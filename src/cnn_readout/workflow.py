"""Portable prepared-data workflow for the field-first CNN.

The module deliberately starts after field forecasts and issuance-time gauge
context have been converted to a leakage-safe tuple cache.  It never opens raw
gauge or radar files.  See :mod:`cnn_readout`'s README for the NPZ schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .loss import importance_corrected_mse
from .model import CNN
from .loss import millimetres_from_normalized_prediction, normalize_target
from .sampling import (
    TARGET_STRATIFIED_V1,
    UNIFORM_WITHOUT_REPLACEMENT,
    build_target_stratum_pools,
    sample_target_stratified,
    validate_sampling_recipe,
)


PREPARED_SCHEMA = "cnn_readout_prepared_npz_v1"
CHECKPOINT_SCHEMA = "cnn_readout_checkpoint_v1"
PREDICTION_SCHEMA = "cnn_readout_predictions_v1"
SELECTION_SCHEMA = "cnn_readout_month_station_oof_v1"
PATCH_SHAPE = (6, 3, 3)
AUX_DIM = 39
STATION_FOLDS = (0, 1, 2)
HELD_MONTHS = np.asarray(
    ("2023-06", "2023-07", "2023-08", "2023-09"), dtype="datetime64[M]"
)
SELECTION_LEADS = np.arange(60, 181, 10, dtype=np.int16)
SELECTION_THRESHOLDS = np.asarray((1.0, 5.0, 10.0), dtype=np.float32)
PURGE_HOURS = 6


class PreparedDataError(ValueError):
    """Raised when a prepared tuple archive violates the public schema."""


@dataclass(frozen=True)
class PreparedData:
    patches: np.ndarray
    auxiliary: np.ndarray
    issue_time_ns: np.ndarray
    station_id: np.ndarray
    station_fold: np.ndarray
    lead_min: np.ndarray
    target_mm: np.ndarray | None
    oof_auxiliary: np.ndarray | None
    dense_shape: tuple[int, int, int] | None = None

    @property
    def size(self) -> int:
        return int(len(self.patches))

    def subset(self, indices: np.ndarray) -> "PreparedData":
        index = np.asarray(indices, dtype=np.int64)
        return PreparedData(
            patches=self.patches[index],
            auxiliary=self.auxiliary[index],
            issue_time_ns=self.issue_time_ns[index],
            station_id=self.station_id[index],
            station_fold=self.station_fold[index],
            lead_min=self.lead_min[index],
            target_mm=None if self.target_mm is None else self.target_mm[index],
            oof_auxiliary=(
                None if self.oof_auxiliary is None else self.oof_auxiliary[:, index]
            ),
            dense_shape=None,
        )


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_string(archive: np.lib.npyio.NpzFile, name: str) -> str:
    if name not in archive.files:
        raise PreparedDataError(f"missing NPZ field: {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise PreparedDataError(f"{name} must be a scalar string")
    return str(value.item())


def _one_dimensional(
    archive: np.lib.npyio.NpzFile, name: str, size: int, dtype: np.dtype
) -> np.ndarray:
    if name not in archive.files:
        raise PreparedDataError(f"missing NPZ field: {name}")
    value = np.asarray(archive[name], dtype=dtype)
    if value.shape != (size,):
        raise PreparedDataError(f"{name} must have shape ({size},), found {value.shape}")
    return np.ascontiguousarray(value)


def _validate_tuple_uniqueness(data: PreparedData) -> None:
    order = np.lexsort((data.lead_min, data.station_id, data.issue_time_ns))
    if len(order) < 2:
        return
    left, right = order[:-1], order[1:]
    duplicate = (
        (data.issue_time_ns[left] == data.issue_time_ns[right])
        & (data.station_id[left] == data.station_id[right])
        & (data.lead_min[left] == data.lead_min[right])
    )
    if np.any(duplicate):
        raise PreparedDataError("duplicate issue-time/station/lead tuple")


def load_prepared_npz(
    path: str | Path,
    *,
    require_target: bool,
    require_oof_auxiliary: bool = False,
) -> PreparedData:
    """Load and strictly validate the public prepared-NPZ contract."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        if _scalar_string(archive, "schema") != PREPARED_SCHEMA:
            raise PreparedDataError(f"unexpected schema in {source}")
        patches = np.asarray(archive["patches"], dtype=np.float32)
        dense_shape: tuple[int, int, int] | None = None
        if patches.ndim == 6 and tuple(patches.shape[3:]) == PATCH_SHAPE:
            # Canonical adapter schema: [issue, station, lead, history, y, x].
            flat_keys = {
                "issue_time_ns",
                "station_id",
                "station_fold",
                "lead_min",
                "target_mm",
            }
            if flat_keys.intersection(archive.files):
                raise PreparedDataError("canonical and flat NPZ keys must not be mixed")
            issues, stations, leads = map(int, patches.shape[:3])
            dense_shape = (issues, stations, leads)
            if min(dense_shape) < 1:
                raise PreparedDataError("dense prepared axes must be non-empty")
            issue_axis = _one_dimensional(
                archive, "issue_times_ns", issues, np.dtype("int64")
            )
            station_axis = _one_dimensional(
                archive, "station_ids", stations, np.dtype("int64")
            )
            lead_axis = _one_dimensional(
                archive, "lead_minutes", leads, np.dtype("int16")
            )
            fold_axis = _one_dimensional(
                archive, "station_folds", stations, np.dtype("int8")
            )
            if np.any(np.diff(issue_axis) <= 0):
                raise PreparedDataError("issue_times_ns must be strictly increasing")
            if len(np.unique(station_axis)) != stations:
                raise PreparedDataError("station_ids must be unique")
            if np.any(np.diff(lead_axis) <= 0):
                raise PreparedDataError("lead_minutes must be strictly increasing")
            if not np.array_equal(lead_axis, SELECTION_LEADS):
                raise PreparedDataError(
                    "canonical lead_minutes must equal +60,...,+180 min"
                )
            issue = np.repeat(issue_axis, stations * leads)
            station = np.tile(np.repeat(station_axis, leads), issues)
            fold = np.tile(np.repeat(fold_axis, leads), issues)
            lead = np.tile(lead_axis, issues * stations)
            size = issues * stations * leads
            patches = patches.reshape(size, *PATCH_SHAPE)
            if "auxiliary" in archive.files:
                auxiliary_dense = np.asarray(archive["auxiliary"], dtype=np.float32)
                if auxiliary_dense.shape != dense_shape + (AUX_DIM,):
                    raise PreparedDataError(
                        f"auxiliary must have shape {dense_shape + (AUX_DIM,)}"
                    )
                auxiliary = auxiliary_dense.reshape(size, AUX_DIM)
            else:
                auxiliary = np.zeros((size, AUX_DIM), dtype=np.float32)
            target_name = "targets_mm"
            oof_shape = (3,) + dense_shape + (AUX_DIM,)
            if "oof_auxiliary" in archive.files:
                oof_auxiliary = np.asarray(archive["oof_auxiliary"], dtype=np.float32)
                if oof_auxiliary.shape != oof_shape:
                    raise PreparedDataError(
                        f"oof_auxiliary must have shape {oof_shape}"
                    )
                oof_auxiliary = oof_auxiliary.reshape(3, size, AUX_DIM)
            else:
                oof_auxiliary = None
        elif patches.ndim == 4 and tuple(patches.shape[1:]) == PATCH_SHAPE:
            # Flat tuple compatibility schema used by lightweight adapters.
            dense_keys = {
                "issue_times_ns",
                "station_ids",
                "station_folds",
                "lead_minutes",
                "targets_mm",
            }
            if dense_keys.intersection(archive.files):
                raise PreparedDataError("canonical and flat NPZ keys must not be mixed")
            size = int(len(patches))
            if "auxiliary" in archive.files:
                auxiliary = np.asarray(archive["auxiliary"], dtype=np.float32)
            else:
                auxiliary = np.zeros((size, AUX_DIM), dtype=np.float32)
            issue = _one_dimensional(
                archive, "issue_time_ns", size, np.dtype("int64")
            )
            station = _one_dimensional(
                archive, "station_id", size, np.dtype("int64")
            )
            fold = _one_dimensional(
                archive, "station_fold", size, np.dtype("int8")
            )
            lead = _one_dimensional(archive, "lead_min", size, np.dtype("int16"))
            target_name = "target_mm"
            if "oof_auxiliary" in archive.files:
                oof_auxiliary = np.asarray(archive["oof_auxiliary"], dtype=np.float32)
            else:
                oof_auxiliary = None
        else:
            raise PreparedDataError(
                "patches must be [N,S,L,6,3,3] (canonical) or [T,6,3,3] (flat)"
            )
        if size == 0 or not np.isfinite(patches).all():
            raise PreparedDataError("patches must be non-empty and finite")
        if auxiliary.shape != (size, AUX_DIM) or not np.isfinite(auxiliary).all():
            raise PreparedDataError(f"auxiliary must be finite [{size},{AUX_DIM}]")
        if np.any(issue == np.iinfo(np.int64).min):
            raise PreparedDataError("issue_time_ns contains NaT")
        if np.any(station < 0):
            raise PreparedDataError("station_id must be nonnegative")
        if not set(np.unique(fold)).issubset(STATION_FOLDS):
            raise PreparedDataError("station_fold must contain only 0, 1 and 2")
        if not set(np.unique(lead)).issubset(set(SELECTION_LEADS.tolist())):
            raise PreparedDataError(
                "lead_min must be drawn from the RN60 target grid +60,...,+180 min"
            )
        for station_value in np.unique(station):
            if len(np.unique(fold[station == station_value])) != 1:
                raise PreparedDataError("each station_id must map to exactly one station_fold")

        target: np.ndarray | None = None
        if target_name in archive.files:
            target_value = np.asarray(archive[target_name], dtype=np.float32)
            expected_target_shape = dense_shape if dense_shape is not None else (size,)
            if target_value.shape != expected_target_shape:
                raise PreparedDataError(
                    f"{target_name} must have shape {expected_target_shape}"
                )
            target = np.ascontiguousarray(target_value.reshape(size))
            invalid_target = np.isinf(target) | (np.isfinite(target) & (target < 0))
            if np.any(invalid_target):
                raise PreparedDataError(
                    f"{target_name} must be nonnegative or NaN; infinity is invalid"
                )
        elif require_target:
            raise PreparedDataError(
                f"{target_name} is required for fitting or OOF selection"
            )

        if oof_auxiliary is not None:
            if oof_auxiliary.shape != (3, size, AUX_DIM):
                raise PreparedDataError(
                    f"oof_auxiliary must have shape [3,{size},{AUX_DIM}]"
                )
            if not np.isfinite(oof_auxiliary).all():
                raise PreparedDataError("oof_auxiliary must be finite")
        elif require_oof_auxiliary:
            raise PreparedDataError(
                "oof_auxiliary is required for station-fold-clean OOF selection"
            )

    result = PreparedData(
        patches=np.ascontiguousarray(patches),
        auxiliary=np.ascontiguousarray(auxiliary),
        issue_time_ns=issue,
        station_id=station,
        station_fold=fold,
        lead_min=lead,
        target_mm=target,
        oof_auxiliary=None if oof_auxiliary is None else np.ascontiguousarray(oof_auxiliary),
        dense_shape=dense_shape,
    )
    _validate_tuple_uniqueness(result)
    return result


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _stratum_fractions(target: np.ndarray) -> np.ndarray:
    strata = np.searchsorted(
        np.asarray((1.0, 5.0, 10.0, 20.0), dtype=np.float32), target, side="right"
    )
    counts = np.bincount(strata, minlength=5).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("target-stratified training requires every target range")
    return counts / counts.sum()


def _training_order(
    target: np.ndarray,
    recipe: str,
    sample_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    count = min(len(target), int(sample_count)) if sample_count > 0 else len(target)
    if count < 1:
        raise ValueError("samples_per_epoch must select at least one tuple")
    if recipe == UNIFORM_WITHOUT_REPLACEMENT:
        return rng.permutation(len(target))[:count].astype(np.int64), None, None
    if count < 5:
        raise ValueError(
            "target_stratified_v1 requires at least five samples per epoch"
        )
    pools = build_target_stratum_pools(target, np.arange(len(target), dtype=np.int64))
    order, quotas, _ = sample_target_stratified(pools, count, rng)
    if np.any(quotas == 0):
        raise ValueError(
            "samples_per_epoch is too small to represent all five target strata"
        )
    return order, _stratum_fractions(target), quotas.astype(np.float64) / count


def train_epoch(
    model: CNN,
    optimizer: torch.optim.Optimizer,
    data: PreparedData,
    indices: np.ndarray,
    *,
    auxiliary: np.ndarray,
    recipe: str,
    samples_per_epoch: int,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    if data.target_mm is None:
        raise PreparedDataError("training requires target_mm")
    selected = np.asarray(indices, dtype=np.int64)
    target_all = data.target_mm[selected]
    order, natural, sampled = _training_order(
        target_all, recipe, samples_per_epoch, rng
    )
    model.train()
    total, seen = 0.0, 0
    for start in range(0, len(order), batch_size):
        local = order[start : start + batch_size]
        rows = selected[local]
        patch = torch.from_numpy(data.patches[rows]).to(device)
        aux = torch.from_numpy(auxiliary[rows]).to(device)
        target_mm = torch.from_numpy(data.target_mm[rows]).to(device)
        target_norm = normalize_target(target_mm)
        prediction = model(patch, aux)
        if recipe == TARGET_STRATIFIED_V1:
            assert natural is not None and sampled is not None
            loss = importance_corrected_mse(
                prediction,
                target_norm,
                target_mm,
                natural_fractions=natural,
                sampled_fractions=sampled,
            )
        else:
            loss = F.mse_loss(prediction, target_norm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * len(rows)
        seen += len(rows)
    return total / seen


@torch.inference_mode()
def predict_mm(
    model: CNN,
    data: PreparedData,
    indices: np.ndarray,
    *,
    auxiliary: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    selected = np.asarray(indices, dtype=np.int64)
    result = np.empty(len(selected), dtype=np.float32)
    for start in range(0, len(selected), batch_size):
        rows = selected[start : start + batch_size]
        patch = torch.from_numpy(data.patches[rows]).to(device)
        aux = torch.from_numpy(auxiliary[rows]).to(device)
        value = millimetres_from_normalized_prediction(model(patch, aux))
        result[start : start + len(rows)] = value.cpu().numpy().astype(np.float32)
    return result


def categorical_counts(
    prediction: np.ndarray,
    truth: np.ndarray,
    lead_min: np.ndarray,
    leads: np.ndarray = SELECTION_LEADS,
    thresholds: np.ndarray = SELECTION_THRESHOLDS,
) -> np.ndarray:
    counts = np.zeros((len(leads), len(thresholds), 4), dtype=np.int64)
    for lead_index, lead in enumerate(leads):
        lead_mask = lead_min == lead
        for threshold_index, threshold in enumerate(thresholds):
            observed = truth >= threshold
            forecast = prediction >= threshold
            counts[lead_index, threshold_index] = (
                np.sum(lead_mask & observed & forecast),
                np.sum(lead_mask & observed & ~forecast),
                np.sum(lead_mask & ~observed & forecast),
                np.sum(lead_mask & ~observed & ~forecast),
            )
    return counts


def csi_from_counts(counts: np.ndarray) -> np.ndarray:
    value = np.asarray(counts, dtype=np.float64)
    denominator = value[..., 0] + value[..., 1] + value[..., 2]
    return np.divide(
        value[..., 0],
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def macro_csi_from_counts(counts: np.ndarray) -> float:
    values = csi_from_counts(counts)
    if not np.isfinite(values).any():
        raise RuntimeError("all macro-CSI cells are undefined")
    return float(np.nanmean(values))


def month_boundary_keep(issue_time_ns: np.ndarray) -> np.ndarray:
    times = np.asarray(issue_time_ns, dtype=np.int64).astype("datetime64[ns]")
    months = np.unique(times.astype("datetime64[M]"))
    keep = np.ones(len(times), dtype=bool)
    margin = np.timedelta64(PURGE_HOURS, "h")
    for boundary in months[1:]:
        keep &= np.abs(times - boundary.astype("datetime64[ns]")) >= margin
    return keep


def select_epoch_oof(
    data: PreparedData,
    *,
    max_epochs: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    recipe: str,
    samples_per_epoch: int,
    train_batch_size: int,
    validation_batch_size: int,
    device: torch.device,
    selection_thresholds: Sequence[float] | None = None,
) -> dict[str, object]:
    """Run four held-month by three held-station-fold OOF selection."""

    if data.target_mm is None or data.oof_auxiliary is None:
        raise PreparedDataError("OOF selection requires target_mm and oof_auxiliary")
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    thresholds = np.asarray(SELECTION_THRESHOLDS if selection_thresholds is None
                            else selection_thresholds, dtype=np.float32)
    if tuple(thresholds.tolist()) not in ((1., 5., 10.), (1., 5., 10., 20.)):
        raise ValueError("Selection thresholds must be 1,5,10 (JHRS) or 1,5,10,20 (historical)")
    months = data.issue_time_ns.astype("datetime64[ns]").astype("datetime64[M]")
    if not np.array_equal(np.unique(months), HELD_MONTHS):
        raise PreparedDataError("OOF selection requires June–September 2023 only")
    if set(np.unique(data.station_fold)) != set(STATION_FOLDS):
        raise PreparedDataError("OOF selection requires all three station folds")
    selection = np.isin(data.lead_min, SELECTION_LEADS)
    finite_target = np.isfinite(data.target_mm)
    keep = month_boundary_keep(data.issue_time_ns)
    pooled = np.zeros(
        (max_epochs, len(SELECTION_LEADS), len(thresholds), 4),
        dtype=np.int64,
    )
    cells: list[dict[str, object]] = []
    for month_index, month in enumerate(HELD_MONTHS):
        for fold in STATION_FOLDS:
            train = np.flatnonzero(
                selection
                & finite_target
                & keep
                & (months != month)
                & (data.station_fold != fold)
            )
            valid = np.flatnonzero(
                selection
                & finite_target
                & keep
                & (months == month)
                & (data.station_fold == fold)
            )
            if not len(train) or not len(valid):
                raise PreparedDataError(f"empty OOF cell month={month} fold={fold}")
            cell_seed = int(seed + 100 * month_index + fold)
            set_deterministic_seed(cell_seed)
            model = CNN().to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            auxiliary = data.oof_auxiliary[fold]
            losses: list[float] = []
            for epoch_index in range(max_epochs):
                rng = np.random.default_rng(cell_seed + (epoch_index + 1) * 1_000_003)
                losses.append(
                    train_epoch(
                        model,
                        optimizer,
                        data,
                        train,
                        auxiliary=auxiliary,
                        recipe=recipe,
                        samples_per_epoch=samples_per_epoch,
                        batch_size=train_batch_size,
                        rng=rng,
                        device=device,
                    )
                )
                prediction = predict_mm(
                    model,
                    data,
                    valid,
                    auxiliary=auxiliary,
                    batch_size=validation_batch_size,
                    device=device,
                )
                pooled[epoch_index] += categorical_counts(
                    prediction,
                    data.target_mm[valid],
                    data.lead_min[valid],
                    thresholds=thresholds,
                )
            cells.append(
                {
                    "held_month": str(month),
                    "held_station_fold": fold,
                    "seed": cell_seed,
                    "training_tuples": int(len(train)),
                    "validation_tuples": int(len(valid)),
                    "training_loss_by_epoch": losses,
                }
            )
    macro = np.asarray([macro_csi_from_counts(value) for value in pooled])
    selected_epoch = int(np.nanargmax(macro)) + 1
    return {
        "schema": SELECTION_SCHEMA,
        "selection_period": "2023-06 through 2023-09",
        "fold_contract": "4 held months × 3 held station folds",
        "training_axes": "month != held month and station fold != held fold",
        "validation_axes": "held month and held station fold",
        "month_boundary_purge_hours": PURGE_HOURS,
        "selection_leads_min": SELECTION_LEADS.astype(int).tolist(),
        "selection_thresholds_mm": thresholds.astype(float).tolist(),
        "selection_rule": "maximum pooled OOF macro CSI; earliest exact tie",
        "macro_csi_by_epoch": macro.tolist(),
        "pooled_contingency_counts_by_epoch": pooled.tolist(),
        "selected_epoch": selected_epoch,
        "maximum_epoch": int(max_epochs),
        "seed": int(seed),
        "sampling_recipe": recipe,
        "samples_per_epoch": int(samples_per_epoch),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "cells": cells,
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def run_fit(args: argparse.Namespace) -> None:
    data = load_prepared_npz(args.input, require_target=True)
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    recipe = validate_sampling_recipe(args.sampling_recipe)
    device = resolve_device(args.device)
    set_deterministic_seed(args.seed)
    model = CNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    assert data.target_mm is not None
    indices = np.flatnonzero(np.isfinite(data.target_mm))
    if not len(indices):
        raise PreparedDataError("no finite targets are available for fitting")
    losses = []
    for _ in range(args.epochs):
        losses.append(
            train_epoch(
                model,
                optimizer,
                data,
                indices,
                auxiliary=data.auxiliary,
                recipe=recipe,
                samples_per_epoch=args.samples_per_epoch,
                batch_size=args.batch_size,
                rng=rng,
                device=device,
            )
        )
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": {
            "patch_steps": PATCH_SHAPE[0],
            "patch_size": PATCH_SHAPE[1],
            "aux_dim": AUX_DIM,
        },
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "input_sha256": sha256_file(args.input),
        "training_loss_by_epoch": losses,
        "sampling_recipe": recipe,
        "samples_per_epoch": int(args.samples_per_epoch),
        "optimizer": "AdamW",
        "target_scale": {"minimum_mm": 0.0, "maximum_mm": 735.0, "source": "frozen fitting-station Direct R2P scale"},
        "objective": "importance-corrected normalized RN60 MSE",
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
    }
    atomic_torch(args.output, payload)


def load_checkpoint(path: Path, device: torch.device) -> tuple[CNN, dict]:
    # CNN checkpoints contain only tensors and primitive metadata, so the
    # restricted loader avoids Python pickle code execution.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported CNN checkpoint")
    expected = {"patch_steps": 6, "patch_size": 3, "aux_dim": 39}
    if payload.get("model_config") != expected:
        raise RuntimeError("checkpoint model contract changed")
    model = CNN().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def run_predict(args: argparse.Namespace) -> None:
    data = load_prepared_npz(args.input, require_target=False)
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    indices = np.arange(data.size, dtype=np.int64)
    prediction = predict_mm(
        model,
        data,
        indices,
        auxiliary=data.auxiliary,
        batch_size=args.batch_size,
        device=device,
    )
    # All primary routes are compared after the same storage-precision round trip.
    prediction = prediction.astype(np.float16).astype(np.float32)
    values: dict[str, object] = {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "checkpoint_sha256": np.asarray(sha256_file(args.checkpoint)),
        "checkpoint_epoch": np.asarray(checkpoint["epochs"], dtype=np.int16),
        "checkpoint_seed": np.asarray(checkpoint["seed"], dtype=np.int64),
    }
    if data.dense_shape is not None:
        issues, stations, leads = data.dense_shape
        values.update(
            {
                "predictions_mm": prediction.reshape(data.dense_shape),
                "issue_times_ns": data.issue_time_ns.reshape(data.dense_shape)[:, 0, 0],
                "station_ids": data.station_id.reshape(data.dense_shape)[0, :, 0],
                "lead_minutes": data.lead_min.reshape(data.dense_shape)[0, 0, :],
            }
        )
        if data.target_mm is not None:
            values["targets_mm"] = data.target_mm.reshape(issues, stations, leads)
    else:
        values.update(
            {
                "prediction_mm": prediction,
                "issue_time_ns": data.issue_time_ns,
                "station_id": data.station_id,
                "lead_min": data.lead_min,
            }
        )
        if data.target_mm is not None:
            values["target_mm"] = data.target_mm
    atomic_npz(args.output, **values)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sampling-recipe", default=TARGET_STRATIFIED_V1)
    parser.add_argument("--samples-per-epoch", type=int, default=2_097_152)
    parser.add_argument("--batch-size", type=int, default=4096)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a prepared NPZ")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--require-target", action="store_true")
    validate.add_argument("--require-oof-auxiliary", action="store_true")

    select = commands.add_parser("select-epoch", help="run 4×3 OOF epoch selection")
    select.add_argument("--input", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--max-epochs", type=int, default=50)
    select.add_argument("--seed", type=int, default=7131000,
                        help="CNN/context cell-seed base; month adds 100 and fold adds 1")
    select.add_argument("--validation-batch-size", type=int, default=65_536)
    select.add_argument("--selection-thresholds", type=float, nargs="+",
                        help="Use 1 5 10 for JHRS; omitted retains historical 1 5 10 20")
    add_training_arguments(select)

    fit = commands.add_parser("fit", help="fit one final CNN")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--epochs", type=int, required=True)
    fit.add_argument("--seed", type=int, required=True)
    add_training_arguments(fit)

    predict = commands.add_parser("predict", help="predict from a fitted checkpoint")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--batch-size", type=int, default=65_536)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        data = load_prepared_npz(
            args.input,
            require_target=args.require_target,
            require_oof_auxiliary=args.require_oof_auxiliary,
        )
        print(json.dumps({"schema": PREPARED_SCHEMA, "tuples": data.size}, indent=2))
    elif args.command == "select-epoch":
        data = load_prepared_npz(
            args.input, require_target=True, require_oof_auxiliary=True
        )
        recipe = validate_sampling_recipe(args.sampling_recipe)
        result = select_epoch_oof(
            data,
            max_epochs=args.max_epochs,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            recipe=recipe,
            samples_per_epoch=args.samples_per_epoch,
            train_batch_size=args.batch_size,
            validation_batch_size=args.validation_batch_size,
            device=resolve_device(args.device),
            selection_thresholds=args.selection_thresholds,
        )
        result["input_sha256"] = sha256_file(args.input)
        atomic_json(args.output, result)
    elif args.command == "fit":
        run_fit(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
