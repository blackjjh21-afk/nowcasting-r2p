#!/usr/bin/env python3
"""Train, evaluate and export the matched-input vanilla Held-out R2P.

The program is intentionally command-driven so long jobs can run as separate
processes, retain logs, and restart safely.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR.parent))

from r2p_4km10min.data import (  # noqa: E402
    GAUGE_OFFSETS_MIN,
    LEAD_MINUTES,
    ORIGINAL_RADAR_RESOLUTION_KM,
    PRIMARY_REPORT_LEADS,
    RADAR_OFFSETS_MIN,
    GaugeScaler,
    R2PWindowDataset,
    build_issue_times,
    build_station_features,
    build_train514_and_eval642_features,
    discover_tiff_index,
    load_station_contract,
    read_gauge_times,
    read_gauge_subset,
    sha256_array,
    sha256_file,
    validate_tiff_archive,
)
from r2p_4km10min.model import (  # noqa: E402
    R2PModelConfig,
    UpgradedNowcaster,
    load_query_expanded_state,
)


TRAIN_YEARS = (2019, 2020, 2021, 2022)
VALIDATION_YEARS = (2023,)
EVALUATION_YEARS = (2024, 2025)
EVAL_THRESHOLDS = (1.0, 5.0, 10.0)
SELECTION_THRESHOLDS = (1.0, 5.0, 10.0, 20.0)
SELECTION_LEADS = tuple(range(60, 181, 10))
SELECTION_WARMUP_EPOCHS = 3
CONTRACT_NAME = "vanilla_heldout_r2p_official4km10min_input_only_v1"


def _validate_managed_child_directory(
    target: Path, parent: Path, expected_name: str
) -> None:
    """Constrain recursive deletion to one named, non-symlink child directory."""

    if not expected_name or target.name != expected_name:
        raise RuntimeError(f"refusing to delete unexpected managed path: {target}")
    if target.is_symlink() or not target.is_dir():
        raise RuntimeError(f"refusing to delete a symlink or non-directory: {target}")
    if target.resolve().parent != parent.resolve():
        raise RuntimeError(f"refusing to delete a path outside {parent}: {target}")


def _validate_prediction_store_for_overwrite(destination: Path) -> None:
    """Require an R2P completion marker before replacing a prediction store."""

    if destination.is_symlink() or not destination.is_dir():
        raise RuntimeError(
            f"refusing to overwrite a symlink or non-directory store: {destination}"
        )
    marker_path = destination / "COMPLETED"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "refusing to overwrite an existing directory without a valid R2P "
            f"completion marker: {destination}"
        ) from exc
    if marker.get("completed") is not True or marker.get("contract") != CONTRACT_NAME:
        raise RuntimeError(
            "refusing to overwrite an existing directory not owned by this R2P "
            f"contract: {destination}"
        )


def _load_trusted_local_checkpoint(
    path: Path, *, trusted_root: Path, purpose: str
) -> Mapping:
    """Load a full-state checkpoint only from the caller's local output tree.

    R2P resume checkpoints contain NumPy and Python RNG states in addition to
    tensor state dictionaries, so PyTorch's restricted ``weights_only`` loader
    cannot deserialize this established format. The path guard and warning make
    the remaining trust boundary explicit.
    """

    if path.is_symlink() or not path.is_file() or path.suffix != ".pt":
        raise RuntimeError(f"invalid trusted-local checkpoint path: {path}")
    resolved_root = trusted_root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            f"checkpoint is outside the trusted local output root: {path}"
        ) from exc
    warnings.warn(
        f"Loading {purpose} checkpoint with Python pickle semantics. Only use "
        "checkpoints generated locally by this R2P runner; never use an "
        "untrusted or downloaded .pt file.",
        RuntimeWarning,
        stacklevel=2,
    )
    payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("trusted-local checkpoint payload must be a mapping")
    return payload


def write_json(path: Path, value: Mapping | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def contract_digest(contract: Mapping) -> str:
    encoded = json.dumps(contract, sort_keys=True).encode("utf-8")
    return sha256_array(np.frombuffer(encoded, dtype=np.uint8))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist")
    return device


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.project_root.resolve()
    return {
        "root": root,
        "radar": args.radar_root.resolve(),
        "gauge": args.gauge_csv.resolve(),
        "split": args.split_csv.resolve(),
        # Real station identifiers and coordinates are not redistributed in
        # this public source release.  Both contracts are therefore explicit
        # command-line inputs instead of silently resolving to an author path.
        "mapping": args.mapping_csv.resolve(),
        "stations": args.stations_csv.resolve(),
        "output": args.output_root.resolve(),
    }


def verify_inputs(paths: Mapping[str, Path], enforce_hashes: bool) -> tuple[dict, object]:
    missing = [
        str(paths[key])
        for key in ("radar", "gauge", "split", "mapping", "stations")
        if not paths[key].exists()
    ]
    if missing:
        raise FileNotFoundError("required input is missing:\n" + "\n".join(missing))
    archive = validate_tiff_archive(paths["radar"])
    stations = load_station_contract(
        paths["mapping"],
        paths["stations"],
        paths["split"],
        archive["longitude"],
        archive["latitude"],
        archive["source_valid"],
        enforce_frozen_hashes=enforce_hashes,
    )
    return archive, stations


def model_for_stations(
    stations: pd.DataFrame,
    config: R2PModelConfig,
    device: torch.device,
) -> UpgradedNowcaster:
    y = stations.exprecast_y.to_numpy(dtype=np.int64)
    x = stations.exprecast_x.to_numpy(dtype=np.int64)
    # Attention geometry stays in the original R2P's frozen 2-km coordinate
    # frame.  Only radar patch/deformable sampling uses the 4-km TIFF indices.
    original_y = stations.iy.to_numpy(dtype=np.int64)
    original_x = stations.ix.to_numpy(dtype=np.int64)
    positions = np.stack(
        [original_x * ORIGINAL_RADAR_RESOLUTION_KM,
         original_y * ORIGINAL_RADAR_RESOLUTION_KM],
        axis=1,
    ).astype(np.float32)
    return UpgradedNowcaster(config, y, x, positions).to(device)


def compute_metrics_arrays(
    predictions: np.ndarray,
    truth: np.ndarray,
    lead_minutes: np.ndarray = LEAD_MINUTES,
    thresholds: Iterable[float] = EVAL_THRESHOLDS,
) -> list[dict[str, float | int]]:
    if predictions.shape != truth.shape:
        raise ValueError("prediction and truth shapes differ")
    rows: list[dict[str, float | int]] = []
    for lead_index, lead in enumerate(lead_minutes):
        prediction = np.asarray(predictions[..., lead_index], dtype=np.float64)
        target = np.asarray(truth[..., lead_index], dtype=np.float64)
        finite = np.isfinite(prediction) & np.isfinite(target)
        if not np.isfinite(prediction).all():
            raise RuntimeError("non-finite model prediction")
        error = prediction[finite] - target[finite]
        base: dict[str, float | int] = {
            "lead_min": int(lead),
            "n_finite": int(finite.sum()),
            "rmse": float(np.sqrt(np.mean(np.square(error)))) if error.size else math.nan,
            "mae": float(np.mean(np.abs(error))) if error.size else math.nan,
            "bias": float(np.mean(error)) if error.size else math.nan,
        }
        for threshold in thresholds:
            observed = target >= threshold
            forecast = prediction >= threshold
            hits = int(np.sum(finite & observed & forecast))
            misses = int(np.sum(finite & observed & ~forecast))
            false_alarms = int(np.sum(finite & ~observed & forecast))
            denominator = hits + misses + false_alarms
            base[f"hits_{threshold:g}"] = hits
            base[f"misses_{threshold:g}"] = misses
            base[f"false_alarms_{threshold:g}"] = false_alarms
            base[f"csi_{threshold:g}"] = hits / denominator if denominator else math.nan
        rows.append(base)
    return rows


def selection_score(rows: list[dict[str, float | int]]) -> tuple[float, int]:
    """Return the historical training-monitor RN60 macro CSI over 52 cells."""

    by_lead = {int(row["lead_min"]): row for row in rows}
    values: list[float] = []
    missing: list[str] = []
    for lead in SELECTION_LEADS:
        row = by_lead.get(lead)
        if row is None:
            missing.append(f"lead={lead}")
            continue
        for threshold in SELECTION_THRESHOLDS:
            key = f"csi_{threshold:g}"
            value = float(row.get(key, math.nan))
            if not math.isfinite(value):
                missing.append(f"lead={lead},threshold={threshold:g}")
            else:
                values.append(value)
    expected = len(SELECTION_LEADS) * len(SELECTION_THRESHOLDS)
    if missing or len(values) != expected:
        raise RuntimeError(
            "checkpoint selection lacks required RN60 CSI cells: "
            + ", ".join(missing[:8])
        )
    return float(np.mean(values)), expected


def stratified_validation_issues(
    dataset: R2PWindowDataset,
    *,
    rain_threshold: float = 0.1,
    buffer_minutes: int = 180,
) -> np.ndarray:
    """Retain all rainy issues plus dry issues within 180 min of rain."""

    issues = np.asarray(dataset.issue_times_ns, dtype=np.int64)
    rows = np.asarray(
        [
            [
                dataset.gauge_positions[int(issue) + int(lead) * 60 * 1_000_000_000]
                for lead in LEAD_MINUTES
            ]
            for issue in issues
        ],
        dtype=np.int64,
    )
    target = dataset.target_rn60[rows]
    rainy = (np.nan_to_num(target, nan=0.0) >= rain_threshold).any(axis=(1, 2))
    if not rainy.any():
        return issues
    rainy_times = np.sort(issues[rainy])
    dry_times = issues[~rainy]
    position = np.searchsorted(rainy_times, dry_times)
    left = np.clip(position - 1, 0, len(rainy_times) - 1)
    right = np.clip(position, 0, len(rainy_times) - 1)
    distance = np.minimum(
        np.abs(dry_times - rainy_times[left]),
        np.abs(dry_times - rainy_times[right]),
    )
    keep_dry = distance <= buffer_minutes * 60 * 1_000_000_000
    return np.sort(np.concatenate([issues[rainy], dry_times[keep_dry]]))


def select_best_epoch(trace_frame: pd.DataFrame) -> int:
    """Select the earliest exact tie after the three-epoch warm-up."""

    required = {"epoch", "selection_macro_csi"}
    missing = required.difference(trace_frame.columns)
    if missing:
        raise RuntimeError(
            "selection trace lacks required columns: " + ", ".join(sorted(missing))
        )
    pool = trace_frame.loc[
        trace_frame["epoch"] > SELECTION_WARMUP_EPOCHS,
        ["epoch", "selection_macro_csi"],
    ].copy()
    if pool.empty:
        raise RuntimeError(
            f"checkpoint selection requires at least {SELECTION_WARMUP_EPOCHS + 1} epochs"
        )
    if not np.isfinite(pool["selection_macro_csi"].to_numpy(dtype=float)).all():
        raise RuntimeError("selection trace contains non-finite macro CSI")
    best_score = float(pool["selection_macro_csi"].max())
    tied = pool.loc[pool["selection_macro_csi"] == best_score].sort_values("epoch")
    return int(tied.iloc[0]["epoch"])


def apply_station_dropout(
    station: torch.Tensor,
    *,
    full_probability: float,
    p_min: float,
    p_max: float,
) -> float:
    p = 1.0 if np.random.rand() < full_probability else float(np.random.uniform(p_min, p_max))
    count = int(round(p * station.shape[2]))
    if count:
        indices = torch.randperm(station.shape[2], device=station.device)[:count]
        station[:, :, indices, :2] = 0.0
        station[:, :, indices, 2] = 1.0
    return p


def collate_prediction(
    model: UpgradedNowcaster,
    loader: DataLoader,
    scaler: GaugeScaler,
    device: torch.device,
    station_slice: slice | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    issues: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            radar = batch["radar"].to(device, non_blocking=True)
            station = batch["station"].to(device, non_blocking=True)
            prediction_n = model(radar, station).cpu().numpy()
            prediction = np.clip(scaler.inverse_target(prediction_n), 0.0, None)
            truth = batch["target"].numpy()
            if station_slice is not None:
                prediction = prediction[:, station_slice]
                truth = truth[:, station_slice]
            if not np.isfinite(prediction).all():
                raise RuntimeError("non-finite prediction during evaluation")
            predictions.append(prediction.astype(np.float32))
            truths.append(truth.astype(np.float32))
            issues.append(batch["issue_time_ns"].numpy().astype(np.int64))
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(truths, axis=0),
        np.concatenate(issues, axis=0),
    )


def save_contract(
    output: Path,
    paths: Mapping[str, Path],
    archive: Mapping,
    config: R2PModelConfig,
    args: argparse.Namespace,
) -> dict:
    contract = {
        "contract": CONTRACT_NAME,
        "radar_input": {
            "format": "official float32 KMA TIFF",
            "shape": [256, 256],
            "resolution_km": 4,
            "offsets_min": RADAR_OFFSETS_MIN.astype(int).tolist(),
            "normalization": "max(raw_100xdbz, 0) / 10000",
        },
        "gauge_context": {
            "stations": "fitting514 only",
            "variables": ["RN60", "RN15", "joint missing", "lat", "lon"],
            "offsets_min": GAUGE_OFFSETS_MIN.astype(int).tolist(),
            "coordinate_channels": "frozen original 2-km R2P grid normalization",
            "station_attention_geometry": "frozen original 2-km R2P station indices",
        },
        "target": {
            "variable": "gauge RN60 ending at valid time",
            "lead_minutes": LEAD_MINUTES.astype(int).tolist(),
            "report_leads": PRIMARY_REPORT_LEADS.astype(int).tolist(),
        },
        "split": {
            "training_years": list(TRAIN_YEARS),
            "selection_years": list(VALIDATION_YEARS),
            "evaluation_years": list(EVALUATION_YEARS),
            "fitting_stations": 514,
            "heldout_stations": 128,
        },
        "model": config.to_dict(),
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "checkpoint_selection": {
                "metric": "unweighted RN60 macro CSI",
                "lead_minutes": list(SELECTION_LEADS),
                "thresholds_mm": list(SELECTION_THRESHOLDS),
                "cell_count": len(SELECTION_LEADS) * len(SELECTION_THRESHOLDS),
                "validation_support": (
                    "all issues with any fitting-station RN60 target >=0.1 mm, "
                    "plus dry issues within 180 min of such an issue"
                ),
                "validation_input_condition": {
                    "query_site_gauge_history": "available through issuance",
                    "station_dropout": "disabled",
                },
                "smoothing": "none",
                "excluded_epochs": list(range(1, SELECTION_WARMUP_EPOCHS + 1)),
                "tie_break": "earliest epoch",
            },
            "station_dropout": {
                "p_range": [args.station_dropout_min, args.station_dropout_max],
                "full_probability": args.station_dropout_full_probability,
            },
        },
        "source_hashes": {
            "manifest": sha256_file(archive["manifest_path"]),
            "grid_coordinates": sha256_file(archive["coordinate_path"]),
            "mapping": sha256_file(paths["mapping"]),
            "stations": sha256_file(paths["stations"]),
            "split": sha256_file(paths["split"]),
            "gauge_csv": sha256_file(paths["gauge"]),
        },
    }
    contract_path = output / "scientific_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError(
                "existing scientific_contract.json differs; use a new output root"
            )
    else:
        write_json(contract_path, contract)
    return contract


def prepare_fit_data(paths, archive, station_contract, years, *, scaler=None):
    gauge_times, rn60, rn15, station_ids = read_gauge_subset(
        paths["gauge"], station_contract.train, years
    )
    expected_ids = station_contract.train.station_id.to_numpy(dtype=np.int64)
    if not np.array_equal(station_ids, expected_ids):
        raise RuntimeError("train514 gauge station order mismatch")
    features, scaler = build_station_features(
        rn60,
        rn15,
        station_contract.train,
        archive["longitude"],
        archive["latitude"],
        scaler=scaler,
    )
    radar_index = discover_tiff_index(paths["radar"], years)
    issues = build_issue_times(radar_index, gauge_times, require_target_horizon=True)
    dataset = R2PWindowDataset(radar_index, gauge_times, features, rn60, issues)
    return dataset, scaler


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    seed,
    scaler,
    contract,
    selection_trace,
):
    return {
        "contract": CONTRACT_NAME,
        "contract_sha256": contract_digest(contract),
        "epoch": int(epoch),
        "seed": int(seed),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_min": scaler.minimum,
        "scaler_max": scaler.maximum,
        "selection_trace": selection_trace,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                [value.cpu() for value in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def run_preflight(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    archive, stations = verify_inputs(paths, not args.allow_unfrozen_contract)
    counts = {}
    for label, years in (
        ("train", TRAIN_YEARS),
        ("validation", VALIDATION_YEARS),
        ("evaluation", EVALUATION_YEARS),
    ):
        gauge_times = read_gauge_times(paths["gauge"], years)
        radar_index = discover_tiff_index(paths["radar"], years)
        issues = build_issue_times(radar_index, gauge_times, require_target_horizon=True)
        counts[label] = len(issues)
        print(
            f"{label}: {len(issues):,} issue times; "
            f"{np.asarray(issues[0], dtype='datetime64[ns]')}--"
            f"{np.asarray(issues[-1], dtype='datetime64[ns]')} KST"
        )
    expected = {"train": 69616, "validation": 17339, "evaluation": 35088}
    if counts != expected:
        raise RuntimeError(f"current anchor counts {counts} differ from audited counts {expected}")
    print("preflight passed", counts)


def run_memory_smoke(args: argparse.Namespace) -> None:
    device = require_device(args.device)
    config = R2PModelConfig()
    stations = args.smoke_stations
    station_y = np.linspace(10, 245, stations, dtype=np.int64)
    station_x = np.linspace(15, 240, stations, dtype=np.int64)
    position = np.stack([station_x * 4.0, station_y * 4.0], axis=1).astype(np.float32)
    model = UpgradedNowcaster(config, station_y, station_x, position).to(device)
    radar = torch.rand(args.batch_size, 7, 256, 256, device=device)
    gauge = torch.rand(args.batch_size, 12, stations, 5, device=device)
    target = torch.rand(args.batch_size, stations, 36, device=device)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    output = model(radar, gauge)
    loss = torch.square(output - target).mean()
    loss.backward()
    result = {
        "input_radar": list(radar.shape),
        "input_station": list(gauge.shape),
        "output": list(output.shape),
        "loss": float(loss.detach().cpu()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        ),
    }
    print(json.dumps(result, indent=2))


def run_train(args: argparse.Namespace) -> None:
    device = require_device(args.device)
    paths = resolve_paths(args)
    output = paths["output"]
    archive, stations = verify_inputs(paths, not args.allow_unfrozen_contract)
    config = R2PModelConfig()
    contract = save_contract(output, paths, archive, config, args)
    train_dataset, scaler = prepare_fit_data(
        paths, archive, stations, TRAIN_YEARS, scaler=None
    )
    validation_dataset, _ = prepare_fit_data(
        paths, archive, stations, VALIDATION_YEARS, scaler=scaler
    )
    selected_validation_issues = stratified_validation_issues(validation_dataset)
    validation_dataset = R2PWindowDataset(
        validation_dataset.radar_index,
        validation_dataset.gauge_times_ns,
        validation_dataset.station_features,
        validation_dataset.target_rn60,
        selected_validation_issues,
    )
    np.save(output / "scaler_min.npy", scaler.minimum, allow_pickle=False)
    np.save(output / "scaler_max.npy", scaler.maximum, allow_pickle=False)
    stations.train.to_csv(output / "fitting_stations.csv", index=False)
    print(
        f"train={len(train_dataset):,}, validation={len(validation_dataset):,}, "
        f"batch={args.batch_size}, device={device}"
    )

    for seed in args.seeds:
        seed_dir = output / f"seed_{seed}"
        if (seed_dir / "COMPLETED").exists() and not args.overwrite:
            print(f"seed {seed}: completed; skipping")
            continue
        if seed_dir.exists() and not args.resume and not args.overwrite:
            raise FileExistsError(f"seed directory exists; use --resume or --overwrite: {seed_dir}")
        if seed_dir.exists() and args.overwrite:
            # ``seed_dir`` is an internally derived direct child of the output
            # root; lock both its parent and exact seed name before rmtree.
            _validate_managed_child_directory(seed_dir, output, f"seed_{seed}")
            shutil.rmtree(seed_dir)
        seed_dir.mkdir(parents=True, exist_ok=True)
        set_seed(args.base_seed + seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            # Preserve the established R2P's global RandomSampler protocol.
            sampler=RandomSampler(train_dataset),
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.validation_batch_size,
            sampler=SequentialSampler(validation_dataset),
            num_workers=args.validation_workers,
            pin_memory=True,
            persistent_workers=args.validation_workers > 0,
        )
        model = model_for_stations(stations.train, config, device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=4, threshold=1e-6
        )
        start_epoch = 1
        trace: list[dict] = []
        last_path = seed_dir / "last.pt"
        if args.resume and last_path.is_file():
            payload = _load_trusted_local_checkpoint(
                last_path, trusted_root=output, purpose="training-resume"
            )
            if payload.get("contract") != CONTRACT_NAME:
                raise RuntimeError("resume checkpoint has the wrong contract")
            model.load_state_dict(payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            start_epoch = int(payload["epoch"]) + 1
            trace = list(payload.get("selection_trace", []))
            rng = payload.get("rng", {})
            if rng:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"])
                if device.type == "cuda" and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
            print(f"seed {seed}: resuming at epoch {start_epoch}")

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total_squared_error = 0.0
            total_valid = 0
            dropout_values = []
            start_time = time.time()
            for batch_index, batch in enumerate(train_loader, start=1):
                radar = batch["radar"].to(device, non_blocking=True)
                station = batch["station"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                dropout_values.append(
                    apply_station_dropout(
                        station,
                        full_probability=args.station_dropout_full_probability,
                        p_min=args.station_dropout_min,
                        p_max=args.station_dropout_max,
                    )
                )
                finite = torch.isfinite(target)
                target_n = (
                    torch.nan_to_num(target, nan=0.0) - float(scaler.minimum[0])
                ) / float(scaler.denominator[0])
                optimizer.zero_grad(set_to_none=True)
                prediction = model(radar, station)
                squared_error = torch.square(prediction - target_n)
                loss = (squared_error * finite).sum() / finite.sum().clamp_min(1)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_squared_error += float((squared_error * finite).sum().detach().cpu())
                total_valid += int(finite.sum().detach().cpu())
                if args.progress_every and batch_index % args.progress_every == 0:
                    print(
                        f"seed={seed} epoch={epoch} batch={batch_index}/{len(train_loader)} "
                        f"loss={total_squared_error/max(total_valid,1):.6g}",
                        flush=True,
                    )

            validation_prediction, validation_truth, validation_issues = collate_prediction(
                model, validation_loader, scaler, device
            )
            metrics = compute_metrics_arrays(
                validation_prediction,
                validation_truth,
                thresholds=SELECTION_THRESHOLDS,
            )
            selection_macro_csi, selection_cell_count = selection_score(metrics)
            validation_error = validation_prediction - validation_truth
            finite_validation = np.isfinite(validation_error)
            validation_mse = float(
                np.mean(np.square(validation_error[finite_validation]))
            )
            scheduler.step(validation_mse / float(scaler.denominator[0]) ** 2)
            row = {
                "epoch": epoch,
                "train_mse_norm": total_squared_error / max(total_valid, 1),
                "dropout_p_mean": float(np.mean(dropout_values)),
                "validation_mse_mm2": validation_mse,
                "selection_macro_csi": selection_macro_csi,
                "selection_cell_count": selection_cell_count,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_minutes": (time.time() - start_time) / 60.0,
            }
            trace.append(row)
            pd.DataFrame(trace).to_csv(seed_dir / "selection_trace.csv", index=False)
            pd.DataFrame(metrics).to_csv(
                seed_dir / f"validation_metrics_epoch_{epoch:03d}.csv", index=False
            )
            payload = checkpoint_payload(
                model, optimizer, scheduler, epoch, seed, scaler, contract, trace
            )
            if args.save_every_epoch:
                atomic_torch_save(seed_dir / f"epoch_{epoch:03d}.pt", payload)
            atomic_torch_save(last_path, payload)
            print(
                f"seed={seed} epoch={epoch}/{args.epochs} "
                f"train={row['train_mse_norm']:.6g} val_rmse={math.sqrt(validation_mse):.4f} "
                f"selection_macro_csi={selection_macro_csi:.4f}",
                flush=True,
            )

        trace_frame = pd.DataFrame(trace)
        best_epoch = select_best_epoch(trace_frame)
        source = seed_dir / f"epoch_{best_epoch:03d}.pt"
        if not source.is_file():
            raise RuntimeError(
                "selected epoch checkpoint is unavailable; keep --save-every-epoch enabled"
            )
        shutil.copyfile(source, seed_dir / "best.pt")
        write_json(
            seed_dir / "COMPLETED",
            {
                "completed": True,
                "contract": CONTRACT_NAME,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_checkpoint_sha256": sha256_file(seed_dir / "best.pt"),
            },
        )
        print(f"seed={seed}: selected epoch {best_epoch}")
        del model, optimizer, scheduler, train_loader, validation_loader
        torch.cuda.empty_cache() if device.type == "cuda" else None
        gc.collect()


def prepare_evaluation_data(paths, archive, stations, scaler):
    gauge_times, train60, train15, train_ids = read_gauge_subset(
        paths["gauge"], stations.train, EVALUATION_YEARS
    )
    if not np.array_equal(
        train_ids, stations.train.station_id.to_numpy(dtype=np.int64)
    ):
        raise RuntimeError("evaluation fitting-context order mismatch")
    _, eval_features = build_train514_and_eval642_features(
        train60, train15, stations, scaler=scaler
    )
    # Held-out target columns are read only on this explicit evaluation path.
    held_times, held60, _, held_ids = read_gauge_subset(
        paths["gauge"], stations.heldout, EVALUATION_YEARS
    )
    if not np.array_equal(gauge_times, held_times):
        raise RuntimeError("fitting context and heldout truth time axes differ")
    if not np.array_equal(
        held_ids, stations.heldout.station_id.to_numpy(dtype=np.int64)
    ):
        raise RuntimeError("heldout target order mismatch")
    all_target = np.concatenate(
        [np.full_like(train60, np.nan), held60], axis=1
    ).astype(np.float32)
    radar_index = discover_tiff_index(paths["radar"], EVALUATION_YEARS)
    issues = build_issue_times(radar_index, gauge_times, require_target_horizon=True)
    dataset = R2PWindowDataset(
        radar_index, gauge_times, eval_features, all_target, issues
    )
    return dataset


def save_prediction_store(
    destination: Path,
    predictions: np.ndarray,
    truth: np.ndarray,
    issues: np.ndarray,
    station_ids: np.ndarray,
    metadata: Mapping,
    *,
    overwrite: bool,
) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        _validate_prediction_store_for_overwrite(destination)
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=destination.name + ".partial.", dir=destination.parent))
    try:
        np.save(partial / "preds_mm.npy", predictions.astype(np.float32), allow_pickle=False)
        np.save(partial / "trues_mm.npy", truth.astype(np.float32), allow_pickle=False)
        np.save(partial / "anchor_times_ns.npy", issues.astype(np.int64), allow_pickle=False)
        np.save(partial / "station_ids.npy", station_ids.astype(np.int64), allow_pickle=False)
        np.save(partial / "lead_minutes.npy", LEAD_MINUTES.astype(np.int16), allow_pickle=False)
        write_json(partial / "metadata.json", dict(metadata, completed=True))
        write_json(partial / "COMPLETED", {"completed": True, "contract": CONTRACT_NAME})
        os.replace(partial, destination)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def run_evaluate(args: argparse.Namespace) -> None:
    device = require_device(args.device)
    paths = resolve_paths(args)
    archive, stations = verify_inputs(paths, not args.allow_unfrozen_contract)
    output = paths["output"]
    if not (output / "scientific_contract.json").is_file():
        raise RuntimeError("training contract is missing")
    training_contract = json.loads(
        (output / "scientific_contract.json").read_text(encoding="utf-8")
    )
    if training_contract.get("contract") != CONTRACT_NAME:
        raise RuntimeError("training contract name mismatch")
    scaler = GaugeScaler(
        minimum=np.load(output / "scaler_min.npy", allow_pickle=False),
        maximum=np.load(output / "scaler_max.npy", allow_pickle=False),
    )
    dataset = prepare_evaluation_data(paths, archive, stations, scaler)
    evaluation_mode = "target_masked"
    if args.mask_all_context_gauges:
        # Same-checkpoint radar-only sensitivity experiment used by Fig. 11a.
        # Keep station coordinates and the complete radar input unchanged, but
        # replace every issuance-time gauge history (including fitting514) by
        # the explicit missing-observation representation.  The held-out128
        # columns already have this representation under the normal contract.
        dataset.station_features[:, :, :2] = 0.0
        dataset.station_features[:, :, 2] = 1.0
        if not np.all(dataset.station_features[:, :, :2] == 0.0):
            raise RuntimeError("radar-only evaluation retained gauge values")
        if not np.all(dataset.station_features[:, :, 2] == 1.0):
            raise RuntimeError("radar-only evaluation missing flags are incomplete")
        evaluation_mode = "radar_only_all_context_gauges_masked"
    loader = DataLoader(
        dataset,
        batch_size=args.evaluation_batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=args.validation_workers,
        pin_memory=True,
        persistent_workers=args.validation_workers > 0,
    )
    config = R2PModelConfig()
    all_station_count = len(stations.all_stations)
    heldout_slice = slice(len(stations.train), all_station_count)
    checkpoint_name = str(args.checkpoint_name)
    checkpoint_path = Path(checkpoint_name)
    if (
        checkpoint_path.name != checkpoint_name
        or checkpoint_path.suffix != ".pt"
        or checkpoint_name in {".", ".."}
    ):
        raise ValueError(
            "--checkpoint-name must be a .pt filename inside each seed directory"
        )
    evaluation_tag = args.evaluation_tag
    if evaluation_tag is None and args.mask_all_context_gauges:
        evaluation_tag = "radar_only"
    elif evaluation_tag is None and checkpoint_name != "best.pt":
        evaluation_tag = checkpoint_path.stem
    if evaluation_tag is not None:
        evaluation_tag = str(evaluation_tag).strip()
        if not evaluation_tag or not all(
            character.isalnum() or character in {"-", "_"}
            for character in evaluation_tag
        ):
            raise ValueError(
                "--evaluation-tag may contain only letters, numbers, '-' and '_'"
            )
    output_suffix = "" if evaluation_tag is None else f"_{evaluation_tag}"
    for seed in args.seeds:
        seed_dir = output / f"seed_{seed}"
        marker_path = seed_dir / "COMPLETED"
        checkpoint = seed_dir / checkpoint_name
        if not marker_path.is_file() or not checkpoint.is_file():
            raise RuntimeError(
                f"seed {seed} is not completed or checkpoint is missing: {checkpoint}"
            )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        checkpoint_sha256 = sha256_file(checkpoint)
        if (
            checkpoint_name == "best.pt"
            and marker.get("best_checkpoint_sha256") != checkpoint_sha256
        ):
            raise RuntimeError("best checkpoint changed after completion")
        payload = _load_trusted_local_checkpoint(
            checkpoint, trusted_root=output, purpose="evaluation"
        )
        if payload.get("contract") != CONTRACT_NAME or int(payload.get("seed")) != seed:
            raise RuntimeError("checkpoint contract mismatch")
        if payload.get("contract_sha256") != contract_digest(training_contract):
            raise RuntimeError("checkpoint/scientific-contract digest mismatch")
        model = model_for_stations(stations.all_stations, config, device)
        load_query_expanded_state(model, payload["model_state_dict"])
        prediction, truth, issues = collate_prediction(
            model, loader, scaler, device, station_slice=heldout_slice
        )
        metrics = compute_metrics_arrays(prediction, truth)
        metrics_frame = pd.DataFrame(metrics)
        metrics_frame.insert(0, "seed", seed)
        metrics_frame.to_csv(
            seed_dir / f"evaluation_metrics_by_lead{output_suffix}.csv", index=False
        )
        destination = seed_dir / (
            f"target_masked_heldout128_predictions{output_suffix}"
        )
        save_prediction_store(
            destination,
            prediction,
            truth,
            issues,
            stations.heldout.station_id.to_numpy(dtype=np.int64),
            {
                "contract": CONTRACT_NAME,
                "seed": seed,
                "mode": evaluation_mode,
                "all_context_gauges_masked": bool(args.mask_all_context_gauges),
                "checkpoint_name": checkpoint_name,
                "checkpoint_sha256": checkpoint_sha256,
                "selected_epoch": int(payload["epoch"]),
                "evaluation_tag": evaluation_tag,
                "radar_offsets_min": RADAR_OFFSETS_MIN.astype(int).tolist(),
                "gauge_offsets_min": GAUGE_OFFSETS_MIN.astype(int).tolist(),
                "lead_minutes": LEAD_MINUTES.astype(int).tolist(),
                "prediction_clipping": "max(prediction_mm, 0)",
            },
            overwrite=args.overwrite,
        )
        print(
            f"seed={seed}: evaluated {len(issues):,} anchors with "
            f"{checkpoint_name} -> {destination}"
        )
        del model, prediction, truth
        torch.cuda.empty_cache() if device.type == "cuda" else None
        gc.collect()


def run_inspect(args: argparse.Namespace) -> None:
    paths = resolve_paths(args)
    archive, stations = verify_inputs(paths, not args.allow_unfrozen_contract)
    config = R2PModelConfig()
    model = model_for_stations(stations.train, config, torch.device("cpu"))
    result = {
        "contract": CONTRACT_NAME,
        "paths": {key: str(value) for key, value in paths.items()},
        "train_stations": len(stations.train),
        "heldout_stations": len(stations.heldout),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": config.to_dict(),
        "radar_offsets_min": RADAR_OFFSETS_MIN.astype(int).tolist(),
        "gauge_offsets_min": GAUGE_OFFSETS_MIN.astype(int).tolist(),
        "lead_minutes": LEAD_MINUTES.astype(int).tolist(),
        "grid_valid_cells": int(archive["source_valid"].sum()),
    }
    print(json.dumps(result, indent=2))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--gauge-csv", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--stations-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-unfrozen-contract", action="store_true")


def parse_seeds(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("seeds must be unique nonnegative integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "preflight"):
        subparser = subparsers.add_parser(name)
        add_common_arguments(subparser)
    smoke = subparsers.add_parser("memory-smoke")
    add_common_arguments(smoke)
    smoke.add_argument("--batch-size", type=int, default=16)
    smoke.add_argument("--smoke-stations", type=int, default=514)

    train = subparsers.add_parser("train")
    add_common_arguments(train)
    train.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2])
    train.add_argument("--base-seed", type=int, default=40)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--validation-batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--station-dropout-min", type=float, default=0.0)
    train.add_argument("--station-dropout-max", type=float, default=0.9)
    train.add_argument("--station-dropout-full-probability", type=float, default=0.05)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--validation-workers", type=int, default=1)
    train.add_argument("--progress-every", type=int, default=250)
    train.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume from this runner's trusted local last.pt checkpoint; full-state "
            "checkpoints use Python pickle semantics and must never be downloaded"
        ),
    )
    train.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    add_common_arguments(evaluate)
    evaluate.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2])
    evaluate.add_argument("--evaluation-batch-size", type=int, default=32)
    evaluate.add_argument("--validation-workers", type=int, default=1)
    evaluate.add_argument(
        "--mask-all-context-gauges",
        action="store_true",
        help=(
            "same-checkpoint radar-only sensitivity: zero RN60/RN15 and set "
            "missing=1 for every context station; coordinates remain unchanged"
        ),
    )
    evaluate.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help=(
            "checkpoint filename inside each seed directory; non-best checkpoints "
            "are written to a tagged evaluation output"
        ),
    )
    evaluate.add_argument(
        "--evaluation-tag",
        help=(
            "optional output suffix; defaults to the checkpoint stem when "
            "--checkpoint-name is not best.pt"
        ),
    )
    evaluate.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        run_inspect(args)
    elif args.command == "preflight":
        run_preflight(args)
    elif args.command == "memory-smoke":
        run_memory_smoke(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
