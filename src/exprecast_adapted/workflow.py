#!/usr/bin/env python3
"""Train, validate, and export the adapted primary 4-km/10-min exPreCast.

The short stage is direct 7-to-6 forecasting; the long stage is direct 7-to-18.
The internal best.pt selects last-lead instantaneous CSI and is used only for
short-stage initialization. Final long selection is performed separately by
exprecast_adapted.selection using 2023 native-grid rolling RN60.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    ChunkLocalRandomSampler,
    CropContract,
    KmaTiffWindowDataset,
    dbz_to_rain_rate,
    estimate_full_grid_export_gib,
)
from .model import (
    ExPreCast5MinConfig,
    build_exprecast_5min,
    sha256_file,
)


KMA_THRESHOLDS_MM_H = (1.0, 4.0, 8.0, 10.0, 20.0, 40.0, 80.0)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_years(value: str) -> tuple[int, ...]:
    values = parse_int_tuple(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one year is required")
    return values


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def torch_save_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def assert_finite_tensor_tree(value: Any, label: str, path: str = "") -> None:
    """Reject a checkpoint tree containing nonfinite floating-point state."""

    if torch.is_tensor(value):
        is_numeric = value.is_floating_point() or value.is_complex()
        if is_numeric and not torch.isfinite(value).all():
            location = f"{label}.{path}" if path else label
            raise FloatingPointError(f"nonfinite tensor in {location}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            assert_finite_tensor_tree(item, label, child)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{path}[{index}]"
            assert_finite_tensor_tree(item, label, child)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def model_config_from_args(args: argparse.Namespace) -> ExPreCast5MinConfig:
    depths = parse_int_tuple(args.depths)
    heads = parse_int_tuple(args.num_heads)
    window = parse_int_tuple(args.window_size)
    if len(window) != 3:
        raise ValueError("--window-size must contain temporal,height,width")
    return ExPreCast5MinConfig(
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        frame_minutes=args.frame_minutes,
        profile=args.profile,
        embed_dim=args.embed_dim,
        depths=depths,
        num_heads=heads,
        window_size=window,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        use_checkpoint=args.use_checkpoint,
    )


def crop_from_args(args: argparse.Namespace) -> CropContract:
    return CropContract(
        height=args.grid_height,
        width=args.grid_width,
        y0=args.crop_y0,
        x0=args.crop_x0,
        spatial_factor=args.spatial_factor,
        spatial_reduction=args.spatial_reduction,
    )


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class StableFACL(torch.nn.Module):
    """Official Fourier Amplitude/Correlation Loss with finite dry-batch gradients."""

    def __init__(self, total_steps: int, transition_fraction: float = 0.60, eps: float = 1e-8):
        super().__init__()
        if eps <= 0:
            raise ValueError("StableFACL eps must be positive")
        self.total_steps = int(total_steps)
        self.transition_steps = max(1, int(total_steps * float(transition_fraction)))
        self.eps = float(eps)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, optimizer_step: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # FFT is deliberately kept in fp32 even when the model uses AMP.
        pred_fft = torch.fft.fftn(prediction.float(), dim=(-2, -1), norm="ortho")
        target_fft = torch.fft.fftn(target.float(), dim=(-2, -1), norm="ortho")
        amplitude = F.mse_loss(pred_fft.abs(), target_fft.abs())
        numerator = (torch.conj(pred_fft) * target_fft).sum().real
        target_energy = target_fft.abs().square().sum()
        prediction_energy = pred_fft.abs().square().sum()
        # Stabilize *inside* sqrt.  Clamping sqrt(energy_product) afterwards
        # keeps the forward value finite but still differentiates through
        # sqrt(0), whose infinite derivative produces NaN gradients on an
        # all-zero target or prediction batch.
        denominator = torch.sqrt(
            target_energy * prediction_energy + self.eps**2
        )
        correlation = 1.0 - numerator / denominator
        transition_denominator = max(1, self.transition_steps - 1)
        alpha = min(
            1.0,
            max(0.0, float(optimizer_step) / transition_denominator),
        )
        weight = math.sqrt(prediction.shape[-2] * prediction.shape[-1])
        loss = weight * (alpha * amplitude + (1.0 - alpha) * correlation)
        return loss, {
            "loss": float(loss.detach()),
            "fal": float(amplitude.detach()),
            "fcl": float(correlation.detach()),
            "fal_weight": alpha,
        }


def lr_multiplier(step: int, total_steps: int, warmup_ratio: float, minimum_ratio: float) -> float:
    warmup = max(1, int(total_steps * warmup_ratio))
    if step <= warmup:
        # Match the official scheduler: the initial optimizer LR is exactly
        # zero and rises linearly to the base LR at the warm-up boundary.
        return float(step) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _pool_max(value: torch.Tensor, size: int) -> torch.Tensor:
    if size == 1:
        return value
    return F.max_pool2d(value, kernel_size=size, stride=max(1, math.ceil(size / 4)))


class StreamingCSI:
    def __init__(
        self,
        output_frames: int,
        thresholds: Iterable[float] = KMA_THRESHOLDS_MM_H,
        pool_sizes: Iterable[int] = (1, 4, 16),
    ) -> None:
        self.thresholds = tuple(float(value) for value in thresholds)
        self.pool_sizes = tuple(int(value) for value in pool_sizes)
        self.counts = torch.zeros(
            len(self.pool_sizes), output_frames, len(self.thresholds), 3, dtype=torch.float64
        )

    @torch.no_grad()
    def update(
        self,
        prediction_norm: torch.Tensor,
        target_norm: torch.Tensor,
        evaluation_mask: torch.Tensor | None = None,
    ) -> None:
        prediction = dbz_to_rain_rate((prediction_norm[:, 0] * 100.0).clamp_min(0)).cpu()
        target = dbz_to_rain_rate((target_norm[:, 0] * 100.0).clamp_min(0)).cpu()
        if evaluation_mask is not None:
            prediction = prediction * evaluation_mask
            target = target * evaluation_mask
        for pool_index, pool_size in enumerate(self.pool_sizes):
            for lead in range(prediction.shape[1]):
                pred = _pool_max(prediction[:, lead : lead + 1], pool_size)
                obs = _pool_max(target[:, lead : lead + 1], pool_size)
                for threshold_index, threshold in enumerate(self.thresholds):
                    pred_event = pred >= threshold
                    obs_event = obs >= threshold
                    self.counts[pool_index, lead, threshold_index, 0] += (pred_event & obs_event).sum()
                    self.counts[pool_index, lead, threshold_index, 1] += (~pred_event & obs_event).sum()
                    self.counts[pool_index, lead, threshold_index, 2] += (pred_event & ~obs_event).sum()

    def result(self, frame_minutes: int) -> dict[str, Any]:
        denominator = self.counts[..., 0] + self.counts[..., 1] + self.counts[..., 2]
        # Official compute_csi uses +1e-6, so empty event cells are zero and
        # remain part of CSI-M rather than being removed by a NaN mean.
        csi = self.counts[..., 0] / (denominator + 1e-6)
        last_frame_csi_m = torch.mean(csi[0, -1]).item()
        return {
            "thresholds_mm_h": list(self.thresholds),
            "pool_sizes": list(self.pool_sizes),
            "lead_minutes": list(range(frame_minutes, csi.shape[1] * frame_minutes + 1, frame_minutes)),
            "csi": csi.numpy().tolist(),
            "last_frame_pool1_csi_m": last_frame_csi_m,
            "counts_hit_miss_false_alarm": self.counts.numpy().tolist(),
        }


def make_dataset(
    args: argparse.Namespace,
    years: tuple[int, ...],
    *,
    require_targets: bool,
    anchor_stride: int = 1,
    max_windows: int | None = None,
) -> KmaTiffWindowDataset:
    common = {
        "input_frames": args.input_frames,
        "output_frames": args.output_frames,
        "frame_minutes": args.frame_minutes,
        "crop": crop_from_args(args),
        "require_targets": require_targets,
        "anchor_stride": anchor_stride,
        "max_windows": max_windows,
    }
    if args.radar_format == "kma-tiff":
        return KmaTiffWindowDataset(
            args.radar_root,
            years,
            supplemental_roots=args.supplemental_radar_root,
            **common,
        )
    raise ValueError(f"unsupported radar format: {args.radar_format}")


def load_evaluation_mask(
    path: Path | None,
    dataset: KmaTiffWindowDataset,
    key: str | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if path is None:
        return None, {
            "path": None,
            "mode": "unmasked; provide the primary source-validity mask for selection",
        }
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            chosen = key
            if chosen is None:
                if len(loaded.files) != 1:
                    raise ValueError(
                        f"evaluation NPZ has keys {loaded.files}; provide --evaluation-mask-key"
                    )
                chosen = loaded.files[0]
            if chosen not in loaded.files:
                raise KeyError(f"evaluation mask key {chosen!r} not in {loaded.files}")
            value = np.asarray(loaded[chosen], dtype=np.float32).squeeze()
        finally:
            loaded.close()
    else:
        if key is not None:
            raise ValueError("--evaluation-mask-key is valid only for an NPZ mask")
        value = np.asarray(loaded, dtype=np.float32).squeeze()
    if value.ndim != 2:
        raise ValueError(f"evaluation mask must be 2-D after squeeze, got {value.shape}")
    crop = dataset.crop
    if tuple(value.shape) == (crop.native_height, crop.native_width):
        mask = torch.from_numpy(value)[None, None]
        if crop.spatial_factor > 1:
            mask = F.interpolate(
                mask,
                size=(crop.reduced_height, crop.reduced_width),
                mode="nearest",
            )
        mask = mask[
            :,
            :,
            crop.y0 : crop.y0 + crop.height,
            crop.x0 : crop.x0 + crop.width,
        ]
    elif tuple(value.shape) == (crop.reduced_height, crop.reduced_width):
        mask = torch.from_numpy(value)[
            None,
            None,
            crop.y0 : crop.y0 + crop.height,
            crop.x0 : crop.x0 + crop.width,
        ]
    elif tuple(value.shape) == (crop.height, crop.width):
        mask = torch.from_numpy(value)[None, None]
    else:
        raise ValueError(
            f"mask shape {value.shape} matches neither native, reduced nor crop grid"
        )
    mask = (mask > 0).to(torch.float32)
    return mask, {
        "path": str(Path(path).resolve()),
        "key": key,
        "sha256": sha256_file(Path(path)),
        "source_shape": list(value.shape),
        "crop_shape": [crop.height, crop.width],
        "valid_fraction": float(mask.mean()),
    }


def make_loader(
    dataset: KmaTiffWindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    samples_per_epoch: int | None,
    locality_span: int,
    seed: int,
    num_workers: int,
) -> DataLoader:
    sampler = None
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        sampler = ChunkLocalRandomSampler(
            dataset,
            num_samples=samples_per_epoch,
            locality_span=locality_span,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )


def load_requested_anchor_times(path: Path, key: str) -> np.ndarray:
    """Load a one-dimensional, unique issue-time axis as int64 nanoseconds."""

    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if key not in loaded.files:
                raise KeyError(f"anchor-time key {key!r} not in {path}: {loaded.files}")
            values = np.asarray(loaded[key])
        finally:
            loaded.close()
    else:
        values = np.asarray(loaded)
    if values.ndim != 1:
        raise ValueError(f"requested anchor times must be one-dimensional: {values.shape}")
    if np.issubdtype(values.dtype, np.datetime64):
        values = values.astype("datetime64[ns]").astype(np.int64)
    elif np.issubdtype(values.dtype, np.integer):
        values = values.astype(np.int64, copy=False)
        if len(values) and np.max(np.abs(values)) < 10**17:
            raise ValueError("integer anchor times must be int64 nanoseconds")
    else:
        raise TypeError(f"unsupported anchor-time dtype {values.dtype}: {path}")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"requested anchor-time axis contains duplicates: {path}")
    if len(values) > 1 and np.any(np.diff(values) <= 0):
        raise ValueError(f"requested anchor-time axis must be strictly increasing: {path}")
    return values


def select_export_anchor_times(
    dataset: KmaTiffWindowDataset,
    requested_path: Path,
    requested_key: str,
    expected_count: int | None,
) -> dict[str, Any]:
    """Restrict an export dataset to exact requested issue times in request order."""

    requested = load_requested_anchor_times(requested_path, requested_key)
    available = np.asarray(
        [dataset._times[year][anchor] for year, anchor in dataset.references],
        dtype=np.int64,
    )
    _, source_rows, requested_rows = np.intersect1d(
        available, requested, assume_unique=True, return_indices=True
    )
    order = np.argsort(requested_rows, kind="stable")
    source_rows = source_rows[order]
    requested_rows = requested_rows[order]
    selected_times = available[source_rows]
    if not np.array_equal(selected_times, requested[requested_rows]):
        raise RuntimeError("exact export-anchor alignment failed")
    if expected_count is not None and len(source_rows) != int(expected_count):
        raise RuntimeError(
            f"export anchor intersection changed: {len(source_rows)} != {expected_count}"
        )
    if len(source_rows) > 1 and (
        np.any(np.diff(source_rows) <= 0) or np.any(np.diff(requested_rows) <= 0)
    ):
        raise RuntimeError("export anchor intersection changed chronological order")
    dataset.references = tuple(dataset.references[int(row)] for row in source_rows)
    if not dataset.references:
        raise RuntimeError("no requested anchor times are exportable")
    return {
        "requested_path": str(requested_path.resolve()),
        "requested_file_sha256": sha256_file(requested_path),
        "requested_key": requested_key,
        "requested_count": len(requested),
        "selected_count": len(source_rows),
        "missing_count": len(requested) - len(source_rows),
        "selected_issue_time_sha256": hashlib.sha256(
            np.ascontiguousarray(selected_times).tobytes()
        ).hexdigest(),
    }


def construct_model(args: argparse.Namespace, device: torch.device):
    model, metadata = build_exprecast_5min(
        model_config_from_args(args),
        args.official_repo,
        initialization_checkpoint=args.initialization_checkpoint,
        encoder_only_initialization=args.encoder_only_initialization,
        freeze_encoder_parameters=args.freeze_encoder,
        trust_checkpoint=getattr(args, "trust_checkpoint", False),
    )
    return model.to(device), metadata


def checkpoint_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"invalid checkpoint object: {type(payload).__name__}")
    value = payload.get("model", payload)
    if not isinstance(value, Mapping):
        raise RuntimeError("checkpoint contains no model state mapping")
    return value


def load_trained_model(args: argparse.Namespace, device: torch.device):
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if not getattr(args, "trust_checkpoint", False):
        raise ValueError(
            "checkpoint deserialization can execute Python code; pass "
            "--trust-checkpoint only for a checkpoint you trust"
        )
    model, metadata = build_exprecast_5min(model_config_from_args(args), args.official_repo)
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint_state_dict(payload), strict=True)
    model.to(device).eval()
    metadata["checkpoint"] = str(args.checkpoint.resolve())
    metadata["checkpoint_sha256"] = sha256_file(args.checkpoint)
    return model, metadata, payload


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    frame_minutes: int,
    max_batches: int | None,
    evaluation_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    metrics = StreamingCSI(model.output_frames)
    mse_sum = torch.zeros(model.output_frames, dtype=torch.float64)
    value_count = 0
    started = time.time()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            predictions = model(inputs)
        error = (predictions.float() - targets.float()).square()
        mse_sum += error.sum(dim=(0, 1, 3, 4)).cpu().double()
        value_count += error.shape[0] * error.shape[-2] * error.shape[-1]
        metrics.update(predictions.float(), targets.float(), evaluation_mask)
        print(f"validation batch {batch_index + 1}/{len(loader)}", end="\r", flush=True)
    print()
    if value_count == 0:
        raise RuntimeError("validation loader yielded no batches")
    result = metrics.result(frame_minutes)
    result.update(
        {
            "normalized_dbz_mse_by_lead": (mse_sum / value_count).tolist(),
            "evaluated_values_per_lead": value_count,
            "elapsed_seconds": time.time() - started,
        }
    )
    return result


def run_inspect(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, metadata = construct_model(args, device)
    dataset = make_dataset(args, args.train_years, require_targets=True, max_windows=1)
    payload = {
        "model": metadata,
        "sample_dataset": dataset.metadata(),
        "device": str(device),
        "recommended_memory_smoke_grid": [128, 128]
        if args.profile == "paper_long_concat"
        else [args.grid_height, args.grid_width],
    }
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True))
    del model


def run_memory_smoke(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("memory-smoke requires CUDA")
    model, metadata = construct_model(args, device)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    inputs = torch.rand(
        args.batch_size,
        1,
        args.input_frames,
        args.grid_height,
        args.grid_width,
        device=device,
    )
    targets = torch.rand(
        args.batch_size,
        1,
        args.output_frames,
        args.grid_height,
        args.grid_width,
        device=device,
    )
    criterion = StableFACL(args.total_steps)
    with autocast_context(device, args.precision):
        predictions = model(inputs)
        loss, components = criterion(predictions, targets, 0)
    loss.backward()
    payload = {
        "model": metadata,
        "input_shape": list(inputs.shape),
        "output_shape": list(predictions.shape),
        "loss": components,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True))


def run_train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)
    schedule_total_steps = (
        args.total_steps
        if args.schedule_total_steps is None
        else args.schedule_total_steps
    )
    if args.total_steps <= 0 or schedule_total_steps <= 0:
        raise ValueError("training and schedule step counts must be positive")
    run_dir = args.output_root / args.run_name / f"seed_{args.seed}"
    if (args.initialization_checkpoint is not None or (args.resume and (run_dir / "last.pt").is_file())) and not getattr(args, "trust_checkpoint", False):
        raise ValueError("initialization/resume requires --trust-checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    model, model_metadata = construct_model(args, device)
    train_dataset = make_dataset(args, args.train_years, require_targets=True)
    validation_dataset = make_dataset(
        args,
        args.validation_years,
        require_targets=True,
        anchor_stride=args.validation_anchor_stride,
        max_windows=args.validation_max_windows,
    )
    loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        samples_per_epoch=args.samples_per_epoch,
        locality_span=args.locality_span,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        samples_per_epoch=None,
        locality_span=1,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    evaluation_mask, evaluation_mask_metadata = load_evaluation_mask(
        args.evaluation_mask, validation_dataset, args.evaluation_mask_key
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(
            step, schedule_total_steps, args.warmup_ratio, args.minimum_lr_ratio
        ),
    )
    scaler_enabled = device.type == "cuda" and args.precision == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    criterion = StableFACL(schedule_total_steps, eps=args.facl_eps)
    global_step = 0
    best_score = -math.inf
    epoch = 0
    if args.resume and (run_dir / "last.pt").is_file():
        payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        assert_finite_tensor_tree(payload.get("model"), "resume model state")
        assert_finite_tensor_tree(payload.get("optimizer"), "resume optimizer state")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        global_step = int(payload["global_step"])
        best_score = float(payload.get("best_score", best_score))
        epoch = int(payload.get("epoch", 0))
        print(f"resumed {run_dir / 'last.pt'} at optimizer step {global_step}")
    run_contract = {
        "model": model_metadata,
        "train_data": train_dataset.metadata(),
        "validation_data": validation_dataset.metadata(),
        "training": {
            "total_optimizer_steps": args.total_steps,
            "optimization_schedule_steps": schedule_total_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "micro_batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "warmup_ratio": args.warmup_ratio,
            "scheduler": "warmup_cosine",
            "loss": "FACL with epsilon inside the correlation square root",
            "nonfinite_policy": (
                "abort before optimizer.step on a nonfinite loss or gradient norm"
            ),
            "max_grad_norm": args.max_grad_norm,
            "gradient_accumulation_caveat": (
                "FACL correlation is a nonlinear global ratio; accumulating "
                "microbatch FACL gradients is not mathematically identical to "
                "computing FACL once on the effective batch"
            ),
            "precision": args.precision,
            "seed": args.seed,
            "save_every_epoch": args.save_every_epoch,
            "epoch_checkpoint_contract": (
                "completed data-pass checkpoints contain model weights and metadata; "
                "last.pt remains the full optimizer-state resume checkpoint"
            ),
            "resume_caveat": (
                "resume restores model/optimizer/scheduler/scaler but restarts "
                "the chunk-local sampler; it is failure recovery, not a bitwise "
                "continuation of the original sample trajectory"
            ),
        },
        "evaluation_mask": evaluation_mask_metadata,
    }
    write_json_atomic(run_dir / "run_contract.json", run_contract)
    log_path = run_dir / "training.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    started = time.time()
    while global_step < args.total_steps:
        epoch += 1
        epoch_batches = 0
        for batch in loader:
            epoch_batches += 1
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            with autocast_context(device, args.precision):
                predictions = model(inputs)
                loss, components = criterion(predictions, targets, global_step)
                scaled_loss = loss / args.gradient_accumulation
            if not math.isfinite(components["loss"]):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "nonfinite FACL loss before backward at "
                    f"optimizer_step={global_step}, epoch={epoch}, "
                    f"microbatch={accumulation + 1}/{args.gradient_accumulation}; "
                    f"components={components}"
                )
            scaler.scale(scaled_loss).backward()
            accumulation += 1
            if accumulation < args.gradient_accumulation:
                continue
            # Always inspect the unscaled global gradient norm before the
            # update.  max_norm=inf is a finite check that preserves the
            # official optimizer when clipping is disabled.
            scaler.unscale_(optimizer)
            max_norm = args.max_grad_norm if args.max_grad_norm > 0 else math.inf
            try:
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    max_norm=max_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "nonfinite accumulated gradient before optimizer.step at "
                    f"optimizer_step={global_step}, epoch={epoch}"
                ) from error
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulation = 0
            global_step += 1
            record = {
                "global_step": global_step,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - started,
                **components,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if global_step == 1 or global_step % args.log_every == 0:
                print(
                    f"step {global_step}/{args.total_steps} loss={components['loss']:.6g} "
                    f"lr={record['learning_rate']:.3g} epoch={epoch}",
                    flush=True,
                )
            validation = None
            if global_step % args.validate_every == 0 or global_step == args.total_steps:
                validation = evaluate_model(
                    model,
                    validation_loader,
                    device=device,
                    precision=args.precision,
                    frame_minutes=args.frame_minutes,
                    max_batches=args.validation_max_batches,
                    evaluation_mask=evaluation_mask,
                )
                validation["global_step"] = global_step
                write_json_atomic(run_dir / f"validation_step_{global_step:07d}.json", validation)
                score = float(validation["last_frame_pool1_csi_m"])
                model.train()
                if score > best_score:
                    best_score = score
                    torch_save_atomic(
                        run_dir / "best.pt",
                        {
                            "model": model.state_dict(),
                            "global_step": global_step,
                            "best_score": best_score,
                            "contract": run_contract,
                            "selection_metric": "validation last-frame pool1 CSI-M",
                        },
                    )
            if global_step % args.save_every == 0 or global_step == args.total_steps:
                torch_save_atomic(
                    run_dir / "last.pt",
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_score": best_score,
                        "contract": run_contract,
                    },
                )
            if global_step >= args.total_steps:
                break
        epoch_complete = epoch_batches == len(loader)
        if args.save_every_epoch and epoch_complete:
            epoch_path = (
                run_dir
                / "epoch_checkpoints"
                / f"epoch_{epoch:04d}_step_{global_step:07d}.pt"
            )
            torch_save_atomic(
                epoch_path,
                {
                    "model": model.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "epoch_complete": True,
                    "pending_gradient_microbatches": accumulation,
                    "best_score": best_score,
                    "contract": run_contract,
                    "checkpoint_kind": "epoch_weights_only",
                    "resume_checkpoint": str(run_dir / "last.pt"),
                },
            )
            print(f"saved completed-epoch weights: {epoch_path}", flush=True)
    print(f"training complete: {run_dir}; best validation CSI-M={best_score:.6f}")


def run_evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, metadata, _ = load_trained_model(args, device)
    dataset = make_dataset(
        args,
        args.evaluation_years,
        require_targets=True,
        anchor_stride=args.evaluation_anchor_stride,
        max_windows=args.evaluation_max_windows,
    )
    loader = make_loader(
        dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        samples_per_epoch=None,
        locality_span=1,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    evaluation_mask, evaluation_mask_metadata = load_evaluation_mask(
        args.evaluation_mask, dataset, args.evaluation_mask_key
    )
    result = evaluate_model(
        model,
        loader,
        device=device,
        precision=args.precision,
        frame_minutes=args.frame_minutes,
        max_batches=args.evaluation_max_batches,
        evaluation_mask=evaluation_mask,
    )
    result.update(
        {
            "model": metadata,
            "data": dataset.metadata(),
            "evaluation_mask": evaluation_mask_metadata,
        }
    )
    write_json_atomic(args.evaluation_output, result)
    print(f"wrote {args.evaluation_output}")


def _open_export(
    path: Path,
    dataset: KmaTiffWindowDataset,
    args: argparse.Namespace,
    contract: Mapping[str, Any],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    output_dtype = np.dtype(args.export_dtype)
    estimated = estimate_full_grid_export_gib(
        len(dataset), args.output_frames, dataset.crop.height, dataset.crop.width, output_dtype.name
    )
    if estimated > args.maximum_export_gib and not args.allow_large_export:
        raise RuntimeError(
            f"estimated forecast array is {estimated:.1f} GiB, above "
            f"--maximum-export-gib={args.maximum_export_gib}; use a stride/smaller crop "
            "or pass --allow-large-export deliberately"
        )
    contract_json = json.dumps(jsonable(contract), sort_keys=True)
    if path.exists() and not args.overwrite:
        if not args.resume_export:
            raise FileExistsError(
                f"output exists; use --resume-export or --overwrite: {path}"
            )
        stream = h5py.File(path, "r+")
        expected_shape = (
            len(dataset),
            args.output_frames,
            dataset.crop.height,
            dataset.crop.width,
        )
        if "completed" not in stream or "forecast_normalized_dbz" not in stream:
            stream.close()
            raise RuntimeError(f"existing export has no resumable contract: {path}")
        if tuple(stream["forecast_normalized_dbz"].shape) != expected_shape:
            stream.close()
            raise RuntimeError("existing export shape differs from the current contract")
        if stream.attrs.get("contract_json") != contract_json:
            stream.close()
            raise RuntimeError("existing export metadata differs from the current contract")
        completed = np.asarray(stream["completed"][:], dtype=bool)
        first_incomplete = int(np.flatnonzero(~completed)[0]) if not completed.all() else len(completed)
        if completed[first_incomplete:].any():
            stream.close()
            raise RuntimeError("completed rows are non-contiguous; refusing ambiguous resume")
        return stream, first_incomplete
    stream = h5py.File(path, "w")
    stream.create_dataset(
        "forecast_normalized_dbz",
        shape=(len(dataset), args.output_frames, dataset.crop.height, dataset.crop.width),
        dtype=output_dtype,
        chunks=(1, 1, dataset.crop.height, dataset.crop.width),
        compression="lzf",
    )
    stream.create_dataset("issue_time_ns", shape=(len(dataset),), dtype="i8")
    stream.create_dataset("completed", shape=(len(dataset),), dtype="bool", fillvalue=False)
    stream.create_dataset(
        "lead_minutes",
        data=np.arange(1, args.output_frames + 1, dtype=np.int16) * args.frame_minutes,
    )
    stream.attrs["encoding"] = "dBZ = 100 * forecast_normalized_dbz"
    stream.attrs["issue_time_timezone"] = "KST"
    stream.attrs["estimated_uncompressed_gib"] = estimated
    stream.attrs["contract_json"] = contract_json
    stream.flush()
    return stream, 0


@torch.no_grad()
def run_export(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, metadata, _ = load_trained_model(args, device)
    dataset = make_dataset(
        args,
        args.export_years,
        require_targets=False,
        anchor_stride=args.export_anchor_stride,
        max_windows=args.export_max_windows,
    )
    anchor_selection = None
    if args.export_anchor_times is not None:
        if args.export_anchor_stride != 1:
            raise ValueError("--export-anchor-times requires --export-anchor-stride 1")
        if args.export_max_windows is not None:
            raise ValueError("--export-anchor-times cannot be combined with --export-max-windows")
        anchor_selection = select_export_anchor_times(
            dataset,
            args.export_anchor_times,
            args.export_anchor_times_key,
            args.expected_export_anchors,
        )
        print(
            f"selected {anchor_selection['selected_count']:,}/"
            f"{anchor_selection['requested_count']:,} requested export anchors",
            flush=True,
        )
    contract = {
        "model": metadata,
        "data": dataset.metadata(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "lead_minutes": list(range(args.frame_minutes, args.output_frames * args.frame_minutes + 1, args.frame_minutes)),
        "anchor_selection": anchor_selection,
    }
    output, position = _open_export(args.export_output, dataset, args, contract)
    try:
        remaining = torch.utils.data.Subset(dataset, range(position, len(dataset)))
        loader = make_loader(
            remaining,
            batch_size=args.export_batch_size,
            shuffle=False,
            samples_per_epoch=None,
            locality_span=1,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        if position:
            print(f"resuming {args.export_output} at row {position}/{len(dataset)}")
        for batch_index, batch in enumerate(loader):
            inputs = batch["inputs"].to(device, non_blocking=True)
            with autocast_context(device, args.precision):
                predictions = model(inputs)
            value = predictions[:, 0].float().cpu().numpy().astype(args.export_dtype)
            count = len(value)
            output["forecast_normalized_dbz"][position : position + count] = value
            output["issue_time_ns"][position : position + count] = batch["issue_time_ns"].numpy()
            output.flush()
            output["completed"][position : position + count] = True
            position += count
            output.flush()
            print(f"export batch {batch_index + 1}/{len(loader)} ({position}/{len(dataset)})", flush=True)
        if position != len(dataset):
            raise RuntimeError(f"incomplete export: wrote {position}/{len(dataset)} windows")
    finally:
        output.close()
    write_json_atomic(args.export_output.with_suffix(".contract.json"), contract)
    print(f"wrote {args.export_output}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    root = Path.cwd()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--radar-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--radar-format",
        choices=("kma-tiff",),
        default="kma-tiff",
    )
    parser.set_defaults(supplemental_radar_root=[])
    parser.add_argument(
        "--official-repo", type=Path, required=True
    )
    parser.add_argument("--input-frames", type=int, choices=(7,), default=7)
    parser.add_argument("--output-frames", type=int, choices=(6, 18), default=18)
    parser.add_argument("--frame-minutes", type=int, choices=(10,), default=10)
    parser.add_argument(
        "--profile",
        choices=("paper_long_concat", "memory_safe_add"),
        default="paper_long_concat",
    )
    parser.add_argument("--grid-height", type=int, default=256)
    parser.add_argument("--grid-width", type=int, default=256)
    parser.add_argument("--crop-y0", type=int)
    parser.add_argument("--crop-x0", type=int)
    parser.add_argument("--spatial-factor", type=int, default=1)
    parser.add_argument(
        "--spatial-reduction", choices=("uniform", "area"), default="uniform"
    )
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--depths", default="2,6,2,2")
    parser.add_argument("--num-heads", default="3,6,12,24")
    parser.add_argument("--window-size", default="2,7,7")
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.2)
    parser.add_argument("--use-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6455)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--evaluation-mask", type=Path)
    parser.add_argument("--evaluation-mask-key", default="source_index_valid_mask")
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--encoder-only-initialization", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--trust-checkpoint", action="store_true",
        help="allow Python deserialization of a checkpoint that you trust",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    add_common_arguments(inspect_parser)
    inspect_parser.add_argument("--train-years", type=parse_years, default=(2019, 2020, 2021, 2022))
    inspect_parser.set_defaults(handler=run_inspect)

    smoke = subparsers.add_parser("memory-smoke")
    add_common_arguments(smoke)
    smoke.add_argument("--total-steps", type=int, default=100_000)
    smoke.set_defaults(handler=run_memory_smoke)

    train = subparsers.add_parser("train")
    add_common_arguments(train)
    train.add_argument("--train-years", type=parse_years, default=(2019, 2020, 2021, 2022))
    train.add_argument("--validation-years", type=parse_years, default=(2023,))
    train.add_argument("--output-root", type=Path, default=Path("outputs/exprecast"))
    train.add_argument("--run-name", required=True)
    train.add_argument("--total-steps", type=int, default=100_000)
    train.add_argument(
        "--schedule-total-steps",
        type=int,
        help=(
            "LR/FACL schedule horizon; defaults to --total-steps. Set this to "
            "100000 for a shorter stability run that matches the beginning of "
            "the official optimization schedule."
        ),
    )
    train.add_argument("--samples-per-epoch", type=int, default=0)
    train.add_argument("--locality-span", type=int, default=16)
    train.add_argument("--gradient-accumulation", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--warmup-ratio", type=float, default=0.2)
    train.add_argument("--minimum-lr-ratio", type=float, default=1e-3)
    train.add_argument(
        "--facl-eps",
        type=float,
        default=1e-8,
        help="epsilon added as eps^2 inside the FACL correlation square root",
    )
    train.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help=(
            "clip the unscaled global gradient norm when positive; zero disables "
            "clipping but still aborts on nonfinite gradients"
        ),
    )
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--save-every", type=int, default=1_000)
    train.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "retain a weights-only .pt after every completed pass through the "
            "training loader; last.pt remains the full resume checkpoint"
        ),
    )
    train.add_argument("--validate-every", type=int, default=5_000)
    train.add_argument("--validation-batch-size", type=int, default=1)
    train.add_argument("--validation-anchor-stride", type=int, default=12)
    train.add_argument("--validation-max-windows", type=int)
    train.add_argument("--validation-max-batches", type=int)
    train.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    train.set_defaults(handler=run_train)

    evaluate = subparsers.add_parser("evaluate")
    add_common_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--evaluation-years", type=parse_years, default=(2023,))
    evaluate.add_argument("--evaluation-anchor-stride", type=int, default=12)
    evaluate.add_argument("--evaluation-max-windows", type=int)
    evaluate.add_argument("--evaluation-max-batches", type=int)
    evaluate.add_argument("--validation-batch-size", type=int, default=1)
    evaluate.add_argument("--evaluation-output", type=Path, required=True)
    evaluate.set_defaults(handler=run_evaluate, batch_size=1)

    export = subparsers.add_parser("export")
    add_common_arguments(export)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--export-years", type=parse_years, required=True)
    export.add_argument("--export-anchor-stride", type=int, default=12)
    export.add_argument("--export-max-windows", type=int)
    export.add_argument(
        "--export-anchor-times",
        type=Path,
        help="optional .npy/.npz issue-time axis; export only exact available matches",
    )
    export.add_argument("--export-anchor-times-key", default="anchor_times")
    export.add_argument("--expected-export-anchors", type=int)
    export.add_argument("--export-batch-size", type=int, default=1)
    export.add_argument("--export-dtype", choices=("float16", "float32"), default="float16")
    export.add_argument("--export-output", type=Path, required=True)
    export.add_argument("--maximum-export-gib", type=float, default=10.0)
    export.add_argument("--allow-large-export", action="store_true")
    export.add_argument(
        "--resume-export", action=argparse.BooleanOptionalAction, default=True
    )
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(handler=run_export, batch_size=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.radar_root = args.radar_root.resolve()
    if args.evaluation_mask is None:
        args.evaluation_mask = args.radar_root / "grid_coordinates.npz"
    else:
        args.evaluation_mask = args.evaluation_mask.resolve()
    if args.command == "train":
        if tuple(args.train_years) != (2019, 2020, 2021, 2022):
            raise ValueError("primary fitting years are 2019,2020,2021,2022")
        if tuple(args.validation_years) != (2023,):
            raise ValueError("primary checkpoint-selection years are 2023 only")
        if args.output_frames == 18:
            if args.profile != "paper_long_concat":
                raise ValueError("the long stage requires paper_long_concat")
            if args.initialization_checkpoint is None or not args.freeze_encoder:
                raise ValueError(
                    "long training requires the same-seed short best.pt via "
                    "--initialization-checkpoint and --freeze-encoder"
                )
            if args.encoder_only_initialization:
                raise ValueError("long training initializes all shape-compatible tensors")
        elif args.profile != "memory_safe_add":
            raise ValueError("the short stage requires --profile memory_safe_add")
    if args.command == "evaluate" and tuple(args.evaluation_years) != (2023,):
        raise ValueError("checkpoint evaluation is restricted to 2023")
    args.official_repo = args.official_repo.resolve()
    if getattr(args, "export_anchor_times", None) is not None:
        if not args.export_anchor_times.is_absolute():
            args.export_anchor_times = (args.project_root / args.export_anchor_times).resolve()
        else:
            args.export_anchor_times = args.export_anchor_times.resolve()
    if hasattr(args, "output_root") and not args.output_root.is_absolute():
        args.output_root = (args.project_root / args.output_root).resolve()
    args.handler(args)


if __name__ == "__main__":
    main()
