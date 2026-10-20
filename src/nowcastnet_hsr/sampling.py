"""Train-only rain-importance sampling weights, with sequential frame caches.

The original NowcastNet uses g(R)=1-exp(-R) for training acceptance weights.
Here one fixed 4-km domain replaces the original spatial crop hierarchy. Each
eligible 25-frame window receives sum(g(R)) over supported cells plus epsilon.
Validation/test windows are never inspected by this module. The trainer uses
these weights with replacement and a fixed number of draws per epoch.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .config import digest, identity, lock
from .data import STEP_NS

SCHEMA = "nowcastnet-training-rain-importance-v1"


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _stat(path):
    path = Path(path)
    value = path.stat()
    return {"path": str(path.resolve()), "bytes": value.st_size,
            "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def _month_contract(dataset, block):
    """Bind weights to source identities without hashing multi-GB radar again."""
    month = dataset._months[block]
    if month["backend"] == "monthly_radar_memmap":
        current_source = {name: _stat(month["folder"] / name)
                          for name in ("radar.npy", "radar_valid.npy", "times_kst_ns.npy",
                                       "spatial_mask.npy", "complete.json")}
    elif month["backend"] == "official_tiff":
        # TIFF inventories otherwise identify timestamps, not modified values.
        records = [(int(stamp), _stat(path)) for stamp, path in sorted(month["paths"].items())]
        current_source = {"tiff_stats_sha256": identity(records), "frame_count": len(records)}
    else:
        raise ValueError("Unsupported HSR source for importance sampling")
    return {"schema": SCHEMA, "month": dataset.months[block],
            "source": month["source"], "source_file_stats": current_source,
            "grid": dataset.provenance["grid"],
            "times_sha256": _array_hash(month["times"]),
            "available_sha256": _array_hash(month["available"]),
            "support_sha256": _array_hash(dataset.spatial_mask),
            "shape": list(dataset.shape), "rain_rate_cap_mm_h": dataset.rain_rate_cap,
            "conversion": dataset.provenance["conversion"],
            "formula": "sum supported pixels of -expm1(-R_mm_h); unavailable frames NaN",
            "split": "train", "large_radar_payload_hash_recomputed": False}


def _validate_scores(scores, available):
    if scores.shape != available.shape or scores.dtype != np.float64:
        raise ValueError("Importance-cache score shape or dtype changed")
    if not np.isfinite(scores[available]).all() or np.any(scores[available] < 0):
        raise ValueError("Importance-cache available scores must be finite and nonnegative")
    if not np.isnan(scores[~available]).all():
        raise ValueError("Importance-cache missing frames must remain marked unavailable")


def _frame_scores(dataset, block, directory):
    contract = _month_contract(dataset, block)
    key = identity(contract)
    path = directory / f"{dataset.months[block]}_{key}.npz"
    month = dataset._months[block]
    available = np.asarray(month["available"], dtype=bool)
    reused = path.is_file()
    print(f"Importance weights {dataset.months[block]}: {'reuse cache' if reused else 'scan training frames'} "
          f"({int(available.sum())} available)", flush=True)
    if reused:
        with np.load(path, allow_pickle=False) as cached:
            if json.loads(str(cached["contract"].item())) != contract:
                raise ValueError(f"Importance-cache source contract changed: {path}")
            if not np.array_equal(cached["times"], month["times"]):
                raise ValueError("Importance-cache timestamps changed")
            scores = np.array(cached["scores"], dtype=np.float64, copy=True)
            if str(cached["scores_sha256"].item()) != _array_hash(scores):
                raise ValueError("Importance-cache frame score checksum changed")
    else:
        scores = np.full(len(available), np.nan, dtype=np.float64)
        offset = int(dataset._starts[block])
        for local_index in np.flatnonzero(available):
            # One chronological pass per training month, not 25 reads per
            # overlapping issue. HSRDataset handles read-only memmap/TIFF IO.
            rain = np.asarray(dataset._read_frame(offset + int(local_index)), dtype=np.float64)
            selected = rain[dataset.spatial_mask]
            if not np.isfinite(selected).all() or np.any(selected < 0):
                raise ValueError("Sampling weights require valid nonnegative training rain rates")
            scores[local_index] = np.sum(-np.expm1(-selected), dtype=np.float64)
        _validate_scores(scores, available)
        # Refuse to publish weights if source files changed while being read.
        if _month_contract(dataset, block) != contract:
            raise RuntimeError("HSR source changed during importance-cache preparation")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, scores=scores, times=month["times"],
                         scores_sha256=_array_hash(scores),
                         contract=json.dumps(contract, sort_keys=True, separators=(",", ":")))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    _validate_scores(scores, available)
    return scores, {"month": dataset.months[block], "path": str(path),
                    "sha256": digest(path), "contract_sha256": key,
                    "available_frames": int(available.sum())}


def build_training_weights(dataset, config):
    """Return (CPU float64 weights in dataset order, JSON-safe provenance).

    Only a future-labelled training HSRDataset is accepted. The sampler must use
    replacement=True and a fixed documented draw count for each epoch. The
    trainer may round up len(dataset) to a whole number of physical batches.
The cache stores per-frame scores, so changing epsilon, issuance subsets, or
epoch seeds does not require reading the images again.
"""
    if dataset.split != "train" or not dataset.require_future:
        raise ValueError("Importance sampling accepts only training windows with future targets")
    if set(dataset.provenance["years"]) - set(config["train_years"]):
        raise ValueError("Importance sampling may not read nontraining years")
    options = config["field"]
    enabled = options.get("importance_sampling", True)
    if type(enabled) is not bool:
        raise ValueError("field.importance_sampling must be boolean")
    epsilon = options.get("sampling_epsilon", 1.0)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("field.sampling_epsilon must be finite and positive")
    if not len(dataset):
        raise ValueError("Cannot sample an empty training dataset")
    provenance = {"schema": SCHEMA, "split": "train", "enabled": enabled,
                  "eligible_window_count": len(dataset), "replacement": True,
                  "epsilon": float(epsilon), "units": "R in mm h-1",
                  "rain_rate_cap_mm_h": dataset.rain_rate_cap,
                  "issue_times_sha256": _array_hash(dataset.issue_times_ns),
                  "formula": "sum over 25 frames and supported cells of (1-exp(-R)) + epsilon",
                  "adaptation": "fixed domain; no original spatial-crop sampling hierarchy",
                  "validation_test": "not read, sampled or reweighted"}
    if not enabled:
        provenance.update(formula="uniform weights", months=[])
        return torch.ones(len(dataset), dtype=torch.float64), provenance
    destination = Path(config["cache_root"]).expanduser()
    if not destination.is_absolute():
        destination = Path(config["project_root"]) / destination
    destination = destination.resolve() / "importance_sampling_frames"
    # Keep generated weights outside all read-only source locations, even if
    # this helper is called directly without the production config validator.
    for source in (dataset.radar_root, dataset.cache_root):
        source = source.resolve()
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("Importance-cache destination overlaps protected HSR source")
    with lock(destination):
        blocks, records = [], []
        for block in range(len(dataset._months)):
            values, record = _frame_scores(dataset, block, destination)
            blocks.append(values)
            records.append(record)
    frame_scores = np.concatenate(blocks)
    indices = np.asarray(dataset.indices, dtype=np.int64)
    starts, ends = indices - 6, indices + 19
    if np.any(starts < 0) or np.any(ends > len(frame_scores)):
        raise ValueError("Training sampling window extends beyond source frames")
    available = np.isfinite(frame_scores)
    count_prefix = np.r_[0, np.cumsum(available, dtype=np.int64)]
    if np.any(count_prefix[ends] - count_prefix[starts] != 25):
        raise ValueError("Training sampling windows include unavailable frames")
    # Endpoint span plus strictly ordered exact-slot source timestamps detects
    # season/year gaps while allowing a contiguous June/July window.
    if np.any(dataset.times[ends - 1] - dataset.times[starts] != 24 * STEP_NS):
        raise ValueError("Training sampling windows cross temporal gaps")
    prefix = np.r_[0.0, np.cumsum(np.nan_to_num(frame_scores, nan=0.0), dtype=np.float64)]
    weights = prefix[ends] - prefix[starts] + float(epsilon)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Training sampling weights must be finite and positive")
    provenance.update(months=records, weights_sha256=_array_hash(weights),
                      weight_min=float(weights.min()), weight_max=float(weights.max()),
                      weight_sum=float(weights.sum()), available_frame_count=int(available.sum()))
    return torch.from_numpy(weights.copy()).to(dtype=torch.float64, device="cpu"), provenance
