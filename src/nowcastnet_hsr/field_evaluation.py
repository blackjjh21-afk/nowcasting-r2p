"""Streaming HSR-MP field verification and explicitly bounded input-only export.

No CNN, stations, AWS, or full forecast bank is used here. The selected,
completed generator is the only accepted forecaster. Results occupy separate,
locked directories and existing or interrupted results are never overwritten.
"""
from __future__ import annotations

import csv
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .config import digest, lock, write_json
from .data import HSRDataset, STEP_NS, rate_to_dbz_numpy
from .trainer import load_forecaster, predict_members, predict_product

RATE_LEADS = tuple(range(10, 181, 10))
RN60_LEADS = tuple(range(60, 181, 10))
THRESHOLDS = (1., 5., 10.)


def _arguments(config, split, max_issues, *, prediction):
    if config.get("schema") != "nowcastnet-hsr-field-v5" or config.get("test_issue_policy") != "hsr_available_windows":
        raise ValueError("Field-only commands require the field schema and hsr_available_windows policy")
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation or test")
    if (max_issues is not None and (type(max_issues) is not int or max_issues < 1)) or (prediction and max_issues is None):
        raise ValueError("max_issues must be a positive integer for bounded prediction or a diagnostic subset")


def _field(value, steps, mask, label):
    """Validate every supported cell; unsupported cells are excluded, not scored."""
    value = np.asarray(value)
    if value.shape != (steps, *mask.shape):
        raise ValueError(f"{label} must have shape {(steps, *mask.shape)}")
    supported = value[:, mask]
    if not np.isfinite(supported).all() or np.any(supported < 0):
        raise ValueError(f"{label} contains nonfinite or negative values on the valid spatial mask")
    result = np.array(value, dtype=np.float32, copy=True)
    result[:, ~mask] = 0
    if not np.isfinite(result).all():
        raise ValueError(f"{label} overflows float32 on the valid spatial mask")
    return result


def _rn60(rate):
    """Trailing six 10-minute rates / 6, ending at +60 through +180 min."""
    return (np.lib.stride_tricks.sliding_window_view(rate, 6, axis=0)
            .sum(axis=-1, dtype=np.float64) / 6.).astype(np.float32)


class StreamingScores:
    """Pooled space/time scores with O(number-of-leads) retained state."""
    def __init__(self, leads, units, thresholds=THRESHOLDS):
        self.leads, self.units, self.thresholds = tuple(leads), units, tuple(thresholds)
        shape = (len(self.leads), len(self.thresholds))
        self.hits = np.zeros(shape, dtype=np.int64)
        self.misses = np.zeros(shape, dtype=np.int64)
        self.false_alarms = np.zeros(shape, dtype=np.int64)
        self.pairs = np.zeros(len(self.leads), dtype=np.int64)
        self.absolute_error = np.zeros(len(self.leads), dtype=np.float64)
        self.squared_error = self.absolute_error.copy()
        self.signed_error = self.absolute_error.copy()

    def update(self, prediction, observation, mask):
        pred = _field(prediction, len(self.leads), mask, "prediction")[:, mask].astype(np.float64)
        obs = _field(observation, len(self.leads), mask, "observation")[:, mask].astype(np.float64)
        error = pred - obs
        self.pairs += pred.shape[1]
        self.absolute_error += np.abs(error).sum(axis=1)
        self.squared_error += np.square(error).sum(axis=1)
        self.signed_error += error.sum(axis=1)
        for index, threshold in enumerate(self.thresholds):
            p, o = pred >= threshold, obs >= threshold
            self.hits[:, index] += (p & o).sum(axis=1)
            self.misses[:, index] += (~p & o).sum(axis=1)
            self.false_alarms[:, index] += (p & ~o).sum(axis=1)

    def result(self):
        if np.any(self.pairs == 0):
            raise ValueError("No valid field pairs at one or more leads")
        result = {"leads_minutes": list(self.leads), "units": self.units,
                  "sample_count": self.pairs.tolist(),
                  "mae": (self.absolute_error / self.pairs).tolist(),
                  "rmse": np.sqrt(self.squared_error / self.pairs).tolist(),
                  "mean_bias": (self.signed_error / self.pairs).tolist(),
                  "mean_bias_definition": "mean(prediction - observation)"}
        if self.thresholds:
            denominator = self.hits + self.misses + self.false_alarms
            csi = np.divide(self.hits, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
            frequency_bias = [[float((h + f) / (h + m)) if h + m else None
                               for h, m, f in zip(hits, misses, alarms)]
                              for hits, misses, alarms in zip(self.hits, self.misses, self.false_alarms)]
            result.update(thresholds=list(self.thresholds), hits=self.hits.tolist(), misses=self.misses.tolist(),
                          false_alarms=self.false_alarms.tolist(), csi=csi.tolist(), frequency_bias=frequency_bias,
                          macro_csi=float(csi.mean()),
                          frequency_bias_definition="(hits + false_alarms) / (hits + misses); null if no observed events",
                          zero_event_convention="CSI=0 when hits+misses+false_alarms=0")
        return result


class StreamingEnsembleScores:
    """Score the mean of the model's matched stochastic draws."""
    def __init__(self, member_count, mask, *, spatial_resolution_km=4.):
        if type(member_count) is not int or member_count < 1:
            raise ValueError("member_count must be a positive integer")
        self.mask = np.array(mask, copy=True)
        if self.mask.ndim != 2 or self.mask.dtype != bool or not self.mask.any():
            raise ValueError("Ensemble scoring requires a nonempty boolean mask")
        if not np.isfinite(spatial_resolution_km) or spatial_resolution_km <= 0:
            raise ValueError("spatial_resolution_km must be finite and positive")
        self.member_count = member_count
        self.products = {
            "rain_rate": StreamingScores(RATE_LEADS, "mm h-1"),
            "rn60": StreamingScores(RN60_LEADS, "mm"),
        }

    def update(self, members, observation, mask):
        if not np.array_equal(mask, self.mask):
            raise ValueError("Common spatial mask changed during ensemble scoring")
        members = np.asarray(members)
        if members.shape != (self.member_count, 18, *self.mask.shape):
            raise ValueError("Ensemble must have shape [member,18,y,x] on the common grid")
        members = np.stack([_field(member, 18, self.mask, "forecast member") for member in members])
        observation = _field(observation, 18, self.mask, "observation")
        mean_field = members.mean(axis=0, dtype=np.float32)
        self.products["rain_rate"].update(mean_field, observation, self.mask)
        self.products["rn60"].update(_rn60(mean_field), _rn60(observation), self.mask)
        return mean_field

    def result(self):
        return {"products": {name: score.result() for name, score in self.products.items()}}


def _coordinates(dataset, mask):
    record = dataset.provenance.get("grid", {}).get("hsr_coordinates", {})
    if not record.get("path"):
        return {}
    with np.load(record["path"], allow_pickle=False) as source:
        result = {name: np.array(source[name], copy=True) for name in ("latitude", "longitude")}
    for name, value in result.items():
        if value.shape != mask.shape or not np.isfinite(value[mask]).all():
            raise ValueError(f"Invalid {name} coordinate grid")
    return result


def _preflight(directory, issue_count, example_count, mask, *, evaluation):
    # Compressed NPZs never require a larger estimate than raw fields plus
    # generous container/metadata overhead. No [issue,lead,y,x] bank is allocated.
    planes = 49 * (2 if evaluation else 1) + 2
    estimate = int(example_count * (mask.size * planes * 8 + 1024 ** 2)
                   + issue_count * 8 + mask.nbytes + 64 * 1024 ** 2)
    free = shutil.disk_usage(directory).free
    if free < estimate:
        raise OSError(f"Field export needs approximately {estimate} free bytes; found {free}")
    return {"required_free_bytes": estimate, "available_free_bytes": free,
            "full_forecast_bank": False, "maximum_example_count": example_count}


def _save_example(directory, issue, prediction, predicted_rn60, coordinates, observation=None):
    values = {"issue_time_kst_ns": np.int64(issue), "rate_leads_minutes": np.array(RATE_LEADS, dtype=np.int64),
              "rn60_leads_minutes": np.array(RN60_LEADS, dtype=np.int64),
              "rate_valid_times_kst_ns": issue + np.arange(1, 19, dtype=np.int64) * STEP_NS,
              "rn60_valid_times_kst_ns": issue + np.arange(6, 19, dtype=np.int64) * STEP_NS,
              "predicted_rate_mm_h": prediction, "predicted_rn60_mm": predicted_rn60,
              "predicted_equivalent_dbz": rate_to_dbz_numpy(prediction, normalized=False), **coordinates}
    if observation is not None:
        values.update(observed_rate_mm_h=observation, observed_rn60_mm=_rn60(observation),
                      observed_equivalent_dbz=rate_to_dbz_numpy(observation, normalized=False))
    path = directory / f"example_{issue}_kst_ns.npz"
    # Exclusive creation protects the individual example even inside our lock.
    with path.open("xb") as output:
        np.savez_compressed(output, **values)
    return path.name


def _write_csv(path, products):
    names = ("product", "lead_minutes", "units", "threshold", "sample_count", "hits", "misses", "false_alarms",
             "csi", "frequency_bias", "mae", "rmse", "mean_bias")
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=names)
        writer.writeheader()
        for product, scores in products.items():
            for lead_index, lead in enumerate(scores["leads_minutes"]):
                for threshold_index, threshold in enumerate(scores.get("thresholds", [None])):
                    row = {"product": product, "lead_minutes": lead, "units": scores["units"], "threshold": threshold,
                           **{name: scores[name][lead_index] for name in ("sample_count", "mae", "rmse", "mean_bias")}}
                    if threshold is not None:
                        row.update({name: scores[name][lead_index][threshold_index]
                                    for name in ("hits", "misses", "false_alarms", "csi", "frequency_bias")})
                    writer.writerow(row)


def _run(config, split, device, max_issues, *, evaluation):
    _arguments(config, split, max_issues, prediction=not evaluation)
    kind = "field_evaluation" if evaluation else "field_prediction"
    selection = "full" if max_issues is None else f"diagnostic_first_{max_issues}"
    directory = Path(config["experiment_root"]) / kind / f"{split}_{selection}"
    with lock(directory):
        if any(path.name != "run.lock" for path in directory.iterdir()):
            raise FileExistsError(f"Refusing existing field results: {directory}; use a new experiment_root or subset")
        dataset = HSRDataset(config, split, require_future=evaluation, max_issues=max_issues)
        issues = np.asarray(dataset.issue_times_ns)
        mask = np.asarray(dataset.spatial_mask)
        if (issues.dtype != np.int64 or issues.ndim != 1 or not len(issues) or len(issues) != len(dataset)
                or np.any(np.diff(issues) <= 0) or np.any(issues % STEP_NS)):
            raise ValueError("Dataset must expose ordered, unique 10-minute KST-nanosecond issues")
        if mask.ndim != 2 or mask.dtype != bool or not mask.any():
            raise ValueError("Dataset must expose a nonempty boolean spatial mask")
        example_count = min(3, len(issues)) if evaluation else len(issues)
        space = _preflight(directory, len(issues), example_count, mask, evaluation=evaluation)
        coordinates = _coordinates(dataset, mask)
        model, checkpoint = load_forecaster(config, device=device)
        model.eval()
        evaluation_config = config.get("evaluation", {})
        resolution = evaluation_config.get("spatial_resolution_km", 4.)
        metadata = {"schema": "nowcastnet-hsr-field-results-v5", "command": kind,
                    "output_directory": str(directory.resolve()), "split": split, "device": str(device),
                    "diagnostic_only": max_issues is not None, "max_issues": max_issues,
                    "issue_count": len(issues), "supported_cells": int(mask.sum()), "hsr_data": dataset.provenance,
                    "forecaster": checkpoint, "forecast_product": config["forecast_product"], "space_preflight": space,
                    "reference": "Future observed HSR converted by fixed Marshall-Palmer; not gauge observations" if evaluation else None,
                    "prediction_uses_future_observations": False,
                    "selection": "Deterministic first eligible issues" if max_issues is not None else "All HSR-eligible issues",
                    "examples_selection": "Deterministic first eligible issues; no skill/event-based selection",
                    "timezone": "Asia/Seoul", "timestamp_encoding": "Naive KST wall-clock nanoseconds, not UTC",
                    "rn60_definition": "Sum of six consecutive future 10-minute rates divided by 6; trailing hour ending +60..+180",
                    "equivalent_dbz_definition": "max(10*log10(200*R**1.6),0); R=0 maps to 0; inverse-MP equivalent, not direct radar output",
                    "invalid_spatial_cells": "Excluded from every score; exported rain fields are zero outside spatial_mask.npy",
                    "ensemble_scoring": "Scores of the fixed four-draw mean field",
                    "comparison_requirement": "For exPreCast or other methods use the identical issuance list, HSR-MP targets, grid, cap/units, and mask; score arrays with StreamingEnsembleScores",
                    "evaluation_settings": {"spatial_resolution_km": resolution},
                    "member_indexing": "One-based exported member IDs correspond to zero-based deterministic noise draw IDs"}
        write_json(directory / "contract.json", {"config": config, **metadata})
        write_json(directory / "status.json", {"state": "running", "issue_count": len(issues)})
        started = time.monotonic()
        processed = 0
        try:
            np.save(directory / "issue_times_kst_ns.npy", issues, allow_pickle=False)
            np.save(directory / "spatial_mask.npy", mask, allow_pickle=False)
            ensemble = StreamingEnsembleScores(config["forecast_product"]["ensemble_members"], mask,
                                               spatial_resolution_km=resolution) if evaluation else None
            dbz_scores = StreamingScores(RATE_LEADS, "dBZ", thresholds=()) if evaluation else None
            examples = []
            for index, stamp in enumerate(issues):
                issue = int(stamp)
                sample = dataset[index]
                if int(sample["issue_ns"]) != issue or not np.array_equal(sample["mask"], mask):
                    raise ValueError("Dataset issuance/mask changed during streaming")
                history = _field(sample["history"], 7, mask, "history")
                with torch.inference_mode():
                    predictor = predict_members if evaluation else predict_product
                    forecast = predictor(model, torch.as_tensor(history[None], device=device), [issue], config)
                forecast = forecast.detach().cpu().numpy() if torch.is_tensor(forecast) else np.asarray(forecast)
                observation = _field(sample["target"], 18, mask, "target") if evaluation else None
                if evaluation:
                    if forecast.shape != (config["forecast_product"]["ensemble_members"], 1, 18, *mask.shape):
                        raise ValueError("NowcastNet members must be [member,1,18,y,x]")
                    prediction = ensemble.update(forecast[:, 0], observation, mask)
                    dbz_scores.update(rate_to_dbz_numpy(prediction, normalized=False),
                                      rate_to_dbz_numpy(observation, normalized=False), mask)
                else:
                    if forecast.shape != (1, 18, *mask.shape):
                        raise ValueError("NowcastNet product must be [1,18,y,x]")
                    prediction = _field(forecast[0], 18, mask, "forecast")
                predicted_rn60 = _rn60(prediction)
                if index < example_count:
                    examples.append(_save_example(directory, issue, prediction, predicted_rn60, coordinates, observation))
                processed += 1
                if index == 0 or processed % 100 == 0 or processed == len(issues):
                    print(f"{kind} {split}: {processed}/{len(issues)} diagnostic={max_issues is not None}", flush=True)
            metadata.update(seconds=time.monotonic() - started, example_files=examples)
            if evaluation:
                metadata.update(ensemble.result())
                metadata["products"]["equivalent_dbz"] = dbz_scores.result()
                _write_csv(directory / "scores.csv", metadata["products"])
            metadata["files"] = {path.name: {"size_bytes": path.stat().st_size, "sha256": digest(path)}
                                 for path in sorted(directory.iterdir()) if path.name not in ("run.lock", "status.json")}
            result_name = "metrics.json" if evaluation else "metadata.json"
            write_json(directory / result_name, metadata)
            write_json(directory / "status.json", {"state": "completed", "issue_count": processed,
                                                   "result_file": result_name, "result_sha256": digest(directory / result_name)})
        except BaseException as error:
            write_json(directory / "status.json", {"state": "failed", "processed_issues": processed,
                                                   "error": f"{type(error).__name__}: {error}"})
            raise
    return metadata


def evaluate(config, split="test", device="cpu", max_issues=None):
    """Stream field metrics against observed future HSR; save at most 3 examples.

    Any explicit max_issues produces a separately named diagnostic result and is
    never presented as full-split verification. No checkpoint is fitted/selected.
    """
    return _run(config, split, device, max_issues, evaluation=True)


def predict(config, split="test", device="cpu", max_issues=3):
    """Export only a positive, explicitly bounded number of input-only forecasts.

    require_future=False is mandatory: neither future pixel values nor future
    availability may select these issues. This command does not report scores.
    """
    return _run(config, split, device, max_issues, evaluation=False)
