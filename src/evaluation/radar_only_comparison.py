#!/usr/bin/env python3
"""Audit and compare standard versus radar-only 4-km/10-min Direct R2P.

This is a same-checkpoint inference-input sensitivity analysis.  The standard
route masks the 128 held-out target histories while retaining gauge context
from the 514 fitting stations.  Radar-only inference uses the same three
selected checkpoints and additionally masks every fitting-station gauge
history.  No model is retrained.

The script validates the array, axis, truth, metadata and checkpoint contracts,
then computes CSI and paired issuance-date block-bootstrap intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


LEADS = np.asarray([60, 90, 120, 150, 180], dtype=np.int16)
THRESHOLDS = np.asarray([1.0, 5.0, 10.0], dtype=np.float64)
STANDARD_STORE = "target_masked_heldout128_predictions"
RADAR_ONLY_STORE = "target_masked_heldout128_predictions_radar_only"
EXPECTED_SHAPE = (35_088, 128, 36)
EXPECTED_CONTRACT = "vanilla_heldout_r2p_official4km10min_input_only_v1"


class RadarOnlyAuditError(RuntimeError):
    """Raised when the matched radar-only contract is violated."""


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RadarOnlyAuditError(f"expected JSON object: {path}")
    return value


def required_store_files(store: Path) -> tuple[Path, ...]:
    return tuple(
        store / name
        for name in (
            "COMPLETED",
            "metadata.json",
            "preds_mm.npy",
            "trues_mm.npy",
            "anchor_times_ns.npy",
            "station_ids.npy",
            "lead_minutes.npy",
        )
    )


def load_axes(store: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.load(store / name, allow_pickle=False)
        for name in ("anchor_times_ns.npy", "station_ids.npy", "lead_minutes.npy")
    )


def assert_array_equal(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if not np.array_equal(left, right, equal_nan=True):
        raise RadarOnlyAuditError(f"array mismatch: {label}")


def roundtrip_f16(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(values, dtype=np.float16), dtype=np.float32)


def date_axes(anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dates = anchors.astype("datetime64[ns]").astype("datetime64[D]")
    unique, codes = np.unique(dates, return_inverse=True)
    if len(unique) != 244:
        raise RadarOnlyAuditError(f"expected 244 issuance dates, found {len(unique)}")
    return unique, codes.astype(np.int16)


def bootstrap_weights(n_dates: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_dates, np.full(n_dates, 1.0 / n_dates), size=replicates
    ).astype(np.int16)
    if not np.all(weights.sum(axis=1) == n_dates):
        raise RadarOnlyAuditError("invalid date-bootstrap weights")
    return weights


def date_counts(
    prediction: np.ndarray,
    truth: np.ndarray,
    finite: np.ndarray,
    date_index: np.ndarray,
    threshold: float,
    n_dates: int,
) -> np.ndarray:
    observed = truth >= threshold
    forecast = prediction >= threshold
    hit = np.sum(finite & observed & forecast, axis=1, dtype=np.int64)
    miss = np.sum(finite & observed & ~forecast, axis=1, dtype=np.int64)
    false_alarm = np.sum(finite & ~observed & forecast, axis=1, dtype=np.int64)
    correct_negative = np.sum(finite & ~observed & ~forecast, axis=1, dtype=np.int64)
    return np.stack(
        [
            np.bincount(date_index, weights=value, minlength=n_dates).astype(np.int64)
            for value in (hit, miss, false_alarm, correct_negative)
        ],
        axis=1,
    )


def csi(counts: np.ndarray) -> np.ndarray:
    denominator = counts[..., 0] + counts[..., 1] + counts[..., 2]
    return np.divide(
        counts[..., 0],
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def validate_contract(
    root: Path,
    hash_sources: bool,
    standard_store_name: str = STANDARD_STORE,
    radar_only_store_name: str = RADAR_ONLY_STORE,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    audit: dict[str, Any] = {"seeds": {}, "source_hashes": {}}
    reference_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    reference_truth_hash: str | None = None
    expected_leads = np.arange(5, 181, 5, dtype=np.int16)

    for seed in range(3):
        seed_root = root / f"seed_{seed}"
        standard = seed_root / standard_store_name
        radar_only = seed_root / radar_only_store_name
        missing = [
            str(path)
            for store in (standard, radar_only)
            for path in required_store_files(store)
            if not path.is_file()
        ]
        if missing:
            raise RadarOnlyAuditError("missing matched inputs:\n" + "\n".join(missing))

        standard_meta = read_json(standard / "metadata.json")
        radar_meta = read_json(radar_only / "metadata.json")
        declared_standard_hash = str(standard_meta.get("checkpoint_sha256", ""))
        declared_radar_hash = str(radar_meta.get("checkpoint_sha256", ""))
        if not declared_standard_hash or declared_standard_hash != declared_radar_hash:
            raise RadarOnlyAuditError(f"seed {seed}: checkpoint hash mismatch")
        candidate_names = {
            str(value)
            for value in (
                standard_meta.get("checkpoint_name"),
                radar_meta.get("checkpoint_name"),
                standard_meta.get("staged_checkpoint_name"),
                radar_meta.get("staged_checkpoint_name"),
                "best.pt",
            )
            if value
        }
        candidate_paths = [
            seed_root / name
            for name in candidate_names
            if Path(name).name == name and (seed_root / name).is_file()
        ]
        checkpoint = next(
            (
                path
                for path in candidate_paths
                if sha256_file(path) == declared_standard_hash
            ),
            None,
        )
        if checkpoint is None:
            raise RadarOnlyAuditError(
                f"seed {seed}: no local checkpoint matches the declared hash"
            )
        actual_checkpoint_hash = declared_standard_hash
        for label, metadata in (("standard", standard_meta), ("radar_only", radar_meta)):
            if metadata.get("contract") != EXPECTED_CONTRACT:
                raise RadarOnlyAuditError(f"seed {seed} {label}: wrong scientific contract")
            if int(metadata.get("seed", -1)) != seed:
                raise RadarOnlyAuditError(f"seed {seed} {label}: metadata seed mismatch")
            if metadata.get("completed") is not True:
                raise RadarOnlyAuditError(f"seed {seed} {label}: incomplete metadata")
        if standard_meta.get("mode") != "target_masked":
            raise RadarOnlyAuditError(f"seed {seed}: standard mode is not target_masked")
        if radar_meta.get("mode") != "radar_only_all_context_gauges_masked":
            raise RadarOnlyAuditError(f"seed {seed}: wrong radar-only mode")
        if radar_meta.get("all_context_gauges_masked") is not True:
            raise RadarOnlyAuditError(f"seed {seed}: gauge-mask flag is not true")
        if not str(radar_meta.get("evaluation_tag", "")).startswith("radar_only"):
            raise RadarOnlyAuditError(f"seed {seed}: wrong radar-only evaluation tag")

        standard_axes = load_axes(standard)
        radar_axes = load_axes(radar_only)
        for name, left, right in zip(
            ("anchors", "stations", "leads"), standard_axes, radar_axes, strict=True
        ):
            assert_array_equal(left, right, f"seed {seed} standard/radar-only {name}")
        if reference_axes is None:
            reference_axes = standard_axes
        else:
            for name, left, right in zip(
                ("anchors", "stations", "leads"), reference_axes, standard_axes, strict=True
            ):
                assert_array_equal(left, right, f"seed {seed} cross-seed {name}")
        assert reference_axes is not None
        if not np.array_equal(standard_axes[2].astype(np.int16), expected_leads):
            raise RadarOnlyAuditError(f"seed {seed}: unexpected lead axis")

        array_contract: dict[str, Any] = {}
        for mode, store in (("standard", standard), ("radar_only", radar_only)):
            prediction = np.load(store / "preds_mm.npy", mmap_mode="r", allow_pickle=False)
            truth = np.load(store / "trues_mm.npy", mmap_mode="r", allow_pickle=False)
            if prediction.shape != EXPECTED_SHAPE or truth.shape != EXPECTED_SHAPE:
                raise RadarOnlyAuditError(
                    f"seed {seed} {mode}: shape mismatch {prediction.shape}/{truth.shape}"
                )
            if prediction.dtype != np.float32 or truth.dtype != np.float32:
                raise RadarOnlyAuditError(
                    f"seed {seed} {mode}: dtype mismatch {prediction.dtype}/{truth.dtype}"
                )
            array_contract[mode] = {
                "prediction_shape": list(prediction.shape),
                "prediction_dtype": str(prediction.dtype),
                "truth_shape": list(truth.shape),
                "truth_dtype": str(truth.dtype),
            }

        standard_truth_hash = sha256_file(standard / "trues_mm.npy")
        radar_truth_hash = sha256_file(radar_only / "trues_mm.npy")
        if standard_truth_hash != radar_truth_hash:
            raise RadarOnlyAuditError(f"seed {seed}: standard/radar-only truth file differs")
        if reference_truth_hash is None:
            reference_truth_hash = standard_truth_hash
        elif standard_truth_hash != reference_truth_hash:
            raise RadarOnlyAuditError(f"seed {seed}: truth differs across seeds")

        source_paths: Iterable[Path] = (
            checkpoint,
            standard / "metadata.json",
            radar_only / "metadata.json",
            standard / "trues_mm.npy",
            radar_only / "trues_mm.npy",
            standard / "preds_mm.npy",
            radar_only / "preds_mm.npy",
        )
        if hash_sources:
            for path in source_paths:
                audit["source_hashes"][str(path.relative_to(root))] = sha256_file(path)
        audit["seeds"][str(seed)] = {
            "checkpoint_sha256": actual_checkpoint_hash,
            "standard_mode": standard_meta.get("mode"),
            "radar_only_mode": radar_meta.get("mode"),
            "truth_sha256": standard_truth_hash,
            "arrays": array_contract,
        }

    assert reference_axes is not None
    anchors, stations, leads = reference_axes
    if (len(anchors), len(stations), len(leads)) != EXPECTED_SHAPE:
        raise RadarOnlyAuditError("reference axes do not match the frozen 35,088×128×36 contract")
    dates, _ = date_axes(anchors)
    audit["common_axes"] = {
        "issue_count": len(anchors),
        "station_count": len(stations),
        "lead_count": len(leads),
        "issuance_date_count": len(dates),
        "anchor_start": str(anchors.astype("datetime64[ns]")[0]),
        "anchor_end": str(anchors.astype("datetime64[ns]")[-1]),
        "station_id_min": int(np.min(stations)),
        "station_id_max": int(np.max(stations)),
    }
    audit["truth_exact_equal_across_modes_and_seeds"] = True
    audit["checkpoint_exact_match_standard_radar_only_and_local_checkpoint_file"] = True
    return audit, reference_axes


def compute_csi_tables(
    root: Path,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    bootstrap_reps: int,
    bootstrap_seed: int,
    standard_store_name: str = STANDARD_STORE,
    radar_only_store_name: str = RADAR_ONLY_STORE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    anchors, _, lead_axis = axes
    dates, date_index = date_axes(anchors)
    weights = bootstrap_weights(len(dates), bootstrap_reps, bootstrap_seed)
    weights_hash = hashlib.sha256(weights.tobytes(order="C")).hexdigest()
    standard_stores = [
        root / f"seed_{seed}" / standard_store_name for seed in range(3)
    ]
    radar_stores = [
        root / f"seed_{seed}" / radar_only_store_name for seed in range(3)
    ]
    truth_memmap = np.load(standard_stores[0] / "trues_mm.npy", mmap_mode="r", allow_pickle=False)
    standard_memmaps = [
        np.load(store / "preds_mm.npy", mmap_mode="r", allow_pickle=False)
        for store in standard_stores
    ]
    radar_memmaps = [
        np.load(store / "preds_mm.npy", mmap_mode="r", allow_pickle=False)
        for store in radar_stores
    ]

    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for lead in LEADS.astype(int):
        matches = np.flatnonzero(lead_axis.astype(int) == lead)
        if len(matches) != 1:
            raise RadarOnlyAuditError(f"lead {lead} is not unique")
        lead_index = int(matches[0])
        truth = np.asarray(truth_memmap[:, :, lead_index], dtype=np.float32)
        standard_predictions = [
            roundtrip_f16(np.asarray(values[:, :, lead_index])) for values in standard_memmaps
        ]
        radar_predictions = [
            roundtrip_f16(np.asarray(values[:, :, lead_index])) for values in radar_memmaps
        ]
        finite = np.isfinite(truth)
        for prediction in standard_predictions + radar_predictions:
            finite &= np.isfinite(prediction)
        for threshold in THRESHOLDS:
            standard_counts = np.stack(
                [
                    date_counts(
                        prediction, truth, finite, date_index, float(threshold), len(dates)
                    )
                    for prediction in standard_predictions
                ]
            )
            radar_counts = np.stack(
                [
                    date_counts(
                        prediction, truth, finite, date_index, float(threshold), len(dates)
                    )
                    for prediction in radar_predictions
                ]
            )
            standard_seed_csi = csi(standard_counts.sum(axis=1))
            radar_seed_csi = csi(radar_counts.sum(axis=1))
            standard_boot = np.einsum("bd,rdc->brc", weights, standard_counts, optimize=True)
            radar_boot = np.einsum("bd,rdc->brc", weights, radar_counts, optimize=True)
            delta_boot = np.nanmean(csi(radar_boot), axis=1) - np.nanmean(
                csi(standard_boot), axis=1
            )
            standard_mean = float(np.nanmean(standard_seed_csi))
            radar_mean = float(np.nanmean(radar_seed_csi))
            delta = radar_mean - standard_mean
            low, high = np.nanpercentile(delta_boot, [2.5, 97.5])
            rows.append(
                {
                    "lead_min": lead,
                    "threshold_mm": float(threshold),
                    "standard_csi_mean": standard_mean,
                    "standard_csi_sd": float(np.nanstd(standard_seed_csi, ddof=1)),
                    "radar_only_csi_mean": radar_mean,
                    "radar_only_csi_sd": float(np.nanstd(radar_seed_csi, ddof=1)),
                    "delta_csi_radar_only_minus_standard": delta,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                    "n_finite_common": int(finite.sum()),
                    "bootstrap_valid_reps": int(np.isfinite(delta_boot).sum()),
                    "bootstrap_reps": bootstrap_reps,
                }
            )
            for seed in range(3):
                seed_rows.append(
                    {
                        "seed": seed,
                        "lead_min": lead,
                        "threshold_mm": float(threshold),
                        "standard_csi": float(standard_seed_csi[seed]),
                        "radar_only_csi": float(radar_seed_csi[seed]),
                        "delta_csi_radar_only_minus_standard": float(
                            radar_seed_csi[seed] - standard_seed_csi[seed]
                        ),
                    }
                )
                for date_index_value, date in enumerate(dates):
                    standard_row = standard_counts[seed, date_index_value]
                    radar_row = radar_counts[seed, date_index_value]
                    daily_rows.append(
                        {
                            "date": str(date),
                            "seed": seed,
                            "lead_min": lead,
                            "threshold_mm": float(threshold),
                            "standard_hit": int(standard_row[0]),
                            "standard_miss": int(standard_row[1]),
                            "standard_false_alarm": int(standard_row[2]),
                            "standard_correct_negative": int(standard_row[3]),
                            "radar_only_hit": int(radar_row[0]),
                            "radar_only_miss": int(radar_row[1]),
                            "radar_only_false_alarm": int(radar_row[2]),
                            "radar_only_correct_negative": int(radar_row[3]),
                        }
                    )
        del truth, standard_predictions, radar_predictions, finite
    return pd.DataFrame(rows), pd.DataFrame(seed_rows), pd.DataFrame(daily_rows), weights_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r2p-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--standard-store", default=STANDARD_STORE)
    parser.add_argument("--radar-only-store", default=RADAR_ONLY_STORE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--bootstrap-reps", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument(
        "--skip-source-hashes",
        action="store_true",
        help="Skip expensive SHA-256 hashes of the full prediction tensors.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.bootstrap_reps < 1_000:
        raise RadarOnlyAuditError("at least 1,000 bootstrap replicates are required")
    for name in (args.standard_store, args.radar_only_store):
        if Path(name).name != name or name in {".", ".."}:
            raise RadarOnlyAuditError("store names must be single directory names")
    audit, axes = validate_contract(
        args.r2p_root,
        not args.skip_source_hashes,
        args.standard_store,
        args.radar_only_store,
    )
    comparison, by_seed, daily, weights_hash = compute_csi_tables(
        args.r2p_root,
        axes,
        args.bootstrap_reps,
        args.bootstrap_seed,
        args.standard_store,
        args.radar_only_store,
    )
    outputs = {
        "csi_by_lead_threshold.csv": comparison,
        "csi_by_seed.csv": by_seed,
        "daily_categorical_counts.csv": daily,
    }
    for name, frame in outputs.items():
        atomic_csv(args.output_dir / name, frame)

    output_paths = [args.output_dir / name for name in outputs]
    audit_manifest = {
        "schema": "r2p-radar-only-4km10min-2024-2025-v1",
        "status": "complete",
        "scientific_contract": {
            "estimand": "CSI(radar-only inference) - CSI(standard Direct R2P)",
            "training_change": "none; same selected checkpoint within each seed",
            "input_change": "all fitting-station RN60/RN15 histories additionally masked",
            "issues": 35_088,
            "heldout_stations": 128,
            "dates": 244,
            "leads_min": LEADS.astype(int).tolist(),
            "thresholds_mm": THRESHOLDS.tolist(),
            "members": 3,
            "prediction_numeric_contract": "float16 round-trip before thresholding",
            "bootstrap_unit": "issuance date",
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_weights_sha256": weights_hash,
            "intervals": "pointwise percentile 95%; not multiplicity adjusted",
            "standard_store": args.standard_store,
            "radar_only_store": args.radar_only_store,
        },
        "input_audit": audit,
        "summary": {
            "maximum_absolute_delta_csi": float(
                comparison.delta_csi_radar_only_minus_standard.abs().max()
            ),
            "cells_with_pointwise_ci_excluding_zero": int(
                comparison.ci_excludes_zero.sum()
            ),
            "cell_count": int(len(comparison)),
        },
        "outputs": {
            f"metrics/{path.name}": {"sha256": sha256_file(path)}
            for path in output_paths
        },
        "script": {
            "path": "src/evaluation/radar_only_comparison.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    manifest_path = args.output_dir / "audit_manifest.json"
    atomic_json(manifest_path, audit_manifest)

    print(f"validated exact matched contract: {EXPECTED_SHAPE}")
    print(f"saved: {args.output_dir / 'csi_by_lead_threshold.csv'}")
    print(f"saved: {manifest_path}")


if __name__ == "__main__":
    main()
