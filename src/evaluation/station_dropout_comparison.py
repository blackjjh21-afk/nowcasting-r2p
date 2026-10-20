#!/usr/bin/env python3
"""Compare Direct R2P with its separately trained no-station-dropout control.

Both routes must contain three complete held-out evaluation stores on exactly
the same issue-time, station, lead and truth axes.  The scientific contracts
must be identical except for the station-dropout configuration.  The script
computes CSI for the three-member route means, estimates paired pointwise 95%
intervals by resampling whole issuance dates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SEEDS = (0, 1, 2)
LEADS = np.asarray([60, 90, 120, 150, 180], dtype=np.int16)
THRESHOLDS = np.asarray([1.0, 5.0, 10.0], dtype=np.float64)
EXPECTED_CONTRACT = "vanilla_heldout_r2p_official4km10min_input_only_v1"
EXPECTED_LEAD_AXIS = np.arange(5, 181, 5, dtype=np.int16)


class StationDropoutAuditError(RuntimeError):
    """Raised when the matched training-time ablation contract is violated."""


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
        raise StationDropoutAuditError(f"expected a JSON object: {path}")
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
        raise StationDropoutAuditError(f"array mismatch: {label}")


def roundtrip_f16(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(values, dtype=np.float16), dtype=np.float32)


def date_axes(
    anchors: np.ndarray, expected_date_count: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    dates = anchors.astype("datetime64[ns]").astype("datetime64[D]")
    unique, codes = np.unique(dates, return_inverse=True)
    if expected_date_count is not None and len(unique) != expected_date_count:
        raise StationDropoutAuditError(
            f"expected {expected_date_count} issuance dates, found {len(unique)}"
        )
    return unique, codes.astype(np.int32)


def bootstrap_weights(n_dates: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_dates, np.full(n_dates, 1.0 / n_dates), size=replicates
    ).astype(np.int16)
    if not np.all(weights.sum(axis=1) == n_dates):
        raise StationDropoutAuditError("invalid issuance-date bootstrap weights")
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


def _dropout_configuration(contract: dict[str, Any]) -> dict[str, Any]:
    try:
        value = contract["optimization"]["station_dropout"]
    except (KeyError, TypeError) as error:
        raise StationDropoutAuditError(
            "scientific_contract.json lacks optimization.station_dropout"
        ) from error
    if not isinstance(value, dict):
        raise StationDropoutAuditError("station-dropout configuration must be an object")
    return value


def validate_scientific_contracts(
    direct_root: Path, no_dropout_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    direct = read_json(direct_root / "scientific_contract.json")
    no_dropout = read_json(no_dropout_root / "scientific_contract.json")
    if direct.get("contract") != EXPECTED_CONTRACT:
        raise StationDropoutAuditError("Direct R2P has the wrong scientific contract")
    if no_dropout.get("contract") != EXPECTED_CONTRACT:
        raise StationDropoutAuditError(
            "no-station-dropout control has the wrong scientific contract"
        )

    direct_dropout = _dropout_configuration(direct)
    no_dropout_config = _dropout_configuration(no_dropout)
    direct_range = np.asarray(direct_dropout.get("p_range", []), dtype=float)
    no_dropout_range = np.asarray(no_dropout_config.get("p_range", []), dtype=float)
    if direct_range.shape != (2,) or not np.allclose(direct_range, [0.0, 0.9]):
        raise StationDropoutAuditError(
            "Direct R2P contract does not declare p_range=[0, 0.9]"
        )
    if not np.isclose(float(direct_dropout.get("full_probability", np.nan)), 0.05):
        raise StationDropoutAuditError(
            "Direct R2P contract does not declare full_probability=0.05"
        )
    if no_dropout_range.shape != (2,) or np.any(no_dropout_range != 0):
        raise StationDropoutAuditError(
            "control contract does not disable the station-mask range"
        )
    if float(no_dropout_config.get("full_probability", np.nan)) != 0.0:
        raise StationDropoutAuditError(
            "control contract does not disable full station masking"
        )

    direct_matched = copy.deepcopy(direct)
    no_dropout_matched = copy.deepcopy(no_dropout)
    direct_matched["optimization"].pop("station_dropout")
    no_dropout_matched["optimization"].pop("station_dropout")
    if direct_matched != no_dropout_matched:
        raise StationDropoutAuditError(
            "route scientific contracts differ beyond the station-dropout setting"
        )
    return direct, no_dropout


def _find_matching_checkpoint(seed_root: Path, metadata: dict[str, Any]) -> Path:
    declared_hash = str(metadata.get("checkpoint_sha256", ""))
    if not declared_hash:
        raise StationDropoutAuditError(
            f"missing checkpoint_sha256 in {seed_root.name} metadata"
        )
    selected_epoch = metadata.get("selected_epoch")
    candidates = {
        str(value)
        for value in (
            metadata.get("checkpoint_name"),
            metadata.get("staged_checkpoint_name"),
            f"epoch_{int(selected_epoch):03d}.pt" if selected_epoch is not None else None,
            "best.pt",
        )
        if value
    }
    paths = [
        seed_root / name
        for name in candidates
        if Path(name).name == name and (seed_root / name).is_file()
    ]
    checkpoint = next((path for path in paths if sha256_file(path) == declared_hash), None)
    if checkpoint is None:
        raise StationDropoutAuditError(
            f"{seed_root.name}: no local checkpoint matches metadata checkpoint_sha256"
        )
    return checkpoint


def _check_complete_seed_stores(root: Path, store_name: str, route_label: str) -> None:
    missing: list[str] = []
    for seed in SEEDS:
        store = root / f"seed_{seed}" / store_name
        absent = [path.name for path in required_store_files(store) if not path.is_file()]
        if absent:
            missing.append(
                f"{route_label} seed {seed}: {store_name}/" + ", ".join(absent)
            )
    if missing:
        detail = "\n".join(missing)
        raise StationDropoutAuditError(
            "The comparison requires complete seeds 0, 1 and 2 for both routes. "
            "Missing inputs:\n" + detail
        )


def validate_contract(
    direct_root: Path,
    no_dropout_root: Path,
    direct_store_name: str,
    no_dropout_store_name: str,
    *,
    expected_issue_count: int | None = 35_088,
    expected_station_count: int | None = 128,
    expected_date_count: int | None = 244,
    hash_prediction_sources: bool = True,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    validate_scientific_contracts(direct_root, no_dropout_root)
    _check_complete_seed_stores(direct_root, direct_store_name, "Direct R2P")
    _check_complete_seed_stores(
        no_dropout_root, no_dropout_store_name, "no-station-dropout"
    )

    audit: dict[str, Any] = {
        "seeds": {},
        "source_hashes": {
            "station_dropout/scientific_contract.json": sha256_file(
                direct_root / "scientific_contract.json"
            ),
            "no_station_dropout/scientific_contract.json": sha256_file(
                no_dropout_root / "scientific_contract.json"
            ),
        },
        "scientific_contracts_matched_except_station_dropout": True,
    }
    reference_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    reference_truth_hash: str | None = None

    for seed in SEEDS:
        direct_seed_root = direct_root / f"seed_{seed}"
        no_dropout_seed_root = no_dropout_root / f"seed_{seed}"
        direct_store = direct_seed_root / direct_store_name
        no_dropout_store = no_dropout_seed_root / no_dropout_store_name
        route_info: dict[str, Any] = {}

        for route_label, seed_root, store in (
            ("station_dropout", direct_seed_root, direct_store),
            ("no_station_dropout", no_dropout_seed_root, no_dropout_store),
        ):
            metadata = read_json(store / "metadata.json")
            if metadata.get("contract") != EXPECTED_CONTRACT:
                raise StationDropoutAuditError(
                    f"{route_label} seed {seed}: metadata contract mismatch"
                )
            if int(metadata.get("seed", -1)) != seed:
                raise StationDropoutAuditError(
                    f"{route_label} seed {seed}: metadata seed mismatch"
                )
            if metadata.get("completed") is not True:
                raise StationDropoutAuditError(
                    f"{route_label} seed {seed}: incomplete metadata"
                )
            if metadata.get("mode") != "target_masked":
                raise StationDropoutAuditError(
                    f"{route_label} seed {seed}: expected target_masked evaluation"
                )
            if metadata.get("all_context_gauges_masked") is not False:
                raise StationDropoutAuditError(
                    f"{route_label} seed {seed}: fitting-station context was not retained"
                )
            checkpoint = _find_matching_checkpoint(seed_root, metadata)
            axes = load_axes(store)
            if reference_axes is None:
                reference_axes = axes
            else:
                for axis_name, expected, actual in zip(
                    ("anchors", "stations", "leads"), reference_axes, axes, strict=True
                ):
                    assert_array_equal(
                        expected, actual, f"{route_label} seed {seed} {axis_name}"
                    )

            prediction = np.load(
                store / "preds_mm.npy", mmap_mode="r", allow_pickle=False
            )
            truth = np.load(store / "trues_mm.npy", mmap_mode="r", allow_pickle=False)
            expected_shape = (len(axes[0]), len(axes[1]), len(axes[2]))
            if prediction.shape != expected_shape or truth.shape != expected_shape:
                raise StationDropoutAuditError(
                    f"{route_label} seed {seed}: tensor/axis shape mismatch"
                )
            if prediction.dtype != np.float32 or truth.dtype != np.float32:
                raise StationDropoutAuditError(
                    f"{route_label} seed {seed}: predictions and truth must be float32"
                )
            truth_hash = sha256_file(store / "trues_mm.npy")
            if reference_truth_hash is None:
                reference_truth_hash = truth_hash
            elif truth_hash != reference_truth_hash:
                raise StationDropoutAuditError(
                    f"{route_label} seed {seed}: truth tensor differs across routes/seeds"
                )

            if hash_prediction_sources:
                for name in ("metadata.json", "preds_mm.npy", "trues_mm.npy"):
                    store_name = store_name_for(
                        route_label, direct_store_name, no_dropout_store_name
                    )
                    audit["source_hashes"][
                        f"{route_label}/seed_{seed}/{store_name}/{name}"
                    ] = sha256_file(store / name)
            audit["source_hashes"][
                f"{route_label}/seed_{seed}/{checkpoint.name}"
            ] = sha256_file(checkpoint)
            route_info[route_label] = {
                "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
                "selected_epoch": int(metadata["selected_epoch"]),
                "evaluation_tag": str(metadata.get("evaluation_tag", "")),
                "prediction_shape": list(prediction.shape),
                "prediction_dtype": str(prediction.dtype),
                "truth_sha256": truth_hash,
            }

        if (
            route_info["station_dropout"]["evaluation_tag"]
            != route_info["no_station_dropout"]["evaluation_tag"]
        ):
            raise StationDropoutAuditError(
                f"seed {seed}: evaluation tags differ between matched routes"
            )
        audit["seeds"][str(seed)] = route_info

    assert reference_axes is not None
    anchors, stations, lead_axis = reference_axes
    if expected_issue_count is not None and len(anchors) != expected_issue_count:
        raise StationDropoutAuditError(
            f"expected {expected_issue_count} issue times, found {len(anchors)}"
        )
    if expected_station_count is not None and len(stations) != expected_station_count:
        raise StationDropoutAuditError(
            f"expected {expected_station_count} stations, found {len(stations)}"
        )
    assert_array_equal(
        lead_axis.astype(np.int16), EXPECTED_LEAD_AXIS, "frozen 5-min lead axis"
    )
    dates, _ = date_axes(anchors, expected_date_count)
    audit["common_axes"] = {
        "issue_count": len(anchors),
        "station_count": len(stations),
        "lead_count": len(lead_axis),
        "issuance_date_count": len(dates),
        "anchor_start": str(anchors.astype("datetime64[ns]")[0]),
        "anchor_end": str(anchors.astype("datetime64[ns]")[-1]),
    }
    audit["truth_exact_equal_across_routes_and_seeds"] = True
    return audit, reference_axes


def store_name_for(
    route_label: str, direct_store_name: str, no_dropout_store_name: str
) -> str:
    return (
        direct_store_name
        if route_label == "station_dropout"
        else no_dropout_store_name
    )


def compute_csi_tables(
    direct_root: Path,
    no_dropout_root: Path,
    direct_store_name: str,
    no_dropout_store_name: str,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    bootstrap_reps: int,
    bootstrap_seed: int,
    *,
    expected_date_count: int | None = 244,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    anchors, _, lead_axis = axes
    dates, date_index = date_axes(anchors, expected_date_count)
    weights = bootstrap_weights(len(dates), bootstrap_reps, bootstrap_seed)
    weights_hash = hashlib.sha256(weights.tobytes(order="C")).hexdigest()
    direct_stores = [
        direct_root / f"seed_{seed}" / direct_store_name for seed in SEEDS
    ]
    no_dropout_stores = [
        no_dropout_root / f"seed_{seed}" / no_dropout_store_name for seed in SEEDS
    ]
    truth_memmap = np.load(
        direct_stores[0] / "trues_mm.npy", mmap_mode="r", allow_pickle=False
    )
    direct_memmaps = [
        np.load(store / "preds_mm.npy", mmap_mode="r", allow_pickle=False)
        for store in direct_stores
    ]
    no_dropout_memmaps = [
        np.load(store / "preds_mm.npy", mmap_mode="r", allow_pickle=False)
        for store in no_dropout_stores
    ]

    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for lead in LEADS.astype(int):
        matches = np.flatnonzero(lead_axis.astype(int) == lead)
        if len(matches) != 1:
            raise StationDropoutAuditError(f"lead {lead} is not unique")
        lead_index = int(matches[0])
        truth = np.asarray(truth_memmap[:, :, lead_index], dtype=np.float32)
        direct_predictions = [
            roundtrip_f16(np.asarray(values[:, :, lead_index]))
            for values in direct_memmaps
        ]
        no_dropout_predictions = [
            roundtrip_f16(np.asarray(values[:, :, lead_index]))
            for values in no_dropout_memmaps
        ]
        finite = np.isfinite(truth)
        for prediction in direct_predictions + no_dropout_predictions:
            finite &= np.isfinite(prediction)

        for threshold in THRESHOLDS:
            direct_counts = np.stack(
                [
                    date_counts(
                        prediction,
                        truth,
                        finite,
                        date_index,
                        float(threshold),
                        len(dates),
                    )
                    for prediction in direct_predictions
                ]
            )
            no_dropout_counts = np.stack(
                [
                    date_counts(
                        prediction,
                        truth,
                        finite,
                        date_index,
                        float(threshold),
                        len(dates),
                    )
                    for prediction in no_dropout_predictions
                ]
            )
            direct_seed_csi = csi(direct_counts.sum(axis=1))
            no_dropout_seed_csi = csi(no_dropout_counts.sum(axis=1))
            direct_boot = np.einsum(
                "bd,rdc->brc", weights, direct_counts, optimize=True
            )
            no_dropout_boot = np.einsum(
                "bd,rdc->brc", weights, no_dropout_counts, optimize=True
            )
            delta_boot = np.nanmean(csi(direct_boot), axis=1) - np.nanmean(
                csi(no_dropout_boot), axis=1
            )
            direct_mean = float(np.nanmean(direct_seed_csi))
            no_dropout_mean = float(np.nanmean(no_dropout_seed_csi))
            delta = direct_mean - no_dropout_mean
            low, high = np.nanpercentile(delta_boot, [2.5, 97.5])
            rows.append(
                {
                    "lead_min": lead,
                    "threshold_mm": float(threshold),
                    "station_dropout_csi_mean": direct_mean,
                    "station_dropout_csi_sd": float(
                        np.nanstd(direct_seed_csi, ddof=1)
                    ),
                    "no_station_dropout_csi_mean": no_dropout_mean,
                    "no_station_dropout_csi_sd": float(
                        np.nanstd(no_dropout_seed_csi, ddof=1)
                    ),
                    "delta_csi_station_dropout_minus_no_station_dropout": delta,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                    "n_finite_common": int(finite.sum()),
                    "bootstrap_valid_reps": int(np.isfinite(delta_boot).sum()),
                    "bootstrap_reps": bootstrap_reps,
                }
            )
            for seed_index, seed in enumerate(SEEDS):
                seed_rows.append(
                    {
                        "seed": seed,
                        "lead_min": lead,
                        "threshold_mm": float(threshold),
                        "station_dropout_csi": float(direct_seed_csi[seed_index]),
                        "no_station_dropout_csi": float(
                            no_dropout_seed_csi[seed_index]
                        ),
                        "delta_csi_station_dropout_minus_no_station_dropout": float(
                            direct_seed_csi[seed_index]
                            - no_dropout_seed_csi[seed_index]
                        ),
                    }
                )
                for date_position, date in enumerate(dates):
                    direct_row = direct_counts[seed_index, date_position]
                    no_dropout_row = no_dropout_counts[seed_index, date_position]
                    daily_rows.append(
                        {
                            "date": str(date),
                            "seed": seed,
                            "lead_min": lead,
                            "threshold_mm": float(threshold),
                            "station_dropout_hit": int(direct_row[0]),
                            "station_dropout_miss": int(direct_row[1]),
                            "station_dropout_false_alarm": int(direct_row[2]),
                            "station_dropout_correct_negative": int(direct_row[3]),
                            "no_station_dropout_hit": int(no_dropout_row[0]),
                            "no_station_dropout_miss": int(no_dropout_row[1]),
                            "no_station_dropout_false_alarm": int(no_dropout_row[2]),
                            "no_station_dropout_correct_negative": int(
                                no_dropout_row[3]
                            ),
                        }
                    )
        del truth, direct_predictions, no_dropout_predictions, finite
    return (
        pd.DataFrame(rows),
        pd.DataFrame(seed_rows),
        pd.DataFrame(daily_rows),
        weights_hash,
    )


def _single_directory_name(value: str, option: str) -> str:
    if Path(value).name != value or value in {".", ".."}:
        raise StationDropoutAuditError(
            f"{option} must be a single store directory name"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--no-dropout-root", type=Path, required=True)
    parser.add_argument("--direct-store", required=True)
    parser.add_argument("--no-dropout-store", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--expected-issue-count", type=int, default=35_088)
    parser.add_argument("--expected-station-count", type=int, default=128)
    parser.add_argument("--expected-date-count", type=int, default=244)
    parser.add_argument(
        "--skip-prediction-hashes",
        action="store_true",
        help="Skip expensive SHA-256 hashes of prediction tensors; contract checks remain enabled.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.bootstrap_reps < 1_000:
        raise StationDropoutAuditError("at least 1,000 bootstrap replicates are required")
    direct_store = _single_directory_name(args.direct_store, "--direct-store")
    no_dropout_store = _single_directory_name(
        args.no_dropout_store, "--no-dropout-store"
    )
    audit, axes = validate_contract(
        args.direct_root,
        args.no_dropout_root,
        direct_store,
        no_dropout_store,
        expected_issue_count=args.expected_issue_count,
        expected_station_count=args.expected_station_count,
        expected_date_count=args.expected_date_count,
        hash_prediction_sources=not args.skip_prediction_hashes,
    )
    comparison, by_seed, daily, weights_hash = compute_csi_tables(
        args.direct_root,
        args.no_dropout_root,
        direct_store,
        no_dropout_store,
        axes,
        args.bootstrap_reps,
        args.bootstrap_seed,
        expected_date_count=args.expected_date_count,
    )
    outputs = {
        "csi_by_lead_threshold.csv": comparison,
        "csi_by_seed.csv": by_seed,
        "daily_categorical_counts.csv": daily,
    }
    for name, frame in outputs.items():
        atomic_csv(args.output_dir / name, frame)

    output_paths = [args.output_dir / name for name in outputs]
    manifest = {
        "schema": "r2p-station-dropout-4km10min-2024-2025-v1",
        "status": "complete",
        "scientific_contract": {
            "estimand": "CSI(station-dropout Direct R2P) - CSI(no-station-dropout control)",
            "training_change": "station masking enabled versus disabled; separately trained three-member routes",
            "evaluation_support": "identical issue-time, station, lead, truth and finite-value axes",
            "issues": int(len(axes[0])),
            "heldout_stations": int(len(axes[1])),
            "dates": int(audit["common_axes"]["issuance_date_count"]),
            "leads_min": LEADS.astype(int).tolist(),
            "thresholds_mm": THRESHOLDS.tolist(),
            "members_per_route": len(SEEDS),
            "prediction_numeric_contract": "float16 round-trip before thresholding",
            "bootstrap_unit": "issuance date",
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_weights_sha256": weights_hash,
            "intervals": "pointwise percentile 95%; not multiplicity adjusted",
            "direct_store": direct_store,
            "no_dropout_store": no_dropout_store,
        },
        "input_audit": audit,
        "summary": {
            "cell_count": int(len(comparison)),
            "cells_with_pointwise_ci_excluding_zero": int(
                comparison.ci_excludes_zero.sum()
            ),
        },
        "outputs": {
            f"metrics/{path.name}": {"sha256": sha256_file(path)}
            for path in output_paths
        },
        "script": {
            "path": "src/evaluation/station_dropout_comparison.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    manifest_path = args.output_dir / "audit_manifest.json"
    atomic_json(manifest_path, manifest)
    print("validated complete matched station-dropout comparison for seeds 0, 1 and 2")
    print(f"saved: {args.output_dir / 'csi_by_lead_threshold.csv'}")
    print(f"saved: {manifest_path}")


if __name__ == "__main__":
    main()
