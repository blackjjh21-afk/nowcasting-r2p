from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "src" / "evaluation" / "radar_only_comparison.py"
    spec = importlib.util.spec_from_file_location("radar_only_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["radar_only_comparison"] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_csi_uses_hit_miss_false_alarm_only() -> None:
    counts = np.asarray([[4, 2, 2, 99], [0, 0, 0, 10]], dtype=np.int64)
    values = MODULE.csi(counts)
    assert values[0] == pytest.approx(0.5)
    assert np.isnan(values[1])


def test_date_counts_preserve_paired_issuance_date_blocks() -> None:
    truth = np.asarray([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    prediction = np.asarray([[2.0, 2.0], [0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    finite = np.ones_like(truth, dtype=bool)
    date_index = np.asarray([0, 0, 1], dtype=np.int16)
    counts = MODULE.date_counts(
        prediction, truth, finite, date_index, threshold=1.0, n_dates=2
    )
    # Date 0: hit=1, miss=1, false alarm=1, correct negative=1.
    # Date 1: hit=1, miss=1, false alarm=0, correct negative=0.
    np.testing.assert_array_equal(counts, [[1, 1, 1, 1], [1, 1, 0, 0]])


def test_axis_audit_rejects_standard_radar_only_mismatch() -> None:
    left = np.asarray([60, 90, 120], dtype=np.int16)
    MODULE.assert_array_equal(left, left.copy(), "same")
    with pytest.raises(MODULE.RadarOnlyAuditError, match="array mismatch"):
        MODULE.assert_array_equal(left, np.asarray([60, 90, 150]), "lead")


def test_bootstrap_resamples_whole_dates() -> None:
    weights = MODULE.bootstrap_weights(n_dates=5, replicates=64, seed=123)
    assert weights.shape == (64, 5)
    np.testing.assert_array_equal(weights.sum(axis=1), np.full(64, 5))
    assert np.issubdtype(weights.dtype, np.integer)


def test_cli_requires_portable_input_and_output_paths() -> None:
    parser = MODULE.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--r2p-root",
            "results",
            "--output-dir",
            "derived",
        ]
    )
    assert args.r2p_root == Path("results")


def test_cli_writes_scores_and_paired_intervals(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "predictions"
    anchors = (
        np.datetime64("2024-06-01", "ns")
        + np.arange(244) * np.timedelta64(1, "D")
    ).astype(np.int64)
    stations = np.asarray([101, 202], dtype=np.int64)
    leads = np.arange(5, 181, 5, dtype=np.int16)
    shape = (len(anchors), len(stations), len(leads))
    monkeypatch.setattr(MODULE, "EXPECTED_SHAPE", shape)
    truth = np.full(shape, 25.0, dtype=np.float32)
    for seed in range(3):
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir(parents=True)
        checkpoint = seed_root / "epoch_001.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        for store_name, radar_only in (
            (MODULE.STANDARD_STORE, False),
            (MODULE.RADAR_ONLY_STORE, True),
        ):
            store = seed_root / store_name
            store.mkdir()
            np.save(store / "preds_mm.npy", np.zeros_like(truth) if radar_only else truth)
            np.save(store / "trues_mm.npy", truth)
            np.save(store / "anchor_times_ns.npy", anchors)
            np.save(store / "station_ids.npy", stations)
            np.save(store / "lead_minutes.npy", leads)
            (store / "COMPLETED").write_text("complete\n", encoding="utf-8")
            metadata = {
                "checkpoint_name": checkpoint.name,
                "checkpoint_sha256": MODULE.sha256_file(checkpoint),
                "completed": True,
                "contract": MODULE.EXPECTED_CONTRACT,
                "evaluation_tag": "radar_only" if radar_only else "standard",
                "mode": "radar_only_all_context_gauges_masked" if radar_only else "target_masked",
                "all_context_gauges_masked": radar_only,
                "seed": seed,
            }
            (store / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    output = tmp_path / "scores"
    MODULE.main([
        "--r2p-root", str(root),
        "--output-dir", str(output),
        "--bootstrap-reps", "1000",
    ])
    assert {path.name for path in output.iterdir()} == {
        "csi_by_lead_threshold.csv", "csi_by_seed.csv",
        "daily_categorical_counts.csv", "audit_manifest.json",
    }
    scores = MODULE.pd.read_csv(output / "csi_by_lead_threshold.csv")
    assert len(scores) == 15
    np.testing.assert_allclose(scores.delta_csi_radar_only_minus_standard, -1.0)
    np.testing.assert_allclose(scores.ci_low, -1.0)
    np.testing.assert_allclose(scores.ci_high, -1.0)
    assert len(MODULE.pd.read_csv(output / "csi_by_seed.csv")) == 45
    assert len(MODULE.pd.read_csv(output / "daily_categorical_counts.csv")) == 244 * 45
    manifest = json.loads((output / "audit_manifest.json").read_text())
    assert manifest["input_audit"]["truth_exact_equal_across_modes_and_seeds"] is True
    assert manifest["scientific_contract"]["bootstrap_reps"] == 1000
    for relative, record in manifest["outputs"].items():
        assert record["sha256"] == MODULE.sha256_file(output / Path(relative).name)
