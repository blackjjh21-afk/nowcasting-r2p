"""Small-grid, CPU-only field evaluation/export tests; no production data/GPU."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nowcastnet_hsr import data, field_evaluation as field


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    config = {"schema": "nowcastnet-hsr-field-v5", "test_issue_policy": "hsr_available_windows",
              "experiment_root": str(tmp_path / "experiment"),
              "forecast_product": {"kind": "mean", "ensemble_members": 4, "noise_seed": 20261015}}
    mask = np.array([[True, True], [True, False]])
    issues = int(np.datetime64("2024-06-01T01:00", "ns").astype(np.int64)) + np.arange(5, dtype=np.int64) * data.STEP_NS
    target = np.broadcast_to(np.arange(1, 19, dtype=np.float32)[:, None, None], (18, 2, 2)).copy()
    target[:, ~mask] = np.nan
    calls, accesses, products = [], [], []

    class InputOnly(dict):
        def __getitem__(self, key):
            assert key != "target", "prediction read a future target"
            return super().__getitem__(key)

    class Dataset:
        def __init__(self, supplied, split, require_future=True, max_issues=None):
            assert supplied is config
            calls.append((split, require_future, max_issues))
            self.issue_times_ns = issues[:max_issues]
            self.spatial_mask = mask
            self.provenance = {"synthetic": True, "require_future": require_future}
            self.require_future = require_future

        def __len__(self):
            return len(self.issue_times_ns)

        def __getitem__(self, index):
            accesses.append(index)
            sample = {"history": np.ones((7, 2, 2), dtype=np.float32), "mask": mask,
                      "issue_ns": self.issue_times_ns[index]}
            if self.require_future:
                sample["target"] = target.copy()
                return sample
            return InputOnly(sample)

    def predict_product(model, history, stamps, supplied):
        assert history.device.type == "cpu" and history.shape == (1, 7, 2, 2)
        assert supplied is config and not torch.is_grad_enabled()
        products.append(list(stamps))
        return torch.from_numpy((target + 1)[None])

    def predict_members(model, history, stamps, supplied):
        return predict_product(model, history, stamps, supplied)[None].repeat(4, 1, 1, 1, 1)

    monkeypatch.setattr(field, "HSRDataset", Dataset)
    monkeypatch.setattr(field, "load_forecaster", lambda supplied, device: (
        torch.nn.Identity(), {"stage": "generator", "diagnostic_only": False, "checkpoint_sha256": "synthetic"}))
    monkeypatch.setattr(field, "predict_product", predict_product)
    monkeypatch.setattr(field, "predict_members", predict_members)
    return SimpleNamespace(config=config, mask=mask, issues=issues, target=target,
                           calls=calls, accesses=accesses, products=products)


def test_streaming_evaluation_scores_alignment_and_only_three_examples(synthetic):
    result = field.evaluate(synthetic.config)
    directory = Path(result["output_directory"])
    assert synthetic.calls == [("test", True, None)]
    assert synthetic.accesses == list(range(5))
    assert len(synthetic.products) == 5 and all(len(stamps) == 1 for stamps in synthetic.products)
    assert result["diagnostic_only"] is False and result["issue_count"] == 5
    assert len(result["example_files"]) == 3
    assert len(list(directory.glob("*.npz"))) == 3
    for product, leads in (("rain_rate", field.RATE_LEADS), ("rn60", field.RN60_LEADS)):
        scores = result["products"][product]
        assert scores["leads_minutes"] == list(leads)
        assert scores["sample_count"] == [15] * len(leads)
        np.testing.assert_allclose(scores["mae"], 1)
        np.testing.assert_allclose(scores["rmse"], 1)
        np.testing.assert_allclose(scores["mean_bias"], 1)
    rain = result["products"]["rain_rate"]
    assert rain["hits"][0][0] == 15 and rain["csi"][0][0] == 1
    assert rain["false_alarms"][3][1] == 15 and rain["frequency_bias"][3][1] is None
    assert rain["csi"][2][1] == 0  # no-event convention
    assert result["products"]["rn60"]["false_alarms"][1][1] == 15
    with np.load(directory / result["example_files"][0]) as example:
        assert example["predicted_rn60_mm"].shape == (13, 2, 2)
        assert example["predicted_rn60_mm"][0, 0, 0] == 4.5
        assert example["observed_rn60_mm"][0, 0, 0] == 3.5
        assert example["observed_rn60_mm"][-1, 0, 0] == 15.5
        assert example["rn60_valid_times_kst_ns"][0] == synthetic.issues[0] + 6 * data.STEP_NS
        assert example["rate_valid_times_kst_ns"][-1] == synthetic.issues[0] + 18 * data.STEP_NS
        assert np.all(example["predicted_rate_mm_h"][:, ~synthetic.mask] == 0)
        assert example["predicted_equivalent_dbz"].shape == (18, 2, 2)
    np.testing.assert_array_equal(np.load(directory / "issue_times_kst_ns.npy"), synthetic.issues)
    np.testing.assert_array_equal(np.load(directory / "spatial_mask.npy"), synthetic.mask)
    assert json.loads((directory / "metrics.json").read_text())["products"] == result["products"]
    assert json.loads((directory / "status.json").read_text())["state"] == "completed"
    with (directory / "scores.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 18 * 3 + 13 * 3 + 18
    assert {row["product"] for row in rows} == {"rain_rate", "rn60", "equivalent_dbz"}
    assert not list(directory.glob("*bank*"))

def test_input_only_prediction_is_bounded_and_never_accesses_targets(synthetic):
    result = field.predict(synthetic.config, max_issues=2)
    directory = Path(result["output_directory"])
    assert synthetic.calls == [("test", False, 2)] and synthetic.accesses == [0, 1]
    assert result["diagnostic_only"] and result["prediction_uses_future_observations"] is False
    assert len(result["example_files"]) == 2
    assert "products" not in result and not (directory / "scores.csv").exists()
    for filename in result["example_files"]:
        with np.load(directory / filename) as example:
            assert not any(name.startswith("observed") for name in example.files)
            assert example["predicted_rate_mm_h"].shape == (18, 2, 2)
    assert (directory / "metadata.json").is_file()


@pytest.mark.parametrize("maximum", [None, 0, -1, True, 1.5])
def test_prediction_requires_positive_explicit_bound(synthetic, maximum):
    with pytest.raises(ValueError, match="positive integer"):
        field.predict(synthetic.config, max_issues=maximum)
    assert not synthetic.calls


def test_diagnostic_evaluation_has_separate_destination_and_never_overwrites(synthetic):
    limited = field.evaluate(synthetic.config, max_issues=1)
    assert limited["diagnostic_only"] and limited["issue_count"] == 1
    before = (Path(limited["output_directory"]) / "metrics.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing existing"):
        field.evaluate(synthetic.config, max_issues=1)
    assert before == (Path(limited["output_directory"]) / "metrics.json").read_bytes()
    complete = field.evaluate(synthetic.config)
    prediction = field.predict(synthetic.config, max_issues=1)
    assert len({limited["output_directory"], complete["output_directory"], prediction["output_directory"]}) == 3


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.])
def test_scores_reject_invalid_supported_cells_instead_of_dropping(synthetic, bad):
    prediction = np.nan_to_num(synthetic.target)
    prediction[0, 0, 0] = bad
    with pytest.raises(ValueError, match="valid spatial mask"):
        field.StreamingScores(field.RATE_LEADS, "mm h-1").update(prediction, synthetic.target, synthetic.mask)


def test_failed_results_are_preserved_and_not_reused(synthetic, monkeypatch):
    def broken(*args):
        return torch.full((4, 1, 18, 2, 2), float("nan"))
    monkeypatch.setattr(field, "predict_members", broken)
    with pytest.raises(ValueError, match="valid spatial mask"):
        field.evaluate(synthetic.config, max_issues=1)
    status = Path(synthetic.config["experiment_root"]) / "field_evaluation/test_diagnostic_first_1/status.json"
    assert json.loads(status.read_text())["state"] == "failed"
    with pytest.raises(FileExistsError):
        field.evaluate(synthetic.config, max_issues=1)


def test_disk_preflight_runs_before_forecast_or_arrays(synthetic, monkeypatch):
    monkeypatch.setattr(field.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="free bytes"):
        field.predict(synthetic.config)
    assert not synthetic.products
    assert not list(Path(synthetic.config["experiment_root"]).rglob("*.npy"))


def test_only_field_schema_and_hsr_policy_allowed(synthetic):
    for changes in ({"schema": "nowcastnet-hsr-cnn-v1"}, {"test_issue_policy": "common_axis"}):
        with pytest.raises(ValueError, match="Field-only"):
            field.evaluate({**synthetic.config, **changes})
    assert not synthetic.calls


def test_completed_forecaster_is_mandatory(synthetic, monkeypatch):
    from nowcastnet_hsr.trainer import load_forecaster
    monkeypatch.setattr(field, "load_forecaster", load_forecaster)
    with pytest.raises(RuntimeError, match="completed full generator"):
        field.predict(synthetic.config, max_issues=1)
    assert not synthetic.products


def test_real_hsr_prediction_reads_only_seven_input_tiffs(tmp_path, synthetic, monkeypatch):
    """No future TIFFs exist; station/common-axis access must also be impossible."""
    radar = tmp_path / "radar"
    radar.mkdir()
    mask = synthetic.mask
    np.savez(radar / "grid_coordinates.npz", latitude=np.full((2, 2), 37.), longitude=np.full((2, 2), 127.),
             source_index_valid_mask=mask)
    (radar / "manifest.json").write_text(json.dumps({"timezone": "Asia/Seoul", "prepared_contract": {
        "shape": [2, 2], "dtype": "float32 TIFF", "normalization": data.NORMALIZATION,
        "spatial_resolution_km": 4, "temporal_resolution_minutes": 10, "minute_phase": 0}}))
    times = data._month_times("202406")
    for stamp in times[:7]:
        label = str(stamp.astype("datetime64[ns]")).replace("-", "").replace(":", "").replace("T", "")[:12]
        path = radar / "2024/06/01" / f"{label}.tiff"
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, np.full((2, 2), .2, dtype=np.float32))
    config = synthetic.config
    config.update(project_root=str(tmp_path), radar_root=str(radar), hsr_cache_root=str(tmp_path / "cache"),
                  train_years=[2019], validation_years=[2023], test_years=[2024], months=[6],
                  model={"grid_size": 2, "input_steps": 7, "output_steps": 18})
    read, original = [], tifffile.imread
    def tracked(path):
        read.append(Path(path).stem)
        return original(path)
    monkeypatch.setattr(tifffile, "imread", tracked)
    monkeypatch.setattr(data, "_common_test_issues", lambda config: pytest.fail("field prediction read station common axis"))
    monkeypatch.setattr(field, "HSRDataset", data.HSRDataset)
    result = field.predict(config, max_issues=1)
    assert len(read) == 7 and max(read) == "202406010100"
    with np.load(Path(result["output_directory"]) / result["example_files"][0]) as example:
        assert {"latitude", "longitude"} <= set(example.files)
        assert int(example["issue_time_kst_ns"]) == times[6]


def test_scores_mean_field_not_mean_of_draw_scores():
    mask = np.ones((2, 2), bool)
    draws = np.stack([np.full((18, 2, 2), 20., np.float32),
                      np.zeros((18, 2, 2), np.float32)])
    observation = np.full((18, 2, 2), 10., np.float32)
    scorer = field.StreamingEnsembleScores(2, mask)
    prediction = scorer.update(draws, observation, mask)
    np.testing.assert_allclose(prediction, 10)
    for result in scorer.result()["products"].values():
        np.testing.assert_allclose(result["rmse"], 0)
        np.testing.assert_allclose(result["csi"], 1)
        assert result["thresholds"] == [1, 5, 10]
    assert set(scorer.result()) == {"products"}
