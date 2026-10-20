"""CPU-only small-grid tests: no external images, GPU, gauges, or RRBF reads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_nowcastnet_hsr_data", ROOT / "src/nowcastnet_hsr/data.py")
data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = data
SPEC.loader.exec_module(data)


def write_json(path, value):
    path.write_text(json.dumps(value))


def setup_grid(tmp_path, months=None):
    radar = tmp_path / "radar"
    radar.mkdir()
    mask = np.array([[True, True], [True, False]])
    np.savez(radar / "grid_coordinates.npz", latitude=np.full((2, 2), 37.), longitude=np.full((2, 2), 127.), source_index_valid_mask=mask)
    write_json(radar / "manifest.json", {"timezone": "Asia/Seoul", "prepared_contract": {
        "shape": [2, 2], "dtype": "float32 TIFF", "normalization": data.NORMALIZATION,
        "spatial_resolution_km": 4, "temporal_resolution_minutes": 10, "minute_phase": 0,
        "spatial_transform": "already reversed y; no further transform"}})
    config = {"project_root": str(tmp_path), "radar_root": "radar", "hsr_cache_root": "cache",
              "train_years": [2019], "validation_years": [2023], "test_years": [2024],
              "months": months or [6], "train_stride_minutes": 10, "validation_stride_minutes": 10,
              "model": {"grid_size": 2, "input_steps": 7, "output_steps": 18}}
    return config, mask


def make_cache(config, mask, month="201906", available=None):
    root = Path(config["project_root"])
    folder = root / config["hsr_cache_root"] / month
    folder.mkdir(parents=True)
    times = data._month_times(month)
    frames = np.broadcast_to(np.array([[.1, .2], [.3, 99]], np.float32), (len(times), 2, 2)).copy()
    # Distinct rows/columns check that no transpose/flip is introduced.
    if available is None:
        available = np.zeros(len(times), bool)
        available[:30] = True
    np.save(folder / "radar.npy", frames)
    np.save(folder / "times_kst_ns.npy", times)
    np.save(folder / "radar_valid.npy", available)
    np.save(folder / "spatial_mask.npy", mask)
    radar = root / config["radar_root"]
    identity = {"schema": data.CACHE_SCHEMA, "month": month,
                "times_sha256": hashlib.sha256(times.tobytes()).hexdigest(),
                "hsr_manifest": {"sha256": data._hash(radar / "manifest.json")},
                "hsr_coordinates": {"sha256": data._hash(radar / "grid_coordinates.npz")}}
    manifest = {"schema": data.CACHE_SCHEMA, "identity": identity, "shape": [len(times), 2, 2],
                "radar_normalization": data.NORMALIZATION, "files": {
                    p.name: {"size": p.stat().st_size, "sha256": data._hash(p)} for p in folder.iterdir()}}
    write_json(folder / "complete.json", manifest)
    return folder, times


def make_tiffs(config, month, indices):
    times = data._month_times(month)
    paths = []
    for index in indices:
        stamp = str(times[index].astype("datetime64[ns]")).replace("-", "").replace(":", "").replace("T", "")[:12]
        path = Path(config["project_root"]) / config["radar_root"] / stamp[:4] / stamp[4:6] / stamp[6:8] / (stamp + ".tiff")
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, np.array([[.1, .2], [.3, 0]], np.float32))
        paths.append(path)
    return paths


def test_mp_conversion_dry_units_and_roundtrip():
    dbz = np.array([0, .1, .2, .3, .4], np.float32)
    rates = data.dbz_to_rate(dbz)
    assert rates[0] == 0 and rates.dtype == np.float32
    np.testing.assert_allclose(rates[1:], (10 ** (dbz[1:].astype(float) * 10) / 200) ** (1 / 1.6), rtol=1e-6)
    np.testing.assert_allclose(data.rate_to_dbz_numpy(rates), dbz, atol=1e-7)
    np.testing.assert_allclose(data.dbz_to_rate(dbz * 100, normalized=False), rates, rtol=1e-6)
    assert data.rate_to_dbz_numpy(np.array([0., 1e-6])) .tolist() == [0., 0.]


@pytest.mark.parametrize("value", [np.array([np.nan]), np.array([np.inf]), np.array([-1.])])
def test_invalid_normalized_or_rate_values_fail(value):
    with pytest.raises(ValueError):
        data.dbz_to_rate(value)
    with pytest.raises(ValueError):
        data.rate_to_dbz_numpy(value)


def test_cache_hsr_only_without_any_rrbf_files_or_other_split(tmp_path, monkeypatch):
    config, mask = setup_grid(tmp_path)
    folder, times = make_cache(config, mask)
    original = data.np.load
    opened = []
    def guarded(path, *args, **kwargs):
        opened.append(str(path))
        assert "rates_mm_h" not in str(path) and "rate_found" not in str(path)
        assert "2023" not in str(path) and "2024" not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(data.np, "load", guarded)
    dataset = data.HSRDataset(config, "train")
    assert len(dataset) == 6  # 30 valid frames -> issues 6..11.
    assert dataset.issues is dataset.issue_times_ns
    sample = dataset[0]
    assert sample["history"].shape == (7, 2, 2) and sample["target"].shape == (18, 2, 2)
    np.testing.assert_allclose(sample["history"][0], data.dbz_to_rate(np.array([[.1, .2], [.3, 0]], np.float32)))
    assert sample["history"].dtype == np.float32 and sample["mask"].dtype == bool
    assert sample["issue_ns"] == times[6]
    assert not (folder / "rates_mm_h.npy").exists()
    assert dataset.provenance["large_cache_payload_hashes_recomputed"] is False


def test_missing_cache_input_is_not_replaced_by_existing_tiff(tmp_path):
    config, mask = setup_grid(tmp_path)
    available = np.zeros(len(data._month_times("201906")), bool)
    available[:30] = True
    available[3] = False
    _, times = make_cache(config, mask, available=available)
    make_tiffs(config, "201906", range(30))
    with pytest.raises(ValueError, match="missing HSR"):
        data.HSRDataset(config, "train", issues=times[[6]])


def test_future_availability_required_only_when_future_requested(tmp_path):
    config, mask = setup_grid(tmp_path)
    available = np.zeros(len(data._month_times("201906")), bool)
    available[:7] = True
    _, times = make_cache(config, mask, available=available)
    prediction = data.HSRDataset(config, "train", require_future=False, issues=times[[6]])
    assert "target" not in prediction[0]
    with pytest.raises(ValueError, match="missing HSR"):
        data.HSRDataset(config, "train", require_future=True, issues=times[[6]])


def test_prediction_output_labels_cannot_cross_season_boundary(tmp_path):
    config, mask = setup_grid(tmp_path)
    times = data._month_times("201906")
    available = np.ones(len(times), bool)
    make_cache(config, mask, available=available)
    with pytest.raises(ValueError, match="boundary"):
        data.HSRDataset(config, "train", require_future=False, issues=times[[-1]])


def test_test_uses_common_axis_and_never_reads_future_tiffs(tmp_path, monkeypatch):
    config, _ = setup_grid(tmp_path)
    times = data._month_times("202406")
    make_tiffs(config, "202406", range(7))
    monkeypatch.setattr(data, "_common_test_issues", lambda _: (times[[6]], {"fixed": True}))
    original, reads = tifffile.imread, []
    def tracked(path):
        reads.append(Path(path).stem)
        return original(path)
    monkeypatch.setattr(tifffile, "imread", tracked)
    dataset = data.HSRDataset(config, "test", require_future=False)
    assert dataset.issue_times_ns.tolist() == times[[6]].tolist()
    sample = dataset[0]
    assert "target" not in sample and len(reads) == 7
    assert max(reads) == "202406010100"
    with pytest.raises(ValueError, match="fixed published"):
        data.HSRDataset(config, "test", require_future=False, issues=times[[7]])
    with pytest.raises(ValueError, match="missing HSR"):
        data.HSRDataset(config, "test", require_future=True)


def test_test_missing_input_fails_instead_of_filtering_common_axis(tmp_path, monkeypatch):
    config, _ = setup_grid(tmp_path)
    times = data._month_times("202406")
    make_tiffs(config, "202406", [0, 1, 2, 4, 5, 6])
    monkeypatch.setattr(data, "_common_test_issues", lambda _: (times[[6]], {}))
    with pytest.raises(ValueError, match="missing HSR"):
        data.HSRDataset(config, "test", require_future=False)


def test_field_v5_test_windows_use_only_hsr_and_reject_old_schema(tmp_path, monkeypatch):
    config, mask = setup_grid(tmp_path)
    make_cache(config, mask, "202406")
    config.update(schema="nowcastnet-hsr-field-v5", test_issue_policy="hsr_available_windows")
    monkeypatch.setattr(data, "_common_test_issues", lambda _: pytest.fail("Field-only test uses no station axis"))
    assert len(data.HSRDataset(config, "test")) == 6
    config["schema"] = "nowcastnet-hsr-field-v4"
    with pytest.raises(ValueError, match="field-only schema"):
        data.HSRDataset(config, "test")


def test_contiguous_month_crossing_allowed_but_season_gap_not(tmp_path):
    config, mask = setup_grid(tmp_path, months=[6, 7])
    june = data._month_times("201906")
    availability = np.zeros(len(june), bool)
    availability[-7:] = True
    make_cache(config, mask, "201906", availability)
    july = data._month_times("201907")
    availability = np.zeros(len(july), bool)
    availability[:18] = True
    make_cache(config, mask, "201907", availability)
    sample = data.HSRDataset(config, "train", issues=june[[-1]])[0]
    assert sample["target"].shape == (18, 2, 2)
    config["months"] = [6, 8]
    august = data._month_times("201908")
    availability = np.ones(len(august), bool)
    make_cache(config, mask, "201908", availability)
    with pytest.raises(ValueError, match="boundary"):
        data.HSRDataset(config, "train", issues=june[[-1]])


def test_month_cache_mask_tamper_rejected(tmp_path):
    config, mask = setup_grid(tmp_path)
    folder, _ = make_cache(config, mask)
    np.save(folder / "spatial_mask.npy", np.ones((2, 2), bool))
    with pytest.raises(ValueError, match="hash"):
        data.HSRDataset(config, "train")


def test_corrupt_or_incomplete_cache_not_silently_bypassed(tmp_path):
    config, mask = setup_grid(tmp_path)
    folder, _ = make_cache(config, mask)
    (folder / "complete.json").unlink()
    make_tiffs(config, "201906", range(30))
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        data.HSRDataset(config, "train")


def test_stride_subset_worker_pickle_and_lru(tmp_path):
    config, mask = setup_grid(tmp_path)
    make_cache(config, mask)
    config["train_stride_minutes"] = 20
    dataset = data.HSRDataset(config, "train", max_issues=2)
    assert len(dataset) == 2 and np.diff(dataset.issue_times_ns).tolist() == [2 * data.STEP_NS]
    dataset[0]
    assert len(dataset._maps) == 1
    cloned = pickle.loads(pickle.dumps(dataset))
    assert len(cloned._maps) == 0
    np.testing.assert_array_equal(cloned[0]["history"], dataset[0]["history"])


def test_split_overlap_and_invalid_grid_rejected(tmp_path):
    config, mask = setup_grid(tmp_path)
    make_cache(config, mask)
    config["validation_years"] = [2019]
    with pytest.raises(ValueError, match="overlap"):
        data.HSRDataset(config, "train")
    config["validation_years"] = [2023]
    config["model"]["input_steps"] = 6
    with pytest.raises(ValueError, match="7 input"):
        data.HSRDataset(config, "train")
