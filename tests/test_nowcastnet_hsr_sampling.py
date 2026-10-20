"""Small CPU fixtures for training-only, cached rain-importance sampling."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nowcastnet_hsr.sampling import build_training_weights
from test_nowcastnet_hsr_data import data, setup_grid, make_cache, make_tiffs


def fixture(tmp_path, months=None):
    config, mask = setup_grid(tmp_path, months=months)
    config["cache_root"] = "training_weights"
    config["field"] = {"importance_sampling": True, "sampling_epsilon": 0.5}
    config["model"]["rain_rate_cap"] = 128.0
    return config, mask


def rewrite_frames(folder, values):
    path = folder / "radar.npy"
    frames = np.load(path)
    frames[:len(values)] = values
    np.save(path, frames)
    manifest = json.loads((folder / "complete.json").read_text())
    manifest["files"]["radar.npy"] = {"size": path.stat().st_size, "sha256": data._hash(path)}
    (folder / "complete.json").write_text(json.dumps(manifest))


def test_weights_exact_25_frame_sum_dataset_order_and_only_train_reads(tmp_path, monkeypatch):
    config, mask = fixture(tmp_path)
    folder, times = make_cache(config, mask)
    frames = np.zeros((30, 2, 2), np.float32)
    frames[:, 0, 0] = np.linspace(0, 0.4, 30)
    frames[:, 1, 1] = 99  # Padding must never affect scores.
    rewrite_frames(folder, frames)
    dataset = data.HSRDataset(config, "train", issues=times[[6, 8, 11]])
    real_read, read_indices = dataset._read_frame, []
    def record(index):
        read_indices.append(index)
        return real_read(index)
    monkeypatch.setattr(dataset, "_read_frame", record)
    weights, provenance = build_training_weights(dataset, config)
    rate = data.dbz_to_rate(frames[:, 0, 0]).astype(np.float64)
    expected = [(-np.expm1(-rate[i - 6:i + 19])).sum() + 0.5 for i in (6, 8, 11)]
    np.testing.assert_allclose(weights.numpy(), expected, rtol=1e-12)
    assert weights.dtype == torch.float64 and weights.device.type == "cpu"
    assert read_indices == list(range(30))  # Not 3 x 25 repeated reads.
    assert provenance["eligible_window_count"] == 3 and provenance["replacement"]
    assert [item["month"] for item in provenance["months"]] == ["201906"]
    assert provenance["available_frame_count"] == 30
    json.dumps(provenance)
    # Cache re-use performs no image reads, and epsilon needs no new scan.
    monkeypatch.setattr(dataset, "_read_frame", lambda _: pytest.fail("Cache must be reused"))
    config["field"]["sampling_epsilon"] = 2.0
    second, record = build_training_weights(dataset, config)
    np.testing.assert_allclose(second.numpy(), weights.numpy() + 1.5)
    assert record["months"] == provenance["months"]


def test_contiguous_month_boundary_is_25_real_frames(tmp_path):
    config, mask = fixture(tmp_path, months=[6, 7])
    june, july = data._month_times("201906"), data._month_times("201907")
    first, second = np.zeros(len(june), bool), np.zeros(len(july), bool)
    first[-7:], second[:18] = True, True
    make_cache(config, mask, "201906", first)
    make_cache(config, mask, "201907", second)
    dataset = data.HSRDataset(config, "train", issues=june[[-1]])
    weights, provenance = build_training_weights(dataset, config)
    frame_sum = -np.expm1(-dataset._read_frame(len(june) - 1)[mask].astype(float)).sum()
    np.testing.assert_allclose(weights.numpy(), [25 * frame_sum + 0.5])
    assert len(provenance["months"]) == 2


def test_missing_frames_not_read_and_no_window_bridges_gap(tmp_path, monkeypatch):
    config, mask = fixture(tmp_path)
    available = np.zeros(len(data._month_times("201906")), bool)
    available[:60] = True
    available[30] = False
    make_cache(config, mask, available=available)
    dataset = data.HSRDataset(config, "train")
    real, read = dataset._read_frame, []
    def track(index):
        assert index != 30
        read.append(index)
        return real(index)
    monkeypatch.setattr(dataset, "_read_frame", track)
    values, provenance = build_training_weights(dataset, config)
    assert len(read) == 59 and provenance["available_frame_count"] == 59
    assert np.all((dataset.indices + 18 < 30) | (dataset.indices - 6 > 30))
    assert len(values) == len(dataset)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_rejects_other_splits_before_frame_io(tmp_path, split, monkeypatch):
    config, mask = fixture(tmp_path)
    make_cache(config, mask)
    dataset = data.HSRDataset(config, "train")
    dataset.split = split
    monkeypatch.setattr(dataset, "_read_frame", lambda _: pytest.fail("Must not read nontraining frame"))
    with pytest.raises(ValueError, match="only training"):
        build_training_weights(dataset, config)
    assert not (tmp_path / "training_weights").exists()


def test_rejects_prediction_only_training_dataset(tmp_path):
    config, mask = fixture(tmp_path)
    make_cache(config, mask)
    with pytest.raises(ValueError, match="future targets"):
        build_training_weights(data.HSRDataset(config, "train", require_future=False), config)


def test_disabled_returns_uniform_without_cache_or_images(tmp_path, monkeypatch):
    config, mask = fixture(tmp_path)
    make_cache(config, mask)
    dataset = data.HSRDataset(config, "train")
    config["field"]["importance_sampling"] = False
    monkeypatch.setattr(dataset, "_read_frame", lambda _: pytest.fail("Disabled sampler reads no images"))
    weights, provenance = build_training_weights(dataset, config)
    assert torch.equal(weights, torch.ones(len(dataset), dtype=torch.float64))
    assert not provenance["enabled"] and not (tmp_path / "training_weights").exists()


def test_source_change_and_cap_change_invalidate_frame_cache(tmp_path):
    config, mask = fixture(tmp_path)
    folder, _ = make_cache(config, mask)
    first, p1 = build_training_weights(data.HSRDataset(config, "train"), config)
    rewrite_frames(folder, np.zeros((30, 2, 2), np.float32))
    second, p2 = build_training_weights(data.HSRDataset(config, "train"), config)
    assert not torch.equal(first, second) and torch.all(second == 0.5)
    assert p1["months"][0]["path"] != p2["months"][0]["path"]
    config["model"]["rain_rate_cap"] = 64.0
    _, p3 = build_training_weights(data.HSRDataset(config, "train"), config)
    assert p2["months"][0]["path"] != p3["months"][0]["path"]
    assert p3["rain_rate_cap_mm_h"] == 64.0


def test_tiff_value_change_invalidates_frame_cache(tmp_path):
    import tifffile
    config, _ = fixture(tmp_path)
    paths = make_tiffs(config, "201906", range(25))
    first, p1 = build_training_weights(data.HSRDataset(config, "train"), config)
    tifffile.imwrite(paths[0], np.zeros((2, 2), np.float32))
    second, p2 = build_training_weights(data.HSRDataset(config, "train"), config)
    assert second[0] < first[0]
    assert p1["months"][0]["path"] != p2["months"][0]["path"]


@pytest.mark.parametrize("split,month", [("train", "201906"), ("validation", "202306"), ("test", "202406")])
def test_physical_128_cap_on_both_input_target_all_splits(tmp_path, split, month, monkeypatch):
    config, mask = fixture(tmp_path)
    folder, times = make_cache(config, mask, month)
    high = np.full((30, 2, 2), 0.8, np.float32)
    rewrite_frames(folder, high)
    if split == "test":
        monkeypatch.setattr(data, "_common_test_issues", lambda _: (times[[6]], {}))
    dataset = data.HSRDataset(config, split)
    sample = dataset[0]
    assert np.all(sample["history"][:, mask] == 128.0)
    assert np.all(sample["target"][:, mask] == 128.0)
    assert np.all(sample["history"][:, ~mask] == 0.0)
    assert dataset.provenance["rain_rate_cap_mm_h"] == 128.0
    assert data.dbz_to_rate(np.array([0.8]))[0] > 128.0  # Conversion utility stays physical.


@pytest.mark.parametrize("cap", [0, -1, float("nan"), float("inf"), True])
def test_bad_cap_rejected(tmp_path, cap):
    config, _ = fixture(tmp_path)
    config["model"]["rain_rate_cap"] = cap
    with pytest.raises(ValueError, match="rain_rate_cap"):
        data.HSRDataset(config, "train")


@pytest.mark.parametrize("epsilon", [0, -1, float("nan"), float("inf"), True])
def test_bad_epsilon_rejected(tmp_path, epsilon):
    config, mask = fixture(tmp_path)
    make_cache(config, mask)
    config["field"]["sampling_epsilon"] = epsilon
    with pytest.raises(ValueError, match="sampling_epsilon"):
        build_training_weights(data.HSRDataset(config, "train"), config)


def test_weight_cache_tamper_is_detected(tmp_path):
    config, mask = fixture(tmp_path)
    make_cache(config, mask)
    dataset = data.HSRDataset(config, "train")
    _, provenance = build_training_weights(dataset, config)
    path = Path(provenance["months"][0]["path"])
    with np.load(path, allow_pickle=False) as content:
        payload = {key: np.array(content[key], copy=True) for key in content}
    payload["scores"][0] += 1.0
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        build_training_weights(dataset, config)


def test_weighted_draws_fixed_length_reproducible(tmp_path):
    config, mask = fixture(tmp_path)
    make_cache(config, mask)
    dataset = data.HSRDataset(config, "train")
    weights, _ = build_training_weights(dataset, config)
    def draw():
        return list(torch.utils.data.WeightedRandomSampler(weights, len(dataset), replacement=True,
                    generator=torch.Generator().manual_seed(40)))
    assert draw() == draw() and len(draw()) == len(dataset)
