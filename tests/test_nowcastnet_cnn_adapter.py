import numpy as np
import pytest

from cnn_readout.workflow import load_prepared_npz
from nowcastnet_hsr import cnn_adapter as adapter


def inputs(tmp_path, *, oof=True):
    issues = np.array(["2023-06-15", "2023-07-15", "2023-08-15", "2023-09-15"],
                      dtype="datetime64[ns]").astype(np.int64)
    rates = np.broadcast_to(np.arange(1, 19, dtype=np.float32)[None, :, None, None],
                             (4, 18, 256, 256)).copy()
    rate_path, issue_path = tmp_path / "mean.npy", tmp_path / "issues.npy"
    np.save(rate_path, rates); np.save(issue_path, issues)
    features = np.arange(4*2*36, dtype=np.float32).reshape(4, 2, 36)
    values = dict(schema=np.asarray(adapter.CONTEXT_SCHEMA), issue_times_ns=issues,
        station_ids=np.array([100, 200]), station_folds=np.array([0, 1], np.int8),
        grid_y=np.array([10, 20]), grid_x=np.array([30, 40]), xy_norm=np.array([[.1, .2], [.3, .4]]),
        context36=features, context_fitting_stations_only=np.asarray(True),
        context_available_at_issuance=np.asarray(True), targets_mm=np.ones((4, 2, 13), np.float32))
    if oof:
        values.update(oof_context36=np.stack([features + 1000*f for f in range(3)]),
                      oof_context_excludes_held_fold=np.asarray(True))
    context_path = tmp_path / "context.npz"
    np.savez(context_path, **values)
    return rate_path, issue_path, context_path, values


def test_equivalent_reflectivity_exact_native_precision_and_order():
    rate = np.zeros((18, 256, 256), np.float32)
    rate[:, 9:12, 19:22] = np.arange(1, 19)[:, None, None]
    result = adapter.rate_to_patches(rate, np.array([10, 40]), np.array([20, 40]))
    expected = ((10*np.log10(200.) + 16*np.log10(np.arange(1, 19, dtype=np.float32)))/100).astype(np.float16)
    assert result.dtype == np.float16 and result.shape == (2, 18, 3, 3)
    np.testing.assert_array_equal(result[0, :, 1, 1], expected)
    assert not result[1].any()


def test_adapter_directly_loads_in_public_workflow(tmp_path):
    rate_path, issue_path, context_path, values = inputs(tmp_path)
    output = tmp_path / "prepared.npz"
    receipt = adapter.prepare(rate_path, issue_path, context_path, output, purpose="select")
    prepared = load_prepared_npz(output, require_target=True, require_oof_auxiliary=True)
    assert prepared.dense_shape == (4, 2, 13)
    assert receipt["rain_rate_mean_recomputed"] is False
    shaped = prepared.patches.reshape(4, 2, 13, 6, 3, 3)
    native = adapter.rate_to_patches(np.load(rate_path)[0], values["grid_y"], values["grid_x"])
    np.testing.assert_array_equal(shaped[0, 0, 0], native[0, :6].astype(np.float32))
    np.testing.assert_array_equal(shaped[0, 0, -1], native[0, 12:18].astype(np.float32))
    aux = prepared.auxiliary.reshape(4, 2, 13, 39)
    np.testing.assert_array_equal(aux[0, 0, 0, 3:], values["context36"][0, 0])
    assert aux[0, 0, 0, 2] == pytest.approx(60/180)
    assert aux[0, 0, -1, 2] == 1
    np.testing.assert_array_equal(prepared.oof_auxiliary[2, 0, 3:], values["context36"][0, 0]+2000)
    with pytest.raises(FileExistsError):
        adapter.prepare(rate_path, issue_path, context_path, output)


def test_select_requires_fold_clean_context(tmp_path):
    paths = inputs(tmp_path, oof=False)[:3]
    with pytest.raises(ValueError, match="fold-clean"):
        adapter.prepare(*paths, tmp_path / "bad.npz", purpose="select")


def test_calibration_rejects_upstream_training_year(tmp_path):
    rate, issue, context, values = inputs(tmp_path)
    old = np.array(["2022-06-15", "2022-07-15", "2022-08-15", "2022-09-15"], dtype="datetime64[ns]").astype(np.int64)
    np.save(issue, old)
    with pytest.raises(ValueError, match="2023 only"):
        adapter.prepare(rate, issue, context, tmp_path / "bad.npz", purpose="fit")


def test_exact_context_time_axis_required(tmp_path):
    rate, issue, context, values = inputs(tmp_path)
    values["issue_times_ns"] = values["issue_times_ns"] + 600_000_000_000
    np.savez(context, **values)
    with pytest.raises(ValueError, match="axes differ"):
        adapter.prepare(rate, issue, context, tmp_path / "bad.npz")


@pytest.mark.parametrize("kind", ["negative", "boundary", "fractional"])
def test_invalid_fields_or_station_centres_rejected(kind):
    rate = np.ones((18, 256, 256), np.float32)
    y, x = np.array([20]), np.array([30])
    if kind == "negative": rate[0, 0, 0] = -1
    if kind == "boundary": x[0] = 0
    if kind == "fractional": x = np.array([30.5])
    with pytest.raises(ValueError):
        adapter.rate_to_patches(rate, y, x)
