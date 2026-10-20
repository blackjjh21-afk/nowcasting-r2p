from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catchment_kriging import (
    ExponentialVariogram, OrdinaryKriging, fit_variogram_from_arrays,
)


XY = np.asarray([[0, 0], [0, 10], [10, 0], [10, 10]], dtype=float)
MODEL = ExponentialVariogram(0.2, 0.8, 7.0)


def solver(target=None, xy=XY, **kwargs):
    return OrdinaryKriging(xy, np.asarray([[5, 5]]) if target is None else target, MODEL, **kwargs)


def test_exponential_scale_and_effective_range_and_nugget():
    assert MODEL.semivariance(0) == 0
    assert MODEL.semivariance(21) == pytest.approx(0.2 + 0.8 * (1 - np.exp(-3)))
    assert MODEL.as_dict()["effective_range_km"] == 21
    assert MODEL.covariance(0) == pytest.approx(1)
    with pytest.raises(ValueError):
        ExponentialVariogram(-0.1, 1, 2)


def test_ok_reproduces_constant_and_observations_with_unit_weight_constraint():
    engine = solver(np.vstack([XY, [[5, 5]]]))
    prediction = engine.predict(np.asarray([2, 4, 6, 8]))
    np.testing.assert_allclose(prediction["rainfall_mm"][:4], [2, 4, 6, 8], atol=1e-10)
    np.testing.assert_allclose(prediction["kriging_variance_mm2"][:4], 0, atol=1e-10)
    assert prediction["rainfall_mm"][-1] == pytest.approx(5)
    weights = next(iter(engine._cache.values()))[0]
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
    constant = engine.predict(np.full(4, 3.0))
    np.testing.assert_allclose(constant["rainfall_mm"], 3)
    np.testing.assert_array_equal(constant["kriging_variance_mm2"], 0)
    assert len(engine._cache) == 1


def test_missing_not_dry_and_negative_gauge_is_excluded():
    engine = solver()
    dry = engine.predict(np.zeros(4))
    assert dry["rainfall_mm"][0] == 0 and dry["valid_mask"][0]
    absent = engine.predict(np.full(4, np.nan))
    assert not absent["valid_mask"].any()
    assert np.isnan(absent["rainfall_mm"]).all()
    negative = engine.predict([1, 1, -999, np.nan])
    assert negative["n_gauges"] == 2
    assert negative["status"] == "insufficient_gauges"


def test_variance_scaled_by_same_time_station_variance():
    engine = solver()
    a, b = engine.predict([0, 1, 2, 4]), engine.predict([0, 2, 4, 8])
    np.testing.assert_allclose(b["rainfall_mm"], 2 * a["rainfall_mm"])
    np.testing.assert_allclose(b["kriging_variance_mm2"], 4 * a["kriging_variance_mm2"])
    assert a["kriging_variance_mm2"][0] > 0


def test_support_hull_radius_and_nearest_threshold():
    locations = np.asarray([[5, 5], [-1, 5], [50, 50]])
    result = solver(locations).predict([0, 1, 2, 3])
    np.testing.assert_array_equal(result["valid_mask"], [True, False, False])
    assert np.isnan(result["rainfall_mm"][1:]).all()
    assert not solver(max_nearest_km=2).predict([1, 2, 3, 4])["valid_mask"].any()
    assert not solver(support_radius_km=2).predict([1, 2, 3, 4])["valid_mask"].any()
    collinear = np.asarray([[0, 0], [1, 0], [2, 0]])
    assert not solver([[1, 0]], xy=collinear).predict([1, 2, 3])["valid_mask"].any()


def test_duplicates_pool_only_finite_observations_and_cache_is_bounded():
    xy = np.vstack([XY, XY[0]])
    engine = solver(xy=xy, max_cache_entries=1)
    actual = engine.predict([0, 2, 4, 6, 2])
    expected = solver().predict([1, 2, 4, 6])
    np.testing.assert_allclose(actual["rainfall_mm"], expected["rainfall_mm"])
    assert actual["n_gauges"] == 4 and actual["n_input_gauges_finite"] == 5
    engine.predict([np.nan, 2, 4, 6, np.nan])
    assert len(engine._cache) == 1
    engine.close()
    assert not engine._cache


def test_solver_fails_explicitly_when_condition_limit_cannot_be_met():
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        solver(condition_limit=1.01).predict([0, 1, 2, 3])


def test_negative_prediction_policy_is_an_explicit_extension():
    # Positive observations can yield negative OK estimates with extrapolation.
    xy = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    candidates = np.asarray([[a, b] for a in np.linspace(-1, 2, 9) for b in np.linspace(-1, 2, 9)])
    raw = OrdinaryKriging(xy, candidates, ExponentialVariogram(0, 1, 10), require_hull=False, negative_policy="retain")
    clipped = OrdinaryKriging(xy, candidates, ExponentialVariogram(0, 1, 10), require_hull=False)
    a, b = raw.predict([10, 0, 0, 0]), clipped.predict([10, 0, 0, 0])
    assert (a["rainfall_mm"] < 0).any()
    assert b["n_negative_clipped"] > 0
    np.testing.assert_allclose(b["rainfall_mm"], np.maximum(a["rainfall_mm"], 0))
    np.testing.assert_allclose(a["kriging_variance_mm2"], b["kriging_variance_mm2"])


def test_seasonal_variogram_is_normalized_per_timestep_without_log_transform():
    rng = np.random.default_rng(42)
    xy = rng.uniform(0, 60, (24, 2))
    values = rng.gamma(1.5, 2.0, (18, 24))
    values[2, :3] = np.nan
    fit = fit_variogram_from_arrays(xy, values, cutoff_km=65, nlags=9)
    # An independent amplitude factor at EVERY timestep must cancel exactly.
    scaled = fit_variogram_from_arrays(xy, values * np.arange(1, 19)[:, None], cutoff_km=65, nlags=9)
    np.testing.assert_allclose(fit["fit"]["empirical_normalized_gamma"], scaled["fit"]["empirical_normalized_gamma"], atol=1e-12)
    assert fit["fit"]["n_wet_variable_timesteps"] == 18
    assert fit["fit"]["variance_ddof"] == 1
    assert fit["variogram"]["scale_km"] > 0
    with pytest.raises(ValueError, match="populated lag"):
        fit_variogram_from_arrays(xy, np.zeros_like(values), cutoff_km=65, nlags=9)


