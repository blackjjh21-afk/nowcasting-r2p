"""Shared point-verification support and undefined-score regression tests."""
import warnings

import numpy as np
import pytest

from common.bootstrap import paired_date_block_csi_difference
from common.metrics import contingency_counts, csi_by_lead


DATES = np.array(["2024-06-01T00:00", "2024-06-02T00:00"], dtype="datetime64[ns]")


def bootstrap(a, b, truth, *, thresholds=(1,), **kwargs):
    return paired_date_block_csi_difference(
        a, b, truth, DATES, thresholds, n_resamples=1000, seed=17, **kwargs
    )


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("missing_route", ["a", "b"])
def test_bootstrap_uses_common_finite_support(missing, missing_route):
    truth = np.ones((2, 2, 1))
    a, b = truth.copy(), truth.copy()
    a[:, 1, :] = missing
    b[:, 1, :] = 0.0
    if missing_route == "b":
        a, b = b, a
    original_a, original_b, original_truth = a.copy(), b.copy(), truth.copy()
    result = bootstrap(a, b, truth)
    for key in ("delta_csi", "ci_low", "ci_high", "bootstrap_delta"):
        np.testing.assert_array_equal(result[key], np.zeros_like(result[key]))
    np.testing.assert_array_equal(result["n_valid_resamples"], [[1000]])
    for actual, expected in ((a, original_a), (b, original_b), (truth, original_truth)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_bootstrap_excludes_nonfinite_truth(missing):
    truth = np.array([1.0, missing, 1.0, missing]).reshape(2, 2, 1)
    a = np.ones_like(truth)
    b = a.copy()
    b[:, 1, :] = 0.0
    result = bootstrap(a, b, truth)
    np.testing.assert_array_equal(result["delta_csi"], [[0.0]])
    np.testing.assert_array_equal(result["n_valid_resamples"], [[1000]])


@pytest.mark.parametrize("value", [0.0, np.nan])
def test_empty_unions_are_undefined_without_runtime_warnings(value):
    values = np.full((2, 2, 2), value)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        score = csi_by_lead(values, values, [1, 5])
        counts = contingency_counts(values, values, 1)
        result = bootstrap(values, values, values, thresholds=(1, 5))
    assert np.isnan(score).all() and np.isnan(counts.csi)
    for key in ("delta_csi", "ci_low", "ci_high", "bootstrap_delta"):
        assert np.isnan(result[key]).all()
    np.testing.assert_array_equal(result["n_valid_resamples"], np.zeros((2, 2), dtype=int))


def test_one_undefined_route_does_not_become_zero_difference():
    truth = np.zeros((2, 1, 1))
    result = bootstrap(np.ones_like(truth), truth.copy(), truth)
    assert np.isnan(result["delta_csi"]).all()
    assert np.isnan(result["ci_low"]).all() and np.isnan(result["ci_high"]).all()
    np.testing.assert_array_equal(result["n_valid_resamples"], [[0]])


def test_undefined_resamples_excluded_per_cell_not_replaced_by_zero():
    truth = np.array([[[1.0, 1.0]], [[0.0, 1.0]]])
    a, b = truth.copy(), np.zeros_like(truth)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = bootstrap(a, b, truth, thresholds=(1, 5))
    weights = np.random.default_rng(17).multinomial(2, [0.5, 0.5], size=1000)
    wet_draws = int((weights[:, 0] > 0).sum())
    np.testing.assert_array_equal(result["n_valid_resamples"], [[wet_draws, 0], [1000, 0]])
    assert 0 < wet_draws < 1000
    np.testing.assert_array_equal(result["delta_csi"][:, 0], [1.0, 1.0])
    np.testing.assert_array_equal(result["ci_low"][:, 0], [1.0, 1.0])
    np.testing.assert_array_equal(result["ci_high"][:, 0], [1.0, 1.0])
    assert np.isnan(result["bootstrap_delta"][weights[:, 0] == 0, 0, 0]).all()
    assert np.isnan(result["ci_low"][:, 1]).all()
    assert np.isnan(result["ci_high"][:, 1]).all()


def test_date_blocks_remain_paired_and_seed_is_reproducible():
    times = np.array(["2024-06-01T00:00", "2024-06-01T00:10", "2024-06-02T00:00"],
                     dtype="datetime64[ns]")
    truth = np.ones((3, 1, 1))
    a = np.array([1.0, 1.0, 0.0]).reshape(3, 1, 1)
    b = 1.0 - a
    options = {"n_resamples": 200, "seed": 31, "confidence": 0.9}
    first = paired_date_block_csi_difference(a, b, truth, times, [1], **options)
    second = paired_date_block_csi_difference(a, b, truth, times, [1], **options)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    weights = np.random.default_rng(31).multinomial(2, [0.5, 0.5], size=200)
    expected = (2 * weights[:, 0] - weights[:, 1]) / (2 * weights[:, 0] + weights[:, 1])
    np.testing.assert_allclose(first["bootstrap_delta"][:, 0, 0], expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(first["delta_csi"], [[1 / 3]])
    np.testing.assert_allclose([first["ci_low"].item(), first["ci_high"].item()],
                               np.quantile(expected, [0.05, 0.95]))
    np.testing.assert_array_equal(first["n_valid_resamples"], [[200]])
    assert first["block"] == "issuance_date" and first["n_resamples"] == 200


@pytest.mark.parametrize("thresholds", [[np.nan], [np.inf], [-np.inf], [1, np.nan]])
def test_public_csi_helpers_reject_nonfinite_thresholds(thresholds):
    values = np.ones((2, 1, 1))
    with pytest.raises(ValueError, match="thresholds must be finite"):
        bootstrap(values, values, values, thresholds=thresholds)
    with pytest.raises(ValueError, match="thresholds must be finite"):
        csi_by_lead(values, values, thresholds)


@pytest.mark.parametrize("thresholds", [[], [[1, 5]]])
def test_public_csi_helpers_reject_invalid_threshold_shape(thresholds):
    values = np.ones((2, 1, 1))
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        bootstrap(values, values, values, thresholds=thresholds)
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        csi_by_lead(values, values, thresholds)
