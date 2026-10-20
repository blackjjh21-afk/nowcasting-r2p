"""Paired issuance-date block bootstrap for route-level CSI differences."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_arrays(
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    truth: np.ndarray,
    issue_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction_a = np.asarray(prediction_a, dtype=np.float64)
    prediction_b = np.asarray(prediction_b, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction_a.shape != prediction_b.shape or prediction_a.shape != truth.shape:
        raise ValueError(
            "paired routes and truth must have identical [issue, station, lead] shapes"
        )
    if prediction_a.ndim != 3:
        raise ValueError("expected [issue, station, lead] arrays")
    issue_times = np.asarray(issue_times).astype("datetime64[ns]")
    if issue_times.shape != (prediction_a.shape[0],):
        raise ValueError("issue_times must have one entry per issue-time row")
    if np.isnat(issue_times).any():
        raise ValueError("issue_times contain NaT")
    return prediction_a, prediction_b, truth, issue_times


def _daily_counts(
    prediction: np.ndarray,
    truth: np.ndarray,
    day_index: np.ndarray,
    n_days: int,
    thresholds: np.ndarray,
    finite: np.ndarray,
) -> np.ndarray:
    n_leads = prediction.shape[-1]
    counts = np.zeros((n_days, n_leads, len(thresholds), 3), dtype=np.int64)
    for threshold_index, threshold in enumerate(thresholds):
        predicted_event = finite & (prediction >= threshold)
        observed_event = finite & (truth >= threshold)
        cells = (
            predicted_event & observed_event,
            ~predicted_event & observed_event,
            predicted_event & ~observed_event,
        )
        for day in range(n_days):
            selected = day_index == day
            for count_index, cell in enumerate(cells):
                counts[day, :, threshold_index, count_index] = cell[selected].sum(
                    axis=(0, 1), dtype=np.int64
                )
    return counts


def _csi_from_counts(counts: np.ndarray) -> np.ndarray:
    denominator = counts[..., 0] + counts[..., 1] + counts[..., 2]
    return np.divide(
        counts[..., 0],
        denominator,
        out=np.full_like(denominator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def paired_date_block_csi_difference(
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    truth: np.ndarray,
    issue_times: np.ndarray,
    thresholds: np.ndarray | list[float] | tuple[float, ...],
    *,
    n_resamples: int = 5_000,
    seed: int = 20260713,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Estimate ``CSI(route A) - CSI(route B)`` with paired date blocks.

    Counts use the intersection of finite route-A, route-B and truth values
    for each issue/station/lead cell. The same multinomial date weights are
    applied to both routes in every replicate, retaining all stations and
    leads within each date block.

    CSI is undefined when its event-union denominator is zero. A bootstrap
    difference is included in the percentile interval only when both route
    scores are finite, separately for each lead/threshold cell. The returned
    ``n_valid_resamples`` records that effective sample size; cells with no
    valid differences have NaN interval bounds. Undefined draws remain NaN
    in ``bootstrap_delta``. Inputs must already use the intended local
    issuance-date convention.
    """

    prediction_a, prediction_b, truth, issue_times = _validate_arrays(
        prediction_a, prediction_b, truth, issue_times
    )
    thresholds_array = np.asarray(thresholds, dtype=np.float64)
    if thresholds_array.ndim != 1 or len(thresholds_array) == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence")
    if not np.isfinite(thresholds_array).all():
        raise ValueError("thresholds must be finite")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie strictly between zero and one")

    days, day_index = np.unique(issue_times.astype("datetime64[D]"), return_inverse=True)
    if len(days) < 2:
        raise ValueError("date-block bootstrap requires at least two issuance dates")
    common_finite = np.isfinite(prediction_a) & np.isfinite(prediction_b) & np.isfinite(truth)
    counts_a = _daily_counts(
        prediction_a, truth, day_index, len(days), thresholds_array, common_finite
    )
    counts_b = _daily_counts(
        prediction_b, truth, day_index, len(days), thresholds_array, common_finite
    )

    point_a = _csi_from_counts(counts_a.sum(axis=0))
    point_b = _csi_from_counts(counts_b.sum(axis=0))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(days),
        np.full(len(days), 1.0 / len(days), dtype=np.float64),
        size=n_resamples,
    )
    bootstrap_a = _csi_from_counts(np.einsum("rd,dltc->rltc", weights, counts_a))
    bootstrap_b = _csi_from_counts(np.einsum("rd,dltc->rltc", weights, counts_b))
    bootstrap_delta = bootstrap_a - bootstrap_b
    alpha = (1.0 - confidence) / 2.0
    valid = np.isfinite(bootstrap_delta)
    n_valid_resamples = valid.sum(axis=0)
    ci_low = np.full(point_a.shape, np.nan, dtype=np.float64)
    ci_high = np.full(point_a.shape, np.nan, dtype=np.float64)
    for lead, threshold in np.ndindex(point_a.shape):
        values = bootstrap_delta[valid[:, lead, threshold], lead, threshold]
        if values.size:
            ci_low[lead, threshold], ci_high[lead, threshold] = np.quantile(
                values, (alpha, 1.0 - alpha)
            )
    return {
        "delta_csi": point_a - point_b,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_delta": bootstrap_delta,
        "n_valid_resamples": n_valid_resamples,
        "dates": days,
        "thresholds": thresholds_array,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "block": "issuance_date",
    }
