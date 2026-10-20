"""Strict basin aggregation and common-sample rainfall scores."""
from __future__ import annotations

import numpy as np
import pandas as pd


def area_mean(values, weights, *, tolerance=1e-5):
    """Integrate [...,cell] against [basin,cell] complete-basin area fractions.

    No renormalization over missing cells. Any nonfinite/negative cell with
    positive weight makes that basin/time invalid. Tiny geometry closure error
    is retained, not silently corrected. Zero-weight missing cells are ignored.
    """
    values, weights = np.asarray(values), np.asarray(weights, dtype=float)
    if (weights.ndim != 2 or values.ndim < 1 or values.shape[-1] != weights.shape[1]
            or not weights.shape[0] or not weights.shape[1]
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=1), 1., rtol=0, atol=tolerance)):
        raise ValueError("Require nonnegative, complete-basin weights and matching cell axis")
    good = np.isfinite(values) & (values >= 0)
    result = np.where(good, values, 0).astype(float) @ weights.T
    missing = (~good).astype(np.int32) @ (weights > 0).T.astype(np.int32)
    return np.where(missing == 0, result, np.nan)


def common_scores(reference, forecasts, thresholds=(1., 5., 10., 20.)):
    """All routes/members use the SAME finite pairs, per basin/lead upstream.

    Arrays here are one-dimensional time series. No member-mean forecast is
    substituted for individual member scores. Undefined CSI remains NaN.
    """
    ref = np.asarray(reference, dtype=float)
    if ref.ndim != 1 or not forecasts:
        raise ValueError("Nonempty forecast mapping and 1-D reference required")
    arrays = {k: np.asarray(v, dtype=float) for k, v in forecasts.items()}
    if any(v.shape != ref.shape for v in arrays.values()):
        raise ValueError("Forecast and reference axes differ")
    common = np.isfinite(ref) & (ref >= 0)
    for value in arrays.values():
        common &= np.isfinite(value) & (value >= 0)
    obs = ref[common]
    rows = []
    for name, value in arrays.items():
        pred = value[common]
        err = pred - obs
        rows.append(dict(series=name, metric="mae_mm", threshold_mm=np.nan,
                         value=float(np.mean(abs(err))) if len(err) else np.nan,
                         n_pairs=len(err)))
        rows.append(dict(series=name, metric="rmse_mm", threshold_mm=np.nan,
                         value=float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
                         n_pairs=len(err)))
        rows.append(dict(series=name, metric="bias_mm", threshold_mm=np.nan,
                         value=float(np.mean(err)) if len(err) else np.nan,
                         n_pairs=len(err)))
        for threshold in thresholds:
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("Thresholds must be finite and positive")
            observed, predicted = obs >= threshold, pred >= threshold
            hit = int((observed & predicted).sum())
            miss = int((observed & ~predicted).sum())
            false = int((~observed & predicted).sum())
            rows.append(dict(series=name, metric="csi", threshold_mm=threshold,
                             value=hit / (hit + miss + false) if hit + miss + false else np.nan,
                             n_pairs=len(err), hits=hit, misses=miss, false_alarms=false))
    return pd.DataFrame(rows), common


def match_reference(reference_times, reference, issuance_times, lead_minutes):
    """Exact matching on naive KST nanoseconds; NEVER nearest-time matching."""
    times, issues = np.asarray(reference_times), np.asarray(issuance_times)
    ref = np.asarray(reference)
    for axis in (times, issues):
        if (axis.ndim != 1 or axis.dtype.kind not in "iu" or not len(axis)
                or np.any(np.diff(axis) <= 0) or np.any(axis % (600 * 10**9))):
            raise ValueError("Unique increasing exact 10-min KST integer axes required")
    if (ref.shape[0] != len(times) or isinstance(lead_minutes, bool)
            or not isinstance(lead_minutes, (int, np.integer)) or lead_minutes < 0
            or lead_minutes % 10):
        raise ValueError("Invalid reference or lead")
    query = issues + int(lead_minutes) * 60 * 10**9
    ix = np.searchsorted(times, query)
    if np.any(ix >= len(times)) or not np.array_equal(times[ix], query):
        raise ValueError("Reference lacks exact forecast valid times")
    return ref[ix]
