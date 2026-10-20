"""Small, route-agnostic verification primitives.

Arrays may come from a direct point model or from any field-first readout.  The
functions deliberately know nothing about radar resolution, frame spacing, or
station split.  Callers are responsible for aligning route predictions to the
same issuance times, stations, leads, and truth before invoking them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Contingency:
    """Event-contingency counts with the standard CSI definition."""

    hit: np.ndarray
    miss: np.ndarray
    false_alarm: np.ndarray
    correct_negative: np.ndarray

    @property
    def csi(self) -> np.ndarray:
        """Return CSI, with NaN for an empty event union."""
        denominator = self.hit + self.miss + self.false_alarm
        return np.divide(
            self.hit,
            denominator,
            out=np.full_like(denominator, np.nan, dtype=np.float64),
            where=denominator > 0,
        )


def _paired_finite(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape:
        raise ValueError(
            f"prediction/truth shape mismatch: {prediction.shape} != {truth.shape}"
        )
    finite = np.isfinite(prediction) & np.isfinite(truth)
    return prediction, truth, finite


def contingency_counts(
    prediction: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Contingency:
    """Return paired-finite event counts at ``prediction/truth >= threshold``."""

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    prediction, truth, finite = _paired_finite(prediction, truth)
    predicted_event = finite & (prediction >= threshold)
    observed_event = finite & (truth >= threshold)
    return Contingency(
        hit=np.sum(predicted_event & observed_event, axis=axis, dtype=np.int64),
        miss=np.sum(~predicted_event & observed_event, axis=axis, dtype=np.int64),
        false_alarm=np.sum(predicted_event & ~observed_event, axis=axis, dtype=np.int64),
        correct_negative=np.sum(
            finite & ~predicted_event & ~observed_event,
            axis=axis,
            dtype=np.int64,
        ),
    )


def csi_by_lead(
    prediction: np.ndarray,
    truth: np.ndarray,
    thresholds: np.ndarray | list[float] | tuple[float, ...],
    *,
    lead_axis: int = -1,
) -> np.ndarray:
    """Compute CSI with output shape ``[lead, threshold]``.

    Every non-lead dimension is pooled into the contingency counts. Empty
    event unions are undefined and return NaN, including cells with no finite
    prediction/truth pairs. Other evaluation conventions must be applied
    explicitly by their callers.
    """

    prediction, truth, _ = _paired_finite(prediction, truth)
    prediction = np.moveaxis(prediction, lead_axis, -1)
    truth = np.moveaxis(truth, lead_axis, -1)
    thresholds_array = np.asarray(thresholds, dtype=np.float64)
    if thresholds_array.ndim != 1 or len(thresholds_array) == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence")
    if not np.isfinite(thresholds_array).all():
        raise ValueError("thresholds must be finite")
    result = np.empty((prediction.shape[-1], len(thresholds_array)), dtype=np.float64)
    reduce_axes = tuple(range(prediction.ndim - 1))
    for threshold_index, threshold in enumerate(thresholds_array):
        counts = contingency_counts(
            prediction,
            truth,
            float(threshold),
            axis=reduce_axes,
        )
        result[:, threshold_index] = counts.csi
    return result


def rmse_and_bias(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float, int]:
    """Return paired-finite RMSE, signed bias, and support."""

    prediction, truth, finite = _paired_finite(prediction, truth)
    support = int(finite.sum())
    if support == 0:
        return float("nan"), float("nan"), 0
    error = prediction[finite] - truth[finite]
    return float(np.sqrt(np.mean(error**2))), float(np.mean(error)), support
