"""Observation-defined events shared by the current peak-RMSE diagnostic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

MINUTE_NS = 60_000_000_000


class DiagnosticError(ValueError):
    """Invalid observed time series or event selection inputs."""


@dataclass(frozen=True)
class EpisodeConfig:
    selection_step_minutes: int = 10
    episode_threshold_mm: float = 5.0
    threshold_mm: float = 20.0
    qc_upper_mm: float = 300.0


def select_episodes(times_ns: np.ndarray, truth: np.ndarray, station_ids: Sequence[int],
                    config: EpisodeConfig) -> pd.DataFrame:
    """Keep all wet runs in an audit, including censored/below-threshold/QC runs.

    No interpolation or bridging missing values. A wet fragment touching a gap
    is censored and cannot be promoted into an independently bounded episode.
    """
    times = np.asarray(times_ns, dtype=np.int64)
    values = np.asarray(truth, dtype=np.float64)
    stations = np.asarray(station_ids, dtype=np.int64)
    step = config.selection_step_minutes * MINUTE_NS
    if (values.shape != (len(times), len(stations)) or not len(times)
            or np.any(np.diff(times) <= 0) or np.any(times % step)
            or len(np.unique(stations)) != len(stations)):
        raise DiagnosticError("Invalid observation/station/time axes.")
    contiguous = np.diff(times) == step
    records = []
    for col, station in enumerate(stations):
        observed = values[:, col]
        valid = np.isfinite(observed) & (observed >= 0)
        wet = valid & (observed >= config.episode_threshold_mm)
        starts = np.flatnonzero(wet & ~np.r_[False, wet[:-1] & contiguous])
        ends = np.flatnonzero(wet & ~np.r_[wet[1:] & contiguous, False])
        for start, end in zip(starts, ends):
            series = observed[start:end + 1]
            max_mm = float(series.max())
            peak = start + int(series.argmax())  # earliest exact tie
            tie_positions = np.flatnonzero(series == max_mm) + start
            left = "observed_dry"
            right = "observed_dry"
            if start == 0 or not contiguous[start - 1]:
                left = "evaluation_boundary"
            elif not valid[start - 1]:
                left = "missing_or_invalid_observation"
            if end == len(times) - 1 or not contiguous[end]:
                right = "evaluation_boundary"
            elif not valid[end + 1]:
                right = "missing_or_invalid_observation"
            qc_count = int((series >= config.qc_upper_mm).sum())
            complete = left == right == "observed_dry"
            status = ("qc_high_observation" if qc_count else "observation_censored" if not complete
                      else "below_peak_threshold" if max_mm < config.threshold_mm else "candidate")
            start_time, end_time, peak_time = pd.to_datetime(times[[start, end, peak]])
            records.append({
                "event_id": f"{station}:{start_time:%Y%m%dT%H%M}", "station_id": int(station),
                "start_index": int(start), "end_index": int(end), "station_index": col,
                "start_time": start_time, "end_time": end_time,
                "observed_peak_time": peak_time, "observed_max_mm": max_mm,
                "valid_date": peak_time.normalize(), "year": peak_time.year,
                "month": peak_time.month, "n_slots": len(series),
                "duration_min": len(series) * config.selection_step_minutes,
                "first_to_last_span_min": (times[end] - times[start]) / MINUTE_NS,
                "crosses_midnight": start_time.normalize() != end_time.normalize(),
                "observed_max_tie_count": len(tie_positions),
                "observed_max_tie_span_min": (times[tie_positions[-1]] - times[tie_positions[0]]) / MINUTE_NS,
                "left_boundary": left, "right_boundary": right, "boundaries_complete": complete,
                "qc_high_observation_count": qc_count, "observation_status": status, "status": status,
            })
    if not records:
        raise DiagnosticError("No observed wet runs were found.")
    return pd.DataFrame(records).sort_values(["start_time", "station_id"]).reset_index(drop=True)


