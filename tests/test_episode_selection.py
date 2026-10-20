"""Tests of the unchanged observation-defined event selector."""
import numpy as np
import pandas as pd
import pytest

from evaluation.episodes import EpisodeConfig, select_episodes


def catalog(values):
    times = pd.date_range("2024-06-01 23:40", periods=len(values), freq="10min").asi8
    return select_episodes(times, np.asarray(values, float)[:, None], [7], EpisodeConfig())


def test_cross_midnight_run_and_earliest_peak_tie():
    row = catalog([0, 5, 25, 25, 5, 0]).iloc[0]
    assert row.status == "candidate" and row.crosses_midnight
    assert row.duration_min == 40 and row.observed_max_tie_count == 2
    assert row.observed_peak_time == pd.Timestamp("2024-06-02 00:00")


@pytest.mark.parametrize("missing", [np.nan, -1.])
def test_missing_observation_censors_both_fragments(missing):
    rows = catalog([0, 25, missing, 25, 0])
    assert rows.status.tolist() == ["observation_censored"] * 2


def test_thresholds_and_whole_run_qc():
    rows = catalog([0, 5, 19.9, 0, 20, 0, 300, 5, 0])
    assert rows.status.tolist() == ["below_peak_threshold", "candidate", "qc_high_observation"]


def test_evaluation_boundary_is_not_dry():
    assert catalog([25, 5, 0]).iloc[0].status == "observation_censored"


def test_temporal_gap_does_not_join_runs():
    times = pd.date_range("2024-06-01", periods=5, freq="10min").asi8
    times[3:] += 600_000_000_000
    rows = select_episodes(times, np.array([0, 20, 25, 20, 0])[:, None], [7], EpisodeConfig())
    assert rows.status.tolist() == ["observation_censored"] * 2
