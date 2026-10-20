import json

import numpy as np
import pandas as pd
import pytest

from basin_verification.core import area_mean, common_scores
from jhrs_verification.cli import run, load_manifest, aggregate_cells
from jhrs_verification.scores import (THRESHOLDS, METRICS, STEP_NS, common_support,
    daily_statistics, scores_from_sums, aggregate_scores, bootstrap_scores, result_tables)
from jhrs_verification.peaks import evaluate_peaks, peak_summary, peak_bootstrap


def fixture_arrays():
    times = pd.Timestamp("2024-06-01").value + np.arange(432, dtype=np.int64) * STEP_NS
    issues, leads = times[:-18], [60, 90, 120, 150, 180]
    obs = np.zeros((432, 2), np.float64)
    obs[140:145, 0] = [5, 10, 30, 20, 5]  # Crosses midnight without splitting.
    obs[290:294, 0] = [5, 20, 15, 5]
    obs[200:203, 1] = [5, 40, 5]
    aligned = np.stack([obs[np.arange(len(issues)) + lead // 10] for lead in leads], axis=1)
    delayed = np.stack([obs[np.arange(len(issues)) + lead // 10 - 1] for lead in leads], axis=1)
    preds = {f"route_{ri}": (delayed[:, None] + np.array([0., 1., 3.])[None, :, None, None] + ri * .1).astype(np.float32)
             for ri in range(4)}
    members = {r: [0, 1, 2] for r in preds}
    return times, issues, leads, obs, aligned, preds, members


def test_common_twelve_member_mask_and_three_threshold_scores():
    _, issues, _, _, obs, preds, _ = fixture_arrays()
    preds["route_3"][1, 2, 3, 0] = np.nan
    obs[2, 1, 1] = -1
    mask = common_support(preds, obs)
    assert not mask[1, 3, 0] and not mask[2, 1, 1]
    assert mask.sum() == mask.size - 2
    _, daily = daily_statistics(preds, obs, mask, issues)
    calculated = scores_from_sums(daily.sum(0))
    assert THRESHOLDS == (1., 5., 10.)
    assert daily.shape[-1] == 10 and calculated.shape[-1] == 6
    for li in range(obs.shape[1]):
        for si in range(obs.shape[2]):
            all_members = {f"{ri}:{mi}": p[:, mi, li, si] for ri, p in enumerate(preds.values()) for mi in range(3)}
            oracle, oracle_mask = common_scores(obs[:, li, si], all_members, thresholds=THRESHOLDS)
            np.testing.assert_array_equal(oracle_mask, mask[:, li, si])
            for ri in range(4):
                for mi in range(3):
                    rows = oracle.loc[oracle.series.eq(f"{ri}:{mi}")]
                    expected = rows.set_index("metric").loc[["rmse_mm", "mae_mm", "bias_mm"], "value"].tolist()
                    expected += rows.loc[rows.metric.eq("csi"), "value"].tolist()
                    np.testing.assert_allclose(calculated[ri, mi, li, si], expected, rtol=1e-14, equal_nan=True)


def test_member_scores_not_ensemble_mean_and_station_not_equal_site_mean():
    _, issues, _, _, obs, preds, _ = fixture_arrays()
    mask = common_support(preds, obs)
    _, daily = daily_statistics(preds, obs, mask, issues)
    total = daily.sum(0)
    site = scores_from_sums(total)
    np.testing.assert_allclose(aggregate_scores(total, "catchment"), site.mean(1).mean(2), equal_nan=True)
    station = aggregate_scores(total, "station")
    np.testing.assert_allclose(station, scores_from_sums(total.sum(3)).mean(1), equal_nan=True)
    assert not np.allclose(station, aggregate_scores(total, "catchment"), equal_nan=True)
    ensemble = {r: p.mean(axis=1, keepdims=True) for r, p in preds.items()}
    _, ensemble_daily = daily_statistics(ensemble, obs, mask, issues)
    assert not np.allclose(station, aggregate_scores(ensemble_daily.sum(0), "station"), equal_nan=True)


@pytest.mark.parametrize("kind", ["station", "catchment"])
def test_shared_bootstrap_counts_match_explicit_resampling(kind):
    _, issues, leads, _, obs, preds, members = fixture_arrays()
    _, daily = daily_statistics(preds, obs, common_support(preds, obs), issues)
    counts = np.array([[3, 0, 0], [1, 1, 1], [0, 2, 1]], dtype=np.int64)
    values, actual = bootstrap_scores(daily, kind, counts=counts, chunk=2)
    np.testing.assert_array_equal(actual, counts)
    expected = np.stack([aggregate_scores(daily[np.repeat(np.arange(3), c)].sum(0), kind) for c in counts])
    np.testing.assert_allclose(values, expected, equal_nan=True)
    tables = result_tables(daily, list(preds), members, leads, ["a", "b"], kind, values)
    csi = tables["route_scores"].query("metric == 'CSI'")
    assert set(csi.threshold_mm) == {1., 5., 10.}
    assert tables["paired_differences"].shape[0] == 6 * 5 * 7
    # A catchment with undefined CSI is not dropped from the equal-site mean.
    total = daily.sum(0)
    total[..., 1, 8:10] = 0
    assert np.isnan(aggregate_scores(total, "catchment")[..., -1]).all()


def test_strict_area_mean_no_renormalization():
    values = np.array([[2., 6., np.nan], [2., np.nan, 3.]])
    weights = np.array([[.25, .75, 0.], [0., .5, .5]])
    expected = area_mean(values, weights)
    np.testing.assert_allclose(aggregate_cells(values, weights, chunk=1), expected, equal_nan=True)
    assert expected[0, 0] == 5
    assert np.isnan(expected[0, 1]) and np.isnan(expected[1]).all()
    with pytest.raises(ValueError, match="complete-basin"):
        area_mean(values, weights * .5)


def test_fixed_lead_peak_tolerances_and_common_outer_halo():
    times, issues, leads, obs, _, preds, members = fixture_arrays()
    result = evaluate_peaks(times, obs, issues, leads, preds, members, ["a", "b"], years=[2024])
    errors = result["peak_event_errors"]
    assert errors.event_id.nunique() == 3
    event = errors.query("route == 'route_0' and member_id == '0' and lead_minutes == 60").iloc[:3]
    np.testing.assert_allclose(event.predicted_peak_mm, [10., 30., 30.])
    np.testing.assert_allclose(event.signed_error_mm, [-20., 0., 0.])
    assert event.n_window_forecasts.tolist() == [1, 3, 5]
    assert str(event.iloc[0].start_time)[:10] != str(event.iloc[0].end_time)[:10]
    # One invalid member at +20 removes that episode even from delta=0 scores.
    peak_ns = pd.Timestamp(event.iloc[0].observed_peak_time).value
    issue_pos = np.where(issues == peak_ns + 20 * STEP_NS // 10 - 180 * STEP_NS // 10)[0][0]
    preds["route_3"][issue_pos, 2, 4, 0] = np.nan
    second = evaluate_peaks(times, obs, issues, leads, preds, members, ["a", "b"], years=[2024])
    assert second["peak_event_errors"].event_id.nunique() == 2
    assert "forecast_values_invalid" in second["event_eligibility"].status.values


def test_peak_station_pool_and_equal_catchments_and_bootstrap():
    times, issues, leads, obs, _, preds, members = fixture_arrays()
    errors = evaluate_peaks(times, obs, issues, leads, preds, members, ["a", "b"], years=[2024])["peak_event_errors"]
    for kind in ("station", "catchment"):
        tables = peak_summary(errors, list(preds), members, leads, ["a", "b"], kind)
        counts = np.array([[1, 1, 1], [3, 0, 0], [0, 3, 0]])
        ci, actual = peak_bootstrap(errors, list(preds), members, leads, times, kind, draws=3,
                                   counts=counts, minimum_valid=1, years=[2024])
        np.testing.assert_array_equal(actual, counts)
        merged = tables["peak_route_scores"].merge(ci["peak_route_intervals"], on=["route", "lead_minutes", "tolerance_minutes"])
        np.testing.assert_allclose(merged.rmse_mm_x, merged.rmse_mm_y)
        assert set(ci["peak_route_intervals"].n_defined_resamples) == ({1} if kind == "catchment" else {3})


def write_manifest(tmp_path):
    times, issues, leads, obs, _, preds, members = fixture_arrays()
    np.save(tmp_path / "issues.npy", issues)
    np.save(tmp_path / "times.npy", times)
    np.save(tmp_path / "obs.npy", obs)
    routes = []
    for name, array in preds.items():
        np.save(tmp_path / f"{name}.npy", array)
        routes.append(dict(name=name, member_ids=members[name], predictions=f"{name}.npy"))
    manifest = dict(schema="jhrs-verification-v1", kind="catchment", timezone="Asia/Seoul",
        timestamp_encoding="naive_kst_nanoseconds", issues="issues.npy", valid_times="times.npy", observations="obs.npy",
        lead_minutes=leads, site_ids=["a", "b"], expected_site_count=2, routes=routes, years=[2024], months=[6, 7, 8, 9])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_manifest_cli_complete_and_immutable_inputs(tmp_path):
    path = write_manifest(tmp_path)
    data = load_manifest(path)
    output = tmp_path / "result"
    report = run(path, output, draws=8, seed=42, minimum_peak_resamples=1)
    assert report["completed"] and report["n_routes"] == 4
    assert report["thresholds_mm"] == [1., 5., 10.]
    assert report["input_sha256"] == data["input_sha256"]
    assert (output / "peak_paired_intervals.csv").exists()
    assert report == json.loads((output / "COMPLETE.json").read_text())
    assert pd.read_csv(output / "peak_event_errors.csv").event_id.nunique() == 3
    with pytest.raises(FileExistsError):
        run(path, output, draws=8)


def test_manifest_rejects_wrong_axes_and_hash(tmp_path):
    path = write_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["sha256"] = {"obs.npy": "bad"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(path)


def test_dry_only_period_has_no_fabricated_events_or_zero_rmse(tmp_path):
    path = write_manifest(tmp_path)
    np.save(tmp_path / "obs.npy", np.zeros((432, 2), np.float32))
    report = run(path, tmp_path / "dry", draws=4, minimum_peak_resamples=1)
    assert report["completed"]
    errors = pd.read_csv(tmp_path / "dry/peak_event_errors.csv")
    scores = pd.read_csv(tmp_path / "dry/peak_route_scores.csv")
    intervals = pd.read_csv(tmp_path / "dry/peak_route_intervals.csv")
    assert errors.empty
    assert scores.rmse_mm.isna().all()
    assert scores.n_events.eq(0).all()
    assert intervals.ci_status.eq("no_events").all()
