"""Common observation-defined events and fixed-lead 0/10/20-minute peaks."""
from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace
import numpy as np
import pandas as pd

from evaluation.episodes import select_episodes
from .scores import MINUTE_NS, STEP_NS, time_axis, validate_predictions, date_counts, interval

TOLERANCES = (0, 10, 20)
DAY_NS = 1440 * MINUTE_NS


def season_mask(times, years=(2024, 2025), months=(6, 7, 8, 9)):
    dates = pd.DatetimeIndex(np.asarray(times).astype("datetime64[ns]"))
    return np.isin(dates.year, years) & np.isin(dates.month, months)


def _lookup(axis, wanted):
    pos = np.searchsorted(axis, wanted)
    safe = np.minimum(pos, len(axis) - 1)
    return pos, (pos < len(axis)) & (axis[safe] == wanted)


def evaluate_peaks(valid_times, observed, issues, leads, predictions, member_ids, site_ids,
                   *, years=(2024, 2025), months=(6, 7, 8, 9)):
    """Select events from observations; intersect full episode+20-minute halos.

    Predictions are [issue,member,lead,site]. At a fixed lead, different
    valid times come from successive issuances. The ±window is a maximum,
    never a sum of overlapping RN60 accumulations. Tied maxima use the first
    time. Events touching missing data, evaluation gaps or boundaries fail
    the inherited dry-bounded >=5 mm run / >=20 mm maximum definition.
    """
    times, issues = time_axis(valid_times), time_axis(issues, "issues")
    observed, leads = np.asarray(observed), np.asarray(leads)
    if observed.shape != (len(times), len(site_ids)):
        raise ValueError("Valid-time observations and site axes differ")
    if leads.ndim != 1 or leads.dtype.kind not in "iu" or not len(leads) or np.any(leads <= 0) or np.any(leads % 10) or len(np.unique(leads)) != len(leads):
        raise ValueError("Distinct positive ten-minute leads are required")
    n_members = validate_predictions(predictions, np.broadcast_to(0., (len(issues), len(leads), len(site_ids))))
    for route in predictions:
        if len(member_ids[route]) != n_members or len(set(member_ids[route])) != n_members:
            raise ValueError("Member identities must match the forecast member axis")
    obs = np.asarray(observed, dtype=float).copy()
    obs[~season_mask(times, years, months)] = np.nan
    config = SimpleNamespace(selection_step_minutes=10, episode_threshold_mm=5., threshold_mm=20., qc_upper_mm=300.)
    columns = ["event_id", "site_id", "site_index", "start_index", "end_index", "start_time", "end_time",
               "observed_peak_time", "observed_max_mm", "duration_min", "year", "observation_status"]
    if np.any(np.isfinite(obs) & (obs >= 5)):
        catalog = select_episodes(times, obs, np.arange(len(site_ids)), config)
        catalog = catalog.rename(columns={"station_index": "site_index"}).drop(columns=["station_id", "status"])
        catalog["site_id"] = [str(site_ids[int(i)]) for i in catalog.site_index]
        catalog["event_id"] = [f"{r.site_id}:{pd.Timestamp(r.start_time):%Y%m%dT%H%M}" for r in catalog.itertuples()]
    else:
        catalog = pd.DataFrame(columns=columns)
    eligibility = catalog[["event_id", "site_id", "year", "observation_status"]].copy()
    eligibility["status"] = eligibility.observation_status
    eligibility["included"] = False
    rows = []
    for index, event in catalog.loc[catalog.observation_status.eq("candidate")].iterrows():
        site = int(event.site_index)
        episode = times[int(event.start_index):int(event.end_index) + 1]
        peak = pd.Timestamp(event.observed_peak_time).value
        halo = peak + np.arange(-20, 21, 10, dtype=np.int64) * MINUTE_NS
        requested = np.union1d(episode, halo)
        pos, present = _lookup(times, requested)
        if not present.all() or not season_mask(requested, years, months).all() or np.any(np.diff(requested) != STEP_NS):
            eligibility.at[index, "status"] = "observation_window_grid_incomplete"
            continue
        reference = obs[pos, site]
        if not np.all(np.isfinite(reference) & (reference >= 0) & (reference < 300)):
            eligibility.at[index, "status"] = "observation_window_values_invalid"
            continue
        positions, present = _lookup(issues, requested[:, None] - leads[None] * MINUTE_NS)
        if not present.all():
            eligibility.at[index, "status"] = "forecast_grid_incomplete"
            continue
        cubes = {route: p[positions[:, :, None], np.arange(n_members)[None, None, :],
                         np.arange(len(leads))[None, :, None], site].transpose(0, 2, 1)
                 for route, p in predictions.items()}
        if any(not np.all(np.isfinite(p) & (p >= 0)) for p in cubes.values()):
            eligibility.at[index, "status"] = "forecast_values_invalid"
            continue
        eligibility.at[index, "included"] = True
        eligibility.at[index, "status"] = "included"
        halo_pos = np.searchsorted(requested, halo)
        base = {key: event[key] for key in ["event_id", "site_id", "start_time", "end_time", "observed_peak_time",
                                           "observed_max_mm", "duration_min", "year"]}
        for route, cube in cubes.items():
            for mi, member in enumerate(member_ids[route]):
                for li, lead in enumerate(leads):
                    forecast = cube[halo_pos, mi, li].astype(np.float64)
                    for tolerance in TOLERANCES:
                        use = np.abs(np.arange(-20, 21, 10)) <= tolerance
                        selected = forecast[use]
                        amount = float(selected.max())
                        rows.append(base | dict(route=route, member_id=str(member), lead_minutes=int(lead),
                            tolerance_minutes=tolerance, n_window_forecasts=int(use.sum()),
                            predicted_peak_mm=amount, signed_error_mm=amount - float(event.observed_max_mm),
                            earliest_window_forecast_max_time=pd.Timestamp(halo[use][selected.argmax()]),
                            block_date=pd.Timestamp(peak).normalize()))
    error_columns = [*columns[:2], "start_time", "end_time", "observed_peak_time", "observed_max_mm", "duration_min", "year",
                     "route", "member_id", "lead_minutes", "tolerance_minutes", "n_window_forecasts",
                     "predicted_peak_mm", "signed_error_mm", "earliest_window_forecast_max_time", "block_date"]
    errors = pd.DataFrame(rows, columns=error_columns)
    expected = len(predictions) * n_members * len(leads) * len(TOLERANCES)
    if len(errors) and (errors.groupby("event_id").size().ne(expected).any() or
                       errors.duplicated(["event_id", "route", "member_id", "lead_minutes", "tolerance_minutes"]).any()):
        raise ValueError("Peak samples differ between routes, members, leads or windows")
    return {"observation_catalogue": catalog, "event_eligibility": eligibility, "peak_event_errors": errors}


def peak_summary(errors, routes, member_ids, leads, site_ids, kind):
    """Station: pooled event RMSE. Catchment: member RMSE then equal active-site mean."""
    grouped = {key: frame for key, frame in errors.groupby(["site_id", "route", "member_id", "lead_minutes", "tolerance_minutes"], sort=False)}
    member_rows, site_rows, route_rows = [], [], []
    for route in routes:
        for lead in leads:
            for tolerance in TOLERANCES:
                by_site = []
                member_pooled = [[] for _ in member_ids[route]]
                for site in site_ids:
                    values, counts = [], []
                    for mi, member in enumerate(member_ids[route]):
                        frame = grouped.get((str(site), route, str(member), int(lead), tolerance))
                        error = frame.signed_error_mm.to_numpy(float) if frame is not None else np.empty(0)
                        if not np.isfinite(error).all():
                            raise ValueError("Peak errors must be finite")
                        value = float(np.sqrt(np.mean(error ** 2))) if len(error) else np.nan
                        values.append(value); counts.append(len(error)); member_pooled[mi].extend(error)
                        member_rows.append(dict(route=route, site_id=str(site), member_id=str(member), lead_minutes=int(lead),
                            tolerance_minutes=tolerance, rmse_mm=value, n_events=len(error)))
                    if len(set(counts)) != 1:
                        raise ValueError("Peak event counts differ between members")
                    by_site.append(values)
                    site_rows.append(dict(route=route, site_id=str(site), lead_minutes=int(lead), tolerance_minutes=tolerance,
                        rmse_mm=float(np.mean(values)), member_sd=float(np.std(values, ddof=1)), n_events=counts[0]))
                values = np.asarray(by_site)
                active = np.isfinite(values).all(axis=1)
                if kind == "station":
                    per_member = np.array([np.sqrt(np.mean(np.square(e))) if len(e) else np.nan for e in member_pooled])
                elif kind == "catchment":
                    per_member = values[active].mean(axis=0) if active.any() else np.full(len(member_ids[route]), np.nan)
                else:
                    raise ValueError("kind must be station or catchment")
                route_rows.append(dict(route=route, lead_minutes=int(lead), tolerance_minutes=tolerance,
                    rmse_mm=float(per_member.mean()), member_sd=float(per_member.std(ddof=1)),
                    n_events=len(member_pooled[0]), n_event_sites=int(active.sum()), n_sites=len(site_ids)))
    return {"peak_member_scores": pd.DataFrame(member_rows), "peak_site_scores": pd.DataFrame(site_rows),
            "peak_route_scores": pd.DataFrame(route_rows)}


def peak_bootstrap(errors, routes, member_ids, leads, valid_times, kind, *, draws=2000, seed=20261018,
                   minimum_valid=None, years=(2024, 2025), months=(6, 7, 8, 9), counts=None):
    """Date of observed peak, full reference calendar, fixed event-evaluable sites.

    Dates with no event remain in the calendar. Draws omitting every event at
    any originally active catchment are undefined, not renormalized. The same
    date draws are used for all three tolerances and every comparison.
    """
    valid_times = time_axis(valid_times)
    if kind not in ("station", "catchment"):
        raise ValueError("kind must be station or catchment")
    calendar = np.unique(valid_times[season_mask(valid_times, years, months)] // DAY_NS)
    weights = date_counts(len(calendar), draws, seed) if counts is None else np.asarray(counts)
    if (weights.ndim != 2 or weights.shape != (draws, len(calendar)) or weights.dtype.kind not in "iu"
            or np.any(weights < 0) or not np.all(weights.sum(axis=1) == len(calendar))):
        raise ValueError("Each peak bootstrap draw must resample the full reference date calendar")
    minimum_valid = max(100, int(np.ceil(.8 * draws))) if minimum_valid is None else int(minimum_valid)
    if minimum_valid < 1:
        raise ValueError("minimum_valid must be positive")
    members = len(member_ids[routes[0]])
    shape = (len(routes), members, len(leads))
    route_rows, pair_rows = [], []
    for tolerance in TOLERANCES:
        frame = errors.loc[errors.tolerance_minutes.eq(tolerance)]
        events = frame.drop_duplicates("event_id")
        sites = list(events.site_id.unique()) if kind == "catchment" else ["pooled"] if len(events) else []
        count = np.zeros((len(calendar), len(sites)))
        squared = np.zeros((len(calendar), len(sites), *shape))
        groups = {key: value.set_index("event_id") for key, value in frame.groupby(["route", "member_id", "lead_minutes"])}
        for si, site in enumerate(sites):
            selected = events.loc[events.site_id.eq(site)] if kind == "catchment" else events
            days = pd.to_datetime(selected.block_date).astype("int64").to_numpy() // DAY_NS
            positions, found = _lookup(calendar, days)
            if not found.all():
                raise ValueError("Event date not present in the reference calendar")
            np.add.at(count[:, si], positions, 1)
            for ri, route in enumerate(routes):
                for mi, member in enumerate(member_ids[route]):
                    for li, lead in enumerate(leads):
                        error = groups[(route, str(member), int(lead))].loc[selected.event_id].signed_error_mm.to_numpy(float)
                        np.add.at(squared[:, si, ri, mi, li], positions, error ** 2)
        point = np.full((len(routes), len(leads)), np.nan)
        replicates = np.full((draws, len(routes), len(leads)), np.nan)
        if sites:
            point = np.sqrt(squared.sum(axis=0) / count.sum(axis=0)[:, None, None, None]).mean(axis=(0, 2))
            n = weights @ count
            good = (n > 0).all(axis=1)
            ss = (weights[good] @ squared.reshape(len(calendar), -1)).reshape(-1, len(sites), *shape)
            replicates[good] = np.sqrt(ss / n[good, :, None, None, None]).mean(axis=(1, 3))
        n_valid = int(np.isfinite(replicates[:, 0, 0]).sum())
        status = "defined" if n_valid >= minimum_valid else "no_events" if not sites else "insufficient_complete_site_resamples"
        base = dict(tolerance_minutes=tolerance, n_resamples=draws, n_defined_resamples=n_valid,
                    minimum_valid_required=minimum_valid, ci_status=status, n_date_blocks=len(calendar))
        low, high, _ = interval(replicates, point)
        if status != "defined":
            low[:], high[:] = np.nan, np.nan
        for ri, route in enumerate(routes):
            for li, lead in enumerate(leads):
                route_rows.append(base | dict(route=route, lead_minutes=int(lead), rmse_mm=float(point[ri, li]),
                    ci_low=float(low[ri, li]), ci_high=float(high[ri, li])))
        for ai, bi in combinations(range(len(routes)), 2):
            delta = point[ai] - point[bi]
            lo, hi, _ = interval(replicates[:, ai] - replicates[:, bi], delta)
            for li, lead in enumerate(leads):
                pair_rows.append(base | dict(route_a=routes[ai], route_b=routes[bi], lead_minutes=int(lead),
                    difference=float(delta[li]), ci_low=float(lo[li]) if status == "defined" else np.nan,
                    ci_high=float(hi[li]) if status == "defined" else np.nan))
    return {"peak_route_intervals": pd.DataFrame(route_rows), "peak_paired_intervals": pd.DataFrame(pair_rows)}, weights
