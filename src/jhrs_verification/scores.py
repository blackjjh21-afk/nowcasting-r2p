"""Numerical counterpart of the JHRS four-route national-basin evaluator.

All calculations use original saved prediction precision, promoted to float64
for sums. A station score pools pairs; a catchment score averages separately
scored catchments. Neither substitutes an ensemble-mean forecast for members.
"""
from __future__ import annotations

from itertools import combinations
import warnings
import numpy as np
import pandas as pd

THRESHOLDS = (1.0, 5.0, 10.0)
METRICS = (("RMSE", np.nan), ("MAE", np.nan), ("Bias", np.nan),
           *(("CSI", t) for t in THRESHOLDS))
MINUTE_NS = 60_000_000_000
STEP_NS = 10 * MINUTE_NS


def time_axis(values, name="times"):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype != np.dtype("int64") or not len(values)
            or np.any(values == np.iinfo(np.int64).min) or np.any(values % STEP_NS)
            or np.any(np.diff(values) <= 0)):
        raise ValueError(f"{name} must contain increasing unique KST-naive 10-minute int64 nanoseconds")
    return values


def validate_predictions(predictions, observed):
    observed = np.asarray(observed)
    if observed.ndim != 3 or not predictions:
        raise ValueError("Reference must be [issue,lead,site] and predictions nonempty")
    members = None
    for name, p in predictions.items():
        if p.ndim != 4 or p.shape[0] != observed.shape[0] or p.shape[2:] != observed.shape[1:] or p.dtype.kind != "f":
            raise ValueError(f"{name}: expected floating [issue,member,lead,site] predictions")
        members = p.shape[1] if members is None else members
        if not members or p.shape[1] != members:
            raise ValueError("Routes must have the same nonzero member count")
    return members


def common_support(predictions, observed, reference_valid=None):
    """One mask per issue/lead/site, intersecting every route and member."""
    observed = np.asarray(observed)
    validate_predictions(predictions, observed)
    good = np.isfinite(observed) & (observed >= 0)
    if reference_valid is not None:
        valid = np.asarray(reference_valid)
        if valid.dtype != np.bool_ or valid.shape != good.shape:
            raise ValueError("Reference validity must be a matching Boolean array")
        good &= valid
    for p in predictions.values():
        good &= np.all(np.isfinite(p) & (p >= 0), axis=1)
    return good


def daily_statistics(predictions, observed, support, issues):
    """Return dates and [date,route,member,lead,site,10] sufficient statistics.

    Slots are N, sum(error), sum(abs(error)), sum(error²), then hit/union
    pairs for 1, 5 and 10 mm. Dates are KST labels; no UTC conversion occurs.
    """
    issues = time_axis(issues, "issuance times")
    observed, support = np.asarray(observed), np.asarray(support)
    members = validate_predictions(predictions, observed)
    if len(issues) != len(observed) or support.dtype != np.bool_ or support.shape != observed.shape:
        raise ValueError("Issue/reference/support axes differ")
    dates, starts = np.unique(issues.astype("datetime64[ns]").astype("datetime64[D]"), return_index=True)
    stops = np.r_[starts[1:], len(issues)]
    daily = np.empty((len(dates), len(predictions), members, *observed.shape[1:], 10), np.float64)
    for di, (start, stop) in enumerate(zip(starts, stops)):
        use = support[start:stop, None]
        obs = np.asarray(observed[start:stop, None], np.float64)
        if np.any(use & (~np.isfinite(obs) | (obs < 0))):
            raise ValueError("Invalid reference on common support")
        for ri, source in enumerate(predictions.values()):
            pred = np.asarray(source[start:stop], np.float64)
            if np.any(use & (~np.isfinite(pred) | (pred < 0))):
                raise ValueError("Invalid prediction on common support")
            error = np.where(use, pred - obs, 0.)
            daily[di, ri, ..., 0] = use.sum(axis=0)
            daily[di, ri, ..., 1] = error.sum(axis=0)
            daily[di, ri, ..., 2] = np.abs(error).sum(axis=0)
            daily[di, ri, ..., 3] = (error * error).sum(axis=0)
            for ti, threshold in enumerate(THRESHOLDS):
                o, p = obs >= threshold, pred >= threshold
                daily[di, ri, ..., 4 + 2 * ti] = (use & o & p).sum(axis=0)
                daily[di, ri, ..., 5 + 2 * ti] = (use & (o | p)).sum(axis=0)
    return dates, daily


def scores_from_sums(sums):
    sums = np.asarray(sums, dtype=np.float64)
    if sums.shape[-1] != 10:
        raise ValueError("Ten three-threshold sufficient statistics are required")
    result = np.full((*sums.shape[:-1], len(METRICS)), np.nan)
    n = sums[..., 0]
    np.divide(sums[..., 3], n, out=result[..., 0], where=n > 0)
    result[..., 0] = np.sqrt(result[..., 0])
    np.divide(sums[..., 2], n, out=result[..., 1], where=n > 0)
    np.divide(sums[..., 1], n, out=result[..., 2], where=n > 0)
    for ti in range(3):
        denominator = sums[..., 5 + 2 * ti]
        np.divide(sums[..., 4 + 2 * ti], denominator, out=result[..., 3 + ti], where=denominator > 0)
    return result


def aggregate_scores(sums, kind):
    """[...route,member,lead,site,stat] -> [...route,lead,metric]."""
    if kind == "station":
        return scores_from_sums(sums.sum(axis=-2)).mean(axis=-3)
    if kind == "catchment":
        return scores_from_sums(sums).mean(axis=-4).mean(axis=-2)
    raise ValueError("kind must be station or catchment")


def date_counts(n_dates, draws=2000, seed=20261018):
    if n_dates < 1 or draws < 1:
        raise ValueError("Positive date and resample counts required")
    return np.random.default_rng(seed).multinomial(n_dates, np.full(n_dates, 1. / n_dates), size=draws)


def bootstrap_scores(daily, kind, *, draws=2000, seed=20261018, counts=None, chunk=64):
    """Shared date multiplicities for every route/member/lead/site; no spatial resampling."""
    if daily.ndim != 6 or daily.shape[-1] != 10 or not len(daily) or chunk < 1:
        raise ValueError("Expected nonempty daily sufficient statistics")
    counts = date_counts(len(daily), draws, seed) if counts is None else np.asarray(counts)
    if (counts.ndim != 2 or counts.shape[1] != len(daily) or counts.dtype.kind not in "iu"
            or np.any(counts < 0) or not np.all(counts.sum(axis=1) == len(daily))):
        raise ValueError("Each bootstrap draw must resample the full date calendar")
    output = np.empty((len(counts), daily.shape[1], daily.shape[3], len(METRICS)))
    matrix = daily.reshape(len(daily), -1)
    for start in range(0, len(counts), chunk):
        sums = (counts[start:start + chunk] @ matrix).reshape(-1, *daily.shape[1:])
        output[start:start + chunk] = aggregate_scores(sums, kind)
    return output, counts


def interval(draws, point):
    """Conditional finite-replicate percentiles; undefined point estimates stay undefined."""
    finite = np.isfinite(draws)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanquantile(draws, [.025, .975], axis=0)
    return np.where(np.isfinite(point), low, np.nan), np.where(np.isfinite(point), high, np.nan), finite.sum(axis=0)


def result_tables(daily, routes, member_ids, leads, site_ids, kind, draws, *, period="all"):
    """Tidy member/site/overall scores and paired route intervals, including CSI-M."""
    total = daily.sum(axis=0)
    member_site = scores_from_sums(total)
    aggregate = aggregate_scores(total, kind)
    records = {"member_scores": [], "site_scores": [], "route_scores": [], "paired_differences": []}
    labels = [*METRICS, ("CSI-M", np.nan)]
    def add_macro(x):
        return np.concatenate((x, x[..., 3:].mean(axis=-1, keepdims=True)), axis=-1)
    member_site, aggregate, draws = map(add_macro, (member_site, aggregate, draws))
    member_overall = add_macro(scores_from_sums(total.sum(axis=-2))) if kind == "station" else member_site.mean(axis=-2)
    low, high, n = interval(draws, aggregate)
    for ri, route in enumerate(routes):
        for li, lead in enumerate(leads):
            for ki, (metric, threshold) in enumerate(labels):
                base = dict(period=period, route=route, lead_minutes=int(lead), metric=metric, threshold_mm=threshold)
                for si, site in enumerate(site_ids):
                    values = member_site[ri, :, li, si, ki]
                    records["site_scores"].append(base | dict(site_id=str(site), value=float(values.mean()), member_sd=float(values.std(ddof=1)), n_pairs=int(total[ri, 0, li, si, 0])))
                    for mi, member in enumerate(member_ids[route]):
                        records["member_scores"].append(base | dict(site_id=str(site), member_id=str(member), value=float(values[mi])))
                v = member_overall[ri, :, li, ki]
                records["route_scores"].append(base | dict(value=float(aggregate[ri, li, ki]), member_sd=float(v.std(ddof=1)),
                    ci_low=float(low[ri, li, ki]), ci_high=float(high[ri, li, ki]), n_defined_resamples=int(n[ri, li, ki]), n_resamples=len(draws)))
    for ai, bi in combinations(range(len(routes)), 2):
        delta = aggregate[ai] - aggregate[bi]
        low, high, n = interval(draws[:, ai] - draws[:, bi], delta)
        for li, lead in enumerate(leads):
            for ki, (metric, threshold) in enumerate(labels):
                records["paired_differences"].append(dict(period=period, route_a=routes[ai], route_b=routes[bi],
                    lead_minutes=int(lead), metric=metric, threshold_mm=threshold, difference=float(delta[li, ki]),
                    ci_low=float(low[li, ki]), ci_high=float(high[li, ki]), n_defined_resamples=int(n[li, ki]),
                    defined_fraction=float(n[li, ki] / len(draws)), n_resamples=len(draws)))
    return {name: pd.DataFrame(rows) for name, rows in records.items()}
