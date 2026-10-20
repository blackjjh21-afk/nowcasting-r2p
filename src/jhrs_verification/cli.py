"""Portable array-manifest interface for station and catchment verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from basin_verification.core import area_mean, match_reference
from .scores import (THRESHOLDS, common_support, daily_statistics, bootstrap_scores,
                     result_tables, time_axis, validate_predictions)
from .peaks import evaluate_peaks, peak_summary, peak_bootstrap, season_mask, DAY_NS

SCHEMA = "jhrs-verification-v1"
PRIMARY_LEADS = (60, 90, 120, 150, 180)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_cells(values, weights, chunk=64):
    """Preserve complete-polygon weights, using bounded first-axis chunks."""
    output = np.empty((*values.shape[:-1], len(weights)), np.float64)
    for start in range(0, len(values), chunk):
        output[start:start + chunk] = area_mean(values[start:start + chunk], weights)
    return output


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Expected manifest schema {SCHEMA}")
    if manifest.get("timezone") != "Asia/Seoul" or manifest.get("timestamp_encoding") != "naive_kst_nanoseconds":
        raise ValueError("Declare Asia/Seoul and naive_kst_nanoseconds; UTC offsets are not inferred")
    kind = manifest.get("kind")
    if kind not in ("station", "catchment"):
        raise ValueError("kind must be station or catchment")
    paths, hashes = {}, {}

    def load(name):
        if not isinstance(name, str) or not name:
            raise ValueError("Array filenames must be nonempty strings")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Use paths relative to the manifest, without parent traversal")
        target = (path.parent / relative).resolve()
        if not target.is_relative_to(path.parent):
            raise ValueError("Array symlinks must stay within the manifest directory")
        if name not in paths:
            hashes[name] = file_hash(target)
            expected = manifest.get("sha256", {}).get(name)
            if expected is not None and hashes[name] != expected:
                raise ValueError(f"Input hash mismatch: {name}")
            paths[name] = np.load(target, allow_pickle=False, mmap_mode="r")
        return paths[name]

    issues = time_axis(load(manifest["issues"]), "issues")
    valid_times = time_axis(load(manifest["valid_times"]), "valid_times")
    observed = load(manifest["observations"])
    if observed.ndim != 2 or observed.shape[0] != len(valid_times) or observed.dtype.kind != "f":
        raise ValueError("observations must be floating [valid_time,site_or_cell]")
    leads = manifest["lead_minutes"]
    if leads != list(PRIMARY_LEADS) or any(type(v) is not int for v in leads):
        raise ValueError("This protocol uses leads [60,90,120,150,180] minutes")
    route_records = manifest["routes"]
    if len(route_records) != 4 or len({r["name"] for r in route_records}) != 4:
        raise ValueError("Declare four distinct forecast routes")
    predictions, member_ids = {}, {}
    for route in route_records:
        name = route["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("Route names must be nonempty strings")
        member_ids[name] = [str(v) for v in route["member_ids"]]
        if len(member_ids[name]) != 3 or len(set(member_ids[name])) != 3:
            raise ValueError("Declare three distinct member IDs per route")
        predictions[name] = load(route["predictions"])
    validate_predictions(predictions, np.broadcast_to(0., (len(issues), len(leads), observed.shape[1])))
    if "observation_valid" in manifest:
        valid = load(manifest["observation_valid"])
        if valid.dtype != np.bool_ or valid.shape != observed.shape:
            raise ValueError("observation_valid must match observations with Boolean dtype")
        observed = np.where(valid, observed, np.nan)
    if "area_weights" in manifest:
        if kind != "catchment":
            raise ValueError("Area aggregation applies only to catchments")
        weights = load(manifest["area_weights"])
        observed = aggregate_cells(observed, weights)
        predictions = {name: aggregate_cells(p, weights) for name, p in predictions.items()}
    site_ids = [str(v) for v in manifest["site_ids"]]
    expected = manifest.get("expected_site_count", 47 if kind == "catchment" else 128)
    if (len(site_ids) != expected or len(set(site_ids)) != len(site_ids)
            or observed.shape[1] != len(site_ids)):
        raise ValueError("site_ids must uniquely label the declared site count and final site axis")
    years, months = manifest.get("years", [2024, 2025]), manifest.get("months", [6, 7, 8, 9])
    if (not years or len(set(years)) != len(years) or any(type(v) is not int for v in years)
            or not months or len(set(months)) != len(months) or any(type(v) is not int or not 1 <= v <= 12 for v in months)):
        raise ValueError("Declare unique integer years and calendar months")
    if not season_mask(issues, years, months).all():
        raise ValueError("Issuance times fall outside declared evaluation seasons")
    aligned = np.stack([match_reference(valid_times, observed, issues, lead) for lead in leads], axis=1)
    return dict(manifest=manifest, manifest_path=path, manifest_sha256=file_hash(path), input_sha256=hashes,
                issues=issues, valid_times=valid_times, observed=observed, aligned=aligned,
                predictions=predictions, member_ids=member_ids, site_ids=site_ids,
                leads=leads, kind=kind, years=years, months=months)


def run(manifest_path, output, *, draws=2000, seed=20261018, peaks=True, minimum_peak_resamples=None):
    """Write a new result directory; never mutate forecasts or overwrite results."""
    if draws < 1:
        raise ValueError("Bootstrap resample count must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; use a new result directory")
    data = load_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    routes, members = list(data["predictions"]), data["member_ids"]
    common = common_support(data["predictions"], data["aligned"])
    dates, daily = daily_statistics(data["predictions"], data["aligned"], common, data["issues"])
    np.save(output / "common_support.npy", common)
    np.save(output / "daily_statistics.npy", daily)
    np.save(output / "issuance_date_blocks.npy", dates.astype("datetime64[ns]").astype(np.int64))
    years = data["years"]
    periods = [(str(year), [year]) for year in years]
    if len(years) > 1:
        periods.append(("-".join(str(y) for y in years), years))
    tables, resamples = {}, {}
    all_peaks = None
    if peaks:
        all_peaks = evaluate_peaks(data["valid_times"], data["observed"], data["issues"], data["leads"],
                                  data["predictions"], members, data["site_ids"], years=years, months=data["months"])
        for name, table in all_peaks.items():
            table.to_csv(output / f"{name}.csv", index=False)
    # Historical basin scores use seed+period_index; peak resamples share one
    # RNG stream across annual periods and then the pooled period.
    peak_rng = np.random.default_rng(seed)
    for pi, (label, period_years) in enumerate(periods):
        use = np.isin(pd.DatetimeIndex(dates).year, period_years)
        period_daily = daily[use]
        if not len(period_daily):
            raise ValueError(f"No issuance dates for period {label}")
        replicates, counts = bootstrap_scores(period_daily, data["kind"], draws=draws, seed=seed + pi)
        resamples[f"issuance_counts_{label}"] = counts
        resamples[f"issuance_dates_{label}"] = dates[use].astype("datetime64[ns]").astype(np.int64)
        result = result_tables(period_daily, routes, members, data["leads"], data["site_ids"], data["kind"], replicates, period=label)
        if peaks:
            errors = all_peaks["peak_event_errors"]
            errors = errors.loc[errors.year.isin(period_years)]
            peak_result = peak_summary(errors, routes, members, data["leads"], data["site_ids"], data["kind"])
            calendar = np.unique(data["valid_times"][season_mask(data["valid_times"], period_years, data["months"])] // DAY_NS)
            counts = peak_rng.multinomial(len(calendar), np.full(len(calendar), 1. / len(calendar)), size=draws)
            intervals, _ = peak_bootstrap(errors, routes, members, data["leads"], data["valid_times"], data["kind"],
                draws=draws, counts=counts, minimum_valid=minimum_peak_resamples, years=period_years, months=data["months"])
            resamples[f"peak_counts_{label}"] = counts
            resamples[f"peak_dates_{label}"] = calendar * DAY_NS
            for name, table in (peak_result | intervals).items():
                table.insert(0, "period", label)
            result |= peak_result | intervals
        for name, table in result.items():
            tables.setdefault(name, []).append(table)
    for name, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    np.savez_compressed(output / "bootstrap_date_counts.npz", **resamples)
    # A long analysis must not publish a receipt for inputs changed while it ran.
    if file_hash(data["manifest_path"]) != data["manifest_sha256"]:
        raise ValueError("Manifest changed during verification")
    for name, expected in data["input_sha256"].items():
        if file_hash(data["manifest_path"].parent / name) != expected:
            raise ValueError(f"Input changed during verification: {name}")
    report = dict(schema=SCHEMA, completed=True, kind=data["kind"], thresholds_mm=list(THRESHOLDS),
        csi_m="Arithmetic mean of CSI at 1, 5 and 10 mm", member_aggregation="Score members before averaging",
        spatial_aggregation="Equal catchment mean" if data["kind"] == "catchment" else "Pooled station pairs",
        n_routes=len(routes), n_members=3, n_sites=len(data["site_ids"]), n_issues=len(data["issues"]),
        n_common_pairs=int(common.sum()), leads_minutes=data["leads"], periods=[p[0] for p in periods],
        bootstrap_resamples=draws, bootstrap_seed=seed, peaks=bool(peaks),
        manifest_sha256=data["manifest_sha256"], input_sha256=data["input_sha256"],
        output_sha256={p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--no-peaks", action="store_true")
    parser.add_argument("--minimum-peak-resamples", type=int, default=None,
                        help="Default max(100, ceil(0.8 * bootstrap)); smaller values are for tests")
    args = parser.parse_args(argv)
    report = run(args.manifest, args.output, draws=args.bootstrap, seed=args.seed,
                 peaks=not args.no_peaks, minimum_peak_resamples=args.minimum_peak_resamples)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
