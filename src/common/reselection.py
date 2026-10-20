"""Offline final CSI(1,5,10 mm) selection from saved validation/OOF scores.

This command does not train a model, inspect TEST predictions, rewrite an old
trace, or change the historical optimizer/LR/early-stop monitor. For a CNN it
selects a fresh full-2023 refit budget; an OOF checkpoint is not a final model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

THRESHOLDS = (1.0, 5.0, 10.0)
LEADS = tuple(range(60, 181, 10))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _epoch(value):
    number = float(value)
    if not np.isfinite(number) or number < 1 or int(number) != number:
        raise ValueError("Epoch must be a positive integer")
    return int(number)


def _counts_to_csi(hits, misses, false_alarms):
    arrays = [np.asarray(value, np.float64) for value in (hits, misses, false_alarms)]
    if any(value.shape != arrays[0].shape or not np.isfinite(value).all()
           or (value < 0).any() for value in arrays):
        raise ValueError("Contingency counts must have matching finite nonnegative shapes")
    denominator = sum(arrays)
    return np.divide(arrays[0], denominator, out=np.full(denominator.shape, np.nan),
                     where=denominator > 0)


def _matrix_record(epoch, csi, leads, thresholds):
    value = np.asarray(csi, np.float64)
    leads, thresholds = tuple(map(int, leads)), tuple(map(float, thresholds))
    if (len(set(leads)) != len(leads) or len(set(thresholds)) != len(thresholds)
            or value.shape != (len(leads), len(thresholds))):
        raise ValueError("CSI matrix and unique source lead/threshold axes disagree")
    return {"epoch": _epoch(epoch), "leads": leads, "thresholds": thresholds, "csi": value}


def _json_records(payload):
    if payload.get("test_period_accessed") is True or payload.get("test_accessed") is True:
        raise ValueError("Selection source reports TEST access")
    if "calibration_years" in payload and payload["calibration_years"] != [2023]:
        raise ValueError("CNN final selection requires 2023-only calibration")
    # Public 12-fold OOF output: counts have already been pooled over the cells.
    if "pooled_contingency_counts_by_epoch" in payload:
        counts = np.asarray(payload["pooled_contingency_counts_by_epoch"], np.float64)
        if counts.ndim != 4 or counts.shape[-1] != 4:
            raise ValueError("Expected pooled [epoch,lead,threshold,H/M/F/CN] counts")
        scores = _counts_to_csi(counts[..., 0], counts[..., 1], counts[..., 2])
        return [_matrix_record(i + 1, matrix, payload["selection_leads_min"],
                               payload["selection_thresholds_mm"])
                for i, matrix in enumerate(scores)]
    # The archived NowcastNet CNN selection stores pooled counts and CSI.
    if "pooled_epochs" in payload:
        result = []
        for record in payload["pooled_epochs"]:
            csi = _counts_to_csi(record["hits"], record["misses"], record["false_alarms"])
            # This archived schema fixes +60..180 by10 and 1/5/10/20 mm.
            result.append(_matrix_record(record["epoch"], csi,
                payload.get("selection_leads_min", LEADS),
                payload.get("selection_thresholds_mm", (1., 5., 10., 20.))))
        return result
    record = payload.get("validation", payload)
    matrix = record.get("csi_by_lead_threshold", record.get("csi"))
    if matrix is None:
        raise ValueError("JSON lacks per-lead/per-threshold validation CSI")
    leads = record.get("lead_minutes", record.get("leads_min"))
    thresholds = record.get("thresholds_mm")
    if leads is None or thresholds is None:
        raise ValueError("Validation JSON must identify its source axes")
    return [_matrix_record(payload["epoch"], matrix, leads, thresholds)]


def _csv_records(path, *, family=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty validation CSV")
    if family is not None:
        rows = [row for row in rows if row.get("family") == family]
    elif len({row.get("family") for row in rows if row.get("family")}) > 1:
        raise ValueError("Multiple model families; specify --family")
    if not rows:
        raise ValueError("No rows for the requested family")
    filename_epoch = re.search(r"epoch[_-]?(\d+)", Path(path).name)
    grouped = {}
    for row in rows:
        number = row.get("epoch") or (filename_epoch.group(1) if filename_epoch else None)
        if number is None:
            raise ValueError("CSV requires an epoch column or epoch_NNN filename")
        grouped.setdefault(_epoch(number), []).append(row)
    result = []
    for epoch, group in grouped.items():
        columns = group[0]
        if all(f"csi_{q:g}mm_mean_13lead" in columns for q in THRESHOLDS):
            if len(group) != 1:
                raise ValueError("Duplicate per-epoch lead-mean rows")
            # Mean over the three column means equals the equal 13x3-cell mean.
            means = [float(group[0][f"csi_{q:g}mm_mean_13lead"]) for q in THRESHOLDS]
            record = _matrix_record(epoch, np.tile(means, (len(LEADS), 1)), LEADS, THRESHOLDS)
            record["preaggregated_lead_means"] = True
            result.append(record)
            continue
        cells = {}
        if "threshold_mm" in columns and "csi" in columns:
            for row in group:
                lead = int(row.get("lead_min") or row.get("lead_minutes") or 0)
                key = (lead, float(row["threshold_mm"]))
                if key in cells:
                    raise ValueError("Duplicate epoch/lead/threshold; pool OOF counts before selecting")
                cells[key] = float(row["csi"])
        else:
            names = [name for name in columns if re.fullmatch(r"csi_(\d+(?:\.\d+)?)", name)]
            if not names:
                raise ValueError("CSV lacks recognized per-threshold CSI columns")
            for row in group:
                lead = int(row.get("lead_min") or row.get("lead_minutes") or 0)
                for name in names:
                    key = (lead, float(name[4:]))
                    if key in cells:
                        raise ValueError("Duplicate epoch/lead/threshold")
                    cells[key] = float(row[name])
        leads = sorted({key[0] for key in cells})
        thresholds = sorted({key[1] for key in cells})
        matrix = [[cells.get((lead, threshold), np.nan) for threshold in thresholds] for lead in leads]
        result.append(_matrix_record(epoch, matrix, leads, thresholds))
    return result


def read_records(paths, *, family=None):
    records = []
    for path in map(Path, paths):
        if path.suffix.lower() == ".csv":
            current = _csv_records(path, family=family)
        elif path.suffix.lower() == ".jsonl":
            current = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    current.extend(_json_records(json.loads(line)))
        else:
            current = _json_records(json.loads(path.read_text(encoding="utf-8")))
        for record in current:
            record["source_name"] = path.name
        records.extend(current)
    return records


def select_csi3(records, *, leads=LEADS, exclude_through=0):
    """Equal-cell mean, earliest exact maximum, on the original epoch domain."""
    leads = tuple(map(int, leads))
    if not leads or len(set(leads)) != len(leads) or exclude_through < 0:
        raise ValueError("Require unique selection leads and a nonnegative warm-up")
    scores, seen = [], set()
    for record in records:
        epoch = _epoch(record["epoch"])
        if epoch in seen:
            raise ValueError("Epoch appears more than once; select each model/seed separately")
        seen.add(epoch)
        if record.get("preaggregated_lead_means") and leads != LEADS:
            raise ValueError("Preaggregated 13-lead means cannot select a different lead set")
        try:
            rows = [record["leads"].index(lead) for lead in leads]
            columns = [record["thresholds"].index(q) for q in THRESHOLDS]
        except ValueError as error:
            raise ValueError("A required selection lead or threshold is absent") from error
        values = np.asarray(record["csi"], np.float64)[np.ix_(rows, columns)]
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("All requested selection CSI cells must be defined and within [0,1]")
        scores.append({"epoch": epoch, "csi3": float(values.mean()),
                       "eligible": epoch > exclude_through})
    eligible = [row for row in scores if row["eligible"]]
    if not eligible:
        raise ValueError("No eligible validation epochs")
    best = max(eligible, key=lambda row: (row["csi3"], -row["epoch"]))
    return {
        "schema": "jhrs_final_csi3_selection_v1", "validation_year": 2023,
        "selection_leads_min": list(leads), "selection_thresholds_mm": list(THRESHOLDS),
        "selection_rule": "equal mean of lead-threshold CSI; earliest exact maximum",
        "excluded_initial_epochs": list(range(1, exclude_through + 1)),
        "selected_epoch": best["epoch"], "selected_score": best["csi3"],
        "scores_by_epoch": sorted(scores, key=lambda row: row["epoch"]),
        "training_changed": False, "historical_monitor_changed": False,
        "test_used_for_selection": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leads", type=int, nargs="+", default=LEADS,
                        help="Default +60..180 by10; use --leads 0 for observed-HSR readouts")
    parser.add_argument("--exclude-through", type=int, default=0,
                        help="Direct R2P and its ablations use 3; field/CNN selection uses 0")
    parser.add_argument("--family", help="Select one family from a multi-family OOF CSV")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve() in [p.resolve() for p in args.input]:
        raise FileExistsError("Use a new selection JSON; existing sources/results are not overwritten")
    result = select_csi3(read_records(args.input, family=args.family),
                         leads=args.leads, exclude_through=args.exclude_through)
    result["source_statistics"] = [{"name": path.name, "sha256": _sha(path)} for path in args.input]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_epoch": result["selected_epoch"],
                      "selected_score": result["selected_score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
