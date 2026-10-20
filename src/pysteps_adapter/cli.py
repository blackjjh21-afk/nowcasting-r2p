"""Command-line interface for explicit NPZ/NPY pySTEPS adapter workflows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import (
    FIELD_LEAD_MINUTES,
    INPUT_OFFSETS_MINUTES,
    TARGET_LEAD_MINUTES,
    extract_station_patches,
    forecast_normalized_hsr_lk,
    lead_window_indices,
    make_six_frame_lead_windows,
)


def load_array(path: Path, *, key: str) -> np.ndarray:
    """Load one array from an explicit .npy or .npz path."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            if key not in handle.files:
                raise KeyError(f"{path} does not contain key {key!r}; found {handle.files}")
            return np.asarray(handle[key])
    raise ValueError(f"expected an .npy or .npz path, got {path}")


def _load_optional_npz_key(path: Path, key: str) -> np.ndarray | None:
    if path.suffix.lower() != ".npz":
        return None
    with np.load(path, allow_pickle=False) as handle:
        return None if key not in handle.files else np.asarray(handle[key])


def _parse_station_ids(values: list[str]) -> np.ndarray:
    try:
        return np.asarray([int(value) for value in values], dtype=np.int64)
    except ValueError:
        return np.asarray(values, dtype=np.str_)


def _load_mapping_csv(
    path: Path,
    *,
    station_id_column: str,
    y_column: str,
    x_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {station_id_column, y_column, x_column}
        missing = sorted(required - fields)
        if missing:
            raise KeyError(f"{path} is missing mapping columns {missing}")
        station_ids: list[str] = []
        station_y: list[int] = []
        station_x: list[int] = []
        for row in reader:
            station_ids.append(row[station_id_column])
            station_y.append(int(row[y_column]))
            station_x.append(int(row[x_column]))
    if not station_ids:
        raise ValueError(f"mapping CSV contains no stations: {path}")
    return (
        _parse_station_ids(station_ids),
        np.asarray(station_y, dtype=np.int64),
        np.asarray(station_x, dtype=np.int64),
    )


def _write_output(
    path: Path,
    *,
    primary: np.ndarray,
    payload: dict[str, np.ndarray],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    if path.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError(f"output must end in .npy or .npz, got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        if path.suffix.lower() == ".npy":
            np.save(handle, primary, allow_pickle=False)
        else:
            np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _forecast(args: argparse.Namespace) -> None:
    input_hsr = load_array(args.input, key=args.input_key)
    forecast_hsr, forecast_rate = forecast_normalized_hsr_lk(input_hsr)
    _write_output(
        args.output,
        primary=forecast_hsr,
        payload={
            "forecast_normalized_hsr": forecast_hsr,
            "forecast_rain_rate_mm_h": forecast_rate,
            "lead_minutes": FIELD_LEAD_MINUTES,
            "input_offsets_minutes": INPUT_OFFSETS_MINUTES,
            "issue_times_ns": np.asarray([args.issue_time_ns], dtype=np.int64),
        },
        overwrite=args.overwrite,
    )


def _patches(args: argparse.Namespace) -> None:
    forecast = load_array(args.forecast, key=args.forecast_key)
    if args.mapping_csv is not None:
        if any(path is not None for path in (args.station_ids, args.station_y, args.station_x)):
            raise ValueError(
                "use either --mapping-csv or --station-ids/--station-y/--station-x"
            )
        station_ids, station_y, station_x = _load_mapping_csv(
            args.mapping_csv,
            station_id_column=args.station_id_column,
            y_column=args.y_column,
            x_column=args.x_column,
        )
    else:
        if any(path is None for path in (args.station_ids, args.station_y, args.station_x)):
            raise ValueError(
                "provide --mapping-csv, or provide all of --station-ids, --station-y, and --station-x"
            )
        station_ids = load_array(args.station_ids, key=args.station_ids_key)
        station_y = load_array(args.station_y, key=args.station_y_key)
        station_x = load_array(args.station_x, key=args.station_x_key)
    valid_mask = (
        None
        if args.valid_mask is None
        else load_array(args.valid_mask, key=args.valid_mask_key)
    )
    issue_times = (
        None
        if args.issue_times is None
        else load_array(args.issue_times, key=args.issue_times_key)
    )
    if issue_times is None:
        issue_times = _load_optional_npz_key(args.forecast, args.issue_times_key)
    batch_size = 1 if forecast.ndim == 3 else forecast.shape[0] if forecast.ndim == 4 else -1
    if issue_times is None:
        raise ValueError(
            "issue_times_ns must be present in the forecast NPZ or supplied with --issue-times"
        )
    issue_times = np.asarray(issue_times, dtype=np.int64)
    station_ids = np.asarray(station_ids)
    if issue_times.ndim != 1 or len(issue_times) != batch_size:
        raise ValueError(
            f"issue_times_ns must have shape [{batch_size}], got {issue_times.shape}"
        )
    if station_ids.ndim != 1 or len(station_ids) != len(np.asarray(station_y)):
        raise ValueError("station_ids must be 1-D and aligned with mapping coordinates")
    patches = extract_station_patches(
        forecast,
        station_y,
        station_x,
        patch_size=args.patch_size,
        valid_mask=valid_mask,
    )
    windows = make_six_frame_lead_windows(patches)
    if windows.ndim == 5:
        windows = windows[None, ...]
    indices = lead_window_indices()
    _write_output(
        args.output,
        primary=windows,
        payload={
            "patches": windows,
            "issue_times_ns": issue_times,
            "station_ids": station_ids,
            "lead_minutes": TARGET_LEAD_MINUTES,
            "station_y": np.asarray(station_y, dtype=np.int64),
            "station_x": np.asarray(station_x, dtype=np.int64),
            "source_field_lead_minutes": FIELD_LEAD_MINUTES,
            "window_field_lead_minutes": FIELD_LEAD_MINUTES[indices],
        },
        overwrite=args.overwrite,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic pySTEPS fields or CNN lead windows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    forecast = subparsers.add_parser(
        "forecast", help="forecast 18 normalized-HSR fields from seven inputs"
    )
    forecast.add_argument("--input", type=Path, required=True)
    forecast.add_argument("--input-key", default="normalized_hsr")
    forecast.add_argument("--issue-time-ns", type=int, required=True)
    forecast.add_argument("--output", type=Path, required=True)
    forecast.add_argument("--overwrite", action="store_true")
    forecast.set_defaults(handler=_forecast)

    patches = subparsers.add_parser(
        "patches", help="extract station patches and assemble six-frame lead windows"
    )
    patches.add_argument("--forecast", type=Path, required=True)
    patches.add_argument("--forecast-key", default="forecast_normalized_hsr")
    patches.add_argument("--mapping-csv", type=Path)
    patches.add_argument("--station-id-column", default="station_id")
    patches.add_argument("--y-column", default="exprecast_y")
    patches.add_argument("--x-column", default="exprecast_x")
    patches.add_argument("--station-ids", type=Path)
    patches.add_argument("--station-ids-key", default="station_ids")
    patches.add_argument("--station-y", type=Path)
    patches.add_argument("--station-y-key", default="station_y")
    patches.add_argument("--station-x", type=Path)
    patches.add_argument("--station-x-key", default="station_x")
    patches.add_argument("--issue-times", type=Path)
    patches.add_argument("--issue-times-key", default="issue_times_ns")
    patches.add_argument("--valid-mask", type=Path)
    patches.add_argument("--valid-mask-key", default="valid_mask")
    patches.add_argument("--patch-size", type=int, default=3)
    patches.add_argument("--output", type=Path, required=True)
    patches.add_argument("--overwrite", action="store_true")
    patches.set_defaults(handler=_patches)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
