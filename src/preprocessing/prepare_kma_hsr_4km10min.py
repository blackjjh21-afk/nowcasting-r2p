#!/usr/bin/env python3
"""Prepare the shared 4-km/10-min KMA HSR inputs used in the manuscript.

This is an independent, audited implementation of the transform published in
``tony890048/Processing-Radar-Datasets/kma_preprocessing.py``.  It accepts the
provider's 500-m HSR binary containers and writes the normalized 4-km,
256 x 256 float32 TIFF files used by all four forecast routes.

The transform intentionally uses aligned uniform subsampling.  It must not be
replaced by block averaging or bilinear resizing.

Input timestamps and output paths use Korea Standard Time (KST), at minute
phase 0 with ten-minute spacing.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import struct
import time
import uuid
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile


REFERENCE_URL = (
    "https://github.com/tony890048/Processing-Radar-Datasets/"
    "blob/main/kma_preprocessing.py"
)
REFERENCE_PAPER = "https://arxiv.org/html/2602.05204#A1.SS1"
# Provider container contract.
RAW_HEIGHT = 2881
RAW_WIDTH = 2305
RAW_SHAPE = (RAW_HEIGHT, RAW_WIDTH)
RAW_DTYPE = np.dtype("<i2")
ARCHIVE_HEADER_BYTES = 1024
API_HEADER_BYTES = 4
OUTSIDE_DOMAIN = -30000
NO_ECHO = -25000

# Frozen indices from the authors' published RegionTransform.  These are the
# rounded LCC indices for 29.0--42.0 N and 120.5--137.5 E under:
# +proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 ...
REGION_X1 = 45
REGION_Y1 = -253
REGION_X2 = 2959
REGION_Y2 = 2675
SQUARE_SIZE = 2936
UNIFORM_STRIDE = 8
SUBSAMPLED_SIZE = 367
TARGET_SIZE = 256
TARGET_START = 55
NORMALIZATION_DIVISOR = np.float32(10000.0)
SOURCE_RESOLUTION_METRES = 500.0
LCC_PROJ4 = (
    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 "
    "+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
)

TIMESTAMP_RE = re.compile(r"(?<!\d)(20\d{10})(?!\d)")
TEN_MINUTES = timedelta(minutes=10)


class PreparationError(RuntimeError):
    """Raised when an input cannot satisfy the shared HSR input contract."""


@dataclass(frozen=True)
class SourceFrame:
    timestamp: datetime
    path: Path
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class FrameResult:
    timestamp_kst: str
    source: str
    output: str
    status: str
    container: str
    compressed_bytes: int
    output_min: float
    output_max: float
    positive_pixels: int
    outside_domain_pixels: int
    no_echo_pixels: int
    source_sha256: str
    error: str


def parse_years(specification: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated years and inclusive ranges."""

    if specification is None or not specification.strip():
        return None
    years: set[int] = set()
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise PreparationError(f"descending year range: {token!r}")
            years.update(range(start, stop + 1))
        else:
            years.add(int(token))
    if not years:
        raise PreparationError("no years selected")
    if min(years) < 1900 or max(years) > 2200:
        raise PreparationError(f"year outside supported range: {sorted(years)}")
    return tuple(sorted(years))


def parse_timestamp(path: Path) -> datetime:
    match = TIMESTAMP_RE.search(path.name)
    if match is None:
        raise PreparationError(f"no 12-digit timestamp in source name: {path}")
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M")
    except ValueError as error:
        raise PreparationError(f"invalid source timestamp in {path}: {error}") from error


def discover_sources(
    raw_root: Path,
    *,
    years: tuple[int, ...] | None,
    minute_phase: int,
) -> tuple[list[SourceFrame], dict[str, int]]:
    if not raw_root.is_dir():
        raise PreparationError(f"raw root not found: {raw_root}")
    if minute_phase != 0:
        raise PreparationError("the manuscript requires minute phase 0")

    all_hsr = 0
    excluded_year = 0
    excluded_phase = 0
    seen: dict[datetime, Path] = {}
    selected: list[SourceFrame] = []
    inventory_digest = hashlib.sha256()
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or "RDR_CMP_HSR" not in path.name:
            continue
        if not (path.name.endswith(".bin") or path.name.endswith(".bin.gz")):
            continue
        stat = path.stat()
        inventory_digest.update(
            f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
        all_hsr += 1
        timestamp = parse_timestamp(path)
        if years is not None and timestamp.year not in years:
            excluded_year += 1
            continue
        if timestamp.minute % 10 != minute_phase:
            excluded_phase += 1
            continue
        previous = seen.get(timestamp)
        if previous is not None:
            raise PreparationError(
                f"duplicate timestamp {timestamp:%Y-%m-%d %H:%M}: "
                f"{previous} and {path}"
            )
        seen[timestamp] = path
        selected.append(
            SourceFrame(
                timestamp=timestamp,
                path=path.resolve(),
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )

    selected.sort(key=lambda item: item.timestamp)
    if not selected:
        raise PreparationError(
            f"no phase-{minute_phase} KMA HSR files found below {raw_root}"
        )
    return selected, {
        "all_hsr_files": all_hsr,
        "excluded_by_year": excluded_year,
        "excluded_by_minute_phase": excluded_phase,
        "selected_files": len(selected),
        "inventory_sha256": inventory_digest.hexdigest(),
    }


def _decompress_if_needed(file_bytes: bytes) -> tuple[bytes, bool]:
    if file_bytes.startswith(b"\x1f\x8b"):
        try:
            return gzip.decompress(file_bytes), True
        except (gzip.BadGzipFile, EOFError, OSError, zlib.error) as error:
            raise PreparationError(f"invalid gzip HSR payload: {error}") from error
    return file_bytes, False


def decode_hsr_payload(payload: bytes) -> tuple[np.ndarray, str]:
    """Decode one provider HSR payload without altering its raw values."""

    values = RAW_HEIGHT * RAW_WIDTH
    archive_size = ARCHIVE_HEADER_BYTES + values * RAW_DTYPE.itemsize
    api_size = API_HEADER_BYTES + values * RAW_DTYPE.itemsize
    if len(payload) == archive_size:
        offset = ARCHIVE_HEADER_BYTES
        container = "archive_header_1024"
        nx, ny = struct.unpack_from("<hh", payload, 20)
    elif len(payload) == api_size:
        offset = API_HEADER_BYTES
        container = "api_header_4"
        nx, ny = struct.unpack_from("<hh", payload, 0)
    else:
        raise PreparationError(
            f"unexpected decompressed size {len(payload):,}; expected "
            f"{archive_size:,} or {api_size:,} bytes"
        )
    if (nx, ny) != (RAW_WIDTH, RAW_HEIGHT):
        raise PreparationError(
            f"unexpected header dimensions {(nx, ny)}; expected "
            f"{(RAW_WIDTH, RAW_HEIGHT)}"
        )
    array = np.frombuffer(
        payload, dtype=RAW_DTYPE, count=values, offset=offset
    ).reshape(RAW_SHAPE)
    return array, container


def read_hsr(path: Path) -> tuple[np.ndarray, str, int, bytes]:
    file_bytes = path.read_bytes()
    payload, was_gzip = _decompress_if_needed(file_bytes)
    raw, container = decode_hsr_payload(payload)
    if was_gzip:
        container += "_gzip"
    return raw, container, len(file_bytes), file_bytes


def official_region_transform_reference(raw_hsr: np.ndarray) -> np.ndarray:
    """Apply the exact public exPreCast KMA 500-m-to-4-km transform.

    The output is normalized float32 reflectivity.  A value ``v`` represents
    ``100 * v`` dBZ.  Negative provider codes and negative finite dBZ values
    become zero; positive values are not upper-clipped, matching the reference.
    """

    if raw_hsr.shape != RAW_SHAPE:
        raise PreparationError(
            f"raw field shape {raw_hsr.shape}; expected {RAW_SHAPE}"
        )
    x = np.asarray(raw_hsr, dtype=np.float32)

    # Exact crop/pad sequence from the public RegionTransform implementation.
    x = x[: REGION_Y2 + 1, REGION_X1 :]
    x = np.pad(
        x,
        ((abs(REGION_Y1), 0), (0, REGION_X2 - RAW_WIDTH + 1)),
        mode="constant",
        constant_values=OUTSIDE_DOMAIN,
    )
    x = np.pad(
        x,
        ((0, 7), (21, 0)),
        mode="constant",
        constant_values=OUTSIDE_DOMAIN,
    )
    if x.shape != (SQUARE_SIZE, SQUARE_SIZE):
        raise PreparationError(
            f"official square construction produced {x.shape}; expected "
            f"{(SQUARE_SIZE, SQUARE_SIZE)}"
        )

    # Uniform aligned decimation, not pooling or interpolation.
    x = x[::UNIFORM_STRIDE, ::UNIFORM_STRIDE]
    if x.shape != (SUBSAMPLED_SIZE, SUBSAMPLED_SIZE):
        raise PreparationError(
            f"uniform subsampling produced {x.shape}; expected "
            f"{(SUBSAMPLED_SIZE, SUBSAMPLED_SIZE)}"
        )
    x = x[
        TARGET_START : TARGET_START + TARGET_SIZE,
        TARGET_START : TARGET_START + TARGET_SIZE,
    ]

    x = np.maximum(x, np.float32(0.0)) / NORMALIZATION_DIVISOR
    x = x[::-1].copy()
    if x.shape != (TARGET_SIZE, TARGET_SIZE) or x.dtype != np.float32:
        raise PreparationError(
            f"prepared frame contract mismatch: shape={x.shape}, dtype={x.dtype}"
        )
    return x


def official_region_transform(raw_hsr: np.ndarray) -> np.ndarray:
    """Memory-efficient transform exactly equivalent to the published one.

    After the published padding, decimation, and center crop, output columns
    0--230 correspond to raw columns 464, 472, ..., 2304.  Columns 231--255
    sample the published outside-domain padding and are therefore zero.  Before
    the final y reversal, rows sample raw rows 187, 195, ..., 2227.
    """

    if raw_hsr.shape != RAW_SHAPE:
        raise PreparationError(
            f"raw field shape {raw_hsr.shape}; expected {RAW_SHAPE}"
        )
    prepared = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    sampled = raw_hsr[187:2228:8, 464:2305:8]
    if sampled.shape != (TARGET_SIZE, 231):
        raise PreparationError(
            f"optimized official sampling produced {sampled.shape}; expected "
            f"{(TARGET_SIZE, 231)}"
        )
    prepared[:, :231] = sampled
    np.maximum(prepared, np.float32(0.0), out=prepared)
    prepared /= NORMALIZATION_DIVISOR
    return prepared[::-1].copy()


def output_raw_index_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw/virtual indices and the source-backed output-pixel mask."""

    # These are the exact indices implied by the authors' crop, padding,
    # decimation, center crop, and final y reversal.
    raw_rows = 2227 - UNIFORM_STRIDE * np.arange(TARGET_SIZE, dtype=np.int32)
    raw_columns = 464 + UNIFORM_STRIDE * np.arange(TARGET_SIZE, dtype=np.int32)
    source_columns = raw_columns < RAW_WIDTH
    source_index_valid = np.broadcast_to(
        source_columns[None, :], (TARGET_SIZE, TARGET_SIZE)
    ).copy()
    return raw_rows, raw_columns, source_index_valid


def build_geographic_grid(
    coordinate_netcdf: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build and validate the final exPreCast grid's two-dimensional lon/lat.

    Pixel selection remains exactly the authors' published transform.  The
    coordinates are labelled using the provider grid's zero-based center from
    the supplied 500-m lon/lat NetCDF.  This avoids treating the documented
    one-based center pair as NumPy indices, which would shift labels by 500 m.
    """

    try:
        import xarray as xr
        from pyproj import CRS, Transformer
    except ImportError as error:
        raise PreparationError(
            "coordinate auditing requires xarray and pyproj"
        ) from error

    coordinate_netcdf = coordinate_netcdf.expanduser().resolve()
    if not coordinate_netcdf.is_file():
        raise PreparationError(f"coordinate NetCDF not found: {coordinate_netcdf}")
    with xr.open_dataset(coordinate_netcdf) as dataset:
        if "lon" not in dataset or "lat" not in dataset:
            raise PreparationError(
                f"coordinate NetCDF lacks lon/lat variables: {coordinate_netcdf}"
            )
        source_lon = np.asarray(dataset["lon"].values)
        source_lat = np.asarray(dataset["lat"].values)
        if source_lon.shape != RAW_SHAPE or source_lat.shape != RAW_SHAPE:
            raise PreparationError(
                f"coordinate grids have shapes {source_lon.shape}/{source_lat.shape}; "
                f"expected {RAW_SHAPE}"
            )
        attributes = dict(dataset.attrs)

    center_x = float(attributes.get("map_sx", 1120.0))
    center_y = float(attributes.get("map_sy", 1680.0))
    central_longitude = float(attributes.get("map_slon", 126.0))
    central_latitude = float(attributes.get("map_slat", 38.0))
    expected_center = (1120.0, 1680.0, 126.0, 38.0)
    found_center = (center_x, center_y, central_longitude, central_latitude)
    if not np.allclose(found_center, expected_center, rtol=0, atol=1e-6):
        raise PreparationError(
            "unexpected provider coordinate center: "
            f"found {found_center}, expected {expected_center}"
        )

    raw_rows, raw_columns, source_index_valid = output_raw_index_axes()
    row_grid, column_grid = np.meshgrid(raw_rows, raw_columns, indexing="ij")
    x_metres = (column_grid.astype(np.float64) - center_x) * SOURCE_RESOLUTION_METRES
    y_metres = (row_grid.astype(np.float64) - center_y) * SOURCE_RESOLUTION_METRES
    inverse = Transformer.from_crs(
        CRS.from_proj4(LCC_PROJ4), "EPSG:4326", always_xy=True
    )
    longitude, latitude = inverse.transform(x_metres, y_metres)
    longitude = np.asarray(longitude, dtype=np.float32)
    latitude = np.asarray(latitude, dtype=np.float32)

    # Validate every source-backed coordinate directly against the provider's
    # 500-m lon/lat arrays.  The remaining 25 columns are the authors' padded
    # virtual grid and therefore have no source-array indices.
    valid_columns = raw_columns[raw_columns < RAW_WIDTH]
    sampled_lon = source_lon[np.ix_(raw_rows, valid_columns)].astype(np.float32)
    sampled_lat = source_lat[np.ix_(raw_rows, valid_columns)].astype(np.float32)
    projected_lon = longitude[:, : len(valid_columns)]
    projected_lat = latitude[:, : len(valid_columns)]
    max_lon_error = float(np.nanmax(np.abs(projected_lon - sampled_lon)))
    max_lat_error = float(np.nanmax(np.abs(projected_lat - sampled_lat)))
    tolerance_degrees = 2.0e-5
    if max_lon_error > tolerance_degrees or max_lat_error > tolerance_degrees:
        raise PreparationError(
            "final exPreCast grid does not align with the provider coordinate "
            f"NetCDF: max errors lon={max_lon_error:.8g}, "
            f"lat={max_lat_error:.8g} degrees"
        )

    arrays = {
        "longitude": longitude,
        "latitude": latitude,
        "source_index_valid_mask": source_index_valid,
        "raw_row_index": raw_rows,
        "raw_column_index": raw_columns,
    }
    audit: dict[str, object] = {
        "coordinate_netcdf": str(coordinate_netcdf),
        "coordinate_netcdf_sha256": file_sha256(coordinate_netcdf),
        "coordinate_center_zero_based_xy": [center_x, center_y],
        "projection": LCC_PROJ4,
        "source_resolution_metres": SOURCE_RESOLUTION_METRES,
        "source_backed_columns": int(len(valid_columns)),
        "published_padding_columns": int(TARGET_SIZE - len(valid_columns)),
        "max_longitude_validation_error_degrees": max_lon_error,
        "max_latitude_validation_error_degrees": max_lat_error,
        "validation_tolerance_degrees": tolerance_degrees,
        "final_longitude_bounds_degrees_east": [
            float(np.nanmin(longitude)),
            float(np.nanmax(longitude)),
        ],
        "final_latitude_bounds_degrees_north": [
            float(np.nanmin(latitude)),
            float(np.nanmax(latitude)),
        ],
        "final_bounds_semantics": (
            "Extrema of the two-dimensional curvilinear LCC lon/lat arrays; "
            "not the edges of a rectilinear longitude-latitude box."
        ),
        "source_backed_longitude_bounds_degrees_east": [
            float(np.nanmin(longitude[source_index_valid])),
            float(np.nanmax(longitude[source_index_valid])),
        ],
        "source_backed_latitude_bounds_degrees_north": [
            float(np.nanmin(latitude[source_index_valid])),
            float(np.nanmax(latitude[source_index_valid])),
        ],
        "paper_intermediate_bounds": {
            "longitude_degrees_east": [120.5, 137.5],
            "latitude_degrees_north": [29.0, 42.0],
            "note": (
                "These coordinates define the broad pre-padding crop, not "
                "the final model-grid corners."
            ),
        },
        "published_geometry_trace": {
            "padded_intermediate_shape": [SQUARE_SIZE, SQUARE_SIZE],
            "padded_intermediate_side_km": (
                SQUARE_SIZE * SOURCE_RESOLUTION_METRES / 1000.0
            ),
            "subsampled_shape": [SUBSAMPLED_SIZE, SUBSAMPLED_SIZE],
            "subsampled_resolution_km": (
                UNIFORM_STRIDE * SOURCE_RESOLUTION_METRES / 1000.0
            ),
            "central_crop_shape": [TARGET_SIZE, TARGET_SIZE],
            "central_crop_side_km": (
                TARGET_SIZE
                * UNIFORM_STRIDE
                * SOURCE_RESOLUTION_METRES
                / 1000.0
            ),
            "discarded_4km_samples_before_after": [
                TARGET_START,
                SUBSAMPLED_SIZE - TARGET_START - TARGET_SIZE,
            ],
            "discarded_km_before_after": [
                TARGET_START
                * UNIFORM_STRIDE
                * SOURCE_RESOLUTION_METRES
                / 1000.0,
                (SUBSAMPLED_SIZE - TARGET_START - TARGET_SIZE)
                * UNIFORM_STRIDE
                * SOURCE_RESOLUTION_METRES
                / 1000.0,
            ],
            "authors_rectangular_plotting_extent_lonlat": [
                122.54572384823177,
                134.47248320969067,
                31.00876360780117,
                40.24300283359723,
            ],
            "plotting_extent_note": (
                "The authors use this rectangular imshow approximation in "
                "kma_postprocessing.ipynb; it is not an exact curvilinear "
                "coordinate envelope."
            ),
        },
    }
    return arrays, audit


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{uuid.uuid4().hex}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def output_path(output_root: Path, timestamp: datetime) -> Path:
    return (
        output_root
        / f"{timestamp:%Y}"
        / f"{timestamp:%m}"
        / f"{timestamp:%d}"
        / f"{timestamp:%Y%m%d%H%M}.tiff"
    )


def validate_tiff(path: Path) -> np.ndarray:
    try:
        value = tifffile.imread(path)
    except Exception as error:
        raise PreparationError(f"cannot read existing TIFF {path}: {error}") from error
    if value.shape != (TARGET_SIZE, TARGET_SIZE) or value.dtype != np.float32:
        raise PreparationError(
            f"existing TIFF {path} has shape={value.shape}, dtype={value.dtype}; "
            f"expected {(TARGET_SIZE, TARGET_SIZE)} float32"
        )
    if not np.isfinite(value).all() or float(value.min()) < 0:
        raise PreparationError(f"existing TIFF {path} contains invalid values")
    return value


def write_tiff_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{uuid.uuid4().hex}.tiff")
    try:
        tifffile.imwrite(
            temporary,
            np.asarray(value, dtype=np.float32),
            photometric="minisblack",
            metadata=None,
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def process_one(
    source: SourceFrame,
    *,
    output_root: Path,
    resume: bool,
    overwrite: bool,
    hash_sources: bool,
    skip_invalid_source: bool,
) -> FrameResult:
    destination = output_path(output_root, source.timestamp)
    if destination.exists() and resume and not overwrite:
        prepared = validate_tiff(destination)
        return FrameResult(
            timestamp_kst=source.timestamp.isoformat(timespec="minutes"),
            source=str(source.path),
            output=str(destination),
            status="reused",
            container="not_reopened",
            compressed_bytes=source.path.stat().st_size,
            output_min=float(prepared.min()),
            output_max=float(prepared.max()),
            positive_pixels=int(np.count_nonzero(prepared > 0)),
            outside_domain_pixels=-1,
            no_echo_pixels=-1,
            source_sha256="",
            error="",
        )
    if destination.exists() and not overwrite:
        raise PreparationError(
            f"output exists: {destination}; use --resume or --overwrite"
        )

    stat_before = source.path.stat()
    if (
        stat_before.st_size != source.size_bytes
        or stat_before.st_mtime_ns != source.mtime_ns
    ):
        raise PreparationError(
            f"source changed after inventory snapshot: {source.path}; "
            "wait for the raw-data copy to finish and rerun"
        )
    try:
        raw, container, compressed_bytes, file_bytes = read_hsr(source.path)
    except PreparationError as error:
        message = f"{source.path}: {error}"
        if not skip_invalid_source:
            raise PreparationError(message) from error
        return FrameResult(
            timestamp_kst=source.timestamp.isoformat(timespec="minutes"),
            source=str(source.path),
            output="",
            status="invalid_source_skipped",
            container="invalid_source",
            compressed_bytes=stat_before.st_size,
            output_min=float("nan"),
            output_max=float("nan"),
            positive_pixels=-1,
            outside_domain_pixels=-1,
            no_echo_pixels=-1,
            source_sha256=(file_sha256(source.path) if hash_sources else ""),
            error=message,
        )
    stat_after = source.path.stat()
    if (
        stat_after.st_size != source.size_bytes
        or stat_after.st_mtime_ns != source.mtime_ns
    ):
        raise PreparationError(
            f"source changed while it was being read: {source.path}; "
            "wait for the raw-data copy to finish and rerun"
        )
    prepared = official_region_transform(raw)
    write_tiff_atomic(destination, prepared)
    # Verify the exact on-disk contract before reporting success.
    roundtrip = validate_tiff(destination)
    if not np.array_equal(prepared, roundtrip):
        raise PreparationError(f"TIFF round-trip changed values: {destination}")
    return FrameResult(
        timestamp_kst=source.timestamp.isoformat(timespec="minutes"),
        source=str(source.path),
        output=str(destination),
        status="written",
        container=container,
        compressed_bytes=compressed_bytes,
        output_min=float(prepared.min()),
        output_max=float(prepared.max()),
        positive_pixels=int(np.count_nonzero(prepared > 0)),
        outside_domain_pixels=int(np.count_nonzero(raw == OUTSIDE_DOMAIN)),
        no_echo_pixels=int(np.count_nonzero(raw == NO_ECHO)),
        source_sha256=sha256_bytes(file_bytes) if hash_sources else "",
        error="",
    )


def _worker(
    arguments: tuple[SourceFrame, str, bool, bool, bool, bool]
) -> FrameResult:
    (
        source,
        output_root_text,
        resume,
        overwrite,
        hash_sources,
        skip_invalid_source,
    ) = arguments
    return process_one(
        source,
        output_root=Path(output_root_text),
        resume=resume,
        overwrite=overwrite,
        hash_sources=hash_sources,
        skip_invalid_source=skip_invalid_source,
    )


def iter_expected_slots(start: datetime, stop: datetime) -> Iterable[datetime]:
    current = start
    while current <= stop:
        yield current
        current += TEN_MINUTES


def year_slots(year: int, minute_phase: int) -> set[datetime]:
    start = datetime(year, 1, 1, 0, minute_phase)
    stop = datetime(year + 1, 1, 1, 0, minute_phase) - TEN_MINUTES
    return set(iter_expected_slots(start, stop))


def coverage_summary(
    sources: Sequence[SourceFrame],
    *,
    selected_years: Sequence[int],
    minute_phase: int,
    limited: bool,
) -> dict[str, object]:
    timestamps = {item.timestamp for item in sources}
    first = min(timestamps)
    last = max(timestamps)
    span_expected = set(iter_expected_slots(first, last))
    summary: dict[str, object] = {
        "first_timestamp_kst": first.isoformat(timespec="minutes"),
        "last_timestamp_kst": last.isoformat(timespec="minutes"),
        "selected_frames": len(timestamps),
        "expected_slots_within_observed_span": len(span_expected),
        "missing_slots_within_observed_span": len(span_expected - timestamps),
        "valid_contiguous_7_frame_input_anchors": sum(
            all(timestamp - step * TEN_MINUTES in timestamps for step in range(7))
            for timestamp in timestamps
        ),
        "minute_phase": minute_phase,
        "limited_run": limited,
    }
    if not limited:
        expected_full: set[datetime] = set()
        for year in selected_years:
            expected_full.update(year_slots(year, minute_phase))
        summary.update(
            {
                "expected_slots_in_selected_full_years": len(expected_full),
                "missing_slots_in_selected_full_years": len(expected_full - timestamps),
            }
        )
    return summary


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_frame_index(path: Path, results: Sequence[FrameResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    fields = list(FrameResult.__dataclass_fields__)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    coordinate = parser.add_mutually_exclusive_group(required=True)
    coordinate.add_argument(
        "--coordinate-netcdf",
        type=Path,
        help=(
            "provider 500-m lon/lat NetCDF used to verify the published crop "
            "and write grid_coordinates.npz"
        ),
    )
    coordinate.add_argument(
        "--skip-coordinate-audit",
        action="store_true",
        help="omit lon/lat validation and grid_coordinates.npz (not recommended)",
    )
    parser.add_argument(
        "--years",
        help="comma-separated years/ranges; inferred from source names if omitted",
    )
    parser.add_argument(
        "--minute-phase",
        type=int,
        choices=(0,),
        default=0,
        help="the shared manuscript input uses phase 0",
    )
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1)
    )
    parser.add_argument("--limit", type=int, help="global frame limit for smoke tests")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-sources", action="store_true")
    parser.add_argument(
        "--skip-invalid-sources",
        action="store_true",
        help=(
            "treat malformed source payloads as explicitly recorded missing "
            "frames instead of aborting; frames.csv and manifest.json retain "
            "their timestamps, paths and errors"
        ),
    )
    parser.add_argument("--require-complete-year", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--inventory-stability-seconds",
        type=float,
        default=5.0,
        help=(
            "wait and rescan before processing; abort if files are still "
            "being copied (0 disables the check)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise PreparationError("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise PreparationError("--limit must be positive")
    if args.progress_every < 1:
        raise PreparationError("--progress-every must be positive")
    if args.inventory_stability_seconds < 0:
        raise PreparationError("--inventory-stability-seconds cannot be negative")
    if args.resume and args.overwrite:
        raise PreparationError("--resume and --overwrite are mutually exclusive")

    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    coordinate_arrays: dict[str, np.ndarray] | None = None
    coordinate_audit: dict[str, object] | None = None
    if not args.skip_coordinate_audit:
        coordinate_arrays, coordinate_audit = build_geographic_grid(
            args.coordinate_netcdf
        )
    requested_years = parse_years(args.years)
    sources, discovery = discover_sources(
        raw_root, years=requested_years, minute_phase=args.minute_phase
    )
    if args.inventory_stability_seconds:
        print(
            "checking that the raw inventory is stable for "
            f"{args.inventory_stability_seconds:g} s ..."
        )
        time.sleep(args.inventory_stability_seconds)
        sources_after, discovery_after = discover_sources(
            raw_root, years=requested_years, minute_phase=args.minute_phase
        )
        if (
            discovery_after["inventory_sha256"]
            != discovery["inventory_sha256"]
            or sources_after != sources
        ):
            raise PreparationError(
                "raw inventory changed during the stability check; the copy "
                "is probably still running. Wait for it to finish before "
                "preprocessing."
            )
    inferred_years = tuple(sorted({item.timestamp.year for item in sources}))
    selected_years = requested_years or inferred_years
    coverage_before_limit = coverage_summary(
        sources,
        selected_years=selected_years,
        minute_phase=args.minute_phase,
        limited=False,
    )
    if args.require_complete_year and int(
        coverage_before_limit["missing_slots_in_selected_full_years"]
    ):
        raise PreparationError(
            "selected raw files do not cover every official ten-minute slot: "
            f"{coverage_before_limit['missing_slots_in_selected_full_years']:,} "
            "slots are missing"
        )
    if args.limit is not None:
        sources = sources[: args.limit]

    print(f"raw root: {raw_root}")
    print(f"output root: {output_root}")
    if coordinate_audit is not None:
        print(
            "final model-grid bounds: lon "
            f"{coordinate_audit['final_longitude_bounds_degrees_east']}, lat "
            f"{coordinate_audit['final_latitude_bounds_degrees_north']}"
        )
        print(
            "coordinate validation max error (deg): lon "
            f"{coordinate_audit['max_longitude_validation_error_degrees']:.3g}, "
            "lat "
            f"{coordinate_audit['max_latitude_validation_error_degrees']:.3g}"
        )
    print(f"minute phase: {args.minute_phase} ({'official' if args.minute_phase == 0 else 'adapted'})")
    print(f"selected frames: {len(sources):,}")
    print(
        "source coverage: "
        f"{coverage_before_limit['first_timestamp_kst']} to "
        f"{coverage_before_limit['last_timestamp_kst']}"
    )
    print(
        "missing official slots within source span: "
        f"{coverage_before_limit['missing_slots_within_observed_span']:,}"
    )
    if args.dry_run:
        print("dry run: no TIFF files were written")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    coordinate_path: Path | None = None
    if coordinate_arrays is not None:
        coordinate_path = output_root / "grid_coordinates.npz"
        write_npz_atomic(coordinate_path, coordinate_arrays)
    jobs = [
        (
            source,
            str(output_root),
            args.resume,
            args.overwrite,
            args.hash_sources,
            args.skip_invalid_sources,
        )
        for source in sources
    ]
    results: list[FrameResult] = []
    if args.workers == 1:
        iterator = map(_worker, jobs)
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if result.status == "invalid_source_skipped":
                print(f"skipped invalid source: {result.error}")
            if index % args.progress_every == 0 or index == len(jobs):
                print(f"processed {index:,}/{len(jobs):,}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, result in enumerate(executor.map(_worker, jobs), start=1):
                results.append(result)
                if result.status == "invalid_source_skipped":
                    print(f"skipped invalid source: {result.error}")
                if index % args.progress_every == 0 or index == len(jobs):
                    print(f"processed {index:,}/{len(jobs):,}")

    results.sort(key=lambda item: item.timestamp_kst)
    invalid_results = [
        item for item in results if item.status == "invalid_source_skipped"
    ]
    index_path = output_root / "frames.csv"
    write_frame_index(index_path, results)
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "purpose": "shared 4-km/10-min HSR inputs for manuscript forecast routes",
        "reference_preprocessing": REFERENCE_URL,
        "reference_paper": REFERENCE_PAPER,
        "implementation": str(script_path),
        "implementation_sha256": file_sha256(script_path),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "timezone": "Asia/Seoul",
        "coordinate_audit": coordinate_audit,
        "discovery": discovery,
        "coverage_before_limit": coverage_before_limit,
        "processed_coverage": coverage_summary(
            sources,
            selected_years=selected_years,
            minute_phase=args.minute_phase,
            limited=args.limit is not None,
        ),
        "raw_contract": {
            "shape": list(RAW_SHAPE),
            "dtype": "little-endian int16",
            "unit": "100 * dBZ",
            "archive_header_bytes": ARCHIVE_HEADER_BYTES,
            "api_header_bytes": API_HEADER_BYTES,
            "outside_domain": OUTSIDE_DOMAIN,
            "no_echo": NO_ECHO,
            "gzip_detected_by_magic": True,
        },
        "prepared_contract": {
            "shape": [TARGET_SIZE, TARGET_SIZE],
            "dtype": "float32 TIFF",
            "spatial_resolution_km": 4,
            "temporal_resolution_minutes": 10,
            "minute_phase": args.minute_phase,
            "official_minute_phase": args.minute_phase == 0,
            "normalization": "max(raw_100xdbz, 0) / 10000",
            "physical_decode": "dBZ = 100 * value",
            "spatial_transform": (
                "published crop/pad; aligned x[::8,::8] uniform subsampling; "
                "central 256x256 crop; reverse y axis"
            ),
            "path_template": "YYYY/MM/DD/YYYYMMDDHHMM.tiff",
            "grid_coordinates": (
                coordinate_path.name if coordinate_path is not None else None
            ),
        },
        "run": {
            "workers": args.workers,
            "limit": args.limit,
            "resume": args.resume,
            "overwrite": args.overwrite,
            "hash_sources": args.hash_sources,
            "skip_invalid_sources": args.skip_invalid_sources,
            "written_frames": sum(item.status == "written" for item in results),
            "reused_frames": sum(item.status == "reused" for item in results),
            "invalid_source_frames": len(invalid_results),
        },
        "invalid_sources": [
            {
                "timestamp_kst": item.timestamp_kst,
                "source": item.source,
                "compressed_bytes": item.compressed_bytes,
                "error": item.error,
            }
            for item in invalid_results
        ],
        "frame_index": index_path.name,
    }
    manifest_path = output_root / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    print(f"wrote: {index_path}")
    print(f"wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
