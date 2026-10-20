"""Leakage-safe data contract for the 4-km/10-min vanilla Held-out R2P.

Radar and gauge cadences deliberately remain separate:

* radar: seven official TIFF frames at -60,-50,...,0 min;
* gauge context: the established twelve 5-min histories at -55,...,0 min;
* point target: gauge RN60 ending at +5,+10,...,+180 min.

Training and checkpoint selection construct only the fitting-514 gauge table.
Held-out columns are opened only by the explicit evaluation path.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset


MINUTE_NS = 60 * 1_000_000_000
RADAR_OFFSETS_MIN = np.arange(-60, 1, 10, dtype=np.int16)
GAUGE_OFFSETS_MIN = np.arange(-55, 1, 5, dtype=np.int16)
LEAD_MINUTES = np.arange(5, 181, 5, dtype=np.int16)
PRIMARY_REPORT_LEADS = np.asarray([60, 90, 120, 150, 180], dtype=np.int16)
EXPECTED_MAPPING_SHA256 = "ce2c923865322060d8d3c9e57c7d0b85af39e456ab35739a2156b6951a3c8cf4"
EXPECTED_PACKAGED_MAPPING_SHA256 = "cce9b676bc6baca56bd2f6d41f7664f6121b335e5dbb777a03476f16de6c19e6"
EXPECTED_SPLIT_SHA256 = "87128f117bf8736237ac86cb934192fb32210a48716b84a932a84e5ad84394eb"
EXPECTED_GRID_SHA256 = "c0de0575376ad91ac46cac14e759fe8e2baa0476473f2967d517b6d219bb6b8e"
EXPECTED_STATIONS_SHA256 = "24ad5d4357987ca1f84b30e9575bf27901abbfc82f6708857f111714db104177"
# Coordinate channels and station-attention distances are part of the frozen
# vanilla R2P contract.  They retain the original 2-km project-grid frame even
# though radar patch indices below refer to the replacement 4-km TIFF grid.
ORIGINAL_GRID_LAT_MIN = np.float32(32.40345)
ORIGINAL_GRID_LAT_MAX = np.float32(39.247715)
ORIGINAL_GRID_LON_MIN = np.float32(123.363205)
ORIGINAL_GRID_LON_MAX = np.float32(132.52771)
ORIGINAL_RADAR_RESOLUTION_KM = np.float32(2.0)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _timestamp_from_tiff(path: Path) -> int:
    stem = path.stem
    if len(stem) != 12 or not stem.isdigit():
        raise ValueError(f"unexpected TIFF filename: {path}")
    timestamp = np.datetime64(
        f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}T{stem[8:10]}:{stem[10:12]}",
        "ns",
    )
    return int(timestamp.astype(np.int64))


def _as_ns(values: Sequence | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.issubdtype(array.dtype, np.integer):
        result = array.astype(np.int64, copy=False)
    else:
        result = array.astype("datetime64[ns]").astype(np.int64)
    if np.any(result == np.iinfo(np.int64).min):
        raise ValueError(f"{name} contains NaT")
    return result


def validate_tiff_archive(radar_root: Path) -> dict:
    radar_root = Path(radar_root)
    manifest_path = radar_root / "manifest.json"
    coordinate_path = radar_root / "grid_coordinates.npz"
    if not manifest_path.is_file() or not coordinate_path.is_file():
        raise FileNotFoundError("official TIFF manifest/grid_coordinates.npz is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared = manifest.get("prepared_contract", {})
    expected = {
        "shape": [256, 256],
        "dtype": "float32 TIFF",
        "spatial_resolution_km": 4,
        "temporal_resolution_minutes": 10,
        "minute_phase": 0,
        "normalization": "max(raw_100xdbz, 0) / 10000",
    }
    for key, value in expected.items():
        if prepared.get(key) != value:
            raise RuntimeError(f"TIFF manifest mismatch for {key}: {prepared.get(key)!r}")
    if sha256_file(coordinate_path) != EXPECTED_GRID_SHA256:
        raise RuntimeError("grid_coordinates.npz differs from the frozen official grid")
    with np.load(coordinate_path, allow_pickle=False) as coordinates:
        required = {"longitude", "latitude", "source_index_valid_mask"}
        if not required.issubset(coordinates.files):
            raise RuntimeError("grid coordinate archive is incomplete")
        longitude = np.asarray(coordinates["longitude"], dtype=np.float32)
        latitude = np.asarray(coordinates["latitude"], dtype=np.float32)
        source_valid = np.asarray(coordinates["source_index_valid_mask"], dtype=bool)
    if longitude.shape != (256, 256) or latitude.shape != (256, 256):
        raise RuntimeError("official grid coordinate shape changed")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "coordinate_path": coordinate_path,
        "longitude": longitude,
        "latitude": latitude,
        "source_valid": source_valid,
    }


@dataclass(frozen=True)
class StationContract:
    train: pd.DataFrame
    heldout: pd.DataFrame
    all_stations: pd.DataFrame
    longitude: np.ndarray
    latitude: np.ndarray


def load_station_contract(
    mapping_csv: Path,
    stations_csv: Path,
    split_csv: Path,
    longitude: np.ndarray,
    latitude: np.ndarray,
    source_valid: np.ndarray,
    *,
    enforce_frozen_hashes: bool = True,
) -> StationContract:
    mapping_csv, stations_csv, split_csv = map(
        Path, (mapping_csv, stations_csv, split_csv)
    )
    if enforce_frozen_hashes:
        mapping_hash = sha256_file(mapping_csv)
        # The packaged copy is CSV-content identical to the original but uses
        # portable LF line endings instead of CRLF, hence its distinct byte hash.
        if mapping_hash not in {
            EXPECTED_MAPPING_SHA256,
            EXPECTED_PACKAGED_MAPPING_SHA256,
        }:
            raise RuntimeError(f"frozen mapping contract changed: {mapping_csv}")
        for path, expected in (
            (stations_csv, EXPECTED_STATIONS_SHA256),
            (split_csv, EXPECTED_SPLIT_SHA256),
        ):
            if sha256_file(path) != expected:
                raise RuntimeError(f"frozen contract file changed: {path}")

    mapping = pd.read_csv(mapping_csv)
    stations = pd.read_csv(stations_csv)
    split = pd.read_csv(split_csv, encoding="utf-8-sig")[["station_id", "split"]]
    if len(mapping) != 642 or len(stations) != 642 or len(split) != 642:
        raise RuntimeError("station files must contain exactly 642 rows")
    if any(frame.station_id.duplicated().any() for frame in (mapping, stations, split)):
        raise RuntimeError("station files contain duplicate IDs")
    expected_ids = set(mapping.station_id.astype(int))
    if expected_ids != set(stations.station_id.astype(int)) or expected_ids != set(
        split.station_id.astype(int)
    ):
        raise RuntimeError("mapping, station metadata and frozen split disagree")

    station_columns = [
        "station_id",
        "name",
        "target_col",
        "rn15m_col",
        "iy",
        "ix",
        "lat",
        "lon",
        "split",
    ]
    merged = stations[station_columns].merge(
        split, on="station_id", validate="one_to_one", suffixes=("", "_frozen")
    )
    if "split_frozen" in merged.columns:
        if not np.array_equal(
            merged["split"].astype(str), merged["split_frozen"].astype(str)
        ):
            raise RuntimeError("stations_eval split differs from station_split_v1")
        merged = merged.drop(columns="split").rename(columns={"split_frozen": "split"})
    merged = merged.merge(
        mapping[
            [
                "station_id",
                "exprecast_y",
                "exprecast_x",
                "exprecast_nearest_distance_km",
            ]
        ],
        on="station_id",
        validate="one_to_one",
    )
    y = merged.exprecast_y.to_numpy(dtype=np.int64)
    x = merged.exprecast_x.to_numpy(dtype=np.int64)
    if np.any(y < 0) or np.any(y >= 256) or np.any(x < 0) or np.any(x >= 256):
        raise RuntimeError("mapped station lies outside the official grid")
    if not np.all(source_valid[y, x]):
        raise RuntimeError("mapped station lies on public preprocessing padding")
    if float(merged.exprecast_nearest_distance_km.max()) > 2.761:
        raise RuntimeError("station mapping distance differs from the frozen mapping")
    merged["grid_lat"] = latitude[y, x]
    merged["grid_lon"] = longitude[y, x]
    train = merged.loc[merged.split == "train"].reset_index(drop=True)
    heldout = merged.loc[merged.split == "test"].reset_index(drop=True)
    if len(train) != 514 or len(heldout) != 128:
        raise RuntimeError("frozen split is not 514/128")
    all_stations = pd.concat([train, heldout], ignore_index=True)
    return StationContract(
        train=train,
        heldout=heldout,
        all_stations=all_stations,
        longitude=longitude,
        latitude=latitude,
    )


@dataclass(frozen=True)
class RadarIndex:
    paths: tuple[Path, ...]
    times_ns: np.ndarray
    time_to_position: dict[int, int]


def discover_tiff_index(radar_root: Path, years: Iterable[int]) -> RadarIndex:
    pairs: list[tuple[int, Path]] = []
    for year in map(int, years):
        year_root = Path(radar_root) / f"{year:04d}"
        if not year_root.is_dir():
            raise FileNotFoundError(year_root)
        for path in sorted(year_root.glob("*/*/*.tiff")):
            timestamp = _timestamp_from_tiff(path)
            minute = int(path.stem[-2:])
            if minute % 10 != 0:
                raise RuntimeError(f"non-phase-0 TIFF found in official archive: {path}")
            pairs.append((timestamp, path))
    pairs.sort(key=lambda item: item[0])
    times = np.asarray([item[0] for item in pairs], dtype=np.int64)
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise RuntimeError("TIFF timestamps must be non-empty and strictly increasing")
    return RadarIndex(
        paths=tuple(item[1] for item in pairs),
        times_ns=times,
        time_to_position={int(value): index for index, value in enumerate(times)},
    )


def build_issue_times(
    radar_index: RadarIndex,
    gauge_times_ns: Sequence | np.ndarray,
    *,
    require_target_horizon: bool,
) -> np.ndarray:
    gauge_times = _as_ns(gauge_times_ns, "gauge_times_ns")
    gauge_set = set(map(int, gauge_times))
    radar_set = set(map(int, radar_index.times_ns))
    issues: list[int] = []
    for issue in radar_index.times_ns:
        issue_value = int(issue)
        if all(
            issue_value + int(offset) * MINUTE_NS in radar_set
            for offset in RADAR_OFFSETS_MIN
        ) and all(
            issue_value + int(offset) * MINUTE_NS in gauge_set
            for offset in GAUGE_OFFSETS_MIN
        ):
            if require_target_horizon and not all(
                issue_value + int(lead) * MINUTE_NS in gauge_set
                for lead in LEAD_MINUTES
            ):
                continue
            issues.append(issue_value)
    result = np.asarray(issues, dtype=np.int64)
    if len(result) == 0 or np.any(np.diff(result) <= 0):
        raise RuntimeError("no valid issue times satisfy the R2P input contract")
    return result


@lru_cache(maxsize=1024)
def read_tiff(path_text: str) -> np.ndarray:
    value = np.asarray(tifffile.imread(path_text), dtype=np.float32)
    if value.shape != (256, 256) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid official TIFF: {path_text}")
    if np.any(value < 0):
        raise RuntimeError(f"official TIFF contains negative normalized dBZ: {path_text}")
    return np.ascontiguousarray(value)


def read_gauge_subset(
    csv_path: Path,
    stations: pd.DataFrame,
    years: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read only the explicitly supplied station IDs from the gauge CSV."""

    time_columns = ["Year", "Month", "Day", "Hour", "Minute"]
    target_columns = stations.target_col.astype(str).tolist()
    rn15_columns = stations.rn15m_col.astype(str).tolist()
    requested = time_columns + target_columns + rn15_columns
    frame = pd.read_csv(csv_path, usecols=requested)
    frame = frame.loc[frame.Year.isin(tuple(map(int, years)))].copy()
    times = pd.to_datetime(
        {
            "year": frame.Year,
            "month": frame.Month,
            "day": frame.Day,
            "hour": frame.Hour,
            "minute": frame.Minute,
        },
        errors="raise",
    ).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise RuntimeError("gauge timestamps must be non-empty and strictly increasing")
    rn60 = frame[target_columns].to_numpy(dtype=np.float32, copy=True)
    rn15 = frame[rn15_columns].to_numpy(dtype=np.float32, copy=True)
    return times, rn60, rn15, np.asarray(stations.station_id, dtype=np.int64)


def read_gauge_times(csv_path: Path, years: Iterable[int]) -> np.ndarray:
    columns = ["Year", "Month", "Day", "Hour", "Minute"]
    frame = pd.read_csv(csv_path, usecols=columns)
    frame = frame.loc[frame.Year.isin(tuple(map(int, years)))]
    times = pd.to_datetime(
        {
            "year": frame.Year,
            "month": frame.Month,
            "day": frame.Day,
            "hour": frame.Hour,
            "minute": frame.Minute,
        },
        errors="raise",
    ).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise RuntimeError("gauge timestamps must be non-empty and strictly increasing")
    return times


@dataclass(frozen=True)
class GaugeScaler:
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def denominator(self) -> np.ndarray:
        return np.where(self.maximum - self.minimum > 1e-6, self.maximum - self.minimum, 1.0)

    def scale_features(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.minimum) / self.denominator).astype(np.float32)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return values * float(self.denominator[0]) + float(self.minimum[0])


def build_station_features(
    rn60: np.ndarray,
    rn15: np.ndarray,
    stations: pd.DataFrame,
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    scaler: GaugeScaler | None,
) -> tuple[np.ndarray, GaugeScaler]:
    invalid60 = ~np.isfinite(rn60)
    invalid15 = ~np.isfinite(rn15)
    missing = invalid60 | invalid15
    rn60_filled = np.where(invalid60, 0.0, rn60).astype(np.float32)
    rn15_filled = np.where(invalid15, 0.0, rn15).astype(np.float32)
    lat = stations.lat.to_numpy(dtype=np.float32)
    lon = stations.lon.to_numpy(dtype=np.float32)
    lat_norm = (lat - ORIGINAL_GRID_LAT_MIN) / (
        ORIGINAL_GRID_LAT_MAX - ORIGINAL_GRID_LAT_MIN
    )
    lon_norm = (lon - ORIGINAL_GRID_LON_MIN) / (
        ORIGINAL_GRID_LON_MAX - ORIGINAL_GRID_LON_MIN
    )
    features = np.stack(
        [
            rn60_filled,
            rn15_filled,
            missing.astype(np.float32),
            np.broadcast_to(lat_norm[None], rn60.shape),
            np.broadcast_to(lon_norm[None], rn60.shape),
        ],
        axis=-1,
    ).astype(np.float32)
    if scaler is None:
        minimum = np.zeros(5, dtype=np.float32)
        maximum = np.ones(5, dtype=np.float32)
        observations = features[:, :, :2].reshape(-1, 2)
        minimum[:2] = observations.min(axis=0)
        maximum[:2] = observations.max(axis=0)
        scaler = GaugeScaler(minimum=minimum, maximum=maximum)
    if np.max(np.abs(scaler.minimum[:2])) > 1e-6:
        raise RuntimeError("zero-filled observations must map to normalized zero")
    return scaler.scale_features(features), scaler


def build_train514_and_eval642_features(
    train_rn60: np.ndarray,
    train_rn15: np.ndarray,
    station_contract: StationContract,
    *,
    scaler: GaugeScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Create identical fitting context for V=514 and target-masked V=642.

    Real held-out histories never enter this function.  The appended 128 query
    nodes have zero RN60/RN15, missing=1 and their fixed coordinate channels.
    """

    train_features, _ = build_station_features(
        train_rn60,
        train_rn15,
        station_contract.train,
        station_contract.longitude,
        station_contract.latitude,
        scaler=scaler,
    )
    time_count = train_features.shape[0]
    heldout_count = len(station_contract.heldout)
    heldout60 = np.zeros((time_count, heldout_count), dtype=np.float32)
    heldout15 = np.zeros_like(heldout60)
    heldout_features, _ = build_station_features(
        heldout60,
        heldout15,
        station_contract.heldout,
        station_contract.longitude,
        station_contract.latitude,
        scaler=scaler,
    )
    # Zero observations above are valid by construction, so force the explicit
    # target-masked missing representation used by the original Held-out R2P.
    heldout_features[:, :, 2] = 1.0
    eval_features = np.concatenate([train_features, heldout_features], axis=1)
    if not np.array_equal(eval_features[:, : len(station_contract.train)], train_features):
        raise RuntimeError("fitting-station gauge context changed during eval composition")
    return train_features, eval_features


class R2PWindowDataset(Dataset):
    def __init__(
        self,
        radar_index: RadarIndex,
        gauge_times_ns: np.ndarray,
        station_features: np.ndarray,
        target_rn60: np.ndarray,
        issue_times_ns: np.ndarray,
    ):
        self.radar_index = radar_index
        self.gauge_times_ns = _as_ns(gauge_times_ns, "gauge_times_ns")
        self.station_features = np.asarray(station_features, dtype=np.float32)
        self.target_rn60 = np.asarray(target_rn60, dtype=np.float32)
        self.issue_times_ns = _as_ns(issue_times_ns, "issue_times_ns")
        if self.station_features.shape[:2] != self.target_rn60.shape:
            raise RuntimeError("station feature/target axes disagree")
        if self.station_features.shape[0] != len(self.gauge_times_ns):
            raise RuntimeError("gauge feature time axis mismatch")
        self.gauge_positions = {int(value): index for index, value in enumerate(self.gauge_times_ns)}

    def __len__(self) -> int:
        return len(self.issue_times_ns)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        issue = int(self.issue_times_ns[position])
        radar_paths = [
            self.radar_index.paths[
                self.radar_index.time_to_position[issue + int(offset) * MINUTE_NS]
            ]
            for offset in RADAR_OFFSETS_MIN
        ]
        radar = np.stack([read_tiff(str(path)) for path in radar_paths], axis=0)
        context_rows = [
            self.gauge_positions[issue + int(offset) * MINUTE_NS]
            for offset in GAUGE_OFFSETS_MIN
        ]
        target_rows = [
            self.gauge_positions[issue + int(lead) * MINUTE_NS]
            for lead in LEAD_MINUTES
        ]
        return {
            "radar": torch.from_numpy(np.ascontiguousarray(radar)),
            "station": torch.from_numpy(
                np.ascontiguousarray(self.station_features[context_rows])
            ),
            "target": torch.from_numpy(
                np.ascontiguousarray(self.target_rn60[target_rows].T)
            ),
            "issue_time_ns": torch.tensor(issue, dtype=torch.int64),
        }


__all__ = [
    "EXPECTED_GRID_SHA256",
    "EXPECTED_MAPPING_SHA256",
    "EXPECTED_PACKAGED_MAPPING_SHA256",
    "EXPECTED_SPLIT_SHA256",
    "EXPECTED_STATIONS_SHA256",
    "GAUGE_OFFSETS_MIN",
    "GaugeScaler",
    "LEAD_MINUTES",
    "MINUTE_NS",
    "ORIGINAL_GRID_LAT_MAX",
    "ORIGINAL_GRID_LAT_MIN",
    "ORIGINAL_GRID_LON_MAX",
    "ORIGINAL_GRID_LON_MIN",
    "ORIGINAL_RADAR_RESOLUTION_KM",
    "PRIMARY_REPORT_LEADS",
    "RADAR_OFFSETS_MIN",
    "R2PWindowDataset",
    "StationContract",
    "build_issue_times",
    "build_station_features",
    "build_train514_and_eval642_features",
    "discover_tiff_index",
    "load_station_contract",
    "read_gauge_subset",
    "read_gauge_times",
    "sha256_array",
    "sha256_file",
    "validate_tiff_archive",
]
