"""Read-only, HSR-only NowcastNet samples on the existing official 4-km grid.

Optional monthly radar-array caches are supported. Cache validity is checked from HSR metadata,
small-payload hashes and NPY headers, without rescanning multi-GB payloads. Missing
cache months use official TIFFs; missing/invalid frames in an existing cache do
not silently fall back to TIFFs. NumPy outputs work with PyTorch default_collate,
but this module itself imports neither PyTorch nor a model.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import tifffile

STEP_NS = 600_000_000_000
NORMALIZATION = "max(raw_100xdbz, 0) / 10000"
CACHE_SCHEMA = "hsr-monthly-cache-v1"


def dbz_to_rate(values, *, normalized=True):
    """HSR dBZ/100 (default) -> MP mm/h, preserving prepared zeros as dry.

With normalized=False, values are physical dBZ; nonpositive dBZ is treated as
dry, consistently with the existing clipped HSR preprocessing. No 0.1-mm/h
threshold is applied.
"""
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or (normalized and np.any(array < 0)):
        raise ValueError("HSR reflectivity must be finite and normalized HSR cannot be negative")
    dbz = array * 100.0 if normalized else array
    result = np.zeros_like(dbz)
    positive = dbz > 0
    with np.errstate(over="raise", invalid="raise"):
        result[positive] = np.power(np.power(10.0, dbz[positive] / 10.0) / 200.0, 1.0 / 1.6)
        result = result.astype(np.float32)
    return result


def rate_to_dbz_numpy(rates, *, normalized=True):
    """MP mm/h -> clipped HSR dBZ/100 (default), or physical dBZ.

Zero rain maps to zero. Positive rates whose implied dBZ is negative also map
to zero, matching prepared HSR; those very small rates cannot be round-tripped.
"""
    array = np.asarray(rates, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Rain rates must be finite and nonnegative")
    result = np.zeros_like(array)
    positive = array > 0
    result[positive] = np.maximum(10.0 * (np.log10(200.0) + 1.6 * np.log10(array[positive])), 0.0)
    if normalized:
        result /= 100.0
    return result.astype(np.float32)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _month_times(month):
    start = np.datetime64(f"{month[:4]}-{month[4:]}", "M")
    first, end = start.astype("datetime64[ns]").astype(np.int64), (start + 1).astype("datetime64[ns]").astype(np.int64)
    return np.arange(first, end, STEP_NS, dtype=np.int64)


def _integer_axis(values, label):
    result = np.asarray(values)
    if result.ndim != 1 or not len(result) or result.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a nonempty one-dimensional integer KST-nanosecond axis")
    result = result.astype(np.int64)
    if np.any(np.diff(result) <= 0) or np.any(result % STEP_NS):
        raise ValueError(f"{label} must be unique, ordered and on exact 10-minute KST slots")
    return result


def _grid(root, shape):
    manifest_path, coordinate_path = root / "manifest.json", root / "grid_coordinates.npz"
    manifest = _read_json(manifest_path)
    contract = manifest.get("prepared_contract", {})
    expected = {"shape": list(shape), "dtype": "float32 TIFF", "normalization": NORMALIZATION,
                "spatial_resolution_km": 4, "temporal_resolution_minutes": 10, "minute_phase": 0}
    if manifest.get("timezone") != "Asia/Seoul" or any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("HSR manifest grid/normalization/time contract differs from the requested model")
    with np.load(coordinate_path, allow_pickle=False) as coordinates:
        mask = np.array(coordinates["source_index_valid_mask"], copy=True)
        if mask.shape != shape or mask.dtype != bool or not mask.any():
            raise ValueError("HSR source mask must be nonempty boolean [H,W]")
        for name, limit in (("latitude", 90), ("longitude", 180)):
            values = coordinates[name]
            if values.shape != shape or not np.isfinite(values[mask]).all() or np.any(np.abs(values[mask]) > limit):
                raise ValueError("HSR coordinates do not match the physical grid")
    return mask, {"hsr_manifest": {"path": str(manifest_path), "sha256": _hash(manifest_path)},
                  "hsr_coordinates": {"path": str(coordinate_path), "sha256": _hash(coordinate_path)},
                  "spatial_transform": contract.get("spatial_transform", "prepared grid retained without transpose or flip")}


def _cached_month(folder, month, shape, mask, grid_identity):
    complete = _read_json(folder / "complete.json")
    identity = complete.get("identity", {})
    if (complete.get("schema") != CACHE_SCHEMA or identity.get("schema") != CACHE_SCHEMA
            or identity.get("month") != month or complete.get("radar_normalization") != NORMALIZATION):
        raise ValueError(f"Completed HSR cache contract changed: {month}")
    for key in ("hsr_manifest", "hsr_coordinates"):
        if identity.get(key, {}).get("sha256") != grid_identity[key]["sha256"]:
            raise ValueError(f"Cached HSR source {key} differs from current grid/manifest")
    times = _month_times(month)
    if identity.get("times_sha256") != hashlib.sha256(times.tobytes()).hexdigest():
        raise ValueError("Cached monthly time identity changed")
    arrays = {}
    expected = {"times_kst_ns.npy": (times.shape, np.dtype("int64")),
                "radar_valid.npy": (times.shape, np.dtype("bool")),
                "spatial_mask.npy": (shape, np.dtype("bool")),
                "radar.npy": ((len(times), *shape), np.dtype("float32"))}
    for name, (wanted_shape, dtype) in expected.items():
        path = folder / name
        record = complete.get("files", {}).get(name, {})
        if (not path.is_file() or path.stat().st_size != record.get("size")
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))):
            raise ValueError(f"Missing or changed cached HSR payload: {path}")
        if name != "radar.npy" and _hash(path) != record["sha256"]:
            raise ValueError(f"Cached HSR small-payload hash changed: {name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != wanted_shape or value.dtype != dtype:
            raise ValueError(f"Cached HSR array shape/dtype changed: {name}")
        if name != "radar.npy":
            arrays[name] = np.array(value, copy=True)
        if getattr(value, "_mmap", None) is not None:
            value._mmap.close()
    if not np.array_equal(arrays["times_kst_ns.npy"], times) or not np.array_equal(arrays["spatial_mask.npy"], mask):
        raise ValueError("Cached HSR timestamps/mask differ from official source")
    if complete.get("shape") != [len(times), *shape]:
        raise ValueError("Completed HSR shape declaration changed")
    return {"backend": "monthly_radar_memmap", "folder": folder, "times": times,
            "available": arrays["radar_valid.npy"], "source": {
                "month": month, "backend": "monthly_radar_memmap", "folder": str(folder),
                "complete_sha256": _hash(folder / "complete.json"),
                "hsr_payloads": {name: complete["files"][name] for name in expected},
                "large_radar_payload_hash_recomputed": False}}


def _tiff_month(root, month):
    times = _month_times(month)
    paths = {}
    for path in sorted((root / month[:4] / month[4:]).glob("*/*.tiff")):
        stem = path.stem
        if not re.fullmatch(r"\d{12}", stem) or not stem.startswith(month) or path.parent.name != stem[6:8]:
            raise ValueError(f"Unexpected HSR TIFF path: {path}")
        stamp = int(np.datetime64(f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}T{stem[8:10]}:{stem[10:12]}", "ns").astype(np.int64))
        if stamp % STEP_NS or stamp in paths:
            raise ValueError(f"HSR TIFF must be unique and phase-0: {path}")
        paths[stamp] = path
    if not paths:
        raise FileNotFoundError(f"No completed HSR cache or official TIFFs for {month}")
    available = np.isin(times, np.fromiter(paths, dtype=np.int64))
    return {"backend": "official_tiff", "paths": paths, "times": times, "available": available,
            "source": {"month": month, "backend": "official_tiff", "folder": str(root / month[:4] / month[4:]),
                       "frame_count": len(paths), "timestamps_sha256": hashlib.sha256(np.array(sorted(paths), dtype=np.int64).tobytes()).hexdigest()}}


def _common_test_issues(config):
    """Read a user-supplied common issue axis, never an author workspace."""
    if not config.get("common_test_issues"):
        raise ValueError("Set common_test_issues to an explicit NPY common issuance axis")
    path = _resolve(Path(config["project_root"]), config["common_test_issues"])
    return np.load(path, allow_pickle=False), {"path": str(path), "sha256": _hash(path)}


class HSRDataset:
    """Seven observed HSR-MP frames -> 18 future HSR-MP frames, in mm/h.

Train/validation selection is based solely on exact HSR availability. The test
axis defaults to the published common axis. The separate field-only workflow
explicitly uses all HSR-eligible windows, without gauge/common-axis imports.
For prediction, require_future=False never reads future image values and never
conditions sample selection on future HSR availability. Explicit missing input
windows fail rather than being silently dropped or imputed.
"""
    def __init__(self, config, split, require_future=True, issues=None, max_issues=None):
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        if not isinstance(require_future, (bool, np.bool_)):
            raise ValueError("require_future must be boolean")
        self.config, self.split, self.require_future = dict(config), split, bool(require_future)
        test_policy = config.get("test_issue_policy", "published_common_axis")
        if test_policy not in ("published_common_axis", "hsr_available_windows"):
            raise ValueError("Unknown test issue policy")
        if test_policy == "hsr_available_windows" and config.get("schema") != "nowcastnet-hsr-field-v5":
            raise ValueError("HSR-only test windows require the explicit field-only schema")
        self.project_root = Path(config["project_root"]).expanduser().resolve()
        self.radar_root = _resolve(self.project_root, config["radar_root"])
        self.cache_root = _resolve(self.project_root, config["hsr_cache_root"])
        model = config["model"]
        self.rain_rate_cap = model.get("rain_rate_cap", 128.0)
        if (isinstance(self.rain_rate_cap, bool) or not isinstance(self.rain_rate_cap, (int, float))
                or not np.isfinite(self.rain_rate_cap) or self.rain_rate_cap <= 0):
            raise ValueError("model.rain_rate_cap must be a finite positive mm/h value")
        self.rain_rate_cap = float(self.rain_rate_cap)
        size = model["grid_size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("model.grid_size must be a positive integer")
        self.shape = (size, size)
        if model.get("input_steps", 7) != 7 or model.get("output_steps", 18) != 18:
            raise ValueError("HSR contract requires 7 input and 18 future 10-minute steps")
        splits = {name: config[name + "_years"] for name in ("train", "validation", "test")}
        if any(not values or any(type(y) is not int for y in values) or len(set(values)) != len(values) for values in splits.values()):
            raise ValueError("Split years must be nonempty unique integer lists")
        if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("Training, validation and test years must not overlap")
        months = config.get("months", [6, 7, 8, 9])
        if not months or any(type(m) is not int or not 1 <= m <= 12 for m in months) or len(set(months)) != len(months):
            raise ValueError("months must be unique integers in 1..12")
        self.months = [f"{year:04d}{month:02d}" for year in sorted(splits[split]) for month in sorted(months)]
        self.spatial_mask, grid_identity = _grid(self.radar_root, self.shape)
        self._months = []
        for month in self.months:
            folder = self.cache_root / month
            if (folder / "complete.json").is_file():
                item = _cached_month(folder, month, self.shape, self.spatial_mask, grid_identity)
            elif folder.exists():
                raise FileNotFoundError(f"Incomplete monthly HSR cache is preserved, not silently bypassed: {folder}")
            else:
                item = _tiff_month(self.radar_root, month)
            self._months.append(item)
        self.times = np.concatenate([item["times"] for item in self._months])
        self._available = np.concatenate([item["available"] for item in self._months])
        self._ends = np.cumsum([len(item["times"]) for item in self._months])
        self._starts = np.r_[0, self._ends[:-1]]
        offsets = np.arange(-6, 19 if require_future else 1)
        candidates = np.arange(6, len(self.times) - (18 if require_future else 0), dtype=np.int64)
        good = np.ones(len(candidates), dtype=bool)
        for offset in offsets:
            good &= self.times[candidates + offset] == self.times[candidates] + int(offset) * STEP_NS
            good &= self._available[candidates + offset]
        years = self.times.astype("datetime64[ns]").astype("datetime64[Y]")
        good &= years[candidates - 6] == years[candidates]
        if require_future:
            good &= years[candidates + 18] == years[candidates]
        else:
            # Forecast *labels* remain inside this season/year, even when no
            # future observations are required or inspected for prediction.
            for offset in range(1, 19):
                label = (self.times[candidates] + offset * STEP_NS).astype("datetime64[ns]")
                good &= label.astype("datetime64[Y]") == years[candidates]
                good &= np.isin(label.astype("datetime64[M]").astype(int) % 12 + 1, months)
        stride = config.get(split + "_stride_minutes", 10) if split != "test" else 10
        if type(stride) is not int or stride < 10 or stride % 10:
            raise ValueError("Issuance stride must be a positive multiple of ten minutes")
        good &= self.times[candidates] % (stride * 60_000_000_000) == 0
        eligible = candidates[good]
        canonical = None
        if split == "test" and test_policy == "published_common_axis":
            canonical, common_provenance = _common_test_issues({**config, "project_root": str(self.project_root)})
            canonical = _integer_axis(canonical, "common test issues")
            if issues is None:
                issues = canonical
        else:
            common_provenance = None
        if issues is not None:
            wanted = _integer_axis(issues, "requested issues")
            if canonical is not None and not np.isin(wanted, canonical).all():
                raise ValueError("Requested test issues must belong to the fixed published common axis")
            if not np.isin(wanted, self.times[eligible]).all():
                missing = wanted[~np.isin(wanted, self.times[eligible])]
                raise ValueError(f"Requested {split} windows have missing HSR, cross a year/season boundary, or violate stride: {missing[:5].astype('datetime64[ns]')}")
            self.indices = np.searchsorted(self.times, wanted)
        else:
            self.indices = eligible
        if max_issues is not None:
            if type(max_issues) is not int or max_issues < 1:
                raise ValueError("max_issues must be a positive integer")
            self.indices = self.indices[:max_issues]
        self.issue_times_ns = self.times[self.indices].copy()
        self.issues = self.issue_times_ns
        if not len(self.indices):
            raise ValueError(f"No complete HSR-only {split} windows")
        self._maps, self._frames = OrderedDict(), OrderedDict()
        self.max_open_months = int(config.get("max_open_hsr_months", 2))
        self.max_tiff_frames = int(config.get("worker_cache_frames", 32))
        if self.max_open_months < 1 or self.max_tiff_frames < 1:
            raise ValueError("Read-only cache limits must be positive")
        self.provenance = {"split": split, "years": list(splits[split]), "months": self.months,
                           "grid": grid_identity, "shape": list(self.shape), "supported_cells": int(self.spatial_mask.sum()),
                           "sources": [item["source"] for item in self._months], "require_future": self.require_future,
                           "issue_count": len(self.indices), "issue_times_sha256": hashlib.sha256(self.issue_times_ns.tobytes()).hexdigest(),
                           "eligible_issue_count_before_subset": len(eligible),
                           "canonical_test_issue_count": len(canonical) if canonical is not None else None,
                           "common_test_axis": common_provenance, "issuance_stride_minutes": stride,
                           "test_issue_policy": test_policy,
                           "input_offsets_minutes": list(range(-60, 1, 10)), "target_offsets_minutes": list(range(10, 181, 10)),
                           "units": "mm h-1", "conversion": "HSR dBZ=100*stored; Z=200*R**1.6; stored zero treated dry; invalid-domain zero",
                           "rain_rate_cap_mm_h": self.rain_rate_cap,
                           "orientation": "Existing official TIFF/cache orientation retained; no transpose, flip, interpolation or RRBF mapping.",
                           "selection": "HSR frame availability only; no RRBF availability, gauge data, or other-split samples read.",
                           "large_cache_payload_hashes_recomputed": False,
                           "diagnostic_subset": max_issues is not None or (canonical is not None and len(self.indices) != len(canonical))
                           or (canonical is None and issues is not None and len(self.indices) != len(eligible))}

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"], state["_frames"] = OrderedDict(), OrderedDict()
        return state

    def _read_frame(self, index):
        block = int(np.searchsorted(self._ends, index, side="right"))
        month = self._months[block]
        if not self._available[index]:
            raise ValueError("Unavailable HSR frame cannot be imputed or replaced")
        if month["backend"] == "monthly_radar_memmap":
            if block not in self._maps:
                self._maps[block] = np.load(month["folder"] / "radar.npy", mmap_mode="r", allow_pickle=False)
            self._maps.move_to_end(block)
            value = np.array(self._maps[block][index - self._starts[block]], copy=True)
            while len(self._maps) > self.max_open_months:
                _, old = self._maps.popitem(last=False)
                old._mmap.close()
        else:
            stamp = int(self.times[index])
            if stamp not in self._frames:
                path = month["paths"][stamp]
                value = tifffile.imread(path)
                if value.dtype != np.float32 or value.shape != self.shape:
                    raise ValueError(f"Invalid official HSR TIFF dtype/shape: {path}")
                self._frames[stamp] = value
            self._frames.move_to_end(stamp)
            value = np.array(self._frames[stamp], copy=True)
            while len(self._frames) > self.max_tiff_frames:
                self._frames.popitem(last=False)
        value[~self.spatial_mask] = 0.0
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("Declared valid HSR frame contains nonfinite or negative values on the source mask")
        # The original NowcastNet data preparation caps physical rain rate at
        # 128 mm/h. Apply the configured cap equally to history and targets;
        # rate_scale divides evolution conditioning only, not this physical cap.
        return np.minimum(dbz_to_rate(value), self.rain_rate_cap)

    def __getitem__(self, index):
        issue = int(self.indices[index])
        sample = {"history": np.stack([self._read_frame(i) for i in range(issue - 6, issue + 1)]),
                  "mask": self.spatial_mask.copy(), "issue_ns": int(self.times[issue])}
        if self.require_future:
            sample["target"] = np.stack([self._read_frame(i) for i in range(issue + 1, issue + 19)])
        return sample
