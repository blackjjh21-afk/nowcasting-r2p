#!/usr/bin/env python3
"""Leakage-safe June–September 4-km/10-min normalized KMA TIFF windows.

Preserves primary-field input ordering, continuity checks, and chunk-local
sampling. Raw KMA data and prepared TIFF archives are supplied by the user.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


NATIVE_FRAME_MINUTES = 10
SEASON_MONTHS = (6, 7, 8, 9)
DAY_NS = 24 * 60 * 60 * 1_000_000_000


@dataclass(frozen=True)
class CropContract:
    height: int = 256
    width: int = 256
    y0: int | None = None
    x0: int | None = None
    spatial_factor: int = 1
    spatial_reduction: str = "uniform"

    def resolve(self, native_shape: tuple[int, int]) -> "ResolvedCrop":
        native_height, native_width = native_shape
        factor = int(self.spatial_factor)
        if factor < 1:
            raise ValueError("spatial_factor must be >= 1")
        if self.spatial_reduction not in {"uniform", "area"}:
            raise ValueError("spatial_reduction must be 'uniform' or 'area'")
        reduced_height = native_height // factor
        reduced_width = native_width // factor
        height, width = int(self.height), int(self.width)
        if height < 32 or width < 32 or height % 32 or width % 32:
            raise ValueError("crop height and width must be positive multiples of 32")
        if height > reduced_height or width > reduced_width:
            raise ValueError(
                f"crop {(height, width)} exceeds reduced source grid "
                f"{(reduced_height, reduced_width)}; reduce crop or spatial_factor"
            )
        y0 = (reduced_height - height) // 2 if self.y0 is None else int(self.y0)
        x0 = (reduced_width - width) // 2 if self.x0 is None else int(self.x0)
        if y0 < 0 or x0 < 0 or y0 + height > reduced_height or x0 + width > reduced_width:
            raise ValueError(
                f"crop origin {(y0, x0)} and shape {(height, width)} are outside "
                f"reduced grid {(reduced_height, reduced_width)}"
            )
        return ResolvedCrop(
            native_height=native_height,
            native_width=native_width,
            reduced_height=reduced_height,
            reduced_width=reduced_width,
            height=height,
            width=width,
            y0=y0,
            x0=x0,
            spatial_factor=factor,
            spatial_reduction=self.spatial_reduction,
        )


@dataclass(frozen=True)
class ResolvedCrop:
    native_height: int
    native_width: int
    reduced_height: int
    reduced_width: int
    height: int
    width: int
    y0: int
    x0: int
    spatial_factor: int
    spatial_reduction: str

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def exprecast_normalized_to_dbz(value: torch.Tensor) -> torch.Tensor:
    return value * 100.0


def dbz_to_rain_rate(dbz: torch.Tensor) -> torch.Tensor:
    """KMA Z=200 R^1.6 conversion used by the exPreCast evaluation."""

    return torch.pow(torch.pow(10.0, dbz / 10.0) / 200.0, 1.0 / 1.6)


def transform_normalized_spatial(
    frames: torch.Tensor, crop: ResolvedCrop
) -> torch.Tensor:
    """Reduce/crop official TIFF frames that are already normalized dBZ/100."""

    if frames.ndim != 3:
        raise ValueError(f"expected [T,H,W], received {tuple(frames.shape)}")
    value = frames.to(torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("official KMA TIFF contains non-finite values")
    if crop.spatial_factor > 1:
        if crop.spatial_reduction == "uniform":
            value = value[
                :,
                : crop.reduced_height * crop.spatial_factor : crop.spatial_factor,
                : crop.reduced_width * crop.spatial_factor : crop.spatial_factor,
            ]
        else:
            value = F.interpolate(
                value[:, None],
                size=(crop.reduced_height, crop.reduced_width),
                mode="area",
            )[:, 0]
    return value[
        :,
        crop.y0 : crop.y0 + crop.height,
        crop.x0 : crop.x0 + crop.width,
    ].contiguous()


def _timestamp_from_tiff(path: Path) -> np.datetime64:
    stem = path.stem
    if len(stem) != 12 or not stem.isdigit():
        raise ValueError(f"unexpected KMA TIFF name: {path}")
    value = (
        f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}T"
        f"{stem[8:10]}:{stem[10:12]}"
    )
    return np.datetime64(value, "ns")


@lru_cache(maxsize=1024)
def _read_normalized_tiff(path_text: str) -> np.ndarray:
    value = np.asarray(tifffile.imread(path_text), dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"expected 2-D KMA TIFF, got {value.shape}: {path_text}")
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite KMA TIFF values: {path_text}")
    return np.ascontiguousarray(value)


class KmaTiffWindowDataset(Dataset):
    """Continuous primary-contract windows from normalized KMA TIFF archives."""

    def __init__(
        self,
        radar_root: Path,
        years: Iterable[int],
        *,
        input_frames: int,
        output_frames: int,
        frame_minutes: int,
        crop: CropContract = CropContract(),
        require_targets: bool = True,
        anchor_stride: int = 1,
        max_windows: int | None = None,
        supplemental_roots: Iterable[Path] = (),
    ) -> None:
        if input_frames < 2 or output_frames < 1:
            raise ValueError("input_frames >= 2 and output_frames >= 1 are required")
        if int(frame_minutes) != 10:
            raise ValueError("the primary KMA TIFF contract requires 10-minute frames")
        if tuple(supplemental_roots):
            raise ValueError("the primary contract uses one phase-0 TIFF root")
        if anchor_stride < 1:
            raise ValueError("anchor_stride must be >= 1")

        self.radar_roots = (Path(radar_root), *map(Path, supplemental_roots))
        self.input_frames = int(input_frames)
        self.output_frames = int(output_frames)
        self.frame_minutes = int(frame_minutes)
        self.require_targets = bool(require_targets)
        self._paths: dict[int, tuple[Path, ...]] = {}
        self._times: dict[int, np.ndarray] = {}
        self._manifests: list[dict[str, object]] = []

        native_shape: tuple[int, int] | None = None
        minute_phases: set[int] = set()
        for root in self.radar_roots:
            manifest_path = root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"KMA TIFF manifest is missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prepared = manifest.get("prepared_contract", {})
            shape = tuple(int(item) for item in prepared.get("shape", ()))
            if shape != (256, 256):
                raise ValueError(f"unsupported KMA TIFF shape {shape}: {manifest_path}")
            if prepared.get("dtype") != "float32 TIFF":
                raise ValueError(f"unexpected KMA TIFF dtype contract: {manifest_path}")
            if prepared.get("normalization") != "max(raw_100xdbz, 0) / 10000":
                raise ValueError(f"unexpected KMA TIFF normalization: {manifest_path}")
            phase = int(prepared.get("minute_phase", -1))
            if phase != 0:
                raise ValueError(f"unexpected KMA TIFF minute phase {phase}: {manifest_path}")
            if int(prepared.get("temporal_resolution_minutes", -1)) != 10:
                raise ValueError(f"unexpected TIFF cadence: {manifest_path}")
            minute_phases.add(phase)
            self._manifests.append(
                {
                    "path": str(manifest_path.resolve()),
                    "minute_phase": phase,
                    "temporal_resolution_minutes": int(
                        prepared.get("temporal_resolution_minutes", -1)
                    ),
                }
            )
            native_shape = shape if native_shape is None else native_shape

        if native_shape is None:
            raise ValueError("no KMA TIFF roots were supplied")
        self.crop = crop.resolve(native_shape)

        references: list[tuple[int, int]] = []
        step_ns = self.frame_minutes * 60 * 1_000_000_000
        for year_value in years:
            year = int(year_value)
            timestamp_to_path: dict[int, Path] = {}
            for root in self.radar_roots:
                year_root = root / f"{year:04d}"
                for path in sorted(year_root.glob("*/*/*.tiff")):
                    timestamp = int(_timestamp_from_tiff(path).astype(np.int64))
                    if int(path.stem[4:6]) not in SEASON_MONTHS:
                        continue
                    if self.frame_minutes == 10:
                        preferred_phase = 0 if 0 in minute_phases else next(iter(minute_phases))
                        minute = int(path.stem[-2:])
                        if minute % 10 != preferred_phase:
                            continue
                    previous = timestamp_to_path.setdefault(timestamp, path)
                    if previous != path:
                        raise ValueError(
                            f"duplicate KMA TIFF timestamp {path.stem}: {previous} and {path}"
                        )
            if not timestamp_to_path:
                raise RuntimeError(f"no KMA TIFF frames found for {year}")
            ordered = sorted(timestamp_to_path.items())
            times = np.asarray([item[0] for item in ordered], dtype=np.int64)
            paths = tuple(item[1] for item in ordered)
            first = self.input_frames - 1
            last_exclusive = len(times) - (self.output_frames if self.require_targets else 0)
            anchors = np.arange(first, last_exclusive, int(anchor_stride), dtype=np.int64)
            if len(anchors):
                starts = anchors - self.input_frames + 1
                input_continuous = (
                    times[anchors] - times[starts]
                    == (self.input_frames - 1) * step_ns
                )
                if self.require_targets:
                    ends = anchors + self.output_frames
                    output_continuous = (
                        times[ends] - times[anchors] == self.output_frames * step_ns
                    )
                    continuous = input_continuous & output_continuous
                else:
                    continuous = input_continuous
                references.extend((year, int(index)) for index in anchors[continuous])
            self._paths[year] = paths
            self._times[year] = times

        if max_windows is not None and int(max_windows) < 1:
            raise ValueError("max_windows must be positive when supplied")
        if max_windows is not None and int(max_windows) < len(references):
            positions = np.linspace(
                0, len(references) - 1, num=int(max_windows), dtype=np.int64
            )
            references = [references[int(position)] for position in positions]
        if not references:
            raise RuntimeError("no continuous KMA TIFF windows satisfy the requested contract")
        self.references = tuple(references)

        groups: dict[tuple[int, int], list[int]] = {}
        for position, (year, anchor) in enumerate(self.references):
            day = int(self._times[year][anchor] // DAY_NS)
            groups.setdefault((year, day), []).append(position)
        self.chunk_local_groups = tuple(
            np.asarray(group, dtype=np.int64) for group in groups.values()
        )

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        year, anchor = self.references[int(position)]
        start = anchor - self.input_frames + 1
        stop = anchor + 1 + (self.output_frames if self.require_targets else 0)
        raw = np.stack(
            [_read_normalized_tiff(str(path)) for path in self._paths[year][start:stop]],
            axis=0,
        )
        value = transform_normalized_spatial(torch.from_numpy(raw), self.crop)
        result = {
            "inputs": value[: self.input_frames].unsqueeze(0),
            "issue_time_ns": torch.tensor(self._times[year][anchor], dtype=torch.int64),
        }
        if self.require_targets:
            result["targets"] = value[self.input_frames :].unsqueeze(0)
        return result

    def metadata(self) -> dict[str, object]:
        years = sorted({year for year, _ in self.references})
        return {
            "radar_roots": [str(path.resolve()) for path in self.radar_roots],
            "radar_format": "kma-tiff",
            "months": list(SEASON_MONTHS),
            "years": years,
            "window_count": len(self),
            "input_frames": self.input_frames,
            "output_frames": self.output_frames,
            "frame_minutes": self.frame_minutes,
            "issue_time_timezone": "KST",
            "require_targets": self.require_targets,
            "crop": self.crop.to_dict(),
            "normalization": "official KMA TIFF: max(raw_100xdbz, 0) / 10000",
            "manifests": self._manifests,
        }


class ChunkLocalRandomSampler(Sampler[int]):
    """Shuffle all windows without replacement while retaining I/O locality."""

    def __init__(
        self,
        dataset: KmaTiffWindowDataset,
        *,
        num_samples: int | None,
        locality_span: int,
        generator: torch.Generator,
    ) -> None:
        self.groups = dataset.chunk_local_groups
        self.num_samples = len(dataset) if num_samples in (None, 0) else int(num_samples)
        self.num_samples = min(self.num_samples, len(dataset))
        self.locality_span = int(locality_span)
        self.generator = generator
        if self.locality_span < 1:
            raise ValueError("locality_span must be >= 1")

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        emitted = 0
        group_order = torch.randperm(len(self.groups), generator=self.generator).tolist()
        for group_index in group_order:
            group = self.groups[group_index]
            order = torch.randperm(len(group), generator=self.generator).numpy()
            blocks: list[np.ndarray] = []
            for offset in range(0, len(order), self.locality_span):
                blocks.append(group[order[offset : offset + self.locality_span]])
            block_order = torch.randperm(len(blocks), generator=self.generator).tolist()
            for block_index in block_order:
                for position in blocks[block_index]:
                    if emitted >= self.num_samples:
                        return
                    yield int(position)
                    emitted += 1


def estimate_full_grid_export_gib(
    windows: int,
    output_frames: int,
    height: int,
    width: int,
    dtype: str,
) -> float:
    itemsize = np.dtype(dtype).itemsize
    return float(windows * output_frames * height * width * itemsize / 2**30)


__all__ = [
    "ChunkLocalRandomSampler",
    "CropContract",
    "KmaTiffWindowDataset",
    "NATIVE_FRAME_MINUTES",
    "ResolvedCrop",
    "dbz_to_rain_rate",
    "estimate_full_grid_export_gib",
    "exprecast_normalized_to_dbz",
    "transform_normalized_spatial",
]

