"""Exact numerical adapter used around deterministic pySTEPS forecasts.

The public HSR contract stores ``x = max(dBZ, 0) / 100``.  pySTEPS receives
rain rate, performs Lucas--Kanade motion estimation in its dB rain-rate
representation, and returns 18 ten-minute forecasts.  This module keeps the
unit changes and station-patch construction explicit and independently
testable.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


ZR_A = 200.0
ZR_B = 1.6
RAIN_THRESHOLD_MM_H = 0.1
DBR_ZERO = -15.0
DBR_INVERSE_THRESHOLD = -10.0
INPUT_OFFSETS_MINUTES = np.arange(-60, 1, 10, dtype=np.int64)
FIELD_LEAD_MINUTES = np.arange(10, 181, 10, dtype=np.int64)
TARGET_LEAD_MINUTES = np.arange(60, 181, 10, dtype=np.int64)
WINDOW_FRAMES = 6


class PySTEPSAdapterError(RuntimeError):
    """Raised when an input or upstream pySTEPS result violates the contract."""


@dataclass(frozen=True)
class PySTEPSBackend:
    """Three injected pySTEPS callables used by :func:`forecast_rain_rate_lk`.

    Dependency injection keeps the numerical contract testable without a
    pySTEPS installation.  Production callers normally omit this argument.
    """

    db_transform: Callable[..., tuple[Any, Any]]
    motion_lk: Callable[[NDArray[np.float32]], Any]
    extrapolate: Callable[..., Any]


def _finite_nonnegative(value: ArrayLike, *, name: str) -> NDArray[np.float32]:
    result = np.asarray(value, dtype=np.float32)
    if np.any(~np.isfinite(result)):
        raise PySTEPSAdapterError(f"{name} contains a non-finite value")
    if np.any(result < 0):
        raise PySTEPSAdapterError(f"{name} contains a negative value")
    return result


def normalized_hsr_to_rain_rate(normalized_hsr: ArrayLike) -> NDArray[np.float32]:
    """Convert public normalized HSR to rain rate in mm h\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}.

    Exact stored zeros remain zero.  Positive values are converted through
    ``Z = 200 R**1.6`` using the same float32 operation order as the analysis.
    """

    result = _finite_nonnegative(normalized_hsr, name="normalized HSR").copy()
    zero = result <= np.float32(0.0)
    result *= np.float32(10.0 * np.log(10.0) / ZR_B)
    result -= np.float32(np.log(ZR_A) / ZR_B)
    np.exp(result, out=result)
    result[zero] = np.float32(0.0)
    if np.any(~np.isfinite(result)):
        raise PySTEPSAdapterError("normalized-HSR conversion overflowed")
    return result


def rain_rate_to_normalized_hsr(rain_rate_mm_h: ArrayLike) -> NDArray[np.float32]:
    """Convert nonnegative rain rate to ``max(dBZ, 0) / 100`` float32 HSR."""

    rate = _finite_nonnegative(rain_rate_mm_h, name="rain rate")
    result = np.zeros(rate.shape, dtype=np.float32)
    positive = rate > np.float32(0.0)
    if np.any(positive):
        dbz = np.float32(10.0) * np.log10(
            np.float32(ZR_A) * np.power(rate[positive], np.float32(ZR_B))
        )
        result[positive] = np.maximum(dbz, np.float32(0.0)) / np.float32(100.0)
    if np.any(~np.isfinite(result)):
        raise PySTEPSAdapterError("rain-rate conversion overflowed")
    return result


def load_pysteps_backend() -> PySTEPSBackend:
    """Load upstream pySTEPS lazily, avoiding a local package-name collision."""

    try:
        from pysteps import motion, nowcasts  # type: ignore[import-not-found]
        from pysteps.utils import transformation  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise PySTEPSAdapterError(
            "forecast generation requires the optional upstream 'pysteps' package"
        ) from exc
    return PySTEPSBackend(
        db_transform=transformation.dB_transform,
        motion_lk=motion.get_method("LK"),
        extrapolate=nowcasts.get_method("extrapolation"),
    )


def forecast_rain_rate_lk(
    input_rain_rate_mm_h: ArrayLike,
    *,
    backend: PySTEPSBackend | None = None,
) -> NDArray[np.float32]:
    """Generate 18 deterministic ten-minute forecasts with LK + extrapolation.

    Parameters
    ----------
    input_rain_rate_mm_h:
        Seven fields ordered from -60 to 0 minutes, shape ``[7, H, W]``.
    backend:
        Optional injected backend for tests.  When omitted, upstream pySTEPS is
        loaded and its ``LK`` motion and ``extrapolation`` nowcast methods are
        used.

    Returns
    -------
    numpy.ndarray
        Nonnegative rain rate for +10 through +180 minutes, shape
        ``[18, H, W]`` and dtype float32.

    Notes
    -----
    This function deliberately has no persistence fallback.  A failed forecast
    raises :class:`PySTEPSAdapterError` rather than silently fabricating output.
    """

    rate = _finite_nonnegative(input_rain_rate_mm_h, name="input rain rate")
    if rate.ndim != 3 or rate.shape[0] != len(INPUT_OFFSETS_MINUTES):
        raise PySTEPSAdapterError(
            f"input rain rate must have shape [7, H, W], got {rate.shape}"
        )
    if rate.shape[1] < 2 or rate.shape[2] < 2:
        raise PySTEPSAdapterError("input spatial dimensions must both be at least 2")

    selected = backend if backend is not None else load_pysteps_backend()
    try:
        with redirect_stdout(StringIO()):
            transformed, _ = selected.db_transform(
                rate,
                threshold=RAIN_THRESHOLD_MM_H,
                zerovalue=DBR_ZERO,
            )
        dbr = np.asarray(transformed, dtype=np.float32)
        dbr = np.where(np.isfinite(dbr), dbr, np.float32(DBR_ZERO)).astype(
            np.float32, copy=False
        )
        velocity = selected.motion_lk(dbr)
        forecast_dbr = selected.extrapolate(
            dbr[-1],
            velocity,
            len(FIELD_LEAD_MINUTES),
            extrap_method="semilagrangian",
            extrap_kwargs={"outval": np.nan},
        )
        with redirect_stdout(StringIO()):
            forecast_rate, _ = selected.db_transform(
                forecast_dbr,
                threshold=DBR_INVERSE_THRESHOLD,
                inverse=True,
            )
    except PySTEPSAdapterError:
        raise
    except Exception as exc:
        raise PySTEPSAdapterError("upstream pySTEPS forecast failed") from exc

    result = np.asarray(forecast_rate, dtype=np.float32)
    expected = (len(FIELD_LEAD_MINUTES), rate.shape[1], rate.shape[2])
    if result.shape != expected:
        raise PySTEPSAdapterError(
            f"pySTEPS returned shape {result.shape}; expected {expected}"
        )
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    if np.any(result < 0):
        raise PySTEPSAdapterError("pySTEPS returned a negative rain rate")
    return np.ascontiguousarray(result)


def forecast_normalized_hsr_lk(
    normalized_hsr: ArrayLike,
    *,
    backend: PySTEPSBackend | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Forecast normalized HSR and also return its intermediate rain rates."""

    rate = normalized_hsr_to_rain_rate(normalized_hsr)
    forecast_rate = forecast_rain_rate_lk(rate, backend=backend)
    return rain_rate_to_normalized_hsr(forecast_rate), forecast_rate


def _station_patch_indices(
    station_y: ArrayLike,
    station_x: ArrayLike,
    *,
    patch_size: int,
    height: int,
    width: int,
    valid_mask: ArrayLike | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    if patch_size <= 0 or patch_size % 2 != 1:
        raise PySTEPSAdapterError("patch_size must be a positive odd integer")
    y = np.asarray(station_y)
    x = np.asarray(station_x)
    if y.ndim != 1 or x.ndim != 1 or y.shape != x.shape:
        raise PySTEPSAdapterError("station_y and station_x must be equal-length 1-D arrays")
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(x.dtype, np.integer):
        raise PySTEPSAdapterError("station coordinates must use an integer dtype")
    y = y.astype(np.int64, copy=False)
    x = x.astype(np.int64, copy=False)
    radius = patch_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    py = y[:, None, None] + dy
    px = x[:, None, None] + dx
    if (
        np.any(py < 0)
        or np.any(py >= height)
        or np.any(px < 0)
        or np.any(px >= width)
    ):
        raise PySTEPSAdapterError("a station patch crosses the forecast-grid edge")
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != (height, width):
            raise PySTEPSAdapterError(
                f"valid_mask must have shape {(height, width)}, got {mask.shape}"
            )
        if not np.all(mask[py, px]):
            raise PySTEPSAdapterError("a station patch enters invalid source support")
    return y, x, py, px


def extract_station_patches(
    forecast_fields: ArrayLike,
    station_y: ArrayLike,
    station_x: ArrayLike,
    *,
    patch_size: int = 3,
    valid_mask: ArrayLike | None = None,
) -> NDArray[np.float32]:
    """Extract station-centred patches without changing field/lead order.

    ``forecast_fields`` may be ``[lead, H, W]`` or ``[batch, lead, H, W]``.
    Outputs are respectively ``[station, lead, P, P]`` or
    ``[batch, station, lead, P, P]``.
    """

    fields = _finite_nonnegative(forecast_fields, name="forecast fields")
    squeezed = fields.ndim == 3
    if squeezed:
        fields = fields[None, ...]
    if fields.ndim != 4:
        raise PySTEPSAdapterError(
            "forecast fields must have shape [lead, H, W] or [batch, lead, H, W]"
        )
    _, _, height, width = fields.shape
    _, _, py, px = _station_patch_indices(
        station_y,
        station_x,
        patch_size=patch_size,
        height=height,
        width=width,
        valid_mask=valid_mask,
    )
    # [B,L,H,W] indexed by [N,P,P] -> [B,N,L,P,P].
    result = np.ascontiguousarray(fields[:, :, py, px].transpose(0, 2, 1, 3, 4))
    return result[0] if squeezed else result


def lead_window_indices(
    field_lead_minutes: Sequence[int] | NDArray[np.integer[Any]] = FIELD_LEAD_MINUTES,
    target_lead_minutes: Sequence[int] | NDArray[np.integer[Any]] = TARGET_LEAD_MINUTES,
) -> NDArray[np.int64]:
    """Return six consecutive field indices ending at each RN60 target lead."""

    field = np.asarray(field_lead_minutes, dtype=np.int64)
    target = np.asarray(target_lead_minutes, dtype=np.int64)
    if field.ndim != 1 or target.ndim != 1:
        raise PySTEPSAdapterError("field and target leads must be 1-D")
    if len(field) < WINDOW_FRAMES or np.any(np.diff(field) <= 0):
        raise PySTEPSAdapterError("field leads must be strictly increasing")
    lookup = {int(lead): index for index, lead in enumerate(field)}
    rows: list[list[int]] = []
    for target_lead in target:
        expected = [int(target_lead) - 10 * offset for offset in range(5, -1, -1)]
        try:
            row = [lookup[lead] for lead in expected]
        except KeyError as exc:
            raise PySTEPSAdapterError(
                f"target lead {int(target_lead)} lacks its six ten-minute fields"
            ) from exc
        if any(b - a != 1 for a, b in zip(row, row[1:])):
            raise PySTEPSAdapterError("six-frame lead windows are not consecutive")
        rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def make_six_frame_lead_windows(
    station_patches: ArrayLike,
    *,
    field_lead_minutes: Sequence[int] | NDArray[np.integer[Any]] = FIELD_LEAD_MINUTES,
    target_lead_minutes: Sequence[int] | NDArray[np.integer[Any]] = TARGET_LEAD_MINUTES,
) -> NDArray[np.float32]:
    """Gather six forecast patches ending at each RN60 target lead.

    Accepted shapes are ``[station, lead, P, P]`` and
    ``[batch, station, lead, P, P]``.  A target-lead axis is inserted before
    the six-frame axis.
    """

    patches = _finite_nonnegative(station_patches, name="station patches")
    squeezed = patches.ndim == 4
    if squeezed:
        patches = patches[None, ...]
    if patches.ndim != 5:
        raise PySTEPSAdapterError(
            "station patches must have shape [station, lead, P, P] or "
            "[batch, station, lead, P, P]"
        )
    field = np.asarray(field_lead_minutes, dtype=np.int64)
    if patches.shape[2] != len(field):
        raise PySTEPSAdapterError(
            f"patch lead axis has length {patches.shape[2]}, expected {len(field)}"
        )
    indices = lead_window_indices(field, target_lead_minutes)
    result = np.ascontiguousarray(np.take(patches, indices, axis=2))
    # np.take result: [B,N,target,frame,P,P].
    return result[0] if squeezed else result
