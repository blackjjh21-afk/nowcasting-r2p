from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysteps_adapter import (
    FIELD_LEAD_MINUTES,
    PySTEPSAdapterError,
    PySTEPSBackend,
    extract_station_patches,
    forecast_rain_rate_lk,
    lead_window_indices,
    make_six_frame_lead_windows,
    normalized_hsr_to_rain_rate,
    rain_rate_to_normalized_hsr,
)
from pysteps_adapter.cli import main


def test_normalized_hsr_rain_rate_round_trip() -> None:
    rate = np.asarray([0.0, 0.1, 1.0, 10.0, 100.0], dtype=np.float32)
    normalized = rain_rate_to_normalized_hsr(rate)
    recovered = normalized_hsr_to_rain_rate(normalized)

    assert normalized.dtype == np.float32
    assert recovered.dtype == np.float32
    assert recovered[0] == 0.0
    np.testing.assert_allclose(recovered[1:], rate[1:], rtol=2e-6, atol=1e-6)


def test_normalized_hsr_conversion_matches_literal_float32_operations() -> None:
    normalized = np.asarray([0.0, 0.01, 0.2, 0.5], dtype=np.float32)
    expected = normalized.copy()
    zero = expected <= np.float32(0.0)
    expected *= np.float32(10.0 * np.log(10.0) / 1.6)
    expected -= np.float32(np.log(200.0) / 1.6)
    np.exp(expected, out=expected)
    expected[zero] = 0.0

    assert np.array_equal(normalized_hsr_to_rain_rate(normalized), expected)
    with pytest.raises(PySTEPSAdapterError, match="negative"):
        normalized_hsr_to_rain_rate(np.asarray([-0.1], dtype=np.float32))
    with pytest.raises(PySTEPSAdapterError, match="non-finite"):
        rain_rate_to_normalized_hsr(np.asarray([np.nan], dtype=np.float32))


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def db_transform(self, value: np.ndarray, **kwargs: object):
        self.calls.append(("db", dict(kwargs)))
        array = np.asarray(value, dtype=np.float32)
        if kwargs.get("inverse") is True:
            result = array + np.float32(20.0)
            result[0, 0, 0] = np.nan
            return result, {"inverse": True}
        result = array + np.float32(2.0)
        result[0, 0, 0] = np.nan
        return result, {"inverse": False}

    def motion_lk(self, value: np.ndarray) -> np.ndarray:
        self.calls.append(("motion_shape", value.shape))
        assert value[0, 0, 0] == np.float32(-15.0)
        return np.zeros((2, *value.shape[-2:]), dtype=np.float32)

    def extrapolate(
        self,
        latest: np.ndarray,
        velocity: np.ndarray,
        timesteps: int,
        **kwargs: object,
    ) -> np.ndarray:
        self.calls.append(("extrapolate", (timesteps, dict(kwargs))))
        assert velocity.shape[0] == 2
        return np.repeat(latest[None], timesteps, axis=0)


def test_forecast_calls_exact_lk_semilagrangian_contract() -> None:
    fake = FakeBackend()
    backend = PySTEPSBackend(fake.db_transform, fake.motion_lk, fake.extrapolate)
    input_rate = np.ones((7, 4, 5), dtype=np.float32)

    result = forecast_rain_rate_lk(input_rate, backend=backend)

    assert result.shape == (18, 4, 5)
    assert result.dtype == np.float32
    assert result[0, 0, 0] == 0.0
    assert fake.calls[0] == ("db", {"threshold": 0.1, "zerovalue": -15.0})
    assert fake.calls[1] == ("motion_shape", (7, 4, 5))
    assert fake.calls[2] == (
        "extrapolate",
        (18, {"extrap_method": "semilagrangian", "extrap_kwargs": {"outval": np.nan}}),
    )
    assert fake.calls[3] == ("db", {"threshold": -10.0, "inverse": True})


def test_station_patch_orientation_edges_and_valid_support() -> None:
    fields = np.arange(18 * 6 * 7, dtype=np.float32).reshape(18, 6, 7)
    patches = extract_station_patches(fields, np.asarray([2]), np.asarray([3]))

    assert patches.shape == (1, 18, 3, 3)
    assert np.array_equal(patches[0, 4], fields[4, 1:4, 2:5])
    with pytest.raises(PySTEPSAdapterError, match="edge"):
        extract_station_patches(fields, np.asarray([0]), np.asarray([3]))

    valid = np.ones((6, 7), dtype=bool)
    valid[1, 2] = False
    with pytest.raises(PySTEPSAdapterError, match="invalid source support"):
        extract_station_patches(
            fields, np.asarray([2]), np.asarray([3]), valid_mask=valid
        )


def test_six_frame_windows_end_at_each_target_lead() -> None:
    indices = lead_window_indices()
    assert np.array_equal(indices[0], np.arange(6))
    assert np.array_equal(indices[-1], np.arange(12, 18))
    assert np.array_equal(FIELD_LEAD_MINUTES[indices[0]], np.arange(10, 61, 10))
    assert np.array_equal(FIELD_LEAD_MINUTES[indices[-1]], np.arange(130, 181, 10))

    patches = np.arange(2 * 18, dtype=np.float32).reshape(2, 18, 1, 1)
    windows = make_six_frame_lead_windows(patches)
    assert windows.shape == (2, 13, 6, 1, 1)
    assert np.array_equal(windows[1, 0, :, 0, 0], patches[1, :6, 0, 0])
    assert np.array_equal(windows[1, -1, :, 0, 0], patches[1, 12:, 0, 0])


def test_patch_cli_reads_and_writes_explicit_npy_npz_paths(tmp_path: Path) -> None:
    forecast_path = tmp_path / "forecast.npz"
    mapping_path = tmp_path / "mapping.csv"
    output_path = tmp_path / "nested/windows.npz"
    forecast = np.ones((2, 18, 7, 7), dtype=np.float32)
    issue_times = np.asarray([1_719_792_000_000_000_000, 1_719_795_600_000_000_000])
    np.savez(
        forecast_path,
        forecast_normalized_hsr=forecast,
        issue_times_ns=issue_times,
    )
    mapping_path.write_text(
        "station_id,exprecast_y,exprecast_x\n101,3,3\n102,4,4\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "patches",
                "--forecast",
                str(forecast_path),
                "--mapping-csv",
                str(mapping_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    with np.load(output_path, allow_pickle=False) as handle:
        assert handle["patches"].shape == (2, 2, 13, 6, 3, 3)
        assert np.array_equal(handle["issue_times_ns"], issue_times)
        assert np.array_equal(handle["station_ids"], np.asarray([101, 102]))
        assert np.array_equal(handle["lead_minutes"], np.arange(60, 181, 10))
        assert handle["window_field_lead_minutes"].shape == (13, 6)

    with pytest.raises(FileExistsError, match="--overwrite"):
        main(
            [
                "patches",
                "--forecast",
                str(forecast_path),
                "--mapping-csv",
                str(mapping_path),
                "--output",
                str(output_path),
            ]
        )
