from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/preprocessing/prepare_kma_hsr_4km10min.py"
SPEC = importlib.util.spec_from_file_location("prepare_kma_hsr_4km10min", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prep
SPEC.loader.exec_module(prep)


def test_production_transform_matches_literal_published_transform() -> None:
    rng = np.random.default_rng(20260828)
    raw = np.full(prep.RAW_SHAPE, prep.NO_ECHO, dtype=np.int16)
    raw[187:2228:8, 464:2305:8] = rng.integers(
        -30000,
        12001,
        size=(prep.TARGET_SIZE, 231),
        dtype=np.int16,
    )

    literal = prep.official_region_transform_reference(raw)
    production = prep.official_region_transform(raw)

    assert production.shape == (256, 256)
    assert production.dtype == np.float32
    assert np.array_equal(production, literal)
    assert np.count_nonzero(production[:, 231:]) == 0


def test_transform_orientation_normalization_and_source_support() -> None:
    raw = np.full(prep.RAW_SHAPE, prep.OUTSIDE_DOMAIN, dtype=np.int16)
    raw[187, 464] = 10000
    raw[2227, 2304] = 12000
    raw[195, 472] = -1

    value = prep.official_region_transform(raw)
    rows, columns, source_valid = prep.output_raw_index_axes()

    assert value[0, 230] == pytest.approx(1.2)
    assert value[255, 0] == pytest.approx(1.0)
    assert value[254, 1] == 0.0
    assert np.array_equal(rows, 2227 - 8 * np.arange(256))
    assert np.array_equal(columns, 464 + 8 * np.arange(256))
    assert source_valid[:, :231].all()
    assert not source_valid[:, 231:].any()
    assert int(source_valid.sum()) == 59136


def test_float32_tiff_round_trip(tmp_path: Path) -> None:
    value = np.random.default_rng(3).random((256, 256), dtype=np.float32)
    path = tmp_path / "2023/06/01/202306010000.tiff"

    prep.write_tiff_atomic(path, value)
    loaded = prep.validate_tiff(path)

    assert loaded.dtype == np.float32
    assert np.array_equal(value, loaded)


def test_cli_requires_explicit_coordinate_choice() -> None:
    parser = prep.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--raw-root", "raw", "--output-root", "out"])
    args = parser.parse_args(
        [
            "--raw-root",
            "raw",
            "--output-root",
            "out",
            "--skip-coordinate-audit",
        ]
    )
    assert args.skip_coordinate_audit is True
