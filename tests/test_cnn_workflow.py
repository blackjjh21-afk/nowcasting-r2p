from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cnn_readout import workflow  # noqa: E402
from cnn_readout.sampling import UNIFORM_WITHOUT_REPLACEMENT  # noqa: E402


def _write_dense(
    path: Path,
    *,
    issue_times: np.ndarray,
    station_ids: np.ndarray,
    station_folds: np.ndarray,
    leads: np.ndarray,
    targets: np.ndarray | None,
    include_oof: bool = False,
) -> None:
    shape = (len(issue_times), len(station_ids), len(leads))
    rng = np.random.default_rng(73)
    values: dict[str, np.ndarray] = {
        "schema": np.asarray(workflow.PREPARED_SCHEMA),
        "patches": rng.normal(size=shape + (6, 3, 3)).astype(np.float32),
        "issue_times_ns": np.asarray(issue_times, dtype=np.int64),
        "station_ids": np.asarray(station_ids, dtype=np.int64),
        "station_folds": np.asarray(station_folds, dtype=np.int8),
        "lead_minutes": np.asarray(leads, dtype=np.int16),
        "auxiliary": rng.normal(size=shape + (39,)).astype(np.float32),
    }
    if targets is not None:
        values["targets_mm"] = np.asarray(targets, dtype=np.float32)
    if include_oof:
        values["oof_auxiliary"] = rng.normal(size=(3,) + shape + (39,)).astype(
            np.float32
        )
    np.savez_compressed(path, **values)


def test_canonical_dense_schema_flattens_without_changing_axis_order(
    tmp_path: Path,
) -> None:
    issue_times = np.asarray(
        ["2023-06-10T00", "2023-06-10T01"], dtype="datetime64[h]"
    ).astype("datetime64[ns]").astype(np.int64)
    leads = workflow.SELECTION_LEADS
    targets = np.arange(2 * 3 * len(leads), dtype=np.float32).reshape(
        2, 3, len(leads)
    )
    targets[0, 0, 0] = np.nan
    source = tmp_path / "prepared.npz"
    _write_dense(
        source,
        issue_times=issue_times,
        station_ids=np.asarray((101, 202, 303)),
        station_folds=np.asarray((0, 1, 2)),
        leads=leads,
        targets=targets,
        include_oof=True,
    )

    data = workflow.load_prepared_npz(
        source, require_target=True, require_oof_auxiliary=True
    )

    assert data.dense_shape == (2, 3, 13)
    assert data.patches.shape == (78, 6, 3, 3)
    assert data.auxiliary.shape == (78, 39)
    assert data.oof_auxiliary is not None
    assert data.oof_auxiliary.shape == (3, 78, 39)
    np.testing.assert_array_equal(data.station_id[:13], np.full(13, 101))
    np.testing.assert_array_equal(data.lead_min[:13], leads)
    assert np.isnan(data.target_mm[0])


def test_canonical_schema_rejects_ambiguous_dense_axes(tmp_path: Path) -> None:
    issue_times = np.asarray(
        ["2023-06-10T01", "2023-06-10T00"], dtype="datetime64[h]"
    ).astype("datetime64[ns]").astype(np.int64)
    source = tmp_path / "bad.npz"
    _write_dense(
        source,
        issue_times=issue_times,
        station_ids=np.asarray((101, 202, 303)),
        station_folds=np.asarray((0, 1, 2)),
        leads=workflow.SELECTION_LEADS,
        targets=np.zeros((2, 3, 13), dtype=np.float32),
    )

    with pytest.raises(workflow.PreparedDataError, match="strictly increasing"):
        workflow.load_prepared_npz(source, require_target=True)


def test_flat_adapter_schema_remains_interoperable(tmp_path: Path) -> None:
    source = tmp_path / "flat.npz"
    np.savez_compressed(
        source,
        schema=np.asarray(workflow.PREPARED_SCHEMA),
        patches=np.zeros((2, 6, 3, 3), dtype=np.float32),
        issue_time_ns=np.asarray((1_686_355_200, 1_686_355_200), dtype=np.int64)
        * 1_000_000_000,
        station_id=np.asarray((101, 202), dtype=np.int64),
        station_fold=np.asarray((0, 1), dtype=np.int8),
        lead_min=np.asarray((60, 60), dtype=np.int16),
    )

    data = workflow.load_prepared_npz(source, require_target=False)

    assert data.dense_shape is None
    assert data.target_mm is None
    assert data.auxiliary.shape == (2, 39)
    np.testing.assert_array_equal(data.auxiliary, 0.0)


def test_flat_schema_rejects_pre_rn60_window_lead(tmp_path: Path) -> None:
    source = tmp_path / "bad_lead.npz"
    np.savez_compressed(
        source,
        schema=np.asarray(workflow.PREPARED_SCHEMA),
        patches=np.zeros((1, 6, 3, 3), dtype=np.float32),
        issue_time_ns=np.asarray((1_686_355_200_000_000_000,), dtype=np.int64),
        station_id=np.asarray((101,), dtype=np.int64),
        station_fold=np.asarray((0,), dtype=np.int8),
        lead_min=np.asarray((50,), dtype=np.int16),
    )
    with pytest.raises(workflow.PreparedDataError, match="RN60 target grid"):
        workflow.load_prepared_npz(source, require_target=False)


def test_macro_csi_is_unweighted_over_lead_threshold_cells() -> None:
    prediction = np.asarray((0.0, 6.0, 12.0, 25.0), dtype=np.float32)
    truth = np.asarray((0.0, 7.0, 8.0, 21.0), dtype=np.float32)
    lead = np.full(4, 60, dtype=np.int16)
    counts = workflow.categorical_counts(
        prediction,
        truth,
        lead,
        leads=np.asarray((60,), dtype=np.int16),
        thresholds=np.asarray((1.0, 5.0, 10.0, 20.0), dtype=np.float32),
    )

    np.testing.assert_allclose(workflow.csi_from_counts(counts), [[1.0, 1.0, 0.5, 1.0]])
    assert workflow.macro_csi_from_counts(counts) == pytest.approx(0.875)


@pytest.mark.parametrize("thresholds", [None, (1., 5., 10.)])
def test_month_station_oof_runs_all_twelve_cells_on_synthetic_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, thresholds
) -> None:
    class TinyCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(6 * 3 * 3 + 39, 1)

        def forward(self, patches: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
            inputs = torch.cat((patches.flatten(1), auxiliary), dim=1)
            return torch.nn.functional.softplus(self.linear(inputs)).squeeze(-1)

    monkeypatch.setattr(workflow, "CNN", TinyCNN)
    issue_times = np.asarray(
        [
            "2023-06-10T00",
            "2023-06-20T00",
            "2023-07-10T00",
            "2023-07-20T00",
            "2023-08-10T00",
            "2023-08-20T00",
            "2023-09-10T00",
            "2023-09-20T00",
        ],
        dtype="datetime64[h]",
    ).astype("datetime64[ns]").astype(np.int64)
    leads = np.arange(60, 181, 10, dtype=np.int16)
    target_values = np.asarray((0.0, 2.0, 7.0, 15.0, 25.0), dtype=np.float32)
    targets = np.resize(target_values, (len(issue_times), 3, len(leads)))
    source = tmp_path / "oof.npz"
    _write_dense(
        source,
        issue_times=issue_times,
        station_ids=np.asarray((101, 202, 303)),
        station_folds=np.asarray((0, 1, 2)),
        leads=leads,
        targets=targets,
        include_oof=True,
    )
    data = workflow.load_prepared_npz(
        source, require_target=True, require_oof_auxiliary=True
    )

    result = workflow.select_epoch_oof(
        data,
        max_epochs=1,
        seed=22500,
        learning_rate=3e-4,
        weight_decay=1e-4,
        recipe=UNIFORM_WITHOUT_REPLACEMENT,
        samples_per_epoch=4,
        train_batch_size=4,
        validation_batch_size=64,
        device=torch.device("cpu"),
        selection_thresholds=thresholds,
    )

    assert result["selected_epoch"] == 1
    assert len(result["cells"]) == 12
    assert result["selection_leads_min"] == leads.astype(int).tolist()
    expected = [1.0, 5.0, 10.0] if thresholds is None else list(thresholds)
    assert result["selection_thresholds_mm"] == expected
    assert np.asarray(result["pooled_contingency_counts_by_epoch"]).shape == (1, 13, len(expected), 4)
    assert len(result["pooled_contingency_counts_by_epoch"]) == 1


def test_fit_and_predict_are_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    issue_times = np.asarray(
        ["2023-06-10T00", "2023-06-10T01"], dtype="datetime64[h]"
    ).astype("datetime64[ns]").astype(np.int64)
    source = tmp_path / "fit.npz"
    _write_dense(
        source,
        issue_times=issue_times,
        station_ids=np.asarray((101, 202)),
        station_folds=np.asarray((0, 1)),
        leads=workflow.SELECTION_LEADS,
        targets=np.resize(
            np.asarray((0.0, 2.0, 7.0, 15.0, 25.0), dtype=np.float32),
            (2, 2, len(workflow.SELECTION_LEADS)),
        ),
    )
    checkpoints = (tmp_path / "a.pt", tmp_path / "b.pt")
    for checkpoint in checkpoints:
        assert (
            workflow.main(
                [
                    "fit",
                    "--input",
                    str(source),
                    "--output",
                    str(checkpoint),
                    "--epochs",
                    "1",
                    "--seed",
                    "21040",
                    "--sampling-recipe",
                    UNIFORM_WITHOUT_REPLACEMENT,
                    "--samples-per-epoch",
                    "8",
                    "--batch-size",
                    "4",
                ]
            )
            == 0
        )

    payload_a = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    payload_b = torch.load(checkpoints[1], map_location="cpu", weights_only=True)
    for name, value in payload_a["state_dict"].items():
        torch.testing.assert_close(value, payload_b["state_dict"][name], rtol=0, atol=0)

    outputs = (tmp_path / "prediction_a.npz", tmp_path / "prediction_b.npz")
    for checkpoint, output in zip(checkpoints, outputs, strict=True):
        assert (
            workflow.main(
                [
                    "predict",
                    "--input",
                    str(source),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output),
                    "--batch-size",
                    "8",
                ]
            )
            == 0
        )
    with np.load(outputs[0], allow_pickle=False) as first, np.load(
        outputs[1], allow_pickle=False
    ) as second:
        assert str(first["schema"].item()) == workflow.PREDICTION_SCHEMA
        assert first["predictions_mm"].shape == (2, 2, 13)
        np.testing.assert_array_equal(first["predictions_mm"], second["predictions_mm"])
