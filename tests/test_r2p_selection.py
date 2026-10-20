from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from r2p_4km10min.run_vanilla_r2p import (
    SELECTION_LEADS,
    SELECTION_THRESHOLDS,
    _load_trusted_local_checkpoint,
    _validate_managed_child_directory,
    _validate_prediction_store_for_overwrite,
    select_best_epoch,
    selection_score,
)


def test_selection_score_is_unweighted_mean_of_52_cells() -> None:
    rows = []
    expected = []
    for lead_index, lead in enumerate(SELECTION_LEADS):
        row = {"lead_min": lead}
        for threshold_index, threshold in enumerate(SELECTION_THRESHOLDS):
            value = (lead_index * len(SELECTION_THRESHOLDS) + threshold_index) / 100.0
            row[f"csi_{threshold:g}"] = value
            expected.append(value)
        rows.append(row)
    score, cell_count = selection_score(rows)
    assert cell_count == 52
    assert score == pytest.approx(float(np.mean(expected)))


def test_selection_score_rejects_missing_cell() -> None:
    rows = [
        {"lead_min": lead, **{f"csi_{threshold:g}": 0.2 for threshold in SELECTION_THRESHOLDS}}
        for lead in SELECTION_LEADS
    ]
    del rows[-1]["csi_20"]
    with pytest.raises(RuntimeError, match="lacks required RN60 CSI cells"):
        selection_score(rows)


def test_best_epoch_excludes_first_three_and_breaks_exact_tie_early() -> None:
    trace = pd.DataFrame(
        {
            "epoch": [1, 2, 3, 4, 5, 6],
            "selection_macro_csi": [0.9, 0.8, 0.7, 0.4, 0.5, 0.5],
        }
    )
    assert select_best_epoch(trace) == 5


def test_trusted_checkpoint_loader_rejects_outside_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    checkpoint = root / "seed_0.pt"
    torch.save({"contract": "synthetic"}, checkpoint)
    with pytest.warns(RuntimeWarning, match="pickle semantics"):
        assert _load_trusted_local_checkpoint(
            checkpoint, trusted_root=root, purpose="test"
        )["contract"] == "synthetic"

    outside = tmp_path / "outside.pt"
    torch.save({"contract": "synthetic"}, outside)
    with pytest.raises(RuntimeError, match="outside the trusted"):
        _load_trusted_local_checkpoint(outside, trusted_root=root, purpose="test")

    link = root / "link.pt"
    link.symlink_to(outside)
    with pytest.raises(RuntimeError, match="invalid trusted-local"):
        _load_trusted_local_checkpoint(link, trusted_root=root, purpose="test")


def test_recursive_delete_guards_lock_scope_and_store_ownership(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    seed = output / "seed_0"
    seed.mkdir()
    _validate_managed_child_directory(seed, output, "seed_0")
    with pytest.raises(RuntimeError, match="unexpected managed path"):
        _validate_managed_child_directory(seed, output, "seed_1")

    store = seed / "target_masked_heldout128_predictions"
    store.mkdir()
    (store / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="completion marker"):
        _validate_prediction_store_for_overwrite(store)
    assert (store / "sentinel").is_file()

    (store / "COMPLETED").write_text(
        '{"completed": true, "contract": '
        '"vanilla_heldout_r2p_official4km10min_input_only_v1"}',
        encoding="utf-8",
    )
    _validate_prediction_store_for_overwrite(store)
