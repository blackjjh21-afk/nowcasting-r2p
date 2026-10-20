import json

import numpy as np
import pytest

from common import reselection as r


def record(epoch, values):
    return r._matrix_record(epoch, np.tile(values, (13, 1)), r.LEADS, (1, 5, 10, 20))


def test_three_thresholds_can_change_winner_without_touching_monitor():
    old = record(21, [.4, .4, .4, .0])
    new = record(26, [.3, .3, .3, 1.])
    result = r.select_csi3([old, new])
    assert result["selected_epoch"] == 21
    assert result["selected_score"] == pytest.approx(.4)
    assert result["historical_monitor_changed"] is False
    assert result["training_changed"] is False


def test_warmup_and_earliest_exact_tie():
    result = r.select_csi3([record(2, [1]*4), record(9, [.5]*4), record(4, [.5]*4)],
                           exclude_through=3)
    assert result["selected_epoch"] == 4
    assert result["excluded_initial_epochs"] == [1, 2, 3]


def test_public_pooled_counts_are_scored_after_pooling(tmp_path):
    counts = np.zeros((2, 13, 4, 4), int)
    counts[0, ..., :3] = (8, 1, 1)
    counts[1, ..., :3] = (5, 2, 3)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(dict(pooled_contingency_counts_by_epoch=counts.tolist(),
        selection_leads_min=list(r.LEADS), selection_thresholds_mm=[1, 5, 10, 20])))
    result = r.select_csi3(r.read_records([path]))
    assert result["selected_epoch"] == 1
    assert result["selected_score"] == pytest.approx(.8)


def test_direct_csv_epoch_filename_and_cli(tmp_path):
    paths = []
    for epoch, value in ((4, .2), (7, .4)):
        path = tmp_path / f"validation_metrics_epoch_{epoch:03d}.csv"
        path.write_text("lead_min,csi_1,csi_5,csi_10,csi_20\n" + "".join(
            f"{lead},{value},{value},{value},0.99\n" for lead in r.LEADS))
        paths.append(path)
    before = [path.read_bytes() for path in paths]
    output = tmp_path / "final.json"
    assert r.main(["--input", *map(str, paths), "--output", str(output), "--exclude-through", "3"]) == 0
    result = json.loads(output.read_text())
    assert result["selected_epoch"] == 7
    assert result["selection_thresholds_mm"] == [1, 5, 10]
    assert [path.read_bytes() for path in paths] == before
    with pytest.raises(FileExistsError):
        r.main(["--input", str(paths[0]), "--output", str(output)])


def test_long_observed_csv_and_family_selection(tmp_path):
    path = tmp_path / "oof.csv"
    path.write_text("family,epoch,threshold_mm,csi\n" + "".join(
        f"{family},{epoch},{threshold},{score}\n"
        for family in ("patch_cnn", "center_mlp")
        for epoch, score in ((33, .5), (45, .6)) for threshold in (1, 5, 10, 20)))
    with pytest.raises(ValueError, match="Multiple"):
        r.read_records([path])
    result = r.select_csi3(r.read_records([path], family="patch_cnn"), leads=(0,))
    assert result["selected_epoch"] == 45


def test_preaggregated_13lead_means(tmp_path):
    path = tmp_path / "epoch_selection.csv"
    path.write_text("epoch,csi_1mm_mean_13lead,csi_5mm_mean_13lead,csi_10mm_mean_13lead\n"
                    "4,0.1,0.3,0.5\n6,0.2,0.4,0.6\n")
    rows = r.read_records([path])
    assert r.select_csi3(rows)["selected_epoch"] == 6
    with pytest.raises(ValueError, match="different lead"):
        r.select_csi3(rows, leads=(60,))


def test_nowcastnet_history_and_field_json(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("\n".join(json.dumps(dict(epoch=epoch, validation=dict(
        csi=np.full((13, 4), score).tolist(), leads_min=list(r.LEADS), thresholds_mm=[1, 5, 10, 20])))
        for epoch, score in ((3, .4), (8, .5))))
    assert r.select_csi3(r.read_records([path]))["selected_epoch"] == 8
    field = tmp_path / "field.json"
    field.write_text(json.dumps(dict(epoch=10, csi_by_lead_threshold=np.full((13, 4), .6).tolist(),
        lead_minutes=list(r.LEADS), thresholds_mm=[1, 5, 10, 20], test_period_accessed=False)))
    assert r.select_csi3(r.read_records([field]))["selected_epoch"] == 10


def test_archived_nowcastnet_pooled_epochs_use_counts():
    counts = np.full((13, 4), 2).tolist()
    payload = dict(calibration_years=[2023], pooled_epochs=[dict(epoch=6, hits=counts,
        misses=counts, false_alarms=counts, csi=np.ones((13, 4)).tolist())])
    assert r.select_csi3(r._json_records(payload))["selected_score"] == pytest.approx(1/3)
    payload["calibration_years"] = [2019, 2020, 2021, 2022]
    with pytest.raises(ValueError, match="2023"):
        r._json_records(payload)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "undefined", "out_of_range"])
def test_invalid_selection_support_fails(problem):
    row = record(4, [.5]*4)
    rows = [row]
    if problem == "duplicate":
        rows.append(row)
    elif problem == "missing":
        row["leads"] = tuple(range(70, 191, 10))
    elif problem == "undefined":
        row["csi"][0, 0] = np.nan
    else:
        row["csi"][0, 0] = 1.1
    with pytest.raises(ValueError):
        r.select_csi3(rows)


def test_duplicate_oof_cells_cannot_be_averaged_as_scores(tmp_path):
    path = tmp_path / "unpooled.csv"
    path.write_text("epoch,lead_min,threshold_mm,csi\n1,60,1,0.5\n1,60,1,0.6\n")
    with pytest.raises(ValueError, match="pool OOF counts"):
        r.read_records([path])
