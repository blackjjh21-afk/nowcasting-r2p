"""CPU fixtures for the complete, resumable field-to-CNN bank interface."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nowcastnet_hsr import export_bank as bank, field_cli
from nowcastnet_hsr.config import digest, identity
from nowcastnet_hsr.field_config import DEFAULT


@pytest.fixture
def synthetic(monkeypatch):
    config = deepcopy(DEFAULT)
    config["field"]["precision"] = "fp32"
    issues = np.array(["2023-06-15T01:00", "2023-07-15T01:00",
                       "2023-08-15T01:00", "2023-09-15T01:00"],
                      dtype="datetime64[ns]").astype(np.int64)
    mask = np.ones((256, 256), bool)
    mask[0] = False
    calls, accesses = [], []

    class InputOnly(dict):
        def __getitem__(self, name):
            assert name != "target", "Inference must not read future observations"
            return super().__getitem__(name)

    class Dataset:
        def __init__(self, supplied, split, require_future=True):
            assert supplied is config
            calls.append((split, require_future))
            self.issue_times_ns, self.spatial_mask = issues, mask
            self.provenance = {"fixture": True, "require_future": require_future}

        def __getitem__(self, index):
            accesses.append(index)
            return InputOnly(history=np.zeros((7, 256, 256), np.float32),
                             mask=mask, issue_ns=int(issues[index]))

    def predict(model, history, stamps, supplied):
        assert supplied is config and history.shape == (1, 7, 256, 256)
        assert history.device.type == "cpu" and not torch.is_grad_enabled()
        offset = float(np.searchsorted(issues, stamps[0]))
        return torch.from_numpy(np.broadcast_to(
            np.arange(4, dtype=np.float32)[:, None, None, None, None] + offset,
            (4, 1, 18, 256, 256)).copy())

    monkeypatch.setattr(bank, "HSRDataset", Dataset)
    monkeypatch.setattr(bank, "load_forecaster", lambda *_: (
        torch.nn.Identity(), {"epoch": 8, "checkpoint_sha256": "a" * 64}))
    monkeypatch.setattr(bank, "predict_members", predict)
    return SimpleNamespace(config=config, issues=issues, mask=mask, predict=predict,
                           calls=calls, accesses=accesses)


def test_complete_bank_is_input_only_cpu_member_mean_and_resumable(synthetic, tmp_path):
    output = tmp_path / "bank"
    selected = synthetic.issues[[0, 2]]
    result = bank.export(synthetic.config, output, split="validation", issue_times=selected)
    assert result["completed"] and result["committed_issues"] == 2
    assert synthetic.calls == [("validation", False)]
    assert synthetic.accesses == [0, 2]
    rates = np.load(output / "mean_rate_mmph.npy")
    np.testing.assert_array_equal(rates[:, :, 20, 20], np.broadcast_to([[1.5], [3.5]], (2, 18)))
    assert not rates[:, :, ~synthetic.mask].any()
    assert result["rate_sha256"] == digest(output / "mean_rate_mmph.npy")
    assert json.loads((output / "contract.json").read_text())["selected_epoch"] == 8
    assert bank.export(synthetic.config, output, split="validation", issue_times=selected, resume=True) == result
    assert synthetic.accesses == [0, 2]  # completed resume does not infer again
    with pytest.raises(FileExistsError):
        bank.export(synthetic.config, output, split="validation", issue_times=selected)
    rates[0, 0, 20, 20] += 1
    np.save(output / "mean_rate_mmph.npy", rates)
    with pytest.raises(ValueError, match="checksum"):
        bank.export(synthetic.config, output, split="validation", issue_times=selected, resume=True)


def test_failed_uncommitted_tail_is_recomputed(synthetic, tmp_path, monkeypatch):
    output = tmp_path / "bank"

    def fail_second(model, history, stamps, supplied):
        if stamps[0] == synthetic.issues[1]:
            raise RuntimeError("fixture interruption")
        return synthetic.predict(model, history, stamps, supplied)

    monkeypatch.setattr(bank, "predict_members", fail_second)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        bank.export(synthetic.config, output, split="validation")
    assert json.loads((output / "status.json").read_text()) == {"state": "failed", "committed_issues": 0}
    monkeypatch.setattr(bank, "predict_members", synthetic.predict)
    result = bank.export(synthetic.config, output, split="validation", resume=True)
    assert result["committed_issues"] == 4
    assert synthetic.accesses == [0, 1, 0, 1, 2, 3]


def test_reject_training_year_bank_or_unknown_issue(synthetic, tmp_path):
    with pytest.raises(ValueError, match="upstream training"):
        bank.export(synthetic.config, tmp_path / "bad", split="train")
    with pytest.raises(ValueError, match="HSR-history eligible"):
        bank.export(synthetic.config, tmp_path / "bad", issue_times=synthetic.issues + bank.STEP_NS)
    assert not (tmp_path / "bad").exists()


def test_explicit_checkpoint_requires_hash_contract_and_matching_model(tmp_path, monkeypatch):
    config = deepcopy(DEFAULT)
    contract = {"config": deepcopy(config), "implementation": {"original": "archived-source-hash"}}
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "epoch_008.pt"
    torch.save({"stage": "generator", "diagnostic_only": False, "epoch": 8,
                "contract_identity": identity(contract), "model": model.state_dict()}, path)
    monkeypatch.setattr(bank, "build_model", lambda _: torch.nn.Linear(2, 1))
    loaded, metadata = bank.load_selected(config, path, digest(path), "cpu")
    assert not loaded.training and metadata["epoch"] == 8
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    with pytest.raises(ValueError, match="SHA256"):
        bank.load_selected(config, path, "f" * 64, "cpu")
    config["model"]["base_channels"] *= 2
    with pytest.raises(ValueError, match="Model geometry"):
        bank.load_selected(config, path, digest(path), "cpu")


def test_console_wrappers_return_success_for_diagnostic_dicts(monkeypatch, tmp_path):
    monkeypatch.setattr(field_cli, "main", lambda: {"diagnostic": "success"})
    assert field_cli.cli_main() == 0
    monkeypatch.setattr(bank, "load_field_config", lambda _: {})
    monkeypatch.setattr(bank, "export", lambda *args, **kwargs: {"completed": True})
    assert bank.main(["--config", str(tmp_path / "config.json"), "--output", str(tmp_path / "bank"),
                      "--split", "validation"]) == 0
