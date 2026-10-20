from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nowcastnet_hsr.config import load_config, validate_config
from nowcastnet_hsr import trainer


def tiny_config(tmp_path):
    c = load_config()
    c["experiment_root"] = str(tmp_path / "experiment")
    c["cache_root"] = str(tmp_path / "cache")
    c["model"].update(grid_size=64, base_channels=4, noise_channels=4)
    c["field"].update(precision="fp32", threads=2, evolution_epochs=1, generator_epochs=1,
                      evolution_iterations=1, generator_iterations=1, importance_sampling=False,
                      batch_size=1, validation_batch_size=1, log_every=10)
    return c


class ToyDataset:
    def __init__(self, config, split, require_future=True):
        assert split in ("train", "validation")  # This entire trainer must never request TEST.
        self.provenance = {"split": split, "synthetic": True}
        self.split = split

    def __len__(self):
        return 1

    def __getitem__(self, index):
        rng = np.random.default_rng(1 if self.split == "train" else 2)
        mask = np.ones((64, 64), dtype=bool)
        mask[:4] = False
        return {"history": rng.uniform(0, 8, (7, 64, 64)).astype("float32") * mask,
                "target": rng.uniform(0, 25, (18, 64, 64)).astype("float32") * mask,
                "mask": mask, "issue_ns": np.int64(1717200000000000000)}


def test_rn60_alignment_and_mask():
    values = torch.arange(1, 19, dtype=torch.float32)[None, :, None, None]
    amount = trainer.rate_to_rn60(values)
    assert amount.shape == (1, 13, 1, 1)
    assert amount[0, 0, 0, 0] == 3.5
    assert amount[0, -1, 0, 0] == 15.5
    score = trainer.FieldScores()
    score.update(values, values, torch.ones(1, 1, 1, dtype=torch.bool))
    assert np.all(np.array(score.result()["csi"])[:, 0] > .999)
    with pytest.raises(ValueError, match="no valid"):
        trainer.FieldScores().result()


@pytest.mark.parametrize("initialization_seed", [0, 17, 40])
def test_noise_product_is_issuance_keyed(tmp_path, initialization_seed):
    torch.set_num_threads(2)
    torch.manual_seed(initialization_seed)
    config = tiny_config(tmp_path)
    model = trainer.build_model(config)
    history = torch.rand(2, 7, 64, 64, generator=torch.Generator().manual_seed(19)) * 8
    noise = model.sample_noise(2, 64, 64, "cpu", torch.Generator().manual_seed(23))
    # The released legacy spectral norm starts with random power vectors.
    # A freshly initialized eval-only model has uncalibrated denominators and
    # can magnify normal batched-vs-single FP32 roundoff. Warm only SN/BN state
    # as training does; no weights, production behavior or tolerance are changed.
    with torch.no_grad():
        for _ in range(3):
            model(history, noise)
    model.eval()
    issues = np.array([1717200000000000000, 1717200600000000000], dtype=np.int64)
    combined = trainer.predict_product(model, history, issues, config)
    singles = torch.cat([trainer.predict_product(model, history[i:i+1], issues[i:i+1], config) for i in range(2)])
    torch.testing.assert_close(combined, singles, atol=2e-4, rtol=2e-5)
    assert (combined >= 0).all()
    assert trainer.noise_seed(config, issues[0], 0) != trainer.noise_seed(config, issues[0], 1)


def test_complete_two_stage_cpu_training_and_loading(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    config = tiny_config(tmp_path)
    monkeypatch.setattr(trainer, "HSRDataset", ToyDataset)
    evo = trainer.train_stage(config, "evolution", "cpu")
    assert evo["state"] == "completed"
    selected_evolution = trainer._load(Path(config["experiment_root"]) / "field/evolution/best.pt")
    gen = trainer.train_stage(config, "generator", "cpu")
    assert gen["state"] == "completed"
    for stage in ("evolution", "generator"):
        directory = Path(config["experiment_root"]) / "field" / stage
        saved = trainer._load(directory / "epoch_001.pt")
        assert trainer.digest(directory / "epoch_001.pt") == trainer.digest(directory / "last.pt")
        assert saved["stage"] == stage and saved["global_step"] == saved["epoch"] == 1
        assert saved["model"] and saved["optimizer"]["state"]
        assert saved["scheduler"]["last_epoch"] == 1
        assert all(key in saved for key in ("shuffle_rng", "torch_rng", "numpy_rng", "python_rng", "cuda_rng"))
        if stage == "generator":
            assert saved["discriminator"] and saved["discriminator_optimizer"]["state"]
            assert saved["discriminator_scheduler"] == saved["scheduler"]
    model, meta = trainer.load_forecaster(config)
    assert not meta["diagnostic_only"] and meta["stage"] == "generator"
    # All evolution parameters AND BN/spectral-normalization buffers remain frozen.
    for key, value in model.state_dict().items():
        if key.startswith("evolution."):
            torch.testing.assert_close(value, selected_evolution["model"][key], rtol=0, atol=0)
    assert trainer.train_stage(config, "generator", "cpu")["best_epoch"] == 1
    changed = deepcopy(config)
    changed["field"]["seed"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        trainer.load_forecaster(changed)


def test_evolution_only_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="completed full generator"):
        trainer.load_forecaster(tiny_config(tmp_path))


def test_generator_step_freezes_discriminator_weights_bn_and_spectral_state(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(40)
    config = tiny_config(tmp_path)
    model = trainer.build_model(config).train()
    for parameter in model.evolution_parameters():
        parameter.requires_grad_(False)
    model.evolution.eval()
    discriminator = trainer.Discriminator(in_steps=7, out_steps=18, base_channels=4,
                                         input_scale=config["field"]["discriminator_input_scale"])
    optimizer = torch.optim.Adam(model.generation_parameters(), lr=3e-5)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3e-5)
    example = ToyDataset(config, "train")[0]
    batch = {key: torch.as_tensor(example[key])[None] for key in ("history", "target", "mask")}
    original_step = trainer._finite_step
    committed_d_state = {}
    norms = {name: module for name, module in discriminator.named_modules()
             if isinstance(module, torch.nn.BatchNorm2d)}
    assert len(norms) == 9
    normalization_calls = {name: [] for name in norms}
    adversarial_path = {}
    handles = []
    def check_batch_normalization(name):
        def check(module, inputs, output):
            assert not module.track_running_stats
            assert module.running_mean is None and module.running_var is None
            assert module.num_batches_tracked is None
            with torch.no_grad():
                expected = torch.nn.functional.batch_norm(
                    inputs[0].detach(), None, None, module.weight.detach(), module.bias.detach(),
                    training=True, momentum=module.momentum, eps=module.eps)
            torch.testing.assert_close(output.detach(), expected, rtol=0, atol=0)
            normalization_calls[name].append((module.training, discriminator.training))
        return check
    def retain_adversarial_input(module, inputs, output):
        if not module.training:
            assert all(not parameter.requires_grad for parameter in module.parameters())
            assert inputs[1].requires_grad and output.requires_grad
            inputs[1].retain_grad()
            adversarial_path["future"] = inputs[1]
    for name, module in norms.items():
        handles.append(module.register_forward_hook(check_batch_normalization(name)))
    handles.append(discriminator.register_forward_hook(retain_adversarial_input))
    def checked_step(loss, selected_optimizer, parameters, clipping):
        result = original_step(loss, selected_optimizer, parameters, clipping)
        if selected_optimizer is d_optimizer:
            committed_d_state.update({key: value.detach().clone()
                                      for key, value in discriminator.state_dict().items()})
        else:
            assert committed_d_state and not discriminator.training
            for key, value in discriminator.state_dict().items():
                torch.testing.assert_close(value, committed_d_state[key], rtol=0, atol=0)
        return result
    monkeypatch.setattr(trainer, "_finite_step", checked_step)
    try:
        result = trainer.training_step(model, discriminator, optimizer, d_optimizer, batch,
                                       config, "generator", "cpu")
    finally:
        for handle in handles:
            handle.remove()
    assert all(np.isfinite(value) for value in result.values())
    # Real/fake D calls are train; the G call is eval to freeze SN, but all
    # three calls use batch statistics. This would fail for default running BN.
    assert all(calls == [(True, True), (True, True), (False, False)]
               for calls in normalization_calls.values())
    gradient = adversarial_path["future"].grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert model.decoder.out.weight.grad is not None and model.decoder.out.weight.grad.abs().sum() > 0
    assert result["weighted_adversarial"] == pytest.approx(6 * result["adversarial"])
    assert result["weighted_pool"] == pytest.approx(20 * result["pool"])
    assert all(parameter.requires_grad for parameter in discriminator.parameters())


def test_config_guards():
    config = load_config()
    validate_config(config)
    bad = deepcopy(config)
    bad["experiment_root"] = str(Path(config["project_root"]) / "src/bad")
    with pytest.raises(ValueError, match="protected"):
        validate_config(bad)
    bad = deepcopy(config)
    bad["test_years"] = [2023]
    with pytest.raises(ValueError, match="freezes"):
        validate_config(bad)
    bad = deepcopy(config)
    bad["model"]["grid_size"] = 128
    with pytest.raises(ValueError, match="Production grid"):
        validate_config(bad)


def test_training_defaults_keep_paper_initial_rates_with_requested_plateau():
    config = load_config()
    f = config["field"]
    assert f["batch_size"] == 16 and f["importance_sampling"]
    assert f["evolution_iterations"] == 300000 and f["generator_iterations"] == 500000
    assert f["evolution_epochs"] == f["generator_epochs"] == 30
    assert f["patience"] == 8 and f["min_delta"] == 0 and f["gradient_clip"] is None
    assert config["model"]["rain_rate_cap"] == 128
    assert f["discriminator_input_scale"] == 1.0
    assert f["evolution_lr"] == 1e-3 and f["generator_lr"] == f["discriminator_lr"] == 3e-5
    assert f["lr_scheduler"] == "plateau" and f["lr_monitor"] == "validation_macro_csi"
    assert f["lr_patience"] == 4 and f["lr_factor"] == .1
    assert f["lr_min"] == 0 and f["lr_eps"] == 1e-8
    assert "evolution_lr_decay_iteration" not in f and "evolution_lr_after_decay" not in f


def test_weighted_loader_uses_full_physical_batches(tmp_path):
    config = tiny_config(tmp_path)
    config["field"].update(batch_size=16, importance_sampling=True)
    dataset = torch.utils.data.TensorDataset(torch.arange(35))
    weights = torch.arange(1, 36, dtype=torch.float64)
    def batches():
        return list(trainer._loader(dataset, config, True, torch.Generator().manual_seed(40), weights))
    first, second = batches(), batches()
    assert len(first) == 3 and all(len(batch[0]) == 16 for batch in first)
    assert torch.equal(torch.cat([x[0] for x in first]), torch.cat([x[0] for x in second]))
    assert trainer.training_draw_count(dataset, config) == 48
    validation = trainer._loader(dataset, config, False)
    assert torch.equal(torch.cat([x[0] for x in validation]), torch.arange(35))


def test_iteration_limit_stops_mid_epoch_and_records_lr(tmp_path, monkeypatch):
    class ThreeWindows(ToyDataset):
        def __len__(self):
            return 3
    config = tiny_config(tmp_path)
    config["field"].update(evolution_epochs=None, evolution_iterations=2)
    monkeypatch.setattr(trainer, "HSRDataset", ThreeWindows)
    result = trainer.train_stage(config, "evolution", "cpu")
    assert result["completed_iterations"] == 2 and result["stopped_by"] == "iteration_budget"
    saved = trainer._load(Path(config["experiment_root"]) / "field/evolution/last.pt")
    assert saved["epoch"] == 1 and saved["global_step"] == 2
    assert saved["history"][0]["examples"] == 2
    assert saved["history"][0]["learning_rate"] == 1e-3
    assert saved["history"][0]["next_learning_rate"] == 1e-3
    assert saved["scheduler"]["last_epoch"] == 1
    assert "accumulation" in saved["history"][0]["loss_components"]


def test_resume_repairs_history_after_checkpoint_commit(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    config["field"].update(evolution_epochs=None, evolution_iterations=2)
    monkeypatch.setattr(trainer, "HSRDataset", ToyDataset)
    write_history = trainer._write_history
    def interrupted(*args):
        raise RuntimeError("simulate history commit interruption")
    monkeypatch.setattr(trainer, "_write_history", interrupted)
    with pytest.raises(RuntimeError, match="history commit"):
        trainer.train_stage(config, "evolution", "cpu")
    directory = Path(config["experiment_root"]) / "field/evolution"
    (directory / "history.jsonl").write_text('{"epoch":999}\n')
    monkeypatch.setattr(trainer, "_write_history", write_history)
    trainer.train_stage(config, "evolution", "cpu", resume=True)
    records = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in records] == [1, 2]
    assert [row["global_step"] for row in records] == [1, 2]


def test_predict_members_mean_matches_product(tmp_path):
    torch.set_num_threads(2)
    config = tiny_config(tmp_path)
    model = trainer.build_model(config).eval()
    history = torch.rand(1, 7, 64, 64)
    issues = [1717200000000000000]
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    members = trainer.predict_members(model, history, issues, config)
    assert members.shape == (4, 1, 18, 64, 64)
    torch.testing.assert_close(members.mean(0), trainer.predict_product(model, history, issues, config), atol=0, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_prediction_restores_noise_spectral_state_on_error(tmp_path, monkeypatch):
    config = tiny_config(tmp_path)
    model = trainer.build_model(config).eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    generate = model.generate
    def broken(*args, **kwargs):
        generate(*args, **kwargs)  # Actually update the released spectral vectors.
        raise RuntimeError("prediction interrupted")
    monkeypatch.setattr(model, "generate", broken)
    with pytest.raises(RuntimeError, match="prediction interrupted"):
        trainer.predict_product(model, torch.rand(1, 7, 64, 64), [1717200000000000000], config)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_resume_repairs_best_commit_and_matches_uninterrupted_weights(tmp_path, monkeypatch):
    config = tiny_config(tmp_path / "reference")
    config["field"].update(evolution_epochs=None, evolution_iterations=2)
    monkeypatch.setattr(trainer, "HSRDataset", ToyDataset)
    trainer.train_stage(config, "evolution", "cpu")
    reference = trainer._load(Path(config["experiment_root"]) / "field/evolution/last.pt")
    config["experiment_root"] = str(tmp_path / "interrupted")
    original_save = trainer._save
    def interrupted(path, state):
        if Path(path).name == "best.pt":
            raise RuntimeError("interrupt before best commit")
        return original_save(path, state)
    monkeypatch.setattr(trainer, "_save", interrupted)
    with pytest.raises(RuntimeError, match="before best"):
        trainer.train_stage(config, "evolution", "cpu")
    directory = Path(config["experiment_root"]) / "field/evolution"
    assert (directory / "last.pt").exists() and not (directory / "best.pt").exists()
    monkeypatch.setattr(trainer, "_save", original_save)
    trainer.train_stage(config, "evolution", "cpu", resume=True)
    resumed = trainer._load(directory / "last.pt")
    assert resumed["global_step"] == reference["global_step"] == 2
    for key, value in reference["model"].items():
        torch.testing.assert_close(resumed["model"][key], value, rtol=0, atol=0)
    assert torch.equal(resumed["shuffle_rng"], reference["shuffle_rng"])
    assert torch.equal(resumed["torch_rng"], reference["torch_rng"])


class StubModel(torch.nn.Module):
    """Tiny parameters keep checkpoint protocol tests independent of GPU cost."""
    def __init__(self):
        super().__init__()
        self.evolution = torch.nn.Linear(1, 1)
        self.generation = torch.nn.Linear(1, 1)
        self.register_buffer("evolution_steps", torch.tensor(0))
        self.register_buffer("generator_steps", torch.tensor(0))

    def evolution_parameters(self):
        return self.evolution.parameters()

    def generation_parameters(self):
        return self.generation.parameters()


class StubDiscriminator(torch.nn.Linear):
    def __init__(self, **kwargs):
        super().__init__(1, 1)


@pytest.fixture
def checkpoint_training(monkeypatch):
    scores = {"evolution": [.9, .4, .3], "generator": [.8, .2, .1]}

    def step(model, discriminator, optimizer, d_optimizer, batch, config, stage, device):
        parameters = list(model.evolution_parameters() if stage == "evolution" else model.generation_parameters())
        # Exercise all restored RNG streams and real Adam state, cheaply on CPU.
        random_value = torch.rand(()) + np.random.random() + trainer.random.random()
        loss = sum((parameter.square() + random_value * parameter).sum() for parameter in parameters)
        trainer._finite_step(loss, optimizer, parameters, None)
        if discriminator is not None:
            d_loss = sum((parameter.square() + torch.rand(()) * parameter).sum()
                         for parameter in discriminator.parameters())
            trainer._finite_step(d_loss, d_optimizer, discriminator.parameters(), None)
        getattr(model, stage + "_steps").add_(1)
        return {"total": float(loss.detach())}

    def validation(model, loader, config, device, stage):
        index = min(int(getattr(model, stage + "_steps")) - 1, len(scores[stage]) - 1)
        return {"macro_csi": scores[stage][index], "seconds": 0.0}

    monkeypatch.setattr(trainer, "HSRDataset", ToyDataset)
    monkeypatch.setattr(trainer, "build_model", lambda config: StubModel())
    monkeypatch.setattr(trainer, "Discriminator", StubDiscriminator)
    monkeypatch.setattr(trainer, "training_step", step)
    monkeypatch.setattr(trainer, "validate", validation)
    return scores


def archive_config(tmp_path, iterations=3):
    config = tiny_config(tmp_path)
    config["field"].update(evolution_epochs=30, generator_epochs=30,
                           evolution_iterations=iterations, generator_iterations=iterations)
    return config


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            assert_nested_equal(value, reference)
    else:
        assert actual == expected


def test_all_epochs_keep_full_state_including_nonbest_final(tmp_path, checkpoint_training):
    config = archive_config(tmp_path)
    for stage in ("evolution", "generator"):
        result = trainer.train_stage(config, stage, "cpu")
        directory = Path(config["experiment_root"]) / "field" / stage
        assert result["best_epoch"] == 1 and result["completed_iterations"] == 3
        assert result["stopped_by"] == "iteration_budget"
        assert [path.name for path in sorted(directory.glob("epoch_*.pt"))] == [
            "epoch_001.pt", "epoch_002.pt", "epoch_003.pt"]
        for epoch in range(1, 4):
            saved = trainer._load(directory / f"epoch_{epoch:03d}.pt")
            assert saved["stage"] == stage and saved["epoch"] == saved["global_step"] == epoch
            assert saved["best_epoch"] == 1 and saved["stale"] == epoch - 1
            assert len(saved["history"]) == epoch and saved["optimizer"]["state"]
            assert saved["scheduler"]["last_epoch"] == epoch
            assert saved["model"][stage + "_steps"] == epoch
            assert all(key in saved for key in ("shuffle_rng", "torch_rng", "numpy_rng", "python_rng", "cuda_rng"))
            if stage == "generator":
                assert saved["discriminator"] and saved["discriminator_optimizer"]["state"]
                assert saved["discriminator_scheduler"] == saved["scheduler"]
        assert trainer._load(directory / "best.pt")["epoch"] == 1
        assert trainer.digest(directory / "epoch_003.pt") == trainer.digest(directory / "last.pt")


def test_patience_eight_is_independent_for_both_stages(tmp_path, checkpoint_training):
    config = archive_config(tmp_path, iterations=100)
    checkpoint_training.update(evolution=[.5], generator=[.4])
    for stage in ("evolution", "generator"):
        result = trainer.train_stage(config, stage, "cpu")
        directory = Path(config["experiment_root"]) / "field" / stage
        saved = trainer._load(directory / "epoch_009.pt")
        assert result["stopped_by"] == "patience" and result["completed_iterations"] == 9
        assert saved["stale"] == 8 and saved["best_epoch"] == 1
        initial_lr = config["field"][stage + "_lr"]
        assert [row["learning_rate"] for row in saved["history"]] == pytest.approx(
            [initial_lr] * 5 + [initial_lr * .1] * 4)
        assert [row["next_learning_rate"] for row in saved["history"]] == pytest.approx(
            [initial_lr] * 4 + [initial_lr * .1] * 4 + [initial_lr * .01])
        assert saved["scheduler"]["num_bad_epochs"] == 0
        assert saved["optimizer"]["param_groups"][0]["lr"] == pytest.approx(initial_lr * .01)
        if stage == "generator":
            assert saved["discriminator_scheduler"] == saved["scheduler"]
            assert saved["discriminator_optimizer"]["param_groups"][0]["lr"] == pytest.approx(initial_lr * .01)
            for row in saved["history"]:
                assert row["learning_rate"] == row["discriminator_learning_rate"]
                assert row["next_learning_rate"] == row["next_discriminator_learning_rate"]
        assert len(list(directory.glob("epoch_*.pt"))) == 9
        assert not (directory / "epoch_010.pt").exists()


def test_zero_delta_improvement_resets_patience(tmp_path, checkpoint_training):
    config = archive_config(tmp_path, iterations=100)
    checkpoint_training["evolution"] = [.5] * 8 + [.50000001] * 9
    result = trainer.train_stage(config, "evolution", "cpu")
    assert result["stopped_by"] == "patience"
    assert result["best_epoch"] == 9 and result["completed_iterations"] == 17
    directory = Path(config["experiment_root"]) / "field/evolution"
    assert trainer._load(directory / "epoch_008.pt")["scheduler"]["num_bad_epochs"] == 3
    improved = trainer._load(directory / "epoch_009.pt")
    assert improved["stale"] == improved["scheduler"]["num_bad_epochs"] == 0
    saved = trainer._load(directory / "last.pt")
    assert [row["learning_rate"] for row in saved["history"]] == pytest.approx(
        [1e-3] * 5 + [1e-4] * 8 + [1e-5] * 4)
    assert saved["history"][-1]["next_learning_rate"] == pytest.approx(1e-6)


def test_both_stages_stop_at_thirty_epochs_when_scores_keep_improving(tmp_path, checkpoint_training):
    config = archive_config(tmp_path, iterations=100)
    checkpoint_training.update(evolution=list(np.linspace(.1, .9, 30)),
                               generator=list(np.linspace(.1, .9, 30)))
    for stage in ("evolution", "generator"):
        result = trainer.train_stage(config, stage, "cpu")
        directory = Path(config["experiment_root"]) / "field" / stage
        saved = trainer._load(directory / "epoch_030.pt")
        assert result["stopped_by"] == "explicit_epoch_limit"
        assert result["completed_iterations"] == result["best_epoch"] == 30
        assert saved["stale"] == saved["scheduler"]["num_bad_epochs"] == 0
        assert saved["optimizer"]["param_groups"][0]["lr"] == config["field"][stage + "_lr"]
        assert not (directory / "epoch_031.pt").exists()


def test_generator_and_discriminator_decay_together_preserving_lr_ratio(tmp_path, checkpoint_training):
    config = archive_config(tmp_path, iterations=9)
    config["field"]["discriminator_lr"] = 6e-5
    trainer.train_stage(config, "evolution", "cpu")
    trainer.train_stage(config, "generator", "cpu")
    directory = Path(config["experiment_root"]) / "field/generator"
    saved = trainer._load(directory / "last.pt")
    for row in saved["history"]:
        assert row["discriminator_learning_rate"] == pytest.approx(row["learning_rate"] * 2)
        assert row["next_discriminator_learning_rate"] == pytest.approx(row["next_learning_rate"] * 2)
    for key in ("num_bad_epochs", "last_epoch", "best"):
        assert saved["scheduler"][key] == saved["discriminator_scheduler"][key]


def test_resume_rejects_old_fixed_schedule_contract(tmp_path, checkpoint_training):
    config = archive_config(tmp_path)
    trainer.train_stage(config, "evolution", "cpu")
    directory = Path(config["experiment_root"]) / "field/evolution"
    contract = json.loads((directory / "contract.json").read_text())
    contract["config"]["field"]["evolution_lr_decay_iteration"] = 200000
    trainer.write_json(directory / "contract.json", contract)
    with pytest.raises(ValueError, match="contract changed"):
        trainer.train_stage(config, "evolution", "cpu", resume=True)


@pytest.mark.parametrize("stage", ["evolution", "generator"])
def test_native_plateau_checkpoint_resume_matches_uninterrupted_weights(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(trainer, "HSRDataset", ToyDataset)
    monkeypatch.setattr(trainer, "validate", lambda *args: {"macro_csi": .5, "seconds": 0.0})
    config = tiny_config(tmp_path / "reference")
    config["field"].update({stage + "_epochs": 6, stage + "_iterations": 6})
    if stage == "generator":
        trainer.train_stage(config, "evolution", "cpu")
    trainer.train_stage(config, stage, "cpu")
    reference = trainer._load(Path(config["experiment_root"]) / "field" / stage / "last.pt")
    config["experiment_root"] = str(tmp_path / "interrupted")
    if stage == "generator":
        trainer.train_stage(config, "evolution", "cpu")
    write_history = trainer._write_history

    def interrupted(path, records):
        if len(records) == 4:
            raise RuntimeError("native interruption before LR reduction")
        write_history(path, records)

    monkeypatch.setattr(trainer, "_write_history", interrupted)
    with pytest.raises(RuntimeError, match="native interruption"):
        trainer.train_stage(config, stage, "cpu")
    monkeypatch.setattr(trainer, "_write_history", write_history)
    trainer.train_stage(config, stage, "cpu", resume=True)
    resumed = trainer._load(Path(config["experiment_root"]) / "field" / stage / "last.pt")
    for key in ("model", "optimizer", "scheduler", "global_step", "epoch", "best_epoch", "stale",
                "shuffle_rng", "torch_rng", "numpy_rng", "python_rng", "cuda_rng"):
        assert_nested_equal(resumed[key], reference[key])
    if stage == "generator":
        for key in ("discriminator", "discriminator_optimizer", "discriminator_scheduler"):
            assert_nested_equal(resumed[key], reference[key])
    assert resumed["epoch"] == 6 and resumed["scheduler"]["num_bad_epochs"] == 1
    assert resumed["optimizer"]["param_groups"][0]["lr"] == pytest.approx(config["field"][stage + "_lr"] * .1)


@pytest.mark.parametrize("stage", ["evolution", "generator"])
def test_resume_repairs_archive_after_last_commit_and_restores_full_state(
        tmp_path, monkeypatch, checkpoint_training, stage):
    config = archive_config(tmp_path / "reference", iterations=9)
    if stage == "generator":
        trainer.train_stage(config, "evolution", "cpu")
    trainer.train_stage(config, stage, "cpu")
    reference = trainer._load(Path(config["experiment_root"]) / "field" / stage / "last.pt")
    config = archive_config(tmp_path / "interrupted", iterations=9)
    if stage == "generator":
        trainer.train_stage(config, "evolution", "cpu")
    archive = trainer._archive_epoch

    def interrupted(directory, saved):
        if saved["epoch"] == 4:
            raise RuntimeError("interrupt before epoch archive publication")
        archive(directory, saved)

    monkeypatch.setattr(trainer, "_archive_epoch", interrupted)
    with pytest.raises(RuntimeError, match="archive publication"):
        trainer.train_stage(config, stage, "cpu")
    directory = Path(config["experiment_root"]) / "field" / stage
    assert trainer._load(directory / "last.pt")["epoch"] == 4
    assert not (directory / "epoch_004.pt").exists()
    monkeypatch.setattr(trainer, "_archive_epoch", archive)
    trainer.train_stage(config, stage, "cpu", resume=True)
    resumed = trainer._load(directory / "last.pt")
    for key in ("model", "optimizer", "scheduler", "global_step", "epoch", "best_epoch", "stale", "stopping_anchor",
                "shuffle_rng", "torch_rng", "numpy_rng", "python_rng", "cuda_rng"):
        assert_nested_equal(resumed[key], reference[key])
    if stage == "generator":
        for key in ("discriminator", "discriminator_optimizer", "discriminator_scheduler"):
            assert_nested_equal(resumed[key], reference[key])
    assert trainer._load(directory / "epoch_004.pt")["epoch"] == 4
    assert [row["epoch"] for row in resumed["history"]] == list(range(1, 10))
    assert resumed["scheduler"]["last_epoch"] == 9


@pytest.mark.parametrize("stage", ["evolution", "generator"])
def test_completed_resume_repairs_only_missing_last_archive(tmp_path, checkpoint_training, stage):
    config = archive_config(tmp_path)
    if stage == "generator":
        trainer.train_stage(config, "evolution", "cpu")
    expected = trainer.train_stage(config, stage, "cpu")
    directory = Path(config["experiment_root"]) / "field" / stage
    first_hash = trainer.digest(directory / "epoch_001.pt")
    (directory / "epoch_002.pt").unlink()
    (directory / "epoch_003.pt").unlink()
    assert trainer.train_stage(config, stage, "cpu", resume=True) == expected
    assert trainer.digest(directory / "epoch_003.pt") == trainer.digest(directory / "last.pt")
    assert trainer.digest(directory / "epoch_001.pt") == first_hash
    assert not (directory / "epoch_002.pt").exists()  # No invented older weights.
    assert trainer.digest(directory / "best.pt") == expected["best_checkpoint_sha256"]


@pytest.mark.parametrize("completed", [False, True])
def test_resume_never_overwrites_existing_epoch_archive(tmp_path, checkpoint_training, completed):
    config = archive_config(tmp_path)
    trainer.train_stage(config, "evolution", "cpu")
    directory = Path(config["experiment_root"]) / "field/evolution"
    if not completed:
        trainer.write_json(directory / "status.json", {"state": "interrupted"})
    (directory / "epoch_003.pt").write_bytes(b"unrelated retained checkpoint")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        trainer.train_stage(config, "evolution", "cpu", resume=True)
    assert (directory / "epoch_003.pt").read_bytes() == b"unrelated retained checkpoint"


def test_orphan_archive_prevents_first_epoch_restart(tmp_path, checkpoint_training):
    config = archive_config(tmp_path)
    trainer.train_stage(config, "evolution", "cpu")
    directory = Path(config["experiment_root"]) / "field/evolution"
    for name in ("last.pt", "best.pt", "history.jsonl"):
        (directory / name).unlink()
    trainer.write_json(directory / "status.json", {"state": "interrupted"})
    with pytest.raises(ValueError, match="Missing last checkpoint"):
        trainer.train_stage(config, "evolution", "cpu", resume=True)


@pytest.mark.parametrize("corruption", ["stage", "contract_identity", "optimizer", "scheduler"])
def test_completed_archive_repair_validates_last_before_writing(tmp_path, checkpoint_training, corruption):
    config = archive_config(tmp_path)
    trainer.train_stage(config, "evolution", "cpu")
    directory = Path(config["experiment_root"]) / "field/evolution"
    (directory / "epoch_003.pt").unlink()
    saved = trainer._load(directory / "last.pt")
    if corruption in ("optimizer", "scheduler"):
        del saved[corruption]
    else:
        saved[corruption] = "invalid"
    trainer._save(directory / "last.pt", saved)
    with pytest.raises(ValueError, match="Last checkpoint"):
        trainer.train_stage(config, "evolution", "cpu", resume=True)
    assert not (directory / "epoch_003.pt").exists()
