"""Staged HSR training and frozen, reproducible NowcastNet forecast products.

This is a KMA adaptation, not the original authors' unpublished training pipeline.
Evolution is fitted first; its selected checkpoint is frozen while the conditional
generator/discriminator are fitted. Only 2023 HSR selects field checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import digest, identity, implementation_identity, lock, write_json
from .data import HSRDataset
from .model import ModelConfig, NowcastNet, Discriminator
from .losses import evolution_loss, generator_loss, discriminator_loss
from .sampling import build_training_weights


def build_model(config):
    names = {f.name for f in fields(ModelConfig)}
    return NowcastNet(ModelConfig(**{k: v for k, v in config["model"].items() if k in names}))


def _autocast(config, device):
    return (torch.autocast("cuda", dtype=torch.bfloat16)
            if torch.device(device).type == "cuda" and config["field"]["precision"] == "bf16"
            else nullcontext())


def noise_seed(config, issue_ns, member):
    value = f'{config["forecast_product"]["noise_seed"]}:{int(issue_ns)}:{int(member)}'
    return int.from_bytes(hashlib.sha256(value.encode("ascii")).digest()[:8], "little") & ((1 << 63) - 1)


@contextmanager
def _preserve_noise_projection_state(model):
    """Keep a selected checkpoint fixed across forecast-product calls.

    The released noise-projector SN updates u/v even in eval mode. Retain its
    exact operations and sequential updates within a four-draw product, but
    restore these non-gradient PARAMETERS afterwards (including on failure).
    This is an explicit evaluation protocol, not a claim about original training.
    """
    projector = getattr(model, "noise_projector", None)
    values = [] if projector is None else [
        value for name, value in projector.named_parameters()
        if name.endswith(("weight_u", "weight_v"))]
    snapshots = [value.detach().clone() for value in values]
    try:
        yield
    finally:
        with torch.no_grad():
            for value, saved in zip(values, snapshots):
                value.copy_(saved)


@torch.no_grad()
def predict_members(model, history, issues_ns, config):
    """Return fixed-checkpoint, issuance-keyed [member,batch,lead,y,x] draws."""
    model.eval()
    stamps = np.asarray(issues_ns, dtype=np.int64).reshape(-1)
    if len(stamps) != len(history):
        raise ValueError("One KST issuance timestamp is required for each history")
    if not torch.isfinite(history).all():
        raise ValueError("Nonfinite HSR history")
    with _preserve_noise_projection_state(model), _autocast(config, history.device):
        evolved = model.evolve(history)
        members = []
        for member in range(config["forecast_product"]["ensemble_members"]):
            noises = []
            for stamp in stamps:
                generator = torch.Generator(device=history.device).manual_seed(noise_seed(config, stamp, member))
                noises.append(model.sample_noise(1, history.shape[-2], history.shape[-1], history.device, generator=generator))
            sample = model.generate(history, evolved, noise=torch.cat(noises, dim=0)).float()
            if not torch.isfinite(sample).all():
                raise FloatingPointError("Nonfinite NowcastNet prediction; do not export a broken field")
            sample = sample.clamp_min(0)
            members.append(sample)
    return torch.stack(members)


@torch.no_grad()
def predict_product(model, history, issues_ns, config):
    """The same four-draw mean for selection, CNN fitting and test prediction."""
    return predict_members(model, history, issues_ns, config).mean(dim=0)


def rate_to_rn60(rate):
    """18 ten-minute rates -> 13 trailing-hour amounts ending +60..+180 min."""
    if rate.ndim != 4 or rate.shape[1] != 18:
        raise ValueError("Expected [batch,18,y,x] instantaneous mm/h")
    return rate.unfold(1, 6, 1).sum(dim=-1) / 6.0


class FieldScores:
    def __init__(self):
        self.hits = np.zeros((13, 4), dtype=np.int64)
        self.misses = self.hits.copy()
        self.false_alarms = self.hits.copy()
        self.pairs = np.zeros(13, dtype=np.int64)

    def update(self, prediction, target, mask):
        pred = rate_to_rn60(prediction.float().clamp_min(0))
        obs = rate_to_rn60(target.float())
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        valid = mask[:, None].bool() & torch.isfinite(pred) & torch.isfinite(obs)
        self.pairs += valid.sum(dim=(0, 2, 3)).cpu().numpy()
        for index, threshold in enumerate((1., 5., 10., 20.)):
            p, o = pred >= threshold, obs >= threshold
            self.hits[:, index] += (valid & p & o).sum(dim=(0, 2, 3)).cpu().numpy()
            self.misses[:, index] += (valid & ~p & o).sum(dim=(0, 2, 3)).cpu().numpy()
            self.false_alarms[:, index] += (valid & p & ~o).sum(dim=(0, 2, 3)).cpu().numpy()

    def result(self):
        if np.any(self.pairs == 0):
            raise ValueError("Field validation has no valid observation/prediction pairs at some leads")
        denominator = self.hits + self.misses + self.false_alarms
        csi = self.hits / (denominator + 1e-6)
        return {"macro_csi": float(csi.mean()), "csi": csi.tolist(),
                "hits": self.hits.tolist(), "misses": self.misses.tolist(),
                "false_alarms": self.false_alarms.tolist(), "pairs": self.pairs.tolist(),
                "leads_min": list(range(60, 181, 10)), "thresholds_mm": [1, 5, 10, 20],
                "reference": "future observed HSR converted by fixed M-P; NOT AWS or RRBF",
                "zero_event_cells": int((denominator == 0).sum()),
                "zero_event_convention": "CSI=0 when H+M+F=0, matching the existing CSI implementation",
                "selection": "equal mean over 13 leads x 4 thresholds; pooled grid/time contingency counts"}


def training_draw_count(dataset, config):
    """A fixed epoch of draws, rounded up to keep EVERY physical batch full."""
    batch_size = config["field"]["batch_size"]
    return math.ceil(len(dataset) / batch_size) * batch_size


def _loader(dataset, config, training, generator=None, weights=None):
    options = config["field"]
    sampler = None
    if training and options["importance_sampling"]:
        if weights is None:
            weights, _ = build_training_weights(dataset, config)
        sampler = WeightedRandomSampler(weights, training_draw_count(dataset, config),
                                        replacement=True, generator=generator)
    worker_options = {}
    if options["num_workers"]:
        # Workers touch CPU arrays only and start cleanly after CUDA initialization.
        # Keep FIFO delivery and recreate workers each epoch: persistent workers
        # would change DataLoader base-seed consumption on the shared sampler RNG.
        worker_options.update(prefetch_factor=options.get("prefetch_factor", 2),
                              multiprocessing_context="spawn", persistent_workers=False)
    return DataLoader(dataset, batch_size=options["batch_size" if training else "validation_batch_size"],
                      shuffle=training and sampler is None, sampler=sampler,
                      generator=generator, num_workers=options["num_workers"],
                      pin_memory=options.get("pin_memory", True), drop_last=False,
                      **worker_options)


def _batch_to_device(batch, device):
    """Queue pinned-host transfers on the current stream before model kernels.

    Host data are never mutated or reused while these transfers are pending.
    CPU/unpinned diagnostic batches remain supported with the same values.
    """
    return tuple(batch[key].to(device, non_blocking=True) for key in ("history", "target", "mask"))


@torch.no_grad()
def validate(model, loader, config, device, stage):
    model.eval()
    scores = FieldScores()
    member_scores = [FieldScores() for _ in range(config["forecast_product"]["ensemble_members"])] if stage == "generator" else []
    started = time.monotonic()
    for batch in loader:
        history, target, mask = _batch_to_device(batch, device)
        if stage == "evolution":
            with _autocast(config, device):
                prediction = model.evolve(history)["prediction"].float().clamp_min(0)
        else:
            members = predict_members(model, history, batch["issue_ns"].numpy(), config)
            prediction = members.mean(dim=0)
            for accumulator, member in zip(member_scores, members):
                accumulator.update(member, target, mask)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite validation output")
        scores.update(prediction, target, mask)
    result = scores.result()
    result["seconds"] = time.monotonic() - started
    if member_scores:
        result["member_scores"] = [value.result() for value in member_scores]
        result["mean_member_macro_csi"] = float(np.mean([value["macro_csi"] for value in result["member_scores"]]))
        result["primary_product"] = "score of four-member mean; not mean of member scores"
    return result


def _finite_step(loss, optimizer, parameters, clipping):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = list(parameters)
    # Infinity retains finite-gradient checks without imposing a clipping rule.
    norm = torch.nn.utils.clip_grad_norm_(parameters, math.inf if clipping is None else clipping,
                                        error_if_nonfinite=True)
    optimizer.step()
    return float(norm.detach())


def build_lr_scheduler(config, optimizer):
    """Reduce after exactly four bad validations, independently of early stop.

    PyTorch reduces when num_bad_epochs > patience, hence the minus one.
    The same absolute improvement rule governs both plateau and stopping.
    """
    settings = config["field"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=settings["lr_factor"], patience=settings["lr_patience"] - 1,
        threshold=settings["min_delta"], threshold_mode="abs", cooldown=0,
        min_lr=settings["lr_min"], eps=settings["lr_eps"])


def training_step(model, discriminator, optimizer, d_optimizer, batch, config, stage, device):
    """One actual physical batch. Used unchanged by training and GPU profiling."""
    f = config["field"]
    history, target, mask = _batch_to_device(batch, device)
    if not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite field target")
    if stage == "evolution":
        with _autocast(config, device):
            evolved = model.evolve(history)
            parts = evolution_loss(evolved["prediction"].float(), target, evolved["motion"].float(), mask,
                                   advected=evolved["advected"].float(), motion_weight=f["motion_weight"])
        norm = _finite_step(parts["total"], optimizer, model.evolution_parameters(), f["gradient_clip"])
        return {**{key: float(value.detach()) for key, value in parts.items()}, "gradient_norm": norm}
    if stage != "generator":
        raise ValueError("Unknown training stage")
    with torch.no_grad(), _autocast(config, device):
        evolved = model.evolve(history)
    with _autocast(config, device):
        discriminator.train()
        samples = torch.stack([model.generate(history, evolved) for _ in range(f["pool_samples"])])
        valid = mask[:, None].bool()
        real_logits = discriminator(history, torch.where(valid, target, 0.))
        fake_logits = discriminator(history, torch.where(valid, samples[0].detach(), 0.))
        d_parts = discriminator_loss(real_logits.float(), fake_logits.float())
    d_norm = _finite_step(d_parts["total"], d_optimizer, discriminator.parameters(), f["gradient_clip"])
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    # Freeze D's spectral state as well as its weights. D-only BN has no
    # running statistics, so eval still uses the current batch, just as in
    # D's real/fake forwards. Gradients through D to the generator stay enabled.
    discriminator.eval()
    with _autocast(config, device):
        logits = discriminator(history, torch.where(valid, samples[0], 0.))
        parts = generator_loss(samples.float(), target, logits.float(), mask,
                               adversarial_weight=f["adversarial_weight"], pool_weight=f["pool_weight"])
    norm = _finite_step(parts["total"], optimizer, model.generation_parameters(), f["gradient_clip"])
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    return {**{key: float(value.detach()) for key, value in parts.items()},
            **{"discriminator_" + key: float(value.detach()) for key, value in d_parts.items()},
            "gradient_norm": norm, "discriminator_gradient_norm": d_norm,
            "weighted_adversarial": f["adversarial_weight"] * float(parts["adversarial"].detach()),
            "weighted_pool": f["pool_weight"] * float(parts["pool"].detach())}


def _write_history(path, records):
    """Checkpoint-owned history: recovery replaces, rather than appends duplicates."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    temporary.replace(path)


def _save(path, state):
    temporary = Path(str(path) + ".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def _load(path):
    # Only locally generated checkpoints, never an arbitrary downloaded pickle.
    return torch.load(path, map_location="cpu", weights_only=False)


def _archive_epoch(directory, saved):
    """Retain the exact full last.pt commit, without replacing any archive.

    Copy first, then publish with an exclusive hard link: a crash leaves at
    most a disposable .partial, never a partially written epoch checkpoint.
    Unlike linking last.pt itself, each archive also owns independent bytes.
    """
    source = directory / "last.pt"
    path = directory / f"epoch_{saved['epoch']:03d}.pt"
    if path.is_symlink():
        raise ValueError(f"Refusing a symlinked epoch checkpoint: {path}")
    if path.exists():
        if digest(path) != digest(source):
            raise ValueError(f"Existing epoch checkpoint differs from last.pt; refusing to overwrite {path}")
        return
    temporary = path.with_name(path.name + ".partial")
    try:
        shutil.copyfile(source, temporary)
        os.link(temporary, path)  # Atomic publication; fails if path already exists.
    finally:
        temporary.unlink(missing_ok=True)


def _recover_epoch_commit(directory, contract, stage, completed=False):
    """Validate and recover only the epoch whose full state exists in last.pt.

    Older missing archives cannot be reconstructed from history or best.pt;
    they are deliberately left missing rather than mislabeled with new weights.
    """
    if not (directory / "last.pt").exists():
        raise ValueError("Missing last checkpoint after an epoch commit; refusing a silent restart")
    saved = _load(directory / "last.pt")
    required = {"stage", "epoch", "model", "optimizer", "scheduler", "best_score", "stale", "global_step",
                "best_epoch", "history", "stopping_anchor", "diagnostic_only", "contract_identity",
                "shuffle_rng", "torch_rng", "numpy_rng", "python_rng", "cuda_rng"}
    if stage == "generator":
        required.update(("discriminator", "discriminator_optimizer", "discriminator_scheduler"))
    if not isinstance(saved, dict) or required - saved.keys():
        raise ValueError("Last checkpoint is missing full training/resume state")
    if (saved["contract_identity"] != identity(contract) or saved["stage"] != stage
            or saved["diagnostic_only"]):
        raise ValueError("Last checkpoint does not match field contract/stage")
    epoch, records = saved["epoch"], saved["history"]
    if (type(epoch) is not int or epoch < 1 or not isinstance(records, list)
            or len(records) != epoch or [row.get("epoch") for row in records] != list(range(1, epoch + 1))
            or records[-1].get("global_step") != saved["global_step"]
            or not 1 <= saved["best_epoch"] <= epoch):
        raise ValueError("Last checkpoint epoch/history/iteration state is inconsistent")
    for archive in directory.glob("epoch_*.pt"):
        try:
            archive_epoch = int(archive.stem.removeprefix("epoch_"))
        except ValueError:
            raise ValueError(f"Unrecognized epoch checkpoint: {archive}") from None
        if archive_epoch > epoch:
            raise ValueError(f"Epoch checkpoint is ahead of last.pt; refusing a silent restart: {archive}")
    # last.pt commits before best/history, so a winning state can repair an
    # interrupted best commit. A completed run's selected hash is already checked.
    if not completed and saved["best_epoch"] == epoch:
        _save(directory / "best.pt", saved)
    chosen = _load(directory / "best.pt")
    if (chosen["stage"] != stage or chosen["diagnostic_only"]
            or chosen["contract_identity"] != saved["contract_identity"]
            or chosen["epoch"] != saved["best_epoch"] or chosen["best_score"] != saved["best_score"]):
        raise ValueError("Best/last checkpoint pair is inconsistent (possibly interrupted commit); do not silently resume")
    _archive_epoch(directory, saved)
    _write_history(directory / "history.jsonl", records)
    return saved


def _model_metadata(config, path, checkpoint):
    return {"checkpoint_path": str(Path(path).resolve()), "checkpoint_sha256": digest(path),
            "diagnostic_only": bool(checkpoint["diagnostic_only"]), "stage": checkpoint["stage"],
            "epoch": checkpoint["epoch"], "validation_macro_csi": checkpoint["best_score"],
            "forecast_product": config["forecast_product"], "model": config["model"],
            "hsr_source": "M-P rain rate from HSR; no RRBF target",
            "contract_identity": checkpoint["contract_identity"]}


def load_forecaster(config, device="cpu"):
    directory = Path(config["experiment_root"]) / "field" / "generator"
    status_path = directory / "status.json"
    if not status_path.exists() or json.loads(status_path.read_text())["state"] != "completed":
        raise RuntimeError("A completed full generator stage is required; evolution-only is not the full NowcastNet product")
    path = directory / "best.pt"
    checkpoint = _load(path)
    contract = json.loads((directory / "contract.json").read_text())
    if (checkpoint["stage"] != "generator" or checkpoint["diagnostic_only"]
            or identity(contract) != checkpoint["contract_identity"]
            or contract["implementation"] != implementation_identity()
            or contract["config"] != config):
        raise ValueError("Selected field checkpoint/config/implementation mismatch or diagnostic run")
    if digest(path) != json.loads(status_path.read_text())["best_checkpoint_sha256"]:
        raise ValueError("Selected field checkpoint bytes changed")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, _model_metadata(config, path, checkpoint)


def train_stage(config, stage, device="cuda:0", resume=False, *, before_training=None):
    if stage not in ("evolution", "generator"):
        raise ValueError("stage must be evolution or generator")
    directory = Path(config["experiment_root"]) / "field" / stage
    with lock(directory):
        return _train_stage_locked(config, stage, device, resume, directory, before_training)


def _train_stage_locked(config, stage, device, resume, directory, before_training=None):
    f = config["field"]
    training = HSRDataset(config, "train", require_future=True)
    validation = HSRDataset(config, "validation", require_future=True)
    parent = None
    if stage == "generator":
        previous = directory.parent / "evolution"
        if not (previous / "status.json").exists() or json.loads((previous / "status.json").read_text())["state"] != "completed":
            raise RuntimeError("Finish evolution training before fitting the stochastic generator")
        parent = {"checkpoint_path": str(previous / "best.pt"), "sha256": digest(previous / "best.pt")}
        if parent["sha256"] != json.loads((previous / "status.json").read_text())["best_checkpoint_sha256"]:
            raise ValueError("Selected evolution checkpoint was changed")
    weights, sampling = build_training_weights(training, config) if f["importance_sampling"] else (
        None, {"method": "uniform shuffled traversal; explicit non-paper override"})
    sampling = {**sampling, "draws_per_epoch": training_draw_count(training, config) if weights is not None else len(training),
                "physical_batch_size": f["batch_size"], "replacement": weights is not None}
    contract = {"config": config, "implementation": implementation_identity(), "stage": stage,
                "train_data": training.provenance, "validation_data": validation.provenance,
                "parent": parent, "diagnostic_only": False, "training_sampling": sampling,
                "selection": "2023 HSR-MP RN60 macro CSI, all 13 leads x 4 thresholds; no TEST"}
    contract_path = directory / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Existing field run contract changed; use a new experiment_root")
        state_path = directory / "status.json"
        if state_path.exists() and json.loads(state_path.read_text())["state"] == "completed":
            if digest(directory / "best.pt") != json.loads(state_path.read_text())["best_checkpoint_sha256"]:
                raise ValueError("Completed field checkpoint bytes changed")
            if resume:
                _recover_epoch_commit(directory, contract, stage, completed=True)
            print(f"{stage}: already completed; keeping selected checkpoint", flush=True)
            return json.loads(state_path.read_text())
        if not resume:
            raise FileExistsError("Incomplete field run exists; pass --resume explicitly")
    elif resume:
        raise FileNotFoundError("No matching field run to resume")
    else:
        if any(path.name != "run.lock" for path in directory.iterdir()):
            raise FileExistsError("Refusing to overwrite an unowned nonempty field run")
    # Contract/checkpoint recovery above is CPU-only and holds the stage lock.
    # Defer the launcher guard and all runtime setup until training is needed.
    if before_training is not None:
        before_training()
    torch.set_num_threads(f["threads"])
    torch.manual_seed(f["seed"])
    np.random.seed(f["seed"])
    random.seed(f["seed"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
        if f["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Configured bf16 is not supported; set field.precision to fp32 explicitly")
    if shutil.disk_usage(directory).free < 8 * 1024 ** 3:
        raise OSError("At least 8 GiB free required for full epoch/best/last field checkpoints")
    if not contract_path.exists():
        write_json(contract_path, contract)
    model = build_model(config).to(device)
    discriminator = None
    if parent:
        previous = _load(parent["checkpoint_path"])
        previous_contract = json.loads((Path(parent["checkpoint_path"]).parent / "contract.json").read_text())
        if (previous_contract["config"] != config or previous_contract["implementation"] != implementation_identity()
                or previous_contract["train_data"] != training.provenance
                or previous_contract["validation_data"] != validation.provenance):
            raise ValueError("Evolution checkpoint configuration/code changed")
        model.load_state_dict(previous["model"])
        del previous
        for parameter in model.evolution_parameters():
            parameter.requires_grad_(False)
        discriminator = Discriminator(in_steps=7, out_steps=18, base_channels=config["model"]["base_channels"],
                                      input_scale=f["discriminator_input_scale"]).to(device)
    parameters = list(model.evolution_parameters() if stage == "evolution" else model.generation_parameters())
    optimizer = torch.optim.Adam(parameters, lr=f[stage + "_lr"], betas=tuple(f["adam_betas"]))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=f["discriminator_lr"], betas=tuple(f["adam_betas"])) if discriminator else None
    scheduler = build_lr_scheduler(config, optimizer)
    d_scheduler = build_lr_scheduler(config, d_optimizer) if discriminator else None
    shuffle = torch.Generator().manual_seed(f["seed"])
    start, best, stale, anchor = 1, -math.inf, 0, -math.inf
    global_step, best_epoch, records = 0, 0, []
    if resume and not (directory / "last.pt").exists():
        if ((directory / "best.pt").exists() or (directory / "history.jsonl").exists()
                or any(directory.glob("epoch_*.pt"))):
            raise ValueError("Missing last checkpoint after an epoch commit; refusing a silent restart")
        print("No completed epoch checkpoint; restarting the interrupted first epoch with the same seed", flush=True)
    if resume and (directory / "last.pt").exists():
        saved = _recover_epoch_commit(directory, contract, stage)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if discriminator:
            discriminator.load_state_dict(saved["discriminator"])
            d_optimizer.load_state_dict(saved["discriminator_optimizer"])
            d_scheduler.load_state_dict(saved["discriminator_scheduler"])
        start, best, stale, anchor = saved["epoch"] + 1, saved["best_score"], saved["stale"], saved["stopping_anchor"]
        global_step, best_epoch, records = saved["global_step"], saved["best_epoch"], saved["history"]
        shuffle.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if torch.device(device).type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        del saved
    train_loader, val_loader = _loader(training, config, True, shuffle, weights), _loader(validation, config, False)
    iteration_limit = f[stage + "_iterations"]
    epoch_limit = math.ceil(iteration_limit / len(train_loader))
    if f[stage + "_epochs"] is not None:
        epoch_limit = min(epoch_limit, f[stage + "_epochs"])
    write_json(directory / "status.json", {"state": "training", "stage": stage, "next_epoch": start})
    print(f"{stage}: train={len(training)} validation={len(validation)} actual batch={f['batch_size']} device={device}; no TEST samples", flush=True)
    print(f"Iterations={iteration_limit}; draws/epoch={sampling['draws_per_epoch']}; importance_sampling={f['importance_sampling']}; "
          f"patience={f['patience']} (0 disables); explicit epoch limit={f[stage + '_epochs']}; "
          f"LR x{f['lr_factor']} after {f['lr_patience']} non-improving validation macro CSI epochs; no automatic batch fallback", flush=True)
    print(f"Data loader: workers={f['num_workers']}; pin_memory={f.get('pin_memory', True)}; "
          f"prefetch_factor={f.get('prefetch_factor', 2) if f['num_workers'] else None}; "
          f"multiprocessing={'spawn' if f['num_workers'] else 'main'}; persistent_workers=False; "
          f"non_blocking_transfers=True; max_open_hsr_months={config.get('max_open_hsr_months', 2)}", flush=True)
    try:
        for epoch in range(start, epoch_limit + 1):
            if (f["patience"] and stale >= f["patience"]) or global_step >= iteration_limit:
                break
            begun = time.monotonic()
            model.train()
            if discriminator:
                # Keep the frozen evolution BN statistics fixed as well as its weights.
                for module in model.modules():
                    if module is getattr(model, "evolution", None):
                        module.eval()
                discriminator.train()
            totals, examples = {}, 0
            lr = optimizer.param_groups[0]["lr"]
            d_lr = d_optimizer.param_groups[0]["lr"] if discriminator else None
            for step, batch in enumerate(train_loader, 1):
                parts = training_step(model, discriminator, optimizer, d_optimizer, batch, config, stage, device)
                count = len(batch["history"])
                for key, value in parts.items():
                    totals[key] = totals.get(key, 0.) + value * count
                examples += count
                global_step += 1
                if step == 1 or step % f["log_every"] == 0:
                    elapsed = time.monotonic() - begun
                    detail = " ".join(f"{key}={value/examples:.6g}" for key, value in totals.items())
                    print(f"{stage} epoch={epoch} batch={step}/{len(train_loader)} iteration={global_step}/{iteration_limit} "
                          f"lr={lr:.6g} {detail} issues/s={examples/elapsed:.3f}", flush=True)
                if global_step >= iteration_limit:
                    break
            train_seconds = time.monotonic() - begun
            metrics = validate(model, val_loader, config, device, stage)
            score = metrics["macro_csi"]
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite validation macro CSI")
            improved = score > best
            best = max(best, score)
            if improved:
                best_epoch = epoch
            if score > anchor + f["min_delta"]:
                stale, anchor = 0, score
            else:
                stale += 1
            scheduler.step(score)
            if d_scheduler:
                d_scheduler.step(score)
            record = {"epoch": epoch, "stage": stage, "global_step": global_step,
                      "train_loss": totals["total"] / examples,
                      "loss_components": {key: value / examples for key, value in totals.items()},
                      "examples": examples, "learning_rate": lr,
                      "next_learning_rate": optimizer.param_groups[0]["lr"],
                      "train_seconds": train_seconds, "validation": metrics, "best_score": best, "stale": stale}
            if discriminator:
                record.update(discriminator_learning_rate=d_lr,
                              next_discriminator_learning_rate=d_optimizer.param_groups[0]["lr"])
            records.append(record)
            saved = {"stage": stage, "epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "best_score": best, "stale": stale,
                     "global_step": global_step, "best_epoch": best_epoch, "history": records,
                     "stopping_anchor": anchor, "diagnostic_only": False, "contract_identity": identity(contract),
                     "shuffle_rng": shuffle.get_state(), "torch_rng": torch.get_rng_state(),
                     "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                     "cuda_rng": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}
            if discriminator:
                saved.update(discriminator=discriminator.state_dict(), discriminator_optimizer=d_optimizer.state_dict(),
                             discriminator_scheduler=d_scheduler.state_dict())
            _save(directory / "last.pt", saved)
            _archive_epoch(directory, saved)
            if improved:
                _save(directory / "best.pt", saved)
            _write_history(directory / "history.jsonl", records)
            write_json(directory / "status.json", {"state": "training", "stage": stage, "epoch": epoch,
                                                   "global_step": global_step, "best_score": best, "stale": stale})
            print(f"{stage} epoch={epoch}: validation macro CSI={score:.6f}; best={best:.6f}; stale={stale}/{f['patience']}; train={train_seconds:.1f}s val={metrics['seconds']:.1f}s", flush=True)
    except BaseException as error:
        write_json(directory / "status.json", {"state": "interrupted", "stage": stage, "error": str(error),
                                               "resume": "last.pt is the last fully completed epoch"})
        raise
    selected = _load(directory / "best.pt")
    result = {"state": "completed", "stage": stage, "best_epoch": selected["epoch"],
              "best_score": selected["best_score"], "best_checkpoint_sha256": digest(directory / "best.pt"),
              "completed_iterations": global_step, "iteration_budget": iteration_limit,
              "stopped_by": "patience" if f["patience"] and stale >= f["patience"] else (
                  "iteration_budget" if global_step >= iteration_limit else "explicit_epoch_limit")}
    write_json(directory / "status.json", result)
    return result
