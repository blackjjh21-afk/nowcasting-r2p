"""Field-only entrypoint: no CNN, AWS, RRBF labels, downloads or implicit TEST."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

from .config import lock
from .field_config import load_field_config


def check(config):
    from .data import HSRDataset
    report = {"route": "HSR -> fixed M-P rain rate -> KMA-adapted NowcastNet",
              "training_launched": False, "test_opened": False,
              "cnn_used": False, "gauge_used": False, "rrbf_used": False, "splits": {}}
    for split in ("train", "validation"):
        dataset = HSRDataset(config, split, require_future=True, max_issues=1)
        item = dataset[0]
        report["splits"][split] = {
            "eligible_issues": dataset.provenance["eligible_issue_count_before_subset"],
            "history_shape": list(item["history"].shape), "target_shape": list(item["target"].shape),
            "valid_grid_cells": int(item["mask"].sum()),
            "backends": sorted({source["backend"] for source in dataset.provenance["sources"]})}
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def ensure_device(device, allow_shared_gpu=False):
    import torch
    selected = torch.device(device)
    if selected.type == "cpu":
        return
    if selected.type != "cuda" or selected.index is None:
        raise ValueError("Choose cpu or an explicit CUDA device, e.g. cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(selected)
    free, total = torch.cuda.mem_get_info(selected)
    if total - free > 2 * 1024**3 and not allow_shared_gpu:
        raise RuntimeError(f"{device} already uses {(total-free)/1024**3:.1f} GiB; "
                           "no work launched. Choose a free GPU; no other job is stopped.")
    if free < 10 * 1024**3:
        raise RuntimeError("Less than 10 GiB GPU memory free; no automatic batch fallback")


def smoke(config):
    """Synthetic CPU test of both training stages; temporary checkpoints only."""
    from copy import deepcopy
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    import numpy as np
    import torch
    from . import trainer

    class Synthetic:
        def __init__(self, config, split, require_future=True):
            if split not in ("train", "validation"):
                raise AssertionError("Smoke training must not open TEST")
            self.split = split
            self.provenance = {"split": split, "synthetic": True}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            rng = np.random.default_rng(1 if self.split == "train" else 2)
            mask = np.ones((64, 64), dtype=bool)
            mask[:4] = False
            return {"history": rng.uniform(0, 8, (7, 64, 64)).astype("float32") * mask,
                    "target": rng.uniform(0, 25, (18, 64, 64)).astype("float32") * mask,
                    "mask": mask, "issue_ns": 1717200000000000000}

    with TemporaryDirectory(prefix="nowcastnet_field_smoke_") as folder:
        tiny = deepcopy(config)
        tiny.update(experiment_root=str(Path(folder) / "experiment"), cache_root=str(Path(folder) / "cache"))
        tiny["model"].update(grid_size=64, base_channels=4, noise_channels=4)
        tiny["field"].update(precision="fp32", threads=2, evolution_epochs=1, generator_epochs=1,
                             evolution_iterations=1, generator_iterations=1,
                             importance_sampling=False, batch_size=1, validation_batch_size=1, num_workers=0)
        with patch.object(trainer, "HSRDataset", Synthetic):
            trainer.train_stage(tiny, "evolution", "cpu")
            trainer.train_stage(tiny, "generator", "cpu")
            model, _ = trainer.load_forecaster(tiny, "cpu")
            sample = Synthetic(tiny, "validation")[0]
            history = torch.from_numpy(sample["history"])[None]
            issues = np.array([sample["issue_ns"]], dtype=np.int64)
            one = trainer.predict_product(model, history, issues, tiny)
            two = trainer.predict_product(model, history, issues, tiny)
            if not torch.equal(one, two) or not torch.isfinite(one).all() or (one < 0).any():
                raise AssertionError("Forecast reproducibility/finiteness failed")
    report = {"diagnostic_only": True, "device": "cpu", "two_stage_training": "passed",
              "repeatable_four_draw_product": True, "shape": list(one.shape),
              "production_training_launched": False, "temporary_checkpoints_removed": True}
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "smoke", "train", "train-evolution", "train-generator", "predict", "evaluate"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-issues", type=int, help="Explicit diagnostic subset; predict defaults to 3; evaluate defaults to full split")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.max_issues is not None and (args.max_issues < 1 or args.action not in ("predict", "evaluate")):
        parser.error("--max-issues must be positive and is only allowed for predict/evaluate")
    config = load_field_config(args.config)
    if args.action == "check":
        return check(config)
    if args.action == "smoke":
        return smoke(config)
    deferred_guard = args.action.startswith("train") and args.resume
    if not deferred_guard:
        ensure_device(args.device, args.allow_shared_gpu)
    started = time.monotonic()
    with lock(Path(config["experiment_root"]) / "launcher"):
        if args.action.startswith("train"):
            from .trainer import train_stage
            stages = ("evolution", "generator") if args.action == "train" else (args.action.removeprefix("train-"),)
            prepared = False
            def before_training():
                nonlocal prepared
                if prepared:
                    return
                if deferred_guard:
                    ensure_device(args.device, args.allow_shared_gpu)
                prepared = True

            if not deferred_guard:
                before_training()
            for stage in stages:
                exists = (Path(config["experiment_root"]) / "field" / stage / "contract.json").exists()
                options = {"before_training": before_training} if deferred_guard else {}
                train_stage(config, stage, args.device, resume=args.resume and exists, **options)
        else:
            from .field_evaluation import evaluate, predict
            if args.action == "evaluate":
                evaluate(config, split=args.split, device=args.device, max_issues=args.max_issues)
            else:
                predict(config, split=args.split, device=args.device,
                        max_issues=args.max_issues if args.max_issues is not None else 3)
    print(f"Finished field-only {args.action} in {time.monotonic()-started:.1f}s. "
          "No CNN was run. Process exit releases GPU memory.", flush=True)


def cli_main():
    """Console-script wrapper: diagnostic dictionaries are not exit statuses."""
    main()
    return 0


if __name__ == "__main__":
    main()
