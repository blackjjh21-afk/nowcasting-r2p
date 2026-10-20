"""Stream the selected NowcastNet four-draw mean to a reusable NPY bank.

Each issue is inferred separately, with the study's issuance-keyed draws and
CPU float32 arithmetic mean after nonnegative member clipping. Future radar
and future gauge observations are never loaded. The output feeds cnn_adapter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import digest, identity, lock, write_json
from .data import HSRDataset, STEP_NS
from .field_config import load_field_config
from .trainer import build_model, load_forecaster, predict_members


def load_selected(config, checkpoint, checksum, device):
    """Load an explicitly hash-pinned, trusted local generator checkpoint.

    This permits archived study checkpoints whose author-side path/code audit
    differs from the portable package. The model geometry and selected weights
    are checked; no source contract or training artifact is rewritten.
    PyTorch checkpoints must come from a trusted source (they contain RNG state).
    """
    checkpoint = Path(checkpoint)
    if not isinstance(checksum, str) or len(checksum) != 64 or digest(checkpoint) != checksum.lower():
        raise ValueError("Selected checkpoint SHA256 differs")
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (value.get("stage") != "generator" or value.get("diagnostic_only") is not False
            or type(value.get("epoch")) is not int or value["epoch"] < 1):
        raise ValueError("Require a non-diagnostic generator checkpoint")
    contract_path = checkpoint.parent / "contract.json"
    if not contract_path.is_file():
        raise ValueError("Archived checkpoint requires its adjacent training contract.json")
    contract = json.loads(contract_path.read_text())
    if value.get("contract_identity") != identity(contract):
        raise ValueError("Checkpoint and original training contract disagree")
    original = contract["config"]
    if (original["model"] != config["model"]
            or original["forecast_product"] != config["forecast_product"]
            or original["field"]["precision"] != config["field"]["precision"]):
        raise ValueError("Model geometry, product or precision differs from selected checkpoint")
    model = build_model(config).to(device)
    model.load_state_dict(value["model"], strict=True)
    model.eval()
    return model, {"epoch": value["epoch"], "checkpoint_sha256": checksum.lower(),
                   "source_contract_sha256": digest(contract_path)}


def export(config, output, *, split="test", device="cpu", issue_times=None,
           checkpoint=None, checkpoint_sha256=None, resume=False):
    if split not in ("validation", "test"):
        raise ValueError("CNN banks use 2023 validation or 2024–2025 TEST, not upstream training years")
    if (checkpoint is None) != (checkpoint_sha256 is None):
        raise ValueError("Provide both explicit checkpoint and its SHA256")
    output = Path(output)
    existed = output.exists()
    if existed and not resume:
        raise FileExistsError(output)
    dataset = HSRDataset(config, split, require_future=False)
    available = np.asarray(dataset.issue_times_ns)
    issues = available if issue_times is None else np.asarray(issue_times)
    if (issues.dtype != np.int64 or issues.ndim != 1 or not len(issues)
            or (np.diff(issues) <= 0).any() or (issues % STEP_NS).any()):
        raise ValueError("Require strictly increasing 10-min KST-naive int64 issues")
    rows = np.searchsorted(available, issues)
    if (rows >= len(available)).any() or not np.array_equal(available[rows], issues):
        raise ValueError("Requested issues are not all HSR-history eligible")
    mask = np.asarray(dataset.spatial_mask)
    if mask.dtype != bool or mask.shape != (256, 256) or not mask.any():
        raise ValueError("Expected a nonempty 256x256 HSR support mask")
    if checkpoint is None:
        model, metadata = load_forecaster(config, device)
    else:
        model, metadata = load_selected(config, checkpoint, checkpoint_sha256, device)
    contract = {"schema": "nowcastnet_public_mean_bank_v1", "split": split,
        "issue_count": len(issues), "shape": [len(issues), 18, 256, 256], "dtype": "float32",
        "units": "mm h-1", "axes": ["issue", "lead", "y", "x"],
        "checkpoint_sha256": metadata["checkpoint_sha256"], "selected_epoch": metadata["epoch"],
        "forecast_product": config["forecast_product"], "precision": config["field"]["precision"],
        "aggregation": "CPU numpy mean of four nonnegative float32 members",
        "prediction_uses_future_observations": False,
        "input_data_provenance_sha256": identity(dataset.provenance),
        "timestamp_encoding": "KST-naive nanoseconds (not UTC Unix timestamps)",
        "issue_axis_sha256": identity(issues.tolist()), "mask_sha256": identity(mask.tolist())}
    with lock(output):
        contract_path, status_path = output / "contract.json", output / "status.json"
        rate_path = output / "mean_rate_mmph.npy"
        if not existed and any(path.exists() for path in (contract_path, status_path, rate_path)):
            raise FileExistsError("Another exporter created this bank while inputs were being prepared")
        if existed:
            if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
                raise ValueError("Resume source checkpoint, issue axis, mask or product changed")
            status = json.loads(status_path.read_text())
            start = int(status["committed_issues"])
            rates = np.load(rate_path, mmap_mode="r+", allow_pickle=False)
            if rates.shape != tuple(contract["shape"]) or rates.dtype != np.float32:
                raise ValueError("Resume bank shape/dtype differs")
            if not np.array_equal(np.load(output / "issue_times_kst_ns.npy", allow_pickle=False), issues):
                raise ValueError("Resume saved issue axis differs")
            if not np.array_equal(np.load(output / "spatial_mask.npy", allow_pickle=False), mask):
                raise ValueError("Resume saved spatial mask differs")
            if status.get("state") == "completed":
                if (start != len(issues) or status.get("contract_sha256") != digest(contract_path)
                        or status.get("rate_sha256") != digest(rate_path)):
                    raise ValueError("Completed bank checksum/cursor differs")
                return status
        else:
            write_json(contract_path, contract)
            np.save(output / "issue_times_kst_ns.npy", issues, allow_pickle=False)
            np.save(output / "spatial_mask.npy", mask, allow_pickle=False)
            rates = np.lib.format.open_memmap(rate_path, mode="w+", dtype=np.float32,
                                             shape=tuple(contract["shape"]))
            start = 0
        if not 0 <= start <= len(issues):
            raise ValueError("Invalid resume cursor")
        committed = start
        write_json(status_path, {"state": "running", "committed_issues": committed})
        try:
            for index in range(start, len(issues)):
                sample = dataset[int(rows[index])]
                if int(sample["issue_ns"]) != int(issues[index]) or not np.array_equal(sample["mask"], mask):
                    raise ValueError("Dataset sample axis or spatial mask changed")
                history = torch.as_tensor(sample["history"][None], device=device)
                with torch.inference_mode():
                    members = predict_members(model, history, [int(issues[index])], config)
                members = members.detach().cpu().numpy().astype(np.float32, copy=False)
                if members.shape != (4, 1, 18, 256, 256) or not np.isfinite(members).all() or (members < 0).any():
                    raise ValueError("Unexpected four-member prediction product")
                mean = members[:, 0].mean(axis=0, dtype=np.float32)
                mean[:, ~mask] = 0
                rates[index] = mean
                if (index + 1) % 100 == 0 or index + 1 == len(issues):
                    rates.flush()
                    committed = index + 1
                    write_json(status_path, {"state": "running", "committed_issues": committed})
                    print(f"NowcastNet bank {split}: {committed}/{len(issues)}", flush=True)
            result = {"state": "completed", "completed": True, "committed_issues": len(issues),
                "contract_sha256": digest(contract_path), "rate_file": rate_path.name,
                "rate_sha256": digest(rate_path),
                "issue_file": "issue_times_kst_ns.npy", "issue_axis_sha256": digest(output / "issue_times_kst_ns.npy")}
            write_json(status_path, result)
            return result
        except BaseException:
            write_json(status_path, {"state": "failed", "committed_issues": committed})
            raise
        finally:
            del rates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--issues", type=Path, help="Optional canonical issuance-axis NPY")
    parser.add_argument("--checkpoint", type=Path, help="Trusted selected generator checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    issues = None if args.issues is None else np.load(args.issues, allow_pickle=False)
    result = export(load_field_config(args.config), args.output, split=args.split,
                    device=args.device, issue_times=issues, checkpoint=args.checkpoint,
                    checkpoint_sha256=args.checkpoint_sha256, resume=args.resume)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
