"""Select adapted exPreCast checkpoints using 2023 native-grid RN60 CSI.

Evaluate all 23 completed epochs on source-backed 4-km cells. The score is
an equal-weight mean over 13 leads (+60 to +180 min) and three thresholds
(1, 5 and 10 mm). Exact ties select the earliest epoch. Evaluation years
are fixed to 2023; test-period data are not used for selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SCHEMA = "exprecast-native-rn60-checkpoint-selection-v1"
EPOCH_SCHEMA = "exprecast-native-rn60-epoch-evaluation-v1"
PREFLIGHT_SCHEMA = "exprecast-native-rn60-preflight-v1"
YEARS = (2023,)
INPUT_FRAMES = 7
OUTPUT_FRAMES = 18
FRAME_MINUTES = 10
LEADS_MIN = tuple(range(60, 181, 10))
THRESHOLDS_MM = (1.0, 5.0, 10.0)
WINDOW_FRAMES = 6
FRAME_HOURS = 10.0 / 60.0
EPOCH_PATTERN = re.compile(r"^epoch_(\d{4})_step_(\d{7})\.pt$")


class SelectionError(RuntimeError):
    """Raised when an input or saved artifact violates the selection contract."""


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise SelectionError(f"non-finite value cannot enter an audit artifact: {value}")
    return value


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_csv_atomic(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise SelectionError(f"expected a JSON object: {path}")
    return payload


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.project_root = args.project_root.expanduser().resolve()
    args.base_code_dir = Path(__file__).resolve().parent
    args.runner = args.base_code_dir / "workflow.py"
    args.official_repo = args.official_repo.expanduser().resolve()
    args.radar_root = args.radar_root.expanduser().resolve()
    if args.evaluation_mask is None:
        args.evaluation_mask = args.radar_root / "grid_coordinates.npz"
    args.evaluation_mask = args.evaluation_mask.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.run_dir / "epoch_checkpoints"
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.run_dir / "checkpoint_evaluations_2023_rn60_13x3"
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def load_runner(args: argparse.Namespace) -> ModuleType:
    from . import workflow

    return workflow

def runner_namespace(args: argparse.Namespace, checkpoint: Path | None = None) -> argparse.Namespace:
    """Build the field workflow arguments for checkpoint evaluation."""

    return argparse.Namespace(
        project_root=args.project_root,
        radar_root=args.radar_root,
        radar_format="kma-tiff",
        supplemental_radar_root=[],
        official_repo=args.official_repo,
        input_frames=INPUT_FRAMES,
        output_frames=OUTPUT_FRAMES,
        frame_minutes=FRAME_MINUTES,
        profile="paper_long_concat",
        grid_height=256,
        grid_width=256,
        crop_y0=None,
        crop_x0=None,
        spatial_factor=1,
        spatial_reduction="uniform",
        embed_dim=96,
        depths="2,6,2,2",
        num_heads="3,6,12,24",
        window_size="2,7,7",
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        use_checkpoint=True,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        evaluation_mask=args.evaluation_mask,
        evaluation_mask_key=args.evaluation_mask_key,
        initialization_checkpoint=None,
        encoder_only_initialization=False,
        freeze_encoder=False,
        checkpoint=checkpoint,
        trust_checkpoint=args.trust_checkpoint,
        evaluation_years=YEARS,
        evaluation_anchor_stride=args.anchor_stride,
        evaluation_max_windows=None,
        evaluation_max_batches=None,
        validation_batch_size=args.batch_size,
    )


def checkpoint_inventory(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {args.checkpoint_dir}")
    rows: list[dict[str, Any]] = []
    for path in sorted(args.checkpoint_dir.glob("epoch_*.pt")):
        match = EPOCH_PATTERN.match(path.name)
        if match is None:
            raise SelectionError(f"unexpected epoch-checkpoint name: {path.name}")
        rows.append(
            {
                "epoch": int(match.group(1)),
                "global_step": int(match.group(2)),
                "checkpoint": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    expected_epochs = list(range(1, args.expected_checkpoints + 1))
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != expected_epochs:
        raise SelectionError(
            "epoch candidate set changed: "
            f"found {epochs}, expected exactly {expected_epochs}"
        )
    return rows


def scientific_contract(
    args: argparse.Namespace,
    dataset: Any,
    mask_metadata: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = dataset.metadata()
    if list(metadata.get("years", [])) != [2023]:
        raise SelectionError(f"test-period blindness failed: dataset years={metadata.get('years')}")
    if int(metadata.get("input_frames", -1)) != INPUT_FRAMES:
        raise SelectionError("input-frame contract changed")
    if int(metadata.get("output_frames", -1)) != OUTPUT_FRAMES:
        raise SelectionError("output-frame contract changed")
    if int(metadata.get("frame_minutes", -1)) != FRAME_MINUTES:
        raise SelectionError("frame cadence changed")
    if len(dataset) != args.expected_windows:
        raise SelectionError(
            f"2023 selection support changed: {len(dataset)} != {args.expected_windows}"
        )
    valid_fraction = float(mask_metadata.get("valid_fraction", float("nan")))
    valid_cells = int(round(valid_fraction * 256 * 256))
    if valid_cells != args.expected_valid_cells:
        raise SelectionError(
            f"source-valid mask changed: {valid_cells} != {args.expected_valid_cells} cells"
        )
    return {
        "schema": SCHEMA,
        "selection_period": [2023],
        "test_period_accessed": False,
        "truth": "observed official-format KMA HSR decoded to rolling RN60",
        "prediction": "long exPreCast native-grid forecast decoded to rolling RN60",
        "grid": "native 4-km 256x256 crop",
        "support": {
            "mode": "source-validity mask; pool 1",
            "mask": jsonable(mask_metadata),
            "valid_cells": valid_cells,
            "window_count": len(dataset),
            "anchor_stride": args.anchor_stride,
        },
        "temporal_contract": {
            "input_frames": INPUT_FRAMES,
            "output_frames": OUTPUT_FRAMES,
            "frame_minutes": FRAME_MINUTES,
            "rn60_frames": WINDOW_FRAMES,
            "lead_minutes": list(LEADS_MIN),
        },
        "thresholds_mm": list(THRESHOLDS_MM),
        "score": (
            "raw equal-weight arithmetic mean of pooled CSI over "
            "13 leads x 3 thresholds"
        ),
        "tie_break": "earliest epoch only for an exactly equal raw macro CSI",
        "candidate_epochs": [int(row["epoch"]) for row in checkpoints],
        "runner": {
            "path": str(args.runner),
            "sha256": sha256_file(args.runner),
            "portable_data_sha256": sha256_file(args.base_code_dir / "data.py"),
            "portable_model_sha256": sha256_file(args.base_code_dir / "model.py"),
            "selector_path": str(Path(__file__).resolve()),
            "selector_sha256": sha256_file(Path(__file__).resolve()),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "data": jsonable(metadata),
    }


def preflight(args: argparse.Namespace) -> tuple[ModuleType, Any, torch.Tensor, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    for path, label in (
        (args.radar_root, "2023 radar root"),
        (args.official_repo / "model.py", "pinned official exPreCast model"),
        (args.evaluation_mask, "source-validity mask"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    checkpoints = checkpoint_inventory(args)
    runner = load_runner(args)
    namespace = runner_namespace(args)
    dataset = runner.make_dataset(
        namespace,
        YEARS,
        require_targets=True,
        anchor_stride=args.anchor_stride,
        max_windows=None,
    )
    mask, mask_metadata = runner.load_evaluation_mask(
        args.evaluation_mask, dataset, args.evaluation_mask_key
    )
    if mask is None:
        raise SelectionError("a source-validity mask is mandatory")
    contract = scientific_contract(args, dataset, mask_metadata, checkpoints)
    contract_hash = sha256_json(contract)
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "scientific_contract": contract,
        "scientific_contract_sha256": contract_hash,
        "checkpoints": checkpoints,
        "output_dir": str(args.output_dir),
        "device_requested": args.device,
        "cuda_available": torch.cuda.is_available(),
    }
    return runner, dataset, mask, mask_metadata, checkpoints, payload


def accumulate_counts(
    counts: torch.Tensor,
    valid_counts: torch.Tensor,
    prediction_rate: torch.Tensor,
    target_rate: torch.Tensor,
    spatial_mask: torch.Tensor,
) -> None:
    """Accumulate [lead, threshold, hit/miss/false] counts on the device."""

    if prediction_rate.shape != target_rate.shape:
        raise SelectionError("prediction and target tensors differ in shape")
    if prediction_rate.ndim != 4 or prediction_rate.shape[1] != OUTPUT_FRAMES:
        raise SelectionError(
            f"expected [batch,{OUTPUT_FRAMES},height,width] rates, got "
            f"{tuple(prediction_rate.shape)}"
        )
    if tuple(spatial_mask.shape) != tuple(prediction_rate.shape[-2:]):
        raise SelectionError("source-validity mask and forecast grid differ")
    for lead_index, end_frame in enumerate(range(WINDOW_FRAMES - 1, OUTPUT_FRAMES)):
        start_frame = end_frame - WINDOW_FRAMES + 1
        prediction_rn60 = (
            prediction_rate[:, start_frame : end_frame + 1].sum(dim=1) * FRAME_HOURS
        )
        target_rn60 = (
            target_rate[:, start_frame : end_frame + 1].sum(dim=1) * FRAME_HOURS
        )
        valid = (
            spatial_mask.unsqueeze(0)
            & torch.isfinite(prediction_rn60)
            & torch.isfinite(target_rn60)
        )
        valid_counts[lead_index] += valid.sum(dtype=torch.int64)
        for threshold_index, threshold in enumerate(THRESHOLDS_MM):
            observed = target_rn60 >= threshold
            forecast = prediction_rn60 >= threshold
            counts[lead_index, threshold_index, 0] += (
                valid & observed & forecast
            ).sum(dtype=torch.int64)
            counts[lead_index, threshold_index, 1] += (
                valid & observed & ~forecast
            ).sum(dtype=torch.int64)
            counts[lead_index, threshold_index, 2] += (
                valid & ~observed & forecast
            ).sum(dtype=torch.int64)


def summarize_counts(counts: np.ndarray) -> dict[str, Any]:
    if counts.shape != (len(LEADS_MIN), len(THRESHOLDS_MM), 3):
        raise SelectionError(f"unexpected count shape: {counts.shape}")
    denominator = counts.sum(axis=2, dtype=np.int64)
    if np.any(denominator <= 0):
        where = np.argwhere(denominator <= 0).tolist()
        raise SelectionError(f"CSI denominator is zero in cells {where}")
    csi = counts[:, :, 0].astype(np.float64) / denominator
    if not np.isfinite(csi).all():
        raise SelectionError("non-finite CSI produced")
    return {
        "counts_hit_miss_false_alarm": counts.tolist(),
        "csi_by_lead_threshold": csi.tolist(),
        "macro_csi_13lead_3threshold": float(csi.mean()),
        "mean_csi_by_threshold": csi.mean(axis=0).tolist(),
        "mean_csi_by_lead": csi.mean(axis=1).tolist(),
    }


@torch.inference_mode()
def evaluate_epoch(
    args: argparse.Namespace,
    runner: ModuleType,
    dataset: Any,
    spatial_mask: torch.Tensor,
    checkpoint: Mapping[str, Any],
    scientific_contract_sha256: str,
) -> dict[str, Any]:
    checkpoint_path = Path(str(checkpoint["checkpoint"]))
    namespace = runner_namespace(args, checkpoint_path)
    device = runner.resolve_device(args.device)
    model, model_metadata, _ = runner.load_trained_model(namespace, device)
    loader = runner.make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        samples_per_epoch=None,
        locality_span=1,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    mask = spatial_mask.squeeze().to(device=device, dtype=torch.bool)
    counts = torch.zeros(
        (len(LEADS_MIN), len(THRESHOLDS_MM), 3),
        dtype=torch.int64,
        device=device,
    )
    valid_counts = torch.zeros(len(LEADS_MIN), dtype=torch.int64, device=device)
    evaluated_windows = 0
    started = time.time()
    for batch_index, batch in enumerate(loader):
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with runner.autocast_context(device, args.precision):
            predictions = model(inputs)
        prediction_norm = predictions[:, 0].float()
        target_norm = targets[:, 0].float()
        prediction_rate = runner.dbz_to_rain_rate(
            (prediction_norm * 100.0).clamp_min(0.0)
        )
        target_rate = runner.dbz_to_rain_rate((target_norm * 100.0).clamp_min(0.0))
        # Preserve the physical no-echo convention used by the paper's native
        # RN60 evaluator.  It is immaterial above 1 mm but makes the contract
        # explicit and avoids adding six formula-derived trace amounts.
        prediction_rate = torch.where(
            prediction_norm > 0.0, prediction_rate, torch.zeros_like(prediction_rate)
        )
        target_rate = torch.where(
            target_norm > 0.0, target_rate, torch.zeros_like(target_rate)
        )
        accumulate_counts(counts, valid_counts, prediction_rate, target_rate, mask)
        evaluated_windows += prediction_rate.shape[0]
        if (
            (batch_index + 1) % args.progress_every == 0
            or batch_index + 1 == len(loader)
        ):
            print(
                f"epoch {int(checkpoint['epoch']):02d}: "
                f"batch {batch_index + 1}/{len(loader)}",
                flush=True,
            )
    if evaluated_windows != len(dataset):
        raise SelectionError(
            f"incomplete epoch evaluation: {evaluated_windows} != {len(dataset)}"
        )
    summary = summarize_counts(counts.cpu().numpy())
    payload = {
        "schema": EPOCH_SCHEMA,
        "status": "complete",
        "selection_period": [2023],
        "test_period_accessed": False,
        "scientific_contract_sha256": scientific_contract_sha256,
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "model": jsonable(model_metadata),
        "lead_minutes": list(LEADS_MIN),
        "thresholds_mm": list(THRESHOLDS_MM),
        "evaluated_windows": evaluated_windows,
        "evaluated_values_by_lead": valid_counts.cpu().tolist(),
        "elapsed_seconds": time.time() - started,
        **summary,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def epoch_output_path(output_dir: Path, epoch: int) -> Path:
    return output_dir / "epochs" / f"epoch_{epoch:04d}_rn60.json"


def load_or_evaluate_epoch(
    args: argparse.Namespace,
    runner: ModuleType,
    dataset: Any,
    mask: torch.Tensor,
    checkpoint: Mapping[str, Any],
    contract_hash: str,
) -> dict[str, Any]:
    path = epoch_output_path(args.output_dir, int(checkpoint["epoch"]))
    if path.exists() and not args.overwrite:
        payload = load_json(path)
        expected = {
            "schema": EPOCH_SCHEMA,
            "status": "complete",
            "scientific_contract_sha256": contract_hash,
            "checkpoint_sha256": str(checkpoint["sha256"]),
            "epoch": int(checkpoint["epoch"]),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise SelectionError(
                    f"saved epoch artifact differs at {key}: {path}; "
                    "use a new output directory or --overwrite deliberately"
                )
        print(f"reusing {path.name}", flush=True)
        return payload
    payload = evaluate_epoch(
        args, runner, dataset, mask, checkpoint, contract_hash
    )
    write_json_atomic(path, payload)
    print(f"wrote {path}", flush=True)
    return payload


def candidate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    threshold_means = [float(value) for value in payload["mean_csi_by_threshold"]]
    lead_means = [float(value) for value in payload["mean_csi_by_lead"]]
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "checkpoint": str(payload["checkpoint"]),
        "checkpoint_sha256": str(payload["checkpoint_sha256"]),
        "macro_csi_13lead_3threshold": float(
            payload["macro_csi_13lead_3threshold"]
        ),
        **{
            f"mean_csi_{int(threshold)}mm": threshold_means[index]
            for index, threshold in enumerate(THRESHOLDS_MM)
        },
        **{
            f"mean_csi_lead_{lead}min": lead_means[index]
            for index, lead in enumerate(LEADS_MIN)
        },
    }


def select_candidates(
    args: argparse.Namespace,
    payloads: Sequence[Mapping[str, Any]],
    contract_hash: str,
) -> dict[str, Any]:
    if len(payloads) != args.expected_checkpoints:
        raise SelectionError(
            f"candidate result count changed: {len(payloads)} != {args.expected_checkpoints}"
        )
    candidates: list[dict[str, Any]] = []
    for payload in payloads:
        if payload.get("scientific_contract_sha256") != contract_hash:
            raise SelectionError("candidate scientific contracts differ")
        row = candidate_summary(payload)
        row["epoch_evaluation_json"] = str(
            epoch_output_path(args.output_dir, int(row["epoch"]))
        )
        candidates.append(row)
    candidates.sort(key=lambda row: int(row["epoch"]))
    epochs = [int(row["epoch"]) for row in candidates]
    expected = list(range(1, args.expected_checkpoints + 1))
    if epochs != expected:
        raise SelectionError(f"candidate epochs changed: {epochs} != {expected}")
    best_score = max(float(row["macro_csi_13lead_3threshold"]) for row in candidates)
    exact_ties = [
        row
        for row in candidates
        if float(row["macro_csi_13lead_3threshold"]) == best_score
    ]
    selected = min(exact_ties, key=lambda row: int(row["epoch"]))
    return {
        "schema": SCHEMA,
        "status": "selected",
        "selection_period": [2023],
        "test_period_accessed": False,
        "scientific_contract_artifact": "preflight.json",
        "scientific_contract_sha256": contract_hash,
        "selection_rule": (
            "maximum raw equal-weight native-grid RN60 CSI mean over "
            "60:10:180 min x 1/5/10 mm; earliest epoch on exact tie"
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "exact_best_tie_epochs": [int(row["epoch"]) for row in exact_ties],
        "selected": selected,
    }


def csv_fields() -> list[str]:
    return [
        "epoch",
        "global_step",
        "checkpoint",
        "checkpoint_sha256",
        "macro_csi_13lead_3threshold",
        *[f"mean_csi_{int(value)}mm" for value in THRESHOLDS_MM],
        *[f"mean_csi_lead_{lead}min" for lead in LEADS_MIN],
        "epoch_evaluation_json",
        "is_selected",
    ]


def finalize_outputs(
    args: argparse.Namespace,
    selection: Mapping[str, Any],
    preflight_payload: Mapping[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selection.json"
    scores_path = args.output_dir / "epoch_scores.csv"
    preflight_path = args.output_dir / "preflight.json"
    write_json_atomic(preflight_path, preflight_payload)
    write_json_atomic(selection_path, selection)
    selected_epoch = int(selection["selected"]["epoch"])
    rows = []
    for candidate in selection["candidates"]:
        row = dict(candidate)
        row["is_selected"] = int(candidate["epoch"]) == selected_epoch
        rows.append(row)
    write_csv_atomic(scores_path, rows, csv_fields())
    artifact_paths = [preflight_path, selection_path, scores_path]
    artifact_paths.extend(
        epoch_output_path(args.output_dir, int(row["epoch"])) for row in rows
    )
    hashes = {
        str(path.relative_to(args.output_dir)): sha256_file(path)
        for path in artifact_paths
    }
    manifest = {
        "schema": "exprecast-native-rn60-artifact-manifest-v1",
        "status": "complete",
        "scientific_contract_sha256": selection["scientific_contract_sha256"],
        "selected_epoch": selected_epoch,
        "selected_checkpoint": selection["selected"]["checkpoint"],
        "selected_checkpoint_sha256": selection["selected"]["checkpoint_sha256"],
        "artifacts_sha256": hashes,
    }
    manifest_path = args.output_dir / "artifact_manifest.json"
    write_json_atomic(manifest_path, manifest)
    completed = {
        "schema": "exprecast-native-rn60-selection-completed-v1",
        "status": "complete",
        "test_period_accessed": False,
        "selected_epoch": selected_epoch,
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "selection_sha256": sha256_file(selection_path),
    }
    write_json_atomic(args.output_dir / "COMPLETED.json", completed)


def command_preflight(args: argparse.Namespace) -> None:
    _, _, _, _, _, payload = preflight(args)
    if args.dry_run:
        print(json.dumps(jsonable(payload), indent=2, sort_keys=True))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "preflight.json", payload)
    print(f"preflight passed; wrote {args.output_dir / 'preflight.json'}")


def command_run(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.trust_checkpoint:
        raise SelectionError("checkpoint evaluation requires --trust-checkpoint for trusted local files")
    runner, dataset, mask, _, checkpoints, preflight_payload = preflight(args)
    if args.dry_run:
        print(json.dumps(jsonable(preflight_payload), indent=2, sort_keys=True))
        print("dry run only; no checkpoint was loaded and no output was written")
        return
    if args.output_dir.exists() and args.overwrite:
        # Never recursively delete an output directory.  Individual files are
        # overwritten atomically as they are regenerated.
        print(f"deliberate overwrite enabled for {args.output_dir}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_hash = str(preflight_payload["scientific_contract_sha256"])
    payloads = [
        load_or_evaluate_epoch(
            args, runner, dataset, mask, checkpoint, contract_hash
        )
        for checkpoint in checkpoints
    ]
    selection = select_candidates(args, payloads, contract_hash)
    finalize_outputs(args, selection, preflight_payload)
    chosen = selection["selected"]
    print(
        f"selected epoch {int(chosen['epoch'])}: "
        f"macro CSI={float(chosen['macro_csi_13lead_3threshold']):.8f}",
        flush=True,
    )
    print(chosen["checkpoint"], flush=True)


def command_select(args: argparse.Namespace) -> None:
    runner, dataset, mask, _, checkpoints, preflight_payload = preflight(args)
    del runner, dataset, mask
    contract_hash = str(preflight_payload["scientific_contract_sha256"])
    payloads = []
    for checkpoint in checkpoints:
        path = epoch_output_path(args.output_dir, int(checkpoint["epoch"]))
        if not path.is_file():
            raise FileNotFoundError(f"epoch evaluation is missing: {path}")
        payload = load_json(path)
        expected = {
            "schema": EPOCH_SCHEMA,
            "status": "complete",
            "scientific_contract_sha256": contract_hash,
            "checkpoint_sha256": str(checkpoint["sha256"]),
            "epoch": int(checkpoint["epoch"]),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise SelectionError(f"saved epoch artifact differs at {key}: {path}")
        payloads.append(payload)
    selection = select_candidates(args, payloads, contract_hash)
    finalize_outputs(args, selection, preflight_payload)
    print(f"selected epoch {selection['selected']['epoch']}")


def command_self_test(_: argparse.Namespace) -> None:
    prediction = torch.zeros((1, OUTPUT_FRAMES, 1, 1), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    prediction[:, :, 0, 0] = 25.0
    target[:, :, 0, 0] = 25.0
    counts = torch.zeros(
        (len(LEADS_MIN), len(THRESHOLDS_MM), 3), dtype=torch.int64
    )
    valid_counts = torch.zeros(len(LEADS_MIN), dtype=torch.int64)
    accumulate_counts(
        counts,
        valid_counts,
        prediction,
        target,
        torch.ones((1, 1), dtype=torch.bool),
    )
    summary = summarize_counts(counts.numpy())
    if valid_counts.tolist() != [1] * len(LEADS_MIN):
        raise SelectionError("valid-value count self-test failed")
    if summary["macro_csi_13lead_3threshold"] != 1.0:
        raise SelectionError("perfect-forecast metric self-test failed")
    candidates = [
        {"epoch": 2, "macro_csi_13lead_3threshold": 0.5},
        {"epoch": 1, "macro_csi_13lead_3threshold": 0.5},
    ]
    score = max(float(row["macro_csi_13lead_3threshold"]) for row in candidates)
    tied = [row for row in candidates if row["macro_csi_13lead_3threshold"] == score]
    if min(tied, key=lambda row: int(row["epoch"]))["epoch"] != 1:
        raise SelectionError("exact-tie self-test failed")
    print("self-test passed")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument(
        "--trust-checkpoint", action="store_true",
        help="allow loading trusted training checkpoints that contain Python state",
    )
    parser.add_argument("--evaluation-mask", type=Path)
    parser.add_argument(
        "--evaluation-mask-key", default="source_index_valid_mask"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-checkpoints", type=int, default=23)
    parser.add_argument("--expected-windows", type=int, default=1441)
    parser.add_argument("--expected-valid-cells", type=int, default=59136)
    parser.add_argument("--anchor-stride", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=6455)
    parser.add_argument("--progress-every", type=int, default=100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser(
        "preflight", help="audit all inputs without loading checkpoint weights"
    )
    add_common_arguments(preflight_parser)
    preflight_parser.add_argument("--dry-run", action="store_true")
    preflight_parser.set_defaults(handler=command_preflight)

    run_parser = subparsers.add_parser(
        "run", help="preflight, evaluate all epochs, select, and hash outputs"
    )
    add_common_arguments(run_parser)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the complete plan without loading a model or writing outputs",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically regenerate per-epoch outputs instead of resuming them",
    )
    run_parser.set_defaults(handler=command_run)

    select_parser = subparsers.add_parser(
        "select", help="select from already completed per-epoch JSON files"
    )
    add_common_arguments(select_parser)
    select_parser.set_defaults(handler=command_select, overwrite=False, dry_run=False)

    self_test_parser = subparsers.add_parser(
        "self-test", help="run deterministic metric and tie-break unit checks"
    )
    self_test_parser.set_defaults(handler=command_self_test)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "self-test":
        args = resolve_defaults(args)
        if args.anchor_stride <= 0 or args.batch_size <= 0 or args.progress_every <= 0:
            parser.error("stride, batch size, and progress interval must be positive")
        if args.expected_checkpoints <= 0:
            parser.error("--expected-checkpoints must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
