"""Convert saved NowcastNet mean rain-rate fields to the public CNN NPZ format.

No model is trained here. The input is the already generated four-draw mean in
mm/h, not an individual draw or mean of four CNN predictions. Context features
and targets are supplied separately on explicitly matching issuance/station
axes; OOF context must exclude the held station group. See the package README.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from cnn_readout.workflow import (AUX_DIM, PREPARED_SCHEMA, SELECTION_LEADS,
    atomic_npz, sha256_file)

CONTEXT_SCHEMA = "nowcastnet_cnn_context_v1"
WINDOWS = np.arange(13)[:, None] + np.arange(6)[None]


def rate_to_patches(rate, station_y, station_x):
    """Native 3x3 station patches, M-P equivalent dBZ/100, float16 storage.

    This is the numerical conversion used in the study: no upper clipping of
    equivalent reflectivity; zero rates map to zero. It does not apply M-P a
    second time to rainfall or change the existing saved mean field.
    """
    values = np.asarray(rate, np.float32)
    y, x = np.asarray(station_y), np.asarray(station_x)
    if values.shape != (18, 256, 256) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Mean rate must be finite nonnegative [18,256,256] mm/h")
    if (y.ndim != 1 or y.shape != x.shape or not len(y)
            or y.dtype.kind not in "iu" or x.dtype.kind not in "iu"
            or (y < 1).any() or (y > 254).any() or (x < 1).any() or (x > 254).any()):
        raise ValueError("Integer station centres must support complete 3x3 grid patches")
    dy, dx = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
    patches = values[:, y[:, None, None] + dy, x[:, None, None] + dx].transpose(1, 0, 2, 3)
    result = np.zeros_like(patches)
    wet = patches > 0
    result[wet] = np.maximum(10. * np.log10(200.) + 16. * np.log10(patches[wet]), 0.) / 100.
    return np.ascontiguousarray(result, dtype=np.float16)


def _declared(archive, name):
    value = np.asarray(archive[name]) if name in archive else np.asarray(False)
    if value.shape != () or value.dtype.kind != "b" or not value.item():
        raise ValueError(f"Context must declare {name}=True")


def _axis(values, label):
    values = np.asarray(values)
    if (values.ndim != 1 or not len(values) or values.dtype != np.int64
            or (np.diff(values) <= 0).any() or (values % 600_000_000_000).any()):
        raise ValueError(f"{label} must be increasing int64 10-min KST-naive timestamps")
    return values


def _auxiliary(context, xy, leads, output):
    output[..., :2] = xy[None, :, None, :]
    output[..., 2] = leads[None, None, :].astype(np.float32) / 180.
    output[..., 3:] = context[:, :, None, :]


def prepare(mean_rate_path, issue_axis_path, context_path, output, *, purpose="predict"):
    """Build a bounded-memory NPZ from a memmapped [issue,18,256,256] bank.

    Input context has [issue,station,36] features and optional [3,issue,station,36]
    fold-clean features. The resulting dense NPZ is consumed directly by
    cnn_readout.workflow's validate/select-epoch/fit/predict commands.
    """
    mean_rate_path, issue_axis_path = Path(mean_rate_path), Path(issue_axis_path)
    context_path, output = Path(context_path), Path(output)
    if purpose not in ("select", "fit", "predict"):
        raise ValueError("Purpose must be select, fit or predict")
    if output.exists():
        raise FileExistsError(output)
    status_path = mean_rate_path.parent / "status.json"
    if status_path.is_file() and mean_rate_path.name == "mean_rate_mmph.npy":
        status = json.loads(status_path.read_text())
        if status.get("state") != "completed":
            raise ValueError("The field bank has not completed")
        if "rate_sha256" in status and status["rate_sha256"] != sha256_file(mean_rate_path):
            raise ValueError("Completed field bank checksum differs")
    issues = _axis(np.load(issue_axis_path, allow_pickle=False), "Field issue axis")
    rates = np.load(mean_rate_path, mmap_mode="r", allow_pickle=False)
    if rates.shape != (len(issues), 18, 256, 256) or rates.dtype != np.float32:
        raise ValueError("Expected float32 saved four-draw mean [issue,18,256,256]")
    if purpose != "predict":
        months = issues.astype("datetime64[ns]").astype("datetime64[M]").astype(str)
        if set(months) != {"2023-06", "2023-07", "2023-08", "2023-09"}:
            raise ValueError("CNN selection/fitting uses June–September 2023 only")
    with np.load(context_path, allow_pickle=False) as context:
        if np.asarray(context["schema"]).shape != () or str(context["schema"].item()) != CONTEXT_SCHEMA:
            raise ValueError("Unexpected context NPZ schema")
        if not np.array_equal(_axis(context["issue_times_ns"], "Context issue axis"), issues):
            raise ValueError("Field and context issue axes differ")
        _declared(context, "context_fitting_stations_only")
        _declared(context, "context_available_at_issuance")
        ids, folds = np.asarray(context["station_ids"]), np.asarray(context["station_folds"])
        xy = np.asarray(context["xy_norm"], np.float32)
        y, x = np.asarray(context["grid_y"]), np.asarray(context["grid_x"])
        if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu" or (ids < 0).any()
                or len(set(ids.tolist())) != len(ids) or folds.shape != ids.shape
                or folds.dtype.kind not in "iu" or not set(folds.tolist()) <= {0, 1, 2}
                or xy.shape != (len(ids), 2) or not np.isfinite(xy).all()
                or y.shape != ids.shape or x.shape != ids.shape):
            raise ValueError("Invalid station IDs, folds, normalized coordinates or grid centres")
        features = np.asarray(context["context36"], np.float32)
        if features.shape != (len(issues), len(ids), 36) or not np.isfinite(features).all():
            raise ValueError("context36 must be finite [issue,station,36]")
        fold_context = None
        if "oof_context36" in context:
            _declared(context, "oof_context_excludes_held_fold")
            fold_context = np.asarray(context["oof_context36"], np.float32)
            if fold_context.shape != (3, *features.shape) or not np.isfinite(fold_context).all():
                raise ValueError("oof_context36 must be finite [3,issue,station,36]")
        if purpose == "select" and fold_context is None:
            raise ValueError("OOF selection requires fold-clean oof_context36")
        shape = (len(issues), len(ids), len(SELECTION_LEADS))
        targets = None
        if "targets_mm" in context:
            targets = np.asarray(context["targets_mm"], np.float32)
            if targets.shape != shape or np.isinf(targets).any() or (targets < 0).any():
                raise ValueError("targets_mm must be nonnegative/NaN [issue,station,13]")
        if purpose != "predict" and targets is None:
            raise ValueError("CNN fitting/selection requires RN60 targets")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="nowcastnet-cnn-prepare-", dir=output.parent) as folder:
            root = Path(folder)
            patches = np.lib.format.open_memmap(root / "patches.npy", mode="w+", dtype=np.float32,
                                               shape=shape + (6, 3, 3))
            aux = np.lib.format.open_memmap(root / "aux.npy", mode="w+", dtype=np.float32,
                                           shape=shape + (AUX_DIM,))
            for index in range(len(issues)):
                patches[index] = rate_to_patches(rates[index], y, x)[:, WINDOWS].astype(np.float32)
            _auxiliary(features, xy, SELECTION_LEADS, aux)
            values = dict(schema=np.asarray(PREPARED_SCHEMA), patches=patches, auxiliary=aux,
                issue_times_ns=issues, station_ids=ids.astype(np.int64), station_folds=folds.astype(np.int8),
                lead_minutes=SELECTION_LEADS, source_product=np.asarray("NowcastNet four-draw rain-rate mean"),
                source_rate_sha256=np.asarray(sha256_file(mean_rate_path)),
                source_context_sha256=np.asarray(sha256_file(context_path)))
            if targets is not None:
                values["targets_mm"] = targets
            if fold_context is not None:
                oof = np.lib.format.open_memmap(root / "oof.npy", mode="w+", dtype=np.float32,
                                               shape=(3,) + shape + (AUX_DIM,))
                for fold in range(3):
                    _auxiliary(fold_context[fold], xy, SELECTION_LEADS, oof[fold])
                values["oof_auxiliary"] = oof
            atomic_npz(output, **values)
        # Public loader verification is done by the caller on demand. Loading
        # the entire large NPZ here would defeat the streaming preparation.
    return {"schema": "nowcastnet_cnn_prepare_receipt_v1", "purpose": purpose,
            "issue_count": len(issues), "station_count": len(ids), "lead_count": 13,
            "output_sha256": sha256_file(output), "patch_roundtrip": "float16 then float32",
            "rain_rate_mean_recomputed": False, "target_scale_refitted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=Path, required=True, help="float32 mean_rate_mmph.npy")
    parser.add_argument("--issues", type=Path, required=True, help="int64 issue_times_kst_ns.npy")
    parser.add_argument("--context", type=Path, required=True, help="nowcastnet_cnn_context_v1 NPZ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=("select", "fit", "predict"), default="predict")
    args = parser.parse_args(argv)
    result = prepare(args.rates, args.issues, args.context, args.output, purpose=args.purpose)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
