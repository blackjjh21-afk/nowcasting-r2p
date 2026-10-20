"""Public-only, CPU synthetic checks; no upstream source or weights are needed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tifffile = pytest.importorskip("tifffile")
h5py = pytest.importorskip("h5py")

from exprecast_adapted import model, selection, workflow
from exprecast_adapted.data import CropContract, KmaTiffWindowDataset
from exprecast_adapter.adapter import inspect_hdf5, validate_field_axes


def _common_arguments(tmp_path: Path) -> list[str]:
    return [
        "--radar-root", str(tmp_path / "radar"),
        "--official-repo", str(tmp_path / "upstream"),
    ]


def test_primary_configuration_and_long_decoder_contract() -> None:
    config = model.ExPreCast5MinConfig()
    config.validate()
    assert (config.input_frames, config.output_frames, config.frame_minutes) == (7, 18, 10)
    assert config.input_offsets_minutes == (-60, -50, -40, -30, -20, -10, 0)
    assert config.lead_minutes == tuple(range(10, 181, 10))
    assert config.depths == (2, 6, 2, 2)
    assert config.num_heads == (3, 6, 12, 24)
    assert config.embed_dim == 96
    assert config.window_size == (2, 7, 7)
    assert config.model_kwargs["patch_embed_size"] == (2, 4, 4)
    assert config.model_kwargs["patch_expan_size"] == (1, 4, 4)
    assert config.model_kwargs["upsampling_scale"] == (2, 2, 2)
    assert config.model_kwargs["skip_connection"] == "concat"
    assert config.model_kwargs["use_checkpoint"] is True
    assert model.expected_latent_time(config) == 60
    with pytest.raises(model.ModelContractError, match="four stages"):
        replace(config, depths=(2, 6, 2)).validate()
    with pytest.raises(model.ModelContractError, match="not divisible"):
        replace(config, embed_dim=95).validate()


def test_cli_primary_defaults_and_separate_train_evaluation_batches(tmp_path: Path) -> None:
    parser = workflow.build_parser()
    common = _common_arguments(tmp_path)
    train = parser.parse_args(["train", *common, "--run-name", "synthetic"])
    assert (train.input_frames, train.output_frames, train.frame_minutes) == (7, 18, 10)
    assert train.radar_format == "kma-tiff"
    assert train.train_years == (2019, 2020, 2021, 2022)
    assert train.validation_years == (2023,)
    assert train.batch_size == 4
    assert train.gradient_accumulation == 4
    assert train.validation_batch_size == 1
    assert train.seed == 6455
    assert train.trust_checkpoint is False
    evaluation = parser.parse_args([
        "evaluate", *common, "--checkpoint", "synthetic.pt",
        "--evaluation-output", "synthetic.json",
    ])
    assert evaluation.evaluation_years == (2023,)
    assert evaluation.batch_size == evaluation.validation_batch_size == 1
    export = parser.parse_args([
        "export", *common, "--checkpoint", "synthetic.pt",
        "--export-output", "synthetic.h5", "--export-years", "2023",
    ])
    assert export.batch_size == export.export_batch_size == 1
    assert workflow.model_config_from_args(train) == model.ExPreCast5MinConfig()


@pytest.mark.parametrize("missing", ["--radar-root", "--official-repo"])
def test_cli_requires_user_supplied_data_and_upstream(tmp_path: Path, missing: str) -> None:
    arguments = _common_arguments(tmp_path)
    index = arguments.index(missing)
    del arguments[index:index + 2]
    with pytest.raises(SystemExit):
        workflow.build_parser().parse_args(["train", *arguments, "--run-name", "synthetic"])


@pytest.mark.parametrize("option,value", [
    ("--input-frames", "13"), ("--frame-minutes", "5"), ("--radar-format", "zarr"),
])
def test_cli_rejects_nonprimary_temporal_or_storage_routes(
    tmp_path: Path, option: str, value: str,
) -> None:
    with pytest.raises(SystemExit):
        workflow.build_parser().parse_args([
            "train", *_common_arguments(tmp_path), "--run-name", "synthetic", option, value,
        ])


@pytest.mark.parametrize("option,value,message", [
    ("--train-years", "2023", "fitting years"),
    ("--validation-years", "2024", "selection years"),
])
def test_training_cli_rejects_changed_fitting_or_selection_periods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str, value: str, message: str,
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "exprecast", "train", *_common_arguments(tmp_path),
        "--run-name", "synthetic", option, value,
    ])
    with pytest.raises(ValueError, match=message):
        workflow.main()


def test_selector_defaults_keep_2023_source_mask_and_all_23_epochs(tmp_path: Path) -> None:
    args = selection.build_parser().parse_args([
        "run", *_common_arguments(tmp_path), "--run-dir", str(tmp_path / "training"),
    ])
    assert args.expected_checkpoints == 23
    assert args.batch_size == 1
    assert args.anchor_stride == 12
    assert args.seed == 6455
    assert args.evaluation_mask_key == "source_index_valid_mask"
    assert args.trust_checkpoint is False
    namespace = selection.runner_namespace(selection.resolve_defaults(args))
    assert tuple(namespace.evaluation_years) == (2023,)
    assert namespace.evaluation_mask == (tmp_path / "radar" / "grid_coordinates.npz").resolve()
    assert (namespace.input_frames, namespace.output_frames, namespace.frame_minutes) == (7, 18, 10)


@pytest.mark.parametrize("optimizer_step", [0, 17, 59, 100])
def test_stable_facl_matches_manual_fourier_formula(optimizer_step: int) -> None:
    prediction = torch.linspace(0.01, 0.91, 144, dtype=torch.float64).reshape(2, 1, 3, 4, 6)
    target = prediction.flip(-1) * 0.8 + 0.03
    eps = 1e-8
    actual, components = workflow.StableFACL(total_steps=100, eps=eps)(
        prediction, target, optimizer_step,
    )
    pred_fft = torch.fft.fftn(prediction.float(), dim=(-2, -1), norm="ortho")
    truth_fft = torch.fft.fftn(target.float(), dim=(-2, -1), norm="ortho")
    amplitude = ((pred_fft.abs() - truth_fft.abs()) ** 2).mean()
    numerator = (pred_fft.conj() * truth_fft).sum().real
    energy_product = (pred_fft.abs() ** 2).sum() * (truth_fft.abs() ** 2).sum()
    correlation = 1 - numerator / torch.sqrt(energy_product + eps ** 2)
    alpha = min(1.0, max(0.0, optimizer_step / 59))
    expected = math.sqrt(24) * (alpha * amplitude + (1 - alpha) * correlation)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual.dtype == torch.float32
    assert components["fal"] == pytest.approx(amplitude.item())
    assert components["fcl"] == pytest.approx(correlation.item())
    assert components["fal_weight"] == alpha


@pytest.mark.parametrize("prediction_value,target_value", [(0.0, 0.0), (0.0, 0.3), (0.2, 0.0)])
def test_stable_facl_dry_batches_have_finite_gradients(
    prediction_value: float, target_value: float,
) -> None:
    prediction = torch.full((1, 1, 2, 4, 4), prediction_value, requires_grad=True)
    target = torch.full_like(prediction, target_value)
    loss, components = workflow.StableFACL(total_steps=100)(prediction, target, 17)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(math.isfinite(value) for value in components.values())
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError, match="positive"):
        workflow.StableFACL(total_steps=100, eps=0)


def _write_tiff_block(root: Path, start: str, *, omit: int | None = None) -> None:
    for index in range(25):
        if index == omit:
            continue
        timestamp = np.datetime64(start, "m") + np.timedelta64(index * 10, "m")
        text = np.datetime_as_string(timestamp, unit="m")
        stamp = text.replace("-", "").replace("T", "").replace(":", "")
        directory = root / stamp[:4] / stamp[4:6] / stamp[6:8]
        directory.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            directory / f"{stamp}.tiff",
            np.full((256, 256), index / 100, dtype=np.float32),
            compression="deflate",
        )


@pytest.fixture
def tiff_archive(tmp_path: Path) -> Path:
    root = tmp_path / "radar"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "prepared_contract": {
            "shape": [256, 256], "dtype": "float32 TIFF",
            "normalization": "max(raw_100xdbz, 0) / 10000",
            "minute_phase": 0, "temporal_resolution_minutes": 10,
        },
    }), encoding="utf-8")
    for year in range(2019, 2024):
        _write_tiff_block(root, f"{year}-06-10T00:00")
    # None of these blocks may become a complete training/selection window:
    # the first lacks in-season inputs, the second in-season targets, the
    # third has a missing frame, and the fourth is outside June--September.
    _write_tiff_block(root, "2023-05-31T23:00")
    _write_tiff_block(root, "2023-09-30T20:00")
    _write_tiff_block(root, "2023-07-10T00:00", omit=12)
    _write_tiff_block(root, "2023-10-15T00:00")
    return root


def _dataset(root: Path, years: tuple[int, ...], *, require_targets: bool = True):
    return KmaTiffWindowDataset(
        root, years, input_frames=7, output_frames=18, frame_minutes=10,
        crop=CropContract(height=32, width=32), require_targets=require_targets,
    )


def test_tiff_windows_keep_all_valid_times_within_disjoint_seasonal_splits(
    tiff_archive: Path,
) -> None:
    train = _dataset(tiff_archive, (2019, 2020, 2021, 2022))
    validation = _dataset(tiff_archive, (2023,))
    assert len(train) == 4
    assert len(validation) == 1
    assert train.metadata()["years"] == [2019, 2020, 2021, 2022]
    assert validation.metadata()["years"] == [2023]
    assert validation.metadata()["months"] == [6, 7, 8, 9]
    assert validation.metadata()["issue_time_timezone"] == "KST"
    for dataset, years in [(train, {2019, 2020, 2021, 2022}), (validation, {2023})]:
        for position in range(len(dataset)):
            sample = dataset[position]
            assert sample["inputs"].shape == (1, 7, 32, 32)
            assert sample["targets"].shape == (1, 18, 32, 32)
            torch.testing.assert_close(sample["inputs"][0, :, 0, 0], torch.arange(7) / 100)
            torch.testing.assert_close(sample["targets"][0, :, 0, 0], torch.arange(7, 25) / 100)
            issue = np.datetime64(sample["issue_time_ns"].item(), "ns")
            valid_times = issue + np.arange(-60, 181, 10) * np.timedelta64(1, "m")
            strings = np.datetime_as_string(valid_times, unit="m")
            assert {int(value[:4]) for value in strings} <= years
            assert all(6 <= int(value[5:7]) <= 9 for value in strings)


def test_tiff_input_only_export_does_not_require_future_targets(tiff_archive: Path) -> None:
    dataset = _dataset(tiff_archive, (2023,), require_targets=False)
    last = dataset[len(dataset) - 1]
    assert "targets" not in last
    assert np.datetime64(last["issue_time_ns"].item(), "ns") == np.datetime64("2023-09-30T23:50")
    assert dataset.metadata()["require_targets"] is False


def test_rn60_counts_match_manual_rolling_windows_masks_and_finite_filtering() -> None:
    rng = np.random.default_rng(6455)
    prediction = rng.uniform(0, 35, size=(1, 18, 2, 3)).astype(np.float32)
    target = rng.uniform(0, 35, size=prediction.shape).astype(np.float32)
    prediction[:, :, 0, 0] = target[:, :, 0, 0] = 30
    mask = np.asarray([[True, True, False], [True, True, True]])
    prediction[0, 4, 1, 0] = np.nan
    target[0, 10, 1, 1] = np.inf
    expected = np.zeros((13, 3, 3), dtype=np.int64)
    expected_valid = np.zeros(13, dtype=np.int64)
    for lead_index in range(13):
        forecast = prediction[:, lead_index:lead_index + 6].sum(axis=1) / 6
        observed = target[:, lead_index:lead_index + 6].sum(axis=1) / 6
        valid = mask[None] & np.isfinite(forecast) & np.isfinite(observed)
        expected_valid[lead_index] = valid.sum()
        for threshold_index, threshold in enumerate((1, 5, 10)):
            yes_prediction, yes_target = forecast >= threshold, observed >= threshold
            expected[lead_index, threshold_index] = (
                np.count_nonzero(valid & yes_prediction & yes_target),
                np.count_nonzero(valid & ~yes_prediction & yes_target),
                np.count_nonzero(valid & yes_prediction & ~yes_target),
            )
    counts = torch.zeros((13, 3, 3), dtype=torch.int64)
    valid_counts = torch.zeros(13, dtype=torch.int64)
    for _ in range(2):
        selection.accumulate_counts(
            counts, valid_counts, torch.from_numpy(prediction),
            torch.from_numpy(target), torch.from_numpy(mask),
        )
    np.testing.assert_array_equal(counts.numpy(), 2 * expected)
    np.testing.assert_array_equal(valid_counts.numpy(), 2 * expected_valid)
    assert tuple(selection.LEADS_MIN) == tuple(range(60, 181, 10))
    assert tuple(selection.THRESHOLDS_MM) == (1, 5, 10)
    summary = selection.summarize_counts(counts.numpy())
    csi = expected[:, :, 0] / expected.sum(axis=2)
    np.testing.assert_allclose(summary["csi_by_lead_threshold"], csi)
    assert summary["macro_csi_13lead_3threshold"] == pytest.approx(csi.mean())
    assert np.isfinite(summary["csi_by_lead_threshold"]).all()


def test_selection_score_is_equal_weight_over_all_39_cells() -> None:
    counts = np.zeros((13, 3, 3), dtype=np.int64)
    counts[:, :, 0] = np.arange(39).reshape(13, 3)
    counts[:, :, 1] = 39 - counts[:, :, 0]
    # Unequal sample counts must not change equal lead/threshold weighting.
    counts *= np.arange(1, 40).reshape(13, 3, 1)
    summary = selection.summarize_counts(counts)
    assert summary["macro_csi_13lead_3threshold"] == pytest.approx(np.arange(39).mean() / 39)
    with pytest.raises(selection.SelectionError, match="denominator is zero"):
        selection.summarize_counts(np.zeros((13, 3, 3), dtype=np.int64))
    with pytest.raises(selection.SelectionError, match="shape"):
        selection.summarize_counts(np.zeros((12, 3, 3), dtype=np.int64))


def _candidate(epoch: int, score: float) -> dict:
    return {
        "scientific_contract_sha256": "synthetic-contract",
        "epoch": epoch, "global_step": epoch * 100,
        "checkpoint": f"synthetic_epoch_{epoch:04d}.pt",
        "checkpoint_sha256": f"{epoch:064x}",
        "macro_csi_13lead_3threshold": score,
        "mean_csi_by_threshold": [score] * 3,
        "mean_csi_by_lead": [score] * 13,
    }


def test_selection_includes_all_23_epochs_and_breaks_only_exact_ties(tmp_path: Path) -> None:
    args = argparse.Namespace(expected_checkpoints=23, output_dir=tmp_path)
    payloads = [_candidate(epoch, 0.9 if epoch in (1, 23) else 0.1) for epoch in range(1, 24)]
    result = selection.select_candidates(args, payloads[::-1], "synthetic-contract")
    assert result["candidate_count"] == 23
    assert result["selected"]["epoch"] == 1
    assert result["exact_best_tie_epochs"] == [1, 23]
    assert result["selection_period"] == [2023]
    assert result["test_period_accessed"] is False
    payloads[-1] = _candidate(23, float(np.nextafter(0.9, 1.0)))
    assert selection.select_candidates(args, payloads, "synthetic-contract")["selected"]["epoch"] == 23
    with pytest.raises(selection.SelectionError, match="count"):
        selection.select_candidates(args, payloads[:-1], "synthetic-contract")
    payloads[-1]["scientific_contract_sha256"] = "different-contract"
    with pytest.raises(selection.SelectionError, match="contracts differ"):
        selection.select_candidates(args, payloads, "synthetic-contract")


def test_checkpoint_inventory_hashes_exact_candidate_set_without_deserializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_load(*args, **kwargs):
        pytest.fail("inventory must hash files without deserializing Python state")

    monkeypatch.setattr(torch, "load", forbidden_load)
    for epoch in range(1, 24):
        (tmp_path / f"epoch_{epoch:04d}_step_{epoch * 100:07d}.pt").write_bytes(
            f"synthetic non-checkpoint {epoch}".encode(),
        )
    args = argparse.Namespace(checkpoint_dir=tmp_path, expected_checkpoints=23)
    inventory = selection.checkpoint_inventory(args)
    assert [row["epoch"] for row in inventory] == list(range(1, 24))
    first = Path(inventory[0]["checkpoint"])
    assert inventory[0]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    first.unlink()
    with pytest.raises(selection.SelectionError, match="candidate set changed"):
        selection.checkpoint_inventory(args)


@pytest.mark.parametrize("replace_checkpoint", [False, True])
def test_cached_selection_checks_current_checkpoint_hash_before_finalizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_checkpoint: bool,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "epoch_0001_step_0000100.pt"
    checkpoint.write_bytes(b"original synthetic checkpoint")
    args = argparse.Namespace(
        expected_checkpoints=1, checkpoint_dir=checkpoint_dir,
        output_dir=tmp_path / "evaluations",
    )
    payload = _candidate(1, 0.5) | {
        "schema": selection.EPOCH_SCHEMA, "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    cached = selection.epoch_output_path(args.output_dir, 1)
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps(payload), encoding="utf-8")
    if replace_checkpoint:
        checkpoint.write_bytes(b"replacement synthetic checkpoint")
    preflight = {"scientific_contract_sha256": "synthetic-contract"}
    monkeypatch.setattr(selection, "preflight", lambda actual_args: (
        None, None, None, None, selection.checkpoint_inventory(actual_args), preflight,
    ))
    finalized = []
    monkeypatch.setattr(
        selection, "finalize_outputs",
        lambda actual_args, result, audit: finalized.append((actual_args, result, audit)),
    )
    if replace_checkpoint:
        with pytest.raises(selection.SelectionError, match="checkpoint_sha256"):
            selection.command_select(args)
        assert finalized == []
        assert not (args.output_dir / "COMPLETED.json").exists()
    else:
        selection.command_select(args)
        assert len(finalized) == 1
        assert finalized[0][0] is args
        assert finalized[0][1]["selected"]["checkpoint_sha256"] == payload["checkpoint_sha256"]
        assert finalized[0][2] is preflight


def test_upstream_source_hash_mismatch_is_rejected_before_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    (tmp_path / "model.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n", encoding="utf-8",
    )
    with pytest.raises(model.ModelContractError, match="differs from the audited source"):
        model.load_official_module(tmp_path)
    assert not marker.exists()


def test_upstream_commit_mismatch_is_rejected_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "model.py").write_text("raise AssertionError('must not execute')\n", encoding="utf-8")
    monkeypatch.setattr(model, "sha256_file", lambda _: model.PINNED_OFFICIAL_MODEL_SHA256)
    monkeypatch.setattr(model, "official_repo_commit", lambda _: "0" * 40)
    with pytest.raises(model.ModelContractError, match="audited commit"):
        model.load_official_module(tmp_path)


def test_python_state_checkpoint_paths_require_explicit_trust_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_load(*args, **kwargs):
        pytest.fail("untrusted checkpoint must never be deserialized")

    monkeypatch.setattr(torch, "load", forbidden_load)
    checkpoint = tmp_path / "nonexistent.pt"
    with pytest.raises(ValueError, match="trust_checkpoint"):
        model.load_shape_compatible_weights(torch.nn.Linear(1, 1), checkpoint)
    with pytest.raises(ValueError, match="trust_checkpoint"):
        model.build_exprecast_5min(
            model.ExPreCast5MinConfig(), tmp_path / "absent-upstream",
            initialization_checkpoint=checkpoint,
        )
    with pytest.raises(ValueError, match="--trust-checkpoint"):
        workflow.load_trained_model(
            argparse.Namespace(checkpoint=checkpoint, trust_checkpoint=False), torch.device("cpu"),
        )
    with pytest.raises(selection.SelectionError, match="--trust-checkpoint"):
        selection.command_run(argparse.Namespace(dry_run=False, trust_checkpoint=False))


@pytest.mark.parametrize("mode", ["initialization", "resume"])
def test_training_checkpoint_gate_precedes_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    args = workflow.build_parser().parse_args([
        "train", *_common_arguments(tmp_path), "--run-name", "synthetic",
        "--output-root", str(tmp_path / "training"), "--device", "cpu",
    ])
    if mode == "initialization":
        args.initialization_checkpoint = tmp_path / "untrusted.pt"
    else:
        run = args.output_root / args.run_name / f"seed_{args.seed}"
        run.mkdir(parents=True)
        (run / "last.pt").write_bytes(b"not a real checkpoint")

    def forbidden(*args, **kwargs):
        pytest.fail("an untrusted training checkpoint must be rejected before model construction")

    monkeypatch.setattr(workflow, "construct_model", forbidden)
    monkeypatch.setattr(workflow, "set_seed", lambda _: None)
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="--trust-checkpoint"):
        workflow.run_train(args)


def test_synthetic_export_matches_existing_adapter_schema_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    issue_ns = int(np.datetime64("2023-06-10T01:00", "ns").astype(np.int64))

    class TinyDataset(torch.utils.data.Dataset):
        crop = CropContract().resolve((256, 256))

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "inputs": torch.zeros((1, 7, 256, 256)),
                "issue_time_ns": torch.tensor(issue_ns, dtype=torch.int64),
            }

        def metadata(self):
            return {"synthetic": True, "window_count": 1}

    class TinyModel(torch.nn.Module):
        calls = 0

        def forward(self, inputs):
            self.calls += 1
            assert tuple(inputs.shape) == (1, 1, 7, 256, 256)
            return torch.linspace(0, 1, 18).reshape(1, 1, 18, 1, 1).expand(1, 1, 18, 256, 256)

    tiny = TinyModel()
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic marker; never deserialized")
    output = tmp_path / "fields.h5"
    args = workflow.build_parser().parse_args([
        "export", *_common_arguments(tmp_path), "--checkpoint", str(checkpoint),
        "--trust-checkpoint", "--export-years", "2023", "--export-output", str(output),
        "--device", "cpu", "--precision", "fp32",
    ])
    monkeypatch.setattr(workflow, "load_trained_model", lambda *args: (tiny, {"synthetic": True}, {}))
    monkeypatch.setattr(workflow, "make_dataset", lambda *args, **kwargs: TinyDataset())
    workflow.run_export(args)
    assert tiny.calls == 1
    with h5py.File(output, "r") as fields:
        assert set(fields) == {"forecast_normalized_dbz", "issue_time_ns", "lead_minutes", "completed"}
        assert fields["forecast_normalized_dbz"].shape == (1, 18, 256, 256)
        assert fields["forecast_normalized_dbz"].dtype == np.dtype("float16")
        np.testing.assert_array_equal(fields["issue_time_ns"], [issue_ns])
        np.testing.assert_array_equal(fields["completed"], [True])
        np.testing.assert_array_equal(fields["lead_minutes"], np.arange(10, 181, 10))
        np.testing.assert_allclose(fields["forecast_normalized_dbz"][0, :, 0, 0], np.linspace(0, 1, 18), atol=3e-4)
        assert fields.attrs["encoding"] == "dBZ = 100 * forecast_normalized_dbz"
        assert fields.attrs["issue_time_timezone"] == "KST"
        contract = json.loads(fields.attrs["contract_json"])
        assert contract["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        validate_field_axes(
            fields["forecast_normalized_dbz"].shape,
            np.asarray(fields["lead_minutes"]), np.asarray(fields["issue_time_ns"]),
        )
    audit = inspect_hdf5(output)
    assert audit["field_shape"] == [1, 18, 256, 256]
    assert audit["issue_count"] == 1
    assert audit["checkpoint_sha256"] == contract["checkpoint_sha256"]
    assert output.with_suffix(".contract.json").is_file()
    args.resume_export = True
    workflow.run_export(args)
    assert tiny.calls == 1, "a completed resume must not infer or overwrite any row"
