"""Explicit, versioned experiment settings; no test selection or implicit launch."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT = {
    "schema": "nowcastnet-hsr-field-v5",
    "test_issue_policy": "hsr_available_windows",
    "project_root": str(ROOT),
    "experiment_root": "outputs/nowcastnet_field",
    "cache_root": "cache/nowcastnet_field",
    "radar_root": "data/private/radar_4km10min_tiff",
    "hsr_cache_root": "data/private/hsr_monthly_cache",
    "max_open_hsr_months": 2,
    "train_years": [2019, 2020, 2021, 2022],
    "validation_years": [2023], "test_years": [2024, 2025],
    "months": [6, 7, 8, 9],
    "train_stride_minutes": 10, "validation_stride_minutes": 10,
    "model": {"grid_size": 256, "input_steps": 7, "output_steps": 18,
              "base_channels": 32, "noise_channels": 32, "rate_scale": 128.0,
              "rain_rate_cap": 128.0},
    "field": {"seed": 40, "batch_size": 16, "validation_batch_size": 1,
              "num_workers": 0, "pin_memory": True, "prefetch_factor": 2,
              "threads": 4, "precision": "bf16",
              "evolution_epochs": 30, "generator_epochs": 30, "patience": 8,
              "evolution_iterations": 300000, "generator_iterations": 500000,
              "lr_scheduler": "plateau", "lr_monitor": "validation_macro_csi",
              "lr_patience": 4, "lr_factor": 0.1, "lr_min": 0.0, "lr_eps": 1e-8,
              "importance_sampling": True, "sampling_epsilon": 1e-6,
              "adam_betas": [0.9, 0.999],
              "min_delta": 0.0, "evolution_lr": 1e-3, "generator_lr": 3e-5,
              "discriminator_lr": 3e-5, "discriminator_input_scale": 1.0, "motion_weight": 0.01,
              "adversarial_weight": 6.0, "pool_weight": 20.0,
              "pool_samples": 4, "gradient_clip": None, "log_every": 100},
    "evaluation": {"spatial_resolution_km": 4.0},
    "forecast_product": {"kind": "mean", "ensemble_members": 4,
                         "noise_seed": 20261015,
                         "rule": "sha256(seed,issue_ns,member)",
                         "nonnegative": "clamp each member before averaging",
                         "spectral_state": "restore noise-projector u/v after each four-member product"},
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_config(path=None):
    config = deepcopy(DEFAULT)
    if path is not None:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(supplied) - set(config)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        for key, value in supplied.items():
            if isinstance(config[key], dict):
                if not isinstance(value, dict) or set(value) - set(config[key]):
                    raise ValueError(f"Unknown or invalid settings in {key}")
                config[key].update(value)
            else:
                config[key] = value
    root = Path(config["project_root"]).expanduser().resolve()
    for key in ("project_root", "experiment_root", "cache_root", "radar_root", "hsr_cache_root"):
        value = Path(config[key]).expanduser()
        config[key] = str((value if value.is_absolute() else root / value).resolve())
    validate_config(config)
    return config


def validate_config(config):
    for key in ("schema", "test_issue_policy", "train_years", "validation_years", "test_years", "months"):
        if config[key] != DEFAULT[key]:
            raise ValueError(f"This experiment freezes {key}={DEFAULT[key]}")
    model, field, product = (config[k] for k in ("model", "field", "forecast_product"))
    if model["grid_size"] != 256 or model["input_steps"] != 7 or model["output_steps"] != 18:
        raise ValueError("Production grid/time contract is 256x256, 7 observed and 18 future 10-min frames")
    for key in ("base_channels", "noise_channels"):
        if not isinstance(model[key], int) or isinstance(model[key], bool) or model[key] < 4:
            raise ValueError(f"Invalid {key}")
    if model["rate_scale"] != 128.0 or model["rain_rate_cap"] <= 0 or field["precision"] not in ("fp32", "bf16"):
        raise ValueError("Invalid normalization or precision")
    for key in ("train_stride_minutes", "validation_stride_minutes"):
        if config[key] != 10:
            raise ValueError("Production train/validation use all eligible 10-min issues")
    for key in ("batch_size", "validation_batch_size", "threads", "evolution_iterations", "generator_iterations",
                "lr_patience", "pool_samples", "log_every", "prefetch_factor"):
        if not isinstance(field[key], int) or isinstance(field[key], bool) or field[key] <= 0:
            raise ValueError(f"field.{key} must be a positive integer")
    for key in ("evolution_epochs", "generator_epochs"):
        if field[key] is not None and (type(field[key]) is not int or field[key] < 1):
            raise ValueError(f"field.{key} must be null (iteration budget) or a positive explicit epoch limit")
    if type(field["patience"]) is not int or field["patience"] < 0:
        raise ValueError("field.patience must be nonnegative; 0 disables early stopping")
    if field["lr_scheduler"] != "plateau" or field["lr_monitor"] != "validation_macro_csi":
        raise ValueError("Field LR scheduling uses plateau on validation_macro_csi")
    for key in ("lr_factor", "lr_min", "lr_eps"):
        if type(field[key]) not in (int, float) or not math.isfinite(field[key]):
            raise ValueError(f"field.{key} must be finite numeric")
    if not 0 < field["lr_factor"] < 1 or field["lr_min"] < 0 or field["lr_eps"] < 0:
        raise ValueError("Invalid plateau LR factor, minimum or epsilon")
    if field["lr_min"] > min(field[key] for key in ("evolution_lr", "generator_lr", "discriminator_lr")):
        raise ValueError("field.lr_min cannot exceed any initial learning rate")
    if type(field["importance_sampling"]) is not bool or field["sampling_epsilon"] <= 0:
        raise ValueError("Invalid importance sampling settings")
    if (not isinstance(field["adam_betas"], list) or len(field["adam_betas"]) != 2
            or any(not 0 <= value < 1 for value in field["adam_betas"])):
        raise ValueError("Adam betas must be two values in [0,1)")
    if type(field["num_workers"]) is not int or field["num_workers"] < 0 or field["min_delta"] < 0 or field["seed"] < 0:
        raise ValueError("Invalid worker count, stopping tolerance or seed")
    if type(field["pin_memory"]) is not bool:
        raise ValueError("field.pin_memory must be boolean")
    if type(config["max_open_hsr_months"]) is not int or config["max_open_hsr_months"] < 1:
        raise ValueError("max_open_hsr_months must be a positive integer")
    for key in ("evolution_lr", "generator_lr", "discriminator_lr"):
        if field[key] <= 0:
            raise ValueError(f"field.{key} must be positive")
    if field["gradient_clip"] is not None and field["gradient_clip"] <= 0:
        raise ValueError("gradient_clip must be null (disabled) or positive")
    if field["discriminator_input_scale"] != 1.0:
        raise ValueError("The v5 discriminator uses raw physical rain rates: discriminator_input_scale=1")
    evaluation = config["evaluation"]
    if evaluation["spatial_resolution_km"] != 4.0:
        raise ValueError("The HSR grid spacing is 4 km")
    if product != DEFAULT["forecast_product"]:
        raise ValueError("Freeze the four-draw, issuance-keyed forecast product for this experiment")
    if Path(config["experiment_root"]) == Path(config["cache_root"]):
        raise ValueError("Separate experiment and cache directories")
    root = Path(config["project_root"]).resolve()
    protected = [root / name for name in ("src", "data", "scripts", "configs", "nowcasting-r2p", "external")]
    protected.extend(Path(config[key]).resolve() for key in ("radar_root", "hsr_cache_root"))
    destinations = [Path(config[key]).resolve() for key in ("experiment_root", "cache_root")]
    for destination in destinations:
        if destination == root or destination in root.parents:
            raise ValueError("Output/cache cannot replace the project or an ancestor")
        for source in protected:
            if destination == source or destination in source.parents or source in destination.parents:
                raise ValueError(f"Output/cache overlaps protected source: {source}")
    if destinations[0] in destinations[1].parents or destinations[1] in destinations[0].parents:
        raise ValueError("Output and cache roots must not be nested")


@contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def implementation_identity():
    directory = Path(__file__).parent
    return {p.name: digest(p) for p in sorted(directory.glob("*.py"))}
