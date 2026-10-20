#!/usr/bin/env python3
"""Adapted exPreCast model for the primary 4-km/10-min field workflow.

Uses the separately obtained, hash-pinned upstream model implementation.
Adapted from exPreCast by Changhoon Song, Teng Yuan Chang and Youngjoon Hong;
shared with permission. See THIRD_PARTY_NOTICES.md for attribution and scope.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import subprocess
import sys
import types
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


OFFICIAL_REPOSITORY = "https://github.com/tony890048/exPreCast"
PINNED_OFFICIAL_COMMIT = "092922c126bdf6098fbda2e08b2f1f3b2873bd9a"
PINNED_OFFICIAL_MODEL_SHA256 = (
    "1d8fa4d16ea9631dd8b3addfeaaa3ca36d0f641f9dfc307588b7ee9424e70625"
)


class ModelContractError(RuntimeError):
    """Raised when model and temporal/spatial contracts are incompatible."""


class _DropPath(nn.Module):
    """Training-safe equivalent of timm.layers.DropPath."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return value
        keep = 1.0 - self.drop_prob
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        random_tensor = value.new_empty(shape).bernoulli_(keep)
        return value.div(keep) * random_tensor


def _rearrange(value: torch.Tensor, pattern: str) -> torch.Tensor:
    normalized = " ".join(pattern.split()).lower()
    permutations = {
        "b c d h w -> b d h w c": (0, 2, 3, 4, 1),
        "b d h w c -> b c d h w": (0, 4, 1, 2, 3),
        "b t h w c -> b c t h w": (0, 4, 1, 2, 3),
        "b c t h w -> b t h w c": (0, 2, 3, 4, 1),
        "n c d h w -> n d h w c": (0, 2, 3, 4, 1),
        "n d h w c -> n c d h w": (0, 4, 1, 2, 3),
    }
    try:
        return value.permute(*permutations[normalized])
    except KeyError as error:
        raise ModelContractError(
            f"unsupported rearrange expression in pinned source: {pattern}"
        ) from error


def install_official_compatibility_modules() -> dict[str, str]:
    """Install minimal timm/einops shims if those optional packages fail.

    The project environment used for the radar pipeline intentionally does not
    require timm or einops.  The official model needs only DropPath,
    trunc_normal_, and six fixed 5-D permutations.
    """

    compatibility: dict[str, str] = {}
    try:
        from timm.layers import DropPath as _  # noqa: F401
    except Exception:
        timm_module = types.ModuleType("timm")
        timm_layers = types.ModuleType("timm.layers")
        timm_layers.DropPath = _DropPath
        timm_layers.trunc_normal_ = torch.nn.init.trunc_normal_
        timm_module.layers = timm_layers
        sys.modules["timm"] = timm_module
        sys.modules["timm.layers"] = timm_layers
        compatibility["timm"] = "local DropPath/trunc_normal_ compatibility shim"
    try:
        from einops import rearrange as _  # noqa: F401
    except Exception:
        einops_module = types.ModuleType("einops")
        einops_module.rearrange = _rearrange
        sys.modules["einops"] = einops_module
        compatibility["einops"] = "local fixed-permutation compatibility shim"
    return compatibility


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_repo_commit(path: Path) -> str | None:
    # An enclosing downstream Git repository is not an upstream checkout.
    if not (Path(path) / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def load_official_module(official_repo: Path):
    official_repo = Path(official_repo)
    model_path = official_repo / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"official exPreCast model.py not found: {model_path}")
    model_sha256 = sha256_file(model_path)
    if model_sha256 != PINNED_OFFICIAL_MODEL_SHA256:
        raise ModelContractError(
            "official model.py differs from the audited source: "
            f"expected {PINNED_OFFICIAL_MODEL_SHA256}, found {model_sha256}"
        )
    commit = official_repo_commit(official_repo)
    if commit is not None and commit != PINNED_OFFICIAL_COMMIT:
        raise ModelContractError(
            "official repository is not at the audited commit: "
            f"expected {PINNED_OFFICIAL_COMMIT}, found {commit}"
        )
    compatibility = install_official_compatibility_modules()
    module_name = f"official_exprecast_model_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ModelContractError(f"could not load official model source: {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    # Preserve the official re-entrant checkpoint behavior while making the
    # choice explicit for current/future PyTorch versions.
    official_checkpoint = module.checkpoint.checkpoint

    def checkpoint_explicit(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", True)
        return official_checkpoint(function, *args, **kwargs)

    module.checkpoint = types.SimpleNamespace(checkpoint=checkpoint_explicit)
    provenance = {
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit or PINNED_OFFICIAL_COMMIT,
        "model_path": str(model_path.resolve()),
        "model_sha256": model_sha256,
        "compatibility_modules": compatibility,
    }
    return module, provenance


@dataclass(frozen=True)
class ExPreCast5MinConfig:
    input_frames: int = 7
    output_frames: int = 18
    frame_minutes: int = 10
    profile: str = "paper_long_concat"
    patch_embed_size: tuple[int, int, int] = (2, 4, 4)
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 6, 2, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    window_size: tuple[int, int, int] = (2, 7, 7)
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2
    use_checkpoint: bool = True

    def validate(self) -> None:
        if self.input_frames < 2 or self.output_frames < 1:
            raise ModelContractError("input_frames >= 2 and output_frames >= 1 are required")
        if self.frame_minutes < 1:
            raise ModelContractError("frame_minutes must be positive")
        if self.profile not in {"paper_long_concat", "memory_safe_add"}:
            raise ModelContractError(f"unknown exPreCast profile: {self.profile}")
        if not (len(self.depths) == len(self.num_heads) == 4):
            raise ModelContractError("the audited exPreCast topology requires four stages")
        for stage, (heads, width) in enumerate(
            zip(self.num_heads, [self.embed_dim * 2**i for i in range(4)])
        ):
            if width % heads:
                raise ModelContractError(
                    f"stage {stage} width {width} is not divisible by {heads} heads"
                )

    @property
    def input_offsets_minutes(self) -> tuple[int, ...]:
        return tuple(
            range(-(self.input_frames - 1) * self.frame_minutes, 1, self.frame_minutes)
        )

    @property
    def lead_minutes(self) -> tuple[int, ...]:
        return tuple(range(self.frame_minutes, self.output_frames * self.frame_minutes + 1, self.frame_minutes))

    @property
    def model_kwargs(self) -> dict[str, Any]:
        self.validate()
        if self.profile == "paper_long_concat":
            # Table 9 long-range decoder: temporal CDU x2, temporal concat
            # skips, and no temporal expansion in the final patch decoder.
            upsampling_scale = (2, 2, 2)
            patch_expan_size = (1, 4, 4)
            skip_connection = "concat"
        else:
            # Explicit OOM fallback based on the paper's short-range topology.
            # It preserves LSTA/CDU/TE but is not the Table 9 long decoder.
            upsampling_scale = (1, 2, 2)
            patch_expan_size = (2, 4, 4)
            skip_connection = "add"
        return {
            "input_frames": self.input_frames,
            "output_frames": self.output_frames,
            "in_chans": 1,
            "out_chans": 1,
            "patch_embed_size": self.patch_embed_size,
            "patch_expan_size": patch_expan_size,
            "upsampling_scale": upsampling_scale,
            "downsampling_scale": (1, 2, 2),
            "embed_dim": self.embed_dim,
            "depths": list(self.depths),
            "num_heads": list(self.num_heads),
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": True,
            "drop_path_rate": self.drop_path_rate,
            "patch_norm": False,
            "frozen_stages": None,
            "skip_connection": skip_connection,
            "use_checkpoint": self.use_checkpoint,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "input_offsets_minutes": list(self.input_offsets_minutes),
            "lead_minutes": list(self.lead_minutes),
            "model_kwargs": self.model_kwargs,
        }


def expected_latent_time(config: ExPreCast5MinConfig) -> int:
    embedded = math.ceil(config.input_frames / config.patch_embed_size[0])
    if config.profile == "paper_long_concat":
        current = embedded
        for _ in range(3):
            current = current * 2 + embedded
        return current
    return embedded * 2


def _clean_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping):
        raise ModelContractError(f"checkpoint object is {type(payload).__name__}, expected mapping")
    return {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }


def load_shape_compatible_weights(
    model: nn.Module,
    checkpoint: Path,
    *,
    encoder_only: bool = False,
    trust_checkpoint: bool = False,
) -> dict[str, Any]:
    if not trust_checkpoint:
        raise ValueError(
            "checkpoint deserialization can execute Python code; pass "
            "trust_checkpoint=True only for a checkpoint you trust"
        )
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    source = _clean_state_dict(payload)
    target = model.state_dict()
    prefixes = ("patch_embed.", "encoder.", "pos_drop.")
    compatible = {
        key: value
        for key, value in source.items()
        if key in target
        and target[key].shape == value.shape
        and (not encoder_only or key.startswith(prefixes))
    }
    model.load_state_dict(compatible, strict=False)
    skipped = sorted(set(source) - set(compatible))
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(checkpoint)),
        "source_tensor_count": len(source),
        "loaded_tensor_count": len(compatible),
        "skipped_tensor_count": len(source) - len(compatible),
        "loaded_tensor_keys": sorted(compatible),
        "skipped_tensor_keys": skipped,
        "encoder_only": bool(encoder_only),
    }


def freeze_encoder(model: nn.Module) -> int:
    """Freeze the paper's encoder while leaving stage 4 as trainable bottleneck.

    This reproduces official ``frozen_stages=5``: PatchEmbed, Dropout, and
    encoder stages 1--3 are frozen; the fourth encoder block is the trainable
    bottleneck.  Freezing all four encoder blocks would not reproduce the
    paper's reported 25.5-M trainable-parameter transfer model.
    """

    frozen = 0
    modules = [model.patch_embed, *list(model.encoder[:-1])]
    for module in modules:
        module.eval()
        for child in module.modules():
            if hasattr(child, "use_checkpoint"):
                child.use_checkpoint = False
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    model._explicitly_frozen_modules = tuple(modules)
    return frozen


def build_exprecast_5min(
    config: ExPreCast5MinConfig,
    official_repo: Path,
    *,
    initialization_checkpoint: Path | None = None,
    encoder_only_initialization: bool = False,
    freeze_encoder_parameters: bool = False,
    trust_checkpoint: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    config.validate()
    if initialization_checkpoint is not None and not trust_checkpoint:
        raise ValueError("initialization requires explicit trust_checkpoint=True")
    module, provenance = load_official_module(Path(official_repo))

    class QuietExPreCast(module.exPreCast):
        """Exact official modules with a print-free, shape-checked forward."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            if value.ndim != 5 or value.shape[1] != 1:
                raise ModelContractError(
                    f"expected [B,1,T,H,W], received {tuple(value.shape)}"
                )
            if value.shape[2] != self.input_frames:
                raise ModelContractError(
                    f"expected {self.input_frames} input frames, received {value.shape[2]}"
                )
            original_height, original_width = value.shape[-2:]
            if original_height % 32 or original_width % 32:
                raise ModelContractError(
                    "height and width must be multiples of 32 for exact encoder-decoder skips; "
                    f"received {(original_height, original_width)}"
                )
            value = self.pos_drop(self.patch_embed(value))
            skips = []
            for index, layer in enumerate(self.encoder):
                if (
                    index == self.num_layers - 1
                    and self.training
                    and getattr(self, "_explicit_encoder_freeze", False)
                    and self.use_checkpoint
                    and not value.requires_grad
                ):
                    # Re-entrant torch checkpointing needs at least one input
                    # requiring gradients.  Earlier stages are intentionally
                    # frozen, so create a leaf at the trainable bottleneck
                    # boundary without reconnecting the frozen graph.
                    value = value.detach().requires_grad_(True)
                value, skip = layer(value.contiguous())
                if index < self.num_layers - 1:
                    skips.append(skip)
            value = value.permute(0, 2, 3, 4, 1)
            value = self.bottleneck_upscale(value)
            value = value.permute(0, 4, 1, 2, 3)
            for index, layer in enumerate(self.decoder):
                skip = skips[-(index + 1)]
                if self.skip_connection == "concat":
                    value = torch.cat([value, skip], dim=2)
                elif self.skip_connection == "add":
                    if value.shape != skip.shape:
                        raise ModelContractError(
                            f"add-skip shape mismatch: {tuple(value.shape)} vs {tuple(skip.shape)}"
                        )
                    value = value + skip
                else:
                    raise ModelContractError(f"unknown skip connection {self.skip_connection}")
                value, _ = layer(value.contiguous())
            value = self.patch_expand3d(value)
            if self.last_time_dim != self.output_frames:
                value = value.permute(0, 2, 3, 4, 1)
                value = self.time_extractor(value)
                value = value.permute(0, 4, 1, 2, 3)
            if tuple(value.shape[-2:]) != (original_height, original_width):
                raise ModelContractError(
                    f"output grid changed unexpectedly: {tuple(value.shape[-2:])}"
                )
            return value

        def train(self, mode: bool = True):
            super().train(mode)
            # The official frozen_stages indexing is ambiguous for transfer.
            # Explicitly frozen encoder parameters stay in eval mode here.
            if getattr(self, "_explicit_encoder_freeze", False):
                for frozen_module in self._explicitly_frozen_modules:
                    frozen_module.eval()
            return self

    model = QuietExPreCast(**config.model_kwargs)
    model.use_checkpoint = bool(config.use_checkpoint)
    model.init_weights()
    if model.last_time_dim != expected_latent_time(config):
        raise ModelContractError(
            f"temporal-contract audit failed: model={model.last_time_dim}, "
            f"expected={expected_latent_time(config)}"
        )
    initialization = None
    if initialization_checkpoint is not None:
        initialization = load_shape_compatible_weights(
            model,
            Path(initialization_checkpoint),
            encoder_only=encoder_only_initialization,
            trust_checkpoint=trust_checkpoint,
        )
    frozen_parameter_count = 0
    if freeze_encoder_parameters:
        frozen_parameter_count = freeze_encoder(model)
        model._explicit_encoder_freeze = True
    metadata = {
        "official_source": provenance,
        "contract": config.to_dict(),
        "latent_time_features": model.last_time_dim,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameter_count": frozen_parameter_count,
        "initialization": initialization,
    }
    return model, metadata


__all__ = [
    "ExPreCast5MinConfig",
    "ModelContractError",
    "PINNED_OFFICIAL_COMMIT",
    "PINNED_OFFICIAL_MODEL_SHA256",
    "build_exprecast_5min",
    "expected_latent_time",
    "freeze_encoder",
    "load_shape_compatible_weights",
]
