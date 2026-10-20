"""CNN used for gauge-referenced point readout after field forecasts."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
N_FRAMES = 6
PATCH_SIZE = 3

def _validate_patch_tensor(
    patches: torch.Tensor,
    name: str,
    patch_steps: int = N_FRAMES,
    patch_size: int = PATCH_SIZE,
) -> None:
    if not isinstance(patches, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if patches.ndim < 4:
        raise ValueError(
            f"{name} must end in [{patch_steps},{patch_size},{patch_size}], "
            f"got shape {tuple(patches.shape)}"
        )
    if tuple(patches.shape[-3:]) != (patch_steps, patch_size, patch_size):
        raise ValueError(
            f"{name} must end in [{patch_steps},{patch_size},{patch_size}], "
            f"got shape {tuple(patches.shape)}"
        )
    if not torch.is_floating_point(patches):
        raise TypeError(f"{name} must be a floating-point tensor")


def _validate_aux_tensor(
    patches: torch.Tensor,
    auxiliary: Optional[torch.Tensor],
    auxiliary_dim: int,
    name: str,
) -> None:
    leading = tuple(patches.shape[:-3])
    if auxiliary_dim == 0:
        if auxiliary is not None and auxiliary.numel() != 0:
            raise ValueError(f"{name} must be None when auxiliary_dim=0")
        return
    if auxiliary is None:
        raise ValueError(f"{name} is required because auxiliary_dim={auxiliary_dim}")
    if not isinstance(auxiliary, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected = leading + (auxiliary_dim,)
    if tuple(auxiliary.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(auxiliary.shape)}")
    if not torch.is_floating_point(auxiliary):
        raise TypeError(f"{name} must be a floating-point tensor")
    if auxiliary.device != patches.device:
        raise ValueError(f"{name} and patches must be on the same device")


class ResidualConvBlock(nn.Module):
    """Query-independent residual block using GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError((in_channels, out_channels))
        groups = 8 if out_channels % 8 == 0 else 1
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.main(inputs) + self.skip(inputs))


class AttentionPool2d(nn.Module):
    """Learned spatial pooling followed by a fixed-width projection."""

    def __init__(self, in_channels: int, output_dim: int):
        super().__init__()
        if in_channels <= 0 or output_dim <= 0:
            raise ValueError((in_channels, output_dim))
        self.score = nn.Conv2d(in_channels, 1, 1)
        self.project = nn.Linear(in_channels, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"inputs must be [N,C,H,W], got {tuple(inputs.shape)}")
        weights = self.score(inputs).flatten(2).softmax(dim=-1)
        values = inputs.flatten(2)
        pooled = torch.sum(values * weights, dim=-1)
        return self.project(pooled)


class CNN(nn.Module):
    """Six-frame 3 x 3 CNN predicting normalized RN60."""

    def __init__(
        self,
        patch_steps: int = 6,
        patch_size: int = 3,
        aux_dim: int = 39,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_steps <= 0 or patch_size <= 0 or aux_dim < 0 or not 0.0 <= dropout < 1.0:
            raise ValueError((patch_steps, patch_size, aux_dim, dropout))
        self.patch_steps = int(patch_steps)
        self.patch_size = int(patch_size)
        self.aux_dim = int(aux_dim)
        self.projection = nn.Sequential(
            nn.Conv2d(self.patch_steps, 16, 1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            ResidualConvBlock(16, 32),
            ResidualConvBlock(32, 32),
        )
        self.pool = AttentionPool2d(32, 128)
        self.auxiliary_projection = (
            nn.Linear(self.aux_dim, 128) if self.aux_dim else None
        )
        self.norm = nn.LayerNorm(128)
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        patches: torch.Tensor,
        auxiliary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _validate_patch_tensor(
            patches, "patches", self.patch_steps, self.patch_size
        )
        _validate_aux_tensor(patches, auxiliary, self.aux_dim, "auxiliary")
        leading = patches.shape[:-3]
        values = patches.reshape(
            -1, self.patch_steps, self.patch_size, self.patch_size
        )
        latent = self.pool(self.backbone(self.projection(values)))
        if self.aux_dim:
            assert auxiliary is not None and self.auxiliary_projection is not None
            latent = latent + self.auxiliary_projection(
                auxiliary.reshape(-1, self.aux_dim)
            )
        latent = self.norm(F.gelu(latent))
        result = F.softplus(self.output(latent).squeeze(-1), beta=5.0)
        return result.reshape(leading)
