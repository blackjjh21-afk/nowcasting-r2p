"""Vanilla Held-out R2P architecture on the official 4-km KMA grid.

Only the radar input contract differs from the established Held-out R2P:
seven 10-min TIFF frames replace twelve 5-min 2-km HSR frames.  Gauge history,
station dropout, point leads, model widths and the point decoder are retained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class R2PModelConfig:
    radar_steps: int = 7
    station_feature_count: int = 5
    grid_height: int = 256
    grid_width: int = 256
    d_model: int = 128
    base_channels: int = 32
    point_leads: int = 36
    readout_window: int = 5
    heads: int = 4
    cross_lead_layers: int = 1
    station_hidden: int = 64
    dropout: float = 0.1
    deform_points: int = 6
    deform_offset_scale: float = 0.5

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DepthwiseSeparableConv(nn.Module):
    """Exact depthwise-multiplier block used by the established R2P."""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel: int = 3,
        kernels_per_layer: int = 2,
    ):
        super().__init__()
        padding = kernel // 2
        self.depthwise = nn.Conv2d(
            channels_in,
            channels_in * kernels_per_layer,
            kernel,
            padding=padding,
            groups=channels_in,
        )
        self.pointwise = nn.Conv2d(
            channels_in * kernels_per_layer, channels_out, 1
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(value))


class DoubleConvDS(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(channels_in, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(channels_out, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class ConvGRUCell(nn.Module):
    def __init__(self, channels_in: int, channels_hidden: int, kernel: int = 3):
        super().__init__()
        padding = kernel // 2
        self.channels_hidden = channels_hidden
        self.gates = nn.Conv2d(
            channels_in + channels_hidden, 2 * channels_hidden, kernel, padding=padding
        )
        self.candidate = nn.Conv2d(
            channels_in + channels_hidden, channels_hidden, kernel, padding=padding
        )

    def forward(
        self, value: torch.Tensor, hidden: torch.Tensor | None
    ) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                value.shape[0],
                self.channels_hidden,
                value.shape[2],
                value.shape[3],
                device=value.device,
                dtype=value.dtype,
            )
        update, reset = torch.sigmoid(
            self.gates(torch.cat([value, hidden], dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat([value, reset * hidden], dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class SharedEncoder(nn.Module):
    """The established vanilla R2P ConvGRU spatial pyramid."""

    def __init__(self, base_channels: int, d_model: int):
        super().__init__()
        self.frame_stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.gru = ConvGRUCell(base_channels, base_channels * 2)
        self.post = DoubleConvDS(base_channels * 2, base_channels * 2)
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConvDS(base_channels * 2, base_channels * 2)
        )
        self.down8 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConvDS(base_channels * 2, base_channels * 3)
        )
        self.down16 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConvDS(base_channels * 3, base_channels * 4)
        )
        self.project2 = nn.Conv2d(base_channels * 2, d_model, 1)
        self.project8 = nn.Conv2d(base_channels * 3, d_model, 1)
        self.project16 = nn.Conv2d(base_channels * 4, d_model, 1)

    def forward(self, radar: torch.Tensor) -> dict[int, torch.Tensor]:
        batch, steps, height, width = radar.shape
        frames = self.frame_stem(radar.reshape(batch * steps, 1, height, width))
        channels, height2, width2 = frames.shape[1:]
        frames = frames.reshape(batch, steps, channels, height2, width2)
        hidden = None
        for step in range(steps):
            hidden = self.gru(frames[:, step], hidden)
        feature2 = self.post(hidden)
        feature8 = self.down8(self.down4(feature2))
        feature16 = self.down16(feature8)
        return {
            2: self.project2(feature2),
            8: self.project8(feature8),
            16: self.project16(feature16),
        }


class StationAttention(nn.Module):
    def __init__(
        self, dimension: int, positions_km: np.ndarray, heads: int = 4, hidden: int = 32
    ):
        super().__init__()
        self.heads = heads
        positions = torch.as_tensor(positions_km, dtype=torch.float32)
        relative = (positions[None] - positions[:, None]) / 100.0
        distance = torch.linalg.norm(relative, dim=-1, keepdim=True)
        # Geometry is regenerated for the 514-node fit and 642-node evaluation
        # query sets and therefore must not enter a checkpoint.
        self.register_buffer(
            "relative_features", torch.cat([relative, distance], dim=-1), persistent=False
        )
        self.bias = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, heads)
        )
        self.attention = nn.MultiheadAttention(
            dimension, heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dimension)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, stations, _ = value.shape
        mask = self.bias(self.relative_features).permute(2, 0, 1)
        mask = mask.unsqueeze(0).expand(batch, -1, -1, -1)
        mask = mask.reshape(batch * self.heads, stations, stations)
        normalized = self.norm(value)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask
        )
        return value + attended


class DeformableUpstream(nn.Module):
    def __init__(
        self, dimension: int, points: int = 6, offset_scale: float = 0.5
    ):
        super().__init__()
        self.points = points
        self.offset_scale = offset_scale
        self.offset = nn.Linear(dimension, points * 2)
        self.project = nn.Linear(dimension, dimension)
        self.point_embedding = nn.Parameter(
            torch.randn(1, 1, points, dimension) * 0.02
        )

    def forward(
        self,
        fine_feature: torch.Tensor,
        station_norm: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        batch, dimension, _, _ = fine_feature.shape
        stations = station_norm.shape[0]
        leads = query.shape[2]
        offset = torch.tanh(self.offset(query)) * self.offset_scale
        offset = offset.reshape(batch, stations, leads, self.points, 2)
        base = station_norm.reshape(1, stations, 1, 1, 2)
        positions = (base + offset).clamp(-1.2, 1.2)
        grid = positions.reshape(batch, stations * leads * self.points, 1, 2)
        sampled = F.grid_sample(
            fine_feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        sampled = sampled.reshape(batch, stations, leads, self.points, dimension)
        return self.project(sampled) + self.point_embedding


def _window_indices(
    station_y: np.ndarray,
    station_x: np.ndarray,
    padded_height: int,
    padded_width: int,
    strides: tuple[int, ...],
    radius: int,
) -> torch.Tensor:
    station_y = np.asarray(station_y, dtype=np.int64)
    station_x = np.asarray(station_x, dtype=np.int64)
    offsets = np.arange(-radius, radius + 1)
    delta_y, delta_x = np.meshgrid(offsets, offsets, indexing="ij")
    delta_y, delta_x = delta_y.ravel(), delta_x.ravel()
    result = np.zeros(
        (len(strides), len(station_y), len(delta_y)), dtype=np.int64
    )
    for level, stride in enumerate(strides):
        level_width = padded_width // stride
        center_y, center_x = station_y // stride, station_x // stride
        for station in range(len(station_y)):
            result[level, station] = (
                (center_y[station] + radius + delta_y)
                * (level_width + 2 * radius)
                + center_x[station]
                + radius
                + delta_x
            )
    return torch.from_numpy(result)


class UpgradedNowcaster(nn.Module):
    """The established direct R2P route with a matched TIFF radar input."""

    def __init__(
        self,
        config: R2PModelConfig,
        station_y: np.ndarray,
        station_x: np.ndarray,
        station_positions_km: np.ndarray,
    ):
        super().__init__()
        self.config = config
        self.station_count = len(station_y)
        self.dimension = config.d_model
        multiple = 16
        self.pad_height = (multiple - config.grid_height % multiple) % multiple
        self.pad_width = (multiple - config.grid_width % multiple) % multiple
        padded_height = config.grid_height + self.pad_height
        padded_width = config.grid_width + self.pad_width

        self.encoder = SharedEncoder(config.base_channels, config.d_model)
        self.window_radius = config.readout_window // 2
        self.levels = (2, 8)
        indices = _window_indices(
            station_y,
            station_x,
            padded_height,
            padded_width,
            self.levels,
            self.window_radius,
        )
        self.register_buffer("window_indices", indices, persistent=False)
        fine_height, fine_width = padded_height // 2, padded_width // 2
        x_norm = (np.asarray(station_x) / 2.0) / (fine_width - 1) * 2.0 - 1.0
        y_norm = (np.asarray(station_y) / 2.0) / (fine_height - 1) * 2.0 - 1.0
        self.register_buffer(
            "station_norm",
            torch.tensor(np.stack([x_norm, y_norm], axis=1), dtype=torch.float32),
            persistent=False,
        )

        self.station_encoder = nn.Sequential(
            nn.Conv1d(config.station_feature_count, config.station_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.station_hidden, config.station_hidden, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.station_attention = StationAttention(
            config.station_hidden, station_positions_km, config.heads
        )
        self.global_projection = nn.Linear(config.d_model, config.d_model)
        self.deformable = DeformableUpstream(
            config.d_model, config.deform_points, config.deform_offset_scale
        )
        self.deform_cross = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.deform_norm = nn.LayerNorm(config.d_model)

        token_count = len(self.levels) * config.readout_window**2
        self.token_position = nn.Parameter(
            torch.randn(1, token_count, config.d_model) * 0.02
        )
        self.station_projection = nn.Linear(config.station_hidden, config.d_model)
        self.lead_embedding = nn.Parameter(
            torch.randn(1, config.point_leads, config.d_model) * 0.02
        )
        self.station_to_query = nn.Linear(config.station_hidden, config.d_model)
        self.query_norm = nn.LayerNorm(config.d_model)
        self.key_value_norm = nn.LayerNorm(config.d_model)
        self.cross = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(config.dropout)
        if config.cross_lead_layers > 0:
            layer = nn.TransformerEncoderLayer(
                config.d_model,
                config.heads,
                config.d_model * 2,
                config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cross_lead = nn.TransformerEncoder(
                layer, config.cross_lead_layers
            )
        else:
            self.cross_lead = None
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward_pre_cross(
        self, radar: torch.Tensor, station: torch.Tensor
    ) -> torch.Tensor:
        """Return lead-wise latent states immediately before cross-lead mixing."""

        if radar.ndim != 4 or station.ndim != 4:
            raise ValueError("radar and station inputs must be [B,T,H,W] and [B,T,V,F]")
        batch, radar_steps, height, width = radar.shape
        if radar_steps != self.config.radar_steps:
            raise ValueError(f"expected {self.config.radar_steps} radar steps, got {radar_steps}")
        if (height, width) != (self.config.grid_height, self.config.grid_width):
            raise ValueError("radar grid differs from the model contract")
        if station.shape[0] != batch or station.shape[2] != self.station_count:
            raise ValueError("station batch/count differs from the model contract")
        if station.shape[3] != self.config.station_feature_count:
            raise ValueError("station feature count differs from the model contract")

        padded = F.pad(radar, (0, self.pad_width, 0, self.pad_height))
        features = self.encoder(padded)
        local_tokens: list[torch.Tensor] = []
        radius = self.window_radius
        for level_index, stride in enumerate(self.levels):
            feature = F.pad(features[stride], (radius, radius, radius, radius))
            _, dimension, level_height, level_width = feature.shape
            gathered = feature.reshape(batch, dimension, level_height * level_width)[
                :, :, self.window_indices[level_index]
            ]
            local_tokens.append(gathered.permute(0, 2, 3, 1))
        window_tokens = torch.cat(local_tokens, dim=2) + self.token_position

        station_steps = station.shape[1]
        station_context = station.permute(0, 2, 3, 1).reshape(
            batch * self.station_count,
            self.config.station_feature_count,
            station_steps,
        )
        station_context = self.station_encoder(station_context).squeeze(-1)
        station_context = station_context.reshape(
            batch, self.station_count, self.config.station_hidden
        )
        station_context = self.station_attention(station_context)

        global_summary = self.global_projection(
            features[16].mean(dim=(2, 3))
        ).reshape(batch, 1, 1, self.dimension)
        query = (
            self.lead_embedding
            + self.station_to_query(station_context).reshape(
                batch * self.station_count, 1, self.dimension
            )
        ).reshape(
            batch,
            self.station_count,
            self.config.point_leads,
            self.dimension,
        ) + global_summary
        query_flat = self.query_norm(
            query.reshape(
                batch * self.station_count,
                self.config.point_leads,
                self.dimension,
            )
        )
        station_token = self.station_projection(station_context).reshape(
            batch * self.station_count, 1, self.dimension
        )
        key_value = torch.cat(
            [
                window_tokens.reshape(
                    batch * self.station_count,
                    window_tokens.shape[2],
                    self.dimension,
                ),
                station_token,
            ],
            dim=1,
        )
        key_value = self.key_value_norm(key_value)
        attended, _ = self.cross(
            query_flat, key_value, key_value, need_weights=True
        )
        output = query_flat + self.drop(attended)

        deform_tokens = self.deformable(features[2], self.station_norm, query)
        points = deform_tokens.shape[3]
        deform_key = self.deform_norm(
            deform_tokens.reshape(
                batch * self.station_count * self.config.point_leads,
                points,
                self.dimension,
            )
        )
        deform_query = output.reshape(
            batch * self.station_count * self.config.point_leads,
            1,
            self.dimension,
        )
        deform_attended, _ = self.deform_cross(
            deform_query, deform_key, deform_key, need_weights=True
        )
        output = (deform_query + self.drop(deform_attended)).reshape(
            batch * self.station_count,
            self.config.point_leads,
            self.dimension,
        )
        return output

    def forward(self, radar: torch.Tensor, station: torch.Tensor) -> torch.Tensor:
        batch = radar.shape[0]
        output = self.forward_pre_cross(radar, station)
        if self.cross_lead is not None:
            output = self.cross_lead(output)
        return self.head(output).squeeze(-1).reshape(
            batch, self.station_count, self.config.point_leads
        )


def load_query_expanded_state(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    """Strictly transplant all learned tensors into a new station query set."""

    missing, unexpected = model.load_state_dict(dict(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"learned checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )


__all__ = [
    "R2PModelConfig",
    "UpgradedNowcaster",
    "load_query_expanded_state",
]
