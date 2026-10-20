"""Trainable KMA adaptation aligned to the released NowcastNet forward model.

Architecture source: Zhang et al. (2023), doi:10.1038/s41586-023-06184-4,
Code Ocean doi:10.24433/CO.0832447.v1, inspected in the supplied official capsule
nowcastnet/capsule-3935105. Module names differ, but matching-geometry weights
can be explicitly mapped to the official forward network. The released capsule
does not contain the discriminator or full training pipeline.

Changes: configurable 7/18 time lengths and small CPU-test widths; device-neutral
execution and a discriminator reconstructed from Extended Data Fig. 1c,e.
The released capsule has no discriminator implementation. Official scaling is
preserved: raw rainfall enters evolution; raw history and evolution/rate_scale
enter generation; the decoder returns raw mm/hour without multiplying by 128.
The official unusual noise rearrangement is retained whenever its reshape is
defined, including 256/512 grids. A clearly separate shape-general fallback
supports tiny CPU-test geometries where that reshape cannot be performed.
Outputs are raw/unconstrained during training, like a linear radar decoder.
Inference callers must clamp each member to >=0 before averaging/accumulating.
The dataset caps input/target rain rates at 128 mm/h; the linear model output
itself has no upper clamp, log transform or RN60 output.

The evolution uses the paper's nearest advection for its forward trajectory,
bilinear advection for the motion-gradient auxiliary loss, and a detached
previous field at each time step. Generation detaches evolution conditioning.

MIT License (architecture adapted from released code):
Copyright (c) 2022 Yuchen Zhang
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to
do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


@dataclass(frozen=True)
class ModelConfig:
    input_steps: int = 7
    output_steps: int = 18
    base_channels: int = 32
    noise_channels: int = 32
    rate_scale: float = 128.0

    def __post_init__(self):
        for key in ("input_steps", "output_steps", "base_channels", "noise_channels"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.base_channels < 2 or self.rate_scale <= 0 or not torch.isfinite(torch.tensor(self.rate_scale)):
            raise ValueError("base_channels >=2 and finite positive rate_scale required")


def _conv(cin, cout, kernel=3, **kwargs):
    return spectral_norm(nn.Conv2d(cin, cout, kernel, **kwargs))


class ResidualConv(nn.Module):
    """Released two-convolution, preactivation residual block."""
    def __init__(self, cin, cout, mid=None):
        super().__init__()
        mid = cout if mid is None else mid
        self.main = nn.Sequential(nn.BatchNorm2d(cin), nn.ReLU(),
            _conv(cin, mid, padding=1), nn.BatchNorm2d(mid), nn.ReLU(), _conv(mid, cout, padding=1))
        self.skip = nn.Sequential(nn.BatchNorm2d(cin), _conv(cin, cout, padding=1))

    def forward(self, x):
        return self.main(x) + self.skip(x)


class Down(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.MaxPool2d(2), ResidualConv(cin, cout))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = ResidualConv(cin, cout, mid=cin // 2)

    def forward(self, value, skip):
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat((skip, value), dim=1))


class EvolutionNetwork(nn.Module):
    """Shared four-level encoder with separate intensity and motion decoders."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        c, t = config.base_channels, config.output_steps
        self.inc = ResidualConv(config.input_steps, c)
        self.down = nn.ModuleList([Down(c, 2*c), Down(2*c, 4*c), Down(4*c, 8*c), Down(8*c, 8*c)])
        self.intensity_up = nn.ModuleList([Up(16*c, 4*c), Up(8*c, 2*c), Up(4*c, c), Up(2*c, c)])
        self.motion_up = nn.ModuleList([Up(16*c, 4*c), Up(8*c, 2*c), Up(4*c, c), Up(2*c, c)])
        self.intensity_head = nn.Conv2d(c, t, 1)
        self.motion_head = nn.Conv2d(c, 2*t, 1)
        self.gamma = nn.Parameter(torch.zeros(1, t, 1, 1))

    def forward(self, history):
        features = [self.inc(history)]
        for block in self.down:
            features.append(block(features[-1]))
        intensity = motion = features[-1]
        for intensity_block, motion_block, skip in zip(self.intensity_up, self.motion_up, reversed(features[:-1])):
            intensity = intensity_block(intensity, skip)
            motion = motion_block(motion, skip)
        return self.intensity_head(intensity) * self.gamma, self.motion_head(motion)


def warp(inputs: Tensor, motion: Tensor, *, mode="bilinear") -> Tensor:
    """Backward semi-Lagrangian warp, motion=(departure dx,dy) in grid cells.

    Positive dx samples the source to the right. This is not an asserted
    physical eastward velocity. Border extension matches the released code.
    """
    if inputs.ndim != 4 or motion.shape != (len(inputs), 2, *inputs.shape[-2:]):
        raise ValueError("warp requires [B,C,H,W] and [B,2,H,W]")
    if mode not in ("nearest", "bilinear"):
        raise ValueError("Supported warp modes: nearest, bilinear")
    h, w = inputs.shape[-2:]
    # Float32 coordinates avoid coarse BF16 pixel positions on large grids.
    yy, xx = torch.meshgrid(torch.arange(h, device=inputs.device, dtype=torch.float32),
                           torch.arange(w, device=inputs.device, dtype=torch.float32), indexing="ij")
    x = 2 * (xx[None] + motion[:, 0].float()) / max(w - 1, 1) - 1
    y = 2 * (yy[None] + motion[:, 1].float()) / max(h - 1, 1) - 1
    return F.grid_sample(inputs.float(), torch.stack((x, y), dim=-1), mode=mode,
                         padding_mode="border", align_corners=True)


class NoiseProjectionBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.skip = OfficialSpectralNorm(nn.Conv2d(cin, cout - cin, 1))
        self.main = nn.Sequential(OfficialSpectralNorm(nn.Conv2d(cin, cout, 3, padding=1)), nn.ReLU(),
                                  OfficialSpectralNorm(nn.Conv2d(cout, cout, 3, padding=1)))

    def forward(self, x):
        return torch.cat((x, self.skip(x)), dim=1) + self.main(x)


class OfficialSpectralNorm(nn.Module):
    """Noise-projector SN from the release, including its eval-time update.

    The official u/v are non-trainable Parameters, not buffers. Detached clones
    in sigma preserve each draw's autograd operands when four forwards precede
    one backward. This does not change the released forward values or updates.
    """
    def __init__(self, module, name="weight", power_iterations=1):
        super().__init__()
        self.module, self.name, self.power_iterations = module, name, power_iterations
        weight = getattr(module, name)
        height = weight.shape[0]
        width = weight.reshape(height, -1).shape[1]
        u = nn.Parameter(weight.new_empty(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(weight.new_empty(width).normal_(0, 1), requires_grad=False)
        with torch.no_grad():
            u.div_(u.norm() + 1e-12)
            v.div_(v.norm() + 1e-12)
        weight_bar = nn.Parameter(weight.detach())
        del module._parameters[name]
        module.register_parameter(name + "_u", u)
        module.register_parameter(name + "_v", v)
        module.register_parameter(name + "_bar", weight_bar)

    def forward(self, *args, **kwargs):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        weight = getattr(self.module, self.name + "_bar")
        matrix = weight.reshape(weight.shape[0], -1)
        with torch.no_grad():
            for _ in range(self.power_iterations):
                next_v = torch.mv(matrix.detach().t(), u)
                next_v = next_v / (next_v.norm() + 1e-12)
                next_u = torch.mv(matrix.detach(), next_v)
                next_u = next_u / (next_u.norm() + 1e-12)
                v.copy_(next_v)
                u.copy_(next_u)
        sigma = u.detach().clone().dot(matrix.mv(v.detach().clone()))
        setattr(self.module, self.name, weight / sigma)
        return self.module(*args, **kwargs)


def rearrange_projected_noise(projected):
    """The release's reshape/permute, NOT PixelShuffle.

    At production 256/512 inputs the projector maps to 8x8/16x16 respectively;
    preserve the release's hard-coded inner 8x8 ordering for both. Use that same
    expression at smaller shapes whenever it is defined. A fallback handles
    tiny synthetic CPU-test widths for which the release's reshape is invalid.
    """
    batch, _, height, width = projected.shape
    if projected[0].numel() % (4 * 4 * 8 * 8) == 0:
        return (projected.reshape(batch, -1, 4, 4, 8, 8)
                .permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height * 4, width * 4))
    return (projected.reshape(batch, -1, 4, 4, height, width)
            .permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height * 4, width * 4))


class NoiseProjector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = OfficialSpectralNorm(nn.Conv2d(channels, 2*channels, 3, padding=1))
        self.blocks = nn.Sequential(*(NoiseProjectionBlock(a*channels, b*channels)
                                     for a, b in ((2, 4), (4, 8), (8, 16), (16, 32))))

    def forward(self, noise):
        return rearrange_projected_noise(self.blocks(self.first(noise)))


class SpatialAdaptiveNorm(nn.Module):
    def __init__(self, channels, output_steps, hidden=64):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.shared = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(output_steps, hidden, 3), nn.ReLU())
        self.gamma = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(hidden, channels, 3))
        self.beta = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(hidden, channels, 3))

    def forward(self, x, evolution):
        condition = F.adaptive_avg_pool2d(evolution, x.shape[-2:])
        hidden = self.shared(condition)
        return self.norm(x) * (1 + self.gamma(hidden)) + self.beta(hidden)


class GenerationBlock(nn.Module):
    def __init__(self, cin, cout, output_steps, hidden, double=True):
        super().__init__()
        self.double_conv = double
        mid = min(cin, cout)
        self.norm0 = SpatialAdaptiveNorm(cin, output_steps, hidden)
        self.conv0 = _conv(cin, mid, padding=0)
        # The release registers these even in its two single-convolution blocks.
        self.norm1 = SpatialAdaptiveNorm(mid, output_steps, hidden)
        self.conv1 = _conv(mid, cout, padding=0)
        self.short_norm = SpatialAdaptiveNorm(cin, output_steps, hidden) if cin != cout else None
        self.short_conv = _conv(cin, cout, kernel=1, bias=False) if cin != cout else None

    def forward(self, x, evolution):
        skip = x if self.short_conv is None else self.short_conv(self.short_norm(x, evolution))
        value = self.conv0(F.pad(F.leaky_relu(self.norm0(x, evolution), .2), (1, 1, 1, 1), mode="reflect"))
        if self.double_conv:
            value = self.conv1(F.pad(F.leaky_relu(self.norm1(value, evolution), .2), (1, 1, 1, 1), mode="reflect"))
        return skip + value


class GenerationEncoder(nn.Sequential):
    def __init__(self, config):
        c = config.base_channels
        super().__init__(ResidualConv(config.input_steps + config.output_steps, c),
                         Down(c, 2*c), Down(2*c, 4*c), Down(4*c, 8*c))


class GenerationDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        c, t = config.base_channels, config.output_steps
        hidden = 64
        self.fc = nn.Conv2d(8*c + 2*config.noise_channels, 8*c, 3, padding=1)
        self.head = GenerationBlock(8*c, 8*c, t, hidden, double=False)
        self.middle0 = GenerationBlock(8*c, 4*c, t, hidden)
        self.middle1 = GenerationBlock(4*c, 4*c, t, hidden)
        self.up0 = GenerationBlock(4*c, 2*c, t, hidden, double=False)
        self.up1 = GenerationBlock(2*c, c, t, hidden)
        self.up2 = GenerationBlock(c, c, t, hidden)
        self.out = nn.Conv2d(c, t, 3, padding=1)

    def forward(self, features, evolution):
        value = self.head(self.fc(features), evolution)
        value = self.middle0(F.interpolate(value, scale_factor=2, mode="nearest"), evolution)
        value = self.middle1(value, evolution)
        value = self.up0(F.interpolate(value, scale_factor=2, mode="nearest"), evolution)
        value = self.up1(F.interpolate(value, scale_factor=2, mode="nearest"), evolution)
        value = self.up2(value, evolution)
        return self.out(F.leaky_relu(value, .2))


class NowcastNet(nn.Module):
    """Complete stochastic generator conditioned on nonlinear evolution."""
    def __init__(self, config: ModelConfig | Mapping | None = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        if isinstance(config, Mapping):
            config = ModelConfig(**config)
        self.config = config
        self.evolution = EvolutionNetwork(config)
        self.encoder = GenerationEncoder(config)
        self.noise_projector = NoiseProjector(config.noise_channels)
        self.decoder = GenerationDecoder(config)

    def evolution_parameters(self):
        return self.evolution.parameters()

    def generation_parameters(self):
        return chain(self.encoder.parameters(), self.noise_projector.parameters(), self.decoder.parameters())

    @staticmethod
    def _check_geometry(height, width):
        if min(height, width) < 64 or height % 32 or width % 32:
            raise ValueError("Height/width must each be >=64 and divisible by32; pad and preserve masks upstream")

    def _check_history(self, history):
        if history.ndim != 4 or history.shape[1] != self.config.input_steps or not history.is_floating_point():
            raise ValueError("History must be floating [B,input_steps,H,W] in mm/hour")
        self._check_geometry(*history.shape[-2:])

    def sample_noise(self, batch, height, width, device=None, generator=None):
        self._check_geometry(height, width)
        if batch < 1:
            raise ValueError("Positive batch required")
        return torch.randn((batch, self.config.noise_channels, height // 32, width // 32),
                           device=device, generator=generator)

    def evolve(self, history_rate: Tensor):
        self._check_history(history_rate)
        b, _, h, w = history_rate.shape
        intensity, motion = self.evolution(history_rate)
        intensity = intensity.float()
        motion = motion.float().reshape(b, self.config.output_steps, 2, h, w)
        previous = history_rate[:, -1:].float()
        predictions, advected = [], []
        for step in range(self.config.output_steps):
            # Distinct forward/gradient paths are described in Methods Eq.4:
            # nearest forward trajectory; bilinear auxiliary for flow gradients.
            previous = previous.detach()
            advected.append(warp(previous, motion[:, step], mode="bilinear"))
            previous = warp(previous, motion[:, step], mode="nearest") + intensity[:, step:step+1]
            predictions.append(previous)
        return dict(prediction=torch.cat(predictions, dim=1), motion=motion,
                    advected=torch.cat(advected, dim=1), intensity=intensity)

    def generate(self, history_rate: Tensor, evolution_dict_or_tensor, noise: Tensor | None = None):
        self._check_history(history_rate)
        evolution = evolution_dict_or_tensor["prediction"] if isinstance(evolution_dict_or_tensor, Mapping) else evolution_dict_or_tensor
        b, _, h, w = history_rate.shape
        if evolution.shape != (b, self.config.output_steps, h, w):
            raise ValueError("Evolution shape differs from configured future sequence")
        if noise is None:
            noise = self.sample_noise(b, h, w, history_rate.device)
        if noise.shape != (b, self.config.noise_channels, h // 32, w // 32) or noise.device != history_rate.device:
            raise ValueError("Explicit noise shape/device differs from model/history")
        condition = evolution.detach() / self.config.rate_scale
        encoded = self.encoder(torch.cat((history_rate, condition), dim=1))
        noise_features = self.noise_projector(noise.to(dtype=history_rate.dtype))
        return self.decoder(torch.cat((encoded, noise_features), dim=1), condition).float()

    def forward(self, history_rate, noise=None):
        return self.generate(history_rate, self.evolve(history_rate), noise)


class DiscriminatorLBlock(nn.Module):
    """Fig. 1e: BN--conv skip plus BN--ReLU--conv residual branch.

    Stride is applied in both 3x3 convolutions to obtain the spatial sizes in
    Fig. 1c. The diagram does not separately specify this placement or padding.
    D-only BN uses batch statistics in both optimizer phases. Keeping running
    statistics would switch normalization when D is frozen/eval for G's step.
    This is a training-loop consistency choice, not an official-code setting.
    """
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.skip = nn.Sequential(nn.BatchNorm2d(cin, track_running_stats=False),
                                  _conv(cin, cout, kernel=3, stride=stride, padding=1))
        self.main = nn.Sequential(nn.BatchNorm2d(cin, track_running_stats=False), nn.ReLU(),
                                  _conv(cin, cout, kernel=3, stride=stride, padding=1))

    def forward(self, value):
        return self.skip(value) + self.main(value)


class Discriminator(nn.Module):
    """Reconstruct Extended Data Fig. 1c,e, not released training-code parity.

    Concatenate history and future as shown in the diagram, then combine one
    2D branch with two temporally valid 3D branches. Flatten each 3D output's
    channel/time axes WITHOUT averaging time. Four BN residual L-blocks and a
    BN--LeakyReLU--3x3 head return patch logits [B,1,H/16,W/16], no sigmoid.

    At the diagram's 4 history/20 future frames and base width 32, the first
    branches contribute 64 + 4*21 + 8*5 = 188 channels. Its long branch's kernel
    label says 4, inconsistent with its displayed 5 output timesteps; a valid
    temporal kernel of 20 matches those dimensions. We explicitly infer that
    kernel as out_steps (18 for KMA), not as a recovered official code value.
    Spatial padding (4 for 9x9; 1 for 3x3) and L-block convolution strides are
    inferred to realize the displayed spatial sizes. Tiny CPU-test widths use
    at least one channel per temporal branch.

    Input is raw mm/hour by default; it is NOT divided by the generator's 128
    condition scale. Mask real/fake identically upstream. No mask is inferred
    by the discriminator. Spectral normalization uses legacy torch hooks.
    All discriminator BN layers are constructed without running statistics so
    train/eval mode does not change their normalization rule. The discriminator
    remains batch-dependent; evolution/generator BN is not changed.
    """
    def __init__(self, in_steps=7, out_steps=18, base_channels=32, input_scale=1.0):
        super().__init__()
        for value in (in_steps, out_steps, base_channels):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Positive integer discriminator dimensions required")
        if input_scale <= 0 or not torch.isfinite(torch.tensor(input_scale)):
            raise ValueError("Finite positive discriminator input scale required")
        self.in_steps, self.out_steps, self.input_scale = in_steps, out_steps, input_scale
        length, c = in_steps + out_steps, base_channels
        short_channels, long_channels = max(1, c // 8), max(1, c // 4)
        short_kernel, long_kernel = min(4, length), out_steps
        self.spatial_front = _conv(length, 2*c, kernel=9, stride=2, padding=4)
        self.temporal = nn.ModuleList([
            spectral_norm(nn.Conv3d(1, channels, (kernel, 9, 9),
                                    stride=(1, 2, 2), padding=(0, 4, 4)))
            for channels, kernel in ((short_channels, short_kernel), (long_channels, long_kernel))])
        self.front_channels = (2*c + short_channels*(length-short_kernel+1)
                               + long_channels*(length-long_kernel+1))
        self.blocks = nn.Sequential(
            DiscriminatorLBlock(self.front_channels, 4*c, stride=2),
            DiscriminatorLBlock(4*c, 8*c, stride=2),
            DiscriminatorLBlock(8*c, 16*c, stride=2),
            DiscriminatorLBlock(16*c, 16*c, stride=1))
        self.head = nn.Sequential(nn.BatchNorm2d(16*c, track_running_stats=False), nn.LeakyReLU(.2),
                                  _conv(16*c, 1, kernel=3, padding=1))

    @staticmethod
    def flatten_time(features):
        """[B,C,T,H,W] -> [B,C*T,H,W], retaining each temporal position."""
        if features.ndim != 5:
            raise ValueError("Temporal features require [B,C,T,H,W]")
        return features.flatten(start_dim=1, end_dim=2)

    def front_features(self, history, future):
        if history.ndim != 4 or future.ndim != 4 or history.shape[1] != self.in_steps or future.shape[1] != self.out_steps:
            raise ValueError("Discriminator requires configured history/future [B,T,H,W]")
        if history.shape[0] != future.shape[0] or history.shape[-2:] != future.shape[-2:]:
            raise ValueError("History/future geometry mismatch")
        value = torch.cat((history, future), dim=1) / self.input_scale
        features = [self.spatial_front(value)]
        features.extend(self.flatten_time(layer(value.unsqueeze(1))) for layer in self.temporal)
        return torch.cat(features, dim=1)

    def forward(self, history, future):
        return self.head(self.blocks(self.front_features(history, future)))
