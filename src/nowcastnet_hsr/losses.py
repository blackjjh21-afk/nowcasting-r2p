"""Masked paper-guided NowcastNet objectives, all rain rates in mm/hour.

Equations 4–12 in Zhang et al. (2023), doi:10.1038/s41586-023-06184-4.
The released package contains inference, not the original training program.
This implementation uses means over valid pixels/times instead of unnormalised
sums so grid size or masked domain does not implicitly change loss scaling.
Weights min(24,1+target) cap LOSS WEIGHTS only. Dataset input/target rain rates
are independently capped at 128 mm/h before reaching these losses.
Pool loss is max_pool each stochastic member THEN ensemble mean (not reverse).
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _valid_target(target, mask):
    if target.ndim != 4 or not target.is_floating_point():
        raise ValueError("Targets must be floating [B,T,H,W]")
    valid = torch.isfinite(target) & (target >= 0)
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[:, None]
        try:
            valid = valid & torch.broadcast_to(mask.bool(), target.shape)
        except RuntimeError as exc:
            raise ValueError("Mask cannot broadcast to target") from exc
    if not bool(valid.any()):
        raise ValueError("No valid target pixels")
    return valid, torch.where(valid, target.float(), 0)


def _safe_prediction(prediction, valid):
    if prediction.shape != valid.shape:
        raise ValueError("Prediction and target shape differ")
    if bool((valid & ~torch.isfinite(prediction)).any()):
        raise ValueError("Nonfinite prediction at a valid target")
    return torch.where(valid, prediction.float(), 0)


def weighted_l1(prediction: Tensor, target: Tensor, mask: Tensor | None = None):
    valid, observed = _valid_target(target, mask)
    prediction = _safe_prediction(prediction, valid)
    weights = (observed + 1).clamp(max=24)
    return (abs(prediction - observed) * weights * valid).sum() / valid.sum()


def motion_regularization(motion: Tensor, target: Tensor, mask: Tensor | None = None):
    valid, observed = _valid_target(target, mask)
    if motion.shape != (target.shape[0], target.shape[1], 2, *target.shape[-2:]):
        raise ValueError("Motion must have [B,T,2,H,W] shape")
    if not bool(torch.isfinite(motion).all()):
        raise ValueError("Nonfinite motion")
    b, t, _, h, w = motion.shape
    field = motion.float().reshape(b*t*2, 1, h, w)
    sobel_x = field.new_tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])[None, None]
    sobel_y = sobel_x.transpose(-1, -2)
    field = F.pad(field, (1, 1, 1, 1), mode="replicate")
    dx = F.conv2d(field, sobel_x).reshape(b, t, 2, h, w)
    dy = F.conv2d(field, sobel_y).reshape(b, t, 2, h, w)
    # A regularizer cell needs valid surrounding target support. Replication
    # does not introduce an artificial gradient for constant motion at edges.
    support = F.avg_pool2d(F.pad(valid.float().reshape(b*t, 1, h, w), (1, 1, 1, 1), mode="replicate"), 3, 1)
    support = support.reshape(b, t, h, w) >= (1 - 1e-6)
    norm = (dx.square() + dy.square()).sum(dim=2)
    weights = (observed + 1).clamp(max=24)
    return (norm * weights * support).sum() / support.sum().clamp_min(1)


def evolution_loss(prediction: Tensor, target: Tensor, motion: Tensor, mask: Tensor | None = None,
                   *, advected: Tensor | None = None, motion_weight: float = .01):
    if advected is None:
        raise ValueError("Bilinear advected auxiliary is required for the motion learning objective")
    if motion_weight < 0:
        raise ValueError("motion_weight must be nonnegative")
    accumulation = weighted_l1(prediction, target, mask) + weighted_l1(advected, target, mask)
    regularization = motion_regularization(motion, target, mask)
    return dict(total=accumulation + motion_weight*regularization,
                accumulation=accumulation, motion=regularization)


def pool_regularization(ensemble_samples: Tensor, target: Tensor, mask: Tensor | None = None,
                        *, kernel_size=5, stride=2):
    if ensemble_samples.ndim != 5 or ensemble_samples.shape[1:] != target.shape or len(ensemble_samples) < 1:
        raise ValueError("Samples must be nonempty [K,B,T,H,W], matching target [B,T,H,W]")
    if min(target.shape[-2:]) < kernel_size:
        raise ValueError("Spatial size is smaller than pooling kernel")
    valid, observed = _valid_target(target, mask)
    samples = torch.stack([_safe_prediction(value, valid) for value in ensemble_samples])
    k, b, t, h, w = samples.shape
    pooled_prediction = F.max_pool2d(samples.reshape(k*b*t, 1, h, w), kernel_size, stride)
    ph, pw = pooled_prediction.shape[-2:]
    pooled_prediction = pooled_prediction.reshape(k, b, t, ph, pw).mean(dim=0)
    pooled_target = F.max_pool2d(observed.reshape(b*t, 1, h, w), kernel_size, stride).reshape(b, t, ph, pw)
    # No score is assigned to a pool touching a missing target; never fill
    # unknown pixels with zero rainfall and treat that as a valid observation.
    pooled_valid = F.avg_pool2d(valid.float().reshape(b*t, 1, h, w), kernel_size, stride)
    pooled_valid = pooled_valid.reshape(b, t, ph, pw) >= (1 - 1e-6)
    if not bool(pooled_valid.any()):
        raise ValueError("No fully valid pooling windows")
    return weighted_l1(pooled_prediction, pooled_target, pooled_valid)


def generator_loss(ensemble_samples: Tensor, target: Tensor, discriminator_logits: Tensor,
                   mask: Tensor | None = None, *, adversarial_weight=6., pool_weight=20.):
    if adversarial_weight < 0 or pool_weight < 0:
        raise ValueError("Loss weights must be nonnegative")
    if not bool(torch.isfinite(discriminator_logits).all()):
        raise ValueError("Nonfinite discriminator logits")
    adversarial = F.binary_cross_entropy_with_logits(discriminator_logits.float(), torch.ones_like(discriminator_logits, dtype=torch.float32))
    pooled = pool_regularization(ensemble_samples, target, mask)
    return dict(total=adversarial_weight*adversarial + pool_weight*pooled,
                adversarial=adversarial, pool=pooled)


def discriminator_loss(real_logits: Tensor, fake_logits: Tensor):
    if not bool(torch.isfinite(real_logits).all() & torch.isfinite(fake_logits).all()):
        raise ValueError("Nonfinite discriminator logits")
    real = F.binary_cross_entropy_with_logits(real_logits.float(), torch.ones_like(real_logits, dtype=torch.float32))
    fake = F.binary_cross_entropy_with_logits(fake_logits.float(), torch.zeros_like(fake_logits, dtype=torch.float32))
    return dict(total=real + fake, real=real, fake=fake)
