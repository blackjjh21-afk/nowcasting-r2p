"""Natural-prevalence objective used for final CNN fitting."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


TARGET_RANGE_EDGES_MM = (1.0, 5.0, 10.0, 20.0)


def importance_corrected_mse(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    target_mm: torch.Tensor,
    *,
    natural_fractions: Sequence[float] | torch.Tensor,
    sampled_fractions: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Estimate unweighted normalized-RN60 MSE after stratified sampling.

    Each sampled tuple receives only ``p_s/q_s`` importance correction. The
    five rain ranges all have coefficient one, so the population objective is
    ordinary MSE on normalized RN60; stratification changes coverage and
    variance but not the target risk.
    """

    if prediction_norm.shape != target_norm.shape or target_mm.shape != target_norm.shape:
        raise ValueError("prediction, normalized target, and millimetre target shapes differ")
    per_tuple = F.mse_loss(prediction_norm, target_norm, reduction="none")
    natural = torch.as_tensor(
        natural_fractions, dtype=per_tuple.dtype, device=per_tuple.device
    )
    sampled = torch.as_tensor(
        sampled_fractions, dtype=per_tuple.dtype, device=per_tuple.device
    )
    if tuple(natural.shape) != (5,) or tuple(sampled.shape) != (5,):
        raise ValueError("natural_fractions and sampled_fractions must have length five")
    if torch.any(natural <= 0) or torch.any(sampled <= 0):
        raise ValueError("stratum fractions must be strictly positive")
    one = torch.ones((), dtype=per_tuple.dtype, device=per_tuple.device)
    tolerance = 32 * torch.finfo(per_tuple.dtype).eps
    if not torch.isclose(natural.sum(), one, rtol=0.0, atol=tolerance):
        raise ValueError("natural_fractions do not sum to one")
    if not torch.isclose(sampled.sum(), one, rtol=0.0, atol=tolerance):
        raise ValueError("sampled_fractions do not sum to one")
    boundaries = torch.as_tensor(
        TARGET_RANGE_EDGES_MM, dtype=target_mm.dtype, device=target_mm.device
    )
    stratum = torch.bucketize(target_mm.contiguous(), boundaries, right=True)
    return torch.mean((natural[stratum] / sampled[stratum]) * per_tuple)


def normalize_target(target_mm: torch.Tensor) -> torch.Tensor:
    """Use the frozen fitting-station RN60 scale also used by Direct R2P."""
    return target_mm / 735.0


def millimetres_from_normalized_prediction(prediction: torch.Tensor) -> torch.Tensor:
    """Invert the fixed scale without upper clipping."""
    return torch.clamp_min(prediction * 735.0, 0.0)
