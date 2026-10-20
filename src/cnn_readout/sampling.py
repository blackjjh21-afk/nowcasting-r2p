"""Deterministic tuple-sampling recipes for spatial point readouts.

``target_stratified_v1`` guarantees deterministic coverage of five target
ranges. It changes only the composition of each epoch and never inspects
predictions or evaluation-period data. The final loss applies exact
natural-prevalence/realized-sampling correction, so the optimized population
objective remains unweighted mean squared error.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


UNIFORM_WITHOUT_REPLACEMENT = "uniform_without_replacement"
TARGET_STRATIFIED_V1 = "target_stratified_v1"
SAMPLING_RECIPES = (UNIFORM_WITHOUT_REPLACEMENT, TARGET_STRATIFIED_V1)

TARGET_STRATA_EDGES_MM = (1.0, 5.0, 10.0, 20.0)
# Mild redistribution relative to the observed full-2023 frequencies
# 92.91/4.84/1.31/0.71/0.23%.  The >=20-mm range is 2% rather than 0.23%; the
# loss applies exact p_s/q_s correction; these quotas affect coverage and
# estimator variance, not the population-risk weights.
TARGET_STRATIFIED_V1_FRACTIONS = (0.82, 0.10, 0.035, 0.025, 0.02)


def validate_sampling_recipe(value: str) -> str:
    if value not in SAMPLING_RECIPES:
        raise ValueError(
            f"unknown sampling recipe {value!r}; allowed={SAMPLING_RECIPES}"
        )
    return value


def sampling_contract(recipe: str) -> dict[str, object]:
    validate_sampling_recipe(recipe)
    if recipe == UNIFORM_WITHOUT_REPLACEMENT:
        return {
            "name": recipe,
            "eligible_tuples": "finite target >= 0 mm",
            "replacement": False,
        }
    return {
        "name": recipe,
        "eligible_tuples": "finite target >= 0 mm",
        "target_ranges_mm": ["<1", "1--<5", "5--<10", "10--<20", ">=20"],
        "epoch_fractions": list(TARGET_STRATIFIED_V1_FRACTIONS),
        "quota_rounding": (
            "largest remainder; deterministic lower-stratum-index tie break"
        ),
        "within_stratum_replacement": (
            "without replacement when quota <= available tuples; otherwise "
            "with replacement only in that stratum"
        ),
        "post_concatenation_shuffle": True,
        "target_only": True,
    }


def largest_remainder_quotas(
    sample_count: int,
    fractions: Sequence[float] = TARGET_STRATIFIED_V1_FRACTIONS,
) -> np.ndarray:
    """Allocate an exact integer epoch size using stable largest remainders."""

    count = int(sample_count)
    values = np.asarray(fractions, dtype=np.float64)
    if count <= 0:
        raise ValueError("sample_count must be positive")
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
        raise ValueError(f"invalid sampling fractions: {fractions}")
    if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"sampling fractions must sum to one: {values.sum()}")
    exact = values * count
    quotas = np.floor(exact).astype(np.int64)
    remainder = count - int(quotas.sum())
    if remainder:
        # mergesort preserves lower stratum indices for exact ties.
        order = np.argsort(-(exact - quotas), kind="mergesort")
        quotas[order[:remainder]] += 1
    if int(quotas.sum()) != count:
        raise RuntimeError("largest-remainder allocation changed epoch size")
    return quotas


def build_target_stratum_pools(
    truth_flat: np.ndarray,
    finite_flat: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Partition local flat tuple indices into the five fixed target ranges."""

    truth = np.asarray(truth_flat).reshape(-1)
    finite = np.asarray(finite_flat, dtype=np.int64)
    if finite.ndim != 1 or len(finite) == 0:
        raise ValueError("finite_flat must be a non-empty one-dimensional array")
    values = np.asarray(truth[finite], dtype=np.float32)
    if not np.all(np.isfinite(values) & (values >= 0)):
        raise ValueError("finite_flat includes an invalid target")
    strata = np.searchsorted(
        np.asarray(TARGET_STRATA_EDGES_MM, dtype=np.float32),
        values,
        side="right",
    )
    pools = tuple(
        np.ascontiguousarray(finite[strata == index], dtype=np.int64)
        for index in range(len(TARGET_STRATIFIED_V1_FRACTIONS))
    )
    if sum(len(pool) for pool in pools) != len(finite):
        raise RuntimeError("target stratification lost eligible tuples")
    if any(len(pool) == 0 for pool in pools):
        raise RuntimeError(
            "target_stratified_v1 requires at least one tuple in every stratum"
        )
    return pools


def sample_target_stratified(
    pools: Sequence[np.ndarray],
    sample_count: int,
    rng: np.random.Generator,
    fractions: Sequence[float] = TARGET_STRATIFIED_V1_FRACTIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw and shuffle one exact deterministic stratified epoch.

    Returns ``(flat_indices, quotas, replacement_flags)`` so callers can audit
    the exact composition and which strata required replacement.
    """

    quotas = largest_remainder_quotas(sample_count, fractions)
    if len(pools) != len(quotas):
        raise ValueError((len(pools), len(quotas)))
    selected: list[np.ndarray] = []
    replacement = np.zeros(len(quotas), dtype=bool)
    for index, (pool_value, quota_value) in enumerate(zip(pools, quotas, strict=True)):
        pool = np.asarray(pool_value, dtype=np.int64)
        quota = int(quota_value)
        if pool.ndim != 1 or len(pool) == 0:
            raise ValueError(f"stratum {index} pool is empty or not one-dimensional")
        replace = quota > len(pool)
        replacement[index] = replace
        selected.append(rng.choice(pool, size=quota, replace=replace))
    order = np.concatenate(selected)
    order = order[rng.permutation(len(order))]
    if len(order) != int(sample_count):
        raise RuntimeError("stratified sampler changed the requested epoch size")
    return np.ascontiguousarray(order, dtype=np.int64), quotas, replacement
