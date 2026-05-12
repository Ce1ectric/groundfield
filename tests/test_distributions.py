"""Tests for ``groundfield.generators.distributions``.

Validation programme of ADR-0009:

1. ``.sample(rng)`` reproducibility under fixed seed.
2. statistical sanity: 10 000-sample mean / std within 5 %
   of the analytic value.
3. JSON round-trip via Pydantic discriminated union.
4. Bound enforcement on truncated Normal.
5. Validation of malformed inputs (negative weights, mismatched
   lengths, non-positive scale, ...).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from groundfield.generators.distributions import (
    AnyDistribution,
    Categorical,
    Constant,
    Discrete,
    Distribution,
    LogNormal,
    Normal,
    Uniform,
    Weibull,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class _Wrapper(BaseModel):
    """Helper model used to drive the discriminated-union round-trip."""

    dist: AnyDistribution


def _samples(d: Distribution, n: int = 10_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray([d.sample(rng) for _ in range(n)])


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist",
    [
        Constant(value=3.14),
        Uniform(low=0.0, high=2.0),
        Normal(mean=10.0, std=2.0),
        LogNormal(mu=1.0, sigma=0.5),
        Weibull(shape=2.0, scale=3.0),
        Discrete(values=[5, 10, 30, 80, 200]),
        Categorical(values=["foundation", "rod", "mesh"]),
    ],
)
def test_sample_is_reproducible_under_seed(dist) -> None:
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    seq_a = [dist.sample(rng_a) for _ in range(50)]
    seq_b = [dist.sample(rng_b) for _ in range(50)]
    assert seq_a == seq_b, f"{type(dist).__name__} not reproducible under seed"


# ---------------------------------------------------------------------
# Statistical sanity
# ---------------------------------------------------------------------


def test_uniform_mean_and_std() -> None:
    d = Uniform(low=0.0, high=4.0)
    s = _samples(d)
    assert abs(s.mean() - 2.0) < 0.05
    # Var = (high - low)^2 / 12 = 16/12 ≈ 1.333; std ≈ 1.155
    assert abs(s.std() - math.sqrt(16.0 / 12.0)) < 0.05


def test_normal_mean_and_std() -> None:
    d = Normal(mean=10.0, std=2.0)
    s = _samples(d)
    assert abs(s.mean() - 10.0) < 0.05
    assert abs(s.std() - 2.0) < 0.1


def test_normal_truncation_respects_bounds() -> None:
    d = Normal(mean=0.0, std=1.0, truncate_low=-1.0, truncate_high=1.0)
    s = _samples(d, n=2000, seed=7)
    assert s.min() >= -1.0 - 1e-12
    assert s.max() <= 1.0 + 1e-12


def test_lognormal_mean_matches_analytic() -> None:
    d = LogNormal(mu=1.0, sigma=0.5)
    s = _samples(d)
    # E[X] = exp(mu + sigma^2/2)
    expected = math.exp(1.0 + 0.5**2 / 2.0)
    assert abs(s.mean() - expected) / expected < 0.05


def test_lognormal_from_moments_recovers_target() -> None:
    target_mean, target_std = 100.0, 50.0
    d = LogNormal.from_moments(mean=target_mean, std=target_std)
    s = _samples(d, n=20_000, seed=0)
    assert abs(s.mean() - target_mean) / target_mean < 0.05
    assert abs(s.std() - target_std) / target_std < 0.1


def test_weibull_mean_matches_analytic() -> None:
    shape, scale = 2.0, 3.0
    d = Weibull(shape=shape, scale=scale)
    s = _samples(d)
    # E[X] = scale * Gamma(1 + 1/shape)
    expected = scale * math.gamma(1.0 + 1.0 / shape)
    assert abs(s.mean() - expected) / expected < 0.05


def test_discrete_uniform_frequencies() -> None:
    values = [5, 10, 30, 80, 200]
    d = Discrete(values=values)
    s = _samples(d)
    counts = {v: int(np.sum(s == v)) for v in values}
    expected = len(s) / len(values)
    for v, c in counts.items():
        assert abs(c - expected) / expected < 0.05, f"value {v} biased"


def test_discrete_weighted_frequencies() -> None:
    values = [1, 2, 3]
    weights = [0.7, 0.2, 0.1]
    d = Discrete(values=values, weights=weights)
    s = _samples(d, n=20_000)
    n = len(s)
    for v, w in zip(values, weights):
        c = int(np.sum(s == v))
        assert abs(c / n - w) < 0.02, f"weight bias for {v}: got {c/n}, expect {w}"


def test_categorical_uniform_frequencies() -> None:
    d = Categorical(values=["foundation", "rod", "mesh"])
    s = [d.sample(np.random.default_rng(seed)) for seed in range(3000)]
    counts = {v: s.count(v) for v in d.values}
    expected = len(s) / len(d.values)
    for v, c in counts.items():
        assert abs(c - expected) / expected < 0.10


# ---------------------------------------------------------------------
# JSON round-trip via discriminated union
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist",
    [
        Constant(value=42.0),
        Uniform(low=-1.0, high=1.0),
        Normal(mean=0.0, std=2.0, truncate_low=-3.0, truncate_high=3.0),
        LogNormal(mu=0.0, sigma=1.0),
        Weibull(shape=1.5, scale=4.0),
        Discrete(values=[5, 10, 30], weights=[0.5, 0.3, 0.2]),
        Categorical(values=["a", "b"], weights=[0.8, 0.2]),
    ],
)
def test_distribution_json_roundtrip(dist) -> None:
    wrapped = _Wrapper(dist=dist)
    payload = wrapped.model_dump_json()
    restored = _Wrapper.model_validate_json(payload)
    assert type(restored.dist) is type(dist), (
        f"discriminator dispatch failed for {type(dist).__name__}: "
        f"got {type(restored.dist).__name__}"
    )
    # Same fields
    assert restored.dist.model_dump() == dist.model_dump()


def test_json_payload_carries_kind_discriminator() -> None:
    n = Normal(mean=1.0, std=1.0)
    payload = json.loads(_Wrapper(dist=n).model_dump_json())
    assert payload["dist"]["kind"] == "normal"


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def test_uniform_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError):
        Uniform(low=2.0, high=1.0)


def test_normal_rejects_non_positive_std() -> None:
    with pytest.raises(ValidationError):
        Normal(mean=0.0, std=0.0)


def test_normal_rejects_inverted_truncation() -> None:
    with pytest.raises(ValidationError):
        Normal(mean=0.0, std=1.0, truncate_low=2.0, truncate_high=-2.0)


def test_weibull_rejects_non_positive_shape() -> None:
    with pytest.raises(ValidationError):
        Weibull(shape=0.0, scale=1.0)


def test_discrete_rejects_negative_weights() -> None:
    with pytest.raises(ValidationError):
        Discrete(values=[1, 2], weights=[-0.1, 1.0])


def test_discrete_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError):
        Discrete(values=[1, 2, 3], weights=[0.5, 0.5])


def test_categorical_rejects_duplicate_values() -> None:
    with pytest.raises(ValidationError):
        Categorical(values=["a", "a"])


def test_normal_truncation_exhaustion_raises() -> None:
    """truncate band so far from the mean that 1000 attempts can't hit it."""
    d = Normal(mean=0.0, std=1.0, truncate_low=100.0, truncate_high=110.0,
               max_attempts=50)
    with pytest.raises(RuntimeError, match="rejection sampling exhausted"):
        d.sample(np.random.default_rng(0))
