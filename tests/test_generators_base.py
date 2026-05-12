"""Tests for ``groundfield.generators.base``.

Validates the abstract framework: ``cfg.sample`` traversal,
``has_distributions`` introspection, ``Generator.build`` guard,
JSON round-trip on configs with mixed fixed-and-distribution fields,
and RNG wiring.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pytest
from pydantic import Field

from groundfield.generators.base import (
    GeneratorConfig,
    WorldGenerator,
    resolve_value,
)
from groundfield.generators.distributions import (
    AnyDistribution,
    Categorical,
    Constant,
    Discrete,
    LogNormal,
    Normal,
    Uniform,
)


# ---------------------------------------------------------------------
# Fixtures: a minimal toy generator + config
# ---------------------------------------------------------------------


class _Inner(GeneratorConfig):
    inner_value: Union[float, AnyDistribution] = 1.0


class _ToyConfig(GeneratorConfig):
    a: Union[float, AnyDistribution] = 1.0
    b: Union[float, AnyDistribution] = 2.0
    label: Union[str, Categorical] = "x"
    fixed_int: int = 5
    inner: _Inner = Field(default_factory=_Inner)


class _ToyGenerator(WorldGenerator[_ToyConfig]):
    """Tiny generator that doesn't actually build a World — returns
    a sentinel string so we can probe the framework. ``sample_world``
    pairs this sentinel with the resolved config automatically."""

    def build(self, cfg=None):  # type: ignore[override]
        cfg = cfg or self.cfg
        self._assert_resolved(cfg)
        return "ok"


# ---------------------------------------------------------------------
# resolve_value
# ---------------------------------------------------------------------


def test_resolve_value_passes_through_constants() -> None:
    rng = np.random.default_rng(0)
    assert resolve_value(3.14, rng) == 3.14
    assert resolve_value(7, rng) == 7
    assert resolve_value("hello", rng) == "hello"


def test_resolve_value_samples_distributions() -> None:
    rng = np.random.default_rng(0)
    out = resolve_value(Constant(value=42.0), rng)
    assert out == 42.0


# ---------------------------------------------------------------------
# has_distributions
# ---------------------------------------------------------------------


def test_has_distributions_false_on_fully_fixed() -> None:
    cfg = _ToyConfig()
    assert cfg.has_distributions() is False


def test_has_distributions_true_on_top_level_field() -> None:
    cfg = _ToyConfig(a=Uniform(low=0.0, high=1.0))
    assert cfg.has_distributions() is True


def test_has_distributions_true_on_nested_field() -> None:
    cfg = _ToyConfig(inner=_Inner(inner_value=Normal(mean=0.0, std=1.0)))
    assert cfg.has_distributions() is True


def test_has_distributions_true_on_categorical() -> None:
    cfg = _ToyConfig(label=Categorical(values=["a", "b"]))
    assert cfg.has_distributions() is True


# ---------------------------------------------------------------------
# cfg.sample()
# ---------------------------------------------------------------------


def test_sample_resolves_top_level_distribution() -> None:
    cfg = _ToyConfig(a=Constant(value=99.0))
    resolved = cfg.sample(0)
    assert resolved.a == 99.0
    assert resolved.has_distributions() is False


def test_sample_resolves_nested_distribution() -> None:
    cfg = _ToyConfig(inner=_Inner(inner_value=Constant(value=7.0)))
    resolved = cfg.sample(0)
    assert resolved.inner.inner_value == 7.0
    assert resolved.has_distributions() is False


def test_sample_resolves_categorical_to_string() -> None:
    cfg = _ToyConfig(label=Categorical(values=["foundation"]))
    resolved = cfg.sample(0)
    assert resolved.label == "foundation"


def test_sample_is_idempotent_when_no_distributions() -> None:
    cfg = _ToyConfig()
    resolved = cfg.sample(0)
    # When nothing changes the same instance is returned (model_copy short-circuit)
    assert resolved is cfg


def test_sample_is_reproducible_under_seed() -> None:
    cfg = _ToyConfig(
        a=Uniform(low=0.0, high=1.0),
        b=Normal(mean=10.0, std=1.0),
        inner=_Inner(inner_value=LogNormal(mu=0.0, sigma=0.5)),
    )
    r1 = cfg.sample(seed_or_rng := 42)
    r2 = cfg.sample(seed_or_rng)
    assert r1.a == r2.a
    assert r1.b == r2.b
    assert r1.inner.inner_value == r2.inner.inner_value


# ---------------------------------------------------------------------
# Generator-side guard
# ---------------------------------------------------------------------


def test_build_raises_on_unresolved_config() -> None:
    cfg = _ToyConfig(a=Uniform(low=0.0, high=1.0))
    with pytest.raises(ValueError, match="still carries Distribution"):
        _ToyGenerator(cfg).build()


def test_build_succeeds_after_sample() -> None:
    cfg = _ToyConfig(a=Uniform(low=0.0, high=1.0))
    gen = _ToyGenerator(cfg, seed=0)
    out, resolved = gen.sample_world()
    assert out == "ok"
    assert isinstance(resolved.a, float)
    assert 0.0 <= resolved.a < 1.0


def test_sample_world_uses_internal_rng_by_default() -> None:
    cfg = _ToyConfig(a=Uniform(low=0.0, high=1.0))
    gen_a = _ToyGenerator(cfg, seed=42)
    gen_b = _ToyGenerator(cfg, seed=42)
    _, ra = gen_a.sample_world()
    _, rb = gen_b.sample_world()
    assert ra.a == rb.a


def test_sample_world_explicit_rng_overrides_internal() -> None:
    cfg = _ToyConfig(a=Uniform(low=0.0, high=1.0))
    gen = _ToyGenerator(cfg, seed=0)
    _, r1 = gen.sample_world(rng=42)
    _, r2 = gen.sample_world(rng=42)
    assert r1.a == r2.a


# ---------------------------------------------------------------------
# JSON round-trip on a mixed config
# ---------------------------------------------------------------------


def test_config_json_roundtrip_with_distributions() -> None:
    cfg = _ToyConfig(
        a=Uniform(low=0.0, high=1.0),
        b=Discrete(values=[1.0, 2.0, 3.0]),
        label=Categorical(values=["x", "y"]),
        inner=_Inner(inner_value=Normal(mean=0.0, std=1.0)),
    )
    payload = cfg.model_dump_json()
    restored = _ToyConfig.model_validate_json(payload)
    assert isinstance(restored.a, Uniform)
    assert isinstance(restored.b, Discrete)
    assert isinstance(restored.label, Categorical)
    assert isinstance(restored.inner.inner_value, Normal)
