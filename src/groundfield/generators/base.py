"""Base framework for ``World`` generators.

Every concrete generator (e.g. ``TnNetworkGenerator``) ships its own
:class:`GeneratorConfig` subclass and a thin
:class:`WorldGenerator` subclass. The common machinery — JSON
serialisation, distribution-aware sampling, RNG wiring — lives here.

Architecture summary (see ADR-0009 for the full design)
-------------------------------------------------------
* :class:`GeneratorConfig` is a Pydantic v2 model whose numerical
  fields may carry **either** a fixed value **or** a
  :class:`~groundfield.generators.distributions.Distribution`. The
  method :meth:`GeneratorConfig.sample` traverses the model and
  resolves every distribution to a concrete value, returning a
  *resolved* config of the same type.
* :class:`WorldGenerator` takes a :class:`GeneratorConfig`, optionally
  resolves it, and constructs a :class:`groundfield.World` via the
  abstract :meth:`WorldGenerator.build` method. Concrete generators
  override :meth:`build` only.
* :func:`resolve_value` is the workhorse utility that turns a
  ``T | Distribution`` field into a ``T`` either by passing fixed
  values through or by sampling the distribution.

Reproducibility
---------------
``WorldGenerator`` accepts an integer ``seed`` or an existing
:class:`numpy.random.Generator` at construction time and exposes its
RNG via :attr:`WorldGenerator.rng`. Explicit ``rng=`` arguments to
:meth:`sample_world` always take precedence.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, Optional, TypeVar, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from groundfield.generators.distributions import (
    Categorical,
    Distribution,
)
from groundfield.world import World

__all__ = [
    "GeneratorConfig",
    "WorldGenerator",
    "resolve_value",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def resolve_value(value: Any, rng: np.random.Generator) -> Any:
    """Resolve a ``T | Distribution`` field to a concrete value.

    Parameters
    ----------
    value
        Either a fixed Python value (passed through unchanged) or a
        :class:`Distribution` instance (sampled via
        :meth:`Distribution.sample`).
    rng
        Random generator forwarded to ``Distribution.sample``.

    Returns
    -------
    object
        The fixed value, or one sample from the distribution.
    """
    if isinstance(value, Distribution):
        return value.sample(rng)
    return value


def _coerce_rng(
    rng: Optional[Union[int, np.random.Generator]],
) -> np.random.Generator:
    """Turn ``int | Generator | None`` into an actual ``np.random.Generator``."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    raise TypeError(
        f"rng must be int, numpy.random.Generator, or None, got {type(rng).__name__}."
    )


# ---------------------------------------------------------------------
# GeneratorConfig — Pydantic-v2 base for every generator's configuration
# ---------------------------------------------------------------------


class GeneratorConfig(BaseModel):
    """Base class for generator configurations.

    Subclasses declare their parameters as Pydantic fields. Numerical
    fields are typed as ``T | <distribution alias>`` so the user can
    pass a fixed value or a :class:`Distribution`. Categorical
    fields use ``str | Categorical``.

    Two introspection helpers are provided:

    * :meth:`has_distributions` returns ``True`` if any field still
      carries a :class:`Distribution` instance — useful as a guard
      before passing the config to :meth:`WorldGenerator.build`.
    * :meth:`sample` returns a copy of the config with every
      :class:`Distribution` field resolved to a concrete value via
      its ``.sample(rng)`` method. Nested :class:`GeneratorConfig`
      sub-fields are recursed into; lists are processed element-wise.
    """

    # JSON round-trip (with discriminated-union ``Distribution``
    # fields) requires arbitrary types and forbids extras to keep the
    # schema tight.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    def has_distributions(self) -> bool:
        """Return whether any field still carries a :class:`Distribution`.

        Recursive: a nested :class:`GeneratorConfig` whose own field
        carries a distribution counts as "has distributions".
        """
        for value in self.__dict__.values():
            if _has_dist(value):
                return True
        return False

    def sample(self: "GeneratorConfig", rng: Optional[Union[int, np.random.Generator]] = None) -> "GeneratorConfig":
        """Return a copy with every :class:`Distribution` field resolved.

        Each distribution is replaced by exactly one ``.sample(rng)``
        draw. Non-distribution fields are passed through unchanged.
        Nested :class:`GeneratorConfig` instances are recursed into.

        Parameters
        ----------
        rng
            Either an integer seed, a :class:`numpy.random.Generator`,
            or ``None`` (fresh entropy).

        Returns
        -------
        GeneratorConfig
            A *resolved* config of the same concrete type, in which
            ``has_distributions()`` is guaranteed to return ``False``
            for any field that the base traversal can reach.
        """
        rng_obj = _coerce_rng(rng)
        update: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            new_value = _sample_value(value, rng_obj)
            if new_value is not value:  # only set if changed
                update[name] = new_value
        if not update:
            return self
        return self.model_copy(update=update)


def _has_dist(value: Any) -> bool:
    """Recursive distribution detector for :meth:`GeneratorConfig.has_distributions`."""
    if isinstance(value, Distribution):
        return True
    if isinstance(value, GeneratorConfig):
        return value.has_distributions()
    if isinstance(value, (list, tuple)):
        return any(_has_dist(v) for v in value)
    if isinstance(value, dict):
        return any(_has_dist(v) for v in value.values())
    return False


def _sample_value(value: Any, rng: np.random.Generator) -> Any:
    """Recursive sample helper used by :meth:`GeneratorConfig.sample`."""
    if isinstance(value, Distribution):
        return value.sample(rng)
    if isinstance(value, GeneratorConfig):
        return value.sample(rng)
    if isinstance(value, list):
        return [_sample_value(v, rng) for v in value]
    if isinstance(value, tuple):
        return tuple(_sample_value(v, rng) for v in value)
    return value


# ---------------------------------------------------------------------
# WorldGenerator — abstract base
# ---------------------------------------------------------------------


C = TypeVar("C", bound=GeneratorConfig)


class WorldGenerator(Generic[C]):
    """Abstract base for ``World``-producing generators.

    Subclasses parameterise themselves on a concrete
    :class:`GeneratorConfig` subclass ``C`` and implement
    :meth:`build`, which constructs a :class:`World` from a
    *resolved* config (one with no remaining
    :class:`Distribution` fields).

    The base class supplies:

    * RNG wiring via the ``seed`` constructor argument;
    * :meth:`sample_world` that resolves any distributions and
      hands the resolved config to :meth:`build`;
    * a guard in :meth:`_assert_resolved` that surfaces a clear
      error if :meth:`build` is called with an unresolved config.
    """

    cfg: C

    def __init__(
        self,
        cfg: C,
        *,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        if not isinstance(cfg, GeneratorConfig):
            raise TypeError(
                f"cfg must be a GeneratorConfig, got {type(cfg).__name__}."
            )
        self.cfg = cfg
        self._rng = _coerce_rng(seed)

    @property
    def rng(self) -> np.random.Generator:
        """The generator's :class:`numpy.random.Generator`."""
        return self._rng

    @abstractmethod
    def build(self, cfg: Optional[C] = None) -> World:
        """Build a :class:`World` from a *resolved* config.

        Subclasses must implement this method. They may assume that
        ``cfg`` (or ``self.cfg`` if ``cfg is None``) has no remaining
        :class:`Distribution` fields and may call
        :meth:`_assert_resolved` defensively.

        Parameters
        ----------
        cfg
            Optional override for ``self.cfg``. Useful in sweeps that
            compute many resolved configs upfront and feed them into
            the same generator instance.
        """

    def sample_world(
        self,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> tuple[World, C]:
        """Resolve any distributions in ``self.cfg`` and build a world.

        Parameters
        ----------
        rng
            Optional RNG override. ``None`` uses the generator's own
            RNG (``self.rng``), set at construction time.

        Returns
        -------
        tuple[World, GeneratorConfig]
            The constructed world and the resolved config (same type
            as ``self.cfg``). Persist the resolved config alongside
            the result for reproducibility.
        """
        rng_obj = self._rng if rng is None else _coerce_rng(rng)
        resolved = self.cfg.sample(rng_obj)
        return self.build(resolved), resolved

    def _assert_resolved(self, cfg: C) -> None:
        """Raise :class:`ValueError` if ``cfg`` still carries distributions."""
        if cfg.has_distributions():
            raise ValueError(
                f"{type(self).__name__}.build received a config that still "
                "carries Distribution instances. Call cfg.sample(rng) first "
                "or use Generator.sample_world(rng)."
            )
