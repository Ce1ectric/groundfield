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
from types import UnionType
from typing import (
    Annotated,
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import numpy as np
from pydantic import BaseModel, ConfigDict

from groundfield.generators._compat import PickleUpgrade
from groundfield.generators.distributions import (
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


class GeneratorConfig(PickleUpgrade, BaseModel):
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

    A config pickled by an earlier release unpickles with its renamed
    fields moved to the current names (``_field_renames`` of the
    subclass) and fields added since filled with their defaults
    (:class:`~groundfield.generators._compat.PickleUpgrade`, audit A2G-07).
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

    def sample(
        self: "GeneratorConfig", rng: Optional[Union[int, np.random.Generator]] = None
    ) -> "GeneratorConfig":
        """Return a copy with every :class:`Distribution` resolved.

        Each distribution anywhere in the config tree — fields of this
        config, of nested generator configs and of the spec models they
        hold (soil, placement, electrode, measurement specs), list items
        and dict values (e.g. ``building_counts``) — is replaced by
        exactly one ``.sample(rng)`` draw. Fields are visited in
        declaration order, lists in order, dicts in sorted key order.

        Parameters
        ----------
        rng
            Either an integer seed, a :class:`numpy.random.Generator`,
            or ``None`` (fresh entropy).

        Returns
        -------
        GeneratorConfig
            A *resolved* config of the same concrete type;
            ``has_distributions()`` returns ``False``. ``self`` is
            returned when there is nothing to resolve.

        Notes
        -----
        Population parameters that are plain numbers stay as they are:
        a ``presence_prob`` of ``0.5`` or a positive placement
        ``jitter_m`` still makes per-site draws when the resolved config
        is built (see :meth:`WorldGenerator.sample_world`).
        """
        return _sample_value(self, _coerce_rng(rng))


def _has_dist(value: Any) -> bool:
    """Recursive distribution detector for :meth:`GeneratorConfig.has_distributions`.

    Walks Pydantic models of any kind (generator configs and the spec
    models nested in them), lists, tuples and dict values.
    """
    if isinstance(value, Distribution):
        return True
    if isinstance(value, BaseModel):
        return any(_has_dist(v) for v in value.__dict__.values())
    if isinstance(value, (list, tuple)):
        return any(_has_dist(v) for v in value)
    if isinstance(value, dict):
        return any(_has_dist(v) for v in value.values())
    return False


def _sample_value(value: Any, rng: np.random.Generator, annotation: Any = None) -> Any:
    """Recursive sample helper used by :meth:`GeneratorConfig.sample`.

    Every :class:`Distribution` is replaced by one draw. Pydantic models
    are walked field by field in declaration order, lists and tuples in
    order, dicts in **sorted key order** (so a mapping re-serialised with
    another key order draws the same values). Objects without any
    distribution are returned unchanged (identity preserved).

    ``annotation`` is the declared type of the slot ``value`` sits in. A
    draw for an integer slot (``int | AnyDistribution``, also as a dict or
    list item, e.g. ``building_counts``) is stored as ``int(round(x))``,
    the value the generators build with; it stayed a float before, which
    made ``model_dump_json`` warn (stream DOC-B).
    """
    if isinstance(value, Distribution):
        drawn = value.sample(rng)
        if _integer_slot(annotation) and isinstance(drawn, (int, float)):
            return int(round(float(drawn)))
        return drawn
    if isinstance(value, BaseModel):
        fields = type(value).model_fields
        update: dict[str, Any] = {}
        for name, item in value.__dict__.items():
            info = fields.get(name)
            new_item = _sample_value(
                item, rng, None if info is None else info.annotation
            )
            if new_item is not item:
                update[name] = new_item
        return value.model_copy(update=update) if update else value
    item_annotation = _item_annotation(annotation)
    if isinstance(value, list):
        items = [_sample_value(v, rng, item_annotation) for v in value]
        return (
            value if all(a is b for a, b in zip(items, value, strict=True)) else items
        )
    if isinstance(value, tuple):
        items = [_sample_value(v, rng, item_annotation) for v in value]
        if all(a is b for a, b in zip(items, value, strict=True)):
            return value
        return tuple(items)
    if isinstance(value, dict):
        resolved = {
            key: _sample_value(value[key], rng, item_annotation)
            for key in sorted(value, key=str)
        }
        if all(resolved[k] is value[k] for k in value):
            return value
        return {key: resolved[key] for key in value}
    return value


def _strip(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """``(origin, args)`` of an annotation, through ``Annotated`` and ``Optional``."""
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return get_origin(annotation), get_args(annotation)


def _is_distribution_type(annotation: Any) -> bool:
    origin, args = _strip(annotation)
    if origin in (Union, UnionType):
        return all(_is_distribution_type(a) for a in args)
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return isinstance(annotation, type) and issubclass(annotation, Distribution)


def _integer_slot(annotation: Any) -> bool:
    """``True`` when the plain values a slot accepts are integers only."""
    origin, args = _strip(annotation)
    if origin in (Union, UnionType):
        plain = [
            a for a in args if a is not type(None) and not _is_distribution_type(a)
        ]
        return bool(plain) and all(a is int for a in plain)
    return annotation is int


def _item_annotation(annotation: Any) -> Any:
    """Declared type of the items of a list / tuple / dict slot (or ``None``)."""
    origin, args = _strip(annotation)
    if origin is dict and len(args) == 2:
        return args[1]
    if origin in (list, tuple) and args:
        return args[0]
    if origin in (Union, UnionType):
        for a in args:
            item = _item_annotation(a)
            if item is not None:
                return item
    return None


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
            raise TypeError(f"cfg must be a GeneratorConfig, got {type(cfg).__name__}.")
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
            Optional seed or generator. When given, **every** random
            number of the realisation is drawn from it: first the config
            resolution, then the per-site draws of the build (presence
            Bernoullis, placement jitter). The generator's construction
            seed then has no influence. ``None`` uses the generator's own
            RNG (``self.rng``) for both.

        Returns
        -------
        tuple[World, GeneratorConfig]
            The constructed world and the resolved config (same type
            as ``self.cfg``, without distributions).

        Notes
        -----
        Reproducing a realisation: ``sample_world(rng=seed)`` with the
        same config and seed always gives the same world. The resolved
        config alone replays the world bit-exactly with any seed when it
        has no per-site randomness (every ``presence_prob`` 0 or 1, no
        placement jitter); otherwise persist the seed, or the network
        plan / world JSON.
        """
        rng_obj = self._rng if rng is None else _coerce_rng(rng)
        resolved = self.cfg.sample(rng_obj)
        # ``build`` draws from ``self._rng``; route it to ``rng_obj`` for
        # this call (also works for subclasses whose ``build`` has no
        # ``rng`` parameter).
        saved = self._rng
        self._rng = rng_obj
        try:
            world = self.build(resolved)
        finally:
            self._rng = saved
        return world, resolved

    def _assert_resolved(self, cfg: C) -> None:
        """Raise :class:`ValueError` if ``cfg`` still carries distributions."""
        if cfg.has_distributions():
            raise ValueError(
                f"{type(self).__name__}.build received a config that still "
                "carries Distribution instances. Call cfg.sample(rng) first "
                "or use Generator.sample_world(rng)."
            )
