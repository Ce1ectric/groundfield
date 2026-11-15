"""Backend registry: names, capabilities, options and the solve contract.

Every numerical backend of :class:`~groundfield.solver.engine.Engine`
is a *registered* backend. A registration ties a name to

* a **solve function** ``solve_fn(world, engine, **options) ->
  FieldResult``;
* the **capabilities** of the backend (:class:`BackendCapabilities`):
  which soil models, conductor models and source kinds it can
  represent;
* an optional **options model** (a :class:`BackendOptions` subclass)
  that validates the backend-specific settings in
  :attr:`Engine.options <groundfield.solver.engine.Engine.options>`;
  the validated options are passed to the solve function as keyword
  arguments;
* optional **aliases**, a **deprecated-alias** target, an
  **experimental** flag and an **auto-forwarding** rule by soil model.

:meth:`Engine.solve <groundfield.solver.engine.Engine.solve>` resolves
the requested name, applies the auto-forwarding, checks the
capabilities against the world *before* any backend code runs, calls
the solve function and fills the common metadata core of the result.
Adding a backend therefore needs one :func:`register_backend` call and
no change to ``Engine.solve``.

Capability checks
-----------------
A world that a backend **cannot represent** raises
:class:`BackendCapabilityError` naming the backends that can solve it:

* the soil: homogeneous (one layer), two-layer, or n-layer
  (:math:`n \\ge 3`). Adjacent layers of equal resistivity are one
  layer physically, so a stack is acceptable to a backend that supports
  any layer count between the merged and the declared one (the check is
  conservative: it never rejects a world the backend could solve);
* electrodes with a concrete-shell correction (ADR-0012);
* leakage-free junction nodes (``World.junctions``) on a backend that
  does not build its nodal system through the shared pipeline;
* a source kernel requested explicitly that the backend does not
  implement.

A world that a backend **solves with a documented simplification**
emits :class:`BackendCapabilityWarning` and solves:

* conductor models the backend reduces (no built-in backend since
  2026.11; ``fem`` treated distributed conductors as lumped branches and
  ignored inductance and the skin effect up to its conforming model, user
  decision Q37);
* sources with a ``return_to`` electrode on a backend that returns the
  current through remote earth instead.

Backend options
---------------
The validated options of the backend that runs are passed as keyword
arguments. Settings that several backends and the post-solve layer
share are common :class:`~groundfield.solver.engine.Engine` fields, not
backend options: ``segment_length``, ``earth_inductive_model`` and the
Tagg / Sunde image-series controls ``image_max_terms`` /
``image_series_tol`` (consumed by ``image_2layer``, ``image_nlayer``,
``mom``, ``mom_sommerfeld`` on its delegated two-layer path and
``solve_mutual_*``, and recorded in ``FieldResult.metadata["series"]``,
from which :meth:`FieldResult.potential` takes them for the post-solve
field evaluation; the deprecated names ``cim`` and ``bem`` run
``image`` and ``mom``). Their single
documented defaults are :data:`DEFAULT_IMAGE_MAX_TERMS` and
:data:`DEFAULT_IMAGE_SERIES_TOL`.

Deprecated aliases and experimental backends
--------------------------------------------
Solving with a deprecated alias emits a :class:`DeprecationWarning`
that names the replacement and the earliest removal
(:data:`DEPRECATION_REMOVAL`). The first solve with an experimental
backend in a process emits :class:`ExperimentalBackendWarning` once.
Neither is emitted for the internal auto-forwarding.

Since review pass 10 (10.76) ``cim`` and ``bem`` are pure deprecated
aliases of ``image`` and ``mom`` (both assembled the same system as
their target), ``image_nlayer`` is a deprecated name of the ``image``
auto-dispatch that keeps its own solve function, so the forwarding of
``image`` to it stays silent, and ``fem`` is experimental.
"""

from __future__ import annotations

import copy
import difflib
import importlib
import os
import warnings
from collections.abc import Callable, Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - type-hint imports only
    from groundfield.soil.models import SoilModel
    from groundfield.solver.engine import Engine
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = [
    "DEFAULT_IMAGE_MAX_TERMS",
    "DEFAULT_IMAGE_SERIES_TOL",
    "DEPRECATION_REMOVAL",
    "BackendCapabilities",
    "BackendCapabilityError",
    "BackendCapabilityWarning",
    "BackendOptions",
    "BackendOptionsWarning",
    "BackendSpec",
    "ExperimentalBackendWarning",
    "FemOptions",
    "MomSommerfeldOptions",
    "available_backends",
    "get_backend",
    "register_backend",
    "unregister_backend",
]

#: Earliest calendar version in which deprecated names are removed.
DEPRECATION_REMOVAL = "2027.5"

#: Default cap on the number of Tagg / Sunde image-series terms. One
#: value for the engine, every layered backend and the post-solve field
#: evaluation. The geometric tail bound |K|^(n+1)/(1-|K|) falls below
#: the default tolerance after 277 terms for the AP1 corner soil
#: rho_1 = 1000, rho_2 = 30 Ohm m (|K| = 0.9417), so the cap stays
#: unused there.
DEFAULT_IMAGE_MAX_TERMS = 300

#: Default stop tolerance of the image series: summation ends when the
#: geometric tail bound |K|^(n+1)/(1-|K|) falls below this value.
DEFAULT_IMAGE_SERIES_TOL = 1e-6

SoilKind = Literal["homogeneous", "two_layer", "multi_layer"]
ConductorFeature = Literal["lumped", "distributed", "inductive", "skin"]
KernelName = Literal["point", "line"]

_SOIL_KINDS: tuple[str, ...] = ("homogeneous", "two_layer", "multi_layer")
_CONDUCTOR_FEATURES: tuple[str, ...] = ("lumped", "distributed", "inductive", "skin")
_KERNELS: tuple[str, ...] = ("point", "line")

SolveFunction = Callable[..., "FieldResult"]


# ---------------------------------------------------------------------
# Warnings and errors
# ---------------------------------------------------------------------


class BackendCapabilityError(ValueError, NotImplementedError):
    """The selected backend cannot represent the world.

    Raised by :meth:`Engine.solve` before any backend code runs. The
    message names the missing capability and the registered backends
    that can solve the world. The backend functions raise it as well
    when they are called directly with a soil model they cannot solve
    (``solve_image`` with a layered soil, ``solve_image_2layer`` with a
    homogeneous soil, ``solve_image_nlayer`` and ``solve_mom`` with three
    or more distinct layers), and so does the concrete-shell guard of
    the backends without the shell correction: one error type for one
    situation (audit A1-08). :class:`TypeError` is left for a
    ``world.soil`` that is no soil model at all.

    Subclasses both :class:`ValueError` and :class:`NotImplementedError`,
    the types ``image_nlayer``, ``cim``, ``bem`` and the concrete-shell
    guard raised for the same situations in 0.15.0. Changed in 2026.11:
    ``image_2layer`` (homogeneous soil, layer stack) and ``mom`` (three
    or more layers) raised :class:`TypeError`, which does not catch this
    error; catch :class:`BackendCapabilityError` (or :class:`ValueError`).
    """


class BackendCapabilityWarning(UserWarning):
    """The backend solves the world with a documented simplification."""


class ExperimentalBackendWarning(UserWarning):
    """An experimental backend was used (emitted once per process)."""


class BackendOptionsWarning(UserWarning):
    """``Engine.options`` configures a backend that will not run."""


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
_PYDANTIC_DIR = os.path.dirname(os.path.abspath(pydantic.__file__)) + os.sep
# The stem, not the file name: CPython's ``warnings.warn`` compares a prefix
# with ``filename[:-1]``, so a prefix equal to the whole file name never
# matches. The stem also covers ``copyreg``, whose frames are no user line
# either.
_COPY_STEM = os.path.splitext(os.path.abspath(copy.__file__))[0]


def warn_external(message: str, category: type[Warning]) -> None:
    """Emit a warning attributed to the first frame outside the package.

    Frames inside ``groundfield`` and ``pydantic`` (model validators) and
    in the standard library's :mod:`copy` module (``copy.replace`` on
    Python >= 3.13 calls pydantic's ``__replace__``; audit R2B-06) are
    skipped, so the warning points at the user's line. This matters most
    for :class:`DeprecationWarning`, which Python only shows by default
    when it is attributed to ``__main__``.

    Parameters
    ----------
    message
        Warning text.
    category
        Warning class.
    """
    warnings.warn(
        message,
        category,
        skip_file_prefixes=(_PACKAGE_DIR, _PYDANTIC_DIR, _COPY_STEM),
    )


# ---------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can represent.

    Parameters
    ----------
    soils
        Soil layer counts the backend solves: ``"homogeneous"`` (one
        layer), ``"two_layer"``, ``"multi_layer"`` (three or more).
        Any set of names is accepted (``{"homogeneous"}``); the
        collections are stored as :class:`frozenset`.
    conductors
        Conductor models the backend honours: ``"lumped"`` (a single
        series branch), ``"distributed"`` (``discretize_segment_length``
        set), ``"inductive"`` (``inductance_model`` set) and ``"skin"``
        (``resistance_model="skin"``, the frequency-dependent resistance
        of review pass 10, 10.37; audit round 2, A2C-4). A world using a
        model outside this set solves with a
        :class:`BackendCapabilityWarning`.
    concrete_shells
        Whether the backend consumes the per-electrode concrete-shell
        coefficient ``concrete_shell_coefficient_ohm_m`` (ADR-0012).
    two_terminal_sources
        Whether the backend returns a source current through
        ``Source.return_to``. If not, a world with a return electrode
        solves with a :class:`BackendCapabilityWarning` and the current
        returns through remote earth.
    junctions
        Whether the backend represents leakage-free junction nodes
        (``World.junctions``): conductors ending on them and sources
        attached to them. A world with junctions is rejected by a
        backend without this capability.
    voltage_sources
        Whether the backend solves impressed voltage sources
        (:class:`~groundfield.sources.VoltageSource`, modified nodal
        analysis; 2026.11). A world with a voltage source is rejected by
        a backend without this capability.
    kernels
        Source kernels the backend implements (``"point"``,
        ``"line"``). Empty for backends without a source kernel
        (volume methods); those accept any kernel setting.
    """

    # Annotated with the accepted input type (audit A1-06): a set literal
    # type-checks; __post_init__ stores a frozenset.
    soils: AbstractSet[str]
    conductors: AbstractSet[str] = frozenset(_CONDUCTOR_FEATURES)
    concrete_shells: bool = False
    two_terminal_sources: bool = False
    kernels: AbstractSet[str] = frozenset({"point"})
    junctions: bool = False
    voltage_sources: bool = False

    def __post_init__(self) -> None:
        for name, allowed in (
            ("soils", _SOIL_KINDS),
            ("conductors", _CONDUCTOR_FEATURES),
            ("kernels", _KERNELS),
        ):
            raw = getattr(self, name)
            if isinstance(raw, str):
                raise TypeError(
                    f"BackendCapabilities.{name} must be a collection of "
                    f"names, got the string {raw!r}."
                )
            value = frozenset(raw)
            unknown = sorted(value - set(allowed))
            if unknown:
                raise ValueError(
                    f"BackendCapabilities.{name}: unknown value(s) {unknown}; "
                    f"allowed: {list(allowed)}."
                )
            object.__setattr__(self, name, value)
        if not self.soils:
            raise ValueError("BackendCapabilities.soils must not be empty.")


# ---------------------------------------------------------------------
# Backend options
# ---------------------------------------------------------------------


class BackendOptions(BaseModel):
    """Base class of the backend-specific option models.

    Subclasses declare the keyword arguments of one backend's solve
    function as validated, immutable fields. Unknown keys are rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class MomSommerfeldOptions(BackendOptions):
    """Accuracy controls of the ``mom_sommerfeld`` backend.

    Forwarded to
    :func:`groundfield.solver.mom_sommerfeld.solve_mom_sommerfeld`. The
    defaults reproduce the backend's own signature defaults.
    """

    lambda_max_factor: float = Field(
        default=200.0,
        gt=0.0,
        description=(
            "Truncation of the reflected-remainder quadrature in units of "
            "1/d_rem, the decay length of the remainder (dimensionless). "
            "The discarded tail is bounded by exp(-lambda_max_factor); the "
            "factor saturates at 60 (~1e-26), so raising it above the "
            "default cannot change the top-layer result. On the delegated "
            "n = 2 cross-layer path the unit is 1/char_length (uncapped), "
            "see mom_sommerfeld.sommerfeld_kernel_value."
        ),
    )
    epsabs: float = Field(
        default=1e-9,
        gt=0.0,
        description=(
            "Absolute threshold of the grid-refinement convergence test of "
            "the reflected-remainder quadrature, in the units of the kernel "
            "(1/m)."
        ),
    )
    epsrel: float = Field(
        default=1e-7,
        gt=0.0,
        description=(
            "Relative threshold of the grid-refinement convergence test, "
            "measured against the magnitude of the whole kernel "
            "(dimensionless). Missing it raises a "
            "SommerfeldConvergenceWarning."
        ),
    )
    max_osc_panels: int = Field(
        default=200_000,
        gt=0,
        description=(
            "Budget of half-period (pi/s) Gauss panels that resolve the "
            "J0(lambda*s) oscillation of the reflected remainder (count). "
            "Binds only for s/d_rem >~ 1.9e4; there the backend raises a "
            "SommerfeldConvergenceWarning and raising this budget is the "
            "remedy. Cost grows linearly with it."
        ),
    )
    tabulate: bool = Field(
        default=True,
        description=(
            "Tabulate the reflected kernel per depth group for n >= 3 "
            "distinct layers (review pass 10, 10.61): a cubic spline in "
            "ln(s), checked at midpoints against the quadrature to 1e-7 of "
            "|G|, with a fallback to the quadrature for a group whose check "
            "fails. False evaluates every distinct pair argument by "
            "quadrature, bit-identical to the 0.15.0 pair loop (27 times "
            "slower for eleven foundation rings, N = 273)."
        ),
    )


class FemOptions(BackendOptions):
    """Mesh, extrapolation and truncation controls of the ``fem`` backend.

    Forwarded to :func:`groundfield.solver.fem.solve_fem`. The defaults
    reproduce the backend's own signature defaults. Since 2026.11 ``fem``
    is a conforming finite-element model (user decision Q37): axisymmetric
    for rods on one axis and rings around it, three-dimensional for every
    other world (``model``, ``source_radius``, ``growth``,
    ``concrete_resistivity``; the optional ``fem3d`` extra). The controls of
    the equivalent-hemisphere mesh (``n_radial``, ``n_axial``,
    ``z_far_factor``) are gone.
    """

    refinement: int = Field(
        default=0,
        ge=0,
        le=6,
        description=(
            "Refinement level of the coarser mesh (count): each level bisects "
            "every interval of the graded mesh lines (about four times the "
            "nodes). The finer mesh is one level above it."
        ),
    )
    extrapolate: bool = Field(
        default=True,
        description=(
            "Solve at refinement and refinement + 1 and extrapolate the port "
            "conductance matrix to zero mesh size (Richardson, second order). "
            "False solves the level refinement alone (a lower bound of the "
            "resistance)."
        ),
    )
    r_far_factor: float = Field(
        default=1000.0,
        gt=1.0,
        description=(
            "Size of the truncation box in characteristic lengths "
            "(dimensionless): the electrodes' depth and radius, or the depth "
            "of the deepest interface times the layer contrast if larger."
        ),
    )
    far_field: Literal["robin", "dirichlet"] = Field(
        default="robin",
        description=(
            "Condition on the truncation box: 'robin' applies the exact "
            "monopole Dirichlet-to-Neumann map, 'dirichlet' grounds the box "
            "(biases R low by about rho_n / (2 pi R_box))."
        ),
    )
    max_nodes: int = Field(
        default=4_000_000,
        gt=0,
        description=(
            "Resource limit (count): the predicted node count of the finer "
            "mesh above which the solve is refused (ResourceLimitError)."
        ),
    )
    model: Literal["auto", "axisymmetric", "3d"] = Field(
        default="auto",
        description=(
            "Which finite-element model solves the world: 'auto' takes the "
            "axisymmetric model where it applies (rods on one axis, rings "
            "around it, no galvanic distributed conductor) and the "
            "three-dimensional model otherwise; '3d' forces the 3-D model "
            "(needs the fem3d extra); 'axisymmetric' refuses every other world."
        ),
    )
    source_radius: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "3-D model: radius in m of the disc over which each wire's leakage "
            "is smoothed (the analytic radius correction covers the rest). None: "
            "0.06 m, at least 2.2 bare wire radii, limited by the depth of the "
            "wires and their gap to the layer interfaces."
        ),
    )
    growth: float = Field(
        default=0.5,
        ge=0.05,
        le=2.0,
        description=(
            "3-D model: growth of the element size per metre away from the "
            "wires (dimensionless). Smaller is finer and slower."
        ),
    )
    concrete_resistivity: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "3-D model: None applies each encased wire's shell coefficient "
            "(ADR-0012, each wire its own shell). A resistivity in Ohm m "
            "resolves the concrete as square prisms of that resistivity around "
            "the encased wires (axis-aligned straight wires), so overlapping "
            "shells at corners, T-joints and crossings share their concrete "
            "(user decision Q40)."
        ),
    )


# ---------------------------------------------------------------------
# Specs and the registry
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BackendSpec:
    """Registration record of one backend name.

    Attributes
    ----------
    name
        Registered name.
    solve_fn
        Solve function, or a lazy ``"module:function"`` reference that
        is imported on every call (so monkeypatching the module works).
        ``None`` for a pure deprecated alias, which runs its target.
    capabilities
        Declared capabilities; ``None`` only for a pure alias.
    options_model
        :class:`BackendOptions` subclass validating ``Engine.options``
        for this backend, or ``None``.
    aliases
        Further accepted names (resolved silently).
    deprecated_alias_of
        Replacement name if this name is deprecated.
    experimental
        Whether solving with the backend emits
        :class:`ExperimentalBackendWarning` (once per process).
    forward
        Auto-forwarding rule: pairs ``(soil model kind, backend name)``.
    summary
        One-line description.
    """

    name: str
    solve_fn: SolveFunction | str | None
    capabilities: BackendCapabilities | None
    options_model: type[BackendOptions] | None = None
    aliases: tuple[str, ...] = ()
    deprecated_alias_of: str | None = None
    experimental: bool = False
    forward: tuple[tuple[str, str], ...] = ()
    summary: str = ""

    @property
    def is_pure_alias(self) -> bool:
        """Whether the name only forwards to :attr:`deprecated_alias_of`."""
        return self.solve_fn is None

    def resolve_solve_fn(self) -> SolveFunction:
        """Return the callable solve function (imports lazy references).

        Returns
        -------
        Callable
            ``solve_fn(world, engine, **options) -> FieldResult``.

        Raises
        ------
        TypeError
            For a pure alias, which has no solve function of its own.
        """
        fn = self.solve_fn
        if fn is None:
            raise TypeError(f"Backend '{self.name}' is a pure alias.")
        if isinstance(fn, str):
            module_name, _, attr = fn.partition(":")
            return getattr(importlib.import_module(module_name), attr)
        return fn

    def forward_target(self, soil_kind: str) -> str | None:
        """Backend the call is forwarded to for a soil model kind, if any."""
        for kind, target in self.forward:
            if kind == soil_kind:
                return target
        return None


_REGISTRY: dict[str, BackendSpec] = {}
_ALIASES: dict[str, str] = {}
_EXPERIMENTAL_WARNED: set[str] = set()


def register_backend(
    name: str,
    solve_fn: SolveFunction | str | None,
    *,
    capabilities: BackendCapabilities | None = None,
    options_model: type[BackendOptions] | None = None,
    aliases: Iterable[str] = (),
    deprecated_alias_of: str | None = None,
    experimental: bool = False,
    forward: Mapping[str, str] | None = None,
    summary: str = "",
    replace: bool = False,
) -> BackendSpec:
    """Register a backend name.

    Parameters
    ----------
    name
        Backend name as used in ``Engine(backend=name)``.
    solve_fn
        ``solve_fn(world, engine, **options) -> FieldResult``, or a lazy
        ``"package.module:function"`` reference. ``None`` registers a
        pure deprecated alias that runs ``deprecated_alias_of``.
    capabilities
        What the backend can represent. Required unless ``solve_fn`` is
        ``None`` (a pure alias inherits its target's capabilities).
    options_model
        :class:`BackendOptions` subclass for ``Engine.options[name]``.
        Its fields must be keyword arguments of ``solve_fn``.
    aliases
        Further accepted names, resolved without a warning.
    deprecated_alias_of
        Registered replacement name. Solving with ``name`` then emits a
        :class:`DeprecationWarning` naming the replacement and
        :data:`DEPRECATION_REMOVAL`. With a ``solve_fn`` the old
        implementation keeps running; without, the replacement runs.
    experimental
        Emit :class:`ExperimentalBackendWarning` on the first solve in a
        process.
    forward
        Auto-forwarding by soil model kind, e.g. ``{"two_layer":
        "image_2layer"}``. Targets are resolved at solve time.
    summary
        One-line description.
    replace
        Allow replacing an existing registration of ``name``; the name
        keeps its position in :func:`available_backends`.

    Returns
    -------
    BackendSpec
        The stored registration record.

    Raises
    ------
    ValueError
        If the name is taken (and ``replace`` is false), the alias
        target is unknown or itself deprecated, or the arguments are
        inconsistent.
    """
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError(f"Backend name must be a non-empty string, got {name!r}.")
    alias_names = tuple(aliases)
    taken = [n for n in (name, *alias_names) if n in _REGISTRY or n in _ALIASES]
    if replace and name in _REGISTRY:
        own = {name, *_REGISTRY[name].aliases}
        taken = [n for n in taken if n not in own]
    if taken:
        raise ValueError(
            f"Backend name(s) {taken} already registered; pass replace=True "
            "to replace a registration."
        )
    if deprecated_alias_of is not None:
        target = _REGISTRY.get(deprecated_alias_of)
        if target is None:
            raise ValueError(
                f"deprecated_alias_of={deprecated_alias_of!r} is not a "
                "registered backend."
            )
        if target.deprecated_alias_of is not None:
            raise ValueError(
                f"Backend '{deprecated_alias_of}' is itself deprecated; "
                "point the alias at its replacement."
            )
    if solve_fn is None:
        if deprecated_alias_of is None:
            raise ValueError("solve_fn=None is only allowed for a deprecated alias.")
        if capabilities is not None or options_model is not None or forward:
            raise ValueError(
                "A pure deprecated alias inherits capabilities, options and "
                "forwarding from its target; do not pass them."
            )
    elif capabilities is None:
        raise ValueError(f"Backend '{name}' needs capabilities.")
    if options_model is not None and not (
        isinstance(options_model, type) and issubclass(options_model, BackendOptions)
    ):
        raise TypeError("options_model must be a BackendOptions subclass.")
    forward_pairs = tuple((forward or {}).items())
    bad_kinds = sorted({k for k, _ in forward_pairs} - set(_SOIL_KINDS))
    if bad_kinds:
        raise ValueError(f"forward: unknown soil model kind(s) {bad_kinds}.")

    if deprecated_alias_of is not None:
        dependants = [
            s.name for s in _REGISTRY.values() if s.deprecated_alias_of == name
        ]
        if dependants:
            raise ValueError(
                f"Backend(s) {dependants} are deprecated aliases of '{name}'; it "
                "cannot become a deprecated alias itself."
            )
    if replace and name in _REGISTRY:
        # Re-assigning the key below keeps the registration order.
        for alias in _REGISTRY[name].aliases:
            _ALIASES.pop(alias, None)
        _EXPERIMENTAL_WARNED.discard(name)
    spec = BackendSpec(
        name=name,
        solve_fn=solve_fn,
        capabilities=capabilities,
        options_model=options_model,
        aliases=alias_names,
        deprecated_alias_of=deprecated_alias_of,
        experimental=bool(experimental),
        forward=forward_pairs,
        summary=summary,
    )
    _REGISTRY[name] = spec
    for alias in alias_names:
        _ALIASES[alias] = name
    return spec


def unregister_backend(name: str) -> BackendSpec:
    """Remove a registration together with its aliases.

    Parameters
    ----------
    name
        Registered name (not an alias).

    Returns
    -------
    BackendSpec
        The removed record.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    spec = _REGISTRY.pop(name)
    for alias in spec.aliases:
        _ALIASES.pop(alias, None)
    _EXPERIMENTAL_WARNED.discard(name)
    return spec


def available_backends(*, include_deprecated: bool = False) -> tuple[str, ...]:
    """Registered backend names in registration order.

    Parameters
    ----------
    include_deprecated
        Also list deprecated aliases.

    Returns
    -------
    tuple of str
        Backend names (aliases from ``aliases=`` are not listed).
    """
    return tuple(
        name
        for name, spec in _REGISTRY.items()
        if include_deprecated or spec.deprecated_alias_of is None
    )


def get_backend(name: str) -> BackendSpec:
    """Look up a backend by name or alias.

    Parameters
    ----------
    name
        Backend name or alias.

    Returns
    -------
    BackendSpec
        The registration record (for a deprecated alias: the alias's
        own record, whose :attr:`BackendSpec.deprecated_alias_of` names
        the replacement).

    Raises
    ------
    ValueError
        If the name is unknown; the message suggests the closest
        registered name.
    """
    key = _ALIASES.get(name, name)
    spec = _REGISTRY.get(key)
    if spec is None:
        known = list(_REGISTRY) + list(_ALIASES)
        close = difflib.get_close_matches(str(name), known, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(
            f"Unknown backend '{name}'.{hint} Registered backends: "
            f"{', '.join(available_backends())}."
        )
    return spec


def _implementation(spec: BackendSpec) -> BackendSpec:
    """The record whose solve function runs for ``spec``."""
    if spec.is_pure_alias:
        return _REGISTRY[spec.deprecated_alias_of]  # type: ignore[index]
    return spec


def _capabilities(spec: BackendSpec) -> BackendCapabilities:
    """Capabilities of the implementation behind ``spec``."""
    caps = _implementation(spec).capabilities
    assert caps is not None  # enforced by register_backend
    return caps


def options_key(name: str) -> str:
    """Canonical ``Engine.options`` key of a backend name or alias."""
    return _implementation(get_backend(name)).name


def reachable_backends(name: str) -> frozenset[str]:
    """Backends that can run when ``name`` is requested (any soil)."""
    spec = _implementation(get_backend(name))
    names = {spec.name}
    for _, target in spec.forward:
        if target in _REGISTRY or target in _ALIASES:
            names.add(_implementation(get_backend(target)).name)
    return frozenset(names)


@dataclass(frozen=True)
class ResolvedBackend:
    """Outcome of :func:`resolve_backend` for one world.

    Attributes
    ----------
    requested
        Name as given in ``Engine.backend``.
    spec
        Record of the backend that runs (after alias resolution and
        auto-forwarding).
    deprecated
        Record of the requested deprecated alias, or ``None``.
    """

    requested: str
    spec: BackendSpec
    deprecated: BackendSpec | None


def resolve_backend(requested: str, world: World) -> ResolvedBackend:
    """Resolve aliases and the auto-forwarding for ``world``.

    Parameters
    ----------
    requested
        Backend name as given in ``Engine.backend``.
    world
        World to be solved (its soil model kind drives the forwarding).

    Returns
    -------
    ResolvedBackend
        The backend that runs; emits no warning.
    """
    first = get_backend(requested)
    deprecated = first if first.deprecated_alias_of is not None else None
    spec = _implementation(first)
    soil_kind = getattr(world.soil, "kind", None)
    target = spec.forward_target(soil_kind) if soil_kind else None
    if target is not None:
        spec = _implementation(get_backend(target))
    return ResolvedBackend(requested=requested, spec=spec, deprecated=deprecated)


def emit_resolution_warnings(resolved: ResolvedBackend) -> None:
    """Deprecation warning for a requested alias; experimental warning once."""
    if resolved.deprecated is not None:
        warn_external(
            f"Backend name '{resolved.deprecated.name}' is deprecated; use "
            f"'{resolved.deprecated.deprecated_alias_of}' instead. The alias "
            f"will be removed no earlier than {DEPRECATION_REMOVAL}.",
            DeprecationWarning,
        )
    spec = resolved.spec
    if spec.experimental and spec.name not in _EXPERIMENTAL_WARNED:
        warn_external(
            f"Backend '{spec.name}' is experimental: its results are not "
            "validated as a cross-check and may change between releases.",
            ExperimentalBackendWarning,
        )
        # Marked after the warning, so an "error" filter raises on every
        # solve instead of only the first one.
        _EXPERIMENTAL_WARNED.add(spec.name)


# ---------------------------------------------------------------------
# Capability check
# ---------------------------------------------------------------------


def _layer_counts(soil: SoilModel) -> tuple[int, int]:
    """Merged and declared layer counts of a soil model."""
    from groundfield.solver._layered import as_layer_stack

    # The *declared* stack: as_layer_stack merges equal layers by default.
    rhos = [float(r) for r in as_layer_stack(soil, canonical=False).rhos]
    merged = 1 + sum(1 for a, b in zip(rhos, rhos[1:], strict=False) if a != b)
    return merged, len(rhos)


def _kind_of(n_layers: int) -> str:
    if n_layers <= 1:
        return "homogeneous"
    if n_layers == 2:
        return "two_layer"
    return "multi_layer"


def soil_kinds(soil: SoilModel) -> frozenset[str]:
    """Layer-count kinds a soil model can be solved as.

    A stack whose adjacent layers share a resistivity can be solved with
    any layer count between the merged and the declared one.

    Parameters
    ----------
    soil
        Soil model of the world.

    Returns
    -------
    frozenset of str
        Subset of ``{"homogeneous", "two_layer", "multi_layer"}``.
    """
    merged, declared = _layer_counts(soil)
    return frozenset(_kind_of(n) for n in range(merged, declared + 1))


def _soil_text(soil: SoilModel) -> str:
    merged, declared = _layer_counts(soil)
    label = {1: "homogeneous", 2: "two-layer"}.get(declared, f"{declared}-layer")
    if merged != declared:
        return f"{label} soil ({merged} layer(s) after merging equal resistivities)"
    return f"{label} soil"


def _shelled_electrodes(world: World) -> list[str]:
    return [
        e.name
        for e in world.electrodes
        if float(getattr(e, "concrete_shell_coefficient_ohm_m", 0.0) or 0.0) != 0.0
    ]


def _conductor_features(world: World) -> set[str]:
    features: set[str] = set()
    for c in world.conductors:
        features.add("distributed" if c.is_distributed else "lumped")
        if c.inductance_model is not None:
            features.add("inductive")
        if getattr(c, "resistance_model", "dc") == "skin":
            features.add("skin")
    return features


def _hard_problems(
    spec: BackendSpec, world: World, kernel: str | None, kernel_explicit: bool
) -> list[str]:
    caps = _capabilities(spec)
    problems: list[str] = []
    if world.soil is not None and not soil_kinds(world.soil) & caps.soils:
        problems.append(f"it does not support {_soil_text(world.soil)}")
    shelled = _shelled_electrodes(world)
    if shelled and not caps.concrete_shells:
        problems.append(
            "it does not implement the concrete-shell correction (ADR-0012) "
            f"carried by electrode(s) {shelled}"
        )
    junctions = [j.name for j in world.junctions]
    if junctions and not caps.junctions:
        problems.append(f"it does not support junction nodes ({junctions})")
    voltage = [s.name for s in world.sources if s.kind == "voltage"]
    if voltage and not caps.voltage_sources:
        problems.append(f"it does not implement impressed voltage sources ({voltage})")
    if (
        kernel is not None
        and kernel_explicit
        and caps.kernels
        and kernel not in caps.kernels
    ):
        problems.append(
            f"it does not implement kernel '{kernel}' (supported: "
            f"{', '.join(repr(k) for k in sorted(caps.kernels))})"
        )
    return problems


def _capable_backends(
    world: World, kernel: str | None, kernel_explicit: bool
) -> list[str]:
    names = []
    soil_kind = getattr(world.soil, "kind", None)
    for name in available_backends():
        spec = _REGISTRY[name]
        target = spec.forward_target(soil_kind) if soil_kind else None
        run = _implementation(get_backend(target)) if target else spec
        if not _hard_problems(run, world, kernel, kernel_explicit):
            names.append(f"'{name}'" + (" (experimental)" if run.experimental else ""))
    return names


def _voltage_capable_backends(world: World) -> str:
    """Registered backends that solve ``world`` with its voltage sources.

    The same list as the message of :func:`check_capabilities`
    (:func:`_capable_backends`; audit W2A-07: up to then every backend
    with the ``voltage_sources`` capability was named, also one that
    does not support the world's soil).
    """
    names = _capable_backends(world, None, False)
    return ", ".join(names) if names else "none"


def check_capabilities(
    resolved: ResolvedBackend,
    world: World,
    *,
    kernel: str | None = None,
    kernel_explicit: bool = False,
) -> None:
    """Check that the resolved backend can represent ``world``.

    Parameters
    ----------
    resolved
        Outcome of :func:`resolve_backend`.
    world
        World to be solved.
    kernel
        Source kernel requested by the engine, if the engine has one.
    kernel_explicit
        Whether the kernel was set explicitly (a default kernel that the
        backend does not implement is replaced by the backend's own).

    Raises
    ------
    BackendCapabilityError
        If the backend cannot represent the world.

    Warns
    -----
    BackendCapabilityWarning
        If the backend solves the world with a documented
        simplification (reduced conductor model, ignored
        ``return_to``).
    """
    spec = resolved.spec
    caps = _capabilities(spec)
    label = f"'{resolved.requested}'"
    if resolved.requested != spec.name:
        label += f" (running '{spec.name}')"
    problems = _hard_problems(spec, world, kernel, kernel_explicit)
    if problems:
        capable = _capable_backends(world, kernel, kernel_explicit)
        alternatives = (
            f"Backends that can: {', '.join(capable)}."
            if capable
            else "No registered backend supports this combination."
        )
        raise BackendCapabilityError(
            f"Backend {label} cannot solve this world: "
            f"{'; '.join(problems)}. {alternatives}"
        )

    reduced = sorted(_conductor_features(world) - caps.conductors)
    if reduced:
        able = [
            f"'{n}'"
            for n in available_backends()
            if set(reduced) <= _capabilities(_REGISTRY[n]).conductors
        ]
        dc = (
            " at their DC resistance (the skin effect of resistance_model="
            "'skin' is ignored)"
            if "skin" in reduced
            else ""
        )
        warn_external(
            f"Backend '{spec.name}' does not implement the {', '.join(reduced)} "
            "conductor model(s) of this world and solves them as lumped, "
            f"purely resistive branches{dc}. Backends that implement them: "
            f"{', '.join(able)}.",
            BackendCapabilityWarning,
        )
    two_terminal = [s.name for s in world.sources if getattr(s, "return_to", None)]
    if two_terminal and not caps.two_terminal_sources:
        warn_external(
            f"Backend '{spec.name}' ignores Source.return_to (source(s) "
            f"{two_terminal}): the current returns through remote earth, not "
            "through the return electrode.",
            BackendCapabilityWarning,
        )


def effective_kernel(spec: BackendSpec, requested: str | None) -> str | None:
    """Kernel a backend uses when the engine requests ``requested``.

    Parameters
    ----------
    spec
        Record of the backend that runs.
    requested
        Kernel requested by the engine, or ``None``.

    Returns
    -------
    str or None
        ``None`` for backends without a source kernel; otherwise the
        requested kernel if implemented, else the backend's own
        (``"point"`` when available).
    """
    kernels = spec.capabilities.kernels if spec.capabilities else frozenset()
    if not kernels:
        return None
    if requested in kernels:
        return requested
    return "point" if "point" in kernels else sorted(kernels)[0]


# ---------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------


def _solve_image_2layer(world: World, engine: Engine) -> FieldResult:
    """``image_2layer`` with the engine's common series controls."""
    from groundfield.solver import image_2layer

    return image_2layer.solve_image_2layer(
        world, engine, max_terms=engine.image_max_terms, tol=engine.image_series_tol
    )


def _solve_image_nlayer(world: World, engine: Engine) -> FieldResult:
    """``image_nlayer`` with the engine's common series controls.

    ``image_nlayer`` re-dispatches to ``solve_image_2layer`` for n = 2,
    so the series controls are forwarded here as well.
    """
    from groundfield.solver import image_nlayer

    return image_nlayer.solve_image_nlayer(
        world, engine, max_terms=engine.image_max_terms, tol=engine.image_series_tol
    )


_ALL_CONDUCTORS = frozenset(_CONDUCTOR_FEATURES)
# The image family will also offer the exact line-source kernel
# (Engine.kernel, review finding 10.28); declaring it here keeps an
# explicit kernel="line" request valid once that field exists.
_IMAGE_KERNELS = frozenset({"point", "line"})


def _register_builtins() -> None:
    register_backend(
        "image",
        "groundfield.solver.image:solve_image",
        capabilities=BackendCapabilities(
            soils={"homogeneous"},
            conductors=_ALL_CONDUCTORS,
            concrete_shells=True,
            two_terminal_sources=True,
            kernels=_IMAGE_KERNELS,
            junctions=True,
            voltage_sources=True,
        ),
        forward={"two_layer": "image_2layer", "multi_layer": "image_nlayer"},
        summary="Homogeneous soil, closed-form image-charge sum.",
    )
    register_backend(
        "image_2layer",
        _solve_image_2layer,
        capabilities=BackendCapabilities(
            soils={"two_layer"},
            conductors=_ALL_CONDUCTORS,
            concrete_shells=True,
            two_terminal_sources=True,
            kernels=_IMAGE_KERNELS,
            junctions=True,
            voltage_sources=True,
        ),
        summary="Two-layer soil, Tagg / Sunde image series.",
    )
    # Deprecated name (10.76): requesting it warns; the forwarding of
    # 'image' to it does not (only the requested name is checked).
    register_backend(
        "image_nlayer",
        _solve_image_nlayer,
        capabilities=BackendCapabilities(
            soils={"homogeneous", "two_layer"},
            conductors=_ALL_CONDUCTORS,
            concrete_shells=True,
            two_terminal_sources=True,
            kernels=_IMAGE_KERNELS,
            junctions=True,
            voltage_sources=True,
        ),
        deprecated_alias_of="image",
        summary="Layered soil routed to image (n = 1) or image_2layer (n = 2).",
    )
    register_backend(
        "cim",
        None,
        deprecated_alias_of="image",
        summary="Deprecated name of the image auto-dispatch (10.76).",
    )
    # The point-matching family runs the shared inject / build_topology
    # steps (10.75): return_to, junction nodes and voltage sources are
    # honoured.
    register_backend(
        "mom",
        "groundfield.solver.mom:solve_mom",
        capabilities=BackendCapabilities(
            soils={"homogeneous", "two_layer"},
            concrete_shells=True,
            two_terminal_sources=True,
            junctions=True,
            voltage_sources=True,
        ),
        summary="Method of moments, point-matching collocation.",
    )
    register_backend(
        "mom_sommerfeld",
        "groundfield.solver.mom_sommerfeld:solve_mom_sommerfeld",
        capabilities=BackendCapabilities(
            soils={"homogeneous", "two_layer", "multi_layer"},
            concrete_shells=True,
            two_terminal_sources=True,
            junctions=True,
            voltage_sources=True,
        ),
        options_model=MomSommerfeldOptions,
        summary="Method of moments with Sommerfeld quadrature, any n.",
    )
    register_backend(
        "bem",
        None,
        deprecated_alias_of="mom",
        summary="Deprecated name of 'mom' (10.76).",
    )
    # Conforming finite-element models (user decision Q37): axisymmetric or
    # three-dimensional; the earth part is a port matrix, the network the
    # shared pipeline's, so conductors,
    # junctions, return_to and voltage sources are honoured. No source
    # kernel: the engine's kernel only selects the form of the point
    # sources for the post-solve field evaluation.
    register_backend(
        "fem",
        "groundfield.solver.fem:solve_fem",
        capabilities=BackendCapabilities(
            soils={"homogeneous", "two_layer", "multi_layer"},
            conductors=_ALL_CONDUCTORS,
            concrete_shells=True,
            two_terminal_sources=True,
            kernels=frozenset(),
            junctions=True,
            voltage_sources=True,
        ),
        options_model=FemOptions,
        experimental=True,
        summary=(
            "Conforming finite elements: axisymmetric (rods and rings on one "
            "axis) or three-dimensional (any electrodes; fem3d extra)."
        ),
    )


_register_builtins()


def _describe_options(model: type[BackendOptions]) -> str:
    return ", ".join(model.model_fields)


def validate_options_entry(key: str, value: Any) -> tuple[str, BackendOptions]:
    """Validate one ``Engine.options`` entry.

    Parameters
    ----------
    key
        Backend name or alias.
    value
        Options model instance or mapping of option values.

    Returns
    -------
    tuple
        ``(canonical backend name, options model instance)``.

    Raises
    ------
    ValueError
        For an unknown backend, a backend without options, an unknown
        option name (with a suggestion) or a wrong value type.
    """
    canonical = options_key(key)
    model = _REGISTRY[canonical].options_model
    if model is None:
        raise ValueError(
            f"Backend '{canonical}' has no backend-specific options. Settings "
            "shared by several backends are Engine fields: segment_length, "
            "earth_inductive_model, image_max_terms, image_series_tol."
        )
    if isinstance(value, model):
        return canonical, value
    if isinstance(value, BackendOptions):
        raise ValueError(
            f"Engine.options['{canonical}'] must be a {model.__name__}, got "
            f"{type(value).__name__}."
        )
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Engine.options['{canonical}'] must be a {model.__name__} or a "
            f"mapping, got {type(value).__name__}."
        )
    unknown = [k for k in value if k not in model.model_fields]
    if unknown:
        hints = []
        for k in unknown:
            close = difflib.get_close_matches(str(k), list(model.model_fields), n=1)
            hints.append(f"'{k}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        raise ValueError(
            f"Unknown option(s) {', '.join(hints)} for backend '{canonical}'. "
            f"Valid options: {_describe_options(model)}."
        )
    return canonical, model.model_validate(dict(value))
