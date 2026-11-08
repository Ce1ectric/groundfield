"""Numerical computation core (``Engine``).

An ``Engine`` bundles **everything that is purely numerical**: the
chosen backend (image, MoM, FEM, ...), the frequency list, the
discretisation and the backend settings. It is deliberately separated
from the ``World``, so that the same physical configuration can be run
against different backends.

Backends are registered in :mod:`groundfield.solver.registry`, which
records for each name the solve function, the declared capabilities
(soil layer counts, conductor models, concrete shells, two-terminal
sources, source kernels) and the backend-specific options model. The
built-in backends fall into three families:

*Closed-form / image-charge.*
``image`` is the homogeneous-soil image-charge sum (closed form,
uniform current per unit length, cheapest backend). ``image_2layer``
implements the Tagg / Sunde geometric series for two-layer soil with
convergence $|K|^n$. Both use the exact line-source kernel by default
(``Engine.kernel``). ``image_nlayer`` (deprecated name) routes a layered
soil with one or two distinct layers to ``image`` / ``image_2layer``
and rejects three or more; ``cim`` is a deprecated name of the ``image``
auto-dispatch.

*Integral equation.* ``mom`` solves for the segment currents by point
matching (collocation) with the closed-form kernels above;
``mom_sommerfeld`` does the same with direct numerical Sommerfeld
quadrature of the layered Green's function and is the only integral
backend for $n \\ge 3$ layers. ``bem`` is a deprecated name of ``mom``.

*Volume PDE.* ``fem`` (experimental) is the axisymmetric finite-element
solver with the equivalent-hemisphere reduction described in
:mod:`groundfield.solver.fem`.

Notes
-----
:meth:`Engine.solve` performs an automatic dispatch when
``backend="image"`` is requested: the call is forwarded to
``"image_2layer"`` if the world holds a :class:`TwoLayerSoil`, and to
``"image_nlayer"`` if it holds a :class:`MultiLayerSoil`. Notebooks
written for the homogeneous case therefore keep working when the soil
is replaced by a layered one.

Every result carries the metadata core :data:`CORE_METADATA_KEYS`,
filled here for all backends, next to the backend-specific keys.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    ValidationInfo,
    field_validator,
    model_validator,
)

from groundfield.solver import registry as _registry
from groundfield.solver._kernels import DEFAULT_KERNEL
from groundfield.solver.registry import (
    DEFAULT_IMAGE_MAX_TERMS,
    DEFAULT_IMAGE_SERIES_TOL,
    DEPRECATION_REMOVAL,
    BackendOptions,
    BackendOptionsWarning,
    MomSommerfeldOptions,
    warn_external,
)
from groundfield.solver.result import FieldResult, SourceRecord, solve_provenance
from groundfield.utils.logging import get_logger
from groundfield.utils.suggest import did_you_mean

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.world import World

__all__ = [
    "CORE_METADATA_KEYS",
    "DEFAULT_IMAGE_MAX_TERMS",
    "DEFAULT_IMAGE_SERIES_TOL",
    "DEFAULT_KERNEL",
    "Engine",
    "Backend",
    "EarthInductiveModel",
    "EngineFrequencyOrderWarning",
]

#: Metadata keys that :meth:`Engine.solve` writes into every result.
#:
#: ``backend``
#:     Backend name as requested (``Engine.backend``).
#: ``effective_backend``
#:     Registered backend that ran, after alias resolution and the
#:     auto-forwarding by soil model.
#: ``kernel``
#:     Source kernel used (``"point"`` / ``"line"``); ``None`` for
#:     backends without a source kernel (``fem``).
#: ``groundfield_version``
#:     Version of the package that solved.
#: ``frequencies``
#:     Frequencies in Hz, in solve order.
#: ``n_segments``
#:     Number of current-carrying source segments (``0`` for ``fem``).
#: ``soil_kind``
#:     ``kind`` of the soil model (``"homogeneous"``, ``"two_layer"``,
#:     ``"multi_layer"``).
#: ``earth_inductive_model``
#:     Earth model of the inductive coupling.
#: ``solve_time_s``
#:     Wall time of the backend call in s.
#: ``series``
#:     Image-series controls of the engine, ``{"max_terms", "tol"}``,
#:     for the post-solve field evaluation.
#: ``max_segments``
#:     Problem-size limit the solve was checked against
#:     (:attr:`Engine.max_segments`, or
#:     :data:`groundfield.diagnostics.HARD_LIMIT` when the engine leaves
#:     it at ``None``); ``fem`` has no segments and is not limited by it.
#: ``provenance``
#:     Versions, platform, engine configuration, world fingerprint and
#:     UTC timestamp (see :func:`groundfield.solver.result.solve_provenance`).
CORE_METADATA_KEYS: tuple[str, ...] = (
    "backend",
    "effective_backend",
    "kernel",
    "groundfield_version",
    "frequencies",
    "n_segments",
    "soil_kind",
    "earth_inductive_model",
    "solve_time_s",
    "series",
    "max_segments",
    "provenance",
)


class EngineFrequencyOrderWarning(UserWarning):
    """``Engine.frequencies`` is not strictly monotonically increasing.

    A dedicated category for the monotonic-order warning emitted by
    :meth:`Engine._validate_frequencies`. Using a dedicated class lets
    notebook authors silence the diagnostic with a stable
    ``warnings.simplefilter("once",
    EngineFrequencyOrderWarning)`` — the default ``UserWarning`` filter
    deduplicates by *message text*, and because the message embeds the
    offending list literal, every distinct non-monotonic list
    triggers a fresh warning even when the underlying convention is
    the same.
    """


def _check_frequency_values(values: list[float]) -> None:
    """Validate the scalar contract of a frequency list.

    Shared by the ``frequencies`` field validator and by
    :meth:`Engine.with_frequencies`, so both entry points reject the
    same inputs with the same messages. The *order* check lives in the
    field validator only, because ``with_frequencies(preserve_order=True)``
    opts out of it.

    Parameters
    ----------
    values
        Candidate frequency list in Hz.

    Raises
    ------
    ValueError
        If the list is empty, or contains a negative, NaN or infinite
        entry. DC (``f == 0.0``) is a legitimate quasi-static operating
        point and is accepted.
    """
    if not values:
        raise ValueError("Engine.frequencies must not be empty.")
    for f in values:
        if f != f:  # NaN guard — must precede the comparison below
            raise ValueError("Engine.frequencies must be finite, got NaN.")
        if f == float("inf"):
            raise ValueError("Engine.frequencies must be finite, got inf.")
        # DC (``f == 0``) is a legitimate operating point for
        # quasi-static grounding studies — reject only negatives
        # and non-finite values.
        if not (f >= 0.0):
            raise ValueError(f"Engine.frequencies must be non-negative, got {f}.")


#: Names of the built-in backends (for type hints; the registry is the
#: authority at runtime and also accepts registered third-party names).
Backend = Literal[
    "image",
    "image_2layer",
    "image_nlayer",
    "cim",
    "mom",
    "mom_sommerfeld",
    "bem",
    "fem",
]

# ADR-0005 / ADR-0006: how the earth contributes to the inductive coupling.
EarthInductiveModel = Literal[
    "perfect_mirror",  # ADR-0004 default — sigma_earth -> infinity
    "carson_series",  # ADR-0005 — Carson 1926 per-m asymptotic correction
    "sommerfeld",  # ADR-0006 — geometric Sommerfeld kernel integration
]

_log = get_logger(__name__)

# Flat ``mom_sommerfeld`` fields of 0.15.0 and their option names.
_LEGACY_SOMMERFELD_FIELDS: dict[str, str] = {
    "sommerfeld_lambda_max_factor": "lambda_max_factor",
    "sommerfeld_epsabs": "epsabs",
    "sommerfeld_epsrel": "epsrel",
    "sommerfeld_max_osc_panels": "max_osc_panels",
}

# Inert fields (no backend reads them) and their defaults.
_INERT_DEFAULTS: dict[str, float] = {"tolerance": 1e-6, "max_iterations": 200}


def _inert_message(name: str) -> str:
    return (
        f"Engine.{name} has no effect: no backend reads it. The field is "
        f"deprecated and will be removed no earlier than "
        f"{DEPRECATION_REMOVAL}; remove the argument."
    )


def _source_records(world: "World", n_freq: int) -> list[SourceRecord]:
    """Snapshot of ``world.sources`` for :attr:`FieldResult.sources`."""
    records = []
    for src in world.sources:
        current: list[complex] = []
        if src.kind == "current":
            i = complex(src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg)))
            current = [i] * n_freq
        records.append(
            SourceRecord(
                name=src.name,
                kind=src.kind,
                attached_to=src.attached_to,
                return_to=src.return_to,
                current=current,
            )
        )
    return records


_GUARDED_WORLDS: ContextVar[frozenset[int]] = ContextVar(
    "groundfield_guarded_worlds", default=frozenset()
)


@contextmanager
def _guarded_sources(world: "World") -> Iterator[None]:
    """Restore ``world.sources`` after a solve, preserving object identity.

    Shared by every solve entry point (:meth:`Engine.solve`,
    :meth:`World.solve <groundfield.world.World.solve>` and, through it,
    :func:`groundfield.api.run_simulation`), so they follow one
    source-snapshot semantics (review finding 10.72):

    * field values a backend rewrites in flight are written back onto
      the very objects the caller holds (handles returned by
      ``create_source`` stay attached);
    * list-level changes (insert, remove, replace) are undone and the
      original list object is re-bound if the attribute was replaced.

    Source models are flat (str / float / None), so a ``__dict__``
    update is a complete field-wise roll-back. A guard nested inside an
    active guard of the same world (``World.solve`` around
    ``Engine.solve``) is a no-op, so the snapshot is taken once.
    """
    active = _GUARDED_WORLDS.get()
    if id(world) in active:
        yield
        return
    source_list = world.sources
    originals = list(source_list)
    snapshots = [s.model_copy(deep=True) for s in originals]
    token = _GUARDED_WORLDS.set(active | {id(world)})
    try:
        yield
    finally:
        _GUARDED_WORLDS.reset(token)
        for original, snapshot in zip(originals, snapshots, strict=True):
            if original != snapshot:
                original.__dict__.update(snapshot.__dict__)
                original.__pydantic_fields_set__ = set(snapshot.__pydantic_fields_set__)
        if len(source_list) != len(originals) or any(
            live is not original
            for live, original in zip(source_list, originals, strict=True)
        ):
            source_list[:] = originals
        if world.sources is not source_list:
            world.sources = source_list


def _legacy_sommerfeld_property(option: str) -> property:
    """Deprecated read / write access ``Engine.sommerfeld_<option>``."""
    flat = f"sommerfeld_{option}"
    message = (
        f"Engine.{flat} is deprecated; use Engine.options['mom_sommerfeld']"
        f".{option} (or Engine.backend_options('mom_sommerfeld').{option}). "
        f"It will be removed no earlier than {DEPRECATION_REMOVAL}."
    )

    def getter(self: "Engine") -> Any:
        warn_external(message, DeprecationWarning)
        return getattr(self.backend_options("mom_sommerfeld"), option)

    def setter(self: "Engine", value: Any) -> None:
        warn_external(message, DeprecationWarning)
        current = self.backend_options("mom_sommerfeld")
        updated = MomSommerfeldOptions.model_validate(
            {**current.model_dump(), option: value}
        )
        self.options = {**self.options, "mom_sommerfeld": updated}

    return property(getter, setter, doc=f"Deprecated alias of the option ``{option}``.")


class Engine(BaseModel):
    """Configuration of the numerical kernel.

    Attributes
    ----------
    backend
        Numerical method, a name registered in
        :mod:`groundfield.solver.registry`. Built in: ``image``,
        ``image_2layer``, ``image_nlayer``, ``cim``, ``mom``,
        ``mom_sommerfeld``, ``bem``, ``fem``. See the module docstring
        above for the family overview, and ADR-0002 for the selection
        heuristic. An unknown name raises with the closest registered
        one as suggestion.
    frequencies
        Frequency list in Hz. Default ``[50.0]``.
    segment_length
        Maximum segment length used to discretise the geometry, in m.
    max_segments
        Resource limit of this engine: the largest problem size it
        solves, counted as the number of leakage segments and as the
        number of unknowns of the dense nodal system (current owners,
        nodes and branches, every conductor sub-piece included). A world
        above it raises :class:`~groundfield.diagnostics.ResourceLimitError`
        before anything of that size is allocated, in every segment
        backend and in :func:`~groundfield.solver.mutual.solve_mutual_matrix`
        / ``solve_mutual_field``. ``None`` (default) takes
        :data:`groundfield.diagnostics.HARD_LIMIT` (20 000), read at
        solve time; a positive integer raises or lowers the limit for
        this engine only (user decision Q17 of 2026.11, finding 10.06).
        The dense reaction matrix of $N$ segments takes $8N^2$ bytes
        (3.0 GiB at 20 000), the nodal system of $n$ unknowns $16n^2$
        bytes; :func:`~groundfield.diagnostics.expected_segments` predicts
        both. The limit is recorded in ``FieldResult.metadata
        ["max_segments"]``; ``fem`` meshes a volume and is not limited by
        it.
    tolerance, max_iterations
        **Deprecated, no effect** (no backend reads them). A non-default
        value warns; both fields will be removed no earlier than
        ``2027.5``.
    earth_inductive_model
        How the earth contributes to the inductive coupling between
        distributed-conductor segments (only effective when at least
        one conductor sets ``inductance_model = "neumann"``):

        - ``"perfect_mirror"`` (ADR-0004, the pre-0.13 default) — the
          earth is a perfect magnetic mirror. Cheap, frequency-independent
          inductance assembly, but its additive image is far from the
          f < 1 kHz physics of buried conductors (audit finding F5); kept
          as a fast diagnostic baseline.
        - ``"carson_series"`` (ADR-0005) — the earth has finite
          conductivity. Adds Carson 1926's per-meter earth-return
          correction $\\Delta Z_\\text{Carson}(\\omega)$, scaled by the
          geometric length of each segment pair, to the Neumann matrix
          built on Carson's own baseline, the *anti-parallel* image
          (since 2026.11; up to 0.15.0 it was added to the additive
          perfect-mirror matrix, which roughly doubled the reactance, review
          pass 10, 10.20). Asymptotically correct for long parallel
          wires over homogeneous earth; an approximation for short
          wires or layered soils. The per-metre tiling is kept only for
          near-parallel pairs of near-horizontal segments on one side of
          the surface: a pair across the surface has no anti-parallel
          baseline and couples through the field transmitted into the
          soil (:func:`~groundfield.coupling.carson.carson_mixed_mutual_correction`),
          and oblique pairs as well as segments inclined by more than
          8.1° get the Sommerfeld correction of a homogeneous earth of
          the same conductivity; at that boundary the result steps by
          the error of the per-metre model (audit round 2, A2C-1).
        - ``"sommerfeld"`` (ADR-0006, Pollaczek kernel since the
          2026-07-08 audit, WP-C1) — geometric integration of the
          σ-dependent vector-potential Green's function over the
          actual segment-pair geometry, dispatched by conductor
          side: buried pairs carry the in-soil attenuation
          $e^{-\\gamma R}$ (and with it the **positive**
          earth-return resistance $\\approx \\omega\\mu_0/8$ per
          metre in the long-wire limit), overhead pairs use the
          air-side reflection (Carson's geometry). The Neumann matrix
          carries no image: the correction holds the whole earth
          effect, so conductors in or crossing the surface plane are
          solved too (since 2026.11). Supports layered earth; see
          :attr:`sommerfeld_assembly` for the assembly (tabulated by
          default). Limits: $\\sigma \\to 0$ → free space (no image);
          $\\sigma \\to \\infty$ → screened (buried) / anti-parallel
          PEC image (overhead). Validated against Pollaczek's
          analytic $K_0$ mutual and Carson's line impedance
          (``tests/test_earth_return_benchmark.py``). **The default
          since 0.13.0** — the rigorous choice whenever a quantitative
          ``Z(f)`` matters.
    image_max_terms, image_series_tol
        Controls of the Tagg / Sunde image-charge series (maximum
        number of terms; stop tolerance on the geometric tail bound
        $|K|^{n+1}/(1-|K|)$, dimensionless). Common to every layered
        backend that evaluates the series (``image_2layer``, ``mom``,
        the delegated two-layer path of ``mom_sommerfeld``; the
        deprecated names run these) and to ``solve_mutual_*``; recorded
        in ``FieldResult.metadata["series"]``, which
        :meth:`FieldResult.potential` uses for the post-solve field
        evaluation. Defaults: :data:`DEFAULT_IMAGE_MAX_TERMS` (300) and
        :data:`DEFAULT_IMAGE_SERIES_TOL` (1e-6).
    options
        Backend-specific settings, keyed by backend name and validated
        against the backend's options model:
        ``{"mom_sommerfeld": MomSommerfeldOptions(...)}`` (quadrature
        accuracy) and ``{"fem": FemOptions(...)}`` (mesh and
        truncation). Plain mappings are accepted and validated into the
        options models; unknown option names raise with a suggestion.
        Options for a backend other than the selected one (or one it
        forwards to) emit
        :class:`~groundfield.solver.registry.BackendOptionsWarning`.
        Entries put into the mapping in place
        (``engine.options["fem"] = {...}``) bypass that validation; they
        are validated when they are read (:meth:`backend_options`,
        :meth:`solve`). The 0.15.0 flat fields
        ``sommerfeld_lambda_max_factor``, ``sommerfeld_epsabs``,
        ``sommerfeld_epsrel`` and ``sommerfeld_max_osc_panels`` are
        still accepted (constructor, JSON, attribute access and
        :meth:`model_copy`) and mapped onto
        ``options["mom_sommerfeld"]`` with a :class:`DeprecationWarning`
        (the constructor drops a value equal to the default silently, so
        engine JSON of 0.15.0 loads without warnings).
    sommerfeld_assembly
        Assembly of the Sommerfeld earth-return correction of inductive
        conductors (``earth_inductive_model="sommerfeld"``):

        - ``"tabulated"`` (default since 2026.11, user decision Q16) —
          one table of the spectral integral $S(\\rho)$ per pair of depth
          groups and frequency (cubic spline, 100 nodes), the outer
          quadrature of all segment pairs vectorised. Blocks of fewer
          than 64 sub-segment pairs and blocks whose $\\lambda$ grid
          cannot resolve their largest distance (conductors in the
          surface plane) keep the per-pair rule.
        - ``"exact"`` — every segment pair with its own quadrature and
          24 interpolation nodes of $S(\\rho)$; the default up to 0.15.0.

        On a TN network of 8 houses with an inductive PEN (78 inductive
        branches, four frequencies) the tabulated solve took 1.0 s CPU
        instead of 16.6 s, and its input impedance is within 4e-10 of
        two refined references (per-pair rule with 200 nodes, tables of
        400 nodes), the per-pair rule within 1.2e-8. Read by every
        backend that assembles the inductive branches through the shared
        pipeline (the image family, ``mom``, ``mom_sommerfeld``) and by
        ``solve_mutual_*``; see
        :func:`~groundfield.coupling.sommerfeld_inductance.build_sommerfeld_correction_matrices`,
        ``method``. The Sommerfeld fallback of ``carson_series`` for
        oblique pairs is always tabulated.

    kernel
        Green's-function kernel of the image family (``image``,
        ``image_2layer`` and ``image_nlayer`` for one or two layers),
        also used by :meth:`FieldResult.potential` and
        :func:`~groundfield.solver.mutual.solve_mutual_matrix` /
        ``solve_mutual_field``:

        - ``None`` (default since 2026.11) — the backend's preferred
          kernel: ``"line"`` for the image family, the backend's own
          kernel (``"point"``) for every other backend.
          ``FieldResult.metadata["kernel"]`` records the kernel that ran.
        - ``"line"`` — every segment is a uniform line current
          (Galerkin form of the exact line-source potential, see
          :mod:`groundfield.solver._line_kernel`). In homogeneous soil it
          is exact for straight wires at any segment length. In two-layer
          soil the Tagg/Sunde images farther than six segment lengths and
          the smooth part of the cross-layer correction stay in point
          form: a straight 10 m strip deviates by up to 0.05 % from its
          exact uniform-current resistance for ds = 0.125–2 m, not
          monotonically in ds (point kernel: up to 22 %), and the
          resistance of a rod through the layer interface changes by less
          than 0.01 % per halving of ds. Cost: a solve of 16 house grids
          (N = 1920 segments, homogeneous soil) takes 2.7 times as long
          as with ``"point"`` when numba is installed (it is used
          automatically) and 5.7 times without it; the extra time grows
          with :math:`N^2`.
        - ``"point"`` — each segment's current sits at its midpoint;
          reproduces the results of releases up to 0.15.0. Its row sums
          carry an error of about :math:`-0.23/L`, so the grounding
          resistance converges to a value 0.9–1.9 % below the exact
          uniform-current solution (review pass 10, 10.28).

        An explicit ``"line"`` on a backend without the line kernel
        raises :class:`~groundfield.solver.registry.BackendCapabilityError`.
        The request is carried in the value: an engine rebuilt from its
        JSON, from ``with_frequencies`` or from a result's provenance
        block runs exactly like the original.

    Notes
    -----
    Every field is validated on construction, on assignment
    (``engine.frequencies = []`` raises) and in :meth:`model_copy`
    (unlike pydantic's default ``model_copy``, which stores ``update``
    unchecked). Lists mutated in place are checked when the engine
    solves.

    Examples
    --------
    >>> eng = Engine(backend="fem", options={"fem": {"n_radial": 120}})
    >>> eng.backend_options().n_radial
    120
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    backend: str = Field(default="image")
    frequencies: list[float] = Field(
        default_factory=lambda: [50.0],
        description=(
            "Frequency list in Hz. **The given order is preserved** — "
            "the solver iterates over the list as-is and "
            ":attr:`FieldResult.frequencies` is the user-supplied list "
            "without sorting. A non-monotonic list (e.g. ``[5000, 50]``) "
            "is accepted but triggers a :class:`UserWarning` to surface "
            "the convention; use :meth:`Engine.with_frequencies` with "
            "``preserve_order=True`` to silence the warning."
        ),
    )
    segment_length: float = Field(default=0.5, gt=0.0)
    tolerance: float = Field(
        default=_INERT_DEFAULTS["tolerance"],
        gt=0.0,
        deprecated=_inert_message("tolerance"),
    )
    max_iterations: int = Field(
        default=int(_INERT_DEFAULTS["max_iterations"]),
        gt=0,
        deprecated=_inert_message("max_iterations"),
    )
    earth_inductive_model: EarthInductiveModel = Field(default="sommerfeld")
    image_max_terms: int = Field(
        default=DEFAULT_IMAGE_MAX_TERMS,
        gt=0,
        description=(
            "Maximum number of series terms of the Tagg / Sunde "
            "image-charge series. High layer contrast (|K| close to 1) "
            "needs more terms to bring the geometric tail bound "
            "|K|^(n+1)/(1-|K|) below image_series_tol. The default 300 "
            "covers the AP1 corner soil rho_1 = 1000 / rho_2 = 30 Ohm m "
            "(|K| = 0.94, 277 terms); the solver emits a "
            "SeriesTruncationWarning when the cap is reached. Common to "
            "every layered backend (image_2layer, mom, the two-layer path "
            "of mom_sommerfeld) and to solve_mutual_*; recorded in "
            "FieldResult.metadata['series'] for FieldResult.potential."
        ),
    )
    kernel: Literal["point", "line"] | None = Field(
        default=None,
        description=(
            "Green's-function kernel: None (default) lets the backend choose "
            "its preferred kernel ('line' for the image family, the backend's "
            "own elsewhere); 'line' (exact uniform line sources, Galerkin "
            "form) or 'point' (midpoint point sources, the kernel of releases "
            "up to 0.15.0) request one explicitly."
        ),
    )
    image_series_tol: float = Field(
        default=DEFAULT_IMAGE_SERIES_TOL,
        gt=0.0,
        description=(
            "Stop tolerance for the image-charge series: iteration "
            "ends when the geometric tail bound |K|^(n+1)/(1-|K|) "
            "falls below this value."
        ),
    )
    # The annotation states the accepted input (option models or plain
    # mappings, audit A1-06); after validation every value is the
    # backend's options model.
    options: Mapping[str, SerializeAsAny[BackendOptions] | Mapping[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Backend-specific settings keyed by backend name, validated "
            "against the backend's registered options model "
            "(mom_sommerfeld: MomSommerfeldOptions, fem: FemOptions)."
        ),
    )
    # user decision Q16 (2026-11-08): the tabulated assembly is the default.
    sommerfeld_assembly: Literal["exact", "tabulated"] = Field(
        default="tabulated",
        description=(
            "Assembly of the Sommerfeld earth-return correction of inductive "
            "conductors: 'tabulated' (default since 2026.11; one table per "
            "pair of depth groups and frequency, faster and closer to a "
            "converged reference) or 'exact' (one quadrature per segment "
            "pair, the default up to 0.15.0). See coupling."
            "sommerfeld_inductance.build_sommerfeld_correction_matrices"
            "(method=...)."
        ),
    )
    max_segments: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Resource limit of this engine: the largest number of leakage "
            "segments and of unknowns of the dense nodal system it solves; a "
            "larger world raises ResourceLimitError before allocating. None "
            "(default) takes groundfield.diagnostics.HARD_LIMIT (20 000), "
            "read at solve time."
        ),
    )

    # Deprecated flat access to the mom_sommerfeld options (0.15.0 API).
    sommerfeld_lambda_max_factor = _legacy_sommerfeld_property("lambda_max_factor")
    sommerfeld_epsabs = _legacy_sommerfeld_property("epsabs")
    sommerfeld_epsrel = _legacy_sommerfeld_property("epsrel")
    sommerfeld_max_osc_panels = _legacy_sommerfeld_property("max_osc_panels")

    # ------------------------------------------------------------------
    # Validation of backend, options and legacy fields
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: Any) -> Any:
        """Map the 0.15.0 flat ``sommerfeld_*`` fields onto ``options``.

        A flat value equal to the 0.15.0 default carries no information
        and is dropped silently, so engine JSON written by 0.15.0 loads
        without warnings. A non-default value is moved into
        ``options["mom_sommerfeld"]`` and emits a
        :class:`DeprecationWarning`.
        """
        if not isinstance(data, Mapping):
            return data
        flat = {k: data[k] for k in _LEGACY_SOMMERFELD_FIELDS if k in data}
        if not flat:
            return data
        data = {k: v for k, v in data.items() if k not in _LEGACY_SOMMERFELD_FIELDS}
        defaults = MomSommerfeldOptions()
        given = {
            _LEGACY_SOMMERFELD_FIELDS[k]: v
            for k, v in flat.items()
            if v != getattr(defaults, _LEGACY_SOMMERFELD_FIELDS[k])
        }
        if not given:
            return data
        options = dict(data.get("options") or {})
        nested = options.get("mom_sommerfeld", {})
        if isinstance(nested, BaseModel):
            nested = nested.model_dump(exclude_unset=True)
        if not isinstance(nested, Mapping):
            raise ValueError(
                "Engine.options['mom_sommerfeld'] must be a mapping or a "
                f"MomSommerfeldOptions, got {type(nested).__name__}."
            )
        clash = sorted(k for k in given if k in nested and nested[k] != given[k])
        if clash:
            raise ValueError(
                f"mom_sommerfeld option(s) {clash} given twice (flat "
                "sommerfeld_* field and Engine.options) with different values."
            )
        options["mom_sommerfeld"] = {**nested, **given}
        data["options"] = options
        names = ", ".join(f"sommerfeld_{k}" for k in sorted(given))
        warn_external(
            f"Engine field(s) {names} are deprecated; pass "
            "options={'mom_sommerfeld': {...}} (or a MomSommerfeldOptions) "
            "instead. The flat fields will be removed no earlier than "
            f"{DEPRECATION_REMOVAL}.",
            DeprecationWarning,
        )
        return data

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        """Accept registered names and aliases only (with a suggestion)."""
        _registry.get_backend(value)
        return value

    @field_validator("tolerance", "max_iterations")
    @classmethod
    def _warn_inert(cls, value: float, info: ValidationInfo) -> float:
        """Warn when an inert field carries a non-default value (10.51)."""
        if value != _INERT_DEFAULTS[info.field_name]:
            warn_external(_inert_message(info.field_name), DeprecationWarning)
        return value

    @field_validator("options", mode="before")
    @classmethod
    def _validate_options(cls, value: Any) -> Any:
        """Validate every entry against its backend's options model."""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(
                "Engine.options must be a mapping backend name -> options, "
                f"got {type(value).__name__}."
            )
        out: dict[str, BackendOptions] = {}
        for key, entry in value.items():
            canonical, options = _registry.validate_options_entry(key, entry)
            if canonical in out:
                raise ValueError(
                    f"Engine.options configures backend '{canonical}' twice "
                    f"(key {key!r} is an alias)."
                )
            out[canonical] = options
        return out

    @model_validator(mode="after")
    def _warn_foreign_options(self) -> "Engine":
        """Options for a backend that cannot run are ignored: warn."""
        if self.options:
            reachable = _registry.reachable_backends(self.backend)
            for key in self.options:
                if key not in reachable:
                    warn_external(
                        f"Engine.options['{key}'] configures backend '{key}', "
                        f"but the selected backend is '{self.backend}'; these "
                        "options are ignored.",
                        BackendOptionsWarning,
                    )
        return self

    def backend_options(self, name: str | None = None) -> BackendOptions | None:
        """Options a backend runs with: the stored entry or the defaults.

        Parameters
        ----------
        name
            Backend name or alias. Default: :attr:`backend`.

        Returns
        -------
        BackendOptions or None
            The validated options model of the backend, or ``None`` for
            a backend without backend-specific options.

        Raises
        ------
        ValueError
            If an entry put into :attr:`options` in place does not
            validate (unknown backend or option name, wrong value).
        """
        key = _registry.options_key(self.backend if name is None else name)
        model = _registry.get_backend(key).options_model
        if model is None:
            return None
        stored = self._checked_options().get(key)
        return stored if stored is not None else model()

    def _checked_options(self) -> dict[str, BackendOptions]:
        """:attr:`options` with entries mutated in place validated (A1-04)."""
        options = self.options
        checked = {
            key: entry
            for key, entry in options.items()
            if isinstance(entry, BackendOptions) and _registry.options_key(key) == key
        }
        if len(checked) == len(options):
            return checked
        return type(self)._validate_options(options)

    # ------------------------------------------------------------------
    # Copies
    # ------------------------------------------------------------------

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy of the engine with ``update`` validated like the constructor.

        pydantic's ``model_copy`` stores ``update`` without validation; a
        deprecated flat name such as ``sommerfeld_epsrel`` was then
        shadowed by its property and silently ignored (audit A1-02), and
        an empty frequency list or a raw option mapping went through
        (A1-04).

        Parameters
        ----------
        update
            Field values to change. Validated together with the unchanged
            fields; the deprecated flat ``sommerfeld_*`` names are mapped
            onto ``options["mom_sommerfeld"]`` with a
            :class:`DeprecationWarning`.
        deep
            Deep copy instead of a shallow one.

        Returns
        -------
        Engine
            The copy (the receiver is not changed).

        Raises
        ------
        ValueError
            If a key of ``update`` is not a field (with a suggestion), a
            value does not validate, or a flat name contradicts
            ``update["options"]``.

        Notes
        -----
        Only warnings that the update introduces are emitted; those of
        the unchanged fields (e.g. a non-monotonic frequency list) were
        emitted when the engine was built.
        """
        if not update:
            return super().model_copy(deep=deep)
        fields = type(self).model_fields
        unknown = [
            k for k in update if k not in fields and k not in _LEGACY_SOMMERFELD_FIELDS
        ]
        if unknown:
            key = str(unknown[0])
            raise ValueError(
                f"Engine.model_copy: {key!r} is not a field."
                f"{did_you_mean(key, fields)} Fields: {', '.join(fields)}."
            )
        changes = dict(update)
        flat = {
            _LEGACY_SOMMERFELD_FIELDS[k]: changes.pop(k)
            for k in list(changes)
            if k in _LEGACY_SOMMERFELD_FIELDS
        }
        if flat:
            changes["options"] = self._options_with_flat(
                changes.get("options", self.options), flat, explicit="options" in update
            )
        # The unchanged fields as set on this engine (not the defaults), so
        # the copy keeps pydantic's ``model_fields_set`` semantics.
        base = {name: self.__dict__[name] for name in self.model_fields_set}
        with warnings.catch_warnings(record=True) as before:
            warnings.simplefilter("always")
            type(self).model_validate(base)
        with warnings.catch_warnings(record=True) as after:
            warnings.simplefilter("always")
            copy = type(self).model_validate({**base, **changes})
        seen = {(w.category, str(w.message)) for w in before}
        for w in after:
            if (w.category, str(w.message)) not in seen:
                warn_external(str(w.message), w.category)
        return super(Engine, copy).model_copy(deep=True) if deep else copy

    def _with(self, **changes: Any) -> Self:
        """Unvalidated copy with trusted values set by the solver itself.

        For internal copies on the solve path (the resolved kernel, one
        frequency of the engine's own list): no validation and, above
        all, no ``warnings.catch_warnings``, which mutates the
        process-wide warning filters (resetting every module's "shown
        once" registry and racing with other threads).
        """
        return super().model_copy(update=changes)

    @staticmethod
    def _options_with_flat(
        options: Any, flat: dict[str, Any], *, explicit: bool
    ) -> dict[str, Any]:
        """``options`` with the flat ``sommerfeld_*`` values merged in."""
        if not isinstance(options, Mapping):
            raise ValueError(
                "Engine.options must be a mapping backend name -> options, "
                f"got {type(options).__name__}."
            )
        merged = dict(options)
        key = next(
            (k for k in merged if _registry.options_key(k) == "mom_sommerfeld"),
            "mom_sommerfeld",
        )
        entry = merged.pop(key, {})
        if isinstance(entry, BaseModel):
            entry = entry.model_dump(exclude_unset=True)
        if not isinstance(entry, Mapping):
            raise ValueError(
                "Engine.options['mom_sommerfeld'] must be a mapping or a "
                f"MomSommerfeldOptions, got {type(entry).__name__}."
            )
        clash = sorted(
            k for k in flat if explicit and k in entry and entry[k] != flat[k]
        )
        if clash:
            raise ValueError(
                f"mom_sommerfeld option(s) {clash} given twice (flat "
                "sommerfeld_* name and options) with different values."
            )
        merged["mom_sommerfeld"] = {**entry, **flat}
        names = ", ".join(f"sommerfeld_{k}" for k in sorted(flat))
        warn_external(
            f"Engine field(s) {names} are deprecated; pass "
            "options={'mom_sommerfeld': {...}} (or a MomSommerfeldOptions) "
            "instead. The flat fields will be removed no earlier than "
            f"{DEPRECATION_REMOVAL}.",
            DeprecationWarning,
        )
        return merged

    # ------------------------------------------------------------------
    # Frequencies
    # ------------------------------------------------------------------

    @field_validator("frequencies")
    @classmethod
    def _validate_frequencies(cls, value: list[float]) -> list[float]:
        """Validate ``frequencies`` and warn on non-monotonic input.

        The validator deliberately *does not sort*. Notebooks that pass
        ``[5000, 50]`` get back a result with ``frequencies == [5000, 50]``
        and the per-frequency column order in the matching order. A
        :class:`UserWarning` makes the convention visible so users who
        previously relied on an implicit sort notice it during the
        migration to ``v0.5.0``.
        """
        _check_frequency_values(value)
        # Strict-monotone-increasing detection. Use a tolerance-free
        # comparison: explicit duplicates are also flagged because they
        # silently double-evaluate the kernel at the same frequency.
        non_monotonic = any(b <= a for a, b in zip(value, value[1:], strict=False))
        if non_monotonic:
            # Deliberately *stable* message text — the per-call list
            # literal is logged at debug level so users can still find
            # the offending Engine in their notebook, but the warning
            # text itself is kept identical across every distinct
            # non-monotonic list. That lets a single
            # ``warnings.simplefilter("once",
            # EngineFrequencyOrderWarning)`` collapse a 10-engine
            # sweep down to a single notification.
            _log.debug(
                "Engine.frequencies non-monotonic: %r — preserving order.",
                value,
            )
            warnings.warn(
                "Engine.frequencies is not strictly increasing. The "
                "solver preserves the given order — result columns "
                "appear in the same order as the input. Wrap your "
                "list with sorted(set(...)) for ascending unique "
                "frequencies, or call "
                "Engine.with_frequencies(*freqs, preserve_order=True) "
                "to opt in explicitly and silence this warning.",
                EngineFrequencyOrderWarning,
                stacklevel=2,
            )
        return list(value)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    def with_frequencies(
        self,
        *frequencies: float,
        preserve_order: bool = False,
    ) -> "Engine":
        """Return a copy of this engine with a new ``frequencies`` list.

        Parameters
        ----------
        *frequencies
            One or more frequencies in Hz. Variadic to make the
            common case readable: ``engine.with_frequencies(50, 5000)``.
        preserve_order
            If ``True``, the frequencies are stored verbatim and no
            non-monotonic warning is raised even for a decreasing or
            duplicate-bearing list. This is the explicit opt-in for
            sweeps that require a specific iteration order. If ``False``
            (default) the standard :func:`_validate_frequencies` checks
            run.

        Returns
        -------
        Engine
            A new :class:`Engine` instance — the receiver is **not**
            mutated.

        Examples
        --------
        >>> eng = Engine(backend="image").with_frequencies(50, 5000,
        ...                                                preserve_order=True)
        >>> eng.frequencies
        [50.0, 5000.0]
        """
        freqs = [float(f) for f in frequencies]
        # The scalar contract holds for *both* paths. Checking it here
        # as well as in the validator keeps the error attributable to
        # ``with_frequencies()`` rather than to a nested model rebuild.
        _check_frequency_values(freqs)
        data = self.model_dump()
        data["frequencies"] = freqs
        if preserve_order:
            # Re-validate the rest, but suppress the field validator's
            # monotone warning — a deliberately ordered sweep is the
            # documented opt-in.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineFrequencyOrderWarning)
                return self.__class__.model_validate(data)
        # Default path: run the full field validation, including the
        # monotone-order warning. ``model_copy(update=...)`` would skip
        # every validator (pydantic v2 semantics), which is the
        # pre-0.14.1 defect. Re-emit any warning at the caller's frame
        # so the notebook line that built the engine is the one shown.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine = self.__class__.model_validate(data)
        for entry in caught:
            warnings.warn(entry.message, entry.category, stacklevel=2)
        return engine

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def solve(self, world: "World") -> FieldResult:
        """Run the simulation with the configured backend.

        Resolves :attr:`backend` in the registry, checks the backend's
        declared capabilities against ``world`` and runs it. When
        ``backend == "image"`` the backend is auto-forwarded to
        ``"image_2layer"`` for a
        :class:`~groundfield.soil.models.TwoLayerSoil` and to
        ``"image_nlayer"`` for a
        :class:`~groundfield.soil.models.MultiLayerSoil`, so notebooks
        written for the homogeneous case keep working when the soil
        model is replaced.

        The sources of ``world`` are snapshotted before the backend runs
        and restored afterwards onto the caller's objects, exactly as in
        :meth:`World.solve <groundfield.world.World.solve>`: solving
        never rewrites the input world, whichever entry point is used.

        Parameters
        ----------
        world : World
            The assembled world (soil, electrodes, conductors and at
            least one source) the backend should solve.

        Returns
        -------
        FieldResult
            Container with per-frequency potentials, currents, cluster
            impedances, the source snapshot and metadata: the core keys
            :data:`CORE_METADATA_KEYS` (including the provenance block)
            plus the backend-specific keys.

        Raises
        ------
        ValueError
            If ``world.soil`` is unset, ``world`` contains no electrodes,
            fails :meth:`World.check <groundfield.world.World.check>`
            (no current source, short-circuited source, zero or non-finite
            magnitude; the same for every backend and for
            :meth:`World.solve <groundfield.world.World.solve>`),
            :attr:`backend` is not registered, :attr:`frequencies` was
            emptied or given a non-finite or negative entry in place, or
            an entry put into :attr:`options` in place does not validate.
        KeyError
            If a source terminal or conductor end names no node of
            ``world`` (:meth:`World.check <groundfield.world.World.check>`).
        BackendCapabilityError
            If the backend cannot represent the world (soil layer count,
            concrete-shell electrodes, explicitly requested kernel). The
            message names the registered backends that can.
        ResourceLimitError
            If the predicted number of segments or of nodal unknowns
            exceeds :attr:`max_segments` (default
            :data:`groundfield.diagnostics.HARD_LIMIT`); raised before
            anything of that size is allocated (subclass of
            ``ValueError``).

        Warns
        -----
        BackendCapabilityWarning
            If the backend solves the world with a documented
            simplification (reduced conductor model, ignored
            ``return_to``).
        DeprecationWarning
            If :attr:`backend` is a deprecated alias.
        ExperimentalBackendWarning
            On the first solve with an experimental backend in a process.
        """
        if world.soil is None:
            raise ValueError("World has no soil model. Set one before calling solve().")
        if not world.electrodes:
            raise ValueError("World contains no electrodes.")
        # Assignments are validated; a list mutated in place is not (A1-04).
        _check_frequency_values(self.frequencies)
        # Central validation of references and excitation (10.66) for
        # every backend and entry point, a current source included: a world
        # without one solved to zeros in fem only and raised in every other
        # backend (audit round 2, A2S-05).
        world.check()

        resolved = _registry.resolve_backend(self.backend, world)
        _log.info(
            "Engine.solve: backend=%s (running %s), n_electrodes=%d, n_freq=%d",
            self.backend,
            resolved.spec.name,
            len(world.electrodes),
            len(self.frequencies),
        )
        with _guarded_sources(world):
            return self._run(world, resolved)

    def _run(
        self, world: "World", resolved: "_registry.ResolvedBackend"
    ) -> FieldResult:
        """Checks, dispatch and metadata core (see :meth:`solve`)."""
        _registry.emit_resolution_warnings(resolved)
        # The kernel request is carried in the value (audit A1-03): None
        # leaves the choice to the backend, so an engine rebuilt from JSON,
        # with_frequencies or a provenance block runs like the original.
        requested = self.kernel
        _registry.check_capabilities(
            resolved,
            world,
            kernel=requested,
            kernel_explicit=requested is not None,
        )

        # Pre-flight discretisation check (advisory, never raises).
        # Surfaces thin-wire violations, degenerate electrode sizes
        # and segment-budget overruns as warnings *before* the solve
        # spends minutes on a misconfigured world.
        from groundfield.diagnostics import _hard_limit, check_segment_resolution

        for finding in check_segment_resolution(world, self):
            warn_external(finding, UserWarning)
        # The resource limit the backend checks the problem size against
        # (Engine.max_segments or the module default; user decision Q17).
        max_segments = _hard_limit(self)

        # Snapshot before dispatch: the record and the fingerprint
        # describe the world as the caller built it, even if a backend
        # rewrites a source in flight (rolled back by _guarded_sources).
        records = _source_records(world, len(self.frequencies))
        provenance = solve_provenance(world, self)

        spec = resolved.spec
        options = self.backend_options(spec.name)
        kwargs = options.model_dump() if options is not None else {}
        # The backend sees the kernel it runs with: the explicit request,
        # else DEFAULT_KERNEL where implemented, else its own.
        effective = _registry.effective_kernel(
            spec, requested if requested is not None else DEFAULT_KERNEL
        )
        run_engine = self
        if effective is not None and effective != requested:
            run_engine = self._with(kernel=effective)
        solve_fn = spec.resolve_solve_fn()
        t0 = time.perf_counter()
        result = solve_fn(world, run_engine, **kwargs)
        solve_time_s = time.perf_counter() - t0

        if not result.sources:
            result.sources = records
        provenance["solved_at_utc"] = _utc_now()
        md = result.metadata
        md["backend"] = self.backend
        md["effective_backend"] = spec.name
        md.setdefault("kernel", effective)
        md["groundfield_version"] = provenance["groundfield_version"]
        md["frequencies"] = [float(f) for f in result.frequencies]
        md["n_segments"] = int(md.get("n_segments", len(result.point_sources)))
        md["soil_kind"] = str(world.soil.kind)
        md.setdefault("earth_inductive_model", self.earth_inductive_model)
        md["solve_time_s"] = float(solve_time_s)
        # A backend that reports the series controls it ran with wins.
        md.setdefault(
            "series",
            {
                "max_terms": int(self.image_max_terms),
                "tol": float(self.image_series_tol),
            },
        )
        md["max_segments"] = max_segments
        md["provenance"] = provenance
        return result


def _utc_now() -> str:
    """Current UTC time as ISO 8601 with a ``Z`` suffix."""
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return stamp.replace("+00:00", "Z")
