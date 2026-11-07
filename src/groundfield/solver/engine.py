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
convergence $|K|^n$. ``image_nlayer`` routes a layered soil with one or
two layers to ``image`` / ``image_2layer`` and rejects three or more
layers. ``cim`` evaluates the same image series for $n \\le 2$ (its
complex-image fit is not used on any solve path, see
:mod:`groundfield.solver.cim`).

*Integral equation.* ``mom`` solves for the segment currents by point
matching (collocation) with the closed-form kernels above;
``mom_sommerfeld`` does the same with direct numerical Sommerfeld
quadrature of the layered Green's function and is the only integral
backend for $n \\ge 3$ layers. ``bem`` is the collocation solver on the
``cim`` kernel.

*Volume PDE.* ``fem`` is the axisymmetric finite-element solver with
the equivalent-hemisphere reduction described in
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
from typing import TYPE_CHECKING, Any, Literal

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

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.world import World

__all__ = [
    "CORE_METADATA_KEYS",
    "DEFAULT_IMAGE_MAX_TERMS",
    "DEFAULT_IMAGE_SERIES_TOL",
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
          correction $\\Delta Z_\\text{Carson}(\\omega)$ on top of the
          perfect-mirror Neumann matrix, scaled by the geometric
          length of each segment-pair. Asymptotically correct for
          long parallel wires over homogeneous earth; an approximation
          for short wires or layered soils.
        - ``"sommerfeld"`` (ADR-0006, Pollaczek kernel since the
          2026-07-08 audit, WP-C1) — geometric integration of the
          σ-dependent vector-potential Green's function over the
          actual segment-pair geometry, dispatched by conductor
          side: buried pairs carry the in-soil attenuation
          $e^{-\\gamma R}$ (and with it the **positive**
          earth-return resistance $\\approx \\omega\\mu_0/8$ per
          metre in the long-wire limit), overhead pairs use the
          air-side reflection (Carson's geometry). Supports layered
          earth. Limits: $\\sigma \\to 0$ → free space (no image);
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
        backend that evaluates the series (``image_2layer``,
        ``image_nlayer``, ``mom``, ``cim``, ``bem``, the delegated
        two-layer path of ``mom_sommerfeld``) and recorded in
        ``FieldResult.metadata["series"]`` for the post-solve field
        evaluation. Defaults: :data:`DEFAULT_IMAGE_MAX_TERMS` (300) and
        :data:`DEFAULT_IMAGE_SERIES_TOL` (1e-6).
    options
        Backend-specific settings, keyed by backend name and validated
        against the backend's options model:
        ``{"mom_sommerfeld": MomSommerfeldOptions(...)}`` (quadrature
        accuracy) and ``{"fem": FemOptions(...)}`` (mesh and
        truncation). Plain mappings are accepted and validated; unknown
        option names raise with a suggestion. Options for a backend
        other than the selected one (or one it forwards to) emit
        :class:`~groundfield.solver.registry.BackendOptionsWarning`.
        The 0.15.0 flat fields ``sommerfeld_lambda_max_factor``,
        ``sommerfeld_epsabs``, ``sommerfeld_epsrel`` and
        ``sommerfeld_max_osc_panels`` are still accepted (constructor
        and JSON) and mapped onto ``options["mom_sommerfeld"]``; a
        non-default value emits a :class:`DeprecationWarning`.

    Examples
    --------
    >>> eng = Engine(backend="fem", options={"fem": {"n_radial": 120}})
    >>> eng.backend_options().n_radial
    120
    """

    model_config = ConfigDict(extra="forbid")

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
            "every layered backend (image_2layer, image_nlayer, mom, "
            "cim, bem, mom_sommerfeld) and recorded in "
            "FieldResult.metadata['series']."
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
    options: dict[str, SerializeAsAny[BackendOptions]] = Field(
        default_factory=dict,
        description=(
            "Backend-specific settings keyed by backend name, validated "
            "against the backend's registered options model "
            "(mom_sommerfeld: MomSommerfeldOptions, fem: FemOptions)."
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
        """
        key = _registry.options_key(self.backend if name is None else name)
        model = _registry.get_backend(key).options_model
        if model is None:
            return None
        stored = self.options.get(key)
        return stored if stored is not None else model()

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
            or :attr:`backend` is not registered.
        BackendCapabilityError
            If the backend cannot represent the world (soil layer count,
            concrete-shell electrodes, explicitly requested kernel). The
            message names the registered backends that can.

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
        # ``Engine.kernel`` is added by the line-kernel work (10.28); until
        # then no kernel is requested and the backends use their own.
        kernel = getattr(self, "kernel", None)
        _registry.check_capabilities(
            resolved,
            world,
            kernel=kernel,
            kernel_explicit="kernel" in self.model_fields_set,
        )

        # Pre-flight discretisation check (advisory, never raises).
        # Surfaces thin-wire violations, degenerate electrode sizes
        # and segment-budget overruns as warnings *before* the solve
        # spends minutes on a misconfigured world.
        from groundfield.diagnostics import check_segment_resolution

        for finding in check_segment_resolution(world, self):
            warn_external(finding, UserWarning)

        # Snapshot before dispatch: the record and the fingerprint
        # describe the world as the caller built it, even if a backend
        # rewrites a source in flight (rolled back by _guarded_sources).
        records = _source_records(world, len(self.frequencies))
        provenance = solve_provenance(world, self)

        spec = resolved.spec
        options = self.backend_options(spec.name)
        kwargs = options.model_dump() if options is not None else {}
        solve_fn = spec.resolve_solve_fn()
        t0 = time.perf_counter()
        result = solve_fn(world, self, **kwargs)
        solve_time_s = time.perf_counter() - t0

        if not result.sources:
            result.sources = records
        provenance["solved_at_utc"] = _utc_now()
        md = result.metadata
        md["backend"] = self.backend
        md["effective_backend"] = spec.name
        md.setdefault("kernel", _registry.effective_kernel(spec, kernel))
        md["groundfield_version"] = provenance["groundfield_version"]
        md["frequencies"] = [float(f) for f in result.frequencies]
        md["n_segments"] = int(md.get("n_segments", len(result.point_sources)))
        md["soil_kind"] = str(world.soil.kind)
        md.setdefault("earth_inductive_model", self.earth_inductive_model)
        md["solve_time_s"] = float(solve_time_s)
        md["series"] = {
            "max_terms": int(self.image_max_terms),
            "tol": float(self.image_series_tol),
        }
        md["provenance"] = provenance
        return result


def _utc_now() -> str:
    """Current UTC time as ISO 8601 with a ``Z`` suffix."""
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    return stamp.replace("+00:00", "Z")
