"""Numerical computation core (``Engine``).

An ``Engine`` bundles **everything that is purely numerical**: the
chosen backend (image, MoM, FEM, ...), the frequency list, mesh
resolution and convergence tolerances. It is deliberately separated
from the ``World``, so that the same physical configuration can be
run against different backends.

The available backends fall into three families:

*Closed-form / image-charge.*
``image`` is the homogeneous-soil image-charge sum (closed form,
uniform current per unit length, cheapest backend). ``image_2layer``
implements the Tagg / Sunde geometric series for 2-layer soil with
convergence $|K|^n$ and is auto-selected for a
:class:`TwoLayerSoil`. ``image_nlayer`` is the general n-layer
image-series expansion of $\\Gamma_1(\\lambda)$ (see
:mod:`groundfield.solver.image_nlayer`); it reduces to ``image`` for
``n = 1`` and to ``image_2layer`` for ``n = 2``. ``cim`` is the
Complex-Image-Method approximation of the layered Green's function
via matrix-pencil fit (see :mod:`groundfield.solver.cim`); it stays
closed-form and its cost is independent of the layer count once
fitted.

*Integral equation.* ``mom`` is the Galerkin Method-of-Moments using
the closed-form layered kernels above (Tagg / Sunde for ``n = 2``,
homogeneous for ``n = 1``). ``mom_sommerfeld`` is the Galerkin MoM
with direct numerical Sommerfeld quadrature of the layered Green's
function — slow but methodologically independent and used as the
reference engine in the layered cross-check. ``bem`` is the
Boundary-Element collocation solver using the CIM kernel.

*Volume PDE.* ``fem`` is the axisymmetric finite-element solver
(volume form) with the equivalent-hemisphere reduction described in
:mod:`groundfield.solver.fem`; it is the only volume-PDE engine in
the suite and serves as a third-line cross-check.

Notes
-----
:meth:`Engine.solve` performs an automatic dispatch when
``backend="image"`` is requested: the call is forwarded to
``"image_2layer"`` if the world holds a :class:`TwoLayerSoil`, and to
``"image_nlayer"`` if it holds a :class:`MultiLayerSoil`. Notebooks
written for the homogeneous case therefore keep working when the soil
is replaced by a layered one.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from groundfield.solver.result import FieldResult
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.world import World

__all__ = [
    "Engine",
    "Backend",
    "EarthInductiveModel",
    "EngineFrequencyOrderWarning",
]


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
    the same (fifth 2026-05-13 audit pass).
    """

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
    "perfect_mirror",   # ADR-0004 default — sigma_earth -> infinity
    "carson_series",    # ADR-0005 — Carson 1926 per-m asymptotic correction
    "sommerfeld",       # ADR-0006 — geometric Sommerfeld kernel integration
]

_log = get_logger(__name__)


class Engine(BaseModel):
    """Configuration of the numerical kernel.

    Attributes
    ----------
    backend
        Numerical method. One of ``image``, ``image_2layer``,
        ``image_nlayer``, ``cim``, ``mom``, ``mom_sommerfeld``,
        ``bem``, ``fem``. See the module docstring above for the
        family overview, and ADR-0002 for the selection heuristic.
    frequencies
        Frequency list in Hz. Default ``[50.0]``.
    segment_length
        Maximum segment length used to discretise the geometry, in m.
    tolerance
        Relative convergence threshold for iterative solvers.
    max_iterations
        Maximum iterations for iterative solvers.
    earth_inductive_model
        How the earth contributes to the inductive coupling between
        distributed-conductor segments (only effective when at least
        one conductor sets ``inductance_model = "neumann"``):

        - ``"perfect_mirror"`` (default, ADR-0004) — the earth is a
          perfect magnetic mirror. Cheap, frequency-independent
          inductance assembly. Bit-exact reproduction of all ADR-0004
          tests.
        - ``"carson_series"`` (ADR-0005) — the earth has finite
          conductivity. Adds Carson 1926's per-meter earth-return
          correction $\\Delta Z_\\text{Carson}(\\omega)$ on top of the
          perfect-mirror Neumann matrix, scaled by the geometric
          length of each segment-pair. Asymptotically correct for
          long parallel wires over homogeneous earth; an approximation
          for short wires or layered soils.
        - ``"sommerfeld"`` (ADR-0006) — geometric integration of the
          σ-dependent vector-potential Green's function over the
          actual segment-pair geometry. Rigorous for arbitrary wire
          lengths and orientations, and supports layered earth
          natively (Pollaczek/Wait kernel). Reduces to
          ``perfect_mirror`` at $\\sigma \\to \\infty$ and to free
          space at $\\sigma \\to 0$; converges to ``carson_series``
          on the cluster-impedance level for long parallel wires
          over homogeneous earth.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Backend = Field(default="image")
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
    tolerance: float = Field(default=1e-6, gt=0.0)
    max_iterations: int = Field(default=200, gt=0)
    earth_inductive_model: EarthInductiveModel = Field(default="perfect_mirror")

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
        if not value:
            raise ValueError("Engine.frequencies must not be empty.")
        for f in value:
            # DC (``f == 0``) is a legitimate operating point for
            # quasi-static grounding studies — reject only negatives
            # and non-finite values.
            if not (f >= 0.0):
                raise ValueError(
                    "Engine.frequencies must be non-negative, got "
                    f"{f}."
                )
            if f != f:  # NaN guard
                raise ValueError(
                    "Engine.frequencies must be finite, got NaN."
                )
        # Strict-monotone-increasing detection. Use a tolerance-free
        # comparison: explicit duplicates are also flagged because they
        # silently double-evaluate the kernel at the same frequency.
        non_monotonic = any(b <= a for a, b in zip(value, value[1:]))
        if non_monotonic:
            # Deliberately *stable* message text — the per-call list
            # literal is logged at debug level so users can still find
            # the offending Engine in their notebook, but the warning
            # text itself is kept identical across every distinct
            # non-monotonic list. That lets a single
            # ``warnings.simplefilter("once",
            # EngineFrequencyOrderWarning)`` collapse a 10-engine
            # sweep down to a single notification (fifth 2026-05-13
            # audit pass).
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
        if preserve_order:
            # Bypass the validator's monotone-check by using
            # ``model_construct`` for the field. We still want type /
            # positivity checks, so do them explicitly here.
            if not freqs:
                raise ValueError(
                    "with_frequencies(): at least one frequency required."
                )
            for f in freqs:
                if not (f >= 0.0):
                    raise ValueError(
                        "with_frequencies(): frequencies must be "
                        f"non-negative, got {f}."
                    )
            data = self.model_dump()
            data["frequencies"] = freqs
            # Re-validate the rest, but inject the preserved list
            # directly so the field-validator's monotone-warning is
            # silenced.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineFrequencyOrderWarning)
                return self.__class__.model_validate(data)
        return self.model_copy(update={"frequencies": freqs})

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def solve(self, world: "World") -> FieldResult:
        """Run the simulation with the configured backend."""
        if world.soil is None:
            raise ValueError(
                "World has no soil model. Set one before calling solve()."
            )
        if not world.electrodes:
            raise ValueError("World contains no electrodes.")

        _log.info(
            "Engine.solve: backend=%s, n_electrodes=%d, n_freq=%d",
            self.backend,
            len(world.electrodes),
            len(self.frequencies),
        )

        # Auto-forwarding: ``backend="image"`` transparently picks
        # ``image_2layer`` for a 2-layer soil and ``image_nlayer`` for
        # a multilayer soil. Notebooks therefore do not need to change
        # the backend string when the soil model changes.
        from groundfield.soil.models import MultiLayerSoil, TwoLayerSoil

        effective_backend = self.backend
        if effective_backend == "image":
            if isinstance(world.soil, TwoLayerSoil):
                effective_backend = "image_2layer"
                _log.info(
                    "Engine.solve: TwoLayerSoil detected, "
                    "switching automatically to 'image_2layer'."
                )
            elif isinstance(world.soil, MultiLayerSoil):
                effective_backend = "image_nlayer"
                _log.info(
                    "Engine.solve: MultiLayerSoil detected, "
                    "switching automatically to 'image_nlayer'."
                )

        if effective_backend == "image":
            from groundfield.solver.image import solve_image

            return solve_image(world, self)

        if effective_backend == "image_2layer":
            from groundfield.solver.image_2layer import solve_image_2layer

            return solve_image_2layer(world, self)

        if effective_backend == "image_nlayer":
            from groundfield.solver.image_nlayer import solve_image_nlayer

            return solve_image_nlayer(world, self)

        if effective_backend == "cim":
            from groundfield.solver.cim import solve_cim

            return solve_cim(world, self)

        if effective_backend == "mom":
            from groundfield.solver.mom import solve_mom

            return solve_mom(world, self)

        if effective_backend == "mom_sommerfeld":
            from groundfield.solver.mom_sommerfeld import solve_mom_sommerfeld

            return solve_mom_sommerfeld(world, self)

        if effective_backend == "bem":
            from groundfield.solver.bem import solve_bem

            return solve_bem(world, self)

        if effective_backend == "fem":
            from groundfield.solver.fem import solve_fem

            return solve_fem(world, self)

        raise ValueError(f"Unknown backend '{self.backend}'.")

    # ------------------------------------------------------------------
    # Stub for backends that are not implemented yet
    # ------------------------------------------------------------------

    def _stub_result(self, world: "World") -> FieldResult:
        """Placeholder result for backends that are not implemented yet."""
        _log.warning(
            "Backend '%s' is not implemented yet — returning stub.",
            self.backend,
        )
        n_freq = len(self.frequencies)
        potentials = {e.name: [complex(1.0, 0.0)] * n_freq for e in world.electrodes}
        currents = {e.name: [complex(0.0, 0.0)] * n_freq for e in world.electrodes}
        return FieldResult(
            backend=self.backend,
            frequencies=list(self.frequencies),
            electrode_potentials=potentials,
            electrode_currents=currents,
            metadata={
                "stub": True,
                "world_name": world.name,
                "n_conductors": len(world.conductors),
                "n_sources": len(world.sources),
            },
        )
