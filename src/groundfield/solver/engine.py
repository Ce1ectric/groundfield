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

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from groundfield.solver.result import FieldResult
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.world import World

__all__ = ["Engine", "Backend", "EarthInductiveModel"]

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
    frequencies: list[float] = Field(default_factory=lambda: [50.0])
    segment_length: float = Field(default=0.5, gt=0.0)
    tolerance: float = Field(default=1e-6, gt=0.0)
    max_iterations: int = Field(default=200, gt=0)
    earth_inductive_model: EarthInductiveModel = Field(default="perfect_mirror")

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
