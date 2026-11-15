"""Finite-element backend (``fem``): conforming axisymmetric and 3-D models.

Since 2026.11 (user decision Q37) ``fem`` solves the volume problem of the
electrodes in the soil; the equivalent-hemisphere reduction of earlier
releases is gone. Two models share the network solve:

* the **axisymmetric** model (part A): rods on one vertical axis and
  horizontal rings centred on it, as the solid bodies they are, in any
  number of horizontal layers, bare or encased (ADR-0012), to about 1e-4;
* the **three-dimensional** model (part B, the optional ``fem3d`` extra):
  any electrodes in one or several clusters, galvanic distributed
  conductors, concrete shells per wire or resolved as a volume (user
  decision Q40); thin wires as smoothed line sources with an analytic
  radius correction, layer interfaces as partially penalised immersed
  elements on an isotropic tetrahedral mesh
  (:mod:`groundfield.solver.fem._solid`), to about 2e-3.

``FemOptions.model = "auto"`` (default) takes the axisymmetric model where
it applies and the three-dimensional one otherwise. ``fem`` is a reference
backend: the integral engines (the image family, ``mom``,
``mom_sommerfeld``) are the production engines, ``fem`` checks them with a
method that shares nothing with them but the network solve (ADR-0016).

The rest of this docstring describes the axisymmetric model; the 3-D model
is documented in :mod:`groundfield.solver.fem._solid` and the modules it
names.

Mathematical / physical model
-----------------------------
The quasi-static current field in the soil,
$$
-\\nabla\\cdot\\bigl(\\sigma(z)\\,\\nabla\\varphi\\bigr) = 0,
\\qquad \\sigma = 1/\\rho \\text{ piecewise constant in depth},
$$
with no current into the air ($\\partial_z\\varphi = 0$ at $z = 0$) and
$\\varphi \\to 0$ far away, reduces for a body of revolution to the
meridian half-plane $(s, z)$,
$$
-\\frac{1}{s}\\partial_s\\bigl(s\\,\\sigma\\,\\partial_s\\varphi\\bigr)
-\\partial_z\\bigl(\\sigma\\,\\partial_z\\varphi\\bigr) = 0 .
$$
Each electrode is a perfect conductor at its own potential $U_k$ (a
*port*): a rod is the solid cylinder of its ``wire_radius`` with flat end
faces, a ring the solid torus of its ``wire_radius``. An encased electrode
(``concrete_shell_coefficient_ohm_m`` $= C > 0$) keeps its metal at $U_k$
and leaks through its shell surface $r_b$ (the ``wire_radius``) with the
radial resistance $C$ per unit length, i.e. the surface conductance
$1/(2\\pi r_b C)$ — Sunde's cylindrical shell, the per-length coefficient
of ADR-0012 applied locally, so the current may redistribute along the
electrode.

Discretisation
--------------
Linear triangles on a graded tensor grid in $(s, z)$ whose lines include
every electrode surface, every rod end face and every layer interface
(:mod:`~groundfield.solver.fem._mesh`); the circular cross section of a
ring wire is resolved by an O-grid block conforming to the grid, and an
interface that passes through such a block is cut into its elements. The
truncation box lies ``r_far_factor`` characteristic lengths away and
carries the exact Dirichlet-to-Neumann condition of the monopole far
field (:mod:`~groundfield.solver.fem._assembly`). The port conductance
matrix $G$ is the Schur complement of the stiffness matrix onto the ports;
it is computed at two nested refinement levels and extrapolated to zero
mesh size with the observed second order (Richardson). The difference to
the finer level is reported as an error estimate.

Network and results
-------------------
The port resistance matrix $R = G^{-1}$ enters the shared pipeline of the
integral backends as the multi-port matrix
(:mod:`~groundfield.solver.fem._network`): conductors (lumped, isolated
distributed chains, inductive branches with their earth-return models,
the skin effect), junctions, current sources with ``return_to`` and
voltage sources are honoured as there, at every frequency of the engine
(the soil part is quasi-static). The result carries point sources along
the electrodes with the model's leakage distribution, so
:meth:`FieldResult.potential <groundfield.solver.result.FieldResult.potential>`
and the touch-voltage helpers work as for the other backends (homogeneous
and two-layer soil).

Interfaces for the 3-D model
----------------------------
A model of the soil delivers an
:class:`~groundfield.solver.fem._ports.EarthPorts` (port conductance
matrix, display segments with their leakage per port volt, metadata);
:func:`~groundfield.solver.fem._network.solve_network` does the rest.
See ADR-0016.

References
----------
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover, ch. 2 and §4.4 (concrete shell).
- Givoli, D. (1992). *Numerical Methods for Problems in Infinite
  Domains*, Elsevier — Dirichlet-to-Neumann truncation; the monopole map
  is its lowest mode.
- Strang, G., & Fix, G. (2008). *An Analysis of the Finite Element
  Method*, 2nd ed. — minimum-energy principle and the one-sided
  convergence of the conductance.
- Richardson, L. F. (1911). The approximate arithmetical solution by
  finite differences of physical problems. *Phil. Trans. R. Soc. A* 210.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from groundfield.solver.fem._legacy import (
    FemReductionWarning,
    equivalent_hemisphere_radius,
)
from groundfield.solver.result import FieldResult

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["FemReductionWarning", "equivalent_hemisphere_radius", "solve_fem"]

_FAR_FIELDS = ("robin", "dirichlet")
_MODELS = ("auto", "axisymmetric", "3d")


def _warn_deprecated_mesh_arguments(**values: object) -> None:
    """The mesh arguments of the hemisphere mesh (0.15.0) have no effect."""
    given = [name for name, value in values.items() if value is not None]
    if not given:
        return
    from groundfield.solver.registry import DEPRECATION_REMOVAL, warn_external

    warn_external(
        f"solve_fem({', '.join(f'{n}=...' for n in given)}): the mesh arguments of "
        "the equivalent-hemisphere mesh have no effect since 2026.11 (user "
        "decision Q37). The conforming mesh is refined with refinement=... "
        "(FemOptions.refinement). They will be removed no earlier than "
        f"{DEPRECATION_REMOVAL}.",
        DeprecationWarning,
    )


def _galvanic_conductors(world: "World") -> list[str]:
    """Names of the galvanic distributed conductors of ``world``."""
    return [
        c.name
        for c in world.conductors
        if getattr(c, "is_distributed", False)
        and getattr(c, "coupling_to_soil", "isolated") == "galvanic"
    ]


def _reject_galvanic_conductors(world: "World", engine: "Engine") -> None:
    """A galvanic distributed conductor leaks along a line the model cannot hold."""
    leaking = _galvanic_conductors(world)
    if not leaking:
        return
    from groundfield.solver.fem._guards import remedy
    from groundfield.solver.registry import BackendCapabilityError

    raise BackendCapabilityError(
        "Backend 'fem' cannot solve this world: conductor(s) "
        f"{', '.join(repr(n) for n in leaking)} (coupling_to_soil='galvanic', "
        "distributed) leak into the soil along their route, which the "
        "axisymmetric model cannot represent. " + remedy(world, engine)
    )


def solve_fem(
    world: "World",
    engine: "Engine",
    *,
    refinement: int = 0,
    extrapolate: bool = True,
    r_far_factor: float = 1000.0,
    far_field: str = "robin",
    max_nodes: int = 2_000_000,
    max_nodes_3d: int = 2_000_000,
    model: str = "auto",
    source_radius: float | None = None,
    growth: float = 0.5,
    concrete_resistivity: float | None = None,
    _concrete_mesh: float | None = None,
    n_radial: int | None = None,
    n_axial: int | None = None,
    z_far_factor: float | None = None,
) -> FieldResult:
    """Conforming finite-element solve (see the module docstring).

    Parameters
    ----------
    world
        World to solve, any horizontally layered soil, bare or encased
        electrodes (see ``model``).
    engine
        Engine; its ``frequencies``, ``earth_inductive_model`` (conductor
        branches) and ``segment_length`` (the display segments that carry
        the leakage distribution of the result) are used.
    refinement
        Refinement level of the coarser of the two meshes (axisymmetric
        model: each level bisects every interval of the graded mesh lines,
        about four times the nodes; 3-D model: about eight times). ``0``
        (default) gives a port resistance within about 1e-4 of the
        converged value for coaxial rods and rings (3.3e-4 for a ring wire
        within 1.05 radii of an interface to a more conductive layer).
    extrapolate
        Solve at ``refinement`` and ``refinement + 1`` and extrapolate the
        port conductance matrix with the second order (default). ``False``
        solves the level ``refinement`` alone; its resistance usually lies
        below the converged value (see *Notes*) and
        ``metadata["fem_error_estimate"]`` is ``NaN`` (one level has no
        estimate).
    r_far_factor
        Size of the truncation box in characteristic lengths (the
        electrodes' size, or the depth of the deepest interface times the
        layer contrast if larger); dimensionless, default 1000.
    far_field
        ``"robin"`` (default): the exact Dirichlet-to-Neumann map of the
        monopole far field on the box. ``"dirichlet"``: the box grounded,
        which biases the resistance low by about $\\rho_n/(2\\pi R)$ (for
        truncation studies).
    max_nodes
        Resource limit of the axisymmetric model: the node count of the
        finer mesh, estimated before anything is assembled (the estimate
        errs high). Default 2 million: the solve needs about 4 GB per
        million nodes of the finer mesh, so about 8 GB at the limit.
    max_nodes_3d
        Resource limit of the three-dimensional model: the node count of
        its finer mesh (3.0 ... 3.4 GB per million nodes), predicted before
        meshing and counted before the finer level is built.
    model
        ``"auto"`` (default): the axisymmetric model where it applies (rods
        on one axis, rings around it, no galvanic distributed conductor),
        the three-dimensional model otherwise; ``"axisymmetric"`` refuses
        every other world; ``"3d"`` forces the three-dimensional model.
    source_radius
        3-D model: radius in m of the disc over which each wire's leakage
        is smoothed; ``None`` (default) takes 0.06 m (at least 2.2 bare
        wire radii, at most what the depth of the wires and their gap to
        the interfaces allow).
    growth
        3-D model: growth of the element size per metre away from the
        wires (default 0.5; smaller is finer and slower).
    concrete_resistivity
        3-D model: ``None`` (default) applies every encased wire's own
        shell coefficient; a resistivity in Ω·m resolves the concrete as
        square prisms of conformal radius ``wire_radius`` around the
        encased (axis-aligned, straight) wires, shared where they meet
        (user decision Q40).
    _concrete_mesh
        Private, for the overlap study of user decision Q40 (see
        :func:`~groundfield.solver.fem._solid.solid_earth`).
    n_radial, n_axial, z_far_factor
        Deprecated: the mesh controls of the equivalent-hemisphere mesh
        (0.15.0); a value has no effect and warns.

    Returns
    -------
    FieldResult
        Electrode potentials and currents (every electrode an
        equipotential port), point sources with the leakage distribution of
        the model, and ``metadata`` with ``fem_port_resistance`` (the
        extrapolated port resistance matrix in Ω, ports in world order,
        ``fem_ports``), ``fem_port_resistance_levels`` (per mesh level),
        ``fem_error_estimate`` (largest relative difference of a diagonal
        entry between the extrapolated and the finer level; ``NaN`` with
        ``extrapolate=False``), ``fem_model`` (``"axisymmetric"`` or
        ``"3d"``: the model that ran), ``fem_mesh`` (size and time per
        level) and ``fem_r_far`` (box size in m).

    Raises
    ------
    BackendCapabilityError
        With ``model="axisymmetric"``: if the world is not axisymmetric (an
        electrode other than a rod or a ring, a rod or ring off the common
        axis by more than 1e-3 of the smallest wire radius or gap, a
        galvanic distributed conductor), its electrodes touch, a ring wire
        reaches within 5 % of its radius to the soil surface, or a ring has
        no room for its mesh block (within about a wire radius, diagonally,
        of a rod's end edge); ``model="auto"`` solves these worlds with the
        three-dimensional model.
        3-D model: a wire too close to the soil surface or an interface for
        its smoothed source, resolved concrete around a wire that is not
        straight along x, y or z. The message names the backends that solve
        the world. A subclass of ``ValueError`` and ``NotImplementedError``.
    ImportError
        If the world needs the three-dimensional model and the ``fem3d``
        extra (scikit-fem, pyamg) is not installed.
    ValueError
        If an electrode or a galvanic distributed conductor lies at or
        above the soil surface (as every backend), or the world fails
        :meth:`World.check <groundfield.world.World.check>`.
    groundfield.diagnostics.ResourceLimitError
        If the finer mesh would exceed ``max_nodes`` nodes (the 3-D model
        checks a prediction before meshing and the exact count before the
        finer level).
    TypeError
        If ``world.soil`` is no soil model.

    Warns
    -----
    MixedCoordinatesWarning
        For a world that mixes local and projected coordinates (user
        decision D5), as every backend.
    DeprecationWarning
        For ``n_radial``, ``n_axial`` or ``z_far_factor``.

    Notes
    -----
    Without extrapolation the resistance of the axisymmetric model
    converges at second order and, in every case tested away from
    interfaces, from below; this is not guaranteed (the O-grid of a ring
    is not nested between levels), and next to an interface cut into a
    ring's mesh block it did not hold before that block was refined (audit
    W4A-01). The error estimate bounds the error of the extrapolated value
    while both levels lie on the same side of the converged value and the
    finer level's error is at most 1/2.5 of the coarser level's (about
    1/3.9 is observed for rods and rings).
    """
    from groundfield.soil.models import HomogeneousSoil, MultiLayerSoil, TwoLayerSoil
    from groundfield.solver import _frame

    _warn_deprecated_mesh_arguments(
        n_radial=n_radial, n_axial=n_axial, z_far_factor=z_far_factor
    )
    if far_field not in _FAR_FIELDS:
        raise ValueError(
            "far_field must be 'robin' (monopole DtN) or 'dirichlet' "
            f"(grounded truncation box), got {far_field!r}."
        )
    if int(refinement) != refinement or refinement < 0:
        raise ValueError(f"refinement must be an integer >= 0, got {refinement!r}.")
    if not r_far_factor > 1.0:
        raise ValueError(f"r_far_factor must be > 1, got {r_far_factor!r}.")
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'fem' supports HomogeneousSoil, TwoLayerSoil, "
            f"and MultiLayerSoil. Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")
    if model not in _MODELS:
        raise ValueError(
            f"model must be one of {', '.join(repr(m) for m in _MODELS)}, got {model!r}."
        )
    if source_radius is not None and not (
        math.isfinite(float(source_radius)) and float(source_radius) > 0.0
    ):
        raise ValueError(
            f"source_radius must be a positive length in m, got {source_radius!r}."
        )
    if not (math.isfinite(float(growth)) and 0.05 <= float(growth) <= 2.0):
        raise ValueError(f"growth must lie in [0.05, 2], got {growth!r}.")
    if concrete_resistivity is not None and not (
        math.isfinite(float(concrete_resistivity)) and float(concrete_resistivity) > 0.0
    ):
        raise ValueError(
            "concrete_resistivity must be a positive resistivity in Ohm m or None, "
            f"got {concrete_resistivity!r}."
        )
    frame = _frame.LocalFrame.of(world)
    with frame.active():
        result = _solve(
            frame.world,
            engine,
            refinement=int(refinement),
            extrapolate=bool(extrapolate),
            r_far_factor=float(r_far_factor),
            far_field=far_field,
            max_nodes=int(max_nodes),
            model=model,
            solid={
                "source_radius_m": None
                if source_radius is None
                else float(source_radius),
                "growth": float(growth),
                "concrete_resistivity": None
                if concrete_resistivity is None
                else float(concrete_resistivity),
                "concrete_mesh": _concrete_mesh,
                "max_nodes_3d": int(max_nodes_3d),
            },
        )
    return frame.to_user(result)


def _is_axisymmetric(world: "World", engine: "Engine") -> bool:
    """Whether the axisymmetric model accepts ``world`` (no galvanic conductor).

    Runs the classification of the axisymmetric model without building the
    refusal's remedy text, including the room of every ring's mesh block
    (audit W4A-03: a ring without it ended ``model="auto"`` in a bare
    ``ValueError``).
    """
    from groundfield.solver.fem._axisymmetric import NotAxisymmetric, classify

    del engine
    if _galvanic_conductors(world):
        return False
    try:
        classify(world)
    except NotAxisymmetric:
        return False
    return True


def _solve(
    world: "World", engine: "Engine", *, model: str, solid: dict, **options
) -> FieldResult:
    """Body of :func:`solve_fem`, in the frame of ``world``."""
    from groundfield.solver._kernels import DEFAULT_KERNEL
    from groundfield.solver._layered import as_layer_stack
    from groundfield.solver.fem._axisymmetric import axisymmetric_earth
    from groundfield.solver.fem._guards import reject_electrodes_out_of_the_soil
    from groundfield.solver.fem._network import port_topology, solve_network

    reject_electrodes_out_of_the_soil(world)  # audit D2A-06, R2A-07
    world.check()  # the errors of every backend before any mesh is built
    three_d = model == "3d" or (model == "auto" and not _is_axisymmetric(world, engine))
    topology = None
    if three_d:
        from groundfield.solver.fem._solid import require_extra, solid_earth

        require_extra()
        disc, topo = port_topology(world)
        topology = (disc, topo)
        extra = disc.segments[len(world.electrodes) :]
        solid_options = dict(options)
        solid_options["max_nodes"] = solid["max_nodes_3d"]
        rest = {k: v for k, v in solid.items() if k != "max_nodes_3d"}
        earth = solid_earth(
            world, engine, extra_segments=extra, **solid_options, **rest
        )
        if not options["extrapolate"]:
            # one level has no estimate (audit W4A-02), as in the
            # axisymmetric model
            earth.metadata["fem_error_estimate"] = math.nan
        approximation = (
            "thin wires as smoothed line sources with an analytic radius "
            "correction in three-dimensional finite elements"
        )
    else:
        _reject_galvanic_conductors(world, engine)
        if (
            solid["concrete_resistivity"] is not None
            or solid["concrete_mesh"] is not None
        ):
            raise ValueError(
                "concrete_resistivity resolves the concrete in the three-dimensional "
                "model; set FemOptions.model='3d' (the axisymmetric model applies "
                "the shell coefficient on the electrode surface)."
            )
        earth = axisymmetric_earth(world, engine, **options)
        approximation = "none: conforming axisymmetric finite elements"
    stack = as_layer_stack(world.soil)
    metadata = {
        "world_name": world.name,
        "n_layers": int(stack.n_layers),
        "rhos": [float(r) for r in stack.rhos],
        "h": [float(h) for h in stack.h],
        "segment_length": float(engine.segment_length),
        "stub": False,
        "approximation": approximation,
    }
    kernel = engine.kernel if engine.kernel is not None else DEFAULT_KERNEL
    return solve_network(
        world, engine, earth, metadata=metadata, kernel=kernel, topology=topology
    )
