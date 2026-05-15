"""Top-level factory functions.

This module provides the convenient, notebook-friendly API of
``groundfield``. Instead of instantiating each class manually, the
caller writes:

>>> import groundfield as gf
>>> world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
>>> rod = gf.create_electrode(world, "rod", name="g1",
...                           position=(0.0, 0.0, 0.0), length=1.5)
>>> ring = gf.create_electrode(world, "ring", name="g2",
...                            center=(10.0, 0.0, 0.8), radius=2.0)
>>> conn = gf.create_conductor(world, name="l1", start=rod, end=ring,
...                            conductor_type="bare_copper")
>>> src = gf.create_source(world, name="s1", attached_to="g1",
...                        magnitude=10.0)
>>> world.set_boundary_conditions(far_field="dirichlet")
>>> eng = gf.create_engine(backend="image", frequencies=[50.0])
>>> result = gf.run_simulation(world, eng)        # or world.solve(eng)

Conventions
-----------
- Every ``create_*`` function **registers** the new object in the
  ``World`` and returns it.
- ``start`` / ``end`` of a conductor may be electrodes, electrode
  names, or coordinate tuples.
- Names are auto-generated when omitted (``electrode_0``,
  ``conductor_0``, ``source_0``).
"""

from __future__ import annotations

from typing import Any, Union

from groundfield.boundary import BoundaryConditions
from groundfield.conductors.conductor import Conductor, ConductorType
from groundfield.geometry.electrodes import (
    Electrode,
    GridMeshElectrode,
    MeshElectrode,
    RingElectrode,
    RodElectrode,
    StripElectrode,
    _ElectrodeBase,
)
from groundfield.solver.engine import Backend, Engine
from groundfield.solver.result import FieldResult
from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    SoilModel,
    TwoLayerSoil,
)
from groundfield.sources import CurrentSource, Source, VoltageSource
from groundfield.world import World

__all__ = [
    "create_world",
    "create_electrode",
    "create_conductor",
    "create_source",
    "create_engine",
    "run_simulation",
]

# Type alias used by the factories
_PointLike = Union[tuple[float, float, float], _ElectrodeBase, str]


# ---------------------------------------------------------------------
# World
# ---------------------------------------------------------------------


def create_world(
    *,
    name: str = "world",
    soil: SoilModel | None = None,
    boundary: BoundaryConditions | None = None,
) -> World:
    """Create an empty :class:`World`.

    Parameters
    ----------
    name
        Human-readable name (used in logs and result metadata).
    soil
        Optional pre-built soil model. Can also be set later via
        ``world.soil = ...``.
    boundary
        Optional pre-built boundary configuration.
    """
    kwargs: dict[str, Any] = {"name": name}
    if soil is not None:
        kwargs["soil"] = soil
    if boundary is not None:
        kwargs["boundary"] = boundary
    return World(**kwargs)


# ---------------------------------------------------------------------
# Electrodes
# ---------------------------------------------------------------------


def _auto_name(world: World, attr: str, prefix: str) -> str:
    """Generate a unique default name ``{prefix}_{n}``."""
    existing = {item.name for item in getattr(world, attr)}
    n = 0
    while f"{prefix}_{n}" in existing:
        n += 1
    return f"{prefix}_{n}"


def create_electrode(
    world: World,
    kind: str = "rod",
    *,
    name: str | None = None,
    **params: Any,
) -> _ElectrodeBase:
    """Create an electrode and register it with ``world``.

    Parameters
    ----------
    world
        Target world that will own the new electrode.
    kind
        Geometry type: ``"rod"``, ``"ring"`` or ``"mesh"``.
    name
        Optional unique name. Auto-generated when ``None``.
    **params
        Remaining parameters are forwarded to the geometry class
        (see :mod:`groundfield.geometry.electrodes`).
    """
    if name is None:
        name = _auto_name(world, "electrodes", "electrode")

    cls_map: dict[str, type[_ElectrodeBase]] = {
        "rod": RodElectrode,
        "ring": RingElectrode,
        "strip": StripElectrode,
        "mesh": MeshElectrode,
        "grid_mesh": GridMeshElectrode,
    }
    if kind not in cls_map:
        raise ValueError(
            f"Unknown electrode kind '{kind}'. "
            f"Allowed: {sorted(cls_map.keys())}."
        )
    electrode = cls_map[kind](name=name, **params)
    world.add_electrode(electrode)
    return electrode


# ---------------------------------------------------------------------
# Conductors
# ---------------------------------------------------------------------


def _resolve_point(world: World, point: _PointLike) -> tuple[float, float, float]:
    """Resolve an electrode / name / tuple to a coordinate triple."""
    if isinstance(point, _ElectrodeBase):
        return point.connection_point
    if isinstance(point, str):
        return world.get_electrode(point).connection_point
    # Assume iterable of length 3
    p = tuple(point)
    if len(p) != 3:
        raise ValueError(
            f"Point must be (x, y, z), got {p!r} of length {len(p)}."
        )
    return (float(p[0]), float(p[1]), float(p[2]))


def create_conductor(
    world: World,
    *,
    start: _PointLike,
    end: _PointLike,
    name: str | None = None,
    conductor_type: ConductorType = "generic",
    wire_radius: float = 0.005,
    resistivity: float = 1.68e-8,
    cross_section: float | str | None = None,
    discretize_segment_length: float | None = None,
    coupling_to_soil: str = "isolated",
    inductance_model: str | None = None,
    lumped_series_resistance_ohm: float | None = None,
) -> Conductor:
    """Create a conductor and register it with ``world``.

    Parameters
    ----------
    world
        Target world.
    start, end
        Start and end point. Each may be an :class:`Electrode`, the
        name of a registered electrode, or a coordinate tuple
        ``(x, y, z)``.
    name
        Optional unique name.
    conductor_type
        Conductor type (see :data:`ConductorType`).
    wire_radius, resistivity
        Wire radius (m) and material resistivity (Ω·m).
    cross_section
        Conductor cross section in $\\mathrm{m}^2$. ``None`` (default)
        keeps the historic ideal-galvanic-short behaviour: both end
        electrodes are fused into one cluster with a common potential.
        A finite value activates the finite-impedance branch model
        with $R_\\text{ser} = \\rho_\\text{mat}\\, L / A$. The string
        ``"from_radius"`` resolves to $\\pi\\, r_\\text{wire}^2$.
    discretize_segment_length
        Maximum sub-segment length in m for the distributed-conductor
        model (see ADR-0003). ``None`` (default) keeps the conductor
        lumped (single segment, leakage only at the end electrodes).
    coupling_to_soil
        ``"isolated"`` (default) — the conductor does not exchange
        current with the soil along its length. ``"galvanic"`` — every
        segment leaks current into the soil. For an insulated cable
        keep ``"isolated"``; for buried bare copper or an exposed
        shield use ``"galvanic"``.
    inductance_model
        ``None`` (default) keeps the longitudinal-branch impedance
        purely resistive (DC behaviour). ``"neumann"`` activates the
        Neumann self- and mutual-inductance integral for every
        distributed-conductor segment pair (ADR-0004) — the system
        becomes frequency-dependent and complex per frequency.
    """
    if name is None:
        name = _auto_name(world, "conductors", "conductor")
    s = _resolve_point(world, start)
    e = _resolve_point(world, end)

    def _electrode_name(p: _PointLike) -> str | None:
        if isinstance(p, _ElectrodeBase):
            return p.name
        if isinstance(p, str):
            # Validate that the referenced electrode exists.
            world.get_electrode(p)
            return p
        return None

    cond = Conductor(
        name=name,
        start=s,
        end=e,
        start_electrode=_electrode_name(start),
        end_electrode=_electrode_name(end),
        conductor_type=conductor_type,
        wire_radius=wire_radius,
        resistivity=resistivity,
        cross_section=cross_section,
        discretize_segment_length=discretize_segment_length,
        coupling_to_soil=coupling_to_soil,
        inductance_model=inductance_model,
        lumped_series_resistance_ohm=lumped_series_resistance_ohm,
    )
    world.add_conductor(cond)
    return cond


# ---------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------


def create_source(
    world: World,
    *,
    attached_to: str | _ElectrodeBase | Conductor,
    name: str | None = None,
    kind: str = "current",
    magnitude: float = 1.0,
    phase_deg: float = 0.0,
    return_to: str | _ElectrodeBase | None = None,
) -> Source:
    """Create a source and register it with ``world``.

    Parameters
    ----------
    world
        Target world.
    attached_to
        Electrode or conductor (or its name) the source feeds into.
    name
        Optional unique name.
    kind
        ``"current"`` (default) or ``"voltage"``.
    magnitude
        Amplitude (A or V).
    phase_deg
        Phase angle in degrees.
    return_to
        Optional return / auxiliary electrode. ``None`` means return
        through the remote earth.
    """
    if name is None:
        name = _auto_name(world, "sources", "source")

    def _name_of(obj: Any) -> str:
        if isinstance(obj, str):
            return obj
        if hasattr(obj, "name"):
            return obj.name
        raise TypeError(
            "attached_to / return_to must be a string or an object with "
            f".name, got {type(obj).__name__}."
        )

    attached_name = _name_of(attached_to)
    return_name: str | None
    return_name = _name_of(return_to) if return_to is not None else None

    if kind == "current":
        src: Source = CurrentSource(
            name=name,
            attached_to=attached_name,
            return_to=return_name,
            magnitude=magnitude,
            phase_deg=phase_deg,
        )
    elif kind == "voltage":
        src = VoltageSource(
            name=name,
            attached_to=attached_name,
            return_to=return_name,
            magnitude=magnitude,
            phase_deg=phase_deg,
        )
    else:
        raise ValueError(f"Unknown source kind '{kind}'.")

    world.add_source(src)
    return src


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------


def create_engine(
    *,
    backend: Backend = "image",
    frequencies: list[float] | None = None,
    segment_length: float = 0.5,
    tolerance: float = 1e-6,
    max_iterations: int = 200,
    earth_inductive_model: str = "perfect_mirror",
) -> Engine:
    """Create a configured :class:`Engine`.

    The :class:`Engine` performs an automatic backend dispatch inside
    :meth:`Engine.solve`: when ``backend="image"`` (the default) is
    combined with a :class:`TwoLayerSoil` the call is forwarded to
    ``"image_2layer"``, and with a :class:`MultiLayerSoil` to
    ``"image_nlayer"``. Notebooks written for the homogeneous case
    therefore keep working when the soil model is replaced by a
    layered one.

    Parameters
    ----------
    backend
        Numerical method. One of ``"image"``, ``"image_2layer"``,
        ``"image_nlayer"``, ``"cim"``, ``"mom"``, ``"mom_sommerfeld"``,
        ``"bem"`` or ``"fem"``. Leaving the default ``"image"`` lets
        the engine pick the matching layered image backend
        automatically based on the world's soil model.
    frequencies
        List of frequencies in Hz. Default: ``[50.0]``.
    segment_length
        Maximum segment length used to discretise the geometry, in m.
    tolerance, max_iterations
        Convergence parameters for iterative solvers.
    earth_inductive_model
        Earth model for the inductive coupling between distributed
        conductor segments. Three values are supported:

        - ``"perfect_mirror"`` (default, ADR-0004) — perfect magnetic
          mirror, frequency-independent Neumann inductance assembly.
        - ``"carson_series"`` (ADR-0005) — Carson 1926 per-meter
          asymptotic correction × geometric length. Cheap,
          appropriate for long parallel wires over homogeneous earth.
        - ``"sommerfeld"`` (ADR-0006) — geometric integration of the
          σ-dependent vector-potential Green's function. Rigorous for
          arbitrary wire lengths and for layered earth.
    """
    return Engine(
        backend=backend,
        frequencies=frequencies if frequencies is not None else [50.0],
        segment_length=segment_length,
        tolerance=tolerance,
        max_iterations=max_iterations,
        earth_inductive_model=earth_inductive_model,
    )


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------


def run_simulation(world: World, engine: Engine) -> FieldResult:
    """Run a simulation.

    Convenience wrapper; equivalent to ``world.solve(engine)``.
    """
    return world.solve(engine)
