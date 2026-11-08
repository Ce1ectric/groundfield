"""Top-level factory functions.

This module provides the convenient, notebook-friendly API of
``groundfield``. Instead of instantiating each class manually, the
caller writes:

>>> import groundfield as gf
>>> world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
>>> rod = gf.create_electrode(world, "rod", name="g1",
...                           position=(0.0, 0.0, 0.0), length=1.5)
>>> ring = gf.create_electrode(world, "ring", name="g2",
...                            center=(10.0, 0.0, 0.8), radius=2.0)
>>> conn = gf.create_conductor(world, name="l1", start=rod, end=ring,
...                            conductor_type="bare_copper")
>>> src = gf.create_source(world, name="s1", attached_to="g1",
...                        magnitude=10.0)
>>> j = gf.create_junction(world, name="J1", position=(10.0, 0.0, 0.5))
>>> eng = gf.create_engine(backend="image", frequencies=[50.0])
>>> result = gf.run_simulation(world, eng)        # or world.solve(eng)

Conventions
-----------
- Every ``create_*`` function **registers** the new object in the
  ``World`` and returns it.
- ``start`` / ``end`` of a conductor may be electrodes, junctions,
  their names, or coordinate tuples.
- Names are auto-generated when omitted (``electrode_0``,
  ``conductor_0``, ``source_0``, ``junction_0``).
- Coordinates are in metres with ``z`` pointing downwards (``z > 0`` is
  in the soil).

Static typing
-------------
:func:`create_electrode` has one ``@overload`` per electrode kind, so an
IDE shows the parameters of the chosen kind and a type checker reports a
misspelled kind or keyword and infers the concrete return type
(``create_electrode(w, "ring", ...)`` is a :class:`RingElectrode`). The
string options are ``Literal`` types (:data:`ElectrodeKind`,
:data:`SourceKind`, :data:`KernelKind`, and
:data:`~groundfield.solver.engine.Backend`,
:data:`~groundfield.solver.engine.EarthInductiveModel`,
:data:`~groundfield.conductors.conductor.ConductorType`,
:data:`~groundfield.conductors.conductor.SoilCoupling`,
:data:`~groundfield.conductors.conductor.InductanceModel` where they are
defined). A kind chosen at run time (a plain ``str``) needs a
``typing.cast`` to :data:`ElectrodeKind` — or build the class directly
(``gf.RodElectrode(...)`` plus ``world.add_electrode``). At run time,
an unknown kind, keyword or electrode name raises with the closest
valid name ("did you mean …?").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt

from groundfield.boundary import BoundaryConditions
from groundfield.conductors.conductor import (
    Conductor,
    ConductorType,
    InductanceModel,
    SoilCoupling,
)
from groundfield.geometry.electrodes import (
    Electrode,
    GridMeshElectrode,
    MeshElectrode,
    PolylineElectrode,
    RingElectrode,
    RodElectrode,
    StarElectrode,
    StripElectrode,
    _ElectrodeBase,
)
from groundfield.geometry.junction import Junction
from groundfield.solver.engine import Backend, EarthInductiveModel, Engine
from groundfield.solver.result import FieldResult
from groundfield.soil.models import SoilModel
from groundfield.sources import CurrentSource, Source, VoltageSource
from groundfield.utils.suggest import did_you_mean, suggest
from groundfield.world import World

__all__ = [
    "create_world",
    "create_electrode",
    "create_conductor",
    "create_junction",
    "create_source",
    "create_engine",
    "run_simulation",
    "ElectrodeKind",
    "SourceKind",
    "KernelKind",
    "PointLike",
    "ConductorEnd",
]

#: Electrode geometry accepted by :func:`create_electrode`.
ElectrodeKind: TypeAlias = Literal[
    "rod", "ring", "strip", "polyline", "star", "mesh", "grid_mesh"
]

#: Source type accepted by :func:`create_source`.
SourceKind: TypeAlias = Literal["current", "voltage"]

#: Field / reaction-matrix kernel accepted by :func:`create_engine`:
#: ``"point"`` (point source per segment) or ``"line"`` (exact uniform
#: line source per segment).
KernelKind: TypeAlias = Literal["point", "line"]

#: A point ``(x, y, z)`` in m: a tuple, a list or a numpy array.
PointLike: TypeAlias = Sequence[float] | npt.NDArray[Any]

#: A conductor end: an electrode or junction, the name of a registered
#: electrode or junction, or a point ``(x, y, z)`` in m.
ConductorEnd: TypeAlias = PointLike | Electrode | Junction | str

_ELECTRODE_CLASSES: dict[str, type[_ElectrodeBase]] = {
    "rod": RodElectrode,
    "ring": RingElectrode,
    "strip": StripElectrode,
    "polyline": PolylineElectrode,
    "star": StarElectrode,
    "mesh": MeshElectrode,
    "grid_mesh": GridMeshElectrode,
}

_SOURCE_CLASSES: dict[str, type[CurrentSource] | type[VoltageSource]] = {
    "current": CurrentSource,
    "voltage": VoltageSource,
}


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
        Optional pre-built boundary configuration. No backend reads it
        (see :class:`~groundfield.BoundaryConditions`).

    Returns
    -------
    World
        The new, empty world.
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


def _parameters_of(cls: type[_ElectrodeBase]) -> list[str]:
    """Keyword parameters :func:`create_electrode` accepts for ``cls``.

    Required parameters first, then the optional ones, each in field order.
    """
    names = [f for f in cls.model_fields if f not in ("kind", "name")]
    return sorted(names, key=lambda f: not cls.model_fields[f].is_required())


def _unknown_keyword_message(kind: str, unknown: list[str], allowed: list[str]) -> str:
    parts = []
    for key in unknown:
        close = suggest(key, allowed, n=1)
        owners = [
            k for k, cls in _ELECTRODE_CLASSES.items() if key in _parameters_of(cls)
        ]
        if close:
            parts.append(f"'{key}' (did you mean '{close[0]}'?)")
        elif owners:
            kinds = ", ".join(f"'{k}'" for k in owners)
            parts.append(f"'{key}' (a parameter of kind {kinds})")
        else:
            parts.append(f"'{key}'")
    noun = "argument" if len(unknown) == 1 else "arguments"
    return (
        f"create_electrode(kind={kind!r}) got an unexpected keyword {noun} "
        f"{', '.join(parts)}. Parameters of kind {kind!r}: name, "
        f"{', '.join(allowed)}."
    )


@overload
def create_electrode(
    world: World,
    kind: Literal["rod"] = ...,
    *,
    name: str | None = ...,
    position: PointLike,
    length: float,
    wire_radius: float = ...,
    concrete_shell_coefficient_ohm_m: float = ...,
) -> RodElectrode: ...


@overload
def create_electrode(
    world: World,
    kind: Literal["ring"],
    *,
    name: str | None = ...,
    center: PointLike,
    radius: float,
    wire_radius: float = ...,
    concrete_shell_coefficient_ohm_m: float = ...,
) -> RingElectrode: ...


@overload
def create_electrode(
    world: World,
    kind: Literal["strip"],
    *,
    name: str | None = ...,
    start: PointLike,
    end: PointLike,
    wire_radius: float = ...,
    concrete_shell_coefficient_ohm_m: float = ...,
) -> StripElectrode: ...


@overload
def create_electrode(
    world: World,
    kind: Literal["polyline"],
    *,
    name: str | None = ...,
    vertices: Sequence[PointLike] | npt.NDArray[Any],
    closed: bool = ...,
    wire_radius: float = ...,
    concrete_shell_coefficient_ohm_m: float = ...,
) -> PolylineElectrode: ...


@overload
def create_electrode(
    world: World,
    kind: Literal["star"],
    *,
    name: str | None = ...,
    center: PointLike,
    n_arms: int,
    arm_length: float,
    orientation_deg: float = ...,
    wire_radius: float = ...,
    concrete_shell_coefficient_ohm_m: float = ...,
) -> StarElectrode: ...


@overload
def create_electrode(
    world: World,
    kind: Literal["mesh"],
    *,
    name: str | None = ...,
    corner: PointLike,
    size: Sequence[float] | npt.NDArray[Any],
    spacing: float = ...,
    wire_radius: float = ...,
    concrete_shell_coefficient_ohm_m: float = ...,
) -> MeshElectrode: ...


@overload
def create_electrode(
    world: World,
    kind: Literal["grid_mesh"],
    *,
    name: str | None = ...,
    corner: PointLike,
    size: Sequence[float] | npt.NDArray[Any],
    n_x: int,
    n_y: int,
    wire_radius: float = ...,
    concrete_shell_coefficient_ohm_m: float = ...,
) -> GridMeshElectrode: ...


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
        Geometry type (:data:`ElectrodeKind`). The keyword parameters of
        each kind (lengths in m, angles in degrees, ``z > 0`` in the
        soil; ``wire_radius`` defaults to 0.005 m and the concrete-shell
        coefficient ``concrete_shell_coefficient_ohm_m`` (Ω·m, ADR-0012)
        to 0 for every kind):

        - ``"rod"`` — ``position`` (top end), ``length``.
        - ``"ring"`` — ``center``, ``radius``.
        - ``"strip"`` — ``start``, ``end``.
        - ``"polyline"`` — ``vertices``, ``closed`` (default ``True``).
        - ``"star"`` — ``center``, ``n_arms``, ``arm_length``,
          ``orientation_deg`` (default 0).
        - ``"mesh"`` — ``corner``, ``size`` ``(x, y)``, ``spacing``
          (default 1 m).
        - ``"grid_mesh"`` — ``corner``, ``size`` ``(x, y)``, ``n_x``,
          ``n_y``.

        See :mod:`groundfield.geometry.electrodes` for the geometry of
        each class.
    name
        Optional unique name. Auto-generated (``electrode_<n>``) when
        ``None``.
    **params
        The parameters of ``kind`` listed above.

    Returns
    -------
    Electrode
        The registered electrode, of the class that belongs to ``kind``
        (:class:`RodElectrode` for ``"rod"``, :class:`RingElectrode` for
        ``"ring"``, …); type checkers infer the concrete class from the
        overloads.

    Raises
    ------
    ValueError
        Unknown ``kind`` (with the closest valid kind), or a parameter
        value the electrode model rejects.
    TypeError
        A keyword that ``kind`` does not have (with the closest valid
        keyword, or the kinds that do have it).
    """
    cls = _ELECTRODE_CLASSES.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown electrode kind {kind!r}."
            f"{did_you_mean(str(kind), _ELECTRODE_CLASSES)} "
            f"Valid kinds: {', '.join(repr(k) for k in _ELECTRODE_CLASSES)}."
        )
    allowed = _parameters_of(cls)
    unknown = [key for key in params if key not in allowed]
    if unknown:
        raise TypeError(_unknown_keyword_message(kind, unknown, allowed))
    if name is None:
        name = _auto_name(world, "electrodes", "electrode")
    electrode = cls(name=name, **params)
    world.add_electrode(electrode)
    return electrode


# ---------------------------------------------------------------------
# Conductors
# ---------------------------------------------------------------------


def _get_electrode(world: World, name: str) -> Electrode:
    """``world.get_electrode`` with a "did you mean" hint on a miss."""
    try:
        return world.get_electrode(name)
    except KeyError:
        known = [e.name for e in world.electrodes]
        hint = did_you_mean(name, known, n=3)
        raise KeyError(
            f"Electrode {name!r} not found in world {world.name!r}.{hint}"
        ) from None


def _get_node(world: World, name: str) -> Electrode | Junction:
    """Electrode or junction ``name`` with a "did you mean" hint on a miss."""
    for e in world.electrodes:
        if e.name == name:
            return e
    for j in world.junctions:
        if j.name == name:
            return j
    known = [e.name for e in world.electrodes] + [j.name for j in world.junctions]
    hint = did_you_mean(name, known, n=3)
    raise KeyError(
        f"No electrode or junction {name!r} in world {world.name!r}.{hint}"
    ) from None


def _resolve_point(world: World, point: ConductorEnd) -> tuple[float, float, float]:
    """Resolve an electrode / junction / name / tuple to a coordinate triple."""
    if isinstance(point, (_ElectrodeBase, Junction)):
        return point.connection_point
    if isinstance(point, str):
        return _get_node(world, point).connection_point
    # Assume iterable of length 3
    p = tuple(point)
    if len(p) != 3:
        raise ValueError(f"Point must be (x, y, z), got {p!r} of length {len(p)}.")
    return (float(p[0]), float(p[1]), float(p[2]))


def create_conductor(
    world: World,
    *,
    start: ConductorEnd,
    end: ConductorEnd,
    name: str | None = None,
    conductor_type: ConductorType = "generic",
    wire_radius: float = 0.005,
    resistivity: float = 1.68e-8,
    cross_section: float | Literal["from_radius"] | None = None,
    discretize_segment_length: float | None = None,
    coupling_to_soil: SoilCoupling = "isolated",
    inductance_model: InductanceModel | None = None,
    lumped_series_resistance_ohm: float | None = None,
) -> Conductor:
    """Create a conductor and register it with ``world``.

    Parameters
    ----------
    world
        Target world.
    start, end
        Start and end point. Each may be an electrode or a junction
        (:func:`create_junction`), the name of a registered electrode or
        junction, or a coordinate tuple ``(x, y, z)`` in m. A conductor
        joins the network only if both ends are nodes.
    name
        Optional unique name. Auto-generated (``conductor_<n>``) when
        ``None``.
    conductor_type
        Conductor type (:data:`~groundfield.conductors.conductor.ConductorType`).
    wire_radius
        Wire radius in m.
    resistivity
        Material resistivity of the conductor in Ω·m (default: copper,
        1.68e-8 Ω·m).
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
    lumped_series_resistance_ohm
        Optional additional lumped series resistance in Ω.

    Returns
    -------
    Conductor
        The registered conductor.

    Raises
    ------
    KeyError
        ``start`` or ``end`` names an electrode or junction that does
        not exist (with the closest names).
    """
    if name is None:
        name = _auto_name(world, "conductors", "conductor")
    s = _resolve_point(world, start)
    e = _resolve_point(world, end)

    def _electrode_name(p: ConductorEnd) -> str | None:
        if isinstance(p, (_ElectrodeBase, Junction)):
            return p.name
        if isinstance(p, str):
            return _get_node(world, p).name
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


@overload
def create_source(
    world: World,
    *,
    attached_to: str | Electrode | Junction,
    name: str | None = ...,
    kind: Literal["current"] = ...,
    magnitude: float = ...,
    phase_deg: float = ...,
    return_to: str | Electrode | Junction | None = ...,
) -> CurrentSource: ...


@overload
def create_source(
    world: World,
    *,
    attached_to: str | Electrode | Junction,
    name: str | None = ...,
    kind: Literal["voltage"],
    magnitude: float = ...,
    phase_deg: float = ...,
    return_to: str | Electrode | Junction | None = ...,
) -> VoltageSource: ...


def create_source(
    world: World,
    *,
    attached_to: str | Electrode | Junction,
    name: str | None = None,
    kind: SourceKind = "current",
    magnitude: float = 1.0,
    phase_deg: float = 0.0,
    return_to: str | Electrode | Junction | None = None,
) -> Source:
    """Create a source and register it with ``world``.

    Parameters
    ----------
    world
        Target world.
    attached_to
        Node the source feeds: an electrode or a junction
        (:func:`create_junction`), or its name.
    name
        Optional unique name. Auto-generated (``source_<n>``) when
        ``None``.
    kind
        ``"current"`` (default) or ``"voltage"`` (:data:`SourceKind`).
        No backend implements voltage sources: a world containing one
        raises :class:`NotImplementedError` when solved.
    magnitude
        Amplitude in A (current source) or V (voltage source), RMS;
        finite. A source of magnitude 0 is rejected when the world is
        solved.
    phase_deg
        Phase angle in degrees; finite.
    return_to
        Optional second terminal (electrode or junction, or its name):
        the source injects $+I$ at ``attached_to`` and $-I$ at
        ``return_to`` (auxiliary electrode of a fall-of-potential
        measurement, return conductor end). ``None`` (default) returns
        the current through remote earth.

    Returns
    -------
    CurrentSource or VoltageSource
        The registered source.

    Raises
    ------
    ValueError
        Unknown ``kind`` (with the closest valid kind); ``attached_to``
        names a conductor (a conductor is not a node — put a junction
        at its end); ``attached_to`` and ``return_to`` are the same
        node; non-finite ``magnitude`` / ``phase_deg``.
    KeyError
        ``attached_to`` / ``return_to`` names no electrode or junction
        of ``world`` (with the closest names).
    """
    cls = _SOURCE_CLASSES.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown source kind {kind!r}.{did_you_mean(str(kind), _SOURCE_CLASSES)} "
            f"Valid kinds: {', '.join(repr(k) for k in _SOURCE_CLASSES)}."
        )
    if name is None:
        name = _auto_name(world, "sources", "source")

    def _name_of(obj: Any) -> str:
        if isinstance(obj, str):
            return obj
        if hasattr(obj, "name"):
            return str(obj.name)
        raise TypeError(
            "attached_to / return_to must be a string or an object with "
            f".name, got {type(obj).__name__}."
        )

    src = cls(
        name=name,
        attached_to=_name_of(attached_to),
        return_to=_name_of(return_to) if return_to is not None else None,
        magnitude=magnitude,
        phase_deg=phase_deg,
    )
    world.add_source(src)
    return src


# ---------------------------------------------------------------------
# Junctions
# ---------------------------------------------------------------------


def create_junction(
    world: World,
    *,
    position: PointLike,
    name: str | None = None,
) -> Junction:
    """Create a leakage-free junction node and register it with ``world``.

    A junction is a node of the conductor network without soil contact:
    conductors can start or end there (:func:`create_conductor`) and
    sources can attach to it (:func:`create_source`), but it injects no
    current into the soil. Its potential follows from Kirchhoff's
    current law and Ohm's law along the conductors; solved results
    report it in ``result.metadata["junction_potentials"][name]`` (V per
    frequency). Use it for cable joints and for the terminals of a
    measuring instrument (see :class:`~groundfield.geometry.junction.Junction`).

    Parameters
    ----------
    world
        Target world.
    position
        Location ``(x, y, z)`` in m (``z`` positive downwards; a point
        above the surface is allowed). Conductors ending on the junction
        use it as their end point.
    name
        Optional unique node name (electrodes and junctions share one
        namespace). Auto-generated (``junction_<n>``) when ``None``.

    Returns
    -------
    Junction
        The registered junction.

    Raises
    ------
    ValueError
        The name is already taken by an electrode or junction, or the
        position is not finite.

    Notes
    -----
    A junction must be connected through conductors to an electrode or
    to a galvanically coupled conductor; a subnetwork without soil
    contact has an undefined potential and is rejected when solved.
    Only backends that declare the ``junctions`` capability (the image
    family) accept worlds with junctions.
    """
    if name is None:
        taken = {e.name for e in world.electrodes} | {j.name for j in world.junctions}
        n = 0
        while f"junction_{n}" in taken:
            n += 1
        name = f"junction_{n}"
    p = tuple(position)
    if len(p) != 3:
        raise ValueError(f"position must be (x, y, z), got {p!r} of length {len(p)}.")
    junction = Junction(name=name, position=(float(p[0]), float(p[1]), float(p[2])))
    world.add_junction(junction)
    return junction


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------


def create_engine(
    *,
    backend: Backend = "image",
    frequencies: Sequence[float] | npt.NDArray[Any] | None = None,
    segment_length: float | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    earth_inductive_model: EarthInductiveModel | None = None,
    image_max_terms: int | None = None,
    image_series_tol: float | None = None,
    kernel: KernelKind | None = None,
    options: Mapping[str, Any] | None = None,
) -> Engine:
    """Create a configured :class:`~groundfield.Engine`.

    Only the arguments you pass are forwarded; every argument left at
    ``None`` takes the :class:`~groundfield.Engine` default, so the engine
    is the single source of its defaults and option-level warnings of the
    engine concern only options you actually set. Options without a
    parameter here (e.g. the ``sommerfeld_*`` accuracy controls) are set
    on :class:`~groundfield.Engine` directly.

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
        Numerical method (:data:`~groundfield.solver.engine.Backend`).
        Leaving the default ``"image"`` lets the engine pick the
        matching layered image backend from the world's soil model.
    frequencies
        Frequencies in Hz (order preserved). Default: ``[50.0]``.
    segment_length
        Maximum segment length used to discretise the geometry, in m.
        Default: the engine default (0.5 m).
    tolerance, max_iterations
        Convergence parameters for iterative solvers. No backend reads
        them; they are forwarded only when given.
    earth_inductive_model
        Earth model for the inductive coupling between distributed
        conductor segments, effective only when a conductor sets
        ``inductance_model="neumann"``
        (:data:`~groundfield.solver.engine.EarthInductiveModel`):
        ``"sommerfeld"`` (engine default since 0.13.0; geometric
        integration of the σ-dependent vector-potential Green's function,
        validated against Pollaczek and Carson), ``"carson_series"``
        (Carson 1926 per-metre correction × length) or
        ``"perfect_mirror"`` (frequency-independent Neumann baseline).
    image_max_terms
        Maximum number of image-charge series terms of the layered
        backends; a truncated series emits a
        :class:`~groundfield.solver.image_2layer.SeriesTruncationWarning`.
    image_series_tol
        Series stop tolerance: iteration ends when the geometric tail
        bound ``|K|^(n+1)/(1-|K|)`` falls below this value.
    kernel
        Field / reaction-matrix kernel (:data:`KernelKind`):
        ``"point"`` reproduces the results of earlier releases,
        ``"line"`` integrates each segment as a uniform line source.
        ``None`` keeps the engine default.
    options
        Backend-specific settings, keyed by backend name, e.g.
        ``{"mom_sommerfeld": {"epsabs": 1e-8}}`` or option models such as
        :class:`~groundfield.solver.MomSommerfeldOptions`; forwarded to
        :attr:`Engine.options <groundfield.Engine.options>`.

    Returns
    -------
    Engine
        The configured engine.

    Raises
    ------
    TypeError
        An unknown keyword argument.
    ValueError
        A value the engine rejects.
    """
    engine_kwargs: dict[str, Any] = {"backend": backend}
    if options is not None:
        engine_kwargs["options"] = dict(options)
    if frequencies is not None:
        # Only a numpy array is converted; anything else (list, tuple, but
        # also a mistaken str or scalar) is validated by the Engine itself.
        if isinstance(frequencies, np.ndarray):
            engine_kwargs["frequencies"] = frequencies.tolist()
        else:
            engine_kwargs["frequencies"] = frequencies
    explicit = {
        "segment_length": segment_length,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "earth_inductive_model": earth_inductive_model,
        "image_max_terms": image_max_terms,
        "image_series_tol": image_series_tol,
    }
    engine_kwargs.update({k: v for k, v in explicit.items() if v is not None})
    if kernel is not None:
        if "kernel" in Engine.model_fields:
            engine_kwargs["kernel"] = kernel
        elif kernel != "point":
            # Transitional: an engine without the ``kernel`` option only
            # has the point kernel.
            raise ValueError(
                f"kernel={kernel!r} is not available in this build; only "
                "the point kernel is."
            )
    return Engine(**engine_kwargs)


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------


def run_simulation(world: World, engine: Engine) -> FieldResult:
    """Solve ``world`` with ``engine``; identical to ``world.solve(engine)``.

    The call goes through :meth:`World.solve` and therefore has exactly
    its semantics, including the source snapshot: the world's sources
    are restored after the solve even if a backend changes them in
    flight, and the handles returned by :func:`create_source` stay
    attached to the world.

    Parameters
    ----------
    world
        The world to solve.
    engine
        The engine configuration.

    Returns
    -------
    FieldResult
        The solution.
    """
    return world.solve(engine)
