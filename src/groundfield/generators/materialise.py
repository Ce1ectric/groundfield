r"""Materialise a :class:`~groundfield.generators.plan.NetworkPlan` into a world.

This module is the **single** place where generated networks become
:class:`groundfield.World` objects (review finding 10.78). Both front-ends
(:class:`~groundfield.generators.tn_network.TnNetworkGenerator` and
:class:`~groundfield.generators.layout.LvNetworkLayout`) and
:meth:`GroundingSystemSpec.build_at
<groundfield.generators.grounding.GroundingSystemSpec.build_at>` call it.
It owns, in one function each:

* :func:`materialise_site` — the electrodes of a grounding site and
  the bonds to its anchor;
* :func:`materialise_electrode` — one electrode, including the
  foundation geometry (rectangle or building outline, its owners) and
  the concrete shell of ADR-0012;
* :func:`materialise_junction` — a PEN cable joint (a leakage-free
  :class:`~groundfield.geometry.junction.Junction` node);
* :func:`materialise_branch` — a conductor between two nodes (PEN
  sections, service drops, measurement leads);
* :func:`materialise_plan` — everything, in a fixed order.

No random numbers are drawn here: every value of a plan is concrete.

Order of construction
---------------------
``materialise_plan`` builds sites, measurement sites, junctions,
instrument terminals, branches, measurement leads and sources, each list
in plan order. The order fixes the order of ``world.electrodes`` /
``world.junctions`` / ``world.conductors`` (and therefore the canonical
world JSON), not the physics.

Concrete shell (ADR-0012)
-------------------------
A foundation encased in concrete of resistivity :math:`\rho_c` and
thickness :math:`t` around a wire of radius :math:`r_a` is seen by the
soil through the outer shell radius :math:`r_b = r_a + t`. The radial
shell adds the per-length resistance coefficient

.. math::

   C = \frac{\rho_c}{2\pi}\,\ln\frac{r_b}{r_a}\quad[\Omega\,\mathrm{m}],

which every wire of an encased foundation carries as
``concrete_shell_coefficient_ohm_m``, with the radius :math:`r_b`; the
solver adds :math:`C/\Delta s` to the diagonal of each of its segments,
i.e. the lumped resistance :math:`R_\text{shell} = C / L` in series with an
electrode of length :math:`L` under a uniform leakage current. Both
``concrete_model`` values use this one mechanism (review findings
10.12–10.14). The cross-braces of a mesh foundation are encased like its
perimeter: a foundation earth electrode per DIN 18014 lies in the concrete
together with its connections (user decision Q20 of 2026.11; up to then
the braces kept the bare radius :math:`r_a` and no coefficient). Without a
shell every wire keeps :math:`r_a`. Rods and other electrodes of a site
are never encased.

Owners of a foundation
----------------------
The image family solves for one uniform leakage current per electrode
(an *owner*). A foundation's owners depend only on its structure and on
``concrete_model`` — never on its orientation, on whether the outline is
given as a rectangle or as a polygon, or on whether a shell is configured
(audit A2G-04, review finding 10.39):

* a mesh foundation (cross-braces) has one owner per wire line, for both
  ``concrete_model`` values. One owner over perimeter and braces puts the
  same current density on both and is +1.6 / +2.5 / +3.2 % high for
  2 × 2 / 3 × 3 / 4 × 4 cells (10 / 20 / 30 m, line kernel, 1 m segments)
  against the equipotential reference of the same segments (one owner
  per segment; with the point kernel identical to ``mom``); one owner per
  wire is +0.3 / +0.5 / +0.6 %. Against ``mom`` itself (point kernel) one
  owner per wire is +1.1 % with the default line kernel and +0.3 / +0.5 /
  +0.5 % with the point kernel;
* a ``style="ring"`` rectangle has one owner per side with
  ``"distributed"`` at every angle, also without a shell; with
  ``"lumped"`` it is one owner (one
  :class:`~groundfield.geometry.electrodes.GridMeshElectrode`, exactly the
  lumped series resistance :math:`C/L_\text{perim}` of the shell) when
  axis-aligned and one owner per side when rotated. For a perimeter one
  owner and one per side differ by at most 0.06 % up to aspect ratio 4
  (0.3 % at 10 to 50). (One closed polyline would make the rotated lumped
  ring one owner too, but the touch-voltage envelope of
  :mod:`groundfield.postprocess.safety` does not sample inside a closed
  polyline by default and would lose the interior point that decides the
  envelope of rings of 20 m and more.)
* a building outline (``polygon_xy_m``) is one closed polyline (one
  owner) for both values.

A foundation of several wires is bonded to its first wire, the anchor:
the local :math:`-y` side of a rectangle, which starts at the local
:math:`(-d_x/2, -d_y/2)` corner (the connection point of incoming
conductors).
"""

from __future__ import annotations

import math
from typing import Any, Optional

from groundfield.api import (
    create_conductor,
    create_electrode,
    create_junction,
    create_source,
    create_world,
)
from groundfield.conductors.conductor import Conductor
from groundfield.geometry.electrodes import concrete_shell_coefficient
from groundfield.generators.electrode_specs import (
    ElectrodeSpec,
    FoundationElectrodeSpec,
    RingElectrodeSpec,
    RodElectrodeSpec,
    StripElectrodeSpec,
)
from groundfield.generators.plan import (
    NetworkPlan,
    PlannedBranch,
    PlannedJunction,
    PlannedSite,
)
from groundfield.soil.models import HomogeneousSoil
from groundfield.world import World

__all__ = [
    "MIN_POLYGON_EDGE_M",
    "clean_polygon",
    "concrete_shell",
    "materialise_branch",
    "materialise_electrode",
    "materialise_junction",
    "materialise_plan",
    "materialise_site",
    "site_anchor_point",
]

MIN_POLYGON_EDGE_M = 0.5
"""Shortest polygon edge (m) kept in a foundation along a building outline.

Shorter edges are merged away: a segment shorter than the
(concrete-inflated) wire radius would trip the solver's thin-wire guard,
and OSM footprints routinely carry centimetre-scale edges without
physical meaning for the strip foundation."""


# ---------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------


def clean_polygon(
    polygon_xy: "tuple[tuple[float, float], ...] | list[tuple[float, float]]",
    *,
    min_edge_m: float = MIN_POLYGON_EDGE_M,
) -> list[tuple[float, float]]:
    """Drop duplicate and sub-``min_edge_m`` vertices from a closed ring.

    Walks the ring once and keeps a vertex only when it is at least
    ``min_edge_m`` away from the last kept one; finally drops trailing
    vertices whose closing edge (last → first) would be too short. An
    explicitly repeated closing vertex is removed first.

    Parameters
    ----------
    polygon_xy
        Ring vertices ``(x, y)`` in m.
    min_edge_m
        Shortest edge in m that is kept.

    Returns
    -------
    list of tuple
        The cleaned, open ring.
    """
    pts = [(float(x), float(y)) for x, y in polygon_xy]
    if len(pts) > 1 and math.dist(pts[0], pts[-1]) < 1e-9:
        pts = pts[:-1]
    kept: list[tuple[float, float]] = []
    for p in pts:
        if not kept or math.dist(kept[-1], p) >= min_edge_m:
            kept.append(p)
    while len(kept) >= 3 and math.dist(kept[-1], kept[0]) < min_edge_m:
        kept.pop()
    return kept


def concrete_shell(
    spec: FoundationElectrodeSpec, *, encased_length_m: float
) -> tuple[float, float, float]:
    r"""Effective radius, lumped resistance and coefficient of a shell.

    Parameters
    ----------
    spec
        Resolved foundation spec (concrete values only).
    encased_length_m
        Total length :math:`L` in m of the encased wire: every wire of the
        foundation — the perimeter of a ring or of a building outline,
        the perimeter plus the cross-braces of a mesh (user decision Q20).

    Returns
    -------
    effective_radius_m : float
        :math:`r_b = r_a + t` in m when a shell is configured, else the
        bare wire radius :math:`r_a`.
    shell_total_ohm : float
        :math:`R_\text{shell} = C / L` in Ω (0 without shell): the shell
        resistance of the whole foundation under a uniform leakage
        current, equal to the per-wire shells :math:`C / L_j` in
        parallel. Informational; the solver sees the coefficient.
    shell_coefficient_ohm_m : float
        :math:`C = \rho_c/(2\pi)\,\ln(r_b/r_a)` in Ω·m (0 without shell),
        :func:`~groundfield.geometry.electrodes.concrete_shell_coefficient`.

    Notes
    -----
    Up to the user decision Q20 of 2026.11 the braces of a mesh were bare
    and the length was the perimeter (keyword ``perimeter_m``).
    """
    if spec.concrete_rho_ohm_m is None:
        return float(spec.wire_radius_m), 0.0, 0.0
    rho_c = float(spec.concrete_rho_ohm_m)
    t = float(spec.concrete_thickness_m)
    if rho_c <= 0.0 or t <= 0.0:
        return float(spec.wire_radius_m), 0.0, 0.0
    r_a = float(spec.wire_radius_m)
    coefficient = concrete_shell_coefficient(rho_c, r_a, t)
    length = float(encased_length_m)
    r_shell = coefficient / length if length > 0.0 else 0.0
    return r_a + t, float(r_shell), float(coefficient)


# ---------------------------------------------------------------------
# Electrodes and sites
# ---------------------------------------------------------------------


def materialise_electrode(
    world: World,
    spec: ElectrodeSpec,
    *,
    centre_xy: tuple[float, float],
    name: str,
    bond_conductor_type: str = "bare_copper",
) -> None:
    """Create one resolved electrode spec in ``world``.

    Parameters
    ----------
    world
        Target world.
    spec
        Resolved electrode spec (concrete values only).
    centre_xy
        Electrode reference point ``(x, y)`` in m (site position plus
        the spec's ``offset_xy_m``).
    name
        Electrode name; a foundation built from several primitives uses
        it for its anchor primitive.
    bond_conductor_type
        Conductor type of the internal bonds of a multi-primitive
        foundation.
    """
    ex, ey = centre_xy
    if isinstance(spec, RodElectrodeSpec):
        create_electrode(
            world,
            "rod",
            name=name,
            position=(ex, ey, float(spec.depth_m)),
            length=float(spec.length_m),
            wire_radius=spec.wire_radius_m,
        )
        return
    if isinstance(spec, RingElectrodeSpec):
        create_electrode(
            world,
            "ring",
            name=name,
            center=(ex, ey, float(spec.depth_m)),
            radius=float(spec.radius_m),
            wire_radius=spec.wire_radius_m,
        )
        return
    if isinstance(spec, StripElectrodeSpec):
        length = float(spec.length_m)
        angle = math.radians(float(spec.orientation_deg))
        depth = float(spec.depth_m)
        sx = ex - 0.5 * length * math.cos(angle)
        sy = ey - 0.5 * length * math.sin(angle)
        tx = ex + 0.5 * length * math.cos(angle)
        ty = ey + 0.5 * length * math.sin(angle)
        create_electrode(
            world,
            "strip",
            name=name,
            start=(sx, sy, depth),
            end=(tx, ty, depth),
            wire_radius=spec.wire_radius_m,
        )
        return
    if isinstance(spec, FoundationElectrodeSpec):
        _materialise_foundation(
            world,
            spec,
            centre_xy=(ex, ey),
            name=name,
            bond_conductor_type=bond_conductor_type,
        )
        return
    raise TypeError(
        f"materialise_electrode: unknown electrode spec {type(spec).__name__}."
    )


def _materialise_foundation(
    world: World,
    spec: FoundationElectrodeSpec,
    *,
    centre_xy: tuple[float, float],
    name: str,
    bond_conductor_type: str,
) -> None:
    """Foundation electrode: building outline or rectangle (ring or mesh).

    The owners follow the module rule (see *Owners of a foundation*): a
    mesh is one owner per wire line; a ring is one owner per side, except
    an axis-aligned ``"lumped"`` ring (one grid mesh); neither depends on
    whether a shell is configured. The concrete shell (ADR-0012) is the
    same for both ``concrete_model`` values: every wire of the foundation,
    the cross-braces of a mesh included (user decision Q20, DIN 18014),
    gets the radius :math:`r_b` and the coefficient :math:`C`.
    """
    depth = float(spec.depth_m)
    if spec.polygon_xy_m is not None:
        _materialise_polygon_foundation(
            world, spec, centre_xy=centre_xy, depth=depth, name=name
        )
        return
    if spec.size_xy_m is not None:
        dx, dy = spec.size_xy_m
    else:
        size = float(spec.size_m)
        dx, dy = size, size
    # ``style="ring"`` is the perimeter only: one mesh cell.
    if spec.style == "ring":
        nx, ny = 1, 1
    else:
        nx, ny = spec.n_x, spec.n_y
    # Every wire is encased: ny + 1 lines of length dx, nx + 1 of length dy.
    effective_radius, _, shell_coefficient = concrete_shell(
        spec, encased_length_m=(ny + 1) * dx + (nx + 1) * dy
    )

    # A rectangle turned by a multiple of 90 deg is axis-aligned: realise
    # it with swapped sides (and mesh counts), so the same rectangle always
    # gets the same discretisation whichever way it was described (10.39).
    angle_deg, dx, dy, nx, ny = _canonical_right_angle(
        spec.orientation_deg or 0.0, dx, dy, nx, ny
    )
    braced = nx > 1 or ny > 1
    if angle_deg == 0.0 and not braced and spec.concrete_model == "lumped":
        create_electrode(
            world,
            "grid_mesh",
            name=name,
            corner=(centre_xy[0] - dx / 2.0, centre_xy[1] - dy / 2.0, depth),
            size=(dx, dy),
            n_x=1,
            n_y=1,
            wire_radius=effective_radius,
            concrete_shell_coefficient_ohm_m=shell_coefficient,
        )
        return
    _materialise_foundation_wires(
        world,
        centre_xy=centre_xy,
        depth=depth,
        dx=dx,
        dy=dy,
        n_x=nx,
        n_y=ny,
        angle_deg=angle_deg,
        name=name,
        wire_radius=effective_radius,
        concrete_shell_coefficient=shell_coefficient,
        bond_conductor_type=bond_conductor_type,
    )


_RIGHT_ANGLE_TOL_DEG = 1e-9
"""Angles closer than this (degrees) to a multiple of 90 count as one."""


def _canonical_right_angle(
    angle_deg: float, dx: float, dy: float, nx: int, ny: int
) -> tuple[float, float, float, int, int]:
    """Map a rectangle turned by k·90° onto the equivalent 0° rectangle.

    Parameters
    ----------
    angle_deg
        Rotation of the local ``+x`` axis in degrees.
    dx, dy
        Side lengths in m along local ``x`` / ``y``.
    nx, ny
        Mesh cells along local ``x`` / ``y``.

    Returns
    -------
    tuple
        ``(angle_deg, dx, dy, nx, ny)``. For an angle within
        :data:`_RIGHT_ANGLE_TOL_DEG` of an even multiple of 90° the angle
        becomes ``0.0``; for an odd multiple it becomes ``0.0`` with the
        sides and mesh counts swapped. Any other angle is returned
        unchanged.
    """
    quarter_turns = round(angle_deg / 90.0)
    if abs(angle_deg - 90.0 * quarter_turns) > _RIGHT_ANGLE_TOL_DEG:
        return angle_deg, dx, dy, nx, ny
    if quarter_turns % 2:
        return 0.0, dy, dx, ny, nx
    return 0.0, dx, dy, nx, ny


def _materialise_polygon_foundation(
    world: World,
    spec: FoundationElectrodeSpec,
    *,
    centre_xy: tuple[float, float],
    depth: float,
    name: str,
) -> None:
    """Foundation along the building outline (one closed polyline)."""
    cx, cy = centre_xy
    verts = clean_polygon(spec.polygon_xy_m, min_edge_m=MIN_POLYGON_EDGE_M)
    if len(verts) < 3:
        raise ValueError(
            f"Foundation {name!r}: the polygon outline collapses to fewer than "
            "3 vertices after removing edges shorter than MIN_POLYGON_EDGE_M = "
            f"{MIN_POLYGON_EDGE_M:g} m. Use an outline at least "
            f"{MIN_POLYGON_EDGE_M:g} m wide (GroundingSystemSpec.with_footprint "
            "widens thin building outlines), or outline='ombr'."
        )
    perimeter = 0.0
    for k in range(len(verts)):
        x0, y0 = verts[k]
        x1, y1 = verts[(k + 1) % len(verts)]
        perimeter += math.hypot(x1 - x0, y1 - y0)
    effective_radius, _, shell_coefficient = concrete_shell(
        spec, encased_length_m=perimeter
    )
    create_electrode(
        world,
        "polyline",
        name=name,
        vertices=[(cx + vx, cy + vy, depth) for vx, vy in verts],
        closed=True,
        wire_radius=effective_radius,
        concrete_shell_coefficient_ohm_m=shell_coefficient,
    )


def _bond_to_anchor(
    world: World, name: str, wires: list[Any], bond_conductor_type: str
) -> None:
    """Bond every wire of a foundation to its first wire (ideal shorts).

    ``wires`` are the electrode objects (no name lookup in the world).
    """
    anchor = wires[0]
    for j, wire in enumerate(wires[1:], start=1):
        create_conductor(
            world,
            name=f"{name}_bond_{j}",
            start=anchor,
            end=wire,
            conductor_type=bond_conductor_type,
        )


def _materialise_foundation_wires(
    world: World,
    *,
    centre_xy: tuple[float, float],
    depth: float,
    dx: float,
    dy: float,
    n_x: int,
    n_y: int,
    angle_deg: float,
    name: str,
    wire_radius: float,
    concrete_shell_coefficient: float,
    bond_conductor_type: str,
) -> None:
    r"""Rectangular foundation as one owner per wire line, bonded.

    ``n_y + 1`` longitudinal wires (along the local :math:`+x` axis,
    rotated by ``angle_deg``) and ``n_x + 1`` transverse wires; with
    ``n_x = n_y = 1`` exactly the four perimeter edges. The first wire
    (local :math:`-y` edge) carries ``name`` and is the anchor; the
    others are ``f"{name}_w{j}"``, bonded to the anchor by
    ``f"{name}_bond_{j}"``. Every wire, perimeter and cross-braces alike,
    carries ``wire_radius`` (the shell's outer radius :math:`r_b` when a
    shell is configured, else the bare :math:`r_a`) and the concrete-shell
    coefficient ``concrete_shell_coefficient`` (user decision Q20: the
    braces lie in the concrete too, DIN 18014; up to then they kept
    :math:`r_a` and no coefficient, review finding 10.13).

    Notes
    -----
    A wire that is crossed by other wires of the mesh is built as an open
    :class:`~groundfield.geometry.electrodes.PolylineElectrode` with a
    vertex at every crossing; a wire without crossings (every wire of a
    ``style="ring"`` foundation) stays a
    :class:`~groundfield.geometry.electrodes.StripElectrode`. Each
    polyline edge (one mesh span) is discretised on its own, so crossings
    fall on segment ends — the rule of the axis-aligned
    ``GridMeshElectrode`` — and segment midpoints of crossing wires never
    coincide (10.39). Each wire is one uniform-current owner, at every
    orientation (audit A2G-04).
    """
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    cx, cy = centre_xy
    half_x = 0.5 * dx
    half_y = 0.5 * dy

    def to_world(lx: float, ly: float) -> tuple[float, float, float]:
        return cos_a * lx - sin_a * ly + cx, sin_a * lx + cos_a * ly + cy, depth

    xs = [-half_x + k * (dx / n_x) for k in range(n_x + 1)]
    ys = [-half_y + k * (dy / n_y) for k in range(n_y + 1)]
    # Local vertices of every wire: longitudinal wires at each y, through
    # every x; transverse wires at each x, through every y.
    wires = [[(x, y) for x in xs] for y in ys] + [[(x, y) for y in ys] for x in xs]
    created: list[Any] = []
    for j, local in enumerate(wires):
        sub_name = name if j == 0 else f"{name}_w{j}"
        vertices = [to_world(lx, ly) for lx, ly in local]
        if len(vertices) == 2:
            wire = create_electrode(
                world,
                "strip",
                name=sub_name,
                start=vertices[0],
                end=vertices[1],
                wire_radius=wire_radius,
                concrete_shell_coefficient_ohm_m=concrete_shell_coefficient,
            )
        else:
            wire = create_electrode(
                world,
                "polyline",
                name=sub_name,
                vertices=vertices,
                closed=False,
                wire_radius=wire_radius,
                concrete_shell_coefficient_ohm_m=concrete_shell_coefficient,
            )
        created.append(wire)
    _bond_to_anchor(world, name, created, bond_conductor_type)


def site_anchor_point(site: PlannedSite) -> Optional[tuple[float, float, float]]:
    """Connection point in m of the anchor electrode of ``site``.

    The site is materialised into a scratch world, so the point is exactly
    the one the conductors of :func:`materialise_branch` attach to
    (``None`` for a site without present electrodes).
    """
    if site.anchor is None:
        return None
    scratch = create_world(soil=HomogeneousSoil(rho=100.0))
    anchor = materialise_site(scratch, site)
    return tuple(float(v) for v in _node_point(scratch, anchor))  # type: ignore[return-value]


def materialise_site(world: World, site: PlannedSite) -> Optional[str]:
    """Create the electrodes of one site and bond them.

    Parameters
    ----------
    world
        Target world.
    site
        The planned site.

    Returns
    -------
    str or None
        The anchor electrode name, or ``None`` for a site without
        present electrodes.

    Notes
    -----
    Every non-anchor electrode is bonded to the anchor by a conductor of
    type ``site.bond_conductor_type`` (an ideal galvanic short), so the
    site is one cluster. A concrete shell lives on the encased wires of
    its foundation (:func:`materialise_electrode`), so the current of a
    bonded rod does not pass through it; nothing is written to
    ``World.concrete_shell_corrections`` (up to 0.15.0 the lumped shells
    of a site were folded there for the PEN service drop to carry,
    review findings 10.12 / 10.13).
    """
    cx, cy = site.position_xy
    created: list[Any] = []
    for planned in site.electrodes:
        spec = planned.spec
        name = site.electrode_name(planned)
        first = len(world.electrodes)
        materialise_electrode(
            world,
            spec,
            centre_xy=(cx + spec.offset_xy_m[0], cy + spec.offset_xy_m[1]),
            name=name,
            bond_conductor_type=site.bond_conductor_type,
        )
        # The first primitive of an electrode spec carries its name.
        created.append(world.electrodes[first])
    if not created:
        return None
    anchor = created[0]
    for k, electrode in enumerate(created[1:], start=1):
        create_conductor(
            world,
            name=f"{site.name}_bond_{k}",
            start=anchor,
            end=electrode,
            conductor_type=site.bond_conductor_type,
        )
    return anchor.name


# ---------------------------------------------------------------------
# Junctions, branches, sources
# ---------------------------------------------------------------------


def materialise_junction(world: World, junction: PlannedJunction) -> str:
    """Create a PEN cable joint and return its node name.

    The joint is a leakage-free :class:`~groundfield.geometry.junction.Junction`
    at ``(x, y, depth_m)``: a node of the conductor network with
    Kirchhoff's current law and no soil contact. Up to 0.15.0 it was a
    5 cm driven rod, which leaked 3 to 7 % of a test current through the
    nominally isolated cable (review finding 10.38).
    """
    create_junction(
        world,
        position=(junction.position_xy[0], junction.position_xy[1], junction.depth_m),
        name=junction.name,
    )
    return junction.name


def _node_point(world: World, name: str) -> tuple[float, float, float]:
    """Connection point (m) of the electrode or junction called ``name``."""
    for electrode in world.electrodes:
        if electrode.name == name:
            return electrode.connection_point
    return world.get_junction(name).connection_point


def materialise_branch(
    world: World,
    branch: PlannedBranch,
    nodes: dict[str, str],
    *,
    index: Optional[dict[str, Any]] = None,
) -> Conductor:
    """Create the conductor of one branch.

    Parameters
    ----------
    world
        Target world.
    branch
        The planned branch.
    nodes
        Map from node name (site or junction) to electrode name; a
        name missing from the map is used verbatim.
    index
        Optional map from electrode / junction name to the object in
        ``world``; saves a search through the world per end (the
        materialiser of a large network passes it).

    Returns
    -------
    Conductor
        The created conductor.
    """
    start = nodes.get(branch.start, branch.start)
    end = nodes.get(branch.end, branch.end)
    if index is not None:
        start_node, end_node = index.get(start, start), index.get(end, end)
    else:
        start_node, end_node = start, end
    p = branch.conductor
    if branch.axis_z_m is None:
        return create_conductor(
            world,
            name=branch.name,
            start=start_node,
            end=end_node,
            conductor_type=p.conductor_type,
            wire_radius=p.wire_radius_m,
            cross_section=p.cross_section,
            discretize_segment_length=p.segment_length_m,
            coupling_to_soil=p.coupling_to_soil,
            inductance_model=p.inductance_model,
        )
    z = float(branch.axis_z_m)
    sx, sy, _ = (
        start_node.connection_point
        if not isinstance(start_node, str)
        else _node_point(world, start)
    )
    ex, ey, _ = (
        end_node.connection_point
        if not isinstance(end_node, str)
        else _node_point(world, end)
    )
    cond = Conductor(
        name=branch.name,
        start=(sx, sy, z),
        end=(ex, ey, z),
        start_electrode=start,
        end_electrode=end,
        conductor_type=p.conductor_type,
        wire_radius=p.wire_radius_m,
        cross_section=p.cross_section,
        discretize_segment_length=p.segment_length_m,
        coupling_to_soil=p.coupling_to_soil,
        inductance_model=p.inductance_model,
    )
    world.add_conductor(cond)
    return cond


def materialise_plan(plan: NetworkPlan) -> World:
    """Build the :class:`~groundfield.World` described by ``plan``.

    Parameters
    ----------
    plan
        A concrete network plan.

    Returns
    -------
    World
        The world, ready for ``world.solve(engine)``.

    Notes
    -----
    Construction order: sites, measurement sites, junctions, instrument
    terminals, branches, measurement leads, sources (each in plan order).
    No random numbers are drawn, so the same plan always gives the same
    world.
    """
    world = create_world(name=plan.name, soil=plan.soil)
    nodes: dict[str, str] = {}
    for site in plan.all_sites:
        anchor = materialise_site(world, site)
        if anchor is not None:
            nodes[site.name] = anchor
    for junction in plan.all_junctions:
        nodes[junction.name] = materialise_junction(world, junction)
    index: dict[str, Any] = {e.name: e for e in world.electrodes}
    index.update((j.name, j) for j in world.junctions)
    for branch in plan.all_branches:
        materialise_branch(world, branch, nodes, index=index)
    for source in plan.sources:
        kwargs: dict = {}
        if source.name is not None:
            kwargs["name"] = source.name
        create_source(
            world,
            attached_to=nodes.get(source.attached_to, source.attached_to),
            return_to=(
                None
                if source.return_to is None
                else nodes.get(source.return_to, source.return_to)
            ),
            kind=source.kind,
            magnitude=source.magnitude,
            phase_deg=source.phase_deg,
            **kwargs,
        )
    return world
