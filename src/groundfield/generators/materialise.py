r"""Materialise a :class:`~groundfield.generators.plan.NetworkPlan` into a world.

This module is the **single** place where generated networks become
:class:`groundfield.World` objects (review finding 10.78). Both front-ends
(:class:`~groundfield.generators.tn_network.TnNetworkGenerator` and
:class:`~groundfield.generators.layout.LvNetworkLayout`) and
:meth:`GroundingSystemSpec.build_at
<groundfield.generators.grounding.GroundingSystemSpec.build_at>` call it.
It owns, in one function each:

* :func:`materialise_site` — the electrodes of a grounding site, the
  bonds to its anchor and the lumped concrete-shell bookkeeping;
* :func:`materialise_electrode` — one electrode, including the
  foundation geometry (axis-aligned grid, rotated mesh, building
  outline) and the concrete shell of ADR-0012;
* :func:`materialise_junction` — a PEN cable joint (a leakage-free
  :class:`~groundfield.geometry.junction.Junction` node);
* :func:`materialise_branch` — a conductor between two nodes (PEN
  sections, service drops, measurement leads);
* :func:`materialise_plan` — everything, in a fixed order.

No random numbers are drawn here: every value of a plan is concrete.

Order of construction
---------------------
``materialise_plan`` builds sites, measurement sites, junctions,
branches, measurement leads and sources, each list in plan order. The
order fixes the order of ``world.electrodes`` / ``world.conductors``
(and therefore the canonical world JSON), not the physics.

Concrete shell (ADR-0012)
-------------------------
A foundation encased in concrete of resistivity :math:`\rho_c` and
thickness :math:`t` around a wire of radius :math:`r_a` is seen by the
soil through the outer shell radius :math:`r_b = r_a + t`. The radial
shell adds the per-length resistance coefficient

.. math::

   C = \frac{\rho_c}{2\pi}\,\ln\frac{r_b}{r_a}\quad[\Omega\,\mathrm{m}],

i.e. :math:`R_\text{shell} = C / L_\text{perim}` for a perimeter of
length :math:`L_\text{perim}`. ``concrete_model="distributed"`` (V2)
puts :math:`C` on every perimeter segment; ``"lumped"`` (V1) records
:math:`R_\text{shell}` in ``world.concrete_shell_corrections`` under the
site anchor (several encased electrodes of one site combine in parallel)
for a branch with :attr:`PlannedBranch.concrete_shell_of
<groundfield.generators.plan.PlannedBranch.concrete_shell_of>` to carry.
"""

from __future__ import annotations

import math
from typing import Optional

from groundfield.api import (
    create_conductor,
    create_electrode,
    create_junction,
    create_source,
    create_world,
)
from groundfield.conductors.conductor import Conductor
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
    spec: FoundationElectrodeSpec, *, perimeter_m: float
) -> tuple[float, float, float]:
    r"""Effective radius, lumped resistance and coefficient of a shell.

    Parameters
    ----------
    spec
        Resolved foundation spec (concrete values only).
    perimeter_m
        Length in m of the encased wire (the foundation perimeter;
        internal cross-braces are not encased).

    Returns
    -------
    effective_radius_m : float
        :math:`r_b = r_a + t` in m when a shell is configured, else the
        bare wire radius :math:`r_a`.
    shell_total_ohm : float
        :math:`R_\text{shell} = C / L_\text{perim}` in Ω (0 without shell).
    shell_coefficient_ohm_m : float
        :math:`C = \rho_c/(2\pi)\,\ln(r_b/r_a)` in Ω·m (0 without shell).
    """
    if spec.concrete_rho_ohm_m is None:
        return float(spec.wire_radius_m), 0.0, 0.0
    rho_c = float(spec.concrete_rho_ohm_m)
    t = float(spec.concrete_thickness_m)
    if rho_c <= 0.0 or t <= 0.0:
        return float(spec.wire_radius_m), 0.0, 0.0
    r_a = float(spec.wire_radius_m)
    r_b = r_a + t
    perimeter = float(perimeter_m)
    coefficient = rho_c / (2.0 * math.pi) * math.log(r_b / r_a)
    r_shell = coefficient / perimeter if perimeter > 0.0 else 0.0
    return r_b, float(r_shell), float(coefficient)


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


def _wire_shell_coefficient(
    world: World,
    spec: FoundationElectrodeSpec,
    name: str,
    *,
    shell_total_ohm: float,
    shell_coefficient_ohm_m: float,
) -> float:
    """Book a lumped (V1) shell; return the coefficient wires carry (V2).

    ``concrete_model="lumped"`` records ``shell_total_ohm`` under the
    electrode name and returns 0 (the wires carry no coefficient);
    ``"distributed"`` returns the per-length coefficient.
    """
    if spec.concrete_model == "lumped":
        if shell_total_ohm > 0.0:
            world.concrete_shell_corrections[name] = shell_total_ohm
        return 0.0
    if spec.concrete_model == "distributed":
        return shell_coefficient_ohm_m
    raise ValueError(
        f"Unknown concrete_model: {spec.concrete_model!r}"
    )  # pragma: no cover


def _materialise_foundation(
    world: World,
    spec: FoundationElectrodeSpec,
    *,
    centre_xy: tuple[float, float],
    name: str,
    bond_conductor_type: str,
) -> None:
    """Foundation electrode: building outline, axis-aligned grid or rotated mesh."""
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
    effective_radius, shell_total_ohm, shell_coeff = concrete_shell(
        spec, perimeter_m=2.0 * (dx + dy)
    )
    strip_shell_coefficient = _wire_shell_coefficient(
        world,
        spec,
        name,
        shell_total_ohm=shell_total_ohm,
        shell_coefficient_ohm_m=shell_coeff,
    )

    # A rectangle turned by a multiple of 90 deg is axis-aligned: realise
    # it on the grid-mesh path with swapped sides (and mesh counts), so
    # the same rectangle always gets the same discretisation whichever
    # way it was described (10.39).
    angle_deg, dx, dy, nx, ny = _canonical_right_angle(
        spec.orientation_deg or 0.0, dx, dy, nx, ny
    )
    # A distributed shell needs per-wire primitives (the coefficient is
    # carried per segment), so it always takes the per-wire path.
    force_strip_path = strip_shell_coefficient > 0.0
    if angle_deg == 0.0 and not force_strip_path:
        create_electrode(
            world,
            "grid_mesh",
            name=name,
            corner=(centre_xy[0] - dx / 2.0, centre_xy[1] - dy / 2.0, depth),
            size=(dx, dy),
            n_x=nx,
            n_y=ny,
            wire_radius=effective_radius,
        )
        return
    _materialise_rotated_foundation(
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
        concrete_shell_coefficient=strip_shell_coefficient,
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
            f"Foundation {name!r}: polygon collapses to fewer than 3 "
            "vertices after removing sub-centimetre edges — check the "
            "footprint."
        )
    perimeter = 0.0
    for k in range(len(verts)):
        x0, y0 = verts[k]
        x1, y1 = verts[(k + 1) % len(verts)]
        perimeter += math.hypot(x1 - x0, y1 - y0)
    effective_radius, shell_total_ohm, shell_coeff = concrete_shell(
        spec, perimeter_m=perimeter
    )
    ring_coeff = _wire_shell_coefficient(
        world,
        spec,
        name,
        shell_total_ohm=shell_total_ohm,
        shell_coefficient_ohm_m=shell_coeff,
    )
    create_electrode(
        world,
        "polyline",
        name=name,
        vertices=[(cx + vx, cy + vy, depth) for vx, vy in verts],
        closed=True,
        wire_radius=effective_radius,
        concrete_shell_coefficient_ohm_m=ring_coeff,
    )


def _materialise_rotated_foundation(
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
    r"""Rotated rectangular foundation as bonded straight wires.

    ``n_y + 1`` longitudinal wires (along the local :math:`+x` axis,
    rotated by ``angle_deg``) and ``n_x + 1`` transverse wires; with
    ``n_x = n_y = 1`` exactly the four perimeter edges. The first wire
    (local :math:`-y` edge) carries ``name`` and is the anchor; the
    others are ``f"{name}_w{j}"``, bonded to the anchor by
    ``f"{name}_bond_{j}"``. Only perimeter wires carry the distributed
    concrete-shell coefficient (internal braces are not encased).

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
    coincide (10.39). Each wire is one uniform-current owner, as before.
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
    perimeter = {0, n_y, n_y + 1, n_y + 1 + n_x}
    sub_names: list[str] = []
    for j, local in enumerate(wires):
        sub_name = name if j == 0 else f"{name}_w{j}"
        coefficient = concrete_shell_coefficient if j in perimeter else 0.0
        vertices = [to_world(lx, ly) for lx, ly in local]
        if len(vertices) == 2:
            create_electrode(
                world,
                "strip",
                name=sub_name,
                start=vertices[0],
                end=vertices[1],
                wire_radius=wire_radius,
                concrete_shell_coefficient_ohm_m=coefficient,
            )
        else:
            create_electrode(
                world,
                "polyline",
                name=sub_name,
                vertices=vertices,
                closed=False,
                wire_radius=wire_radius,
                concrete_shell_coefficient_ohm_m=coefficient,
            )
        sub_names.append(sub_name)
    anchor = sub_names[0]
    for j, sn in enumerate(sub_names[1:], start=1):
        create_conductor(
            world,
            name=f"{name}_bond_{j}",
            start=anchor,
            end=sn,
            conductor_type=bond_conductor_type,
        )


def materialise_site(world: World, site: PlannedSite) -> Optional[str]:
    r"""Create the electrodes of one site, bond them and fold the shells.

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
    site is one cluster. Lumped concrete shells (ADR-0012 V1) are booked
    per electrode while the electrodes are created and then folded onto
    the anchor: several encased electrodes of one site leak through
    their own shells in parallel, so the site value is
    :math:`(\sum_i 1/R_i)^{-1}`.
    """
    cx, cy = site.position_xy
    created: list[str] = []
    for planned in site.electrodes:
        spec = planned.spec
        name = site.electrode_name(planned)
        materialise_electrode(
            world,
            spec,
            centre_xy=(cx + spec.offset_xy_m[0], cy + spec.offset_xy_m[1]),
            name=name,
            bond_conductor_type=site.bond_conductor_type,
        )
        created.append(name)
    if not created:
        return None
    anchor = created[0]
    for k, name in enumerate(created[1:], start=1):
        create_conductor(
            world,
            name=f"{site.name}_bond_{k}",
            start=anchor,
            end=name,
            conductor_type=site.bond_conductor_type,
        )
    conductances = 0.0
    for name in created:
        r_i = world.concrete_shell_corrections.pop(name, 0.0)
        if r_i > 0.0:
            conductances += 1.0 / r_i
    if conductances > 0.0:
        world.concrete_shell_corrections[anchor] = 1.0 / conductances
    return anchor


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
    world: World, branch: PlannedBranch, nodes: dict[str, str]
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

    Returns
    -------
    Conductor
        The created conductor.
    """
    start = nodes.get(branch.start, branch.start)
    end = nodes.get(branch.end, branch.end)
    p = branch.conductor
    lumped: Optional[float] = None
    if branch.concrete_shell_of is not None:
        shell_ohm = world.concrete_shell_corrections.get(
            nodes.get(branch.concrete_shell_of, "")
        )
        if shell_ohm is not None and shell_ohm > 0.0:
            lumped = float(shell_ohm)
    if branch.axis_z_m is None:
        return create_conductor(
            world,
            name=branch.name,
            start=start,
            end=end,
            conductor_type=p.conductor_type,
            wire_radius=p.wire_radius_m,
            cross_section=p.cross_section,
            discretize_segment_length=p.segment_length_m,
            coupling_to_soil=p.coupling_to_soil,
            inductance_model=p.inductance_model,
            lumped_series_resistance_ohm=lumped,
        )
    z = float(branch.axis_z_m)
    sx, sy, _ = _node_point(world, start)
    ex, ey, _ = _node_point(world, end)
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
        lumped_series_resistance_ohm=lumped,
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
    Construction order: sites, measurement sites, junctions, branches,
    measurement leads, sources (each in plan order). No random numbers
    are drawn, so the same plan always gives the same world.
    """
    world = create_world(name=plan.name, soil=plan.soil)
    nodes: dict[str, str] = {}
    for site in plan.all_sites:
        anchor = materialise_site(world, site)
        if anchor is not None:
            nodes[site.name] = anchor
    for junction in plan.junctions:
        nodes[junction.name] = materialise_junction(world, junction)
    for branch in plan.all_branches:
        materialise_branch(world, branch, nodes)
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
