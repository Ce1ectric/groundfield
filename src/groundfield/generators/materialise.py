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
electrode of length :math:`L` under a uniform leakage current. This one
mechanism (review findings 10.12–10.14) does not depend on the deprecated
``concrete_model``, which has no effect at all since user decision Q38.
The cross-braces of a mesh foundation are encased like its
perimeter: a foundation earth electrode per DIN 18014 lies in the concrete
together with its connections (user decision Q20 of 2026.11). In 0.15.0
every wire got :math:`r_b` but the braces no coefficient (and with
``"lumped"`` no wire did); the bare braces (:math:`r_a`, no coefficient)
of the 2026.11 development line before the decision were never released.
Without a shell every wire keeps :math:`r_a`. Rods and other electrodes of
a site are never encased.

Owners of a foundation
----------------------
The image family solves for one uniform leakage current per straight wire
of an electrode (an *owner*; since user decisions Q30/Q44 also inside a
grid, star or polyline electrode, whose collinear consecutive edges form one
wire). A foundation's owners depend only on its structure — never
on its orientation, on whether a shell is configured (audit A2G-04, review
finding 10.39) or on ``concrete_model`` (user decision Q38):

* a mesh foundation (cross-braces) has one owner per wire line. One owner
  over perimeter and braces puts the same current density on both and is
  +1.6 / +2.5 / +3.2 % high for 2 × 2 / 3 × 3 / 4 × 4 cells (10 / 20 / 30 m,
  line kernel, 1 m segments) against the equipotential reference of the
  same segments (one owner per segment; with the point kernel identical
  to ``mom``); one owner per wire is +0.3 / +0.5 / +0.6 %. Against ``mom``
  itself (point kernel) one owner per wire is +1.1 % with the default line
  kernel and +0.3 / +0.5 / +0.5 % with the point kernel;
* a rectangle without cross-braces (``style="ring"``, or ``"mesh"`` with
  ``n_x = n_y = 1``) has one owner per side, at every angle (user decision
  Q38 of 2026.11). Up to then the default ``"lumped"`` made an
  axis-aligned one — also one turned by a multiple of 90°, which was built
  as the axis-aligned rectangle with swapped sides — a single owner (one
  :class:`~groundfield.geometry.electrodes.GridMeshElectrode`, connected
  at its centre), and only ``"distributed"`` or a rotation by another
  angle split it (audit D3B-04). The
  four bonded owners carry the split of the current among the sides that
  dissipates least (the owner matrix is symmetric and positive definite),
  and the single owner's split, in proportion to the side lengths, is one
  of those they can take: per side the input impedance is never higher,
  and equal for a square. Over the soils of the AP1 grid, for a 5 mm wire
  at 0.8 m, bare or in concrete of 30 to 1000 Ω·m up to 0.2 m thick, it
  is lower by less than 1 % up to aspect ratio 4 (short side at least
  1 m, sides up to 80 m) and 2 % up to aspect ratio 50 (short side at
  least 1 m, sides up to 150 m); bare or with the default 5 cm shell by
  less than 0.4 % and 1 % (largest decreases found −0.686 / −1.529 % and
  −0.320 / −0.907 %; audit R2B-01). Against one current per segment one
  owner per side is at most 1.4 % high over that scope (+1.34 % for
  150 m × 3 m, bare, in 30 over 200 Ω·m with :math:`h_1` = 5 m; +1.35 %
  in a 0.2 m shell) and at most 0.25 % up to aspect ratio 4 bare or with
  the default shell; the single owner was up to 2.2 % high (audits D3B-01,
  R4B-05;
  at the largest decreases above 0.23 … 0.90 % against 0.55 … 2.16 %);
* a building outline (``polygon_xy_m``) is one closed polyline, which the
  image family solves as one owner per straight side since user decisions
  Q30/Q44 (up to then one owner for the whole outline); a rectangular
  outline therefore equals the same rectangle given by its size.

A foundation of several wires is bonded to its first wire, the anchor:
the local :math:`-y` side of a rectangle, which starts at the local
:math:`(-d_x/2, -d_y/2)` corner — its connection point, where conductors
attached to the foundation by name start. A rectangle at a multiple of 90°
is rotated with the exact cosine and sine, so its coordinates carry no
round-off and the local :math:`-y` side stays the anchor (up to audit
D3B-04 it was built as the 0° rectangle with swapped sides, whose anchor
is another side: the connection point jumped by up to the rectangle's
diagonal between 179.9° and 180°). Per-electrode results under the
foundation's name (its current,
:meth:`~groundfield.solver.result.FieldResult.grounding_impedance`)
therefore refer to that side; the cluster-level results refer to the
whole foundation.

Where conductors meet a site
----------------------------
Every conductor the generators route to a site with a horizontal earth
electrode ends at the point of that electrode nearest to it: at a
foundation since user decision Q41 of 2026.11, at a circular ring
(:class:`~groundfield.generators.electrode_specs.RingElectrodeSpec`, e.g.
the default substations of both generators) and a strip since user
decision Q43. A service drop is the shortest connection in plan between its
feeding cable and the horizontal wires of the building (their closest pair),
a cable leaving a substation or a cabinet starts at the point of its
horizontal wires nearest to the cable's next node; the conductor is attached
to the wire that holds the point
(:class:`~groundfield.generators.plan.PlannedContact`,
:func:`site_contact_wires`). A site with horizontal electrodes and rods is
met on the horizontal ones; rods (vertical) keep their head, so a site with
rods only keeps the connection point of its anchor
(:func:`site_anchor_point`), and so does everything attached by name in
user code. Ties go to the lowest wire index, then to the smallest arc
position along the wire: a cable parallel to the side facing it meets that
side at the start of the range of equal distance, for a side it passes
along its whole length the corner where the wire starts. A circle is met at
its exact nearest point (:mod:`groundfield.generators._contact`).
Physically the conductor is a finite-impedance branch of the nodal analysis
(ADR-0003) and, with an inductance model, a Neumann filament (ADR-0004):
where it ends on the electrode does not change the galvanic model, whose
bonded wires are one cluster, but sets the length of the conductor and its
coupling to the electrode's wires and to the other conductors — at 1 kHz
the reactance of a small ring-house network changed by +8 … +18 % when the
drops ran from the corners (audit D3A-02).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from typing import Any, NamedTuple, Optional

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
from groundfield.generators._contact import (
    FoundationWire,
    Outline,
    circle_polygon,
    convex_outline,
)
from groundfield.generators.plan import (
    NetworkPlan,
    PlannedBranch,
    PlannedContact,
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
    "site_contact_wires",
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
    New in 2026.11. Before the user decision Q20 the development line kept
    the braces of a mesh bare and took the perimeter as the length (keyword
    ``perimeter_m``); 0.15.0 had no such function (its private
    ``grounding._maybe_concrete_shell`` divided by the perimeter).
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
    mesh with cross-braces is one owner per wire line; a rectangle without
    braces (a ring, or a mesh with ``n_x = n_y = 1``) is one owner per
    side, at every angle (user decision Q38); a building outline is one
    closed polyline with one owner per straight side (one current per
    straight wire, user decisions Q30, Q44). None of them depends on whether
    a shell is configured
    or on the deprecated ``concrete_model``, which has no effect. The
    concrete shell (ADR-0012) puts the radius :math:`r_b` and the
    coefficient :math:`C` on every wire of the foundation, the cross-braces
    of a mesh included (user decision Q20, DIN 18014).
    """
    depth = float(spec.depth_m)
    if spec.polygon_xy_m is not None:
        _materialise_polygon_foundation(
            world, spec, centre_xy=centre_xy, depth=depth, name=name
        )
        return
    if spec.size_xy_m is not None:
        dx, dy = (float(v) for v in spec.size_xy_m)
    else:
        size = float(spec.size_m)
        dx, dy = size, size
    # The sides are checked here, after the draws of a Distribution: four
    # strips accept a negative side and build the mirrored rectangle (audit
    # D3A-01; the grid mesh that the default ring was up to user decision
    # Q38 refused it). ``not dx > 0`` also catches NaN.
    if not (dx > 0.0 and dy > 0.0):
        raise ValueError(
            f"Foundation {name!r}: the sides of the rectangle must be positive, "
            f"got {dx:g} m x {dy:g} m."
        )
    # ``style="ring"`` is the perimeter only: one mesh cell.
    if spec.style == "ring":
        nx, ny = 1, 1
    else:
        nx, ny = spec.n_x, spec.n_y
    # Every wire is encased: ny + 1 lines of length dx, nx + 1 of length dy.
    effective_radius, _, shell_coefficient = concrete_shell(
        spec, encased_length_m=(ny + 1) * dx + (nx + 1) * dy
    )

    # One owner per wire line at every angle; for a rectangle without
    # cross-braces one per side (user decision Q38: the axis-aligned
    # "lumped" one was a single GridMeshElectrode up to then).
    _materialise_foundation_wires(
        world,
        centre_xy=centre_xy,
        depth=depth,
        dx=dx,
        dy=dy,
        n_x=nx,
        n_y=ny,
        angle_deg=float(spec.orientation_deg or 0.0),
        name=name,
        wire_radius=effective_radius,
        concrete_shell_coefficient=shell_coefficient,
        bond_conductor_type=bond_conductor_type,
    )


_RIGHT_ANGLE_TOL_DEG = 1e-9
"""Angles closer than this (degrees) to a multiple of 90 count as one."""

_QUARTER_TURNS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
"""Exact ``(cos, sin)`` of 0, 90, 180 and 270 degrees."""


def _rotation(angle_deg: float) -> tuple[float, float]:
    """``(cos, sin)`` of a rotation, exact at multiples of 90 degrees.

    Parameters
    ----------
    angle_deg
        Rotation of the local ``+x`` axis in degrees.

    Returns
    -------
    tuple of float
        ``(cos a, sin a)``. Within :data:`_RIGHT_ANGLE_TOL_DEG` of a multiple
        of 90 degrees the exact values ``0`` and ``±1``, so the wires of a
        rectangle at a right angle get coordinates without round-off (and
        0 degrees, 360 degrees, ... build the axis-aligned rectangle bit for
        bit). The rectangle is rotated like at any other angle: its local
        :math:`-y` side stays the anchor, so the anchor and its connection
        point are continuous in the orientation (audit D3B-04; up to then a
        multiple of 90 degrees was realised as the 0-degree rectangle with
        swapped sides, whose anchor was another side).
    """
    quarter_turns = round(angle_deg / 90.0)
    if abs(angle_deg - 90.0 * quarter_turns) <= _RIGHT_ANGLE_TOL_DEG:
        return _QUARTER_TURNS[quarter_turns % 4]
    rad = math.radians(angle_deg)
    return math.cos(rad), math.sin(rad)


def _materialise_polygon_foundation(
    world: World,
    spec: FoundationElectrodeSpec,
    *,
    centre_xy: tuple[float, float],
    depth: float,
    name: str,
) -> None:
    """Foundation along the building outline (one closed polyline, solved as one owner per straight side, user decisions Q30/Q44)."""
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
    braces lie in the concrete too, DIN 18014; in 0.15.0 they had
    :math:`r_b` without a coefficient, and the 2026.11 development line
    before the decision kept them at :math:`r_a` without one, review
    finding 10.13).

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
    cos_a, sin_a = _rotation(angle_deg)
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
    the one a branch without a contact attaches to (``None`` for a site
    without present electrodes). For a rectangular foundation it is a
    corner — the start of its local :math:`-y` side (meshes since audit
    A2G-04, rectangles without cross-braces since user decision Q38) — and
    for a building outline its first vertex, for a circular ring its
    :math:`+x` point, for a strip its start, for a rod its head. The
    generators connect their cables to a site with a horizontal electrode (a
    foundation, a ring, a strip) at the point of those electrodes nearest to
    each cable instead (:func:`site_contact_wires`, user decisions Q41,
    Q43); this point stays the end of everything attached to the site by
    name, and of the generators' conductors at a site with rods only.
    """
    if site.anchor is None:
        return None
    scratch = create_world(soil=HomogeneousSoil(rho=100.0))
    anchor = materialise_site(scratch, site)
    return tuple(float(v) for v in _node_point(scratch, anchor))  # type: ignore[return-value]


_HORIZONTAL_SPECS = (FoundationElectrodeSpec, RingElectrodeSpec, StripElectrodeSpec)
"""Electrode specs a generated conductor meets at their nearest point (user
decisions Q41, Q43): foundations, circular rings and strips; rods keep their
head."""


def site_contact_wires(site: PlannedSite) -> list[FoundationWire]:
    """The horizontal wires of ``site``, in plan (user decisions Q41, Q43).

    Parameters
    ----------
    site
        A planned site.

    Returns
    -------
    list of FoundationWire
        One entry per electrode built from a
        :class:`~groundfield.generators.electrode_specs.FoundationElectrodeSpec`
        (every side or mesh line of a rectangle, the closed outline of a
        building), a
        :class:`~groundfield.generators.electrode_specs.RingElectrodeSpec`
        (the circle; ``circle`` holds its centre and radius, ``xy`` the
        inscribed search polygon of
        :func:`~groundfield.generators._contact.circle_polygon`) or a
        :class:`~groundfield.generators.electrode_specs.StripElectrodeSpec`
        of the site, in world order: its name, its index in this list, its
        plan vertices from its connection point and its depth. Empty for a
        site with rods only.

    Notes
    -----
    New in 2026.11 (as ``site_foundation_wires`` with the foundations only,
    user decision Q41; renamed and extended to rings and strips by user
    decision Q43). The generators end every conductor they route to a site
    with horizontal electrodes at the point of these wires nearest to the
    conductor (:mod:`groundfield.generators._contact`); a site with rods only
    keeps the connection point of its anchor, the head of its first rod
    (:func:`site_anchor_point`). The site is materialised into a scratch
    world, so the wires are exactly those of the materialised plan.
    """
    return _site_geometry(site, anchor=False).wires


class SiteGeometry(NamedTuple):
    """Plan geometry of a site where generated conductors meet it."""

    point: Optional[tuple[float, float, float]]
    """Connection point in m of the anchor (``None``: no electrodes)."""
    wires: list[FoundationWire]
    """Horizontal wires (:func:`site_contact_wires`)."""
    outlines: list[Outline]
    """Areas enclosed by the closed horizontal electrodes: a rectangle, a
    building outline, a circle (strips enclose none)."""


def _site_geometry(site: PlannedSite, *, anchor: bool = True) -> SiteGeometry:
    """Anchor connection point, horizontal wires and outlines of ``site``.

    The electrodes are materialised into one scratch world without the
    bonds, so the anchor's connection point is the one of
    :func:`site_anchor_point` (``None`` for a site without electrodes) and
    the wires are those of :func:`site_contact_wires`. With ``anchor=False``
    only the horizontal electrode specs are materialised.
    """
    scratch = create_world(soil=HomogeneousSoil(rho=100.0))
    cx, cy = site.position_xy
    point: Optional[tuple[float, float, float]] = None
    wires: list[FoundationWire] = []
    outlines: list[Outline] = []
    for planned in site.electrodes:
        spec = planned.spec
        horizontal = isinstance(spec, _HORIZONTAL_SPECS)
        if not horizontal and not (anchor and point is None):
            continue
        first = len(scratch.electrodes)
        materialise_electrode(
            scratch,
            spec,
            centre_xy=(cx + spec.offset_xy_m[0], cy + spec.offset_xy_m[1]),
            name=site.electrode_name(planned),
            bond_conductor_type=site.bond_conductor_type,
        )
        if anchor and point is None:
            point = tuple(float(v) for v in scratch.electrodes[first].connection_point)  # type: ignore[assignment]
        if not horizontal:
            continue
        runs: list[tuple[tuple[float, float], ...]] = []
        for electrode in scratch.electrodes[first:]:
            z = float(electrode.connection_point[2])
            if isinstance(spec, RingElectrodeSpec):
                ex, ey, _ = electrode.center
                circle = (float(ex), float(ey), float(electrode.radius))
                xy = circle_polygon(*circle)
                outlines.append(Outline(xy=xy, circle=circle))
                wires.append(FoundationWire(electrode.name, len(wires), xy, z, circle))
                continue
            (run,) = electrode.wire_polylines()
            xy = tuple((float(x), float(y)) for x, y, _ in run)
            runs.append(xy)
            wires.append(FoundationWire(electrode.name, len(wires), xy, z))
        if isinstance(spec, FoundationElectrodeSpec):
            if spec.polygon_xy_m is not None:
                # One closed polyline along the building outline.
                outlines.append(Outline(xy=runs[0]))
            else:
                # A rectangle (sides, or the lines of a mesh): its corners.
                outlines.append(
                    Outline(xy=convex_outline([p for run in runs for p in run]))
                )
    return SiteGeometry(point, wires, outlines)


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
    return _materialise_site(world, site)[0]


def _materialise_site(
    world: World, site: PlannedSite
) -> tuple[Optional[str], set[str]]:
    """:func:`materialise_site`, and the names of its horizontal wires.

    The second item holds every electrode built from a
    :class:`~groundfield.generators.electrode_specs.FoundationElectrodeSpec`,
    :class:`~groundfield.generators.electrode_specs.RingElectrodeSpec` or
    :class:`~groundfield.generators.electrode_specs.StripElectrodeSpec` of
    the site: the wires a :class:`~groundfield.generators.plan.PlannedContact`
    may name (audit R4A-05; rings and strips since user decision Q43).
    """
    cx, cy = site.position_xy
    created: list[Any] = []
    foundation: set[str] = set()
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
        if isinstance(spec, _HORIZONTAL_SPECS):
            foundation.update(e.name for e in world.electrodes[first:])
    if not created:
        return None, foundation
    anchor = created[0]
    for k, electrode in enumerate(created[1:], start=1):
        create_conductor(
            world,
            name=f"{site.name}_bond_{k}",
            start=anchor,
            end=electrode,
            conductor_type=site.bond_conductor_type,
        )
    return anchor.name, foundation


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


def _branch_end(
    world: World,
    node: Any,
    name: str,
    contact: Optional[PlannedContact],
    index: Optional[dict[str, Any]],
    *,
    branch: Optional[PlannedBranch] = None,
    site: Optional[str] = None,
    foundation_wires: Optional[Mapping[str, Collection[str]]] = None,
) -> tuple[tuple[float, float, float], str]:
    """End point (m) and attached node name of one end of a branch.

    Without a contact the connection point of the node; with one the
    contact's point on the named wire, at the wire's depth (user decision
    Q41), after :func:`_check_contact`.
    """
    if contact is None:
        point = node.connection_point if not isinstance(node, str) else None
        if point is None:
            point = _node_point(world, name)
        return tuple(float(v) for v in point), name  # type: ignore[return-value]
    wire = _check_contact(world, contact, index, branch, site, foundation_wires)
    x, y = contact.position_xy
    return (float(x), float(y), float(wire.connection_point[2])), contact.electrode


def _check_contact(
    world: World,
    contact: PlannedContact,
    index: Optional[dict[str, Any]],
    branch: Optional[PlannedBranch],
    site: Optional[str],
    foundation_wires: Optional[Mapping[str, Collection[str]]],
) -> Any:
    """The wire a contact names, after checking that the contact is on it.

    A contact must name a horizontal wire of the site the branch ends at — a
    wire of its foundation, ring or strip (``foundation_wires[site]`` when
    given, as :func:`materialise_plan` does; otherwise any horizontal wire,
    a strip, polyline or ring electrode) — and lie within one minimum cable
    section of it in plan (of the exact circle for a ring): a joined cable's
    point (rule 4 of the PEN router) lies up to
    :func:`~groundfield.generators._pen_routing.min_section_length` off the
    wire it crosses or touches, 1 cm for wire radii up to 7.3 mm (audit
    R4A-05; rings and strips since user decision Q43).

    Raises
    ------
    ValueError
        If the electrode is unknown, not a horizontal wire of the site, or
        farther than that from the point.
    """
    from groundfield.generators._pen_routing import (
        MIN_DROP_LENGTH_M,
        min_section_length,
    )

    label = f"branch {branch.name!r}" if branch is not None else "a branch"
    wire = (index or {}).get(contact.electrode)
    if wire is None:
        try:
            wire = world.get_electrode(contact.electrode)
        except KeyError:
            raise ValueError(
                f"materialise_branch: {label} meets {contact.electrode!r}, "
                "which is no electrode of the world; a contact names a "
                "horizontal wire of the site the branch ends at."
            ) from None
    kind = getattr(wire, "kind", None)
    if foundation_wires is not None and site is not None:
        allowed = contact.electrode in foundation_wires.get(site, ())
    else:
        allowed = kind in ("strip", "polyline", "ring")
    if not allowed:
        raise ValueError(
            f"materialise_branch: {label} meets {contact.electrode!r}, which "
            f"is not a foundation, ring or strip wire of site {site!r}; a "
            "contact names a wire of a FoundationElectrodeSpec, "
            "RingElectrodeSpec or StripElectrodeSpec of that site."
        )
    tolerance = (
        MIN_DROP_LENGTH_M if branch is None else min_section_length(branch.conductor)
    )
    x, y = (float(v) for v in contact.position_xy)
    off = math.inf
    if kind == "ring":
        # The exact circle (user decision Q43).
        cx, cy, _ = (float(v) for v in wire.center)
        off = abs(math.hypot(x - cx, y - cy) - float(wire.radius))
    for run in [] if kind == "ring" else wire.wire_polylines():
        pts = [(float(p[0]), float(p[1])) for p in run]
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:], strict=True):
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 == 0.0 else ((x - ax) * dx + (y - ay) * dy) / length2
            t = min(max(t, 0.0), 1.0)
            off = min(off, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    if not off <= tolerance * (1.0 + 1e-9):
        raise ValueError(
            f"materialise_branch: {label} meets {contact.electrode!r} at "
            f"({x:g}, {y:g}) m, {off:.3g} m from the wire in plan; a contact "
            f"lies on its wire, at most {tolerance:g} m off (the point of a "
            "cable joined to a wire it crosses)."
        )
    return wire


def materialise_branch(
    world: World,
    branch: PlannedBranch,
    nodes: dict[str, str],
    *,
    index: Optional[dict[str, Any]] = None,
    foundation_wires: Optional[Mapping[str, Collection[str]]] = None,
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
    foundation_wires
        Optional map from site name to the names of its horizontal wires
        (foundations, rings, strips; :func:`materialise_plan` passes it); a
        contact must name one of the wires of the site the branch ends at.
        Without it the check is weaker: the contact must name a strip,
        polyline or ring electrode, which need not belong to that site (a
        :class:`~groundfield.generators.plan.NetworkPlan` rejects a contact
        on another site's electrode when it is validated).

    Returns
    -------
    Conductor
        The created conductor.

    Notes
    -----
    An end with a :class:`~groundfield.generators.plan.PlannedContact` is
    the contact's point on the wire it names, and the conductor is attached
    to that wire (user decisions Q41, Q43); an end without one is the connection
    point of the node (the site's anchor electrode, or the junction). With
    ``axis_z_m`` both ends are moved to that depth.

    Raises
    ------
    ValueError
        If a contact names an electrode that is not a horizontal wire
        (foundation, ring, strip) of the site the branch ends at (with
        ``foundation_wires``; without it, an electrode that is no strip,
        polyline or ring), or lies farther from that wire in plan (from the
        exact circle of a ring) than one minimum cable section
        (:func:`~groundfield.generators._pen_routing.min_section_length` of
        the branch's conductor: 1 cm for wire radii up to 7.3 mm; audit
        R4A-05).
    """
    start = nodes.get(branch.start, branch.start)
    end = nodes.get(branch.end, branch.end)
    if index is not None:
        start_node, end_node = index.get(start, start), index.get(end, end)
    else:
        start_node, end_node = start, end
    p = branch.conductor
    if (
        branch.axis_z_m is None
        and branch.start_contact is None
        and branch.end_contact is None
    ):
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
    (sx, sy, sz), start = _branch_end(
        world,
        start_node,
        start,
        branch.start_contact,
        index,
        branch=branch,
        site=branch.start,
        foundation_wires=foundation_wires,
    )
    (ex, ey, ez), end = _branch_end(
        world,
        end_node,
        end,
        branch.end_contact,
        index,
        branch=branch,
        site=branch.end,
        foundation_wires=foundation_wires,
    )
    if branch.axis_z_m is not None:
        sz = ez = float(branch.axis_z_m)
    cond = Conductor(
        name=branch.name,
        start=(sx, sy, sz),
        end=(ex, ey, ez),
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
    foundation_wires: dict[str, set[str]] = {}
    for site in plan.all_sites:
        anchor, foundation_wires[site.name] = _materialise_site(world, site)
        if anchor is not None:
            nodes[site.name] = anchor
    for junction in plan.all_junctions:
        nodes[junction.name] = materialise_junction(world, junction)
    index: dict[str, Any] = {e.name: e for e in world.electrodes}
    index.update((j.name, j) for j in world.junctions)
    for branch in plan.all_branches:
        materialise_branch(
            world, branch, nodes, index=index, foundation_wires=foundation_wires
        )
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
