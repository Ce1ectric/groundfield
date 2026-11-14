"""Generated conductors end at the foundation point nearest to them (Q41).

User decision Q41 of 2026-11-14: every conductor that the generators route
to a foundation — a PEN service drop of ``TnNetworkGenerator`` or of the
layout builder, a cable that leaves a substation or a cabinet — ends at the
point of that foundation nearest to the conductor (the connection lug on the
side facing the cable), for every foundation form: rectangles (ring or mesh,
any orientation), building outlines, the substation's foundation. A service
drop is the shortest connection in plan between its feeding cable and the
foundation wires (the closest pair); ties go to the lowest wire index, then
to the smallest arc position along the wire.

Before (base ``500127c``) the generators attached every conductor at the
connection point of the site's anchor electrode: the start of the local -y
side of a rectangle (a corner; meshes since audit A2G-04, rings since user
decision Q38) or the first vertex of an outline. 47 of the 282 drops of the
audit D3A setup ran along a side of their own foundation (up to 8 m), and a
house beyond its cable was connected at its far corner, the drop crossing
the house (audits D3A-02, D3B-03).

The checks below use a second, independent method: dense sampling of the
wires and cables in plan.
"""

from __future__ import annotations

import itertools
import math
import pathlib
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import (
    BuildingTypeSpec,
    CabinetConfig,
    ExplicitPlacement,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    HomogeneousSoilSpec,
    LvNetworkLayout,
    ManhattanGridPlacement,
    PenConfig,
    RadialTrunkTopology,
    RodElectrodeSpec,
    StarCabinetTopology,
    SubstationConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
    rod_circle,
)
from groundfield.generators._pen_routing import MIN_DROP_LENGTH_M

pytestmark = pytest.mark.finding("10.38")

# ---------------------------------------------------------------------
# Independent reference: dense sampling in plan
# ---------------------------------------------------------------------

_STEP_M = 0.002


def _sample(points: np.ndarray, step: float = _STEP_M) -> np.ndarray:
    """Points every ``step`` m along a polyline (plan coordinates)."""
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:], strict=True):
        n = max(1, math.ceil(float(np.hypot(*(b - a))) / step))
        out.append(a + np.linspace(0.0, 1.0, n + 1)[1:, None] * (b - a))
    return np.concatenate(out)


def _wires_of(
    world: gf.World, root: str
) -> list[gf.geometry.electrodes._ElectrodeBase]:
    """The wires of the foundation called ``root`` (anchor and ``_w<j>``)."""
    return [
        e for e in world.electrodes if e.name == root or e.name.startswith(f"{root}_w")
    ]


def _plan(electrode) -> np.ndarray:
    (run,) = electrode.wire_polylines()
    return np.asarray(run)[:, :2]


def _min_distance(points: np.ndarray, polylines: list[np.ndarray]) -> float:
    """Smallest plan distance from sampled ``points`` to the polylines.

    The points are samples of one geometry, the distance to each straight
    piece of the other is exact (projection): the result is the sampled
    minimum, at most one sampling step above the exact one.
    """
    best = math.inf
    for line in polylines:
        for a, b in zip(line[:-1], line[1:], strict=True):
            d = b - a
            t = np.clip((points - a) @ d / (d @ d), 0.0, 1.0)
            dist = np.hypot(*(points - (a + t[:, None] * d)).T)
            best = min(best, float(dist.min()))
    return best


def _on_wire(point: tuple[float, float], electrode, tol: float = 1e-9) -> bool:
    """Whether ``point`` lies on the wire (plan)."""
    pts = _plan(electrode)
    p = np.asarray(point, dtype=float)
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        d = b - a
        t = np.clip((p - a) @ d / (d @ d), 0.0, 1.0)
        if np.hypot(*(p - (a + t * d))) <= tol:
            return True
    return False


def _foundation_of(world: gf.World, wire: str) -> str:
    """Root name of the foundation that the wire ``wire`` belongs to."""
    return wire.split("_w")[0] if "_foundation_" in wire else wire


def _overlap_along(drop, wires, tol: float = 0.3) -> float:
    """Length in m over which ``drop`` runs within ``tol`` of, and parallel
    to, a wire (the metric of audit D3A, ``drop_overlap.py``)."""
    s, t = np.asarray(drop.start[:2]), np.asarray(drop.end[:2])
    total = 0.0
    for w in wires:
        pts = _plan(w)
        for b0, b1 in zip(pts[:-1], pts[1:], strict=True):
            da, db = t - s, b1 - b0
            la, lb = np.linalg.norm(da), np.linalg.norm(db)
            if la < 1e-9 or lb < 1e-9:
                continue
            ua, ub = da / la, db / lb
            if abs(abs(ua @ ub) - 1.0) > 1e-6:
                continue
            n = np.array([-ub[1], ub[0]])
            if abs((s - b0) @ n) > tol:
                continue
            t0, t1 = sorted([(s - b0) @ ub, (t - b0) @ ub])
            total += max(0.0, min(t1, lb) - max(t0, 0.0))
    return total


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (counter-clockwise) of plan points, monotone chain."""
    pts = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})

    def half(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while (
                len(out) >= 2
                and (
                    (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                    - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])
                )
                <= 0.0
            ):
                out.pop()
            out.append(p)
        return out[:-1]

    return np.asarray(half(pts) + half(pts[::-1]))


def _crosses_interior(drop, wires) -> bool:
    """Whether the drop passes over the inside of its (convex) foundation."""
    hull = _hull(np.concatenate([_plan(w) for w in wires]))
    s, t = np.asarray(drop.start[:2]), np.asarray(drop.end[:2])
    probe = s + np.linspace(0.0, 1.0, 401)[1:-1, None] * (t - s)
    edges = np.roll(hull, -1, axis=0) - hull
    rel = probe[:, None, :] - hull[None, :, :]
    cross = edges[None, :, 0] * rel[..., 1] - edges[None, :, 1] * rel[..., 0]
    return bool(np.any(np.all(cross > 1e-6, axis=1)))


def _ring_house(size=(12.0, 8.0), **kw) -> BuildingTypeSpec:
    return BuildingTypeSpec(
        name="residential",
        grounding=GroundingSystemSpec(
            electrodes=[
                FoundationElectrodeSpec(style="ring", size_xy_m=size, depth_m=0.8, **kw)
            ]
        ),
    )


def _build(**kw) -> gf.World:
    params: dict[str, object] = {"soil": HomogeneousSoilSpec(rho=100.0)}
    params.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TnNetworkGenerator(TnNetworkConfig(**params), seed=1).build()


def _drops(world: gf.World) -> list[gf.Conductor]:
    return [
        c for c in world.conductors if c.name.startswith(("pen_service_", "service_"))
    ]


# ---------------------------------------------------------------------
# The closest pair (unit level, two methods)
# ---------------------------------------------------------------------


# user decision Q41
def test_closest_pair_agrees_with_dense_sampling() -> None:
    """The analytic segment-segment solver against 2 mm sampling: 300 random
    cable paths of one to three sections against rotated rectangles."""
    from groundfield.generators._contact import FoundationWire, closest_pair

    rng = np.random.default_rng(20261114)
    worst = 0.0
    for _ in range(300):
        segments = [
            (tuple(rng.uniform(-20, 20, 2)), tuple(rng.uniform(-20, 20, 2)))
            for _ in range(int(rng.integers(1, 4)))
        ]
        cx, cy = rng.uniform(-10, 10, 2)
        dx, dy = rng.uniform(1, 12, 2)
        c, s = (
            math.cos(rng.uniform(0, 2 * math.pi)),
            math.sin(rng.uniform(0, 2 * math.pi)),
        )
        c, s = c / math.hypot(c, s), s / math.hypot(c, s)
        local = [
            (-dx / 2, -dy / 2),
            (dx / 2, -dy / 2),
            (dx / 2, dy / 2),
            (-dx / 2, dy / 2),
        ]
        corners = [(cx + c * x - s * y, cy + s * x + c * y) for x, y in local]
        wires = [
            FoundationWire(f"w{k}", k, (corners[k], corners[(k + 1) % 4]), 0.8)
            for k in range(4)
        ]
        i, t, contact = closest_pair(segments, wires)
        (x0, y0), (x1, y1) = segments[i]
        cable_point = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        assert math.dist(cable_point, contact.xy) == pytest.approx(
            contact.distance, abs=1e-9
        )
        sampled = _min_distance(
            np.concatenate([_sample(np.array(seg)) for seg in segments]),
            [np.array([corners[k], corners[(k + 1) % 4]]) for k in range(4)],
        )
        # The exact minimum is never above the sampled one and at most one
        # sampling step below it.
        assert contact.distance <= sampled + 1e-12
        worst = max(worst, sampled - contact.distance)
    assert worst < _STEP_M


# user decision Q41
def test_ties_go_to_the_lowest_wire_then_the_smallest_arc() -> None:
    from groundfield.generators._contact import (
        FoundationWire,
        closest_pair,
        nearest_point,
    )

    sides = [((-5.0, 5.0), (5.0, 5.0)), ((5.0, 5.0), (5.0, 15.0)),
             ((5.0, 15.0), (-5.0, 15.0)), ((-5.0, 15.0), (-5.0, 5.0))]  # fmt: skip
    wires = [FoundationWire(f"w{k}", k, side, 0.8) for k, side in enumerate(sides)]
    # A cable parallel to the first side: the whole side is 5 m away; the
    # start of the side (arc 0) wins.
    i, t, contact = closest_pair([((-50.0, 0.0), (50.0, 0.0))], wires)
    assert (contact.electrode, contact.xy, contact.distance) == ("w0", (-5.0, 5.0), 5.0)
    assert t == pytest.approx(0.45)
    # A cable through the house crosses w1 and w3: the lower index wins.
    _, _, contact = closest_pair([((-50.0, 10.0), (50.0, 10.0))], wires)
    assert contact.electrode == "w1" and contact.xy == pytest.approx((5.0, 10.0))
    assert contact.distance == pytest.approx(0.0, abs=1e-12)
    # A corner shared by two wires: the lower index (the end of w1).
    contact = nearest_point((20.0, 20.0), wires)
    assert (contact.electrode, contact.xy) == ("w1", (5.0, 15.0))


# ---------------------------------------------------------------------
# TnNetworkGenerator: service drops
# ---------------------------------------------------------------------

_D3A_PLACEMENT = ManhattanGridPlacement(
    spacing_x_m=25.0, spacing_y_m=30.0, n_per_row=6, jitter_m=4.0
)


def _d3a_world(n: int, topology: str, seed: int, **kw) -> gf.World:
    cfg = TnNetworkConfig(
        building_types=[_ring_house(**kw)],
        building_counts={"residential": n},
        placement=_D3A_PLACEMENT,
        pen_topology=StarCabinetTopology()
        if topology == "star"
        else RadialTrunkTopology(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TnNetworkGenerator(cfg).build(rng=seed)


# user decision Q41, audit D3A-02
@pytest.mark.parametrize(
    ("n", "topology", "seed"),
    list(itertools.product([5, 12, 30], ["star", "radial"], [0, 1, 2])),
)
def test_no_service_drop_runs_along_or_across_its_foundation(
    n: int, topology: str, seed: int
) -> None:
    """The 282 drops of the audit D3A setup (12 m x 8 m ring houses on a
    jittered grid): base 47 of them ran along a side of their own
    foundation (up to 8 m), and houses beyond their cable were connected at
    the far corner across the house."""
    world = _d3a_world(n, topology, seed)
    for drop in _drops(world):
        wires = _wires_of(world, _foundation_of(world, drop.end_electrode))
        assert _overlap_along(drop, wires) < 1e-6, drop.name
        assert not _crosses_interior(drop, wires), drop.name


# user decision Q41
@pytest.mark.parametrize("topology", ["star", "radial"])
@pytest.mark.parametrize("angle", [None, 30.0, 90.0])
def test_service_drops_are_the_closest_pair_of_cable_and_foundation(
    topology: str, angle: float | None
) -> None:
    """Every drop ends on a wire of its own house at the point nearest to
    the drop's start, and a drop from a tap junction is as short as the
    plan distance between its feeding cable and the foundation (2 mm
    sampling)."""
    world = _d3a_world(12, topology, 1, orientation_deg=angle)
    conductors = {c.name: c for c in world.conductors}
    cables: dict[str, list[gf.Conductor]] = {}
    for c in world.conductors:
        if c.name.startswith(("pen_main_", "pen_trunk_")):
            cables.setdefault(c.name.split("_s")[0], []).append(c)
    by_end = {}
    for c in conductors.values():
        if c.name.startswith(("pen_main_", "pen_trunk_")):
            by_end.setdefault(c.end_electrode, []).append(c.name.split("_s")[0])
    drops = _drops(world)
    assert drops
    for drop in drops:
        wire = world.get_electrode(drop.end_electrode)
        assert _on_wire(drop.end[:2], wire)
        root = _foundation_of(world, drop.end_electrode)
        assert root.startswith("residential_") and drop.name.endswith(root)
        lines = [_plan(w) for w in _wires_of(world, root)]
        start = np.asarray([drop.start[:2]])
        assert math.dist(drop.start[:2], drop.end[:2]) == pytest.approx(
            _min_distance(start, lines), abs=_STEP_M
        )
        if not drop.start_electrode.startswith("pen_tap_"):
            continue
        (cable,) = set(by_end[drop.start_electrode])
        path = np.concatenate(
            [_sample(np.array([c.start[:2], c.end[:2]]), 0.01) for c in cables[cable]]
        )
        assert math.dist(drop.start[:2], drop.end[:2]) == pytest.approx(
            _min_distance(path, lines), abs=0.01
        )


# user decision Q41
def test_a_drop_is_attached_to_the_wire_that_holds_its_end() -> None:
    """A house north of its cable is met on its -y side (the anchor, named
    after the foundation), a house south of it on its +y side (``_w1``); the
    sides stay bonded to the anchor."""
    world = _build(
        building_types=[_ring_house()],
        building_counts={"residential": 2},
        placement=ExplicitPlacement(positions=[(-40.0, 15.0), (-80.0, -15.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(-150.0, 0.0)]), fixed_count=1
        ),
    )
    drops = {d.name: d for d in _drops(world)}
    north = drops["pen_service_residential_0_foundation_0"]
    south = drops["pen_service_residential_1_foundation_0"]
    assert north.end_electrode == "residential_0_foundation_0"
    assert north.start[:2] == pytest.approx((-46.0, 0.0))
    assert north.end == pytest.approx((-46.0, 11.0, 0.6))
    assert south.end_electrode == "residential_1_foundation_0_w1"
    assert south.end == pytest.approx((-86.0, -11.0, 0.6))
    bonds = {c.name: (c.start_electrode, c.end_electrode) for c in world.conductors}
    assert bonds["residential_1_foundation_0_bond_1"] == (
        "residential_1_foundation_0",
        "residential_1_foundation_0_w1",
    )


# user decision Q41, audit D3B-03
def test_a_house_on_the_main_is_joined_where_the_main_crosses_it() -> None:
    """A 10 m ring house centred on the main. Base: a 5 m drop from the main
    to the (-5, -5) m corner (the 0.15.0 grid mesh had its centre on the main
    and no drop). The main crosses the -x side (``_w2``, x = -65 m) and the
    +x side (``_w3``, x = -55 m); the lower index wins and the main is joined
    to ``_w2`` there, without a drop."""
    house = BuildingTypeSpec(
        name="residential",
        grounding=GroundingSystemSpec(
            electrodes=[FoundationElectrodeSpec(style="ring", size_m=10.0, depth_m=0.8)]
        ),
    )
    world = _build(
        building_types=[house],
        building_counts={"residential": 1},
        placement=ExplicitPlacement(positions=[(-60.0, 0.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(-150.0, 0.0)]), fixed_count=1
        ),
    )
    assert _drops(world) == []
    sections = {c.name: c for c in world.conductors if c.name.startswith("pen_main_")}
    first, second = (
        sections["pen_main_kvs_0_rod_0"],
        sections["pen_main_kvs_0_rod_0_s1"],
    )
    assert (
        first.end_electrode == second.start_electrode == "residential_0_foundation_0_w2"
    )
    assert first.end[:2] == pytest.approx((-65.0, 0.0))
    assert second.start[:2] == pytest.approx((-65.0, 0.0))


# user decision Q41
def test_a_cabinet_inside_a_foundation_meets_it_one_section_from_the_cabinet() -> None:
    """50 default houses with five cabinets on the middle row: the rod of
    ``kvs_3`` at (-60, 0) m stands inside house 22, on the brace of its
    2 x 2 mesh, which its cable runs along. The cable is joined to the
    foundation one minimum section length before the cabinet instead of a
    drop of no length (the first attempt of the decision raised "coincides")."""
    cfg = TnNetworkConfig(
        building_counts={"residential": 50},
        cabinets=CabinetConfig(
            count_per_100_buildings=10.0,
            placement=ManhattanGridPlacement(
                spacing_x_m=40.0, spacing_y_m=1.0, n_per_row=10
            ),
        ),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    pen = [c for c in world.conductors if c.conductor_type == "pen"]
    assert min(c.length for c in pen) >= MIN_DROP_LENGTH_M - 1e-12
    joined = [
        c for c in pen
        if c.end_electrode == "kvs_3_rod_0" and c.start_electrode.startswith("residential_22")
    ]  # fmt: skip
    (last,) = joined
    assert last.length == pytest.approx(MIN_DROP_LENGTH_M)


# user decision Q41
def test_sites_without_a_foundation_keep_their_connection_point() -> None:
    """Houses with a rod only: the drop ends at the rod head (its connection
    point), attached to the rod, as before the decision."""
    house = BuildingTypeSpec(
        name="residential",
        grounding=GroundingSystemSpec(electrodes=[RodElectrodeSpec(length_m=2.0)]),
    )
    world = _build(
        building_types=[house],
        building_counts={"residential": 3},
        placement=ManhattanGridPlacement(
            spacing_x_m=25.0, spacing_y_m=30.0, n_per_row=3
        ),
    )
    for drop in _drops(world):
        rod = world.get_electrode(drop.end_electrode)
        assert rod.kind == "rod"
        assert drop.end[:2] == pytest.approx(rod.connection_point[:2])


# user decision Q41
def test_cables_leave_a_substation_foundation_at_its_nearest_point() -> None:
    """AP1-like substation: a 4 m x 4 m ring at 0.6 m with three 4 m rods.
    Each cabinet cable starts at the point of the ring nearest to its
    cabinet, attached to that side; the connection point of the anchor (the
    (-2, -2) m corner) is no longer the start of every cable."""
    substation = SubstationConfig(
        grounding=GroundingSystemSpec(
            electrodes=[
                FoundationElectrodeSpec(
                    style="ring",
                    size_xy_m=(4.0, 4.0),
                    depth_m=0.6,
                    wire_radius_m=0.0075,
                ),
                *rod_circle(
                    3, radius_m=1.5, length_m=4.0, depth_m=0.6, wire_radius_m=0.0125
                ),
            ]
        )
    )
    world = _build(
        substation=substation,
        building_counts={"residential": 2},
        placement=ExplicitPlacement(positions=[(-40.0, 20.0), (25.0, 60.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(-100.0, 0.0), (30.0, 100.0)]),
            fixed_count=2,
        ),
    )
    mains = {c.name: c for c in world.conductors if c.name.startswith("pen_main_")}
    west = mains["pen_main_kvs_0_rod_0"]
    north = mains["pen_main_kvs_1_rod_0"]
    assert west.start_electrode == "trafo_foundation_0_w2"
    assert west.start[:2] == pytest.approx((-2.0, 0.0))
    assert north.start_electrode == "trafo_foundation_0_w1"
    # The nearest point of the +y side to (30, 100): its +x end.
    assert north.start[:2] == pytest.approx((2.0, 2.0))
    lines = [_plan(w) for w in _wires_of(world, "trafo_foundation_0")]
    for cable, cabinet in ((west, (-100.0, 0.0)), (north, (30.0, 100.0))):
        assert math.dist(cable.start[:2], cabinet) == pytest.approx(
            _min_distance(np.asarray([cabinet]), lines), abs=_STEP_M
        )


# ---------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------


def _l_footprint(cx: float, cy: float, osm_id: int) -> gf.BuildingFootprint:
    """An L-shaped outline, 12 m x 10 m with a 5 m x 4 m notch."""
    pts = [(0, 0), (12, 0), (12, 6), (7, 6), (7, 10), (0, 10)]
    return gf.BuildingFootprint(
        polygon_xy_m=[(cx + x - 6.0, cy + y - 5.0) for x, y in pts],
        building_use="residential",
        osm_id=osm_id,
    )


# user decision Q41
def test_layout_drops_are_the_closest_pair_of_cable_and_building_outline() -> None:
    """AP1-like layout: building outlines (``foundation_outline="polygon"``)
    in a 5 cm shell of 100 ohm m, the station ring 4 m x 4 m with three rods.
    Each drop runs from its cable to the point of the outline nearest to the
    cable; each cable leaves the station at the point of the ring nearest to
    its next node."""
    fps = [_l_footprint(20.0 + 25.0 * i, 15.0 if i % 2 else -15.0, i) for i in range(4)]
    layout = LvNetworkLayout.from_footprints(fps, substation_xy=(0.0, 0.0))
    layout.add_pen_cable(start="substation", end_xy=(110.0, 0.0))
    layout.add_pen_cable(start="substation", end_xy=(0.0, -60.0))
    layout.connect_buildings()
    house = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                style="ring",
                depth_m=0.8,
                concrete_rho_ohm_m=100.0,
                concrete_thickness_m=0.05,
            )
        ]
    )
    station = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                style="ring", size_xy_m=(4.0, 4.0), depth_m=0.6, wire_radius_m=0.0075
            ),
            *rod_circle(
                3, radius_m=1.5, length_m=4.0, depth_m=0.6, wire_radius_m=0.0125
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = layout.to_world(
            house_grounding=house,
            substation_grounding=station,
            foundation_outline="polygon",
            seed=0,
        )
    drops = _drops(world)
    assert len(drops) == 4
    cables = {c.name: [] for c in layout.pen_cables}
    for c in world.conductors:
        if c.name.startswith("pen_"):
            cables[c.name.rsplit("_seg_", 1)[0]].append(c)
    for drop in drops:
        outline = world.get_electrode(drop.end_electrode)
        assert outline.kind == "polyline" and _on_wire(drop.end[:2], outline)
        (connection,) = [
            c for c in layout.connections if f"service_{c.house_idx:03d}" == drop.name
        ]
        path = np.concatenate(
            [
                _sample(np.array([c.start[:2], c.end[:2]]), 0.01)
                for c in cables[connection.cable_name]
            ]
        )
        plan_length = math.dist(drop.start[:2], drop.end[:2])
        assert plan_length == pytest.approx(
            _min_distance(path, [_plan(outline)]), abs=0.01
        )
        assert drop.end[2] == pytest.approx(0.8)
    starts = [
        c
        for c in world.conductors
        if c.conductor_type == "pen" and c.start_electrode.startswith("substation_")
    ]
    assert {c.start[:2] for c in starts} == {(2.0, 0.0), (0.0, -2.0)}


# ---------------------------------------------------------------------
# Plan data
# ---------------------------------------------------------------------


# user decision Q41
def test_plan_contacts_round_trip_and_are_checked() -> None:
    from groundfield.generators.plan import NetworkPlan, PlannedContact

    plan = TnNetworkGenerator(
        TnNetworkConfig(
            building_types=[_ring_house()], building_counts={"residential": 4}
        ),
        seed=0,
    ).plan()
    contacts = [b.end_contact for b in plan.branches if b.end_contact is not None]
    assert contacts and all(isinstance(c, PlannedContact) for c in contacts)
    again = NetworkPlan.model_validate_json(plan.model_dump_json())
    assert gf.world_fingerprint(again.to_world()) == gf.world_fingerprint(
        plan.to_world()
    )
    data = plan.model_dump()
    drop = next(b for b in data["branches"] if b["end_contact"] is not None)
    drop["end_contact"]["electrode"] = "trafo_ring_0"
    with pytest.raises(ValueError, match="contact names an electrode of the site"):
        NetworkPlan.model_validate(data)


# user decision Q41 (item 4: not changed)
def test_the_connection_point_of_a_foundation_stays_its_corner() -> None:
    """``site_anchor_point`` and conductors attached by name still use the
    connection point of the anchor: the start of the local -y side."""
    from groundfield.generators.materialise import site_anchor_point

    plan = TnNetworkGenerator(
        TnNetworkConfig(
            building_types=[_ring_house()],
            building_counts={"residential": 1},
            placement=ExplicitPlacement(positions=[(30.0, 20.0)]),
        ),
        seed=0,
    ).plan()
    house = plan.site("residential_0")
    assert site_anchor_point(house) == (24.0, 16.0, 0.8)
    world = plan.to_world()
    lead = gf.create_conductor(
        world, name="lead", start="residential_0_foundation_0", end="trafo_ring_0"
    )
    assert lead.start == (24.0, 16.0, 0.8)


# user decision Q41
def test_inductive_networks_with_nearest_point_drops_stay_passive() -> None:
    """Ring houses with a Neumann PEN at 1 kHz (perfect mirror)."""
    world = _build(
        building_types=[_ring_house()],
        building_counts={"residential": 6},
        placement=ManhattanGridPlacement(
            spacing_x_m=25.0, spacing_y_m=30.0, n_per_row=5
        ),
        pen=PenConfig(inductance_model="neumann", segment_length_m=5.0),
    )
    engine = gf.create_engine(
        segment_length=2.0,
        frequencies=[50.0, 1000.0],
        earth_inductive_model="perfect_mirror",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = np.asarray(engine.solve(world).input_impedance())
    assert np.all(z.real > 0.0) and np.all(z.imag > 0.0)


# user decision Q41
def test_a_layout_drop_from_the_station_leaves_its_foundation_at_the_nearest_point() -> (
    None
):
    """A house east of a cable that runs west from the station: its closest
    point to the cable is the cable's start at the station, so the drop
    leaves the station. It runs between the closest pair of the station ring
    and the house outline, (0, 2) m on the ring's +y side to the outline's
    first vertex (0, 15) m (first version: from the ring's (-2, -2) m
    corner, the anchor's connection point)."""
    layout = LvNetworkLayout.from_footprints(
        [_l_footprint(6.0, 20.0, 0)], substation_xy=(0.0, 0.0)
    )
    layout.add_pen_cable(start="substation", end_xy=(-80.0, 0.0))
    layout.connect_buildings()
    station = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(style="ring", size_xy_m=(4.0, 4.0), depth_m=0.6),
            *rod_circle(3, radius_m=1.5, length_m=4.0, depth_m=0.6),
        ]
    )
    house = GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", depth_m=0.8)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = layout.to_world(
            house_grounding=house,
            substation_grounding=station,
            foundation_outline="polygon",
        )
    (drop,) = _drops(world)
    ring = [_plan(w) for w in _wires_of(world, "substation_foundation_0")]
    outline = _plan(world.get_electrode(drop.end_electrode))
    assert drop.start_electrode == "substation_foundation_0_w1"
    assert drop.start[:2] == pytest.approx((0.0, 2.0))
    assert drop.end[:2] == pytest.approx((0.0, 15.0))
    assert _on_wire(drop.start[:2], world.get_electrode(drop.start_electrode))
    sampled = _min_distance(_sample(outline, 0.01), ring)
    assert math.dist(drop.start[:2], drop.end[:2]) == pytest.approx(sampled, abs=0.01)


# user decision Q41 (attack of stream f4gen)
def test_layout_houses_without_a_foundation_keep_the_rod_head() -> None:
    """Rod-only houses in the layout builder: the drop ends at the rod head,
    attached to the rod, as before the decision (the first version of the
    decision raised a KeyError for them)."""
    fps = [
        gf.BuildingFootprint(
            polygon_xy_m=[
                (20 + 25 * i, 8),
                (30 + 25 * i, 8),
                (30 + 25 * i, 16),
                (20 + 25 * i, 16),
            ],
            osm_id=i,
        )
        for i in range(3)
    ]
    layout = LvNetworkLayout.from_footprints(fps, substation_xy=(0.0, 0.0))
    layout.add_pen_cable(start="substation", end_xy=(110.0, 0.0))
    layout.connect_buildings()
    rods = GroundingSystemSpec(electrodes=[RodElectrodeSpec(length_m=2.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = layout.to_world(house_grounding=rods)
    drops = _drops(world)
    assert len(drops) == 3
    for drop in drops:
        rod = world.get_electrode(drop.end_electrode)
        assert rod.kind == "rod" and drop.end == rod.connection_point


# ---------------------------------------------------------------------
# Audits R4A / R4B: the drop taps its cable, crossed buildings are joined,
# no drop lies on a cable
# ---------------------------------------------------------------------


def _collinear_overlap(a, b, tol: float = MIN_DROP_LENGTH_M) -> float:
    """Plan length in m over which the straight conductors ``a`` and ``b``
    run on one line: both ends of ``b`` within ``tol`` of the line of ``a``
    (default one minimum section, 1 cm: the tolerance of the router's rule
    that a drop leaves a cable it would run along, audit R5A-01)."""
    a0, a1 = np.asarray(a.start[:2]), np.asarray(a.end[:2])
    b0, b1 = np.asarray(b.start[:2]), np.asarray(b.end[:2])
    la, lb = np.linalg.norm(a1 - a0), np.linalg.norm(b1 - b0)
    if la < 1e-12 or lb < 1e-12:
        return 0.0
    ua = (a1 - a0) / la
    normal = np.array([-ua[1], ua[0]])
    if abs((b0 - a0) @ normal) > tol or abs((b1 - a0) @ normal) > tol:
        return 0.0
    lo, hi = sorted([(b0 - a0) @ ua, (b1 - a0) @ ua])
    return max(0.0, min(hi, la) - max(lo, 0.0))


def _cables(world: gf.World) -> list[gf.Conductor]:
    return [
        c
        for c in world.conductors
        if c.name.startswith(("pen_main_", "pen_trunk_")) and c.length > 0
    ]


def _guard_world(kind: str, n: int, topology: str, seed: int) -> gf.World:
    if kind == "d3a":
        return _d3a_world(n, topology, seed)
    cfg = TnNetworkConfig(
        building_counts={"residential": n},
        pen_topology=StarCabinetTopology()
        if topology == "star"
        else RadialTrunkTopology(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TnNetworkGenerator(cfg).build(rng=seed)


# user decision Q41, audits R4A-02, R4B-01
@pytest.mark.parametrize("topology", ["star", "radial"])
@pytest.mark.parametrize(
    ("kind", "n", "seed"),
    [("default", n, s) for n in (30, 50, 150) for s in (0, 1)]
    + [("d3a", n, s) for n in (5, 12, 30) for s in (0, 1, 2)],
)
def test_no_drop_lies_on_a_cable(kind: str, n: int, topology: str, seed: int) -> None:
    """No service drop runs along a cabinet cable or a trunk. Snapshot
    ``e91f3c0``: the substation-slot drops of the radial networks ran along
    the trunk through the houses it crosses (default 30 houses: 6 drops,
    195 m; D3A's set: 18 drops, 566 m), and in the star default grid of 50 /
    150 houses a house on the row between collinear cabinets got a drop
    along the mains of the cabinets beyond its own (7.5 m)."""
    world = _guard_world(kind, n, topology, seed)
    cables = _cables(world)
    for drop in _drops(world):
        along = sum(_collinear_overlap(drop, c) for c in cables)
        assert along < 1e-6, (drop.name, along)


# user decision Q41, audit R4A-01
@pytest.mark.parametrize("n", [30, 50, 150])
def test_every_star_drop_starts_on_its_cable_not_on_a_neighbours_foundation(
    n: int,
) -> None:
    """Default star networks: a drop whose closest pair lies at the joint of
    a house that the main crosses started on that house's foundation, 5 m
    from the cable (snapshot: 10 / 16 / 70 drops for n = 30 / 50 / 150). Now
    every drop starts on a cable, at its tap point: the foot of the drop's
    end on the cable, up to the snapping distance."""
    world = _guard_world("default", n, "star", 0)
    cables = _cables(world)
    for drop in _drops(world):
        foot = min(
            (_nearest_foot(c, drop.start[:2]) for c in cables),
            key=lambda p: math.dist(p, drop.start[:2]),
        )
        assert math.dist(foot, drop.start[:2]) < 1e-9, drop.name
        if drop.start_electrode.startswith("residential_"):
            # It starts at the joint of the crossed house, on its own
            # foundation wire's contact with the cable.
            assert _on_wire(
                drop.start[:2], world.get_electrode(drop.start_electrode), 0.01
            )


def _nearest_foot(conductor: gf.Conductor, point) -> tuple[float, float]:
    """Point of the straight conductor nearest to ``point`` (plan)."""
    a, b = np.asarray(conductor.start[:2]), np.asarray(conductor.end[:2])
    d = b - a
    t = float(np.clip((np.asarray(point) - a) @ d / (d @ d), 0.0, 1.0))
    return (float(a[0] + t * d[0]), float(a[1] + t * d[1]))


# user decision Q41, audit R4A-02 (radial)
def test_radial_houses_on_a_trunk_are_joined_and_get_no_drop() -> None:
    """Default radial network of 30 houses: the trunks pen_trunk_f0_0 and
    pen_trunk_f2_0 cross houses 12 ... 17, which are fed from the substation
    slots. Snapshot: none was joined, their drops from the substation ran
    along the trunk (195 m); base 500127c: drops to the corner. Now each is
    joined to the trunk where it crosses the house, without a drop."""
    world = _guard_world("default", 30, "radial", 0)
    drops = {d.name for d in _drops(world)}
    ends = {
        end.split("_w")[0]
        for c in _cables(world)
        for end in (c.start_electrode, c.end_electrode)
        if c.name.startswith("pen_trunk_")
    }
    for k in range(12, 18):
        root = f"residential_{k}_foundation_0"
        assert root in ends
        assert f"pen_service_{root}" not in drops
    assert len(drops) == 24


# user decision Q41, audit R4A-02 (radial, a trunk beyond the tap source)
def test_a_trunk_beyond_the_tap_source_that_crosses_a_house_is_joined() -> None:
    """Default radial network of 50 houses: house 28 is fed by the cabinet
    kvs_f0_0 at (60, 0) m; the trunk on to kvs_f0_1 crosses it. Snapshot: its
    drop ran 22.5 m along that trunk from the cabinet."""
    world = _guard_world("default", 50, "radial", 0)
    assert "pen_service_residential_28_foundation_0" not in {
        d.name for d in _drops(world)
    }
    joined = [
        c
        for c in _cables(world)
        if c.name.startswith("pen_trunk_f0_1")
        and "residential_28_foundation_0" in (c.start_electrode + c.end_electrode)
    ]
    assert joined


# user decision Q41, audits R4A-02, R4B-01 (star)
def test_a_house_that_a_main_crosses_is_fed_by_that_main() -> None:
    """A 10 m ring house at (-60, 0) m: the main of the cabinet at
    (-150, 0) m runs through it, its nearest cabinet is the one at
    (-62, 30) m. Snapshot: fed by the nearest cabinet with a 19.9 m drop,
    the other main running through the house unjoined. Now it is joined to
    the main that crosses it and fed by that main's cabinet."""
    house = BuildingTypeSpec(
        name="residential",
        grounding=GroundingSystemSpec(
            electrodes=[FoundationElectrodeSpec(style="ring", size_m=10.0, depth_m=0.8)]
        ),
    )
    world = _build(
        building_types=[house],
        building_counts={"residential": 1},
        placement=ExplicitPlacement(positions=[(-60.0, 0.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(-150.0, 0.0), (-62.0, 30.0)]),
            fixed_count=2,
        ),
    )
    assert _drops(world) == []
    main = [c for c in _cables(world) if c.name.startswith("pen_main_kvs_0_rod_0")]
    assert any(
        c.end_electrode == "residential_0_foundation_0_w2"
        and c.end[:2] == pytest.approx((-65.0, 0.0))
        for c in main
    )


# user decision Q41, audit R4A-02 (star, several mains)
def test_a_house_on_several_mains_is_fed_by_the_nearest_of_their_cabinets() -> None:
    """Default star network of 50 houses: house 20 at (-112.5, 0) m lies on
    the mains of kvs_0 (-180, 0) and kvs_1 (-140, 0) m; its nearest cabinet is
    kvs_2 (-100, 0) m. It is joined to the main of kvs_1, the nearer of the
    two cabinets whose mains cross it; the main of kvs_0 passes it unjoined."""
    world = _guard_world("default", 50, "star", 0)
    assert "pen_service_residential_20_foundation_0" not in {
        d.name for d in _drops(world)
    }

    def touches(prefix: str) -> bool:
        return any(
            "residential_20_foundation_0" in (c.start_electrode + c.end_electrode)
            for c in _cables(world)
            if c.name.startswith(prefix)
        )

    assert touches("pen_main_kvs_1_rod_0")
    assert not touches("pen_main_kvs_0_rod_0")
    assert not touches("pen_main_kvs_2_rod_0")


# user decision Q41, audit R4A-03
@pytest.mark.parametrize("end_site", ["station", "cabinet"])
def test_layout_tap_never_lies_inside_a_site_foundation(end_site: str) -> None:
    """A house outline (0.5, 6)-(8.5, 12) m north of a cable between a 4 m x
    4 m station ring at (0, 0) and (60, 0) m. Snapshot: the closest pair was
    taken on the layout polyline, which starts at the station's waypoint
    inside the ring: the tap junction (0.5, 0) m lay inside the ring, and the
    cable ran from the ring's +x side (2, 0) m back to it and forward again
    over the same 1.5 m. Now the closest pair is taken on the cable as built
    (from (2, 0) m): its point at the ring is the station, and the drop is
    the closest pair of ring and outline, (0.5, 2) -> (0.5, 6) m. The same
    holds at a cabinet with a foundation at the cable's end (a ring at
    (60, 0) m, the outline from x = 58.5 m: snapshot tap (58.5, 0) m)."""
    ring = FoundationElectrodeSpec(style="ring", size_xy_m=(4.0, 4.0), depth_m=0.6)
    station = GroundingSystemSpec(
        electrodes=[ring, *rod_circle(3, radius_m=1.5, length_m=4.0, depth_m=0.6)]
    )
    if end_site == "station":
        outline = [(0.5, 6.0), (8.5, 6.0), (8.5, 12.0), (0.5, 12.0)]
        sub_xy, cab_xy = (0.0, 0.0), None
    else:
        outline = [(58.5, 6.0), (66.5, 6.0), (66.5, 12.0), (58.5, 12.0)]
        sub_xy, cab_xy = (-40.0, 0.0), (60.0, 0.0)
    fp = gf.BuildingFootprint(
        polygon_xy_m=outline, building_use="residential", osm_id=1
    )
    lay = LvNetworkLayout.from_footprints(
        [fp], substation_xy=sub_xy, obstacle_clearance_m=0.0
    )
    if cab_xy is None:
        lay.add_pen_cable(start="substation", end_xy=(60.0, 0.0))
    else:
        cabinet = lay.add_cabinet(cab_xy, name="kvs_0")
        lay.add_pen_cable(start="substation", end=cabinet)
    lay.connect_buildings()
    house = GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", depth_m=0.8)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = lay.to_world(
            house_grounding=house,
            substation_grounding=station,
            cabinet_grounding=station,
            foundation_outline="polygon",
        )
    centre = (0.0, 0.0) if cab_xy is None else cab_xy
    for j in world.junctions:
        x, y, _ = j.connection_point
        assert not (abs(x - centre[0]) < 2.0 and abs(y - centre[1]) < 2.0), j.name
    pen = [c for c in world.conductors if c.name.startswith("pen_")]
    shared = sum(
        _collinear_overlap(a, b) for k, a in enumerate(pen) for b in pen[k + 1 :]
    )
    assert shared < 1e-9
    (drop,) = _drops(world)
    x_drop = 0.5 if cab_xy is None else 58.5
    assert drop.start[:2] == pytest.approx((x_drop, 2.0))
    assert drop.end[:2] == pytest.approx((x_drop, 6.0))


def _segment_distance(a0, a1, b0, b1) -> float:
    """Exact plan distance in m between the segments ``a0 a1`` and ``b0 b1``
    (zero when they cross; otherwise the smallest of the four end-point
    projections)."""
    a0, a1, b0, b1 = (np.asarray(p, dtype=float)[:2] for p in (a0, a1, b0, b1))

    def orient(p, q, r) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    if (
        orient(a0, a1, b0) * orient(a0, a1, b1) < 0.0
        and orient(b0, b1, a0) * orient(b0, b1, a1) < 0.0
    ):
        return 0.0

    def to_segment(p, s0, s1) -> float:
        d = s1 - s0
        n2 = float(d @ d)
        t = 0.0 if n2 == 0.0 else float(np.clip((p - s0) @ d / n2, 0.0, 1.0))
        return float(np.hypot(*(p - (s0 + t * d))))

    return min(
        to_segment(a0, b0, b1),
        to_segment(a1, b0, b1),
        to_segment(b0, a0, a1),
        to_segment(b1, a0, a1),
    )


_SITE_FOUNDATIONS = {
    "station rotated 30": ("station", {"size_m": 4.0, "orientation_deg": 30.0}),
    "station polygon": (
        "station",
        {"polygon_xy_m": ((-2.0, -2.0), (2.5, -1.0), (2.0, 2.0), (-1.5, 2.5))},
    ),
    "station offset": ("station", {"size_m": 4.0, "offset_xy_m": (0.0, 3.0)}),
    # attack of stream f6gen: the station's foundation beyond the cable's
    # first waypoint; the clip lands on that waypoint, the drops start there
    "station beyond the first waypoint": (
        "station",
        {"size_m": 4.0, "offset_xy_m": (90.0, -3.0)},
    ),
    "cabinet rotated 30": ("cabinet", {"size_m": 2.0, "orientation_deg": 30.0}),
    "cabinet polygon": (
        "cabinet",
        {"polygon_xy_m": ((-1.0, -1.0), (1.2, -0.5), (1.0, 1.0), (-0.8, 1.3))},
    ),
}


# user decision Q41, audit R5A-02 (regression of the R4A-03 fix)
@pytest.mark.parametrize("case", list(_SITE_FOUNDATIONS))
def test_layout_drops_tap_the_cable_next_to_any_site_foundation(case: str) -> None:
    """Two houses north of an 80 m layout cable whose station (or end
    cabinet) has a rotated, polygon or offset foundation. ``e91f3c0`` tapped
    the waypoint line: 6 m drops (up to 0.13 m above the closest pair of the
    built cable, whose first / last section is slanted). Snapshot
    ``f7db449``: the taps were computed on the slanted section, refused by
    the axis-aligned cable, and the drops fell back to the station, the
    cabinet or the far end node, 14.9 … 36.5 m. Now the taps stay on the
    waypoint line, clipped at the projection of the site contact on it."""
    where, spec = _SITE_FOUNDATIONS[case]
    site = GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", depth_m=0.6, **spec)]
    )
    plain = GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", size_m=4.0, depth_m=0.6)]
    )
    xs = [20.0, 40.0] if where == "station" else [40.0, 60.0]
    footprints = [
        gf.BuildingFootprint(
            polygon_xy_m=[(x - 4, 6), (x + 4, 6), (x + 4, 14), (x - 4, 14)],
            building_use="residential",
            osm_id=k + 1,
        )
        for k, x in enumerate(xs)
    ]
    lay = LvNetworkLayout.from_footprints(
        footprints, substation_xy=(0.0, 0.0), obstacle_clearance_m=0.0
    )
    if where == "station":
        lay.add_pen_cable(start="substation", end_xy=(80.0, 0.0))
    else:
        lay.add_pen_cable(
            start="substation", end=lay.add_cabinet((80.0, 0.0), name="kvs_0")
        )
    lay.connect_buildings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = lay.to_world(
            house_grounding=GroundingSystemSpec(
                electrodes=[FoundationElectrodeSpec(style="ring", depth_m=0.8)]
            ),
            substation_grounding=site if where == "station" else plain,
            cabinet_grounding=site,
            foundation_outline="polygon",
        )
    cable = [c for c in world.conductors if c.name.startswith("pen_")]
    junctions = {j.name for j in world.junctions}
    drops = _drops(world)
    assert len(drops) == 2
    for drop in drops:
        house = _plan(world.get_electrode(drop.end_electrode))
        assert drop.start_electrode in junctions, drop.name
        assert (
            min(
                math.dist(_nearest_foot(c, drop.start[:2]), drop.start[:2])
                for c in cable
            )
            < 1e-9
        )
        pair = min(
            _segment_distance(c.start, c.end, a, b)
            for c in cable
            for a, b in zip(house[:-1], house[1:], strict=True)
        )
        assert drop.length <= pair + 0.2, (drop.name, drop.length, pair)
    # No tap or corner junction inside the site foundation (R4A-03): even-odd
    # ray casting against the site's ring wires.
    root = "substation_foundation_0" if where == "station" else "kvs_0_foundation_0"
    sides = [
        (a, b)
        for wire in _wires_of(world, root)
        for a, b in zip(_plan(wire)[:-1], _plan(wire)[1:], strict=True)
    ]
    for j in world.junctions:
        x, y, _ = j.connection_point
        inside = False
        for (ax, ay), (bx, by) in sides:
            if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                inside = not inside
        assert not inside, j.name


# user decision Q41, audit R5A-01
@pytest.mark.parametrize("dy", [0.0, 2e-6, 1e-4, 1e-3, 9e-3])
def test_a_drop_beside_a_trunk_leaves_it_at_its_end(dy: float) -> None:
    """Radial feeder along +x with cabinets at 60 and 120 m; house 8, a 10 m
    ring at (150, 5 + dy) m, is fed by the cabinet at 60 m and its -y side
    lies ``dy`` beside the trunk line. Snapshot ``f7db449``: at dy = 0 the
    drop leaves the trunk at kvs_f0_1 (25 m), from dy = 2 um its drop runs
    85 m from kvs_f0_0, 60 m of it within 0.7 mm of the trunk (the rule
    tested collinearity to 1e-6 m). Now a section counts as shared when
    both its ends lie within one minimum section (1 cm) of the drop line."""
    positions = (
        [(10.0, 40.0)]
        + [(70.0 + 12.0 * k, 40.0) for k in range(7)]
        + [(150.0, 5.0 + dy), (200.0, 40.0)]
    )
    world = _build(
        building_types=[_ring_house((10.0, 10.0))],
        building_counts={"residential": len(positions)},
        placement=ExplicitPlacement(positions=positions),
        pen_topology=RadialTrunkTopology(
            n_feeders=1, slots_per_substation=1, slots_per_cabinet=8
        ),
    )
    trunks = [c for c in _cables(world) if c.name.startswith("pen_trunk_")]
    for drop in _drops(world):
        near = _length_within(drop, trunks, MIN_DROP_LENGTH_M)
        assert near < 0.5, (drop.name, near)
    (drop,) = [
        d for d in _drops(world) if d.name == "pen_service_residential_8_foundation_0"
    ]
    assert drop.start_electrode == "kvs_f0_1_rod_0"
    assert drop.length == pytest.approx(25.0, abs=1e-3)


def _length_within(drop: gf.Conductor, cables, tol: float, step: float = 0.01) -> float:
    """Plan length in m of ``drop`` that lies within ``tol`` of a cable
    (sampled every ``step`` m, exact distances to the straight cables)."""
    points = _sample(np.array([drop.start[:2], drop.end[:2]]), step)
    lines = [np.array([c.start[:2], c.end[:2]]) for c in cables]
    close = np.zeros(len(points), dtype=bool)
    for line in lines:
        a, b = line
        d = b - a
        t = np.clip((points - a) @ d / (d @ d), 0.0, 1.0)
        close |= np.hypot(*(points - (a + t[:, None] * d)).T) <= tol
    return float(close.sum()) * drop.length / max(len(points) - 1, 1)


_ROD_STATION = SubstationConfig(
    position=(50.0, 100.0),
    grounding=GroundingSystemSpec(electrodes=[RodElectrodeSpec(length_m=2.0)]),
)


# user decision Q41, audit R5A-03 (star)
def test_a_main_to_a_cabinet_on_a_house_wire_is_joined_where_it_crosses() -> None:
    """A main from the substation at (50, 100) m runs south to a cabinet at
    (50, -5) m on the south wire of house 0 (10 m ring at (50, 0) m) and
    crosses the house's north wire at (50, 5) m. Snapshot ``f7db449`` (and
    ``e91f3c0``): the tie of the two zero distances went to the cabinet's
    node, the 1 cm step back left 1 cm, the house got a 1 cm drop on the
    main and the main crossed it unjoined. Now the main is joined to the
    house where it crosses it."""
    world = _build(
        building_types=[_ring_house((10.0, 10.0))],
        building_counts={"residential": 2},
        placement=ExplicitPlacement(positions=[(50.0, 0.0), (50.0, 40.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(50.0, -5.0)]), fixed_count=1
        ),
        substation=_ROD_STATION,
    )
    assert "pen_service_residential_0_foundation_0" not in {
        d.name for d in _drops(world)
    }
    joints = [
        c.end[:2]
        for c in _cables(world)
        if c.end_electrode.startswith("residential_0_foundation_0")
    ]
    assert joints == [pytest.approx((50.0, 5.0))]
    assert all(d.length > 1.0 for d in _drops(world))


# user decision Q41, audit R5A-03 (radial)
def test_a_trunk_beyond_a_cabinet_on_a_house_wire_is_joined() -> None:
    """Radial feeder along +x; the cabinet kvs_f0_0 at (60, 0) m lies on the
    west wire of house 1 at (65, 0) m, which it feeds; the trunk on to
    kvs_f0_1 crosses the house's east wire at (70, 0) m. Snapshot
    ``f7db449``: a 1 cm drop from a tap 1 cm before the cabinet, the trunk
    through the house unjoined. Now that trunk is joined to the house where
    it crosses its east wire, and the house has no drop."""
    positions = [(20.0, 15.0), (65.0, 0.0)] + [
        (80.0 + 10.0 * k, 20.0) for k in range(8)
    ]
    world = _build(
        building_types=[_ring_house((10.0, 10.0))],
        building_counts={"residential": len(positions)},
        placement=ExplicitPlacement(positions=positions),
        pen_topology=RadialTrunkTopology(
            n_feeders=1, slots_per_substation=1, slots_per_cabinet=8
        ),
        substation=SubstationConfig(
            position=(0.0, 0.0),
            grounding=GroundingSystemSpec(electrodes=[RodElectrodeSpec(length_m=2.0)]),
        ),
    )
    assert "pen_service_residential_1_foundation_0" not in {
        d.name for d in _drops(world)
    }
    joints = [
        c.end[:2]
        for c in _cables(world)
        if c.name.startswith("pen_trunk_f0_1")
        and c.end_electrode.startswith("residential_1_foundation_0")
    ]
    assert joints == [pytest.approx((70.0, 0.0))]
    assert all(d.length > 1.0 for d in _drops(world))


# user decision Q41, audit R4A-05
def test_a_contact_off_its_wire_or_on_a_rod_is_rejected() -> None:
    """Snapshot: a plan whose contact names an unknown wire raised a
    KeyError, one on the site's rod built a drop attached to the rod that
    ended 5 m from it, one 50 m off the wire a drop 50 m short."""
    from groundfield.generators.plan import NetworkPlan

    types = [
        BuildingTypeSpec(
            name="residential",
            grounding=GroundingSystemSpec(
                electrodes=[
                    FoundationElectrodeSpec(style="ring", size_m=10.0),
                    RodElectrodeSpec(length_m=2.0),
                ]
            ),
        )
    ]
    plan = TnNetworkGenerator(
        TnNetworkConfig(building_types=types, building_counts={"residential": 2}),
        seed=0,
    ).plan()
    data = plan.model_dump()
    i = next(k for k, b in enumerate(data["branches"]) if b["end_contact"] is not None)
    branch = data["branches"][i]
    x, y = branch["end_contact"]["position_xy"]
    wire = branch["end_contact"]["electrode"]
    for contact, match in (
        ({"electrode": branch["end"] + "_rod_1", "position_xy": (x, y)}, "not a foundation wire"),
        ({"electrode": branch["end"] + "_nothing", "position_xy": (x, y)}, "no electrode"),
        ({"electrode": wire, "position_xy": (x + 50.0, y)}, "from the wire"),
        ({"electrode": wire, "position_xy": (x + 0.011, y + 0.011)}, "from the wire"),
    ):  # fmt: skip
        data["branches"][i] = {**branch, "end_contact": contact}
        with pytest.raises(ValueError, match=match):
            NetworkPlan.model_validate(data).to_world()
    # Within one minimum section of the wire (a joined cable's point): built.
    data["branches"][i] = {
        **branch,
        "end_contact": {"electrode": wire, "position_xy": (x + 0.006, y + 0.006)},
    }
    NetworkPlan.model_validate(data).to_world()


# audit R4A-06
def test_the_closest_pair_prefilter_changes_no_bit(monkeypatch) -> None:
    """The bounding-box prefilter of the closest-pair search (planning time)
    against the full search over every segment: identical index, parameter
    and contact, bit for bit, for 400 random paths of 9 to 60 sections
    (ties included: axis-aligned paths along the sides)."""
    from groundfield.generators import _contact
    from groundfield.generators._contact import FoundationWire, closest_pair

    rng = np.random.default_rng(20261114)
    cases = []
    for k in range(400):
        n = int(rng.integers(9, 61))
        if k % 4 == 0:  # an axis-aligned path through and along a rectangle
            xs = np.sort(rng.choice(np.arange(-80.0, 81.0, 2.5), n + 1, replace=False))
            pts = np.stack([xs, np.full(n + 1, rng.choice([-5.0, 0.0, 6.0]))], axis=1)
        else:
            pts = np.cumsum(rng.uniform(-8, 8, (n + 1, 2)), axis=0)
        segments = [
            (tuple(a), tuple(b)) for a, b in zip(pts[:-1], pts[1:], strict=True)
        ]
        cx, cy = rng.uniform(-10, 10, 2)
        dx, dy = rng.uniform(1, 12, 2)
        corners = [(cx - dx / 2, cy - dy / 2), (cx + dx / 2, cy - dy / 2),
                   (cx + dx / 2, cy + dy / 2), (cx - dx / 2, cy + dy / 2)]  # fmt: skip
        wires = [
            FoundationWire(f"w{j}", j, (corners[j], corners[(j + 1) % 4]), 0.8)
            for j in range(4)
        ]
        cases.append((segments, wires))
    fast = [closest_pair(s, w) for s, w in cases]
    monkeypatch.setattr(
        _contact, "_candidate_segments", lambda seg, wires: np.arange(seg.shape[0])
    )
    full = [closest_pair(s, w) for s, w in cases]
    assert fast == full


# ---------------------------------------------------------------------
# Texts (audits R5B-01, R5B-06, R5B-13)
# ---------------------------------------------------------------------


def _flat_text(text: str) -> str:
    return " ".join(text.split())


def _doc(*parts: str) -> str:
    root = pathlib.Path(__file__).resolve().parents[1]
    return _flat_text(root.joinpath(*parts).read_text(encoding="utf-8"))


# user decision Q41, audits R5B-01, R5B-06
def test_the_texts_state_the_drops_over_other_foundations_and_the_1_cm_bound() -> None:
    """The texts said a drop may pass over another building's foundation for
    "the house behind a house on the main, 5 m", and "no drop runs along a
    cable" without its 1 cm bound. In the default grids 0 ... 80 % of the
    drops run 0.2 m above other houses' side wires (star, 30 mesh houses: 14
    of 25 drops, 150 m; over_foundations/extent.py of stream f6gen), and a
    foundation that touches the end node of its cable from beyond is tapped
    1 cm before the node, its drop sharing that 1 cm with the cable."""
    from groundfield.generators import _pen_routing

    for text in (
        _flat_text(_pen_routing.__doc__ or ""),
        _doc("docs", "reference", "api", "generators.md"),
        _doc("docs", "adr", "0009-world-generators.md"),
    ):
        assert "0.2 m above their wires" in text or "0.2 m above them" in text
        assert "14 of 25" in text
        assert "farther than one minimum section" in text
    assert "0 … 80 %" in _doc("docs", "reference", "api", "generators.md")


# user decision Q41, audit R5B-13
def test_the_materialiser_text_states_the_weaker_check_without_foundation_wires() -> (
    None
):
    """``materialise_branch`` said that without ``foundation_wires`` "any
    strip or polyline electrode is accepted" while its Raises section named
    the site's foundation wires only."""
    from groundfield.generators.materialise import materialise_branch

    doc = _flat_text(materialise_branch.__doc__ or "")
    assert "Without it any strip or polyline electrode is accepted" not in doc
    assert "need not belong to that site" in doc
    assert "without it, an electrode that is no strip or polyline" in doc
