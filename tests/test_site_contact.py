"""Generated conductors end at the nearest point of every horizontal electrode (Q43).

User decision Q43 of 2026-11-14: the nearest-point rule of user decision Q41
(every conductor the generators route to a foundation ends at the point of
that foundation nearest to it) also holds for sites without a foundation
whose earthing is a horizontal electrode: a ``RingElectrodeSpec`` circle —
among them the default substations of both generators (a 4 m ring with four
rods) — or a ``StripElectrodeSpec`` strip. Rods (vertical) keep their head.
A site with a horizontal electrode and rods is met at the nearest point of
the horizontal ones; a site with several horizontal electrodes (a ring, a
mesh foundation and strips) at the nearest point of all of them.

Before (base ``e6124cd``) such sites were met at the connection point of
their anchor electrode: the +x point of a ring, the start of a strip. Every
cable of the default star network left the substation at (4, 0) m, the
western mains crossing the ring.

The checks below use a second, independent method: dense sampling of the
circle in plan.
"""

from __future__ import annotations

import math
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
    RingElectrodeSpec,
    RodElectrodeSpec,
    StripElectrodeSpec,
    SubstationConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
    rod_circle,
)

pytestmark = pytest.mark.finding("10.38")

_RING_R = 4.0
"""Radius in m of the default substation ring of both generators."""


def _circle_samples(cx: float, cy: float, r: float, n: int = 200_000) -> np.ndarray:
    """``n`` points of the exact circle (spacing 2 pi r / n)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])


def _segment_distance(points: np.ndarray, a, b) -> np.ndarray:
    """Exact plan distance of ``points`` to the segment ``a -> b``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    dd = float(d @ d)
    t = np.zeros(len(points)) if dd == 0.0 else np.clip((points - a) @ d / dd, 0, 1)
    return np.hypot(*(points - (a + t[:, None] * d)).T)


def _build(**kw) -> gf.World:
    params: dict[str, object] = {"soil": HomogeneousSoilSpec(rho=100.0)}
    params.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TnNetworkGenerator(TnNetworkConfig(**params), seed=1).build()


def _at(world: gf.World, electrode: str) -> list[tuple[gf.Conductor, str]]:
    """Generated PEN conductors with an end on ``electrode`` (and which end)."""
    out = []
    for c in world.conductors:
        if not c.name.startswith("pen_"):
            continue
        if c.start_electrode == electrode:
            out.append((c, "start"))
        if c.end_electrode == electrode:
            out.append((c, "end"))
    return out


def _star_world(**substation) -> gf.World:
    """Default substation (or ``substation``), two rod cabinets, two houses."""
    kw: dict[str, object] = dict(
        building_counts={"residential": 2},
        placement=ExplicitPlacement(positions=[(-50.0, 20.0), (15.0, -20.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(-100.0, 0.0), (60.0, 80.0)]),
            fixed_count=2,
        ),
    )
    if substation:
        kw["substation"] = SubstationConfig(**substation)
    return _build(**kw)


# ---------------------------------------------------------------------
# TnNetworkGenerator
# ---------------------------------------------------------------------


# user decision Q43
def test_cables_leave_the_default_substation_ring_at_its_nearest_point() -> None:
    """The default substation (4 m ring at 0.6 m, four rods): each main
    starts at the point of the circle nearest to its cabinet's rod head,
    exactly on the circle, attached to the ring. Base: both started at the
    ring's connection point (4, 0) m, the western one crossing the ring."""
    world = _star_world()
    mains = {c.name: c for c in world.conductors if c.name.startswith("pen_main_")}
    west, north_east = mains["pen_main_kvs_0_rod_0"], mains["pen_main_kvs_1_rod_0"]
    assert west.start_electrode == north_east.start_electrode == "trafo_ring_0"
    assert west.start[:2] == (-4.0, 0.0)
    assert north_east.start[:2] == pytest.approx((2.4, 3.2), abs=1e-12)
    samples = _circle_samples(0.0, 0.0, _RING_R)
    for cable, cabinet in ((west, (-100.0, 0.0)), (north_east, (60.0, 80.0))):
        assert math.hypot(*cable.start[:2]) == pytest.approx(_RING_R, abs=1e-12)
        sampled = samples[np.argmin(np.hypot(*(samples - cabinet).T))]
        # Second method: the nearest of 200 000 points of the circle.
        assert math.dist(cable.start[:2], sampled) < 2e-4


# user decision Q43
def test_a_drop_from_the_substation_ring_is_the_closest_pair_with_the_house() -> None:
    """The house south-east of the substation is fed by the western cabinet;
    its closest point on that main is the main's start, so its drop leaves
    the substation: at the point of the circle nearest to the house's
    foundation (its corner (10, -15) m), not at (4, 0) m."""
    world = _star_world()
    house = "residential_1"
    (drop,) = [
        c
        for c in world.conductors
        if c.name.startswith("pen_service_") and c.end_electrode.startswith(house)
    ]
    assert drop.start_electrode == "trafo_ring_0"
    corner = np.array([10.0, -15.0])
    expected = _RING_R * corner / np.linalg.norm(corner)
    assert drop.start[:2] == pytest.approx(tuple(expected), abs=1e-12)
    # Second method: the closest pair of 200 000 circle points and the
    # foundation wires (exact distances to the wire segments).
    samples = _circle_samples(0.0, 0.0, _RING_R)
    wires = [e for e in world.electrodes if e.name.startswith(f"{house}_foundation_0")]
    best = math.inf
    for e in wires:
        (run,) = e.wire_polylines()
        pts = np.asarray(run)[:, :2]
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            best = min(best, float(_segment_distance(samples, a, b).min()))
    plan_length = math.dist(drop.start[:2], drop.end[:2])
    assert plan_length == pytest.approx(best, abs=1e-6)


# user decision Q43
def test_a_strip_substation_is_met_at_its_nearest_point() -> None:
    """A substation earthed by one 20 m strip along x: the main to the
    cabinet at (5, 40) m starts at (5, 0) m on the strip. Base: at the
    strip's start (-10, 0) m."""
    world = _build(
        substation=SubstationConfig(
            grounding=GroundingSystemSpec(
                electrodes=[StripElectrodeSpec(length_m=20.0, depth_m=0.6)]
            )
        ),
        building_counts={"residential": 1},
        placement=ExplicitPlacement(positions=[(-60.0, -60.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(5.0, 40.0)]), fixed_count=1
        ),
    )
    (main,) = [c for c in world.conductors if c.name.startswith("pen_main_")]
    assert main.start_electrode == "trafo_strip_0"
    assert main.start[:2] == (5.0, 0.0)


# user decision Q43
def test_rods_keep_their_head() -> None:
    """A substation of four rods only: the mains start at the head of the
    first rod, as before the decision (guard)."""
    world = _star_world(
        grounding=GroundingSystemSpec(
            electrodes=rod_circle(4, radius_m=2.0, length_m=2.5)
        )
    )
    rod = world.get_electrode("trafo_rod_0")
    ends = _at(world, "trafo_rod_0")
    assert len(ends) >= 2
    for cable, end in ends:
        assert end == "start"
        assert cable.start[:2] == pytest.approx(rod.connection_point[:2])


# user decision Q43
def test_a_site_with_a_ring_a_foundation_and_strips_is_met_at_the_nearest_of_them() -> (
    None
):
    """The ``large_industry`` earthing at the substation: a 25 m ring, a
    30 m mesh foundation inside it, eight rods on the ring and two 40 m
    strips at y = +-28 m. The main to the cabinet at (0, 100) m starts on the
    northern strip at (0, 28) m (72 m; the ring is 75 m away), the one to
    (100, 0) m on the ring at (25, 0) m. Base: both on the foundation, the
    connection point of none of these."""
    world = _build(
        substation=SubstationConfig(
            grounding=GroundingSystemSpec(
                electrodes=[
                    RingElectrodeSpec(radius_m=25.0, depth_m=0.8),
                    FoundationElectrodeSpec(size_m=30.0, depth_m=0.8, n_x=4, n_y=4),
                    *rod_circle(n=8, radius_m=25.0, length_m=3.0),
                    StripElectrodeSpec(
                        length_m=40.0, depth_m=0.8, offset_xy_m=(0.0, 28.0)
                    ),
                    StripElectrodeSpec(
                        length_m=40.0, depth_m=0.8, offset_xy_m=(0.0, -28.0)
                    ),
                ]
            )
        ),
        building_counts={"residential": 1},
        placement=ExplicitPlacement(positions=[(-150.0, -150.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(0.0, 100.0), (100.0, 0.0)]),
            fixed_count=2,
        ),
    )
    mains = {c.name: c for c in world.conductors if c.name.startswith("pen_main_")}
    north, east = mains["pen_main_kvs_0_rod_0"], mains["pen_main_kvs_1_rod_0"]
    strip = world.get_electrode(north.start_electrode)
    assert strip.kind == "strip" and strip.start[1] == 28.0
    assert north.start[:2] == pytest.approx((0.0, 28.0), abs=1e-12)
    assert east.start_electrode == "trafo_ring_0"
    assert east.start[:2] == (25.0, 0.0)


# user decision Q43
def test_a_ring_house_is_joined_where_a_main_crosses_it() -> None:
    """A house earthed by a 6 m ring centred 1 m north of the main is joined
    to the ring where the main crosses the circle, without a drop (rule 4 of
    the router, now for rings): of the two crossings at x = -30 -+ sqrt(8) m
    the tie rule (smallest arc position from the ring's +x point) takes the
    western one, at 199.5 degrees. Base: a 1 m drop from the main to the
    ring's connection point (-27, 1) m."""
    ring_house = BuildingTypeSpec(
        name="residential",
        grounding=GroundingSystemSpec(
            electrodes=[RingElectrodeSpec(radius_m=3.0, depth_m=0.8)]
        ),
    )
    world = _build(
        building_types=[ring_house],
        building_counts={"residential": 1},
        placement=ExplicitPlacement(positions=[(-30.0, 1.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(-100.0, 0.0)]), fixed_count=1
        ),
    )
    assert not [c for c in world.conductors if c.name.startswith("pen_service_")]
    sections = _at(world, "residential_0_ring_0")
    assert len(sections) == 2
    x = -30.0 - math.sqrt(8.0)
    for cable, end in sections:
        point = cable.end[:2] if end == "end" else cable.start[:2]
        assert point == pytest.approx((x, 0.0), abs=1e-9)


# ---------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------


def _layout_world() -> gf.World:
    """Default layout station (4 m ring, four rods) at the origin, a cable
    to the west, one house north of it and one north-east of the station."""

    def box(xc, yc, w, h):
        return [
            (xc - w / 2, yc - h / 2),
            (xc + w / 2, yc - h / 2),
            (xc + w / 2, yc + h / 2),
            (xc - w / 2, yc + h / 2),
        ]

    fps = [
        gf.BuildingFootprint(polygon_xy_m=poly, building_use="residential", osm_id=k)
        for k, poly in enumerate([box(-40.0, 10.0, 8.0, 6.0), box(9.0, 9.0, 6.0, 6.0)])
    ]
    layout = LvNetworkLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0), obstacle_clearance_m=0.0
    )
    layout.pen_cables.append(
        gf.generators.layout.routing.PenCable(
            name="pen_w",
            start_anchor="substation",
            waypoints=[(0.0, 0.0), (-60.0, 0.0)],
        )
    )
    layout.connect_buildings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return layout.to_world()


# user decision Q43
def test_layout_cables_and_drops_leave_the_station_ring_at_its_nearest_point() -> None:
    """The cable to the west leaves the station ring at (-4, 0) m (base: at
    (4, 0) m, through the ring); the house north-east of the station is fed
    from the station at the point of the circle nearest to its outline (its
    corner (6, 6) m, direction 45 degrees)."""
    world = _layout_world()
    first = next(c for c in world.conductors if c.name == "pen_w_seg_0")
    assert first.start_electrode == "substation_ring_0"
    assert first.start[:2] == (-4.0, 0.0)
    (drop,) = [c for c in world.conductors if c.name == "service_001"]
    assert drop.start_electrode == "substation_ring_0"
    r = _RING_R / math.sqrt(2.0)
    assert drop.start[:2] == pytest.approx((r, r), abs=1e-12)
    assert drop.end[:2] == pytest.approx((6.0, 6.0), abs=1e-12)


# ---------------------------------------------------------------------
# Unit level: the exact circle, two methods
# ---------------------------------------------------------------------


# user decision Q43
def test_the_circle_closest_pair_agrees_with_dense_sampling() -> None:
    """300 random segments against a circle: the exact closest pair (crossing,
    inside, outside) against the nearest of 200 000 circle points to the
    segment; the pair's point is on the circle and on the segment's
    nearest point."""
    from groundfield.generators._contact import (
        FoundationWire,
        circle_polygon,
        closest_pair,
        nearest_point,
    )

    rng = np.random.default_rng(43)
    circle = (1.5, -2.0, 4.0)
    wire = FoundationWire("ring", 0, circle_polygon(*circle), 0.6, circle)
    samples = _circle_samples(*circle)
    for _ in range(300):
        a, b = rng.uniform(-12.0, 12.0, (2, 2))
        _, t, contact = closest_pair([(tuple(a), tuple(b))], [wire])
        exact = float(_segment_distance(samples, a, b).min())
        assert contact.distance <= exact + 1e-12
        # The sampling spacing is 2 pi r / 200 000 = 0.126 mm.
        assert contact.distance == pytest.approx(exact, abs=7e-5)
        cx, cy, r = circle
        assert math.hypot(contact.xy[0] - cx, contact.xy[1] - cy) == pytest.approx(r)
        on_segment = a + t * (b - a)
        assert math.dist(on_segment, contact.xy) == pytest.approx(
            contact.distance, abs=1e-9
        )
        p = rng.uniform(-12.0, 12.0, 2)
        near = nearest_point(tuple(p), [wire])
        sampled = samples[np.argmin(np.hypot(*(samples - p).T))]
        assert math.dist(near.xy, sampled) < 2e-4
        assert near.distance == pytest.approx(
            abs(math.hypot(p[0] - cx, p[1] - cy) - r), abs=1e-12
        )


# user decision Q43
def test_a_contact_on_a_ring_is_checked_against_the_exact_circle() -> None:
    """A plan contact on the substation ring between two vertices of the
    search polygon (on the exact circle) builds; 1.1 cm off the circle it
    is refused, like a contact off a foundation wire."""
    from groundfield.generators.plan import NetworkPlan

    plan = TnNetworkGenerator(
        TnNetworkConfig(building_counts={"residential": 2}), seed=0
    ).plan()
    data = plan.model_dump()
    i = next(
        k
        for k, b in enumerate(data["branches"])
        if b["start_contact"] is not None
        and b["start_contact"]["electrode"] == "trafo_ring_0"
    )
    branch = data["branches"][i]
    angle = 2.0 * math.pi * 0.5 / 144  # half-way between two polygon vertices
    on = (4.0 * math.cos(angle), 4.0 * math.sin(angle))
    off = (4.011 * math.cos(angle), 4.011 * math.sin(angle))
    data["branches"][i] = {
        **branch,
        "start_contact": {"electrode": "trafo_ring_0", "position_xy": on},
    }
    NetworkPlan.model_validate(data).to_world()
    data["branches"][i] = {
        **branch,
        "start_contact": {"electrode": "trafo_ring_0", "position_xy": off},
    }
    with pytest.raises(ValueError, match="from the wire"):
        NetworkPlan.model_validate(data).to_world()
    data["branches"][i] = {
        **branch,
        "start_contact": {"electrode": "trafo_rod_1", "position_xy": on},
    }
    with pytest.raises(ValueError, match="not a foundation, ring or strip wire"):
        NetworkPlan.model_validate(data).to_world()


# user decision Q43
def test_site_contact_wires_lists_the_horizontal_electrodes_in_world_order() -> None:
    """``site_contact_wires``: the ring (with its circle), the strips and the
    foundation wires of a site, in world order; nothing for rods."""
    from groundfield.generators.materialise import site_contact_wires

    spec = GroundingSystemSpec(
        electrodes=[
            RodElectrodeSpec(length_m=2.0),
            RingElectrodeSpec(radius_m=4.0, depth_m=0.6),
            FoundationElectrodeSpec(style="ring", size_m=6.0, depth_m=0.8),
            StripElectrodeSpec(length_m=10.0, depth_m=0.5, offset_xy_m=(0.0, 9.0)),
        ]
    )
    site = spec.plan_site("s", role="substation", position_xy=(10.0, 0.0), rng=0)
    wires = site_contact_wires(site)
    assert [w.electrode for w in wires] == [
        "s_ring_1",
        "s_foundation_2",
        "s_foundation_2_w1",
        "s_foundation_2_w2",
        "s_foundation_2_w3",
        "s_strip_3",
    ]
    assert [w.index for w in wires] == list(range(6))
    assert wires[0].circle == (10.0, 0.0, 4.0) and wires[0].z == 0.6
    assert all(w.circle is None for w in wires[1:])
    assert wires[-1].xy == ((5.0, 9.0), (15.0, 9.0))
    rods = GroundingSystemSpec(electrodes=rod_circle(3, radius_m=1.5, length_m=3.0))
    assert (
        site_contact_wires(
            rods.plan_site("r", role="substation", position_xy=(0.0, 0.0), rng=0)
        )
        == []
    )


# audit W1B-07
def test_the_router_rules_name_every_horizontal_electrode() -> None:
    """Rules 2 and 4 of the router apply to every horizontal earth electrode
    of a building (foundation, ring, strip; rule 0, user decision Q43); up to
    then they named the foundation only."""
    from groundfield.generators import _pen_routing

    text = " ".join((_pen_routing.__doc__ or "").split())
    rule2 = text.split("2. **")[1].split("3. **")[0]
    rule4 = text.split("4. **")[1].split("What follows")[0]
    assert "horizontal electrode wires (foundation, ring, strip" in rule2
    assert "the wires of the building's foundation" not in rule2
    assert "horizontal electrodes (foundation, ring, strip)" in rule4
    assert "without a foundation" not in rule2 + rule4
