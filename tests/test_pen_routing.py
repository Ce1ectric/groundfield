"""PEN cables of generated TN networks (``TnNetworkGenerator``).

Stream DOC-B found a 2-house network whose single cable cabinet sat 180 m
beyond the houses (first slot of the default cabinet row): both service
drops ran from the cabinet back along the PEN main, coaxial with it at the
same depth; with ``inductance_model="neumann"`` the network was
non-passive (Im Z_in = -0.633 ohm at 1 kHz). The first fix built a
nearest-point tree; the third adversarial audit found that it crashed on
1-3 % of random networks (A3SG-01), chained collinear cabinets on one
cable against the documented star topology (A3SG-02) and hung service
drops off other houses' drops (A3SG-03).

What the router guarantees since 2026.11 (``generators._pen_routing``):
every cabinet has its own cable (star: from the substation; radial trunk:
one trunk cable per segment); a service drop taps the cable path that
feeds its own building at the nearest point (a tap junction) or starts at
its end node, never on another drop; no section is shorter than the
router's minimum length (taps snap to nearby nodes); a drop never runs
along the cable it taps. Cables of different cabinets and drops may share
a route (one trench); the network stays passive.
"""

from __future__ import annotations

import itertools
import math
import re
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
    ManhattanGridPlacement,
    PenConfig,
    RadialTrunkTopology,
    RodElectrodeSpec,
    SubstationConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
)
from groundfield.generators._pen_routing import MIN_DROP_LENGTH_M, min_section_length
from groundfield.generators.plan import ConductorParameters, NetworkPlan

_TWO_HOUSES = ManhattanGridPlacement(spacing_x_m=25.0, spacing_y_m=30.0, n_per_row=2)
_RING_HOUSE = BuildingTypeSpec(
    name="residential",
    grounding=GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", size_m=10.0, depth_m=0.8)]
    ),
)
_ROD_SUBSTATION = SubstationConfig(
    grounding=GroundingSystemSpec(electrodes=[RodElectrodeSpec(length_m=2.5)])
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _pen(world: gf.World) -> list[gf.Conductor]:
    return [c for c in world.conductors if c.conductor_type == "pen"]


def _cable(name: str) -> str:
    """Cable a section belongs to (``<cable>_s<k>`` -> ``<cable>``)."""
    return re.sub(r"_s[0-9]+$", "", name)


def _is_main(c: gf.Conductor) -> bool:
    return c.name.startswith(("pen_main_", "pen_trunk_"))


def _drops(world: gf.World) -> list[gf.Conductor]:
    return [c for c in _pen(world) if c.name.startswith("pen_service_")]


def _overlap(c1: gf.Conductor, c2: gf.Conductor, tol: float = 1e-6) -> float:
    """Length in m over which two straight conductors run along each other."""
    ax, ay, bx, by = c1.start[0], c1.start[1], c1.end[0], c1.end[1]
    length = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / length, (by - ay) / length
    ends = [(c2.start[0] - ax, c2.start[1] - ay), (c2.end[0] - ax, c2.end[1] - ay)]
    if any(abs(-uy * px + ux * py) > tol for px, py in ends):
        return 0.0
    lo, hi = sorted(px * ux + py * uy for px, py in ends)
    return max(0.0, min(length, hi) - max(0.0, lo))


def _build(**kw: object) -> gf.World:
    params: dict[str, object] = {"soil": HomogeneousSoilSpec(rho=100.0)}
    params.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TnNetworkGenerator(TnNetworkConfig(**params), seed=1).build()


def _check_routing(world: gf.World, substation_anchor: str) -> None:
    """Assert the documented guarantees on a built network."""
    pen = _pen(world)
    mains = [c for c in pen if _is_main(c)]
    drops = _drops(world)
    assert len(mains) + len(drops) == len(pen)
    # No section or drop shorter than the router's minimum length.
    assert min(math.dist(c.start, c.end) for c in pen) >= MIN_DROP_LENGTH_M
    # Every drop starts on the mains (a node of a cabinet cable or trunk,
    # or the substation), never on another drop.
    nodes = {substation_anchor}
    for c in mains:
        nodes.update((c.start_electrode, c.end_electrode))
    assert all(d.start_electrode in nodes for d in drops)
    # A drop never runs along the cable it taps (the cable whose section
    # ends at the drop's start: its tap junction, cabinet or a building the
    # cable runs through).
    sections: dict[str, list[gf.Conductor]] = {}
    tapped: dict[str, set[str]] = {}
    for c in mains:
        sections.setdefault(_cable(c.name), []).append(c)
        tapped.setdefault(c.end_electrode, set()).add(_cable(c.name))
    for d in drops:
        for cable in tapped.get(d.start_electrode, ()):
            for c in sections[cable]:
                assert _overlap(d, c) < 0.01, (d.name, c.name)


# ---------------------------------------------------------------------
# DOC-B: drops ran back along the main (10.38)
# ---------------------------------------------------------------------

_CASES = {
    "two-houses": dict(building_counts={"residential": 2}, placement=_TWO_HOUSES),
    "two-ring-houses": dict(
        building_counts={"residential": 2},
        placement=_TWO_HOUSES,
        building_types=[_RING_HOUSE],
    ),
    "default-5": dict(building_counts={"residential": 5}),
    "default-30": dict(),
    "houses-in-a-column": dict(
        building_counts={"residential": 4},
        placement=ExplicitPlacement(
            positions=[(-60.0, 20.0), (-60.0, 45.0), (-60.0, 70.0), (-60.0, -25.0)]
        ),
        building_types=[_RING_HOUSE],
    ),
    "radial": dict(
        building_counts={"residential": 12},
        pen_topology=RadialTrunkTopology(
            n_feeders=1, feeder_directions_deg=[180.0], slots_per_substation=2
        ),
    ),
    "radial-4-feeders": dict(
        building_counts={"residential": 16},
        pen_topology=RadialTrunkTopology(slots_per_substation=2, slots_per_cabinet=2),
    ),
    # audit A3SG-03: the houses of the middle row sit on the east / west
    # trunks; fed from the substation slots, their drops run along the trunk
    "radial-default-grid-ring-houses": dict(
        building_types=[_RING_HOUSE],
        building_counts={"residential": 30},
        pen_topology=RadialTrunkTopology(),
    ),
    # audit A3SG-03: two houses above one point, fed by different cabinets
    "star-two-feeders-share-a-tap": dict(
        building_types=[_RING_HOUSE],
        building_counts={"residential": 2},
        placement=ExplicitPlacement(positions=[(10.0, 10.0), (10.0, 45.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(60.0, 0.0), (60.0, 80.0)]),
            fixed_count=2,
        ),
    ),
    # audit A3SG-03: a house beside another cabinet's cable
    "star-drop-along-another-cabinet-cable": dict(
        building_types=[_RING_HOUSE],
        building_counts={"residential": 1},
        placement=ExplicitPlacement(positions=[(30.0, 10.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(60.0, 0.0), (30.0, 60.0)]),
            fixed_count=2,
        ),
    ),
}


# stream DOC-B, audit A3SG-03
@pytest.mark.finding("10.38")
@pytest.mark.parametrize("case", sorted(_CASES))
def test_routing_guarantees(case: str) -> None:
    """Before DOC-B: the drops of the 2-house networks ran back along the
    main over 167.5 m. Before A3SG-03: in the radial default grid 22 of 33
    drops started on another house's drop."""
    _check_routing(_build(**_CASES[case]), "trafo_ring_0")


# stream DOC-B
@pytest.mark.finding("10.38")
def test_a_drop_starts_at_the_nearest_point_of_its_feeding_cable() -> None:
    world = _build(building_counts={"residential": 2}, placement=_TWO_HOUSES)
    drops = {c.name: c for c in _drops(world)}
    near = drops["pen_service_residential_0_foundation_0"]
    # A tap junction on the main, straight beside the house's connection point.
    tap = world.get_junction(near.start_electrode)
    assert tap.position[:2] == pytest.approx((-17.5, 0.0))
    assert math.dist(near.start, near.end) == pytest.approx(5.0)
    # The house beside the substation is fed from the substation (the end of
    # its cabinet's cable), not from the cabinet 180 m away.
    far = drops["pen_service_residential_1_foundation_0"]
    assert far.start_electrode == "trafo_ring_0"
    assert math.dist(far.start, far.end) < 10.0
    total = sum(math.dist(c.start, c.end) for c in _pen(world))
    assert total == pytest.approx(184.0 + 5.0 + math.hypot(3.5, 5.0))


# stream DOC-B
@pytest.mark.finding("10.38")
def test_a_building_on_the_main_splits_it() -> None:
    """A ring foundation's anchor (the grid-mesh centre) lies on the main:
    the main runs through it, the building gets no drop."""
    world = _build(
        building_counts={"residential": 2},
        placement=_TWO_HOUSES,
        building_types=[_RING_HOUSE],
    )
    names = {c.name: (c.start_electrode, c.end_electrode) for c in _pen(world)}
    assert "pen_service_residential_0_foundation_0" not in names
    assert names["pen_main_kvs_0_rod_0"] == (
        "trafo_ring_0",
        "residential_0_foundation_0",
    )
    assert names["pen_main_kvs_0_rod_0_s1"] == (
        "residential_0_foundation_0",
        "kvs_0_rod_0",
    )


# stream DOC-B
@pytest.mark.finding("10.38")
@pytest.mark.parametrize("building_types", [None, [_RING_HOUSE]], ids=["mesh", "ring"])
def test_inductive_pen_network_is_passive(building_types: list | None) -> None:
    """Base: Im Z_in = -0.633 ohm at 1 kHz (perfect mirror) with the drops
    on the main."""
    kw: dict[str, object] = dict(
        building_counts={"residential": 2},
        placement=_TWO_HOUSES,
        pen=PenConfig(inductance_model="neumann"),
    )
    if building_types is not None:
        kw["building_types"] = building_types
    world = _build(**kw)
    engine = gf.create_engine(
        segment_length=1.0,
        frequencies=[50.0, 1000.0],
        earth_inductive_model="perfect_mirror",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        z = engine.solve(world).input_impedance()
    assert z[0].imag > 0.0 and z[1].imag > 0.0


# stream DOC-B
@pytest.mark.finding("10.38")
def test_tap_junctions_are_plan_junctions_and_round_trip() -> None:
    gen = TnNetworkGenerator(
        TnNetworkConfig(building_counts={"residential": 5}), seed=0
    )
    plan = gen.plan()
    assert [j.name for j in plan.junctions] == [f"pen_tap_{k}" for k in range(5)]
    again = type(plan).model_validate_json(plan.model_dump_json())
    world, world_again = plan.to_world(), again.to_world()
    assert world.model_dump_json() == world_again.model_dump_json()
    assert [j.name for j in world.junctions] == [j.name for j in plan.junctions]


# stream DOC-B
@pytest.mark.finding("10.38")
def test_a_building_on_top_of_a_node_is_rejected_clearly() -> None:
    """A house whose connection point coincides with the cabinet's would get
    a cable of no length (a Conductor validation error deep in the build)."""
    cfg = TnNetworkConfig(
        building_types=[_RING_HOUSE],
        building_counts={"residential": 2},
        placement=ExplicitPlacement(positions=[(-60.0, 0.0), (-180.0, 0.0)]),
    )
    with pytest.raises(ValueError, match="coincides"):
        TnNetworkGenerator(cfg, seed=0).build()


# ---------------------------------------------------------------------
# A3SG-02: every cabinet has its own cable
# ---------------------------------------------------------------------


# audit A3SG-02
@pytest.mark.finding("10.38")
def test_star_topology_wires_every_cabinet_directly_to_the_substation() -> None:
    """StarCabinetTopology: 'The substation feeds each cable cabinet
    directly'. Base: the default 30-house network chained kvs_0 off kvs_1."""
    world = _build()
    mains = [c for c in _pen(world) if c.name.startswith("pen_main_")]
    first = {c.name: c for c in mains if _cable(c.name) == c.name}
    assert sorted(first) == ["pen_main_kvs_0_rod_0", "pen_main_kvs_1_rod_0"]
    assert all(c.start_electrode == "trafo_ring_0" for c in first.values())
    # Each cabinet's cable ends at its cabinet.
    for name in first:
        last = [c for c in mains if _cable(c.name) == name][-1]
        assert last.end_electrode == name.removeprefix("pen_main_")


# audit A3SG-02
@pytest.mark.finding("10.38")
def test_collinear_cabinets_have_separate_cables_in_one_trench() -> None:
    """The two cabinets of the default 30-house network lie west of the
    substation on one line: two cables along one route (144 m in common),
    evaluated as touching wires, so the inductive network stays passive."""
    world = _build(pen=PenConfig(inductance_model="neumann"))
    mains = {c.name: c for c in _pen(world) if c.name.startswith("pen_main_")}
    assert _overlap(
        mains["pen_main_kvs_0_rod_0"], mains["pen_main_kvs_1_rod_0"]
    ) == pytest.approx(21.5)
    total = sum(
        _overlap(mains["pen_main_kvs_0_rod_0"], c)
        for name, c in mains.items()
        if _cable(name) == "pen_main_kvs_1_rod_0"
    )
    assert total == pytest.approx(144.0)
    engine = gf.create_engine(
        segment_length=2.0,
        frequencies=[50.0, 1000.0],
        earth_inductive_model="perfect_mirror",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        z = np.asarray(engine.solve(world).input_impedance())
    assert np.all(z.real > 0.0) and np.all(z.imag > 0.0)


# audit A3SG-02
@pytest.mark.finding("10.38")
@pytest.mark.slow
def test_z_in_of_a_100_house_network_drops_against_the_old_routing() -> None:
    """100 ring houses, default catalogue grid and cabinets, image backend,
    ds = 2 m, 50 Hz. Z_in of the world built by the 0.15.0 routing (drops
    from the cabinets, measured with this solver): 0.202724 ohm; the
    nearest-point tree with chained cabinets (base): 0.206457 ohm (+1.8 %,
    against the documented direction); now 0.200434 ohm."""
    world = _build(building_types=[_RING_HOUSE], building_counts={"residential": 100})
    engine = gf.create_engine(backend="image", segment_length=2.0, frequencies=[50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        z = complex(np.asarray(engine.solve(world).input_impedance())[0]).real
    assert z == pytest.approx(0.200434, rel=1e-5)
    assert z < 0.202724


# ---------------------------------------------------------------------
# A3SG-03: a drop taps the cable of its own cabinet
# ---------------------------------------------------------------------


# audit A3SG-03
@pytest.mark.finding("10.38")
def test_a_drop_taps_the_cable_of_its_own_cabinet() -> None:
    """The house at (30, 10) is fed by the cabinet at (60, 0); base: its drop
    started at the tap from which the cable of the other cabinet (30, 60)
    left the chained main and ran 10 m along that cable."""
    world = _build(**_CASES["star-drop-along-another-cabinet-cable"])
    (drop,) = _drops(world)
    tap = world.get_junction(drop.start_electrode)
    assert tap.position[:2] == pytest.approx((30.0, 0.0))
    own = [c for c in _pen(world) if _cable(c.name) == "pen_main_kvs_0_rod_0"]
    assert drop.start_electrode in {c.end_electrode for c in own}
    other = [c for c in _pen(world) if _cable(c.name) == "pen_main_kvs_1_rod_0"]
    assert all(_overlap(drop, c) == 0.0 for c in other)


# audit A3SG-03
@pytest.mark.finding("10.38")
def test_no_drop_starts_on_another_drop_in_the_radial_default_grid() -> None:
    """Base: 22 of the drops of this network started on another house's drop."""
    world = _build(**_CASES["radial-default-grid-ring-houses"])
    drops = _drops(world)
    assert len(drops) == 30
    houses = {c.end_electrode for c in drops}
    assert not any(c.start_electrode in houses for c in drops)
    assert not any(c.start_electrode.startswith("residential_") for c in drops)


# audit A3SG-03
@pytest.mark.finding("10.38")
def test_overlapping_drops_and_trunks_stay_passive() -> None:
    """Drops of the substation slots run along the trunks (one trench)."""
    world = _build(
        **_CASES["radial-default-grid-ring-houses"],
        pen=PenConfig(inductance_model="neumann"),
    )
    trunks = [c for c in _pen(world) if c.name.startswith("pen_trunk_")]
    shared = sum(_overlap(d, t) for d in _drops(world) for t in trunks)
    assert shared > 0.0
    engine = gf.create_engine(
        segment_length=2.0,
        frequencies=[50.0, 1000.0],
        earth_inductive_model="perfect_mirror",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        z = np.asarray(engine.solve(world).input_impedance())
    assert np.all(z.real > 0.0) and np.all(z.imag > 0.0)


# ---------------------------------------------------------------------
# A3SG-01: no cable section shorter than the minimum length
# ---------------------------------------------------------------------


# audit A3SG-01
@pytest.mark.finding("10.38")
@pytest.mark.parametrize("offset_m", [1e-7, 1e-4, 1e-3, 3e-3, 4.5e-3])
def test_a_building_millimetres_beside_a_node_still_builds(offset_m: float) -> None:
    """Substation rod at the origin, one cabinet 100 m east, a ring house
    20 m north of the substation but ``offset_m`` east of it. Base: its tap
    left a main section of ``offset_m`` that the Conductor model refuses
    (pydantic ValidationError, series resistance below 1e-6 ohm). The tap
    snaps to the substation."""
    world = _build(
        substation=_ROD_SUBSTATION,
        building_types=[_RING_HOUSE],
        building_counts={"residential": 1},
        placement=ExplicitPlacement(positions=[(offset_m, 20.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(100.0, 0.0)]), fixed_count=1
        ),
    )
    lengths = [math.dist(c.start, c.end) for c in _pen(world)]
    assert min(lengths) >= MIN_DROP_LENGTH_M
    (drop,) = _drops(world)
    assert drop.start_electrode == "trafo_rod_0"
    assert world.junctions == []


# audit A3SG-01
@pytest.mark.finding("10.38")
def test_a_tap_millimetres_from_an_earlier_tap_snaps_to_it() -> None:
    """Two houses on either side of the main whose taps would lie 2 mm apart
    share one tap junction."""
    world = _build(
        substation=_ROD_SUBSTATION,
        building_types=[_RING_HOUSE],
        building_counts={"residential": 2},
        placement=ExplicitPlacement(positions=[(50.0, 20.0), (50.002, -30.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(100.0, 0.0)]), fixed_count=1
        ),
    )
    drops = _drops(world)
    assert len(drops) == 2 and len(world.junctions) == 1
    assert {d.start_electrode for d in drops} == {"pen_tap_0"}
    assert min(math.dist(c.start, c.end) for c in _pen(world)) > 10.0


# 60 houses uniformly in 400 m x 240 m (12 m apart), default catalogue, star
# topology: network 71 of the audit's Monte-Carlo run, positions rounded to
# 1 mm. Base: ValidationError for 'pen_main_kvs_2_rod_0_s3' (a tap 2 mm from
# the previous one).
_RANDOM_60 = [
    (-25.281, -21.318), (-185.569, -8.933), (171.17, 37.169), (24.075, 24.43),
    (-51.872, -92.925), (64.192, -80.589), (154.722, 88.469), (-166.267, -85.762),
    (104.167, 74.973), (184.928, 60.125), (-13.802, -42.358), (-142.761, -56.018),
    (121.629, 110.773), (32.599, -38.131), (-164.596, 65.276), (-165.297, -51.953),
    (-99.971, 5.242), (176.004, 78.287), (-107.061, 87.736), (-184.267, 97.618),
    (80.044, -51.824), (-120.761, -19.741), (-74.92, 102.483), (-1.014, -25.708),
    (91.3, -43.432), (2.173, 58.865), (-69.085, -46.9), (74.18, -0.292),
    (-68.233, 63.369), (-141.768, -71.946), (-61.187, 48.117), (21.14, -119.929),
    (56.038, -18.098), (139.939, 42.008), (-61.189, -15.987), (184.815, -3.724),
    (115.789, 7.414), (62.877, 34.957), (-42.005, -34.98), (87.949, -91.723),
    (146.322, -104.844), (15.219, 37.077), (-90.732, -43.333), (8.885, -76.118),
    (182.51, -41.902), (-108.168, -72.27), (-111.986, 107.175), (107.357, -44.446),
    (-172.158, -112.184), (-130.184, 103.743), (-190.898, -30.602), (-96.925, 23.089),
    (68.425, -65.004), (-122.107, 16.81), (50.41, -93.042), (78.12, -82.493),
    (42.408, -47.346), (-197.38, -49.677), (76.285, -112.681), (-34.758, -73.299),
]  # fmt: skip


# audit A3SG-01
@pytest.mark.finding("10.38")
def test_random_position_network_builds() -> None:
    world = _build(
        building_counts={"residential": len(_RANDOM_60)},
        placement=ExplicitPlacement(positions=_RANDOM_60),
    )
    assert len(_drops(world)) > 0
    _check_routing(world, "trafo_ring_0")


# audit A3SG-01
@pytest.mark.finding("10.38")
@pytest.mark.parametrize(
    ("conductor", "expected_m"),
    [
        (ConductorParameters(wire_radius_m=0.005), 0.01),
        (ConductorParameters(wire_radius_m=0.0087), 0.028308),
        (ConductorParameters(cross_section=240e-6), 0.028571),
        (ConductorParameters(cross_section=None), 0.01),
    ],
    ids=["default-radius", "8.7mm", "240mm2", "ideal"],
)
def test_min_section_length(conductor: ConductorParameters, expected_m: float) -> None:
    """At least 1 cm, and long enough that a section of copper keeps twice the
    1e-6 ohm below which a distributed conductor is refused."""
    assert min_section_length(conductor) == pytest.approx(expected_m, rel=1e-4)


# audit A3SG-01
@pytest.mark.finding("10.38")
def test_taps_on_a_thick_pen_keep_sections_the_conductor_model_accepts() -> None:
    """PEN radius 8.7 mm (238 mm²): a 1.2 cm section has 0.85e-6 ohm, below
    the 1e-6 ohm the Conductor model needs. The minimum section is 2.83 cm
    for this radius, so the tap snaps to the substation."""
    world = _build(
        substation=_ROD_SUBSTATION,
        building_types=[_RING_HOUSE],
        building_counts={"residential": 1},
        placement=ExplicitPlacement(positions=[(0.012, 20.0)]),
        cabinets=CabinetConfig(
            placement=ExplicitPlacement(positions=[(100.0, 0.0)]), fixed_count=1
        ),
        pen=PenConfig(wire_radius_m=0.0087),
    )
    (drop,) = _drops(world)
    assert drop.start_electrode == "trafo_rod_0"
    assert all(not c.is_ideal() for c in _pen(world))


def _random_positions(rng: np.random.Generator, n: int) -> list[tuple[float, float]]:
    """``n`` sites uniform in 400 m x 240 m, 12 m apart and from the origin."""
    pts = np.empty((0, 2))
    while len(pts) < n:
        p = rng.uniform([-200.0, -120.0], [200.0, 120.0])
        if np.hypot(*p) < 12.0 or (len(pts) and np.min(np.hypot(*(pts - p).T)) < 12.0):
            continue
        pts = np.vstack([pts, p])
    return [(float(x), float(y)) for x, y in pts]


def _feeding_cables(plan: NetworkPlan) -> dict[str, str]:
    """Cable of the nearest cabinet (Manhattan) of every building anchor (star)."""
    cabinets = [s for s in plan.sites if s.role == "cabinet" and s.anchor is not None]
    out = {}
    for b in plan.sites:
        if b.role != "building" or b.anchor is None:
            continue
        bx, by = b.position_xy
        best = min(
            cabinets,
            key=lambda c: abs(c.position_xy[0] - bx) + abs(c.position_xy[1] - by),
        )
        out[b.anchor] = f"pen_main_{best.anchor}"
    return out


def _nearest_on(sections: list[gf.Conductor], p: tuple[float, float]) -> tuple:
    """Point of ``sections`` nearest to ``p`` (horizontal plane)."""
    best, foot = math.inf, None
    for c in sections:
        ax, ay, bx, by = c.start[0], c.start[1], c.end[0], c.end[1]
        dx, dy = bx - ax, by - ay
        t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)
        t = min(max(t, 0.0), 1.0)
        q = (ax + t * dx, ay + t * dy)
        if math.dist(p, q) < best - 1e-12:
            best, foot = math.dist(p, q), q
    assert foot is not None
    return foot


# audit A3SG-01, A3SG-03: random-network stress test
@pytest.mark.finding("10.38")
@pytest.mark.slow
@pytest.mark.parametrize("topology", ["star", "radial"])
@pytest.mark.parametrize("n_houses", [60, 200])
def test_random_networks_build_and_route_as_documented(
    topology: str, n_houses: int
) -> None:
    """100 random networks per case (default catalogue for 60 houses, ring
    houses for 200 to keep the run short). Base: 1 of 100 (60 houses) and 2
    of 60 (200 houses) star networks failed with a ValidationError."""
    rng = np.random.default_rng(20261108 + n_houses)
    for seed in range(100):
        kw: dict[str, object] = dict(
            soil=HomogeneousSoilSpec(rho=100.0),
            building_counts={"residential": n_houses},
            placement=ExplicitPlacement(positions=_random_positions(rng, n_houses)),
        )
        if n_houses > 60:
            kw["building_types"] = [_RING_HOUSE]
        if topology == "radial":
            kw["pen_topology"] = RadialTrunkTopology()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = TnNetworkGenerator(TnNetworkConfig(**kw), seed=seed).plan()
            world = plan.to_world()
        _check_routing(world, "trafo_ring_0")
        pen = _pen(world)
        connected = {c.start_electrode for c in pen} | {c.end_electrode for c in pen}
        anchors = {s.anchor for s in plan.sites if s.role == "building"}
        assert anchors <= connected
        if topology != "star":
            continue
        # Star: every drop starts on the cable of the building's own cabinet,
        # at the point nearest to the building (up to the snapping distance).
        own: dict[str, list[gf.Conductor]] = {}
        for c in pen:
            if _is_main(c):
                own.setdefault(_cable(c.name), []).append(c)
        feeding = _feeding_cables(plan)
        for d in _drops(world):
            cable = own[feeding[d.end_electrode]]
            ends = {c.start_electrode for c in cable} | {c.end_electrode for c in cable}
            assert d.start_electrode in ends, d.name
            foot = _nearest_on(cable, d.end[:2])
            assert math.dist(d.start[:2], foot) < MIN_DROP_LENGTH_M


def test_overlap_helper_is_symmetric_for_collinear_pieces() -> None:
    """Self-check of the test helper on two collinear conductors."""
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="a", position=(0.0, 0.0, 0.0), length=1.0)
    gf.create_electrode(w, "rod", name="b", position=(10.0, 0.0, 0.0), length=1.0)
    gf.create_electrode(w, "rod", name="c", position=(4.0, 0.0, 0.0), length=1.0)
    c1 = gf.create_conductor(w, name="c1", start="a", end="b")
    c2 = gf.create_conductor(w, name="c2", start="c", end="b")
    for x, y in itertools.permutations([c1, c2]):
        assert _overlap(x, y) == pytest.approx(6.0)
