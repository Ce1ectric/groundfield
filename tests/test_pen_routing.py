"""The PEN cables of generated TN networks never overlap.

Stream DOC-B found a 2-house :class:`TnNetworkGenerator` network whose
single cable cabinet sat 180 m beyond the houses (first slot of the
default cabinet row): both service drops ran from the cabinet back along
the PEN main, coaxial with it at the same depth. Coaxial overlapping
conductors have no finite Neumann mutual inductance; with
``inductance_model="neumann"`` the network was non-passive
(Im Z_in = -0.633 ohm at 1 kHz) and ``diagnose()`` reported nothing. The
default 30-house network fed its two collinear cabinets by two cables
overlapping over 144 m.

Since 2026.11 the PEN network is a nearest-point tree
(``generators._pen_routing``): every cable starts at the point of the
network already built nearest to its far end, inside a cable at a
leakage-free tap junction, so no cable runs along another.
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import (
    BuildingTypeSpec,
    ExplicitPlacement,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    HomogeneousSoilSpec,
    ManhattanGridPlacement,
    PenConfig,
    RadialTrunkTopology,
    TnNetworkConfig,
    TnNetworkGenerator,
)

pytestmark = pytest.mark.finding("10.38")

_TWO_HOUSES = ManhattanGridPlacement(spacing_x_m=25.0, spacing_y_m=30.0, n_per_row=2)
_RING_HOUSE = BuildingTypeSpec(
    name="residential",
    grounding=GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", size_m=10.0, depth_m=0.8)]
    ),
)


def _pen(world: gf.World) -> list[gf.Conductor]:
    return [c for c in world.conductors if c.conductor_type == "pen"]


def _overlapping_pairs(world: gf.World, tol: float = 1e-6) -> list[tuple]:
    """PEN conductors that share a stretch longer than 1 cm (collinear)."""
    pairs = []
    for c1, c2 in itertools.combinations(_pen(world), 2):
        a1, b1 = np.array(c1.start[:2]), np.array(c1.end[:2])
        a2, b2 = np.array(c2.start[:2]), np.array(c2.end[:2])
        length = float(np.linalg.norm(b1 - a1))
        u = (b1 - a1) / length
        n = np.array([-u[1], u[0]])
        if abs((a2 - a1) @ n) > tol or abs((b2 - a1) @ n) > tol:
            continue
        lo, hi = sorted([(a2 - a1) @ u, (b2 - a1) @ u])
        if min(length, hi) - max(0.0, lo) > 0.01:
            pairs.append((c1.name, c2.name))
    return pairs


def _build(**kw: object) -> gf.World:
    params: dict[str, object] = {"soil": HomogeneousSoilSpec(rho=100.0)}
    params.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TnNetworkGenerator(TnNetworkConfig(**params), seed=1).build()


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
}


# stream DOC-B: service drops ran back along the PEN main
@pytest.mark.parametrize("case", sorted(_CASES))
def test_no_two_pen_cables_overlap(case: str) -> None:
    """Before: the drops of the 2-house networks overlapped the main (and
    each other) over 167.5 m; the default 30-house network's cabinet cables
    overlapped over 144 m."""
    world = _build(**_CASES[case])
    assert _overlapping_pairs(world) == []


# stream DOC-B
def test_a_drop_starts_at_the_nearest_point_of_its_feeding_cable() -> None:
    world = _build(building_counts={"residential": 2}, placement=_TWO_HOUSES)
    drops = {c.name: c for c in _pen(world) if c.name.startswith("pen_service_")}
    near = drops["pen_service_residential_0_foundation_0"]
    # A tap junction on the main, straight beside the house's connection point.
    tap = world.get_junction(near.start_electrode)
    assert tap.position[:2] == pytest.approx((-17.5, 0.0))
    assert math.dist(near.start, near.end) == pytest.approx(5.0)
    # The house beside the substation is fed from the substation, not from
    # the cabinet 180 m away.
    far = drops["pen_service_residential_1_foundation_0"]
    assert far.start_electrode == "trafo_ring_0"
    assert math.dist(far.start, far.end) < 10.0
    total = sum(math.dist(c.start, c.end) for c in _pen(world))
    assert total == pytest.approx(184.0 + 5.0 + math.hypot(3.5, 5.0))


# stream DOC-B
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
def test_collinear_cabinets_are_chained() -> None:
    world = _build()
    mains = {c.name: c for c in _pen(world) if c.name.startswith("pen_main_")}
    # The farther cabinet hangs off the nearer one instead of a second
    # cable from the substation along the same line.
    assert mains["pen_main_kvs_0_rod_0"].start_electrode == "kvs_1_rod_0"


# stream DOC-B
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
    z = engine.solve(world).input_impedance()
    assert z[0].imag > 0.0 and z[1].imag > 0.0


# stream DOC-B
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
