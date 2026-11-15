"""Service drops that leave one tap in one direction lie in a row (decision D9).

The PEN router of ``TnNetworkGenerator`` draws the drops of houses behind
each other from one tap in one direction: the five drops of the 5-house
radial network all start at the substation ring and run along -x, the
10-house star feeds five houses from the substation the same way. Up to
098bbbc they were drawn on one axis, and the inductance matrix read them as
a bundle in which every pair touches (no cross-section of four or more equal
wires realises that); ``diagnose()`` reported every pair as an overlap (5
houses: 10 pairs, 80 houses radial: 31, 200: 247).

The author's decision D9 of 2026-11-15: such drops lie side by side at the
touching distance, so that only neighbours touch. The router sorts the drops
of a row by length (shortest first, equal lengths in plan order) and lays
drop k of n at (k - (n - 1) / 2) touching distances from the common axis,
towards the left normal of the drop direction; the plan records the offset as
``PlannedBranch.axis_offset_m``, which the materialiser applies to both ends.
Plans written before (no offsets) build the worlds they built; a placement
without shared directions is bit-identical.
"""

from __future__ import annotations

import itertools
import json
import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import (
    HomogeneousSoilSpec,
    PenConfig,
    RadialTrunkTopology,
    StarCabinetTopology,
    TnNetworkConfig,
    TnNetworkGenerator,
)
from groundfield.generators.materialise import materialise_plan
from groundfield.generators.plan import NetworkPlan, PlannedBranch

pytestmark = pytest.mark.finding("10.78")

R = 0.005  # default PEN radius in m
TOUCH = 2.0 * R


def _generator(n: int, topology, *, sites: bool = False, **pen) -> TnNetworkGenerator:
    counts = {"residential": n}
    if sites:
        counts.update(small_industry=5, medium_industry=2)
    cfg = TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=100.0),
        building_counts=counts,
        pen_topology=topology,
        pen=PenConfig(inductance_model="neumann", segment_length_m=5.0, **pen),
    )
    return TnNetworkGenerator(cfg, seed=0)


def _overlaps(world: gf.World) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [d for d in gf.diagnose(world).findings if d.category == "overlap"]


def _plan(n: int, topology, **kw) -> NetworkPlan:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _generator(n, topology, **kw).plan()


def _plan_xy(world: gf.World, name: str) -> tuple[np.ndarray, np.ndarray]:
    c = world.get_conductor(name)
    return np.asarray(c.start, float)[:2], np.asarray(c.end, float)[:2]


# user decision D9
def test_the_five_drops_of_the_5_house_radial_network_lie_in_a_row() -> None:
    """Base 098bbbc: all five drops start at the substation ring
    (-4, 0, 0.6) and run along -x on one axis. Now they are parallel, one
    touching distance apart, sorted by length, centred on the old axis."""
    plan = _plan(5, RadialTrunkTopology())
    drops = [b for b in plan.all_branches if b.role == "service_drop"]
    assert len(drops) == 5
    world = materialise_plan(plan)
    lengths = []
    for b in drops:
        a, e = _plan_xy(world, b.name)
        lengths.append(float(np.linalg.norm(e - a)))
    order = sorted(range(5), key=lambda k: (lengths[k], k))
    offsets = [drops[k].axis_offset_m for k in order]
    assert offsets == pytest.approx([(k - 2.0) * TOUCH for k in range(5)], abs=1e-15)
    # parallel lines, neighbours one touching distance apart, the middle one
    # on the axis of 098bbbc (through the start point at the substation ring)
    a0, e0 = _plan_xy(world, drops[order[2]].name)
    u = (e0 - a0) / np.linalg.norm(e0 - a0)
    normal = np.array([-u[1], u[0]])
    for k, idx in enumerate(order):
        a, e = _plan_xy(world, drops[idx].name)
        assert abs(float(np.dot(a - a0, normal)) - (k - 2) * TOUCH) < 1e-12
        assert abs(float(np.dot(e - a0, normal)) - (k - 2) * TOUCH) < 1e-12
    assert np.allclose(a0, (-4.0, 0.0), atol=1e-12)


# user decision D9, audit W3B-03 / W3B-05
@pytest.mark.parametrize(
    ("n", "topology", "sites", "before"),
    [
        (5, RadialTrunkTopology(), False, 10),
        (10, RadialTrunkTopology(), False, 20),
        (30, RadialTrunkTopology(), True, 3),
        (80, RadialTrunkTopology(), False, 31),
        (10, StarCabinetTopology(), False, 10),
        (80, StarCabinetTopology(), True, 2),
    ],
    ids=["radial5", "radial10", "radial30_sites", "radial80", "star10", "star80_sites"],
)
def test_generated_networks_have_no_overlapping_drops(
    n, topology, sites, before
) -> None:
    """Base 098bbbc: ``before`` overlap findings, every one a pair of drops
    leaving one tap on one axis (stream w3perf, audit W3B-05). With the rows
    and the tolerance of W3B-03 none."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = _generator(n, topology, sites=sites).build()
    assert _overlaps(world) == []


# user decision D9
def test_drops_in_different_directions_keep_their_axes() -> None:
    """A placement with no shared directions has no offsets: star networks of
    30 and 200 houses (their drops leave every tap in different
    directions)."""
    for n in (30, 200):
        plan = _plan(n, StarCabinetTopology())
        assert all(b.axis_offset_m == 0.0 for b in plan.all_branches)


# user decision D9
def test_a_plan_written_before_builds_the_world_it_built() -> None:
    """A plan of 098bbbc has no ``axis_offset_m``: its drops stay on one
    axis, as they were (the materialiser does not regroup drops)."""
    plan = _plan(5, RadialTrunkTopology())
    payload = json.loads(plan.model_dump_json())
    for group in ("branches",):
        for branch in payload.get(group, []):
            branch.pop("axis_offset_m", None)
    old = NetworkPlan.model_validate(payload)
    assert all(b.axis_offset_m == 0.0 for b in old.all_branches)
    world = materialise_plan(old)
    starts = {
        tuple(np.round(np.asarray(world.get_conductor(b.name).start), 12))
        for b in old.all_branches
        if b.role == "service_drop"
    }
    assert starts == {(-4.0, 0.0, 0.6)}
    # the new plan round-trips through JSON with its offsets
    again = NetworkPlan.model_validate_json(plan.model_dump_json())
    assert (
        materialise_plan(again).model_dump_json()
        == materialise_plan(plan).model_dump_json()
    )


# user decision D9
def test_only_neighbours_touch_in_a_row() -> None:
    """The axis distances of the drops of a row: k touching distances for
    the k-th neighbour (exactly the drawn spacing, to round-off)."""
    world = _generator(5, RadialTrunkTopology()).build()
    drops = [c for c in world.conductors if c.name.startswith("pen_service_")]
    for a, b in itertools.combinations(drops, 2):
        pa, pb = np.asarray(a.start, float), np.asarray(b.start, float)
        d = abs(pa[1] - pb[1])
        k = round(d / TOUCH)
        assert k >= 1 and abs(d - k * TOUCH) < 1e-12


# user decision D9
def test_the_row_follows_the_pen_radius() -> None:
    """The spacing is the touching distance of the PEN of the network."""
    plan = _plan(5, RadialTrunkTopology(), wire_radius_m=0.008)
    offsets = sorted(
        b.axis_offset_m for b in plan.all_branches if b.role == "service_drop"
    )
    assert offsets == pytest.approx([k * 0.016 for k in (-2, -1, 0, 1, 2)], abs=1e-15)


# user decision D9
def test_the_layout_path_is_unaffected() -> None:
    """The AP1 catalogue path (LvNetworkLayout, drops attached by name)
    draws no rows: every branch of its plan keeps axis_offset_m = 0."""
    from groundfield.generators import LvNetworkLayout

    fps = []
    for i in range(4):
        cx, cy = 20.0 + 25.0 * i, 18.0 if i % 2 == 0 else -18.0
        fps.append(
            gf.BuildingFootprint(
                polygon_xy_m=[
                    (cx - 5.0, cy - 5.0),
                    (cx + 5.0, cy - 5.0),
                    (cx + 5.0, cy + 5.0),
                    (cx - 5.0, cy + 5.0),
                ],
                building_use="residential",
                osm_id=1000 + i,
            )
        )
    layout = LvNetworkLayout.from_footprints(fps, substation_xy=(0.0, 0.0))
    layout.add_pen_cable(start="substation", end_xy=(110.0, 0.0))
    layout.connect_buildings()
    plan = layout.plan(seed=0)
    assert plan.all_branches
    assert all(b.axis_offset_m == 0.0 for b in plan.all_branches)


# user decision D9
def test_the_materialiser_applies_the_offset_to_both_ends() -> None:
    """``axis_offset_m`` moves both ends of the conductor towards the left
    normal of its plan direction; a branch without horizontal extent cannot
    take one."""
    plan = _plan(5, RadialTrunkTopology())
    drop = next(b for b in plan.all_branches if b.role == "service_drop")
    plain = materialise_plan(
        plan.model_copy(
            update={
                "branches": [
                    b.model_copy(update={"axis_offset_m": 0.0})
                    if b.role == "service_drop"
                    else b
                    for b in plan.branches
                ]
            }
        )
    )
    moved = materialise_plan(
        plan.model_copy(
            update={
                "branches": [
                    b.model_copy(update={"axis_offset_m": 0.25})
                    if b.name == drop.name
                    else b
                    for b in plan.branches
                ]
            }
        )
    )
    a0, e0 = _plan_xy(plain, drop.name)
    a1, e1 = _plan_xy(moved, drop.name)
    u = (e0 - a0) / np.linalg.norm(e0 - a0)
    normal = np.array([-u[1], u[0]])
    assert np.allclose(a1 - a0, 0.25 * normal, atol=1e-14)
    assert np.allclose(e1 - e0, 0.25 * normal, atol=1e-14)
    c = moved.get_conductor(drop.name)
    assert (c.start_electrode, c.end_electrode) == (
        plain.get_conductor(drop.name).start_electrode,
        plain.get_conductor(drop.name).end_electrode,
    )
    with pytest.raises(ValueError, match="axis_offset_m"):
        PlannedBranch(
            name="x",
            role="pen",
            start="a",
            end="b",
            conductor=drop.conductor,
            axis_offset_m=math.inf,
        )
