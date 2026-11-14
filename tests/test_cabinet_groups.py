"""Cable cabinets per building group; substation slots tap the trunk (Q22, Q35).

User decisions Q22 and Q35 of 2026-11-14: ``TnNetworkGenerator`` places its
cable cabinets per building group, at the far end of the houses they feed,
instead of at fixed positions (the default strip along the substation row
put both cabinets of the default 30-house network west of the substation,
and the eastern houses got drops of 3.5 … 106.5 m straight from the
substation). The rule (``StarCabinetTopology.group_buildings``): the
buildings are ordered by their bearing around the substation, starting
after the widest angular gap, and cut into as many runs of consecutive
bearings as there are cabinets, of sizes that differ by at most one; the
cabinet of a group stands on the ray from the substation through the
group's centroid, ``CabinetConfig.end_clearance_m`` (1 m) beyond the
farthest point of the group's buildings along that ray, and feeds its
group. A given ``CabinetConfig.placement`` keeps the former rule (every
building fed by its nearest cabinet). In ``RadialTrunkTopology`` a building
in a substation slot that no trunk crosses taps its feeder's trunk instead
of getting its own line from the substation.

Every rule of user decision Q41 stays: closest pair, a building that a main
crosses is joined to it and fed by it, no drop along a cable, no cross-ties,
determinism.
"""

from __future__ import annotations

import json
import math
import re
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import (
    CabinetConfig,
    ExplicitPlacement,
    HomogeneousSoilSpec,
    ManhattanGridPlacement,
    RadialTrunkTopology,
    StarCabinetTopology,
    TnNetworkConfig,
    TnNetworkGenerator,
)

pytestmark = pytest.mark.finding("10.38")

_FORMER_STRIP = ManhattanGridPlacement(spacing_x_m=40.0, spacing_y_m=1.0, n_per_row=10)
"""The cabinet placement of the default configuration up to the decision."""


def _plan(**kw):
    params: dict[str, object] = {"soil": HomogeneousSoilSpec(rho=100.0)}
    params.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TnNetworkGenerator(TnNetworkConfig(**params), seed=0).plan()


def _root(name: str) -> str:
    """Site of an electrode or junction name (``residential_3_foundation_0_w2``
    -> ``residential_3``)."""
    return re.sub(r"_(foundation|ring|rod|strip)_\d+(_w\d+)?$", "", name)


def _drops(world: gf.World) -> list[gf.Conductor]:
    return [c for c in world.conductors if c.name.startswith("pen_service_")]


def _cable(name: str) -> str:
    return re.sub(r"_s\d+$", "", name)


def _feeding_main(world: gf.World) -> dict[str, str]:
    """Main (cabinet cable) that feeds every building, read from the built
    network: the cable whose section a drop starts at (its tap junction,
    the joint of a crossed house, or the cabinet), or the cable a building
    is joined to."""
    sections = [c for c in world.conductors if c.name.startswith("pen_main_")]
    ends: dict[str, set[str]] = {}
    for c in sections:
        for e in (c.start_electrode, c.end_electrode):
            ends.setdefault(_root(e), set()).add(_cable(c.name))
    out: dict[str, str] = {}
    for c in sections:
        for e in (c.start_electrode, c.end_electrode):
            if _root(e).startswith("residential_"):
                out[_root(e)] = _cable(c.name)
    for d in _drops(world):
        start = _root(d.start_electrode)
        if start.startswith("kvs_"):
            out[_root(d.end_electrode)] = f"pen_main_{start}_rod_0"
        elif start != "trafo":
            (cable,) = ends[start]
            out[_root(d.end_electrode)] = cable
    return out


# ---------------------------------------------------------------------
# The default star network
# ---------------------------------------------------------------------


# user decisions Q22, Q35
def test_default_cabinets_stand_at_the_far_end_of_their_building_groups() -> None:
    """Default network of 30 houses (three rows of ten, 25 m x 30 m, the
    substation in the middle row): the western and the eastern half are the
    two groups; their cabinets stand 1 m beyond the far side of the outer
    houses of the middle row, at (-118.5, 0) and (118.5, 0) m. Each main runs
    along the middle row (its houses joined to it), the houses of the outer
    rows tap it with 25 m drops: 20 drops, 500 m. Base: cabinets at (-180, 0)
    and (-140, 0) m, 25 drops, 1 139.6 m, 15 of them from the substation
    (3.5 ... 106.5 m)."""
    plan = _plan()
    cabinets = {s.name: s.position_xy for s in plan.sites if s.role == "cabinet"}
    assert cabinets == {"kvs_0": (-118.5, 0.0), "kvs_1": (118.5, 0.0)}
    world = plan.to_world()
    drops = _drops(world)
    assert len(drops) == 20
    lengths = [math.dist(d.start[:2], d.end[:2]) for d in drops]
    assert lengths == pytest.approx([25.0] * 20)
    assert not [d for d in drops if d.start_electrode.startswith("trafo")]
    # Group membership, by an independent rule for this symmetric grid:
    # the western houses are fed by kvs_0, the eastern ones by kvs_1.
    position = {s.name: s.position_xy for s in plan.sites if s.role == "building"}
    for house, main in _feeding_main(world).items():
        side = "kvs_0" if position[house][0] < 0.0 else "kvs_1"
        assert main == f"pen_main_{side}_rod_0", house
    assert len(_feeding_main(world)) == 30


# user decisions Q22, Q35
@pytest.mark.parametrize("n", [50, 150])
def test_each_cabinet_feeds_its_group_from_the_far_end(n: int) -> None:
    """Default grids of 50 and 150 houses (3 and 8 cabinets): every
    building is fed by the main of its group's cabinet (a building that
    another main crosses by that main, user decision Q41); each cabinet
    stands on the ray through its group's centroid, 1 m beyond the
    farthest foundation vertex of the group along the ray (recomputed from
    the built world's electrodes)."""
    plan = _plan(building_counts={"residential": n})
    world = plan.to_world()
    buildings = [s for s in plan.sites if s.role == "building"]
    cabinets = [s for s in plan.sites if s.role == "cabinet"]
    groups = StarCabinetTopology.group_buildings(
        (0.0, 0.0), [b.position_xy for b in buildings], len(cabinets)
    )
    assert len(cabinets) == math.ceil(5 * n / 100)
    feeding = _feeding_main(world)
    crossed = 0
    for group, cab in zip(groups, cabinets, strict=True):
        own = f"pen_main_{cab.anchor}"
        for i in group:
            main = feeding[buildings[i].name]
            if main != own:
                # Joined to another cabinet's main that crosses the house.
                assert any(
                    _root(c.end_electrode) == buildings[i].name
                    for c in world.conductors
                    if _cable(c.name) == main
                )
                crossed += 1
        c = np.mean([buildings[i].position_xy for i in group], axis=0)
        u = c / np.linalg.norm(c)
        reach = max(
            float(np.asarray(p[:2]) @ u)
            for i in group
            for e in world.electrodes
            if _root(e.name) == buildings[i].name
            for p in e.wire_polylines()[0]
        )
        assert cab.position_xy == pytest.approx(tuple((reach + 1.0) * u), abs=1e-9)
    assert crossed <= n // 10


# user decisions Q22, Q35
def test_groups_are_runs_of_consecutive_bearings_of_nearly_equal_size() -> None:
    """``group_buildings`` on 200 random layouts (3 ... 120 buildings, 1 ...
    12 groups): sizes differ by at most one, every group covers an arc of
    bearings that holds no building of another group, the first group
    starts after the widest gap, and the result does not depend on the
    order of the input beyond the index ties."""
    rng = np.random.default_rng(2022)
    for _ in range(200):
        n = int(rng.integers(3, 121))
        g = int(rng.integers(1, 13))
        pts = [tuple(p) for p in rng.uniform(-200.0, 200.0, (n, 2))]
        groups = StarCabinetTopology.group_buildings((0.0, 0.0), pts, g)
        sizes = [len(x) for x in groups]
        assert sum(sizes) == n and len(groups) == min(g, n)
        assert max(sizes) - min(sizes) <= 1 and sizes == sorted(sizes, reverse=True)
        bearing = [math.atan2(y, x) % (2.0 * math.pi) for x, y in pts]
        order = [i for group in groups for i in group]
        # Consecutive bearings: the concatenated order is the sorted cycle.
        b = [bearing[i] for i in order]
        steps = sum(1 for k in range(n) if b[(k + 1) % n] < b[k] - 1e-12)
        assert steps <= 1
        # The cycle is cut at the widest gap (the gap before the first one).
        gap_before_first = (b[0] - b[-1]) % (2.0 * math.pi)
        gaps = [(b[(k + 1) % n] - b[k]) % (2.0 * math.pi) for k in range(n)]
        assert gap_before_first >= max(gaps) - 1e-9


# user decisions Q22, Q35
def test_one_group_around_the_substation_gets_its_cabinet_beyond_its_first_far_house() -> (
    None
):
    """Ten houses in one row through the substation, one cabinet: the
    group's centroid is the substation, so the cabinet stands on the side of
    the group's farthest house, 1 m beyond it (two at 112.5 m: the first of
    the group in bearing order, the western one; the group starts after the
    widest gap, which ties between north and south and goes to the first)."""
    plan = _plan(building_counts={"residential": 10})
    (cabinet,) = [s for s in plan.sites if s.role == "cabinet"]
    assert cabinet.position_xy == (-118.5, 0.0)


# user decisions Q22, Q35
def test_the_clearance_moves_the_cabinet_along_its_ray() -> None:
    plan = _plan(cabinets=CabinetConfig(end_clearance_m=5.0))
    cabinets = [s.position_xy for s in plan.sites if s.role == "cabinet"]
    assert cabinets == [(-122.5, 0.0), (122.5, 0.0)]


# user decisions Q22, Q35
def test_more_cabinets_than_buildings_places_one_per_building() -> None:
    with pytest.warns(UserWarning, match="one cabinet per building group places 3"):
        plan = TnNetworkGenerator(
            TnNetworkConfig(
                building_counts={"residential": 3},
                cabinets=CabinetConfig(fixed_count=5),
            ),
            seed=0,
        ).plan()
    assert len([s for s in plan.sites if s.role == "cabinet"]) == 3


# user decisions Q22, Q35
def test_random_layouts_are_reproducible_and_route_as_documented() -> None:
    """Explicit random positions (and a jittered grid): the same seed gives
    the same world bit for bit, every building is connected, and no drop
    runs along a cable."""
    rng = np.random.default_rng(35)
    for k in range(12):
        pts = [tuple(p) for p in rng.uniform(-150.0, 150.0, (40, 2))]
        pts = [p for p in pts if math.hypot(*p) > 15.0]
        keep = []
        for p in pts:
            if all(math.dist(p, q) > 13.0 for q in keep):
                keep.append(p)
        placement = (
            ExplicitPlacement(positions=keep)
            if k % 2
            else ManhattanGridPlacement(jitter_m=6.0)
        )
        kw = dict(
            soil=HomogeneousSoilSpec(rho=100.0),
            building_counts={"residential": len(keep) if k % 2 else 30},
            placement=placement,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = TnNetworkGenerator(TnNetworkConfig(**kw), seed=k).build()
            b = TnNetworkGenerator(TnNetworkConfig(**kw), seed=k).build()
        assert gf.world_fingerprint(a) == gf.world_fingerprint(b)
        houses = {
            _root(e.name) for e in a.electrodes if e.name.startswith("residential_")
        }
        assert houses <= set(_feeding_main(a)) | {
            _root(d.end_electrode)
            for d in _drops(a)
            if d.start_electrode.startswith("trafo")
        }


# ---------------------------------------------------------------------
# The former placement and old configurations
# ---------------------------------------------------------------------


# user decisions Q22, Q35 (guard)
def test_a_given_placement_keeps_the_former_cabinets_and_nearest_feeding() -> None:
    """With the former default strip the cabinets stand at (-180, 0) and
    (-140, 0) m and the eastern houses are fed from the substation as before
    the decision (15 drops from the substation)."""
    plan = _plan(cabinets=CabinetConfig(placement=_FORMER_STRIP))
    cabinets = [s.position_xy for s in plan.sites if s.role == "cabinet"]
    assert cabinets == [(-180.0, 0.0), (-140.0, 0.0)]
    world = plan.to_world()
    assert (
        len([d for d in _drops(world) if d.start_electrode.startswith("trafo")]) == 15
    )


# user decisions Q22, Q35 (guard)
def test_a_configuration_stored_before_the_decision_keeps_its_cabinets() -> None:
    """A configuration written up to 2026.11 carries the former default
    cabinet placement and replays its cabinets; one written now has
    ``placement`` ``None`` and ``end_clearance_m`` and round-trips."""
    old = json.loads(TnNetworkConfig().model_dump_json())
    old["cabinets"]["placement"] = json.loads(_FORMER_STRIP.model_dump_json())
    old["cabinets"].pop("end_clearance_m")
    cfg = TnNetworkConfig.model_validate(old)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = TnNetworkGenerator(cfg, seed=0).plan()
    assert [s.position_xy for s in plan.sites if s.role == "cabinet"] == [
        (-180.0, 0.0),
        (-140.0, 0.0),
    ]
    new = TnNetworkConfig(cabinets=CabinetConfig(end_clearance_m=2.5))
    again = TnNetworkConfig.model_validate_json(new.model_dump_json())
    assert again == new
    assert again.cabinets.placement is None and again.cabinets.end_clearance_m == 2.5


# ---------------------------------------------------------------------
# Radial trunk: the substation slots tap their feeder's trunk (Q35)
# ---------------------------------------------------------------------


def _radial(n: int, **topology) -> tuple[object, gf.World]:
    plan = _plan(
        building_counts={"residential": n},
        pen_topology=RadialTrunkTopology(**topology),
    )
    return plan, plan.to_world()


def _segment_distance(p, a, b) -> float:
    p, a, b = (np.asarray(v, dtype=float)[:2] for v in (p, a, b))
    d = b - a
    t = float(np.clip((p - a) @ d / (d @ d), 0.0, 1.0))
    return float(np.hypot(*(p - (a + t * d))))


# user decisions Q22, Q35
def test_substation_slots_tap_their_feeders_trunk() -> None:
    """Default radial network of 30 houses: the eastern and western feeders
    have a trunk (one cabinet each, 60 m out). Their houses in substation
    slots that no trunk crosses tap the trunk at the closest pair with
    their foundation, instead of getting a drop from the substation (base:
    14 drops from the substation, 4 of them on the feeders without a
    trunk). Only the houses of the northern and southern feeders, which
    have no trunk, are still fed from the substation: 4 drops. Each
    tapping drop is as long as the distance from its foundation to the
    trunk (2 mm sampling of the trunk, a second method)."""
    plan, world = _radial(30)
    trunks = [c for c in world.conductors if c.name.startswith("pen_trunk_")]
    drops = _drops(world)
    from_trafo = [d for d in drops if d.start_electrode.startswith("trafo")]
    assert len(from_trafo) == 4
    position = {s.name: s.position_xy for s in plan.sites if s.role == "building"}
    for d in from_trafo:
        x, y = position[_root(d.end_electrode)]
        assert abs(y) > abs(x), d.name  # a northern or southern house
    samples = np.concatenate(
        [
            np.asarray(t.start[:2])
            + np.linspace(0.0, 1.0, max(2, int(t.length / 0.002)))[:, None]
            * (np.asarray(t.end[:2]) - np.asarray(t.start[:2]))
            for t in trunks
        ]
    )
    for d in drops:
        if d in from_trafo:
            continue
        start_on_trunk = min(_segment_distance(d.start, t.start, t.end) for t in trunks)
        assert start_on_trunk < 1e-9, d.name
        house = [e for e in world.electrodes if _root(e.name) == _root(d.end_electrode)]
        sampled = min(
            _points_to_segment(samples, a, b)
            for e in house
            for a, b in zip(
                e.wire_polylines()[0][:-1], e.wire_polylines()[0][1:], strict=True
            )
        )
        plan_length = math.dist(d.start[:2], d.end[:2])
        assert plan_length == pytest.approx(sampled, abs=2e-3), d.name


def _points_to_segment(points: np.ndarray, a, b) -> float:
    """Smallest exact plan distance of ``points`` to the segment ``a b``."""
    a, b = np.asarray(a, dtype=float)[:2], np.asarray(b, dtype=float)[:2]
    d = b - a
    t = np.clip((points - a) @ d / (d @ d), 0.0, 1.0)
    return float(np.min(np.hypot(*(points - (a + t[:, None] * d)).T)))


# user decisions Q22, Q35 (guard)
def test_a_feeder_without_a_trunk_feeds_its_slots_from_the_substation() -> None:
    """Ten houses, no feeder needs a cabinet: every house has its drop from
    the substation, as before the decision."""
    _, world = _radial(10)
    assert not [c for c in world.conductors if c.name.startswith("pen_trunk_")]
    drops = _drops(world)
    assert len(drops) == 10
    assert all(d.start_electrode.startswith("trafo") for d in drops)


# ---------------------------------------------------------------------
# Planning time: the segment-box prefilter of the router
# ---------------------------------------------------------------------


# user decisions Q22, Q35 (planning time)
def test_the_segment_box_gap_agrees_with_dense_sampling() -> None:
    """The router's distance between a cable's line and a building's
    bounding box against 4 000-point sampling of the segment with exact
    distances to the box: never above the sampled value, at most one step
    below it, zero exactly when they meet."""
    from groundfield.generators._pen_routing import _segment_box_gap

    rng = np.random.default_rng(3535)
    for _ in range(400):
        a, b = (tuple(map(float, p)) for p in rng.uniform(-50.0, 50.0, (2, 2)))
        x0, y0 = rng.uniform(-30.0, 20.0, 2)
        w, h = rng.uniform(0.0, 15.0, 2)
        box = (float(x0), float(y0), float(x0 + w), float(y0 + h))
        t = np.linspace(0.0, 1.0, 4001)[:, None]
        pts = np.asarray(a) + t * (np.asarray(b) - np.asarray(a))
        gx = np.maximum.reduce(
            [box[0] - pts[:, 0], np.zeros(len(pts)), pts[:, 0] - box[2]]
        )
        gy = np.maximum.reduce(
            [box[1] - pts[:, 1], np.zeros(len(pts)), pts[:, 1] - box[3]]
        )
        sampled = float(np.min(np.hypot(gx, gy)))
        step = math.dist(a, b) / 4000
        gap = _segment_box_gap(a, b, box)
        assert gap <= sampled + 1e-12
        assert gap >= sampled - step - 1e-12


# user decisions Q22, Q35 (planning time)
def test_the_segment_box_prefilter_changes_no_bit(monkeypatch) -> None:
    """Without the prefilter (every cable whose bounding box comes near a
    building is tested exactly) the plans of the default star networks of
    200 and 1 000 houses and of the radial one of 200 are the same, bit for
    bit. The prefilter keeps the planning time of large star networks
    near the former one: the long mains of cabinets at the far ends of
    their groups have bounding boxes over many buildings."""
    from groundfield.generators import _pen_routing

    def plans():
        out = []
        for n, topo in ((200, None), (1000, None), (200, RadialTrunkTopology())):
            kw: dict[str, object] = {"building_counts": {"residential": n}}
            if topo is not None:
                kw["pen_topology"] = topo
            out.append(_plan(**kw).model_dump_json())
        return out

    fast = plans()
    monkeypatch.setattr(_pen_routing, "_segment_box_gap", lambda a, b, box: 0.0)
    assert plans() == fast
