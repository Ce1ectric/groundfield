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
group's centroid, ``CabinetConfig.end_clearance_m`` (1 m) beyond the group's
electrodes along it (rods included since audit W1B-02), when it stands there
within ``CabinetConfig.max_far_end_gap_m`` (20 m) of its houses, otherwise on
the ray through the group's far-end building (the largest projection on the
direction of the centroid), 1 m beyond that building (audit W1B-01); it feeds
its group; a cabinet that would come closer than ``end_clearance_m`` to
another site raises (audit W1B-02). A given ``CabinetConfig.placement`` keeps the former rule (every
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
    Discrete,
    ExplicitPlacement,
    GroundingSystemSpec,
    HomogeneousSoilSpec,
    ManhattanGridPlacement,
    RadialTrunkTopology,
    RingElectrodeSpec,
    RodElectrodeSpec,
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
    stands where the rule of audit W1B-01 and the coordinator's decision on
    it puts it (recomputed from the built world's electrodes): on the ray
    through its group's centroid when it stands there within 20 m of its
    houses, otherwise on the ray through the far-end building."""
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
            if buildings[i].name not in feeding:
                # A drop from the substation: the point of the group's main
                # nearest to the house is the main's substation end (the
                # closest-pair rule, user decision Q41).
                _assert_nearest_point_is_the_substation_end(
                    world, own, buildings[i].name
                )
                continue
            main = feeding[buildings[i].name]
            if main != own:
                # Joined to another cabinet's main that crosses the house.
                assert any(
                    _root(c.end_electrode) == buildings[i].name
                    for c in world.conductors
                    if _cable(c.name) == main
                )
                crossed += 1
    # The cabinet on the ray through its group's far-end building, 1 m
    # beyond that building (audit W1B-01; up to then on the ray through the
    # centroid, 1 m beyond the group's farthest foundation vertex).
    expected = _expected_cabinet(plan, world)
    for cab, (xy, far, *_) in zip(cabinets, expected, strict=True):
        assert cab.position_xy == pytest.approx(tuple(xy), abs=1e-9), far
    assert crossed <= n // 10


def _assert_nearest_point_is_the_substation_end(
    world: gf.World, main: str, house: str
) -> None:
    """The point of ``main`` nearest to the electrodes of ``house`` (2 mm
    sampling of the house) is the main's start at the substation, within
    one minimum section."""
    from groundfield.generators._pen_routing import MIN_DROP_LENGTH_M

    sections = [c for c in world.conductors if _cable(c.name) == main]
    start = next(c for c in sections if _root(c.start_electrode) == "trafo")
    points, _ = _site_samples(world, house)
    best, foot = math.inf, None
    for c in sections:
        a, b = np.asarray(c.start[:2], float), np.asarray(c.end[:2], float)
        d = b - a
        t = np.clip((points - a) @ d / float(d @ d), 0.0, 1.0)
        q = a + t[:, None] * d
        dist = np.hypot(*(points - q).T)
        k = int(np.argmin(dist))
        if dist[k] < best:
            best, foot = float(dist[k]), q[k]
    assert foot is not None
    assert math.dist(foot, start.start[:2]) <= MIN_DROP_LENGTH_M + 1e-3, (house, foot)


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
    the same world bit for bit, every building is connected, the PEN is one
    tree that feeds every house once (a drop, or joined to a main, never
    both), and no drop runs along a cable."""
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
        # audit W1B-08: the routing checks the docstring promises (the
        # base 6f63deb passed this test without them).
        _assert_one_tree_feeding_every_house_once(a, houses)
        _assert_no_drop_runs_along_a_cable(a)


def _pen_node(name: str) -> str:
    """Node of the PEN graph: the site of an electrode, or a junction."""
    return _root(name)


def _assert_one_tree_feeding_every_house_once(world: gf.World, houses: set) -> None:
    """The PEN conductors form one tree through the substation; every house
    is a node of it, fed either by one drop or by being joined to a main."""
    pen = [c for c in world.conductors if c.name.startswith("pen_")]
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in pen:
        a, b = find(_pen_node(c.start_electrode)), find(_pen_node(c.end_electrode))
        assert a != b, f"{c.name} closes a loop"
        parent[a] = b
    roots = {find(n) for n in list(parent)}
    assert len(roots) == 1, "the PEN is not one tree"
    assert houses <= set(parent) and "trafo" in parent
    drops_to: dict[str, int] = {}
    for d in _drops(world):
        drops_to[_root(d.end_electrode)] = drops_to.get(_root(d.end_electrode), 0) + 1
    joined = {
        _root(e)
        for c in pen
        if c.name.startswith("pen_main_")
        for e in (c.start_electrode, c.end_electrode)
    }
    for house in houses:
        assert drops_to.get(house, 0) + (house in joined) == 1, house


def _cross(a, b) -> float:
    """z component of the cross product of two plan vectors."""
    return float(a[0] * b[1] - a[1] * b[0])


def _collinear_overlap(c1: gf.Conductor, c2: gf.Conductor, tol: float) -> float:
    """Length in m over which two straight conductors run on one line."""
    a, b = np.asarray(c1.start[:2], float), np.asarray(c1.end[:2], float)
    p, q = np.asarray(c2.start[:2], float), np.asarray(c2.end[:2], float)
    d = b - a
    length = float(np.hypot(*d))
    if length == 0.0:
        return 0.0
    u = d / length
    off = [abs(_cross(u, x - a)) for x in (p, q)]
    if max(off) > tol:
        return 0.0
    s = sorted(float((x - a) @ u) for x in (p, q))
    return max(0.0, min(length, s[1]) - max(0.0, s[0]))


def _assert_no_drop_runs_along_a_cable(world: gf.World) -> None:
    """No drop shares more than one minimum section (1 cm) with a main
    (audits R4A-02, R4B-01; the router's tolerance)."""
    from groundfield.generators._pen_routing import MIN_DROP_LENGTH_M

    mains = [c for c in world.conductors if c.name.startswith("pen_main_")]
    for d in _drops(world):
        shared = max(
            (_collinear_overlap(d, c, MIN_DROP_LENGTH_M) for c in mains), default=0.0
        )
        assert shared <= MIN_DROP_LENGTH_M * (1 + 1e-6), (d.name, shared)


# ---------------------------------------------------------------------
# Wide groups and the clearance of the cabinet's electrodes (audit W1B)
# ---------------------------------------------------------------------

_SAMPLE_M = 0.002
"""Sampling step in m of the independent geometry checks below."""


def _site_samples(world: gf.World, root: str) -> tuple[np.ndarray, np.ndarray]:
    """Plan geometry of a site's electrodes, independent of the generator:
    points (rod heads and 2 mm samples of every wire and ring circle) and
    the straight wire segments ``(n, 2, 2)``."""
    points: list[np.ndarray] = []
    segments: list[tuple] = []
    for e in world.electrodes:
        if _root(e.name) != root:
            continue
        if e.kind == "rod":
            points.append(np.asarray([e.connection_point[:2]], dtype=float))
        elif e.kind == "ring":
            cx, cy, _ = e.center
            n = max(64, math.ceil(2.0 * math.pi * e.radius / _SAMPLE_M))
            t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
            points.append(
                np.column_stack([cx + e.radius * np.cos(t), cy + e.radius * np.sin(t)])
            )
        else:
            for run in e.wire_polylines():
                xy = np.asarray(run, dtype=float)[:, :2]
                for a, b in zip(xy[:-1], xy[1:], strict=True):
                    segments.append((a, b))
                    n = max(2, math.ceil(float(np.hypot(*(b - a))) / _SAMPLE_M) + 1)
                    points.append(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a))
    return (
        np.concatenate(points) if points else np.zeros((0, 2)),
        np.asarray(segments, dtype=float).reshape(-1, 2, 2),
    )


def _points_to_segments(points: np.ndarray, segments: np.ndarray) -> float:
    """Smallest exact plan distance of ``points`` to ``segments``."""
    best = math.inf
    for a, b in segments:
        d = b - a
        dd = float(d @ d)
        t = np.zeros(len(points)) if dd == 0.0 else (points - a) @ d / dd
        t = np.clip(t, 0.0, 1.0)
        best = min(best, float(np.min(np.hypot(*(points - (a + t[:, None] * d)).T))))
    return best


def _site_gap(world: gf.World, a: str, b: str) -> float:
    """Plan distance in m between the electrodes of two sites (sampled to
    within 1 mm)."""
    pa, sa = _site_samples(world, a)
    pb, sb = _site_samples(world, b)
    gap = math.inf
    if len(pa) and len(pb):
        diff = pa[:, None, :] - pb[None, :, :] if len(pa) * len(pb) < 4e6 else None
        if diff is not None:
            gap = float(np.min(np.hypot(diff[..., 0], diff[..., 1])))
    if len(sb):
        gap = min(gap, _points_to_segments(pa, sb))
    if len(sa):
        gap = min(gap, _points_to_segments(pb, sa))
    return gap


def _reach(world: gf.World, root: str, v: np.ndarray) -> float:
    """Largest projection in m on the unit vector ``v`` of a site's
    electrodes in plan: wire vertices, ring circles (centre and radius),
    rod heads."""
    best = -math.inf
    for e in world.electrodes:
        if _root(e.name) != root:
            continue
        if e.kind == "rod":
            best = max(best, float(np.asarray(e.connection_point[:2]) @ v))
        elif e.kind == "ring":
            best = max(best, float(np.asarray(e.center[:2]) @ v) + e.radius)
        else:
            for run in e.wire_polylines():
                best = max(best, float(np.max(np.asarray(run)[:, :2] @ v)))
    return best


def _point_gap(world: gf.World, root: str, xy) -> float:
    """Plan distance in m from ``xy`` to the electrodes of a site (2 mm
    sampling of its wires and circles, exact to its rods and wire pieces)."""
    points, segments = _site_samples(world, root)
    p = np.asarray(xy, dtype=float)[None, :]
    gap = float(np.min(np.hypot(*(points - p).T))) if len(points) else math.inf
    if len(segments):
        gap = min(gap, _points_to_segments(p, segments))
    return gap


def _expected_cabinet(
    plan, world: gf.World, clearance: float = 1.0, max_gap: float = 20.0
) -> list:
    """Cabinet positions by the rule of audits W1B-01 / W1B-02 and the
    coordinator's decision on them, recomputed from the built world: on the
    ray through the group's centroid, ``clearance`` m beyond the largest
    projection on it of every electrode of the group (rods included) when
    that point lies within ``max_gap`` m of an electrode of the group;
    otherwise on the ray through the far-end building (the largest
    projection of its position on the centroid direction; ties: nearest to
    the ray), ``clearance`` m beyond that building's electrodes (the default
    cabinet rod has no plan extent). Returns (position, the building it
    stands beyond, group, rule, the gap in m between the centroid-ray
    position and the nearest electrode of the group)."""
    buildings = [s for s in plan.sites if s.role == "building"]
    n = len([s for s in plan.sites if s.role == "cabinet"])
    out = []
    groups = StarCabinetTopology.group_buildings(
        (0.0, 0.0), [b.position_xy for b in buildings], n
    )
    for group in groups:
        pos = np.asarray([buildings[i].position_xy for i in group], dtype=float)
        u = pos.mean(axis=0)
        u = u / np.hypot(*u)
        reach = max(_reach(world, buildings[i].name, u) for i in group)
        centre = (reach + clearance) * u
        gap = min(_point_gap(world, buildings[i].name, centre) for i in group)
        if gap <= max_gap:
            out.append((centre, None, group, "centroid", gap))
            continue
        proj = pos @ u
        ties = [k for k in range(len(group)) if proj[k] >= proj.max() - 1e-9]
        far = min(ties, key=lambda k: (abs(_cross(u, pos[k])), k))
        v = pos[far] / np.hypot(*pos[far])
        reach = _reach(world, buildings[group[far]].name, v)
        out.append(
            ((reach + clearance) * v, buildings[group[far]].name, group, "far", gap)
        )
    return out


# audit W1B-01; user decisions Q22, Q35
@pytest.mark.parametrize(
    "counts",
    [
        {"residential": 10, "small_industry": 1},
        {"residential": 10, "medium_industry": 1},
        {"residential": 5, "small_industry": 5, "medium_industry": 1},
    ],
)
def test_the_cabinet_of_a_wide_group_stands_next_to_its_far_end_building(
    counts: dict,
) -> None:
    """AP1 combinations with one group of about 180 degrees: the ray through
    the group's centroid passed no building, and the base ``6f63deb`` put
    the cabinet 50.4 m from the nearest one (10 houses + 1 small commercial:
    drops 527.4 m, base strip 334.9 m). The cabinet stands on the ray through
    the group's far-end building, 1 m beyond that building's electrodes
    along the ray (recomputed from the built world), next to it."""
    plan = _plan(building_counts=counts)
    world = plan.to_world()
    cabinets = [s for s in plan.sites if s.role == "cabinet"]
    buildings = [s for s in plan.sites if s.role == "building"]
    for cab, (xy, far, group, rule, _gap) in zip(
        cabinets, _expected_cabinet(plan, world), strict=True
    ):
        assert rule == "far"  # 50.4 m from the houses on the centroid ray
        assert cab.position_xy == pytest.approx(tuple(xy), abs=1e-9), far
        nearest = min(_site_gap(world, cab.name, buildings[i].name) for i in group)
        assert 1.0 - 1e-3 <= nearest <= 2.0, (cab.name, nearest)
        assert _site_gap(world, cab.name, far) == pytest.approx(nearest, abs=1e-3)


# audit W1B-01 (CabinetConfig.max_far_end_gap_m); user decisions Q22, Q35
def test_the_50_house_grid_keeps_its_cabinets_on_the_centroid_rays() -> None:
    """The three groups of the default 50-house grid stand 8.8 ... 12.8 m from
    their houses on the rays through their centroids (as up to audit W1B):
    within ``max_far_end_gap_m`` (20 m), so they stay there and no house is
    fed straight from the substation. With the far-end rule alone (the first
    fix of W1B-01, also ``max_far_end_gap_m`` below 8.8 m) six houses of the
    southern group, whose main ran to its eastern corner, had a drop from
    the substation ring, up to 116.8 m."""
    plan = _plan(building_counts={"residential": 50})
    world = plan.to_world()
    rules = [e[3] for e in _expected_cabinet(plan, world)]
    assert rules == ["centroid"] * 3
    cabinets = [s.position_xy for s in plan.sites if s.role == "cabinet"]
    for xy, (expected, *_rest) in zip(
        cabinets, _expected_cabinet(plan, world), strict=True
    ):
        assert xy == pytest.approx(tuple(expected), abs=1e-9)
    assert not [d for d in _drops(world) if d.start_electrode.startswith("trafo")]
    far = _plan(
        building_counts={"residential": 50},
        cabinets=CabinetConfig(max_far_end_gap_m=5.0),
    ).to_world()
    from_trafo = [d for d in _drops(far) if d.start_electrode.startswith("trafo")]
    assert len(from_trafo) == 6
    assert max(math.dist(d.start[:2], d.end[:2]) for d in from_trafo) == pytest.approx(
        116.8, abs=0.05
    )


# audit W1B-01 (CabinetConfig.max_far_end_gap_m)
@pytest.mark.parametrize("value", [-1.0, 0.0, float("nan"), float("inf")])
def test_a_non_positive_or_infinite_far_end_gap_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="max_far_end_gap_m"):
        CabinetConfig(max_far_end_gap_m=value)
    cfg = TnNetworkConfig(
        cabinets=CabinetConfig(max_far_end_gap_m=Discrete(values=[-2.0]))
    )
    with pytest.raises(ValueError, match="max_far_end_gap_m.*-2"):
        TnNetworkGenerator(cfg, seed=0).plan()


# ---------------------------------------------------------------------
# The threshold of the cabinet rule (audit W1R2; user decision D7: keep
# the 20 m threshold for 2026.11, document the step, record the rule)
# ---------------------------------------------------------------------

_RULE = {"centroid": "centroid_ray", "far": "far_end_ray"}


def _spacing(spacing_x_m: float) -> ManhattanGridPlacement:
    """The default building grid with another column spacing."""
    return ManhattanGridPlacement(
        spacing_x_m=spacing_x_m, spacing_y_m=30.0, n_per_row=10
    )


# audit W1R2-02; user decision D7
@pytest.mark.parametrize(
    "counts, rules",
    [
        ({"residential": 50}, ["centroid_ray"] * 3),
        ({"residential": 10, "small_industry": 1}, ["far_end_ray"]),
        (
            {"residential": 100},
            ["centroid_ray", "far_end_ray", "centroid_ray", "far_end_ray"]
            + ["centroid_ray"],
        ),
        (
            {"residential": 80, "medium_industry": 5},
            ["far_end_ray", "centroid_ray", "far_end_ray", "centroid_ray"]
            + ["far_end_ray"],
        ),
    ],
    ids=["grid-50", "res10+small1", "grid-100", "res80+medium5"],
)
def test_every_cabinet_records_the_rule_that_placed_it_and_its_gap(
    counts: dict, rules: list
) -> None:
    """The plan records per cabinet which rule placed it (``cabinet_rule``:
    ``"centroid_ray"`` / ``"far_end_ray"``) and the gap between its position
    on the centroid ray and its group (``centroid_ray_gap_m``, the value
    compared with ``max_far_end_gap_m``), so a parameter study sees where it
    crosses the threshold. Both agree with the rule recomputed from the
    built world (own projections, exact distances to wire pieces and rods).
    Up to then the plan did not say which rule a cabinet took (80 houses + 5
    medium commercial: one group 20.21 m from its houses, 0.21 m above the
    threshold). Every other site records nothing; the plan's JSON round
    trip keeps both fields, and materialising ignores them."""
    plan = _plan(building_counts=counts)
    world = plan.to_world()
    cabinets = [s for s in plan.sites if s.role == "cabinet"]
    expected = _expected_cabinet(plan, world)
    assert [c.cabinet_rule for c in cabinets] == rules
    assert [_RULE[e[3]] for e in expected] == rules
    for cab, (*_rest, gap) in zip(cabinets, expected, strict=True):
        assert cab.centroid_ray_gap_m == pytest.approx(gap, abs=1e-9), cab.name
    assert all(
        s.cabinet_rule is None and s.centroid_ray_gap_m is None
        for s in plan.all_sites
        if s.role != "cabinet"
    )
    again = type(plan).model_validate_json(plan.model_dump_json())
    assert again == plan
    bare = plan.model_copy(
        update={
            "sites": [
                s.model_copy(update={"cabinet_rule": None, "centroid_ray_gap_m": None})
                for s in plan.sites
            ]
        }
    )
    assert bare.to_world().model_dump_json() == world.model_dump_json()


# audit W1R2-02; user decision D7
def test_cabinets_not_placed_per_building_group_record_no_rule() -> None:
    """A given placement and the radial topology place their cabinets
    without the rule; their cabinets record nothing."""
    for kw in (
        {"cabinets": CabinetConfig(placement=_FORMER_STRIP)},
        {"pen_topology": RadialTrunkTopology(), "building_counts": {"residential": 60}},
    ):
        cabinets = [s for s in _plan(**kw).sites if s.role == "cabinet"]
        assert cabinets
        assert all(
            c.cabinet_rule is None and c.centroid_ray_gap_m is None for c in cabinets
        )


# audit W1R2-02; user decision D7
@pytest.mark.parametrize(
    "kw, message",
    [
        (
            dict(role="building", cabinet_rule="centroid_ray", centroid_ray_gap_m=3.0),
            r"cabinets only.*'building'",
        ),
        (dict(role="cabinet", cabinet_rule="far_end_ray"), "recorded together"),
        (dict(role="cabinet", centroid_ray_gap_m=3.0), "recorded together"),
        (
            dict(role="cabinet", cabinet_rule="centroid_ray", centroid_ray_gap_m=-1.0),
            r"centroid_ray_gap_m\n.*greater than or equal to 0",
        ),
        (
            dict(
                role="cabinet",
                cabinet_rule="centroid_ray",
                centroid_ray_gap_m=float("nan"),
            ),
            r"centroid_ray_gap_m\n.*finite number",
        ),
        (
            dict(role="cabinet", cabinet_rule="nearest", centroid_ray_gap_m=1.0),
            r"cabinet_rule\n.*'centroid_ray' or 'far_end_ray'",
        ),
    ],
)
def test_a_planned_site_records_the_cabinet_rule_consistently(
    kw: dict, message: str
) -> None:
    """Only a cabinet records a rule, always with its gap (finite, at
    least 0); the rule is one of the two names."""
    from groundfield.generators.plan import PlannedSite

    with pytest.raises(ValueError, match=message):
        PlannedSite(name="kvs_0", position_xy=(0.0, 0.0), **kw)


# audit W1R2-02; user decision D7
def test_the_cabinet_jumps_where_the_recorded_gap_crosses_the_threshold() -> None:
    """User decision D7 keeps the 20 m threshold for 2026.11 and documents
    its step. In the 50-house grid with its house columns 32.0 / 32.2 m apart
    the southern group's gap on its centroid ray goes 19.99 -> 20.22 m,
    across ``max_far_end_gap_m``: its cabinet moves from (11.75, -84.99) m on
    the centroid ray to (151.86, -62.88) m on the ray through its far-end
    building, 141.8 m, and the drops of the network from 45 / 1 788.9 m to
    43 / 1 443.0 m (the other two cabinets move by less than 0.9 m). The
    recorded rule and gap show the step."""
    before = _plan(building_counts={"residential": 50}, placement=_spacing(32.0))
    after = _plan(building_counts={"residential": 50}, placement=_spacing(32.2))
    cab0 = [s for s in before.sites if s.role == "cabinet"]
    cab1 = [s for s in after.sites if s.role == "cabinet"]
    assert [c.cabinet_rule for c in cab0] == ["centroid_ray"] * 3
    assert [c.cabinet_rule for c in cab1] == ["centroid_ray", "far_end_ray"] + [
        "centroid_ray"
    ]
    assert cab0[1].centroid_ray_gap_m == pytest.approx(19.987, abs=5e-4)
    assert cab1[1].centroid_ray_gap_m == pytest.approx(20.216, abs=5e-4)
    assert cab0[1].position_xy == pytest.approx((11.751, -84.987), abs=5e-4)
    assert cab1[1].position_xy == pytest.approx((151.859, -62.882), abs=5e-4)
    assert math.dist(cab0[1].position_xy, cab1[1].position_xy) == pytest.approx(
        141.8, abs=0.05
    )
    for a, b in ((cab0[0], cab1[0]), (cab0[2], cab1[2])):
        assert math.dist(a.position_xy, b.position_xy) < 0.9
    for plan, (n, length) in ((before, (45, 1788.9)), (after, (43, 1443.0))):
        drops = _drops(plan.to_world())
        assert len(drops) == n
        assert sum(math.dist(d.start, d.end) for d in drops) == pytest.approx(
            length, abs=0.05
        )


# audit W1R2-04; user decision D7
def test_the_threshold_moves_two_groups_of_the_100_house_grid_either_way() -> None:
    """15 / 20 / 25 m give 4 / 2 / 0 groups of the default 100-house grid on
    the far-end ray (the grid is symmetric, its groups come in pairs: gaps
    10.05, 2 x 16.62 and 2 x 23.01 m); the texts said one group. Counted from
    the recorded rule and from the rule recomputed from the built world."""
    counts = []
    for max_gap in (15.0, 20.0, 25.0):
        plan = _plan(
            building_counts={"residential": 100},
            cabinets=CabinetConfig(max_far_end_gap_m=max_gap),
        )
        recorded = sum(s.cabinet_rule == "far_end_ray" for s in plan.sites)
        rebuilt = _expected_cabinet(plan, plan.to_world(), max_gap=max_gap)
        assert recorded == sum(e[3] == "far" for e in rebuilt)
        counts.append(recorded)
    assert counts == [4, 2, 0]


# audit W1R2-09; coordinator's decision on W1R2 (both fields new in 2026.11)
@pytest.mark.parametrize(
    "clearance, max_gap", [(20.0, 20.0), (21.0, 20.0), (5.0, 4.0), (1.0, 1.0)]
)
def test_an_end_clearance_not_below_the_far_end_gap_is_rejected(
    clearance: float, max_gap: float
) -> None:
    """A cabinet on the centroid ray stands at least ``end_clearance_m`` from
    its group, so with ``end_clearance_m >= max_far_end_gap_m`` every group
    took the far-end ray, silently (base: 0 / 1 / 3 / 3 far-end groups of
    the 50-house grid at 1 / 10 / 19 / 21 m)."""
    with pytest.raises(ValueError, match=r"end_clearance_m.*max_far_end_gap_m"):
        CabinetConfig(end_clearance_m=clearance, max_far_end_gap_m=max_gap)


# audit W1R2-09; coordinator's decision on W1R2
def test_a_drawn_end_clearance_not_below_the_far_end_gap_is_rejected() -> None:
    """Distributions are checked where they are drawn; just below the
    threshold the network builds."""
    for cabinets in (
        CabinetConfig(end_clearance_m=Discrete(values=[25.0])),
        CabinetConfig(max_far_end_gap_m=Discrete(values=[0.5])),
        CabinetConfig(
            end_clearance_m=Discrete(values=[3.0]),
            max_far_end_gap_m=Discrete(values=[3.0]),
        ),
    ):
        cfg = TnNetworkConfig(cabinets=cabinets)
        with pytest.raises(ValueError, match=r"end_clearance_m.*max_far_end_gap_m"):
            TnNetworkGenerator(cfg, seed=0).plan()
    plan = _plan(
        building_counts={"residential": 50},
        cabinets=CabinetConfig(end_clearance_m=Discrete(values=[19.0])),
    )
    assert [s.cabinet_rule for s in plan.sites if s.role == "cabinet"] == [
        "far_end_ray"
    ] * 3


# audit W1B-02; user decisions Q22, Q35
@pytest.mark.parametrize(
    "kw",
    [
        dict(building_counts={"small_industry": 30}),
        dict(building_counts={"residential": 20, "medium_industry": 10}),
        dict(
            building_counts={"residential": 30},
            cabinets=CabinetConfig(
                grounding=GroundingSystemSpec(
                    electrodes=[RingElectrodeSpec(radius_m=2.0, depth_m=0.6)]
                )
            ),
        ),
        dict(
            building_counts={"residential": 30},
            cabinets=CabinetConfig(
                grounding=GroundingSystemSpec(
                    electrodes=[
                        RingElectrodeSpec(radius_m=1.5, depth_m=0.6),
                        RodElectrodeSpec(length_m=1.5, offset_xy_m=(2.5, 0.0)),
                    ]
                ),
                end_clearance_m=2.0,
            ),
        ),
    ],
    ids=["small_industry-rod", "medium_industry-rods", "cabinet-ring-2m", "ring+rod"],
)
def test_the_cabinet_electrodes_keep_the_end_clearance_from_every_electrode(
    kw: dict,
) -> None:
    """The clearance counts every electrode of the group, rods included, and
    the plan extent of the cabinet's own grounding. Base ``6f63deb``: the
    cabinet rod stood on the rod of ``small_industry_19`` at (119.5, 0) m
    (0.000 m), a 2 m cabinet ring crossed the outer house's foundation.
    Here the cabinet's electrodes keep ``end_clearance_m`` from every other
    electrode of the world (2 mm sampling)."""
    plan = _plan(**kw)
    world = plan.to_world()
    clearance = _plan_clearance(kw)
    sites = {_root(e.name) for e in world.electrodes}
    for cab in (s for s in sites if s.startswith("kvs_")):
        gap = min(_site_gap(world, cab, other) for other in sites if other != cab)
        assert gap >= clearance - 1e-3, (cab, gap)


def _plan_clearance(kw: dict) -> float:
    """``end_clearance_m`` of a test configuration."""
    cabinets = kw.get("cabinets")
    return 1.0 if cabinets is None else float(cabinets.end_clearance_m)


# audit W1B-02; user decisions Q22, Q35
def test_a_cabinet_that_cannot_keep_the_clearance_raises_naming_the_sites() -> None:
    """Four houses, two cabinets: the far-end house of the first group stands
    at 5 degrees, a house of the second group at 6.5 degrees 0.8 m beyond
    where that group's cabinet would stand. The generator raises and names
    the cabinet, its far-end building and the site in the way."""
    positions = [
        (40.0, 0.0),
        (50.0 * math.cos(math.radians(5.0)), 50.0 * math.sin(math.radians(5.0))),
        (62.0 * math.cos(math.radians(6.5)), 62.0 * math.sin(math.radians(6.5))),
        (0.0, 60.0),
    ]
    cfg = TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=100.0),
        building_counts={"residential": 4},
        placement=ExplicitPlacement(positions=positions),
        cabinets=CabinetConfig(fixed_count=2),
    )
    with pytest.raises(ValueError, match=r"'kvs_0'.*'residential_1'.*'residential_2'"):
        TnNetworkGenerator(cfg, seed=0).plan()


# audit W1B-05; user decisions Q22, Q35
@pytest.mark.parametrize("value", [-5.0, -0.5, 0.0, float("nan"), float("inf")])
def test_a_non_positive_or_infinite_end_clearance_is_rejected(value: float) -> None:
    """Base ``6f63deb``: -5 m put both cabinets of the default network at the
    centres of the outer houses of the middle row, 0 m on a foundation wire,
    NaN failed late in the rod constructor."""
    with pytest.raises(ValueError, match="end_clearance_m"):
        CabinetConfig(end_clearance_m=value)


# audit W1B-05; user decisions Q22, Q35
def test_a_drawn_end_clearance_must_be_positive() -> None:
    """A distribution is checked where it is drawn."""
    cfg = TnNetworkConfig(
        cabinets=CabinetConfig(end_clearance_m=Discrete(values=[-1.0]))
    )
    with pytest.raises(ValueError, match="end_clearance_m.*-1"):
        TnNetworkGenerator(cfg, seed=0).plan()
    cfg = TnNetworkConfig(
        cabinets=CabinetConfig(end_clearance_m=Discrete(values=[2.0]))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = TnNetworkGenerator(cfg, seed=0).plan()
    assert [s.position_xy for s in plan.sites if s.role == "cabinet"] == [
        (-119.5, 0.0),
        (119.5, 0.0),
    ]


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
