"""Regression tests for review pass 9, group ``generators/`` (0.15.0).

Covered findings
---------------
* **F33** — ``building_counts`` is a *mapping*, so its key order is
  semantically void, yet up to 0.14.1 the per-type counts were drawn
  from the RNG in dict-insertion order. Two configs differing only in
  the order the keys were written produced different worlds for the
  same seed (seed 42: ``residential`` = 5 vs 30 houses), which breaks
  the "a seed pins a world" contract the AP1 Monte-Carlo study rests
  on. Sampling now follows the catalog order of ``building_types``.
* **F25** — footprint-driven foundations inherited the OMBR side
  lengths and orientation but not its centre, so for a non-symmetric
  outline the foundation ring was a *translated* copy of the oriented
  minimum bounding rectangle (3.49 m off for an L-shaped house,
  8.04 m for a thin L), leaving its own plot and overlapping the
  neighbour's footprint.
* **F42** — ``route_manhattan``'s ``escape_radius_m`` blind spot was
  applied inside the A* loop only; the trailing bridge from the last
  grid cell to ``end`` was tested against the raw obstacle list, so
  an ``end`` anchor inside a footprint raised ``RuntimeError`` for
  *every* escape radius — the release valve never covered the case it
  exists for.

Physical context: all three defects are purely geometric /
book-keeping. F25 and F42 move conductors (and therefore change the
solved potential distribution of an OSM-driven world); F33 changes
which world a seed denotes.
"""

from __future__ import annotations

import math

import pytest

from groundfield.generators import (
    BuildingTypeSpec,
    Discrete,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    TnNetworkConfig,
    TnNetworkGenerator,
)
from groundfield.generators.manhattan_routing import (
    ObstacleBox,
    route_manhattan,
    segment_intersects_box,
)
from groundfield.generators.tn_network import (
    _override_foundation_from_footprint,
)

# --- Fixtures shared by the F25 tests --------------------------------
# An L-shaped outline: area centroid (7.529, 8.529), OMBR centre
# (10.0, 11.0), OMBR 22 x 20 m at 90 deg -> 3.494 m displacement.
L_SHAPE = [
    (0.0, 0.0),
    (20.0, 0.0),
    (20.0, 8.0),
    (8.0, 8.0),
    (8.0, 22.0),
    (0.0, 22.0),
]
# A thin L: centroid (9.318, 9.318) vs OMBR centre (15, 15) -> 8.035 m.
THIN_L = [
    (0.0, 0.0),
    (30.0, 0.0),
    (30.0, 5.0),
    (5.0, 5.0),
    (5.0, 30.0),
    (0.0, 30.0),
]


def _footprint(polygon, osm_id=1):
    """Build a :class:`BuildingFootprint`, skipping without shapely."""
    pytest.importorskip("shapely")
    from groundfield.geo.footprint import BuildingFootprint

    return BuildingFootprint(polygon_xy_m=list(polygon), osm_id=osm_id)


# =====================================================================
# F33 — building_counts key order must not move the RNG draws
# =====================================================================


def _counts_config(pairs) -> TnNetworkConfig:
    """Config whose ``building_counts`` keys follow ``pairs`` order."""
    return TnNetworkConfig(
        building_counts={
            name: Discrete(values=list(values)) for name, values in pairs
        },
    )


ORDER_A = [("residential", [5, 30]), ("small_industry", [1, 9])]
ORDER_B = [("small_industry", [1, 9]), ("residential", [5, 30])]


def test_f33_building_counts_key_order_does_not_change_counts() -> None:
    """Same seed + same specs -> same counts, whatever the key order.

    Pre-fix (0.14.1): order A -> residential 5 / small_industry 9,
    order B -> residential 30 / small_industry 1.
    """
    resolved = []
    for pairs in (ORDER_A, ORDER_B):
        cfg = _counts_config(pairs)
        gen = TnNetworkGenerator(cfg, seed=42)
        pairs_out = gen._resolve_building_counts(cfg, gen.rng)
        resolved.append([(t.name, n) for t, n in pairs_out])
    assert resolved[0] == resolved[1]
    # The canonical pairing is the catalog order of ``building_types``:
    # the first draw belongs to ``residential`` (catalog index 0).
    assert resolved[0] == [("residential", 5), ("small_industry", 9)]


def test_f33_key_order_gives_bit_identical_world() -> None:
    """Two differently-ordered but equal configs build one world."""
    worlds = []
    for pairs in (ORDER_A, ORDER_B):
        cfg = _counts_config(pairs)
        worlds.append(TnNetworkGenerator(cfg, seed=42).build())
    a, b = worlds
    assert [e.name for e in a.electrodes] == [e.name for e in b.electrodes]
    assert len(a.electrodes) == len(b.electrodes)
    # Full structural equality: names, geometry, conductors, source.
    assert a.model_dump_json() == b.model_dump_json()


def test_f33_counts_still_follow_catalog_order() -> None:
    """A reordered *catalog* still drives the output order (unchanged)."""
    cfg = _counts_config(ORDER_B)
    gen = TnNetworkGenerator(cfg, seed=7)
    names = [t.name for t, _ in gen._resolve_building_counts(cfg, gen.rng)]
    catalog = [t.name for t in cfg.building_types]
    assert names == [n for n in catalog if n in cfg.building_counts]


def test_f33_unknown_type_error_is_order_independent() -> None:
    """The KeyError names the same offender for either key order."""
    messages = []
    orders = (
        ("aaa_missing", "zzz_missing"),
        ("zzz_missing", "aaa_missing"),
    )
    for keys in orders:
        cfg = TnNetworkConfig(building_counts={k: 1 for k in keys})
        gen = TnNetworkGenerator(cfg, seed=0)
        with pytest.raises(KeyError) as excinfo:
            gen._resolve_building_counts(cfg, gen.rng)
        messages.append(str(excinfo.value))
    assert messages[0] == messages[1]
    assert "aaa_missing" in messages[0]


# =====================================================================
# F25 — the OMBR centre must survive into the foundation spec
# =====================================================================


def _foundation_grounding(offset=(0.0, 0.0)) -> GroundingSystemSpec:
    return GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(size_m=10.0, n_x=2, n_y=2,
                                    offset_xy_m=offset),
        ],
    )


@pytest.mark.parametrize(
    ("polygon", "expected_centre", "expected_shift"),
    [
        (L_SHAPE, (10.0, 11.0), 3.4939393893923527),
        (THIN_L, (15.0, 15.0), 8.035304331665312),
    ],
)
def test_f25_offset_recentres_on_ombr(
    polygon, expected_centre, expected_shift,
) -> None:
    """``offset_xy_m`` compensates centroid -> OMBR-centre offset.

    Pre-fix the offset stayed at ``(0, 0)``, i.e. the rectangle was
    built on the polygon centroid and thus displaced by
    ``expected_shift`` metres.
    """
    fp = _footprint(polygon)
    centre, (dx, dy), angle = fp.oriented_bounding_rectangle()
    centroid = fp.centroid_xy_m()
    assert centre == pytest.approx(expected_centre)

    spec = _override_foundation_from_footprint(_foundation_grounding(), fp)
    foundation = spec.electrodes[0]
    assert foundation.size_xy_m == pytest.approx((dx, dy))
    assert foundation.orientation_deg == pytest.approx(angle)
    # The electrode is placed at ``site_xy + offset_xy_m`` and the
    # site is the centroid -> the sum must be the OMBR centre.
    placed = (
        centroid[0] + foundation.offset_xy_m[0],
        centroid[1] + foundation.offset_xy_m[1],
    )
    assert placed == pytest.approx(expected_centre)
    assert math.hypot(*foundation.offset_xy_m) == pytest.approx(
        expected_shift
    )


def test_f25_user_offset_is_preserved_not_overwritten() -> None:
    """A user ``offset_xy_m`` composes with the recentring shift."""
    fp = _footprint(L_SHAPE)
    centroid = fp.centroid_xy_m()
    user = (1.5, -2.5)
    spec = _override_foundation_from_footprint(
        _foundation_grounding(offset=user), fp,
    )
    off = spec.electrodes[0].offset_xy_m
    assert (centroid[0] + off[0], centroid[1] + off[1]) == pytest.approx(
        (10.0 + user[0], 11.0 + user[1])
    )


def test_f25_explicit_site_xy_is_honoured() -> None:
    """Passing the site position removes the centroid assumption."""
    fp = _footprint(L_SHAPE)
    spec = _override_foundation_from_footprint(
        _foundation_grounding(), fp, (10.0, 11.0),
    )
    assert spec.electrodes[0].offset_xy_m == pytest.approx((0.0, 0.0))


def test_f25_symmetric_footprint_needs_no_shift() -> None:
    """For a rectangle, centroid == OMBR centre -> zero offset."""
    fp = _footprint([(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)])
    spec = _override_foundation_from_footprint(_foundation_grounding(), fp)
    assert spec.electrodes[0].offset_xy_m == pytest.approx(
        (0.0, 0.0), abs=1e-9
    )


def test_f25_built_world_foundation_sits_on_the_ombr() -> None:
    """End-to-end: the built strips span the OMBR, not a shifted copy.

    Pre-fix the bounding box of the foundation strips was centred on
    the polygon centroid (7.529, 8.529) instead of (10.0, 11.0).
    """
    pytest.importorskip("shapely")
    from groundfield.geo.placement import OsmBuildingPlacement

    fp = _footprint(L_SHAPE)
    catalog = [
        BuildingTypeSpec(
            name="residential", grounding=_foundation_grounding(),
        ),
    ]
    cfg = TnNetworkConfig(
        building_types=catalog,
        building_counts={"residential": 1},
        placement=OsmBuildingPlacement(footprints=[fp]),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    xs: list[float] = []
    ys: list[float] = []
    for e in world.electrodes:
        if not e.name.startswith("residential_0_foundation_0"):
            continue
        for point in (e.start, e.end):
            xs.append(point[0])
            ys.append(point[1])
    assert xs, "no foundation electrode was built"
    centre = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))
    assert centre == pytest.approx((10.0, 11.0), abs=1e-9)
    assert (max(xs) - min(xs), max(ys) - min(ys)) == pytest.approx(
        (20.0, 22.0), abs=1e-9
    )


# =====================================================================
# F42 — escape_radius_m must reach the trailing bridge
# =====================================================================

# ``end`` = (53, 3) lies inside the obstacle; the last grid cell is
# (50, 0), also inside it after the 0.5 m clearance inflation.
BRIDGE_OBSTACLE = ObstacleBox(x_min=40.0, y_min=-10.0, x_max=60.0, y_max=10.0)
BRIDGE_START = (0.0, 0.0)
BRIDGE_END = (53.0, 3.0)


def _assert_manhattan(waypoints, start, end) -> None:
    assert waypoints[0] == pytest.approx(start)
    assert waypoints[-1] == pytest.approx(end)
    for p, q in zip(waypoints, waypoints[1:]):
        assert math.isclose(p[0], q[0]) or math.isclose(p[1], q[1])


@pytest.mark.parametrize("escape_radius_m", [None, 50.0, 200.0])
def test_f42_end_anchor_inside_obstacle_routes(escape_radius_m) -> None:
    """An ``end`` inside a footprint routes instead of raising.

    Pre-fix all three cases raised
    ``RuntimeError: cannot bridge final cell (50.0, 0.0) to end
    (53.0, 3.0) ...`` — raising ``escape_radius_m`` had no effect.
    """
    waypoints = route_manhattan(
        BRIDGE_START,
        BRIDGE_END,
        [BRIDGE_OBSTACLE],
        grid_size=10.0,
        clearance_m=0.5,
        escape_radius_m=escape_radius_m,
    )
    _assert_manhattan(waypoints, BRIDGE_START, BRIDGE_END)


def test_f42_large_escape_radius_shortens_the_route() -> None:
    """With the anchor exempt, no detour around its own building."""
    waypoints = route_manhattan(
        BRIDGE_START,
        BRIDGE_END,
        [BRIDGE_OBSTACLE],
        grid_size=10.0,
        clearance_m=0.5,
        escape_radius_m=200.0,
    )
    assert waypoints == [(0.0, 0.0), (53.0, 0.0), (53.0, 3.0)]


def test_f42_start_anchor_inside_obstacle_routes() -> None:
    """The symmetric case (anchor inside the box) also routes."""
    waypoints = route_manhattan(
        BRIDGE_END,
        BRIDGE_START,
        [BRIDGE_OBSTACLE],
        grid_size=10.0,
        clearance_m=0.5,
        escape_radius_m=50.0,
    )
    _assert_manhattan(waypoints, BRIDGE_END, BRIDGE_START)


def test_f42_zero_escape_radius_stays_strict() -> None:
    """``escape_radius_m = 0`` keeps the pre-v0.7 hard failure."""
    with pytest.raises(RuntimeError):
        route_manhattan(
            BRIDGE_START,
            BRIDGE_END,
            [BRIDGE_OBSTACLE],
            grid_size=10.0,
            clearance_m=0.5,
            escape_radius_m=0.0,
        )


def test_f42_obstacle_away_from_anchors_is_still_enforced() -> None:
    """The exemption is local: a mid-route building still blocks.

    The obstacle contains neither anchor, so the route must detour
    around it rather than cut through.
    """
    obstacle = ObstacleBox(x_min=20.0, y_min=-10.0, x_max=30.0, y_max=10.0)
    waypoints = route_manhattan(
        BRIDGE_START,
        BRIDGE_END,
        [obstacle],
        grid_size=10.0,
        clearance_m=0.5,
    )
    _assert_manhattan(waypoints, BRIDGE_START, BRIDGE_END)
    inflated = obstacle.inflated(0.5)
    for p, q in zip(waypoints, waypoints[1:]):
        assert not segment_intersects_box(p, q, inflated)


def test_f42_bridge_still_raises_outside_the_escape_disk() -> None:
    """A box crossing both bridge corners outside the disk raises.

    Guards against the fix degenerating into "the bridge ignores
    obstacles". The obstacle contains neither anchor and the escape
    radius (0.5 m) is far smaller than the 3 m bridge legs, so both
    candidate corners must be rejected.
    """
    obstacle = ObstacleBox(x_min=51.0, y_min=-5.0, x_max=52.0, y_max=5.0)
    with pytest.raises(RuntimeError, match="cannot bridge final cell"):
        route_manhattan(
            BRIDGE_START,
            BRIDGE_END,
            [obstacle],
            grid_size=10.0,
            clearance_m=0.0,
            escape_radius_m=0.5,
        )


def test_f42_bridge_error_message_mentions_escape_radius() -> None:
    """The bridge error must name the knob that can fix it."""
    obstacle = ObstacleBox(x_min=51.0, y_min=-5.0, x_max=52.0, y_max=5.0)
    with pytest.raises(RuntimeError, match="escape_radius_m"):
        route_manhattan(
            BRIDGE_START,
            BRIDGE_END,
            [obstacle],
            grid_size=10.0,
            clearance_m=0.0,
            escape_radius_m=0.5,
        )


def test_f42_route_is_deterministic() -> None:
    """Repeated calls return the identical polyline (no RNG, no sets)."""
    kwargs = dict(grid_size=10.0, clearance_m=0.5, escape_radius_m=25.0)
    first = route_manhattan(
        BRIDGE_START, BRIDGE_END, [BRIDGE_OBSTACLE], **kwargs
    )
    second = route_manhattan(
        BRIDGE_START, BRIDGE_END, [BRIDGE_OBSTACLE], **kwargs
    )
    assert first == second
