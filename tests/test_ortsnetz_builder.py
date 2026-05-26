"""Tests for the imperative TN-Ortsnetz layout builder.

Covers
------
* :func:`route_manhattan` — basic axis-aligned routing,
  obstacle avoidance, unreachable failure, min-segment-length
  preservation after collinear merging.
* :func:`segment_intersects_box` and the merge helper.
* :class:`OrtsnetzLayout` end-to-end pipeline: OSM-style
  footprint ingest, substation placement, KVS placement, PEN
  cable routing through a sparse obstacle field, building →
  cable connection, materialisation to :class:`World`, plotting.

All tests build their own synthetic footprints (no live OSM
fetch) so the suite stays offline-friendly.
"""

from __future__ import annotations

import math
import warnings

import pytest

import groundfield as gf
from groundfield.generators import (
    AuxiliaryElectrodePlacement,
    BuildingConnection,
    KvsPlacement,
    ObstacleBox,
    OrtsnetzLayout,
    PenCable,
    VoltageProbePlacement,
    merge_collinear_waypoints,
    route_manhattan,
    segment_intersects_box,
)
from groundfield.geo.footprint import BuildingFootprint


# ---------------------------------------------------------------------
# Synthetic footprints (offline-friendly)
# ---------------------------------------------------------------------


def _rect_footprint(
    cx: float, cy: float, dx: float = 10.0, dy: float = 10.0,
    osm_id: int = 0,
) -> BuildingFootprint:
    """Axis-aligned rectangle as a BuildingFootprint."""
    return BuildingFootprint(
        polygon_xy_m=[
            (cx - dx / 2.0, cy - dy / 2.0),
            (cx + dx / 2.0, cy - dy / 2.0),
            (cx + dx / 2.0, cy + dy / 2.0),
            (cx - dx / 2.0, cy + dy / 2.0),
        ],
        building_use="residential",
        osm_id=osm_id,
    )


# ---------------------------------------------------------------------
# Manhattan routing primitives
# ---------------------------------------------------------------------


def test_segment_intersects_box_axis_aligned() -> None:
    box = ObstacleBox(x_min=10.0, y_min=10.0, x_max=20.0, y_max=20.0)
    # Horizontal segment cutting through the middle: blocked.
    assert segment_intersects_box((5.0, 15.0), (25.0, 15.0), box)
    # Horizontal segment passing well above: clear.
    assert not segment_intersects_box((5.0, 30.0), (25.0, 30.0), box)
    # Vertical segment grazing the right edge: not counted as
    # intersecting (boundary clearance).
    assert not segment_intersects_box((20.0, 5.0), (20.0, 25.0), box)


def test_segment_intersects_box_rejects_diagonal() -> None:
    box = ObstacleBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
    with pytest.raises(ValueError, match="axis-aligned"):
        segment_intersects_box((0.0, 0.0), (10.0, 10.0), box)


def test_merge_collinear_removes_intermediate_collinear() -> None:
    wp = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (20.0, 10.0)]
    out = merge_collinear_waypoints(wp)
    assert out == [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0)]


def test_route_manhattan_direct_line_no_obstacles() -> None:
    out = route_manhattan(
        start=(0.0, 0.0), end=(50.0, 0.0),
        obstacles=[], grid_size=10.0,
    )
    assert out[0] == (0.0, 0.0)
    assert out[-1] == (50.0, 0.0)
    # No corners needed on a clear line.
    assert all(math.isclose(y, 0.0) for _x, y in out)


def test_route_manhattan_segments_are_axis_aligned() -> None:
    out = route_manhattan(
        start=(0.0, 0.0), end=(50.0, 30.0),
        obstacles=[], grid_size=10.0,
    )
    for (x0, y0), (x1, y1) in zip(out[:-1], out[1:]):
        assert math.isclose(x0, x1) or math.isclose(y0, y1), (
            f"diagonal segment {(x0, y0)} -> {(x1, y1)}"
        )


def test_route_manhattan_detours_around_obstacle() -> None:
    # Single obstacle directly on the straight-line path.
    blocking = ObstacleBox(
        x_min=15.0, y_min=-5.0, x_max=35.0, y_max=5.0,
    )
    out = route_manhattan(
        start=(0.0, 0.0), end=(50.0, 0.0),
        obstacles=[blocking], grid_size=5.0, clearance_m=0.5,
    )
    # The path must contain at least one corner (it cannot be a
    # straight line through the obstacle).
    assert len(out) >= 3
    # No segment touches the inflated box interior.
    for p0, p1 in zip(out[:-1], out[1:]):
        assert not segment_intersects_box(p0, p1, blocking.inflated(0.5))


def test_route_manhattan_raises_when_enclosed() -> None:
    # Substation surrounded by a ring of obstacles too tight to escape.
    wall = [
        ObstacleBox(x_min=-5.0, y_min=5.0, x_max=5.0, y_max=15.0),
        ObstacleBox(x_min=-5.0, y_min=-15.0, x_max=5.0, y_max=-5.0),
        ObstacleBox(x_min=5.0, y_min=-5.0, x_max=15.0, y_max=5.0),
        ObstacleBox(x_min=-15.0, y_min=-5.0, x_max=-5.0, y_max=5.0),
    ]
    with pytest.raises(RuntimeError):
        route_manhattan(
            start=(0.0, 0.0), end=(50.0, 0.0),
            obstacles=wall, grid_size=2.0, clearance_m=1.0,
            max_iterations=20_000,
        )


def test_route_manhattan_escape_radius_unblocks_anchor() -> None:
    """With ``escape_radius_m=0`` the substation enclosed by tight
    obstacles is unroutable; with the default escape radius the
    neighbours become reachable so the route succeeds.

    Geometry: four small 1 m x 1 m obstacles centred on the *four
    immediate cardinal neighbours* of the substation cell (grid
    size = 2 m so the neighbours are at (±2, 0) / (0, ±2)). The
    obstacles only block their own cell -- the next ring outward
    is free, so the path exists as soon as A* can take a first
    step.
    """
    blocking_neighbours = [
        # East / West / North / South of the substation cell.
        ObstacleBox(x_min=+1.5, y_min=-0.5, x_max=+2.5, y_max=+0.5),
        ObstacleBox(x_min=-2.5, y_min=-0.5, x_max=-1.5, y_max=+0.5),
        ObstacleBox(x_min=-0.5, y_min=+1.5, x_max=+0.5, y_max=+2.5),
        ObstacleBox(x_min=-0.5, y_min=-2.5, x_max=+0.5, y_max=-1.5),
    ]
    # Strict mode (escape_radius_m=0) -- every immediate neighbour
    # is blocked, A* cannot move.
    with pytest.raises(RuntimeError):
        route_manhattan(
            start=(0.0, 0.0), end=(50.0, 0.0),
            obstacles=blocking_neighbours, grid_size=2.0,
            clearance_m=0.2, escape_radius_m=0.0,
        )
    # Default escape radius (= grid_size = 2 m) unblocks the four
    # cardinal neighbours; the next ring outward is free, so the
    # route succeeds.
    out = route_manhattan(
        start=(0.0, 0.0), end=(50.0, 0.0),
        obstacles=blocking_neighbours, grid_size=2.0,
        clearance_m=0.2,
    )
    assert out[0] == (0.0, 0.0)
    assert out[-1] == (50.0, 0.0)


def test_route_manhattan_error_message_names_offending_anchor() -> None:
    """The RuntimeError mentions which endpoint is enclosed so the
    user gets an actionable hint."""
    wall = [
        ObstacleBox(x_min=-5.0, y_min=5.0, x_max=5.0, y_max=15.0),
        ObstacleBox(x_min=-5.0, y_min=-15.0, x_max=5.0, y_max=-5.0),
        ObstacleBox(x_min=5.0, y_min=-5.0, x_max=15.0, y_max=5.0),
        ObstacleBox(x_min=-15.0, y_min=-5.0, x_max=-5.0, y_max=5.0),
    ]
    with pytest.raises(RuntimeError, match=r"start|end|both"):
        route_manhattan(
            start=(0.0, 0.0), end=(50.0, 0.0),
            obstacles=wall, grid_size=2.0, clearance_m=1.0,
            escape_radius_m=0.0, max_iterations=20_000,
        )


def test_add_pen_cable_forwards_escape_radius() -> None:
    """``OrtsnetzLayout.add_pen_cable`` forwards ``escape_radius_m``
    so users can relax routing when an anchor is wedged into a
    dense foundation cluster.

    Geometry: four 1 m x 1 m footprints centred on the four
    immediate cardinal neighbours of the substation cell (grid
    size = 2 m). With ``escape_radius_m=0`` the substation cell
    has no free neighbour; with a larger radius it does.
    """
    fps = [
        _rect_footprint(+2.0,  0.0, dx=1.0, dy=1.0, osm_id=1),
        _rect_footprint(-2.0,  0.0, dx=1.0, dy=1.0, osm_id=2),
        _rect_footprint( 0.0, +2.0, dx=1.0, dy=1.0, osm_id=3),
        _rect_footprint( 0.0, -2.0, dx=1.0, dy=1.0, osm_id=4),
    ]
    layout = OrtsnetzLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0),
        obstacle_clearance_m=0.2,
    )
    # With escape_radius_m=0 the immediate neighbours are blocked.
    with pytest.raises(RuntimeError):
        layout.add_pen_cable(
            start="substation", end_xy=(50.0, 0.0),
            min_segment_length_m=2.0, escape_radius_m=0.0,
        )
    # With a generous escape radius (>= one cell) the route
    # succeeds.
    cable = layout.add_pen_cable(
        start="substation", end_xy=(50.0, 0.0),
        min_segment_length_m=2.0, escape_radius_m=4.0,
    )
    assert cable.waypoints[0] == (0.0, 0.0)
    assert cable.waypoints[-1] == (50.0, 0.0)


def test_route_manhattan_min_segment_length_after_merge() -> None:
    out = route_manhattan(
        start=(0.0, 0.0), end=(120.0, 60.0),
        obstacles=[], grid_size=10.0,
    )
    # After merging, every interior leg must be at least the grid
    # size (the trailing bridge to ``end`` is exempt because it
    # only fires when ``end`` doesn't align with the grid).
    for (x0, y0), (x1, y1) in zip(out[:-1], out[1:]):
        length = abs(x1 - x0) + abs(y1 - y0)
        assert length >= 10.0 - 1e-9


# ---------------------------------------------------------------------
# OrtsnetzLayout: KVS placement, naming
# ---------------------------------------------------------------------


def test_add_kvs_autogenerates_unique_names() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    n0 = layout.add_kvs((100.0, 0.0))
    n1 = layout.add_kvs((200.0, 0.0))
    assert n0 != n1
    assert [k.name for k in layout.kvs_placements] == [n0, n1]


def test_add_kvs_rejects_duplicate_names() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_kvs((100.0, 0.0), name="my_kvs")
    with pytest.raises(ValueError, match="already taken"):
        layout.add_kvs((200.0, 0.0), name="my_kvs")


def test_add_kvs_rejects_substation_name() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
        substation_name="trafo",
    )
    with pytest.raises(ValueError, match="substation"):
        layout.add_kvs((100.0, 0.0), name="trafo")


# ---------------------------------------------------------------------
# OrtsnetzLayout: PEN cable routing
# ---------------------------------------------------------------------


def test_add_pen_cable_substation_to_kvs() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_kvs((100.0, 0.0), name="kvs_a")
    cable = layout.add_pen_cable(start="substation", end="kvs_a")
    assert cable.start_anchor == "substation"
    assert cable.end_anchor == "kvs_a"
    assert cable.waypoints[0] == (0.0, 0.0)
    assert cable.waypoints[-1] == (100.0, 0.0)
    # No obstacles -> the path is a single straight line.
    assert len(cable.waypoints) == 2


def test_add_pen_cable_requires_one_endpoint_spec() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="exactly one"):
        layout.add_pen_cable(start="substation")
    layout.add_kvs((100.0, 0.0), name="kvs_a")
    with pytest.raises(ValueError, match="exactly one"):
        layout.add_pen_cable(
            start="substation",
            end="kvs_a",
            end_xy=(50.0, 0.0),
        )


def test_add_pen_cable_routes_around_house() -> None:
    house = _rect_footprint(50.0, 0.0, dx=10.0, dy=10.0)
    layout = OrtsnetzLayout.from_footprints(
        [house], substation_xy=(0.0, 0.0),
    )
    cable = layout.add_pen_cable(
        start="substation",
        end_xy=(100.0, 0.0),
        min_segment_length_m=5.0,
    )
    # The straight x-axis path crosses the foundation; the router
    # must produce at least one corner.
    assert len(cable.waypoints) >= 3
    # No segment crosses the inflated obstacle.
    obstacle_box = layout._obstacle_boxes()[0].inflated(
        layout.obstacle_clearance_m,
    )
    for p0, p1 in zip(cable.waypoints[:-1], cable.waypoints[1:]):
        assert not segment_intersects_box(p0, p1, obstacle_box)


def test_add_pen_cable_rejects_duplicate_name() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(50.0, 0.0),
                        name="bus_main")
    with pytest.raises(ValueError, match="taken"):
        layout.add_pen_cable(
            start="substation",
            end_xy=(60.0, 0.0),
            name="bus_main",
        )


# ---------------------------------------------------------------------
# OrtsnetzLayout: building → cable connection
# ---------------------------------------------------------------------


def _simple_bus_layout(n_houses: int = 6) -> OrtsnetzLayout:
    """A bus topology with houses lined up north and south of the
    substation-to-end cable on y = 0.
    """
    footprints: list[BuildingFootprint] = []
    for i in range(n_houses):
        cx = 20.0 + 25.0 * i
        cy = 18.0 if i % 2 == 0 else -18.0
        footprints.append(_rect_footprint(cx, cy, dx=10.0, dy=10.0, osm_id=i))
    layout = OrtsnetzLayout.from_footprints(
        footprints, substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(180.0, 0.0))
    return layout


def test_connect_buildings_connects_every_house_for_single_cable() -> None:
    layout = _simple_bus_layout(n_houses=6)
    new = layout.connect_buildings()
    assert len(new) == 6
    assert {c.house_idx for c in new} == set(range(6))
    # Every connection targets the only cable in the layout.
    assert {c.cable_name for c in new} == {"pen_0"}
    # Tap point of every house has the *same* y as the cable.
    for c in new:
        assert math.isclose(c.tap_point_xy[1], 0.0, abs_tol=1e-6)


def test_connect_buildings_picks_closest_cable_when_two_present() -> None:
    # Two parallel buses, the southern one being much closer to
    # a south-side house.
    fp_south = _rect_footprint(60.0, -25.0, osm_id=0)
    fp_north = _rect_footprint(60.0, 60.0, osm_id=1)
    layout = OrtsnetzLayout.from_footprints(
        [fp_south, fp_north], substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(
        start="substation", end_xy=(120.0, 0.0), name="north_bus",
    )
    layout.add_kvs((0.0, 80.0), name="kvs_north")
    layout.add_pen_cable(
        start="kvs_north", end_xy=(120.0, 80.0), name="far_bus",
    )
    layout.connect_buildings()
    cable_map = {c.house_idx: c.cable_name for c in layout.connections}
    # South house ends up on the north_bus (at y=0).
    assert cable_map[0] == "north_bus"
    # North house ends up on the far_bus (at y=80) — it's strictly
    # closer than the one at y=0.
    assert cable_map[1] == "far_bus"


def test_connect_buildings_skips_unreachable_and_warns() -> None:
    # Sandwich a house between two other houses so its only
    # access lanes to the cable cross those neighbours.
    target = _rect_footprint(50.0, 30.0, dx=8.0, dy=8.0, osm_id=0)
    blocker_n = _rect_footprint(50.0, 50.0, dx=80.0, dy=8.0, osm_id=1)
    blocker_s = _rect_footprint(50.0, 10.0, dx=80.0, dy=8.0, osm_id=2)
    layout = OrtsnetzLayout.from_footprints(
        [target, blocker_n, blocker_s],
        substation_xy=(0.0, 0.0),
        obstacle_clearance_m=1.0,
    )
    layout.add_pen_cable(
        start="substation", end_xy=(120.0, 0.0),
        min_segment_length_m=5.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new = layout.connect_buildings()
    # The target house is fully boxed in -> warning + skipped.
    assert any(c.house_idx == 0 for c in new) is False
    assert any("cannot be tapped" in str(w.message) for w in caught)


def test_connect_buildings_requires_existing_cable() -> None:
    fp = _rect_footprint(20.0, 18.0, osm_id=0)
    layout = OrtsnetzLayout.from_footprints(
        [fp], substation_xy=(0.0, 0.0),
    )
    with pytest.raises(RuntimeError, match="no PEN cables"):
        layout.connect_buildings()


# ---------------------------------------------------------------------
# OrtsnetzLayout: materialisation
# ---------------------------------------------------------------------


def test_to_world_emits_expected_anchors_and_conductors() -> None:
    layout = _simple_bus_layout(n_houses=4)
    layout.connect_buildings()
    world = layout.to_world(seed=0)

    # Substation cluster anchor + KVS rod + 4 house foundations + PEN nodes.
    electrode_names = [e.name for e in world.electrodes]
    assert "substation_ring_0" in electrode_names
    assert any(n.startswith("house_000_foundation_0") for n in electrode_names)
    # Service drops -- one per house.
    service_conductors = [
        c for c in world.conductors if c.name.startswith("service_")
    ]
    assert len(service_conductors) == 4
    # The current source attaches to the substation cluster.
    assert len(world.sources) == 1
    assert world.sources[0].attached_to == "substation_ring_0"


def test_to_world_runs_without_source_when_disabled() -> None:
    layout = _simple_bus_layout(n_houses=2)
    layout.connect_buildings()
    world = layout.to_world(include_source=False, seed=0)
    assert len(world.sources) == 0


def test_pen_cable_length_matches_manhattan_distance() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    cable = layout.add_pen_cable(
        start="substation", end_xy=(50.0, 30.0),
    )
    # On an obstacle-free corridor the Manhattan length equals
    # the Manhattan distance between the endpoints.
    assert math.isclose(cable.total_length_m(), 50.0 + 30.0, abs_tol=1e-6)


# ---------------------------------------------------------------------
# GPS / lat-lon convenience helpers
# ---------------------------------------------------------------------


def test_layout_without_frame_origin_rejects_lat_lon() -> None:
    """A layout without ``frame_origin_lat_lon`` cannot consume
    ``position_lat_lon`` / ``end_lat_lon`` arguments."""
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    with pytest.raises(RuntimeError, match="frame_origin_lat_lon"):
        layout.add_kvs(position_lat_lon=(51.9136, 10.8432))
    with pytest.raises(RuntimeError, match="frame_origin_lat_lon"):
        layout.add_pen_cable(
            start="substation",
            end_lat_lon=(51.9140, 10.8460),
        )


def test_layout_with_frame_origin_accepts_lat_lon_anchors() -> None:
    """When ``frame_origin_lat_lon`` is set, the layout projects
    lat / lon arguments through it automatically."""
    origin = (51.9136, 10.8432)
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
        frame_origin_lat_lon=origin,
    )
    # KVS placed ~50 m east of the origin → x ≈ +50, y ≈ 0.
    delta_lon = 50.0 / (111_320.0 * math.cos(math.radians(origin[0])))
    kvs_name = layout.add_kvs(
        position_lat_lon=(origin[0], origin[1] + delta_lon),
    )
    placement = next(k for k in layout.kvs_placements if k.name == kvs_name)
    assert math.isclose(placement.position_xy[0], 50.0, abs_tol=1.0)
    assert abs(placement.position_xy[1]) < 1.0


def test_add_kvs_rejects_both_position_kinds() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
        frame_origin_lat_lon=(51.9136, 10.8432),
    )
    with pytest.raises(ValueError, match="exactly one"):
        layout.add_kvs(
            position_xy=(50.0, 0.0),
            position_lat_lon=(51.9136, 10.8432),
        )
    with pytest.raises(ValueError, match="exactly one"):
        layout.add_kvs()  # neither provided


def test_add_pen_cable_lat_lon_end_projects_through_frame() -> None:
    origin = (51.9136, 10.8432)
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
        frame_origin_lat_lon=origin,
    )
    delta_lon = 80.0 / (111_320.0 * math.cos(math.radians(origin[0])))
    cable = layout.add_pen_cable(
        start="substation",
        end_lat_lon=(origin[0], origin[1] + delta_lon),
    )
    assert math.isclose(cable.waypoints[-1][0], 80.0, abs_tol=1.0)
    assert abs(cable.waypoints[-1][1]) < 1.0


def test_lat_lon_round_trip_via_layout_projector() -> None:
    """``lat_lon_to_xy`` ∘ ``xy_to_lat_lon`` is the identity (within
    a millimetre)."""
    origin = (51.9136, 10.8432)
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
        frame_origin_lat_lon=origin,
    )
    test_pts = [(51.9140, 10.8460), (51.9100, 10.8400), (51.9200, 10.8500)]
    for lat, lon in test_pts:
        x, y = layout.lat_lon_to_xy(lat, lon)
        lat_back, lon_back = layout.xy_to_lat_lon(x, y)
        assert math.isclose(lat, lat_back, abs_tol=1e-7)
        assert math.isclose(lon, lon_back, abs_tol=1e-7)


# ---------------------------------------------------------------------
# Deterministic foundation-electrode penetration mask
# ---------------------------------------------------------------------


def _layout_with_osm_ids(n: int = 12) -> OrtsnetzLayout:
    """Synthetic layout with distinct, stable osm_id values."""
    footprints = [
        _rect_footprint(20.0 + 25.0 * i, 18.0 if i % 2 == 0 else -18.0,
                        osm_id=1000 + i)
        for i in range(n)
    ]
    return OrtsnetzLayout.from_footprints(
        footprints, substation_xy=(0.0, 0.0),
    )


def test_foundation_mask_reproducible_across_calls() -> None:
    layout = _layout_with_osm_ids(n=20)
    m1 = layout.foundation_mask(0.30)
    m2 = layout.foundation_mask(0.30)
    assert m1 == m2


def test_foundation_mask_zero_and_one_extremes() -> None:
    layout = _layout_with_osm_ids(n=20)
    assert layout.foundation_mask(0.0) == [False] * 20
    assert layout.foundation_mask(1.0) == [True] * 20


def test_foundation_mask_is_nested_in_penetration() -> None:
    """A house with a foundation at p1 also has one at p2 > p1."""
    layout = _layout_with_osm_ids(n=20)
    for p1, p2 in [(0.10, 0.25), (0.25, 0.50), (0.50, 0.90)]:
        m1 = layout.foundation_mask(p1)
        m2 = layout.foundation_mask(p2)
        for b1, b2 in zip(m1, m2):
            # b1 implies b2
            assert (not b1) or b2


def test_foundation_mask_rough_fraction_matches_penetration() -> None:
    """On a larger sample the realised fraction tracks p within a
    few percent — the assignment is uniform, not adversarial."""
    footprints = [
        _rect_footprint(0.0, 5.0 * i, osm_id=10_000 + i)
        for i in range(200)
    ]
    layout = OrtsnetzLayout.from_footprints(
        footprints, substation_xy=(0.0, 0.0),
    )
    for p in (0.25, 0.5, 0.75):
        mask = layout.foundation_mask(p)
        realised = sum(mask) / len(mask)
        assert abs(realised - p) < 0.10  # generous bound


def test_foundation_mask_salt_shifts_assignment() -> None:
    """Two different ``salt`` values produce independent masks at
    the same penetration."""
    layout = _layout_with_osm_ids(n=20)
    m_a = layout.foundation_mask(0.40, salt=0)
    m_b = layout.foundation_mask(0.40, salt=1)
    # They must differ in at least one house (Monte-Carlo
    # realisation changed) but each one is still self-consistent.
    assert m_a != m_b
    assert m_a == layout.foundation_mask(0.40, salt=0)
    assert m_b == layout.foundation_mask(0.40, salt=1)


def test_foundation_mask_rejects_out_of_range() -> None:
    layout = _layout_with_osm_ids(n=4)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        layout.foundation_mask(-0.01)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        layout.foundation_mask(1.01)


def test_foundation_mask_falls_back_to_index_without_osm_id() -> None:
    """When ``osm_id`` is ``None`` the mask uses the footprint
    index, so the assignment is still deterministic for synthetic
    footprints."""
    footprints = [
        _rect_footprint(20.0 + 25.0 * i, 0.0)  # osm_id default = 0
        for i in range(10)
    ]
    # Wipe osm_id so the fallback path is exercised.
    footprints = [
        BuildingFootprint(polygon_xy_m=fp.polygon_xy_m,
                          building_use="residential")
        for fp in footprints
    ]
    layout = OrtsnetzLayout.from_footprints(
        footprints, substation_xy=(0.0, 0.0),
    )
    m1 = layout.foundation_mask(0.40)
    m2 = layout.foundation_mask(0.40)
    assert m1 == m2


def test_plot_accepts_foundation_mask() -> None:
    """``plot(foundation_mask=...)`` colours houses by the mask
    without raising on the Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = _layout_with_osm_ids(n=10)
    mask = layout.foundation_mask(0.40)
    ax = layout.plot(foundation_mask=mask)
    plt.close(ax.figure)


def test_plot_rejects_mismatched_foundation_mask_length() -> None:
    import matplotlib
    matplotlib.use("Agg")

    layout = _layout_with_osm_ids(n=10)
    with pytest.raises(ValueError, match="foundation_mask length"):
        layout.plot(foundation_mask=[True] * 9)


# ---------------------------------------------------------------------
# Auxiliary current electrode (Hilfserder) + measurement workflow
# ---------------------------------------------------------------------


def test_add_auxiliary_electrode_distance_and_direction() -> None:
    """``distance_m`` + ``direction_deg`` places the electrode along
    a ray from the substation."""
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(10.0, 20.0),
    )
    # 200 m east of the substation.
    layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)
    aux = layout.auxiliary_electrode
    assert aux is not None
    assert math.isclose(aux.position_xy[0], 210.0, abs_tol=1e-9)
    assert math.isclose(aux.position_xy[1], 20.0, abs_tol=1e-9)
    # 100 m north of the substation.
    layout.add_auxiliary_electrode(distance_m=100.0, direction_deg=90.0)
    aux = layout.auxiliary_electrode
    assert aux is not None
    assert math.isclose(aux.position_xy[0], 10.0, abs_tol=1e-9)
    assert math.isclose(aux.position_xy[1], 120.0, abs_tol=1e-9)


def test_add_auxiliary_electrode_explicit_xy() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_auxiliary_electrode(position_xy=(123.0, -45.0))
    assert layout.auxiliary_electrode is not None
    assert layout.auxiliary_electrode.position_xy == (123.0, -45.0)


def test_add_auxiliary_electrode_rejects_negative_distance() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="distance_m must be > 0"):
        layout.add_auxiliary_electrode(distance_m=-10.0)
    with pytest.raises(ValueError, match="distance_m must be > 0"):
        layout.add_auxiliary_electrode(distance_m=0.0)


def test_add_auxiliary_electrode_rejects_multiple_positions() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="exactly one"):
        layout.add_auxiliary_electrode()
    with pytest.raises(ValueError, match="exactly one"):
        layout.add_auxiliary_electrode(
            distance_m=200.0, position_xy=(0.0, 0.0),
        )


def test_add_auxiliary_electrode_name_clash() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
        substation_name="trafo",
    )
    with pytest.raises(ValueError, match="substation"):
        layout.add_auxiliary_electrode(distance_m=200.0, name="trafo")
    layout.add_kvs((100.0, 0.0), name="kvs_a")
    with pytest.raises(ValueError, match="KVS"):
        layout.add_auxiliary_electrode(distance_m=200.0, name="kvs_a")


def test_add_auxiliary_electrode_replaces_previous() -> None:
    """Calling add_auxiliary_electrode twice replaces the placement
    (only one aux in v1)."""
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)
    layout.add_auxiliary_electrode(distance_m=300.0, direction_deg=180.0)
    assert layout.auxiliary_electrode is not None
    assert math.isclose(layout.auxiliary_electrode.position_xy[0], -300.0)


def test_to_world_skips_houses_without_foundation() -> None:
    """``foundation_mask=[True, False]`` builds only one house
    foundation; the masked-out house has no electrode in the world.
    """
    fps = [
        _rect_footprint(30.0, 20.0, osm_id=1),
        _rect_footprint(60.0, 20.0, osm_id=2),
    ]
    layout = OrtsnetzLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(100.0, 0.0))
    layout.connect_buildings()

    world = layout.to_world(
        foundation_mask=[True, False],
        seed=0,
    )
    foundation_names = [e.name for e in world.electrodes
                        if "_foundation_0" in e.name]
    assert any(n.startswith("house_000_") for n in foundation_names)
    assert not any(n.startswith("house_001_") for n in foundation_names)


def test_to_world_rejects_mismatched_foundation_mask() -> None:
    fps = [_rect_footprint(30.0, 20.0, osm_id=1)]
    layout = OrtsnetzLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(100.0, 0.0))
    layout.connect_buildings()
    with pytest.raises(ValueError, match="foundation_mask length"):
        layout.to_world(foundation_mask=[True, False, True])


def test_to_world_wires_source_return_through_aux() -> None:
    """When an auxiliary electrode is configured the layout adds the
    substation-side source AND an opposite-sign return source at
    the aux anchor so the measurement loop is physically closed.

    The v0.7 image solver only consumes :attr:`Source.attached_to`;
    :attr:`Source.return_to` is recorded for documentation but not
    honoured by the engine. The extra ``phase_deg=180`` source is
    the workaround that makes the engine see the test current
    leaving via the Hilfserder. With the v0.7 default geometry the
    aux anchor is ``aux_rod_0`` (first rod of the 3-rod triangle).
    """
    fps = [_rect_footprint(30.0, 20.0, osm_id=1)]
    layout = OrtsnetzLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(100.0, 0.0))
    layout.connect_buildings()
    layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0,
                                   name="aux")

    world = layout.to_world(seed=0)
    # The triangle creates 3 rod electrodes, all prefixed ``aux_``.
    aux_rod_electrodes = [
        e for e in world.electrodes if e.name.startswith("aux_rod_")
    ]
    assert len(aux_rod_electrodes) == 3
    # Two sources -- substation injection + aux return.
    assert len(world.sources) == 2
    sub_source = next(
        s for s in world.sources if s.attached_to == "substation_ring_0"
    )
    aux_source = next(
        s for s in world.sources if s.attached_to == "aux_rod_0"
    )
    # The substation source still records the aux as its (logical)
    # return_to (= the first present aux electrode) even though the
    # solver ignores it.
    assert sub_source.return_to == "aux_rod_0"
    assert math.isclose(sub_source.phase_deg, 0.0)
    # The auxiliary return source is the same magnitude but
    # 180 deg out of phase.
    assert math.isclose(aux_source.magnitude, sub_source.magnitude)
    assert math.isclose(aux_source.phase_deg, 180.0)


def test_to_world_remote_return_when_no_aux() -> None:
    """Without an auxiliary electrode the source uses the remote-
    earth return (``return_to=None``), preserving the pre-v0.7
    behaviour. Only one source is created (no return source)."""
    fps = [_rect_footprint(30.0, 20.0, osm_id=1)]
    layout = OrtsnetzLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(100.0, 0.0))
    layout.connect_buildings()
    world = layout.to_world(seed=0)
    assert len(world.sources) == 1
    assert world.sources[0].return_to is None


def test_engine_sees_opposite_currents_at_sub_and_aux() -> None:
    """End-to-end physics check: after solving, the positive side
    of the measurement loop (substation + KVS + foundation
    electrodes -- everything PEN-bonded to the substation cluster)
    carries the +I injection, and the auxiliary bundle carries -I
    -- confirming the loop is closed in the engine's view.

    Note that the substation cluster *alone* does NOT see +I:
    only a fraction of the injected current leaks via the
    substation grounding, the rest flows out via the PEN cables
    to the connected foundations. That's why we sum over *every*
    non-aux electrode.
    """
    import groundfield as gf

    fps = [_rect_footprint(20.0 + 25.0 * i, 18.0, osm_id=10 + i)
           for i in range(2)]
    layout = OrtsnetzLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(120.0, 0.0))
    layout.connect_buildings()
    layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)

    world = layout.to_world(
        soil=gf.HomogeneousSoil(resistivity=100.0),
        source_magnitude_A=1.0, seed=0,
    )
    engine = gf.create_engine(
        backend="image", segment_length=0.5, frequencies=[50.0],
    )
    result = engine.solve(world)

    # Sum every leakage current on each side of the loop. The
    # auxiliary side consists of every electrode whose name starts
    # with ``aux_`` (the 3-rod triangle); the positive side is
    # everything else.
    aux_prefix = "aux_"
    pos_i = sum(
        result.electrode_currents[e.name][0] for e in world.electrodes
        if not e.name.startswith(aux_prefix)
    )
    aux_i = sum(
        result.electrode_currents[e.name][0] for e in world.electrodes
        if e.name.startswith(aux_prefix)
    )
    assert pos_i.real > 0.0
    assert aux_i.real < 0.0
    # Magnitudes match the test current to within 2 %.
    assert abs(pos_i.real - 1.0) < 0.02
    assert abs(aux_i.real + 1.0) < 0.02
    # And the two sides sum (almost) to zero -- Kirchhoff.
    assert abs((pos_i + aux_i).real) < 0.02


def test_plot_includes_auxiliary_marker() -> None:
    """``plot()`` adds the Hilfserder marker without raising."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = _layout_with_osm_ids(n=6)
    layout.add_pen_cable(start="substation", end_xy=(180.0, 0.0))
    layout.connect_buildings()
    layout.add_auxiliary_electrode(distance_m=200.0)
    ax = layout.plot()
    plt.close(ax.figure)


# ---------------------------------------------------------------------
# Voltage probe (Spannungssonde) + measured impedance
# ---------------------------------------------------------------------


def test_add_voltage_probe_inline_fraction_midpoint() -> None:
    """``inline_fraction=0.5`` puts the probe at the midpoint of the
    substation -> aux axis."""
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)
    layout.add_voltage_probe(inline_fraction=0.5)
    probe = layout.voltage_probe
    assert probe is not None
    assert math.isclose(probe.position_xy[0], 100.0, abs_tol=1e-9)
    assert math.isclose(probe.position_xy[1], 0.0, abs_tol=1e-9)


def test_add_voltage_probe_perpendicular_left_and_right() -> None:
    """``perpendicular_fraction=0.5`` + side selects north / south
    when the aux points east."""
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)
    # left = CCW from +x = north (positive y).
    layout.add_voltage_probe(
        perpendicular_fraction=0.5, perpendicular_side="left",
    )
    assert layout.voltage_probe is not None
    assert math.isclose(layout.voltage_probe.position_xy[0], 0.0, abs_tol=1e-9)
    assert math.isclose(layout.voltage_probe.position_xy[1], 100.0, abs_tol=1e-9)
    # right = CW from +x = south (negative y).
    layout.add_voltage_probe(
        perpendicular_fraction=0.5, perpendicular_side="right",
    )
    assert math.isclose(layout.voltage_probe.position_xy[1], -100.0, abs_tol=1e-9)


def test_add_voltage_probe_requires_aux_for_fraction_modes() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    with pytest.raises(RuntimeError, match="auxiliary electrode"):
        layout.add_voltage_probe(inline_fraction=0.5)
    with pytest.raises(RuntimeError, match="auxiliary electrode"):
        layout.add_voltage_probe(perpendicular_fraction=0.5)


def test_add_voltage_probe_explicit_xy_works_without_aux() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_voltage_probe(position_xy=(45.0, 30.0))
    assert layout.voltage_probe is not None
    assert layout.voltage_probe.position_xy == (45.0, 30.0)


def test_add_voltage_probe_rejects_multiple_positions() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_auxiliary_electrode(distance_m=100.0)
    with pytest.raises(ValueError, match="exactly one"):
        layout.add_voltage_probe()
    with pytest.raises(ValueError, match="exactly one"):
        layout.add_voltage_probe(
            inline_fraction=0.5, position_xy=(0.0, 0.0),
        )


def test_add_voltage_probe_rejects_invalid_fractions() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_auxiliary_electrode(distance_m=100.0)
    with pytest.raises(ValueError, match="inline_fraction"):
        layout.add_voltage_probe(inline_fraction=-0.1)
    with pytest.raises(ValueError, match="inline_fraction"):
        layout.add_voltage_probe(inline_fraction=1.5)
    with pytest.raises(ValueError, match="perpendicular_fraction"):
        layout.add_voltage_probe(perpendicular_fraction=2.0)


def test_add_voltage_probe_rejects_invalid_side() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    layout.add_auxiliary_electrode(distance_m=100.0)
    with pytest.raises(ValueError, match="perpendicular_side"):
        layout.add_voltage_probe(
            perpendicular_fraction=0.5,
            perpendicular_side="north",  # invalid; must be left / right
        )


def test_voltage_probe_is_not_materialised_as_electrode() -> None:
    """The probe is a sampling point; ``to_world`` must not create a
    physical rod for it (so it doesn't perturb the potential field).
    """
    fp = _rect_footprint(30.0, 0.0, osm_id=1)
    layout = OrtsnetzLayout.from_footprints(
        [fp], substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(100.0, 0.0))
    layout.connect_buildings()
    layout.add_auxiliary_electrode(distance_m=200.0)
    layout.add_voltage_probe(
        position_xy=(75.0, 25.0), name="my_probe",
    )
    world = layout.to_world(seed=0)
    names = [e.name for e in world.electrodes]
    assert "my_probe" not in names  # not materialised
    # ... but the auxiliary electrode IS in the world (3 rods of
    # the default triangle).
    assert any(n.startswith("aux_rod_") for n in names)


def test_measured_grounding_impedance_with_and_without_probe() -> None:
    """End-to-end: build a small world, solve, and compute the
    fall-of-potential reading with and without a probe. The probe
    reading is *lower* than the remote-earth reading because the
    probe is at a finite, positive potential."""
    import groundfield as gf

    fps = [_rect_footprint(20.0 + 25.0 * i, 18.0, osm_id=10 + i)
           for i in range(4)]
    layout = OrtsnetzLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(120.0, 0.0))
    layout.connect_buildings()
    layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)

    world = layout.to_world(
        soil=gf.HomogeneousSoil(resistivity=100.0),
        source_magnitude_A=1.0, seed=0,
    )
    engine = gf.create_engine(
        backend="image", segment_length=0.5, frequencies=[50.0],
    )
    result = engine.solve(world)

    # Without a probe: remote-earth reference.
    z_remote = layout.measured_grounding_impedance(
        result, source_magnitude_A=1.0,
    )
    assert math.isfinite(abs(z_remote))
    assert abs(z_remote) > 0.0

    # With a 50 % inline probe at (100, 0): U_probe > 0, so the
    # measured value is smaller (a real fall-of-potential meter
    # under-reads when the probe sits in the field's near zone).
    layout.add_voltage_probe(inline_fraction=0.5)
    z_probe = layout.measured_grounding_impedance(
        result, source_magnitude_A=1.0,
    )
    assert math.isfinite(abs(z_probe))
    assert abs(z_probe) < abs(z_remote) + 1e-9


def test_measured_grounding_impedance_requires_aux() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    with pytest.raises(RuntimeError, match="auxiliary electrode"):
        layout.measured_grounding_impedance(object())


def test_measured_grounding_impedance_probe_xy_override() -> None:
    """``probe_xy=`` lets the caller sample any probe position
    from a single solve, overriding ``self.voltage_probe`` for
    that call only. Two different probe positions give two
    different impedance readings on the same field result."""
    import groundfield as gf

    fps = [_rect_footprint(20.0 + 25.0 * i, 18.0, osm_id=10 + i)
           for i in range(2)]
    layout = OrtsnetzLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(120.0, 0.0))
    layout.connect_buildings()
    layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)
    layout.add_voltage_probe(inline_fraction=0.5)  # stored placement

    world = layout.to_world(
        soil=gf.HomogeneousSoil(resistivity=100.0),
        source_magnitude_A=1.0, seed=0,
    )
    engine = gf.create_engine(
        backend="image", segment_length=0.5, frequencies=[50.0],
    )
    result = engine.solve(world)

    # Default call -- uses the stored inline probe at (100, 0).
    z_default = layout.measured_grounding_impedance(result)
    # Override with the SAME position -- must yield the same value.
    z_inline_explicit = layout.measured_grounding_impedance(
        result, probe_xy=(100.0, 0.0),
    )
    assert abs(z_default - z_inline_explicit) < 1e-9
    # Override with the perpendicular position -- yields a
    # different reading because the field is anisotropic with the
    # Hilfserder on the +x axis.
    z_perp = layout.measured_grounding_impedance(
        result, probe_xy=(0.0, 100.0),
    )
    assert abs(z_inline_explicit - z_perp) > 1e-6


def test_verify_current_balance_closes_the_loop() -> None:
    """End-to-end Kirchhoff check: positive-side leakage sums to
    +I_src and the Hilfserder bundle to -I_src."""
    import groundfield as gf

    fps = [_rect_footprint(20.0 + 25.0 * i, 18.0, osm_id=10 + i)
           for i in range(3)]
    layout = OrtsnetzLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(120.0, 0.0))
    layout.connect_buildings()
    layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)

    world = layout.to_world(
        soil=gf.HomogeneousSoil(resistivity=100.0),
        source_magnitude_A=1.0, seed=0,
    )
    engine = gf.create_engine(
        backend="image", segment_length=0.5, frequencies=[50.0],
    )
    result = engine.solve(world)

    balance = layout.verify_current_balance(
        result, source_magnitude_A=1.0,
    )
    assert balance["ok"], balance
    # Positive side sums to +1 A (substation + KVS + foundations).
    assert abs(balance["positive_side_A"].real - 1.0) < 0.02
    # Auxiliary side sums to -1 A.
    assert abs(balance["auxiliary_side_A"].real + 1.0) < 0.02


def test_verify_current_balance_requires_aux() -> None:
    layout = OrtsnetzLayout.from_footprints(
        [], substation_xy=(0.0, 0.0),
    )
    with pytest.raises(RuntimeError, match="auxiliary electrode"):
        layout.verify_current_balance(object())


def test_plot_surface_potential_two_slope_runs() -> None:
    """``two_slope=True`` produces a valid figure with the TwoSlopeNorm
    colour mapping -- smoke test on the Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import groundfield as gf

    fps = [_rect_footprint(20.0 + 25.0 * i, 18.0, osm_id=10 + i)
           for i in range(2)]
    layout = OrtsnetzLayout.from_footprints(
        fps, substation_xy=(0.0, 0.0),
    )
    layout.add_pen_cable(start="substation", end_xy=(120.0, 0.0))
    layout.connect_buildings()
    layout.add_auxiliary_electrode(distance_m=150.0, direction_deg=0.0)
    layout.add_voltage_probe(inline_fraction=0.5)

    world = layout.to_world(
        soil=gf.HomogeneousSoil(resistivity=100.0),
        source_magnitude_A=1.0, seed=0,
    )
    engine = gf.create_engine(
        backend="image", segment_length=0.5, frequencies=[50.0],
    )
    result = engine.solve(world)

    for mode in ({"two_slope": True}, {"symmetric": True}, {"log": True}):
        fig = layout.plot_surface_potential(
            result, world,
            padding_m=40.0, n=60, levels=21,
            **mode,
        )
        plt.close(fig)


def test_plot_includes_voltage_probe_marker() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = _layout_with_osm_ids(n=4)
    layout.add_pen_cable(start="substation", end_xy=(180.0, 0.0))
    layout.connect_buildings()
    layout.add_auxiliary_electrode(distance_m=200.0)
    layout.add_voltage_probe(inline_fraction=0.5)
    ax = layout.plot()
    plt.close(ax.figure)


# ---------------------------------------------------------------------
# Plot smoke test
# ---------------------------------------------------------------------


def test_plot_returns_axes_without_displaying() -> None:
    import matplotlib
    matplotlib.use("Agg")  # off-screen backend so CI stays headless
    import matplotlib.pyplot as plt

    layout = _simple_bus_layout(n_houses=4)
    layout.connect_buildings()
    ax = layout.plot()
    assert ax is not None
    plt.close(ax.figure)
