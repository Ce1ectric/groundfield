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
    BuildingConnection,
    KvsPlacement,
    ObstacleBox,
    OrtsnetzLayout,
    PenCable,
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
