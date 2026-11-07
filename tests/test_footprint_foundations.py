"""Foundations follow the building footprints in the layout builder (10.54, F28).

``LvNetworkLayout.to_world`` used to give every house the same 10 x 10 m
foundation at the footprint centroid, whatever the building. Foundations
now follow the footprint with the rule of the OSM placement of
``TnNetworkGenerator`` (oriented minimum bounding rectangle: size,
orientation and centre). ``foundation_outline="polygon"`` lays a
ring-style foundation along the real outline (DIN 18014);
``foundation_outline="spec"`` keeps the spec geometry (v0.15.0).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import (
    BuildingTypeSpec,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    LvNetworkLayout,
    OsmBuildingPlacement,
    TnNetworkConfig,
    TnNetworkGenerator,
)

pytestmark = pytest.mark.finding("10.54")


def _layout(footprints: list[gf.BuildingFootprint]) -> LvNetworkLayout:
    layout = LvNetworkLayout.from_footprints(footprints, substation_xy=(0.0, -30.0))
    layout.add_pen_cable(
        start="substation", end_xy=(160.0, -30.0), min_segment_length_m=10.0
    )
    layout.connect_buildings()
    return layout


def _foundations(world: gf.World, prefix: str) -> list:
    return [e for e in world.electrodes if e.name.startswith(prefix)]


GARAGE = gf.BuildingFootprint(
    polygon_xy_m=[(40, 0), (46, 0), (46, 4), (40, 4)], osm_id=1
)
BLOCK = gf.BuildingFootprint(
    polygon_xy_m=[(70, 0), (110, 0), (110, 20), (70, 20)], osm_id=2
)


def _rotated(cx: float, cy: float, w: float, h: float, angle: float, osm_id: int):
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return gf.BuildingFootprint(
        polygon_xy_m=[(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts],
        osm_id=osm_id,
    )


def test_foundation_size_follows_the_footprint() -> None:
    """The pass-9 F28 reproduction: a garage and an apartment block."""
    world = _layout([GARAGE, BLOCK]).to_world()
    (garage,) = _foundations(world, "house_000_foundation")
    (block,) = _foundations(world, "house_001_foundation")
    assert garage.kind == block.kind == "grid_mesh"
    assert (garage.corner[:2], garage.size) == ((40.0, 0.0), (6.0, 4.0))
    assert (block.corner[:2], block.size) == ((70.0, 0.0), (40.0, 20.0))


def test_rotated_footprint_gives_a_rotated_foundation_on_the_ombr() -> None:
    fp = _rotated(60.0, 15.0, 14.0, 9.0, 30.0, osm_id=7)
    world = _layout([fp]).to_world()
    wires = _foundations(world, "house_000_foundation")
    points = np.array(
        [
            v[:2]
            for e in wires
            for v in (e.vertices if e.kind == "polyline" else [e.start, e.end])
        ]
    )
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    assert centre == pytest.approx([60.0, 15.0], abs=1e-9)
    (cx, cy), (dx, dy), angle = fp.oriented_bounding_rectangle()
    assert (dx, dy, angle) == pytest.approx((14.0, 9.0, 30.0))
    first = wires[0]  # local -y edge, along the long axis
    direction = np.subtract(first.vertices[-1][:2], first.vertices[0][:2])
    assert np.degrees(np.arctan2(direction[1], direction[0])) == pytest.approx(30.0)
    assert np.hypot(*direction) == pytest.approx(14.0)


def test_layout_and_tn_generator_build_the_same_foundation() -> None:
    """One footprint rule for both front-ends (10.78 consolidation)."""
    fp = gf.BuildingFootprint(
        polygon_xy_m=[(0, 0), (20, 0), (20, 8), (8, 8), (8, 22), (0, 22)], osm_id=3
    )
    spec = GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", depth_m=0.7)]
    )
    world_layout = _layout([fp]).to_world(house_grounding=spec)
    cfg = TnNetworkConfig(
        building_types=[BuildingTypeSpec(name="residential", grounding=spec)],
        building_counts={"residential": 1},
        placement=OsmBuildingPlacement(footprints=[fp]),
    )
    world_tn = TnNetworkGenerator(cfg, seed=0).build()

    def geometry(world: gf.World, prefix: str) -> str:
        items = []
        for e in _foundations(world, prefix):
            d = e.model_dump(mode="json")
            d.pop("name")
            items.append(d)
        return json.dumps(items, sort_keys=True)

    assert geometry(world_layout, "house_000_foundation") == geometry(
        world_tn, "residential_0_foundation"
    )


def test_polygon_outline_follows_the_real_footprint() -> None:
    l_shape = gf.BuildingFootprint(
        polygon_xy_m=[(30, 0), (42, 0), (42, 9), (35, 9), (35, 6), (30, 6)], osm_id=4
    )
    ring = GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", depth_m=0.7)]
    )
    world = _layout([l_shape]).to_world(
        house_grounding=ring, foundation_outline="polygon"
    )
    (foundation,) = _foundations(world, "house_000_foundation")
    assert foundation.kind == "polyline" and foundation.closed
    got = np.array([v[:2] for v in foundation.vertices])
    np.testing.assert_allclose(
        got, np.array(l_shape.polygon_xy_m, dtype=float), atol=1e-12
    )
    assert all(v[2] == 0.7 for v in foundation.vertices)


def test_polygon_outline_rejects_mesh_foundations() -> None:
    with pytest.raises(ValueError, match="style='ring'"):
        _layout([GARAGE]).to_world(foundation_outline="polygon")


def test_spec_outline_reproduces_the_fixed_foundation() -> None:
    world = _layout([GARAGE, BLOCK]).to_world(foundation_outline="spec")
    for i, fp in enumerate((GARAGE, BLOCK)):
        (foundation,) = _foundations(world, f"house_{i:03d}_foundation")
        cx, cy = fp.centroid_xy_m()
        assert foundation.size == (10.0, 10.0)
        assert foundation.corner[:2] == pytest.approx((cx - 5.0, cy - 5.0))


def test_foundation_outline_is_recorded_in_the_plan() -> None:
    plan = _layout([GARAGE]).plan(foundation_outline="ombr")
    (house,) = plan.sites_with_role("building")
    assert house.footprint == GARAGE
    assert house.electrodes[0].spec.size_xy_m == (6.0, 4.0)
