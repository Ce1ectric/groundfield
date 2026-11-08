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


def _extent(wires: list) -> tuple[tuple[float, float], tuple[float, float]]:
    """``(corner, size)`` of the bounding box of a foundation's wires (a
    mesh foundation is one electrode per wire, audit A2G-04)."""
    points = [v for e in wires for line in e.wire_polylines() for v in line]
    xs = [float(v[0]) for v in points]
    ys = [float(v[1]) for v in points]
    return (min(xs), min(ys)), (max(xs) - min(xs), max(ys) - min(ys))


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
    garage = _foundations(world, "house_000_foundation")
    block = _foundations(world, "house_001_foundation")
    assert {e.kind for e in garage + block} == {"polyline"}
    assert _extent(garage) == ((40.0, 0.0), (6.0, 4.0))
    assert _extent(block) == ((70.0, 0.0), (40.0, 20.0))


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
        corner, size = _extent(_foundations(world, f"house_{i:03d}_foundation"))
        cx, cy = fp.centroid_xy_m()
        assert size == pytest.approx((10.0, 10.0))
        assert corner == pytest.approx((cx - 5.0, cy - 5.0))


def test_foundation_outline_is_recorded_in_the_plan() -> None:
    plan = _layout([GARAGE]).plan(foundation_outline="ombr")
    (house,) = plan.sites_with_role("building")
    assert house.footprint == GARAGE
    assert house.electrodes[0].spec.size_xy_m == (6.0, 4.0)


def _square(cx: float, cy: float, osm_id: int, w: float = 10.0, h: float = 10.0):
    return gf.BuildingFootprint(
        polygon_xy_m=[
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ],
        osm_id=osm_id,
    )


def test_footprint_outline_does_not_shift_the_random_draws() -> None:
    """Audit M2: with a stochastic foundation size the ``"ombr"`` default
    skipped the size draw, the random stream shifted and optional rods
    moved to other houses (6 -> 8 rods) although the footprints were exact
    10 x 10 m squares. The draws are made first, the footprint shapes the
    resolved foundation afterwards."""
    from groundfield.generators import RodElectrodeSpec, Uniform

    footprints = [
        _square(30.0 + 25.0 * (i // 2), 18.0 if i % 2 == 0 else -18.0, 500 + i)
        for i in range(12)
    ]
    spec = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                style="ring", size_m=Uniform(low=9.9999, high=10.0001)
            ),
            RodElectrodeSpec(length_m=1.5, presence_prob=0.5, offset_xy_m=(5.0, 0.0)),
        ]
    )
    layout = LvNetworkLayout.from_footprints(footprints, substation_xy=(0.0, 0.0))
    layout.add_pen_cable(start="substation", end_xy=(200.0, 0.0))
    layout.connect_all_buildings()
    rods = {}
    for outline in ("spec", "ombr", "polygon"):
        world = layout.to_world(
            seed=3, house_grounding=spec, foundation_outline=outline
        )
        rods[outline] = sorted(
            e.name
            for e in world.electrodes
            if e.name.startswith("house") and e.kind == "rod"
        )
    assert rods["ombr"] == rods["spec"] == rods["polygon"]
    assert 0 < len(rods["spec"]) < 12


def test_a_degenerate_footprint_gets_a_minimum_foundation() -> None:
    """Audit L5: a mapped wall (10 m x 5 mm) became a 10 m x 5 mm mesh and
    the whole ``to_world`` failed (segment shorter than the wire radius);
    v0.15.0 (fixed 10 x 10 m spec foundation) built the layout. The OMBR
    side is widened to MIN_FOUNDATION_SIDE_M (1 m), with a warning."""
    layout = _layout(
        [_square(40.0, 18.0, 1, 10.0, 8.0), _square(70.0, 18.0, 2, 10.0, 0.005)]
    )
    with pytest.warns(UserWarning, match="widened"):
        world = layout.to_world(seed=0)
    _, size = _extent(_foundations(world, "house_001_foundation"))
    assert sorted(size) == pytest.approx([1.0, 10.0])
    res = gf.create_engine(backend="image", segment_length=1.0).solve(world)
    assert np.isfinite(res.input_impedance()[0])

    from groundfield.generators.grounding import MIN_FOUNDATION_SIDE_M

    assert MIN_FOUNDATION_SIDE_M == 1.0


# audit A2G-06
@pytest.mark.parametrize(
    ("outline", "size"),
    [
        ([(60.0, 10.0), (70.0, 10.0), (80.0, 10.0), (60.0, 10.0)], [1.0, 20.0]),
        ([(60.0, 10.0)] * 4, [1.0, 1.0]),
    ],
    ids=["zero-width", "point"],
)
def test_a_footprint_without_area_gets_a_minimum_foundation(outline, size) -> None:
    """Audit L5 limit: at 0 mm width the OMBR raised IndexError and
    ``to_world`` crashed instead of widening the foundation."""
    wall = gf.BuildingFootprint(polygon_xy_m=outline, osm_id=2)
    layout = _layout([_square(40.0, 18.0, 1, 10.0, 8.0), wall])
    with pytest.warns(UserWarning, match="widened"):
        world = layout.to_world(seed=0)
    (x0, y0), (dx, dy) = _extent(_foundations(world, "house_001_foundation"))
    assert sorted((dx, dy)) == pytest.approx(size)
    centre = (x0 + dx / 2.0, y0 + dy / 2.0)
    assert centre == pytest.approx((70.0, 10.0) if size[1] > 1 else (60.0, 10.0))
    res = gf.create_engine(backend="image", segment_length=0.5).solve(world)
    assert np.isfinite(res.input_impedance()[0])


# audit A2G-06
def test_tn_generator_widens_a_footprint_without_area() -> None:
    """The OSM placement of TnNetworkGenerator uses the same rule (its
    default ``min_area_m2=16`` would drop the wall before)."""
    wall = gf.BuildingFootprint(
        polygon_xy_m=[(60.0, 10.0), (70.0, 10.0), (80.0, 10.0)], osm_id=2
    )
    placement = OsmBuildingPlacement(
        footprints=[_square(40.0, 18.0, 1), wall], min_area_m2=0.0
    )
    cfg = TnNetworkConfig(building_counts={"residential": 2}, placement=placement)
    with pytest.warns(UserWarning, match="widened"):
        world = TnNetworkGenerator(cfg, seed=0).build()
    assert any(e.name.startswith("residential_1_foundation") for e in world.electrodes)


# audit A3SG-07
@pytest.mark.parametrize("width_m", [0.005, 0.3, 0.49, 0.8])
def test_a_thin_outline_with_a_polygon_foundation_is_widened(width_m: float) -> None:
    """``foundation_outline="polygon"`` with an outline narrower than
    MIN_FOUNDATION_SIDE_M: base: ``to_world`` raised ValueError ("collapses to
    fewer than 3 vertices after removing sub-centimetre edges", although the
    edge limit is 0.5 m) below 0.5 m, and laid a ring along both faces of a
    0.8 m wall. Now the widened OMBR, with the OMBR path's warning."""
    thin = gf.BuildingFootprint(
        polygon_xy_m=[
            (60.0, 10.0),
            (72.0, 10.0),
            (72.0, 10.0 + width_m),
            (60.0, 10.0 + width_m),
        ],
        osm_id=2,
    )
    assert not thin.is_degenerate()  # the Overpass parser keeps it
    layout = _layout([_square(40.0, 18.0, 1, 10.0, 8.0), thin])
    ring = GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", size_m=10.0, depth_m=0.8)]
    )
    with pytest.warns(UserWarning, match="widened"):
        world = layout.to_world(
            seed=0, foundation_outline="polygon", house_grounding=ring
        )
    (wall,) = _foundations(world, "house_001_foundation")
    corner, size = _extent([wall])
    assert sorted(size) == pytest.approx([1.0, 12.0])
    centre = (corner[0] + size[0] / 2.0, corner[1] + size[1] / 2.0)
    assert centre == pytest.approx((66.0, 10.0 + width_m / 2.0))
    res = gf.create_engine(backend="image", segment_length=0.5).solve(world)
    assert np.isfinite(res.input_impedance()[0])


# audit A3SG-07
def test_a_polygon_foundation_of_an_ordinary_outline_is_unchanged() -> None:
    """No widening, no warning and no shapely call for outlines wider than
    MIN_FOUNDATION_SIDE_M on average."""
    import warnings

    layout = _layout([_square(40.0, 18.0, 1, 10.0, 8.0)])
    ring = GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", size_m=10.0, depth_m=0.8)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plan = layout.plan(seed=0, foundation_outline="polygon", house_grounding=ring)
    (spec,) = [e.spec for s in plan.sites if s.role == "building" for e in s.electrodes]
    assert spec.polygon_xy_m == ((-5.0, -4.0), (5.0, -4.0), (5.0, 4.0), (-5.0, 4.0))


# audit A3SG-07
def test_a_collapsing_polygon_outline_names_the_real_edge_limit() -> None:
    """The materialiser's last check (a spec built by hand) names the 0.5 m
    edge limit instead of "sub-centimetre edges" and an alternative."""
    from groundfield.generators.materialise import materialise_site
    from groundfield.generators.plan import PlannedElectrode, PlannedSite

    spec = FoundationElectrodeSpec(
        style="ring",
        size_m=10.0,
        depth_m=0.8,
        polygon_xy_m=((0.0, 0.0), (12.0, 0.0), (12.0, 0.3), (0.0, 0.3)),
    )
    site = PlannedSite(
        name="h",
        role="building",
        position_xy=(0.0, 0.0),
        electrodes=[PlannedElectrode(index=0, spec=spec)],
    )
    world = gf.World(soil=gf.HomogeneousSoil(rho=100.0))
    with pytest.raises(ValueError, match=r"shorter than MIN_POLYGON_EDGE_M = 0\.5 m"):
        materialise_site(world, site)


# audit A3SG-07
@pytest.mark.parametrize("outline", ["ombr", "polygon"])
def test_a_thin_outline_without_a_foundation_is_left_alone(outline: str) -> None:
    """A grounding without a foundation electrode has nothing to shape: no
    widening warning for a wall outline (it warned on the OMBR path)."""
    import warnings

    from groundfield.generators import RodElectrodeSpec

    rods = GroundingSystemSpec(electrodes=[RodElectrodeSpec(length_m=2.0)])
    wall = gf.BuildingFootprint(
        polygon_xy_m=[(0.0, 0.0), (12.0, 0.0), (12.0, 0.01), (0.0, 0.01)], osm_id=3
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert rods.with_footprint(wall, outline=outline) is rods
    with pytest.raises(ValueError, match="outline must be"):
        rods.with_footprint(wall, outline="hull")
