"""Tests for the footprint-driven placement bridge.

Covers :class:`groundfield.geo.placement.OsmBuildingPlacement` itself
plus the integration with :class:`TnNetworkGenerator`:

* The placement's :meth:`generate` returns polygon centroids in
  declared order and obeys the ``min_area_m2`` filter.
* :meth:`footprint_at` looks up the polygon associated with a
  generator-built site.
* :class:`PlacementSpec` JSON round-trips select ``OsmBuildingPlacement``
  via its ``kind == "osm"`` discriminator.
* The rotated-foundation branch of
  :meth:`GroundingSystemSpec.build_at` registers the expected
  number of strip electrodes and bonds them internally.
* The OSM-driven integration in :class:`TnNetworkGenerator` rewrites
  the per-building :class:`FoundationElectrodeSpec` so that
  ``size_xy_m`` and ``orientation_deg`` reflect the polygon's
  oriented minimum bounding rectangle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import TypeAdapter

from groundfield.api import create_world
from groundfield.generators.electrode_specs import FoundationElectrodeSpec
from groundfield.generators.grounding import GroundingSystemSpec
from groundfield.generators.placement import (
    ExplicitPlacement,
    OsmBuildingPlacement,
    PlacementSpec,
)
from groundfield.geo.footprint import BuildingFootprint
from groundfield.soil.models import TwoLayerSoil


# ---------------------------------------------------------------------
# OsmBuildingPlacement basics
# ---------------------------------------------------------------------


def _rectangle(centre: tuple[float, float], size: tuple[float, float]) -> BuildingFootprint:
    """Helper: axis-aligned rectangle with the requested centre/size."""
    cx, cy = centre
    dx, dy = size
    return BuildingFootprint(
        polygon_xy_m=[
            (cx - dx / 2, cy - dy / 2),
            (cx + dx / 2, cy - dy / 2),
            (cx + dx / 2, cy + dy / 2),
            (cx - dx / 2, cy + dy / 2),
        ],
    )


def test_osm_placement_generate_returns_centroids_in_order() -> None:
    fps = [
        _rectangle((10.0, 20.0), (8.0, 6.0)),
        _rectangle((30.0, 40.0), (10.0, 4.0)),
        _rectangle((-5.0, 0.0), (12.0, 8.0)),
    ]
    p = OsmBuildingPlacement(footprints=fps, min_area_m2=0.0)
    positions = p.generate(3, np.random.default_rng(0))
    assert positions == [(10.0, 20.0), (30.0, 40.0), (-5.0, 0.0)]


def test_osm_placement_filters_below_min_area() -> None:
    fps = [
        _rectangle((0.0, 0.0), (10.0, 8.0)),   # 80 m²
        _rectangle((50.0, 0.0), (2.0, 2.0)),  # 4 m²  — dropped
    ]
    p = OsmBuildingPlacement(footprints=fps, min_area_m2=16.0)
    positions = p.generate(1, np.random.default_rng(0))
    assert positions == [(0.0, 0.0)]


def test_osm_placement_raises_when_requesting_too_many() -> None:
    fps = [_rectangle((0.0, 0.0), (10.0, 8.0))]
    p = OsmBuildingPlacement(footprints=fps, min_area_m2=0.0)
    with pytest.raises(ValueError):
        p.generate(5, np.random.default_rng(0))


def test_osm_placement_selection_all_ignores_n() -> None:
    fps = [
        _rectangle((0.0, 0.0), (10.0, 8.0)),
        _rectangle((50.0, 0.0), (10.0, 8.0)),
    ]
    p = OsmBuildingPlacement(footprints=fps, min_area_m2=0.0, selection="all")
    assert len(p.generate(1, np.random.default_rng(0))) == 2


def test_osm_placement_footprint_at_returns_corresponding_polygon() -> None:
    fps = [
        _rectangle((0.0, 0.0), (10.0, 8.0)),
        _rectangle((50.0, 0.0), (12.0, 6.0)),
    ]
    p = OsmBuildingPlacement(footprints=fps, min_area_m2=0.0)
    assert p.footprint_at(0).area_m2() == pytest.approx(80.0)
    assert p.footprint_at(1).area_m2() == pytest.approx(72.0)
    assert p.footprint_at(2) is None


def test_osm_placement_round_trips_via_placement_union() -> None:
    fps = [_rectangle((1.0, 2.0), (10.0, 6.0))]
    p = OsmBuildingPlacement(footprints=fps)
    adapter = TypeAdapter(PlacementSpec)
    payload = adapter.dump_json(p)
    restored = adapter.validate_json(payload)
    assert isinstance(restored, OsmBuildingPlacement)
    assert len(restored.footprints) == 1
    assert restored.footprints[0].area_m2() == pytest.approx(60.0)


# ---------------------------------------------------------------------
# Rotated foundation path
# ---------------------------------------------------------------------


def test_rotated_foundation_registers_perimeter_strips() -> None:
    """A rotated ring-style foundation (style='ring', orientation=30 deg)
    must register exactly four :class:`StripElectrode` primitives
    (the four perimeter sides) plus three bonds, all sharing the
    spec-level anchor name as the first sub-electrode.
    """
    world = create_world(
        name="rot_found_ring",
        soil=TwoLayerSoil(rho_1=100.0, rho_2=50.0, h_1=2.0),
    )
    gs = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                style="ring", size_xy_m=(10.0, 6.0),
                depth_m=0.8, orientation_deg=30.0,
            )
        ],
    )
    anchor = gs.build_at(
        world, site_xy=(0.0, 0.0), name_prefix="house",
        rng=np.random.default_rng(0),
    )
    assert anchor == "house_foundation_0"
    names = [e.name for e in world.electrodes]
    # The anchor name + three follow-up strips: w1, w2, w3.
    assert "house_foundation_0" in names
    assert "house_foundation_0_w1" in names
    assert "house_foundation_0_w2" in names
    assert "house_foundation_0_w3" in names
    # No GridMeshElectrode was created.
    assert all(e.kind != "grid_mesh" for e in world.electrodes)


def test_rotated_foundation_mesh_style_adds_internal_braces() -> None:
    """style='mesh' with n_x = n_y = 2 yields 6 wires
    (3 longitudinal + 3 transverse) when rotated."""
    world = create_world(
        name="rot_found_mesh",
        soil=TwoLayerSoil(rho_1=100.0, rho_2=50.0, h_1=2.0),
    )
    gs = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                style="mesh", size_xy_m=(10.0, 6.0),
                depth_m=0.8, orientation_deg=15.0, n_x=2, n_y=2,
            )
        ],
    )
    gs.build_at(
        world, site_xy=(0.0, 0.0), name_prefix="house",
        rng=np.random.default_rng(0),
    )
    strips = [e for e in world.electrodes if e.kind == "strip"]
    # 3 long-axis wires + 3 short-axis wires.
    assert len(strips) == 6


def test_axis_aligned_foundation_stays_grid_mesh() -> None:
    """When ``orientation_deg`` is ``None`` (the default), the
    foundation must still materialise as a single
    :class:`GridMeshElectrode` — Phase A is strictly additive."""
    world = create_world(
        name="aa_found",
        soil=TwoLayerSoil(rho_1=100.0, rho_2=50.0, h_1=2.0),
    )
    gs = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                style="mesh", size_xy_m=(10.0, 6.0), depth_m=0.8,
            )
        ],
    )
    gs.build_at(
        world, site_xy=(0.0, 0.0), name_prefix="house",
        rng=np.random.default_rng(0),
    )
    kinds = [e.kind for e in world.electrodes]
    assert kinds == ["grid_mesh"]


def test_zero_orientation_uses_axis_aligned_path() -> None:
    """``orientation_deg = 0.0`` must take the fast path too —
    avoids needlessly synthesising 4 strips for an axis-aligned house."""
    world = create_world(
        name="zero_deg",
        soil=TwoLayerSoil(rho_1=100.0, rho_2=50.0, h_1=2.0),
    )
    gs = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                style="ring", size_xy_m=(10.0, 6.0),
                depth_m=0.8, orientation_deg=0.0,
            )
        ],
    )
    gs.build_at(
        world, site_xy=(0.0, 0.0), name_prefix="house",
        rng=np.random.default_rng(0),
    )
    kinds = [e.kind for e in world.electrodes]
    assert kinds == ["grid_mesh"]


# ---------------------------------------------------------------------
# End-to-end TN-Ortsnetz with OSM placement
# ---------------------------------------------------------------------


def _build_tn_with_osm_placement(
    footprints: list[BuildingFootprint],
) -> "groundfield.World":
    """Helper: build a tiny TN-Ortsnetz world driven by the given
    OSM footprints."""
    from groundfield.generators.building import BuildingTypeSpec
    from groundfield.generators.tn_network import (
        SubstationConfig,
        TnNetworkConfig,
        TnNetworkGenerator,
    )

    placement = OsmBuildingPlacement(footprints=footprints, min_area_m2=0.0)
    btype = BuildingTypeSpec(
        name="residential",
        grounding=GroundingSystemSpec(
            electrodes=[
                FoundationElectrodeSpec(
                    style="ring", size_m=8.0, depth_m=0.8,
                )
            ],
        ),
    )
    cfg = TnNetworkConfig(
        name="osm_tn",
        substation=SubstationConfig(position=(0.0, 0.0)),
        placement=placement,
        building_types=[btype],
        building_counts={"residential": len(footprints)},
    )
    gen = TnNetworkGenerator(cfg, seed=0)
    return gen.build()


def test_tn_generator_uses_footprint_dimensions_for_each_house() -> None:
    r"""Two houses with very different OMBR sizes / orientations
    produce two foundations with the matching size_xy_m and
    orientation_deg. We verify by checking the strip electrodes
    that the rotated-foundation path emitted."""
    # House A: 10 x 6 rectangle rotated by 30 deg, centred at (0, 0).
    angle_a = math.radians(30.0)
    ca, sa = math.cos(angle_a), math.sin(angle_a)
    rect_a = [(-5.0, -3.0), (5.0, -3.0), (5.0, 3.0), (-5.0, 3.0)]
    poly_a = [(ca * x - sa * y, sa * x + ca * y) for x, y in rect_a]
    fp_a = BuildingFootprint(polygon_xy_m=poly_a)

    # House B: 14 x 8 rectangle rotated by -20 deg, centred at (100, 100).
    angle_b = math.radians(-20.0)
    cb, sb = math.cos(angle_b), math.sin(angle_b)
    rect_b = [(-7.0, -4.0), (7.0, -4.0), (7.0, 4.0), (-7.0, 4.0)]
    poly_b = [(cb * x - sb * y + 100.0, sb * x + cb * y + 100.0) for x, y in rect_b]
    fp_b = BuildingFootprint(polygon_xy_m=poly_b)

    world = _build_tn_with_osm_placement([fp_a, fp_b])

    # Every building emitted four perimeter strips (style='ring'),
    # so we expect 8 strip electrodes total from the foundations.
    strips = [e for e in world.electrodes if e.kind == "strip"]
    assert len(strips) == 8

    # Pick the long-axis strip of house A — its length must match
    # the OMBR's long side (10 m). The two long strips of each
    # rotated ring foundation are the (-y) and (+y) edges, both
    # 10 m long for house A.
    house_a_strips = [
        e for e in strips
        if e.name.startswith("residential_0_foundation_")
    ]
    assert len(house_a_strips) == 4
    lengths_a = sorted(e.length for e in house_a_strips)
    # Two long sides ≈ 10 m, two short sides ≈ 6 m.
    assert lengths_a[0] == pytest.approx(6.0, abs=1e-6)
    assert lengths_a[1] == pytest.approx(6.0, abs=1e-6)
    assert lengths_a[2] == pytest.approx(10.0, abs=1e-6)
    assert lengths_a[3] == pytest.approx(10.0, abs=1e-6)

    house_b_strips = [
        e for e in strips
        if e.name.startswith("residential_1_foundation_")
    ]
    lengths_b = sorted(e.length for e in house_b_strips)
    # House B: two long sides ≈ 14 m, two short sides ≈ 8 m.
    assert lengths_b[0] == pytest.approx(8.0, abs=1e-6)
    assert lengths_b[1] == pytest.approx(8.0, abs=1e-6)
    assert lengths_b[2] == pytest.approx(14.0, abs=1e-6)
    assert lengths_b[3] == pytest.approx(14.0, abs=1e-6)


def test_tn_generator_reproducibility_under_seed() -> None:
    """A footprint-driven build with a fixed RNG seed produces the
    same electrodes (names, positions, sizes) across reruns."""
    fps = [
        _rectangle((0.0, 0.0), (10.0, 6.0)),
        _rectangle((50.0, 0.0), (12.0, 8.0)),
    ]
    w1 = _build_tn_with_osm_placement(fps)
    w2 = _build_tn_with_osm_placement(fps)
    names_1 = sorted(e.name for e in w1.electrodes)
    names_2 = sorted(e.name for e in w2.electrodes)
    assert names_1 == names_2
