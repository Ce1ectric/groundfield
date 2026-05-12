"""Tests for the pre-solve world diagnostics.

Validates :mod:`groundfield.diagnostics` against:

- the structural snapshot returned by :func:`world_statistics`
  (counts, lengths, bounds, footprint, conductor stats),
- the **exact** match between :func:`expected_segments` and
  the image-family discretiser — every electrode kind is
  cross-checked against the actual segment count produced by
  ``Engine(backend='image').solve(world).point_sources``,
- :func:`check_segment_resolution` warnings (thin-wire ratio,
  electrode size vs. segment length, total-segment budget),
- top-level export.
"""

from __future__ import annotations

import math

import pytest

import groundfield as gf
from groundfield.diagnostics import (
    _MIN_THINWIRE_RATIO,
    check_segment_resolution,
    expected_segments,
    world_statistics,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _mixed_world() -> gf.World:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil, name="mixed")

    g_rod = gf.create_electrode(
        world, "rod", name="rod_a", position=(0.0, 0.0, 0.5), length=1.5
    )
    gf.create_electrode(
        world, "ring", name="ring_a", center=(10.0, 0.0, 0.8), radius=2.5
    )
    gf.create_electrode(
        world,
        "strip",
        name="strip_a",
        start=(0.0, 5.0, 0.6),
        end=(8.0, 5.0, 0.6),
    )
    gf.create_electrode(
        world,
        "grid_mesh",
        name="mesh_a",
        corner=(15.0, -3.0, 0.7),
        size=(6.0, 4.0),
        n_x=3,
        n_y=2,
    )
    gf.create_conductor(
        world, name="bond", start=g_rod, end="ring_a",
        conductor_type="bare_copper",
    )
    # A second, distributed PEN conductor.
    gf.create_conductor(
        world,
        name="pen_trunk",
        start=(-5.0, 0.0, -0.2),
        end=(20.0, 0.0, -0.2),
        conductor_type="pen",
        cross_section=50e-6,
        coupling_to_soil="isolated",
        discretize_segment_length=2.5,
    )
    gf.create_source(world, name="src", attached_to=g_rod, magnitude=10.0)
    return world


# ---------------------------------------------------------------------
# world_statistics
# ---------------------------------------------------------------------


def test_world_statistics_basic_counts() -> None:
    stats = world_statistics(_mixed_world())
    assert stats["n_electrodes"] == 4
    assert stats["n_electrodes_by_kind"] == {
        "rod": 1, "ring": 1, "strip": 1, "grid_mesh": 1,
    }
    assert stats["n_conductors"] == 2
    assert stats["n_conductors_by_type"] == {"bare_copper": 1, "pen": 1}
    assert stats["n_distributed_conductors"] == 1
    assert stats["n_lumped_conductors"] == 1
    assert stats["n_galvanic_conductors"] == 0  # both isolated by default
    assert stats["n_isolated_conductors"] == 2
    assert stats["n_sources"] == 1


def test_world_statistics_total_wire_length_matches_geometry() -> None:
    """Sum of analytic wire lengths must match the structural total."""
    stats = world_statistics(_mixed_world())
    # rod 1.5 m, ring 2 pi 2.5, strip 8.0, grid_mesh (3 × 4 + 4 × 6 +
    # n_y+1 × dx + n_x+1 × dy) = (n_y+1)*dx + (n_x+1)*dy
    #   with n_x=3, n_y=2, dx=6, dy=4 -> 3*6 + 4*4 = 18 + 16 = 34
    expected = 1.5 + 2.0 * math.pi * 2.5 + 8.0 + 34.0
    assert stats["total_electrode_wire_length_m"] == pytest.approx(expected)


def test_world_statistics_conductor_length_stats() -> None:
    stats = world_statistics(_mixed_world())
    cs = stats["conductor_length_stats"]
    assert set(cs.keys()) == {"min", "median", "max", "mean"}
    assert cs["min"] <= cs["median"] <= cs["max"]
    assert cs["mean"] > 0.0


def test_world_statistics_bounds_and_footprint_consistent() -> None:
    stats = world_statistics(_mixed_world())
    x_min, x_max, y_min, y_max, _, _ = stats["bounds_3d"]
    assert stats["footprint_xy"] == (x_min, x_max, y_min, y_max)
    assert stats["footprint_area_m2"] == pytest.approx(
        (x_max - x_min) * (y_max - y_min)
    )


def test_world_statistics_layered_soil_flag() -> None:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.0)
    stats = world_statistics(world)
    assert stats["has_layered_soil"] is True


def test_world_statistics_empty_world_safe() -> None:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    stats = world_statistics(world)
    assert stats["n_electrodes"] == 0
    assert stats["conductor_length_stats"] == {}
    assert stats["footprint_area_m2"] == 0.0


# ---------------------------------------------------------------------
# expected_segments — exact match with the image discretiser
# ---------------------------------------------------------------------


def test_expected_segments_rod_matches_solver() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.05)

    pred = expected_segments(world, eng)
    actual = len(eng.solve(world).point_sources)
    assert pred["total"] == actual
    assert pred["per_electrode"]["g1"] == actual


def test_expected_segments_ring_matches_solver() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "ring", name="g1", center=(0, 0, 0.8), radius=2.0)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.5)

    pred = expected_segments(world, eng)
    actual = len(eng.solve(world).point_sources)
    assert pred["total"] == actual


def test_expected_segments_strip_matches_solver() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world, "strip", name="g1", start=(0, 0, 0.6), end=(10, 0, 0.6)
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.3)

    pred = expected_segments(world, eng)
    actual = len(eng.solve(world).point_sources)
    assert pred["total"] == actual


def test_expected_segments_grid_mesh_matches_solver() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world,
        "grid_mesh",
        name="g1",
        corner=(0, 0, 0.7),
        size=(6.0, 4.0),
        n_x=3,
        n_y=2,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.5)

    pred = expected_segments(world, eng)
    actual = len(eng.solve(world).point_sources)
    assert pred["total"] == actual


def test_expected_segments_mesh_matches_solver() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world,
        "mesh",
        name="g1",
        corner=(0, 0, 0.7),
        size=(5.0, 4.0),
        spacing=1.0,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.5)

    pred = expected_segments(world, eng)
    actual = len(eng.solve(world).point_sources)
    assert pred["total"] == actual


def test_expected_segments_per_kind_aggregates() -> None:
    pred = expected_segments(
        _mixed_world(), gf.create_engine(backend="image", segment_length=0.5)
    )
    assert set(pred["per_kind"].keys()) == {"rod", "ring", "strip", "grid_mesh"}
    assert sum(pred["per_kind"].values()) == pred["electrode_total"]
    assert pred["total"] == pred["electrode_total"] + pred["conductor_total"]


def test_expected_segments_galvanic_distributed_conductor_counts() -> None:
    """Distributed *galvanic* conductor contributes one segment per
    sub-piece — its midpoints leak current and end up in
    ``point_sources``."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    g1 = gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.0)
    g2 = gf.create_electrode(world, "rod", name="g2", position=(20, 0, 0.5), length=1.0)
    gf.create_conductor(
        world, name="bare_trunk", start=g1, end=g2,
        conductor_type="bare_copper",
        cross_section=50e-6,
        coupling_to_soil="galvanic",
        discretize_segment_length=2.5,
    )
    gf.create_source(world, attached_to=g1, magnitude=1.0)

    eng = gf.create_engine(backend="image", segment_length=1.0)
    pred = expected_segments(world, eng)
    # 20 m trunk / 2.5 m -> 8 sub-pieces.
    assert pred["per_conductor"] == {"bare_trunk": 8}
    # Bit-exact match against the actual solve.
    actual = len(eng.solve(world).point_sources)
    assert pred["total"] == actual


def test_expected_segments_isolated_distributed_conductor_zero() -> None:
    """An *isolated* distributed conductor (jacketed PEN cable) does
    not contribute to point_sources — its longitudinal-branch chain
    is represented by interior KCL nodes only."""
    pred = expected_segments(
        _mixed_world(), gf.create_engine(backend="image", segment_length=1.0)
    )
    # The mixed world's PEN trunk is isolated -> 0 contribution.
    assert pred["per_conductor"] == {}
    assert pred["conductor_total"] == 0


def test_expected_segments_rejects_invalid_segment_length() -> None:
    world = _mixed_world()
    with pytest.raises(ValueError, match="segment_length"):
        bad = gf.create_engine(backend="image", segment_length=0.1)
        bad.segment_length = 0.0
        expected_segments(world, bad)


# ---------------------------------------------------------------------
# check_segment_resolution
# ---------------------------------------------------------------------


def test_check_segment_resolution_clean_world_returns_empty() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world, "rod", name="g1", position=(0, 0, 0.5), length=1.5,
        wire_radius=0.005,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.05)  # ratio = 10
    assert check_segment_resolution(world, eng) == []


def test_check_segment_resolution_thin_wire_warning() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world, "rod", name="g1", position=(0, 0, 0.5), length=1.5,
        wire_radius=0.05,            # bigger wire
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.05)  # ratio = 1.0
    msgs = check_segment_resolution(world, eng)
    assert any("thin-wire" in m and "g1" in m for m in msgs)


def test_check_segment_resolution_distributed_conductor_warning() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.0)
    gf.create_electrode(world, "rod", name="g2", position=(20, 0, 0.5), length=1.0)
    gf.create_conductor(
        world,
        name="bad_pen",
        start="g1", end="g2",
        conductor_type="pen",
        cross_section=50e-6,
        wire_radius=0.05,                  # 5 cm radius
        discretize_segment_length=0.1,     # ratio = 2 < 5
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.5)
    msgs = check_segment_resolution(world, eng)
    assert any("thin-wire" in m and "bad_pen" in m for m in msgs)


def test_check_segment_resolution_electrode_smaller_than_segment_warning() -> None:
    """A 0.3 m rod with segment_length = 1.0 should trigger the
    'smaller than one segment' warning."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.5), length=0.3,
                        wire_radius=0.005)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=1.0)
    msgs = check_segment_resolution(world, eng)
    assert any("resolution" in m and "g1" in m for m in msgs)


def test_check_segment_resolution_budget_warning() -> None:
    """A massive grid mesh with a fine segment length should cross
    the soft budget threshold."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world,
        "grid_mesh",
        name="big",
        corner=(0, 0, 0.7),
        size=(100.0, 100.0),
        n_x=20, n_y=20,
    )
    gf.create_source(world, attached_to="big", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.5)
    msgs = check_segment_resolution(world, eng)
    assert any("budget" in m for m in msgs)


# ---------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------


def test_top_level_exports_diagnostics() -> None:
    needed = {"world_statistics", "expected_segments", "check_segment_resolution"}
    assert needed.issubset(set(gf.__all__))
    assert all(hasattr(gf, name) for name in needed)
