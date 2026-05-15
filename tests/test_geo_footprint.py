"""Tests for :mod:`groundfield.geo.footprint`.

Covers the :class:`BuildingFootprint` Pydantic model plus its
geometric helpers (``signed_area``, ``ensure_orientation``,
``area_m2``, ``centroid_xy_m``, axis-aligned bounding rectangle,
oriented minimum bounding rectangle). The OMBR test is the
key Phase-A validation for ADR-0011: an arbitrary L-shape's
OMBR must come out as the dominant edge rectangle, and a
known-rotated rectangle must round-trip its rotation angle to
within numerical noise.
"""

from __future__ import annotations

import math

import pytest

from groundfield.geo.footprint import (
    BuildingFootprint,
    ensure_orientation,
    signed_area,
)


# ---------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------


def test_signed_area_ccw_positive() -> None:
    ring = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    assert signed_area(ring) == pytest.approx(12.0)


def test_signed_area_cw_negative() -> None:
    ring = [(0.0, 0.0), (0.0, 3.0), (4.0, 3.0), (4.0, 0.0)]
    assert signed_area(ring) == pytest.approx(-12.0)


def test_signed_area_degenerate_is_zero() -> None:
    assert signed_area([(0.0, 0.0), (1.0, 1.0)]) == 0.0
    assert signed_area([]) == 0.0


def test_ensure_orientation_flips_cw_to_ccw() -> None:
    ring = [(0.0, 0.0), (0.0, 3.0), (4.0, 3.0), (4.0, 0.0)]
    out = ensure_orientation(ring, ccw=True)
    assert signed_area(out) > 0


def test_ensure_orientation_strips_duplicate_closing_vertex() -> None:
    ring = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
    out = ensure_orientation(ring, ccw=True)
    assert out[0] != out[-1]


# ---------------------------------------------------------------------
# BuildingFootprint
# ---------------------------------------------------------------------


def test_building_footprint_normalises_ring_orientation() -> None:
    # Input is CW; the model must normalise to CCW so the area is positive.
    fp = BuildingFootprint(
        polygon_xy_m=[(0.0, 0.0), (0.0, 8.0), (10.0, 8.0), (10.0, 0.0)],
    )
    assert signed_area(fp.polygon_xy_m) > 0
    assert fp.area_m2() == pytest.approx(80.0)


def test_building_footprint_rejects_degenerate_ring() -> None:
    with pytest.raises(Exception):
        BuildingFootprint(polygon_xy_m=[(0.0, 0.0), (1.0, 1.0)])


def test_building_footprint_centroid_of_rectangle() -> None:
    fp = BuildingFootprint(
        polygon_xy_m=[(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
    )
    cx, cy = fp.centroid_xy_m()
    assert (cx, cy) == pytest.approx((5.0, 4.0))


def test_building_footprint_area_subtracts_holes() -> None:
    fp = BuildingFootprint(
        polygon_xy_m=[(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
        holes_xy_m=[
            # CW hole — must be normalised to CW on input.
            [(2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)],
        ],
    )
    # Outer area 80 m² minus a 4 m² hole = 76 m².
    assert fp.area_m2() == pytest.approx(76.0)


def test_building_footprint_is_frozen() -> None:
    fp = BuildingFootprint(
        polygon_xy_m=[(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)],
    )
    with pytest.raises(Exception):
        fp.osm_id = 42


# ---------------------------------------------------------------------
# Bounding rectangles
# ---------------------------------------------------------------------


def test_aabb_of_axis_aligned_rectangle() -> None:
    fp = BuildingFootprint(
        polygon_xy_m=[(2.0, 3.0), (12.0, 3.0), (12.0, 9.0), (2.0, 9.0)],
    )
    centre, size = fp.axis_aligned_bounding_rectangle()
    assert centre == pytest.approx((7.0, 6.0))
    assert size == pytest.approx((10.0, 6.0))


def test_ombr_axis_aligned_rectangle_matches_aabb() -> None:
    fp = BuildingFootprint(
        polygon_xy_m=[(0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)],
    )
    centre, size, angle_deg = fp.oriented_bounding_rectangle()
    assert centre == pytest.approx((5.0, 3.0))
    # The long side (10 m) should be reported first.
    assert size[0] == pytest.approx(10.0)
    assert size[1] == pytest.approx(6.0)
    # An axis-aligned rectangle has a long axis along +x (angle = 0).
    assert abs(angle_deg) < 1e-6


def test_ombr_round_trips_rotation_angle() -> None:
    r"""A known 10 x 6 rectangle rotated by 37 deg must report
    ``angle_deg == 37`` and the same side lengths.

    This is the canonical AP1 use case: the footprint of a house
    that does not run E-W. The OMBR's principal angle is what
    :class:`FoundationElectrodeSpec.orientation_deg` is set to in
    the OSM-driven build path.
    """
    angle = 37.0
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    rect_local = [(-5.0, -3.0), (5.0, -3.0), (5.0, 3.0), (-5.0, 3.0)]
    rotated = [(c * x - s * y, s * x + c * y) for x, y in rect_local]
    fp = BuildingFootprint(polygon_xy_m=rotated)
    (cx, cy), (dx, dy), angle_deg = fp.oriented_bounding_rectangle()
    assert (cx, cy) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert (dx, dy) == pytest.approx((10.0, 6.0), abs=1e-6)
    assert angle_deg == pytest.approx(37.0, abs=1e-4)


def test_ombr_handles_l_shape() -> None:
    """OMBR of an L-shape must yield a rectangle that contains the
    L and whose long axis aligns with the dominant edge.

    The L below is 12 m along x by 8 m along y; the inner corner
    is at (8, 4). The OMBR is therefore the 12 x 8 rectangle and
    its area is 96 m² (vs. the L's actual area of 80 m²).
    """
    l_shape = [
        (0.0, 0.0), (12.0, 0.0), (12.0, 4.0),
        (8.0, 4.0), (8.0, 8.0), (0.0, 8.0),
    ]
    fp = BuildingFootprint(polygon_xy_m=l_shape)
    _, (dx, dy), angle_deg = fp.oriented_bounding_rectangle()
    assert (dx, dy) == pytest.approx((12.0, 8.0), abs=1e-6)
    # Long axis aligns with the world +x (or +y after 90-deg flip).
    # Both 0 and ±90 are valid representations of the same rectangle;
    # the convention in the implementation maps to [-90, 90], so
    # for this L the angle is exactly 0.
    assert abs(angle_deg) < 1e-6
    assert fp.area_m2() == pytest.approx(80.0)
