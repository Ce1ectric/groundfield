"""Rotated mesh foundations and the canonical OMBR angle (10.39).

A rotated (or distributed-shell) mesh foundation used to be built from
independent full-length strips. When two crossing wires both had an odd
segment count, their middle-segment midpoints coincided at the crossing:
the reaction matrix hit the 1 mm singularity clamp and the impedance
jumped by about 10 % at ds = 2.0 / 1.5 m. The wires are now split at the
crossings, so crossings fall on segment ends as in ``GridMeshElectrode``.

GEOS reports axis-aligned squares (and rectangles that are long in y) at
+90 deg, which sent them to the rotated path while the same rectangle at
0 deg used the grid-mesh path. The OMBR angle of a square is now chosen
canonically, and a foundation at a multiple of 90 deg is realised as the
axis-aligned one with swapped sides. Since audit A2G-04 (meshes) and user
decision Q38 (rectangles without cross-braces) no foundation uses a grid
mesh any more: both are one wire per line at every angle.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import FoundationElectrodeSpec, GroundingSystemSpec

pytestmark = pytest.mark.finding("10.39")


def _solve(
    spec: FoundationElectrodeSpec, ds: float
) -> tuple[float, list[str], gf.World]:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    anchor = GroundingSystemSpec(electrodes=[spec]).build_at(
        world, site_xy=(0.0, 0.0), name_prefix="h", rng=np.random.default_rng(0)
    )
    gf.create_source(world, attached_to=anchor, magnitude=1.0)
    engine = gf.create_engine(backend="image", segment_length=ds, frequencies=[50.0])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = engine.solve(world)
    clamps = [str(w.message) for w in rec if "clamp" in str(w.message)]
    return res.cluster_impedance(anchor)[0].real, clamps, world


@pytest.mark.parametrize("angle", [30.0, -63.0])
def test_rotated_mesh_has_no_clamp_and_converges_smoothly(angle: float) -> None:
    spec = FoundationElectrodeSpec(
        size_xy_m=(10.0, 10.0), orientation_deg=angle, style="mesh", n_x=2, n_y=2
    )
    z = {}
    for ds in (2.5, 2.0, 1.5, 1.25, 1.0, 0.5):
        z[ds], clamps, _ = _solve(spec, ds)
        assert clamps == [], f"ds={ds}: {clamps}"
    # v0.15.0: +11.7 % at ds = 2.0 and +11.0 % at ds = 1.5 against ds = 0.5.
    for ds, value in z.items():
        assert abs(value / z[0.5] - 1.0) < 0.01, (ds, value, z[0.5])


def test_distributed_shell_mesh_at_zero_degrees_has_no_clamp() -> None:
    # An encased mesh at 0 deg: one wire per line (since audit A2G-04 at
    # every angle and for both values; concrete_model has no effect, user
    # decision Q38).
    spec = FoundationElectrodeSpec(
        size_xy_m=(10.0, 10.0),
        style="mesh",
        n_x=2,
        n_y=2,
        concrete_rho_ohm_m=150.0,
        concrete_model="distributed",
    )
    _, clamps, _ = _solve(spec, 2.0)
    assert clamps == []


def test_rotated_mesh_segments_never_meet_at_crossings() -> None:
    from groundfield.solver.image import _discretize_electrode

    spec = FoundationElectrodeSpec(
        size_xy_m=(10.0, 10.0), orientation_deg=30.0, style="mesh", n_x=2, n_y=2
    )
    _, _, world = _solve(spec, 2.0)
    mids = np.array(
        [s.midpoint for e in world.electrodes for s in _discretize_electrode(e, 2.0)]
    )
    d = np.linalg.norm(mids[:, None, :] - mids[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 0.5


def test_rotated_mesh_matches_grid_mesh_segment_counts() -> None:
    """Each mesh span gets the grid-mesh segment count.

    Step sizes are chosen so that no span length is an integer multiple
    of ``ds``: rotated vertices carry floating-point noise, and the
    solver's ``ceil(L / ds)`` then adds a segment for an edge of exactly
    ``k * ds`` (reported to the solver stream).
    """
    from groundfield.solver.image import _discretize_electrode

    for ds in (2.2, 1.7, 1.1, 0.7):
        rotated = FoundationElectrodeSpec(
            size_xy_m=(12.0, 9.0), orientation_deg=30.0, style="mesh", n_x=2, n_y=3
        )
        axis = FoundationElectrodeSpec(
            size_xy_m=(12.0, 9.0), style="mesh", n_x=2, n_y=3
        )
        counts = []
        for spec in (rotated, axis):
            _, _, world = _solve(spec, ds)
            counts.append(
                sum(len(_discretize_electrode(e, ds)) for e in world.electrodes)
            )
        assert counts[0] == counts[1], (ds, counts)


def test_ring_style_rotated_foundation_keeps_its_four_strips() -> None:
    spec = FoundationElectrodeSpec(
        size_xy_m=(12.0, 9.0), orientation_deg=30.0, style="ring"
    )
    _, _, world = _solve(spec, 1.0)
    assert [e.kind for e in world.electrodes] == ["strip"] * 4


# ---------------------------------------------------------------------
# Canonical angle
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "ring",
    [
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [(10, 0), (10, 10), (0, 10), (0, 0)],
        [(0, 0), (0, 10), (10, 10), (10, 0)],
        [(0, 10), (0, 0), (10, 0), (10, 10)],
    ],
)
def test_ombr_angle_of_an_axis_aligned_square_is_zero(ring: list) -> None:
    (cx, cy), (dx, dy), angle = gf.BuildingFootprint(
        polygon_xy_m=ring
    ).oriented_bounding_rectangle()
    assert (cx, cy, dx, dy) == pytest.approx((5.0, 5.0, 10.0, 10.0))
    assert angle == 0.0


def test_ombr_of_a_rotated_square_is_in_the_canonical_range() -> None:
    c, s = np.cos(np.radians(70.0)), np.sin(np.radians(70.0))
    ring = [
        (c * x - s * y, s * x + c * y) for x, y in [(-4, -4), (4, -4), (4, 4), (-4, 4)]
    ]
    _, _, angle = gf.BuildingFootprint(polygon_xy_m=ring).oriented_bounding_rectangle()
    assert angle == pytest.approx(-20.0)


def _world_electrodes(spec: FoundationElectrodeSpec) -> str:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    GroundingSystemSpec(electrodes=[spec]).build_at(
        world, site_xy=(3.0, 4.0), name_prefix="h", rng=np.random.default_rng(0)
    )
    return json.dumps(
        [e.model_dump(mode="json") for e in world.electrodes], sort_keys=True
    )


def _edges(spec: FoundationElectrodeSpec) -> set[frozenset]:
    """Every straight wire piece of the generated foundation, unordered."""
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    GroundingSystemSpec(electrodes=[spec]).build_at(
        world, site_xy=(3.0, 4.0), name_prefix="h", rng=np.random.default_rng(0)
    )
    out = set()
    for e in world.electrodes:
        (run,) = e.wire_polylines()
        for a, b in zip(run[:-1], run[1:], strict=True):
            out.add(frozenset((tuple(a), tuple(b))))
    return out


def _anchor_start(spec: FoundationElectrodeSpec) -> tuple[float, float, float]:
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    anchor = GroundingSystemSpec(electrodes=[spec]).build_at(
        world, site_xy=(0.0, 0.0), name_prefix="h", rng=np.random.default_rng(0)
    )
    return world.get_electrode(anchor).connection_point


# user decision Q41 (item 5), audit D3B-04
@pytest.mark.parametrize("style", ["mesh", "ring"])
@pytest.mark.parametrize("angle", [90.0, -90.0, 180.0, 270.0, 450.0])
def test_foundation_at_right_angles_is_the_rotated_rectangle_without_round_off(
    angle: float, style: str
) -> None:
    """Up to the audit a multiple of 90 deg was realised as the 0-deg
    rectangle with swapped sides: the same wires, but another side became
    the anchor, so its connection point jumped (20 m x 5 m ring: (2.483,
    -10.004) m at 89.9 deg, (-2.5, -10) m at 90 deg; 20.6 m between 179.9
    and 180 deg). Now it is rotated with the exact cos / sin: the wires are
    those of the swapped rectangle (to 1e-12 m, exactly for a ring), the
    local -y side stays the anchor, and its start is continuous in the
    angle."""
    turned = FoundationElectrodeSpec(
        size_xy_m=(10.0, 12.0), orientation_deg=angle, style=style, n_x=3, n_y=2
    )
    swapped = FoundationElectrodeSpec(
        size_xy_m=(12.0, 10.0) if angle % 180 else (10.0, 12.0),
        style=style,
        n_x=2 if angle % 180 else 3,
        n_y=3 if angle % 180 else 2,
    )
    a, b = _edges(turned), _edges(swapped)
    if style == "ring":
        assert a == b
    else:
        assert len(a) == len(b)
        rounded = {frozenset(tuple(np.round(p, 9)) for p in e) for e in a}
        assert rounded == {frozenset(tuple(np.round(p, 9)) for p in e) for e in b}
    near = FoundationElectrodeSpec(
        size_xy_m=(10.0, 12.0), orientation_deg=angle - 1e-7, style=style, n_x=3, n_y=2
    )
    assert math.dist(_anchor_start(turned), _anchor_start(near)) < 1e-6


# user decision Q41 (item 5), audit D3B-04
@pytest.mark.parametrize(
    ("angle", "start"),
    [
        (90.0, (2.5, -10.0)),
        (180.0, (10.0, 2.5)),
        (270.0, (-2.5, 10.0)),
        (-90.0, (-2.5, 10.0)),
    ],
)
def test_right_angle_coordinates_are_exact(angle: float, start: tuple) -> None:
    """The anchor of a 20 m x 5 m ring starts exactly at the rotated local
    (-10, -2.5) m corner; the other side ends agree without round-off."""
    spec = FoundationElectrodeSpec(
        style="ring", size_xy_m=(20.0, 5.0), depth_m=0.8, orientation_deg=angle
    )
    assert _anchor_start(spec) == (*start, 0.8)
    corners = {p for e in _edges(spec) for p in e}
    expected = {
        (x + 3.0, y + 4.0, 0.8)
        for x in (-10.0, 10.0, -2.5, 2.5)
        for y in (-10.0, 10.0, -2.5, 2.5)
    }
    assert corners <= expected and len(corners) == 4


# user decision Q41 (item 5)
@pytest.mark.parametrize("style", ["mesh", "ring"])
def test_right_angles_give_the_impedance_of_the_swapped_rectangle(style: str) -> None:
    turned = FoundationElectrodeSpec(
        size_xy_m=(10.0, 12.0), orientation_deg=90.0, style=style, n_x=3, n_y=2
    )
    swapped = FoundationElectrodeSpec(size_xy_m=(12.0, 10.0), style=style, n_x=2, n_y=3)
    z_turned, clamps, _ = _solve(turned, 1.0)
    z_swapped, _, _ = _solve(swapped, 1.0)
    assert not clamps
    assert z_turned == pytest.approx(z_swapped, rel=1e-12)


def test_axis_aligned_footprint_square_gives_an_axis_aligned_foundation() -> None:
    from groundfield.generators.tn_network import _override_foundation_from_footprint

    fp = gf.BuildingFootprint(polygon_xy_m=[(0, 0), (10, 0), (10, 10), (0, 10)])
    spec = GroundingSystemSpec(electrodes=[FoundationElectrodeSpec(style="mesh")])
    shaped = _override_foundation_from_footprint(spec, fp, (5.0, 5.0))
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    shaped.build_at(
        world, site_xy=(5.0, 5.0), name_prefix="h", rng=np.random.default_rng(0)
    )
    # One owner per wire (audit A2G-04), every wire along x or y.
    assert [e.kind for e in world.electrodes] == ["polyline"] * 6
    for e in world.electrodes:
        (x0, y0, _), (x1, y1, _) = e.vertices[0], e.vertices[-1]
        assert x0 == x1 or y0 == y1
