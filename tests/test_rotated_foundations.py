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
canonically, and a foundation at a multiple of 90 deg is realised on the
grid-mesh path.
"""

from __future__ import annotations

import json
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
    # A distributed concrete shell forces the per-wire path even at 0 deg.
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


@pytest.mark.parametrize("angle", [90.0, -90.0])
def test_foundation_at_right_angles_uses_the_grid_mesh_path(angle: float) -> None:
    turned = FoundationElectrodeSpec(
        size_xy_m=(10.0, 12.0), orientation_deg=angle, style="mesh", n_x=3, n_y=2
    )
    straight = FoundationElectrodeSpec(
        size_xy_m=(12.0, 10.0), style="mesh", n_x=2, n_y=3
    )
    assert _world_electrodes(turned) == _world_electrodes(straight)


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
