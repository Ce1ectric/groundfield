"""Degenerate electrode geometry is rejected at construction (review pass 10, 10.67).

Up to 0.15.0 a zero-length strip, a negative mesh size or a wire radius
typed in millimetres reached the solver and failed there as a thin-wire
violation of one segment (or as an ``OverflowError`` for infinite
lengths); a NaN coordinate solved silently to ``Z = nan``. The electrode
models now reject such input when they are built.
"""

from __future__ import annotations

import math

import pytest

import groundfield as gf

_FINDING = pytest.mark.finding("10.67")


@_FINDING
def test_zero_length_strip_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero length"):
        gf.StripElectrode(name="s", start=(1.0, 2.0, 0.5), end=(1.0, 2.0, 0.5))


@_FINDING
@pytest.mark.parametrize("size", [(-4.0, 4.0), (4.0, 0.0), (0.0, 0.0)])
def test_non_positive_grid_mesh_size_is_rejected(size) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        gf.GridMeshElectrode(name="g", corner=(0.0, 0.0, 0.5), size=size, n_x=2, n_y=2)


@_FINDING
def test_non_positive_mesh_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        gf.MeshElectrode(name="m", corner=(0.0, 0.0, 0.5), size=(0.0, 4.0))


@_FINDING
@pytest.mark.parametrize(
    "build",
    [
        lambda: gf.RodElectrode(name="r", position=(math.nan, 0.0, 0.5), length=2.0),
        lambda: gf.RodElectrode(name="r", position=(0.0, 0.0, 0.5), length=math.inf),
        lambda: gf.RingElectrode(name="r", center=(0.0, math.inf, 0.8), radius=2.0),
        lambda: gf.PolylineElectrode(
            name="p", vertices=[(0.0, 0.0, 0.5), (math.inf, 0.0, 0.5), (0.0, 3.0, 0.5)]
        ),
        lambda: gf.StarElectrode(
            name="s", center=(0.0, 0.0, 0.5), n_arms=3, arm_length=2.0,
            orientation_deg=math.nan,
        ),
        lambda: gf.RodElectrode(
            name="r", position=(0.0, 0.0, 0.5), length=2.0, wire_radius=math.inf
        ),
    ],
    ids=["rod-nan", "rod-inf-length", "ring-inf", "polyline-inf", "star-nan", "radius-inf"],
)  # fmt: skip
def test_non_finite_values_are_rejected(build) -> None:
    with pytest.raises(ValueError, match="finite"):
        build()


@_FINDING
@pytest.mark.parametrize(
    "build",
    [
        # wire radius typed in mm (5 instead of 0.005) on a 2 m ring
        lambda: gf.RingElectrode(
            name="g", center=(0.0, 0.0, 0.8), radius=2.0, wire_radius=5.0
        ),
        lambda: gf.RodElectrode(
            name="g", position=(0.0, 0.0, 0.5), length=0.004, wire_radius=0.005
        ),
        lambda: gf.StripElectrode(
            name="g", start=(0.0, 0.0, 0.5), end=(0.003, 0.0, 0.5), wire_radius=0.005
        ),
        lambda: gf.StarElectrode(
            name="g", center=(0.0, 0.0, 0.5), n_arms=3, arm_length=0.004
        ),
        lambda: gf.PolylineElectrode(
            name="g",
            vertices=[(0.0, 0.0, 0.5), (3.0, 0.0, 0.5), (3.0, 0.004, 0.5)],
        ),
        lambda: gf.GridMeshElectrode(
            name="g", corner=(0.0, 0.0, 0.5), size=(0.01, 4.0), n_x=4, n_y=2
        ),
        lambda: gf.MeshElectrode(
            name="g", corner=(0.0, 0.0, 0.5), size=(4.0, 4.0), spacing=0.004
        ),
    ],
    ids=["ring", "rod", "strip", "star", "polyline", "grid_mesh", "mesh"],
)
def test_dimension_not_longer_than_the_wire_radius_is_rejected(build) -> None:
    with pytest.raises(ValueError, match="wire_radius"):
        build()


@_FINDING
def test_create_electrode_reports_the_problem_before_the_solve() -> None:
    world = gf.create_world()
    with pytest.raises(ValueError, match="zero length"):
        gf.create_electrode(
            world, "strip", name="s", start=(0.0, 0.0, 0.5), end=(0.0, 0.0, 0.5)
        )
    assert world.electrodes == []


def test_valid_geometries_still_build() -> None:
    """Guard: ordinary electrodes, including small and thick ones, are accepted."""
    gf.RodElectrode(name="r", position=(0.0, 0.0, 0.5), length=0.05)
    gf.RingElectrode(name="g", center=(0.0, 0.0, 0.8), radius=0.06, wire_radius=0.05)
    gf.StripElectrode(name="s", start=(0.0, 0.0, 0.5), end=(10.0, 0.0, 0.5))
    gf.StarElectrode(name="t", center=(0.0, 0.0, 0.5), n_arms=3, arm_length=3.0)
    gf.GridMeshElectrode(
        name="m", corner=(0.0, 0.0, 0.5), size=(10.0, 10.0), n_x=2, n_y=2
    )
    gf.MeshElectrode(name="n", corner=(0.0, 0.0, 0.5), size=(10.0, 4.0), spacing=2.0)
    gf.PolylineElectrode(
        name="p", vertices=[(0.0, 0.0, 0.5), (12.0, 0.0, 0.5), (12.0, 9.0, 0.5)]
    )
