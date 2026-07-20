"""Tests for :class:`~groundfield.geometry.electrodes.StarElectrode`.

Covers
------
* geometry: radial arms, tips, total wire length, hub connection point,
* the ``n_arms >= 2`` guard,
* a single galvanic cluster with a monotone ``R(n_arms)``,
* rotational invariance of the grounding resistance in homogeneous soil,
* plot / VTK smoke, so the new electrode kind is handled by every
  geometry consumer.

The quantitative Dwight (1936) validation lives in
``test_dwight_references.py::test_image_n_point_star_vs_dwight``.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

import groundfield as gf  # noqa: E402
from groundfield.geometry.electrodes import StarElectrode  # noqa: E402


def _solve_star(
    n_arms: int, orientation_deg: float = 0.0, arm_length: float = 4.0
) -> float:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world,
        "star",
        name="g1",
        center=(0.0, 0.0, 0.5),
        n_arms=n_arms,
        arm_length=arm_length,
        wire_radius=0.006,
        orientation_deg=orientation_deg,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    engine = gf.create_engine(backend="image", segment_length=0.2, frequencies=[50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = world.solve(engine)
    return res.cluster_impedance("g1")[0].real


def test_star_geometry() -> None:
    s = StarElectrode(
        name="g1",
        wire_radius=0.01,
        center=(0.0, 0.0, 0.5),
        n_arms=4,
        arm_length=5.0,
    )
    assert len(s.edges) == 4
    assert s.length == pytest.approx(20.0)
    assert s.connection_point == (0.0, 0.0, 0.5)
    tips = s.arm_tips
    assert tips[0] == pytest.approx((5.0, 0.0, 0.5))
    assert tips[1] == pytest.approx((0.0, 5.0, 0.5))
    # Every arm starts at the shared hub.
    assert all(start == (0.0, 0.0, 0.5) for start, _ in s.edges)


def test_star_requires_at_least_two_arms() -> None:
    with pytest.raises(ValidationError):
        StarElectrode(
            name="g1",
            wire_radius=0.01,
            center=(0.0, 0.0, 0.5),
            n_arms=1,
            arm_length=5.0,
        )


def test_star_more_arms_lower_resistance() -> None:
    # More radial copper spreads the current -> lower grounding resistance.
    assert _solve_star(8) < _solve_star(4)


def test_star_resistance_is_rotation_invariant() -> None:
    # A rigid rotation about the vertical axis leaves the grounding
    # resistance of a symmetric star unchanged in homogeneous soil.
    assert _solve_star(4, orientation_deg=37.0) == pytest.approx(
        _solve_star(4, orientation_deg=0.0), rel=1e-6
    )


def test_star_plot_and_vtk_smoke(tmp_path) -> None:
    from groundfield.io.vtk import export_geometry_vtk
    from groundfield.postprocess.geometry_plot import (
        plot_world,
        plot_world_3d,
    )

    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world,
        "star",
        name="g1",
        center=(0.0, 0.0, 0.5),
        n_arms=6,
        arm_length=4.0,
        wire_radius=0.006,
    )
    plot_world(world)
    plot_world_3d(world)
    out = export_geometry_vtk(world, tmp_path / "star.vtk")
    assert out.exists() and out.stat().st_size > 0
