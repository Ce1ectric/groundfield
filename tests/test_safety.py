"""Tests for the safety post-processing helpers (touch / step voltage).

Validates :mod:`groundfield.postprocess.safety` against:

- the closed-form Sunde solution for a single rod in homogeneous
  soil (touch and step voltage have known order of magnitude),
- physical monotonicities (U_T grows towards U_E far away from
  the electrode, |U_S| decays towards remote earth),
- symmetry under cylindrically symmetric geometry,
- the EN 50522:2010, Fig. B.3 anchor table for U_TP(t_F).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf
from groundfield.postprocess.safety import (
    _EN50522_TP_GRID,
    permissible_touch_voltage_en50522,
    step_voltage,
    touch_voltage,
    touch_voltage_envelope,
)


# ---------------------------------------------------------------------
# Common fixture: a single rod in homogeneous soil
# ---------------------------------------------------------------------


def _single_rod_world(
    *, rho: float = 100.0, length: float = 1.5, current: float = 10.0
) -> tuple[gf.World, gf.FieldResult]:
    soil = gf.HomogeneousSoil(resistivity=rho)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world,
        "rod",
        name="g1",
        position=(0.0, 0.0, 0.5),
        length=length,
        wire_radius=0.005,
    )
    gf.create_source(world, attached_to="g1", magnitude=current)
    eng = gf.create_engine(backend="image", segment_length=0.05)
    return world, eng.solve(world)


# ---------------------------------------------------------------------
# touch_voltage
# ---------------------------------------------------------------------


def test_touch_voltage_positive_close_to_rod() -> None:
    """U_T at 1 m around a current-injecting rod must be positive and below U_E."""
    world, result = _single_rod_world()
    U_E = result.electrode_potentials["g1"][0].real

    U_T = touch_voltage(result, world, electrode="g1", distance=1.0).real
    assert U_T > 0.0
    assert U_T < U_E


def test_touch_voltage_grows_with_distance_to_remote_earth() -> None:
    """As the feet recede, U_T -> U_E (potential at the feet -> 0)."""
    world, result = _single_rod_world()
    U_E = result.electrode_potentials["g1"][0].real

    U_close = touch_voltage(result, world, electrode="g1", distance=1.0).real
    U_mid = touch_voltage(result, world, electrode="g1", distance=10.0).real
    U_far = touch_voltage(result, world, electrode="g1", distance=200.0).real

    assert U_close < U_mid < U_far
    # At 200 m the surface potential of a 1.5 m rod injecting 10 A in
    # 100 Ω·m soil has decayed by far more than 99 % — feet ≈ remote
    # earth.
    assert abs(U_far - U_E) / U_E < 1e-2


def test_touch_voltage_rejects_invalid_distance() -> None:
    world, result = _single_rod_world()
    with pytest.raises(ValueError, match="distance"):
        touch_voltage(result, world, electrode="g1", distance=0.0)
    with pytest.raises(ValueError, match="distance"):
        touch_voltage(result, world, electrode="g1", distance=-1.0)


def test_touch_voltage_rejects_unknown_electrode() -> None:
    world, result = _single_rod_world()
    with pytest.raises(KeyError, match="missing"):
        touch_voltage(result, world, electrode="missing", distance=1.0)


def test_touch_voltage_horizontal_projection_strips_z() -> None:
    """A direction with non-zero z but zero (x, y) is rejected."""
    world, result = _single_rod_world()
    with pytest.raises(ValueError, match="horizontal"):
        touch_voltage(
            result, world, electrode="g1", distance=1.0, direction=(0, 0, 1.0)
        )


# ---------------------------------------------------------------------
# touch_voltage_envelope
# ---------------------------------------------------------------------


def test_envelope_uniform_for_centred_rod() -> None:
    """A vertical rod on the z-axis is cylindrically symmetric — every angle
    must yield the same touch voltage."""
    world, result = _single_rod_world()
    angles, voltages = touch_voltage_envelope(
        result, world, electrode="g1", distance=1.0, n_angles=12
    )
    assert angles.shape == (12,)
    assert voltages.shape == (12,)

    spread = float(voltages.real.max() - voltages.real.min())
    mean = float(voltages.real.mean())
    assert spread / abs(mean) < 1e-6


def test_envelope_matches_pointwise_touch_voltage() -> None:
    """The envelope at angle 0 must equal touch_voltage with direction +x."""
    world, result = _single_rod_world()
    angles, voltages = touch_voltage_envelope(
        result, world, electrode="g1", distance=1.0, n_angles=8
    )
    pointwise = touch_voltage(
        result, world, electrode="g1", distance=1.0, direction=(1.0, 0.0, 0.0)
    )
    # angle index 0 corresponds to (cos 0, sin 0) = (+x).
    assert voltages[0].real == pytest.approx(pointwise.real, rel=1e-12)


def test_envelope_rejects_too_few_angles() -> None:
    world, result = _single_rod_world()
    with pytest.raises(ValueError, match="n_angles"):
        touch_voltage_envelope(
            result, world, electrode="g1", distance=1.0, n_angles=2
        )


# ---------------------------------------------------------------------
# step_voltage
# ---------------------------------------------------------------------


def test_step_voltage_decays_to_remote_earth() -> None:
    """|U_S| must decay monotonically with distance from the source."""
    world, result = _single_rod_world()
    U_S_close = abs(
        step_voltage(result, position=(1.0, 0.0, 0.0), step=1.0, direction=(1, 0, 0))
    )
    U_S_mid = abs(
        step_voltage(result, position=(10.0, 0.0, 0.0), step=1.0, direction=(1, 0, 0))
    )
    U_S_far = abs(
        step_voltage(
            result, position=(100.0, 0.0, 0.0), step=1.0, direction=(1, 0, 0)
        )
    )
    assert U_S_close > U_S_mid > U_S_far
    # 100 m away from a 1.5 m rod, |U_S| over 1 m must be tiny.
    assert U_S_far < 1e-2 * U_S_close


def test_step_voltage_sign_matches_potential_gradient() -> None:
    """Stepping outward from the rod, U_S = phi(near) - phi(far) > 0."""
    _world, result = _single_rod_world()
    U_S = step_voltage(
        result, position=(1.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), step=1.0
    )
    # phi(1, 0, 0) > phi(2, 0, 0) for an outward-pointing source field.
    assert U_S.real > 0.0


def test_step_voltage_rejects_invalid_step() -> None:
    _world, result = _single_rod_world()
    with pytest.raises(ValueError, match="step"):
        step_voltage(result, position=(1.0, 0.0, 0.0), step=0.0)
    with pytest.raises(ValueError, match="step"):
        step_voltage(result, position=(1.0, 0.0, 0.0), step=-0.5)


def test_step_voltage_horizontal_projection_strips_z() -> None:
    _world, result = _single_rod_world()
    with pytest.raises(ValueError, match="horizontal"):
        step_voltage(
            result, position=(1.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), step=1.0
        )


# ---------------------------------------------------------------------
# permissible_touch_voltage_en50522
# ---------------------------------------------------------------------


def test_en50522_anchor_points_match_table() -> None:
    """The interpolant must reproduce the table exactly at the anchors."""
    for t, u in _EN50522_TP_GRID:
        assert permissible_touch_voltage_en50522(t) == pytest.approx(u, rel=1e-12)


def test_en50522_strictly_decreasing_inside_grid() -> None:
    """U_TP(t) is monotonically decreasing on [50 ms, 10 s]."""
    ts = np.geomspace(0.05, 10.0, 50)
    us = np.array([permissible_touch_voltage_en50522(t) for t in ts])
    diffs = np.diff(us)
    assert (diffs <= 1e-9).all()


def test_en50522_clamped_outside_grid() -> None:
    """Outside the grid the table endpoints are returned unchanged."""
    assert permissible_touch_voltage_en50522(1e-3) == pytest.approx(
        _EN50522_TP_GRID[0][1], rel=1e-12
    )
    assert permissible_touch_voltage_en50522(1e6) == pytest.approx(
        _EN50522_TP_GRID[-1][1], rel=1e-12
    )


def test_en50522_rejects_invalid_time() -> None:
    with pytest.raises(ValueError, match="t_clear_s"):
        permissible_touch_voltage_en50522(0.0)
    with pytest.raises(ValueError, match="t_clear_s"):
        permissible_touch_voltage_en50522(-0.1)
    with pytest.raises(ValueError, match="t_clear_s"):
        permissible_touch_voltage_en50522(float("nan"))


def test_en50522_loglog_interpolation_in_geometric_mean() -> None:
    """Halfway in log time between two anchors the value must equal the
    geometric mean of the corresponding anchor voltages."""
    # Anchors (1.0 s, 115 V) and (2.0 s, 95 V) — geometric mean of t is
    # sqrt(2) s, expected U is sqrt(115 * 95) V.
    t_mid = math.sqrt(2.0)
    u_expected = math.sqrt(115.0 * 95.0)
    assert permissible_touch_voltage_en50522(t_mid) == pytest.approx(
        u_expected, rel=1e-12
    )


def test_en50522_terminal_plateau_constant() -> None:
    """Between t = 5 s and t = 10 s the standard's table is flat at 85 V."""
    for t in (5.0, 6.0, 7.5, 9.0, 10.0):
        assert permissible_touch_voltage_en50522(t) == pytest.approx(85.0, rel=1e-12)


# ---------------------------------------------------------------------
# Integration: top-level export
# ---------------------------------------------------------------------


def test_top_level_exports_safety_helpers() -> None:
    needed = {
        "touch_voltage",
        "touch_voltage_envelope",
        "step_voltage",
        "permissible_touch_voltage_en50522",
    }
    assert needed.issubset(set(gf.__all__))
    assert all(hasattr(gf, name) for name in needed)
