"""Short segments and interface slivers (review pass 10, 10.32).

* The collocation self term :math:`2\\ln(L/a)/L` of the point kernel tends to
  zero as :math:`L \\to a`: a sliver left by the split at a layer interface
  acted as a short, so ``mom`` gave a rod whose tip lies 9 mm below
  :math:`h_1` 27 % less resistance than the rod 1 mm above it, and a rod
  whose tip lies 1 mm below :math:`h_1` raised a ``ValueError``. Below
  :math:`L/a = 2.99` the Galerkin self term takes over, and an interface cut
  that would leave a part shorter than the wire diameter is dropped.
* ``ceil(L / ds)`` added a segment when ``L / ds`` exceeded an integer only
  through floating-point noise (GEN-1: 52 instead of 50 segments).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

_FINDING = pytest.mark.finding("10.32")
_SOIL = dict(rho_1=1000.0, rho_2=100.0, h_1=5.0)


def _rod_z(bottom: float, backend: str, **engine_kw) -> float:
    w = gf.create_world(soil=gf.TwoLayerSoil(**_SOIL))
    gf.create_electrode(
        w, "rod", name="r", position=(0, 0, 0.5), length=bottom - 0.5, wire_radius=0.008
    )
    gf.create_source(w, attached_to="r", magnitude=1.0)
    eng = gf.create_engine(backend=backend, segment_length=0.5, **engine_kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return w.solve(eng).cluster_impedance("r")[0].real


@_FINDING
def test_rod_tip_just_below_the_interface_solves() -> None:
    """A tip 1 mm below h_1 raised (segment shorter than the wire radius)."""
    z = _rod_z(5.001, "image_2layer")
    # Same model as the rod ending at h_1 (10 instead of 9 segments).
    assert z == pytest.approx(_rod_z(5.0, "image_2layer"), rel=5e-3)


@_FINDING
def test_mom_resistance_decreases_with_rod_length_across_the_interface() -> None:
    """R1-04: mom gave 196.6 -> 144.2 -> 166.8 Ohm for bottoms 4.99 / 5.009 / 5.02 m."""
    bottoms = [4.99, 5.009, 5.02, 5.05, 5.2]
    z = [_rod_z(b, "mom") for b in bottoms]
    assert all(b < a for a, b in zip(z, z[1:], strict=False)), z


@pytest.mark.parametrize("kernel", ["point", "line"])
def test_image_resistance_decreases_with_rod_length_across_the_interface(
    kernel: str,
) -> None:
    bottoms = [4.99, 5.001, 5.009, 5.02, 5.05, 5.2]
    z = [_rod_z(b, "image_2layer", kernel=kernel) for b in bottoms]
    assert all(b < a for a, b in zip(z, z[1:], strict=False)), z


@_FINDING
def test_point_self_term_takes_the_galerkin_form_for_short_segments() -> None:
    from groundfield.solver._line_kernel import (
        SELF_TERM_CROSSOVER,
        galerkin_self,
        point_self_term,
    )

    a = np.full(7, 0.008)
    ratio = np.array([1.05, 1.13, 1.5, 2.0, 2.5, 2.9, 2.98])
    L = ratio * a
    np.testing.assert_array_equal(point_self_term(L, a), galerkin_self(L, a))
    long_ratio = np.array([2.991, 3.0, 5.0, 62.5, 1e3])
    L = long_ratio * a[:5]
    np.testing.assert_array_equal(
        point_self_term(L, a[:5]), 2.0 * np.log(L / a[:5]) / L
    )
    # continuous at the switch
    a0 = np.array([0.01])
    below = point_self_term(np.array([SELF_TERM_CROSSOVER * (1 - 1e-9)]) * a0, a0)
    above = point_self_term(np.array([SELF_TERM_CROSSOVER * (1 + 1e-9)]) * a0, a0)
    assert below[0] == pytest.approx(above[0], rel=1e-8)


@_FINDING
def test_sliver_threshold_is_the_wire_diameter() -> None:
    """Parts shorter than 2 wire radii are not split off; longer ones are."""
    a, h = 0.008, (5.0,)
    short = gf.RodElectrode(
        name="r", position=(0, 0, 0.5), length=4.5 + 1.9 * a, wire_radius=a
    )
    kept = gf.RodElectrode(
        name="r", position=(0, 0, 0.5), length=4.5 + 2.1 * a, wire_radius=a
    )
    z_short = short.segments(0.5, h).midpoints[:, 2]
    z_kept = kept.segments(0.5, h).midpoints[:, 2]
    assert (z_short < 5.0).all()  # one part, all midpoints above h_1
    assert (z_kept >= 5.0).sum() == 1  # the 2.1 a part is its own segment
    assert kept.segments(0.5, h).lengths.min() == pytest.approx(2.1 * a)


@_FINDING
def test_segment_count_ignores_floating_point_noise() -> None:
    # Star arms at 120 / 240 degrees: 2.9999999999999996 and 3.000000000000001 m.
    star = gf.StarElectrode(name="s", center=(0, 0, 0.6), n_arms=3, arm_length=3.0)
    assert star.segment_count(0.5) == 18
    edge = 3.0000000000000004
    poly = gf.PolylineElectrode(
        name="p", vertices=[(0, 0, 0.5), (edge, 0, 0.5)], closed=False
    )
    assert poly.segment_count(1.5) == 2
    rod = gf.RodElectrode(name="r", position=(0, 0, 0.5), length=0.1 + 0.2)
    assert rod.segment_count(0.1) == 3
    assert math.isclose(rod.segments(0.1).lengths[0], 0.1)
