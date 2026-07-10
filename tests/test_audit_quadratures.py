"""Regression tests for the WP-B quadrature fixes (audit 2026-07-08).

Covers the two measurably wrong integrals found by the audit:

* **N3 — Carson intermediate regime.** The historic 64-node
  Gauss–Legendre on ``[0, 30/p]`` failed for near-perpendicular
  geometries (buried-pair mutuals, small ``p``): measured errors up
  to −36 % (P) / +67 % (Q). The oscillatory path now uses QUADPACK's
  Fourier integrator; reference values below were computed with an
  independent composite per-half-period Gauss rule plus Euler
  acceleration of the alternating tail (agreement 1e-11).
* **F2 — large-a asymptote.** The ``a^-3`` term of Q was missing its
  ``1/sqrt(2)``; the expansion additionally degrades for
  ``theta > ~80 deg`` and is now bypassed there.
* **N1 — layered_green Hankel panel.** The fixed 96-node linear
  panel under-resolved the J0 oscillations for large radial
  distance: measured errors 2.5 % at s = 100 m and 46 % at
  s = 200 m for cross-layer pairs. The grid is now
  oscillation-resolved (one 16-node panel per J0 period, log-region
  capped below the first oscillation); reference values below were
  cross-checked against adaptive scipy quadrature of the spectral
  difference (agreement ~1e-9, limited by the reference tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest

from groundfield.coupling.carson import carson_p_q
from groundfield.coupling.layered_green import (
    two_layer_layered_correction_real_space,
)

# ---------------------------------------------------------------------
# Carson P/Q — oscillatory intermediate regime (audit finding N3)
# ---------------------------------------------------------------------

# (a, theta_deg, P_ref, Q_ref) — independent panel+Euler reference.
_CARSON_REFERENCE = [
    # The audit's headline failures (historic errors -36 % / +67 %):
    (0.44, 88.6, 0.3644770, 0.7301439),
    (1.60, 89.3, 0.2368087, 0.1788292),
    # Exactly perpendicular (p = 0) — historic code fell back to the
    # small-a form, wrong for a > 0.25:
    (0.44, 90.0, 0.3661619, 0.7277049),
    (1.60, 90.0, 0.2374283, 0.1755649),
    # Large-a with theta > 80 deg now routes through the quadrature:
    (8.00, 88.6, 0.0176047, 0.0021053),
]


@pytest.mark.parametrize("a, deg, P_ref, Q_ref", _CARSON_REFERENCE)
def test_carson_oscillatory_regime(a, deg, P_ref, Q_ref) -> None:
    P, Q = carson_p_q(a, np.deg2rad(deg))
    assert P == pytest.approx(P_ref, rel=1e-4)
    assert Q == pytest.approx(Q_ref, rel=1e-4)


def test_carson_theta_zero_unchanged() -> None:
    """The non-oscillatory (self-correction) path keeps its historic
    machine-accurate Gauss–Legendre evaluation."""
    P, Q = carson_p_q(2.0, 0.0)
    # Adaptive-quadrature reference: machine agreement.
    assert P == pytest.approx(0.19124329, rel=1e-6)
    assert Q == pytest.approx(0.30452142, rel=1e-6)


def test_carson_large_a_q_cubed_term_fixed() -> None:
    """Q's a^-3 asymptote term carries 1/sqrt(2) (audit finding F2).

    At a = 8, theta = 0 the historic form was off by
    (1 - 1/sqrt(2))*cos(3*theta)/a^3 = 5.7e-4 absolute; the fixed
    expansion must agree with the quadrature path to ~2e-5 (the
    remaining truncation of the asymptotic series).
    """
    from groundfield.coupling.carson import _p_q_large, _p_q_quadrature

    P_asym, Q_asym = _p_q_large(8.0, 0.0)
    P_quad, Q_quad = _p_q_quadrature(8.0, 0.0)
    assert abs(Q_asym - Q_quad) < 5e-5
    assert abs(P_asym - P_quad) < 5e-5


def test_carson_regime_continuity_at_a5() -> None:
    """Quadrature and asymptote stay continuous at the a = 5 boundary."""
    for deg in (0.0, 30.0, 60.0, 80.0):
        th = np.deg2rad(deg)
        P_lo, Q_lo = carson_p_q(4.999, th)
        P_hi, Q_hi = carson_p_q(5.001, th)
        # The jump equals the asymptote's truncation error at a = 5
        # (~5e-4 at theta = 80 deg, within the ADR-0005 5e-3 band).
        assert abs(P_hi - P_lo) < 2e-3
        assert abs(Q_hi - Q_lo) < 2e-3


# ---------------------------------------------------------------------
# layered_green — oscillation-resolved Hankel grid (audit finding N1)
# ---------------------------------------------------------------------

_PARS = dict(rho_1=100.0, rho_2=1000.0, h_1=5.0)

# (s, z, z_s, reference) — adaptive scipy reference of the spectral
# difference, agreement ~1e-9. The historic fixed panel was off by
# 2.5 % (s=100), 46 % (s=200) and 16 % (near-interface s=100).
_GREEN_REFERENCE = [
    (50.0, 0.7, 7.0, 12.9155576),
    (100.0, 0.7, 7.0, 7.7681534),
    (200.0, 0.7, 7.0, 4.2656567),
    (100.0, 4.9, 4.9, 7.8113960),
]


@pytest.mark.parametrize("s, z, z_s, ref", _GREEN_REFERENCE)
def test_layered_correction_large_radius(s, z, z_s, ref) -> None:
    g = two_layer_layered_correction_real_space(s, z, z_s, **_PARS)
    assert g == pytest.approx(ref, rel=1e-6)


def test_layered_correction_small_radius_unchanged() -> None:
    """Small s keeps the historic single-panel path (bit-compatible):
    the default grid must agree with a heavily refined one."""
    g_def = two_layer_layered_correction_real_space(1.0, 0.7, 7.0, **_PARS)
    g_ref = two_layer_layered_correction_real_space(
        1.0, 0.7, 7.0, **_PARS, n_log=256, n_lin=4096,
    )
    assert g_def == pytest.approx(g_ref, rel=1e-10)


def test_layered_correction_homogeneous_limit_zero() -> None:
    g = two_layer_layered_correction_real_space(
        100.0, 0.7, 7.0, rho_1=100.0, rho_2=100.0, h_1=5.0,
    )
    assert g == 0.0
