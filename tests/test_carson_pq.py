"""Carson's P, Q: large-a asymptote and memoisation (10.46, 10.63)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from groundfield.coupling.carson import _p_q_large, _p_q_quadrature, carson_p_q


@pytest.mark.finding("10.46")
@pytest.mark.parametrize("theta", [0.0, 0.5])
def test_q_asymptote_a7_term_has_positive_sign(theta: float) -> None:
    """Residual of the a^-7 asymptote scaled by a^7 is small (correct sign).

    With the historic -45 the scaled residual tends to
    -90 cos(7 theta)/sqrt(2) = -63.6 cos(7 theta) (measured -60.5 at
    a = 20, theta = 0); the quadrature path is machine-accurate there.
    """
    a = 20.0
    _, q_asym = _p_q_large(a, theta)
    _, q_quad = _p_q_quadrature(a, theta)
    assert abs((q_asym - q_quad) * a**7) < 5.0


@pytest.mark.finding("10.46")
@pytest.mark.parametrize("a_boundary", [5.0, 15.0])
@pytest.mark.parametrize("theta", [0.0, 0.7, 1.2, 1.39])
def test_large_a_regime_is_continuous(a_boundary: float, theta: float) -> None:
    """No jump at the old (a = 5) or new (a = 15) asymptote boundary.

    v0.15.0 jumped by 4.5e-4 (theta = 0) to 8.2e-4 (theta = 1.39) at a = 5.
    """
    p_lo, q_lo = carson_p_q(a_boundary * (1.0 - 1e-9), theta)
    p_hi, q_hi = carson_p_q(a_boundary * (1.0 + 1e-9), theta)
    assert abs(p_hi - p_lo) < 1e-6
    assert abs(q_hi - q_lo) < 1e-6


@pytest.mark.finding("10.46")
@pytest.mark.xfail(
    strict=True,
    reason="wave 2: Carson's a^2 term of Q moves every carson_series result; "
    "it ships with the carson_series image-baseline fix (10.20)",
)
def test_small_a_regime_is_continuous() -> None:
    """The small-a form omits -(pi/64) a^2 cos(2 theta): 2.9e-3 jump in Q."""
    for theta in (0.0, 1.39):
        _, q_lo = carson_p_q(0.25 * (1.0 - 1e-9), theta)
        _, q_hi = carson_p_q(0.25 * (1.0 + 1e-9), theta)
        assert abs(q_hi - q_lo) < 1e-6


@pytest.mark.finding("10.63")
def test_multifrequency_carson_matrices_are_bit_identical_and_cached() -> None:
    from groundfield.coupling import carson as gc
    from groundfield.coupling.inductance import (
        build_carson_correction_matrices,
        build_carson_correction_matrix,
    )

    # two parallel streets of 10 m segments plus a service drop
    eps = [[[10.0 * k, 0.0, 0.7], [10.0 * (k + 1), 0.0, 0.7]] for k in range(6)]
    eps += [[[10.0 * k, 8.0, 0.7], [10.0 * (k + 1), 8.0, 0.7]] for k in range(6)]
    eps += [[[25.0, 0.0, 0.7], [31.0, 5.0, 0.7]]]
    eps = np.array(eps)
    radii = np.full(len(eps), 0.005)
    omegas = [2.0 * math.pi * f for f in (50.0, 250.0, 1000.0)]
    gc._carson_p_q_cached.cache_clear()
    multi = build_carson_correction_matrices(
        eps, radii, omegas=omegas, sigma_earth=0.02
    )
    info = gc._carson_p_q_cached.cache_info()
    # identical (a, theta) are evaluated once: repeated street spacings
    assert info.hits > info.misses
    for k, om in enumerate(omegas):
        single = build_carson_correction_matrix(eps, radii, omega=om, sigma_earth=0.02)
        assert np.array_equal(multi[k], single)


@pytest.mark.finding("10.63")
def test_cached_values_equal_uncached_values() -> None:
    from groundfield.coupling import carson as gc

    uncached = gc._carson_p_q_cached.__wrapped__
    for a in (0.1, 0.8, 3.0, 9.0, 40.0):
        for theta in (0.0, 0.6, 1.45):
            assert carson_p_q(a, theta) == uncached(a, theta)
