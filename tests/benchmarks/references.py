"""Independent reference values for the engine benchmark suite.

Nothing here imports a groundfield solver: every value is either a
closed form or the result of a small, independent solver of the same
physical problem, so a benchmark compares a backend against a number it
did not help to compute.

Conventions: SI units (Ω·m, m, A, Ω), soil surface at ``z = 0``,
``z`` positive into the soil, quasi-static current field.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ellipk

__all__ = [
    "hemisphere_resistance",
    "ring_thin_wire_resistance",
    "rod_thin_wire_galerkin",
    "rod_thin_wire_limit",
    "three_layer_kernel_mpmath",
    "two_layer_series_kernel",
]


def hemisphere_resistance(rho: float, radius: float) -> float:
    """Resistance of a hemisphere at the soil surface, ``rho / (2 pi a)``.

    Parameters
    ----------
    rho
        Soil resistivity in Ω·m.
    radius
        Hemisphere radius in m.

    Returns
    -------
    float
        Resistance in Ω (exact).
    """
    return rho / (2.0 * math.pi * radius)


def _h(t: np.ndarray, a: float) -> np.ndarray:
    """Antiderivative of ``asinh(t / a)`` (double integral of the reduced kernel)."""
    return t * np.arcsinh(t / a) - np.sqrt(t * t + a * a)


def rod_thin_wire_galerkin(length: float, radius: float, rho: float, n: int) -> float:
    """Galerkin thin-wire resistance of a vertical rod with its top at ``z = 0``.

    Pulse basis on ``n`` equal segments, Galerkin testing (test = basis),
    the reduced kernel ``1 / sqrt(t^2 + a^2)`` and the air image; every
    matrix entry is the analytic double integral. The Galerkin value is
    an upper bound of the thin-wire resistance that decreases
    monotonically under refinement (energy principle); see
    :func:`rod_thin_wire_limit` for the extrapolated limit.

    Parameters
    ----------
    length
        Rod length in m.
    radius
        Wire radius in m.
    rho
        Soil resistivity in Ω·m.
    n
        Number of equal segments.

    Returns
    -------
    float
        Resistance in Ω.
    """
    e = np.linspace(0.0, length, n + 1)
    z1, z2 = e[:-1], e[1:]
    zi, zf = z1[:, None], z2[:, None]
    w1, w2 = z1[None, :], z2[None, :]
    direct = (
        _h(zf - w1, radius)
        - _h(zi - w1, radius)
        - _h(zf - w2, radius)
        + _h(zi - w2, radius)
    )
    image = (
        _h(zf + w2, radius)
        - _h(zi + w2, radius)
        - _h(zf + w1, radius)
        + _h(zi + w1, radius)
    )
    ds = z2 - z1
    z_mat = (direct + image) / ds[:, None] / ds[None, :] * rho / (4.0 * math.pi)
    a_mat = np.block([[z_mat, -np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    return float(np.linalg.solve(a_mat, b)[-1])


def rod_thin_wire_limit(length: float, radius: float, rho: float) -> float:
    """Thin-wire limit of :func:`rod_thin_wire_galerkin` (Richardson).

    Galerkin values on ``n = 96`` and ``n = 192`` segments, extrapolated
    with an ``O(h^2)``-type step. For ``L = 3 m, a = 8 mm, rho = 100``
    this gives 33.1957 Ω, against 33.4927 Ω from Sunde's closed form
    ``rho / (2 pi L) [ln(4 L / a) - 1]`` (which assumes a uniform line
    current) and 33.3254 Ω for the Galerkin solution at ``ds = 0.5 m``.

    Returns
    -------
    float
        Resistance in Ω.
    """
    coarse = rod_thin_wire_galerkin(length, radius, rho, 96)
    fine = rod_thin_wire_galerkin(length, radius, rho, 192)
    return fine + (fine - coarse) / 3.0


def ring_thin_wire_resistance(
    rho: float, ring_radius: float, wire_radius: float, depth: float
) -> float:
    """Thin-wire resistance of a horizontal ring buried at ``depth``.

    By rotational symmetry the leakage current along a ring is uniform,
    so the thin-wire problem has the closed-form solution

    ``R = rho / (2 pi^2) [K(m) / sqrt(a^2 + 4 R0^2) + K(m') / sqrt(4 z^2 + 4 R0^2)]``

    with ``m = 4 R0^2 / (a^2 + 4 R0^2)`` (the ring seen from its own wire
    surface) and ``m' = 4 R0^2 / (4 z^2 + 4 R0^2)`` (its air image), and
    ``K`` the complete elliptic integral of the first kind (parameter
    convention of :func:`scipy.special.ellipk`).

    Parameters
    ----------
    rho
        Soil resistivity in Ω·m.
    ring_radius
        Ring radius ``R0`` in m.
    wire_radius
        Wire radius ``a`` in m.
    depth
        Burial depth ``z`` in m.

    Returns
    -------
    float
        Resistance in Ω.
    """
    r0, a, z = ring_radius, wire_radius, depth
    m_self = 4.0 * r0**2 / (a**2 + 4.0 * r0**2)
    m_img = 4.0 * r0**2 / (4.0 * z**2 + 4.0 * r0**2)
    self_term = ellipk(m_self) / math.sqrt(a**2 + 4.0 * r0**2)
    image_term = ellipk(m_img) / math.sqrt(4.0 * z**2 + 4.0 * r0**2)
    return rho / (2.0 * math.pi**2) * float(self_term + image_term)


def three_layer_kernel_mpmath(
    rhos: tuple[float, float, float],
    h: tuple[float, float],
    s: float,
    z: float,
    zs: float,
) -> float:
    """Top-layer three-layer Green's function by mpmath quadrature.

    The direct pair and its air mirror are taken in closed form; the
    reflected remainder

    ``int_0^inf [xi(lambda) - e^{-lambda|dz|} - e^{-lambda p}] J0(lambda s) dlambda``

    with ``xi`` the full top-layer spectral kernel (both image families
    and the multiple-reflection denominator, ``Gamma_1`` from the
    two-interface recursion) is integrated with ``mpmath.quadosc`` at
    30 digits (``mpmath.quad`` for ``s = 0``). Shares no code with
    :mod:`groundfield.solver.mom_sommerfeld`.

    Parameters
    ----------
    rhos
        Layer resistivities ``(rho_1, rho_2, rho_3)`` in Ω·m.
    h
        Thicknesses ``(h_1, h_2)`` in m.
    s
        Horizontal distance in m.
    z, zs
        Depths of the field point and the source in m (both ``<= h_1``).

    Returns
    -------
    float
        ``G`` in 1/m (potential ``rho_1 I G / (4 pi)``).
    """
    import mpmath as mp

    with mp.workdps(30):
        r1, r2, r3 = (mp.mpf(r) for r in rhos)
        h1, h2 = mp.mpf(h[0]), mp.mpf(h[1])
        k1 = (r2 - r1) / (r2 + r1)
        k2 = (r3 - r2) / (r3 + r2)
        adz, p = abs(mp.mpf(z) - mp.mpf(zs)), mp.mpf(z) + mp.mpf(zs)

        def remainder(lam):
            e = mp.e ** (-2 * lam * h2)
            g = (k1 + k2 * e) / (1 + k1 * k2 * e)
            e2h = mp.e ** (-2 * lam * h1)
            xi = (
                mp.e ** (-lam * adz)
                + g * mp.e ** (-lam * (2 * h1 - adz))
                + mp.e ** (-lam * p)
                + g * mp.e ** (-lam * (2 * h1 - p))
            ) / (1 - g * e2h)
            out = xi - mp.e ** (-lam * adz) - mp.e ** (-lam * p)
            return out * mp.besselj(0, lam * s) if s > 0 else out

        if s > 0:
            refl = mp.quadosc(remainder, [0, mp.inf], period=2 * mp.pi / s)
        else:
            refl = mp.quad(remainder, [0, 1, 10, mp.inf])
        direct = 1 / max(math.hypot(s, float(adz)), 1e-3) + 1 / math.hypot(s, float(p))
        return direct + float(refl)


def two_layer_series_kernel(k: float, h: float, s: float, z: float, zs: float) -> float:
    """Top-layer two-layer Green's function as the Tagg/Sunde image series.

    ``G = sum_m K^m [1/R(2mh + |dz|) + K/R(2(m+1)h - |dz|) + 1/R(2mh + p)
    + K/R(2(m+1)h - p)]`` with ``R(d) = sqrt(s^2 + d^2)`` floored at 1 mm
    and ``p = z + zs``, summed (small terms first) until ``|K|^m < 1e-17``.
    The potential of a point source ``I`` is ``rho_1 I G / (4 pi)``.

    Parameters
    ----------
    k
        Reflection coefficient ``(rho_2 - rho_1) / (rho_2 + rho_1)``.
    h
        Top-layer thickness in m.
    s
        Horizontal distance in m.
    z, zs
        Depths of the field point and the source in m (both ``<= h``).

    Returns
    -------
    float
        ``G`` in 1/m.
    """
    adz, p = abs(z - zs), z + zs

    def r(d):
        return np.maximum(np.hypot(s, d), 1e-3)

    if k == 0.0:
        return float(1.0 / r(adz) + 1.0 / r(p))
    m_max = int(math.ceil(math.log(1e-17) / math.log(abs(k)))) + 10
    m = np.arange(m_max + 1, dtype=float)
    terms = k**m * (
        1.0 / r(2 * m * h + adz)
        + k / r(2 * (m + 1) * h - adz)
        + 1.0 / r(2 * m * h + p)
        + k / r(2 * (m + 1) * h - p)
    )
    return float(np.sum(terms[::-1]))
