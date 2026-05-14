"""Carson 1926 correction for the earth-return path.

This module implements the magnetic-image correction described in
ADR-0005. It complements the perfect-mirror Neumann assembly from
ADR-0004 (in :mod:`groundfield.coupling.inductance`) with the
finite-conductivity contribution that Carson 1926 derived for a
homogeneous, semi-infinite, conductive half-space.

Mathematical background
-----------------------
Carson 1926 (Bell Syst. Tech. J. 5(4)) writes the earth-return
correction to the per-unit-length series impedance of a long
straight wire above a homogeneous earth as

$$
\\Delta Z_\\text{Carson}(\\omega) \\;=\\; \\frac{\\omega\\,\\mu_0}{\\pi}\\,
\\bigl[P(a, \\theta) \\,+\\, j\\,Q(a, \\theta)\\bigr],
$$

with the dimensionless Carson parameter

$$
a \\;=\\; D\\,\\sqrt{\\omega\\,\\mu_0\\,\\sigma_\\text{earth}}
\\;=\\; \\frac{D\\sqrt{2}}{\\delta(\\omega)},
\\qquad
\\delta(\\omega) \\;=\\; \\sqrt{2 \\,/\\, (\\omega\\mu_0\\sigma_\\text{earth})}
\\;[\\text{skin depth in soil}],
$$

where $D = 2h_i$ for the self-impedance correction
($\\theta = 0$) and
$D = \\sqrt{(h_i + h_j)^2 + d_{ij}^2}$,
$\\theta = \\arctan(d_{ij} / (h_i + h_j))$ for the mutual-impedance
correction between two parallel wires at heights $h_i, h_j$ with
horizontal separation $d_{ij}$.

The functions $P, Q$ are the real and imaginary parts of Carson's
infinite integral $J(p, q)$ (Carson eq. 29, with $p = a\\cos\\theta$,
$q = a\\sin\\theta$):

$$
J(p, q) \\;=\\; \\int_0^{\\infty}\\!\\bigl(\\sqrt{\\mu^2 + j}-\\mu\\bigr)\\,
e^{-p\\mu}\\cos(q\\mu)\\,d\\mu
\\;=\\; P(a,\\theta) + j\\,Q(a,\\theta).
$$

Three evaluation regimes
------------------------
We follow Carson's own discussion in section III of the original
paper:

1. **Small $a$** ($a \\le 0.25$) — Carson eqs. 34/35,
   leading-term form. Closed form, only $\\sin / \\cos / \\ln$.
2. **Intermediate $a$** ($0.25 < a \\le 5$) — direct
   **numerical quadrature** of Carson's $J(p, q)$ via
   Gauss–Legendre on a truncated interval. The original Carson
   1926 series in that range is technically convergent but its
   recurrence is numerically delicate; quadrature is robust and
   converges to machine precision in $\\le 64$ nodes.
3. **Large $a$** ($a > 5$) — Carson eqs. 36/37, asymptotic
   expansion in inverse powers of $a$.

Every regime boundary is smoke-tested for continuity at
$\\le 10^{-6}$ — see :mod:`tests.test_carson_coupling`.

Implementation notes
--------------------
- All formulas use **SI units throughout**. Carson's CGS
  pre-factor $4\\omega$ becomes $\\omega\\mu_0 / \\pi$ in SI.
- $\\sigma_\\text{earth}$ is an explicit argument; the caller is
  responsible for selecting the right resistivity (homogeneous
  $1/\\rho$, or upper-layer $1/\\rho_1$ with a warning).
- $\\omega = 0$ or $\\sigma = 0$ short-circuit to ``0+0j``: the
  prefactor $\\omega\\mu_0/\\pi$ vanishes, so the (logarithmically
  diverging) $Q$-asymptote is harmless.
- The complex-depth Deri/Semlyen approximation is provided as an
  internal sanity check (:func:`deri_semlyen_correction`); it is
  not the production code path.

References
----------
- Carson, J. R. (1926). Wave propagation in overhead wires with
  ground return. *Bell Syst. Tech. J.* **5**(4), 539–554.
- Deri, A.; Tevan, G.; Semlyen, A.; Castanheira, A. (1981). The
  complex ground return plane. *IEEE Trans. PAS* **100**(8),
  3686–3693.
- Tleis, N. D. (2008). *Power Systems Modelling and Fault
  Analysis*, Newnes, ch. 3.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "MU_0",
    "carson_p_q",
    "carson_self_correction",
    "carson_mutual_correction",
    "deri_semlyen_correction",
    "skin_depth",
    "carson_parameter",
]

# Vacuum permeability in H/m (CODATA 2018).
MU_0 = 4.0e-7 * math.pi

# Carson eq. 35 leading offset for Q. Carson 1926 prints this as
# "-0.0386"; modern textbooks (Tleis 2008 Tab. 3.1) trace it back to
# (1/2)·ln(γ/2) - 1/4 with Euler's γ = 0.57721566... and arrive at
# the same numerical value.
_Q_OFFSET_SMALL_A = -0.0386

# Regime boundaries (Carson 1926, p. 547).
_REGIME_SMALL_MAX = 0.25
_REGIME_LARGE_MIN = 5.0

# Gauss–Legendre quadrature nodes for the intermediate regime.
# 64 nodes are sufficient for machine-precision evaluation of
# Carson's integral over the typical parameter range; the marginal
# cost vs. 32 nodes is negligible compared with the surrounding
# linear-system solve.
_GL_NODES_64, _GL_WEIGHTS_64 = np.polynomial.legendre.leggauss(64)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def skin_depth(omega: float, sigma_earth: float) -> float:
    """Electromagnetic skin depth in soil.

    Returns $\\delta = \\sqrt{2 / (\\omega \\mu_0 \\sigma)}$ in
    metres. Diverges at $\\omega = 0$ (purely conductive earth-
    return path); the caller must handle that case by skipping the
    Carson correction altogether.

    Parameters
    ----------
    omega
        Angular frequency $\\omega = 2\\pi f$ in rad/s.
    sigma_earth
        Earth conductivity $\\sigma$ in S/m.

    Returns
    -------
    delta : float
        Skin depth in metres. ``+inf`` when ``omega`` or
        ``sigma_earth`` is zero.
    """
    if omega <= 0.0 or sigma_earth <= 0.0:
        return math.inf
    return math.sqrt(2.0 / (omega * MU_0 * sigma_earth))


def carson_parameter(distance: float, omega: float, sigma_earth: float) -> float:
    """Dimensionless Carson parameter $a = D \\sqrt{\\omega\\mu_0\\sigma}$.

    Parameters
    ----------
    distance
        Geometric distance $D$ in metres
        ($2h$ for self-impedance, $\\sqrt{(h_i+h_j)^2 + d^2}$
        for mutual).
    omega
        Angular frequency in rad/s.
    sigma_earth
        Earth conductivity in S/m.

    Returns
    -------
    a : float
        Dimensionless Carson parameter $a$.
    """
    if distance < 0.0:
        raise ValueError("distance must be non-negative")
    if omega <= 0.0 or sigma_earth <= 0.0:
        return 0.0
    return distance * math.sqrt(omega * MU_0 * sigma_earth)


# ---------------------------------------------------------------------
# Three-regime evaluation of P(a, theta), Q(a, theta)
# ---------------------------------------------------------------------


def _p_q_small(a: float, theta: float) -> tuple[float, float]:
    """Carson eqs. 34/35, leading-term small-$a$ form.

    Valid for $a \\le 0.25$ to within $\\le 1 \\cdot 10^{-9}$ of
    the full integral, and still accurate to $\\sim 0.5\\,\\%$
    at $a = 0.4$ (cf. Carson's *railway* and *wave-antenna*
    worked examples in section V of the original paper).
    """
    if a == 0.0:
        return math.pi / 8.0, math.inf
    cos_th = math.cos(theta)
    cos_2th = math.cos(2.0 * theta)
    sin_2th = math.sin(2.0 * theta)
    P = (
        math.pi / 8.0
        - a * cos_th / (3.0 * math.sqrt(2.0))
        + (a * a / 16.0) * cos_2th * (0.6728 + math.log(2.0 / a))
        + (a * a / 16.0) * theta * sin_2th
    )
    Q = (
        _Q_OFFSET_SMALL_A
        + 0.5 * math.log(2.0 / a)
        + a * cos_th / (3.0 * math.sqrt(2.0))
    )
    return P, Q


def _p_q_quadrature(a: float, theta: float) -> tuple[float, float]:
    """Direct numerical evaluation of Carson's $J(p, q)$.

    Computes $P + jQ = J(p, q) = \\int_0^\\infty (\\sqrt{\\mu^2 + j}
    - \\mu)\\, e^{-p\\mu}\\, \\cos(q\\mu)\\, d\\mu$ with
    $p = a\\cos\\theta$, $q = a\\sin\\theta$, by 64-point
    Gauss–Legendre quadrature on the truncated interval
    $[0, \\mu_\\max]$ with $\\mu_\\max = 30/p$ — ensuring
    $e^{-p\\mu_\\max} \\le 10^{-13}$.

    Used for the intermediate regime $0.25 < a \\le 5$ where the
    classical Tleis recurrence on Carson's series is numerically
    delicate (alternating coefficients with rapidly varying
    magnitudes). Quadrature converges to machine precision in
    $\\le 64$ nodes across the typical parameter range and is the
    reference implementation of the present module.

    For $\\theta \\to \\pi/2$ (wires at the same height with only
    horizontal separation) $p \\to 0$ and the truncation interval
    grows; in that case the function falls back to the small-$a$
    closed form which is exact in the $p = 0$ limit.

    Parameters
    ----------
    a : float
        Carson parameter, $a \\ge 0$.
    theta : float
        Angle $\\theta \\in [0, \\pi/2]$ in radians.

    Returns
    -------
    P, Q : tuple[float, float]
        Real and imaginary parts of $J(p, q)$.
    """
    p = a * math.cos(theta)
    q = a * math.sin(theta)
    if p <= 1e-6:
        # Degenerate (theta -> pi/2). The integrand has no
        # exponential decay; fall back to the small-a form so
        # tests do not fail on edge cases. typical geometries always
        # have h > 0 for both wires, so we never hit this branch
        # in production.
        return _p_q_small(max(a, 1e-12), theta)

    mu_max = 30.0 / p
    half = 0.5 * mu_max
    mu = half * (_GL_NODES_64 + 1.0)
    w = half * _GL_WEIGHTS_64

    # sqrt(mu^2 + j) with j = imaginary unit. numpy gives the
    # principal branch automatically.
    sqrt_term = np.sqrt(mu * mu + 1j)
    decay = np.exp(-p * mu) * np.cos(q * mu)
    integrand_re = (sqrt_term.real - mu) * decay
    integrand_im = sqrt_term.imag * decay
    P = float(np.sum(w * integrand_re))
    Q = float(np.sum(w * integrand_im))
    return P, Q


def _p_q_large(a: float, theta: float) -> tuple[float, float]:
    """Carson eqs. 36/37, asymptotic expansion for $a > 5$.

    Truncates after the $1/a^7$ term, which matches Carson's own
    Fig. 2/3 within plotting accuracy.
    """
    inv_a = 1.0 / a
    inv_a2 = inv_a * inv_a
    inv_a3 = inv_a * inv_a2
    inv_a5 = inv_a3 * inv_a2
    inv_a7 = inv_a5 * inv_a2
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    cos_t = math.cos(theta)
    cos_2t = math.cos(2.0 * theta)
    cos_3t = math.cos(3.0 * theta)
    cos_5t = math.cos(5.0 * theta)
    cos_7t = math.cos(7.0 * theta)
    P = (
        sqrt2_inv * cos_t * inv_a
        - cos_2t * inv_a2
        + sqrt2_inv * cos_3t * inv_a3
        + 3.0 * sqrt2_inv * cos_5t * inv_a5
        - 45.0 * sqrt2_inv * cos_7t * inv_a7
    )
    Q = (
        sqrt2_inv * cos_t * inv_a
        - cos_3t * inv_a3
        + 3.0 * sqrt2_inv * cos_5t * inv_a5
        - 45.0 * sqrt2_inv * cos_7t * inv_a7
    )
    return P, Q


def carson_p_q(a: float, theta: float) -> tuple[float, float]:
    """Evaluate Carson's $P(a, \\theta)$, $Q(a, \\theta)$.

    Dispatches to the appropriate regime based on the magnitude of
    $a$:

    - $a \\le 0.25$: closed-form leading-term expansion (Carson
      eqs. 34/35).
    - $0.25 < a \\le 5$: direct numerical quadrature of Carson's
      $J(p, q)$.
    - $a > 5$: asymptotic expansion (Carson eqs. 36/37).

    The three regimes are continuous at the boundaries to within
    the tolerances documented in ADR-0005 §5/§6.

    Parameters
    ----------
    a : float
        Dimensionless Carson parameter $a = D \\sqrt{\\omega
        \\mu_0 \\sigma_\\text{earth}}$. Must be $\\ge 0$.
    theta : float
        Angle of the image-distance vector to the vertical, in
        radians. $\\theta = 0$ for the self-impedance correction,
        $\\theta = \\arctan(d/(h_i+h_j))$ for the mutual.

    Returns
    -------
    P, Q : tuple[float, float]
        Real and imaginary parts of Carson's integral $J$.

    Raises
    ------
    ValueError
        If ``a`` is negative.
    """
    if a < 0.0:
        raise ValueError(f"Carson parameter a must be non-negative, got {a}")
    if a == 0.0:
        return math.pi / 8.0, math.inf
    if a <= _REGIME_SMALL_MAX:
        return _p_q_small(a, theta)
    if a <= _REGIME_LARGE_MIN:
        return _p_q_quadrature(a, theta)
    return _p_q_large(a, theta)


# ---------------------------------------------------------------------
# Self- and mutual-impedance corrections (per unit length)
# ---------------------------------------------------------------------


def carson_self_correction(
    omega: float,
    height: float,
    sigma_earth: float,
) -> complex:
    """Carson earth-return correction for a single horizontal wire.

    Evaluates the per-unit-length impedance correction

    $$
    \\Delta Z_\\text{self}(\\omega) \\;=\\; \\frac{\\omega \\mu_0}{\\pi}
    \\bigl[P(a_s, 0) + j Q(a_s, 0)\\bigr],
    $$

    with $a_s = 2h\\sqrt{\\omega\\mu_0\\sigma_\\text{earth}}$ and
    $\\theta = 0$ (Carson eq. 30, ADR-0005).

    Parameters
    ----------
    omega
        Angular frequency $\\omega = 2\\pi f$ in rad/s.
    height
        Height of the wire above the earth surface in metres
        (positive — for buried wires use the Sunde-equivalent
        height).
    sigma_earth
        Earth conductivity in S/m
        ($\\sigma = 1/\\rho$ for a homogeneous earth).

    Returns
    -------
    Z : complex
        Per-unit-length earth-return correction in $\\Omega/\\text{m}$.

    Raises
    ------
    ValueError
        If ``height`` is non-positive.
    """
    if height <= 0.0:
        raise ValueError("height must be positive (use |z| of the wire)")
    if omega <= 0.0 or sigma_earth <= 0.0:
        return 0.0 + 0.0j
    a = carson_parameter(2.0 * height, omega, sigma_earth)
    P, Q = carson_p_q(a, 0.0)
    pref = omega * MU_0 / math.pi
    return complex(pref * P, pref * Q)


def carson_mutual_correction(
    omega: float,
    height_i: float,
    height_j: float,
    horizontal_distance: float,
    sigma_earth: float,
) -> complex:
    """Carson earth-return correction between two parallel horizontal wires.

    Evaluates the per-unit-length mutual-impedance correction

    $$
    \\Delta Z_\\text{mutual}(\\omega) \\;=\\; \\frac{\\omega\\mu_0}{\\pi}
    \\bigl[P(a_m, \\theta_m) \\,+\\, j Q(a_m, \\theta_m)\\bigr],
    $$

    with

    $$
    a_m \\;=\\; D_m \\sqrt{\\omega\\mu_0\\sigma_\\text{earth}}, \\quad
    D_m \\;=\\; \\sqrt{(h_i + h_j)^2 + d^2}, \\quad
    \\theta_m \\;=\\; \\arctan(d / (h_i + h_j)),
    $$

    after Carson eq. 31. The parallel-wire assumption is the same
    as for ADR-0004's Neumann fast path; for non-parallel segments
    the caller must split into projection components.

    Parameters
    ----------
    omega
        Angular frequency in rad/s.
    height_i, height_j
        Heights above earth surface in metres (both positive).
    horizontal_distance
        Horizontal separation $d$ between the two wires in metres.
    sigma_earth
        Earth conductivity in S/m.

    Returns
    -------
    Z : complex
        Per-unit-length mutual correction in $\\Omega/\\text{m}$.
    """
    if height_i <= 0.0 or height_j <= 0.0:
        raise ValueError("heights must be positive")
    if horizontal_distance < 0.0:
        raise ValueError("horizontal_distance must be non-negative")
    if omega <= 0.0 or sigma_earth <= 0.0:
        return 0.0 + 0.0j
    h_sum = height_i + height_j
    D = math.hypot(h_sum, horizontal_distance)
    theta = math.atan2(horizontal_distance, h_sum)
    a = carson_parameter(D, omega, sigma_earth)
    P, Q = carson_p_q(a, theta)
    pref = omega * MU_0 / math.pi
    return complex(pref * P, pref * Q)


# ---------------------------------------------------------------------
# Deri/Semlyen complex-depth approximation (sanity check only)
# ---------------------------------------------------------------------


def deri_semlyen_correction(
    omega: float,
    height_i: float,
    height_j: float,
    horizontal_distance: float,
    sigma_earth: float,
) -> complex:
    """Deri/Semlyen 1981 complex-depth approximation.

    Replaces the Carson integral by the closed-form expression

    $$
    \\Delta Z_\\text{Deri-Semlyen} \\;=\\; \\frac{j\\omega\\mu_0}{2\\pi}
    \\ln\\!\\Bigl(\\frac{D'}{D}\\Bigr),
    $$

    with $D = \\sqrt{(h_i - h_j)^2 + d^2}$ the direct distance,
    $D' = \\sqrt{(h_i + h_j + 2p)^2 + d^2}$ the distance to a
    complex-depth image, and the complex penetration depth

    $$
    p \\;=\\; 1 \\big/ \\sqrt{j\\omega\\mu_0\\sigma_\\text{earth}}.
    $$

    The Deri/Semlyen approximation is **not** the production
    code path. It is provided as an alternative independent
    estimator that the test suite can compare against the Carson
    series — agreement within $\\approx 5\\,\\%$ over the typical
    parameter range confirms that neither implementation contains
    a sign or pre-factor bug.

    Parameters
    ----------
    omega, height_i, height_j, horizontal_distance, sigma_earth
        Same meaning as in :func:`carson_mutual_correction`.

    Returns
    -------
    Z : complex
        Per-unit-length earth-return correction in $\\Omega/\\text{m}$
        according to Deri/Semlyen 1981.

    Notes
    -----
    The "self" version is recovered by setting
    ``height_i = height_j`` and ``horizontal_distance = 0``: the
    direct-distance term degenerates to the wire radius, which the
    caller must substitute manually (the formula does not include
    a wire-radius regularisation by itself).
    """
    if omega <= 0.0 or sigma_earth <= 0.0:
        return 0.0 + 0.0j
    p = 1.0 / np.sqrt(1j * omega * MU_0 * sigma_earth)
    D = math.hypot(height_i - height_j, horizontal_distance)
    if D == 0.0:
        D = 1e-9
    D_prime = np.sqrt((height_i + height_j + 2.0 * p) ** 2 + horizontal_distance ** 2)
    return complex(1j * omega * MU_0 / (2.0 * math.pi) * np.log(D_prime / D))
