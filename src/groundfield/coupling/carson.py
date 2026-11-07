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
3. **Large $a$** ($a > 15$) — Carson eqs. 36/37, asymptotic
   expansion in inverse powers of $a$ up to $a^{-7}$ (for
   $\\theta \\le 1.4$ rad; steeper angles stay on the quadrature).

Accuracy against mpmath (review pass 10): the quadrature is
machine-accurate for every $a > 0.25$; the asymptote is accurate to
$\\le 10^{-7}$ above $a = 15$, so the large-$a$ boundary is continuous
to that level. The small-$a$ form omits Carson's $a^2$ term of $Q$
($-(\\pi/64)\\,a^2\\cos 2\\theta$) and rounds the constant to
$-0.0386$: the boundary at $a = 0.25$ jumps by up to $2.9\\cdot10^{-3}$ in
$Q$ (about 0.3 %, decaying $\\propto a^2$ below it). Changing it moves
every ``carson_series`` result and is scheduled together with the
``carson_series`` image-baseline fix (finding 10.20).

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
from functools import lru_cache

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

# Vacuum permeability in H/m: the conventional 4*pi*1e-7 (the CODATA
# 2018 value differs by 5.5e-10 relative).
MU_0 = 4.0e-7 * math.pi

# Carson eq. 35 leading offset for Q. Carson 1926 prints this as
# "-0.0386"; the exact closed form is (1/2)·(1/2 − γ_E) with Euler's
# constant γ_E = 0.57721566... → −0.0386078...
_Q_OFFSET_SMALL_A = -0.0386

# Regime boundaries. Carson (1926, p. 547) switches to the asymptote at
# a > 5; with the asymptote truncated after a^-7 that leaves jumps of up
# to 1.6e-3 at a = 5 (review pass 10, finding 10.46). The quadrature is
# machine-accurate for every a > 0.25 and theta, so the asymptote is
# only used where its truncation error is below 1e-7 over
# theta <= _ASYMPTOTE_THETA_MAX (measured against mpmath: 8.4e-8 at
# a = 15, 1.1e-5 at a = 10).
_REGIME_SMALL_MAX = 0.25
_REGIME_LARGE_MIN = 15.0
# Above this angle the a > 5 asymptotic expansion degrades (measured
# absolute errors up to ~2e-3 near the regime boundary at
# theta ~ 89°); route through the oscillatory quadrature instead
# (audit 2026-07-08, WP-B2).
_ASYMPTOTE_THETA_MAX = 1.4  # rad, ~80.2°

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

    Truncation accuracy (measured against adaptive quadrature of the
    full integral, audit 2026-07-08): $|\\Delta Q| \\approx
    3 \\cdot 10^{-3}$ absolute at $a = 0.25$ (the dropped
    $a^2$-term of the Q series), decaying $\\propto a^2$;
    $|\\Delta P| \\le 10^{-4}$ over the whole $a \\le 0.25$ range.
    Adequate for the per-metre correction because the prefactor
    $\\omega\\mu_0/\\pi$ keeps the absolute impedance error in the
    µΩ/m range at $f \\le 1$ kHz.
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
    $p = a\\cos\\theta$, $q = a\\sin\\theta$.

    Used for the intermediate regime $0.25 < a \\le 5$ where the
    classical Tleis recurrence on Carson's series is numerically
    delicate (alternating coefficients with rapidly varying
    magnitudes).

    Two evaluation paths (audit 2026-07-08, finding N3):

    - **Non-oscillatory** ($q \\approx 0$, i.e.
      $\\theta \\approx 0$, the self-correction geometry): 64-point
      Gauss–Legendre on the truncated interval $[0, 30/p]$
      ($e^{-p\\mu_\\max} \\le 10^{-13}$). Machine-accurate here,
      verified against adaptive references.
    - **Oscillatory** ($q > 0$): QUADPACK's Fourier integrator
      (``scipy.integrate.quad`` with ``weight='cos'``), designed for
      $\\int_0^\\infty f(\\mu)\\cos(q\\mu)\\,d\\mu$ with slowly
      decaying $f$. This replaces the historic fixed-node
      Gauss–Legendre on $[0, 30/p]$, which failed for
      $\\theta \\to \\pi/2$ (small $p$, huge unresolved interval):
      measured errors up to $-36\\,\\%$ in $P$ and $+67\\,\\%$ in
      $Q$ at buried-pair mutual geometries ($a = 0.44$–$1.6$,
      $\\theta \\approx 89°$). The Fourier path also covers the
      degenerate $p \\to 0$ limit exactly (the imaginary part
      decays only $\\propto 1/(2\\mu)$ and converges conditionally
      through the oscillation — precisely QAWF's use case), so the
      former inaccurate small-$a$ fallback is gone.

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

    if q <= 1e-9 * max(a, 1.0):
        # theta ~ 0: pure exponential decay, no oscillation.
        mu_max = 30.0 / p
        half = 0.5 * mu_max
        mu = half * (_GL_NODES_64 + 1.0)
        w = half * _GL_WEIGHTS_64
        # sqrt(mu^2 + j) with j = imaginary unit. numpy gives the
        # principal branch automatically.
        sqrt_term = np.sqrt(mu * mu + 1j)
        decay = np.exp(-p * mu)
        P = float(np.sum(w * (sqrt_term.real - mu) * decay))
        Q = float(np.sum(w * sqrt_term.imag * decay))
        return P, Q

    from scipy.integrate import quad

    def _f_re(mu: float) -> float:
        return (math.sqrt(0.5 * (math.hypot(mu * mu, 1.0) + mu * mu)) - mu) * math.exp(
            -p * mu
        )

    def _f_im(mu: float) -> float:
        # Im sqrt(mu^2 + j) = 1 / (2 · Re sqrt(mu^2 + j))
        re = math.sqrt(0.5 * (math.hypot(mu * mu, 1.0) + mu * mu))
        return 0.5 / re * math.exp(-p * mu)

    P = quad(_f_re, 0.0, np.inf, weight="cos", wvar=q, limit=200, limlst=200)[0]
    Q = quad(_f_im, 0.0, np.inf, weight="cos", wvar=q, limit=200, limlst=200)[0]
    return P, Q


def _p_q_large(a: float, theta: float) -> tuple[float, float]:
    """Carson eqs. 36/37, asymptotic expansion (used for $a > 15$).

    Truncated after the $1/a^7$ term. Measured against mpmath for
    $\\theta \\le 1.4$ rad (review pass 10): worst absolute error
    $1.6\\cdot 10^{-3}$ at $a = 5$, $1.1\\cdot 10^{-5}$ at $a = 10$,
    $8.4\\cdot 10^{-8}$ at $a = 15$.
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
    # The a^-7 term of Q is +45 (review pass 10, finding 10.46; it was
    # -45, which made the regime boundary jump by 4.5e-4 at a = 5): the
    # Laplace-transform expansion of sqrt(mu^2 + j) - mu gives
    # Q ~ (cos t)/(sqrt2 a) - (cos 3t)/(sqrt2 a^3)
    #     + 3 (cos 5t)/(sqrt2 a^5) + 45 (cos 7t)/(sqrt2 a^7).
    Q = (
        sqrt2_inv * cos_t * inv_a
        - sqrt2_inv * cos_3t * inv_a3
        + 3.0 * sqrt2_inv * cos_5t * inv_a5
        + 45.0 * sqrt2_inv * cos_7t * inv_a7
    )
    return P, Q


def carson_p_q(a: float, theta: float) -> tuple[float, float]:
    """Evaluate Carson's $P(a, \\theta)$, $Q(a, \\theta)$.

    Dispatches to the appropriate regime based on the magnitude of
    $a$:

    - $a \\le 0.25$: closed-form leading-term expansion (Carson
      eqs. 34/35).
    - $0.25 < a \\le 15$: direct numerical quadrature of Carson's
      $J(p, q)$ (Gauss–Legendre for $\\theta \\approx 0$, QUADPACK
      Fourier integrator otherwise — audit 2026-07-08, WP-B2).
    - $a > 15$: asymptotic expansion (Carson eqs. 36/37, $a^{-7}$
      term of $Q$ with the corrected sign), except for
      $\\theta > 1.4$ rad where the expansion degrades and the
      quadrature path is used instead.

    The large-$a$ boundary is continuous to $\\le 10^{-7}$; the
    small-$a$ boundary jumps by up to $2.9\\cdot 10^{-3}$ in $Q$ (see
    the module docstring). Results are memoised on ``(a, theta)``.

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
    return _carson_p_q_cached(float(a), float(theta))


@lru_cache(maxsize=1 << 16)
def _carson_p_q_cached(a: float, theta: float) -> tuple[float, float]:
    """Regime dispatch of :func:`carson_p_q`, memoised on ``(a, theta)``.

    The assembled correction matrices evaluate the same ``(a, theta)``
    many times (equal heights and repeated axis separations in
    street-routed networks, and the self term of every segment), and
    the oscillatory regime costs two QUADPACK calls with Python
    integrands (review pass 10, finding 10.63). The cache returns the
    identical floats, so results are bit-for-bit unchanged.
    """
    if a == 0.0:
        return math.pi / 8.0, math.inf
    if a <= _REGIME_SMALL_MAX:
        return _p_q_small(a, theta)
    if a <= _REGIME_LARGE_MIN or theta > _ASYMPTOTE_THETA_MAX:
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
    D_prime = np.sqrt((height_i + height_j + 2.0 * p) ** 2 + horizontal_distance**2)
    return complex(1j * omega * MU_0 / (2.0 * math.pi) * np.log(D_prime / D))
