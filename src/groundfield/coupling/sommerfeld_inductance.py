"""Geometric Sommerfeld earth-return Green function (ADR-0006).

This module implements the rigorous geometric formulation of the
inductive earth-return coupling described in ADR-0006. Compared to
ADR-0005's Carson per-meter scaling, it integrates the actual
vector-potential Green's function over the segment-pair geometry,
which correctly handles short wires, non-parallel arrangements,
and layered earth.

Mathematical background
-----------------------
*(Rewritten in the 2026-07-08 audit, WP-C1: the historic kernel
``1/R + ∫Γ·e^{-λ(z+z')}J0`` dropped the in-soil propagation of the
primary term — precisely where the buried-wire earth-return
resistance lives — and applied the soil-side reflection sign to
overhead conductors. The implemented kernel is now the genuine
quasi-static Pollaczek form, dispatched by which side of the soil
surface the two segments are on.)*

For a horizontal current element in a conducting half-space
($z > 0$ soil, $z < 0$ air, $\\gamma^2 = j\\omega\\mu_0\\sigma_1$,
$u_1 = \\sqrt{\\lambda^2 + \\gamma^2}$), the quasi-static
vector-potential Green's functions are:

**Buried source, buried observer** (depths $h, h' > 0$):

$$
G_\\text{bb} \\;=\\; \\frac{e^{-\\gamma R}}{R} \\;+\\; \\int_0^{\\infty}\\!
\\frac{\\lambda}{u_1}\\,\\Gamma(\\lambda)\\,
e^{-u_1 (h + h')}\\,J_0(\\lambda\\rho)\\,d\\lambda,
$$

**Overhead source and observer** (heights $H, H' > 0$):

$$
G_\\text{oo} \\;=\\; \\frac{1}{R} \\;-\\; \\int_0^{\\infty}\\!
\\Gamma(\\lambda)\\,
e^{-\\lambda (H + H')}\\,J_0(\\lambda\\rho)\\,d\\lambda
\\qquad\\text{(Carson's geometry)},
$$

**Mixed** (buried $h$, overhead $H$):

$$
G_\\text{bo} \\;=\\; \\int_0^{\\infty}\\!
\\frac{2\\lambda}{u_1 + \\lambda}\\,
e^{-u_1 h - \\lambda H}\\,J_0(\\lambda\\rho)\\,d\\lambda,
$$

with the **reflection coefficient**

- homogeneous earth (Pillar A):
  $\\Gamma(\\lambda) = (u_1 - \\lambda)/(u_1 + \\lambda)$,
- $n$-layer earth (Pillar B): the recursive Wait-style reflection
  coefficient (Wait 1972 §3, Tleis 2008 §3.5); the $\\lambda/u_1$
  weight and the exponents use the top-layer $u_1$ (the conductors
  live in layer 1).

The functions in this module return the **correction beyond the
ADR-0004 additive perfect-mirror baseline** ($1/R + 1/R'$), so the
solver assembles ``Z_b = jω·(L_Neumann+mirror) + ΔZ_Sommerfeld``.
The mirror deduplication $-1/R'$ is carried in closed form; only
the smooth reflected/transmitted remainder is integrated
numerically.

Limit checks (built into the test suite)
----------------------------------------
- $\\sigma_e\\to 0$: every kernel collapses to free space — the
  correction tends to $-1/R'$ and exactly cancels the ADR-0004
  mirror. (At DC the soil is magnetically transparent: **no image**.)
- $\\sigma_e\\to\\infty$, overhead pair: $-\\Gamma \\to -1$, the net
  image becomes the **anti-parallel** PEC image $-1/R'$ (Carson's
  baseline).
- $\\sigma_e\\to\\infty$, buried pair: $e^{-\\gamma R} \\to 0$ — the
  conductor is screened by the surrounding medium and the total
  external coupling tends to zero.
- Long parallel buried wires: the per-metre limit reproduces
  Pollaczek's mutual $\\frac{j\\omega\\mu_0}{2\\pi}\\bigl[K_0(\\gamma d)
  - K_0(\\gamma D') + J_P\\bigr]$, whose low-frequency real part is
  Carson's $\\omega\\mu_0/8$ (with the **positive** sign — the
  historic kernel produced $-\\omega\\mu_0/8$).

References
----------
- Stratton, J. A. (1941). *Electromagnetic Theory*, McGraw-Hill,
  §9-10 — derivation of the half-space vector potential.
- Sommerfeld, A. (1909). Über die Ausbreitung der Wellen in der
  drahtlosen Telegraphie. *Ann. Phys.* **28**(4), 665–736.
- Wait, J. R. (1972). *Electromagnetic Waves in Stratified
  Media*, Pergamon. Ch. 3.
- Tleis, N. D. (2008). *Power Systems Modelling and Fault
  Analysis*, Newnes. Ch. 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import j0

__all__ = [
    "MU_0",
    "earth_return_correction_homogeneous",
    "earth_return_correction_layered",
    "sommerfeld_pair_integral_homogeneous",
    "sommerfeld_pair_integral_layered",
    "build_sommerfeld_correction_matrix",
    "LayeredEarth",
    "reflection_coefficient_homogeneous",
    "reflection_coefficient_layered",
]


MU_0 = 4.0e-7 * math.pi

# Quadrature configuration. Numbers chosen so that:
# - Outer (geometry) Gauss-Legendre uses the same 16x16 grid as
#   coupling/inductance.py for consistency.
# - Inner Sommerfeld quadrature uses 200 nodes split between a
#   logarithmic part [0, lambda_break] and a linear tail
#   [lambda_break, lambda_max]. Calibrated against the perfect-
#   mirror limit and the Carson long-wire asymptote.
_OUTER_GL_NODES, _OUTER_GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
_INNER_GL_NODES, _INNER_GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


# ---------------------------------------------------------------------
# Layered-earth data class
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LayeredEarth:
    """Frozen layered-earth configuration for the Sommerfeld kernel.

    Attributes
    ----------
    rhos
        Resistivities $\\rho_1, \\dots, \\rho_n$ of the layers in
        $\\Omega\\,\\mathrm{m}$. The last entry is the
        semi-infinite bottom layer.
    thicknesses
        Thicknesses $h_1, \\dots, h_{n-1}$ in metres. The bottom
        layer has no thickness (semi-infinite). For
        ``len(thicknesses) == len(rhos) - 1``.

    Notes
    -----
    **Numerical precision contract.** The reflection-coefficient
    evaluators (:func:`reflection_coefficient_homogeneous`,
    :func:`reflection_coefficient_layered`) and every consumer of
    this dataclass operate in **IEEE-754 double precision (FP64)**.
    Future hardware-accelerated backends (e.g. an MLX path on Apple
    silicon) must honour the same precision or the cross-backend
    cross-check in
    ``tests/test_layered_green.py::test_cross_backend_precision``
    will fail. ``np.complex128`` is the default at every entry point
    and is preserved through the Sommerfeld quadrature; do not
    silently down-cast to FP32 in a derived backend.
    """

    rhos: tuple[float, ...]
    thicknesses: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.rhos) == 0:
            raise ValueError("LayeredEarth needs at least one layer")
        if len(self.thicknesses) != len(self.rhos) - 1:
            raise ValueError(
                f"thicknesses must have len(rhos)-1 entries; "
                f"got {len(self.rhos)} rhos, {len(self.thicknesses)} thicknesses"
            )
        for r in self.rhos:
            if r <= 0.0:
                raise ValueError(f"rho must be positive, got {r}")
        for h in self.thicknesses:
            if h <= 0.0:
                raise ValueError(f"layer thickness must be positive, got {h}")

    @property
    def n_layers(self) -> int:
        return len(self.rhos)


# ---------------------------------------------------------------------
# Reflection coefficients
# ---------------------------------------------------------------------


def reflection_coefficient_homogeneous(
    lambdas: np.ndarray, *, omega: float, sigma_earth: float,
) -> np.ndarray:
    """Magnetic reflection coefficient for a homogeneous half-space.

    .. math::

        \\Gamma_\\text{mag}^{(1)}(\\lambda) \\;=\\;
        \\frac{u_e - \\lambda}{u_e + \\lambda},
        \\qquad u_e \\;=\\; \\sqrt{\\lambda^2 + j\\omega\\mu_0\\sigma_e}.

    Parameters
    ----------
    lambdas : np.ndarray
        Spectral variable, shape ``(N_lam,)``. Must be non-negative.
    omega
        Angular frequency in rad/s.
    sigma_earth
        Earth conductivity in S/m.

    Returns
    -------
    Gamma : np.ndarray, complex, shape (N_lam,)
        Reflection coefficient at each $\\lambda$.
    """
    if omega <= 0.0 or sigma_earth <= 0.0:
        return np.zeros_like(lambdas, dtype=complex)
    u_e = np.sqrt(lambdas * lambdas + 1j * omega * MU_0 * sigma_earth)
    return (u_e - lambdas) / (u_e + lambdas)


def reflection_coefficient_layered(
    lambdas: np.ndarray, *, omega: float, earth: LayeredEarth,
) -> np.ndarray:
    """Magnetic reflection coefficient for an $n$-layer earth.

    Implements the recursive Tagg/Sunde-Wait formula

    .. math::

        \\Gamma_k(\\lambda) \\;=\\;
        \\frac{u_k - u_{k+1} - (u_k + u_{k+1})\\,\\Gamma_{k+1}\\,e^{-2 u_k h_k}}
             {u_k + u_{k+1} + (u_k - u_{k+1})\\,\\Gamma_{k+1}\\,e^{-2 u_k h_k}},

    starting from $\\Gamma_n = 0$ (semi-infinite bottom layer)
    and walking up to layer 1. The top-layer reflection is
    $\\Gamma_\\text{mag}^{(n)}(\\lambda) = (u_e - \\lambda)/(u_e+\\lambda)$
    with $u_e \\to u_1$ in the homogeneous limit, but for $n>1$
    the recursion modifies $u_1$ effectively. The formulation
    below is the standard one in Tleis 2008 §3.5.

    For $n=1$ this collapses to
    :func:`reflection_coefficient_homogeneous`.

    Parameters
    ----------
    lambdas : np.ndarray, shape (N_lam,)
    omega
        Angular frequency in rad/s.
    earth : LayeredEarth
        Layered-earth configuration.

    Returns
    -------
    Gamma : np.ndarray, complex
    """
    if omega <= 0.0:
        return np.zeros_like(lambdas, dtype=complex)
    n = earth.n_layers
    if n == 1:
        return reflection_coefficient_homogeneous(
            lambdas, omega=omega, sigma_earth=1.0 / earth.rhos[0],
        )
    # u_k for each layer k=1..n
    sigmas = np.array([1.0 / r for r in earth.rhos])
    u = [np.sqrt(lambdas * lambdas + 1j * omega * MU_0 * s) for s in sigmas]
    # Walk up from the bottom: Gamma_n = 0 at the bottom interface
    # (semi-infinite). At each interface k → k+1 we compose.
    # Convention: the air-layer above has u_a = lambda (quasi-static).
    # We compute the *effective* u_1 seen from above by recursing the
    # reflection at each subsurface interface. The final Gamma is
    # built from u_1 versus lambda.
    # Implementation follows Tleis 2008 eq. (3.55)+: build R_{n-1, n} = 0,
    # then propagate up with
    #   R_{k,k+1}_visible = (R_{k,k+1} + R_{k+1,k+2}_visible · e^{-2 u_{k+1} h_{k+1}})
    #                       / (1 + R_{k,k+1} · R_{k+1,k+2}_visible · e^{-2 u_{k+1} h_{k+1}})
    # then the top reflection seen by the air is the standard
    # (u_1 - lambda)/(u_1 + lambda) but with u_1 effectively replaced.
    # For the first implementation we use the simpler "top-layer
    # reflection composed with the next interface" form.
    # Build R_{k, k+1} for k=1..n-1
    R_internal = []
    for k in range(n - 1):
        R_k = (u[k] - u[k + 1]) / (u[k] + u[k + 1])
        R_internal.append(R_k)
    # Walk up: combined reflection at the top of layer k seen from layer 1
    # is built recursively. For the default case n=2, this is just
    #   Gamma_eff = (R_{0,1} + R_{1,2} e^{-2 u_1 h_1}) / (1 + R_{0,1} R_{1,2} e^{-2 u_1 h_1})
    # with R_{0,1} = (u_a - u_1)/(u_a + u_1) = (lambda - u_1)/(lambda + u_1) (sign opposite).
    # Hmm, for the magnetic case the air-side coefficient is
    # (u_1 - lambda)/(u_1 + lambda) (using the sign convention we picked).
    # The top reflection coefficient as seen by a wire in the air (or
    # at z=0) is *not* (u_1 - lambda)/(u_1 + lambda) for n>1; it is
    # the composed multilayer reflection. We compute it by walking
    # bottom-up:
    #
    #   R_eff[n-1] = 0  (no reflection from a semi-infinite bottom)
    #   R_eff[k]   = (R_{k,k+1} + R_eff[k+1] * exp(-2 u_{k+1} h_{k+1}))
    #                / (1 + R_{k,k+1} * R_eff[k+1] * exp(-2 u_{k+1} h_{k+1}))
    #               for k = n-2, n-3, ..., 0
    #   Gamma_top  = (u_1 - lambda + R_eff[0] (u_1 + lambda) ...) [doesn't quite work]
    #
    # Simpler: the magnetic reflection coefficient seen from above is
    #   Gamma_top(lambda) = (u_1 - lambda + (u_1 + lambda) R) / (u_1 + lambda + (u_1 - lambda) R)
    # where R is the *internal* reflection seen at the top of layer 1.
    # For n=2 this gives:
    #   R = R_{1,2} * exp(-2 u_1 h_1)
    # For n=3:
    #   R = (R_{1,2} + R_{2,3} exp(-2 u_2 h_2)) / (1 + R_{1,2} R_{2,3} exp(-2 u_2 h_2))
    #       times exp(-2 u_1 h_1)
    # In general we compute R_eff recursively:
    if n >= 2:
        # Start from below: R_eff[n-1] = 0 (no reflection past the
        # semi-infinite bottom).
        R_eff = np.zeros_like(lambdas, dtype=complex)
        for k in range(n - 2, -1, -1):
            # Interface k between layer k+1 and layer k+2 (1-indexed
            # k+1 and k+2 in math, 0-indexed in Python).
            R_k = R_internal[k]
            if k + 1 < len(earth.thicknesses):
                # Round-trip across layer (k+2) of thickness h_{k+2}
                # is captured below; here we account for the round trip
                # across layer (k+1)... wait, this gets confusing with
                # the indexing. Stick with the simpler 2-layer first.
                phase = np.exp(-2.0 * u[k + 1] * earth.thicknesses[k + 1])
            else:
                phase = np.zeros_like(lambdas, dtype=complex)
            R_eff = (R_k + R_eff * phase) / (1.0 + R_k * R_eff * phase)
        # Apply the round trip across layer 1
        R_top_internal = R_eff * np.exp(-2.0 * u[0] * earth.thicknesses[0])
        # Compose with the air-to-layer-1 reflection
        Gamma_top = (
            (u[0] - lambdas + (u[0] + lambdas) * R_top_internal)
            / (u[0] + lambdas + (u[0] - lambdas) * R_top_internal)
        )
        return Gamma_top
    # Should not reach here.
    return np.zeros_like(lambdas, dtype=complex)


# ---------------------------------------------------------------------
# Sommerfeld kernel evaluation (vectorised over rho/lambda)
# ---------------------------------------------------------------------


def _build_lambda_grid(
    *, z_sum: float, rho_max: float, omega: float, sigma_top: float,
    n_panels_per_oscillation: int = 8,
    n_log_nodes: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a two-regime Sommerfeld quadrature grid.

    The integrand has two distinct length scales:

    1. **Small $\\lambda$** — the reflection coefficient
       $\\Gamma_\\text{mag}$ varies smoothly between its two
       limits ($\\Gamma \\to +1$ at $\\lambda \\to 0$,
       $\\Gamma \\to 0$ at $\\lambda \\gg p_\\text{skin}$, with
       $p_\\text{skin} = \\sqrt{\\omega\\mu_0\\sigma}$). The
       Bessel function is approximately constant
       ($J_0(\\lambda\\rho) \\approx 1$) for $\\lambda\\rho \\ll 1$.
       Use a **logarithmic** grid here.
    2. **Large $\\lambda$** — $\\Gamma_\\text{mag} \\approx 0$,
       $(\\Gamma-1) \\approx -1$, and the integrand is dominated
       by $e^{-\\lambda z}\\,J_0(\\lambda\\rho)$. The Bessel
       function oscillates rapidly. Use a **uniform** grid with
       enough panels to resolve the oscillations.

    The split point is taken as
    $\\lambda_\\text{break} = \\max(p_\\text{skin}, 1/z_\\text{sum})$
    to ensure both sides of the σ transition are captured.

    Parameters
    ----------
    z_sum, rho_max, omega, sigma_top : float
        Geometry and material parameters; see module docstring.
    n_panels_per_oscillation
        Resolution of the Bessel oscillations on the uniform tail.
        8 × 8-point GL ≅ machine precision for typical ranges.
    n_log_nodes
        Number of nodes on the logarithmic small-$\\lambda$ part.

    Returns
    -------
    lambdas, weights : np.ndarray
        Combined Gauss-Legendre nodes and weights.
    """
    z_eff = max(z_sum, 1e-3)
    lambda_max = 30.0 / z_eff

    # Skin-depth-derived natural break.
    if omega > 0.0 and sigma_top > 0.0:
        p_skin = math.sqrt(omega * MU_0 * sigma_top)
    else:
        p_skin = 0.0
    lambda_break = max(p_skin, 1.0 / z_eff, 1e-9)
    if lambda_break > 0.5 * lambda_max:
        lambda_break = 0.5 * lambda_max

    # Logarithmic small-lambda region [eps, lambda_break]. Variable
    # change t = ln(lambda); the transformed integrand picks up a
    # Jacobian of lambda·dt = dlambda. Apply Gauss-Legendre on
    # [-1, 1] mapped to [ln(eps), ln(lambda_break)].
    eps = max(lambda_max * 1e-12, 1e-15)
    nodes_log_x, weights_log_x = np.polynomial.legendre.leggauss(n_log_nodes)
    a_log = math.log(eps)
    b_log = math.log(lambda_break)
    t = 0.5 * (nodes_log_x + 1.0) * (b_log - a_log) + a_log
    lambdas_log = np.exp(t)
    weights_log = 0.5 * weights_log_x * (b_log - a_log) * lambdas_log

    # Uniform panel grid on [lambda_break, lambda_max] resolving the
    # Bessel oscillations.
    span = lambda_max - lambda_break
    if rho_max > 0.0:
        oscillations = max(1, int(np.ceil(span * rho_max / (2.0 * math.pi))))
    else:
        oscillations = 1
    n_panels = min(n_panels_per_oscillation * oscillations, 4096)
    panel_width = span / n_panels

    nodes_panel, weights_panel = np.polynomial.legendre.leggauss(8)
    nodes_in_panel = 0.5 * (nodes_panel + 1.0)
    weights_in_panel = 0.5 * weights_panel

    panel_starts = lambda_break + np.arange(n_panels) * panel_width
    lambdas_lin = (
        panel_starts[:, None] + nodes_in_panel[None, :] * panel_width
    ).ravel()
    weights_lin = (
        np.ones(n_panels)[:, None] * (weights_in_panel * panel_width)[None, :]
    ).ravel()

    lambdas = np.concatenate([lambdas_log, lambdas_lin])
    weights = np.concatenate([weights_log, weights_lin])
    return lambdas, weights


def earth_return_correction_homogeneous(
    *,
    rho: float, z_i: float, z_j: float,
    omega: float, sigma_earth: float,
) -> complex:
    """σ-dependent earth-return correction beyond the additive mirror.

    Point-kernel form of the **buried–buried** Pollaczek correction
    (audit 2026-07-08, WP-C1; see the module docstring):

    $$
    \\Delta G \\;=\\; \\frac{e^{-\\gamma R} - 1}{R}
    \\;-\\; \\frac{1}{R'}
    \\;+\\; \\int_0^\\infty \\frac{\\lambda}{u_1}\\,
    \\Gamma(\\lambda)\\, e^{-u_1 (z + z')}\\,
    J_0(\\lambda\\rho)\\,d\\lambda,
    $$

    with $R = \\sqrt{\\rho^2 + (z - z')^2}$,
    $R' = \\sqrt{\\rho^2 + (z + z')^2}$. This is the σ-dependent
    piece *to be added to the ADR-0004 result* (which carries
    $1/R + 1/R'$): the $-1/R'$ removes the additive mirror, the
    attenuated direct term carries the buried-path earth-return
    resistance, and the $\\lambda/u_1$-weighted reflection is the
    genuine soil-side boundary response.

    Limit checks:

    - $\\sigma_e \\to 0$: $\\gamma \\to 0$, $\\Gamma \\to 0$ —
      correction $\\to -1/R'$ → cancels the ADR-0004 image, total
      → free space $1/R$ (at DC the soil is magnetically
      transparent: no image). ✓
    - $\\sigma_e \\to \\infty$: $e^{-\\gamma R} \\to 0$,
      $\\lambda/u_1 \\to 0$ — correction $\\to -1/R - 1/R'$, total
      → 0 (buried conductor screened by the surrounding medium). ✓
      *(The historic kernel instead returned the additive mirror in
      this limit, which is the electrostatic — not the magnetic —
      image; see the audit report.)*

    Parameters
    ----------
    rho
        Horizontal distance between source and field point in m.
        Must be ≥ 0.
    z_i, z_j
        Depths (positive into soil) in m. Must be > 0 for the
        integral to converge; for wires *at* the surface
        (Sunde-equivalent depth = 0) use a small regularisation
        $z = \\max(|\\text{depth}|, r)$ where $r$ is the wire
        radius.
    omega
        Angular frequency in rad/s.
    sigma_earth
        Earth conductivity in S/m.

    Returns
    -------
    correction : complex
        The kernel above, dimensionless (the calling
        :func:`build_sommerfeld_correction_matrix` multiplies by
        $\\mu_0 / (4\\pi)$ × ... × line integrations).
    """
    if omega <= 0.0 or sigma_earth <= 0.0:
        return 0.0 + 0.0j
    z_sum = z_i + z_j
    if z_sum <= 0.0:
        z_sum = 1e-3
    gamma2 = 1j * omega * MU_0 * sigma_earth
    gamma = complex(np.sqrt(gamma2))
    R_dir = math.sqrt(rho * rho + (z_i - z_j) ** 2)
    R_mir = math.sqrt(rho * rho + z_sum * z_sum)
    if R_dir < 1e-9:
        direct = -gamma
    else:
        direct = (np.exp(-gamma * R_dir) - 1.0) / R_dir
    lambdas, weights = _build_lambda_grid(
        z_sum=z_sum, rho_max=max(rho, 1e-6),
        omega=omega, sigma_top=sigma_earth,
    )
    Gamma = reflection_coefficient_homogeneous(
        lambdas, omega=omega, sigma_earth=sigma_earth,
    )
    u1 = np.sqrt(lambdas * lambdas + gamma2)
    decay = np.exp(-u1 * z_sum)
    bessel = j0(lambdas * rho)
    reflected = complex(
        np.sum(weights * (lambdas / u1) * Gamma * decay * bessel)
    )
    return complex(direct) - 1.0 / max(R_mir, 1e-9) + reflected


def earth_return_correction_layered(
    *,
    rho: float, z_i: float, z_j: float,
    omega: float, earth: LayeredEarth,
) -> complex:
    """Layered-earth analogue of :func:`earth_return_correction_homogeneous`.

    Uses :func:`reflection_coefficient_layered` for the reflected
    term; the $\\lambda/u_1$ weight, the exponents and the direct
    attenuation use the top-layer parameters (conductors live in
    layer 1). For ``earth.n_layers == 1`` it short-circuits to the
    single-layer formula.
    """
    if omega <= 0.0:
        return 0.0 + 0.0j
    if earth.n_layers == 1:
        return earth_return_correction_homogeneous(
            rho=rho, z_i=z_i, z_j=z_j,
            omega=omega, sigma_earth=1.0 / earth.rhos[0],
        )
    z_sum = z_i + z_j
    if z_sum <= 0.0:
        z_sum = 1e-3
    sigma_top = 1.0 / earth.rhos[0]
    gamma2 = 1j * omega * MU_0 * sigma_top
    gamma = complex(np.sqrt(gamma2))
    R_dir = math.sqrt(rho * rho + (z_i - z_j) ** 2)
    R_mir = math.sqrt(rho * rho + z_sum * z_sum)
    if R_dir < 1e-9:
        direct = -gamma
    else:
        direct = (np.exp(-gamma * R_dir) - 1.0) / R_dir
    lambdas, weights = _build_lambda_grid(
        z_sum=z_sum, rho_max=max(rho, 1e-6),
        omega=omega, sigma_top=sigma_top,
    )
    Gamma = reflection_coefficient_layered(lambdas, omega=omega, earth=earth)
    u1 = np.sqrt(lambdas * lambdas + gamma2)
    decay = np.exp(-u1 * z_sum)
    bessel = j0(lambdas * rho)
    reflected = complex(
        np.sum(weights * (lambdas / u1) * Gamma * decay * bessel)
    )
    return complex(direct) - 1.0 / max(R_mir, 1e-9) + reflected


# ---------------------------------------------------------------------
# Segment-pair integration via 16x16 Gauss-Legendre outer
# ---------------------------------------------------------------------


def _spectral_rho_interp(
    rho_grid: np.ndarray,
    z_ref: float,
    spectral_weights: np.ndarray,   # (nλ,) complex — kernel × λ-weights
    lambdas: np.ndarray,
    *,
    n_nodes: int = 24,
) -> np.ndarray:
    """Evaluate $S(\\rho) = \\sum_\\lambda w(\\lambda) J_0(\\lambda\\rho)$
    on the outer grid via 1-D interpolation.

    The spectral integral is a *smooth, monotone-decaying* function
    of $\\rho$ (the $J_0$ oscillations integrate out), so it is
    evaluated exactly on ``n_nodes`` log-spaced $\\rho$-nodes and
    linearly interpolated in $t = \\log(\\rho + z_\\text{ref})$ to
    the full 16×16 grid. This replaces the historic
    ``(16, 16, nλ)`` tensor evaluation, whose cost exploded with the
    oscillation-resolved λ-grid for distant segment pairs
    (~2 s/pair at ρ = 200 m; now milliseconds) — audit 2026-07-08,
    WP-C1 performance note.
    """
    rho_min = float(rho_grid.min())
    rho_max = float(rho_grid.max())
    if rho_max - rho_min < 1e-9:
        s_val = complex(np.sum(spectral_weights * j0(lambdas * rho_min)))
        return np.full(rho_grid.shape, s_val, dtype=complex)
    z_off = max(z_ref, 1e-3)
    t_lo = math.log(rho_min + z_off)
    t_hi = math.log(rho_max + z_off)
    t_nodes = np.linspace(t_lo, t_hi, n_nodes)
    rho_nodes = np.exp(t_nodes) - z_off
    # Exact evaluation at the nodes: (n_nodes, nλ) — small.
    s_nodes = (
        j0(lambdas[None, :] * np.maximum(rho_nodes, 0.0)[:, None])
        @ spectral_weights
    )
    t_grid = np.log(rho_grid + z_off)
    s_re = np.interp(t_grid.ravel(), t_nodes, s_nodes.real)
    s_im = np.interp(t_grid.ravel(), t_nodes, s_nodes.imag)
    return (s_re + 1j * s_im).reshape(rho_grid.shape)


def _pollaczek_inner_kernel(
    rho_grid: np.ndarray,      # (16, 16) horizontal distances
    dz_grid: np.ndarray,       # (16, 16) signed z_a - z_b
    depth_a: np.ndarray,       # (16,)   |z| along segment a
    depth_b: np.ndarray,       # (16,)   |z| along segment b
    buried_a: bool,
    buried_b: bool,
    *,
    omega: float,
    sigma_top: float,
    lambdas: np.ndarray,
    lambda_weights: np.ndarray,
    Gamma: np.ndarray,
) -> np.ndarray:
    """Pollaczek correction kernel on the outer 16×16 node grid.

    Returns the correction **beyond the ADR-0004 additive-mirror
    baseline** (see module docstring): the closed-form mirror
    deduplication $-1/R'$, plus (buried–buried) the closed-form
    direct-term attenuation $(e^{-\\gamma R} - 1)/R$, plus the
    numerically integrated reflected/transmitted remainder of the
    case-specific kernel. The spectral part uses the pair's mean
    depth sum (segments assumed approximately horizontal — same
    assumption as the historic kernel reuse, ADR-0006 numerical
    notes) and is interpolated over $\\rho$
    (:func:`_spectral_rho_interp`).
    """
    gamma2 = 1j * omega * MU_0 * sigma_top
    gamma = np.sqrt(gamma2)
    u1 = np.sqrt(lambdas * lambdas + gamma2)

    ha = depth_a[:, None]                     # (16, 1)
    hb = depth_b[None, :]                     # (1, 16)

    if buried_a and buried_b:
        z_sum = ha + hb
        z_ref = float(z_sum.mean())
        # Closed-form mirror dedup −1/R' (mirror at z = −h_b).
        R_mir = np.sqrt(rho_grid ** 2 + z_sum ** 2)
        inner = -1.0 / np.maximum(R_mir, 1e-9) + 0.0j
        # Closed-form direct-term attenuation (e^{−γR} − 1)/R —
        # smooth at R → 0 (limit −γ); carries the earth-return
        # resistance of the buried path.
        R3 = np.sqrt(rho_grid ** 2 + dz_grid ** 2)
        small = R3 < 1e-9
        R3_safe = np.where(small, 1.0, R3)
        direct = np.where(
            small, -gamma, (np.exp(-gamma * R3_safe) - 1.0) / R3_safe,
        )
        inner = inner + direct
        # Reflected remainder: (λ/u1)·Γ·e^{−u1(h+h')} — decays like
        # Γ ~ γ²/(4λ²) at large λ, absolutely convergent.
        spectral_w = (lambdas / u1) * Gamma * np.exp(-u1 * z_ref) \
            * lambda_weights
        inner = inner + _spectral_rho_interp(
            rho_grid, z_ref, spectral_w, lambdas,
        )
        return inner

    if (not buried_a) and (not buried_b):
        # Overhead pair — Carson geometry: air-side reflection −Γ.
        z_sum = ha + hb
        z_ref = float(z_sum.mean())
        R_mir = np.sqrt(rho_grid ** 2 + z_sum ** 2)
        inner = -1.0 / np.maximum(R_mir, 1e-9) + 0.0j
        spectral_w = -Gamma * np.exp(-lambdas * z_ref) * lambda_weights
        inner = inner + _spectral_rho_interp(
            rho_grid, z_ref, spectral_w, lambdas,
        )
        return inner

    # Mixed pair: transmission through the soil surface. Assign the
    # buried depth h and overhead height H per grid axis.
    if buried_a:
        h_g, H_g = ha, hb
    else:
        h_g, H_g = hb, ha
    h_ref = float(h_g.mean())
    H_ref = float(H_g.mean())
    # Mirror dedup: the ADR-0004 mirror of the buried segment sits at
    # −h, i.e. on the air side → mirror distance carries |h − H|.
    R_mir = np.sqrt(rho_grid ** 2 + (h_g - H_g) ** 2)
    inner = -1.0 / np.maximum(R_mir, 1e-9) + 0.0j
    # Transmitted kernel minus the free-space direct term (the
    # direct 1/R lives in the Neumann matrix): both decay with
    # (h + H), so the remainder is integrable on the shared grid.
    spectral_w = (
        (2.0 * lambdas / (u1 + lambdas))
        * np.exp(-u1 * h_ref - lambdas * H_ref)
        - np.exp(-lambdas * (h_ref + H_ref))
    ) * lambda_weights
    inner = inner + _spectral_rho_interp(
        rho_grid, h_ref + H_ref, spectral_w, lambdas,
    )
    return inner


def _pollaczek_pair_integral(
    p1_a: np.ndarray, p2_a: np.ndarray,
    p1_b: np.ndarray, p2_b: np.ndarray,
    *,
    omega: float,
    sigma_top: float,
    gamma_provider,
) -> complex:
    """Shared geometric integration for both earth models.

    ``gamma_provider(lambdas)`` returns the reflection coefficient
    array (homogeneous or layered).
    """
    p1_a = np.asarray(p1_a, dtype=float)
    p2_a = np.asarray(p2_a, dtype=float)
    p1_b = np.asarray(p1_b, dtype=float)
    p2_b = np.asarray(p2_b, dtype=float)
    if omega <= 0.0 or sigma_top <= 0.0:
        return 0.0 + 0.0j
    da = p2_a - p1_a
    db = p2_b - p1_b
    la = float(np.linalg.norm(da))
    lb = float(np.linalg.norm(db))
    if la <= 0.0 or lb <= 0.0:
        return 0.0 + 0.0j
    ua = da / la
    ub = db / lb
    dot = float(ua @ ub)
    if abs(dot) < 1e-12:
        return 0.0 + 0.0j  # orthogonal segments

    # Outer grids.
    s_nodes = 0.5 * (_OUTER_GL_NODES + 1.0)
    w_nodes = 0.5 * _OUTER_GL_WEIGHTS
    pts_a = p1_a[None, :] + s_nodes[:, None] * da[None, :]
    pts_b = p1_b[None, :] + s_nodes[:, None] * db[None, :]
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    rho_grid = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
    dz_grid = diff[:, :, 2]
    z_a = pts_a[:, 2]
    z_b = pts_b[:, 2]
    depth_a = np.abs(z_a)
    depth_b = np.abs(z_b)
    # Side dispatch by segment midpoint (segments crossing z = 0 are
    # blocked upstream by the surface-plane guard in
    # build_inductance_matrix).
    buried_a = bool(0.5 * (z_a[0] + z_a[-1]) >= 0.0)
    buried_b = bool(0.5 * (z_b[0] + z_b[-1]) >= 0.0)

    z_sum_grid = depth_a[:, None] + depth_b[None, :]
    z_sum_worst = max(float(z_sum_grid.min()), 1e-3)
    rho_max = float(rho_grid.max() + 1e-9)
    lambdas, lambda_weights = _build_lambda_grid(
        z_sum=z_sum_worst, rho_max=rho_max,
        omega=omega, sigma_top=sigma_top,
    )
    Gamma = gamma_provider(lambdas)

    inner = _pollaczek_inner_kernel(
        rho_grid, dz_grid, depth_a, depth_b, buried_a, buried_b,
        omega=omega, sigma_top=sigma_top,
        lambdas=lambdas, lambda_weights=lambda_weights, Gamma=Gamma,
    )

    outer_w = w_nodes[:, None] * w_nodes[None, :]
    geom_integral = complex(np.sum(outer_w * inner))
    geom_integral *= la * lb * dot

    return 1j * omega * MU_0 / (4.0 * math.pi) * geom_integral


def sommerfeld_pair_integral_homogeneous(
    p1_a: np.ndarray, p2_a: np.ndarray,
    p1_b: np.ndarray, p2_b: np.ndarray,
    *,
    omega: float, sigma_earth: float,
) -> complex:
    """Integrate the σ-dependent magnetic Green function over a segment pair.

    Computes

    $$
    \\Delta Z^{(i,j)}_\\text{Sommerfeld} \\;=\\; \\frac{j\\omega\\mu_0}{4\\pi}
    \\int_{C_i}\\!\\!\\int_{C_j}
    (\\hat{l}_i\\cdot\\hat{l}_j)\\,
    \\Delta G_\\text{mag}(\\vec{r}_i, \\vec{r}_j;\\,\\omega,\\sigma_e)\\,
    dl_i\\,dl_j,
    $$

    where $\\Delta G_\\text{mag}$ is the σ-dependent earth-return
    correction beyond the ADR-0004 additive-mirror baseline
    (vanishes for σ → 0 together with the mirror; see the module
    docstring for the case-dispatched Pollaczek kernels). Used by
    :func:`build_sommerfeld_correction_matrix`.

    The integration is 16×16 Gauss–Legendre over the two segment
    parameterisations. The Sommerfeld inner integral is **not**
    re-evaluated at every outer node — the spectral kernel is
    computed once per segment pair on a shared λ-grid and reused
    (valid because the segments are assumed approximately horizontal
    so the depth sum varies negligibly along them; see ADR-0006
    numerical notes). The closed-form pieces (mirror deduplication
    $-1/R'$; buried–buried direct attenuation
    $(e^{-\\gamma R} - 1)/R$) are evaluated exactly per outer node.

    Returns
    -------
    Z : complex
        Per-pair earth-return correction in $\\Omega$ (already
        includes the $j\\omega\\mu_0/(4\\pi)$ pre-factor).
    """
    if omega <= 0.0 or sigma_earth <= 0.0:
        return 0.0 + 0.0j

    def _gamma(lambdas: np.ndarray) -> np.ndarray:
        return reflection_coefficient_homogeneous(
            lambdas, omega=omega, sigma_earth=sigma_earth,
        )

    return _pollaczek_pair_integral(
        p1_a, p2_a, p1_b, p2_b,
        omega=omega, sigma_top=sigma_earth, gamma_provider=_gamma,
    )


def sommerfeld_pair_integral_layered(
    p1_a: np.ndarray, p2_a: np.ndarray,
    p1_b: np.ndarray, p2_b: np.ndarray,
    *,
    omega: float, earth: LayeredEarth,
) -> complex:
    """Layered-earth analogue of :func:`sommerfeld_pair_integral_homogeneous`.

    The reflected term uses the layered reflection coefficient; the
    $\\lambda/u_1$ weight, the $e^{-u_1(\\cdot)}$ exponents and the
    buried direct-term attenuation use the **top-layer** $u_1$ /
    $\\gamma_1$ (the conductors live in layer 1 — deeper layers act
    through the reflection coefficient only).
    """
    if earth.n_layers == 1:
        return sommerfeld_pair_integral_homogeneous(
            p1_a, p2_a, p1_b, p2_b,
            omega=omega, sigma_earth=1.0 / earth.rhos[0],
        )
    if omega <= 0.0:
        return 0.0 + 0.0j
    sigma_top = 1.0 / earth.rhos[0]

    def _gamma(lambdas: np.ndarray) -> np.ndarray:
        return reflection_coefficient_layered(
            lambdas, omega=omega, earth=earth,
        )

    return _pollaczek_pair_integral(
        p1_a, p2_a, p1_b, p2_b,
        omega=omega, sigma_top=sigma_top, gamma_provider=_gamma,
    )


# ---------------------------------------------------------------------
# Full M×M correction matrix
# ---------------------------------------------------------------------


def build_sommerfeld_correction_matrix(
    seg_endpoints: np.ndarray,        # shape (M, 2, 3)
    wire_radii: np.ndarray,           # shape (M,) — currently unused but kept for API parity
    *,
    omega: float,
    earth: LayeredEarth,
) -> np.ndarray:
    """Assemble the dense Sommerfeld earth-return correction matrix.

    The output is the σ-dependent addition to the perfect-mirror
    Neumann inductance matrix from
    :func:`groundfield.coupling.inductance.build_inductance_matrix`.
    The two should be added (after multiplying $L_\\text{Neumann}$
    by $j\\omega$):

    .. code-block:: python

        Z_b = jω · L_Neumann + dZ_Sommerfeld

    For ``earth.n_layers == 1`` this is the homogeneous-earth
    Sommerfeld kernel; for ``n >= 2`` the layered Pollaczek-Wait
    kernel is used.

    Parameters
    ----------
    seg_endpoints
        Array of shape ``(M, 2, 3)`` with the start- and end-points
        of every distributed-conductor longitudinal-branch segment.
    wire_radii
        Per-branch wire radii. Currently unused (the radius is
        already in the perfect-mirror diagonal handled by
        ``build_inductance_matrix``); reserved for future use when
        the diagonal needs a wire-radius regularisation in the
        Sommerfeld self-pair integral.
    omega
        Angular frequency in rad/s.
    earth
        Layered-earth configuration.

    Returns
    -------
    dZ : np.ndarray, shape (M, M), dtype complex
        Symmetric Sommerfeld correction matrix in $\\Omega$.
    """
    if omega <= 0.0:
        return np.zeros(
            (seg_endpoints.shape[0], seg_endpoints.shape[0]),
            dtype=complex,
        )
    M = seg_endpoints.shape[0]
    dZ = np.zeros((M, M), dtype=complex)
    for i in range(M):
        p1_i = seg_endpoints[i, 0]
        p2_i = seg_endpoints[i, 1]
        for j in range(i, M):
            p1_j = seg_endpoints[j, 0]
            p2_j = seg_endpoints[j, 1]
            val = sommerfeld_pair_integral_layered(
                p1_i, p2_i, p1_j, p2_j,
                omega=omega, earth=earth,
            )
            dZ[i, j] = dZ[j, i] = val
    return dZ
