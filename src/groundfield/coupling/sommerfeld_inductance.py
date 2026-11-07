"""Geometric Sommerfeld earth-return Green function (ADR-0006).

This module implements the rigorous geometric formulation of the
inductive earth-return coupling described in ADR-0006. Compared to
ADR-0005's Carson per-meter scaling, it integrates the actual
vector-potential Green's function over the segment-pair geometry,
which correctly handles short wires, non-parallel arrangements,
and layered earth.

Mathematical background
-----------------------
For a horizontal current element in a stratified, quasi-static
medium ($z > 0$ soil, $z < 0$ air, layer $k$ with
$\\gamma_k^2 = j\\omega\\mu_0\\sigma_k$ and vertical wavenumber
$u_k = \\sqrt{\\lambda^2 + \\gamma_k^2}$, $u_0 = \\lambda$ in air;
time convention $e^{+j\\omega t}$), the vector-potential Green's
function of a source at depth $z'$ and an observer at depth $z$ is

$$
G(\\rho, z, z') \\;=\\; \\delta_{kj}\\,\\frac{e^{-\\gamma_k R}}{R}
\\;+\\; \\int_0^{\\infty}\\! \\lambda\\,F_{kj}(\\lambda; z, z')\\,
J_0(\\lambda\\rho)\\,d\\lambda ,
$$

with $k$ / $j$ the layers of observer / source. The spectral amplitude
$F$ follows from the continuity of $A$ and $\\partial A/\\partial z$ at
every interface. For source and observer in the **same** layer $k$
(top at $z_{k-1}$, thickness $t$, depths $h, h'$ measured from its
top) it is the closed form

$$
F_{kk} \\;=\\; \\frac{\\tilde R_\\uparrow e^{-u_k(h+h')}
 + \\tilde R_\\downarrow e^{-u_k(2t-h-h')}
 + \\tilde R_\\uparrow\\tilde R_\\downarrow\\bigl[e^{-u_k(2t-(h-h'))}
 + e^{-u_k(2t+(h-h'))}\\bigr]}
{u_k\\,\\bigl(1 - \\tilde R_\\uparrow \\tilde R_\\downarrow e^{-2u_k t}\\bigr)},
$$

where $\\tilde R_\\uparrow$, $\\tilde R_\\downarrow$ are the
**generalised reflection coefficients** of the stack above and below
layer $k$ (bottom-up / top-down recursions over
$r_{k,m} = (u_k - u_m)/(u_k + u_m)$). For the homogeneous half-space
this is the familiar $(\\lambda/u_1)\\Gamma e^{-u_1(h+h')}$ with
$\\Gamma = (u_1-\\lambda)/(u_1+\\lambda)$; for two layers it is the
four-term buried-in-layer-1 amplitude validated against Tsiamitros
(2005) Eq. (8) (WP-D2, 2026-07-20); for $n \\ge 3$ layers the
generalised $\\tilde R_\\downarrow$ makes it exact as well (review
pass 10, finding 10.35 — the historic $n \\ge 3$ path used the
single air-image form and was 17–25 % off in $X'$). Pairs in
different layers (including a buried / overhead pair over a layered
earth) use the transmitted amplitude, obtained from the same interface
conditions by a small linear solve per $\\lambda$.

The overhead pair keeps Carson's geometry,
$\\lambda F_{00} = -\\Gamma_s(\\lambda) e^{-\\lambda(H+H')}$, with the
surface reflection coefficient $\\Gamma_s$ of the whole stack.

Earth-return correction and image conventions
---------------------------------------------
The functions here return the **earth-return correction**
$\\Delta Z$ that is added to $j\\omega L$, the Neumann
partial-inductance matrix of
:func:`groundfield.coupling.inductance.build_inductance_matrix`. The
correction depends on which image that matrix carries (``image=``):

- ``"none"`` — $\\Delta G = G - 1/R$, the exact quasi-static
  correction to free space. At DC the soil is magnetically
  transparent: no image at all.
- ``"additive"`` (historic default, ADR-0004) — the matrix carries
  the perfect-magnetic-mirror image $L_\\text{img}$; the correction
  removes it **exactly**:
  $\\Delta Z_\\text{additive} = \\Delta Z_\\text{none} - j\\omega
  L_\\text{img}$, with $L_\\text{img}$ computed by the same routine
  that added it (closed form for parallel pairs, the mirrored
  orientation for inclined and vertical segments).
- ``"antiparallel"`` — $\\Delta Z_\\text{none} + j\\omega
  L_\\text{img}$.

Before review pass 10 the image was removed as $-1/R'$ integrated by
the 16x16 outer rule and weighted with the *direct* orientation:
vertical segments kept twice the image in the $\\sigma \\to 0$ limit,
and segments much longer than twice their depth lost 0.6 % (25 m) to
5.7 % (50 m) of the image term (finding 10.46).

Numerical method
----------------
Each segment pair is integrated by 16x16 Gauss–Legendre over the two
segments; the closed-form direct term $(e^{-\\gamma R}-1)/R$ is
evaluated at every node, the spectral part
$S(\\rho) = \\int \\lambda F J_0(\\lambda\\rho)\\,d\\lambda$ on 24
log-spaced $\\rho$ nodes and interpolated (segments assumed
approximately horizontal, mean depths per segment). Segments that cross
the soil surface or a layer interface are split there first.

The $\\lambda$ quadrature (review pass 10, finding 10.36) follows the
F14 policy of :mod:`groundfield.coupling.layered_green`: a
logarithmic region on $[\\epsilon, \\lambda_\\text{b}]$ with
$\\lambda_\\text{b} \\le \\pi/(4\\rho_\\text{max})$ so that it is free
of $J_0$ oscillations, followed by Gauss panels of width
$\\le \\pi/\\rho_\\text{max}$ (16 nodes per $J_0$ period), geometrically
graded near the skin and decay scales, up to
$\\lambda_\\text{max} = 30/z$. Before the fix the log region ended at
$\\max(p_\\text{skin}, 1/z)$ regardless of the distance, so $J_0$ went
unresolved for distant pairs (Re −24 % at 200 m, +274 % at 400 m),
and a 4096-panel cap truncated silently. The node budget is now
explicit: beyond a soft limit the density drops to 8 nodes per period,
and a :class:`~groundfield.coupling.layered_green.SommerfeldResolutionWarning`
is raised instead of a silent truncation when even that does not fit.

:func:`build_sommerfeld_correction_matrices` evaluates several
frequencies at once (finding 10.60): geometry, $\\lambda$ grid and the
Bessel table are built once per pair and reused for every frequency
whose grid is identical (the common case — the grid depends on the
frequency only through skin scales above $1/z$). Each frequency's
matrix equals the single-frequency result bit for bit.

Limit checks (built into the test suite)
----------------------------------------
- $\\sigma_e\\to 0$: $\\Delta Z_\\text{none} \\to 0$, and with
  ``image="additive"`` the correction cancels the mirror exactly for
  every orientation.
- $\\sigma_e\\to\\infty$, overhead pair: $-\\Gamma \\to -1$, the net
  image becomes the anti-parallel PEC image (Carson's baseline).
- $\\sigma_e\\to\\infty$, buried pair: $e^{-\\gamma R} \\to 0$ — the
  conductor is screened by the surrounding medium.
- Long parallel buried wires: the per-metre limit reproduces
  Pollaczek's $\\frac{j\\omega\\mu_0}{2\\pi}\\bigl[K_0(\\gamma d)
  - K_0(\\gamma D') + J_P\\bigr]$.
- A three-layer earth with $\\rho_2 = \\rho_3$ reproduces the
  two-layer result.

References
----------
- Stratton, J. A. (1941). *Electromagnetic Theory*, McGraw-Hill,
  §9-10 — derivation of the half-space vector potential.
- Sommerfeld, A. (1909). Über die Ausbreitung der Wellen in der
  drahtlosen Telegraphie. *Ann. Phys.* **28**(4), 665–736.
- Wait, J. R. (1972). *Electromagnetic Waves in Stratified
  Media*, Pergamon. Ch. 3.
- Chew, W. C. (1995). *Waves and Fields in Inhomogeneous Media*,
  IEEE Press, §2.1 (generalised reflection coefficients).
- Tsiamitros, D. A. et al. (2005). Earth return path impedances of
  underground cables for the two-layer earth case. *IEEE Trans.
  Power Delivery* **20**(3), 2174–2181.
"""

from __future__ import annotations

import bisect
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import j0

from groundfield.coupling.inductance import (
    ImageConvention,
    _check_image_geometry,
    _image_inductance_matrix,
    _mirror,
    _resolve_image,
    neumann_mutual,
)
from groundfield.coupling.layered_green import SommerfeldResolutionWarning

__all__ = [
    "MU_0",
    "ImageConvention",
    "SommerfeldResolutionWarning",
    "earth_return_correction_homogeneous",
    "earth_return_correction_layered",
    "sommerfeld_pair_integral_homogeneous",
    "sommerfeld_pair_integral_layered",
    "build_sommerfeld_correction_matrix",
    "build_sommerfeld_correction_matrices",
    "LayeredEarth",
    "reflection_coefficient_homogeneous",
    "reflection_coefficient_layered",
]


MU_0 = 4.0e-7 * math.pi

# Outer (geometry) Gauss–Legendre rule: the same 16x16 grid as
# coupling/inductance.py.
_OUTER_GL_NODES, _OUTER_GL_WEIGHTS = np.polynomial.legendre.leggauss(16)

# --- lambda quadrature policy (review pass 10, finding 10.36) ---------
#
# Log region: n_log Gauss nodes in t = ln(lambda) on [eps, lambda_b].
_N_LOG_NODES = 64
# Gauss order of the panels of the linear region.
_PANEL_ORDER = 8
# Target: two panels (16 nodes) per J0 period 2 pi / rho_max.
_PANELS_PER_PERIOD = 2
# Minimum number of panels of the linear region (historic floor).
_MIN_PANELS = 8
# Above this panel count the density drops towards one panel (8 nodes)
# per period, the floor that layered_green also accepts.
_SOFT_PANEL_BUDGET = 1 << 14
# Hard allocation budget (2^17 panels = 1 Mi nodes). Exceeding it
# raises SommerfeldResolutionWarning instead of truncating silently.
_HARD_PANEL_BUDGET = 1 << 17
# Graded panels resolve the reflection-coefficient transitions up to
# this multiple of the largest physical scale (skin / decay / layer).
_GRADING_SPAN = 4.0
# Number of rho interpolation nodes per segment pair (historic design).
_N_RHO_NODES = 24


@lru_cache(maxsize=None)
def _leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Read-only Gauss–Legendre nodes / weights of order ``n`` (cached).

    ``numpy.polynomial.legendre.leggauss`` solves an eigenvalue problem
    on every call; the historic code paid it twice per segment pair
    and frequency (14 % of an inductive solve, review pass 10, 10.60).
    """
    x, w = np.polynomial.legendre.leggauss(n)
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


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
    silicon) must honour the same precision. ``np.complex128`` is the
    default at every entry point and is preserved through the
    Sommerfeld quadrature; do not silently down-cast to FP32 in a
    derived backend.
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
            if not (r > 0.0) or not math.isfinite(r):
                raise ValueError(f"rho must be positive and finite, got {r}")
        for h in self.thicknesses:
            if not (h > 0.0) or not math.isfinite(h):
                raise ValueError(
                    f"layer thickness must be positive and finite, got {h}"
                )

    @property
    def n_layers(self) -> int:
        """Number of layers, including the semi-infinite bottom layer."""
        return len(self.rhos)

    @property
    def interface_depths(self) -> tuple[float, ...]:
        """Depths of the soil-layer interfaces below the surface, in m."""
        out: list[float] = []
        acc = 0.0
        for h in self.thicknesses:
            acc += float(h)
            out.append(acc)
        return tuple(out)

    def layer_of(self, depth: float) -> int:
        """Layer index (1-based) that contains ``depth`` (m, > 0 in soil).

        A point exactly on an interface belongs to the lower layer.
        """
        return 1 + bisect.bisect_right(self.interface_depths, float(depth))


# ---------------------------------------------------------------------
# Reflection coefficients
# ---------------------------------------------------------------------


def reflection_coefficient_homogeneous(
    lambdas: np.ndarray,
    *,
    omega: float,
    sigma_earth: float,
) -> np.ndarray:
    """Magnetic reflection coefficient for a homogeneous half-space.

    .. math::

        \\Gamma_\\text{mag}^{(1)}(\\lambda) \\;=\\;
        \\frac{u_e - \\lambda}{u_e + \\lambda},
        \\qquad u_e \\;=\\; \\sqrt{\\lambda^2 + j\\omega\\mu_0\\sigma_e}.

    Parameters
    ----------
    lambdas : np.ndarray
        Spectral variable in 1/m, shape ``(N_lam,)``. Must be
        non-negative.
    omega
        Angular frequency in rad/s.
    sigma_earth
        Earth conductivity in S/m.

    Returns
    -------
    Gamma : np.ndarray, complex, shape (N_lam,)
        Reflection coefficient at each $\\lambda$ (dimensionless).
    """
    if omega <= 0.0 or sigma_earth <= 0.0:
        return np.zeros_like(lambdas, dtype=complex)
    u_e = np.sqrt(lambdas * lambdas + 1j * omega * MU_0 * sigma_earth)
    return (u_e - lambdas) / (u_e + lambdas)


def reflection_coefficient_layered(
    lambdas: np.ndarray,
    *,
    omega: float,
    earth: LayeredEarth,
) -> np.ndarray:
    """Magnetic surface reflection coefficient of an $n$-layer earth.

    The generalised reflection coefficient of the whole stack seen
    from the air, in the convention of
    :func:`reflection_coefficient_homogeneous` (so that the air-side
    reflection of a downward wave is $-\\Gamma$). It is built from the
    deepest interface upwards,

    .. math::

        \\Gamma \\leftarrow \\frac{r_k + \\Gamma\\,e^{-2u_{k+1}h_{k+1}}}
                                  {1 + r_k\\,\\Gamma\\,e^{-2u_{k+1}h_{k+1}}},
        \\qquad r_k = \\frac{u_{k+1} - u_k}{u_{k+1} + u_k},

    with $u_0 = \\lambda$ (air) and $u_k$ the vertical wavenumber of
    layer $k$. It reduces to the homogeneous $\\rho_1$ half-space as
    $h_1 \\to \\infty$ and to the $\\rho_2$ half-space as $h_1 \\to 0$.
    For $n = 1$ it is :func:`reflection_coefficient_homogeneous`.

    Parameters
    ----------
    lambdas : np.ndarray
        Spectral variable in 1/m, shape ``(N_lam,)``.
    omega
        Angular frequency in rad/s.
    earth : LayeredEarth
        Layered-earth configuration.

    Returns
    -------
    Gamma : np.ndarray, complex
        Surface reflection coefficient (dimensionless).
    """
    if omega <= 0.0:
        return np.zeros_like(lambdas, dtype=complex)
    n = earth.n_layers
    if n == 1:
        return reflection_coefficient_homogeneous(
            lambdas,
            omega=omega,
            sigma_earth=1.0 / earth.rhos[0],
        )
    u = _vertical_wavenumbers(lambdas, omega, earth)
    gamma = (u[n] - u[n - 1]) / (u[n] + u[n - 1])  # r_{n-1}, deepest interface
    for k in range(n - 2, -1, -1):
        r_k = (u[k + 1] - u[k]) / (u[k + 1] + u[k])
        phase = np.exp(-2.0 * u[k + 1] * earth.thicknesses[k])
        gamma = (r_k + gamma * phase) / (1.0 + r_k * gamma * phase)
    return gamma


def _vertical_wavenumbers(
    lambdas: np.ndarray, omega: float, earth: LayeredEarth
) -> list[np.ndarray]:
    """``[u_0, u_1, ..., u_n]`` with ``u_0 = lambda`` (air), in 1/m."""
    u = [np.asarray(lambdas, dtype=float).astype(complex)]
    for rho_k in earth.rhos:
        u.append(np.sqrt(lambdas * lambdas + 1j * omega * MU_0 / rho_k))
    return u


def _reflection_up(u: list[np.ndarray], earth: LayeredEarth, k: int) -> np.ndarray:
    """Generalised reflection coefficient looking **up** from layer ``k``.

    $\\tilde R_\\uparrow$ at the top interface of layer ``k`` (1-based),
    including the multiple reflections in every layer above it and the
    air. Local coefficients $r_{k,m} = (u_k - u_m)/(u_k + u_m)$.
    """
    R = (u[1] - u[0]) / (u[1] + u[0])  # layer 1 looking into the air
    for m in range(2, k + 1):
        r = (u[m] - u[m - 1]) / (u[m] + u[m - 1])
        phase = np.exp(-2.0 * u[m - 1] * earth.thicknesses[m - 2])
        R = (r + R * phase) / (1.0 + r * R * phase)
    return R


def _reflection_down(
    u: list[np.ndarray], earth: LayeredEarth, k: int
) -> np.ndarray | None:
    """Generalised reflection coefficient looking **down** from layer ``k``.

    $\\tilde R_\\downarrow$ at the bottom interface of layer ``k``;
    ``None`` for the semi-infinite bottom layer.
    """
    n = earth.n_layers
    if k >= n:
        return None
    R = (u[n - 1] - u[n]) / (u[n - 1] + u[n])
    for m in range(n - 2, k - 1, -1):
        r = (u[m] - u[m + 1]) / (u[m] + u[m + 1])
        phase = np.exp(-2.0 * u[m + 1] * earth.thicknesses[m])
        R = (r + R * phase) / (1.0 + r * R * phase)
    return R


def _same_layer_weight(
    lambdas: np.ndarray,
    u: list[np.ndarray],
    earth: LayeredEarth,
    k: int,
    h_obs: float,
    h_src: float,
) -> np.ndarray:
    """$\\lambda F_{kk}$ — reflected amplitude, both points in layer ``k``.

    ``h_obs`` / ``h_src`` are the depths below the **top of layer k**
    in m. Exact for any number of layers (generalised reflection
    coefficients, see the module docstring).
    """
    uk = u[k]
    R_up = _reflection_up(u, earth, k)
    R_dn = _reflection_down(u, earth, k)
    if R_dn is None:
        return lambdas * R_up * np.exp(-uk * (h_obs + h_src)) / uk
    t = float(earth.thicknesses[k - 1])
    dh = h_obs - h_src
    num = (
        R_up * np.exp(-uk * (h_obs + h_src))
        + R_dn * np.exp(-uk * (2.0 * t - h_obs - h_src))
        + R_up * R_dn * (np.exp(-uk * (2.0 * t - dh)) + np.exp(-uk * (2.0 * t + dh)))
    )
    den = uk * (1.0 - R_up * R_dn * np.exp(-2.0 * uk * t))
    return lambdas * num / den


def _mixed_layer1_weight(
    lambdas: np.ndarray,
    u: list[np.ndarray],
    earth: LayeredEarth,
    h_buried: float,
    height_air: float,
) -> np.ndarray:
    """$\\lambda F_{01}$ — buried point in layer 1, the other in the air.

    Transmitted amplitude through the soil surface including the
    reflections from the layers below (exact for any $n$); reduces to
    $2\\lambda e^{-u_1 h - \\lambda H}/(u_1 + \\lambda)$ for a
    homogeneous earth.
    """
    u1 = u[1]
    R_up = _reflection_up(u, earth, 1)
    R_dn = _reflection_down(u, earth, 1)
    base = 2.0 * lambdas / (u1 + lambdas) * np.exp(-lambdas * height_air)
    if R_dn is None:
        return base * np.exp(-u1 * h_buried)
    t = float(earth.thicknesses[0])
    num = np.exp(-u1 * h_buried) + R_dn * np.exp(-u1 * (2.0 * t - h_buried))
    den = 1.0 - R_up * R_dn * np.exp(-2.0 * u1 * t)
    return base * num / den


def _generic_weight(
    lambdas: np.ndarray,
    u: list[np.ndarray],
    earth: LayeredEarth,
    z_obs: float,
    z_src: float,
) -> np.ndarray:
    """$\\lambda F$ for arbitrary source / observer layers (linear solve).

    Solves the interface conditions (continuity of $A$ and
    $\\partial A/\\partial z$ at $z = 0$ and every soil interface) per
    $\\lambda$ for the secondary field of a unit spectral primary
    $e^{-u_j|z-z'|}/u_j$ and returns $\\lambda$ times the secondary
    field at the observer: the reflected amplitude if both points are
    in one layer, the full transmitted amplitude otherwise. ``z_src``
    must lie in the soil (``> 0``); ``z_obs`` may be in the air
    (``< 0``). Used for cross-layer pairs and as an independent check
    of the closed forms.
    """
    n = earth.n_layers
    tops = (0.0, *earth.interface_depths)  # top depth of layer m = tops[m-1]
    j_src = earth.layer_of(z_src)
    k_obs = 0 if z_obs < 0.0 else earth.layer_of(z_obs)
    nl = lambdas.shape[0]
    size = 2 * n
    A = np.zeros((nl, size, size), dtype=complex)
    b = np.zeros((nl, size), dtype=complex)

    # Unknown indices: c -> 0; a_m -> 2m-1, b_m -> 2m (m = 1..n-1);
    # a_n -> 2n-1.
    def idx_a(m: int) -> int:
        return 2 * m - 1

    def idx_b(m: int) -> int:
        return 2 * m

    E = [None] + [
        np.exp(-u[m] * earth.thicknesses[m - 1]) if m < n else None
        for m in range(1, n + 1)
    ]

    for i in range(n):  # interface i at depth tops[i], layers i (up) / i+1 (low)
        rv, rd = 2 * i, 2 * i + 1
        # upper medium at its bottom
        if i == 0:
            A[:, rv, 0] += 1.0
            A[:, rd, 0] += u[0]
        else:
            m = i
            A[:, rv, idx_a(m)] += E[m]
            A[:, rv, idx_b(m)] += 1.0
            A[:, rd, idx_a(m)] += -u[m] * E[m]
            A[:, rd, idx_b(m)] += u[m]
        # lower medium at its top (subtract)
        m = i + 1
        if m < n:
            A[:, rv, idx_a(m)] -= 1.0
            A[:, rv, idx_b(m)] -= E[m]
            A[:, rd, idx_a(m)] -= -u[m]
            A[:, rd, idx_b(m)] -= u[m] * E[m]
        else:
            A[:, rv, idx_a(m)] -= 1.0
            A[:, rd, idx_a(m)] -= -u[m]
        # primary: rhs = P_low - P_up at the interface
        z_i = tops[i]
        if j_src == i + 1:  # source in the lower medium; interface above it
            e = np.exp(-u[j_src] * (z_src - z_i))
            b[:, rv] += e / u[j_src]
            b[:, rd] += e
        if j_src == i and i >= 1:  # source in the upper medium; interface below
            e = np.exp(-u[j_src] * (z_i - z_src))
            b[:, rv] -= e / u[j_src]
            b[:, rd] -= -e
    x = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    if k_obs == 0:
        field = x[:, 0] * np.exp(lambdas * z_obs)
    elif k_obs < n:
        top = tops[k_obs - 1]
        field = x[:, idx_a(k_obs)] * np.exp(-u[k_obs] * (z_obs - top)) + x[
            :, idx_b(k_obs)
        ] * np.exp(-u[k_obs] * (tops[k_obs] - z_obs))
    else:
        top = tops[n - 1]
        field = x[:, idx_a(n)] * np.exp(-u[n] * (z_obs - top))
    return lambdas * field


# ---------------------------------------------------------------------
# lambda quadrature
# ---------------------------------------------------------------------


@lru_cache(maxsize=512)
def _grid_from_params(
    lambda_break: float,
    lambda_max: float,
    rho_max: float,
    grading_top: float,
    n_log_nodes: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Nodes / weights of the two-regime grid (cached, read-only).

    Returns ``(lambdas, weights, resolved)``; ``resolved`` is ``False``
    when the hard node budget forced fewer than eight nodes per $J_0$
    period.
    """
    # Logarithmic region [eps, lambda_break]: Gauss–Legendre in ln(lambda).
    eps = max(lambda_max * 1e-12, 1e-15)
    nodes_log_x, weights_log_x = _leggauss(n_log_nodes)
    a_log = math.log(eps)
    b_log = math.log(lambda_break)
    t = 0.5 * (nodes_log_x + 1.0) * (b_log - a_log) + a_log
    lambdas_log = np.exp(t)
    weights_log = 0.5 * weights_log_x * (b_log - a_log) * lambdas_log

    # Linear region [lambda_break, lambda_max]: uniform panels resolving
    # J0 (two per period) merged with geometrically graded panels that
    # resolve the reflection-coefficient transitions.
    span = lambda_max - lambda_break
    resolved = True
    if rho_max > 0.0:
        n_target = int(math.ceil(span * rho_max * _PANELS_PER_PERIOD / (2.0 * math.pi)))
        if n_target > _SOFT_PANEL_BUDGET:
            n_floor = int(math.ceil(span * rho_max / (2.0 * math.pi)))
            n_uniform = max(_SOFT_PANEL_BUDGET, n_floor)
            if n_uniform > _HARD_PANEL_BUDGET:
                n_uniform = _HARD_PANEL_BUDGET
                resolved = False
        else:
            n_uniform = n_target
    else:
        n_uniform = 1
    n_uniform = max(n_uniform, _MIN_PANELS)
    edges = np.linspace(lambda_break, lambda_max, n_uniform + 1)
    if grading_top > lambda_break:
        graded = [lambda_break * 2.0**k for k in range(1, 200)]
        graded = [g for g in graded if g < min(grading_top, lambda_max)]
        if graded:
            edges = np.union1d(edges, np.asarray(graded))
            gaps = np.diff(edges)
            keep = np.concatenate([[True], gaps > 1e-9 * lambda_max])
            edges = edges[keep]
            edges[-1] = lambda_max
    nodes_p, weights_p = _leggauss(_PANEL_ORDER)
    half = 0.5 * np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    lambdas_lin = (centres[:, None] + half[:, None] * nodes_p[None, :]).ravel()
    weights_lin = (half[:, None] * weights_p[None, :]).ravel()

    lambdas = np.concatenate([lambdas_log, lambdas_lin])
    weights = np.concatenate([weights_log, weights_lin])
    lambdas.setflags(write=False)
    weights.setflags(write=False)
    return lambdas, weights, resolved


def _grid_params(
    *,
    z_sum: float,
    rho_max: float,
    p_ref: float,
    scales: Sequence[float] = (),
) -> tuple[float, float, float, float]:
    """``(lambda_break, lambda_max, rho_max, grading_top)`` of a pair grid.

    ``z_sum`` is the decay length of the spectral weight (m), ``rho_max``
    the largest horizontal distance whose $J_0$ must be resolved (m),
    ``p_ref`` the skin wavenumber $\\sqrt{\\omega\\mu_0\\sigma}$ of the
    medium around the conductors (1/m) and ``scales`` further physical
    wavenumbers (other layers' skin wavenumbers, 1/thickness).
    """
    z_eff = max(z_sum, 1e-3)
    lambda_max = 30.0 / z_eff
    lambda_break = max(p_ref, 1.0 / z_eff, 1e-9)
    lambda_break = min(lambda_break, 0.5 * lambda_max)
    rho_max = float(rho_max)
    if rho_max > 0.0:
        # F14 policy: keep the log region free of J0 oscillations.
        lambda_break = max(min(lambda_break, 0.25 * math.pi / rho_max), 1e-12)
    top_scale = max([p_ref, 1.0 / z_eff, *scales])
    grading_top = min(_GRADING_SPAN * top_scale, lambda_max)
    return lambda_break, lambda_max, rho_max, grading_top


def _build_lambda_grid(
    *,
    z_sum: float,
    rho_max: float,
    omega: float,
    sigma_top: float,
    n_log_nodes: int = _N_LOG_NODES,
    scales: Sequence[float] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Build the two-regime Sommerfeld quadrature grid (finding 10.36).

    Logarithmic region on $[\\epsilon, \\lambda_\\text{b}]$ with
    $\\lambda_\\text{b} = \\min(\\max(p_\\text{skin}, 1/z),
    \\lambda_\\text{max}/2, \\pi/(4\\rho_\\text{max}))$ — free of $J_0$
    oscillations at any distance — and 8-node Gauss panels of width
    $\\le \\pi/\\rho_\\text{max}$ (16 nodes per $J_0$ period), graded
    geometrically up to four times the largest physical scale, on
    $[\\lambda_\\text{b}, \\lambda_\\text{max} = 30/z]$.

    Parameters
    ----------
    z_sum
        Decay length of the spectral weight in m (e.g. $z + z'$).
    rho_max
        Largest horizontal distance in m whose $J_0(\\lambda\\rho)$ must
        be resolved.
    omega
        Angular frequency in rad/s.
    sigma_top
        Conductivity in S/m of the medium around the conductors.
    n_log_nodes
        Number of nodes of the logarithmic region.
    scales
        Further wavenumbers (1/m) to grade the panels for (other
        layers' skin wavenumbers, inverse layer thicknesses).

    Returns
    -------
    lambdas, weights : np.ndarray
        Combined Gauss nodes (1/m) and weights (read-only).

    Warns
    -----
    SommerfeldResolutionWarning
        If the node budget prevents resolving $J_0$ with at least eight
        nodes per period (instead of the historic silent 4096-panel
        cap).
    """
    p_ref = math.sqrt(omega * MU_0 * sigma_top) if omega > 0 and sigma_top > 0 else 0.0
    params = _grid_params(z_sum=z_sum, rho_max=rho_max, p_ref=p_ref, scales=scales)
    lambdas, weights, resolved = _grid_from_params(*params, n_log_nodes)
    if not resolved:
        _warn_unresolved(params)
    return lambdas, weights


def _warn_unresolved(params: tuple[float, float, float, float]) -> None:
    lambda_break, lambda_max, rho_max, _ = params
    warnings.warn(
        "sommerfeld_inductance: the lambda quadrature cannot resolve the J0 "
        f"oscillations of a segment pair {rho_max:.4g} m apart on "
        f"[{lambda_break:.3g}, {lambda_max:.3g}] 1/m within the node budget "
        f"({_HARD_PANEL_BUDGET} panels); the earth-return correction of "
        "this pair is NOT converged. Such pairs are several kilometres "
        "apart; split the study or check that their coupling is negligible.",
        SommerfeldResolutionWarning,
        stacklevel=4,
    )


# ---------------------------------------------------------------------
# Point kernels (public, correction beyond the additive mirror)
# ---------------------------------------------------------------------


def earth_return_correction_homogeneous(
    *,
    rho: float,
    z_i: float,
    z_j: float,
    omega: float,
    sigma_earth: float,
) -> complex:
    """σ-dependent earth-return correction beyond the additive mirror.

    Point-kernel form of the **buried–buried** Pollaczek correction
    (see the module docstring):

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
      → free space $1/R$.
    - $\\sigma_e \\to \\infty$: $e^{-\\gamma R} \\to 0$,
      $\\lambda/u_1 \\to 0$ — correction $\\to -1/R - 1/R'$, total
      → 0 (buried conductor screened by the surrounding medium).

    Parameters
    ----------
    rho
        Horizontal distance between source and field point in m.
        Must be ≥ 0.
    z_i, z_j
        Depths (positive into soil) in m. Must be > 0 for the
        integral to converge.
    omega
        Angular frequency in rad/s.
    sigma_earth
        Earth conductivity in S/m.

    Returns
    -------
    correction : complex
        The kernel above in 1/m (the pair integrals multiply by
        $j\\omega\\mu_0/(4\\pi)$ and integrate over both segments).
    """
    if omega <= 0.0 or sigma_earth <= 0.0:
        return 0.0 + 0.0j
    earth = LayeredEarth(rhos=(1.0 / sigma_earth,), thicknesses=())
    return earth_return_correction_layered(
        rho=rho, z_i=z_i, z_j=z_j, omega=omega, earth=earth
    )


def earth_return_correction_layered(
    *,
    rho: float,
    z_i: float,
    z_j: float,
    omega: float,
    earth: LayeredEarth,
) -> complex:
    """Layered-earth analogue of :func:`earth_return_correction_homogeneous`.

    Buried–buried point kernel beyond the additive mirror,
    $\\Delta G = G - 1/R - 1/R'$, with the exact layered Green's
    function of the module docstring: the generalised-reflection
    closed form for points in the same layer, the transmitted amplitude
    otherwise. Review pass 10 (finding 10.46, pass-9 F50): the
    historic implementation used the single air-image form with the
    stack's surface reflection coefficient, whose reflected term has
    the **opposite sign** of the exact two-layer kernel (e.g.
    +8.7e-3 instead of −6.3e-3 at $\\rho = 0$ for earth (100, 10)
    Ω·m, $h_1$ = 3 m, 1 kHz, $z = z' = 0.6$ m).

    Parameters
    ----------
    rho
        Horizontal distance in m (≥ 0).
    z_i, z_j
        Depths of observer and source in m (> 0, in the soil).
    omega
        Angular frequency in rad/s.
    earth
        Layered-earth configuration.

    Returns
    -------
    correction : complex
        Kernel value in 1/m.
    """
    if omega <= 0.0:
        return 0.0 + 0.0j
    if z_i <= 0.0 or z_j <= 0.0:
        raise ValueError(
            f"earth_return_correction_layered needs buried points (z > 0); "
            f"got z_i={z_i}, z_j={z_j}"
        )
    rho = float(rho)
    R_dir = math.sqrt(rho * rho + (z_i - z_j) ** 2)
    R_mir = math.sqrt(rho * rho + (z_i + z_j) ** 2)
    k_i = earth.layer_of(z_i)
    k_j = earth.layer_of(z_j)
    plan = _SpectralPlan.buried(earth, z_i, z_j, k_i, k_j)
    lambdas, weights = _build_lambda_grid(
        z_sum=plan.decay,
        rho_max=max(rho, 1e-6),
        omega=omega,
        sigma_top=1.0 / earth.rhos[plan.ref_layer - 1],
        scales=plan.scales(omega, earth),
    )
    spec = plan.weight(lambdas, omega, earth)
    reflected = complex(np.sum(weights * spec * j0(lambdas * rho)))
    if k_i == k_j:
        gamma = complex(np.sqrt(1j * omega * MU_0 / earth.rhos[k_i - 1]))
        if R_dir < 1e-9:
            direct = -gamma
        else:
            direct = (np.exp(-gamma * R_dir) - 1.0) / R_dir
    else:
        direct = 0.0
    return complex(direct) - 1.0 / max(R_mir, 1e-9) + reflected


# ---------------------------------------------------------------------
# Spectral plans: which weight a (sub)segment pair needs
# ---------------------------------------------------------------------


class _SpectralPlan:
    """Spectral part of the Green's function for one (sub)segment pair.

    ``kind`` is ``"same"`` (both buried in layer ``ref_layer``),
    ``"air"`` (both overhead), ``"mixed1"`` (one in layer 1, one
    overhead) or ``"generic"`` (any other combination, linear solve).
    ``decay`` is the decay length of the weight in m (sets
    $\\lambda_\\text{max}$), ``direct`` whether the closed-form direct
    term $(e^{-\\gamma R}-1)/R$ applies and ``subtract_free`` whether the
    free-space $e^{-\\lambda|\\Delta z|}$ is subtracted inside the
    integral (pairs in different media).
    """

    __slots__ = (
        "kind",
        "ref_layer",
        "decay",
        "direct",
        "subtract_free",
        "h_a",
        "h_b",
        "z_a",
        "z_b",
    )

    def __init__(self, kind, ref_layer, decay, direct, subtract_free, **depths):
        self.kind = kind
        self.ref_layer = ref_layer
        self.decay = decay
        self.direct = direct
        self.subtract_free = subtract_free
        self.h_a = depths.get("h_a", 0.0)
        self.h_b = depths.get("h_b", 0.0)
        self.z_a = depths.get("z_a", 0.0)
        self.z_b = depths.get("z_b", 0.0)

    @classmethod
    def buried(cls, earth, z_a, z_b, k_a, k_b) -> _SpectralPlan:
        if k_a == k_b:
            tops = (0.0, *earth.interface_depths)
            top = tops[k_a - 1]
            h_a, h_b = z_a - top, z_b - top
            decay = h_a + h_b
            if k_a < earth.n_layers:
                t = earth.thicknesses[k_a - 1]
                decay = min(decay, 2.0 * t - h_a - h_b)
            return cls("same", k_a, decay, True, False, h_a=h_a, h_b=h_b)
        decay = max(abs(z_a - z_b), 1e-3)
        return cls("generic", k_b, decay, False, True, z_a=z_a, z_b=z_b)

    @classmethod
    def for_pair(cls, earth, z_a: float, z_b: float) -> _SpectralPlan:
        """Plan for observer depth ``z_a`` and source depth ``z_b`` (m)."""
        buried_a = z_a >= 0.0
        buried_b = z_b >= 0.0
        if buried_a and buried_b:
            return cls.buried(
                earth,
                max(z_a, 0.0),
                max(z_b, 0.0),
                earth.layer_of(z_a),
                earth.layer_of(z_b),
            )
        if not buried_a and not buried_b:
            return cls("air", 1, -(z_a + z_b), False, False, h_a=-z_a, h_b=-z_b)
        z_soil, z_air = (z_a, z_b) if buried_a else (z_b, z_a)
        if earth.layer_of(z_soil) == 1:
            return cls("mixed1", 1, z_soil - z_air, False, True, h_a=z_soil, h_b=-z_air)
        # Reciprocity: the generic solve needs the source in the soil.
        return cls(
            "generic",
            earth.layer_of(z_soil),
            max(z_soil - z_air, 1e-3),
            False,
            True,
            z_a=z_air,
            z_b=z_soil,
        )

    def scales(self, omega: float, earth: LayeredEarth) -> tuple[float, ...]:
        """Physical wavenumbers (1/m) the panels are graded for."""
        out = [math.sqrt(omega * MU_0 / r) for r in earth.rhos]
        out += [1.0 / t for t in earth.thicknesses]
        return tuple(out)

    def weight(
        self, lambdas: np.ndarray, omega: float, earth: LayeredEarth
    ) -> np.ndarray:
        """Spectral weight $w(\\lambda)$ with $\\Delta G \\ni \\int w J_0$."""
        if self.kind == "air":
            gamma = reflection_coefficient_layered(lambdas, omega=omega, earth=earth)
            return -gamma * np.exp(-lambdas * (self.h_a + self.h_b))
        u = _vertical_wavenumbers(lambdas, omega, earth)
        if self.kind == "same":
            return _same_layer_weight(
                lambdas, u, earth, self.ref_layer, self.h_a, self.h_b
            )
        if self.kind == "mixed1":
            w = _mixed_layer1_weight(lambdas, u, earth, self.h_a, self.h_b)
            return w - np.exp(-lambdas * (self.h_a + self.h_b))
        w = _generic_weight(lambdas, u, earth, self.z_a, self.z_b)
        return w - np.exp(-lambdas * abs(self.z_a - self.z_b))


# ---------------------------------------------------------------------
# Segment-pair integration via 16x16 Gauss–Legendre outer
# ---------------------------------------------------------------------


def _rho_interp_setup(
    rho_grid: np.ndarray, z_ref: float
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Interpolation nodes of $S(\\rho)$ for one pair.

    Returns ``(t_nodes, t_grid, rho_nodes)``; ``t_nodes`` is ``None``
    when all outer nodes share one distance (single evaluation).
    """
    rho_min = float(rho_grid.min())
    rho_max = float(rho_grid.max())
    if rho_max - rho_min < 1e-9:
        return None, None, np.array([rho_min])
    z_off = max(z_ref, 1e-3)
    t_lo = math.log(rho_min + z_off)
    t_hi = math.log(rho_max + z_off)
    t_nodes = np.linspace(t_lo, t_hi, _N_RHO_NODES)
    rho_nodes = np.maximum(np.exp(t_nodes) - z_off, 0.0)
    t_grid = np.log(rho_grid + z_off)
    return t_nodes, t_grid, rho_nodes


def _interp_spectral(
    s_nodes: np.ndarray,
    t_nodes: np.ndarray | None,
    t_grid: np.ndarray | None,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Linear interpolation of $S$ in $t = \\ln(\\rho + z)$ to the grid."""
    if t_nodes is None:
        return np.full(shape, complex(s_nodes[0]), dtype=complex)
    s_re = np.interp(t_grid.ravel(), t_nodes, s_nodes.real)
    s_im = np.interp(t_grid.ravel(), t_nodes, s_nodes.imag)
    return (s_re + 1j * s_im).reshape(shape)


def _split_points(earth: LayeredEarth | None) -> tuple[float, ...]:
    """Planes (depths in m) at which segments are split: surface + interfaces."""
    if earth is None:
        return (0.0,)
    return (0.0, *earth.interface_depths)


def _split_segment(
    p1: np.ndarray, p2: np.ndarray, planes: Sequence[float]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a segment at the horizontal planes ``z = plane`` it crosses.

    Returns the sub-segments in order; the cut points lie exactly on
    their planes. A segment that only touches a plane is not split.
    """
    z1, z2 = float(p1[2]), float(p2[2])
    length = float(np.linalg.norm(p2 - p1))
    tol = 1e-9 * max(length, 1e-12)
    cuts = sorted(
        ((zc - z1) / (z2 - z1), zc)
        for zc in planes
        if min(z1, z2) < zc - tol and max(z1, z2) > zc + tol
    )
    if not cuts:
        return [(p1, p2)]
    pts = [p1]
    for s, zc in cuts:
        q = p1 + s * (p2 - p1)
        q[2] = zc
        pts.append(q)
    pts.append(p2)
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _subpair_values(
    pa1: np.ndarray,
    pa2: np.ndarray,
    pb1: np.ndarray,
    pb2: np.ndarray,
    dot: float,
    omegas: Sequence[float],
    earth: LayeredEarth,
) -> np.ndarray:
    """ΔZ (image ``"none"``) of one sub-segment pair for every frequency.

    Both sub-segments lie in one medium each (split at the surface and
    the interfaces). Returns complex values in Ω, shape ``(n_f,)``.
    """
    n_f = len(omegas)
    out = np.zeros(n_f, dtype=complex)
    da = pa2 - pa1
    db = pb2 - pb1
    la = float(np.linalg.norm(da))
    lb = float(np.linalg.norm(db))
    if la <= 0.0 or lb <= 0.0:
        return out
    s_nodes = 0.5 * (_OUTER_GL_NODES + 1.0)
    w_nodes = 0.5 * _OUTER_GL_WEIGHTS
    pts_a = pa1[None, :] + s_nodes[:, None] * da[None, :]
    pts_b = pb1[None, :] + s_nodes[:, None] * db[None, :]
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    rho_grid = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
    R3 = np.sqrt(rho_grid**2 + diff[:, :, 2] ** 2)
    small = R3 < 1e-9
    R3_safe = np.where(small, 1.0, R3)
    outer_w = w_nodes[:, None] * w_nodes[None, :]

    z_a = 0.5 * float(pa1[2] + pa2[2])
    z_b = 0.5 * float(pb1[2] + pb2[2])
    plan = _SpectralPlan.for_pair(earth, z_a, z_b)
    t_nodes, t_grid, rho_nodes = _rho_interp_setup(rho_grid, plan.decay)
    rho_max = float(rho_grid.max() + 1e-9)
    sigma_ref = 1.0 / earth.rhos[plan.ref_layer - 1]

    table_key = None
    table = None
    for k, omega in enumerate(omegas):
        if omega <= 0.0:
            continue
        p_ref = math.sqrt(omega * MU_0 * sigma_ref)
        params = _grid_params(
            z_sum=plan.decay,
            rho_max=rho_max,
            p_ref=p_ref,
            scales=plan.scales(omega, earth),
        )
        lambdas, lam_w, resolved = _grid_from_params(*params, _N_LOG_NODES)
        if not resolved:
            _warn_unresolved(params)
        if params != table_key:
            # Bessel table J0(lambda * rho_node) — the expensive part,
            # shared by every frequency with the same grid.
            table = j0(lambdas[None, :] * rho_nodes[:, None])
            table_key = params
        spec = plan.weight(lambdas, omega, earth) * lam_w
        s_vals = table @ spec
        inner = _interp_spectral(s_vals, t_nodes, t_grid, rho_grid.shape)
        if plan.direct:
            gamma = np.sqrt(1j * omega * MU_0 * sigma_ref)
            inner = inner + np.where(
                small, -gamma, (np.exp(-gamma * R3_safe) - 1.0) / R3_safe
            )
        geom = complex(np.sum(outer_w * inner)) * la * lb * dot
        out[k] = 1j * omega * MU_0 / (4.0 * math.pi) * geom
    return out


def _pair_values_none(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    omegas: Sequence[float],
    earth: LayeredEarth,
) -> np.ndarray:
    """ΔZ of a segment pair relative to free space (image ``"none"``)."""
    p1_a = np.asarray(p1_a, dtype=float)
    p2_a = np.asarray(p2_a, dtype=float)
    p1_b = np.asarray(p1_b, dtype=float)
    p2_b = np.asarray(p2_b, dtype=float)
    out = np.zeros(len(omegas), dtype=complex)
    da = p2_a - p1_a
    db = p2_b - p1_b
    la = float(np.linalg.norm(da))
    lb = float(np.linalg.norm(db))
    if la <= 0.0 or lb <= 0.0:
        return out
    dot = float((da / la) @ (db / lb))
    if abs(dot) < 1e-12:
        return out  # orthogonal segments: the A_x-type kernel vanishes
    planes = _split_points(earth)
    subs_a = _split_segment(p1_a, p2_a, planes)
    subs_b = _split_segment(p1_b, p2_b, planes)
    for sa1, sa2 in subs_a:
        for sb1, sb2 in subs_b:
            out += _subpair_values(sa1, sa2, sb1, sb2, dot, omegas, earth)
    return out


def _convert_image(
    values: np.ndarray, omegas: Sequence[float], m_img: float, mode: str
) -> np.ndarray:
    """Move a ``"none"`` correction to another image convention."""
    if mode == "none":
        return values
    sign = -1.0 if mode == "additive" else 1.0
    om = np.asarray(omegas, dtype=float)
    return values + np.where(om > 0.0, sign * 1j * om * m_img, 0.0)


def _pair_image_mutual(p1_a, p2_a, p1_b, p2_b) -> float:
    """Neumann mutual of segment a and the mirrored segment b (H)."""
    p1_b = np.asarray(p1_b, dtype=float)
    p2_b = np.asarray(p2_b, dtype=float)
    seg = np.array([[p1_a, p2_a], [p1_b, p2_b]], dtype=float)
    _check_image_geometry(
        seg[:, 0], seg[:, 1], np.zeros(2), where="earth-return correction"
    )
    return neumann_mutual(
        np.asarray(p1_a, dtype=float),
        np.asarray(p2_a, dtype=float),
        _mirror(p1_b),
        _mirror(p2_b),
    )


def sommerfeld_pair_integral_homogeneous(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    *,
    omega: float,
    sigma_earth: float,
    image: ImageConvention = "additive",
) -> complex:
    """Integrate the σ-dependent magnetic Green function over a segment pair.

    Computes

    $$
    \\Delta Z^{(i,j)} \\;=\\; \\frac{j\\omega\\mu_0}{4\\pi}
    \\int_{C_i}\\!\\!\\int_{C_j}
    (\\hat{l}_i\\cdot\\hat{l}_j)\\,
    \\Delta G(\\vec{r}_i, \\vec{r}_j;\\,\\omega,\\sigma_e)\\,
    dl_i\\,dl_j,
    $$

    the earth-return correction of a homogeneous earth relative to the
    image convention ``image`` of the inductance matrix it is added to
    (see the module docstring). Used by
    :func:`build_sommerfeld_correction_matrix`.

    Parameters
    ----------
    p1_a, p2_a, p1_b, p2_b
        End points of the two segments in m ($z > 0$ in the soil).
    omega
        Angular frequency in rad/s.
    sigma_earth
        Earth conductivity in S/m.
    image
        ``"additive"`` (default, the historic ADR-0004 baseline),
        ``"none"`` or ``"antiparallel"``.

    Returns
    -------
    Z : complex
        Per-pair earth-return correction in $\\Omega$.
    """
    if omega <= 0.0 or sigma_earth <= 0.0:
        return 0.0 + 0.0j
    earth = LayeredEarth(rhos=(1.0 / sigma_earth,), thicknesses=())
    return sommerfeld_pair_integral_layered(
        p1_a, p2_a, p1_b, p2_b, omega=omega, earth=earth, image=image
    )


def sommerfeld_pair_integral_layered(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    *,
    omega: float,
    earth: LayeredEarth,
    image: ImageConvention = "additive",
) -> complex:
    """Layered-earth analogue of :func:`sommerfeld_pair_integral_homogeneous`.

    Exact layered kernel for any number of layers (generalised
    reflection coefficients; transmitted amplitudes for pairs in
    different media). Segments that cross the soil surface or a layer
    interface are split there.

    Parameters
    ----------
    p1_a, p2_a, p1_b, p2_b
        End points of the two segments in m.
    omega
        Angular frequency in rad/s.
    earth
        Layered-earth configuration.
    image
        Image convention of the inductance matrix the correction is
        added to (``"additive"`` default, ``"none"``,
        ``"antiparallel"``). With an image, a segment that crosses the
        soil surface raises ``ValueError`` (its image is singular).

    Returns
    -------
    Z : complex
        Per-pair earth-return correction in $\\Omega$.
    """
    mode = _resolve_image(image, None)
    if omega <= 0.0:
        return 0.0 + 0.0j
    m_img = 0.0
    if mode != "none":
        m_img = _pair_image_mutual(p1_a, p2_a, p1_b, p2_b)
    vals = _pair_values_none(p1_a, p2_a, p1_b, p2_b, [omega], earth)
    return complex(_convert_image(vals, [omega], m_img, mode)[0])


# ---------------------------------------------------------------------
# Full M×M correction matrix
# ---------------------------------------------------------------------


def build_sommerfeld_correction_matrices(
    seg_endpoints: np.ndarray,  # shape (M, 2, 3)
    wire_radii: np.ndarray | None = None,
    *,
    omegas: Sequence[float],
    earth: LayeredEarth,
    image: ImageConvention = "additive",
) -> np.ndarray:
    """Sommerfeld earth-return correction matrices for several frequencies.

    Multi-frequency form of :func:`build_sommerfeld_correction_matrix`
    (finding 10.60): per segment pair the outer geometry, the
    $\\lambda$ grid and the Bessel table $J_0(\\lambda\\rho)$ — about
    three quarters of the historic per-frequency cost — are built once
    and reused for every frequency with the same grid. Entry ``k``
    equals ``build_sommerfeld_correction_matrix(..., omega=omegas[k])``
    bit for bit.

    Parameters
    ----------
    seg_endpoints
        Array of shape ``(M, 2, 3)``: start and end point (m) of every
        inductive branch segment.
    wire_radii
        Per-branch wire radii in m (unused: the radius enters through
        the self terms of the inductance matrix). Kept for API parity.
    omegas
        Angular frequencies in rad/s; entries ``<= 0`` give zeros.
    earth
        Layered-earth configuration.
    image
        Image convention of the inductance matrix the corrections are
        added to: ``"additive"`` (default — what the solver assembles
        today), ``"none"`` (exact baseline, cheapest) or
        ``"antiparallel"``.

    Returns
    -------
    dZ : np.ndarray, shape (n_f, M, M), dtype complex
        Symmetric correction matrices in Ω.
    """
    mode = _resolve_image(image, None)
    omegas = [float(om) for om in np.atleast_1d(np.asarray(omegas, dtype=float))]
    seg_endpoints = np.asarray(seg_endpoints, dtype=float)
    M = seg_endpoints.shape[0]
    out = np.zeros((len(omegas), M, M), dtype=complex)
    if M == 0 or all(om <= 0.0 for om in omegas):
        return out
    L_img = _image_inductance_matrix(seg_endpoints) if mode != "none" else None
    for i in range(M):
        p1_i = seg_endpoints[i, 0]
        p2_i = seg_endpoints[i, 1]
        for j in range(i, M):
            vals = _pair_values_none(
                p1_i, p2_i, seg_endpoints[j, 0], seg_endpoints[j, 1], omegas, earth
            )
            out[:, i, j] = vals
            out[:, j, i] = vals
    if L_img is not None:
        sign = -1.0 if mode == "additive" else 1.0
        for k, om in enumerate(omegas):
            if om > 0.0:
                out[k] += sign * 1j * om * L_img
    return out


def build_sommerfeld_correction_matrix(
    seg_endpoints: np.ndarray,  # shape (M, 2, 3)
    wire_radii: np.ndarray | None = None,
    *,
    omega: float,
    earth: LayeredEarth,
    image: ImageConvention = "additive",
) -> np.ndarray:
    """Assemble the dense Sommerfeld earth-return correction matrix.

    The output is the σ-dependent addition to the Neumann inductance
    matrix of
    :func:`groundfield.coupling.inductance.build_inductance_matrix`
    built with the same image convention:

    .. code-block:: python

        Z_b = jω · L(image=image) + dZ_Sommerfeld(image=image)

    For ``earth.n_layers == 1`` this is the homogeneous-earth
    Pollaczek kernel; for ``n >= 2`` the exact layered kernel is used.
    Several frequencies at once: :func:`build_sommerfeld_correction_matrices`.

    Parameters
    ----------
    seg_endpoints
        Array of shape ``(M, 2, 3)`` with the start- and end-points
        (m) of every distributed-conductor longitudinal-branch segment.
    wire_radii
        Per-branch wire radii in m (unused, kept for API parity).
    omega
        Angular frequency in rad/s.
    earth
        Layered-earth configuration.
    image
        ``"additive"`` (default), ``"none"`` or ``"antiparallel"`` —
        see the module docstring.

    Returns
    -------
    dZ : np.ndarray, shape (M, M), dtype complex
        Symmetric Sommerfeld correction matrix in $\\Omega$.
    """
    if omega <= 0.0:
        return np.zeros(
            (np.asarray(seg_endpoints).shape[0],) * 2,
            dtype=complex,
        )
    return build_sommerfeld_correction_matrices(
        seg_endpoints, wire_radii, omegas=[omega], earth=earth, image=image
    )[0]
