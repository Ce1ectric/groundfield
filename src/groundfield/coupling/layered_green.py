"""Cross-layer scalar Green's function for layered soil (ADR-0007).

This module implements the electric (scalar-potential) Green's
function for a 2-layer soil where source and observer can be in
**either** layer. It complements the existing image-series solver in
:mod:`groundfield.solver.image_2layer` (which assumes both points in
the upper layer) and the Sommerfeld kernel in
:mod:`groundfield.solver.mom_sommerfeld` (same restriction).

Mathematical background
-----------------------
For a 2-layer soil with upper-layer resistivity $\\rho_1$ and
thickness $h_1$, lower-layer resistivity $\\rho_2$ semi-infinite,
the scalar Green's function for a unit point source at depth $z_s$
observed at depth $z$ takes the Hankel-transform form

$$
\\varphi(s, z, z_s) \\;=\\; \\int_0^{\\infty}\\!
\\Phi(\\lambda, z, z_s)\\,J_0(\\lambda s)\\,\\lambda\\,d\\lambda,
$$

where the spectral kernel $\\Phi(\\lambda, z, z_s)$ is determined by
the boundary-value problem

- $\\partial\\Phi/\\partial z = 0$ at $z = 0$ (free surface),
- $\\Phi$ continuous at $z = h_1$,
- $\\sigma_k\\,\\partial\\Phi/\\partial z$ continuous at $z = h_1$.

In each layer $\\Phi$ is a sum of $e^{\\pm\\lambda z}$ plus a
particular solution $e^{-\\lambda|z-z_s|}/(2\\lambda\\sigma_k)$ in
the layer that contains the source. The four layer-pair cases —
$\\Phi_{uu}, \\Phi_{ul}, \\Phi_{lu}, \\Phi_{ll}$ — correspond to
the four (source-layer, observer-layer) combinations and are
related by reciprocity ($\\Phi_{ul} = \\Phi_{lu}$).

Implementation
--------------
The 4×4 boundary-value system is solved numerically once per
$(\\lambda, z_s)$ pair (4 unknowns: amplitudes in each layer for
each direction). Per call the cost is dominated by $J_0$
evaluation and the linear solve at each $\\lambda$ node — fully
vectorisable with `numpy`.

The implementation focuses on the 2-layer case
(:class:`groundfield.soil.models.TwoLayerSoil`); the n-layer
extension follows the same recursion and is deferred to ADR-0007
Phase C.

Limit checks (in the test suite)
--------------------------------
- $\\rho_2 = \\rho_1$: all four pair kernels reduce to the
  homogeneous Green's function $\\rho/(4\\pi r) + \\rho/(4\\pi r')$.
- $\\rho_2 \\to \\infty$ (PEC bottom): for source in upper layer,
  $G_{ul} \\to 0$ (no penetration into the lower layer);
  $G_{uu} \\to G_{uu}^{K \\to 1}$ — image structure with
  $K \\to 1$.
- $\\rho_2 \\to 0$ (sink): $G_{uu} \\to G_{uu}^{K \\to -1}$,
  spreading resistance drops dramatically.
- Source/observer in upper layer: bit-exact match to the
  Tagg/Sunde image series in
  :func:`groundfield.solver.image_2layer.solve_image_2layer`.

References
----------
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover, Ch. 3.
- Tagg, G. F. (1964). *Earth Resistances*, Newnes.
- Wait, J. R. (1972). *Electromagnetic Waves in Stratified Media*,
  Pergamon.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import j0

__all__ = [
    "two_layer_spectral_kernel",
    "two_layer_real_space_kernel",
    "two_layer_layered_correction_real_space",
]


def _spectral_amplitudes(
    lambdas: np.ndarray,
    z_s: float,
    rho_1: float,
    rho_2: float,
    h_1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 2-layer matching for amplitudes at every $\\lambda$.

    Returns ``(a_1, b_1, a_2, b_2)`` such that the spectral kernel
    in each layer is

    - layer 1 ($0 \\le z \\le h_1$):
      $\\Phi_1(\\lambda, z) = a_1 e^{\\lambda z} + b_1 e^{-\\lambda z}
      + \\delta_{s\\in 1}\\,\\rho_1/(2\\lambda)\\,e^{-\\lambda|z-z_s|}$,
    - layer 2 ($z > h_1$):
      $\\Phi_2(\\lambda, z) = a_2 e^{\\lambda z} + b_2 e^{-\\lambda z}
      + \\delta_{s\\in 2}\\,\\rho_2/(2\\lambda)\\,e^{-\\lambda|z-z_s|}$.

    Numerical stability
    -------------------
    Direct use of $a_1 e^{\\lambda z}$ overflows for moderate
    $\\lambda h_1$ ($e^{200} \\approx 10^{86}$ already destroys
    the matrix conditioning). We therefore solve internally for
    the **stable amplitudes** $(A, B, C)$ via the change of
    variables

    .. math::

       a_1 = A\\,e^{-\\lambda h_1}, \\quad b_1 = B, \\quad
       b_2 = C\\,e^{-\\lambda h_1}.

    All coefficients in the resulting matrix are bounded by 1 in
    magnitude for any $\\lambda > 0$, so the linear solve is
    well-conditioned. The returned $(a_1, b_1, a_2, b_2)$ are then
    the unstable representation; callers that consume them must
    take care to use them only inside expressions where the
    growing exponential is multiplied by a decaying one (the
    spectral kernel evaluators in this module do exactly that).
    """
    # Source-in-which-layer flags
    source_in_upper = z_s <= h_1

    # Pre-compute the only exponential we need: e^{-λh_1} (bounded by 1).
    e_mlh = np.exp(-lambdas * h_1)

    # Source contribution coefficients at z=0 and z=h_1.
    if source_in_upper:
        # z_s in (0, h_1)
        # particular solution in layer 1: rho_1/(2λ) * exp(-λ|z - z_s|)
        # at z=0: ∂_z Φ_p|_{0} = +ρ_1/2 · exp(-λ z_s)
        dz_phi_p_0 = (rho_1 / 2.0) * np.exp(-lambdas * z_s)
        # at z=h_1: Φ_p(h_1) = ρ_1/(2λ) · exp(-λ(h_1 - z_s))
        phi_p_h = (rho_1 / (2.0 * lambdas)) * np.exp(-lambdas * (h_1 - z_s))
        # ∂_z Φ_p|_{h_1} = -ρ_1/2 · exp(-λ(h_1 - z_s))
        dz_phi_p_h = -(rho_1 / 2.0) * np.exp(-lambdas * (h_1 - z_s))
        phi_p_layer2_h = np.zeros_like(lambdas)
        dz_phi_p_layer2_h = np.zeros_like(lambdas)
    else:
        # z_s > h_1, source in layer 2
        dz_phi_p_0 = np.zeros_like(lambdas)
        phi_p_h = np.zeros_like(lambdas)
        dz_phi_p_h = np.zeros_like(lambdas)
        # at z=h_1: Φ_p(h_1) = ρ_2/(2λ) · exp(-λ(z_s - h_1))
        phi_p_layer2_h = (rho_2 / (2.0 * lambdas)) * np.exp(-lambdas * (z_s - h_1))
        # ∂_z Φ_p|_{h_1} = +ρ_2/2 · exp(-λ(z_s - h_1)) (since z_s > h_1)
        dz_phi_p_layer2_h = (rho_2 / 2.0) * np.exp(-lambdas * (z_s - h_1))

    # Solve the *stable* matrix system in (A, B, C) defined via
    #    a_1 = A·e^{-λh_1},   b_1 = B,   b_2 = C·e^{+λh_1}.
    #
    # The choice ``b_2 = C · e^{+λh_1}`` is essential for stability
    # in the lower layer: with it, the spectral kernel reads
    # ``Phi_2(z) = b_2·e^{-λz} = C·e^{-λ(z-h_1)}``, which is bounded
    # by ``|C|`` for any ``z ≥ h_1`` and any ``λ``. All exponentials
    # appearing in the matrix coefficients below are ``e^{-λh_1}``
    # (≤ 1) — no overflow even at the deepest λ in the quadrature.
    #
    # The original equations transform as follows:
    #
    # Eq 1 (∂_z Φ at z=0 = 0): λ a_1 - λ b_1 = -dz_phi_p_0
    #    ⇒ λ A·e^{-λh_1} - λ B = -dz_phi_p_0
    #
    # Eq 2 (continuity at z=h_1):
    #    a_1·e^{λh_1} + b_1·e^{-λh_1} + phi_p_h
    #      = b_2·e^{-λh_1} + phi_p_layer2_h
    #    ⇒ A + B·e^{-λh_1} - C = phi_p_layer2_h - phi_p_h
    #      (since a_1·e^{λh_1} = A and b_2·e^{-λh_1} = C·e^{λh_1}·e^{-λh_1} = C)
    #
    # Eq 3 (σ-weighted ∂_z continuity at z=h_1):
    #    σ_1·[λ a_1·e^{λh_1} - λ b_1·e^{-λh_1} + dz_phi_p_h]
    #      = σ_2·[-λ b_2·e^{-λh_1} + dz_phi_p_layer2_h]
    #    ⇒ σ_1·λ·A - σ_1·λ·B·e^{-λh_1} + σ_2·λ·C
    #      = σ_2·dz_phi_p_layer2_h - σ_1·dz_phi_p_h
    sigma_1 = 1.0 / rho_1
    sigma_2 = 1.0 / rho_2

    n = lambdas.size
    A_mat = np.zeros((n, 3, 3), dtype=float)
    rhs = np.zeros((n, 3), dtype=float)

    # Eq 1
    A_mat[:, 0, 0] = lambdas * e_mlh
    A_mat[:, 0, 1] = -lambdas
    A_mat[:, 0, 2] = 0.0
    rhs[:, 0] = -dz_phi_p_0

    # Eq 2
    A_mat[:, 1, 0] = 1.0
    A_mat[:, 1, 1] = e_mlh
    A_mat[:, 1, 2] = -1.0
    rhs[:, 1] = phi_p_layer2_h - phi_p_h

    # Eq 3
    A_mat[:, 2, 0] = sigma_1 * lambdas
    A_mat[:, 2, 1] = -sigma_1 * lambdas * e_mlh
    A_mat[:, 2, 2] = sigma_2 * lambdas
    rhs[:, 2] = sigma_2 * dz_phi_p_layer2_h - sigma_1 * dz_phi_p_h

    sol = np.linalg.solve(A_mat, rhs[..., None]).squeeze(-1)
    A = sol[:, 0]
    B = sol[:, 1]
    C = sol[:, 2]
    # Return the **stable** amplitudes (A, B, C). Callers should
    # use _phi_from_stable_amplitudes() to evaluate Phi(z) at any
    # observer depth without intermediate overflows.
    zeros = np.zeros_like(C)
    return A, B, zeros, C


def _phi_from_stable_amplitudes(
    lambdas: np.ndarray,
    A: np.ndarray, B: np.ndarray, C: np.ndarray,
    z: float, z_s: float,
    rho_1: float, rho_2: float, h_1: float,
) -> np.ndarray:
    """Evaluate $\\Phi(\\lambda, z, z_s)$ from the stable amplitudes.

    Builds the spectral kernel using **only** decaying exponentials
    so the result is bounded for any $\\lambda \\cdot h_1$.

    Decompositions:

    - layer 1 ($0 \\le z \\le h_1$):
      $\\Phi_1(\\lambda, z) = A\\,e^{-\\lambda(h_1-z)}
      + B\\,e^{-\\lambda z}
      + \\delta_{s\\in 1}\\,\\frac{\\rho_1}{2\\lambda}e^{-\\lambda|z-z_s|}$.
    - layer 2 ($z > h_1$):
      $\\Phi_2(\\lambda, z) = C\\,e^{-\\lambda(z-h_1)}
      + \\delta_{s\\in 2}\\,\\frac{\\rho_2}{2\\lambda}e^{-\\lambda|z-z_s|}$.
    """
    if z <= h_1:
        Phi = (
            A * np.exp(-lambdas * (h_1 - z))
            + B * np.exp(-lambdas * z)
        )
        if z_s <= h_1:
            Phi = Phi + (rho_1 / (2.0 * lambdas)) * np.exp(
                -lambdas * abs(z - z_s)
            )
    else:
        Phi = C * np.exp(-lambdas * (z - h_1))
        if z_s > h_1:
            Phi = Phi + (rho_2 / (2.0 * lambdas)) * np.exp(
                -lambdas * abs(z - z_s)
            )
    return Phi


def two_layer_spectral_kernel(
    lambdas: np.ndarray,
    z: float,
    z_s: float,
    *,
    rho_1: float,
    rho_2: float,
    h_1: float,
) -> np.ndarray:
    """Spectral Green's function $\\Phi(\\lambda, z, z_s)$.

    Returns the value of the spectral kernel at every $\\lambda$
    in ``lambdas`` for given source depth $z_s$ and observer depth
    $z$. Both depths can be in either layer.

    Parameters
    ----------
    lambdas
        1-D array of spectral variable values, $\\lambda > 0$ in
        units of m⁻¹.
    z, z_s
        Observer / source depth in metres ($z \\ge 0$, positive into
        the soil).
    rho_1, rho_2
        Resistivities of upper / lower layer in $\\Omega\\,\\mathrm{m}$.
    h_1
        Upper-layer thickness in m.

    Returns
    -------
    Phi : np.ndarray, shape (n_lambda,)
    """
    A, B, _, C = _spectral_amplitudes(
        lambdas, z_s, rho_1, rho_2, h_1,
    )
    return _phi_from_stable_amplitudes(
        lambdas, A, B, C, z, z_s, rho_1, rho_2, h_1,
    )


def two_layer_real_space_kernel(
    s: float,
    z: float,
    z_s: float,
    *,
    rho_1: float,
    rho_2: float,
    h_1: float,
    lambda_max_factor: float = 200.0,
    n_log: int = 32,
    n_lin: int = 96,
) -> float:
    """Cross-layer real-space scalar Green's function $\\varphi(s, z, z_s)$.

    Computes $\\varphi(s, z, z_s) / I = \\int_0^\\infty
    \\Phi(\\lambda, z, z_s)\\,J_0(\\lambda s)\\,\\lambda\\,d\\lambda$
    for a unit point source at $(0, 0, z_s)$ observed at $(s, 0, z)$.
    The result already includes the $1/(4\\pi)$-style prefactor that
    is conventional in the existing `mom_sommerfeld` kernel
    representation.

    Numerical strategy: split-grid Sommerfeld quadrature
    (logarithmic for small $\\lambda$, uniform-with-Bessel-resolution
    for large $\\lambda$) consistent with
    :mod:`groundfield.coupling.sommerfeld_inductance` (ADR-0006).

    For ``s = 0`` (coaxial limit) the kernel diverges logarithmically
    when ``z = z_s``; the caller must regularise at the wire radius.

    Parameters
    ----------
    s : float
        Cylindrical radius in m, $\\ge 0$.
    z, z_s : float
        Observer / source depths in m.
    rho_1, rho_2, h_1 : float
        Two-layer soil parameters.
    lambda_max_factor
        Upper bound of the quadrature, given as a multiple of
        $1 / \\bar h$ with $\\bar h = \\min(h_1, s + z + z_s)$.
    n_log, n_lin
        Number of nodes in the logarithmic / linear part of the
        quadrature grid.

    Returns
    -------
    G : float
        Real-space scalar Green's function (dimensionless after
        dividing by $\\rho$).
    """
    char_length = max(min(h_1, s + z + z_s + 1e-9), 1e-3)
    lambda_max = lambda_max_factor / char_length
    lambda_break = max(min(1.0 / char_length, 0.5 * lambda_max), 1e-9)

    # Logarithmic part [eps, lambda_break]
    nodes_log_x, weights_log_x = np.polynomial.legendre.leggauss(n_log)
    eps = max(lambda_max * 1e-12, 1e-15)
    a_log = math.log(eps)
    b_log = math.log(lambda_break)
    t = 0.5 * (nodes_log_x + 1.0) * (b_log - a_log) + a_log
    lambdas_log = np.exp(t)
    weights_log = 0.5 * weights_log_x * (b_log - a_log) * lambdas_log

    # Uniform part [lambda_break, lambda_max] with Bessel resolution
    nodes_lin_x, weights_lin_x = np.polynomial.legendre.leggauss(n_lin)
    half = 0.5 * (lambda_max - lambda_break)
    lambdas_lin = half * (nodes_lin_x + 1.0) + lambda_break
    weights_lin = half * weights_lin_x

    lambdas = np.concatenate([lambdas_log, lambdas_lin])
    weights = np.concatenate([weights_log, weights_lin])

    Phi = two_layer_spectral_kernel(
        lambdas, z, z_s, rho_1=rho_1, rho_2=rho_2, h_1=h_1,
    )
    integrand = Phi * j0(lambdas * s) * lambdas
    return float(np.sum(weights * integrand))


def two_layer_layered_correction_real_space(
    s: float,
    z: float,
    z_s: float,
    *,
    rho_1: float,
    rho_2: float,
    h_1: float,
    rho_baseline: float | None = None,
    lambda_max_factor: float = 200.0,
    n_log: int = 32,
    n_lin: int = 96,
) -> float:
    """Real-space layered *correction* over the homogeneous-rho_1 baseline.

    Computes
    $$
    \\Delta G(s, z, z_s) \\;=\\;
        G_{\\text{2-layer}}(s, z, z_s; \\rho_1, \\rho_2, h_1)
        \\;-\\; G_{\\text{homog}}(s, z, z_s; \\rho_1)
    $$
    by subtracting the spectral kernels **before** the Hankel
    integration. The singular source term
    $\\rho_1/(2\\lambda)\\,e^{-\\lambda|z-z_s|}$ is identical in
    both spectral kernels and **cancels exactly**, leaving a
    smooth, exponentially decaying integrand. Two big advantages:

    1. In the homogeneous limit $\\rho_2 = \\rho_1$ the spectral
       difference is *identically zero* at every $\\lambda$, so the
       integrator returns 0 to machine precision (no spurious
       diagonal contribution from finite-Sommerfeld-quadrature
       error at $z = z_s$).
    2. The kernel difference $(a_1 - a_1^{\\text{hom}})e^{\\lambda z}
       + (b_1 - b_1^{\\text{hom}})e^{-\\lambda z}$ has no
       $1/(2\\lambda)$ piece, so the small-$\\lambda$ region is
       finite and easy to integrate.

    The caller adds this correction on top of the homogeneous
    potential computed with the existing
    :func:`groundfield.solver.image._self_corrected_kernel` (which
    correctly handles the line-self diagonal).

    The optional ``rho_baseline`` parameter selects which homog
    soil to subtract:

    - ``None`` (default): subtract homog with $\\rho_1$. Suitable
      when the calling solver builds $\\phi_\\text{hom}$ uniformly
      with $\\rho_1$.
    - explicit float: subtract homog with the given resistivity.
      Use ``rho_2`` for source segments in the lower layer when
      the calling solver matches its $\\phi_\\text{hom}$ baseline
      to the source layer (recommended for AP1-grade solver,
      ADR-0007 §"Phase A.1").

    Returns 0 in the homogeneous limit by construction; otherwise
    a finite real value with the same units as
    :func:`two_layer_real_space_kernel`.
    """
    if rho_baseline is None:
        rho_baseline = rho_1
    if abs(rho_2 - rho_1) < 1e-12 * max(rho_1, 1.0) and \
            abs(rho_baseline - rho_1) < 1e-12 * max(rho_1, 1.0):
        return 0.0

    char_length = max(min(h_1, s + z + z_s + 1e-9), 1e-3)
    lambda_max = lambda_max_factor / char_length
    lambda_break = max(min(1.0 / char_length, 0.5 * lambda_max), 1e-9)

    nodes_log_x, weights_log_x = np.polynomial.legendre.leggauss(n_log)
    eps = max(lambda_max * 1e-12, 1e-15)
    a_log = math.log(eps)
    b_log = math.log(lambda_break)
    t = 0.5 * (nodes_log_x + 1.0) * (b_log - a_log) + a_log
    lambdas_log = np.exp(t)
    weights_log = 0.5 * weights_log_x * (b_log - a_log) * lambdas_log

    nodes_lin_x, weights_lin_x = np.polynomial.legendre.leggauss(n_lin)
    half = 0.5 * (lambda_max - lambda_break)
    lambdas_lin = half * (nodes_lin_x + 1.0) + lambda_break
    weights_lin = half * weights_lin_x

    lambdas = np.concatenate([lambdas_log, lambdas_lin])
    weights = np.concatenate([weights_log, weights_lin])

    # Layered amplitudes
    A, B, _, C = _spectral_amplitudes(
        lambdas, z_s, rho_1, rho_2, h_1,
    )
    # Baseline (homog) amplitudes — both layers at rho_baseline.
    A_h, B_h, _, C_h = _spectral_amplitudes(
        lambdas, z_s, rho_baseline, rho_baseline, h_1,
    )

    # Evaluate the full Phi for both and subtract. When the source
    # is in the same layer as the baseline-rho region, the
    # singular particular term cancels and the residual is bounded.
    # When the baseline differs from rho_at_source (e.g.,
    # rho_baseline=rho_2 with source in upper at z_s<h_1, where
    # the layered Phi has rho_1/(2λ)·exp() but the baseline-homog
    # Phi has rho_2/(2λ)·exp()), the residual carries a
    # (rho_at_source - rho_baseline)/(4πr) divergence — but the
    # caller is responsible for selecting rho_baseline = rho at
    # source layer to avoid this.
    Phi_lay = _phi_from_stable_amplitudes(
        lambdas, A, B, C, z, z_s, rho_1, rho_2, h_1,
    )
    Phi_hom = _phi_from_stable_amplitudes(
        lambdas, A_h, B_h, C_h, z, z_s, rho_baseline, rho_baseline, h_1,
    )
    delta_Phi = Phi_lay - Phi_hom

    integrand = delta_Phi * j0(lambdas * s) * lambdas
    return float(np.sum(weights * integrand))
