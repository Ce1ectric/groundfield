"""Layered-soil Green's-function helpers (shared by several backends).

This module collects the **physics building blocks** that are shared by
``image_nlayer``, ``cim``, ``mom_sommerfeld`` and ``bem`` whenever they
need to evaluate the potential of a point current source inside a
horizontally stratified, semi-infinite half-space.

Mathematical / physical model
-----------------------------
For a point current source $I$ placed at depth $z_s > 0$
in the **uppermost layer** (resistivity $\\rho_1$, thickness
$h_1$) of an n-layer half-space with layer interfaces at
$z_i = h_1 + h_2 + \\dots + h_i$ (positive $z$ axis pointing
into the soil, surface at $z = 0$), the quasi-static potential at
a field point $(s, z)$ inside the upper layer is given by the
**Sommerfeld integral**
$$
\\varphi(s, z) \\;=\\; \\frac{\\rho_1\\, I}{4\\pi}\\,
\\int_{0}^{\\infty}
\\bigl[\\, e^{-\\lambda |z - z_s|}
     + \\Gamma_1(\\lambda)\\, e^{-\\lambda (z + z_s)}\\bigr]
\\, J_0(\\lambda s)\\, d\\lambda,
$$
where $s$ is the cylindrical radius $\\sqrt{x^2 + y^2}$,
$J_0$ the Bessel function of the first kind, and
$\\Gamma_1(\\lambda)$ the **upward-looking reflection coefficient**
seen at the bottom of layer 1, computed recursively from the bottom up:
$$
\\Gamma_{n-1}(\\lambda) &= K_{n-1}, \\\\
\\Gamma_i(\\lambda)     &= \\frac{K_i + \\Gamma_{i+1}(\\lambda)\\,
                                    e^{-2\\lambda h_{i+1}}}
                                   {1 + K_i\\,\\Gamma_{i+1}(\\lambda)\\,
                                    e^{-2\\lambda h_{i+1}}},
\\qquad i = n-2, \\dots, 1,
$$
with the layer-pair Fresnel coefficients
$$
K_i \\;=\\; \\frac{\\rho_{i+1} - \\rho_i}{\\rho_{i+1} + \\rho_i}.
$$
The free soil surface at $z = 0$ reflects perfectly
($R_{\\text{air}} = +1$); the source's image at $z = -z_s$
is therefore included with weight 1. Multiple bouncing between the air
boundary and the layer system below gives the closed expression
multiplying $\\Gamma_1(\\lambda)$, which the four backends below
each evaluate by a different numerical strategy (image series, complex
images, direct quadrature, BEM). The integral is **quasi-static** —
valid for the frequency band targeted by ``groundfield``
($f < 1\\,\\mathrm{kHz}$).

Reduction to special cases
--------------------------
- ``n = 1`` (homogeneous): $\\Gamma_1 \\equiv 0$, so the integral
  collapses to $1/r + 1/r_{\\text{img}}$ — the classical
  homogeneous image-charge backend.
- ``n = 2`` (two-layer): $\\Gamma_1 \\equiv K_1$ constant in
  $\\lambda$, and the closed-form Tagg/Sunde image-charge
  geometric series follows directly.
- ``n \\ge 3``: $\\Gamma_1(\\lambda)$ is a rational function of
  $e^{-2\\lambda h_i}$ that no longer has a simple geometric
  expansion; this is the regime where the four engines below have to
  pick a numerical strategy (see ADR-0002).

References
----------
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover.
- Dawalibi, F. P., & Barbeito, N. (1991). Measurements and computations
  of the performance of grounding systems buried in multilayer soils,
  IEEE PWRD 6(4).
- Li, Z.-X. et al. (2006). A novel mathematical modeling of grounding
  system buried in multilayer earth, IEEE PWRD 21(3).
- Zou, J. et al. (2015). Fast calculation of the Green function of a
  point current source in a horizontal layered soil with a new complex
  path, IEEE Trans. Magn. 51(3).
- Dan, Y. et al. (2021). Segmented sampling least squares algorithm
  for Green's function of arbitrary layered soil, IEEE PWRD 36(3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    SoilModel,
    TwoLayerSoil,
)

__all__ = [
    "LayerStack",
    "as_layer_stack",
    "reflection_gamma",
    "image_series_offsets",
]


# ---------------------------------------------------------------------
# Layer stack — neutral container for n layers
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LayerStack:
    """Neutral container for an n-layer soil model.

    Parameters
    ----------
    rhos : np.ndarray, shape (n,)
        Layer resistivities $\\rho_1, \\dots, \\rho_n$.
    h : np.ndarray, shape (n-1,)
        Finite layer thicknesses $h_1, \\dots, h_{n-1}$. The
        last (n-th) layer is semi-infinite. For ``n == 1`` ``h`` is
        empty.
    """

    rhos: np.ndarray
    h: np.ndarray

    @property
    def n_layers(self) -> int:
        return int(self.rhos.size)

    @property
    def K(self) -> np.ndarray:
        """Per-interface Fresnel coefficient $K_i = (\\rho_{i+1} - \\rho_i) / (\\rho_{i+1} + \\rho_i)$."""
        if self.n_layers <= 1:
            return np.zeros(0)
        rho = self.rhos
        return (rho[1:] - rho[:-1]) / (rho[1:] + rho[:-1])


def as_layer_stack(soil: SoilModel) -> LayerStack:
    """Cast any ``SoilModel`` to a :class:`LayerStack`.

    Notes
    -----
    Inverse of :class:`groundfield.soil.SoilModel`: a homogeneous soil
    becomes ``n=1``, a 2-layer soil becomes ``n=2``, and a multilayer
    soil keeps its layer count.
    """
    if isinstance(soil, HomogeneousSoil):
        return LayerStack(
            rhos=np.array([soil.resistivity], dtype=float),
            h=np.zeros(0, dtype=float),
        )
    if isinstance(soil, TwoLayerSoil):
        return LayerStack(
            rhos=np.array([soil.rho_1, soil.rho_2], dtype=float),
            h=np.array([soil.h_1], dtype=float),
        )
    if isinstance(soil, MultiLayerSoil):
        rhos = np.array([ly.resistivity for ly in soil.layers], dtype=float)
        h = np.array(
            [ly.thickness for ly in soil.layers[:-1]], dtype=float
        )
        return LayerStack(rhos=rhos, h=h)
    raise TypeError(
        f"Cannot convert {type(soil).__name__} to LayerStack."
    )


# ---------------------------------------------------------------------
# Recursive reflection coefficient Γ_1(λ)
# ---------------------------------------------------------------------


def reflection_gamma(stack: LayerStack, lam: np.ndarray) -> np.ndarray:
    """Upward-looking reflection coefficient $\\Gamma_1(\\lambda)$.

    Built bottom-up from the per-interface Fresnel coefficients,
    $\\Gamma_{n-1} = K_{n-1}$,
    $\\Gamma_i = (K_i + \\Gamma_{i+1} e^{-2\\lambda h_{i+1}}) /
    (1 + K_i \\Gamma_{i+1} e^{-2\\lambda h_{i+1}})$.

    Parameters
    ----------
    stack
        Layer stack.
    lam
        Sommerfeld variable $\\lambda$, real and non-negative.
        Any shape; the operation is element-wise.

    Returns
    -------
    gamma : np.ndarray
        $\\Gamma_1(\\lambda)$, same shape as ``lam``. For
        ``n_layers == 1`` returns zero (homogeneous half-space).
    """
    lam_arr = np.asarray(lam, dtype=float)
    if stack.n_layers <= 1:
        return np.zeros_like(lam_arr)

    K = stack.K  # shape (n-1,)
    h = stack.h  # shape (n-1,)
    # Bottom-up recursion.
    # Last interface (between layers n-1 and n): Γ_{n-1}(λ) = K_{n-1}
    gamma = np.full_like(lam_arr, K[-1])
    for i in range(stack.n_layers - 3, -1, -1):
        # Γ_i ← (K_i + Γ_{i+1}·e^{-2λ·h_{i+1}}) / (1 + K_i·Γ_{i+1}·e^{-2λ·h_{i+1}})
        # Note: h_{i+1} is the thickness of layer i+1 (1-based).
        e = np.exp(-2.0 * lam_arr * h[i + 1])
        num = K[i] + gamma * e
        den = 1.0 + K[i] * gamma * e
        gamma = num / den
    return gamma


# ---------------------------------------------------------------------
# Image-series expansion of Γ_1(λ) at fixed depths
# ---------------------------------------------------------------------


def image_series_offsets(
    stack: LayerStack,
    *,
    max_terms: int = 200,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometric expansion of Γ_1(λ) in $e^{-2\\lambda \\bar h}$.

    The recursive Γ_1(λ) of an n-layer stack can be expanded
    formally as a power series in the per-layer attenuation factors
    $e^{-2\\lambda h_i}$, producing a (truncated) expansion of
    the form
    $$
    \\Gamma_1(\\lambda) \\;\\approx\\; \\sum_{k=1}^{M} w_k\\,
    e^{-2\\lambda \\Delta_k},
    $$
    with $\\Delta_k = \\sum_i n_{k,i}\\,h_i$ an integer
    combination of the layer thicknesses. Each term turns directly
    into one image-charge offset of the spatial Green's function.

    The expansion is generated by recursively expanding
    $1 / (1 + K_i\\Gamma_{i+1} e^{-2\\lambda h_{i+1}})$ as a
    geometric series and accumulating contributions until the weight
    falls below ``tol`` or ``max_terms`` is reached.

    Parameters
    ----------
    stack
        Layer stack.
    max_terms
        Hard upper bound on the number of returned image terms.
    tol
        Absolute weight cutoff. A term is dropped (and its descendants
        not generated) when its absolute weight falls below ``tol``.

    Returns
    -------
    weights : np.ndarray
        Real per-term weight $w_k$. Empty for ``n_layers == 1``.
    deltas : np.ndarray
        Per-term offset $\\Delta_k$ (in metres), same shape as
        ``weights``. The matching spatial image of a source at
        $(x_s, y_s, z_s)$ sits at
        $(x_s, y_s, 2\\Delta_k - z_s)$ and the image of its
        air-reflected mirror at $(x_s, y_s, 2\\Delta_k + z_s)$,
        each weighted by $w_k$.

    Notes
    -----
    The expansion converges geometrically as long as
    $|K_i\\,\\Gamma_{i+1}| < 1$ for every interface — which is
    the case for every physically reasonable soil contrast (it would
    take $|K| = 1$ to fail, i.e. an interface to a perfect
    insulator or a perfect conductor).
    """
    if stack.n_layers <= 1:
        return np.zeros(0), np.zeros(0)

    K = stack.K
    h = stack.h
    n = stack.n_layers

    # Worklist entries: (current_layer_index, current_weight, current_delta).
    # We expand the recursion from the top (i=0) downwards. Each "step"
    # at interface i contributes:
    #   Γ_i ≈ K_i + (1 - K_i^2) Γ_{i+1} e^{-2λ h_{i+1}}
    #         + Σ_{m≥1} (-K_i)^m Γ_{i+1}^{m+1} e^{-2λ (m+1) h_{i+1}}
    # This is the formal expansion of (K + x)/(1 + K x) with
    # x = Γ_{i+1} e^{-2λ h_{i+1}}.
    #
    # The implementation walks recursively: at interface i, multiply by
    # K_i (deepest contribution) or by the (1 − K_i²)·Γ_{i+1}·e^{-2λ h_{i+1}}
    # factor and recurse into Γ_{i+1}; for higher-order terms, multiply by
    # an extra power of (−K_i·Γ_{i+1}·e^{-2λ h_{i+1}}). For practical
    # n_layers ≤ 4 the number of generated terms is well below max_terms.

    weights: list[float] = []
    deltas: list[float] = []

    def _expand(layer: int, w: float, delta: float, depth: int) -> None:
        """Recurse on interface ``layer`` (0-based) with running weight ``w``
        and offset ``delta``."""
        if depth >= max_terms or abs(w) < tol:
            return
        if layer >= n - 1:
            # No further interface: terminate with the running term.
            weights.append(w)
            deltas.append(delta)
            return
        if layer == n - 2:
            # Bottom-most interface: Γ_{n-1} = K_{n-1}, no recursion.
            weights.append(w * K[layer])
            deltas.append(delta)
            return
        # Constant part:  K_i  →  emit a term and stop.
        weights.append(w * K[layer])
        deltas.append(delta)
        # First-order part: (1 − K_i²) · Γ_{i+1} · e^{-2λ h_{i+1}}
        # → recurse into layer i+1 with weight w·(1 − K_i²) and offset
        # delta + h_{i+1}.
        first_w = w * (1.0 - K[layer] ** 2)
        first_delta = delta + h[layer + 1]
        _expand(layer + 1, first_w, first_delta, depth + 1)
        # Higher-order: ·(−K_i·Γ_{i+1}·e^{-2λ h_{i+1}})^m for m ≥ 1.
        # We keep m up to the point where the prefactor drops below tol.
        # Since |Γ_{i+1}| ≤ 1, the geometric step factor is at most |K_i|.
        m = 1
        cur_w = first_w
        cur_delta = first_delta
        while True:
            cur_w *= -K[layer]
            cur_delta += h[layer + 1]
            if abs(cur_w) < tol or depth + m + 1 >= max_terms:
                break
            _expand(layer + 1, cur_w, cur_delta, depth + m + 1)
            m += 1

    _expand(layer=0, w=1.0, delta=0.0, depth=0)

    # Aggregate identical offsets (numerical robustness).
    if not weights:
        return np.zeros(0), np.zeros(0)
    deltas_arr = np.array(deltas)
    weights_arr = np.array(weights)
    # Round delta to 1 µm to deduplicate.
    keys = np.round(deltas_arr / 1e-6).astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    agg_w = np.zeros(uniq.size)
    np.add.at(agg_w, inv, weights_arr)
    agg_d = uniq.astype(float) * 1e-6
    # Sort by offset.
    order = np.argsort(agg_d)
    return agg_w[order], agg_d[order]


# ---------------------------------------------------------------------
# Vectorised distance helpers — used by image_nlayer and cim
# ---------------------------------------------------------------------


def cylindrical_radius(field_pts: np.ndarray, source_pts: np.ndarray) -> np.ndarray:
    """Cylindrical radius $s_{mn}$ between every (field, source) pair.

    Parameters
    ----------
    field_pts : np.ndarray, shape (M, 3)
    source_pts : np.ndarray, shape (N, 3)

    Returns
    -------
    s : np.ndarray, shape (M, N)
        $s_{mn} = \\sqrt{(x_m - x_n)^2 + (y_m - y_n)^2}$.
    """
    diff = field_pts[:, None, 0:2] - source_pts[None, :, 0:2]
    return np.sqrt(np.einsum("mnk,mnk->mn", diff, diff))


def _split_layer_thicknesses(soil: SoilModel) -> Iterable[float]:
    """Iterate over the finite layer thicknesses of a soil model."""
    stack = as_layer_stack(soil)
    return list(stack.h.tolist())
