"""Layered-soil Green's-function helpers (shared by several backends).

This module collects the **physics building blocks** that are shared by
``image_nlayer``, ``cim``, ``mom``, ``mom_sommerfeld``, ``bem`` and
``fem`` whenever they need the layer structure of a horizontally
stratified, semi-infinite half-space: the neutral :class:`LayerStack`
container, its canonical form (:func:`canonical_layer_stack`) and the
upward-looking reflection coefficient (:func:`reflection_gamma`).

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
\\frac{e^{-\\lambda |z - z_s|}
     + \\Gamma_1\\, e^{-\\lambda (2 h_1 - |z - z_s|)}
     + e^{-\\lambda (z + z_s)}
     + \\Gamma_1\\, e^{-\\lambda (2 h_1 - z - z_s)}}
     {1 - \\Gamma_1(\\lambda)\\, e^{-2 \\lambda h_1}}
\\, J_0(\\lambda s)\\, d\\lambda,
$$
with **both** reflected image families in the numerator and the
surface-interface multiple-reflection denominator. *(Corrected in
the 2026-07-08 audit, WP-E: the formula previously displayed here —
$e^{-\\lambda|z-z_s|} + \\Gamma_1 e^{-\\lambda(z+z_s)}$ — is
incomplete; it misses the $2h_1$ families and every multiple
bounce. The historic ``cim``/``bem`` $n \\ge 3$ kernels
implemented that incomplete form literally, which is why those
backends now reject $n \\ge 3$ soils; ``mom_sommerfeld``
implements the full form.)*
Here $s$ is the cylindrical radius $\\sqrt{x^2 + y^2}$,
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

Canonical layer stack
---------------------
The reflection coefficient only sees *contrasts*: an interface between
two layers of equal resistivity has $K_i = 0$ and is physically absent,
and a layer of zero thickness contributes $e^{-2\\lambda \\cdot 0} = 1$
in the recursion, i.e. nothing. A declared stack can therefore describe
a soil with fewer layers than it lists — ``[100, 400, 400]`` Ω·m over
``h = [2, 3]`` m *is* the two-layer soil ``(100, 400, h_1 = 2 m)``, and
``[ρ, ρ, ρ]`` is homogeneous. Since the backends choose their kernel by
the layer count (closed-form series for $n \\le 2$, quadrature for
$n \\ge 3$), the declared count must not decide: :func:`as_layer_stack`
returns the **canonical** stack by default
(:func:`canonical_layer_stack`), in which adjacent layers with
resistivities equal to a relative $10^{-12}$ are merged and zero-thickness
layers are dropped (review pass 10, 10.48). Up to 0.15.0 the literal
stack was used, so physically two-layer or homogeneous
``MultiLayerSoil`` declarations were rejected by ``cim`` / ``bem`` /
``mom`` and, for electrodes below $h_1$, by ``mom_sommerfeld`` and
``image_nlayer``.

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

import math
from dataclasses import dataclass

import numpy as np

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    SoilLayer,
    SoilModel,
    TwoLayerSoil,
)

__all__ = [
    "LayerStack",
    "as_layer_stack",
    "canonical_layer_stack",
    "canonical_soil",
    "interface_depths",
    "reflection_gamma",
]

#: Relative tolerance below which two adjacent layer resistivities count
#: as equal and the layers are merged by :func:`canonical_layer_stack`.
#: Two layers that agree to 1e-12 have a reflection coefficient
#: $|K| \le 5 \cdot 10^{-13}$, i.e. no physical interface.
EQUAL_RESISTIVITY_RTOL = 1e-12


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


def as_layer_stack(soil: SoilModel, *, canonical: bool = True) -> LayerStack:
    """Cast any ``SoilModel`` to a :class:`LayerStack`.

    Parameters
    ----------
    soil
        Homogeneous, two-layer or multilayer soil model.
    canonical
        Return the canonical stack (default): adjacent layers of equal
        resistivity merged and zero-thickness layers dropped, see
        :func:`canonical_layer_stack`. ``False`` returns the stack
        exactly as declared.

    Returns
    -------
    LayerStack
        Resistivities in Ω·m and finite thicknesses in m.

    Raises
    ------
    TypeError
        If ``soil`` is not one of the three soil models.

    Notes
    -----
    Inverse of :class:`groundfield.soil.SoilModel`: a homogeneous soil
    becomes ``n=1``, a 2-layer soil ``n=2`` and a multilayer soil keeps
    its declared layer count — before canonicalisation. Every layered
    backend reads the canonical stack, so the kernel is chosen by the
    number of physical interfaces, not by the number of declared layers
    (review pass 10, 10.48).
    """
    if isinstance(soil, HomogeneousSoil):
        stack = LayerStack(
            rhos=np.array([soil.resistivity], dtype=float),
            h=np.zeros(0, dtype=float),
        )
    elif isinstance(soil, TwoLayerSoil):
        stack = LayerStack(
            rhos=np.array([soil.rho_1, soil.rho_2], dtype=float),
            h=np.array([soil.h_1], dtype=float),
        )
    elif isinstance(soil, MultiLayerSoil):
        rhos = np.array([ly.resistivity for ly in soil.layers], dtype=float)
        h = np.array([ly.thickness for ly in soil.layers[:-1]], dtype=float)
        stack = LayerStack(rhos=rhos, h=h)
    else:
        raise TypeError(f"Cannot convert {type(soil).__name__} to LayerStack.")
    return canonical_layer_stack(stack) if canonical else stack


def canonical_layer_stack(
    stack: LayerStack, *, rtol: float = EQUAL_RESISTIVITY_RTOL
) -> LayerStack:
    """Merge layers that the Green's function cannot tell apart.

    Two rules, applied in this order:

    1. A finite layer of zero thickness is dropped — its attenuation
       factor $e^{-2 \\lambda h}$ is 1 for every $\\lambda$, so it
       leaves $\\Gamma_1(\\lambda)$ unchanged.
    2. Adjacent layers whose resistivities agree to the relative
       tolerance ``rtol`` are merged into one layer whose thickness is
       the sum of theirs (semi-infinite if the lower one is the bottom
       half-space). The interface between them has
       $K = (\\rho_{i+1} - \\rho_i)/(\\rho_{i+1} + \\rho_i) \\approx 0$
       and is physically absent. The upper layer's resistivity is kept.

    Parameters
    ----------
    stack
        Layer stack as declared.
    rtol
        Relative resistivity tolerance of rule 2 (dimensionless).

    Returns
    -------
    LayerStack
        The canonical stack. ``stack`` itself (same object) when it is
        already canonical, so the common case costs nothing and stays
        bit-identical.

    Examples
    --------
    ``[100, 400, 400]`` Ω·m over ``[2, 3]`` m becomes ``[100, 400]``
    over ``[2]`` m; ``[100, 100, 100]`` becomes the homogeneous
    ``[100]``.
    """
    groups = _canonical_groups(stack, rtol=rtol)
    if len(groups) == stack.n_layers:
        return stack
    thick = [float(t) for t in stack.h] + [math.inf]
    return LayerStack(
        rhos=np.array([float(stack.rhos[g[0]]) for g in groups], dtype=float),
        h=np.array([sum(thick[i] for i in g) for g in groups[:-1]], dtype=float),
    )


def _canonical_groups(
    stack: LayerStack, *, rtol: float = EQUAL_RESISTIVITY_RTOL
) -> list[list[int]]:
    """Indices of the declared layers that form each canonical layer.

    Zero-thickness layers belong to no group; a layer joins the group
    above it when its resistivity agrees with the group's (upper)
    resistivity to ``rtol`` (see :func:`canonical_layer_stack`).
    """
    rhos = [float(r) for r in stack.rhos]
    thick = [float(t) for t in stack.h] + [math.inf]
    groups: list[list[int]] = []
    for i, (rho, t) in enumerate(zip(rhos, thick, strict=True)):
        if t == 0.0:
            continue
        if groups:
            top = rhos[groups[-1][0]]
            if abs(rho - top) <= rtol * max(abs(rho), abs(top)):
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups


def interface_depths(soil: SoilModel) -> tuple[float, ...]:
    """Depths of the physical layer interfaces of ``soil`` in m.

    The interfaces of the canonical stack (:func:`canonical_layer_stack`):
    only where the resistivity changes. Every segment-based backend cuts
    wires at these depths and the pre-flight prediction
    (:func:`groundfield.diagnostics.expected_segments`) counts with the
    same ones, so both agree (audit round 2, A2S-07: a declared interface
    between layers of one resistivity was counted but not cut).

    Parameters
    ----------
    soil
        Homogeneous, two-layer or multilayer soil model.

    Returns
    -------
    tuple of float
        Interface depths in m, increasing; empty for a soil that is
        homogeneous after canonicalisation.
    """
    stack = as_layer_stack(soil)
    if stack.n_layers <= 1:
        return ()
    return tuple(float(z) for z in np.cumsum(stack.h))


def _relative_permittivities(soil: SoilModel) -> list[float]:
    """Relative permittivity of every declared layer of ``soil``."""
    if isinstance(soil, MultiLayerSoil):
        return [float(layer.relative_permittivity) for layer in soil.layers]
    n = 2 if isinstance(soil, TwoLayerSoil) else 1
    return [float(soil.relative_permittivity)] * n


def canonical_soil(soil: SoilModel) -> SoilModel:
    """The soil model of the canonical layer stack of ``soil``.

    Backends store this model in :attr:`FieldResult.soil`, so that the
    post-solve potential evaluation picks the kernel of the soil that
    was actually solved (a ``MultiLayerSoil`` with three declared but
    two physical layers is evaluated with the two-layer series instead
    of being rejected as $n \\ge 3$).

    Parameters
    ----------
    soil
        Homogeneous, two-layer or multilayer soil model.

    Returns
    -------
    SoilModel
        A :class:`HomogeneousSoil` ($n = 1$), :class:`TwoLayerSoil`
        ($n = 2$) or :class:`MultiLayerSoil` ($n \\ge 3$) of the
        canonical stack. ``soil`` itself is returned when its stack is
        already canonical and its type already matches that layer count
        (a ``MultiLayerSoil`` with one or two layers is mapped to the
        plain model).

    Notes
    -----
    The relative permittivity is carried over from the upper declared
    layer of each merged group (``TwoLayerSoil`` holds one common value:
    that of its upper layer). No backend uses it below 1 kHz; it is
    kept so that the result's soil does not silently fall back to the
    default.
    """
    raw = as_layer_stack(soil, canonical=False)
    stack = canonical_layer_stack(raw)
    if stack is raw and not (isinstance(soil, MultiLayerSoil) and stack.n_layers <= 2):
        return soil
    eps_declared = _relative_permittivities(soil)
    eps = [eps_declared[g[0]] for g in _canonical_groups(raw)]
    if stack.n_layers == 1:
        return HomogeneousSoil(
            resistivity=float(stack.rhos[0]), relative_permittivity=eps[0]
        )
    if stack.n_layers == 2:
        return TwoLayerSoil(
            rho_1=float(stack.rhos[0]),
            rho_2=float(stack.rhos[1]),
            h_1=float(stack.h[0]),
            relative_permittivity=eps[0],
        )
    layers = [
        SoilLayer(resistivity=float(r), thickness=float(t), relative_permittivity=e)
        for r, t, e in zip(stack.rhos[:-1], stack.h, eps[:-1], strict=True)
    ]
    layers.append(
        SoilLayer(
            resistivity=float(stack.rhos[-1]),
            thickness=None,
            relative_permittivity=eps[-1],
        )
    )
    return MultiLayerSoil(layers=layers)


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
