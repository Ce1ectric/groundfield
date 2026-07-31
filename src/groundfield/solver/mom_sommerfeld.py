"""Method-of-Moments backend with **direct Sommerfeld quadrature** (``mom_sommerfeld``).

Mathematical / physical model
-----------------------------
The other layered backends (``image_nlayer``, ``cim``, ``bem``) rely
on a *closed-form* representation of the layered Green's function
(real image series or complex images). This backend instead evaluates
the Sommerfeld integral
$$
\\varphi(s, z; z_s) \\;=\\; \\frac{\\rho_1\\, I}{4\\pi}
\\!\\int_0^{\\infty}\\! \\bigl[ e^{-\\lambda |z - z_s|}
    + \\Gamma_1(\\lambda)\\, e^{-\\lambda (z + z_s)}\\bigr]
J_0(\\lambda s)\\, d\\lambda
$$
**numerically**, point by point, with adaptive Gauss–Kronrod
quadrature (``scipy.integrate.quad``). The recursive
$\\Gamma_1(\\lambda)$ from
:func:`groundfield.solver._layered.reflection_gamma` is used as is —
no expansion, no fit. The result is therefore **independent** of the
expansion choices in ``image_nlayer`` / ``cim`` and serves as the
absolute reference inside the cross-engine validation.

Quadrature strategy
-------------------
The integrand is **not** absolutely integrable: the direct term
$e^{-\\lambda |z - z_s|} J_0(\\lambda s)$ does not decay at all
for two segments at the same depth ($|z - z_s| = 0$), and the
integral converges only through the oscillation of $J_0$. A
finite real-axis truncation $[0, \\lambda_{\\max}]$ therefore
leaves a sign-alternating error of order
$\\sqrt{2 / (\\pi \\lambda_{\\max} s)}$ that no quadrature
tolerance can remove (fixed in 0.15.0; before that the backend could
return a *negative* Green's function).

The kernel is therefore split into an analytic part and a genuinely
decaying remainder, using **Lipschitz' integral**
$$
\\int_0^{\\infty} e^{-\\lambda d}\\, J_0(\\lambda s)\\, d\\lambda
= \\frac{1}{\\sqrt{s^2 + d^2}}, \\qquad d \\ge 0 .
$$
Three terms are integrated in closed form — the direct term
($d = |z - z_s|$), its air mirror ($d = z + z_s$) and the
leading interface reflection with the $\\lambda \\to \\infty$
limit $\\Gamma_1(\\infty) = K_1$ of the reflection coefficient
($d = 2 h_1 - z - z_s$). The remainder carries at least the
exponential $e^{-\\lambda \\min(2 h_1 - |z - z_s|,\\;
2 h_1 - z - z_s + \\Delta)}$, i.e. it decays on a strictly positive
length scale even when both points sit on the layer interface, so a
truncated quadrature is now *controlled*: the discarded tail is
bounded by $e^{-\\lambda_{\\max} d_{\\text{rem}}}$.

The remainder is evaluated with fixed-order Gauss–Legendre panels on
a grid that (a) is graded towards $\\lambda = 0$ to resolve the
exponentials and $\\Gamma_1(\\lambda)$, (b) carries a geometric
(log-spaced) family down to the $\\lambda$-scale
$(1 - \\Gamma_1(0)) / 2 h_1$ of the multiple-reflection multiplier,
which collapses towards $\\lambda = 0$ for $|K_1| \\to 1$, and
(c) is subdivided at half-period spacing $\\pi / s$ to resolve the
$J_0$ oscillation. The panel count is refined until two successive
grids agree within ``max(epsabs, epsrel · |G|)``, which makes the
tolerance arguments meaningful (the previous adaptive
``scipy.integrate.quad`` call could not honour them at all, because
its ``limit=400`` subdivision cap was reached long before the
requested accuracy).

Failing that test is **not** silent: a
:class:`SommerfeldConvergenceWarning` is raised naming the achieved
versus the requested tolerance, the fact that the sign may be wrong
and the parameter to raise. Up to 0.15.0 the same condition was only
logged at ``DEBUG`` level, and — because the oscillation panel budget
made the resolved $\\lambda$-range *shrink* with refinement — the
**least** accurate of the four grid values was returned. In the
corner $s / d_{\\text{rem}} \\gtrsim 2 \\cdot 10^4$ (a
centimetre-thin intermediate layer with a kilometre-scale horizontal
separation) that reproduced the original F04 failure mode: the kernel
returned $-9.47 \\cdot 10^{-5}$ where the far-field asymptote
$G \\to (2/s)\\,\\rho_n/\\rho_1$ requires $+5 \\cdot 10^{-4}$,
with **no** Python-level warning.

The MoM resolution itself (Galerkin scheme with one constraint per
cluster) is the same as :mod:`groundfield.solver.mom`; only the
underlying Z-matrix kernel differs.

Validity
--------
- Quasi-static, $f < 1\\,\\mathrm{kHz}$.
- All segments in the **upper** layer, except for the $n = 2$
  cross-layer case, which is delegated to
  :mod:`groundfield.coupling.layered_green`.
- Slower than the closed-form backends but methodologically
  independent. Use it as the **reference** in cross-engine tests on
  layered worlds with hard contrasts: the pointwise kernel matches
  the analytic Tagg/Sunde series to $\\lesssim 10^{-13}$ relative
  (median $\\sim 10^{-16}$) for
  $\\rho_2 / \\rho_1 \\in [10^{-2}, 10^{2}]$, so any deviation of
  a closed-form backend is that backend's series / fit error.
- Distances are floored at 1 mm (``_MIN_DISTANCE``). That floor is
  applied to the *analytic* $K_1 / \\sqrt{s^2 + d_{\\text{int}}^2}$
  interface-image term but **not** to the matching
  $K_1 e^{-\\lambda d_{\\text{int}}}$ subtraction inside the
  numerically integrated residual, so for
  $d_{\\text{int}} = 2 h_1 - z - z_s < 1\\,\\mathrm{mm}$ the split
  is no longer algebraically exact and the reflected part is
  saturated by the floor:
  ``_reflected_integral(s=0, dz=0, z+z_s=2(h_1 - 10^{-5}))`` returns
  667.561 where the exact reflected series is 33334.23 (−98 %); at
  $d_{\\text{int}} = 2\\,\\mathrm{mm}$ the same call is exact. This
  is the documented distance-floor convention (identical to the
  homogeneous ``image`` backend), not a sign error, but it is far
  sharper here than a 1 mm geometric floor suggests. Reachable
  through the $n \\ge 3$ diagonal when a segment midpoint sits
  within 0.5 mm of the layer interface — keep segment midpoints at
  least a few millimetres clear of $h_1$.

References
----------
- Sommerfeld, A. (1909). Über die Ausbreitung der Wellen in der
  drahtlosen Telegraphie. *Annalen der Physik* 28.
- Watson, G. N. (1944). *A Treatise on the Theory of Bessel
  Functions*, 2nd ed., §13.2 — Lipschitz' integral
  $\\int_0^\\infty e^{-\\lambda d} J_0(\\lambda s)\\,d\\lambda
  = (s^2 + d^2)^{-1/2}$, the identity used for the analytic
  extraction.
- Lucas, S. K., & Stone, H. A. (1995). Evaluating infinite integrals
  involving Bessel functions of arbitrary order. *J. Comput. Appl.
  Math.* 64 — panel quadrature between the zeros of $J_0$.
- Zou, J., Du, X., & Zhou, C. (2015). Fast calculation of the Green
  function of a point current source in a horizontal layered soil
  with a new complex path. *IEEE Trans. Magn.* 51(3). A complex
  contour is an *alternative* to the analytic extraction used here;
  it is not implemented.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.special import j0

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.solver._layered import (
    LayerStack,
    as_layer_stack,
    reflection_gamma,
)
from groundfield.solver.image import (
    _MIN_DISTANCE,
    _assemble_inductance_matrix,
    _build_clusters,
    _build_distributed_topology,
    _build_finite_branches,
    _discretize_electrode,
    _reject_concrete_shells,
    _self_corrected_kernel,
    _Segment,
    _warn_ignored_sources,
)
from groundfield.solver.mom import _galerkin_solve
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = [
    "solve_mom_sommerfeld",
    "sommerfeld_kernel_value",
    "SommerfeldConvergenceWarning",
]

_log = get_logger(__name__)


class SommerfeldConvergenceWarning(UserWarning):
    """The reflected-remainder quadrature did not reach ``epsrel``.

    Emitted (in addition to the logger record) whenever the grid
    refinement ladder of :func:`_reflected_integral` exhausts all of
    :data:`_REFINEMENTS` without two successive grids agreeing within
    ``max(epsabs, epsrel · |G|)``. The returned value is then **not**
    converged, and because the integrand is oscillatory
    ($J_0(\\lambda s)$) a non-converged panel grid does not merely
    lose digits — it can return a Green's function with the **wrong
    sign** (the pre-0.15.0 failure mode of this backend, review-pass-9
    finding F04). Up to 0.15.0 the same situation was only logged at
    ``DEBUG`` level and the *least* accurate of the four grid values
    was returned silently.

    Two mechanisms reach this branch:

    * **Oscillation budget.** ``max_osc_panels`` (default 200 000,
      i.e. ``Engine.sommerfeld_max_osc_panels``) bounds the
      half-period panel family $\\pi / s$ that resolves
      $J_0(\\lambda s)$ on
      $[0, 34 / d_{\\text{rem}}]$. The budget binds when
      $s / d_{\\text{rem}} \\gtrsim 1.9 \\cdot 10^4$ — a
      centimetre-thin intermediate layer ($d_{\\text{rem}} =
      2\\min(h_1,h_2)$) combined with a kilometre-scale horizontal
      separation. Remedy: **raise** ``max_osc_panels`` (runtime and
      memory grow linearly with it).
    * **Layer contrast.** For $|K_1| \\to 1$ the multiple-reflection
      multiplier $1/(1 - \\Gamma_1 e^{-2\\lambda h_1})$ develops a
      $\\lambda$-scale $(1 - \\Gamma_1(0)) / 2 h_1$ that the graded
      base grid has to resolve. Since 0.15.0 a geometric panel family
      covers that scale explicitly, which removed the historic
      $-12.9\\,\\%$ error at $\\rho_2/\\rho_1 = 10^8$; if the
      warning still appears here the contrast is outside the validated
      envelope $\\rho_2/\\rho_1 \\in [10^{-2}, 10^{2}]$ and no knob
      fixes it — use ``image_2layer`` (exact geometric series) instead.

    Silence with
    ``warnings.simplefilter("ignore", SommerfeldConvergenceWarning)``
    only if the affected entries are known to be negligible.
    """


# --- quadrature parameters of the remainder integral -----------------
#
# Gauss–Legendre order per panel. A 12-point rule integrates one half
# period of J_0 and a smooth exponential to machine precision.
_GAUSS_ORDER = 12
# Panels of the graded (non-oscillatory) base grid. The grid is
# quadratically graded towards λ = 0, so it resolves both the slowly
# decaying remainder terms and the fast ones (e^{-λ(2h_1+z+z_s)}).
_N_DECAY_PANELS = 96
# Largest exponent honoured for the truncation λ_max·d_rem. e^{-60} ≈
# 9e-27, so raising ``lambda_max_factor`` beyond this value cannot
# change the result — the knob saturates instead of destabilising the
# quadrature (regression guard for the historic sign flip).
_MAX_DECAY_EXPONENT = 60.0
# Above λ·d_rem = 34 the remainder envelope is below 2e-15 relative,
# so the J_0 oscillation no longer needs to be resolved there.
_OSC_DECAY_EXPONENT = 34.0
# Default budget for the number of oscillation panels. Only reachable
# for extreme s / d_rem ratios (>~ 1.9e4); exposed as
# ``max_osc_panels`` / ``Engine.sommerfeld_max_osc_panels`` so the
# corner has a documented escape hatch.
_MAX_OSC_PANELS = 200_000
# Geometric (log-spaced) panels per decade used to resolve the
# λ-scale (1 - Γ_1(0))/(2 h_1) of the multiple-reflection multiplier
# 1/(1 - Γ_1 e^{-2λh_1}). For |K_1| → 1 that scale collapses towards
# λ = 0 and the quadratically graded base grid misses it entirely:
# at ρ_2/ρ_1 = 10^8 the kernel came out 12.9 % low (18.0327 instead of
# 20.4209 at s = 0.5, z = z_s = 0.8, h_1 = 2) *after* the internal
# convergence test had already failed. 8 panels/decade of 12 Gauss
# nodes integrate the smooth Lorentzian-like multiplier to machine
# precision.
_POLE_PANELS_PER_DECADE = 8
# The geometric family starts a factor _POLE_MARGIN below the pole
# scale, so the innermost (plain) panel [0, λ_lo] sits inside the flat
# region of the multiplier.
_POLE_MARGIN = 0.25
# Successive grid refinements used for the error estimate.
_REFINEMENTS = (1, 2, 4, 8)

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GAUSS_ORDER)


class _PanelGrid(NamedTuple):
    """Panel edges plus the diagnostics needed for the error report."""

    edges: np.ndarray
    #: λ up to which the J_0 half periods are actually resolved.
    lam_covered: float
    #: λ up to which they *should* be resolved (``min(lam_osc, lam_max)``).
    lam_required: float
    #: Half-period panels needed for full coverage at refinement 1.
    osc_panels_required: int


class _QuadratureIssue(NamedTuple):
    """One non-converged remainder quadrature (see :func:`_reflected_integral`)."""

    achieved: float
    tol: float
    s: float
    abs_dz: float
    z_plus: float
    grid: _PanelGrid


def _format_convergence_warning(
    issue: _QuadratureIssue,
    *,
    epsabs: float,
    epsrel: float,
    max_osc_panels: int,
    n_affected: int = 1,
) -> str:
    """Human-readable message for :class:`SommerfeldConvergenceWarning`."""
    grid = issue.grid
    capped = grid.lam_covered < grid.lam_required
    where = (
        f"s = {issue.s:.4g} m, |z - z_s| = {issue.abs_dz:.4g} m, "
        f"z + z_s = {issue.z_plus:.4g} m"
    )
    head = (
        "mom_sommerfeld: the reflected-remainder quadrature did not "
        f"converge. Worst pair ({where}): the last two grids of the "
        f"refinement ladder {list(_REFINEMENTS)} still differ by "
        f"{issue.achieved:.3e}, against a requested tolerance of "
        f"{issue.tol:.3e} (epsabs={epsabs:.1e}, epsrel={epsrel:.1e})."
    )
    if n_affected > 1:
        head += f" {n_affected} reaction-matrix entries are affected."
    cause = (
        (
            " Cause: the J_0(lambda*s) oscillation is resolved only up "
            f"to lambda = {grid.lam_covered:.4g} 1/m of the required "
            f"{grid.lam_required:.4g} 1/m, because full coverage needs "
            f"{grid.osc_panels_required} half-period panels but "
            f"max_osc_panels = {max_osc_panels}. Remedy: raise "
            "'max_osc_panels' (Engine.sommerfeld_max_osc_panels); "
            "runtime and memory grow linearly with it."
        )
        if capped
        else (
            " The oscillation grid is fully covered, so the residual "
            "error sits in the layer contrast: for |K_1| -> 1 the "
            "multiple-reflection multiplier 1/(1 - Gamma_1*exp(-2*lambda*h_1)) "
            "concentrates at lambda -> 0. Tighten "
            "'epsabs'/'epsrel' only if the soil is inside the validated "
            "envelope rho_2/rho_1 in [1e-2, 1e2]; otherwise use "
            "backend='image_2layer', whose geometric series is exact "
            "for n = 2."
        )
    )
    return (
        head
        + " The returned value is NOT converged and, because the "
        "integrand is oscillatory, its SIGN may be wrong."
        + cause
    )


def _emit_convergence_warning(
    issues: list[_QuadratureIssue],
    *,
    epsabs: float,
    epsrel: float,
    max_osc_panels: int,
    stacklevel: int = 3,
) -> None:
    """Emit one aggregated warning for a batch of non-converged pairs.

    The reaction-matrix assembly evaluates $O(N^2)$ kernels; warning
    per pair would produce thousands of distinct messages (the default
    warning filter deduplicates by message *text*, and the text embeds
    the geometry). The worst offender is reported once, with the number
    of affected entries.
    """
    if not issues:
        return
    worst = max(issues, key=lambda i: i.achieved - i.tol)
    warnings.warn(
        _format_convergence_warning(
            worst,
            epsabs=epsabs,
            epsrel=epsrel,
            max_osc_panels=max_osc_panels,
            n_affected=len(issues),
        ),
        SommerfeldConvergenceWarning,
        stacklevel=stacklevel,
    )


# ---------------------------------------------------------------------
# Panel quadrature of the (decaying) remainder integrand
# ---------------------------------------------------------------------


def _panel_edges(
    lam_max: float,
    s: float,
    lam_osc: float,
    refine: int,
    *,
    lam_pole: float = 0.0,
    max_osc_panels: int = _MAX_OSC_PANELS,
) -> _PanelGrid:
    """Panel edges for the remainder quadrature on $[0, \\lambda_{\\max}]$.

    Three length scales have to be resolved simultaneously:

    * the **exponential decay** of the remainder (scale
      $1 / d_{\\text{rem}}$), captured by a quadratically graded
      base grid — fine near $\\lambda = 0$ where the fast
      exponentials live, coarse in the tail;
    * the **multiple-reflection pole** of
      $1/(1 - \\Gamma_1 e^{-2\\lambda h_1})$ at
      $\\lambda \\sim (1 - \\Gamma_1(0)) / 2 h_1$, captured by a
      geometric (log-spaced) family between
      ``_POLE_MARGIN · lam_pole`` and the innermost base panel. For
      $|K_1| \\to 1$ that scale collapses towards $\\lambda = 0$
      and the quadratic grading misses it completely — the source of
      the historic $-12.9\\,\\%$ error at $\\rho_2/\\rho_1 = 10^8$;
    * the **oscillation** of $J_0(\\lambda s)$ (period
      $2 \\pi / s$), captured by extra edges at half-period
      spacing $\\pi / s$ up to ``lam_osc``, beyond which the
      envelope is numerically negligible.

    Refinement policy for the oscillation family (0.15.0)
    -----------------------------------------------------
    Up to 0.15.0 the half-period step was ``π / (s · refine)`` and the
    resulting panel count was *truncated* at ``max_osc_panels``, so
    once the budget bound, the covered range
    ``n_osc · step = max_osc_panels · π / (s · refine)`` **shrank** by
    the refinement factor: the later rungs of the refinement ladder
    covered *less* than the earlier ones, and because
    :func:`_reflected_integral` returns the last value, more refinement
    produced a worse answer (measured: a sign-flipped Green's function,
    $-9.47 \\cdot 10^{-5}$ instead of $+5 \\cdot 10^{-4}$).

    The budget is now spent coverage-first:

    * as long as full coverage fits, the *refinement* is reduced to the
      largest multiplier the budget allows (coverage stays complete);
    * if not even one panel per half period fits, the step stays at the
      unrefined ``π / s`` and the covered range becomes
      ``max_osc_panels · π / s`` — **independent of** ``refine``.

    The covered range is therefore monotonically non-decreasing along
    :data:`_REFINEMENTS`, and the returned :class:`_PanelGrid` reports
    what was achieved so the caller can warn instead of returning a
    silently truncated value.

    Parameters
    ----------
    lam_max
        Upper end of the integration range.
    s
        Cylindrical radius; ``s = 0`` switches the oscillation grid
        off ($J_0 \\equiv 1$).
    lam_osc
        Upper end of the oscillation-resolved region.
    refine
        Refinement multiplier (panel count scales linearly with it).
    lam_pole
        $\\lambda$-scale of the multiple-reflection multiplier,
        $(1 - \\Gamma_1(0)) / 2 h_1$. ``0`` switches the geometric
        family off.
    max_osc_panels
        Budget for the half-period panel family.

    Returns
    -------
    grid : _PanelGrid
        ``grid.edges`` is strictly increasing with ``edges[0] == 0.0``
        and ``edges[-1] == lam_max``; the remaining fields describe how
        much of the oscillatory region is resolved.
    """
    n_dec = _N_DECAY_PANELS * refine
    t = np.arange(n_dec + 1, dtype=float) / n_dec
    edges = lam_max * t ** 2

    # Geometric family for the multiple-reflection pole.
    lam_first = lam_max / float(n_dec) ** 2
    lam_lo = max(_POLE_MARGIN * float(lam_pole), lam_max * 1e-15)
    if 0.0 < lam_lo < lam_first:
        n_decades = math.log10(lam_first / lam_lo)
        n_pole = max(1, int(math.ceil(n_decades * _POLE_PANELS_PER_DECADE * refine)))
        edges = np.concatenate(
            [edges, np.geomspace(lam_lo, lam_first, n_pole + 1)]
        )

    lam_required = min(lam_osc, lam_max) if (s > 0.0 and lam_osc > 0.0) else 0.0
    lam_covered = lam_required
    n_half = 0
    if lam_required > 0.0:
        step_1 = np.pi / s
        # Half-period panels needed for full coverage at refine == 1.
        n_half = int(np.floor(lam_required / step_1))
        if n_half >= 1:
            budget = max(1, int(max_osc_panels))
            if n_half <= budget:
                # Coverage fits: spend the rest of the budget on
                # refinement, never on shrinking the covered range.
                eff = max(1, min(refine, budget // n_half))
                n_osc = n_half * eff
                step = step_1 / eff
            else:
                # Budget binds even unrefined — keep the coarsest
                # (refine-independent) step so coverage cannot shrink.
                n_osc = budget
                step = step_1
                _log.debug(
                    "mom_sommerfeld: oscillation panels capped at %d of "
                    "%d needed (s=%.3g, lam_osc=%.3g)",
                    budget, n_half, s, lam_required,
                )
            lam_covered = min(n_osc * step, lam_required)
            edges = np.concatenate(
                [edges, step * np.arange(1, n_osc + 1, dtype=float)]
            )
    return _PanelGrid(
        edges=np.unique(np.clip(edges, 0.0, lam_max)),
        lam_covered=float(lam_covered),
        lam_required=float(lam_required),
        osc_panels_required=int(n_half),
    )


def _integrate_panels(func, edges: np.ndarray) -> float:
    """Composite fixed-order Gauss–Legendre integral of ``func``.

    ``func`` is called **once**, vectorised, with all
    ``(len(edges) - 1) × _GAUSS_ORDER`` abscissae.

    Parameters
    ----------
    func
        Vectorised integrand ``f(lam: np.ndarray) -> np.ndarray``.
    edges
        Panel edges from :func:`_panel_edges`.

    Returns
    -------
    value : float
    """
    a = edges[:-1]
    b = edges[1:]
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    lam = (mid[:, None] + half[:, None] * _GL_NODES[None, :]).ravel()
    vals = np.asarray(func(lam), dtype=float).reshape(a.size, _GAUSS_ORDER)
    return float(np.dot(half, vals @ _GL_WEIGHTS))


def _reflected_integral(
    stack: LayerStack,
    s: float,
    abs_dz: float,
    z_plus: float,
    *,
    lambda_max_factor: float,
    epsabs: float,
    epsrel: float,
    reference: float = 0.0,
    max_osc_panels: int = _MAX_OSC_PANELS,
    issues: list[_QuadratureIssue] | None = None,
) -> float:
    """Reflected (layered) part of the top-layer Sommerfeld kernel.

    Evaluates
    $$
    G_{\\text{refl}} = \\int_0^{\\infty}
      \\bigl[\\xi(\\lambda)
             - e^{-\\lambda d_z} - e^{-\\lambda p}\\bigr]
      J_0(\\lambda s)\\, d\\lambda ,
    $$
    where $d_z = |z - z_s|$, $p = z + z_s$ and
    $\\xi$ is the full top-layer spectral kernel
    $$
    \\xi(\\lambda) = \\frac{e^{-\\lambda d_z}
        + \\Gamma_1 e^{-\\lambda (2 h_1 - d_z)}
        + e^{-\\lambda p}
        + \\Gamma_1 e^{-\\lambda (2 h_1 - p)}}
        {1 - \\Gamma_1(\\lambda)\\, e^{-2 \\lambda h_1}} .
    $$
    Subtracting the two $\\Gamma_1 \\to 0$ terms removes the
    conditionally convergent part of the integral; what is left is
    $$
    \\xi - e^{-\\lambda d_z} - e^{-\\lambda p}
    = \\frac{\\Gamma_1 \\bigl[e^{-\\lambda(2h_1 - d_z)}
             + e^{-\\lambda(2h_1 - p)}
             + e^{-2\\lambda h_1}(e^{-\\lambda d_z}
             + e^{-\\lambda p})\\bigr]}
            {1 - \\Gamma_1 e^{-2\\lambda h_1}},
    $$
    every term of which carries an exponential
    $e^{-\\lambda d}$ with
    $d \\ge \\min(2 h_1 - d_z,\\, 2 h_1 - p) > 0$ — except in the
    corner $p \\to 2 h_1$ (both points *at* the interface). That
    corner is handled by extracting one more term analytically, with
    $\\Gamma_1$ replaced by its high-$\\lambda$ limit
    $\\Gamma_1(\\infty) = K_1$ (exact for $n = 2$, where
    $\\Gamma_1 \\equiv K_1$):
    $$
    \\int_0^\\infty K_1 e^{-\\lambda (2h_1 - p)} J_0(\\lambda s)
    \\,d\\lambda = \\frac{K_1}{\\sqrt{s^2 + (2 h_1 - p)^2}} .
    $$
    The numerically integrated residual then decays on the scale
    $d_{\\text{rem}} = \\min(2 h_1 - d_z,\\;
    (2 h_1 - p) + \\Delta)$ with
    $\\Delta = 2 h_1$ for $n = 2$ and
    $\\Delta = 2 \\min(h_1, h_2)$ for $n \\ge 3$ — the scale on
    which $\\Gamma_1(\\lambda) - K_1$ and $e^{-2\\lambda h_1}$
    die out.

    Parameters
    ----------
    stack
        Layer stack, ``n_layers >= 2``.
    s
        Cylindrical radius in metres.
    abs_dz
        $|z - z_s|$ in metres.
    z_plus
        $z + z_s$ in metres; must not exceed $2 h_1$ (both
        points inside the upper layer).
    lambda_max_factor
        Truncation of the remainder quadrature in units of
        $1 / d_{\\text{rem}}$, saturating at
        ``_MAX_DECAY_EXPONENT``.
    epsabs, epsrel
        Convergence thresholds of the grid refinement.
    reference
        Magnitude the relative threshold is measured against
        (typically the analytic direct part). Passing the *total*
        kernel magnitude avoids demanding a relative accuracy of a
        remainder that is legitimately close to zero.
    max_osc_panels
        Budget for the half-period panel family that resolves
        $J_0(\\lambda s)$; see :func:`_panel_edges`. The budget binds
        for $s / d_{\\text{rem}} \\gtrsim 1.9 \\cdot 10^{4}$.
    issues
        Optional collector. When given, a failed convergence test
        appends a :class:`_QuadratureIssue` to it instead of warning
        immediately, so an $O(N^2)$ matrix assembly can emit a single
        aggregated :class:`SommerfeldConvergenceWarning`.

    Returns
    -------
    G_refl : float

    Warns
    -----
    SommerfeldConvergenceWarning
        If the refinement ladder is exhausted without meeting
        ``max(epsabs, epsrel · |G|)`` and no ``issues`` collector is
        supplied. Up to 0.15.0 this condition was only logged at
        ``DEBUG`` level.
    """
    h_top = float(stack.h[0])
    k_1 = float(stack.K[0])
    d_int = 2.0 * h_top - z_plus
    if d_int < 0.0:
        raise ValueError(
            "mom_sommerfeld: the top-layer Sommerfeld kernel requires "
            f"z + z_s <= 2·h_1 (got z + z_s = {z_plus:.4f} m, "
            f"h_1 = {h_top:.4f} m). Cross-layer geometries are only "
            "supported for n_layers == 2 (backend 'image_2layer')."
        )

    # Analytic term: leading interface reflection with Γ_1 → K_1.
    r_int = float(np.hypot(s, d_int))
    g_analytic = k_1 / max(r_int, _MIN_DISTANCE)
    if 0.0 < r_int < _MIN_DISTANCE:
        # The 1 mm distance floor (the documented convention shared with
        # the homogeneous ``image`` backend) is applied to the analytic
        # K_1 interface-image term but *not* to the matching
        # K_1·e^{-λ·d_int} subtraction in ``_residual`` below, so the
        # analytic split is no longer algebraically exact here and the
        # reflected part saturates hard: at s = 0, d_int = 1e-5 m the
        # result is 667.561 against an exact reflected series of
        # 33334.23 (-98 %); at d_int = 2 mm it is exact again. Only
        # reachable through the n >= 3 diagonal for a segment midpoint
        # within 0.5 mm of the interface. No warning category is raised
        # (this is the distance-floor convention, not a defect), but the
        # saturation is logged so it can be found in a solve log.
        _log.warning(
            "mom_sommerfeld: interface-image distance "
            "sqrt(s^2 + (2*h_1 - z - z_s)^2) = %.3e m is below the "
            "%.0e m distance floor (s=%.3g, z+z_s=%.6g, h_1=%.6g). The "
            "reflected kernel is saturated by the floor and can be one "
            "to two orders of magnitude low. Keep segment midpoints a "
            "few millimetres clear of the layer interface.",
            r_int, _MIN_DISTANCE, s, z_plus, h_top,
        )

    if k_1 == 0.0 and stack.n_layers == 2:
        # Γ_1 ≡ 0: the reflected part vanishes identically. Keeping
        # this exact makes the ρ_1 == ρ_2 limit bit-exact.
        return 0.0

    def _residual(lam: np.ndarray) -> np.ndarray:
        gamma = reflection_gamma(stack, lam)
        e_2h = np.exp(-2.0 * lam * h_top)
        denom = 1.0 - gamma * e_2h
        num = gamma * (
            np.exp(-lam * (2.0 * h_top - abs_dz))
            + np.exp(-lam * d_int)
            + e_2h * (np.exp(-lam * abs_dz) + np.exp(-lam * z_plus))
        )
        out = num / denom - k_1 * np.exp(-lam * d_int)
        if s > 0.0:
            out = out * j0(lam * s)
        return out

    # Decay scale of the residual, see the formula in the docstring.
    if stack.n_layers >= 3:
        delta = 2.0 * min(h_top, float(stack.h[1]))
    else:
        delta = 2.0 * h_top
    d_rem = min(2.0 * h_top - abs_dz, d_int + delta)
    d_rem = max(d_rem, _MIN_DISTANCE)

    exponent = min(float(lambda_max_factor), _MAX_DECAY_EXPONENT)
    lam_max = exponent / d_rem
    lam_osc = min(lam_max, _OSC_DECAY_EXPONENT / d_rem)

    # λ-scale of the multiple-reflection multiplier 1/(1 - Γ_1 e^{-2λh}).
    # Γ_1(0) is the DC reflection of the whole stack below the interface;
    # for |Γ_1(0)| → 1 the multiplier concentrates at λ → 0 and needs a
    # geometric panel family (see :func:`_panel_edges`).
    gamma_0 = float(np.atleast_1d(reflection_gamma(stack, np.array([0.0])))[0])
    lam_pole = max(1.0 - gamma_0, 1e-16) / (2.0 * h_top)

    value = 0.0
    previous: float | None = None
    grid = _PanelGrid(np.zeros(2), 0.0, 0.0, 0)
    achieved = math.inf
    tol = math.inf
    for refine in _REFINEMENTS:
        grid = _panel_edges(
            lam_max, s, lam_osc, refine,
            lam_pole=lam_pole, max_osc_panels=max_osc_panels,
        )
        value = _integrate_panels(_residual, grid.edges)
        if previous is not None:
            tol = max(
                epsabs,
                epsrel * max(abs(reference), abs(value + g_analytic)),
            )
            achieved = abs(value - previous)
            if achieved <= tol:
                break
        previous = value
    else:
        _log.debug(
            "mom_sommerfeld: remainder quadrature not converged to "
            "epsrel=%.1e (s=%.3g, |dz|=%.3g, z+z_s=%.3g, "
            "achieved=%.3e > tol=%.3e, lam_covered=%.4g of %.4g)",
            epsrel, s, abs_dz, z_plus, achieved, tol,
            grid.lam_covered, grid.lam_required,
        )
        issue = _QuadratureIssue(
            achieved=float(achieved),
            tol=float(tol),
            s=float(s),
            abs_dz=float(abs_dz),
            z_plus=float(z_plus),
            grid=grid,
        )
        if issues is None:
            _emit_convergence_warning(
                [issue],
                epsabs=epsabs, epsrel=epsrel,
                max_osc_panels=max_osc_panels,
                stacklevel=4,
            )
        else:
            issues.append(issue)

    return g_analytic + value


# ---------------------------------------------------------------------
# Pointwise evaluation of the Sommerfeld kernel
# ---------------------------------------------------------------------


def sommerfeld_kernel_value(
    stack: LayerStack,
    s: float,
    z: float,
    z_s: float,
    *,
    lambda_max_factor: float = 200.0,
    epsabs: float = 1e-9,
    epsrel: float = 1e-7,
    max_osc_panels: int = _MAX_OSC_PANELS,
    issues: list[_QuadratureIssue] | None = None,
) -> float:
    """Evaluate the layered-soil Sommerfeld kernel at one point.

    Returns the full top-layer kernel
    $$
    G(s, z, z_s) \\;=\\; \\int_0^{\\infty}
    \\frac{e^{-\\lambda |z - z_s|}
         + \\Gamma_1 e^{-\\lambda (2 h_1 - |z - z_s|)}
         + e^{-\\lambda (z + z_s)}
         + \\Gamma_1 e^{-\\lambda (2 h_1 - z - z_s)}}
         {1 - \\Gamma_1(\\lambda)\\, e^{-2 \\lambda h_1}}
    J_0(\\lambda s)\\, d\\lambda ,
    $$
    i.e. both image families plus the full multiple-reflection
    multiplier between the free surface ($R_{\\text{air}} = +1$)
    and the layer interface. The potential of a point source $I$
    is then ``ρ_1 · I · G / (4 π)``.

    The integral is **not** evaluated by a bare real-axis quadrature:
    the two $\\Gamma_1 \\to 0$ terms (which converge only through
    the oscillation of $J_0$ and are the reason the pre-0.15.0
    implementation could return a negative value) are taken from
    Lipschitz' integral
    $\\int_0^\\infty e^{-\\lambda d} J_0(\\lambda s) d\\lambda
    = (s^2 + d^2)^{-1/2}$, so that
    $$
    G = \\frac{1}{\\sqrt{s^2 + (z - z_s)^2}}
      + \\frac{1}{\\sqrt{s^2 + (z + z_s)^2}}
      + G_{\\text{refl}},
    $$
    and only the exponentially decaying reflected remainder
    $G_{\\text{refl}}$ is integrated numerically — see
    :func:`_reflected_integral`.

    Parameters
    ----------
    stack
        Layer stack.
    s
        Cylindrical radius $s = \\sqrt{(x - x_s)^2 + (y - y_s)^2}$
        in metres.
    z, z_s
        Field-point depth and source depth (positive into the soil).
    lambda_max_factor
        Truncation of the *remainder* quadrature, in units of
        $1 / d_{\\text{rem}}$, where $d_{\\text{rem}}$ is the
        decay length of the remainder (not, as before 0.15.0, a
        characteristic length of the geometry). The discarded tail is
        bounded by $e^{-\\texttt{lambda\\_max\\_factor}}$; the factor
        saturates at 60 ($\\sim 10^{-26}$), so raising it beyond
        the default cannot change the result.

        .. warning::
           **The same keyword has a different meaning on the
           delegated $n = 2$ cross-layer path.** When
           ``stack.n_layers == 2`` and $z > h_1$ or $z_s > h_1$,
           the call is forwarded to
           :func:`groundfield.coupling.layered_green.two_layer_real_space_kernel`,
           where ``lambda_max_factor`` is measured in units of
           $1 / \\ell_{\\text{char}}$ with
           $\\ell_{\\text{char}}$ a *geometric* characteristic
           length (``max(h_1, s, |z ± z_s|)``-like), is **not**
           saturated at 60, and where *raising* it degrades the
           result: the Hankel grid there has to resolve
           $J_0(\\lambda s)$ up to $\\lambda_{\\max}$, so a large
           factor exhausts the node budget and triggers
           :class:`groundfield.coupling.layered_green.SommerfeldResolutionWarning`.
           On that path the documented remedy is to *lower* the
           factor; on the top-layer path here it is harmless to raise
           it. The two conventions are deliberately left unmerged —
           the delegated backend is shared with the inductive-coupling
           code and its calibration is validated separately — but the
           unit difference is a real trap and is repeated at the
           delegation site in the source.
    epsabs, epsrel
        Absolute / relative thresholds of the grid-refinement
        convergence test of the remainder quadrature. ``epsrel`` is
        measured against the magnitude of the *whole* kernel. They are
        **not** forwarded to the delegated $n = 2$ cross-layer path,
        which carries its own fixed-order grid.
    max_osc_panels
        Budget for the half-period panel family that resolves the
        $J_0(\\lambda s)$ oscillation, default 200 000 panels
        (× 12 Gauss nodes). Only binds for
        $s / d_{\\text{rem}} \\gtrsim 1.9 \\cdot 10^{4}$, i.e. a
        centimetre-thin intermediate layer combined with a
        kilometre-scale horizontal separation; there the coverage is
        incomplete, a :class:`SommerfeldConvergenceWarning` is raised
        and raising this budget is the remedy (cost grows linearly).
    issues
        Optional collector for non-convergence diagnostics; see
        :func:`_reflected_integral`. Used by
        :func:`_build_Z_sommerfeld` to aggregate $O(N^2)$ warnings
        into one.

    Returns
    -------
    G : float

    Raises
    ------
    ValueError
        For $n \\ge 3$ with $z > h_1$ or $z_s > h_1$: the
        expression above is a top-layer form, so the kernel refuses
        instead of silently returning a diverging number (before
        0.15.0 it returned $\\approx -2 \\cdot 10^{127}$ there).

    Warns
    -----
    SommerfeldConvergenceWarning
        If the remainder quadrature does not reach the requested
        tolerance. The returned value is then not converged and its
        sign may be wrong.

    Notes
    -----
    Limits and special cases:

    - ``stack.n_layers == 1`` (homogeneous, $\\Gamma_1 \\equiv 0$):
      short-circuited to $1/r + 1/r_{\\text{img}}$, bit-exact.
    - $\\rho_1 = \\rho_2$ in a declared 2-layer stack
      ($K_1 = 0$, hence $\\Gamma_1 \\equiv 0$): the reflected
      remainder is identically zero, so the same closed form is
      returned bit-exact. This is the strongest available accuracy
      test of the backend.
    - ``stack.n_layers == 2`` with $z > h_1$ or $z_s > h_1$:
      delegated to
      :func:`groundfield.coupling.layered_green.two_layer_real_space_kernel`,
      which solves the cross-layer spectral matching.

    Distances are floored at 1 mm (``_MIN_DISTANCE``), consistently
    with the homogeneous ``image`` backend, so the coincident-point
    limit stays finite.
    """
    # Homogeneous shortcut.
    if stack.n_layers <= 1:
        r = np.sqrt(s ** 2 + (z - z_s) ** 2)
        r_img = np.sqrt(s ** 2 + (z + z_s) ** 2)
        return float(1.0 / max(r, _MIN_DISTANCE) + 1.0 / max(r_img, _MIN_DISTANCE))

    # ADR-0006 Phase B: 2-layer soil with cross-layer source/observer.
    # Delegate to coupling.layered_green which solves the full 2-layer
    # spectral matching for any (z_layer, z_s_layer) combination. The
    # caller multiplies by rho_1/(4π); the layered_green kernel
    # internally carries the source-layer rho factor, so we divide
    # out rho_1 here to match the existing convention.
    if stack.n_layers == 2 and (z > stack.h[0] or z_s > stack.h[0]):
        from groundfield.coupling.layered_green import (
            two_layer_real_space_kernel,
        )

        # NOTE (unit trap): ``lambda_max_factor`` changes meaning here.
        # In the top-layer path below it is measured in units of
        # 1/d_rem (the *remainder* decay length) and saturates at 60.
        # ``two_layer_real_space_kernel`` measures it in units of
        # 1/char_length (a geometric length) and does **not** saturate
        # it; there a large value exhausts the Hankel node budget and
        # raises SommerfeldResolutionWarning, so the remedy on that
        # path is to *lower* the factor. The value is forwarded
        # verbatim (the historic behaviour) rather than rescaled,
        # because the delegated kernel is calibrated and validated
        # against its own reference; see the ``lambda_max_factor``
        # entry in this function's docstring.
        rho_1_local = float(stack.rhos[0])
        rho_2_local = float(stack.rhos[1])
        h_1_local = float(stack.h[0])
        G_phys = two_layer_real_space_kernel(
            s=s, z=z, z_s=z_s,
            rho_1=rho_1_local, rho_2=rho_2_local, h_1=h_1_local,
            lambda_max_factor=lambda_max_factor,
        )
        # G_phys has rho/2·(1/r + ...) structure; the existing caller
        # forms Z = rho_1/(4π)·G_old, so we want G_old = 2·G_phys/rho_1.
        return float(2.0 * G_phys / rho_1_local)

    # Layered case — full top-layer Sommerfeld form.
    #
    # The complete Green's function of a top-layer source observed in the
    # top layer combines four exponentials, each propagated through the
    # multiple-reflection multiplier 1/(1 - Γ_1(λ)·e^{-2λh_1}):
    #
    #   ξ(λ) = (e^{-λ|z-z_s|} + Γ_1·e^{-λ(2h_1-|z-z_s|)}
    #         + e^{-λ(z+z_s)}  + Γ_1·e^{-λ(2h_1-z-z_s)})
    #          / (1 - Γ_1(λ)·e^{-2λh_1})
    #
    # In the limits Γ_1 → 0 this reduces to the homogeneous form
    # e^{-λ|z-z_s|} + e^{-λ(z+z_s)}; for Γ_1 = K_1 = const it expands
    # into the classical Tagg/Sunde geometric series in K_1^n at images
    # ±2nh_1 ± z_s.
    #
    # The two Γ_1 → 0 terms are the conditionally convergent part of
    # the integral and are taken from Lipschitz' integral in closed
    # form; only the decaying reflected remainder is quadratured.
    h_top = float(stack.h[0])
    z_tol = 1e-9 * max(h_top, 1.0)
    if z > h_top + z_tol or z_s > h_top + z_tol:
        # Only reachable for n >= 3 (n == 2 is handled above). The
        # top-layer form is not valid below the interface; before
        # 0.15.0 it silently returned a diverging number (≈ -2e127).
        raise ValueError(
            "mom_sommerfeld: the n>=3 top-layer Sommerfeld kernel "
            "requires both the field point and the source inside the "
            f"upper layer (got z = {z:.4f} m, z_s = {z_s:.4f} m, "
            f"h_1 = {h_top:.4f} m). Use backend='image_2layer' for "
            "cross-layer geometries on a 2-layer soil."
        )
    abs_dz = abs(z - z_s)
    z_plus = z + z_s
    r = float(np.hypot(s, abs_dz))
    r_img = float(np.hypot(s, z_plus))
    G_direct = 1.0 / max(r, _MIN_DISTANCE) + 1.0 / max(r_img, _MIN_DISTANCE)

    G_refl = _reflected_integral(
        stack, s, abs_dz, z_plus,
        lambda_max_factor=lambda_max_factor,
        epsabs=epsabs, epsrel=epsrel,
        reference=G_direct,
        max_osc_panels=max_osc_panels,
        issues=issues,
    )
    return float(G_direct + G_refl)


def _build_Z_sommerfeld(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    stack: LayerStack,
    *,
    lambda_max_factor: float,
    epsabs: float,
    epsrel: float,
    max_osc_panels: int = _MAX_OSC_PANELS,
    max_terms: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    """N×N Sommerfeld reaction matrix.

    Off-diagonal entries: pointwise Sommerfeld evaluation through
    :func:`sommerfeld_kernel_value`. Diagonal entries: line
    self-potential of the homogeneous bulk plus the layered-soil
    correction. For ``n_layers == 2`` we take the closed-form
    Tagg/Sunde self-kernel as the diagonal source (consistent with
    the off-diagonal Sommerfeld integral, which evaluates the same
    physics by direct quadrature). For ``n_layers >= 3`` we add the
    *reflected remainder* of the analytic split
    (:func:`_reflected_integral`) evaluated at ``s = 0, z = z_s``:
    the direct $1/r$ singularity lives entirely in the analytic
    part that the homogeneous line self-potential already accounts
    for, so the remainder is regular point-on-source.

    Non-convergence of any single pair is collected and reported as
    **one** aggregated :class:`SommerfeldConvergenceWarning` naming the
    worst offender and the number of affected entries — an $O(N^2)$
    loop that warned per pair would emit thousands of distinct messages.
    """
    n = seg_points.shape[0]
    rho_1 = float(stack.rhos[0])

    if stack.n_layers <= 1:
        eye = np.eye(n)
        return _self_corrected_kernel(
            seg_points, seg_lengths, wire_radii, eye, rho_1
        )

    Z = np.zeros((n, n), dtype=float)
    issues: list[_QuadratureIssue] = []

    # Off-diagonal: pointwise Sommerfeld evaluation.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = seg_points[i, 0] - seg_points[j, 0]
            dy = seg_points[i, 1] - seg_points[j, 1]
            s_ij = float(np.sqrt(dx * dx + dy * dy))
            G = sommerfeld_kernel_value(
                stack,
                s=s_ij,
                z=float(seg_points[i, 2]),
                z_s=float(seg_points[j, 2]),
                lambda_max_factor=lambda_max_factor,
                epsabs=epsabs,
                epsrel=epsrel,
                max_osc_panels=max_osc_panels,
                issues=issues,
            )
            Z[i, j] = rho_1 / (4.0 * np.pi) * G

    # Diagonal: layered self-potential.
    if stack.n_layers == 2:
        # Closed-form Tagg/Sunde self-kernel — consistent with the
        # off-diagonal Sommerfeld integral.
        from groundfield.soil.models import TwoLayerSoil
        from groundfield.solver.image_2layer import (
            _two_layer_self_kernel_factory,
        )
        soil = TwoLayerSoil(
            rho_1=float(stack.rhos[0]),
            rho_2=float(stack.rhos[1]),
            h_1=float(stack.h[0]),
        )
        # allow_cross_layer=True keeps the diagonal consistent with
        # the off-diagonal Sommerfeld integral for interface-crossing
        # geometries (previously the diagonal silently fell back to
        # the upper-layer series).
        self_kern = _two_layer_self_kernel_factory(
            soil, max_terms=max_terms, tol=tol, allow_cross_layer=True,
        )
        eye = np.eye(n)
        Z_layered_diag = self_kern(seg_points, seg_lengths, wire_radii, eye)
        np.fill_diagonal(Z, np.diag(Z_layered_diag))
        _emit_convergence_warning(
            issues, epsabs=epsabs, epsrel=epsrel,
            max_osc_panels=max_osc_panels, stacklevel=4,
        )
        return Z

    # n >= 3: homogeneous line self-potential plus a layered
    # reflection-only correction. The reflection-only kernel is exactly
    # the remainder of the analytic split (the direct 1/r singularity
    # sits entirely in the subtracted homogeneous part), so it can be
    # evaluated at s = 0, |z - z_s| = 0 — where the physical decay
    # scale is 2·(h_1 - z_i), *not* min(h_1, z_i) as assumed before
    # 0.15.0.
    eye = np.eye(n)
    Z_homog = _self_corrected_kernel(seg_points, seg_lengths, wire_radii, eye, rho_1)
    for i in range(n):
        z_i = float(seg_points[i, 2])
        refl = _reflected_integral(
            stack, 0.0, 0.0, 2.0 * z_i,
            lambda_max_factor=lambda_max_factor,
            epsabs=epsabs, epsrel=epsrel,
            reference=4.0 * np.pi * Z_homog[i, i] / rho_1,
            max_osc_panels=max_osc_panels,
            issues=issues,
        )
        layered_offset = rho_1 / (4.0 * np.pi) * refl
        Z[i, i] = Z_homog[i, i] + layered_offset
    _emit_convergence_warning(
        issues, epsabs=epsabs, epsrel=epsrel,
        max_osc_panels=max_osc_panels, stacklevel=4,
    )
    return Z


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_mom_sommerfeld(
    world: "World",
    engine: "Engine",
    *,
    lambda_max_factor: float = 200.0,
    epsabs: float = 1e-9,
    epsrel: float = 1e-7,
    max_osc_panels: int = _MAX_OSC_PANELS,
) -> FieldResult:
    """Galerkin MoM with direct Sommerfeld quadrature for layered soil.

    This is a methodologically independent backend used as the
    reference inside :func:`groundfield.compare_engines` for layered
    worlds with hard contrasts. Every off-diagonal reaction-matrix
    entry is one evaluation of :func:`sommerfeld_kernel_value`, i.e.
    the closed-form direct pair plus a numerically integrated
    reflected remainder; no image series and no fit enter the kernel.

    Parameters
    ----------
    world
        World to evaluate.
    engine
        Engine configuration.
    lambda_max_factor
        Truncation of the remainder quadrature in units of the
        remainder decay length $1 / d_{\\text{rem}}$; saturates at
        60. See :func:`sommerfeld_kernel_value` — note the unit change
        on the delegated $n = 2$ cross-layer path documented there.
    epsabs, epsrel
        Convergence thresholds of the remainder quadrature.
    max_osc_panels
        Budget for the $J_0$ half-period panel family.

    Returns
    -------
    FieldResult

    Warns
    -----
    SommerfeldConvergenceWarning
        Once per solve, if any reaction-matrix entry's remainder
        quadrature failed to converge.

    Notes
    -----
    The four accuracy keywords are wired to the
    :class:`~groundfield.solver.engine.Engine` fields
    ``sommerfeld_lambda_max_factor``, ``sommerfeld_epsabs``,
    ``sommerfeld_epsrel`` and ``sommerfeld_max_osc_panels``. The
    signature defaults are kept identical to those field defaults, so a
    direct call without keywords behaves exactly like
    :meth:`~groundfield.solver.engine.Engine.solve`. Up to 0.14.1
    ``Engine.solve`` called this function with no keywords at all, so
    none of the knobs was reachable through the public API.
    """
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'mom_sommerfeld' supports HomogeneousSoil, "
            "TwoLayerSoil, and MultiLayerSoil. "
            f"Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")
    _reject_concrete_shells(world, "mom_sommerfeld")
    _warn_ignored_sources(world, "mom_sommerfeld")

    stack = as_layer_stack(world.soil)
    ds = engine.segment_length
    _log.info(
        "mom_sommerfeld: n_layers=%d, segment_length=%.3f, lam_max_fac=%.1f",
        stack.n_layers, ds, lambda_max_factor,
    )

    # 1) Discretisation.
    all_segments: list[_Segment] = []
    elec_to_segidx: dict[str, list[int]] = {}
    interfaces = (
        (float(stack.h[0]),) if stack.n_layers >= 2 else None
    )
    for e in world.electrodes:
        segs = _discretize_electrode(e, ds, layer_interfaces=interfaces)
        elec_to_segidx[e.name] = list(
            range(len(all_segments), len(all_segments) + len(segs))
        )
        all_segments.extend(segs)

    # 2) Per-electrode input currents.
    elec_input_current: dict[str, complex] = {
        e.name: 0j for e in world.electrodes
    }
    for src in world.sources:
        if src.kind != "current":
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in elec_input_current:
            elec_input_current[src.attached_to] += i_complex

    cluster_id = _build_clusters(world.electrodes, world.conductors)
    finite_branches = _build_finite_branches(world.conductors, cluster_id)

    # 2b) Distributed-conductor topology (ADR-0003) + ADR-0004
    #     inductive coupling.
    cond_segs, distributed_branches_objs, interior_nodes = _build_distributed_topology(
        world.conductors, cluster_id
    )
    for s in cond_segs:
        pn = s.electrode_name
        elec_to_segidx[pn] = [len(all_segments)]
        all_segments.append(s)
        cluster_id[pn] = pn
    for n_ in interior_nodes:
        if n_ not in cluster_id:
            cluster_id[n_] = n_
            elec_to_segidx[n_] = []
    n_lumped_branches = len(finite_branches)
    distributed_branch_tuples = [
        (db.node_a, db.node_b, db.R) for db in distributed_branches_objs
    ]
    finite_branches = list(finite_branches) + distributed_branch_tuples
    earth_inductive_model = getattr(
        engine, "earth_inductive_model", "perfect_mirror"
    )
    sigma_earth_for_carson: float | None = None
    layered_earth_for_sommerfeld: object = None
    if earth_inductive_model == "carson_series":
        from groundfield.coupling import resolve_earth_conductivity

        sigma_earth_for_carson = resolve_earth_conductivity(world.soil)
    elif earth_inductive_model == "sommerfeld":
        from groundfield.coupling import resolve_earth_layers

        layered_earth_for_sommerfeld = resolve_earth_layers(world.soil)
    inductance_matrix_full, has_inductance, carson_builder = _assemble_inductance_matrix(
        distributed_branches_objs,
        n_lumped_branches=n_lumped_branches,
        n_total_branches=len(finite_branches),
        earth_model=earth_inductive_model,
        sigma_earth=sigma_earth_for_carson,
        layered_earth=layered_earth_for_sommerfeld,
    )

    n_segments = len(all_segments)
    seg_points = np.array([s.midpoint for s in all_segments])
    seg_lengths = np.array([s.length for s in all_segments])
    wire_radii = np.array([s.wire_radius for s in all_segments])

    if stack.n_layers >= 3:
        z_max = seg_points[:, 2].max()
        h_1 = float(stack.h[0])
        if z_max >= h_1:
            # ADR-0007 Phase B (n≥3): not implemented — the
            # reflection-only diagonal correction below assumes all
            # segments in the top layer. Hard error instead of a
            # warning followed by wrong physics.
            raise ValueError(
                f"mom_sommerfeld: cross-layer geometry on "
                f"n_layers={stack.n_layers} is not supported "
                f"(z_max = {z_max:.3f} m >= h_1 = {h_1:.3f} m). "
                "Use backend='image_2layer' for n=2 cross-layer "
                "worlds; for n>=3 thicken the upper layer."
            )

    # 3) Z-matrix via direct Sommerfeld quadrature (series knobs for
    #    the n=2 diagonal from the engine).
    Z = _build_Z_sommerfeld(
        seg_points, seg_lengths, wire_radii, stack,
        lambda_max_factor=lambda_max_factor,
        epsabs=epsabs, epsrel=epsrel,
        max_osc_panels=max_osc_panels,
        max_terms=engine.image_max_terms, tol=engine.image_series_tol,
    )

    # 4) Frequency loop (Galerkin solve + Z · I_seg for phi).
    n_freq = len(engine.frequencies)
    omegas = [2.0 * np.pi * float(f) for f in engine.frequencies]
    real_electrode_names = {e.name for e in world.electrodes}

    def _solve_at(omega: float) -> tuple[np.ndarray, np.ndarray]:
        carson_dz = (
            carson_builder(omega) if (has_inductance and carson_builder is not None)
            else None
        )
        sc, _ = _galerkin_solve(
            Z=Z,
            elec_input_current=elec_input_current,
            cluster_id=cluster_id,
            elec_to_segidx=elec_to_segidx,
            n_segments=n_segments,
            finite_branches=finite_branches,
            omega=omega if has_inductance else 0.0,
            inductance_matrix=inductance_matrix_full if has_inductance else None,
            carson_correction=carson_dz,
        )
        ph = np.zeros(n_segments, dtype=complex)
        if sc.any():
            ph = Z @ sc.real + 1j * (Z @ sc.imag)
        return sc, ph

    sc_per_freq: list[np.ndarray] = []
    phi_per_freq: list[np.ndarray] = []
    if has_inductance:
        for omega in omegas:
            sc, ph = _solve_at(omega)
            sc_per_freq.append(sc)
            phi_per_freq.append(ph)
    else:
        sc, ph = _solve_at(0.0)
        sc_per_freq = [sc] * n_freq
        phi_per_freq = [ph] * n_freq

    electrode_potentials: dict[str, list[complex]] = {}
    electrode_currents: dict[str, list[complex]] = {}
    conductor_currents: dict[str, list[complex]] = {}
    conductor_potentials: dict[str, list[complex]] = {}
    for ename, idxs in elec_to_segidx.items():
        if not idxs:
            continue
        u_list = [
            complex(np.mean(phi_per_freq[k][idxs])) for k in range(n_freq)
        ]
        i_list = [
            complex(sc_per_freq[k][idxs].sum()) for k in range(n_freq)
        ]
        if ename in real_electrode_names:
            electrode_potentials[ename] = u_list
            electrode_currents[ename] = i_list
        else:
            conductor_potentials[ename] = u_list
            conductor_currents[ename] = i_list

    point_sources = [
        PointSource(
            position=tuple(seg_points[i].tolist()),
            current=[complex(sc_per_freq[k][i]) for k in range(n_freq)],
            electrode_name=all_segments[i].electrode_name,
            length=float(seg_lengths[i]),
        )
        for i in range(n_segments)
    ]
    cluster_members: dict[str, list[str]] = {}
    for ename in real_electrode_names:
        cluster_members[ename] = sorted(
            n for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_electrode_names
        )

    metadata = {
        "world_name": world.name,
        "n_segments": n_segments,
        "segment_length": ds,
        "n_layers": int(stack.n_layers),
        "rhos": stack.rhos.tolist(),
        "h": stack.h.tolist(),
        "lambda_max_factor": float(lambda_max_factor),
        "epsabs": float(epsabs),
        "epsrel": float(epsrel),
        "max_osc_panels": int(max_osc_panels),
        "solver": "galerkin",
        "stub": False,
        "earth_inductive_model": earth_inductive_model,
    }
    if has_inductance:
        from groundfield.coupling.carson import skin_depth

        sigma_ref = (
            sigma_earth_for_carson
            if sigma_earth_for_carson is not None
            else 1.0 / float(stack.rhos[0])
        )
        metadata["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma_ref)
            for f in engine.frequencies
        }
    if conductor_currents:
        metadata["conductor_node_currents"] = conductor_currents
        metadata["conductor_node_potentials"] = conductor_potentials

    return FieldResult(
        backend="mom_sommerfeld",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=float(stack.rhos[0]),
        soil=world.soil,
        clusters=cluster_members,
        metadata=metadata,
    )
