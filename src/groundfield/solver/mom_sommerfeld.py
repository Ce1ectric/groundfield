"""Point-matching method of moments with **direct Sommerfeld quadrature** (``mom_sommerfeld``).

Mathematical / physical model
-----------------------------
The other layered backends (``image_2layer``, ``mom``, ``cim``,
``bem``) rely on the *closed-form* Tagg/Sunde image series of the
two-layer Green's function. This backend instead evaluates
the Sommerfeld integral
$$
\\varphi(s, z; z_s) \\;=\\; \\frac{\\rho_1\\, I}{4\\pi}
\\!\\int_0^{\\infty}\\! \\bigl[ e^{-\\lambda |z - z_s|}
    + \\Gamma_1(\\lambda)\\, e^{-\\lambda (z + z_s)}\\bigr]
J_0(\\lambda s)\\, d\\lambda
$$
**numerically**, point by point, with adaptive Gauss–Kronrod
quadrature (see below). The recursive
$\\Gamma_1(\\lambda)$ from
:func:`groundfield.solver._layered.reflection_gamma` is used as is —
no expansion, no fit. The *kernel* is therefore independent of the
series expansion of the other backends and is the only kernel for
$n \\ge 3$ distinct layers. The *discretisation* is not independent:
it is the point-matching scheme of :mod:`groundfield.solver.mom`, and
for $n = 2$ the diagonal of the reaction matrix is ``mom``'s series
diagonal. Agreement between ``mom_sommerfeld`` and ``mom`` on two-layer
soil is a kernel test, not an independent validation of the scheme
(review pass 10, 10.31).

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

The MoM resolution itself (point matching at the segment midpoints
with a point-source near field, one constraint per node, source-free
clusters floating) is that of :mod:`groundfield.solver.mom`; only the
off-diagonal kernel evaluation differs. Its accuracy limits (about 1 %
to 2 % below the exact thin-wire solution, drifting under refinement)
are those documented there.

Validity
--------
- Quasi-static, $f < 1\\,\\mathrm{kHz}$.
- All segments in the **upper** layer, except for the $n = 2$
  cross-layer case, which is delegated to
  :mod:`groundfield.coupling.layered_green`. The layer stack is the
  canonical one (equal adjacent layers merged, 10.48), so a declared
  three-layer stack with two equal layers is the $n = 2$ case.
- Slower than the closed-form backends. Its value is the kernel: the
  pointwise kernel matches the analytic Tagg/Sunde series to
  $\\lesssim 10^{-13}$ relative (median $\\sim 10^{-16}$) for
  $\\rho_2 / \\rho_1 \\in [10^{-2}, 10^{2}]$, so for $n = 2$ any
  deviation of a closed-form backend *with the same discretisation*
  is that backend's series error; for $n \\ge 3$ it is the only
  kernel available.
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
from groundfield.solver._guards import _MIN_DISTANCE
from groundfield.solver._kernels import HalfSpaceKernel
from groundfield.solver._layered import (
    LayerStack,
    as_layer_stack,
    canonical_soil,
    reflection_gamma,
)
from groundfield.solver._pipeline import _reject_concrete_shells
from groundfield.solver._series import (
    ImageSeries,
    SeriesTruncationWarning,
    _series_tail_bound,
)
from groundfield.solver.image_2layer import _two_layer_self_kernel_factory
from groundfield.solver.mom import _PointMatchingGreen, solve_point_matching
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.solver.result import FieldResult
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
    #: The oscillation budget bound (``osc_panels_required >
    #: max_osc_panels``): the J_0 half periods are not all resolved, and
    #: every rung of the refinement ladder shares that capped grid.
    capped: bool = False


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
    where = (
        f"s = {issue.s:.4g} m, |z - z_s| = {issue.abs_dz:.4g} m, "
        f"z + z_s = {issue.z_plus:.4g} m"
    )
    ladder = (
        f"the last two grids of the refinement ladder {list(_REFINEMENTS)} "
        f"differ by {issue.achieved:.3e}, against a requested tolerance of "
        f"{issue.tol:.3e} (epsabs={epsabs:.1e}, epsrel={epsrel:.1e})"
    )
    if issue.capped:
        head = (
            "mom_sommerfeld: the reflected-remainder quadrature did not "
            f"converge. Worst pair ({where}): {ladder}, but every grid of "
            "the ladder shares a capped oscillation grid, so their "
            "agreement does not certify the value."
        )
    else:
        head = (
            "mom_sommerfeld: the reflected-remainder quadrature did not "
            f"converge. Worst pair ({where}): {ladder}."
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
        if issue.capped
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
        head + " The returned value is NOT converged and, because the "
        "integrand is oscillatory, its SIGN may be wrong." + cause
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
    edges = lam_max * t**2

    # Geometric family for the multiple-reflection pole.
    lam_first = lam_max / float(n_dec) ** 2
    lam_lo = max(_POLE_MARGIN * float(lam_pole), lam_max * 1e-15)
    if 0.0 < lam_lo < lam_first:
        n_decades = math.log10(lam_first / lam_lo)
        n_pole = max(1, int(math.ceil(n_decades * _POLE_PANELS_PER_DECADE * refine)))
        edges = np.concatenate([edges, np.geomspace(lam_lo, lam_first, n_pole + 1)])

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
                    budget,
                    n_half,
                    s,
                    lam_required,
                )
            lam_covered = min(n_osc * step, lam_required)
            edges = np.concatenate([edges, step * np.arange(1, n_osc + 1, dtype=float)])
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
    g_analytic, value, issue = _reflected_parts(
        stack,
        s,
        abs_dz,
        z_plus,
        lambda_max_factor=lambda_max_factor,
        epsabs=epsabs,
        epsrel=epsrel,
        reference=reference,
        max_osc_panels=max_osc_panels,
    )
    if issue is not None:
        if issues is None:
            _emit_convergence_warning(
                [issue],
                epsabs=epsabs,
                epsrel=epsrel,
                max_osc_panels=max_osc_panels,
                stacklevel=4,
            )
        else:
            issues.append(issue)
    return g_analytic + value


def _reflected_parts(
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
) -> tuple[float, float, _QuadratureIssue | None]:
    """Return the two parts of :func:`_reflected_integral` and its issue.

    Returns ``(g_analytic, value, issue)``: the closed-form leading
    interface reflection $K_1 / \\sqrt{s^2 + (2 h_1 - p)^2}$ (1/m), the
    numerically integrated, exponentially decaying remainder (1/m), and
    a :class:`_QuadratureIssue` if the refinement ladder did not converge
    (else ``None``). ``g_analytic + value`` is the reflected kernel;
    the remainder is the smooth part the depth-group tabulation of
    :func:`_build_Z_sommerfeld` interpolates (10.61). Parameters as in
    :func:`_reflected_integral`; nothing is warned here.
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
            r_int,
            _MIN_DISTANCE,
            s,
            z_plus,
            h_top,
        )

    if k_1 == 0.0 and stack.n_layers == 2:
        # Γ_1 ≡ 0: the reflected part vanishes identically. Keeping
        # this exact makes the ρ_1 == ρ_2 limit bit-exact (``g_analytic``
        # is +0.0 here, so the sum is the +0.0 returned up to 0.15.0).
        return g_analytic, 0.0, None

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
    ladder_converged = False
    for refine in _REFINEMENTS:
        grid = _panel_edges(
            lam_max,
            s,
            lam_osc,
            refine,
            lam_pole=lam_pole,
            max_osc_panels=max_osc_panels,
        )
        value = _integrate_panels(_residual, grid.edges)
        if previous is not None:
            tol = max(
                epsabs,
                epsrel * max(abs(reference), abs(value + g_analytic)),
            )
            achieved = abs(value - previous)
            if achieved <= tol:
                ladder_converged = True
                break
        previous = value
    # Review pass 10, 10.47: once the oscillation budget binds, every rung
    # of the ladder uses the same capped half-period grid (the budget is
    # spent coverage-first), so two rungs can agree although the J_0
    # oscillation beyond ``grid.lam_covered`` was never resolved — and
    # with the absolute floor ``epsabs`` dominating for |G| < epsabs /
    # epsrel they did, silently (3.1e-3 error at s = 5 km over a 5 cm
    # layer). A capped budget is therefore non-convergence by itself.
    # ``lam_covered < lam_required`` alone is not the test: it also holds
    # for the last, partial half period of a fully resolved grid.
    capped = grid.osc_panels_required > max(1, int(max_osc_panels))
    issue = None
    if not ladder_converged or capped:
        _log.debug(
            "mom_sommerfeld: remainder quadrature not converged to "
            "epsrel=%.1e (s=%.3g, |dz|=%.3g, z+z_s=%.3g, "
            "achieved=%.3e vs tol=%.3e, lam_covered=%.4g of %.4g, "
            "oscillation budget bound: %s)",
            epsrel,
            s,
            abs_dz,
            z_plus,
            achieved,
            tol,
            grid.lam_covered,
            grid.lam_required,
            capped,
        )
        issue = _QuadratureIssue(
            achieved=float(achieved),
            tol=float(tol),
            s=float(s),
            abs_dz=float(abs_dz),
            z_plus=float(z_plus),
            grid=grid,
            capped=bool(capped),
        )
    return g_analytic, value, issue


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

        On the delegated $n = 2$ cross-layer path ($z > h_1$ or
        $z_s > h_1$, forwarded to
        :func:`groundfield.coupling.layered_green.two_layer_real_space_kernel`)
        the keyword is measured in units of a *geometric*
        characteristic length instead. Since the F14 fix of 0.15.0
        that kernel adds every image term it does not resolve back in
        closed form, so the factor is only an upper bound there as
        well: values from 30 to 20 000 change the delegated kernel by
        at most 1.3e-8 and raise no warning (review pass 10; the
        pre-0.15.0 advice to *lower* the factor on that path is
        obsolete).
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
        r = np.sqrt(s**2 + (z - z_s) ** 2)
        r_img = np.sqrt(s**2 + (z + z_s) ** 2)
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

        # ``lambda_max_factor`` is measured in units of a geometric
        # length here (1/d_rem on the top-layer path below). It is
        # forwarded verbatim: since F14 (0.15.0) the delegated kernel
        # adds the unresolved image terms back in closed form, so the
        # factor is an upper bound only (30 ... 20 000 change the
        # kernel by <= 1.3e-8, review pass 10).
        rho_1_local = float(stack.rhos[0])
        rho_2_local = float(stack.rhos[1])
        h_1_local = float(stack.h[0])
        G_phys = two_layer_real_space_kernel(
            s=s,
            z=z,
            z_s=z_s,
            rho_1=rho_1_local,
            rho_2=rho_2_local,
            h_1=h_1_local,
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
        stack,
        s,
        abs_dz,
        z_plus,
        lambda_max_factor=lambda_max_factor,
        epsabs=epsabs,
        epsrel=epsrel,
        reference=G_direct,
        max_osc_panels=max_osc_panels,
        issues=issues,
    )
    return float(G_direct + G_refl)


# ---------------------------------------------------------------------
# Reaction matrix: grouped evaluation and depth-group tabulation
# ---------------------------------------------------------------------

#: Log-spaced table nodes per decade of horizontal distance in the first
#: tabulation attempt of a depth group (10.61); a failed accuracy check
#: doubles the density once. Measured on four 3- and 4-layer stacks over
#: s = 0.3 … 300 m: largest error at the interval midpoints 1.1e-7 with 64
#: nodes per decade and 7e-9 with 128 (32 per decade: 2.1e-6).
_TAB_NODES_PER_DECADE = 64
#: Smallest number of table nodes of a depth group.
_TAB_MIN_NODES = 12
#: Number of node intervals whose midpoint is checked against the
#: pointwise quadrature after a table is built.
_TAB_CHECKS = 8
#: Accepted relative error of the interpolated kernel at the check
#: points, measured against the whole kernel :math:`|G|`. The check
#: points are interval midpoints, where a cubic spline's error peaks;
#: the error of the assembled matrix stays below 1e-6 (benchmark B12).
_TAB_TOL = 1e-7


def _direct_kernel(s: np.ndarray, abs_dz: float, z_plus: float) -> np.ndarray:
    """Closed-form direct pair ``1/r + 1/r_img`` in 1/m (floored at 1 mm)."""
    r = np.maximum(np.hypot(s, abs_dz), _MIN_DISTANCE)
    r_img = np.maximum(np.hypot(s, z_plus), _MIN_DISTANCE)
    return 1.0 / r + 1.0 / r_img


def _interface_kernel(stack: LayerStack, s: np.ndarray, z_plus: float) -> np.ndarray:
    """Closed-form leading interface reflection ``K_1 / r_int`` in 1/m."""
    d_int = 2.0 * float(stack.h[0]) - z_plus
    return float(stack.K[0]) / np.maximum(np.hypot(s, d_int), _MIN_DISTANCE)


def _tabulated_kernel(
    stack: LayerStack,
    abs_dz: float,
    z_plus: float,
    s_values: np.ndarray,
    quad: dict,
) -> tuple[np.ndarray, float, int] | None:
    """Kernel of one depth group from a table of the quadrature remainder.

    For a fixed depth pair the top-layer kernel is a function of the
    horizontal distance $s$ alone,
    $G(s) = G_\\text{direct}(s) + K_1/r_\\text{int}(s) + R(s)$, where only
    the remainder $R$ needs the quadrature (:func:`_reflected_parts`).
    $R$ is smooth and decays on the scale $d_\\text{rem}$, so it is
    tabulated on log-spaced nodes in $[s_\\min, s_\\max]$ and
    interpolated with a cubic spline in $\\ln s$; the two closed-form
    terms are evaluated exactly (review pass 10, 10.61; R2-06).

    The table is verified at the midpoints of :data:`_TAB_CHECKS` node
    intervals against the pointwise quadrature; if the relative error
    exceeds :data:`_TAB_TOL` the node density is doubled once.

    Parameters
    ----------
    stack
        Layer stack, ``n_layers >= 3`` (both points in the top layer).
    abs_dz, z_plus
        $|z - z_s|$ and $z + z_s$ of the group in m.
    s_values
        Distinct horizontal distances in m, sorted ascending, all > 0.
    quad
        Quadrature controls (``lambda_max_factor``, ``epsabs``,
        ``epsrel``, ``max_osc_panels``).

    Returns
    -------
    tuple or None
        ``(G, check_error, n_evaluations)``: the kernel in 1/m at every
        ``s_values`` entry, the largest relative error at the check
        points and the number of quadratures spent — or ``None`` when
        pointwise evaluation is cheaper, a node quadrature did not
        converge, or the table failed its check twice (the caller then
        evaluates every distance pointwise).
    """
    from scipy.interpolate import CubicSpline

    s_lo, s_hi = float(s_values[0]), float(s_values[-1])
    decades = math.log10(s_hi / s_lo) if s_hi > s_lo else 0.0

    def remainder(s: float) -> tuple[float, float]:
        reference = float(_direct_kernel(np.array(s), abs_dz, z_plus))
        g_analytic, value, issue = _reflected_parts(
            stack, s, abs_dz, z_plus, reference=reference, **quad
        )
        if issue is not None:
            raise _TableFailure
        return value, reference + g_analytic + value

    evaluations = 0
    density = _TAB_NODES_PER_DECADE
    for _attempt in range(2):
        n_nodes = max(_TAB_MIN_NODES, int(math.ceil(decades * density)) + 1)
        if n_nodes + _TAB_CHECKS >= s_values.size:
            return None
        t_nodes = np.linspace(math.log(s_lo), math.log(s_hi), n_nodes)
        s_nodes = np.exp(t_nodes)
        s_nodes[0], s_nodes[-1] = s_lo, s_hi
        try:
            values = np.array([remainder(float(s))[0] for s in s_nodes])
            spline = CubicSpline(t_nodes, values)
            picks = np.unique(
                np.linspace(0, n_nodes - 2, min(_TAB_CHECKS, n_nodes - 1)).round()
            ).astype(int)
            t_check = 0.5 * (t_nodes[picks] + t_nodes[picks + 1])
            exact = [remainder(float(np.exp(t))) for t in t_check]
        except _TableFailure:
            return None
        evaluations += n_nodes + len(t_check)
        error = max(
            abs(float(spline(t)) - v) / max(abs(g), 1e-300)
            for t, (v, g) in zip(t_check, exact, strict=True)
        )
        if error <= _TAB_TOL:
            G = (
                _direct_kernel(s_values, abs_dz, z_plus)
                + _interface_kernel(stack, s_values, z_plus)
                + spline(np.log(s_values))
            )
            return G, float(error), evaluations
        density *= 2
    return None


class _TableFailure(Exception):
    """A table node did not converge; evaluate the group pointwise."""


def _kernel_matrix(
    seg_points: np.ndarray,
    stack: LayerStack,
    quad: dict,
    *,
    tabulate: bool,
    report: dict,
) -> tuple[np.ndarray, list[tuple[int, int, _QuadratureIssue]]]:
    """Off-diagonal Sommerfeld kernel ``G`` (1/m) of every segment pair.

    The kernel of a pair depends on the horizontal distance $s$ and the
    two depths only, and for two points in the top layer on
    $(|z - z_s|, z + z_s)$, i.e. symmetrically. Pairs are therefore
    grouped by their depth classes and every distinct argument is
    evaluated once, by exactly the scalar call
    :func:`sommerfeld_kernel_value` of the historic pair loop — the
    entries are bit-identical to it, and each value serves both
    $(i, j)$ and $(j, i)$ (review pass 10, R2-06 item 2). With
    ``tabulate`` a top-layer depth group of an $n \\ge 3$ stack with
    many distinct distances is interpolated instead
    (:func:`_tabulated_kernel`, 10.61).

    Parameters
    ----------
    seg_points
        Segment midpoints in m, shape ``(N, 3)``.
    stack
        Layer stack, ``n_layers >= 2``.
    quad
        Quadrature controls of :func:`sommerfeld_kernel_value`.
    tabulate
        Allow the depth-group tabulation (``n_layers >= 3`` only).
    report
        Updated in place: ``groups_tabulated``, ``groups_pointwise``,
        ``table_evaluations``, ``max_check_error``.

    Returns
    -------
    G
        Kernel per ordered pair, shape ``(N, N)``; the diagonal is zero.
    issues
        ``(i, j, issue)`` for every pair whose quadrature did not
        converge, in row-major order.
    """
    n = seg_points.shape[0]
    x = seg_points[:, 0]
    y = seg_points[:, 1]
    z = seg_points[:, 2]
    G = np.zeros((n, n), dtype=float)
    issue_at: dict[tuple[int, int], _QuadratureIssue] = {}
    z_vals, z_cls = np.unique(z, return_inverse=True)
    members = [np.flatnonzero(z_cls == c) for c in range(z_vals.size)]
    h_top = stack.h[0]
    two_layer = stack.n_layers == 2
    for ci in range(z_vals.size):
        for cj in range(z_vals.size):
            z_i, z_j = float(z_vals[ci]), float(z_vals[cj])
            # The n = 2 cross-layer kernel is evaluated per ordered depth
            # pair (it is reciprocal only up to its quadrature); every
            # other pair is symmetric and done once for (ci, cj), ci <= cj.
            cross = two_layer and (z_i > h_top or z_j > h_top)
            if cj < ci and not cross:
                continue
            rows, cols = members[ci], members[cj]
            dx = x[rows][:, None] - x[cols][None, :]
            dy = y[rows][:, None] - y[cols][None, :]
            s = np.sqrt(dx * dx + dy * dy)
            off = rows[:, None] != cols[None, :]
            if not off.any():
                continue
            s_unique, inverse = np.unique(s[off], return_inverse=True)
            values = None
            if tabulate and not cross and stack.n_layers >= 3:
                positive = s_unique > 0.0
                table = (
                    _tabulated_kernel(
                        stack, abs(z_i - z_j), z_i + z_j, s_unique[positive], quad
                    )
                    if positive.sum() > 1
                    else None
                )
                if table is not None:
                    values = np.empty(s_unique.size)
                    values[positive] = table[0]
                    for k in np.flatnonzero(~positive):
                        values[k] = sommerfeld_kernel_value(
                            stack, s=0.0, z=z_i, z_s=z_j, **quad
                        )
                    report["groups_tabulated"] += 1
                    report["table_evaluations"] += table[2]
                    report["max_check_error"] = max(report["max_check_error"], table[1])
            group_issues: dict[int, _QuadratureIssue] = {}
            if values is None:
                values = np.empty(s_unique.size)
                for k, s_k in enumerate(s_unique):
                    found: list[_QuadratureIssue] = []
                    values[k] = sommerfeld_kernel_value(
                        stack, s=float(s_k), z=z_i, z_s=z_j, issues=found, **quad
                    )
                    if found:
                        group_issues[k] = found[0]
                report["groups_pointwise"] += 1
            block = np.zeros(s.shape)
            block[off] = values[inverse]
            G[np.ix_(rows, cols)] = block
            if ci != cj and not cross:
                G[np.ix_(cols, rows)] = block.T
            if group_issues:
                index = np.full(s.shape, -1)
                index[off] = inverse
                for a, b in zip(
                    *np.nonzero(np.isin(index, list(group_issues))), strict=True
                ):
                    issue = group_issues[int(index[a, b])]
                    issue_at[(int(rows[a]), int(cols[b]))] = issue
                    if ci != cj and not cross:
                        issue_at[(int(cols[b]), int(rows[a]))] = issue
    issues = [(i, j, issue_at[(i, j)]) for i, j in sorted(issue_at)]
    return G, issues


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
    issues: list[_QuadratureIssue] | None = None,
    tabulate: bool = False,
    report: dict | None = None,
    stacklevel: int = 4,
) -> np.ndarray:
    """N×N Sommerfeld reaction matrix in Ω.

    Off-diagonal entries: Sommerfeld kernel through
    :func:`sommerfeld_kernel_value`, evaluated once per distinct
    argument (:func:`_kernel_matrix`) or, with ``tabulate`` and
    ``n_layers >= 3``, interpolated per depth group (10.61). Diagonal
    entries: line self-potential of the homogeneous bulk plus the
    layered-soil correction. For ``n_layers == 2`` the diagonal is the
    closed-form Tagg/Sunde reaction diagonal (consistent with the
    off-diagonal Sommerfeld integral, which evaluates the same physics
    by direct quadrature). For ``n_layers >= 3`` the *reflected
    remainder* of the analytic split (:func:`_reflected_integral`)
    evaluated at ``s = 0, z = z_s`` is added: the direct $1/r$
    singularity lives entirely in the analytic part that the
    homogeneous line self-potential already accounts for, so the
    remainder is regular point-on-source. Both the homogeneous and the
    two-layer diagonal come from the directly assembled reaction
    matrices (10.59), not from the kernel applied to the identity.

    Non-convergence of any single pair is collected and reported as
    **one** aggregated :class:`SommerfeldConvergenceWarning` naming the
    worst offender and the number of affected entries — an $O(N^2)$
    loop that warned per pair would emit thousands of distinct messages.
    If ``issues`` is a list, the collected issues are appended to it as
    well, so the caller can report convergence in the result metadata.

    Parameters
    ----------
    seg_points, seg_lengths, wire_radii
        Segment midpoints (m), lengths (m) and wire radii (m).
    stack
        Canonical layer stack.
    lambda_max_factor, epsabs, epsrel, max_osc_panels
        Quadrature controls, see :func:`sommerfeld_kernel_value`.
    max_terms, tol
        Tagg/Sunde truncation of the ``n_layers == 2`` diagonal.
    issues
        Optional collector of the non-converged entries.
    tabulate
        Interpolate the reflected remainder per depth group for
        ``n_layers >= 3`` (10.61). ``False`` (the default of this
        private builder) evaluates every distinct pair by quadrature.
    report
        Optional dict, updated with the tabulation counters
        ``groups_tabulated``, ``groups_pointwise``,
        ``table_evaluations`` and ``max_check_error``.
    stacklevel
        ``stacklevel`` of the aggregated warning, counted from
        :func:`_emit_convergence_warning`.
    """
    n = seg_points.shape[0]
    rho_1 = float(stack.rhos[0])
    if report is None:
        report = {}
    for key, start in (
        ("groups_tabulated", 0),
        ("groups_pointwise", 0),
        ("table_evaluations", 0),
        ("max_check_error", 0.0),
    ):
        report.setdefault(key, start)

    if stack.n_layers <= 1:
        return HalfSpaceKernel(rho_1).matrix(seg_points, seg_lengths, wire_radii)

    quad = {
        "lambda_max_factor": lambda_max_factor,
        "epsabs": epsabs,
        "epsrel": epsrel,
        "max_osc_panels": max_osc_panels,
    }
    G, pair_issues = _kernel_matrix(
        seg_points, stack, quad, tabulate=bool(tabulate), report=report
    )
    found = [issue for _i, _j, issue in pair_issues]
    off = ~np.eye(n, dtype=bool)
    Z = np.zeros((n, n), dtype=float)
    Z[off] = rho_1 / (4.0 * np.pi) * G[off]

    if stack.n_layers == 2:
        # Closed-form Tagg/Sunde reaction diagonal — consistent with the
        # off-diagonal Sommerfeld integral. allow_cross_layer=True keeps
        # it consistent for interface-crossing geometries as well.
        soil = TwoLayerSoil(
            rho_1=float(stack.rhos[0]),
            rho_2=float(stack.rhos[1]),
            h_1=float(stack.h[0]),
        )
        layered = _two_layer_self_kernel_factory(
            soil,
            max_terms=max_terms,
            tol=tol,
            allow_cross_layer=True,
        ).matrix(seg_points, seg_lengths, wire_radii)
        np.fill_diagonal(Z, np.diag(layered))
    else:
        # n >= 3: homogeneous line self-potential plus the reflected
        # remainder at s = 0, |z - z_s| = 0, where the physical decay
        # scale is 2·(h_1 - z_i), *not* min(h_1, z_i) as assumed before
        # 0.15.0. Segments with the same depth and self term share one
        # quadrature.
        homogeneous = HalfSpaceKernel(rho_1).matrix(seg_points, seg_lengths, wire_radii)
        diag_cache: dict[
            tuple[float, float], tuple[float, _QuadratureIssue | None]
        ] = {}
        for i in range(n):
            z_plus = 2.0 * float(seg_points[i, 2])
            reference = 4.0 * np.pi * homogeneous[i, i] / rho_1
            key = (z_plus, float(reference))
            if key not in diag_cache:
                g_analytic, value, issue = _reflected_parts(
                    stack, 0.0, 0.0, z_plus, reference=reference, **quad
                )
                diag_cache[key] = (g_analytic + value, issue)
            refl, issue = diag_cache[key]
            if issue is not None:
                found.append(issue)
            layered_offset = rho_1 / (4.0 * np.pi) * refl
            Z[i, i] = homogeneous[i, i] + layered_offset

    _emit_convergence_warning(
        found,
        epsabs=epsabs,
        epsrel=epsrel,
        max_osc_panels=max_osc_panels,
        stacklevel=stacklevel,
    )
    if issues is not None:
        issues.extend(found)
    return Z


# ---------------------------------------------------------------------
# Green's function and main entry point
# ---------------------------------------------------------------------


class _MatrixKernel:
    """Reaction operator of an assembled matrix ``Z`` in Ω."""

    materialised = True

    def __init__(self, Z: np.ndarray) -> None:
        self.Z = Z

    def matrix(
        self, seg_points: np.ndarray, seg_lengths: np.ndarray, wire_radii: np.ndarray
    ) -> np.ndarray:
        """Return the assembled matrix (no copy)."""
        return self.Z

    def __call__(
        self,
        seg_points: np.ndarray,
        seg_lengths: np.ndarray,
        wire_radii: np.ndarray,
        currents: np.ndarray,
    ) -> np.ndarray:
        return self.Z @ currents


class _SommerfeldGreen(_PointMatchingGreen):
    """Green's function of ``mom_sommerfeld``: quadrature of the layered kernel.

    Parameters
    ----------
    soil
        Soil model of the world; it is solved on its canonical stack
        (10.48).
    lambda_max_factor, epsabs, epsrel, max_osc_panels
        Quadrature controls, see :func:`sommerfeld_kernel_value`.
    max_terms, tol
        Tagg/Sunde truncation of the ``n = 2`` reaction diagonal.
    tabulate
        Depth-group tabulation for ``n >= 3`` (10.61).
    """

    backend = "mom_sommerfeld"
    where = "mom_sommerfeld"

    def __init__(
        self,
        soil: HomogeneousSoil | TwoLayerSoil | MultiLayerSoil,
        *,
        lambda_max_factor: float,
        epsabs: float,
        epsrel: float,
        max_osc_panels: int,
        max_terms: int,
        tol: float,
        tabulate: bool,
    ) -> None:
        self.stack = as_layer_stack(soil)
        self.soil = canonical_soil(soil)
        stack = self.stack
        self.layer_interfaces = (float(stack.h[0]),) if stack.n_layers >= 2 else None
        self.soil_resistivity = float(stack.rhos[0])
        self.reference_resistivity = self.soil_resistivity
        self.lambda_max_factor = lambda_max_factor
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.max_osc_panels = max_osc_panels
        self.max_terms = max_terms
        self.tol = tol
        self.tabulate = bool(tabulate) and stack.n_layers >= 3
        self.quadrature_issues: list[_QuadratureIssue] = []
        self.tabulation: dict = {}
        self.z_max = 0.0

    def prepare(self, segments, arrays, *, stacklevel: int):  # noqa: ANN001
        """Assemble the Sommerfeld reaction matrix (the expensive step)."""
        stack = self.stack
        self.z_max = float(arrays.points[:, 2].max())
        if stack.n_layers >= 3:
            h_1 = float(stack.h[0])
            if self.z_max >= h_1:
                # ADR-0007 Phase B (n≥3): not implemented — the
                # reflection-only diagonal correction assumes all
                # segments in the top layer. Hard error instead of a
                # warning followed by wrong physics.
                raise ValueError(
                    f"mom_sommerfeld: cross-layer geometry on "
                    f"n_layers={stack.n_layers} is not supported "
                    f"(z_max = {self.z_max:.3f} m >= h_1 = {h_1:.3f} m). "
                    "Use backend='image_2layer' for n=2 cross-layer "
                    "worlds; for n>=3 thicken the upper layer."
                )
        Z = _build_Z_sommerfeld(
            arrays.points,
            arrays.lengths,
            arrays.radii,
            stack,
            lambda_max_factor=self.lambda_max_factor,
            epsabs=self.epsabs,
            epsrel=self.epsrel,
            max_osc_panels=self.max_osc_panels,
            max_terms=self.max_terms,
            tol=self.tol,
            issues=self.quadrature_issues,
            tabulate=self.tabulate,
            report=self.tabulation,
            stacklevel=stacklevel + 2,
        )
        return _MatrixKernel(Z)

    def metadata(self, solution, *, stacklevel: int) -> dict:  # noqa: ANN001
        """Quadrature controls and the convergence report (10.47, 10.61)."""
        stack = self.stack
        n_freq = len(solution.segment_currents)
        # For n = 2 inside the upper layer the diagonal is the truncated
        # Tagg/Sunde series, which is checked like in ``mom`` (it was
        # silent here up to 0.15.0, R2-09).
        series_meta: dict[str, int] = {}
        series_converged = True
        if stack.n_layers == 2 and self.z_max < float(stack.h[0]):
            k_1 = float(stack.K[0])
            n_terms_used = ImageSeries.from_coefficients(
                k_1, float(stack.h[0]), self.max_terms, self.tol
            ).n_terms
            series_meta = {"n_terms_used": int(n_terms_used)}
            if n_terms_used:
                tail = _series_tail_bound(abs(k_1), n_terms_used)
                series_converged = bool(tail < self.tol)
                if not series_converged:
                    warnings.warn(
                        "mom_sommerfeld: the Tagg/Sunde series of the reaction-"
                        f"matrix diagonal was truncated at max_terms="
                        f"{self.max_terms} with a remaining tail bound of "
                        f"{tail:.2e} > tol={self.tol:.2e} "
                        f"(|K| = {abs(k_1):.3f}). Raise Engine.image_max_terms "
                        "for high layer contrasts.",
                        SeriesTruncationWarning,
                        stacklevel=stacklevel,
                    )
        converged = series_converged and not self.quadrature_issues
        return {
            "n_layers": int(stack.n_layers),
            "rhos": stack.rhos.tolist(),
            "h": stack.h.tolist(),
            "lambda_max_factor": float(self.lambda_max_factor),
            "epsabs": float(self.epsabs),
            "epsrel": float(self.epsrel),
            "max_osc_panels": int(self.max_osc_panels),
            # The reaction matrix is frequency-independent, so one flag
            # holds for every frequency; the per-frequency list is aligned
            # with ``frequencies`` for callers that tabulate per frequency.
            "converged": bool(converged),
            "converged_per_frequency": [bool(converged)] * n_freq,
            "n_unconverged_entries": len(self.quadrature_issues),
            **series_meta,
            "kernel_tabulation": {
                "enabled": bool(self.tabulate),
                "groups_tabulated": int(self.tabulation.get("groups_tabulated", 0)),
                "groups_pointwise": int(self.tabulation.get("groups_pointwise", 0)),
                "table_evaluations": int(self.tabulation.get("table_evaluations", 0)),
                "max_check_error": float(self.tabulation.get("max_check_error", 0.0)),
            },
        }


def solve_mom_sommerfeld(
    world: World,
    engine: Engine,
    *,
    lambda_max_factor: float = 200.0,
    epsabs: float = 1e-9,
    epsrel: float = 1e-7,
    max_osc_panels: int = _MAX_OSC_PANELS,
    tabulate: bool = True,
) -> FieldResult:
    """Point-matching MoM with direct Sommerfeld quadrature for layered soil.

    The discretisation and the linear solve are those of ``mom`` (point
    matching at the segment midpoints, point-source near field,
    source-free clusters floating), run through the shared solve
    pipeline (10.75). Every off-diagonal reaction-matrix entry is the
    Sommerfeld kernel of :func:`sommerfeld_kernel_value`, i.e. the
    closed-form direct pair plus a numerically integrated reflected
    remainder; no image series and no fit enter the off-diagonal
    kernel. For $n = 2$ the diagonal is the Tagg/Sunde series diagonal
    of ``mom``, so on two-layer soil this backend reproduces ``mom`` and
    checks the kernel, not the scheme; for $n \\ge 3$ distinct layers it
    is the only integral-equation backend.

    Cost (review pass 10, 10.61): pairs that share a depth pair and a
    horizontal distance share one quadrature, which halves the work of
    every world and collapses it for regular geometries. For
    $n \\ge 3$ the reflected remainder of a depth group with many
    distinct distances is tabulated on log-spaced nodes and interpolated
    (``tabulate=True``, the default): about $10^{-7}$ relative error of
    the kernel at the verification points and below $10^{-6}$ on the
    reaction matrix of the benchmark grid B12, for a cost that no longer
    grows with the number of pairs. ``tabulate=False`` evaluates every
    distinct pair by quadrature (the kernel reference). For $n = 2$
    every pair is evaluated by quadrature: this backend is the kernel
    check of the closed-form series there.

    Parameters
    ----------
    world
        World to evaluate.
    engine
        Engine configuration.
    lambda_max_factor
        Truncation of the remainder quadrature in units of the
        remainder decay length $1 / d_{\\text{rem}}$; saturates at
        60. See :func:`sommerfeld_kernel_value` for the delegated
        $n = 2$ cross-layer path.
    epsabs, epsrel
        Convergence thresholds of the remainder quadrature (absolute
        in 1/m, relative).
    max_osc_panels
        Budget for the $J_0$ half-period panel family.
    tabulate
        Depth-group tabulation of the reflected kernel for $n \\ge 3$
        (see above).

    Returns
    -------
    FieldResult
        ``metadata['converged']`` (one bool) and
        ``metadata['converged_per_frequency']`` report the quadrature
        and series convergence (10.47);
        ``metadata['kernel_tabulation']`` the tabulation counters and
        the largest relative error at its verification points.

    Warns
    -----
    SommerfeldConvergenceWarning
        Once per solve, if any reaction-matrix entry's remainder
        quadrature failed to converge.

    Notes
    -----
    The quadrature keywords are reachable through
    ``Engine.options["mom_sommerfeld"]``
    (:class:`~groundfield.solver.registry.MomSommerfeldOptions`); the
    signature defaults equal the option defaults, so a direct call
    without keywords behaves exactly like
    :meth:`~groundfield.solver.engine.Engine.solve`.
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
    green = _SommerfeldGreen(
        world.soil,
        lambda_max_factor=lambda_max_factor,
        epsabs=epsabs,
        epsrel=epsrel,
        max_osc_panels=max_osc_panels,
        max_terms=engine.image_max_terms,
        tol=engine.image_series_tol,
        tabulate=tabulate,
    )
    _log.info(
        "mom_sommerfeld: n_layers=%d, segment_length=%.3f, lam_max_fac=%.1f",
        green.stack.n_layers,
        engine.segment_length,
        lambda_max_factor,
    )
    return solve_point_matching(world, engine, green, stacklevel=3)
