# `mom_sommerfeld` — layered Green's function by quadrature

This page explains the engine for soils of three and more layers: the
layered Green's function as a Sommerfeld integral, how the integral is
evaluated without silent truncation, what it costs and where its limits
are. Read it when your soil model has more than two layers, or when you
want the kernel reference for two-layer soil.

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | any number of layers (canonical stack; equal adjacent layers merged) |
    | Green's function | the Sommerfeld integral with the recursive reflection coefficient, evaluated numerically |
    | Current model | one pulse current per segment, point matching (the scheme of [`mom`](mom.md)) |
    | Kernel | point only |
    | Options | `MomSommerfeldOptions`: `lambda_max_factor`, `epsabs`, `epsrel`, `max_osc_panels`, `tabulate` |
    | Accuracy | the kernel to $10^{-9}$ or better; the point-matching deficit of `mom` on impedances |
    | Status | current; the only engine for three or more distinct layers |

## What it solves

`mom_sommerfeld` computes the quasi-static current field of thin-wire
electrodes in a horizontally stratified half-space: layers
$i = 1 \dots n$ of resistivity $\rho_i$ and thickness $h_i$, the last one
infinite, insulating air above. It evaluates the layered Green's function
without expansion or fit: the reflection coefficient $\Gamma_1(\lambda)$
of the whole stack enters the Sommerfeld integral as it is, and the
integral is computed by quadrature for every pair of segments.

That makes it the reference for the **kernel**: on two-layer soil it must
reproduce the Tagg / Sunde series of [`mom`](mom.md) on the same
discretisation, and for three and more layers it is the only kernel there
is. The discretisation and the current model are those of `mom` (one
current per segment, point matching), so it is not an independent check
of the point-matching scheme.

Concrete-encased wires carry the per-segment shell term of `mom`
(ADR-0012). On two-layer soil the two engines agree to about $10^{-8}$
with and without shells where neither engine warns; under
`SeriesTruncationWarning` (a truncated series) they can differ by 9 %.
An encased ring in three layers rises by exactly $C/(2\pi r)$ over the
bare ring of the outer radius ($C$ the shell coefficient, $r$ the ring
radius).

## Mathematical and physical context

### The Sommerfeld integral of a layered soil

The potential of a point current $I$ at depth $z_s$ in the top layer,
observed at horizontal distance $s$ and depth $z$ in the top layer, is

$$
\varphi = \frac{\rho_1 I}{4\pi}\int_0^\infty
\frac{e^{-\lambda\lvert z-z_s\rvert} + e^{-\lambda(z+z_s)}
+ \Gamma_1(\lambda)\bigl[e^{-\lambda(2h_1-\lvert z-z_s\rvert)} + e^{-\lambda(2h_1-z-z_s)}\bigr]}
{1 - \Gamma_1(\lambda)\,e^{-2\lambda h_1}}\,J_0(\lambda s)\,\mathrm d\lambda ,
$$

with the spatial frequency $\lambda$ (in 1/m) and the Bessel function
$J_0$. It is the static limit of Sommerfeld's (1909) half-space problem:
the potential is decomposed into cylindrical waves $J_0(\lambda s)$, each
of which is reflected by the stack below the top layer with the
coefficient $\Gamma_1(\lambda)$. The numerator holds the direct field, its
air image and the first reflection; the denominator sums the reflections
back and forth between the surface and the first interface.

### The reflection coefficient of the stack

The reflection coefficient seen from the top layer is built bottom-up
from the interface coefficients
$K_i = (\rho_{i+1}-\rho_i)/(\rho_{i+1}+\rho_i)$ of the interfaces
$i = 1 \dots n-1$:

$$
\Gamma_{n-1} = K_{n-1},\qquad
\Gamma_i(\lambda) = \frac{K_i + \Gamma_{i+1}(\lambda)\,e^{-2\lambda h_{i+1}}}
{1 + K_i\,\Gamma_{i+1}(\lambda)\,e^{-2\lambda h_{i+1}}}.
$$

Each step combines the reflection at interface $i$ with what comes back,
attenuated by the round trip $e^{-2\lambda h_{i+1}}$ through layer $i+1$,
from the stack below it. Short wavelengths (large $\lambda$) see only the
first interface, $\Gamma_1(\infty) = K_1$; long wavelengths see the deep
layers. $\Gamma_1$ is therefore constant only for two layers: for $n = 2$
the integral expands into the image series of
[`image_2layer`](image_2layer.md#from-the-sommerfeld-integral-to-the-image-series),
for $n \ge 3$ no closed form exists.

### Electrodes and layers

For $n = 2$ with every segment in the top layer the diagonal of the
reaction matrix is the series diagonal of `mom`, and the off-diagonal
entries are quadratures; a world with an electrode at or below the
interface is delegated to the two-layer Sommerfeld kernel of
`groundfield.coupling.layered_green` (ADR-0007). For $n \ge 3$ every
segment must lie in the top layer.

## Numerics

### Quadrature

The integrand does not decay for two points at the same depth: the direct
term $e^{-\lambda\lvert z-z_s\rvert}J_0(\lambda s)$ with $z = z_s$
converges only through the oscillation of $J_0$, and a truncation at
$\lambda_\text{max}$ leaves an error of order
$\sqrt{2/(\pi\lambda_\text{max}s)}$ that no quadrature tolerance can
remove. The kernel is therefore split with Lipschitz' integral
$\int_0^\infty e^{-\lambda d}J_0(\lambda s)\,\mathrm d\lambda = (s^2+d^2)^{-1/2}$:

1. the direct term ($d = \lvert z - z_s\rvert$), its air image
   ($d = z + z_s$) and the leading interface reflection with
   $\Gamma_1(\infty) = K_1$ ($d = 2h_1 - z - z_s$) are taken in closed
   form;
2. the remainder decays at least like $e^{-\lambda d_\text{rem}}$ with a
   strictly positive length $d_\text{rem}$, even when both points sit on
   the interface, so its truncation error is bounded by
   $e^{-\lambda_\text{max}d_\text{rem}}$;
3. the remainder is integrated with Gauss–Legendre panels on a grid graded
   towards $\lambda = 0$ (exponentials and $\Gamma_1$), with a geometric
   family down to the scale $(1-\Gamma_1(0))/(2h_1)$ of the reflection
   multiplier (which collapses for $\lvert K_1\rvert \to 1$), and
   subdivided at the half period $\pi/s$ of $J_0$;
4. the grid is refined until two successive grids agree within
   $\max(\varepsilon_\text{abs}, \varepsilon_\text{rel}\lvert G\rvert)$,
   $G$ the kernel value.

Failing that is reported, never silent: `SommerfeldConvergenceWarning`
names the achieved and the requested tolerance; a capped oscillation
budget (`max_osc_panels`) counts as non-convergence;
`metadata["converged"]`, `["converged_per_frequency"]` and
`["n_unconverged_entries"]` record it. Distances are floored at 1 mm like
in the image family; keep segment midpoints a few millimetres clear of an
interface, where the floor would saturate the interface term.

### Cost: distinct pairs and depth-group tables

Every distinct argument $(s, \lvert z-z_s\rvert, z+z_s)$ is integrated
once. For $n \ge 3$ the pairs of a depth class in the top layer share one
function $G(s)$ of the horizontal distance alone; with `tabulate=True`
(the default, option of `MomSommerfeldOptions`) its smooth quadrature
remainder is tabulated on 64 log-spaced nodes per decade of $s$ and
interpolated with a cubic spline in $\ln s$, while the closed-form terms
stay exact. Each table is checked at eight interval midpoints against the
pointwise quadrature to $10^{-7}$ of $\lvert G\rvert$, refined once, and
dropped in favour of pointwise quadrature when it fails. Benchmark B12
bounds the effect on the assembled matrix and the impedance by $10^{-6}$
(generated table below); the measured speed-up is on the
[performance page](../performance.md). The current model adds the dense
LU of `mom`, $O(N^3)$ for $N$ segments.

## Validation

The kernel is checked against independent references: the exact Tagg /
Sunde series for two layers (B4, within 2.0e-10 over 1050 points) and
`mpmath` quadrature for three layers (B5, within 2.3e-13). On the same
segments it reproduces `mom` on two-layer soil to about $10^{-8}$ (B6,
B7, B11). Impedances carry the point-matching deficit of
[`mom`](mom.md#numerics), which these checks do not measure.

<!-- generated:begin mom_sommerfeld -->
Generated from the benchmark run measured on 2026-11-16 with groundfield 2026.11.0 (Python 3.12.16, NumPy 2.4.4, SciPy 1.17.1, numba installed: yes; Darwin arm64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

| Benchmark | Measure | Bound | Worst measured | Cases | Status | Note |
|---|---|---|---|---|---|---|
| B4 Two-layer Sommerfeld kernel | rel. kernel error, or SommerfeldConvergenceWarning | 1e-9 | 2.0e-10 | 1 | pass | never silent (10.47); no warning for h_1 >= 0.2 m |
| B5 Three-layer Sommerfeld kernel | rel. kernel error | 1e-10 | 2.3e-13 | 2 | pass |  |
| B6 Top-layer electrodes, two-layer soil | abs(Z/Z_ref - 1) | 1e-7 | 1.2e-8 | 10 | pass | kernel check |
| B7 Rod through the layer interface | abs(Z/Z_ref - 1) | 1e-7 | 1.4e-10 | 2 | pass | kernel check |
| B8 Two-port reciprocity | abs(Z_AB/Z_BA - 1) | 1e-7 | 4.4e-16 | 2 | pass |  |
| B9 Limits of the layered kernels (`equal_rho_integral`) | abs(Z/Z_ref - 1) | 1e-10 | 0 | 3 | pass | image_2layer, mom, mom_sommerfeld, cim, bem |
| B9 Limits of the layered kernels (`layer_limit_integral`) | abs(Z/Z_ref - 1) | 0.1 % | 5.1e-4 | 2 | pass | physical limit |
| B11 Grid refinement | abs(Z/Z_mom - 1) at ds = 5 m | 1e-7 | 7.5e-10 | 1 | pass | kernel check |
| B12 Three-layer grid | max abs(Z_ij,tab / Z_ij - 1) and abs(Z_g1,tab / Z_g1 - 1) | 1e-6 | 6.9e-9 | 4 | pass | depth-group tabulation of the reflected kernel (10.61) |
<!-- generated:end mom_sommerfeld -->

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Soil | any canonical stack | — |
| Electrodes, $n \ge 3$ | every segment in the top layer | `ValueError` |
| Electrodes, $n = 2$ | anywhere (cross-layer path of `layered_green`) | — |
| Quadrature | converged within `epsrel` / `epsabs` (defaults $10^{-7}$ / $10^{-9}$) | `SommerfeldConvergenceWarning`, `metadata["converged"] = False` |
| Interface distance | segment midpoints more than about 1 mm from an interface | interface term saturated by the 1 mm floor |
| Line kernel | not supported | `BackendCapabilityError` |
| Concrete shells | thin shell per segment, as in [`mom`](mom.md) | — |
| Size | dense matrix, one quadrature per distinct pair argument | seconds to minutes ([performance](../performance.md)) |
| Frequency | quasi-static, $f < 1$ kHz | — |

The thin-wire limits next to an interface (rod ends, horizontal wires,
the point form near $h_1$) are those of [`mom`](mom.md#validity-range).
What to do instead, and the other limits:
[known limitations](../../reference/known-limitations.md#current-ansatz-and-source-kernels);
the warnings: [warnings reference](../../reference/warnings.md).

## Literature

- Sommerfeld, A. (1909). Über die Ausbreitung der Wellen in der drahtlosen
  Telegraphie. *Annalen der Physik* 28(4), 665–736.
- Watson, G. N. (1944). *A Treatise on the Theory of Bessel Functions*,
  2nd ed. Cambridge University Press, §13.2 (Lipschitz' integral).
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission Systems*.
  Dover, New York, sect. 3.5 (the layered-earth integral and its
  recursive reflection coefficient).
- Lucas, S. K., & Stone, H. A. (1995). Evaluating infinite integrals
  involving Bessel functions of arbitrary order. *Journal of Computational
  and Applied Mathematics* 64 (panel quadrature between the zeros of
  $J_0$).
- Zou, J., Du, X., & Zhou, C. (2015). Fast calculation of the Green
  function of a point current source in a horizontal layered soil with a
  new complex path. *IEEE Transactions on Magnetics* 51(3) (a complex
  contour, an alternative to the analytic extraction; not implemented).

Architecture decisions: [ADR-0002](../../adr/0002-engine-family.md),
[ADR-0006](../../adr/0006-sommerfeld-earth-return.md) (the inductive
Sommerfeld kernel shares the quadrature ideas),
[ADR-0007](../../adr/0007-cross-layer-electrodes.md). API:
`groundfield.solver.mom_sommerfeld.solve_mom_sommerfeld`,
`sommerfeld_kernel_value`.

## Audit trail

??? note "Concrete shells against `mom` (since 2026.11; audits W2B-07, W2R2-09)"

    Since 2026.11 concrete-encased wires carry the per-segment shell term of
    `mom` (ADR-0012). On two-layer soil the two engines agree with and
    without shells:

    - $10^{-9}$ … $1.5\cdot10^{-8}$ for a 10 m grid in four soils (audit
      W2B-07);
    - $9\cdot10^{-8}$ with the layer interface 0.2 m below the wires, under
      `PointKernelInterfaceWarning`, and 9 % for a truncated series under
      `SeriesTruncationWarning` (audit W2R2-09);
    - the shell increments agree to $1.4\cdot10^{-10}$, the agreement of the
      two kernels.

??? note "Convergence reporting and tabulation (findings 10.47, 10.61)"

    `SommerfeldConvergenceWarning` and the `converged` metadata were added
    by finding 10.47, so a quadrature that does not converge is never
    silent. The depth-group tabulation with its midpoint check and fallback
    is finding 10.61.
