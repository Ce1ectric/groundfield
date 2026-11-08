# `mom_sommerfeld` — layered Green's function by quadrature

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | any number of layers (canonical stack; equal adjacent layers merged) |
    | Green's function | the Sommerfeld integral with the recursive reflection coefficient, evaluated numerically |
    | Current model | one pulse current per segment, point matching (the scheme of [`mom`](mom.md)) |
    | Kernel | point only |
    | Options | `MomSommerfeldOptions`: `lambda_max_factor`, `epsabs`, `epsrel`, `max_osc_panels`, `tabulate` |
    | Status | current; the only engine for three or more distinct layers |

`mom_sommerfeld` evaluates the layered Green's function without expansion
or fit: the reflection coefficient $\Gamma_1(\lambda)$ of the whole stack
enters the Sommerfeld integral as it is, and the integral is computed by
quadrature for every pair of segments. That makes it the reference for the
**kernel**: on two-layer soil it must reproduce the Tagg / Sunde series of
[`mom`](mom.md) on the same discretisation, and for three and more layers
it is the only kernel there is. The discretisation and the current model
are those of `mom`, so it is not an independent check of the
point-matching scheme.

## Mathematical and physical context

For a horizontally stratified half-space with layers $i = 1 \dots n$
(resistivity $\rho_i$, thickness $h_i$, the last one infinite), the
potential of a point current $I$ at depth $z_s$ in the top layer,
observed at horizontal distance $s$ and depth $z$ in the top layer, is

$$
\varphi = \frac{\rho_1 I}{4\pi}\int_0^\infty
\frac{e^{-\lambda|z-z_s|} + e^{-\lambda(z+z_s)}
+ \Gamma_1(\lambda)\bigl[e^{-\lambda(2h_1-|z-z_s|)} + e^{-\lambda(2h_1-z-z_s)}\bigr]}
{1 - \Gamma_1(\lambda)\,e^{-2\lambda h_1}}\,J_0(\lambda s)\,\mathrm d\lambda .
$$

The reflection coefficient seen from the top layer is built bottom-up
from the interface coefficients $K_i = (\rho_{i+1}-\rho_i)/(\rho_{i+1}+\rho_i)$,

$$
\Gamma_{n-1} = K_{n-1},\qquad
\Gamma_i(\lambda) = \frac{K_i + \Gamma_{i+1}(\lambda)\,e^{-2\lambda h_{i+1}}}
{1 + K_i\,\Gamma_{i+1}(\lambda)\,e^{-2\lambda h_{i+1}}},
$$

so $\Gamma_1(\infty) = K_1$ and $\Gamma_1$ is constant only for two
layers. The integral is the static limit of Sommerfeld's (1909) half-space
problem; for $n = 2$ it expands into the image series of
[`image_2layer`](image_2layer.md), for $n \ge 3$ no closed form exists.

## Quadrature

The integrand does not decay for two points at the same depth: the direct
term $e^{-\lambda|z-z_s|}J_0(\lambda s)$ with $z = z_s$ converges only
through the oscillation of $J_0$, and a truncation at
$\lambda_\text{max}$ leaves an error of order
$\sqrt{2/(\pi\lambda_\text{max}s)}$ that no quadrature tolerance can
remove. The kernel is therefore split with Lipschitz' integral
$\int_0^\infty e^{-\lambda d}J_0(\lambda s)\,\mathrm d\lambda = (s^2+d^2)^{-1/2}$:
the direct term ($d = |z - z_s|$), its air image ($d = z + z_s$) and the
leading interface reflection with $\Gamma_1(\infty) = K_1$
($d = 2h_1 - z - z_s$) are taken in closed form. The remainder decays at
least like $e^{-\lambda d_\text{rem}}$ with a strictly positive length
$d_\text{rem}$, even when both points sit on the interface, so its
truncation error is bounded by $e^{-\lambda_\text{max}d_\text{rem}}$.

The remainder is integrated with Gauss–Legendre panels on a grid graded
towards $\lambda = 0$ (exponentials and $\Gamma_1$), with a geometric
family down to the scale $(1-\Gamma_1(0))/(2h_1)$ of the reflection
multiplier (which collapses for $|K_1| \to 1$), and subdivided at the
half period $\pi/s$ of $J_0$. The grid is refined until two successive
grids agree within $\max(\varepsilon_\text{abs}, \varepsilon_\text{rel}|G|)$.
Failing that is reported, never silent: `SommerfeldConvergenceWarning`
names the achieved and the requested tolerance; a capped oscillation
budget (`max_osc_panels`) counts as non-convergence;
`metadata["converged"]`, `["converged_per_frequency"]` and
`["n_unconverged_entries"]` record it (finding 10.47). Distances are
floored at 1 mm like in the image family; keep segment midpoints a few
millimetres clear of an interface, where the floor would saturate the
interface term.

### Two layers and electrodes across interfaces

For $n = 2$ with every segment in the top layer the diagonal of the
reaction matrix is the series diagonal of `mom`, and the off-diagonal
entries are quadratures; a world with an electrode at or below the
interface is delegated to the two-layer Sommerfeld kernel of
`groundfield.coupling.layered_green` (ADR-0007). For $n \ge 3$ every
segment must lie in the top layer.

### Cost: distinct pairs and depth-group tables

Every distinct argument $(s, |z-z_s|, z+z_s)$ is integrated once. For
$n \ge 3$ the pairs of a depth class in the top layer share one function
$G(s)$ of the horizontal distance alone; with `tabulate=True` (the
default, option of `MomSommerfeldOptions`) its smooth quadrature
remainder is tabulated on 64 log-spaced nodes per decade of $s$ and
interpolated with a cubic spline in $\ln s$, while the closed-form terms
stay exact. Each table is checked at eight interval midpoints against the
pointwise quadrature to $10^{-7}$ of $|G|$, refined once, and dropped in
favour of pointwise quadrature when it fails (finding 10.61). Benchmark B12
bounds the effect on the assembled matrix and the impedance by
$10^{-6}$ (generated table below); the measured speed-up is on the
[performance page](../performance.md).

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Soil | any canonical stack | — |
| Electrodes, $n \ge 3$ | every segment in the top layer | `ValueError` |
| Electrodes, $n = 2$ | anywhere (cross-layer path delegated to `layered_green`) | — |
| Quadrature | convergence within `epsrel` / `epsabs` (defaults $10^{-7}$ / $10^{-9}$) | `SommerfeldConvergenceWarning`, `metadata["converged"] = False` |
| Interface distance | segment midpoints more than about 1 mm from an interface | interface term saturated by the 1 mm floor |
| Concrete shells, line kernel | not supported | `BackendCapabilityError` |
| Size | dense matrix, one quadrature per distinct pair argument | seconds to minutes, see [performance](../performance.md) |
| Frequency | quasi-static, $f < 1$ kHz | — |

## Validation

<!-- generated:begin mom_sommerfeld -->
Generated from the benchmark run measured on 2026-11-08 with groundfield 0.15.0 (Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, numba installed: yes; Linux x86_64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

| Benchmark | Measure | Bound | Worst measured | Cases | Status | Note |
|---|---|---|---|---|---|---|
| B4 Two-layer Sommerfeld kernel | rel. kernel error, or SommerfeldConvergenceWarning | 1e-9 | 2.0e-10 | 1 | pass | never silent (10.47); no warning for h_1 >= 0.2 m |
| B5 Three-layer Sommerfeld kernel | rel. kernel error | 1e-10 | 2.7e-13 | 2 | pass |  |
| B6 Top-layer electrodes, two-layer soil | abs(Z/Z_ref - 1) | 1e-7 | 1.2e-8 | 10 | pass | kernel check |
| B7 Rod through the layer interface | abs(Z/Z_ref - 1) | 1e-7 | 1.4e-10 | 2 | pass | kernel check |
| B8 Two-port reciprocity | abs(Z_AB/Z_BA - 1) | 1e-7 | 4.4e-16 | 2 | pass |  |
| B9 Limits of the layered kernels (`equal_rho_integral`) | abs(Z/Z_ref - 1) | 1e-10 | 0 | 3 | pass | image_2layer, mom, mom_sommerfeld, cim, bem |
| B9 Limits of the layered kernels (`layer_limit_integral`) | abs(Z/Z_ref - 1) | 0.1 % | 5.1e-4 | 2 | pass | physical limit |
| B11 Grid refinement | abs(Z/Z_mom - 1) at ds = 5 m | 1e-7 | 7.5e-10 | 1 | pass | kernel check |
| B12 Three-layer grid | max abs(Z_ij,tab / Z_ij - 1) and abs(Z_g1,tab / Z_g1 - 1) | 1e-6 | 6.9e-9 | 4 | pass | depth-group tabulation of the reflected kernel (10.61) |
<!-- generated:end mom_sommerfeld -->

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
