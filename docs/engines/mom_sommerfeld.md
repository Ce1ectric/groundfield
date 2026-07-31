# `mom_sommerfeld` — Galerkin MoM with direct Sommerfeld quadrature

## Physical context

Every other layered backend in the family relies on a *closed-form*
representation of the layered Green's function:

- [`image_2layer`](image_2layer.md) — geometric Tagg / Sunde series.
- [`image_nlayer`](image_nlayer.md) — dispatcher to the closed-form
  variants.
- [`cim`](cim.md) — matrix-pencil approximation by complex
  exponentials.
- [`bem`](bem.md) — collocation on top of the CIM kernel.

`mom_sommerfeld` is methodologically distinct: it evaluates the
Sommerfeld integral **numerically**, point by point. The recursive
$\Gamma_1(\lambda)$ enters the integrand as is — no expansion, no
fit. The price is speed (a single $N \times N$ reaction matrix can
take seconds rather than milliseconds), but the result is an
**absolute reference**: any disagreement between the closed-form
engines and `mom_sommerfeld` indicates a fit / series accuracy issue,
not a kernel bug.

In ADR-0002 this is the engine that anchors the cross-validation
envelope for the $n \ge 3$ regime, and provides the independent
methodology that the closed-form layered family lacks.

## Governing equation: layered Sommerfeld integral

For a top-layer source at depth $z_s$ and a top-layer field point
at $(s, z)$, the full layered Green's function is the Sommerfeld
integral with multiple-reflection multiplier:

$$
G(s, z, z_s) \;=\; \int_0^{\infty}
\frac{ e^{-\lambda |z - z_s|}
     + e^{-\lambda (z + z_s)}
     + \Gamma_1(\lambda)\, e^{-\lambda (2 h_1 - |z - z_s|)}
     + \Gamma_1(\lambda)\, e^{-\lambda (2 h_1 - z - z_s)} }
     { 1 - \Gamma_1(\lambda)\, e^{-2\lambda h_1} }\,
J_0(\lambda s)\, d\lambda.
$$

The integrand carries:

- **Two direct exponentials** — the source and its air mirror.
- **Two reflected exponentials** — the source's reflection at
  $z = h_1$ and the air mirror's reflection at $z = h_1$.
- **A multiplexion factor** $1 / (1 - \Gamma_1(\lambda) e^{-2\lambda h_1})$ that captures the infinite back-and-forth bouncing
  between the air boundary ($R_{\text{air}} = +1$) and the layer
  interface ($\Gamma_1$).

For $\Gamma_1 \to 0$ (homogeneous) the integral reduces to
$1/r + 1/r_{\text{img}}$. For $\Gamma_1 \equiv K_1$ (2-layer) the
geometric expansion of the multiplier reproduces the Tagg / Sunde
series. For $n \ge 3$ no closed form is available — the engine
just integrates.

## Numerical strategy

### The integral is only conditionally convergent

The first term of the numerator, $e^{-\lambda |z - z_s|}$, does
**not** decay for two points at the same depth — and
$|z - z_s| = 0$ is the normal case for the off-diagonal entries of
a buried horizontal grid. There the integral converges only through
the oscillation of $J_0(\lambda s)$, and a truncation at any
finite $\lambda_{\max}$ leaves a sign-alternating error of order

$$
\varepsilon_{\text{trunc}} \sim
\sqrt{\frac{2}{\pi\, \lambda_{\max}\, s}},
$$

which **no** quadrature tolerance can reduce. Up to and including
0.14.1 the engine did exactly that (a bare
`scipy.integrate.quad(0, lam_max)` with `limit=400`), and returned
$G = -0.012551$ where the exact value is $+0.039990$ for
$s = 50\,\text{m}$, $z = z_s = 0.8\,\text{m}$,
$h_1 = 1\,\text{m}$ — a negative Green's function in the engine
that all other backends are validated against.

### Analytic extraction (0.15.0)

The fix is exact, not a heuristic. Lipschitz' integral

$$
\int_0^{\infty} e^{-\lambda d}\, J_0(\lambda s)\, d\lambda
= \frac{1}{\sqrt{s^2 + d^2}}, \qquad d \ge 0,
$$

gives the closed form of every *single* exponential against
$J_0$. The kernel is therefore split as

$$
G = \underbrace{\frac{1}{\sqrt{s^2 + (z - z_s)^2}}
    + \frac{1}{\sqrt{s^2 + (z + z_s)^2}}
    + \frac{K_1}{\sqrt{s^2 + (2 h_1 - z - z_s)^2}}}_{\text{closed form}}
  \; + \; G_{\text{refl}},
$$

where the first two terms are the $\Gamma_1 \to 0$ limit of the
integrand (the conditionally convergent part — for
$\rho_1 = \rho_2$ they are the *whole* answer) and the third is
the leading interface reflection evaluated with the high-$\lambda$
limit $\Gamma_1(\infty) = K_1$ (exact for $n = 2$, where
$\Gamma_1 \equiv K_1$). What remains,

$$
G_{\text{refl}} = \int_0^{\infty}\!\!\left[
\frac{\Gamma_1 \bigl(e^{-\lambda (2h_1 - |z-z_s|)}
      + e^{-\lambda (2h_1 - z - z_s)}
      + e^{-2\lambda h_1}(e^{-\lambda |z-z_s|}
      + e^{-\lambda (z+z_s)})\bigr)}
     {1 - \Gamma_1 e^{-2\lambda h_1}}
- K_1 e^{-\lambda (2 h_1 - z - z_s)}\right] J_0(\lambda s)\,
d\lambda,
$$

decays on the strictly positive length scale

$$
d_{\text{rem}} = \min\bigl(2 h_1 - |z - z_s|,\;
(2 h_1 - z - z_s) + \Delta\bigr), \qquad
\Delta = \begin{cases}
2 h_1 & n = 2\\
2 \min(h_1, h_2) & n \ge 3
\end{cases}
$$

— even in the corner where both points sit *on* the interface
($z + z_s \to 2 h_1$), which is why the $K_1$ term is extracted
as well. The discarded tail is then bounded by
$e^{-\lambda_{\max} d_{\text{rem}}}$, i.e. **controlled** — provided
the panel grid actually resolves $[0, \lambda_{\max}]$. When it
cannot (see [Non-convergence is loud](#non-convergence-is-loud-0150)
below) that bound does not hold and the engine says so.

!!! warning "1 mm distance floor at the interface"
    Distances are floored at 1 mm (`_MIN_DISTANCE`), the same
    convention as the homogeneous [`image`](image.md) backend. The
    floor is applied to the **analytic**
    $K_1 / \sqrt{s^2 + d_{\text{int}}^2}$ term with
    $d_{\text{int}} = 2 h_1 - z - z_s$, but *not* to the matching
    $K_1 e^{-\lambda d_{\text{int}}}$ subtraction inside the
    numerically integrated residual, so for
    $\sqrt{s^2 + d_{\text{int}}^2} < 1\,\text{mm}$ the split is no
    longer algebraically exact and the reflected part saturates at the
    floor. This is not a sign error, but it is much sharper than a
    1 mm geometric floor suggests:
    `_reflected_integral(s=0, dz=0, z+z_s=2(h_1-10^{-5}))` returns
    **667.561** where the exact reflected series is **33334.23**
    (−98 %); at $d_{\text{int}} = 2\,\text{mm}$ the same call is
    exact to $10^{-9}$. The only route there is the $n \ge 3$
    diagonal with a segment midpoint within 0.5 mm of the interface
    ($s = 0$, $z = z_s$). The saturation is logged at `WARNING`
    level; keep segment midpoints a few millimetres clear of $h_1$.

### Quadrature truncation and $\lambda$ grid

$\lambda_{\max} = \texttt{lambda\_max\_factor} / d_{\text{rem}}$ —
the decay length of the *remainder*, not (as before 0.15.0) a
characteristic length of the geometry
$\min(h_1, s + z + z_s)$. The old bound ignored
$|z - z_s|$ altogether and truncated the largest off-diagonal
entries of $\mathbf{Z}$ by up to 45 % (two segments 2 cm apart at
3 m depth: 27.6997 instead of 50.1661), an error that *grew* under
mesh refinement. The same omission affected the $n \ge 3$ diagonal
correction, whose true decay scale is $2 (h_1 - z_i)$: at
$z_i = 1.995\,\text{m}$ in a $h_1 = 2\,\text{m}$ layer the
reflected self-term came out 36 % low.

`lambda_max_factor` saturates at 60 ($e^{-60} \approx 10^{-26}$):
raising the knob beyond the default can no longer change the result.
Before 0.15.0 raising it made things *worse* — 200 → −1.4 %,
20000 → −57 % with a flipped sign — because `quad`'s `limit=400`
subdivision cap could not resolve the additional Bessel half-waves.
The knob is settable through `Engine.sommerfeld_lambda_max_factor`
since 0.15.0; **its unit differs on the delegated $n = 2$
cross-layer path** — see the warning box below.

The remainder is integrated with fixed-order (12-point)
Gauss–Legendre panels on a grid built from three families:

- a quadratically graded base grid on $[0, \lambda_{\max}]$ —
  dense near $\lambda = 0$, where the fast exponentials
  $e^{-\lambda (2 h_1 + z + z_s)}$ and the variation of
  $\Gamma_1(\lambda)$ live;
- a **geometric (log-spaced) family** below the innermost base panel,
  reaching down to a quarter of the $\lambda$-scale
  $\lambda_{\text{pole}} = (1 - \Gamma_1(0)) / 2 h_1$ of the
  multiple-reflection multiplier
  $1/(1 - \Gamma_1 e^{-2\lambda h_1})$, at 8 panels per decade. For
  $|K_1| \to 1$ that scale collapses towards $\lambda = 0$
  ($2.5 \cdot 10^{-9}\,\text{m}^{-1}$ at
  $\rho_2/\rho_1 = 10^8$, $h_1 = 2\,\text{m}$) and the quadratic
  grading misses it completely — see
  [Convergence and cost](#convergence-and-cost);
- extra edges at half-period spacing $\pi / s$ up to
  $\lambda\, d_{\text{rem}} = 34$ (envelope $\sim 10^{-15}$),
  which resolves the $J_0$ oscillation the way Lucas & Stone
  (1995) prescribe — 12 Gauss nodes per half period integrate a
  Bessel half-wave to machine precision.

The grid is refined (×2, ×4, ×8) until two successive values agree
within $\max(\texttt{epsabs}, \texttt{epsrel} \cdot |G|)$, so the
tolerance arguments are now meaningful. All abscissae of one panel
family are evaluated in a single vectorised call, which is why the
engine is *faster* than before despite the higher accuracy (a 32-
segment ring on a 2-layer soil: 121 s → 1 s).

### Oscillation panel budget and the refinement ladder

The half-period family needs $\lambda_{\max}\, s / \pi$ panels, so
its size grows with $s / d_{\text{rem}}$. It is bounded by
`max_osc_panels` (`Engine.sommerfeld_max_osc_panels`, default
200 000 panels = 2.4 M abscissae per grid), which binds once
$s / d_{\text{rem}} \gtrsim 1.9 \cdot 10^{4}$.

Up to 0.15.0 the budget was spent the wrong way round. The step was
$\pi / (s \cdot \texttt{refine})$ and the panel *count* was
truncated at the cap, so the resolved range
$n_{\text{osc}} \cdot \text{step} =
\texttt{max\_osc\_panels} \cdot \pi / (s \cdot \texttt{refine})$
**shrank** by the refinement factor: rung ×8 of the ladder covered
one eighth of what rung ×1 covered, and since the loop returns the
*last* value, refining made the answer worse. The budget is now spent
**coverage first**:

1. while full coverage of $[0, \min(\lambda_{\text{osc}},
   \lambda_{\max})]$ fits in the budget, the *refinement multiplier*
   is reduced to the largest value that fits — coverage stays
   complete;
2. only if not even one panel per half period fits does the covered
   range become $\texttt{max\_osc\_panels} \cdot \pi / s$, and it is
   then **independent of `refine`**.

The covered $\lambda$-range is therefore monotonically
non-decreasing along the ladder.

### Non-convergence is loud (0.15.0)

If the ladder is exhausted without two successive grids agreeing
within $\max(\texttt{epsabs}, \texttt{epsrel}\cdot|G|)$, the engine
raises a
`groundfield.solver.mom_sommerfeld.SommerfeldConvergenceWarning`
naming the achieved versus the requested tolerance, the offending
$(s, |z - z_s|, z + z_s)$, whether the oscillation budget bound (and
how much of the required $\lambda$-range was covered), the fact that
the **sign may be wrong**, and which parameter to raise. The
$O(N^2)$ matrix assembly aggregates all failures into a single
warning naming the worst pair and the number of affected entries.

Up to 0.15.0 this condition was only `logger.debug`, i.e. silent at
Python level, **and** the least accurate rung of the ladder was the
one returned. The combination reproduced the original F04 failure
mode in the very engine that is the package's designated reference.
Measured on `LayerStack(rhos=[100, 1000, 50], h=[2, h_2])` with
$z = z_s$, checked against the far-field asymptote
$G \to (2/s)\,\rho_3/\rho_1$:

| $h_2$ [m] | $z$ [m] | $s$ [m] | 0.14.x | 0.15.0 | exact | warned? |
|---|---|---|---|---|---|---|
| 0.01 | 2.0 | 2000 | **−9.4749e-05** (−119 %, wrong sign) | +5.0139e-04 (+0.28 %) | 5.000e-04 | yes |
| 0.001 | 2.0 | 200 | **−9.4761e-04** (−119 %, wrong sign) | +5.0138e-03 (+0.28 %) | 5.000e-03 | yes |
| 0.01 | 2.0 | 500 | 1.92531e-03 (−3.7 %) | 1.999995e-03 (−2.6e-6) | 2.000e-03 | no |
| 0.01 | 1.99 | 1000 | 9.58002e-04 (−4.2 %) | 9.999994e-04 (−6.4e-7) | 1.000e-03 | no |

The two rows that still warn are genuinely budget-limited: raising
`Engine.sommerfeld_max_osc_panels` to `1_200_000` makes the first row
converge silently to 4.9999992e-04 (−1.5e-7) in 19 s instead of 2 s.
Across the 1260 in-envelope parameter sets of the accuracy table
below the ladder converged in 2–3 rungs every time, so escalating the
diagnostic from `DEBUG` to a warning does not fire in normal use.

<!-- skip-doctest: illustrative fragment — ``world`` is not defined here -->

```python
import warnings
import groundfield as gf
from groundfield.solver.mom_sommerfeld import SommerfeldConvergenceWarning

engine = gf.Engine(
    backend="mom_sommerfeld",
    segment_length=0.5,
    sommerfeld_max_osc_panels=2_000_000,   # escape hatch
    sommerfeld_epsrel=1e-9,                # tighter convergence test
)
with warnings.catch_warnings():
    warnings.simplefilter("error", SommerfeldConvergenceWarning)
    result = engine.solve(world)   # now raises instead of returning junk
```

### Accuracy knobs are reachable from the public API (0.15.0)

`Engine.solve` used to call `solve_mom_sommerfeld(world, self)` with
no keywords, so `lambda_max_factor`, `epsabs` and `epsrel` could not
be set by a user at all — a caller who hit the corner above had
nothing to tighten. They are now forwarded from four `Engine` fields,
following the `image_max_terms` / `image_series_tol` precedent:

| `Engine` field | backend keyword | default |
|---|---|---|
| `sommerfeld_lambda_max_factor` | `lambda_max_factor` | 200.0 |
| `sommerfeld_epsabs` | `epsabs` | $10^{-9}$ |
| `sommerfeld_epsrel` | `epsrel` | $10^{-7}$ |
| `sommerfeld_max_osc_panels` | `max_osc_panels` | 200 000 |

The defaults equal the previous signature defaults, so existing
callers get bit-identical results. The fields are **not** exposed
through `gf.create_engine(...)`; construct `gf.Engine(...)` directly.
All four are echoed into `FieldResult.metadata`.

!!! danger "`lambda_max_factor` has two different units"
    On the top-layer path (all segments above $h_1$) the factor is
    measured in units of $1 / d_{\text{rem}}$ — the decay length of
    the *remainder* — and saturates at 60, so raising it is harmless
    and eventually a no-op.

    On the delegated $n = 2$ **cross-layer** path
    (`sommerfeld_kernel_value` forwards to
    `coupling.layered_green.two_layer_real_space_kernel` when
    $z > h_1$ or $z_s > h_1$) the very same keyword is measured in
    units of $1 / \ell_{\text{char}}$ with $\ell_{\text{char}}$ a
    *geometric* characteristic length, is **not** saturated, and
    raising it *degrades* the answer: the Hankel grid there must
    resolve $J_0(\lambda s)$ out to $\lambda_{\max}$, so a large
    factor exhausts the node budget and triggers
    `layered_green.SommerfeldResolutionWarning`. On that path the
    documented remedy is to **lower** the factor.

    The two conventions are deliberately left unmerged — the
    delegated kernel is shared with the inductive-coupling code and
    calibrated against its own reference — but the difference is a
    real trap and is flagged at both sites in the source.

### Reaction matrix assembly

`mom_sommerfeld` builds the $N \times N$ reaction matrix in three
modes:

- **Homogeneous ($n = 1$).** Falls back on the closed-form
  homogeneous self-kernel — quadrature is unnecessary and would
  introduce numerical noise.
- **Two-layer ($n = 2$).** Uses the Tagg / Sunde self-kernel for
  the diagonal (the closed-form multi-image self-action) and the
  direct Sommerfeld quadrature for the off-diagonals. The two are
  consistent — each off-diagonal entry is the same physics
  evaluated by quadrature, and the diagonal is the closed-form
  geometric series for the same physics.
- **Multilayer ($n \ge 3$).** Uses the homogeneous line
  self-potential plus a layered self-correction obtained from the
  reflected remainder $G_{\text{refl}}$ evaluated at $s = 0$,
  $|z - z_s| = 0$: the direct $1/r$ singularity sits entirely in
  the analytically extracted part, which the line self-potential
  already accounts for, so the remainder is regular
  point-on-source. Its decay scale there is $2 (h_1 - z_i)$.

### Galerkin solve

The actual linear-system solve reuses the `_galerkin_solve` helper
from [`mom`](mom.md) — the cluster augmenting rows are identical;
only the kernel changes.

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `HomogeneousSoil`, `TwoLayerSoil`, `MultiLayerSoil` |
| Frequency | quasi-static, $f < 1\,\text{kHz}$ |
| Electrode placement, $n \le 2$ | any; the **cross-layer** case ($z > h_1$ or $z_s > h_1$) is supported and delegated to `coupling.layered_green.two_layer_real_space_kernel` (note the different `lambda_max_factor` units) |
| Electrode placement, $n \ge 3$ | every segment strictly in the upper layer — a `ValueError` is raised for $z_{\max} \ge h_1$; keep midpoints a few mm clear of $h_1$ (1 mm distance floor) |
| Layer contrast | validated $\rho_2/\rho_1 \in [10^{-2}, 10^{2}]$; accurate to $\sim 10^{-11}$ up to $10^{8}$ since 0.15.0 |
| Quadrature tolerances | $\text{epsabs} = 10^{-9}$, $\text{epsrel} = 10^{-7}$ (`Engine.sommerfeld_epsabs` / `_epsrel`) |
| Panel rule | 12-point Gauss–Legendre, 96 graded panels + 8 log panels/decade at the multiplier pole + one panel per $J_0$ half period |
| $\lambda_{\max}$ factor | 200, saturating at 60 / $d_{\text{rem}}$ (`Engine.sommerfeld_lambda_max_factor`) |
| Oscillation panel budget | 200 000 (`Engine.sommerfeld_max_osc_panels`); binds for $s / d_{\text{rem}} \gtrsim 1.9\cdot10^{4}$ |
| Mesh size $N$ | practical limit $\sim 300$ segments at acceptable runtime — **but see the cost table below**, which can move that limit down by three orders of magnitude |

The runtime cost of the engine grows as $O(N^2)$ remainder
quadratures. In the ordinary regime each costs one vectorised
evaluation of a few thousand abscissae (≈ 1–3 ms), so a 32-segment
ring on a 2-layer soil takes ≈ 1 s (0.14.1: 121 s), $N = 200$ ≈ 40 s
and $N = 1000$ of the order of a quarter hour. This is acceptable for
cross-validation but not for production sweeps — use `image_2layer` /
`cim` / `bem` there.

The per-pair cost is **not** bounded by those few thousand abscissae.
It scales with $s / d_{\text{rem}}$, the number of $J_0$ half
periods that have to be resolved, and $d_{\text{rem}}$ collapses to
$2\min(h_1, h_2)$ as soon as an intermediate layer is thin.
Measured (summed over the refinement rungs actually used, one kernel
evaluation each):

| Geometry | abscissae | wall time |
|---|---|---|
| `TwoLayerSoil(100, 500, h_1=2)`, $s = 5$, $z = z_s = 0.8$ | 3 924 | 1.3 ms |
| same, $s = 50$ | 8 316 | 2.2 ms |
| `[100, 1000, 50]`, $h = [2, 1]$, $s = 50$, $z = z_s = 1.9$ | 12 276 | 2.6 ms |
| `[100, 1000, 50]`, $h = [2, 0.01]$, $s = 50$, $z = z_s = 2$ | 977 532 | **488 ms** |
| same, $s = 500$ | 4 803 516 | **2.13 s** |
| same, $s = 2000$ (budget-capped, warns) | 9 617 340 | **3.25 s** |

Trigger: a **centimetre-scale intermediate layer** ($n \ge 3$) with
both points near the interface and a horizontal separation of order
$10^2$–$10^3\,\text{m}$. At 0.5–3 s per pair an $O(N^2)$ assembly
for a 60-segment mesh is 30 min – 3 h, and raising
`sommerfeld_max_osc_panels` to escape the accuracy warning scales that
up linearly. In such a soil do not use this engine for a solve — use
it pointwise, on the pairs you actually need.

## Convergence and cost

- **Quadrature accuracy.** Measured against the Tagg / Sunde image
  series for $n = 2$ over $\rho_2/\rho_1 \in [0.2, 10]$,
  $h_1 \in [1, 5]\,\text{m}$ and seven $(z, z_s)$ combinations
  including $z = z_s = h_1$ — 1260 parameter sets in total:

  | radial range | worst relative error | 99th pct | median |
  |---|---|---|---|
  | $s \in [0.05, 50]\,\text{m}$ (stated envelope) | 3.4e-15 | 2.0e-15 | 0 (bit-exact) |
  | $s \in [100, 200]\,\text{m}$ | 8.7e-15 | 3.9e-15 | 2.2e-16 |

  The previously documented "$10^{-14}$ level" was measured on the
  analytic split *alone* and was about 10× optimistic there (worst
  1.1e-13 inside the envelope, 1.4e-13 at $s = 200$, median 1.4e-16);
  the pole-resolving log panel family added in 0.15.0 brings it back
  under $10^{-14}$ with margin. Do not read the table as a guarantee:
  it is a measurement on a grid, and the median being exactly 0 means
  the *reference* series and the quadrature agree bit for bit on most
  of it.
- **Extreme layer contrast.** Outside the soil range the accuracy no
  longer degrades the way it did up to 0.15.0. Measured at
  $s = 0.5$, $z = z_s = 0.8$, $h_1 = 2$ against the resummed
  image series:

  | $\rho_2/\rho_1$ | before (split only) | 0.15.0 |
  |---|---|---|
  | $10^{2}$ | −2.2e-16 | −2.2e-16 |
  | $10^{4}$ | −1.5e-13 | −1.5e-13 |
  | $10^{6}$ | −3.4e-4 | −7.7e-13 |
  | $10^{8}$ | −5.2e-2 | −2.8e-11 |

  The $10^{6}$ / $10^{8}$ errors came from the
  $\lambda$-scale $(1 - \Gamma_1(0)) / 2 h_1$ of the
  multiple-reflection multiplier
  $1/(1 - \Gamma_1 e^{-2\lambda h_1})$, which the quadratically
  graded base grid cannot reach ($2.5\cdot10^{-9}\,\text{m}^{-1}$ at
  $10^{8}$ versus an innermost base panel of
  $1.6\cdot10^{-3}\,\text{m}^{-1}$). The geometric panel family now
  covers it. Note that these were returned **after** the internal
  convergence test had already failed and logged at `DEBUG` — they
  would now warn.
- **Per-pair cost.** ≈ 4000–13000 abscissae per (field, source) pair
  in the ordinary regime (1–3 ms), evaluated vectorised. The count
  scales with $s / d_{\text{rem}}$ (the number of $J_0$ half
  periods that have to be resolved) and reaches $10^{6}$–$10^{7}$
  abscissae / 0.5–3 s per single pair for a centimetre-thin
  intermediate layer — see the cost table in
  [Validity envelope](#validity-envelope).
- **Reduction.** At $\Gamma_1 \to 0$ — including a 2-layer stack
  declared with $\rho_1 = \rho_2$, where $K_1 = 0$ — the
  reflected remainder is identically zero and the engine returns
  $1/r + 1/r_{\text{img}}$ bit-exact. A 25 m ring at
  $z = 0.8\,\text{m}$ in `TwoLayerSoil(100, 100, h_1=1)` then
  reproduces `image` and `mom` to the last bit (1.5750513784 Ω;
  0.14.1 gave 1.4952 Ω, −5.1 %). With a real contrast
  (`TwoLayerSoil(100, 500, h_1=2)`) the same ring agrees with `mom`
  and `image_2layer` to 4e-8 (0.14.1: −5.7e-4).

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| `image` ($n = 1$) | bit-exact | quadrature short-circuits to closed form |
| `image` ($n = 2$, $\rho_1 = \rho_2$) | bit-exact | $\Gamma_1 \equiv 0$: the remainder vanishes identically |
| `mom` ($n \le 2$) | $\lesssim 10^{-7}$ | **same** Galerkin scheme, kernel by quadrature instead of the closed-form series — this is the sharpest kernel-only comparison |
| `image_2layer` ($n = 2$) | $\le 5\,\%$ | quadrature reproduces the geometric series; the residual is the different discretisation scheme, not the kernel |
| `cim` (any $n$) | $\le 5\,\%$ | matrix-pencil fit accuracy |
| `bem` (any $n$) | $\le 5\,\%$ | collocation on the same physics |

The engine's role is the **absolute reference**. Whenever the
closed-form layered engines disagree, the disagreement is measured
against `mom_sommerfeld`. The cross-engine validation harness report agreement tables relative to this engine.

## References

- **Sommerfeld, A.** (1909). Über die Ausbreitung der Wellen in der
  drahtlosen Telegraphie. *Annalen der Physik* 28. The original
  paper introducing the integral.
- **Watson, G. N.** (1944). *A Treatise on the Theory of Bessel
  Functions*, 2nd ed., §13.2. Lipschitz' integral
  $\int_0^\infty e^{-\lambda d} J_0(\lambda s)\,d\lambda
  = (s^2 + d^2)^{-1/2}$ — the identity behind the analytic
  extraction.
- **Lucas, S. K. & Stone, H. A.** (1995). Evaluating infinite
  integrals involving Bessel functions of arbitrary order.
  *J. Comput. Appl. Math.* 64. Panel quadrature between the zeros of
  $J_0$, the strategy used for the oscillatory factor.
- **Zou, J., Du, X. & Zhou, C.** (2015). Fast calculation of the
  Green function of a point current source in a horizontal layered
  soil with a new complex path. *IEEE Trans. Magn.* 51(3). A
  deformed complex contour is an *alternative* way of taming the
  Bessel oscillation; it is not implemented here — the analytic
  extraction above removes the conditionally convergent part
  instead.
- **Dwight, H. B.** (1936). Calculation of resistances to ground.
  Reference DC resistances used in the cross-validation tests.

## Example

```python
import groundfield as gf

soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
world = gf.create_world(soil=soil)
gf.create_electrode(world, "rod", name="g1",
                    position=(0.0, 0.0, 0.0), length=1.5)
gf.create_source(world, attached_to="g1", magnitude=1.0)

# Reference engine — slow but methodologically independent.
engine = gf.create_engine(backend="mom_sommerfeld",
                          segment_length=0.1,
                          frequencies=[50.0])
result = world.solve(engine)
print(result.cluster_impedance("g1")[0])
```

## API reference

::: groundfield.solver.mom_sommerfeld

## Related material

- ADR-0002 — engine selection heuristic; this engine is the
  reference for the layered family.
