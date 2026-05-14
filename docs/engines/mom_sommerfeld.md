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
Sommerfeld integral **numerically**, point by point, with adaptive
Gauss–Kronrod quadrature. The recursive $\Gamma_1(\lambda)$ enters
the integrand as is — no expansion, no fit. The price is speed (a
single $N \times N$ reaction matrix can take seconds rather than
milliseconds), but the result is an **absolute reference**: any
disagreement between the closed-form engines and `mom_sommerfeld`
indicates a fit / series accuracy issue, not a kernel bug.

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

### Quadrature truncation

The integrand decays exponentially in $\lambda$ at the rate of the
fastest decaying exponential $\min(|z - z_s|, z + z_s, 2 h_1 - z - z_s, 2 h_1 - |z - z_s|)$. We bound the upper limit at
$\lambda_{\max} = \texttt{lambda\_max\_factor} / \bar h$ with
$\bar h = \min(h_1, s + z + z_s + \epsilon)$ — i.e. the
characteristic length scale of the geometry, and a default factor
of 200. This puts the residual integrand contribution above
$\lambda_{\max}$ at $e^{-200} \sim 10^{-87}$, well below any
practical tolerance.

Inside that bound, `scipy.integrate.quad` (Gauss–Kronrod adaptive
quadrature with a 21-point rule) handles the integration to
absolute / relative tolerances $10^{-9} / 10^{-7}$. The Bessel
function $J_0(\lambda s)$ oscillates with period $2\pi/s$; the
adaptive rule subdivides until the per-subinterval rule and a
higher-order rule agree to within tolerance, so the oscillation is
captured automatically.

### Why not the Zou et al. complex contour?

Zou, Du & Zhou (2015) proposed a deformed integration contour in
the complex $\lambda$-plane that breaks the Bessel oscillation and
makes the integrand non-oscillatory along the contour. The current
implementation uses the simpler real-axis adaptive quadrature.
Pragmatic justification:

- For $s \lesssim 100\,\text{m}$ (the typical range) the real-axis
  adaptive rule converges in 200–400 kernel evaluations per pair,
  which is acceptable for the engine's role as cross-check
  reference.
- The Zou contour requires careful handling of the branch points
  in $\Gamma_1$ for $n \ge 3$, which would make the implementation
  significantly larger.
- The cross-validation tests show that the real-axis rule already
  agrees with the closed-form engines to better than 1 % across
  the typical contrast range.

The module's docstring still cites Zou et al. as the reference for
the *idea*; switching to a complex contour is a documented future
optimisation (ADR-0002 action items).

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
  reflection-only Sommerfeld integral evaluated at $s = 0$,
  $|z - z_s| = 0$ (the integrand has no $1/r$ singularity in this
  reflection-only form, so it can be integrated point-on-source).

### Galerkin solve

The actual linear-system solve reuses the `_galerkin_solve` helper
from [`mom`](mom.md) — the cluster augmenting rows are identical;
only the kernel changes.

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `HomogeneousSoil`, `TwoLayerSoil`, `MultiLayerSoil` |
| Frequency | quasi-static, $f < 1\,\text{kHz}$ |
| Electrode placement | every segment in the upper layer |
| Quadrature tolerances | $\text{epsabs} = 10^{-9}$, $\text{epsrel} = 10^{-7}$ |
| Quadrature limit | 400 subdivisions |
| $\lambda_{\max}$ factor | 200 / characteristic length (default) |
| Mesh size $N$ | practical limit $\sim 200$ segments at acceptable runtime |

The runtime cost of the engine grows as $O(N^2)$ Sommerfeld
quadrature evaluations, each costing a few hundred kernel
evaluations. For $N = 200$ on a 2-layer world the engine completes
in a few seconds; for $N = 1000$ it would take of the order of a
minute. This is acceptable for cross-validation but not for
production sweeps — use `image_2layer` / `cim` / `bem` there.

## Convergence and cost

- **Quadrature accuracy.** With the default tolerances the
  per-pair error is $\sim 10^{-8}$ relative to the analytical
  closed forms. The cluster-impedance error in the assembled
  result stays below 1 % of the closed-form reference.
- **Per-pair cost.** Adaptive Gauss–Kronrod typically converges in
  100–400 kernel evaluations per (field, source) pair.
- **Reduction.** At $\Gamma_1 \to 0$ the engine short-circuits to
  $1/r + 1/r_{\text{img}}$ — bit-exact match with `image`.

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| `image` ($n = 1$) | bit-exact | quadrature short-circuits to closed form |
| `image_2layer` ($n = 2$) | $\le 5\,\%$ | quadrature reproduces the geometric series |
| `cim` (any $n$) | $\le 5\,\%$ | matrix-pencil fit accuracy |
| `bem` (any $n$) | $\le 5\,\%$ | collocation on the same physics |
| `mom` ($n \le 2$) | $\le 2\,\%$ | same Galerkin scheme, different kernel evaluation |

The engine's role is the **absolute reference**. Whenever the
closed-form layered engines disagree, the disagreement is measured
against `mom_sommerfeld`. The cross-engine notebooks
(`06_mom_sommerfeld.ipynb`,  `09_cross_engine_extended.ipynb`)
report agreement tables relative to this engine.

## References

- **Sommerfeld, A.** (1909). Über die Ausbreitung der Wellen in der
  drahtlosen Telegraphie. *Annalen der Physik* 28. The original
  paper introducing the integral.
- **Zou, J., Du, X. & Zhou, C.** (2015). Fast calculation of the
  Green function of a point current source in a horizontal layered
  soil with a new complex path. *IEEE Trans. Magn.* 51(3). The
  reference for a deformed contour that the current implementation
  does *not* use, but which is on the optimisation roadmap.
- **Piessens, R. et al.** (1983). *QUADPACK*, Springer. The
  Gauss–Kronrod adaptive quadrature implementation that scipy
  wraps.
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
- Notebook `06_mom_sommerfeld.ipynb` — kernel sanity checks at the
  homogeneous / $K = 0$ limits, single rod and bonded-rod
  fixtures, and a hard-contrast validation table where
  `mom_sommerfeld` is the absolute reference.
