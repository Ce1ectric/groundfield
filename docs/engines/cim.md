# `cim` — Complex Image Method

!!! warning "Status: reduced to closed-form kernels ($n \le 2$ only)"

    Since 0.11.0 `cim` **rejects** $n \ge 3$ soils with
    `NotImplementedError` (the historic complex-image kernel was
    structurally incomplete — audit 2026-07-08, WP-E), and since
    0.15.0 it no longer computes a complex-image fit at all. What runs
    is:

    | Soil | Kernel | Relation to other backends |
    |---|---|---|
    | $n = 1$ | homogeneous image-charge self-kernel | **bit-identical** to `image` |
    | $n = 2$ | exact Tagg / Sunde series self-kernel | **bit-identical** to `image_2layer` |
    | $n \ge 3$ | — | `NotImplementedError`; use `mom_sommerfeld` or `fem` |

    `cim` is therefore **not an independent cross-check** of the
    image-charge family: in a cross-validation table it must be counted
    as the *same* computation as `image` / `image_2layer`, not as a
    second engine (ADR-0002 amendment 2026-07-09). The sections below
    document the complex-image theory and
    [`fit_complex_images`](#api-reference), which remains available as a
    standalone spectral helper for a future complete $n \ge 3$
    kernel, but is **not** on any solve path.

## Physical context

The Complex Image Method (CIM) is the modern workhorse for layered
Green's functions in the grounding and antenna literature. It
trades a small, controllable approximation error for two important
properties:

- **Closed-form spatial Green's function** of the same shape as the
  homogeneous image-charge sum.
- **Cost independent of the layer count** once the fit is done.

The trick is that the only object in the layered Sommerfeld
representation that *changes* with the layer count is the recursive
reflection coefficient $\Gamma_1(\lambda)$ — every other piece is
the same exponential / Bessel kernel as in the homogeneous case. If
$\Gamma_1(\lambda)$ can be approximated by a finite sum of complex
exponentials,

$$
\Gamma_1(\lambda) \;\approx\; \sum_{k=1}^{P} a_k\,
e^{-2\lambda \beta_k},
\qquad a_k \in \mathbb{C}, \quad \Re\{\beta_k\} > 0,
$$

then substituting that approximation into the Sommerfeld integral
gives a **closed-form** spatial form via the Sommerfeld identity

$$
\int_0^{\infty} e^{-\lambda d}\, J_0(\lambda s)\, d\lambda
\;=\; \frac{1}{\sqrt{s^2 + d^2}},
\qquad \Re\{d\} > 0.
$$

The result is a $1/r$-type kernel, but with **complex image
positions** $z = -(z_s + 2\beta_k)$. Each pole $(a_k, \beta_k)$
contributes one image; the per-evaluation cost is the same as the
homogeneous backend multiplied by $P$, and is independent of $n$.

## Governing equation: complex-image kernel

Substituting the fit into the layered Sommerfeld integral gives

$$
\varphi(s, z) \;=\; \frac{\rho_1\, I}{4\pi}
\Biggl[
   \frac{1}{r} + \frac{1}{r_{\text{air}}}
 + \sum_{k=1}^{P} a_k\,
    \frac{1}{\sqrt{s^2 + (z + z_s + 2\beta_k)^2}}
\Biggr],
$$

with $r = \sqrt{s^2 + (z-z_s)^2}$, $r_{\text{air}} = \sqrt{s^2 + (z+z_s)^2}$.
The first two terms are the homogeneous direct + air-mirror; the
third is the closed-form layered correction. With complex
$\beta_k$, the square-root denominator is complex too — but the
imaginary parts cancel by symmetry of the fit, so the final
potential is real (an implementation would take the real part to
suppress numerical residue).

Two caveats, both of which is why this kernel is *not* wired into a
solver today: the form above carries only the $(z + z_s)$ image
family, and it lacks the surface-interface multiple-reflection
denominator $1 / (1 - \Gamma_1 e^{-2\lambda h_1})$ of the full
layered Green's function (see
[`_layered`](../api/solver.md) and
[`mom_sommerfeld`](mom_sommerfeld.md)). Completing it is the
prerequisite for letting `cim` accept $n \ge 3$ again.

## Numerical strategy

### Matrix-pencil fit

We approximate $\Gamma_1(\lambda)$ on a uniform sample grid
$\lambda_j = \lambda_{\min} + j \Delta$, $j = 0, \dots, N_s - 1$ by
$P$ complex exponentials using the **matrix-pencil method** (Sarkar
& Pereira 1995). The procedure:

1. Split off the asymptote (see below) and sample the *decaying*
   remainder $g_j = \Gamma_1(\lambda_j) - K_1$ on the uniform grid.
2. Form the rectangular Hankel pencil $[Y_0, Y_1]$ with $Y_0[i, k] = g_{i+k}$, $Y_1[i, k] = g_{i+k+1}$, $i = 0, \dots, N_s - L - 1$,
   $k = 0, \dots, L - 1$, with pencil parameter $L \approx N_s/3$.
3. SVD $Y_0 = U \Sigma V^*$ and project both blocks onto the
   $P$-dimensional dominant singular subspace
   ($U_P, \Sigma_P, V_P$).
4. The poles $p_k = e^{-2\Delta \beta_k}$ of the sum of
   exponentials are the eigenvalues of
   $\Sigma_P^{-1} U_P^* Y_1 V_P^*$.
5. Recover $\beta_k = -\ln p_k / (2\Delta)$. Discard poles with
   $|p_k| \ge 1$ (non-decaying / non-physical) or with
   $\Re\{\beta_k\} \le 0$.
6. Solve the linear least-squares system for the coefficients
   $a_k$ on the original samples of $\Gamma_1$, over the
   *surviving* set $\{\beta_k\} \cup \{0\}$ — i.e. **after** the pole
   filter of step 5, so that the reported weights belong to the
   reported poles.

The fit is **adaptive** in the sense that the SVD truncates poles
whose singular value falls below $10^{-10}$ relative to the
dominant one, so the effective $P$ may be smaller than the
requested target. This handles the degenerate case $\Gamma_1 \equiv 0$ (homogeneous soil) gracefully — the fit returns
$P = 0$ and the engine collapses to the homogeneous closed form
exactly.

### The $\lambda \to \infty$ asymptote must be split off

A sum of decaying exponentials vanishes for
$\lambda \to \infty$, but $\Gamma_1$ does not:

$$
\lim_{\lambda \to \infty} \Gamma_1(\lambda) \;=\; K_1
\;=\; \frac{\rho_2 - \rho_1}{\rho_2 + \rho_1} \;\neq\; 0,
$$

because every $e^{-2\lambda h_i}$ in the recursion dies and leaves
the top-interface Fresnel coefficient. The constant is carried by an
image at $\beta_0 = 0$ — i.e. at the air-mirror position
$z = -z_s$, with weight $K_1$, still integrable through the
Sommerfeld identity since $d = z + z_s > 0$ — and only
$\Gamma_1(\lambda) - K_1$ is handed to the matrix pencil. This
follows Li et al. 2006 / Dan et al. 2021.

Up to 0.14.1 `groundfield` fitted $\Gamma_1$ itself, and the pole
filters of step 5 removed precisely the constant term (pole at
$p = 1$, $\beta = 0$) that would have carried the asymptote. The
residual was therefore of order $|K_1|$ for *every* stack — e.g.
$\rho = [100, 400, 50]\ \Omega\text{m}$, $h = [2, 3]\ \text{m}$:
$|K_1| = 0.600$, reported `rms` $= 0.585$ — and a two-layer stack
(where the constant is the *whole* function) returned $P = 0$ with
`rms = nan`. With the split, the same stacks give `rms` $\approx
2 \cdot 10^{-13}$ and $0$ exactly (review pass 9, F39).

### Failure reporting

A fit that cannot represent $\Gamma_1$ no longer passes silently.
`fit_complex_images` reports

- `rms` — residual on the uniform fitting grid, **always finite**,
- `rms_extrapolated` — residual on an independent 512-point
  log-spaced verification grid spanning the full requested
  $\lambda$ range (catches a window that stopped short of the
  decay band, which the fitting-grid residual cannot see),
- `converged` — both residuals below `rms_tol` (default $10^{-3}$)
  and the pencil produced at least one usable pole,
- `k_inf` — the asymptote $K_1$ carried by the $\beta = 0$ image,

and emits a `ComplexImageFitWarning` whenever `converged` is
`False`.

### Sample grid choice

Only the thicknesses $h_2, \dots, h_{n-1}$ enter $\Gamma_1$ — the
recursion never uses $h_1$. The grid is uniform on
$\lambda \in [\lambda_{\min}, \lambda_{\max}]$ with

- $\lambda_{\min} = 10^{-3} / h_{\max}$, $h_{\max} = \max_{i \ge 2} h_i$
  (well below the band where $\Gamma_1$ has structure),
- $\Delta \le 1 / (8\, h_{\max})$, so the *slowest* exponential
  (scale $1/(2 h_{\max})$) is resolved by at least eight samples,
- $\lambda_{\max} = 50 / h_{\min}$, $h_{\min} = \min_{i \ge 2} h_i$
  (above the decay scale of the thinnest interface); with $\Delta$
  capped as above the grid may stop earlier, which is exactly what
  `rms_extrapolated` monitors.

Up to 0.14.1 *both* bounds were derived from $h_{\min}$, so for a
stack like $h = [0.5, 10]\ \text{m}$ the step was $\Delta = 1.59$
against a structure scale of $0.05$ — the grid stepped straight over
the transition and the pencil collapsed to $P = 1$.

This is a single-segment fit. Dan et al. 2021 propose a
*segmented-sampling* variant in which the $\lambda$-axis is split
into pieces and a separate fit is run on each. For the typical
contrast and layer-count range a single-segment fit with $P = 8$
is sufficient; the segmented variant becomes attractive when
$n \gtrsim 5$ or contrasts are extreme.

### Self-action strategy

For $n = 1$ the self-action is the homogeneous self-kernel
($\Gamma_1 \equiv 0$, no extra images).

For $n = 2$ the engine uses the closed-form Tagg / Sunde self-kernel
of [`image_2layer`](image_2layer.md), with
`allow_cross_layer=True` so that a rod crossing the interface
dispatches to the rigorous cross-layer path (ADR-0007). Here
$\Gamma_1 \equiv K_1$ is constant in $\lambda$, so the geometric
image series *is* the complex-image representation (the single
$\beta = 0$ image) — evaluating it in closed form is exact and
cheaper than any fit.

For $n \ge 3$ the engine raises `NotImplementedError`. **No
complex-image fit is evaluated on any solve path.** Up to 0.14.1 the
fit was computed on every solve and its diagnostics were published as
`cim_n_images` / `cim_rms` although neither reachable path consumed
them; readers were invited to judge "how good is the complex-image
approximation here" from a NaN produced by a fit with no influence on
the answer (review pass 9, F34). The metadata now carries

| Key | Value | Meaning |
|---|---|---|
| `cim_fit_used` | `False` | no complex-image fit entered this result |
| `cim_n_images` | `0` | consequently, no images |
| `cim_rms` | `None` | no fit, hence no residual (was `nan`) |
| `reduces_to` | `"image"` / `"image_2layer"` | backend this result is bit-identical to |

### Reaction matrix

The closed-form self-kernel handles direct + air-mirror + layered
correction in one call. Cluster constraints are enforced through the
same multi-port system as in [`image`](image.md).

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `HomogeneousSoil`, `TwoLayerSoil` (a `MultiLayerSoil` is accepted only while it reduces to $n \le 2$) |
| Frequency | quasi-static, $f < 1\,\text{kHz}$ |
| Electrode placement | free — $n = 2$ interface crossings dispatch to the ADR-0007 cross-layer kernel |
| Number of layers | $n \le 2$; $n \ge 3$ raises `NotImplementedError` |
| Number of images $P$ | not applicable — no fit runs (`fit_complex_images` itself: 4–12 typical, 8 default) |
| Layer-contrast range | as `image_2layer` (exact series, truncation controlled by `Engine.image_max_terms` / `image_series_tol`) |

## Convergence and cost

- **Accuracy.** Identical to `image` ($n = 1$) / `image_2layer`
  ($n = 2$) by construction — the same kernels are called.
- **Fit accuracy** (of the standalone `fit_complex_images`, not of
  the engine). With $P = 8$ decaying images on a 64-sample grid plus
  the analytic $\beta = 0$ asymptote image, the $\Gamma_1$ residual
  measured on three- and four-layer stacks with $|K_i|$ up to
  $0.9$ is $\le 5 \cdot 10^{-8}$ on the fitting grid and
  $\le 3 \cdot 10^{-4}$ on the wide verification grid. Anything
  worse raises a `ComplexImageFitWarning`.
- **Per-segment cost.** $O(N^2)$ kernel evaluations plus the
  multi-port solve — the same as `image` / `image_2layer`.
- **Reduction.** At $n = 1$ bit-exact match with `image`; at
  $n = 2$ bit-exact match with `image_2layer`.

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| `image` ($n = 1$) | bit-exact | *same computation* — the homogeneous self-kernel; no independent information |
| `image_2layer` ($n = 2$) | bit-exact | *same computation* — the Tagg / Sunde kernel; no independent information |
| `mom` / `bem` (any $n \le 2$) | $\le 5\,\%$ | uniform-current vs. Galerkin / collocation on the same kernel |
| `mom_sommerfeld` ($n = 2$) | $\le 5\,\%$ | direct quadrature of the full layered Green's function — the only genuinely independent kernel |
| `fem` ($n \le 2$) | $\le 10\,\%$ | volume PDE, independent problem form (equivalent-hemisphere bias) |

**`cim` is not an independent cross-validation engine.** For every
soil it accepts it evaluates the same closed-form kernel as the
image-charge family, so a table that lists `image_2layer` and `cim`
as two agreeing engines is reporting one computation twice. Real
independence comes from `mom_sommerfeld` (quadrature) and `fem`
(volume PDE); see the [ADR-0002
amendment](../adr/0002-engine-family.md).

## References

- **Sarkar, T. K. & Pereira, O.** (1995). Using the matrix pencil
  method to estimate the parameters of a sum of complex
  exponentials. *IEEE Antennas & Propagation Magazine* 37(1). The
  primary reference for the fit algorithm.
- **Li, Z.-X., Chen, W., Fan, J.-B. & Lu, J.** (2006). A novel
  mathematical modeling of grounding system buried in multilayer
  earth. *IEEE PWRD* 21(3). Quasi-static CIM (QCIM) for grounding
  systems; the closed-form Green's function in the form used here.
- **Dan, Y. et al.** (2021). Segmented sampling least squares
  algorithm for Green's function of arbitrary layered soil.
  *IEEE PWRD* 36(3). The segmented-sampling refinement and a
  systematic study of the accuracy versus $P$.
- **Hua, Y. & Sarkar, T. K.** (1990). Matrix pencil method for
  estimating parameters of exponentially damped/undamped
  sinusoids in noise. *IEEE Trans. ASSP* 38(5). The original
  matrix-pencil derivation.

## Example

A two-layer solve — the deepest stack `cim` accepts:

```python
import groundfield as gf

soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
world = gf.create_world(soil=soil)
gf.create_electrode(world, "rod", name="g1",
                    position=(0.0, 0.0, 0.0), length=3.0)
gf.create_source(world, attached_to="g1", magnitude=1.0)

engine = gf.create_engine(backend="cim",
                          segment_length=0.1,
                          frequencies=[50.0])
result = world.solve(engine)
print(result.cluster_impedance("g1")[0])   # (67.6864+0j) Ohm
print(result.metadata["cim_fit_used"])     # False — no fit runs
print(result.metadata["reduces_to"])       # 'image_2layer'
```

Three or more layers raise, and point at the engines that do
implement the full layered Green's function:

```python
soil = gf.MultiLayerSoil(layers=[
    gf.SoilLayer(resistivity=80.0, thickness=0.5),
    gf.SoilLayer(resistivity=300.0, thickness=2.0),
    gf.SoilLayer(resistivity=50.0),  # semi-infinite bottom
])
world = gf.create_world(soil=soil)
gf.create_electrode(world, "rod", name="g1",
                    position=(0.0, 0.0, 0.0), length=0.4)
gf.create_source(world, attached_to="g1", magnitude=1.0)

# gf.create_engine(backend="cim", ...).solve(world)
#   -> NotImplementedError: cim: n_layers = 3 >= 3 is not supported ...
engine = gf.create_engine(backend="mom_sommerfeld", segment_length=0.1)
result = world.solve(engine)
print(result.cluster_impedance("g1")[0])
```

The spectral helper can still be called directly — for instance to
check how well a stack's $\Gamma_1$ *would* be represented:

```python
import numpy as np
from groundfield.solver._layered import LayerStack
from groundfield.solver.cim import fit_complex_images

stack = LayerStack(rhos=np.array([100.0, 400.0, 50.0]),
                   h=np.array([2.0, 3.0]))
fit = fit_complex_images(stack)
print(fit.a.size, fit.k_inf, fit.rms, fit.converged)
# 8 0.6 2.1e-13 True   (index 0 is the beta = 0 asymptote image)
```

## API reference

::: groundfield.solver.cim

## Related material

- ADR-0002 — engine selection heuristic.
