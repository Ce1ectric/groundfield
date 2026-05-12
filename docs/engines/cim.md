# `cim` — Complex Image Method

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
potential is real (the implementation takes the real part to
suppress numerical residue).

## Numerical strategy

### Matrix-pencil fit

We approximate $\Gamma_1(\lambda)$ on a uniform sample grid
$\lambda_j = \lambda_{\min} + j \Delta$, $j = 0, \dots, N_s - 1$ by
$P$ complex exponentials using the **matrix-pencil method** (Sarkar
& Pereira 1995). The procedure:

1. Sample $g_j = \Gamma_1(\lambda_j)$ on the uniform grid.
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
   $a_k$ on the original samples.

The fit is **adaptive** in the sense that the SVD truncates poles
whose singular value falls below $10^{-10}$ relative to the
dominant one, so the effective $P$ may be smaller than the
requested target. This handles the degenerate case $\Gamma_1 \equiv 0$ (homogeneous soil) gracefully — the fit returns
$P = 0$ and the engine collapses to the homogeneous closed form
exactly.

### Sample grid choice

The grid is uniform on $\lambda \in [\lambda_{\min}, \lambda_{\max}]$
with bounds expressed as multiples of $1 / h_{\min}$:

- $\lambda_{\min} = 10^{-3} / h_{\min}$ (well below the band where
  $\Gamma_1$ has structure),
- $\lambda_{\max} = 50 / h_{\min}$ (well above the decay scale of
  every interface).

This is a single-segment fit. Dan et al. 2021 propose a
*segmented-sampling* variant in which the $\lambda$-axis is split
into pieces and a separate fit is run on each. For the AP1
contrast and layer-count range a single-segment fit with $P = 8$
is sufficient (sub-percent residual); the segmented variant becomes
attractive when $n \gtrsim 5$ or contrasts are extreme.

### Self-action strategy

For $n = 1$ the fit returns $P = 0$ and the self-action falls back
exactly on the homogeneous self-kernel.

For $n = 2$ the engine **deliberately bypasses the matrix-pencil
fit** and reuses the closed-form Tagg / Sunde self-kernel of
[`image_2layer`](image_2layer.md). The reason is numerical
conditioning: $\Gamma_1 \equiv K_1$ is a constant in $\lambda$, so
the matrix pencil sees zero variation in the Hankel block and would
have to fit a single pole on the unit circle. This is
ill-conditioned and would produce $P = 1$ with a near-singular pole
at $\beta = 0$. Falling back to the geometric series gives a
bit-exact match at zero numerical risk.

For $n \ge 3$ the genuine matrix-pencil fit runs. The result
metadata exposes `cim_n_images` (the effective $P$) and
`cim_rms` (the residual on the sample grid) for diagnostics.

### Reaction matrix

After the fit, the homogeneous self-kernel handles the direct +
air-mirror part; the layered correction is a sum over the $P$
complex images, evaluated with point-source kernels at every (field,
source) segment pair. Cluster constraints are enforced through the
same multi-port system as in [`image`](image.md).

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `HomogeneousSoil`, `TwoLayerSoil`, or `MultiLayerSoil` |
| Frequency | quasi-static, $f < 1\,\text{kHz}$ |
| Electrode placement | every segment must lie inside the upper layer |
| Number of images $P$ | 4–12 typical; 8 default |
| Layer-contrast range | $|K_i| \le 0.95$ verified; harder contrasts may need more images |
| Number of layers | tested up to $n = 8$ (Dan et al. report) |

## Convergence and cost

- **Fit accuracy.** With $P = 8$ on a 64-sample grid, the
  $\Gamma_1$ residual stays below $10^{-3}$ for AP1 contrasts. The
  cluster-impedance error in the final result is typically smaller
  than the fit residual by an order of magnitude (the spatial
  integration averages out the spectral residual).
- **Per-segment cost.** $O(N^2 \cdot P)$ kernel evaluations, plus
  $O(N_s P^2)$ for the fit (one-shot, dominated by the SVD). Total
  cost is a small constant multiple of the homogeneous engine, and
  is independent of $n$.
- **Reduction.** At $n = 1$ the fit gives $P = 0$ and the engine
  collapses to `image` exactly. At $n = 2$ the engine reuses the
  exact Tagg / Sunde kernel — bit-exact match with `image_2layer`.

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| `image` ($n = 1$) | bit-exact | fit returns $P = 0$ |
| `image_2layer` ($n = 2$) | bit-exact | engine reuses the Tagg / Sunde kernel |
| `mom_sommerfeld` ($n \ge 3$) | $\le 5\,\%$ | quadrature is the reference |
| `bem` (any $n$) | $\le 5\,\%$ | shares the CIM kernel for $n \ge 3$ |
| Three-layer collapse $\rho_1 = \rho_2 = \rho_3$ | $\le 2\,\%$ | engine reduces to the 1-layer case via the empty-fit shortcut |

The engine's cross-validation role is the **fast layered solver**:
it covers the $n \ge 3$ regime where `image_nlayer` refuses, and is
checked against `mom_sommerfeld` for absolute correctness.

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

## Related material

- API reference: `groundfield.solver.cim`,
  `groundfield.solver.cim.fit_complex_images`,
  `groundfield.solver.cim.ComplexImageFit`.
- ADR-0002 — engine selection heuristic.
- Notebook `05_cim.ipynb` — visualises the matrix-pencil fit on a
  3-layer stack and exercises the engine on a two-bonded-rod
  fixture under multiple soil contrasts.
