# `image_2layer` — two-layer soil

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | `TwoLayerSoil` (also a `MultiLayerSoil` whose canonical stack has two layers); `backend="image"` forwards here |
    | Green's function | Tagg / Sunde image series inside a layer; the Sommerfeld kernel of ADR-0007 as soon as an electrode reaches the interface |
    | Current model | one uniform leakage current per unit length per electrode and soil layer |
    | Kernel | exact line source (default), `kernel="point"` reproduces 0.15.0 |
    | Status | default for two-layer soil |

The two-layer soil — an upper layer of resistivity $\rho_1$ and thickness
$h_1$ over a half-space $\rho_2$ — is the simplest soil that captures the
first-order effect of stratification (a dry or frozen top layer over wet
ground, weathered soil over rock) without the identification problem of
more layers. It is the soil model of the AP1 catalogue. `image_2layer`
solves it with the classical image series and the uniform-current model of
[`image`](image.md); everything said there about the discretisation, the
network solve and the field-grade limits applies here too.

## Mathematical and physical context

The interface reflects the current field with the coefficient

$$
K = \frac{\rho_2 - \rho_1}{\rho_2 + \rho_1} \in (-1, 1).
$$

$K < 0$ (conductive subsoil) draws current downwards and lowers the
grounding impedance; $K > 0$ (resistive subsoil) pushes current into the
top layer and raises it. For two layers the reflection coefficient of the
Sommerfeld integral is the constant $\Gamma_1 = K$, so its
multiple-reflection denominator $1/(1 - K e^{-2\lambda h_1})$ expands into
a geometric series, and Lipschitz' integral
$\int_0^\infty e^{-\lambda d}J_0(\lambda s)\,\mathrm d\lambda = 1/\sqrt{s^2+d^2}$
turns every term into a point source. For a source at depth $z_s$ and a
field point at depth $z$, both in the top layer,

$$
\varphi = \frac{\rho_1 I}{4\pi}\Biggl[\frac{1}{r_0^+} + \frac{1}{r_0^-}
+ \sum_{n=1}^{\infty} K^n\Bigl(\frac{1}{r_n^{++}} + \frac{1}{r_n^{+-}}
+ \frac{1}{r_n^{-+}} + \frac{1}{r_n^{--}}\Bigr)\Biggr],
\qquad
r_n^{\sigma\tau} = \sqrt{s^2 + (z - \sigma\,2nh_1 - \tau z_s)^2},
$$

with $\sigma, \tau \in \{+1, -1\}$: four images of weight $K^n$ per order
$n$ at the depths $\pm 2nh_1 \pm z_s$, on top of the source and its air
image (Stefanescu 1930; Tagg 1964; Sunde 1968). For $K = 0$ only the
$n = 0$ pair remains and the result is that of `image` bit for bit.

### Truncation of the series

The weights decay geometrically, so the neglected tail after order $n$ is
bounded by $|K|^{n+1}/(1 - |K|)$. The series (one `ImageSeries` object
shared by the solver, `FieldResult.potential` and `solve_mutual_*`) stops
at the first $n$ whose tail bound is below `Engine.image_series_tol`
(default $10^{-6}$), or at `Engine.image_max_terms` (default 300) with a
`SeriesTruncationWarning`. The number of terms depends only on $|K|$; the
generated table in the [validation section](#validation) lists it for
typical contrasts — the default cap covers the AP1 corner soil
1000 / 30 Ω·m. `metadata["series"]` records the controls of the solve, and
the post-solve field evaluation uses the same controls (up to 0.15.0 it
was capped at 100 terms, finding 10.25).

## Electrodes in the top layer

Every segment carries the source, the air image and the image series. With
the exact line kernel (default) the direct and air-image pairs are the
Galerkin integrals of the [kernels page](../kernels.md); the images of
order $n \ge 1$ lie at least $2(h_1 - z)$ away and use the point form,
except for image pairs closer than six segment lengths (wires close to the
interface), which are integrated exactly too. For a straight strip the
result deviates from the exact uniform-current resistance by a few
$10^{-5}$ at any segment length (generated kernel check below); the point
kernel sits about 1–2 % low, like in homogeneous soil.

## Electrodes that reach the interface

A rod driven through $h_1$, or any wire at or below the interface, needs
the Green's function with source and field point in different layers. As
soon as one segment reaches the interface, `image_2layer` evaluates every
pair with the two-layer Sommerfeld kernel of
`groundfield.coupling.layered_green` (ADR-0007): segments are first split
at $h_1$; the reaction matrix is the homogeneous kernel scaled with the
resistivity of the source layer plus a layered correction $\Delta Z$
evaluated by spectral quadrature, grouped by depth. For a pair in
different layers the direct term carries the transmitted resistivity
$2\rho_1\rho_2/(\rho_1+\rho_2)$, which keeps the matrix reciprocal; the
near-singular interface image of $\Delta Z$ is integrated in Galerkin form
for near pairs (line kernel).

Physically the leakage current density of a rod through the interface
scales roughly with the conductivity of the layer it is in. Since 2026.11
such an electrode is therefore **one owner per layer** (finding 10.15), all
in the electrode's cluster — the same model as two rods bonded at $h_1$:
the current density may differ above and below the interface, the
reported potential is the length-weighted average over the whole
electrode. Up to 0.15.0 one uniform density ran across the interface,
which put a 12 m rod through $h_1 = 5$ m in 1000 / 30 Ω·m soil 566 % above
the per-segment `mom` solution (`tests/test_cross_layer_owners.py`).
Every interface cut is kept, however short the piece beyond it: the
Galerkin self term and the line kernel's parallel-pair integral are
evaluated without cancellation for such pieces, so the impedance is
continuous in the depth of a rod tip. (Earlier in 2026.11 cuts that would
leave a piece shorter than the wire diameter were dropped, which ignored
the contact with the other layer: a 4.5 m rod in 1000 / 30 Ω·m read 195.0 Ω
with its tip 15.9 mm into the lower layer and 128.6 Ω at 16.1 mm; second
adversarial audit, A2S-01.) What remains is the onset of the contact
itself, between a tip that ends at the interface and one that reaches
past it, which `mom` shows as well.

!!! warning "Point kernel next to the interface"

    With `kernel="point"` (and in the `mom` family, which shares the point
    form) a horizontal wire just below or above $h_1$ needs segments much
    shorter than twice its distance to the interface: a ring of radius 3 m
    5 cm below $h_1$ gives 29.7 Ω at ds = 0.5 m against 54.7 Ω at
    ds = 0.025 m, while the line kernel gives 55.6–55.7 Ω at every segment
    length (`tests/test_cross_layer_line_kernel.py`). The default line
    kernel needs no such care.

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Soil | two distinct layers after canonicalisation; $\rho_1 = \rho_2$ is solved as homogeneous | `BackendCapabilityError` for three distinct layers (use `mom_sommerfeld`) |
| Contrast | $\lvert K\rvert < 1$; the series length grows like $\ln(\text{tol}) / \ln\lvert K\rvert$ | `SeriesTruncationWarning` at `image_max_terms` |
| Electrodes | anywhere in the soil; below or through $h_1$ via the Sommerfeld path (slower, see [performance](../performance.md)) | — |
| Thin wire, burial, shallow segments | as for [`image`](image.md#validity-range) | `ValueError` / `ShallowSegmentWarning` (point kernel) |
| Field quantities near grids, stars, strip ends | uniform-current ansatz, see [`image`](image.md#field-quantities-and-the-uniform-current-ansatz) | `UniformCurrentFieldWarning` in the safety helpers |
| Frequency | quasi-static, $f < 1$ kHz | — |

## Validation

<!-- generated:begin image_2layer -->
Generated from the benchmark run measured on 2026-11-08 with groundfield 0.15.0 (Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, numba installed: yes; Linux x86_64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

| Benchmark | Measure | Bound | Worst measured | Cases | Status | Note |
|---|---|---|---|---|---|---|
| B6 Top-layer electrodes, two-layer soil | abs(Z/Z_ref - 1) | 1 % | 0.37 % | 10 | pass | uniform-current ansatz |
| B7 Rod through the layer interface | abs(Z/Z_ref - 1) | 2 % | 0.38 % | 2 | pass | one uniform current across the interface (F02); bound met since the fix of 10.15 |
| B8 Two-port reciprocity | abs(Z_AB/Z_BA - 1) | 1e-7 | 7.3e-9 | 2 | pass |  |
| B9 Limits of the layered kernels (`equal_rho_integral`) | abs(Z/Z_ref - 1) | 1e-10 | 0 | 2 | pass | image_2layer, mom, mom_sommerfeld, cim, bem |
| B9 Limits of the layered kernels (`layer_limit_integral`) | abs(Z/Z_ref - 1) | 0.1 % | 5.1e-4 | 2 | pass | physical limit |
| B11 Grid refinement | abs(Z(ds)/Z(ds/2) - 1) | 0.5 % | 0.43 % | 2 | pass | not monotone |

Length of the image series (independent of h₁: the number of terms depends only on |K| and the tolerance):

| ρ₁ / ρ₂ in Ω·m | K | terms for a tail bound below 1e-6 | within the default cap of 300 terms |
|---|---|---|---|
| 300 / 100 | -0.5000 | 20 | yes |
| 100 / 400 | +0.6000 | 28 | yes |
| 1000 / 100 | -0.8182 | 77 | yes |
| 500 / 30 | -0.8868 | 133 | yes |
| 1000 / 30 | -0.9417 | 277 | yes |
| 2000 / 30 | -0.9704 | 577 | no — SeriesTruncationWarning |

Kernel check — signed deviation from the exact uniform-current resistance:

| Case | Exact uniform-current R | Kernel | ds = 2 m | ds = 0.5 m | ds = 0.25 m | ds = 0.125 m |
|---|---|---|---|---|---|---|
| strip 10 m at 0.5 m, a = 5 mm, two-layer 1000/30 Ω·m, h₁ = 5 m | 129.9464 Ω | `point` | +0.88 % | −1.1 % |  | −1.3 % |
|  |  | `line` | +6.1e-5 | −2.7e-5 |  | −7.8e-7 |
<!-- generated:end image_2layer -->

## Literature

- Stefanescu, S. S., & Schlumberger, C. (1930). Sur la distribution
  électrique potentielle autour d'une prise de terre ponctuelle dans un
  terrain à couches horizontales, homogènes et isotropes. *Journal de
  Physique et le Radium* 1.
- Tagg, G. F. (1964). *Earth Resistances*. George Newnes, London, ch. 5
  (the image series in engineering form).
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission Systems*.
  Dover, New York, sect. 3.5 (Sommerfeld integral of the layered earth and
  its two-layer reduction).
- Dawalibi, F. P., & Barbeito, N. (1991). Measurements and computations of
  the performance of grounding systems buried in multilayer soils. *IEEE
  Transactions on Power Delivery* 6(4).
- Colominas, I., París, J., Navarrina, F., & Casteleiro, M. (2012).
  Improvement of computer methods for grounding analysis in layered soils
  by using high-efficient convergence acceleration techniques. *Advances in
  Engineering Software* 44 (acceleration of the series for $|K| \to 1$, not
  used here).

Architecture decisions: [ADR-0001](../../adr/0001-two-layer-method.md),
[ADR-0007](../../adr/0007-cross-layer-electrodes.md),
[ADR-0015](../../adr/0015-line-source-kernel-default.md). API:
`groundfield.solver.image_2layer.solve_image_2layer`.
