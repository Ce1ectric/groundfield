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
Galerkin integrals of the [kernels page](../kernels.md), and so are the
images of order $n \ge 1$: exact for image pairs closer than six segment
lengths, blended into the far form up to 7.5, and beyond the point form
with the quadrupole term of the segment average — tapered off between 60
and 120 segment lengths of vertical distance, where it falls below the
remainder of the far form (since wave 3 of the 2026.11 work; before, far
images were point sources, which `Engine.image_far_field = "point"` keeps,
see below). The exact integral of an image pair takes the image's
potential at the transverse distance $\max(\rho_\perp, a)$ of the wire from
the image's axis, its average over the wire's circumference (audit
W3A-02; the reduced kernel of direct pairs left out $a^2/(2d^2)$ of an image
$d$ away). A straight strip then reproduces its uniform-current resistance
at any segment length: a 10 m strip at 0.5 m in 1000 / 50 Ω·m with
$h_1$ = 1 m within 3.6e-6 from 2 m to 0.125 m segments, not monotonically
(before wave 3 up to +5.1e-4; about 2e-6 of the rest is the reduced kernel
the soil-surface image keeps); 0.1 m above $h_1$ within 3.8e-6 (+6.8e-5
with the reduced kernel on the images); a ring 1 cm above a 1000 / 30 Ω·m
interface within 0.004 % of its exact thin ring (+1.60 %); the surface
potential above the strip within 8e-6 of the closed-form image sum (before
up to 0.94 %: the alternating series of a more conductive lower layer
amplifies the error of its far images). The point kernel sits about 1–2 %
low, like in homogeneous soil.

### Far images in point form

`Engine(image_far_field="point")` (the author's decision of 2026-11-15)
evaluates the far images as point sources, as before wave 3, and on the
cross-layer path the line form of the layered correction for the interface
image of pairs in one layer only; the continuous switch and the transverse
distance of near image pairs stay. It saves the far-image work (cost on
the [kernels page](../kernels.md#cost)) and moves networks by 2e-5 … 1.8e-4
against the default, single wires near the interface by up to 4e-4 and the
surface potential above them by up to 0.8 %. `metadata["series"]` records
the form, and the post-solve field evaluation and `solve_mutual_*` use it.

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
$2\rho_1\rho_2/(\rho_1+\rho_2)$, which keeps the matrix reciprocal. With
the line kernel $\Delta Z$ is in line form as well: in two-layer soil it is
an exact superposition of images in every layer combination (Sunde), and
its images get the treatment of the Tagg/Sunde images above — the far
quadrupole term over all pairs, the Galerkin integral for pairs within
six segment lengths blended up to 7.5, the Galerkin self images on the
diagonal — on top of the spectral values between midpoints (since wave 3
of the 2026.11 work; before, only the interface image of near pairs in one
layer was a Galerkin integral, which `image_far_field="point"` keeps). A
rod through the interface then no longer depends on the segment length
(6 m from 0.6 m through $h_1$ = 5 m in 500 / 100, 1000 / 30 and
30 / 1000 Ω·m: within 6.7e-7 from 2 m to 0.125 m segments, before
1.3e-4 … 2.4e-4; within 1.4e-6 of the exact two-owner value), and for
wires in the upper layer, where both paths apply, the cross-layer path
reproduces the series to 1.8e-7 (before up to 6.2e-4).

Its cost: one spectral evaluation per pair of segment depths. Large depth
groups (all house grids at one depth, for example) are planned once and
evaluated in row chunks, so the path needs the memory of the series path —
the reaction matrix, $16N^2$ bytes, within the budget and row blocks beyond
it (finding 10.56; before, 41 instead of 18 bytes per segment pair for 16
house grids below $h_1$, and no row blocks). The boundary-value system of
the spectral amplitudes is eliminated in closed form, vectorised over the
quadrature nodes (one LAPACK solve per node before: half the time of the
path).

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
adversarial audit, A2S-01.)

The diagonal of a segment is the layered correction at the wire radius,
$s = a$, with the interface image exchanged for its Galerkin integral. Its
closed-form $1/R$ terms are evaluated at that radius: the default
regularisation of `layered_green` clamps them at $h_1/200$ (25 mm for
$h_1 = 5$ m), meant for coincident points, and took 68–84 % of the
layered correction of every piece within $h_1/400$ of the interface — a
ring of radius 3 m at $h_1 = 5$ m read −5.29 Ω, the contact values of a
rod tip were 10–32 % low (third adversarial audit, A3S-01). The diagonal
now equals the image solution to $10^{-13}$, a ring is continuous through
$h_1$ (4.1826 Ω 1 µm above, 4.1805 Ω at $h_1$), and pairs closer than
$h_1/200$ are regularised at the 1 mm of the point forms they combine
with.

!!! note "Rod ends at the interface: a limit of the thin-wire model"

    The thin wire has no end face. A rod that ends exactly at $h_1$ is a
    wire entirely in the upper layer; one that reaches past it by a
    fraction of its radius touches the lower layer through a short piece
    only. The solid rod is continuous in between, because its flat end
    face carries the contact. Against an axisymmetric finite-volume model
    of the solid rod (4.5 m, $a$ = 8 mm, 1000 / 30 Ω·m) the line kernel
    reads:

    | tip below $h_1$ | −16 mm | −8 mm | −1 mm | 0 | +1 mm | +4 mm | +8 mm | +16 mm |
    |---|---|---|---|---|---|---|---|---|
    | solid rod | 190.2 Ω | 188.9 Ω | 182.8 Ω | 161.4 Ω | 156.3 Ω | 148.5 Ω | 141.2 Ω | 130.2 Ω |
    | thin wire | +3.3 % | +3.9 % | +7.2 % | +21.4 % | −4.6 % | −1.2 % | +0.7 % | +2.1 % |

    The point kernel and `mom` behave alike (+19.0 … +20.5 % at 0). Every
    segment backend therefore warns with `InterfaceContactWarning` when a
    rod head or tip lies within one wire radius of an interface, and
    `diagnose()` lists it under `interface_contact`. Keep rod ends a few
    wire radii from an interface, or bracket the result by moving the end
    by ±2 wire radii.

!!! warning "Point kernel next to the interface"

    With `kernel="point"` (and in the `mom` family, which shares the point
    form) a horizontal wire just below or above $h_1$ needs segments much
    shorter than twice its distance to the interface: a ring of radius 3 m
    5 cm below $h_1$ gives 29.7 Ω at ds = 0.5 m against 54.7 Ω at
    ds = 0.025 m, while the line kernel gives 55.6–55.7 Ω at every segment
    length (`tests/test_cross_layer_line_kernel.py`). Above $h_1$ the
    point form takes the image of a segment in the interface as a point
    source: a ring 1 µm / 1 cm above $h_1 = 5$ m read −1931 / −57.9 Ω
    (line kernel 4.18 / 15.4 Ω). On either side of the interface a wire
    closer than 0.387 segment lengths therefore raises
    `BackendCapabilityError` and one within 0.6 segment lengths warns with
    `PointKernelInterfaceWarning`. The default line kernel needs no such
    care.

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Soil | two distinct layers after canonicalisation; $\rho_1 = \rho_2$ is solved as homogeneous | `BackendCapabilityError` for three distinct layers (use `mom_sommerfeld`) |
| Contrast | $\lvert K\rvert < 1$; the series length grows like $\ln(\text{tol}) / \ln\lvert K\rvert$ | `SeriesTruncationWarning` at `image_max_terms` |
| Electrodes | anywhere in the soil; below or through $h_1$ via the Sommerfeld path (slower, see [performance](../performance.md)) | — |
| Rod ends | head and tip more than one wire radius from $h_1$ (thin wire without end face, see above) | `InterfaceContactWarning` |
| Point kernel near $h_1$ | horizontal wires at least 0.6 segment lengths from $h_1$, on either side | `PointKernelInterfaceWarning`; within 0.387: `BackendCapabilityError` |
| Thin wire, burial, shallow segments | as for [`image`](image.md#validity-range) | `ValueError` / `ShallowSegmentWarning` (point kernel) |
| Field quantities near grids, stars, strip ends | uniform-current ansatz, see [`image`](image.md#field-quantities-and-the-uniform-current-ansatz) | `UniformCurrentFieldWarning` in the safety helpers |
| Frequency | quasi-static, $f < 1$ kHz | — |

## Validation

<!-- generated:begin image_2layer -->
Generated from the benchmark run measured on 2026-11-14 with groundfield 2026.11.0 (Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, numba installed: yes; Linux x86_64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

| Benchmark | Measure | Bound | Worst measured | Cases | Status | Note |
|---|---|---|---|---|---|---|
| B6 Top-layer electrodes, two-layer soil | abs(Z/Z_ref - 1) | 1 % | 0.37 % | 10 | pass | uniform-current ansatz |
| B7 Rod through the layer interface | abs(Z/Z_ref - 1) | 2 % | 0.38 % | 2 | pass | one uniform current across the interface (F02); bound met since the fix of 10.15 |
| B8 Two-port reciprocity | abs(Z_AB/Z_BA - 1) | 1e-7 | 7.3e-9 | 2 | pass |  |
| B9 Limits of the layered kernels (`equal_rho_integral`) | abs(Z/Z_ref - 1) | 1e-10 | 0 | 2 | pass | image_2layer, mom, mom_sommerfeld, cim, bem |
| B9 Limits of the layered kernels (`layer_limit_integral`) | abs(Z/Z_ref - 1) | 0.1 % | 5.1e-4 | 2 | pass | physical limit |
| B11 Grid refinement | abs(Z(ds)/Z(ds/2) - 1) | 0.6 % | 0.54 % | 2 | pass | not monotone (point kernel); 0.43 % / 0.39 % with one owner for the grid before user decision Q30 |

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
