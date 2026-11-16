# `image_2layer` — two-layer soil

This page explains the default engine for two-layer soil: how the layer
interface enters the Green's function, how electrodes in, at and through
the interface are treated, how accurate the results are and which
geometries next to the interface need care. Read the [`image`](image.md)
page first; everything said there about the current model, the network
solve and the field-grade limits applies here too.

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | `TwoLayerSoil`, or a `MultiLayerSoil` whose canonical stack has two layers; `backend="image"` forwards here |
    | Green's function | Tagg / Sunde image series inside a layer; the two-layer Sommerfeld kernel of ADR-0007 once an electrode reaches the interface |
    | Current model | one uniform leakage current per straight wire and soil layer (user decision Q30) |
    | Kernel | exact line source (default); `kernel="point"` reproduces 0.15.0 |
    | Accuracy | exact for its model; the current-model error of [`image`](image.md#field-quantities-and-the-uniform-current-ansatz); thin-wire limits right at the interface |
    | Status | default for two-layer soil |

## What it solves

The two-layer soil is an upper layer of resistivity $\rho_1$ and thickness
$h_1$ over a half-space of resistivity $\rho_2$. It is the simplest soil
that captures the first-order effect of stratification — a dry or frozen
top layer over wet ground, weathered soil over rock — without the
identification problem of more layers, and it is the soil model of the AP1
catalogue. `image_2layer` solves it with the classical image series and
the uniform-current model of `image`:

- **Soil.** Two homogeneous, isotropic layers with a horizontal interface
  at depth $h_1$, insulating air above; depth $z$ counts positive
  downwards.
- **Quasi-static field**, thin wires and average potential as for
  [`image`](image.md#what-it-solves).
- **Current model.** One uniform leakage current per straight wire, and
  per soil layer for a wire that crosses the interface: a rod driven
  through $h_1$ is **one owner per layer**, so its current density may
  differ above and below the interface.
- **Electrodes anywhere in the soil**: in the top layer (image series), at
  or below the interface (Sommerfeld kernel).

## Mathematical and physical context

### Reflection at the interface

The interface reflects the current field with the coefficient

$$
K = \frac{\rho_2 - \rho_1}{\rho_2 + \rho_1} \in (-1, 1).
$$

$K < 0$ (a conductive subsoil) draws current downwards and lowers the
grounding impedance; $K > 0$ (a resistive subsoil) pushes current into the
top layer and raises it. For $K = 0$ the soil is homogeneous.

### From the Sommerfeld integral to the image series

For a point current $I$ at depth $z_s$ and a field point at depth $z$ and
horizontal distance $s$, both in the top layer, the potential is the
Sommerfeld integral (the static limit of Sommerfeld's half-space problem;
the general stack is on the [`mom_sommerfeld`](mom_sommerfeld.md) page)

$$
\varphi = \frac{\rho_1 I}{4\pi}\int_0^\infty
\frac{e^{-\lambda\lvert z-z_s\rvert} + e^{-\lambda(z+z_s)}
+ K\bigl[e^{-\lambda(2h_1-\lvert z-z_s\rvert)} + e^{-\lambda(2h_1-z-z_s)}\bigr]}
{1 - K\,e^{-2\lambda h_1}}\,J_0(\lambda s)\,\mathrm d\lambda ,
$$

with the spatial frequency $\lambda$ (in 1/m) and the Bessel function
$J_0$. The numerator holds the direct field, its air image and the first
reflection at the interface; the denominator sums the reflections back and
forth between the surface and the interface. For two layers the
reflection coefficient is the constant $K$, so three steps give a closed
form:

1. expand the denominator in a geometric series,
   $1/(1 - K e^{-2\lambda h_1}) = \sum_{m\ge0} K^m e^{-2m\lambda h_1}$;
2. multiply out: every term is $K^n e^{-\lambda d}J_0(\lambda s)$ with a
   vertical distance $d$ of $2nh_1 \pm (z - z_s)$, $2nh_1 + z + z_s$ or
   $2nh_1 - z - z_s$;
3. integrate each term with Lipschitz' integral
   $\int_0^\infty e^{-\lambda d}J_0(\lambda s)\,\mathrm d\lambda = 1/\sqrt{s^2+d^2}$,
   which is the potential of a point source at distance
   $\sqrt{s^2 + d^2}$.

The result is the image series

$$
\varphi = \frac{\rho_1 I}{4\pi}\Biggl[\frac{1}{r_0^+} + \frac{1}{r_0^-}
+ \sum_{n=1}^{\infty} K^n\Bigl(\frac{1}{r_n^{++}} + \frac{1}{r_n^{+-}}
+ \frac{1}{r_n^{-+}} + \frac{1}{r_n^{--}}\Bigr)\Biggr],
\qquad
r_n^{\sigma\tau} = \sqrt{s^2 + (z - \sigma\,2nh_1 - \tau z_s)^2},
$$

with $\sigma, \tau \in \{+1, -1\}$ and $r_0^{\pm} = \sqrt{s^2 + (z \mp z_s)^2}$:
the source and its air image, plus four images of weight $K^n$ per order
$n$ at the depths $\pm 2nh_1 \pm z_s$ (Stefanescu 1930; Tagg 1964; Sunde
1968). Each order is one more round trip of the current field between the
surface and the interface. For $K = 0$ only the $n = 0$ pair remains and
the result is that of `image` bit for bit.

### Electrodes in the top layer

Every segment carries the source, the air image and the image series.
With the exact line kernel (default) all of them are line sources, treated
like the direct pairs of the [kernels page](../kernels.md):

- image pairs closer than six segment lengths are exact Galerkin
  integrals, blended into the far form up to 7.5 segment lengths;
- beyond, the point form with the quadrupole term of the segment average,
  tapered off between 60 and 120 segment lengths of vertical distance,
  where it falls below the remainder of the far form;
- the exact integral of an image pair takes the image's potential at the
  transverse distance $\max(\rho_\perp, a)$ of the wire from the image's
  axis ($\rho_\perp$ the distance of the two axes, $a$ the wire radius):
  its average over the wire's circumference.

A straight wire then reproduces its uniform-current resistance at any
segment length ([Numerics](#numerics)). The point kernel sits about
1–2 % low, as in homogeneous soil.

### Electrodes that reach the interface

A rod driven through $h_1$, or any wire at or below the interface, needs
the Green's function with source and field point in different layers. As
soon as one segment reaches the interface, `image_2layer` evaluates every
pair with the two-layer Sommerfeld kernel of
`groundfield.coupling.layered_green` (ADR-0007):

- segments are first split at $h_1$;
- the reaction matrix is the homogeneous kernel scaled with the
  resistivity of the source layer, plus a layered correction $\Delta Z$
  evaluated by spectral quadrature, grouped by depth;
- for a pair in different layers the direct term carries the transmitted
  resistivity $2\rho_1\rho_2/(\rho_1+\rho_2)$, which keeps the matrix
  reciprocal;
- with the line kernel $\Delta Z$ is in line form as well: in two-layer
  soil it is an exact superposition of images in every layer combination
  (Sunde), and its images get the treatment of the Tagg / Sunde images
  above — the far quadrupole term over all pairs, the Galerkin integral for
  pairs within six segment lengths blended up to 7.5, the Galerkin self
  images on the diagonal — on top of the spectral values between
  midpoints.

Physically the leakage current density of a rod through the interface
scales roughly with the conductivity of the layer it is in. Such an
electrode is therefore **one owner per layer**, all in the electrode's
cluster — the same model as two rods bonded at $h_1$: the current density
may differ above and below the interface, and the reported potential is
the length-weighted average over the whole electrode. Every interface cut
is kept, however short the piece beyond it: the Galerkin self term and the
line kernel's parallel-pair integral are evaluated without cancellation
for such pieces, so the impedance is continuous in the depth of a rod
tip.

The diagonal entry of a segment is the layered correction at the wire
radius, $s = a$, with the interface image exchanged for its Galerkin
integral; its closed-form $1/R$ terms are evaluated at that radius (the
default regularisation of `layered_green`, a clamp at $h_1/200$, is meant
for coincident points). The diagonal then equals the image solution to
$10^{-13}$, a ring is continuous through $h_1$ (4.1826 Ω 1 µm above,
4.1805 Ω at $h_1$), and pairs closer than $h_1/200$ are regularised at the
1 mm of the point forms they combine with.

## Numerics

### Truncation of the series

The weights decay geometrically, so the neglected tail after order $n$ is
bounded by $\lvert K\rvert^{n+1}/(1 - \lvert K\rvert)$. The series (one
`ImageSeries` object shared by the solver, `FieldResult.potential` and
`solve_mutual_*`) stops at the first $n$ whose tail bound is below
`Engine.image_series_tol` (default $10^{-6}$), or at
`Engine.image_max_terms` (default 300) with a `SeriesTruncationWarning`.
The number of terms depends only on $\lvert K\rvert$, about
$\ln(\text{tol})/\ln\lvert K\rvert$; the generated table in the
[validation section](#validation) lists it for typical contrasts — the
default cap covers the AP1 corner soil 1000 / 30 Ω·m.
`metadata["series"]` records the controls of the solve, and the post-solve
field evaluation uses the same controls.

### Convergence with the segment length

With the line kernel a straight wire reproduces its exact uniform-current
resistance at any segment length:

- a 10 m strip at 0.5 m in 1000 / 50 Ω·m with $h_1$ = 1 m: within 3.6e-6
  from 2 m to 0.125 m segments, not monotonically; 0.1 m above $h_1$
  within 3.8e-6;
- a ring 1 cm above a 1000 / 30 Ω·m interface: within 0.004 % of its exact
  thin ring;
- the surface potential above the strip: within 8e-6 of the closed-form
  image sum;
- a 75 m strip below a 0.5 m top layer: within 1.1e-5 at every segment
  length from 0.5 to 37.5 m in 500 / 100 and 1000 / 30 Ω·m (within 4.1e-6
  from 1 m); over a more resistive lower layer, 100 / 500 Ω·m, +1.2e-4 at
  0.5 m and +4.8e-5 at 1 m (the far terms of an image series that does not
  alternate);
- a 6 m rod from 0.6 m through $h_1$ = 5 m in 500 / 100, 1000 / 30
  and 30 / 1000 Ω·m: within 6.7e-7 from 2 m to 0.125 m segments, and
  within 1.4e-6 of the exact two-owner value;
- for wires in the upper layer, where both paths apply, the cross-layer
  path reproduces the series to 1.8e-7.

About 2e-6 of the strip's rest is the reduced kernel that the soil-surface
image keeps. The point kernel converges to a value 1–2 % low.

### Far images in point form

`Engine(image_far_field="point")` (the author's decision of 2026-11-15)
leaves out the dense far term: the far Tagg / Sunde images and the far
images of the layered correction are point sources, while the near image
pairs — within 7.5 segment lengths, a segment's own images included — stay
Galerkin integrals blended into the point form, on the series and the
cross-layer path alike. It saves the far-image work (cost on the
[kernels page](../kernels.md#cost)) and against the default moves

- networks by 3e-6 … 4.3e-4 (16 house grids 0.4 m above a 1000 / 10 Ω·m
  interface: 4.3e-4; 16 house grids with a 75 m bare conductor in two
  pieces below a 0.5 m top layer: within 4e-6),
- single wires near the interface by up to 4e-4 and the surface potential
  above them by up to 0.8 %.

A wire on the interface is up to 3e-4 apart on the two paths in this form
(the default: 1.3e-6), because the far terms the point form leaves out are
split differently between the two decompositions. The continuous switch
and the transverse distance of near image pairs stay, so `"point"` is not
the result of the earlier code: within the upper layer it differs from it
by up to 2.3e-4. `metadata["series"]` records the form, the post-solve
field evaluation and `solve_mutual_*` use it, and any value other than
`"quadrupole"` or `"point"` raises `ValueError`.

### Cost

The series path costs the image terms of every pair: their number grows
with the contrast (table below). The cross-layer path costs one spectral
evaluation per pair of segment depths. Large depth groups (all house grids
at one depth, for example) are planned once and evaluated in row chunks,
so the path needs the memory of the series path — the reaction matrix,
$16N^2$ bytes, within the budget and row blocks beyond it. The
boundary-value system of the spectral amplitudes is eliminated in closed
form, vectorised over the quadrature nodes. Timings: [kernels](../kernels.md#cost)
and [performance](../performance.md).

## Validation

Against `mom` on the same segments, electrodes in the top layer agree to
0.37 % (B6) and a rod through the interface to 0.38 % (B7): the difference
is the uniform-current ansatz (both run with the point kernel in that
comparison). For straight wires the line kernel reproduces the exact
uniform-current resistance (kernel check at the end of the generated
block). The field-grade limits of the ansatz next to grids and stars are
those of [`image`](image.md#field-quantities-and-the-uniform-current-ansatz).
Next to the interface the thin-wire model itself has limits, measured
against a solid-rod and a solid-torus model ([Validity range](#validity-range)).

<!-- generated:begin image_2layer -->
Generated from the benchmark run measured on 2026-11-16 with groundfield 2026.11.0 (Python 3.12.16, NumPy 2.4.4, SciPy 1.17.1, numba installed: yes; Darwin arm64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

| Benchmark | Measure | Bound | Worst measured | Cases | Status | Note |
|---|---|---|---|---|---|---|
| B6 Top-layer electrodes, two-layer soil | abs(Z/Z_ref - 1) | 1 % | 0.37 % | 10 | pass | uniform-current ansatz |
| B7 Rod through the layer interface | abs(Z/Z_ref - 1) | 2 % | 0.38 % | 2 | pass | one uniform current across the interface (F02); bound met since the fix of 10.15 |
| B8 Two-port reciprocity | abs(Z_AB/Z_BA - 1) | 1e-7 | 7.3e-9 | 2 | pass |  |
| B9 Limits of the layered kernels (`equal_rho_integral`) | abs(Z/Z_ref - 1) | 1e-10 | 4.4e-16 | 2 | pass | image_2layer, mom, mom_sommerfeld, cim, bem |
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
|  |  | `line` | −2.2e-7 | −9.3e-7 |  | −1.1e-6 |
<!-- generated:end image_2layer -->

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Soil | two distinct layers; $\rho_1 = \rho_2$ is solved as homogeneous | `BackendCapabilityError` for three distinct layers (use `mom_sommerfeld`) |
| Contrast | $\lvert K\rvert < 1$; terms grow like $\ln(\text{tol}) / \ln\lvert K\rvert$ | `SeriesTruncationWarning` at `image_max_terms` |
| Electrodes | anywhere in the soil; at or through $h_1$ via the slower Sommerfeld path | — |
| Rod ends | head and tip more than one wire radius from $h_1$ | `InterfaceContactWarning` |
| Horizontal wires next to $h_1$ | axis outside the reach of the interface (3.9 wire radii at contrast 33) | `InterfaceContactWarning` |
| Point kernel near $h_1$ | horizontal wires at least 0.6 segment lengths from $h_1$ | `PointKernelInterfaceWarning`; within 0.387: `BackendCapabilityError` |
| Thin wire, burial, shallow segments | as for [`image`](image.md#validity-range) | `ValueError` / `ShallowSegmentWarning` |
| Field quantities near grids, stars, strip ends | uniform-current ansatz, see [`image`](image.md#field-quantities-and-the-uniform-current-ansatz) | `UniformCurrentFieldWarning` |
| Frequency | quasi-static, $f < 1$ kHz | — |

The two thin-wire limits at the interface and the point-kernel limit are
explained below. What to do instead, and the other limits of the family:
[known limitations](../../reference/known-limitations.md#current-ansatz-and-source-kernels);
the warnings and their reach: [warnings reference](../../reference/warnings.md#interfacecontactwarning).

### Rod ends at the interface: a limit of the thin-wire model

The thin wire has no end face. A rod that ends exactly at $h_1$ is a wire
entirely in the upper layer; one that reaches past it by a fraction of its
radius touches the lower layer through a short piece only. The solid rod
is continuous in between, because its flat end face carries the contact.
Against an axisymmetric finite-volume model of the solid rod (4.5 m,
$a$ = 8 mm, 1000 / 30 Ω·m) the line kernel reads:

| Tip below $h_1$ | Solid rod | Thin wire |
|---|---|---|
| −16 mm | 190.2 Ω | +3.3 % |
| −8 mm | 188.9 Ω | +3.9 % |
| −1 mm | 182.8 Ω | +7.2 % |
| 0 | 161.4 Ω | +21.4 % |
| +1 mm | 156.3 Ω | −4.6 % |
| +4 mm | 148.5 Ω | −1.2 % |
| +8 mm | 141.2 Ω | +0.7 % |
| +16 mm | 130.2 Ω | +2.1 % |

The point kernel and `mom` behave alike (+19.0 … +20.5 % at 0). Every
segment backend therefore warns with `InterfaceContactWarning` when a rod
head or tip lies within one wire radius of an interface, and `diagnose()`
lists it under `interface_contact`. **Keep rod ends a few wire radii from
an interface**, or bracket the result by moving the end by ±2 wire radii;
`fem` models the solid rod.

### Horizontal wires next to the interface

A horizontal wire has the same kind of limit: next to a more conductive
layer the solid wire's current crowds to its near side, which the current
on the axis cannot follow. A ring ($a$ = 5 mm) reads +1.2 % at 3 wire
radii, +3.6 % at 2 and +8.6 % at 1.5 radii above a layer 33 times more
conductive, against the solid torus of the axisymmetric `fem`. Within the
reach of the interface — 3.9 radii at that contrast, tabulated under
`InterfaceContactWarning` in the
[warnings reference](../../reference/warnings.md) — the solve warns with
`InterfaceContactWarning` as well.

### Point kernel next to the interface

With `kernel="point"` (and in the `mom` family, which shares the point
form) a horizontal wire just below or above $h_1$ needs segments much
shorter than twice its distance to the interface:

- 5 cm below $h_1$, a ring of radius 3 m gives 29.7 Ω at ds = 0.5 m
  against 54.7 Ω at ds = 0.025 m, while the line kernel gives
  55.6–55.7 Ω at every segment length
  (`tests/test_cross_layer_line_kernel.py`);
- above $h_1$ the point form takes the image of a segment in the
  interface as a point source: a ring 1 µm / 1 cm above $h_1 = 5$ m read
  −1931 / −57.9 Ω (line kernel 4.18 / 15.18 Ω).

On either side of the interface a wire closer than 0.387 segment lengths
therefore raises `BackendCapabilityError`, and one within 0.6 segment
lengths warns with `PointKernelInterfaceWarning`. The default line kernel
needs no such care.

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
  Engineering Software* 44 (acceleration of the series for
  $\lvert K\rvert \to 1$, not used here).

Architecture decisions: [ADR-0001](../../adr/0001-two-layer-method.md),
[ADR-0007](../../adr/0007-cross-layer-electrodes.md),
[ADR-0015](../../adr/0015-line-source-kernel-default.md). API:
`groundfield.solver.image_2layer.solve_image_2layer`.

## Audit trail

??? note "Line-source images (wave 3 of the 2026.11 work, audit W3A-02)"

    Before wave 3 of the 2026.11 work, far images were point sources (which
    `Engine.image_far_field = "point"` offers in its own form today), and
    on the cross-layer path only the interface image of near pairs in one
    layer was a Galerkin integral. The reduced kernel of direct pairs,
    applied to images, left out $a^2/(2d^2)$ of an image $d$ away; audit
    W3A-02 replaced it by the transverse distance. Measured before against
    now:

    - the 10 m strip at 0.5 m in 1000 / 50 Ω·m: up to +5.1e-4 before;
      0.1 m above $h_1$: +6.8e-5 with the reduced kernel on the images;
    - the ring 1 cm above a 1000 / 30 Ω·m interface: +1.60 % against its
      exact thin ring before (now −0.004 %); the point-kernel example ring
      1 cm above $h_1 = 5$ m: line kernel 15.42 Ω before audit W3A-02,
      15.18 Ω now;
    - the surface potential above the strip: up to 0.94 % before (the
      alternating series of a more conductive lower layer amplifies the
      error of its far images);
    - the rod through the interface: 1.3e-4 … 2.4e-4 over the segment
      lengths before; the cross-layer path against the series in the upper
      layer: up to 6.2e-4 before.

??? note "Point form of the far images (audits W3R2A-01, W3R2A-02, user decision D2)"

    Up to audit W3R2A-01 the cross-layer path kept the line form of the
    interface image only, as before wave 3; the own surface and transmitted
    images of a segment below the interface were point sources, and the
    75 m strip below a 0.5 m top layer read −1.0e-3 at a segment length of
    2.5 m and −58 % at 37.5 m (16 house grids with a 75 m bare conductor in
    two pieces: +1.1 % against the default). The point form keeps the
    continuous switch of user decision D2, which blends up to 7.5 segment
    lengths where the base switched at six in one step; hence its
    difference of up to 2.3e-4 from the code before wave 3. Audit W3R2A-02
    added the `ValueError` for other values of `image_far_field`.

??? note "Series length and memory (findings 10.25, 10.56)"

    Up to 0.15.0 the post-solve field evaluation capped the series at 100
    terms (finding 10.25); it now uses the controls of the solve. Before
    finding 10.56 the cross-layer path needed 41 instead of 18 bytes per
    segment pair for 16 house grids below $h_1$, and had no row blocks; one
    LAPACK solve per quadrature node took half the time of the path before
    the closed-form elimination.

??? note "Rods through the interface (finding 10.15, audits A2S-01, A3S-01)"

    Up to 0.15.0 one uniform density ran across the interface, which put a
    12 m rod through $h_1 = 5$ m in 1000 / 30 Ω·m soil 566 % above the
    per-segment `mom` solution (`tests/test_cross_layer_owners.py`); since
    2026.11 the electrode is one owner per layer (finding 10.15).

    Earlier in 2026.11, cuts that would leave a piece shorter than the wire
    diameter were dropped, which ignored the contact with the other layer:
    a 4.5 m rod in 1000 / 30 Ω·m read 195.0 Ω with its tip 15.9 mm into the
    lower layer and 128.6 Ω at 16.1 mm (second adversarial audit, A2S-01).

    The default regularisation of `layered_green` clamps the closed-form
    $1/R$ terms at $h_1/200$ (25 mm for $h_1 = 5$ m). On the diagonal it
    took 68–84 % of the layered correction of every piece within $h_1/400$
    of the interface: a ring of radius 3 m at $h_1 = 5$ m read −5.29 Ω, and
    the contact values of a rod tip were 10–32 % low (third adversarial
    audit, A3S-01). The diagonal is now evaluated at the wire radius.

??? note "Horizontal wires next to the interface (audits W4A-07, W4R2A-04)"

    The ring next to a layer 33 times more conductive was measured against
    the solid torus of the axisymmetric `fem` in audit W4A-07; the reach of
    the interface, 3.9 radii at that contrast, comes from audit W4R2A-04.
