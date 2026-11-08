# `fem` — equivalent hemisphere by finite elements (experimental)

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | any number of layers |
    | Model | the volume PDE of one **equivalent hemisphere** per cluster, axisymmetric |
    | Discretisation | linear triangles on a spherical-shell mesh that conforms to the hemisphere; layer interfaces cut into the mesh |
    | Far field | exact monopole Dirichlet-to-Neumann (Robin) condition on the truncation sphere |
    | Options | `FemOptions`: `n_radial`, `n_axial`, `r_far_factor`, `far_field` |
    | Status | **experimental** (`ExperimentalBackendWarning` once per process); valid for one compact electrode inside one layer |

`fem` solves the volume form of the current-flow problem instead of an
integral equation. It does not mesh the electrodes: every cluster is
replaced by a hemisphere with the same resistance in homogeneous soil, and
that hemisphere is solved in the layered soil. In homogeneous soil the
result is therefore Dwight's closed form by construction, and in layered
soil it describes a different geometry from the one in the world. It is
kept as a fast plausibility check for a single compact electrode inside
one layer, not as a cross-check of the integral engines (finding 10.23).

## Mathematical and physical context

The stationary current field obeys the volume equation

$$
-\nabla\cdot\bigl(\sigma(z)\,\nabla\varphi\bigr) = 0
\quad\text{in the soil},\qquad
\frac{\partial\varphi}{\partial z}\Big|_{z=0} = 0,
$$

with $\sigma = 1/\rho$ piecewise constant in depth and the electrode a
surface of prescribed potential. For a body of revolution about a vertical
axis the problem reduces to the meridian half-plane $(s, z)$,

$$
-\frac{1}{s}\frac{\partial}{\partial s}\Bigl(s\,\sigma\,\frac{\partial\varphi}{\partial s}\Bigr)
-\frac{\partial}{\partial z}\Bigl(\sigma\,\frac{\partial\varphi}{\partial z}\Bigr) = 0 ,
$$

and the resistance follows from the energy: with $\varphi = 1$ V on the
electrode, the conductance is $G = 1/R = \int \sigma|\nabla\varphi|^2\,\mathrm dV$
(Dirichlet principle).

### The equivalent hemisphere

An electrode of a cluster is replaced by the hemisphere of radius
$a_\text{eq}$ whose resistance in homogeneous soil of the top-layer
resistivity equals the electrode's closed-form resistance,
$\rho_1/(2\pi a_\text{eq}) = R_\text{Dwight}$ (Dwight 1936; Sverak's
formula for grids). The radii of the electrodes of one cluster are summed,
which puts them in parallel without their mutual resistance. The
reduction is exact in what it promises — the Dwight value in homogeneous
soil — and wrong wherever the geometry matters: a hemisphere reaches the
depth $a_\text{eq}$, not the depth of the electrode, so it feeds a
different layer than a rod or a ring does.

## Discretisation

The mesh is built in spherical-shell coordinates around the electrode,
$s = r\sin\vartheta$, $z = r\cos\vartheta$, $r \in [a_\text{eq}, r_\text{far}]$,
with a geometric ladder in $r$ and a uniform one in $\vartheta$. The
electrode surface $r = a_\text{eq}$ is an exact mesh line (no staircase),
the soil surface $\vartheta = \pi/2$ carries the natural Neumann
condition, and element shapes are scale-invariant. Linear elements on
straight-sided triangles with the centroid radius integrate the
axisymmetric weak form exactly. Elements crossed by a layer interface are
cut along it, so each element has one conductivity.

Refining $(n_\text{radial}, n_\text{axial})$ by $n \mapsto 2n-1$ produces
nested spaces; by the minimum-energy principle the discrete conductance
can only decrease, so $R_h$ **increases monotonically towards the exact
value from below** (benchmark B1 reports the observed order). The domain
is truncated at $r_\text{far} = $ `r_far_factor` $\cdot\,(a_\text{eq} + \sum h_i)$
with the Robin condition $\partial\varphi/\partial r = -\varphi/r_\text{far}$,
the exact Dirichlet-to-Neumann map of the monopole far field; it removes
the $O(a_\text{eq}/r_\text{far})$ error of a grounded truncation sphere.
Since 2026.11 the number of radial lines grows with
$\ln(r_\text{far}/a_\text{eq})$ when a deep layer stack stretches the
domain, so the near-field resolution no longer degrades with $h_1$
(finding 10.49).

## Guards

The registry marks `fem` experimental; the reduction itself is guarded
(findings 10.23, the audit rounds of the release and the D2 audit). All
layer rules ask which layers the electrodes touch with one predicate, the
contact of the thin-wire model the integral engines solve: a horizontal
wire touches the layers that hold its axis (a wire lying on an interface
touches both, to round-off), a rod the layers its axis passes through —
an end lying on an interface does not count, the thin-wire model has no
end face (audit D2A-01, D2A-02). Only the first error below also counts
a wire within its radius of a layer as touching it, the one rule that
this makes stricter; counting it in the other two lifted errors where
`fem` is far too low (a ring 3 mm above a more conductive layer: −56 %,
a warning in between; audit R2A-02).

- **Error** (`ValueError`, before anything else): an electrode, driven or
  passive, lies at or above the soil surface, in the surface plane
  ($2z \le a$) or reaches above it — what every other backend refuses as
  well (audit D2A-06). A rod with its head at $z = 0$ is the ordinary
  driven rod and stays valid.
- **Error** (`BackendCapabilityError`, before meshing): an electrode
  touches a layer that its hemisphere does not reach — a rod through the
  top layer, a wire on (or within its radius above) an interface below
  the hemisphere. The hemisphere would inject the current into the wrong
  layer; a 3 m rod through $h_1 = 2$ m was +1092 % off in v0.15.0,
  silently (benchmark B7 records the exclusion), and a 2 m ring on
  $h_1 = 2$ m ($a_\text{eq} = 1.3$ m) is +1008 % (1000/30 Ω·m) and −18 %
  (30/1000 Ω·m) against `image_2layer`, silent up to the D2 audit.
- **Error** (`BackendCapabilityError`): every electrode of a driven
  cluster lies below an interface at a depth between $a_\text{eq}/2$ and
  $a_\text{eq}$, so more than half of the hemisphere sits in a layer the
  electrodes do not touch (a 3 m rod below $h_1 = 0.4$ m in
  1000 / 30 Ω·m read 31.1 Ω against 10.5 Ω from `mom`, +197 %; second
  adversarial audit, A2E-2; a 2 m ring 1 … 7 mm below $h_1 = 0.8$ m:
  −5 … −84 % in 30/1000 and 30/10000 Ω·m, +11 … +19 % in 1000/30 and
  10000/30 Ω·m).
- **Error** (`BackendCapabilityError`, user decision Q23 of 2026.11): a
  layer that no electrode of the cluster touches and that conducts better
  than every layer the electrodes touch, with its top above
  $\max(a_\text{eq}, r_\text{ext})$ — the hemisphere radius or the
  electrodes' horizontal half-extent (ring radius, half the strip length,
  half the grid diagonal). The hemisphere injects the current straight
  into a layer it spans and reaches one below it much less than the
  electrodes do: `fem` misplaces the current, mostly far too low, i.e. not
  conservative. The
  rule is structural, independent of the size of the contrast, and works
  on the canonical layer stack (equal resistivities are no interface).
  The fem error is continuous across $h_1 = a_\text{eq}$ (−67.9 % on both
  sides for a 10 m ring over 1000/30 Ω·m), so the rule covers the
  half-extent as well since the D2 audit (D2A-03; a coordinator decision
  flagged for the author's review). The message names the cluster, the
  layer (depth range and $\rho$), the electrodes' $\rho$,
  $a_\text{eq}$, the measured error and the backends whose own checks
  accept the world. Measured against `image_2layer` and `mom`:

    | World (two layers) | `fem` against `image_2layer` / `mom` |
    |---|---|
    | 3 m rod at 0.5 m below a more conductive top layer, $h_1 = 0.05 \ldots 0.49\,a_\text{eq}$ (22 … 217 mm), or with its head on $h_1$ | 100/1000 Ω·m: −12 … −58 %; 30/1000: −35 … −80 %; 30/10000: −80 … −97 % |
    | 10 m strip at 0.8 m below such a top layer, $h_1 = 0.22$ and $0.50$ m | 100/1000: −32 … −47 %; 30/1000: −58 … −71 %; 30/10000: −91 … −95 % |
    | ring $r = 10$ m at 0.8 m above a more conductive layer at $h_1 = 2$ m | 1000/300 and 100/30: −45 %; 1000/30: −92 % |
    | ring $r = 10$ m at 0.8 m above a more conductive layer at $h_1 = a_\text{eq} \ldots 10$ m (below the hemisphere, within the radius) | 1000/30: −68 … −7 %; 3000/30 and 1000/10: −78 … −8 %; 100/30 and 1000/300: −28 … −4 % |
    | 10 m strip at 0.8 m above a more conductive layer at $h_1 = 1.2 \ldots 5$ m | 1000/30 and 3000/30: −45 … +1 % |
    | 20 m grid at 0.5 m above 1000/30 Ω·m at $h_1 = a_\text{eq} \ldots 14$ m | −60 … +13 % (its formula reads +12 % high in homogeneous soil) |
    | near-equal layers (100/120 and 120/100 Ω·m), rods, strips and rings | −7 … +5 % (raises as well: the rule is structural) |
    | ring $r = 5$ / 10 m within its wire radius (1 µm … 7 mm) below a more conductive top layer, $h_1 = 0.5$ / 0.8 m (`mom` refuses these; second reference an independent ring series) | 30/1000 and 100/1000 Ω·m: −6 … +30 %; 30/10000: −59 … +7 %; 100/120: +2 … +3 % |
    | ring $r = 5$ / 2 m within its wire radius (1 µm … 7 mm) above a more conductive lower layer, $h_1 = 0.5$ / 0.8 m (likewise) | 1000/30, 3000/30, 10000/30 Ω·m: −96 … +11 %; 120/100: −3 … +1 % |

  The rule stops at the half-extent: just beyond it fem is silent at
  −9.8 … −0.8 % (rings $r$ = 2, 5, 10 m), −1.4 % (10 m strip) and +9.9 %
  (20 m grid) against `image_2layer` (−8.3 … +0.1 %, +0.6 % and +12.7 %
  against `mom`), and the deviation fades further down (−1.3 % for the
  10 m ring over $h_1$ = 20 m).
- **Error** (`BackendCapabilityError`): the closed form of the
  hemisphere is used outside its range (audit D2A, O-1). Dwight's strip
  formula is the series of the wire and its image in $s/l$ ($s = 2t$
  the image distance, $l$ the length) and converges only for a burial
  depth $t < l/2$; his ring formula needs $s \ll D$. A driven strip deeper
  than half its length and a driven ring deeper than its radius raise
  (a 1.88 m strip at 5 m and 9.8 m read +369 % and +6120 % in homogeneous
  soil, silently). Every driven `MeshElectrode` raises: its formula, one
  strip along the diagonal, misses the wires of the mesh (+118 … +193 %
  against `image` at 0.8 m); a `GridMeshElectrode` of the same wires is
  reduced with Sverak's formula (−1 … +21 %).
- **Error** (`BackendCapabilityError`): a driven cluster contains a star
  or polyline electrode, which has no closed-form hemisphere radius. The
  generators build mesh foundations of polyline wires, so generated
  networks cannot be solved with `fem`.
- The backends an error message names pass their own checks for the
  world — the capability check, the resource limit, the burial depth of
  every segment, the point form at the interface and the cross-layer
  limit of `mom_sommerfeld`; a backend that refuses is named with its
  reason, and if none accepts, the message says that no backend solves
  the world yet (audit D2A-06).
- **Warning** (`FemReductionWarning`, once per solve, naming every reason
  and the cluster members): a cluster with several electrodes (radii
  summed; −35 % for a ring with three rods in homogeneous soil, −55 … −75 %
  for two bonded rods of which one reaches into a conductive top layer);
  several clusters (each solved alone, transfer impedances 0, passive
  clusters at 0 V); a layer interface above the deeper of the electrode
  and the hemisphere that the errors above do not cover: the spanned
  layer conducts no better than the electrodes' layer (−1 … +48 %
  measured, +23 % for a 10 m ring over a more resistive layer at 2 m), an
  electrode touches the conductive layer only partly (a rod reaching
  1 µm … 8 cm into a conductive top layer of $0.2\,a_\text{eq}$:
  −44 … −78 %; a ring lying exactly on the interface, which touches both
  layers: −21 … +30 % against `image_2layer` and a ring series);
  an interface below the electrode and the hemisphere but within the
  horizontal half-extent of the electrode over a layer that conducts no
  better (+9 % for a 10 m ring over 30/1000 Ω·m at 5 m; −16 … +16 % over
  30/1000, 100/1000 and 30/10000 Ω·m). Escalate with
  `warnings.simplefilter("error", FemReductionWarning)` when `fem` serves
  as a check.
- Concrete shells are rejected; distributed and inductive conductors are
  solved as lumped branches, and `resistance_model="skin"` is ignored, each
  with a `BackendCapabilityWarning`.
- **Not covered** (silent): an interface just below a horizontal electrode
  that is buried almost as deep as its half-extent, with the interface
  deeper than both the hemisphere radius and the half-extent — a 2 m ring
  at 1.9 m over $h_1 = 2$ m in 1000/30 Ω·m reads +34 %, at 1.993 m +250 %.
  Rods are hardly affected (+5.7 % with the interface 1 mm below the tip).
- The size notice of `Engine.solve` ("solve time may run from seconds to
  minutes") and its resource limit count the segments of the integral
  engines, which `fem` does not use: its problem is the mesh of each
  driven cluster (about 2 400 nodes at the default resolution, 3 700 with
  a 30 m top layer, independent of `segment_length`), and `fem` does not
  refuse a world above `Engine.max_segments`.

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Electrodes | one compact rod, ring, strip or grid per cluster, inside one layer, with the hemisphere inside that layer | error (wrong layer, or a more conductive layer within the hemisphere or the half-extent that no electrode touches) or `FemReductionWarning` (see above) |
| Burial depth | strip: at most half its length; ring: at most its radius; grid and rod: no limit (Sverak's grid formula stays bounded, +6 … +21 % from 0.25 m to 20 m; Dwight's rod ignores the head depth, +3.9 % at 0.5 m … +10 % at 10 m) | error (the closed form diverges: up to +474 % for strips, −28 % for rings) |
| Kinds | rod, ring, strip, grid (`GridMeshElectrode`) | error for a driven `MeshElectrode`, star or polyline |
| Clusters | one driven cluster | `FemReductionWarning`: no coupling between clusters |
| Conductors | lumped | distributed and inductive conductors solved as lumped, `BackendCapabilityWarning` |
| Accuracy | discretisation error of the hemisphere itself: B1 below; the reduction is the Dwight formula | — |
| Frequency | quasi-static, frequency independent | — |

## Validation

<!-- generated:begin fem -->
Generated from the benchmark run measured on 2026-11-08 with groundfield 2026.11.0 (Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, numba installed: yes; Linux x86_64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

| Benchmark | Measure | Bound | Worst measured | Cases | Status | Note |
|---|---|---|---|---|---|---|
| B1 Hemisphere | abs(Z/Z_ref - 1) | 0.5 % | 0.16 % | 1 | pass | observed order of convergence 1.99; discretisation of the conforming shell mesh alone |
| B2 Rod, homogeneous soil | abs(Z/Z_ref - 1) | 1 % | 0.70 % | 3 | pass | equals Dwight's rod formula |
| B3 Ring, homogeneous soil | abs(Z/Z_ref - 1) | 1 % | 0.34 % | 3 | pass | equals Dwight's ring formula |
| B6 Top-layer electrodes, two-layer soil | abs(Z/Z_ref - 1) | 5 % | 4.3 % | 10 | pass | experimental; equivalent hemisphere in layered soil (10.23) |
| B7 Rod through the layer interface | abs(Z/Z_ref - 1) | excluded | — | — | excluded | raises BackendCapabilityError (v0.15.0: +1092 %, silent); excluded: the rod reaches a layer its equivalent hemisphere does not (10.23) |
| B8 Two-port reciprocity | abs(Z_AB/Z_BA - 1) | excluded | — | — | excluded | Z_AB = Z_BA = 0; excluded: clusters solved without coupling (10.23) |
| B9 Limits of the layered kernels (`equal_rho_fem`) | abs(Z/Z_ref - 1) | 0.1 % | 2.0e-4 | 3 | pass | declared stack; <= 1e-5 for interfaces below the hemisphere (10.49) |
| B9 Limits of the layered kernels (`layer_limit_fem`) | abs(Z/Z_ref - 1) | 1 % | 0.20 % | 2 | pass | experimental; h₁ = 4 mm emits FemReductionWarning (structural interface rule, audit F01) |
| B10 Ring with three bonded rods | abs(Z/Z_ref - 1) | excluded | — | — | excluded | -35 % / -45 %; excluded: radii summed (10.23) |
<!-- generated:end fem -->

## Literature

- Dwight, H. B. (1936). Calculation of resistances to ground. *Electrical
  Engineering* 55(12), 1319–1328.
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission Systems*.
  Dover, New York, ch. 2 (equivalent hemisphere).
- Güemes, J. A., & Hernando, F. E. (2004). Method for calculating the
  ground resistance of grounding grids using FEM. *IEEE Transactions on
  Power Delivery* 19(2).
- Givoli, D. (1992). *Numerical Methods for Problems in Infinite Domains*.
  Elsevier (Dirichlet-to-Neumann truncation; the monopole map is its
  lowest mode).
- Strang, G., & Fix, G. (2008). *An Analysis of the Finite Element
  Method*, 2nd ed. Wellesley-Cambridge Press (minimum-energy principle and
  one-sided convergence).

Architecture decision: [ADR-0002](../../adr/0002-engine-family.md) and its
amendments. API: `groundfield.solver.fem.solve_fem`,
`equivalent_hemisphere_radius`, `FemReductionWarning`.
