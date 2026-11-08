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
(findings 10.23 and the first audit round of the release):

- **Error** (`BackendCapabilityError`, before meshing): an electrode
  reaches below a layer interface that its hemisphere does not reach — a
  rod through the top layer. The hemisphere would inject the current into
  the wrong layer; a 3 m rod through $h_1 = 2$ m was +1092 % off in
  v0.15.0, silently (benchmark B7 records the exclusion).
- **Error** (`BackendCapabilityError`): every electrode of a driven
  cluster lies below an interface at a depth between $a_\text{eq}/2$ and
  $a_\text{eq}$, so more than half of the hemisphere sits in a layer the
  electrodes do not touch (a 3 m rod below $h_1 = 0.4$ m in
  1000 / 30 Ω·m read 31.1 Ω against 10.5 Ω from `mom`, +197 %; second
  adversarial audit, A2E-2). Thinner top layers keep the warning below
  (+16 … +46 % and −38 … −58 % measured).
- **Error** (`BackendCapabilityError`): a driven cluster contains a star
  or polyline electrode, which has no closed-form hemisphere radius. The
  generators build mesh foundations of polyline wires, so generated
  networks cannot be solved with `fem`.
- **Warning** (`FemReductionWarning`, once per solve, naming every reason
  and the cluster members): a cluster with several electrodes (radii
  summed); several clusters (each solved alone, transfer impedances 0,
  passive clusters at 0 V); a layer interface above the deeper of the
  electrode and the hemisphere; an interface within the horizontal
  half-extent of the electrode. Escalate with
  `warnings.simplefilter("error", FemReductionWarning)` when `fem` serves
  as a check.
- Concrete shells are rejected; distributed and inductive conductors are
  solved as lumped branches, and `resistance_model="skin"` is ignored, each
  with a `BackendCapabilityWarning`.

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Electrodes | one compact rod, ring, strip or grid per cluster, inside one layer, with the hemisphere inside that layer | error or `FemReductionWarning` (see above) |
| Clusters | one driven cluster | `FemReductionWarning`: no coupling between clusters |
| Conductors | lumped | distributed and inductive conductors solved as lumped, `BackendCapabilityWarning` |
| Accuracy | discretisation error of the hemisphere itself: B1 below; the reduction is the Dwight formula | — |
| Frequency | quasi-static, frequency independent | — |

## Validation

<!-- generated:begin fem -->
Generated from the benchmark run measured on 2026-11-08 with groundfield 0.15.0 (Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, numba installed: yes; Linux x86_64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

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
