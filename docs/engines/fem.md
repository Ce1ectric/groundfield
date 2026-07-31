# `fem` — axisymmetric volume Finite Elements

## Physical context

The Finite Element Method is the only **volume-PDE engine** in the
family. Every other backend solves an integral equation on the
electrode boundary — either with a closed-form Green's function
(`image`, `image_2layer`, `cim`) or with a numerically integrated
one (`mom`, `mom_sommerfeld`, `bem`). FEM instead discretises the
volume of the soil and solves the partial differential equation
directly:

$$
- \nabla \cdot (\sigma(\mathbf{r})\, \nabla \varphi) \;=\; q,
\qquad \sigma(\mathbf{r}) = 1/\rho(\mathbf{r}),
$$

with insulating boundary at the soil surface ($\partial \varphi / \partial z = 0$ at $z = 0$) and the far-field decay
$\varphi \to 0$ as $|\mathbf{r}| \to \infty$, truncated to a finite
outer radius $r_{\text{far}}$ that carries the monopole
Dirichlet-to-Neumann condition (see
[Boundary conditions](#boundary-conditions)). The source-current
density $q$ is concentrated on the electrode surfaces.

The role of FEM in the engine family is the **third independent
methodology**. The integral-equation engines all share a thin-wire
approximation and the same Sommerfeld kernel; if that kernel had a
bug, every integral engine would inherit it. FEM does not touch
the kernel at all — it solves the underlying PDE on the volume
mesh — so a comparison to FEM checks the kernel implementation
itself.

## Governing equation: weak form

Multiplying the PDE by a test function $v$ and integrating by parts
gives the weak form:

$$
\int_{\Omega} \sigma\, \nabla \varphi \cdot \nabla v\, dV
\;=\; \int_{\Omega} q\, v\, dV
\quad \forall v \in V_0,
$$

with $V_0$ the test-function space (functions vanishing on the
Dirichlet boundary). The Neumann boundary at $z = 0$ contributes no
boundary term — its zero-flux condition is the natural boundary
condition of this weak form.

## Axisymmetric reduction

For typical reference electrodes (rod, ring, mesh) the
problem is **rotationally symmetric** around the cluster centroid
to a good approximation. Exploiting this symmetry reduces the
problem from 3-D to 2-D in cylindrical coordinates $(s, z)$:

$$
- \frac{1}{s} \frac{\partial}{\partial s}
  \!\left(s\, \sigma\, \frac{\partial \varphi}{\partial s}\right)
- \frac{\partial}{\partial z}
  \!\left(\sigma\, \frac{\partial \varphi}{\partial z}\right)
\;=\; q.
$$

The weak form picks up an additional factor $2\pi s$ from the
volume element $dV = 2\pi s\, ds\, dz$, so the per-element
stiffness contribution becomes

$$
K^T_{ij} \;=\; 2\pi\, \sigma_T\, \bar s_T\,
(\nabla \phi_i \cdot \nabla \phi_j)\, |T|,
$$

with $\bar s_T$ the centroid radius and $|T|$ the planar area of
triangle $T$.

## Equivalent-hemisphere reduction

The axisymmetric formulation is exact for true hemispheres but only
approximate for finite-length electrodes (rods, rings, meshes).
The implementation reduces every cluster to its **equivalent
hemisphere**: a hemisphere of radius

$$
a_{\text{eq}} \;=\; \frac{\rho_1}{2\pi\, R_{\text{Dwight}}},
$$

with $R_{\text{Dwight}}$ the closed-form DC resistance of the
electrode in homogeneous soil (computed via
`groundfield.references.dwight1936`). The hemisphere is centred at
the cluster centroid; a multi-electrode cluster is reduced to a
single equivalent hemisphere via the parallel-conductance rule
$a_{\text{eq, cluster}} = \sum_e a_{\text{eq}, e}$ (the *radii*
add, because the hemisphere conductance scales linearly with $a$).

This reduction is **exact for hemispheres**, **good (better than
5 %) for rods and shallow meshes**, and documented as a known
approximation. The FEM engine is therefore best read as "the
volume-PDE solver for the **equivalent-hemisphere** of the input
cluster, in the actual layered soil". The bias is bounded — at
worst $\sim 10\,\%$ on rings and meshes far from the hemisphere
limit — and reported in `result.metadata['equivalent_hemisphere_radius']`.

Because the reduction defines the geometry the PDE is solved on, it
also defines the reference answer the discretisation must reproduce:
in homogeneous soil the correct result of the *discrete* problem is
exactly $\rho / (2\pi a_{\text{eq}})$, and the mesh is built so that
it converges to it (see [Convergence](#convergence-and-cost)). This
is what separates the reduction bias (a modelling choice, quoted
above) from a discretisation error (a bug).

## Numerical strategy

### Mesh construction — a boundary-conforming spherical shell

The Dirichlet boundary of this backend *is* the equivalent
hemisphere $r = a_{\text{eq}}$. A mesh that does not resolve that
surface geometrically replaces it by a staircase whose shape — and
therefore whose capacitance — changes discontinuously with the mesh,
so refining the mesh changes the answer without converging. The
mesh is therefore built in **spherical shell coordinates** centred
on the electrode,

$$
s = r \sin\vartheta, \qquad z = r \cos\vartheta, \qquad
r \in [a_{\text{eq}}, r_{\text{far}}], \quad
\vartheta \in [0, \pi/2],
$$

as a structured $(r, \vartheta)$ grid, each cell split into two
triangles:

- **Radial node lines** ($n_{\text{radial}} = 60$ by default):
  geometrically spaced from $a_{\text{eq}}$ to $r_{\text{far}}$.
  Equal *relative* steps is the natural grading for a $1/r$ field —
  every element then carries the same share of the discretisation
  error.
- **Transverse node lines** ($n_{\text{axial}} = 40$ by default):
  uniform in the polar angle $\vartheta$, from the downward
  symmetry axis ($\vartheta = 0$, $s = 0$) to the soil surface
  ($\vartheta = \pi/2$, $z = 0$).
- **Truncation**: a single sphere
  $r_{\text{far}} = f_r \cdot \bar L$ with $f_r =$ `r_far_factor`
  (default 30) and $\bar L = a_{\text{eq}} + \sum_i h_i$. With a
  spherical truncation there is only one number, so `r_far_factor`
  is the whole knob and takes effect for *every* value. The
  historical `z_far_factor` defaults to `None` and is ignored; if a
  caller passes it, it acts as a lower bound on the same radius and
  the backend logs a warning when it actually overrides
  `r_far_factor` (the first spherical-shell rewrite silently combined
  the two as `max(...)`, which made `r_far_factor` a dead knob for
  every value $\le$ `z_far_factor`).
- **Layer interfaces**: a shell mesh is not aligned with the
  horizontal interfaces, so every element crossed by
  $z = \sum_{i \le k} h_i$ is **cut along the interface** into two
  or three sub-triangles. Cut nodes are keyed on the edge they
  split, so neighbouring elements share them and the mesh stays
  conforming (no hanging nodes). The conductivity jump is
  consequently mesh-aligned and each element carries exactly one
  layer conductivity from the `LayerStack`. An interface shallower
  than $a_{\text{eq}}$ also splits one **chord of the electrode
  polyline**; that cut node is added to the Dirichlet set (the
  criterion is topological — both end nodes of the split edge are
  electrode nodes — not a radius tolerance), so the energised
  surface stays closed.

Three properties follow from the choice of coordinates rather than
from any tolerance, and all three were the source of real defects in
the earlier $(s, z)$ tensor grid:

| Property | Consequence |
|---|---|
| $r = a_{\text{eq}}$ is an exact mesh line | the discrete electrode is the inscribed polyline of the intended hemisphere at *every* resolution, and it is closed (all its nodes are Dirichlet, including interface cut nodes) |
| radial step $\propto r \ln(r_{\text{far}}/a_{\text{eq}}) / n_{\text{radial}}$ | near-electrode resolution depends on the truncation only through $\ln r_{\text{far}}$ — a thick declared soil layer can no longer starve the electrode of nodes |
| element aspect ratio $\approx 2 n_{\text{axial}} \ln(r_{\text{far}}/a_{\text{eq}}) / (\pi n_{\text{radial}})$, independent of $r$ | $O(1)$ elements **in the uncut shell** — measured shape quality $\max \sum \ell^2/\lvert T\rvert = 10.7$ at the default resolution (2.31 = equilateral) instead of the 50:1 slivers of a linearly spaced axial grid |

The interface cut is the one exception to the last row: where a layer
boundary grazes a node it produces genuine slivers — worst measured
$\sum \ell^2/\lvert T\rvert = 1.1 \cdot 10^5$ at an area of
$6.7 \cdot 10^{-12}\,\text{m}^2$ over an adversarial sweep of 66
grazing depths (each node depth of the shell approached from both
sides at relative offsets $10^{-9}$ and $10^{-5}$). Nodes within a
relative tolerance of $10^{-6}$ of the plane are snapped onto it
first and degenerate elements carry no volume and are skipped in the
assembly, so the answer is unaffected in practice: at fixed
$r_{\text{far}}$ that sweep stays inside
$-0.23837 \ldots -0.23816\,\%$ — a spread of $2 \cdot 10^{-4}$
percentage points around the uncut $-0.23939\,\%$ — with no zero-area
element produced. The claim is therefore *bounded* element quality
away from the cut, not $O(1)$ quality everywhere.

### Stiffness assembly

The element stiffness uses linear hat functions on each triangle.
The per-element 3×3 matrix entries are built from the gradient
vectors $\nabla \phi_i = (b_i, c_i)/(2 |T|)$, weighted by
$\sigma_T \cdot 2\pi \bar s_T$ as derived above. Since the
gradients are constant on a straight-sided triangle and
$\int_T s\, dA = \bar s_T |T|$ holds exactly, this element rule is
an **exact** quadrature of the axisymmetric weak form, not a
one-point approximation. The global matrix is assembled in COO
format and converted to CSR for the sparse solve.

### Boundary conditions

- **Dirichlet inner** (electrode surface): the node line
  $r = a_{\text{eq}}$ is fixed at $\varphi = 1$ (the
  unit-potential probe). The set is tracked by node index, so it is
  exactly the conforming hemisphere — it is not selected by a
  geometric predicate such as $s^2 + z^2 \le a_{\text{eq}}^2$,
  which on a non-conforming grid degenerates into a flat disc.
  Tracking it by index means the interface cut has to *maintain* it:
  a cut node inserted on an electrode chord is appended to the set,
  so no free (natural, zero-flux) node is ever left on the energised
  surface. `tests/test_pass9_fem.py::` `test_fem_electrode_boundary_`
  `is_closed_under_interface_cuts` asserts this for every boundary
  edge, over interface depths straddling $a_{\text{eq}}$ and over a
  100 × 100 m mesh electrode ($a_{\text{eq}} = 6.6$ m).
- **Robin outer** (far-field truncation, default): the leading
  far-field term of any grounding electrode is the monopole
  $\varphi \propto 1/r$ — in a layered soil too, where only the
  amplitude changes. Its exact Dirichlet-to-Neumann map on a sphere
  is $\partial \varphi / \partial r = -\varphi / r_{\text{far}}$,
  which contributes the boundary mass matrix

  $$
  M^e \;=\; \frac{2\pi \sigma \ell}{12\, r_{\text{far}}}
  \begin{pmatrix} 3 s_1 + s_2 & s_1 + s_2 \\
                  s_1 + s_2 & s_1 + 3 s_2 \end{pmatrix}
  $$

  per boundary edge of length $\ell$ between radii $s_1, s_2$
  (exact for linear hat functions). Grounding the truncation
  sphere instead ($\varphi = 0$, available as
  `far_field="dirichlet"`) short-circuits the remaining half-space
  and biases $R$ low by $\approx a_{\text{eq}}/r_{\text{far}}$;
  the DtN condition removes that term — it is satisfied identically
  by $\varphi = a_{\text{eq}}/r$ — and leaves only the multipole
  residual $O((a_{\text{eq}}/r_{\text{far}})^3)$.
- **Neumann surface** (insulating air): the natural boundary
  condition of the weak form takes care of
  $\partial \varphi / \partial z = 0$ at $z = 0$, which the shell
  mesh places exactly on the $\vartheta = \pi/2$ node line.

The Dirichlet condition is eliminated by reducing the system to
the free-node sub-block and folding the Dirichlet contribution
into the right-hand side.

### Resistance recovery

After the unit-potential boundary problem is solved, the cluster
**conductance** is the full bilinear form of the discrete solution,

$$
\frac{1}{R} \;=\; a(\varphi, \varphi)
\;=\; \underbrace{2\pi \sum_T \sigma_T\, \bar s_T\,
|\nabla \varphi_T|^2\, |T|}_{\text{dissipation inside } r_{\text{far}}}
\;+\; \underbrace{\oint_{r_{\text{far}}}
\frac{\sigma}{r_{\text{far}}}\, \varphi^2\, dS}_{\text{power leaving the sphere}},
$$

evaluated as $\varphi^{\mathsf T} \mathbf{A} \varphi$ with
$\mathbf{A}$ the assembled system matrix. At $\varphi = 1$ V on
the electrode this is numerically identical to the injected
current, $a(\varphi, \varphi) = a(\varphi, 1)$, which the test
suite checks to machine precision. The cluster resistance
$R_{\text{cluster}} = 1/G$ is returned as the cluster impedance of
the `FieldResult`.

Because $a(\cdot, \cdot)$ is symmetric positive definite, the
discrete $\varphi$ *minimises* it over the finite-element space
(Dirichlet's principle), and $G$ *is* that minimum. A finer space
can therefore only lower the energy minimum — i.e. **lower $G$ and
raise $R$** towards the exact values — so with
$V_h \subset V_{h/2} \subset V$

$$
G_h \;\ge\; G_{h/2} \;\ge\; G, \qquad
R_h \;\le\; R_{h/2} \;\le\; R \quad \text{for nested spaces},
$$

which is why $R_h$ approaches the exact hemisphere resistance
**monotonically from below** rather than oscillating around it.
Refining as $n \mapsto 2n - 1$ in both directions keeps the spaces
nested. Measured on the reference rod, $G_h = 0.015640831 \to
0.015516853 \to 0.015485931 \to 0.015478205$ against
$G = 0.0154757$ — a decreasing sequence, exactly as the inequality
above requires.

### Per-electrode current split

Within a cluster the engine splits the cluster current onto the
member electrodes proportionally to their individual hemisphere
conductances:

$$
I_e \;=\; I_{c,\text{in}} \cdot
\frac{a_{\text{eq}, e}}{\sum_{e' \in c} a_{\text{eq}, e'}}.
$$

This is the parallel-conductance rule applied to hemispheres. For
a cluster of identical electrodes it splits the current evenly
(physically expected); for a heterogeneous cluster it weights
toward the lower-resistance electrodes.

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `HomogeneousSoil`, `TwoLayerSoil`, `MultiLayerSoil` |
| Frequency | quasi-static, frequency-independent |
| Geometry coverage | rods, rings, mesh — all reduced to equivalent hemispheres |
| Cluster topology | per-cluster reduction; no inter-cluster coupling |
| Mesh resolution | 60 radial × 40 transverse node lines (default) |
| Truncation | sphere at $r_{\text{far}} = $ `r_far_factor` $\cdot \bar L$ (default 30), monopole-DtN boundary |
| Discretisation error | $< 0.3\,\%$ at the default mesh, $O(h^2)$ |

## Convergence and cost

### Mesh convergence

For homogeneous soil the discrete problem has a known exact answer —
the analytic hemisphere resistance
$R = \rho / (2\pi a_{\text{eq}})$ — and the solver converges to it
monotonically from below at second order. Single rod, $L = 1.5$ m,
$r_w = 5$ mm, $\rho = 100\,\Omega\text{m}$
($a_{\text{eq}} = 0.2463023$ m, $R = 64.61772\,\Omega$), refining
as $n \mapsto 2n - 1$ so that the finite-element spaces stay nested:

| $n_{\text{radial}} \times n_{\text{axial}}$ | nodes | $R_h$ [Ω] | error | order | wall |
|---|---|---|---|---|---|
| 15 × 10 | 150 | 61.9523 | $-4.12\,\%$ | — | 6 ms |
| 29 × 19 | 551 | 63.9352 | $-1.06\,\%$ | 1.97 | 11 ms |
| 57 × 37 | 2 109 | 64.4461 | $-0.266\,\%$ | 1.99 | 31 ms |
| 113 × 73 | 8 249 | 64.5747 | $-0.0665\,\%$ | 2.00 | 134 ms |
| 225 × 145 | 32 625 | 64.6070 | $-0.0166\,\%$ | 2.00 | 680 ms |
| 449 × 289 | 129 761 | 64.6150 | $-0.0042\,\%$ | 2.00 | 4.5 s |

The error is strictly negative and shrinks by a factor of four per
refinement — the signature of a conforming boundary plus the
minimum-energy bound. Plain doubling (60 × 40 → 120 × 80 → …) gives
the same $O(h^2)$ rate, only without the strict nesting guarantee.
The default 60 × 40 mesh sits at $-0.24\,\%$.

### Truncation

With the monopole-DtN (Robin) sphere the truncation is no longer the
limiting error: at 60 × 40 the result moves from $-0.065\,\%$ at
$r_{\text{far}} = 15\,a_{\text{eq}}$ to $-0.51\,\%$ at
$r_{\text{far}} = 1500\,a_{\text{eq}}$ — and it *degrades* with a
larger domain, because the graded ladder is stretched over more
decades. A grounded ($\varphi = 0$) sphere instead needs
$r_{\text{far}} \gtrsim 500\,a_{\text{eq}}$ to reach the same
accuracy, since its error decays only as
$a_{\text{eq}}/r_{\text{far}}$ ($-6.7\,\%$ at
$15\,a_{\text{eq}}$, $-0.90\,\%$ at $150\,a_{\text{eq}}$).

### Other error sources

- **Equivalent-hemisphere bias.** With the discretisation error
  below $0.3\,\%$, this is now the *only* significant error of the
  backend. $\le 10\,\%$ for rods, $\le 5\,\%$ for thin shallow
  meshes; for rings the bias depends on the
  ring-radius / wire-radius ratio. In layered soil the reduction is
  only meaningful while the hemisphere does not straddle an
  interface: for $h_1 \gtrsim 8\,a_{\text{eq}}$ the FEM agrees with
  `image_2layer` on a rod to $\le 3\,\%$ over
  $\rho_2/\rho_1 \in [0.1, 10]$, whereas for
  $h_1 \lesssim 2\,a_{\text{eq}}$ the reduced hemisphere no longer
  represents the rod's penetration into the second layer and the
  deviation reaches a factor of two (a 1.5 m rod through
  $h_1 = 0.5$ m into $\rho_2 = 10\,\Omega\text{m}$: 45 Ω against
  19 Ω from `image_2layer`) — a property of the geometric
  reduction, not of the mesh. Prefer `image_2layer`, `mom` or
  `mom_sommerfeld` for thin-top-layer geometries.
- **Sparse-solve cost.** Dominated by scipy's sparse LU; for the
  default mesh ($\sim 2400$ nodes) mesh build plus solve completes
  in $\sim 15$ ms.
- **Uniform-$\rho$ collapse.** When $\rho$ is uniform across all
  layers the result must be independent of the declared layer
  thicknesses. The interface cut leaves only the logarithmic loss
  of radial resolution from the larger $r_{\text{far}}$: over
  $h_1 \in [0.5, 100]$ m the cluster resistance of the reference rod
  stays within $0.61\,\%$ of the homogeneous answer (checked in
  `tests/test_pass9_fem.py`).

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| $\rho / (2\pi a_{\text{eq}})$, homogeneous | $\le 0.3\,\%$, $O(h^2)$ | the discretisation itself (analytic answer) |
| `image` ($n = 1$) | $\le 10\,\%$ | volume PDE vs. integral equation |
| `image_2layer` ($n = 2$), $h_1 \gtrsim 8 a_{\text{eq}}$ | $\le 10\,\%$ | layered PDE vs. closed-form image series |
| `cim` (any $n$) | $\le 10\,\%$ | layered PDE vs. CIM |
| Layer-contrast monotonicity | strict | $\rho_2 \uparrow \Rightarrow R_{\text{cluster}} \uparrow$ |
| Uniform-$\rho$ layer collapse | $\le 1\,\%$ | declared $h_i$ must not change a homogeneous answer |

The 10 % envelope is the price of the equivalent-hemisphere
reduction, and — since 0.15.0 — nothing else: the homogeneous row
above pins the numerics against a closed-form answer, so a
disagreement with an integral engine can be attributed to the
reduction. The engine's role is **methodological independence**:
when an integral engine and FEM agree to within 10 %, the kernel
and the volume PDE are giving consistent physics. Disagreements
beyond that envelope point to the reduction itself (in layered soil
first of all to a hemisphere straddling the interface), not to the
underlying physics.

## Roadmap

A full 3-D FEM (via `scikit-fem` or comparable) without the
equivalent-hemisphere reduction is on the roadmap as a future
upgrade. It would:

- Cover multi-cluster volume worlds (currently every cluster is
  reduced separately and the per-cluster meshes do not "see" each
  other).
- Eliminate the $\le 10\,\%$ reduction bias.
- Cost one to two orders of magnitude more in mesh-build and solve
  time.

The current axisymmetric implementation is sufficient for typical
use cases and provides the volume-PDE cross-check at minimal
implementation cost. Upgrading to a full 3-D FEM is deferred until a
concrete use case demands it.

## References

- **Güemes, J. A. & Hernando, F. E.** (2004). Method for
  calculating the ground resistance of grounding grids using FEM.
  *IEEE PWRD* 19(2). The reference paper for FEM in grounding
  analysis.
- **Sunde, E. D.** (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover, ch. 2.1. Equivalent-hemisphere reduction
  formulas.
- **Dwight, H. B.** (1936). Calculation of resistances to ground.
  *AIEE Transactions* 55. The closed-form $R_{\text{Dwight}}$
  formulas used to compute the equivalent-hemisphere radius.
- **Reddy, J. N.** (2005). *An Introduction to the Finite Element
  Method*, McGraw-Hill. The FEM textbook.
- **Givoli, D.** (1992). *Numerical Methods for Problems in Infinite
  Domains*, Elsevier. Dirichlet-to-Neumann truncation boundaries;
  the monopole DtN map used here is its lowest mode.
- **Strang, G. & Fix, G.** (2008). *An Analysis of the Finite
  Element Method*, 2nd ed. The minimum-energy principle behind the
  one-sided, monotone convergence of $R_h$.

## Example

```python
import groundfield as gf

soil = gf.HomogeneousSoil(resistivity=100.0)
world = gf.create_world(soil=soil)
gf.create_electrode(world, "rod", name="g1",
                    position=(0.0, 0.0, 0.0), length=1.5)
gf.create_source(world, attached_to="g1", magnitude=1.0)

engine = gf.create_engine(backend="fem", frequencies=[50.0])
result = world.solve(engine)
print(result.cluster_impedance("g1")[0])
print(result.metadata.get("equivalent_hemisphere_radius"))
```

Refining the mesh is a meaningful accuracy knob: the solver
approaches the analytic hemisphere resistance of the reduced
geometry from below.

```python
import math

from groundfield.solver.fem import solve_fem

for n_r, n_a in [(29, 19), (57, 37), (113, 73)]:
    res = solve_fem(world, engine, n_radial=n_r, n_axial=n_a)
    a_eq = res.metadata["equivalent_hemisphere_radius"]["g1"]
    R_h = res.metadata["fem_cluster_resistance"]["g1"]
    R_exact = 100.0 / (2.0 * math.pi * a_eq)
    print(f"{n_r:4d} x {n_a:3d}  R = {R_h:8.4f} Ohm  "
          f"({100.0 * (R_h - R_exact) / R_exact:+.3f} %)")
#   29 x  19  R =  63.9352 Ohm  (-1.056 %)
#   57 x  37  R =  64.4461 Ohm  (-0.266 %)
#  113 x  73  R =  64.5747 Ohm  (-0.067 %)
```

## API reference

::: groundfield.solver.fem

## Related material

- ADR-0002 — engine selection heuristic; the FEM is the volume-PDE
  cross-check.
