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

with insulating boundary at the soil surface ($\partial \varphi / \partial z = 0$ at $z = 0$) and a Dirichlet far-field
($\varphi \to 0$ as $|\mathbf{r}| \to \infty$, truncated to a
finite outer radius $R_{\text{far}}$). The source-current density
$q$ is concentrated on the electrode surfaces.

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

For work-package-1 reference electrodes (rod, ring, mesh) the
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

## Numerical strategy

### Mesh construction

A 2-D structured mesh is built on
$(s, z) \in [0, R_{\text{far}}] \times [0, Z_{\text{far}}]$ with

- **Radial nodes**: log-spaced from $0.05\,a_{\text{eq}}$ to
  $R_{\text{far}}$, prepended by $s = 0$.
- **Axial nodes**: linearly spaced, plus an explicit z-line at
  every layer interface so that the conductivity step is
  mesh-aligned.
- **Truncation factors**: $R_{\text{far}} = 30 \cdot \bar L$,
  $Z_{\text{far}} = 20 \cdot \bar L$, with $\bar L = a_{\text{eq}} + \sum_i h_i$.

Each grid cell is split into two right triangles. Element
centroids are tagged with the layer index, and the per-element
conductivity is taken from the `LayerStack`.

### Stiffness assembly

The element stiffness uses linear hat functions on each triangle.
The per-element 3×3 matrix entries are built from the gradient
vectors $\nabla \phi_i = (b_i, c_i)/(2 |T|)$, weighted by
$\sigma_T \cdot 2\pi \bar s_T$ as derived above. The global
matrix is assembled in COO format and converted to CSR for the
sparse solve.

### Boundary conditions

- **Dirichlet inner** (electrode surface): every node within
  $\sqrt{s^2 + z^2} \le a_{\text{eq}}$ is fixed at $\varphi = 1$
  (the unit-potential probe).
- **Dirichlet outer** (far-field truncation): every node on the
  outer boundary of the mesh is fixed at $\varphi = 0$.
- **Neumann surface** (insulating air): the natural boundary
  condition of the weak form takes care of $\partial \varphi / \partial z = 0$ at $z = 0$.

The Dirichlet conditions are eliminated by reducing the system to
the free-node sub-block and folding the Dirichlet contribution
into the right-hand side.

### Resistance recovery

After the unit-potential boundary problem is solved, the cluster
**conductance** is the integrated dissipation:

$$
\frac{1}{R} \;=\; \int_{\Omega} \sigma\, |\nabla \varphi|^2\, dV
\;=\; 2\pi\, \sum_T \sigma_T\, \bar s_T\,
|\nabla \varphi_T|^2\, |T|,
$$

with the gradient $\nabla \varphi_T$ piecewise constant on each
triangle. The cluster resistance $R_{\text{cluster}} = 1/G$ is
returned as the cluster impedance of the `FieldResult`.

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
| Mesh resolution | 60 radial × 40 axial nodes (default) |
| Truncation factors | $R_{\text{far}} = 30 \bar L$, $Z_{\text{far}} = 20 \bar L$ |

## Convergence and cost

- **Mesh discretisation.** The default mesh gives $\sim 5\,\%$
  agreement with `image` on a single rod in homogeneous soil.
  Doubling the mesh resolution shrinks the bias to $\sim 1\,\%$
  but increases solve time by an order of magnitude.
- **Equivalent-hemisphere bias.** Documented in the test suite as
  $\le 10\,\%$ for rods (the Dwight rod formula and the hemisphere
  formula are within that envelope already), $\le 5\,\%$ for thin
  shallow meshes. For rings the bias depends on the
  ring-radius / wire-radius ratio.
- **Sparse-solve cost.** $O(N \log N)$ thanks to scipy's sparse
  LU; for the default mesh ($\sim 2400$ nodes) the solve completes
  in milliseconds.
- **Reduction.** When ρ is uniform across all layers, the FEM
  collapses to the homogeneous PDE on the same mesh; the only
  residual is the explicit z-line at the layer interfaces,
  bounded at $\sim 25\,\%$ on the cluster impedance (documented in
  `tests/test_fem.py::test_fem_two_layer_K_zero_collapses`).

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| `image` ($n = 1$) | $\le 10\,\%$ | volume PDE vs. integral equation |
| `image_2layer` ($n = 2$) | $\le 10\,\%$ | layered PDE vs. closed-form image series |
| `cim` (any $n$) | $\le 10\,\%$ | layered PDE vs. CIM |
| Layer-contrast monotonicity | strict | $\rho_2 \uparrow \Rightarrow R_{\text{cluster}} \uparrow$ |

The 10 % envelope is the price of the equivalent-hemisphere
reduction. The engine's role is **methodological independence**:
when an integral engine and FEM agree to within 10 %, the kernel
and the volume PDE are giving consistent physics. Disagreements
beyond that envelope point to the reduction itself, not the
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

The current axisymmetric implementation is sufficient for the AP1
work package and provides the volume-PDE cross-check at minimal
implementation cost. Upgrading to a full 3-D FEM is deferred until
AP1 demands it.

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

## Related material

- API reference: `groundfield.solver.fem`,
  `groundfield.solver.fem.equivalent_hemisphere_radius`.
- ADR-0002 — engine selection heuristic; the FEM is the volume-PDE
  cross-check.
- Notebook `08_fem.ipynb` — equivalent-hemisphere visualisation,
  single rod and bonded-rod cluster, layer-contrast trend, mesh-
  refinement sanity check.
