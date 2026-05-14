# `bem` — Boundary-Element Collocation

## Physical context

The Boundary-Element Method is a sister to the Method of Moments:
both reduce a continuous boundary integral equation to a finite
linear system by discretising the boundary into elements (here:
wire segments) and choosing a finite set of test functions. The
two methods only differ in the choice of test function:

- **Galerkin** ([`mom`](mom.md)): test function = basis function.
  The reaction matrix entry $Z_{ij}$ is the *average* potential of
  segment $i$ caused by a unit current on segment $j$.
- **Collocation** (`bem`): test function = Dirac delta at the
  segment midpoint. The reaction matrix entry $Z_{ij}$ is the
  *point-evaluated* potential of segment $i$ at its midpoint
  caused by a unit current on segment $j$.

In the grounding literature, Colominas, Navarrina & Casteleiro
(2007, 2012) document the collocation BEM as the historically
dominant variant for layered-soil grounding analysis. It has
roughly half the cost of Galerkin per matrix entry (one integration
instead of a double integration) and gives essentially identical
accuracy on smooth electrodes. The price is slightly higher
sensitivity to the segment-length / wire-radius ratio at the wire
end-points.

`bem` was added to the engine family to provide a **methodological
alternative** to the Galerkin scheme. When `mom` and `bem` agree
on a given problem, the answer is robust against the choice of
test function; when they disagree, the disagreement is reproducible
and quantifiable.

## Governing equation: boundary integral

The same boundary integral equation as in [`mom`](mom.md):

$$
\sum_{j=1}^{N} Z_{ij}\, I_j \;=\; \varphi_c
\qquad \forall\, i \in c,
\qquad
\sum_{j \in c} I_j \;=\; I_{c,\text{in}},
$$

with the **reaction matrix entries**

$$
Z_{ij} \;=\; \frac{1}{4\pi}
\int_{\Sigma_j} G(\mathbf{r}_i, \mathbf{r}'_j)\, dS_j.
$$

The difference: $\mathbf{r}_i$ is now the *centre* of segment $i$
rather than the average over its length. The $j$-side integration
is unchanged.

For thin-wire grounding electrodes (radius $a \ll L_i$) the
$\Sigma_j$ surface integral collapses to a line integral with
appropriate kernel; for the off-diagonal entries the line integral
itself further reduces to a point-source evaluation at the segment
midpoint (the segment is short compared to its distance to the
field point). The diagonal carries the analytical line
self-potential, the same as in `mom` and `image`.

## Numerical strategy

### Kernel choice

`bem` uses the **same kernel infrastructure** as the rest of the
engine family. For each soil class:

- **`HomogeneousSoil`** → homogeneous self-kernel ($1/r + 1/r_{\text{air}}$ point-source off-diagonal, line self-potential
  on the diagonal). Bit-exact match to `image` and `mom` at the
  Galerkin level for $n = 1$.
- **`TwoLayerSoil`** → the closed-form Tagg / Sunde self-kernel
  (`_two_layer_self_kernel_factory`). The matrix-pencil-fit-based
  CIM kernel is intentionally *not* used here, even though `bem`
  takes a `fit_complex_images` instance: at $n = 2$ the
  $\Gamma_1 \equiv K_1$ constant makes the matrix-pencil fit
  ill-conditioned (a single pole at $\beta = 0$). Falling back on
  the geometric series gives a bit-exact match with
  `image_2layer`.
- **`MultiLayerSoil`** ($n \ge 3$) → homogeneous self-kernel for
  the direct + air-mirror part plus the closed-form complex-image
  contribution from `cim`. This is the same closed-form layered
  Green's function the `cim` engine uses, but evaluated with
  collocation rather than Galerkin averaging.

The end result: `bem` and `mom` differ only in the *test function*;
the *kernel* is identical for $n = 1, 2$, and shares the same CIM
approximation for $n \ge 3$.

### Reaction matrix assembly

For the homogeneous and 2-layer cases the assembly is one call to
the existing self-kernel factory with the identity matrix as the
"currents" argument — the resulting matrix is exactly $Z$. The
diagonal carries the line self-potential; the off-diagonals carry
the point-source approximation.

For $n \ge 3$ the homogeneous part is built first, then the
complex-image contribution is added explicitly:

$$
Z^{(\text{layered})}_{ij} \;=\; Z^{(\text{hom})}_{ij}
+ \frac{\rho_1}{4\pi}\,
\sum_{k=1}^{P} \frac{a_k}{\sqrt{s_{ij}^2
                       + (z_i + z_j + 2\beta_k)^2}},
$$

with $s_{ij}$ the radial distance between the segment midpoints
and $a_k, \beta_k$ the matrix-pencil fit coefficients. The
imaginary part of the sum cancels by symmetry, so we take the real
part to suppress numerical residue.

### Linear-system solve

The cluster augmenting rows and the Galerkin solve are reused
from [`mom`](mom.md) — the only difference between `mom` and `bem`
in the assembled system is the matrix entries themselves.

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `HomogeneousSoil`, `TwoLayerSoil`, `MultiLayerSoil` |
| Frequency | quasi-static, $f < 1\,\text{kHz}$ |
| Electrode placement | every segment in the upper layer |
| Wire radius / segment ratio | thin-wire, $a \ll L_i$ |
| Mesh size $N$ | $\le 1000$ at acceptable runtime |
| Number of complex images $P$ | inherited from `cim` (default 8) |

## Convergence and cost

- **Per-segment accuracy.** Comparable to `mom` for smooth
  electrodes (rods, rings, meshes); collocation converges
  somewhat faster on the segment-length axis but is slightly
  more sensitive to the wire-radius / segment-length ratio at the
  electrode ends. For the typical geometries the difference is
  negligible.
- **Computational cost.** $O(N^2)$ matrix build, $O((N + K)^3)$
  solve. For the layered case the matrix build is dominated by the
  one-shot CIM fit ($O(N_s P^2)$, $N_s = 64$, $P = 8$, so
  negligible).
- **Reduction.** At $K_1 = 0$ the engine collapses bit-exactly to
  the homogeneous `bem` solution, which itself agrees with `image`
  to within the segment-discretisation envelope.

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| `image` ($n = 1$) | $\le 5\,\%$ | uniform-current vs. collocation |
| `image_2layer` ($n = 2$) | bit-exact | bem reuses the Tagg / Sunde kernel for $n = 2$ |
| `mom` (any $n$) | $\le 5\,\%$ | Galerkin vs. collocation on the same kernel |
| `cim` ($n \ge 3$) | $\le 5\,\%$ | shares the CIM kernel; only the test function differs |
| `mom_sommerfeld` (any $n$) | $\le 5\,\%$ | quadrature reference |
| Sunde / Dwight closed forms | $\le 5\,\%$ | tighter than the image backend |

`bem` is the **alternative-weighting cross-check** in the matrix.
It is paired with `mom` and `cim` in the test suite; together they
form a triangle that detects bugs in the kernel (`mom` vs. `cim`
disagreement), the test function (`mom` vs. `bem` disagreement), or
the layered approximation (`cim` vs. `mom_sommerfeld`
disagreement).

## References

- **Colominas, I., Navarrina, F. & Casteleiro, M.** (2007).
  Numerical simulation of transferred potentials in earthing grids
  considering layered soil models. *IEEE PWRD* 22(3). Layered BEM
  for grounding systems.
- **Colominas, I., París, J., Navarrina, F. & Casteleiro, M.**
  (2012). Improvement of computer methods for grounding analysis
  in layered soils by using high-efficient convergence
  acceleration techniques. *Adv. Eng. Soft.* 44. Aitken / Pade
  acceleration of the BEM kernel; cross-checks against
  measurement.
- **Brebbia, C. A. & Dominguez, J.** (1992). *Boundary Elements:
  An Introductory Course*, McGraw-Hill. The BEM textbook.
- **Harrington, R. F.** (1968). *Field Computation by Moment
  Methods*, Macmillan. Cross-reference for the Galerkin
  alternative.

## Example

```python
import groundfield as gf

soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
world = gf.create_world(soil=soil)
gf.create_electrode(world, "rod", name="g1",
                    position=(0.0, 0.0, 0.0), length=1.5)
gf.create_source(world, attached_to="g1", magnitude=1.0)

engine = gf.create_engine(backend="bem",
                          segment_length=0.1,
                          frequencies=[50.0])
result = world.solve(engine)
print(result.cluster_impedance("g1")[0])
```

## API reference

::: groundfield.solver.bem

## Related material

- ADR-0002 — engine selection heuristic; `bem` is the
  alternative-weighting cross-check in the layered family.
- Notebook `07_bem.ipynb` — single rod, bonded-rod cluster,
  surface-potential profile, mesh-refinement convergence study.
