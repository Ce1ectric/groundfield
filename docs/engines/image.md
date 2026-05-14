# `image` — homogeneous image-charge sum

## Physical context

A grounding electrode embedded in a homogeneous half-space of
resistivity $\rho$ injects a current $I$ into the soil. The
quasi-static potential field is governed by Laplace's equation
($-\nabla \cdot (\sigma \nabla \varphi) = q$ with $\sigma = 1/\rho$
and $q$ the source-current density), with an insulating boundary at
the soil surface ($\partial\varphi/\partial z = 0$ at $z = 0$) and
$\varphi \to 0$ at infinity.

For frequencies $f < 1\,\text{kHz}$ the displacement-current term
is negligible — the relaxation time of moist soil
($\tau = \varepsilon/\sigma$) is on the order of 100 ns, well below
the millisecond regime. The static potential field is therefore
representative of the entire quasi-static frequency window.

## Governing equation: image-charge solution

A point current source $I$ at depth $z_s > 0$ in a homogeneous
half-space satisfies the Neumann boundary at $z = 0$ exactly through
the **image-charge construction**: place a virtual source of
identical strength at the mirror position $z = -z_s$. The
superposition of source and image gives

$$
\varphi(x, y, z) \;=\; \frac{\rho\, I}{4\pi}
\left( \frac{1}{r} + \frac{1}{r'} \right),
\qquad
r  = \sqrt{(x{-}x_s)^2 + (y{-}y_s)^2 + (z{-}z_s)^2},
$$

with $r' = \sqrt{(x{-}x_s)^2 + (y{-}y_s)^2 + (z{+}z_s)^2}$ the
distance to the air-mirrored image.

This is the smallest, cleanest closed form in the engine family and
provides the baseline against which every layered engine collapses
when its layer contrast vanishes.

## Numerical strategy

### Wire-segment discretisation

A finite electrode (rod, ring, mesh) is discretised into $N$
collinear segments of length $L_i \le \Delta s$ (the
`engine.segment_length` parameter). Each segment carries one point
current source at its midpoint. The total current of an electrode
is distributed **uniformly per unit length** across its segments —
i.e. the segment current $I_i = I_{\text{electrode}} \cdot L_i / \sum_j L_j$.

The uniform-current ansatz is an approximation: the true current
distribution along a wire is non-uniform, with end-point
concentrations on the order of $\sim 5\,\%$. This residual is
handled either by accepting a $\sim 5\,\%$ Dwight-bias on the input
impedance (cheap), or by switching to the [`mom`](mom.md) backend
which solves for the actual distribution at $O(N^3)$ cost.

### Self-action correction

For the average-potential evaluation at segment midpoints, the
diagonal of the kernel matrix carries a $1/r$ singularity that the
point-source representation cannot handle. We replace the
direct-source self-distance by the **analytical line
self-potential**

$$
\varphi_{\text{self,line}} \;=\; \frac{\rho\, I_i}{2\pi\, L_i}\,
\ln\!\frac{L_i}{a_i},
$$

with $a_i$ the wire radius. This is the classic Howe / Sunde
average-potential formula for a thin wire of finite length. The
image contribution at the same segment carries no singularity (the
image is at $z = -2 z_s$, distance $2 z_s$ away), so a point-source
evaluation suffices there.

### Cluster constraints

Multiple electrodes connected by a `Conductor` form a *galvanic
cluster* with a shared (unknown) cluster potential $\varphi_c$ and
a known total injected current $I_{c,\text{in}} = \sum_{e\in c} I_{\text{src},e}$. The current sharing within the cluster is solved
through the multi-port grounding matrix $Z_{ij}$ (average potential
at electrode $i$ for unit current at electrode $j$):

$$
\begin{bmatrix} Z & -C \\ C^{\top} & 0 \end{bmatrix}
\begin{bmatrix} I \\ \varphi_c \end{bmatrix}
= \begin{bmatrix} 0 \\ I_{\text{in}} \end{bmatrix},
$$

with $C$ the cluster-membership indicator. The first $N$ rows
enforce $\varphi_i = \varphi_c$ for every electrode in cluster $c$;
the last $K$ rows enforce $\sum_{i \in c} I_i = I_{c,\text{in}}$.

### Postprocessing

After the cluster currents are known, every segment current is
fixed by the uniform-per-unit-length rule. Field-point evaluations
(profiles, contours, transferred potentials) reuse the same kernel
$1/r + 1/r'$ at the actual field point. The
[`FieldResult.potential`](../api/solver.md) helper is a thin
wrapper around this evaluation.

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `HomogeneousSoil` only |
| Frequency | quasi-static, $f < 1\,\text{kHz}$ |
| Wire radius | $a \ll L_i$ (thin-wire) |
| Segment length | $L_i \lesssim a_{\text{eq}} / 5$ for stable averaging |
| Air boundary | insulating (Neumann at $z = 0$) |
| Far-field | $\varphi \to 0$ as $|\mathbf{r}| \to \infty$ |

## Convergence and cost

- **Discretisation error.** The uniform-per-unit-length ansatz
  carries a $\sim 4{-}5\,\%$ residual compared to the Sunde rod
  formula at the canonical 1.5 m / 5 mm rod, and shrinks to
  $< 1\,\%$ at sub-centimetre segment lengths and short rods.
- **Computational cost.** $O(N^2)$ matrix build for the cluster
  reaction matrix; $O(K^3)$ for the constraint solve, where $K$ is
  the cluster count (typically 1–3). For typical geometries with
  $N \le 10^3$, the homogeneous engine completes in milliseconds.
- **Numerical singularity.** Distances below `_MIN_DISTANCE = 1 mm`
  are clamped at the floor to keep the kernel finite during plot
  evaluations near the wire axis.

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| Dwight 1936 closed forms | $\le 10\,\%$ | rod / ring / mesh DC resistance |
| `mom` (Galerkin) | $\le 2\,\%$ | same kernel, different test function |
| `image_2layer` at $K = 0$ | bit-exact | layered family collapses to homogeneous |
| `cim` at $n = 1$ | bit-exact | matrix-pencil fit returns $P = 0$, kernel matches |
| `mom_sommerfeld` at $n = 1$ | bit-exact | quadrature short-circuits to closed form |
| `fem` (axisymmetric volume PDE) | $\le 10\,\%$ | reduction to equivalent hemisphere |

These bounds are codified as parametric pytest fixtures; see
`tests/test_cross_engines.py` and
`tests/test_cross_engines_extended.py`.

## References

- **Sunde, E. D.** (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover. Chapter 2 — image-charge construction and
  average-potential method.
- **Dwight, H. B.** (1936). Calculation of resistances to ground.
  *AIEE Transactions* 55. Reference DC resistances for canonical
  geometries.
- **Tagg, G. F.** (1964). *Earth Resistances*, Pitman. The
  practitioner's reference for image methods.

## Example

```python
import groundfield as gf

soil = gf.HomogeneousSoil(resistivity=100.0)
world = gf.create_world(soil=soil)
gf.create_electrode(world, "rod", name="g1",
                    position=(0.0, 0.0, 0.0), length=1.5)
gf.create_source(world, attached_to="g1", magnitude=1.0)

engine = gf.create_engine(backend="image",
                          segment_length=0.05,
                          frequencies=[50.0])
result = world.solve(engine)
print(result.cluster_impedance("g1")[0])
```

## API reference

::: groundfield.solver.image

## Related material

- ADR-0001 documents why this homogeneous engine sits at the root
  of the engine family.
- Notebook `01_smoke_test.ipynb` exercises the full API of this
  backend on a single rod and on a two-electrode cluster.
