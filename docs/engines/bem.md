# `bem` — Boundary-Element Collocation

!!! warning "Status: $n \le 2$ only, and numerically identical to `mom`"

    Since 0.11.0 `bem` **rejects** $n \ge 3$ soils with
    `NotImplementedError` (it shared the structurally incomplete
    complex-image kernel with [`cim`](cim.md) — audit 2026-07-08,
    WP-E), and since 0.15.0 it computes no complex-image fit at all.
    In the two regimes it accepts it assembles the **same reaction
    matrix** as [`mom`](mom.md) and solves it with the **same**
    constraint solver; the measured relative difference on a ring case
    is 4e-16. `bem` is therefore **not an independent cross-check** of
    `mom` — treating "`mom` and `bem` agree" as corroboration counts
    one computation twice (ADR-0002 amendment 2026-07-09). It is kept
    as the collocation-flavoured entry point of the family.

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
alternative** to the Galerkin scheme. That intent is only partly
realised: on the diagonal both engines use the same analytical line
self-potential and off the diagonal both reduce to a point-source
evaluation at the segment midpoint, so on the soils `bem` accepts
($n \le 2$) the two assembled matrices coincide to floating-point
noise. The distinction between Galerkin and collocation is real in
theory but has no numerical footprint in this implementation — see
the status box above.

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
  (`_two_layer_self_kernel_factory`, with
  `allow_cross_layer=True` so interface-crossing geometries take the
  rigorous ADR-0007 path). At $n = 2$ the constant
  $\Gamma_1 \equiv K_1$ *is* a single complex image at
  $\beta = 0$, so the geometric series is the exact complex-image
  representation — there is nothing for a fit to add.
- **`MultiLayerSoil`** ($n \ge 3$) → `NotImplementedError`. The
  historic contribution for this regime came from an incomplete
  Green's function (no $2 h_1$ image families, no
  surface-interface multiple-reflection denominator; see
  `solver/_layered.py`). Use `mom_sommerfeld` or `fem`.

No complex-image fit is computed. Up to 0.14.1 `solve_bem` called
`fit_complex_images` on every solve and published its (failed)
diagnostics as `cim_n_images` / `cim_rms`, although the only
reachable branches never looked at them (review pass 9, F34). The
metadata now records `cim_fit_used = False`, `cim_n_images = 0`,
`cim_rms = None` and `reduces_to = "mom"`.

The end result: `bem` and `mom` differ only in the *test function*,
and on the soils `bem` accepts even that difference cancels — the
*matrices* are identical.

### Reaction matrix assembly

For the homogeneous and 2-layer cases the assembly is one call to
the existing self-kernel factory with the identity matrix as the
"currents" argument — the resulting matrix is exactly $Z$. The
diagonal carries the line self-potential; the off-diagonals carry
the point-source approximation.

The $n \ge 3$ branch used to add a complex-image contribution

$$
Z^{(\text{layered})}_{ij} \;=\; Z^{(\text{hom})}_{ij}
+ \frac{\rho_1}{4\pi}\,
\sum_{k=1}^{P} \frac{a_k}{\sqrt{s_{ij}^2
                       + (z_i + z_j + 2\beta_k)^2}},
$$

with $s_{ij}$ the radial distance between the segment midpoints
and $a_k, \beta_k$ the matrix-pencil fit coefficients. That branch
has been unreachable since 0.11.0 (the $n \ge 3$ rejection sits in
front of it) and was **deleted in 0.15.0**: the sum represents only a
single image family of the layered Green's function, so keeping it
alive invited a silent wrong-physics path. The matrix builder now
raises for $n \ge 3$ instead. The formula stays documented here
because it is the shape a *complete* $n \ge 3$ kernel would take
once the missing families are added.

### Linear-system solve

The cluster augmenting rows and the Galerkin solve are reused
from [`mom`](mom.md) — the only difference between `mom` and `bem`
in the assembled system is the matrix entries themselves.

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `HomogeneousSoil`, `TwoLayerSoil` (a `MultiLayerSoil` is accepted only while it reduces to $n \le 2$) |
| Frequency | quasi-static, $f < 1\,\text{kHz}$ |
| Number of layers | $n \le 2$; $n \ge 3$ raises `NotImplementedError` |
| Electrode placement | free — $n = 2$ interface crossings dispatch to the ADR-0007 cross-layer kernel |
| Wire radius / segment ratio | thin-wire, $a \ll L_i$ |
| Mesh size $N$ | $\le 1000$ at acceptable runtime |
| Number of complex images $P$ | not applicable — no fit runs |

## Convergence and cost

- **Per-segment accuracy.** Equal to `mom` on the soils this backend
  accepts: the assembled matrices agree to 4e-16, so the two engines
  converge along the segment-length axis in lockstep. (The textbook
  difference between collocation and Galerkin — slightly faster
  convergence, slightly higher sensitivity to the wire-radius /
  segment-length ratio at wire ends — would only appear with a
  genuinely averaged Galerkin kernel.)
- **Computational cost.** $O(N^2)$ matrix build, $O((N + K)^3)$
  solve.
- **Reduction.** At $K_1 = 0$ the engine collapses bit-exactly to
  the homogeneous `bem` solution, which itself agrees with `image`
  to within the segment-discretisation envelope.

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| `mom` ($n \le 2$) | bit-exact (4e-16) | *same computation* — identical reaction matrix and constraint solve; no independent information |
| `image` ($n = 1$) | $\le 5\,\%$ | uniform-current vs. collocation weighting |
| `image_2layer` ($n = 2$) | $\le 5\,\%$ | uniform-current vs. collocation weighting on the same Tagg / Sunde kernel |
| `mom_sommerfeld` ($n \le 2$) | $\le 5\,\%$ | direct quadrature of the full layered Green's function — the genuinely independent kernel |
| `fem` ($n \le 2$) | $\le 10\,\%$ | volume PDE, independent problem form |
| Sunde / Dwight closed forms | $\le 5\,\%$ | tighter than the image backend |

`bem` is a **flavour** of the integral-equation family, not an
independent line of defence: the historic "`mom` / `bem` / `cim`
triangle" collapses to a single point, because `bem` reproduces
`mom` bit-for-bit and `cim` reproduces `image_2layer` bit-for-bit.
The independent checks are `mom_sommerfeld` (quadrature of the full
layered kernel) and `fem` (volume PDE); see the [ADR-0002
amendment](../adr/0002-engine-family.md).

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
print(result.metadata["cim_fit_used"])   # False -- no fit runs
print(result.metadata["reduces_to"])     # 'mom'
```

For $n \ge 3$ the call raises; use `mom_sommerfeld` (full layered
Green's function) or `fem` (volume PDE) instead.

## API reference

::: groundfield.solver.bem

## Related material

- ADR-0002 — engine selection heuristic, and the 2026-07-09
  amendment that revoked `bem`'s role as an independent
  cross-validation engine.
