# `image_2layer` — Tagg / Sunde series for 2-layer soil

## Physical context

The 2-layer model is the canonical typical layered-soil description: a finite
upper layer of thickness $h_1$ and resistivity $\rho_1$ over a
semi-infinite lower layer of resistivity $\rho_2$. It is the
simplest soil that captures the dominant first-order effect of
heterogeneity — a frozen, weathered, or saturated surface layer
above a different sub-stratum — without introducing the
identification problem that more layers create. A typical
parameter range is $\rho_1 \in [50, 1000]\,\Omega\, \text{m}$,
$\rho_2 \in [10, 5000]\,\Omega\,\text{m}$,
$h_1 \in [0.5, 5]\,\text{m}$.

The single design degree of freedom that controls the layered
behaviour is the **interface reflection coefficient**

$$
K_1 \;=\; \frac{\rho_2 - \rho_1}{\rho_2 + \rho_1} \in (-1, +1).
$$

For $\rho_2 < \rho_1$ ($K_1 < 0$, conductive bottom) the lower
layer pulls current away from the surface and the cluster impedance
falls. For $\rho_2 > \rho_1$ ($K_1 > 0$, resistive bottom) the
opposite holds. The limiting cases $K_1 \to \pm 1$ correspond to a
perfect insulator below or a perfect conductor below.

## Governing equation: Tagg / Sunde geometric series

A point current source $I$ at depth $z_s \in (0, h_1)$ produces a
potential field that satisfies the Neumann boundary at $z = 0$ and
the continuity-of-current boundary at $z = h_1$. The
spectral-domain solution involves the Sommerfeld integral

$$
\varphi(s, z) \;=\; \frac{\rho_1\, I}{4\pi}
\int_0^{\infty}
\frac{ e^{-\lambda |z-z_s|}
     + e^{-\lambda (z+z_s)}
     + K_1\, e^{-\lambda (2 h_1 - z - z_s)}
     + K_1\, e^{-\lambda (2 h_1 - |z-z_s|)} }
     { 1 - K_1\, e^{-2\lambda h_1} } \,
J_0(\lambda s)\, d\lambda.
$$

Because $\Gamma_1(\lambda) \equiv K_1$ is constant in $\lambda$
(only one interface contributes) the multiple-reflection multiplier
$1 / (1 - K_1 e^{-2\lambda h_1})$ expands as a **geometric series**
in $K_1 e^{-2\lambda h_1}$, and the Sommerfeld integral collapses
into a closed-form sum of point sources at mirrored positions.

After substitution and simplification, the spatial form is the
classical Tagg / Sunde series:

$$
\varphi(s, z) \;=\; \frac{\rho_1\, I}{4\pi}
\Biggl[
   \tfrac{1}{r_0^+} + \tfrac{1}{r_0^-}
 + \sum_{n=1}^{\infty} K_1^n
   \Bigl(\tfrac{1}{r_n^{++}} + \tfrac{1}{r_n^{+-}}
       + \tfrac{1}{r_n^{-+}} + \tfrac{1}{r_n^{--}}\Bigr)
\Biggr],
$$

with the image distances

$$
r_n^{\sigma\tau} \;=\;
\sqrt{(x{-}x_s)^2 + (y{-}y_s)^2
      + (z - \sigma\,2 n h_1 - \tau\,z_s)^2},
\qquad \sigma, \tau \in \{+1, -1\}.
$$

Per index $n$ four images are placed at $z_n^{\sigma\tau} = \sigma\,2 n h_1 + \tau\,z_s$ with weight $K_1^n$. The $n=0$ term
contributes two images (the original source at $z = +z_s$ and the
air mirror at $z = -z_s$, both with weight 1) — this is exactly the
homogeneous image-charge solution.

## Numerical strategy

### Series truncation

The series is truncated at the smallest index for which
$|K_1|^n < \text{tol}$ (default $\text{tol} = 10^{-6}$) or at
`max_terms` (default $200$), whichever happens first. For
practically observed contrasts ($|K_1| \le 0.96$, i.e. $\rho_2 / \rho_1 \le 50$), the tolerance criterion is reached in $\le 200$ terms.
For harder contrasts (closer to $|K_1| = 1$), the convergence slows
significantly and the per-segment cost grows linearly with the
required number of image terms.

When `max_terms` is reached without meeting the tolerance,
`FieldResult.metadata['converged']` is set to `False` and a warning
is logged.

### Self-action

The wire-segment self-action splits cleanly:

- The $n = 0$ direct contribution carries the singular line
  self-potential — handled by the homogeneous formula
  $2 \ln(L_i / a_i) / L_i$.
- The $n = 0$ air-mirror term and all $n \ge 1$ image terms are at
  least $2 z_s$ (or $2 h_1 - z - z_s$, etc.) away from the segment
  midpoint, so the point-source approximation is safe.

This is the same construction as in the homogeneous backend, with
$\rho \to \rho_1$ as the prefactor and the geometric image series
added on top.

### Auto-dispatch

`Engine.solve` picks `image_2layer` automatically when the user
passed `backend="image"` and `world.soil` is a `TwoLayerSoil`. A
notebook written for the homogeneous backend therefore keeps
working when only the soil model is swapped — no string change
required.

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `TwoLayerSoil` only |
| Frequency | quasi-static, $f < 1\,\text{kHz}$ |
| Electrode placement | every segment must have $z < h_1$ (raises `ValueError` otherwise) |
| Reflection coefficient | $|K_1| < 1$; convergence slows as $|K_1| \to 1$ |
| Series truncation | adaptive in $|K_1|^n < \text{tol}$, capped at `max_terms` |
| Wire radius / segment ratio | thin-wire, $a \ll L_i$ |

The hard constraint that all segments lie inside the upper layer is
a direct consequence of the series construction: the image
positions $\pm 2 n h_1 \pm z_s$ are derived assuming the source is
*above* the interface. A segment at $z = h_1$ would put a source
*on* the interface and break the geometric expansion.

## Convergence and cost

- **Series convergence.** Geometric in $|K_1|$:
  - $|K_1| = 0.5$ → 17 terms for $10^{-6}$ accuracy.
  - $|K_1| = 0.8$ → 62 terms.
  - $|K_1| = 0.9$ → 131 terms.
  - $|K_1| = 0.95$ → 270 terms (above the default `max_terms`).
- **Per-segment cost.** $O(N^2 \cdot M)$ kernel evaluations with $M$
  the truncated series length. For typical cases with $N \le 1000$ and
  $M \le 200$ the engine still runs in seconds.
- **Reduction to homogeneous.** At $K_1 = 0$ the engine collapses
  to the homogeneous backend bit-exactly — the only series term
  with non-zero weight is $n = 0$ with weight $1$, identical to the
  `image` backend with $\rho = \rho_1$.

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| `image` at $K_1 = 0$ | bit-exact | series collapse |
| `mom` on the same world | $\le 2\,\%$ | same kernel, different resolution scheme |
| `cim` on the same world | bit-exact (CIM falls back on this kernel for $n = 2$) | `cim` deliberately reuses the Tagg / Sunde self-kernel for $n = 2$ |
| `bem` on the same world | bit-exact (BEM falls back on this kernel for $n = 2$) | same reason |
| `mom_sommerfeld` on the same world | $\le 5\,\%$ | quadrature reproduces the geometric series |
| Sunde / Dwight closed forms | $\le 10\,\%$ | reference values for canonical geometries (rods, rings) |

The engine's role inside the cross-validation matrix is to be the
**closed-form anchor**: every other engine is checked against it
for $n = 2$.

## References

- **Tagg, G. F.** (1964). *Earth Resistances*, Pitman, ch. 5. The
  classical statement of the geometric image series.
- **Sunde, E. D.** (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover, sect. 3.5. Derivation of the multi-layer
  Sommerfeld integral and the 2-layer reduction.
- **Stefanescu, S. & Schlumberger, C.** (1930). *Sur la
  distribution électrique potentielle autour d'une prise de terre
  ponctuelle*, Journal de Physique. Original geophysical
  formulation of the image-charge series.
- **Dawalibi, F. P. & Barbeito, N.** (1991). Measurements and
  computations of the performance of grounding systems buried in
  multilayer soils. *IEEE PWRD* 6(4) — extension to $n$ layers and
  modern computational practice.
- **Colominas, I., París, J., Navarrina, F. & Casteleiro, M.**
  (2012). Improvement of computer methods for grounding analysis in
  layered soils by using high-efficient convergence acceleration
  techniques. *Adv. Eng. Soft.* 44 — Aitken / Pade acceleration of
  the series for $|K_1| \to 1$.

## Example

```python
import groundfield as gf

soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
world = gf.create_world(soil=soil)
gf.create_electrode(world, "ring", name="g1",
                    center=(0.0, 0.0, 0.8), radius=5.0,
                    wire_radius=0.005)
gf.create_source(world, attached_to="g1", magnitude=1.0)

engine = gf.create_engine(backend="image_2layer",
                          segment_length=0.1,
                          frequencies=[50.0])
result = world.solve(engine)
print(result.cluster_impedance("g1")[0])
```

## API reference

::: groundfield.solver.image_2layer

## Related material

- ADR-0001 — original methodology decision.
- Notebook `02_two_layer.ipynb` — parameter sweep over $K_1$ and
  $h_1$, trumpet comparison homogeneous vs. 2-layer, exact $K_1=0$
  collapse.
