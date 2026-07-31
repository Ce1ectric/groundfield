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
average-potential formula for a thin wire of finite length.

The image contribution at the same segment is evaluated as a **point
image** at distance $2 z_i$,

$$
K_{ii}^{\text{img}} \;=\; \frac{1}{2 z_i}
\;=\; \lim_{L_i \to 0}\;
\frac{2}{L_i}\, \operatorname{arsinh}\!\frac{L_i}{4 z_i},
$$

i.e. the $L_i \ll 4 z_i$ limit of the exact potential of the
segment's *image line* at the segment midpoint. Two consequences
are enforced explicitly since 0.15.0 (they were silent before):

- **Hard error for vanishing image separation.** As $z_i \to 0$ the
  image line merges with the conductor and the self-image term
  diverges — like $1/z$ in the point form, like
  $\ln\bigl(L/(2z)\bigr)$ in the exact form. A conductor lying *in*
  the surface plane has no image separation at all, and its correct
  model is a *coincident* source and image (twice the free-space
  line self-potential), which is a different kernel from the one
  assembled here. Any segment whose **midpoint** has $z \le 0$ or
  $2|z| \le a$ therefore raises `ValueError` — the same criterion the
  inductive path uses in `build_inductance_matrix`. Up to 0.14.1 the
  term was instead *clamped* at `_MIN_DISTANCE = 1 mm`, which returned
  a mesh-dependent grounding resistance up to ~15x too high (a 10 m
  tape at $z = 0$: 810 Ω at $\Delta s = 1$ m, 413 Ω at 0.5 m,
  98 Ω at 0.1 m — never converging, against a physical ~23 Ω), and
  an electrode at negative $z$ (in air) was silently mirrored into
  the soil.

  Both midpoint conditions are checked inside the *shared*
  reaction-matrix kernel `_self_corrected_kernel`, so `mom`, `bem`,
  `cim`, `mom_sommerfeld` and `mutual` reject them identically. In
  0.15.0 development the $z \le 0$ half still sat in `solve_image` /
  `solve_image_2layer` alone, so `compare_engines` on an airborne ring
  ($z = -100$ m) raised for the image family and returned 5.5162 Ω
  for the other four.

  **Who is named in the message.** `solve_image` /
  `solve_image_2layer` pass owner labels down, so the message reads
  `electrode 'g1'` — or `conductor 'pen'` for a galvanically coupled
  distributed conductor, whose leakage segments carry the internal
  pseudo-node name `__cond_pen__seg_0` that the user never wrote. The
  sibling backends call the kernel positionally and get the segment
  index plus its `(x, y, z)` coordinates instead; the subject of the
  sentence degrades to "the model" there.
- **Hard error for a segment that straddles the surface.** A segment
  whose midpoint is buried but whose upper end
  $z_{\text{top}} = z_i - \tfrac{1}{2} L_i |e_{z,i}|$ is negative has
  part of its conductor in the air. Checking midpoints only let
  `RodElectrode(position=(0, 0, -0.5), length=1.2)` solve to
  106.2444 Ω with nothing but a `ShallowSegmentWarning`. This check
  needs the segment tangent (`_Segment.direction`), which the shared
  kernel does not receive, so it runs in `solve_image` /
  `solve_image_2layer` only. A segment whose upper end is *exactly* at
  $z = 0$ stays legal: that is the ordinary driven rod
  (`position=(x, y, 0)`), whose source-plus-image is the length-$2L$
  line Dwight's closed form is derived from.
- **`ShallowSegmentWarning` for $L_i > 4 z_i$.** The regime is legal
  but the point image then overestimates $K_{ii}^{\text{img}}$ by
  more than 13 % (unboundedly as $z \to 0$), so the impedance is
  biased high and still moves under refinement. The warning
  quantifies the bias of the worst offender with the
  **horizontal-segment** formula
  $\bigl(L/4z\bigr)/\operatorname{arsinh}\bigl(L/4z\bigr) - 1$;
  measured at $z = 0.01$ m, $\Delta s = 0.5$ m: 35.5 Ω against
  Dwight's 21.0 Ω (+69 %).

  That formula is the applicable one for every segment that can still
  reach the warning, because the straddling guard above removes the
  steep ones: $L > 4 z$ together with $L |e_z| \le 2 z$ forces
  $|e_z| < 1/2$, i.e. an inclination below $30°$ to the horizontal.
  The message states the inclination it assumed. Up to 0.15.0
  development the same horizontal figure was quoted for *vertical*
  worst offenders, where the vertical closed form
  $(1/L)\ln\bigl((4z+L)/(4z-L)\bigr)$ has no real value at all
  ($4z < L$ makes the argument negative) and the point form
  *under*-estimates instead of over-estimating — the sign of the
  quoted bias was wrong for exactly the class that triggered it.

  **The warning is not restricted to distributed conductors.** Any
  geometry with segments long compared with the burial depth triggers
  it: a ring of radius 25 m at $z = 0.8$ m discretised at
  $\Delta s = 5$ m produces 4.909 m arc segments and warns
  ("by 26 %"), and so do coarse meshes, counterpoises, PEN conductors
  and surface-near tapes. Measured across the test suite it fires at
  20 test nodes (≈35 emissions) in 9 modules.

  **It is also asymmetric across the engine family.** `mom`, `bem`,
  `cim`, `mom_sommerfeld` and `mutual` inherit the identical biased
  diagonal from the shared kernel but do not emit the warning, so a
  `compare_engines` run warns for `image` and stays silent for the
  others while all of them carry the bias. The check stays in
  `solve_image*` because the shared kernel is re-entered per frequency
  and per excitation inside the current-solving backends, where the
  advisory would repeat several times per solve with a `stacklevel`
  pointing into solver internals rather than at user code.

### Residual bias of the point-image diagonal (not fixed in 0.15.0)

The orientation-aware image-line term is still *not* used on the
diagonal. The bias is real and large: a 10 m tape at $z = 0.01$ m
gives 54.361 / 35.491 / 26.603 / 22.039 / 21.043 Ω for
$\Delta s = 1.0 / 0.5 / 0.25 / 0.1 / 0.05$ m, i.e. **+69 % at
$\Delta s = 0.5$ m** against the converged ≈21 Ω.

Three points a reader should have straight:

1. Replacing $1/(2z)$ by the *horizontal* image-line form
   unconditionally would make the **vertical** case worse, not
   better. At $z = L = 0.5$ m the true vertical value is 1.021651,
   the point form 1.000000, the horizontal `arsinh` form 0.989866 —
   the point form is the closer of the two for a rod.
2. The expression $(2/L)\operatorname{arsinh}\bigl(L/(4z)\bigr)$
   quoted above is the **collocation** value — the image line's
   potential *at the segment midpoint*. The reduction the rest of the
   pipeline uses since 0.15.0 (see "Node potential" below) is the
   **Galerkin**, surface-averaged one,
   $\bigl(2/L^2\bigr)\bigl[L \operatorname{arsinh}(L/2z)
   - \sqrt{L^2 + 4z^2} + 2z\bigr]$, which is 0.980575 at
   $L = z = 0.5$ m against the collocation 0.989866. A future fix
   should use the Galerkin form for consistency with
   `_weighted_node_potential`.
3. The mechanical blocker previously recorded here — "the kernel is
   shared and receives no segment directions, so this needs a
   signature change for the whole family" — overstates the case.
   `_Segment`, `_discretize_electrode` and `_self_corrected_kernel`
   all live in `solver/image.py`, and an *optional*
   `seg_directions=None` keyword (point form when `None`) would leave
   every sibling backend bit-exact. What actually defers the fix is
   that switching the image family onto a different diagonal changes
   every stored reference impedance in the benchmark catalogue at
   once; that belongs in its own release step, not in a review-pass
   bug fix.

Until then the guards plus the warning delimit the validity envelope
instead of hiding the bias.

### Cluster constraints

Multiple electrodes connected by a `Conductor` form a *galvanic
cluster* with a shared (unknown) cluster potential $\varphi_c$ and
a known total injected current $I_{c,\text{in}} = \sum_{e\in c} I_{\text{src},e}$. The current sharing within the cluster is solved
through the multi-port grounding matrix $Z_{ij}$ (average potential
at electrode $i$ for unit current at electrode $j$, reduced with the
length-weighted average of the next subsection):

$$
\begin{bmatrix} Z & -C \\ C^{\top} & 0 \end{bmatrix}
\begin{bmatrix} I \\ \varphi_c \end{bmatrix}
= \begin{bmatrix} 0 \\ I_{\text{in}} \end{bmatrix},
$$

with $C$ the cluster-membership indicator. The first $N$ rows
enforce $\varphi_i = \varphi_c$ for every electrode in cluster $c$;
the last $K$ rows enforce $\sum_{i \in c} I_i = I_{c,\text{in}}$.

### Node potential: the length-weighted (Galerkin) average

The "potential of electrode $i$" that enters the row reduction of
$Z_{ij}$ — and that is reported back as
`FieldResult.electrode_potentials` — is the **length-weighted**
average of the segment-midpoint potentials,

$$
\varphi_e \;=\;
\frac{\sum_{k \in e} L_k\, \varphi_k}{\sum_{k \in e} L_k},
$$

which is the pairing dual to the uniform-per-unit-length current
ansatz $I_k = I_e L_k / \sum_j L_j$: it averages over the electrode
*surface*, not over its segment *list*, and it restores exact
discrete reciprocity $Z_{ij} = Z_{ji}$. Consequently
`grounding_impedance(e)` equals the diagonal $Z_{ii}$ of the matrix
the solver actually inverted, to round-off, and
`cluster_impedance(e)` uses the cluster-wide weighted potential
$\varphi_c = \sum_e L_e \varphi_e / \sum_e L_e$ — a physical
quantity, invariant under renaming of the members.

For an electrode with equal segment lengths — every plain rod, ring
or strip — the weighted and unweighted averages coincide, so those
results are unchanged. They differ wherever segment lengths are
mixed: mesh electrodes with $\Delta x/n_x \neq \Delta y/n_y$,
polylines with legs of incommensurable length, and rods split at a
soil-layer interface (ADR-0007). Up to 0.14.1 the reduction inside
the solve was weighted (0.11.0, WP-B3) but the *reported* potential
was a plain mean, so the impedance the user read back was not the
impedance the solver enforced: measured $-10.9\,\%$ on a 4.509 m rod
split at $h_1 = 5$ m (45.686 Ω enforced, 40.701 Ω reported), and
ideally bonded electrodes reported different potentials for one and
the same constrained node.

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
| Burial depth (midpoints) | $z > 0$ and $2 z > a$ for every segment midpoint — hard `ValueError` otherwise, raised in the shared reaction-matrix kernel so `mom` / `bem` / `cim` / `mom_sommerfeld` / `mutual` reject the same geometries |
| Burial depth (extent) | $z_i - \tfrac{1}{2} L_i \lvert e_{z,i}\rvert \ge 0$: no segment may cross the surface. Hard `ValueError`, needs the segment tangent and therefore fires in `image` / `image_2layer` only. An upper end *exactly* at $z = 0$ is legal (ordinary driven rod) |
| Self-image bias | $L_i \le 4 z_i$ for an unbiased self-image term; `ShallowSegmentWarning` otherwise, quantified with the horizontal-segment formula. Emitted by `image` / `image_2layer` only although the bias is shared by the whole family |
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
  evaluations near the wire axis, and off-diagonal reaction-matrix
  pairs that hit the clamp raise a warning (two distinct conductors
  overlap). The *diagonal* image term is no longer clamped: a
  vanishing image separation is a modelling error, not a numerical
  one, and is rejected outright (see "Self-action correction").

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
