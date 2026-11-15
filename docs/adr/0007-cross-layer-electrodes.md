# ADR-0007: Cross-layer electrodes and conductors

| | |
|---|---|
| **Status** | Accepted — step 1 (interface split), layer-aware post-processing and the mom lift implemented 2026-07-09 (audit WP-D; see the amendment at the end); amended 2026.11 (review pass 10: one owner per layer, slivers, reciprocity) |
| **Date** | 2026-05-07 |
| **Deciders** | Project maintainers |
| **Scope** | `groundfield` |

## Context

Every layered-soil backend currently rejects worlds in which any
electrode segment crosses a layer interface:

```
ValueError: image_2layer: a segment lies below the layer interface
(z_max = 2.500 m, h_1 = 1.000 m). The current backend assumes all
electrodes sit in the upper layer.
```

This precondition exists in `image_2layer`, `mom_sommerfeld`,
`cim` and `bem`. The reason is purely numerical: the Tagg/Sunde
image series and the Sommerfeld kernel as currently implemented
both assume **source and observer in the same upper layer**. The
mathematical Green's function for two-layer earth supports
arbitrary layer combinations, the implementation just hasn't been
extended yet.

For typical applications this is a **blocking limitation**.
Realistic geometries that we *must* be able to compute include:

- **Driven rods (Tiefenerder)** of 1.5–3 m length at house
  connections that pass through the typical $h_1 = 0.5\,$m to
  $5\,$m upper layer into the lower one.
- **Foundation electrodes** at depths around 0.8–1.5 m that may
  straddle a thin top layer.
- **MV cable shields** routed at 1.0–1.5 m depth, often below the
  resistive top layer of dry-fill / paving.
- **Cable cabinets and substations** with deep grounding stars.

Without cross-layer support the user has to either truncate the
geometry artificially (wrong physics) or thicken the upper layer
beyond physical reality (wrong soil model). Neither option is
acceptable for typical cases.

## Decision

### Physical model

For a 2-layer earth (upper layer $\rho_1$ thick $h_1$, lower
layer $\rho_2$ semi-infinite), the electric Green's function for
a source at depth $z_s$ observed at depth $z$ has four pair-type
cases that depend on which layer the source and the observer
inhabit:

$$
G(\vec r, \vec r')\;=\;\begin{cases}
G_{uu}(\vec r,\vec r') & 0\le z, z_s\le h_1 \quad\text{(both upper)} \\
G_{ul}(\vec r,\vec r') & 0\le z\le h_1,\; z_s>h_1 \\
G_{lu}(\vec r,\vec r') & z>h_1,\; 0\le z_s\le h_1 \\
G_{ll}(\vec r,\vec r') & z, z_s>h_1 \quad\text{(both lower)}
\end{cases}
$$

With $G_{ul} = G_{lu}$ by reciprocity. Each of the four kernels
has a closed-form Tagg/Sunde image-series representation
*and* a direct Sommerfeld-quadrature representation, both
documented in Sunde 1968 ch. 3 and Tagg 1964. In the limit
$\rho_2 = \rho_1$ all four collapse to the homogeneous kernel
(K=0); in the limit $\rho_2 \to \infty$ they collapse to "PEC at
$z = h_1$" (no current can enter the lower layer).

### Implementation strategy — three phases

This ADR ships **Phase A** in the same release. **Phase B** and
**C** are explicit follow-up work documented here so the API
contract is stable.

#### Phase A — Discretiser split + uniform numerical kernel

1. **Discretiser split.** `_discretize_electrode` and
   `_discretize_conductor` get a new optional argument
   `layer_interfaces=[h_1, h_2, ...]`. When a segment would cross
   a layer boundary at depth $h_k$, it is split at that depth
   into two segments. After this step every segment lies fully
   in exactly one layer; a new `_Segment.layer_index` field
   records which.
2. **Layered Green's-function kernel.** A new
   `coupling.layered_green` module evaluates
   $G_{ij}(\vec r, \vec r';\,\rho_1,\rho_2,h_1)$ for any pair
   $(i, j) \in \{u, l\}^2$ via direct Hankel-transform numerical
   integration of the spectral solution to the 2-layer matching
   problem. The kernel is consumed by all four layered backends.
3. **Lift the preconditions.** `z_max < h_1` is replaced by a
   discretiser-split call. Every segment-pair entry in the
   reaction matrix uses the appropriate kernel
   $G_{uu}, G_{ul}, G_{ll}$ depending on the layer indices.
4. **Image-series fast path** (Tagg/Sunde) is preserved for the
   common $G_{uu}$ case; the slower spectral-quadrature path is
   used for $G_{ul}$, $G_{lu}$, $G_{ll}$.

#### Phase B — Closed-form image series for ll and ul (planned)

The Tagg/Sunde image series for $G_{ll}$ uses the lower-to-upper
reflection coefficient $-K$ and an image structure analogous to
the $uu$ series; the $ul$ case carries a transmission factor
$(1 + K)$ and progressively deeper "transmitted-then-reflected"
images. Both are derivable in closed form (Sunde 1968, eqs.
3.32–3.34) and will replace the slower spectral-quadrature path
in `_layered_green_kernel` for typical worlds with many cross-layer
segments.

**Speedup target**: Phase A's geometric Sommerfeld quadrature is
$\mathcal{O}(N_\lambda)$ per pair (≈ 128 spectral evaluations
per pair × 9 line-line samples = 1 152 evaluations per diagonal
entry, 128 per off-diagonal). Phase B's truncated image series
truncates at $|K|^n < \text{tol}$, typically 10–30 terms, with
each term being a cheap $1/r$ — speedup factor ≈ **30–100x**.

**API contract**: no public API change. The dispatcher in
`_two_layer_self_kernel_factory(allow_cross_layer=True)` will
automatically pick the closed-form path when both
$z_i, z_j \le h_1$ + Tagg/Sunde-uu (already implemented), and
the new closed-form $ll$ and $ul$ helpers when one or both
segments are in the lower layer. The Sommerfeld quadrature stays
as a sanity check.

**Validation programme for Phase B** (when implemented):

1. ll vs Phase-A Sommerfeld at $f = 0$ Hz, ρ_2/ρ_1 = 10, h_1 = 1 m,
   z = z_s = 2 m, s = 0.1 m → ≤ 0.5 % rel difference.
2. ul vs Phase-A Sommerfeld in the symmetric configuration
   (same parameters, different z, z_s) → reciprocity check
   ≤ 1e-9 abs.
3. ρ_1 = ρ_2 limit: every term beyond $K^0$ vanishes, the series
   reduces to the homogeneous free-surface image — bit-exact.
4. End-to-end: 1 km PEN on a 2-layer world with rod tip 1 m below
   h_1 — Phase B vs Phase A on the cluster impedance ≤ 1 % rel,
   wall-clock speedup ≥ 30x.

**Status**: deferred to a focused follow-up where the
derivation can be cross-checked against Sunde 1968 and Tagg 1964
worked examples. For production-grade single-frequency runs at $\lesssim
500$ segments, the Phase A path is acceptable (a few seconds per
solve); Phase B becomes essential for the **parameter-sweep
phase** of large parameter studies (many EFH counts × multiple soil
models × multiple frequencies).

#### Phase C — `n \ge 3` layers (deferred)

The `MultiLayerSoil` case generalises the 2-layer image series
to a recursive Tagg-Sunde stack with reflection coefficients
$\Gamma_k(\lambda)$ at each interface. This is consistent with
the existing `image_nlayer` dispatcher and the Sommerfeld
formulation in `solver/cim.py` / `solver/bem.py`. Phase C is a
straightforward extension once Phase A is stable.

### API impact

- `Engine` schema is unchanged.
- `World` / `Conductor` / `Electrode` are unchanged.
- `_Segment` gains a `layer_index: int` field.
- `_discretize_electrode`, `_discretize_conductor` accept
  `layer_interfaces` and split there.
- All four layered backends drop the `if z_max >= h_1: raise`
  check and dispatch the kernel by `(layer_i, layer_j)`.
- The user-visible behaviour is **strictly additive**: every
  geometry that worked before continues to produce the same
  result; geometries that were rejected before now produce a
  meaningful FieldResult.

## Validation

`tests/test_cross_layer.py`:

1. **Homogeneous limit** ($\rho_1 = \rho_2$): for every backend
   and every layer-pair case, the cross-layer kernel reproduces
   the existing homogeneous kernel to within $10^{-9}$.
2. **PEC limit** ($\rho_2 \to \infty$): for source in upper and
   $\rho_2 / \rho_1 = 10^6$, the spreading resistance approaches
   the *upper-layer-only* hemispherical asymptote.
3. **Sink limit** ($\rho_2 \to 0$): for source in upper and
   $\rho_2 / \rho_1 = 10^{-6}$, spreading resistance drops by
   the textbook factor.
4. **Discretiser split**: a 3-m rod at $z = 0.5$ over a 2-layer
   soil with $h_1 = 1.5$ m is split into 2 segments inside layer
   1 and 1 segment inside layer 2; segment count and per-segment
   layer indices are checked explicitly.
5. **Cross-engine consistency** at 50 Hz: image_2layer,
   mom_sommerfeld, cim, bem agree on the cluster impedance of a
   rod-through-interface within 5 %.
6. **driven rod**: 3-m rod, $h_1 = 1$ m, $\rho_1 = 300$,
   $\rho_2 = 50\,\Omega\,\mathrm{m}$ — sanity check on the
   cluster impedance against the Dwight 1936 formula adapted
   for 2-layer (Dwight + correction factor).
7. **Notebook 16 layered sweep** runs through every $h_1 \in
   \{0.5, 1.0, 2.0, 5.0, 10.0\}$ without error.
8. **DC reproducibility regression**: every old test that uses
   `z_max < h_1` continues to pass bit-exact.

## Consequences

### Positive

- **realistic geometries run.** Driven rods, foundation
  electrodes, deep meshes — all supported on every layered
  backend.
- The Sommerfeld kernel in `mom_sommerfeld` becomes more
  general; this also lifts ADR-0006's Pillar B layered
  Sommerfeld coupling for cross-layer wires.
- Layered earth no longer needs a `UserWarning` anywhere in the
  stack. Carson series remains as the asymptotic option but is
  honest about its limitation: per-meter formula × length, top
  layer only.

### Negative

- Phase A's spectral-quadrature path for cross-layer pairs is
  slower than the closed-form image series. For typical worlds with
  ≲ 1000 segments and a single-frequency or short-frequency
  sweep this is acceptable; for very large worlds Phase B is
  the optimisation lever.
- Discretiser splits add segments at layer interfaces; the total
  segment count grows by ~$O(n_\text{cross})$ where
  $n_\text{cross}$ is the number of conductors crossing
  interfaces. For typical this is small.

### Neutral

- Default behaviour unchanged for all geometries that previously
  worked.
- The new `_Segment.layer_index` field is exposed but not
  documented as a public API — internal solver detail.

## References

- **Sunde, E. D.** (1968). *Earth Conduction Effects in
  Transmission Systems*, Dover. Chapter 3, §3.5–3.7.
- **Tagg, G. F.** (1964). *Earth Resistances*, George Newnes.
  Two-layer image series for source/observer in either layer.
- **Wait, J. R.** (1972). *Electromagnetic Waves in Stratified
  Media*, Pergamon. General reflection-coefficient algebra.
- **Tleis, N. D.** (2008). *Power Systems Modelling and Fault
  Analysis*, Newnes. Modern transmission-line interpretation of
  the layered Green's function.


---

## Amendment 2026-07-09 — WP-D (audit 2026-07-08)

Three gaps between this ADR and the implementation were closed:

1. **Step 1 (interface-split discretiser) is now implemented.** The
   original phase plan promised segments that never straddle the
   interface; the discretiser had ignored `layer_interfaces` and
   assigned straddling segments by midpoint (mis-assignment up to
   `ds/2`). Rods and sloped strips now split at `h_1` in every
   layered backend; a margin warning fires when the deepest segment
   ends within `2·segment_length` above `h_1`.
2. **Post-solve field evaluation dispatches by layer.**
   `FieldResult.potential` and the `solve_mutual_*` probe kernels
   used the upper-layer image series for all points (measured +7 %
   at z = 7 m, +25 % at z = 12 m for K = +0.818); layer-2 pairs now
   use the spectral kernel (`two_layer_probe_matrix`).
3. **`mom` no longer rejects cross-layer worlds** — the action item
   "lift the remaining backends" is done for the n = 2 family
   (mom/cim/bem and the mom_sommerfeld diagonal all dispatch the
   cross-layer kernel; n ≥ 3 cross-layer raises everywhere).

Performance note: the cross-layer reaction-matrix correction is now
grouped by depth pairs (one spectral BVP solve per (z_i, z_j) group
instead of per matrix entry) and cached across the kernel calls of
one solve — a 3-rod cross-layer world dropped from 33.7 s to 4.6 s
on the audit sandbox with identical results.

---

## Amendment 2026.11 — one owner per layer, slivers, reciprocity, line kernel (review pass 10)

Findings 10.15 (F02), 10.25, 10.26, 10.28, 10.32 and 10.50 (F43).

1. **Current per layer.** The image family carries one uniform current per
   owner. An electrode that crosses an interface is now one owner **per
   layer it reaches**, all in its cluster — the model of two rods bonded
   at $h_1$ (10.15). With one owner across the interface a 12 m rod
   through $h_1 = 5$ m in 1000 / 30 Ω·m soil was 566 % above the
   per-segment `mom` solution; it is now within the ansatz error of
   `mom`. `solve_mutual_matrix` / `solve_mutual_field` follow the same
   split, and the diagonal of the mutual matrix is the length-weighted
   cluster impedance over all owners (10.26).
2. **Slivers.** The point kernel takes the Galerkin self term below
   $L/a = 2.99$ (10.32). Every interface cut is kept, however short the
   piece beyond it. (A first version dropped cuts that would leave a
   piece shorter than the wire diameter, `SLIVER_RADII = 2`; that
   ignored the contact with the other layer — a 4.5 m rod in
   1000 / 30 Ω·m jumped from 195.0 to 128.6 Ω between 15.9 and 16.1 mm of
   tip penetration — and was replaced before the release, second
   adversarial audit A2S-01. The Galerkin self term and the line
   kernel's parallel-pair integral are evaluated without cancellation
   for such pieces.) The diagonal evaluates the closed-form $1/R$ terms
   of the layered correction at the wire radius instead of their
   default clamp $h_1/200$, which took 68–84 % of the correction of every
   piece within $h_1/400$ of the interface (a ring at $h_1$ = 5 m read
   −5.29 Ω; third adversarial audit A3S-01); off the diagonal they are
   regularised at the 1 mm of the point forms. What remains at a rod end
   is a limit of the thin-wire model, which has no end face: +21 % with
   the tip exactly at $h_1$ against a solid-rod reference, −4.6 % 1 mm
   past it; such ends warn (`InterfaceContactWarning`, A3S-05).
3. **Reciprocity and the line kernel.** With the exact line kernel
   (ADR-0015) the direct term of a pair in different layers carries the
   transmitted resistivity $2\rho_1\rho_2/(\rho_1+\rho_2)$ and the
   near-singular interface image of the layered correction is a Galerkin
   integral for near pairs; the reaction matrix is reciprocal to round-off
   and the impedance of a rod through the interface changes by less than
   $10^{-4}$ per halving of the segment length.
4. **Point form on both sides.** The point-form guard of a horizontal
   wire within 0.387 (error) or 0.6 (warning) segment lengths of $h_1$
   applies above the interface too, where the series takes the image of
   the segment in the interface as a point source: a ring 1 µm above
   $h_1$ read −1931 Ω with the point kernel and the `mom` family (A3S-02).
5. **Series bookkeeping.** Cross-layer solves report `series_used = False`
   and no false `SeriesTruncationWarning` (10.50); the post-solve field
   evaluation uses the series controls of the solve (10.25).
6. **`fem`** raises for an electrode through an interface its equivalent
   hemisphere does not reach (ADR-0002 amendment 2026.11).

## Amendment 2026.11 (wave 3) — line form, memory, speed

Finding 10.28 (the line kernel on the cross-layer path), 10.56 (memory),
the roadmap row *Vectorise the ADR-0007 cross-layer Sommerfeld path*.

1. **Line form of the layered correction.** With the line kernel the
   layered correction was evaluated between segment midpoints, except the
   interface image of near pairs in one layer. In two-layer soil it is an
   exact superposition of images in every layer combination (the
   generations of `layered_green._image_generation`): with
   $K = (\rho_2 - \rho_1)/(\rho_2 + \rho_1)$,
   $c = 2\rho_1\rho_2/(\rho_1 + \rho_2)$, $q = 2h_1$ and relative to the
   baseline of the source layer, $\rho_1 K^m$ at the four Tagg/Sunde images
   for two points in layer 1; $-\rho_2 K$ at $q - z_s$, $-\rho_2 K^2$ at
   $-z_s$ and $\rho_2(1 - K^2)K^m$ at $-z_s - mq$ in layer 2; $cK^m$ at
   $z_s \pm mq$ and $-z_s - mq$ ($m \ge 1$) across the interface. The
   spectral values stay the point form; each image adds its line-kernel
   correction as on the series path (far quadrupole term, blended Galerkin
   integrals of near pairs, Galerkin self images). A rod through the
   interface no longer depends on the segment length (within 9e-7 from
   2 m to 0.125 m, before 1.3e-4 … 2.4e-4) and the path reproduces the
   series for wires in the upper layer to 1.7e-7 (before 6.2e-4). The
   field of sources in the lower layer (`FieldResult.potential`,
   `solve_mutual_field`) stays in point form.
2. **Memory.** A depth group larger than one block is planned once from
   the range of its distances and evaluated in row chunks (bit-identical),
   and beyond the matrix budget the path works in row blocks like the
   series path: the AP1 station at 40 % penetration ($N$ = 2711) needs
   134 instead of 278 MiB (the within-layer path 126 MiB).
3. **Speed.** The path was grouped by depth since the 2026-07-08 audit;
   what remained were one LAPACK solve of a 3 × 3 system per quadrature
   node, twice per group — half the time of a cross-layer solve, and
   batching them into one call gains nothing. The system is eliminated in
   closed form, vectorised over the nodes: with the point kernel, which
   sees only that change, the AP1 station solves 1.6 … 2.0 times faster
   (5 … 40 % penetration); results move by round-off (the elimination is at
   least as accurate as LAPACK against a 50-digit reference). With the line
   kernel the far terms of point 1 cost about as much again at 40 %
   (performance page).
