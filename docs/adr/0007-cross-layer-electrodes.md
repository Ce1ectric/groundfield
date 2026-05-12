# ADR-0007: Cross-layer electrodes and conductors

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-05-07 |
| **Deciders** | Christian Ehlert |
| **Scope** | `groundfield`, work package 1 of the dissertation |

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

For the AP1 dissertation work this is a **blocking limitation**.
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
acceptable for AP1.

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
in `_layered_green_kernel` for AP1 worlds with many cross-layer
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
worked examples. For AP1-grade single-frequency runs at $\lesssim
500$ segments, the Phase A path is acceptable (a few seconds per
solve); Phase B becomes essential for the **parameter-sweep
phase** of AP1 (5/10/30/80/200 EFH × multiple soil models ×
multiple frequencies).

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
6. **AP1 driven rod**: 3-m rod, $h_1 = 1$ m, $\rho_1 = 300$,
   $\rho_2 = 50\,\Omega\,\mathrm{m}$ — sanity check on the
   cluster impedance against the Dwight 1936 formula adapted
   for 2-layer (Dwight + correction factor).
7. **Notebook 16 layered sweep** runs through every $h_1 \in
   \{0.5, 1.0, 2.0, 5.0, 10.0\}$ without error.
8. **DC reproducibility regression**: every old test that uses
   `z_max < h_1` continues to pass bit-exact.

## Consequences

### Positive

- **AP1-realistic geometries run.** Driven rods, foundation
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
  slower than the closed-form image series. For AP1 worlds with
  ≲ 1000 segments and a single-frequency or short-frequency
  sweep this is acceptable; for very large worlds Phase B is
  the optimisation lever.
- Discretiser splits add segments at layer interfaces; the total
  segment count grows by ~$O(n_\text{cross})$ where
  $n_\text{cross}$ is the number of conductors crossing
  interfaces. For AP1 this is small.

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
