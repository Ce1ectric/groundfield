# ADR-0003: Distributed conductor model

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-05-04 |
| **Deciders** | Christian Ehlert |
| **Scope** | `groundfield`, work package 1 of the dissertation |

## Context

The previous step in the conductor stack added a **lumped**
finite-impedance branch model: a `Conductor` between two electrodes
collapses to a single $R = \rho_\text{mat} L / A$ branch in the
nodal-analysis system. That model is sufficient when the conductor
is *short* compared to its electrical size — for a few-metre run
between two rod electrodes the lumped approximation matches the
distributed solution to within a fraction of a percent.

For work package 1 (TN distribution network) several questions
**fall outside that regime**:

1. **PEN-strand spanning the village.** A 100 – 200 m PEN section
   between the transformer station and the cable cabinets does not
   leak its current at one end — every house tap, every joint, every
   buried section adds a leakage path. The current distribution
   along the conductor is non-uniform and depends on the local soil
   conditions.
2. **Bare-copper interconnects.** Inside MV substations and at
   industrial sites, bare-copper conductors are deliberately laid
   into the soil and act as longitudinal earth electrodes. They are
   essentially strip electrodes with a finite length — not lumped
   branches.
3. **Cable shields with a continuous earth path.** For shielded MV
   cables the metallic screen is grounded at multiple points; the
   shield itself is a distributed earth conductor whose leakage and
   longitudinal current are coupled through Carson-type effects
   below 1 kHz.
4. **Inductive coupling between conductor sections.** The AP1
   research question on the *coupling between the measurement lead
   and the current injection* requires the geometry of the leads to
   be resolved spatially — Neumann integrals are point-pair
   integrals; lumped branches do not capture them.

The lumped model is therefore insufficient for AP1 in its current
form. ADR-0003 settles **how to extend** the conductor stack so
that it remains compatible with the eight existing backends, the
finite-impedance branches added in the previous step, and the
upcoming inductive-coupling and Carson extensions.

## Decision

`Conductor` becomes a **distributable wire**. Two new fields drive
the model:

| field | type | meaning |
|---|---|---|
| `discretize_segment_length` | `float` (m), default `None` | Maximum segment length used to discretise the conductor. ``None`` keeps the conductor lumped (single-segment, equivalent to today's behaviour). |
| `coupling_to_soil` | `Literal["isolated", "galvanic"]`, default `"isolated"` | Selects whether the conductor exchanges current with the soil along its length. `"isolated"` (cable, PEN inside an insulating jacket) keeps the conductor purely longitudinal. `"galvanic"` (bare copper, exposed shield) lets every segment leak current into the soil through the same Green's-function kernel as the electrode segments. |

The discretiser produces $n$ collinear sub-segments along the
conductor's axis, each carrying:

- a **midpoint-leakage current** $I_\text{leak}^{(k)}$
  (only when `coupling_to_soil == "galvanic"`),
- a **longitudinal current** $I_\text{long}^{(k)}$ flowing along
  the conductor axis from node $K_{k-1}$ to node $K_k$.

The conductor introduces $n+1$ nodes:
$K_0 = $ start-electrode cluster, $K_n = $ end-electrode cluster,
and $n-1$ interior nodes at the segment midpoints. Each
**longitudinal segment** is one branch in the nodal-analysis system,
with series impedance

$$
Z_\text{long}^{(k)} \;=\; R^{(k)} \;+\; j\omega\, L^{(k)}
$$

where $R^{(k)} = \rho_\text{mat}\, \ell_k / A$ and the inductance
$L^{(k)}$ is **deferred to ADR-0004** (inductive coupling). For the
present step we keep $L^{(k)} = 0$, so the longitudinal impedance
remains purely resistive. The system stays real for f < 1 kHz.

### Augmented linear system

Let:

- $N_e$ = number of electrode segments (as today),
- $N_c$ = number of conductor segments (new, only the
  `coupling_to_soil == "galvanic"` ones contribute to the Z-matrix),
- $K$ = number of cluster nodes (electrode clusters merged by ideal
  conductors, plus the interior nodes introduced by distributed
  conductors),
- $M$ = number of longitudinal branches (one per conductor segment).

The combined unknown vector is

$$
\mathbf{x} \;=\;
\begin{pmatrix} \mathbf{I}_\text{leak} \\
                \boldsymbol{\varphi}_n \\
                \mathbf{I}_\text{long} \end{pmatrix}
\in \mathbb{C}^{(N_e + N_c) + K + M}.
$$

The augmented system reads

$$
\begin{bmatrix}
   Z & -C_s & 0 \\
   C_s^{\top} & 0 & B^{\top} \\
   0 & B & -Z_b
\end{bmatrix}
\begin{bmatrix}
   \mathbf{I}_\text{leak} \\
   \boldsymbol{\varphi}_n \\
   \mathbf{I}_\text{long}
\end{bmatrix}
\;=\;
\begin{bmatrix}
   \mathbf{0} \\
   \mathbf{I}_\text{in} \\
   \mathbf{0}
\end{bmatrix},
$$

where

- $Z \in \mathbb{R}^{(N_e + N_c) \times (N_e + N_c)}$ is the
  *enlarged* multi-port grounding matrix, built from the same
  Green's-function kernel as before but evaluated on **all** segments
  (electrode + galvanic conductor segments).
- $C_s \in \{0,1\}^{(N_e+N_c) \times K}$ is the
  segment-to-node incidence matrix. Electrode segments map to the
  cluster of their owning electrode; conductor segments map to their
  midpoint node.
- $B \in \{-1,0,+1\}^{M \times K}$ is the branch-to-node
  incidence matrix (`+1` at branch start, `−1` at branch end). The
  branch endpoints are the conductor's interior midpoint nodes plus
  the start / end cluster nodes.
- $Z_b \in \mathbb{R}^{M \times M}$ is the diagonal matrix of
  longitudinal segment impedances $R^{(k)}$ (purely resistive
  in this ADR; complex from ADR-0004 on).

For `coupling_to_soil == "isolated"` the conductor's segments
contribute to $C_s$ and $B$ but **not** to $Z$ — there are no
leakage rows for these segments. The full system collapses gracefully
to the lumped finite-branch model when `discretize_segment_length is
None` (single segment, midpoint-only leakage).

### Boundary conditions at the conductor ends

The endpoints of a distributed conductor sit on the cluster of the
attached electrodes. Both ideal galvanic shorts and finite branches
to other conductors are handled by the existing cluster + branch
infrastructure — distributed conductors integrate without changing
those rules.

If a conductor's start- or end-electrode is `None` (purely geometric
end), the corresponding endpoint becomes a **floating node**: it
participates in the linear system but has no external source, no
leakage, and no constraint other than KCL. This case will be useful
for future work on partial measurement-lead geometries; for AP1 the
ends are always anchored to electrodes.

### Implementation strategy

1. **Conductor schema.** Add `discretize_segment_length`,
   `coupling_to_soil`. Backwards-compatible: both default to the
   previous lumped behaviour (`None`, `"isolated"`).
2. **Discretiser.** New `_discretize_conductor(c, ds)` in
   `solver/image.py` produces `_Segment` instances tagged with
   `kind="conductor"` plus the conductor's name and the per-segment
   node indices `(K_in, K_out)`.
3. **Cluster / node / branch builders.** Extend
   `_build_clusters`, `_build_finite_branches`, and the new
   `_build_distributed_branches` so that interior conductor nodes
   become regular nodes in the active set.
4. **Solvers.** Each backend that consumes
   `_solve_cluster_currents` / `_galerkin_solve` is updated to:
   - assemble the enlarged $Z$ over electrode + galvanic
     conductor segments,
   - feed the new branch list into the existing nodal-analysis
     block.
5. **Frequency loop.** As long as $L^{(k)} = 0$, the linear
   system is real per frequency and the solution is identical for
   all entries of `engine.frequencies`. The frequency dispatch
   becomes meaningful with ADR-0004.

### Validation

- **Limit `discretize_segment_length is None`** → bit-exact match
  with the lumped finite-branch solution from the previous step
  (regression test).
- **Limit `coupling_to_soil == "isolated"`** with finite
  `discretize_segment_length` → solver behaves like a lumped branch
  with $R = \sum_k R^{(k)}$ (the segment chain is a series
  resistor).
- **Convergence in `n_segments`** → as the conductor is refined the
  earth current and EPR profile converge to a limit. Test
  $\Delta < 1\%$ for $n \ge 8$ on a 30 m PEN strand.
- **Cross-engine consistency** → `image`, `image_2layer`, `mom`,
  `cim`, `bem` agree to ≤ 3 % on a galvanic 30 m bare-copper strip
  electrode modelled three ways: as a `StripElectrode`, as a
  distributed `Conductor` with `coupling_to_soil="galvanic"`, and
  as a fine-grained chain of small electrodes connected by ideal
  conductors. The three formulations should give the same
  cluster-impedance to within the discretisation tolerance.
- **Strip-equivalence** → a `Conductor(galvanic, n=20)` between
  two rods reproduces the cluster impedance of an equivalent
  `StripElectrode` of the same length placed at the same depth, to
  within 5 % (the difference reflects the strip's symmetric
  current feed at both ends vs. the conductor's directed feed).

## Consequences

### Positive

- Sets the geometric stage for ADR-0004 (inductive coupling): once
  the conductor has segments, Neumann integrals between segment
  pairs become a straightforward addition to the longitudinal
  impedance.
- Enables AP1 questions that the lumped model cannot answer:
  PEN-strand voltage drop with intermediate leakage, bare-copper
  conductor as a distributed earth element, partial-shield current
  redistribution.
- One unified discretisation kernel for both electrodes and
  conductors — they share `_discretize_conductor` for line geometries
  with `_discretize_strip` (the strip electrode is now a special
  case).

### Negative

- Larger linear system. For an AP1 world with ~100 rods and
  ~150 m of PEN at $\Delta s = 1\,\mathrm{m}$, the additional
  conductor segments push the segment count from ~1 500
  (electrodes only) towards ~3 000. The dense $O(N^2)$ memory
  and $O(N^3)$ LU solve still fit in the 32 GB workstation budget,
  but the runtime grows by ≈ 8×. This motivates the iterative-solver
  roadmap item.
- Two-layer / n-layer kernels need to be evaluated on the conductor
  midpoints in the upper layer. The existing precondition
  ($z < h_1$) extends to all segments — including conductor
  segments. The user has to make sure that buried conductors stay in
  the upper layer; otherwise a clear `ValueError` is raised.

### Neutral

- The default behaviour stays lumped. Distributed mode is opt-in;
  existing notebooks and tests continue to work.
- The augmented system has the same block structure as the
  lumped-finite-branch system from the previous step. The change is
  *quantitative* (more rows / columns) rather than *qualitative*
  — readers familiar with the previous formulation will recognise
  the same pattern.

## References

- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*. Dover, ch. 7 — wire antenna formulation, distributed
  current and voltage along buried conductors.
- Dawalibi, F. P. (1986). Electromagnetic fields generated by
  overhead and buried short conductors. *IEEE Trans. PWRD* 1(4) —
  the canonical reference for distributed conductor segments with
  a layered Green's function.
- Meliopoulos, A. P. S., & Moharam, M. G. (1983). Transient
  analysis of grounding systems. *IEEE Trans. PAS* 102(2).
- Colominas, I., Navarrina, F., & Casteleiro, M. (1999). A boundary
  element formulation for the substation grounding design.
  *Adv. Eng. Soft.* 30(9–11).
