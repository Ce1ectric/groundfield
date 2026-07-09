# ADR-0010: Tier-0 performance optimisations

| | |
|---|---|
| **Status** | Accepted (Tier 0b implemented; Tier 0a/1 implemented 2026-07-09, audit WP-F; 0c deferred) |
| **Date** | 2026-05-09 |
| **Deciders** | Project maintainers |
| **Scope** | `groundfield.coupling.inductance`, `groundfield.solver.image` (LU caching), discretiser heuristics |

## Context

The empirical benchmarks in
the project's internal performance-tuning harness
identified three classes of low-risk optimisation that do not
change the underlying physics but cut wall-clock time
substantially:

* **0a — LU caching across frequencies.** Today the solver
  rebuilds the dense reaction matrix and re-factorises per
  frequency. For the default path
  (`earth_inductive_model="perfect_mirror"`, no
  per-conductor `inductance_model`) the matrix is identical
  across frequencies; the solve currently does $n_\text{freq}$
  redundant LU factorisations.
* **0b — Vectorised Neumann assembly.** The Python double loop
  in :func:`build_inductance_matrix` calls
  :func:`neumann_mutual` for each of the $M(M-1)/2$ off-diagonal
  pairs. The Neumann integrand itself is already vectorised
  (a single `np.einsum`), but the per-pair Python overhead
  dominates: array conversions, branch on parallel-vs-quadrature,
  per-call function dispatch. Empirically this is ~10–30 s for
  $M = 1\,000$.
* **0c — Geometry-adaptive discretisation.** A single global
  `segment_length` over-resolves long uniform PEN runs while
  potentially under-resolving the field near substation rings.
  A heuristic refinement that uses fine segments near electrode
  attachments and coarse segments along homogeneous mid-sections
  could halve $M$ without accuracy loss — at the cost of more
  complex convergence behaviour.

ADR-0009 v3 (the spec layer) and ADR-0008 (the `groundinsight`
bridge) are not affected; this ADR is purely about the inner
numerical loop.

## Decision

Implement Tier 0 in three independent, individually
test-isolatable steps. Each preserves the current public API and
must be **bit-exactly verified** against the pre-optimisation
implementation on the existing test suite.

### 0b (this ADR, implemented now)

Rewrite :func:`build_inductance_matrix` to vectorise the off-
diagonal assembly:

* Compute every segment's unit vector and length once.
* Pair-wise dot products in a single $O(M^2)$ NumPy
  multiplication.
* Mask the parallel pairs and evaluate the closed-form Grover
  expression in a vectorised batch.
* Mask the non-parallel pairs and evaluate the 16×16
  Gauss–Legendre quadrature in a row-at-a-time vectorised batch
  (so peak memory stays at $O(M \cdot 16^2 \cdot 3)$ rather
  than $O(M^2 \cdot 16^2 \cdot 3)$).
* The diagonal stays identical (still
  :func:`perfect_mirror_self_pair_inductance` per row).

The legacy implementation is kept as
:func:`_build_inductance_matrix_loop` for regression testing
and as the reference oracle in :file:`tests/test_inductance_vectorised.py`.

### 0a (follow-up — deferred)

Add an LU cache to the per-frequency solver loop in
:mod:`groundfield.solver.image` (and friends). When the matrix
is frequency-independent (`earth_inductive_model ==
"perfect_mirror"` and no per-conductor `inductance_model`),
factor once and reuse the LU across the whole frequency list.
For the inductive paths, factor per frequency as today.

**Status**: implemented 2026-07-09 (audit 2026-07-08, WP-F), in a
slightly different — and stronger — form than originally scoped. The
2026-07-08 audit profile showed that the dominant cost was not the LU
factorisation but the **multiport-Z assembly**: the self-kernel was
invoked once per electrode column *and* per frequency, rebuilding its
O(N²) geometry tensors every time (97 % of wall time on AP1-sized
worlds). The implemented Tier 1 therefore covers:

* one-shot batched Z assembly — all ``N_a`` excitation columns in a
  single kernel call with an ``(n_segments, N_a)`` matrix;
* a ``multiport_cache`` shared across the per-frequency calls in
  ``image`` / ``image_2layer`` / ``cim`` (Z is
  frequency-independent);
* batched per-frequency potential evaluation (one stacked kernel
  call for the whole frequency set);
* multi-RHS ``solve(A, [b_re, b_im])`` on every real DC path
  (image family, ``_galerkin_solve``, ``fem``).

Measured effect: 50-building TN world (N = 1476) 22.0 s → 0.18 s;
100 buildings (N = 2926) ≈ 2 min → 0.85 s. Results are numerically
identical to FP rounding (verified against the audit reference
values on the galvanic and inductive paths). The originally scoped
LU cache across frequencies remains unimplemented — after Tier 1 the
augmented-system LU is no longer a relevant cost below N ≈ 10⁴.

### 0c (deferred)

Adaptive discretisation requires:

* A heuristic that classifies segments into "near-feature"
  vs. "homogeneous-trunk".
* Calibration against the global-`segment_length` results to
  choose tolerance bands.
* A new convergence test suite — the existing tests assume a
  uniform discretisation.

The benefit is a 1.5–2× reduction in $M$ at equivalent
accuracy, but the work is research-grade and risks introducing
hard-to-diagnose convergence drift. This ADR therefore notes
adaptive discretisation as a **roadmap item**, not a Tier-0
deliverable.

## Validation programme

For 0b (this release):

1. **Bit-exact regression** against the legacy loop on every
   geometric class used by the existing test suite — single
   rod, ring, strip, mesh, multi-electrode network with
   bonding straps, reference networks. Tolerance:
   $10^{-12}$ relative on every matrix entry.
2. **Numerical stability** at extreme aspect ratios — very long
   thin PEN runs, very short bonding straps, near-parallel
   geometries.
3. **Performance** — measured speed-up on a representative typical
   network with $M \in [100, 5\,000]$, plotted in
   the project's internal Tier-0 performance harness.

For 0a (follow-up):

1. Bit-exact regression on every multi-frequency test
   (`tests/test_*` that use `frequencies=[...]`).
2. End-to-end frequency-sweep speed-up on the reference
   network.

## Consequences

* The vectorised assembly is **drop-in** — every caller of
  `build_inductance_matrix` keeps working unchanged.
* Memory: $O(M \cdot 16^2 \cdot 3) = O(M)$ peak per
  computed row, identical $O(M^2)$ for the result. No surprise.
* Maintenance: the vectorised code is denser than the original
  and harder to read. A docstring with a worked example and a
  comment block per processing stage compensates for that.
* The pre-optimisation legacy function stays in the module as
  `_build_inductance_matrix_loop` so future regressions can be
  diagnosed by comparing against it.

## References

- Notebook 21 (empirical benchmarks).
- ADR-0004 (the underlying Neumann inductance model).
- ADR-0005 (the Carson correction; **not** vectorised in this
  ADR — orthogonal scope).
- Profiling output cross-checked against the Stage-1 results in
  the conversation log.
