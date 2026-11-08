---
title: Roadmap
description: Planned and possible work on groundfield and the items left open after 2026.11.0.
---

# Roadmap

This page collects work that is planned or possible but not scheduled for a
release. Up to 0.15.0 the roadmap was the last section of the changelog; it
moved here in 2026.11.0, when the release notes were condensed (see the
[changelog](changelog.md)). Ideas that a review pass recorded as *roadmap
candidates* stay on the page of that pass in the [audit history](audit/index.md).

## Left open by 2026.11.0

The work of release 2026.11.0 left these items open on purpose. Where an item
limits a result, the reference documentation states the limit.

- **Far images of the two-layer image series.** With the exact line kernel the
  distant Tagg/Sunde images are still point sources without a quadrupole term,
  so a straight wire in two-layer soil is not reproduced exactly: a 10 m strip
  at 0.5 m depth in 1000 over 50 Ω·m with h₁ = 1 m deviates from its exact
  uniform-current resistance by at most +0.051 % for segment lengths from
  0.125 m to 2 m, not monotonically in the segment length (point kernel −0.6 %
  to −1.9 %). A quadrupole term for every image would remove the residual at
  roughly twice the cost of the series assembly (10.28).
- **One current per wire line for grids and stars in the image family.**
  The image family solves one uniform current per electrode; a
  `grid_mesh` or star electrode is therefore 3.9 % (20 m × 20 m, 4 × 4
  meshes), 5.3 % (40 m × 40 m, 8 × 8) and 4.3 % (4-arm star) above the
  per-segment `mom` solution, and its near-field potentials are off by tens
  of per cent. The generators build mesh foundations as one electrode per
  wire line since 2026.11 (+1.1 %, point kernel +0.3 … +0.5 %); doing the same inside the solver
  for every grid and star would move their results by −3 … −5 % (second
  adversarial audit, A2G-04).
- **Line kernel on the cross-layer path.** For electrodes that cross the layer
  interface (ADR-0007) the homogeneous part of the reaction matrix uses the line
  kernel, the layered correction stays in point form (10.28).
- **Compiled Carson integrals.** A `LowLevelCallable` for Carson's P and Q was
  not built; memoisation gives the bulk of the gain bit-identically (10.63).
- **Tabulated Sommerfeld assembly as the default.** The tabulated assembly of
  the Sommerfeld earth-return correction is opt-in
  (`Engine(sommerfeld_assembly="tabulated")`); the default stays the per-pair
  quadrature (10.60).
- **Voltage sources.** Solving a world with a `VoltageSource` raises
  `NotImplementedError`; drive the world with a 1 A current source and scale by
  U / Z_in (10.51).
- **`fem` beyond the equivalent hemisphere.** `fem` is experimental: it reduces
  every cluster to an equivalent hemisphere, lumps distributed and inductive
  conductors, returns the current of a two-terminal source through remote
  earth and does not support junction nodes (10.23, 10.74). A full 3-D volume
  method remains deferred (see below).
- **Concrete shells in the `mom` family.** `mom` and `mom_sommerfeld` reject
  electrodes with a concrete shell; the shell is modelled by the image family
  and `solve_mutual_*` (10.12–10.14).
- **Overlapping inductive conductors in `diagnose()`.** Conductors that lie
  side by side or overlap (two cables in one trench, a service drop along its
  PEN main) are evaluated as touching wires since 2026.11; `diagnose()` reports
  close and interpenetrating leakage wires but not overlapping insulated
  inductive conductors, so an unintended overlap goes unnoticed.
- **Memory of the cross-layer path.** Worlds with rods that reach the lower
  layer need more memory than the within-layer path; this path was not
  optimised further in 2026.11.0 (10.56).

## Status of the roadmap items carried in the changelog

| Item (verbatim list below) | Status after 2026.11.0 |
|---|---|
| Vectorise the ADR-0007 cross-layer Sommerfeld path | open |
| Exact buried two-layer inductive kernel for `n ≥ 3` layers | done in 2026.11.0: the Sommerfeld kernel uses generalised reflection coefficients for any number of layers and any layer (10.35) |
| FEM support for distributed conductors | open (`fem` is experimental, 10.23) |
| Full-3-D FEM backend | deferred |
| Scaling the integral solvers to ~10 000 segments | partly: the image-family kernels need memory linear in the segment count (10.56); the dense LU solve remains |
| `ParameterSweep` API | partly: `sweep(..., n_jobs=...)` runs the sweep in a process pool (10.64) |
| TN-Ortsnetz topology generator | implemented in 0.7.0; the corner-routed service drop is open |
| Penetration depth for multilayer soil | open |
| Export of `BusType` and `BranchType` into groundinsight databases | `BusType`: `save_bustype_to_db`; `BranchType` open |
| Import of measurement geometries from groundmeas | open |
| Reference-case library for AP1 | open |
| Notebook suite | the tutorials T01–T10 cover the main workflows (2026.11.0) |

## Roadmap as carried in the changelog up to 0.15.0

The list below is the *Roadmap* section of `CHANGELOG.md` as it stood before
2026.11.0, moved verbatim (reST roles rendered as code); its references to the
`[Unreleased]` block are to that time.

<!-- verbatim roadmap: begin -->

Backlog of feature ideas that are not yet scheduled. Updated as
work package 1 progresses.

### Performance

- **Vectorise the ADR-0007 cross-layer Sommerfeld path.** After caching the
  Gauss–Legendre nodes (see `[Unreleased]`), the remaining cost of the
  cross-layer fallback is the genuine $O(N^2)$ per-segment-pair work in
  `coupling/layered_green.py::two_layer_layered_correction_real_space` /
  `_spectral_amplitudes`: many small `np.linalg.solve` calls (one
  spectral-amplitude system per pair) plus per-pair quadrature. Batching the
  per-pair spectral solves into a single vectorised `np.linalg.solve` over the
  pair axis, and evaluating the Bessel/exponential kernels on the shared
  quadrature grid in one array op, should give another ~2–5×. Motivation: AP1
  catalogues two-layer soils with a thin upper layer (`h_1 = 5 m`) where a deep
  station rod can cross the interface — currently the dominant per-station cost
  even after the leggauss cache (~155 s vs ~3 s within-layer at 5 %
  penetration). Cost scales $O(N^2)$ in segment count, so larger /
  higher-penetration geometries benefit most.

### Core functionality

- **Exact buried two-layer inductive kernel → `n ≥ 3` layers.** The 0.13.0
  fix (`coupling/sommerfeld_inductance.py::_two_layer_reflected_weight`)
  solves the exact boundary-value problem for a conductor buried in the
  upper layer of a **two-layer** earth, validated against Tsiamitros (2005)
  Eq. (8) across 50 Hz – 1 MHz. For **`n ≥ 3`** layers a *buried* source
  still falls back to the approximate single-air-image correction (the
  surface reflection cascade `reflection_coefficient_layered` is already
  exact for any `n`); a general `n`-layer buried spectral kernel would
  extend the exact inductive earth-return to arbitrary multilayer soils.
  The `n ≥ 3` buried path is currently pinned only by the homogeneous and
  bounding checks — there is no exact multilayer reference for it yet.
  Galvanic multilayer solves are unaffected (they use `layered_green`).
- **FEM support for distributed conductors.** The current FEM
  backend falls back to a lumped branch and logs a warning when
  it sees a distributed conductor — the equivalent-hemisphere
  reduction does not resolve the midpoint pseudo-electrodes. A
  full-3-D FEM (deferred below) would lift this restriction.
- Full-3-D FEM backend (via `scikit-fem` or comparable) replacing
  the equivalent-hemisphere reduction in the current `fem` engine.
  **Deferred** — explicitly *not* on the AP1 roadmap. The target
  AP1 geometries (~20 houses with foundation electrodes plus a
  transformer station with ring + driven rods plus driven rods at
  cable cabinets) already push the integral solvers towards the
  ~1000-segment regime where the dense $O(N^3)$ LU solve dominates;
  a 3-D volume FEM on the same configuration would be one to two
  orders of magnitude more expensive without adding physical
  insight. This task is therefore reserved for follow-up work where
  a multi-cluster volume computation is genuinely needed.
- Scaling the integral solvers to ~10 000-segment networks: replace
  the dense LU solve in `mom`, `bem` and `cim` by a preconditioned
  iterative solver (GMRES with a block-diagonal or fast-multipole
  preconditioner). Unlocks the full AP1 parameter space (large
  TN-Ortsnetz with foundation electrodes, drives, and cable
  cabinets) without giving up the per-segment current resolution.
  Concrete prerequisites: ACA (adaptive cross approximation) for
  the dense reaction matrix to bring memory from $O(N^2)$ down to
  $O(N \log N)$ and a block-Jacobi preconditioner per cluster.

### Studies and tooling

- **`ParameterSweep` API** — wrap a `world_factory(**params)` plus a
  parameter grid into a parallel sweep (`joblib`-driven, target 12
  cores) that returns a `pandas.DataFrame` of cluster impedances,
  EPRs and per-electrode currents. Drives the AP1 statistical
  studies (soil layering, electrode count and position, Monte Carlo
  realisations).
- **TN-Ortsnetz topology generator** — *implemented 2026-05-18 in
  the `[Unreleased]` block above.* The radial-trunk PEN topology
  (`RadialTrunkTopology`) and the integration with
  `OsmBuildingPlacement` are live; what remains as a v2
  enhancement is the *corner-routed* service drop (tap → axis-
  aligned corner → building), which currently runs as a single
  straight line between anchors. For typical street-aligned plots
  the corner offset is small compared with the full feeder length
  and the residual error in the longitudinal branch impedance
  stays well below 1 % at 50 Hz.

- **penetration depth** Calculate the depth of the earth current as it is used in Carson integrals. It should be possible to create the earth current depth of any soild multilayer problem to use an equivilent for the typical formulas for calculating the self and coupling impedances of a cable or overheadline with earth return part.

### Integration

- Direct export of `BusType` and `BranchType` into `groundinsight`
  databases.
- Import of measurement geometries from `groundmeas` (positions,
  lengths, auxiliary-electrode set-up).

### Documentation and typing

- Reference-case library for AP1: clusters of single-family houses
  (5 / 10 / 30 / 80 / 200), small and medium commercial buildings,
  cable cabinets.
- Notebook suite that covers the full parameter space described in
  the dissertation proposal.

<!-- verbatim roadmap: end -->
