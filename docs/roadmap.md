---
title: Roadmap
description: Planned and possible work on groundfield and the items left open after 2026.11.0.
---

# Roadmap

This page collects work that is planned or possible. The first section lists
what is scheduled for the release after 2026.11.0; everything below it is not
scheduled. Up to 0.15.0 the roadmap was the last section of the changelog; it
moved here in 2026.11.0, when the release notes were condensed (see the
[changelog](changelog.md)). Ideas that a review pass recorded as *roadmap
candidates* stay on the page of that pass in the [audit history](audit/index.md).

## Scheduled for the next release

Nothing is scheduled yet. The items decided on 2026-11-08 for the release
after 2026.11.0 (the questions and answers are listed on the
[pass-10 page](audit/pass-10.md#decisions-on-the-questions-left-open-2026-11-08))
were moved into 2026.11.0 itself on 2026-11-14 and are done.

## Left open by 2026.11.0

The work of release 2026.11.0 left these items open on purpose. Where an item
limits a result, the reference documentation states the limit.

- **A current that varies along each wire in the image family.** Since user
  decision Q30 every straight wire of an electrode carries one uniform
  current per unit length. The arms of a star carry equal currents by
  symmetry, so the split left a star alone as it was: +1.8 % (3 arms),
  +3.4 % (4), +10.4 % (8 arms of 10 m) above one current per segment in
  100 Ω·m, +12.3 % in 1000 over 100 Ω·m — the current crowds to the tips.
  Grids are within +0.36 … +0.70 %. A current varying along each wire
  (linear or per sub-piece, as `mom` solves it) would close the gap; until
  then `backend="mom"` gives the per-segment solution.
- **Group rule for wide groups.** A group of about 180° is served by one main to its
  far-end building; a street-row rule or a minimum of two groups for buildings
  around the substation would serve it better (a decision).
- **One cabinet around the substation.** Ten houses in a row through the substation get
  their only cabinet at one end; the other half is fed from the substation.
- **A cabinet that cannot keep its clearance** raises; moving it on along the ray past
  the blocking site would always place it (dense random layouts only so far).
- **A continuous cabinet rule** (user decision D7: not in 2026.11). The 20 m threshold
  between the centroid ray and the far-end ray is a step: a group that crosses it moves its
  cabinet by up to the width of the group (141.8 m in the 50-house grid for a 0.2 m change
  of the house spacing, X at 1 kHz about −20 %). A rule that moves the cabinet continuously
  with the group's geometry would remove the step; the plan's `cabinet_rule` /
  `centroid_ray_gap_m` show meanwhile where a study crosses it.
- **Line form of the field of lower-layer sources.** `FieldResult.potential`
  and `solve_mutual_field` evaluate sources in the lower layer with the
  spectral point kernel; the reaction matrix uses the line form since
  2026.11 (wave 3).
- **A faster far-image kernel.** The compiled far-image term is not
  vectorised by numba on arm64 (nor is the series kernel); a hand-vectorised
  or fused point + quadrupole kernel would cut the +60 … +90 % of two-layer
  line solves.
- **Cost of `carson_series` on large networks.** The Sommerfeld fallback of
  the oblique pairs takes 72 % of a 100-house sweep and the Python loop of
  the Carson pair plan 15 % of a 40-house sweep. Compiled Carson integrals
  were measured and not built (5 … 13 % of an inductive sweep after
  memoisation, below the 20 % threshold; 10.63).
- **Spacing of service drops in a trench.** The drops of one tap and
  direction lie side by side at the touching distance (user decision D9); a
  spacing parameter of the generator would cover wider trenches (5 cm moves
  the 5-house radial network's X at 1 kHz by −12.5 / −14.6 %,
  `carson_series` / `sommerfeld`).
- **Galerkin near image pairs of long pieces.** A long piece below a thin top
  layer is within 7.5 segment lengths of the image of every short segment
  (1.15 M pairs, 5 times the solve before wave 3); integrating them with the
  shorter segment as observer and mirroring the result (the corrections are
  symmetric) would remove it.
- **Far term per image family** (audit W3A-01, proposal 2). Thin top layers
  and high contrasts put hundreds of images within reach (3.6 … 4.5 times the
  solve before wave 3 with the default far-image form).
- **The soil-surface image at the transverse distance.** Image pairs use the
  transverse distance since audit W3A-02; the soil-surface image keeps the
  reduced kernel so that homogeneous worlds stay as they are in 2026.11
  (applied to it, homogeneous house grids would move by +3.9e-7 at 0.8 m,
  +8.8e-7 at 0.5 m and +1.7e-5 at 0.1 m).
- **A finer 3-D `fem` for groups.** The 3-D model refuses the house mesh at
  its defaults (`max_nodes_3d`) and needs about 3 GB per million nodes; a
  third level with the observed order of convergence (instead of the assumed
  second order) would tighten its 2e-3, and per-feature grading or a
  matrix-free multigrid would reach foundations and groups at the defaults.
- **Ties of the star topology under placement jitter.** The cabinet
  placement and the contact rule of the star topology have ties that a
  jitter of 1e-7 … 1e-6 m flips (the 10-house star: −0.24 % and −11.8 % X at
  1 kHz); making them robust, or documenting the realisations as discrete,
  would serve Monte Carlo studies.

## Status of the roadmap items carried in the changelog

| Item (verbatim list below) | Status after 2026.11.0 |
|---|---|
| Vectorise the ADR-0007 cross-layer Sommerfeld path | done in 2026.11.0: the spectral amplitudes in closed form (1.6 … 2.0 × at the AP1 station, round-off); the path was grouped by depth since 2026-07-08, the remaining cost is the `J0` contraction |
| Exact buried two-layer inductive kernel for `n ≥ 3` layers | done in 2026.11.0: the Sommerfeld kernel uses generalised reflection coefficients for any number of layers and any layer (10.35) |
| FEM support for distributed conductors | done in 2026.11.0: isolated chains in both `fem` models, galvanic ones in the 3-D model (every leaking piece a port) |
| Full-3-D FEM backend | done in 2026.11.0: the 3-D model of `fem` (scikit-fem + pyamg as the optional `fem3d` extra; no separate backend; user decisions Q37, D6) |
| Scaling the integral solvers to ~10 000 segments | partly: the image-family kernels need memory linear in the segment count, the cross-layer path included (10.56); the dense LU solve remains |
| `ParameterSweep` API | partly: `sweep(..., n_jobs=...)` runs the sweep in a process pool (10.64) |
| TN-Ortsnetz topology generator | implemented in 0.7.0; service drops keep the shortest connection (user decision Q42), cabinets per building group (user decisions Q22, Q35) |
| Penetration depth for multilayer soil | done in 2026.11.0: `penetration_depth` (complex depth and equivalent resistivity of any layered soil) |
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
