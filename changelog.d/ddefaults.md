### Added

- **Per-engine resource limit `Engine.max_segments` (10.06, user decision
  Q17):** the hard problem-size limit of the segment backends — the number
  of leakage segments and the number of unknowns of the dense nodal system —
  is set per engine: `Engine(max_segments=...)` (also a `create_engine`
  parameter) raises or lowers it for one engine, a positive integer;
  `None` (default) takes the module constant
  `groundfield.diagnostics.HARD_LIMIT` (20 000), read at solve time, so the
  constants stay the defaults. Every segment backend (the image family,
  `mom`, `mom_sommerfeld`), `solve_mutual_matrix` and `solve_mutual_field`
  refuse a larger world with `ResourceLimitError` before anything of that
  size is allocated; `check_segment_budget`, `expected_segments` (new key
  `"limit"`), `check_segment_resolution` and `diagnose`
  (`report.context["max_segments"]`) check against the same limit, and the
  messages name the setting to change. The limit is recorded in the new core
  metadata key `FieldResult.metadata["max_segments"]` (the engine block of
  the provenance keeps the request itself). `SOFT_LIMIT` (5 000, the size
  above which the solve reports that it may take minutes) stays a module
  constant. Tests: `tests/test_resource_limit.py`.

### Changed

- ⚠ **`solve_mutual_matrix` uses the galerkin off-diagonal for every call
  (10.34, user decision Q15):** `offdiag="galerkin"` (the reaction of the
  leakage distributions: reciprocal, and the matrix with which
  (Y_L + Z_G⁻¹)u = i reproduces the solve of the clusters joined by branches)
  is the default whatever the call; `offdiag="probe"` (the potential at one
  probe point per cluster, the definition of releases up to 0.15.0) stays
  available. A call that passes `probe_points` without `offdiag="probe"` —
  every 0.15.0 call, the AP1 catalogue pipeline among them — returns the
  galerkin matrix; the points are ignored, not validated, with the new
  `groundfield.solver.mutual.ProbePointsIgnoredWarning`, which names
  `offdiag="probe"` as the way back. This supersedes the selection rule
  added earlier in this release cycle (`offdiag=None` choosing `"probe"` when
  probe points were given; `None` is no longer accepted). The diagonal does
  not move. On an AP1-like site (station ring and three rods, twelve polygon
  foundations, ds = 2 m, five soils) the symmetrised probe matrix missed the
  input impedance of the joined network by −0.15 … −0.88 % and single
  off-diagonal entries by up to 41 % (median 0.15 … 1.1 %); the galerkin
  matrix reproduces it to 7e-13. On the AP1 catalogue path (group-2 site,
  7 corner soils × 4 penetrations) |Z_ons| at 50 Hz moves by −0.05 … +0.29 %
  at penetrations up to 20 % and by +0.25 … +2.67 % at 40 % (largest at
  ρ₁ = 1000 Ω·m), the ONS current share at 50 Hz by up to +2.7 %. Who is
  affected: every `Z_G` built with probe points, the AP1 catalogue. Golden
  cases: `mutual_foundations_lumped_concrete`,
  `mutual_foundations_distributed_concrete`, `mutual_three_rods_crosslayer`,
  `line_mutual_two_layer` (Z_G norm-wise 7.7e-4 … 8.8e-4, single off-diagonal
  entries 0.48 … 5.9 %). Tests:
  `tests/test_mutual_matrix.py::test_galerkin_is_the_default_whatever_the_call`,
  `::test_ignored_probe_points_are_not_validated`,
  `::test_offdiag_selection_and_validation`.
- ⚠ **Tabulated Sommerfeld assembly by default (10.60, user decision Q16):**
  `Engine.sommerfeld_assembly` defaults to `"tabulated"` (one table of the
  spectral integral per pair of depth groups and frequency); `"exact"` (one
  quadrature per segment pair, the default up to 0.15.0) stays available.
  The image family, `mom`, `mom_sommerfeld` and `solve_mutual_*` follow the
  setting alike; engine JSON without the field (0.15.0) gets the new
  default. Blocks of fewer than 64 sub-segment pairs keep the per-pair rule,
  so small inductive worlds do not move. On a TN network with an inductive
  PEN (`TnNetworkGenerator`, homogeneous 100 Ω·m, ds = 1 m, four
  frequencies 50 … 1000 Hz) the input impedance of the tabulated solve lies
  within 2e-10 of two independent refined references (per-pair rule with 200
  instead of 24 interpolation nodes; tables of 400 instead of 100 nodes; both
  with a refined λ grid), which agree with each other to 2e-10, while the
  per-pair rule is 1.2e-8 off (8 houses); CPU time 1.0 s instead of 16.6 s
  (8 houses, 78 inductive branches) and 12.2 s instead of 306 s (30 houses,
  338 inductive branches; input impedance equal to 1.1e-8, the tabulated one
  1.8e-11 from the 400-node tables). The correction matrices of small
  conductor sets in twelve critical geometries (buried, overhead, 1 cm below
  the surface, in the surface plane, crossing the surface or a layer
  interface, two- and three-layer earth, 250 m segments, 10 and 1000 Ω·m;
  50 Hz and 1 kHz) are finite, symmetric and free of warnings; against a
  reference refined in S(ρ) and in the λ grid the tabulated rule is closer
  than the per-pair rule in ten of them (1.5e-7 … 1.1e-5 against
  4.2e-6 … 7.6e-5 of the largest entry) and equal in two (the surface plane,
  where both use the per-pair rule, and a riser through h₁, 4.3e-6). Who is
  affected: inductive conductors under `earth_inductive_model="sommerfeld"`
  (the default) with blocks of at least 64 sub-segment pairs; numbers move
  by about 1e-8 … 2.4e-6. Golden case: `cond_lumped_long_pen_sommerfeld`
  (≤ 2.4e-6; against the refined per-pair reference the tabulated result is
  4.2e-7 off, the per-pair rule 2.0e-6). Tests:
  `tests/test_sommerfeld_assembly_default.py`.
