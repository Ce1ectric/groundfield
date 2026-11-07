### Added

- **`mom_sommerfeld` kernel tabulation report (10.61):**
  `metadata["kernel_tabulation"]` (`enabled`, `groups_tabulated`,
  `groups_pointwise`, `table_evaluations`, `max_check_error`) and the
  keyword `solve_mom_sommerfeld(..., tabulate=True)`; through the engine
  as `options={"mom_sommerfeld": {"tabulate": False}}` once the registry
  entry `MomSommerfeldOptions.tabulate` is in place. `converged` and
  `converged_per_frequency` (10.47) are kept. Test:
  `tests/test_mom_sommerfeld.py::test_solve_reports_the_kernel_tabulation`.
- **Benchmarks B12 and B13 (10.76, R2 §3.3):** B12 checks the depth-group
  table of `mom_sommerfeld` against the pointwise quadrature on a 40 m
  grid in 1000/30/200 Ω·m (bound 1e-6; measured 0 at ds = 2.5 m,
  6.9e-9 for single entries and 1.5e-10 for the grid impedance at
  ds = 1.25 m). B13 compares the per-metre series impedance of an
  isolated line (difference of a 1600 m and an 800 m line at 1 kHz,
  which cancels the end effects) with the Pollaczek conductor–earth loop
  impedance (reactance bound 5e-3, measured 4.1e-6; resistance bound
  2e-2, measured −1.04 %). B7 keeps the strict
  xfail of `image_2layer` until 10.15 and records that `fem` raises.
  `python -m tests.benchmarks.tolerances` prints the table.
- **`compare_engines` names engines that ran the same computation
  (10.76):** a deprecated name and its replacement with identical
  settings (`cim` and `image_2layer`, `bem` and `mom`) get a note that
  their agreement is not an independent cross-check. Test:
  `tests/test_engine_aliases.py::test_compare_engines_notes_that_an_alias_is_no_independent_check`.

### Changed

- **`mom`, `mom_sommerfeld` and `bem` run the shared solve pipeline
  (10.75, 10.59).** Discretisation, injection, topology, branches and
  the solve step are those of `groundfield.solver._pipeline`; each
  backend contributes only its Green's function, and the reaction matrix
  is assembled directly instead of as the product of the kernel with the
  identity. The numbers are unchanged: 41 of 41 stored golden cases and
  47 further results are bit-identical to the wave-2 base. The direct
  assembly makes `mom_sommerfeld` on two-layer soil faster (40 m grid,
  ds = 2.5 m: 15.2 → 0.41 s). New result metadata: `n_segments`,
  `segment_length`, `solver`, `stub`, `earth_inductive_model`.
- ⚠ **`mom` and `mom_sommerfeld` honour `Source.return_to` and junction
  nodes (10.75 with 10.16 / 10.38).** Because they now use the shared
  injection and topology steps, the return terminal of a two-terminal
  source draws the current back through its electrode (0.15.0: ignored,
  the current returned through remote earth), and
  `metadata["junction_potentials"]` reports junction nodes as the image
  family does. Numbers change for every world with a `return_to` source
  solved with these backends; no stored golden case contains one. Needs
  the registry flags `two_terminal_sources` and `junctions` for both
  backends. Tests:
  `tests/test_point_matching_pipeline.py::test_mom_family_honours_return_to`,
  `tests/test_point_matching_pipeline.py::test_mom_family_reports_the_junction_potential`.
- ⚠ **`mom_sommerfeld` tabulates the reflected kernel for n ≥ 3 layers
  (10.61).** Segment pairs of one depth class in the top layer use a
  cubic spline in ln s (64 nodes per decade) instead of one quadrature
  per pair; eight midpoints per table are checked against the quadrature
  (tolerance 1e-7 of |G|, one refinement), and a group whose check fails
  or whose nodes do not converge falls back to the quadrature with the
  same convergence report. Every other pair is evaluated once per
  distinct (s, |Δz|, z + z') instead of once per pair, bit-identically.
  Results move where a table is used, by ≤ 1.1e-9 relative for the
  impedances measured and ≤ 6.9e-9 for single matrix entries (40 m grid
  at ds = 1.25 m: 4.8703316096 → 4.8703316089 Ω). Three-layer 40 m grid:
  19.0 → 0.25 s (ds = 2.5 m, deduplication only, bit-identical) and
  66.6 → 0.30 s (ds = 1.25 m, tabulated); a village of eleven rings at
  irregular positions (N = 273) takes 1.25 s instead of 34.1 s with the
  deduplication alone (impedance 1.8e-11 apart). Golden cases: none of the
  stored ones moves; new case `mom_sommerfeld_three_layer_grid_tabulated`.
  Test:
  `tests/test_mom_sommerfeld.py::test_tabulated_reaction_matrix_is_within_1e_6_of_the_quadrature`.
- ⚠ **`fem` raises when an electrode reaches a layer its hemisphere does
  not (10.23).** An electrode below a layer interface that lies above
  the equivalent-hemisphere radius (a rod through or below the top
  layer) now raises `BackendCapabilityError` (a `ValueError` and
  `NotImplementedError`) before meshing. 0.15.0 returned +633 … +3138 %
  over a conductive lower layer (1000/30 Ω·m) and −10 … −85 % over a
  resistive one silently. Affected: rods in layered soil solved with
  `fem`; golden cases: none. Test:
  `tests/test_fem.py::test_fem_rejects_an_electrode_below_its_hemisphere`.
- **The `mom` family shares the depth advisories of the image family
  (10.75):** `ShallowSegmentWarning` for segments long against their
  burial depth.

### Fixed

- **`mom`, `mom_sommerfeld`, `bem` solved a segment that crosses the soil
  surface as if it were buried (10.75):** a 1 m rod from z = −0.1 m gave
  94.27 Ω; the shared depth guard now raises as it has for `image` since
  0.15.0. Test:
  `tests/test_point_matching_pipeline.py::test_mom_family_rejects_a_segment_partly_above_the_surface`.
- **`FemReductionWarning` missed interfaces above the electrode
  (10.23, audit F01):** the hemisphere spans the soil from the surface,
  but only interfaces between the electrode and the hemisphere radius
  were checked; a 10 m strip and a ring r = 5 m at 0.8 m below
  h₁ = 0.5 m (30/1000 Ω·m) came out −70 % and −47 % against `mom`
  without a warning. Every interface above the deeper of the electrode
  and its hemisphere is now reported, and so is an interface below both
  but within the electrode's horizontal half-extent (ring r = 10 m over
  h₁ = 5 m in 1000/30 Ω·m: −46 %, silent before). Where no rule applies,
  rings and strips measured in 1000/30, 100/1000 and 100/400 Ω·m stay
  within −7.4 … +4.3 % of `mom`. Tests:
  `tests/test_fem.py::test_fem_warns_when_the_electrode_lies_below_a_thin_top_layer`,
  `tests/test_fem.py::test_fem_warns_when_an_interface_lies_within_the_electrode_extent`.
- **`compare_engines` reported 34.3 Ω against 0 Ω as consistent
  (10.71, audit F03):** a quantity was skipped as "undefined for every
  engine" as soon as one engine returned zero. Only a quantity that is
  zero for every engine is skipped now; a zero for some engines fails
  with an infinite deviation and a note. Test:
  `tests/test_compare_engines.py::test_compare_engines_counts_a_zero_of_one_engine_as_failure`.

### Deprecated

- **Backend names `cim`, `bem` and `image_nlayer` (10.76).** They keep
  working, emit a `DeprecationWarning` naming the replacement and are
  removed no earlier than 2027.5: `cim` runs the image auto-dispatch
  (`image`, `image_2layer`; bit-identical on every world compared),
  `bem` runs `mom` (it assembled the identical point-matching system;
  results move by ≤ 3.5e-16), `image_nlayer` is the name of the
  auto-dispatch target of `image` for `MultiLayerSoil` and keeps its own
  solve function, so `Engine(backend="image")` stays silent. Three or
  more distinct layers raise and name `mom_sommerfeld`. As a pure alias,
  `cim` has the capabilities of `image`: concrete-shell electrodes and
  `kernel="line"`, which it rejected before, now run. The modules
  `groundfield.solver.cim` (with `fit_complex_images`) and
  `groundfield.solver.bem` stay importable; `solve_cim` and `solve_bem`
  warn and delegate. `fem` is registered as experimental
  (`ExperimentalBackendWarning`, once per process). The registry entries
  are applied by the release coordinator. Tests:
  `tests/test_engine_aliases.py`.

### Removed

- Private helpers without a remaining caller:
  `groundfield.solver.mom._carson_skin_metadata`,
  `groundfield.solver.bem._build_Z_collocation`,
  `groundfield.solver.cim._cim_self_kernel_factory`; the re-imports of
  image-family helpers from `mom`, `mom_sommerfeld`, `bem` and `cim`.
