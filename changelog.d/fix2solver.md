### Added

- **`groundfield.solver.PointKernelInterfaceWarning`:** warning category for
  a horizontal wire within 0.6 segment lengths of the layer interface on the
  point-form cross-layer path (see *Fixed*, point form of the cross-layer
  path).
- **`expected_segments(...)["nodal_unknowns"]` (10.06):** the size of the
  dense augmented nodal system (current owners, nodes and branches, every
  conductor sub-piece included), an upper bound computed in closed form and
  checked against `HARD_LIMIT` like the segment count. Test:
  `tests/test_diagnostics_report.py::test_nodal_unknowns_are_the_size_of_the_solved_system`.

### Changed

- **A world without a current source raises in every backend (10.66, audit
  A2S-05):** `Engine.solve` validated with `World.check(require_source=False)`,
  and an earlier entry of this release said such a world "stays solvable
  through `Engine.solve` (all results zero)". Only `fem` did; the image and
  `mom` families and `World.solve` raised. `Engine.solve` now runs the full
  check, so every backend and both entry points raise the same `ValueError`
  ("has no current source"), as the 10.66 concept lists. Test:
  `tests/test_source_validation.py::test_a_world_without_a_source_raises_in_every_backend`.
- ⚠ **`TwoLayerSoil(rho, rho, h)` is cut nowhere (10.48, audit A2S-07):**
  `image_2layer` (also reached through `image` and the auto dispatch) cut
  wires at `h` even when both layers have one resistivity, while `mom`,
  `mom_sommerfeld` and `image_nlayer` did not. Every backend now cuts where
  the resistivity changes only (the canonical stack), so such a soil solves
  exactly like homogeneous soil: a rod 0.3–6.3 m in `TwoLayerSoil(100, 100,
  2.15)` had 13 segments and sat 3.8e-4 (point kernel) and 3.3e-7 (line
  kernel) from the homogeneous result; now 12 segments, equal to 2.2e-16 and
  bit for bit. Who is affected: worlds with an equal-resistivity two-layer
  soil and a wire through `h` (not AP1: its station rods end above `h_1`).
  Golden cases: none. Test:
  `tests/test_diagnostics.py::test_equal_resistivity_two_layer_soil_is_the_homogeneous_soil`.

### Fixed

- ⚠ **Interface contact (10.32, audit A2S-01):** an interface cut that would
  leave a wire part shorter than the wire diameter was dropped, and the part
  joined its neighbour in the other layer, so the contact with that layer was
  ignored: a 4.5 m rod (a = 8 mm) in 1000/30 Ω·m read 195.0 Ω with its tip
  15.9 mm into the lower layer and 128.6 Ω at 16.1 mm (+52 %), in
  `image_2layer` and the `mom` family alike, and the same at the head of a
  rod. Every interface cut is now kept, however short the part beyond it;
  the thin-wire guard concerns the chosen segment length only. The Galerkin
  self term (it lost every digit below L/a = 1e-8) and the line kernel's
  parallel-pair integral (two parts of 1e-11 m: 0 instead of 125 1/m, the rod
  read 72.7 instead of 133.5 Ω) are evaluated without cancellation for such
  parts; every regular segment keeps its arithmetic bit for bit. The
  impedance is now continuous in the penetration depth (line kernel 133.5 →
  128.6 Ω from 10 pm to 16.1 mm, the same shape for the point kernel, `mom`
  and `mom_sommerfeld`); the step that remains is the onset of the contact
  itself between a tip that ends at the interface and one that reaches past
  it (195.9 → 133.5 Ω here). Who is affected: rods whose tip or head lies
  within 16 mm (two wire radii) of an interface: −34 % in this example.
  The constant `geometry.electrodes.SLIVER_RADII` introduced earlier in this
  release is gone. Golden cases: none. Tests:
  `tests/test_interface_slivers.py` (`::test_every_interface_cut_is_kept`,
  `::test_rod_resistance_is_smooth_in_the_tip_penetration`,
  `::test_rod_resistance_is_continuous_in_h1_at_the_former_sliver_threshold`,
  `::test_a_tip_twelve_millimetres_into_a_conductive_layer_is_felt`,
  `::test_a_head_just_inside_a_conductive_upper_layer_is_felt`,
  `::test_parallel_pair_integral_of_parts_much_shorter_than_their_distance`).
- ⚠ **Point form of the cross-layer path (10.28, ADR-0007):** with
  `kernel="point"` (and in the `mom` family, `mom_sommerfeld` with two layers
  and `solve_mutual_*` with the point kernel) the diagonal of the cross-layer
  reaction matrix averages the layered correction over a 3-point rule along
  the depth axis, ±0.387 L, whatever the segment's orientation. For a
  horizontal wire next to the interface the rule reached across it and
  returned wrong, even negative impedances (a rod through h₁ = 1.2 m with a
  20 m strip at 0.8 m, 300/50 Ω·m, ds = 2 m: −19.37 Ω against 9.62 Ω with
  the line kernel; v0.15.0: −14.24 Ω; −14 % … −552 % over the measured
  cases), or reached above the soil surface and raised a bare `ValueError`
  about negative depths. Such worlds now raise `BackendCapabilityError`
  naming the line kernel (or `image_2layer` for the `mom` family) and a
  segment length that fits; a horizontal wire within 0.6 segment lengths of
  h₁ warns with `PointKernelInterfaceWarning` (−3 % … −17 %). Rods (cut at
  h₁) and every valid configuration keep their numbers; the default line
  kernel is unaffected. Golden cases: none. Tests:
  `tests/test_cross_layer_point_form.py`,
  `tests/test_cross_layer_line_kernel.py::test_point_kernel_keeps_its_numbers_near_the_interface`.
- ⚠ **`fem` below a thin top layer (10.23, audit A2E-2):** the guard raised
  only when an interface lies between the hemisphere radius `a_eq` and the
  electrode's deepest point; a 3 m rod wholly below h₁ = 0.4 m < `a_eq`
  (1000/30 Ω·m) only warned and returned 31.1 Ω (`mom`: 10.5 Ω, +197 %). It
  now also raises when every electrode of a cluster lies below an interface
  between `a_eq/2` and `a_eq` (more than half of the hemisphere in a layer
  the electrodes do not touch; +51 … +130 % and −81 … −98 % measured for a
  3 m rod against `image_2layer`). Thinner top layers keep the warning
  (+16 … +46 %, −38 … −58 % for 100/1000 Ω·m), as do bonded rods through the
  top layer. Golden cases: none. Tests:
  `tests/test_fem.py::test_fem_rejects_a_rod_below_a_top_layer_holding_most_of_its_hemisphere`,
  `::test_fem_only_warns_below_a_thin_top_layer`,
  `::test_fem_guard_applies_to_a_bonded_cluster_wholly_below_the_interface`.
- **Resource guard (10.06, audit A2S-02):** `check_segment_budget` counted
  leakage segments only, so every sub-piece of an isolated distributed
  conductor (a branch and an interior node of the dense nodal system and a
  row of the inductance matrix) counted as zero: a 100 m PEN in 2000 pieces
  predicted 12 segments and took 407 MiB, in millimetre pieces the solve died
  with a `MemoryError`. The guard now also predicts the unknowns of the
  nodal system in closed form and raises `ResourceLimitError` above
  `HARD_LIMIT` before anything of that size is allocated (an inductive chain
  before the inductance assembly); `diagnose()` reports it. Tests:
  `tests/test_diagnostics_report.py::test_conductor_chain_beyond_the_limit_is_rejected`,
  `::test_a_chain_in_millimetre_pieces_is_rejected_before_allocation`,
  `::test_diagnose_reports_a_nodal_system_beyond_the_limit`.
- **Floating clusters in `solve_mutual_*` (10.29, audit A2S-03):** every
  excitation rebuilt the multi-port matrix of all floating clusters and
  factorised the full nodal system again, so the cost grew with the square
  of the number of bonded grounding nodes (256 ring-and-rod nodes: 199.9 s
  against 24.3 s before floating). The floating core is now assembled and
  factorised once with one right-hand side per anchor; an anchor outside the
  core enters through its coupling column. 256 nodes: 5.6 s; 100 rotated
  foundations of four bonded strips: 2.7 s (24.0 s); results equal to
  1e-12 (golden values bit-identical). Tests:
  `tests/test_mutual_matrix.py::test_shared_factorisation_equals_one_floating_solve_per_anchor`,
  `::test_one_core_assembly_and_factorisation_for_all_anchors`.
- **`expected_segments` for layer stacks (10.79, audit A2S-07):** the
  prediction cut wires at every declared interface, the backends at the
  interfaces of the canonical stack only: a rod 0.3–6.3 m in
  `MultiLayerSoil([100 Ω·m, h = 2.15 m], [100 Ω·m])` was predicted at 13
  segments and solved with 12. One rule now feeds both. Test:
  `tests/test_diagnostics.py::test_expected_segments_matches_the_solve_for_every_layer_stack`.
- **`fem` ignored the skin effect silently (10.37, audit A2C-4):** `"skin"` is
  a conductor feature of the backend registry; `fem` (lumped, DC) warns with
  `BackendCapabilityWarning` naming it, while `image` and `mom` raise the
  rod-to-rod voltage of a 300 m 150 mm² Al PEN by +54.9 % from 0 Hz to 1 kHz.
  A registered backend that lists its conductor features without `"skin"`
  warns too. Tests:
  `tests/test_backend_registry.py::test_a_backend_that_ignores_the_skin_model_says_so`,
  `::test_registered_backend_without_the_skin_feature_warns`.
- **Duplicate names in a loaded or edited world (10.66, audit A2S-06):**
  `World.check()` did not check that names are unique (only the `add_*`
  methods did): two electrodes of one name from JSON failed in the solver
  with a singular matrix, an electrode and a junction of one name were merged
  into one node. `check()` (run by every solve) now rejects a name used twice
  among the nodes, the conductors or the sources. Test:
  `tests/test_source_validation.py::test_check_reports_names_used_twice`.
- **`convergence_study` repeated show-once warnings (A1-02, audit A2S-08):**
  it copied the engine per step with the validating `Engine.model_copy`,
  whose warning capture resets every module's registry; a shallow strip
  showed the same `ShallowSegmentWarning` once per step. It now copies
  internally and validates the segment lengths itself (finite and strictly
  positive). Tests:
  `tests/test_convergence.py::test_convergence_study_shows_each_distinct_warning_once`,
  `::test_convergence_study_rejects_non_finite_segment_lengths`.

### Security

- **Crafted result files (A1-01, audit A2S-04):** a `$complex` metadata tag
  holding an integer beyond the float range (`{"$complex": [10**400, 0]}`,
  also inside `$tuple`, as a `$dict` key or an `$ndarray` item) escaped
  `load_result` as a bare `OverflowError` instead of the documented
  `ValueError`; it is now a `ValueError` naming the metadata key. Test:
  `tests/test_result_metadata_codec.py::test_a_complex_tag_beyond_the_float_range_is_a_value_error`.
