### Added

- **Skin effect in the solve (10.37, opt-in):** a conductor with
  `resistance_model="skin"` now changes the branch impedance per frequency:
  each of its branches (lumped branches and the half stubs of galvanic chains
  included) adds `l_b · Conductor.skin_correction_per_length(f)` to its diagonal,
  so the resistance becomes R_ac(f) and the internal inductance L_int(f) (exact
  solid-round Bessel solution). A conductor without an inductance model gets the
  resistance part only and makes the world frequency dependent. Through the
  solver, a 150 mm² Al PEN has R_ac/R_dc = 1.0556 / 1.2308 / 1.5496 at
  250 / 550 / 1000 Hz. The default `resistance_model="dc"` changes nothing.
  Tests: `tests/test_inductive_solver_integration.py::test_skin_model_raises_the_pen_resistance_at_1khz`,
  `::test_skin_correction_of_an_inductive_conductor`,
  `::test_skin_correction_covers_the_half_stubs_of_a_galvanic_chain`,
  `::test_skin_effect_without_inductance_model_changes_the_resistance_only`.
- **Tabulated Sommerfeld assembly (10.60, opt-in):**
  `build_sommerfeld_correction_matrices(..., method="tabulated")` (and the
  single-frequency builder) groups the segments by depth and shares one table of
  the spectral integral S(ρ) per pair of depth groups and frequency (100 nodes,
  cubic spline in ln(ρ + z)); the outer quadrature of all segment pairs runs
  vectorised. Against a converged reference (200 instead of 24 ρ nodes per pair)
  it is closer than the per-pair rule: 1.7e-6 against 1.4e-5 of the largest
  entry on a 10-house street network (two-layer earth: 2.7e-6 against 3.6e-6);
  the substation impedance agrees with the converged reference to 5e-10
  (per-pair rule 1.1e-8). On R8's W3 network a full solve with 11 frequencies
  takes 2.6 s instead of 27 s CPU (10 houses) and 7.8–9.1 s instead of 242 s
  (30 houses, 249 inductive branches; impedances equal to 3e-8). Blocks whose
  λ grid cannot be resolved (conductors in the surface plane across a large
  network) keep the per-pair rule. The solver selects it through the
  engine attribute `sommerfeld_assembly` (`"exact"` when absent); the default
  stays the per-pair assembly. Tests: `tests/test_sommerfeld_tabulated.py`.

### Changed

- ⚠ **Exact touching segment pairs by default (10.46):** chain neighbours and
  bends share an end point where the Neumann integrand is log-singular; the
  16×16 Gauss–Legendre rule is −0.3 % … +0.8 % off for such pairs. The solver and
  `build_inductance_matrix`, `neumann_mutual`, `perfect_mirror_self_pair_inductance`
  now use the closed forms (`touching="exact"`); `touching="quadrature"`
  reproduces 0.15.0. A straight conductor keeps the inductance of the unsplit
  wire for every discretisation. Every result with an inductive conductor moves:
  1.2e-6 (50 Hz) and 2.3e-5 (1 kHz) on the substation impedance of R8's 2-house
  street network (1.2e-4 at 1 kHz on the 30-house network, together with the
  lumped-branch fix below), ≤ 3.2e-6 in the golden cases
  `cond_galvanic_neumann_perfect_mirror`, `cond_galvanic_neumann_carson_series`,
  `cond_galvanic_neumann_sommerfeld`, `defaults_mixed_cluster_inductive`,
  `line_cond_galvanic_neumann`, `cond_coupled_leads_open_circuit`,
  `cond_isolated_pen_sommerfeld_two_layer`, `gen_tn_measurement_overhead_leads`,
  `fit_rho_f_standard_sweep`.
- **`sommerfeld` on the image-free baseline (10.46):** the solver assembles the
  Neumann matrix and the Sommerfeld correction against free space
  (`image="none"`) instead of adding the additive image and removing it again:
  exact, cheaper, the same numbers to round-off (≤ 4e-18 relative). Golden cases:
  the three `sommerfeld` cases, round-off only.
- **One earth-return assembly for all frequencies (10.60):** the solver builds
  the Sommerfeld or Carson correction for the engine's whole frequency set at
  once on first use (geometry, λ grid and Bessel tables once per segment pair);
  every frequency is bit-identical to a single-frequency assembly. Sommerfeld
  matrices of R8's W3 network for 11 frequencies: 7.4–8.2 s → 1.5–1.9 s CPU
  (2 houses), 137–141 s → 27–28 s (10 houses); Carson matrices of W3c
  (30 houses): 4.4 s → 2.4 s. Full solves (with the wave-1 gains): W3, 2 houses
  35.3 s in 0.15.0 → 2.1 s; W3, 30 houses about 3 600 s (review estimate) →
  242 s; W3c, 30 houses 16.4 s → 3.3 s. The correction keeps at most 256 MiB
  of matrices (longer sweeps are assembled in chunks). Golden cases: none.
  Tests: `tests/test_inductive_solver_integration.py::test_sommerfeld_pair_work_does_not_grow_with_the_number_of_frequencies`,
  `::test_branch_correction_is_built_once_for_all_frequencies`,
  `::test_branch_correction_memory_bound_keeps_the_numbers`.

### Fixed

- ⚠ **`carson_series` on the wrong image baseline (10.20, F32):** Carson's
  correction (ωμ₀/π)(P + jQ) is defined against the anti-parallel image, but the
  solver added it on top of the additive image of ADR-0004, which counts the
  image twice: the earth-return reactance of a 1 km PEN was 2.04× (50 Hz) and
  2.19× (1 kHz) Carson's value. The solver now builds the inductance matrix with
  `image="antiparallel"`; the 1 km PEN matches l·X′_Carson(GMR = a·e^(−1/4)) to
  7e-6 (50 Hz) and 4.2e-4 (1 kHz). On R8's W3c network (30 houses, PEN Neumann)
  the substation impedance at 1 kHz drops from 0.4288 + 0.3439j Ω to
  0.3964 + 0.2301j Ω (|Z| −16.6 %). Affects every `carson_series` result with an
  inductive conductor (all backends that share the inductive assembly). Golden
  case: `cond_galvanic_neumann_carson_series` (6.0e-3).
  Test: `tests/test_inductive_solver_integration.py::test_carson_series_reactance_matches_carson`.
- ⚠ **Carson's small-a series (10.46):** for a ≤ 0.25 only the leading terms
  were used (no a² term −(π/64)a²cos2θ of Q, constants rounded): P and Q were up
  to 2.4e-4 and 3.0e-3 off at a = 0.25, and the θ = 0 quadrature above it
  6.4e-6. The series is now summed to a¹² with the exact constants and the
  θ = 0 quadrature uses a graded composite rule; both are at round-off against
  mpmath and the regime boundary is continuous. Moves `carson_series` results
  (included in the bound of the entry above).
  Tests: `tests/test_carson_pq.py::test_small_a_regime_is_continuous`,
  `::test_small_a_series_matches_an_independent_quadrature`,
  `::test_theta_zero_quadrature_is_at_round_off`.
- ⚠ **Neumann inductance of lumped and single-segment conductors (10.24):** a
  conductor with `inductance_model="neumann"` that the topology keeps as one
  lumped branch (`discretize_segment_length=None`, or a single sub-segment) lost
  its inductance silently: a 300 m PEN was 0.0564 + 0j Ω at 1 kHz. It is now one
  inductive branch from start to end, coupled to every other inductive branch
  (0.2101 + 3.7839j Ω with `sommerfeld`; a 10 m chain gives 0.2096 + 3.7840j Ω;
  with `perfect_mirror` and `carson_series` the single branch equals the refined
  chain to 1e-9). Affects PEN service drops shorter than the PEN segment length
  (5 m by default) and lumped leads when the conductor is inductive; on R8's
  2-house street network the substation impedance moves by 1.3e-6 at 1 kHz.
  Golden cases: none. Tests:
  `tests/test_inductive_solver_integration.py::test_neumann_is_honoured_on_lumped_and_single_segment_conductors`,
  `::test_lumped_branch_equals_the_distributed_chain`,
  `::test_lumped_inductive_conductor_couples_to_a_parallel_chain`.
- **Conductors that leave the soil under `sommerfeld` (10.46, F52):** a
  conductor crossing z = 0, or lying in the surface plane, is solved with the
  default earth model (the pair integrals split it at the surface; a PEN in the
  plane agrees with one 1 mm below it to 1e-7); with `perfect_mirror` and
  `carson_series` its image is singular and the solve raises as before. Tests:
  `tests/test_inductive_solver_integration.py::test_sommerfeld_accepts_a_surface_crossing_conductor`,
  `tests/test_audit_guards.py::test_surface_pen_with_neumann_is_continuous_under_sommerfeld`.
- **Invalid escape sequences in `coupling.inductance` docstrings (audit A1-12,
  10.95):** three docstrings raised `SyntaxWarning` on compilation (an error under
  `python -W error`), and `\text` in the module docstring turned into TAB
  characters; they are raw strings now. Tests:
  `tests/test_coupling_conventions.py::test_modules_compile_without_invalid_escape_sequences`,
  `::test_docstrings_contain_no_control_characters`.
- **`segment_distance` with a zero-length segment (audit F04, 10.44):** a point
  segment was measured from the start of the other segment (5.099 m instead of
  1.0 m for a point 1 m beside the middle of a 10 m cable), so
  `find_close_wires` / `find_close_conductors` missed a point-like bond 2 cm from
  a cable; points now take the clamped orthogonal projection. Tests:
  `tests/test_conductor_proximity.py::test_segment_distance_with_point_segments`,
  `::test_point_like_bond_next_to_a_cable_is_reported`.
