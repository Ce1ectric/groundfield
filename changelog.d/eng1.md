### Added

- **Engine benchmark suite B1–B13 (10.76, R2 §3.3):** `tests/benchmarks/`
  holds reference worlds with analytic or independently computed reference
  values (hemisphere, thin-wire Galerkin rod, closed-form thin ring,
  Tagg/Sunde series, mpmath three-layer kernel, floating-port two-port)
  and one explicit tolerance table per backend
  (`tests/benchmarks/tolerances.py`; `python -m tests.benchmarks.tolerances`
  prints it as Markdown for the validation page). Skipped by default; run
  with `pytest tests/benchmarks --run-benchmark`. The image-family
  cross-layer rod (B7) is a strict xfail until 10.15 lands.
- **`FemReductionWarning` (10.23):** `fem` names every reason why its
  equivalent-hemisphere reduction does not model the world (several
  electrodes in a cluster, several clusters, a layer interface between the
  electrode and its hemisphere), once per solve. Exported from
  `groundfield.solver.fem`. Test:
  `tests/test_fem.py::test_fem_warns_when_electrode_and_hemisphere_straddle_an_interface`.
- **`compare_engines` tolerance table (10.71):** `EngineComparison.rows`
  (one `ComparisonRow` per quantity, frequency and real / imaginary part,
  with values, reference, deviation, tolerance and verdict), `frequencies`,
  `impedances` (complex values per engine and frequency),
  `input_deviations`; new keyword `component_floor` (default 1e-3).
  `summary()` prints the table. Test:
  `tests/test_compare_engines.py::test_compare_engines_tolerance_table_is_explicit`.
- **`mom_sommerfeld` convergence metadata (10.47):** `converged` (bool,
  valid for every frequency), `converged_per_frequency`,
  `n_unconverged_entries` and, for two-layer soil, `n_terms_used`. Test:
  `tests/test_mom_sommerfeld.py::test_solve_reports_convergence_in_the_metadata`.
- **`ComplexImageFit.max_error` (10.49):** maximum error of the
  complex-image fit on the verification grid including λ = 0. Test:
  `tests/test_cim.py::test_fit_converged_flag_bounds_the_maximum_error`.

### Changed

- ⚠ **`mom`, `bem`, `mom_sommerfeld`: source-free clusters float (10.29).**
  A cluster without a source is now an unknown with zero net current (a
  floating equipotential) instead of being dropped and read back as the
  mean of the incident field. Numbers move for every world with a passive
  cluster in these backends: potentials read at a passive electrode by
  −59 % (rod through h₁ = 2 m in 1000/30 Ω·m: 3.548 → 1.440 Ω transfer
  impedance) to +1.5 % (100/1000 Ω·m), −0.4 % in homogeneous soil; the
  driven electrode's impedance by −0.009 % to −0.49 % because the floating
  neighbour now perturbs the field (R2 two-port worlds, 4 m apart).
  Passive networks joined by finite conductors carry their equalising
  current. Affected: probe and auxiliary electrodes, unbonded neighbours
  and passive conductor networks solved with the MoM family; the image
  family and the AP1 catalogue path are unaffected. Golden cases: none of
  the stored cases has a passive cluster in these backends (bit-identical);
  new case `mom_two_port_passive_ring_two_layer`.
- ⚠ **`cim` reports the length-weighted node potential (10.30)**, as
  `image_2layer` has since 0.15.0 (F10). Numbers move on electrodes with
  unequal segment lengths: −3.9 % (rod split unevenly at h₁ = 2 m:
  71.060 → 68.262 Ω), −1.1 % (the F10 rod shape); uniform segments are
  unchanged. Golden cases: `cim_rod_two_layer` bit-identical; new case
  `cim_grid_unequal_segments`.
- ⚠ **Canonical layer stack (10.48).** The layered backends (`mom`, `bem`,
  `cim`, `mom_sommerfeld`, and `image_nlayer` through `as_layer_stack`)
  merge adjacent layers of equal resistivity and drop zero-thickness layers
  before choosing a kernel: `MultiLayerSoil([ρ, ρ, ρ])` is homogeneous and
  `[ρ₁, ρ₂, ρ₂]` is the two-layer soil it describes. Stacks that raised
  before now solve; `mom` accepts a `MultiLayerSoil` with at most two
  distinct layers; `FieldResult.soil` carries the canonical soil model, so
  `potential()` works for such stacks. Numbers move only for equal-
  resistivity declarations whose electrodes cross the declared interface
  (no longer split there): +0.07 % (`mom`, `bem`, `mom_sommerfeld`) and
  +0.42 % (`cim`) for a 3 m rod through h = 2 m in `TwoLayerSoil(100, 100, 2)`.
  `fem` keeps the declared stack. Golden cases: none moved; new case
  `mom_sommerfeld_duplicated_stack`.
- ⚠ **`fem`: the truncation sphere no longer costs radial resolution
  (10.49).** For a layer stack deeper than 1 m the radial line count grows
  with ln(r_far / a_eq), so the radial step ratio stays that of the
  homogeneous mesh. Layered `fem` results move up by +0.16 % (h₁ = 5 m),
  +0.41 % (30 m), +1.07 % (1 km) and +1.64 % (10 km) for a 1.5 m rod with
  ρ₂ = ρ₁ (from −0.16 % … −1.62 % to within ±0.005 % of the homogeneous
  value); homogeneous soil and stacks with Σh ≤ 1 m are unchanged.
  `metadata["fem_mesh"][cluster]["n_radial"]` reports the count used.
  Affected: every layered `fem` result. Golden cases: `fem_rod_hom`
  bit-identical; new case `fem_rod_two_layer_deep`.
- **`fem` is experimental (10.23).** Module and `solve_fem` docstrings say
  what the equivalent-hemisphere reduction is and is not (no longer "the
  third independent cross-check" and "better than 5 %"); the registry flag
  follows with the backend registry.
- **`compare_engines` compares complex impedances at every frequency
  (10.71):** cluster impedances and the input impedance of every current
  source, real and imaginary part separately, at all frequencies (up to
  0.15.0: Re Z of the clusters at the first frequency). Engines with
  different frequency lists raise `ValueError`; a value that is
  non-finite for some engines only is a failure. Comparisons that passed
  before may now be reported inconsistent; `deviations` per cluster is
  the worst over all frequencies and both parts.
- **`metadata["solver"]` is `"point_matching"` for `mom`, `bem` and
  `mom_sommerfeld` (10.31)** (was `"galerkin"` / `"collocation"` /
  `"galerkin"` for one and the same scheme).

### Fixed

- **Transfer impedances of the MoM family are reciprocal (10.29).** Z_AB
  and Z_BA from two solves differed by 4.0e-3 (homogeneous) to 1.46 (rod
  through the interface); now below 1e-8 (cross-layer quadrature) and
  7e-16 (closed form). Test:
  `tests/test_mom_family.py::test_two_port_transfer_impedance_is_reciprocal`.
- **`cim` is bit-identical to `image_2layer` and reciprocal (10.30)**, as
  its `reduces_to` metadata and the 0.15.0 notes claimed. Test:
  `tests/test_cim.py::test_cim_is_bit_identical_to_image_2layer_for_unequal_segments`.
- **Physically two-layer or homogeneous `MultiLayerSoil` stacks no longer
  raise (10.48)** in `cim`, `bem`, `mom`, `mom_sommerfeld` and
  `image_nlayer`. Test:
  `tests/test_layer_stack.py::test_duplicated_layer_stack_is_solved_like_two_layers`.
- **`mom_sommerfeld`: a capped oscillation budget is no longer silent
  (10.47).** When `max_osc_panels` binds, every rung of the refinement
  ladder shares the capped J₀ grid, so two rungs could agree while the
  absolute floor `epsabs` dominated: kernels 5 km apart over a 5 cm layer
  were up to 3.1e-3 wrong without a `SommerfeldConvergenceWarning` (7 of
  1050 stress points). A binding budget now counts as non-convergence;
  the largest silent error on the stress grid is 2.0e-10. The truncated
  Tagg/Sunde series of the two-layer diagonal warns like in `mom`. Test:
  `tests/test_mom_sommerfeld.py::test_kernel_warns_or_converges_when_the_oscillation_budget_binds`.
- **`fit_complex_images` no longer reports `converged=True` for a fit that
  misses Γ₁ by more than `rms_tol` (10.49).** The flag was RMS-gated:
  26 of 300 random 3–6-layer stacks passed with a maximum error of up to
  3.5e-3 between the fitting samples; the maximum error on the
  verification grid including λ = 0 is now part of the gate (0 of 300
  silent). No solver path uses the fit. Test:
  `tests/test_cim.py::test_fit_converged_flag_bounds_the_maximum_error`.
- **`compare_engines` sees the reactance, every frequency and non-finite
  results (10.71).** Perfect-mirror and Sommerfeld earth return (input
  reactance 0.600 vs 0.376 Ω at 1 kHz) were reported consistent. Test:
  `tests/test_compare_engines.py::test_compare_engines_sees_the_reactance_at_every_frequency`.
- **`fem` warns where its reduction is not a valid cross-check (10.23):**
  +1092 % for a 3 m rod through h₁ = 2 m, −92 % for a 10 m ring above it,
  −35 % for a ring with three bonded rods, transfer impedance 0 between
  clusters — all silent before. Test:
  `tests/test_fem.py::test_fem_warns_for_a_multi_electrode_cluster`.
- **`fem` truncation (10.49):** see Changed. Test:
  `tests/test_fem.py::test_fem_deep_interface_does_not_degrade_the_resolution`.

### Documentation

- **`mom`, `bem` and `mom_sommerfeld` are documented as one point-matching
  scheme (10.31):** pulse basis, delta test at the segment midpoints,
  point-source near field. `mom` is not Galerkin and its bordered system is
  not symmetric; it lies 1.0 % (ds = 0.5 m) to 1.7 % (ds = 0.0625 m) below
  the exact thin-wire Galerkin solution of a 3 m rod and drifts by −0.96 %
  over that ladder, while `image` is closer at every ds; `bem` assembles
  the identical matrix; `mom_sommerfeld` shares the discretisation and is a
  kernel reference, not an independent scheme. The stale advice to lower
  `lambda_max_factor` on the delegated cross-layer path is removed (factors
  30 … 20 000 change that kernel by ≤ 1.3e-8). No physics change. Test:
  `tests/test_mom_family.py::test_mom_family_reports_one_point_matching_scheme`.

### Removed

- The unused private helpers `cylindrical_radius` and
  `_split_layer_thicknesses` of `groundfield.solver._layered` (10.81).
