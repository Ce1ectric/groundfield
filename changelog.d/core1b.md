### Added
- **Backend registry (10.74):** `groundfield.solver.registry` with
  `register_backend(name, solve_fn, *, capabilities, options_model=None,
  aliases=(), deprecated_alias_of=None, experimental=False, forward=None)`,
  `get_backend`, `available_backends`, `unregister_backend`,
  `BackendCapabilities` and `BackendSpec`. `Engine.solve` dispatches through
  the registry instead of an if-chain, so a backend is added without editing
  `Engine.solve`; the automatic forwarding of `image` to `image_2layer` /
  `image_nlayer` by soil model is kept. A deprecated alias warns with its
  replacement and the earliest removal (`2027.5`); an experimental backend
  warns once per process (`ExperimentalBackendWarning`). Tests:
  `tests/test_backend_registry.py::test_new_backend_runs_without_editing_engine_solve`,
  `::test_deprecated_alias_warns_with_replacement_and_removal_version`,
  `::test_experimental_backend_warns_once_per_process`,
  `::test_image_is_auto_forwarded_by_soil_model`.
- **Capability check before the solve (10.74):** a world the backend cannot
  represent (soil layer count, concrete-shell electrodes, an explicitly
  requested source kernel) raises `BackendCapabilityError` before any
  backend code runs and names the registered backends that can solve it; a
  stack of equal layers counts as the merged stack. A backend that solves
  the world with a simplification warns with `BackendCapabilityWarning`:
  `fem` reduces distributed and inductive conductors to lumped resistive
  branches, and every backend still returns the current of a source with
  `return_to` through remote earth. Tests:
  `tests/test_backend_registry.py::test_capability_error_names_a_backend_that_can_solve_the_world`,
  `::test_capability_check_runs_before_the_backend`,
  `::test_two_terminal_source_warns_when_the_backend_ignores_return_to`,
  `::test_fem_warns_when_it_reduces_a_distributed_conductor`.
- **Metadata core (10.74):** every result carries `backend`,
  `effective_backend`, `kernel`, `groundfield_version`, `frequencies`,
  `n_segments`, `soil_kind`, `earth_inductive_model`, `solve_time_s`,
  `series` and `provenance` (`groundfield.solver.engine.CORE_METADATA_KEYS`)
  next to the backend-specific keys, which are unchanged. `series` records
  the engine's image-series controls for the post-solve field evaluation.
  Test: `tests/test_backend_registry.py::test_every_backend_reports_the_metadata_core`.
- **Backend options (10.74):** `Engine.options`, keyed by backend name and
  validated by `MomSommerfeldOptions` (`mom_sommerfeld`) and `FemOptions`
  (`fem`), with a did-you-mean message for misspelled names;
  `Engine.backend_options(name)` returns the options a backend runs with.
  Tests: `tests/test_engine_options.py::test_nested_sommerfeld_options_reach_the_backend`,
  `::test_misspelled_option_is_rejected_with_a_suggestion`.
- **Provenance (10.77, F22):** `metadata["provenance"]` of every solve holds
  the groundfield, Python, numpy, scipy and pydantic versions, the platform,
  the engine configuration, `world_fingerprint` (SHA-256 of the canonical
  world JSON, `groundfield.io.world_fingerprint`) and the UTC time of the
  solve. Tests: `tests/test_result_persistence.py::test_every_solve_records_provenance`,
  `::test_world_fingerprint_is_canonical_and_sensitive`,
  `::test_fingerprint_describes_the_world_as_built`.
- **Versioned result files (10.77, F23):** `groundfield.io.save_result` /
  `load_result` write and read one standard-JSON file with the envelope
  `{"format": "groundfield.result", "schema_version": 1, "written_by",
  "provenance", "result"}` (no pickle, atomic write, no silent overwrite);
  a newer schema raises `ResultFormatError`. Tests:
  `tests/test_result_persistence.py::test_save_and_load_result_round_trip`,
  `::test_load_result_rejects_a_newer_schema`.
- **`FieldResult.to_dataframe()` (10.69, 10.72):** tidy table with one row
  per frequency and electrode (`frequency_hz`, `electrode`, `cluster`,
  `potential_v`, `current_a`, `cluster_impedance_ohm`); `to_dataframe("sources")`
  gives one row per frequency and source with the impressed current and the
  input impedance. pandas is imported on first use. Tests:
  `tests/test_result_dataframe.py::test_electrode_table_is_tidy_and_matches_the_accessors`,
  `::test_source_table_carries_the_input_impedance`.

### Changed
- **`Engine.backend`** is validated against the registry (a plain `str`
  annotation; `Backend` stays the `Literal` of the built-in names for type
  hints). An unknown name raises with the closest registered name as
  suggestion. Test:
  `tests/test_backend_registry.py::test_unknown_backend_name_suggests_the_closest_registered_one`.
- **Unsupported worlds fail earlier:** `cim`, `bem`, `mom` and `image_nlayer`
  on soils with three or more distinct layers, and every non-image backend on
  concrete-shell electrodes, now raise `BackendCapabilityError` from
  `Engine.solve` and list the backends that can. It subclasses `ValueError`
  and `NotImplementedError`, the types `image_nlayer`, `cim`, `bem` and the
  concrete-shell guard raised before; `mom` raised `TypeError` for three or
  more layers.
- **One default for the image-series cap (10.74):** `DEFAULT_IMAGE_MAX_TERMS
  = 300` and `DEFAULT_IMAGE_SERIES_TOL = 1e-6` (in
  `groundfield.solver.registry`) are the engine defaults and are recorded on
  every result as `metadata["series"]`. The image-series controls stay
  common `Engine` fields because six backends and the post-solve layer read
  them. Test: `tests/test_engine_options.py::test_series_default_is_one_documented_value`.
- **Engine JSON:** `options` replaces the four flat `sommerfeld_*` keys;
  engine JSON written by 0.15.0 still loads, silently when it carries the
  default values. Test: `tests/test_engine_options.py::test_v0150_engine_json_loads_without_warnings`.
- **Result JSON** writes non-finite floats of typed fields as `"Infinity"` /
  `"NaN"` strings instead of `null`, and tagged objects in `metadata` (see
  the `groundfield.solver.result` module docstring), so it is standard JSON
  and round-trips exactly.
- Warnings of the pre-flight resolution check of `Engine.solve` are
  attributed to the caller's line instead of a line inside groundfield.

### Fixed
- **Lossy result JSON (10.77):** `metadata["penetration_depth"]` lost its
  float keys and its `inf`, and numpy values in the metadata could not be
  serialised at all. Tests:
  `tests/test_result_persistence.py::test_result_json_round_trip_is_lossless`,
  `::test_numpy_and_non_json_metadata_survive_the_round_trip`.
- **Backend options silently accepted for another backend (10.74):** the
  `sommerfeld_*` fields were accepted by every backend without effect;
  options for a backend that will not run now warn
  (`BackendOptionsWarning`). Test:
  `tests/test_engine_options.py::test_options_for_another_backend_warn`.
- **`fem` knobs unreachable (10.74):** `n_radial`, `n_axial`, `r_far_factor`,
  `z_far_factor` and `far_field` can now be set through
  `Engine(options={"fem": {...}})`. Tests:
  `tests/test_engine_options.py::test_fem_knobs_are_reachable_through_engine_options`,
  `::test_fem_far_field_option_reaches_the_solver`.
- **`Engine.solve` let a backend rewrite the caller's sources (10.72):**
  `World.solve`, `Engine.solve` and `run_simulation` now share one
  identity-preserving snapshot and restore of `World.sources`. Test:
  `tests/test_solve_entry_points.py::test_every_entry_point_restores_the_sources`.
- **Inert engine settings accepted silently (10.51):** `Engine.tolerance` and
  `Engine.max_iterations` are read by no backend; a non-default value now
  warns. Test:
  `tests/test_engine_options.py::test_tolerance_and_max_iterations_warn_that_they_have_no_effect`.

### Deprecated
- `Engine.tolerance` and `Engine.max_iterations` (no effect): a non-default
  value or attribute access emits `DeprecationWarning`; removal no earlier
  than `2027.5`.
- The flat fields `Engine.sommerfeld_lambda_max_factor`, `sommerfeld_epsabs`,
  `sommerfeld_epsrel` and `sommerfeld_max_osc_panels`: use
  `Engine(options={"mom_sommerfeld": {...}})` or `MomSommerfeldOptions`. They
  are still accepted (constructor, JSON, attribute read and write); removal
  no earlier than `2027.5`. Test:
  `tests/test_engine_options.py::test_flat_sommerfeld_fields_are_mapped_and_deprecated`.
- `World.solve(snapshot_sources=False)`: every entry point takes the
  snapshot now, so the opt-out saves nothing; removal no earlier than
  `2027.5`. Test: `tests/test_solve_entry_points.py::test_snapshot_opt_out_is_deprecated`.

### Removed
- The private, unreachable `Engine._stub_result` and `Engine._dispatch`
  (replaced by the registry; part of 10.81).
