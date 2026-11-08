### Added

- **`Engine.sommerfeld_assembly` (10.60):** `"exact"` (default) or
  `"tabulated"` assembly of the Sommerfeld earth-return correction of
  inductive conductors, a validated engine field (also a `create_engine`
  parameter, recorded in the provenance). The solver read it through
  `getattr` on a field that did not exist. Tests:
  `tests/test_sommerfeld_tabulated.py::test_the_assembly_is_a_validated_engine_field`,
  `::test_solve_with_an_engine_that_selects_the_tabulated_assembly`.

### Changed

- **One error type when a backend cannot take the soil model (audit A1-08,
  F09):** the backend functions called directly raise
  `BackendCapabilityError` like `Engine.solve` does: `solve_image` with a
  layered soil, `solve_image_2layer` with a homogeneous soil,
  `solve_image_nlayer` and `solve_mom` with three or more distinct layers,
  and the concrete-shell guard of the mom family. `BackendCapabilityError`
  subclasses `ValueError` and `NotImplementedError` (the types of
  `image_nlayer`, `cim`, `bem` and the shell guard in 0.15.0), but not
  `TypeError`: code that caught `TypeError` from `image_2layer` (homogeneous
  soil or layer stack) or `mom` (three or more layers) must catch
  `BackendCapabilityError` or `ValueError`. `TypeError` stays for a
  `world.soil` that is no soil model. Tests:
  `tests/test_soil_capability_errors.py`,
  `tests/test_api_smoke.py::test_solve_image_directly_rejects_non_homogeneous`.
- **Engine copies and assignments are validated (audit A1-02, A1-04):**
  `Engine.model_copy(update=...)` validates the update like the constructor
  (unknown keys raise with a suggestion; only warnings that the update
  introduces are emitted), every field is validated on assignment
  (`engine.frequencies = []` raises), and entries put into `Engine.options`
  in place are validated when they are read (`backend_options`, `solve`).
  `Engine.options` and `BackendCapabilities.soils` / `conductors` /
  `kernels` are annotated with the accepted input types (mappings, sets),
  so the documented calls type-check (audit A1-06).
- **Soil keyword typos are type errors again (audit A1-09):** the deprecated
  soil names `resistivity` / `thickness` are translated in a model validator
  instead of `validation_alias=AliasChoices(...)`, which made the pydantic
  mypy plugin accept any keyword (`HomogeneousSoil(rhoo=100.0)` passed).
  Runtime behaviour is unchanged (deprecation warning, 0.15.0 JSON loads, a
  rejected value is reported under the name it was given); type checkers now
  also report the deprecated keywords. Test: `tests/test_typing_contract.py`.
- **W605 joins the blocking lint subset (audit A1-12):** pre-commit and CI
  run `ruff check --select F,E9,B,W605`; an invalid escape sequence (a
  `SyntaxWarning` today, a `SyntaxError` under `python -W error` and in a
  future Python) no longer passes. The 35 findings of the blocking subset in
  `tests/` were cleared outside the generator / geo test modules.
- **Release tool checks the remote (audit A1-11, 10.09):** before anything
  is written, `scripts/release.py` reads the upstream branch and the remote
  tags with `git ls-remote` (no fetch, `--dry-run` changes nothing): a
  clone that is behind or has diverged from its upstream is refused, remote
  tags count for the `MICRO` of `calver` and for "the tag exists", an
  unreadable remote or a branch without upstream (in a repository with a
  remote) aborts; a repository without any remote is released with a note.
  A stale clone used to create a second, different `v2026.11.0` that only
  the push rejected. Tests: `tests/test_release_script.py`.
- **`solve_mutual_matrix` / `solve_mutual_field` assemble the inductive branch
  correction for the solved frequency only (10.60):** an engine with four
  frequencies built the correction four times per call, now once; results
  are bit-identical. Test:
  `tests/test_mutual_matrix.py::test_one_call_assembles_the_branch_correction_for_its_frequency_only`.

### Fixed

- ⚠ **A galvanic conductor with one sub-piece leaks (10.24, galvanic half):**
  a distributed conductor with `coupling_to_soil="galvanic"` whose
  `discretize_segment_length` is at least its length fell back to the
  leakage-free lumped branch. Two 2 m rods 30 m apart joined by a bare
  50 mm² conductor in 100 Ω·m: input impedance 22.74 Ω with one sub-piece
  against 5.54 Ω with two (line kernel; point kernel 22.36 Ω against
  6.91 Ω); now 5.54 Ω (point kernel 8.59 Ω, the coarsest member of its
  convergence sequence towards 5.48 Ω). The sub-piece is one midpoint
  pseudo-node between two R/2 stubs and also carries its inductance with
  `inductance_model="neumann"`. Affects only galvanic distributed conductors
  shorter than `discretize_segment_length` (no golden case; isolated ones
  keep the lumped branch). Tests:
  `tests/test_conductors_distributed.py::test_a_single_segment_galvanic_conductor_leaks`,
  `::test_a_single_segment_galvanic_conductor_keeps_its_inductance`.
- **Deprecated flat names in `Engine.model_copy` (audit A1-02):**
  `engine.model_copy(update={"sommerfeld_epsrel": 1e-2})` was silently
  ignored (the copy ran with `1e-7`; the name is a deprecated property that
  shadowed the stored value). The flat `sommerfeld_*` names are now mapped
  onto `options["mom_sommerfeld"]` with a `DeprecationWarning`, also back to
  a default value. Tests: `tests/test_engine_validation.py`.
- **Engine assignments (audit A1-04, engine part of 10.67):**
  `engine.options = {"fem": {...}}` stored a raw dict and the solve failed
  with an `AttributeError` (a misspelled option was never named);
  `engine.frequencies = []` returned an empty result and `[nan]` solved.
  Tests: `tests/test_engine_validation.py`.
- **Result files honour the umask (audit A1-05):** `save_result` wrote mode
  `0600` (`tempfile.mkstemp`); new files now get `0666` reduced by the umask
  (`0644` under `022`), an overwritten file keeps its mode. Tests:
  `tests/test_result_persistence.py::test_result_file_mode_follows_the_umask`,
  `::test_overwriting_keeps_the_mode_of_the_replaced_file`.
- **`ShallowSegmentWarning` follows the kernel (audit F08):** the advisory
  quantifies the bias of the point-form self image of a segment longer than
  four times its depth. It was also emitted for the exact line kernel (the
  image family's default), which has no such bias: a 10 m strip at 0.1 m
  depth gives 17.37863 Ω with 2 m segments and 17.37859 Ω with 0.25 m ones
  (point kernel 21.57 Ω and 17.23 Ω). It is now emitted for the point kernel
  and the `mom` family only, names the line kernel as the remedy and no
  longer claims that the `mom` family stays silent. Tests:
  `tests/test_shallow_segment_advisory.py`.
- **Stale docstrings (audit A1-07):** `create_engine` (backend-specific
  settings such as the `mom_sommerfeld` accuracy controls go into `options`,
  not the deprecated flat fields), `groundfield.solver` (the registered
  backends instead of "reserved" MoM / FEM), the engine's family overview
  (`cim` / `bem` aliases, `fem` experimental), `Engine.earth_inductive_model`
  (`carson_series` on the anti-parallel image, `sommerfeld` on the image-free
  matrix), the consumers of the series controls, and the branch block (skin
  term).
- **Control characters in docstrings (10.95):** `\rho` in a non-raw
  docstring of the two-layer solver started with a carriage return (a valid
  escape, invisible to W605). Test: `tests/test_source_escapes.py`.

### Security

- **Result files from untrusted sources (audit A1-01):** the `$ndarray`
  decoder of `FieldResult.metadata` accepted any numpy dtype string: a
  396-byte result file decoded to a 4 MiB unicode array, 413 bytes to an
  8 MiB sub-array, and object arrays loaded. It now reads only what the
  encoder writes (bool, integer, float16/32/64, complex64/128 and unicode
  dtypes), checks that the shape matches the number of items and that every
  item is a scalar of the dtype's kind before allocating, limits string
  arrays to 64 bytes per character of their items (plus 4 KiB), bounds the
  nesting depth (`MAX_METADATA_DEPTH = 100`) and raises `ValueError`
  (`pydantic.ValidationError` through `load_result`) naming the metadata key
  for every malformed tag; `load_result` reports non-UTF-8 and too deeply
  nested JSON as `ResultFormatError`. The encoder applies the same limits.
  Tests: `tests/test_result_metadata_codec.py`.
