### Security
- **BusType impedance formulas are parsed, never executed (10.01):**
  `evaluate_spec` passed `impedance_formula` to `sympy.sympify`, which
  evaluates the text as Python code, so a crafted BusType JSON file ran
  arbitrary code or exhausted memory (`7**7**7*0+rho` builds an integer of
  about 696 000 digits, `9**9**9**9` never finishes). A new allow-list
  parser (`groundfield.io._formula`) parses the text with `ast` and builds
  the SymPy expression itself: numbers, `f`, `rho`, `j` / `I`, `pi`, `E`,
  `+ - * /`, `**` with a literal exponent (at most 64 in magnitude) and
  `sqrt` / `exp` / `log`, after Unicode NFKC normalisation, ASCII only, at
  most 10 000 characters, 2 000 syntax nodes and nesting depth 100. Every
  number stays within 1e-300 to 1e300 and powers are checked before they
  are computed, which also stops towers such as `(((10**60*f)**64)**64)**64`
  (the review prototype accepted this one after 24 to 46 s with a
  15.7-million-digit integer). `BusTypeSpec.from_dict` and
  `load_bustype_json` validate the formula on load and raise
  `EvaluateSpecError` naming the offending element (a missing formula
  raises `EvaluateSpecError` instead of `KeyError`). All 384 formulas the
  0.15.0 exporter wrote parse to identical values, so existing BusType files
  keep loading. The corpus `tests/data/formula_corpus.json` (115 accepted
  formulas with reference values, 79 rejected strings) is shared with
  groundinsight's guard. Tests:
  `tests/test_formula_security.py::test_evaluate_spec_does_not_run_code_from_a_formula`,
  `tests/test_formula_security.py::test_every_rejected_formula_fails_on_load_within_bounds`,
  `tests/test_formula_security.py::test_every_rejected_formula_fails_in_evaluate_spec_within_bounds`.

### Added
- **Passivity check before BusType export (10.40):** `to_bustype_dict`,
  `save_bustype_json`, `to_bustype` and `save_bustype_to_db` evaluate the
  exported formula from 0 Hz to twice the highest fitted frequency (dense
  linear and logarithmic grid, the samples and the midpoints between them,
  points around every pole; for the standard form over the fitted `rho`
  range) and refine the minima. `metadata["passivity"]` records `passed`,
  `min_re_Z_Ohm`, `f_at_min_Hz`, `rho_at_min_Ohm_m` and the checked ranges.
  `Re Z < 0` emits `NonPassiveFitWarning`; the new keyword
  `require_passive=True` raises `NonPassiveFitError` (a `ValueError`) and
  writes nothing. Example from the review: a 4-pole fit of passive data
  (rms 4.4 mOhm) has Re Z = -0.0194 Ohm at 5.507 Hz between the 0 Hz and
  50 Hz samples and is now flagged. Test:
  `tests/test_io_groundinsight.py::test_in_band_non_passive_vector_fit_is_flagged_on_export`.

### Changed
- ⚠ **Readable exported formulas (10.73):** vector fits are written as
  partial fractions with float coefficients, each rounded once to `decimals`
  significant digits with Python's correctly rounded formatting (identical
  on every platform), e.g. `28.4652054274 - 54.0434542796/(2*pi*f*j +
  27.5749267026)`. The exporter no longer calls `nsimplify` (exact
  fractions with integers of up to 254 digits) and no longer merges the
  partial fractions into one rational function (`sympy.simplify` produced
  terms up to `f**4`). Formulas of 2 to 10 poles shrink from 273-622 to
  126-469 characters, and the export takes 0.04-0.14 s instead of
  0.39-2.23 s (same session, passivity check included). `decimals` must be
  a positive integer; values above 17 count as 17. Numbers move only at the
  rounding level, at most half a unit in the last printed digit per
  coefficient (round trip below 1e-10 relative at the default 12 digits).
  Affected: anyone comparing formula text, e.g. diffs of BusType JSON
  files. Golden cases: none moved (`fit_rho_f_standard_sweep`
  `bustype/eval` is bit-identical). Tests:
  `tests/test_io_groundinsight.py::test_vector_fit_formula_prints_floats_not_long_rationals`,
  `tests/test_io_groundinsight.py::test_round_trip_accuracy_follows_decimals`.

### Fixed
- **Actionable `rho_at_fit` error for vector-fit exports (10.84):** the
  documented `save_bustype_json(fit, ...)` call for a vector fit raised
  "rho_at_fit is required" without saying what to pass. `rho_at_fit` stays
  required for a `VectorFitResult` (the BusType records the soil
  resistivity it belongs to); all four export functions now raise the same
  `ValueError` naming the function, an example call and the layered-soil
  convention, before a file is written or groundinsight is imported, and
  `rho_at_fit` must be positive and finite. Test:
  `tests/test_io_groundinsight.py::test_vector_fit_without_rho_at_fit_raises_an_actionable_error`.
- **BusType provenance stamp:** `metadata["source"]` read
  `groundfield unknown` when the package ran from a source checkout (and
  went stale under an editable install); it now uses
  `groundfield.__version__` (review item R5-12). Test:
  `tests/test_io_groundinsight.py::test_bustype_metadata_records_the_running_groundfield_version`.

### Internal
- The groundinsight round-trip test for vector fits asserts 1e-9 (ADR-0008)
  instead of 1e-3; a new test compares `evaluate_spec` with groundinsight's
  `compute_impedance` to 1e-9 and skips without groundinsight
  (`tests/test_io_groundinsight.py::test_evaluate_spec_matches_groundinsight_evaluation`).
  The bridge tests no longer skip the whole module when groundinsight is
  missing.
