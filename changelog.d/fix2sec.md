### Security
- **Formula parser: SymPy no longer evaluates the expression being built (10.01):**
  the second audit of the calendar release found two more evaluation paths
  that the parser's rules did not bound (after two rounds of rules against
  particular inputs): SymPy's `exp` asks whether the factors of its
  argument are comparable and expands powers of complex sums to find out,
  and `Pow` computes the real part and the argument of the inner base of
  `(b**e)**(p/q)`; 67- and 75-character formulas such as
  `exp(pi*(((pi/4 + j/2)**64 + pi/4 + j/2)**64 + pi/4 + j/2)**64/1e30)` and
  `(((((f/4 + j/2)**64/1000 + j/2)**64/1000 + j/2)**64/1000 + j/2)**63)**(1/2)`
  ran for more than a minute. The class is fixed instead of the examples:
  every node is now built unevaluated (`evaluate=False`), and every number
  is computed by the parser itself in bounded floating point (128 bits,
  every bound checked before the operation) and stored as a SymPy float;
  within one product the numeric factors form one coefficient, within one
  sum the numbers are added and like terms collected (so `1/(f - f)` and
  `exp(f - f + 1000)` stay rejected). Formulas within every limit that took
  2.2 s to 3.3 s to parse before now take at most 0.04 s, every corpus
  string keeps its verdict, and the time of a parse grows linearly with
  the length of the formula. The exponents of nested powers now multiply to
  at most 64 (`MAX_POW`), the bound groundinsight's guard applies; the
  audit's strings are rejected within 3 ms. `evaluate_spec` and the exporter's
  passivity check evaluate the parsed expression node by node with NumPy
  (new `formula_function`) instead of `sympy.lambdify`, whose printer
  evaluates sub-expressions: 6.8 s for a 1 439-character formula within
  every limit, now 2 ms. `tests/data/formula_corpus.json` gains 16
  resource-exhaustion and 2 out-of-grammar strings, all rejected by
  groundinsight's guard as well. Tests:
  `tests/test_formula_security.py::test_nested_powers_of_audit_2_are_rejected_quickly`,
  `tests/test_formula_security.py::test_formulas_within_every_limit_parse_and_evaluate_quickly`,
  `tests/test_formula_security.py::test_parsing_and_evaluation_never_let_sympy_evaluate`,
  `tests/test_formula_security.py::test_evaluate_spec_does_not_use_sympy_code_generation`,
  `tests/test_formula_security.py::test_terms_that_cancel_within_one_sum_are_collected`.
- **Formula parser: numeric literals outside the float range raise `FormulaError` (10.01):**
  `1e-` followed by 320 nines is a valid Python float literal that rounds
  to 0.0; the parser converted its exponent to a float and raised
  `OverflowError`, which leaked through `load_bustype_json`. Every literal,
  also in an exponent, is now bounded by its exact decimal value before
  any float is formed, and no arithmetic error leaves the parser as
  anything but `FormulaError`; a BusType document whose `voltage_level`
  or `schema_version` is outside the float range raises `ValueError`, and
  so does a crafted document that is nested too deeply for Python's JSON
  reader, is not a JSON object, lacks `name`, `system_type` or
  `voltage_level`, or has samples that are not numbers (it raised
  `RecursionError`, `AttributeError` or `KeyError`). Tests:
  `tests/test_formula_security.py::test_numeric_literals_outside_the_bounds_raise_formula_error`,
  `tests/test_formula_security.py::test_load_bustype_json_reports_a_huge_literal_as_evaluate_spec_error`,
  `tests/test_formula_security.py::test_bustype_header_numbers_outside_the_float_range_raise_value_error`,
  `tests/test_io_groundinsight.py::test_crafted_bustype_documents_raise_value_error`.

### Added
- **`formula_function(expr, symbols=("f", "rho"))`** in
  `groundfield.io._formula`: NumPy function of a parsed impedance formula,
  built from the expression tree without SymPy's code generation (real
  arithmetic for real values, as `lambdify` with NumPy; SymPy's principal
  branch for roots and logarithms of negative numbers). Test:
  `tests/test_formula_security.py::test_formula_function_keeps_the_real_limits_at_singular_points`.

### Changed
- ⚠ **Form of a parsed impedance formula (10.01):** `parse_impedance_formula`
  returns an unevaluated expression whose numbers are SymPy floats of 17
  significant digits: `2*pi*f*j/1000 + 403191/50000` parses to
  `(0.0062831853071795866*I)*f + 8.0638199999999998`, `log(2)` to
  `0.69314718055994529`; symbolic `exp`, `log` and powers keep the
  structure of the text. Code that compared parsed expressions with exact
  SymPy expressions must compare values instead. Evaluated values move by
  at most 4.1e-16 relative for the 103 exporter formulas of the corpus and
  3.3e-15 for the hand-written ones (closer to a 50-digit evaluation than
  before: 1.5e-16 instead of 3.4e-15 for the worst one). Formulas that
  loaded before and are now rejected: nested powers whose exponents
  multiply to more than 64, also through sums and function calls and for
  powers of numbers, e.g. `((f**8)**8)**2` or `(f*10**40)**2` (write
  `1e40`), and exponent literals below `1e-300` such as `f**1e-400`
  (which was read as `f**0`). `evaluate_spec` gives the principal-branch
  value where `lambdify` gave `nan`, e.g. `(f - 100)**0.5` below 100 Hz.
  groundfield's exporters write neither nested powers nor roots, so
  exported formulas are not affected. Golden cases: none (bit-identical
  to the base of the change). Tests:
  `tests/test_formula_security.py::test_exponents_of_nested_powers_multiply_to_at_most_max_pow`,
  `tests/test_formula_security.py::test_parsed_expression_is_unevaluated_with_numbers_as_floats`,
  `tests/test_formula_security.py::test_formula_function_takes_the_principal_branch_where_real_arithmetic_has_none`.

### Fixed
- **Strict JSON for unknown fit quality (10.19):** a fit whose rms error
  is unknown (NaN, e.g. a hand-built fit) exported `"rms_error_Ohm": NaN`,
  which strict JSON parsers reject, and a non-finite sample did the same;
  every non-finite number of a BusType document is now `null`,
  `save_bustype_json` serialises with `allow_nan=False`, the export refuses
  a `voltage_level` that is not finite, and `fit_quality_summary` reports
  an unknown rms error as `rms=unknown`. Tests:
  `tests/test_io_groundinsight.py::test_unknown_fit_quality_and_samples_are_written_as_null`,
  `tests/test_io_groundinsight.py::test_export_refuses_a_voltage_level_that_is_not_finite`.
