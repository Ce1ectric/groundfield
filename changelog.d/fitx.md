### Security
- **Formula parser cost is bounded for sums, products and powers (10.01; audit 1, H1, H2, L1, L2):**
  formulas inside every documented limit could still cost minutes (times
  below: fresh process on a shared machine). A sum of fractions with
  distinct 15-digit prime denominators, `((b+2)/b)**64` per term, took 22 s
  (120 terms) and 105 s (222 terms) before it was rejected, because the
  bound was checked after SymPy had built the common denominator; the least
  common multiple of the fraction coefficients is now accumulated term by
  term before the sum is built, and the formula is rejected at the second
  term (0.05 s). Accepted products such as 332 logarithms (13 s) or 111
  vanishing log sums (55 s) re-evaluated the growing product with `evalf`
  after every factor (2 802 and 2 857 `evalf` calls); values now come from a
  numeric shadow evaluation (mpmath, 128 bits) computed once per node, and
  chains of 32 or more factors are multiplied pairwise (2.1 s and 0.8 s;
  176 and 110 `evalf` calls, all SymPy's own). `(exp(600)*f)**64` was
  accepted as `f**64*exp(38400)`: the power bound now includes the
  symbol-free factor of a product base, and
  every symbol-free sub-expression of the result is bounded
  (`f*exp(600)*exp(600)` is rejected). The documented exponent `(-3/2)`
  (Python reads `(-3)/2`) is accepted, so `format_impedance_formula` output
  such as `f**(-3/2)` parses. A deterministic budget, `MAX_WORK` = 200 000
  bookkeeping steps, caps every parse. Exported formulas parse to the same
  values as before. `tests/data/formula_corpus.json` gains 4 accepted and 6
  rejected audit strings for groundinsight's guard. Tests:
  `tests/test_formula_security.py::test_sum_of_fractions_with_distinct_denominators_is_rejected_quickly`,
  `tests/test_formula_security.py::test_long_product_chains_parse_in_bounded_time`,
  `tests/test_formula_security.py::test_symbol_free_factor_of_a_power_base_is_bounded`,
  `tests/test_formula_security.py::test_negative_rational_literal_exponents_are_accepted`.

### Added
- **`touch_voltage_envelope(..., interior=...)` (10.21):** samples the inside
  of closed loops, the analogue of the IEEE Std 80 mesh voltage (the worst
  touch voltage inside a mesh, near its centre). `interior=None` (default):
  the mesh centres of `mesh` / `grid_mesh` electrodes and the centre of a
  `ring`, i.e. loops in open ground; a closed `polyline`, the generators'
  building-outline foundation, is not sampled because the building stands
  inside. `interior=True` adds the point inside a closed polyline farthest
  from its wires (non-convex outlines stay inside); `interior=False` gives
  the offset curve only. The default cannot see buildings: pass
  `interior=False` for an axis-aligned generated foundation (`grid_mesh`) or
  a ring around a station housing. It replaces the `mesh_centres=` keyword
  added earlier in this release; `anchor="connection_point",
  interior=False` reproduces v0.15.0. Test:
  `tests/test_touch_voltage_envelope.py::test_ring_envelope_samples_the_ring_centre_by_default`.
- **Fit metadata in BusType documents (10.19, 10.27):** standard-form exports
  record all coefficients `k0` ... `k5`, `fit_quality.max_relative`, the
  `weights` and the fitted impedance `quantity` (`"input"`, `"cluster"`,
  `"electrode"`, or `null` for raw samples); vector-fit exports record
  `fit_quality.rms_relative` and `quantity`. `fit_quality_summary` names
  them. Test:
  `tests/test_io_groundinsight.py::test_fit_metadata_records_which_impedance_was_fitted`.
- **`MAX_WORK`** in `groundfield.io._formula` (see Security).

### Changed
- **One formula renderer (10.73):** `fit_to_sympy` keeps its signature but
  returns the partial fractions of the fit (one term per real pole or
  conjugate pole pair) instead of passing them through `sympy.simplify`,
  which merged them into one rational function: 6 of reviewer R5's 98 vector
  fits had a term above second order (`s**4` over `s**3` for a pole pair, a
  real pole and `L_inf`), now none. `fit_to_sympy`, `fit_to_sympy_standard`
  and the groundinsight exporter share one builder (one pole-pairing rule)
  and one rounding rule: every coefficient is rounded once to `decimals`
  significant digits (`decimals` must be a positive integer, also a NumPy
  integer, and every other value raises `ValueError`; above 17 counts as
  17). Accuracy is unchanged: worst deviation from `fit.evaluate` over R5's
  fits 4.0e-6 at 6 digits and 4.8e-12 at 12. Exported vector-fit formulas
  are unchanged character for character; exported standard-form formulas
  keep every coefficient and value, but a complex coefficient now keeps its
  own sign (`- f*(12.5592 - 1338.31*j)` is written `+ f*(-12.5592 +
  1338.31*j)`; 23 of R5's 31 standard fits). Test:
  `tests/test_fit_formula_renderer.py::test_fit_to_sympy_keeps_one_term_per_pole_or_pole_pair`.
- ⚠ **One passivity check (10.40):** the BusType export runs `check_passivity`
  (on the rounded formula it writes) instead of its own implementation, so
  both use one band, 0 Hz to twice the highest fitted frequency (the default
  of `check_passivity` was the highest fitted frequency: a fit with
  Re Z = -1.0 Ohm at 1.5 kHz, passive at every sample up to 1 kHz, passed
  `check_passivity` but failed the export), the pole-adapted grid with the
  midpoints between samples, an exact corner check for the standard form, a
  tolerance of 1e-9 max|Z|, and non-finite values fail (a pole at s = 0
  passed `check_passivity`). `PassivityReport.as_dict()` is the strict-JSON
  `metadata["passivity"]` block (`passed`, `min_re_Z_Ohm` or `null`,
  `f_at_min_Hz`, `rho_at_min_Ohm_m`, `f_checked_Hz`, `rho_checked_Ohm_m`,
  `n_points`); `PassivityReport` gains `n_points`. Affected: callers of
  `check_passivity` without `f_max`, whose verdict now covers twice the
  band (fits that passed may fail); impedances do not move. Golden cases:
  none. Test:
  `tests/test_passivity.py::test_violation_between_the_highest_sample_and_twice_it_is_found`.
- **`rho_at_fit` for layered soil (10.84):** the error message and the
  export docstrings state one convention, the resistivity of the upper
  layer, `rho_1` (the parameter of the standard form's rho_1 sweep), so
  vector-fit and standard-form BusTypes pair with the same groundinsight
  `Bus`. Test:
  `tests/test_io_groundinsight.py::test_rho_at_fit_convention_for_layered_soil_is_the_upper_layer`.
- **CSV frequency column `frequency_hz` (10.69):** the column the electrode and
  cluster tables gain in this release, and the header keys `frequency_hz` /
  `frequencies_hz`, use the lower-case unit suffix of the spec layer and of
  `FieldResult.to_dataframe`. The potential path keeps its v0.15.0 column
  `frequency_Hz`. Test:
  `tests/test_io_csv.py::test_new_frequency_column_uses_the_lower_case_unit_suffix`.
- **Footprint from the electrode geometry (10.79):** the touch-voltage
  footprint reads the wires from `Electrode.wire_polylines()` instead of a
  per-kind re-implementation (a ring keeps its exact circle). Foot positions
  are unchanged up to round-off (at most 1.4e-14 relative in the touch
  voltage, 2.3e-10 at UTM coordinates). Test:
  `tests/test_touch_voltage_footprint.py::test_mesh_centres_lie_midway_between_the_discretiser_wires`.

### Fixed
- ⚠ **Touch voltage of rotated foundations (10.21; audit 1, H4):** the
  generators build a foundation that is not axis-aligned from bonded wires
  `<name>`, `<name>_w1`, ... (10.39), and with 10.54 every OSM house
  foundation is rotated. The footprint came from the named edge wire only:
  for a 10 m x 10 m 2 x 2 mesh at 30 deg (100 Ohm m, 100 A) `touch_voltage`
  read 214.4 V on +x / -x and 93.9 V on +y (feet 1 m inside the building),
  31 of 72 envelope feet lay inside the foundation and no cell centre was
  sampled. The footprint is now every wire of the foundation (from any of its
  names, within its galvanic cluster) and the envelope samples the cells of
  the rotated grid: 161.5 V on every side (167.5 V at 0 deg; one uniform
  current per wire instead of per grid), no foot inside. Affected: touch
  voltages at rotated foundations, i.e. OSM-based layouts. Golden cases:
  none. Test:
  `tests/test_touch_voltage_footprint.py::test_touch_voltage_of_a_rotated_foundation_matches_the_unrotated_one`.
- **`touch_voltage_spectrum` counted a frequency twice (10.52; audit 1, L4):**
  a frequency carried by two results entered the RMS twice (701.6 V instead of
  496.1 V); it now raises `ValueError` naming the frequency. Test:
  `tests/test_safety.py::test_touch_voltage_spectrum_rejects_a_frequency_given_twice`.
- **Strict-JSON fit metadata (10.27):** hand-built standard fits wrote
  `max_relative` as NaN, which is not JSON; non-finite quality figures are
  now `null`. Test:
  `tests/test_io_groundinsight.py::test_metadata_of_hand_built_fits_is_strict_json`.
- ⚠ **Envelope of a ring includes its centre (10.21):** a consequence of
  `interior=None` above; where the centre dominates, the envelope maximum
  rises (ring of 10 m radius at 0.8 m, 100 Ohm m, 100 A: 139.4 V on the
  offset curve, 182.6 V in the centre; a 4 m ring's centre, 346.0 V
  against 360.2 V, does not).
  Affected: envelopes of ring electrodes. Golden cases: none. Test:
  `tests/test_touch_voltage_envelope.py::test_ring_envelope_samples_the_ring_centre_by_default`.

### Deprecated
- Nothing.
