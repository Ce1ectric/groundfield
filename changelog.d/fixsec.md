### Security
- **Formula parser: SymPy no longer factorises numbers or rewrites `exp(c*log(x))` while parsing (10.01):**
  strings inside the allow-list grammar still kept SymPy busy for minutes.
  SymPy simplifies a fractional power of an exact number by factorising the
  number and keeps a radicand of up to `q - 1` times its digits:
  `(N)**(63/64)` with the 300-digit `N = 10**300 - 1` (also written
  `(f*N)**(63/64)`, `(-N*f*rho)**(63/64)` or `(N/8)**(63/64)`) keeps a
  radicand of about 18 800 digits and tests a 16 280-digit factor of it for
  primality, and `18**(10**30/(10**30 + 1))` needs `2**(10**30)`. It also
  rewrites `exp(c*log(x))` as `x**c`, whose exponent is no literal:
  `exp(log(N)*63/64)` factorises like the power, `exp(log(2*f)*10**12)`
  builds `2**(10**12)`. Of 19 such strings (four reported by the
  groundinsight stream), 15 did not finish within 20 s, three took 1.8 to
  2.6 s and one was accepted. Now no exact number is left under a fractional
  power: a fractional power (`x**(p/q)`, `sqrt`) of a symbol-free base is
  evaluated numerically, and the rational coefficient `c` of a product base
  is taken out as the float `|c|**(p/q)` (17 significant digits, principal
  branch); a powered sum with a float coefficient gets float coefficients
  throughout (SymPy 1.13, which the dependency range allows, takes the
  largest coefficient of such a sum out of the power: `(2.0*f + N)**(63/64)`
  did not finish within 10 s there); an exact number under a fractional
  power needs `q <= 64` and digits times `q <= 300` (`MAX_ROOT_DENOMINATOR`,
  `MAX_RADICAND_DIGITS` in `groundfield.io._formula`, the limits of
  groundinsight's guard); `log()` inside `exp()` is rejected before anything
  is built; `log` of a symbol-free argument other than a positive rational
  or a float is evaluated numerically (SymPy's `log(r + r*j)` and the square
  root of `r + r*j/3` factorised numbers of about 1 800 digits: 0.7 to 1.4 s
  for 900-digit fractions `r`). Each of the 19 strings is now rejected in
  at most 3 ms. Exported formulas parse to identical expressions (all 119
  accepted corpus formulas and 387 exporter outputs of reviewer R5's
  generator). `tests/data/formula_corpus.json` gains 18 resource-exhaustion
  strings and one out-of-grammar string, all rejected by groundinsight's
  guard as well; the corpus test now parses every rejected string first in a
  worker process with a 20 s deadline and a 1 GiB address-space limit, so a
  regression can neither hang the test run nor exhaust the machine. Tests:
  `tests/test_formula_security.py::test_every_rejected_formula_is_rejected_quickly_in_a_worker_process`,
  `tests/test_formula_security.py::test_fractional_powers_of_large_exact_numbers_are_rejected_before_sympy`,
  `tests/test_formula_security.py::test_fractional_powers_of_exact_numbers_are_evaluated_in_floating_point`,
  `tests/test_formula_security.py::test_log_inside_exp_is_rejected`,
  `tests/test_formula_security.py::test_sum_with_a_float_coefficient_keeps_no_exact_coefficient_under_a_root`,
  `tests/test_formula_security.py::test_complex_numbers_with_large_fractions_are_evaluated_quickly`.

### Added
- **`MAX_ROOT_DENOMINATOR`** (64) and **`MAX_RADICAND_DIGITS`** (300) in
  `groundfield.io._formula` (see Security).

### Changed
- ⚠ **Hand-written formulas with roots or logarithms of numbers (10.01):**
  `sqrt(2)`, `3**(1/3)`, `(2*pi*f)**(1/2)`, `log(-2)` or `log(2*pi)` now
  parse to floats of 17 significant digits instead of exact radicals and
  logarithms (`log` of a positive rational such as `log(2)` stays exact);
  evaluated values move by at most 3.6e-16 relative (2 units in the last
  place) in 17 sample formulas. Formulas that loaded before and are now
  rejected: `log` anywhere inside `exp` (e.g. `exp(log(f)*3)`, which SymPy
  turned into `f**3`), and fractional powers of exact numbers beyond the
  new limits, e.g. `2**(1/65)`, `(2*f)**(1/1000)`, a 5-digit integer under
  `**(63/64)` or the square root of a 151-digit integer; writing the number
  as a float (`2.0**(1/65)`) keeps such a formula valid. Affects
  hand-written BusType formulas only: groundfield's exporters write
  neither roots nor `exp`/`log` (checked for every exporter formula of the
  corpus). Golden cases: none (bit-identical). Test:
  `tests/test_formula_security.py::test_fractional_power_limits_for_exact_numbers`.
