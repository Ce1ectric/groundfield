### Fixed

- **Cost of the exact touching-pair rule (10.46, audit A2C-5):** the closed
  forms of coaxial segment pairs were evaluated with one Python call per
  pair, and on a straight conductor of M sub-segments all M(M−1)/2 pairs are
  coaxial: `build_inductance_matrix` took 2.4–3.2× (straight PEN, M = 100–800)
  and 7–26× (oblique PEN) the time of the 0.15.0 quadrature rule, and a solve
  of a straight 2 km PEN (400 branches, 2 frequencies) 2.2–2.6 s instead of
  1.0–1.3 s CPU. The coaxial pairs of a segment are now evaluated in one
  vectorised call — bit-identical (the 3-element dot products go through the
  same BLAS routine, the logarithms through `math.log`): 0.9–1.6× the
  quadrature time on straight chains, 2.0–3.7× on oblique ones (where the
  quadrature rule uses the cheap parallel closed form; 0.07–0.26 s for
  M = 200–400), 1.1–1.4 s for the 2 km PEN solve. No number changes (golden cases bit-identical). Tests:
  `tests/test_neumann_touching.py::test_scalar_closed_form_calls_grow_linearly`,
  `::test_touching_batch_is_bit_identical_with_the_scalar_closed_forms`,
  `::test_exact_rule_on_a_straight_chain_is_not_slower_than_twice_the_quadrature`
  (benchmark).
