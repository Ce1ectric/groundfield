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
- ⚠ **`carson_series`: an overhead wire over a buried one (10.20, audit
  A2C-1):** the solver treated the buried wire of such a pair as an overhead
  wire at height |z| on the mirrored-endpoint image, whose image lands above
  ground next to the overhead wire. The mutual reactance per metre was
  35.9 % low for a lead 1 m above a PEN at 0.7 m depth (1 kHz, 100 Ω·m),
  13.5 % at 1 m horizontal offset, 3.4 % for a generator surface lead 0.1 m
  above ground, 10.3 % at 50 Hz. A perfect conductor screens the soil, so the
  anti-parallel baseline of a pair on opposite sides of the surface is now
  zero (`build_inductance_matrix(image="antiparallel")`), and its coupling is
  the field transmitted into the soil, new
  `coupling.carson.carson_mixed_mutual_correction` (Carson's J plus the
  bounded remainder of the buried wire's attenuation; Ametani et al. 2009):
  equal to an independent mpmath quadrature to 1e-9 and to the per-metre
  Sommerfeld kernel to 1e-3 at 1 kHz. Affects worlds with an overhead and a
  buried inductive conductor under `carson_series`; golden cases: none.
  Tests: `tests/test_carson_validity_domain.py::test_overhead_wire_over_buried_wire_matches_the_transmitted_field`,
  `::test_mixed_mutual_correction_matches_an_independent_quadrature`,
  `::test_mixed_pair_agrees_with_the_sommerfeld_model_per_metre`,
  `::test_antiparallel_baseline_of_a_pair_across_the_surface_is_zero`,
  `::test_every_image_convention_gives_one_impedance`.
- ⚠ **`carson_series` outside its validity domain (10.20, stream DOC-B):**
  oblique segment pairs got `sqrt(l_i l_j)·|cos φ|` times the per-metre value
  once per segment pair, which grows with the number of segments: a probe lead
  turning 8.1° against the feed lead changed the measured reactance 17.6-fold,
  two 1 km leads at 9° gave 20× the Sommerfeld coupling. Carson's per-metre
  tiling is now kept for near-parallel pairs of near-horizontal segments only;
  oblique pairs and every pair and self term of a segment inclined by more
  than 8.1° get the exact Sommerfeld correction of a homogeneous earth of the
  same conductivity (tabulated assembly). At the boundary the result steps
  from the per-metre tiling to the Sommerfeld value by the model error of the
  tiling there: 12 % for two 1 km leads 2 m apart at 1 kHz; in the DOC-B
  measurement setup (93 m leads, 50 Hz, short against the 711 m skin depth)
  Im Z_meas goes from 0.032 Ω (6.2°) to 0.018 Ω (8.6°), which is the
  `sommerfeld` value. Affects `carson_series` worlds with oblique, inclined or
  vertical inductive conductors; golden cases: none. Tests:
  `tests/test_carson_validity_domain.py::test_pairs_outside_the_domain_get_the_sommerfeld_correction`,
  `::test_oblique_coupling_does_not_grow_with_the_number_of_segments`,
  `::test_oblique_leads_couple_as_in_the_sommerfeld_model`.
- ⚠ **Wires lying side by side (10.46, streams GEN-2 / DOC-B):** two
  near-parallel segments whose axes stay closer than the sum of their radii
  over a common length (coincident leads, a service drop routed back along its
  PEN main) had no finite filament mutual inductance — the 16×16 rule stopped
  at its 1e-6 m floor: two coincident 50 m leads read 1.87 H of mutual
  inductance (+587j Ω at 50 Hz), and a two-house TN network whose drops ran
  back along the PEN main had a negative input reactance at 1 kHz
  (−0.633 Ω perfect_mirror, −0.643 Ω sommerfeld). Such pairs are now evaluated
  as touching wires at the axis distance r₁ + r₂ (in every image convention and
  in the image block), and `carson_series` gives two conductors overlapping on
  one axis the earth-return mutual of touching wires (it skipped them as
  "the same wire"). The two-house network now reads +0.244 / +0.233 / +0.250 Ω
  (perfect_mirror / sommerfeld / carson_series). Also moves the reading of the
  generator's in-line fall-of-potential setup, whose probe lead starts 1 cm
  beside the feed lead and ends on its axis: Im Z_meas at 50 Hz 0.063 → 0.054 Ω
  (sommerfeld). Golden cases: none. Tests:
  `tests/test_interpenetrating_wires.py`.
- ⚠ **`sommerfeld`: long segments (10.24, audit A2C-2):** the earth-return
  correction of a segment pair was one 16×16 Gauss–Legendre rule over the two
  segments, which cannot resolve exp(−γR) along a segment much longer than the
  skin depth — every lumped inductive conductor is one such segment. At 1 kHz
  in 100 Ω·m (skin depth 159 m) the earth-return resistance of a straight
  buried conductor was +1.7 % (1 km), +13.4 % (3 km), +34.9 % (5 km) against
  the converged chain (+0.2 / +0.6 / +1.7 % at 50 Hz), and a lumped 3 km PEN
  between two rods read +11 % series resistance, without a warning. A segment
  longer than a quarter skin depth of the most conductive layer now gets a
  composite outer rule (16 nodes per piece, at most 512 pieces, beyond that a
  `SommerfeldResolutionWarning`) and an S(ρ) table as dense as a short pair's
  (24 nodes over a km-long range left 1–2e-3); frequencies with equal piece
  counts share one pass, so every frequency still equals a single-frequency
  assembly bit for bit, and the tabulated assembly splits the same way. The
  lumped conductors now match their chains to ≤ 1e-4 (3 km at 1 kHz:
  2.78766 against 2.78773 Ω). Numbers move for every `sommerfeld` world with
  an inductive segment longer than a quarter skin depth — at 1 kHz 40 m
  (100 Ω·m), 22 m (30 Ω·m), 12.6 m (10 Ω·m); at 50 Hz 178 m (100 Ω·m) — by
  about 1e-4 just above that length up to tens of per cent for km-long lumped
  conductors; segments below it keep the historic rule bit for bit. Golden
  cases: none. Tests: `tests/test_sommerfeld_long_segments.py`.
