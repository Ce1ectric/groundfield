### Fixed

- ⚠ **Wires lying beside each other: order-free, pointwise rule (10.46, audit
  A3C-1, A3C-2):** the touching-wire rule of audit round 2 switched a whole
  near-parallel pair to touching wires when the partner's end points lay
  inside a cylinder of radius r_a + r_b around the *row* segment, so the
  result depended on the order of the conductors: a V pair of 1 m and 0.5 m
  at 1.5° read 0.362 or 0.503 µH (+38.7 %), the generator's in-line
  fall-of-potential setup with lumped leads read Im Z_meas(50 Hz) =
  0.134442 or 0.126676 Ω (perfect_mirror) depending on whether the feed or
  the probe lead came first. Tilted V and X pairs whose far ends lay just
  outside the cylinder kept a filament value above the self inductance: an
  indefinite inductance matrix (minimum eigenvalue −1.3e-8 H), a two-wire
  loop with Im Z_in(1 kHz) = −0.14 mΩ (−0.51 mΩ with the skin model). The
  rule now acts pointwise inside the Neumann integral: where a point of one
  wire lies beside the other wire (its foot on the other axis falls inside
  that wire) closer than r_a + r_b, that distance is used; both one-sided
  integrals (inner integral in closed form) are averaged on end points in a
  fixed order — bit-identical for every order and direction of the
  segments, continuous in the geometry, 4e-15 against an independent mpmath
  quadrature. The in-line reading is 0.126676 Ω in both orders (5 m lead
  segments: 0.126683 Ω); 2000 random close configurations × 6
  conventions gave no indefinite matrix (round-2 state: 700). Numbers move
  for pairs that lie beside each other closer than touching except exactly
  coincident ones: partly overlapping parallel wires (+2.3e-4 for a 2 mm
  offset against the whole-pair value), V pairs leaving a junction at an
  acute angle, wires crossing at one depth, branches starting on another
  segment (crossing at 30° with r = 0.1 mm: −7.9e-5). Chain neighbours,
  bends turning by less than 90° and pairs farther apart keep their values
  bit for bit. Golden cases: none (all bit-identical). Tests:
  `tests/test_interpenetrating_wires.py::test_beside_rule_does_not_depend_on_the_order_or_direction`,
  `::test_beside_rule_matches_an_independent_quadrature`,
  `::test_matrix_does_not_depend_on_the_order_of_the_conductors`,
  `::test_inline_reading_does_not_depend_on_the_order_of_the_leads`,
  `::test_partial_inductance_matrix_stays_positive_definite`,
  `::test_random_close_configurations_are_positive_definite`,
  `::test_two_wire_loop_has_a_positive_reactance`,
  `::test_mutual_is_continuous_across_the_touching_distance`,
  `::test_chain_neighbours_and_bends_keep_their_closed_forms`.
- ⚠ **A wire inside another (10.46, audit A3C-5):** a core on the axis of a
  cable shield was coupled like two wires touching on the outside (axis
  distance r_c + r_s): loop inductance +36 % against the geometric mean
  distance, +61 % against the analytic coax, silently. Where the thinner
  wire runs inside the thicker one, the thicker radius is used — a filament
  inside a thin tube couples to it as at the tube radius: with the internal
  inductance on the core only, the loop of a 100 m core (5 mm) in a shield
  (15 mm) is 26.966 µH against the analytic coax 26.972 µH. Equal radii
  never count as one wire inside the other (cables in one trench drawn on
  one axis stay touching wires); for unequal radii the value steps where
  the thinner axis leaves the thicker wire (concentric cable vs two cables
  side by side). Golden cases: none. Tests:
  `tests/test_interpenetrating_wires.py::test_wire_inside_another_couples_at_the_radius_of_the_thicker_wire`,
  `::test_wire_inside_another_uses_the_geometric_mean_distance`,
  `::test_coincident_wires_of_equal_radius_stay_touching_wires`.
- ⚠ **`carson_series`: segment order of near-parallel pairs (10.20, audit
  round 3, evidence e13):** a near-parallel pair that is not exactly
  parallel was projected on the axis of its first segment, so reversing the
  segment order moved entries of the correction by up to 3.1 % of the
  largest one (a pair got an overlap in one order and none in the other).
  Such pairs are now projected on the bisector of the two axes from the
  mean of their mid-points (3e-17 after reversal); exactly parallel pairs
  keep their numbers bit for bit. Golden cases: none. Test:
  `tests/test_carson_validity_domain.py::test_carson_tiling_does_not_depend_on_the_segment_order`.
- ⚠ **Filaments at tiny angles (10.46, audit A3C-4):** two filaments from
  one vertex below the coaxial tolerance (4.5e-5 rad) were taken as
  overlapping coaxial filaments and fell to the 16 × 16 rule: 16 % off at
  1.5e-6 rad, 3.8 % at 1e-5 rad; just above it the vertex formula lost
  2.7e-9. Below 0.81° the vertex formula is evaluated without cancellation:
  round-off on axis-parallel geometry, at most 4e-10 in a general
  orientation. Filaments crossing at tiny angles lost the crossing-point
  mark of the close-pair rule (computed from 1 − cos², cut off at 1e-14):
  9.5e-4 at 1e-8 rad; below 1 − cos² = 1e-6 the mark now comes from a
  cross product: 1.0e-9 at 1e-8 … 3e-8 rad, 5e-11 from 1e-6 rad on. Numbers
  move for filaments (`neumann_mutual`) and segment pairs at such angles
  that are farther apart than r_a + r_b. Golden cases: none. Tests:
  `tests/test_neumann_touching.py::test_filaments_from_one_vertex_at_tiny_angles`,
  `tests/test_neumann_close_pairs.py::test_crossing_filaments_at_tiny_angles`.
- **Accuracy statements say where they hold (10.24, 10.46, 10.20, audit
  A3C-4):** measured worst cases replace statements that held only where
  they were measured. Sommerfeld correction of a long segment (composite
  rule above a quarter skin depth; audit round 2 stated 7e-5 / 1e-6 and
  "lumped conductors match their chains to 1e-4" from buried cases):
  against a refined rule, 50 Hz and 1 kHz, up to 3 km, within 5.9e-4 in
  the earth-return resistance and 2.4e-4 in the reactance (worst: a 3 km
  overhead line at 10 m over 10 Ω·m at 1 kHz; a 1 km one 4.8e-4 / 1.2e-4;
  buried in 100 Ω·m 1.7e-4), tabulated 3.3e-4 / 9.6e-5; the historic rule
  below a quarter skin depth reaches 4.8e-4, so the correction steps by
  3.5e-4 (resistance) at L = δ/4. Close pairs (stated "5e-11"): 5e-11 above
  1e-6 rad, 1.0e-9 at 1e-8 rad; pairs inside the parallel tolerance use the
  parallel closed form at the start offset (4.2e-9 for wires 1 cm apart
  tilted by 1e-10 rad, 1.7e-5 at 1 µm). Overhead wire over a buried one
  (stated "1e-9"): 4.2e-9 up to x = 300 m (1 kHz, 10 Ω·m; 1e-10 at
  50 Hz), 1.2e-7 at x = 1 km, inherited from Carson's J (1.2e-7 at a = 28,
  θ = 89.9°, QUADPACK's Fourier rule). Tests:
  `tests/test_sommerfeld_long_segments.py::test_long_segment_rule_meets_its_stated_accuracy`,
  `::test_the_step_at_a_quarter_skin_depth_is_the_stated_one`,
  `tests/test_neumann_close_pairs.py::test_crossing_filaments_at_tiny_angles`.
- **`carson_series` validity domain in the documentation (10.20, audit
  A3C-3):** the explanation page and the how-to still described the
  correction as "scaled by the geometric length of each segment pair" (the
  heuristic removed in audit round 2). They now describe the per-metre
  tiling of near-parallel horizontal pairs, the transmitted field across
  the surface, the Sommerfeld fallback with its step at 8.1°, the
  long-segment rule and the rule for wires beside each other; ADR-0004,
  -0005 and -0006 carry the corrections in their 2026.11 amendments. Test:
  `tests/test_carson_validity_domain.py::test_docs_describe_the_carson_validity_domain`.
