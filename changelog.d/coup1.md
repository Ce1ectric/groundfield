### Added

- **Image conventions for the inductive stack (10.20 preparation, F32):**
  `build_inductance_matrix(..., image="none" | "additive" | "antiparallel")`
  (the legacy `use_image=True/False` maps to `"additive"`/`"none"`), and the
  same `image=` argument on `build_carson_correction_matrix`,
  `build_sommerfeld_correction_matrix` and the Sommerfeld pair integrals: a
  correction states which image the inductance matrix it is added to carries,
  so `j·ω·L(image) + ΔZ(image)` is the same physical matrix for every
  convention. Carson's correction is documented as defined against the
  anti-parallel image; the defaults reproduce the numbers of 0.15.0 for
  `carson_series` and `perfect_mirror`. The solver switches `carson_series` to
  the anti-parallel baseline in a later step of this release.
  Tests: `tests/test_sommerfeld_accuracy.py::test_total_impedance_does_not_depend_on_the_image_convention`,
  `tests/test_sommerfeld_accuracy.py::test_carson_on_the_antiparallel_baseline_matches_carson`.
- **Multi-frequency earth-return builders (10.60, 10.63):**
  `build_sommerfeld_correction_matrices(seg_endpoints, radii, omegas=..., earth=...)`
  and `build_carson_correction_matrices(...)` return all frequencies at once;
  geometry, λ grid and the Bessel table are built once per segment pair. Each
  frequency is bit-identical to the single-frequency builder. On the inductive
  PEN branches of a 2-house street network (34 branches, 11 frequencies) the
  Sommerfeld assembly is 4.8–6.7× faster than eleven single-frequency calls.
  Carson's `P`, `Q` are memoised on `(a, θ)` (bit-identical; 9130 → 2321
  evaluations and 3.9× less CPU time on a 10-house network).
  Tests: `tests/test_sommerfeld_accuracy.py::test_multifrequency_matrices_are_bit_identical`,
  `tests/test_carson_pq.py::test_multifrequency_carson_matrices_are_bit_identical_and_cached`.
- **Exact touching segment pairs, opt-in (10.46):** `touching="exact"` on
  `neumann_mutual` and `build_inductance_matrix` uses Grover's closed forms for
  segments that share an end point (chain neighbours, bends) or lie on one
  axis; the 16×16 Gauss–Legendre rule is −0.3 % … +0.8 % off for such pairs, and
  a straight wire split into `n` segments now keeps the self inductance of the
  unsplit wire exactly. The default stays `"quadrature"` until the solver
  switches. Test: `tests/test_neumann_touching.py::test_chain_sum_is_refinement_invariant`.
- **Per-segment internal inductance (10.46):** `include_internal` of
  `build_inductance_matrix` accepts a boolean array. The package-wide convention
  (DC internal inductance μ₀/8π per metre included; compare with references at
  `GMR = a·e^(−1/4)`) is documented in `groundfield.coupling.inductance`, the
  Carson and Pollaczek references. Test:
  `tests/test_coupling_conventions.py::test_internal_inductance_convention_matches_references`.
- **Opt-in skin effect (10.37):** `Conductor(resistance_model="skin")` with
  `Conductor.internal_impedance_per_length(f)` and
  `Conductor.skin_correction_per_length(f)` (the correction relative to the DC
  convention), based on the new
  `groundfield.conductors.solid_round_internal_impedance` (exact Bessel
  solution): 150 mm² Al has R_ac/R_dc = 1.231 at 550 Hz and 1.550 at 1 kHz,
  internal inductance 0.740× at 1 kHz. The default `"dc"` changes nothing; the
  solver uses the correction in a later step of this release. Test:
  `tests/test_skin_effect.py::test_150mm2_aluminium_resistance_ratio`.
- **Close, interpenetrating and crossing wires (10.44):**
  `groundfield.conductors.find_close_wires(endpoints, radii, names, ...)`,
  `find_close_conductors(conductors, ...)` and `segment_distance` report wire
  pairs of different owners closer than `rel_threshold·(a_i + a_j)`, classified
  as crossing / interpenetrating / close (junctions at shared end points are
  skipped by default). Test: `tests/test_conductor_proximity.py::test_parallel_rods`.
- **`LayeredEarth.interface_depths` and `LayeredEarth.layer_of(depth)`.**

### Changed

- ⚠ **`references.pollaczek` is exact for two layers (10.45):** the self and
  mutual impedances use the exact buried-conductor amplitude (lower-interface
  term included) and the `K0(γ·GMR)` head instead of `ln(2h/GMR)`. R′ moves by
  +0.15 % (1 m) and +1.0 % (3 m) at 30 Ω·m, 1 kHz (homogeneous), by up to +0.5 %
  in band near a layer interface, and by +40 % at 1 MHz (Tsiamitros Case V); X′
  by less than 0.02 % in band. Conductors below the first interface and pairs in
  different layers are now supported exactly. Affects users of
  `pollaczek_self_impedance`, `pollaczek_mutual_impedance`,
  `loop_impedance_conductor_earth` (the AP1 catalogue builds its own z′ table
  from these). Golden cases: none.
- **Carson P, Q (10.46):** the asymptote is used only for a > 15 (was 5), where
  it is accurate to 1e-7; moves `carson_series` only for pairs with a > 5
  (e.g. more than 560 m apart at 1 kHz in 100 Ω·m). Golden cases: none.

### Fixed

- ⚠ **Sommerfeld λ grid resolves distant pairs (10.36):** the logarithmic
  region ended at max(p_skin, 1/z) whatever the pair distance, so J0 was
  unresolved there, and the linear region was capped at 4096 panels without a
  check: pair entries were off by −24 % (Re, 200 m) and +274 % (Re, 400 m),
  +2.35 % (Im, 400 m). The grid now follows the F14 policy of `layered_green`
  (log region below π/(4ρ_max), 16 nodes per J0 period, graded panels at the
  skin and layer scales) and raises `SommerfeldResolutionWarning` instead of
  truncating; entries agree with an independent quadrature to 1.2e-5 (R′ part)
  and 2.7e-6 (X′ part) for pairs 50–400 m apart. Golden cases:
  `cond_galvanic_neumann_sommerfeld`, `defaults_mixed_cluster_inductive`
  (≤ 1.8e-9), `cond_isolated_pen_sommerfeld_two_layer` (≤ 1.9e-5). Test:
  `tests/test_sommerfeld_accuracy.py::test_distant_pair_entry_matches_independent_quadrature`.
- ⚠ **Sommerfeld kernel exact for n ≥ 3 layers and below the first interface
  (10.35):** the buried kernel used the single air-image form outside the
  two-layer/layer-1 case; a three-layer earth with ρ₂ = ρ₃ gave +17 … +25 % X′
  against the identical two-layer earth. The kernel now uses generalised
  reflection coefficients (any number of layers, any layer) and the transmitted
  amplitude for pairs in different media. Affects `sommerfeld` worlds on
  `MultiLayerSoil` and conductors deeper than `h_1`. Golden cases: none. Tests:
  `tests/test_sommerfeld_accuracy.py::test_three_layer_duplicate_equals_two_layer`,
  `tests/test_sommerfeld_accuracy.py::test_conductor_below_first_interface`.
- ⚠ **`earth_return_correction_layered` had the wrong sign (10.46, F50):** the
  reflected term was +8.739e-3 + 6.484e-3j instead of −6.346e-3 − 4.580e-3j
  (earth 100/10 Ω·m, h₁ = 3 m, 1 kHz, z = z′ = 0.6 m); it now uses the exact
  layered kernel. Test:
  `tests/test_sommerfeld_accuracy.py::test_layered_point_kernel_has_the_exact_sign`.
- ⚠ **Exact removal of the additive image in the Sommerfeld correction
  (10.46):** the image was removed as −1/R′ integrated by the 16×16 outer rule
  with the direct orientation, so vertical segments kept twice the image in the
  σ → 0 limit and 50 m segments at 0.7 m lost 2 % of it. The correction now
  subtracts the very image block the inductance matrix added. Golden cases: see
  the λ-grid entry. Test:
  `tests/test_sommerfeld_accuracy.py::test_transparent_earth_cancels_the_additive_mirror`.
- **Segments crossing the soil surface (10.46, F52):** an inductive segment
  whose end points lie on both sides of z = 0 was dispatched by its midpoint
  (`World.solve` solved silently) or rejected as "lying in the soil-surface
  plane". With an image convention it now raises a `ValueError` that says the
  segment crosses the surface (its image intersects it); with `image="none"` the
  Sommerfeld pair integrals split such segments at the surface and at layer
  interfaces. Tests:
  `tests/test_sommerfeld_accuracy.py::test_surface_crossing_segment_is_rejected_with_the_right_message`,
  `tests/test_sommerfeld_accuracy.py::test_world_solve_rejects_surface_crossing_inductive_conductor`.
- **Carson's Q asymptote had the wrong sign of its a⁻⁷ term (10.46):** +45, not
  −45; together with the boundary move the large-a regime is continuous to
  1e-7 (it jumped by up to 9.3e-4 at a = 5). Tests:
  `tests/test_carson_pq.py::test_q_asymptote_a7_term_has_positive_sign`,
  `tests/test_carson_pq.py::test_large_a_regime_is_continuous`.
- **Conductor input validation (10.43, 10.67):** `cross_section <= 0` or
  non-finite, non-finite `resistivity` / `wire_radius` / end points raise a
  `ValueError` naming the parameter (they failed late with `ZeroDivisionError`
  or `LinAlgError`, or gave an active element); a cross section above 1e-2 m²
  warns with `CrossSectionUnitWarning` (`cross_section=50` meant as mm² was a
  silent 6.7 nΩ short); an ideal conductor with `discretize_segment_length`
  raises (it was `LinAlgError: Singular matrix` at solve time). Tests:
  `tests/test_conductor_validation.py::test_nonpositive_or_nonfinite_cross_section_raises`,
  `tests/test_conductor_validation.py::test_cross_section_in_mm2_warns`,
  `tests/test_conductor_validation.py::test_ideal_distributed_conductor_raises_with_parameter_name`.
- **Documentation in the coupling package:** English prose in
  `coupling/layered_green.py` (10.95); the Carson layered-soil warning points to
  `earth_inductive_model="sommerfeld"` (it named `backend='mom_sommerfeld'`);
  the internal-inductance statement of `references.earth_return` ("external
  inductance only") is corrected. Test:
  `tests/test_coupling_conventions.py::test_layered_green_prose_is_english`.
