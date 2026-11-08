### Added
- **Reading of a simulated earth tester (10.17):**
  `groundfield.generators.measured_impedance(result, probe=None, source=None)`
  returns the three-pole reading Z_meas = (phi_E - phi_S) / I per frequency:
  E is the earthing system the measurement current is fed into (the source's
  `attached_to` cluster), S the voltage terminal (`meas_S` of a network built
  with a probe lead, else the probe electrode). The two-pole reading (E-H) is
  `result.input_impedance()`. New constants `MEASUREMENT_H_TERMINAL = "meas_H"`
  and `MEASUREMENT_S_TERMINAL = "meas_S"`; `PlannedMeasurement.terminals` and
  `NetworkPlan.all_junctions` carry the instrument terminals of a plan.
  Test: `tests/test_measurement_loop.py`.
- **`GroundingSystemSpec.plan_site(..., outline=)`:** shapes the resolved
  foundations after the footprint once the random draws are made
  (`"spec"` by default). Test:
  `tests/test_footprint_foundations.py::test_footprint_outline_does_not_shift_the_random_draws`.
- **`groundfield.generators.grounding.MIN_FOUNDATION_SIDE_M` (1 m):** shortest
  side of a foundation taken from a footprint's oriented bounding rectangle.

### Changed
- ⚠ **Measurement leads end on the instrument (10.17):** in
  `TnNetworkGenerator` measurement set-ups the feed lead runs from the
  junction `meas_H` to the auxiliary electrode and the probe lead from
  `meas_S` to the probe (both junctions at the substation position, at the
  lead height); the source drives the test current between the substation and
  `meas_H` (`return_to`; the auxiliary electrode itself without a feed lead).
  With both leads, `meas_S` lies the touching distance (sum of the two wire
  radii) beside `meas_H`, so an in-line probe lead runs beside the feed lead
  instead of through it. Numbers of every measurement world with leads change
  completely: the old probe lead bonded the probe to the substation (reading
  2.8e-4 instead of 1.6096 ohm in a 3-house network) and the old feed lead
  closed the loop metallically (99.93 % of the current never reached the
  soil). Golden cases: `gen_tn_measurement_overhead_leads`
  (auxiliary rod -1 A instead of +0.024 A, network potentials -1.8 %).
- ⚠ **One concrete-shell mechanism (10.12, 10.13, 10.14):** both
  `concrete_model` values put the Sunde coefficient
  C = rho_c / (2 pi) ln(r_b / r_a) and the radius r_b = r_a + t on every encased
  (perimeter) wire; nothing is written to `World.concrete_shell_corrections`
  any more and PEN service drops keep their geometric resistance. `"lumped"`
  keeps one grid-mesh owner for an axis-aligned ring foundation (exactly the
  lumped resistance C / L_perim), `"distributed"` uses one owner per wire;
  polygon, rotated and mesh foundations are the same for both. Mesh braces are
  not encased (bare radius, C = 0); this also moves `"distributed"` mesh
  foundations, whose braces had the inflated radius (4-house network, 2 x 2
  mesh: +2.6 / +7.8 / +12 % at rho_c = 150 / 1000 / 5000 Ohm m). The shell now reaches `solve_mutual_matrix`
  / `solve_mutual_field` (the AP1 Z_G path), `Engine.solve` of `build_at`
  worlds, the layout builder and substation / cabinet foundations. AP1-like
  site (rho_c = 100 Ohm m, 5 cm, 12 polygon foundations): house Z_G diagonal
  +42 / +19 / +2.8 % at rho_1 = 30 / 100 / 1000 Ohm m against v0.15.0, network
  input impedance +13.6 / +8.8 / +2.2 %; with rho_c = rho_soil the shell is
  transparent (0.0 %). Affected: every world with `concrete_rho_ohm_m` set.
  Golden cases: `mutual_foundations_lumped_concrete` (Z_G diagonal +42.2 %).
- ⚠ **PEN cable joints are junction nodes (10.38):** the layout builder's cable
  corners and tap points are leakage-free `Junction` nodes (reported in
  `result.metadata["junction_potentials"]`) instead of 5 cm driven rods, which
  leaked 0.24 to 0.53 % of the test current at full foundation penetration and
  3.7 % at 5 % penetration (substation potential +0.07 ... +0.33 % and +3.5 %).
  The joints are no electrodes any more. Golden cases: `gen_layout_builder`
  (+0.33 % potentials, +0.43 ... +0.57 % currents).
- ⚠ **`PenConfig.depth_m` is honoured:** every PEN section and service drop of
  `TnNetworkGenerator` runs horizontally at z = `depth_m` between its anchors
  (it ran straight between the electrode connection points, e.g. from the
  substation ring at 0.6 m to a cabinet rod head at 0 m); `depth_m` must be
  > 0. 10-house network: resistive PEN -2.7e-7, bare galvanic PEN -1.4 %,
  Neumann PEN at 1 kHz X 0.736 -> 0.167 ohm (the old sections ended in the
  surface plane, where the perfect-mirror image meets the wire). Golden cases:
  `gen_tn_default` (<= 3.5e-7).
- **Layout reading uses the solved test current:**
  `LvNetworkLayout.measured_grounding_impedance` and `verify_current_balance`
  take `source_magnitude_a=None` by default and read the current of the solved
  substation source (was 1 A unless repeated by the caller). Test:
  `tests/test_measurement_loop.py::test_layout_reading_uses_the_solved_source_current`.
- **Overpass response cap 64 -> 16 MiB:** `DEFAULT_MAX_RESPONSE_BYTES` also
  bounds the decoding memory (6x the body for building payloads, 25x for a
  body of empty objects); the largest AP1 payload is 3.0 MiB. Pass
  `max_response_bytes` for wider areas. Test:
  `tests/test_overpass_ingest.py::test_default_response_cap_bounds_the_decoded_size`.
- **Naming rule of the spec layer (10.69):** spec fields carry lower-case unit
  suffixes; the soil specs are the deliberate exception and mirror the
  literature symbols of their soil model (`rho`, `rho_1`, `rho_2`, `h_1`, `h`).
  Test: `tests/test_generator_naming.py::test_soil_specs_mirror_their_soil_models`.

### Fixed
- **Measurement leads short-circuited the instrument (10.17):** resistive leads
  now leave the reading unchanged (1.609633 ohm with and without leads; the
  feed lead adds its resistance to the two-pole reading only). Tests:
  `tests/test_measurement_loop.py::test_resistive_leads_do_not_change_the_reading`,
  `::test_feed_lead_adds_its_resistance_to_the_loop_impedance`,
  `::test_two_terminal_reading_is_the_superposition_of_one_terminal_solves`,
  `::test_in_line_leads_lie_side_by_side`.
- **Lumped concrete shell dropped outside TN service drops (10.12):** Test:
  `tests/test_concrete_one_mechanism.py::test_lumped_polygon_foundation_equals_the_distributed_one`,
  `::test_lumped_shell_resistivity_moves_the_mutual_matrix`.
- **Rods and braces leaked through the foundation's shell; substation and
  cabinet shells dropped (10.13):** lumped vs distributed +1.4 / +9.7 / +26.8 %
  (foundation + rod) and +9.6 / +56 / +148 % (mesh foundation) in v0.15.0, now
  within 2 % and identical. Tests:
  `tests/test_concrete_one_mechanism.py::test_mixed_site_lumped_equals_distributed`,
  `::test_mesh_braces_are_not_encased`, `::test_substation_and_cabinet_shells_are_consumed`.
- **Layout builder ignored the concrete shell (10.14, F08):** Test:
  `tests/test_concrete_one_mechanism.py::test_layout_builder_honours_the_shell`.
- **PEN junction rods leaked current (10.38):** Tests:
  `tests/test_pen_junctions.py::test_the_earthing_system_carries_the_whole_test_current`,
  `::test_a_resistive_cable_through_joints_equals_one_series_resistor`.
- **`PenConfig.depth_m` had no effect:** Test: `tests/test_pen_depth.py`.
- **Malformed Overpass payloads (10.05; audit M1):** elements with wrong types
  crashed the parser with `AttributeError` / `TypeError` after the payload had
  been cached (every later call crashed from the cache); NaN / 1e308 / lat 95
  vertices gave NaN-area footprints that passed `min_area_m2`. Every element
  and vertex is validated before caching and parsing (`OverpassError`), and a
  cached payload that fails the check is fetched again. Tests:
  `tests/test_overpass_ingest.py::test_malformed_elements_raise_overpass_error`,
  `::test_non_finite_or_out_of_range_vertices_are_rejected`,
  `::test_malformed_payload_is_neither_cached_nor_parsed`,
  `::test_malformed_cache_file_is_refetched`.
- **Footprint outline shifted the random draws (10.54; audit M2):** with a
  stochastic foundation size the `"ombr"` default skipped the size draw and
  moved optional rods to other houses (6 -> 8 rods). Test:
  `tests/test_footprint_foundations.py::test_footprint_outline_does_not_shift_the_random_draws`.
- **Degenerate footprints broke `to_world` (10.54; audit L5):** a 10 m x 5 mm
  outline is widened to a 1 m foundation with a warning. Test:
  `tests/test_footprint_foundations.py::test_a_degenerate_footprint_gets_a_minimum_foundation`.
- **Layouts pickled with v0.15.0 (10.69; audit L6):** `to_world` raised
  `AttributeError: 'cabinet_placements'`; `kvs_placements` is migrated on
  unpickling. Test:
  `tests/test_generator_naming.py::test_layout_pickled_with_v0150_still_builds`.
- **Inert permittivity of soil specs (10.51; audit A1-10):** a non-default
  `relative_permittivity` of a soil spec now warns with
  `InertConfigurationWarning` when the soil is built (it was dropped silently on
  the generator path). Test:
  `tests/test_soil_specs.py::test_inert_permittivity_warns_on_the_generator_path`.
- **Carriage return in a docstring (10.95):** the soil-spec module docstring
  held `\rho` in a non-raw string. Test:
  `tests/test_generator_docstrings.py::test_generator_docstrings_contain_no_control_characters`.

### Removed
- Unreleased names of the first 2026.11 consolidation: `PlannedBranch.concrete_shell_of`
  and `materialise.PEN_JUNCTION_ROD_LENGTH_M`.
