### Added

- **Junction nodes (10.38, prerequisite of 10.17):** `groundfield.api.create_junction(world, position=..., name=None)`
  creates a leakage-free network node (`groundfield.geometry.junction.Junction`,
  `World.junctions`, `World.add_junction` / `get_junction`). Conductors start or
  end at junctions (`create_conductor` accepts junctions and their names) and
  sources attach to or return at them; a junction has no soil contact, takes
  part in Kirchhoff's current law and reports its potential per frequency in
  `FieldResult.metadata["junction_potentials"]`. Electrodes and junctions share
  one node namespace. A junction subnetwork without soil contact (no path to an
  electrode or a galvanic conductor) is rejected before the solve. Measurement
  leads that end on junctions at the instrument leave the soil problem exactly
  unchanged (fall-of-potential Z_meas equal to the lead-free value to 5.4e-16),
  where a probe lead ending on the station short-circuits the voltmeter
  (0.46 % of the true value); a junction leaks nothing where a 5 cm joint rod
  leaked 0.35 % of the test current in the test world. Worlds without junctions
  serialise exactly as before. Supported by the image family
  (`BackendCapabilities.junctions`); other backends reject such worlds with
  `BackendCapabilityError`. Tests: `tests/test_junction_nodes.py`.
- **`World.check()` (10.66):** validates source terminals and conductor ends
  (with the closest names), the excitation, voltage sources and short-circuited
  two-terminal sources; called by every solve. Test:
  `tests/test_source_validation.py`.
- **Aggregated diagnostics (10.70):** `groundfield.diagnostics.diagnose(world,
  engine=None, *, result=None, proximity=True)` returns a `DiagnosticsReport` of
  `Diagnostic` findings (category, severity, subjects, message, data) for thin
  wires, resolution, segment budget, close or interpenetrating leakage wires
  (`groundfield.conductors.find_close_wires`, 10.44), inert settings (conductors
  not attached at both ends, the deprecated shell registry, non-default boundary
  conditions or permittivity) and, with a result, series truncation. `str()`
  prints one line per cause, `.warn()` emits one `DiagnosticsWarning`. Close
  wires are a warning with the point kernel (two bonded 3 m rods 2 mm apart
  solve to 5.7 times the one-rod impedance) and information with the exact line
  kernel (0.998 times). Test: `tests/test_diagnostics_report.py`.
- **`diagnostics.check_segment_budget` and `ResourceLimitError` (10.06):** the
  hard segment limit of the solver pipeline (see *Fixed*).

### Changed

- ⚠ **`Source.return_to` is honoured by the image family (10.16, F01/F30).**
  `image`, `image_2layer` and `image_nlayer` inject +I at `attached_to` and -I at
  `return_to` through one shared injection step; the leakage currents of a world
  of two-terminal sources sum to zero and `input_impedance()` is
  (φ_a − φ_r)/I. Up to 0.15.0 the auxiliary electrode of a simulated
  fall-of-potential measurement carried 0 A and the station potential did not
  depend on its distance (13.107017 V for d_aux = 20 … 1600 m in the test
  world); now 12.310 V at 20 m rising to 13.097 V at 1600 m, and the probe at
  61.8 % reads the remote impedance to −1.6e-3 (20 m), −3.5e-6 (200 m), as
  Tagg's rule says (0.15.0: −9.9 % and −0.98 %). Who is affected: every world
  whose sources set `return_to` — the measurement worlds of
  `TnNetworkGenerator` and `LvNetworkLayout`. The generators still wire the
  feed lead from the station straight to the auxiliary electrode, so that lead
  now closes the loop metallically (station potential 1.5008 V → 4.15e-3 V
  in the golden case); they are re-wired through junctions in the generator
  work (10.17). `LvNetworkLayout` still adds its 180° counter-source until then,
  which now counts the return current twice; the solve warns about such a
  counter-source. Golden case: `gen_tn_measurement_overhead_leads`.
- ⚠ **Source-free clusters float in the image family (10.29, image half).** A
  bonded group of passive electrodes and a passive network joined by finite
  conductors now enter the nodal system with zero net impressed current instead
  of being dropped and read back as the incident field: one potential per
  cluster, equalising currents, and members of an isolated floating cluster
  report currents that sum to exactly zero (cluster impedance `nan`). Passive
  ring + rod bonded pair beside a driven rod: 2.5735 V / 1.8055 V and 0 A
  became one 2.4119 V with ±14.3 mA (`mom`: 2.4189 V, ±13.1 mA). A lone
  passive electrode is unchanged (exactly equivalent under the uniform-current
  ansatz), so worlds without bonded passive groups or passive networks are
  bit-identical. Who is affected: probe and auxiliary electrode groups,
  unbonded neighbours with several electrodes, passive measurement leads.
  Golden case: `cond_coupled_leads_open_circuit` (the grounded measurement lead
  beside the source lead: φ(m1) +42.5 %, driven side −0.89 % / −3.8 %).
- **One warning per cause instead of per electrode (10.70):**
  `check_segment_resolution` aggregates its findings (docs example 07: 81
  resolution warnings became 9, one per solve); the `return_to` capability
  warning disappears for the image family.
- **`expected_segments` follows the solver's discretisation (10.79)**, including
  wires split at soil-layer interfaces (a 3 m rod across h₁ = 1.3 m: 7 segments,
  the 0.15.0 prediction said 6). Test:
  `tests/test_diagnostics_report.py::test_expected_segments_is_exact_for_rods_split_at_the_interface`.

### Fixed

- **Invalid source references solved silently (10.66).** A misspelled
  `attached_to` / `return_to`, a source on a conductor (documented as allowed,
  injected nothing), a source returning to its own node, a world without a
  current source, a source of magnitude 0, and a conductor end naming no node
  (the conductor dropped out of the network) now raise — at `create_source` /
  `World.add_source` / `World.add_conductor`, or when an edited or loaded world
  is solved — with the closest names. `magnitude` and `phase_deg` must be
  finite, also when assigned later. Tests: `tests/test_source_validation.py`.
- **`VoltageSource` solved to all-zero potentials (10.51).** Solving a world with
  a voltage source raises `NotImplementedError` naming the alternative (a 1 A
  current source scaled by U / Z_in). Test:
  `tests/test_source_validation.py::test_voltage_source_raises_not_implemented_naming_the_alternative`.
- **`HARD_LIMIT` only warned (10.06).** A world whose predicted segment count
  exceeds `groundfield.diagnostics.HARD_LIMIT` (20 000, read at solve time) is
  rejected with `ResourceLimitError` before anything of the problem size is
  allocated, also in `solve_mutual_*`: a 5 m ring entered in millimetres
  (62 832 segments) fails after 0.06 s with a 0.67 MiB allocation peak; the
  wave-1 code solved the full problem (173 s, 140 MiB peak on the review
  machine) and returned 0.0096 Ω. Test:
  `tests/test_diagnostics_report.py::test_world_beyond_the_hard_limit_is_rejected_before_any_allocation`.
- **Polyline and star electrodes skipped by the resolution check (F53, F54).**
  Listed as fixed in 0.15.0, but `diagnostics.py` was unchanged since 0.13.0;
  every electrode kind is now checked through `Electrode.min_feature_length()`.
  Test: `tests/test_diagnostics_report.py::test_resolution_check_covers_star_and_polyline`.
- **Close wires went unreported (10.44, diagnostics part):** see `diagnose` above.
  Test: `tests/test_diagnostics_report.py::test_close_and_interpenetrating_wires_are_reported`.

### Deprecated

- **`World.concrete_shell_corrections`** (the lumped ADR-0012 V1 registry,
  10.12–10.14): the solver consumes concrete shells only as the per-length
  coefficient `concrete_shell_coefficient_ohm_m` of the electrodes and never
  read this registry (nor did `solve_mutual_*`). Solving a world with entries
  emits a `DeprecationWarning` naming the coefficient; the field still loads
  from JSON and will be removed no earlier than 2027.5. Test:
  `tests/test_concrete_shell_registry.py`.
