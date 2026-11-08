### Added

- **`groundfield.solver.InterfaceContactWarning` and the `diagnose()` category
  `interface_contact` (10.32, audit A3S-05):** a rod head or tip within one
  wire radius of a layer interface. The thin-wire model has no end face: a
  rod that ends exactly at the interface is a wire on one side of it, one
  that reaches past it by a fraction of its radius touches the other layer
  through a short piece only, while the solid rod is continuous in between.
  Against an axisymmetric finite-volume model of the solid rod (4.5 m,
  a = 8 mm, 1000/30 Ω·m) the line kernel reads +3.3 / +3.9 / +7.2 / +21.4 %
  with the tip 16 / 8 / 1 / 0 mm above the interface and −4.6 / −1.2 /
  +0.7 / +2.1 % 1 / 4 / 8 / 16 mm past it (point kernel and `mom` alike).
  The step at zero penetration that an earlier entry of this release called
  the onset of the contact is this limit of the model. No end-face term:
  the solid rod's 161.4 Ω at the interface lies between the two branches of
  the thin wire (196 Ω above, at most 149.2 Ω past it). Tests:
  `tests/test_interface_slivers.py::test_rod_tip_within_one_wire_radius_of_the_interface_warns`,
  `::test_rod_head_within_one_wire_radius_of_the_interface_warns`,
  `tests/test_diagnostics_report.py::test_diagnose_reports_a_rod_end_at_the_interface`.
- **`min_distance` of `two_layer_real_space_kernel`,
  `two_layer_layered_correction_real_space` and
  `two_layer_layered_correction_group` (audit A3S-01):** the regularisation
  length of the closed-form 1/R terms in m; they are clamped at
  `min(1/lambda_clamp, min_distance)` instead of `1/lambda_clamp`
  (`h_1/200` near the interface). `None` (default) keeps the clamp; every
  value at or beyond it is bit-identical. Test:
  `tests/test_cross_layer_line_kernel.py::test_min_distance_leaves_every_farther_image_bit_for_bit`.
- **`groundfield.world.RESERVED_NAME_PREFIXES`** (`"__layer_"`,
  `"__cond_"`), see *Fixed*.

### Fixed

- ⚠ **Cross-layer diagonal at the interface (10.28, 10.32, audit A3S-01):**
  the diagonal of a segment is the layered correction at the wire radius
  with the interface image exchanged for its Galerkin integral, but the
  correction clamped its closed-form 1/R terms at `h_1/200` (25 mm for
  `h_1 = 5` m), so every piece within `h_1/400` of an interface lost 68–84 %
  of its layered correction. A ring (r = 3 m, a = 5 mm) at `h_1 = 5` m read
  −5.29 Ω with the default engine (1000/30 Ω·m; 322 Ω instead of 6.80 Ω in
  30/1000 Ω·m), and the contact values of a rod tip just past the interface
  were 10–32 % low in every segment backend. The solver now evaluates the
  diagonal at the wire radius and pairs closer than `h_1/200` at the 1 mm of
  its point forms (also `mom_sommerfeld`'s two-layer off-diagonal kernel and
  `two_layer_probe_matrix`, which clamps at its own `min_distance`). The
  diagonal equals the image solution to 1e-13; the ring is continuous through
  `h_1` (4.1826 Ω 1 µm above, 4.1805 Ω at `h_1`). Numbers move for a rod tip
  or head within `h_1/400` of an interface (12.5 mm for `h_1 = 5` m, 75 mm for
  30 m; 4.5 m rod, a = 8 mm, 1000/30 Ω·m, tip 1 µm past `h_1`: 133.5 → 149.2 Ω,
  +11.7 %; 10000/10 Ω·m: 123.2 → 180.7 Ω; against the solid rod −4.6 / −1.2 /
  +0.7 % at 1 / 4 / 8 mm, before −14.6 / −10.7 / −6.8 %), for horizontal wires
  at or within `h_1/400` below an interface, and for pairs of segments closer
  than `h_1/200` (segment lengths below about `h_1/100` near the interface:
  a 2 m rod through `h_1 = 30` m, ds = 0.05 m, line kernel −0.014 %). Not
  affected: AP1 (station rods end 0.2 m or more above `h_1`). Golden cases:
  none. Tests: `tests/test_cross_layer_line_kernel.py`
  (`::test_ring_resistance_is_continuous_through_a_deep_interface`,
  `::test_diagonal_correction_at_the_wire_radius_matches_the_image_solution`,
  `::test_pairs_closer_than_the_clamp_match_the_image_solution`),
  `tests/test_interface_slivers.py::test_rod_tip_contact_against_the_solid_rod_reference`,
  `::test_rod_head_contact_against_the_solid_rod_reference`,
  `::test_rod_tip_contact_does_not_depend_on_the_interface_clamp`.
- **Point form above the interface (10.28, audit A3S-02):** the point-form
  guard of audit round 2 ran only when a segment reaches `h_1`. With every
  wire above `h_1` the point form takes the image of a segment in the
  interface as a point source 2(`h_1` − z) away: a ring (r = 3 m,
  ds = 0.5 m) 1 µm / 1 mm / 1 cm / 5 cm above `h_1 = 5` m read −1931 / −945 /
  −57.9 / 21.2 Ω (line kernel 4.18 / 4.77 / 15.4 / 28.0 Ω) with
  `kernel="point"`, `mom`, `mom_sommerfeld` (two and three layers) and
  `solve_mutual_*`, also in 0.15.0; +853 % … +0.1 % in 30/1000 Ω·m. The guard
  now applies on both sides of the interface with the same thresholds: a
  horizontal wire closer than 0.387 segment lengths raises
  `BackendCapabilityError`, one within 0.6 warns with
  `PointKernelInterfaceWarning` (−2.5 % / −0.4 % at 0.2 m above `h_1`). The
  default line kernel is unaffected. Tests:
  `tests/test_cross_layer_point_form.py::test_point_form_rejects_a_horizontal_wire_just_above_the_interface`,
  `::test_point_form_warns_for_a_horizontal_wire_above_the_interface`,
  `::test_mutual_matrix_point_form_is_guarded_above_the_interface`,
  `::test_rods_ending_just_above_the_interface_are_not_refused`.
- **Reserved name prefixes (10.66, audit A3S-03):** an electrode called
  `__layer_r__L1` next to a rod `r` through `h_1`, or `__cond_pen__seg_0`
  next to a galvanic conductor `pen`, took the place of the solver's internal
  owner or node of that name and the solve failed with
  `numpy.linalg.LinAlgError`. `add_electrode`, `add_junction` and
  `World.check()` (every solve, worlds loaded from JSON) reject node names
  with the prefixes `__layer_` and `__cond_`. Tests:
  `tests/test_source_validation.py::test_check_rejects_a_name_with_a_reserved_prefix`,
  `::test_reserved_prefixes_are_rejected_when_a_node_is_added`.
- ⚠ **Field of a soil of one resistivity declared as layers (10.48, audit
  A3S-04):** `TwoLayerSoil(rho, rho, h)` solved like homogeneous soil, but
  `FieldResult.potential` and `solve_mutual_field` evaluated sources below
  `h` with the point form of the two-layer kernel instead of the line sources
  of the homogeneous result: +2.7 % at (0.2, 0, 4) m next to a 0.3–6.3 m rod,
  21 % at 5 cm from it. Both now read the canonical layer stack and equal the
  homogeneous result bit for bit; a stored `MultiLayerSoil` with at most two
  distinct layers is evaluated as two-layer soil (it raised
  `NotImplementedError` for three declared layers). Who is affected: field
  values of equal-resistivity layered declarations near wires below `h`.
  `fem` keeps the declared interface in its mesh on purpose (mesh topology
  continuous over a ρ₂ sweep): 1.7e-6 / 1.0e-5 from the homogeneous solve for
  two / three equal layers. Golden cases: none. Tests:
  `tests/test_layer_stack.py::test_field_of_an_equal_resistivity_stack_is_the_homogeneous_field`,
  `::test_potential_of_a_stored_three_layer_declaration_with_two_distinct_layers`.
- **Building large networks (audit round 3, generator note):** since
  generated mesh foundations are one electrode per wire line,
  `TnNetworkGenerator.build()` was 10–80 times slower for 100–1000 houses:
  `World.add_*` scanned every list for the new name. A name index next to the
  world (not part of the model: equality, copies, pickles and JSON are
  unchanged) makes the check O(1) per add; generated worlds are bit-identical
  (fingerprints for 100 / 400 / 1000 houses). 1000 houses: 6.97 → 2.44 s
  (star) and 7.52 → 2.65 s (radial) on the measuring machine. Tests:
  `tests/test_world_names.py::test_each_add_costs_constant_name_work`,
  `::test_lists_edited_by_hand_are_seen_by_the_next_add`,
  `::test_the_index_is_no_state_of_the_world`.
- **Quoted near-field deviations (10.21, 10.22):** `UniformCurrentFieldWarning`,
  the safety module and the known limitations quoted "+19 % / −21 %" (touch /
  step 1 m outside a 20 m grid corner), the point-kernel numbers, and the
  `image` engine page −28 % for a step that starts above the corner. They now
  give the default (line) kernel and the step: touch 1 m outside +20 %, step
  from 1 m to 2 m outside −21 %, from the point above the corner 1 m outwards
  −28 % (point kernel +19 / −21 / −28 %). The `interior` docstring of
  `touch_voltage_envelope` called a generated foundation a `grid_mesh`; it is
  a group of bonded wires whose cells the default samples. Tests:
  `tests/test_touch_voltage_footprint.py::test_the_quoted_corner_deviations_are_those_of_the_default_kernel`,
  `tests/test_touch_voltage_envelope.py::test_a_generated_mesh_foundation_is_sampled_in_its_cells_by_default`.
