### Added
- **`groundfield.io.groundinsight.MAX_METADATA_DEPTH` (32):** deepest nesting of
  objects and lists in the metadata of a BusType document (the exporters write
  four levels); a deeper document is refused on load. Test:
  `tests/test_io_groundinsight.py::test_crafted_bustype_documents_raise_value_error`.

### Changed
- ⚠ **PEN cables of `TnNetworkGenerator` (10.38; stream DOC-B; audit A3SG-01,
  A3SG-02, A3SG-03):** every cable cabinet has its own cable, as the topology
  documents (star: a straight cable from the substation; radial trunk: one
  cable per trunk segment); cabinets on one line get separate cables in one
  trench. A service drop taps the cable path that feeds its own building — the
  cable of its cabinet, or the trunk from the substation to its tap source — at
  the point nearest to the building's connection point: a leakage-free tap
  junction `pen_tap_<k>` that splits the cable into sections `<cable>_s<k>`, or
  the substation / the cabinet at the end of the cable; a building in a
  substation slot of the radial trunk is connected to the substation. What the
  router guarantees: a drop never starts on another building's drop and never
  runs along the cable it taps; no cable section or drop is shorter than 1 cm (a
  tap closer to a node is that node; longer for PEN radii above 7.3 mm, so that
  every section keeps twice the resistance below which a distributed conductor
  is refused); a building whose connection point lies on its feeding cable
  splits it (no drop); two nodes closer than that raise `ValueError`. A drop
  may share its route with another cable (another cabinet's cable, a trunk,
  another drop), as cables in one trench do; the inductive coupling treats such
  conductors as touching wires and the networks stay passive. Against 0.15.0,
  where every drop ran from its cabinet (60 to 160 m back along the main in the
  default network, coaxial with it): the routing moves the golden networks
  (one cabinet west of every house) by −2.1 % in the substation potential,
  −3.1 to −4.8 % in the substation leakage, −0.2 to +1.0 % in the house
  potentials and −1.1 % in the measurement network;
  the input impedance of default-grid networks with ring foundations (image
  backend, 50 Hz, 100 Ω·m) moves by −6.4 / −1.1 / −0.2 % for 30 / 100 / 200
  houses (star) and 0.0 / −0.1 % for 30 / 100 houses (radial trunk). Against
  the nearest-point tree of the first 2026.11 draft (chained cabinets, drops
  hung on drops): golden networks bit-identical; 30 / 100 / 200 houses −4.1 /
  −2.9 / −2.7 % (ring foundations) and −2.4 / −3.4 / −2.8 % (default mesh
  foundations), radial trunk −0.5 / −0.3 %. Golden cases: `gen_tn_default`,
  `gen_tn_measurement_overhead_leads` (moves declared in `fix2gen.toml`).
- **Polygon foundations of thin outlines (10.54; audit A3SG-07):** with
  `foundation_outline="polygon"` an outline whose oriented bounding rectangle
  is narrower than `MIN_FOUNDATION_SIDE_M` (1 m) is replaced by that rectangle
  widened to 1 m, with the same `UserWarning` as for `"ombr"` (a ring along an
  outline 0.5 to 1 m wide ran along both faces of a wall). Outlines wider than
  1 m on average are unchanged.
- **`TnNetworkGenerator.build()` of large networks (no finding):** the
  materialiser hands objects instead of names to the world, so bonds and PEN
  sections need no search through the world: 1000 default houses 6.5 → 4.4 s
  (star) and 7.3 → 5.4 s (radial trunk), worlds byte-identical. The rest of the
  cost is the name checks of `World.add_electrode` / `add_conductor`, linear in
  the size of the world.

### Fixed
- **`build()` failed for 1–3 % of random TN networks (10.38; audit A3SG-01):**
  a service drop whose tap fell 0.1 µm to 4.7 mm from a node of the cable left
  a section that the `Conductor` model refuses (series resistance below
  1e-6 Ω), so `build()` raised `pydantic.ValidationError` (1 of 100 random
  60-house and 2 of 60 random 200-house networks). The layout builder had the
  same defect for taps a few millimetres from a cable corner or from another
  tap; it now merges cable nodes closer than the shortest section and refuses a
  cable between two sites closer than it with `ValueError`. Tests:
  `tests/test_pen_routing.py::test_a_building_millimetres_beside_a_node_still_builds`,
  `::test_a_tap_millimetres_from_an_earlier_tap_snaps_to_it`,
  `::test_random_position_network_builds`,
  `::test_taps_on_a_thick_pen_keep_sections_the_conductor_model_accepts`,
  `::test_random_networks_build_and_route_as_documented` (400 random networks,
  60 / 200 houses, both topologies),
  `tests/test_lv_network_layout.py::test_taps_millimetres_apart_share_one_node`,
  `::test_a_tap_millimetres_from_a_cable_corner_is_the_corner`,
  `::test_a_cable_between_two_sites_closer_than_a_section_is_rejected`.
- **Cabinets chained on one cable, drops hung on other drops (10.38; audit
  A3SG-02, A3SG-03):** the nearest-point tree of the first 2026.11 draft fed a
  cabinet from the cable of a nearer one, although `StarCabinetTopology` and
  `TnNetworkConfig.pen_topology` document a direct cable per cabinet (input
  impedance +1.8 / +2.6 % for 100 / 200 houses against the 0.15.0 routing,
  against the documented direction), and started most drops on other houses'
  drops (26 of 30 in the radial default grid with ring foundations, 19 of 30 in
  the default star network). Tests:
  `tests/test_pen_routing.py::test_star_topology_wires_every_cabinet_directly_to_the_substation`,
  `::test_collinear_cabinets_have_separate_cables_in_one_trench`,
  `::test_z_in_of_a_100_house_network_drops_against_the_old_routing`,
  `::test_a_drop_taps_the_cable_of_its_own_cabinet`,
  `::test_no_drop_starts_on_another_drop_in_the_radial_default_grid`,
  `::test_overlapping_drops_and_trunks_stay_passive`, `::test_routing_guarantees`.
- **A loaded BusType document crashed the other entry points (10.01; audit
  A3SG-04):** `load_bustype_json` accepted documents for which
  `fit_quality_summary` raised `AttributeError`, `TypeError`, `ValueError` or
  `OverflowError` (`fit_quality` null or a list; a figure that is an object, a
  list, a string or a 400-digit integer), `to_dict()` raised `RecursionError`
  (metadata nested 3 000 levels, which the JSON decoder reads) or wrote a NaN
  description (not strict JSON). The loader now refuses a `name` or
  `system_type` that is an object or a list, a `description` that is neither a
  string nor null, an `impedance_formula` that is an object or a list
  (`EvaluateSpecError`) and metadata deeper than `MAX_METADATA_DEPTH`;
  `to_dict()` writes a non-string description as null and walks the metadata
  without recursion; `fit_quality_summary` reads only finite numbers and
  strings. Tests: `tests/test_io_groundinsight.py::test_crafted_bustype_documents_raise_value_error`,
  `::test_a_loaded_document_never_crashes_the_other_entry_points`,
  `::test_fit_quality_summary_reads_the_figures_it_can`,
  `::test_to_dict_of_a_spec_built_in_memory_is_strict_json_or_refused`.
- **A sum with an exact small imaginary part parsed to zero (10.01; audit
  A3SG-05):** the formula parser drops a collected sum that cancels within its
  128-bit working precision, but compared the complex magnitude of the sum with
  its largest term: `(1 + 1e-31*j - 1)*1e31*f` evaluated to 0 instead of 50j at
  50 Hz. Real and imaginary parts are now judged separately; the 119 accepted
  formulas of the corpus evaluate as before (worst 1.9e-15 against a 60-digit
  mpmath evaluation). Test:
  `tests/test_formula_security.py::test_real_and_imaginary_parts_of_a_sum_cancel_separately`.
- **Polygon foundations of outlines narrower than 0.5 m (10.54; audit
  A3SG-07):** `LvNetworkLayout.to_world(foundation_outline="polygon")` raised
  `ValueError` ("polygon collapses ... after removing sub-centimetre edges",
  although the edge limit is 0.5 m) for every such outline the Overpass parser
  keeps; it is widened now (see Changed), and the materialiser's last check
  names the real limit. Tests:
  `tests/test_footprint_foundations.py::test_a_thin_outline_with_a_polygon_foundation_is_widened`,
  `::test_a_polygon_foundation_of_an_ordinary_outline_is_unchanged`,
  `::test_a_collapsing_polygon_outline_names_the_real_edge_limit`.
- **Margin of generated mesh foundations against `mom` (10.39; audit
  A3SG-06):** the documentation said one current per wire is +0.3 / +0.5 /
  +0.6 % "against `mom`"; that is the line kernel against a line-kernel
  per-segment reference. Against `mom` (point kernel) the default line kernel
  is +1.1 % and `kernel="point"` +0.3 … +0.5 % for 10 m 2 × 2 to 30 m 4 × 4
  meshes; one current for the whole mesh was +2.4 … +3.7 % (point kernel
  +1.5 … +3.1 %). Test:
  `tests/test_foundation_owners.py::test_documented_margin_of_mesh_foundations_against_mom`.
