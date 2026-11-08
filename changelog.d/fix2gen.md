### Added
- **Outline width of a footprint (audit A2G-06):** `BuildingFootprint.mean_width_m()`
  (twice the area over the perimeter, the width of a thin rectangle),
  `BuildingFootprint.is_degenerate()`, `groundfield.geo.mean_width` and
  `groundfield.geo.MIN_OUTLINE_WIDTH_M` (1 mm). Test:
  `tests/test_geo_footprint.py::test_mean_width_of_an_outline`.
- **`groundfield.geo.MAX_PAYLOAD_DEPTH` (16):** deepest container nesting of an
  accepted Overpass payload (an Overpass result has 7 levels). Test:
  `tests/test_overpass_ingest.py::test_payload_nested_deeper_than_an_overpass_result_is_rejected`.

### Changed
- ⚠ **Owners of generated mesh foundations (10.12, 10.39; audit A2G-04):** the
  image family solves one uniform leakage current per electrode (an owner). A
  mesh foundation (cross-braces) is now one owner per wire line at every
  orientation, bare or encased, for both `concrete_model` values; an
  axis-aligned bare one was one `grid_mesh` owner. One owner over perimeter and
  braces was +1.6 / +2.5 / +3.2 % high for 2 × 2 / 3 × 3 / 4 × 4 meshes against
  the equipotential reference (one owner per segment, equal to `mom`), one owner
  per wire is +0.3 / +0.5 / +0.6 %. A `"distributed"` ring foundation is one
  owner per side also without a shell (it was one owner when bare and
  axis-aligned). Numbers move: bare axis-aligned mesh foundations −1.2 % (10 m,
  2 × 2) to −2.5 % (30 m, 4 × 4); generated networks with the default mesh
  foundations −0.7 to −0.8 % in every node potential and up to 4.7 % in surface
  potentials; the largest touch voltage over the default house mesh −3.6 %
  (2.528 → 2.437 V per ampere, reference 2.402); bare distributed rings at most
  0.06 % up to aspect ratio 4 (0.24 % at 10). A `"lumped"` ring keeps one
  `grid_mesh` owner when axis-aligned and one owner per side when rotated
  (0.24 % apart at aspect ratio 10), and polygon outlines stay one closed wire:
  one closed polyline would drop the interior sample of the default
  touch-voltage envelope, which decides it for rings of 20 m and more. AP1
  (lumped polygon foundations) does not move. For a mesh foundation,
  `electrode_currents[<anchor>]` and `grounding_impedance(<anchor>)` now refer
  to its first wire, as for every rotated foundation: use `cluster_impedance`
  or the sum over `<name>` and `<name>_w<j>`. Golden cases: `gen_tn_default`,
  `gen_layout_builder`, `gen_tn_measurement_overhead_leads`.
- ⚠ **PEN cables of `TnNetworkGenerator` form a nearest-point tree (10.38;
  stream DOC-B):** every cable starts at the point of the network already built
  nearest to its far end, inside a cable at a leakage-free tap junction
  (`pen_tap_<k>`, a plan junction): cabinet cables in the order of their
  distance from the substation (collinear cabinets are chained), service drops
  from the cable path that feeds their cabinet or radial-trunk tap (or from a
  drop already routed for it, closer buildings first); a building whose
  connection point lies on a cable splits it and gets no drop. Service drops
  started at their cabinet before, 60 to 160 m back along the main in the
  default network, and now tap the main beside each house (5 m). Numbers move
  for every generated TN network: default network substation potential −2.1 %,
  substation leakage −3.1 to −4.8 %, house potentials −0.2 to +1.0 % (the
  houses are tied closer to the substation); measurement network −1.1 %. Main
  sections are named `<cable>_s<k>`. Golden cases: `gen_tn_default`,
  `gen_tn_measurement_overhead_leads`.
- **Overpass ids and levels:** element ids and member refs must be 64-bit
  integers (`OverpassError` otherwise); a non-finite or negative
  `building:levels` tag gives `levels=None`; outlines without area are skipped
  (see Fixed).

### Fixed
- **A shell that is electrically absent moved a mesh foundation (10.12; audit
  A2G-04):** with ρ_c = ρ_soil (or a 0.1 mm shell) the default 10 × 10 m mesh
  foundation lost 1.27 % of its impedance and its largest touch voltage on the
  offset curve dropped from 232.26 to 221.21 V (100 A), because the shell
  switched it from one owner to one per wire; a distributed 20 × 2 m ring moved
  by 0.24 %. Tests:
  `tests/test_foundation_owners.py::test_vanishing_shell_on_a_mesh_foundation_is_continuous`,
  `::test_vanishing_shell_on_a_distributed_ring_is_continuous`.
- **Mesh foundation impedance depended on the orientation (10.39; audit
  A2G-04):** the same bare mesh gave 4.79618 Ω at 0° and 4.73771 Ω at 30°
  (100 Ω·m, line kernel). Tests:
  `tests/test_foundation_owners.py::test_mesh_foundation_does_not_depend_on_its_orientation`,
  `::test_distributed_ring_does_not_depend_on_its_orientation`,
  `::test_mesh_foundation_is_close_to_the_equipotential_reference`.
- **Untrusted Overpass JSON raised other exceptions than `OverpassError` (10.05;
  audit A2G-05):** a body nested deeper than the decoder's limit (about 10 000
  levels) raised `RecursionError`, also from a cache file; an unknown key nested
  1 500 levels decoded, passed every check and crashed the cache write
  (`json.dump` recurses in Python); a 400-digit integer coordinate raised
  `OverflowError`; `"id": null` crashed the parser with `TypeError`. Tests:
  `tests/test_overpass_ingest.py::test_deeply_nested_body_raises_overpass_error`,
  `::test_payload_nested_deeper_than_an_overpass_result_is_rejected`,
  `::test_deeply_nested_cache_file_is_refetched`,
  `::test_vertex_with_a_huge_integer_coordinate_raises_overpass_error`,
  `::test_huge_integer_coordinate_in_the_json_text_raises_overpass_error`,
  `::test_ids_beyond_64_bits_are_rejected`,
  `::test_element_with_a_null_id_is_parsed_without_an_id`,
  `::test_non_finite_or_negative_levels_are_unknown`.
- **Outlines without area (10.05, 10.54; audit A2G-06):** `parse_overpass_payload`
  kept outlines of identical or collinear vertices with the default
  `min_area_m2=0` (although documented as skipped); their oriented bounding
  rectangle raised `IndexError`, so `LvNetworkLayout.to_world` and
  `TnNetworkGenerator` crashed instead of widening the foundation to
  `MIN_FOUNDATION_SIDE_M`. The parser now skips outlines narrower than 1 mm on
  average; the bounding rectangle of such an outline is its covering segment,
  which the generators widen with a warning. Tests:
  `tests/test_overpass_ingest.py::test_outline_without_area_is_skipped`,
  `::test_thin_but_real_outline_is_kept`,
  `tests/test_geo_footprint.py::test_ombr_of_a_degenerate_outline_is_its_segment`,
  `tests/test_footprint_foundations.py::test_a_footprint_without_area_gets_a_minimum_foundation`,
  `::test_tn_generator_widens_a_footprint_without_area`.
- **Generator objects pickled by v0.15.0 (10.69; audit A2G-07):** a pickled
  `TnNetworkConfig` unpickled and then failed in `build()` with `AttributeError`
  (renamed fields of soil specs, cabinets, PEN topology and the config itself).
  Pickles are now upgraded like JSON input: renamed fields and values, defaults
  for fields added since, and instances of the deprecated aliases (`KvsConfig`,
  `StarKvsTopology`, `KvsPlacement`, `OrtsnetzLayout`) load as the current
  classes, equal to the same object loaded from its v0.15.0 JSON. Tests:
  `tests/test_generator_pickles.py`.
- **Overlapping PEN cables made an inductive network non-passive (10.38; stream
  DOC-B):** in a 2-house network both service drops ran from the cabinet 180 m
  out back along the PEN main (Im Z_in = −0.633 Ω at 1 kHz with
  `inductance_model="neumann"`); now +0.072 / +0.055 / +0.051 Ω (perfect mirror /
  Sommerfeld / Carson series). Tests: `tests/test_pen_routing.py`.
- **Integer fields of a resolved config held floats (10.53; stream DOC-B):**
  `GeneratorConfig.sample` stored the float draw of a distribution in
  `building_counts` and `CabinetConfig.fixed_count`, and `model_dump_json`
  warned; a draw for an integer field is now `int(round(x))`, the value the
  generators build with. Test:
  `tests/test_generators_base.py::test_resolved_integer_fields_hold_integers`.
