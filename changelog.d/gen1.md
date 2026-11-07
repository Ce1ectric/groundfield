### Added
- **Network plan (10.78):** `NetworkPlan` with `PlannedSite`, `PlannedElectrode`,
  `PlannedJunction`, `PlannedBranch`, `ConductorParameters`, `PlannedMeasurement`
  and `PlannedSource` is the pure-data description of one realisation (sites with
  concrete electrodes, PEN junctions, cables, measurement set-up, sources, soil).
  `TnNetworkGenerator.plan()` and `LvNetworkLayout.plan()` produce it,
  `materialise_plan(plan)` turns it into a `World`; `build()` / `to_world()` are
  plan + materialise. A plan round-trips through JSON and rebuilds the same world.
  `GroundingSystemSpec.resolve_electrodes`, `plan_site` and `with_footprint`
  expose the steps of `build_at`. Test: `tests/test_network_plan.py`.
- **Foundation outline of the layout builder (10.54):** `LvNetworkLayout.to_world`
  and `plan` take `foundation_outline`: `"ombr"` (default, oriented minimum
  bounding rectangle of the footprint, the rule of the OSM placement of
  `TnNetworkGenerator`), `"polygon"` (ring along the real outline, DIN 18014,
  ring-style specs only) or `"spec"` (the fixed spec geometry of v0.15.0).
  Test: `tests/test_footprint_foundations.py`.
- **Explicit OSM building-tag filter:** every `building=*` value is a footprint
  by default (also `garage`, `shed` and `building=no`), as before.
  `parse_overpass_payload`, `query_and_project` and `LvNetworkLayout.from_osm`
  take `exclude_building_tags`; `groundfield.geo.NON_SERVICE_BUILDING_TAGS`
  (carport, container, garage, garages, greenhouse, hut, no, roof, ruins, shed,
  shelter, tent) is the documented opt-in set. Test:
  `tests/test_overpass_ingest.py::test_building_tag_filter_drops_non_service_buildings`.
- **Reproducible generator draws (10.53):** `TnNetworkGenerator.build()` and
  `plan()` accept `rng=` (seed or `numpy.random.Generator`).

### Changed
- **One network plan and one materialiser (10.78):** both front-ends
  (`TnNetworkGenerator`, `LvNetworkLayout`) produce a `NetworkPlan`; one module
  (`groundfield.generators.materialise`) builds foundations and their concrete
  shells, PEN junctions, PEN and service cables, the measurement loop and the
  sources. The layout module (`ortsnetz_builder.py`, about 2700 lines) is
  split into the package `groundfield.generators.layout` (model, routing,
  measurement, penetration, plot). Worlds are unchanged: 18 of 20 reference
  worlds are byte-identical to v0.15.0, the other two list the same conductors
  in another order and solve bit-identically; the refactor alone passes all
  golden values in strict mode. Test:
  `tests/test_network_plan.py::test_tn_build_is_materialised_plan`.
- **English names in the generator layer (10.69):** see *Deprecated* for the
  full table. Generated element names (`trafo_*`, `kvs_*`, `house_*`) are data
  and stay unchanged, so stored worlds and results keep their keys.
- ⚠ **`sample_world` collapses nested distributions once per realisation
  (10.53):** as documented, `cfg.sample(rng)` now replaces every distribution of
  the config tree, also those inside the electrode specs of a building type, so
  all buildings of a type share the drawn value in `sample_world`. v0.15.0
  left these to per-building draws while reporting the config as resolved.
  `build()` on an unresolved config still draws per building; plain
  `presence_prob` and `jitter_m` still draw per site. Affected: `sample_world`
  realisations of configs with distributions inside specs. Golden cases: none
  moved.
- **Overpass retries:** also on HTTP 503, honouring `Retry-After` (capped at
  60 s); `max_retries` is limited to 5. Non-integer `timeout_s` raises
  `TypeError` (it is part of the query text). Test:
  `tests/test_overpass_ingest.py::test_retries_honour_retry_after_and_are_bounded`.

### Fixed
- **Resolved configs replay their realisation (10.53, F07):** `cfg.sample()`
  left distributions in nested specs and in `building_counts`; replays of the
  "resolved" config with seeds 0/1/2 gave 15/15/9 electrodes at 337/280/168
  Ohm m instead of 15 electrodes at 331 Ohm m. Every distribution is now
  resolved (dicts in sorted key order), and `sample_world(rng)` draws the whole
  realisation from `rng` (the construction seed gave 5 vs 7 houses for the same
  `rng`). Test:
  `tests/test_generator_reproducibility.py::test_resolved_config_replays_the_world_for_any_seed`.
- ⚠ **Stable penetration mask without OSM ids (10.53):**
  `LvNetworkLayout.foundation_mask` keyed footprints without `osm_id` by their
  list index; reversing the list changed the equipped houses and Z by 2.8 %.
  They are now keyed by their outline; footprints with `osm_id` keep their keys,
  so stored masks of OSM layouts stay valid. Affected: masks of footprints
  without `osm_id` at penetration below 1. Golden cases: none moved. Test:
  `tests/test_generator_reproducibility.py::test_mask_without_osm_id_ignores_footprint_order`.
- ⚠ **Layout foundations follow the building footprints (10.54, F28):**
  `LvNetworkLayout.to_world` gave every house the same 10 x 10 m foundation at
  the footprint centroid. Foundations now take size, orientation and centre
  from the footprint (oriented minimum bounding rectangle), like the OSM
  placement of `TnNetworkGenerator`. In three test streets (8 houses, 100 Ohm m,
  50 Hz, image backend) the substation impedance moves by 0 % (10 x 10 m
  squares), +0.6 % (12 x 9 m houses rotated by 0 to 35 deg) and +5.2 %
  (6 x 4 m garages alternating with 30 x 15 m blocks). Affected: every layout
  world whose footprints are not 10 x 10 m axis-aligned squares;
  `foundation_outline="spec"` reproduces v0.15.0. Golden cases:
  `gen_layout_builder` unchanged (10 x 10 m squares). Test:
  `tests/test_footprint_foundations.py::test_foundation_size_follows_the_footprint`.
- ⚠ **Rotated mesh foundations without the singularity clamp (10.39):** rotated
  mesh foundations were built from full-length strips whose segment midpoints
  could coincide at the crossings; the solver's 1 mm clamp then made a
  10 x 10 m, 2 x 2 mesh at 30 deg 11.8 % too high at a segment length of
  2.0 m and 11.0 % at 1.5 m (no error at 2.5 m or 1.0 m). Crossed wires are now
  polylines with a vertex at every crossing, so crossings fall on segment ends
  as in `GridMeshElectrode`; ring-style foundations keep their four strips.
  The oriented bounding rectangle of a square reports the axis closest to +x,
  and a foundation at a multiple of 90 deg is built on the axis-aligned path:
  mesh foundations of squares and north-south rectangles change by +1.0 to
  +1.2 % (or -9.5 % where the clamp fired), ring foundations by less than
  0.002 %. Note: with the uniform-current `image` backend a rotated mesh stays
  about 1.1 % below the axis-aligned mesh of the same rectangle (six wires with
  one current each instead of one); `mom` and `bem` give the same value for
  both, closer to the rotated one. Affected: mesh-style foundations on rotated
  or right-angle footprints (OSM placement, layout builder). Golden cases:
  none moved. Test:
  `tests/test_rotated_foundations.py::test_rotated_mesh_has_no_clamp_and_converges_smoothly`.
- **Jitter distribution (10.55, F47):** `ManhattanGridPlacement(jitter_m=Uniform(...))`
  raised `TypeError` ("Unable to apply constraint ge"); the bound now applies to
  the float only, a distribution is drawn once per site and a draw of 0 or less
  leaves the site on its grid point. Test:
  `tests/test_placement.py::test_manhattan_grid_jitter_accepts_a_distribution`.
- ⚠ **Measurement leads at their configured depth (10.55, F48):**
  `MeasurementLeadConfig.depth_m` was drawn and discarded: buried and overhead
  leads ran on the same diagonal from the substation ring (z = 0.6 m) to the
  auxiliary rod head (z = 0). A lead now runs horizontally at z = `depth_m`
  (z > 0 below the surface) between the anchors and stays connected to both;
  `depth_m = 0` is placed 0.1 m above ground (`SURFACE_LEAD_HEIGHT_M`; the
  surface plane is singular for the image kernels and the Sommerfeld kernel
  degrades within about ten wire radii of it), a galvanic lead above ground is
  rejected. In the golden geometry (perfect-mirror earth) the probe potential
  moves by 2.5e-4 at 50 Hz and its coupled imaginary part by +37 % (overhead)
  and +21 % (buried) at 1 kHz; v0.15.0 gave identical results for both lead
  types. Affected: `TnNetworkGenerator` worlds with measurement leads. Golden
  cases: `gen_tn_measurement_overhead_leads` (potentials up to 2.5e-4, currents
  up to 2.8e-4, lead end points). Test:
  `tests/test_measurement.py::test_buried_lead_runs_at_its_depth`,
  `tests/test_measurement.py::test_overhead_lead_lies_above_ground`.

### Deprecated
- English names replace the German identifiers and capitalised unit suffixes
  of the generator layer (10.69). Every old name keeps working, emits a
  `DeprecationWarning` naming the replacement, and is removed no earlier than
  2027.5. Configurations written by v0.15.0 (JSON keys `kvs`,
  `source_magnitude_A`, `star_kvs`, `resistivity`, `thickness_m`, ...) still load.

  | old (v0.15.0) | new |
  |---|---|
  | module `groundfield.generators.ortsnetz_builder` | `groundfield.generators.layout` |
  | `OrtsnetzLayout` | `LvNetworkLayout` |
  | `KvsPlacement` | `CabinetPlacement` |
  | `KvsConfig` | `CabinetConfig` |
  | `StarKvsTopology`, `kind="star_kvs"` | `StarCabinetTopology`, `kind="star_cabinet"` |
  | `TnNetworkConfig.kvs` | `TnNetworkConfig.cabinets` |
  | `TnNetworkConfig.source_magnitude_A` | `TnNetworkConfig.source_magnitude_a` |
  | `CabinetConfig.quote_per_100_buildings` | `CabinetConfig.count_per_100_buildings` |
  | `RadialTrunkTopology.slots_per_kvs` | `RadialTrunkTopology.slots_per_cabinet` |
  | `RadialTrunkTopology.kvs_spacing_m` | `RadialTrunkTopology.cabinet_spacing_m` |
  | `RadialTrunkTopology.plan_feeder_kvs_positions()` | `RadialTrunkTopology.plan_feeder_cabinet_positions()` |
  | `LvNetworkLayout.add_kvs()` | `LvNetworkLayout.add_cabinet()` |
  | `LvNetworkLayout.kvs_placements` | `LvNetworkLayout.cabinet_placements` |
  | `from_osm(kvs_lat_lons=, kvs_xys=)` | `from_osm(cabinet_lat_lons=, cabinet_xys=)` |
  | `to_world(kvs_grounding=)` | `to_world(cabinet_grounding=)` (also on the new `plan`) |
  | `source_magnitude_A=` of `to_world`, `measured_grounding_impedance`, `verify_current_balance` | `source_magnitude_a=` |
  | result keys `positive_side_A`, `auxiliary_side_A`, `expected_*_A`, `abs_error_*_A`, `tolerance_A` of `verify_current_balance` | `positive_side_a`, `auxiliary_side_a`, `expected_*_a`, `abs_error_*_a`, `tolerance_a` |
  | `HomogeneousSoilSpec.resistivity` | `HomogeneousSoilSpec.rho` |
  | `SoilLayerSpec.resistivity`, `SoilLayerSpec.thickness_m` | `SoilLayerSpec.rho`, `SoilLayerSpec.h` |

  The module alias `groundfield.generators.tn_ortsnetz` keeps working as
  before. Test: `tests/test_generator_naming.py`.

### Security
- **Bounded, validated Overpass requests (10.05):** `timeout_s` was put into
  the query text unchecked (a string could inject Overpass QL); NaN or
  out-of-range coordinates and radii reached the server; the response was read
  into memory without limit; an HTTP-200 payload with a runtime-error `remark`
  was cached as an empty result; a writer killed mid-way left a truncated cache
  file that broke every later call. Now `timeout_s` must be an integer in
  [1, 900] s, coordinates finite WGS84 values and the radius at most 10 km;
  endpoints must use https (http only for localhost); the body is streamed and
  capped (`max_response_bytes`, default 64 MiB); error remarks and malformed
  payloads raise `OverpassError` and are not cached; cache files are written
  atomically and an unreadable one is fetched again with a warning; a
  non-default endpoint is part of the cache key (the default endpoint keeps the
  old key, existing caches stay valid); the query text of valid requests is
  unchanged. Test: `tests/test_overpass_ingest.py` (mocked transport; one real
  request behind the `network` marker).
- **MD5 on FIPS systems (10.07):** the penetration-mask hash called
  `hashlib.md5()` without `usedforsecurity=False` and failed on FIPS builds of
  Python. The digest is unchanged, so stored masks stay valid. Test:
  `tests/test_generator_reproducibility.py::test_mask_works_where_md5_is_blocked_for_security`.
