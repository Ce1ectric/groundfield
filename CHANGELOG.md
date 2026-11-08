# Changelog
<!-- --8<-- [start:changelog] -->

All notable changes to `groundfield` are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

These are concise release notes. The full narrative of every review and audit
pass (findings, measurements, decisions) is kept verbatim in the
[audit history][audit-history]; work that is planned but not scheduled is on
the [roadmap][roadmap]. Up to 0.15.0 both stood in this file; they moved in
2026.11.0, and the release notes of 0.1.0 to 0.15.0 were condensed. The
complete text as published is in the history of this file (for example at the
tag `v0.15.0`).

**Versioning.** From `2026.11.0` on, the project uses
[Calendar Versioning](https://calver.org/) with the scheme `YYYY.MM.MICRO` in
its [PEP 440](https://peps.python.org/pep-0440/) form: the month has no leading
zero and there is no hyphen, `MICRO` counts the earlier releases of the same
month, and tags carry a `v` prefix. `2026.11.0` is the first release of
November 2026, `2026.11.1` the second one, `2027.1.0` the first release of
January 2027. Releases up to `0.15.0` followed
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Deprecation policy.** A calendar version carries no compatibility promise;
compatibility follows this policy instead:

- A deprecated name, parameter or behaviour keeps working and emits a
  `DeprecationWarning` that names its replacement and the earliest release in
  which it may be removed.
- It is removed no earlier than two releases **and** six months after the
  release that announced the deprecation (announced in `2026.11.0`: removed in
  `2027.5.0` at the earliest).
- Every deprecation is listed under **Deprecated** in the release that
  announces it, every removal under **Removed** in the release that performs
  it.

**Categories.** Each release lists its changes under **Added**, **Changed**,
**Deprecated**, **Removed**, **Fixed** and **Security**. An entry marked ⚠
changes numerical results; from 2026.11.0 on, every release that moves numbers
opens with a **Numbers that change** block (what moves, by how much, for whom,
and how to get the old numbers back), and every **Fixed** entry names the test
that fails on the previous release. Finding identifiers (`F01`–`F57` of review
pass 9, `10.01`–`10.95` of review pass 10) refer to the audit history.

**Contributing.** Add your entry under the matching category of the
`[Unreleased]` block; `python scripts/release.py calver` turns that block into
the section `## [YYYY.M.MICRO] - YYYY-MM-DD` of the new release.

## Releases that changed numbers

Results computed with an older version can differ from the current ones. This
table lists every release that moved numbers on purpose; the release notes
give the details.

| Release | What moved | Size of the change, or of the error it removed | Who is affected |
|---|---|---|---|
| 2026.11.0 | default line-source kernel, return currents and measurement loop, electrodes through and at a layer interface, concrete shells, floating passive clusters, `solve_mutual_matrix` (galerkin off-diagonal), touch voltages and the permissible touch voltage beyond 10 s, rho-f fits and exports, `carson_series` and other inductive kernels, the tabulated Sommerfeld assembly, `fem`, generator geometry and PEN routing | from round-off to a complete change of simulated measurement readings (see *Numbers that change* below) | nearly every study: AP1 catalogue, touch-voltage assessments, rho-f exports, inductive and measurement studies |
| 0.15.0 | `fem`, layered `mom_sommerfeld`, `coupling.layered_green`, the `image` family for unequal segment lengths, `references.pollaczek`, two `references.dwight1936` formulas, `rho_f_standard` with dead frequencies | `fem` errors up to −79 % removed; image family up to +12 %; `mom_sommerfeld` kernels of the wrong sign removed | layered and `fem` studies, mixed-length geometries, reference comparisons |
| 0.14.1 | exported groundinsight formulas of vector fits | 28 % at 16.7 Hz in the test case | BusTypes exported from fits with a negative imaginary residue |
| 0.13.0 | default `earth_inductive_model` (`sommerfeld`), two-layer inductive kernel, grids discretised with a half-integer number of segments per span | grid resistance was 1.97× to 2.89× too high; layer attribution of the reactance | inductive studies, grid electrodes |
| 0.11.0 | multiport matrices with unequal segment lengths, layer-2 field points, `sommerfeld` and `carson_series` inductive results, the series truncation criterion, vector fits with complex residues | layer-2 potential errors of +7 % and +25 % removed; Carson integrals were −36 % and +67 % off | layered studies, inductive studies, rho-f fits |
| 0.5.0 | order of `FieldResult.frequencies` for a non-increasing frequency list | order only | sweeps over unsorted frequency lists |

## [Unreleased]

The first calendar-versioned release. It implements review pass 10 against
0.15.0 (findings 10.01–10.95; [audit history, pass 10][pass-10]) and closes the
pass-9 findings F01, F02 and F32 that 0.15.0 left open. The release notes the
work streams wrote, with every measurement and test, are reproduced on the
pass-10 page. Many results move on purpose: read *Numbers that change* first.

### Numbers that change

⚠ Every change of this release that moves a result computed with 0.15.0.
Sizes are relative to 0.15.0; *golden* names the stored reference cases that
moved (`python -m tests.golden.harness report`: 222 of 453 stored quantities
in 26 of 48 cases moved, each declared with its finding, 10 are no longer
produced and 120 are new (generated mesh foundations are one electrode per
wire); 19 cases are bit-identical and 3 differ by round-off). Where a switch
reproduces 0.15.0, it is named.

| Finding | What moves | Size | Who is affected |
|---|---|---|---|
| 10.28, 10.33 | the exact line-source kernel is the default of `image` and `image_2layer` (also through the deprecated alias `cim`), of `FieldResult.potential` and of `solve_mutual_*` | 3 m rod +0.60 % (surface potential up to +1.9 %), two-layer ring +1.23 %, ring r = 2 m through `cim` +0.85 %, two rods joined by a bare conductor with 5 m segments: currents 10.9 %, cluster impedance 8.4 % (golden `defaults_rod_hom`, `defaults_two_layer_ring_auto`, `defaults_mixed_cluster_inductive`) | every image-family study, the AP1 catalogue (Z_G, potentials); `Engine(kernel="point")` reproduces 0.15.0 |
| 10.15, F02 | an electrode through the layer interface has one current per layer in `image_2layer` and `solve_mutual_*` | a 12 m rod through h₁ = 5 m was +566 % (1000/30 Ω·m), +148 % and +123 % off; now +0.11 … +0.31 % (point kernel) and +0.7 … +1.3 % (line kernel) from `mom` (golden `img2_rod_crosslayer` −46.5 %, `mutual_three_rods_crosslayer`) | rods through h₁; not the AP1 station rods (4 m from 0.6 m depth, h₁ ≥ 5 m) |
| 10.12–10.14, F08 | one concrete-shell mechanism, also in `solve_mutual_*`, the layout builder and station and cabinet foundations | AP1-like site (ρ_c = 100 Ω·m, 5 cm): house Z_G diagonal +42 / +19 / +2.8 % and network input impedance +13.6 / +8.8 / +2.2 % at ρ₁ = 30 / 100 / 1000 Ω·m; mesh foundations with `"distributed"`: +2.6 … +12 % (golden `mutual_foundations_lumped_concrete` +42.2 %) | every world with `concrete_rho_ohm_m`, including AP1 runs with concrete shells |
| 10.26, 10.29, 10.34 | `solve_mutual_matrix`: length-weighted diagonal over all members and layers, the other clusters float, and the reciprocal galerkin off-diagonal for every call (probe points are ignored with `ProbePointsIgnoredWarning` unless `offdiag="probe"`; user decision Q15) | AP1-like site: house diagonal up to 0.10 %, station 0.000 %; the probe off-diagonal missed the input impedance of the joined network by −0.15 … −0.88 % (single entries up to 41 %), galerkin reproduces it to 7e-13; AP1 catalogue path \|Z_ons\| at 50 Hz −0.05 … +0.29 % up to 20 % penetration, up to +2.67 % at 40 %; rods split at h₁ were −3.8 / −9.1 % off (golden `mutual_foundations_distributed_concrete` R 1.6 %, Z_G of four mutual cases 7.7e-4 … 8.8e-4) | AP1 mutual matrices Z_G; `offdiag="probe"` reproduces the 0.15.0 off-diagonal |
| 10.25 | `potential` and `solve_mutual_*` sum the image series with the engine's controls (was 100 terms) | K = −0.94 with `image_max_terms=2000`: potential 100 m from a ring 0.8 % | high-contrast two-layer studies with a raised term cap |
| 10.16, 10.17, F01/F30 | `Source.return_to` is honoured; measurement leads end on junction nodes at the instrument | auxiliary rod 20 m away: the probe at 61.8 % now reads the remote impedance to −1.6e-3 (0.15.0: −9.9 %); with leads the old reading was 2.8e-4 Ω instead of 1.6096 Ω (golden `gen_tn_measurement_overhead_leads`: auxiliary rod −1 A instead of +0.024 A, network potentials −1.8 %) | simulated fall-of-potential measurements (AP1 Phase B), every world with `return_to` |
| 10.29 | passive clusters float (zero net current, one potential, equalising currents) in every backend | a passive electrode in `mom`: −59 % … +1.5 %; a bonded passive pair in the image family: one potential instead of two (golden `floating_bonded_pair_two_layer` 60.5 %, `cond_coupled_leads_open_circuit` 42.5 %) | probes, auxiliary electrodes, unbonded neighbours with several electrodes, passive leads |
| 10.38 | PEN cable joints of the layout builder are leakage-free junctions | substation potential +0.07 … +0.33 % (full foundation penetration), +3.5 % at 5 % penetration (golden `gen_layout_builder`) | layout-builder worlds |
| 10.54, 10.39 | layout foundations follow the footprint; rotated mesh foundations split at their crossings | substation impedance 0 / +0.6 / +5.2 % in three test streets; mesh foundations +1.0 … +1.2 % (−9.5 % where the clamp fired) | layout and OSM worlds; `foundation_outline="spec"` reproduces 0.15.0 |
| 10.53 | reproducible draws; masks of footprints without OSM id keyed by outline | reordering the footprints moved Z by 2.8 % before | `sample_world`, masks of footprints without `osm_id` |
| 10.55 | `MeasurementLeadConfig.depth_m` and `PenConfig.depth_m` are honoured | coupled imaginary part of the probe +37 % (overhead) and +21 % (buried) at 1 kHz; inductive PEN at 1 kHz X 0.736 → 0.167 Ω; bare PEN −1.4 % (golden `gen_tn_default` ≤ 3.5e-7) | generator worlds with leads or inductive PEN |
| 10.21, F13 | touch voltage 1 m from the nearest wire of the footprint; the envelope follows that offset curve and samples no interior unless asked (`interior="meshes"` for open-air grids; user decision Q3) | 20 m × 20 m grid (100 A, 100 Ω·m): 9.92 → 69.6 V at +x, envelope 9.92 → 111.4 V; star 275 → 818 V; 10 m ring envelope 139.4 V unchanged (182.6 V in its centre with `interior="meshes"`); rotated foundation 214.4 / 93.9 V → 161.5 V (golden `img_mixed_cluster_hom`: strip ×3.2, star ×2.1) | touch-voltage assessments of grids, meshes, stars, strips, polylines, rings and rotated foundations; rods unchanged; `anchor="connection_point"` reproduces 0.15.0 |
| 10.42 | the permissible touch voltage beyond 10 s is the 80 V of the note to Figure 8 of EN 50522 (user decision Q4) | `permissible_touch_voltage_en50522(t_f > 10 s)` 85 → 80 V (−5.9 %); `math.inf` (fault not cleared) raised, now 80 V; prospective value with 1000 Ω shoes on 100 Ω·m 142.5 → 137.5 V | assessments with fault durations above 10 s, resonant-earthed networks |
| 10.18, 10.19 | `sweep`, `convergence_study` and the rho-f helpers use the input impedance | the cluster impedance of a station was 1.26 / 1.79 / 3.09 × the input impedance with 1 / 3 / 8 houses on a PEN; a ring with two bonded rods was fitted 17 % too high | sweeps of TN worlds and BusTypes exported with the rho-f helpers; the AP1 catalogue computes its own driving-point impedance |
| 10.27 | the standard rho-f form gains a constant term `k0` and relative weights | worst pointwise error 15–27 % (ring) and 43 % (AP1-like station) → 0.8–3.3 % and 6.2 % (golden `fit_rho_f_standard_sweep`) | rho-f formulas exported to groundinsight, AP1 exports included; `k0=False, weights="uniform"` reproduces 0.15.0 |
| 10.73, 10.01 | exported formulas are partial fractions with float coefficients; parsed formulas hold their numbers as floats and are evaluated without `lambdify` | at most half a unit in the last printed digit; evaluated formulas at most 4.1e-16 (exporter output) and 3.3e-15 (hand-written) | text diffs of BusType files, hand-written formulas |
| 10.20, F32 | `carson_series` on the anti-parallel image; a pair across the surface couples through the field transmitted into the soil; oblique, inclined and vertical segments (beyond 8.1°) get the Sommerfeld correction | the earth-return reactance of a 1 km PEN was 2.04 × (50 Hz) and 2.19 × (1 kHz) Carson's; 30-house network \|Z\| −16.6 % at 1 kHz (golden `cond_galvanic_neumann_carson_series` 0.60 %); an overhead lead 1 m above a PEN at 0.7 m: mutual reactance was 35.9 % low (1 kHz, 100 Ω·m); a probe lead turning 8.1° against the feed lead changed the reading 17.6-fold | `carson_series` studies, measurement leads over buried conductors |
| 10.46 | exact touching segment pairs, exact removal of the additive image, Carson's small-a series; wires closer than the sum of their radii are touching wires; close non-parallel pairs are integrated semi-analytically | Im Z +6.0 % for a conductor inclined by 30° and +18.4 % for a steep riser (default `sommerfeld`, point kernel), Re Z unchanged; at most 3.2e-6 in the golden cases; two coincident 50 m leads 1.87 H → 82 µH of mutual inductance; near-parallel pairs 1 cm apart were +200 … +250 %; a core inside a cable shield couples at the shield radius (loop inductance was +36 %); the generator's in-line fall-of-potential reading Im Z_meas 0.063 → 0.054 Ω (50 Hz, `sommerfeld`) | inductive studies with inclined, vertical or closely spaced conductors, simulated measurements with leads (AP1 Phase B) |
| 10.35, 10.36, F52 | Sommerfeld kernel for three or more layers and below h₁; λ grid of distant pairs | a three-layer earth with ρ₂ = ρ₃ was +17 … +25 % in X′; pair entries 200 m and 400 m apart were −24 % and +274 % off (golden at most 2.1e-5) | `sommerfeld` on `MultiLayerSoil`, deep conductors, large networks |
| 10.24 | inductance of lumped and single-segment Neumann conductors; leakage of a one-piece galvanic conductor; the `sommerfeld` correction of a segment longer than a quarter skin depth is integrated piecewise | 300 m PEN at 1 kHz 0.0564 + 0j Ω → 0.2096 + 3.7840j Ω (equal to its 10 m chain); two rods joined by a bare one-piece conductor 22.74 → 5.54 Ω; a lumped 3 km conductor at 1 kHz (100 Ω·m) had +13.4 % earth-return resistance against its refined chain, now ≤ 1e-4; distributed conductors with pieces longer than 40 m (1 kHz, 100 Ω·m) move by 1e-4 and more | service drops shorter than the PEN segment length, lumped leads, long lumped or coarsely divided conductors under `sommerfeld` |
| 10.45 | `references.pollaczek` uses the exact buried amplitude and the K₀ head | R′ at 1 kHz +0.15 % (1 m) and +1.0 % (3 m) in homogeneous soil, up to +1.4 % for a conductor in the upper layer and +3.5 % below the interface; X′ at most −0.09 % | callers of the Pollaczek references |
| 10.30, 10.48 | `cim` reads the length-weighted potential; equal adjacent layers are merged in every backend | a rod split unevenly at h₁ = 2 m −3.9 % in `cim`; a rod through an equal-resistivity interface +0.07 % (`mom` family), +0.42 % (`cim`); `TwoLayerSoil(ρ, ρ, h)` in `image_2layer` was 3.8e-4 (point kernel) and 3.3e-7 (line kernel) from homogeneous soil, now equal bit for bit | `cim` and declared equal layers; not AP1 |
| 10.49, 10.23 | `fem` truncation keeps its radial resolution; electrodes below the hemisphere's layer raise | layered `fem` +0.16 … +1.64 % (golden `fem_rod_two_layer_deep`); the silent +633 … +3138 % cases now raise | every layered `fem` result |
| 10.75 | `mom` and `mom_sommerfeld` honour `return_to` and junctions | the return current flows through the return electrode instead of remote earth | `mom` family worlds with two-terminal sources |
| 10.32, 10.28 | every interface cut is kept, however short the part beyond it, and the diagonal of a piece at the interface is evaluated at the wire radius, so a rod tip or head next to the interface is in contact with the layer it reaches and a wire lying at the interface is continuous across it | a 4.5 m rod (a = 8 mm) in 1000/30 Ω·m with its tip 12 mm into the lower layer: 193.3 → 137.4 Ω (−29 %), within −1.2 % of a solid-rod reference; a ring (r = 3 m) at h₁ = 5 m: 9.56 Ω (0.15.0, point kernel) → 4.18 Ω, continuous across h₁ (the point form raises there); a rod end within one wire radius of h₁ warns (`InterfaceContactWarning`, the thin-wire model has no end face) (golden: none) | rods whose tip or head lies within 16 mm of an interface, horizontal wires at an interface; not AP1 |
| 10.12, 10.39 | generated mesh foundations are one current owner per wire line at every orientation, bare or encased | bare axis-aligned mesh foundations −1.2 % (10 m, 2 × 2) to −2.5 % (30 m, 4 × 4), now +1.1 % against `mom` with the default line kernel (+0.3 … +0.5 % with the point kernel); generated networks −0.7 … −0.8 % in every node potential and up to 4.7 % in surface potentials; largest touch voltage over the default house mesh −3.6 % (golden `gen_tn_default`, `gen_layout_builder`, `gen_tn_measurement_overhead_leads`) | generator and layout-builder worlds with mesh foundations; not AP1 (lumped polygon foundations) |
| 10.38 | the PEN cables of `TnNetworkGenerator`: every cabinet keeps its own cable, a service drop taps the cable that feeds its building at the point nearest to it instead of running from the cabinet back along the main | golden networks: substation potential −2.1 %, substation leakage −3.1 … −4.8 %, house potentials −0.2 … +1.0 %, measurement network −1.1 %; input impedance of default-grid networks with ring foundations −6.4 / −1.1 / −0.2 % for 30 / 100 / 200 houses (star) (golden `gen_tn_default`, `gen_tn_measurement_overhead_leads`) | `TnNetworkGenerator` worlds; not the layout builder |
| 10.60 | the Sommerfeld correction of inductive conductors is assembled from shared tables by default (`sommerfeld_assembly="tabulated"`; user decision Q16; `"exact"` is the per-pair rule of 0.15.0) | 1e-8 … 2.4e-6 (golden `cond_lumped_long_pen_sommerfeld` ≤ 2.4e-6); closer to refined references than the per-pair rule (2e-10 against 1.2e-8 on a TN network); a 30-house network 306 s → 12 s | inductive conductors under `sommerfeld` with at least 64 sub-segment pairs per block |
| 10.32, 10.58, 10.61 | segment counts tolerant to round-off, series summation order, kernel tabulation | at most 6.9e-5 (point kernel), 7.3e-16 and 1.1e-9 | none in practice |

<!-- Release coordinator: entries from the second adversarial audit go below
     their category; numbers that move go into the table above. -->

### Security

- **BusType formulas are parsed, never executed (10.01).** `evaluate_spec`
  passed `impedance_formula` to `sympy.sympify`, so a crafted BusType file ran
  code or exhausted memory. An allow-list parser (`groundfield.io._formula`)
  builds the expression from the syntax tree, every node unevaluated,
  computes its numbers itself in bounded floating point and evaluates it
  with its own numeric function instead of `lambdify`, with bounds on
  length, nodes, depth, number size, work and the product of nested
  exponents (64); formulas are validated on load (`EvaluateSpecError`), and
  crafted documents raise `ValueError` (metadata nested deeper than
  `MAX_METADATA_DEPTH`, fields of the wrong type), so every other entry point
  works on what the loader accepts (audit A3SG-04). (A first version still let SymPy
  expand symbolic `exp` and fractional powers of large powers: 67- to
  75-character formulas ran longer than 60 s; second adversarial audit,
  A2G-01 to A2G-03.) Every formula the 0.15.0 exporter wrote still loads
  with the same value. The attack corpus `tests/data/formula_corpus.json`
  (119 accepted, 122 rejected) is shared with groundinsight's guard. Tests:
  `tests/test_formula_security.py`.
- **Result files from untrusted sources (audit A1-01).** `load_result` decodes
  only the array types the encoder writes, checks shapes and sizes before
  allocating and bounds the nesting depth; a number beyond the float range
  raises `ValueError` (audit A2S-04). Tests:
  `tests/test_result_metadata_codec.py`.
- **Overpass requests (10.05).** Validated parameters (no query injection
  through `timeout_s`), https only, a capped response (16 MiB by default),
  error payloads raise and are not cached, atomic cache files, every element
  checked before caching; payloads nested deeper than an Overpass result, ids
  beyond 64 bits and oversized integer coordinates raise `OverpassError`
  (audit A2G-05). Tests: `tests/test_overpass_ingest.py`.
- **Wheel contents (10.02).** The wheel installed `scripts/release.py` and the
  `release` command like groundinsight 0.3.0 does, so installing both
  overwrote one with the other; it now ships only `groundfield`. Test:
  `tests/test_packaging.py::test_wheel_contents`.
- **Supply chain (10.03, 10.04).** Workflows with minimal permissions and
  actions pinned to commit SHAs (Dependabot keeps them current), no persisted
  credentials, tag names only through environment variables, the release gated
  by the full CI; locked dependencies updated until `pip-audit` reports no
  known advisory.
- **Documentation site (10.08).** MathJax 3.2.2 is vendored with the site and
  pinned by checksums, and no Google Fonts are loaded: the site requests no
  third-party resource. Test: `tests/test_docs_assets.py`.
- **VTK export (10.10).** A newline in `world.name` can no longer break the
  file header; coordinates are written as `double`. Tests:
  `tests/test_io_vtk.py::test_world_name_cannot_break_the_header`,
  `tests/test_io_vtk.py::test_points_are_double_precision_at_utm_scale`.
- **Release tool (10.09).** `scripts/release.py` runs git in the repository
  root, validates everything (also against the remote) before it writes and
  restores the files if the commit fails. Tests:
  `tests/test_release_script.py`.
- **FIPS builds (10.07).** The penetration-mask hash uses
  `md5(usedforsecurity=False)`; the digests and stored masks are unchanged.
  Test: `tests/test_generator_reproducibility.py::test_mask_works_where_md5_is_blocked_for_security`.

### Added

**Solver and kernels**

- **Exact line-source kernel (10.28, 10.33):** `Engine(kernel="line" | "point")`
  and `create_engine(kernel=...)`; segments are uniform line currents in the
  reaction matrix, in `FieldResult.potential` and in `solve_mutual_*`. Results
  record `metadata["kernel"]`; `PointSource` gains an optional `direction`.
  Test: `tests/test_line_kernel.py`.
- **Backend registry (10.74):** `groundfield.solver.registry`
  (`register_backend`, `get_backend`, `available_backends`,
  `BackendCapabilities`), a capability check before every solve
  (`BackendCapabilityError`, `BackendCapabilityWarning`), backend options
  `Engine(options={...})` with `MomSommerfeldOptions` and `FemOptions`, and
  `Engine.sommerfeld_assembly`. Tests: `tests/test_backend_registry.py`,
  `tests/test_engine_options.py`.
- **Junction nodes (10.38):** `create_junction(world, position=...)`, a
  leakage-free network node for conductor ends, source terminals and
  measurement instruments; potentials in `metadata["junction_potentials"]`.
  Test: `tests/test_junction_nodes.py`.
- **`solve_mutual_matrix(..., offdiag="galerkin" | "probe")` (10.34):**
  reciprocal off-diagonal entries that reproduce a network solve of the
  joined clusters, the default of every call (user decision Q15);
  `ProbePointsIgnoredWarning` when probe points are passed without
  `offdiag="probe"`. Test: `tests/test_mutual_matrix.py`.
- **Per-engine resource limit (10.06; user decision Q17):**
  `Engine(max_segments=...)` and `create_engine(max_segments=...)` set the
  hard limit of leakage segments and nodal unknowns for one engine (default
  `diagnostics.HARD_LIMIT`, 20 000), honoured by every segment backend,
  `solve_mutual_*` and the diagnostics, recorded in
  `metadata["max_segments"]`. Test: `tests/test_resource_limit.py`.
- **Concrete shells on every electrode kind (10.12–10.14):**
  `concrete_shell_coefficient_ohm_m` on rods, rings and meshes, with the
  conversions `concrete_shell_coefficient`,
  `shell_coefficient_from_resistance` and `shell_resistance_from_coefficient`.
  Test: `tests/test_concrete_shell_coefficient.py`.
- **Geometry queries (10.79):** every electrode provides `segments`,
  `segment_count`, `wire_polylines`, `bounds`, `total_length` and
  `min_feature_length`. Test: `tests/test_electrode_geometry.py`.
- **Optional `fast` extra:** with `numba` the two-layer series and the far
  field of the line kernel run compiled, bit-identical to numpy.

**Results, diagnostics and files**

- **`FieldResult.input_impedance(source)` and `FieldResult.sources` (10.18):**
  the driving-point impedance (φ_attached − φ_return) / I a measurement or a
  fault sees, and the sources as solved. Test: `tests/test_input_impedance.py`.
- **Metadata and provenance (10.74, 10.77, F22):** every result carries a
  metadata core (backend, kernel, versions, segment count, series controls,
  solve time) and `metadata["provenance"]` with `world_fingerprint`. Test:
  `tests/test_result_persistence.py`.
- **Versioned result files (10.77, F23):** `groundfield.io.save_result` and
  `load_result` (standard JSON, schema version, atomic write). Test:
  `tests/test_result_persistence.py`.
- **`FieldResult.to_dataframe()` (10.69, 10.72)** and
  `read_csv_with_provenance(path)` (10.73). Tests:
  `tests/test_result_dataframe.py`, `tests/test_io_csv.py`.
- **`diagnose(world, engine=None, result=None)` (10.70)** returns a
  `DiagnosticsReport` (thin wires, resolution, segment budget, close or
  crossing wires, inert settings, series truncation); `World.check()` (10.66)
  validates source terminals and conductor ends; `check_segment_budget` and
  `ResourceLimitError` (10.06); `find_close_wires`, `find_close_conductors`,
  `segment_distance` (10.44); `expected_segments(...)["nodal_unknowns"]`, the
  size of the dense nodal system the guard checks (audit A2S-02);
  `groundfield.solver.PointKernelInterfaceWarning` (10.28);
  `groundfield.solver.InterfaceContactWarning` and the `diagnose()` category
  `interface_contact` (a rod end within one wire radius of an interface,
  audit A3S-05); `groundfield.world.RESERVED_NAME_PREFIXES`. Tests:
  `tests/test_diagnostics_report.py`, `tests/test_source_validation.py`,
  `tests/test_conductor_proximity.py`, `tests/test_cross_layer_point_form.py`.

**Coupling**

- **Skin effect, opt-in (10.37):** `Conductor(resistance_model="skin")` and
  `create_conductor(resistance_model=...)`; exact solid-round internal
  impedance (`solid_round_internal_impedance`). Tests:
  `tests/test_skin_effect.py`, `tests/test_inductive_solver_integration.py`.
- **Earth-return builders (10.20, 10.46, 10.60):** `image="none" | "additive" |
  "antiparallel"` on the inductance and correction builders, `touching="exact"`,
  per-segment internal inductance, multi-frequency builders
  `build_sommerfeld_correction_matrices` / `build_carson_correction_matrices`,
  and the tabulated Sommerfeld assembly (`Engine(sommerfeld_assembly=
  "tabulated" | "exact")`, tabulated by default). Tests:
  `tests/test_sommerfeld_accuracy.py`, `tests/test_sommerfeld_tabulated.py`.
- `LayeredEarth.interface_depths` and `LayeredEarth.layer_of(depth)`.
- **`coupling.carson.carson_mixed_mutual_correction` (10.20, audit A2C-1):**
  the earth-return coupling of an overhead and a buried wire (Carson's J and
  the remainder of the buried wire's attenuation, Ametani et al. 2009).
  Test: `tests/test_carson_validity_domain.py`.

**Safety, fits and export**

- **EN 50522:2022 (F51, 10.42, 10.52):**
  `permissible_prospective_touch_voltage_en50522` (Annex B.4) and
  `touch_voltage_spectrum` (true RMS over a fault spectrum). Tests:
  `tests/test_safety.py`.
- **Touch-voltage options (10.21, 10.22):** `touch_voltage(anchor=)`,
  `touch_voltage_envelope(anchor=, interior=False | "meshes" | True,
  return_points=)`, `step_voltage(world=)`, `UniformCurrentFieldWarning`.
  Tests: `tests/test_touch_voltage_footprint.py`,
  `tests/test_touch_voltage_envelope.py`.
- **Touch points (10.21; user decision Q3):** `touch_voltage_at_part(result,
  world, position=, bonded_to=, distance=1.0, ...)`, the touch voltage of an
  accessible metal part above ground (a station door, a mast, a fence post, a
  service connection box) bonded to an electrode or a PEN junction, with the
  feet on the 1 m circle around it, optionally only in given `directions` or
  outside given outlines (`exclude`); also `gf.safety.touch_voltage_at_part`.
  How-to *Assess touch voltages*. Test: `tests/test_touch_voltage_at_part.py`.
- **Effective resistivity (10.84; user decision Q6):**
  `effective_resistivity(soil, *, w0=0.13, beta=0.8, h_ref=10.0)`,
  ρ_eff = ρ₁^(1−w)·ρ₂^w with w = w0·(h₁/h_ref)^(−β) limited to [0, 1], exact
  for homogeneous soil and equal layers, canonical stacks reduced, an error
  for three or more distinct layers; bit-identical to the AP1 / AP2 scripts
  inside their calibration range. Test: `tests/test_effective_resistivity.py`.
- **Impedance quantity (10.18, 10.19):** `quantity="input" | "cluster" |
  "electrode"` and `source=` on `sweep`, `convergence_study` and the rho-f
  helpers. Test: `tests/test_impedance_quantity.py`.
- **Passivity (10.40, 10.41):** `check_passivity` and `PassivityReport`; the
  BusType export checks passivity, records it in `metadata["passivity"]` and
  raises with `require_passive=True` (`NonPassiveFitWarning`,
  `NonPassiveFitError`). Tests: `tests/test_passivity.py`,
  `tests/test_io_groundinsight.py`.
- **Fit quality (10.27, 10.41):** `RhoFStandardFit.k0`, `.max_relative`,
  `.weights`, `.quantity`, `VectorFitResult.rms_relative`, `vector_fit(rms_tol=)`,
  `RhoFStandardQualityWarning`, `VectorFitQualityWarning`; BusType documents
  record the coefficients, the fit quality and the fitted quantity. Tests:
  `tests/test_rho_f_standard.py`, `tests/test_vector_fitting.py`.
- **`sweep(n_jobs=...)` (10.64)** runs a process pool, bit-identical to the
  serial sweep. Test: `tests/test_sweep.py`.
- **Plots (10.73):** `component="real" | "imag" | "abs" | "phase"` on the
  potential plots. Test: `tests/test_plotting.py`.

**Engines and validation**

- **Engine benchmark suite (10.76):** `tests/benchmarks` (B1–B13) with
  analytic or independent reference values and one tolerance table per backend
  (`pytest tests/benchmarks --run-benchmark`).
- **`compare_engines` (10.71, 10.76):** an explicit tolerance table
  (`EngineComparison.rows`), complex impedances at every frequency,
  `component_floor`, and a note when two engines run the same computation.
  Tests: `tests/test_compare_engines.py`, `tests/test_engine_aliases.py`.
- **`mom_sommerfeld` (10.47, 10.61):** convergence and kernel-tabulation
  reports in the metadata. Test: `tests/test_mom_sommerfeld.py`.
- `FemReductionWarning`, `ExperimentalBackendWarning` and
  `ComplexImageFit.max_error` (10.23, 10.49).

**Generators**

- **Network plan (10.78):** `NetworkPlan` and `materialise_plan`;
  `TnNetworkGenerator.plan()` and `LvNetworkLayout.plan()`. Test:
  `tests/test_network_plan.py`.
- **Simulated earth tester (10.17):** `measured_impedance(result, probe=None,
  source=None)` (three-pole reading) with the terminals `meas_H` and
  `meas_S`. Test: `tests/test_measurement_loop.py`.
- **Footprints (10.54):** `foundation_outline="ombr" | "polygon" | "spec"`,
  `exclude_building_tags` with `NON_SERVICE_BUILDING_TAGS`, `rng=` on `build()`
  and `plan()` (10.53); `BuildingFootprint.mean_width_m()` and
  `.is_degenerate()`, `geo.mean_width`, `geo.MIN_OUTLINE_WIDTH_M` and
  `geo.MAX_PAYLOAD_DEPTH` (audit A2G-05, A2G-06). Tests:
  `tests/test_footprint_foundations.py`, `tests/test_overpass_ingest.py`,
  `tests/test_geo_footprint.py`.

**API**

- **Typed factories (10.68):** one overload per electrode kind, `Literal`
  option types (`ElectrodeKind`, `SourceKind`, `KernelKind`, ...) and
  “did you mean …?” errors. Test: `tests/test_api_factories.py`.
- **Namespaces (10.72):** `gf.plot`, `gf.safety` and `gf.fit`; the flat names
  keep working. Test: `tests/test_package_namespace.py`.
- `groundfield.utils`: `suggest`, `did_you_mean`, `warn_deprecated`,
  `warn_inert`, `InertConfigurationWarning`, `REMOVAL_VERSION`.
- The package ships a `py.typed` marker (10.02).

### Changed

**Versioning, packaging and tooling**

- **Calendar versioning and release tool (10.09):** `python scripts/release.py
  calver [--date YYYY-MM-DD] [--dry-run]` or `set X.Y.Z`; it updates the three
  version files and the changelog, commits and tags, and never pushes.
- **Package metadata (10.11):** PEP 621 `[project]` table, SPDX licence,
  classifiers for Python 3.12–3.14, `poetry-core>=2.2.0,<3.0.0`.
- **Development standards (10.80, 10.94, 10.95):** ruff format and lint (NumPy
  docstring convention), mypy, pre-commit, CI for lint, types, tests on
  Python 3.12–3.14 and a strict docs build on pull requests; tests are
  organised by feature and carry a `finding` marker.

**Solver**

- ⚠ **Default kernel (10.28):** `Engine.kernel` defaults to `None`, the
  backend's preferred kernel: `"line"` for the image family, also through the
  deprecated alias `cim`. A 16-house solve
  takes 2.7× (with `numba`) or 5.7× (without) the time of the point kernel.
- ⚠ **Two-terminal sources (10.16, 10.75, F01/F30):** `image`,
  `image_2layer`, `mom` and `mom_sommerfeld` inject +I at `attached_to` and −I
  at `return_to`; `fem` still returns the current through remote earth and
  warns.
- ⚠ **Floating passive clusters (10.29)** in the image family, the `mom`
  family and `solve_mutual_*`.
- ⚠ **Cross-layer electrodes (10.15, F02), slivers and segment counts (10.32),
  post-solve series controls (10.25), `solve_mutual_matrix` diagonal (10.26)**:
  see *Numbers that change*. Cross-layer solves no longer report an image
  series they did not evaluate (10.50, F43).
- ⚠ **Concrete shells (10.12–10.14):** the solver sees a shell only as the
  per-length coefficient C; `concrete_model="lumped"` and `"distributed"` put C
  on every encased wire; mesh braces are not encased.
- **`solve_mutual_matrix` / `solve_mutual_field` (10.34):** resolve and check
  the backend like `Engine.solve`; a backend outside the image family raises
  `BackendCapabilityError`.
- **Errors (10.74, audit A1-08):** worlds a backend cannot represent raise
  `BackendCapabilityError` (a `ValueError` and `NotImplementedError`) before
  the solve, also from the backend functions; code that caught `TypeError`
  from `image_2layer` or `mom` must catch `BackendCapabilityError` or
  `ValueError`.
- **Engine validation (audit A1-02, A1-04):** `Engine.model_copy(update=...)`
  and attribute assignments are validated; `create_engine` forwards only the
  arguments it is given.
- **`mom`, `bem`, `mom_sommerfeld` (10.31, 10.75):** documented and reported
  (`metadata["solver"] == "point_matching"`) as one point-matching scheme; they
  share one solve pipeline and the depth advisories of the image family.
- ⚠ **`cim` (10.30)** reads the length-weighted node potential; ⚠ **layer
  stacks (10.48)** are merged to their canonical form in every backend
  (`TwoLayerSoil(ρ, ρ, h)` is homogeneous soil, audit A2S-07); ⚠ **`fem`
  (10.23, 10.49)** is experimental, warns where its reduction does not model
  the world or ignores the skin model, and raises for electrodes below its
  hemisphere's layer, for clusters below a top layer that holds most of the
  hemisphere (audit A2E-2) and for driven polyline and star electrodes.
- **A world without a current source (10.66, audit A2S-05)** raises the same
  `ValueError` in every backend and entry point (`fem` solved it to zeros).
- **`compare_engines` (10.71)** compares complex impedances at every frequency;
  comparisons that passed before may now fail.
- **Warnings (10.70):** one warning per cause instead of per electrode;
  pre-flight warnings point at the caller's line; `ShallowSegmentWarning`
  follows the kernel (audit F08).

**Coupling**

- ⚠ **Inductive assembly (10.20, 10.46):** `carson_series` on the
  anti-parallel image, `sommerfeld` on the image-free matrix, exact touching
  pairs by default (`touching="quadrature"` reproduces 0.15.0), Carson's P and
  Q with the a² term and the asymptote only above a = 15. `carson_series`
  keeps Carson's per-metre formula for near-parallel, near-horizontal pairs
  on one side of the surface; a pair across the surface has no anti-parallel
  baseline and couples through the transmitted field, and oblique, inclined
  and vertical segments get the Sommerfeld correction of a homogeneous earth
  (at the 8.1° boundary the result steps by the error of the per-metre model,
  12 % for two 1 km leads at 1 kHz). Wires closer than the sum of their radii
  are evaluated as touching wires, close non-parallel pairs semi-analytically,
  and `sommerfeld` integrates segments longer than a quarter skin depth
  piecewise (audit A2C-1, A2C-2).
- ⚠ **`references.pollaczek` (10.45)** is exact for two layers, also for
  conductors below the first interface and pairs in different layers.
- **One earth-return assembly for all frequencies (10.60):** bit-identical per
  frequency; a 30-house Carson network 16.4 s → 3.3 s.
- ⚠ **Tabulated Sommerfeld assembly by default (10.60; user decision Q16):**
  one table of the spectral integral per pair of depth groups and frequency;
  blocks of fewer than 64 sub-segment pairs keep the per-pair rule, which
  stays available as `sommerfeld_assembly="exact"`. Test:
  `tests/test_sommerfeld_assembly_default.py`.

**Safety, fits and export**

- ⚠ **Rho-f defaults (10.18, 10.19, 10.27):** input impedance as the fitted
  quantity; standard form with `k0` and relative weights.
- ⚠ **Formulas (10.73):** partial fractions with float coefficients rounded
  once to `decimals` significant digits; one renderer for `fit_to_sympy`,
  `fit_to_sympy_standard` and the exporter; exports take 0.04–0.14 s instead
  of 0.39–2.23 s.
- ⚠ **One passivity check (10.40):** the export and `check_passivity` check 0 Hz
  to twice the highest fitted frequency; fits that passed may now fail.
- ⚠ **Parsed formulas (10.01):** `parse_impedance_formula` returns an
  unevaluated expression whose numbers are 17-digit floats (`log(2)` is
  `0.69314718055994529`); compare parsed expressions by value. Nested powers
  whose exponents multiply to more than 64 (`((f**8)**8)**2`, `(f*10**40)**2`:
  write `1e40`), `log` inside `exp` and fractional powers of large exact
  numbers are rejected; `evaluate_spec` gives the principal branch where
  `lambdify` gave `nan`. groundfield's exporters write none of these.
- ⚠ **`rho` of a BusType in layered soil (10.84; user decision Q6)** is the
  effective resistivity `effective_resistivity(soil)`, the
  `specific_earth_resistance` an AP2 bus carries: pass it as `rho_at_fit` of a
  vector fit and fit standard forms over ρ_eff samples. Documents record
  `metadata["rho_definition"] = "effective_resistivity"`. A compact electrode
  in the upper layer follows ρ₁ more closely than the network calibration of
  ρ_eff (station of the IO example: standard form within 13.4 / 8.3 / 3.6 %
  over ρ_eff, 3.9 / 1.9 / 0.6 % over ρ₁ for h₁ = 5 / 10 / 30 m); check
  `max_relative`, or pass group-specific `w0`, `beta`.
- ⚠ **Touch-voltage envelope (10.21; user decision Q3)** samples no interior
  by default: inside the footprint stands the station, the house, the steel
  structure or the mast; touch points are defined with
  `touch_voltage_at_part`, open-air grids are sampled with
  `interior="meshes"`.
- ⚠ **Permissible touch voltage beyond 10 s (10.42; user decision Q4):** 80 V
  for every fault duration above 10 s, including `math.inf`, after the note
  to Figure 8 of EN 50522:2022 (the curve ends at 10 s);
  `permissible_prospective_touch_voltage_en50522` uses the same value.
- **CSV (10.69, 10.73):** provenance header, a `frequency_hz` column, no
  silent overwrite (`overwrite=True`).
- **Results JSON (10.77):** standard JSON (non-finite numbers as strings),
  engine JSON with `options`; 0.15.0 files still load.
- **`split_factor` (10.18)** divides by the solved source current; plot labels
  name the drawn component (10.73).

**Generators**

- ⚠ **Measurement loop (10.17):** feed and probe leads end on the junctions
  `meas_H` and `meas_S` at the substation; `LvNetworkLayout` readings use the
  solved test current.
- ⚠ **Layout builder (10.38, 10.54):** PEN joints are junctions; foundations
  follow the footprints.
- ⚠ **Mesh foundations (10.12, 10.39, audit A2G-04):** one wire electrode per
  mesh line at every orientation, bare or encased (`<name>`, `<name>_w<j>`);
  `electrode_currents[<name>]` and `grounding_impedance(<name>)` refer to the
  first wire, use `cluster_impedance` for the foundation. An axis-aligned
  `"lumped"` ring stays one `grid_mesh`, a polygon outline one closed wire.
- ⚠ **PEN cables of `TnNetworkGenerator` (10.38):** every cable cabinet has its
  own cable, as the topology documents (cabinets on one line: separate cables
  in one trench); a service drop taps the cable that feeds its building at the
  point nearest to the building (a leakage-free junction `pen_tap_<k>` that
  splits the cable into sections `<cable>_s<k>`), never another drop; no
  section or drop is shorter than 1 cm. A drop may share its route with other
  cables, which the inductive coupling treats as touching wires (10.46).
- **`TnNetworkGenerator.build()` of large networks:** `World` keeps a name
  index (O(1) per added object) and the materialiser passes objects instead
  of names: 1000 houses about 7 s → 2.4 s, worlds bit-identical.
- ⚠ **`sample_world` (10.53)** resolves every nested distribution once per
  realisation.
- ⚠ **Depths (10.55):** measurement leads and PEN sections run at their
  configured depth.
- **One network plan and one materialiser (10.78):** both front-ends produce a
  `NetworkPlan`; the layout module became the package
  `groundfield.generators.layout`.
- **English names (10.69):** see *Deprecated*; generated element names
  (`trafo_*`, `kvs_*`, `house_*`) are unchanged.
- **Overpass:** retries on HTTP 503 honour `Retry-After`; `max_retries` is at
  most 5; the response cap is 16 MiB (pass `max_response_bytes` for wider
  areas); element ids are 64-bit integers, a non-finite or negative
  `building:levels` tag gives `levels=None`, outlines without area are
  skipped.

**API and performance**

- **Soil parameters (10.69):** `HomogeneousSoil.rho`, `SoilLayer.rho`,
  `SoilLayer.h`; soil models validate assignments and reject unknown keys in
  `model_copy` (10.67).
- **Import time (10.72):** heavy dependencies load on first use; median
  `import groundfield` 798 ms → 223 ms.
- **Memory and speed (10.56–10.65), bit-identical unless marked:** kernels
  without `(N, N, 3)` tensors (30-house network 1 248 → 115 MiB peak; an
  80-house network with 9 712 segments solves in about 2 s); one
  reaction-matrix assembly per solve; `solve_mutual_matrix` 23–31× faster;
  ⚠ two-layer series summed before the product (round-off, 10–15× faster on a
  30-house network); chunked field evaluation; multi-frequency earth return;
  ⚠ tabulated `mom_sommerfeld` kernels for three or more layers (66.6 → 0.30 s
  on a 40 m grid).

**Documentation (10.82–10.93)**

- The documentation follows the Diátaxis structure: tutorials T01–T10 as
  executed notebooks with value checks (`tests/test_tutorials.py`), how-to
  guides, a reference with glossary (English and German), units and
  conventions, warnings catalogue, known limitations and deprecations, an
  explanation section with one theory page per engine and a validation page
  generated from the benchmark suite, the [audit history][audit-history] and
  the [roadmap][roadmap]. Versioned with `mike`.
- Formulas on the documentation site are typeset once; pages that load a TeX
  extension no longer fail.
- Moved pages (no redirects): `api/*` → `reference/api/*`, `examples/*` →
  `tutorials/` and `how-to/`, `engines/*` → `explanation/engines/`,
  `concepts.md` → `explanation/concepts.md`, `performance.md` →
  `explanation/performance.md`, `installation.md` and `quickstart.md` →
  `getting-started/`.

### Deprecated

Every name below keeps working, emits a `DeprecationWarning` that names its
replacement, and is removed in `2027.5.0` at the earliest.

| Deprecated | Replacement |
|---|---|
| backend `cim` (10.76) | `image` (auto-dispatch; `cim` already ran `image_2layer`, whose default kernel is now `"line"`, see *Numbers that change*) |
| backend `bem` (10.76) | `mom` (identical point-matching system) |
| backend `image_nlayer` (10.76) | `image` (auto-dispatch by soil) |
| `Engine.tolerance`, `Engine.max_iterations` (10.51) | none, they have no effect |
| `Engine.sommerfeld_lambda_max_factor`, `sommerfeld_epsabs`, `sommerfeld_epsrel`, `sommerfeld_max_osc_panels` (10.74) | `Engine(options={"mom_sommerfeld": {...}})` or `MomSommerfeldOptions` |
| `World.solve(snapshot_sources=False)` (10.72) | none, every entry point takes the snapshot |
| `World.concrete_shell_corrections` (10.12–10.14) | `concrete_shell_coefficient_ohm_m` of the electrodes |
| `HomogeneousSoil(resistivity=...)`, `SoilLayer(resistivity=...)` (10.69) | `rho` |
| `SoilLayer(thickness=...)` (10.69) | `h` |
| module `groundfield.generators.ortsnetz_builder` (10.69) | `groundfield.generators.layout` |
| `OrtsnetzLayout` | `LvNetworkLayout` |
| `KvsPlacement`, `KvsConfig` | `CabinetPlacement`, `CabinetConfig` |
| `StarKvsTopology`, `kind="star_kvs"` | `StarCabinetTopology`, `kind="star_cabinet"` |
| `TnNetworkConfig.kvs`, `TnNetworkConfig.source_magnitude_A` | `TnNetworkConfig.cabinets`, `TnNetworkConfig.source_magnitude_a` |
| `CabinetConfig.quote_per_100_buildings` | `CabinetConfig.count_per_100_buildings` |
| `RadialTrunkTopology.slots_per_kvs`, `.kvs_spacing_m`, `.plan_feeder_kvs_positions()` | `.slots_per_cabinet`, `.cabinet_spacing_m`, `.plan_feeder_cabinet_positions()` |
| `LvNetworkLayout.add_kvs()`, `.kvs_placements` | `.add_cabinet()`, `.cabinet_placements` |
| `from_osm(kvs_lat_lons=, kvs_xys=)`, `to_world(kvs_grounding=)` | `cabinet_lat_lons=`, `cabinet_xys=`, `cabinet_grounding=` |
| `source_magnitude_A=` of `to_world`, `measured_grounding_impedance`, `verify_current_balance`, and the `*_A` result keys of `verify_current_balance` | `source_magnitude_a=` and `*_a` keys |
| `HomogeneousSoilSpec.resistivity`, `SoilLayerSpec.resistivity`, `SoilLayerSpec.thickness_m` | `rho`, `rho`, `h` |

Configurations and worlds written by 0.15.0 still load. `solve_cim` and
`solve_bem` warn and delegate; their results carry the metadata of the backend
that ran. `fem` is registered as experimental (`ExperimentalBackendWarning`).

### Removed

- **From the wheel (10.02, 10.11):** the console scripts `release` and
  `generate-third-party-licenses`, the top-level `scripts` package and the
  unused `polars` dependency (groundinsight 0.3.0 still installs it with the
  `groundinsight` extra); run the tools from a checkout
  (`python scripts/release.py ...`).
- **Release tool options (10.09):** the `major` / `minor` / `patch` bumps and
  `--no-push`, `--allow-dirty`, `--allow-empty-changelog`.
- **Metadata of `cim` results (10.76):** `cim_fit_used`, `cim_n_images`,
  `cim_rms`, `reduces_to`, `n_layers`, `h`, `rhos`.
- **Private names (10.75, 10.79, 10.81):** `solver.image._potential_kernel`,
  `_segment_z_extent`, the `_discretize_*` helpers, `_grid_segments` and the
  electrode classes and `PointSource` that `solver.image` re-imported;
  `image_2layer._two_layer_potential_kernel`; the module `solver.field_study`;
  `Engine._stub_result` and `_dispatch`; `_layered.cylindrical_radius` and
  `_split_layer_thicknesses`; `mom._carson_skin_metadata`,
  `bem._build_Z_collocation`, `cim._cim_self_kernel_factory`,
  `_galerkin_solve` of `bem` and `mom_sommerfeld`, the private Pollaczek
  kernels of `coupling.sommerfeld_inductance` and the helpers the solver
  modules re-imported from each other. Import public names from `groundfield`
  or `groundfield.geometry`.
- `gf.annotations` (a leaked `__future__` object, 10.72).

### Fixed

**Inputs that solved silently**

- **Source references (10.66):** misspelled `attached_to` / `return_to`, a
  source on a conductor, a source returning to its own node, no current
  source, magnitude 0 and dangling conductor ends raise with the closest
  names. Tests: `tests/test_source_validation.py`.
- **`VoltageSource` solved to zero (10.51)** and now raises
  `NotImplementedError` naming the alternative. Test:
  `tests/test_source_validation.py::test_voltage_source_raises_not_implemented_naming_the_alternative`.
- **`HARD_LIMIT` only warned (10.06):** worlds beyond the segment limit, or
  whose dense nodal system with every conductor sub-piece exceeds it (audit
  A2S-02: a 100 m PEN in 2000 pieces was predicted at 12 segments), raise
  `ResourceLimitError` before any allocation. Test:
  `tests/test_diagnostics_report.py::test_world_beyond_the_hard_limit_is_rejected_before_any_allocation`.
- **Conductors (10.43, 10.67):** non-positive or non-finite cross sections and
  resistivities raise; a cross section above 1e-2 m² warns (mm² given?); an
  ideal conductor with `discretize_segment_length` raises. Tests:
  `tests/test_conductor_validation.py`.
- **Soils and geometry (10.67):** infinite resistivities and thicknesses,
  malformed `MultiLayerSoil` stacks and degenerate electrode geometry raise
  when built. Tests: `tests/test_soil_models.py`,
  `tests/test_electrode_validation.py`.
- **Inert settings (10.51, F57):** non-default boundary conditions, soil
  permittivity and engine settings that no backend reads warn. Tests:
  `tests/test_boundary_conditions.py`, `tests/test_engine_options.py`,
  `tests/test_soil_specs.py`.
- **Engine assignments and copies (audit A1-02, A1-04):** `engine.frequencies
  = []` and a raw `options` dict were accepted, and deprecated flat names in
  `model_copy` were ignored. Tests: `tests/test_engine_validation.py`.
- **Names used twice in a loaded or edited world (10.66, audit A2S-06)** failed
  with a singular matrix or merged an electrode and a junction, and names with
  the solver's internal prefixes `__layer_` / `__cond_` broke the solve (audit
  A3S-03); `World.check()` and the `add_*` methods reject both. Tests:
  `tests/test_source_validation.py::test_check_reports_names_used_twice`,
  `tests/test_source_validation.py::test_check_rejects_a_name_with_a_reserved_prefix`.

**Solver**

- **Reciprocity (10.28, 10.29):** transfer impedances of the `mom` family
  (were up to 1.46 Ω apart) and of the line kernel across the interface (were
  1.1–4.7 % non-reciprocal) are reciprocal. Tests:
  `tests/test_mom_family.py::test_two_port_transfer_impedance_is_reciprocal`,
  `tests/test_reciprocity.py`.
- **`cim` is bit-identical to `image_2layer` (10.30)** of the same release,
  default line kernel included (against 0.15.0 both move, see *Numbers that
  change*). Test:
  `tests/test_cim.py::test_cim_is_bit_identical_to_image_2layer_for_unequal_segments`.
- **Layer stacks (10.48):** physically two-layer or homogeneous
  `MultiLayerSoil` stacks no longer raise. Tests: `tests/test_layer_stack.py`,
  `tests/test_layer_stack_dispatch.py`.
- ⚠ **Interface slivers and contact (10.32, audit A2S-01):** a rod tip just
  below h₁ no longer raises or shorts the `mom` solve, and the contact with
  the layer it reaches is kept: the first fix dropped cuts shorter than the
  wire diameter, so a 4.5 m rod in 1000/30 Ω·m jumped from 195.0 to 128.6 Ω
  between 15.9 and 16.1 mm of tip penetration (audit A2S-01). The diagonal of
  a piece at the interface clamped its closed-form image terms at h₁/200, so a
  ring lying at h₁ = 5 m read −5.29 Ω with the default engine and contact
  values were 10–32 % low (audit A3S-01); it is evaluated at the wire radius.
  Tests: `tests/test_interface_slivers.py`,
  `tests/test_cross_layer_line_kernel.py::test_ring_resistance_is_continuous_through_a_deep_interface`.
- ⚠ **Point form next to the interface (10.28):** on the cross-layer path a
  horizontal wire next to h₁ gave wrong, even negative impedances with
  `kernel="point"` and the `mom` family (−14 % … −552 %, also in 0.15.0) or a
  bare `ValueError`; it raises `BackendCapabilityError` naming the line
  kernel, and one within 0.6 segment lengths warns
  (`PointKernelInterfaceWarning`), on both sides of the interface (a ring 1 mm
  above h₁ read −945 Ω; audit A3S-02). Test:
  `tests/test_cross_layer_point_form.py`.
- ⚠ **Field of equal-resistivity layers (10.48, audit A3S-04):**
  `FieldResult.potential` and `solve_mutual_field` of `TwoLayerSoil(ρ, ρ, h)`
  were up to 21 % off next to wires below h; they read the canonical stack
  and equal the homogeneous field. Test:
  `tests/test_layer_stack.py::test_field_of_an_equal_resistivity_stack_is_the_homogeneous_field`.
- **Floating clusters in `solve_mutual_*` (10.29, audit A2S-03):** one
  factorisation for all excitations (256 bonded nodes 199.9 s → 5.6 s,
  results equal to 1e-12). Test:
  `tests/test_mutual_matrix.py::test_shared_factorisation_equals_one_floating_solve_per_anchor`.
- **`mom` family above ground (10.75):** a segment crossing the surface
  raises instead of being solved as buried. Test:
  `tests/test_point_matching_pipeline.py::test_mom_family_rejects_a_segment_partly_above_the_surface`.
- **`mom_sommerfeld` budget (10.47):** a binding oscillation budget counts as
  non-convergence. Test:
  `tests/test_mom_sommerfeld.py::test_kernel_warns_or_converges_when_the_oscillation_budget_binds`.
- **`fem` warnings (10.23, audit F01):** interfaces above or beside the
  electrode are reported. Tests: `tests/test_fem.py`.
- ⚠ **`fem` below a thin top layer (10.23, audit A2E-2):** a 3 m rod below
  h₁ = 0.4 m in 1000/30 Ω·m read 31.1 Ω against 10.5 Ω (`mom`) with a warning;
  it raises, and driven polyline and star electrodes raise
  `BackendCapabilityError` instead of a bare `TypeError`; `fem` warns that it
  ignores `resistance_model="skin"` (audit A2C-4). Tests: `tests/test_fem.py`,
  `tests/test_backend_registry.py::test_a_backend_that_ignores_the_skin_model_says_so`.
- **`fit_complex_images` (10.49)** no longer reports convergence for a fit
  that misses Γ₁. Test: `tests/test_cim.py::test_fit_converged_flag_bounds_the_maximum_error`.
- **`compare_engines` (10.71, audit F03)** sees the reactance, every frequency,
  non-finite results and a zero of one engine. Tests:
  `tests/test_compare_engines.py`.
- **Resolution check (F53, F54):** polyline and star electrodes are checked.
  Test: `tests/test_diagnostics_report.py::test_resolution_check_covers_star_and_polyline`.
- **`expected_segments` for layer stacks (10.79, audit A2S-07)** cuts where the
  backends cut. Test:
  `tests/test_diagnostics.py::test_expected_segments_matches_the_solve_for_every_layer_stack`.
- **`convergence_study` (A1-02, audit A2S-08)** shows each warning once and
  rejects non-finite segment lengths. Tests: `tests/test_convergence.py`.
- **Line-kernel cost (10.28):** one long segment no longer slows down the
  whole near-pair search (16 houses: 29 s and 1.7 GiB before). Test:
  `tests/test_line_kernel_near_pairs.py`.
- **Engine knobs (10.74):** `fem` options are reachable, options for another
  backend warn, `Engine.solve` keeps the caller's sources (10.72). Tests:
  `tests/test_engine_options.py`, `tests/test_solve_entry_points.py`.
- **Memory (10.56, 10.57, 10.59, 10.62, 10.65):** see *Changed*. Tests:
  `tests/test_kernel_performance.py`.

**Coupling**

- ⚠ **`carson_series` baseline (10.20, F32).** Test:
  `tests/test_inductive_solver_integration.py::test_carson_series_reactance_matches_carson`.
- ⚠ **`carson_series` across the surface and outside its domain (10.20, audit
  A2C-1):** an overhead wire over a buried one was coupled through an image
  above ground (mutual reactance 35.9 % low for a lead 1 m above a PEN at
  0.7 m, 1 kHz), and oblique pairs got the per-metre value once per segment
  pair, which grew with the number of segments (two 1 km leads at 9°: 20× the
  Sommerfeld coupling). Tests: `tests/test_carson_validity_domain.py`.
- ⚠ **Wires lying side by side (10.46):** coincident or overlapping parallel
  conductors had no finite mutual inductance (two coincident 50 m leads read
  1.87 H), which made a two-house network with drops along the PEN main
  non-passive. Where a point of one wire lies beside the other closer than
  r_a + r_b, the touching distance is used, pointwise inside the Neumann
  integral and independent of the order of the conductors (a first version
  switched whole pairs from the row segment's view: +38.7 % depending on the
  order, indefinite matrices; audit A3C-1, A3C-2); a thinner wire inside a
  thicker one couples at the thicker radius (core in a shield: the loop was
  +36 %, audit A3C-5). Tests: `tests/test_interpenetrating_wires.py`.
- ⚠ **Close non-parallel pairs (10.46):** nearly parallel, crossing, skew and
  T-junction pairs were integrated with a 16 × 16 rule that does not resolve
  1/r (+12 % at 10 cm, up to +250 % at 1 cm, −13 % for T-junctions); they are
  integrated semi-analytically, to 5e-11 for directions that differ by more
  than 1e-6 rad (1.0e-9 at 1e-8 rad); filaments from one vertex at tiny
  angles were up to 16 % off (audit A3C-4). Tests:
  `tests/test_neumann_close_pairs.py`, `tests/test_neumann_touching.py`.
- ⚠ **`carson_series` and the segment order (10.20, audit round 3):** a
  near-parallel pair was projected on the axis of its first segment, so the
  order of the segments moved entries by up to 3.1 % of the largest one;
  such pairs are projected on the bisector (two 1 km leads 2 m apart: +2e-5
  at 0.5° … +6.4e-3 at 8°). Test:
  `tests/test_carson_validity_domain.py::test_carson_tiling_does_not_depend_on_the_segment_order`.
- ⚠ **Long segments under `sommerfeld` (10.24, audit A2C-2):** one 16 × 16 rule
  over a segment much longer than the skin depth put a lumped 3 km PEN's
  earth-return resistance +13.4 % high at 1 kHz (+34.9 % at 5 km), without a
  warning; the lumped 3 km PEN now matches its refined chain to 1e-4, and
  the composite rule stays within 5.9e-4 (resistance) and 2.4e-4 (reactance)
  of a refined rule up to 3 km, with a step of 3.5e-4 at a quarter skin
  depth (audit A3C-4). Tests: `tests/test_sommerfeld_long_segments.py`.
- **Cost of the exact touching-pair rule (10.46, audit A2C-5):** the coaxial
  closed forms are evaluated in one vectorised call per segment, bit for bit
  (2 km PEN solve 2.2–2.6 s → 1.1–1.4 s). Tests:
  `tests/test_neumann_touching.py`.
- ⚠ **Sommerfeld kernel (10.35, 10.36)** for three or more layers, conductors
  below h₁ and distant pairs; `SommerfeldResolutionWarning` instead of silent
  truncation. Tests: `tests/test_sommerfeld_accuracy.py`.
- ⚠ **`earth_return_correction_layered` sign (10.46, F50).** Test:
  `tests/test_sommerfeld_accuracy.py::test_layered_point_kernel_has_the_exact_sign`.
- ⚠ **Carson P and Q (10.46):** the a⁻⁷ term of the Q asymptote, the small-a
  series and the θ = 0 quadrature. Tests: `tests/test_carson_pq.py`.
- ⚠ **Lumped and single-segment conductors (10.24):** the Neumann inductance
  of a lumped conductor and the leakage of a one-piece galvanic conductor.
  Tests:
  `tests/test_inductive_solver_integration.py::test_neumann_is_honoured_on_lumped_and_single_segment_conductors`,
  `tests/test_conductors_distributed.py::test_a_single_segment_galvanic_conductor_leaks`.
- **Conductors at the surface (10.46, F52):** crossing or lying in the surface
  plane is solved under `sommerfeld` and raises with the image models. Tests:
  `tests/test_inductive_solver_integration.py::test_sommerfeld_accepts_a_surface_crossing_conductor`.
- **`segment_distance` with a point segment (10.44, audit F04).** Test:
  `tests/test_conductor_proximity.py::test_segment_distance_with_point_segments`.
- **Docstrings (10.95):** invalid escape sequences and control characters.
  Tests: `tests/test_coupling_conventions.py`, `tests/test_source_escapes.py`.

**Safety, fits and export**

- ⚠ **Touch voltage at the electrode centre (F13, 10.21),** also for rotated
  foundations (audit H4). Tests:
  `tests/test_touch_voltage_footprint.py::test_grid_envelope_covers_the_one_metre_offset_curve`,
  `tests/test_touch_voltage_footprint.py::test_touch_voltage_of_a_rotated_foundation_matches_the_unrotated_one`.
- **Uniform-current caveat (10.22):** touch and step voltages near grids,
  meshes, stars, strips and bonded clusters from the image family warn. Test:
  `tests/test_touch_voltage_footprint.py::test_uniform_current_backend_warns_once_per_call_near_a_grid`.
- **EN 50522 anchors (F44, 10.42):** checked against literals of EN 50522:2022
  Table B.4 (values unchanged); docstrings cite the 2022 edition. Test:
  `tests/test_safety.py::test_en50522_docstrings_cite_the_current_edition`.
- **`touch_voltage_spectrum` (10.52, audit L4)** rejects a frequency given
  twice. Test:
  `tests/test_safety.py::test_touch_voltage_spectrum_rejects_a_frequency_given_twice`.
- **Vector fit with a 0 Hz sample (10.41):** no collapse to Z ≈ 0. Test:
  `tests/test_vector_fitting.py::test_dc_sample_with_imaginary_part_does_not_collapse_the_fit`.
- **Export errors and metadata (10.84, 10.27, 10.19):** an actionable error
  without `rho_at_fit`, the running version in the BusType metadata, every
  non-finite number of a BusType document as `null` (audit A2G-08). Tests:
  `tests/test_io_groundinsight.py::test_vector_fit_without_rho_at_fit_raises_an_actionable_error`,
  `tests/test_io_groundinsight.py::test_metadata_of_hand_built_fits_is_strict_json`,
  `tests/test_io_groundinsight.py::test_unknown_fit_quality_and_samples_are_written_as_null`.
- **Result JSON (10.77):** float keys, infinities and numpy values in the
  metadata survive the round trip; result files honour the umask (audit
  A1-05). Tests: `tests/test_result_persistence.py`.

**Generators**

- ⚠ **Measurement leads (10.17, 10.55, F48):** leads no longer short-circuit
  the instrument and run at their depth. Tests:
  `tests/test_measurement_loop.py::test_resistive_leads_do_not_change_the_reading`,
  `tests/test_measurement.py::test_buried_lead_runs_at_its_depth`.
- ⚠ **Concrete shells (10.12–10.14, F08):** the lumped shell outside TN
  service drops, rods and braces inside a shell, substation and cabinet shells,
  and the layout builder. Tests: `tests/test_concrete_one_mechanism.py`.
- ⚠ **PEN joints and depth (10.38):** joints no longer leak, `PenConfig.depth_m`
  acts. Tests: `tests/test_pen_junctions.py`, `tests/test_pen_depth.py`.
- ⚠ **Foundations (10.39, 10.54, F28):** rotated meshes without the clamp,
  layout foundations from the footprint, no shift of the random draws,
  degenerate footprints get a minimum foundation. Tests:
  `tests/test_rotated_foundations.py`, `tests/test_footprint_foundations.py`.
- ⚠ **Mesh foundations depended on their orientation and on an absent shell
  (10.12, 10.39, audit A2G-04):** the same bare mesh gave 4.796 Ω at 0° and
  4.738 Ω at 30°, and a shell with ρ_c = ρ_soil lowered the largest touch
  voltage of a 10 m mesh from 232.3 to 221.2 V (100 A). Tests:
  `tests/test_foundation_owners.py`.
- ⚠ **Overlapping PEN cables (10.38):** service drops ran from the cabinet back
  along the main, coaxial with it, which made an inductive 2-house network
  non-passive (Im Z_in −0.633 Ω at 1 kHz); drops now tap their feeding cable,
  and conductors lying side by side are touching wires (10.46). Tests:
  `tests/test_pen_routing.py`, `tests/test_interpenetrating_wires.py`.
- **Outlines without area (10.05, 10.54, audit A2G-06)** crashed the
  generators; they are skipped on ingest and widened with a warning by the
  generators, and polygon foundations of outlines narrower than 1 m are
  widened like the OMBR ones (audit A3SG-07). Tests:
  `tests/test_geo_footprint.py`, `tests/test_footprint_foundations.py`.
- **Integer fields of a resolved config (10.53)** hold `int`. Test:
  `tests/test_generators_base.py::test_resolved_integer_fields_hold_integers`.
- **Reproducibility (10.53, F07):** resolved configs replay their realisation;
  masks ignore the footprint order. Tests:
  `tests/test_generator_reproducibility.py`.
- **`ManhattanGridPlacement(jitter_m=<distribution>)` (10.55, F47).** Test:
  `tests/test_placement.py::test_manhattan_grid_jitter_accepts_a_distribution`.
- **Malformed Overpass payloads (10.05, audit M1, A2G-05)** raise
  `OverpassError` before caching, also deeply nested ones (`RecursionError`
  before). Tests: `tests/test_overpass_ingest.py`.
- **Layouts and generator configs pickled with 0.15.0 (10.69, audit A2G-07)**
  still load and build. Tests: `tests/test_generator_pickles.py`,
  `tests/test_generator_naming.py::test_layout_pickled_with_v0150_still_builds`.


### Corrections to the 0.15.0 release notes

The 0.15.0 notes contain statements that review pass 10 found to be wrong
(finding 10.82). They are left as published and marked with a footnote:

- **F47, F48, F53, F54** were listed as fixed, but `generators/placement.py`
  and `generators/measurement.py` were unchanged since 0.14.1 and
  `diagnostics.py` since 0.13.0. They are fixed in this release (see *Fixed*).
- **`cim` bit-identical to `image_2layer`:** false for electrodes with unequal
  segment lengths, +4.1 % for a 3 m rod in `TwoLayerSoil(100, 400, h_1=2)` at a
  segment length of 0.4 m (71.060 Ω against 68.262 Ω). Fixed in this release
  (10.30).
- **F16** quoted the symptom of F08: the layout builder still ignored the
  concrete shell; the 0.15.0 fix covered `TnNetworkGenerator` only. Fixed in
  this release (10.14).
- **F17** was fixed only in part: the engine-comparison example still
  recommended `image_nlayer` or `cim` for more than two layers, which both
  raise. The documentation was rewritten in this release.

---

## [0.15.0] - 2026-07-31

The rest of review pass 9 ([audit history, pass 9][pass-09]). F01, F02 and
F32 stayed open (closed in 2026.11.0).

⚠ Numbers change for every `fem` result, layered `mom_sommerfeld`,
`coupling.layered_green` near the surface and the interface and beyond about
160 m, the `image` family for electrodes with unequal segment lengths (up to
+12 %), `references.pollaczek`, two `references.dwight1936` formulas, and
`rho_f_standard` fits of sweeps with dead frequencies.

### Added

- `Engine.sommerfeld_lambda_max_factor`, `sommerfeld_epsabs`,
  `sommerfeld_epsrel`, `sommerfeld_max_osc_panels`.
- `SommerfeldConvergenceWarning`, `SommerfeldTailTruncationWarning`,
  `ShallowSegmentWarning`.
- Executable documentation: a test runs every Python block of the
  documentation (F27).

### Changed

- `cim` and `bem` no longer compute a complex-image fit; “`cim` returns values
  bit-identical to `image_2layer` and `bem` to `mom`” (F34, F39).[^c-cim]
- `coupling.layered_green`: `lambda_max_factor` is an upper bound on the
  truncation.
- `generators`: `MeasurementLeadConfig.depth_m` is honoured (F48).[^c-f48]
- Faster: `layered_green` over kilometres (199 → 71 ms), `mom_sommerfeld`
  150–8000×, `fem` about 15×.

### Deprecated

- The `fem` option `z_far_factor` (`r_far_factor` sets the truncation sphere).

### Fixed

- ⚠ `mom_sommerfeld`: the top-layer kernel no longer truncates a
  conditionally convergent integral (a 25 m ring 1.4952 → 1.5751 Ω), and an
  unresolved oscillation no longer returns a negative kernel silently (F04,
  F40).
- ⚠ `coupling.layered_green`: the λ truncation is bounded by construction
  (errors up to −39.5 % near the interface before) and the 64-pair
  interpolation step is gone (F14, F15).
- ⚠ `image`, `image_2layer`: reported potentials are the length-weighted
  average the solver enforces, independent of electrode names; mixed-length
  geometries move by up to +12 % (F10, F11).
- ⚠ `fem`: the mesh no longer degrades with the declared layer thickness (up
  to −79 % before), and the Dirichlet electrode is conforming (F05, F06, F24).
- ⚠ `references.pollaczek` used Carson's air exponent (R′ −0.5 … −5.8 % in
  band); the 0.14.0 notes had described this as physics (F09).
- ⚠ `references.dwight1936`: `horizontal_strip` (about 10 % low) and
  `vertical_round_plate` (F37, F38).
- ⚠ `rho_f_standard`: dead frequencies are no longer fitted as Z = 0 (F41).
- The `z <= 0` check covers every backend and the segment extent (F26, F29).
- `cluster_impedance` sums the current of every member; `layered_green` raises
  for negative depths; `Engine.solve` forwards the series controls to
  `image_nlayer` (F18); `World.solve` keeps the `Source` objects (F31);
  `compare_engines` uses a pairwise metric and reports an empty comparison as
  inconsistent (F35, F36).
- Generators: the `building_counts` order (F33), the OMBR centre (F25), the
  escape radius of `route_manhattan` (F42), and “the ADR-0012 concrete shell is
  no longer silently dropped (F16)”.[^c-f16]
- `io.csv` column constants (F46); `fem.r_far_factor` acts.
- Documentation: the Pollaczek exponent in ADR-0006, `offset_xy_m` in
  ADR-0011, accuracy claims, the reactance trend of `earth_return` (F55), the
  meaning of k₂ and k₃ (F56); backends that raise for three or more layers are
  no longer advertised (F17);[^c-f17] “`check_segment_resolution` skipped
  polyline and star electrodes (F53, F54)”;[^c-f53] “`ManhattanGridPlacement.jitter_m`
  carried a `ge=0.0` constraint on a `float | Distribution` union” (F47).[^c-f47]

[^c-cim]: Correction (2026.11.0): not bit-identical for electrodes with unequal segment lengths (+4.1 % for a 3 m rod split unevenly at h₁ = 2 m); fixed in 2026.11.0 (10.30).
[^c-f48]: Correction (2026.11.0): not fixed in 0.15.0, `generators/measurement.py` was unchanged since 0.14.1; fixed in 2026.11.0 (10.55).
[^c-f16]: Correction (2026.11.0): the symptom quoted for F16 (a factor-33 change of `concrete_rho_ohm_m` gave bit-identical impedances) is that of F08, the layout builder, which the 0.15.0 fix did not cover; fixed in 2026.11.0 (10.14).
[^c-f17]: Correction (2026.11.0): fixed in part only; the engine-comparison example still recommended `image_nlayer` or `cim` for more than two layers.
[^c-f53]: Correction (2026.11.0): not fixed in 0.15.0, `diagnostics.py` was unchanged since 0.13.0; fixed in 2026.11.0.
[^c-f47]: Correction (2026.11.0): not fixed in 0.15.0, `generators/placement.py` was unchanged since 0.14.1; fixed in 2026.11.0 (10.55).

---

## [0.14.1] - 2026-07-30

The three findings of review pass 9 that corrupt data or crash
([audit history, pass 9][pass-09]).

### Fixed

- ⚠ `fit_to_sympy` lost the sign of the residue's imaginary part, so the
  exported groundinsight formula was a different function from the fit (28.1 %
  at 16.7 Hz in the test case). Re-export affected BusTypes; fits and solver
  results were correct (F03).
- `plot_potential_contour` without `extent` raised `AttributeError` on NumPy 2
  (F12).
- `Engine.with_frequencies()` validates the frequencies (F19).
- Documentation: the cross-section snippet of the plot gallery.

---

## [0.14.0] - 2026-07-20

### Added

- `groundfield.references.pollaczek`: `pollaczek_self_impedance`,
  `pollaczek_mutual_impedance`, `loop_impedance_conductor_earth` and
  `layered_effective_wavenumber`, the two-layer earth-return impedance of a
  buried conductor per unit length (Pollaczek/Sunde), an independent reference
  for the assembled Sommerfeld stack. Its exponent was corrected in 0.15.0
  (F09).

---

## [0.13.0] - 2026-07-20

### Added

- `groundfield.references.earth_return` (Carson's closed form) with a
  benchmark of the assembled Sommerfeld stack (reactance within about 1 % of
  Carson).
- `groundfield.references.ieee80` and the IEEE Std 80-2013 Annex H benchmark:
  Grids 1–3 within about 1 % of the published grid resistance, GPR and touch
  voltage.
- `StarElectrode` (`kind="star"`), within about 1 % of Dwight's formula.

### Changed

- ⚠ The default `earth_inductive_model` is `sommerfeld` (was
  `perfect_mirror`); this affects solves with Neumann conductors at f > 0.

### Fixed

- ⚠ The two-layer inductive earth-return kernel for conductors in the upper
  layer mis-weighted the layers (attribution 0.51 instead of 0.08 at 10 kHz);
  `reflection_coefficient_layered` uses the bottom-up cascade.
- ⚠ Grids whose segment length gave a half-integer number of segments per
  span returned 1.97 to 2.89 times the correct resistance.
- The cross-layer two-layer solve of IEEE Std 80 Grid 3 needed about 263 GiB;
  it now runs in well under 1 GB.
- `PolylineElectrode` rejects zero-length edges; a closed ring of two vertices
  warns.
- Documentation: the `grid_mesh` example; the governing equations on the
  boundary page.

---

## [0.12.2] - 2026-07-18

### Added

- A changelog and roadmap page on the documentation site.

### Changed

- `mkdocs.yml` sets `strict: true`.

### Fixed

- `mkdocs build --strict` is clean (51 docstring warnings fixed); Shapely's
  benign OMBR `RuntimeWarning` is silenced.

---

## [0.12.1] - 2026-07-17

### Fixed

- The geometry plots and bounds handle the `polyline` electrode kind.

---

## [0.12.0] - 2026-07-14

### Added

- `PolylineElectrode` (`kind="polyline"`) and
  `FoundationElectrodeSpec.polygon_xy_m`: a foundation ring along the real
  building outline. It avoids coincident conductors of neighbouring OMBR
  foundations (184 of 196 AP1 sites had such pairs) and the perimeter bias of
  the OMBR (−6 % on an L-shaped building). A rectangular polygon reproduces
  the `GridMeshElectrode` ring bit-identically.

---

## [0.11.0] - 2026-07-10

The physics and numerics audit of 2026-07-08
([audit history][audit-2026-07-08]). ⚠ Numbers change for multiport matrices
with unequal segment lengths, field points in the second layer, `sommerfeld`
and `carson_series` inductive results, the series truncation, and vector fits
with complex residues.

### Added

- Guard rails (WP-A): surface-laid inductive conductors raise, a thin-wire
  hard guard, pre-flight diagnostics on every solve, `SeriesTruncationWarning`,
  non-image backends reject concrete shells, warnings for dropped non-current
  sources and for the singularity clamp.
- `vector_fit`: `VectorFitConvergenceWarning`, `n_iter_used`, `pole_shift`,
  `weighting="inverse_magnitude"`.

### Changed

- ⚠ Vector fitting on the conjugate-pair real basis: complex residues (WP-B4).
- ⚠ The series stops on the geometric tail bound; `image_max_terms` default
  100 → 300; the series controls reach every layered backend.
- One-shot multiport assembly (WP-F): a 50-building world 22.0 s → 0.18 s.

### Removed

- `_layered.image_series_offsets` (dead code).

### Fixed

- ⚠ The multiport matrix is exactly reciprocal (WP-B3).
- `cim` and `bem` raise for three or more layers (incomplete kernel, WP-E).
- ⚠ `potential` and `solve_mutual_*` use the spectral kernel for points in the
  second layer (+7 % at 7 m and +25 % at 12 m before); rods are split at the
  interface; `mom` accepts cross-layer worlds (WP-D).
- ⚠ `sommerfeld` is a genuine Pollaczek kernel (the earth-return resistance had
  the wrong sign) and the `carson_series` matrix converges under refinement
  (WP-C).
- ⚠ Quadratures (WP-B): the `layered_green` Hankel grid (46 % at 200 m), Carson
  P and Q near θ = 90° (−36 % and +67 %), the large-a asymptote.
- Single-segment distributed conductors no longer vanish; `cim`, `bem` and the
  `mom_sommerfeld` diagonal use the cross-layer kernel;
  `rho_f_from_field_result` masks dead frequencies.

---

## [0.10.0] - 2026-06-28

### Added

- `solve_mutual_field(world, engine, anchors, field_points)`: the potential at
  arbitrary points under unit excitation of each cluster.

---

## [0.9.0] - 2026-06-19

### Added

- `solve_mutual_matrix`: the mutual grounding-impedance matrix from one
  assembly, bit-identical to separate solves and 20–40× faster.

---

## [0.8.1] - 2026-06-19

### Fixed

- The `image_2layer` cross-layer path is about 20× faster (cached
  Gauss–Legendre nodes, bit-identical).

---

## [0.8.0] - 2026-06-18

### Added

- `create_engine(image_max_terms=..., image_series_tol=...)` (defaults
  unchanged).
- `OrtsnetzLayout.connect_all_buildings()`: every house is connected to the
  LV network.

---

## [0.7.0] - 2026-05-26

Includes the fixes of review pass 7 and the findings of passes 7 and 8
([pass 7][pass-07], [pass 8][pass-08]).

### Added

- `OrtsnetzLayout`: an imperative builder of a TN-Ortsnetz (OSM footprints,
  substation, cable cabinets, Manhattan-routed PEN cables, house connections,
  simulated fall-of-potential measurement, foundation masks).
- `generators.manhattan_routing` and `RadialTrunkTopology`
  (`TnNetworkConfig.pen_topology`).
- `EvaluateSpecError`, `World.reset_concrete_corrections()`,
  `OsmBuildingPlacement.footprint_at(i, strict=...)`,
  `query_buildings(max_retries=...)` (pass 7).

### Fixed

- The measurement loop of `OrtsnetzLayout.to_world` is closed by a second
  source of 180° at the auxiliary electrode, because the solver ignored
  `return_to` (replaced in 2026.11.0, 10.16, 10.17).

---

## [0.6.0] - 2026-05-15

### Added

- Concrete encasement of foundation electrodes (ADR-0012).
- `groundfield.geo`: OSM building footprints and footprint-driven foundations
  (ADR-0011), `OsmBuildingPlacement`, `FoundationElectrodeSpec.orientation_deg`,
  the `geo` extra.

### Changed

- The under-determined check of `vector_fit` respects conjugate pole pairs
  ([pass 6][pass-06]).

---

## [0.5.0] - 2026-05-14

The fixes of review passes 4 and 5 ([pass 4][pass-04], [pass 5][pass-05]).

### Added

- Top-level `evaluate_spec`, `fit_quality_summary`, `LayeredEarth` and
  `SourceAdapter`; frozen CSV column names; public `MIN_THINWIRE_RATIO`,
  `SOFT_LIMIT` and `HARD_LIMIT`.

### Changed

- `Source` is a discriminated union.
- ⚠ `Engine.frequencies` keeps the given order (it was sorted), with
  `EngineFrequencyOrderWarning` and `Engine.with_frequencies(...,
  preserve_order=True)`.
- `TnNetworkConfig.source_return_to` overrides the return electrode.

### Fixed

- `World.set_boundary_conditions` also warns on a revert; `vector_fit` rejects
  `n_poles=0` and warns on under-determined input; `evaluate_spec` raises
  `ValueError` on malformed specs; `TnNetworkConfig.source_kind` is validated;
  `World.solve` restores the sources; the documentation no longer loads
  `polyfill.io`.

---

## [0.4.0] - 2026-05-12

Housekeeping release without user-visible changes; the first version published
on PyPI (0.1.0 and 0.2.0 exist in the repository history only).

---

## [0.2.0] - 2026-05-12

The first release with a solver.

### Added

- Data model (pydantic v2): `World`, soils, electrodes, conductors (ideal,
  finite and distributed, ADR-0003), sources, `Engine`; the factories
  `create_world`, `create_electrode`, `create_conductor`, `create_source` and
  `create_engine`.
- The engine family (ADR-0001, ADR-0002): `image`, `image_2layer` (Tagg/Sunde
  series), `image_nlayer`, `cim`, `mom`, `mom_sommerfeld`, `bem`, `fem`, with
  dispatch by soil model and `compare_engines`.
- Inductive coupling (ADR-0004), Carson earth return (ADR-0005), Sommerfeld
  earth return (ADR-0006), cross-layer electrodes (ADR-0007).
- The rho-f model: vector fitting, the standard form, SymPy export and the
  groundinsight bridge (ADR-0008).
- World generators and the AP1 TN-Ortsnetz generator (ADR-0009) with a
  measurement set-up.
- Post-processing: potentials, plots, convergence studies, sweeps,
  diagnostics, current balance, touch and step voltages, CSV and VTK export.
- References: `dwight1936`, `oeding`, `carson`.

### Changed

- The inductance matrix is vectorised (ADR-0010, bit-exact).
- The project language is English.

### Deprecated

- The module `groundfield.generators.tn_ortsnetz`; use
  `groundfield.generators.tn_network`.

### Removed

- The flat v1 `TnNetworkConfig` schema.

### Fixed

- `FieldResult.potential` raises for three or more layers instead of using the
  homogeneous kernel; the rho-f functions are exported at the top level;
  `BoundaryConditions` warns about settings no backend reads.

---

## [0.1.0] - 2026-04-24

### Added

- Project skeleton: package layout, Poetry configuration, MkDocs scaffold,
  release script, GitHub workflows, citation file and licence (MIT).

[Unreleased]: https://github.com/Ce1ectric/groundfield/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.15.0
[0.14.1]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.14.1
[0.14.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.14.0
[0.13.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.13.0
[0.12.2]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.12.2
[0.12.1]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.12.1
[0.12.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.12.0
[0.11.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.11.0
[0.10.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.10.0
[0.9.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.9.0
[0.8.1]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.8.1
[0.8.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.8.0
[0.7.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.7.0
[0.6.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.6.0
[0.5.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.5.0
[0.4.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.4.0

<!-- --8<-- [end:changelog] -->

<!-- Link targets for reading this file in the repository. The documentation
     page docs/changelog.md includes the section above and defines the same
     names with paths relative to docs/. -->
[audit-history]: docs/audit/index.md
[roadmap]: docs/roadmap.md
[pass-01]: docs/audit/pass-01.md
[pass-02]: docs/audit/pass-02.md
[pass-03]: docs/audit/pass-03.md
[pass-04]: docs/audit/pass-04.md
[pass-05]: docs/audit/pass-05.md
[pass-06]: docs/audit/pass-06.md
[pass-07]: docs/audit/pass-07.md
[pass-08]: docs/audit/pass-08.md
[audit-2026-07-08]: docs/audit/audit-2026-07-08.md
[pass-09]: docs/audit/pass-09.md
[pass-10]: docs/audit/pass-10.md
