### Added

- **Concrete shell on every electrode kind (10.12–10.14):** rods, rings,
  meshes and grid meshes carry `concrete_shell_coefficient_ohm_m` like strips,
  polylines and stars. It is the one mechanism through which the solver sees
  a concrete encasement (ADR-0012): every segment of length Δs adds C/Δs to its
  reaction-matrix diagonal, in the image family and in `solve_mutual_matrix` /
  `solve_mutual_field`; with the uniform current per electrode this is exactly
  the lumped shell resistance R = C/L in series with the electrode. Conversion
  helpers in `groundfield.geometry.electrodes`:
  `concrete_shell_coefficient(rho_c, wire_radius, thickness)` (C =
  ρ_c/(2π)·ln((r_a + t)/r_a), 38.16 Ω·m for 100 Ω·m and 5 cm around a 5 mm
  wire), `shell_coefficient_from_resistance(electrode, R)` (C = R·L) and
  `shell_resistance_from_coefficient(electrode, C=None)` (R = C/L), which the
  generators use for `concrete_model="lumped"`. Rods, rings and meshes without a
  shell serialise exactly as before. Tests: `tests/test_concrete_shell_coefficient.py`.
- **`solve_mutual_matrix(..., offdiag="galerkin")` (10.34):** the off-diagonal
  entry Z_ij is the reaction T_iᵀ(K + S)T_j of the two leakage distributions —
  the potential of cluster i under excitation j, reciprocal to round-off
  without symmetrisation, and the matrix with which (Y_L + Z_G⁻¹)u = i
  reproduces the solve of the clusters joined by branches. `probe_points` is
  optional; `offdiag=None` (default) selects `"galerkin"` without probe points
  and `"probe"` with them, so existing calls keep their numbers apart from the
  diagonal (see *Changed*). On an AP1-like site (station ring + three rods,
  12 polygon foundations, ds = 2 m, five soils) the galerkin matrix reproduces
  a network solve with PEN-like branches to ≤ 2e-5, while the symmetrised probe
  matrix misses the input impedance by −0.15 … −0.88 % and single entries by up
  to 46 %; for Z_G as a network element `offdiag="galerkin"` is recommended.
  Tests: `tests/test_mutual_matrix.py::test_galerkin_offdiagonal_is_reciprocal`,
  `::test_galerkin_matrix_reproduces_the_joined_network`,
  `::test_offdiag_selection_and_validation`.

### Changed

- ⚠ **Exact line-source kernel is the default of the image family (10.28):**
  `Engine.kernel` defaults to `None`, the backend's preferred kernel: `"line"`
  for `image`, `image_2layer` and `image_nlayer` (also used by
  `FieldResult.potential` and `solve_mutual_*`), the backend's own kernel
  elsewhere; `metadata["kernel"]` records the kernel that ran and
  `kernel="point"` reproduces releases up to 0.15.0. The request is carried in
  the value, so engines rebuilt from JSON, `with_frequencies` or a provenance
  block run like the original. Numbers move for every default solve of the
  image family: rod 3 m +0.60 % (surface potential up to 1.9 %), two-layer ring
  +1.23 %, two rods joined by a galvanic conductor of 5 m segments: currents
  10.9 %, cluster impedance 8.4 %. Cost: a 16-house solve (N = 1920) takes 2.7
  times as long as with the point kernel with numba installed and 5.7 times
  without. In two-layer soil a straight 10 m strip stays within 0.05 % of its
  exact uniform-current resistance for ds = 0.125–2 m (point kernel: up to
  22 %). Golden cases: `defaults_rod_hom`, `defaults_two_layer_ring_auto`,
  `defaults_mixed_cluster_inductive`.
- ⚠ **Electrodes through a soil-layer interface (10.15, F02):** in
  `image_2layer` and `solve_mutual_*` an electrode that crosses h₁ (where the
  resistivity changes) has one uniform-current owner per layer, all in its
  cluster, so the current density may differ above and below the interface.
  A 12 m rod through h₁ = 5 m now equals the manual split into two bonded rods
  and lies +0.11 … +0.31 % from `mom` with the point kernel (+0.7 … +1.3 % with
  the line kernel, its offset from point-matching methods; 0.15.0: +566 % in
  1000/30 Ω·m, +148 % in 1000/100, +123 % in 100/1000). `TwoLayerSoil(ρ, ρ, h)` stays bit-identical to
  the homogeneous backend. Who is affected: rods through h₁ — not the AP1
  station rods (4 m from 0.6 m, h₁ ≥ 5 m). Golden cases: `img2_rod_crosslayer`
  (−46.5 %), `line_img2_rod_crosslayer` (−48.2 %), `mutual_three_rods_crosslayer`.
- ⚠ **Interface slivers and segment counts (10.32):** an interface cut that
  would leave a wire part shorter than the wire diameter
  (`geometry.electrodes.SLIVER_RADII = 2` radii) is dropped; the point kernel
  takes the Galerkin self term for segments shorter than 2.99 wire radii
  (identical above); segment counts ignore floating-point noise
  (`ceil(L/ds − 1e-9)`): a star arm of 3.000000000000001 m is cut into 6, not 7
  segments at ds = 0.5 m (≤ 6.9e-5 point kernel, ≤ 8.3e-8 line kernel).
  Golden cases: `img_mixed_cluster_hom`, `line_img_mixed_cluster_hom`.
- ⚠ **Post-solve series controls follow the engine (10.25):**
  `FieldResult.potential`, `solve_mutual_matrix` and `solve_mutual_field` sum
  the Tagg/Sunde series with the controls recorded with the solve
  (`metadata["series"]`, the engine defaults for results without the record)
  instead of a hard-coded 100 terms / 1e-6, and warn with
  `SeriesTruncationWarning` when the series is truncated. For K = −0.94 and
  `image_max_terms=2000` the potential 100 m from a ring moved by 0.8 %. The backend
  entry points `solve_image_2layer` / `solve_image_nlayer` take the engine's
  controls instead of private defaults 100 / 200 (10.74). Golden cases: none
  (the cases use the default 300 terms).
- ⚠ **`solve_mutual_matrix` diagonal (10.26):** `Z[j, j]` is the cluster
  impedance reduced like `FieldResult.cluster_impedance` — length-weighted
  over every member and every layer owner — instead of the unweighted mean over
  the alphabetically first member; equal to the per-excitation solve to
  ≤ 2.2e-16. 0.15.0 deviated by +0.150 % for an L-shaped footprint at
  ds = 2 m and by −3.8 / −9.1 % for a rod split at h₁. AP1-like site
  (ds = 2 m): house foundations up to 0.10 %, station 0.000 %. Golden cases:
  `mutual_*` (round-off; `mutual_three_rods_crosslayer` see 10.15).
- **`solve_mutual_matrix` / `solve_mutual_field` check the backend (10.34):**
  the engine's backend is resolved and checked like in `Engine.solve`
  (aliases, capability check, kernel); a backend outside the image family
  raises `BackendCapabilityError` (a `ValueError`) instead of returning image
  numbers (an 8-arm star: 7.854 Ω returned for a `mom` engine whose own solve
  gives 7.128 Ω). Test:
  `tests/test_mutual_matrix.py::test_non_image_backend_is_rejected`.
- ⚠ **`solve_mutual_matrix` / `solve_mutual_field`: the other clusters float
  (10.29):** each excitation is solved with the source-free clusters floating
  (zero net current, equalising currents in bonded groups and between the
  layer owners of a rod through h₁), as `Engine.solve` does since 10.29; the
  galerkin off-diagonal is then the exact potential of the passive cluster.
  AP1-like site: diagonal 3.3e-5, galerkin off-diagonal up to 0.09 %. Golden
  case: `mutual_foundations_distributed_concrete` (Z_G 5.2e-4, R 1.6 %).
- **`Engine.solve` validates the world (10.66):** it runs
  `World.check(require_source=False)` for every backend and entry point
  (dangling references, short-circuited sources, zero or non-finite
  magnitudes); a world without a source stays solvable through
  `Engine.solve` (all results zero), as before.
- **Cross-layer solves report no image series (10.50, F43):** a solve on the
  ADR-0007 path (an electrode reaches h₁) no longer reports the term count or
  the truncation of a series it never evaluates; `metadata` gains
  `series_used` and `series`. Test:
  `tests/test_series_controls.py::test_cross_layer_solve_reports_no_series`.

### Fixed

- **Degenerate electrode geometry (10.67):** zero-length strips, non-positive
  mesh or grid extents, non-finite coordinates or lengths and dimensions not
  longer than the wire radius (typically a radius typed in mm) are rejected
  when the electrode is built; up to 0.15.0 they failed in the solver as a
  thin-wire violation, an `OverflowError`, or solved to Z = nan. Test:
  `tests/test_electrode_validation.py`.
- **Reciprocity of the line kernel across the layer interface (10.28):** the
  cross-layer reaction matrix scaled the line corrections of the direct and
  surface-image terms with the source-layer resistivity while the layered
  correction kept the compensating terms in point form: 3.8 % asymmetric, and
  a two-port of a rod through h₁ and a ring non-reciprocal by 1.1–4.7 % (point
  kernel: 1e-8). Pairs in different layers now carry the transmitted
  resistivity 2ρ₁ρ₂/(ρ₁ + ρ₂), and the near-singular interface image of the
  layered correction is a Galerkin integral: symmetric to ~1e-8, and a rod
  through h₁ changes by less than 0.01 % per halving of ds (was up to 0.22 %).
  The interface image matters most for horizontal wires just below h₁: a ring
  (r = 3 m) 5 cm below the interface of 100 over 1000 Ω·m now gives
  55.6–55.7 Ω at every ds from 0.5 to 0.025 m; the point kernel (and `mom`,
  and the line kernel before) gave 29.7 Ω at ds = 0.5 m (−46 %), converging to
  54.7 Ω only at ds = 0.025 m. Keep ds well below 2|z − h₁| with
  `kernel="point"`. Tests: `tests/test_reciprocity.py`,
  `tests/test_cross_layer_line_kernel.py`.
- **`image_2layer` with a layer stack (10.48):** a `MultiLayerSoil` with two
  distinct layers raised a bare `TypeError` after passing the capability check;
  it now solves its canonical two-layer form, three distinct layers raise
  `BackendCapabilityError`. Test: `tests/test_layer_stack_dispatch.py`.
- **Interface slivers (10.32):** a rod whose tip lay 1 mm below h₁ raised a
  `ValueError`, one 9 mm below acted as a short in `mom` (196.6 → 144.2 →
  166.8 Ω for tips at 4.99 / 5.009 / 5.02 m). Test:
  `tests/test_interface_slivers.py`.
- **Junction terminals in `FieldResult.input_impedance` (10.38):** a source
  attached to or returning at a junction reads the junction potential. Test:
  `tests/test_junction_nodes.py::test_input_impedance_of_a_source_on_a_junction`.
- **Line-kernel cost with one long segment (10.28):** the near-pair search used
  the longest segment of the whole set as radius for every observer; one
  conductor discretised at 50 m turned a 16-house line solve from 0.8 s into
  29 s and 1.7 GiB. Each segment now searches with its own radius (identical
  results); the reaction corrections of 3000 short plus three 35 m segments
  peak below 40 MiB (940.9 MiB before). Test:
  `tests/test_line_kernel_near_pairs.py`.
- **Two-layer `mom` assembled its series matrix twice per solve:** the matrix
  is kept from the reaction-matrix assembly (bit-identical; 20 × 20 m grid,
  N = 400: 0.294 s → 0.154 s).
