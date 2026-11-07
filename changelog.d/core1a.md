### Added

- **Exact line-source kernel (10.28):** `Engine(kernel="line")` treats every
  segment as a uniform line current (Galerkin form of the exact line potential
  `ln((R1+R2+L)/(R1+R2-L))/L`, self term `(2/L)[asinh(L/a) - sqrt(1+(a/L)^2) + a/L]`,
  images with mirrored end points) for the reaction matrix, `FieldResult.potential`
  and `solve_mutual_matrix` / `solve_mutual_field`. Against the exact
  uniform-current resistance (100 Ω·m) a 3 m rod, a 10 m strip, a 10 × 10 m
  foundation and a 20 × 20 m grid agree within 0.0004 % at every segment length
  from 2 m to 0.125 m (point kernel: −0.4 … −1.7 %); a ring of 5 m radius
  converges monotonically as O(ds²) (−0.60 / −0.17 / −0.044 / −0.011 % at
  ds = 2 / 1 / 0.5 / 0.25 m; point kernel −0.95 % at every ds). The default stays
  `kernel="point"`, which reproduces earlier numbers; backends outside the image
  family reject `kernel="line"`. Results record `metadata["kernel"]`;
  `PointSource` gains an optional `direction` (results written before this
  release load unchanged and are evaluated with the point kernel). For 16 house
  grids (1 920 segments) a solve with the line kernel takes about 2.5× the time
  of the point kernel with `numba` installed and about 5× without.
  Test: `tests/test_line_kernel.py`.
- **Geometry queries on the electrode models (10.79):** every electrode kind
  provides `segments(ds, layer_interfaces=None)`, `segment_count(ds)`,
  `wire_polylines()`, `bounds()`, `total_length()` and `min_feature_length()`;
  `groundfield.geometry.WireSegments` holds midpoints, lengths and tangents.
  The solver discretises through them (midpoints, lengths and tangents
  bit-identical to the previous discretiser), so diagnostics, plots and exports
  can use the same conventions instead of their own `isinstance` ladders.
  Test: `tests/test_electrode_geometry.py`.
- **Optional compiled kernels (10.58, 10.28):** when `numba` is installed the
  Tagg/Sunde series matrix and the far-field term of the line kernel are
  evaluated by compiled kernels that give the same matrices bit for bit: the
  series about 3× faster than the numpy path on one core (1 200 segments,
  148 images), the far-field term about 5× (1 920 segments). Tests:
  `tests/test_kernel_performance.py::test_numba_series_kernel_is_bit_identical`,
  `tests/test_line_kernel.py::test_numba_far_field_block_is_bit_identical`.

### Changed

- ⚠ **Two-layer image series summed before the product (10.58):** the series is
  accumulated image by image into one matrix and then multiplied with the
  currents, instead of one matrix product per image. Results of `image_2layer`
  (electrodes inside the upper layer) and `cim` move by floating-point round-off
  only: at most 7.3e-16 relative on the golden cases `cim_rod_two_layer`,
  `cond_isolated_pen_sommerfeld_two_layer`, `defaults_two_layer_ring_auto`,
  `img2_grid_upper_resistive` and `img2_ring_rods_within_layer` (all
  `within_tol`). The reaction matrices of `solve_mutual_*` and of the `mom`
  family are unchanged bit for bit. A 30-house two-layer network
  (3 669 segments, 11 frequencies) solved 10–15× faster in two back-to-back
  comparisons on the same machine (memory 1 236 → 219 MiB).
- **Solver structure (10.75, 10.79):** `solver/image.py` (2 526 lines) is split
  into private modules (discretisation, guards, kernels, inductive assembly,
  topology, pipeline); `solve_image` and `solve_image_2layer` run through one
  pipeline and differ only in the Green's-function object; the Tagg/Sunde
  series is one `ImageSeries` object used by the solver, `FieldResult.potential`
  and `solve_mutual_*`. Every previously importable name of
  `groundfield.solver.image` remains importable. Numbers are unchanged
  (golden values bit-identical). Tests: `tests/test_solver_structure.py`,
  `tests/test_image_pipeline.py`.
- The warning for ignored non-current sources of `image` and `image_2layer`
  uses the common text of the other backends; `FieldResult.clusters` lists the
  electrodes in world order.

### Fixed

- **Memory of the image-family kernels (10.56):** the kernels no longer build
  `(N, N, 3)` difference tensors (96 bytes per segment pair). A 30-house network
  (3 687 segments) peaks at 115 MiB instead of 1 248 MiB and solved about 10×
  faster in back-to-back runs on the same machine; an 80-house network
  (9 712 segments) solves in about 2 s with 773 MiB (before: about 8.4 GiB
  needed). With rods reaching the lower layer of a two-layer soil (ADR-0007
  path) the 30-house network peaks at 510 MiB instead of 1 250 MiB. Results are
  bit-identical. Above 1 GiB per reaction matrix the kernels switch to row
  blocks (results then differ by round-off only). Tests:
  `tests/test_kernel_performance.py::test_solve_memory_peak_is_bounded`,
  `tests/test_kernel_performance.py::test_cross_layer_solve_memory_peak_is_bounded`.
- **Reaction matrix evaluated twice per solve (10.57):** the segment potentials
  after the solve reuse the assembled matrix. Bit-identical. Test:
  `tests/test_kernel_performance.py::test_one_reaction_matrix_assembly_per_solve`.
- **`solve_mutual_matrix` built the reaction matrix as `kernel(eye(N))`
  (10.59):** the matrix is assembled directly; Z_G is bit-identical and a
  20-house two-layer catalogue matrix (2 466 segments) was 23–31× faster in two
  back-to-back comparisons (memory 604 → 146 MiB). Test:
  `tests/test_kernel_performance.py::test_mutual_matrix_without_identity_product`.
- **`FieldResult.potential` unchunked (10.62):** large evaluations run in row
  blocks; a 200 × 200 grid over 3 687 sources peaks at 17 MiB (v0.15.0 needs about
  96 bytes per point·source pair, 13 GiB here). Two-layer evaluations share every
  `1/r` block between real and imaginary part and skip an all-zero imaginary
  part. Evaluations up to 16 MiB of kernel are bit-identical. Test:
  `tests/test_kernel_performance.py::test_potential_on_a_large_grid_is_memory_bounded`.
- **Point-source arrays rebuilt on every `potential()` call (10.65):** cached
  privately on the result (not serialised). Test:
  `tests/test_kernel_performance.py::test_point_source_arrays_are_cached_and_invalidated`.

### Removed

- **Dead code (10.81):** `solver.image._potential_kernel`,
  `solver.image._segment_z_extent`, `solver.image_2layer._two_layer_potential_kernel`
  and the empty module `solver.field_study`. Test:
  `tests/test_solver_structure.py::test_dead_helpers_are_removed`.
