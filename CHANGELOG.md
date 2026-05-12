# Changelog

All notable changes to `groundfield` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Change categories follow the Keep-a-Changelog vocabulary:

- **Added** — new features and public API.
- **Changed** — behaviour changes to existing public API.
- **Deprecated** — features that still work but will be removed.
- **Removed** — features taken out of the public API.
- **Fixed** — bug fixes.
- **Security** — vulnerability fixes.
- **Docs** — documentation-only changes.
- **Internal** — refactors, tests, packaging, CI; no observable behaviour change.

The backlog of ideas that are not yet scheduled is kept at the end of this
file under **Roadmap**. During regular work, add your entry under the
matching category in `[Unreleased]`; the release script
(`scripts/release.py`) moves the whole `[Unreleased]` block into a new
version section when a release is cut.

---

## [Unreleased]

### Fixed

- **`FieldResult.potential` no longer silently falls back to the
  homogeneous kernel for a `MultiLayerSoil` with three or more
  layers** (`solver/result.py`). Prior to this fix, calling
  `result.potential(points)` on a world with `n ≥ 3` soil layers
  returned potentials computed against a single image charge at
  $z \to -z$ — wrong by the same margin as the layered Green's
  function deviates from the homogeneous half-space. The error was
  silent because the dispatcher only branched on
  `isinstance(self.soil, TwoLayerSoil)` and otherwise fell through
  to `_potential_homogeneous`, affecting profiles, surface plots,
  touch- and step-voltage post-processing and VTK exports
  alike. The new dispatcher casts a degenerate 1-layer
  `MultiLayerSoil` to the homogeneous kernel, casts a 2-layer
  `MultiLayerSoil` to `TwoLayerSoil` and re-uses the Tagg/Sunde
  series, and raises `NotImplementedError` for `n ≥ 3` with a
  clear pointer that the *solve* itself remains correct via the
  `cim` / `mom_sommerfeld` / `bem` backends and that
  `result.electrode_potentials` / `result.cluster_impedance` stay
  accessible. A closed-form n-layer Green's-function kernel on
  the post-solve potential path is queued for a later release.
- **Top-level `groundfield` package re-exports the `rho-f` fit
  API** (`__init__.py`). `vector_fit`, `VectorFitResult`,
  `fit_to_sympy`, `rho_f_from_field_result`, `RhoFStandardFit`,
  `fit_rho_f_standard`, `fit_to_sympy_standard` and
  `rho_f_standard_from_results` are now reachable via
  `groundfield.<name>` (the README and ADR-0008 advertise this
  short import path; prior to this fix the user had to spell out
  `from groundfield.postprocess.vector_fitting import …` because
  the symbols were missing from both the top-level imports and
  `__all__`).
- **`BoundaryConditions` no longer pretends to honour non-default
  settings** (`boundary.py`, `world.py`). The integral /
  image-charge backends in `groundfield.solver` enforce
  `far_field="dirichlet"`, `surface="neumann"` and
  `reference_node=None` implicitly through the choice of
  Green's function; any other value set via
  `World.set_boundary_conditions(...)` was silently dropped at
  solve time. The fields are now documented as accepted but not
  consumed in v0.2.0, and `set_boundary_conditions` emits a
  `UserWarning` whenever a non-default value is supplied. The
  fields remain reserved for the upcoming FEM backend, which will
  resolve them explicitly.

### Added

- **Disk I/O — CSV and legacy ASCII VTK
  (`groundfield.io.csv`, `groundfield.io.vtk`).** Closes the
  long-standing ``io.csv`` / ``io.vtk`` *Reserved* slots in
  the ``io`` package docstring. Six writers, no new
  dependencies (pandas is already a runtime dep; the VTK
  writer is ~30 lines of pure-Python in legacy ASCII format):
  * ``save_potential_path_csv(result, path, *, start, direction,
    distance, n, frequency_indices=None)`` — sample
    :meth:`FieldResult.potential` along a line and write
    ``(s, x, y, z, frequency_Hz, phi_re, phi_im, abs_phi)``.
  * ``save_electrode_table_csv(result, path, *, world=None,
    frequency_index=0)`` — wrap
    :func:`electrode_current_table` and dump to CSV.
  * ``save_cluster_impedances_csv(result, path, *,
    frequency_index=0)`` — wrap :func:`cluster_current_balance`
    and dump to CSV; the per-cluster ``members`` list is
    flattened into a ``';'``-joined string for tabular
    compatibility.
  * ``export_geometry_vtk(world, path)`` — legacy ASCII VTK
    PolyData with electrodes (rod / ring / strip /
    grid_mesh perimeter + interior wires) and conductor line
    segments as 3-D polylines. Cell data carries an integer
    ``role`` field (0 = electrode, 1 = conductor) so colour-
    by-role works directly in ParaView.
  * ``export_field_vtk(result, path, *, extent, z=0.0,
    n=(120, 120), frequency_index=0)`` — sample the potential
    on a regular :math:`N_x \times N_y` grid in the slice
    plane :math:`z = z_0` and write a STRUCTURED_POINTS file
    with ``potential_re`` and ``potential_im`` scalars.
- ``tests/test_io_csv.py`` (8 tests) — round-trip of the
  potential-path writer (re-evaluating
  :meth:`FieldResult.potential` at the saved coordinates
  matches the saved values to 12+ significant figures), parent-
  directory creation, multi-frequency dump, electrode-table
  CSV ↔ in-memory DataFrame equivalence (with / without
  ``world``), cluster-impedance CSV with the flattened
  ``members`` column, and bad-argument error paths.
- ``tests/test_io_vtk.py`` (9 tests) — POLYDATA header
  integrity, exact polyline count for a mixed world (1 rod
  + 1 ring + 1 strip + 1 ``GridMeshElectrode(n_x=3, n_y=2)``
  + 1 conductor = 11 polylines), empty-world header-only
  output, ``role`` cell-data scalar covers both 0 and 1,
  STRUCTURED_POINTS ``DIMENSIONS`` / ``ORIGIN`` /
  ``SPACING`` correctness, scalar payload size matches
  ``n_x * n_y``, and bad-extent / too-few-grid-points error
  paths.

- **Convergence study over `engine.segment_length`
  (`groundfield.postprocess.convergence`).** Single-axis
  refinement helper — the canonical "halve the segment length,
  watch what happens" experiment as one function call:
  * ``convergence_study(world, engine, *, segment_lengths,
    response=None) -> pd.DataFrame`` — clones the engine via
    :meth:`Engine.model_copy` so the original is **not**
    mutated, solves at every refinement step, and returns a
    long-format DataFrame with ``segment_length_m``,
    ``frequency_Hz``, ``n_segments`` and the response columns.
    Default response: cluster impedance + EPR at the source's
    cluster (same extractor as :func:`sweep`).
  * ``plot_convergence(df, *, response="abs_Z",
    reference=None, ...)`` — log-x plot with the x-axis
    **inverted** so finer ``segment_length`` lands on the right
    (the asymptote direction). Optional reference line for the
    analytical asymptote (Sunde, Dwight, IEEE 80).
- ``tests/test_convergence.py`` (12 tests) — monotone growth
  of ``n_segments`` with refinement, monotone convergence of
  ``abs_Z`` to within 5 % of the Sunde reference for a 1.5 m
  rod, **engine-mutation guard** (the original engine's
  ``segment_length`` is unchanged after the helper runs),
  validation of the ``segment_lengths`` argument (empty,
  single value, repeated values, non-positive entries),
  multi-frequency row count, plot-helper smoke (single +
  multi-frequency, with reference line), and the inverted
  x-axis check.

- **Cartesian-product parameter sweeps
  (`groundfield.postprocess.sweep`).** New module that turns
  the AP1 parameter axes (:math:`\rho_1`, :math:`\rho_2`,
  :math:`h_1`, geometry, frequency) into a single tabular
  response, ready for vector-fitting and the
  :math:`\rho`-:math:`f` regression:
  * ``sweep(world_factory, engine, *, axes, response=None) ->
    pd.DataFrame`` — Cartesian product over arbitrary named
    axes. ``engine`` is either a static :class:`Engine`
    (reused for every combination) or a callable
    ``engine(**combination) -> Engine`` (rebuilt per
    combination for axis-dependent discretisation). For each
    combination + frequency, a row is emitted with the axis
    values, ``frequency_Hz``, and the response keys.
  * Default response: cluster impedance and EPR at the
    source's cluster, per frequency (``Z_re``, ``Z_im``,
    ``abs_Z``, ``arg_Z_deg``, ``U_E_re``, ``U_E_im``,
    ``abs_U_E``, ``I_re``, ``I_im``, ``abs_I``).
  * ``plot_sweep_lines(df, *, x, y="abs_Z", color=None,
    log_x=False, log_y=False, ...)`` — line plot of the
    response, one curve per ``color`` value if set.
  * ``plot_sweep_heatmap(df, *, x, y, response="abs_Z",
    frequency_Hz=None, agg="mean", ...)`` — pivot-table
    heatmap of one response over a ``(x_axis, y_axis)``
    pair, optionally selecting a frequency slice.
- ``tests/test_sweep.py`` (14 tests) — Cartesian-product row
  count (axes × frequencies), default-response columns
  present, AP1 linearity check (:math:`Z` is proportional to
  :math:`\rho` for homogeneous soil to better than
  :math:`10^{-6}` relative), per-combination engine factory
  is invoked exactly once per Cartesian point, custom response
  extractor replaces the default, empty-axes / empty-axis-
  values error paths, plot-helper smokes (single curve, multi-
  curve with logs, heatmap with frequency slice, unknown-column
  guard, missing-frequency guard).

- **Pre-solve world diagnostics
  (`groundfield.diagnostics`).** New top-level module — the
  pre-solve counterpart to :mod:`groundfield.validation`'s
  post-solve cross-engine check. Three helpers, all of which
  work on a :class:`World` (and optionally an :class:`Engine`)
  without invoking the solver:
  * ``world_statistics(world) -> dict`` — structural snapshot:
    counts per electrode kind / conductor type / coupling mode,
    total electrode wire length, total conductor length plus
    ``min/median/max/mean`` of the conductor-length distribution,
    full :math:`(x, y, z)` bounding box and footprint area, and a
    ``has_layered_soil`` flag. Complements
    :meth:`World.summary` (one-line text) with a richer
    machine-readable dictionary that scales to AP1-grade
    networks.
  * ``expected_segments(world, engine) -> dict`` — predicts the
    number of point-source segments that the image-family
    discretiser will produce. Mirrors
    :mod:`groundfield.solver.image` exactly for every electrode
    kind (rod / ring / strip / mesh / grid_mesh) and adds the
    ``ceil(L / discretize_segment_length)`` count for
    distributed conductors (ADR-0003). The prediction is
    bit-exact for the ``image`` / ``image_2layer`` /
    ``image_nlayer`` / ``mom`` / ``mom_sommerfeld`` / ``cim``
    / ``bem`` backends; FEM uses an unrelated axisymmetric
    volume mesh.
  * ``check_segment_resolution(world, engine) -> list[str]`` —
    heuristic warnings for common AP1 modelling pitfalls:
    thin-wire ratio :math:`\Delta s / r_\text{wire} \ge 5`,
    electrode smaller than one segment (degenerate
    discretisation), distributed-conductor wire / segment
    mismatch, and total-segment budget thresholds (5 000 soft
    warning, 20 000 hard warning to flag the dense-system
    :math:`O(N^2)` memory and :math:`O(N^3)` solve scaling).
    Returns an empty list when everything looks healthy.

  Helpers re-exported at the package top level
  (``gf.world_statistics``, ``gf.expected_segments``,
  ``gf.check_segment_resolution``).
- ``tests/test_diagnostics.py`` (20 tests) — counts and lengths
  match the analytic geometry exactly, footprint and bounds
  are consistent, layered-soil flag, empty-world safety,
  ``expected_segments`` **bit-exact** against the image solver
  for every electrode kind (rod / ring / strip / mesh /
  grid_mesh) plus distributed-conductor counts and per-kind
  aggregation, ``check_segment_resolution`` clean-world empty
  return, thin-wire warning trigger, distributed-conductor
  mismatch trigger, electrode-smaller-than-segment trigger,
  budget-threshold trigger on a 100 m × 100 m grid mesh,
  invalid-segment-length error path, and a top-level export
  check.

- **World-geometry plots without solving
  (`groundfield.postprocess.geometry_plot`).** New module that
  renders the *physical* world — electrodes, conductors and
  current sources — as a sanity check **before**
  ``world.solve(...)``. Useful for AP1-grade networks where a
  typo in an electrode position or a missing conductor would
  otherwise only surface several minutes of solver time later:
  * ``world_bounds_3d(world) -> (x_min, x_max, y_min, y_max,
    z_min, z_max)`` — full :math:`(x, y, z)` bounding box of the
    electrodes plus conductor endpoints. Extension of
    :func:`world_bounds_xy` to the third axis; correctly tracks
    rod feet (``position[2] + length``) and overhead conductor
    routing (negative :math:`z`).
  * ``plot_world(world, *, plane="xy"|"xz", extent=None,
    padding_m=5.0, show_conductors=True, show_sources=True,
    annotate_electrodes=False, ax=None, ...) -> Figure`` —
    top-down or vertical 2-D plot. Electrodes drawn via the
    existing :func:`_draw_electrodes` helper (consistent style
    with the field plots); conductors as colour-coded line
    segments (PEN green, bare_copper orange, cable_shield grey,
    overhead steel blue, generic dark grey; solid for
    ``coupling_to_soil="galvanic"``, dashed for ``"isolated"``);
    sources as red star at the anchor electrode plus a thin
    arrow to ``return_to`` if set. Synthetic legend over
    conductor types and source markers; soil surface (``z=0``)
    drawn as a dotted grey line in the ``xz`` view.
  * ``plot_world_3d(world, *, show_conductors=True,
    show_sources=True, show_surface=True, elev=22.0, azim=-55.0,
    ...) -> Figure`` — 3-D wireframe via
    :mod:`mpl_toolkits.mplot3d`. The :math:`z`-axis is **inverted**
    so depth grows downwards on screen (groundfield convention);
    a translucent grey plane at :math:`z=0` marks the soil
    surface. Rods drawn as vertical line segments, rings as 64-point
    circles, strips as line segments, mesh / grid_mesh electrodes
    as outer rectangle plus inner wires.

  Helpers re-exported at the package top level
  (``gf.plot_world``, ``gf.plot_world_3d``, ``gf.world_bounds_3d``).
- ``tests/test_geometry_plot.py`` (18 tests) — bounding-box
  correctness across all electrode kinds (rod foot, ring extremes,
  strip, mesh / grid_mesh, conductor endpoints), empty-world
  zero-box, ``plot_world`` smoke on both planes (``xy`` /
  ``xz``), inverted ``y`` axis on ``xz``, conductor / source
  toggle reduces line count, padding-around-bounds default,
  explicit-extent override, unknown-plane error, optional
  axes hand-over (``ax=...``), annotation creates one
  :class:`matplotlib.text.Annotation` per electrode,
  ``plot_world_3d`` smoke + inverted z-axis check + empty-world
  + options-off variants, and a top-level export check.

- **Current-sharing post-processing
  (`groundfield.postprocess.current_balance`).** New module that
  turns the per-electrode currents in ``FieldResult`` into the
  AP1 *"where does the injected current actually return?"*
  quantities:
  * ``cluster_current_balance(result, *, frequency_index=0) ->
    pd.DataFrame`` — per-cluster soil leakage
    :math:`I_c = \sum_{e \in c} I_e`, cluster potential
    :math:`U_c`, cluster impedance :math:`Z_c = U_c / I_c`,
    sorted by descending :math:`|I_c|`.
  * ``electrode_current_table(result, world=None, *,
    frequency_index=0) -> pd.DataFrame`` — per-electrode
    potential, current, two-terminal impedance, and the
    fractional share of the cluster total :math:`s_{e \mid c} =
    I_e / I_c`. With ``world`` set, the table also reports the
    electrode kind and connection-point depth — the small AP1
    annotations that turn a 200-EFH run from "wall of numbers"
    into "what's actually loaded."
  * ``split_factor(result, world, *, source_name=None,
    frequency_index=0) -> complex`` — galvanic current-split
    factor :math:`s = I_{c_\text{src}} / I_\text{src}`. Returns
    :math:`s = 1` when the entire injected current leaves the
    source cluster through the soil; :math:`s < 1` when a
    metallic parallel path (PEN trunk, parallel measurement
    lead, cable shield) carries part of the current as a
    parallel resistive path. Raises on missing / multiple /
    zero-magnitude sources or on unknown ``source_name``.

    *Naming note.* This is **not** the *Reduktionsfaktor* of the
    German EVU / Schirmtechnik literature (Oeding & Oswald 2016)
    — that latter quantity is the additional **transformatorische
    / inductive coupling correction** between a current-carrying
    conductor and a parallel grounding / shield conductor and
    vanishes for perpendicular geometry. The split factor here is
    purely galvanic and applies whenever there are parallel
    resistive paths, regardless of conductor angle. The proper
    Reduktionsfaktor is on the roadmap; the inductance backends in
    :mod:`groundfield.coupling` are already in place.
  * ``plot_current_sharing(result, world=None, *, by="electrode",
    top_n=15, frequency_index=0)`` — quick top-N bar chart of
    :math:`|I|` (per electrode or per cluster). The default
    ``by="electrode"`` is the AP1 default for spotting which
    physical electrode actually carries the test current.

  Backed by :class:`pandas.DataFrame` (already a runtime
  dependency); helpers re-exported at the package top level
  (``gf.cluster_current_balance``, ``gf.electrode_current_table``,
  ``gf.split_factor``, ``gf.plot_current_sharing``).
- ``tests/test_current_balance.py`` (19 tests) — KCL on a single
  rod (``s = 1``), KCL on an ideally bonded multi-electrode
  cluster (``s = 1`` regardless of member count), AP1
  measurement scenario with a finite-impedance Cu feed lead in
  parallel to the soil return path (``s < 1`` plus an explicit
  branch-current consistency check
  :math:`(1 - s) I_\text{src} = (U_a - U_b) / R`), per-cluster
  KCL Σ leakages = source magnitude, sort order by descending
  :math:`|I|`, share-of-cluster sum = 1 + 0j over each cluster,
  optional ``world`` annotation columns (kind / depth_m),
  unknown / missing / multiple source error paths, plot smoke
  tests, and a top-level export check.

- **Touch- and step-voltage helpers (`groundfield.postprocess.safety`).**
  New module that closes the long-standing gap between
  ``FieldResult`` and the engineering safety quantities promised
  in the ``postprocess`` package docstring:
  * ``touch_voltage(result, world, *, electrode, distance=1.0,
    direction=(1, 0, 0), surface_z=0.0, frequency_index=0)`` —
    pointwise :math:`U_T = U_E - \varphi(\mathbf r_\text{feet})`
    on the soil surface.
  * ``touch_voltage_envelope(result, world, *, electrode,
    distance=1.0, n_angles=24, ...)`` — same evaluation along an
    equidistant horizontal circle around the touched electrode;
    the maximum of ``|U_T|`` is the conservative envelope used in
    safety verification.
  * ``step_voltage(result, *, position, direction=(1, 0, 0),
    step=1.0, surface_z=None, frequency_index=0)`` —
    :math:`U_S = \varphi(\mathbf r_1) - \varphi(\mathbf r_1 +
    d_\text{step}\,\hat{\mathbf e})` between two surface points.
  * ``permissible_touch_voltage_en50522(t_clear_s)`` — reference
    curve :math:`U_{TP}(t_F)` taken **verbatim** from EN 50522:2010
    **Table B.4** ("Berechnete Werte der zulässigen
    Berührungsspannung U_TP in Abhängigkeit von der Fehlerdauer
    t_f", values rounded to 5 V in the standard); log-log
    interpolation over the eight canonical anchors :math:`(0.05,
    725),\ (0.10, 655),\ (0.20, 525),\ (0.50, 225),\ (1.00, 115),\
    (2.00, 95),\ (5.00, 85),\ (10.00, 85)` (s, V), clamped to the
    table endpoints outside the grid (the standard's terminal 85 V
    plateau between 5 s and 10 s is reproduced exactly).

  All evaluations stay in the complex frequency-domain phasor
  convention used by the rest of ``groundfield`` so that
  inductive-coupling effects above DC remain visible. Helpers are
  re-exported at the package top level (``gf.touch_voltage``,
  ``gf.step_voltage``, …).
- ``tests/test_safety.py`` (19 tests) — closed-form Sunde
  homogeneous-soil reference (positive U_T below U_E, U_T → U_E
  at remote earth, |U_S| decay to remote earth, sign of U_S
  matching the radial potential gradient), cylindrical symmetry
  of the envelope around a vertical rod, envelope vs. pointwise
  cross-check, EN 50522 Table B.4 anchor-point exact match,
  monotone decrease, log-log interpolation geometric-mean check,
  terminal 85 V plateau between 5 s and 10 s, endpoint clamping,
  validation of distance / step / direction / unknown-electrode
  error paths, and a top-level export check.

- **World generator framework + AP1 TN-Ortsnetz generator
  (ADR-0009).** New subpackage `groundfield.generators` with:
  * `GeneratorConfig` — Pydantic v2 base for generator
    configurations. Numerical / categorical fields accept either a
    fixed value or a :class:`Distribution`. ``cfg.sample(rng)``
    walks the model and resolves every distribution to a concrete
    value, recursing into nested `GeneratorConfig` instances and
    list/tuple containers; ``cfg.has_distributions()`` is the
    introspection counterpart.
  * `WorldGenerator(Generic[C])` — abstract base. Concrete
    generators implement `build(cfg)` and inherit RNG wiring,
    `sample_world(rng)`, and the `_assert_resolved` guard.
  * `groundfield.generators.distributions` — distribution catalogue
    `Constant`, `Uniform`, `Normal` (with rejection-sampling
    truncation), `LogNormal` (with `from_moments` constructor),
    `Weibull`, `Discrete`, `Categorical`. All are Pydantic v2
    models with a literal `kind` discriminator and a `.sample(rng)`
    method; the `AnyDistribution` discriminated union enables
    JSON round-tripping inside `GeneratorConfig` fields.
  * `TnNetworkGenerator` (+ `TnNetworkConfig` and the four sub-
    configs `SoilConfig`, `TrafoStationConfig`,
    `HouseElectrodeConfig`, `PenConfig`) — first concrete
    generator. Parameterises the AP1 axes from
    `999_projektmanagement/arbeitspakete/AP1_tn_ortsnetz.md`:
    $n_\text{EFH} \in \{5, 10, 30, 80, 200\}$, small / medium
    commercial buildings, cable-cabinet quota
    $q$ per 100 EFH, two-layer soil $(\rho_1, \rho_2, h_1)$,
    house electrode kind drawn per house from a
    `Categorical({"foundation", "rod", "mesh"})`. Topology:
    Manhattan-grid house placement → cable cabinets along the
    substation row → PEN backbone (substation ↔ KVS, KVS ↔
    nearest house) as a *distributed conductor* (ADR-0003), with
    optional `inductance_model="neumann"` (ADR-0004) and
    selectable `coupling_to_soil`.
- **ADR-0009** (`docs/adr/0009-world-generators.md`) — design
  rationale, architecture (ABC + Pydantic config), distribution
  layer, sampling semantics, topology contract, validation
  programme.
- `tests/test_distributions.py` (24 tests) — reproducibility under
  seed, statistical sanity (mean/std within 5 % over 10 000
  samples), JSON discriminated-union round-trip per
  distribution, validation of malformed inputs (negative weights,
  inverted bounds, non-positive scale, duplicate categorical
  values), and rejection-sampling exhaustion on truncated Normal.
- `tests/test_generators_base.py` (15 tests) — `resolve_value`
  pass-through, `has_distributions` recursion, `cfg.sample`
  resolution at top-level / nested / categorical, idempotence on
  fully-fixed configs, reproducibility under seed, build-side
  guard against unresolved configs, RNG wiring, JSON round-trip
  on a mixed-distribution config.
- `tests/test_tn_ortsnetz.py` (10 tests) — smallest-preset build &
  solve, default-segment-budget at $n_\text{EFH}=30$,
  bit-exact reproducibility under fixed seed, stochastic
  reproducibility under fixed seed, categorical-mix electrode
  kinds, $|Z| \propto \rho_1$ monotonicity at low frequency for
  $n_\text{EFH} \in \{5, 10, 30\}$, JSON round-trip preserving
  every distribution kind, edge cases (zero houses, unknown
  electrode kind), KVS quota reproduction.
- Notebook `notebooks/20_tn_ortsnetz_generator.ipynb` — deterministic
  AP1 reference build (10 EFH foundation electrodes), three
  side-by-side stochastic samples with a Categorical kind mix,
  full $n_\text{EFH} \times \rho_1$ heatmap at 50 Hz showing the
  Dwight asymptote $|Z| \propto \rho_1$ and the parallel-cluster
  scaling with $n_\text{EFH}$, JSON round-trip of the stochastic
  config with a bit-exact replay of the seed-42 sample.

### Changed

- **`build_inductance_matrix` is now vectorised
  (ADR-0010 Tier 0b).** The Python double loop over $M(M-1)/2$
  segment pairs is replaced by a row-at-a-time NumPy assembly:
  pair-wise dot products in a single $O(M^2)$ multiplication,
  closed-form Grover formula vectorised over the parallel pairs,
  16×16 Gauss–Legendre quadrature batched over the non-parallel
  pairs (peak memory $O(M \cdot 256 \cdot 24)$ bytes per row).
  Reproduces the legacy loop bit-exactly to floating-point
  precision (max relative drift ~ $10^{-12}$ — pure summation
  roundoff). Empirical speed-up **5–10×** across $M \in [25,
  400]$ on a typical laptop, scaling cleanly. The legacy
  loop is kept as `_build_inductance_matrix_loop` for
  regression-test reference.
- `tests/test_inductance_vectorised.py` (8 tests) — bit-exact
  regression against the legacy loop on hand-crafted mixed
  geometry, on random non-degenerate segments at $M \in \{5, 20,
  60\}$ for both `use_image=True/False`, on a fully-parallel
  grid (closed-form-only path), plus symmetry / positive-
  diagonal / single-segment / empty / zero-length-segment edge
  cases.
- **ADR-0010** (`docs/adr/0010-tier0-performance.md`) — design
  rationale and validation programme. Documents 0b as
  implemented, 0a (LU caching across frequencies) and 0c
  (geometry-adaptive discretisation) as scoped follow-ups.
- Notebook `notebooks/22_tier0_speedup.ipynb` — bit-exact
  cross-check between loop and vectorised implementation, plus
  a speed-up scan over $N$ for both random-geometry (mixed
  parallel + quadrature path) and all-parallel-grid
  (closed-form-only path) inputs.

### Docs

- **`docs/performance.md`** — comprehensive performance and
  scaling guide. Empirical wall-clock characteristics from
  Notebook 21 (Sommerfeld is ~1200× slower than Carson at AP1
  frequencies and gives identical answers for parallel-wire
  geometries; `segment_length ≤ 1 m` mandatory for AP1
  accuracy; PEN-Neumann is ~1 % at 50 Hz at 3× cost). Wall-clock
  estimates per AP1 study size (5 EFH up to 200 EFH).
  Monte-Carlo strategy with a fully worked `joblib.Parallel`
  pattern, throughput estimates per study size, and tips for
  reproducible / resumable runs. Roadmap pointer to ACA + GMRES
  with a clear threshold for when it becomes worth the
  implementation effort.
- **Eight ground-up `docs/examples/`** — a guided tour:
  * 01 First solve (a single rod, Sunde reference);
  * 02 Substation grounding (ring + 4 rods, Dwight 1936
    reference);
  * 03 TN-Ortsnetz generator basics;
  * 04 AP1 Analysis 1 — galvanic measurement and the 62 % rule;
  * 05 AP1 Analysis 2 — inductive coupling on the measurement
    leads, frequency-dependent error;
  * 06 Deterministic parameter sweep over $\rho_1$;
  * 07 Monte-Carlo sweep with `joblib`, persistent Parquet
    storage, statistical bands;
  * 08 Full pipeline `groundfield` → ρ-f fit → `groundinsight`
    `BusType` for downstream fault analyses.
  Each page is self-contained and runnable end-to-end. Linked
  from `examples/index.md`, navigation in `mkdocs.yml`.

### Added

- **`TnNetworkConfig.source_magnitude_A` accepts a Distribution.**
  The driving current at the substation cluster (test current for
  measurements, fault current for fault simulations) was previously
  a plain ``float``; it now also takes any
  :class:`Distribution`. Sweep e.g. measurement vs. fault on the
  same world via
  ``source_magnitude_A=Discrete(values=[1.0, 5000.0])`` or define
  a Monte-Carlo current with ``LogNormal.from_moments(...)``.
  Three new tests in ``tests/test_measurement.py`` cover the
  pass-through, the discrete-distribution case, and seed
  reproducibility.

- **Measurement-setup layer for AP1 grounding-resistance studies
  (Analysis 1 + 2).** New module
  `groundfield.generators.measurement` with:
  * `MeasurementLeadConfig` — one physical measurement lead
    (overhead at surface or buried cable) modelled as a finite-
    impedance :class:`Conductor`. Default settings:
    ``coupling_to_soil="isolated"``,
    ``inductance_model="neumann"`` so the lead generates a
    magnetic field that couples to every parallel conductor (PEN,
    cable shields, the parallel measurement lead).
  * `MeasurementInjectionConfig` — auxiliary current electrode
    (Hilfserder) at a configurable remote position with its own
    :class:`GroundingSystemSpec`, plus an optional
    `feed_lead` (`None` = AP1 Analysis 1 galvanic only;
    a :class:`MeasurementLeadConfig` enables AP1 Analysis 2
    inductive coupling).
  * `MeasurementProbeConfig` — voltage probe (Spannungssonde) at a
    configurable position with its own grounding (default: short
    rod at the 62 % point), plus an optional metallic measurement
    `lead`.
  * `MeasurementSetupConfig` — top-level measurement spec used as
    the new optional `TnNetworkConfig.measurement` field. When
    set, the generator builds the aux electrode, the voltage
    probe, the configured leads, and re-routes the source's
    `return_to` to the aux anchor — the test current physically
    returns through the auxiliary electrode (and, with metallic
    leads enabled, mostly through them). When `None` (default)
    the source returns through *remote earth* and no aux/probe
    are added.
  * Convenience factories `overhead_lead()` (surface bare copper,
    Neumann-coupled) and `buried_lead(depth_m=0.6)` (cable shield),
    plus `single_rod_grounding()` (Erdungsspieß) and
    `neighbour_substation_grounding()` (ring + 4 rods, for
    measurements against a neighbour substation).
- `tests/test_measurement.py` (10 tests) — default-no-measurement
  hygiene, galvanic-only setup adds aux + probe but no leads,
  source `return_to` wiring to aux anchor, inductive setup adds
  both leads with `inductance_model="neumann"` active, lead
  helper inspections, neighbour-substation aux variant, smallest
  preset solves with measurement enabled, JSON round-trip.
- Notebook 20 grows section **9. Erdungsmessaufbau (AP1 Analyse
  1 + 2)** with three sub-sections: 9a galvanic-only Surface
  potential overlay, 9b inductive-coupling case with overhead
  feed and probe leads + Sommerfeld earth-return correction, and
  9c neighbour-substation aux variant.

- **`plot_surface_potential(result, world, …)`** — new helper in
  :mod:`groundfield.postprocess.plotting` that plots
  $\varphi(x, y, z)$ over the *entire world* instead of over the
  point-source bounding box. Default extent is derived from
  :func:`world_bounds_xy(world)` plus a configurable
  ``padding_m`` (default 15 m), so all buildings, cable cabinets
  and the substation appear together with a strip of remote-earth
  decay around them. Options: ``log`` (multi-decade colour scale
  on $|\varphi|$), ``symmetric`` (signed $[-\varphi_\text{max}, +\varphi_\text{max}]$
  scale around 0), ``show_electrodes`` overlay,
  ``show_contour_lines`` for iso-potential annotations, custom
  ``cmap`` / ``figsize`` / ``title``.
- :func:`world_bounds_xy(world)` — public helper that returns
  the smallest axis-aligned $(x, y)$ bounding box of the world's
  electrodes. Each electrode kind is unwrapped to its true
  footprint (rods → point, rings → enclosing square, strips →
  endpoint bbox, grid meshes → ``corner``-to-``corner+size``).
- Notebook 20 grows section **8. Surface-Potential über die
  gesamte Welt** with three plots on the existing 5-EFH
  minimal-example: linear scale with iso-lines, log $|\varphi|$
  for the boundary decay, and a wide (60 m) padding view that
  shows how far the field still influences the *remote earth*.
- ``tests/test_api_smoke.py`` (4 new tests) — `world_bounds_xy`
  bounding-box correctness, default-extent honouring on
  `plot_surface_potential`, log-mode smoke, explicit-extent
  override.

### Changed

- **`_draw_electrodes` no longer annotates electrode names** in
  the potential / surface-potential plots. In dense AP1 networks
  the per-electrode labels overlapped heavily and obscured the
  contours. The geometric outlines (rod markers, ring/strip lines,
  mesh wires) stay; callers that genuinely need labels can iterate
  ``world.electrodes`` and call ``ax.annotate`` themselves. Affects
  :func:`plot_potential_contour` and :func:`plot_surface_potential`.

- **`FoundationElectrodeSpec` gains an explicit ring/mesh style.**
  New field ``style: Literal["ring", "mesh"] = "mesh"``:
  ``"ring"`` realises only the rectangle's perimeter (closed wire
  loop, no internal cross-bracing — the *Ringerder*-style
  foundation electrode common in residential buildings),
  ``"mesh"`` (default) keeps the previous behaviour with
  perimeter plus ``n_x × n_y`` internal cross-braces (the
  classical *Maschenerder*). Internally both styles materialise as
  a :class:`GridMeshElectrode`; ``"ring"`` forces ``n_x = n_y = 1``
  (one mesh cell = perimeter wires only). Tests and the variant
  catalog notebook (notebook 20) are updated.
- **Notebook 20 plot rendering**: the foundation / grid-mesh
  electrode is now drawn with all its longitudinal *and*
  transverse wires (instead of just the outer rectangle), so the
  internal cross-bracing is visible at a glance. Adds a new "Ring
  vs. Mesh" variant section that compares the three common
  realisations (perimeter only, 2×2 mesh, 4×4 mesh) side by side.

- **`TnNetworkGenerator` refactored onto a composable spec layer
  (ADR-0009 v2).** The flat v1 config (`n_efh`, `house_electrode`,
  hardcoded substation ring + rods, single-rod KVS) is replaced
  by five reusable spec modules:
  * `electrode_specs` — discriminated union `ElectrodeSpec`
    (`RodElectrodeSpec`, `RingElectrodeSpec`, `StripElectrodeSpec`,
    `FoundationElectrodeSpec`) with `presence_prob` and
    `offset_xy_m`. Helper `rod_circle(n, radius_m, …)` returns N
    rods on a circle — the typical substation Tiefenerder layout.
  * `grounding.GroundingSystemSpec` — `electrodes:
    list[ElectrodeSpec]` plus `build_at(world, site_xy,
    name_prefix, rng)`. Bonds every present electrode into one
    cluster. Used identically by substation, KVS and every
    building.
  * `placement` — discriminated union `PlacementSpec`
    (`ManhattanGridPlacement` with optional jitter,
    `ExplicitPlacement` with caller-supplied coordinates).
  * `soil_specs` — discriminated union `SoilSpec`
    (`HomogeneousSoilSpec`, `TwoLayerSoilSpec`,
    `MultiLayerSoilSpec`) with distributions per parameter and a
    `to_soil(rng)` materialiser.
  * `building.BuildingTypeSpec` + `default_building_catalog()` —
    AP1-typical four-type catalog (`residential`,
    `small_industry`, `medium_industry`, `large_industry`) with
    distinct grounding systems.
  The new `TnNetworkConfig` exposes `soil`, `substation`, `kvs`,
  `placement`, `building_types`, `building_counts`, `pen`,
  `source_magnitude_A`. Substation grounding is now any AND/OR
  combination of ring / rods / strip / foundation; KVS grounding
  the same; per-building grounding is type-driven and supports
  multi-electrode systems (foundation + extra rod, ring + grid +
  strips, …) with per-electrode `presence_prob` for stochastic
  fleets.
- **`TnNetworkGenerator.build` adopts lazy distribution
  resolution.** The previous strict guard
  `_assert_resolved(cfg)` rejected any unresolved
  `Distribution` field and prevented the per-house Categorical
  electrode-kind mix from being drawn at build time. The
  generator now resolves top-level distributions lazily (one
  draw per `build` call) and samples per-instance fields
  (currently `house_electrode.kind`) once per house. Strict
  upfront resolution is still available via
  `gen.sample_world(rng)` or a manual `cfg.sample(rng)` —
  see ADR-0009 for the documented trade-off (`sample_world`
  collapses the Categorical to a single kind for all houses;
  `build` gives the per-house mix).

### Deprecated

- **`groundfield.generators.tn_ortsnetz`** module path. Use
  `groundfield.generators.tn_network` instead, or import
  `TnNetworkGenerator` / `TnNetworkConfig` from the top-level
  `groundfield` namespace. The old module re-exports the new
  classes under their old names (`TnOrtsnetzGenerator`,
  `TnOrtsnetzConfig`) and emits a `DeprecationWarning`. The
  rename is part of aligning the public surface with the
  project-wide English-only naming convention; the AP1
  work-package keeps its German name *TN-Ortsnetz* in the
  research documentation and file paths.

### Removed

- The flat v1 `TnNetworkConfig` schema (`n_efh`,
  `n_klein_gewerbe`, `n_mittel_gewerbe`,
  `kvs_per_100_efh`, `kvs_x_max_m`, `n_houses_per_row`,
  `house_spacing_m`, `row_spacing_m`, the v1 `house_electrode`
  block with its `kind` Categorical) is gone. Use
  `building_types` + `building_counts`, plus the new sub-configs
  `substation`, `kvs`, `placement`. Notebook 19 (groundinsight
  bridge) is unaffected; only generator-using code paths need
  updates.

- **`groundinsight` bridge — `BusType` export from a `rho-f` fit
  (ADR-0008).** New module `groundfield.io.groundinsight` that
  closes the family pipeline `groundfield → groundinsight`:
  * `BusTypeSpec` — neutral, in-memory representation of an exported
    `BusType` carrying name, description, system_type, voltage_level,
    the SymPy `impedance_formula` string, the parallel sample table
    `(frequency_Hz, rho_Ohm_m, Z_real_Ohm, Z_imag_Ohm)`, and a
    free-form `metadata` block with the fit method, fit quality,
    coefficients/poles, source-package version and creation
    timestamp.
  * `to_bustype_dict(fit, ...)` / `save_bustype_json(fit, path, ...)`
    / `load_bustype_json(path)` — JSON path with a versioned schema
    (`schema = "groundfield.bustype"`, `schema_version = 1`). Works
    without `groundinsight` installed.
  * `to_bustype(fit, ...)` / `save_bustype_to_db(fit, ...)` —
    Python-API path that returns a live
    `groundinsight.models.core_models.BusType` Pydantic instance via
    a lazy import of `groundinsight`. Raises a clear `ImportError`
    pointing at the optional install
    `pip install groundfield[groundinsight]` when the package is
    missing.
  * `evaluate_spec(spec, frequencies, rho)` — re-evaluate an exported
    formula at arbitrary `(f, rho)` points without round-tripping
    through `groundinsight`.
  * Symbol convention: `RhoFStandardFit` exports natively in
    `(rho, f)`; `VectorFitResult` exports via the symbolic
    substitution $s \to j\,2\pi f$ so the resulting formula matches
    the `groundinsight.BusType.impedance_formula` parser.
- **`groundinsight` as an optional dependency** (extras group
  `[groundinsight]` in `pyproject.toml`). `pip install groundfield`
  works as before; `pip install groundfield[groundinsight]` enables
  the live-`BusType` path.
- **ADR-0008** (`docs/adr/0008-groundinsight-bridge.md`) — design
  rationale, JSON schema (v1), symbol convention, optional-dependency
  strategy, and the validation programme (symbol round-trip via
  `groundinsight.compute_impedance`, JSON round-trip,
  optional-dependency hygiene, end-to-end notebook).
- `tests/test_io_groundinsight.py` (12 tests): schema and dict
  shape, JSON round-trip with bit-exact sample preservation, schema
  rejection (wrong name / future version),
  `groundinsight.compute_impedance` symbol round-trip for both fit
  methods (`< 1e-9` for `rho_f_standard`, `< 1e-3` for
  `vector_fit`), end-to-end `BusType → Bus → calculate_impedance`
  consistency, `evaluate_spec` consistency, optional-dependency
  hygiene (`to_bustype` raises `ImportError` when `groundinsight`
  is unimportable while the JSON path keeps working).
- Notebook `notebooks/19_groundinsight_export.ipynb` — AP1-style
  end-to-end demonstration: transformer station with ring earth +
  rods on a 2-layer soil, $\rho_1$ sweep, `RhoFStandardFit`,
  `BusType` export to JSON and to a live `groundinsight.Network`,
  comparison of the field-grade impedance against the reduced
  formula evaluation.

- **Vector fitting + SymPy export for the rho-f model.** New module
  `groundfield.postprocess.vector_fitting` with:
  * `vector_fit(frequencies, Z_values, n_poles, ...)` — clean
    Gustavsen/Semlyen 1999 vector-fitting implementation (single-
    output, complex/real poles, optional R_∞ and L_∞ residuals).
  * `VectorFitResult` dataclass with poles, residues, and
    `evaluate(frequencies)` for re-evaluation.
  * `fit_to_sympy(fit, decimals)` — produces a SymPy expression
    in a single free symbol `s`, with complex-conjugate pole pairs
    combined into real second-order terms. Compatible with
    `groundinsight.BusType.impedance_formula`.
  * `rho_f_from_field_result(result, electrode_name, n_poles)`
    — convenience wrapper that takes a `FieldResult`, computes
    `Z(s) = U(s)/I(s)` per frequency, and runs the fit.
- `tests/test_vector_fitting.py` — synthetic ground-truth recovery
  (real and complex poles), evaluate round-trip, SymPy export
  consistency, and end-to-end FieldResult flow.
- **ADR-0006 / ADR-0007 Phase B (n=2 cross-layer in
  mom_sommerfeld, cim, bem).** The Sommerfeld off-diagonal kernel
  in `mom_sommerfeld.sommerfeld_kernel_value` now dispatches to
  `coupling.layered_green.two_layer_real_space_kernel` for any
  source/observer pair on a 2-layer soil. Combined with the
  shared cross-layer-aware diagonal from
  `_two_layer_self_kernel_factory` (Phase A), this lifts the
  cross-layer precondition in `mom_sommerfeld`, `cim`, and `bem`
  for `n_layers == 2`. `n_layers >= 3` still emits a documented
  `UserWarning`.

- **Cross-layer electrodes and conductors (ADR-0007, Phase A).**
  Lifts the long-standing precondition `z_max < h_1` in the layered
  backends. Driven rods (Tiefenerder), foundation electrodes, deep
  meshes and any conductor that crosses the upper-layer interface
  can now be computed directly. The implementation:
  * **`groundfield.coupling.layered_green`** — new module with
    `two_layer_spectral_kernel` and `two_layer_real_space_kernel`
    that solve the 2-layer matching problem numerically and
    produce the rigorous Green's function for any
    (source-layer, observer-layer) pair.
  * **`solver/image_2layer`** automatically dispatches to the
    Sommerfeld Green's function for cross-layer geometries while
    keeping the historic Tagg/Sunde image series as a fast path
    for pure-upper-layer worlds (bit-exact regression preserved).
  * **`_Segment.layer_index`** — new internal field tagging which
    layer each segment lives in (0 = upper, 1 = next, …).
  * **`mom_sommerfeld`, `cim`, `bem`** emit a `UserWarning` instead
    of raising on cross-layer geometries, pointing the user at
    `image_2layer` as the Phase A path. Phase B will extend their
    kernels too (planned).
- **ADR-0007** (`docs/adr/0007-cross-layer-electrodes.md`) —
  derivation of the cross-layer matching problem, three-phase
  rollout (A: discretiser + uniform numerical kernel; B: closed-
  form image series for ll and ul; C: n ≥ 3 layers), and
  validation programme.
- `tests/test_cross_layer.py` — homogeneous-limit checks for
  `layered_green`, reciprocity, image_2layer cross-layer
  acceptance test, regression test for pure-upper-layer worlds,
  warning behaviour test for `mom_sommerfeld / cim / bem`.
- Notebook `notebooks/17_cross_layer_ap1.ipynb` — AP1-realistic
  driven-rod-through-interface sweep showing the spreading-
  resistance drop when the rod tip reaches the conductive lower
  layer; ρ₂ = ρ₁ limit check.

### Changed

- The `z_max >= h_1` precondition in `image_2layer` is no longer
  fatal: the backend transparently dispatches to the Sommerfeld
  cross-layer kernel. Behaviour for in-bounds geometries is
  unchanged.
- `mom_sommerfeld`, `cim`, `bem`: same precondition replaced by a
  `UserWarning` that names the recommended workaround.

- **Geometric Sommerfeld earth-return Green's function (ADR-0006).**
  New `Engine.earth_inductive_model="sommerfeld"` option that
  integrates the σ-dependent vector-potential Green's function over
  the actual segment-pair geometry. Rigorous for arbitrary wire
  lengths and orientations; supports **layered earth natively**
  (Pollaczek/Wait kernel) without warnings. In the long-parallel-
  wires + homogeneous-earth limit it converges (on the cluster
  level) to ADR-0005's per-meter Carson asymptote.
- **`groundfield.coupling.sommerfeld_inductance`** — new module
  with:
  * `LayeredEarth` — frozen layered-earth dataclass.
  * `reflection_coefficient_homogeneous` /
    `reflection_coefficient_layered` — magnetic Fresnel /
    Pollaczek-Wait coefficients $\Gamma_\text{mag}^{(n)}(\lambda)$.
  * `earth_return_correction_homogeneous` /
    `earth_return_correction_layered` — point-wise σ-dependent
    Green's function correction (uses Lipschitz–Hankel identity
    in the σ→0 limit).
  * `sommerfeld_pair_integral_homogeneous` /
    `sommerfeld_pair_integral_layered` — 16×16 Gauss–Legendre
    outer integration over a segment pair, vectorised inner
    Sommerfeld quadrature with split log-then-uniform λ-grid that
    resolves both the σ-transition and the Bessel oscillations.
  * `build_sommerfeld_correction_matrix` — dense $M\times M$
    correction matrix consumed by all distributed-capable
    backends, drop-in alongside `build_carson_correction_matrix`.
- **`groundfield.coupling.resolve_earth_layers`** — extracts a
  `LayeredEarth` from any soil model. No warning for layered
  configurations — they are first-class citizens.
- All distributed-capable backends (`image`, `image_2layer`,
  `mom`, `mom_sommerfeld`, `cim`, `bem`) consume the new
  `"sommerfeld"` switch and pass the layered-earth configuration
  into the per-frequency builder. `fem` continues to log a
  warning (its equivalent-hemisphere reduction is DC only).
- **ADR-0006** (`docs/adr/0006-sommerfeld-earth-return.md`) —
  derivation of the magnetic Green's function, three-regime limit
  checks, two-pillar API (homogeneous Pillar A this release,
  layered Pillar B same release), numerical strategy, validation
  programme, and the explicit hand-off from ADR-0005's per-m
  asymptote to this geometric formulation.
- `tests/test_sommerfeld_inductance.py` — reflection-coefficient
  limits, Lipschitz–Hankel σ→0 identity, σ→∞ collapse,
  ω→0 DC reproducibility, Sommerfeld-vs-Carson agreement on a
  long PEN at 50 Hz (cluster level), no-warning behaviour on
  TwoLayerSoil, layered-vs-homogeneous deviation, cross-engine
  consistency over `image / mom / cim / bem`.
- Notebook `notebooks/16_carson_vs_sommerfeld.ipynb` — wire-length
  sweep showing where Carson's per-m asymptote breaks down,
  layered-earth comparison Carson(top-rho) vs.
  Sommerfeld(layered), AP1-relevant TN-Ortsnetz example.

### Changed

- `Engine.earth_inductive_model` accepts a third value
  `"sommerfeld"` (in addition to `"perfect_mirror"` and
  `"carson_series"`). Default unchanged. ADR-0005's Carson series
  is now documented as the *asymptotic* option; ADR-0006's
  Sommerfeld kernel is the *geometric* option for AP1-grade
  studies.
- `_assemble_inductance_matrix` signature: gains an optional
  `layered_earth` argument (used when `earth_model="sommerfeld"`).

- **Carson earth-return correction (ADR-0005).** Adds the Carson 1926
  finite-conductivity correction $\Delta Z_\text{Carson}(\omega)$ on
  top of the perfect-mirror Neumann inductance from ADR-0004. New
  `Engine.earth_inductive_model` field with values `"perfect_mirror"`
  (default, bit-exact reproduction of ADR-0004) and `"carson_series"`
  (Carson 1926). The branch-impedance block becomes
  $Z_b(\omega) = R + j\omega L_\text{Neumann} + \Delta Z_\text{Carson}(\omega)$
  with the Carson correction rebuilt at every frequency.
- **`groundfield.coupling.carson`** — three-regime evaluation of
  Carson's $J(p, q) = P(a, \theta) + jQ(a, \theta)$ following the
  original 1926 paper:
  * `_p_q_small` — leading-term form for $a \le 0.25$ (Carson eqs. 34/35),
  * `_p_q_quadrature` — direct 64-point Gauss–Legendre numerical
    quadrature of Carson eq. 29 for $0.25 < a \le 5$ (replaces the
    classical Tleis recurrence with a robust numerical kernel),
  * `_p_q_large` — asymptotic expansion for $a > 5$ (Carson eqs. 36/37),
  * `carson_p_q`, `carson_self_correction`,
    `carson_mutual_correction` — public API,
  * `deri_semlyen_correction` — Deri/Semlyen 1981 complex-depth
    approximation (sanity-check, not the production path),
  * `skin_depth`, `carson_parameter` — diagnostic helpers.
- **`groundfield.coupling.resolve_earth_conductivity`** — extracts
  $\sigma_\text{earth}$ from a soil model. Exact for
  `HomogeneousSoil`, approximate (upper-layer $\rho_1$ with a
  `UserWarning`) for `TwoLayerSoil` and `MultiLayerSoil`.
- **`groundfield.coupling.inductance.build_carson_correction_matrix`**
  — assembles the dense complex Carson correction matrix
  $\Delta Z_\text{Carson}(\omega)$ over distributed-conductor branches
  using midpoint-rule projection onto Carson's parallel-wire formula
  (orthogonal components contribute zero by Neumann symmetry).
- All distributed-capable backends (`image`, `image_2layer`, `cim`,
  `mom`, `mom_sommerfeld`, `bem`) consume the new
  `earth_inductive_model` switch and rebuild the Carson correction
  matrix per frequency. `fem` logs a warning and ignores Carson
  (its equivalent-hemisphere reduction is DC only).
- `_assemble_inductance_matrix` returns an additional third value:
  a closure `omega -> dZ_carson(omega)` (or `None` for the
  perfect-mirror default). `_solve_cluster_currents` and
  `_galerkin_solve` accept a new `carson_correction` argument that
  is added to $j\omega L$ inside Block 3.
- **Penetration-depth diagnostic.** `FieldResult.metadata` now
  exposes `"penetration_depth"` — a `dict[float, float]` keyed by
  frequency that returns $\delta(\omega)$ in metres, populated by
  every backend that runs an inductive-coupling frequency loop.
  Together with `"earth_inductive_model"` this lets notebooks
  diagnose "is my geometry small or large compared to $\delta$?"
  without re-deriving the formula.
- **`groundfield.references.carson`** — Carson 1926 Section V worked
  examples (wave antenna at $r = 4.0$ and $r = 0.4$, railway at
  $r = 0.2$, $\theta = 63°30'$) plus four self-consistent
  regression anchors in the intermediate and asymptotic regimes.
- **ADR-0005**
  (`docs/adr/0005-carson-earth-return.md`) — Carson series
  derivation, three-regime split, linear-system integration, soil
  conductivity source, layered-earth handoff to the Pollaczek
  follow-up (deferred), and the full validation programme.
- `tests/test_carson_coupling.py` — Carson 1926 worked examples,
  regime-boundary continuity, $\sigma\to\infty / \omega\to 0$
  limits, skin-depth/Carson-parameter relation, Deri/Semlyen
  cross-check, engine-side regression and frequency-dependence
  tests, two-layer warning, 1 km PEN textbook benchmark, and
  cross-engine consistency at 50 Hz.
- Notebook `notebooks/15_carson_correction.ipynb` — Carson functions
  across the three regimes, skin-depth diagnostic, 1 km PEN self
  impedance perfect-mirror vs. Carson, mutual-coupling open-circuit
  voltage, cross-engine table at 50 Hz / 1 kHz.
- `scripts/benchmarks/pen_1km_carson.py` — stand-alone CLI tool
  that reproduces the AP1-typical 1 km PEN benchmark (self
  impedance, mutual to a parallel measurement lead, all four
  distributed-capable backends, side-by-side closed-form
  Oeding+Carson reference).

### Changed

- `Engine` schema gains the `earth_inductive_model` field (default
  `"perfect_mirror"`). Existing notebooks and tests keep their
  results bit-exact because the default reproduces ADR-0004.
- `_assemble_inductance_matrix` signature: returns a 3-tuple
  `(L_full, has_inductance, carson_builder)` instead of the previous
  2-tuple. All in-tree callers updated; out-of-tree subclasses must
  unpack the third element.

### Added

- **Inductive coupling (ADR-0004).** Distributed conductors now
  carry a partial inductance per longitudinal segment. With
  `Conductor.inductance_model = "neumann"` the solver assembles
  the Neumann self- and mutual-inductance matrix
  $L \in \mathbb{R}^{M \times M}$ over every distributed-conductor
  branch and plugs it into the branch block as
  $Z_b(\omega) = R + j\omega L$. The resulting complex linear
  system is solved per frequency in `engine.frequencies`. With
  `inductance_model = None` (default) the system stays real and
  the historic DC fast path is preserved bit-exact. Earth is
  modelled as a perfect magnetic mirror in this release; finite-
  conductivity (Carson) correction is the subject of ADR-0005.
- **`groundfield.coupling.inductance`** — partial-inductance
  helpers used by the assembly:
  * `thin_wire_self_inductance` — closed-form Grover formula for a
    straight thin wire,
  * `parallel_segments_mutual` — closed-form mutual inductance for
    two equal-length parallel coaxial segments,
  * `neumann_mutual` — generic Neumann double-line integral with a
    closed-form fast path for parallel segments and a 16×16
    Gauss–Legendre quadrature fallback for arbitrary 3-D geometry,
  * `build_inductance_matrix` — assembles the dense per-branch
    matrix including the perfect-mirror image contribution.
- All distributed-capable backends (`image`, `image_2layer`, `cim`,
  `mom`, `mom_sommerfeld`, `bem`) now run a frequency loop when
  inductive coupling is active. The `_solve_cluster_currents` and
  `_galerkin_solve` helpers grew matching `omega` and
  `inductance_matrix` parameters; the inductance matrix is built
  once before the loop. `fem` logs a warning and falls back to
  the resistive solution (its equivalent-hemisphere reduction is
  DC only).
- New helper `_assemble_inductance_matrix` and a typed
  `_DistributedBranch` dataclass that carries every branch's
  endpoints and wire radius, so the inductance assembly works
  uniformly across backends.
- `FieldResult.metadata["conductor_node_currents"]` and
  `["conductor_node_potentials"]` now hold one entry **per
  frequency** (matching the public `electrode_currents` /
  `electrode_potentials` shape).
- **ADR-0004**
  (`docs/adr/0004-inductive-coupling.md`) — derivation, linear
  system, validation programme, and explicit hand-off to the
  Carson follow-up.
- `tests/test_inductive_coupling.py` (12 tests):
  Grover self-formula, parallel-mutual closed-form vs. Neumann
  quadrature, perpendicular = 0, anti-parallel sign flip, DC
  reproducibility, $\omega \to 0$ collapse to the resistive
  solution, frequency-induced phase shift on the source-rod
  current, frequency-scaling open-circuit voltage on a parallel
  measurement lead (the AP1 question), cross-engine consistency
  at 50 Hz, FEM warning-and-fallback test.
- Notebook `notebooks/13_inductive_coupling.ipynb` — Neumann
  helpers validation, frequency sweep on the source-rod current,
  loop-coupling open-circuit voltage profile, and the
  cross-engine table at 50 Hz.
- **`groundfield.references.oeding`** — closed-form per-unit-length
  loop self- and mutual-inductance formulas from Oeding & Oswald
  (2016) *Elektrische Kraftwerke und Netze*, ch. 9 (Eq. 9.8, 9.9,
  9.13c) plus the helpers
  `two_wire_loop_radius` ($r' = r\,e^{-1/4}$) and
  `internal_inductance_per_length` ($\mu_0/(8\pi)$). Used as the
  textbook reference against which the Neumann implementation is
  validated in the long-wire limit.
- `thin_wire_self_inductance` gained an `include_internal` flag
  (default `True`). The internal-field contribution
  $\mu_0\,\ell/(8\pi)$ is now added to the Grover external term by
  default, so that summing two parallel-wire partial inductances
  reproduces Oeding Eq. (9.13c) including the +1/4 term. Pass
  `include_internal=False` to keep the pure external-field
  expression for high-frequency / skin-effect studies.
- `tests/test_oeding_inductance.py` (10 tests): direct algebraic
  checks of `oeding.*`, plus four parametric Neumann-vs-Oeding
  9.13c regressions across (length, distance, radius), a
  $1/L$-convergence test, and two loop-to-loop mutual checks
  against Eq. 9.8 (in-line and offset geometries).
- Notebook `notebooks/14_oeding_validation.ipynb` — wire-length
  sweep showing the $1/L$ convergence of the Neumann
  partial-inductance combination to Eq. 9.13c, parameter sweep
  over $(d, r)$, and loop-to-loop mutual against Eq. 9.8.

### Changed

- `metadata["conductor_node_currents"]` and
  `metadata["conductor_node_potentials"]` are now
  `dict[str, list[complex]]` (one entry per
  `engine.frequencies`) instead of `dict[str, complex]`. The shape
  is now consistent with the public `electrode_*` mappings;
  callers should index `[k]` for the *k*-th frequency.

### Added

- **Distributed conductor model (ADR-0003).** `Conductor` now
  carries two new fields, `discretize_segment_length` (m) and
  `coupling_to_soil` (`"isolated"` | `"galvanic"`). When the
  segment length is finite the conductor is split into
  $n = \lceil L / \Delta s\rceil$ collinear sub-pieces, each with
  its own longitudinal current. With
  `coupling_to_soil="galvanic"` every midpoint additionally leaks
  current into the soil through the same Green's-function kernel
  as electrode segments — turning a buried bare-copper conductor
  or an exposed cable shield into a true distributed earth
  electrode. The augmented linear system in
  `_solve_cluster_currents` and `_galerkin_solve` couples the
  enlarged multi-port grounding matrix $Z$ (over electrode +
  galvanic conductor segments) with KCL per cluster node and
  Ohm's law on each longitudinal segment branch. Per ADR-0003 the
  longitudinal impedance is purely resistive in this release;
  inductive and Carson corrections come with ADR-0004.
- New helpers `_discretize_conductor`,
  `_build_distributed_topology`, `Conductor.is_distributed`,
  `Conductor.n_segments`. The cluster builder
  `_build_clusters` and the lumped-branch builder
  `_build_finite_branches` skip distributed conductors so they
  are routed exclusively through the new topology builder.
- All seven integral-equation backends (`image`, `image_2layer`,
  `image_nlayer` via dispatch, `cim`, `mom`, `mom_sommerfeld`,
  `bem`) consume the distributed topology natively. `fem` falls
  back to a lumped branch and logs a warning, because its
  equivalent-hemisphere reduction is not defined for the small
  midpoint pseudo-electrodes the discretiser would produce.
- `FieldResult.metadata["conductor_node_currents"]` and
  `["conductor_node_potentials"]` expose the per-midpoint leakage
  and EPR for galvanic distributed conductors. The public
  `electrode_currents` / `electrode_potentials` mappings stay
  restricted to real electrodes for backwards compatibility.
- **ADR-0003** (`docs/adr/0003-distributed-conductor-model.md`)
  documenting the linear system, the boundary conditions, the
  validation strategy, and the deferred items
  (inductive coupling → ADR-0004).
- `tests/test_conductors_distributed.py` (13 tests):
  schema additions, lumped-fallback regression, isolated chain
  series-resistor equivalence (n=2/4/8/16), galvanic chain
  current conservation and convergence in `n_seg`, cross-engine
  consistency on `image`, `mom`, `cim`, `bem`, FEM
  warning-and-lumped-fallback test, three-rod chain end-to-end
  test.
- Notebook `notebooks/12_distributed_conductor.ipynb`:
  algebraic-equivalence sweep, EPR + leakage profile along a
  galvanic chain, convergence in `n_seg`, side-by-side
  cross-engine table, `StripElectrode` vs distributed-conductor
  cluster-impedance comparison.
- **Finite-impedance conductor branches** — first step toward the
  AP1 distributed-conductor model. `Conductor` now carries a
  `cross_section` field (in m²; `None` keeps the historic ideal
  galvanic short, `"from_radius"` resolves to π·r²). Finite
  conductors enter the solver as branches with series resistance
  $R_\text{ser} = \rho_\text{mat} L / A$ in an augmented
  nodal-analysis system that couples the multi-port grounding
  matrix $Z$ with Kirchhoff's current law per cluster node and
  Ohm's law per branch:
  ```
  [ Z      -C       0    ] [ I_e    ]   [ 0       ]
  [ C^T    0        B^T  ] [ phi_n  ] = [ I_in    ]
  [ 0      B       -R_b  ] [ I_b    ]   [ 0       ]
  ```
  For `cross_section is None` the branch block is empty and the
  system collapses bit-exactly to the legacy
  cluster-equipotential constraint.
- New helpers `Conductor.series_resistance`,
  `Conductor.effective_cross_section`, `Conductor.is_ideal()` and
  the routing helper `solver.image._build_finite_branches`. The
  cluster builder `_build_clusters` now ignores finite conductors
  by design — they stay separate clusters and are reattached as
  branches.
- Branch support in **all eight backends**: `image`, `image_2layer`
  and `cim` route through the augmented `_solve_cluster_currents`;
  `mom`, `mom_sommerfeld` and `bem` route through the augmented
  `_galerkin_solve`; `image_nlayer` inherits via dispatch; `fem`
  adds a small per-cluster nodal analysis on top of the
  equivalent-hemisphere reduction.
- `tests/test_conductors_finite.py` (15 tests): regression of the
  `cross_section=None` default, closed-form two-rod / one-branch
  split, $R_b \to 0$ and $R_b \to \infty$ limits, monotonicity
  in $R_b$, three-rod chain transitive activation, and
  cross-engine consistency for every backend on a homogeneous and
  on a 2-layer soil (≤ 3 % vs. image/image_2layer reference, ≤ 7 %
  for FEM through its equivalent-hemisphere reduction).
- Notebook `notebooks/11_finite_conductor.ipynb`: two-rod
  analytical check, sweep over $R_b$, five-rod PEN chain with
  per-rod EPR profile, and side-by-side ideal vs. finite
  comparison.
- Pydantic v2 data model: `World`, `BoundaryConditions`,
  `HomogeneousSoil`, `TwoLayerSoil`, `MultiLayerSoil`, `RodElectrode`,
  `RingElectrode`, `MeshElectrode`, `Conductor`, `CurrentSource`,
  `VoltageSource`.
- `Engine` (backend selection `image | image_2layer | mom | fem`,
  frequency list, tolerances) and `FieldResult` as the unified
  result object with `point_sources` (discretised current
  distribution) and `result.potential(points)` for evaluating the
  potential at arbitrary field points.
- **Image backend** for homogeneous soil: image-charge sum over
  discretised wire segments, line self-action correction. Returns
  grounding impedances within < 5 % of the Sunde formula for typical
  driven rods.
- **Cluster logic** for galvanically connected electrodes:
  conductors with electrode end-points form a shared potential
  cluster. Current sharing within a cluster is solved exactly via
  the multi-port grounding matrix (constraint: φ_i = const ∀ i ∈
  cluster, Σ I_i = I_input).
- `Conductor.start_electrode` / `end_electrode` (Pydantic fields),
  set automatically by `create_conductor`.
- `FieldResult.clusters` (mapping electrode → cluster members) and
  `FieldResult.cluster_impedance(name)` for the physically
  interpretable cluster grounding impedance (parallel combination).
- Top-level factories `create_world`, `create_electrode`,
  `create_conductor`, `create_source`, `create_engine`,
  `run_simulation`.
- Post-processing plots: `plot_potential_contour` (xy or xz slice),
  `plot_potential_profile` (potential along an arbitrary line for
  several depths), `plot_potential_radial` (trumpet around an
  electrode).
- Smoke-test notebook `notebooks/01_smoke_test.ipynb` with Sunde
  comparison, "without vs. with connection" section
  (current/potential table, before/after contour, line profile),
  and the full plot family.
- pytest suite `tests/test_api_smoke.py` (22 tests): parametric
  Sunde comparison, monotonicity, cluster current balance,
  cluster-impedance plausibility, plot smoke tests.
- **`groundfield.references.dwight1936`** — module with all the
  closed-form grounding formulas from Dwight, H. B., *Calculation
  of Resistances to Ground*, AIEE 1936 (Tab. I): rod, rod pairs
  (close / far), buried horizontal wire, right-angle, 3/4/6/8-point
  star, ring, strip, round and vertical plate, hemisphere. Every
  formula reproduces the worked examples published in the paper to
  within 0.2 Ω.
- `tests/test_dwight_references.py` (19 tests) — image-backend vs.
  Dwight per geometry. 10 % tolerance for image vs. Dwight, < 0.5 Ω
  for module vs. paper.
- **ADR-0001** (`docs/adr/0001-two-layer-method.md`) — analysis and
  decision regarding the numerical methods for 2-layer soil: two
  engines side by side (Tagg/Sunde image-charge series + MoM with
  Sommerfeld quadrature). Rationale, convergence, action items.
- **`gf.compare_engines(world, engines, ...)`** and the
  `EngineComparison` report — cross-engine self-validation per
  ADR-0001. Tests in `tests/test_compare_engines.py` (6 tests).
- **Engine A `image_2layer`** — Tagg/Sunde image-charge series for
  2-layer soil (see ADR-0001, action items 1, 4, 6). Adaptive
  series truncation at `|K|^n < 1e-6` (max. 100 terms), series
  diagnostics in `FieldResult.metadata` (`K`, `n_terms_used`,
  `converged`). Precondition: every electrode must lie within the
  upper layer, otherwise a clear `ValueError` is raised.
- **Auto-dispatch** in `Engine.solve`: `backend="image"` with a
  world that holds a `TwoLayerSoil` transparently switches to
  `image_2layer` — notebooks do not need to change the backend
  string when the soil model changes.
- **`FieldResult.soil`** as a new optional field. `FieldResult.potential`
  picks the appropriate Green's-function kernel automatically
  (homogeneous or 2-layer). For homogeneous solutions the existing
  `soil_resistivity` continues to suffice — no breaking migration.
- Tests in `tests/test_two_layer.py` (8 tests): `ρ₁=ρ₂` limit,
  sign of K, auto-dispatch, precondition check, series convergence,
  cross-engine sanity.
- Notebook `notebooks/02_two_layer.ipynb` with parameter sweeps
  over K and h₁, trumpet comparison homogeneous vs. 2-layer,
  K=0 sanity plot.
- **Engine B `mom` — Galerkin Method-of-Moments backend**
  (`solver/mom.py`). Independent second engine for cross-validation
  (ADR-0001 action item 2). Builds the full segment-level reaction
  matrix and solves a (N+K)×(N+K) linear system for the per-segment
  current distribution and the cluster potentials, instead of
  assuming uniform per-unit-length currents like the image
  backends. Supports both `HomogeneousSoil` and `TwoLayerSoil`
  through the same Green's-function kernels as the image backends;
  only the resolution scheme differs.
- Cross-engine tests in `tests/test_cross_engines.py` (15 tests):
  image vs. mom on homogeneous worlds (< 2 %), image_2layer vs. mom
  on 2-layer worlds across the full K range (< 2 %), exact K=0
  collapse, multi-electrode cluster cross-check, potential-field
  consistency.
- Notebook `notebooks/03_cross_engine.ipynb` demonstrating the
  side-by-side validation: cluster-impedance sweep over K, surface
  potential profiles, and a `compare_engines` programmatic check.
- Logging helpers `groundfield.utils.configure` / `get_logger`.
- **`solver/_layered.py`** — shared layered-soil helpers:
  `LayerStack`, `as_layer_stack`, `reflection_gamma` (recursive
  $\Gamma_1(\lambda)$ from per-interface Fresnel coefficients),
  `image_series_offsets`, `cylindrical_radius`. Used by every new
  layered backend.
- **Engine `image_nlayer`** (`solver/image_nlayer.py`) — n-layer
  image-charge dispatcher. Forwards `n=1` to `image`, `n=2` to
  `image_2layer`, raises `ValueError` for `n ≥ 3` with a clear
  pointer to `cim` / `mom_sommerfeld` / `bem`. Auto-selected from
  `backend="image"` when the world holds a `MultiLayerSoil`.
- **Engine `cim`** (`solver/cim.py`) — Complex Image Method via
  matrix-pencil fit of $\Gamma_1(\lambda)$ as a sum of complex
  exponentials. Closed-form for any layer count once the fit is
  done. Exposes the fit through
  `result.metadata["cim_n_images"]` and `cim_rms`.
  Public helper: `fit_complex_images(stack, …)`.
- **Engine `mom_sommerfeld`** (`solver/mom_sommerfeld.py`) — Galerkin
  MoM with **direct numerical Sommerfeld quadrature** of the layered
  Green's function (`scipy.integrate.quad`). Slow but
  methodologically independent reference engine. Public helper:
  `sommerfeld_kernel_value(stack, s, z, z_s, …)`.
- **Engine `bem`** (`solver/bem.py`) — boundary-element collocation
  with the closed-form CIM kernel. Companion to `mom` (Galerkin)
  with a different test-function weighting.
- **Engine `fem`** (`solver/fem.py`) — axisymmetric volume PDE on a
  cylindrical $(s, z)$ triangular mesh, layer boundaries as
  element-wise conductivities, sparse `spsolve`. Reduces every
  cluster to its equivalent hemisphere via Dwight 1936 closed forms;
  exposes the reduction through
  `result.metadata["equivalent_hemisphere_radius"]`.
- **Backend literal extended** to
  `image | image_2layer | image_nlayer | cim | mom | mom_sommerfeld | bem | fem`,
  and `Engine.solve` auto-dispatches `backend="image"` to
  `image_nlayer` for `MultiLayerSoil` worlds.
- Test suite extended by ≈ 50 tests:
  `tests/test_image_nlayer.py` (5 tests),
  `tests/test_cim.py` (7 tests),
  `tests/test_mom_sommerfeld.py` (6 tests),
  `tests/test_bem.py` (6 tests),
  `tests/test_fem.py` (5 tests),
  `tests/test_cross_engines_extended.py` (≈ 25 parametric tests
  covering homogeneous-rod agreement, 2-layer engine agreement,
  two-rod cluster agreement, and layer-contrast monotonicity for
  every engine).
- Notebooks `04_image_nlayer.ipynb`, `05_cim.ipynb`,
  `06_mom_sommerfeld.ipynb`, `07_bem.ipynb`, `08_fem.ipynb`,
  `09_cross_engine_extended.ipynb`. Each contains a single-rod
  sanity, the user-mandated **two interconnected grounding systems**
  test case (potential profile, cluster impedance, current split),
  a layer sweep, and a self-checks block with explicit collapse /
  monotonicity verification.
- Notebook `10_trafostation.ipynb` — application-style cross-engine
  reference for the grounding system of a small transformer station
  (ring earth electrode + two driven rods + optional strip earth
  electrode in a 2-layer soil). All `(x, y)` electrode positions are
  exposed as parameters; rods and strip can be switched on/off via
  flags. Reports cluster impedance for every engine, `compare_engines`
  consistency check, and surface-potential 3-D and contour plots.
  Uses the native `StripElectrode` primitive.
- **`StripElectrode`** (`kind="strip"`) — first-class horizontal
  straight strip earth electrode (Banderder). One wire from `start`
  to `end` at a fixed depth, arbitrary in-plane direction. Replaces
  the prior degenerate-mesh workaround. Discretiser added to
  `solver/image.py:_discretize_strip` and shared with every
  integral-equation backend through the existing dispatcher.
  Plausibility test against `dwight1936.horizontal_wire` activates
  the previously reserved entry in `tests/test_dwight_references.py`.
  Dedicated tests in `tests/test_strip.py`.
- **`GridMeshElectrode`** (`kind="grid_mesh"`) — rectangular meshed
  earth electrode with explicit `n_x` × `n_y` inner meshes
  (`n_x + 1` transverse and `n_y + 1` longitudinal wires). Cleaner
  alternative to `MeshElectrode` (which keeps its uniform-spacing
  API for backwards compatibility). Discretiser shares
  `_grid_segments` with the legacy mesh. The FEM equivalent-hemisphere
  reduction for `GridMeshElectrode` uses the Schwarz / Sverak / IEEE
  Std 80 formula
  $R \approx \rho/L_C + \rho/\sqrt{20 A}\,(1 + 1/(1 + h\sqrt{20/A}))$
  so that the inner mesh density is reflected in the FEM cross-check
  (the legacy `MeshElectrode` keeps the prior strip-along-diagonal
  proxy).
- **ADR-0002** (`docs/adr/0002-engine-family.md`) — engine selection
  heuristic and cross-validation envelope.

### Changed

- `Engine.solve` now dispatches on `world.soil`. `solve_image`
  (private API) remains strictly limited to `HomogeneousSoil`; the
  friendly forwarding only happens inside `Engine.solve`.
- Internal refactoring: `_solve_cluster_currents` is now
  kernel-agnostic and takes a `self_kernel` closure, so both the
  homogeneous and the 2-layer backend share the same cluster logic.
- **Project language switched to English**: every Python source
  file, docstring, comment, Markdown documentation page, ADR,
  notebook, and CHANGELOG entry is now in English. This brings the
  project in line with the open-source conventions for
  international distribution. Variable names were already English;
  only prose changed. The repo-internal `CLAUDE.md` was updated to
  reflect the new language convention.

### Internal

- The `mom` backend now returns real results (Galerkin scheme on
  segment level). The `fem` backend now returns real results
  (axisymmetric volume PDE with equivalent-hemisphere reduction);
  the prior stub is removed.
- Bugfix in `_two_layer_self_kernel_factory`: the `extra` accumulator
  was hard-wired to a 1-D shape, which broke the matrix-mode call
  used by the new MoM backend (identity-matrix probe to obtain the
  full reaction matrix in one pass). Changed to `np.zeros_like(phi)`.
- Notebook file `02_zweischicht.ipynb` was renamed to
  `02_two_layer.ipynb`; ADR file `0001-zweischicht-verfahren.md`
  was renamed to `0001-two-layer-method.md`. MkDocs navigation and
  cross-references updated.

### Docs

- **`docs/graphify.md`** — new tooling page describing the
  hand-driven workflow for building a queryable knowledge graph of
  `groundfield` with [`graphify`](https://github.com/safishamsi/graphify)
  (PyPI: `graphifyy`, MIT) for use by AI coding assistants. Covers
  installation via `pipx` (so the ~25 transitive Tree-sitter language
  packages stay out of the project lock file), the build-vs-query
  cost split, the manual `extract` / `update` workflow, the
  recommended `graphify claude install` Claude-Code integration
  (which routes day-to-day queries through Claude Code's `Read`
  tool so they count against a Claude Pro / Max subscription
  rather than billed API tokens), the `--global` cross-repo graph
  spanning `groundfield`, `groundinsight` and `groundmeas`, and
  the limits (LaTeX / `.ipynb` narrative unparsed). The page is
  registered under a new **Tooling** section in `mkdocs.yml`.
- `.gitignore` — exclude the `graphify-out/` working cache so the
  locally generated graph (which depends on the chosen LLM backend
  and revision) is not committed.

### Internal

- **`scripts/generate_graphify_report.py`** — workaround that renders
  `graphify-out/GRAPH_REPORT.md` from the JSON artefacts that
  `graphify extract` *does* write (`.graphify_analysis.json` plus
  `graph.json`). Necessary because `graphify` 0.7.x silently stopped
  emitting the Markdown report itself, while the
  `graphify claude install` PreToolUse hook and the matching
  `CLAUDE.md` directive both still reference that file as the entry
  point AI coding assistants should read first. The script is
  AST/JSON-only — no network, no API access, no LLM call — and is
  intended to be removed once upstream `graphify` re-emits the
  Markdown report directly. Run via
  `poetry run python scripts/generate_graphify_report.py` after
  every `graphify extract` invocation.

No code, public API or solver behaviour is touched by this entry —
`graphifyy` is intentionally **not** added to `pyproject.toml`,
mirroring the way other developer tools (`black`, `ruff`, `mypy`)
are kept out of library dependency graphs.

---

## [0.1.0] — 2026-04-24

Initial project skeleton for `groundfield`.

### Added

- Package layout analogous to `groundinsight`, with
  `src/groundfield/` and the subpackages `soil`, `geometry`,
  `conductors`, `solver`, `coupling`, `postprocess`, `io`, `utils`.
- Poetry configuration (`pyproject.toml`) with dev and docs groups.
- MkDocs Material scaffold (`mkdocs.yml`, `docs/`) for the future
  documentation site on GitHub Pages.
- Release automation (`scripts/release.py`) and third-party
  license report (`scripts/generate_third_party_licenses.py`)
  ported from `groundinsight`.
- GitHub-Actions workflows `ci.yml`, `docs.yml`, `release.yml`.
- `CITATION.cff`, `LICENSE` (MIT), `.gitignore`, `CLAUDE.md`
  (context for AI assistants).
- First smoke test (`tests/test_import.py`).

### Internal

- Project shell as the foundation for **work package 1** of the
  dissertation on networked grounding systems: TN distribution
  network with layered soil, house connections, and cable cabinets.

---

## Backlog (2026-05-10 audit)

Items identified in the cross-cutting code-review pass on
2026-05-10 that did **not** ship in [Unreleased]. Each entry is
queued for a future maintenance release; once an item ships, the
matching CHANGELOG line moves into [Unreleased]. The list is
kept here (rather than in [Unreleased]) so the published release
notes describe shipped work only.

### Fixed (pending implementation)

> The following bugs and inconsistencies were identified in the
> code-review pass on 2026-05-10. They are queued for the next
> maintenance release; the entries are recorded here so each one
> ships with a referenced fix commit.

- **`references.oeding` and `references.carson` are not
  re-exported on the `groundfield` package**, while
  `references.dwight1936` is. The asymmetry surprises users who
  expect `gf.oeding.loop_self_inductance_per_length(...)` to
  exist after seeing `gf.dwight1936.hemisphere(...)`.
- **`coupling.resolve_earth_conductivity` raises a generic
  `TypeError`** for unsupported soil models. Use the package's
  own exception type (or document the type explicitly), and
  cover the failure path in `tests/test_carson_coupling.py` —
  currently the only call sites are inside the engine builders
  and the failure is never exercised.
- **`coupling/inductance._build_inductance_matrix_loop` is
  exported in `__all__` with a leading underscore** — `from
  groundfield.coupling.inductance import *` therefore does not
  expose it (Python skips underscore names) and the only caller
  is the regression test. Either drop it from `__all__` or
  rename without the leading underscore.
- **`generators.tn_ortsnetz` re-export is documented as
  deprecated** but the deprecation warning is only emitted when
  the *module* is imported, not when individual symbols are
  accessed via `from groundfield.generators.tn_ortsnetz import
  TnOrtsnetzGenerator`. Add a `__getattr__` so the symbol-level
  access also warns. Plan the removal target (e.g. `0.5.0`).
- **`generators.measurement.MeasurementSetupConfig` mutates
  `source.return_to`** of the parent `TnNetworkConfig` during
  `build`. If the same `TnNetworkConfig` instance is reused for
  a non-measurement build afterwards, the now-stale `return_to`
  pointer leaks into the second world. Either deep-copy the
  source spec inside the generator, or document the
  one-shot-config contract.
- **`coupling/sommerfeld_inductance.build_sommerfeld_correction_matrix`
  uses a fixed 16 × 16 Gauss–Legendre node count** for the
  outer pair integration. There is no mechanism to refine the
  grid when the dimensionless length is large; the docstring
  promises convergence but the production path runs at a
  single resolution. Either expose the resolution knob or
  document the upper bound where the fixed grid still resolves
  the kernel.
- **`solver/result.FieldResult.cluster_impedance(name)` returns
  one `complex` per frequency** but the summary docstring at
  the top of `result.py` describes the helper as "scalar". Fix
  the summary docstring; the per-call docstrings are correct.

> Additional findings from the **second 2026-05-10 review pass**:

- **`generators.measurement.MeasurementSetupConfig.build`** has
  no early-out if both `feed_lead` and `probe.lead` are
  `None`; it still walks every spec and conditionally skips
  the lead creation. The early-out makes the "no inductive
  coupling, just galvanic Analysis 1" case explicit and
  protects against future regressions where a new lead-related
  field is added.
- **`solver.image._build_finite_branches` and the augmented
  multi-port system in `_solve_cluster_currents`** silently
  ignore the case where a `Conductor.cross_section` resolves
  to a non-positive value. The `Conductor.series_resistance`
  helper does check for `<= 0` and falls back to the ideal
  short, but the path is not exercised in
  `tests/test_conductors_finite.py`. Add a regression test for
  `cross_section = 0.0` and `cross_section = "from_radius"`
  with `radius_m = 0` so future refactors of the helper cannot
  introduce a silent NaN.
- **`coupling/sommerfeld_inductance.sommerfeld_pair_integral_layered`**
  uses a split log-then-uniform λ-grid (changelog
  describes the strategy) but the split point is hard-wired
  to the per-pair σ-transition. For pairs with very large
  dimensionless length the linear part of the grid silently
  under-samples the Bessel oscillations. Expose the
  cross-over knob and lock it down with a notebook plot of
  the integrand to make the convergence visible.
- **`postprocess.convergence.convergence_study` clones the
  engine via `engine.model_copy`** but does not deep-copy the
  optional `frequencies` list. If a caller passes the same
  Python list as `frequencies` to two engines, mutating one
  list will silently mutate the cloned engine's list too.
  Either deep-copy or document that `engine.frequencies` is
  treated as immutable.
- **`generators.distributions.Discrete` weights are not
  normalised eagerly**, so a `Discrete(values=[1, 2],
  weights=[10, 5])` calls `numpy.random.choice` with
  un-normalised weights (numpy normalises internally, but the
  un-normalised weights stay on the model and round-trip
  through JSON). Add a `model_validator(mode="after")` that
  normalises weights to sum to 1 and remembers the original
  sum in `metadata` if a future audit needs it.
- **`io/csv.save_potential_path_csv`** writes a header line
  with `frequency_Hz` even when the saved frame contains a
  single frequency. Callers piping the output back into the
  multi-frequency reader path get a one-row-per-frequency
  layout regardless of source dimensions — fine, but the
  docstring should explicitly state the long-format contract.
- **`io/vtk.export_field_vtk`** silently truncates the field
  on `n=(120, 120)` even for very large worlds (a 200-EFH AP1
  network spans ~150 m × 150 m; 120 grid points gives ~1.25 m
  resolution, comparable with `engine.segment_length` and
  therefore aliased). Either raise a `UserWarning` when the
  grid spacing exceeds `engine.segment_length / 2` or expose
  an `auto_grid=True` heuristic.
- **`postprocess/current_balance.split_factor` raises on
  *any* source whose magnitude evaluates to zero** at
  `frequencies[frequency_index]`. For frequency sweeps where
  the source is intentionally inactive at some harmonics
  (e.g. `{50: 1.0, 250: 0.0}`) the helper aborts the whole
  table. Return `complex("nan")` (or a documented sentinel)
  for the zero-magnitude rows instead.
- **`generators.measurement.single_rod_grounding` / 
  `neighbour_substation_grounding`** signatures accept no
  RNG, so a stochastic placement (a `Uniform` over the rod
  position) cannot reproduce a previous draw. Add an
  optional `rng` parameter that defaults to a deterministic
  default (e.g. `np.random.default_rng(0)`) so test cases
  stay reproducible.
- **`postprocess.geometry_plot.plot_world_3d`** uses
  `ax.invert_zaxis()` after every call, but does **not**
  store / restore the original `zlim` if `ax` is passed in by
  the caller. Successive `plot_world_3d` calls on the same
  axis flip the z-axis back and forth.
- **`solver/result.FieldResult.summary` is still missing a
  docstring** as flagged in the 2026-05-09 audit; the second
  pass confirms that the public method has not been touched.
  Re-list here so it is not lost between audits.

> Additional findings from the **third 2026-05-12 review pass**
> (focus: `FieldResult.potential` dispatch, `BoundaryConditions`
> wiring, `references.dwight1936` validation, `compare_engines`
> coverage; deeper read of the `postprocess.sweep` heatmap
> plotter and the new measurement-injection layer):

- **`sources.Source.attached_to` is not validated against the
  world's electrode / conductor list.** A typoed
  `attached_to="g_1"` (instead of `"g1"`) silently drops the
  source current at solve time, because the backends iterate
  with `if src.attached_to in elec_input_current:` and just
  skip unknown keys. Raise a clear `KeyError` in
  `World.add_source(...)` or at the top of every backend's
  `solve_*`.
- **`solver/result.FieldResult.potential` (and the VTK export)
  mirror about a hard-coded `z = 0`** even though
  `safety.touch_voltage` exposes a `surface_z` argument. If the
  user sets up a world with the soil surface at any other plane
  (local-terrain coordinate), the image charges sit at the wrong
  depth. Either add `surface_z` to `FieldResult.potential` /
  `io.vtk.export_field_vtk` or document `z = 0` as a hard
  invariant.
- **`conductors.Conductor` accepts `start == end` (zero
  length).** No `model_validator` rejects coincident endpoints,
  and the discretisation step `_discretize_conductor` later
  divides by `L`, producing a hard `ZeroDivisionError` for any
  user who set `discretize_segment_length` on such a conductor.
  Add `@model_validator(mode="after")` that raises if
  `length < 1e-9 m`.
- **`validation.compare_engines` checks `Z.real` only**
  (`validation.py:~192,197`). For Carson / Sommerfeld engines or
  any inductive-coupling configuration the imaginary part can
  dominate at higher frequencies — a 30 % imag-part deviation
  goes undetected. Also `if zmean == 0.0: continue` silently
  skips clusters with symmetric real-part sign cancellation.
  Compare `abs(Z)`, or compare `Z.real` and `Z.imag` separately
  with the same tolerance.
- **`generators/distributions.Discrete` allows duplicate
  `values`.** Unlike `Categorical` (which has `_check_unique`),
  `Discrete(values=[5, 5, 10], weights=[1, 1, 1])` is accepted
  and silently double-weights `5`. Either merge duplicates +
  add their weights, or add a `_check_unique` validator and
  raise.
- **`io/vtk._format_lines_block` is dead code** — defined but
  never called; `export_geometry_vtk` re-implements the same
  logic inline. Either delete the helper or have the exporter
  use it.
- **`references.dwight1936` validation is inconsistent.** Only
  `rod`, `two_rods_far` and `hemisphere` validate `length > 0` /
  `radius > 0`; `two_rods_close`, `horizontal_wire`,
  `right_angle_wire`, `n_point_star`, `buried_ring`,
  `horizontal_strip`, `horizontal_round_plate` and
  `vertical_round_plate` accept any input. Calling
  `dw.buried_ring(rho=100, ring_diameter=-1, ...)` returns a
  meaningless float rather than raising. Apply the same
  `length > 0` / `radius > 0` guard at every entry point.
- **`postprocess/sweep.plot_sweep_heatmap` uses `imshow(...,
  extent=(min, max, ...))` on a `pivot_table`.** Non-uniformly
  spaced axis values (e.g. log-spaced soil resistivity
  `[10, 100, 1000]`) render as equally-wide rectangular cells —
  visually misleading. Switch to
  `ax.pcolormesh(pivot.columns, pivot.index, pivot.values)`
  which honours the actual coordinate spacing.

### Changed (pending implementation)

- **`generators/measurement.MeasurementInjectionConfig.feed_lead`
  uses `Optional[MeasurementLeadConfig]`** while the rest of
  the module uses `MeasurementLeadConfig | None`. Pick one and
  apply consistently across the spec layer.
- **`generators/distributions.LogNormal.from_moments`** —
  verify that its constructor takes the *standard deviation*
  (consistent with `Normal`) rather than the variance, and that
  the docstring matches the implementation. Two adjacent
  docstrings disagree on whether `sigma` or `var` is the
  expected argument.
- **`postprocess/geometry_plot.plot_world_3d`** uses
  `mpl_toolkits.mplot3d` directly — the inverted z-axis is
  correct under `%matplotlib inline` but occasionally flips
  with `widget`/`ipympl` backends. Add a regression notebook
  cell + screenshot to
  `notebooks/20_tn_ortsnetz_generator.ipynb`.

### Docs (pending implementation)

The package now has a sprawling public surface that is only
partially mirrored in the docs site. The next docs sweep should:

- **Add `docs/api/references.md`** — the `references` subpackage
  (`dwight1936`, `oeding`, `carson`) is reachable via
  `gf.dwight1936.*` and via direct `from groundfield.references
  import *`, but neither the API navigation nor the doc index
  links to it. The classical reference formulas are exactly
  what a reviewer would look up first.
- **Add `docs/api/validation.md`** — `gf.compare_engines` and
  `EngineComparison` are top-level exports but only mentioned
  in passing in `docs/api/index.md` and in the engine theory
  pages. Promote to a dedicated page.
- **Add `docs/api/world.md`** — `World`, `BoundaryConditions`
  and the `world.summary()` helper are top-level exports
  without their own page.
- **`docs/api/postprocess.md` does not yet have dedicated
  sections for the safety, current-balance and geometry-plot
  modules** — only `vector_fitting` and `rho_f_standard` are
  written up. Add headings and `:::` directives so the new
  helpers (touch / step voltage, EN 50522 limit, cluster /
  electrode current tables, `split_factor`, `plot_world` /
  `plot_world_3d`) become part of the navigable reference.
- **`docs/quickstart.md` and `docs/concepts.md` were updated in
  the previous audit pass** to cover all eight backends — the
  next step is to extend `docs/concepts.md` with a dedicated
  section on the **current-balance / split-factor**
  postprocessing (today the only entry point is
  `examples/04_grounding_measurement.md`).
- **`docs/performance.md` references Notebook 21 (Sommerfeld vs
  Carson scan)** — verify the link still resolves after the
  ADR-0010 cleanup.
- **`docs/api/coupling.md`** still uses
  `resolve_earth_conductivity` and `resolve_earth_layers` as
  the only documented entry points to the soil-resolver layer,
  even though the layered-Green-function and Sommerfeld
  inductance modules have grown several public helpers. Add a
  dedicated heading per public helper so mkdocstrings produces
  a discoverable index.
- **`docs/api/postprocess.md` refers to `notebooks/01..22`** in
  prose; the `mkdocs.yml` `examples/` nav lists eight
  hand-curated example pages but does not enumerate the raw
  notebooks. Decide whether to publish the full notebook
  collection under a dedicated nav section
  (`Notebooks: notebooks/01_*.ipynb` ... ) or to keep curation
  as today.
- **`docs/api/diagnostics.md` example uses `PenConfig`** — the
  symbol moved to `generators.tn_network`; verify the import
  path still works and update the example if not.
- **README** does not yet mention the world generator, the
  measurement-setup layer, the safety helpers or the
  current-balance helpers — the public surface has grown
  without a corresponding README rewrite.
- **mkdocs `extra_javascript: polyfill.io` reference**
  (`mkdocs.yml:90`) — the `groundmeas` sister project removed
  this URL because the domain was sold and the CDN later
  served malicious JavaScript. The same removal should happen
  here; modern browsers do not need the polyfill for MathJax 3.

> Additional doc gaps from the **second 2026-05-10 review pass**:

- **No `docs/api/generators/measurement.md` sub-page.** The
  measurement-setup layer (`MeasurementLeadConfig`,
  `MeasurementInjectionConfig`, `MeasurementProbeConfig`,
  `MeasurementSetupConfig` and the four convenience
  factories `overhead_lead`, `buried_lead`,
  `single_rod_grounding`, `neighbour_substation_grounding`)
  is the largest single API addition in [Unreleased] but the
  only entry point on the doc site is its mention in
  `docs/api/generators.md`. Add a sub-page that explains
  Analysis 1 vs. Analysis 2, the
  return-path-via-`return_to` wiring, and the
  galvanic/inductive switch flow.
- **No `docs/api/postprocess/sweep.md` or
  `convergence.md` sub-pages.** Both are stand-alone modules
  with their own plot helpers and public DataFrames; they
  deserve dedicated headings under
  `docs/api/postprocess.md` rather than being lumped under
  `Postprocess`.
- **`docs/api/diagnostics.md`** describes
  `world_statistics`, `expected_segments` and
  `check_segment_resolution` but does not list the **AP1
  budget thresholds** (`5000` soft, `20000` hard) that the
  warning text refers to. Quote them from the source so the
  user reading the docs can match the warning before opening
  the file.
- **No `docs/engines/_index_table.md` style overview** of
  which backend supports which feature combination
  (`HomogeneousSoil` × `TwoLayerSoil` × `MultiLayerSoil` ×
  `inductance_model` × `earth_inductive_model` ×
  `distributed conductors`). Today the user has to read the
  six engine pages individually. A single capability matrix
  closes the gap.
- **README quickstart** still uses
  `gf.run_simulation(...)`-style imports while the
  `Engine.solve(world)` pattern is now the canonical entry
  point (every example notebook 01..22 uses it). Update.
- **`docs/installation.md`** — the optional
  `[groundinsight]` extra is declared in `pyproject.toml` and
  referenced from `docs/api/io.md`, but the install page does
  not say `pip install groundfield[groundinsight]`. Mirror
  the `pandapower` callout from `groundinsight/docs`.
- **`docs/concepts.md` "current balance and split factor"** —
  add a new section under Concepts that ties together
  `cluster_current_balance`, `electrode_current_table` and
  `split_factor`. The example currently lives only in
  `examples/04_grounding_measurement.md` and is therefore
  invisible to a reader who lands on Concepts first.
- **`docs/api/world.md` (still missing) blocks the
  `world.summary()` cross-layer warning entry** from the
  Roadmap below — `summary` does not have a doc page to
  attach the new behaviour to.
- **README Python-version badge** still reads
  `3.12 | 3.13`; `pyproject.toml` declares `^3.12` so
  3.14 is supported. Extend the badge so users do not assume
  3.14 is unsupported.
- **CLAUDE.md project notes for `groundfield`** still claim
  `0.1.0` while the public surface is several releases ahead.
  Either bump the field manually or hook
  `scripts/release.py` to update it from `pyproject.toml`
  (same drift problem the cross-repo audit identified in
  `groundinsight/CLAUDE.md` and `groundmeas/CLAUDE.md`).

> Additional doc gaps from the **third 2026-05-12 review pass**:

- **No `docs/api/sources.md` page.** `Source`,
  `CurrentSource`, `VoltageSource` and the new
  measurement-injection wiring are only documented inline in
  `docs/api/index.md`. A dedicated page lets us spell out the
  `attached_to` contract (which is currently silently
  best-effort — see the bug above) and the
  `source_type='voltage'` mutual-coupling restriction.
- **No `docs/api/boundary.md` page.** `BoundaryConditions` is
  in the public API but undocumented. Until the fields are
  consumed by any backend (see bug above), users should be
  told in writing that `world.set_boundary_conditions(...)` is
  decorative.
- **`docs/engines/fem.md` does not document the cluster
  equivalent-hemisphere approximation** (radii-add rule) or its
  error envelope when the cluster electrodes share a field.
  Without that note a reader comparing FEM with the integral
  backends sees a small but unexplained delta on dense
  clusters.
- **`docs/api/references.md` is still missing** as flagged in
  pass 2, *and* the test-suite-facing entry points
  (`references.carson.CarsonExample`, `all_examples()`,
  `REGRESSION_ANCHORS`) need to land on the page so
  contributors can find them when adding regression cases.
- **`docs/api/io.md` does not document
  `io.groundinsight.evaluate_spec` /
  `io.groundinsight.fit_quality_summary` /
  `BusTypeSpec.from_dict`.** All three are publicly importable
  helpers but the rendered page lists only the `save_*` /
  `to_*` family. Add a `:::` directive for each.
- **`docs/concepts.md` "World object → Boundary"** section
  should document the no-op status of `BoundaryConditions`
  explicitly until the fields are wired through. Two lines
  prevent the most likely "why is my Dirichlet not applied"
  support question.

### Tests (pending implementation)

- **No cross-engine consistency test that exercises the
  Sommerfeld earth-return path on layered soil for every
  distributed-capable backend at 50 Hz / 1 kHz / 5 kHz.** The
  current `tests/test_sommerfeld_inductance.py` covers the
  homogeneous-vs-Carson agreement at 50 Hz only.
- **No regression test for the `mutual_branches` skip in
  `coupling/sommerfeld_inductance` for voltage sources.** The
  changelog notes that voltage sources skip the mutual
  contribution and emit a one-time warning; add a test that
  asserts the warning text and that the skip indeed produces
  the no-coupling baseline.
- **`tests/test_distributions.py`** asserts mean/std within 5 %
  over 10 000 samples, but no test verifies that the
  rejection-sampling path on the truncated Normal does not
  exhaust silently for very narrow `(low, high)` windows.
  Already mentioned as a roadmap item; lock it in with a test.
- **`tests/test_generators_base.py`** does not exercise
  `cfg.has_distributions()` on the new
  `TnNetworkConfig.measurement` sub-config. Add a test that
  flipping a single measurement-lead field to a `Distribution`
  flips `has_distributions()` from `False` to `True`.
- **`tests/test_geometry_plot.py`** runs 18 smoke tests but none
  asserts that `plot_world` raises (or returns an empty figure
  cleanly) on an empty world with `extent=None` and
  `padding_m=0` (degenerate bounding box).

### Roadmap candidates (from the 2026-05-10 audit)

The following items emerged from the cross-cutting review of the
public surface, the docs site and the test suite. Triage into
the appropriate Roadmap section in the next planning cycle.

- **`Reduktionsfaktor` proper.** The galvanic `split_factor`
  documentation explicitly notes that the helper is *not* the
  German EVU/Schirmtechnik *Reduktionsfaktor* (the inductive
  coupling correction). The inductance backends in
  `coupling/inductance.py` are already in place — a thin
  wrapper that integrates the inductive contribution along the
  parallel-conductor path and returns the magnitude / phase of
  the additional EMF would close the gap.
- **`Network → groundinsight` round-trip** — given a multi-port
  Z matrix produced by `gf` (Kron-reduced from a field-grade
  world), assemble the corresponding `groundinsight.Network`
  with one bus per port. Symmetric counterpart to the
  `BusType` exporter.
- **EN 50522 clearing-time helper bundle** —
  `gf.assess_touch_voltage(result, world, electrode,
  t_clear_s)` returning a single namedtuple (`U_T_max`,
  `U_TP_limit`, `passes`). Trivial wrapper but missing today;
  the measurement-setup notebook (notebook 20 §9)
  re-implements the comparison inline.
- **`world.summary()` should report cross-layer electrodes
  explicitly** — today it lists every electrode kind and depth,
  but a user staring at a `(z_min, z_max)` pair has to mentally
  intersect with `soil.h_1` to know whether the ADR-0007
  cross-layer path is exercised.
- **Notebook front-matter** — five of the recent notebooks
  (12–22) carry similar names ("inductive coupling", "Carson",
  "Sommerfeld") that get hard to disambiguate in the file
  browser. Adding a one-line YAML front-matter (`adr: 0005`,
  `feature: tier0`) would let the docs build group them by
  topic.

> Additional roadmap candidates from the **second 2026-05-10
> review pass**:

- **`MeasurementSetup` → `ResultMeasurement` post-processor.**
  Today the AP1 measurement notebook (notebook 20 §9) reads
  the source / probe potentials by hand and computes
  `Z_meas = U_probe / I_inj`. Wrap the pattern into a
  post-processor that pulls the right potentials from a
  `FieldResult` and returns the measured-vs-true earthing
  impedance plus the 62 %-rule diagnostic. Closes the AP1
  Analysis 1 + 2 measurement-side workflow as a single call.
- **`gf.export_capability_matrix(file)`** — auto-generated
  Markdown of the engine × feature support matrix described
  in the docs gap above. Lives next to the existing
  `compare_engines` so the docs and the test suite share a
  single source of truth.
- **`postprocess.convergence.convergence_study` with two
  axes** — extend the current single-axis (`segment_length`)
  helper to a 2-D mesh-refinement sweep so the user can
  jointly tighten `segment_length` and (e.g.)
  `discretize_segment_length` for distributed conductors.
  The natural counterpart of `sweep.sweep` for refinement
  studies.
- **`generators.tn_network.TnNetworkGenerator.summary()`** —
  one-line text summary that mirrors `World.summary()`, but
  for the *config* (before sampling) and the *built world*
  (after sampling) side by side. Useful for the AP1 parameter
  notebooks where the user wants to confirm what was actually
  drawn before invoking the solver.
- **`io.groundinsight.BusTypeSpec.to_branchtype()` /
  `save_branchtype_to_db()`** — symmetric counterpart of the
  existing `to_bustype` family. Many `groundinsight` networks
  expose the cable shield / PEN trunk as a `Branch` whose
  type's `R_mutual_formula` / `M_mutual_formula` come from a
  field-grade `gf` solve. The full bridge is incomplete until
  `BranchType` is also exportable.
- **`postprocess.safety.touch_voltage_envelope` with a
  layered-soil correction.** The current envelope uses the
  scalar `result.potential(...)`; on a 2-layer soil the
  envelope should integrate the bus-current image series too.
  The kernel is already in `coupling.layered_green`; wire it
  in behind a `surface_potential_model="layered"` switch and
  document the speed cost.

> Additional test gaps from the **third 2026-05-12 review pass**:

- **No regression test that `FieldResult.potential` raises (or
  dispatches correctly) on a `MultiLayerSoil` world.** Build a
  3-layer fixture, call `result.potential((x, y, 0.0))` and
  assert the result is not silently produced by the homogeneous
  kernel.
- **No test that `World.add_source(src)` rejects a typoed
  `attached_to`** (e.g. `"g_1"` when the world has `"g1"`).
  Currently the bug surfaces only at solve time and only as a
  zero-injection result.
- **No regression test for `Conductor(start=p, end=p)`**
  (zero length). Once the validator above lands, this becomes
  a one-line `pytest.raises(ValidationError)` case.
- **No `compare_engines` test on the imaginary part.** Build a
  Carson coupling fixture, set the existing real-only check to
  pass, then add a `Z.imag` check at 1 kHz that *would* fail
  without the imag branch.
- **No `Discrete(values=[5,5,10], weights=[1,1,1])` test.** A
  one-line assertion that duplicates are rejected (or merged)
  pins the chosen behaviour from bug 7.

> Additional roadmap candidates from the **third 2026-05-12
> review pass**:

- **`FieldResult.potential(..., surface_z=0.0)` argument.**
  Allow the user to plug in a non-zero soil-surface plane in
  local-terrain coordinates. The plumbing already exists in
  `safety.touch_voltage`; surface it on the post-solve
  potential evaluator too so the VTK export and the safety
  helper agree by construction.
- **`fem.equivalent_hemisphere_radius` with mutual-resistance
  correction.** Tagg-style cluster equivalence: replace the
  pure radii-add rule with a per-electrode mutual-resistance
  matrix lookup so clusters of driven rods produce the right
  conductance.
- **`gf.evaluate_spec` / `gf.fit_quality_summary` as top-level
  re-exports.** Both are publicly importable helpers in
  `io/groundinsight.py` but reachable only via the fully
  qualified path. Promote next to `vector_fit` for symmetry.
- **`EarthInductiveModel` as a top-level re-export** — the
  Literal lives in `solver.engine.__all__` but not in
  `groundfield.__all__`; users writing type annotations have
  to dig into the submodule.
- **`postprocess.geometry_plot.plot_world_top_down(...,
  annotate=True)`** — for worlds with ≤ 30 electrodes the
  annotated plot is invaluable for debugging measurement-setup
  geometry. Add an opt-in `annotate` flag with a sensible
  `annotate_threshold` default.
- **`postprocess.sweep.plot_sweep_heatmap` log-spaced cell
  rendering.** Switch to `pcolormesh` so the rendered cell
  widths match the actual log-spaced axis (bug 10). Stand-alone
  visual fix, no API impact.

## Roadmap

Backlog of feature ideas that are not yet scheduled. Updated as
work package 1 progresses.

### Core functionality

- **Inductive coupling (ADR-0004 — to be written).** Now that the
  distributed-conductor topology is in place (see
  `[Unreleased]`/ADR-0003), the next step adds
  Neumann self- and mutual-induction integrals between conductor
  segments. Drops the quasi-static frequency limit from ≈ DC to
  ~1 kHz with frequency-dependent longitudinal impedance
  $Z_\text{long} = R + j\omega L$. Required for the AP1 question
  on coupling between the measurement lead and the current
  injection.
- ~~**Carson correction** for the earth-return path below 1 kHz —~~
  ~~AP1 question "diffusion field and Carson relevance for earth~~
  ~~currents".~~ → done in `[Unreleased]` as ADR-0005.
- **FEM support for distributed conductors.** The current FEM
  backend falls back to a lumped branch and logs a warning when
  it sees a distributed conductor — the equivalent-hemisphere
  reduction does not resolve the midpoint pseudo-electrodes. A
  full-3-D FEM (deferred below) would lift this restriction.
- Full-3-D FEM backend (via `scikit-fem` or comparable) replacing
  the equivalent-hemisphere reduction in the current `fem` engine.
  **Deferred** — explicitly *not* on the AP1 roadmap. The target
  AP1 geometries (~20 houses with foundation electrodes plus a
  transformer station with ring + driven rods plus driven rods at
  cable cabinets) already push the integral solvers towards the
  ~1000-segment regime where the dense $O(N^3)$ LU solve dominates;
  a 3-D volume FEM on the same configuration would be one to two
  orders of magnitude more expensive without adding physical
  insight. This task is therefore reserved for follow-up work where
  a multi-cluster volume computation is genuinely needed.
- Scaling the integral solvers to ~10 000-segment networks: replace
  the dense LU solve in `mom`, `bem` and `cim` by a preconditioned
  iterative solver (GMRES with a block-diagonal or fast-multipole
  preconditioner). Unlocks the full AP1 parameter space (large
  TN-Ortsnetz with foundation electrodes, drives, and cable
  cabinets) without giving up the per-segment current resolution.
  Concrete prerequisites: ACA (adaptive cross approximation) for
  the dense reaction matrix to bring memory from $O(N^2)$ down to
  $O(N \log N)$ and a block-Jacobi preconditioner per cluster.
- Vector fitting for the `rho-f` curve.

### Studies and tooling

- **`ParameterSweep` API** — wrap a `world_factory(**params)` plus a
  parameter grid into a parallel sweep (`joblib`-driven, target 12
  cores) that returns a `pandas.DataFrame` of cluster impedances,
  EPRs and per-electrode currents. Drives the AP1 statistical
  studies (soil layering, electrode count and position, Monte Carlo
  realisations).
- **TN-Ortsnetz topology generator** — helper that builds an
  AP1-style world from high-level parameters
  (#single-family / small-commercial / mid-commercial buildings,
  cable-cabinet ratio, soil layering). Generates electrodes,
  finite PEN sections, cable cabinets, and the transformer station
  in one call.
- **penetration depth** Calculate the depth of the earth current as it is used in Carson integrals. It should be possible to create the earth current depth of any soild multilayer problem to use an equivilent for the typical formulas for calculating the self and coupling impedances of a cable or overheadline with earth return part.

### Features

- Ingest open building map data to model foundation electrodes from
  a map slice. The user decides whether every building has its own
  grounding system, or whether they are sampled from a stochastic
  distribution. Per building type (mesh / foundation / driven rod
  or any combination) the geometry is determined.
- Connection conductors need a switch between euclidean routing and
  Manhattan routing (along x or y only, no diagonal). For
  distribution networks Manhattan is often more realistic because
  cables follow streets and footpaths.

### Integration

- Direct export of `BusType` and `BranchType` into `groundinsight`
  databases.
- Import of measurement geometries from `groundmeas` (positions,
  lengths, auxiliary-electrode set-up).

### Documentation and typing

- Reference-case library for AP1: clusters of single-family houses
  (5 / 10 / 30 / 80 / 200), small and medium commercial buildings,
  cable cabinets.
- Notebook suite that covers the full parameter space described in
  the dissertation proposal.

[Unreleased]: https://github.com/Ce1ectric/groundfield/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ce1ectric/groundfield/releases/tag/v0.1.0
