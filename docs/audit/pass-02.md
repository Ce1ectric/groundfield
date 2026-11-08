---
title: Review pass 2 (2026-05-10)
description: Review pass 2 (2026-05-10) of groundfield, as recorded in the changelog.
---

# Review pass 2 (2026-05-10)

| | |
|---|---|
| Date | 2026-05-10 |
| Recorded in | the backlog section at the end of the changelog |
| Overview | [Audit history](index.md) · [Changelog](../changelog.md) |

The second review pass appended its findings to the backlog of the first pass.

!!! note "Where this text comes from"
    This page was moved verbatim from `CHANGELOG.md` when the release notes were
    condensed in 2026.11.0 (source: commit `30a122d`). Only the markup was adapted:
    reST roles are rendered as code or math, relative links point to the current
    pages, trailing whitespace is removed, and each block stands under a heading
    that names where it was published.
    Paths, test names and numbers are those of the time; the code has changed since.

## From the backlog section of the changelog (“Backlog (2026-05-10 audit)”)

### Fixed (pending implementation)

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

### Docs (pending implementation)

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

### Roadmap candidates (from the 2026-05-10 audit)

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
