---
title: Review pass 3 (2026-05-12)
description: Review pass 3 (2026-05-12) of groundfield, as recorded in the changelog.
---

# Review pass 3 (2026-05-12)

| | |
|---|---|
| Date | 2026-05-12 |
| Recorded in | the backlog section at the end of the changelog |
| Overview | [Audit history](index.md) · [Changelog](../changelog.md) |

The third review pass appended its findings to the backlog. One block of test gaps stood under the roadmap-candidates heading of the backlog and is reproduced where it stood.

!!! note "Where this text comes from"
    This page was moved verbatim from `CHANGELOG.md` when the release notes were
    condensed in 2026.11.0 (source: commit `30a122d`). Only the markup was adapted:
    reST roles are rendered as code or math, relative links point to the current
    pages, trailing whitespace is removed, and each block stands under a heading
    that names where it was published.
    Paths, test names and numbers are those of the time; the code has changed since.

## From the backlog section of the changelog (“Backlog (2026-05-10 audit)”)

### Fixed (pending implementation)

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

### Docs (pending implementation)

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

### Roadmap candidates (from the 2026-05-10 audit)

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

### Roadmap candidates (from the 2026-05-10 audit)

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
