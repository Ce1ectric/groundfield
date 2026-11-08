---
title: Review pass 1 (2026-05-10)
description: Review pass 1 (2026-05-10) of groundfield, as recorded in the changelog.
---

# Review pass 1 (2026-05-10)

| | |
|---|---|
| Date | 2026-05-10 (the second pass refers back to findings “flagged in the 2026-05-09 audit”) |
| Reviewed | the development state between 0.1.0 and 0.2.0 |
| Recorded in | the backlog section at the end of the changelog |
| Overview | [Audit history](index.md) · [Changelog](../changelog.md) |

The first review pass is recorded in the changelog as “the cross-cutting code-review pass on 2026-05-10”. Its findings were filed in a backlog section at the end of the changelog, and passes 2 to 6 appended their own findings to that section. Entries without a pass label (the *Changed* and *Tests* categories) are reproduced on this page; the changelog does not say which of the first passes raised them.

!!! note "Where this text comes from"
    This page was moved verbatim from `CHANGELOG.md` when the release notes were
    condensed in 2026.11.0 (source: commit `30a122d`). Only the markup was adapted:
    reST roles are rendered as code or math, relative links point to the current
    pages, trailing whitespace is removed, and each block stands under a heading
    that names where it was published.
    Paths, test names and numbers are those of the time; the code has changed since.

## From the backlog section of the changelog (“Backlog (2026-05-10 audit)”)

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
