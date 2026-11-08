---
title: Review pass 8 (2026-05-25)
description: Review pass 8 (2026-05-25) of groundfield, as recorded in the changelog.
---

# Review pass 8 (2026-05-25)

| | |
|---|---|
| Date | 2026-05-25 |
| Recorded in | the release notes of 0.7.0 |
| Overview | [Audit history](index.md) · [Changelog](../changelog.md) |

The eighth review pass was a changelog-only pass; it was published in the 0.7.0 release notes.

!!! note "Where this text comes from"
    This page was moved verbatim from `CHANGELOG.md` when the release notes were
    condensed in 2026.11.0 (source: commit `30a122d`). Only the markup was adapted:
    reST roles are rendered as code or math, relative links point to the current
    pages, trailing whitespace is removed, and each block stands under a heading
    that names where it was published.
    Paths, test names and numbers are those of the time; the code has changed since.

## Findings, published in the 0.7.0 release notes

### Fixed (Backlog — eighth 2026-05-25 review pass)

> The eighth audit pass was run on 2026-05-25, ten days after the
> `0.6.0` release (`geo` subpackage + concrete-encasement model) and
> seven days after the seventh CHANGELOG-only audit pass. The
> seventh-pass *code* fixes (`EvaluateSpecError`, `World.reset_concrete_corrections`,
> `OsmBuildingPlacement.footprint_at` bounds-check, `query_buildings`
> `max_retries` keyword) landed in commit `4c27695` on
> `feature/audit-pass7-fixes` together with the new OrtsnetzLayout +
> Manhattan PEN router + radial-trunk PEN topology in `60a9cf5`. The
> bullets below capture both the **still-unmerged release-train risk**
> and a new Pass-8 wave of behavioural / docs gaps. Only the
> CHANGELOG is edited in this pass; no program code is touched.

- **`feature/audit-pass7-fixes` carries two commits plus a large
  unstaged delta — release cut blocked.** The branch holds the
  seventh-pass code fixes (`4c27695`) and the AP1-feature commit
  (`60a9cf5` — `OrtsnetzLayout`, `manhattan_routing`, `pen_topology`),
  *plus* twenty-six modified files in the working tree (audit-label
  cleanup in inline comments, AP1-reference removal in module
  docstrings, the new `escape_radius_m` keyword on
  `route_manhattan` and `OrtsnetzLayout.add_pen_cable`,
  CITATION.cff edits). Three CHANGELOG `Added`-blocks dated
  2026-05-18 (escape-radius safety valve, OrtsnetzLayout GPS layer,
  Manhattan router + Ortsnetz builder + radial-trunk topology) are
  also unstaged. The 0.7.0 / 0.6.1 release cannot be tagged until
  the working tree is split into a `chore(audit-cleanup)` commit
  (audit-label / AP1-reference scrub) and a `feat(routing)` commit
  (escape-radius valve + diagnostic error). Eighth-pass forcing
  function: open the release PR before the next audit pass or the
  backlog ledger keeps growing past the implementation.
- **Audit-pass labels inside docstrings and inline comments are
  being scrubbed without a Changelog `Internal` block.** The
  working-tree diff against `60a9cf5` strips the parenthetical
  *„(fifth 2026-05-13 audit pass)"* / *„(sixth 2026-05-14 audit
  pass)"* / *„(seventh 2026-05-18 audit pass)"* breadcrumbs from
  `src/groundfield/coupling/sommerfeld_inductance.py`,
  `diagnostics.py`, `generators/electrode_specs.py`,
  `generators/ortsnetz_builder.py`, `generators/tn_network.py`,
  `geo/osm.py`, `geo/placement.py`, `io/csv.py`,
  `io/groundinsight.py`, `postprocess/vector_fitting.py`,
  `solver/engine.py` and `world.py`. The intent is reasonable (the
  audit pass numbers are noise once the fix is shipped) but the
  change is invisible to a CHANGELOG reader. Add an
  `Internal (audit-label scrub — 2026-05-25)` block before the
  cleanup commit lands.
- **AP1-reference cleanup in module docstrings is undocumented.**
  Same diff removes the *„AP1"* project name from the top-level
  docstrings of `generators/manhattan_routing.py` and
  `generators/ortsnetz_builder.py` (`AP1 PEN cable layout` →
  `PEN cable layout`, `AP1 LV cable router` → `LV cable router`,
  `AP1 reference worlds` → `reference worlds`). Project-neutral
  docstrings are correct (`docs_review_report.md` from 2026-05-18
  explicitly recommended this) but the change is not visible in the
  Changelog. Add a `Docs (project-neutral docstrings — 2026-05-25)`
  bullet referencing the 2026-05-18 docs review.
- **`route_manhattan(..., escape_radius_m=...)` has no negative
  test for `escape_radius_m = -1.0`.** The implementation in
  `src/groundfield/generators/manhattan_routing.py` validates
  ``if escape_radius_m < 0.0:`` and raises `ValueError`, but the
  three new regression tests in `tests/test_ortsnetz_builder.py`
  (running total 31) only cover positive defaults, `0.0` strict
  mode and `6.0` rescue mode. Add a parametrised test on
  `[-1.0, -0.001]` so the contract stays explicit.
- **`OrtsnetzLayout.from_osm` accepts `kvs_lat_lons` without a
  shape contract.** The classmethod (lines 311–432 of
  `src/groundfield/generators/ortsnetz_builder.py`) lets the user
  pass either `kvs_lat_lons` (list of `(lat, lon)` tuples) or
  `kvs_xys` (list of `(x, y)` tuples), guarded by an XOR validation
  on whether both are non-`None`. There is **no** validation that
  the user-supplied `kvs_names` list (if given) has the same
  length as `kvs_lat_lons` / `kvs_xys`, nor that each tuple has
  exactly two elements. A typo like `kvs_lat_lons=[(51.0,)]`
  reaches `projector.to_xy_m(*tup)` and raises a bare
  `TypeError`. Add a `_validate_kvs_arguments(kvs_lat_lons,
  kvs_xys, kvs_names)` helper that produces a self-describing
  error at the API boundary.
- **`evaluate_spec`'s `EvaluateSpecError` is in `docs/api/io.md`
  but not in `docs/api/index.md`.** Pass-7 added the *„Errors"*
  subsection to `docs/api/io.md` (commit `4c27695`); the
  `api/index.md` overview prose still enumerates only the
  read/write helpers and lacks a one-line pointer to the new
  typed exception. Add it to the IO row so users browsing the
  Overview see the typed-error contract before drilling into
  the page.
- **`groundfield.diagnostics` exposes `MIN_THINWIRE_RATIO`,
  `SOFT_LIMIT`, `HARD_LIMIT` as public constants but they are not
  listed in `docs/api/diagnostics.md`.** The constants were promoted
  from the private leading-underscore aliases in pass 5 (commit
  `72ee1b2`) and the audit-label cleanup currently underway in the
  working tree keeps them but removes the *„fifth 2026-05-13 audit
  pass"* breadcrumb. The mkdocstrings dump in `docs/api/diagnostics.md`
  picks up the function symbols but not the module-level constants
  because the page uses `:::groundfield.diagnostics:` without
  `members:` or `filters:`. Add an explicit `:::groundfield.diagnostics:
  members: [MIN_THINWIRE_RATIO, SOFT_LIMIT, HARD_LIMIT, expected_segments,
  world_statistics, check_segment_resolution]` so the public surface
  matches what `from groundfield.diagnostics import *` exposes.
- **`docs/concepts.md` still has no V1 vs V2 Sunde-shell decision
  table (eighth pass in a row for the V1/V2 distinction; second
  pass that explicitly calls for the *table* form).** Same finding
  as the seventh-pass `Fixed (Backlog)` entry. The two
  implementation variants (`concrete_model="lumped"` vs
  `"distributed"`) and their respective complexity / accuracy
  trade-offs are still only discoverable through ADR-0012.
- **`docs/index.md` 0.6.0 migration callout still missing
  (seventh-pass finding re-flagged).** Same as the
  Docs-backlog entry under the seventh review pass.
- **`docs/quickstart.md` does not show the new 0.6.0 `geo`
  exports (seventh-pass finding re-flagged).**
- **`docs/performance.md` lacks a `geo` cost section
  (seventh-pass finding re-flagged).**
- **`docs/api/geo.md` does not document the offline fallback
  contract (seventh-pass finding re-flagged).**
- **ADR-0003 and ADR-0009 still have no reciprocal „See also"
  link to ADR-0011 / ADR-0012 (seventh-pass finding re-flagged
  — eighth pass).**
- **Notebooks `30..36` not in `mkdocs.yml` nav (seventh-pass
  finding re-flagged with two additional notebooks).** The 0.6.0
  release added `32_osm_footprints.ipynb` and `33_concrete_encasement.ipynb`;
  pass 7 added `34_audit_pass7_fixes.ipynb`; the AP1 feature
  branch added `35_radial_trunk_pen_topology.ipynb` and
  `36_ortsnetz_builder.ipynb`. None of the seven are reachable
  from the rendered docs nav.
- **`graphify-out/manifest.json` is stale.** `GRAPH_REPORT.md`
  reports commit `aa9f0978` (the 0.6.0 release tag) while
  `git log` is at `60a9cf5`; the manifest mtimes are all from
  before commit `4c27695`. Run `graphify update .` before the
  next audit pass so the knowledge-graph is in sync with the
  Pass-7 / OrtsnetzLayout commits.

### Docs (Backlog — eighth 2026-05-25 review pass)

- **`README.md` still does not mention the OrtsnetzLayout +
  Manhattan router + radial-trunk PEN topology (uncommitted
  AP1 features).** The README's „New in 0.6.0" section lists
  OSM-driven footprints and concrete encasement but not the
  three uncommitted modules. Add a short *„Upcoming in 0.7.0
  / 0.6.1"* paragraph (or wait until release-cut) describing
  the imperative builder and the GPS / OSM convenience layer.
- **`docs/examples/` has no end-to-end OrtsnetzLayout walkthrough.**
  The `36_ortsnetz_builder.ipynb` notebook lives in `notebooks/`
  (gitignored) and is rendered nowhere; an `examples/10_ortsnetz_layout.md`
  page that imports the Mulmke OSM extract narrative would close
  the gap for docs-site readers.
- **`docs/adr/` has no ADR for the OrtsnetzLayout builder.** The
  module is a substantive architectural addition (imperative
  single-network builder vs. the stochastic `TnNetworkGenerator`)
  but no ADR records the design rationale. Open ADR-0013 (or
  ADR-0014 if ADR-0013 is reserved for the cross-repo
  `show_versions` convention) — `OrtsnetzLayout: imperative
  single-network builder for TN-Ortsnetze`.
- **`docs/adr/` has no ADR for the Manhattan PEN router.** The
  4-connected A* + AABB obstacle test is a new code path with
  its own performance / correctness contract (cell size,
  clearance, escape radius, max iterations). One paragraph of
  rationale per knob is enough.
- **`docs/api/generators.md` does not enumerate the new public
  symbols** `OrtsnetzLayout`, `PenCable`, `KvsPlacement`,
  `BuildingConnection`, `ObstacleBox`, `PenTopology`,
  `StarKvsTopology`, `RadialTrunkTopology`,
  `route_manhattan`, `segment_intersects_box`,
  `merge_collinear_waypoints`. The 0.6.0 changelog block lists
  them but the `docs/api/generators.md` mkdocstrings dump still
  uses the default `:::groundfield.generators:` and the
  pre-0.6.0 module docstring at the top of the page does not
  reference them.
- **`docs/api/io.md` *„Errors"* subsection does not show the
  `try / except EvaluateSpecError` example.** The seventh-pass
  edit added the subsection but only documents the class
  surface. Add a five-line example so consumers can copy-paste
  the pattern.

### Tests (Backlog — eighth 2026-05-25 review pass)

- **`route_manhattan(..., escape_radius_m=-1.0)` negative-input
  parametrised test.** Tied to the Fixed-backlog entry above.
- **`OrtsnetzLayout.from_osm` shape-contract regression tests.**
  Add `tests/test_ortsnetz_builder.py::test_from_osm_kvs_shape_contract`
  parametrised on the four failure modes: `kvs_lat_lons` with
  mis-sized tuples, `kvs_names` length mismatch, both
  `kvs_lat_lons` and `kvs_xys` set, both `substation_lat_lon`
  and `substation_xy` set.
- **No regression test against `graphify-out` staleness.** A
  one-line `test_repo_hygiene.py::test_graphify_manifest_matches_head`
  helper (read `GRAPH_REPORT.md` *„Built from commit"* line,
  compare against `git rev-parse HEAD`) would surface the
  knowledge-graph drift at PR time. Same Forcing-Function
  pattern that `groundmeas` is rolling out for repo-root junk.
- **JSON round-trip test for `Conductor.lumped_series_resistance_ohm`
  (seventh-pass finding re-flagged).**
- **JSON round-trip test for
  `StripElectrode.concrete_shell_coefficient_ohm_m`
  (seventh-pass finding re-flagged).**
- **`mkdocs build --strict` CI hook (seven passes in a row
  flagged this — eighth pass elevates it again).** With the
  five long-standing API pages now present (commit `4c27695`)
  the `--strict` build would *no longer* fail on the missing
  pages; the remaining gates would catch the missing
  `Geo / OSM` index entry, the unreferenced `0.7.0` notebooks
  and any future broken cross-link. This is the single highest-
  leverage forcing function for the docs backlog.

### Roadmap candidates — eighth 2026-05-25 review pass

- **`gf.OrtsnetzLayout.add_multi_stop_trunk(start, stops)`** —
  the Mulmke walkthrough in `notebooks/36_ortsnetz_builder.ipynb`
  ends with a four-item rough-edges list; the multi-stop trunk
  helper (a single call that routes through N intermediate KVS
  in sequence) is the first item and the only one that affects
  the public API surface. Defer the basemap overlay and the
  coverage report to follow-on PRs.
- **`gf.OrtsnetzLayout.coverage_report()`** — Mulmke rough-edge
  item 3. A read-only diagnostic that lists for each building
  which PEN cable it was connected to, the stub length and any
  building that ended up unconnected. Pairs with the proposed
  `assert_api_pages_exist()` as a self-describing pre-solve
  check.
- **`gf.geo.OsmBuildingPlacement.per_footprint_specs`** —
  Mulmke rough-edge item 4. Allow the user to attach a per-
  building `FoundationElectrodeSpec` instead of the current
  „one spec for all footprints" contract. Default behaviour
  unchanged.
- **`gf.cross_repo` namespace + `docs/cross-repo.md`** — once
  ADR-0013 (Cross-repo `show_versions` convention) lands, mirror
  the proposed `gi.cross_repo` / `gm.cross_repo` modules so the
  three packages have a uniform façade for cross-repo helpers
  (`show_versions`, `assert_api_pages_exist`,
  `audit_apply(report_path)`).
- **`gf.docs.assert_api_pages_exist()`** — same as Pass-7;
  seven passes have flagged the missing-API-pages problem,
  five of which are now closed, so the helper is the right
  forcing function to prevent regression.
