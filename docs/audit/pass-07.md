---
title: Review pass 7 (2026-05-18)
description: Review pass 7 (2026-05-18) of groundfield, as recorded in the changelog.
---

# Review pass 7 (2026-05-18)

| | |
|---|---|
| Date | 2026-05-18 |
| Reviewed | the state three days after 0.6.0 |
| Recorded in | the release notes of 0.7.0 |
| Implemented in | 0.7.0 (2026-05-26) |
| Overview | [Audit history](index.md) · [Changelog](../changelog.md) |

The seventh review pass was a changelog-only pass; its code fixes shipped with 0.7.0 (regression tests `tests/test_audit_pass7_fixes.py`). Both blocks were published in the 0.7.0 release notes.

!!! note "Where this text comes from"
    This page was moved verbatim from `CHANGELOG.md` when the release notes were
    condensed in 2026.11.0 (source: commit `30a122d`). Only the markup was adapted:
    reST roles are rendered as code or math, relative links point to the current
    pages, and each block stands under a heading that names where it was published.
    Paths, test names and numbers are those of the time; the code has changed since.

## Findings, published in the 0.7.0 release notes

### Fixed (Backlog — seventh 2026-05-18 review pass)

> The seventh audit pass was run on 2026-05-18, three days after the
> `0.6.0` release (`geo` subpackage + concrete-encasement model). The
> entries below capture new findings that emerged either as a
> side-effect of the 0.6.0 surface (`World.concrete_shell_corrections`
> registry, `OsmBuildingPlacement.footprint_at`, `geo.osm` retry
> contract) or that have **still not been addressed** through six
> previous audit passes (missing API pages, ADR cross-references).
> Only the CHANGELOG is edited in this pass; no program code is
> touched.

- **`docs/api/sources.md`, `boundary.md`, `references.md`,
  `validation.md`, `world.md` still missing — seventh pass in a row.**
  The `0.6.0` release added `docs/api/geo.md` for the new OSM
  subpackage but did **not** address the five long-standing missing
  pages flagged in pass 1 through pass 6. `groundfield.world.World`,
  `groundfield.solver.sources`, `groundfield.solver.boundary` and
  `groundfield.validation` are all part of the public ``__all__`` of
  `src/groundfield/__init__.py` (confirmed on 2026-05-18) and the
  user has no rendered API documentation for any of them. The
  six-pass-in-a-row finding is therefore re-elevated; without a
  forcing CI gate it will reappear in the eighth pass.
- **`docs/api/index.md` does not list the new `Geo / OSM` page.** The
  rendered API-reference index enumerates Soil, Geometry, Conductors,
  Solver, Coupling, Postprocess, Diagnostics, IO and Generators but
  is missing the `Geo / OSM` entry that the 0.6.0 `mkdocs.yml` nav
  added (``api/geo.md``). Drift between API overview prose and the
  rendered nav appeared on the same release that landed the
  subpackage — straightforward bullet-list addition.
- **`World.concrete_shell_corrections` registry has no documented
  cleanup contract.** Re-using a single `World` across
  multiple `TnNetworkGenerator.build` calls (the canonical
  parameter-sweep pattern) accumulates entries from previous builds
  in the same dict; the `0.6.0` release records the registry as a
  public ``dict[str, float]`` but offers neither a
  ``world.reset_concrete_corrections()`` helper nor a
  per-build invalidation hook on the generator side. AP1 Monte-Carlo
  studies that flip ``concrete_rho_ohm_m`` per realisation will
  carry stale shell resistances from earlier samples into later
  ones unless the user knows to clear the dict by hand.
- **`OsmBuildingPlacement.footprint_at(i)` lacks defensive bounds
  checking.** The current implementation indexes
  ``self.footprints[i]`` directly; for ``i`` out of range CPython
  raises a bare ``IndexError`` whose message does not name the
  placement, the requested index, or the available range. Wrap with
  an explicit ``raise IndexError(f"OsmBuildingPlacement: requested
  index {i} but only {n} footprints are available; check
  placement.n_buildings.")`` so the AP1 generator pipeline produces
  a self-describing failure instead of an opaque traceback at
  ``TnNetworkGenerator.build`` time.
- **`geo.osm.query_buildings` retry count is hardcoded as ``1``.**
  The docstring promises "one retry on ``429`` / ``504`` with
  exponential backoff" but no parameter is exposed for users
  working over flaky cellular / VPN links. Promote the retry count
  to a ``max_retries: int = 1`` keyword on
  `query_buildings` / `query_and_project`; the default
  preserves current behaviour.
- **`groundfield.io.groundinsight.evaluate_spec` ValueError class is
  not re-exported.** Pass-5 widened `evaluate_spec` to raise
  ``ValueError`` on non-canonical free symbols (and the Pass-5
  Fixed block lists this as a public-surface improvement), but
  downstream code that wants to catch the new exception has no
  named class — it has to compare against ``str(exc)`` substrings
  or fall back to the broad ``ValueError`` parent. Define an
  ``EvaluateSpecError(ValueError)`` subclass in
  `groundfield.io.groundinsight` and add it to the module ``__all__``;
  follow-on `groundinsight` consumers will benefit from the typed
  surface.
- **`Conductor.lumped_series_resistance_ohm` and
  `StripElectrode.concrete_shell_coefficient_ohm_m` are not
  JSON-round-tripped in `tests/test_concrete_encasement.py`.** The
  new test file covers the V2 Sunde-shell closed-form match, the
  V1 registry, the discriminator and the end-to-end OSM smoke,
  but does **not** assert that a `World` with a populated
  `Conductor.lumped_series_resistance_ohm` or
  `StripElectrode.concrete_shell_coefficient_ohm_m` survives
  ``World.model_dump_json()`` → ``World.model_validate_json(...)``.
  Add a single test case so the persistence contract is locked in
  before users start sharing concrete-encasement-bearing world
  files across runs.
- **ADR cross-references are one-directional.** `ADR-0011`
  (OSM building footprints) references `ADR-0009` (World
  generators) and `ADR-0012` (Concrete encasement) references
  `ADR-0003` (Distributed conductor) — but the *target* ADRs
  carry no reciprocal "See also" link. A maintainer reading
  ADR-0003 today sees no breadcrumb to the V1 lumped path that
  reuses its framework; same for ADR-0009 → ADR-0011.
  Bidirectional ADR linking is a one-line edit per file and
  prevents future architectural drift.
- **`docs/concepts.md` is silent on the lumped vs. distributed
  Sunde-shell trade-off.** The 0.6.0 release added two
  implementation variants (``concrete_model="lumped"`` vs.
  ``"distributed"``) with materially different physics (V1
  records the total shell resistance and injects a series
  resistance on the PEN drop; V2 augments the post-kernel
  diagonal per segment). The user-facing `concepts.md` page
  describes neither variant nor the regime where each is
  preferred. AP1 readers reach the ADR-0012 file as a fallback,
  but the canonical "Concepts" page should carry a short
  decision-guide table.
- **`README.md` does not mention `0.6.0` features.** The README
  Roadmap and Features sections were not updated alongside the
  0.6.0 release — neither the `geo` extra, the
  `BuildingFootprint` / `OsmBuildingPlacement` exports nor the
  concrete-encasement model appear in the project description.
  AP1 reviewers landing on the GitHub page have no signal that
  the OSM and concrete-encasement features exist.
- **Notebooks `30..33` are not wired into the docs nav.** All four
  audit / OSM / concrete-encasement notebooks exist on disk and
  are referenced in the 0.6.0 CHANGELOG "Docs" block, but
  `mkdocs.yml` only loads `docs/examples/*.md`. The
  `mkdocs-jupyter` plugin *is* enabled (``plugins: mkdocs-jupyter``
  block, ``execute: false``), so adding a ``Notebooks:`` nav
  section under Examples is a four-line edit. Without it the
  notebook walkthroughs that the 0.6.0 changelog calls out are
  invisible to docs-site readers.

### Docs (Backlog — seventh 2026-05-18 review pass)

- **`docs/index.md` Migration callout for `0.6.0`.** A short
  "Migrating from 0.5.x" section under the index — listing the
  new optional ``geo`` extra, the additive
  ``concrete_rho_ohm_m`` field on
  `FoundationElectrodeSpec`, and the no-op default — would
  spare users a CHANGELOG dive on upgrade. Mirror the pattern
  that the 0.5 release introduced for the discriminated
  ``Source`` union.
- **`docs/quickstart.md` does not show the new exports.** The
  page still ends at ``gf.create_engine(...).solve(world)``; the
  0.6.0 exports `gf.BuildingFootprint`, `gf.OsmBuildingPlacement`,
  `gf.Projector`, `gf.query_buildings`, `gf.query_and_project`,
  `gf.OverpassError` are not introduced anywhere in the quickstart
  flow. A short "Optional: OSM-driven footprints" appendix would
  close the gap.
- **`docs/performance.md` has no `geo` cost section.** The
  on-disk Overpass cache, the projection cost and the OMBR
  reduction are all new performance-relevant code paths in
  `0.6.0`. AP1 readers running long Monte-Carlo sweeps over
  10–100 buildings need an order-of-magnitude estimate
  (current local: ~ms per footprint, network: ~hundreds of ms
  on cold cache) and an explicit pointer at the cache directory
  to clear between experiments.
- **`docs/api/geo.md` lacks the offline-fallback contract.** The
  0.6.0 page describes the live Overpass path and the cache
  layer but does not state the contract of the *purely offline*
  variant — the path through `BuildingFootprint` + manual
  centroid placement that does not import `requests` /
  `pyproj` / `shapely`. A "Working offline" subsection that
  enumerates the no-dependency code path keeps the AP1 reproducible-
  computation guarantee visible.
- **`docs/adr/0003-distributed-conductor-model.md` and
  `docs/adr/0009-world-generators.md`** do not yet reference
  `ADR-0011` and `ADR-0012`. Same finding as the bidirectional
  ADR cross-reference Fixed-backlog entry above.

### Tests (Backlog — seventh 2026-05-18 review pass)

- **No JSON round-trip test for `Conductor.lumped_series_resistance_ohm`.**
  Tied to the ADR-0012 V1 persistence-contract Fixed-backlog entry above.
- **No JSON round-trip test for
  `StripElectrode.concrete_shell_coefficient_ohm_m`.** Tied to the
  ADR-0012 V2 persistence-contract Fixed-backlog entry above.
- **No regression test for `World.concrete_shell_corrections` cleanup
  contract.** Once a per-build `reset_concrete_corrections()` helper
  (or a `TnNetworkGenerator.build(reset_shell_corrections=True)`
  keyword) lands, pin the contract with a two-build test that
  asserts the second build sees a clean registry.
- **No regression test for `OsmBuildingPlacement.footprint_at(i)`
  out-of-range error message.** Once the bounds check lands, pin
  the structured error string so the message does not regress to
  a bare `IndexError`.
- **No timing test for the `geo.osm` cache.** A trivial fixture
  that mocks the Overpass `_post` boundary, asserts a single POST
  per ``(origin, radius)`` tuple over five `query_buildings`
  invocations and checks the cache directory carries exactly one
  on-disk file is a one-screen test that would lock in the
  ADR-0011 reproducibility guarantee.
- **No `mkdocs build --strict` test.** Sixth-pass entry remains
  open; seventh pass elevates it again because the new `geo.md`
  page and the still-missing `sources.md` / `boundary.md` /
  `references.md` / `validation.md` / `world.md` pages would all
  be flagged by ``--strict`` immediately. Without the CI gate,
  the eighth audit pass will inevitably re-list the same
  five-page gap.

### Roadmap candidates — seventh 2026-05-18 review pass

- **`gf.show_versions()`** — same return-shape convention as the
  proposed `gi.show_versions()` and `gm-cli doctor`. Sixth-pass
  carried this; seventh pass re-emphasises the cross-repo
  contract: ``dict[str, str]`` with keys ``package``, ``python``,
  ``numpy``, ``scipy``, ``sympy``, ``pydantic``, ``polars``,
  ``shapely``, ``pyproj``, ``requests`` (the last three reflect
  the new ``geo`` extra) plus ``_meta = {timestamp, platform}``.
- **`gf.docs.assert_api_pages_exist()`** — walk ``__all__`` and
  assert every public symbol has at least one mkdocstrings
  ``:::`` directive somewhere under ``docs/api/``. Companion to
  the proposed `mkdocs build --strict` CI gate. Six passes have
  flagged the missing-API-pages problem and *seven* passes the
  forcing-function lack; this is now the highest-leverage
  cross-repo roadmap item.
- **`gf.World.reset_concrete_corrections()`** — public helper
  matching the ADR-0012 V1 registry. Default behaviour stays
  additive; the helper is opt-in for parameter sweeps. Tie to
  the Fixed-backlog entry above.
- **`gf.geo.OverpassCache(path, max_age_s=...)`** — promote the
  current hidden ``$XDG_CACHE_HOME/groundfield/osm/`` directory
  to a configurable cache object. Users running CI builds want
  per-job cache control; AP1 Monte-Carlo runs want to pin the
  cache for reproducibility.
- **ADR-0013 — Cross-repo `show_versions` convention.** Bundle
  the three-repo `show_versions` proposal into a single ADR
  that lives in `groundfield` and is linked from
  `groundinsight` / `groundmeas`. Five passes have proposed
  ``show_versions``; the ADR is the forcing function that pins
  the return shape before any of the three packages ships its
  own implementation.

## Released in 0.7.0 (2026-05-26)

### Added (Audit pass 7 — implemented 2026-05-18)

- **`groundfield.io.groundinsight.EvaluateSpecError`** — new typed
  exception (subclass of `ValueError`) raised by
  `evaluate_spec` on malformed spec input. The class is
  re-exported at the top level as `groundfield.EvaluateSpecError`
  and listed in `groundfield.__all__`. Legacy
  ``except ValueError`` blocks keep working unchanged because the
  new class inherits from `ValueError`; downstream
  ``groundinsight`` consumers can now catch the typed exception
  without grepping ``str(exc)`` (closes the *seventh 2026-05-18
  audit pass* "no named class for evaluate_spec failures" bullet).
- **`groundfield.World.reset_concrete_corrections()`** — explicit
  cleanup helper for the ADR-0012 V1 concrete-shell registry
  (`world.py`). Re-using a single `World` across multiple
  `TnNetworkGenerator.build` calls — the canonical AP1
  Monte-Carlo pattern over ``concrete_rho_ohm_m`` — previously
  accumulated stale entries in the public
  `World.concrete_shell_corrections` dict. The new method
  clears the registry and returns a shallow copy of the dropped
  entries, so callers can inspect or log them. Calling on a fresh
  world is an idempotent no-op. Closes the *seventh 2026-05-18
  audit pass* "registry has no documented cleanup contract" bullet.
- **`OsmBuildingPlacement.footprint_at(i, strict=...)`** — opt-in
  defensive bounds check (`geo/placement.py`). Default
  ``strict=False`` preserves the v0.6.0 "out-of-range → ``None``"
  contract that `TnNetworkGenerator.build` relies on;
  ``strict=True`` raises a self-describing `IndexError`
  that names the placement, the requested index, the filtered
  footprint count (after ``min_area_m2``) and the raw list length.
  The placement also exposes a new
  `OsmBuildingPlacement.n_footprints` property identical to
  ``len(placement)`` so consumers can pre-flight ``i`` without
  invoking the dunder. Closes the *seventh 2026-05-18 audit pass*
  "footprint_at lacks defensive bounds checking" bullet.
- **`groundfield.geo.osm.query_buildings(..., max_retries=N)` and
  `query_and_project(..., max_retries=N)`** — promote the
  previously hard-coded ``1`` retry to a keyword
  (`geo/osm.py`). Defaults to ``1``, matching the historic v0.6.0
  contract; users on flaky cellular / VPN links can raise it,
  CI pipelines that prefer fail-fast semantics can set ``0``.
  ``max_retries < 0`` is rejected at the API boundary with a
  clear `ValueError`. The module docstring and the
  `_post_overpass` internal docstring are updated to match.
  Closes the *seventh 2026-05-18 audit pass* "retry count
  hardcoded as 1" bullet.

### Tests (Audit pass 7 — implemented 2026-05-18)

- **`tests/test_audit_pass7_fixes.py`** — 14 regression tests
  mapped 1:1 to the user-visible bullets above:
  `EvaluateSpecError` is a `ValueError` subclass,
  is raised on missing / empty / unknown-symbol formulas, and is
  reachable from the top level (4 tests);
  `World.reset_concrete_corrections` clears the registry,
  returns a defensive copy of the previous state, and is
  idempotent on a fresh world (3 tests);
  `OsmBuildingPlacement.footprint_at` keeps the historic
  ``None`` default, raises a structured `IndexError` under
  ``strict=True`` (negative and out-of-range indices), the message
  carries the expected diagnostic tokens, and the new
  `n_footprints` property tracks ``len(placement)`` after
  ``min_area_m2`` filtering (4 tests);
  `query_buildings` accepts and forwards ``max_retries``,
  rejects negative values, defaults to ``1``, and
  `query_and_project` exposes the same keyword on its
  signature (3 tests).

### Notebooks (Audit pass 7 — implemented 2026-05-18)

- **`notebooks/34_audit_pass7_fixes.ipynb`** — narrative walk-through
  of the four behaviour changes, one short section per fix. The
  notebook runs offline (no Overpass call) and uses synthetic
  footprints for the placement demo.

### Docs (Audit pass 7 — implemented 2026-05-18)

- **`docs/api/world.md`, `docs/api/sources.md`, `docs/api/boundary.md`,
  `docs/api/validation.md`, `docs/api/references.md`** — the five
  long-standing missing API-reference pages (flagged from pass 1
  through pass 6) now exist as mkdocstrings stubs that render the
  module-level public surface. Wired into ``mkdocs.yml`` nav under
  *API reference*. Closes the seven-pass-in-a-row backlog bullet.
- **`docs/api/index.md`** — the API-reference index now lists the
  ``Geo / OSM`` page that was added in 0.6.0 but missing from the
  rendered index, alongside the five newly-added pages.
- **`docs/api/io.md`** — adds an "Errors" subsection that documents
  the new `EvaluateSpecError` class and the typed-exception
  contract for downstream ``groundinsight`` consumers.
