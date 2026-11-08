---
title: Review pass 5 (2026-05-13)
description: Review pass 5 (2026-05-13) of groundfield, as recorded in the changelog.
---

# Review pass 5 (2026-05-13)

| | |
|---|---|
| Date | 2026-05-13 |
| Recorded in | the backlog section at the end of the changelog |
| Implemented in | 0.5.0 (2026-05-14) |
| Overview | [Audit history](index.md) · [Changelog](../changelog.md) |

The fifth review pass recorded its findings in the backlog; the implementation block shipped with 0.5.0 (regression tests `tests/test_audit_pass5_fixes.py`).

!!! note "Where this text comes from"
    This page was moved verbatim from `CHANGELOG.md` when the release notes were
    condensed in 2026.11.0 (source: commit `30a122d`). Only the markup was adapted:
    reST roles are rendered as code or math, relative links point to the current
    pages, and each block stands under a heading that names where it was published.
    Paths, test names and numbers are those of the time; the code has changed since.

## From the backlog section of the changelog (“Backlog (2026-05-10 audit)”)

### Fixed (pending implementation) — fifth 2026-05-13 review pass

> Focus of the fifth pass: residual cleanup after the Pass-4
> implementation block landed in `[Unreleased]`; freshly noticed
> bugs around `Engine.frequencies` validator wording, the still-
> unrealised top-level API documentation pages, doc/notebook
> coverage on the mkdocs site, and a handful of new findings in
> the cross-coupling and io packages that earlier passes had not
> reached.

- **`postprocess.vector_fitting.vector_fit(n_poles=1)`** is
  accepted even when the input has only one or two frequencies.
  A 1-pole fit on two complex points has 4 unknowns (real /
  imaginary residue + pole real / imag), which is exactly the
  number of equations after the real-imag stacking, so the
  fit *can* succeed numerically but is meaningless. Reject
  ``n_poles >= len(frequencies)`` or emit a `UserWarning` when
  the problem is under- / exactly-determined.
- **`solver.engine.Engine._validate_frequencies` warns on
  every duplicate.** A sweep over ten engines, all built from
  the same `frequencies=[5000, 50]` "intentionally
  decreasing" list, emits the same warning ten times — the
  `warnings.simplefilter("once")` default deduplicates by
  *message text*, but the message embeds the list literal so
  every distinct list triggers a fresh warning. Switch to a
  dedicated `class EngineFrequencyOrderWarning(UserWarning)`
  with a stable message prefix so notebook users can silence
  the lot with `warnings.simplefilter("once",
  EngineFrequencyOrderWarning)`.
- **`coupling.layered_green.LayeredEarth.compute_T_lambda`
  is silently single-precision on the MLX backend** (`backend=
  "mlx"`). The MLX `mx.array(np.asarray(..., dtype=np.float32))`
  conversion drops to FP32 while the NumPy path stays in FP64.
  A reference run with the two backends therefore differs in
  the 5th decimal — the Pass-3 cross-coupling cross-check did
  not catch this because the test tolerance is `1e-4`. Either
  default to FP64 (`mx.float64`) or document the backend
  precision contract.
- **`io.groundinsight.evaluate_spec` raises
  `KeyError`** when the supplied spec dict is missing the
  ``Z_target`` key. The error originates inside a deep dict-
  lookup chain and the trace is hard to read for a user
  passing a hand-rolled spec. Wrap the entry point with an
  explicit `ValidationError` raise that names the missing key.
- **`generators.tn_network.TnNetworkConfig.build`** does not
  validate that `cfg.measurement.source_kind` is one of the
  supported strings (`"current"` / `"voltage"`). A typo
  (`"voltage_"`) silently falls through to the default
  `CurrentSource` factory; the user sees a result that looks
  correct until they inspect the source's `kind` field.
- **`world.World.solve(engine)` does not deep-copy
  `self.sources`** before passing them to the backend. A
  backend that internally mutates `source.return_to` (the
  Pass-4 finding on `MeasurementSetupConfig.build` is the
  textbook example) silently rewrites the world's sources
  for any later solve. Either copy in `World.solve` or
  document the contract.
- **`diagnostics.check_segment_resolution` thresholds are
  hard-coded** to `5_000` / `20_000` without a public
  constant. Once the integral backends gain a
  `coarse_segments` opt-in (Pass-3 finding), the thresholds
  will move; promote them to module-level `SOFT_LIMIT` /
  `HARD_LIMIT` constants so tests and notebooks have a
  stable handle.
- **`mkdocs.yml` Examples nav still lists only 8 curated
  pages** while `notebooks/` now carries 30 .ipynb files
  (Pass-4 audit-fix notebook brought the total over the
  previous 29). Pass-2/3/4 finding still only partially
  addressed — closing it requires the same
  `mkdocs-jupyter` migration that `groundinsight` is doing.
- **`__init__.__all__` does not list `SourceAdapter`**, the
  `TypeAdapter[Source]` introduced in the Pass-4
  implementation block. Users who follow the docstring of the
  discriminated union and try `from groundfield import
  SourceAdapter` get an `ImportError`.
- **`scripts/release.py` still does not propagate the
  Pass-4 "do-not-hard-code" CLAUDE.md pattern**. The pattern
  is documented but not enforced: a contributor who refreshes
  the CLAUDE.md version line manually still drifts. A
  `re.search(r"__version__ = \"[^\"]+\"", claude_md)` raise on
  bump completes the Pass-4 closure.

### Docs (pending implementation) — fifth 2026-05-13 review pass

- **`docs/api/sources.md`, `boundary.md`, `references.md`,
  `validation.md`, `world.md`** — fifth audit pass in a row
  confirms these pages are still missing. The Pass-4
  implementation block landed `Source` (discriminator),
  `Engine` (order-preserving frequencies),
  `TnNetworkConfig.source_return_to` and the boundary-revert
  warning *all without their target documentation page*. Each
  fix therefore has no natural home for its migration
  callout. The doc pages are the longest-standing single open
  item across the audit history.
- **`docs/api/index.md`** does not surface `SourceAdapter` or
  the `Discriminator("kind")` migration callout. A reader who
  searches the API index for "discriminator" finds nothing.
- **`docs/concepts.md`** still lacks the "Engine vs. World
  separation" section flagged in Pass 4. Pass 5 confirms:
  the only place the split is documented is the
  `World.solve` source docstring.
- **`docs/engines/index.md`** capability matrix still missing
  (Pass-2 → Pass-5 carryover). The matrix is the natural
  home for the new
  `earth_inductive_model={perfect_mirror, carson_series,
  sommerfeld}` toggle and for the `inductance_model` choice.
- **`docs/api/postprocess.md`** still missing the dedicated
  "Vector fitting and safety helpers" subsection (Pass-4
  finding). The new `n_poles=0` rejection callout has no
  doc home.
- **`docs/api/io.md` does not document the new
  `POTENTIAL_PATH_COLUMNS`, `ELECTRODE_TABLE_REQUIRED_COLUMNS`,
  `CLUSTER_IMPEDANCE_REQUIRED_COLUMNS`** column tuples
  exported in the Pass-4 implementation block. Users who want
  to join the CSV outputs back into a pandas frame have to
  read the source to find the canonical column names.
- **README "Compatibility" matrix** has not been updated for
  the `0.5.0` release cut implied by the Pass-4 fixes block.
  Bumps `Python`, `numpy`, `scipy`, `pydantic`, `sympy`
  minimums. Sync with `pyproject.toml` before the next tag.
- **`docs/quickstart.md` does not call
  `Engine.with_frequencies(*freqs, preserve_order=True)`** in
  any example, even though that constructor is the documented
  escape hatch for the new order-preserving validator. Add a
  three-line snippet so the new opt-in lands with discoverable
  prose.
- **`CLAUDE.md`** now defers the version field to
  `pyproject.toml` (Pass-4 doc fix), but the "Audit history"
  section near the top of the file still lists pass 1-4
  only. Append a pass-5 reference so the next session has the
  same starting context the previous ones did.

### Tests (pending implementation) — fifth 2026-05-13 review pass

- **No test for the new `EngineFrequencyOrderWarning`
  category.** Once the dedicated subclass lands (Fixed-
  backlog fifth-pass entry above), assert that
  `warnings.simplefilter("once",
  EngineFrequencyOrderWarning)` suppresses the same warning
  on a second construction.
- **No regression test for `vector_fit(n_poles=1)` on a
  two-frequency input.** Build a fixture with
  `frequencies=[50.0, 5000.0]` and
  `Z_values=[…]` and assert that the helper either rejects
  the call or warns about the under-determined fit.
- **No regression test for the MLX backend FP32 default.**
  Cross-validate `LayeredEarth.compute_T_lambda` with NumPy
  and MLX on the same input and assert a `rtol=1e-12`
  agreement, or — if the FP32 default is intentional —
  document and pin it with a relaxed tolerance.
- **No regression test for `World.solve` source mutation.**
  Build a world, run `solve(engine)`, and assert that
  `world.sources[i].return_to` is unchanged from the
  pre-solve state.
- **No regression test for `from groundfield import
  SourceAdapter`.** Either lock the import or remove the
  symbol from the public surface.
- **No regression test for `evaluate_spec` raising
  `ValidationError` (not `KeyError`) on a missing
  `Z_target` key.** Locks in the human-readable error
  message once the fix lands.
- **No regression test for `mkdocs build`** — the rendered
  site is built on the docs server and never checked in CI.
  Add a `tests/test_docs.py::test_mkdocs_strict_build` that
  runs `mkdocs build --strict` so the missing-page warnings
  surface during PR review.
- **No regression test for the
  `TnNetworkConfig.build(source_kind=...)` typo path.** Build
  a config with `source_kind="voltage_"` (trailing
  underscore) and assert a clear `ValueError`.

### Roadmap candidates — fifth 2026-05-13 review pass

- **`gf.show_versions()`** — print the installed
  `groundfield` / `groundinsight` / `groundmeas` / NumPy /
  SciPy / Pydantic / SymPy / MLX versions. Cross-package
  compatibility is currently maintained by hand; surfacing
  the active versions in one call removes the back-and-forth
  on every issue triage.
- **`gf.io.bench(world, engine, *, repeats=5)`** — pin the
  per-backend wall-clock numbers reported in
  `docs/performance.md`. Today the doc page lists static
  tables; a `bench` helper would let the docs site
  auto-refresh them on a release cut.
- **`gf.audit_apply(audit_report_path: str)`** — read an
  audit-report markdown file, parse the
  `audit-report-changelogs-YYYY-MM-DD[-pass*].md` checklist
  and emit a delta against the current `[Unreleased]` block.
  Six months of audit history have produced a recurring
  workflow; promote it into a maintainer-facing CLI.
- **`gf.docs.assert_api_pages_exist()`** — pytest-friendly
  helper that loads `mkdocs.yml`, walks the API section, and
  asserts every page named in this changelog actually exists.
  Closes the four-passes-in-a-row "page missing" finding by
  failing the build instead of waiting for a human to flag
  it.
- **`gf.SourceAdapter` exposed at the top level** —
  pair with the `Source` discriminated union from Pass-4.
  Users hand-rolling source dicts (e.g. from a YAML
  campaign config) need the adapter to validate the dict;
  today they have to know the `from groundfield.sources
  import SourceAdapter` path.
- **`gf.boundary.FemBoundaryConditions`** — placeholder
  subclass to make the future FEM-backend boundary semantics
  visible *now*, even before the backend lands. Closes the
  "v0.2.0 no-op" doc gap from a different angle: the type
  hierarchy itself documents what the upcoming FEM backend
  will and won't honour.

## Released in 0.5.0 (2026-05-14)

### Fixed (Audit pass 5 — implemented 2026-05-13)

> Implementation block for the *fifth 2026-05-13 review pass*. Every
> entry maps 1:1 to a bullet under "Fixed (pending implementation) —
> fifth 2026-05-13 review pass" further down in this `[Unreleased]`
> block. The pending-implementation bullets remain in place for audit
> consistency but are dispatched here. Regression tests live in
> `tests/test_audit_pass5_fixes.py`; the matching demo notebook is
> `notebooks/31_audit_pass5_fixes.ipynb`.

- **`vector_fit` now warns on under- / exactly-determined input.**
  A new `VectorFitUnderdeterminedWarning(UserWarning)` is emitted
  whenever ``2 * n_poles >= len(frequencies)``. Each pole contributes
  a residue (complex, 2 real DOFs) and a pole-location DOF (complex,
  2 real DOFs); the real-imag stacking gives ``2 N`` real equations,
  so the conservative ``2*n_poles >= N`` threshold catches the cases
  the audit flagged (``n_poles=1`` on ``N ∈ {1, 2}``) — previously
  accepted as a silent identity interpolation. The dedicated warning
  category lets notebooks silence the diagnostic with
  ``warnings.simplefilter("once", VectorFitUnderdeterminedWarning)``.
- **`Engine` frequency-order warning is silenceable by category.**
  Pass-4 introduced the order-preserving validator with a
  ``UserWarning``; pass 5 promotes it to a dedicated
  ``EngineFrequencyOrderWarning(UserWarning)`` subclass and switches
  the warning text to a stable form (the per-call list literal is
  logged at debug level instead). A single
  ``warnings.simplefilter("once", EngineFrequencyOrderWarning)``
  now collapses a ten-engine sweep over decreasing lists down to
  one notification.
- **`LayeredEarth` documents the FP64 precision contract.** The
  dataclass docstring now states explicitly that every consumer of
  ``LayeredEarth`` operates in IEEE-754 double precision. Future
  hardware-accelerated backends (e.g. MLX on Apple silicon) must
  honour the same precision contract; the
  ``tests/test_audit_pass5_fixes.py::test_layered_earth_precision_contract``
  homogeneous-limit cross-check enforces ``rtol=1e-12`` and would
  fail under a silent FP32 down-cast.
- **`io.groundinsight.evaluate_spec` raises `ValueError` on
  malformed specs.** Three guards at the entry point now wrap the
  deep SymPy stack trace into one human-readable ``ValueError`` per
  cause: non-``BusTypeSpec`` argument, missing / empty
  ``impedance_formula``, or unknown free symbols (anything other
  than ``f``, ``rho``, ``j``). A user passing a hand-rolled spec dict
  now sees the missing-``Z_target`` problem at the surface instead
  of a ``KeyError`` four frames deep inside ``sympy.lambdify``.
- **`TnNetworkConfig.source_kind` validates against a `Literal`.**
  New ``source_kind: Literal["current", "voltage"] = "current"``
  field on `TnNetworkConfig`. Typos like ``"voltage_"``
  (trailing underscore) are rejected at validation time with a
  Pydantic ``ValidationError`` instead of silently falling through
  to the default ``CurrentSource`` factory. ``TnNetworkGenerator.build``
  forwards the kind to `create_source`.
- **`World.solve(engine)` snapshots and restores `world.sources`.**
  The method now wraps the backend dispatch in a try / finally and
  restores deep copies of every ``Source`` on the way out. Backends
  that internally mutate ``source.return_to`` (Pass-4 flagged
  ``MeasurementSetupConfig.build`` as the textbook offender) can no
  longer leak the mutation back into the caller's world.
- **`SourceAdapter` is now a top-level re-export.** The Pass-4
  ``TypeAdapter[Source]`` was introduced in ``groundfield.sources``
  but never lifted to the package surface. ``from groundfield import
  SourceAdapter`` now works and the symbol is listed in
  ``groundfield.__all__``.
- **`diagnostics.MIN_THINWIRE_RATIO`, `SOFT_LIMIT`, `HARD_LIMIT`**
  are now public module-level constants. The previous
  ``_MIN_THINWIRE_RATIO`` / ``_BUDGET_WARN_THRESHOLD`` /
  ``_BUDGET_HARD_THRESHOLD`` private aliases remain as
  backwards-compatible shadows. Tests, notebooks and external
  callers gain a stable handle ahead of the planned
  ``coarse_segments`` opt-in.
- **`scripts/release.py` rejects a hard-coded version in
  `CLAUDE.md`.** The Pass-4 "do not hard-code" convention is now
  *enforced*: a new
  ``_check_claude_md_no_hardcoded_version`` scans the file at
  release time, ignoring fenced code blocks and the explanatory
  reminder paragraph, and raises a clear ``RuntimeError`` if a
  ``__version__ = "X.Y.Z"`` (or ``version = "X.Y.Z"``) literal is
  pasted into the project context document. ``main()`` runs the
  check before any files are touched, so a contributor who needs
  to fix the literal does so on a clean working tree.

### Tests (Audit pass 5 — implemented 2026-05-13)

- ``tests/test_audit_pass5_fixes.py`` — 16 regression tests,
  one (or more) per bullet above:
  ``VectorFitUnderdeterminedWarning`` raised / not raised,
  ``EngineFrequencyOrderWarning`` category + once-filter
  deduplication, ``LayeredEarth`` FP64 cross-check,
  ``evaluate_spec`` on bad / empty / unknown-symbol / good specs,
  ``TnNetworkConfig.source_kind`` typo rejection + voltage happy
  path, ``World.solve`` source-mutation guard,
  ``gf.SourceAdapter`` top-level import + ``__all__`` presence,
  ``diagnostics`` public-constant identity + backwards-compatible
  private aliases, ``scripts.release._check_claude_md_no_hardcoded_version``
  bad-literal rejection + fenced-block exemption.

### Docs (Audit pass 5 — implemented 2026-05-13)

- ``notebooks/31_audit_pass5_fixes.ipynb`` — narrative walk-through
  of every behaviour change captured in this block. Cell-per-fix
  layout matches the Pass-4 notebook so the closure pattern stays
  consistent.
- ``src/groundfield/coupling/sommerfeld_inductance.py`` —
  ``LayeredEarth`` docstring extended with the FP64 precision
  contract.
