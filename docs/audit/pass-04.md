---
title: Review pass 4 (2026-05-12)
description: Review pass 4 (2026-05-12) of groundfield, as recorded in the changelog.
---

# Review pass 4 (2026-05-12)

| | |
|---|---|
| Date | 2026-05-12 |
| Recorded in | the backlog section at the end of the changelog |
| Implemented in | 0.5.0 (2026-05-14) |
| Overview | [Audit history](index.md) · [Changelog](../changelog.md) |

The fourth review pass recorded its findings in the backlog; the implementation shipped with 0.5.0 (regression tests `tests/test_audit_pass4_fixes.py`).

!!! note "Where this text comes from"
    This page was moved verbatim from `CHANGELOG.md` when the release notes were
    condensed in 2026.11.0 (source: commit `30a122d`). Only the markup was adapted:
    reST roles are rendered as code or math, relative links point to the current
    pages, trailing whitespace is removed, and each block stands under a heading
    that names where it was published.
    Paths, test names and numbers are those of the time; the code has changed since.

## From the backlog section of the changelog (“Backlog (2026-05-10 audit)”)

### Fixed (pending implementation) — fourth 2026-05-12 review pass

> Focus of the fourth pass: discriminated-union annotation on the
> `Source` re-export, boundary-condition warning semantics on
> *reverts* to the default, missing top-level re-exports vs. the
> sister project's flat `gi.*` surface, mkdocs notebook nav gap.

- **`groundfield.sources.Source = Union[CurrentSource,
  VoltageSource]`** lacks Pydantic's `Discriminator("kind")`
  annotation. Pydantic falls back to attempting validation against
  each member in declaration order during JSON deserialisation; a
  malformed payload with `kind="voltage"` but missing `magnitude`
  silently falls through to `CurrentSource` and is rejected there
  with a `magnitude` error pointing at the wrong class. Mark the
  union as
  `Annotated[Union[CurrentSource, VoltageSource],
  Discriminator("kind")]` so the error message matches the
  declared `kind`.
- **`World.set_boundary_conditions(...)` warns only on *non-
  default kwargs***. A user who first set
  `set_boundary_conditions(far_field="neumann")` and *then* calls
  `set_boundary_conditions(far_field="dirichlet")` to revert sees
  no warning, even though the latter call silently re-arms the
  v0.2.0 default. Either also warn on every transition that
  flips a previously non-default field back to its default
  (because the previous non-default state was never actually
  consumed) or document the asymmetry.
- **`World.add_source(source)` does not validate `attached_to` or
  `return_to` against the world's electrodes/conductors.** A
  typo `attached_to="g_1"` on a world with electrode `"g1"` is
  accepted; the bug surfaces at solve time as a zero-injection
  result with no diagnostic. Already flagged in pass 3 — pass 4
  confirms the path is still unguarded.
- **`io.groundinsight.evaluate_spec`,
  `io.groundinsight.fit_quality_summary` and
  `coupling.LayeredEarth` are publicly importable but missing
  from `groundfield.__all__`** (`__init__.py`). Users who write
  `gf.evaluate_spec(...)` get an `AttributeError`; consistency
  with the `gi.*` flat surface (which re-exports analogous
  helpers) is broken. Promote all three to top-level re-exports.
- **`solver.engine.Engine.frequencies` is silently sorted on
  validation**, but the public docstring does not mention the
  reorder. Notebooks that pass `frequencies=[5000, 50]` because
  they want the dominant 5 kHz tone first in the result get a
  silently-reordered result array — a real surprise when joining
  against an external DataFrame indexed by the *input* order.
  Either preserve order on construction or document the reorder.
- **`generators.measurement.MeasurementSetupConfig.build` sets
  `source.return_to` to the auxiliary anchor**, but does not
  guard against a `TnNetworkConfig.source` whose `return_to` was
  already explicitly set by the user (e.g. for a multi-source
  AP 1 study with two parallel measurements). The user-supplied
  `return_to` is silently overwritten. Either copy-on-write the
  source spec inside the generator (already on the pass 1
  backlog as a deep-copy hint) or raise a clear error.
- **`postprocess.vector_fitting.vector_fit` accepts
  `n_poles=0`** and returns a constant-only fit, but
  `fit_to_sympy` then produces an expression with no `s`
  dependence — the matching `groundinsight.BusType` is degenerate
  (Z(rho, f) = const, ignoring rho and f). Validate `n_poles >=
  1` at function entry or document the degenerate case
  explicitly.
- **`io.csv.save_potential_path_csv` writes
  `abs_phi = |phi_re + 1j*phi_im|`** under the column
  `abs_phi`, but the corresponding
  `save_cluster_impedances_csv` and `save_electrode_table_csv`
  use the suffix `_abs` (`abs_Z`, not `Z_abs`). Pick one
  convention and apply it across `io.csv` so DataFrames
  produced by the three writers can be joined on a uniform
  column-naming scheme.
- **`mkdocs.yml:90` still references
  `https://polyfill.io/v3/polyfill.min.js?features=es6`** —
  fourth audit pass in a row that flags it. `groundmeas`
  already removed the URL; do the same here and in
  `groundinsight`.
- **`mkdocs.yml` nav lists the eight curated `examples/`
  pages but not the 29 raw notebooks (`notebooks/01..29`).**
  Mirror the same gap reported for `groundinsight`: the
  feature work in `notebooks/22_tier0_speedup.ipynb`,
  `notebooks/23_safety_touch_step.ipynb`,
  `notebooks/29_io_csv_vtk.ipynb` etc. is invisible to the
  docs-site reader unless they go to the GitHub tree.
- **`generators/__init__.py` does not re-export the
  `single_rod_grounding(rng=...)` factory's `rng` argument
  contract** through a short doctest. The factory's RNG
  default is `None` (no jitter), but the only example in the
  docs (`docs/api/generators.md`) shows the deterministic
  call — the stochastic Monte-Carlo path is not advertised
  even though the rest of the generator framework's pass-3
  bug list lists `rng` as a missing parameter.

### Docs (pending implementation) — fourth 2026-05-12 review pass

- **`docs/api/sources.md` still missing** (third audit pass in a
  row). The pass-3 entry remains in the backlog above; pass 4
  confirms the file has not been created. Without the page,
  the discriminated-union fix above has no natural home for
  its migration callout.
- **`docs/api/boundary.md` still missing** (third audit pass in
  a row). Pass 4 confirms the gap; `BoundaryConditions` and
  the v0.2.0 no-op contract live only in source docstrings.
- **`docs/api/references.md`, `docs/api/validation.md`,
  `docs/api/world.md`** are all still missing — pass 4
  confirms three consecutive audits have flagged them.
- **`docs/concepts.md` has no "Engine vs. World separation"
  section** describing why `engine.solve(world)` and
  `world.solve(engine)` are duals, why `Engine` carries
  `frequencies` while `World` carries `boundary`, and how the
  v0.2.0 `BoundaryConditions` no-op interacts with the
  upcoming FEM backend. The split is a deliberate part of the
  architecture but the only place it is documented is the
  `World.solve` source docstring.
- **`docs/api/diagnostics.md` does not document the
  `check_segment_resolution`-budget threshold rationale.** The
  thresholds (5 000 soft, 20 000 hard) appear in the source
  code and in this changelog but the rendered doc page lists
  them as bare numbers. Add a short "Why 5 000?" subsection
  that ties them to the $O(N^2)$ memory / $O(N^3)$ solve
  scaling of the integral backends.
- **`docs/engines/index.md` has no capability matrix.** The
  pass-2 entry already noted the gap; pass 4 confirms it. The
  matrix is the natural home for the
  `inductance_model` / `earth_inductive_model` / distributed-
  conductor support tabulation referenced in the Roadmap.
- **`README` Python-version badge is still
  `3.12 | 3.13`.** Pass 4 confirms the gap from pass 2.
- **`CLAUDE.md` version drift is still `0.1.0`** while the
  package ships `0.4.0`. Pass 4 confirms the gap; the
  `scripts/release.py` hook flagged in pass 2 has not
  landed.
- **No `docs/api/coupling/sommerfeld.md`** dedicated
  page. The Sommerfeld earth-return module (`coupling/
  sommerfeld_inductance.py`) contains six public helpers and
  the Pollaczek-Wait kernel; users reading
  `docs/api/coupling.md` find the heading but not the kernel
  signature.
- **`docs/api/postprocess.md` lacks a "Safety helpers"
  subsection** distinct from "Current sharing" and "Plotting"
  — the EN 50522 lookup helper is the most safety-critical
  function in the package and currently shows up only via
  the autodoc.

### Tests (pending implementation) — fourth 2026-05-12 review pass

- **No test that `Source` JSON round-trip preserves the
  discriminator-vs-non-discriminator union behaviour.** Once
  the `Discriminator("kind")` annotation lands, pin the
  resulting error message for a malformed payload so the bug
  cannot regress silently.
- **No test that `World.set_boundary_conditions(...)` warns on
  the *revert-to-default* path** flagged above. Asserting
  `pytest.warns(UserWarning, match="default")` on the second
  call locks in whatever semantics the team picks.
- **No test that `vector_fit(n_poles=0)` is rejected.** A
  one-line `pytest.raises(ValueError)` regression once the
  guard lands.
- **No test for the `evaluate_spec` / `fit_quality_summary`
  re-export.** Once the helpers are promoted to the top-level
  `__all__`, assert that `gf.evaluate_spec is
  groundfield.io.groundinsight.evaluate_spec` so a future
  silent rename cannot break the public alias.
- **No test for the `Engine.frequencies` sort behaviour.**
  Pass `frequencies=[5000, 50]`, call `engine.solve(world)`,
  and assert either (a) the returned cluster impedance is
  indexed in the input order or (b) a `UserWarning` was
  raised at construction time. Either way the user gets
  determinism.
- **No regression test on `MeasurementSetupConfig.build`
  for a pre-set `source.return_to`.** Build a
  `TnNetworkConfig` whose source already has
  `return_to="explicit_aux"`, run the generator with a
  `MeasurementSetupConfig`, and assert either that the
  user-supplied value is honoured or that the override is
  reported via a clear log entry.

### Roadmap candidates — fourth 2026-05-12 review pass

- **`gf.create_source(world, ..., validate_attached_to=True)`
  switch** — opt-in (default `True` from the next minor release,
  default `False` until then) for the validation contract
  flagged repeatedly in passes 1–4. Gives the existing test
  suite a deprecation window before the validator becomes
  mandatory.
- **`gf.SourceUnion` with `Discriminator("kind")`** — public
  type alias that ships the discriminated annotation alongside
  the legacy `Source = Union[...]`. Lets type checkers in
  downstream notebooks pick up the fix without a major-version
  bump.
- **`generators.tn_network.TnNetworkConfig.measurement_array(
  ...)`** — convenience accessor that returns a per-house
  measurement-setup catalog (Analysis 1 only, Analysis 2 only,
  both) so the AP 1 sweep over `n_efh ∈ {5, 10, 30, 80, 200}`
  can be expressed as a single sweep axis rather than three
  nested ones.
- **`solver.engine.Engine.with_frequencies(freqs,
  preserve_order=True)`** — copy-on-write factory that
  produces a new engine without the silent sort surprise above.
  Closes the order-preservation gap until the validator change
  lands.
- **`scripts/release.py` — sync `CLAUDE.md` version from
  `pyproject.toml`** — same hook every pass-1..4 of the cross-
  repo audit has requested. A six-line `re.sub` in
  `scripts/release.py` ends the drift permanently.
- **`io.csv.save_field_csv(result, path, *, extent, z=0.0,
  n=(120, 120), frequency_index=0)`** — companion to the
  existing VTK exporter that writes a long-format
  `(x, y, frequency_Hz, phi_re, phi_im, abs_phi)` CSV. Closes
  the symmetry gap: today CSV exports cover paths and
  electrode tables, VTK covers fields; the slice export is
  available only as VTK.

## Released in 0.5.0 (2026-05-14)

### Changed

- **`Source = Union[CurrentSource, VoltageSource]` is now a Pydantic
  discriminated union** (`sources.py`). The annotation is
  ``Annotated[Union[...], Discriminator("kind")]``, with a companion
  ``SourceAdapter = TypeAdapter(Source)`` for stand-alone source-dict
  validation. JSON / dict round-trips now report errors against the
  selected sub-class (``CurrentSource`` / ``VoltageSource``) instead
  of dumping the whole union's validator chain — fourth 2026-05-12
  audit pass closure. The user-facing constructor signatures are
  unchanged; only the error messages and the public ``Source``
  annotation differ.
- **`Engine.frequencies` is documented and validated as
  order-preserving** (`solver/engine.py`). A new
  ``field_validator("frequencies")`` rejects empty or non-positive
  inputs and emits a ``UserWarning`` when the list is not strictly
  increasing — but the order is *preserved* in
  ``Engine.frequencies`` and downstream in
  ``FieldResult.frequencies``. The new
  ``Engine.with_frequencies(*freqs, preserve_order=True)``
  constructor is the explicit opt-in for sweeps that intentionally
  iterate non-monotonically (e.g. ``[5000, 50]``); it silences the
  warning and returns a fresh ``Engine`` without mutating the
  receiver. Closes the *fourth 2026-05-12 review pass* "silent sort"
  finding.
- **`TnNetworkConfig.source_return_to` exposes an explicit override
  for the source's return path** (`generators/tn_network.py`). When
  set, it takes precedence over the auxiliary electrode derived
  from ``cfg.measurement``; when both are present and differ,
  ``TnNetworkGenerator.build`` emits a ``UserWarning`` that makes
  the precedence explicit. Previously the measurement setup
  silently overwrote any caller-side intent for ``return_to``.

### Fixed

- **`World.set_boundary_conditions` now warns on revert as well as
  on set** (`world.py`). Setting a non-default value already
  emitted a ``UserWarning`` (v0.2.0); the same call signature now
  *also* warns when a previously-set non-default value is reverted
  back to the default. The previous non-default value was never
  consumed by any backend, so the revert is a silent no-op — the
  new warning makes that visible (fourth 2026-05-12 audit pass).
- **`vector_fit(n_poles=0)` is rejected at the API boundary with a
  detailed `ValueError`** (`postprocess/vector_fitting.py`). A
  zero-pole fit produces an ``s``-free SymPy expression whose
  ``groundinsight.BusType.impedance_formula`` consumer cannot
  recover the frequency dependence; the failure mode was silent
  before. ``fit_to_sympy`` additionally guards against
  programmatically constructed ``VectorFitResult`` objects whose
  resulting expression is ``s``-free, emitting a ``UserWarning``
  with the same diagnostic.
- **`mkdocs.yml` no longer references `polyfill.io`**. The CDN's
  ownership change in early 2024 (malicious payloads injected via
  the original domain) made the URL unsafe to ship in the docs
  build. Modern MathJax 3 / ``tex-mml-chtml`` does not require an
  ES6 polyfill for any browser the docs target. Four audit passes
  in a row flagged this URL — fixed in this release.

### Added

- **Top-level re-exports `evaluate_spec`, `fit_quality_summary` and
  `LayeredEarth`** (`__init__.py`). The three helpers were
  publicly importable from their sub-modules but were missing from
  the top-level surface and from ``__all__``. They are now
  reachable as ``gf.evaluate_spec`` / ``gf.fit_quality_summary`` /
  ``gf.LayeredEarth`` and listed in ``__all__``.
- **Frozen CSV column-name tuples** in `groundfield.io.csv`
  (``POTENTIAL_PATH_COLUMNS``, ``ELECTRODE_TABLE_REQUIRED_COLUMNS``,
  ``CLUSTER_IMPEDANCE_REQUIRED_COLUMNS``). The convention is
  ``<symbol>_re`` / ``<symbol>_im`` / ``abs_<symbol>`` per
  physical quantity (``phi`` for potentials, ``I`` for currents,
  ``Z`` for impedances) — magnitude columns therefore differ per
  writer by design. Locking the tuples makes accidental renames
  test-detectable and documents the join-rename requirement.

### Tests

- ``tests/test_audit_pass4_fixes.py`` — eight regression tests
  mapped 1:1 to the user-visible bullets above: discriminator
  round-trip, discriminator error path, boundary revert warning,
  ``Engine`` order-preserving validator + warning, opt-in
  ``with_frequencies`` silencer, ``TnNetworkConfig.source_return_to``
  precedence with warning, ``vector_fit(n_poles=0)`` rejection,
  CSV column-schema lock, top-level re-export presence, and
  ``mkdocs.yml`` polyfill cleanup.

### Docs

- ``notebooks/30_audit_pass4_fixes.ipynb`` — narrative
  walk-through of all behaviour changes captured in this block.
  The notebook is short and per-fix: every cell illustrates one
  change in isolation so the diff is easy to verify when closing
  the audit.
- ``CLAUDE.md`` — added a "Version (do not hard-code in this
  file)" section that points at ``pyproject.toml`` /
  ``groundfield.__version__`` / ``CITATION.cff`` as the canonical
  sources. Closes the four-passes-in-a-row CLAUDE.md version-drift
  finding from the cross-cutting recommendations.
- ``mkdocs.yml`` — inline comment explaining why
  ``polyfill.io`` was removed.
