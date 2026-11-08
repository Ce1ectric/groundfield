---
title: Review pass 6 (2026-05-14)
description: Review pass 6 (2026-05-14) of groundfield, as recorded in the changelog.
---

# Review pass 6 (2026-05-14)

| | |
|---|---|
| Date | 2026-05-14 |
| Reviewed | 0.5.0 |
| Recorded in | the backlog section at the end of the changelog |
| Implemented in | one fix in 0.6.0 (2026-05-15) |
| Overview | [Audit history](index.md) · [Changelog](../changelog.md) |

The sixth review pass ran against 0.5.0. One finding, the under-determined check of `vector_fit`, was fixed in 0.6.0.

!!! note "Where this text comes from"
    This page was moved verbatim from `CHANGELOG.md` when the release notes were
    condensed in 2026.11.0 (source: commit `30a122d`). Only the markup was adapted:
    reST roles are rendered as code or math, relative links point to the current
    pages, trailing whitespace is removed, and each block stands under a heading
    that names where it was published.
    Paths, test names and numbers are those of the time; the code has changed since.

## From the backlog section of the changelog (“Backlog (2026-05-10 audit)”)

### Fixed (pending implementation) — sixth 2026-05-14 review pass

> Sixth-pass findings against `0.5.0` (released 2026-05-14).
> Focus: secondary effects of the just-shipped 0.5.0 surface,
> notebook / docs reality vs. CHANGELOG claim, ADR-0011
> follow-through, FP-precision contract enforcement, and the
> `World.solve` deep-copy cost on engine-sweep workloads.

- **`World.solve(engine)` deep-copies `self.sources` on every
  call.** Pass-5 added the snapshot/restore pattern to defend
  against backend-side `Source.return_to` mutation
  (`MeasurementSetupConfig.build` was the textbook offender).
  The implementation does
  ``[s.model_copy(deep=True) for s in self.sources]`` on entry
  *unconditionally*. On a 100-frequency sweep that re-uses the
  same `World` across `Engine.with_frequencies(...).solve(...)`
  calls the deep-copy cost dominates for sources that carry
  large `return_to`-electrode meshes. Either gate the snapshot
  behind a `World.solve(engine, snapshot_sources=True)`
  default-on keyword (so power users can opt out), or do a
  shallow snapshot of just the mutable fields
  (`return_to`, `value`) instead of a full `model_copy(deep=True)`.
- **`VectorFitUnderdeterminedWarning` threshold mis-fires on
  conjugate-pair fits.** The current check
  ``if 2 * n_poles >= len(frequencies)`` treats every pole as
  contributing 4 real DOFs (residue real+imag, pole real+imag).
  For a conjugate-pair fit (`complex_conj=True`, default) the
  residues and pole locations come in complex-conjugate pairs,
  halving the actual free parameters. The result is a false
  positive on `n_poles=2, N=2` (warns) where the conjugate-pair
  fit is in fact uniquely determined. Tighten the check to
  ``2 * n_independent_poles >= len(frequencies)`` where
  `n_independent_poles = n_poles // 2 + (n_poles % 2)`
  under `complex_conj=True`.
- **`LayeredEarth.compute_T_lambda` FP-precision contract is
  documented but not enforced.** Pass-5 added the docstring
  note "every consumer of LayeredEarth operates in IEEE-754
  double precision" and a homogeneous-limit cross-check at
  `rtol=1e-12`. What is still missing is the actual *guard*:
  a future MLX-on-Apple backend silently downgrading to FP32
  would pass the homogeneous check (which is a single-layer
  identity, no roundoff to speak of) and fail only on
  five-layer realistic soils. Add an assertion at the top of
  `compute_T_lambda`: `assert out.dtype == np.float64,
  "LayeredEarth contract: FP64 required"` — cheap, catches
  the regression at the source.
- **`Engine` frequency-list cache key.** `Engine.frequencies`
  is now order-preserving with `EngineFrequencyOrderWarning`.
  Pass 5 surfaces but does not solve the equality contract:
  `Engine(frequencies=[50, 100]) == Engine(frequencies=[100, 50])`
  evaluates to `False` (the warning is *per-instance*), so two
  engines that map to the same physics produce two different
  cache keys downstream in `compare_engines` / `convergence_study`
  result tables. Either document the "order matters" contract
  on `Engine.__eq__` *or* override `__eq__` / `__hash__` to
  compare frozenset(frequencies).
- **`io.groundinsight.evaluate_spec` free-symbol allow-list
  is name-only.** The Pass-5 fix accepts every symbol whose
  `name in {"f", "rho"}`, ignoring assumptions. A
  `sp.Symbol("rho", real=False, positive=False)` slips through
  and produces a complex-valued `rho` that crashes deep in
  `lambdify` instead of at the validation step. Compare on
  full symbol identity (`s == f_sym or s == rho_sym`) and
  reject anything else.
- **`Engine.with_frequencies(preserve_order=False)` is the
  silent default.** The docstring shows
  ``preserve_order=True`` but the constructor body checks
  ``if not preserve_order`` and falls through; if the caller
  omits the keyword the engine *does* preserve order but
  the warning is still suppressed by the
  `simplefilter("ignore")` block. Result: a sweep author who
  copies the docstring example sees the same silenced
  behaviour as the explicit `preserve_order=True` user, but
  the silencing is by accident. Either flip the default
  to `False` (loud) and document the explicit silencer, or
  document that `Engine.with_frequencies` is the *opt-in*
  silencer regardless of `preserve_order` value.
- **`SourceAdapter` validates a single source dict but does
  not validate a *list*.** A user-side helper that imports a
  campaign YAML with a `sources: [...]` field has to call
  `SourceAdapter.validate_python` per element. Add a
  `SourcesAdapter = TypeAdapter(list[Source])` companion so
  the bulk path is one call.
- **`World.set_boundary_conditions(**kwargs)` revert-warning
  is emitted regardless of whether the warning was already
  raised for the same value.** Pass-4 added the set-and-revert
  warning, Pass-5 left it as-is. A loop that toggles a
  boundary value across 100 sweeps emits 200 warnings. Apply
  a `simplefilter("once", BoundaryConditionWarning)` strategy
  on first set, OR coalesce the warning by value identity
  inside the method body.
- **`vector_fit` initial pole placement is silent for
  `n_poles == 1`.** `_initial_poles` chooses a single real
  pole at the geometric mean of `omega_min`/`omega_max`. For
  data with a complex-pole structure (e.g. capacitive return
  path), the iteration cannot bend a real pole into a
  conjugate pair within `max_iter=20` and the fit converges
  to a poor RMSE. A `UserWarning` "vector_fit(n_poles=1)
  cannot fit complex resonance peaks; pass n_poles=2 for
  conjugate-pair coverage" at fit time would save a
  Stack-Overflow round-trip.
- **`compare_engines` returns the discrepancy matrix but
  does not call `EngineFrequencyOrderWarning` silencer.**
  A 4×4 cross-engine matrix over decreasing frequencies fires
  16 warnings; the helper should suppress them once internally
  while still surfacing them on the per-engine path.

### Docs (pending implementation) — sixth 2026-05-14 review pass

- **`docs/api/sources.md`, `boundary.md`, `references.md`,
  `validation.md`, `world.md`** — **sixth audit pass in a
  row** these pages are still missing. Pass 5 promised
  `mkdocs build --strict` would catch this in CI; the test
  has not landed and the pages have not been written. This
  is now the longest-standing structural doc gap in the
  three-package family. Sixth-pass acknowledges that one
  of two things has to happen before the next release tag:
  either the five pages land, or the `mkdocs build --strict`
  test lands (the latter is the one-line forcing function).
- **`docs/adr/0011-cross-repo-release-shared.md`** does not
  exist. Pass-5 cross-cutting recommendation #5 ("a single
  shared `scripts/_release_shared.py` would lose three
  release-flow problems in one diff") was elevated to an
  explicit ADR-0011 proposal in the cross-cutting summary;
  the ADR itself was not drafted. As the architectural
  authority on cross-repo conventions sits in `groundfield`
  (eight existing ADRs), ADR-0011 belongs here even though
  the implementation will ultimately live in all three
  repos.
- **`docs/api/index.md` does not document the FP64 contract
  on `LayeredEarth`** that Pass-5 added in code. The
  rendered API index page therefore does not tell the user
  that hardware-accelerated backends are required to keep
  IEEE-754 double precision. Add a one-paragraph admonition
  before the `LayeredEarth` `:::` directive.
- **`docs/api/postprocess.md` does not document
  `VectorFitUnderdeterminedWarning`** — the warning category
  is publicly importable but unmentioned. A reader who tries
  to `warnings.simplefilter("once", ...)` has to discover
  the class name from the source.
- **`docs/api/sources.md`** (once it ships) must enumerate
  the `SourceAdapter` / proposed `SourcesAdapter` helpers
  with a copy-pasteable round-trip example. The Pass-4 +
  Pass-5 Source-discriminator surface is currently
  documented only inside the source-file docstring.
- **`docs/concepts.md`** does not describe the "Engine
  re-use across `World.solve` calls" pattern. Tied to the
  sixth-pass `World.solve` deep-copy finding: the user who
  wants the snapshot-free behaviour should see a recipe for
  the planned `snapshot_sources=False` opt-out before the
  perf cost bites them.
- **`docs/engines/*.md` — Capability matrix sub-section**
  still missing on each engine page. Pass-5 listed this; the
  pages have not been edited. A two-column matrix
  (`feature` × `supported / partial / planned`) at the top
  of every `bem.md` / `mom.md` / `fem.md` / ... gives the
  reader a one-glance picture of why one engine is in beta
  and another is reference-quality.
- **`README.md` does not yet list the `0.5.0` release**.
  The CHANGELOG carries `[0.5.0] — 2026-05-14`, but the
  README "Latest release" / badges sub-section still points
  at `0.4.0` (verified 2026-05-14). README/CHANGELOG drift
  reappears on every release that does not run through
  `scripts/release.py` end-to-end.
- **CLAUDE.md "Audit history"** — append a sixth-pass
  reference. Pass-5 already failed to do this for itself
  (verified above as a Pass-5 Docs-backlog item); sixth
  pass re-confirms.

### Tests (pending implementation) — sixth 2026-05-14 review pass

- **No regression test for the
  `VectorFitUnderdeterminedWarning` *false-positive* path
  under `complex_conj=True`.** Build a `vector_fit(n_poles=2,
  N=2, complex_conj=True)` instance and assert no warning
  fires; today the Pass-5 threshold flags it as
  underdetermined.
- **No regression test for FP64 contract enforcement on
  `LayeredEarth.compute_T_lambda`.** Construct a stub backend
  that returns an FP32 array, invoke `compute_T_lambda`, and
  assert the proposed runtime assertion (`out.dtype == np.float64`)
  trips with a clear message.
- **No regression test for `World.solve(engine,
  snapshot_sources=False)` cost.** Once the keyword lands,
  benchmark a 100-frequency sweep with and without the
  snapshot and assert the difference is below the
  `tests/test_perf_smoke.py` threshold; pin the opt-out so
  the cost is documented.
- **No regression test for `Engine.__eq__` / `Engine.__hash__`
  contract.** Once the sixth-pass `frequency-order` finding
  is resolved, pin the chosen contract
  (`Engine([50,100]) == Engine([100,50])` true or false) so
  it cannot regress in a refactor.
- **No regression test for `io.groundinsight.evaluate_spec`
  rejecting a non-canonical-assumption `rho` symbol.**
  Construct `sp.Symbol("rho", real=False)`-bearing formula,
  call `evaluate_spec` and assert the sixth-pass tightened
  check raises `ValueError`. Today the symbol slips through
  and the failure surfaces deep in `sympy.lambdify`.
- **No regression test for the `compare_engines`
  warning-suppression path.** Build a 4×4 comparison over
  decreasing frequencies and assert exactly one
  `EngineFrequencyOrderWarning` fires (after the sixth-pass
  per-engine internal silencer lands), not 16.
- **No regression test that `set_boundary_conditions` does
  not double-warn on repeat assignment of the same
  non-default value.** Today the helper warns on every
  call, even with identical kwargs.
- **No regression test for `SourcesAdapter = TypeAdapter
  (list[Source])`** once it ships. Round-trip a heterogeneous
  list with one `CurrentSource` and one `VoltageSource` and
  assert each element keeps its discriminator.

### Roadmap candidates — sixth 2026-05-14 review pass

- **`gf.SourcesAdapter = TypeAdapter(list[Source])`** —
  bulk-validation companion to the Pass-4 `SourceAdapter`,
  needed by any YAML / JSON campaign-config import path
  that has more than one source per scenario.
- **`World.solve(engine, snapshot_sources: bool = True)`** —
  explicit opt-out for the Pass-5 deep-copy snapshot. The
  perf-sensitive callers (`convergence_study`, parameter
  sweeps, the cross-engine matrix) can disable the snapshot
  once they have proven their backend does not mutate
  `source.return_to`.
- **`gf.set_warning_policy("verbose" | "once" | "silent")`** —
  global helper to bulk-configure the pass-5 family of
  warning categories (`EngineFrequencyOrderWarning`,
  `VectorFitUnderdeterminedWarning`, the proposed
  `BoundaryConditionWarning`, the proposed
  `VectorFitRealPoleWarning`). Replaces the
  `warnings.simplefilter("once", X)` boilerplate that
  notebooks currently copy-paste per category.
- **`gf.audit_apply(report_path)`** — twin of the
  `groundinsight` proposal. Read a markdown audit's backlog
  bullets and insert them into `[Unreleased]`. Six passes of
  hand-merging strongly motivate the helper.
- **ADR-0011 — cross-repo `_release_shared.py`**. Pass-5
  cross-cutting recommendation #5; sixth pass elevates it
  to a tracked roadmap candidate so the ADR file lands as
  the next merge-able artefact. Mirrors the
  `gi.scripts.release.py` and `gm.scripts.release.py` gaps
  in a single shared implementation.
- **`gf.show_versions()`** — keep the
  cross-package-consistency lemma in the roadmap: when
  `gi.show_versions()` (Pass-5 idea) and
  `gm.show_versions()` (sixth-pass idea, see groundmeas
  roadmap above) land, this one needs to land too with the
  same return-shape contract.
- **`gf.docs.assert_api_pages_exist()`** — a tiny
  walker that asserts every `__all__` member has a
  corresponding mkdocstrings directive somewhere under
  `docs/api/`. Sixth pass elevates this from the Pass-5
  proposal status to an explicit roadmap item because the
  five-pass-in-a-row missing-pages backlog clearly demands
  a structural forcing function.

## Released in 0.6.0 (2026-05-15)

### Changed

- **`vector_fit` under-determined check now respects conjugate-pair
  symmetry** (`postprocess/vector_fitting.py`, sixth 2026-05-14
  audit pass). The previous trigger counted four real DOFs per
  pole regardless of whether the search was constrained to
  conjugate pairs. Under the default ``complex_poles=True`` the
  pair constraint halves the actual free parameters; the
  underdetermined-test is now ``2 * n_independent_poles >
  len(frequencies)`` with
  ``n_independent_poles = n_poles // 2 + (n_poles % 2)``. The
  strict ``>`` admits the uniquely-determined boundary case
  (``n_poles=2, N=2``) that the previous ``>=`` trigger
  rejected as a false positive. The warning text now states
  which ratio applies given the fit mode. Behaviour for
  ``complex_poles=False`` is unchanged. No public API change;
  this is a tightening of the
  `VectorFitUnderdeterminedWarning` envelope.
