# ADR-0014: Backend registry and shared solve pipeline

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-11-07 |
| **Deciders** | Project maintainers |
| **Scope** | `groundfield.solver.registry`, `Engine.solve`, `Engine.options`, the private modules `solver._pipeline`, `_discretize`, `_topology`, `_inductive`, `_kernels`, `_series`, `_guards`, result provenance and persistence (`groundfield.io.results`) |

## Context

Review pass 10 (findings 10.74, 10.75, 10.77, 10.79, 10.81) found the
solver grown by copying:

- `Engine.solve` was an eight-branch `if` chain. Backends declared nothing
  about what they could solve; an unsupported world failed deep inside a
  backend, or worse, solved with a silent simplification (`return_to`
  ignored, distributed conductors reduced).
- The solve pipeline — discretise, inject the sources, build clusters and
  branches, assemble the inductive block, solve the nodal system, build
  the result — existed in seven copies (991 duplicate lines). The
  Tagg/Sunde series existed in four or five copies with two truncation
  policies, the injection loop seven times, the point kernel five times.
  This is how the length-weighted read-back fix of 0.11 missed `cim` and
  how the post-solve series kept a cap of 100 terms.
- `solver/image.py` (2 500 lines) served as a private library for the
  other backends (63 private cross-module imports).
- Of 32 metadata keys only two were common to all backends; results
  carried no provenance and had no versioned file format.
- Backend settings were flat `Engine` fields (`sommerfeld_*`), accepted
  silently for every backend; the `fem` knobs were unreachable.

## Decision

### Registry

`groundfield.solver.registry` is the authority on backend names:
`register_backend(name, solve_fn, capabilities=..., options_model=...,
deprecated_alias_of=..., experimental=..., forward=...)`.

- **Capabilities** (`BackendCapabilities`): soil layer counts, conductor
  models, concrete shells, two-terminal sources, junctions, source
  kernels. `Engine.solve` checks a world before the backend runs. Hard
  mismatches (layer count, concrete shells, an explicitly requested
  kernel, junctions) raise `BackendCapabilityError` — a subclass of
  `ValueError` and `NotImplementedError`, so earlier `except` clauses keep
  working — and name a backend that can solve the world; soft ones (a
  conductor model the backend reduces, `return_to` ignored) warn with
  `BackendCapabilityWarning`.
- **Names.** Unknown names raise with the closest registered name.
  Deprecated names (`cim`, `bem`, `image_nlayer`) warn with the
  replacement and the removal version; an experimental backend (`fem`)
  warns once per process. Only the requested name is checked: the
  forwarding of `image` to `image_2layer` for a two-layer soil stays
  silent. Alias chains are not allowed.
- **Options.** Backend settings live in `Engine.options`, keyed by
  backend name and validated by the backend's options model
  (`MomSommerfeldOptions`, `FemOptions`); misspelled names raise with a
  suggestion, options for another backend warn. The flat `sommerfeld_*`
  fields of 0.15.0 are deprecated aliases. The image-series controls
  (`image_max_terms`, `image_series_tol`) stay common `Engine` fields
  because every layered backend and the post-solve evaluation read them.
- **Metadata core.** `Engine.solve` writes `backend`,
  `effective_backend`, `kernel`, `groundfield_version`, `frequencies`,
  `n_segments`, `soil_kind`, `earth_inductive_model`, `solve_time_s`,
  `series` and `provenance` into every result (`CORE_METADATA_KEYS`).
- **Third-party backends** register a solve function with the same
  contract, `solve_fn(world, engine, **options) -> FieldResult`, without
  editing `Engine.solve`.

### Shared pipeline

`groundfield.solver._pipeline` implements the solve once:

```
discretise → inject → build_topology → assemble_branches → solve → build_result
```

A backend supplies a Green's-function object (the `ImageGreen` protocol:
`prepare(segments, arrays) -> kernel`, `metadata(solution)`) and, for the
point-matching family, the segment-level solve and read-back
(`_PointMatchingGreen`). The steps are shared by `image`, `image_2layer`,
`mom` and `mom_sommerfeld`, so a fix in one step — two-terminal sources,
junction nodes, the inductive branch block, the burial-depth guards —
reaches every backend at once. `fem`, which solves a different model, is
registered but not on the pipeline. Around the pipeline:

- `_kernels` holds every kernel of the closed-form family (point and line
  kernel, field evaluation, row-block and memory budget), `_series` the
  one `ImageSeries` object of the Tagg/Sunde series used by the solver,
  `FieldResult.potential` and `solve_mutual_*`.
- Geometry queries that were `isinstance` ladders are methods of the
  electrode classes (`segments`, `segment_count`, `wire_polylines`,
  `bounds`, `total_length`, `min_feature_length`).
- `image.py` keeps re-exports of the moved private names for one release
  cycle.

### Provenance and persistence

Every result records `metadata["provenance"]`: package and dependency
versions, platform, the engine configuration and a SHA-256 fingerprint of
the world as built, plus the solve time stamp. `save_result` /
`load_result` write and read a versioned envelope (`groundfield.result`,
schema version 1) with a lossless metadata codec (complex numbers,
arrays, non-finite values); a newer schema raises.

## Consequences

**Positive.** One place for each piece of physics; capability errors at
the call site instead of wrong numbers; results that can be traced and
reloaded; a new backend is a registration plus a Green's-function object.
The refactor was verified against the golden-value suite: bit-identical,
except for the round-off of the reordered two-layer series summation
(at most $7.3\cdot 10^{-16}$ relative).

**Negative.** The private module structure is new; external scripts that
imported private helpers from `groundfield.solver.image` may need the
re-exports (kept) or the electrode methods. The two-layer owner split of
cross-layer electrodes (ADR-0007 amendment 2026.11) is a composed path
inside `image_2layer` until the pipeline offers a split/merge hook.

**Neutral.** Backend names and results are unchanged; deprecated names
keep working until 2027.5 at the earliest (ADR-0013).

## References

- Review pass 10 (2026-11-07), findings 10.74–10.77, 10.79, 10.81,
  decision question 9.
- ADR-0002 (engine family) and its amendment 2026.11.
