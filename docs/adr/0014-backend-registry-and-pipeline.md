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
  models, concrete shells, two-terminal sources, junctions, voltage
  sources (amendment below), source kernels. `Engine.solve` checks a world
  before the backend runs. Hard mismatches (layer count, concrete shells,
  an explicitly requested kernel, junctions, voltage sources) raise
  `BackendCapabilityError` — a subclass of
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

## Amendment 2026.11 — impressed voltage sources in the pipeline (finding 10.51)

`VoltageSource` was a data model only: up to 0.15.0 every backend dropped
it with a warning and solved to zero potentials, the release preparation
made it raise. Decision (scope of the 2026.11 release, 2026-11-14): the
shared pipeline solves it by **modified nodal analysis**. Step 2 lists the
voltage sources next to the impressed currents
(`_pipeline.impressed_voltages`); step 5 appends, after the branch block,
one unknown per voltage source — its current $I_v$, entering Kirchhoff's
current law at `attached_to` ($-I_v$ on the left-hand side) and
`return_to` ($+I_v$) — and one row $\varphi_a - \varphi_r = U$
($\varphi_r = 0$ for remote earth). The image family and the point-matching
family call the same helpers (`_topology._voltage_source_entries`), so the
cross-layer path, inductive branches, frequency sweeps and junction
terminals need no code of their own. A voltage-source terminal is a driven
node: its cluster does not float. A world without voltage sources assembles
the same matrix as before, bit for bit (golden suite strict against the
base). The backend reports the solved currents in `FieldResult.sources`
(`SourceRecord.current`, with the impressed `SourceRecord.voltage`);
`input_impedance()` is $U/I$.

`World.check` rejects what makes the system singular: a voltage source
with both terminals in one ideally bonded cluster, and voltage sources that
close a loop through ideal bonds or remote earth (union-find over clusters
with remote earth as one more node). A voltage source of 0 V is an ideal
short and allowed. The capability `voltage_sources` is declared by `image`,
`image_2layer`, `mom` and `mom_sommerfeld`; `fem` raises
`BackendCapabilityError`. `solve_mutual_*` ignore the sources of the world,
as before. Validation (`tests/test_voltage_sources.py`): the voltage solve
equals the 1 A current-source solve scaled by $U/Z_\text{in}$ in every
backend, two- and three-layer soil, the cross-layer path and inductive
conductors at 0, 50 and 1000 Hz (at most $1.4\cdot 10^{-15}$ relative); a
two-terminal source equals $U/(Z_{11}+Z_{22}-Z_{12}-Z_{21})$ of
`solve_mutual_matrix`, three voltage sources the hand-inverted transfer
matrix of three 1 A solves, a rod Dwight's formula (−0.012 %) and its far
field the hemispherical $\rho I/(2\pi r)$.

## Amendment 2026.11 — internal impedance of a voltage source (user decision D1)

Decision of 2026-11-15: a `VoltageSource` carries an internal series
impedance $Z_q(f) = R + j\omega L$ as two fields,
`internal_resistance_ohm` and `internal_inductance_h` (both ≥ 0, default 0:
the ideal source, which assembles the matrix of the first amendment entry
for entry). $Z_q$ is the one diagonal entry of the source's constraint row,
$\varphi_a - \varphi_r + Z_q I_v = U$, with $I_v$ flowing from the source
into `attached_to` (the terminal voltage is $U - Z_q I_v$). The pipeline
evaluates it per angular frequency (`ImpressedVoltage.terminals(cluster_id,
omega)`); `_voltage_source_entries` writes $\operatorname{Re} Z_q$ into the
real matrix and `_source_reactance_entries` adds $j\operatorname{Im} Z_q$
to its complex copy. A real (galvanic) system with a non-zero $\omega L$
therefore **takes the complex path** rather than being refused. The image
family solves its owner-level system once per frequency (cheap: ×1.00 …
×1.09 the R-only solve at 1 … 20 frequencies on a 1 488-segment station,
homogeneous and two-layer). The point-matching family would factorise its segment-level
system once per frequency — ×2.0 / ×8.1 / ×28 the R-only solve at 1 / 5 /
20 frequencies on 1 488 segments (audit W2R2-02) — so it keeps the one
real factorisation instead: $j\omega L$ enters only the diagonal of the
V source rows, the real block $A_{11}$ (the world with its voltage sources
open) is factorised once with the V + 2 right-hand sides
$[\operatorname{Re} b_1, \operatorname{Im} b_1, A_{12}]$, and each
frequency solves the V × V Schur complement
$(A_{22} - A_{21} A_{11}^{-1} A_{12} + j\omega L)\,x_2 = b_2 - A_{21}
A_{11}^{-1} b_1$ (`_point_matching_source_sweep`; exact, the Woodbury
update the auditor proposed). Measured on the same world (best of three):
×1.06 / ×1.04 / ×1.02, and the results agree with the per-frequency path
to 3.3 · 10⁻¹⁵ and with the scaled 1 A solve to 1.3 · 10⁻¹⁵. A solve at
$f = 0$ alone remains the R-only solve, bit for bit; inside a sweep the
$f = 0$ entry agrees with it to round-off. The same two helpers serve
the image family (including the cross-layer path) and the point-matching
family, so `image`, `image_2layer`, `mom` and `mom_sommerfeld` honour $Z_q$
with conductors, inductive coupling and frequency sweeps; `fem` keeps
refusing voltage sources.

`input_impedance()` of a voltage source is the impedance the source sees,
$U/I - Z_q$, without its own $Z_q$; `SourceRecord` carries $R$ and $L$ so a
result read back from a file computes it alone. Records and sources written
before have 0 / 0: in JSON by the field defaults, in pickles since audit
W2R2-01 through a `__setstate__` that fills the two fields. A non-zero $R$
or $L$ below the smallest normal float is rejected (audit W2R2-10).
`World.check` treats a source with $R > 0$ as no edge of the loop check —
two such sources in parallel, or a ring with one of them, are solvable —
while an inductance alone keeps the source an edge, because $Z_q$ vanishes
at $f = 0$ and `World.check` does not know the engine's frequencies. A
source short-circuited by an ideal bond stays rejected also with $Z_q$ (its
current $U/Z_q$ would circulate through the bond and the source, none
through the soil — the wording of the message since audit W2R2-04).
Validation (`tests/test_voltage_source_impedance.py`): the solve
equals the junction + `lumped_series_resistance_ohm` pattern for a
resistance, and the 1 A current-source solve with $I = U/(Z_q + Z_\text{in})$
at 0, 50 and 1000 Hz in every backend, with inductive conductors (three
earth models) and on the cross-layer path; several sources satisfy
$(Z_T + \operatorname{diag} Z_q)\,\mathbf I = \mathbf U$ with the
transfer matrix of 1 A solves, a two-terminal source
$U/(Z_q + Z_{11}+Z_{22}-Z_{12}-Z_{21})$ of `solve_mutual_matrix`.
