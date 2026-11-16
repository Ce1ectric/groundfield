---
title: Review pass 10 (2026-11-07)
description: Review pass 10 (2026-11-07) of groundfield, as recorded in the changelog.
---

# Review pass 10 (2026-11-07)

| | |
|---|---|
| Date | 2026-11-07 |
| Reviewed | 0.15.0 |
| Findings | 10.01–10.95 (pass-9 identifiers kept where a finding was already known) |
| Released in | 2026.11.0 |
| Overview | [Audit history](index.md) · [Changelog](../changelog.md) |

The tenth review pass ran against 0.15.0. Its change concept was decided on 2026-11-07 and recorded in the `[Unreleased]` block of the changelog; the work was carried out by parallel work streams, each of which wrote its release notes as a changelog fragment. The release notes of 2026.11.0 condense these fragments; the concept and the fragments are reproduced here verbatim.

!!! note "Where this text comes from"
    This page was moved verbatim from `CHANGELOG.md` when the release notes were
    condensed in 2026.11.0 (source: commit `30a122d`). Only the markup was adapted:
    reST roles are rendered as code or math, relative links point to the current
    pages, trailing whitespace is removed, and each block stands under a heading
    that names where it was published.
    Paths, test names and numbers are those of the time; the code has changed since.

## Change concept (decided 2026-11-07)

### Planned — review pass 10 (change concept, decided 2026-11-07)

> **Not yet implemented.** This block records the change concept of the tenth
> review pass (2026-11-07, against `0.15.0`) as decided on 2026-11-07. Findings
> are referenced as `10.NN`; pass-9 identifiers (`Fxx`) are kept where an item
> was already known. Each item moves into Added / Changed / Fixed / Security /
> Removed below once it lands together with its test; whatever is still listed
> here when the release is cut moves to the Roadmap instead. Items marked ⚠
> change numerical results.
>
> **One release, calendar versioning.** All items below ship together in the
> first calendar-versioned release. The version scheme changes from SemVer
> (`0.15.0`) to CalVer `YYYY.MM.MICRO` (e.g. `2026.11.0`; month unpadded and no
> hyphen, because PEP 440 drops leading zeros and reads `2026-11` as a
> post-release). `MICRO` counts further releases within the same month. CalVer
> carries no compatibility promise, so deprecations follow a written policy:
> a deprecated name keeps working, warns with `DeprecationWarning`, and is
> removed no earlier than two releases and six months after its announcement.

- **Versioning and release tooling:** `scripts/release.py` computes the CalVer
  version from the release date, validates everything before it writes, and
  runs git in the repository root; tags become `vYYYY.MM.MICRO`.
- **Security (10.01):** `evaluate_spec` passes `impedance_formula` from a
  BusType JSON to `sympy.sympify`, which evaluates Python code. Replace it by
  an allow-list parser that builds the SymPy expression from the AST (numbers,
  `f`, `rho`, `j`, `pi`, `+ − * / **`, `sqrt`/`exp`/`log`, size and exponent
  bounds) and validate on load. groundinsight (local `0.6.0`) already guards
  its formulas; its guard is re-verified against the same attack corpus and
  against every formula groundfield exports.
- **Packaging (10.02, 10.11):** no top-level `scripts` package and no
  `release` console script in the wheel (the same collision exists in
  groundinsight); `py.typed`; `polars` dropped (never imported); `profile.out`
  untracked; `poetry-core` bounded.
- **Silent failures become errors (10.06, 10.43, 10.51, 10.66, 10.67):** unknown
  `attached_to` / `return_to` names, worlds without a finite current source,
  `magnitude=nan`, `cross_section <= 0`, non-finite resistivities, zero-length
  geometry, an ideal conductor with `discretize_segment_length`, and
  `VoltageSource` (currently solved to zero) raise with a named parameter and a
  suggestion; a cross section above 1e-2 m² warns ("mm² given?"); worlds beyond
  `HARD_LIMIT` raise before any allocation unless `max_segments` is raised;
  inert settings warn.
- **⚠ Two-terminal sources and measurement (F01/F30, 10.16, 10.17, 10.38):**
  `Source.return_to` honoured by every backend through one shared injection
  step; leakage-free junction nodes; measurement leads re-wired through them
  (replacing the 5 cm pseudo-rods); `FieldResult.input_impedance(source)`.
- **⚠ Cross-layer electrodes (F02, 10.15, 10.32, 10.50):** one current share per
  electrode and layer in `image_2layer`, `cim` and `solve_mutual_*`; interface
  slivers merged; Galerkin self-term.
- **⚠ Concrete encasement (10.12–10.14):** one mechanism — the solver always
  treats the concrete shell as a distributed series resistance; the lumped
  model only converts its input. Today the lumped shell is dropped by
  `solve_mutual_matrix`, `solve_mutual_field` and every world without a PEN
  service drop while the inflated wire radius is kept (−3 % to −29 % on a
  foundation's self-impedance). `OrtsnetzLayout` honours the shells. An impact
  study on representative AP1 sites compares the old and new numbers.
- **⚠ Exact line-source kernel (10.28):** default reaction-matrix and field
  kernel; the point kernel converges 0.9–1.9 % below the exact uniform-current
  solution. `Engine(kernel="point")` reproduces earlier numbers.
- **⚠ Earth return (F32, F50, F52, 10.20, 10.24, 10.35–10.37, 10.45, 10.46):**
  `carson_series` on the anti-parallel image baseline; Sommerfeld λ-grid
  resolves distant pairs; exact `n >= 3` buried kernel; Neumann inductance also
  on lumped and single-segment conductors; consistent internal-inductance
  convention; opt-in skin effect for conductor resistance.
- **⚠ Post-solve consistency (10.25, 10.26, 10.33, 10.34):** `FieldResult.potential`
  and `solve_mutual_*` use the engine's series controls (today capped at 100
  terms); length-weighted `solve_mutual_matrix` diagonal; backend check and an
  optional reciprocal off-diagonal; line-source evaluation near conductors.
- **⚠ Safety (F13, F51, 10.21, 10.22, 10.42, 10.52):** touch voltage over the
  electrode footprint; uniform-current caveat; the EN 50522 anchors are
  confirmed against EN 50522:2022 Table B.4 (DIN EN 50522:2023-10) and pinned
  as literals in the test; docstrings cite the current edition; optional
  body-impedance / additional-resistance model and spectrum touch voltage.
- **⚠ Fits and export (10.18, 10.19, 10.27, 10.40, 10.41):** `quantity=` on
  `sweep` and the rho-f helpers with the input impedance as default; passivity
  check before export; robust vector fitting with a 0 Hz sample; standard form
  with optional constant term and relative weights; readable exported formulas.
- **⚠ Engines (10.23, 10.29–10.31, 10.47–10.49):** floating passive clusters in
  the `mom` family (reciprocal transfer impedances); `cim` weighted read-back;
  `fem` marked experimental with guards; duplicated layers merged; `cim`,
  `bem` and `image_nlayer` become deprecated aliases; shared engine benchmark
  suite.
- **⚠ Generators (F07, F08, F28, F47, F48, 10.39, 10.53–10.55, 10.78):** one
  network plan and one materialiser behind `TnNetworkGenerator` and the
  layout builder; reproducible resolved configs; rotated mesh foundations split
  at crossings; per-footprint foundations; lead depth honoured; bounded and
  validated Overpass requests with atomic caching; MD5 marked
  `usedforsecurity=False` (identical digests — stored masks stay valid).
- **Performance (10.56–10.65):** row-chunked kernels (memory from 96 B·N² to
  O(N)), one reaction-matrix evaluation per solve, direct matrix assembly in
  `solve_mutual_matrix`, vectorised two-layer series (optional `numba` extra),
  multi-frequency Sommerfeld assembly, chunked field evaluation,
  `sweep(n_jobs=...)`, performance and memory regression tests.
- **Architecture (10.74–10.77, 10.79–10.81):** one kernel module and one solve
  pipeline shared by all backends; backend registry with a common metadata
  core and provenance (F22); versioned `save_result` / `load_result` (F23);
  nested backend options.
- **Usability (10.68–10.73):** typed electrode factories, consistent parameter
  names with aliases, English names for German identifiers with aliases,
  curated namespaces (`gf.plot`, `gf.safety`, `gf.fit`),
  `FieldResult.to_dataframe()`, complex-valued `compare_engines` over all
  frequencies.
- **Standards (10.80, 10.94, 10.95):** `ruff format`, ruff lint with the NumPy
  docstring convention and a physics naming allow-list, mypy, pre-commit, CI
  jobs for lint, types, docs and Python 3.14; tests organised by feature with a
  `finding` marker; supply-chain hardening of the workflows (10.03, 10.04).
- **Documentation (10.82–10.93):** tutorials / how-to / reference / explanation
  structure, versioned tutorial notebooks executed in CI, glossary, units and
  conventions, warnings catalogue, validation page, known limitations,
  factories in the API reference, corrections to the 0.15.0 release notes
  (F47, F48 and F53/F54 were listed as fixed but the code was unchanged; `cim`
  is not bit-identical to `image_2layer` on unequal segment lengths; the F16
  entry describes the symptom of F08), a concise changelog with a list of
  releases that changed numbers.

## Release notes of the work streams

The fragments are reproduced as the work streams wrote them (from `changelog.d/`
of the release branch); only reST roles would be adapted, and none occur. Each
stream wrote its fragment against the state of the branch at that time, so a later
stream may have changed what an earlier one describes.

!!! warning "Corrections (first adversarial audit of this release)"
    Three statements in the fragments below were found to be wrong; the release
    notes of 2026.11.0 carry the corrected wording:

    - **coup1, `references.pollaczek` (10.45):** R′ moves by up to +3.5 % in band
      (100/10 Ω·m, h₁ = 1 m, conductor at 3 m, 1 kHz; +1.4 % for 1000/30 Ω·m,
      h₁ = 5 m, conductor at 3 m in the upper layer), not by “up to +0.5 %”; X′ by up to
      −0.09 %, not “less than 0.02 %”.
    - **coup1, exact removal of the additive image (10.46):** only kernel-level
      numbers are quoted. At solver level (default `sommerfeld` earth return,
      point kernel; two 1 m rods in 100 Ω·m joined by a 95 mm² conductor) Im Z
      moves by +6.0 % for a conductor inclined by 30° and by +18.4 % (50 Hz) to
      +18.5 % (1 kHz) for a steep riser; Re Z is unchanged.
    - **core1a, solver structure (10.75, 10.79):** “every previously importable name
      of `groundfield.solver.image` remains importable” is false for private names
      and re-imports: the `_discretize_*` helpers, `_grid_segments`, the electrode
      classes and `PointSource` that the module re-imported, `_galerkin_solve` of
      `bem` and `mom_sommerfeld`, private helpers re-imported by the other solver
      modules, and the private Pollaczek kernels of `coupling.sommerfeld_inductance`
      are gone.
    - Several fragments of the first waves describe intermediate states that later
      streams changed: the default kernel became `"line"` (core2b, after core1a),
      `touching="exact"` and the anti-parallel `carson_series` baseline became the
      solver default (coup2, after coup1), `cim`, `bem` and `image_nlayer` became
      deprecated aliases (eng2), the 180° counter-source of the layout builder was
      replaced by the junction-based measurement loop (gen2, after core2a), the two
      passivity checks were merged (fitx, after post and secio), and
      `touch_voltage_envelope` replaced `mesh_centres=` by `interior=` (fitx).

!!! warning "Corrections (second adversarial audit of this release)"
    The second audit found these statements of the fragments below to be false
    at the time they were written; the fragments of the `fix2*` streams at the end
    of the page record the fixes, and the release notes carry the corrected
    wording:

    - **core2b, interface slivers (10.32):** dropping an interface cut that would
      leave a part shorter than the wire diameter ignored the contact with the
      other layer (a 4.5 m rod in 1000/30 Ω·m: 195.0 Ω with its tip 15.9 mm into
      the lower layer, 128.6 Ω at 16.1 mm); every cut is now kept (fix2solver).
    - **core2b, “a world without a source stays solvable through `Engine.solve`
      (all results zero)”:** only `fem` did; every backend now raises (fix2solver).
    - **The segment budget (10.06):** conductor sub-pieces were not counted (a
      100 m PEN in 2000 pieces was predicted at 12 segments); the guard now counts
      the nodal system (fix2solver).
    - **Floating clusters in `solve_mutual_*` (10.29):** not only an accuracy
      change; the cost grew with the square of the number of bonded nodes (256
      nodes: 24.3 s → 199.9 s); one factorisation now serves all excitations
      (fix2solver).
    - **`carson_series` on the anti-parallel image (10.20):** wrong for a pair on
      opposite sides of the surface (an overhead lead 1 m above a PEN at 0.7 m:
      mutual reactance 35.9 % low at 1 kHz) and outside Carson's per-metre domain
      (fix2coup).
    - **Lumped inductive conductors (10.24):** under `sommerfeld` a km-long lumped
      conductor was one 16 × 16 quadrature (3 km at 1 kHz: +13.4 % earth-return
      resistance, +11 % at solver level, no warning) (fix2coup).
    - **fixsec, “parse cost bounded” (10.01):** SymPy still expanded symbolic `exp`
      and fractional powers of large powers (67- to 75-character formulas ran longer
      than 60 s), and a literal of 320 nines raised `OverflowError` (fix2sec).
    - **gen2, a shell with ρ_c = ρ_soil (10.12):** a shell of any size switched a
      mesh foundation from one current owner to one per wire (largest touch voltage
      232.3 → 221.2 V at 100 A) (fix2gen).
    - **eng2, deprecated `cim` “bit-identical on every world compared” (10.76):** it is
      bit-identical to `image_2layer` of the same release; through the alias it uses
      the new default line kernel (core2b), so a ring of r = 2 m in 100 Ω·m moves by
      +0.85 % against 0.15.0 (13.31207 → 13.42534 Ω).

!!! warning "Superseded by the third adversarial audit and the user's decisions"
    These statements of the fragments below no longer describe the release; the
    fragments at the end of the page and the release notes do:

    - **core2b, `solve_mutual_matrix`:** `offdiag=None` selected `"probe"` when probe
      points were given. The galerkin off-diagonal is the default of every call; probe
      points need `offdiag="probe"` (ddefaults, user decision Q15).
    - **coup2 and fix1, Sommerfeld assembly:** “the default stays the per-pair
      assembly”, “`"exact"` (default)”. The default is `"tabulated"`; `"exact"` is the
      per-pair rule (ddefaults, user decision Q16).
    - **fitx and fix3solver, `touch_voltage_envelope(interior=None)`** sampled mesh
      cells and ring centres (and the cells of generated mesh foundations) by default.
      The default samples no interior; `interior="meshes"` does (dsafety, user
      decision Q3).
    - **post, `long_duration=True`** for 80 V beyond 10 s. 80 V applies to every
      fault duration above 10 s; the keyword is gone (dsafety, user decision Q4).
    - **secio / fitx, `rho_at_fit` = ρ₁ for layered soil.** The convention is the effective
      resistivity `effective_resistivity(soil)` (drhoeff, user decision Q6).
    - **fix2gen, the nearest-point tree of PEN cables** (chained cabinets, drops hung
      on other drops). Every cabinet has its own cable and a drop taps the cable that
      feeds its building (fix3gen).
    - **fix2solver, “the step at zero penetration is the onset of the contact”.** It is
      a limit of the thin-wire model; a rod end within one wire radius of an interface
      warns (`InterfaceContactWarning`), and the interface diagonal no longer clamps at
      h₁/200 (fix3solver).
    - **fix2coup, wires lying side by side** were switched to touching wires as whole
      pairs, from one segment's view (order dependent); the rule acts pointwise and
      order-free, and the accuracy statements “5e-11”, “1e-9” and “≤ 1e-4” hold only
      where they were measured (fix3coup).

## Stream `infra`: Release tooling, packaging and CI

### Changed

- **Calendar versioning:** releases are numbered `YYYY.MM.MICRO` in PEP 440
  form (`2026.11.0`; month without leading zero, no hyphen; `MICRO` counts
  the earlier releases of the month; tags `v2026.11.0`). The deprecation
  policy is stated at the top of this changelog. Test:
  `tests/test_release_script.py::test_calver_version_from_date_and_tags`.
- **Release tool (10.09):** `python scripts/release.py calver [--date
  YYYY-MM-DD] [--dry-run]` derives the version from the date and the
  existing tags; `python scripts/release.py set X.Y.Z` takes an explicit
  canonical PEP 440 calendar version (pre-releases such as `2026.11.0rc1`
  included). The tool updates `pyproject.toml`, `__version__`,
  `CITATION.cff` (`version`, `date-released`) and turns `[Unreleased]` into
  `## [X] - YYYY-MM-DD`, then commits and tags; it never pushes. The
  `major` / `minor` / `patch` bumps and the options `--no-push`,
  `--allow-dirty` and `--allow-empty-changelog` are gone. Test:
  `tests/test_release_script.py`.
- **Package metadata (10.11):** metadata moved from the legacy
  `[tool.poetry]` tables to `[project]` (PEP 621; license as SPDX expression
  `MIT`, PEP 639); explicit classifiers for Python 3.12, 3.13 and 3.14
  (the auto-generated 3.15 classifier is gone), OS independent,
  Scientific/Engineering, `Typing :: Typed`; build backend bounded to
  `poetry-core>=2.2.0,<3.0.0`. Test:
  `tests/test_packaging.py::test_metadata_uses_the_project_table`.
- **Typed package (10.02):** `groundfield` ships a `py.typed` marker, so type
  checkers use its annotations. Test:
  `tests/test_packaging.py::test_wheel_contents`.
- **Development standards (10.94, 10.80):** ruff lint configuration (F, E,
  W, I, B, UP, D with NumPy convention, N with a physics naming
  allow-list, C901 with complexity 15), mypy configuration with the
  pydantic plugin, pre-commit hooks (ruff format, ruff check, nbstripout,
  file hygiene), ruff 0.17.4 and mypy 2.4.1 pinned in the dev group,
  black removed. CI runs lint, types (non-blocking), tests on Python 3.12,
  3.13 and 3.14 with `pytest -n auto` and coverage, a strict docs build on
  pull requests, and a non-blocking weekly dependency and workflow audit.

### Removed

- **`polars` dependency (10.11):** it was never imported. It is still
  installed with the `groundinsight` extra, because groundinsight 0.3.0
  requires it. Test:
  `tests/test_packaging.py::test_polars_is_neither_a_dependency_nor_imported`.
- **Console scripts and the top-level `scripts` package (10.02):** the wheel
  no longer installs `release`, `generate-third-party-licenses` or a
  `scripts` package. Run the tools from a checkout:
  `python scripts/release.py ...`,
  `python scripts/generate_third_party_licenses.py`. Test:
  `tests/test_packaging.py::test_no_scripts_package_and_no_console_scripts`.

### Security

- **Package collision with groundinsight (10.02):** the v0.15.0 wheel
  installed the same `scripts/release.py` and `release` command as
  groundinsight 0.3.0; installing both overwrote one with the other and
  uninstalling one broke the other. The wheel now contains only
  `groundfield/` and its metadata. Test:
  `tests/test_packaging.py::test_wheel_contents`.
- **CI supply chain (10.03):** every workflow starts with `permissions: {}`
  and grants each job only what it needs; every action is pinned to a full
  commit SHA with the tag in a comment; checkouts do not persist
  credentials (except the docs deployment, which pushes `gh-pages`);
  `snok/install-poetry` and `softprops/action-gh-release` are replaced by
  a pinned `pip install poetry` and the GitHub CLI; tag names reach shell
  code only through environment variables. The release workflow checks
  that the tag equals the version in all three version files, runs the
  full CI as a gate before building, and keeps PyPI trusted publishing.
  zizmor 1.32.0 reports 26 findings on the v0.15.0 workflows and none on
  the new ones.
- **Locked dependencies (10.04):** `pip-audit` on the exported lock (all
  groups and extras) found 100 known advisories in 8 packages; after
  updating pillow (13.0.0), tornado (6.6), urllib3 (2.8.0),
  pymdown-extensions (12.3), certifi (2026.10.23), idna (3.20), pytest
  (9.2.0) and mkdocs-material (9.8.1) it finds none. numpy, scipy, sympy,
  pydantic, matplotlib and pandas keep their locked versions.
- **MathJax (10.08):** the documentation loaded MathJax from a floating
  `cdn.jsdelivr.net/npm/mathjax@3` URL without integrity check. MathJax
  3.2.2 is now vendored with the site (60 files byte-identical to the
  verified npm release, pinned by `SHA256SUMS`); the site loads no
  third-party JavaScript. Test: `tests/test_docs_assets.py`.
- **Release tool (10.09):** the v0.15.0 tool ran git in the caller's
  working directory, rewrote the version files before it validated the
  changelog (an aborted run left bumped files behind), committed on a
  detached HEAD or onto an existing tag, and could not finalise a
  pre-release. It now runs git in the repository root, validates
  everything before it writes (clean tree, branch, tag, agreeing version
  files, non-empty `[Unreleased]`) and restores the files if the commit
  fails. Tests: `tests/test_release_script.py` (marked `finding("10.09")`).

### Docs

- **Formula rendering:** the MathJax configuration started a second
  typesetting pass concurrently with MathJax's own; every formula was
  typeset twice (the copy lands in the hidden assistive-MathML region),
  and pages that autoload a TeX extension (`\boldsymbol` in ADR-0003 and
  ADR-0004) failed with "Cannot read properties of null (reading
  'replaceChild')". Re-typesetting after instant navigation is now
  serialised through `MathJax.startup.promise`.

## Stream `core1a`: Exact line-source kernel, solver structure and memory

### Added

- **Exact line-source kernel (10.28):** `Engine(kernel="line")` treats every
  segment as a uniform line current (Galerkin form of the exact line potential
  `ln((R1+R2+L)/(R1+R2-L))/L`, self term `(2/L)[asinh(L/a) - sqrt(1+(a/L)^2) + a/L]`,
  images with mirrored end points) for the reaction matrix, `FieldResult.potential`
  and `solve_mutual_matrix` / `solve_mutual_field`. Against the exact
  uniform-current resistance (100 Ω·m) a 3 m rod, a 10 m strip, a 10 × 10 m
  foundation and a 20 × 20 m grid agree within 0.0004 % at every segment length
  from 2 m to 0.125 m (point kernel: −0.4 … −1.7 %); a ring of 5 m radius
  converges monotonically as O(ds²) (−0.60 / −0.17 / −0.044 / −0.011 % at
  ds = 2 / 1 / 0.5 / 0.25 m; point kernel −0.95 % at every ds). The default stays
  `kernel="point"`, which reproduces earlier numbers; backends outside the image
  family reject `kernel="line"`. Results record `metadata["kernel"]`;
  `PointSource` gains an optional `direction` (results written before this
  release load unchanged and are evaluated with the point kernel). For 16 house
  grids (1 920 segments) a solve with the line kernel takes about 2.5× the time
  of the point kernel with `numba` installed and about 5× without.
  Test: `tests/test_line_kernel.py`.
- **Geometry queries on the electrode models (10.79):** every electrode kind
  provides `segments(ds, layer_interfaces=None)`, `segment_count(ds)`,
  `wire_polylines()`, `bounds()`, `total_length()` and `min_feature_length()`;
  `groundfield.geometry.WireSegments` holds midpoints, lengths and tangents.
  The solver discretises through them (midpoints, lengths and tangents
  bit-identical to the previous discretiser), so diagnostics, plots and exports
  can use the same conventions instead of their own `isinstance` ladders.
  Test: `tests/test_electrode_geometry.py`.
- **Optional compiled kernels (10.58, 10.28):** when `numba` is installed the
  Tagg/Sunde series matrix and the far-field term of the line kernel are
  evaluated by compiled kernels that give the same matrices bit for bit: the
  series about 3× faster than the numpy path on one core (1 200 segments,
  148 images), the far-field term about 5× (1 920 segments). Tests:
  `tests/test_kernel_performance.py::test_numba_series_kernel_is_bit_identical`,
  `tests/test_line_kernel.py::test_numba_far_field_block_is_bit_identical`.

### Changed

- ⚠ **Two-layer image series summed before the product (10.58):** the series is
  accumulated image by image into one matrix and then multiplied with the
  currents, instead of one matrix product per image. Results of `image_2layer`
  (electrodes inside the upper layer) and `cim` move by floating-point round-off
  only: at most 7.3e-16 relative on the golden cases `cim_rod_two_layer`,
  `cond_isolated_pen_sommerfeld_two_layer`, `defaults_two_layer_ring_auto`,
  `img2_grid_upper_resistive` and `img2_ring_rods_within_layer` (all
  `within_tol`). The reaction matrices of `solve_mutual_*` and of the `mom`
  family are unchanged bit for bit. A 30-house two-layer network
  (3 669 segments, 11 frequencies) solved 10–15× faster in two back-to-back
  comparisons on the same machine (memory 1 236 → 219 MiB).
- **Solver structure (10.75, 10.79):** `solver/image.py` (2 526 lines) is split
  into private modules (discretisation, guards, kernels, inductive assembly,
  topology, pipeline); `solve_image` and `solve_image_2layer` run through one
  pipeline and differ only in the Green's-function object; the Tagg/Sunde
  series is one `ImageSeries` object used by the solver, `FieldResult.potential`
  and `solve_mutual_*`. Every previously importable name of
  `groundfield.solver.image` remains importable. Numbers are unchanged
  (golden values bit-identical). Tests: `tests/test_solver_structure.py`,
  `tests/test_image_pipeline.py`.
- The warning for ignored non-current sources of `image` and `image_2layer`
  uses the common text of the other backends; `FieldResult.clusters` lists the
  electrodes in world order.

### Fixed

- **Memory of the image-family kernels (10.56):** the kernels no longer build
  `(N, N, 3)` difference tensors (96 bytes per segment pair). A 30-house network
  (3 687 segments) peaks at 115 MiB instead of 1 248 MiB and solved about 10×
  faster in back-to-back runs on the same machine; an 80-house network
  (9 712 segments) solves in about 2 s with 773 MiB (before: about 8.4 GiB
  needed). With rods reaching the lower layer of a two-layer soil (ADR-0007
  path) the 30-house network peaks at 510 MiB instead of 1 250 MiB. Results are
  bit-identical. Above 1 GiB per reaction matrix the kernels switch to row
  blocks (results then differ by round-off only). Tests:
  `tests/test_kernel_performance.py::test_solve_memory_peak_is_bounded`,
  `tests/test_kernel_performance.py::test_cross_layer_solve_memory_peak_is_bounded`.
- **Reaction matrix evaluated twice per solve (10.57):** the segment potentials
  after the solve reuse the assembled matrix. Bit-identical. Test:
  `tests/test_kernel_performance.py::test_one_reaction_matrix_assembly_per_solve`.
- **`solve_mutual_matrix` built the reaction matrix as `kernel(eye(N))`
  (10.59):** the matrix is assembled directly; Z_G is bit-identical and a
  20-house two-layer catalogue matrix (2 466 segments) was 23–31× faster in two
  back-to-back comparisons (memory 604 → 146 MiB). Test:
  `tests/test_kernel_performance.py::test_mutual_matrix_without_identity_product`.
- **`FieldResult.potential` unchunked (10.62):** large evaluations run in row
  blocks; a 200 × 200 grid over 3 687 sources peaks at 17 MiB (v0.15.0 needs about
  96 bytes per point·source pair, 13 GiB here). Two-layer evaluations share every
  `1/r` block between real and imaginary part and skip an all-zero imaginary
  part. Evaluations up to 16 MiB of kernel are bit-identical. Test:
  `tests/test_kernel_performance.py::test_potential_on_a_large_grid_is_memory_bounded`.
- **Point-source arrays rebuilt on every `potential()` call (10.65):** cached
  privately on the result (not serialised). Test:
  `tests/test_kernel_performance.py::test_point_source_arrays_are_cached_and_invalidated`.

### Removed

- **Dead code (10.81):** `solver.image._potential_kernel`,
  `solver.image._segment_z_extent`, `solver.image_2layer._two_layer_potential_kernel`
  and the empty module `solver.field_study`. Test:
  `tests/test_solver_structure.py::test_dead_helpers_are_removed`.

## Stream `core1b`: Backend registry, metadata, provenance and result files

### Added
- **Backend registry (10.74):** `groundfield.solver.registry` with
  `register_backend(name, solve_fn, *, capabilities, options_model=None,
  aliases=(), deprecated_alias_of=None, experimental=False, forward=None)`,
  `get_backend`, `available_backends`, `unregister_backend`,
  `BackendCapabilities` and `BackendSpec`. `Engine.solve` dispatches through
  the registry instead of an if-chain, so a backend is added without editing
  `Engine.solve`; the automatic forwarding of `image` to `image_2layer` /
  `image_nlayer` by soil model is kept. A deprecated alias warns with its
  replacement and the earliest removal (`2027.5`); an experimental backend
  warns once per process (`ExperimentalBackendWarning`). Tests:
  `tests/test_backend_registry.py::test_new_backend_runs_without_editing_engine_solve`,
  `::test_deprecated_alias_warns_with_replacement_and_removal_version`,
  `::test_experimental_backend_warns_once_per_process`,
  `::test_image_is_auto_forwarded_by_soil_model`.
- **Capability check before the solve (10.74):** a world the backend cannot
  represent (soil layer count, concrete-shell electrodes, an explicitly
  requested source kernel) raises `BackendCapabilityError` before any
  backend code runs and names the registered backends that can solve it; a
  stack of equal layers counts as the merged stack. A backend that solves
  the world with a simplification warns with `BackendCapabilityWarning`:
  `fem` reduces distributed and inductive conductors to lumped resistive
  branches, and every backend still returns the current of a source with
  `return_to` through remote earth. Tests:
  `tests/test_backend_registry.py::test_capability_error_names_a_backend_that_can_solve_the_world`,
  `::test_capability_check_runs_before_the_backend`,
  `::test_two_terminal_source_warns_when_the_backend_ignores_return_to`,
  `::test_fem_warns_when_it_reduces_a_distributed_conductor`.
- **Metadata core (10.74):** every result carries `backend`,
  `effective_backend`, `kernel`, `groundfield_version`, `frequencies`,
  `n_segments`, `soil_kind`, `earth_inductive_model`, `solve_time_s`,
  `series` and `provenance` (`groundfield.solver.engine.CORE_METADATA_KEYS`)
  next to the backend-specific keys, which are unchanged. `series` records
  the engine's image-series controls for the post-solve field evaluation.
  Test: `tests/test_backend_registry.py::test_every_backend_reports_the_metadata_core`.
- **Backend options (10.74):** `Engine.options`, keyed by backend name and
  validated by `MomSommerfeldOptions` (`mom_sommerfeld`) and `FemOptions`
  (`fem`), with a did-you-mean message for misspelled names;
  `Engine.backend_options(name)` returns the options a backend runs with.
  Tests: `tests/test_engine_options.py::test_nested_sommerfeld_options_reach_the_backend`,
  `::test_misspelled_option_is_rejected_with_a_suggestion`.
- **Provenance (10.77, F22):** `metadata["provenance"]` of every solve holds
  the groundfield, Python, numpy, scipy and pydantic versions, the platform,
  the engine configuration, `world_fingerprint` (SHA-256 of the canonical
  world JSON, `groundfield.io.world_fingerprint`) and the UTC time of the
  solve. Tests: `tests/test_result_persistence.py::test_every_solve_records_provenance`,
  `::test_world_fingerprint_is_canonical_and_sensitive`,
  `::test_fingerprint_describes_the_world_as_built`.
- **Versioned result files (10.77, F23):** `groundfield.io.save_result` /
  `load_result` write and read one standard-JSON file with the envelope
  `{"format": "groundfield.result", "schema_version": 1, "written_by",
  "provenance", "result"}` (no pickle, atomic write, no silent overwrite);
  a newer schema raises `ResultFormatError`. Tests:
  `tests/test_result_persistence.py::test_save_and_load_result_round_trip`,
  `::test_load_result_rejects_a_newer_schema`.
- **`FieldResult.to_dataframe()` (10.69, 10.72):** tidy table with one row
  per frequency and electrode (`frequency_hz`, `electrode`, `cluster`,
  `potential_v`, `current_a`, `cluster_impedance_ohm`); `to_dataframe("sources")`
  gives one row per frequency and source with the impressed current and the
  input impedance. pandas is imported on first use. Tests:
  `tests/test_result_dataframe.py::test_electrode_table_is_tidy_and_matches_the_accessors`,
  `::test_source_table_carries_the_input_impedance`.

### Changed
- **`Engine.backend`** is validated against the registry (a plain `str`
  annotation; `Backend` stays the `Literal` of the built-in names for type
  hints). An unknown name raises with the closest registered name as
  suggestion. Test:
  `tests/test_backend_registry.py::test_unknown_backend_name_suggests_the_closest_registered_one`.
- **Unsupported worlds fail earlier:** `cim`, `bem`, `mom` and `image_nlayer`
  on soils with three or more distinct layers, and every non-image backend on
  concrete-shell electrodes, now raise `BackendCapabilityError` from
  `Engine.solve` and list the backends that can. It subclasses `ValueError`
  and `NotImplementedError`, the types `image_nlayer`, `cim`, `bem` and the
  concrete-shell guard raised before; `mom` raised `TypeError` for three or
  more layers.
- **One default for the image-series cap (10.74):** `DEFAULT_IMAGE_MAX_TERMS
  = 300` and `DEFAULT_IMAGE_SERIES_TOL = 1e-6` (in
  `groundfield.solver.registry`) are the engine defaults and are recorded on
  every result as `metadata["series"]`. The image-series controls stay
  common `Engine` fields because six backends and the post-solve layer read
  them. Test: `tests/test_engine_options.py::test_series_default_is_one_documented_value`.
- **Engine JSON:** `options` replaces the four flat `sommerfeld_*` keys;
  engine JSON written by 0.15.0 still loads, silently when it carries the
  default values. Test: `tests/test_engine_options.py::test_v0150_engine_json_loads_without_warnings`.
- **Result JSON** writes non-finite floats of typed fields as `"Infinity"` /
  `"NaN"` strings instead of `null`, and tagged objects in `metadata` (see
  the `groundfield.solver.result` module docstring), so it is standard JSON
  and round-trips exactly.
- Warnings of the pre-flight resolution check of `Engine.solve` are
  attributed to the caller's line instead of a line inside groundfield.

### Fixed
- **Lossy result JSON (10.77):** `metadata["penetration_depth"]` lost its
  float keys and its `inf`, and numpy values in the metadata could not be
  serialised at all. Tests:
  `tests/test_result_persistence.py::test_result_json_round_trip_is_lossless`,
  `::test_numpy_and_non_json_metadata_survive_the_round_trip`.
- **Backend options silently accepted for another backend (10.74):** the
  `sommerfeld_*` fields were accepted by every backend without effect;
  options for a backend that will not run now warn
  (`BackendOptionsWarning`). Test:
  `tests/test_engine_options.py::test_options_for_another_backend_warn`.
- **`fem` knobs unreachable (10.74):** `n_radial`, `n_axial`, `r_far_factor`,
  `z_far_factor` and `far_field` can now be set through
  `Engine(options={"fem": {...}})`. Tests:
  `tests/test_engine_options.py::test_fem_knobs_are_reachable_through_engine_options`,
  `::test_fem_far_field_option_reaches_the_solver`.
- **`Engine.solve` let a backend rewrite the caller's sources (10.72):**
  `World.solve`, `Engine.solve` and `run_simulation` now share one
  identity-preserving snapshot and restore of `World.sources`. Test:
  `tests/test_solve_entry_points.py::test_every_entry_point_restores_the_sources`.
- **Inert engine settings accepted silently (10.51):** `Engine.tolerance` and
  `Engine.max_iterations` are read by no backend; a non-default value now
  warns. Test:
  `tests/test_engine_options.py::test_tolerance_and_max_iterations_warn_that_they_have_no_effect`.

### Deprecated
- `Engine.tolerance` and `Engine.max_iterations` (no effect): a non-default
  value or attribute access emits `DeprecationWarning`; removal no earlier
  than `2027.5`.
- The flat fields `Engine.sommerfeld_lambda_max_factor`, `sommerfeld_epsabs`,
  `sommerfeld_epsrel` and `sommerfeld_max_osc_panels`: use
  `Engine(options={"mom_sommerfeld": {...}})` or `MomSommerfeldOptions`. They
  are still accepted (constructor, JSON, attribute read and write); removal
  no earlier than `2027.5`. Test:
  `tests/test_engine_options.py::test_flat_sommerfeld_fields_are_mapped_and_deprecated`.
- `World.solve(snapshot_sources=False)`: every entry point takes the
  snapshot now, so the opt-out saves nothing; removal no earlier than
  `2027.5`. Test: `tests/test_solve_entry_points.py::test_snapshot_opt_out_is_deprecated`.

### Removed
- The private, unreachable `Engine._stub_result` and `Engine._dispatch`
  (replaced by the registry; part of 10.81).

## Stream `secio`: Formula security and the groundinsight export

### Security
- **BusType impedance formulas are parsed, never executed (10.01):**
  `evaluate_spec` passed `impedance_formula` to `sympy.sympify`, which
  evaluates the text as Python code, so a crafted BusType JSON file ran
  arbitrary code or exhausted memory (`7**7**7*0+rho` builds an integer of
  about 696 000 digits, `9**9**9**9` never finishes). A new allow-list
  parser (`groundfield.io._formula`) parses the text with `ast` and builds
  the SymPy expression itself: numbers, `f`, `rho`, `j` / `I`, `pi`, `E`,
  `+ - * /`, `**` with a literal exponent (at most 64 in magnitude) and
  `sqrt` / `exp` / `log`, after Unicode NFKC normalisation, ASCII only, at
  most 10 000 characters, 2 000 syntax nodes and nesting depth 100. Every
  number stays within 1e-300 to 1e300 and powers are checked before they
  are computed, which also stops towers such as `(((10**60*f)**64)**64)**64`
  (the review prototype accepted this one after 24 to 46 s with a
  15.7-million-digit integer). `BusTypeSpec.from_dict` and
  `load_bustype_json` validate the formula on load and raise
  `EvaluateSpecError` naming the offending element (a missing formula
  raises `EvaluateSpecError` instead of `KeyError`). All 384 formulas the
  0.15.0 exporter wrote parse to identical values, so existing BusType files
  keep loading. The corpus `tests/data/formula_corpus.json` (115 accepted
  formulas with reference values, 79 rejected strings) is shared with
  groundinsight's guard. Tests:
  `tests/test_formula_security.py::test_evaluate_spec_does_not_run_code_from_a_formula`,
  `tests/test_formula_security.py::test_every_rejected_formula_fails_on_load_within_bounds`,
  `tests/test_formula_security.py::test_every_rejected_formula_fails_in_evaluate_spec_within_bounds`.

### Added
- **Passivity check before BusType export (10.40):** `to_bustype_dict`,
  `save_bustype_json`, `to_bustype` and `save_bustype_to_db` evaluate the
  exported formula from 0 Hz to twice the highest fitted frequency (dense
  linear and logarithmic grid, the samples and the midpoints between them,
  points around every pole; for the standard form over the fitted `rho`
  range) and refine the minima. `metadata["passivity"]` records `passed`,
  `min_re_Z_Ohm`, `f_at_min_Hz`, `rho_at_min_Ohm_m` and the checked ranges.
  `Re Z < 0` emits `NonPassiveFitWarning`; the new keyword
  `require_passive=True` raises `NonPassiveFitError` (a `ValueError`) and
  writes nothing. Example from the review: a 4-pole fit of passive data
  (rms 4.4 mOhm) has Re Z = -0.0194 Ohm at 5.507 Hz between the 0 Hz and
  50 Hz samples and is now flagged. Test:
  `tests/test_io_groundinsight.py::test_in_band_non_passive_vector_fit_is_flagged_on_export`.

### Changed
- ⚠ **Readable exported formulas (10.73):** vector fits are written as
  partial fractions with float coefficients, each rounded once to `decimals`
  significant digits with Python's correctly rounded formatting (identical
  on every platform), e.g. `28.4652054274 - 54.0434542796/(2*pi*f*j +
  27.5749267026)`. The exporter no longer calls `nsimplify` (exact
  fractions with integers of up to 254 digits) and no longer merges the
  partial fractions into one rational function (`sympy.simplify` produced
  terms up to `f**4`). Formulas of 2 to 10 poles shrink from 273-622 to
  126-469 characters, and the export takes 0.04-0.14 s instead of
  0.39-2.23 s (same session, passivity check included). `decimals` must be
  a positive integer; values above 17 count as 17. Numbers move only at the
  rounding level, at most half a unit in the last printed digit per
  coefficient (round trip below 1e-10 relative at the default 12 digits).
  Affected: anyone comparing formula text, e.g. diffs of BusType JSON
  files. Golden cases: none moved (`fit_rho_f_standard_sweep`
  `bustype/eval` is bit-identical). Tests:
  `tests/test_io_groundinsight.py::test_vector_fit_formula_prints_floats_not_long_rationals`,
  `tests/test_io_groundinsight.py::test_round_trip_accuracy_follows_decimals`.

### Fixed
- **Actionable `rho_at_fit` error for vector-fit exports (10.84):** the
  documented `save_bustype_json(fit, ...)` call for a vector fit raised
  "rho_at_fit is required" without saying what to pass. `rho_at_fit` stays
  required for a `VectorFitResult` (the BusType records the soil
  resistivity it belongs to); all four export functions now raise the same
  `ValueError` naming the function, an example call and the layered-soil
  convention, before a file is written or groundinsight is imported, and
  `rho_at_fit` must be positive and finite. Test:
  `tests/test_io_groundinsight.py::test_vector_fit_without_rho_at_fit_raises_an_actionable_error`.
- **BusType provenance stamp:** `metadata["source"]` read
  `groundfield unknown` when the package ran from a source checkout (and
  went stale under an editable install); it now uses
  `groundfield.__version__` (review item R5-12). Test:
  `tests/test_io_groundinsight.py::test_bustype_metadata_records_the_running_groundfield_version`.

### Internal
- The groundinsight round-trip test for vector fits asserts 1e-9 (ADR-0008)
  instead of 1e-3; a new test compares `evaluate_spec` with groundinsight's
  `compute_impedance` to 1e-9 and skips without groundinsight
  (`tests/test_io_groundinsight.py::test_evaluate_spec_matches_groundinsight_evaluation`).
  The bridge tests no longer skip the whole module when groundinsight is
  missing.

## Stream `post`: Safety, sweeps, fits and plots

### Added
- **Prospective touch voltage (F51, 10.52):** `permissible_prospective_touch_voltage_en50522(t_f, *, R_H=0, R_F1=0, rho_s=None, HF=1, long_duration=False)` implements EN 50522:2022, Annex B.4 / Equation (A.3): U_vTp = U_Tp(t_f) + I_B(t_f)/HF · R_A with R_A = R_H + R_F1 + 1.5 m⁻¹ · ρ_s and the body current of Table B.1. Test: `tests/test_safety.py::test_prospective_touch_voltage_figure_b3_parameter_sets`.
- **True-RMS touch voltage over a fault spectrum (10.52):** `touch_voltage_spectrum(result_or_results, world, *, electrode, currents, ...)` scales the touch voltage of every harmonic with its fault current and returns sqrt(Σ|U_T,k|²) at one foot position (AP2 harmonic cases). Test: `tests/test_safety.py::test_touch_voltage_spectrum_is_the_rms_of_the_scaled_harmonics`.
- **80 V for long-duration faults (10.42):** `permissible_touch_voltage_en50522(t_f, long_duration=True)` returns the 80 V that EN 50522:2022 allows for current-flow durations considerably longer than 10 s (requires `t_f > 10 s`; `math.inf` accepted). The default keeps the 85 V end value of Table B.4. Test: `tests/test_safety.py::test_en50522_long_duration_allows_80_volt`.
- **`sweep(n_jobs=...)` (10.64):** process pool (spawn, BLAS pinned to one thread per worker); rows in serial order, worker warnings re-issued, bit-identical to `n_jobs=1` when the caller also runs a single-threaded BLAS. Callables must be importable (not lambdas or notebook functions; a clear `TypeError` says so). Test: `tests/test_sweep.py::test_parallel_sweep_is_bit_identical_to_the_serial_sweep`.
- **`check_passivity(fit, f_max, n=...)` (10.41, 10.40):** returns a `PassivityReport` (`passed`, `min_re_Z`, `f_at_min`, `as_dict()`); dense grid plus pole frequencies plus bounded refinement for vector fits, exact corner check for standard-form fits. It finds the −0.019 Ω dip at 5.5 Hz of the review's 4-pole fit. The export bridge has its own check (SEC/IO); the two are to be unified. Test: `tests/test_vector_fitting.py::test_check_passivity_finds_an_in_band_violation_between_samples`.
- **Plot components (10.73):** `plot_potential_contour`, `plot_potential_profile`, `plot_potential_radial` and `plot_surface_potential` take `component="real" | "imag" | "abs" | "phase"`. Test: `tests/test_plotting.py::test_profile_plots_and_labels_the_requested_component`.
- **`read_csv_with_provenance(path)` (10.73):** reads the CSV files of `groundfield.io.csv` and returns the table plus the provenance header. Test: `tests/test_io_csv.py::test_csv_files_carry_a_provenance_header`.
- **New names:** `UniformCurrentFieldWarning`, `NEAR_FIELD_DISTANCE_M`, `RhoFStandardQualityWarning`, `VectorFitQualityWarning`, `PassivityReport`; fields `RhoFStandardFit.k0`, `.max_relative`, `.weights`, `.quantity` and `VectorFitResult.rms_relative`, `.quantity`; keywords `touch_voltage(anchor=)`, `touch_voltage_envelope(anchor=, mesh_centres=, return_points=)`, `step_voltage(world=)`, `vector_fit(rms_tol=)`.

### Changed
- ⚠ **Input impedance is the default of `sweep`, `convergence_study` and the rho-f exporters (10.18, 10.19):** all take `quantity="input" | "cluster" | "electrode"` and `source=`. `"input"` (default) is the driving-point impedance (φ_attached − φ_return)/I_source; the v0.15.0 behaviour is `quantity="cluster"` for `sweep` / `convergence_study` and `quantity="electrode"` for `rho_f_from_field_result` / `rho_f_standard_from_results` (`electrode_name` is now optional). Numbers move wherever conductors carry current out of the feeding cluster: the cluster impedance of a station was 1.26 / 1.79 / 3.09 times the input impedance with 1 / 3 / 8 houses on a PEN, and the exporters fitted a ring with two bonded rods 17 % too high. Affected: groundinsight BusTypes exported with the rho-f helpers and sweeps of TN worlds; the AP1 catalogue is not affected (it computes its own driving-point impedance). With several current sources `source=` is now required (v0.15.0 silently used the first). Golden cases: `fit_rho_f_standard_sweep` (model halves: the two rods act in parallel through the PEN). Test: `tests/test_impedance_quantity.py::test_sweep_default_reports_the_input_impedance`.
- ⚠ **Standard rho-f form with constant term and relative weights (10.27):** `fit_rho_f_standard` fits Z = k0 + k1 ρ + (k2 + j k3) f + (k4 + j k5) f ρ with `k0=True` and `weights="relative"` (1/|Z|) by default, reports `max_relative` (worst pointwise relative error) next to `rms_relative` and warns with `RhoFStandardQualityWarning` above 5 %. In ρ₁ sweeps of two-layer soil the worst pointwise error drops from 15–27 % (ring) and 43 % (AP1-like station) to 0.8–3.3 % and 6.2 %; homogeneous single-electrode sweeps are unchanged (k0 → 0). Exported formulas gain the constant term. `k0=False, weights="uniform"` reproduces v0.15.0 bit for bit. Affected: rho-f formulas exported to groundinsight, including AP1 exports (the stored samples are unchanged). Golden cases: `fit_rho_f_standard_sweep`. Test: `tests/test_rho_f_standard.py::test_two_layer_ring_sweep_is_fitted_within_five_percent`.
- **CSV files (10.73):** the electrode and cluster tables gain a `frequency_Hz` column, every file starts with `# key: value` provenance lines (groundfield version, UTC time, backend, frequency, solve provenance when present), and the writers raise `FileExistsError` instead of overwriting unless `overwrite=True`. Read with `read_csv_with_provenance` or `pandas.read_csv(path, comment="#")`; `provenance=False` writes a plain CSV. Test: `tests/test_io_csv.py::test_csv_writers_refuse_to_overwrite`.
- **Plot labels (10.73):** potential plots label the drawn component (`Re φ in V` for the default) instead of `Potential φ in V`. Test: `tests/test_plotting.py::test_default_component_is_the_real_part_with_an_honest_label`.
- **`split_factor` (10.18)** divides by the source current of the solve (snapshot in `FieldResult.sources`) instead of the world's current source; the current-balance tables document that their impedances are cluster / electrode impedances, related to the input impedance by Z_in = s · Z_cluster. Test: `tests/test_impedance_quantity.py::test_split_factor_uses_the_solved_source_current`.

### Fixed
- ⚠ **Touch voltage anchored at the electrode centre (F13, 10.21):** `touch_voltage` and `touch_voltage_envelope` measured the 1 m from the connection point, so for a 20 m × 20 m grid (100 A, 100 Ω·m) the feet stood above the grid: 9.92 V instead of 69.6 V at +x, envelope 9.92 V instead of 111.4 V; a 4-arm star 275 V instead of 818 V beyond the arm tip. The feet now stand `distance` from the nearest wire of the electrode's footprint, on the side given by `direction`; the envelope walks that offset curve and adds the mesh centres of mesh / grid electrodes. `anchor="connection_point"` reproduces v0.15.0; rods are unchanged. Affected: touch-voltage assessments of grids, meshes, stars, strips, polylines and rings (rings at +x unchanged). Golden cases: `img_mixed_cluster_hom` (`safety/touch_1m`: strip ×3.2, star ×2.1). Test: `tests/test_touch_voltage_footprint.py::test_grid_envelope_covers_the_one_metre_offset_curve`.
- **Uniform-current field caveat (10.22):** touch and step voltages from the `image` family (one uniform current per electrode) near grids, meshes, stars, strips, open polylines and bonded clusters deviate from `mom` by up to +19 % (touch) and −21 % (step, non-conservative) 1 m outside a 20 m grid corner; rings, rods and closed loops agree within 1 %. `touch_voltage*`, `step_voltage` and `touch_voltage_spectrum` now emit `UniformCurrentFieldWarning` once per call when a uniform-current result is evaluated within 5 m of such an electrode. Test: `tests/test_touch_voltage_footprint.py::test_uniform_current_backend_warns_once_per_call_near_a_grid`.
- **EN 50522 permissible touch voltage (F44, 10.42):** the test compared the interpolant with the constant it interpolates; it now checks literals from EN 50522:2022, Table B.4 (DIN EN 50522 (VDE 0101-2):2023-10). The module values were confirmed against the standard and do not change; the deviations quoted in the review came from a secondary reproduction. Docstrings cite the 2022 edition (Table B.4, Figure 8) instead of "EN 50522:2010, Figure B.3" and no longer call it a low-voltage curve. Test: `tests/test_safety.py::test_en50522_docstrings_cite_the_current_edition`.
- **Vector fit collapse with a 0 Hz sample (10.41):** a 0 Hz sample with an imaginary part (measured or averaged data) made the pole relocation move a pole onto the origin; the fit collapsed to Z ≈ 0 (relative rms 0.99 in 20 of 20 noisy data sets, no warning) or numpy raised `LinAlgError`. 0 Hz samples are now projected onto their real part (bit-neutral for real DC samples), an imaginary DC part above 0.1 % warns, a relative rms above `rms_tol` (1 %) warns with `VectorFitQualityWarning`, and solver failures raise `ValueError`. Test: `tests/test_vector_fitting.py::test_dc_sample_with_imaginary_part_does_not_collapse_the_fit`.
- **VTK header injection and float precision (10.10):** a newline in `world.name` corrupted the legacy VTK header, and `float` declarations quantised UTM northings (5.2e6–6.2e6 m) by up to 0.25 m. The header line is sanitised and limited to 256 bytes; POINTS and SCALARS are `double`; the z-down convention is documented. Tests: `tests/test_io_vtk.py::test_world_name_cannot_break_the_header`, `tests/test_io_vtk.py::test_points_are_double_precision_at_utm_scale`.

## Stream `api`: Factories, namespaces and soil parameter names

### Added

- **Typed factories (10.68):** `create_electrode` has one `@overload` per
  electrode kind and returns the concrete class
  (`create_electrode(world, "ring", ...)` is a `RingElectrode`), so IDEs
  show the parameters of the chosen kind and mypy reports a misspelled
  kind or keyword. Every string option of the factories is a `Literal`
  type: `ElectrodeKind`, `SourceKind` and `KernelKind` (new, in
  `groundfield.api`), and `Backend`, `EarthInductiveModel`,
  `ConductorType`, `SoilCoupling`, `InductanceModel` where they are
  defined. mypy now finds all four typos of the review example (wrong
  kind, wrong keyword, `coupling_to_soil="galvanik"`, `kind="Current"`);
  it found none before. Test:
  `tests/test_api_factories.py::test_mypy_finds_the_four_typos`.
- **"Did you mean …?" errors (10.68):** an unknown electrode kind
  (`ValueError`), an unknown keyword of a kind (`TypeError`, naming the
  kinds that do have it), an unknown source kind (`ValueError`) and an
  unknown electrode name as a conductor end (`KeyError`) name the
  closest valid names. Tests:
  `tests/test_api_factories.py::test_unknown_electrode_kind_suggests_the_closest`,
  `::test_unknown_electrode_keyword_suggests_the_closest`,
  `::test_keyword_of_another_kind_is_explained`,
  `::test_unknown_source_kind_suggests_the_closest`,
  `::test_unknown_conductor_end_suggests_the_closest`.
- **Curated namespaces (10.72):** `gf.plot` (all plot helpers), `gf.safety`
  (touch and step voltages, EN 50522 limit) and `gf.fit` (vector fitting
  and the rho-f standard form) are new modules (`groundfield.plot`,
  `groundfield.safety`, `groundfield.fit`); `gf.io` groups the export.
  They hold the same objects as the flat names, which keep working;
  `dir(gf)` lists the namespaces. Test:
  `tests/test_package_namespace.py::test_namespace_is_an_importable_module`.
- **`create_engine(kernel=...)`:** chooses the field / reaction-matrix
  kernel (`"point"` or `"line"`, `KernelKind`). Test:
  `tests/test_api_factories.py::test_create_engine_accepts_the_point_kernel`.
- **`groundfield.utils`:** `suggest`, `did_you_mean`, `warn_deprecated`,
  `warn_inert`, `InertConfigurationWarning` (a `UserWarning`),
  `REMOVAL_VERSION` (`"2027.5"`). Deprecation and inert-configuration
  warnings point at the caller's line and are not emitted when
  groundfield's own code is the caller. Test:
  `tests/test_utils_helpers.py::test_helpers_stay_silent_for_groundfield_callers`.

### Changed

- **Faster `import groundfield` (10.72):** post-processing, IO,
  generators, the geo ingest, references, cross-engine validation and
  diagnostics are imported on first use (PEP 562); pandas, matplotlib,
  sympy and scipy are no longer loaded at startup. Median import time
  798 ms → 223 ms and 1169 → 346 loaded modules (15 fresh interpreters,
  shared 2-core sandbox). `__all__` is unchanged. Test:
  `tests/test_package_namespace.py::test_import_does_not_load_heavy_dependencies`.
- **Soil parameter names (10.69):** the model layer uses the literature
  symbols — `HomogeneousSoil.rho`, `SoilLayer.rho`, `SoilLayer.h`
  (`TwoLayerSoil.rho_1`, `rho_2`, `h_1` unchanged). The canonical world
  JSON writes `rho` / `h`; JSON written by 0.15.0 still loads. No
  numerical change. Golden: `gen_layout_builder` (`world/json` only).
  Test: `tests/test_soil_models.py::test_world_json_written_by_v0150_loads`.
- **`create_electrode` with an unknown keyword raises `TypeError`** (was a
  pydantic `ValidationError`), like the other factories, whose explicit
  signatures always raised `TypeError`.
- **`create_engine` forwards only the arguments you pass;** everything
  else takes the `Engine` default (single source of the defaults).
  `create_engine()` builds the same engine as `Engine()`. Test:
  `tests/test_api_factories.py::test_create_engine_forwards_only_explicit_arguments`.
- **Soil models validate assignment,** and `SoilModel` is a union
  discriminated by `kind` (inferred from the keys when a mapping omits
  it), so a soil is validated once and errors name the selected model.
- **`model_copy(update=...)` of a soil model rejects unknown keys** (a
  misspelled key was silently stored next to the fields and the copy
  kept the old value). Test:
  `tests/test_soil_models.py::test_model_copy_rejects_unknown_fields`.
- **`run_simulation(world, engine)`** is documented and tested as
  `world.solve(engine)`, with the same source-snapshot semantics. Test:
  `tests/test_api_factories.py::test_run_simulation_has_world_solve_semantics`.

### Fixed

- **Non-finite soil parameters (10.67):** `resistivity = inf`,
  `rho_1 = inf`, `rho_2 = inf`, `h_1 = inf`, an infinite layer thickness
  or `relative_permittivity` were accepted; an infinite resistivity then
  failed inside the solver with `LinAlgError: Singular matrix`. They are
  now rejected at construction and on assignment, naming the parameter.
  Tests: `tests/test_soil_models.py::test_soil_parameters_must_be_finite_and_positive`,
  `::test_assignment_is_validated`.
- **`MultiLayerSoil` structure errors (10.67):** an empty stack, a
  single layer with a finite thickness, a half-space above the bottom
  and a finite bottom layer each raise a message that names the layer
  and the fix; a one-layer stack with `h=None` stays a valid
  homogeneous half-space. Test:
  `tests/test_soil_models.py::test_multilayer_structure_errors_are_clear`.
- **Inert settings were silent (10.51 part, F57):** a non-default
  `BoundaryConditions` field set through the constructor,
  `create_world(boundary=...)` or a world JSON, and a non-default soil
  `relative_permittivity`, now warn with `InertConfigurationWarning`. No
  backend reads them: with every boundary field changed, all eight
  backends solve bit-identically, and ε_r = 1 and 81 give identical
  impedances (below 1 kHz the displacement current is at most
  ωε₀ε_rρ = 4.5·10⁻³ of the conduction current for ρ = 1000 Ω·m,
  ε_r = 81). `World.set_boundary_conditions` keeps its single warning.
  Tests: `tests/test_boundary_conditions.py::test_non_default_field_warns_on_construction`,
  `::test_boundary_conditions_are_inert_in_every_backend`,
  `tests/test_soil_models.py::test_non_default_relative_permittivity_warns`.
- **`annotations` leaked into the top-level namespace (10.72):**
  `gf.annotations` (the `__future__` feature object) is gone. Test:
  `tests/test_package_namespace.py::test_annotations_does_not_leak`.

### Deprecated

Old soil parameter names (10.69). They keep working as keyword, JSON key,
attribute and `model_copy(update=...)` key, warn with
`DeprecationWarning`, and will be removed in 2027.5 at the earliest.
Passing an old and its new name together raises. Type checkers already
report the old keywords (only the canonical names are part of the typed
signature); at run time they keep working.

| Deprecated | Replacement |
|---|---|
| `HomogeneousSoil(resistivity=...)`, `HomogeneousSoil.resistivity` | `rho` |
| `SoilLayer(resistivity=...)`, `SoilLayer.resistivity` | `rho` |
| `SoilLayer(thickness=...)`, `SoilLayer.thickness` | `h` |

Test: `tests/test_soil_models.py::test_old_keyword_warns_and_maps_to_canonical_field`.

## Stream `gen1`: Generators: network plan, footprints, reproducibility

### Added
- **Network plan (10.78):** `NetworkPlan` with `PlannedSite`, `PlannedElectrode`,
  `PlannedJunction`, `PlannedBranch`, `ConductorParameters`, `PlannedMeasurement`
  and `PlannedSource` is the pure-data description of one realisation (sites with
  concrete electrodes, PEN junctions, cables, measurement set-up, sources, soil).
  `TnNetworkGenerator.plan()` and `LvNetworkLayout.plan()` produce it,
  `materialise_plan(plan)` turns it into a `World`; `build()` / `to_world()` are
  plan + materialise. A plan round-trips through JSON and rebuilds the same world.
  `GroundingSystemSpec.resolve_electrodes`, `plan_site` and `with_footprint`
  expose the steps of `build_at`. Test: `tests/test_network_plan.py`.
- **Foundation outline of the layout builder (10.54):** `LvNetworkLayout.to_world`
  and `plan` take `foundation_outline`: `"ombr"` (default, oriented minimum
  bounding rectangle of the footprint, the rule of the OSM placement of
  `TnNetworkGenerator`), `"polygon"` (ring along the real outline, DIN 18014,
  ring-style specs only) or `"spec"` (the fixed spec geometry of v0.15.0).
  Test: `tests/test_footprint_foundations.py`.
- **Explicit OSM building-tag filter:** every `building=*` value is a footprint
  by default (also `garage`, `shed` and `building=no`), as before.
  `parse_overpass_payload`, `query_and_project` and `LvNetworkLayout.from_osm`
  take `exclude_building_tags`; `groundfield.geo.NON_SERVICE_BUILDING_TAGS`
  (carport, container, garage, garages, greenhouse, hut, no, roof, ruins, shed,
  shelter, tent) is the documented opt-in set. Test:
  `tests/test_overpass_ingest.py::test_building_tag_filter_drops_non_service_buildings`.
- **Reproducible generator draws (10.53):** `TnNetworkGenerator.build()` and
  `plan()` accept `rng=` (seed or `numpy.random.Generator`).

### Changed
- **One network plan and one materialiser (10.78):** both front-ends
  (`TnNetworkGenerator`, `LvNetworkLayout`) produce a `NetworkPlan`; one module
  (`groundfield.generators.materialise`) builds foundations and their concrete
  shells, PEN junctions, PEN and service cables, the measurement loop and the
  sources. The layout module (`ortsnetz_builder.py`, about 2700 lines) is
  split into the package `groundfield.generators.layout` (model, routing,
  measurement, penetration, plot). Worlds are unchanged: 18 of 20 reference
  worlds are byte-identical to v0.15.0, the other two list the same conductors
  in another order and solve bit-identically; the refactor alone passes all
  golden values in strict mode. Test:
  `tests/test_network_plan.py::test_tn_build_is_materialised_plan`.
- **English names in the generator layer (10.69):** see *Deprecated* for the
  full table. Generated element names (`trafo_*`, `kvs_*`, `house_*`) are data
  and stay unchanged, so stored worlds and results keep their keys.
- ⚠ **`sample_world` collapses nested distributions once per realisation
  (10.53):** as documented, `cfg.sample(rng)` now replaces every distribution of
  the config tree, also those inside the electrode specs of a building type, so
  all buildings of a type share the drawn value in `sample_world`. v0.15.0
  left these to per-building draws while reporting the config as resolved.
  `build()` on an unresolved config still draws per building; plain
  `presence_prob` and `jitter_m` still draw per site. Affected: `sample_world`
  realisations of configs with distributions inside specs. Golden cases: none
  moved.
- **Overpass retries:** also on HTTP 503, honouring `Retry-After` (capped at
  60 s); `max_retries` is limited to 5. Non-integer `timeout_s` raises
  `TypeError` (it is part of the query text). Test:
  `tests/test_overpass_ingest.py::test_retries_honour_retry_after_and_are_bounded`.

### Fixed
- **Resolved configs replay their realisation (10.53, F07):** `cfg.sample()`
  left distributions in nested specs and in `building_counts`; replays of the
  "resolved" config with seeds 0/1/2 gave 15/15/9 electrodes at 337/280/168
  Ohm m instead of 15 electrodes at 331 Ohm m. Every distribution is now
  resolved (dicts in sorted key order), and `sample_world(rng)` draws the whole
  realisation from `rng` (the construction seed gave 5 vs 7 houses for the same
  `rng`). Test:
  `tests/test_generator_reproducibility.py::test_resolved_config_replays_the_world_for_any_seed`.
- ⚠ **Stable penetration mask without OSM ids (10.53):**
  `LvNetworkLayout.foundation_mask` keyed footprints without `osm_id` by their
  list index; reversing the list changed the equipped houses and Z by 2.8 %.
  They are now keyed by their outline; footprints with `osm_id` keep their keys,
  so stored masks of OSM layouts stay valid. Affected: masks of footprints
  without `osm_id` at penetration below 1. Golden cases: none moved. Test:
  `tests/test_generator_reproducibility.py::test_mask_without_osm_id_ignores_footprint_order`.
- ⚠ **Layout foundations follow the building footprints (10.54, F28):**
  `LvNetworkLayout.to_world` gave every house the same 10 x 10 m foundation at
  the footprint centroid. Foundations now take size, orientation and centre
  from the footprint (oriented minimum bounding rectangle), like the OSM
  placement of `TnNetworkGenerator`. In three test streets (8 houses, 100 Ohm m,
  50 Hz, image backend) the substation impedance moves by 0 % (10 x 10 m
  squares), +0.6 % (12 x 9 m houses rotated by 0 to 35 deg) and +5.2 %
  (6 x 4 m garages alternating with 30 x 15 m blocks). Affected: every layout
  world whose footprints are not 10 x 10 m axis-aligned squares;
  `foundation_outline="spec"` reproduces v0.15.0. Golden cases:
  `gen_layout_builder` unchanged (10 x 10 m squares). Test:
  `tests/test_footprint_foundations.py::test_foundation_size_follows_the_footprint`.
- ⚠ **Rotated mesh foundations without the singularity clamp (10.39):** rotated
  mesh foundations were built from full-length strips whose segment midpoints
  could coincide at the crossings; the solver's 1 mm clamp then made a
  10 x 10 m, 2 x 2 mesh at 30 deg 11.8 % too high at a segment length of
  2.0 m and 11.0 % at 1.5 m (no error at 2.5 m or 1.0 m). Crossed wires are now
  polylines with a vertex at every crossing, so crossings fall on segment ends
  as in `GridMeshElectrode`; ring-style foundations keep their four strips.
  The oriented bounding rectangle of a square reports the axis closest to +x,
  and a foundation at a multiple of 90 deg is built on the axis-aligned path:
  mesh foundations of squares and north-south rectangles change by +1.0 to
  +1.2 % (or -9.5 % where the clamp fired), ring foundations by less than
  0.002 %. Note: with the uniform-current `image` backend a rotated mesh stays
  about 1.1 % below the axis-aligned mesh of the same rectangle (six wires with
  one current each instead of one); `mom` and `bem` give the same value for
  both, closer to the rotated one. Affected: mesh-style foundations on rotated
  or right-angle footprints (OSM placement, layout builder). Golden cases:
  none moved. Test:
  `tests/test_rotated_foundations.py::test_rotated_mesh_has_no_clamp_and_converges_smoothly`.
- **Jitter distribution (10.55, F47):** `ManhattanGridPlacement(jitter_m=Uniform(...))`
  raised `TypeError` ("Unable to apply constraint ge"); the bound now applies to
  the float only, a distribution is drawn once per site and a draw of 0 or less
  leaves the site on its grid point. Test:
  `tests/test_placement.py::test_manhattan_grid_jitter_accepts_a_distribution`.
- ⚠ **Measurement leads at their configured depth (10.55, F48):**
  `MeasurementLeadConfig.depth_m` was drawn and discarded: buried and overhead
  leads ran on the same diagonal from the substation ring (z = 0.6 m) to the
  auxiliary rod head (z = 0). A lead now runs horizontally at z = `depth_m`
  (z > 0 below the surface) between the anchors and stays connected to both;
  `depth_m = 0` is placed 0.1 m above ground (`SURFACE_LEAD_HEIGHT_M`; the
  surface plane is singular for the image kernels and the Sommerfeld kernel
  degrades within about ten wire radii of it), a galvanic lead above ground is
  rejected. In the golden geometry (perfect-mirror earth) the probe potential
  moves by 2.5e-4 at 50 Hz and its coupled imaginary part by +37 % (overhead)
  and +21 % (buried) at 1 kHz; v0.15.0 gave identical results for both lead
  types. Affected: `TnNetworkGenerator` worlds with measurement leads. Golden
  cases: `gen_tn_measurement_overhead_leads` (potentials up to 2.5e-4, currents
  up to 2.8e-4, lead end points). Test:
  `tests/test_measurement.py::test_buried_lead_runs_at_its_depth`,
  `tests/test_measurement.py::test_overhead_lead_lies_above_ground`.

### Deprecated
- English names replace the German identifiers and capitalised unit suffixes
  of the generator layer (10.69). Every old name keeps working, emits a
  `DeprecationWarning` naming the replacement, and is removed no earlier than
  2027.5. Configurations written by v0.15.0 (JSON keys `kvs`,
  `source_magnitude_A`, `star_kvs`, `resistivity`, `thickness_m`, ...) still load.

  | old (v0.15.0) | new |
  |---|---|
  | module `groundfield.generators.ortsnetz_builder` | `groundfield.generators.layout` |
  | `OrtsnetzLayout` | `LvNetworkLayout` |
  | `KvsPlacement` | `CabinetPlacement` |
  | `KvsConfig` | `CabinetConfig` |
  | `StarKvsTopology`, `kind="star_kvs"` | `StarCabinetTopology`, `kind="star_cabinet"` |
  | `TnNetworkConfig.kvs` | `TnNetworkConfig.cabinets` |
  | `TnNetworkConfig.source_magnitude_A` | `TnNetworkConfig.source_magnitude_a` |
  | `CabinetConfig.quote_per_100_buildings` | `CabinetConfig.count_per_100_buildings` |
  | `RadialTrunkTopology.slots_per_kvs` | `RadialTrunkTopology.slots_per_cabinet` |
  | `RadialTrunkTopology.kvs_spacing_m` | `RadialTrunkTopology.cabinet_spacing_m` |
  | `RadialTrunkTopology.plan_feeder_kvs_positions()` | `RadialTrunkTopology.plan_feeder_cabinet_positions()` |
  | `LvNetworkLayout.add_kvs()` | `LvNetworkLayout.add_cabinet()` |
  | `LvNetworkLayout.kvs_placements` | `LvNetworkLayout.cabinet_placements` |
  | `from_osm(kvs_lat_lons=, kvs_xys=)` | `from_osm(cabinet_lat_lons=, cabinet_xys=)` |
  | `to_world(kvs_grounding=)` | `to_world(cabinet_grounding=)` (also on the new `plan`) |
  | `source_magnitude_A=` of `to_world`, `measured_grounding_impedance`, `verify_current_balance` | `source_magnitude_a=` |
  | result keys `positive_side_A`, `auxiliary_side_A`, `expected_*_A`, `abs_error_*_A`, `tolerance_A` of `verify_current_balance` | `positive_side_a`, `auxiliary_side_a`, `expected_*_a`, `abs_error_*_a`, `tolerance_a` |
  | `HomogeneousSoilSpec.resistivity` | `HomogeneousSoilSpec.rho` |
  | `SoilLayerSpec.resistivity`, `SoilLayerSpec.thickness_m` | `SoilLayerSpec.rho`, `SoilLayerSpec.h` |

  The module alias `groundfield.generators.tn_ortsnetz` keeps working as
  before. Test: `tests/test_generator_naming.py`.

### Security
- **Bounded, validated Overpass requests (10.05):** `timeout_s` was put into
  the query text unchecked (a string could inject Overpass QL); NaN or
  out-of-range coordinates and radii reached the server; the response was read
  into memory without limit; an HTTP-200 payload with a runtime-error `remark`
  was cached as an empty result; a writer killed mid-way left a truncated cache
  file that broke every later call. Now `timeout_s` must be an integer in
  [1, 900] s, coordinates finite WGS84 values and the radius at most 10 km;
  endpoints must use https (http only for localhost); the body is streamed and
  capped (`max_response_bytes`, default 64 MiB); error remarks and malformed
  payloads raise `OverpassError` and are not cached; cache files are written
  atomically and an unreadable one is fetched again with a warning; a
  non-default endpoint is part of the cache key (the default endpoint keeps the
  old key, existing caches stay valid); the query text of valid requests is
  unchanged. Test: `tests/test_overpass_ingest.py` (mocked transport; one real
  request behind the `network` marker).
- **MD5 on FIPS systems (10.07):** the penetration-mask hash called
  `hashlib.md5()` without `usedforsecurity=False` and failed on FIPS builds of
  Python. The digest is unchanged, so stored masks stay valid. Test:
  `tests/test_generator_reproducibility.py::test_mask_works_where_md5_is_blocked_for_security`.

## Stream `eng1`: Engines: floating clusters, cim, layer stacks, fem, benchmarks

### Added

- **Engine benchmark suite B1–B13 (10.76, R2 §3.3):** `tests/benchmarks/`
  holds reference worlds with analytic or independently computed reference
  values (hemisphere, thin-wire Galerkin rod, closed-form thin ring,
  Tagg/Sunde series, mpmath three-layer kernel, floating-port two-port)
  and one explicit tolerance table per backend
  (`tests/benchmarks/tolerances.py`; `python -m tests.benchmarks.tolerances`
  prints it as Markdown for the validation page). Skipped by default; run
  with `pytest tests/benchmarks --run-benchmark`. The image-family
  cross-layer rod (B7) is a strict xfail until 10.15 lands.
- **`FemReductionWarning` (10.23):** `fem` names every reason why its
  equivalent-hemisphere reduction does not model the world (several
  electrodes in a cluster, several clusters in a driven world, a layer
  interface between the electrode and its hemisphere), once per solve.
  Exported from
  `groundfield.solver.fem`. Test:
  `tests/test_fem.py::test_fem_warns_when_electrode_and_hemisphere_straddle_an_interface`.
- **`compare_engines` tolerance table (10.71):** `EngineComparison.rows`
  (one `ComparisonRow` per quantity, frequency and real / imaginary part,
  with values, reference, deviation, tolerance and verdict), `frequencies`,
  `impedances` (complex values per engine and frequency),
  `input_deviations`; new keyword `component_floor` (default 1e-3).
  `summary()` prints the table. Test:
  `tests/test_compare_engines.py::test_compare_engines_tolerance_table_is_explicit`.
- **`mom_sommerfeld` convergence metadata (10.47):** `converged` (bool,
  true only if every frequency converged), `converged_per_frequency`,
  `n_unconverged_entries` and, when the two-layer diagonal is the
  Tagg/Sunde series (every segment in the upper layer), `n_terms_used`.
  Test:
  `tests/test_mom_sommerfeld.py::test_solve_reports_convergence_in_the_metadata`.
- **`ComplexImageFit.max_error` (10.49):** maximum error of the
  complex-image fit on the verification grid including λ = 0. Test:
  `tests/test_cim.py::test_fit_converged_flag_bounds_the_maximum_error`.

### Changed

- ⚠ **`mom`, `bem`, `mom_sommerfeld`: source-free clusters float (10.29).**
  A cluster without a source is now an unknown with zero net current (a
  floating equipotential) instead of being dropped and read back as the
  mean of the incident field. Numbers move for every world with a passive
  cluster in these backends: potentials read at a passive electrode by
  −59 % (rod through h₁ = 2 m in 1000/30 Ω·m: 3.548 → 1.440 Ω transfer
  impedance) to +1.5 % (100/1000 Ω·m), −0.4 % in homogeneous soil; the
  driven electrode's impedance by −0.009 % to −0.49 % because the floating
  neighbour now perturbs the field (R2 two-port worlds, 4 m apart).
  Passive networks joined by finite conductors carry their equalising
  current. Affected: probe and auxiliary electrodes, unbonded neighbours
  and passive conductor networks solved with the MoM family; the image
  family and the AP1 catalogue path are unaffected. Golden cases: none of
  the stored cases has a passive cluster in these backends (bit-identical);
  new case `mom_two_port_passive_ring_two_layer`.
- ⚠ **`cim` reports the length-weighted node potential (10.30)**, as
  `image_2layer` has since 0.15.0 (F10). Numbers move on electrodes with
  unequal segment lengths: −3.9 % (rod split unevenly at h₁ = 2 m:
  71.060 → 68.262 Ω), −1.1 % (the F10 rod shape); uniform segments are
  unchanged. Golden cases: `cim_rod_two_layer` bit-identical; new case
  `cim_grid_unequal_segments`.
- ⚠ **Canonical layer stack (10.48).** The layered backends (`mom`, `bem`,
  `cim`, `mom_sommerfeld`, and `image_nlayer` through `as_layer_stack`)
  merge adjacent layers of equal resistivity and drop zero-thickness layers
  before choosing a kernel: `MultiLayerSoil([ρ, ρ, ρ])` is homogeneous and
  `[ρ₁, ρ₂, ρ₂]` is the two-layer soil it describes. Stacks that raised
  before now solve; `mom` accepts a `MultiLayerSoil` with at most two
  distinct layers; `FieldResult.soil` carries the canonical soil model
  (with the relative permittivity of the upper layer of each merged
  group), so `potential()` works for such stacks. Numbers move only for
  equal-resistivity declarations whose electrodes cross the declared
  interface (no longer split there): +0.07 % (`mom`, `bem`,
  `mom_sommerfeld`) and +0.42 % (`cim`) for a 3 m rod through h = 2 m in
  `TwoLayerSoil(100, 100, 2)`.
  `fem` keeps the declared stack. Golden cases: none moved; new case
  `mom_sommerfeld_duplicated_stack`.
- ⚠ **`fem`: the truncation sphere no longer costs radial resolution
  (10.49).** For a layer stack deeper than 1 m the radial line count grows
  with ln(r_far / a_eq), so the radial step ratio stays that of the
  homogeneous mesh. Layered `fem` results move up by +0.16 % (h₁ = 5 m),
  +0.41 % (30 m), +1.07 % (1 km) and +1.64 % (10 km) for a 1.5 m rod with
  ρ₂ = ρ₁ (from −0.16 % … −1.62 % to within ±0.005 % of the homogeneous
  value); homogeneous soil and stacks with Σh ≤ 1 m are unchanged.
  `metadata["fem_mesh"][cluster]["n_radial"]` reports the count used.
  Affected: every layered `fem` result. Golden cases: `fem_rod_hom`
  bit-identical; new case `fem_rod_two_layer_deep`.
- **`fem` is experimental (10.23).** Module and `solve_fem` docstrings say
  what the equivalent-hemisphere reduction is and is not (no longer "the
  third independent cross-check" and "better than 5 %"); the registry flag
  follows with the backend registry.
- **`compare_engines` compares complex impedances at every frequency
  (10.71):** cluster impedances and the input impedance of every current
  source, real and imaginary part separately, at all frequencies (up to
  0.15.0: Re Z of the clusters at the first frequency). Engines with
  different frequency lists raise `ValueError`; a value that is
  non-finite for some engines only is a failure. Comparisons that passed
  before may now be reported inconsistent; `deviations` per cluster is
  the worst over all frequencies and both parts.
- **`metadata["solver"]` is `"point_matching"` for `mom`, `bem` and
  `mom_sommerfeld` (10.31)** (was `"galerkin"` / `"collocation"` /
  `"galerkin"` for one and the same scheme).

### Fixed

- **Transfer impedances of the MoM family are reciprocal (10.29).** Z_AB
  and Z_BA from two solves differed by 4.0e-3 (homogeneous) to 1.46 (rod
  through the interface); now below 1e-8 (cross-layer quadrature) and
  7e-16 (closed form). Test:
  `tests/test_mom_family.py::test_two_port_transfer_impedance_is_reciprocal`.
- **`cim` is bit-identical to `image_2layer` and reciprocal (10.30)**, as
  its `reduces_to` metadata and the 0.15.0 notes claimed. Test:
  `tests/test_cim.py::test_cim_is_bit_identical_to_image_2layer_for_unequal_segments`.
- **Physically two-layer or homogeneous `MultiLayerSoil` stacks no longer
  raise (10.48)** in `cim`, `bem`, `mom`, `mom_sommerfeld` and
  `image_nlayer`. Test:
  `tests/test_layer_stack.py::test_duplicated_layer_stack_is_solved_like_two_layers`.
- **`mom_sommerfeld`: a capped oscillation budget is no longer silent
  (10.47).** When `max_osc_panels` binds, every rung of the refinement
  ladder shares the capped J₀ grid, so two rungs could agree while the
  absolute floor `epsabs` dominated: kernels 5 km apart over a 5 cm layer
  were up to 3.1e-3 wrong without a `SommerfeldConvergenceWarning` (7 of
  1050 stress points). A binding budget now counts as non-convergence;
  the largest silent error on the stress grid is 2.0e-10. The truncated
  Tagg/Sunde series of the two-layer diagonal warns like in `mom`. Test:
  `tests/test_mom_sommerfeld.py::test_kernel_warns_or_converges_when_the_oscillation_budget_binds`.
- **`fit_complex_images` no longer reports `converged=True` for a fit that
  misses Γ₁ by more than `rms_tol` (10.49).** The flag was RMS-gated:
  26 of 300 random 3–6-layer stacks passed with a maximum error of up to
  3.5e-3 between the fitting samples; the maximum error on the
  verification grid including λ = 0 is now part of the gate (0 of 300
  silent). No solver path uses the fit. Test:
  `tests/test_cim.py::test_fit_converged_flag_bounds_the_maximum_error`.
- **`compare_engines` sees the reactance, every frequency and non-finite
  results (10.71).** Perfect-mirror and Sommerfeld earth return (input
  reactance 0.600 vs 0.376 Ω at 1 kHz) were reported consistent. Test:
  `tests/test_compare_engines.py::test_compare_engines_sees_the_reactance_at_every_frequency`.
- **`fem` warns where its reduction is not a valid cross-check (10.23):**
  +1092 % for a 3 m rod through h₁ = 2 m, −92 % for a 10 m ring above it,
  −35 % for a ring with three bonded rods, transfer impedance 0 between
  clusters — all silent before. Test:
  `tests/test_fem.py::test_fem_warns_for_a_multi_electrode_cluster`.
- **`fem` truncation (10.49):** see Changed. Test:
  `tests/test_fem.py::test_fem_deep_interface_does_not_degrade_the_resolution`.

### Documentation

- **`mom`, `bem` and `mom_sommerfeld` are documented as one point-matching
  scheme (10.31):** pulse basis, delta test at the segment midpoints,
  point-source near field. `mom` is not Galerkin and its bordered system is
  not symmetric; it lies 1.0 % (ds = 0.5 m) to 1.7 % (ds = 0.0625 m) below
  the exact thin-wire Galerkin solution of a 3 m rod and drifts by −0.96 %
  over that ladder, while `image` is closer at every ds; `bem` assembles
  the identical matrix; `mom_sommerfeld` shares the discretisation and is a
  kernel reference, not an independent scheme. The stale advice to lower
  `lambda_max_factor` on the delegated cross-layer path is removed (factors
  30 … 20 000 change that kernel by ≤ 1.3e-8). No physics change. Test:
  `tests/test_mom_family.py::test_mom_family_reports_one_point_matching_scheme`.

### Removed

- The unused private helpers `cylindrical_radius` and
  `_split_layer_thicknesses` of `groundfield.solver._layered` (10.81).

## Stream `coup1`: Coupling: image conventions, Sommerfeld, Pollaczek, Carson

### Added

- **Image conventions for the inductive stack (10.20 preparation, F32):**
  `build_inductance_matrix(..., image="none" | "additive" | "antiparallel")`
  (the legacy `use_image=True/False` maps to `"additive"`/`"none"`), and the
  same `image=` argument on `build_carson_correction_matrix`,
  `build_sommerfeld_correction_matrix` and the Sommerfeld pair integrals: a
  correction states which image the inductance matrix it is added to carries,
  so `j·ω·L(image) + ΔZ(image)` is the same physical matrix for every
  convention. Carson's correction is documented as defined against the
  anti-parallel image; the defaults reproduce the numbers of 0.15.0 for
  `carson_series` and `perfect_mirror`. The solver switches `carson_series` to
  the anti-parallel baseline in a later step of this release.
  Tests: `tests/test_sommerfeld_accuracy.py::test_total_impedance_does_not_depend_on_the_image_convention`,
  `tests/test_sommerfeld_accuracy.py::test_carson_on_the_antiparallel_baseline_matches_carson`.
- **Multi-frequency earth-return builders (10.60, 10.63):**
  `build_sommerfeld_correction_matrices(seg_endpoints, radii, omegas=..., earth=...)`
  and `build_carson_correction_matrices(...)` return all frequencies at once;
  geometry, λ grid and the Bessel table are built once per segment pair. Each
  frequency is bit-identical to the single-frequency builder. On the inductive
  PEN branches of a 2-house street network (34 branches, 11 frequencies) the
  Sommerfeld assembly takes 1.5 s CPU against 7.6 s for eleven single-frequency
  calls and 31.8 s in 0.15.0.
  Carson's `P`, `Q` are memoised on `(a, θ)` (bit-identical; 9130 → 2321
  evaluations and 3.9× less CPU time on a 10-house network).
  Tests: `tests/test_sommerfeld_accuracy.py::test_multifrequency_matrices_are_bit_identical`,
  `tests/test_carson_pq.py::test_multifrequency_carson_matrices_are_bit_identical_and_cached`.
- **Exact touching segment pairs, opt-in (10.46):** `touching="exact"` on
  `neumann_mutual` and `build_inductance_matrix` uses Grover's closed forms for
  segments that share an end point (chain neighbours, bends) or lie on one
  axis; the 16×16 Gauss–Legendre rule is −0.3 % … +0.8 % off for such pairs, and
  a straight wire split into `n` segments now keeps the self inductance of the
  unsplit wire exactly. The default stays `"quadrature"` until the solver
  switches. Test: `tests/test_neumann_touching.py::test_chain_sum_is_refinement_invariant`.
- **Per-segment internal inductance (10.46):** `include_internal` of
  `build_inductance_matrix` accepts a boolean array. The package-wide convention
  (DC internal inductance μ₀/8π per metre included; compare with references at
  `GMR = a·e^(−1/4)`) is documented in `groundfield.coupling.inductance`, the
  Carson and Pollaczek references. Test:
  `tests/test_coupling_conventions.py::test_internal_inductance_convention_matches_references`.
- **Opt-in skin effect (10.37):** `Conductor(resistance_model="skin")` with
  `Conductor.internal_impedance_per_length(f)` and
  `Conductor.skin_correction_per_length(f)` (the correction relative to the DC
  convention), based on the new
  `groundfield.conductors.solid_round_internal_impedance` (exact Bessel
  solution): 150 mm² Al has R_ac/R_dc = 1.231 at 550 Hz and 1.550 at 1 kHz,
  internal inductance 0.740× at 1 kHz. The default `"dc"` changes nothing; the
  solver uses the correction in a later step of this release. Test:
  `tests/test_skin_effect.py::test_150mm2_aluminium_resistance_ratio`.
- **Close, interpenetrating and crossing wires (10.44):**
  `groundfield.conductors.find_close_wires(endpoints, radii, names, ...)`,
  `find_close_conductors(conductors, ...)` and `segment_distance` report wire
  pairs of different owners closer than `rel_threshold·(a_i + a_j)`, classified
  as crossing / interpenetrating / close (junctions at shared end points are
  skipped by default). Test: `tests/test_conductor_proximity.py::test_parallel_rods`.
- **`LayeredEarth.interface_depths` and `LayeredEarth.layer_of(depth)`.**

### Changed

- ⚠ **`references.pollaczek` is exact for two layers (10.45):** the self and
  mutual impedances use the exact buried-conductor amplitude (lower-interface
  term included) and the `K0(γ·GMR)` head instead of `ln(2h/GMR)`. R′ moves by
  +0.15 % (1 m) and +1.0 % (3 m) at 30 Ω·m, 1 kHz (homogeneous), by up to +0.5 %
  in band near a layer interface, and by +40 % at 1 MHz (Tsiamitros Case V); X′
  by less than 0.02 % in band. Conductors below the first interface and pairs in
  different layers are now supported exactly. Affects users of
  `pollaczek_self_impedance`, `pollaczek_mutual_impedance`,
  `loop_impedance_conductor_earth` (the AP1 catalogue builds its own z′ table
  from these). Golden cases: none.
- **Carson P, Q (10.46):** the asymptote is used only for a > 15 (was 5), where
  it is accurate to 1e-7; moves `carson_series` only for pairs with a > 5
  (e.g. more than 560 m apart at 1 kHz in 100 Ω·m). Golden cases: none.

### Fixed

- ⚠ **Sommerfeld λ grid resolves distant pairs (10.36):** the logarithmic
  region ended at max(p_skin, 1/z) whatever the pair distance, so J0 was
  unresolved there, and the linear region was capped at 4096 panels without a
  check: pair entries were off by −24 % (Re, 200 m) and +274 % (Re, 400 m),
  +2.35 % (Im, 400 m). The grid now follows the F14 policy of `layered_green`
  (log region below π/(4ρ_max), 16 nodes per J0 period, graded panels at the
  skin and layer scales) and raises `SommerfeldResolutionWarning` instead of
  truncating; entries agree with an independent quadrature to 1.2e-5 (R′ part)
  and 2.7e-6 (X′ part) for pairs 50–400 m apart. Golden cases:
  `cond_galvanic_neumann_sommerfeld`, `defaults_mixed_cluster_inductive`
  (≤ 1.8e-9), `cond_isolated_pen_sommerfeld_two_layer` (≤ 1.9e-5). Test:
  `tests/test_sommerfeld_accuracy.py::test_distant_pair_entry_matches_independent_quadrature`.
- ⚠ **Sommerfeld kernel exact for n ≥ 3 layers and below the first interface
  (10.35):** the buried kernel used the single air-image form outside the
  two-layer/layer-1 case; a three-layer earth with ρ₂ = ρ₃ gave +17 … +25 % X′
  against the identical two-layer earth. The kernel now uses generalised
  reflection coefficients (any number of layers, any layer) and the transmitted
  amplitude for pairs in different media. Affects `sommerfeld` worlds on
  `MultiLayerSoil` and conductors deeper than `h_1`. Golden cases: none. Tests:
  `tests/test_sommerfeld_accuracy.py::test_three_layer_duplicate_equals_two_layer`,
  `tests/test_sommerfeld_accuracy.py::test_conductor_below_first_interface`.
- ⚠ **`earth_return_correction_layered` had the wrong sign (10.46, F50):** the
  reflected term was +8.739e-3 + 6.484e-3j instead of −6.346e-3 − 4.580e-3j
  (earth 100/10 Ω·m, h₁ = 3 m, 1 kHz, z = z′ = 0.6 m); it now uses the exact
  layered kernel. Test:
  `tests/test_sommerfeld_accuracy.py::test_layered_point_kernel_has_the_exact_sign`.
- ⚠ **Exact removal of the additive image in the Sommerfeld correction
  (10.46):** the image was removed as −1/R′ integrated by the 16×16 outer rule
  with the direct orientation, so vertical segments kept twice the image in the
  σ → 0 limit and 50 m segments at 0.7 m lost 2 % of it. The correction now
  subtracts the very image block the inductance matrix added. Golden cases: see
  the λ-grid entry. Test:
  `tests/test_sommerfeld_accuracy.py::test_transparent_earth_cancels_the_additive_mirror`.
- **Segments crossing the soil surface (10.46, F52):** an inductive segment
  whose end points lie on both sides of z = 0 was dispatched by its midpoint
  (`World.solve` solved silently) or rejected as "lying in the soil-surface
  plane". With an image convention it now raises a `ValueError` that says the
  segment crosses the surface (its image intersects it); with `image="none"` the
  Sommerfeld pair integrals split such segments at the surface and at layer
  interfaces. Tests:
  `tests/test_sommerfeld_accuracy.py::test_surface_crossing_segment_is_rejected_with_the_right_message`,
  `tests/test_sommerfeld_accuracy.py::test_world_solve_rejects_surface_crossing_inductive_conductor`.
- **Carson's Q asymptote had the wrong sign of its a⁻⁷ term (10.46):** +45, not
  −45; together with the boundary move the large-a regime is continuous to
  1e-7 (it jumped by up to 9.3e-4 at a = 5). Tests:
  `tests/test_carson_pq.py::test_q_asymptote_a7_term_has_positive_sign`,
  `tests/test_carson_pq.py::test_large_a_regime_is_continuous`.
- **Conductor input validation (10.43, 10.67):** `cross_section <= 0` or
  non-finite, non-finite `resistivity` / `wire_radius` / end points raise a
  `ValueError` naming the parameter (they failed late with `ZeroDivisionError`
  or `LinAlgError`, or gave an active element); a cross section above 1e-2 m²
  warns with `CrossSectionUnitWarning` (`cross_section=50` meant as mm² was a
  silent 6.7 nΩ short); an ideal conductor with `discretize_segment_length`
  raises (it was `LinAlgError: Singular matrix` at solve time). Tests:
  `tests/test_conductor_validation.py::test_nonpositive_or_nonfinite_cross_section_raises`,
  `tests/test_conductor_validation.py::test_cross_section_in_mm2_warns`,
  `tests/test_conductor_validation.py::test_ideal_distributed_conductor_raises_with_parameter_name`.
- **Documentation in the coupling package:** English prose in
  `coupling/layered_green.py` (10.95); the Carson layered-soil warning points to
  `earth_inductive_model="sommerfeld"` (it named `backend='mom_sommerfeld'`);
  the internal-inductance statement of `references.earth_return` ("external
  inductance only") is corrected. Test:
  `tests/test_coupling_conventions.py::test_layered_green_prose_is_english`.

## Stream `core2a`: Two-terminal sources, junctions and diagnostics

### Added

- **Junction nodes (10.38, prerequisite of 10.17):** `groundfield.api.create_junction(world, position=..., name=None)`
  creates a leakage-free network node (`groundfield.geometry.junction.Junction`,
  `World.junctions`, `World.add_junction` / `get_junction`). Conductors start or
  end at junctions (`create_conductor` accepts junctions and their names) and
  sources attach to or return at them; a junction has no soil contact, takes
  part in Kirchhoff's current law and reports its potential per frequency in
  `FieldResult.metadata["junction_potentials"]`. Electrodes and junctions share
  one node namespace. A junction subnetwork without soil contact (no path to an
  electrode or a galvanic conductor) is rejected before the solve. Measurement
  leads that end on junctions at the instrument leave the soil problem exactly
  unchanged (fall-of-potential Z_meas equal to the lead-free value to 5.4e-16),
  where a probe lead ending on the station short-circuits the voltmeter
  (0.46 % of the true value); a junction leaks nothing where a 5 cm joint rod
  leaked 0.35 % of the test current in the test world. Worlds without junctions
  serialise exactly as before. Supported by the image family
  (`BackendCapabilities.junctions`); other backends reject such worlds with
  `BackendCapabilityError`. Tests: `tests/test_junction_nodes.py`.
- **`World.check()` (10.66):** validates source terminals and conductor ends
  (with the closest names), the excitation, voltage sources and short-circuited
  two-terminal sources; called by every solve. Test:
  `tests/test_source_validation.py`.
- **Aggregated diagnostics (10.70):** `groundfield.diagnostics.diagnose(world,
  engine=None, *, result=None, proximity=True)` returns a `DiagnosticsReport` of
  `Diagnostic` findings (category, severity, subjects, message, data) for thin
  wires, resolution, segment budget, close or interpenetrating leakage wires
  (`groundfield.conductors.find_close_wires`, 10.44), inert settings (conductors
  not attached at both ends, the deprecated shell registry, non-default boundary
  conditions or permittivity) and, with a result, series truncation. `str()`
  prints one line per cause, `.warn()` emits one `DiagnosticsWarning`. Close
  wires are a warning with the point kernel (two bonded 3 m rods 2 mm apart
  solve to 5.7 times the one-rod impedance) and information with the exact line
  kernel (0.998 times). Test: `tests/test_diagnostics_report.py`.
- **`diagnostics.check_segment_budget` and `ResourceLimitError` (10.06):** the
  hard segment limit of the solver pipeline (see *Fixed*).

### Changed

- ⚠ **`Source.return_to` is honoured by the image family (10.16, F01/F30).**
  `image`, `image_2layer` and `image_nlayer` inject +I at `attached_to` and -I at
  `return_to` through one shared injection step; the leakage currents of a world
  of two-terminal sources sum to zero and `input_impedance()` is
  (φ_a − φ_r)/I. Up to 0.15.0 the auxiliary electrode of a simulated
  fall-of-potential measurement carried 0 A and the station potential did not
  depend on its distance (13.107017 V for d_aux = 20 … 1600 m in the test
  world); now 12.310 V at 20 m rising to 13.097 V at 1600 m, and the probe at
  61.8 % reads the remote impedance to −1.6e-3 (20 m), −3.5e-6 (200 m), as
  Tagg's rule says (0.15.0: −9.9 % and −0.98 %). Who is affected: every world
  whose sources set `return_to` — the measurement worlds of
  `TnNetworkGenerator` and `LvNetworkLayout`. The generators still wire the
  feed lead from the station straight to the auxiliary electrode, so that lead
  now closes the loop metallically (station potential 1.5008 V → 4.15e-3 V
  in the golden case); they are re-wired through junctions in the generator
  work (10.17). `LvNetworkLayout` still adds its 180° counter-source until then,
  which now counts the return current twice; the solve warns about such a
  counter-source. Golden case: `gen_tn_measurement_overhead_leads`.
- ⚠ **Source-free clusters float in the image family (10.29, image half).** A
  bonded group of passive electrodes and a passive network joined by finite
  conductors now enter the nodal system with zero net impressed current instead
  of being dropped and read back as the incident field: one potential per
  cluster, equalising currents, and members of an isolated floating cluster
  report currents that sum to exactly zero (cluster impedance `nan`). Passive
  ring + rod bonded pair beside a driven rod: 2.5735 V / 1.8055 V and 0 A
  became one 2.4119 V with ±14.3 mA (`mom`: 2.4189 V, ±13.1 mA). A lone
  passive electrode is unchanged (exactly equivalent under the uniform-current
  ansatz), so worlds without bonded passive groups or passive networks are
  bit-identical. Who is affected: probe and auxiliary electrode groups,
  unbonded neighbours with several electrodes, passive measurement leads.
  Golden case: `cond_coupled_leads_open_circuit` (the grounded measurement lead
  beside the source lead: φ(m1) +42.5 %, driven side −0.89 % / −3.8 %).
- **One warning per cause instead of per electrode (10.70):**
  `check_segment_resolution` aggregates its findings (docs example 07: 81
  resolution warnings became 9, one per solve); the `return_to` capability
  warning disappears for the image family.
- **`expected_segments` follows the solver's discretisation (10.79)**, including
  wires split at soil-layer interfaces (a 3 m rod across h₁ = 1.3 m: 7 segments,
  the 0.15.0 prediction said 6). Test:
  `tests/test_diagnostics_report.py::test_expected_segments_is_exact_for_rods_split_at_the_interface`.

### Fixed

- **Invalid source references solved silently (10.66).** A misspelled
  `attached_to` / `return_to`, a source on a conductor (documented as allowed,
  injected nothing), a source returning to its own node, a world without a
  current source, a source of magnitude 0, and a conductor end naming no node
  (the conductor dropped out of the network) now raise — at `create_source` /
  `World.add_source` / `World.add_conductor`, or when an edited or loaded world
  is solved — with the closest names. `magnitude` and `phase_deg` must be
  finite, also when assigned later. Tests: `tests/test_source_validation.py`.
- **`VoltageSource` solved to all-zero potentials (10.51).** Solving a world with
  a voltage source raised `NotImplementedError` in the release preparation,
  naming the alternative (a 1 A current source scaled by U / Z_in); voltage
  sources are solved since 2026.11 (stream `w2src`, audit W2A-11). Test:
  `tests/test_source_validation.py::test_voltage_source_solves_and_fixes_the_feeding_potential`.
- **`HARD_LIMIT` only warned (10.06).** A world whose predicted segment count
  exceeds `groundfield.diagnostics.HARD_LIMIT` (20 000, read at solve time) is
  rejected with `ResourceLimitError` before anything of the problem size is
  allocated, also in `solve_mutual_*`: a 5 m ring entered in millimetres
  (62 832 segments) fails after 0.06 s with a 0.67 MiB allocation peak; the
  wave-1 code solved the full problem (173 s, 140 MiB peak on the review
  machine) and returned 0.0096 Ω. Test:
  `tests/test_diagnostics_report.py::test_world_beyond_the_hard_limit_is_rejected_before_any_allocation`.
- **Polyline and star electrodes skipped by the resolution check (F53, F54).**
  Listed as fixed in 0.15.0, but `diagnostics.py` was unchanged since 0.13.0;
  every electrode kind is now checked through `Electrode.min_feature_length()`.
  Test: `tests/test_diagnostics_report.py::test_resolution_check_covers_star_and_polyline`.
- **Close wires went unreported (10.44, diagnostics part):** see `diagnose` above.
  Test: `tests/test_diagnostics_report.py::test_close_and_interpenetrating_wires_are_reported`.

### Deprecated

- **`World.concrete_shell_corrections`** (the lumped ADR-0012 V1 registry,
  10.12–10.14): the solver consumes concrete shells only as the per-length
  coefficient `concrete_shell_coefficient_ohm_m` of the electrodes and never
  read this registry (nor did `solve_mutual_*`). Solving a world with entries
  emits a `DeprecationWarning` naming the coefficient; the field still loads
  from JSON and will be removed no earlier than 2027.5. Test:
  `tests/test_concrete_shell_registry.py`.

## Stream `core2b`: Default line kernel, cross-layer electrodes, concrete shells, mutual matrix

### Added

- **Concrete shell on every electrode kind (10.12–10.14):** rods, rings,
  meshes and grid meshes carry `concrete_shell_coefficient_ohm_m` like strips,
  polylines and stars. It is the one mechanism through which the solver sees
  a concrete encasement (ADR-0012): every segment of length Δs adds C/Δs to its
  reaction-matrix diagonal, in the image family and in `solve_mutual_matrix` /
  `solve_mutual_field`; with the uniform current per electrode this is exactly
  the lumped shell resistance R = C/L in series with the electrode. Conversion
  helpers in `groundfield.geometry.electrodes`:
  `concrete_shell_coefficient(rho_c, wire_radius, thickness)` (C =
  ρ_c/(2π)·ln((r_a + t)/r_a), 38.16 Ω·m for 100 Ω·m and 5 cm around a 5 mm
  wire), `shell_coefficient_from_resistance(electrode, R)` (C = R·L) and
  `shell_resistance_from_coefficient(electrode, C=None)` (R = C/L), which the
  generators use for `concrete_model="lumped"`. Rods, rings and meshes without a
  shell serialise exactly as before. Tests: `tests/test_concrete_shell_coefficient.py`.
- **`solve_mutual_matrix(..., offdiag="galerkin")` (10.34):** the off-diagonal
  entry Z_ij is the reaction T_iᵀ(K + S)T_j of the two leakage distributions —
  the potential of cluster i under excitation j, reciprocal to round-off
  without symmetrisation, and the matrix with which (Y_L + Z_G⁻¹)u = i
  reproduces the solve of the clusters joined by branches. `probe_points` is
  optional; `offdiag=None` (default) selects `"galerkin"` without probe points
  and `"probe"` with them, so existing calls keep their numbers apart from the
  diagonal (see *Changed*). On an AP1-like site (station ring + three rods,
  12 polygon foundations, ds = 2 m, five soils) the galerkin matrix reproduces
  a network solve with PEN-like branches to ≤ 2e-5, while the symmetrised probe
  matrix misses the input impedance by −0.15 … −0.88 % and single entries by up
  to 46 %; for Z_G as a network element `offdiag="galerkin"` is recommended.
  Tests: `tests/test_mutual_matrix.py::test_galerkin_offdiagonal_is_reciprocal`,
  `::test_galerkin_matrix_reproduces_the_joined_network`,
  `::test_offdiag_selection_and_validation`.

### Changed

- ⚠ **Exact line-source kernel is the default of the image family (10.28):**
  `Engine.kernel` defaults to `None`, the backend's preferred kernel: `"line"`
  for `image`, `image_2layer` and `image_nlayer` (also used by
  `FieldResult.potential` and `solve_mutual_*`), the backend's own kernel
  elsewhere; `metadata["kernel"]` records the kernel that ran and
  `kernel="point"` reproduces releases up to 0.15.0. The request is carried in
  the value, so engines rebuilt from JSON, `with_frequencies` or a provenance
  block run like the original. Numbers move for every default solve of the
  image family: rod 3 m +0.60 % (surface potential up to 1.9 %), two-layer ring
  +1.23 %, two rods joined by a galvanic conductor of 5 m segments: currents
  10.9 %, cluster impedance 8.4 %. Cost: a 16-house solve (N = 1920) takes 2.7
  times as long as with the point kernel with numba installed and 5.7 times
  without. In two-layer soil a straight 10 m strip stays within 0.05 % of its
  exact uniform-current resistance for ds = 0.125–2 m (point kernel: up to
  22 %). Golden cases: `defaults_rod_hom`, `defaults_two_layer_ring_auto`,
  `defaults_mixed_cluster_inductive`.
- ⚠ **Electrodes through a soil-layer interface (10.15, F02):** in
  `image_2layer` and `solve_mutual_*` an electrode that crosses h₁ (where the
  resistivity changes) has one uniform-current owner per layer, all in its
  cluster, so the current density may differ above and below the interface.
  A 12 m rod through h₁ = 5 m now equals the manual split into two bonded rods
  and lies +0.11 … +0.31 % from `mom` with the point kernel (+0.7 … +1.3 % with
  the line kernel, its offset from point-matching methods; 0.15.0: +566 % in
  1000/30 Ω·m, +148 % in 1000/100, +123 % in 100/1000). `TwoLayerSoil(ρ, ρ, h)` stays bit-identical to
  the homogeneous backend. Who is affected: rods through h₁ — not the AP1
  station rods (4 m from 0.6 m, h₁ ≥ 5 m). Golden cases: `img2_rod_crosslayer`
  (−46.5 %), `line_img2_rod_crosslayer` (−48.2 %), `mutual_three_rods_crosslayer`.
- ⚠ **Interface slivers and segment counts (10.32):** an interface cut that
  would leave a wire part shorter than the wire diameter
  (`geometry.electrodes.SLIVER_RADII = 2` radii) is dropped; the point kernel
  takes the Galerkin self term for segments shorter than 2.99 wire radii
  (identical above); segment counts ignore floating-point noise
  (`ceil(L/ds − 1e-9)`): a star arm of 3.000000000000001 m is cut into 6, not 7
  segments at ds = 0.5 m (≤ 6.9e-5 point kernel, ≤ 8.3e-8 line kernel).
  Golden cases: `img_mixed_cluster_hom`, `line_img_mixed_cluster_hom`.
- ⚠ **Post-solve series controls follow the engine (10.25):**
  `FieldResult.potential`, `solve_mutual_matrix` and `solve_mutual_field` sum
  the Tagg/Sunde series with the controls recorded with the solve
  (`metadata["series"]`, the engine defaults for results without the record)
  instead of a hard-coded 100 terms / 1e-6, and warn with
  `SeriesTruncationWarning` when the series is truncated. For K = −0.94 and
  `image_max_terms=2000` the potential 100 m from a ring moved by 0.8 %. The backend
  entry points `solve_image_2layer` / `solve_image_nlayer` take the engine's
  controls instead of private defaults 100 / 200 (10.74). Golden cases: none
  (the cases use the default 300 terms).
- ⚠ **`solve_mutual_matrix` diagonal (10.26):** `Z[j, j]` is the cluster
  impedance reduced like `FieldResult.cluster_impedance` — length-weighted
  over every member and every layer owner — instead of the unweighted mean over
  the alphabetically first member; equal to the per-excitation solve to
  ≤ 2.2e-16. 0.15.0 deviated by +0.150 % for an L-shaped footprint at
  ds = 2 m and by −3.8 / −9.1 % for a rod split at h₁. AP1-like site
  (ds = 2 m): house foundations up to 0.10 %, station 0.000 %. Golden cases:
  `mutual_*` (round-off; `mutual_three_rods_crosslayer` see 10.15).
- **`solve_mutual_matrix` / `solve_mutual_field` check the backend (10.34):**
  the engine's backend is resolved and checked like in `Engine.solve`
  (aliases, capability check, kernel); a backend outside the image family
  raises `BackendCapabilityError` (a `ValueError`) instead of returning image
  numbers (an 8-arm star: 7.854 Ω returned for a `mom` engine whose own solve
  gives 7.128 Ω). Test:
  `tests/test_mutual_matrix.py::test_non_image_backend_is_rejected`.
- ⚠ **`solve_mutual_matrix` / `solve_mutual_field`: the other clusters float
  (10.29):** each excitation is solved with the source-free clusters floating
  (zero net current, equalising currents in bonded groups and between the
  layer owners of a rod through h₁), as `Engine.solve` does since 10.29; the
  galerkin off-diagonal is then the exact potential of the passive cluster.
  AP1-like site: diagonal 3.3e-5, galerkin off-diagonal up to 0.09 %. Golden
  case: `mutual_foundations_distributed_concrete` (Z_G 5.2e-4, R 1.6 %).
- **`Engine.solve` validates the world (10.66):** it runs
  `World.check(require_source=False)` for every backend and entry point
  (dangling references, short-circuited sources, zero or non-finite
  magnitudes); a world without a source stays solvable through
  `Engine.solve` (all results zero), as before.
- **Cross-layer solves report no image series (10.50, F43):** a solve on the
  ADR-0007 path (an electrode reaches h₁) no longer reports the term count or
  the truncation of a series it never evaluates; `metadata` gains
  `series_used` and `series`. Test:
  `tests/test_series_controls.py::test_cross_layer_solve_reports_no_series`.

### Fixed

- **Degenerate electrode geometry (10.67):** zero-length strips, non-positive
  mesh or grid extents, non-finite coordinates or lengths and dimensions not
  longer than the wire radius (typically a radius typed in mm) are rejected
  when the electrode is built; up to 0.15.0 they failed in the solver as a
  thin-wire violation, an `OverflowError`, or solved to Z = nan. Test:
  `tests/test_electrode_validation.py`.
- **Reciprocity of the line kernel across the layer interface (10.28):** the
  cross-layer reaction matrix scaled the line corrections of the direct and
  surface-image terms with the source-layer resistivity while the layered
  correction kept the compensating terms in point form: 3.8 % asymmetric, and
  a two-port of a rod through h₁ and a ring non-reciprocal by 1.1–4.7 % (point
  kernel: 1e-8). Pairs in different layers now carry the transmitted
  resistivity 2ρ₁ρ₂/(ρ₁ + ρ₂), and the near-singular interface image of the
  layered correction is a Galerkin integral: symmetric to ~1e-8, and a rod
  through h₁ changes by less than 0.01 % per halving of ds (was up to 0.22 %).
  The interface image matters most for horizontal wires just below h₁: a ring
  (r = 3 m) 5 cm below the interface of 100 over 1000 Ω·m now gives
  55.6–55.7 Ω at every ds from 0.5 to 0.025 m; the point kernel (and `mom`,
  and the line kernel before) gave 29.7 Ω at ds = 0.5 m (−46 %), converging to
  54.7 Ω only at ds = 0.025 m. Keep ds well below 2|z − h₁| with
  `kernel="point"`. Tests: `tests/test_reciprocity.py`,
  `tests/test_cross_layer_line_kernel.py`.
- **`image_2layer` with a layer stack (10.48):** a `MultiLayerSoil` with two
  distinct layers raised a bare `TypeError` after passing the capability check;
  it now solves its canonical two-layer form, three distinct layers raise
  `BackendCapabilityError`. Test: `tests/test_layer_stack_dispatch.py`.
- **Interface slivers (10.32):** a rod whose tip lay 1 mm below h₁ raised a
  `ValueError`, one 9 mm below acted as a short in `mom` (196.6 → 144.2 →
  166.8 Ω for tips at 4.99 / 5.009 / 5.02 m). Test:
  `tests/test_interface_slivers.py`.
- **Junction terminals in `FieldResult.input_impedance` (10.38):** a source
  attached to or returning at a junction reads the junction potential. Test:
  `tests/test_junction_nodes.py::test_input_impedance_of_a_source_on_a_junction`.
- **Line-kernel cost with one long segment (10.28):** the near-pair search used
  the longest segment of the whole set as radius for every observer; one
  conductor discretised at 50 m turned a 16-house line solve from 0.8 s into
  29 s and 1.7 GiB. Each segment now searches with its own radius (identical
  results); the reaction corrections of 3000 short plus three 35 m segments
  peak below 40 MiB (940.9 MiB before). Test:
  `tests/test_line_kernel_near_pairs.py`.
- **Two-layer `mom` assembled its series matrix twice per solve:** the matrix
  is kept from the reaction-matrix assembly (bit-identical; 20 × 20 m grid,
  N = 400: 0.294 s → 0.154 s).

## Stream `eng2`: Shared point-matching pipeline, Sommerfeld tabulation, fem guards

### Added

- **`mom_sommerfeld` kernel tabulation report (10.61):**
  `metadata["kernel_tabulation"]` (`enabled`, `groups_tabulated`,
  `groups_pointwise`, `table_evaluations`, `max_check_error`) and the
  keyword `solve_mom_sommerfeld(..., tabulate=True)`; through the engine
  as `options={"mom_sommerfeld": {"tabulate": False}}` once the registry
  entry `MomSommerfeldOptions.tabulate` is in place. `converged` and
  `converged_per_frequency` (10.47) are kept. Test:
  `tests/test_mom_sommerfeld.py::test_solve_reports_the_kernel_tabulation`.
- **Benchmarks B12 and B13 (10.76, R2 §3.3):** B12 checks the depth-group
  table of `mom_sommerfeld` against the pointwise quadrature on a 40 m
  grid in 1000/30/200 Ω·m (bound 1e-6; measured 0 at ds = 2.5 m,
  6.9e-9 for single entries and 1.5e-10 for the grid impedance at
  ds = 1.25 m). B13 compares the per-metre series impedance of an
  isolated line (difference of a 1600 m and an 800 m line at 1 kHz,
  which cancels the end effects) with the Pollaczek conductor–earth loop
  impedance (reactance bound 5e-3, measured 4.1e-6; resistance bound
  2e-2, measured −1.04 %). B7 keeps the strict
  xfail of `image_2layer` until 10.15 and records that `fem` raises.
  `python -m tests.benchmarks.tolerances` prints the table.
- **`compare_engines` names engines that ran the same computation
  (10.76):** a deprecated name and its replacement with identical
  settings (`cim` and `image_2layer`, `bem` and `mom`) get a note that
  their agreement is not an independent cross-check. Test:
  `tests/test_engine_aliases.py::test_compare_engines_notes_that_an_alias_is_no_independent_check`.

### Changed

- **`mom`, `mom_sommerfeld` and `bem` run the shared solve pipeline
  (10.75, 10.59).** Discretisation, injection, topology and branch
  assembly are the steps of `groundfield.solver._pipeline`; the family
  keeps one shared point-matching solve and read-back, each backend
  contributes only its Green's function, and the reaction matrix is
  assembled directly instead of as the product of the kernel with the
  identity. The numbers are unchanged: 41 of 41 stored golden cases are
  bit-identical to the wave-2 base, and so are the numeric fields of 47
  further results except `bem` (≤ 3.5e-16, it now runs `mom`) and the
  tabulated three-layer case of 10.61 (1.1e-9). The direct
  assembly makes `mom_sommerfeld` on two-layer soil faster (40 m grid,
  ds = 2.5 m: 15.2 → 0.41 s).
- ⚠ **`mom` and `mom_sommerfeld` honour `Source.return_to` and junction
  nodes (10.75 with 10.16 / 10.38).** Because they now use the shared
  injection and topology steps, the return terminal of a two-terminal
  source draws the current back through its electrode (0.15.0: ignored,
  the current returned through remote earth), and
  `metadata["junction_potentials"]` reports junction nodes as the image
  family does. Numbers change for every world with a `return_to` source
  solved with these backends; no stored golden case contains one. Needs
  the registry flags `two_terminal_sources` and `junctions` for both
  backends. Tests:
  `tests/test_point_matching_pipeline.py::test_mom_family_honours_return_to`,
  `tests/test_point_matching_pipeline.py::test_mom_family_reports_the_junction_potential`.
- ⚠ **`mom_sommerfeld` tabulates the reflected kernel for n ≥ 3 layers
  (10.61).** Segment pairs of one depth class in the top layer use a
  cubic spline in ln s (64 nodes per decade) instead of one quadrature
  per pair; eight midpoints per table are checked against the quadrature
  (tolerance 1e-7 of |G|, one refinement), and a group whose check fails
  or whose nodes do not converge falls back to the quadrature with the
  same convergence report. Every other pair is evaluated once per
  distinct (s, |Δz|, z + z') instead of once per pair, bit-identically.
  Results move where a table is used, by ≤ 1.1e-9 relative for the
  impedances measured and ≤ 6.9e-9 for single matrix entries (40 m grid
  at ds = 1.25 m: 4.8703316096 → 4.8703316089 Ω). Three-layer 40 m grid:
  19.0 → 0.25 s (ds = 2.5 m, deduplication only, bit-identical) and
  66.6 → 0.30 s (ds = 1.25 m, tabulated); a village of eleven rings at
  irregular positions (N = 273) takes 1.25 s instead of 34.1 s with the
  deduplication alone (impedance 1.8e-11 apart). Golden cases: none of the
  stored ones moves; new case `mom_sommerfeld_three_layer_grid_tabulated`.
  Test:
  `tests/test_mom_sommerfeld.py::test_tabulated_reaction_matrix_is_within_1e_6_of_the_quadrature`.
- ⚠ **`fem` raises when an electrode reaches a layer its hemisphere does
  not (10.23).** An electrode below a layer interface that lies above
  the equivalent-hemisphere radius (a rod through or below the top
  layer) now raises `BackendCapabilityError` (a `ValueError` and
  `NotImplementedError`) before meshing. 0.15.0 returned +633 … +3138 %
  over a conductive lower layer (1000/30 Ω·m) and −10 … −85 % over a
  resistive one silently. Affected: rods in layered soil solved with
  `fem`; golden cases: none. Test:
  `tests/test_fem.py::test_fem_rejects_an_electrode_below_its_hemisphere`.
- **The `mom` family shares the depth advisories of the image family
  (10.75):** `ShallowSegmentWarning` for segments long against their
  burial depth.

### Fixed

- **`mom`, `mom_sommerfeld`, `bem` solved a segment that crosses the soil
  surface as if it were buried (10.75):** a 1 m rod from z = −0.1 m gave
  94.27 Ω; the shared depth guard now raises as it has for `image` since
  0.15.0. Test:
  `tests/test_point_matching_pipeline.py::test_mom_family_rejects_a_segment_partly_above_the_surface`.
- **`FemReductionWarning` missed interfaces above the electrode
  (10.23, audit F01):** the hemisphere spans the soil from the surface,
  but only interfaces between the electrode and the hemisphere radius
  were checked; a 10 m strip and a ring r = 5 m at 0.8 m below
  h₁ = 0.5 m (30/1000 Ω·m) came out −70 % and −47 % against `mom`
  without a warning. Every interface above the deeper of the electrode
  and its hemisphere is now reported, and so is an interface below both
  but within the electrode's horizontal half-extent (ring r = 10 m over
  h₁ = 5 m in 1000/30 Ω·m: −46 %, silent before). Where no rule applies,
  rings and strips measured in 1000/30, 100/1000 and 100/400 Ω·m stay
  within −7.4 … +4.3 % of `mom`. Tests:
  `tests/test_fem.py::test_fem_warns_when_the_electrode_lies_below_a_thin_top_layer`,
  `tests/test_fem.py::test_fem_warns_when_an_interface_lies_within_the_electrode_extent`.
- **`compare_engines` reported 34.3 Ω against 0 Ω as consistent
  (10.71, audit F03):** a quantity was skipped as "undefined for every
  engine" as soon as one engine returned zero. Only a quantity that is
  zero for every engine is skipped now; a zero for some engines fails
  with an infinite deviation and a note. Test:
  `tests/test_compare_engines.py::test_compare_engines_counts_a_zero_of_one_engine_as_failure`.

### Deprecated

- **Backend names `cim`, `bem` and `image_nlayer` (10.76).** They keep
  working, emit a `DeprecationWarning` naming the replacement and are
  removed no earlier than 2027.5: `cim` runs the image auto-dispatch
  (`image`, `image_2layer`; bit-identical on every world compared),
  `bem` runs `mom` (it assembled the identical point-matching system;
  results move by ≤ 3.5e-16), `image_nlayer` is the name of the
  auto-dispatch target of `image` for `MultiLayerSoil` and keeps its own
  solve function, so `Engine(backend="image")` stays silent. Three or
  more distinct layers raise and name `mom_sommerfeld`. As a pure alias,
  `cim` has the capabilities of `image`: concrete-shell electrodes and
  `kernel="line"`, which it rejected before, now run. The modules
  `groundfield.solver.cim` (with `fit_complex_images`) and
  `groundfield.solver.bem` stay importable; `solve_cim` and `solve_bem`
  warn and delegate. Their results carry the metadata of the backend
  that ran: `cim_fit_used`, `cim_n_images`, `cim_rms`, `reduces_to`,
  `n_layers`, `h` and `rhos` are gone, a direct call adds
  `dispatched_to`. `fem` is registered as experimental
  (`ExperimentalBackendWarning`, once per process). The registry entries
  are applied by the release coordinator. Tests:
  `tests/test_engine_aliases.py`.

### Removed

- Private helpers without a remaining caller:
  `groundfield.solver.mom._carson_skin_metadata`,
  `groundfield.solver.bem._build_Z_collocation`,
  `groundfield.solver.cim._cim_self_kernel_factory`; the re-imports of
  image-family helpers from `mom`, `mom_sommerfeld`, `bem` and `cim`.

## Stream `coup2`: Carson baseline, skin effect, touching pairs, earth-return assembly

### Added

- **Skin effect in the solve (10.37, opt-in):** a conductor with
  `resistance_model="skin"` now changes the branch impedance per frequency:
  each of its branches (lumped branches and the half stubs of galvanic chains
  included) adds `l_b · Conductor.skin_correction_per_length(f)` to its diagonal,
  so the resistance becomes R_ac(f) and the internal inductance L_int(f) (exact
  solid-round Bessel solution). A conductor without an inductance model gets the
  resistance part only and makes the world frequency dependent. Through the
  solver, a 150 mm² Al PEN has R_ac/R_dc = 1.0556 / 1.2308 / 1.5496 at
  250 / 550 / 1000 Hz. The default `resistance_model="dc"` changes nothing.
  Tests: `tests/test_inductive_solver_integration.py::test_skin_model_raises_the_pen_resistance_at_1khz`,
  `::test_skin_correction_of_an_inductive_conductor`,
  `::test_skin_correction_covers_the_half_stubs_of_a_galvanic_chain`,
  `::test_skin_effect_without_inductance_model_changes_the_resistance_only`.
- **Tabulated Sommerfeld assembly (10.60, opt-in):**
  `build_sommerfeld_correction_matrices(..., method="tabulated")` (and the
  single-frequency builder) groups the segments by depth and shares one table of
  the spectral integral S(ρ) per pair of depth groups and frequency (100 nodes,
  cubic spline in ln(ρ + z)); the outer quadrature of all segment pairs runs
  vectorised. Against a converged reference (200 instead of 24 ρ nodes per pair)
  it is closer than the per-pair rule: 1.7e-6 against 1.4e-5 of the largest
  entry on a 10-house street network (two-layer earth: 2.7e-6 against 3.6e-6;
  lines 1 cm below the surface: 8.0e-7 against 6.1e-5);
  the substation impedance agrees with the converged reference to 5e-10
  (per-pair rule 1.1e-8). On R8's W3 network a full solve with 11 frequencies
  takes 2.6 s instead of 27 s CPU (10 houses) and 7.8–9.1 s instead of 242 s
  (30 houses, 249 inductive branches; impedances equal to 3e-8). Blocks whose
  λ grid cannot be resolved (conductors in the surface plane across a large
  network) keep the per-pair rule. The solver selects it through the
  engine attribute `sommerfeld_assembly` (`"exact"` when absent); the default
  stays the per-pair assembly. Tests: `tests/test_sommerfeld_tabulated.py`.

### Changed

- ⚠ **Exact touching segment pairs by default (10.46):** chain neighbours and
  bends share an end point where the Neumann integrand is log-singular; the
  16×16 Gauss–Legendre rule is −0.3 % … +0.8 % off for such pairs. The solver and
  `build_inductance_matrix`, `neumann_mutual`, `perfect_mirror_self_pair_inductance`
  now use the closed forms (`touching="exact"`); `touching="quadrature"`
  reproduces 0.15.0. A straight conductor keeps the inductance of the unsplit
  wire for every discretisation. Every result with an inductive conductor moves:
  1.2e-6 (50 Hz) and 2.3e-5 (1 kHz) on the substation impedance of R8's 2-house
  street network (1.2e-4 at 1 kHz on the 30-house network, together with the
  lumped-branch fix below), ≤ 3.2e-6 in the golden cases
  `cond_galvanic_neumann_perfect_mirror`, `cond_galvanic_neumann_carson_series`,
  `cond_galvanic_neumann_sommerfeld`, `defaults_mixed_cluster_inductive`,
  `line_cond_galvanic_neumann`, `cond_coupled_leads_open_circuit`,
  `cond_isolated_pen_sommerfeld_two_layer`, `gen_tn_measurement_overhead_leads`,
  `fit_rho_f_standard_sweep`.
- **`sommerfeld` on the image-free baseline (10.46):** the solver assembles the
  Neumann matrix and the Sommerfeld correction against free space
  (`image="none"`) instead of adding the additive image and removing it again:
  exact, cheaper, the same numbers to round-off (≤ 4e-18 relative). Golden cases:
  the three `sommerfeld` cases, round-off only.
- **One earth-return assembly for all frequencies (10.60):** the solver builds
  the Sommerfeld or Carson correction for the engine's whole frequency set at
  once on first use (geometry, λ grid and Bessel tables once per segment pair);
  every frequency is bit-identical to a single-frequency assembly. Sommerfeld
  matrices of R8's W3 network for 11 frequencies: 7.4–8.2 s → 1.5–1.9 s CPU
  (2 houses), 137–141 s → 27–28 s (10 houses); Carson matrices of W3c
  (30 houses): 4.4 s → 2.4 s. Full solves (with the wave-1 gains): W3, 2 houses
  35.3 s in 0.15.0 → 2.1 s; W3, 30 houses about 3 600 s (review estimate) →
  242 s; W3c, 30 houses 16.4 s → 3.3 s. The correction keeps at most 256 MiB
  of matrices (longer sweeps are assembled in chunks). Golden cases: none.
  Tests: `tests/test_inductive_solver_integration.py::test_sommerfeld_pair_work_does_not_grow_with_the_number_of_frequencies`,
  `::test_branch_correction_is_built_once_for_all_frequencies`,
  `::test_branch_correction_memory_bound_keeps_the_numbers`.

### Fixed

- ⚠ **`carson_series` on the wrong image baseline (10.20, F32):** Carson's
  correction (ωμ₀/π)(P + jQ) is defined against the anti-parallel image, but the
  solver added it on top of the additive image of ADR-0004, which counts the
  image twice: the earth-return reactance of a 1 km PEN was 2.04× (50 Hz) and
  2.19× (1 kHz) Carson's value. The solver now builds the inductance matrix with
  `image="antiparallel"`; the 1 km PEN matches l·X′_Carson(GMR = a·e^(−1/4)) to
  7e-6 (50 Hz) and 4.2e-4 (1 kHz). On R8's W3c network (30 houses, PEN Neumann)
  the substation impedance at 1 kHz drops from 0.4288 + 0.3439j Ω to
  0.3964 + 0.2301j Ω (|Z| −16.6 %). Affects every `carson_series` result with an
  inductive conductor (all backends that share the inductive assembly). Golden
  case: `cond_galvanic_neumann_carson_series` (6.0e-3).
  Test: `tests/test_inductive_solver_integration.py::test_carson_series_reactance_matches_carson`.
- ⚠ **Carson's small-a series (10.46):** for a ≤ 0.25 only the leading terms
  were used (no a² term −(π/64)a²cos2θ of Q, constants rounded): P and Q were up
  to 2.4e-4 and 3.0e-3 off at a = 0.25, and the θ = 0 quadrature above it
  6.4e-6. The series is now summed to a¹² with the exact constants and the
  θ = 0 quadrature uses a graded composite rule; both are at round-off against
  mpmath and the regime boundary is continuous. Moves `carson_series` results
  (included in the bound of the entry above).
  Tests: `tests/test_carson_pq.py::test_small_a_regime_is_continuous`,
  `::test_small_a_series_matches_an_independent_quadrature`,
  `::test_theta_zero_quadrature_is_at_round_off`.
- ⚠ **Neumann inductance of lumped and single-segment conductors (10.24):** a
  conductor with `inductance_model="neumann"` that the topology keeps as one
  lumped branch (`discretize_segment_length=None`, or a single sub-segment) lost
  its inductance silently: a 300 m PEN was 0.0564 + 0j Ω at 1 kHz. It is now one
  inductive branch from start to end, coupled to every other inductive branch
  (0.2101 + 3.7839j Ω with `sommerfeld`; a 10 m chain gives 0.2096 + 3.7840j Ω;
  with `perfect_mirror` and `carson_series` the single branch equals the refined
  chain to 1e-9). Affects PEN service drops shorter than the PEN segment length
  (5 m by default) and lumped leads when the conductor is inductive; on R8's
  2-house street network the substation impedance moves by 1.3e-6 at 1 kHz.
  Golden cases: none. Tests:
  `tests/test_inductive_solver_integration.py::test_neumann_is_honoured_on_lumped_and_single_segment_conductors`,
  `::test_lumped_branch_equals_the_distributed_chain`,
  `::test_lumped_inductive_conductor_couples_to_a_parallel_chain`.
- **Conductors that leave the soil under `sommerfeld` (10.46, F52):** a
  conductor crossing z = 0, or lying in the surface plane, is solved with the
  default earth model (the pair integrals split it at the surface; a PEN in the
  plane agrees with one 1 mm below it to 1e-7); with `perfect_mirror` and
  `carson_series` its image is singular and the solve raises as before. Tests:
  `tests/test_inductive_solver_integration.py::test_sommerfeld_accepts_a_surface_crossing_conductor`,
  `tests/test_audit_guards.py::test_surface_pen_with_neumann_is_continuous_under_sommerfeld`.
- **Invalid escape sequences in `coupling.inductance` docstrings (audit A1-12,
  10.95):** three docstrings raised `SyntaxWarning` on compilation (an error under
  `python -W error`), and `\text` in the module docstring turned into TAB
  characters; they are raw strings now. Tests:
  `tests/test_coupling_conventions.py::test_modules_compile_without_invalid_escape_sequences`,
  `::test_docstrings_contain_no_control_characters`.
- **`segment_distance` with a zero-length segment (audit F04, 10.44):** a point
  segment was measured from the start of the other segment (5.099 m instead of
  1.0 m for a point 1 m beside the middle of a 10 m cable), so
  `find_close_wires` / `find_close_conductors` missed a point-like bond 2 cm from
  a cable; points now take the clamped orthogonal projection. Tests:
  `tests/test_conductor_proximity.py::test_segment_distance_with_point_segments`,
  `::test_point_like_bond_next_to_a_cable_is_reported`.

## Stream `fitx`: Touch-voltage envelope, formula renderer, passivity

### Security
- **Formula parser cost is bounded for sums, products and powers (10.01; audit 1, H1, H2, L1, L2):**
  formulas inside every documented limit could still cost minutes (times
  below: fresh process on a shared machine). A sum of fractions with
  distinct 15-digit prime denominators, `((b+2)/b)**64` per term, took 22 s
  (120 terms) and 105 s (222 terms) before it was rejected, because the
  bound was checked after SymPy had built the common denominator; the least
  common multiple of the fraction coefficients is now accumulated term by
  term before the sum is built, and the formula is rejected at the second
  term (0.05 s). Accepted products such as 332 logarithms (13 s) or 111
  vanishing log sums (55 s) re-evaluated the growing product with `evalf`
  after every factor (2 802 and 2 857 `evalf` calls); values now come from a
  numeric shadow evaluation (mpmath, 128 bits) computed once per node, and
  chains of 32 or more factors are multiplied pairwise (2.1 s and 0.8 s;
  176 and 110 `evalf` calls, all SymPy's own). `(exp(600)*f)**64` was
  accepted as `f**64*exp(38400)`: the power bound now includes the
  symbol-free factor of a product base, and
  every symbol-free sub-expression of the result is bounded
  (`f*exp(600)*exp(600)` is rejected). The documented exponent `(-3/2)`
  (Python reads `(-3)/2`) is accepted, so `format_impedance_formula` output
  such as `f**(-3/2)` parses. A deterministic budget, `MAX_WORK` = 200 000
  bookkeeping steps, caps every parse. Exported formulas parse to the same
  values as before. `tests/data/formula_corpus.json` gains 4 accepted and 6
  rejected audit strings for groundinsight's guard. Tests:
  `tests/test_formula_security.py::test_sum_of_fractions_with_distinct_denominators_is_rejected_quickly`,
  `tests/test_formula_security.py::test_long_product_chains_parse_in_bounded_time`,
  `tests/test_formula_security.py::test_symbol_free_factor_of_a_power_base_is_bounded`,
  `tests/test_formula_security.py::test_negative_rational_literal_exponents_are_accepted`.

### Added
- **`touch_voltage_envelope(..., interior=...)` (10.21):** samples the inside
  of closed loops, the analogue of the IEEE Std 80 mesh voltage (the worst
  touch voltage inside a mesh, near its centre). `interior=None` (default):
  the mesh centres of `mesh` / `grid_mesh` electrodes and the centre of a
  `ring`, i.e. loops in open ground; a closed `polyline`, the generators'
  building-outline foundation, is not sampled because the building stands
  inside. `interior=True` adds the point inside a closed polyline farthest
  from its wires (non-convex outlines stay inside); `interior=False` gives
  the offset curve only. The default cannot see buildings: pass
  `interior=False` for an axis-aligned generated foundation (`grid_mesh`) or
  a ring around a station housing. It replaces the `mesh_centres=` keyword
  added earlier in this release; `anchor="connection_point",
  interior=False` reproduces v0.15.0. Test:
  `tests/test_touch_voltage_envelope.py::test_ring_envelope_samples_the_ring_centre_by_default`.
- **Fit metadata in BusType documents (10.19, 10.27):** standard-form exports
  record all coefficients `k0` ... `k5`, `fit_quality.max_relative`, the
  `weights` and the fitted impedance `quantity` (`"input"`, `"cluster"`,
  `"electrode"`, or `null` for raw samples); vector-fit exports record
  `fit_quality.rms_relative` and `quantity`. `fit_quality_summary` names
  them. Test:
  `tests/test_io_groundinsight.py::test_fit_metadata_records_which_impedance_was_fitted`.
- **`MAX_WORK`** in `groundfield.io._formula` (see Security).

### Changed
- **One formula renderer (10.73):** `fit_to_sympy` keeps its signature but
  returns the partial fractions of the fit (one term per real pole or
  conjugate pole pair) instead of passing them through `sympy.simplify`,
  which merged them into one rational function: 6 of reviewer R5's 98 vector
  fits had a term above second order (`s**4` over `s**3` for a pole pair, a
  real pole and `L_inf`), now none. `fit_to_sympy`, `fit_to_sympy_standard`
  and the groundinsight exporter share one builder (one pole-pairing rule)
  and one rounding rule: every coefficient is rounded once to `decimals`
  significant digits (`decimals` must be a positive integer, also a NumPy
  integer, and every other value raises `ValueError`; above 17 counts as
  17). Accuracy is unchanged: worst deviation from `fit.evaluate` over R5's
  fits 4.0e-6 at 6 digits and 4.8e-12 at 12. Exported vector-fit formulas
  are unchanged character for character; exported standard-form formulas
  keep every coefficient and value, but a complex coefficient now keeps its
  own sign (`- f*(12.5592 - 1338.31*j)` is written `+ f*(-12.5592 +
  1338.31*j)`; 23 of R5's 31 standard fits). Test:
  `tests/test_fit_formula_renderer.py::test_fit_to_sympy_keeps_one_term_per_pole_or_pole_pair`.
- ⚠ **One passivity check (10.40):** the BusType export runs `check_passivity`
  (on the rounded formula it writes) instead of its own implementation, so
  both use one band, 0 Hz to twice the highest fitted frequency (the default
  of `check_passivity` was the highest fitted frequency: a fit with
  Re Z = -1.0 Ohm at 1.5 kHz, passive at every sample up to 1 kHz, passed
  `check_passivity` but failed the export), the pole-adapted grid with the
  midpoints between samples, an exact corner check for the standard form, a
  tolerance of 1e-9 max|Z|, and non-finite values fail (a pole at s = 0
  passed `check_passivity`). `PassivityReport.as_dict()` is the strict-JSON
  `metadata["passivity"]` block (`passed`, `min_re_Z_Ohm` or `null`,
  `f_at_min_Hz`, `rho_at_min_Ohm_m`, `f_checked_Hz`, `rho_checked_Ohm_m`,
  `n_points`); `PassivityReport` gains `n_points`. Affected: callers of
  `check_passivity` without `f_max`, whose verdict now covers twice the
  band (fits that passed may fail); impedances do not move. Golden cases:
  none. Test:
  `tests/test_passivity.py::test_violation_between_the_highest_sample_and_twice_it_is_found`.
- **`rho_at_fit` for layered soil (10.84):** the error message and the
  export docstrings state one convention, the resistivity of the upper
  layer, `rho_1` (the parameter of the standard form's rho_1 sweep), so
  vector-fit and standard-form BusTypes pair with the same groundinsight
  `Bus`. Test:
  `tests/test_io_groundinsight.py::test_rho_at_fit_convention_for_layered_soil_is_the_upper_layer`.
- **CSV frequency column `frequency_hz` (10.69):** the column the electrode and
  cluster tables gain in this release, and the header keys `frequency_hz` /
  `frequencies_hz`, use the lower-case unit suffix of the spec layer and of
  `FieldResult.to_dataframe`. The potential path keeps its v0.15.0 column
  `frequency_Hz`. Test:
  `tests/test_io_csv.py::test_new_frequency_column_uses_the_lower_case_unit_suffix`.
- **Footprint from the electrode geometry (10.79):** the touch-voltage
  footprint reads the wires from `Electrode.wire_polylines()` instead of a
  per-kind re-implementation (a ring keeps its exact circle). Foot positions
  are unchanged up to round-off (at most 1.4e-14 relative in the touch
  voltage, 2.3e-10 at UTM coordinates). Test:
  `tests/test_touch_voltage_footprint.py::test_mesh_centres_lie_midway_between_the_discretiser_wires`.

### Fixed
- ⚠ **Touch voltage of rotated foundations (10.21; audit 1, H4):** the
  generators build a foundation that is not axis-aligned from bonded wires
  `<name>`, `<name>_w1`, ... (10.39), and with 10.54 every OSM house
  foundation is rotated. The footprint came from the named edge wire only:
  for a 10 m x 10 m 2 x 2 mesh at 30 deg (100 Ohm m, 100 A) `touch_voltage`
  read 214.4 V on +x / -x and 93.9 V on +y (feet 1 m inside the building),
  31 of 72 envelope feet lay inside the foundation and no cell centre was
  sampled. The footprint is now every wire of the foundation (from any of its
  names, within its galvanic cluster) and the envelope samples the cells of
  the rotated grid: 161.5 V on every side (167.5 V at 0 deg; one uniform
  current per wire instead of per grid), no foot inside. Affected: touch
  voltages at rotated foundations, i.e. OSM-based layouts. Golden cases:
  none. Test:
  `tests/test_touch_voltage_footprint.py::test_touch_voltage_of_a_rotated_foundation_matches_the_unrotated_one`.
- **`touch_voltage_spectrum` counted a frequency twice (10.52; audit 1, L4):**
  a frequency carried by two results entered the RMS twice (701.6 V instead of
  496.1 V); it now raises `ValueError` naming the frequency. Test:
  `tests/test_safety.py::test_touch_voltage_spectrum_rejects_a_frequency_given_twice`.
- **Strict-JSON fit metadata (10.27):** hand-built standard fits wrote
  `max_relative` as NaN, which is not JSON; non-finite quality figures are
  now `null`. Test:
  `tests/test_io_groundinsight.py::test_metadata_of_hand_built_fits_is_strict_json`.
- ⚠ **Envelope of a ring includes its centre (10.21):** a consequence of
  `interior=None` above; where the centre dominates, the envelope maximum
  rises (ring of 10 m radius at 0.8 m, 100 Ohm m, 100 A: 139.4 V on the
  offset curve, 182.6 V in the centre; a 4 m ring's centre, 346.0 V
  against 360.2 V, does not).
  Affected: envelopes of ring electrodes. Golden cases: none. Test:
  `tests/test_touch_voltage_envelope.py::test_ring_envelope_samples_the_ring_centre_by_default`.

### Deprecated
- Nothing.

## Stream `fixsec`: Formula parser: resource bounds

### Security
- **Formula parser: SymPy no longer factorises numbers or rewrites `exp(c*log(x))` while parsing (10.01):**
  strings inside the allow-list grammar still kept SymPy busy for minutes.
  SymPy simplifies a fractional power of an exact number by factorising the
  number and keeps a radicand of up to `q - 1` times its digits:
  `(N)**(63/64)` with the 300-digit `N = 10**300 - 1` (also written
  `(f*N)**(63/64)`, `(-N*f*rho)**(63/64)` or `(N/8)**(63/64)`) keeps a
  radicand of about 18 800 digits and tests a 16 280-digit factor of it for
  primality, and `18**(10**30/(10**30 + 1))` needs `2**(10**30)`. It also
  rewrites `exp(c*log(x))` as `x**c`, whose exponent is no literal:
  `exp(log(N)*63/64)` factorises like the power, `exp(log(2*f)*10**12)`
  builds `2**(10**12)`. Of 19 such strings (four reported by the
  groundinsight stream), 15 did not finish within 20 s, three took 1.8 to
  2.6 s and one was accepted. Now no exact number is left under a fractional
  power: a fractional power (`x**(p/q)`, `sqrt`) of a symbol-free base is
  evaluated numerically, and the rational coefficient `c` of a product base
  is taken out as the float `|c|**(p/q)` (17 significant digits, principal
  branch); a powered sum with a float coefficient gets float coefficients
  throughout (SymPy 1.13, which the dependency range allows, takes the
  largest coefficient of such a sum out of the power: `(2.0*f + N)**(63/64)`
  did not finish within 10 s there); an exact number under a fractional
  power needs `q <= 64` and digits times `q <= 300` (`MAX_ROOT_DENOMINATOR`,
  `MAX_RADICAND_DIGITS` in `groundfield.io._formula`, the limits of
  groundinsight's guard); `log()` inside `exp()` is rejected before anything
  is built; `log` of a symbol-free argument other than a positive rational
  or a float is evaluated numerically (SymPy's `log(r + r*j)` and the square
  root of `r + r*j/3` factorised numbers of about 1 800 digits: 0.7 to 1.4 s
  for 900-digit fractions `r`). Each of the 19 strings is now rejected in
  at most 3 ms. Exported formulas parse to identical expressions (all 119
  accepted corpus formulas and 387 exporter outputs of reviewer R5's
  generator). `tests/data/formula_corpus.json` gains 18 resource-exhaustion
  strings and one out-of-grammar string, all rejected by groundinsight's
  guard as well; the corpus test now parses every rejected string first in a
  worker process with a 20 s deadline and a 1 GiB address-space limit, so a
  regression can neither hang the test run nor exhaust the machine. Tests:
  `tests/test_formula_security.py::test_every_rejected_formula_is_rejected_quickly_in_a_worker_process`,
  `tests/test_formula_security.py::test_fractional_powers_of_large_exact_numbers_are_rejected_before_sympy`,
  `tests/test_formula_security.py::test_fractional_powers_of_exact_numbers_are_evaluated_in_floating_point`,
  `tests/test_formula_security.py::test_log_inside_exp_is_rejected`,
  `tests/test_formula_security.py::test_sum_with_a_float_coefficient_keeps_no_exact_coefficient_under_a_root`,
  `tests/test_formula_security.py::test_complex_numbers_with_large_fractions_are_evaluated_quickly`.

### Added
- **`MAX_ROOT_DENOMINATOR`** (64) and **`MAX_RADICAND_DIGITS`** (300) in
  `groundfield.io._formula` (see Security).

### Changed
- ⚠ **Hand-written formulas with roots or logarithms of numbers (10.01):**
  `sqrt(2)`, `3**(1/3)`, `(2*pi*f)**(1/2)`, `log(-2)` or `log(2*pi)` now
  parse to floats of 17 significant digits instead of exact radicals and
  logarithms (`log` of a positive rational such as `log(2)` stays exact);
  evaluated values move by at most 3.6e-16 relative (2 units in the last
  place) in 17 sample formulas. Formulas that loaded before and are now
  rejected: `log` anywhere inside `exp` (e.g. `exp(log(f)*3)`, which SymPy
  turned into `f**3`), and fractional powers of exact numbers beyond the
  new limits, e.g. `2**(1/65)`, `(2*f)**(1/1000)`, a 5-digit integer under
  `**(63/64)` or the square root of a 151-digit integer; writing the number
  as a float (`2.0**(1/65)`) keeps such a formula valid. Affects
  hand-written BusType formulas only: groundfield's exporters write
  neither roots nor `exp`/`log` (checked for every exporter formula of the
  corpus). Golden cases: none (bit-identical). Test:
  `tests/test_formula_security.py::test_fractional_power_limits_for_exact_numbers`.

## Stream `gen2`: Measurement loop, one concrete-shell mechanism, PEN junctions

### Added
- **Reading of a simulated earth tester (10.17):**
  `groundfield.generators.measured_impedance(result, probe=None, source=None)`
  returns the three-pole reading Z_meas = (phi_E - phi_S) / I per frequency:
  E is the earthing system the measurement current is fed into (the source's
  `attached_to` cluster), S the voltage terminal (`meas_S` of a network built
  with a probe lead, else the probe electrode). The two-pole reading (E-H) is
  `result.input_impedance()`. New constants `MEASUREMENT_H_TERMINAL = "meas_H"`
  and `MEASUREMENT_S_TERMINAL = "meas_S"`; `PlannedMeasurement.terminals` and
  `NetworkPlan.all_junctions` carry the instrument terminals of a plan.
  Test: `tests/test_measurement_loop.py`.
- **`GroundingSystemSpec.plan_site(..., outline=)`:** shapes the resolved
  foundations after the footprint once the random draws are made
  (`"spec"` by default). Test:
  `tests/test_footprint_foundations.py::test_footprint_outline_does_not_shift_the_random_draws`.
- **`groundfield.generators.grounding.MIN_FOUNDATION_SIDE_M` (1 m):** shortest
  side of a foundation taken from a footprint's oriented bounding rectangle.

### Changed
- ⚠ **Measurement leads end on the instrument (10.17):** in
  `TnNetworkGenerator` measurement set-ups the feed lead runs from the
  junction `meas_H` to the auxiliary electrode and the probe lead from
  `meas_S` to the probe (both junctions at the substation position, at the
  lead height); the source drives the test current between the substation and
  `meas_H` (`return_to`; the auxiliary electrode itself without a feed lead).
  With both leads, `meas_S` lies the touching distance (sum of the two wire
  radii) beside `meas_H`, so an in-line probe lead runs beside the feed lead
  instead of through it. Numbers of every measurement world with leads change
  completely: the old probe lead bonded the probe to the substation (reading
  2.8e-4 instead of 1.6096 ohm in a 3-house network) and the old feed lead
  closed the loop metallically (99.93 % of the current never reached the
  soil). Golden cases: `gen_tn_measurement_overhead_leads`
  (auxiliary rod -1 A instead of +0.024 A, network potentials -1.8 %).
- ⚠ **One concrete-shell mechanism (10.12, 10.13, 10.14):** both
  `concrete_model` values put the Sunde coefficient
  C = rho_c / (2 pi) ln(r_b / r_a) and the radius r_b = r_a + t on every encased
  (perimeter) wire; nothing is written to `World.concrete_shell_corrections`
  any more and PEN service drops keep their geometric resistance. `"lumped"`
  keeps one grid-mesh owner for an axis-aligned ring foundation (exactly the
  lumped resistance C / L_perim), `"distributed"` uses one owner per wire;
  polygon, rotated and mesh foundations are the same for both. Mesh braces are
  not encased (bare radius, C = 0); this also moves `"distributed"` mesh
  foundations, whose braces had the inflated radius (4-house network, 2 x 2
  mesh: +2.6 / +7.8 / +12 % at rho_c = 150 / 1000 / 5000 Ohm m). The shell now reaches `solve_mutual_matrix`
  / `solve_mutual_field` (the AP1 Z_G path), `Engine.solve` of `build_at`
  worlds, the layout builder and substation / cabinet foundations. AP1-like
  site (rho_c = 100 Ohm m, 5 cm, 12 polygon foundations): house Z_G diagonal
  +42 / +19 / +2.8 % at rho_1 = 30 / 100 / 1000 Ohm m against v0.15.0, network
  input impedance +13.6 / +8.8 / +2.2 %; with rho_c = rho_soil the shell is
  transparent (0.0 %). Affected: every world with `concrete_rho_ohm_m` set.
  Golden cases: `mutual_foundations_lumped_concrete` (Z_G diagonal +42.2 %).
- ⚠ **PEN cable joints are junction nodes (10.38):** the layout builder's cable
  corners and tap points are leakage-free `Junction` nodes (reported in
  `result.metadata["junction_potentials"]`) instead of 5 cm driven rods, which
  leaked 0.24 to 0.53 % of the test current at full foundation penetration and
  3.7 % at 5 % penetration (substation potential +0.07 ... +0.33 % and +3.5 %).
  The joints are no electrodes any more. Golden cases: `gen_layout_builder`
  (+0.33 % potentials, +0.43 ... +0.57 % currents).
- ⚠ **`PenConfig.depth_m` is honoured:** every PEN section and service drop of
  `TnNetworkGenerator` runs horizontally at z = `depth_m` between its anchors
  (it ran straight between the electrode connection points, e.g. from the
  substation ring at 0.6 m to a cabinet rod head at 0 m); `depth_m` must be
  > 0. 10-house network: resistive PEN -2.7e-7, bare galvanic PEN -1.4 %,
  Neumann PEN at 1 kHz X 0.736 -> 0.167 ohm (the old sections ended in the
  surface plane, where the perfect-mirror image meets the wire). Golden cases:
  `gen_tn_default` (<= 3.5e-7).
- **Layout reading uses the solved test current:**
  `LvNetworkLayout.measured_grounding_impedance` and `verify_current_balance`
  take `source_magnitude_a=None` by default and read the current of the solved
  substation source (was 1 A unless repeated by the caller). Test:
  `tests/test_measurement_loop.py::test_layout_reading_uses_the_solved_source_current`.
- **Overpass response cap 64 -> 16 MiB:** `DEFAULT_MAX_RESPONSE_BYTES` also
  bounds the decoding memory (6x the body for building payloads, 25x for a
  body of empty objects); the largest AP1 payload is 3.0 MiB. Pass
  `max_response_bytes` for wider areas. Test:
  `tests/test_overpass_ingest.py::test_default_response_cap_bounds_the_decoded_size`.
- **Naming rule of the spec layer (10.69):** spec fields carry lower-case unit
  suffixes; the soil specs are the deliberate exception and mirror the
  literature symbols of their soil model (`rho`, `rho_1`, `rho_2`, `h_1`, `h`).
  Test: `tests/test_generator_naming.py::test_soil_specs_mirror_their_soil_models`.

### Fixed
- **Measurement leads short-circuited the instrument (10.17):** resistive leads
  now leave the reading unchanged (1.609633 ohm with and without leads; the
  feed lead adds its resistance to the two-pole reading only). Tests:
  `tests/test_measurement_loop.py::test_resistive_leads_do_not_change_the_reading`,
  `::test_feed_lead_adds_its_resistance_to_the_loop_impedance`,
  `::test_two_terminal_reading_is_the_superposition_of_one_terminal_solves`,
  `::test_in_line_leads_lie_side_by_side`.
- **Lumped concrete shell dropped outside TN service drops (10.12):** Test:
  `tests/test_concrete_one_mechanism.py::test_lumped_polygon_foundation_equals_the_distributed_one`,
  `::test_lumped_shell_resistivity_moves_the_mutual_matrix`.
- **Rods and braces leaked through the foundation's shell; substation and
  cabinet shells dropped (10.13):** lumped vs distributed +1.4 / +9.7 / +26.8 %
  (foundation + rod) and +9.6 / +56 / +148 % (mesh foundation) in v0.15.0, now
  within 2 % and identical. Tests:
  `tests/test_concrete_one_mechanism.py::test_mixed_site_lumped_equals_distributed`,
  `::test_mesh_braces_are_not_encased`, `::test_substation_and_cabinet_shells_are_consumed`.
- **Layout builder ignored the concrete shell (10.14, F08):** Test:
  `tests/test_concrete_one_mechanism.py::test_layout_builder_honours_the_shell`.
- **PEN junction rods leaked current (10.38):** Tests:
  `tests/test_pen_junctions.py::test_the_earthing_system_carries_the_whole_test_current`,
  `::test_a_resistive_cable_through_joints_equals_one_series_resistor`.
- **`PenConfig.depth_m` had no effect:** Test: `tests/test_pen_depth.py`.
- **Malformed Overpass payloads (10.05; audit M1):** elements with wrong types
  crashed the parser with `AttributeError` / `TypeError` after the payload had
  been cached (every later call crashed from the cache); NaN / 1e308 / lat 95
  vertices gave NaN-area footprints that passed `min_area_m2`. Every element
  and vertex is validated before caching and parsing (`OverpassError`), and a
  cached payload that fails the check is fetched again. Tests:
  `tests/test_overpass_ingest.py::test_malformed_elements_raise_overpass_error`,
  `::test_non_finite_or_out_of_range_vertices_are_rejected`,
  `::test_malformed_payload_is_neither_cached_nor_parsed`,
  `::test_malformed_cache_file_is_refetched`.
- **Footprint outline shifted the random draws (10.54; audit M2):** with a
  stochastic foundation size the `"ombr"` default skipped the size draw and
  moved optional rods to other houses (6 -> 8 rods). Test:
  `tests/test_footprint_foundations.py::test_footprint_outline_does_not_shift_the_random_draws`.
- **Degenerate footprints broke `to_world` (10.54; audit L5):** a 10 m x 5 mm
  outline is widened to a 1 m foundation with a warning. Test:
  `tests/test_footprint_foundations.py::test_a_degenerate_footprint_gets_a_minimum_foundation`.
- **Layouts pickled with v0.15.0 (10.69; audit L6):** `to_world` raised
  `AttributeError: 'cabinet_placements'`; `kvs_placements` is migrated on
  unpickling. Test:
  `tests/test_generator_naming.py::test_layout_pickled_with_v0150_still_builds`.
- **Inert permittivity of soil specs (10.51; audit A1-10):** a non-default
  `relative_permittivity` of a soil spec now warns with
  `InertConfigurationWarning` when the soil is built (it was dropped silently on
  the generator path). Test:
  `tests/test_soil_specs.py::test_inert_permittivity_warns_on_the_generator_path`.
- **Carriage return in a docstring (10.95):** the soil-spec module docstring
  held `\rho` in a non-raw string. Test:
  `tests/test_generator_docstrings.py::test_generator_docstrings_contain_no_control_characters`.

### Removed
- Unreleased names of the first 2026.11 consolidation: `PlannedBranch.concrete_shell_of`
  and `materialise.PEN_JUNCTION_ROD_LENGTH_M`.

## Stream `fix1`: Fixes from the first adversarial audit

### Added

- **`Engine.sommerfeld_assembly` (10.60):** `"exact"` (default) or
  `"tabulated"` assembly of the Sommerfeld earth-return correction of
  inductive conductors, a validated engine field (also a `create_engine`
  parameter, recorded in the provenance). The solver read it through
  `getattr` on a field that did not exist. Tests:
  `tests/test_sommerfeld_tabulated.py::test_the_assembly_is_a_validated_engine_field`,
  `::test_solve_with_an_engine_that_selects_the_tabulated_assembly`.

- **`create_conductor(resistance_model=...)` (10.37):** the opt-in skin
  effect (`"skin"`; default `"dc"`) is reachable through the factory; it was
  accepted by `Conductor(...)` only. Test:
  `tests/test_api_factories.py::test_create_conductor_reaches_the_skin_effect_model`.

### Changed

- **One error type when a backend cannot take the soil model (audit A1-08,
  F09):** the backend functions called directly raise
  `BackendCapabilityError` like `Engine.solve` does: `solve_image` with a
  layered soil, `solve_image_2layer` with a homogeneous soil,
  `solve_image_nlayer` and `solve_mom` with three or more distinct layers,
  and the concrete-shell guard of the mom family. `BackendCapabilityError`
  subclasses `ValueError` and `NotImplementedError` (the types of
  `image_nlayer`, `cim`, `bem` and the shell guard in 0.15.0), but not
  `TypeError`: code that caught `TypeError` from `image_2layer` (homogeneous
  soil or layer stack) or `mom` (three or more layers) must catch
  `BackendCapabilityError` or `ValueError`. `TypeError` stays for a
  `world.soil` that is no soil model. Tests:
  `tests/test_soil_capability_errors.py`,
  `tests/test_api_smoke.py::test_solve_image_directly_rejects_non_homogeneous`.
- **Engine copies and assignments are validated (audit A1-02, A1-04):**
  `Engine.model_copy(update=...)` validates the update like the constructor
  (unknown keys raise with a suggestion; only warnings that the update
  introduces are emitted), every field is validated on assignment
  (`engine.frequencies = []` raises), and entries put into `Engine.options`
  in place are validated when they are read (`backend_options`, `solve`).
  `Engine.options` and `BackendCapabilities.soils` / `conductors` /
  `kernels` are annotated with the accepted input types (mappings, sets),
  so the documented calls type-check (audit A1-06).
- **Soil keyword typos are type errors again (audit A1-09):** the deprecated
  soil names `resistivity` / `thickness` are translated in a model validator
  instead of `validation_alias=AliasChoices(...)`, which made the pydantic
  mypy plugin accept any keyword (`HomogeneousSoil(rhoo=100.0)` passed).
  Runtime behaviour is unchanged (deprecation warning, 0.15.0 JSON loads, a
  rejected value is reported under the name it was given); type checkers now
  also report the deprecated keywords. Test: `tests/test_typing_contract.py`.
- **W605 joins the blocking lint subset (audit A1-12):** pre-commit and CI
  run `ruff check --select F,E9,B,W605`; an invalid escape sequence (a
  `SyntaxWarning` today, a `SyntaxError` under `python -W error` and in a
  future Python) no longer passes. The 35 findings of the blocking subset in
  `tests/` were cleared outside the generator / geo test modules.
- **Release tool checks the remote (audit A1-11, 10.09):** before anything
  is written, `scripts/release.py` reads the upstream branch and the remote
  tags with `git ls-remote` (no fetch, `--dry-run` changes nothing): a
  clone that is behind or has diverged from its upstream is refused, remote
  tags count for the `MICRO` of `calver` and for "the tag exists", an
  unreadable remote or a branch without upstream (in a repository with a
  remote) aborts; a repository without any remote is released with a note.
  A stale clone used to create a second, different `v2026.11.0` that only
  the push rejected. Tests: `tests/test_release_script.py`.
- **`solve_mutual_matrix` / `solve_mutual_field` assemble the inductive branch
  correction for the solved frequency only (10.60):** an engine with four
  frequencies built the correction four times per call, now once; results
  are bit-identical. Test:
  `tests/test_mutual_matrix.py::test_one_call_assembles_the_branch_correction_for_its_frequency_only`.

### Fixed

- ⚠ **A galvanic conductor with one sub-piece leaks (10.24, galvanic half):**
  a distributed conductor with `coupling_to_soil="galvanic"` whose
  `discretize_segment_length` is at least its length fell back to the
  leakage-free lumped branch. Two 2 m rods 30 m apart joined by a bare
  50 mm² conductor in 100 Ω·m: input impedance 22.74 Ω with one sub-piece
  against 5.54 Ω with two (line kernel; point kernel 22.36 Ω against
  6.91 Ω); now 5.54 Ω (point kernel 8.59 Ω, the coarsest member of its
  convergence sequence towards 5.48 Ω). The sub-piece is one midpoint
  pseudo-node between two R/2 stubs and also carries its inductance with
  `inductance_model="neumann"`. Affects only galvanic distributed conductors
  shorter than `discretize_segment_length` (no golden case; isolated ones
  keep the lumped branch). A single sub-piece is as accurate as the kernel
  is for a long segment: with the line kernel it is converged in homogeneous
  soil and in two-layer soil without an electrode through the interface
  (12.134 Ω against 12.136 Ω with 32 sub-pieces, 300 over 50 Ω·m,
  h₁ = 2 m); with an electrode through the interface the cross-layer
  correction needs sub-pieces of the order of h₁ (one: 7.18 Ω, four:
  9.41 Ω, 32: 9.50 Ω). Tests:
  `tests/test_conductors_distributed.py::test_a_single_segment_galvanic_conductor_leaks`,
  `::test_a_single_segment_galvanic_conductor_keeps_its_inductance`.
- **Deprecated flat names in `Engine.model_copy` (audit A1-02):**
  `engine.model_copy(update={"sommerfeld_epsrel": 1e-2})` was silently
  ignored (the copy ran with `1e-7`; the name is a deprecated property that
  shadowed the stored value). The flat `sommerfeld_*` names are now mapped
  onto `options["mom_sommerfeld"]` with a `DeprecationWarning`, also back to
  a default value. Tests: `tests/test_engine_validation.py`.
- **Engine assignments (audit A1-04, engine part of 10.67):**
  `engine.options = {"fem": {...}}` stored a raw dict and the solve failed
  with an `AttributeError` (a misspelled option was never named);
  `engine.frequencies = []` returned an empty result and `[nan]` solved.
  Tests: `tests/test_engine_validation.py`.
- **Result files honour the umask (audit A1-05):** `save_result` wrote mode
  `0600` (`tempfile.mkstemp`); new files now get `0666` reduced by the umask
  (`0644` under `022`), an overwritten file keeps its mode. Tests:
  `tests/test_result_persistence.py::test_result_file_mode_follows_the_umask`,
  `::test_overwriting_keeps_the_mode_of_the_replaced_file`.
- **`ShallowSegmentWarning` follows the kernel (audit F08):** the advisory
  quantifies the bias of the point-form self image of a segment longer than
  four times its depth. It was also emitted for the exact line kernel (the
  image family's default), which has no such bias: a 10 m strip at 0.1 m
  depth gives 17.37863 Ω with 2 m segments and 17.37859 Ω with 0.25 m ones
  (point kernel 21.57 Ω and 17.23 Ω). It is now emitted for the point kernel
  and the `mom` family only, names the line kernel as the remedy and no
  longer claims that the `mom` family stays silent. Tests:
  `tests/test_shallow_segment_advisory.py`.
- **Stale docstrings (audit A1-07):** `create_engine` (backend-specific
  settings such as the `mom_sommerfeld` accuracy controls go into `options`,
  not the deprecated flat fields), `groundfield.solver` (the registered
  backends instead of "reserved" MoM / FEM), the engine's family overview
  (`cim` / `bem` aliases, `fem` experimental), `Engine.earth_inductive_model`
  (`carson_series` on the anti-parallel image, `sommerfeld` on the image-free
  matrix), the consumers of the series controls, and the branch block (skin
  term).
- **Control characters in docstrings (10.95):** `\rho` in a non-raw
  docstring of the two-layer solver started with a carriage return (a valid
  escape, invisible to W605). Test: `tests/test_source_escapes.py`.

### Security

- **Result files from untrusted sources (audit A1-01):** the `$ndarray`
  decoder of `FieldResult.metadata` accepted any numpy dtype string: a
  396-byte result file decoded to a 4 MiB unicode array, 413 bytes to an
  8 MiB sub-array, and object arrays loaded. It now reads only what the
  encoder writes (bool, integer, float16/32/64, complex64/128 and unicode
  dtypes), checks that the shape matches the number of items and that every
  item is a scalar of the dtype's kind before allocating, limits string
  arrays to 64 bytes per character of their items (plus 4 KiB), bounds the
  nesting depth (`MAX_METADATA_DEPTH = 100`) and raises `ValueError`
  (`pydantic.ValidationError` through `load_result`) naming the metadata key
  for every malformed tag; `load_result` reports non-UTF-8 and too deeply
  nested JSON as `ResultFormatError`. The encoder applies the same limits.
  Tests: `tests/test_result_metadata_codec.py`.

## Stream `fix2solver`: Fixes from the second adversarial audit: solver and engines

### Added

- **`groundfield.solver.PointKernelInterfaceWarning`:** warning category for
  a horizontal wire within 0.6 segment lengths of the layer interface on the
  point-form cross-layer path (see *Fixed*, point form of the cross-layer
  path).
- **`expected_segments(...)["nodal_unknowns"]` (10.06):** the size of the
  dense augmented nodal system (current owners, nodes and branches, every
  conductor sub-piece included), an upper bound computed in closed form and
  checked against `HARD_LIMIT` like the segment count. Test:
  `tests/test_diagnostics_report.py::test_nodal_unknowns_are_the_size_of_the_solved_system`.

### Changed

- **A world without a current source raises in every backend (10.66, audit
  A2S-05):** `Engine.solve` validated with `World.check(require_source=False)`,
  and an earlier entry of this release said such a world "stays solvable
  through `Engine.solve` (all results zero)". Only `fem` did; the image and
  `mom` families and `World.solve` raised. `Engine.solve` now runs the full
  check, so every backend and both entry points raise the same `ValueError`
  ("has no current source"), as the 10.66 concept lists. Test:
  `tests/test_source_validation.py::test_a_world_without_a_source_raises_in_every_backend`.
- ⚠ **`TwoLayerSoil(rho, rho, h)` is cut nowhere (10.48, audit A2S-07):**
  `image_2layer` (also reached through `image` and the auto dispatch) cut
  wires at `h` even when both layers have one resistivity, while `mom`,
  `mom_sommerfeld` and `image_nlayer` did not. Every backend now cuts where
  the resistivity changes only (the canonical stack), so such a soil solves
  exactly like homogeneous soil: a rod 0.3–6.3 m in `TwoLayerSoil(100, 100,
  2.15)` had 13 segments and sat 3.8e-4 (point kernel) and 3.3e-7 (line
  kernel) from the homogeneous result; now 12 segments, equal to 2.2e-16 and
  bit for bit. Who is affected: worlds with an equal-resistivity two-layer
  soil and a wire through `h` (not AP1: its station rods end above `h_1`).
  Golden cases: none. Test:
  `tests/test_diagnostics.py::test_equal_resistivity_two_layer_soil_is_the_homogeneous_soil`.

### Fixed

- ⚠ **Interface contact (10.32, audit A2S-01):** an interface cut that would
  leave a wire part shorter than the wire diameter was dropped, and the part
  joined its neighbour in the other layer, so the contact with that layer was
  ignored: a 4.5 m rod (a = 8 mm) in 1000/30 Ω·m read 195.0 Ω with its tip
  15.9 mm into the lower layer and 128.6 Ω at 16.1 mm (+52 %), in
  `image_2layer` and the `mom` family alike, and the same at the head of a
  rod. Every interface cut is now kept, however short the part beyond it;
  the thin-wire guard concerns the chosen segment length only. The Galerkin
  self term (it lost every digit below L/a = 1e-8) and the line kernel's
  parallel-pair integral (two parts of 1e-11 m: 0 instead of 125 1/m, the rod
  read 72.7 instead of 133.5 Ω) are evaluated without cancellation for such
  parts; every regular segment keeps its arithmetic bit for bit. The
  impedance is now continuous in the penetration depth (line kernel 133.5 →
  128.6 Ω from 10 pm to 16.1 mm, the same shape for the point kernel, `mom`
  and `mom_sommerfeld`); the step that remains is the onset of the contact
  itself between a tip that ends at the interface and one that reaches past
  it (195.9 → 133.5 Ω here). Who is affected: rods whose tip or head lies
  within 16 mm (two wire radii) of an interface: −34 % in this example.
  The constant `geometry.electrodes.SLIVER_RADII` introduced earlier in this
  release is gone. Golden cases: none. Tests:
  `tests/test_interface_slivers.py` (`::test_every_interface_cut_is_kept`,
  `::test_rod_resistance_is_smooth_in_the_tip_penetration`,
  `::test_rod_resistance_is_continuous_in_h1_at_the_former_sliver_threshold`,
  `::test_a_tip_twelve_millimetres_into_a_conductive_layer_is_felt`,
  `::test_a_head_just_inside_a_conductive_upper_layer_is_felt`,
  `::test_parallel_pair_integral_of_parts_much_shorter_than_their_distance`).
- ⚠ **Point form of the cross-layer path (10.28, ADR-0007):** with
  `kernel="point"` (and in the `mom` family, `mom_sommerfeld` with two layers
  and `solve_mutual_*` with the point kernel) the diagonal of the cross-layer
  reaction matrix averages the layered correction over a 3-point rule along
  the depth axis, ±0.387 L, whatever the segment's orientation. For a
  horizontal wire next to the interface the rule reached across it and
  returned wrong, even negative impedances (a rod through h₁ = 1.2 m with a
  20 m strip at 0.8 m, 300/50 Ω·m, ds = 2 m: −19.37 Ω against 9.62 Ω with
  the line kernel; v0.15.0: −14.24 Ω; −14 % … −552 % over the measured
  cases), or reached above the soil surface and raised a bare `ValueError`
  about negative depths. Such worlds now raise `BackendCapabilityError`
  naming the line kernel (or `image_2layer` for the `mom` family) and a
  segment length that fits; a horizontal wire within 0.6 segment lengths of
  h₁ warns with `PointKernelInterfaceWarning` (−3 % … −17 %). Rods (cut at
  h₁) and every valid configuration keep their numbers; the default line
  kernel is unaffected. Golden cases: none. Tests:
  `tests/test_cross_layer_point_form.py`,
  `tests/test_cross_layer_line_kernel.py::test_point_kernel_keeps_its_numbers_near_the_interface`.
- ⚠ **`fem` below a thin top layer (10.23, audit A2E-2):** the guard raised
  only when an interface lies between the hemisphere radius `a_eq` and the
  electrode's deepest point; a 3 m rod wholly below h₁ = 0.4 m < `a_eq`
  (1000/30 Ω·m) only warned and returned 31.1 Ω (`mom`: 10.5 Ω, +197 %). It
  now also raises when every electrode of a cluster lies below an interface
  between `a_eq/2` and `a_eq` (more than half of the hemisphere in a layer
  the electrodes do not touch; +51 … +130 % and −81 … −98 % measured for a
  3 m rod against `image_2layer`). Thinner top layers keep the warning
  (+16 … +46 %, −38 … −58 % for 100/1000 Ω·m), as do bonded rods through the
  top layer. Golden cases: none. Tests:
  `tests/test_fem.py::test_fem_rejects_a_rod_below_a_top_layer_holding_most_of_its_hemisphere`,
  `::test_fem_only_warns_below_a_thin_top_layer`,
  `::test_fem_guard_applies_to_a_bonded_cluster_wholly_below_the_interface`.
- **Resource guard (10.06, audit A2S-02):** `check_segment_budget` counted
  leakage segments only, so every sub-piece of an isolated distributed
  conductor (a branch and an interior node of the dense nodal system and a
  row of the inductance matrix) counted as zero: a 100 m PEN in 2000 pieces
  predicted 12 segments and took 407 MiB, in millimetre pieces the solve died
  with a `MemoryError`. The guard now also predicts the unknowns of the
  nodal system in closed form and raises `ResourceLimitError` above
  `HARD_LIMIT` before anything of that size is allocated (an inductive chain
  before the inductance assembly); `diagnose()` reports it. Tests:
  `tests/test_diagnostics_report.py::test_conductor_chain_beyond_the_limit_is_rejected`,
  `::test_a_chain_in_millimetre_pieces_is_rejected_before_allocation`,
  `::test_diagnose_reports_a_nodal_system_beyond_the_limit`.
- **Floating clusters in `solve_mutual_*` (10.29, audit A2S-03):** every
  excitation rebuilt the multi-port matrix of all floating clusters and
  factorised the full nodal system again, so the cost grew with the square
  of the number of bonded grounding nodes (256 ring-and-rod nodes: 199.9 s
  against 24.3 s before floating). The floating core is now assembled and
  factorised once with one right-hand side per anchor; an anchor outside the
  core enters through its coupling column. 256 nodes: 5.6 s; 100 rotated
  foundations of four bonded strips: 2.7 s (24.0 s); results equal to
  1e-12 (golden values bit-identical). Tests:
  `tests/test_mutual_matrix.py::test_shared_factorisation_equals_one_floating_solve_per_anchor`,
  `::test_one_core_assembly_and_factorisation_for_all_anchors`.
- **`expected_segments` for layer stacks (10.79, audit A2S-07):** the
  prediction cut wires at every declared interface, the backends at the
  interfaces of the canonical stack only: a rod 0.3–6.3 m in
  `MultiLayerSoil([100 Ω·m, h = 2.15 m], [100 Ω·m])` was predicted at 13
  segments and solved with 12. One rule now feeds both. Test:
  `tests/test_diagnostics.py::test_expected_segments_matches_the_solve_for_every_layer_stack`.
- **`fem` ignored the skin effect silently (10.37, audit A2C-4):** `"skin"` is
  a conductor feature of the backend registry; `fem` (lumped, DC) warns with
  `BackendCapabilityWarning` naming it, while `image` and `mom` raise the
  rod-to-rod voltage of a 300 m 150 mm² Al PEN by +54.9 % from 0 Hz to 1 kHz.
  A registered backend that lists its conductor features without `"skin"`
  warns too. Tests:
  `tests/test_backend_registry.py::test_a_backend_that_ignores_the_skin_model_says_so`,
  `::test_registered_backend_without_the_skin_feature_warns`.
- **Duplicate names in a loaded or edited world (10.66, audit A2S-06):**
  `World.check()` did not check that names are unique (only the `add_*`
  methods did): two electrodes of one name from JSON failed in the solver
  with a singular matrix, an electrode and a junction of one name were merged
  into one node. `check()` (run by every solve) now rejects a name used twice
  among the nodes, the conductors or the sources. Test:
  `tests/test_source_validation.py::test_check_reports_names_used_twice`.
- **`convergence_study` repeated show-once warnings (A1-02, audit A2S-08):**
  it copied the engine per step with the validating `Engine.model_copy`,
  whose warning capture resets every module's registry; a shallow strip
  showed the same `ShallowSegmentWarning` once per step. It now copies
  internally and validates the segment lengths itself (finite and strictly
  positive). Tests:
  `tests/test_convergence.py::test_convergence_study_shows_each_distinct_warning_once`,
  `::test_convergence_study_rejects_non_finite_segment_lengths`.

### Security

- **Crafted result files (A1-01, audit A2S-04):** a `$complex` metadata tag
  holding an integer beyond the float range (`{"$complex": [10**400, 0]}`,
  also inside `$tuple`, as a `$dict` key or an `$ndarray` item) escaped
  `load_result` as a bare `OverflowError` instead of the documented
  `ValueError`; it is now a `ValueError` naming the metadata key. Test:
  `tests/test_result_metadata_codec.py::test_a_complex_tag_beyond_the_float_range_is_a_value_error`.

## Stream `fix2coup`: Fixes from the second adversarial audit: coupling

### Fixed

- **Cost of the exact touching-pair rule (10.46, audit A2C-5):** the closed
  forms of coaxial segment pairs were evaluated with one Python call per
  pair, and on a straight conductor of M sub-segments all M(M−1)/2 pairs are
  coaxial: `build_inductance_matrix` took 2.4–3.2× (straight PEN, M = 100–800)
  and 7–26× (oblique PEN) the time of the 0.15.0 quadrature rule, and a solve
  of a straight 2 km PEN (400 branches, 2 frequencies) 2.2–2.6 s instead of
  1.0–1.3 s CPU. The coaxial pairs of a segment are now evaluated in one
  vectorised call — bit-identical (the 3-element dot products go through the
  same BLAS routine, the logarithms through `math.log`): 0.9–1.6× the
  quadrature time on straight chains, 2.0–3.7× on oblique ones (where the
  quadrature rule uses the cheap parallel closed form; 0.07–0.26 s for
  M = 200–400), 1.1–1.4 s for the 2 km PEN solve. No number changes (golden cases bit-identical). Tests:
  `tests/test_neumann_touching.py::test_scalar_closed_form_calls_grow_linearly`,
  `::test_touching_batch_is_bit_identical_with_the_scalar_closed_forms`,
  `::test_exact_rule_on_a_straight_chain_is_not_slower_than_twice_the_quadrature`
  (benchmark).
- ⚠ **`carson_series`: an overhead wire over a buried one (10.20, audit
  A2C-1):** the solver treated the buried wire of such a pair as an overhead
  wire at height |z| on the mirrored-endpoint image, whose image lands above
  ground next to the overhead wire. The mutual reactance per metre was
  35.9 % low for a lead 1 m above a PEN at 0.7 m depth (1 kHz, 100 Ω·m),
  13.5 % at 1 m horizontal offset, 3.4 % for a generator surface lead 0.1 m
  above ground, 10.3 % at 50 Hz. A perfect conductor screens the soil, so the
  anti-parallel baseline of a pair on opposite sides of the surface is now
  zero (`build_inductance_matrix(image="antiparallel")`), and its coupling is
  the field transmitted into the soil, new
  `coupling.carson.carson_mixed_mutual_correction` (Carson's J plus the
  bounded remainder of the buried wire's attenuation; Ametani et al. 2009):
  equal to an independent mpmath quadrature to 1e-9 and to the per-metre
  Sommerfeld kernel to 1e-3 at 1 kHz. Affects worlds with an overhead and a
  buried inductive conductor under `carson_series`; golden cases: none.
  Tests: `tests/test_carson_validity_domain.py::test_overhead_wire_over_buried_wire_matches_the_transmitted_field`,
  `::test_mixed_mutual_correction_matches_an_independent_quadrature`,
  `::test_mixed_pair_agrees_with_the_sommerfeld_model_per_metre`,
  `::test_antiparallel_baseline_of_a_pair_across_the_surface_is_zero`,
  `::test_every_image_convention_gives_one_impedance`.
- ⚠ **`carson_series` outside its validity domain (10.20, stream DOC-B):**
  oblique segment pairs got `sqrt(l_i l_j)·|cos φ|` times the per-metre value
  once per segment pair, which grows with the number of segments: a probe lead
  turning 8.1° against the feed lead changed the measured reactance 17.6-fold,
  two 1 km leads at 9° gave 20× the Sommerfeld coupling. Carson's per-metre
  tiling is now kept for near-parallel pairs of near-horizontal segments only;
  oblique pairs and every pair and self term of a segment inclined by more
  than 8.1° get the exact Sommerfeld correction of a homogeneous earth of the
  same conductivity (tabulated assembly). At the boundary the result steps
  from the per-metre tiling to the Sommerfeld value by the model error of the
  tiling there: 12 % for two 1 km leads 2 m apart at 1 kHz; in the DOC-B
  measurement setup (93 m leads, 50 Hz, short against the 711 m skin depth)
  Im Z_meas goes from 0.032 Ω (6.2°) to 0.018 Ω (8.6°), which is the
  `sommerfeld` value. Affects `carson_series` worlds with oblique, inclined or
  vertical inductive conductors; golden cases: none. Tests:
  `tests/test_carson_validity_domain.py::test_pairs_outside_the_domain_get_the_sommerfeld_correction`,
  `::test_oblique_coupling_does_not_grow_with_the_number_of_segments`,
  `::test_oblique_leads_couple_as_in_the_sommerfeld_model`.
- ⚠ **Wires lying side by side (10.46, streams GEN-2 / DOC-B):** two
  near-parallel segments whose axes stay closer than the sum of their radii
  over a common length (coincident leads, a service drop routed back along its
  PEN main) had no finite filament mutual inductance — the 16×16 rule stopped
  at its 1e-6 m floor: two coincident 50 m leads read 1.87 H of mutual
  inductance (+587j Ω at 50 Hz), and a two-house TN network whose drops ran
  back along the PEN main had a negative input reactance at 1 kHz
  (−0.633 Ω perfect_mirror, −0.643 Ω sommerfeld). Such pairs are now evaluated
  as touching wires at the axis distance r₁ + r₂ (in every image convention and
  in the image block), and `carson_series` gives two conductors overlapping on
  one axis the earth-return mutual of touching wires (it skipped them as
  "the same wire"). The two-house network now reads +0.244 / +0.233 / +0.250 Ω
  (perfect_mirror / sommerfeld / carson_series). Also moves the reading of the
  generator's in-line fall-of-potential setup, whose probe lead starts 1 cm
  beside the feed lead and ends on its axis: Im Z_meas at 50 Hz 0.063 → 0.054 Ω
  (sommerfeld). Golden cases: none. Tests:
  `tests/test_interpenetrating_wires.py`.
- ⚠ **`sommerfeld`: long segments (10.24, audit A2C-2):** the earth-return
  correction of a segment pair was one 16×16 Gauss–Legendre rule over the two
  segments, which cannot resolve exp(−γR) along a segment much longer than the
  skin depth — every lumped inductive conductor is one such segment. At 1 kHz
  in 100 Ω·m (skin depth 159 m) the earth-return resistance of a straight
  buried conductor was +1.7 % (1 km), +13.4 % (3 km), +34.9 % (5 km) against
  the converged chain (+0.2 / +0.6 / +1.7 % at 50 Hz), and a lumped 3 km PEN
  between two rods read +11 % series resistance, without a warning. A segment
  longer than a quarter skin depth of the most conductive layer now gets a
  composite outer rule (16 nodes per piece, at most 512 pieces, beyond that a
  `SommerfeldResolutionWarning`) and an S(ρ) table as dense as a short pair's
  (24 nodes over a km-long range left 1–2e-3); frequencies with equal piece
  counts share one pass, so every frequency still equals a single-frequency
  assembly bit for bit, and the tabulated assembly splits the same way. The
  lumped conductors now match their chains to ≤ 1e-4 (3 km at 1 kHz:
  2.78766 against 2.78773 Ω). Numbers move for every `sommerfeld` world with
  an inductive segment longer than a quarter skin depth — at 1 kHz 40 m
  (100 Ω·m), 22 m (30 Ω·m), 12.6 m (10 Ω·m); at 50 Hz 178 m (100 Ω·m) — by
  about 1e-4 just above that length up to tens of per cent for km-long lumped
  conductors; segments below it keep the historic rule bit for bit. Golden
  cases: none. Tests: `tests/test_sommerfeld_long_segments.py`.
- ⚠ **Neumann inductance of close, non-parallel segment pairs (10.46, found
  in audit round 2 while attacking the guard above):** pairs that miss the
  closed forms — nearly parallel wires close together whose directions
  differ by more than 1e-9 in cos (about 0.003°), crossing and skew pairs —
  were integrated with the 16×16 rule, which only resolves 1/r for pairs
  apart by about half their length: two 5 m segments with aligned ends and
  1e-5 … 1e-3 rad between them were +12 % at 10 cm and +200 … +250 % at
  1 cm, −19 % at 1 mm with offset ends; crossing pairs about −1 % (−12 % when
  crossing near an end at 2°), skew pairs 1 cm apart −2.3 … +0.5 %, a branch
  starting on the interior of another segment (T-junction without a shared
  vertex) −3.5 … −13 %. With `touching="exact"` (the default) such pairs
  are now integrated semi-analytically (inner integral in closed form, outer
  Gauss–Legendre panels graded towards the near-singular points; ≤ 5e-11
  against an adaptive reference); `touching="quadrature"` keeps the 0.15.0
  numbers. Moves every inductive result with such pairs — conductors from
  real (non-axis-parallel) geometries running close together, the
  generator's in-line probe lead (starts 1 cm beside the feed lead, ends on
  its axis): feed–probe mutual 92.8 → 91.7 µH, the value of touching wires
  over the overlap plus the end-to-end coupling (91.65 µH). Golden cases: none.
  Tests: `tests/test_neumann_close_pairs.py`.

## Stream `fix2sec`: Fixes from the second adversarial audit: formula parser

### Security
- **Formula parser: SymPy no longer evaluates the expression being built (10.01):**
  the second audit of the calendar release found two more evaluation paths
  that the parser's rules did not bound (after two rounds of rules against
  particular inputs): SymPy's `exp` asks whether the factors of its
  argument are comparable and expands powers of complex sums to find out,
  and `Pow` computes the real part and the argument of the inner base of
  `(b**e)**(p/q)`; 67- and 75-character formulas such as
  `exp(pi*(((pi/4 + j/2)**64 + pi/4 + j/2)**64 + pi/4 + j/2)**64/1e30)` and
  `(((((f/4 + j/2)**64/1000 + j/2)**64/1000 + j/2)**64/1000 + j/2)**63)**(1/2)`
  ran for more than a minute. The class is fixed instead of the examples:
  every node is now built unevaluated (`evaluate=False`), and every number
  is computed by the parser itself in bounded floating point (128 bits,
  every bound checked before the operation) and stored as a SymPy float;
  within one product the numeric factors form one coefficient, within one
  sum the numbers are added and like terms collected (so `1/(f - f)` and
  `exp(f - f + 1000)` stay rejected). Formulas within every limit that took
  2.2 s to 3.3 s to parse before now take at most 0.04 s, every corpus
  string keeps its verdict, and the time of a parse grows linearly with
  the length of the formula. The exponents of nested powers now multiply to
  at most 64 (`MAX_POW`), the bound groundinsight's guard applies; the
  audit's strings are rejected within 3 ms. `evaluate_spec` and the exporter's
  passivity check evaluate the parsed expression node by node with NumPy
  (new `formula_function`) instead of `sympy.lambdify`, whose printer
  evaluates sub-expressions: 6.8 s for a 1 439-character formula within
  every limit, now 2 ms. `tests/data/formula_corpus.json` gains 16
  resource-exhaustion and 2 out-of-grammar strings, all rejected by
  groundinsight's guard as well. Tests:
  `tests/test_formula_security.py::test_nested_powers_of_audit_2_are_rejected_quickly`,
  `tests/test_formula_security.py::test_formulas_within_every_limit_parse_and_evaluate_quickly`,
  `tests/test_formula_security.py::test_parsing_and_evaluation_never_let_sympy_evaluate`,
  `tests/test_formula_security.py::test_evaluate_spec_does_not_use_sympy_code_generation`,
  `tests/test_formula_security.py::test_terms_that_cancel_within_one_sum_are_collected`.
- **Formula parser: numeric literals outside the float range raise `FormulaError` (10.01):**
  `1e-` followed by 320 nines is a valid Python float literal that rounds
  to 0.0; the parser converted its exponent to a float and raised
  `OverflowError`, which leaked through `load_bustype_json`. Every literal,
  also in an exponent, is now bounded by its exact decimal value before
  any float is formed, and no arithmetic error leaves the parser as
  anything but `FormulaError`; a BusType document whose `voltage_level`
  or `schema_version` is outside the float range raises `ValueError`, and
  so does a crafted document that is nested too deeply for Python's JSON
  reader, is not a JSON object, lacks `name`, `system_type` or
  `voltage_level`, or has samples that are not numbers (it raised
  `RecursionError`, `AttributeError` or `KeyError`). Tests:
  `tests/test_formula_security.py::test_numeric_literals_outside_the_bounds_raise_formula_error`,
  `tests/test_formula_security.py::test_load_bustype_json_reports_a_huge_literal_as_evaluate_spec_error`,
  `tests/test_formula_security.py::test_bustype_header_numbers_outside_the_float_range_raise_value_error`,
  `tests/test_io_groundinsight.py::test_crafted_bustype_documents_raise_value_error`.

### Added
- **`formula_function(expr, symbols=("f", "rho"))`** in
  `groundfield.io._formula`: NumPy function of a parsed impedance formula,
  built from the expression tree without SymPy's code generation (real
  arithmetic for real values, as `lambdify` with NumPy; SymPy's principal
  branch for roots and logarithms of negative numbers). Test:
  `tests/test_formula_security.py::test_formula_function_keeps_the_real_limits_at_singular_points`.

### Changed
- ⚠ **Form of a parsed impedance formula (10.01):** `parse_impedance_formula`
  returns an unevaluated expression whose numbers are SymPy floats of 17
  significant digits: `2*pi*f*j/1000 + 403191/50000` parses to
  `(0.0062831853071795866*I)*f + 8.0638199999999998`, `log(2)` to
  `0.69314718055994529`; symbolic `exp`, `log` and powers keep the
  structure of the text. Code that compared parsed expressions with exact
  SymPy expressions must compare values instead. Evaluated values move by
  at most 4.1e-16 relative for the 103 exporter formulas of the corpus and
  3.3e-15 for the hand-written ones (closer to a 50-digit evaluation than
  before: 1.5e-16 instead of 3.4e-15 for the worst one). Formulas that
  loaded before and are now rejected: nested powers whose exponents
  multiply to more than 64, also through sums and function calls and for
  powers of numbers, e.g. `((f**8)**8)**2` or `(f*10**40)**2` (write
  `1e40`), and exponent literals below `1e-300` such as `f**1e-400`
  (which was read as `f**0`). `evaluate_spec` gives the principal-branch
  value where `lambdify` gave `nan`, e.g. `(f - 100)**0.5` below 100 Hz.
  groundfield's exporters write neither nested powers nor roots, so
  exported formulas are not affected. Golden cases: none (bit-identical
  to the base of the change). Tests:
  `tests/test_formula_security.py::test_exponents_of_nested_powers_multiply_to_at_most_max_pow`,
  `tests/test_formula_security.py::test_parsed_expression_is_unevaluated_with_numbers_as_floats`,
  `tests/test_formula_security.py::test_formula_function_takes_the_principal_branch_where_real_arithmetic_has_none`.

### Fixed
- **Strict JSON for unknown fit quality (10.19):** a fit whose rms error
  is unknown (NaN, e.g. a hand-built fit) exported `"rms_error_Ohm": NaN`,
  which strict JSON parsers reject, and a non-finite sample did the same;
  every non-finite number of a BusType document is now `null`,
  `save_bustype_json` serialises with `allow_nan=False`, the export refuses
  a `voltage_level` that is not finite, and `fit_quality_summary` reports
  an unknown rms error as `rms=unknown`. Tests:
  `tests/test_io_groundinsight.py::test_unknown_fit_quality_and_samples_are_written_as_null`,
  `tests/test_io_groundinsight.py::test_export_refuses_a_voltage_level_that_is_not_finite`.

## Stream `fix2gen`: Fixes from the second adversarial audit: generators and OSM ingest

### Added
- **Outline width of a footprint (audit A2G-06):** `BuildingFootprint.mean_width_m()`
  (twice the area over the perimeter, the width of a thin rectangle),
  `BuildingFootprint.is_degenerate()`, `groundfield.geo.mean_width` and
  `groundfield.geo.MIN_OUTLINE_WIDTH_M` (1 mm). Test:
  `tests/test_geo_footprint.py::test_mean_width_of_an_outline`.
- **`groundfield.geo.MAX_PAYLOAD_DEPTH` (16):** deepest container nesting of an
  accepted Overpass payload (an Overpass result has 7 levels). Test:
  `tests/test_overpass_ingest.py::test_payload_nested_deeper_than_an_overpass_result_is_rejected`.

### Changed
- ⚠ **Owners of generated mesh foundations (10.12, 10.39; audit A2G-04):** the
  image family solves one uniform leakage current per electrode (an owner). A
  mesh foundation (cross-braces) is now one owner per wire line at every
  orientation, bare or encased, for both `concrete_model` values; an
  axis-aligned bare one was one `grid_mesh` owner. One owner over perimeter and
  braces was +1.6 / +2.5 / +3.2 % high for 2 × 2 / 3 × 3 / 4 × 4 meshes against
  the equipotential reference (one owner per segment, equal to `mom`), one owner
  per wire is +0.3 / +0.5 / +0.6 %. A `"distributed"` ring foundation is one
  owner per side also without a shell (it was one owner when bare and
  axis-aligned). Numbers move: bare axis-aligned mesh foundations −1.2 % (10 m,
  2 × 2) to −2.5 % (30 m, 4 × 4); generated networks with the default mesh
  foundations −0.7 to −0.8 % in every node potential and up to 4.7 % in surface
  potentials; the largest touch voltage over the default house mesh −3.6 %
  (2.528 → 2.437 V per ampere, reference 2.402); bare distributed rings at most
  0.06 % up to aspect ratio 4 (0.24 % at 10). A `"lumped"` ring keeps one
  `grid_mesh` owner when axis-aligned and one owner per side when rotated
  (0.24 % apart at aspect ratio 10), and polygon outlines stay one closed wire:
  one closed polyline would drop the interior sample of the default
  touch-voltage envelope, which decides it for rings of 20 m and more. AP1
  (lumped polygon foundations) does not move. For a mesh foundation,
  `electrode_currents[<anchor>]` and `grounding_impedance(<anchor>)` now refer
  to its first wire, as for every rotated foundation: use `cluster_impedance`
  or the sum over `<name>` and `<name>_w<j>`. Golden cases: `gen_tn_default`,
  `gen_layout_builder`, `gen_tn_measurement_overhead_leads`.
- ⚠ **PEN cables of `TnNetworkGenerator` form a nearest-point tree (10.38;
  stream DOC-B):** every cable starts at the point of the network already built
  nearest to its far end, inside a cable at a leakage-free tap junction
  (`pen_tap_<k>`, a plan junction): cabinet cables in the order of their
  distance from the substation (collinear cabinets are chained), service drops
  from the cable path that feeds their cabinet or radial-trunk tap (or from a
  drop already routed for it, closer buildings first); a building whose
  connection point lies on a cable splits it and gets no drop. Service drops
  started at their cabinet before, 60 to 160 m back along the main in the
  default network, and now tap the main beside each house (5 m). Numbers move
  for every generated TN network: default network substation potential −2.1 %,
  substation leakage −3.1 to −4.8 %, house potentials −0.2 to +1.0 % (the
  houses are tied closer to the substation); measurement network −1.1 %. Main
  sections are named `<cable>_s<k>`. Golden cases: `gen_tn_default`,
  `gen_tn_measurement_overhead_leads`.
- **Overpass ids and levels:** element ids and member refs must be 64-bit
  integers (`OverpassError` otherwise); a non-finite or negative
  `building:levels` tag gives `levels=None`; outlines without area are skipped
  (see Fixed).

### Fixed
- **A shell that is electrically absent moved a mesh foundation (10.12; audit
  A2G-04):** with ρ_c = ρ_soil (or a 0.1 mm shell) the default 10 × 10 m mesh
  foundation lost 1.27 % of its impedance and its largest touch voltage on the
  offset curve dropped from 232.26 to 221.21 V (100 A), because the shell
  switched it from one owner to one per wire; a distributed 20 × 2 m ring moved
  by 0.24 %. Tests:
  `tests/test_foundation_owners.py::test_vanishing_shell_on_a_mesh_foundation_is_continuous`,
  `::test_vanishing_shell_on_a_distributed_ring_is_continuous`.
- **Mesh foundation impedance depended on the orientation (10.39; audit
  A2G-04):** the same bare mesh gave 4.79618 Ω at 0° and 4.73771 Ω at 30°
  (100 Ω·m, line kernel). Tests:
  `tests/test_foundation_owners.py::test_mesh_foundation_does_not_depend_on_its_orientation`,
  `::test_distributed_ring_does_not_depend_on_its_orientation`,
  `::test_mesh_foundation_is_close_to_the_equipotential_reference`.
- **Untrusted Overpass JSON raised other exceptions than `OverpassError` (10.05;
  audit A2G-05):** a body nested deeper than the decoder's limit (about 10 000
  levels) raised `RecursionError`, also from a cache file; an unknown key nested
  1 500 levels decoded, passed every check and crashed the cache write
  (`json.dump` recurses in Python); a 400-digit integer coordinate raised
  `OverflowError`; `"id": null` crashed the parser with `TypeError`. Tests:
  `tests/test_overpass_ingest.py::test_deeply_nested_body_raises_overpass_error`,
  `::test_payload_nested_deeper_than_an_overpass_result_is_rejected`,
  `::test_deeply_nested_cache_file_is_refetched`,
  `::test_vertex_with_a_huge_integer_coordinate_raises_overpass_error`,
  `::test_huge_integer_coordinate_in_the_json_text_raises_overpass_error`,
  `::test_ids_beyond_64_bits_are_rejected`,
  `::test_element_with_a_null_id_is_parsed_without_an_id`,
  `::test_non_finite_or_negative_levels_are_unknown`.
- **Outlines without area (10.05, 10.54; audit A2G-06):** `parse_overpass_payload`
  kept outlines of identical or collinear vertices with the default
  `min_area_m2=0` (although documented as skipped); their oriented bounding
  rectangle raised `IndexError`, so `LvNetworkLayout.to_world` and
  `TnNetworkGenerator` crashed instead of widening the foundation to
  `MIN_FOUNDATION_SIDE_M`. The parser now skips outlines narrower than 1 mm on
  average; the bounding rectangle of such an outline is its covering segment,
  which the generators widen with a warning. Tests:
  `tests/test_overpass_ingest.py::test_outline_without_area_is_skipped`,
  `::test_thin_but_real_outline_is_kept`,
  `tests/test_geo_footprint.py::test_ombr_of_a_degenerate_outline_is_its_segment`,
  `tests/test_footprint_foundations.py::test_a_footprint_without_area_gets_a_minimum_foundation`,
  `::test_tn_generator_widens_a_footprint_without_area`.
- **Generator objects pickled by v0.15.0 (10.69; audit A2G-07):** a pickled
  `TnNetworkConfig` unpickled and then failed in `build()` with `AttributeError`
  (renamed fields of soil specs, cabinets, PEN topology and the config itself).
  Pickles are now upgraded like JSON input: renamed fields and values, defaults
  for fields added since, and instances of the deprecated aliases (`KvsConfig`,
  `StarKvsTopology`, `KvsPlacement`, `OrtsnetzLayout`) load as the current
  classes, equal to the same object loaded from its v0.15.0 JSON. Tests:
  `tests/test_generator_pickles.py`.
- **Overlapping PEN cables made an inductive network non-passive (10.38; stream
  DOC-B):** in a 2-house network both service drops ran from the cabinet 180 m
  out back along the PEN main (Im Z_in = −0.633 Ω at 1 kHz with
  `inductance_model="neumann"`); now +0.072 / +0.055 / +0.051 Ω (perfect mirror /
  Sommerfeld / Carson series). Tests: `tests/test_pen_routing.py`.
- **Integer fields of a resolved config held floats (10.53; stream DOC-B):**
  `GeneratorConfig.sample` stored the float draw of a distribution in
  `building_counts` and `CabinetConfig.fixed_count`, and `model_dump_json`
  warned; a draw for an integer field is now `int(round(x))`, the value the
  generators build with. Test:
  `tests/test_generators_base.py::test_resolved_integer_fields_hold_integers`.

## Stream `fix3solver`: Fixes from the third adversarial audit: solver

### Added

- **`groundfield.solver.InterfaceContactWarning` and the `diagnose()` category
  `interface_contact` (10.32, audit A3S-05):** a rod head or tip within one
  wire radius of a layer interface. The thin-wire model has no end face: a
  rod that ends exactly at the interface is a wire on one side of it, one
  that reaches past it by a fraction of its radius touches the other layer
  through a short piece only, while the solid rod is continuous in between.
  Against an axisymmetric finite-volume model of the solid rod (4.5 m,
  a = 8 mm, 1000/30 Ω·m) the line kernel reads +3.3 / +3.9 / +7.2 / +21.4 %
  with the tip 16 / 8 / 1 / 0 mm above the interface and −4.6 / −1.2 /
  +0.7 / +2.1 % 1 / 4 / 8 / 16 mm past it (point kernel and `mom` alike).
  The step at zero penetration that an earlier entry of this release called
  the onset of the contact is this limit of the model. No end-face term:
  the solid rod's 161.4 Ω at the interface lies between the two branches of
  the thin wire (196 Ω above, at most 149.2 Ω past it). Tests:
  `tests/test_interface_slivers.py::test_rod_tip_within_one_wire_radius_of_the_interface_warns`,
  `::test_rod_head_within_one_wire_radius_of_the_interface_warns`,
  `tests/test_diagnostics_report.py::test_diagnose_reports_a_rod_end_at_the_interface`.
- **`min_distance` of `two_layer_real_space_kernel`,
  `two_layer_layered_correction_real_space` and
  `two_layer_layered_correction_group` (audit A3S-01):** the regularisation
  length of the closed-form 1/R terms in m; they are clamped at
  `min(1/lambda_clamp, min_distance)` instead of `1/lambda_clamp`
  (`h_1/200` near the interface). `None` (default) keeps the clamp; every
  value at or beyond it is bit-identical. Test:
  `tests/test_cross_layer_line_kernel.py::test_min_distance_leaves_every_farther_image_bit_for_bit`.
- **`groundfield.world.RESERVED_NAME_PREFIXES`** (`"__layer_"`,
  `"__cond_"`), see *Fixed*.

### Fixed

- ⚠ **Cross-layer diagonal at the interface (10.28, 10.32, audit A3S-01):**
  the diagonal of a segment is the layered correction at the wire radius
  with the interface image exchanged for its Galerkin integral, but the
  correction clamped its closed-form 1/R terms at `h_1/200` (25 mm for
  `h_1 = 5` m), so every piece within `h_1/400` of an interface lost 68–84 %
  of its layered correction. A ring (r = 3 m, a = 5 mm) at `h_1 = 5` m read
  −5.29 Ω with the default engine (1000/30 Ω·m; 322 Ω instead of 6.80 Ω in
  30/1000 Ω·m), and the contact values of a rod tip just past the interface
  were 10–32 % low in every segment backend. The solver now evaluates the
  diagonal at the wire radius and pairs closer than `h_1/200` at the 1 mm of
  its point forms (also `mom_sommerfeld`'s two-layer off-diagonal kernel and
  `two_layer_probe_matrix`, which clamps at its own `min_distance`). The
  diagonal equals the image solution to 1e-13; the ring is continuous through
  `h_1` (4.1826 Ω 1 µm above, 4.1805 Ω at `h_1`). Numbers move for a rod tip
  or head within `h_1/400` of an interface (12.5 mm for `h_1 = 5` m, 75 mm for
  30 m; 4.5 m rod, a = 8 mm, 1000/30 Ω·m, tip 1 µm past `h_1`: 133.5 → 149.2 Ω,
  +11.7 %; 10000/10 Ω·m: 123.2 → 180.7 Ω; against the solid rod −4.6 / −1.2 /
  +0.7 % at 1 / 4 / 8 mm, before −14.6 / −10.7 / −6.8 %), for horizontal wires
  at or within `h_1/400` below an interface, and for pairs of segments closer
  than `h_1/200` (segment lengths below about `h_1/100` near the interface:
  a 2 m rod through `h_1 = 30` m, ds = 0.05 m, line kernel −0.014 %). Not
  affected: AP1 (station rods end 0.2 m or more above `h_1`). Golden cases:
  none. Tests: `tests/test_cross_layer_line_kernel.py`
  (`::test_ring_resistance_is_continuous_through_a_deep_interface`,
  `::test_diagonal_correction_at_the_wire_radius_matches_the_image_solution`,
  `::test_pairs_closer_than_the_clamp_match_the_image_solution`),
  `tests/test_interface_slivers.py::test_rod_tip_contact_against_the_solid_rod_reference`,
  `::test_rod_head_contact_against_the_solid_rod_reference`,
  `::test_rod_tip_contact_does_not_depend_on_the_interface_clamp`.
- **Point form above the interface (10.28, audit A3S-02):** the point-form
  guard of audit round 2 ran only when a segment reaches `h_1`. With every
  wire above `h_1` the point form takes the image of a segment in the
  interface as a point source 2(`h_1` − z) away: a ring (r = 3 m,
  ds = 0.5 m) 1 µm / 1 mm / 1 cm / 5 cm above `h_1 = 5` m read −1931 / −945 /
  −57.9 / 21.2 Ω (line kernel 4.18 / 4.77 / 15.4 / 28.0 Ω) with
  `kernel="point"`, `mom`, `mom_sommerfeld` (two and three layers) and
  `solve_mutual_*`, also in 0.15.0; +853 % … +0.1 % in 30/1000 Ω·m. The guard
  now applies on both sides of the interface with the same thresholds: a
  horizontal wire closer than 0.387 segment lengths raises
  `BackendCapabilityError`, one within 0.6 warns with
  `PointKernelInterfaceWarning` (−2.5 % / −0.4 % at 0.2 m above `h_1`). Above
  the interface this error threshold is conservative: it also refuses coarse
  but accurate models (a ring r = 25 m 1.2 m above `h_1` with 5 m segments,
  +0.9 % against the line kernel), which a finer `segment_length` — the
  message names one — or the line kernel solves. The default line kernel is
  unaffected. Tests:
  `tests/test_cross_layer_point_form.py::test_point_form_rejects_a_horizontal_wire_just_above_the_interface`,
  `::test_point_form_warns_for_a_horizontal_wire_above_the_interface`,
  `::test_mutual_matrix_point_form_is_guarded_above_the_interface`,
  `::test_rods_ending_just_above_the_interface_are_not_refused`.
- **Reserved name prefixes (10.66, audit A3S-03):** an electrode called
  `__layer_r__L1` next to a rod `r` through `h_1`, or `__cond_pen__seg_0`
  next to a galvanic conductor `pen`, took the place of the solver's internal
  owner or node of that name and the solve failed with
  `numpy.linalg.LinAlgError`. `add_electrode`, `add_junction` and
  `World.check()` (every solve, worlds loaded from JSON) reject node names
  with the prefixes `__layer_` and `__cond_`. Tests:
  `tests/test_source_validation.py::test_check_rejects_a_name_with_a_reserved_prefix`,
  `::test_reserved_prefixes_are_rejected_when_a_node_is_added`.
- ⚠ **Field of a soil of one resistivity declared as layers (10.48, audit
  A3S-04):** `TwoLayerSoil(rho, rho, h)` solved like homogeneous soil, but
  `FieldResult.potential` and `solve_mutual_field` evaluated sources below
  `h` with the point form of the two-layer kernel instead of the line sources
  of the homogeneous result: +2.7 % at (0.2, 0, 4) m next to a 0.3–6.3 m rod,
  21 % at 5 cm from it. Both now read the canonical layer stack and equal the
  homogeneous result bit for bit; a stored `MultiLayerSoil` with at most two
  distinct layers is evaluated as two-layer soil (it raised
  `NotImplementedError` for three declared layers). Who is affected: field
  values of equal-resistivity layered declarations near wires below `h`.
  `fem` keeps the declared interface in its mesh on purpose (mesh topology
  continuous over a ρ₂ sweep): 1.7e-6 / 1.0e-5 from the homogeneous solve for
  two / three equal layers. Golden cases: none. Tests:
  `tests/test_layer_stack.py::test_field_of_an_equal_resistivity_stack_is_the_homogeneous_field`,
  `::test_potential_of_a_stored_three_layer_declaration_with_two_distinct_layers`.
- **Building large networks (audit round 3, generator note):** since
  generated mesh foundations are one electrode per wire line,
  `TnNetworkGenerator.build()` was 10–80 times slower for 100–1000 houses:
  `World.add_*` scanned every list for the new name. A name index next to the
  world (not part of the model: equality, copies, pickles and JSON are
  unchanged) makes the check O(1) per add; generated worlds are bit-identical
  (fingerprints for 100 / 400 / 1000 houses). 1000 houses: 6.97 → 2.44 s
  (star) and 7.52 → 2.65 s (radial) on the measuring machine. Tests:
  `tests/test_world_names.py::test_each_add_costs_constant_name_work`,
  `::test_lists_edited_by_hand_are_seen_by_the_next_add`,
  `::test_the_index_is_no_state_of_the_world`.
- **Quoted near-field deviations (10.21, 10.22):** `UniformCurrentFieldWarning`,
  the safety module and the known limitations quoted "+19 % / −21 %" (touch /
  step 1 m outside a 20 m grid corner), the point-kernel numbers, and the
  `image` engine page −28 % for a step that starts above the corner. They now
  give the default (line) kernel and the step: touch 1 m outside +20 %, step
  from 1 m to 2 m outside −21 %, from the point above the corner 1 m outwards
  −28 % (point kernel +19 / −21 / −28 %). The `interior` docstring of
  `touch_voltage_envelope` called a generated foundation a `grid_mesh`; it is
  a group of bonded wires whose cells the default samples. Tests:
  `tests/test_touch_voltage_footprint.py::test_the_quoted_corner_deviations_are_those_of_the_default_kernel`,
  `tests/test_touch_voltage_envelope.py::test_a_generated_mesh_foundation_is_sampled_in_its_cells_by_default`.

## Stream `fix3coup`: Fixes from the third adversarial audit: coupling

### Fixed

- ⚠ **Wires lying beside each other: order-free, pointwise rule (10.46, audit
  A3C-1, A3C-2):** the touching-wire rule of audit round 2 switched a whole
  near-parallel pair to touching wires when the partner's end points lay
  inside a cylinder of radius r_a + r_b around the *row* segment, so the
  result depended on the order of the conductors: a V pair of 1 m and 0.5 m
  at 1.5° read 0.362 or 0.503 µH (+38.7 %), the generator's in-line
  fall-of-potential setup with lumped leads read Im Z_meas(50 Hz) =
  0.134442 or 0.126676 Ω (perfect_mirror) depending on whether the feed or
  the probe lead came first. Tilted V and X pairs whose far ends lay just
  outside the cylinder kept a filament value above the self inductance: an
  indefinite inductance matrix (minimum eigenvalue −1.3e-8 H), a two-wire
  loop with Im Z_in(1 kHz) = −0.14 mΩ (−0.51 mΩ with the skin model). The
  rule now acts pointwise inside the Neumann integral: where a point of one
  wire lies beside the other wire (its foot on the other axis falls inside
  that wire) closer than r_a + r_b, that distance is used; both one-sided
  integrals (inner integral in closed form) are averaged on end points in a
  fixed order — bit-identical for every order and direction of the
  segments, continuous in the geometry, 4e-15 against an independent mpmath
  quadrature. The in-line reading is 0.126676 Ω in both orders (5 m lead
  segments: 0.126683 Ω); 2000 random close configurations × 6
  conventions gave no indefinite matrix (round-2 state: 700). Numbers move
  for pairs that lie beside each other closer than touching except exactly
  coincident ones: partly overlapping parallel wires (+2.3e-4 for a 2 mm
  offset against the whole-pair value), V pairs leaving a junction at an
  acute angle, wires crossing at one depth, branches starting on another
  segment (crossing at 30° with r = 0.1 mm: −7.9e-5). Chain neighbours,
  bends turning by less than 90° and pairs farther apart keep their values
  bit for bit. Golden cases: none (all bit-identical). Tests:
  `tests/test_interpenetrating_wires.py::test_beside_rule_does_not_depend_on_the_order_or_direction`,
  `::test_beside_rule_matches_an_independent_quadrature`,
  `::test_matrix_does_not_depend_on_the_order_of_the_conductors`,
  `::test_inline_reading_does_not_depend_on_the_order_of_the_leads`,
  `::test_partial_inductance_matrix_stays_positive_definite`,
  `::test_random_close_configurations_are_positive_definite`,
  `::test_two_wire_loop_has_a_positive_reactance`,
  `::test_mutual_is_continuous_across_the_touching_distance`,
  `::test_chain_neighbours_and_bends_keep_their_closed_forms`.
- ⚠ **A wire inside another (10.46, audit A3C-5):** a core on the axis of a
  cable shield was coupled like two wires touching on the outside (axis
  distance r_c + r_s): loop inductance +36 % against the geometric mean
  distance, +61 % against the analytic coax, silently. Where the thinner
  wire runs inside the thicker one, the thicker radius is used — a filament
  inside a thin tube couples to it as at the tube radius: with the internal
  inductance on the core only, the loop of a 100 m core (5 mm) in a shield
  (15 mm) is 26.966 µH against the analytic coax 26.972 µH. Equal radii
  never count as one wire inside the other (cables in one trench drawn on
  one axis stay touching wires); for unequal radii the value steps where
  the thinner axis leaves the thicker wire (concentric cable vs two cables
  side by side). Golden cases: none. Tests:
  `tests/test_interpenetrating_wires.py::test_wire_inside_another_couples_at_the_radius_of_the_thicker_wire`,
  `::test_wire_inside_another_uses_the_geometric_mean_distance`,
  `::test_coincident_wires_of_equal_radius_stay_touching_wires`.
- ⚠ **`carson_series`: segment order of near-parallel pairs (10.20, audit
  round 3, evidence e13):** a near-parallel pair that is not exactly
  parallel was projected on the axis of its first segment, so reversing the
  segment order moved entries of the correction by up to 3.1 % of the
  largest one (a pair got an overlap in one order and none in the other).
  Such pairs are now projected on the bisector of the two axes from the
  mean of their mid-points (3e-17 after reversal); exactly parallel pairs
  keep their numbers bit for bit. Numbers move for `carson_series` worlds
  with horizontal conductors at small angles to each other: the mutual
  impedance of two 300 m or 1 km leads 2 m apart grows by 2e-5 at 0.5°,
  4e-4 at 2°, 1.6e-3 at 4° and up to 6.4e-3 at 8° (50 Hz and 1 kHz); the
  step at the 8.1° boundary becomes 12.8 % for the 1 km leads at 1 kHz
  (12.3 %). Golden cases: none. Test:
  `tests/test_carson_validity_domain.py::test_carson_tiling_does_not_depend_on_the_segment_order`.
- ⚠ **Filaments at tiny angles (10.46, audit A3C-4):** two filaments from
  one vertex below the coaxial tolerance (4.5e-5 rad) were taken as
  overlapping coaxial filaments and fell to the 16 × 16 rule: 16 % off at
  1.5e-6 rad, 3.8 % at 1e-5 rad; just above it the vertex formula lost
  2.7e-9. Below 0.81° the vertex formula is evaluated without cancellation:
  round-off on axis-parallel geometry, at most 4e-10 in a general
  orientation. Filaments crossing at tiny angles lost the crossing-point
  mark of the close-pair rule (computed from 1 − cos², cut off at 1e-14):
  9.5e-4 at 1e-8 rad; below 1 − cos² = 1e-6 the mark now comes from a
  cross product: 1.0e-9 at 1e-8 … 3e-8 rad, 5e-11 from 1e-6 rad on. Numbers
  move for filaments (`neumann_mutual`) and segment pairs at such angles
  that are farther apart than r_a + r_b. Golden cases: none. Tests:
  `tests/test_neumann_touching.py::test_filaments_from_one_vertex_at_tiny_angles`,
  `tests/test_neumann_close_pairs.py::test_crossing_filaments_at_tiny_angles`.
- **Accuracy statements say where they hold (10.24, 10.46, 10.20, audit
  A3C-4):** measured worst cases replace statements that held only where
  they were measured. Sommerfeld correction of a long segment (composite
  rule above a quarter skin depth; audit round 2 stated 7e-5 / 1e-6 and
  "lumped conductors match their chains to 1e-4" from buried cases):
  against a refined rule, 50 Hz and 1 kHz, up to 3 km, within 5.9e-4 in
  the earth-return resistance and 2.4e-4 in the reactance (worst: a 3 km
  overhead line at 10 m over 10 Ω·m at 1 kHz; a 1 km one 4.8e-4 / 1.2e-4;
  buried in 100 Ω·m 1.7e-4), tabulated 3.3e-4 / 9.6e-5; the historic rule
  below a quarter skin depth reaches 4.8e-4, so the correction steps by
  3.5e-4 (resistance) at L = δ/4. Close pairs (stated "5e-11"): 5e-11 above
  1e-6 rad, 1.0e-9 at 1e-8 rad; pairs inside the parallel tolerance use the
  parallel closed form at the start offset (4.2e-9 for wires 1 cm apart
  tilted by 1e-10 rad, 1.7e-5 at 1 µm). Overhead wire over a buried one
  (stated "1e-9"): 4.2e-9 up to x = 300 m (1 kHz, 10 Ω·m; 1e-10 at
  50 Hz), 1.2e-7 at x = 1 km, inherited from Carson's J (1.2e-7 at a = 28,
  θ = 89.9°, QUADPACK's Fourier rule). Tests:
  `tests/test_sommerfeld_long_segments.py::test_long_segment_rule_meets_its_stated_accuracy`,
  `::test_the_step_at_a_quarter_skin_depth_is_the_stated_one`,
  `tests/test_neumann_close_pairs.py::test_crossing_filaments_at_tiny_angles`.
- **`carson_series` validity domain in the documentation (10.20, audit
  A3C-3):** the explanation page and the how-to still described the
  correction as "scaled by the geometric length of each segment pair" (the
  heuristic removed in audit round 2). They now describe the per-metre
  tiling of near-parallel horizontal pairs, the transmitted field across
  the surface, the Sommerfeld fallback with its step at 8.1°, the
  long-segment rule and the rule for wires beside each other; ADR-0004,
  -0005 and -0006 carry the corrections in their 2026.11 amendments. Test:
  `tests/test_carson_validity_domain.py::test_docs_describe_the_carson_validity_domain`.

## Stream `fix3gen`: Fixes from the third adversarial audit: generators and export

### Added
- **`groundfield.io.groundinsight.MAX_METADATA_DEPTH` (32):** deepest nesting of
  objects and lists in the metadata of a BusType document (the exporters write
  four levels); a deeper document is refused on load. Test:
  `tests/test_io_groundinsight.py::test_crafted_bustype_documents_raise_value_error`.

### Changed
- ⚠ **PEN cables of `TnNetworkGenerator` (10.38; stream DOC-B; audit A3SG-01,
  A3SG-02, A3SG-03):** every cable cabinet has its own cable, as the topology
  documents (star: a straight cable from the substation; radial trunk: one
  cable per trunk segment); cabinets on one line get separate cables in one
  trench. A service drop taps the cable path that feeds its own building — the
  cable of its cabinet, or the trunk from the substation to its tap source — at
  the point nearest to the building's connection point: a leakage-free tap
  junction `pen_tap_<k>` that splits the cable into sections `<cable>_s<k>`, or
  the substation / the cabinet at the end of the cable; a building in a
  substation slot of the radial trunk is connected to the substation. What the
  router guarantees: a drop never starts on another building's drop and never
  runs along the cable it taps; no cable section or drop is shorter than 1 cm (a
  tap closer to a node is that node; longer for PEN radii above 7.3 mm, so that
  every section keeps twice the resistance below which a distributed conductor
  is refused); a building whose connection point lies on its feeding cable
  splits it (no drop); two nodes closer than that raise `ValueError`. A drop
  may share its route with another cable (another cabinet's cable, a trunk,
  another drop), as cables in one trench do; the inductive coupling treats such
  conductors as touching wires and the networks stay passive. Against 0.15.0,
  where every drop ran from its cabinet (60 to 160 m back along the main in the
  default network, coaxial with it): the routing moves the golden networks
  (one cabinet west of every house) by −2.1 % in the substation potential,
  −3.1 to −4.8 % in the substation leakage, −0.2 to +1.0 % in the house
  potentials and −1.1 % in the measurement network;
  the input impedance of default-grid networks with ring foundations (image
  backend, 50 Hz, 100 Ω·m) moves by −6.4 / −1.1 / −0.2 % for 30 / 100 / 200
  houses (star) and 0.0 / −0.1 % for 30 / 100 houses (radial trunk). Against
  the nearest-point tree of the first 2026.11 draft (chained cabinets, drops
  hung on drops): golden networks bit-identical; 30 / 100 / 200 houses −4.1 /
  −2.9 / −2.7 % (ring foundations) and −2.4 / −3.4 / −2.8 % (default mesh
  foundations), radial trunk −0.5 / −0.3 %. Golden cases: `gen_tn_default`,
  `gen_tn_measurement_overhead_leads` (moves declared in `fix2gen.toml`).
- **Polygon foundations of thin outlines (10.54; audit A3SG-07):** with
  `foundation_outline="polygon"` an outline whose oriented bounding rectangle
  is narrower than `MIN_FOUNDATION_SIDE_M` (1 m) is replaced by that rectangle
  widened to 1 m, with the same `UserWarning` as for `"ombr"` (a ring along an
  outline 0.5 to 1 m wide ran along both faces of a wall). Outlines wider than
  1 m on average are unchanged.
- **`TnNetworkGenerator.build()` of large networks (no finding):** the
  materialiser hands objects instead of names to the world, so bonds and PEN
  sections need no search through the world: 1000 default houses 6.5 → 4.4 s
  (star) and 7.3 → 5.4 s (radial trunk), worlds byte-identical. The rest of the
  cost is the name checks of `World.add_electrode` / `add_conductor`, linear in
  the size of the world.

### Fixed
- **`build()` failed for 1–3 % of random TN networks (10.38; audit A3SG-01):**
  a service drop whose tap fell 0.1 µm to 4.7 mm from a node of the cable left
  a section that the `Conductor` model refuses (series resistance below
  1e-6 Ω), so `build()` raised `pydantic.ValidationError` (1 of 100 random
  60-house and 2 of 60 random 200-house networks). The layout builder had the
  same defect for taps a few millimetres from a cable corner or from another
  tap; it now merges cable nodes closer than the shortest section and refuses a
  cable between two sites closer than it with `ValueError`. Tests:
  `tests/test_pen_routing.py::test_a_building_millimetres_beside_a_node_still_builds`,
  `::test_a_tap_millimetres_from_an_earlier_tap_snaps_to_it`,
  `::test_random_position_network_builds`,
  `::test_taps_on_a_thick_pen_keep_sections_the_conductor_model_accepts`,
  `::test_random_networks_build_and_route_as_documented` (400 random networks,
  60 / 200 houses, both topologies),
  `tests/test_lv_network_layout.py::test_taps_millimetres_apart_share_one_node`,
  `::test_a_tap_millimetres_from_a_cable_corner_is_the_corner`,
  `::test_a_cable_between_two_sites_closer_than_a_section_is_rejected`.
- **Cabinets chained on one cable, drops hung on other drops (10.38; audit
  A3SG-02, A3SG-03):** the nearest-point tree of the first 2026.11 draft fed a
  cabinet from the cable of a nearer one, although `StarCabinetTopology` and
  `TnNetworkConfig.pen_topology` document a direct cable per cabinet (input
  impedance +1.8 / +2.6 % for 100 / 200 houses against the 0.15.0 routing,
  against the documented direction), and started most drops on other houses'
  drops (26 of 30 in the radial default grid with ring foundations, 19 of 30 in
  the default star network). Tests:
  `tests/test_pen_routing.py::test_star_topology_wires_every_cabinet_directly_to_the_substation`,
  `::test_collinear_cabinets_have_separate_cables_in_one_trench`,
  `::test_z_in_of_a_100_house_network_drops_against_the_old_routing`,
  `::test_a_drop_taps_the_cable_of_its_own_cabinet`,
  `::test_no_drop_starts_on_another_drop_in_the_radial_default_grid`,
  `::test_overlapping_drops_and_trunks_stay_passive`, `::test_routing_guarantees`.
- **A loaded BusType document crashed the other entry points (10.01; audit
  A3SG-04):** `load_bustype_json` accepted documents for which
  `fit_quality_summary` raised `AttributeError`, `TypeError`, `ValueError` or
  `OverflowError` (`fit_quality` null or a list; a figure that is an object, a
  list, a string or a 400-digit integer), `to_dict()` raised `RecursionError`
  (metadata nested 3 000 levels, which the JSON decoder reads) or wrote a NaN
  description (not strict JSON). The loader now refuses a `name` or
  `system_type` that is an object or a list, a `description` that is neither a
  string nor null, an `impedance_formula` that is an object or a list
  (`EvaluateSpecError`) and metadata deeper than `MAX_METADATA_DEPTH`;
  `to_dict()` writes a non-string description as null and walks the metadata
  without recursion; `fit_quality_summary` reads only finite numbers and
  strings. Tests: `tests/test_io_groundinsight.py::test_crafted_bustype_documents_raise_value_error`,
  `::test_a_loaded_document_never_crashes_the_other_entry_points`,
  `::test_fit_quality_summary_reads_the_figures_it_can`,
  `::test_to_dict_of_a_spec_built_in_memory_is_strict_json_or_refused`.
- **A sum with an exact small imaginary part parsed to zero (10.01; audit
  A3SG-05):** the formula parser drops a collected sum that cancels within its
  128-bit working precision, but compared the complex magnitude of the sum with
  its largest term: `(1 + 1e-31*j - 1)*1e31*f` evaluated to 0 instead of 50j at
  50 Hz. Real and imaginary parts are now judged separately; the 119 accepted
  formulas of the corpus evaluate as before (worst 1.9e-15 against a 60-digit
  mpmath evaluation). Test:
  `tests/test_formula_security.py::test_real_and_imaginary_parts_of_a_sum_cancel_separately`.
- **Polygon foundations of outlines narrower than 0.5 m (10.54; audit
  A3SG-07):** `LvNetworkLayout.to_world(foundation_outline="polygon")` raised
  `ValueError` ("polygon collapses ... after removing sub-centimetre edges",
  although the edge limit is 0.5 m) for every such outline the Overpass parser
  keeps; it is widened now (see Changed), and the materialiser's last check
  names the real limit. Tests:
  `tests/test_footprint_foundations.py::test_a_thin_outline_with_a_polygon_foundation_is_widened`,
  `::test_a_polygon_foundation_of_an_ordinary_outline_is_unchanged`,
  `::test_a_collapsing_polygon_outline_names_the_real_edge_limit`.
- **Margin of generated mesh foundations against `mom` (10.39; audit
  A3SG-06):** the documentation said one current per wire is +0.3 / +0.5 /
  +0.6 % "against `mom`"; that is the line kernel against a line-kernel
  per-segment reference. Against `mom` (point kernel) the default line kernel
  is +1.1 % and `kernel="point"` +0.3 … +0.5 % for 10 m 2 × 2 to 30 m 4 × 4
  meshes; one current for the whole mesh was +2.4 … +3.7 % (point kernel
  +1.5 … +3.1 %). Test:
  `tests/test_foundation_owners.py::test_documented_margin_of_mesh_foundations_against_mom`.

## Stream `dsafety`: User decisions Q3 and Q4: touch points, EN 50522 beyond 10 s

### Added
- **Touch points: `touch_voltage_at_part` (10.21; user decision Q3):** the
  EN 50522 touch voltage of an accessible metal part above ground — a
  station housing or its door, a mast, a fence post, a cable cabinet, a
  service connection box. The part is given by its horizontal position and
  the electrode or junction it is bonded to (`bonded_to=`), whose potential
  it carries; a part bonded to a PEN junction expresses a transferred
  potential. The feet stand on the circle of radius `distance` (1 m) around
  the part (`n_angles`, default 24), optionally only in given `directions`,
  and the positions inside the outlines in `exclude` (the housing or the
  building, where nobody can stand) are dropped. Returns the touch-voltage
  phasors and azimuths per foot position like `touch_voltage_envelope`
  (and the positions with `return_points=True`). Available as
  `gf.touch_voltage_at_part` and `gf.safety.touch_voltage_at_part`. A
  service connection box 40 m from a ring earth (100 A, 100 Ω·m), bonded to
  the PEN, has 1081 V of the 1121 V earth potential rise as touch voltage.
  Tests: `tests/test_touch_voltage_at_part.py::test_part_on_a_wire_matches_touch_voltage_on_the_offset_curve`,
  `tests/test_touch_voltage_at_part.py::test_part_at_a_rod_head_against_the_analytic_surface_potential`,
  `tests/test_touch_voltage_at_part.py::test_part_bonded_to_a_pen_junction_carries_the_transferred_potential`.
- **How-to "Assess touch voltages"** (`docs/how-to/assess-touch-voltages.md`):
  touch points of a compact station, the transferred potential over the
  PEN, the meshes of an open-air grid, and the comparison with the
  permissible values; tutorial T08 adds a fence bonded to the substation
  grid (199.5 V at the corner post against 140.7 V 1 m outside the grid,
  `mom`, 150 A) and an earth fault that is not cleared.

### Changed
- ⚠ **`touch_voltage_envelope` samples no interior by default (10.21; user
  decision Q3):** inside an electrode's footprint stands the station, the
  house, the steel structure or the mast, so the default is now
  `interior=False` for every electrode kind: the curve `distance` outside
  the wires only. The mesh sampling of the development snapshots (the
  centres of the meshes of grids and of generated mesh foundations, the
  centre of a ring) is the explicit `interior="meshes"`, for open-air grids
  such as a substation yard; `interior=True` also samples the inside of a
  closed polyline. `interior=None` is rejected with a hint. Against 0.15.0
  the default envelope is the footprint's offset curve of 10.21: a ring of
  10 m radius at 0.8 m (100 A, 100 Ω·m, `image`, point kernel) keeps
  139.4 V (the development snapshots gave 182.6 V, its centre; with the
  default line kernel 142.2 / 185.5 V), a 20 m perimeter loop (1 × 1 grid)
  141.4 V instead of 158.3 V in its centre, and the 20 m × 20 m grid with
  4 × 4 meshes 111.4 V (unchanged: the maximum is 1 m outside a corner, the
  mesh centres reach 63.0 V). Affected: envelopes of rings, grids and
  generated foundations computed with the development snapshots; pass
  `interior="meshes"` for open-air grids. Golden cases: none (no case
  evaluates the envelope). Tests:
  `tests/test_touch_voltage_envelope.py::test_envelope_samples_no_interior_by_default`,
  `tests/test_touch_voltage_envelope.py::test_release_note_envelopes_with_the_new_default`.
- ⚠ **EN 50522 permissible touch voltage beyond 10 s (10.42; user decision
  Q4):** `permissible_touch_voltage_en50522(t_f)` follows Figure 8 /
  Table B.4 of EN 50522:2022 up to 10 s and returns 80 V for every
  t_f > 10 s, including `math.inf` for an earth fault that is not cleared
  (resonant-earthed networks). The curve of Figure 8 ends at 10 s; the
  note to Figure 8 (5.4.1) and the text after Table B.4 (B.3) allow 80 V
  for current-flow durations considerably longer than 10 s, below the
  85 V end value of the table. 0.15.0 returned 85 V beyond 10 s (now
  −5.9 %) and rejected `math.inf`. `permissible_prospective_touch_voltage_en50522`
  uses the same U_Tp: beyond 10 s 80 V + 50 mA · R_A / HF (1000 Ω shoes
  on 100 Ω·m soil: 137.5 V instead of 142.5 V). Affected: assessments
  with fault durations above 10 s, e.g. resonant-earthed networks. Golden
  cases: none (`safety_en50522_curve` covers 0.05 … 10 s). Tests:
  `tests/test_safety.py::test_en50522_beyond_ten_seconds_is_80_volt`,
  `tests/test_safety.py::test_prospective_touch_voltage_beyond_ten_seconds_uses_80_volt`.

### Removed
- **`long_duration=`** of `permissible_touch_voltage_en50522` and
  `permissible_prospective_touch_voltage_en50522` (added during this
  release cycle, never released): 80 V beyond 10 s is the default. Test:
  `tests/test_safety.py::test_en50522_has_no_long_duration_keyword`.

### Deprecated
- Nothing.

## Stream `drhoeff`: User decision Q6: effective resistivity

### Added

- **`effective_resistivity` (10.84, user decision Q6):** the effective
  resistivity of a homogeneous or two-layer soil in Ω·m,
  ρ_eff = ρ₁^(1−w) · ρ₂^w with w = w0 · (h₁ / h_ref)^(−β), h_ref = 10 m and
  the AP1 calibration w0 = 0.13, β = 0.8 (group-specific values as `w0=`,
  `beta=`, `h_ref=`). `gf.effective_resistivity(soil)` (also
  `groundfield.soil.effective_resistivity`, or with the keywords `rho_1`,
  `rho_2`, `h_1`) returns ρ itself for homogeneous soil and for stacks of
  equal layers, without round-off (the plain formula gives
  200.00000000000003 Ω·m for 200 / 200 Ω·m, h₁ = 10 m); reduces a
  `MultiLayerSoil` to its canonical stack (equal adjacent layers merged, as
  the layered backends do, 10.48) and raises `ValueError` for three or more
  distinct layers; limits w to [0, 1], so ρ_eff stays between ρ₁ and ρ₂ (the
  unlimited formula leaves that interval below h₁ = 0.78 m: 17.9 Ω·m for
  100 / 30 Ω·m at h₁ = 0.5 m; with the limit such a thin upper layer gives
  ρ₂). Inside the calibration range of the AP1 catalogue (h₁ = 5 … 30 m,
  ρ₁ = 30 … 1000 Ω·m, ρ₂ = 30 … 200 Ω·m) the value is bit-identical to the
  AP1 / AP2 scripts' `r1 ** (1 - w) * r2 ** w` and agrees with a 50-digit
  evaluation to 1.6e-16. Test: `tests/test_effective_resistivity.py`.

### Changed

- ⚠ **The `rho` of a BusType in layered soil is the effective resistivity
  (10.84, user decision Q6):** `rho_at_fit` of a vector fit and the `rho`
  samples of a standard form fitted in layered soil are ρ_eff of the soil
  they were computed in, `effective_resistivity(soil)`, the
  `specific_earth_resistance` the AP2 network model gives a groundinsight
  bus. This replaces the upper-layer resistivity ρ₁ that the 10.84 entry of
  this release cycle had made the convention; on the AP1 grid ρ_eff differs
  from ρ₁ by −54.8 % (1000 over 30 Ω·m, h₁ = 5 m) to +53.6 % (30 over
  200 Ω·m, h₁ = 5 m). The error message for a missing `rho_at_fit`, the
  docstrings of the four exporters, the IO and Soil references, ADR-0008
  and tutorial T05 state it. Every BusType document records the convention
  as `metadata["rho_definition"] = "effective_resistivity"` (documents of
  0.15.0 have no such entry). No function computes a different number: the
  exporter writes the `rho` it is given. Affected: BusTypes exported in
  layered soil with ρ₁ as `rho`. A compact electrode in the upper layer
  follows ρ₁ more closely than ρ_eff of the AP1 networks: the station of the
  IO example (2 m ring and 3 m rod) swept over ρ₁ fits the standard form to
  13.4 / 8.3 / 3.6 % over ρ_eff and to 3.9 / 1.9 / 0.6 % over ρ₁ for
  h₁ = 5 / 10 / 30 m; check `max_relative`. Golden cases: none. Tests:
  `tests/test_io_groundinsight.py::test_rho_at_fit_convention_for_layered_soil_is_the_effective_resistivity`,
  `::test_bustype_metadata_records_the_rho_definition`,
  `::test_layered_soil_exports_pair_with_an_effective_resistivity_bus`.

## Stream `ddefaults`: User decisions Q15 to Q17: galerkin, tabulated Sommerfeld, resource limit

### Added

- **Per-engine resource limit `Engine.max_segments` (10.06, user decision
  Q17):** the hard problem-size limit of the segment backends — the number
  of leakage segments and the number of unknowns of the dense nodal system —
  is set per engine: `Engine(max_segments=...)` (also a `create_engine`
  parameter) raises or lowers it for one engine, a positive integer;
  `None` (default) takes the module constant
  `groundfield.diagnostics.HARD_LIMIT` (20 000), read at solve time, so the
  constants stay the defaults. Every segment backend (the image family,
  `mom`, `mom_sommerfeld`), `solve_mutual_matrix` and `solve_mutual_field`
  refuse a larger world with `ResourceLimitError` before anything of that
  size is allocated; `check_segment_budget`, `expected_segments` (new key
  `"limit"`), `check_segment_resolution` and `diagnose`
  (`report.context["max_segments"]`) check against the same limit, and the
  messages name the setting to change. The limit is recorded in the new core
  metadata key `FieldResult.metadata["max_segments"]` (the engine block of
  the provenance keeps the request itself). `SOFT_LIMIT` (5 000, the size
  above which the solve reports that it may take minutes) stays a module
  constant. Tests: `tests/test_resource_limit.py`.

### Changed

- ⚠ **`solve_mutual_matrix` uses the galerkin off-diagonal for every call
  (10.34, user decision Q15):** `offdiag="galerkin"` (the reaction of the
  leakage distributions: reciprocal, and the matrix with which
  (Y_L + Z_G⁻¹)u = i reproduces the solve of the clusters joined by branches)
  is the default whatever the call; `offdiag="probe"` (the potential at one
  probe point per cluster, the definition of releases up to 0.15.0) stays
  available. A call that passes `probe_points` without `offdiag="probe"` —
  every 0.15.0 call, the AP1 catalogue pipeline among them — returns the
  galerkin matrix; the points are ignored, not validated, with the new
  `groundfield.solver.mutual.ProbePointsIgnoredWarning`, which names
  `offdiag="probe"` as the way back. This supersedes the selection rule
  added earlier in this release cycle (`offdiag=None` choosing `"probe"` when
  probe points were given; `None` is no longer accepted). The diagonal does
  not move. On an AP1-like site (station ring and three rods, twelve polygon
  foundations, ds = 2 m, five soils) the symmetrised probe matrix missed the
  input impedance of the joined network by −0.15 … −0.88 % and single
  off-diagonal entries by up to 41 % (median 0.15 … 1.1 %); the galerkin
  matrix reproduces it to 7e-13. On the AP1 catalogue path (group-2 site,
  7 corner soils × 4 penetrations) |Z_ons| at 50 Hz moves by −0.05 … +0.29 %
  at penetrations up to 20 % and by +0.25 … +2.67 % at 40 % (largest at
  ρ₁ = 1000 Ω·m), the ONS current share at 50 Hz by up to +2.7 %. Who is
  affected: every `Z_G` built with probe points, the AP1 catalogue. Golden
  cases: `mutual_foundations_lumped_concrete`,
  `mutual_foundations_distributed_concrete`, `mutual_three_rods_crosslayer`,
  `line_mutual_two_layer` (Z_G norm-wise 7.7e-4 … 8.8e-4, single off-diagonal
  entries 0.48 … 5.9 %). Tests:
  `tests/test_mutual_matrix.py::test_galerkin_is_the_default_whatever_the_call`,
  `::test_ignored_probe_points_are_not_validated`,
  `::test_offdiag_selection_and_validation`.
- ⚠ **Tabulated Sommerfeld assembly by default (10.60, user decision Q16):**
  `Engine.sommerfeld_assembly` defaults to `"tabulated"` (one table of the
  spectral integral per pair of depth groups and frequency); `"exact"` (one
  quadrature per segment pair, the default up to 0.15.0) stays available.
  The image family, `mom`, `mom_sommerfeld` and `solve_mutual_*` follow the
  setting alike; engine JSON without the field (0.15.0) gets the new
  default. Blocks of fewer than 64 sub-segment pairs keep the per-pair rule,
  so small inductive worlds do not move. On a TN network with an inductive
  PEN (`TnNetworkGenerator`, homogeneous 100 Ω·m, ds = 1 m, four
  frequencies 50 … 1000 Hz) the input impedance of the tabulated solve lies
  within 2e-10 of two independent refined references (per-pair rule with 200
  instead of 24 interpolation nodes; tables of 400 instead of 100 nodes; both
  with a refined λ grid), which agree with each other to 2e-10, while the
  per-pair rule is 1.2e-8 off (8 houses); CPU time 1.0 s instead of 16.6 s
  (8 houses, 78 inductive branches) and 12.2 s instead of 306 s (30 houses,
  338 inductive branches; input impedance equal to 1.1e-8, the tabulated one
  1.8e-11 from the 400-node tables). The correction matrices of small
  conductor sets in twelve critical geometries (buried, overhead, 1 cm below
  the surface, in the surface plane, crossing the surface or a layer
  interface, two- and three-layer earth, 250 m segments, 10 and 1000 Ω·m;
  50 Hz and 1 kHz) are finite, symmetric and free of warnings; against a
  reference refined in S(ρ) and in the λ grid the tabulated rule is closer
  than the per-pair rule in ten of them (1.5e-7 … 1.1e-5 against
  4.2e-6 … 7.6e-5 of the largest entry) and equal in two (the surface plane,
  where both use the per-pair rule, and a riser through h₁, 4.3e-6). Who is
  affected: inductive conductors under `earth_inductive_model="sommerfeld"`
  (the default) with blocks of at least 64 sub-segment pairs; numbers move
  by about 1e-8 … 2.4e-6. Golden case: `cond_lumped_long_pen_sommerfeld`
  (≤ 2.4e-6; against the refined per-pair reference the tabulated result is
  4.2e-7 off, the per-pair rule 2.0e-6). Tests:
  `tests/test_sommerfeld_assembly_default.py`.

## Decisions on the questions left open (2026-11-08)

After the third adversarial audit and the user's decisions Q1–Q19, the release
preparation left 17 questions to the author, each with a recommendation. The
author followed every recommendation. The table records the question, the
decision and where it went; the work streams `d2solver`, `d2gen` and `d2soil`
below implemented the decisions that change code.

| Q | Question | Decision | Where |
|---|---|---|---|
| 20 | Encase the cross-braces of mesh foundation electrodes in the concrete shell, as DIN 18014 lays the foundation earth electrode with its connections in the concrete? | yes | streams `d2gen`, `f2gen` |
| 21 | Deprecate `FoundationElectrodeSpec.concrete_model`, which only chooses one or four current owners of a ring foundation (≤ 0.06 % apart up to aspect ratio 4)? | yes; the measured difference is larger (< 1 % / < 2 % up to aspect ratio 4 / 50, see the box below; for the author's review) | streams `d2gen`, `f2gen`, `f3gen` |
| 22 | Keep the PEN routing of `TnNetworkGenerator` (an own cable per cabinet, a service drop taps the cable that feeds its building)? Place cabinets per building group in the future? | routing kept; cabinets per building group moved into 2026.11.0 (user decision of 2026-11-14, stream `w1gen`; a wide group's cabinet on the ray through its far-end building since audit W1B-01, stream `fw1gen`; the 20 m step of the rule stays, user decision D7, stream `fw1r2gen`) | [changelog](../changelog.md) |
| 23 | `fem` whose equivalent hemisphere puts current into a more conductive layer that the electrodes do not touch (−38 … −98 %): error instead of warning? | error; extended after the audit to a more conductive layer within the electrodes' horizontal half-extent (coordinator, for the author's review) | streams `d2solver`, `f2solver`, `f3solver` |
| 24 | Point-form guard near a layer interface: keep the conservative thresholds (warning within 0.6 segment lengths, error within 0.387)? | kept | [known limitations](../reference/known-limitations.md#current-ansatz-and-source-kernels) |
| 25 | Accept the step of `carson_series` at the 8.1° boundary of its validity domain as a documented limit? | accepted | [known limitations](../reference/known-limitations.md#earth-return-and-conductors), [earth return](../explanation/earth-return.md) |
| 26 | Accept that the simulated in-line fall-of-potential reading moves because the leads now couple as wires lying side by side (Im Z_meas 0.063 → 0.054 Ω at 50 Hz)? | accepted | release notes, 10.46 |
| 27 | Warn when `perfect_mirror` couples conductors on opposite sides of the soil surface through its additive image? | warning | streams `d2solver`, `f2solver`, `f3solver` |
| 28 | Count powers of numbers (`10**40`) in the rule on nested exponents of the formula parser, as groundinsight does? | yes; already the rule (`(f*10**40)**2` is rejected, `tests/test_formula_security.py::test_exponents_of_nested_powers_multiply_to_at_most_max_pow`) | no change |
| 29 | Only document that printing a parsed formula or passing it to `lambdify` lets SymPy evaluate parts of an untrusted formula again? | documented | [IO reference](../reference/api/io.md) |
| 30 | One current per wire line for grids and stars in the image family (−3 … −5 %)? | moved into 2026.11.0 (user decision of 2026-11-14, stream `w1sol`): grids one owner per wire line (−1.2 … −5.5 %), building outlines per side (Q44); a star alone does not change (equal arm currents), its remaining error is on the [roadmap](../roadmap.md#left-open-by-2026110) | [changelog](../changelog.md) |
| 31 | Make the soft size notice (5 000 segments) adjustable per engine as well? | yes | stream `d2solver` |
| 32 | Read a thin conductor inside a thicker one as a concentric cable only with an explicit marker? | moved into 2026.11.0 (user decision of 2026-11-14, stream `w1sol`): marker `conductor_type="cable_shield"` | [changelog](../changelog.md) |
| 33 | No internal inductance of a solid wire for cable shields (100 m coaxial pair 31.97 instead of 26.97 µH)? | moved into 2026.11.0 together with 32: 26.97 µH; sub-pieces add up since audit W1A-02 (stream `fw1sol`), also tilted against each other since audits W1R2-01, W1R3-01 and W1R3-02 (streams `fw1r2sol`, `fw1r3`) | [changelog](../changelog.md) |
| 34 | Tighten the rule for segments up to a quarter skin depth under `sommerfeld` (step of 3.5e-4 at δ/4)? | a later release (2026-11-08); moved into 2026.11.0 by the user decision of 2026-11-14 | done in 2026.11.0 (streams `w3perf`, `fw3perf`): exact one-dimensional self rule and graded close-pair rule; with the λ grid of audit W3B-01 the self correction is within 1.1e-9 of a refined rule from 1 m and 0.01 δ to 3 km at 50 Hz and 1 kHz for the soils tested (up to 3.4e-9 in high-contrast soils, audit W3R2B-08), no step (was 4.8e-4, step 3.5e-4); close pairs within about 4e-8 (were up to 1.5e-3); five golden cases move by up to 4.9e-7 ([changelog](../changelog.md)) |
| 35 | Radial trunk: a building in a substation slot taps its feeder's trunk instead of getting its own line from the substation? | moved into 2026.11.0 together with 22 (stream `w1gen`) | [changelog](../changelog.md) |
| 36 | Presets of the effective-resistivity weights per electrode class, with the values of the AP2 hand-over? | yes | streams `d2soil`, `f2gen` |

The audits of the decision streams left four more questions, answered on the
same day; the author again followed the recommendations and chose the next
release for 37 and 39, this release for 38, and for 40 a quantification
only once a 3-D volume reference exists. The earlier wording of 37 pointed to
a "conforming axisymmetric" roadmap item that did not exist yet. With the
user decision of 2026-11-14, 37 and 39 moved into 2026.11.0, and the 3-D
model of wave 4 gave the reference that 40 waited for; the rows say where
each was done.

| Q | Question | Decision | Where |
|---|---|---|---|
| 37 | Stop adding rules to the equivalent-hemisphere reduction of `fem`, in which every audit round found new limit cases (an interface 1 cm below a deep ring +191 … +308 % without a message; a rod reaching 1 µm … 8 cm into a more conductive top layer −44 … −78 %, warning only)? | yes; no further rules, replaced by a conforming axisymmetric model (rod on the axis, ring, ring with central rod; other electrodes refused until a full 3-D method exists); planned for a later release on 2026-11-08, moved into 2026.11.0 with the full 3-D model by the user decision of 2026-11-14 | implemented in 2026.11 (streams w4axi, w4fem3d; fixes of audits W4A / W4B in fw4axi, fw4fem3d): conforming axisymmetric model of rods on one axis and rings around it (about 1e-4; rings next to a conductive layer 3.3e-4 since W4A-01), and a three-dimensional conforming model for every other world — strips, grids, meshes, several clusters, galvanic distributed conductors — within 2e-3 of the axisymmetric model away from interfaces and other wires (W4B-01/-02) and between `image_2layer` and `mom` at the AP1 station; the 3-D port limit is the option `FemOptions.max_ports_3d` (stream fw4int); the equivalent-hemisphere reduction is gone ([changelog](../changelog.md)) |
| 38 | Which owner model stays when `concrete_model` is removed: one current per rectangle without cross-braces (up to 2.2 % above the reference with one current per segment) or one per side (up to 0.9 %; short side at least 1 m)? | one per side, already in 2026.11.0: `concrete_model` has no effect | stream `d3gen` |
| 39 | Record the effective-resistivity preset in the BusType document (for example `metadata["rho_preset"]`), so that a bus paired with the wrong preset is detectable? | yes, in a later release (2026-11-08); moved into 2026.11.0 by the user decision of 2026-11-14 | done in 2026.11.0 (streams `w3perf`, `fw3perf`, `fw3r2b`): `rho_preset` of the exporters, `metadata["rho_preset"]`, `BusTypeSpec.rho_preset`, the last line of the description of the live BusType and of its database entry (`bustype_rho_preset`; audit W3B-06), `RhoPresetMismatchWarning` from `evaluate_spec` / `bustype_from_spec`; a document holds a known preset name or null, and only an exact preset line of a description is replaced (audit W3R2B-06); older documents load with `None` ([changelog](../changelog.md)) |
| 40 | Quantify the overlap of the concrete shells at the corners and crossings of encased foundations? | only once a 3-D volume reference is available | quantified in 2026.11 (stream w4fem3d, audit W4B-05) against the concrete resolved as a volume in the 3-D `fem` model: a junction acts like a few cm of wire, at most about 2 r_b, sign not settled (not converged); the generators' house mesh changes by about 1 % or less (0.35 … 0.85 % re-measured after W4B-01); no correction rule ([concrete shells](../explanation/concrete-shells.md#overlapping-shells-at-junctions)) |

The audits of stream `d3gen` (D3A-02, D3A-03, D3B-03, D3B-04) found that the
generators attach PEN conductors at the connection point of a foundation's
anchor, which is a corner for rectangular foundations in this release (meshes
since audit A2G-04, rings since Q38; the centre of one grid mesh in 0.15.0) and
the first vertex for building outlines; the author decided on 2026-11-14:

| Q | Question | Decision | Where |
|---|---|---|---|
| 41 | Where do the conductors the generators route to a foundation end: at the centre as in 0.15.0, at the corner of the anchor side, or at the point of the foundation nearest to the conductor? | at the nearest point, for every foundation form (rectangles, building outlines, the substation's foundation), already in 2026.11.0 | streams `f4gen`, `f5gen`, `f6gen` (audits D3A, D3B, R4A, R4B, R5A, R5B, R6A); rings and strips since Q43 and layout cables clipped at every site they pass since audits R6A-01, W1B-03, W1B-04 (streams `w1gen`, `fw1gen`) ([changelog](../changelog.md)) |


The hand-over of 2026-11-14 left three questions; the author answered them
the same evening and decided to integrate (almost) every open roadmap item
into 2026.11.0 before the release — everything scheduled for the next
release, the items left open, and the full 3-D FEM; out of scope stay the
`BranchType` export to groundinsight, the import of measurement setups from
groundmeas, the AP1 reference-case library and scaling the LU solve to
10 000 segments.

| Q | Question | Decision | Where |
|---|---|---|---|
| 42 | Route service drops around foundations (street geometry) instead of the shortest connection? | no, keep the shortest connection | no code change; [known limitations](../reference/known-limitations.md#generators-and-osm), [roadmap](../roadmap.md#status-of-the-roadmap-items-carried-in-the-changelog) (row of the TN generator) |
| 43 | Apply the nearest-point rule of Q41 also to sites earthed by a `RingElectrodeSpec` circle or a strip (the default substations)? | yes | done in 2026.11.0 (stream `w1gen`): every conductor routed to such a site ends at the point of the electrode nearest to it (a circle exactly), rods keep their head ([changelog](../changelog.md)) |
| 44 | One current per straight wire for building outlines? | yes, together with Q30 (one current per wire line in the solver) | done in 2026.11.0 (stream `w1sol`); straight runs at projected coordinates since audits W1A-01 and W1R2-05 (streams `fw1sol`, `fw1r2sol`), in linear time since audit W1R3-03 (stream `fw1r3`) ([changelog](../changelog.md)) |

The decisions D1–D9 taken while these items were implemented are listed in
[Decisions of 2026-11-15](#decisions-of-2026-11-15), before the sections of
the four waves at the end of this page.


!!! warning "Correction to question 38 (audit D3B-01)"
    The 0.9 % of question 38 was measured at the cases where the owner change
    is largest. Over the whole scope of the owner texts (short side at least
    1 m, aspect ratio up to 50, sides up to 150 m, the AP1 soils) one owner
    per side is at most 1.4 % above one current per segment (150 m × 3 m bare
    in 30 over 200 Ω·m: +1.34 %) and one owner per rectangle up to
    2.2 % (16 m × 1 m in a 0.2 m shell of 30 Ω·m over 1000 / 30 Ω·m: +2.106 /
    +2.160 % at 0.5 / 0.25 m segments);
    one per side is closer everywhere, so the decision stands.


!!! warning "Corrections (adversarial audits of the decision streams)"
    Two adversarial audits of the three decision streams below (D2A / D2B,
    fixed by the streams `f2solver` and `f2gen`; R2A / R2B, fixed by
    `f3solver` and `f3gen`) corrected these statements; the release notes
    describe the release:

    - **d2solver, Q23:** the rule also covers a more conductive layer below
      the hemisphere but within the electrodes' horizontal half-extent when
      no more resistive layer lies in between (the error is continuous
      across a_eq, −67.9 % on both sides for a 10 m ring over 1000/30 Ω·m;
      an extension by the coordinator, for the author's review). A
      horizontal wire touches the layers that hold its axis (within its wire
      radius only for the rule of an electrode below its hemisphere); a rod
      end on an interface does not touch the layer beyond. `fem` misplaces
      the current by −97 … +30 % in the rejected worlds (mostly far too
      low); more resistive spanned layers are −3 … +48 % off, not
      +3 … +45 %; none of the rejected worlds was silent before, they
      warned. `fem` also refuses strips deeper than half their length,
      rings deeper than their diameter, driven `MeshElectrode`s and
      electrodes and galvanic conductors out of the soil, and names only
      backends whose own checks accept the world.
    - **d2gen, Q20:** the brace table is measured against the bare braces of
      the development line, which were never released; in 0.15.0 the braces
      had the shell radius without the shell resistance, and against the
      0.15.0 `"distributed"` model the brace rule alone is +5.9 / +66 /
      +393 % at ρ_c = 150 / 1000 / 5000 Ω·m. The hand assembly that
      reproduces the solve is a check of the generator with the same kernel
      and owner model, not an independent method (an independent boundary-
      element model confirms the numbers; with one current per segment the
      brace effect is +49.2 / +318.5 % instead of +46.9 / +307 %).
    - **d2gen, Q21:** `concrete_model` changes the impedance of an
      axis-aligned rectangle without cross-braces (also `style="mesh"` with
      `n_x = n_y = 1`) by less than 1 % up to aspect ratio 4 and 2 % up to
      50 over the AP1 soils (short side at least 1 m; 5 mm wire at 0.8 m,
      bare or in concrete of 30 … 1000 Ω·m up to 0.2 m thick; largest values
      found 0.69 / 1.55 %), not by at most 0.16 % / 0.67 %, and the premise
      of question 21 (≤ 0.06 %) holds only for bare rings in homogeneous
      soil (0.09 % / 0.33 % measured): flagged for the author's review.
    - **d2soil, Q36:** the medians 1.8 % / 5.0 % are over all 39 AP1 soils
      with the homogeneous ones at zero (2.1 % / 5.5 % over the 36 two-layer
      soils); that the hand-over's 1.7 % / 15.7 % are this metric is an
      inference (groundfield 0.14.0 itself at 1.5 m segments reproduces
      them).

## Stream `d2solver`: User decisions Q23, Q27, Q31: fem and more conductive layers, perfect_mirror across the surface, per-engine size notice

### Changed

- ⚠ `fem` (experimental) raises `BackendCapabilityError` when the equivalent
  hemisphere of a cluster spans a layer that no electrode of the cluster
  touches and that conducts better than every layer the electrodes touch
  (user decision Q23, finding 10.23). The hemisphere injects the current
  straight into that layer and the result is too low: −97 … +5 % against
  `image_2layer` and `mom` in the measured cases (rods, strips and rings
  below a more conductive top layer from 22 mm thickness, rings above a more
  conductive lower layer; −92 % for a 10 m ring at 0.8 m over 1000/30 Ω·m at
  h₁ = 2 m). Up to now a `FemReductionWarning` or silent. The rule is
  structural (near-equal 100/120 Ω·m raises as well), uses the canonical
  layer stack, and an electrode exactly on an interface touches both layers;
  the message names the cluster, the layer, the electrodes' ρ, a_eq, the
  measured range and a backend that solves the world. More resistive spanned
  layers keep the warning (+3 … +45 %). Tests:
  `tests/test_fem.py::test_fem_rejects_a_more_conductive_layer_its_hemisphere_spans`
  and four siblings.
- `earth_inductive_model="perfect_mirror"` emits one `UserWarning` per solve
  when inductive conductor segments lie on both sides of the soil surface,
  naming the number of such pairs and the conductors and pointing to
  `"sommerfeld"` (user decision Q27, finding 10.46): its additive image has
  no physical basis for an overhead/buried pair (2.28 / 2.77 times the
  mutual reactance of `sommerfeld` at 50 Hz / 1 kHz for a lead 1 m above a
  PEN 0.7 m deep). Numbers do not change. Tests:
  `tests/test_inductive_solver_integration.py::test_perfect_mirror_warns_for_inductive_pairs_across_the_surface`
  and siblings.

### Added

- `Engine.warn_segments` (`int | None`, default `None` =
  `diagnostics.SOFT_LIMIT`, 5 000, read at solve time): the size of the soft
  notice "solve time may run from seconds to minutes" per engine, counted
  like `max_segments` (segments and nodal unknowns); honoured by the solve's
  pre-flight warning, `check_segment_resolution` and `diagnose()`
  (`report.context["warn_segments"]`). Not checked against `max_segments`
  (a notice above the hard limit never appears). Booleans rejected; kept by
  JSON, copies and the provenance (user decision Q31, finding 10.06). Tests:
  `tests/test_resource_limit.py::test_warn_segments_is_a_validated_engine_field`
  and four siblings.

### Numbers that change

| What | Size | Who is affected | Golden cases |
|---|---|---|---|
| none (Q23 turns results into errors, Q27 adds a warning, Q31 moves a notice) | — | `fem` users with a conductive layer spanned by the hemisphere get an error instead of a −97 … +5 % result | none |

## Stream `d2gen`: User decisions Q20, Q21: encased mesh braces, concrete_model deprecated

### Changed

- ⚠ **Mesh foundations: the cross-braces are encased (user decision Q20).** A
  foundation earth electrode per DIN 18014 lies in the concrete together with
  its connections, so every wire of an encased
  `FoundationElectrodeSpec(style="mesh")` — the perimeter and the cross-braces —
  carries the outer radius $r_b = r_a + t$ and the shell coefficient $C$; up to
  now the braces stayed bare, and a foundation could not exceed the spreading
  resistance of its braces alone however dry the concrete. Without a shell
  nothing changes. See *Numbers that change*. Tests:
  `tests/test_concrete_one_mechanism.py::test_mesh_braces_are_encased`,
  `::test_encased_mesh_equals_the_assembly_of_its_wires`,
  `::test_shell_of_a_mesh_foundation_spans_every_wire`.
- `groundfield.generators.materialise.concrete_shell` takes `encased_length_m`
  (the keyword `perimeter_m` of the earlier 2026.11 work is gone) and reports
  the shell resistance $C/L$ over every encased wire.

### Deprecated

- **`FoundationElectrodeSpec.concrete_model` (user decision Q21).** The field
  no longer selects a shell model; it only chooses one current owner
  (`"lumped"`, the default) or one per side (`"distributed"`) for an
  axis-aligned `style="ring"` rectangle, at most 0.16 % apart up to aspect
  ratio 4 and 0.67 % up to 50 over the soils of the AP1 grid, bare or in
  concrete of 30 to 1 000 Ω·m (0.06 % / 0.3 % for a bare ring in homogeneous
  soil). `"distributed"` keeps working and emits a `DeprecationWarning` at the
  user's line (constructor, `model_validate`, JSON, `model_copy` that sets it);
  `"lumped"` and the configuration JSON and pickles of 0.15.0 load silently;
  the generators do not repeat the warning. Drop the argument; removal no
  earlier than 2027.5. Tests: `tests/test_foundation_owners.py`.

### Numbers that change

| Q20 (10.13) | the cross-braces of an encased mesh foundation carry the concrete shell like its perimeter (DIN 18014; user decision Q20) | default house mesh (10 m, 2 × 2, 0.8 m, r_a 5 mm, t 5 cm) in 100 Ω·m −1.1 / −0.03 / +1.5 / +47 / +307 % at ρ_c = 50 / 100 / 150 / 1 000 / 5 000 Ω·m, in 30 / 100 Ω·m (h₁ = 5 m) +1.3 / +6.2 / +12 / +154 / +861 %; `medium_industry` mesh (20 m, 3 × 3) −1.3 … +298 % and +1.0 … +671 %; no golden case moves | generated mesh foundations with `concrete_rho_ohm_m` set; not worlds without a shell (the default building catalogue), not ring and building-outline foundations, not AP1 |

## Stream `d2soil`: User decision Q36: effective-resistivity presets per electrode class

### Added
- **Effective-resistivity presets per electrode class (user decision Q36).**
  `gf.effective_resistivity(soil, preset=...)` takes the weights
  $(w_0, \beta)$ from the read-only mapping
  `groundfield.soil.EFFECTIVE_RESISTIVITY_PRESETS`, with the values of the AP2
  hand-over (2026-10-20, AP1 catalogue computed with groundfield 0.14.0):
  `"lv_network"` (0.13, 0.8; the default, the AP1 calibration),
  `"station"` (0.06, 1.3; the station electrode alone) and
  `"lv_group_<g>_p<pp>"` for the four LV network groups (perimeter equivalent
  $n_U$, thresholds 20 / 100 / 250) and 5 / 10 / 20 / 40 % foundation
  electrodes (method A, q50). An explicit `w0` or `beta` replaces that weight
  only; an unknown name raises `ValueError` naming the closest presets. For
  the AP1 station in the 39 AP1 soils, $R = c\,\rho_\text{eff}$ misses the
  field solution by a median 1.8 % and at most 15.7 % with `"station"`
  (hand-over: 1.7 %, 15.7 %), against 5.0 % and 35.9 % with `"lv_network"`.
  Tests: `tests/test_effective_resistivity.py` (presets section,
  `test_the_station_preset_reproduces_the_handover_for_the_ap1_station`,
  `test_the_documented_tables_hold_the_presets`).

### Changed
- `effective_resistivity`: `w0` and `beta` default to `None`, the value of
  the preset; every call keeps its value bit for bit (default preset
  `"lv_network"` = 0.13, 0.8). The IO reference example fits the station over
  $\rho_\text{eff}$ with `preset="station"` (`max_relative` 7.48 % instead of
  13.41 %); Soil reference, ADR-0008 (amendment item 5) and glossary
  describe the presets and that a BusType and its bus must use the same one.

### Numbers that change

none (no library value moves; golden unchanged).

## Streams `f2solver`, `f2gen`, `f3solver`, `f3gen`: fixes from the audits of the decision streams

Two adversarial audits attacked the work of the three decision streams; four
fix streams answered them. Their notes were written as instructions for the
release notes, which now carry the result (the box above lists what they
corrected); in short:

- **`f2solver` (audit D2A):** one contact predicate for the layer rules of
  `fem`; Q23 within the electrodes' horizontal half-extent; documented ranges
  re-measured; a remedy only from backends whose own checks accept the world;
  electrodes out of the soil refused like in every backend; strips and rings
  deeper than their formulas and driven `MeshElectrode`s refused; conductor
  names with a line break in the `perfect_mirror` warning; which entry points
  emit the size notice.
- **`f2gen` (audit D2B):** encased mesh foundations measured against 0.15.0
  (the bare braces existed only on the development line); the `concrete_model`
  bound with its scope and rectangles without cross-braces; `model_copy`
  validates `concrete_model`; the shell helper over every encased wire; the
  hand-over's station metric marked as inferred.
- **`f3solver` (audit R2A):** only a rod is non-horizontal (round-off in a
  strip's end depths); the wire-radius band only for the rule it makes
  stricter, Q23 and the A2E-2 rule on the wire's axis again; no Q23 error for
  a conductive layer behind a more resistive one within the half-extent; ring
  depth limit at the diameter; galvanic conductors out of the soil refused;
  remedies with the setting that lifts a refusal; ranges with converged image
  series.
- **`f3gen` (audit R2B):** the `concrete_model` bound with the short side in
  both clauses, a margin over the largest values found and its model
  uncertainty; `model_copy` of subclasses; `copy.replace` warns at the
  caller's line on Python ≥ 3.13.

## Stream `d3gen`: User decision Q38: one current owner per side

Every rectangular foundation without cross-braces (`style="ring"`, or
`"mesh"` with `n_x = n_y = 1`) is one strip per side, bonded to its local −y
side, at every orientation; `FoundationElectrodeSpec.concrete_model` keeps
its validation and loads from JSON but has no effect (`"distributed"` warns,
`"lumped"` stays silent). Up to this change the default `"lumped"` made an
axis-aligned rectangle one `GridMeshElectrode`. The release notes carry the
details; in short:

- **Size:** the input impedance is never higher and unchanged for a square on
  its own. Over the AP1 soils (short side at least 1 m) the largest decreases
  are −0.32 / −0.91 % bare or with the default shell and −0.69 / −1.53 % in up
  to 0.2 m of wet concrete (aspect ratio up to 4 / 50), confirmed by the
  side-matrix algebra and an independent boundary-element solver; the AP1
  station −0.0009 … −0.0026 %.
- **Connection point and names:** conductors attached to the foundation by
  name start at the corner of its −y side instead of its centre, and the
  electrode named after the foundation is that side (`cluster_impedance` for
  the foundation), as for meshes and rotated rectangles.
- **`fem`:** the four strips are reduced as a cluster; no new rule (user
  decision Q37), the changes are listed in the `fem` row of the release notes
  and in the known limitations.
- **Golden:** only `mutual_foundations_lumped_concrete` moves (Z_G 5.2e-4,
  R 1.6 %, structure of the world), now equal to the `distributed` case.

## Stream `f4gen`: User decision Q41 and the fixes of audits D3A, D3B

Every conductor that `TnNetworkGenerator` and the layout builder route to a
foundation ends at the point of that foundation nearest to the conductor: a
service drop is the closest pair of its feeding cable and the building's
foundation wires, a cable leaving a site with a foundation starts at the
foundation's point nearest to its next node, and the conductor is attached
to the wire that holds the point. Ties go to the lowest wire index, then to
the smallest arc position. Sites without a foundation keep their connection
point, and so does everything attached by name. The first version joined a
cable that crosses a foundation to it without a drop; the audits of the
stream completed that rule (see the f5gen and f6gen sections). In short:

- **Drops:** of the 282 drops of audit D3A's ring-house set (11 968.5 m),
  47 ran along their own foundation and 45 across it; the first version had
  264 drops (10 529.6 m), none along or across its own foundation, but 18
  radial substation-slot drops on the trunk; now 246 drops (9 963.5 m), none
  on a cable (closest pair checked against 2 mm sampling).
- **Networks** (seed 1, 100 Ω·m, default grid; values after the audit
  fixes, see the f5gen section): the resistive PEN at 50 Hz moves by
  −0.005 … −0.12 % (star) and −0.03 … +0.13 % (radial); with a Neumann PEN
  at 1 kHz \|Z\| by −0.71 … +3.3 % (star) and −0.67 … +9.0 % (radial). The
  12 m × 8 m ring houses of audit D3A have +5.7 … +14.0 % more reactance
  than drops to the centre as in 0.15.0; the default 10 m ring networks
  with drops +1.2 … +2.1 % (star, 30 houses) and −10.8 … −1.1 % (radial).
- **A layout of building outlines with an AP1-like station:** −0.026 …
  −0.147 %, drops 215.9 → 140.6 m. The AP1 catalogue itself attaches its
  drops and bonds by name and is not changed by Q41 (audit R4B-14).
- **Fixes:** right angles are exact rotations and keep the anchor side
  (D3B-04, D3A-05); non-positive foundation sides raise (D3A-01);
  `diagnose()` leaves out the pairs of one foundation's wires that meet at
  its corners (D3A-04; since audit R4A-04 only wires bonded by the
  foundation's `_bond_` conductors and only pairs at a shared vertex); the
  per-side bound is 1.4 % (D3B-01, see the correction above); the texts on
  the connection point, the golden field change and the `fem` ranges are
  corrected (D3A-03, D3A-06, D3B-02, D3B-03, D3B-05 … D3B-07).
- **Golden:** `gen_tn_default` (≤ 2.9e-4), `gen_layout_builder` (≤ 1.1e-3)
  and `gen_tn_measurement_overhead_leads` (≤ 1.6e-4, the probe reading
  4.3e-3) move; the other 46 cases are bit-identical.

## Stream `f5gen`: Fixes of audits R4A, R4B (user decision Q41)

The audits of stream `f4gen` found three routing paths that broke the rule —
drops that started on the foundation of a house the cable crosses (R4A-01),
radial drops from the substation that lay on the trunk through the houses it
crosses (R4A-02, R4B-01: 6 drops over 195 m in the default 30-house network),
layout taps inside the station foundation (R4A-03) — an over-broad
`diagnose()` grouping (R4A-04), unchecked plan contacts (R4A-05), a slower
planner (R4A-06) and scoped or stale numbers (R4B-02 … R4B-14). In short:

- **Crossings:** a building that a cable crosses is joined to it and fed by
  it, without a drop: a cable of its feeding path, every trunk of the radial
  topology, and in the star topology the main that crosses it, whose cabinet
  then feeds it (coordinator decision: no cross-ties between cabinets
  through a house; 1 of 50 and 3 of 150 default houses change cabinet). No
  drop runs along a cable farther than one minimum section (1 cm): a drop
  from a node that would run along a trunk or another main starts where it
  leaves it (needed by the random stress set). The guard, with the router's
  1 cm tolerance since audit R5A-01, finds no drop along a cable in the
  default grids and audit D3A's set; in the 400 random networks one drop
  shares the 1 cm of rule 3's bound (a foundation 8 mm beyond a cabinet rod
  at the cable's end), which the guard allows.
- **Numbers, scoped** (seed 1, 100 Ω·m, default grid; `sommerfeld` and
  `carson_series`, two segment lengths): Q41 moves resistive networks by
  −0.005 … −0.12 % (star) and −0.03 … +0.13 % (radial) at 50 Hz; with a
  Neumann PEN at 1 kHz Im −3.1 … +16.3 % (star; the 10-house network with 5
  of its houses on the main +14.2 / +16.3 %) and −8.6 … +37.7 % (radial;
  default 30 houses +10.4 %). Radial networks react to where the drops end
  much more than star ones.
- **Other fixes:** `diagnose()` groups only the wires bonded by a
  foundation's `_bond_` conductors and leaves out only pairs that meet at a
  shared corner; plan contacts off their wire, unknown or on a rod are
  rejected; planning one cabinet feeding 1000 houses 3.14 → 0.89 s;
  `LvNetworkLayout.plot(world=...)` draws the drops as built; one owner up to
  2.2 %, the scanned 145 m × 2.9 m rectangle, the passive-field change
  +5.5 / −4.7 % and the provenance of the touch-voltage values corrected.
- **Golden:** nothing moves against `f4gen`.

## Stream `f6gen`: Fixes of audits R5A, R5B (user decision Q41)

The second audit round found one regression of the first round's fixes and
three smaller routing gaps — layout taps next to a rotated, polygon or
offset station or cabinet foundation were computed on the slanted first
section and refused, so the drops fell back to the station, the cabinet or
the far end of the cable, 14.0 … 36.5 m instead of 6.0 m (R5A-02); "along a
cable" was tested to 1 µm, so a drop 2 µm beside a trunk ran 60 m along it
(R5A-01); a cable touching a house at its end node and crossing it elsewhere
gave it a 1 cm drop (R5A-03); radial planning grew faster than the network
(R5A-04) — and texts that stated the pre-fix rules or unscoped numbers
(R5B-01 … R5B-13). In short:

- **Layout:** the taps stay on the layout polyline, clipped at the
  projection of the site contact onto it; no tap on the first or last
  segment inside a site foundation (a cable corner inside one stays a node,
  audit R6A-01);
  audit R5A's 40 random layouts: drops more than 1 cm longer than the
  closest pair of the built cable 30 → 6 (`e91f3c0`: 7).
- **Routing:** "along" means both ends of a cable section within one
  minimum section (1 cm) of the drop's line, the guards use the same value;
  a crossing beyond a touching node is the joint; set lookup (radial 8000
  houses with `max_feeder_length_m=1e5` 43.6 → 18.4 s, plans bit-identical). The default grids, audit
  D3A's set and the 260 random networks of the stress tests are unchanged.
- **Drops over other foundations:** 0 … 80 % of the drops of the default
  grids run 0.2 m above other houses' side wires; moved 1 m aside they change
  \|Z\| by at most 0.02 % (insulated PEN) and −0.30 % (bare PEN). Routing
  drops around foundations was left to the next release; the author then
  kept the shortest connection (user decision Q42).
- **Golden:** nothing moves.

## Verification audit R6A of stream `f6gen`

A focused third round on the fixes of stream `f6gen` reproduced them (the
layout clip of R5A-02 bit for bit against `e91f3c0` in the auditor's five
cases, the 1 cm rule changing only the 73 near-collinear drops of the
stress set, the joint search, the bit-identical set lookup, and the
sensitivity of drops over other foundations: bare PEN −0.30 %, insulated
+0.02 %, two discretisations each). It found that a router corner inside a
small cabinet ring stays a node that houses tap (R6A-01: 82 of 2 540 drops in
200 random layouts, also before user decision Q41) — documented as a known
limitation and scheduled in the roadmap — and five scoping and wording
points (R6A-02 … R6A-06: an evidence line of the stream's report, the
`max_feeder_length_m=1e5` of the 8000-house timings, the definition of
"along" in the 10.38 row, the frequency of the detour reactance, the
substation drop in tutorial T06), which the release notes, the known
limitations and the tutorial now state.

## Decisions of 2026-11-15

While the items of the user decision of 2026-11-14 were implemented, eight
questions came up (D1–D8), each with a recommendation. On 2026-11-15 the
author followed every recommendation and asked, for D6, to stay with
licences that allow further commercial use and are as free as possible.
During the waves he decided two more: the form of the far images of the
two-layer image series and D9. The table keeps the question, the
recommendation that is now the decision, and where it was done.

| id | Question | Decision | Status and where |
|---|---|---|---|
| D1 | Give `VoltageSource` an internal impedance Z_q = R + jωL (two fields, one diagonal entry of the source's constraint row)? Without it: a junction with a conductor, or the 1 A solve scaled by hand. | yes (small, useful for AP2) | done in 2026.11.0 (stream `fw2src`): `internal_resistance_ohm`, `internal_inductance_h`, 0 by default and then bit-identical to the ideal source; pickles written before it load as ideal sources, and a galvanic `mom` world factorises once for all frequencies (audits W2R2-01, W2R2-02, stream `fw2r2`) ([changelog](../changelog.md)) |
| D2 | Make the near/far switch of the line kernel continuous (a deterministic margin or a far form exact to round-off), so that results do not depend on where a world stands? It moves worlds near the origin, estimated at about 2e-7 (audit W2B-02: 1.15e-5 for the switch alone in golden `line_img2_rod_crosslayer`). | yes, in stream `w3xl` (the numbers move far below every tolerance; reproducibility wins) | done in 2026.11.0 (stream `w3xl`, item 5): a C¹ blend between 6 and 7.5 segment lengths; 1e-13 over local shifts, ≤ 5e-11 at UTM coordinates (audit W3A-04); with the far images and the cross-layer line form of the same stream the golden line-kernel cases move by up to 6.0e-4 ([changelog](../changelog.md)) |
| D3 | A current that varies along each wire in the image family (stars +1.8 … +10.4 %, 8 arms in 1000/100 Ω·m +12.3 %; grids +0.36 … +0.70 % above one current per segment)? | roadmap (`mom` is the per-segment answer today) | open: [roadmap](../roadmap.md#left-open-by-2026110) |
| D4 | The generator rules wave 1 left open: a group rule for wide groups (about 180°), one cabinet around the substation, a cabinet that cannot keep its clearance (it raises; it could move on along its ray)? | roadmap; the raise is a clean failure | open: three items of the [roadmap](../roadmap.md#left-open-by-2026110) |
| D5 | Worlds that mix projected (UTM) and local coordinates are not helped by the shifted local frame of stream `w2mom`: what to do about them? | document them, once audit W2B has quantified them | quantified by audit W2B-04 and done in 2026.11.0 (stream `fw2mom`); by the coordinator's decision on W2B-04 beyond the text: `diagnose()` reports such a world under `coordinates`, and its solve warns with `MixedCoordinatesWarning` ([known limitations](../reference/known-limitations.md#scope-of-the-physics), [changelog](../changelog.md)) |
| D6 | Dependencies of the 3-D FEM as an optional extra of the MIT package (meshers: gmsh GPL-2.0-or-later with exceptions, netgen LGPL, TetGen AGPL; `scikit-fem` BSD-3, `pyamg` MIT)? | permissive licences only (MIT / BSD / Apache): `scikit-fem` with its own tensor-product meshes and `pyamg`; no GPL / AGPL; a weak-copyleft mesher only if nothing permissive works, and then ask again | done in 2026.11.0 (stream `w4fem3d`): the optional extra `fem3d` with scikit-fem (BSD-3-Clause) and pyamg (MIT); the design needs no surface-fitted mesh, so no external mesher and no copyleft package ([changelog](../changelog.md)) |
| D7 | A continuous cabinet rule instead of the 20 m threshold, which moves a cabinet by up to 142 m for a 0.2 m change of the house spacing (X at 1 kHz about −20 % in the 50-house grid; AP1 mixes with groups at 18.8 / 20.2 / 20.7 m; audit W1R2-02)? | keep the threshold for 2026.11, document the step and record the rule per cabinet; the AP1 catalogue does not use the generator's cabinets | done in 2026.11.0 (stream `fw1r2gen`): the plan records `PlannedSite.cabinet_rule` and `PlannedSite.centroid_ray_gap_m` per cabinet, the [known limitations](../reference/known-limitations.md#generators-and-osm) give the step (141.8 m, X at 1 kHz −21.6 / −18.9 % with `carson_series` / `sommerfeld`); a continuous rule is on the [roadmap](../roadmap.md#left-open-by-2026110) |
| D8 | Rebase the golden baseline for 2026.11.0 after the last wave, since the declared bounds of many streams add up and no longer bind (audit W1R2-07)? | yes, as the last step before the release dry run | pending: part of the release preparation after wave 4 |
| D9 | Service drops that leave one tap in one direction (the radial taps, the substation drops of the star) lie on one axis, where the inductance matrix reads every pair as touching (`diagnose()`: 10 overlapping pairs in the 5-house radial network, 247 with 200 houses): keep them, or lay them side by side? | in a row side by side at the touching distance; only neighbours touch | done in 2026.11.0 (stream `fw3perf`; along one common normal under placement jitter since audit W3R2B-01, stream `fw3r2b`): `PlannedBranch.axis_offset_m`; X at 1 kHz of the 5-house radial network −2.33 / −2.70 % (`carson_series` / `sommerfeld`); no generated network reports an overlap for `jitter_m` 0 … 0.1 m; no golden case moves ([changelog](../changelog.md)) |
| far images | Which form do the far images of the two-layer image series take (stream `w3xl`, item 1)? | the quadrupole form stays the default (the reference model); an `Engine` option gives the point form for fast parameter studies | done in 2026.11.0 (stream `fw3xl`): `Engine.image_far_field = "quadrupole" \| "point"`; the TN 40-house network in 1000/30 Ω·m with h₁ = 1.5 m solves in 0.49 of the default's time; since audit W3R2A-01 (stream `fw3r2a`) the point form keeps the near image pairs on the cross-layer path, and a 75 m strip below the interface is within 1.1e-5 of its exact value at any segment length (up to −58 % before) ([changelog](../changelog.md), [known limitations](../reference/known-limitations.md#current-ansatz-and-source-kernels)) |

## Waves 1 to 4 of the extended release

On 2026-11-14 the author moved (almost) every open roadmap item into
2026.11.0 (see the decisions above). The work ran in four waves, each with
implementation streams on disjoint files, the coordinator's texts, a first
round of two adversarial audits, fix streams with the coordinator's decision
per finding, and a second audit round that attacked the fixes, again
followed by fixes; wave 1 had a third, verifying round, and the rounds of
wave 4 after audit W4A follow at the release. The audits number their
findings per round (W1A-01 … W4A-11). Wave 1 with its first fixes and audit W1R2, and the wave-2 streams,
ran in the cloud workspace (Linux x86-64, OpenBLAS); the rest ran on the
author's Mac (macOS arm64, Accelerate), where every claim of bit identity is
made against a golden baseline generated on the same machine. As for the
streams above, the release notes carry the result; the sections below say
what each stream and each audit round did, with the numbers as corrected by
the later rounds.

## Wave 1: one current per straight wire, concentric cables, cabinets per building group

Wave 1 implemented the user decisions Q30 / Q44 and Q32 / Q33 in the solver
and the coupling (stream `w1sol`), and Q22 / Q35, Q43 and the layout fix of
audit R6A-01 in the generators (stream `w1gen`); Q42 needed no code. Its
audits ran three rounds: W1A / W1B, W1R2 on the fixes, and the verification
round W1R3.

### Stream `w1sol`: one current per straight wire, concentric cables by marker (Q30, Q44, Q32, Q33)

`image`, `image_2layer` (with its cross-layer path) and `solve_mutual_*`
give every grid one uniform-current owner per full wire line, a star one per
arm and a polyline one per straight run; the owners of an electrode are
ideally bonded, every result stays per electrode, and
`metadata["wire_owners"]` counts them. Rods, strips and rings stay one wire;
generated mesh foundations, one electrode per wire line already, do not
change. In the coupling, a thinner conductor on the axis of a thicker one is
a core inside a shield only when the thicker one has
`conductor_type="cable_shield"`, and a shield has no internal inductance of
a solid wire. In short:

- **Grids:** −1.2 % (10 m, 2 × 2 meshes) … −4.0 % (40 m, 8 × 8) … −5.5 %
  (100 m, 20 × 20) in 100 Ω·m; above one current per segment +0.36 … +0.70 %
  instead of +1.6 … +4.9 %. Against `mom`, at the corner of the 20 m grid
  with 4 × 4 meshes: touch voltage +6.4 % (+20 % before), step voltage
  −7.6 % (−21 %); the touch-voltage envelope of the 10.21 note 111.44 →
  98.72 V.
- **Outlines and stars:** a building outline is one owner per side and
  equals the rectangle given by its size (rectangle outlines −0.000 …
  −0.31 % bare, −0.62 % in wet concrete; the T06 layout with the AP1 station
  −0.03 … −0.40 %). A star alone does not change (equal arm currents); its
  gap to one current per segment stays on the roadmap (decision D3). A
  source-free grid, star or polyline floats at one potential with exactly
  zero net current.
- **Concentric cables:** two unmarked conductors on one axis are separate
  cables that touch (a 100 m pair of 5 mm and 15 mm radius: loop 31.97 →
  43.47 µH, the analytic 43.48 µH); the self terms of a shield drop μ₀ℓ/(8π)
  (50 nH/m), and the 100 m coax reads the analytic 26.97 µH instead of
  31.97 µH. `buried_lead()` is a shield: two-pole readings with buried leads
  −2.5 % in reactance.
- **Golden:** `img2_grid_upper_resistive` (Zc −0.94 %),
  `cim_grid_unequal_segments` (−1.41 %), `img_mixed_cluster_hom` and
  `line_img_mixed_cluster_hom` (−0.20 %, electrode currents up to 3.4 %)
  move; Q32 / Q33 move no stored case. The refinement bound of benchmark B11
  is 0.6 % instead of 0.5 %.

### Stream `w1gen`: cabinets per building group, the nearest point on rings and strips (Q22, Q35, Q43, R6A-01)

The star topology of `TnNetworkGenerator` orders the buildings by bearing
around the substation, from the widest angular gap on, cuts them into one
run of consecutive bearings per cabinet (sizes differ by at most one) and
places each cabinet on the ray through its group's centroid, 1 m
(`CabinetConfig.end_clearance_m`) beyond the group. In the radial topology a
building in a substation slot taps its feeder's trunk (Q35). Every conductor
routed to a site earthed by a ring or a strip, the default substations
among them, ends at the point of that electrode nearest to it (Q43), and a
layout cable reaches a site where its polyline enters the site foundation
(R6A-01). In short:

- **Default networks:** the 30-house star had both cabinets west of the
  substation and fed its eastern houses by drops of 3.5 … 106.5 m from the
  substation (25 drops, 1 140 m); its cabinets now stand at (−118.5, 0) and
  (118.5, 0) m with 20 drops of 25 m. The 30-house radial network: drops
  1 115 → 778 m, 14 → 4 of them from the substation. The default mains
  leave the substation ring at its point nearest to their cabinets.
- **Numbers** (default grids of 10 … 150 houses, 100 Ω·m, with the fixes of
  `fw1gen`): resistive PEN at 50 Hz star +0.12 … +0.85 %, radial −0.01 …
  +0.49 % (Q43 alone −0.05 … −0.13 %); an isolated Neumann PEN at 1 kHz (up
  to 50 houses) star \|Z\| −1.4 … +13.1 %, radial −1.2 … +15.4 %. The
  reactance of the default 30-house star rises by +64 / +55 %
  (`carson_series` / `sommerfeld`), because its houses share their main
  instead of parallel drops (audit W1B: the rise comes from the tree, +52 %
  in an own lumped model, +54.9 % with `mom`).
- **Golden:** `gen_tn_default` (substation potential +1.18 %, the current of
  the cabinet rod −62.7 %: the rod stands 1 m beyond the houses instead of
  62.5 m) and `gen_tn_measurement_overhead_leads` (+1.76 %, the probe rod's
  potential +117 %) move; the other 47 cases are bit-identical.

### Audits W1A and W1B of the wave-1 streams

Two auditors attacked the merged wave 1 (snapshot `6f63deb`, change from
`e6124cd`) with code of their own, W1A the solver and coupling side, W1B the
generators. Findings:

- W1A-01 (medium): the absolute collinearity tolerance (a sine of 1e-9)
  split the wire lines of rotated generated meshes at UTM coordinates into
  extra owners (a 10 m mesh with 10 × 10 meshes: 142 extra owners, −0.55 %;
  up to −0.64 %).
- W1A-02 (medium): the coax loop inductance shrank with the number of pieces
  (26.966 µH lumped, 24.608 µH in 0.25 m pieces, −8.8 %).
- W1A-03 (low): the 10.21 row still gave 69.6 V at +x; since Q30 it is
  59.95 V (point kernel), 61.31 V (line kernel), 61.58 V (`mom`).
- W1A-04 (low): the owner texts of the generators described one leakage
  current per electrode.
- W1A-05 (low): `model_copy` bypassed the rule that a shield has no skin
  model.
- W1A-06 (low): `diagnose()` counted one owner per electrode (102 instead of
  393 unknowns).
- W1A-07 … W1A-10 (info): a lone floating split electrode reported
  `-0-0j`; the roadmap had lost the star item and three decision rows of this
  page were stale; two documentation points on shields.
- W1B-01 (medium): a group spanning about 180° got its cabinet in empty
  ground (10 houses and one small commercial building: 50.4 m from the
  nearest building, drops 527 instead of 337 m).
- W1B-02 (medium): the clearance looked only at the group's horizontal
  wires; a cabinet rod stood at the point of a building's rod.
- W1B-03, W1B-04 (low): the R6A-01 clip covered only a cable's own end
  sites, and its merge cut off cable legs that serve houses (drops 30.0 →
  142.8 m).
- W1B-05 … W1B-08 (low): `end_clearance_m` without a lower bound, golden
  bounds wider than their quantities and adding up across streams, stale
  texts, tests that did not check what they promised.

Confirmed: one owner per wire line equals bonded strips to 7e-15 in every
engine path; an independent grid code gives the owner gaps (20 m 4 × 4 grid:
+3.297 % with one owner, +0.574 % with one per wire; stars +1.81 / +3.48 /
+10.54 %); 54 records outside the items are bit-identical; 26.97 µH and
43.48 µH follow from closed forms; over about 300 layouts every PEN network
is one tree, every building is fed once and every drop is the closest pair;
ring contacts lie on the circle within 2.5e-13 m. Taken up by `fw1sol`
(W1A except -04 and -08), `fw1gen` (W1B and W1A-04) and the coordinator
(W1A-08: the roadmap item and the rows).

### Streams `fw1sol` and `fw1gen`: fixes of audits W1A and W1B

- **`fw1sol` (W1A):** a vertex continues a straight wire when it lies on the
  chord of its neighbours up to 64 units of round-off of the coordinates
  (`COLLINEAR_ROUNDOFF_ULPS`), so generated meshes at UTM coordinates read
  their local value again (+0.09 … +0.64 %). Pieces of conductors lying
  beside each other add up: a thinner wire parallel inside a shield couples
  at the shield radius at every axial offset, and the coax reads 26.97 µH in
  any parallel pieces (it read 26.91 / 26.37 / 24.61 µH in 10 / 1 / 0.25 m
  pieces), the unmarked pair 43.47 µH. A copied skin shield is rejected at
  the solve, `diagnose()` counts 393 unknowns, a lone floating electrode
  reports `0j`, and the 10.21 note gives the touch voltage after Q30.
  Golden: `cond_drops_along_pen_main_perfect_mirror` moves (X at 1 kHz
  +3.9e-4, Zin 1.3e-5).
- **`fw1gen` (W1B, W1A-04):** a cabinet stays on its centroid ray,
  `end_clearance_m` beyond the group's farthest electrode (rods included)
  plus the plan extent of its own grounding, unless it would stand there
  farther than `max_far_end_gap_m` (20 m) from the nearest electrode of its
  group; then it stands on the ray through the group's far-end building.
  This hybrid rule is the coordinator's: the stream's first version put
  every cabinet on the far-end ray, which gave the 50-house grid six drops
  from the substation. The case of W1B-01 has its cabinet 1.56 m from the
  nearest building and 311.8 m of drops. A cabinet that cannot keep its
  clearance to another site raises, naming both; `end_clearance_m` is
  positive. Layout cables are clipped at every site they pass, and only the
  leading and trailing waypoints inside the end sites are merged.
- **Numbers (`fw1gen`):** resistive PEN at 50 Hz of the default star grids
  of 100 … 200 houses −0.031 … +0.089 %, the mix of 10 houses and one small
  commercial building +0.489 %; the wide mixes with an isolated Neumann PEN
  at 1 kHz \|Z\| +0.3 … +1.9 %; layouts with a cable through a passed
  cabinet −0.16 … −0.45 %. The 10-, 30- and 50-house grids, the radial
  networks and every golden case are unchanged.

### Audit W1R2: round 2 on the wave-1 fixes

The second round attacked the fixes of `fw1sol` and `fw1gen` (snapshot
`1579ed8`). Findings:

- W1R2-01 (medium): the rule of W1A-02 acted only on pieces parallel within
  4.47e-5 rad; at 4.5e-5 rad the 100 m coax read −0.17 … −7.42 % again in
  10 … 0.25 m pieces, with a step at the tolerance.
- W1R2-02 (medium): the 20 m threshold switches a cabinet between two
  positions up to 142 m apart (the 50-house grid with its house columns
  32.0 / 32.2 m apart: X at 1 kHz about −20 %); the texts said it separates
  the group populations, which it does not (decision D7).
- W1R2-03 (low): the bundle rule evaluated all piece pairs at once (two
  600 m conductors: 30 → 292 MiB).
- W1R2-04 (low): texts against the measurements (the AP1 gaps are
  20.21 … 53.56 m, not 23.7 … 50.4 m; thresholds of 15 / 25 m move two groups
  of the 100-house grid, not one; a timing of the first version).
- W1R2-05 (low): straight runs were judged per vertex triple, so a
  near-duplicate vertex or a finely sampled arc counted as straight at UTM
  coordinates.
- W1R2-06 (low): rows 22 and 35 of this page still said "next release",
  and five rows linked a roadmap section that no longer listed them.
- W1R2-07 … W1R2-10 (info): the golden bounds of several streams add up
  (decision D8); without conductor labels unmarked touching pieces keep the
  old drift; `end_clearance_m ≥ max_far_end_gap_m` silently sent every group
  to the far-end ray; `World.check()` let a copied skin shield through.

Confirmed: the coax and the unmarked pair add up exactly at local and UTM
coordinates (1.6e-15), the W1A regressions and the ten W1B tests pass, the
layout clip is clean in 17 hand-made attack cases, and the cabinet rule is
deterministic and raises nowhere over the 80 AP1 mixes. Taken up by
`fw1r2sol` (01, 03, 05, 08, 10), `fw1r2gen` (02 with D7, the code texts of
04, 09) and the coordinator (the other texts of 04, 06; 07 by D8).

### Streams `fw1r2sol` and `fw1r2gen`: fixes of audit W1R2

- **`fw1r2sol`:** pieces of two conductors tilted against each other get the
  bundle rule, with the stretches inside, beside and away from the partner
  found exactly (the length gate and the angular cap of this version went
  in `fw1r3`); the tilted coax returns to its lumped value (+0.004 …
  +8.0 %). The generator's overhead probe lead, drawn from 10 mm beside the
  feed lead to its axis, is such a pair: the probe junction potential moves
  by 6.5e-5 … 1.1e-4. The bundle rule needs memory in proportion to the
  matrix (292 → 30 MiB, bit-identical). Straight runs grow against the chord
  of the whole run (a 2° arc electrode of R = 50 m at UTM coordinates
  +1.27 % → +5.5e-4 against the local origin). `World.check()` rejects a
  copied skin shield. Golden: nothing moves.
- **`fw1r2gen` (D7):** the 20 m threshold stays and the step is documented
  with its measured size (the 50-house grid: 141.8 m, X at 1 kHz with an
  inductive PEN −21.6 / −18.9 %, `carson_series` / `sommerfeld`); the plan
  records per cabinet which rule placed it and the gap it compared
  (`PlannedSite.cabinet_rule`, `PlannedSite.centroid_ray_gap_m`). Over the
  80 AP1 mixes 118 groups stand 20.21 … 53.56 m from their houses on the
  centroid ray and take the far-end ray; the largest gap kept is 18.84 m.
  `end_clearance_m` must be smaller than `max_far_end_gap_m`. Golden:
  nothing moves.

### Verification audit W1R3 of the W1R2 fixes

A third round checked the fixes of `fw1r2sol` (snapshot `8c83d7c`). Its own
integration of the documented readings agreed with the code on 1740 piece
pairs to 5.5e-12 wherever the code claims a reading. Findings:

- W1R3-01 (medium): the length gate of the tilted rule kept the old drift
  for longer pieces: a core crossing the shield's axis stepped by −8.6e-5 of
  the loop at the gate, shield pieces of mixed length −1.6e-3, where the
  texts said about 1e-6.
- W1R3-02 (low): the cap \|cos φ\| ≥ 0.99 made a new step of 1.5e-3 at 8.1°
  for two wires leaving a junction in 0.02 m pieces.
- W1R3-03 (medium): the chord rule took quadratic time: 4000 collinear edges
  13.3 s (0.085 s before W1R2-05), a 2 km polyline electrode in 1 m vertices
  17.7 s (0.46 s).
- W1R3-04 (low): a row of *Numbers that change* said "more than" 0.46 µm,
  where the rule changes polylines whose vertices lie closer than that.
- W1R3-05 … W1R3-07 (info): correct but undeclared moves (wires leaving a
  junction at 3° in fine pieces +0.098 / +0.75 %, drops routed back along a
  split main +6.8e-5 … +1.65e-3); the partition into runs depends on the
  vertex order at UTM coordinates (2.4e-8); the `carson_series` gap between
  a lumped tilted pair and its pieces grows with the square of the tilt
  (4.2e-4 … 4.4e-3 in X at 2e-3 … 8e-3 rad).

Taken up by `fw1r3` (01, 02, 03; 06 measured), the coordinator's texts (04,
05) and stream `w3perf` of wave 3 (07).

### Stream `fw1r3`: fixes of audit W1R3

- **Tilted rule:** no length gate and no angular cap; pairs within 1e-9 of
  orthogonal count as orthogonal. Every case of the audit adds up within
  2.2e-12, the tilted coax within 6.4e-12 for tilts up to 1e-3 rad (an
  unmarked touching pair within 3.1e-10), and the solved loop within 1.1e-15
  under `perfect_mirror` (`sommerfeld` within 1.6e-8, its own piece
  dependence).
- **Chord rule in linear time:** the decisions are the same bit for bit, and
  the runs are cached on the vertices' values: 4000 collinear edges 0.006 s,
  the 2 km polyline electrode 0.25 s instead of 17.3 s. The order dependence
  of W1R3-06 stays at 2.37e-8.
- **Golden:** strict against the stream's own baseline, 49 cases pass; of a
  probe of 178 arrays 177 are bit-identical, and two wires leaving one point
  at 30° in 0.05 m pieces move from −1.04e-4 to −7.8e-16 off their lumped
  pair. Every generated network is bit-identical.

## Wave 2: impressed voltage sources, shells in `mom`, projected coordinates

Wave 2 added impressed voltage sources (stream `w2src`, finding 10.51, the
open item "voltage sources" of the scope of 2026-11-14) and concrete shells
in the moment method, together with solves at projected coordinates that do
not depend on where a world stands (stream `w2mom`, an open point of
`fw1sol`). Decisions D1 and D5 came with its fixes.

### Stream `w2src`: impressed voltage sources (10.51)

A `VoltageSource` holds `attached_to` at U above `return_to` (or remote
earth) and is solved by modified nodal analysis: one current unknown and one
constraint row per source in the nodal system of the shared pipeline, in
`image`, `image_2layer` (also on the cross-layer path), `mom` and
`mom_sommerfeld`, with conductors, inductive coupling and several
frequencies. Up to 0.15.0 every backend dropped it with a warning and solved
to zero potentials. In short:

- **Results:** `SourceRecord.current` holds the solved current (new
  `SourceRecord.voltage` and column `voltage_v`), `input_impedance()` is
  U/I, and `sweep`, the rho-f helpers, `split_factor` and
  `touch_voltage_spectrum` accept the source; several sources superpose and
  0 V is an ideal short; new capability `BackendCapabilities.voltage_sources`.
- **Checks:** voltage sources that close a loop through ideal bonds or remote
  earth, and a source short-circuited by an ideal bond, raise `ValueError`
  naming the sources; `fem` rejected voltage sources with
  `BackendCapabilityError` until its conforming models of wave 4, which
  solve them.
- **Validation:** the voltage solve equals the 1 A current-source solve
  scaled by U/Z_in in every backend, in two- and three-layer soil, on the
  cross-layer path and with inductive conductors at 0 / 50 / 1000 Hz (at most
  1.4e-15), and agrees with `solve_mutual_matrix` for a two-terminal source,
  the hand-inverted transfer matrix of three sources, Dwight's rod (−0.012 %)
  and the hemispherical far field.
- **Golden:** no stored quantity moves; worlds without voltage sources are
  bit-identical.

### Stream `w2mom`: concrete shells in `mom` and `mom_sommerfeld`, solves at projected coordinates

- **Shells:** each segment adds its radial shell resistance C/Δs to the
  diagonal of the point-matching system and the shell drop to its potential;
  with one owner per segment the image family solves the same system
  (7e-16), and a ring rises by exactly C/(2πr) in one, two and three layers
  (1e-14). Up to then both engines raised `BackendCapabilityError`; `fem`
  remained the only backend that refused shells, until wave 4. `mom` is
  thereby the
  per-segment reference for encased foundations: the image family's point
  kernel lies +0.35 … +0.69 % above it for grids in 5 cm of wet concrete and
  up to +18.8 % for stars.
- **Projected coordinates:** a world with a horizontal coordinate beyond
  10 km is solved in an exactly shifted local frame and reported in user
  coordinates, also by `FieldResult.potential`, and the segment counts allow
  the round-off of the coordinates' own magnitude (64 ε\|x\|): a generated
  10 m mesh foundation at UTM coordinates had 508 instead of 440 segments,
  and `mom` / `image(kernel="point")` moved by 2.6e-4 / 2.8e-4, now 3e-12;
  the line kernel stayed within its own spread (1.5e-6; ≤ 5e-11 since
  decision D2). A distributed conductor whose length is a whole number of
  pieces up to round-off gets no extra piece (a 60 m counterpoise at UTM
  coordinates 7 → 6 pieces).
- **Golden:** nothing moves; the AP1 catalogue is not affected. Three open
  points went to the audits: the knife edge of the line kernel's switch
  (D2), worlds that mix coordinates (D5) and the brace effect with `mom`.

### Audits W2A and W2B of the wave-2 streams

Two auditors attacked wave 2 (snapshot `ecff7b6`, change from `1579ed8`),
W2A the voltage sources, W2B the shells and the frame. Findings:

- W2A-01 (medium): `LvNetworkLayout.measured_grounding_impedance` and
  `verify_current_balance` fell back to 1 A for a voltage source (5.604
  instead of 1.635 Ω, balance `ok=False`).
- W2A-02 … W2A-09 (low): `compare_engines` left out the input impedance of
  voltage sources; the alias `_warn_ignored_sources` lost its `stacklevel`
  (`TypeError` in every world); a voltage record without voltage and current
  gave `[]` or `IndexError`; "equals the scaled 1 A solve" holds for one
  source only (two sources −7.5 %); the messages of a same-node voltage
  source and of a loop; a direct `solve_fem` and the shell refusal of `fem`
  named the wrong capable backends.
- W2A-10, W2A-11 (info): `source_kind="voltage"` with a measurement set-up
  solved, but `measured_impedance` refused it; a bullet of this page and
  notebook 83.
- W2B-01, W2B-02 (medium): the near/far switch of the line kernel is a knife
  edge: two-layer image pairs on it spread Z by 1.0e-5 … 2.6e-5 and the touch
  voltage by up to 5.5e-5 with the position of a world (the texts said up to
  2.4e-7), also through the series-image switch, whose far form is the point
  form; the estimate of D2 (about 2e-7) was wrong (golden
  `line_img2_rod_crosslayer` 1.15e-5).
- W2B-03 (medium): "worlds within 10 km unchanged" is false for distributed
  conductors: a 60 m / 10 m counterpoise goes 7 → 6 pieces in 876 of 3600
  directions, Z −0.30 % (`image`) and −2.54 % (`mom`).
- W2B-04 (medium): decision D5 quantified: the round-off grows only from
  1.0e-11 to 1.5e-11 (point kernel), but nothing warns; a return electrode
  given at local coordinates raised Z_in by 0.43 %, a cable end given locally
  made a 33 118 km conductor (7.5 kΩ instead of 5.4 Ω), and `sommerfeld` did
  not finish.
- W2B-05 (medium, pre-existing): one lumped Neumann conductor under
  `sommerfeld` took 0.42 / 4.3 / 28 s and 116 / 364 / 1647 MB at 0.25 / 1 /
  2 km.
- W2B-06, W2B-07 (low, texts): the brace effect with `mom` and the
  `medium_industry` values were missing from the concrete-shell page;
  "`mom_sommerfeld` equals `mom` to 4e-9" holds for one soil only (1.4e-9 …
  1.5e-8 over four).
- W2B-08 (info): the frame shift is exact only for coordinates between o/2
  and 2o, o being the frame's origin; elsewhere it is within 1.9e-9 m.

Confirmed: the modified nodal analysis matches the scaled 1 A solve and
independent closed forms to 1.1e-15 in every backend, layering, terminal kind
and frequency tested; worlds without voltage sources are bit-identical (563
of 563 golden quantities) and cost the same; the shell term and the frame
agree with two independent methods each to round-off (the rise C/(2πr) of a
ring to 5.6e-15; 19 worlds at 10 offsets: Z within 2.6e-11). Taken up by
`fw2src` (W2A-01 … -10, with D1), `fw2mom` (the texts of W2B-01, -03, -04
with D5, -06 … -08), stream `w3xl` of wave 3 (W2B-01, -02 by D2), stream
`w3perf` (W2B-05) and the coordinator (W2A-11).

### Streams `fw2src` and `fw2mom`: fixes of audits W2A and W2B, decisions D1 and D5

- **`fw2src` (D1):** `VoltageSource.internal_resistance_ohm` and
  `internal_inductance_h` (≥ 0, 0 by default: the ideal source, bit for bit)
  and `internal_impedance(f)`; Z_q is the diagonal entry of the source's
  constraint row in every backend that solves voltage sources, and
  `input_impedance()` is the impedance the source sees, U/I − Z_q. A galvanic
  world whose source has an inductance is solved per frequency on the
  complex path. Validated against the junction +
  `lumped_series_resistance_ohm` pattern (≤ 7.2e-16), the 1 A solve with
  I = U/(Z_q + Z_in) in every backend, three earth models and the
  cross-layer path (≤ 4.0e-15), and the transfer matrix of three sources and
  `solve_mutual_matrix` (≤ 3.4e-16). A source with R > 0 opens a loop of
  voltage sources; an inductance alone does not.
- **`fw2src` (W2A):** the layout readings use the solved current (1.635 Ω);
  a record without voltage and current raises `ValueError`; `stacklevel` is
  restored; the loop message names the sources on the loop, and a same-node
  voltage source is reported as short-circuited; `compare_engines` compares
  the input impedance of voltage sources; `source_kind="voltage"` gives the
  readings of `"current"`. Golden: 563 of 563 quantities identical; worlds
  with ideal voltage sources bit-identical.
- **`fw2mom` (D5, W2B):** `diagnose()` has the category `coordinates` (a
  warning when points lie both within 10 km of the origin and beyond 1000 km
  from it, naming both groups and the conductors between them; information
  for a world wider than 100 km that does not mix), and such a solve warns
  once with `MixedCoordinatesWarning` before anything is discretised. The
  texts give the measured line-kernel spread (W2B-01), the local piece
  counts with W2B's pin test (W2B-03), the brace effect with `mom`
  (+49.6 / +321.2 % for the house mesh, +49.5 / +313.7 % for
  `medium_industry` at ρ_c = 1000 / 5000 Ω·m; the owner model with bare braces
  1.6 / 2.8 % and 2.2 / 3.6 % high), the agreement of `mom_sommerfeld` with
  `mom` (about 1e-8), and where the frame shift is exact. Golden: 563 of 563
  quantities identical.

### Audit W2R2: round 2 on the wave-2 fixes

The second round attacked the fixes of `fw2src` and `fw2mom` (snapshot
`34ad269`). D1 holds (the constraint row, its sign and the per-frequency
complex path agree with independent methods to round-off; ideal sources stay
bit-identical), the W2A fixes hold, and D5 raised no false alarm in eleven
projected and local coordinate systems while it caught every stray node or
conductor end of a UTM world; worlds outside the items are bit-identical
(563 of 563). Findings:

- W2R2-01 (low): worlds and results pickled before D1 raised
  `AttributeError`.
- W2R2-02 (low, cost): a galvanic `mom` world whose source has an inductance
  got one complex factorisation per frequency (×2.0 / ×8.1 / ×28 at 1 / 5 /
  20 frequencies, 1488 segments).
- W2R2-03 (low, text): the stated direct-pair spread of the line kernel was
  exceeded (a 12 m strip 5.9e-7 / 1.15e-6).
- W2R2-04, W2R2-05 (low): the message of a short-circuited source with
  R > 0; `measured_impedance` raised "2 sources" for a result with one
  voltage and one current source, which read the current source before.
- W2R2-06 … W2R2-11 (info): "never flagged" is untrue for wide pure worlds;
  a TN network with its substation at UTM coordinates builds a mixed world
  (D5 warns, correctly); "876 of 3600" depends on how the angles are formed
  (829); `mom_sommerfeld` against `mom` is 9e-8 with the interface 0.2 m
  below the wires and 9.2 % with a truncated series (both warned); a large R
  loses log10(R/Z_E) digits, and 1e-310 Ω gave `nan`; what the D5 rule does
  not see (a stray point between 10 and 1000 km, field and probe points).

Taken up by `fw2r2`, the known-limitation texts through the coordinator.

### Stream `fw2r2`: fixes of audit W2R2

- **Pickles:** worlds and results pickled before D1 load as ideal sources
  (W2R2-01).
- **Cost:** a galvanic `mom` / `mom_sommerfeld` world whose source has an
  inductance factorises its segment system once and solves only the V × V
  system of the source currents per frequency (exact): 1.02 times the solve
  without the inductance at 20 frequencies on 1 488 segments, 28 times
  before; results equal to 3.3e-15 (W2R2-02; audit W3B: within 1.4e-14 of
  the former path, ×1.04 … ×1.05).
- **Messages and readings:** a short-circuited source behind Z_q is told
  that its current U/Z_q would circulate through the bond and the source;
  `measured_impedance` and the layout readings take the single current
  source, else the single voltage source, as up to 0.15.0 (W2R2-04, -05). A
  non-zero R or L below the smallest normal float is rejected (W2R2-10), and
  `SubstationConfig.position` is documented as local (W2R2-07).
- **Texts:** the line-kernel spread, the scope of D5, the 829 … 876
  directions and the agreement of `mom_sommerfeld` (W2R2-03, -06, -08,
  -09); the text on what D5 does not see (W2R2-11) went to the known
  limitations.
- **Golden:** 563 of 563 quantities identical.

## Wave 3: the two-layer image series, earth return and soil

Wave 3 took the roadmap items of the two-layer image series and the
cross-layer path with decision D2 (stream `w3xl`), and those of the earth
return, the soil and the BusType document with the user decisions Q34 and
Q39 (stream `w3perf`). The author decided the far-image option and D9
during its fixes.

### Stream `w3xl`: far images, the cross-layer path in line form, memory, the continuous switch (D2, 10.28, 10.56)

Five items. In short:

- **Far images (item 1):** far Tagg/Sunde images carry the quadrupole term
  of the segment average, and the far form of segment pairs the radius term
  of the reduced kernel; the term is tapered off between 60 and 120 segment
  lengths of the pair's vertical distance and left out below 1e-9 of a unit
  point term. A straight strip in two-layer soil is within 3.6e-6 of its
  uniform-current resistance for segment lengths of 0.125 … 2 m, not
  monotonically (+5.1e-4 before; the 7.7e-7 of the stream held at its five
  sampled lengths only, audit W3A-03), and the surface potential above it
  within 8e-6 (0.94 % before).
- **Cross-layer path (items 2–4):** the layered correction in line form (its
  exact image expansion, ADR-0007 amendment): a rod through the interface
  changes by 9e-7 between segment lengths of 2 and 0.125 m (1.3e-4 …
  2.4e-4 before), and the two paths of `image_2layer` agree to 1.7e-7 where
  both apply (6.2e-4). Memory as on the series path, in row chunks and in
  row blocks beyond the budget: 278 → 137 MiB at the AP1 station with 40 %
  penetration with numba (without numba 224 → 140 MiB since audit W3A-05).
  The spectral amplitudes in closed form: the path with the point kernel
  1.6 … 2.0 times faster at the AP1 station, round-off.
- **Continuous switch (item 5, D2):** pairs between 6 and 7.5 segment lengths
  blend the exact and the far value with a C¹ weight (direct pairs, surface
  and Tagg/Sunde images, the images of the cross-layer correction, field
  points): 1e-13 over local shifts and ≤ 5e-11 at UTM coordinates (4.7e-11
  measured by audit W3A-04), where the position had moved results by up to
  5.9e-7 (direct pairs) and 8e-6 … 2.6e-5 (two-layer images).
- **Numbers:** homogeneous soil ≤ 7.8e-7; two layers: house grids 3e-6 …
  1.8e-4, wires near the interface up to 6.2e-4 (audit W3A-04), surface
  potentials near the interface up to 0.9 %; cross-layer rods ≤ 4e-4, the
  AP1 station −4e-5. The cost was larger than the stream stated (audit
  W3A-01, see `fw3xl`).
- **Golden:** 60 quantities in 9 line-kernel cases declared
  (`w3xl.toml`), up to 6.0e-4 (`line_img2_ring_rods_within_layer`);
  `line_img2_rod_crosslayer` moves 4.0e-4, its former point-form error.
  Point-kernel and `mom`-family cases are bit-identical apart from the
  round-off of the closed-form amplitudes (≤ 2.5e-16).

### Stream `w3perf`: penetration depth, `sommerfeld` segment rules, the BusType preset, overlaps (Q34, Q39)

Roadmap items of the earth return and the soil, the user decisions Q34 and
Q39, and two audit leftovers (W2B-05, W1R3-07). In short:

- **Penetration depth:** `penetration_depth(soil, frequencies)` and
  `PenetrationDepth` give the complex penetration depth and the equivalent
  resistivity of a soil with any number of layers; exact for homogeneous
  soil, over the 39 AP1 soils (50 Hz … 1 kHz, 0.1 … 1000 m) within a median
  of 2.2 % of the exact layered earth return (at most 28 / 33 % overhead /
  buried); three and four layers up to 46 / 77 % at 1 km and 1 kHz (audit
  W3B-08). `per_metre_mutual_impedance` gives the exact layered earth-return
  impedance per metre (within 5e-8 of Carson and of the two-layer Pollaczek
  integral; against an independent integral within 6e-8 of the earth-return
  correction up to 300 m and 7e-7 at 1 km for buried pairs, audit W3B-08).
- **Q34:** the self term of every `sommerfeld` segment is the exact
  one-dimensional reduction of its double integral, and close pairs get a
  graded rule: no step at a quarter skin depth (4.8e-4 off before, with a
  step of 3.5e-4); with the λ grid of audit W3B-01 the self correction is
  within 1.1e-9 for the soils tested (3.4e-9 in high-contrast soils, audit
  W3R2B-08) and close pairs within about 4e-8 (the stream had stated 2.7e-9;
  up to 1.5e-3 before).
- **Q39:** BusType documents record the effective-resistivity preset
  (`rho_preset=` of the exporters, `metadata["rho_preset"]`,
  `BusTypeSpec.rho_preset`); `evaluate_spec` and `bustype_from_spec` warn
  with `RhoPresetMismatchWarning` when the bus uses another preset.
- **Overlaps:** `diagnose()` reports inductive conductors that lie beside
  each other closer than touching over more than ten touching distances and
  5 % of the shorter one (category `overlap`); the generated networks had
  such pairs (the 5-house radial network 10, 200 houses 247: drops of one
  tap on one axis), which led to decision D9.
- **Cost:** a long lumped conductor under `sommerfeld` costs at most linear
  time with bounded memory (W2B-05: 2 km 28 s and 1.6 GB → 1.1 s and
  0.21 GB; 100 km 10.6 s within 0.24 GB, but beyond 33 km for that conductor,
  38 km for one at 0.7 m, the result warns with `SommerfeldResolutionWarning`
  and is not converged, audit W3B-10). Compiled Carson integrals were
  measured and not built (5 … 13 % of an inductive sweep after memoisation,
  below the 20 % threshold; 10.63).
- **`carson_series`:** a tilted near-parallel pair that comes close is tiled
  per stretch (W1R3-07: a lumped pair and its pieces within 8e-6 of the loop
  reactance); generated networks move by up to 3e-7 (audit W3B-07). On the
  generated networks of 5 … 100 houses the series reads X 4 … 21 % and R up
  to 11 % high at 1 kHz and X 21 … 50 % and R up to 0.9 % high at 50 Hz
  against `sommerfeld` (audits W3B-02, W3R2B-03).
- **Golden:** five cases move by 8e-10 … 4.9e-7
  (`cond_lumped_long_pen_sommerfeld` 4.9e-7,
  `cond_isolated_pen_sommerfeld_two_layer`,
  `cond_overhead_lead_over_pen_carson_series`,
  `cond_galvanic_neumann_sommerfeld`, `defaults_mixed_cluster_inductive`).

### Audits W3A and W3B of the wave-3 streams

Two auditors attacked wave 3 (snapshot `0b942c9`), W3A stream `w3xl`, W3B
stream `w3perf` together with the fix streams `fw1r3` and `fw2r2`. Findings:

- W3A-01 (medium): the image work grew with the world's longest segment and
  with the image density: eight 9.4 m pieces among 1920 segments made the
  solve 2.7 … 6.5 times slower, thin top layers 7 … 8.5 times, and the TN
  40-house network in 1000/30 Ω·m with h₁ = 1.5 m took 4.4 times (11 times
  without numba) the solve before wave 3, where the texts said +59 … +70 %
  and 3.3 times.
- W3A-02 (medium, pre-existing): image pairs used the reduced kernel
  √(r² + a²): a ring 1 cm above a 1000/30 Ω·m interface read +1.60 % against
  the exact thin ring (+1.74 % before wave 3); the cause of what stream
  `w4axi` had found.
- W3A-03 … W3A-06 (low): the strip bound "within 7.7e-7, monotonically" held
  at the five sampled lengths only; stale texts on the one-step switch and
  on far images as point sources, 6.2e-4 instead of "up to 4e-4" and 4.7e-11
  instead of 3e-11 at UTM coordinates; the memory claims held with numba only
  (161 instead of 79.6 MiB); the near-interface "finer segment_length"
  warning fired for the line kernel.
- W3A-07 (info): switching the quadrupole term off did not give the code
  before wave 3.
- W3B-01 (low): the Q34 bound failed for short segments (1e-5 … 7.4e-5 at
  0.01 skin depths, 7.5e-5 … 3e-4 at 1 m), because the logarithmic region of
  the λ grid had 64 nodes; pre-existing, at most 8e-9 in a solved reactance;
  the close rule 3.5e-8 off instead of the stated 2.7e-9.
- W3B-02 (medium, text): `carson_series` on generated networks of 5 … 30
  houses reads X +3.6 … +20.7 % and R +1.9 … +7.8 % at 1 kHz and X +21 …
  +44 % at 50 Hz, which no text stated.
- W3B-03 (low): the overlap check had no tolerance: cables 1e-14 m inside
  the touching distance were flagged over their whole length.
- W3B-04 (info, decision): drops of one tap in a row at the touching
  distance move the 5-house radial network's X at 1 kHz by −2.0 / −2.7 %
  (decision D9).
- W3B-05, W3B-07 … W3B-09 (low, texts): star networks warn too, and the
  stream's limitation and roadmap texts were not taken over; the stretch
  tiling moves generated networks (the 10-house star with sites 2.8e-7, the
  80-house radial network 3.4e-8); the errors of `penetration_depth` in
  three and four layers (44 … 77 % at 1 km, 1 kHz) and
  `per_metre_mutual_impedance` 6.8e-7 off at 1 km for buried pairs; rows 34
  and 39 of this page.
- W3B-06 (low): `to_bustype` and `save_bustype_to_db` validated the preset
  and dropped it.
- W3B-10 … W3B-12 (info): the 100 km conductor warns as unconverged, and so
  does one of 37 km; the mutual of two long close conductors moved by
  2.6e-5 … 4e-5, not 8e-6; `penetration_depth` read `"50"` and `True` as
  frequencies and raised `ZeroDivisionError` at 1e-320 Hz, and
  `PenetrationDepth` could not be compared or hashed; a proposed test of
  audit W2R2 reads a phrase that D2 removed.

Confirmed: the blend is continuous; translation invariance holds (7.7e-14
over local shifts, 4.7e-11 at UTM coordinates); the cross-layer rod is within
1.6e-6 of an exact closed form; chunking is bit-identical; golden 497
quantities identical, 6 within round-off, 60 moved within their bounds; the
jump of a ring on an interface that `w4axi` found is resolved (≤ 0.0005 %).
`penetration_depth` agrees with the auditor's own reflection coefficient to
1.8e-9 in one to four layers; Q34 has no step (≤ 6.5e-12 over 6700 lengths);
W2B-05 is bounded (2 km: 1.22 s, 0.21 GB); `fw1r3` and `fw2r2` are
bit-identical in the golden cases and right in the auditor's own attacks
(pieces against the lumped pair ≤ 4.6e-15; W2R2-02 within 1.4e-14 of the
former path). Taken up by `fw3xl` (W3A, with the far-image option),
`fw3perf` (W3B-01 … -08, -10, -11; -04 with D9) and the coordinator
(W3B-09, -12).

### Streams `fw3xl` and `fw3perf`: fixes of audits W3A and W3B, the far-image option and D9

- **`fw3xl` (W3A-01, -05):** the far-image work follows the pair class and
  the upper half of the symmetric matrix (bit-identical, round-off in the
  symmetric half): 16 house grids take 3.8 times the point kernel in
  homogeneous soil and 1.6 times in two-layer soil (h₁ = 5 m) with numba (5.8
  and 1.4 times without); against the solve before wave 3 the TN 40-house
  network takes 1.84 times with numba and 3.63 times without (4.2 and 9.7
  times before the fix), eight 9.4 m pieces among the 16 house grids 1.5
  times (2.8), thin top layers 3.6 … 4.5 times (7 … 8.7). Without numba the
  far-image term holds 80 instead of 161 MiB.
- **`fw3xl` (W3A-02, -03, -04, -06):** image pairs act at the transverse
  distance max(ρ⊥, a), collinear images and images within one wire radius
  of their segment keep the reduced kernel, far images have no radius term,
  and the image corrections are symmetric: the ring 1 cm above 1000/30 Ω·m
  −0.004 % instead of +1.60 %, a strip 0.1 m above the interface −3.8e-6
  instead of +6.8e-5, the rod through the interface within 1.4e-6 of its
  exact value. The near-interface warning fires for the point kernel only;
  the texts are corrected.
- **The far-image option (the author's decision of 2026-11-15):**
  `Engine.image_far_field = "point"` keeps the far Tagg/Sunde images and the
  images of the cross-layer correction beyond the near-pair reach as point
  sources, the continuous switch and the transverse distance stay; networks
  move by 2e-5 … 1.8e-4 against the default (16 house grids 0.4 m above a
  1000/10 Ω·m interface 4.25e-4, audit W3R2A-04), surface potentials by up
  to 0.8 %, and the TN 40-house network solves in 0.46 of the default's time
  (0.84 of the solve before wave 3); recorded in
  `metadata["series"]["far_field"]`. Since stream `fw3r2a` the point form
  keeps the near image pairs on the cross-layer path: networks 3e-6 …
  4.3e-4, the TN 40-house network 0.49 of the default's time and 0.91 of the
  solve before wave 3.
- **Golden (`fw3xl`):** W3A-02 moves four cases by at most 3.2e-7
  (`line_img2_ring_rods_within_layer`; 3.19e-7, audit W3R2A-07); the other
  fixes and the option are bit-identical in every stored case.
- **`fw3perf` (W3B):** the logarithmic region of the λ grid has 256 nodes:
  the self term is within 1.1e-9 of a refined rule and an independent
  reference from 1 m and 0.01 skin depths to 3 km, close pairs within about
  4e-8, at no measurable cost. The overlap check allows round-off at the
  touching distance, and `find_close_wires` calls such wires close; the live
  BusType keeps the preset as the last line of its description
  (`groundfield rho_preset: <name>`, read back by `bustype_rho_preset`);
  `penetration_depth` refuses bool, str, complex and subnormal frequencies,
  and `PenetrationDepth` compares by value and is hashable; texts of W3B-02,
  -05, -07, -08, -10. Golden: the five `sommerfeld` cases move by 4.8e-18 …
  2.2e-11.
- **`fw3perf` (D9):** service drops that leave one tap in one direction lie
  side by side at the touching distance, sorted by length and centred on
  their former axis, so that only neighbours touch
  (`PlannedBranch.axis_offset_m`): X at 1 kHz of the radial 5-house network
  −2.33 / −2.70 % (`carson_series` / `sommerfeld`), radial 10 −2.31 /
  −2.82 %, star 10 −1.06 / −1.27 %, radial 80 / 200 ≤ 5e-4, radial 30
  ≤ 1.1e-6; no golden case moves.

### Audits W3R2A and W3R2B: round 2 on the wave-3 fixes

W3R2A attacked `fw3xl` and the far-image option (snapshot `ac556ec`), W3R2B
`fw3perf` and D9 (snapshot `df16521`). Findings:

- W3R2A-01 (medium): with `image_far_field="point"` wires wholly below the
  layer interface lose accuracy as their segments get longer: a 75 m strip
  below a 0.5 m top layer reads −1.0e-3 … −1.8e-3 at 2.5 m segments,
  −1.7 … −4.6 % at 7.5 m and up to −58 % at 37.5 m against an exact
  reference (the default within 1.3e-6); 16 house grids with a 75 m bare
  conductor in 40 m pieces +1.13 % against the default.
- W3R2A-02 (low): `far_field` of the public `solve_image_2layer` is not
  validated; any other value runs the point form, and the field then
  disagrees with the solve.
- W3R2A-03 (low): within one wire radius of the interface Z is not monotone
  in the gap (a ring in 1000/30 Ω·m: 5.993 → 7.319 → 5.992 Ω).
- W3R2A-04 (low, texts): "the point form moves networks by 2e-5 … 1.8e-4"
  is too narrow (4.25e-4), and "as before wave 3" is misleading (up to
  2.3e-4 from the D2 blend, which the option keeps).
- W3R2A-05 … W3R2A-07 (info): "integrated exactly" overstates the axis value
  for short strips near the interface (a 2 m strip 5 mm above 1000/30 Ω·m
  +3.0e-4); direct pairs of different wires keep the reduced kernel (two
  strips 1 cm above the interface, 5 cm apart, −3.5e-4); a bound comment in
  `w3xl.toml` (3.4e-7 against the measured 3.19e-7).
- W3R2B-01 (medium): the D9 rows held only to 1e-11 m: any placement jitter
  from 1 nm to 1 mm put the drops back on one axis (the 10-house radial
  network: 19 / 20 / 14 overlaps), and X at 1 kHz jumped by +2.37 % /
  +2.90 % (`carson_series` / `sommerfeld`) at 1 nm.
- W3R2B-02 (low): the overlap tolerance of 1e-11 m lies below the spacing of
  floating-point numbers at projected coordinates: rows translated to
  (5e5, 5.8e6) give 4 / 10 / 98 false overlaps for the radial networks of
  5 / 80 / 200 houses.
- W3R2B-03 (medium, text): the stated `carson_series` error is exceeded (the
  80-house radial network +49.7 % at 50 Hz; R up to +11.1 % at 1 kHz and
  0.9 % at 50 Hz; the default `sommerfeld` 1.0e-3 from a refined one, stated
  4e-4).
- W3R2B-04 (low, text): the base error of a 1 m self term is 3.0e-4 /
  6.4e-4 in 30/1000 and 10/10 000 Ω·m, not 7.5e-5.
- W3R2B-05 (info): a `PenetrationDepth` built directly is not validated, and
  f = +0.0 and −0.0 compare equal but hash differently.
- W3R2B-06 (low): loading refused only blank presets ("station\nfoo" lost
  the preset, " station " read back as "station", unknown names passed), and
  every user line that merely starts with the marker was deleted.
- W3R2B-07, W3R2B-08 (info): the start points of a row leave their tap by up
  to (n − 1)/2 touching distances, and stars of 40 … 180 houses have rows
  too (the 60-house star X −1.0e-3); the self-term bound of 1.1e-9 reaches
  3.4e-9 in 10/10 000 Ω·m and 1.8e-9 under a 0.3 m top layer.

Confirmed: the paths of the option (round trips, several frequencies,
`solve_mutual_field`, no effect with the point kernel, the `mom` family or
homogeneous soil); with D2 and W3A-02 switched off at run time the point
form reproduces the code before wave 3 to 6.6e-7 in the upper layer and
5.7e-6 on the cross-layer path; the W3A-01 matrices are bit-identical but
for the diagonal rule (Z −5.4e-14); W3A-02 against the auditor's exact
references (strips with a gap of at least one radius ≤ 5.5e-5, the ring
−0.0039 %, reciprocity ≤ 4e-11). The D9 geometry of 80 AP1 networks and 58
variants: every conductor that is not a row drop, every electrode and every
junction bit-identical, the rows equal to the auditor's own construction,
up to 277 overlaps → 0 without jitter, old plans rebuilt bit-identically;
the D9 moves agree between three constructions; W3B-01 holds (self terms
within 1.1e-9 for the stream's soils, close pairs 3.5e-8), golden 43 cases
bit-identical and 5 within 4.8e-18 … 2.2e-11. Outside the round, W3R2B
found that `plan()` followed by `build()` drew the placement jitter again
and that a tie of the star topology flips under jitter (the 10-house star:
X −11.8 %, now on the roadmap). Taken up by `fw3r2a` (W3R2A), `fw3r2b`
(W3R2B-01, -02, -05, -06, the texts of -03 and -07, `plan()` / `build()`)
and the coordinator (the texts of W3R2B-04, -08).

### Stream `fw3r2a`: fixes of audit W3R2A

- **The point form below the interface (W3R2A-01):** on the cross-layer path
  `image_far_field="point"` keeps every image family with all its images;
  the near pairs within 7.5 segment lengths are Galerkin integrals blended
  into the point form, a segment's own image follows the switch of every
  pair, and only the dense far term stays out. The 75 m strip below a 0.5 m
  top layer is within 1.1e-5 of its exact value at every segment length
  (−1.06e-5 at 0.5 m in 1000/30 Ω·m, the option's own far-pair error;
  −3.1e-7 at 37.5 m, where it read −58 %), and 16 house grids with the 75 m
  conductor in 37.5 m pieces read −3.65e-6 against the default instead of
  +1.13 %. The default and the series path in point form are bit-identical.
- **Its cost:** on cross-layer worlds the point form takes 0.49 … 0.94 of
  the default's time; below thin top layers it keeps the Galerkin near image
  pairs it used to save (1.5 … 2.4 times the solve before wave 3 instead of
  1.2). The TN 40-house network in 1000/30 Ω·m with h₁ = 1.5 m is a
  cross-layer world (its transformer rods reach below h₁): its point form
  moves by 2.3e-4 and takes 0.91 of the solve before wave 3 (0.83 before the
  fix).
- **`far_field` (W3R2A-02):** `solve_image_2layer`, the series kernel, the
  image families and the field correction raise `ValueError` for any value
  but `"quadrupole"` or `"point"`.
- **Within one wire radius (W3R2A-03):** the image blend runs over a tenth
  of the wire radius; the bump of 22 % of the ring in 1000/30 Ω·m is gone,
  and a step of at most 0.17 % onto the interface remains (wires 0 < g < a/2
  from the interface move by up to −18 %: the ring 1.75 mm above, 7.319 →
  5.985 Ω; the ring 1 cm above stays at −0.004 %).
- **Texts (W3R2A-04 … -07):** the point form moves networks by 3e-6 …
  4.3e-4 against the default (16 house grids 0.4 m above a 1000/10 Ω·m
  interface) and is not the result before wave 3 (16 grids in 1000/30 Ω·m,
  h₁ = 1.5 m: +2.32e-4); the line kernel is exact for long wires with an end
  effect O(a²/(d·ℓ)) (a 2 m strip at one wire radius +3.03e-4); direct pairs
  of different wires keep their convention (−3.50e-4 against a
  circumference-averaged direct term); the golden comment gives 3.19e-7.
- **Left open (before the stream as well):** in point form a ring at the
  interface differs by up to 2.96e-4 between the series and the cross-layer
  path (the default by 1.3e-6), and `solve_mutual_matrix(symmetrize=False)`
  of a cross-layer world is asymmetric by 2.3e-8 in both forms.
- **Golden:** 563 of 563 quantities identical; 29 new tests.

### Stream `fw3r2b`: fixes of audit W3R2B

- **Rows under jitter (W3R2B-01):** two drops of one node belong to one row
  when their start points lie within the touching distance d_t of each other
  and the shorter drop's end lies within d_t of the longer drop's line;
  drops that are not exactly parallel lie along one common normal
  (`PlannedBranch.axis_offset_normal_xy`), each at least d_t from the
  others, and a drop that would overlap a drop of a row joins it, also a
  drop of a tap 1 … 4 cm away on the same trunk (from 2 cm of jitter). The
  10-house radial network at 1 nm of jitter: −8.5e-10 / −1.1e-9 instead of
  +2.4 % / +2.9 %; no generated network reports an overlap with `jitter_m`
  0 … 0.1 m (240 networks, star and radial, 5 … 200 houses); every network
  without jitter is bit-identical.
- **Projected coordinates (W3R2B-02):** the tolerance at the touching
  distance is the larger of 1e-9 of it and 16 units in the last place of the
  pair's largest coordinate; local worlds keep their findings bit for bit.
- **Values and documents (W3R2B-05, -06):** a `PenetrationDepth` built
  directly is validated and its hash agrees with `==`; `metadata["rho_preset"]`
  must be a known preset name or null, and only exact preset lines of a
  description are removed or replaced.
- **`plan()` and `build()`:** `build()` right after `plan()` on the same
  generator gives the world of that plan; successive calls still draw
  successive realisations.
- **Golden:** 563 of 563 quantities identical.

## Wave 4: a conforming `fem` (Q37, Q40, D6)

Wave 4 replaced the equivalent-hemisphere reduction of `fem` by conforming
finite-element models (user decision Q37): an axisymmetric model (stream
`w4axi`, part A) and a three-dimensional model for every other world (stream
`w4fem3d`, part B), which also quantified the overlap of concrete shells at
junctions (Q40). Its dependencies follow decision D6.

### Stream `w4axi`: the conforming axisymmetric model (part A)

Rods on one vertical axis and horizontal rings centred on it are solved as
the solid bodies they are (a cylinder with flat end faces, a torus), in any
number of horizontal layers, bare or with a concrete shell, each electrode
one port: linear triangles on a graded tensor mesh conforming to every
electrode surface and layer interface, an O-grid around the ring wire, the
monopole Dirichlet-to-Neumann condition on a box 1000 lengths away, and the
port conductance matrix extrapolated from two nested levels; no new
dependency. The network of the shared pipeline runs on top. In short:

- **Accuracy:** within about 1e-4 of independent references (thin-wire
  limits, elliptic and Hankel rings, finite volumes of the solid rod) at
  0.2 … 2 s per solve; the limit cases of the reduction — a ring 1 cm above an
  interface (+191 … +308 % without a message), a rod reaching 1 µm … 8 cm
  into a more conductive top layer (−44 … −78 %, warning only), a ring
  exactly on an interface, rods through one, clusters of several electrodes
  and several clusters — are solved within 1e-4 (within 2e-3 by the 3-D
  model of part B where they are not axisymmetric; audit W4A-01 found the
  exception of a ring wire at an interface, see below).
- **Numbers:** every `fem` result moves: the stored 3 m rods −4.65 % /
  −4.57 % (Dwight's rod ignored the head depth), the B2 rod / B3 ring
  −0.70 % / −0.35 %, a ring with a bonded central rod +20 %, a ring 1 cm
  above a conductive layer −67 %, a rod 2 cm into a conductive top layer
  +164 %. Every other world was refused until part B.
- **Options:** `FemOptions.n_radial`, `n_axial` and `z_far_factor` are gone
  (2026.11 names); the point sources of `fem` count against
  `Engine.max_segments` (found in the stream's own attack); benchmark B1
  checks `fem` against Weber's disc.
- **For `image_2layer`:** the stream's references found a ring 1 cm above an
  interface 1.7 % above the exact thin ring, a jump of 0.03 … 0.38 % between
  the two sides for a ring on an interface, and a truncated series in
  30/10 000 Ω·m (−1.0 %); wave 3 resolved the jump, and audit W3A-02 found
  the cause of the first.
- **Golden:** `fem_rod_hom` −4.65 % and `fem_rod_two_layer_deep` −4.57 %
  (`w4fem.toml`); the other 47 cases are bit-identical.

### Stream `w4fem3d`: the three-dimensional conforming model (part B; Q40, D6)

Every world that is not axisymmetric — strips, polylines, stars, meshes,
grids, rods and rings side by side, several clusters with their transfer
resistances, galvanic distributed conductors with every leaking piece a port
— is solved with smoothed line sources and an analytic radius correction in
an isotropic tetrahedral mesh refined towards them, partially penalised
immersed layer interfaces, conjugate gradients with an algebraic multigrid,
and two nested levels extrapolated. It needs the optional extra `fem3d`
(scikit-fem, BSD-3-Clause; pyamg, MIT; decision D6) and raises `ImportError`
naming it otherwise. In short:

- **Against part A:** within 2.0e-3 for rods and rings in one to three
  layers, inside its own error estimate (away from interfaces and other
  wires since audits W4B-01, W4B-02, see below).
- **The AP1 station** in the corner soils of the AP1 grid lies between the
  thin-wire engines: `image_2layer` +0.7 … +3.0 % above it, `mom` −0.4 …
  −0.9 % below; 1.5 … 3 minutes and 3.6 … 4.4 GB per solve.
- **Brace effect** (the open point of `w2mom`): +49.09 / +317.99 % at
  ρ_c = 1000 / 5000 Ω·m, which confirms the per-segment value of the
  concrete-shell page (+49.2 / +318.5 %) to 0.1 / 0.5 points.
- **Q40:** against the concrete resolved as a volume, each junction of
  encased wires acts like at most about one shell radius of wire (a corner
  −1.5 cm, a crossing −3 … −6 cm, a T-joint within ±5 cm; 5 mm bar, 5 cm
  cover, ρ_c 1 000 / 5 000 Ω·m); the generators' house mesh changes by less
  than 0.6 %, so no correction rule is applied (audit W4B-05 found these
  numbers not converged, see below).
- **Golden:** nothing moves (every golden `fem` case is axisymmetric and
  stays on part A bit for bit); the 3-D model has no golden case.

### Audit W4A: the axisymmetric model and the shared network

The first wave-4 audit checked part A and the network on top of it
(snapshot `4e37760`) against references without groundfield code:
axisymmetric boundary elements for the solid cylinder and torus with the
Tagg/Sunde images as rings, and cell-centred finite volumes of a rod with a
shell. Findings:

- W4A-01 (medium): a ring wire cut by an interface to a more conductive
  layer, or within 0.05 wire radii of it, is off by up to 6.0e-3 at the
  default; the error estimate lies below the error in 7 of 14 cases, and the
  levels are not monotone.
- W4A-02, W4A-03 (low): `extrapolate=False` reports an error estimate of 0
  for a level-0 ring 6.2e-3 low; `model="auto"` ends in a bare `ValueError`
  for a ring diagonal to a rod's end edge, without falling back to the 3-D
  model or naming a remedy.
- W4A-04 (low): tutorial T09 still says that `fem` refuses every other
  world.
- W4A-05 (medium): the node limit of 4 M admits about 16 GB (3.1 / 3.8 /
  12.6 GB at 0.80 / 0.91 / 3.21 M nodes), and no memory figure is
  documented.
- W4A-06 (low): the provenance line of the validation page names Linux and
  2026-11-14 for `fem` rows measured on macOS on 2026-11-15.
- W4A-07 (medium): the known limitation on rings near an interface quotes
  +3.6 % at 1 cm only; closer in, `image_2layer` reads +8.6 … +93 % (at
  1.5 … 0.95 wire radii) without a warning, and `diagnose()` is silent.
- W4A-08, W4A-09 (low): the dispatch threshold of 3 nm sends a rod 1 µm off
  the ring axis to the 3-D model (40.7 s / 2.35 GB instead of 1.8 s /
  0.61 GB, −1.43e-3, and it needs the extra); stale or overstated texts
  ("every other world is refused", "lower bound / monotone", "within 5 %").
- W4A-10, W4A-11 (info): `predicted_nodes` is 1.5 … 2.3 % low for rings; the
  second methods of the validation table are thin-wire models or none (the
  solid-body boundary elements: −1.07e-5 … −3.0e-5).

Confirmed: the default `fem` agrees with the boundary elements to 1.0e-5 …
1.5e-4 for rods, rings, rings near and on interfaces, rods through and
near interfaces and two-port worlds, with an error estimate of 10 … 130 times
the error; three layers and concrete shells against further references; the
network (U = R·I to 8e-17, one branch matrix for `fem` and `image_2layer`,
round trips, a UTM world and the D5 warning); the refusals, the light import
and the extra; golden 559 of 563 quantities identical, the 4 that move in the
two declared `fem` cases. Taken up by the wave-4 fixes below.

### Audit W4B: the three-dimensional model

The second wave-4 audit checked part B, decision Q40 and the extra `fem3d`
(snapshot `4e37760`) against the axisymmetric model, the exact thin rings
and closed forms of the smoothed line source. Findings:

- W4B-01 (medium): the radius correction added terms for the surface image
  and for every wire within 20 R, where the mesh already gives the exact
  field: shallow wires up to +1.9e-3 high, two wires 10 cm apart +4.6e-3,
  above the error estimate.
- W4B-02 (medium): neither "within 2e-3 of the axisymmetric model" nor "the
  estimate bounds the error" held for every world the guards accept: a ring
  1.5 cm above a 1000 / 30 Ω·m interface read +9.1e-3 against an estimate of
  5.4e-3, its two levels on either side of the reference.
- W4B-03 (low–medium): at a layer contrast of 1000 the conjugate gradients
  did not converge in 3000 iterations and ended after about a minute in a
  bare `RuntimeError`; the pages said "any contrast".
- W4B-04, W4B-05 (low–medium): the square concrete prism keeps the shell's
  outer conformal radius, but its inner one is 0.914 r_b, so its concrete
  leaks 3.9 % per metre more than the shell, which the texts blamed on the
  discretisation; the Q40 junction numbers were not converged (one face
  refinement moved the corner by −4.9 / −7.1 cm), although "no correction
  rule" stays justified.
- W4B-06, W4B-07 (low–medium): the face key of the immersed elements
  aliased above 2²¹ nodes (the default limit of 2 million lies 4.6 % below)
  and coupled unrelated elements; results depended on numpy's global random
  state (1e-12 … 1.9e-6 apart), and every solve advanced the user's stream.
- W4B-08, W4B-09, W4B-10 (low): `Engine.max_segments` was checked only after
  both levels were solved, and the number of ports had no limit;
  `fem_peak_memory_gb` reported the peak of the whole process; about 30
  stale texts.

Confirmed: no stored golden number changes (563 of 563 quantities against
the merge parent); the AP1 station, the brace effect (+49.09 / +317.99 %)
and the Q40 numbers reproduce to the last digit; the light import, the
message without the extra, the lock file, the licences and the `ci:` commit
are correct; keeping `fem` experimental is justified. Taken up by the
wave-4 fixes below.

### Streams `fw4axi` and `fw4fem3d`: fixes of audits W4A and W4B

- **`fw4axi`, a ring next to a more conductive layer (W4A-01):** a ring
  wire within 1.05 radii of an interface gets 64 instead of 8 tensor
  intervals per side of its mesh block. The auditor's 14 cases are within
  3.3e-4 of the boundary elements (were 6.0e-3), and the error estimate
  (7.5e-4 … 1.9e-3) is at least the error in each of them (was below it in
  7); such a ring takes about four times the nodes (0.37 → 2.3 … 2.7 s).
- **`fw4axi` (W4A-02, -03, -05, -08, -10):** a single level reports an error
  estimate of NaN; `auto` solves a ring without room for its mesh block with
  the 3-D model, and `model="axisymmetric"` refuses it with a remedy; the
  common axis has a tolerance of 1e-3 of the smallest wire radius or gap
  (the axisymmetric reading then errs by less than 1e-6), so a rod 1 µm off
  a ring's axis stays axisymmetric; `FemOptions.max_nodes` defaults to 2
  million (about 4 GB per million nodes, re-measured 3.92 … 4.21), and the
  node prediction errs high (+0.67 … +10.8 %).
- **`fw4axi`, the thin wire next to a contrasting interface (W4A-07,
  W4B-02):** `InterfaceContactWarning` and the `diagnose()` category
  `interface_contact` for horizontal wires within the reach where the thin
  wire deviates from the solid wire by more than about 1 %, measured against
  the axisymmetric torus and the Hankel thin ring with an exact 2-D
  proximity correction at contrasts 1.5 … 1000 in both directions: 1.0 /
  1.5 / 2.9 / 3.5 / 3.8 / 3.9 wire radii over a layer 2 / 3 / 10 / 33 / 100
  / 1000 times more conductive, 1.0 … 1.25 radii over a more resistive one,
  nothing below a contrast of 2. Emitted by the image family, `mom`,
  `mom_sommerfeld`, `solve_mutual_*` and the 3-D `fem`; no generated
  network, no world of the AP1 soils and no golden case is reported.
- **`fw4axi`, texts (W4A-04, -09, W4B-10):** tutorial T09 describes both
  models; the stale texts of the auditors' lists in the reference pages,
  the docstrings and ADR-0002.
- **`fw4fem3d`, the radius correction (W4B-01):** it counts the observer's
  own wire whole and, of every other wire and surface image, only the part
  within 1.5 R (a clip at 2 R would still keep 2 % of the self term at
  1.5 R). A ring r = 1 m at 0.06 … 0.8 m depth lies −1.62e-3 … −1.83e-3
  from the axisymmetric model (spread 2.1e-4, was 2.1e-3); two coaxial
  rings 10 cm apart −1.19e-3 / −1.48e-3, inside the estimate of 3.87e-3
  (were +4.60e-3 / +3.67e-3); the AP1 station moves by about −1e-4, the
  brace effect to +49.14 / +318.23 %.
- **`fw4fem3d`, the accuracy texts (W4B-02):** "within 2e-3 of the
  axisymmetric model" holds away from interfaces and other wires; the error
  estimate bounds the error while the ratio of the level errors is at least
  2.5 (2.1 … 2.7 times the error for rings), and not for resolved concrete.
  The ring 1.5 cm above a 1000 / 30 Ω·m interface still reads +9.2e-3
  against 5.4e-3: the thin-wire error, which the new warning reports.
- **`fw4fem3d` (W4B-03 … -09):** a conjugate-gradient solve that does not
  converge (contrast above the tested 300) raises `BackendCapabilityError`
  naming the contrast, the axisymmetric model and the integral engines (the
  auditor's preconditioner gave no gain); the texts say that the square
  prism's concrete is 3.8 % less resistive (inner conformal radius
  0.914 r_b, two methods) and mark the junction numbers not converged — a
  few cm of wire per junction, at most about 2 r_b, sign not settled, the
  house mesh 0.35 … 0.85 % re-measured; faces are paired with `np.lexsort`;
  numpy's global random state is saved, seeded and restored around the
  multigrid set-up (results bit-identical whatever the user's seed, the
  user's stream unchanged); `Engine.max_segments` and the number of ports
  (100; about 0.95 s per port and level) are checked before meshing;
  `fem_peak_memory_gb` is what the solve added (about 2.9 GB per million
  nodes of the finer level), `fem_process_peak_memory_gb` the process peak.
- **Golden:** 563 of 563 quantities identical in both streams.

### Stream `fw4int`: the loose ends of the wave-4 fixes

- The texts of `fw4axi` on the `fem` page and in ADR-0016, fitted to those
  of `fw4fem3d`; the validation rows next to and on an interface
  re-measured: `image_2layer` reads a ring 1 cm above a 1000 / 30 Ω·m
  interface +3.46 … +3.57 % above `fem` (the page said +5.2 … +5.4 %, from
  before wave 3), the ring on an interface within 1.2e-5 of the solid-torus
  boundary elements.
- `FemOptions.max_ports_3d` (default 100): the port limit of the 3-D model
  is an option, and its refusal names it.
- Benchmarks B8 and B10 exclude `fem` for its cost; measured once with the
  3-D model, both are reciprocal to round-off, B10 +0.6 / +0.8 % above
  `mom`. The transfer impedance of B8 lies 0.40 … 0.96 % below `mom`, more
  than the error estimate; handed to the second audit round.
- A test that the 3-D model warns once for a wire next to a contrasting
  interface; the call was already in place.
- **Golden:** 563 of 563 quantities identical.

### Audits W4R2A and W4R2B: round 2 on the wave-4 fixes

The second round tried to break the wave-4 fixes (snapshot `4eb1c2c`)
with the boundary elements of round 1, the axisymmetric model, `mom`
refined and closed forms of the smoothed line source. Findings:

- W4R2A-01, W4R2B-03 (medium): the 3-D model next to another wire. The
  fallback of W4A-03 sent a ring without room for its mesh block to the
  3-D model, which read two diagonal rings 1.3 … 3.8 mm apart +11.4 …
  +14.3 % high against an estimate of 0.42 %, without a message; coaxial
  rings 7.5 cm apart, where the texts promised 2e-3, read +4.35e-3, 6 cm
  apart +1.0e-2, and below one source radius the port entries were off by up
  to 4.3 %; the radius correction counted other ports' wires with the
  observer's current.
- W4R2A-02 (medium): the refined ring block held up to a contrast of 33; a
  ring touching a layer 1000 times more conductive read −1.61e-2 against an
  estimate of 1.07e-2, and at `refinement=1` the estimate fell below the
  error at contrasts 100 and 1000.
- W4R2A-03 (medium): the 3-D model neither warned nor refused a rod end at
  an interface (+43 % for a tip on a 1000 / 30 Ω·m interface).
- W4R2B-01, W4R2B-02 (low–medium): on straight pieces longer than about 7 m
  off the coordinate axes the observer's own piece was clipped by round-off
  (a 20 m oblique strip −3.8e-4); an arm of the same polyline folded back at
  any angle but exactly parallel counted whole (a hairpin tilted by 1e-4 rad
  +1.41e-3, a 10° V +8.5e-4 against the same V as two strips).
- W4R2A-04, W4R2A-05 (low): the reach of the near-interface warning lay
  0.2 … 0.4 wire radii short for a thick wire on a small ring, and cut wires
  below a contrast of 2 read up to ±1.5 % without a warning; `diagnose()`
  with a `fem` engine reported a ring that the axisymmetric model solves.
- W4R2B-04 … W4R2B-07, W4R2A-06, W4R2A-07 (low, info): resolved concrete
  with a thin cover ended in a bare `ValueError` after meshing; stale texts
  (level ratios, station memory, the junction table, the refusal of
  W4B-03); tests that failed on the base for the wrong reason.
- **B8 (open point of stream `fw4int`):** the 3-D model is right; `mom`'s
  floating port at the benchmark's 0.5 m is not converged. Refined to
  1/32 m it falls by 0.20 … 0.75 % and ends 0.15 … 0.21 % above `fem`, and
  on a coaxial rod and ring the axisymmetric model confirms the 3-D
  transfer resistance to 1.3e-3.

Confirmed: golden 563 of 563 quantities identical in both audits; one
near-interface warning per solve in every backend that should warn and none
from the axisymmetric model; no false positive over 868 tests and the
tutorials; the depth residual, the coaxial rings at 10 cm and the brace
effect reproduce to the digit; the refusals of W4B-03 and W4B-08, the
determinism of W4B-07 and the memory keys hold.

### Streams `fw4r2a` and `fw4r2b`: fixes of audits W4R2A and W4R2B

- **`fw4r2b`, wires beside each other (W4R2A-01, W4R2B-03):** two wires run
  beside each other where a point of one lies within 1.5 R of the side of
  the other and the two run within 30° of parallel, or where the whole of
  one lies within 1.5 R of another it does not meet; junctions, crossings
  at more than 30°, a rod on a ring and collinear pieces are left alone.
  The model keeps such wires 1.5 R apart by a smaller source radius for the
  whole world and refuses them below 2.2 bare wire radii (3.3 wire radii,
  axis to axis) before meshing, naming the wires, their distance, `mom` and
  `image`. Coaxial rings 4.5 … 10 cm apart lie within 1.72e-3 of the solid
  tori in every port entry and the bonded value, inside the estimate; 3 cm
  is refused by the node limit, the diagonal rings of W4R2A-01 and the ring
  at a rod head before meshing. The window term of another port counts
  only where the two wires meet; the coordinator kept this against the
  literal brief, since dropping it would put an L of two bonded strips
  6.3e-4 below the same L as one polyline (1.0e-4 with it) without fixing
  any audited case.
- **`fw4r2b`, the own wire (W4R2B-01, -02):** the axis distance comes from
  the cross product (the 20 m oblique strip matches its closed form to
  4e-16); the own polyline counts whole only where it has turned by less
  than 120° (the tilted hairpin −1.9e-4, `mom` −2.0e-4; the jump at the
  threshold 1.5e-4; the auditor's axis-line rule moved rings away from the
  solid torus and was not taken).
- **`fw4r2b`, the rest (W4R2B-04 … -07, W4R2A-07):** resolved concrete with
  a thin cover is refused before meshing; the new
  `fem_error_estimate_transfer` (6.3 times the transfer error on the
  coaxial analogue of B8); the B8 note says that `mom` at 0.5 m is not
  converged; the level ratios (rods 3.3 … 4.4, rings 3.0 … 3.4) and the CG
  refusal corrected; the W4B-01 tests fail on their base by assertion;
  `solid_earth` defaults to 2 million nodes. The AP1 station and the house
  mesh are unchanged to 7 digits.
- **`fw4r2a`, the ring block (W4R2A-02):** graded towards the points where
  the interface cuts the wire at a narrow angle or comes closest to it, in
  angle and radius, from cells of 1e-3 wire radii, with 32 instead of 64
  intervals. The estimate bounds the error in all 113 cases of the
  auditor's sweep at the default and at `refinement=1`; the default lies
  within 5.1e-5 of the boundary elements up to a contrast of 1000 (3.2e-5
  at 100), at fewer nodes than before (a touching ring 151 807 nodes,
  1.8 s, 0.76 GB).
- **`fw4r2a`, the warnings (W4R2A-03 … -05):** the 3-D dispatch warns for
  rod ends at an interface; the reach table is re-measured on six ring
  geometries and covers all 2348 measured points that are 1 % off (34 lay
  outside the old one); cut wires are reported from a contrast of 1.4; no
  false positive over 741 tests, T01–T10 and 48 golden cases; `diagnose()`
  with a `fem` engine is silent for a world the axisymmetric model takes.
- **`fw4r2a`, the rest (W4R2A-06, -07):** texts; a ring of radius 1.2 wire
  radii is solved (−7e-5 against the boundary elements) instead of
  refused, since no backend could otherwise.
- **New, found by `fw4r2a`:** two mesh lines 5.6e-17 m apart (an interface
  at 0.3 m and a ring or rod tip at 0.1 + 0.2 m) gave a singular factor or,
  silently, 6809 Ω instead of 647 Ω; features within 1e-12 were one line,
  since audit W4R3A-01 within 1e-9 of max(1 m, x).
- **Golden:** 563 of 563 quantities identical in both streams; full suite
  after the merge in the release section.
