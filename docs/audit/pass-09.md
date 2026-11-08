---
title: Review pass 9 (2026-07-28)
description: Review pass 9 (2026-07-28) of groundfield, as recorded in the changelog.
---

# Review pass 9 (2026-07-28)

| | |
|---|---|
| Date | 2026-07-28 |
| Reviewed | 0.14.0 (commit `428442d`) |
| Findings | F01–F57 |
| Released in | 0.14.1 (2026-07-30: F03, F12, F19) and 0.15.0 (2026-07-31: the rest); F01, F02 and F32 stayed open until 2026.11.0 |
| Overview | [Audit history](index.md) · [Changelog](../changelog.md) |

The ninth review pass ran eight parallel subsystem reviewers and an independent adversarial verification stage. Its findings are referenced by their pass-9 identifiers F01–F57.

!!! note "Where this text comes from"
    This page was moved verbatim from `CHANGELOG.md` when the release notes were
    condensed in 2026.11.0 (source: commit `30a122d`). Only the markup was adapted:
    reST roles are rendered as code or math, relative links point to the current
    pages, and each block stands under a heading that names where it was published.
    Paths, test names and numbers are those of the time; the code has changed since.

!!! warning "Corrections (review pass 10, finding 10.82)"
    Some statements below were found to be wrong in review pass 10. The text is
    kept as it was published; the corrections are listed in the release notes of
    2026.11.0 under *Corrections to the 0.15.0 release notes*:

    - **F47, F48** (`ManhattanGridPlacement.jitter_m`, `MeasurementLeadConfig.depth_m`)
      and **F53, F54** (`check_segment_resolution` for polyline and star electrodes)
      are listed as fixed, but `generators/placement.py` and
      `generators/measurement.py` were unchanged since 0.14.1 and `diagnostics.py`
      since 0.13.0. They were fixed in 2026.11.0 (F47 and F48 as part of 10.55).
    - **`cim` “bit-identical” to `image_2layer`** was false for electrodes with unequal
      segment lengths: +4.1 % for a 3 m rod in `TwoLayerSoil(100, 400, h_1=2)` split
      unevenly at the interface (segment length 0.4 m: 71.060 Ω against 68.262 Ω).
      Fixed in 2026.11.0 (10.30).
    - **F16** describes the symptom of F08: the layout builder (`OrtsnetzLayout.to_world`)
      still ignored the concrete shell; the 0.15.0 fix covered `TnNetworkGenerator`
      only. Fixed in 2026.11.0 (10.14).
    - **F17** (documentation of backends that raise for three or more layers) was
      fixed only in part: the engine-comparison example still recommended
      `image_nlayer` or `cim` for more than two layers. The documentation was
      rewritten in 2026.11.0.

## Released in 0.14.1 (2026-07-30)

### Fixed (Review pass 9 — implemented 2026-07-28)

> The ninth review pass was run on 2026-07-28 against `0.14.0` (commit
> `428442d`) with eight parallel subsystem reviewers and an independent
> adversarial verification stage. Findings are referenced below by their
> pass-9 identifier (F01–F57). This release deliberately carries **only the
> three findings that are either data-corrupting or hard crashes**, so that a
> `groundfield` version number is enough to decide whether a stored
> `groundinsight` `BusType` is affected. The remaining findings ship in
> `0.15.0`.

- **⚠ `fit_to_sympy` lost the sign of the residue's imaginary part —
  the exported `groundinsight` formula did not match the fit it exports**
  (`postprocess/vector_fitting.py`, finding F03). When combining a
  complex-conjugate pole/residue pair into the real second-order term
  $r/(s-p) + r^{*}/(s-p^{*}) = [2\,\Re(r)\,(s-\Re(p)) - 2\,\Im(r)\,\Im(p)] /
  [(s-\Re(p))^2 + \Im(p)^2]$,
  the canonical residue is by construction the one sitting at the pole with
  $\Im(p) > 0$. The previous symmetrisation used
  `0.5 * (abs(r.imag) + abs(r2.imag))`, which forces $\Im(r) \ge 0$; the
  following `if p.imag < 0` correction flips on the sign of the *pole*, not of
  the *residue*. Whenever the canonical residue carried $\Im(r) < 0$ — which
  vector fitting produces routinely for lossy earth-return driving-point
  impedances — the $-2\,\Im(r)\,\Im(p)$ numerator term entered with the wrong
  sign and the exported `BusType.impedance_formula` was a **different rational
  function** from `VectorFitResult`. Measured on a
  $Z_0\coth(\gamma\ell)$ line fit ($R = 0.3$, $L = 1$ mH, $G = 1$ nS,
  $C = 10$ nF, $\ell = 5$ km, 6 poles): 28.1 % magnitude error at 16.7 Hz,
  13.1 % at 50 Hz, and a *sign* difference in $\Re(Z)$ at 500 Hz. The fix
  selects the residue belonging to the positive-imaginary pole and
  symmetrises as $\tfrac12(\Im(r_+) - \Im(r_-))$, which preserves the sign
  while keeping the existing exact-conjugate symmetrisation intent.

  **Numerical results change.** Any `BusType` exported through
  `to_bustype_dict` / `save_bustype_json` / `to_bustype` from a fit with at
  least one negative-imaginary canonical residue is affected and should be
  re-exported. `VectorFitResult` itself, `fit.evaluate()`, and every solver
  result were always correct — the defect was confined to the SymPy rendering
  step, so re-running `fit_to_sympy` on the stored fit is sufficient; the
  field computation does not need repeating.

- **`plot_potential_contour` raised `AttributeError` on every supported
  NumPy when `extent` was omitted** (`postprocess/plotting.py`, finding F12).
  The automatic-extent branch called the `ndarray.ptp()` *method*, removed in
  NumPy 2.0, while `pyproject.toml` requires `numpy = "^2.1.0"` — so the
  documented default call, including the one in
  [Quickstart](../getting-started/quickstart.md), was dead. Now uses the free function
  `np.ptp(...)`. A repo-wide sweep for the other NumPy-2.0 removals
  (`itemset`, `newbyteorder`, `np.float_`, `np.complex_`, `np.NaN`, `np.Inf`,
  `np.alltrue`, `np.product`) found no further occurrence.

- **`Engine.with_frequencies()` bypassed all frequency validation on its
  default path** (`solver/engine.py`, finding F19). The
  `preserve_order=False` branch returned
  `self.model_copy(update={"frequencies": freqs})`, and `model_copy(update=...)`
  runs **no** validators under pydantic v2 — contradicting the method's own
  docstring ("the standard `_validate_frequencies` checks run"). An empty
  list, a negative frequency, NaN and infinity were all silently accepted, and
  the `EngineFrequencyOrderWarning` for a non-monotonic list never fired.
  Both paths now go through `model_validate`; the order warning is re-emitted
  at the caller's frame so the notebook line that built the engine is the one
  reported. The scalar checks moved into a shared
  `_check_frequency_values` helper so the constructor and the convenience
  constructor reject the same inputs with the same messages, and infinity is
  now rejected explicitly (previously it fell through the `f >= 0.0` guard).
  DC (`f == 0.0`) remains a legitimate quasi-static operating point.

New `tests/test_audit_pass9_fixes.py` (14 tests): the residue-sign export for
both signs and across a frequency sweep, the full
`to_bustype_dict` → `BusTypeSpec` → `evaluate_spec` round trip against the
analytic pair sum, `plot_potential_contour` without `extent` in both planes,
and the five `with_frequencies` rejection cases plus the order-warning and
`preserve_order` controls. Eleven of the fourteen fail on `428442d`; the other
three are controls over behaviour that was already correct.

### Docs

- **`docs/examples/09_plot_gallery.md` — corrected the
  `plot_potential_contour` cross-section snippet.** It passed a
  non-existent `fixed=0.0` keyword (the parameter is `y` for `plane="xz"`
  and `z` for `plane="xy"`) and advertised a `plane="yz"` option the
  function does not implement. The surrounding prose now states the
  plane/fixed-coordinate pairing and the `extent=None` behaviour.

## Released in 0.15.0 (2026-07-31)

### Review pass 9 — the remaining findings

> This block closes the rest of the ninth review pass (2026-07-28 against
> `0.14.0`); `0.14.1` shipped only F03, F12 and F19. It is a **minor**, not a
> patch, because several findings could not be closed honestly without also
> taking their behaviour-changing twin: F06 needs F05, F10 needs F11, F26 needs
> F29, and F39 needs F34. Findings are referenced by their pass-9 identifier.
>
> **Read this before regenerating anything.** Entries marked ⚠ change numerical
> results. In descending order of blast radius: every `fem` number moves; every
> layered `mom_sommerfeld` number moves; `coupling.layered_green` moves for
> surface, interface, near-interface and cross-layer pairs and for radii beyond
> ~160 m; the `image` family moves for electrodes with *unequal* segment lengths
> (uniform-segment geometries are unchanged to floating-point rounding — 1 ulp on
> 4 of 11 test worlds, because a BLAS dot replaces `np.mean`); `references.pollaczek`
> and two `references.dwight1936` formulas move; and `rho_f_standard` fits move
> wherever a sweep contained dead frequencies.
>
> Still open after this release: **F01** (the simulated fall-of-potential
> measurement is a null experiment — `Source.return_to` is inert), **F02** (the
> ADR-0007 cross-layer current-sharing ansatz, up to +581 %) and **F32**
> (`carson_series` on the wrong image baseline, 1.65×–2.17× too much earth-return
> reactance). All three change what a study *means* rather than what it computes
> and are staged separately. The deferred half of F26/F29 — the exact image-line
> diagonal — is quantified under *Documentation* below.

### Fixed — the numerical core

- **⚠ `mom_sommerfeld`: the reference engine no longer truncates a conditionally
  convergent integral (F04, F40).** The top-layer kernel was evaluated with a bare
  `scipy.integrate.quad` on a finite `[0, lam_max]`. The direct term
  $e^{-\lambda|z-z_s|}J_0(\lambda s)$ does not decay for two segments at equal
  depth, so the integral converges *only* through the Bessel oscillation and the
  cut left a sign-alternating error no tolerance could reduce:
  `sommerfeld_kernel_value` returned **−0.012551** where the exact value is
  **+0.039990**, and a 25 m ring in `TwoLayerSoil(100, 100, h_1=1)` came out at
  1.4952 Ω instead of 1.5751 Ω. The kernel is now split with Lipschitz' integral
  $\int_0^\infty e^{-\lambda d}J_0(\lambda s)\,d\lambda = 1/\sqrt{s^2+d^2}$ — the
  direct pair and the leading interface reflection are closed-form, and only the
  exponentially decaying remainder is quadratured. `lambda_max` now follows that
  remainder's decay length instead of ignoring $|z-z_s|$ (two segments 2 cm apart
  at 3 m depth previously lost 44.8 % of their reaction-matrix entry, and the error
  *grew* under refinement). Against the analytic Tagg/Sunde series the kernel is
  now accurate to ~3e-15 for $s \le 50$ m; the engine is also **150–8000× faster**
  (32-segment ring: 162 s → 0.02 s).
- **⚠ `mom_sommerfeld`: unresolved oscillation no longer returns a negative
  Green's function, and non-convergence is no longer silent.** The panel budget
  truncated the panel *count*, so the resolved λ-range **shrank** with the
  refinement factor while the loop returns its last rung — refining made the answer
  worse. For a centimetre-scale intermediate layer at kilometre separation the
  kernel returned −9.4749e-05 where the far-field asymptote requires +5.000e-04,
  with **zero** Python-level warnings. The budget is now spent coverage-first, and
  exhausting the ladder raises the new `SommerfeldConvergenceWarning`, naming the
  achieved versus requested tolerance, the offending geometry, that **the sign may
  be wrong**, and which parameter to raise. Verified silent across 1260 in-envelope
  parameter sets. A geometric panel family additionally resolves the
  multiple-reflection pole $(1-\Gamma_1(0))/2h_1$, which the graded grid missed
  entirely at extreme contrast (ρ₂/ρ₁ = 10⁸: −5.2e-2 → −2.8e-11).
- **⚠ `coupling.layered_green`: the λ-truncation is now guaranteed instead of
  assumed (F14).** `lambda_max` was sized from
  `lambda_max_factor / min(h_1, s + z + z_s)`, which bears no relation to the decay
  length $d_\text{img}$ of the reflected remainder. Every image distance can
  degenerate, and the kernel was then wrong with no warning: −2.7e-2 for a pair at
  the air/soil surface, +4.1 % at the layer interface, and **−39.5 %** for a pair
  half a millimetre below it — the last one *growing as the mesh is refined*
  towards $h_1$, which destroyed mesh convergence for exactly the cross-layer rods
  ADR-0007 exists for. Cross-layer pairs straddling the interface had the same
  defect. The reflected kernel is now written as its exact image superposition;
  every term with $\lambda_\text{max}d_j < 30$ is subtracted from the integrand and
  added back through its exact Hankel inverse $c_j/(2\sqrt{s^2+d_j^2})$, so the
  discarded tail is bounded by $e^{-30}$ **by construction**. All of those
  geometries, and radii from 0 to 20 km, now agree with an independently derived
  closed form to ~3e-10 — and the result is identical to ten digits for
  `lambda_max_factor` anywhere in 30 … 30 000.
- **⚠ `coupling.layered_green`: the 48-node interpolation and its 64-pair
  discontinuity (F15).** The group interpolation was 2–3 orders less accurate than
  documented and stepped by ~3 % at the pair-count threshold where the strategy
  switches — a discontinuity in *problem size*, which is indefensible in a
  reference solver. Residual step is now 3.4e-5, and the interpolated-versus-exact
  agreement ~1e-6 (worst 1.1e-5).
- **⚠ `image` / `image_2layer`: reported potentials are the length-weighted
  (Galerkin) average the solver actually enforces (F10, F11).** 0.11.0 (WP-B3)
  moved the multiport matrix to a length-weighted reduction but left the
  result-assembly path on a plain `np.mean`, so `grounding_impedance()` disagreed
  with the impedance the solver had constrained by −10.9 % on a rod split at a
  layer interface, and `cluster_impedance` read `members[0]` — the *alphabetically
  first* name — so **renaming an electrode changed the reported impedance**
  (19.4393 → 19.2314 Ω on identical geometry). `grounding_impedance(e)` now equals
  the diagonal $Z_{ii}$ to ≤2 ulp, all members of one galvanic cluster report equal
  potentials to 2e-16 (was 3.1e-3), and the reported value is name-independent.
  Mixed-length geometries move by up to +12 %; uniform-segment geometries are
  unchanged to rounding.
- **⚠ `fem`: the axial mesh no longer degrades with the declared layer thickness
  (F05, F06).** `_build_axisymmetric_mesh` sized the domain from
  `a_eq + sum(h_layers)` and then filled the axial direction with a fixed 40
  *linearly* spaced nodes, so near-electrode resolution collapsed in proportion to
  a parameter that must have no effect. A physically homogeneous soil declared as
  `TwoLayerSoil(100, 100, h_1=X)` returned 40.36 / 34.94 / 23.69 / 15.66 / 7.51 Ω
  for X = 1/2/5/10/30 m against an exact 64.62 Ω — up to −79 %, monotone in X. The
  mesh is now a geometrically graded spherical shell; the same sweep is flat within
  −0.24 … −0.84 %.
- **⚠ `fem`: the Dirichlet electrode is now genuinely conforming (F24).** It was a
  staircase flat disc, so the discrete electrode did not converge to the intended
  hemisphere: refining changed the answer *without* converging (−31.9 % → −2.0 % →
  **+20.1 %** on a nested ladder). The electrode surface $r = a_\text{eq}$ is now an
  exact mesh line. Verified independently across six geometries: convergence is
  order **2.00**, strictly one-sided and strictly shrinking. A second, subtler
  non-conformity found in re-audit is also closed — the Dirichlet node set was
  captured *before* the layer-interface cut, so for any interface shallower than
  $a_\text{eq}$ the cut left a free node in the middle of the energised surface
  (an insulating patch). That affected every top-layer thickness below
  $a_\text{eq}$, which for a 100 × 100 m mesh electrode is the whole AP1 soil grid.
- **⚠ `references.pollaczek`: the buried-conductor kernel used Carson's air
  exponent (F09).** Both `pollaczek_self_impedance` and
  `pollaczek_mutual_impedance` integrated $e^{-2h\lambda}$, correct only for a
  conductor *in air* where $\gamma \approx 0$. The Pollaczek/Sunde form for a
  conductor buried in the earth requires $e^{-2h\,u}$ with
  $u = \sqrt{\lambda^2 + j\omega\mu_0/\rho}$. Measured error in $R'$: −0.5 % at
  50 Hz / $h$ = 1 m, −2.1 % at 1 kHz / 1 m, −5.8 % at 1 kHz / 3 m, −54 % at 1 MHz;
  $X'$ was affected far less (< 0.2 % in band). **The 0.14.0 notes recorded the
  consequence as physics** — "$R'$ sits just below $\omega\mu_0/8$, the deficit
  growing with frequency" — and that was the bug's signature: with the correct
  kernel $R'$ sits *above* $\omega\mu_0/8$ and *rises* with burial depth, as a
  buried conductor's earth-return resistance must. The claim is corrected in the
  module docstring, the API page, ADR-0006 and the tests that encoded it.
- **⚠ `references.dwight1936`: two reference formulas were wrong.**
  `horizontal_strip` used the round-wire bracket constant −2 instead of the strip
  constant −1 (~10 % low, F37); `vertical_round_plate` applied a factor 2 twice and
  so halved the plate–image term (F38).
- **⚠ `postprocess.rho_f_standard`: dead frequencies are no longer fitted as
  $Z = 0$ (F41)** — the same defect that had already been fixed in the
  vector-fitting sibling. A zero impedance is not missing data; it drags the fit
  towards the origin.

### Fixed — guards that were missing or wrong

- **`image` family: the `z <= 0` rejection now covers every backend (F26, F29).**
  The self-image term $1/(2z)$ was *clamped* at a 1 mm floor instead of guarded, so
  a horizontal electrode at $z = 0$ silently returned a resistance ~15× too high
  and mesh-dependent, and a ring at $z = -100$ m solved to 5.5162 Ω as if buried at
  $+100$ m. The guard now lives in the shared reaction-matrix kernel, so `mom`,
  `bem`, `cim`, `mom_sommerfeld` and `mutual` see it too — previously only `image` /
  `image_2layer` raised, and `compare_engines` therefore disagreed about whether a
  world was valid at all. Validation looks at segment **extent**, not just
  midpoints: a rod with its top 0.5 m in the air used to solve to 106.2 Ω. Messages
  name the electrode — or the *conductor*, not the internal
  `__cond_<name>__seg_<k>` pseudo-node.
- **New `ShallowSegmentWarning` for the legal-but-shallow regime.** $L > 4z$ is
  admissible but the point-image stand-in for the image line biases the diagonal by
  $\gtrsim 13\,\%$, which is why refining the mesh keeps moving the answer there.
  The warning quantifies the bias of the worst offender. Its quoted figure is the
  horizontal-segment one, and the extent guard above provably restricts the warning
  to segments inclined less than 30° from horizontal, so the figure now always
  applies to the case that triggers it.
- **`FieldResult.cluster_impedance` no longer drops a member's current.** The new
  member filter was applied to the current sum as well as the potential average, so
  potentials `{"a": 10 V}` with currents `{"a": 1 A, "b": 1 A}` returned 10 Ω
  instead of 5 Ω. The average keeps the filtered set; the current sum runs over all
  members again.
- **`coupling.layered_green`: negative depths returned `nan` silently** and
  propagated all the way to `grounding_impedance()`. All four entry points now
  raise `ValueError`.
- **`Engine.solve` forwards the image-series controls to `image_nlayer` (F18).**
  `image_max_terms` / `image_series_tol` were forwarded to `solve_image_2layer` but
  the `image_nlayer` call took no arguments, so that backend silently used its own
  defaults although the field documentation promised otherwise.
- **`World.solve` no longer detaches the `Source` objects `create_source` returned
  (F31)** — a handle the user held went stale, silently.
- **`compare_engines`: the deviation metric is pairwise, not mean-referenced
  (F35), and `is_consistent` is no longer `True` when nothing was compared
  (F36).** The old metric admitted about twice the documented tolerance, and an
  empty comparison reported a false green.
- **`generators`: `building_counts` key order no longer changes the sampled world
  for a fixed seed (F33)** — a reproducibility defect, and the AP1 Monte-Carlo runs
  depend on a seed pinning a world.
- **`generators`: footprint-driven foundations land on the OMBR centre (F25)** —
  the oriented-bounding-rectangle centre was discarded, displacing the generated
  rectangle from the footprint it was derived from.
- **`generators`: `route_manhattan`'s `escape_radius_m` now covers the final
  bridge segment (F42)**, so an anchor inside a footprint no longer always raises —
  the escape valve did not cover the case it exists for.
- **`generators`: the ADR-0012 concrete shell is no longer silently dropped
  (F16).** The registry was written under the electrode's own name but read under
  the cluster anchor's, so switching concrete on had no effect whenever the
  foundation was not the first surviving electrode of its site: a factor-33 change
  in `concrete_rho_ohm_m` produced bit-identical impedances.
  `GroundingSystemSpec.build_at` now folds a site's entries onto the anchor once
  the bonds are in place. Several encased electrodes in one cluster combine **in
  parallel**, not in series — each leaks through its own radial shell.
- **`io.csv`: the frozen column-name constants named columns the writers never
  emit (F46)**, so a consumer written against the published constants got a
  `KeyError`.
- **`fem`: `r_far_factor` was a dead knob.** `r_far` was
  `max(r_far_factor, z_far_factor) * base_length`, so every value ≤ 20 returned the
  same truncation radius with no warning — a truncation study swept a parameter
  with no effect. It now sets the sphere on its own; `z_far_factor` is deprecated,
  still acts as a lower bound when passed, and logs when it overrides.

### Changed

- **`cim` and `bem` no longer compute a complex-image fit, and the metadata says
  so (F34, F39).** Both reject $n \ge 3$ up front, so the only reachable paths use
  exact closed-form self-kernels and ignored the fit entirely — yet
  `fit_complex_images` ran on every solve and its failed result was advertised as
  `cim_n_images` / `cim_rms` (a NaN). `cim` returns values bit-identical to
  `image_2layer` and `bem` to `mom`; the metadata now states that plainly.
  `fit_complex_images` itself learned to represent the $\lambda\to\infty$ asymptote
  of $\Gamma_1$ and to report failure instead of returning a fit whose residual is
  meaninglessly $|K_1|$.
- **`coupling.layered_green`: `lambda_max_factor` is now an upper bound on the
  truncation, not the truncation itself.** Raising it is normally a no-op — that is
  the point of the F14 fix — while lowering it moves work into the closed-form
  image sum and is equally exact.
- **`generators`: `MeasurementLeadConfig.depth_m` is honoured (F48)** — it was
  sampled and discarded, so buried and overhead measurement leads were
  geometrically identical.

### Added

- **`Engine.sommerfeld_lambda_max_factor`, `sommerfeld_epsabs`,
  `sommerfeld_epsrel`, `sommerfeld_max_osc_panels`.** The accuracy knobs of the
  package's *reference* engine were unreachable from the public API — `Engine.solve`
  called `solve_mom_sommerfeld(world, self)` with no keywords. Defaults equal the
  previous signature defaults, so existing callers are bit-identical, and all four
  are echoed into `FieldResult.metadata`.
- **`SommerfeldConvergenceWarning`**, **`SommerfeldTailTruncationWarning`** and
  **`ShallowSegmentWarning`**; the last is now re-exported from
  `groundfield.solver` and documented on the solver API page.
- **Executable documentation (F27).** `tests/test_pass9_doc_snippets.py` extracts
  every fenced `python` block under `docs/` and runs it, one namespace per page,
  with an explicit `<!-- skip-doctest -->` marker for illustrative fragments and
  for the network-dependent OSM examples. Nothing had ever executed the
  documentation's code, and six live snippets were broken.

### Performance

- **`coupling.layered_green`: multi-kilometre spans no longer explode.** With the
  tail bounded analytically, `lambda_max` no longer has to be large. A 4000-pair
  group at `s_max = 4 km` runs in **71 ms** against 199 ms on 0.14.1 and 11.2 s in
  the intermediate development state; a scalar 6 km kernel 30 → 8 ms; a 200×20
  `two_layer_probe_matrix` 120 → 9 ms. Accuracy is unchanged or better everywhere.
- **`mom_sommerfeld` is 150–8000× faster** and **`fem` about 15× faster**, both as
  a side effect of the correctness work above rather than of tuning.

### Documentation

- **ADR-0006** carried the wrong Pollaczek exponent in its 2026-07-20 amendment and
  now records the correction and why the old behaviour looked like physics.
  **ADR-0011** claimed `offset_xy_m` is "zero by construction"; it is not.
  **ADR-0002**, `docs/concepts.md`, `docs/quickstart.md` and the engine pages no
  longer advertise `cim` as the primary $n \ge 3$ engine and `bem` as its
  alternative — both have raised `NotImplementedError` since 0.11.0 (F17) — and now
  state plainly that eight backends are **six distinct computations**, with
  `mom_sommerfeld` the only integral-equation path for $n \ge 3$ and `fem` the
  methodologically independent cross-check.
- **Accuracy claims corrected where they were optimistic**, not just where they
  were wrong: four "~1e-8" claims in `layered_green` (measured ~1e-6, worst
  1.1e-5), `mom_sommerfeld`'s "10⁻¹⁴ level" per-pair error and its "≈1 ms" per-pair
  cost (measured up to 3.25 s for a single kernel evaluation in the thin-layer
  corner, with the $O(N^2)$ consequence spelled out), and `fem`'s "element aspect
  ratios O(1)" (true away from the interface cut, not at it).
- **The deferred half of F26/F29 is documented rather than hidden.** Replacing the
  point image $1/(2z)$ by the exact image-line integral was *not* done. The residual
  bias is stated with numbers — a 10 m tape at $z = 0.01$ m gives 54.361 / 35.491 /
  26.603 / 22.039 / 21.043 Ω for `ds` = 1.0 / 0.5 / 0.25 / 0.1 / 0.05, i.e. **+69 %
  at `ds` = 0.5** — together with the trap that made the naive fix wrong (the
  horizontal $\operatorname{arsinh}$ form applied to a *vertical* rod segment is
  worse than the point form) and the collocation-versus-Galerkin distinction of the
  image-line term.
- `earth_return`'s module docstring claimed the earth-return reactance *decreases*
  with frequency; it increases (F55). The `rho_f_standard` docstring swapped the
  physical meaning of $k_2$ and $k_3$ (F56). `check_segment_resolution` skipped
  polyline and star electrodes (F53, F54). `ManhattanGridPlacement.jitter_m`
  carried a `ge=0.0` constraint on a `float | Distribution` union, making the
  documented distribution form unusable (F47).

### Internal

- New regression modules `tests/test_pass9_*.py`, one per subsystem, plus
  `tests/test_audit_pass9_fixes.py` from 0.14.1. Each test was checked against a
  clean `v0.14.1` export so that it demonstrably fails on the old code; the
  deliberate controls over already-correct behaviour are labelled as such in their
  docstrings. One pre-existing assertion was **tightened** (`test_fem.py`, 0.25 →
  0.01) and one narrowed to exempt a single legitimately-earned warning
  (`test_sommerfeld_inductance.py`) — no tolerance was loosened.
