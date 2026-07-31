# ADR-0002: Engine family for layered soils

| | |
|---|---|
| **Status** | Accepted — amended 2026-07-09 (audit WP-E) and 2026-07-30 (review pass 9, F17/F34/F39); see the amendments at the end of this document, which supersede the selection heuristic below |
| **Date** | 2026-05-01 |
| **Deciders** | Project maintainers |
| **Scope** | `groundfield` |

## Context

ADR-0001 settled the methodology for the **2-layer** case: closed-form
Tagg/Sunde image-charge series (`image_2layer`) plus a Galerkin MoM
cross-validation engine (`mom`). For two-layer soils in isolation
that is enough. Two outstanding issues motivate this ADR:

1. **Three or more layers.** Although the default scope targets a
   2-layer soil, several supporting cases involve a weathered or
   frozen surface layer above a representative two-layer stack
   (i.e. effectively 3 layers). The closed-form Tagg/Sunde series
   does not extend cleanly to $n \ge 3$: the upward-looking
   reflection $\Gamma_1(\lambda)$ stops being constant in $\lambda$
   and a real geometric image series develops a doubly-nested
   expansion that becomes fragile at hard contrasts.
2. **Independent cross-validation.** The current engine family
   (`image_2layer` / `mom`) shares the same Green's-function kernel,
   so its agreement reflects the resolution scheme but not the
   physics layer. ADR-0001 already lists the addition of a
   methodologically independent reference engine as an open action
   item.

A literature review of recent layered-soil grounding work
(Dawalibi 1991, Meliopoulos 1993, Güemes 2004, Colominas 2007/2012,
Li 2006, Zou 2015, Dan 2021) yielded four method families that map
cleanly onto the existing `groundfield` data model:

- **Image-charge series** (Tagg/Sunde 2-layer; Stefanescu/Sunde n-layer).
- **Complex Image Method** (matrix-pencil or segmented-sampling
  least-squares fit of $\Gamma_1(\lambda)$).
- **Direct Sommerfeld quadrature**, optionally with the Zou et al.
  complex integration path.
- **Boundary-element method** (Colominas) and **finite-element
  method** (Güemes), each adding either a different test-function
  weighting (collocation vs. Galerkin) or a different problem form
  (volume PDE vs. integral equation).

## Decision

`groundfield` ships eight backends arranged in three families:

| Family | Backends |
|---|---|
| Closed-form image-charge | `image`, `image_2layer`, `image_nlayer`, `cim` |
| Integral equation | `mom`, `mom_sommerfeld`, `bem` |
| Volume PDE | `fem` |

The selection heuristic *as decided in 2026-05* was:

1. **Homogeneous soil** → `image` (cheapest, closed form).
2. **2-layer soil** → `image_2layer` (auto-dispatched from
   `backend="image"`). Cross-check with `mom`, `cim`, `bem`,
   `mom_sommerfeld`.
3. **3+ layer soil** → `cim` is the recommended primary engine
   (closed form, cost independent of layer count once fitted).
   `mom_sommerfeld` is the reference for hard contrasts.
   `image_nlayer` raises a `ValueError` for `n ≥ 3` (the real
   image-charge series is not implemented for that regime).
4. **Volume cross-check** → `fem` (axisymmetric, equivalent-hemisphere
   reduction). Used as a third independent line of defence; not
   intended for general meshing.

!!! danger "Superseded — items 2 and 3 above are no longer valid"

    Item 3 was revoked by the 2026-07-09 amendment (`cim`/`bem`
    reject `n ≥ 3`), and item 2 overstates the cross-check value of
    `cim`/`bem`. The heuristic **in force** is:

    1. **Homogeneous soil** → `image`.
    2. **2-layer soil** → `image_2layer`. Resolution cross-check with
       `mom`; *independent* cross-check with `mom_sommerfeld`
       (quadrature) or `fem` (volume PDE). `cim` reproduces
       `image_2layer` bit-for-bit and `bem` reproduces `mom` to
       4e-16 — running them adds no information.
    3. **3+ layer soil** → `mom_sommerfeld` (the only engine that
       evaluates the full layered Green's function in this regime),
       cross-checked against `fem`. `cim`, `bem` raise
       `NotImplementedError`; `image_nlayer` raises `ValueError`.
    4. **Volume cross-check** → `fem`, as before.

    See the 2026-07-30 amendment for details.

## Mathematical / physical model

All seven layered engines evaluate the same Sommerfeld integral of
the quasi-static layered Green's function

$$
\varphi(s, z) \;=\; \frac{\rho_1\, I}{4\pi}
\int_0^{\infty} \bigl[
  e^{-\lambda |z - z_s|}
+ \Gamma_1(\lambda)\, e^{-\lambda (z + z_s)}
\bigr]\, J_0(\lambda s)\, d\lambda,
$$

with the recursive upward reflection $\Gamma_1(\lambda)$ built bottom
up from the per-interface Fresnel coefficients
$K_i = (\rho_{i+1} - \rho_i)/(\rho_{i+1} + \rho_i)$. Differences
between the engines lie purely in the numerical strategy used to
evaluate the integral (real images, complex images, quadrature,
volume PDE). The frequency band targeted by `groundfield` is
$f < 1\,\text{kHz}$, where the quasi-static formulation is valid.

## Cross-validation envelope

For simple configurations all engines must agree within tightly
defined tolerances (codified in
`tests/test_cross_engines_extended.py`):

| Pair | Tolerance | Notes |
|---|---|---|
| any closed-form image vs. another closed-form image | 1e-9 | exact reduction (e.g. `image_nlayer` → `image_2layer`) |
| closed-form image vs. `cim` | 1e-9 | identity, not a check — `cim` *is* the closed-form kernel |
| `mom` vs. `bem` | 1e-9 | identity, not a check — same reaction matrix (4e-16) |
| closed-form image vs. `mom` / `bem` | 5 % | resolution scheme differs |
| `mom_sommerfeld` vs. closed-form layered engines | 5 % | quadrature is the reference; the only independent kernel |
| `fem` vs. integral engines | 10 % | equivalent-hemisphere bias; independent problem form |

Layer-contrast sweeps must produce a monotonically increasing
cluster impedance for every engine — a basic physics consistency
check.

## Consequences

**Positive.**

- Three independent methodologies (closed-form images, integral
  equation, volume PDE) are now available side by side. A bug in any
  single one is detectable through cross-comparison.
- ~~For 3+ layer soils there is a clear recommended engine (`cim`)
  with an independent reference (`mom_sommerfeld`) and a second-line
  cross-check (`bem`).~~ *Revoked 2026-07-09: for 3+ layers only
  `mom_sommerfeld` (with `fem` as cross-check) remains.*
- `fem` adds a methodologically distinct line of defence at the cost
  of a simple equivalent-hemisphere reduction; the bias is
  documented and bounded.

**Negative / open.**

- `mom_sommerfeld` is slow (per-pair adaptive quadrature). Since it is
  the only $n \ge 3$ path, that cost is now unavoidable in that
  regime rather than optional.
- ~~`cim` quality depends on the matrix-pencil fit; very hard
  contrasts may need more images. The fit RMS is exposed in
  `result.metadata["cim_rms"]`.~~ *Revoked 2026-07-30: no solve path
  evaluates the fit, and `cim_rms` is `None` with an explicit
  `cim_fit_used = False` flag. `fit_complex_images` survives as a
  standalone helper with honest failure reporting.*
- Two of the eight backends (`cim`, `bem`) are aliases of others on
  every soil they accept. They are kept for API stability and as
  entry points for future complete kernels, but the eight-backend
  count overstates the number of independent computations (six).
- `fem` covers single-cluster worlds via the equivalent-hemisphere
  reduction. Multi-cluster volume runs would need a real 3-D mesh
  generator and a stronger FEM kernel — out of scope for typical cases.

## Action items

- [x] Add `_layered.py` with the recursive $\Gamma_1(\lambda)$ and the
      shared `LayerStack` helper.
- [x] Implement `image_nlayer`, `cim`, `mom_sommerfeld`, `bem`, `fem`
      with NumPy docstrings.
- [x] Extend `Engine.solve` dispatcher and the `Backend` literal.
- [x] Notebooks `04`–`09` covering each engine and the joint
      cross-engine view.
- [x] `tests/test_image_nlayer.py`, `test_cim.py`,
      `test_mom_sommerfeld.py`, `test_bem.py`, `test_fem.py`,
      `test_cross_engines_extended.py`.
- [ ] Reference-case library for typical cases (separate notebooks).
- [ ] Optional: scikit-fem-backed 3-D `fem` (lifted from the
      equivalent-hemisphere reduction) for full multi-cluster
      worlds; deferred until a concrete use case demands it.


---

## Amendment 2026-07-09 — engine-family independence (audit 2026-07-08, WP-E)

Two corrections to this ADR's cross-validation narrative:

1. **`bem` and `mom` are not independent for n ≤ 2.** They assemble
   the identical reaction matrix from the same kernels and solve it
   with the same constraint solver (measured relative difference
   4e-16). Their agreement validates the shared discretisation, not
   the physics. Genuine methodological independence in the layered
   cross-checks comes from `mom_sommerfeld` (direct Sommerfeld
   quadrature of the full layered Green's function) and `fem`
   (volume PDE, with its documented equivalent-hemisphere reduction
   bias). `cim` shares the exact Tagg/Sunde kernel with
   `image_2layer` for n = 2.
2. **`cim`/`bem` reject n ≥ 3 soils.** The historic n ≥ 3
   complex-image kernel implemented an incomplete Green's function
   (single (z+z_s) image family; missing the 2·h_1-reflected
   families and the surface-interface multiple-reflection
   denominator — see `solver/_layered.py`) and systematically
   underestimated the layered correction. This ADR's designation of
   `cim` as the primary n ≥ 3 engine is revoked until a complete
   kernel exists; use `mom_sommerfeld` or `fem` for three and more
   layers.

---

## Amendment 2026-07-30 — the engine count is honest now (review pass 9, F17/F34/F39)

Three corrections, all in the direction of *saying what the code
does* rather than what the 2026-05 design intended.

1. **`mom_sommerfeld` is the only $n \ge 3$ path.** The
   documentation (`docs/engines/index.md` decision tree,
   `docs/engines/cim.md`, `docs/engines/bem.md`,
   `docs/concepts.md` backend table) still routed multi-layer users
   to `cim` (primary) and `bem` (alternative) nine months after both
   started raising `NotImplementedError`, including in a runnable
   worked example. Those pages now describe the actual behaviour:
   $n \ge 3$ → `mom_sommerfeld`, with `fem` as the independent
   volume-PDE cross-check, and `cim`/`bem` documented as
   `TwoLayerSoil`-max.

2. **No complex-image fit is computed on any solve path.** Up to
   0.14.1 `solve_cim` and `solve_bem` called `fit_complex_images` on
   every solve although both reachable regimes ($n \le 1$ and
   $n = 2$) use exact closed-form self-kernels that ignore it. Its
   failed result was published as
   `metadata['cim_n_images'] = 0` / `['cim_rms'] = nan`, inviting
   readers to judge an approximation with no influence on the answer.
   The call is gone; the metadata carries `cim_fit_used: False`,
   `cim_rms: None` and a new `reduces_to` key naming the backend the
   numbers are bit-identical to (`image` / `image_2layer` for `cim`,
   `mom` for `bem`). The unreachable $n \ge 3$ code — the complex-image
   self-kernel branch, the collocation-matrix branch, the whole
   `_cim_field_potential` helper and the two `n >= 3` cross-layer
   guards behind the `n >= 3` rejection — was deleted; the remaining
   $n \ge 3$ branches raise instead of computing a single-family
   Green's function.

3. **The fit itself reports failure.** `fit_complex_images` could not
   represent $\lim_{\lambda\to\infty}\Gamma_1 = K_1$ — the pole
   filters discarded exactly the constant term ($p = 1$,
   $\beta = 0$) that carries the asymptote — so its residual was of
   order $|K_1|$ for every stack, silently, and for two-layer stacks
   it returned $P = 0$ with `rms = nan`. It now splits $K_1$ off
   analytically as a $\beta = 0$ image, fits only the decaying
   remainder, re-solves the least-squares weights *after* the pole
   filter, derives the grid step from the structure scale
   $h_{\max}$ (not $h_{\min}$), verifies the result on an
   independent log-spaced grid, and warns via a dedicated
   `ComplexImageFitWarning` when either residual exceeds `rms_tol`.
   Measured on $\rho = [100, 400, 50]\ \Omega\text{m}$,
   $h = [2, 3]\ \text{m}$: RMS $0.585 \to 2.1 \cdot 10^{-13}$; on
   a two-layer stack: `nan` $\to 0$ exactly.

**Consequence for cross-validation tables.** A table listing
`image_2layer`, `cim` and `bem` as three agreeing engines reports one
computation up to three times. The independent checks are
`mom_sommerfeld` (direct quadrature of the full layered Green's
function) and `fem` (volume PDE).

**Still open.** Reviving a genuine complex-image engine needs (a) the
complete layered Green's function of `solver/_layered.py` — both
$2h_1$ image families and the multiple-reflection denominator — and
(b) the now-honest fit above. Only then should `cim`/`bem` accept
$n \ge 3$ again, and only then does `Engine` need to expose
`n_images` / `n_samples` (the parameters were removed from
`solve_cim` / `solve_bem` in 0.15.0 because `Engine.solve` could
never reach them).
