# ADR-0002: Engine family for layered soils

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-05-01 |
| **Deciders** | Christian Ehlert |
| **Scope** | `groundfield`, work package 1 of the dissertation |

## Context

ADR-0001 settled the methodology for the **2-layer** case: closed-form
Tagg/Sunde image-charge series (`image_2layer`) plus a Galerkin MoM
cross-validation engine (`mom`). For work package 1 in isolation
that is enough. Two outstanding issues motivate this ADR:

1. **Three or more layers.** Although AP1 itself targets a 2-layer
   soil, several supporting cases in the dissertation involve a
   weathered or frozen surface layer above a representative AP1 stack
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

The selection heuristic is:

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
| closed-form image vs. `mom` / `bem` / `cim` | 5 % | resolution scheme differs |
| `mom_sommerfeld` vs. closed-form layered engines | 5 % | quadrature is the reference |
| `fem` vs. integral engines | 10 % | equivalent-hemisphere bias |

Layer-contrast sweeps must produce a monotonically increasing
cluster impedance for every engine — a basic physics consistency
check.

## Consequences

**Positive.**

- Three independent methodologies (closed-form images, integral
  equation, volume PDE) are now available side by side. A bug in any
  single one is detectable through cross-comparison.
- For 3+ layer soils there is a clear recommended engine (`cim`)
  with an independent reference (`mom_sommerfeld`) and a second-line
  cross-check (`bem`).
- `fem` adds a methodologically distinct line of defence at the cost
  of a simple equivalent-hemisphere reduction; the bias is
  documented and bounded.

**Negative / open.**

- `mom_sommerfeld` is slow (per-pair adaptive quadrature). Acceptable
  for the cross-check role; not intended for production sweeps.
- `cim` quality depends on the matrix-pencil fit; very hard contrasts
  may need more images. The fit RMS is exposed in
  `result.metadata["cim_rms"]`.
- `fem` covers single-cluster worlds via the equivalent-hemisphere
  reduction. Multi-cluster volume runs would need a real 3-D mesh
  generator and a stronger FEM kernel — out of scope for AP1.

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
- [ ] Reference-case library for AP1 (separate notebooks under
      `notebooks/ap1_*`).
- [ ] Optional: scikit-fem-backed 3-D `fem` (lifted from the
      equivalent-hemisphere reduction) for full multi-cluster
      worlds; deferred until AP1 demands it.
