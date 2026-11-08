# Solver

The solver layer holds the numerical core. Available backends:

| Backend | Soil model | Method | Status |
|---|---|---|---|
| `image` | `HomogeneousSoil` | image-charge sum (point sources + line self-action) | implemented |
| `image_2layer` | `TwoLayerSoil` | Tagg/Sunde image-charge series | implemented |
| `image_nlayer` | `HomogeneousSoil`, `TwoLayerSoil`, `MultiLayerSoil` | image-charge dispatcher (delegates to `image` for `n=1`, to `image_2layer` for `n=2`; raises for `n ≥ 3`) | implemented |
| `cim` | any layered | Complex Image Method (matrix-pencil fit of $\Gamma_1(\lambda)$) | implemented |
| `mom` | `HomogeneousSoil` or `TwoLayerSoil` | Galerkin Method-of-Moments on segment level (independent resolution scheme over the same Green's-function kernels) | implemented |
| `mom_sommerfeld` | any layered | Galerkin MoM with direct Sommerfeld quadrature (reference engine) | implemented |
| `bem` | any layered | Boundary-element collocation with the CIM kernel | implemented |
| `fem` | any layered | Axisymmetric volume PDE with equivalent-hemisphere reduction | implemented |

## Mathematical / physical model

For a point current source $I$ at $z_s$ in the upper layer of a
horizontally stratified half-space (insulating soil surface at
$z = 0$), every backend evaluates the same quasi-static Sommerfeld
representation of the potential:

$$
\varphi(s, z) \;=\; \frac{\rho_1\, I}{4\pi}
\int_0^{\infty} \bigl[
  e^{-\lambda |z - z_s|}
+ \Gamma_1(\lambda)\, e^{-\lambda (z + z_s)}
\bigr]\, J_0(\lambda s)\, d\lambda,
$$

with the upward-looking reflection $\Gamma_1(\lambda)$ built
recursively from the bottom up
(`groundfield.solver._layered.reflection_gamma`). The engines differ
only in **how** they evaluate this integral:

- closed-form real images (`image`, `image_2layer`, `image_nlayer`);
- closed-form complex images (`cim`);
- direct numerical quadrature (`mom_sommerfeld`);
- volume PDE (`fem`).

ADR-0002 (`docs/adr/0002-engine-family.md`) records the selection
heuristic between the engines.

## Auto-dispatch

`Engine.solve` automatically forwards `backend="image"`:

- to `image_2layer` if `world.soil` is a `TwoLayerSoil`;
- to `image_nlayer` if `world.soil` is a `MultiLayerSoil`.

Notebooks therefore keep working unchanged when the soil model is
swapped.

The series controls `Engine.image_max_terms` (default 300) and
`Engine.image_series_tol` (default $10^{-6}$) are forwarded on **both**
dispatch paths. Up to 0.14.x `image_nlayer` was called without them and
fell back to its own signature defaults (200, $10^{-6}$), so raising
`image_max_terms` — the documented remedy for a high layer contrast,
$|K| \to 1$, where the geometric tail bound
$|K|^{n+1} / (1 - |K|)$ decays slowly — had no effect when the same
soil was spelled as a `MultiLayerSoil` instead of a `TwoLayerSoil`.
Since 0.15.0 both spellings return the identical impedance.

## `mom_sommerfeld` accuracy controls

Since 0.15.0 `Engine.solve` also forwards four fields to
[`mom_sommerfeld`](../engines/mom_sommerfeld.md). Up to 0.14.x that
backend was called as `solve_mom_sommerfeld(world, self)` with no
keywords, so `lambda_max_factor`, `epsabs` and `epsrel` were
unreachable from the public API — a user who hit a non-converged corner
in the package's *reference* engine had nothing to tighten.

| `Engine` field | backend keyword | default | meaning |
|---|---|---|---|
| `sommerfeld_lambda_max_factor` | `lambda_max_factor` | 200.0 | truncation of the reflected-remainder quadrature in units of $1/d_\text{rem}$; saturates at 60 |
| `sommerfeld_epsabs` | `epsabs` | $10^{-9}$ | absolute threshold of the grid-refinement convergence test |
| `sommerfeld_epsrel` | `epsrel` | $10^{-7}$ | relative threshold, measured against $\lvert G\rvert$ |
| `sommerfeld_max_osc_panels` | `max_osc_panels` | 200 000 | budget for the $\pi/s$ panel family resolving $J_0(\lambda s)$ |

The defaults equal the previous signature defaults, so existing callers
get bit-identical results. All four are echoed into
`FieldResult.metadata`. They are **not** exposed through
`gf.create_engine(...)` — construct the model directly:

```python
import groundfield as gf

engine = gf.Engine(
    backend="mom_sommerfeld",
    segment_length=0.5,
    sommerfeld_epsrel=1e-9,
    sommerfeld_max_osc_panels=2_000_000,
)
```

!!! danger "`lambda_max_factor` has two different units"
    On the top-layer path it is in units of $1/d_\text{rem}$ (the
    *remainder* decay length) and saturates at 60, so raising it is
    harmless. On the delegated $n = 2$ **cross-layer** path
    (`z > h_1` or `z_s > h_1`, forwarded to
    `coupling.layered_green.two_layer_real_space_kernel`) it is in
    units of $1/\ell_\text{char}$ (a geometric length), is not
    saturated, and *raising* it exhausts that kernel's Hankel node
    budget and triggers `SommerfeldResolutionWarning` — there the
    remedy is to **lower** it. See
    [`mom_sommerfeld`](../engines/mom_sommerfeld.md) for the full note.

### `SommerfeldConvergenceWarning`

`groundfield.solver.mom_sommerfeld.SommerfeldConvergenceWarning` is a
`UserWarning` subclass raised when the reflected-remainder quadrature
exhausts its refinement ladder without two successive grids agreeing
within `max(epsabs, epsrel · |G|)`. It reports the achieved versus the
requested tolerance, the offending $(s, \lvert z - z_s\rvert,
z + z_s)$, whether the oscillation panel budget bound and how much of
the required $\lambda$-range was covered, that the returned value's
**sign may be wrong**, and which parameter to raise. An $O(N^2)$
matrix assembly aggregates all failures into one warning naming the
worst pair and the number of affected entries.

Up to 0.15.0 the same condition was only `logger.debug`, and the
*least* accurate rung of the refinement ladder was the value returned;
that combination could return a **negative** Green's function
(−9.47e-05 where the far-field asymptote requires +5.00e-04) with zero
Python-level warnings. Two mechanisms reach the warning:

- **oscillation budget** — `sommerfeld_max_osc_panels` binds for
  $s/d_\text{rem} \gtrsim 1.9\cdot10^{4}$ (a centimetre-thin
  intermediate layer plus a kilometre-scale separation). Remedy: raise
  it; cost grows linearly.
- **layer contrast** — $\lvert K_1\rvert \to 1$. Remedy: none of the
  knobs; use `image_2layer`, whose geometric series is exact for
  $n = 2$.

Silence with
`warnings.simplefilter("ignore", SommerfeldConvergenceWarning)`, or
promote it with `simplefilter("error", ...)` in a validation harness.

Sibling categories in the same family:
`solver.image_2layer.SeriesTruncationWarning` (image series hit
`image_max_terms`) and
`coupling.layered_green.SommerfeldResolutionWarning` (Hankel grid
cannot resolve $J_0$).

## Example

```python
import groundfield as gf

# Build a small world (single ring electrode in two-layer soil).
soil  = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
world = gf.create_world(soil=soil)
gf.create_electrode(
    world, "ring", name="g1",
    center=(0.0, 0.0, 0.8), radius=5.0, wire_radius=0.005,
)
gf.create_source(world, attached_to="g1", magnitude=1.0)

# Create an engine and solve. Auto-dispatch hands `image` over to
# `image_2layer` because the soil is two-layer.
engine = gf.create_engine(
    backend="image",
    frequencies=[50.0, 150.0, 250.0],
    segment_length=0.05,
)
result = world.solve(engine)

# Cluster impedance of the ring electrode.
print(result.cluster_impedance("g1"))
```

The same `World` can be solved with any of the eight backends —
`compare_engines(world, engines={"image": ..., "mom": ...})` reports
their cluster-impedance agreement (cross-validation rules below).

### Frequency-list order

`Engine.frequencies` is **order-preserving**. The list is iterated
verbatim and the same order propagates to
`FieldResult.frequencies` and to every per-frequency column in the
post-processing helpers (`postprocess.sweep`, the CSV writers,
`fit_to_sympy`, etc.). A non-monotonic list — e.g. `[5000, 50]` —
is accepted but raises a `UserWarning` so the convention is
visible. Use the explicit opt-in if the order is intentional:

```python
engine = (
    gf.create_engine(backend="image")
    .with_frequencies(5000.0, 50.0, preserve_order=True)
)
```

`Engine.with_frequencies` returns a fresh `Engine` instance; the
receiver is not mutated.

## Diagnostics raised by the galvanic image family

### `ShallowSegmentWarning`

Exported from `groundfield.solver` (and from
`groundfield.solver.image`, where it is defined):

```python
import warnings
from groundfield.solver import ShallowSegmentWarning

warnings.simplefilter("ignore", ShallowSegmentWarning)
```

A `UserWarning` subclass emitted when a leakage segment is longer than
four times its burial depth, $L_i > 4 z_i$. The reaction-matrix
diagonal then uses the *point* image at distance $2 z_i$ where the
segment's *image line* would be needed, which overestimates that entry
by more than 13 % (unboundedly as $z \to 0$), so the grounding
impedance is biased high and keeps moving under mesh refinement. The
message quantifies the bias of the worst offender with the
horizontal-segment closed form.

What triggers it, honestly:

- **Not** only distributed conductors. Any geometry whose segments are
  long compared with the burial depth qualifies — a ring of radius
  25 m at $z = 0.8$ m discretised at `segment_length = 5 m` gives arc
  segments of 4.909 m and warns "by 26 %"; so do coarse meshes,
  counterpoises and surface-near tapes. Measured across the test suite it fires at
  20 test nodes in 9 modules (`test_carson_coupling`,
  `test_conductors_distributed`, `test_conductors_finite`,
  `test_convergence`, `test_diagnostics`, `test_earth_return_benchmark`,
  `test_inductive_coupling`, `test_sommerfeld_inductance`,
  `test_pass9_mom_sommerfeld`).
- **Only from `image` and `image_2layer`.** `mom`, `bem`, `cim`,
  `mom_sommerfeld` and `mutual` share the same biased diagonal through
  `solver.image._self_corrected_kernel`, but they do not emit the
  warning, so a `compare_engines` run warns asymmetrically about a
  bias all of them carry. The reasoning is recorded in
  `_check_segment_depths` ("Emission scope"): the shared kernel is
  re-entered per frequency and per excitation, so emitting there would
  repeat the message with a `stacklevel` pointing into solver
  internals.

The **hard** depth guards are not asymmetric: a segment midpoint at
$z \le 0$ or with $2|z| \le a$ raises `ValueError` from the shared
kernel, so all backends of the family reject it identically (0.15.0;
before that only `image` / `image_2layer` rejected $z \le 0$, and
`mom` / `bem` / `cim` / `mom_sommerfeld` returned a number for an
airborne electrode). A segment that *straddles* $z = 0$ — midpoint
buried, upper end in the air — needs the segment direction and is
therefore rejected in `image` / `image_2layer` only.

## Cross-engine validation

`groundfield.compare_engines(world, engines={...})` runs the same
world through several engines and reports the cluster-impedance
agreement. The same rule is enforced by
`tests/test_cross_engines_extended.py`:

The reported deviation is the worst **pairwise** relative difference of
the cluster impedances, `(max Z - min Z) / min |Z|` — see
[Validation](validation.md) for the definition and for the change of
metric in 0.15.0.

- For homogeneous worlds every engine must agree to within 5 %
  (10 % for `fem`).
- For 2-layer worlds the closed-form / image / MoM engines must
  agree with each other to within 5 %.
- A monotone $\rho_2$ sweep at fixed $\rho_1$ must produce a
  monotonically increasing cluster impedance for every engine.

::: groundfield.solver
