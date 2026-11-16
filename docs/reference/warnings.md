# Warnings and errors

`groundfield` warns when a result is valid but limited, and raises when
it cannot produce a meaningful result. Every warning class below is a
subclass of `UserWarning` (except `DeprecationWarning`), so it can be
filtered by category:

```python
import warnings

import groundfield as gf

with warnings.catch_warnings():
    warnings.simplefilter("error", gf.UniformCurrentFieldWarning)   # make it fatal
    warnings.simplefilter("ignore", gf.ExperimentalBackendWarning)  # or silence it
    ...  # solve and post-process here
```

Warnings are attributed to the line of your code that caused them,
not to the inside of the package; warnings that the package's own
internal calls would trigger (for example a generator using an old
keyword) are not shown, because you cannot act on them. Python shows a
`DeprecationWarning` by default only for code in `__main__` and in
notebooks; run tests with `-W error::DeprecationWarning` to find
deprecated names early. The page lists every warning and exception
class defined in `src/groundfield`, then the plain `UserWarning`s, then
the built-in exceptions with a fixed meaning.

## Solver and backends

### `BackendCapabilityError`

`gf.BackendCapabilityError`, subclass of `ValueError` and
`NotImplementedError`. **Raised** by `Engine.solve` before any
numerics when the backend cannot represent the world: an unsupported
number of soil layers, electrodes with a concrete-shell coefficient,
junctions, a world outside the axisymmetric model of `fem` with
`FemOptions.model="axisymmetric"`, an explicitly requested kernel the
backend does not implement, or a horizontal wire so close to the layer interface, on
either side of it, that the point form cannot evaluate it (see
`PointKernelInterfaceWarning`). The message names the backends that can. **Remedy:** choose
one of them (see the [backend registry](api/registry.md)); three or more
distinct layers need `mom_sommerfeld`.

### `BackendCapabilityWarning`

`groundfield.solver.BackendCapabilityWarning`. **Emitted** when the
backend solves the world with a documented simplification: a backend
without two-terminal sources returns the current of a source with
`return_to` through remote earth (no built-in backend since 2026.11; up
to then `fem` also treated distributed and inductive conductors as lumped
resistive branches). **Remedy:** use an image or MoM backend when the
conductor network or the return terminal matters.

### `ExperimentalBackendWarning`

`gf.ExperimentalBackendWarning`. **Emitted** once per process on the
first solve with an experimental backend (`fem`). Its results are not
validated as a cross-check and may change between releases. **Remedy:**
use it for orientation only, or filter the category.

### `BackendOptionsWarning`

`groundfield.solver.BackendOptionsWarning`. **Emitted** when
`Engine.options` configures a backend that will not run (for example
`options={"fem": ...}` with `backend="mom"`); the options are ignored.
**Remedy:** remove them or select that backend.

### `ProbePointsIgnoredWarning`

`gf.ProbePointsIgnoredWarning` (also `groundfield.solver`). **Emitted** by
`solve_mutual_matrix` when it is given `probe_points` but computes the
galerkin off-diagonal, the default of every call since 2026.11. Up to
0.15.0 probe points selected the potential at those points as the
off-diagonal; a call written for 0.15.0 (the AP1 pipeline, for example)
keeps running and now returns the reciprocal galerkin matrix, which
reproduces the solve of the clusters joined by branches (the probe matrix
missed it by −0.15 … −0.88 % on an AP1-like site). **Remedy:** drop
`probe_points`, or pass `offdiag="probe"` to keep the 0.15.0 definition.

### `EngineFrequencyOrderWarning`

`groundfield.solver.engine.EngineFrequencyOrderWarning`. **Emitted**
when `Engine.frequencies` is not strictly increasing. The order is kept,
and every result list follows it. **Remedy:** sort the list, or state the
intent with `engine.with_frequencies(..., preserve_order=True)`.

### `ShallowSegmentWarning`

`groundfield.solver.ShallowSegmentWarning`. **Emitted** for the point
kernel only (`kernel="point"` and the `mom` family) when a leakage segment is
longer than about four times its depth: the self term uses the point
image at distance $2z$ instead of the image line, which overestimates
it by more than 13 % and biases the impedance high. The message
quantifies the worst case. **Remedy:** shorter segments, or the exact
line kernel of the image family (the default), which has no such bias.

### `PointKernelInterfaceWarning`

`groundfield.solver.PointKernelInterfaceWarning`. **Emitted** for the
point form (`kernel="point"` of the image family and of
`solve_mutual_*`, the `mom` family) when a horizontal wire lies within
0.6 segment lengths of the layer interface $h_1$, on either side of it.
With an electrode that reaches the interface (the cross-layer path) the
three-point rule of the diagonal samples the layered correction where it
varies quickly; with every wire above $h_1$ the diagonal takes the image
of a segment in the interface as a point source $2|z - h_1|$ away, while
its image line runs much closer to the wire. Measured: 3–17 % low on the
cross-layer path, −2.5 % and −0.4 % for a ring 0.2 m above $h_1$
(1000/30 and 30/1000 Ω·m, ds = 0.5 m). Closer than 0.387 segment
lengths, or where the rule would reach above the surface, the solve
raises `BackendCapabilityError` instead: a ring 1 µm above $h_1$ read
−1931 Ω where the line kernel gives 4.18 Ω (audit A3S-02).
**Remedy:** the exact line kernel of the image family (the default), or
segments shorter than the distance of the wire to the interface.

### `InterfaceContactWarning`

`groundfield.solver.InterfaceContactWarning`. **Emitted** by every
segment backend (the image family, `mom`, `mom_sommerfeld`), by
`solve_mutual_*` and by the three-dimensional `fem` (both rules, once per
rule and solve; rod ends since audit W4R2A-03) where the thin-wire
model, which carries the current on the wire axis, deviates from the
solid wire next to a layer interface:

- **Rod ends.** A rod head or tip within one wire radius of a layer
  interface. The thin-wire model has no end face: a rod that ends exactly
  at the interface is a wire entirely on one side of it, and one that
  reaches past it by a fraction of its radius touches the other layer
  through a short part only, while the solid rod with a flat end is
  continuous in between. Against an axisymmetric model of the solid rod
  (4.5 m, $a$ = 8 mm, 1000/30 Ω·m) the thin wire reads +21 % with its tip
  at the interface, +7 % 1 mm above it, −4.6 % 1 mm below it, and within
  a few per cent beyond two radii, as anywhere else. The three-dimensional
  `fem` is a thin wire without end faces too: a rod tip on 1000/30 Ω·m
  reads +43 % there against the solid rod, with an error estimate of
  6e-3 (audit W4R2A-03). An end that meets the
  end of another rod — a rod split at the interface into two pieces, the
  usual way to give each layer its own piece — has no free end face and is
  not reported.
- **Horizontal wires** (since 2026.11, audits W4A-07, W4B-02, W4R2A-04
  and W4R3A-02). Every
  electrode but a rod, and the leaking pieces of a galvanic distributed
  conductor, whose axis lies within the reach of a contrasting interface.
  Next to a more conductive layer the current crowds to the wire's near
  side, which a current on the axis cannot follow. Against the solid torus
  (the axisymmetric `fem`; a Hankel thin ring plus the exact 2-D
  proximity correction agrees to 5e-4) a ring of $a$ = 5 mm over a layer
  33 times more conductive reads +0.59 % at 4 radii, +1.2 % at 3, +3.6 % at
  2, +8.6 % at 1.5 and +32 % at 1.1 radii, and +89 % when the interface
  cuts the wire; under a more resistive layer up to +2.6 % (a 55 mm wire on
  a 1 m ring at 1.01 radii, contrast 1000). The reach is the distance of
  the wire axis from the interface inside which the thin wire (converged in
  its segment length) is 1 % or more off: the largest of five ring
  geometries ($r_0/a$ = 18 … 400: rings of 0.5 … 5 m at 0.3 … 2 m depth,
  $a$ = 5 … 55 mm; the more conductive layer below the wire, for the 2 m
  ring also above it; the more resistive layer below and above every one;
  a 55 mm wire on a 1 m ring is the widest at every contrast), rounded up
  to 0.1 $a$, plus 0.1 $a$ for the contrasts between the nodes — the
  resistive side from the converged thin wire, the conductive side from the
  thin wire at one segment length, its node at 1000 raised to 4.3 $a$ so
  that the converged distances lie inside:

  | contrast of the layers | 1.4 | 2 | 3 | 5.5 | 10 | 18 | 33 | 100 | 1000 |
  |---|---|---|---|---|---|---|---|---|---|
  | more conductive beyond | 1.0 $a$ | 1.3 $a$ | 1.9 $a$ | 2.7 $a$ | 3.3 $a$ | 3.7 $a$ | 3.9 $a$ | 4.1 $a$ | 4.3 $a$ |
  | more resistive beyond | 1.0 $a$ | (1.14 $a$) | 1.3 $a$ | (1.60 $a$) | 1.9 $a$ | (2.05 $a$) | 2.2 $a$ | 2.2 $a$ | 2.3 $a$ |

  (log-linear in between, values in brackets interpolated; held beyond
  1000, and at 1.0 $a$ down to a contrast of 1.3; 1.0 $a$: only a wire
  that the interface cuts.) Checked at the nodes and between them
  (conductive 1.3 … 1.9, 2.5, 4, 7, 14, 25, 50, 300; resistive 2, 5.5, 18,
  300): of 2348 conductive distances (the thin wire at one segment length)
  and 2470 resistive ones (converged in it) none with the thin wire 1 % or
  more off lies outside the reach, the largest errors outside it are
  +0.94 % and +0.93 %; converged, the thick wire's conductive 1 % distances
  lie 0.004 … 0.17 $a$ inside the reach (the node at 1000 is 4.3 $a$ for
  the 4.15 $a$ at 300). Below a contrast of 1.3 nothing is reported: a cut
  wire is at most +0.96 % off at 1.3 and +1.06 % at 1.35 (the thick wire
  cut 0.5 $a$ below its axis by a more resistive layer above it). Up to
  audit W4R2A-04 the table came from four geometries (0.2 … 0.4 $a$ short
  of the thick wire on the 1 m ring) and started at a contrast of 2, below
  which a cut wire read up to −1.46 %; up to audit W4R3A-02 its resistive
  side was measured above the wire for the thin ring only (the thick wire
  1 % off up to 0.74 $a$ beyond it: +1.75 % at 1.3 $a$ under a 33 times more
  resistive layer, unreported) and it started at 1.4 (+1.16 % there).
  The table is measured on rings with $r_0/a \ge 18$; a thicker wire on a
  smaller ring is not covered. A piece of a galvanic distributed conductor
  counts as horizontal while its slope $|e_z|$ is at most 0.1 (5.7°): a
  steeper piece passes the reach of an interface over at most
  $2\,\mathrm{reach}/|e_z|$ wire radii of its length, like a rod through an
  interface, and is not reported (a step at the cutoff, kept). The
  three-dimensional `fem`
  shares the limit (its smoothed line source lies between the thin wire
  and the torus); the axisymmetric `fem` solves the wire as the solid torus
  and does not warn. No generated network and no world of the AP1 soils
  ($h_1$ = 5 or 30 m) comes within reach.

`diagnose()` reports both under `interface_contact`; with an `fem` engine
it reports neither for a world that the axisymmetric model takes, which
resolves rod end faces and ring wires (audit W4R2A-05).
**Remedy:** keep rod ends and horizontal wires a few wire radii from a
contrasting interface; bracket a rod end by moving it by ±2 wire radii;
check a ring with `fem`'s axisymmetric model
(`FemOptions.model="axisymmetric"`).

### `SeriesTruncationWarning`

`groundfield.solver.image_2layer.SeriesTruncationWarning` (also in
`groundfield.solver.mom`). **Emitted** when the Tagg/Sunde image series
reaches `Engine.image_max_terms` before its tail bound
$\lvert K\rvert^{n+1}/(1-\lvert K\rvert)$ falls below
`image_series_tol`, typically at very high layer contrast
($\lvert K\rvert \to 1$). **Remedy:** raise `image_max_terms` (the
default 300 covers $\lvert K\rvert = 0.94$).

### `SommerfeldConvergenceWarning`

`groundfield.solver.mom_sommerfeld.SommerfeldConvergenceWarning`.
**Emitted** when the quadrature of the reflected part of the layered
Green's function does not converge to `epsrel`; the value may even have
the wrong sign. The message names the worst pair and the cause.
**Remedy:** raise `max_osc_panels` in
`Engine(options={"mom_sommerfeld": {...}})` for kilometre separations
with thin layers; at a layer contrast $\lvert K\rvert \to 1$ in two-layer
soil use `image_2layer`.

### `FemReductionWarning`

`gf.FemReductionWarning`. **Never emitted since 2026.11**: `fem` solves a
conforming finite-element model (user decision Q37) instead of the
equivalent-hemisphere reduction, and the class is kept importable for
filters of older code. Up to 2026.11 it was emitted once per `fem` solve,
listing the reasons, when the reduction did not represent the
world: several electrodes in a cluster (mutual resistance neglected;
−35 % for a ring with three rods, −55 … −75 % for two bonded rods of
which one reaches into a conductive top layer), several clusters (no
coupling between them), or a layer interface within the hemisphere
radius or the electrode's horizontal extent over a layer that the
electrodes touch or that conducts no better than theirs (−3 … +48 %
measured for a spanned layer, −17 … +16 % for one within the extent;
−44 … −78 % for a rod that reaches 1 µm … 8 cm into a conductive top
layer, −21 … +30 % for a ring lying exactly on the interface; −10 … +46 %
for a more conductive layer within the extent behind one that conducts no
better than the electrodes', audit R2A-09). A layer that no electrode
touches, that conducts better than the electrodes' layer and whose top
lies above the hemisphere radius or the horizontal half-extent (below the
hemisphere: with no such shielding layer in between) is no warning but a
`BackendCapabilityError` since 2026.11 (user decision Q23 and audit
D2A-03: `fem` misplaces the current, mostly far too low), as are
electrodes touching a layer the hemisphere does not reach, a strip deeper
than half its length, a ring deeper than its diameter and a driven
`MeshElectrode` (the closed forms outside their range). See the [fem engine page](../explanation/engines/fem.md#guards)
for the conforming model and its guards. **Remedy:** none needed; remove
filters for this category when convenient.

### `ComplexImageFitWarning`

`groundfield.solver.cim.ComplexImageFitWarning`. **Emitted** only by a
direct call of `groundfield.solver.cim.fit_complex_images` when the
complex-image fit misses its tolerance. The `cim` backend no longer fits
(it is a deprecated alias of `image`). **Remedy:** do not use the fit as
a Green's function.

## Coupling (earth return, layered Green's function)

### `SommerfeldResolutionWarning`

`groundfield.coupling.layered_green.SommerfeldResolutionWarning` (also
raised by `coupling.sommerfeld_inductance`). **Emitted** when the Hankel
quadrature cannot place enough nodes per oscillation of $J_0(\lambda s)$
within its node budget; the result would not be converged. **Remedy:**
lower `lambda_max_factor` (no accuracy is lost: the truncated tail is
added in closed form) or reduce the distance range of the group.

### `SommerfeldTailTruncationWarning`

`groundfield.coupling.layered_green.SommerfeldTailTruncationWarning`.
**Emitted** when the reflected kernel needs more closed-form image
terms than its budget to bound the truncated tail of the quadrature
(a very wide range of horizontal distances in one group, typically
pairs near the surface or the layer interface); the value is then not
bounded by the documented tolerance. **Remedy:** lower
`lambda_max_factor` (fewer image terms are then needed) or reduce the
horizontal distance range of the group.

## Model and input

### `DeprecationWarning`

**Emitted** for every deprecated name, keyword, JSON key or backend
alias. The message names the replacement and the earliest removal
(2027.5). **Remedy:** see [Deprecations](deprecations.md).

### `InertConfigurationWarning`

`groundfield.utils.InertConfigurationWarning`. **Emitted** when a
public setting is stored but read by no backend: a non-default
`BoundaryConditions` field or a soil `relative_permittivity` (also in
the generator soil specs). Every backend uses the quasi-static
conduction-current model with remote earth as reference. **Remedy:**
remove the setting.

### `CrossSectionUnitWarning`

`groundfield.conductors.CrossSectionUnitWarning`. **Emitted** for a
conductor `cross_section` above 10⁻² m², which almost always is a value
in mm² (`50` instead of `50e-6`) and would make the conductor a
near-ideal short. **Remedy:** give the cross section in m².

## Diagnostics

### `DiagnosticsWarning`

`gf.DiagnosticsWarning`. **Emitted** by `diagnose(...).warn()`: one
aggregated warning with all findings, one line per cause (mixed
coordinates, thin-wire ratio, resolution, problem size, close or
interpenetrating wires, rod ends at a layer interface, inert settings,
series truncation). The solve emits the discretisation
findings as individual warnings. **Remedy:** follow the advice in each
line; see [Diagnostics](api/diagnostics.md).

### `MixedCoordinatesWarning`

`groundfield.diagnostics.MixedCoordinatesWarning` (since 2026.11, user
decision D5). **Emitted** once per solve by the image family, `mom`,
`mom_sommerfeld` and `solve_mutual_*`, before anything is discretised,
when points of the world lie both within 10 km of the origin and beyond
1000 km from it (horizontal distance): electrodes at projected (UTM,
Gauss–Krüger) coordinates and a node or electrode near (0, 0), almost
always given in the wrong system by mistake. The world is solved as
given, its two groups some 33 000 km apart; the message names both groups
and the conductors between them. Measured with a 12 m × 7 m outline and a
3 m rod at UTM coordinates (audit W2B-04): a return rod given at local
(80, 15) m raises $Z_{in}$ by 0.43 %; a PEN cable to a cabinet at local coordinates is 33 118 km
long, $Z_{in}$ = 7494 + j25 396 Ω instead of 5.39 + j0.19 Ω at 50 Hz with
`carson_series`, and the default `sommerfeld` earth return does not
finish. `diagnose()` reports the same world under `coordinates` (a
warning), and a world wider than 100 km that does not mix as
information. A world wholly in projected or wholly in local coordinates
never warns. `fem` runs the same check at the solve (since 2026.11, user
decision D5). **Remedy:** give every point in one coordinate system; to stop
such a solve at once, `warnings.simplefilter("error",
MixedCoordinatesWarning)`.

### `ResourceLimitError`

`gf.ResourceLimitError`, subclass of `ValueError`. **Raised** before
any allocation when the predicted number of segments, or of unknowns of
the dense nodal system, exceeds the engine's limit `Engine.max_segments`
(default `groundfield.diagnostics.HARD_LIMIT`, 20 000), by every segment
backend and by `solve_mutual_matrix` / `solve_mutual_field`. The usual
cause is a geometry given in millimetres, a very small `segment_length`
or a conductor `discretize_segment_length` in millimetres. **Remedy:**
check units and segment lengths; to solve a large world on purpose,
raise the limit of the engine, `Engine(max_segments=...)`. Below that
limit, a world at or above the engine's notice size
(`Engine(warn_segments=...)`, default `SOFT_LIMIT` = 5 000) only emits the
`UserWarning` `"budget: ..."` that the solve may take minutes; the notice
comes from `Engine.solve` (with any backend), not from
`solve_mutual_matrix` / `solve_mutual_field`. Both sizes count the
segments of the integral engines; `fem` checks the display segments that
carry the point sources of its result against `Engine.max_segments` too,
and limits its mesh by `FemOptions.max_nodes` (axisymmetric, about 4 GB
per million nodes) and `max_nodes_3d` (3-D; the solve adds about 2.9 GB
per million nodes, 3.0 … 3.1 GB with the process), estimated before
anything is assembled, and the ports of its 3-D model by `max_ports_3d`
(100), checked before meshing.

## Post-processing

### `UniformCurrentFieldWarning`

`gf.UniformCurrentFieldWarning`. **Emitted** by the touch- and
step-voltage functions when a result of the image family (one uniform
current per straight wire of an electrode) is evaluated near a grid, mesh,
star, strip end,
open polyline or bonded multi-electrode cluster, where the real current
crowds to edges and tips; the field values there are impedance-grade,
not field-grade. **Remedy:** re-solve with `backend="mom"` for those
quantities.

### `VectorFitQualityWarning`

`gf.VectorFitQualityWarning`. **Emitted** when the relative rms error of
a vector fit exceeds `rms_tol` (1 %), or when the imaginary part of a
0 Hz sample had to be discarded. **Remedy:** inspect `rms_relative`; try
another `n_poles`, more samples or `weighting="inverse_magnitude"`.

### `VectorFitConvergenceWarning`

`groundfield.fit.VectorFitConvergenceWarning`. **Emitted** when the
poles still move by more than 1 % after `n_iter` relocation steps.
**Remedy:** more iterations, another `n_poles`, or a different
weighting.

### `VectorFitUnderdeterminedWarning`

`groundfield.fit.VectorFitUnderdeterminedWarning`. **Emitted** when
`n_poles` is not smaller than the number of samples: the fit only
interpolates. **Remedy:** fewer poles or more frequencies.

### `RhoFStandardQualityWarning`

`gf.RhoFStandardQualityWarning`. **Emitted** when the standard-form
fit misses a sample by more than `max_relative_tol` (5 %): the fixed
ansatz cannot represent the sampled $Z(\rho, f)$. **Remedy:** check
`max_relative`, restrict the $\rho$ range, or fit per $\rho$ with vector
fitting.

## IO and export

### `NonPassiveFitWarning` / `NonPassiveFitError`

`gf.NonPassiveFitWarning`, `gf.NonPassiveFitError` (subclass of
`ValueError`). **Emitted / raised** (with `require_passive=True`) when
the exported impedance formula has $\operatorname{Re}Z < 0$ somewhere
between 0 Hz and twice the highest fitted frequency; the message names
the frequency (and $\rho$) of the most negative value. **Remedy:**
refit (fewer poles, different weighting) before exporting.

### `EvaluateSpecError`

`gf.EvaluateSpecError`, subclass of `ValueError`. **Raised** by
`evaluate_spec`, `load_bustype_json` and `BusTypeSpec.from_dict` when a
`BusType` document or its formula is rejected: not a `BusTypeSpec`, empty
formula, unknown symbol, or a formula outside the allowed grammar or its
size bounds (the parser never evaluates the text). The parser's own
`groundfield.io._formula.FormulaError` arrives wrapped in this class.
**Remedy:** regenerate the document with `save_bustype_json`, or fix the
formula by hand.

### `ResultFormatError`

`gf.ResultFormatError`, subclass of `ValueError`. **Raised** by
`load_result` for a file that is not a readable result file: wrong
`format`, a newer `schema_version`, invalid JSON or UTF-8, or metadata
outside what the writer produces. **Remedy:** load it with the
groundfield version that wrote it (`written_by`), or regenerate it.

### `OverpassError`

`gf.OverpassError`, subclass of `RuntimeError`. **Raised** by the OSM
queries on a non-success Overpass answer or a malformed payload (after
the retries); `status_code` and the start of the body are attached. The
answer is not cached. **Remedy:** retry later, reduce the radius, or work
from a cached / stored answer.

## Other warnings (plain `UserWarning`)

| Where | When | What to do |
|---|---|---|
| `coupling` (Carson) | `earth_inductive_model="carson_series"` with layered soil: the upper-layer $\rho_1$ is used | use `"sommerfeld"`, which supports layered soil |
| `solver` (inductive assembly) | `earth_inductive_model="perfect_mirror"` with inductive conductor segments on both sides of the soil surface (an overhead lead and a buried PEN): its additive image has no physical basis for such a pair; once per solve, naming the number of pairs and the conductors (user decision Q27) | use `"sommerfeld"` (the default) for numbers; the result of `perfect_mirror` is a diagnostic baseline |
| `solver` (`image_2layer`, `kernel="point"`) | the deepest segment ends less than two segment lengths above the layer interface | refine `segment_length` near the interface, or use the default line kernel |
| `solver` (kernels) | two segments of different wires closer than the 1 mm singularity clamp | merge the wires or move them apart |
| `solver` (pipeline) | a source looks like the hand-made counter-source of another one | use one source with `return_to` |
| `solver` (pipeline) | conductors could not be matched to the branch network; their inductance is not applied | check the conductor end points |
| `World.set_boundary_conditions` | a boundary field set to (or back from) a non-default value | none needed; the fields are not read |
| `geometry` (`PolylineElectrode`) | `closed=True` with only two vertices (treated as open) | give at least three vertices |
| `generators` | buildings that cannot be connected to a PEN cable or a feeder, exhausted feeder slots, a cabinet without electrodes, a footprint widened to 1 m, `source_return_to` overriding the measurement return | check the layout parameters named in the message |
| `geo` | an unreadable Overpass cache file (re-fetched) | none |
| `fit` / `io` | a fit or exported formula that does not depend on frequency | expected for frequency-independent data; otherwise check the samples |

## Built-in exceptions with a fixed meaning

| Exception | Raised when |
|---|---|
| `KeyError` | an electrode, junction, conductor or source name is unknown (the message suggests the closest names) |
| `ValueError` | an invalid value: non-finite or non-positive physical input, a current source of 0 A, a short-circuited two-terminal source, a loop of voltage sources, a junction without soil contact, an empty world |
| `TypeError` | an unknown keyword of a factory, or a soil that is not a soil model |
| `NotImplementedError` | `FieldResult.potential` for three or more soil layers (`BackendCapabilityError`, a subclass, for a world a backend cannot represent, such as a strip with `FemOptions.model="axisymmetric"`) |
| `ImportError` | a function of `groundfield.geo` without the `geo` extra, or `to_bustype` without `groundinsight` (the message gives the install command) |
| `pydantic.ValidationError` | a model field fails validation (subclass of `ValueError`) |
