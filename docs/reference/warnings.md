# Warnings and errors

This page explains every warning and error that `groundfield` emits: when
it appears, what it means for your result, and what to do. Look a message
up here when a solve or a post-processing call warns; the how-to guide
[Read and act on warnings](../how-to/warnings-and-diagnostics.md) shows the
workflow, and [Known limitations](known-limitations.md) gives the size of
each limit behind a warning.

`groundfield` warns when a result is valid but limited, and raises when it
cannot produce a meaningful result. Every warning class below is a
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

Warnings are attributed to the line of your code that caused them, not to
the inside of the package; warnings that the package's own internal calls
would trigger (for example a generator using an old keyword) are not
shown, because you cannot act on them. Python shows a `DeprecationWarning`
by default only for code in `__main__` and in notebooks; run tests with
`-W error::DeprecationWarning` to find deprecated names early. The page
lists every warning and exception class defined in `src/groundfield`,
then the plain `UserWarning`s, then the built-in exceptions with a fixed
meaning.

| Class | Kind | Typical cause |
|---|---|---|
| [`BackendCapabilityError`](#backendcapabilityerror) | error | the engine cannot represent the world |
| [`BackendCapabilityWarning`](#backendcapabilitywarning) | warning | a documented simplification of a custom backend |
| [`ExperimentalBackendWarning`](#experimentalbackendwarning) | warning | first solve with `fem` |
| [`BackendOptionsWarning`](#backendoptionswarning) | warning | options for a backend that does not run |
| [`ProbePointsIgnoredWarning`](#probepointsignoredwarning) | warning | `probe_points` of 0.15.0 code |
| [`EngineFrequencyOrderWarning`](#enginefrequencyorderwarning) | warning | frequencies not increasing |
| [`ShallowSegmentWarning`](#shallowsegmentwarning) | warning | point kernel, segment long against its depth |
| [`PointKernelInterfaceWarning`](#pointkernelinterfacewarning) | warning | point kernel, wire next to the layer interface |
| [`InterfaceContactWarning`](#interfacecontactwarning) | warning | rod end or wire at a layer interface |
| [`SeriesTruncationWarning`](#seriestruncationwarning) | warning | image series cut at its cap |
| [`SommerfeldConvergenceWarning`](#sommerfeldconvergencewarning) | warning | layered quadrature not converged |
| [`FemReductionWarning`](#femreductionwarning) | warning | never emitted since 2026.11 |
| [`ComplexImageFitWarning`](#compleximagefitwarning) | warning | direct call of the old complex-image fit |
| [`SommerfeldResolutionWarning`](#sommerfeldresolutionwarning) | warning | Hankel quadrature out of nodes |
| [`SommerfeldTailTruncationWarning`](#sommerfeldtailtruncationwarning) | warning | tail of the reflected kernel not bounded |
| [`DeprecationWarning`](#deprecationwarning) | warning | deprecated name, keyword or alias |
| [`InertConfigurationWarning`](#inertconfigurationwarning) | warning | a setting no backend reads |
| [`CrossSectionUnitWarning`](#crosssectionunitwarning) | warning | cross section given in mm² |
| [`DiagnosticsWarning`](#diagnosticswarning) | warning | the findings of `diagnose(...).warn()` |
| [`MixedCoordinatesWarning`](#mixedcoordinateswarning) | warning | projected and local coordinates mixed |
| [`ResourceLimitError`](#resourcelimiterror) | error | too many segments, nodes or ports |
| [`UniformCurrentFieldWarning`](#uniformcurrentfieldwarning) | warning | image-family field values near grids and stars |
| [`VectorFitQualityWarning`](#vectorfitqualitywarning) | warning | vector fit misses its rms tolerance |
| [`VectorFitConvergenceWarning`](#vectorfitconvergencewarning) | warning | poles still moving |
| [`VectorFitUnderdeterminedWarning`](#vectorfitunderdeterminedwarning) | warning | more poles than samples |
| [`RhoFStandardQualityWarning`](#rhofstandardqualitywarning) | warning | standard form misses a sample |
| [`NonPassiveFitWarning` / `NonPassiveFitError`](#nonpassivefitwarning-nonpassivefiterror) | warning / error | exported formula not passive |
| [`RhoPresetMismatchWarning`](#rhopresetmismatchwarning) | warning | BusType fitted over another preset |
| [`EvaluateSpecError`](#evaluatespecerror) | error | BusType document or formula rejected |
| [`ResultFormatError`](#resultformaterror) | error | not a readable result file |
| [`OverpassError`](#overpasserror) | error | failed Overpass query |

## Solver and backends

### `BackendCapabilityError`

`gf.BackendCapabilityError`, subclass of `ValueError` and
`NotImplementedError`.

**When:** raised by `Engine.solve` before any numerics when the backend
cannot represent the world: a number of soil layers it does not support
(three or more distinct layers for every engine but `mom_sommerfeld` and
`fem`), an explicitly requested kernel the backend does not implement
(`kernel="line"` for the `mom` family), a world outside the axisymmetric
model of `fem` with `FemOptions.model="axisymmetric"`, a world the 3-D
model of `fem` refuses before meshing (see the
[fem engine page](../explanation/engines/fem.md#guards)), or a horizontal
wire so close to the layer interface, on either side of it, that the point
form cannot evaluate it (see `PointKernelInterfaceWarning`).
`solve_mutual_matrix` and `solve_mutual_field` raise it for a backend
outside the image family and for three or more layers. A backend
registered by the user that declares no support for concrete shells,
junctions or voltage sources raises it for such worlds; every built-in
backend supports them.

**What it means:** nothing was computed. The message names the backends
that can solve the world.

**What to do:** choose one of them (see the
[backend registry](api/registry.md)); three or more distinct layers need
`mom_sommerfeld`.

### `BackendCapabilityWarning`

`groundfield.solver.BackendCapabilityWarning`.

**When:** emitted when the backend solves the world with a documented
simplification: a backend without two-terminal sources returns the current
of a source with `return_to` through remote earth.

**What it means:** the conductor network or the return terminal is
simplified. No built-in backend emits it since 2026.11.

**What to do:** use an image or MoM backend when the conductor network or
the return terminal matters.

??? note "Audit trail"

    Up to 2026.11 `fem` also treated distributed and inductive conductors as
    lumped resistive branches, with this warning.

### `ExperimentalBackendWarning`

`gf.ExperimentalBackendWarning`.

**When:** emitted once per process on the first solve with an experimental
backend (`fem`).

**What it means:** `fem` is a reference backend whose model, options and
results may still change between releases; its measured accuracy is on the
[fem engine page](../explanation/engines/fem.md).

**What to do:** nothing; filter the category once you have read the
limits of `fem`.

### `BackendOptionsWarning`

`groundfield.solver.BackendOptionsWarning`.

**When:** emitted when `Engine.options` configures a backend that will not
run (for example `options={"fem": ...}` with `backend="mom"`).

**What it means:** the options are ignored.

**What to do:** remove them or select that backend.

### `ProbePointsIgnoredWarning`

`gf.ProbePointsIgnoredWarning` (also `groundfield.solver`).

**When:** emitted by `solve_mutual_matrix` when it is given `probe_points`
but computes the galerkin off-diagonal, the default of every call since
2026.11.

**What it means:** a call written for 0.15.0 (the AP1 pipeline, for
example) keeps running and now returns the reciprocal galerkin matrix,
which reproduces the solve of the clusters joined by branches; the probe
matrix missed it by −0.15 … −0.88 % on an AP1-like site.

**What to do:** drop `probe_points`, or pass `offdiag="probe"` to keep the
0.15.0 definition.

??? note "Audit trail"

    Up to 0.15.0 probe points selected the potential at those points as the
    off-diagonal.

### `EngineFrequencyOrderWarning`

`groundfield.solver.engine.EngineFrequencyOrderWarning`.

**When:** emitted when `Engine.frequencies` is not strictly increasing.

**What it means:** the order is kept, and every result list follows it.

**What to do:** sort the list, or state the intent with
`engine.with_frequencies(..., preserve_order=True)`.

### `ShallowSegmentWarning`

`groundfield.solver.ShallowSegmentWarning`.

**When:** emitted for the point kernel only (`kernel="point"` and the `mom`
family) when a leakage segment is longer than about four times its depth.

**What it means:** the self term uses the point image at distance $2z$
instead of the image line, which overestimates it by more than 13 % and
biases the impedance high. The message quantifies the worst case.

**What to do:** shorter segments, or the exact line kernel of the image
family (the default), which has no such bias.

### `PointKernelInterfaceWarning`

`groundfield.solver.PointKernelInterfaceWarning`.

**When:** emitted for the point form (`kernel="point"` of the image family
and of `solve_mutual_*`, the `mom` family) when a horizontal wire lies
within 0.6 segment lengths of the layer interface $h_1$, on either side of
it. Closer than 0.387 segment lengths, or where the rule would reach above
the surface, the solve raises `BackendCapabilityError` instead.

**What it means:** the impedance is biased low. With an electrode that
reaches the interface (the cross-layer path) the three-point rule of the
diagonal samples the layered correction where it varies quickly; with
every wire above $h_1$ the diagonal takes the image of a segment in the
interface as a point source $2\lvert z - h_1\rvert$ away, while its image
line runs much closer to the wire. Measured: 3–17 % low on the
cross-layer path, −2.5 % and −0.4 % for a ring 0.2 m above $h_1$
(1000/30 and 30/1000 Ω·m, ds = 0.5 m).

**What to do:** the exact line kernel of the image family (the default),
or segments shorter than the distance of the wire to the interface.

??? note "Audit trail"

    The refusal below 0.387 segment lengths follows audit A3S-02: a ring
    1 µm above $h_1$ read −1931 Ω where the line kernel gives 4.18 Ω.

### `InterfaceContactWarning`

`groundfield.solver.InterfaceContactWarning`.

**When:** emitted by every segment backend (the image family, `mom`,
`mom_sommerfeld`), by `solve_mutual_*` and by the three-dimensional `fem`
(both rules, once per rule and solve) where the thin-wire model, which
carries the current on the wire axis, deviates from the solid wire next to
a layer interface:

- **Rod ends:** a rod head or tip within one wire radius of a layer
  interface. An end that meets the end of another rod — a rod split at the
  interface into two pieces, the usual way to give each layer its own
  piece — has no free end face and is not reported.
- **Horizontal wires:** every electrode but a rod, and the leaking pieces
  of a galvanic distributed conductor, whose axis lies within the reach of
  a contrasting interface (table below). A piece of a galvanic distributed
  conductor counts as horizontal while its slope $\lvert e_z\rvert$ is at
  most 0.1 (5.7°).

**What it means:** the result carries the thin-wire error next to the
interface. The thin-wire model has no end face: a rod that ends exactly at
the interface is a wire entirely on one side of it, and one that reaches
past it by a fraction of its radius touches the other layer through a
short part only, while the solid rod with a flat end is continuous in
between. Against an axisymmetric model of the solid rod (4.5 m,
$a$ = 8 mm, 1000/30 Ω·m) the thin wire reads +21 % with its tip at the
interface, +7 % 1 mm above it, −4.6 % 1 mm below it, and within a few per
cent beyond two radii, as anywhere else. Next to a more conductive layer
the current of a horizontal wire crowds to its near side, which a current
on the axis cannot follow: against the solid torus a ring of $a$ = 5 mm
over a layer 33 times more conductive reads +0.59 % at 4 radii, +1.2 % at
3, +3.6 % at 2, +8.6 % at 1.5 and +32 % at 1.1 radii, and +89 % when the
interface cuts the wire; under a more resistive layer up to +2.6 %.

Reach of the interface, wire axis to interface plane, in wire radii $a$:

| contrast of the layers | more conductive beyond | more resistive beyond |
|---|---|---|
| 1.4 | 1.0 $a$ | 1.0 $a$ |
| 2 | 1.3 $a$ | (1.14 $a$) |
| 3 | 1.9 $a$ | 1.3 $a$ |
| 5.5 | 2.7 $a$ | (1.60 $a$) |
| 10 | 3.3 $a$ | 1.9 $a$ |
| 18 | 3.7 $a$ | (2.05 $a$) |
| 33 | 3.9 $a$ | 2.2 $a$ |
| 100 | 4.1 $a$ | 2.2 $a$ |
| 1000 | 4.3 $a$ | 2.3 $a$ |

Log-linear in between, values in brackets interpolated; held beyond 1000,
and at 1.0 $a$ down to a contrast of 1.3; 1.0 $a$: only a wire that the
interface cuts. Below a contrast of 1.3 nothing is reported. The table is
measured on rings with $r_0/a \ge 18$; a thicker wire on a smaller ring is
not covered. No generated network and no world of the AP1 soils
($h_1$ = 5 or 30 m) comes within reach.

`diagnose()` reports both under `interface_contact`; with an `fem` engine
it reports neither for a world that the axisymmetric model takes, which
resolves rod end faces and ring wires.

**What to do:** keep rod ends and horizontal wires a few wire radii from a
contrasting interface; bracket a rod end by moving it by ±2 wire radii;
check a ring with `fem`'s axisymmetric model
(`FemOptions.model="axisymmetric"`), which solves the wire as the solid
torus and does not warn.

??? note "How the reach was measured, and audit trail"

    The three-dimensional `fem` is a thin wire without end faces too: a rod
    tip on 1000/30 Ω·m reads +43 % there against the solid rod, with an
    error estimate of 6e-3 (audit W4R2A-03, since which it warns for rod
    ends). It shares the horizontal-wire limit as well (its smoothed line
    source lies between the thin wire and the torus). The horizontal-wire
    rule dates from 2026.11 (audits W4A-07, W4B-02, W4R2A-04 and W4R3A-02).

    The solid torus is the axisymmetric `fem`; a Hankel thin ring plus the
    exact 2-D proximity correction agrees with it to 5e-4. The +2.6 % under
    a more resistive layer is a 55 mm wire on a 1 m ring at 1.01 radii,
    contrast 1000.

    The reach is the distance of the wire axis from the interface inside
    which the thin wire (converged in its segment length) is 1 % or more
    off: the largest of five ring geometries ($r_0/a$ = 18 … 400: rings of
    0.5 … 5 m at 0.3 … 2 m depth, $a$ = 5 … 55 mm; the more conductive layer
    below the wire, for the 2 m ring also above it; the more resistive layer
    below and above every one; a 55 mm wire on a 1 m ring is the widest at
    every contrast), rounded up to 0.1 $a$, plus 0.1 $a$ for the contrasts
    between the nodes — the resistive side from the converged thin wire, the
    conductive side from the thin wire at one segment length, its node at
    1000 raised to 4.3 $a$ so that the converged distances lie inside.

    Checked at the nodes and between them (conductive 1.3 … 1.9, 2.5, 4, 7,
    14, 25, 50, 300; resistive 2, 5.5, 18, 300): of 2348 conductive
    distances (the thin wire at one segment length) and 2470 resistive ones
    (converged in it) none with the thin wire 1 % or more off lies outside
    the reach, the largest errors outside it are +0.94 % and +0.93 %;
    converged, the thick wire's conductive 1 % distances lie 0.004 … 0.17 $a$
    inside the reach (the node at 1000 is 4.3 $a$ for the 4.15 $a$ at 300).
    Below a contrast of 1.3 a cut wire is at most +0.96 % off at 1.3 and
    +1.06 % at 1.35 (the thick wire cut 0.5 $a$ below its axis by a more
    resistive layer above it).

    Up to audit W4R2A-04 the table came from four geometries (0.2 … 0.4 $a$
    short of the thick wire on the 1 m ring) and started at a contrast of 2,
    below which a cut wire read up to −1.46 %; up to audit W4R3A-02 its
    resistive side was measured above the wire for the thin ring only (the
    thick wire 1 % off up to 0.74 $a$ beyond it: +1.75 % at 1.3 $a$ under a
    33 times more resistive layer, unreported) and it started at 1.4
    (+1.16 % there).

    A steeper piece of a galvanic distributed conductor passes the reach of
    an interface over at most $2\,\mathrm{reach}/\lvert e_z\rvert$ wire radii
    of its length, like a rod through an interface, and is not reported (a
    step at the cutoff, kept). That `diagnose()` reports nothing for a world
    of the axisymmetric `fem` is audit W4R2A-05.

### `SeriesTruncationWarning`

`groundfield.solver.image_2layer.SeriesTruncationWarning` (also in
`groundfield.solver.mom` and `groundfield.solver.mom_sommerfeld`).

**When:** emitted when the Tagg/Sunde image series reaches
`Engine.image_max_terms` before its tail bound
$\lvert K\rvert^{n+1}/(1-\lvert K\rvert)$ falls below `image_series_tol`,
typically at very high layer contrast ($\lvert K\rvert \to 1$).

**What it means:** the image series is cut before its tolerance; the
result is not converged in the series.

**What to do:** raise `image_max_terms` (the default 300 covers
$\lvert K\rvert = 0.94$).

### `SommerfeldConvergenceWarning`

`groundfield.solver.mom_sommerfeld.SommerfeldConvergenceWarning`.

**When:** emitted when the quadrature of the reflected part of the layered
Green's function does not converge to `epsrel`. The message names the
worst pair and the cause.

**What it means:** the value may even have the wrong sign.

**What to do:** raise `max_osc_panels` in
`Engine(options={"mom_sommerfeld": {...}})` for kilometre separations with
thin layers; at a layer contrast $\lvert K\rvert \to 1$ in two-layer soil
use `image_2layer`.

### `FemReductionWarning`

`gf.FemReductionWarning`.

**When:** never emitted since 2026.11: `fem` solves a conforming
finite-element model (user decision Q37) instead of the
equivalent-hemisphere reduction.

**What it means:** the class is kept importable for filters of older code.

**What to do:** none needed; remove filters for this category when
convenient. The conforming model and its guards are on the
[fem engine page](../explanation/engines/fem.md#guards).

??? note "Audit trail: the reduction up to 2026.11"

    Up to 2026.11 it was emitted once per `fem` solve, listing the reasons,
    when the reduction did not represent the world: several electrodes in a
    cluster (mutual resistance neglected; −35 % for a ring with three rods,
    −55 … −75 % for two bonded rods of which one reaches into a conductive
    top layer), several clusters (no coupling between them), or a layer
    interface within the hemisphere radius or the electrode's horizontal
    extent over a layer that the electrodes touch or that conducts no better
    than theirs (−3 … +48 % measured for a spanned layer, −17 … +16 % for
    one within the extent; −44 … −78 % for a rod that reaches 1 µm … 8 cm
    into a conductive top layer, −21 … +30 % for a ring lying exactly on the
    interface; −10 … +46 % for a more conductive layer within the extent
    behind one that conducts no better than the electrodes', audit R2A-09).
    A layer that no electrode touches, that conducts better than the
    electrodes' layer and whose top lies above the hemisphere radius or the
    horizontal half-extent (below the hemisphere: with no such shielding
    layer in between) was no warning but a `BackendCapabilityError` (user
    decision Q23 and audit D2A-03: `fem` misplaced the current, mostly far
    too low), as were electrodes touching a layer the hemisphere does not
    reach, a strip deeper than half its length, a ring deeper than its
    diameter and a driven `MeshElectrode` (the closed forms outside their
    range).

### `ComplexImageFitWarning`

`groundfield.solver.cim.ComplexImageFitWarning`.

**When:** emitted only by a direct call of
`groundfield.solver.cim.fit_complex_images` when the complex-image fit
misses its tolerance. The `cim` backend no longer fits (it is a deprecated
alias of `image`).

**What it means:** the fitted images do not reproduce the layered kernel to
the requested tolerance.

**What to do:** do not use the fit as a Green's function.

## Coupling (earth return, layered Green's function)

### `SommerfeldResolutionWarning`

`groundfield.coupling.layered_green.SommerfeldResolutionWarning` (also
raised by `coupling.sommerfeld_inductance`).

**When:** emitted when the Hankel quadrature cannot place enough nodes per
oscillation of $J_0(\lambda s)$ within its node budget — for example a
conductor longer than the spectral range of the earth-return quadrature
(see [Known limitations](known-limitations.md#earth-return-and-conductors)).

**What it means:** the result would not be converged.

**What to do:** lower `lambda_max_factor` (no accuracy is lost: the
truncated tail is added in closed form) or reduce the distance range of
the group.

### `SommerfeldTailTruncationWarning`

`groundfield.coupling.layered_green.SommerfeldTailTruncationWarning`.

**When:** emitted when the reflected kernel needs more closed-form image
terms than its budget to bound the truncated tail of the quadrature (a
very wide range of horizontal distances in one group, typically pairs near
the surface or the layer interface).

**What it means:** the value is then not bounded by the documented
tolerance.

**What to do:** lower `lambda_max_factor` (fewer image terms are then
needed) or reduce the horizontal distance range of the group.

## Model and input

### `DeprecationWarning`

**When:** emitted for every deprecated name, keyword, JSON key or backend
alias.

**What it means:** the name still works; the message names the
replacement and the earliest removal (2027.5).

**What to do:** see [Deprecations](deprecations.md).

### `InertConfigurationWarning`

`groundfield.utils.InertConfigurationWarning`.

**When:** emitted when a public setting is stored but read by no backend:
a non-default `BoundaryConditions` field or a soil `relative_permittivity`
(also in the generator soil specs).

**What it means:** the setting has no effect. Every backend uses the
quasi-static conduction-current model with remote earth as reference.

**What to do:** remove the setting.

### `CrossSectionUnitWarning`

`groundfield.conductors.CrossSectionUnitWarning`.

**When:** emitted for a conductor `cross_section` above 10⁻² m².

**What it means:** almost always a value in mm² (`50` instead of `50e-6`),
which would make the conductor a near-ideal short.

**What to do:** give the cross section in m².

## Diagnostics

### `DiagnosticsWarning`

`gf.DiagnosticsWarning`.

**When:** emitted by `diagnose(...).warn()`: one aggregated warning with
all findings, one line per cause (mixed coordinates, thin-wire ratio,
resolution, problem size, close or interpenetrating wires, overlapping
inductive conductors, rod ends and horizontal wires next to a layer
interface, inert settings, series truncation).

**What it means:** the solve emits the discretisation findings as
individual warnings; this class collects them before a solve.

**What to do:** follow the advice in each line; see
[Diagnostics](api/diagnostics.md).

### `MixedCoordinatesWarning`

`groundfield.diagnostics.MixedCoordinatesWarning`.

**When:** emitted once per solve by the image family, `mom`,
`mom_sommerfeld`, `fem` and `solve_mutual_*`, before anything is
discretised, when points of the world lie both within 10 km of the origin
and beyond 1000 km from it (horizontal distance): electrodes at projected
(UTM, Gauss–Krüger) coordinates and a node or electrode near (0, 0),
almost always given in the wrong system by mistake.

**What it means:** the world is solved as given, its two groups some
33 000 km apart; the message names both groups and the conductors between
them. Measured with a 12 m × 7 m outline and a 3 m rod at UTM coordinates:
a return rod given at local (80, 15) m raises $Z_{in}$ by 0.43 %; a PEN
cable to a cabinet at local coordinates is 33 118 km long,
$Z_{in}$ = 7494 + j25 396 Ω instead of 5.39 + j0.19 Ω at 50 Hz with
`carson_series`, and the default `sommerfeld` earth return does not
finish.

**What to do:** give every point in one coordinate system; to stop such a
solve at once, `warnings.simplefilter("error", MixedCoordinatesWarning)`.
`diagnose()` reports the same world under `coordinates` (a warning), and a
world wider than 100 km that does not mix as information. A world wholly
in projected or wholly in local coordinates never warns.

??? note "Audit trail"

    The warning and the check of `fem` at the solve date from 2026.11 (user
    decision D5); the measurements are those of audit W2B-04.

### `ResourceLimitError`

`gf.ResourceLimitError`, subclass of `ValueError`.

**When:** raised before any allocation when the predicted number of
segments, or of unknowns of the dense nodal system, exceeds the engine's
limit `Engine.max_segments` (default `groundfield.diagnostics.HARD_LIMIT`,
20 000), by every segment backend and by `solve_mutual_matrix` /
`solve_mutual_field`. `fem` checks the display segments that carry the
point sources of its result against `Engine.max_segments` too, and limits
its mesh by `FemOptions.max_nodes` (axisymmetric, about 4 GB per million
nodes) and `max_nodes_3d` (3-D; the solve adds about 2.9 GB per million
nodes, 3.0 … 3.1 GB with the process), estimated before anything is
assembled, and the ports of its 3-D model by `max_ports_3d` (100), checked
before meshing.

**What it means:** nothing was allocated. The usual cause is a geometry
given in millimetres, a very small `segment_length` or a conductor
`discretize_segment_length` in millimetres.

**What to do:** check units and segment lengths; to solve a large world on
purpose, raise the limit of the engine, `Engine(max_segments=...)` (or the
`FemOptions` limit). Below that limit, a world at or above the engine's
notice size (`Engine(warn_segments=...)`, default `SOFT_LIMIT` = 5 000)
only emits the `UserWarning` `"budget: ..."` that the solve may take
minutes; the notice comes from `Engine.solve` (with any backend), not from
`solve_mutual_matrix` / `solve_mutual_field`. Both sizes count the
segments of the integral engines.

## Post-processing

### `UniformCurrentFieldWarning`

`gf.UniformCurrentFieldWarning`.

**When:** emitted once per call by `touch_voltage`, `step_voltage`,
`touch_voltage_envelope`, `touch_voltage_at_part` and
`touch_voltage_spectrum` when a result of the image family (one uniform
current per straight wire of an electrode) is evaluated within 5 m
(`NEAR_FIELD_DISTANCE_M`) of a grid, mesh, star, strip, open polyline or
bonded multi-electrode cluster. Without the world the point sources of the
result stand in for the wires.

**What it means:** the real current crowds to edges and tips, so the field
values there are impedance-grade, not field-grade (against `mom`: a 20 m
grid corner +6 % in the touch voltage 1 m outside and −8 % in the step from
1 m to 2 m outside; see
[Known limitations](known-limitations.md#current-ansatz-and-source-kernels)).

**What to do:** re-solve with `backend="mom"` for those quantities.

### `VectorFitQualityWarning`

`gf.VectorFitQualityWarning`.

**When:** emitted when the relative rms error of a vector fit exceeds
`rms_tol` (1 %), or when the imaginary part of a 0 Hz sample had to be
discarded.

**What it means:** the fitted model misses the samples by more than the
tolerance.

**What to do:** inspect `rms_relative`; try another `n_poles`, more
samples or `weighting="inverse_magnitude"`.

### `VectorFitConvergenceWarning`

`groundfield.fit.VectorFitConvergenceWarning`.

**When:** emitted when the poles still move by more than 1 % after
`n_iter` relocation steps.

**What it means:** the poles are not converged.

**What to do:** more iterations, another `n_poles`, or a different
weighting.

### `VectorFitUnderdeterminedWarning`

`groundfield.fit.VectorFitUnderdeterminedWarning`.

**When:** emitted when `n_poles` is not smaller than the number of
samples.

**What it means:** the fit only interpolates.

**What to do:** fewer poles or more frequencies.

### `RhoFStandardQualityWarning`

`gf.RhoFStandardQualityWarning`.

**When:** emitted when the standard-form fit misses a sample by more than
`max_relative_tol` (5 %).

**What it means:** the fixed ansatz cannot represent the sampled
$Z(\rho, f)$.

**What to do:** check `max_relative`, restrict the $\rho$ range, or fit
per $\rho$ with vector fitting.

## IO and export

### `NonPassiveFitWarning` / `NonPassiveFitError`

`gf.NonPassiveFitWarning`, `gf.NonPassiveFitError` (subclass of
`ValueError`).

**When:** emitted (or raised, with `require_passive=True`) when the
exported impedance formula has $\operatorname{Re}Z < 0$ somewhere between
0 Hz and twice the highest fitted frequency.

**What it means:** the formula is not passive there; the message names the
frequency (and $\rho$) of the most negative value.

**What to do:** refit (fewer poles, different weighting) before exporting.

### `RhoPresetMismatchWarning`

`gf.RhoPresetMismatchWarning` (also `groundfield.io`).

**When:** emitted by `evaluate_spec` and `bustype_from_spec` when the
caller names the preset of its bus (`rho_preset=...`) and the document
records a different one in `metadata["rho_preset"]`.

**What it means:** the `rho` of the bus then means another resistivity
than the `rho` of the formula (see the effective resistivity in
[Known limitations](known-limitations.md#soil-layers)).

**What to do:** evaluate the BusType with the preset it was fitted over,
or refit it over the preset of your bus.

??? note "Audit trail"

    The preset record is user decision Q39.

### `EvaluateSpecError`

`gf.EvaluateSpecError`, subclass of `ValueError`.

**When:** raised by `evaluate_spec`, `load_bustype_json` and
`BusTypeSpec.from_dict` when a `BusType` document or its formula is
rejected: not a `BusTypeSpec`, empty formula, unknown symbol, or a formula
outside the allowed grammar or its size bounds.

**What it means:** the parser never evaluates the text; its own
`groundfield.io._formula.FormulaError` arrives wrapped in this class.

**What to do:** regenerate the document with `save_bustype_json`, or fix
the formula by hand.

### `ResultFormatError`

`gf.ResultFormatError`, subclass of `ValueError`.

**When:** raised by `load_result` for a file that is not a readable result
file: wrong `format`, a newer `schema_version`, invalid JSON or UTF-8, or
metadata outside what the writer produces.

**What it means:** the file is not loaded.

**What to do:** load it with the groundfield version that wrote it
(`written_by`), or regenerate it.

### `OverpassError`

`gf.OverpassError`, subclass of `RuntimeError`.

**When:** raised by the OSM queries on a non-success Overpass answer or a
malformed payload (after the retries); `status_code` and the start of the
body are attached.

**What it means:** the answer is not cached.

**What to do:** retry later, reduce the radius, or work from a cached /
stored answer.

## Other warnings (plain `UserWarning`)

| Where | When | What to do |
|---|---|---|
| `coupling` (Carson) | `earth_inductive_model="carson_series"` with layered soil: the upper-layer $\rho_1$ is used | use `"sommerfeld"`, which supports layered soil |
| `solver` (inductive assembly) | `earth_inductive_model="perfect_mirror"` with inductive segments on both sides of the soil surface (an overhead lead and a buried PEN); once per solve, naming the pairs and conductors | use `"sommerfeld"` (the default) for numbers |
| `solver` (`image_2layer`, `kernel="point"`) | the deepest segment ends less than two segment lengths above the layer interface | refine `segment_length` near the interface, or use the default line kernel |
| `solver` (kernels) | two segments of different wires closer than the 1 mm singularity clamp | merge the wires or move them apart |
| `solver` (pipeline) | a source looks like the hand-made counter-source of another one | use one source with `return_to` |
| `solver` (pipeline) | conductors could not be matched to the branch network; their inductance is not applied | check the conductor end points |
| `World.set_boundary_conditions` | a boundary field set to (or back from) a non-default value | none needed; the fields are not read |
| `geometry` (`PolylineElectrode`) | `closed=True` with only two vertices (treated as open) | give at least three vertices |
| `generators` | buildings that cannot be connected to a PEN cable or a feeder, exhausted feeder slots, a cabinet without electrodes, a footprint widened to 1 m, `source_return_to` overriding the measurement return | check the layout parameters named in the message |
| `geo` | an unreadable Overpass cache file (re-fetched) | none |
| `fit` / `io` | a fit or exported formula that does not depend on frequency | expected for frequency-independent data; otherwise check the samples |

The warning of `perfect_mirror` across the soil surface is user decision
Q27: its additive image has no physical basis for such a pair, and its
result is a diagnostic baseline.

## Built-in exceptions with a fixed meaning

| Exception | Raised when |
|---|---|
| `KeyError` | an electrode, junction, conductor or source name is unknown (the message suggests the closest names) |
| `ValueError` | an invalid value: non-finite or non-positive physical input, a current source of 0 A, a short-circuited two-terminal source, a loop of voltage sources, a junction without soil contact, an empty world |
| `TypeError` | an unknown keyword of a factory, or a soil that is not a soil model |
| `NotImplementedError` | `FieldResult.potential` for three or more soil layers (`BackendCapabilityError`, a subclass, for a world a backend cannot represent, such as a strip with `FemOptions.model="axisymmetric"`) |
| `ImportError` | a function of `groundfield.geo` without the `geo` extra, `fem`'s 3-D model without the `fem3d` extra, or `to_bustype` without `groundinsight` (the message gives the install command) |
| `pydantic.ValidationError` | a model field fails validation (subclass of `ValueError`) |
