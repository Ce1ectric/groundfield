# `image` — homogeneous soil

This page explains the default engine for homogeneous soil: the physical
model, the equations it solves, how accurate its results are and where it
must not be used. Read it before you rely on `image` for touch and step
voltages next to a grid, a star or a bonded cluster.

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | `HomogeneousSoil`; a `TwoLayerSoil` is forwarded to [`image_2layer`](image_2layer.md) |
    | Green's function | point source plus its air image, closed form |
    | Current model | one uniform leakage current per straight wire (user decision Q30) |
    | Kernel | exact line source in Galerkin form (default); `kernel="point"` for the midpoint point source |
    | Unknowns | one current per straight wire, node potentials, branch currents |
    | Accuracy | exact for its model; impedances about 1 % from per-segment `mom` on grids, a few per cent on stars; near-field quantities up to tens of per cent ([Validation](#validation)) |
    | Status | default engine; `cim` and `image_nlayer` are deprecated names that run it |

## What it solves

`image` is the classical grounding calculation of Sunde, Tagg and Dwight.
It computes the stationary current field of buried electrodes and the
potentials and impedances that follow from it, under these assumptions:

- **Soil.** A homogeneous, isotropic half-space of resistivity $\rho$
  (in Ω·m), bounded at $z = 0$ by insulating air. Depth $z$ counts
  positive downwards.
- **Quasi-static field.** No displacement current, and a skin depth large
  against the electrode (frequencies below 1 kHz, see the
  [engine family](index.md)). The galvanic solution is then independent of
  the frequency: one solve serves every frequency of a galvanic world.
- **Thin wires.** The wire radius $a$ is small against the segment
  length. The current flows on the wire axis and the potential is
  observed on the wire surface.
- **Uniform current per straight wire.** Every straight wire of an
  electrode leaks one current that is uniform along its length: a grid has
  one current per full wire line, a star one per arm, a polyline one per
  straight run (a building outline one per side). Rods, strips and rings
  are one wire each. The wires of an electrode are ideally bonded, so they
  share one potential, and every result is reported per electrode (user
  decisions Q30 and Q44).
- **Average potential.** The potential of a wire is the average of the
  field over the wire (Galerkin testing), not the value at one point.

For this model `image` is exact. What remains is the error of the model
itself, chiefly the uniform current, which the [Validation](#validation)
section quantifies.

## Mathematical and physical context

### Green's function of the half-space

A point current $I$ (in A) injected at $\mathbf r_s = (x_s, y_s, z_s)$
with $z_s > 0$ produces the potential $\varphi$ (in V) that satisfies

$$
\nabla^2\varphi = -\rho I\,\delta(\mathbf r - \mathbf r_s),
\qquad
\frac{\partial\varphi}{\partial z}\Big|_{z=0} = 0 .
$$

The first equation says that the injected current is the only source of
the field; the second that no current crosses the soil surface into the
air. The image construction solves the problem in closed form: a second
source of the same sign at the mirror point
$\mathbf r_s' = (x_s, y_s, -z_s)$ makes the normal derivative vanish on
$z = 0$, so

$$
\varphi(\mathbf r) = \frac{\rho I}{4\pi}\left(
\frac{1}{\lvert\mathbf r - \mathbf r_s\rvert}
+ \frac{1}{\lvert\mathbf r - \mathbf r_s'\rvert}\right).
$$

Physically, the insulating surface doubles the potential of a source at
the surface: the current that would flow into the upper half-space is
reflected back into the soil.

### Potential of a straight wire

A straight segment of length $L$ from $A$ to $B$ that leaks the current
$I$ uniformly along its length produces, at a point $X$ with the end-point
distances $R_1 = \lvert X - A\rvert$ and $R_2 = \lvert X - B\rvert$,

$$
\varphi_\text{line}(X) = \frac{\rho I}{4\pi L}
\ln\frac{R_1 + R_2 + L}{R_1 + R_2 - L}.
$$

This is the line integral of the point-source potential along the
segment. It holds for any orientation of the segment; its air image is the
mirrored segment. Since $R_1 + R_2$ is constant on prolate spheroids with
foci $A$ and $B$, the equipotentials of a uniformly loaded segment are
those spheroids.

### Current model and discretisation

Every electrode is cut into straight segments no longer than
`Engine.segment_length` (rings into chords, meshes and grids into their
wire pieces between crossings). The unknown currents belong to *owners*:
an owner is one straight wire of an electrode. Owner $o$ of total length
$L_o$ leaks the current $I_o$, distributed uniformly per unit length, so
its segment $k$ of length $L_k$ carries

$$
I_k = I_o\,\frac{L_k}{L_o}.
$$

Let $K_{kl}$ (in 1/m) be the reaction coefficient: $\rho/(4\pi)$ times
$K_{kl}$ is the average potential on segment $k$ per ampere leaked
uniformly by segment $l$, air image included. The potential of owner $o$
is the length-weighted average of its segment potentials, with the same
weights $L_k/L_o$ that distribute its current. With the weight matrix
$W$, $W_{ko} = L_k/L_o$ for $k \in o$ and zero otherwise, the owner
multiport matrix is

$$
Z_{oo'} = \frac{\rho}{4\pi}\sum_{k \in o}\sum_{l \in o'}
\frac{L_k}{L_o}\,K_{kl}\,\frac{L_l}{L_{o'}}
= \frac{\rho}{4\pi}\,\bigl(W^\top K W\bigr)_{oo'} .
$$

$Z_{oo'}$ is the average potential of wire $o$ per ampere leaked by wire
$o'$. This is the Galerkin reduction dual to the uniform-current ansatz:
the same weights distribute the current and average the potential, so
$Z_{oo'} = Z_{o'o}$ whenever $K$ is symmetric, and the multiport matrices
of the engine are reciprocal.

### Source kernel

$K$ uses the exact line-source kernel by default (`Engine.kernel=None`
resolves to `"line"` for this family). Every pair of segments is the
double integral of the reduced kernel
$1/\sqrt{\lvert\mathbf x - \mathbf y\rvert^2 + a^2}$ over both segments:
the current on the axis seen from the surface, which keeps the self term
finite. For a straight wire the result is then independent of the segment
length. The legacy point kernel (`kernel="point"`) places each segment's
current at its midpoint and uses the collocation self term
$2\ln(L/a)/L$; its grounding resistance converges to a value about
1–2 % below the uniform-current solution, at every segment length. The
[kernels page](../kernels.md) derives both and shows where the difference
comes from.

A concrete encasement enters as the coefficient $C$ (in Ω·m) of the
electrode, the radial resistance of the shell times its length: every
segment adds $C/L_k$ to its diagonal entry
([concrete shells](../concrete-shells.md)).

### Network solve

The owner multiport matrix enters the augmented nodal system of the
[engine family](index.md#network-equations):

- electrodes bonded by ideal conductors share a node potential, and all
  owners of an electrode belong to its node;
- finite conductors are branches with their resistance, internal and
  partial inductance and earth-return correction;
- junctions are nodes without soil contact;
- a current source injects $+I$ at `attached_to` and $-I$ at `return_to`;
- a voltage source adds its current as an unknown and fixes
  $\varphi_a - \varphi_r + Z_q I = U$ (modified nodal analysis), with the
  potentials $\varphi_a$ and $\varphi_r$ of its two terminals and its
  internal impedance $Z_q$.

Source-free clusters float with zero net current. A lone passive
electrode then carries no current (under the uniform ansatz its leakage
would have to be zero everywhere) and reports the potential the field
imposes; a bonded group of passive electrodes exchanges current between
its members to stay one equipotential. The reported potential of an
electrode is the length-weighted average over its wires, which equals the
potential of its cluster. The impedances that can be read from the result
(input, cluster and electrode impedance) are explained on
[sources and measurement](../sources-and-measurement.md#three-impedances).

### Field evaluation after the solve

`FieldResult.potential(points)` superposes the segment sources with the
kernel of the solve:

- with the line kernel, the exact line potential near a segment and the
  point form plus its quadrupole term farther away;
- with the point kernel, point sources with a 1 mm distance floor.

Surface potentials next to a wire are therefore correct with the default
kernel. The point form misses the field above a wire by several per cent
when the segment length is comparable to the distance
([kernels](../kernels.md#field-points)).

## Numerics

**Convergence.** With the line kernel a straight wire gives the same
resistance at every segment length: the generated kernel check below
shows a 3 m rod and a 10 m strip within about $10^{-6}$ of their exact
uniform-current resistance at every segment length it tests. A ring converges
like the square of the segment length, because its chords approximate the
circle (−0.24 % at 0.5 m, −1.5e-4 at 0.125 m for a ring of radius 2 m).
The point kernel converges as well, but to a value 1–2 % below the
uniform-current solution, so a segment-length study cannot reveal that
offset ([kernels](../kernels.md)). `convergence_study` runs such a study
for any world.

**Cost.** The reaction matrix needs one kernel evaluation per pair of
segments: $O(N^2)$ time for $N$ segments and $8N^2$ bytes. It is kept in
memory while it fits 1 GiB ($N \le 11\,585$) and evaluated in row blocks
beyond. The network system has one unknown per straight wire, node and
finite branch and is small. A galvanic world is solved once for all
frequencies; inductive conductors add one complex solve per frequency.
The line kernel takes 3.8 times as long as the point kernel for 16 house
grids ($N$ = 1920) with numba installed, 5.8 times without
([kernels](../kernels.md#cost)). `Engine.max_segments` (default
`diagnostics.HARD_LIMIT` = 20 000 segments) stops a larger world before
anything of size $N$ is allocated. `FieldResult.potential` works in row
blocks of at most 16 MiB, so large grids of field points do not allocate
`(points × sources × 3)` tensors. Timings of whole networks are on the
[performance page](../performance.md).

**Error estimates.** The engine reports no error estimate of its own:
within its model and with the line kernel the discretisation error is
essentially the chord error of curved wires (the far-form remainder of
the kernel stays near $10^{-6}$ in the kernel check). The model error (the uniform current) is estimated by a
second engine on the same world: `compare_engines` with `mom`, or
`diagnose()` for the warnings that apply.

## Validation

Two effects make up the accuracy of `image`, and they differ in size:

- **Impedances** are accurate to about a per cent on grids and a few per
  cent on stars against `mom` on the same discretisation (one current per
  segment, same point kernel). Rings and single rods are exact by symmetry
  or nearly so.
- **Field quantities next to the electrode** (touch, step and mesh
  voltages within a few metres of a grid corner or a star tip) can be off
  by about ten per cent next to a grid corner and by tens of per cent next
  to a star tip, with the step voltage **under**-estimated — the
  non-conservative direction for a safety assessment.

The generated tables below give the measured values; the kernel check at
their end compares both kernels with the exact uniform-current
resistance. References and all cases are on the
[validation page](../validation.md).

### Field quantities and the uniform-current ansatz

The uniform-current ansatz is **impedance-grade, not field-grade**. The
impedance averages the potential over the electrode, and a wrong current
distribution changes that average only to second order. A surface
potential next to a grid corner or the tip of a star arm depends on the
local current density, which the ansatz gets wrong: a real grid leaks more
current at its corners and edges than in its interior, a star more at its
tips. One current per wire line lets the outer lines of a grid carry more
than the inner ones; along each wire, and along each arm of a star, the
current is still uniform.

In the near-field table below the touch voltage has its feet 1 m outside
the corner or tip, and the step starts at the surface point **above** the
corner or tip and reaches 1 m outwards. The warning text and
`groundfield.postprocess.safety` quote the step from 1 m to 2 m outside
instead, next to the touch voltage at 1 m: at the grid corner the image
family reads +6 % / −8 % with the default line kernel (+5 % / −8 % with
the point kernel).

The safety helpers therefore warn: `touch_voltage`, `step_voltage`,
`touch_voltage_envelope` and `touch_voltage_at_part` emit
`UniformCurrentFieldWarning` when a result of the image family is
evaluated within 5 m (`NEAR_FIELD_DISTANCE_M`) of a grid, mesh, star,
strip, open polyline or a bonded multi-electrode cluster. For a safety
assessment of such geometries solve with `mom` (or compare both); for
impedances, rho-f samples and far-field potentials the image family is the
right tool.

<!-- generated:begin image -->
Generated from the benchmark run measured on 2026-11-16 with groundfield 2026.11.0 (Python 3.12.16, NumPy 2.4.4, SciPy 1.17.1, numba installed: yes; Darwin arm64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

| Benchmark | Measure | Bound | Worst measured | Cases | Status | Note |
|---|---|---|---|---|---|---|
| B2 Rod, homogeneous soil | abs(Z/Z_ref - 1) | 1.5 % | 0.79 % | 3 | pass | point kernel |
| B3 Ring, homogeneous soil | abs(Z/Z_ref - 1) | 1.5 % | 1.2 % | 3 | pass | point kernel (10.28) |
| B8 Two-port reciprocity | abs(Z_AB/Z_BA - 1) | 1e-10 | 0 | 1 | pass |  |
| B10 Ring with three bonded rods | abs(Z/Z_ref - 1) | 3 % | 2.8 % | 2 | pass | uniform current per straight wire (10.22); target 2 % |
| B13 Ring with an inductive 400 m line (`image_vs_mom`) | compare_engines row deviation (Re and Im) | 5 % | 1.0 % | 1 | pass | inductive branch shared; galvanic part differs by the ansatz |
| B13 Ring with an inductive 400 m line (`pollaczek_x`) | abs(X'/X'_P - 1), X' = Im(Z_1600 m - Z_800 m) / 800 m at 1 kHz | 0.5 % | 9.0e-6 | 1 | pass | the two-length difference cancels the end effects; the 400 m line itself is 3.7 % (1 kHz) and 10.8 % (50 Hz) below l X'_P (finite length, l / delta = 2.5 and 0.56) |
| B13 Ring with an inductive 400 m line (`pollaczek_r`) | abs(R'/R'_P - 1), same difference | 2 % | 1.1 % | 1 | pass | converges from below with the length (-4.5 % for 400/800 m); the 400 m line is 34 % (1 kHz) and 17 % (50 Hz) below l R'_P |

Near-field quantities against `mom` on the same discretisation (uniform-current ansatz; homogeneous 100 Ω·m, ds = 0.5 m):

| Case | Backend (kernel) | Impedance | Touch voltage, 1 m outside | Step voltage, 1 m | Mesh voltage |
|---|---|---|---|---|---|
| grid 20 m × 20 m, 4 × 4 meshes, at 0.5 m; corner | `mom` (reference) | 2.3473 Ω | 0.9377 V/A | 0.3664 V/A | 0.5305 V/A |
| | `image` (`point`) | +0.57 % | +5.1 % | −11 % | +6.1 % |
| | `image` (`line`) | +1.1 % | +6.4 % | −11 % | +8.6 % |
| star, 4 arms of 10 m, at 0.5 m; arm tip | `mom` (reference) | 5.3784 Ω | 3.4271 V/A | 0.7858 V/A | — |
| | `image` (`point`) | +3.4 % | +8.8 % | −22 % | — |
| | `image` (`line`) | +4.3 % | +10 % | −23 % | — |
| ring r = 5 m at 0.5 m | `mom` (reference) | 6.3596 Ω | 2.9730 V/A | 1.0493 V/A | — |
| | `image` (`point`) | +4.4e-16 | +6.7e-16 | −1.2e-15 | — |
| | `image` (`line`) | +0.88 % | +1.8 % | +5.3e-5 | — |

Kernel check — signed deviation from the exact uniform-current resistance:

| Case | Exact uniform-current R | Kernel | ds = 2 m | ds = 0.5 m | ds = 0.25 m | ds = 0.125 m |
|---|---|---|---|---|---|---|
| rod 3 m from the surface, a = 8 mm, 100 Ω·m (B2 world) | 33.4997 Ω | `point` |  | −1.2 % | −1.5 % | −1.7 % |
|  |  | `line` |  | −3.6e-7 | −8.0e-7 | −1.0e-6 |
| ring r = 2 m at 0.8 m, a = 8 mm, 100 Ω·m (B3 world) | 12.4816 Ω | `point` |  | −1.2 % | −1.2 % | −1.2 % |
|  |  | `line` |  | −0.24 % | −6.1e-4 | −1.5e-4 |
| strip 10 m at 0.5 m, a = 5 mm, 100 Ω·m | 14.9411 Ω | `point` | +0.80 % | −0.96 % |  | −1.2 % |
|  |  | `line` | −1.1e-16 | −8.3e-7 |  | −7.6e-7 |
<!-- generated:end image -->

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Soil | `HomogeneousSoil`; two layers go to `image_2layer` | `BackendCapabilityError` for three distinct layers |
| Frequency | quasi-static, $f < 1$ kHz | not checked |
| Thin wire | $a \ll$ segment length; $L \le a$ rejected, $L < 5a$ reported by `diagnose()` | `ValueError` |
| Burial | every segment in the soil, $2z > a$ | `ValueError` naming the electrode |
| Shallow segments, point kernel | $L \le 4z$ | `ShallowSegmentWarning` with the bias |
| Field quantities within 5 m of grids, stars, strip ends, bonded clusters | ansatz error up to tens of per cent | `UniformCurrentFieldWarning` |
| Size | `Engine.max_segments`, default 20 000 segments | `ResourceLimitError` before allocation |

Notes on the rows:

- **Frequency.** The galvanic kernel is frequency independent, so the
  limit is not checked; above about 1 kHz the quasi-static model itself
  no longer holds.
- **Burial.** The mirror image of a wire must not touch the wire itself,
  so a wire at depth $z$ needs $2z > a$.
- **Shallow segments.** The point form of a segment's own image is biased
  high when the segment is longer than four times its depth; the warning
  quotes the bias in per cent. The default line kernel is not affected.
- **Field quantities.** Use `mom` for touch, step and mesh voltages next to
  such electrodes, or compare both engines ([Validation](#validation)).
- **Size.** The reaction matrix is kept in memory up to
  $N = 11\,585$ segments and evaluated in blocks beyond; raise
  `Engine.max_segments` to solve a larger world on purpose.

What to use instead, and the other limits of the family:
[known limitations](../../reference/known-limitations.md#current-ansatz-and-source-kernels);
the warnings: [warnings reference](../../reference/warnings.md).

## Literature

- Dwight, H. B. (1936). Calculation of resistances to ground. *Electrical
  Engineering* 55(12), 1319–1328 (the uniform-current, average-potential
  closed forms of `groundfield.references.dwight1936`).
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission Systems*.
  Dover, New York, ch. 2–3 (image construction, average-potential method).
- Tagg, G. F. (1964). *Earth Resistances*. George Newnes, London.
- Heppe, R. J. (1979). Computation of potential at surface above an
  energized grid or other electrode, allowing for non-uniform current
  distribution. *IEEE Transactions on Power Apparatus and Systems*
  PAS-98(6), 1978–1989 (why the uniform-current ansatz misplaces surface
  potentials).
- IEEE Std 80-2013. *IEEE Guide for Safety in AC Substation Grounding*
  (mesh and step voltage of grids; `groundfield.references.ieee80`).

Architecture decisions: [ADR-0001](../../adr/0001-two-layer-method.md),
[ADR-0015](../../adr/0015-line-source-kernel-default.md). API:
`groundfield.solver.image.solve_image` (called through `Engine.solve`).

## Audit trail

??? note "One current per straight wire (user decisions Q30 and Q44)"

    Before 2026.11 the uniform current belonged to the whole electrode. Since
    user decisions Q30 and Q44 it belongs to every straight wire of an
    electrode. With one current for the whole grid, the 20 m grid with
    4 × 4 meshes read +3.9 % against `mom`, and the touch and step voltages
    1 m and 1 m … 2 m outside its corner read +20 % / −21 % (now +6 % / −8 %
    with the line kernel, +5 % / −8 % with the point kernel). The generated
    tables are measured with one current per straight wire.

??? note "Kernel default (ADR-0015)"

    The exact line kernel is the default of the image family since 2026.11.
    The point kernel was the only kernel up to 0.15.0 and reproduces its
    numbers (`kernel="point"`).
