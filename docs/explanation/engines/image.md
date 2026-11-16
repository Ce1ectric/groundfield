# `image` — homogeneous soil

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | `HomogeneousSoil`; a `TwoLayerSoil` is forwarded to [`image_2layer`](image_2layer.md) |
    | Green's function | point source plus its air image, closed form |
    | Current model | one uniform leakage current per unit length per straight wire of an electrode (a wire line of a grid, an arm of a star, a straight run of a polyline; since 2026.11, user decision Q30) |
    | Kernel | exact line source in Galerkin form (default since 2026.11); midpoint point source with `kernel="point"` |
    | Unknowns | one current per straight wire (and soil layer) plus node potentials and branch currents |
    | Status | default engine; `cim` and `image_nlayer` are deprecated names that run it |

`image` is the reference implementation of the classical grounding
calculation of Sunde, Tagg and Dwight: every electrode carries a uniform
leakage current, its potential is the average over its surface, and the
soil is a homogeneous half-space bounded by insulating air. It is exact
for that model — the remaining error is the model itself, which this page
quantifies. Since 2026.11 the uniform current belongs to every straight
wire of an electrode rather than to the whole electrode (user decisions
Q30 and Q44): a grid has one current per full wire line, a star one per
arm, a polyline one per straight run (a building outline one per side);
the wires of an electrode are ideally bonded, so they share one potential
and every result is still reported per electrode. Rods, strips and rings
are one wire.

## Mathematical and physical context

In homogeneous soil of resistivity $\rho$ the stationary current field of a
point current $I$ at $\mathbf r_s = (x_s, y_s, z_s)$, $z_s > 0$, satisfies
$\nabla^2\varphi = -\rho I\,\delta(\mathbf r - \mathbf r_s)$ with an
insulating soil surface ($\partial\varphi/\partial z = 0$ at $z = 0$).
The image construction solves it in closed form: a second source of the
same sign at the mirror point $\mathbf r_s' = (x_s, y_s, -z_s)$ makes the
normal derivative vanish on $z = 0$,

$$
\varphi(\mathbf r) = \frac{\rho I}{4\pi}\left(
\frac{1}{|\mathbf r - \mathbf r_s|} + \frac{1}{|\mathbf r - \mathbf r_s'|}\right).
$$

A straight segment of length $L$ from $A$ to $B$ that leaks the current
$I$ uniformly along its length produces, at a point $X$ with the end-point
distances $R_1 = |X - A|$ and $R_2 = |X - B|$,

$$
\varphi_\text{line}(X) = \frac{\rho I}{4\pi L}
\ln\frac{R_1 + R_2 + L}{R_1 + R_2 - L},
$$

valid for any orientation of the segment; its image is the mirrored
segment. The quasi-static assumption (no displacement current, skin depth
large against the electrode, see the [engine family](index.md)) makes the
galvanic solution frequency independent: one solve serves every frequency
of a galvanic world.

## Discretisation and unknowns

Every electrode is cut into straight segments no longer than
`Engine.segment_length` (rings into chords, meshes and grids into their
wire pieces between crossings). An electrode is one *owner* of current:
it leaks the current $I_o$, distributed uniformly per unit length, so
segment $k$ of owner $o$ carries $I_k = I_o L_k/L_o$. With the segment
reaction matrix $K$ (potential at segment $k$ per unit current on segment
$l$, in 1/m) the potential of owner $o$ is the length-weighted average of
its segment potentials, and the owner multiport matrix is

$$
Z_{oo'} = \frac{\rho}{4\pi}\sum_{k \in o}\sum_{l \in o'}
\frac{L_k}{L_o}\,K_{kl}\,\frac{L_l}{L_{o'}}
= \frac{\rho}{4\pi}\,\bigl(W^\top K W\bigr)_{oo'} .
$$

This is the Galerkin reduction dual to the uniform-current ansatz: the
same weights distribute the current and average the potential, so
$Z_{oo'} = Z_{o'o}$ whenever $K$ is symmetric, and `grounding_impedance`
reads back exactly the diagonal the solver inverted.

The reaction matrix uses the exact line-source kernel by default
(`Engine.kernel=None` resolves to `"line"` for this family): every pair of
segments is the double integral of the reduced kernel
$1/\sqrt{|\mathbf x - \mathbf y|^2 + a^2}$ over both segments, which makes
the result for a straight wire independent of the segment length. The
legacy point kernel (`kernel="point"`) places each segment's current at its
midpoint and uses the collocation self term $2\ln(L/a)/L$; its grounding
resistance converges to a value about 1–2 % below the uniform-current
solution, at every segment length. The [kernels page](../kernels.md) derives
both and shows where the difference comes from. A concrete encasement
enters as the coefficient $C$ of the electrode: every segment adds $C/L_k$
to its diagonal ([concrete shells](../concrete-shells.md)).

## Network solve

The owner multiport matrix enters the augmented nodal system of the
[engine family](index.md#network-equations): electrodes bonded by ideal
conductors share a node potential, finite conductors are branches with
their resistance, internal and partial inductance and earth-return
correction, junctions are nodes without soil contact, current sources
inject $+I$ at `attached_to` and $-I$ at `return_to`, and a voltage source
adds its current as an unknown and fixes
$\varphi_a - \varphi_r + Z_q I = U$ (modified nodal analysis, $Z_q$ its
internal impedance). Source-free clusters float
with zero net current: a lone passive electrode then carries no current
(under the uniform ansatz its leakage would have to be zero everywhere)
and reports the potential the field imposes; a bonded group of passive
electrodes exchanges current between its members to stay one
equipotential. The impedances that can be read from the result — input,
cluster and electrode impedance — are explained on
[sources and measurement](../sources-and-measurement.md#three-impedances).

## Field evaluation after the solve

`FieldResult.potential(points)` superposes the segment sources with the
kernel of the solve: with the line kernel the exact line potential near a
segment and the point form plus its quadrupole term farther away, with the
point kernel point sources with a 1 mm distance floor. Surface potentials
next to a wire are therefore correct with the default kernel; the point
form misses the field above a wire by several per cent when the segment
length is comparable to the distance ([kernels](../kernels.md#field-points)).
The evaluation runs in row blocks of at most 16 MiB, so large grids of
field points do not allocate `(points × sources × 3)` tensors.

## Field quantities and the uniform-current ansatz

The uniform-current ansatz is **impedance-grade, not field-grade**. The
impedance averages the potential over the electrode, and a wrong current
distribution changes that average only to second order. A surface
potential next to a grid corner or the tip of a star arm depends on the
local current density, which the ansatz gets wrong: a real grid leaks
more current at its corners and edges than in its interior, a star more
at its tips. One current per wire line lets the outer lines of a grid
carry more than the inner ones (since 2026.11, user decision Q30); along
each wire, and along each arm of a star, the current is still uniform.
Against `mom` on the same discretisation (one current per segment, same
point kernel) the table in the [validation section](#validation) below
shows about a per cent on grid impedances (the 20 m grid with 4 × 4
meshes +3.9 % with one current for the whole grid, before Q30) and a few
per cent on stars, but up to tens of per cent on touch, step and mesh
voltages next to a star tip and up to about ten per cent next to a grid
corner, with the step voltage **under**-estimated — the non-conservative
direction for a safety assessment. Rings and single rods are exact by
symmetry or nearly so. In the table the touch voltage has its feet 1 m
outside the corner or tip and the step starts at the surface point
**above** the corner or tip and reaches 1 m outwards; the warning text and
`groundfield.postprocess.safety` quote the step from 1 m to 2 m outside
instead, next to the touch voltage at 1 m: +6 % / −8 % at the grid corner
with the default line kernel (+5 % / −8 % with the point kernel; one
current for the whole grid gave +20 % / −21 %).

The safety helpers therefore warn: `touch_voltage`, `step_voltage`,
`touch_voltage_envelope` and `touch_voltage_at_part` emit
`UniformCurrentFieldWarning` when a result of
the image family is evaluated within 5 m (`NEAR_FIELD_DISTANCE_M`) of a
grid, mesh, star, strip, open polyline or a bonded multi-electrode
cluster. For a safety assessment of such geometries solve with `mom` (or
compare both); for impedances, rho-f samples and far-field potentials the
image family is the right tool.

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Soil | `HomogeneousSoil` (a two-layer soil is forwarded to `image_2layer`) | `BackendCapabilityError` for three distinct layers |
| Frequency | quasi-static, $f < 1$ kHz | — (not checked; the galvanic kernel is frequency independent) |
| Thin wire | wire radius $a \ll$ segment length; $L \le a$ is rejected, $L < 5a$ is reported by `diagnose()` | `ValueError` |
| Burial | every segment entirely in the soil; the mirror image must not touch the wire ($2z > a$) | `ValueError` naming the electrode |
| Shallow segments (point kernel only) | $L \le 4z$ for the point form of the self image | `ShallowSegmentWarning` with the bias in per cent |
| Field quantities near grids, stars, strip ends, bonded clusters | within 5 m: ansatz error up to tens of per cent near star tips, about ten per cent near grid corners (validation table) | `UniformCurrentFieldWarning` in the safety helpers |
| Size | reaction matrix kept in memory while $8N^2 \le$ 1 GiB ($N \le 11\,585$), block evaluation beyond; the engine's limit `Engine.max_segments` (default `diagnostics.HARD_LIMIT` = 20 000 segments) | `ResourceLimitError` before any allocation |

## Validation

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
