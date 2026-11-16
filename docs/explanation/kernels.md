# Kernels: point source and exact line source

This page explains how the integral engines turn the current of one wire
segment into the potential at another — the *kernel* — and why the two
kernels of `groundfield` differ by a systematic 1–2 % on grounding
resistances. Read it when `kernel="point"` and `kernel="line"` disagree,
when you compare `groundfield` with another program, or before you choose
a segment length.

!!! abstract "At a glance"

    | | point kernel | line kernel |
    |---|---|---|
    | Source of a segment | its current at the midpoint | its current spread uniformly along the axis |
    | Observation | at the midpoint of the other segment (collocation) | averaged over the other segment (Galerkin) |
    | Used by | `mom`, `mom_sommerfeld`; `kernel="point"` of the image family | the image family by default since 2026.11 (ADR-0015) |
    | Resistance of a straight wire | converges to a value about 1–2 % low | exact at any segment length |
    | Cost | lowest | about 1.1 … 5.8 times the point kernel ([Cost](#cost)) |

## Mathematical and physical context

An electrode is cut into straight segments. Segment $j$ has the length
$L_j$, the wire radius $a_j$, the unit tangent $\mathbf t_j$, the midpoint
$\mathbf P_j$ and the axis $\mathbf x_j(l')$, $0 \le l' \le L_j$. The image
family assumes that it leaks its current $I_j$ uniformly along its axis,
with the line density $I_j/L_j$ in A/m.

In homogeneous soil of resistivity $\rho$ with an insulating surface, a
point current $I$ at $\mathbf y$ produces at $\mathbf x$ the potential
$\rho I/(4\pi)$ times $1/\lvert\mathbf x - \mathbf y\rvert$ plus the same
term for the mirror point of $\mathbf y$ above the surface (the *air
image*, which makes the surface current-free). Integrating the line
density of segment $j$ along its axis and averaging the result over the
observing segment $i$ gives the mean potential of segment $i$,

$$
\bar\varphi_i = \frac{\rho}{4\pi}\sum_j K_{ij}\,I_j ,
$$

which says that the potential of every segment is a weighted sum of all
segment currents; the weights $K_{ij}$ (in 1/m) are the *reaction
coefficients*. Exactly,

$$
K_{ij} = \frac{1}{L_i L_j}\int_{0}^{L_i}\!\!\int_{0}^{L_j}
\frac{\mathrm dl\,\mathrm dl'}{\sqrt{\lvert\mathbf x_i(l) - \mathbf x_j(l')\rvert^2 + a^2}}
\quad(+\text{ the same integral with the image of segment } j),
$$

the mean over both segments of the potential that a unit current on one
produces on the other. The **reduced kernel**
$1/\sqrt{r^2 + a^2}$ puts the source current on the wire axis and the
observation point on the wire surface, a distance $a$ away; this keeps the
self term finite. For two different wires $a^2 = \tfrac12(a_i^2 + a_j^2)$.

Averaging the potential with the same weights that distribute the current
is the **Galerkin** choice. It makes $K$ symmetric ($K_{ij} = K_{ji}$),
so every multiport matrix built from it is reciprocal, and for a straight
wire the sum over all segment pairs is the double integral over the whole
wire — the result does not depend on how finely the wire is cut. A kernel
is an approximation of $K_{ij}$; the two kernels below differ in how they
evaluate it.

## The point kernel

The point kernel (`kernel="point"`, the only kernel of 0.15.0 and of the
`mom` family) replaces both integrals by values at the midpoints:

- **off the diagonal** the current of segment $j$ sits at its midpoint and
  is observed at the midpoint of segment $i$: $K_{ij} \approx 1/r_{ij}$,
  $r_{ij} = \lvert\mathbf P_i - \mathbf P_j\rvert$;
- **on the diagonal** it takes the collocation self term
  $2\ln(L_i/a_i)/L_i$ — the potential of the uniformly loaded segment at
  its own midpoint, $(2/L)\operatorname{arsinh}(L/2a)$, for $L \gg a$ —
  plus the self image $1/(2z_i)$ as a point image, $z_i$ the depth of the
  segment.

**Why it is low.** Take a straight wire cut into segments of equal length
$L$ and compare one row of the matrix with the exact Galerkin entries:

- the segment itself: exact $K_{ii} \approx 2[\ln(2L/a) - 1]/L$ against
  $2\ln(L/a)/L$, so the point kernel over-counts by
  $2(1 - \ln 2)/L = +0.61/L$;
- each adjacent collinear segment: exact
  $\frac{1}{L^2}\int_0^L\!\int_L^{2L}\frac{\mathrm dl'\,\mathrm dl}{l' - l} = 2\ln 2/L$
  against $1/L$, so the two neighbours together are under-counted by
  $2(2\ln 2 - 1)/L = -0.77/L$;
- the segment $k$ places away: exact
  $[(k+1)\ln(k+1) - 2k\ln k + (k-1)\ln(k-1)]/L$ against $1/(kL)$, a
  deficit that falls like $1/k^3$ and adds about $-0.07/L$ over both
  sides.

The net row-sum error is about $-0.23/L$. The row sum itself grows like
$2\ln(\ell/a)/L$ for a wire of total length $\ell$, so the relative error
of the resistance — the ratio of the two — depends on $\ell/a$ and not on
the segment length. **The point kernel converges, but to a value about
1–2 % below the uniform-current solution**, and a convergence study cannot
see it. The generated kernel check on the
[validation page](validation.md#kernel-of-the-image-family) shows it for a
rod, a ring and a strip at every segment length.

Two further consequences of the point form:

- **Short segments.** The collocation self term vanishes at $L = a$ and is
  too small for short segments, so a short piece acts like a short
  circuit. The point kernel therefore takes the Galerkin self term below
  $L/a = 2.990$, where both coincide (`SELF_TERM_CROSSOVER`; every segment
  with $L/a \ge 2.990$ keeps its value bit for bit). Every interface cut is
  kept, however short the piece beyond it; the self term of such a piece is
  evaluated without cancellation.
- **Shallow segments.** The self image $1/(2z)$ is the $L \ll 4z$ limit of
  the image line; for segments longer than four times their depth it is
  biased high (`ShallowSegmentWarning`).

??? note "Measurements and audit trail"

    The row-sum analysis is finding 10.28 of review pass 10. Before the
    Galerkin self term for short segments (finding 10.32, second adversarial
    audit A2S-01), a sliver at a layer interface acted as a short: `mom`
    gave a rod whose tip is 9 mm below $h_1$ 27 % less resistance than the
    same rod 1 mm above it.

## The exact line kernel

The line kernel (`kernel="line"`, default of the image family) evaluates
the double integral of $K_{ij}$. Its inner integral, the potential of a
uniformly loaded segment from $A$ to $B$ at a point $X$ with the end-point
distances $R_1 = \lvert X - A\rvert$ and $R_2 = \lvert X - B\rvert$, has the
classical closed form

$$
\frac{1}{L}\int_0^L \frac{\mathrm dl}{\lvert X - A - l\,\mathbf t\rvert}
= \frac{1}{L}\ln\frac{R_1 + R_2 + L}{R_1 + R_2 - L},
$$

valid for any orientation of the segment: the equipotentials of a
uniformly charged segment are ellipsoids with its end points as foci. The
code evaluates it without cancellation and with the reduced kernel. The
outer average is computed as follows.

**Self term.** With $u = l - l'$ the double integral reduces to one
integral, $K_{ii} = \frac{2}{L^2}\int_0^L \frac{L - u}{\sqrt{u^2 + a^2}}\,\mathrm du$,
which is

$$
K_{ii} = \frac{2}{L}\Bigl[\operatorname{arsinh}\frac{L}{a}
- \sqrt{1 + (a/L)^2} + \frac{a}{L}\Bigr],
$$

the mean potential of the segment on its own surface per unit current; for
$L \gg a$ it tends to $2[\ln(2L/a) - 1]/L$.

**Near pairs** — midpoint distance below six times the longer segment
(`NEAR_FACTOR`) — are integrated exactly. Parallel pairs have a closed form
(the integral of Neumann's mutual-inductance formula for parallel
filaments); touching or crossing pairs use Gauss–Legendre points graded
towards the point of closest approach with the exact inner line potential;
separated skew pairs a plain eight-point rule.

**Far pairs** use the expansion of the segment average. Write a point of
segment $k$ as $\mathbf P_k + s\,\mathbf t_k$ with $-L_k/2 \le s \le L_k/2$
and expand $1/\lvert\mathbf R + s\,\mathbf t\rvert$ to second order in $s$,
$\mathbf R = \mathbf P_i - \mathbf P_j$, $R = \lvert\mathbf R\rvert$. The
linear term averages to zero and the mean of $s^2$ is $L_k^2/12$, which
leaves

$$
\frac{1}{R} + \frac{L_i^2(3c_i^2 - 1) + L_j^2(3c_j^2 - 1) - 12a^2}{24R^3},
\qquad c_k = \frac{\mathbf R\cdot\mathbf t_k}{R},
$$

the point form $1/R$, the *quadrupole term* of each segment's length
($c_k$ is the cosine between the segment and the line joining the
midpoints) and the radius term $-a^2/(2R^3)$ of
$1/\sqrt{R^2 + a^2}$. The remainder is of order $(L/R)^4$.

**The switch** between exact and far form is a blend (user decision D2):
pairs between six and 7.5 segment lengths (`SWITCH_FACTOR`) take
$w\,K_\text{exact} + (1 - w)\,K_\text{far}$, the weight $w$ falling from 1
to 0 with zero slope at both ends (smoothstep in $R^2$). The result is
continuous in the geometry: it spreads by at most 1e-13 over local offsets
of 0.1 m … 8.8 km, and the solve at UTM coordinates lies as close to the
local one as with the point kernel (3e-11 in $Z$).

**Images.** The air image is a mirrored segment and is treated like a
source segment. The Tagg / Sunde images of a two-layer soil lie at least
$2(h_1 - z)$ away ($h_1$ the thickness of the top layer); they get the
treatment of direct pairs — exact near, blended, far form with the
quadrupole term — with two differences:

- **Transverse distance.** An image line that passes the wire at a
  transverse distance $\rho_\perp$ acts with its potential averaged over
  the wire's circumference. By the two-dimensional mean-value property —
  the average of $\ln\lvert x - y\rvert$ over a circle of radius $a$ around
  $x$ is $\ln\max(\rho, a)$, $\rho = \lvert x - y\rvert$ — that average is
  its value on the axis for $\rho_\perp \ge a$. The exact integral of an
  image pair therefore uses $\max(\rho_\perp, a)$ in place of
  $\sqrt{\rho_\perp^2 + a^2}$, which keeps the reduced kernel for a
  collinear image ($\rho_\perp = 0$, a rod against its own image), and the
  far form of images has no radius term. A ring 1 cm above a
  1000 / 30 Ω·m interface then lies within −0.004 % of its exact thin ring.
- **Reach of the quadrupole term.** The quadrupole term of an image is
  tapered off between 60 and 120 segment lengths of vertical distance
  between observer and image (`IMAGE_QUADRUPOLE_REACH`). At 60 segment
  lengths it is at most $1/(6\cdot 60^2)$ of the image's point value
  ($1/(12\cdot 60^2)$ for horizontal wires), of the order of the far form's
  remainder at the switch: the image series of a high contrast has hundreds
  of images, of which few are close. An image is evaluated only on the
  pairs whose longer segment reaches it, and in the reaction matrix, which
  is symmetric, on its upper half.

`Engine.image_far_field = "point"` keeps far images as point sources (see
[Cost](#cost)). On the cross-layer path of
[`image_2layer`](engines/image_2layer.md#electrodes-that-reach-the-interface)
the layered correction is the exact image expansion of the two-layer
Green's function (Sunde) in every layer combination, and its images get
the same treatment on top of the spectral point values; pairs in
different layers carry the transmitted resistivity
$2\rho_1\rho_2/(\rho_1+\rho_2)$, which keeps the matrix reciprocal.

??? note "Measurements and audit trail"

    - **Radius term of the far form.** Up to wave 3 of the 2026.11 work
      the far form had no radius term: it was $a^2/(2R^2)$ — 1.4e-6 at six
      segment lengths of 0.5 m — away from the exact pairs it joins.
    - **One-step switch.** Before user decision D2 the form switched at six
      segment lengths in one step — where the pairs six segments apart on a
      straight wire sit exactly. Round-off of the coordinates decided their
      form, and the same world at different positions spread by up to 4.4e-7
      in the impedance (a 12 m strip) and, in two-layer soil with an image
      distance of six segment lengths, by 1e-5 … 2.6e-5 (audit W2B-01/-02).
    - **Far Tagg / Sunde images.** Up to wave 3 of the 2026.11 work they
      were point sources beyond six segment lengths, which left a straight
      strip in two-layer soil up to 5e-4 off and the potential above it up
      to 1 % off.
    - **Transverse distance** (audit W3A-02). The reduced kernel on images
      left out $a^2/(2d^2)$ of an image $d$ away, so a ring 1 cm above a
      1000 / 30 Ω·m interface read +1.60 % against its exact thin ring.
    - **Reach of the far-image work** (audit W3A-01). Before, a few long
      pieces among short segments extended the reach of every pair.
    - **Bound of the tapered term.** Earlier texts gave $1/(8\cdot 60^2)$
      of the image's point value at the start of the taper, which holds for
      horizontal wires; the largest size, reached for a vertical wire and
      its collinear image, is $1/(6\cdot 60^2)$ (the bound
      $\lvert w\rvert(L^2 + 3a^2)/(6\,\mathrm dz^2)$ that the code uses to
      skip images, `IMAGE_QUADRUPOLE_TOL`).

### Accuracy of the line kernel

For a straight wire the line kernel is exact at any segment length, in
homogeneous and in two-layer soil, up to the $(L/R)^4$ remainder of the
far form and the reduced kernel that the soil-surface image keeps: a 10 m
strip in 1000 / 50 Ω·m with $h_1$ = 1 m is within 3.6e-6 of its
closed-form uniform-current resistance (images axis to axis) for segment
lengths of 0.125 … 2 m, not monotonically (the same strip in homogeneous
soil: within 2.1e-6), and a rod through the layer interface is within
1.4e-6 of its exact two-owner value at every segment length. A ring
converges like the square of the segment length, because its chords
approximate the circle.

What remains against the true thin-wire solution is the uniform-current
ansatz itself: for the 3 m rod of benchmark B2 the line kernel sits about
+0.9 % above the thin-wire value at every segment length (generated
numbers on the [validation page](validation.md#kernel-of-the-image-family)).
The point kernel's −1 … −2 % kernel error partly cancels that ansatz error
on impedances, which is why the old numbers looked closer to some
references than they were.

??? note "Measurements and audit trail"

    The strip and rod numbers are those of audit W3A (4 soils, 3 rods).

### Field points

After the solve, `FieldResult.potential` evaluates the field with the
kernel of the solve. With the line kernel that is the exact line potential
near a segment and the point form plus the quadrupole term
$L_j^2(3c_j^2 - 1)/(24R^3)$ farther away, blended like the reaction matrix
and for the Tagg/Sunde images as well (sources in the lower layer of a
two-layer soil are evaluated in point form by the spectral kernel). Every
`PointSource` of a line-kernel result carries its unit tangent
(`direction`), so a stored result is re-evaluated with line sources;
results written by 0.15.0 have no direction and are evaluated with the
point kernel.

Near a wire the difference is large: on the surface of a rod discretised
at 0.5 m the point form spikes at every segment midpoint (16 … 177 V
around an electrode potential of 32 V), while the line form is smooth and
its surface average equals the electrode potential
(`tests/test_line_kernel.py::test_line_potential_is_equipotential_on_the_wire_surface`).
Surface potentials next to a wire — touch and step voltages — therefore
need the line kernel.

??? note "Measurements and audit trail"

    The spikes of the point form on the wire surface are finding 10.33 of
    review pass 10.

### Cost

The exact near pairs and the quadrupole far field over all $N^2$ pairs
make the line kernel more expensive than the point kernel. A solve of 16
house grids ($N$ = 1920 segments) takes 3.8 times as long as with the
point kernel in homogeneous soil and 1.6 times in two-layer soil
($h_1$ = 5 m) with numba installed (it is used automatically, extra
`fast`), 5.8 and 1.4 times without; the extra time grows with $N^2$. The
blended switch adds the integrals of the pairs between six and 7.5 segment
lengths, and in two-layer soil every image within reach carries the
far-field term, evaluated only on the pairs whose longer segment reaches
it and, in the reaction matrix, on its upper half, with one division per
evaluation. A 40-house network in 1000 / 30 Ω·m with $h_1$ = 1.5 m
(82 images within reach) takes 1.8 times the solve before wave 3 of the
2026.11 work with numba and 3.6 times without.

`Engine.image_far_field = "point"` keeps far images as point sources and
leaves out their far-field term; the near image pairs stay Galerkin
integrals. It takes 1.1 … 1.3 times the solve before wave 3 on house grids
within the upper layer. The 40-house network solves on the cross-layer
path (its transformer rods reach below $h_1$): 0.91 times the solve
before wave 3 and 0.49 times the default with numba. Below thin top layers
the near image pairs remain, 1.5 … 2.4 times the solve before wave 3
(0.51 … 0.74 times the default). The [performance page](performance.md)
has the table. `kernel="point"` remains available and reproduces the
numbers of 0.15.0.

??? note "Measurements and audit trail"

    Audit W3A-01 measured the far-image work before its restriction to the
    pairs that reach an image: a few long pieces among short segments
    multiplied the work by 5.6, and the 40-house network in 1000 / 30 Ω·m
    took 4.2 times the solve before wave 3 with numba and 9.7 times
    without. The near image pairs of `"point"` are those of audit
    W3R2A-01.

## Where each kernel is used

| Component | Kernels | Default |
|---|---|---|
| `image`, `image_2layer` (and the deprecated names `cim`, `image_nlayer`) | line, point | line |
| `mom`, `mom_sommerfeld` (and `bem`) | point | point |
| `solve_mutual_matrix`, `solve_mutual_field` | line, point | the engine's kernel |
| `FieldResult.potential` | the kernel of the solve (`metadata["kernel"]`) | — |
| `fem` | none (volume mesh; the 3-D model has its own smoothed line source) | — |

`metadata["kernel"]` records the kernel of every result, and
`metadata["provenance"]["engine"]` the request, so a result can be
reproduced exactly.

## Literature

- Harrington, R. F. (1968). *Field Computation by Moment Methods*.
  Macmillan, New York (Galerkin and point-matching testing).
- Wilton, D. R., Rao, S. M., Glisson, A. W., Schaubert, D. H.,
  Al-Bundak, O. M., & Butler, C. M. (1984). Potential integrals for
  uniform and linear source distributions on polygonal and polyhedral
  domains. *IEEE Transactions on Antennas and Propagation* AP-32(3),
  276–281 (closed-form potentials of uniform sources).
- Grover, F. W. (1946). *Inductance Calculations: Working Formulas and
  Tables*. Van Nostrand; Dover reprint (the double integrals of $1/r$
  over parallel and inclined filaments).
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission Systems*.
  Dover, New York (average-potential method, line-source potentials).

Architecture decision: [ADR-0015](../adr/0015-line-source-kernel-default.md).
Implementation: `groundfield.solver._line_kernel` (private) behind
`Engine.kernel`.
