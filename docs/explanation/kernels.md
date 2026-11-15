# Kernels: point source and exact line source

Every integral engine needs the potential that the current of one wire
segment produces at another segment. This page explains the two kernels
of `groundfield` — the midpoint **point source** used up to 0.15.0 and by
the `mom` family, and the **exact line source** in Galerkin form that the
image family uses by default since 2026.11 (ADR-0015) — and why the
difference is a systematic 1–2 % on grounding resistances that no
refinement of the segment length removes.

## Mathematical and physical context

Under the uniform-current ansatz of the image family a segment $j$ of
length $L_j$ and wire radius $a_j$ leaks the current $I_j$ uniformly along
its axis. The potential it produces is the line integral of the Green's
function, and the potential "of a segment" that enters the solve is the
average over the observing segment. In homogeneous soil (Green's function
$\rho/(4\pi r)$ plus the air image) the exact reaction coefficient between
two segments is therefore the double integral

$$
K_{ij} = \frac{1}{L_i L_j}\int_{0}^{L_i}\!\!\int_{0}^{L_j}
\frac{\mathrm dl\,\mathrm dl'}{\sqrt{|\mathbf x_i(l) - \mathbf x_j(l')|^2 + a^2}}
\quad(+\text{ the same integral with the image of segment } j),
$$

with the **reduced kernel** $1/\sqrt{r^2 + a^2}$: the source current flows
on the wire axis and the potential is observed on its surface, which makes
the self term finite. The potential in V is $\rho/(4\pi)$ times $K$ applied
to the currents in A. Averaging over the observing segment with the same
weights that distribute the current is the Galerkin choice; it makes $K$
symmetric, so multiport matrices are reciprocal, and for a straight wire
the sum over all segment pairs is the double integral over the whole wire
— independent of how finely it is cut.

## The point kernel

The point kernel (`kernel="point"`, the only kernel of 0.15.0 and of the
`mom` family) approximates this:

- **off the diagonal** the current of segment $j$ sits at its midpoint and
  is observed at the midpoint of segment $i$: $K_{ij} \approx 1/r_{ij}$;
- **on the diagonal** the collocation self term
  $2\ln(L_i/a_i)/L_i$, the potential of the uniformly loaded segment at
  its own midpoint for $L \gg a$, plus the self image $1/(2z_i)$ as a
  point image.

Both approximations err in the same row. For a straight wire cut into
segments of length $L$ the exact Galerkin entries are $K_{ii} \approx
2[\ln(2L/a) - 1]/L$ for the segment itself and $2\ln 2/L$ for each
adjacent collinear segment; the point kernel gives $2\ln(L/a)/L$ and
$1/L$. Per row the point kernel therefore over-counts the self term by
$2(1 - \ln 2)/L = +0.61/L$ and under-counts the two neighbours by
$2(2\ln 2 - 1)/L = -0.77/L$, with a small further deficit from the next
neighbours: a net row-sum error of about $-0.23/L$ (review pass 10, finding
10.28). The row sum itself grows like $2\ln(\ell/a)/L$ for a wire of total
length $\ell$, so the relative error of the resistance is set by
$\ell/a$, not by the segment length: **the point kernel converges, but to
a value about 1–2 % below the uniform-current solution**, and a
convergence study cannot see it. The generated kernel check on the
[validation page](validation.md#kernel-of-the-image-family) shows it for a
rod, a ring and a strip at every segment length.

Two further consequences of the point form:

- The collocation self term vanishes at $L = a$ and is too small for
  short segments; a sliver at a layer interface then acted as a short
  (`mom` gave a rod whose tip is 9 mm below $h_1$ 27 % less resistance
  than the same rod 1 mm above it). Since 2026.11 the point kernel takes
  the Galerkin self term below $L/a = 2.990$, where both coincide
  (`SELF_TERM_CROSSOVER`; every segment with $L/a \ge 2.990$ keeps its
  value bit for bit). Every interface cut is kept, however short the
  piece beyond it; the self term of such a piece is evaluated without
  cancellation (finding 10.32, second adversarial audit A2S-01).
- The self image $1/(2z)$ is the $L \ll 4z$ limit of the image line; for
  segments longer than four times their depth it is biased high
  (`ShallowSegmentWarning`).

## The exact line kernel

The line kernel (`kernel="line"`, default of the image family) evaluates
the double integral above. The inner integral has the classical closed
form of a uniformly loaded segment from $A$ to $B$: at a point $X$ with
$R_1 = |X - A|$, $R_2 = |X - B|$,

$$
\frac{1}{L}\int_0^L \frac{\mathrm dl}{|X - A - l\,\mathbf t|}
= \frac{1}{L}\ln\frac{R_1 + R_2 + L}{R_1 + R_2 - L},
$$

valid for any orientation (evaluated without cancellation, with the
reduced kernel). The outer average is computed as follows:

- **Self term:** $K_{ii} = \dfrac{2}{L}\Bigl[\operatorname{arsinh}\dfrac{L}{a}
  - \sqrt{1 + (a/L)^2} + \dfrac{a}{L}\Bigr]$.
- **Near pairs** (midpoint distance below six times the longer segment,
  `NEAR_FACTOR`): exact. Parallel pairs have a closed form (the integral
  of Neumann's mutual-inductance formula for parallel filaments);
  touching or crossing pairs use Gauss–Legendre points graded towards the
  point of closest approach with the exact inner line potential;
  separated skew pairs a plain eight-point rule.
- **Far pairs:** the expansion of the segment average of the reduced
  kernel,
  $$
  \frac{1}{R} + \frac{L_i^2(3c_i^2 - 1) + L_j^2(3c_j^2 - 1) - 12a^2}{24R^3},
  \qquad c_k = \frac{(\mathbf P_i - \mathbf P_j)\cdot\mathbf t_k}{R},
  \quad a^2 = \tfrac12(a_i^2 + a_j^2),
  $$
  the point form, the quadrupole term and the radius term
  $-a^2/(2R^3)$ of $1/\sqrt{R^2 + a^2}$; the remainder is of order
  $(L/R)^4$. (Up to wave 3 of the 2026.11 work the far form had no radius
  term: it was $a^2/(2R^2)$ — 1.4e-6 at six segment lengths of 0.5 m —
  away from the exact pairs it joins.)
- **The switch between the two** is a blend (user decision D2): pairs
  between six and 7.5 segment lengths (`SWITCH_FACTOR`) take
  $w\,K_\text{exact} + (1 - w)\,K_\text{far}$, the weight $w$ falling from
  1 to 0 with zero slope at both ends (smoothstep in $R^2$). Up to then
  the form switched at six segment lengths in one step — where the pairs
  six segments apart on a straight wire sit exactly. Round-off of the
  coordinates decided their form, and the same world at different
  positions spread by up to 4.4e-7 in the impedance (a 12 m strip) and,
  in two-layer soil with an image distance of six segment lengths, by
  1e-5 … 2.6e-5 (audit W2B-01/-02). Now results spread by at most 1e-13
  over local offsets of 0.1 m … 8.8 km and the solve at UTM coordinates
  lies as close to the local one as with the point kernel (3e-11 in Z).
- **Images:** the air image is a mirrored segment and is treated like a
  source segment. The Tagg / Sunde images of a two-layer soil lie at least
  $2(h_1 - z)$ away; they get the treatment of direct pairs — exact near,
  blended, far form with the quadrupole term (since wave 3 of the 2026.11
  work; they were point sources beyond six segment lengths, which left a
  straight strip in two-layer soil up to 5e-4 off and the potential above
  it up to 1 % off). The quadrupole term of an image is tapered off
  between 60 and 120 segment lengths of vertical distance between
  observer and image (`IMAGE_QUADRUPOLE_REACH`; beyond, it is below
  $1/(8\cdot 60^2)$ of the image's point value, the level of the far form's
  remainder at the switch), and images beyond every observer are skipped:
  the image series of a high contrast has hundreds of images, of which few
  are close. On the cross-layer path of
  [`image_2layer`](engines/image_2layer.md#electrodes-that-reach-the-interface)
  the layered correction is the exact image expansion of the two-layer
  Green's function (Sunde) in every layer combination, and its images get
  the same treatment on top of the spectral point values; pairs in
  different layers carry the transmitted resistivity
  $2\rho_1\rho_2/(\rho_1+\rho_2)$, which keeps the matrix reciprocal.

For a straight wire the line kernel is exact at any segment length, in
homogeneous and in two-layer soil (the remaining few $10^{-7}$ are the
$(L/R)^4$ remainder of the far form; a 10 m strip in 1000 / 50 Ω·m with
$h_1$ = 1 m is within 7.7e-7 of its closed-form uniform-current resistance
from 2 m to 0.125 m segments, monotonically, and a rod through the layer
interface changes by less than 1e-6 over the same range); a ring converges
monotonically because its chords approximate the circle. What remains
against the true thin-wire solution is the uniform-current ansatz itself:
for the 3 m rod of benchmark B2 the line kernel sits about +0.9 % above
the thin-wire value at every segment length (generated numbers on the
[validation page](validation.md#kernel-of-the-image-family)). The point
kernel's −1 … −2 % kernel error partly cancels that ansatz error on
impedances, which is why the old numbers looked closer to some references
than they were.

### Field points

After the solve, `FieldResult.potential` evaluates the field with the
kernel of the solve: with the line kernel the exact line potential near a
segment and the point form plus $L_j^2(3c_j^2 - 1)/(24R^3)$ farther away,
blended like the reaction matrix and for the Tagg/Sunde images as well
(sources in the lower layer of a two-layer soil are evaluated in point form
by the spectral kernel).
Every `PointSource` of a line-kernel result carries its unit tangent
(`direction`), so a stored result is re-evaluated with line sources;
results written by 0.15.0 have no direction and are evaluated with the
point kernel. Near a wire the difference is large: on the surface of a
rod discretised at 0.5 m the point form spikes at every segment midpoint
(16 … 177 V around an electrode potential of 32 V), while the line form
is smooth and its surface average equals the electrode potential
(`tests/test_line_kernel.py::test_line_potential_is_equipotential_on_the_wire_surface`;
finding 10.33).

### Cost

The near pairs and the quadrupole far field over all $N^2$ pairs make the
line kernel more expensive than the point kernel. As documented with
`Engine.kernel`, a solve of 16 house grids ($N$ = 1920 segments,
homogeneous soil) took 2.7 times as long with numba installed (it is used
automatically, extra `fast`) and 5.7 times without; the extra time grows
with $N^2$. The blended switch adds the integrals of the pairs between six
and 7.5 segment lengths (+18 … +27 % on those 16 house grids), and in
two-layer soil every image within reach carries the far-field term (16 house
grids at $h_1$ = 5 m: 1.6 times the solve without it, which is 0.9 times the
assembly of the point series). The [performance page](performance.md) puts
this into the context of a whole study. `kernel="point"` remains available
and reproduces the numbers of 0.15.0.

## Where each kernel is used

| Component | Kernels | Default |
|---|---|---|
| `image`, `image_2layer` (and the deprecated names `cim`, `image_nlayer`) | line, point | line |
| `mom`, `mom_sommerfeld` (and `bem`) | point | point |
| `solve_mutual_matrix`, `solve_mutual_field` | line, point | the engine's kernel |
| `FieldResult.potential` | the kernel of the solve (`metadata["kernel"]`) | — |
| `fem` | none (volume mesh) | — |

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
