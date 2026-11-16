# `mom` — point-matching method of moments

This page explains the engine that solves for one current per segment:
the integral equation behind it, the point-matching scheme, why it is the
check of the image family's current model rather than an accuracy
reference, and its limits. Read it when you use `mom` for touch and step
voltages near grids and stars, or to cross-check `image` and
`image_2layer`.

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | homogeneous or two layers (a `MultiLayerSoil` whose canonical stack has at most two layers) |
    | Green's function | the kernels of the image family: point source plus air image, Tagg / Sunde series, the Sommerfeld kernel of ADR-0007 across the interface |
    | Current model | one pulse current per segment |
    | Testing | delta at the segment midpoints (point matching, collocation) with a point-source near field |
    | Kernel | point only (an explicit `kernel="line"` raises `BackendCapabilityError`) |
    | Accuracy | resolves the current distribution; absolute impedances 0.2 … 2 % low, drifting under refinement ([Numerics](#numerics)) |
    | Status | current; `bem` is a deprecated name that runs it |

## What it solves

`mom` solves the same physical problem as the image family — the
quasi-static current field of thin-wire electrodes in homogeneous or
two-layer soil, see [`image`](image.md#what-it-solves) and
[`image_2layer`](image_2layer.md#what-it-solves) — but drops the
uniform-current ansatz. Each segment carries its own leakage current, so
the solve resolves how current distributes along an electrode (end and
corner effects), between the electrodes of a bonded cluster and across a
layer interface.

Its role in the family is to **check the current model** of the image
family on the same kernel and discretisation. It is not an accuracy
reference for impedances at the 1 % level, which its point-source near
field does not reach.

## Mathematical and physical context

### The integral equation

With the Green's function $G(\mathbf r, \mathbf r')$ of the soil (the
potential at $\mathbf r$ per ampere injected at $\mathbf r'$, homogeneous
or two-layer, see [`image`](image.md) and
[`image_2layer`](image_2layer.md)) the potential on the surface of a thin
wire is the integral of the leakage current density $\iota(l)$ (in A/m,
$l$ the arc length) along all wires,

$$
\varphi(\mathbf r) = \int_{\text{wires}} G(\mathbf r, \mathbf r')\,\iota(l')\,\mathrm dl' .
$$

Every point of an electrode must sit at the potential of its cluster, so
the left-hand side is known on the wires and $\iota$ is the unknown: a
Fredholm integral equation of the first kind. Physically, the current
distributes itself so that the metal is an equipotential.

### Method of moments with point matching

The method of moments (Harrington 1968) expands $\iota$ in basis functions
and enforces the equation with test functions. `mom` uses the simplest
choice:

- **basis:** one constant current $I_j$ per segment $j$ (pulse functions);
- **testing:** the equation at the segment midpoints (delta functions) —
  point matching, or collocation.

The reaction matrix, the potential at midpoint $i$ per ampere on segment
$j$, is

$$
Z_{ij} = \frac{\rho}{4\pi}\,K_{ij},\qquad
K_{ij} =
\begin{cases}
\text{potential at midpoint } i \text{ of a unit point source at midpoint } j, & i \ne j,\\[2pt]
2\ln(L_i/a_i)/L_i + \text{images}, & i = j,
\end{cases}
$$

with $L_i$ the length and $a_i$ the wire radius of segment $i$. The
off-diagonal uses the point kernel of the soil (its images included); the
diagonal uses the potential of the uniformly loaded segment on its own
surface (the Galerkin self term for very short segments, $L < 2.99\,a$).
The matrix is assembled directly, entry by entry.

This is **not** a Galerkin scheme. Galerkin testing would average the
potential over each segment and integrate each source along its segment
(the exact line kernel of the image family does exactly that, for a
uniform current). `metadata["solver"] = "point_matching"` says so.

### Network solve

The segment currents $\mathbf I_\text{seg}$ and the node potentials
$\boldsymbol\varphi_n$ enter the bordered system

$$
\begin{bmatrix} Z & -C \\ C^{\top} & 0 \end{bmatrix}
\begin{bmatrix} \mathbf I_\text{seg} \\ \boldsymbol\varphi_n \end{bmatrix}
= \begin{bmatrix} \mathbf 0 \\ \mathbf I_\text{in} \end{bmatrix},
$$

with the incidence matrix $C$ ($C_{in} = 1$ if segment $i$ belongs to node
$n$) and the injected node currents $\mathbf I_\text{in}$, plus one row and
column per finite conductor branch, as in the
[engine family](index.md#network-equations). The first block row says that
every midpoint of a node sits at the node potential, the second that the
segment currents of a node add up to its injected current. $Z$ is
symmetric; the bordered matrix as assembled is not, and is solved with a
dense LU.

Every node with soil contact is an unknown, also without a source: a
source-free cluster floats with zero net current, so transfer impedances
are reciprocal (B8) and passive networks carry their equalising currents.
The node potential is reported as the mean of its segment potentials,
which point matching makes equal.

`mom` runs the shared pipeline of the image family (discretise, inject,
topology, branches; ADR-0014): two-terminal sources (`return_to`), voltage
sources (one current unknown and one constraint row each), junction nodes
and inductive conductors behave exactly as with `image`.

### Concrete-encased wires

A concrete-encased wire (`concrete_shell_coefficient_ohm_m` $= C$, in
Ω·m; ADR-0012) adds the radial resistance of each segment's shell,
$C/\Delta s_k$ for a segment of length $\Delta s_k$, to the segment's
diagonal entry, and the drop $C I_k/\Delta s_k$ to its potential: the
reported potential is that of the metal. With one owner per segment the
image family solves the same system: the two agree to $7\cdot10^{-16}$
with shells, in homogeneous, two-layer and cross-layer soil, and a ring,
whose current stays uniform, rises by exactly $C/(2\pi r)$ (to
$10^{-14}$; see [concrete shells](../concrete-shells.md#in-the-moment-method)).

## Numerics

### Accuracy of the point-matching scheme

The point-source near field underestimates the coupling of neighbouring
segments of the same wire. For a collinear neighbour of length $\Delta s$,
the exact potential at the midpoint is the line integral
$\frac{1}{\Delta s}\int_{\Delta s/2}^{3\Delta s/2}\mathrm dx/x = \ln 3/\Delta s$
per unit current; the point source gives $1/\Delta s$ (−9 %). The self
image of a shallow segment is a point image, too. Both errors scale with
the near-field terms themselves, so `mom` does not converge to the
thin-wire limit under refinement: the B2 rod lies below the independent
thin-wire Galerkin solution and the gap grows as the segments get shorter
(generated table below). Consequences:

- `mom` against `image` with `kernel="point"` isolates the effect of the
  current model: both use the same kernel, so their difference is the
  ansatz (validation: B6, B10, B13 and the
  [field quantities](../validation.md#field-quantities-near-the-electrode)).
- `mom` against `image` with the default line kernel mixes the ansatz with
  the 1–2 % kernel difference.
- Absolute impedances at the 1 % level are better taken from the image
  family with the line kernel where the ansatz holds (rings, rods, compact
  electrodes), or from a convergence study (`convergence_study`).

`ShallowSegmentWarning` is emitted when a segment is longer than four
times its depth, because the point self image is then biased high.

### Cost

The matrix is dense, $N \times N$ for $N$ segments, and its LU
factorisation costs $O(N^3)$: it dominates for large $N$, and an inductive
world needs one factorisation per frequency. Memory and timings are on
the [performance page](../performance.md).

## Validation

`mom` resolves the current distribution that the image family assumes
uniform, so near-field quantities next to grid corners and star tips are
its strength (the comparison table is on the
[`image`](image.md#field-quantities-and-the-uniform-current-ansatz) page).
Its absolute impedances carry the point-kernel deficit: the B2 rod
1.4 % below the thin-wire limit at the finest segment length, the B3 ring
1.2 % below its closed form. Reciprocity holds to $10^{-8}$.

<!-- generated:begin mom -->
Generated from the benchmark run measured on 2026-11-16 with groundfield 2026.11.0 (Python 3.12.16, NumPy 2.4.4, SciPy 1.17.1, numba installed: yes; Darwin arm64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

| Benchmark | Measure | Bound | Worst measured | Cases | Status | Note |
|---|---|---|---|---|---|---|
| B2 Rod, homogeneous soil | abs(Z/Z_ref - 1) | 2 % | 1.4 % | 3 | pass | point-source near field drifts under refinement (10.31); target 0.5 % with an exact near field |
| B3 Ring, homogeneous soil | abs(Z/Z_ref - 1) | 1.5 % | 1.2 % | 3 | pass | uniform current by symmetry |
| B8 Two-port reciprocity | abs(Z_AB/Z_BA - 1) | 1e-7 | 7.2e-9 | 2 | pass |  |
| B9 Limits of the layered kernels (`equal_rho_integral`) | abs(Z/Z_ref - 1) | 1e-10 | 0 | 3 | pass | image_2layer, mom, mom_sommerfeld, cim, bem |
| B9 Limits of the layered kernels (`layer_limit_integral`) | abs(Z/Z_ref - 1) | 0.1 % | 5.1e-4 | 2 | pass | physical limit |
| B11 Grid refinement | abs(Z(ds)/Z(ds/2) - 1) | 1 % | 0.86 % | 2 | pass |  |
| B13 Ring with an inductive 400 m line (`image_vs_mom`) | compare_engines row deviation (Re and Im) | 5 % | 1.0 % | 1 | pass | inductive branch shared; galvanic part differs by the ansatz |
<!-- generated:end mom -->

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Soil | one or two distinct layers | `BackendCapabilityError` for three distinct layers (use `mom_sommerfeld`) |
| Kernel | point | `BackendCapabilityError` for an explicit `kernel="line"` |
| Concrete shells | thin shell per segment, $C/\Delta s$ on the diagonal (ADR-0012) | — |
| Horizontal wires near $h_1$, point form | at least 0.6 segment lengths from $h_1$, on either side | `PointKernelInterfaceWarning`; within 0.387: `BackendCapabilityError` |
| Rod ends at an interface | head and tip more than one wire radius away (+21 % with the tip at $h_1$) | `InterfaceContactWarning` |
| Horizontal wires next to an interface | axis outside the reach (3.9 wire radii at contrast 33; +3.6 % at 2 radii) | `InterfaceContactWarning` |
| Size | dense $N \times N$ matrix, one LU per inductive frequency, $O(N^3)$ | memory and time ([performance](../performance.md)) |
| Frequency | quasi-static, $f < 1$ kHz | — |

The interface limits are those of every thin-wire engine: the point form
of the interface image needs short segments next to $h_1$, and the thin
wire has no end face and no current crowding to its near side; they are
explained on the [`image_2layer`](image_2layer.md#electrodes-that-reach-the-interface)
page. What to do instead, and the other limits:
known limitations of the [current ansatz and kernels](../../reference/known-limitations.md#current-ansatz-and-source-kernels)
and of the [soil layers](../../reference/known-limitations.md#soil-layers);
the warnings: [warnings reference](../../reference/warnings.md).

## Literature

- Harrington, R. F. (1968). *Field Computation by Moment Methods*.
  Macmillan, New York.
- Meliopoulos, A. P. S., Xia, F., Joy, E. B., & Cokkinides, G. J. (1993).
  An advanced computer model for grounding system analysis. *IEEE
  Transactions on Power Delivery* 8(1).
- Heppe, R. J. (1979). Computation of potential at surface above an
  energized grid or other electrode, allowing for non-uniform current
  distribution. *IEEE Transactions on Power Apparatus and Systems*
  PAS-98(6), 1978–1989.
- Colominas, I., Navarrina, F., & Casteleiro, M. (2007). Numerical
  simulation of transferred potentials in earthing grids considering
  layered soil models. *IEEE Transactions on Power Delivery* 22(3)
  (Galerkin boundary elements, the alternative not implemented here).

Architecture decisions: [ADR-0001](../../adr/0001-two-layer-method.md),
[ADR-0002](../../adr/0002-engine-family.md),
[ADR-0014](../../adr/0014-backend-registry-and-pipeline.md). API:
`groundfield.solver.mom.solve_mom`.

## Audit trail

??? note "Assembly, naming and passive clusters (findings 10.29, 10.31, 10.59)"

    - Up to 0.15.0 the reaction matrix was extracted by applying the
      image-family operator to an identity matrix, an $O(N^3)$ product per
      image term with bit-identical entries; it is now assembled directly
      (finding 10.59).
    - Since review pass 10 the documentation and
      `metadata["solver"] = "point_matching"` call the scheme point
      matching, not Galerkin (finding 10.31).
    - Up to 0.15.0 passive clusters were dropped from the unknowns and read
      back as an unweighted field mean, which broke reciprocity; every node
      with soil contact is now an unknown (finding 10.29).

??? note "Concrete shells (since 2026.11)"

    Up to 2026.11 `mom` rejected concrete-encased electrodes. Since then it
    carries the per-segment shell term of ADR-0012.
