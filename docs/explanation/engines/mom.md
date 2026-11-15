# `mom` — point-matching method of moments

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | homogeneous or two layers (a `MultiLayerSoil` whose canonical stack has at most two layers) |
    | Green's function | the kernels of the image family: point source plus air image, Tagg / Sunde series, the Sommerfeld kernel of ADR-0007 across the interface |
    | Current model | one pulse current per segment |
    | Testing | delta at the segment midpoints (point matching, collocation) with a point-source near field |
    | Kernel | point only (an explicit `kernel="line"` raises `BackendCapabilityError`) |
    | Status | current; `bem` is a deprecated name that runs it |

`mom` drops the uniform-current ansatz of the image family. Each segment
carries its own leakage current, so the solve resolves how current
distributes along an electrode (end and corner effects), between the
electrodes of a bonded cluster and across a layer interface. Its role in
the family is to **check the current model** of the image family on the
same kernel and discretisation — not to be an accuracy reference for
impedances at the 1 % level, which its point-source near field does not
reach.

## Mathematical and physical context

With the Green's function $G$ of the soil (homogeneous or two-layer, see
[`image`](image.md) and [`image_2layer`](image_2layer.md)) the potential
on the surface of a thin wire is the integral of the leakage current
density $\iota(l)$ (in A/m) along all wires,

$$
\varphi(\mathbf r) = \int_{\text{wires}} G(\mathbf r, \mathbf r')\,\iota(l')\,\mathrm dl' ,
$$

and every electrode of a cluster must sit at the cluster potential. That
is a Fredholm integral equation of the first kind for $\iota$. The method
of moments (Harrington 1968) expands $\iota$ in basis functions and
enforces the equation with test functions. `mom` uses the simplest
choice:

- **basis:** one constant current $I_j$ per segment (pulse functions);
- **testing:** the equation at the segment midpoints (delta functions) —
  point matching, or collocation.

The reaction matrix is

$$
Z_{ij} = \frac{\rho}{4\pi}\,K_{ij},\qquad
K_{ij} =
\begin{cases}
\text{potential at midpoint } i \text{ of a unit point source at midpoint } j, & i \ne j,\\[2pt]
2\ln(L_i/a_i)/L_i + \text{images}, & i = j,
\end{cases}
$$

the off-diagonal with the point kernel of the soil (its images included),
the diagonal with the potential of the uniformly loaded segment on its own
surface (the Galerkin self term for very short segments, $L < 2.99\,a$).
The matrix is assembled directly, entry by entry; up to 0.15.0 it was
extracted by applying the image-family operator to an identity matrix, an
$O(N^3)$ product per image term with bit-identical entries (finding
10.59).

This is **not** a Galerkin scheme. Galerkin testing would average the
potential over each segment and integrate each source along its segment
(the exact line kernel of the image family does exactly that, for a
uniform current). Since review pass 10 the documentation and
`metadata["solver"] = "point_matching"` say so (finding 10.31).

## Network solve

The segment unknowns enter the bordered system

$$
\begin{bmatrix} Z & -C \\ C^{\top} & 0 \end{bmatrix}
\begin{bmatrix} \mathbf I_\text{seg} \\ \boldsymbol\varphi_n \end{bmatrix}
= \begin{bmatrix} \mathbf 0 \\ \mathbf I_\text{in} \end{bmatrix},
$$

plus one row and column per finite conductor branch, as in the
[engine family](index.md#network-equations): every midpoint of a node sits
at the node potential, and the segment currents of a node add up to its
injected current. $Z$ is symmetric; the bordered matrix as assembled is
not, and is solved with a dense LU. Every node with soil contact is an
unknown, also without a source: a source-free cluster floats with zero
net current, so transfer impedances are reciprocal (B8) and passive
networks carry their equalising currents (finding 10.29; up to 0.15.0
passive clusters were dropped from the unknowns and read back as an
unweighted field mean, which broke reciprocity). The node potential is
reported as the mean of its segment potentials, which point matching makes
equal.

`mom` runs the shared pipeline of the image family (discretise, inject,
topology, branches; ADR-0014): two-terminal sources (`return_to`),
voltage sources (one current unknown and one constraint row each),
junction nodes and inductive conductors behave exactly as with `image`.
A concrete-encased wire (`concrete_shell_coefficient_ohm_m` $= C$,
ADR-0012) adds the radial resistance of each segment's shell, $C/\Delta s$,
to the segment's diagonal entry and the drop $C I_k/\Delta s_k$ to its
potential, the potential of the metal (since 2026.11; up to then `mom`
rejected such electrodes). With one owner per segment the image family
solves the same system: the two agree to $7\cdot10^{-16}$ with shells, in
homogeneous, two-layer and cross-layer soil, and a ring, whose current
stays uniform, rises by exactly $C/(2\pi r)$ (to $10^{-14}$; see
[concrete shells](../concrete-shells.md#in-the-moment-method)).

## Accuracy of the point-matching scheme

The point-source near field underestimates the coupling of neighbouring
segments of the same wire: the exact potential of a collinear neighbour at
the midpoint is $\ln 3/\Delta s$ per unit current, the point source gives
$1/\Delta s$ (−9 %). The self image of a shallow segment is a point image,
too. Both errors scale with the near-field terms themselves, so `mom` does
not converge to the thin-wire limit under refinement: the B2 rod lies
below the independent thin-wire Galerkin solution and the gap grows as
the segments get shorter (generated table below). Consequences:

- `mom` against `image` with `kernel="point"` isolates the effect of the
  current model: both use the same kernel, so their difference is the
  ansatz (validation: B6, B10, B13 and the
  [field quantities](../validation.md#field-quantities-near-the-electrode)).
- `mom` against `image` with the default line kernel mixes the ansatz with
  the 1–2 % kernel difference.
- Absolute impedances at the 1 % level are better taken from the image
  family with the line kernel where the ansatz holds (rings, rods,
  compact electrodes), or from a convergence study.

`ShallowSegmentWarning` is emitted when a segment is longer than four
times its depth, because the point self image is then biased high.

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Soil | one or two distinct layers | `BackendCapabilityError` for three distinct layers (use `mom_sommerfeld`) |
| Kernel | point | `BackendCapabilityError` for an explicit `kernel="line"` |
| Concrete shells | thin shell per segment, $C/\Delta s$ on the diagonal (ADR-0012) | — |
| Near-interface horizontal wires | distance to $h_1$, on either side, of at least 0.6 segment lengths (point form of the interface image, see [`image_2layer`](image_2layer.md#electrodes-that-reach-the-interface)) | `PointKernelInterfaceWarning` within 0.6 segment lengths, `BackendCapabilityError` within 0.387 |
| Rod ends at a layer interface | head and tip more than one wire radius from an interface (the thin wire has no end face: +21 % with the tip at $h_1$) | `InterfaceContactWarning` |
| Horizontal wires next to an interface | the wire axis outside the reach of a contrasting interface (3.5 wire radii over a layer 33 times more conductive; the solid wire's current crowds to its near side: +3.6 % at 2 radii) | `InterfaceContactWarning` |
| Size | dense $N \times N$ matrix and one LU per inductive frequency, $O(N^3)$ | memory and time, see [performance](../performance.md) |
| Frequency | quasi-static, $f < 1$ kHz | — |

## Validation

<!-- generated:begin mom -->
Generated from the benchmark run measured on 2026-11-14 with groundfield 2026.11.0 (Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, numba installed: yes; Linux x86_64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

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
