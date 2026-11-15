# `fem` — conforming finite elements, axisymmetric and 3-D (experimental)

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | any number of horizontal layers |
    | Electrodes | every kind, bare or encased (ADR-0012); rods on one vertical axis and rings around it by the axisymmetric model, every other world by the three-dimensional model (the optional `fem3d` extra) |
    | Axisymmetric model | the electrodes as they are (solid cylinders and tori), linear triangles on a graded tensor mesh conforming to every surface and interface; port resistance within about 1e-4 of independent references |
    | 3-D model | thin wires as smoothed line sources with an analytic radius correction, linear tetrahedra on an isotropic bisection mesh, layer interfaces (and resolved concrete) as partially penalised immersed elements; within about 2e-3 of the axisymmetric model |
    | Far field | exact monopole Dirichlet-to-Neumann (Robin) condition on a box 1000 lengths away |
    | Extrapolation | Richardson over two nested levels, error estimate reported |
    | Network | the shared pipeline: conductors (lumped, distributed isolated and galvanic), junctions, `return_to`, voltage sources, inductive branches |
    | Options | `FemOptions`: `model`, `refinement`, `extrapolate`, `r_far_factor`, `far_field`, `max_nodes`, `max_nodes_3d`, `source_radius`, `growth`, `concrete_resistivity` |
    | Status | **experimental** (`ExperimentalBackendWarning` once per process), a reference backend |

`fem` solves the volume form of the current-flow problem: it meshes the
soil around the electrodes and finds the field of the conductors. Since
2026.11 (user decision Q37) it has two models that share nothing with the
integral engines but the network solve. The **axisymmetric model** takes a
rod, a ring and a ring with a central rod — any set of rods on one vertical
axis and rings around it — as the solid bodies they are, in any
horizontally layered soil, to about 1e-4. The **three-dimensional model**
takes every other world — strips, polylines, stars, meshes and grids, rods
and rings side by side, several clusters, galvanic distributed conductors —
with its wires as smoothed line sources, to about 2e-3, and can resolve a
concrete shell as a volume (user decision Q40). `FemOptions.model="auto"`
(default) chooses the axisymmetric model where it applies. That makes
`fem` the independent reference for the image family, `mom` and
`mom_sommerfeld` where they make their approximations: the uniform current
of the image family, the point kernel of `mom`, the thin wire of both next
to an interface, each wire's own concrete shell at the junctions of a
foundation. Up to 2026.11 `fem` reduced every cluster to an equivalent
hemisphere; that reduction and its rules are gone
([ADR-0016](../../adr/0016-conforming-fem.md)).

The sections up to *Validation* describe the axisymmetric model; the
three-dimensional model has its own section
([below](#the-three-dimensional-model)).

## Mathematical and physical context

The stationary current field obeys

$$
-\nabla\cdot\bigl(\sigma(z)\,\nabla\varphi\bigr) = 0
\quad\text{in the soil},\qquad
\frac{\partial\varphi}{\partial z}\Big|_{z=0} = 0,\qquad
\varphi \to 0 \text{ far away},
$$

with $\sigma = 1/\rho$ piecewise constant in depth. For a body of
revolution about a vertical axis the problem reduces to the meridian
half-plane $(s, z)$, $s$ the distance from the axis,

$$
-\frac{1}{s}\frac{\partial}{\partial s}\Bigl(s\,\sigma\,\frac{\partial\varphi}{\partial s}\Bigr)
-\frac{\partial}{\partial z}\Bigl(\sigma\,\frac{\partial\varphi}{\partial z}\Bigr) = 0 .
$$

Every electrode is a perfect conductor at its own potential $U_k$ — a
*port*. A rod is the solid cylinder of its `wire_radius` with flat end
faces, a ring the solid torus of its `wire_radius`; the current density
along and around them follows from the field (more at the rod ends, more on
the side of a ring that faces a conductive layer). The **port conductance
matrix** $G$, $\mathbf I = G\,\mathbf U$, is the earth part of the solve:
symmetric, positive definite, and by the minimum-energy principle the
discrete $G$ of one mesh is larger than the exact one (the resistance
converges from below).

An **encased electrode** (`concrete_shell_coefficient_ohm_m` $= C$,
[concrete shells](../concrete-shells.md)) keeps its metal at $U_k$ and
leaks through its shell surface $r_b$ = `wire_radius`: every metre adds the
radial resistance $C$, applied locally as the surface conductance
$1/(2\pi r_b C)$ between the metal and the soil at each point of the
surface. That is Sunde's cylindrical shell, exact for a shell thin against
the scale on which the leakage current varies; the current may
redistribute, so the shell adds at least $C/L$ (for a ring $C/(2\pi r)$),
as in `mom`.

## Discretisation

- **Graded tensor mesh.** Vertical lines $s = s_i$ and horizontal lines
  $z = z_j$ through every feature — the rod surfaces and end faces, the
  layer interfaces, the soil surface, the axis and the truncation box —
  graded away from each feature, $h(x) = \min_f (h_f + 0.2\,|x - x_f|)$.
  The cells inside a rod are removed, so the rod is a rectangle of mesh
  lines. The edges of its end faces are re-entrant corners of the
  conductor; with a spacing of $0.002\,a$ there the convergence stays of
  second order, also for a rod tip on an interface. Each cell is split into
  two linear triangles; the weight $2\pi s$ is integrated exactly.
- **O-grid for the ring wire.** A square block of up to three wire radii
  around the wire's cross section is filled by an O-grid that conforms to
  the tensor lines on its boundary and is inscribed in the wire's circle
  (32 chords at level 0). The block keeps clear of the soil surface, the
  axis, the other electrodes and of every interface more than $1.05\,a$
  from the wire axis; a nearer interface — a ring on it or within its
  radius — is cut into the block's elements, so every element lies in one
  layer.
- **Truncation.** A box at `r_far_factor` = 1000 characteristic lengths
  (the electrodes' depth and radius, or the depth of the deepest interface
  times the layer contrast $\max\rho/\min\rho$ if larger, since a
  resistive bottom keeps the current in the layers above it over that
  distance) with the Robin condition
  $\partial_n\varphi = -(\mathbf n\cdot\mathbf r/r^2)\,\varphi$, the exact
  Dirichlet-to-Neumann map of the monopole far field on any boundary; the
  graded mesh reaches it with a few dozen extra lines.
- **Ports.** The surface nodes of a bare electrode are one unknown; $G$ is
  the Schur complement of the stiffness matrix onto the ports, one sparse
  LU factorisation for all of them.
- **Refinement and extrapolation.** Level $\ell$ bisects every interval of
  the level-0 lines $\ell$ times (and doubles the O-grid), so the tensor
  part of the spaces is nested and the resistance of one level rises
  monotonically to the limit. By default `fem` solves levels 0 and 1 and
  extrapolates $G$ with the observed second order (Richardson,
  $G = G_1 + (G_1 - G_0)/3$); `metadata["fem_error_estimate"]` is the
  largest relative difference of a port's resistance between the
  extrapolated and the finer level — a bound of the finer level's error,
  and some ten to a hundred times the error of the extrapolated value.
  `FemOptions(refinement=k)` starts at level $k$; `extrapolate=False`
  solves one level (a lower bound of the resistance).

Convergence (one level without extrapolation; deviation from the
independent reference; time on an M1 Pro core with other jobs running):

| World | level 0 | level 1 | level 2 | level 3 | observed order | default (0/1 extrapolated) |
|---|---|---|---|---|---|---|
| ring $r$ = 2 m at 0.8 m, $a$ = 8 mm, 100 Ω·m (closed form) | −0.63 % | −0.16 % | −0.043 % | −0.013 % | 1.97 / 2.00 | −5.7e-5 |
| rod 0.5 … 3.5 m, $a$ = 8 mm, 100 Ω·m (solid-rod FV) | −0.42 % | −0.11 % | −0.027 % | −0.0069 % | 1.98 / 1.99 | −1.4e-5 |
| the rod through $h_1$ = 2 m, 30 / 1000 Ω·m (FV) | −0.40 % | −0.10 % | −0.025 % | −0.0061 % | 1.98 / 1.99 | −1.0e-5 |
| rod 0.5 … 5 m with its tip on $h_1$, 1000 / 30 Ω·m (FV) | −0.43 % | −0.11 % | −0.029 % | −0.0075 % | 1.96 / 1.95 | −4.1e-5 |
| ring ($a$ = 5 mm) on $h_1$ = 2 m, 30 / 1000 Ω·m (Hankel ring) | −0.51 % | −0.13 % | −0.026 % | −0.0014 % | 1.97 / 2.00 | +4.7e-5 |
| the ring 1 cm above $h_1$, 1000 / 30 Ω·m (Hankel ring + proximity) | −0.69 % | −0.18 % | −0.049 % | −0.015 % | 1.98 / 1.96 | −7.0e-5 |
| ring in three layers 100 / 1000 / 30 Ω·m (Hankel ring) | −0.48 % | −0.12 % | −0.034 % | −0.011 % | 1.97 / 2.00 | −5.2e-5 |

Mesh size and cost: a ring has 7 341 / 28 934 nodes at levels 0 / 1, a
3 m rod 18 868 / 74 871, a ring with a central rod in two layers 56 079 /
223 112; the default solve takes 0.20 s, 0.58 s and 2.0 s (best of three),
`refinement=1` 12 s for the last, level 3 of a rod through an interface
2.4 million nodes and 36 s. The error estimate of the finer level
($1.0 \ldots 1.7 \cdot 10^{-3}$ in these cases) matches its actual error
within 5 %. The truncation is converged: the box at 30, 100, 300 … 10 000
characteristic lengths moves a rod over 30 / 10 000 Ω·m by at most 4e-5
(5e-6 from 100 on), a grounded box at the default size by 7e-5.

## Network, sources and results

The soil model hands an `EarthPorts` record (port conductance matrix,
display segments with their leakage per port volt) to the network solve of
the integral backends ([ADR-0014](../../adr/0014-backend-registry-and-pipeline.md)),
with $Z = G^{-1}$ as the multi-port matrix. Lumped and isolated distributed
conductors, inductive branches with their earth-return models
(`earth_inductive_model`), the skin effect, junctions, `return_to` and
voltage sources with their internal impedance are honoured exactly as
there, at every frequency of the engine; the soil part is quasi-static and
the same at every frequency. Floating electrodes take their transfer
potential. The result reports every electrode's potential and current, the
port resistance matrix (`metadata["fem_port_resistance"]`, ports in world
order `metadata["fem_ports"]`, and per level), the mesh size and time per
level (`fem_mesh`) and the box size (`fem_r_far`). Its point sources are
the display segments of the engine's `segment_length` (rods cut at the
interfaces) with the model's leakage distribution, so
`FieldResult.potential`, the touch-voltage helpers and the exports work as
for the other backends (homogeneous and two-layer soil); the field of
those point sources is the exact field of the model's leakage, and the
nodal field of the mesh converges to it (at most 0.96 %, 0.21 %, 0.062 %
at levels 0, 1, 2 along the surface 1 … 100 m from a 3 m rod). Next to a
rod the equipotential current distribution differs from the uniform one of
the image family: the surface potential 1 … 3 m from the rod lies
0.08 … 0.36 % below that of `image`, the touch voltage at 1 m (per ampere)
1.4 % below. The solve runs the
coordinate check of user decision D5 (`MixedCoordinatesWarning`) and
solves projected coordinates in a local frame.

## Guards

- **The 3-D model instead** (`FemOptions.model="auto"`): an electrode
  other than a rod or a ring; a rod or a ring off the common axis (to 1e-9
  of the world's size); a galvanic distributed conductor (it leaks along
  its route); rods that touch or overlap on the axis, a ring that touches a
  rod or another ring, a ring wire that reaches within 5 % of its radius to
  the soil surface (the solid torus must fit). With
  `FemOptions.model="axisymmetric"` these worlds are refused
  (`BackendCapabilityError`, naming the backends whose own checks accept
  the world).
- **Error** (`ValueError`, before anything else): an electrode, driven or
  passive, at or above the soil surface, in the surface plane
  ($2z \le a$) or reaching above it, and a galvanic distributed conductor
  out of the soil — what every backend refuses (audit D2A-06, R2A-07). A
  rod with its head at $z = 0$ is the ordinary driven rod.
- **Error** (`ResourceLimitError`): the node count of the finer mesh,
  predicted from its lines before anything is assembled, above
  `FemOptions.max_nodes` (4 000 000; level 3 of a rod has about 2.4
  million). The display segments of the point sources count against
  `Engine.max_segments` like the segments of every backend (a unit error
  of `segment_length` fails before anything is allocated).
- The remedy of every refusal comes from the other backends' own checks:
  a backend that a setting of the engine would let solve the world is
  named with that setting (a larger `Engine.max_segments`, a finer
  `segment_length`), and if none is left the message says that no backend
  solves the world yet (audit D2A-06, R2A-03, R2A-04).

## Validity range

| Condition | Limit | Behaviour outside |
|---|---|---|
| Electrodes | axisymmetric model: rods on one vertical axis, horizontal rings centred on it (any number, not touching) | the 3-D model (`model="auto"`), or `BackendCapabilityError` with `model="axisymmetric"` |
| Ring wire | axisymmetric model: its top deeper than 5 % of the wire radius | the 3-D model |
| 3-D model | wires whose disc of radius `source_radius` lies in the soil and in one layer; the `fem3d` extra | `BackendCapabilityError` / `ImportError` |
| Soil | any number of horizontal layers, any contrast | — |
| Concrete shells | per-length coefficient on rods and rings | — |
| Conductors | lumped, inductive, skin effect, isolated distributed chains; galvanic distributed chains in the 3-D model | — |
| Concrete | per-wire coefficient (both models); resolved prisms in the 3-D model (axis-aligned straight wires) | `BackendCapabilityError` for other wires |
| Accuracy | axisymmetric: port resistance about 1e-4 at the default; 3-D: about 2e-3, bounded by the error estimate | raise `FemOptions.refinement` |
| Mesh size | `FemOptions.max_nodes` (axisymmetric), `max_nodes_3d` (3-D) | `ResourceLimitError` |
| Frequency | quasi-static soil; conductor branches per frequency | — |

## Validation

Every number is checked against a second method that shares no code with
`fem` (references in `tests/benchmarks/references.py`, tests in
`tests/test_fem.py`): Weber's disc, the thin ring in closed form, a
Hankel-transform thin ring for any number of layers, the exact 2-D
proximity correction of a wire next to an interface, and a finite-volume
model of the solid rod (cell-centred, its own grid, adapted from audit
A3). The integral engines follow within their documented biases: the
uniform current of the image family lies about 1 % above a rod's
equipotential solution, the point kernel of `mom` 0.2 … 2 % below.

| World | `fem` | second method: fem / reference − 1 | integral engines: engine / fem − 1 |
|---|---|---|---|
| Weber's disc $b$ = 1 m, 100 Ω·m | 24.9923 Ω | $\rho/(4b)$ = 25 Ω: −3.1e-4 | — |
| rod 0 … 3 m, $a$ = 8 mm, 100 Ω·m (B2) | 33.1933 Ω | thin-wire Galerkin limit 33.1957 Ω: −7.3e-5 | `image` (line) +0.93 %, `mom` (0.25 m) −1.1 % |
| rod 0.5 … 3.5 m / 0.5 … 5 m | 31.7170 / 22.7906 Ω | solid-rod FV: −1.4e-5 / −1.4e-5 | `image` +1.1 / +0.91 %, `mom` −0.59 / −0.91 % |
| rings $r$ = 2, 5, 10 m (B3) | 12.4809 … 3.4417 Ω | closed form: −2.7e-5 … −5.7e-5 | `image` +3.0e-5 … +5.0e-5, `mom` −0.85 … −1.2 % |
| ring with a central rod, bonded, 100 Ω·m | 11.5338 Ω | — (port matrix symmetric) | `image` +0.41 %, `mom` −0.87 % |
| rings in two layers (8 soils, 30 / 10 000 … 10 000 / 30 Ω·m) | | Hankel ring: −2.6e-5 … −7.3e-5 | `image_2layer` +3.0e-5 … +5.9e-5 (+3.4e-4 for 10 000 / 30, −1.0 % for 30 / 10 000 Ω·m, whose series it truncates), `mom` −0.43 … −1.8 % |
| rings in three layers (5 stacks, also below $h_1$) | | Hankel ring: −3.4e-5 … −9.1e-5 | `mom_sommerfeld` −0.45 … −2.1 % (top layer only) |
| rods in two layers (6 cases, through and within a layer) | | solid-rod FV: −1.4e-5 … +2.0e-4 | `image_2layer` +0.07 … +1.4 %, `mom` −1.2 … +0.5 % |
| ring 1 cm above an interface, $r$ = 2 / 5 m at $t = r \ldots D$ (12 cases) | | Hankel ring + 2-D proximity: −1.3e-5 … −7.0e-5 | `image_2layer` +5.2 … +5.4 % (1000 / 30 Ω·m), +0.07 … +0.10 % (30 / 1000) |
| the same rings 10 cm above the interface (12 cases) | | −1.9e-5 … −6.3e-5 | `image_2layer` −0.004 … +0.07 % |
| 3 m rod with its head 1 mm below … 8 cm into a 30 Ω·m top layer (14 cases) | | solid-rod FV: within ±1.3e-4 | `image_2layer` jumps by a factor 3.8 when its first segment enters the layer (30 / 10 000 Ω·m) |
| ring exactly on an interface (4 soils) | | Hankel ring: −1.1e-4 … +4.7e-5 | `image_2layer` jumps by 0.03 … 0.38 % between the two sides |
| rod tip 16 mm above … 16 mm below an interface (1000 / 30 Ω·m) | | solid-rod FV: −4.1e-5 … +1.3e-5 | thin wire (`image_2layer`) +22 % with the tip on the interface, +7.2 % 1 mm above, −4.4 % 1 mm below |
| encased ring $r$ = 2 m at 0.5 / 0.8 / 3 m ($r_b$ = 55 mm, $C$ = 38.16 Ω·m) | rise 1.0057 / 1.0040 / 1.0021 $C/(2\pi r)$ | the bound $\ge C/(2\pi r)$ | the image family adds exactly $C/(2\pi r)$ |
| encased 3 m rod, head at 0 / 0.5 m | rise 1.0086 / 1.0005 $C/L$ | Sunde's decomposition with Dwight's rod (uniform current): −0.96 % / −5.2 % | `image` +1.1 / +1.6 %, `mom` −0.84 / −0.21 % |

(Evidence of stream w4axi, `e01`–`e04`.)

The generated table below compares `fem` with the benchmark references of
the suite; in B6 and B7 the reference is `mom` on the same segments, so
the deviation is the deficit of `mom`'s point kernel.

<!-- generated:begin fem -->
Generated from the benchmark run measured on 2026-11-14 with groundfield 2026.11.0 (Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, numba installed: yes; Linux x86_64); bounds from `tests/benchmarks/tolerances.py`. Details and all cases: [Validation and benchmarks](../validation.md).

| Benchmark | Measure | Bound | Worst measured | Cases | Status | Note |
|---|---|---|---|---|---|---|
| B1 Surface disc | abs(Z/Z_ref - 1) | 5e-4 | 3.1e-4 | 1 | pass | observed order of convergence 1.87; discretisation of the conforming tensor mesh alone (user decision Q37; up to 2026.11 the hemisphere of the equivalent-hemisphere mesh) |
| B2 Rod, homogeneous soil | abs(Z/Z_ref - 1) | 0.1 % | 7.3e-5 | 3 | pass | solid rod with flat ends (user decision Q37); the thin-wire limit itself is uncertain at a few 1e-4 (33.185 Ω at 384 segments) |
| B3 Ring, homogeneous soil | abs(Z/Z_ref - 1) | 5e-4 | 5.7e-5 | 3 | pass | solid torus (user decision Q37) |
| B6 Top-layer electrodes, two-layer soil | abs(Z/Z_ref - 1) | 3 % | 1.3 % | 10 | pass | conforming model (user decision Q37): the deficit of mom's point kernel (10.28) |
| B7 Rod through the layer interface | abs(Z/Z_ref - 1) | 3 % | 1.1 % | 2 | pass | conforming model (user decision Q37; up to 2026.11 it raised, v0.15.0: +1092 %, silent): the deficit of mom's point kernel |
| B8 Two-port reciprocity | abs(Z_AB/Z_BA - 1) | excluded | — | — | excluded | refused; excluded: rod and ring on different axes, which the axisymmetric model refuses (user decision Q37); the port matrix of coaxial electrodes is symmetric to round-off |
| B9 Limits of the layered kernels (`equal_rho_fem`) | abs(Z/Z_ref - 1) | 1e-12 | 0 | 3 | pass | the conforming model meshes the canonical stack (user decision Q37) |
| B9 Limits of the layered kernels (`layer_limit_fem`) | abs(Z/Z_ref - 1) | 0.1 % | 5.0e-4 | 2 | pass | physical limit (user decision Q37) |
| B10 Ring with three bonded rods | abs(Z/Z_ref - 1) | excluded | — | — | excluded | refused; excluded: rods off the ring's axis, which the axisymmetric model refuses (user decision Q37; up to 2026.11 -35 % / -45 %, radii summed) |
<!-- generated:end fem -->

## The three-dimensional model

The three-dimensional model (part B of user decision Q37) needs the
optional `fem3d` extra (scikit-fem, BSD-3; pyamg, MIT; see
[Install the optional extras](../../how-to/install-extras.md)).

### Wires as smoothed line sources

Every electrode is a chain of straight pieces along its wire axis (a ring a
polygon of 1° chords), every leaking piece of a galvanic distributed
conductor one more. A piece leaks the current density $j(s)$ in A/m,
spread over the disc of radius $R$ (`FemOptions.source_radius`, default
6 cm) perpendicular to the axis with the bump density

$$
\eta(r) = \frac{3}{\pi R^2}\Bigl(1 - \frac{r^2}{R^2}\Bigr)^2 ,
$$

and the wire sees the $\eta$-weighted disc average $\Pi(s)$ of the
potential. Its field is smooth, so the mesh needs no resolution of the
wire radius $a$. What lies inside the disc is added analytically as a
radial resistance per unit length between the metal and that average,

$$
U - \Pi(s) = \bigl(C + \Delta C(s)\bigr)\,j(s),
\qquad
\Delta C = \frac{\rho}{4\pi}\sum_k \bigl[\Phi_k(s; a) - \Phi_k(s; r_\eta)\bigr],
$$

with the shell coefficient $C$ of an encased wire, $\Phi_k(s; c)$ the
integral of the reduced kernel $1/\sqrt{d^2 + c^2}$ over piece $k$ (closed
form), summed over the own piece, its neighbours at corners and junctions
and their images in the soil surface, and $r_\eta = 0.5442\,R$ the energy
radius of the bump. Far inside a straight wire this is Sunde's radial
resistance $\frac{\rho}{2\pi}\ln(r_\eta/a)$ of the soil between the wire and
the disc; near an end, a corner or a junction it carries the 3-D geometry.

### Mesh, layers and concrete

- **Isotropic mesh.** A grid of $8\times8\times4$ cubes on the truncation
  box, six Kuhn tetrahedra each, refined by a deterministic conforming
  longest-edge bisection until every element is below
  $1.5\,(R/2 + g\max(0, d - R))$, $d$ the distance from the nearest wire
  and $g$ = `FemOptions.growth` (0.5). The elements keep an aspect ratio of
  at most 2. (A graded tensor grid carries its fine lines to the box as
  slabs; the multigrid stalls on them.)
- **Layer interfaces** are no mesh planes. An element that an interface
  cuts is an **immersed element**: its shape functions are linear on either
  side of the plane, continuous across it and carry the flux
  $\sigma\,\partial_z\varphi$ across it; the faces the interface crosses get
  the consistency and penalty terms of the partially penalised method (Lin,
  Lin & Zhang 2015), so a layered field is reproduced exactly. Nodes within
  30 % of their shortest edge of an interface are moved onto it. Elements
  four times larger than the layer they cut (far from the electrodes) take
  the laminate conductivity.
- **Concrete shells.** By default every encased wire carries its own
  coefficient $C$ — the model of the image family and `mom`, so `fem`
  compares with them directly. With `FemOptions.concrete_resistivity`
  $= \rho_c$ the concrete is a volume: a square prism of conformal radius
  $r_b$ = `wire_radius` (side $r_b/0.59017$, the same resistance to the
  soil as a circular shell) around every straight encased wire along x, y
  or z, the metal radius following from
  $C = \frac{\rho_c}{2\pi}\ln(r_b/r_a)$. Where wires meet, the prisms share
  their concrete (user decision Q40, see
  [Concrete shells](../concrete-shells.md)).

### Solve, levels and cost

The disc averages couple all nodes of a disc, so the coupling is never
formed: conjugate gradients run on the operator, preconditioned by the
smoothed-aggregation multigrid of the soil stiffness (pyamg), one solve per
port (16 … 115 iterations). The finer level is the red refinement of the
coarser one; the port conductance matrix is extrapolated with the second
order and the relative difference to the finer level is
`metadata["fem_error_estimate"]` (3 … 5e-3 at the defaults). The levels
are not asymptotic for every geometry (error ratios between the levels of
3.0 … 5.2), so the extrapolated value carries up to about 2e-3; the
estimate bounds the deviation from the axisymmetric model in every case
tested. A ring of 12.6 m takes about a minute and 2 GB, the AP1 station
(28 m of wire, 1.3 million nodes at the finer level) about two minutes and
4 GB. `FemOptions.max_nodes_3d` (2 million nodes of the finer mesh, about
3 GB per million) refuses larger meshes before they are built
(`ResourceLimitError`).

### Guards of the 3-D model

- `ImportError` naming the `fem3d` extra if it is missing.
- `BackendCapabilityError`: a wire so close to the soil surface or a layer
  interface that its disc cannot lie in the soil and in one layer; resolved
  concrete around a wire that is not straight along x, y or z (rings,
  oblique strips) or whose concrete would reach the surface.
- `ValueError`: a `source_radius` that does not reach beyond the bare
  wire (it must be at least 2.2 wire radii).

### Validation of the 3-D model

Two methods each (evidence of stream w4fem3d): the axisymmetric model where
the world allows it, closed forms and the integral engines otherwise.

**Against the axisymmetric model** (defaults of the 3-D model; levels 0/1 extrapolated):

| World | axisymmetric model | 3-D model / axisymmetric − 1 | error estimate | finer-level nodes | thin-wire engine / axisymmetric − 1 | `mom` / axisymmetric − 1 |
|---|---|---|---|---|---|---|
| rod 0 … 3 m, $a$ = 8 mm, 100 Ω·m | 33.19327 Ω | −1.5e-4 | 4.6e-3 | 186 903 | `image` +0.92 % | −1.45 % |
| rod 0.5 … 3.5 m, 100 Ω·m | 31.71703 Ω | +6.5e-4 | 4.8e-3 | 203 383 | `image` +1.11 % | −1.22 % |
| ring $r$ = 2 m at 0.8 m, $a$ = 8 mm, 100 Ω·m | 12.48090 Ω | −1.95e-3 | 4.7e-3 | 600 179 | `image` +0.003 % | −1.17 % |
| ring $r$ = 5 m at 0.5 m, $a$ = 5 mm, 100 Ω·m | 6.41819 Ω | −1.93e-3 | 4.1e-3 | 1 500 815 | `image` +0.003 % | −0.91 % |
| rod 0 … 3 m through $h_1$ = 2 m, 100 / 1000 Ω·m | 61.22214 Ω | −1.30e-3 | 4.3e-3 | 194 897 | `image_2layer` +0.08 % | −1.32 % |
| rod 0 … 3 m above $h_1$ = 4 m, 1000 / 30 Ω·m | 300.66258 Ω | +2.1e-4 | 5.2e-3 | 187 027 | `image_2layer` +1.20 % | −1.59 % |
| rod 0 … 3 m above $h_1$ = 5 m, 100 / 1000 Ω·m | 38.84736 Ω | −3.1e-4 | 4.5e-3 | 195 663 | `image_2layer` +0.74 % | −1.25 % |
| ring $r$ = 2 m at 0.8 m above $h_1$ = 2 m, 1000 / 30 Ω·m | 80.71707 Ω | −2.00e-3 | 4.3e-3 | 595 993 | `image_2layer` +0.004 % | −1.81 % |
| ring $r$ = 2 m at 0.8 m, three layers 100 / 1000 / 30 Ω·m (1 m, 2 m) | 21.41013 Ω | −9.2e-4 | 4.0e-3 | 641 613 | `mom_sommerfeld` −0.68 % | refused |

Every case lies within 2.0e-3 of the axisymmetric model and inside its own error estimate. The free ends of a wire read slightly high, a wire without ends (a ring) about 2e-3 low: the two mesh levels are not yet in the asymptotic range for every geometry (error ratio between the levels 3.0 for rings, 4.2 … 5.2 for rods).

**AP1 station** in the seven corner soils of the AP1 grid: square ring
4 m × 4 m at 0.6 m (5 mm wire), three 4 m rods (8 mm) from three corners,
bonded; input impedance at DC. The 3-D model lies between the two thin-wire
engines, as the axisymmetric model does for rods and rings: the uniform
current per wire of `image_2layer` reads high, the point kernel of `mom`
low.

| $\rho_1$ / $\rho_2$ / $h_1$ (Ω·m, m) | `fem` (3-D) | error estimate | `image_2layer` / `fem` − 1 | `mom` (0.25 m) / `fem` − 1 |
|---|---|---|---|---|
| 1000 / 200 / 30 | 72.328 Ω | 4.3e-3 | +1.52 % | −0.63 % |
| 1000 / 200 / 5 | 57.962 Ω | 4.6e-3 | +2.44 % | −0.80 % |
| 1000 / 30 / 30 | 71.525 Ω | 4.8e-3 | +1.52 % | −0.65 % |
| 1000 / 30 / 5 | 52.747 Ω | 4.8e-3 | +2.98 % | −0.86 % |
| 30 / 200 / 30 | 2.4652 Ω | 4.5e-3 | +1.33 % | −0.56 % |
| 30 / 200 / 5 | 3.5625 Ω | 4.8e-3 | +0.66 % | −0.38 % |
| 30 (homogeneous) | 2.2510 Ω | 4.3e-3 | +1.47 % (`image`) | −0.60 % |

**Two clusters.** Two 2 m rods ($a$ = 8 mm) in 100 Ω·m: the port matrix is
symmetric to round-off ($|R_{12}/R_{21} - 1| \le 1.1\cdot10^{-16}$), and the
transfer resistance tends to the point-source value, $R_{12} = 0.949$ /
$0.994\,\rho/(2\pi d)$ at $d$ = 5 / 20 m. The loop impedance
$R_{11} + R_{22} - 2R_{12}$ of `image` lies +1.12 / +1.11 % above, that of `mom`
(0.1 m) −1.51 / −1.42 % below. **A conductor between two clusters**
(two 2 m rods 8 m apart, heads at 0.6 m, 50 mm² copper at 0.6 m in 0.5 m
pieces): isolated, `image` +1.18 % and `mom` −1.11 %; galvanic (every piece
a port of the 3-D model), `image` +0.57 % and `mom` −1.73 % in the input
impedance of the first rod.

**Encased braces and concrete shells.** The house mesh of the generators
(10 m, 2 × 2 cells, 5 mm bar, 5 cm cover, 100 Ω·m; `source_radius` 0.1 m,
`growth` 0.6, 1.5 million nodes): encasing its braces raises $Z$ by
+49.1 / +318.0 % at $\rho_c$ = 1 000 / 5 000 Ω·m (`mom` +49.6 / +321.3 %), and
the owner model of the image family lies +1.6 / +2.7 % above the 3-D model
with bare braces, +0.14 / +0.03 % with encased ones. On straight bars, an
L, a T-joint and a crossing the per-wire shell of the 3-D model agrees with
`image` to 0.01 … 0.24 % and with `mom` to 0.06 … 0.29 %. **Resolved
concrete** (`concrete_resistivity`, user decision Q40) converges at first
order in its prism faces and edges — it reads the conductance per metre
2.8 … 4.6 % high at the default levels — so it serves differences on one
mesh: against each wire's own shell, every junction of a foundation acts
like at most about one shell radius of wire (a corner −1.5 cm, a crossing
−3 … −6 cm, a T-joint within ±5 cm), and the house mesh changes by less than
0.6 % ([Concrete shells](../concrete-shells.md)).

## Literature

- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission Systems*.
  Dover, New York, ch. 2 and §4.4 (concrete shell).
- Weber, H. (1873). Ueber die Besselschen Functionen und ihre Anwendung
  auf die Theorie der elektrischen Ströme. *J. reine angew. Math.* 75
  (the disc electrode, $R = \rho/(4b)$).
- Givoli, D. (1992). *Numerical Methods for Problems in Infinite Domains*.
  Elsevier (Dirichlet-to-Neumann truncation; the monopole map is its
  lowest mode).
- Strang, G., & Fix, G. (2008). *An Analysis of the Finite Element
  Method*, 2nd ed. Wellesley-Cambridge Press (minimum-energy principle and
  one-sided convergence).
- Richardson, L. F. (1911). The approximate arithmetical solution by finite
  differences of physical problems. *Phil. Trans. R. Soc. A* 210.
- Smythe, W. R. (1968). *Static and Dynamic Electricity*, 3rd ed.
  McGraw-Hill (images in a plane interface and in a circular cylinder).

Architecture decision: [ADR-0016](../../adr/0016-conforming-fem.md) (the
conforming model; [ADR-0002](../../adr/0002-engine-family.md) for the engine
family). API: `groundfield.solver.fem.solve_fem`; `FemOptions`;
`equivalent_hemisphere_radius` (deprecated) and `FemReductionWarning`
(never emitted) are kept for compatibility.
