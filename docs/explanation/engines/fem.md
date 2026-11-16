# `fem` — conforming finite elements, axisymmetric and 3-D (experimental)

!!! abstract "At a glance"

    | | |
    |---|---|
    | Soil | any number of horizontal layers |
    | Electrodes | every kind, bare or encased (ADR-0012); rods on one vertical axis and rings around it by the axisymmetric model, every other world by the three-dimensional model (the optional `fem3d` extra) |
    | Axisymmetric model | the electrodes as they are (solid cylinders and tori), linear triangles on a graded tensor mesh conforming to every surface and interface; port resistance within about 1e-4 of independent references (within 5.1e-5 for a ring wire within 1.05 radii of an interface, up to a layer contrast of 1000) |
    | 3-D model | thin wires as smoothed line sources with an analytic radius correction, linear tetrahedra on an isotropic bisection mesh, layer interfaces (and resolved concrete) as partially penalised immersed elements; within about 2e-3 of the axisymmetric model away from layer interfaces and other wires — it keeps wires that run beside each other at least $1.5\,R$ apart ($R$ = `source_radius`, 6 cm by default; a smaller $R$ for closer wires, `BackendCapabilityError` where none fits); next to a strongly contrasting interface the thin-wire error applies |
    | Far field | exact monopole Dirichlet-to-Neumann (Robin) condition on a box 1000 lengths away |
    | Extrapolation | Richardson over two nested levels, error estimate reported |
    | Network | the shared pipeline: conductors (lumped, distributed isolated and galvanic), junctions, `return_to`, voltage sources, inductive branches |
    | Options | `FemOptions`: `model`, `refinement`, `extrapolate`, `r_far_factor`, `far_field`, `max_nodes`, `max_nodes_3d`, `max_ports_3d`, `source_radius`, `growth`, `concrete_resistivity` |
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
with its wires as smoothed line sources, to about 2e-3 away from layer
interfaces and other wires, and can resolve a concrete shell as a volume
(user decision Q40). `FemOptions.model="auto"`
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
symmetric and positive definite. By the minimum-energy principle the
discrete $G$ of a mesh that holds the electrodes exactly (a rod is a
rectangle of mesh lines) is larger than the exact one, so its resistance
lies below the exact one; a ring's O-grid inscribes a polygon in the
wire's circle, so for rings this is observed, not guaranteed (see
*Refinement and extrapolation*; audit W4A-09).

Mesh features closer than 1e-9 of max(1 m, x) are one line: a run keeps
the coordinate of its physical feature (an interface, a rod end, the soil
surface), so only mesh lines move; an interface is cut at its own line, and
a triangle of zero area is refused as an internal error (audits W4R2A,
W4R3A-01: a ring axis picometres off an interface read +35 … +44 % or
failed).

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
  layer. A ring wire within $1.05\,a$ of an interface gets a finer block:
  the gap between the wire and the interface, and the narrow wedge where
  the interface cuts the wire near its edge, make the field vary over
  $\sqrt{2a\,|d - a|}$ ($0.14\,a$ at $d = 1.01\,a$, $d$ the distance of
  the wire axis from the interface). With the default block the levels
  converged at first order and not monotonically there (up to −6.0e-3 next
  to a layer 33 times more conductive, the error estimate below the error
  in 7 of 14 cases; audit W4A-01). With 64 intervals per side the default
  lay within 3.3e-4 at contrast 33, but not beyond it: touching a 1000
  times more conductive layer it read −1.6e-2 with an estimate of 1.1e-2,
  the levels converged at about first order (audit W4R2A-02). The
  interface cuts the wire at an edge of the conductor on a material
  interface, or touches it in a contact zone that shrinks with the
  contrast. The O-grid of such a block is therefore graded towards the
  points where the interface cuts the wire at a narrow angle (0.7 a or
  farther from the axis) or comes closest to it — the rays geometrically
  in angle, the first layers geometrically in radius, from cells of 1e-3
  wire radii, both bisected with the level — and the block has 32
  intervals per side. The default lies within 5.1e-5 of the auditor's
  boundary elements at contrasts 3 … 1000 on either side, cut or touching
  (60 cases; 3.2e-5 at 100), `refinement=1` within 1.9e-5, the estimate
  (3.9e-4 … 1.5e-3) above the error in all 113 cases of audit W4R2A's
  sweep at both refinements; a ring touching a 1000 times more conductive
  layer takes 38 317 / 151 807 nodes, 1.8 s and 0.76 GB (`refinement=1`:
  604 307 nodes, 11 s, 2.8 GB) — fewer than the 64-interval block
  (evidence `e05`, `e07` of stream fw4r2a). A ring on the interface
  ($d = 0$) reads −1.2e-5 against the same model.
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
  the level-0 lines $\ell$ times (and doubles the O-grid). The tensor part
  of the spaces is nested, the O-grid of a ring is not (its layers move
  with the level, the inscribed polygon grows): one level converges at
  second order and, in every case tested away from interfaces, from below
  — observed, not guaranteed. By default `fem` solves levels 0 and 1 and
  extrapolates $G$ with the observed second order (Richardson,
  $G = G_1 + (G_1 - G_0)/3$); `metadata["fem_error_estimate"]` is the
  largest relative difference of a port's resistance between the
  extrapolated and the finer level — about the finer level's error (0.88
  … 0.99 of it in 36 cases of auditor W4A, 0.94 … 1.07 for a ring that an
  interface cuts or comes within 1.05 radii of, 60 cases of audit W4R2A)
  and 10 … 130 times the error of the
  extrapolated value, which it bounds while both levels lie on the same
  side of the converged value and the finer level's error is at most
  1/2.5 of the coarser one's (1/3.9 observed; 1/3.9 … 1/4.0 for a ring
  touching a 100 or 1000 times more conductive layer). `FemOptions(refinement=k)`
  starts at level $k$; `extrapolate=False` solves one level, usually
  below the converged value, and reports `fem_error_estimate` as NaN
  (audit W4A-02).

Convergence (one level without extrapolation; deviation from the
independent reference; time on an M1 Pro core with other jobs running;
the ring on $h_1$ re-measured with its graded block of audit W4R2A-02,
the script `e05` of stream fw4int on the merged fixes: 24 488 … 1 543 200
nodes at levels 0 … 3, the level 3 in 30 s and 6.4 GB):

| World | level 0 | level 1 | level 2 | level 3 | observed order | default (0/1 extrapolated) |
|---|---|---|---|---|---|---|
| ring $r$ = 2 m at 0.8 m, $a$ = 8 mm, 100 Ω·m (closed form) | −0.63 % | −0.16 % | −0.043 % | −0.013 % | 1.97 / 2.00 | −5.7e-5 |
| rod 0.5 … 3.5 m, $a$ = 8 mm, 100 Ω·m (solid-rod FV) | −0.42 % | −0.11 % | −0.027 % | −0.0069 % | 1.98 / 1.99 | −1.4e-5 |
| the rod through $h_1$ = 2 m, 30 / 1000 Ω·m (FV) | −0.40 % | −0.10 % | −0.025 % | −0.0061 % | 1.98 / 1.99 | −1.0e-5 |
| rod 0.5 … 5 m with its tip on $h_1$, 1000 / 30 Ω·m (FV) | −0.43 % | −0.11 % | −0.029 % | −0.0075 % | 1.96 / 1.95 | −4.1e-5 |
| ring ($a$ = 5 mm) on $h_1$ = 2 m, 30 / 1000 Ω·m (solid-torus boundary elements) | −0.44 % | −0.11 % | −0.028 % | −0.0070 % | 1.98 / 2.00 | −1.1e-5 |
| the ring 1 cm above $h_1$, 1000 / 30 Ω·m (Hankel ring + proximity) | −0.69 % | −0.18 % | −0.049 % | −0.015 % | 1.98 / 1.96 | −7.0e-5 |
| ring in three layers 100 / 1000 / 30 Ω·m (Hankel ring) | −0.48 % | −0.12 % | −0.034 % | −0.011 % | 1.97 / 2.00 | −5.2e-5 |

Mesh size and cost: a ring has 7 341 / 28 934 nodes at levels 0 / 1, a
3 m rod 18 868 / 74 871, a ring with a central rod in two layers 56 079 /
223 112; the default solve takes 0.20 s, 0.58 s and 2.0 s (best of three),
`refinement=1` 12 s for the last, level 3 of a rod through an interface
2.4 million nodes and 36 s (above the default `max_nodes`). The error
estimate of the finer level ($1.0 \ldots 1.7 \cdot 10^{-3}$ in these
cases) is 0.88 … 0.99 of its actual error in the 36 cases of auditor W4A
(0.88 for a rod head on a 30 / 10 000 Ω·m interface). The truncation is
converged: the box at 30, 100, 300 … 10 000 characteristic lengths moves a
rod over 30 / 10 000 Ω·m by at most 4e-5 (5e-6 from 100 on), a grounded
box at the default size by 7e-5.

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
  other than a rod or a ring; a rod or a ring off the common axis; a
  galvanic distributed conductor (it leaks along its route); rods that
  touch or overlap on the axis, a ring that touches a rod or another ring,
  a ring wire that reaches within 5 % of its radius to the soil surface
  (the solid torus must fit), a ring without room for its mesh block
  (within about a wire radius, diagonally, of a rod's end edge; audit
  W4A-03). With `FemOptions.model="axisymmetric"` these worlds are refused
  (`BackendCapabilityError`, naming the backends whose own checks accept
  the world).
- **The common axis** has a tolerance of 1e-3 of the smallest wire radius
  or surface gap between two electrodes in the meridian plane (at least
  1e-12 of the world's size). The resistance is even in the offset
  $\delta$, so the axisymmetric reading is off by $\kappa(\delta/L)^2$:
  $|\kappa| \le$ 6e-4 for $L$ the wire radius and $\le$ 1.1e-2 for $L$ the
  gap (`image` and `mom` on the same discretisations, three worlds), below
  1e-6 at the tolerance; for coaxial rods end to end with a gap $g \le a$
  the overlap of the end faces changes linearly in $|\delta|$ in the
  parallel-plate picture, which with $d\ln Z/d\ln g$ = 0.02 … 0.10
  estimates 6e-6 … 1.3e-5 at the tolerance: below about 1e-5 (an estimate
  that errs high, the offset being at most 1e-3 g; audit W4R2A-06). Up to audit W4A-08 it was 1e-9 of the world's
  size (3 nm): a rod 1 µm off a ring's axis went to the 3-D model (40.7 s,
  2.35 GB, −1.43e-3; evidence `e01` of stream fw4axi).
- **Error** (`ValueError`, before anything else): an electrode, driven or
  passive, at or above the soil surface, in the surface plane
  ($2z \le a$) or reaching above it, and a galvanic distributed conductor
  out of the soil — what every backend refuses (audit D2A-06, R2A-07). A
  rod with its head at $z = 0$ is the ordinary driven rod.
- **Error** (`ResourceLimitError`): the node count of the finer mesh,
  predicted from its lines before anything is assembled, above
  `FemOptions.max_nodes` (2 000 000 nodes, about 8 GB: the axisymmetric
  solve needs about 4 GB per million nodes of the finer mesh — 0.82 /
  3.15 / 5.15 GB at 0.20 / 0.80 / 1.22 million, `/usr/bin/time -l`,
  evidence `e02` of stream fw4axi; level 3 of a rod, 2.4 million, needs it
  raised). The prediction counts the O-grid exactly and errs high
  (+0.7 … +11 % in five worlds; audit W4A-10). The display segments of
  the point sources count against `Engine.max_segments` like the segments
  of every backend (a unit error of `segment_length` fails before anything
  is allocated).
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
| 3-D model | wires whose disc of radius `source_radius` lies in the soil and in one layer; wires that run beside each other at least $1.5\,R$ apart (the model takes a smaller $R$, down to 2.2 bare wire radii); at most `FemOptions.max_ports_3d` ports (100; electrodes and leaking pieces of galvanic conductors); the `fem3d` extra | `BackendCapabilityError` / `ResourceLimitError` before meshing / `ImportError` |
| Soil | any number of horizontal layers; any contrast in the axisymmetric model, up to about 300 tested in the 3-D model | 3-D model: `BackendCapabilityError` when its conjugate gradients do not converge (a rod through a 1 / 1000 Ω·m interface), naming the axisymmetric model and the integral engines |
| Concrete shells | per-length coefficient on rods and rings | — |
| Conductors | lumped, inductive, skin effect, isolated distributed chains; galvanic distributed chains in the 3-D model | — |
| Concrete | per-wire coefficient (both models); resolved prisms in the 3-D model (axis-aligned straight wires) | `BackendCapabilityError` for other wires |
| Accuracy | axisymmetric: port resistance about 1e-4 at the default (within 5.1e-5 for a ring wire within 1.05 radii of an interface, up to a contrast of 1000); 3-D: about 2e-3 away from interfaces and other wires (wires beside each other at least $1.5\,R$ apart), the error estimate a bound while the ratio of the two levels' errors is at least 2.5; next to a strongly contrasting interface the thin-wire error (see *Validation of the 3-D model*) | axisymmetric: raise `FemOptions.refinement`; 3-D: a smaller `source_radius` or `growth` with a larger `max_nodes_3d` (`refinement` multiplies its nodes by eight: the AP1 station would need about 11.5 million), or one of the integral engines as the second method |
| Mesh size | `FemOptions.max_nodes` (axisymmetric), `max_nodes_3d` and `max_ports_3d` (3-D) | `ResourceLimitError` |
| Frequency | quasi-static soil; conductor branches per frequency | — |

## Validation

Every number is checked against a second method that shares no code with
`fem` (references in `tests/benchmarks/references.py`, tests in
`tests/test_fem.py`): Weber's disc, the thin ring in closed form, a
Hankel-transform thin ring for any number of layers, the exact 2-D
proximity correction of a wire next to an interface, a finite-volume
model of the solid rod (cell-centred, its own grid, adapted from audit
A3) and auditor W4A's axisymmetric boundary elements of the solid rods
and tori (audit W4A-11). The integral engines follow within their
documented biases: the uniform current of the image family lies about
1 % above a rod's equipotential solution, the point kernel of `mom`
0.2 … 2 % below.

| World | `fem` | second method: fem / reference − 1 | integral engines: engine / fem − 1 |
|---|---|---|---|
| Weber's disc $b$ = 1 m, 100 Ω·m | 24.9923 Ω | $\rho/(4b)$ = 25 Ω: −3.1e-4 | — |
| rod 0 … 3 m, $a$ = 8 mm, 100 Ω·m (B2) | 33.1933 Ω | solid-rod boundary elements: −1.07e-5; thin-wire Galerkin limit 33.1957 Ω: −7.3e-5 | `image` (line) +0.92 %, `mom` (0.25 m) −1.1 % |
| rod 0.5 … 3.5 m / 0.5 … 5 m | 31.7170 / 22.7906 Ω | solid-rod FV: −1.4e-5 / −1.4e-5 | `image` +1.1 / +0.90 %, `mom` −0.59 / −0.91 % |
| rings $r$ = 2 m at 0.8 m ($a$ = 8 mm, B3), 5 m at 0.5 m and 10 m at 0.8 m ($a$ = 5 mm) | 12.4809 / 6.4182 / 3.4417 Ω | solid-torus boundary elements: −3.00e-5 / −2.63e-5 / −2.56e-5; thin ring in closed form: −5.7e-5 / −3.1e-5 / −2.7e-5 | `image` +2.5e-5 … +3.3e-5, `mom` −0.85 … −1.2 % |
| ring with a central rod, bonded, 100 Ω·m | 11.5338 Ω | boundary elements: −2.26e-5 ($R_{12}$ −1.26e-5); the port matrix symmetric | `image` +0.40 %, `mom` −0.87 % |
| rings in two layers (8 soils, 30 / 10 000 … 10 000 / 30 Ω·m) | | Hankel ring: −2.6e-5 … −7.3e-5 | `image_2layer` +2.3e-5 … +3.8e-5 (+3.1e-4 for 10 000 / 30, −0.99 % for 30 / 10 000 Ω·m, whose series it truncates), `mom` −0.43 … −1.8 % |
| rings in three layers (5 stacks, also below $h_1$) | | Hankel ring: −3.4e-5 … −9.1e-5 | `mom_sommerfeld` −0.45 … −2.1 % (top layer only) |
| rods in two layers (6 cases, through and within a layer) | | solid-rod FV: −1.4e-5 … +2.0e-4 | `image_2layer` +0.08 … +1.4 %, `mom` −1.2 … +0.5 % |
| ring 1 cm above an interface, $r$ = 2 / 5 m at $t = r \ldots D$, $a$ = 5 mm (12 cases) | | Hankel ring + 2-D proximity: −1.3e-5 … −7.0e-5 | `image_2layer` +3.46 … +3.57 % (1000 / 30 Ω·m), +0.17 … +0.23 % (30 / 1000): the thin-wire error next to an interface (`InterfaceContactWarning` for 1000 / 30); it matches its exact thin ring (Hankel) to 4e-5 |
| the same rings 10 cm above the interface (12 cases) | | −1.9e-5 … −6.3e-5 | `image_2layer` +0.003 … +0.019 % |
| 3 m rod with its head 1 mm below … 8 cm into a 30 Ω·m top layer (14 cases) | | solid-rod FV: within ±1.3e-4 | `image_2layer` jumps by a factor 3.8 when its first segment enters the layer (30 / 10 000 Ω·m) |
| ring exactly on an interface, $r$ = 2 m, $a$ = 5 mm (4 soils) | | solid-torus boundary elements: −1.1e-5 … −1.2e-5; Hankel ring: −9.1e-5 … +5.8e-5 | `image_2layer` jumps by 0.001 … 0.012 % between 1 µm above and on the interface |
| rod tip 16 mm above … 16 mm below an interface (1000 / 30 Ω·m) | | solid-rod FV: −4.1e-5 … +1.3e-5 | thin wire (`image_2layer`) +22 % with the tip on the interface, +7.2 % 1 mm above, −4.4 % 1 mm below |
| encased ring $r$ = 2 m at 0.5 / 0.8 / 3 m ($r_b$ = 55 mm, $C$ = 38.16 Ω·m) | rise 1.0057 / 1.0040 / 1.0021 $C/(2\pi r)$ | the bound $\ge C/(2\pi r)$ | the image family adds exactly $C/(2\pi r)$ |
| encased 3 m rod, head at 0 / 0.5 m | rise 1.0086 / 1.0005 $C/L$ | Sunde's decomposition with Dwight's rod (uniform current): −0.96 % / −5.2 % | `image` +1.1 / +1.6 %, `mom` −0.84 / −0.21 % |

(Evidence of stream w4axi, `e01`–`e04`; the boundary elements: evidence
`e07` of stream fw4axi and `e03`, `e05` of stream fw4int; the rows next to
and on an interface and the `image_2layer` columns re-measured after the
wave-3 changes of the image family and the refined ring block of audit
W4A-01, evidence `e02`, `e04`, `e05`, `e07` of stream fw4int.)

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
| B8 Two-port reciprocity | abs(Z_AB/Z_BA - 1) | excluded | — | — | excluded | 3-D model, run once outside the suite (stream fw4int): 0 in all three soils, its port matrix symmetric to 2.2e-16; Z_BA within 0.15 … 0.21 % of mom refined to ds = 1/32 m; mom at the benchmark's 0.5 m reads 0.40 / 0.67 / 0.96 % higher, not converged (on a coaxial rod and ring the axisymmetric model confirms the 3-D transfer resistance to 1.3e-3 and shows mom at 0.5 m 0.3 … 1.3 % high; audit W4R2B-05); excluded from the benchmark run for its cost: rod and ring on different axes are not axisymmetric, so fem solves them with its 3-D model (the fem3d extra; user decision Q37), 0.5 … 1.2 minutes and up to 2 GB per solve, two solves per soil |
| B9 Limits of the layered kernels (`equal_rho_fem`) | abs(Z/Z_ref - 1) | 1e-12 | 0 | 3 | pass | the conforming model meshes the canonical stack (user decision Q37) |
| B9 Limits of the layered kernels (`layer_limit_fem`) | abs(Z/Z_ref - 1) | 0.1 % | 5.0e-4 | 2 | pass | physical limit (user decision Q37) |
| B10 Ring with three bonded rods | abs(Z/Z_ref - 1) | excluded | — | — | excluded | 3-D model, run once outside the suite (stream fw4int): +0.62 % / +0.84 % against mom (error estimate 3.9e-3 / 4.3e-3; image / image_2layer +0.81 / +1.97 % above fem); excluded from the benchmark run for its cost: rods off the ring's axis are not axisymmetric, so fem solves the world with its 3-D model (the fem3d extra; user decision Q37), 2 … 3.5 minutes and up to 3.8 GB per soil (up to 2026.11 -35 % / -45 %, radii summed) |
<!-- generated:end fem -->

B8 and B10 exclude `fem` from the benchmark run for its cost: neither world
is axisymmetric, so `fem` solves it with its 3-D model, 0.5 … 3.5 minutes
and 2 … 4 GB per solve. Run once outside the suite (evidence `e01` of
stream fw4int): in B8 the transfer impedances agree to round-off
($Z_{AB} = Z_{BA}$ in all three soils, the port matrix symmetric to
$2.2\cdot10^{-16}$), and $Z_{BA}$ lies within 0.15 … 0.21 % of `mom` refined
to ds = 1/32 m (at $h_1$ = 5 m 17.16667 Ω against 17.19187 Ω, re-run in
`g04` of stream fw4r2b; the other soils auditor W4R2B). `mom` at the
benchmark's 0.5 m reads 0.40 / 0.67 / 0.96 % higher (homogeneous, $h_1$ =
5 / 2 m): it is not converged and falls at an observed order of about 0.6.
On a coaxial rod and ring (rod 0 … 3 m on the axis, ring $r$ = 2.5 m at
0.8 m) the axisymmetric model confirms the 3-D transfer resistance to
−4.9e-4 at $h_1$ = 5 m (−1.3e-3 … −4.9e-4 in the three soils, auditor
W4R2B) and shows `mom` at 0.5 m +0.63 % high (+0.038 % at 1/32 m; audit
W4R2B-05: up to then the page read `mom`'s coarse value as `fem`'s
deviation). In B10 (the shape of the AP1 station
below) `fem` lies +0.62 / +0.84 % above `mom` on the same segments, within
the pattern of the station, `image` / `image_2layer` +0.81 / +1.97 % above
`fem`.

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
form) and $r_\eta = 0.5442\,R$ the energy radius of the bump. The sum runs
over what the mesh does not resolve:

- the observer's own wire whole: its polyline as far as it has turned by
  at most 90° along the wire (a ring's polygon, a strip, the corners of a
  loop), and every piece on the same axis line that belongs to the
  observer's port or to a wire that meets it — the next piece of a
  galvanic conductor, a rod cut into touching pieces, the surface image of
  a rod whose head reaches the surface;
- of the rest of its electrode — a part of the own polyline that has
  turned by 150° or more along the wire (an arm folded back, the far side
  of a narrow loop or a small ring, the other arm of a narrow V, the next
  arm of a meander or the next winding of a spiral, which run the same way
  again after a full turn), the other wires of a mesh or a star — and of
  every other surface image only the part within $1.5\,R$ of the point;
  between 90° and 150° of turn the own polyline counts by the weight
  $(150° - T)/60°$ whole and by the rest of the weight within $1.5\,R$;
- of a wire of another port the part within $1.5\,R$ (on the observer's
  axis line the whole piece) only where the two wires meet, that is, where
  their surfaces touch: a rod on a ring, a T-joint, a crossing, a galvanic
  conductor and the electrode it is attached to. Wires that touch are one
  conductor and carry about the same current where they meet. Where two
  wires have just stopped touching, this window term fades out linearly
  with the gap between their surfaces, from 1 at contact to 0 at a gap of
  $a_1 + a_2$, so the correction is continuous. Any other wire of another
  port carries its own current, which the observer's correction cannot
  know, and counts nothing.

Each wire reaches its radius from its axis perpendicular to it, up to a
flat end. Two wires that cross therefore touch where their axes come
within $a_1 + a_2$ (the window fades out between $a_1 + a_2$ and
$2(a_1 + a_2)$); a wire that ends at another touches it where its end face
reaches the other's surface; two wires end to end on one axis touch only
where the gap between their ends is zero (to 1e-9 m). An encased wire
reaches its outer shell radius $r_b$, which is what the soil sees: two
encased wires of different ports whose shells touch count as one conductor
in the radius correction, so their separate port entries are not
resolved. Auditor W4A's small
ring 0.75 radii above a rod's head stays 0.49 mm from the rod's end face
although the axes are 15.2 mm apart, within $a_1 + a_2$ = 16 mm. Up to
audits W4R3B-01 and W4R3B-04 the axes had to come within the smaller wire
radius. Two crossing strips whose surfaces still overlapped stopped meeting,
so a 2 m cross jumped by −7.5e-4 in $Z$ between axis gaps of 4.9 and
5.1 mm (`mom` −1.6e-5); it now moves by −2.4e-5. And two stacked rods with
an end gap below the radius met, so the other rod counted whole with the
observer's current.

Farther than $1.5\,R$ the disc average of a smoothed line is its line field
(the 2-D mean-value property of the radial bump), so the mesh already gives
the exact field: for two parallel wires the exact correction is below
$10^{-3}$ of the self term from $1.5\,R$ on, and the surface image of a
horizontal wire ($2z \ge 2.5\,R$ away) never counts. (Up to audit W4B-01
the surface images and every wire within $20\,R$ counted: shallow wires
read up to +1.9e-3 high, two rings 10 cm apart +4.6e-3.) Far inside a
straight wire $\Delta C$ is Sunde's radial resistance
$\frac{\rho}{2\pi}\ln(r_\eta/a)$ of the soil between the wire and the disc;
near an end, a corner or a junction it carries the 3-D geometry. The squared
distance from a piece's axis is taken from the cross product: the
difference $|\mathbf r|^2 - (\mathbf r\cdot\mathbf e)^2$ cancelled on long
pieces off the coordinate axes and clipped the observer's own piece from
about 7 m on (audit W4R2B-01: a 20 m oblique strip read −3.8e-4).

The turn $T$ is the sum of the angles at the corners passed along the
chain from the observer's piece, along a closed chain the shorter way
round. It replaces two earlier rules. Up to audit W4R2B-02 an exactly
parallel test was used: a folded arm tilted by 1e-6 rad counted whole, so a
hairpin's impedance jumped by +1.41e-3 for a tilt of 1e-4 rad. Up to audit
W4R3B-03 a piece counted whole where its direction was within 120° of the
observer's: the third arm of an S-meander, which runs the same way as the
first, 0.20 m beside it, counted whole as soon as it was tilted, and a V
jumped where its turn crossed 120°. The blend makes the correction
continuous in the turn (evidence `h00` of stream fw4r3b, one run each,
another agent running):

| world | base (120° direction rule) | now | `mom` (ds 0.025 m) | `image` |
|---|---|---|---|---|
| S-meander, third arm tilted by 1e-4 rad, $Z$ | +4.65e-4 | −1.24e-4 | −1.26e-4 | −1.19e-4 |
| spiral of 3 windings (pitch 0.15 m), one polyline / bonded windings − 1 | +2.08e-3 | +5.9e-5 | 0 | 0 |
| V of 0.5 m arms, turn 119.9° → 120.1°, $Z$ | −1.26e-3 | +3.30e-4 | +3.44e-4 | +3.48e-4 |

An L corner (turn 90°) and a square loop are unchanged to the last bit.
The far chords of a ring now count by their weight: rings of $r$ = 1 /
0.3 m move by −4.4e-7 / −6.9e-6 in $Z$ (to −1.839e-3 / −5.00e-4 against the
solid torus, from −1.839e-3 / −4.93e-4). A rule that clipped the own polyline by its distance from the
observer's axis line also clipped the curved continuation of a ring and
moved rings away from the solid torus (−1.19e-3 for $r$ = 0.3 m, −2.0e-3
for $r$ = 1 m; evidence `g02` of stream fw4r2b), so it was not taken.

### Wires beside each other

The disc of a smoothed source reaches $R$ from its axis. Where another wire
runs within $1.5\,R$ of it, the mesh no longer gives that wire's exact
field and the radius correction has no term for it (counting the other
wire's near field with the observer's current, as up to audit W4R2B-03,
read two coaxial rings 7.5 cm apart +4.35e-3 in a port entry, above the
estimate). The 3-D model therefore keeps wires that run beside each other
at least $1.5\,R$ apart, as it keeps every disc in the soil and in one
layer: it takes $R$ = (their axis distance) / 1.5 for the whole world
(wires end to end: their gap / 3). Two wires run beside each other where

- a point of one lies within $1.5\,R$ of the *side* of the other (its
  nearest point is not an end of that wire) and the two run within 30° of
  parallel: wires side by side, concentric or diagonal rings, an arm of the
  same polyline folded back beside the point, a galvanic conductor in the
  trench of an earthing strip; or
- the whole of one electrode's wire lies within $1.5\,R$ of another wire it
  does not meet (a small ring around a rod's head); or
- two wires that do not meet lie in line, end to end, with a gap below
  $3\,R$ that runs within 30° of both axes: stacked rods of two electrodes
  (audit W4R3B-01; such rods 1 cm apart read +1.57e-2 in a port entry and
  −7.3e-2 in the transfer entry, against estimates of 4.9e-3 / 4.5e-3).
  Wires end to end keep their gap at $3\,R$, not $1.5\,R$: at $1.5\,R$ the
  transfer entry read 1.5 … 2.0 times outside its estimate.

Stacked rods ($a$ = 8 mm, 1.5 m each, 100 Ω·m) against the axisymmetric
model, two ports, default options (evidence `h07`, `h12` of stream fw4r3b;
base 4f443f5 in brackets):

| gap | $R$ | $R_{11}$ | $R_{12}$ | bonded | estimate (diagonal / transfer) | finer-level nodes, time |
|---|---|---|---|---|---|---|
| 2 mm … 4 cm | refused (below 17.6 mm) | (2 mm / 1 cm / 3 cm: +5.1e-2 / +1.6e-2 / +6.4e-3) | (−2.0e-1 / −7.3e-2 / −2.9e-2) | | | |
| 6 cm | 2 cm | −5.1e-4 | −3.51e-3 | −9.6e-4 | 6.1e-3 / 4.3e-3 | 471 295, 18 s |
| 10 cm | 3.33 cm | +1.15e-3 | **−5.03e-3** | +2.9e-4 | 5.5e-3 / 4.4e-3 | 308 033, 13 s |
| 15 cm | 5 cm | +1.98e-3 | **−5.32e-3** | +1.05e-3 | 4.6e-3 / 4.6e-3 | 256 055, 9 s |
| 18.1 cm | 6 cm | +2.47e-3 | **−5.42e-3** | +1.50e-3 | 5.1e-3 / 4.6e-3 | 203 375, 7 s |
| 30 cm | 6 cm | +2.32e-3 | −3.74e-3 | +1.67e-3 | 5.2e-3 / 4.7e-3 | 201 509, 7 s |
| 1 m | 6 cm | +2.14e-3 | −1.84e-3 | +1.88e-3 | 4.55e-3 / 4.75e-3 | 227 147, 8 s |

Every diagonal entry and the bonded value lie within the estimate. The
transfer entry lies within its estimate at 6 cm and from 30 cm on, but
1.15 … 1.18 times outside it at 10 … 18 cm, where $R$ = gap / 3 is 3.3 …
6 cm (thinner wires get there earlier: 2 mm rods 10 cm apart −3.40e-3
against 4.58e-3). With $R$ at most a fifth of the gap it lies within in
every measured case: 8 mm rods 10 … 30 cm apart at $R$ = 2 … 6 cm read
−2.6e-3 … −3.7e-3 against 4.4 … 4.7e-3 (evidence `h09` of stream fw4r3b).
Where the transfer entry of wires end to end matters, set
`FemOptions.source_radius` to at most a fifth of their gap.

The distance is that of the two wire axes, independent of how the check
samples them. Left alone are wires that meet at 10° or more: the collinear
pieces of a galvanic conductor, a rod cut into touching pieces, a rod on a
ring, a T-joint, a corner, a crossing, a conductor's attachment, and a V or
a branch of 10 … 30°. Two straight wires that meet at an angle $\theta$
come within $1.5\,R$ of each other only within $1.5\,R/\sin\theta$ of where
they meet, and there the window term counts the other wire, whose current
agrees with the observer's (up to audit W4R3B-02 a V narrower than 30° was
refused, with the distance of the sample point nearest the apex, 0.65 mm
for 10°). Wires that meet at less than 10°, nearly parallel or touching
along their length (a cable laid on a strip, the service drops of
`TnNetworkGenerator` laid in a row at the touching distance), run beside
each other like wires that do not meet. They are refused before meshing
where they are closer than 3.3 radii of the thickest bare wire, and the
message names their distance and the angle at which they meet: two bonded
parallel strips 8 mm apart ($a$ = 5 mm) are refused, as a V of 9°.
Touching wires that run together are not one conductor. (Without this
limit, as one conductor, the bonded pair of strips read −1.30e-2 against
one strip of the pair's mean radius $\sqrt{aD}$, `image` −1.16e-2, and a V
of 5° / 2.5° lay on the trend of the wider ones, fem / `image` − 1 =
−1.49e-2 / −1.41e-2; evidence `h03`, `h05` of stream fw4r3b.) Crossings at more than 30° are
left alone as well: near them a point crosses the other wire within a few
$R$. A V whose arms do not touch (a rounded apex, a second strip starting
beside the first) still runs beside: a V of 10° whose second arm starts
3 cm beside the apex takes $R$ = 2 cm.

A V of two 4 m arms meeting at the apex (5 mm wire, 0.8 m deep), as two
bonded strips (two ports) and as one polyline, against `mom` and `image`
(evidence `h03` of stream fw4r3b):

| angle | fem [Ω] | estimate | polyline / two strips − 1 | fem / `mom` (ds 0.0125 m) − 1 | fem / `mom` (0.025 m) − 1 | fem / `image` − 1 |
|---|---|---|---|---|---|---|
| 10° | 21.199700 | 4.2e-3 | −7.0e-6 | +3.35e-3 | +7.36e-3 | −1.47e-2 |
| 20° | 19.942955 | 4.4e-3 | −1.7e-5 | +4.43e-3 | +9.15e-3 | −1.38e-2 |
| 25° | 19.538485 | 4.4e-3 | +8.9e-6 | +4.63e-3 | +9.54e-3 | −1.35e-2 |
| 29° | 19.271152 | 4.5e-3 | −1.4e-5 | +4.80e-3 | +9.82e-3 | −1.33e-2 |
| 31° | 19.151321 | 4.5e-3 | −1.4e-5 | +4.86e-3 | +9.93e-3 | −1.32e-2 |
| 35° | 18.935410 | 4.5e-3 | +2.2e-6 | +5.02e-3 | +1.02e-2 | −1.29e-2 |

The ratios continue the trend of 25 … 35° down to 10°, the two
representations agree to 2e-5, and up to audit W4R3B-02 the worlds below
30° were refused; below 10° they are refused again (see above). (`mom` is not converged at these segment lengths: from
ds = 0.025 to 0.0125 m it moves by +0.40 … +0.51 %.)

One $R$ serves the whole world (the disc quadrature, the energy radius and
the mesh size around the wires use it), so the closest pair refines the
whole mesh: the nodes grow as $1/R$. The smallest $R$ is 2.2 radii of the
thickest bare wire of the world, wherever that wire lies (the energy radius
must exceed 1.2 of its radii): wires beside each other closer than 3.3
such radii, axis to axis, and wires end to end closer than 6.6, are refused
before meshing with `BackendCapabilityError` naming the distance and the
integral engines (stacked 8 mm rods up to a gap of 5.3 cm). Two 4 mm strips 4 cm apart are refused
next to a rod of 20 mm radius elsewhere in the world, and solved at
$R$ = 2.67 cm without it (audit W4R3B-05). Where the finer mesh exceeds
`max_nodes_3d`, `ResourceLimitError` says why $R$ is small.

Two coaxial rings ($r$ = 1 m and 1 m + $D$, 5 mm wire, 0.8 m deep, two
ports) against the solid tori of the axisymmetric model (evidence `g04` of
stream fw4r2b; one run each, another agent running):

| $D$ | $R$ | $R_{11}$ | $R_{22}$ | $R_{12}$ | bonded | estimate (diagonal / transfer) | finer-level nodes, time, memory |
|---|---|---|---|---|---|---|---|
| 3 cm | 2 cm | refused (`ResourceLimitError`, 2.02 million nodes predicted) | | | | | |
| 4.5 cm | 3 cm | −8.4e-5 | −7.2e-4 | −1.72e-3 | −9.9e-4 | 4.1e-3 / 4.1e-3 | 1 034 829, 61 s, 3.2 GB |
| 6 cm | 4 cm | −8.0e-4 | −1.25e-3 | −1.67e-3 | −1.31e-3 | 4.5e-3 / 3.9e-3 | 597 929, 33 s |
| 7.5 cm | 5 cm | −1.25e-3 | −1.61e-3 | −1.70e-3 | −1.56e-3 | 4.2e-3 / 4.1e-3 | 495 591, 24 s |
| 9 cm | 6 cm | −1.12e-3 | −1.41e-3 | −1.61e-3 | −1.42e-3 | 3.9e-3 / 4.1e-3 | 511 257, 27 s |
| 10 cm | 6 cm | −1.20e-3 | −1.48e-3 | −1.57e-3 | −1.45e-3 | 3.9e-3 / 4.1e-3 | 520 285, 27 s |

Every entry lies within 1.8e-3 of the solid tori and inside both
estimates; up to audit W4R2B-03 the port entries read +4.35e-3 at 7.5 cm
and +1.0e-2 at 6 cm (auditor W4R2B). Auditor W4R2A's rings diagonal to each
other 1.6 … 2.05 wire radii apart in $s$ and $z$ (axes 11 … 14.5 mm apart;
+11.4 … +14.3 % against the solid tori before, which `model="auto"` sends to
the 3-D model when the axisymmetric mesh block has no room) and auditor
W4A's small ring 0.75 radii above a rod head (+0.80 %, estimate 0.66 %) are
refused before meshing, naming the integral engines (audits W4R2A-01,
W4R2B-03); `mom` lies within 0.07 … 0.18 % of the boundary elements there
(auditor W4R2A).

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
  $r_b$ = `wire_radius` (side $s = r_b/0.59017$, the same resistance to the
  soil as a circular shell) around every straight encased wire along x, y
  or z, the metal radius following from
  $C = \frac{\rho_c}{2\pi}\ln(r_b/r_a)$. Where wires meet, the prisms share
  their concrete (user decision Q40, see
  [Concrete shells](../concrete-shells.md)). The square keeps the
  resistance to the soil, but in the shell-dominated limit
  ($\rho_c \gg \rho$) its concrete is 3.8 % less resistive than the
  circular shell: the concrete between the bar and a nearly equipotential
  prism surface is set by the square's inner conformal radius at its
  centre, $0.5394\,s = 0.914\,r_b$ (for a 5 mm bar with 5 cm cover
  $\ln(50.3/5)/\ln(55/5) = 0.962$; Schwarz–Christoffel and a least-squares
  Green's function agree, audit W4B-04). One square side cannot match both
  radii.

### Solve, levels and cost

The disc averages couple all nodes of a disc, so the coupling is never
formed: conjugate gradients run on the operator, preconditioned by the
smoothed-aggregation multigrid of the soil stiffness (pyamg, set up under a
fixed seed of numpy's global random state, which is restored afterwards, so
a result does not depend on the user's seed), one solve per port. The
iterations grow with the layer contrast: 16 … 115 up to the contrast 33 of
the AP1 soils, 384 … 530 at 100 and 1 225 … 1 264 at 333 at the finer level
for a rod through the interface (both orders of the layers, each within its
error estimate of the axisymmetric model; evidence `f13` of stream
fw4fem3d); at 1 000 they do not converge in 3 000 iterations and the solve
raises `BackendCapabilityError` naming the contrast, the axisymmetric model
(any contrast) where it takes the world, and the integral engines (audit
W4B-03; a multigrid of the stiffness plus the diagonal of the wire coupling
gives the same counts). Without a layer contrast the refusal names its
circumstances instead — resolved concrete and its contrast to the soil, or
homogeneous soil, where 19 … 27 iterations are the rule (audit W4R2B-06).
The finer level is the red refinement of the coarser one; the port
conductance matrix is extrapolated with the second order and the relative
difference of a diagonal entry to the finer level is
`metadata["fem_error_estimate"]` (3 … 5e-3 at the defaults). The levels are
not asymptotic for every geometry (error ratios between the levels of
3.0 … 4.4), so the extrapolated value carries up to about 2e-3. With the
ratio $r$ of the two levels' errors the extrapolated value errs by
$e_1(4 - r)/3$ and the estimate is $e_1(r - 1)/3$, so the estimate bounds the
error while $r \ge 2.5$: rods run at 3.3 … 4.4 and rings at 3.0 … 3.4, where
the estimate is 2.1 … 4.4 times the error (evidence `f04` of stream
fw4fem3d, the ratios recomputed in `g05` of stream fw4r2b).
`metadata["fem_error_estimate_transfer"]` is the same for the transfer
resistances, the largest relative difference of an off-diagonal entry
(`NaN` for one port or one level; audit W4R2B-05: the estimate covered the
diagonal only). Where the model holds it bounds the error of the
transfer entries with a margin of 1.06 … 6.3: two 3 m rods 10 m apart
1.06 against `image` and 1.8 … 1.9 against `mom`, 30 m apart 1.7 / 1.8,
two coaxial rings 4.5 … 10 cm apart 2.3 … 2.6, and 6.3 on the coaxial
analogue of benchmark B8 (rod 0 … 3 m on the axis, ring $r$ = 2.5 m at
0.8 m, 1000 / 30 Ω·m with $h_1$ = 5 m: 3.1e-3 against an error of $R_{12}$
of −4.9e-4; evidence `g04` of stream fw4r2b, `h00`, `h07`, `h11` of stream
fw4r3b). In B8 itself it is 2.9e-3, with `mom` refined to ds = 1/32 m
1.5e-3 above `fem` and still falling. For two wires end to end the rule
keeps the gap at $3\,R$; there the estimate is 0.85 … 0.87 times the
error of $R_{12}$ for 8 mm rods at $R$ = 3.3 … 6 cm, and bounds it from a
gap of $5\,R$ on in every measured case (see *Wires beside each other*; up to audit W4R3B-01, at
$R$ = 6 cm for a gap of 3 cm, −2.9e-2 against 4.5e-3). Resolved concrete converges at
first order ($r \approx 2$): there the estimate is no bound. Next to a
strongly contrasting interface the two levels can lie on either side of the
solid wire's value, and the estimate does not see the thin-wire error (see
*Validation of the 3-D model*).

Cost (fresh processes, one solve each, other jobs running; evidence `f06`,
`f15` of stream fw4fem3d): a ring of 12.6 m (600 000 nodes at the finer
level) takes 26 s and 1.8 GB, the AP1 station (28 m of wire, 1.2 … 1.4
million nodes) 75 … 140 s and 4.3 GB. The solve adds about 2.9 GB per
million nodes of the finer mesh (3.0 … 3.1 GB with the process itself), so
`FemOptions.max_nodes_3d` (2 million nodes) refuses meshes of more than
about 6 GB before they are built (`ResourceLimitError`).
`metadata["fem_peak_memory_gb"]` is what the solve added to the peak
resident memory of the process — in a fresh process the memory the solve
needs, less (down to zero) when an earlier computation of the process
already needed more — and `metadata["fem_process_peak_memory_gb"]` that
peak itself, the figure `/usr/bin/time -l` reports (audit W4B-09). Both
are in GiB.

### Guards of the 3-D model

- `ImportError` naming the `fem3d` extra if it is missing.
- `BackendCapabilityError` before meshing: a wire so close to the soil
  surface or a layer interface that its disc cannot lie in the soil and in
  one layer; wires that run beside each other closer than 1.5 times the
  smallest source radius (3.3 radii of the thickest bare wire of the world,
  axis to axis), or lie end to end without touching closer than 3 times it
  (see *Wires beside each other*; wires that meet at less than 10° count as
  running beside each other), naming the wires, their distance and the
  integral engines; resolved concrete around a wire that is not straight
  along x, y or z (rings, oblique strips), whose concrete would reach the
  surface, or whose cover is so thin that the smoothed source does not
  reach beyond the metal (a 5 mm bar with 5 mm cover; audit W4R2B-04),
  naming `concrete_resistivity=None`. After meshing: conjugate gradients
  that do not converge (layer contrasts above the tested 300, such as a rod
  through a 1 / 1000 Ω·m interface), naming the contrast, the axisymmetric
  model where it takes the world, and the integral engines.
- `ResourceLimitError` before anything is meshed: the display segments
  above `Engine.max_segments` (a unit error of `segment_length`), more
  ports than `FemOptions.max_ports_3d` (100) — every electrode and every
  leaking piece of a galvanic distributed conductor is a port and costs
  one conjugate-gradient solve per level, about 0.9 s per port and level
  at 600 000 nodes, so the message names the estimated time (coarsen the
  conductor's `discretize_segment_length`, or raise the limit to solve the
  world on purpose) — and the predicted finer mesh above
  `FemOptions.max_nodes_3d`; after the coarser mesh, its exact refinement
  above that limit. Where the source radius was limited by wires beside
  each other, the message says so.
- `ValueError`: a `source_radius` that does not reach beyond the bare
  wire (it must be at least 2.2 wire radii).
- `InterfaceContactWarning` (the solve goes on): a horizontal wire whose
  axis lies within the reach of a contrasting layer interface (3.9 wire
  radii at a contrast of 33; the table of the reach is under the
  [warning](../../reference/warnings.md#interfacecontactwarning)). The
  smoothed line source shares the thin-wire error of the integral engines
  there (+9.2e-3 against the solid torus 3 radii above a 1000 / 30 Ω·m
  interface, beyond the error estimate; audit W4B-02), so the dispatch
  warns before the 3-D model runs, as the integral engines do (audit
  W4A-07); the axisymmetric model solves the wire as a solid torus and
  does not warn. A bare wire closer than about 2.8 radii to an interface
  is refused (its disc would not lie in one layer).

### Validation of the 3-D model

Two methods each (evidence of stream w4fem3d): the axisymmetric model where
the world allows it, closed forms and the integral engines otherwise.

**Against the axisymmetric model** (defaults of the 3-D model; levels 0/1
extrapolated; re-measured after audit W4B-01, evidence `f04` of stream
fw4fem3d — the rows moved by at most 3e-5):

| World | axisymmetric model | 3-D model / axisymmetric − 1 | error estimate | finer-level nodes | thin-wire engine / axisymmetric − 1 | `mom` / axisymmetric − 1 |
|---|---|---|---|---|---|---|
| rod 0 … 3 m, $a$ = 8 mm, 100 Ω·m | 33.19327 Ω | −1.4e-4 | 4.6e-3 | 186 903 | `image` +0.92 % | −1.45 % |
| rod 0.5 … 3.5 m, 100 Ω·m | 31.71703 Ω | +6.5e-4 | 4.8e-3 | 203 383 | `image` +1.11 % | −1.22 % |
| ring $r$ = 2 m at 0.8 m, $a$ = 8 mm, 100 Ω·m | 12.48090 Ω | −1.93e-3 | 4.7e-3 | 600 179 | `image` +0.003 % | −1.17 % |
| ring $r$ = 5 m at 0.5 m, $a$ = 5 mm, 100 Ω·m | 6.41819 Ω | −1.94e-3 | 4.1e-3 | 1 500 815 | `image` +0.003 % | −0.91 % |
| rod 0 … 3 m through $h_1$ = 2 m, 100 / 1000 Ω·m | 61.22214 Ω | −1.30e-3 | 4.3e-3 | 194 897 | `image_2layer` +0.08 % | −1.32 % |
| rod 0 … 3 m above $h_1$ = 4 m, 1000 / 30 Ω·m | 300.66258 Ω | +2.1e-4 | 5.2e-3 | 187 027 | `image_2layer` +1.20 % | −1.59 % |
| rod 0 … 3 m above $h_1$ = 5 m, 100 / 1000 Ω·m | 38.84736 Ω | −3.0e-4 | 4.5e-3 | 195 663 | `image_2layer` +0.74 % | −1.25 % |
| ring $r$ = 2 m at 0.8 m above $h_1$ = 2 m, 1000 / 30 Ω·m | 80.71707 Ω | −1.97e-3 | 4.3e-3 | 595 993 | `image_2layer` +0.004 % | −1.81 % |
| ring $r$ = 2 m at 0.8 m, three layers 100 / 1000 / 30 Ω·m (1 m, 2 m) | 21.41013 Ω | −9.1e-4 | 4.0e-3 | 641 613 | `mom_sommerfeld` −0.68 % | refused |

Every case lies within 2.0e-3 of the axisymmetric model and inside its own
error estimate. The free ends of a wire read slightly high, a wire without
ends (a ring) about 2e-3 low: the two mesh levels are not yet in the
asymptotic range for every geometry (error ratio between the levels 3.0 …
3.4 for rings, 3.3 … 4.4 for rods). A ring $r$ = 1 m ($a$ = 5 mm) reads −1.62e-3 …
−1.83e-3 at 0.06, 0.1, 0.3 and 0.8 m deep (before audit W4B-01 the
surface-image term drifted it from +2.4e-4 to −1.85e-3), and two coaxial
rings 1.0 m and 1.1 m as two ports −1.19e-3 / −1.48e-3 / −1.57e-3 in
$R_{11}$ / $R_{22}$ / $R_{12}$ against an estimate of 3.9e-3 (before:
+4.6e-3 in $R_{11}$; evidence `f08`, `f09` of stream fw4fem3d; `image`, the
exact thin rings, agrees with part A to 3.7e-4); closer rings with the
source radius they need are in the table of *Wires beside each other*.

**Next to a contrasting interface** the thin wire itself errs, and the 3-D
model with it: its current sits on the wire axis, while on the solid wire
it crowds to the side that faces a more conductive layer. A ring ($r$ = 1
m, $a$ = 5 mm, 0.8 m deep) with its axis a gap $g$ above a 1000 / 30 Ω·m
interface, against the solid torus (evidence `f10` of stream fw4fem3d):

| $g$ | 3-D model / axisymmetric − 1 | error estimate | levels L0 / L1 | exact thin ring (`image_2layer`) / axisymmetric − 1 |
|---|---|---|---|---|
| 0.3 m | −1.7e-3 | 3.3e-3 | −1.47 / −0.50 % | +6e-5 |
| 0.075 m | +2.2e-3 | 8.6e-3 | −3.12 / −0.64 % | +3.3e-4 |
| 0.03 m ($R$ = 2.4 cm) | −6e-5 | 4.6e-3 | −1.80 / −0.46 % | +2.3e-3 |
| 0.015 m ($R$ = 1.2 cm) | **+9.2e-3** | 5.4e-3 | −1.24 / **+0.37 %** | +1.24e-2 |

At 1.5 cm (1.9 million nodes) the model lies between the thin ring and the
torus, above its estimate, its two levels on either side of the torus: the
estimate does not see the thin-wire error (audit W4B-02); the solve warns
there (`InterfaceContactWarning`, see *Guards of the 3-D model*). Over a
more resistive layer (30 / 1000 Ω·m) the same ring reads −6.5e-4 at
1.5 cm, within its estimate of 2.3e-3; rods through a 10 / 1000 or
1000 / 10 Ω·m interface read −7.1e-4 / +5.3e-4 (evidence `f13`).

**AP1 station** in the seven corner soils of the AP1 grid: square ring
4 m × 4 m at 0.6 m (5 mm wire), three 4 m rods (8 mm) from three corners,
bonded; input impedance at DC (re-measured after audit W4B-01, evidence
`f06` of stream fw4fem3d: −0.9e-4 … −1.1e-4 against the first version).
The 3-D model lies between the two thin-wire engines, as the axisymmetric
model does for rods and rings: the uniform current per wire of
`image_2layer` reads high, the point kernel of `mom` low.

| $\rho_1$ / $\rho_2$ / $h_1$ (Ω·m, m) | `fem` (3-D) | error estimate | `image_2layer` / `fem` − 1 | `mom` (0.25 m) / `fem` − 1 |
|---|---|---|---|---|
| 1000 / 200 / 30 | 72.322 Ω | 4.3e-3 | +1.53 % | −0.62 % |
| 1000 / 200 / 5 | 57.956 Ω | 4.6e-3 | +2.45 % | −0.79 % |
| 1000 / 30 / 30 | 71.519 Ω | 4.8e-3 | +1.53 % | −0.64 % |
| 1000 / 30 / 5 | 52.741 Ω | 4.8e-3 | +2.99 % | −0.85 % |
| 30 / 200 / 30 | 2.4650 Ω | 4.5e-3 | +1.34 % | −0.55 % |
| 30 / 200 / 5 | 3.5623 Ω | 4.8e-3 | +0.67 % | −0.37 % |
| 30 (homogeneous) | 2.2508 Ω | 4.3e-3 | +1.48 % (`image`) | −0.59 % |

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
`growth` 0.6, 1.5 million nodes; re-measured after audit W4B-01, evidence
`f07` of stream fw4fem3d): encasing its braces raises $Z$ by
+49.1 / +318.2 % at $\rho_c$ = 1 000 / 5 000 Ω·m (`mom` +49.6 / +321.3 %), and
the owner model of the image family lies +1.6 / +2.8 % above the 3-D model
with bare braces, +0.14 / +0.03 % with encased ones. On straight bars, an
L, a T-joint and a crossing the per-wire shell of the 3-D model agrees with
`image` to 0.01 … 0.21 % and with `mom` to 0.06 … 0.28 % (re-measured after
audit W4B-01, evidence `f12` of stream fw4fem3d). **Resolved
concrete** (`concrete_resistivity`, user decision Q40) reads the conductance
per metre 2.8 … 4.6 % high at the default levels: in the shell-dominated
limit the square prism's concrete is 3.8 % less resistive than the circular
shell (see *Mesh, layers and concrete*), and its prism faces and edges
converge at first order. So it serves differences on one mesh: against
each wire's own shell a junction of a foundation acts like a few cm of
wire, at most about $2\,r_b$, sign not settled, and the house mesh of the
generators changes by about 1 % or less (60 m with 4 corners, 4 T-joints
and a crossing: 0.35 … 0.85 % with every sign aligned, from the
extrapolated junctions of both decompositions, 0.5 … 1.2 % from the finer
level alone; evidence `f12` of stream fw4fem3d). The junction numbers are
not converged: one refinement of the prism faces moved the corner by
about 5 cm, the conductance and the impedance decomposition differ in
sign for the corner and the T-joint, and the T-joint and the crossing were
not refined (audit W4B-05; [Concrete shells](../concrete-shells.md)).

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
