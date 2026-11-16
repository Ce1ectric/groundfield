# ADR-0016: A conforming finite-element model in `fem`

| | |
|---|---|
| **Status** | Accepted — part A (axisymmetric model) and part B (three-dimensional model, user decision Q40 on the shell overlap) 2026-11-15 |
| **Date** | 2026-11-15 |
| **Deciders** | Project maintainers (user decision Q37 of 2026-11-14) |
| **Scope** | `groundfield.solver.fem` (now a package), the `fem` entry of the backend registry, `FemOptions`, the optional `fem3d` extra |

## Context

Up to 2026.11 `fem` replaced every galvanic cluster by one *equivalent
hemisphere* — the hemisphere with the closed-form resistance of the
electrodes in homogeneous soil of the top-layer resistivity — and solved
the volume problem of that hemisphere. In homogeneous soil that reproduces
Dwight's formula by construction; in layered soil it describes another
geometry. Every audit round of the release found new limit cases (an
interface 1 cm below a deep ring read +191 … +308 % without a message, a
rod reaching 1 µm … 8 cm into a more conductive top layer −44 … −78 % with
a warning, a ring lying exactly on an interface −21 … +30 %), and the
reduction had grown a dozen structural rules (review pass 10, 10.23; user
decision Q23; audits D2A, R2A). User decision Q37 stops adding rules and
replaces the reduction by a conforming model; the scope decision of
2026-11-15 adds the full 3-D model to the release (part B, with the
overlap of concrete shells at junctions, user decision Q40).

`fem` stays a *reference* backend (grey-box principle): the integral
engines are the production engines, `fem` checks them with a method that
shares nothing with them but the network solve.

## Decision

### 1. Physical model: the electrodes as they are

The quasi-static field $-\nabla\cdot(\sigma(z)\nabla\varphi) = 0$ in the soil,
no current into the air, $\varphi \to 0$ far away. Every electrode is a
perfect conductor at its own potential — a **port**: a rod is the solid
cylinder of its `wire_radius` with flat end faces, a ring the solid torus of
its `wire_radius`. The port conductance matrix $G$ (currents leaked for the
port potentials, symmetric positive definite) is the earth part of the
solve; nothing is reduced or averaged.

A concrete shell (ADR-0012) keeps its meaning: the soil sees the shell
surface $r_b$ (`wire_radius`) and every metre of wire adds the radial
resistance $C$ = `concrete_shell_coefficient_ohm_m`. `fem` applies it
locally, as the surface conductance $g = 1/(2\pi r_b C)$ between the metal
potential and the soil at every point of the electrode surface, so the
current may redistribute along the electrode. For a cylindrical shell that
is exact (Sunde's decomposition) and needs neither $\rho_c$ nor $r_a$,
which the electrode does not carry; resolving the concrete as a volume is
left to part B, where the overlap at junctions (Q40) needs it.

### 2. Axisymmetric discretisation (part A)

A world whose electrodes are rods on one vertical axis and horizontal rings
centred on it is a body of revolution and reduces to the meridian
half-plane $(s, z)$ with the weight $2\pi s$:

- **Graded tensor mesh.** Lines $s = s_i$ and $z = z_j$ through every
  feature — rod surfaces and end faces, layer interfaces, soil surface,
  axis, truncation box — graded geometrically away from them
  ($h(x) = \min_f(h_f + g|x - x_f|)$, $g = 0.2$). The rod's end edges are
  re-entrant corners of the conductor; with a spacing of $0.002\,a$ there
  (a few dozen lines more) the convergence stays of second order even for
  a rod tip lying on an interface. Linear triangles, the weight integrated
  exactly.
- **O-grid for the ring wire.** A square block of up to three wire radii
  around the wire cross section is filled by an O-grid conforming to the
  tensor lines on its boundary and inscribed in the circle of the wire
  (32 nodes at level 0). The block keeps clear of the surface, the axis,
  the other electrodes and every interface more than $1.05\,a$ from the
  wire axis; a nearer interface (a ring on it, or within its radius) is
  cut into the block's elements, so every element lies in one layer. Such
  a block is finer: the gap between the wire and the interface, and the narrow wedge where the interface
  cuts the wire near its edge, make the field vary over
  $\sqrt{2a\,|d - a|}$ ($0.14\,a$ at $d = 1.01\,a$). With the default
  block the levels converged at first order and not monotonically there
  (up to −6.0e-3 next to a layer 33 times more conductive, the error
  estimate below the error in 7 of 14 cases, audit W4A-01); refined to 64
  intervals the default lay within 3.3e-4 at contrast 33 (stream fw4axi),
  but beyond it the levels converged at first order again (touching a 1000
  times more conductive layer −1.6e-2, estimate 1.1e-2; audit W4R2A-02). A
  refinement alone does not resolve this: the field is singular where the
  interface cuts the wire at a narrow angle or touches it, on a scale that
  shrinks with the contrast. The block's O-grid is graded towards those
  points in angle and radius (32 intervals, cells of 1e-3 wire radii),
  which resolves them at fewer nodes: within 5.1e-5 of the boundary
  elements up to a contrast of 1000, the estimate above the error in all
  113 cases of the audit's sweep (evidence `e05` of stream fw4r2a).
- **Truncation.** A box at `r_far_factor` (1000) characteristic lengths —
  the electrodes' size, or the depth of the deepest interface times the
  layer contrast if larger (a resistive bottom keeps the current in the
  layers above it over that distance) — carries the Robin condition
  $\partial_n\varphi = -(\mathbf n\cdot\mathbf r/r^2)\varphi$, the exact
  Dirichlet-to-Neumann map of the monopole far field on any boundary. The
  graded mesh reaches the box with a few dozen extra lines.
- **Ports.** The surface nodes of a bare electrode are one unknown; the
  port conductance matrix is the Schur complement of the stiffness matrix
  onto the ports (one sparse LU factorisation for all ports).
- **Refinement and extrapolation.** Level $\ell$ bisects every interval of
  the level-0 lines $\ell$ times. The tensor part of the spaces is nested,
  the O-grid of a ring is not (its layers move with the level, the
  inscribed polygon grows): one level converges at second order and, in
  every case tested away from interfaces, from below — observed, not
  guaranteed (audit W4A-09). The default solves levels 0 and 1 and
  extrapolates the port conductance matrix with the observed second order
  (Richardson); the relative difference between the extrapolated and the
  finer level is reported as `metadata["fem_error_estimate"]` (about the
  finer level's error, 0.88 … 0.99 of it in 36 cases of auditor W4A and
  0.94 … 1.07 in 60 cases of a ring at an interface, audit W4R2A-02; NaN
  with `extrapolate=False`, audit W4A-02).
- **Resource guard.** `FemOptions.max_nodes` (2 million nodes of the finer
  mesh, about 8 GB: the axisymmetric solve needs about 4 GB per million
  nodes, 0.82 / 3.15 / 5.15 GB at 0.20 / 0.80 / 1.22 million measured with
  `/usr/bin/time -l`, evidence `e02` of stream fw4axi; audit W4A-05)
  against a prediction from the mesh lines before anything is assembled.
  The prediction counts the O-grid exactly and bounds the cut of an
  interface, so it errs high (audit W4A-10).

Every other world — electrodes other than rods and rings, a rod or ring
off the common axis, galvanic distributed conductors (their leakage runs
along their route), touching electrodes, a ring wire that reaches within
5 % of its radius to the soil surface (the solid torus must fit) and a
ring without room for its mesh block (within about a wire radius,
diagonally, of a rod's end edge; audit W4A-03) — goes to the
three-dimensional model of section 5 under the default
`FemOptions.model = "auto"`. With `model = "axisymmetric"` it is refused
with `BackendCapabilityError`, naming the backends that solve it. (Part A
alone, before part B, refused every such world.) The common axis has a
tolerance of 1e-3 of the smallest wire radius or surface gap between two
electrodes (at least 1e-12 of the world's size): the resistance is even
in the offset $\delta$, so the axisymmetric reading is off by
$\kappa(\delta/L)^2$ with $|\kappa| \le$ 6e-4 for $L$ the wire radius and
$\le$ 1.1e-2 for $L$ the gap (`image` and `mom`, three worlds), below 1e-6.
Up to audit W4A-08 it was 1e-9 of the world's size (3 nm): a rod 1 µm off
a ring's axis went to the 3-D model (40.7 s, 2.35 GB, −1.43e-3).

### 3. Network and results on the shared pipeline

The soil model delivers an `EarthPorts` record (`solver/fem/_ports.py`):
the port conductance matrix, the display segments of the electrodes (the
segments the integral backends would use at the engine's
`segment_length`) and the current each of them leaks for 1 V on each
port. `solver/fem/_network.py` runs the pipeline of the integral backends
(ADR-0014) with one pseudo-segment per port and $Z = R = G^{-1}$ as the
multi-port matrix: lumped and isolated distributed conductors, inductive
branches with their earth-return models, the skin effect, junctions,
`return_to` and voltage sources are honoured as there, at every frequency
of the engine. The result carries point sources with the model's leakage
distribution, so `FieldResult.potential` and the touch-voltage helpers
work (homogeneous and two-layer soil, as for the other backends). The
solve runs the coordinate check of user decision D5.

### 4. Package layout (the interface for part B)

```
solver/fem/
  __init__.py       solve_fem (dispatch), public names, module docstring
  _mesh.py          graded lines, tensor grid, O-grid, interface cut (2-D)
  _assembly.py      axisymmetric P1 assembly, far field, shells, port Schur complement
  _axisymmetric.py  world -> coaxial electrodes -> EarthPorts (part A)
  _ports.py         EarthPorts: the interface of any soil model to the network
  _network.py       EarthPorts + world -> FieldResult on the shared pipeline
  _guards.py        refusals shared by every model, the remedy text
  _legacy.py        equivalent_hemisphere_radius (deprecated), FemReductionWarning
```

Part B adds its 3-D model as further modules (for example `_mesh3d.py`,
`_assembly3d.py`, `_solid.py`) whose entry point returns an `EarthPorts`
for any world, and extends the dispatch in `solve_fem` (axisymmetric where
possible, 3-D otherwise); `_network.py`, `_ports.py` and `_guards.py` are
shared unchanged.

### 5. Three-dimensional model (part B)

Every world that the axisymmetric model cannot hold — strips, polylines,
stars, meshes and grids, rods and rings side by side, several clusters,
galvanic distributed conductors — is solved in three dimensions.
`FemOptions.model = "auto"` (default) keeps the axisymmetric model where it
applies (bit for bit) and takes the 3-D model otherwise; `"3d"` forces it,
`"axisymmetric"` refuses every other world. Modules: `_wires3d.py`
(wires, radius correction), `_mesh3d.py` (mesh), `_assembly3d.py`
(stiffness, immersed elements, sources, port solve), `_solid.py` (world
→ `EarthPorts`).

- **Electrodes as smoothed line sources with an analytic radius
  correction.** A resolved wire surface (radius 5 mm in a domain of
  kilometres) cannot be meshed affordably in 3-D, and the brief asked for
  embedded line sources. Every wire axis (a ring as a polygon of 1°
  chords) carries the current density $j(s)$, spread over the disc of
  radius $R$ perpendicular to the axis with the bump
  $\eta = \frac{3}{\pi R^2}(1 - r^2/R^2)^2$; the wire sees the
  $\eta$-weighted disc average $\Pi(s)$. What lies inside the disc is
  added analytically as a radial resistance per length,
  $U - \Pi = (C + \Delta C)\,j$, with the shell coefficient $C$ of ADR-0012
  and the correction $\Delta C$ = the difference between the reduced
  thin-wire kernel of radius $a$ and that of the smoothed source (energy
  radius $0.5442\,R$), integrated in closed form over what the mesh does
  not resolve: the observer's own wire whole (its polyline as far as it
  has turned by at most 90° along the wire, by a weight falling linearly to
  0 at 150°, and every piece on its axis line that belongs to its port or
  to a wire that meets it — the next piece of a galvanic conductor, the
  surface image of a rod reaching the surface); of the rest of its
  electrode (the own polyline beyond, by the rest of the weight), of a wire
  of another port that meets the observer's wire (their surfaces touch: a
  rod on a ring, a T-joint, a crossing; one conductor), fading out with the
  surface gap where two wires have just stopped touching, and of every
  other surface image only the part within $1.5\,R$ of the point; any other
  wire of another port nothing, since it carries its own current (audit
  W4R2B-03). An exact-parallel test of folded arms switched at a tilt of
  1e-6 rad (audit W4R2B-02), a test of the direction within 120° counted a
  meander's arm whole after a full turn and made a V jump at 120°, a
  meeting distance of the smaller radius made two crossing strips jump
  where their surfaces still overlapped, and a collinear rod of another
  electrode counted whole across an end gap (audits W4R3B-01, -03, -04);
  the axis distance from the cross product keeps long oblique pieces whole
  (W4R2B-01). Beyond that the disc
  average of a smoothed line is its line field (2-D mean-value property of
  the radial bump), so the mesh is exact there: for two parallel wires
  the exact correction is below $10^{-3}$ of the self term from $1.5\,R$ on (evidence `f01` of stream
  fw4fem3d). The first version counted the surface images and every wire
  within $20\,R$, with the observer's current: shallow wires read up to
  +1.9e-3 high and two rings 10 cm apart +4.6e-3, above their error
  estimate (audit W4B-01). Far inside a straight wire $\Delta C$ is Sunde's
  $\frac{\rho}{2\pi}\ln(r_\eta/a)$. The result depends little on $R$
  (evidence `e07` of stream w4fem3d, re-measured as `f05` of stream
  fw4fem3d: a 3 m rod and a ring at $R$ = 4, 6 and 10 cm against part A).
  Default $R$ = 6 cm. Wires that do not meet but run beside each other
  (side by side within 30° of parallel, a folded arm, a wire within reach
  of another over its whole length) are kept at least $1.5\,R$ apart, two
  wires end to end in line with a gap at least $3\,R$: $R$ becomes their
  axis distance over 1.5 (end to end: their gap over 3) for the whole
  world, and below 2.2 radii of the thickest bare wire of the world the
  solve is refused before meshing (`BackendCapabilityError`, naming the
  distance and the integral engines). Wires that meet at 10° or more
  (junctions, corners, a V or a branch of 10 … 30°) and crossings at more
  than 30° are left alone; wires that meet at less than 10°, nearly
  parallel or touching along their length, run beside each other (audits
  W4R2B-03, W4R2A-01, W4R3B-01, W4R3B-02, and the coordinator's decisions
  on them; up to then two coaxial rings 7.5 cm apart read +4.35e-3 in a
  port entry, two rings diagonal to each other, surface gaps
  1.3 … 3.8 mm, +11 … +14 %, two stacked rods 1 cm apart +1.6e-2, and a V
  narrower than 30° was refused). An encased wire meets others by its outer
  shell radius $r_b$, what the soil sees: two encased wires of different
  ports whose shells touch are one conductor in the radius correction, and
  their separate port entries are not resolved.
- **Isotropic mesh, deterministic bisection.** A coarse grid of
  $8 \times 8 \times 4$ cubes (six Kuhn tetrahedra each) on the box is
  refined by conforming longest-edge bisection (own implementation with
  ties broken by node numbers; scikit-fem's adaptive refinement asserts on
  some backgrounds) until every element is below
  $1.5\,(R/2 + g\max(0, d - R))$, $d$ the distance from the nearest wire,
  $g = 0.5$. A graded *tensor* grid, the first design, carries the fine
  lines of the layer interfaces and electrodes to the box as slabs
  thousands of times wider than thick: the multigrid needed 345 iterations
  for a 5 m slab and did not converge for a 0.3 m one; the isotropic mesh
  needs 16 … 115 (up to 115 with immersed layer interfaces).
- **Layer interfaces as partially penalised immersed elements.** No
  interface is a mesh plane (a plane through the isotropic mesh would cut
  the large far elements into slivers). A tetrahedron that one interface
  cuts gets the immersed shape functions of Li, Lin & Wu (2003) /
  Kafafy et al. (2005) — linear on either side, continuous, with
  continuous normal flux — and the faces the interface crosses get the
  symmetric consistency and penalty terms of Lin, Lin & Zhang (2015);
  nodes within 30 % of their shortest edge of a plane (and a quarter of
  the thinner layer) are moved onto it so no cut leaves a sliver. The
  scheme reproduces a layered field exactly (to 3e-14; without the face
  terms −1.6e-4 … −6.4e-4 on the same cube, `e01` / `e08` of stream
  w4fem3d). A laminate mean (arithmetic along, harmonic across the layers)
  is a non-conforming average of first order; it is kept only for elements
  more than four times larger than the thinner layer they cut (far from
  the electrodes, where the field varies on scales far above the layer).
- **Solve.** The disc averages couple every node of a disc with every
  other, so the coupling $A^\mathsf T D A$ is never formed: conjugate
  gradients on $K + A^\mathsf T D A$ with the smoothed-aggregation
  multigrid of $K$ (pyamg) as preconditioner, one solve per port
  ($G$ is quadratic in the solution, so its error is the square of the
  solver's). A direct sparse factorisation (SuperLU, `scipy.sparse.linalg.splu`)
  needed 93 s at 155 000 nodes (COLAMD ordering) and had not finished after
  15 minutes at 68 000 nodes with the MMD ordering. pyamg draws the start
  vectors of its spectral-radius estimates from numpy's global random
  state; the hierarchy is set up under a fixed seed and the user's state is
  restored, so results are bit-identical whatever the user's seed and the
  user's stream is not advanced (audit W4B-07). The iterations grow with
  the layer contrast (16 … 115 up to the contrast 33 of AP1, 384 … 530 at
  100 and 1 225 … 1 264 at 333 for a rod through the interface); at 1 000
  they do not converge in 3 000 iterations, and the solve raises
  `BackendCapabilityError` naming the contrast, the axisymmetric model
  where it takes the world, and the integral engines; the tested range is
  up to about 300 (audit W4B-03; without a layer contrast the refusal
  names its circumstances instead, audit W4R2B-06). The multigrid of
  $K + \operatorname{diag}(A^\mathsf T D A)$ gives the same counts (evidence
  `f02` of stream fw4fem3d).
- **Levels and extrapolation.** The finer level is the red refinement of
  the coarser one (nested); $G$ is extrapolated with the second order as
  in part A. The levels are not yet asymptotic for every geometry (error
  ratios $r$ = 3.3 … 4.4 for rods and 3.0 … 3.4 for rings), so the
  extrapolated value carries up to 2e-3. It errs by $e_1(4 - r)/3$, the
  reported estimate is $e_1(r - 1)/3$ (3 … 5e-3): a bound while
  $r \ge 2.5$, 2.1 … 4.4 times the error for rings. The same level
  difference of the transfer entries is reported as
  `metadata["fem_error_estimate_transfer"]` (audit W4R2B-05: the
  estimate covered the diagonal only; where the model holds it is
  1.06 … 6.3 times the error of the transfer entries; for wires end to end
  at the gap of $3\,R$ the rule keeps, 0.85 … 0.87 times it for 8 mm rods at
  $R$ = 3.3 … 6 cm, and a bound from $5\,R$ on). This holds away from layer
  interfaces and other wires — the model keeps wires that run beside each
  other at least $1.5\,R$ apart, and two coaxial rings 4.5 … 10 cm apart lie
  within 1.8e-3 of the solid tori in every port entry. Next to a strongly
  contrasting interface the thin-wire error applies, which the levels do
  not see (a ring 1.5 cm above a 1000 / 30 Ω·m interface lies between the
  exact thin ring and the solid torus, its levels on either side of the torus; audit W4B-02). Resolved
  concrete converges at first order ($r \approx 2$): there the estimate is
  no bound.
- **Thin wire next to an interface.** The smoothed line source shares the
  thin-wire limit of the integral engines next to a contrasting interface
  (+9.2e-3 against the solid torus for a ring 3 radii above a 1000 / 30
  Ω·m interface, beyond its estimate; evidence `f10` of stream fw4fem3d,
  audit W4B-02). The dispatch therefore emits `InterfaceContactWarning`
  for a horizontal wire within the reach of an interface (3.9 radii at a
  contrast of 33, audit W4R2A-04; the table of the reach is in the
  warnings reference)
  before the 3-D model runs, as the integral engines do (audit W4A-07);
  the axisymmetric model resolves the wire and does not warn. The 3-D
  model's own guard refuses bare wires closer than about 2.8 radii (the
  disc of the source must lie in one layer).
- **Ports and network.** Every electrode is one port, every leaking piece
  of a galvanic distributed conductor one more; `_network.solve_network`
  takes the pipeline's topology of those pieces, the rest is section 3.
- **Concrete (user decision Q40).** By default each encased wire carries
  its shell coefficient $C$ (the model of the image family and `mom`).
  With `FemOptions.concrete_resistivity` = $\rho_c$ the concrete is
  resolved: a square prism of conformal radius $r_b$ (side
  $r_b/0.59017$) around every straight axis-aligned encased wire, the
  metal radius following from $C = \frac{\rho_c}{2\pi}\ln(r_b/r_a)$, the
  prism faces as immersed interfaces, so overlapping shells at corners,
  T-joints and crossings share their concrete. The square keeps the
  resistance to the soil, but in the shell-dominated limit its concrete is
  3.8 % less resistive than the circular shell: the concrete is set by the
  square's inner conformal radius, $0.914\,r_b$ (audit W4B-04; one side
  cannot match both radii, so the prism is not resized). Together with the
  first order of the prism faces and edges the resolved concrete reads the
  conductance per metre 2.8 … 4.6 % high at the default levels, so it
  measures differences on one mesh. Against the per-wire shells a junction
  acts like a few cm of wire, at most about $2\,r_b$, sign not settled — the
  junction numbers are not converged (one refinement of the prism faces
  moved the corner by about 5 cm; evidence `e09`, `e11` of stream w4fem3d,
  audit W4B-05) — and the house mesh of the generators changes by about 1 %
  or less (0.35 … 0.85 % with every sign aligned after the fix of W4B-01,
  evidence `f12` of stream fw4fem3d). The per-wire shells keep their place
  as the default (no correction rule). A cover so thin that the smoothed
  source the prism allows does not reach beyond the metal (energy radius
  at most 1.2 metal radii: a 5 mm bar with 5 mm cover) is refused before
  meshing, naming `concrete_resistivity=None` (audit W4R2B-04; up to then a
  bare `ValueError` after meshing).
- **Resource guard.** Before anything is meshed: the display segments
  against `Engine.max_segments`, the number of ports against
  `FemOptions.max_ports_3d` (default 100; every
  port is one conjugate-gradient solve per level, 0.93 … 0.96 s per port
  and level at 583 000 nodes, auditor W4B and evidence `f14` of stream
  fw4fem3d; audit W4B-08) and `FemOptions.max_nodes_3d` (2 million nodes
  of the finer mesh; the solve adds about 2.9 GB per million, evidence
  `f15`) against a prediction; the exact count again before the finer
  level. `metadata["fem_peak_memory_gb"]` is what the
  solve added to the peak resident memory of the process,
  `fem_process_peak_memory_gb` that peak (audit W4B-09).
- **Face pairing.** The immersed elements' face terms pair the faces by
  their sorted node triples (`numpy.lexsort`); a packed integer key
  aliased above $2^{21}$ nodes (audit W4B-06).
- **Dependencies (decision D6).** `scikit-fem` (BSD-3: the Kuhn background
  `MeshTet.init_tensor`, the red refinement) and `pyamg` (MIT), as the
  optional `fem3d` extra; `import groundfield` imports neither. Without
  the extra a world that needs the 3-D model raises `ImportError` naming
  the extra. No GPL or LGPL mesher (gmsh, TetGen, CGAL, netgen) is used:
  the design needs no resolved surface, so nothing beyond permissive
  packages was required.

## Consequences

- Every `fem` number changes (both stored golden cases by −4.6 %; the
  benchmark tables of the fem page). `fem` agrees with independent
  references to about 1e-4 (within 5.1e-5 for a ring wire within 1.05
  radii of an interface, up to a contrast of 1000) at a cost of a second per solve
  ([fem page](../explanation/engines/fem.md#validation)).
- `fem` solves every world again: the axisymmetric model where it applies
  (about 1e-4), the 3-D model otherwise (with the optional `fem3d` extra;
  about 2e-3 away from layer interfaces, wires that run beside each other
  kept at least $1.5\,R$ apart, which refines the mesh or refuses the
  closest pairs); without the extra the 3-D worlds raise `ImportError`.
- `fem` gains concrete shells, junctions, two-terminal and voltage sources
  and every conductor model (registry capabilities).
- `FemOptions` changes: `refinement`, `extrapolate`, `r_far_factor`
  (default 1000), `far_field`, `max_nodes`; for the 3-D model `model`,
  `source_radius`, `growth`, `concrete_resistivity`, `max_nodes_3d`,
  `max_ports_3d`.
  The 2026.11 fields `n_radial`,
  `n_axial`, `z_far_factor` are gone; as keyword arguments of `solve_fem`
  (0.15.0) they warn and have no effect. `equivalent_hemisphere_radius` is
  deprecated; `FemReductionWarning` is kept and never emitted.
- The 3-D model shares the thin-wire assumption of the integral engines
  (uniform current around the wire, no end faces); it is independent of
  them in everything else (the layering, the surface, the far field, the
  current distribution along the wires). Its accuracy (about 2e-3 away
  from interfaces and other wires, the wires beside each other at least
  $1.5\,R$ apart, levels not yet asymptotic for every geometry; next to a
  strongly contrasting interface the thin-wire error
  of the integral engines) is a reference for the integral engines'
  biases of 0.4 … 3 %, not for effects below 1e-3; differences of two
  concrete models on the same mesh (Q40) are finer than that.
- No stored number changes with part B: the golden `fem` cases are
  axisymmetric and stay on part A bit for bit.
- `fem` stays experimental until the audits of the release have passed.

## Alternatives considered

- **Thin-wire line sources in the mesh with a radius correction**
  (Peaceman-type well index). Cheaper meshes, but the correction depends
  on the mesh, and the end faces and the contact with an interface — the
  configurations where the reduction failed — are exactly what a line
  source cannot represent. A resolved surface costs only logarithmically
  many lines on a graded mesh.
- **Extending the spherical-shell mesh of the reduction.** It conforms to a
  hemisphere, not to rods and rings.
- **An unstructured mesher.** Not needed in 2-D; in 3-D the
  smoothed-source formulation needs no surface-fitted mesh, and the
  bisection mesh of section 5 is built with scikit-fem's tensor grid and
  an own refinement (decision D6: no GPL / LGPL mesher).
- **3-D alternatives tried and dropped (stream w4fem3d).** A graded tensor
  grid (slab anisotropy, multigrid stalls); quadratic elements on the
  bisection mesh (the multigrid did not converge in 500 iterations);
  laminate interfaces (first order); a mesh cut along the interface planes
  (slivers where thin layers cross the far elements); plain immersed
  elements without face terms (not consistent: −1.6e-4 … −6.4e-4 on a
  layered cube that the penalised form reproduces exactly); a fixed source
  quadrature coarser than the finer mesh (the finite elements resolve the
  individual points: the rod's level 2 overshot by +0.6 %); forming
  $K + A^\mathsf T D A$ for the multigrid (3.4 GB instead of 1.8 GB at
  600 000 nodes). Singularity subtraction with the analytic segment
  potential (Gjerde et al. 2019) would make the method a moment method
  with a finite-element correction in homogeneous soil.
- **Resolving the concrete as a volume.** Needs the concrete resistivity
  and the metal radius, which the electrode does not carry; for a
  cylindrical shell it is equivalent to the surface conductance.
- **Infinite elements or a mapped layer at the truncation.** The monopole
  Robin condition on a box a thousand electrode lengths away moves the
  result by less than 1e-5 and needs no special elements.

## References

- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission Systems*.
  Dover, ch. 2 and §4.4.
- Givoli, D. (1992). *Numerical Methods for Problems in Infinite Domains*.
  Elsevier.
- Strang, G., & Fix, G. (2008). *An Analysis of the Finite Element
  Method*, 2nd ed. Wellesley-Cambridge Press.
- Richardson, L. F. (1911). The approximate arithmetical solution by finite
  differences of physical problems. *Phil. Trans. R. Soc. A* 210, 307–357.
- Thompson, J. F., Warsi, Z. U. A., & Mastin, C. W. (1985). *Numerical
  Grid Generation*. North-Holland (O-grids and transfinite blocks).
- Rivara, M.-C. (1984). Algorithms for refining triangular grids suitable
  for adaptive and multigrid techniques. *Int. J. Numer. Methods Eng.* 20,
  745–756 (longest-edge bisection).
- Li, Z., Lin, T., & Wu, X. (2003). New Cartesian grid methods for
  interface problems using the finite element formulation. *Numer. Math.*
  96, 61–98.
- Kafafy, R., Lin, T., Lin, Y., & Wang, J. (2005). Three-dimensional
  immersed finite element methods for electric field simulation in
  composite materials. *Int. J. Numer. Methods Eng.* 64, 940–972.
- Lin, T., Lin, Y., & Zhang, X. (2015). Partially penalized immersed
  finite element methods for elliptic interface problems. *SIAM J. Numer.
  Anal.* 53, 1121–1144.
- Dompierre, J., Labbé, P., Vallet, M.-G., & Camarero, R. (1999). How to
  subdivide pyramids, prisms and hexahedra into tetrahedra. *8th Int.
  Meshing Roundtable*, 195–204.
- Vaněk, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
  smoothed aggregation for second and fourth order elliptic problems.
  *Computing* 56, 179–196 (pyamg).
- Pólya, G., & Szegő, G. (1951). *Isoperimetric Inequalities in
  Mathematical Physics*. Princeton (the conformal radius of a square,
  0.59017 times its side).
