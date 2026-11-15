# ADR-0016: A conforming finite-element model in `fem`

| | |
|---|---|
| **Status** | Accepted — part A (axisymmetric model) 2026-11-15; part B (full 3-D model, user decision Q40) amends this record |
| **Date** | 2026-11-15 |
| **Deciders** | Project maintainers (user decision Q37 of 2026-11-14) |
| **Scope** | `groundfield.solver.fem` (now a package), the `fem` entry of the backend registry and `FemOptions` |

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
  cut into the block's elements, so every element lies in one layer.
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
  the level-0 lines $\ell$ times (nested tensor spaces: the resistance of
  one level converges monotonically from below). The default solves levels
  0 and 1 and extrapolates the port conductance matrix with the observed
  second order (Richardson); the relative difference between the
  extrapolated and the finer level is reported as
  `metadata["fem_error_estimate"]`.

Every other world is refused with `BackendCapabilityError`, naming the
backends that solve it: electrodes other than rods and rings, a rod or
ring off the common axis, galvanic distributed conductors (their leakage
runs along their route), touching electrodes and a ring wire that reaches
within 5 % of its radius to the soil surface (the solid torus must fit).

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

## Consequences

- Every `fem` number changes (both stored golden cases by −4.6 %; the
  benchmark tables of the fem page). `fem` agrees with independent
  references to about 1e-4 at a cost of a second per solve
  ([fem page](../explanation/engines/fem.md#validation)).
- `fem` refuses worlds it used to solve with the reduction: strips,
  grids, polylines, stars, several clusters side by side — until part B.
- `fem` gains concrete shells, junctions, two-terminal and voltage sources
  and every conductor model (registry capabilities).
- `FemOptions` changes: `refinement`, `extrapolate`, `r_far_factor`
  (default 1000), `far_field`, `max_nodes`. The 2026.11 fields `n_radial`,
  `n_axial`, `z_far_factor` are gone; as keyword arguments of `solve_fem`
  (0.15.0) they warn and have no effect. `equivalent_hemisphere_radius` is
  deprecated; `FemReductionWarning` is kept and never emitted.
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
- **An unstructured mesher.** Not needed in 2-D; the permissive candidates
  for 3-D are part B's question (decision D6).
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
