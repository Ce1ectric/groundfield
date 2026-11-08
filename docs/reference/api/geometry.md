# Geometry

The `geometry` package describes the buried metal: parametric
electrodes (rod, ring, strip, polyline, star, mesh, grid mesh), the
leakage-free `Junction` node, and the discretisation of every electrode
into straight wire segments that the solvers consume.

| Kind | Class | Parameters (m) | Typical use |
|---|---|---|---|
| `rod` | `RodElectrode` | `position` (top end), `length` | driven rod (Tiefenerder) |
| `ring` | `RingElectrode` | `center`, `radius` | ring earth around a building or substation |
| `strip` | `StripElectrode` | `start`, `end` | horizontal earth strip, counterpoise |
| `polyline` | `PolylineElectrode` | `vertices`, `closed` | foundation earth following a building outline |
| `star` | `StarElectrode` | `center`, `n_arms`, `arm_length`, `orientation_deg` | star earth (Sternerder) |
| `mesh` | `MeshElectrode` | `corner`, `size`, `spacing` | mesh with a given spacing |
| `grid_mesh` | `GridMeshElectrode` | `corner`, `size`, `n_x`, `n_y` | substation grid with a given number of wires |

Every kind also takes `wire_radius` (default 5 mm) and
`concrete_shell_coefficient_ohm_m` (default 0, see below).

## Mathematical and physical context

An electrode is a **thin wire**: its radius $a$ is small against its
length and against the distance to other wires, so the current leaving
the surface can be represented by a current on the wire axis and the
potential is evaluated on the surface (at distance $a$). Each geometry
is cut into straight segments

$$
\Gamma_i = \{\mathbf{x}_i^{(0)} + \xi\,(\mathbf{x}_i^{(1)} - \mathbf{x}_i^{(0)}) : \xi \in [0, 1]\},
\qquad i = 1 \ldots N,
$$

no longer than `Engine.segment_length`. Grid and mesh wires are split
so that wire crossings fall on segment end points, and every wire is
cut at the soil-layer interfaces so that each segment lies in one
layer; every interface cut is kept, however short the piece beyond it,
so a rod tip that reaches a few millimetres into a layer is in contact
with it. How the current is
distributed over the segments — uniform per unit length on each
electrode (image family) or one unknown per segment (`mom` family) — is
a property of the backend, see [Solver](solver.md) and
[source kernels](../../explanation/kernels.md).

**Concrete encasement.** A foundation earth in concrete adds the
radial resistance of the concrete shell. Every electrode kind carries
it as a per-length coefficient

$$
C = \frac{\rho_c}{2\pi}\,\ln\frac{a + t}{a} \quad [\Omega\,\mathrm{m}],
$$

for concrete resistivity $\rho_c$, bare wire radius $a$ and shell
thickness $t$; the electrode's `wire_radius` is then the outer radius
$a + t$ (the soil sees the shell surface), and every segment of length
$\Delta s$ adds the series resistance $C/\Delta s$ of its shell to its
self term. Only the image family (`image`, `image_2layer`) and
`solve_mutual_*` represent a non-zero coefficient; the other backends
reject such a world before they run. The helpers
`concrete_shell_coefficient`, `shell_resistance_from_coefficient` and
`shell_coefficient_from_resistance` convert between the coefficient and
a lumped resistance; the physics is explained in
[concrete shells](../../explanation/concrete-shells.md) and ADR-0012.

**Validity.** Thin-wire approximation ($a \ll$ segment length and
wire spacing); wires must be in the soil ($z > 0$) — the solver rejects
segments in the air and warns about leakage segments that are long
compared with their depth.

## Example

```python
import groundfield as gf
from groundfield.references import dwight1936

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
star = gf.create_electrode(
    world, "star", name="star", center=(0.0, 0.0, 0.5), n_arms=4, arm_length=5.0
)
gf.create_source(world, attached_to="star", magnitude=1.0)
result = gf.create_engine(segment_length=0.5).solve(world)

assert abs(star.total_length() - 20.0) < 1e-12       # four arms of 5 m
R = result.input_impedance()[0].real                  # 10.05 Ω
R_ref = dwight1936.n_point_star(rho=100.0, arm_length=5.0, radius=0.005, depth=0.5, n_arms=4)
assert abs(R - R_ref) / R_ref < 1e-3                  # Dwight (1936), Eqs. 23-26
```

The geometry queries `segments(ds)`, `segment_count(ds)`,
`wire_polylines()`, `bounds()`, `total_length()` and
`min_feature_length()` are available on every electrode without a
solve, for example to check a model before a long computation.

## API reference

::: groundfield.geometry

::: groundfield.geometry.electrodes
    options:
      members:
        - concrete_shell_coefficient
        - shell_resistance_from_coefficient
        - shell_coefficient_from_resistance
