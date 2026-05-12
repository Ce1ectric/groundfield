# Geometry

The ``geometry`` subpackage describes the physical shape of every
buried metallic object — rods, rings, strips, mesh and grid-mesh
electrodes — together with the wire-segment representation that the
field solver consumes.

## Physical background

A grounding electrode is a finite metal body buried in conducting
soil. For frequencies below 1 kHz it can be treated as a thin-wire
boundary along which the potential is enforced and the current
distribution is unknown (or, in the cheaper image backends,
prescribed as uniform). All standard geometries are reduced to a
collection of straight cylindrical segments

$$
\Gamma_i = \{ \mathbf{x}(\xi) = \mathbf{x}_i^{(0)}
                  + \xi\,(\mathbf{x}_i^{(1)} - \mathbf{x}_i^{(0)}) :
                  \xi \in [0, 1] \},\qquad i = 1\ldots N,
$$

each carrying a wire radius $a_i$ and a material tag. The
discretisation length $\Delta s$ is set by
:attr:`groundfield.solver.engine.Engine.segment_length`. The
geometry layer is intentionally decoupled from the solver, so the
same electrodes can be fed into the closed-form image backend, MoM,
or FEM without modification.

## Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))

# Vertical driven rod
rod = gf.create_electrode(
    world, "rod", name="g1",
    position=(0.0, 0.0, 0.0), length=1.5, wire_radius=0.005,
)

# Buried ring earth at house-foundation depth
ring = gf.create_electrode(
    world, "ring", name="g2",
    center=(10.0, 0.0, 0.8), radius=2.0, wire_radius=0.005,
)

# Rectangular mesh (e.g. substation grid)
mesh = gf.create_electrode(
    world, "grid_mesh", name="g3",
    center=(0.0, 0.0, 0.5), size=(20.0, 20.0),
    n_x=11, n_y=11, wire_radius=0.005,
)

# Pre-solve resolution / segment-budget check
for warning in gf.check_segment_resolution(
    world, gf.create_engine(backend="image", segment_length=0.1)
):
    print("WARN:", warning)
```

For programmatic generation of large AP1 worlds (5 – 200
single-family houses, stochastic electrode mixes) see
[Generators](generators.md).

## API reference

::: groundfield.geometry
