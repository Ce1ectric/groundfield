# Factories

The `create_*` functions are the notebook-friendly way to build a
model: each one constructs an object, registers it in the `World` and
returns it. They are thin wrappers around the model classes
(`gf.RodElectrode(...)` plus `world.add_electrode(...)` is equivalent),
with two additions: the parameters of every electrode kind are typed
(IDE completion, type-checker errors for a misspelled kind or keyword),
and unknown names raise with the closest valid name ("did you mean
…?").

| Factory | Creates | Physical meaning |
|---|---|---|
| `create_world(soil=...)` | `World` | container of the physical model; the soil is the conducting half-space |
| `create_electrode(world, kind, ...)` | electrode (`rod`, `ring`, `strip`, `polyline`, `star`, `mesh`, `grid_mesh`) | metal in contact with the soil; leaks current |
| `create_conductor(world, start=, end=, ...)` | `Conductor` | connection between two nodes: ideal, lumped resistance, or a distributed line with resistance, inductance and optional leakage |
| `create_junction(world, position=)` | `Junction` | node without soil contact (cable joint, instrument terminal) |
| `create_source(world, attached_to=, ...)` | `CurrentSource` | impressed current between a node and remote earth or a second node (`return_to`) |
| `create_engine(...)` | `Engine` | the numerics: backend, frequencies, segment length, kernel, earth-return model |
| `run_simulation(world, engine)` | `FieldResult` | the same as `world.solve(engine)` and `engine.solve(world)` |

## Mathematical and physical context

The factories assemble the boundary-value problem that every backend
solves: Laplace's equation $\nabla\cdot(\sigma\nabla\varphi) = 0$ in the
layered soil (`soil`), the electrode surfaces as equipotential
boundaries per galvanic cluster (`create_electrode`), Kirchhoff's laws
for the conductor network between the nodes (`create_conductor`,
`create_junction`), and the impressed currents as the right-hand side
(`create_source`). The `Engine` decides how the problem is discretised
and solved; the `World` holds only physics and can be solved by several
engines for a cross-check.

Coordinates are $(x, y, z)$ in metres with $z$ the depth below the soil
surface (positive downwards); every other quantity is in SI units (see
[Units and conventions](../units-and-conventions.md)).

## Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
rod = gf.create_electrode(world, "rod", name="g1", position=(0.0, 0.0, 0.0), length=1.5)
ring = gf.create_electrode(world, "ring", name="g2", center=(10.0, 0.0, 0.8), radius=2.0)
joint = gf.create_junction(world, name="J1", position=(5.0, 0.0, 0.7))  # cable joint
gf.create_conductor(world, name="l1", start=rod, end=joint, cross_section=50e-6)
gf.create_conductor(world, name="l2", start=joint, end=ring, cross_section=50e-6)
gf.create_source(world, name="s1", attached_to="g1", magnitude=10.0)
engine = gf.create_engine(backend="image", frequencies=[50.0, 150.0])
result = gf.run_simulation(world, engine)

assert type(ring).__name__ == "RingElectrode"
assert len(result.input_impedance("s1")) == 2   # one value per frequency
```

`create_electrode(world, "rnig", ...)` raises
`ValueError: Unknown electrode kind 'rnig'. Did you mean 'ring'?`;
a keyword that the chosen kind does not take raises `TypeError` with the
list of valid keywords. A junction that no conductor connects to an
electrode has no defined potential and is rejected when the world is
solved.

## API reference

::: groundfield.api
    options:
      members:
        - create_world
        - create_electrode
        - create_conductor
        - create_junction
        - create_source
        - create_engine
        - run_simulation
        - ElectrodeKind
        - SourceKind
        - KernelKind
        - PointLike
        - ConductorEnd
