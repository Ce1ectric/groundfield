# World

`groundfield.World` is the container of the physical model: the soil,
the electrodes, the leakage-free junction nodes, the conductors between
them, the sources and the boundary conditions. It holds no numerics;
backend, frequencies and discretisation belong to the
[`Engine`](solver.md), and `world.solve(engine)` runs one against the
other.

## Mathematical and physical context

A world describes one boundary-value problem of the quasi-static
grounding field ($f < 1\,\mathrm{kHz}$):

- in the soil, $\nabla\cdot(\sigma\nabla\varphi) = 0$ with
  $\varphi \to 0$ at remote earth and no current through the soil
  surface;
- every **galvanic cluster** (electrodes joined by ideal conductors) is
  one equipotential body, and the current it leaks is the integral of
  the current density over its surface;
- the **conductor network** (conductors with impedance, junctions)
  obeys Kirchhoff's current law at every node and
  $\varphi_a - \varphi_b = Z_b I_b$ along every branch;
- the **sources** impress currents between a node and remote earth or
  between two nodes.

The names of electrodes and junctions share one namespace: they are the
nodes that conductors and sources refer to.

## Checks before a solve

`World.check()` validates the model before any backend runs: every
source terminal and conductor end must name an existing node (the error
suggests the closest names), the two terminals of a source must not be
bonded into one cluster, and a world needs at least one electrode and
one current source with a finite, non-zero magnitude. A
`VoltageSource` raises `NotImplementedError` (no backend implements an
impressed voltage). Every solve calls the check; the solver pipeline
also rejects junctions without a conductor path to an electrode. The
[diagnostics](diagnostics.md) module adds advisory checks (problem size,
segment resolution, close wires).

## Sources are restored after a solve

`world.solve(engine)` works on a snapshot of the sources: the source
objects returned by `create_source` stay attached to the world, so a
parameter loop can edit them between solves.

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(world, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
src = gf.create_source(world, attached_to="g1", magnitude=10.0)
engine = gf.create_engine(frequencies=[50.0])

potentials = []
for current in (10.0, 20.0, 40.0):
    src.magnitude = current            # src is still world.sources[0]
    potentials.append(world.solve(engine).electrode_potentials["g1"][0])

assert abs(potentials[2] / potentials[0] - 4.0) < 1e-12   # linear problem
```

## API reference

::: groundfield.world
