# API reference

Generated from the source docstrings with
[mkdocstrings](https://mkdocstrings.github.io/). Every physical
parameter is documented with its unit; the conventions shared by all
modules are collected in [Units and conventions](../units-and-conventions.md).

## Top-level namespace

`import groundfield as gf` loads the data model, the factories and the
engine. Everything else is imported on first use, so the import stays
light (no pandas, matplotlib or most of scipy until a function needs
them). Every public function is available under its flat name
(`gf.touch_voltage`) and, for the post-processing API, also in a
curated namespace:

| Namespace | Content |
|---|---|
| `gf.plot` | potential maps and profiles, world geometry, current sharing, sweeps, convergence |
| `gf.safety` | touch and step voltage, touch points (`touch_voltage_at_part`), EN 50522 permissible touch voltage |
| `gf.fit` | `rho-f` models: vector fitting and the research standard form |
| `gf.io` | `BusType` export to groundinsight, CSV, VTK, result files |

```python
import groundfield as gf
from groundfield.references import dwight1936

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(world, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
gf.create_source(world, attached_to="g1", magnitude=1.0)
result = gf.create_engine(frequencies=[50.0]).solve(world)

Z = result.input_impedance()[0]                      # 35.99 Ω
R_dwight = dwight1936.rod(rho=100.0, length=3.0, radius=0.005)
assert abs(Z.real - R_dwight) / R_dwight < 1e-3      # Dwight (1936): 35.99 Ω
assert gf.safety.touch_voltage is gf.touch_voltage   # same function, two names
```

## Pages

| Page | Modules |
|---|---|
| [Factories](factories.md) | `groundfield.api` |
| [World](world.md) | `groundfield.world` |
| [Soil](soil.md) | `groundfield.soil` |
| [Geometry](geometry.md) | `groundfield.geometry` |
| [Conductors](conductors.md) | `groundfield.conductors` |
| [Sources](sources.md) | `groundfield.sources` |
| [Boundary](boundary.md) | `groundfield.boundary` |
| [Solver](solver.md) | `groundfield.solver.engine`, `.result`, `.mutual` |
| [Backend registry](registry.md) | `groundfield.solver.registry` |
| [Coupling](coupling.md) | `groundfield.coupling` |
| [Postprocess](postprocess.md) | `groundfield.postprocess` |
| [Diagnostics](diagnostics.md) | `groundfield.diagnostics` |
| [Validation](validation.md) | `groundfield.validation` |
| [Result files](results-io.md) | `groundfield.io.results` |
| [IO and groundinsight](io.md) | `groundfield.io.groundinsight`, `.csv`, `.vtk` |
| [Geo / OSM](geo.md) | `groundfield.geo` |
| [Generators](generators.md) | `groundfield.generators` |
| [References](references.md) | `groundfield.references` |
| `groundfield.utils` | name suggestions and the warning helpers, see [Deprecations](../deprecations.md) and [Warnings and errors](../warnings.md) |
