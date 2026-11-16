# API reference

This part of the reference documents every public class, function,
constant and type of `groundfield`, generated from the docstrings with
[mkdocstrings](https://mkdocstrings.github.io/). Use the map below to
find the page of a name. Every physical parameter is documented with
its unit; the conventions shared by all modules are collected in
[Units and conventions](../units-and-conventions.md). How the pieces fit
together is explained in [Concepts](../../explanation/concepts.md) and
shown in the [tutorials](../../tutorials/index.md).

## Map of the pages

| Page | Module | What is there |
|---|---|---|
| [Factories](factories.md) | `groundfield.api` | the `create_*` functions and `run_simulation`, with the literal types of their arguments |
| [World](world.md) | `groundfield.world` | `World`, the container of soil, electrodes, junctions, conductors and sources |
| [Soil](soil.md) | `groundfield.soil` | homogeneous, two-layer and multi-layer soil; `effective_resistivity`, `penetration_depth` |
| [Geometry](geometry.md) | `groundfield.geometry` | the seven electrode kinds, `Junction`, the wire segments, the concrete-shell coefficients |
| [Conductors](conductors.md) | `groundfield.conductors` | `Conductor` (PEN, cable screen, lead), skin effect, wire-proximity checks |
| [Sources](sources.md) | `groundfield.sources` | `CurrentSource`, `VoltageSource` and the `Source` union |
| [Boundary](boundary.md) | `groundfield.boundary` | `BoundaryConditions`: remote earth, insulating surface |
| [Solver](solver.md) | `groundfield.solver` | `Engine`, `FieldResult`, mutual-impedance matrices, the backend solve functions, the solver warnings |
| [Backend registry](registry.md) | `groundfield.solver.registry` | `available_backends`, `register_backend`, capabilities, backend options |
| [Coupling](coupling.md) | `groundfield.coupling` | Neumann inductance, Carson and Sommerfeld earth return, two-layer Green's function |
| [Postprocess](postprocess.md) | `groundfield.postprocess` | touch and step voltage, current sharing, sweeps, `rho-f` fits, plots |
| [Diagnostics](diagnostics.md) | `groundfield.diagnostics` | `diagnose` and the pre-solve checks, problem-size limits |
| [Validation](validation.md) | `groundfield.validation` | `compare_engines`: one world, several engines, one table |
| [Result files](results-io.md) | `groundfield.io.results` | `save_result`, `load_result`, `world_fingerprint` |
| [IO and groundinsight](io.md) | `groundfield.io` | BusType export to groundinsight, CSV with provenance, VTK |
| [Geo / OSM](geo.md) | `groundfield.geo` | OpenStreetMap building footprints, the local projection |
| [Generators](generators.md) | `groundfield.generators` | TN low-voltage networks, distributions, specs, measurement set-ups |
| [References](references.md) | `groundfield.references` | closed forms of Dwight, IEEE 80, Oeding, Carson and Pollaczek |
| [Utilities](utils.md) | `groundfield.utils` | logging, name suggestions, deprecation and inert-setting warnings |

Every warning category and exception is documented on the page of the
module that defines it; [Warnings and errors](../warnings.md) explains
what to do when one appears.

## Top-level namespace

`import groundfield as gf` loads the data model, the factories and the
engine. Everything else is imported on first use, so the import stays
light (no pandas, matplotlib or most of scipy until a function needs
them). Every name of `groundfield.__all__` is available under its flat
name (`gf.touch_voltage`) and is the same object as the one of its home
module (`groundfield.postprocess.safety.touch_voltage`), documented
once, on the page of that module. The post-processing API is also
grouped in curated namespaces:

| Namespace | Content | Page |
|---|---|---|
| `gf.plot` | potential maps and profiles, world geometry, current sharing, sweeps, convergence | [Postprocess](postprocess.md) |
| `gf.safety` | touch and step voltage, touch points (`touch_voltage_at_part`), EN 50522 permissible touch voltage | [Postprocess](postprocess.md) |
| `gf.fit` | `rho-f` models: vector fitting and the research standard form | [Postprocess](postprocess.md) |
| `gf.io` | `BusType` export to groundinsight, CSV, VTK, result files | [IO](io.md), [Result files](results-io.md) |

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

## What counts as public

The public API is the `__all__` list of `groundfield` and of every
module whose dotted path has no part that starts with an underscore.
`tests/test_docs_api_coverage.py` checks that each of these names is
rendered on one of the pages above and has a docstring summary, that
every public function documents its parameters and its return value,
and that every public method has a docstring. Modules with a leading
underscore (`groundfield.solver._kernels`, …) are internal and are not
documented here.

## Package version

::: groundfield.__version__
    options:
      heading_level: 3
