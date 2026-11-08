# Reference

Look-up material: what a function takes and returns, which units and
sign conventions apply, what a warning means, which names are
deprecated, and what the package does not (yet) do. For the reasoning
behind the methods see [Explanation](../explanation/index.md); to learn
the package step by step see the [Tutorials](../tutorials/index.md).

## API

The [API reference](api/index.md) is generated from the docstrings
(NumPy style) with mkdocstrings. Each page starts with the
mathematical and physical context of the module, then lists its public
classes and functions.

| Page | Content |
|---|---|
| [Factories](api/factories.md) | `create_world`, `create_electrode`, `create_conductor`, `create_junction`, `create_source`, `create_engine`, `run_simulation` |
| [World](api/world.md), [Soil](api/soil.md), [Geometry](api/geometry.md), [Conductors](api/conductors.md), [Sources](api/sources.md), [Boundary](api/boundary.md) | the physical model |
| [Solver](api/solver.md), [Backend registry](api/registry.md) | `Engine`, `FieldResult`, multi-port matrices, backends and their capabilities |
| [Coupling](api/coupling.md) | Neumann inductance, Carson and Sommerfeld earth return, layered Green's function |
| [Postprocess](api/postprocess.md) | potentials, touch and step voltages, current sharing, sweeps, `rho-f` fits, plots |
| [Diagnostics](api/diagnostics.md), [Validation](api/validation.md) | checks before and after a solve |
| [Result files](api/results-io.md), [IO and groundinsight](api/io.md) | versioned result files, `BusType` export, CSV, VTK |
| [Geo / OSM](api/geo.md), [Generators](api/generators.md) | worlds from OpenStreetMap and from parameter sets |
| [References](api/references.md) | closed-form reference formulas used in the validation (Dwight, IEEE 80, Carson, Pollaczek, Oeding) |

## Conventions and catalogues

- [Glossary (EN / DE)](glossary.md) — terms of the package and their
  German counterparts (Ortsnetz, KVS, Fundamenterder, …).
- [Units and conventions](units-and-conventions.md) — SI units,
  coordinates, phasors, potential reference and signs, frequency range.
- [Warnings and errors](warnings.md) — every warning and exception
  class with its cause and remedy.
- [Known limitations](known-limitations.md) — what is approximate,
  experimental or not implemented, with the evidence.
- [Deprecations](deprecations.md) — deprecated names and parameters,
  their replacements and the earliest removal.
