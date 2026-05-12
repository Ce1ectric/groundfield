# API reference

The API reference is generated directly from the source-code
docstrings via
[`mkdocstrings`](https://mkdocstrings.github.io/). Each subpage
corresponds to one subpackage.

- [Soil](soil.md) — soil models.
- [Geometry](geometry.md) — electrode and conductor geometries.
- [Conductors](conductors.md) — conductors, PEN, cable shields.
- [Solver](solver.md) — numerical field solver.
- [Coupling](coupling.md) — galvanic and inductive coupling.
- [Postprocess](postprocess.md) — potentials, voltages, currents,
  plots.
- [Diagnostics](diagnostics.md) — pre-solve structural checks
  (`world_statistics`, `expected_segments`,
  `check_segment_resolution`).
- [IO](io.md) — JSON export and the bridge to `groundinsight`.
- [Generators](generators.md) — world-generator framework
  (`TnNetworkGenerator`, distributions, spec layer; ADR-0009).

## Top-level package

The most relevant classes and factory functions are re-exported on
the package level:

```python
import groundfield as gf

soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
world = gf.create_world(soil=soil)
gf.create_electrode(world, "ring", name="g1",
                    center=(0.0, 0.0, 0.8), radius=5.0)
gf.create_source(world, attached_to="g1", magnitude=1.0)
result = gf.create_engine(backend="image",
                          frequencies=[50.0]).solve(world)
```
