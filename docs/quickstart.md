# Quickstart

```python
import groundfield as gf

# 1) Soil model: typical two-layer soil from the AP1 parameter space
soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)

# 2) Build a world and add a ring electrode around a substation
world = gf.create_world(soil=soil)
gf.create_electrode(
    world,
    "ring",
    name="g1",
    center=(0.0, 0.0, 0.8),
    radius=5.0,
    wire_radius=0.005,
)
gf.create_source(world, attached_to="g1", magnitude=1.0)

# 3) Configure the engine
engine = gf.create_engine(
    backend="image",                # auto-routes to image_2layer
    segment_length=0.05,
    frequencies=[50.0, 150.0, 250.0, 350.0],
)

# 4) Run the simulation
result = world.solve(engine)

# 5) Evaluate
Z = result.cluster_impedance("g1")             # input impedance per frequency
import numpy as np
xs = np.linspace(0.5, 30.0, 120)
phi = result.potential(np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)]))

# 6) Plots
gf.plot_potential_radial(result, around="g1", world=world,
                         depths=[0.0, 0.5, 1.0])
gf.plot_potential_contour(result, world=world, plane="xy", z=0.0)
```

## Backends

| Backend          | Soil model                                            | Method                                                                  | Status     |
|------------------|-------------------------------------------------------|-------------------------------------------------------------------------|------------|
| `image`          | `HomogeneousSoil`                                     | image-charge sum, closed form                                           | implemented |
| `image_2layer`   | `TwoLayerSoil` (auto-dispatched from `"image"`)       | Tagg/Sunde geometric image-charge series                                | implemented |
| `image_nlayer`   | `HomogeneousSoil`, `TwoLayerSoil`, `MultiLayerSoil`   | dispatcher (delegates to `image` for $n=1$, `image_2layer` for $n=2$)   | implemented |
| `cim`            | any layered                                           | Complex Image Method (matrix-pencil fit of $\Gamma_1(\lambda)$)         | implemented |
| `mom`            | `HomogeneousSoil` or `TwoLayerSoil`                   | Galerkin Method-of-Moments on segment level                             | implemented |
| `mom_sommerfeld` | any layered                                           | Galerkin MoM with direct Sommerfeld quadrature (reference engine)        | implemented |
| `bem`            | any layered                                           | Boundary-element collocation with the CIM kernel                        | implemented |
| `fem`            | any layered                                           | Axisymmetric volume FEM with equivalent-hemisphere reduction            | implemented |

`Engine.solve` automatically forwards `backend="image"` to
`image_2layer` for a `TwoLayerSoil` and to `image_nlayer` for a
`MultiLayerSoil`, so notebooks written for the homogeneous case keep
working when only the soil model is swapped. The full engine theory
lives under [Engine theory](engines/index.md); the selection
heuristic is recorded in
[ADR-0002](adr/0002-engine-family.md).

## Cross-engine validation

```python
from groundfield import compare_engines

report = compare_engines(
    world,
    engines={
        "fine":   gf.create_engine(backend="image", segment_length=0.025),
        "coarse": gf.create_engine(backend="image", segment_length=0.10),
    },
    rel_tolerance=0.05,
)
print(report.summary())
assert report.is_consistent
```

The same pattern is used by `tests/test_cross_engines_extended.py`
to enforce the methodological cross-checks between the closed-form
image engines, the integral-equation engines (`mom`,
`mom_sommerfeld`, `bem`) and the volume-PDE engine (`fem`).

## Where this fits in work package 1

- Vary `soil.rho_1`, `soil.rho_2`, `soil.h_1` and the auxiliary
  electrode position to span the AP1 parameter space.
- The resulting `rho-f` curves feed the model reduction step in
  `groundinsight`.
