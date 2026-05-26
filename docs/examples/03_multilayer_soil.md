# 03 — Using multi-layer soil models

Real soil rarely is homogeneous. A two- or three-layer profile —
often inferred from Wenner or Schlumberger soundings — already
captures the dominant axial variation of $\rho$ and is enough for
most low-frequency grounding studies. `groundfield` ships three
soil models with auto-dispatching backends.

## Soil model classes

| Class                       | Layer count | Used by backends           |
|-----------------------------|-------------|----------------------------|
| `HomogeneousSoil`           | 1           | `image`, `mom`, `bem`, `fem` |
| `TwoLayerSoil`              | 2           | `image_2layer`, `mom`        |
| `MultiLayerSoil`            | ≥ 1         | `image_nlayer` dispatcher, `cim`, `mom_sommerfeld` |

`backend="image"` auto-dispatches to the matching layered backend:
a `TwoLayerSoil` ends up on `image_2layer`, a `MultiLayerSoil`
with $n = 1$ collapses to `image`, $n = 2$ routes to
`image_2layer`, $n \ge 3$ raises a clear `ValueError` directing
the user to `cim` or `mom_sommerfeld`.

## Two-layer setup

A 4 m ring earth electrode sitting in 80 Ω·m topsoil with a much
more conductive 30 Ω·m bedrock 3 m below the surface (the
classical "topsoil over rock" profile).

```python
import groundfield as gf

soil = gf.TwoLayerSoil(rho_1=80.0, rho_2=30.0, h_1=3.0)
world = gf.create_world(soil=soil)
gf.create_electrode(
    world, "ring", name="ring_1",
    center=(0.0, 0.0, 0.8), radius=4.0, wire_radius=0.005,
)
gf.create_source(world, attached_to="ring_1", magnitude=1.0)

engine = gf.create_engine(
    backend="image",                       # auto-dispatch to image_2layer
    segment_length=0.5,
    frequencies=[50.0, 150.0, 250.0],
)
result = engine.solve(world)
print(result.cluster_impedance("ring_1"))
```

## Sweep over the layer contrast

The reflection coefficient between two layers is

$$
K = \frac{\rho_2 - \rho_1}{\rho_2 + \rho_1}.
$$

A small parameter sweep over $K \in [-0.9, +0.9]$ illustrates the
collapse to the homogeneous case at $K = 0$ and the lower / upper
trumpet bounds.

```python
import numpy as np

K_grid = np.linspace(-0.9, +0.9, 9)
Z_per_K: list[complex] = []
for K in K_grid:
    rho_1 = 100.0
    rho_2 = rho_1 * (1.0 + K) / (1.0 - K)
    world = gf.create_world(
        soil=gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=3.0),
    )
    gf.create_electrode(world, "ring", name="ring_1",
                        center=(0.0, 0.0, 0.8), radius=4.0,
                        wire_radius=0.005)
    gf.create_source(world, attached_to="ring_1", magnitude=1.0)
    engine = gf.create_engine(backend="image", segment_length=0.5,
                              frequencies=[50.0])
    Z_per_K.append(engine.solve(world).cluster_impedance("ring_1")[0])
```

The curve is monotonic in $K$ and crosses the homogeneous reference
at $K = 0$ to plotting accuracy — a useful built-in sanity check
for the 2-layer kernel.

## Three or more layers

For $n \ge 3$ use `MultiLayerSoil` with the `cim` (Complex Image
Method) or `mom_sommerfeld` backend:

```python
soil = gf.MultiLayerSoil(layer_resistivities=[80.0, 30.0, 200.0],
                          layer_thicknesses=[3.0, 5.0])
world = gf.create_world(soil=soil)
gf.create_electrode(world, "ring", name="ring_1",
                    center=(0.0, 0.0, 0.8), radius=4.0,
                    wire_radius=0.005)
gf.create_source(world, attached_to="ring_1", magnitude=1.0)

engine = gf.create_engine(backend="cim", segment_length=0.5,
                          frequencies=[50.0])
result = engine.solve(world)
```

`cim` builds a matrix-pencil fit of the spectral Green's function
once and re-uses it at every frequency, so the second and following
frequency points come for free. `mom_sommerfeld` is slower but is
the absolute multi-layer reference — use it to validate `cim` on
new soil profiles before launching long sweeps.

## Where to go next

- Confirm that the layered backends agree on a single problem
  via [02 — Comparison of the solver engines](02_engine_comparison.md).
- Build a complete grounding system on top of a layered soil — see
  [04 — Interconnected grounding system](04_interconnected_grounding.md).
- Inspect the resulting potential field on the layered soil — see
  [09 — All plots in action](09_plot_gallery.md).
