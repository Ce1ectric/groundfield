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
| `MultiLayerSoil`            | ≥ 1         | `image_nlayer` dispatcher, `cim` ($n \le 2$), `mom_sommerfeld`, `fem` |

`backend="image"` auto-dispatches to the matching layered backend:
a `TwoLayerSoil` ends up on `image_2layer`, a `MultiLayerSoil`
with $n = 1$ collapses to `image`, $n = 2$ routes to
`image_2layer`, $n \ge 3$ raises a clear `ValueError` directing
the user to `mom_sommerfeld` or `fem`. `cim` shares that limit:
its closed-form self-kernel is exact for $n \le 2$ and rejects
$n \ge 3$ with a `NotImplementedError` (see
[ADR-0002](../adr/0002-engine-family.md) and the
[`cim` engine page](../engines/cim.md)), because for three or more
layers the reflection coefficient $\Gamma_1(\lambda)$ is no longer
constant in $\lambda$ and the historic complex-image expansion was
structurally incomplete.

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

For $n \ge 3$ build a `MultiLayerSoil` from explicit `SoilLayer`
entries — one per layer, top-down, and the last one semi-infinite
(`thickness=None`, which is the default) — and solve it with the
`mom_sommerfeld` reference backend:

```python
soil = gf.MultiLayerSoil(
    layers=[
        gf.SoilLayer(resistivity=80.0, thickness=3.0),   # topsoil
        gf.SoilLayer(resistivity=30.0, thickness=5.0),   # moist sand
        gf.SoilLayer(resistivity=200.0),                 # bedrock, h = inf
    ],
)
world = gf.create_world(soil=soil)
gf.create_electrode(world, "ring", name="ring_1",
                    center=(0.0, 0.0, 0.8), radius=4.0,
                    wire_radius=0.005)
gf.create_source(world, attached_to="ring_1", magnitude=1.0)

engine = gf.create_engine(backend="mom_sommerfeld", segment_length=0.5,
                          frequencies=[50.0])
result = engine.solve(world)
print(result.cluster_impedance("ring_1"))
```

`mom_sommerfeld` evaluates the full layered Green's function by
direct Sommerfeld quadrature, so it carries no layer-count
restriction and is the absolute multi-layer reference. `fem` is
the cheap alternative for $n \ge 3$ (axisymmetric equivalent
hemisphere, so useful for compact, roughly rotationally symmetric
electrodes only). `cim` and the `image` family stay restricted to
$n \le 2$; use them for the two-layer sweeps above, where `cim`
re-uses its spectral fit across frequencies and the second and
following frequency points come almost for free.

## Where to go next

- Confirm that the layered backends agree on a single problem
  via [02 — Comparison of the solver engines](02_engine_comparison.md).
- Build a complete grounding system on top of a layered soil — see
  [04 — Interconnected grounding system](04_interconnected_grounding.md).
- Inspect the resulting potential field on the layered soil — see
  [09 — All plots in action](09_plot_gallery.md).
