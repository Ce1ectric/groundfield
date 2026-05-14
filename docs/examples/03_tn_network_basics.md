# Example 03 — TN-Ortsnetz generator basics

The previous examples built worlds by hand. For typical you'll
typically have a substation, dozens of houses, several cable
cabinets and a PEN backbone connecting all of them — that's
where `TnNetworkGenerator` comes in. One config object, one
`build()` call, fully populated `World`.

## What you'll see

* The minimal `TnNetworkConfig` and what it produces.
* How to use the building-type catalog (residential / small
  industry / medium industry / large industry).
* A 2-layer soil specification.
* A surface-potential plot of the whole network.

## Code

```python
import matplotlib.pyplot as plt

import groundfield as gf
from groundfield.generators import (
    ManhattanGridPlacement,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
)

# 1. Configuration. Everything has sensible defaults; ``building_counts``
#    is the only field that essentially must be set.
cfg = TnNetworkConfig(
    # Layout: 30 single-family houses on a Manhattan grid
    placement=ManhattanGridPlacement(
        spacing_x_m=20.0, spacing_y_m=25.0, n_per_row=6,
    ),
    building_counts={"residential": 30},
    # Two-layer soil — default
    soil=TwoLayerSoilSpec(rho_1=100.0, rho_2=50.0, h_1=5.0),
)

# 2. Build the world. ``seed=0`` makes the world bit-exact reproducible
#    across runs (matters once distributions are involved; see ex 07).
gen = TnNetworkGenerator(cfg, seed=0)
world = gen.build()
print(f"electrodes: {len(world.electrodes):3d}    "
      f"conductors: {len(world.conductors):3d}    "
      f"sources: {len(world.sources)}")

# 3. Solve. Use ``image_2layer`` since the soil has two layers; pass
#    ``segment_length=1.0`` for production-grade convergence (see performance
#    guide).
engine = gf.create_engine(
    backend="image_2layer",
    segment_length=1.0,
    frequencies=[50.0],
)
result = engine.solve(world)

Z = result.cluster_impedance("trafo_ring_0")[0]
import numpy as np
print(f"trafo cluster |Z(50 Hz)| = {abs(Z):.3f} Ω    "
      f"arg = {np.angle(Z, deg=True):+.2f}°")

# 4. Surface-potential plot — covers the whole world plus 15 m of
#    "remote earth" around the bounding box.
fig = gf.plot_surface_potential(
    result, world,
    z=0.0, padding_m=15.0, n=240,
    title=f"30 EFH TN-Ortsnetz, ρ₁=100, ρ₂=50, h₁=5 m, |Z|={abs(Z):.2f} Ω",
)
plt.show()
```

## What the generator built for you

For the configuration above, the generator instantiated:

* a substation at $(0, 0)$ with a ring earth electrode (radius
  4 m) plus 4 driven rods bonded into it (the default
  substation grounding),
* 30 foundation electrodes on the Manhattan grid, each a
  $10 \times 10$ m grid mesh ($n_x = n_y = 2$ — one
  cross-brace each axis),
* 2 cable cabinets along $y = 0$ (default quota: 5 per 100
  buildings, rounded up; 2 for 30 EFH),
* a PEN backbone connecting substation → each cable cabinet,
  and each cable cabinet → its nearest house, modelled as
  distributed insulated conductors,
* a 1 A source attached to the substation, returning through
  remote earth.

All of that is in **one config object**. Switching the topology
(more houses, finer soil layering, different placement
strategy, different building types) is now a one-line change
to `cfg`.

## Inspecting the catalog

The four default building types are returned by
`default_building_catalog()`. Have a look:

```python
from groundfield.generators import default_building_catalog
for bt in default_building_catalog():
    print(f"{bt.name:18s}  electrodes: "
          f"{[e.kind for e in bt.grounding.electrodes]}")
```

Expected:

```
residential       electrodes: ['foundation']
small_industry    electrodes: ['foundation', 'rod']
medium_industry   electrodes: ['foundation', 'rod', 'rod', 'rod', 'rod']
large_industry    electrodes: ['ring', 'foundation', 'rod', ..., 'strip', 'strip']
```

Mix and match in `building_counts`:

```python
cfg = TnNetworkConfig(
    building_counts={
        "residential": 25,
        "small_industry": 4,
        "medium_industry": 1,
    },
    placement=ManhattanGridPlacement(...),
)
```

## Switching the soil model

The `soil=` field accepts any of three discriminated specs:

```python
from groundfield.generators import (
    HomogeneousSoilSpec, TwoLayerSoilSpec,
    MultiLayerSoilSpec, SoilLayerSpec,
)

# Homogeneous (sanity baseline):
cfg.soil = HomogeneousSoilSpec(resistivity=100.0)

# default:
cfg.soil = TwoLayerSoilSpec(rho_1=100.0, rho_2=50.0, h_1=5.0)

# Multi-layer (e.g. a measurement campaign with three identifiable
# strata):
cfg.soil = MultiLayerSoilSpec(layers=[
    SoilLayerSpec(resistivity=300.0, thickness_m=2.0),
    SoilLayerSpec(resistivity=100.0, thickness_m=5.0),
    SoilLayerSpec(resistivity=30.0,  thickness_m=None),  # semi-infinite
])
```

## Try this next

* Tune `placement` to `ExplicitPlacement(positions=[(x, y), ...])`
  to read coordinates from a CSV or hand-drawn map slice.
* Switch the residential foundation from mesh to ring-style
  using `FoundationElectrodeSpec(style="ring")` — see the
  variant catalog in `notebooks/20_tn_ortsnetz_generator.ipynb`.
* For the actual measurement question (how the substation
  impedance reads when probed from outside), continue with
  example 04.
