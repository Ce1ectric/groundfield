# 01 — Quickstart: two single electrodes

The shortest possible end-to-end run: define a homogeneous soil,
place a rod and a ring electrode, attach a current source, pick a
solver backend, and read the cluster impedance. About 15 lines of
code; nothing optional.

## Physical setup

| Quantity                       | Value                |
|--------------------------------|----------------------|
| Soil model                     | Homogeneous, $\rho = 100\;\Omega\,\mathrm{m}$ |
| Electrode 1 — driven rod       | Head at $(0, 0, 0.5)$ m, length 1.5 m |
| Electrode 2 — horizontal ring  | Centre at $(20, 0, 0.8)$ m, radius 3 m |
| Source                         | $I = 1\;\mathrm{A}$ at the rod, remote-earth return |
| Frequencies                    | $50\;\mathrm{Hz}$ |
| Solver backend                 | `image` (homogeneous Green's function) |

The two electrodes are not bonded — each is its own galvanic
cluster — so the result reports two independent grounding
impedances.

## Code

```python
import groundfield as gf

# 1. World with a homogeneous-soil model.
world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))

# 2. Two electrodes.
gf.create_electrode(
    world, "rod", name="rod_1",
    position=(0.0, 0.0, 0.5), length=1.5, wire_radius=0.01,
)
gf.create_electrode(
    world, "ring", name="ring_1",
    center=(20.0, 0.0, 0.8), radius=3.0, wire_radius=0.005,
)

# 3. Current source on the rod, returning through remote earth.
gf.create_source(world, attached_to="rod_1", magnitude=1.0)

# 4. Solve.
engine = gf.create_engine(
    backend="image",                       # homogeneous image method
    segment_length=0.5,                    # discretisation length, m
    frequencies=[50.0],
)
result = engine.solve(world)

# 5. Read the cluster impedances.
Z_rod = result.cluster_impedance("rod_1")[0]
Z_ring = result.cluster_impedance("ring_1")[0]
print(f"Z_rod  = {Z_rod.real:7.3f}  + {Z_rod.imag:+7.3f}j  Ohm")
print(f"Z_ring = {Z_ring.real:7.3f}  + {Z_ring.imag:+7.3f}j  Ohm")
```

## What to expect

For a 1.5 m driven rod in 100 Ω·m soil the closed-form value
(Sunde) is $R \approx 47\,\Omega$. For a 3 m ring at $z = 0.8$ m
the trumpet-curve value is around $9\,\Omega$. The two clusters
are uncoupled at this distance (mutual term well below 1 % of
either self-term), so the printed values agree with the analytical
single-electrode formulas to plotting accuracy.

## Where to go next

- Compare the answer of the `image` backend against `image_2layer`,
  `mom` and `bem` on the same geometry — see
  [02 — Comparison of the solver engines](02_engine_comparison.md).
- Replace the homogeneous soil with a layered profile — see
  [03 — Multi-layer soil models](03_multilayer_soil.md).
- Bond the two electrodes with a bare-copper strip so they share a
  common potential — see [04 — Interconnected grounding system](04_interconnected_grounding.md).
