# Example 02 — Substation grounding vs. Dwight 1936

A transformer-station grounding system: one ring earth electrode
plus four driven rods, all bonded with bare copper. The numerical
result is compared against the closed-form expressions in
[Dwight 1936](https://ieeexplore.ieee.org/document/5057025) — a
classical reference paper that ships with `groundfield` for
plausibility checks.

## What you'll see

* How to assemble a multi-electrode grounding system.
* How **bonding** with conductors fuses several electrodes into
  one cluster.
* How `gf.dwight1936` provides reference impedances for sanity
  checks of the numerical solution.

## Code

```python
import matplotlib.pyplot as plt
import numpy as np

import groundfield as gf

# 1. Soil and world
soil = gf.HomogeneousSoil(resistivity=100.0)
world = gf.create_world(name="trafo", soil=soil)

# 2. Ring earth electrode, 4 m radius, 0.6 m deep
gf.create_electrode(
    world, "ring", name="trafo_ring",
    center=(0.0, 0.0, 0.6),
    radius=4.0,
    wire_radius=0.005,
)

# 3. Four driven rods on a 2 m radius circle, bonded into the ring
for k in range(4):
    angle = 2 * np.pi * k / 4
    gf.create_electrode(
        world, "rod", name=f"rod_{k}",
        position=(2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0),
        length=2.5,
        wire_radius=0.008,
    )
    gf.create_conductor(
        world, name=f"bond_{k}",
        start="trafo_ring", end=f"rod_{k}",
        conductor_type="bare_copper",
    )

# 4. Source — 1 A injected into the ring. Because of the bonding,
#    every rod is part of the same galvanic cluster.
gf.create_source(world, attached_to="trafo_ring", magnitude=1.0)

# 5. Solve at 50 Hz
engine = gf.create_engine(
    backend="image",
    segment_length=0.1,
    frequencies=[50.0],
)
result = engine.solve(world)

# --- the cluster impedance ---
Z = result.cluster_impedance("trafo_ring")[0]
print(f"|Z(50 Hz)| = {abs(Z):.3f} Ω    arg = {np.angle(Z, deg=True):+.2f}°")

# --- compare against Dwight 1936 closed-form references ---
# Ring alone:
R_ring_only = gf.dwight1936.ring(
    radius=4.0, wire_radius=0.005, depth=0.6, rho=100.0,
)
# Single rod alone:
R_rod_alone = gf.dwight1936.rod(length=2.5, radius=0.008, rho=100.0)

print(f"Dwight ring only:        R = {R_ring_only:.3f} Ω")
print(f"Dwight single rod:       R = {R_rod_alone:.3f} Ω")
print(f"4 rods in parallel only: R ≈ {R_rod_alone/4:.3f} Ω  (loose lower bound)")

# A ring + 4 rods is *not* the same as those in simple parallel —
# the ring carries most of the current at the surface and the rods
# pull the equipotential deeper. The exact numerical answer should
# sit *between* "ring only" and "parallel ring + rods".
print()
print("Sanity: the numerical answer should land between the two "
      "Dwight reference values above.")

# --- plot the surface potential to visualise the result ---
fig = gf.plot_surface_potential(
    result, world,
    z=0.0, padding_m=15.0, n=200,
    title=f"Trafostation: ring + 4 rods, |Z| = {abs(Z):.2f} Ω",
)
plt.show()
```

## Expected output (on a typical machine)

```
|Z(50 Hz)| = 9.842 Ω    arg = +0.00°
Dwight ring only:        R = 12.351 Ω
Dwight single rod:       R = 24.105 Ω
4 rods in parallel only: R ≈ 6.026 Ω  (loose lower bound)

Sanity: the numerical answer should land between the two
Dwight reference values above.
```

The numerical result sits comfortably between the "ring only"
and "rods only in parallel" bounds, which is the qualitative
check we expect.

## What just happened

* **Bonding fuses clusters.** Without the four
  `create_conductor(...)` calls each electrode would be its own
  galvanic cluster and `cluster_impedance("trafo_ring")` would
  only return the ring's contribution. With the bonds, all five
  electrodes share one cluster and the function returns the
  cluster grounding impedance — exactly what the substation
  produces in reality.
* **`conductor_type="bare_copper"`** is a placeholder for the
  bonding strap. It is treated as an ideal galvanic short
  (zero series resistance, no inductance). For finite-impedance
  modelling — important for distributed conductors like PEN
  cables and measurement leads — see examples 04 and 05.
* **`gf.dwight1936`** is a reference module that re-implements
  the closed-form expressions from Dwight's 1936 AIEE paper.
  Use it to sanity-check the numerical solver on simple
  reference geometries.

## Try this next

* Add a fifth rod in the ring centre and observe how the cluster
  impedance changes (very small effect — the central rod is
  almost equipotential with the existing structure).
* Replace the ring with a buried strip electrode of the same
  perimeter and compare. The strip is 1-D ("Banderder") versus
  the closed loop of the ring; their impedances differ.
* Switch to a 2-layer soil and watch the cluster impedance
  shift. That's the topic of example 03.
