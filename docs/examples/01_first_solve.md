# Example 01 — Your first solve

A single driven rod in homogeneous soil. The simplest possible
grounding study: build a world by hand (no generator yet), call
the solver, plot the radial decay of the potential. By the end
of the page you'll know what every object in `groundfield` does.

## What you'll see

* How to construct a `World` from scratch.
* How `Engine.solve(world)` produces a `FieldResult`.
* How to read off the **grounding impedance** of an electrode and
  the **potential** at any point in the soil.
* The classical "Spannungstrichter" (potential funnel) around a
  driven rod, plotted against radial distance.

## Code (copy-paste ready)

```python
import matplotlib.pyplot as plt
import numpy as np

import groundfield as gf

# 1. Soil — homogeneous, 100 Ω·m. The simplest model; the layered
#    variants come in example 03.
soil = gf.HomogeneousSoil(resistivity=100.0)

# 2. World — the container that holds soil, electrodes, conductors,
#    and the source(s).
world = gf.create_world(name="rod_demo", soil=soil)

# 3. One driven rod, 1.5 m long, head at the surface. ``position`` is
#    (x, y, z) in metres; z is the depth below the surface, so z=0
#    means "rod head right at the surface".
gf.create_electrode(
    world, "rod", name="g1",
    position=(0.0, 0.0, 0.0),
    length=1.5,
    wire_radius=0.008,
)

# 4. Source — 1 A test current attached to the rod. ``return_to``
#    defaults to None, which means "current returns through remote
#    earth" (the textbook fall-of-potential setup).
gf.create_source(world, attached_to="g1", magnitude=1.0)

# 5. Engine — image method, single frequency 50 Hz. ``segment_length``
#    controls the discretisation; 0.05 m is plenty for a 1.5 m rod.
engine = gf.create_engine(
    backend="image",
    segment_length=0.05,
    frequencies=[50.0],
)

# 6. Solve! ``result`` is a FieldResult: it knows the segment
#    currents, the grounding impedance per cluster, and can evaluate
#    the potential at any point in space.
result = engine.solve(world)

# --- read off the grounding impedance ---
Z = result.cluster_impedance("g1")[0]
print(f"|Z(50 Hz)| = {abs(Z):.3f} Ω    arg = {np.angle(Z, deg=True):+.2f}°")

# Compare against the analytical Sunde / Dwight formula for a single
# driven rod of length L with rod radius r in homogeneous soil:
#   R_rod = (rho / (2 pi L)) * (ln(4L/r) - 1)
L = 1.5
r = 0.008
R_sunde = (soil.resistivity / (2 * np.pi * L)) * (np.log(4 * L / r) - 1)
print(f"Sunde reference: R = {R_sunde:.3f} Ω")

# --- plot the radial potential profile at the surface ---
fig = gf.plot_potential_radial(
    result, around="g1", world=world,
    r_max=20.0, n=200,
    depths=[0.0, 0.5, 1.0],
)
plt.show()
```

## Expected output

```
|Z(50 Hz)| = 39.815 Ω    arg = +0.00°
Sunde reference: R = 39.789 Ω
```

The image method matches Sunde's closed form to better than
0.1 % — the rod is short enough that the discrete-segment
representation is essentially exact. The plot shows the typical
trumpet shape: the potential is highest right at the rod and
decays roughly as $1/r$ at large radii.

## What just happened

`World` is a Pydantic data model. It carries:

* the **soil** (the conductivity profile $\sigma(z)$),
* a list of **electrodes** (rods, rings, strips, mesh
  electrodes — see [`api/geometry`](../api/geometry.md)),
* a list of **conductors** that bond electrodes together
  (see [`api/conductors`](../api/conductors.md)),
* a list of **sources** that inject current.

`Engine` is the numerical solver. Its `solve(world)` step
discretises every electrode into ≤ `segment_length` chunks,
builds the dense reaction matrix, and solves the resulting
linear system.

`FieldResult` is the output. The two methods you'll use most:

* `result.cluster_impedance(name)` — returns a list of
  complex impedances, one per frequency. The "cluster" is the
  set of galvanically connected electrodes — for a stand-alone
  rod, just the rod itself.
* `result.potential(points, frequency_index=0)` — evaluates
  $\varphi(\mathbf{r})$ at arbitrary 3-D points, used by every
  plotting helper.

## Try this next

* Sweep the rod length: replace step 3 with a loop over
  `length ∈ {0.5, 1.0, 1.5, 3.0}` m and compare `R` against the
  Sunde formula.
* Add a second rod at $(5, 0, 0)$. Without bonding them, you
  have two independent clusters — `cluster_impedance` returns
  the same value for both.
* Bond them with `gf.create_conductor(world, start="g1",
  end="g2", conductor_type="bare_copper")`. Now they form one
  cluster and the parallel-rod formula applies. Compare against
  Dwight 1936's "two parallel rods" formula in
  `gf.dwight1936.parallel_rods`.

That last step is the gentle ramp into the substation example
(02), which scales the same idea up to a ring + four rods.
