# Plot results

This guide draws the standard figures of a grounding study — the
geometry, the surface potential, profiles and cross-sections, the current
sharing and sweeps — with the plot helpers of groundfield. Read it when you
need a first look at a result or a figure for a report.

All plot helpers return matplotlib objects (a `Figure` or an `Axes`) and
are available as flat names (`gf.plot_world`) and in the `gf.plot`
namespace. Potential plots draw the real part of the complex potential by
default and say so in the label; pass `component="imag"`, `"abs"` or
`"phase"` for the other parts.

**Prerequisites:** matplotlib (part of the core install); `plot_world_3d`
draws with matplotlib as well.

## 1. Solve a world to plot

A substation earth (ring and four rods) measured with an auxiliary
electrode 200 m east and a voltage probe at 61.8 %, built with the layout
builder; 1 A at 50 Hz in 100 Ω·m.

```python
import matplotlib.pyplot as plt

import groundfield as gf
from groundfield.generators import LvNetworkLayout

layout = LvNetworkLayout.from_footprints([], substation_xy=(0.0, 0.0))
layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)
layout.add_voltage_probe(inline_fraction=0.618)
world = layout.to_world(soil=gf.HomogeneousSoil(rho=100.0))
result = gf.create_engine(segment_length=0.5, frequencies=[50.0]).solve(world)
print([e.name for e in world.electrodes])
```

<!-- doctest-check
assert abs(layout.measured_grounding_impedance(result) - result.input_impedance()[0]) > 0  # a three-pole set-up
assert layout.verify_current_balance(result)["ok"]
-->

## 2. Plot the geometry

```python
fig = gf.plot_world(world, title="Top view")  # plane="xy" (default)
fig_side = gf.plot_world(world, plane="xz", extent=(-10.0, 15.0, -1.0, 5.0), title="Side view, depth downwards")
ax = layout.plot(figsize=(9, 4))  # the layout view with anchors and the measurement loop
fig_3d = gf.plot_world_3d(world)
```

<!-- doctest-check
from matplotlib.figure import Figure
assert isinstance(fig, Figure) and isinstance(fig_side, Figure)
-->

## 3. Map the surface potential

```python
fig = gf.plot_surface_potential(result, world, extent=(-20.0, 220.0, -40.0, 40.0), n=120)
fig_abs = gf.plot_surface_potential(result, world, extent=(-15.0, 15.0, -15.0, 15.0), n=100, component="abs")
```

<!-- doctest-check
assert fig.axes[-1].get_ylabel().startswith("Re")
assert fig_abs.axes[-1].get_ylabel() == "|φ| in V"
-->

With an auxiliary electrode the potential is strongly negative near it and
only moderately positive around the substation. The layout's own plot
uses a colour scale with equal resolution above and below zero
(`two_slope=True`, the default); `symmetric=True` and `log=True` are the
alternatives:

```python
fig = layout.plot_surface_potential(result, world, padding_m=30.0, n=120, levels=31)
fig_log = layout.plot_surface_potential(result, world, padding_m=30.0, n=120, log=True, cmap="viridis")
```

## 4. Draw profiles and cross-sections

A profile along a line at several depths, the radial decay around an
electrode, and a vertical cross-section through the measurement axis:

```python
fig = gf.plot_potential_profile(result, start=(-30.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), distance=260.0, depths=(0.0, 1.0))
fig_r = gf.plot_potential_radial(result, around="substation_ring_0", world=world, r_max=60.0, depths=(0.0, 0.5, 1.0), log_x=True)
fig_xz = gf.plot_potential_contour(result, world=world, plane="xz", y=0.0, extent=(-20.0, 220.0, 0.0, 30.0), n=120)
```

`plane="xz"` holds `y` fixed, `plane="xy"` holds `z` fixed (depth
positive downwards). Without `extent` the window follows the electrodes.

## 5. Show where the current goes

```python
fig = gf.plot_current_sharing(result, world, by="electrode")
table = gf.electrode_current_table(result, world)
print(table[["name", "abs_I", "share_of_cluster_pct"]].head())
```

<!-- doctest-check
assert abs(table["I_re"].sum()) < 1e-9  # the test current returns through the auxiliary electrode
-->

**How to check it:** the electrode currents of a three-pole set-up add up
to zero, because the test current returns through the auxiliary
electrode; the hidden check of this page asserts it.

## 6. Plot sweeps and convergence studies

`gf.plot_sweep_lines` and `gf.plot_sweep_heatmap` take the table of
`gf.sweep` (see [Run sweeps and Monte Carlo studies](sweeps-and-monte-carlo.md));
`gf.plot_convergence` the table of `gf.convergence_study`
(tutorial [T09](../tutorials/T09_choosing_an_engine_and_convergence.ipynb)):

```python
df = gf.convergence_study(world, gf.create_engine(frequencies=[50.0]), segment_lengths=[0.5, 0.25, 0.125])
fig = gf.plot_convergence(df, response="abs_Z")
plt.close("all")
```

<!-- doctest-check
assert list(df["segment_length_m"]) == [0.5, 0.25, 0.125]
-->

## Tips

- Every helper takes `frequency_index` for multi-frequency results.
- Save a figure with `fig.savefig("name.png", dpi=150)`; in scripts call
  `plt.show()` or close figures with `plt.close("all")` in loops.
- For publication plots, use the helpers for a first look and then plot
  the numbers yourself: `result.potential(points)` returns the complex
  potential at any `(M, 3)` array of points.

## See also

- Tutorial [T07](../tutorials/T07_fall_of_potential_measurement.ipynb):
  the fall-of-potential measurement behind the example world.
- [Export to CSV, VTK and ParaView](export-csv-vtk-paraview.md): the same
  quantities as files, for other tools.
- [Postprocess API](../reference/api/postprocess.md): every plot helper
  with its options.
