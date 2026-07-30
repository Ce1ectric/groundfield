# 09 — All plots in action

`groundfield.postprocess` ships a complete plot toolkit on top of
the solver result and the world. This page demonstrates the
project's plotting surface end-to-end on one solved problem.

## Reference solve

The reference world: a substation grounding (5 m ring + 3 driven
rods) with a 200 m east Hilfserder, a Spannungssonde halfway in
between, and a $1\;\mathrm{A}$ test current at 50 Hz.

```python
import matplotlib.pyplot as plt
import groundfield as gf
from groundfield.generators import OrtsnetzLayout

layout = OrtsnetzLayout.from_footprints(
    [], substation_xy=(0.0, 0.0),
)
layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)
layout.add_voltage_probe(inline_fraction=0.5)

world = layout.to_world(
    soil=gf.HomogeneousSoil(resistivity=100.0),
    source_magnitude_A=1.0,
    seed=0,
)
engine = gf.create_engine(
    backend="image",
    segment_length=0.5,
    frequencies=[50.0],
)
result = engine.solve(world)
```

The same `result` is the input to every plot below.

## 1. Layout view (`OrtsnetzLayout.plot`)

```python
ax = layout.plot(figsize=(11, 7))
plt.show()
```

The layout plot shows footprints, the substation (orange diamond),
the KVS markers (blue squares), the Hilfserder (purple triangle)
with a dotted return-current line, and the Spannungssonde (lime
star). Pass `foundation_mask=...` to colour houses by their
foundation-electrode penetration.

## 2. Surface potential — TwoSlopeNorm (default)

```python
fig = layout.plot_surface_potential(
    result, world,
    padding_m=60.0, n=180, levels=41,
)
plt.show()
```

`two_slope=True` (default) uses
:class:`matplotlib.colors.TwoSlopeNorm` centred at $\varphi = 0$
so the deep Hilfserder trough (negative potentials) and the small
substation + foundation rise (positive potentials) are both
visible at full colour resolution on an `RdBu_r` diverging
colormap. The contour levels are split half-and-half between the
negative and positive ranges.

## 3. Surface potential — symmetric mode

```python
fig = layout.plot_surface_potential(
    result, world,
    padding_m=60.0, n=180, levels=41,
    symmetric=True,        # symmetric range [-|phi|_max, +|phi|_max]
)
plt.show()
```

Use `symmetric=True` when the potential extremes really are
symmetric (mutual-coupling studies, net-zero injection setups).
On a strongly asymmetric range (deep Hilfserder trough vs small
foundation rise) the symmetric mode collapses the small side into
a single colour step — that's exactly why `two_slope` is the
default.

## 4. Surface potential — log scale on $|\varphi|$

```python
fig = layout.plot_surface_potential(
    result, world,
    padding_m=300.0, n=180, levels=41,
    log=True,             # log |phi|, sequential cmap recommended
    cmap="viridis",
)
plt.show()
```

`log=True` switches to a log-magnitude colour scale that resolves
the boundary decay across several decades. Useful for verifying
that the potential reaches remote earth at the chosen `padding_m`.

## 5. Radial-decay profile

```python
from groundfield.postprocess import plot_potential_radial

fig = plot_potential_radial(
    result, around="substation_ring_0", world=world,
    r_max=80.0, depths=(0.0, 0.5, 1.0),
)
plt.show()
```

The classical trumpet curve $\varphi(r)$ measured along the
$+x$ ray from the substation centre, at three depths. Cross-
checks the solver result against the analytic single-electrode
Sunde formula.

## 6. Cross-section in the $x$-$z$ plane

```python
from groundfield.postprocess import plot_potential_contour

fig = plot_potential_contour(
    result, world=world,
    plane="xz", y=0.0,
    extent=(-50.0, 220.0, -1.0, 30.0),
    n=200,
)
plt.show()
```

Vertical slice at $y = 0$ — useful for visualising the current
path through the soil between substation and Hilfserder. The slice
plane is selected with `plane`, and the coordinate held fixed is the
one the plane does not span: `plane="xz"` takes `y=...`, `plane="xy"`
takes `z=...`. Omitting `extent` derives the window from the bounding
box of the point sources.

## 7. World-level matplotlib geometry

```python
from groundfield.postprocess import plot_geometry

fig = plot_geometry(world, figsize=(11, 7))
plt.show()
```

A bare 2-D geometry plot of every electrode and conductor in the
world — without any solver result. Useful for sanity-checking the
generator output before launching a long solve.

## Where to go next

- Build a complete OSM-driven world and run the gallery on it —
  see [08 — OSM pipeline with simulation results](08_osm_pipeline.md).
- Use the same plots to sweep penetration $p$ + Hilfserder
  distance $D$ — see
  [07 — Comparison of measurement distances](07_measurement_distance.md).
