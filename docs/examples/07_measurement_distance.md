# 07 — Comparison of measurement distances on a synthetic grounding system

A fall-of-potential test is the classical field method to measure
the grounding impedance of a substation: inject a known current
$I$ between the substation and an auxiliary current electrode
(Hilfserder) placed at some distance $D$, then measure the
substation potential against a voltage probe (Spannungssonde)
placed *between* the substation and the Hilfserder (or 90° from
the current-feed axis). The reading is biased by the
$U_\text{probe}$ contribution; the bias vanishes only as $D$ grows
toward "remote earth".

This example builds a synthetic grounding world, runs the
measurement at several Hilfserder distances *and* two probe
geometries (0° inline vs 90° perpendicular), and shows the bias
disappearing as the loop expands.

## Build the synthetic world

```python
import groundfield as gf
from groundfield.generators import OrtsnetzLayout

# Synthetic footprints lining a single street, plus a substation
# at the west end.
def make_house(cx, cy, dx=10.0, dy=10.0, osm_id=0):
    return gf.BuildingFootprint(
        polygon_xy_m=[
            (cx - dx / 2.0, cy - dy / 2.0),
            (cx + dx / 2.0, cy - dy / 2.0),
            (cx + dx / 2.0, cy + dy / 2.0),
            (cx - dx / 2.0, cy + dy / 2.0),
        ],
        building_use="residential",
        osm_id=osm_id,
    )

footprints = [make_house(40.0 + 25.0 * i,
                         +20.0 if i % 2 == 0 else -20.0,
                         osm_id=i)
              for i in range(8)]
layout = OrtsnetzLayout.from_footprints(
    footprints,
    substation_xy=(0.0, 0.0),
    obstacle_clearance_m=1.0,
)
layout.add_pen_cable(start="substation", end_xy=(250.0, 0.0),
                    min_segment_length_m=8.0)
layout.connect_buildings()
```

## Configure the measurement loop

`layout.add_auxiliary_electrode(distance_m=D, direction_deg=0.0)`
drops the Hilfserder (default geometry: a 3 × 0.5 m rod triangle,
parallel-bonded) at radius $D$ east of the substation. The
matching `layout.to_world(...)` then wires a second current source
with `phase_deg=180` at the Hilfserder anchor so the engine sees a
proper closed-loop pair (the v0.7 image solver ignores
`Source.return_to`, so the workaround injects the return current
explicitly).

## Sweep over $D$ with two probe positions

```python
import math
import numpy as np

distance_sweep = [25.0, 50.0, 75.0, 100.0, 150.0, 200.0,
                  300.0, 500.0, 800.0]
PENETRATION = 0.30
SOURCE_MAGNITUDE_A = 1.0

sx, sy = layout.substation_xy
theta = math.radians(0.0)                    # Hilfserder direction = east
u_inline = (math.cos(theta), math.sin(theta))
u_perp = (-u_inline[1], u_inline[0])         # 90 deg CCW

z_inline, z_perp = [], []
fixed_mask = layout.foundation_mask(PENETRATION)
engine = gf.create_engine(backend="image",
                          segment_length=0.5,
                          frequencies=[50.0])

for D in distance_sweep:
    layout.add_auxiliary_electrode(distance_m=D, direction_deg=0.0)
    world = layout.to_world(
        foundation_mask=fixed_mask,
        source_magnitude_A=SOURCE_MAGNITUDE_A, seed=0,
    )
    result = engine.solve(world)

    half_D = 0.5 * D
    probe_inline = (sx + half_D * u_inline[0],
                    sy + half_D * u_inline[1])
    probe_perp = (sx + half_D * u_perp[0],
                  sy + half_D * u_perp[1])

    z_inline.append(layout.measured_grounding_impedance(
        result, source_magnitude_A=SOURCE_MAGNITUDE_A,
        probe_xy=probe_inline,
    ))
    z_perp.append(layout.measured_grounding_impedance(
        result, source_magnitude_A=SOURCE_MAGNITUDE_A,
        probe_xy=probe_perp,
    ))

for D, zi, zp in zip(distance_sweep, z_inline, z_perp):
    delta = abs(zp) - abs(zi)
    rel = 100.0 * delta / abs(zp) if abs(zp) else 0.0
    print(f"D = {D:5.0f} m  |Z_inline| = {abs(zi):7.3f}  "
          f"|Z_perp| = {abs(zp):7.3f}  delta = {delta:+.3f} ({rel:+.1f} %)")
```

## What you should see

The two curves both converge to the same asymptote (the *true*
substation grounding impedance) as $D$ grows. For small $D$ the
inline probe sits in the substation's near field and reads too
low, while the perpendicular probe stays closer to remote earth
and reads closer to the asymptote. Above roughly $D = 5 \cdot
r_\text{sub}$ (where $r_\text{sub}$ is the largest dimension of
the substation grounding) the bias drops below a few percent for
both placements.

## Reproducibility and physics check

* `layout.foundation_mask(p)` is **deterministic** in $p$ and the
  footprint OSM id — the same penetration $p = 0.30$ always picks
  the same houses, and increasing $p$ only adds houses to the
  equipped subset.
* `layout.measured_grounding_impedance(result, probe_xy=...)` reads
  $\varphi_\text{sub} - \varphi_\text{probe}$ from a single solve
  — both probe geometries share the same engine call so the
  comparison is apples-to-apples.
* `layout.verify_current_balance(result)` is the Kirchhoff
  plausibility check: positive side (substation + KVS + foundation
  electrodes) sums to $+I_\text{src}$, auxiliary side sums to
  $-I_\text{src}$.

## Where to go next

- Drive the same workflow on a real OSM extract — see
  [08 — OSM pipeline with simulation results](08_osm_pipeline.md).
- Plot the surface potential of the measurement loop with the new
  `TwoSlopeNorm` colormap — see
  [09 — All plots in action](09_plot_gallery.md).
