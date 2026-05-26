# 08 — OSM pipeline with simulation results

The :class:`OrtsnetzLayout` builder turns a real
OpenStreetMap building extract into a fully populated
:class:`groundfield.World` with a few self-documenting calls. This
example runs the full pipeline on a small rural village:

1. Query OSM at a chosen lat/lon centre + radius.
2. Drop the substation + KVS at user coordinates (GPS or ENU
   metres).
3. Route PEN cables in strict Manhattan geometry around the
   foundation polygons.
4. Connect every house to the closest cable.
5. Pick a reproducible foundation-electrode penetration.
6. Drop the Hilfserder + Spannungssonde for a simulated
   fall-of-potential measurement.
7. Materialise + solve.
8. Read the simulated meter reading, verify Kirchhoff balance,
   and plot the surface potential.

## Requirements

The OSM ingest path needs the optional `geo` extra:

```bash
pip install groundfield[geo]
# or, from a Poetry checkout
poetry install --extras geo
```

This pulls in `requests`, `shapely` and `pyproj`. The Overpass
response is cached locally so subsequent runs stay offline.

## End-to-end code

```python
import groundfield as gf
from groundfield.generators import OrtsnetzLayout

# 1. OSM ingest + anchor placement, all in one call. Pick the
#    village centre lat/lon; substation_xy / kvs_xys are in local
#    ENU metres relative to the centre. Use
#    substation_lat_lon / kvs_lat_lons if you prefer GPS picks.
layout = OrtsnetzLayout.from_osm(
    center_lat_deg=51.9136, center_lon_deg=10.8432,
    radius_m=200.0,
    substation_xy=(0.0, -12.0),
    kvs_xys=[(40.0, -12.0), (80.0, -12.0)],
    obstacle_clearance_m=1.0,
    min_area_m2=20.0,
)

# 2. Lay PEN cables Manhattan-style around the foundations.
layout.add_pen_cable(start='substation', end='kvs_0',
                    min_segment_length_m=8.0)
layout.add_pen_cable(start='kvs_0', end='kvs_1',
                    min_segment_length_m=8.0)
layout.add_pen_cable(start='substation', end_xy=(0.0, 60.0),
                    min_segment_length_m=8.0)

# 3. Connect every house to its closest cable.
layout.connect_buildings()

# 4. Reproducible foundation-electrode penetration.
PENETRATION = 0.30
mask = layout.foundation_mask(PENETRATION)

# 5. Add the simulated fall-of-potential measurement loop.
layout.add_auxiliary_electrode(distance_m=200.0, direction_deg=0.0)
layout.add_voltage_probe(inline_fraction=0.5)   # 50 % rule

# 6. Materialise + solve.
world = layout.to_world(
    foundation_mask=mask,
    source_magnitude_A=1.0,
    seed=0,
)
engine = gf.create_engine(
    backend='image', segment_length=0.5, frequencies=[50.0],
)
result = engine.solve(world)

# 7. Plausibility check + meter reading.
balance = layout.verify_current_balance(result)
assert balance['ok'], balance
Z = layout.measured_grounding_impedance(result, source_magnitude_A=1.0)
print(f"Closed measurement loop, |Z_meas| = {abs(Z):.3f} ohm "
      f"(R = {Z.real:+.3f}, X = {Z.imag:+.3f})")
```

## What the helpers do

* `OrtsnetzLayout.from_osm` runs the
  :func:`groundfield.geo.osm.query_and_project` pipeline, projects
  the substation + KVS through the resulting `Projector` and
  stores the WGS84 origin on the layout so later GPS-based methods
  (`add_voltage_probe(position_lat_lon=...)`, `lat_lon_to_xy`,
  `xy_to_lat_lon`) stay consistent.
* `add_pen_cable` routes a PEN cable from any anchor to another
  anchor or a free coordinate using a 4-connected Manhattan-grid
  A* that avoids every building's bounding rectangle (inflated by
  `obstacle_clearance_m`). The ``escape_radius_m`` knob lets the
  cable physically exit the substation cubicle through nearby
  foundations — it defaults to one grid cell of slack.
* `connect_buildings` attaches every house to the closest cable
  via a short axis-aligned stub (1 or 2 segments). Houses with no
  obstacle-free stub emit a `UserWarning` and stay unconnected.
* `foundation_mask(p)` is deterministic and nested in $p$. Same
  $(p,\,\text{salt})$ always returns the same houses, and
  increasing $p$ only grows the foundation-equipped subset.
* `add_auxiliary_electrode` drops the Hilfserder (default geometry
  = 3 × 0.5 m parallel-bonded rods in a 0.5 m triangle); the
  matching `to_world` step injects an opposite-sign return current
  at the aux anchor so the engine sees a physically closed
  measurement loop.
* `add_voltage_probe` drops the Spannungssonde as a pure sampling
  point — no rod is materialised in the world, so the ideal-
  voltmeter assumption is preserved.
* `measured_grounding_impedance(result, probe_xy=...)` returns
  $(\varphi_\text{sub} - \varphi_\text{probe}) / I_\text{src}$ at
  the requested frequency.
* `verify_current_balance(result)` aggregates leakage currents
  on both sides of the loop and asserts they sum to
  $\pm I_\text{src}$ within tolerance.

## Visualising the result

```python
import matplotlib.pyplot as plt

# Layout view: footprints, anchors, cables, service drops, probe.
ax = layout.plot(foundation_mask=mask, figsize=(13, 7))
plt.show()

# Surface potential with TwoSlopeNorm so the deep Hilfserder
# trough and the small substation + foundation rise are both
# visible at full colour resolution.
fig = layout.plot_surface_potential(
    result, world,
    padding_m=80.0, n=180, levels=41,
)
plt.show()
```

See the [plot gallery](09_plot_gallery.md) for the full set of
postprocess helpers and the colour-axis modes (`two_slope`,
`symmetric`, `log`).

## Where to go next

- Sweep penetration $p$ + Hilfserder distance $D$ on the same OSM
  layout and trace the bias as the measurement loop expands — see
  [07 — Comparison of measurement distances](07_measurement_distance.md).
- Use the layout's `xy_to_lat_lon` helper to cross-check the
  Hilfserder coordinate against a satellite-imagery base map.
