# Example 09 — OSM-driven Ortsnetz with measurement setup

This is the **end-to-end AP1 workflow** on a real (or synthetic)
OpenStreetMap neighbourhood. It exercises every piece of ADR-0011
in one go and stitches them together with the fall-of-potential
measurement setup from example 04:

1. **Read OSM building footprints** (Variant B: live Overpass
   query; Variant A: hand-crafted footprints with the same shape).
2. **Set parameters and distributions for the foundation
   electrodes** — in particular `presence_prob` as a Bernoulli on
   each building (not every house has a `Fundamenterder` according
   to DIN 18014).
3. **Place the substation** within (or just outside) the OSM
   cluster.
4. **Place the auxiliary current electrode** (Hilfserder) for the
   measurement loop — typically 100–300 m away from the substation,
   well outside the network's own potential funnel.
5. **Place the voltage probe** (Spannungssonde) at an arbitrary
   (x, y) location, and read the surface potential there. Two
   ways: as a real probe electrode wired into the world, or by
   evaluating ``result.potential`` at the point post-solve.

Two complete code blocks: one with **pseudo-geometries** (always
runs, no network), one with **live Overpass** (needs
`pip install groundfield[geo]` and internet).

## Variant A — pseudo-geometries

```python
import math

import matplotlib.pyplot as plt
import numpy as np

import groundfield as gf
from groundfield.generators import (
    BuildingTypeSpec,
    Categorical,
    ExplicitPlacement,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    KvsConfig,
    MeasurementInjectionConfig,
    MeasurementProbeConfig,
    MeasurementSetupConfig,
    OsmBuildingPlacement,
    RodElectrodeSpec,
    SubstationConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
    Uniform,
)
from groundfield.geo import BuildingFootprint


# 1) ---- Read (or, here, hand-craft) the building footprints --------
def rotated_rectangle(centre, size, angle_deg):
    cx, cy = centre
    dx, dy = size
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return [
        (c*x - s*y + cx, s*x + c*y + cy)
        for x, y in [(-dx/2, -dy/2), (dx/2, -dy/2), (dx/2, dy/2), (-dx/2, dy/2)]
    ]


buildings = [
    BuildingFootprint(polygon_xy_m=rotated_rectangle((0.0, 10.0), (12.0, 8.0), -15.0)),
    BuildingFootprint(polygon_xy_m=rotated_rectangle((20.0, 11.0), (10.0, 9.0), 10.0)),
    BuildingFootprint(polygon_xy_m=rotated_rectangle((40.0, 12.0), (11.0, 8.0), -5.0)),
    BuildingFootprint(polygon_xy_m=rotated_rectangle((60.0, 11.0), (13.0, 7.0), 15.0)),
    BuildingFootprint(polygon_xy_m=rotated_rectangle((80.0, 10.0), (10.0, 10.0), 25.0)),
]

# 2) ---- Parameters and distributions for the foundation electrodes ----
#
# Two stochastic axes on the building-level grounding spec:
#
# - ``presence_prob`` (Bernoulli per building) — DIN 18014 is only
#   mandatory for new constructions since 2007. In a real Ortsnetz we
#   estimate that 70 % of residential buildings have a foundation
#   electrode; this is the only knob that survives the OMBR override
#   (size_xy_m and orientation_deg get overwritten from the polygon).
# - ``depth_m`` (Uniform on the burial depth) — realistic spread
#   between 0.6 m (early DIN 18014) and 1.2 m (deep Streifenfundament).
#
# Every other geometric field of FoundationElectrodeSpec is irrelevant
# here because TnNetworkGenerator rewrites size_xy_m + orientation_deg
# from the polygon's oriented bounding rectangle (ADR-0011).
residential = BuildingTypeSpec(
    name="residential",
    grounding=GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                style="ring",
                size_m=10.0,                    # placeholder — overridden by OMBR
                depth_m=Uniform(low=0.6, high=1.2),
                presence_prob=0.7,              # Bernoulli per house
            ),
            # Optional add-on: 30 % of houses have an additional driven
            # rod next to the foundation. This survives the override
            # because it is not a FoundationElectrodeSpec.
            RodElectrodeSpec(
                length_m=1.5, depth_m=0.0,
                offset_xy_m=(5.0, 0.0),
                presence_prob=0.3,
            ),
        ],
    ),
)

# 3) ---- Place the substation just outside the cluster ---------------
SUB_XY = (40.0, -25.0)    # 35 m south of the cluster centre
KVS_XY = (40.0, -8.0)     # KVS on the inner edge, on the street

# 4) ---- Place the auxiliary current electrode (Hilfserder) ----------
#
# Classic fall-of-potential geometry: aux electrode 200 m south of
# the substation along the +x = 40 line so we get a clean radial
# voltage decay between substation and aux. ``feed_lead=None`` keeps
# this purely galvanic (no metallic measurement lead → no inductive
# coupling, no Carson correction needed).
AUX_XY = (40.0, -225.0)

# 5) ---- Place the voltage probe (Spannungssonde) --------------------
#
# Classic 62 % rule of thumb: probe sits at 62 % of the substation-aux
# distance. With substation at (40, -25) and aux at (40, -225) that
# is (40, -149). Note: in a connected TN-Ortsnetz the 62 % rule does
# **not** recover the standalone substation cluster impedance — the
# probe ends up reading the *system* impedance (substation + every
# bonded foundation electrode in parallel via the PEN trunk), which
# is precisely the AP1-Analyse-1 question.
PROBE_XY = (40.0, -149.0)

cfg = TnNetworkConfig(
    name="osm_pseudo",
    soil=TwoLayerSoilSpec(rho_1=150.0, rho_2=30.0, h_1=4.0),
    substation=SubstationConfig(position=SUB_XY),
    placement=OsmBuildingPlacement(footprints=buildings, min_area_m2=16.0),
    building_types=[residential],
    building_counts={"residential": len(buildings)},
    kvs=KvsConfig(
        fixed_count=1,
        placement=ExplicitPlacement(positions=[KVS_XY]),
    ),
    measurement=MeasurementSetupConfig(
        injection=MeasurementInjectionConfig(position_xy=AUX_XY),
        probe=MeasurementProbeConfig(position_xy=PROBE_XY),
    ),
    source_magnitude_A=1.0,                  # 1 A → impedances directly in Ω
)

world = TnNetworkGenerator(cfg, seed=0).build()
print(f"world: {len(world.electrodes)} electrodes, "
      f"{len(world.conductors)} conductors")

# Bernoulli draws on presence_prob are visible in the electrode list:
present_foundations = sum(
    1 for e in world.electrodes
    if "_foundation_" in e.name and "_w" not in e.name
)
present_rods = sum(
    1 for e in world.electrodes
    if e.name.startswith("residential_") and "_rod_" in e.name
)
print(f"  realised foundations: {present_foundations} / {len(buildings)} "
      f"(expected ~{0.7 * len(buildings):.1f})")
print(f"  realised extra rods : {present_rods} / {len(buildings)} "
      f"(expected ~{0.3 * len(buildings):.1f})")

# ---- Solve ----------------------------------------------------------
engine = gf.create_engine(
    backend="image_2layer",
    segment_length=0.5,
    frequencies=[50.0],
    earth_inductive_model="perfect_mirror",   # galvanic regime
)
result = engine.solve(world)

# 6) ---- Read the measurement quantities -----------------------------
#
# Two equivalent ways to get the probe potential:
#
# (a) Via the real probe electrode that MeasurementSetupConfig added.
#     The probe is materialised as a 1.5 m rod with anchor
#     ``probe_rod_0`` (see groundfield.generators.measurement defaults).
phi_probe_electrode = result.electrode_potentials["probe_rod_0"][0].real

# (b) Via ``result.potential`` at the (x, y, z=0) coordinate. This is
#     the more general path: it works for *any* point in space, no
#     need to add a real electrode there. Useful when sweeping the
#     probe position post-solve.
phi_probe_point = result.potential(
    np.array([[PROBE_XY[0], PROBE_XY[1], 0.0]])
)[0].real

phi_substation = result.electrode_potentials["trafo_ring_0"][0].real

# Two quantities of interest, and they are *different* in a connected
# TN-Ortsnetz:
#
# - Z_cluster: the substation's own cluster impedance against remote
#   earth — what the substation grounding would have in isolation.
# - Z_system: the *effective* impedance the source actually drives —
#   substation in parallel with every bonded foundation electrode
#   via the PEN trunk. The measurement reads Z_system, not Z_cluster.
Z_cluster = result.cluster_impedance("trafo_ring_0")[0].real
Z_system = (phi_substation - phi_probe_electrode) / cfg.source_magnitude_A

print()
print(f"Z_cluster (substation alone, vs. remote earth) = {Z_cluster:.3f} Ω")
print(f"Z_system  (substation || PEN || foundations)   = {Z_system:.3f} Ω")
print(f"Ratio Z_system / Z_cluster                     = "
      f"{Z_system / Z_cluster:.3f}")
print(f"  -> the parallel foundations + PEN reduce the effective")
print(f"     grounding impedance by a factor "
      f"{Z_cluster / Z_system:.2f}.")
print()
print(f"phi at probe — via probe electrode  = {phi_probe_electrode:+.4f} V")
print(f"phi at probe — via result.potential = {phi_probe_point:+.4f} V")
print("  (the two values agree to within numerical noise)")

# 7) ---- Contour plot of the surface potential -----------------------
fig = gf.plot_surface_potential(
    result, world,
    z=0.0, padding_m=20.0, n=180,
    title="phi(x, y, z=0)  —  AP1 Ortsnetz with measurement loop, 50 Hz",
)
plt.show()
```

## Expected output (Variant A)

The `presence_prob` draws produce a heterogeneous fleet (4/5 houses
got a foundation electrode, 2/5 got the additional rod):

```text
world: 26 electrodes, 23 conductors
  realised foundations: 4 / 5 (expected ~3.5)
  realised extra rods : 2 / 5 (expected ~1.5)

Z_cluster (substation alone, vs. remote earth) = 7.693 Ω
Z_system  (substation || PEN || foundations)   = 1.241 Ω
Ratio Z_system / Z_cluster                     = 0.161
  -> the parallel foundations + PEN reduce the effective
     grounding impedance by a factor 6.20.

phi at probe — via probe electrode  = +0.0310 V
phi at probe — via result.potential = +0.0310 V
  (the two values agree to within numerical noise)
```

The exact numbers depend on the RNG seed (`seed=0` is reproducible).
Re-running with a different seed produces a different fleet of
realised buildings (still around 70 % foundation presence and 30 %
add-on rod presence) and slightly different system impedances —
that variability is **the** result of AP1's parameter study.

## Variant B — live Overpass query

The only change is **where** the footprints come from: instead of
hand-crafted rectangles, we ask Overpass for the buildings within
a radius of a real (lat, lon) and feed them into the same pipeline.
The substation, aux electrode and probe positions are derived from
the cluster geometry rather than hard-coded.

```python
import math

import matplotlib.pyplot as plt
import numpy as np

import groundfield as gf
from groundfield.generators import (
    BuildingTypeSpec, ExplicitPlacement, FoundationElectrodeSpec,
    GroundingSystemSpec, KvsConfig,
    MeasurementInjectionConfig, MeasurementProbeConfig,
    MeasurementSetupConfig,
    OsmBuildingPlacement, RodElectrodeSpec, SubstationConfig,
    TnNetworkConfig, TnNetworkGenerator, TwoLayerSoilSpec, Uniform,
)
from groundfield.geo import query_and_project


# 1) ---- Read live OSM data ------------------------------------------
LAT0, LON0 = 52.227, 11.011       # Helmstedt-Stadtmitte (change me!)
RADIUS_M = 120.0
MIN_AREA_M2 = 40.0                # filter Gartenhäuser
MAX_BUILDINGS = 12                # cap solver cost

footprints_live, projector = query_and_project(
    lat0_deg=LAT0, lon0_deg=LON0,
    radius_m=RADIUS_M, min_area_m2=MIN_AREA_M2,
)
fps_used = sorted(
    footprints_live, key=lambda fp: -fp.area_m2()
)[:MAX_BUILDINGS]
print(f"Overpass returned {len(footprints_live)} buildings; "
      f"using the {len(fps_used)} largest.")

# 2) ---- Same stochastic foundation spec as Variant A ----------------
residential = BuildingTypeSpec(
    name="residential",
    grounding=GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                style="ring", size_m=10.0,
                depth_m=Uniform(low=0.6, high=1.2),
                presence_prob=0.7,
            ),
            RodElectrodeSpec(
                length_m=1.5, depth_m=0.0,
                offset_xy_m=(5.0, 0.0),
                presence_prob=0.3,
            ),
        ],
    ),
)

# 3) ---- Substation just outside the cluster on the south side -------
all_x = [p[0] for fp in fps_used for p in fp.polygon_xy_m]
all_y = [p[1] for fp in fps_used for p in fp.polygon_xy_m]
centre_x = 0.5 * (max(all_x) + min(all_x))
y_south = min(all_y) - 25.0
SUB_XY = (centre_x, y_south)
KVS_XY = (centre_x, min(all_y) - 8.0)

# 4) ---- Aux electrode 200 m further south of the substation ---------
AUX_XY = (centre_x, y_south - 200.0)

# 5) ---- Probe at the 62 % point between substation and aux ----------
PROBE_XY = (centre_x, y_south - 0.62 * 200.0)

cfg = TnNetworkConfig(
    name="osm_live",
    soil=TwoLayerSoilSpec(rho_1=150.0, rho_2=30.0, h_1=4.0),
    substation=SubstationConfig(position=SUB_XY),
    placement=OsmBuildingPlacement(
        footprints=fps_used, min_area_m2=MIN_AREA_M2,
    ),
    building_types=[residential],
    building_counts={"residential": len(fps_used)},
    kvs=KvsConfig(
        fixed_count=1,
        placement=ExplicitPlacement(positions=[KVS_XY]),
    ),
    measurement=MeasurementSetupConfig(
        injection=MeasurementInjectionConfig(position_xy=AUX_XY),
        probe=MeasurementProbeConfig(position_xy=PROBE_XY),
    ),
    source_magnitude_A=1.0,
)

world = TnNetworkGenerator(cfg, seed=0).build()
print(f"world: {len(world.electrodes)} electrodes, "
      f"{len(world.conductors)} conductors")

# Slightly coarser discretisation because live worlds usually have
# more buildings than the synthetic demo.
engine = gf.create_engine(
    backend="image_2layer",
    segment_length=0.7,
    frequencies=[50.0],
    earth_inductive_model="perfect_mirror",
)
result = engine.solve(world)

# 6) ---- Read measurement quantities ---------------------------------
phi_substation = result.electrode_potentials["trafo_ring_0"][0].real
phi_probe_electrode = result.electrode_potentials["probe_rod_0"][0].real

# Standalone vs. system impedance — same distinction as in Variant A.
Z_cluster = result.cluster_impedance("trafo_ring_0")[0].real
Z_system = (phi_substation - phi_probe_electrode) / cfg.source_magnitude_A

print()
print(f"Z_cluster (substation alone)               = {Z_cluster:.3f} Ω")
print(f"Z_system  (substation || foundations||PEN) = {Z_system:.3f} Ω")
print(f"Ratio                                      = "
      f"{Z_system / Z_cluster:.3f}")

# 7) ---- Sweep the probe along the substation → aux line -------------
#
# result.potential takes an (M, 3) point array, so a whole
# fall-of-potential curve costs one matrix-vector multiply per point
# — much cheaper than re-solving for each probe location.
probe_line_y = np.linspace(y_south, AUX_XY[1], 200)
probe_line_points = np.column_stack([
    np.full_like(probe_line_y, centre_x),
    probe_line_y,
    np.zeros_like(probe_line_y),
])
phi_along_line = result.potential(probe_line_points).real
distance_from_sub_m = np.abs(probe_line_y - y_south)
Z_apparent = (phi_substation - phi_along_line) / cfg.source_magnitude_A

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(Z_cluster, color="C2", ls="--",
           label=f"standalone Z_cluster = {Z_cluster:.2f} Ω")
ax.axhline(Z_system, color="C4", ls="-.",
           label=f"system Z_system = {Z_system:.2f} Ω")
ax.plot(distance_from_sub_m, Z_apparent, "-", color="C0",
        label="apparent Z(probe along sub→aux line)")
ax.axvline(0.62 * 200.0, color="C3", ls=":", lw=1, label="62 % point")
ax.set_xlabel("probe distance from substation in m")
ax.set_ylabel("apparent grounding impedance in Ω")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
ax.set_title(f"Fall-of-potential curve at ({LAT0}, {LON0})")
fig.tight_layout()
plt.show()

# 8) ---- Contour plot of the live OSM neighbourhood ------------------
fig = gf.plot_surface_potential(
    result, world,
    z=0.0, padding_m=30.0, n=160,
    title=f"phi over live OSM at ({LAT0}, {LON0}), 50 Hz",
)
plt.show()
```

## What just happened

### Distribution functions on `FoundationElectrodeSpec`

`presence_prob=0.7` is the **Bernoulli** trial that decides whether
each building gets a foundation electrode in any given realisation.
This is the **only stochastic axis that survives the OMBR override**:
the geometric fields (`size_xy_m`, `orientation_deg`) are rewritten
from the polygon's oriented bounding rectangle, so the per-building
geometry is deterministic given the OSM data, while the *existence*
of the electrode remains a coin flip. This is exactly the situation
AP1 wants: realistic per-Ortsnetz penetration rate, but reproducible
geometry for each present electrode.

The `depth_m=Uniform(low=0.6, high=1.2)` keeps a *non-geometric*
distribution alive — burial depth is not derivable from the
polygon, so it sensibly stays a random variable. Other distributions
(`Normal`, `LogNormal`, `Weibull`, `Categorical`) work the same way
on every `_to_float` field.

### Where the substation, aux electrode and probe live

`SubstationConfig.position` is just a tuple in the local ENU frame.
It can be anywhere — inside the cluster, on the edge, or far away.
For a real Ortsnetz the position is fixed by the cable map; for a
study you typically vary it as part of a parameter sweep.

`MeasurementSetupConfig` is the umbrella for the measurement-side
electrodes:

- `MeasurementInjectionConfig.position_xy` — the Hilfserder location.
  Defaults to a 1.5 m driven rod; pass a custom
  `MeasurementInjectionConfig.grounding` for a different aux geometry
  (e.g. `neighbour_substation_grounding()` for a real benachbarte
  Trafostation as the current sink).
- `MeasurementProbeConfig.position_xy` — the Spannungssonde location.
  Same idea: defaults to a 1 m driven rod.

The generator automatically wires the source's `return_to` to the
auxiliary electrode, so the test current physically returns through
the Hilfserder rather than through remote earth. That detail
matters for the inductive case (example 05).

### Reading the potential at an arbitrary (x, y) point

Two equivalent ways:

1. **As a real electrode** — let `MeasurementSetupConfig` add a
   probe rod at the chosen point, then read
   `result.electrode_potentials["probe_rod_0"]`.
2. **As a post-solve field evaluation** —
   `result.potential(points)` evaluates the surface potential at
   *any* (M, 3) array of points without having to re-solve. This
   is the right tool for sweeping a probe along a line (as in
   Variant B above) or for reading multiple probe candidates from
   a single solve. The two paths agree to within numerical noise.

### The Z_cluster vs. Z_system gap

In an isolated grounding measurement (substation alone, no buildings
bonded) the probe at the 62 % point would recover `Z_cluster`
within a few percent — that is the textbook 62 % rule. In a
**connected** TN-Ortsnetz the source's current does not stay on the
substation; the PEN trunk distributes it to every house, and each
present foundation electrode contributes a parallel path back into
the soil. The effective system impedance `Z_system` is therefore
substantially lower than `Z_cluster` — typically by a factor 3–10×
depending on the number and quality of the bonded foundations.

That gap is precisely the AP1-Analyse-1 question: *how strongly do
networked foundations and the PEN backbone reduce the apparent
grounding impedance compared with the isolated substation case?*
Running this example with different `presence_prob` values, soil
parameters, or building counts is exactly the parameter sweep
ADR-0009 set up.

### What you see in the contour plot

The substation funnel dominates the picture: a high-potential
hemisphere around `SUB_XY`, decaying with distance. The aux
electrode produces an **inverted funnel** at `AUX_XY` (it draws
$-I$). On the substation side you can see local dimples around
each *present* foundation electrode where the building absorbs a
small fraction of the current; the larger the foundation perimeter
(and therefore the OMBR side lengths from ADR-0011), the deeper
the dimple. The PEN trunk shows up as a thin streak of slightly
elevated potential connecting substation, KVS and houses.

## See also

- [ADR-0011](../adr/0011-osm-building-footprints.md) — full
  decision record for the OSM ingest layer.
- [Example 04](04_grounding_measurement.md) — the same measurement
  setup on a Manhattan-grid reference world; useful as a
  side-by-side comparison.
- [Notebook 32](https://github.com/Ce1ectric/groundfield/blob/main/notebooks/32_osm_footprints.ipynb)
  — interactive variant of this example with parameter sliders.
