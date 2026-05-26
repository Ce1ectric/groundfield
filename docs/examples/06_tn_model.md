# 06 — Create a TN-Model

For an entire LV distribution-grid section — substation, KVS,
many houses with their own foundation electrodes, PEN backbone —
hand-rolling the world quickly becomes unwieldy. `groundfield`
ships two factories:

| Factory                        | Use when                                                                 |
|--------------------------------|--------------------------------------------------------------------------|
| :class:`TnNetworkGenerator`    | Stochastic AP1-style worlds, parameter sweeps, Monte Carlo               |
| :class:`OrtsnetzLayout`        | A *single deterministic* network composed piece by piece from a real map |

This example uses :class:`TnNetworkGenerator` for a synthetic
parameter-sweep-friendly network. See the
[OSM-pipeline example](08_osm_pipeline.md) for the
:class:`OrtsnetzLayout` workflow on a real OSM extract.

## Synthetic TN-Ortsnetz with stochastic axes

```python
import groundfield as gf

cfg = gf.TnNetworkConfig(
    soil=gf.TwoLayerSoilSpec(rho_1=120.0, rho_2=30.0, h_1=3.0),
    building_counts={
        "residential": 30,
        "small_industry": gf.Discrete(values=[0, 1, 2], weights=[0.6, 0.3, 0.1]),
    },
    source_magnitude_A=1.0,
)
gen = gf.TnNetworkGenerator()
world = gen.build(cfg, rng=gf.np.random.default_rng(seed=42))

engine = gf.create_engine(
    backend="image",                       # auto-dispatch to image_2layer
    segment_length=0.5, frequencies=[50.0],
)
result = engine.solve(world)
print("substation cluster Z =",
      result.cluster_impedance("trafo_ring_0")[0])
```

A few things happen behind the scenes:

* The substation grounding (ring + 4 driven rods) is built at the
  origin.
* Buildings are placed on a Manhattan grid centred on the
  substation, with the configured count per type. Each building
  gets the per-type :class:`GroundingSystemSpec` (default
  ``residential`` = 1.0 m × 1.0 m foundation electrode).
* Cable cabinets (KVS) are placed in a row along the substation
  axis at the configured quota.
* The PEN backbone fans out from the substation to every KVS and
  from each KVS to its closest buildings.
* A 1 A current source is attached to the substation cluster.

## Picking the PEN backbone topology

`TnNetworkConfig.pen_topology` now also accepts the new
:class:`RadialTrunkTopology`:

```python
cfg = gf.TnNetworkConfig(
    soil=gf.TwoLayerSoilSpec(rho_1=120.0, rho_2=30.0, h_1=3.0),
    building_counts={"residential": 30},
    pen_topology=gf.RadialTrunkTopology(
        n_feeders=4,                 # north / east / south / west
        slots_per_substation=8,      # houses directly at the trafo
        slots_per_kvs=8,             # houses per inserted KVS
        kvs_spacing_m=60.0,
        max_feeder_length_m=400.0,
    ),
)
world = gen.build(cfg)
```

The substation now feeds four radial LV cables. Once a cable's
``slots_per_substation`` budget is exhausted, a new KVS is inserted
along the cable axis. Buildings beyond ``max_feeder_length_m`` are
dropped with a `UserWarning`.

## Stochastic axes and reproducibility

Every numeric field accepts either a constant or a
:class:`Distribution`:

| Distribution        | Typical use                              |
|---------------------|------------------------------------------|
| `Constant(v)`       | Placeholder                              |
| `Uniform(low, high)`| Soil resistivities, geometry tolerances  |
| `Normal(...)`       | Engineering tolerances                   |
| `Discrete(values, weights=...)` | Categorical counts (n_efh in {5, 10, 30, 80, 200}) |

For bit-exact reproducibility, pass an `np.random.default_rng(seed=...)`
to `gen.build(cfg, rng=...)`. `gen.sample_world(rng=...)` returns
both the world and the *resolved* config so the realisation can be
persisted as JSON alongside the simulation output.

## Where to go next

- Place the substation + KVSes at user-chosen GPS / xy positions
  and route the PEN cables Manhattan-style around the real
  building polygons — see
  [08 — OSM pipeline with simulation results](08_osm_pipeline.md).
- Compare measured grounding impedance vs Hilfserder distance on
  the resulting world — see
  [07 — Comparison of measurement distances](07_measurement_distance.md).
