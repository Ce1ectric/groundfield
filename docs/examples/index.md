# Examples

A guided tour through `groundfield`. Each example is a complete,
self-contained Python snippet — copy it into a script or
interactive session and run it. The examples build on each other
but every page also recaps what you need to follow it on its own.

If you're brand new to the project, work through them in order:
each one introduces a couple of new concepts and shows what to
expect.

## Pre-requisites

```bash
pip install groundfield
# Optional, for real OSM building extracts (examples 07, 08, 09):
pip install "groundfield[geo]"
# Optional, closes the loop with the sister project groundinsight:
pip install "groundfield[groundinsight]"
```

## The nine examples

| # | Topic | What you learn |
|---|---|---|
| [01](01_quickstart_two_electrodes.md) | Quick start (two single electrodes) | Build a `World` by hand, run `engine.solve(world)`, read the cluster impedance |
| [02](02_engine_comparison.md) | Comparison of the solver engines | Cross-check `image`, `image_2layer`, `mom`, `bem`, `fem` on one geometry |
| [03](03_multilayer_soil.md) | Using multi-layer soil models | `HomogeneousSoil`, `TwoLayerSoil`, `MultiLayerSoil` and the auto-dispatching backends |
| [04](04_interconnected_grounding.md) | Interconnected grounding system with line connections | Cluster fusion via ideal bonds, finite-impedance branches (PEN cable) between clusters |
| [05](05_inductive_coupling.md) | Analysing inductive coupling | Neumann self-/mutual inductance and the Carson earth-return correction |
| [06](06_tn_model.md) | Create a TN-Model | Stochastic TN low-voltage network via `TnNetworkGenerator` + `RadialTrunkTopology` |
| [07](07_measurement_distance.md) | Comparison of measurement distances on a synthetic grounding system | Hilfserder distance sweep with inline + perpendicular voltage probes |
| [08](08_osm_pipeline.md) | OSM pipeline with simulation results | End-to-end `OrtsnetzLayout` workflow: OSM ingest → Manhattan PEN → measurement loop |
| [09](09_plot_gallery.md) | All plots in action | Layout, surface potential (`two_slope` / `symmetric` / `log`), radial decay, $x$-$z$ cross-section, world geometry |

## How to use this catalogue

* If you have only 30 minutes, run **01 → 02 → 03**. Those cover
  "build a world", "compare engines on the same geometry", and
  "switch the soil model" — most engineering use cases are
  mash-ups of those three.
* For the cabled / branched-cluster side of the modelling, work
  through **04 → 05** in sequence.
* If your study is bound to a specific real neighbourhood, jump
  straight to **08** — the OSM-driven pipeline closes the loop
  from a lat/lon centre all the way to a simulated grounding-
  measurement reading.
* For sensitivity studies on real or synthetic networks, **06**
  (stochastic generator) and **07** (Hilfserder distance sweep)
  complement each other.
* The plot helpers are all collected in **09** with the same
  reference solve in every cell — bookmark it.

## Related material

* [Concepts](../concepts.md) — the theoretical scaffold (PDE
  formulation, soil models, electrode types).
* [Engine theory](../engines/index.md) — what each backend does
  numerically and when to pick it.
* [Performance guide](../performance.md) — empirical timings
  and parameter-sweep guidance.
* [Generators API](../api/generators.md) — `TnNetworkGenerator`,
  `OrtsnetzLayout`, `PenTopology` reference.
* [Geo / OSM API](../api/geo.md) — `BuildingFootprint`,
  `Projector`, `query_and_project`.
* [ADR-0009 — World generators](../adr/0009-world-generators.md),
  [ADR-0011 — OSM building footprints](../adr/0011-osm-building-footprints.md),
  [ADR-0012 — Foundation concrete encasement](../adr/0012-foundation-concrete-encasement.md).
