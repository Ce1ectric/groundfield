# Generators

The `generators` package builds complete worlds of TN low-voltage
networks (Ortsnetze): a substation earth, the foundation earths of the
buildings, cable distribution cabinets, the PEN network that joins them
and, optionally, a fall-of-potential measurement set-up. It contains no
physics of its own; it decides geometry and topology and hands a
`World` to the solver. Design and validation programme: ADR-0009.

Two front-ends share one data model:

- `TnNetworkGenerator(config, seed=...)` draws a network from
  population parameters (`TnNetworkConfig`: building counts and types,
  soil, placement, PEN topology), every numeric field a fixed value or a
  `Distribution`;
- `LvNetworkLayout` composes one deterministic network step by step, as
  an engineer draws it: real building footprints (OSM), a hand-placed
  substation and cabinets, PEN cables routed on a Manhattan grid around
  the buildings, and the measurement loop.

Both produce a pure-data `NetworkPlan`; one materialiser,
`materialise_plan(plan)`, turns every plan into a `World`, so both
front-ends build foundations, concrete shells, PEN junctions and
measurement leads in the same way.

## Mathematical and physical context

The generated world is solved like any other: Laplace's equation in the
layered soil with the electrodes as equipotential clusters and the PEN
network as branches. What the generator controls is the parameter
space of the study (AP1):

| Axis | Config field |
|---|---|
| building mix (single-family houses, small and medium commercial) | `building_counts`, `building_types` |
| grounding system per building type (foundation, ring, rods, strips) | `BuildingTypeSpec.grounding` (`GroundingSystemSpec` of `ElectrodeSpec`) |
| penetration of foundation earths | `FoundationElectrodeSpec.presence_prob` |
| concrete encasement (moisture) | `FoundationElectrodeSpec.concrete_rho_ohm_m`, `concrete_thickness_m` |
| substation earth | `substation` |
| cable distribution cabinets | `cabinets` |
| PEN topology and cable data | `pen_topology`, `pen` |
| soil $\rho_1$, $\rho_2$, $h_1$ | `soil` (`HomogeneousSoilSpec`, `TwoLayerSoilSpec`, `MultiLayerSoilSpec`) |

**Reproducibility.** All random draws come from one generator seeded by
`seed` (or `rng=` of `build`, `plan` and `sample_world`), in a canonical
order: the same configuration and seed give the same world, independent
of the order in which mapping keys were written. `sample_world(rng=...)`
also returns the resolved configuration (every distribution replaced by
its drawn value); it replays the world exactly when no per-site draws
remain (presence probabilities 0 or 1, no placement jitter), otherwise
keep the seed or the plan / world JSON. The foundation penetration mask
of the layout builder is a stable hash of the building ids and is nested
in the penetration rate (a house equipped at 10 % is also equipped at
20 %).

**Foundations and PEN cables.** A mesh foundation (`style="mesh"`) is
one wire electrode per mesh line at every orientation (`<name>`,
`<name>_w<j>`, bonded into one cluster), bare or encased. The image
family solves one uniform current per electrode: one current per wire
is 0.3–0.6 % above the per-segment reference (`mom`), one current for
the whole mesh was 1.6–3.2 % above it and depended on the orientation.
`electrode_currents[<name>]` and `grounding_impedance(<name>)` therefore
refer to the first wire; use `cluster_impedance` or the sum over the
wires for the foundation. An axis-aligned `"lumped"` ring stays one
`grid_mesh` electrode and a polygon outline one closed wire. The PEN
cables of `TnNetworkGenerator` form a nearest-point tree: every cable
starts at the point of the network already built that is nearest to its
far end, at a leakage-free tap junction (`pen_tap_<k>`), so a service
drop taps the main beside its house instead of running back along it to
the cabinet.

**Measurement set-up.** A fall-of-potential measurement is a
two-terminal source between the earthing system and an auxiliary
electrode; the current and probe leads end on leakage-free junctions at
the instrument, and `measured_impedance(result)` returns the reading of
the simulated earth tester, probe voltage over source current. The leads
can be overhead or buried (`overhead_lead`, `buried_lead`); their
inductive coupling to the PEN network is part of the solve.

## Example

```python
import groundfield as gf

config = gf.TnNetworkConfig(
    soil=gf.TwoLayerSoilSpec(rho_1=gf.Uniform(low=80.0, high=300.0), rho_2=100.0, h_1=5.0),
    building_counts={"residential": 5},
)
world = gf.TnNetworkGenerator(config, seed=42).build()
again = gf.TnNetworkGenerator(config, seed=42).build()
assert gf.world_fingerprint(world) == gf.world_fingerprint(again)   # same seed, same world

stats = gf.world_statistics(world)
# each house has a 2 × 2 mesh foundation, one wire electrode per mesh line
assert stats["n_electrodes_by_kind"]["polyline"] == 5 * 6
result = gf.create_engine(frequencies=[50.0]).solve(world)
assert 1.0 < result.input_impedance()[0].real < 5.0       # Ω, substation with the PEN network
```

Tutorial T06 builds a network synthetically and from OpenStreetMap data,
T07 adds the fall-of-potential measurement. The German names of releases
up to 0.15.0 (`OrtsnetzLayout`, `kvs`, `KvsConfig`, …) keep working with a
`DeprecationWarning`; see [Deprecations](../deprecations.md) and the
[glossary](../glossary.md).

## API reference

::: groundfield.generators
