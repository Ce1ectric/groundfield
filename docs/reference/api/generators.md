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
`<name>_w<j>`, bonded into one cluster), bare or encased; an encased
foundation (`concrete_rho_ohm_m`) carries the concrete shell on every
wire, its cross-braces included (DIN 18014, see
[concrete shells](../../explanation/concrete-shells.md#cross-braces-of-mesh-foundations)).
The image
family solves one uniform current per electrode. Against the
per-segment `mom` solution (100 Ω·m, 1 m segments; `mom` uses the point
kernel), one current per wire is +1.1 % with the default line kernel
and +0.3 … +0.5 % with `kernel="point"` for 10 m 2 × 2 to 30 m 4 × 4
meshes; one current for the whole mesh was +2.4 … +3.7 % (point kernel
+1.5 … +3.1 %) and depended on the orientation.
`electrode_currents[<name>]` and `grounding_impedance(<name>)` therefore
refer to the first wire; use `cluster_impedance` or the sum over the
wires for the foundation. An axis-aligned rectangle without cross-braces
(`style="ring"`, or `"mesh"` with `n_x = n_y = 1`) stays one `grid_mesh`
electrode and a polygon outline one closed wire; the deprecated
`concrete_model="distributed"` makes the axis-aligned rectangle one wire
per side, less than 1 % apart up to aspect ratio 4 and 2 % up to 50 for a
short side of at least 1 m over the soils of the AP1 grid (scope and
numbers on the
[concrete-shell page](../../explanation/concrete-shells.md#owners-of-a-rectangle-without-cross-braces);
drop the argument, see [Deprecations](../deprecations.md)); an outline
narrower than `MIN_FOUNDATION_SIDE_M` (1 m) is widened to it, with a
warning, for both `foundation_outline="ombr"` and `"polygon"`.

In `TnNetworkGenerator` every cable cabinet has its own PEN cable, as
the topology documents: from the substation (`StarCabinetTopology`), or
one cable per trunk segment (`RadialTrunkTopology`). A service drop taps
the cable path that feeds its own building — the cable of its cabinet,
or the trunk from the substation to its tap source — at the point
nearest to the building's connection point: a leakage-free tap junction
`pen_tap_<k>` that splits the cable into sections `<cable>_s<k>`, or the
substation or cabinet at the end of the cable; a building in a
substation slot of the radial trunk is connected to the substation. The
router guarantees that a drop never starts on another building's drop,
never runs along the cable it taps, and that no cable section is
shorter than 1 cm (a tap closer to a node is that node; longer for PEN
radii above 7.3 mm, so that every section stays a finite resistance). A
drop may share its route with another cable — another cabinet's cable,
a trunk, another drop — as cables in one trench do; the inductive
coupling treats such overlapping conductors as touching wires. The
layout builder merges cable nodes closer than the same length.

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
