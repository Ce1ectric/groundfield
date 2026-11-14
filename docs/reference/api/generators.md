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
A rectangle without cross-braces (`style="ring"`, or `"mesh"` with
`n_x = n_y = 1`) is one strip per side at every orientation (user
decision Q38); up to that decision an axis-aligned one was one
`grid_mesh` electrode connected at its centre, and per side its input
impedance is lower by less than 1 % up to aspect ratio 4 and 2 % up to 50
for a short side of at least 1 m over the soils of the AP1 grid, equal for
a square (scope and numbers on the
[concrete-shell page](../../explanation/concrete-shells.md#owners-of-a-rectangle-without-cross-braces)).
`electrode_currents[<name>]` and `grounding_impedance(<name>)` therefore
refer to the first wire (the local −y side, whose start, a corner of the
rectangle, is the connection point of conductors attached to `<name>` by
name); use `cluster_impedance` or the sum over the wires for the
foundation. A rectangle at a multiple of 90° is rotated exactly (no
round-off), so the local −y side stays the first wire. A polygon outline
is one closed wire. The deprecated `concrete_model` has
no effect (drop the argument, see [Deprecations](../deprecations.md)); an
outline narrower than `MIN_FOUNDATION_SIDE_M` (1 m) is widened to it, with
a warning, for both `foundation_outline="ombr"` and `"polygon"`.

In `TnNetworkGenerator` every cable cabinet has its own PEN cable, as
the topology documents: from the substation (`StarCabinetTopology`), or
one cable per trunk segment (`RadialTrunkTopology`). A service drop taps
the cable path that feeds its own building — the cable of its cabinet,
or the trunk from the substation to its tap source — at the point
nearest to the building: a leakage-free tap junction
`pen_tap_<k>` that splits the cable into sections `<cable>_s<k>`, or the
substation or cabinet at the end of the cable; a building in a
substation slot of the radial trunk is connected to the substation. The
router guarantees that a drop never starts on another building's drop,
never runs along a cable (the one it taps, a trunk, another cabinet's
main) farther than one minimum section, and that no cable section is
shorter than that minimum section, 1 cm (a tap closer to a node is that
node; longer for PEN radii above 7.3 mm, so that every section stays a
finite resistance). "Along" means both ends of a cable section within
1 cm of the drop's line; the one overlap left is that 1 cm, where a
foundation touches the end node of its cable from beyond and is tapped
1 cm before the node. A drop may share its route with another drop, as
cables in one trench do; the inductive coupling treats such overlapping
conductors as touching wires. A drop may also pass over other buildings'
foundations, 0.2 m above their wires (PEN at 0.6 m, foundations at
0.8 m): the contact of a house lies at the corner of the side facing its
cable (the tie rule below), so a drop to a house behind others of its
grid column runs above their side wires. In the default grids of 10 to
200 houses that is 0 … 80 % of the drops (star, 30 mesh houses: 14 of 25
drops, 150 m in all, up to 40 m over 4 houses; star 150: 70 of 141 drops,
2 450 m, up to 65 m over 7 houses); drops are not routed around
foundations. A drop has no inductive coupling to these wires in the
field model (an electrode carries no current along itself), a bare one
leaks next to them through the soil; moved 1 m aside as a whole, those
drops change the input impedance by at most 0.02 % with an insulated PEN
and by up to −0.30 % with a bare one (default grids of 30 and 150 houses,
50 Hz and 1 kHz; see the [known limitations](../known-limitations.md)). The layout
builder merges cable nodes closer than the same length.

**Where conductors meet a foundation** (user decision Q41). Every
conductor the generators route to a foundation ends at the point of that
foundation nearest to it, for every form (rectangles at any orientation,
building outlines, a substation's foundation), in `TnNetworkGenerator` and
in the layout builder:

- a service drop is the shortest connection in plan between its feeding
  cable and the wires of the building's foundation (their closest pair):
  the tap on the cable and the end on the foundation, at the PEN depth
  (layout builder: at the wire's depth), attached to the wire that holds
  the point (`PlannedBranch.end_contact`, a `PlannedContact`);
- a cable leaving a substation or cabinet with a foundation starts at the
  foundation's point nearest to its next node;
- a building that a cable crosses is joined to it and fed by it: a cable
  of the building's feeding path that crosses or touches its foundation in
  plan is joined to the wire there, and the building gets no drop; every
  trunk of the radial topology counts as a cable of the feeding path for
  this rule (a building in a substation slot has no cable of its own), and
  in the star topology a building that a main crosses is fed by that
  main's cabinet (the nearest of the cabinets whose mains cross it, ties
  to the lower cabinet index) instead of its nearest one; a main that
  crosses a building it does not feed stays unjoined (no cross-tie between
  cabinets through a house); where the foundation touches the cable at a
  node (a cabinet on a wire) and the cable crosses it elsewhere, it is
  joined where it crosses (audit R5A-03), otherwise the joint lies 1 cm
  from the node (audits R4A-02, R4B-01);
- a drop whose closest pair lies at such a joint starts there, on the
  cable (audit R4A-01); a drop from a node that would run along a trunk or
  another main (both ends of a section within 1 cm of the drop's line,
  audit R5A-01) starts where it leaves it, at its node at the end of the
  shared stretch (audit R4B-01);
- the layout builder keeps its taps on the layout polyline, where the
  cable's junctions are: a cable leaving a site with a foundation starts
  at the foundation's point nearest to its next waypoint, the polyline is
  clipped at the projection of that point onto it, so no tap lies inside
  the site foundation, also for a rotated, polygon or offset one, and a
  tap at the projection is the site (audits R4A-03, R5A-02);
- ties go to the lowest wire index, then to the smallest arc position
  along the wire: a cable parallel to the side facing it meets the side
  at the start of the range of equal distance, for a side it passes along
  its whole length the corner where that wire starts.

Sites without a foundation (rods, rings, strips) keep the connection point
of their anchor, and so does everything attached by name
(`site_anchor_point`, `create_conductor(start=<foundation>)`).
`site_foundation_wires(site)` returns the wires the rule uses.
`materialise_plan` rejects a `PlannedContact` that names no foundation
wire of the branch's site (a direct `materialise_branch` call without
`foundation_wires` checks only that it names a strip or polyline
electrode; a `NetworkPlan` rejects another site's electrode when it is
validated) or lies farther from its wire than one minimum cable section
(1 cm; a joined cable's point may lie that far off the wire it crosses;
audit R4A-05). `LvNetworkLayout.plot(world=...)` draws the drops as built
(tap to foundation contact) instead of the layout's centroid stubs.

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
