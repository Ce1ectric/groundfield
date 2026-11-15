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
family solves one uniform current per straight wire (since user decision
Q30 also inside a grid, so a mesh electrode built by hand and the
generated foundation of bonded wires are the same model). Against the
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
is one closed polyline with one owner per straight side (one current per
straight wire, user decisions Q30, Q44). The deprecated `concrete_model` has
no effect (drop the argument, see [Deprecations](../deprecations.md)); an
outline narrower than `MIN_FOUNDATION_SIDE_M` (1 m) is widened to it, with
a warning, for both `foundation_outline="ombr"` and `"polygon"`.

**Cable cabinets per building group** (user decisions Q22 and Q35). In
the star topology `TnNetworkGenerator` places one cable cabinet per
building group, at the far end of the houses it feeds
(`CabinetConfig.placement=None`, the default): the buildings are ordered
by their bearing around the substation, counter-clockwise from the widest
angular gap between two neighbouring bearings, and cut into as many runs
of consecutive bearings as there are cabinets (`count_per_100_buildings`
or `fixed_count`), of sizes that differ by at most one
(`StarCabinetTopology.group_buildings`); the cabinet of a group stands on
the ray from the substation through the group's centroid,
`CabinetConfig.end_clearance_m` (1 m, positive) beyond the largest
projection on that ray of the group's earth electrodes (horizontal wires
and rods) plus the plan extent of the cabinet's own grounding — when it
stands there within `CabinetConfig.max_far_end_gap_m` (20 m, positive,
larger than `end_clearance_m`) of the nearest electrode of its group: a
cabinet stands next to the houses it feeds. A wider group, whose centroid
ray leads past its houses, has its cabinet on the ray through its far-end
building (the largest projection of its position on the direction of the
centroid, ties to the one nearest to that direction), `end_clearance_m`
beyond that building's electrodes. The cabinet feeds its group (audits
W1B-01, W1B-02). 10 houses and one small commercial building had their
cabinet 50.4 m from the nearest building on the centroid ray (drops
527.4 m), now 1.56 m (311.8 m) on the far-end ray; with the far-end ray for
every group, six houses of the 50-house grid were fed straight from the
substation (up to 116.8 m). The plan records per cabinet which rule placed
it, `PlannedSite.cabinet_rule` (`"centroid_ray"` / `"far_end_ray"`), and its
distance from the group on the centroid ray, `PlannedSite.centroid_ray_gap_m`
(`None` for every other site and for cabinets of a given placement or of
the radial topology; materialising ignores both).

The threshold is a step, kept for 2026.11 (user decision D7, audit
W1R2-02). Groups lie on both sides of it: on the centroid ray the groups of
the default 50-house grid stand 8.8 … 12.8 m from their houses, those of
the 100-house grid 10.05 … 23.01 m (its two groups beyond 20 m are on the
far-end ray; 15 / 25 m put four / none there), and over the 80 AP1 mixes
the 118 groups beyond it stand 20.21 … 53.56 m from their houses and the
largest gap kept is 18.84 m (19 mixes have a group between 15 and 25 m).
A group whose distance crosses the threshold moves its cabinet by up to the
width of the group, and the results jump:

| layout change (homogeneous 100 Ω·m) | cabinet moves | drops | R at 50 Hz | X at 1 kHz, inductive PEN (`carson_series` / `sommerfeld`) |
|---|---|---|---|---|
| 50-house grid, house columns 32.0 → 32.2 m apart (one group 19.99 → 20.22 m) | 141.8 m | 1 788.9 → 1 443.0 m | −0.36 % (−0.14 % beyond the smooth trend) | −21.6 / −18.9 % |
| 100-house grid, columns across 22.95 m (two groups cross 20 m) | 65.6 m each | +15.8 % | −0.03 % | +1.5 / −3.1 % |
| 80 houses + 5 medium commercial (AP1), one group at 20.21 m: `max_far_end_gap_m` 20 → 20.5 m | 31.4 m | 1 960.6 → 1 968.2 m | −0.02 % | +1.5 / +1.1 % |

The 100-house row is the difference of straight lines through three
spacings on each side of the crossing, at the crossing; R at 50 Hz with the
image backend (2 m and 1 m segments agree), Z at 1 kHz with the isolated
Neumann PEN in 5 m pieces. A study that varies the layout can read where a
group crosses from the plan. A cabinet whose
electrodes would come closer than `end_clearance_m` to those of another
site — a building of its own or of another group, the substation, another
cabinet — raises a `ValueError` that names both (3 of 100 random layouts of
200 houses 12 m apart; none of the default grids and AP1 mixes). In the default grid of 30 houses (three
rows of ten, 25 m × 30 m, the substation in the middle row) the groups
are the western and the eastern half, with their cabinets at
(−118.5, 0) m and (118.5, 0) m; each main runs along the middle row,
whose houses are joined to it, and the 20 houses of the outer rows tap it
with drops of 25 m (before: both cabinets west of the substation, at
(−180, 0) m and (−140, 0) m on a strip along the substation row, and 25
drops of 1 140 m, the 15 eastern ones from the substation, 3.5 … 106.5 m).
A single group around the substation (one cabinet for a row through it)
gets its cabinet beyond its farthest house, ties to the first in bearing
order. A given `CabinetConfig.placement` keeps the former rule — the
given positions, every building fed by its nearest cabinet (Manhattan
metric) — and a configuration stored before the decision carries the
former strip and keeps its cabinets. In `RadialTrunkTopology` a building
in a substation slot that no trunk crosses taps its feeder's trunk at the
closest pair, like the buildings of the cabinets, instead of getting its
own line from the substation (default 30 houses: drops 1 115 → 778 m,
14 → 4 from the substation, those of the two feeders without a trunk).

In `TnNetworkGenerator` every cable cabinet has its own PEN cable, as
the topology documents: from the substation (`StarCabinetTopology`), or
one cable per trunk segment (`RadialTrunkTopology`). A service drop taps
the cable path that feeds its own building — the cable of its cabinet,
or the trunk from the substation to its tap source, for a building in a
substation slot its feeder's whole trunk — at the point nearest to the
building: a leakage-free tap junction
`pen_tap_<k>` that splits the cable into sections `<cable>_s<k>`, or the
substation or cabinet at the end of the cable; a building in a
substation slot of a feeder without a trunk is connected to the
substation. The
router guarantees that a drop never starts on another building's drop,
never runs along a cable (the one it taps, a trunk, another cabinet's
main) farther than one minimum section, and that no cable section is
shorter than that minimum section, 1 cm (a tap closer to a node is that
node; longer for PEN radii above 7.3 mm, so that every section stays a
finite resistance). "Along" means both ends of a cable section within
1 cm of the drop's line; the one overlap left is that 1 cm, where a
foundation touches the end node of its cable from beyond and is tapped
1 cm before the node. Drops that leave one tap in one direction — the
drops of houses behind each other, such as the five drops of the 5-house
radial network from the substation ring along −x, or the five drops of the
10-house star from the substation — lie side by side at the touching
distance $2r$ of the PEN (decision D9 of 2026.11), so that only neighbours
touch: sorted by length, shortest first (equal lengths in plan order), and
centred on their common axis, drop $k$ of $n$ at $(k - (n - 1)/2)\,2r$
along the normal $(-d_y, d_x)/|d|$ of the drop direction $d$. The plan
records the offset as `PlannedBranch.axis_offset_m`; the materialiser
moves both ends of the conductor by it, and the drop stays attached to
its tap and its building. Up to then such drops lay on one axis and the
inductance matrix read them as a bundle in which every pair touches,
which no cross-section of four or more equal wires realises;
`gf.diagnose()` reported every pair (audit W3B-05). With an inductive
PEN (5 m pieces, 100 Ω·m) the input reactance moves at 1 kHz by
−2.3 / −2.7 % (`carson_series` / `sommerfeld`) in the 5-house radial
network, −2.3 / −2.8 % with 10 houses, −1.1 / −1.3 % in the 10-house
star, and by at most 1.1e-6 (30 houses) and 5e-4 (80 and 200 houses,
both frequencies) in the radial networks; the star networks of 5, 30, 80 and
200 houses have no such drops and build the same world, and plans
written before (no offsets) build the worlds they built. `LvNetworkLayout`
draws no rows. Conductors that overlap otherwise (in a hand-made plan)
are evaluated as touching wires. A drop may also pass over other buildings'
foundations, 0.2 m above their wires (PEN at 0.6 m, foundations at
0.8 m): the contact of a house lies at the corner of the side facing its
cable (the tie rule below), so a drop to a house behind others of its
grid column runs above their side wires. In the default grids of 10 to
200 houses that is 0 … 100 % of the drops (star, 30 houses: all 20
drops, each over 5 m above a side wire of the middle-row house it passes
on its way from the main, 100 m in all; star 50, 150 and 200: none; radial 150: 87 of
140 drops, 1 805 m, up to 40 m over 4 houses); drops are not routed
around foundations (user decision Q42). A drop has no inductive coupling
to these wires in the field model (an electrode carries no current along
itself), a bare one leaks next to them through the soil; moved 1 m aside
as a whole, those drops change the input impedance by at most 0.006 % with
an insulated PEN and by up to −0.40 % with a bare one (default grids of 30
and 150 houses, 50 Hz and 1 kHz; see the
[known limitations](../known-limitations.md)). The layout
builder merges cable nodes closer than the same length.

**Where conductors meet a site** (user decisions Q41 and Q43). Every
conductor the generators route to a site with a horizontal earth
electrode ends at the point of that electrode nearest to it, in
`TnNetworkGenerator` and in the layout builder: at a foundation of every
form (rectangles at any orientation, building outlines) since user
decision Q41, at a circular ring (`RingElectrodeSpec`, e.g. the default
substation of both generators, a 4 m ring with four rods) and a strip
since user decision Q43. A site with horizontal electrodes and rods is met
at the nearest point of the horizontal ones, a site with several of them
(the `large_industry` ring, mesh and strips) at the nearest point of all;
rods (vertical) keep their head. A circle is met at its exact nearest
point (it is searched on an inscribed polygon with a sagitta of at most
1 mm and refined to the circle). The default mains now leave the
substation ring at its point nearest to their cabinets — the western ones
at (−4, 0) m instead of (4, 0) m, through the ring — and drops from the
substation at the point of the ring nearest to the house.

- a service drop is the shortest connection in plan between its feeding
  cable and the horizontal wires of the building (their closest pair):
  the tap on the cable and the end on the foundation, ring or strip, at
  the PEN depth (layout builder: at the wire's depth), attached to the
  wire that holds the point (`PlannedBranch.end_contact`, a
  `PlannedContact`);
- a cable leaving a substation or cabinet with horizontal electrodes
  starts at their point nearest to its next node;
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
  cable's junctions are. The leading run of waypoints inside the start
  site's closed electrodes (a rectangle, a building outline, a circle) and
  the trailing run inside the end site's are part of those sites (audit
  R6A-01 — a router corner inside a small cabinet ring was a node that
  houses tapped, 82 of 2 540 drops in 200 random layouts with a 2 m cabinet
  ring started inside it), and the segment that leaves (enters) the site
  is clipped at the projection of the site's point nearest to its far
  waypoint. Every other passage of the cable through the closed electrodes
  of the substation or a cabinet — another cabinet on its way, or its own
  start or end site when it comes back through it — is clipped the same
  way: the cable reaches the site where it enters and leaves it where it
  leaves, joined to it there, and the waypoints in between are part of it
  (audits W1B-03, W1B-04; up to then a cable to a far cabinet kept a tap
  inside a near cabinet's ring, and a leg that left its start site and came
  back through it, or overshot its cabinet, was merged into the site and
  lost its taps: 142.8 m instead of 30.0 m of drops in the audit's loop
  case). So no tap and no corner lies inside the foundation of the
  substation or a cabinet, also for a rotated, polygon or offset one; a
  tap at a clip point, or at the site's waypoint, is the site (audits
  R4A-03, R5A-02). A cable through a house foundation is not joined to it:
  the house keeps its own drop, of no length in plan where the cable crosses
  its foundation;
- ties go to the lowest wire index, then to the smallest arc position
  along the wire: a cable parallel to the side facing it meets the side
  at the start of the range of equal distance, for a side it passes along
  its whole length the corner where that wire starts.

Sites with rods only keep the connection point of their anchor (the head
of the first rod), and so does everything attached by name
(`site_anchor_point`, `create_conductor(start=<foundation>)`).
`site_contact_wires(site)` returns the wires the rule uses (named
`site_foundation_wires` before user decision Q43, foundations only).
`materialise_plan` rejects a `PlannedContact` that names no horizontal
wire (foundation, ring, strip) of the branch's site (a direct
`materialise_branch` call without `foundation_wires` checks only that it
names a strip, polyline or ring electrode; a `NetworkPlan` rejects another
site's electrode when it is validated) or lies farther from its wire — from
the exact circle of a ring — than one minimum cable section (1 cm; a joined
cable's point may lie that far off the wire it crosses; audit R4A-05).
`LvNetworkLayout.plot(world=...)` draws the drops as built (tap to
foundation contact) instead of the layout's centroid stubs.

**Coordinates.** Both generators build in local coordinates around the
origin. `SubstationConfig.position` moves the substation only: the
buildings, cabinets and the measurement stay around (0, 0), so a substation
placed at projected coordinates builds a world that mixes projected and
local coordinates (10 houses with the substation at the UTM origin: one PEN
cable of 33 118 km), which `diagnose()` reports and the solve warns about
with `MixedCoordinatesWarning`; `RadialTrunkTopology` then drops every
building and fails to build (audit W2R2-07). To place a network at
projected coordinates, build it around the origin and translate the whole
built world — every electrode, conductor and junction by the same offset;
groundfield has no public helper for that, and the solve does not need it,
since every result depends on coordinate differences only (the translated
network of 10 houses has the input impedance of the local one to twelve
digits).

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
