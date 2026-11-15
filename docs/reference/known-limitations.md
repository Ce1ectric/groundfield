# Known limitations

What `groundfield` does not do, does only approximately, or does only
with certain backends — with the size of the effect where it has been
measured. The numbers are recorded in the docstrings of the modules
named in each entry (they are the authority and are updated with the
code) and in the [changelog](../changelog.md) of the release that
measured them. The [validation page](../explanation/validation.md)
shows what agrees with independent references and how well.

## Scope of the physics

- **Quasi-static, below 1 kHz.** Displacement currents in the soil and
  wave propagation along conductors are neglected. The ratio of
  displacement to conduction current, $\omega\varepsilon_0\varepsilon_r\rho$,
  is below $10^{-2}$ up to 1 kHz for $\rho \le 1000\,\Omega\,\mathrm{m}$ and
  $\varepsilon_r = 81$, but reaches about $4.5\cdot10^{-2}$ in rock of
  10 kΩ·m ([Soil](api/soil.md)). Higher frequencies are accepted without
  a warning but are outside the validated range; impulse and lightning
  transients are out of scope.
- **Linear, isotropic, horizontally layered soil.** No soil ionisation,
  no lateral changes of resistivity, no anisotropy, no frequency
  dependence of $\rho$; soil and metal are non-magnetic.
- **No capacitive coupling** between conductors (negligible below
  1 kHz for buried conductors).
- **Ideal voltage sources.** A `VoltageSource` has no internal impedance;
  model one as a junction at the terminal and a finite conductor to the
  electrode. Voltage sources that form a loop (through ideal bonds or remote
  earth) are rejected; `fem` does not solve voltage sources
  (`BackendCapabilityError`).
- **Not computed:** the inductive reduction factor of cable screens and
  earth wires; mutual impedances (`BranchType`) for groundinsight — one
  `BusType` describes one terminal.
- **Projected coordinates** are solved in an exactly shifted local frame beyond 10 km
  from the origin. A world that mixes projected and local coordinates (electrodes at UTM
  coordinates and a node near (0, 0)) gains nothing from the frame, as its bounding box
  spans the whole range. The straight runs of a polyline (one owner per run)
  are found with a round-off allowance of 64 units of the coordinates,
  0.46 µm at a UTM easting: an arc whose vertices lie closer than that to the
  chord of their neighbours (R = 50 m in 6 mm edges) gets one owner per two
  edges instead of one per edge (+5.5e-4 in Z against the same arc at local
  coordinates).

## Current ansatz and source kernels

- **Uniform current per straight wire (image family).** `image` and
  `image_2layer` assume a uniform current per unit length on each straight
  wire of an electrode (and soil layer): a full wire line of a grid, an arm
  of a star, a straight run of a polyline (user decisions Q30, Q44);
  rods, strips and rings are one wire. Impedances of rods and rings are
  within about 1 % of the per-segment `mom` solution, grids +0.36 … +0.70 %
  above one current per segment with the default line kernel (10 m 2 × 2 …
  40 m 8 × 8; +0.43 … +0.89 % in 5 cm of wet concrete; against `mom` with
  the point kernel +0.29 … +0.66 %, +0.35 … +0.69 % in the same concrete),
  stars a few per cent above it,
  growing with the number of arms (+1.8 % for 3, +3.5 % for 4, +10.5 % for
  8 arms of 10 m in 100 Ω·m; up to +12.6 % in 1000 over 100 Ω·m and
  +19.7 % in wet concrete; against `mom`, point kernel, +1.4 … +11.8 % bare,
  up to +18.8 % encased): the current crowds to the tips, which one
  current per arm cannot represent. Rectangles without cross-braces,
  given by their size or as an outline (one owner per side), with a short
  side of at least 1 m, an aspect ratio up to 50 and sides up to 150 m are
  at most 1.4 % high over the AP1 soils, bare or encased (150 m × 3 m bare
  in 30 over 200 Ω·m: +1.34 %; 0.9 % for 150 m × 7.75 m; at most 0.25 % up
  to aspect ratio 4); narrower ones more (150 m × 1 m: +1.65 %). For shells
  as thick as 0.2 m these numbers carry about 0.2 percentage points of
  model uncertainty. Surface potentials near grids, stars, strip ends and
  bonded multi-electrode clusters are worse. Against the per-segment `mom`
  solution, with the default line kernel, touch voltage with the feet 1 m
  outside the footprint and step voltage from 1 m to 2 m outside: a 20 m ×
  20 m grid with 4 × 4 meshes at a corner +6.4 % / −7.6 % (the step from
  the point above the corner 1 m outwards −11 %, the step at the middle of
  an edge +5.8 %); a 4-arm star with 5 m arms beyond a tip +8 % / −9 %
  (between two arms touch −17 %, step +40 %); rings, rods and closed loops
  within 2 % (the point kernel isolates the ansatz: +5 % / −8 %,
  +7 % / −10 %, within 1 %; `groundfield.postprocess.safety`). The safety
  functions warn with `UniformCurrentFieldWarning`; use `backend="mom"`
  for such field values.
- **Point kernel** (`kernel="point"`, and always in `mom` and
  `mom_sommerfeld`). Its row sums carry an error of about $-0.23/L$, so
  grounding resistances converge 0.9–1.9 % below the exact
  uniform-current value (`Engine.kernel`). Horizontal wires close to the
  layer interface need segments much shorter than $2\lvert z - h_1\rvert$:
  a ring 5 cm below the interface of 100 over 1000 Ω·m reads −46 % at
  0.5 m segments (changelog 2026.11.0, 10.28). A horizontal point-kernel
  segment within 0.6 segment lengths of the interface, on either side of
  it, warns with `PointKernelInterfaceWarning` (3–17 % low in the
  measured cases with an electrode through the interface, −2.5 % for a
  ring 0.2 m above it); one closer than 0.387 segment lengths, or one
  whose three-point rule would reach above the surface, raises
  `BackendCapabilityError` (a ring 1 µm above the interface read −1931 Ω
  instead of 4.18 Ω before). Use the line kernel or shorter segments.
- **Line kernel in two-layer soil.** Tagg/Sunde images farther than six
  segment lengths and the smooth part of the cross-layer correction stay
  in point form: a straight 10 m strip deviates by up to 0.05 % from its
  exact uniform-current resistance, not monotonically in the segment
  length (`Engine.kernel`). With an electrode through the interface, a
  galvanic distributed conductor needs sub-pieces of the order of $h_1$
  (changelog 2026.11.0, 10.24).
- **Cost of the line kernel.** About 2.7 times the solve time of the
  point kernel for 16 house grids ($N$ = 1920 segments) with numba
  installed, 5.7 times without (`Engine.kernel`); the extra work grows
  like $N^2$.
- **Line-kernel switch at six segment lengths.** The line kernel integrates segment
  pairs closer than six segment lengths exactly and farther ones in its far form; on a
  straight wire of equal segments the pairs six segments apart sit exactly on that
  boundary, so round-off decides their form wherever the world stands. Results of the
  same world at different positions (shifted by 0.1 m … 8.8 km, or to UTM coordinates)
  therefore spread by up to 2.4e-7 in the impedance, 4.7e-7 in the mutual matrix and
  1.7e-6 in step voltages. The point kernel (`kernel="point"`, `mom`,
  `mom_sommerfeld`) is invariant to 1e-14 at local offsets and 3e-11 at UTM coordinates.

## Soil layers

- **Three or more layers** are solved only by `mom_sommerfeld` (and the
  experimental `fem`). `FieldResult.potential` — and with it touch and
  step voltages, potential plots, potential-path CSV and the VTK field
  export — supports homogeneous and two-layer soil only and raises
  `NotImplementedError` for three or more layers.
- **Rod ends at a layer interface.** The thin-wire model has no end
  face, so the contact of a rod end with the layer beyond is all or
  nothing: against an axisymmetric model of the solid rod (4.5 m,
  $a$ = 8 mm, 1000 / 30 Ω·m) the thin wire reads +21 % with its tip at
  the interface, +7 % 1 mm above it and −4.6 % 1 mm below it (within a
  few per cent beyond two wire radii). A rod head or tip within one wire
  radius of an interface warns with `InterfaceContactWarning` and is
  listed by `diagnose()` (`interface_contact`).
- **High layer contrast.** The Tagg/Sunde series converges like
  $\lvert K\rvert^n$; the default cap of 300 terms covers
  $\lvert K\rvert = 0.94$ and warns (`SeriesTruncationWarning`) beyond.
  `mom_sommerfeld` may not converge for $\lvert K\rvert \to 1$
  (`SommerfeldConvergenceWarning`); use `image_2layer` for two layers.
- **Effective resistivity** (`effective_resistivity`, the `rho` of a
  BusType in layered soil) is an empirical calibration on the LV
  networks of the AP1 catalogue ($h_1$ = 5 … 30 m, $\rho_1$ = 30 …
  1000 Ω·m, $\rho_2$ = 30 … 200 Ω·m), defined for homogeneous and
  two-layer soil only (three distinct layers raise `ValueError`); its
  weight is limited to $[0, 1]$, so an upper layer thinner than
  0.28 … 1.34 m, depending on the preset (0.78 m with the default), gives
  $\rho_2$. The weight presets per electrode class
  (`groundfield.soil.EFFECTIVE_RESISTIVITY_PRESETS`) are the q50 values of
  the AP2 hand-over, derived from the catalogue computed with groundfield
  0.14.0 and not yet re-derived for this release, which changes LV-network
  results against 0.15.0 alone by up to +13.6 % in the input impedance and
  +42 % in a house's $Z_G$ diagonal at $\rho_1$ = 30 Ω·m (concrete shell,
  line kernel; *Numbers that change* in the changelog). A
  preset holds for its class only (the AP1 station with
  `"lv_group_4_p40"`: rms 39.9 %); even `"station"` leaves up to 15.7 % at
  $h_1$ = 5 m, where the station rods end 0.4 m above the interface. A
  BusType records `rho_definition`, not the preset (the preset follows in
  the next release, user decision Q39): use the same preset
  for the BusType and its bus, and check `max_relative` of a fit over
  $\rho_\text{eff}$ (see [Soil](api/soil.md#effective-resistivity)).

## Backends

- **`fem` is experimental.** It replaces every cluster by an equivalent
  hemisphere: biased for clusters with several electrodes (−35 % for a
  2 m ring with three 4 m rods), blind to the coupling between clusters,
  and off where a layer interface lies within the electrode, the
  hemisphere or the electrode's horizontal extent. It raises when an
  electrode touches a layer its hemisphere does not reach, when more than
  half of the hemisphere lies in a layer the electrodes do not touch, and
  when a more conductive layer that no electrode touches lies above the
  hemisphere radius or — with no more resistive layer in between — above
  the electrodes' horizontal half-extent (user decision Q23; −97 … +30 %
  before, a structural rule that also rejects near-equal layers such as
  100 over 120 Ω·m). It warns and is off by −3 … +48 % where a spanned,
  untouched layer is more resistive, −17 … +16 % for a more resistive
  layer within the half-extent, −10 … +46 % for a conductive layer there
  behind a more resistive one, −44 … −78 % for a rod that reaches
  1 µm … 8 cm into a more conductive top layer, −21 … +30 % for a ring
  lying exactly on an interface and −55 … −75 % for two bonded rods of
  which one reaches into a more conductive top layer. Just beyond the
  half-extent it is silent at −10.3 … −0.3 % (rings), and an interface
  just below a horizontal electrode buried about as deep as its
  half-extent or deeper, deeper than the hemisphere radius and the
  half-extent, is covered by no rule (+34 % for a 2 m ring 0.1 m above it
  in 1000/30 Ω·m, +250 % 7 mm above it; rings 1 cm above such an
  interface: +191 … +308 % over a more conductive layer, −27 … −14 % over
  a more resistive one). Its closed forms: a strip deeper than half its
  length, a ring deeper than its diameter and a driven `MeshElectrode`
  raise; Dwight's rod formula ignores the depth of the rod head
  (+0.8 … +15 % for 1 … 20 m rods with the head 0.5 … 10 m deep) and
  Sverak's grid formula reads high, the more the smaller the grid
  (+51 … +67 % for a 1 m grid, +14 … +39 % for 4 m, +6.5 … +17 % for
  20 m). `Engine.solve` reports the size of `fem` worlds in segments of
  the integral engines, which `fem` does not use (2 400 mesh nodes per
  driven cluster in homogeneous soil, up to about 4 100 with a 30 m and
  6 200 with a 1 km layer stack), and `fem` does not refuse a world above
  `Engine.max_segments`. It ignores inductance, `return_to`, junctions and
  concrete shells (`BackendCapabilityWarning` or `BackendCapabilityError`;
  `groundfield.solver.fem`). Generated rectangular foundations without
  cross-braces are four bonded strips (user decision Q38): their radii are
  summed (−32 % for a 10 m square ring in 100 Ω·m, +21 % as one grid
  before), the layer rule takes half a side as their half-extent, so three
  layered worlds that the grid form refused are solved with the warning
  only (−42 … −58 %), and a rectangle with a side shorter than twice its
  depth is refused. The reduction gets no further rules; the next
  release replaces it by a conforming axisymmetric model
  ([roadmap](../roadmap.md#scheduled-for-the-next-release), user decision
  Q37).
- **Concrete shells** (`concrete_shell_coefficient_ohm_m`) are represented by the image
  family, `solve_mutual_*`, `mom` and `mom_sommerfeld` (`fem` rejects them). Each wire
  carries its own radial shell; where wires meet — the corners of a ring, the crossings
  and T-joints of a mesh foundation, whose braces are encased since 2026.11 — the
  shells overlap within about $r_b$ of the junction and each wire counts the shared
  concrete as its own (a local effect, not quantified separately; it will be quantified
  once a 3-D volume reference is available, user decision Q40).
- **Aliases.** `cim`, `bem` and `image_nlayer` run `image` and `mom`;
  comparing them with their targets adds no independent information.

## Earth return and conductors

- **`carson_series`** is a per-metre approximation for long parallel
  conductors over homogeneous soil; in layered soil it uses $\rho_1$
  (with a warning), and it rejects conductor segments in the soil-surface
  plane. It tiles only near-parallel pairs of segments inclined by less
  than 8.1° and hands every other pair to the Sommerfeld kernel; at that
  boundary the result steps by the error of the per-metre model (13 % for
  two 1 km leads 2 m apart at 1 kHz, 42 % for two 300 m leads at 50 Hz;
  [earth return](../explanation/earth-return.md)). **`perfect_mirror`** is a frequency-independent diagnostic
  baseline, not a model of buried conductors below 1 kHz; across the soil
  surface its additive image has no physical basis (a lead 1 m above a PEN
  0.7 m deep: 2.28 / 2.77 times the mutual reactance of `sommerfeld` at
  50 Hz / 1 kHz), and such a solve warns. Use the default `sommerfeld` for
  numbers.
- **Cost of the Sommerfeld assembly** grows with the number of segment
  pairs and frequencies. The default `sommerfeld_assembly="tabulated"`
  shares one table of the spectral integral per pair of depth groups and
  frequency (17–25× less CPU than the per-pair rule on TN networks of 8
  and 30 houses); blocks of fewer than 64 sub-segment pairs and
  conductors in the surface plane keep the per-pair rule (`"exact"`),
  whose cost grows with every pair.
- **Skin effect** is opt-in (`resistance_model="skin"`, solid round
  conductors); the proximity effect between conductors is not modelled.
- **Concentric cables** are read only with the marker: a thinner conductor
  parallel inside a conductor with `conductor_type="cable_shield"` is its
  core and couples at the shield radius, at every axial offset (so a shield
  of several conductors and a cable in sections add up); the shield has no
  internal inductance of a solid wire (user decisions Q32, Q33). Any two
  other conductors drawn on one axis are separate cables touching each
  other; their sub-pieces add up to the lumped pair. Two unmarked cables in
  *sections* couple as filaments on the axis between sections that do not
  overlap, and a core tilted against the shield axis is read the same way
  while the lateral distance changes by at most the touching distance
  $r_c + r_s$ along a piece (and $|\cos\varphi| \ge 0.99$); beyond, the rule
  acts per segment and the pieces are about 1e-6 off the lumped pair (the
  touching stretch is then about one piece long). Under `carson_series` the
  Carson correction of a tilted pair is tiled at the mid-point distance of
  each pair, so one lumped pair and its pieces differ (1.3e-4 of the loop at
  1e-3 rad). The skin effect of a shield (a tube) is
  not modelled (`resistance_model="skin"` raises for a shield); a three-core
  cable in one shield is three cores at the shield radius each.
## Post-processing and export

- **EN 50522 permissible touch voltage** is interpolated in Table B.4 for
  fault durations from 50 ms to 10 s and clamped to 725 V below 50 ms;
  every duration beyond 10 s, including an uncleared fault (`math.inf`),
  gets 80 V (the curve of Figure 8 ends at 10 s), so the value steps from
  85 V at 10 s to 80 V just above.
- **Touch-voltage envelope** samples a ring of directions around the
  electrode's footprint, a proxy for touched parts that are not known;
  define the accessible metal parts with `touch_voltage_at_part` where
  they are. It samples nothing inside the footprint unless asked:
  `interior="meshes"` adds the centres of the meshes of grids, of rings
  and of the generated mesh foundations (bonded wires, one per wire
  line), `interior=True` also the inside of a closed polyline. The modes
  cannot see buildings: use them where the inside is accessible open
  ground (a substation yard), not for foundations or a ring around a
  housing or a mast.
- **Touch points** (`touch_voltage_at_part`) take the part's potential
  from the electrode or junction it is bonded to; a part bonded to a
  point along a distributed conductor between two nodes needs a junction
  there.
- **Reduced models.** The standard form is a fixed ansatz — check
  `max_relative` (warning above 5 %). A vector fit belongs to one soil
  (`rho_at_fit`). Passivity is checked from 0 Hz to twice the highest
  fitted frequency only.

## Generators and OSM

- **Foundation outline.** By default a footprint becomes its oriented
  minimum bounding rectangle; `foundation_outline="polygon"` follows the
  outline (ring-style specs only). Outlines smaller than 1 m are widened
  with a warning. Every `building=*` object is a footprint unless excluded
  with `exclude_building_tags`. A rectangle is one owner per side (user decision Q38) and so is a building
  outline (one owner per straight side, user decisions Q30 and Q44): a
  rectangular outline equals the same rectangle given by its size. The
  twelve outlines of the tutorial T06 layout moved by at most −0.004 % each
  with that decision, the layout with an AP1-like station by
  −0.03 … −0.40 %. The connection point of a rectangular
  foundation is the start of its local −y side, a corner (also at multiples
  of 90°, which are exact rotations), that of an outline its first vertex;
  conductors attached by name and `anchor="connection_point"` use it. The
  generators end their cables and service drops at the point of the
  foundation nearest to each conductor instead (user decision Q41).
- **Generated service drops** are straight conductors from the closest point
  of the feeding cable to the closest point of the building's foundation
  (horizontal at the PEN depth in `TnNetworkGenerator`). A cable parallel to
  the facing side meets it at the corner where that side's wire starts (the
  tie rule). A building that a cable of its feeding path crosses is joined to
  it and fed by it without a drop (radial: any trunk; star: the main that
  crosses it, whose cabinet then feeds it), and no drop runs along a cable
  farther than one minimum section (1 cm; "along" means both ends of a cable
  section within 1 cm of the drop's line, so a drop that leaves a node at a
  small angle to a cable and stays within 1 cm of it only over part of the
  section keeps its route; a foundation that touches the end node of its
  cable from beyond is tapped 1 cm before the node and its drop shares that
  1 cm).
  Drops keep the shortest connection and are not routed around foundations (user
  decision Q42): since the contact lies at the corner of the side facing the
  cable, a drop to a house behind another runs 0.2 m above that house's side wire
  (PEN 0.6 m, foundations 0.8 m) — with the cabinets per building group 0 … 100 %
  of the drops of the default grids of 10 to 200 houses (star, 30 houses: all 20
  drops, 5 m each over the middle-row house next to the main, 100 m; star 50, 150,
  200: none; radial 150: 87 of 140, 1 805 m). The field model has no inductive
  coupling between a drop and an electrode; a bare PEN couples through the soil.
  Moved 1 m aside as a whole, those drops change |Z| by at most 0.006 % with an
  insulated PEN and by up to −0.40 % with a bare one (default grids of 30 and 150
  houses, 50 Hz and 1 kHz, `carson_series` and `sommerfeld`). Every site with a
  horizontal earth electrode — foundation, `RingElectrodeSpec` circle, strip — is
  met at its nearest point (Q41, Q43); only sites with rods alone keep the head
  of their first rod. A house whose horizontal electrode touches or crosses the
  substation's (e.g. 10 m houses 5 m from the default 4 m ring) gets a drop of no length: the star topology feeds it from its main instead (drops of
  2.39 / 1.61 m in the touching / crossing cases of audit W1B), the radial topology
  raises and names both sites. The following values were measured before user decisions
  Q22, Q35 and Q43 (see the release notes for the change they bring): with an isolated PEN the end of a
  drop changes the resistive model only through its length (radial default
  grid with 2 ring houses, 50 Hz: −0.025 %); the joints and the star
  reassignment of user decision Q41 move it by −0.12 … +0.13 % (star 50
  houses −0.117 %, radial 30 / 50 houses +0.118 / +0.125 %). A galvanic
  (bare) PEN leaks along its drops, so their route moves its earthing too
  (the radial default grid with 2 ring houses, 50 Hz: Re +7.8 % against
  −0.025 % isolated). The inductive coupling of drops and cables depends on
  where the drops end: with a Neumann PEN at 1 kHz (seed 1, default grid,
  `sommerfeld` default and `carson_series`) the reactance moved by −3.1 …
  +16.3 % (star) and −8.6 … +37.7 % (radial) between the anchor corner and
  the nearest point, and lies −10.8 … +14.0 % from drops to the centre as in
  0.15.0 (12 m × 8 m ring houses +5.7 … +14.0 %; default 10 m ring houses:
  star +1.2 … +2.1 % for the 30-house network — the networks of 1 … 5 houses
  have no drop, all on the main, and move by +0.6 … +10.4 % through the joint
  and the Q38 owner —, radial −10.8 … −1.1 %). A layout cable is joined to the substation and to every cabinet whose closed earth
  electrode it passes through, not only at its ends; a cable through a house foundation
  is not joined to it (the house keeps its drop, of no length in plan there). `LvNetworkLayout.plot(world=...)` draws the
  drops as built; without a world it draws the layout's stubs (centroid to
  tap).
- **Cabinets per building group.** The star topology groups the buildings by bearing
  around the substation (equal sizes, from the widest gap) and puts each cabinet on
  the ray through its group's centroid, 1 m beyond the group; a cabinet that would
  stand there farther than `CabinetConfig.max_far_end_gap_m` (20 m) from its houses
  (a group of about 180°, e.g. 10 houses and one small commercial building: 50.4 m)
  stands on the ray through the group's far-end building instead. The threshold is a
  step, not a separation of two populations (kept for 2026.11, user decision D7):
  groups lie on both sides of it — on the centroid ray the groups of the default
  50-house grid stand 8.8 … 12.8 m from their houses (a wide group there, 206 … 341°,
  still has drops up to 113 m to its outer houses), those of the 100-house grid
  10.05 … 23.01 m (15 / 20 / 25 m put four / two / none of its five groups on the
  far-end ray), and over the 80 AP1 mixes the 118 groups beyond 20 m stand
  20.21 … 53.56 m from their houses while the largest gap kept is 18.84 m — and the
  results jump where a group crosses it: with the house columns of the 50-house grid
  32.0 / 32.2 m apart one cabinet moves 141.8 m, the drops by −19 %, the input
  resistance at 50 Hz by −0.14 % beyond the smooth trend and, with an inductive PEN,
  the reactance at 1 kHz by −21.6 / −18.9 % (`carson_series` / `sommerfeld`); the
  100-house grid across 22.95 m moves two cabinets by 65.6 m (X at 1 kHz +1.5 / −3.1 %),
  and 80 houses with 5 medium commercial buildings (a group at 20.21 m) move one by
  31.4 m when the threshold is 20.5 m (X +1.5 / +1.1 %). The plan records per cabinet
  which rule placed it and its distance on the centroid ray
  (`PlannedSite.cabinet_rule`, `PlannedSite.centroid_ray_gap_m`), so a study can see
  where a group crosses. A single group around the substation (one cabinet for a
  row through it) has its cabinet on one side only. A cabinet whose electrodes would
  come closer than `end_clearance_m` to another site raises (3 of 100 random
  200-house layouts 12 m apart); a given `CabinetConfig.placement` avoids it. The
  houses share their group's main (a tree): with an inductive PEN the network
  reactance at 1 kHz is +55 … +64 % (default star 30 houses, `sommerfeld` /
  `carson_series`) above the former cabinet strip with its parallel drops; results of
  default `TnNetworkGenerator` networks therefore depend on this topology choice.
  Planning 8 000 default star houses takes about 14 s (6.9 s with the former strip).
- **Measurement leads** of the in-line set-up are laid side by side; their
  mutual inductance depends on the logarithm of that spacing, which is a
  modelling choice, not a measured value.
- **Overpass** answers are capped at 16 MiB; live queries need network
  access (tests and tutorials use stored answers).

## Diagnostics and problem size

- **Problem size.** More segments or nodal unknowns than the engine's
  limit (`Engine.max_segments`, default `HARD_LIMIT` = 20 000) raise
  `ResourceLimitError` before any allocation; the dense system grows like
  $N^2$ in memory and work per excitation. Raise the limit per engine to
  solve a larger world on purpose. The notice that a solve may take
  minutes comes from `Engine.solve` (not from `solve_mutual_*`, which
  enforce the limit but emit no notice) from `Engine.warn_segments` on
  (default `SOFT_LIMIT` = 5 000); it is not checked against
  `max_segments`, so a notice size above the hard limit never shows.
- **Touching electrodes.** `diagnose` reports wires of different
  electrodes that are closer than five times the sum of their radii, but
  not two electrodes that meet at a common segment end point (for example
  a rod through a node of a separate grid). Metal in contact is one
  galvanic cluster: bond such electrodes with a conductor.
