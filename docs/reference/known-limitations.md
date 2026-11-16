# Known limitations

This page lists what `groundfield` does not do, does only approximately,
or does only with certain engines, with the size of the effect where it has
been measured. Read the entries of your engine and topic before you rely on
a number near one of these limits.

Each entry says **what** the limit is, **how large** its effect is, and
**what to do instead**. Measurement lists, audit references and history are
collapsed under *Details and audit trail*. The numbers are recorded in the
docstrings of the modules named in each entry (they are the authority and
are updated with the code) and in the [changelog](../changelog.md) of the
release that measured them. The [validation page](../explanation/validation.md)
shows what agrees with independent references and how well, the
[warnings reference](warnings.md) explains every warning named here, and
[`diagnose()`](api/diagnostics.md) reports most geometric limits before a
solve.

| Group | Covers |
|---|---|
| [Scope of the physics](#scope-of-the-physics) | frequency range, soil model, what is not computed |
| [Coordinates](#coordinates) | projected and local coordinates in one world |
| [Current ansatz and source kernels](#current-ansatz-and-source-kernels) | the image family's uniform current per wire; the point and the line kernel |
| [Soil layers](#soil-layers) | three or more layers, rod ends and wires at an interface, high contrast, effective resistivity |
| [Backends](#backends): [the `fem` engine](#the-fem-engine), [concrete shells](#concrete-shells) | the axisymmetric and the 3-D model of `fem`, their accuracy, size and cost; overlapping concrete shells at junctions |
| [Earth return and conductors](#earth-return-and-conductors) | `carson_series`, `perfect_mirror`, the Sommerfeld rules, skin effect, concentric cables |
| [Post-processing and export](#post-processing-and-export) | EN 50522 curve, touch-voltage envelope, reduced models |
| [Generators and OSM](#generators-and-osm) | foundation outlines, service drops, cabinets, Overpass |
| [Diagnostics and problem size](#diagnostics-and-problem-size) | segment limits, touching electrodes, backend aliases |

## Scope of the physics

- **Quasi-static, below 1 kHz.** Displacement currents in the soil and
  wave propagation along conductors are neglected; impulse and lightning
  transients are out of scope.

    **How large:** the ratio of displacement to conduction current,
    $\omega\varepsilon_0\varepsilon_r\rho$, is below $10^{-2}$ up to 1 kHz
    for $\rho \le 1000\,\Omega\,\mathrm{m}$ and $\varepsilon_r = 81$, but
    reaches about $4.5\cdot10^{-2}$ in rock of 10 kΩ·m ([Soil](api/soil.md)).

    **Instead:** stay below 1 kHz. Higher frequencies are accepted without a
    warning but are outside the validated range.

- **Linear, isotropic, horizontally layered soil.** No soil ionisation,
  no lateral changes of resistivity, no anisotropy, no frequency
  dependence of $\rho$; soil and metal are non-magnetic.

- **No capacitive coupling** between conductors (negligible below
  1 kHz for buried conductors).

- **Voltage sources.** The internal impedance of a `VoltageSource` is a
  lumped $R + j\omega L$ with constant $R$ and $L$. Voltage sources without
  internal resistance that form a loop (through ideal bonds or remote
  earth) are rejected, also when one of them has an inductance (it vanishes
  at $f = 0$; `World.check` does not know the frequencies).

    **Instead:** for a frequency-dependent source impedance, solve once per
    frequency with its values.

- **Not computed:** the inductive reduction factor of cable screens and
  earth wires; mutual impedances (`BranchType`) for groundinsight — one
  `BusType` describes one terminal.

## Coordinates

- **Projected and local coordinates in one world.** Projected coordinates
  are solved in an exactly shifted local frame beyond 10 km from the
  origin. A world that mixes projected and local coordinates — points
  within 10 km of the origin and beyond 1000 km from it, typically a return
  electrode, a cabinet or a cable end entered in a local system in a UTM
  world — gains nothing from the frame and is solved as given: its two
  groups lie some 33 000 km apart and a conductor between them is that
  long.

    **How large:** a return rod at local (80, 15) m for an outline at UTM
    coordinates: Z_in +0.43 %; a PEN cable to a cabinet at local
    coordinates: a 33 118 km conductor, 7494 + j25396 Ω instead of
    5.39 + j0.19 Ω with `carson_series`, and `sommerfeld` does not finish.

    **Instead:** give every point in one coordinate system. The solve warns
    with [`MixedCoordinatesWarning`](warnings.md#mixedcoordinateswarning)
    and `diagnose()` reports it under `coordinates`. Field and probe points
    are never checked.

    ??? note "Details and audit trail"

        Its round-off is that of the UTM world (1.5e-11 instead of 1.0e-11,
        point kernel). Since 2026.11 the solve warns with
        `MixedCoordinatesWarning` and `diagnose()` reports it under
        `coordinates` (user decision D5). A world wholly in projected or
        wholly in local coordinates is never warned about at the solve;
        `diagnose()` reports one wider than 100 km as information. The rule
        is a point within 10 km of the origin together with one beyond
        1000 km: a stray point between 10 and 1000 km, a northing missing a
        digit, an easting without its zone prefix or a position given in km
        are silent at the solve (`diagnose()` reports them as information
        through their extent, 5 231 / 32 000 / 33 085 km for a rod beside
        one at UTM coordinates), and field and probe points
        (`FieldResult.potential(points)`, `touch_voltage(position=…)`,
        `step_voltage`, the probe and field points of `solve_mutual_*`) are
        never checked (a potential at local (1, 0, 0) m of a UTM world is
        4.8e-7 V, without a warning; audit W2R2-11). The measurements of
        the mixed world are those of audit W2B-04.

- **Straight runs of a polyline at UTM coordinates.** The straight runs of
  a polyline (one owner per run) are found with a round-off allowance of 64
  units of the coordinates, 0.46 µm at a UTM easting, and are grown
  greedily from the first vertex.

    **How large:** an arc whose vertices lie closer than that to the chord
    of their neighbours (R = 50 m in 6 mm edges) gets one owner per two
    edges instead of one per edge: +5.5e-4 in Z against the same arc at
    local coordinates. The same arc given in reverse order has as many
    wires, in another partition, and its impedance differs by 2.4e-8
    (closed) / 4.9e-8 (open).

    **Instead:** for such finely divided arcs give the whole world in local
    coordinates (the reverse-order difference is 4e-15 there).

    ??? note "Details and audit trail"

        The partition depends on the vertex order only where vertices lie
        within the allowance off the chord (audit W1R3-06).

## Current ansatz and source kernels

- **Uniform current per straight wire (image family).** `image` and
  `image_2layer` assume a uniform current per unit length on each straight
  wire of an electrode (and soil layer): a full wire line of a grid, an arm
  of a star, a straight run of a polyline; rods, strips and rings are one
  wire. Impedances are close to the per-segment `mom` solution; surface
  potentials next to grid corners, star tips, strip ends and bonded
  multi-electrode clusters are not, because the real current crowds there.

    **How large:** impedances of rods and rings within about 1 % of the
    per-segment `mom` solution, grids +0.36 … +0.70 %, stars +1.8 %
    (3 arms) … +10.5 % (8 arms of 10 m). Touch / step voltage next to a
    20 m × 20 m grid corner +6.4 % / −7.6 %, beyond a 4-arm star tip
    +8 % / −9 %, between two star arms touch −17 % and step +40 %. Single
    rings, rods and closed loops are within 2 %, but a bonded cluster is
    not: a bare ring bonded 1 m outside a house's foundation mesh reads the
    touch voltage +5 … +16 % above `mom` with the feet right above the
    ring (the foundation alone within 2 %).

    **Instead:** for field values near such geometries solve with
    `backend="mom"` (or compare both); the safety functions warn with
    [`UniformCurrentFieldWarning`](warnings.md#uniformcurrentfieldwarning).
    For impedances, rho-f samples and far-field potentials the image family
    is the right tool.

    ??? note "Details and audit trail"

        The current model is that of user decisions Q30 and Q44. Impedances
        of rods and rings are within about 1 % of the per-segment `mom`
        solution, grids +0.36 … +0.70 % above one current per segment with
        the default line kernel (10 m 2 × 2 … 40 m 8 × 8; +0.43 … +0.89 % in
        5 cm of wet concrete; against `mom` with the point kernel
        +0.29 … +0.66 %, +0.35 … +0.69 % in the same concrete), stars a few
        per cent above it, growing with the number of arms (+1.8 % for 3,
        +3.5 % for 4, +10.5 % for 8 arms of 10 m in 100 Ω·m; up to +12.6 %
        in 1000 over 100 Ω·m and +19.7 % in wet concrete; against `mom`,
        point kernel, +1.4 … +11.8 % bare, up to +18.8 % encased): the
        current crowds to the tips, which one current per arm cannot
        represent.

        Rectangles without cross-braces, given by their size or as an
        outline (one owner per side), with a short side of at least 1 m, an
        aspect ratio up to 50 and sides up to 150 m are at most 1.4 % high
        over the AP1 soils, bare or encased (150 m × 3 m bare in 30 over
        200 Ω·m: +1.34 %; 0.9 % for 150 m × 7.75 m; at most 0.25 % up to
        aspect ratio 4); narrower ones more (150 m × 1 m: +1.65 %). For
        shells as thick as 0.2 m these numbers carry about 0.2 percentage
        points of model uncertainty.

        Surface potentials near grids, stars, strip ends and bonded
        multi-electrode clusters are worse. Against the per-segment `mom`
        solution, with the default line kernel, touch voltage with the feet
        1 m outside the footprint and step voltage from 1 m to 2 m outside:
        a 20 m × 20 m grid with 4 × 4 meshes at a corner +6.4 % / −7.6 %
        (the step from the point above the corner 1 m outwards −11 %, the
        step at the middle of an edge +5.8 %); a 4-arm star with 5 m arms
        beyond a tip +8 % / −9 % (between two arms touch −17 %, step
        +40 %); rings, rods and closed loops within 2 % (the point kernel
        isolates the ansatz: +5 % / −8 %, +7 % / −10 %, within 1 %;
        `groundfield.postprocess.safety`).

        The 2 % hold for single electrodes. Example E02 (stream docex2, the
        executed notebook `docs/examples/E02_single_family_house.ipynb`):
        the 10 m × 10 m mesh foundation of the generators' `residential`
        house (2 × 2 cells, 0.8 m deep, 5 mm bar in 5 cm of 150 Ω·m
        concrete) with a bare ring 1 m outside, bonded; the largest touch
        voltage of `touch_voltage_envelope` 1 m from the bars (72
        directions, the feet right above the ring) per ampere, `image`
        against `mom` at 0.5 m segments: +9.4 % in 100 Ω·m, +5.3 % in
        300 over 30 Ω·m and +15.7 % in 50 over 500 Ω·m ($h_1$ = 2 m);
        the foundation alone +1.2 … +2.0 %. The image family is high there
        (the safe side), and `UniformCurrentFieldWarning` fires for the
        bonded cluster.

- **Point kernel** (`kernel="point"`, and always in `mom` and
  `mom_sommerfeld`). It places each segment's current at its midpoint; its
  row sums carry an error of about $-0.23/L$, which no refinement removes.

    **How large:** grounding resistances converge 0.9–1.9 % below the exact
    uniform-current value (`Engine.kernel`). Horizontal wires close to the
    layer interface need segments much shorter than
    $2\lvert z - h_1\rvert$: a ring 5 cm below the interface of 100 over
    1000 Ω·m reads −46 % at 0.5 m segments.

    **Instead:** use the line kernel (the default of the image family) or
    shorter segments. A horizontal point-kernel segment within 0.6 segment
    lengths of the interface warns with
    [`PointKernelInterfaceWarning`](warnings.md#pointkernelinterfacewarning);
    one closer than 0.387 segment lengths raises `BackendCapabilityError`.

    ??? note "Details and audit trail"

        The ring 5 cm below the interface is documented in changelog
        2026.11.0, finding 10.28. A horizontal point-kernel segment within
        0.6 segment lengths of the interface, on either side of it, warns
        with `PointKernelInterfaceWarning` (3–17 % low in the measured cases
        with an electrode through the interface, −2.5 % for a ring 0.2 m
        above it); one closer than 0.387 segment lengths, or one whose
        three-point rule would reach above the surface, raises
        `BackendCapabilityError` (a ring 1 µm above the interface read
        −1931 Ω instead of 4.18 Ω before).

- **Line kernel in two-layer soil.** The exact line kernel leaves three
  small remainders in two-layer soil: the ends of a finite wire next to the
  interface, the reduced kernel of direct pairs of different wires, and the
  form of far images.

    **How large:** a 2 m strip 5 mm above a 1000 / 30 Ω·m interface reads
    +3.0e-4 against the circumference average, a 10 m strip +5.5e-5; two
    strips 1 cm above that interface, 5 cm apart, −3.5e-4 in their mutual
    resistance. With `Engine.image_far_field = "point"` a wire on the
    interface is up to 3.0e-4 apart between the two paths (the default:
    1.3e-6).

    **Instead:** keep the default `Engine.image_far_field`. With an
    electrode through the interface, give a galvanic distributed conductor
    sub-pieces of the order of $h_1$.

    ??? note "Details and audit trail"

        Image pairs take the image's potential at the transverse distance of
        the wire from the image's axis (audit W3A-02). That is its average
        over the wire's circumference for a long wire; the ends of a finite
        one add a remainder of order a²/(d·ℓ) (audit W3R2A-05): a 2 m strip
        5 mm above a 1000 / 30 Ω·m interface reads +3.0e-4 against the
        circumference average, a 10 m strip +5.5e-5; rings have no ends.

        A ring 1 cm above a 1000 / 30 Ω·m interface is within 0.004 % of its
        exact thin ring. A straight 10 m strip in 1000 / 50 Ω·m with
        $h_1$ = 1 m is within 3.6e-6 of its uniform-current resistance for
        segment lengths of 0.125 … 2 m, not monotonically. A rod through
        the interface is within 1.4e-6 of its exact two-owner value.

        Direct pairs of different wires and the soil-surface image keep the
        reduced kernel of the direct term (audit W3R2A-06). Two strips 1 cm
        above a 1000 / 30 Ω·m interface, 5 cm apart, read −3.5e-4 against
        their mutual resistance with the circumference average of the direct
        term too. A house grid at 0.8 m is 4e-7 lower than with the
        transverse distance on the surface image, at 0.1 m 1.7e-5.

        Far images carry the quadrupole term up to a vertical distance of
        120 segment lengths (tapered from 60) and where it exceeds 1e-9 of a
        unit point term. With `Engine.image_far_field = "point"` they stay
        point sources and their near pairs stay Galerkin integrals. In that
        form a wire on the interface is up to 3.0e-4 apart between the
        series and the cross-layer path (the default: 1.3e-6), and a rod
        through the interface depends on the segment length by up to 6e-5
        (ds 1 … 0.25 m; the default 4e-7).

        The field of sources in the lower layer (`FieldResult.potential`,
        `solve_mutual_field`) is evaluated in point form by the spectral
        kernel. With an electrode through the interface, a galvanic
        distributed conductor needs sub-pieces of the order of $h_1$
        (changelog 2026.11.0, finding 10.24).

- **Cost of the line kernel.** The near pairs and the quadrupole far field
  over all pairs make the line kernel more expensive than the point kernel;
  the extra work grows like $N^2$.

    **How large:** for 16 house grids ($N$ = 1920) 3.8 times the solve time
    of the point kernel in homogeneous soil and 1.6 times in two-layer soil
    ($h_1$ = 5 m) with numba, 5.8 and 1.4 times without (`Engine.kernel`).

    **Instead:** install numba (extra `fast`); for large two-layer studies
    `Engine.image_far_field = "point"` takes 0.49 … 0.94 of the default's
    time; `kernel="point"` reproduces the numbers of 0.15.0. See
    [performance](../explanation/performance.md).

    ??? note "Details and audit trail"

        In two-layer soil the far images cost with the image density: a
        40-house network in 1000 / 30 Ω·m with $h_1$ = 1.5 m takes 1.8 times
        the solve before wave 3 (3.6 times without numba), 16 house grids
        below a 0.1 m top layer 4 times, and with eight 9.4 m pieces below a
        0.5 m top layer 5 times (their near image pairs are Galerkin
        integrals). `Engine.image_far_field = "point"`: 1.1 … 1.3 times
        within the upper layer; on the cross-layer path its near image pairs
        stay Galerkin integrals (audit W3R2A-01): 0.7 … 0.9 times at the AP1
        station and the 40-house network, 1.5 … 2.4 times below thin top
        layers, 4.3 … 4.5 times with long pieces there — 0.49 … 0.94 of the
        default's time.

- **Line-kernel switch.** Pairs closer than six segment lengths are
  integrated exactly, pairs beyond 7.5 use the far form, and between the
  two the values are blended.

    **How large:** results of the same world at different positions agree
    to 8e-14 over local shifts of 0.1 m … 8.8 km and to 5e-11 (impedance)
    at UTM coordinates — the rounding of the input vertices, as with the
    point kernel. Nothing to do.

    ??? note "Details and audit trail"

        The blend is user decision D2.

## Soil layers

- **Three or more layers** are solved only by `mom_sommerfeld` (and `fem`).
  `FieldResult.potential` — and with it touch and step voltages, potential
  plots, potential-path CSV and the VTK field export — supports
  homogeneous and two-layer soil only and raises `NotImplementedError` for
  three or more layers.

    **Instead:** read impedances, electrode potentials and currents of such
    a solve from the result; only the field evaluation after the solve is
    missing.

- **Rod ends at a layer interface.** The thin-wire model has no end face,
  so the contact of a rod end with the layer beyond is all or nothing.

    **How large:** against the conforming `fem` model and a finite-volume
    model of the solid rod (4.5 m, $a$ = 8 mm, 1000 / 30 Ω·m) the thin wire
    reads +22 % with its tip at the interface, +7.2 % 1 mm above it and
    −4.4 % 1 mm below it (+3.3 % / +2.1 % 16 mm above / below).

    **Instead:** keep rod ends a few wire radii from an interface, or
    bracket the result by moving the end by ±2 wire radii; the axisymmetric
    `fem` solves the solid rod. A rod head or tip within one wire radius of
    an interface warns with
    [`InterfaceContactWarning`](warnings.md#interfacecontactwarning) and is
    listed by `diagnose()` (`interface_contact`).

- **Wires within one radius of the layer interface.** A wire at least one
  wire radius from the interface is integrated exactly by the line kernel.
  Closer, its surface crosses the interface and the thin-wire model does
  not apply.

    **How large:** a ring 5 mm above a 1000 / 30 Ω·m interface is within
    4.1e-5 of its exact thin ring (two references). At half a radius the two
    references (image sum, Hankel transform) differ by 8 %, and the line
    kernel lies 2 … 11 % below them.

    **Instead:** solve a ring that touches or crosses an interface with the
    axisymmetric `fem` (solid torus). Such wires lie within the reach of the
    next entry, so the solve warns with `InterfaceContactWarning` wherever
    the contrast is at least 1.3; `kernel="point"` keeps its own warning for
    wires within two segment lengths of the interface.

    ??? note "Details and audit trail"

        At half a radius the line kernel read +65 % before audit W3A-02.
        Within a/20 of the interface the image blends into the reduced
        kernel of the direct pair, so a wire on the interface is the same on
        both sides (4.1806 Ω for a 3 m ring at $h_1$ = 5 m). Through ±a the
        impedance moves with wiggles of at most 0.17 % against its direction
        (strips and rings in 1000 / 30 and 30 / 1000 Ω·m; up to 24 % with
        the blend over one radius before audit W3R2A-03).

        The horizontal-wire rule of `InterfaceContactWarning` (audits
        W4A-07, W4R2A-04, W4R3A-02) covers every wire that the interface
        cuts at a contrast of 1.3 or more.

- **A horizontal wire within a few wire radii of an interface.** The
  thin-wire engines (the image family, `mom`, `mom_sommerfeld`) and the 3-D
  `fem` carry the current on the wire axis; next to a more conductive
  layer the solid wire's current crowds to its near side.

    **How large:** against the solid torus a ring of $a$ = 5 mm over a layer
    33 times more conductive reads +0.59 % at 4 radii, +1.2 % at 3, +3.6 %
    at 2, +8.6 % at 1.5, +32 % at 1.1 and +89 % when the interface cuts the
    wire at 0.99 radii; under a more resistive layer up to +2.6 %. The 3-D
    `fem` reads +9.2e-3 for a ring with its axis 1.5 cm above
    1000 / 30 Ω·m, and a rod tip on such an interface +43 %.

    **Instead:** keep horizontal wires outside the reach below, or solve
    rings with the axisymmetric `fem`, which takes them as solid tori
    (within 5.1e-5 of a boundary-element model down to contact, up to a
    contrast of 1000). Within the reach the solve warns with
    [`InterfaceContactWarning`](warnings.md#interfacecontactwarning)
    (`diagnose()` category `interface_contact`).

    Reach of the interface, wire axis to interface, in wire radii $a$
    (values in brackets interpolated; the warning holds the last value
    beyond 1000):

    | contrast | more conductive beyond | more resistive beyond |
    |---|---|---|
    | 1.4 | 1.0 a | 1.0 a |
    | 2 | 1.3 a | (1.14 a) |
    | 3 | 1.9 a | 1.3 a |
    | 5.5 | 2.7 a | (1.60 a) |
    | 10 | 3.3 a | 1.9 a |
    | 18 | 3.7 a | (2.05 a) |
    | 33 | 3.9 a | 2.2 a |
    | 100 | 4.1 a | 2.2 a |
    | 1000 | 4.3 a | 2.3 a |

    (1.0 a: only a wire the interface cuts; below a contrast of 1.3 nothing
    is reported, a cut wire is within 0.96 % there.) Measured on rings with
    $r_0/a \ge 18$; a thicker wire on a smaller ring, and straight wires
    against a solid reference, are not covered. No generated network and no
    world of the AP1 soils is reported.

    ??? note "Details and audit trail"

        The solid torus is the axisymmetric `fem`; a Hankel thin ring plus
        the exact 2-D proximity correction agrees with it to 5e-4. The
        +2.6 % under a more resistive layer is a 55 mm wire on a 1 m ring at
        1.01 radii, contrast 1000. The 3-D `fem` is a thin wire too: its
        +9.2e-3 for the ring ($r$ = 1 m) at 1.5 cm lies above its error
        estimate. The reach is the largest distance at which the thin wire
        is 1 % off in five ring geometries ($r_0/a$ = 18 … 400, the layer
        beyond below and above the wire), rounded up to 0.1 a, plus 0.1 a
        between the nodes; no measured point 1 % or more off lies outside it
        (the thin wire converged in its segment length). The full
        measurement is under
        [`InterfaceContactWarning`](warnings.md#interfacecontactwarning).
        Audits W4A-07, W4B-02, W4R2A-03, W4R2A-04, W4R3A-02.

- **High layer contrast.** The Tagg/Sunde series converges like
  $\lvert K\rvert^n$; the default cap of 300 terms covers
  $\lvert K\rvert = 0.94$ and warns
  ([`SeriesTruncationWarning`](warnings.md#seriestruncationwarning))
  beyond. `mom_sommerfeld` may not converge for $\lvert K\rvert \to 1$
  ([`SommerfeldConvergenceWarning`](warnings.md#sommerfeldconvergencewarning)).

    **Instead:** raise `Engine.image_max_terms`; use `image_2layer` for two
    layers.

- **Effective resistivity** (`effective_resistivity`, the `rho` of a
  BusType in layered soil) is an empirical calibration on the LV networks
  of the AP1 catalogue ($h_1$ = 5 … 30 m, $\rho_1$ = 30 … 1000 Ω·m,
  $\rho_2$ = 30 … 200 Ω·m), defined for homogeneous and two-layer soil only
  (three distinct layers raise `ValueError`).

    **How large:** its weight is limited to $[0, 1]$, so an upper layer
    thinner than 0.28 … 1.34 m, depending on the preset (0.78 m with the
    default), gives $\rho_2$. A preset holds for its class only (the AP1
    station with `"lv_group_4_p40"`: rms 39.9 %); even `"station"` leaves up
    to 15.7 % at $h_1$ = 5 m. The presets are not yet re-derived for this
    release, which changes LV-network results against 0.15.0 alone by up to
    +13.6 % in the input impedance and +42 % in a house's $Z_G$ diagonal.

    **Instead:** name the preset of your bus (`rho_preset`) and check
    `max_relative` of a fit over $\rho_\text{eff}$ (see
    [Soil](api/soil.md#effective-resistivity)); a mismatch of presets warns
    with [`RhoPresetMismatchWarning`](warnings.md#rhopresetmismatchwarning).

    ??? note "Details and audit trail"

        The weight presets per electrode class
        (`groundfield.soil.EFFECTIVE_RESISTIVITY_PRESETS`) are the q50
        values of the AP2 hand-over, derived from the catalogue computed with
        groundfield 0.14.0 and not yet re-derived for this release, which
        changes LV-network results against 0.15.0 alone by up to +13.6 % in
        the input impedance and +42 % in a house's $Z_G$ diagonal at
        $\rho_1$ = 30 Ω·m (concrete shell, line kernel; *Numbers that
        change* in the changelog). Even `"station"` leaves up to 15.7 % at
        $h_1$ = 5 m, where the station rods end 0.4 m above the interface.
        A BusType records the preset you name (`rho_preset`, user decision
        Q39): the JSON document in `metadata["rho_preset"]`, the live
        `groundinsight.BusType` and the groundinsight database in the last
        line of the description, `groundfield rho_preset: <name>`
        (`bustype_rho_preset` reads it back; audit W3B-06; a document's
        `metadata["rho_preset"]` must be a preset name or null, as the
        exporters write it, and only the exact line is the record, audit
        W3R2B-06); `evaluate_spec` and `bustype_from_spec` warn when the bus
        uses another one, nothing checks `rho_at_fit` against it, and a
        document without a recorded preset is not checked.

- **Penetration depth of layered soil.** `penetration_depth` matches the
  layered earth return to second order in the horizontal wavenumber; it is
  exact only in limits (homogeneous soil, very thin or thick top layers,
  large separations).

    **How large:** in between, the earth-return correction of two wires is
    off by up to 28 % / 33 % (overhead / buried, AP1 soils, separations near
    $\lvert p\rvert$), on top of the complex-depth formula's own error (up
    to 13 % in homogeneous soil); three layers up to 46 %, four layers up to
    77 % at 1 km, 1 kHz (37 % at 50 Hz).

    **Instead:** the reference integral `per_metre_mutual_impedance`, within
    6e-8 of the earth-return correction up to 300 m and 7e-7 at 1 km for
    buried pairs.

    ??? note "Details and audit trail"

        Measured in audit W3B-08.

## Backends

The limits that belong to one backend: the `fem` engine and the concrete
shells of every backend. The limits of the integral engines' current model
and kernels are under [current ansatz and source kernels](#current-ansatz-and-source-kernels),
those at layer interfaces under [soil layers](#soil-layers).

### The `fem` engine

- **`fem` is experimental** (`ExperimentalBackendWarning` once per
  process). It has two conforming models (ADR-0016);
  `FemOptions.model = "auto"` takes the axisymmetric one wherever it
  applies and the three-dimensional one (optional extra `fem3d`; without it
  `ImportError`) for every other world. `fem` is a reference for single
  clusters and small groups, not for networks. Like every backend,
  `FieldResult.potential` of a `fem` result supports homogeneous and
  two-layer soil only.

    ??? note "Details and audit trail"

        The automatic choice of the model is user decision Q37.

- **Axisymmetric model.** Rods on one vertical axis and horizontal rings
  centred on it, solved as the solid bodies they are in any horizontally
  layered soil, bare or encased. A concrete shell enters as the surface
  conductance $1/(2\pi r_b C)$ (exact for a cylindrical shell). A ring wire
  must lie 5 % of its radius below the surface (the thin-wire engines
  accept $2z > a$).

    **How large:** port resistances agree with independent references to
    about 1e-4 at the default (within 5.1e-5 for a ring wire within 1.05
    radii of an interface up to a contrast of 1000, −8.5e-4 at 10 000).
    Eight rings with a central rod need about a million nodes and 17 s
    (1.2 million nodes and 26 s when one ring lies on an interface).

    **Instead:** raise `FemOptions.refinement` for a smaller error;
    `FemOptions.max_nodes` (2 million, about 8 GB at about 4 GB per million
    nodes) refuses larger meshes, raise it on purpose.

    ??? note "Details and audit trail"

        The error estimate is about the finer level's error (0.94 … 1.07 of
        it next to an interface) and 10 … 130 times that of the extrapolated
        value. Audits W4A-01, W4R2A-02 (accuracy next to an interface),
        W4A-05 (mesh size).

- **3-D model: accuracy.** The wires are thin wires like those of the
  integral engines (a smoothed line source with an analytic radius
  correction: uniform around the wire, no end faces), the soil an isotropic
  tetrahedral mesh with immersed layer interfaces.

    **How large:** away from layer interfaces it agrees with the
    axisymmetric model to 2e-3 for rods and rings in one to three layers.
    Its two mesh levels are not asymptotic for every geometry (error ratios
    3.0 … 4.4): the reported error estimate (3 … 5e-3;
    `fem_error_estimate_transfer` for the transfer entries) bounds the error
    while the ratio is at least 2.5, and differences below about 2e-3 are
    not resolved. Next to a strongly contrasting interface the thin-wire
    error applies (see *A horizontal wire within a few wire radii of an
    interface*).

    **Instead:** check a 3-D result with an integral engine as the second
    method; solve rods on one axis and rings around it with the
    axisymmetric model.

    ??? note "Details and audit trail"

        For rings the estimate is 2.1 … 4.4 times the error. Audits W4B-01,
        W4B-02.

- **3-D model: wires beside each other.** Wires that do not meet but run
  beside each other (side by side within 30° of parallel, a folded arm, a
  small ring around a rod head) are kept at least $1.5\,R$ apart ($R$ =
  `source_radius`, 6 cm), two wires end to end in line with a gap, such as
  stacked rods of two electrodes, at least $5\,R$. The model takes a
  smaller $R$ for the whole world, and the nodes grow as $1/R$.

    **How large:** coaxial rings 4.5 cm apart need 1.0 million nodes,
    stacked 8 mm rods 10 cm apart 0.47 million; coaxial rings 4.5 … 10 cm
    apart lie within 1.8e-3 of the solid tori. The model refuses, before
    meshing, wires closer than 3.3 radii of the thickest bare wire of the
    world, axis to axis (end to end 11, so stacked 8 mm rods up to 8.8 cm
    apart; `BackendCapabilityError`).

    **Instead:** an integral engine for such wires (the refusal names them),
    or the axisymmetric model where the world allows it.

    ??? note "Details and audit trail"

        End to end, the transfer entry's error falls with $R$ / gap; at
        $3\,R$ it was still up to 1.18 times its estimate. Wires that meet
        at 10° or more are left alone, so a V or a branch of 10 … 30° is
        solved; wires that meet at less than 10°, nearly parallel or
        touching along their length, count as running beside each other
        (audit W4R3B-02). Two encased wires of different ports whose shells
        touch count as one conductor in the radius correction, so their
        separate port entries are not resolved. Audits W4R2B-03, W4R2A-01,
        W4R3B-01, W4R3B-05.

- **3-D model: layer contrast.** Its conjugate gradients are tested up to a
  layer contrast of about 300.

    **How large:** at 1000 they do not converge and the solve raises
    `BackendCapabilityError`.

    **Instead:** the axisymmetric model (any contrast) where it takes the
    world, otherwise an integral engine.

    ??? note "Details and audit trail"

        Audit W4B-03.

- **3-D model: wires near the surface or an interface.** A horizontal wire
  within $R/0.8$ of the soil surface or a layer interface ($R$ =
  `source_radius`, 6 cm) gets a smaller disc and a finer mesh.

    **How large:** a bare one within about 2.8 wire radii is refused
    (`BackendCapabilityError`); an encased one is solved.

    **Instead:** an integral engine (the refusal names them).

- **3-D model: size, time and ports.** The finer mesh and the number of
  ports set the cost.

    **How large:** the AP1 station (28 m of wire) takes 1.5 … 3 minutes and
    4.1 … 4.3 GB. `FemOptions.max_nodes_3d` (2 million nodes of the finer
    mesh; the solve adds about 2.9 GB per million) refuses larger meshes
    before they are built, the house mesh of the generators (60 m) at the
    defaults included (it solves at `source_radius = 0.1`, `growth = 0.6`:
    1.5 million nodes, 1.5 minutes, 4.6 GB). More than
    `FemOptions.max_ports_3d` ports (100; every leaking piece of a galvanic
    distributed conductor is one) are refused before meshing
    (`ResourceLimitError`).

    **Instead:** raise the option to solve such a world on purpose, or use
    an integral engine for networks.

    ??? note "Details and audit trail"

        The port limit is audit W4B-08.

- **3-D model: resolved concrete** (`concrete_resistivity`) needs straight
  axis-parallel encased wires and converges at first order. It serves
  differences on one mesh, not the shell term itself.

    **How large:** its per-metre shell conductance reads 2.8 … 4.6 % high at
    the default levels: the square prism's concrete is 3.8 % less resistive
    than the circular shell in the shell-dominated limit, and its faces
    converge at first order; its error estimate is no bound. A cover so thin
    that the smoothed source does not reach beyond the metal (a 5 mm bar
    with 5 mm cover) is refused before meshing.

    **Instead:** the per-wire shell coefficient (the default of both
    models) for absolute values.

    ??? note "Details and audit trail"

        Resolved concrete is user decision Q40; audits W4B-04 (accuracy) and
        W4R2B-04 (thin cover).

### Concrete shells

- **Overlapping shells at junctions.** Concrete shells
  (`concrete_shell_coefficient_ohm_m`) are represented by the image
  family, `solve_mutual_*`, `mom` and `mom_sommerfeld`, and `fem` (each
  wire its own shell; the axisymmetric model as a surface conductance, the
  3-D model as a radial resistance per length). Each wire carries its own
  radial shell; where wires meet — the corners of a ring, the crossings and
  T-joints of a mesh foundation, whose braces are encased since 2026.11 —
  the shells overlap within about $r_b$ of the junction and each wire
  counts the shared concrete as its own.

    **How large:** against the concrete resolved as a volume in the
    three-dimensional `fem` model a junction acts like a few cm of wire, at
    most about $2\,r_b$, sign not settled; the house mesh of the generators
    (60 m, nine junctions) changes by about 1 % or less (0.35 … 0.85 % with
    every sign aligned).

    **Instead:** no correction is applied, because the junction numbers are
    not converged ([concrete shells](../explanation/concrete-shells.md)).

    ??? note "Details and audit trail"

        The volume model is user decision Q40 (5 mm bar, 5 cm cover, 1 000
        and 5 000 Ω·m concrete in 100 Ω·m soil). The junction numbers are
        not converged — one refinement of the prism faces moved the corner
        by about 5 cm, and two decompositions of the same runs differ in
        sign (audit W4B-05) — so no correction is applied.

## Earth return and conductors

- **`carson_series`** is a per-metre approximation for long parallel
  conductors over homogeneous soil; in layered soil it uses $\rho_1$
  (with a warning), and it rejects conductor segments in the soil-surface
  plane. It tiles only near-parallel pairs of segments inclined by less
  than 8.1° and hands every other pair to the Sommerfeld kernel.

    **How large:** at that boundary the result steps by the error of the
    per-metre model (13 % for two 1 km leads 2 m apart at 1 kHz, 42 % for
    two 300 m leads at 50 Hz; [earth return](../explanation/earth-return.md)).
    It overstates the earth return of LV networks of short service drops
    (shorter than the skin depth): the generated star, radial and grid
    networks of 5 … 100 houses read X 4 … 21 % and R up to 11 % high at
    1 kHz and X 21 … 50 % and R up to 0.9 % high at 50 Hz against
    `sommerfeld`.

    **Instead:** use the default `sommerfeld` for numbers and for networks.

    ??? note "Details and audit trail"

        The 80-house radial network reads +50 % at 50 Hz; the default
        `sommerfeld` agrees with a refined discretisation to 1e-3 for the
        5-house radial network and 4e-5 for the 80- and 100-house ones. At
        50 Hz `carson_series` is no closer than `perfect_mirror` for the
        10-house radial network, the 30-house star and the 80- and 100-house
        radial networks, but closer for the 5-house radial network (+31 %
        against +45 %; audits W3B-02, W3R2B-03). The error changes with the
        geometry, so a geometry change can move the two models in opposite
        directions (the 100-house star across the cabinet fix: +1.2 %
        against −3.5 %). Under `carson_series` the stretch rule of tilted
        pairs (audit W1R3-07) moves generated networks by up to 3e-7 (the
        10-house star with 5 small and 2 medium sites 2.8e-7, the 80-house
        radial network 3.4e-8; the 30-house star and radial networks are
        bit-identical; audit W3B-07).

- **`perfect_mirror`** is a frequency-independent diagnostic baseline, not
  a model of buried conductors below 1 kHz.

    **How large:** across the soil surface its additive image has no
    physical basis (a lead 1 m above a PEN 0.7 m deep: 2.28 / 2.77 times
    the mutual reactance of `sommerfeld` at 50 Hz / 1 kHz), and such a
    solve warns.

    **Instead:** use the default `sommerfeld` for numbers.

- **Sommerfeld segment rules.** Under `sommerfeld` the self term of every
  segment is integrated exactly in one dimension and close pairs along
  their line of closest approach. Pairs of long segments that are not
  close keep a composite 16×16 rule (pieces of a quarter skin depth, at
  most 512 per segment, then a `SommerfeldResolutionWarning`). The spectral
  quadrature resolves the most distant points of a conductor up to
  2^17 · 2π (z + z′)/30 — 38 km for a conductor at 0.7 m, 33 km from a rod
  head at 0.5 m to a junction at 0.7 m.

    **How large:** the self terms are within 1.1e-9 of a refined rule from
    1 m and 0.01 skin depths to 3 km (up to 3.4e-9 in high-contrast soils),
    close pairs within about 4e-8. A longer conductor than the spectral
    range warns (`SommerfeldResolutionWarning`) and is not converged (a
    lumped 100 km conductor solves in 10.6 s but warns).

    **Instead:** treat a result that carries
    [`SommerfeldResolutionWarning`](warnings.md#sommerfeldresolutionwarning)
    as not converged.

    ??? note "Details and audit trail"

        The segment rules are user decision Q34: within 1.1e-9 of a refined
        rule from 1 m and 0.01 skin depths to 3 km for the soils tested
        (100, 1000, 30 over 1000 and 300/50/2000 Ω·m; up to 3.4e-9 in
        high-contrast soils: 10 over 10 000 Ω·m, 1.8e-9 under a 0.3 m top
        layer; independent references, audits W3B-01, W3R2B-08), close
        pairs within about 4e-8 (3.5e-8 for 1 m wires 1 mm apart in 30 over
        1000 Ω·m at 1 kHz), continuous in the length. The spectral range is
        audit W3B-10. Two long close conductors (two lumped cables 1 m
        apart) cost a few seconds (1.2 / 4.8 s at 1 / 5 km).

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
  internal inductance of a solid wire. The skin effect of a shield (a tube)
  is not modelled (`resistance_model="skin"` raises for a shield); a
  three-core cable in one shield is three cores at the shield radius each.

    **Instead:** mark the shield with `conductor_type="cable_shield"`; any
    two other conductors drawn on one axis are separate cables touching each
    other.

    ??? note "Details and audit trail"

        The marker and the coupling at the shield radius are user decisions
        Q32 and Q33. Any two other conductors drawn on one axis are separate
        cables touching each other; their sub-pieces add up to the lumped
        pair. Two unmarked cables in *sections* couple as filaments on the
        axis between sections that do not overlap, and a core tilted against
        the shield axis, or crossing it, is read the same way at any angle
        and in pieces of any length (they add up to the lumped pair within
        3e-12; audits W1R2-01, W1R3-01). Under `carson_series` a tilted pair
        that comes close within its overlap is tiled per stretch, so a
        lumped pair and its pieces add up within 8e-6 of the loop reactance
        (audit W1R3-07; 5.2e-2 apart before at 4e-2 rad); a pair whose
        distance stays above 30 times its change keeps the mid-point value
        (within 5e-5 of that entry).

## Post-processing and export

- **EN 50522 permissible touch voltage** (EN 50522:2022) is interpolated in
  Table B.4 for fault durations from 50 ms to 10 s and clamped to 725 V
  below 50 ms; every duration beyond 10 s, including an uncleared fault
  (`math.inf`), gets 80 V (the curve of Figure 8 ends at 10 s), so the value
  steps from 85 V at 10 s to 80 V just above.

- **Touch-voltage envelope** samples a ring of directions around the
  electrode's footprint, a proxy for touched parts that are not known. It
  samples nothing inside the footprint unless asked: `interior="meshes"`
  adds the centres of the meshes of grids, of rings and of the generated
  mesh foundations (bonded wires, one per wire line), `interior=True` also
  the inside of a closed polyline. The modes cannot see buildings.

    **Instead:** define the accessible metal parts with
    `touch_voltage_at_part` where they are known; use the interior modes
    where the inside is accessible open ground (a substation yard), not for
    foundations or a ring around a housing or a mast.

- **Touch points** (`touch_voltage_at_part`) take the part's potential
  from the electrode or junction it is bonded to.

    **Instead:** a part bonded to a point along a distributed conductor
    between two nodes needs a junction there.

- **Reduced models.** The standard form is a fixed ansatz; a vector fit
  belongs to one soil (`rho_at_fit`); passivity is checked from 0 Hz to
  twice the highest fitted frequency only.

    **Instead:** check `max_relative` (warning above 5 %,
    [`RhoFStandardQualityWarning`](warnings.md#rhofstandardqualitywarning)).

## Generators and OSM

- **Foundation outline.** By default a footprint becomes its oriented
  minimum bounding rectangle; `foundation_outline="polygon"` follows the
  outline (ring-style specs only). Outlines smaller than 1 m are widened
  with a warning. Every `building=*` object is a footprint unless excluded
  with `exclude_building_tags`. A rectangle is one owner per side and so is
  a building outline (one owner per straight side): a rectangular outline
  equals the same rectangle given by its size. The connection point of a
  rectangular foundation is the start of its local −y side, a corner (also
  at multiples of 90°, which are exact rotations), that of an outline its
  first vertex; conductors attached by name and
  `anchor="connection_point"` use it. The generators end their cables and
  service drops at the point of the foundation nearest to each conductor
  instead.

    ??? note "Details and audit trail"

        One owner per side is user decision Q38 for rectangles and user
        decisions Q30 and Q44 for building outlines. The twelve outlines of
        the tutorial T06 layout moved by at most −0.004 % each with that
        decision, the layout with an AP1-like station by −0.03 … −0.40 %.
        The nearest-point rule of the generators is user decision Q41.

- **Generated service drops** are straight conductors from the closest
  point of the feeding cable to the closest point of the building's
  foundation (horizontal at the PEN depth in `TnNetworkGenerator`). Drops
  keep the shortest connection and are not routed around foundations, so a
  drop to a house behind another runs 0.2 m above that house's side wire
  (PEN 0.6 m, foundations 0.8 m). The field model has no inductive coupling
  between a drop and an electrode; a bare PEN couples through the soil.

    **How large:** moved 1 m aside as a whole, those drops change
    $\lvert Z\rvert$ by at most 0.006 % with an insulated PEN and by up to
    −0.40 % with a bare one (default grids of 30 and 150 houses, 50 Hz and
    1 kHz, `carson_series` and `sommerfeld`).

    ??? note "Details and audit trail"

        A cable parallel to the facing side meets it at the corner where
        that side's wire starts (the tie rule). A building that a cable of
        its feeding path crosses is joined to it and fed by it without a
        drop (radial: any trunk; star: the main that crosses it, whose
        cabinet then feeds it), and no drop runs along a cable farther than
        one minimum section (1 cm; "along" means both ends of a cable
        section within 1 cm of the drop's line, so a drop that leaves a node
        at a small angle to a cable and stays within 1 cm of it only over
        part of the section keeps its route; a foundation that touches the
        end node of its cable from beyond is tapped 1 cm before the node and
        its drop shares that 1 cm).

        The shortest connection is user decision Q42: since the contact lies
        at the corner of the side facing the cable, a drop to a house behind
        another runs over that house's side wire — with the cabinets per
        building group 0 … 100 % of the drops of the default grids of 10 to
        200 houses (star, 30 houses: all 20 drops, 5 m each over the
        middle-row house next to the main, 100 m; star 50, 150, 200: none;
        radial 150: 87 of 140, 1 805 m).

        Every site with a horizontal earth electrode — foundation,
        `RingElectrodeSpec` circle, strip — is met at its nearest point
        (Q41, Q43); only sites with rods alone keep the head of their first
        rod. A house whose horizontal electrode touches or crosses the
        substation's (e.g. 10 m houses 5 m from the default 4 m ring) gets a
        drop of no length: the star topology feeds it from its main instead
        (drops of 2.39 / 1.61 m in the touching / crossing cases of audit
        W1B), the radial topology raises and names both sites.

        The following values were measured before user decisions Q22, Q35
        and Q43 (see the release notes for the change they bring): with an
        isolated PEN the end of a drop changes the resistive model only
        through its length (radial default grid with 2 ring houses, 50 Hz:
        −0.025 %); the joints and the star reassignment of user decision
        Q41 move it by −0.12 … +0.13 % (star 50 houses −0.117 %, radial
        30 / 50 houses +0.118 / +0.125 %). A galvanic (bare) PEN leaks along
        its drops, so their route moves its earthing too (the radial default
        grid with 2 ring houses, 50 Hz: Re +7.8 % against −0.025 %
        isolated). The inductive coupling of drops and cables depends on
        where the drops end: with a Neumann PEN at 1 kHz (seed 1, default
        grid, `sommerfeld` default and `carson_series`) the reactance moved
        by −3.1 … +16.3 % (star) and −8.6 … +37.7 % (radial) between the
        anchor corner and the nearest point, and lies −10.8 … +14.0 % from
        drops to the centre as in 0.15.0 (12 m × 8 m ring houses
        +5.7 … +14.0 %; default 10 m ring houses: star +1.2 … +2.1 % for the
        30-house network — the networks of 1 … 5 houses have no drop, all on
        the main, and move by +0.6 … +10.4 % through the joint and the Q38
        owner —, radial −10.8 … −1.1 %).

        A layout cable is joined to the substation and to every cabinet
        whose closed earth electrode it passes through, not only at its
        ends; a cable through a house foundation is not joined to it (the
        house keeps its drop, of no length in plan there).
        `LvNetworkLayout.plot(world=...)` draws the drops as built; without
        a world it draws the layout's stubs (centroid to tap).

- **Cabinets per building group.** The star topology groups the buildings
  by bearing around the substation and puts each cabinet on the ray
  through its group's centroid, 1 m beyond the group; a cabinet that would
  stand there farther than `CabinetConfig.max_far_end_gap_m` (20 m) from
  its houses stands on the ray through the group's far-end building
  instead. The threshold is a step, and the results jump where a group
  crosses it. The houses share their group's main (a tree), so results of
  default `TnNetworkGenerator` networks depend on this topology choice.

    **How large:** with the house columns of the 50-house grid 32.0 /
    32.2 m apart one cabinet moves 141.8 m, the drops by −19 %, the input
    resistance at 50 Hz by −0.14 % beyond the smooth trend and, with an
    inductive PEN, the reactance at 1 kHz by −21.6 / −18.9 %
    (`carson_series` / `sommerfeld`). With an inductive PEN the network
    reactance at 1 kHz is +55 … +64 % above the former cabinet strip with
    its parallel drops.

    **Instead:** read per cabinet which rule placed it and its distance on
    the centroid ray (`PlannedSite.cabinet_rule`,
    `PlannedSite.centroid_ray_gap_m`) to see where a group crosses; a given
    `CabinetConfig.placement` fixes the cabinets.

    ??? note "Details and audit trail"

        The groups have equal sizes, starting from the widest gap. A group
        of about 180°, e.g. 10 houses and one small commercial building,
        would put its cabinet 50.4 m from its houses on the centroid ray.
        The threshold is a step, not a separation of two populations (kept
        for 2026.11, user decision D7): groups lie on both sides of it — on
        the centroid ray the groups of the default 50-house grid stand
        8.8 … 12.8 m from their houses (a wide group there, 206 … 341°,
        still has drops up to 113 m to its outer houses), those of the
        100-house grid 10.05 … 23.01 m (15 / 20 / 25 m put four / two / none
        of its five groups on the far-end ray), and over the 80 AP1 mixes
        the 118 groups beyond 20 m stand 20.21 … 53.56 m from their houses
        while the largest gap kept is 18.84 m. The 100-house grid across
        22.95 m moves two cabinets by 65.6 m (X at 1 kHz +1.5 / −3.1 %), and
        80 houses with 5 medium commercial buildings (a group at 20.21 m)
        move one by 31.4 m when the threshold is 20.5 m (X +1.5 / +1.1 %).

        A single group around the substation (one cabinet for a row through
        it) has its cabinet on one side only. A cabinet whose electrodes
        would come closer than `end_clearance_m` to another site raises (3
        of 100 random 200-house layouts 12 m apart); a given
        `CabinetConfig.placement` avoids it. The +55 … +64 % of the tree
        topology are the default star of 30 houses (`sommerfeld` /
        `carson_series`). Planning 8 000 default star houses takes about
        14 s (6.9 s with the former strip). Audits W1R2-02, W1R2-04.

- **Measurement leads** of the in-line set-up are laid side by side; their
  mutual inductance depends on the logarithm of that spacing, which is a
  modelling choice, not a measured value.

- **Overpass** answers are capped at 16 MiB; live queries need network
  access (tests and tutorials use stored answers).

- **Service drops of one tap.** The service drops that leave one tap in
  one direction lie side by side at the touching distance, as cables in one
  trench; a wider spacing is not modelled.

    **How large:** 5 cm apart the 5-house radial network's X at 1 kHz is
    12.5 / 14.6 % lower (`carson_series` / `sommerfeld`).

    ??? note "Details and audit trail"

        The touching distance is user decision D9. Drops that are not
        exactly parallel (placement jitter) lie along one common normal,
        and a drop that would come closer than touching to a drop of a row
        joins it, also a drop of a tap 1 … 4 cm away on the same trunk. No
        generated network reports an overlap in `diagnose()`, also with
        `jitter_m` 0 … 0.1 m (audit W3R2B-01). A drop of a row starts up to
        (n − 1)/2 touching distances off its tap (2 cm in a row of five).
        With a bare PEN, `diagnose()` without an engine warns about those
        starts as "interpenetrating" against the substation ring (the
        10-house radial network: 9 pairs at 0.5 … 5.0 mm) and about the row
        neighbours as "close"; with the default line kernel all of these are
        information (audit W3R2B-07).

- **Ties of the star topology under placement jitter.** Two ties of the
  10-house star move its reactance by steps.

    **How large:** from about 1e-7 m of jitter the cabinet main joins
    `residential_4` at another foundation wire (−0.24 % at 1 kHz), and from
    1e-6 m it runs to the other side (−x → +x, −11.8 % `carson_series`,
    −12.1 % `sommerfeld`).

    **Instead:** expect these steps in a Monte Carlo study over the jitter.

## Diagnostics and problem size

- **Problem size.** More segments or nodal unknowns than the engine's
  limit (`Engine.max_segments`, default `HARD_LIMIT` = 20 000) raise
  [`ResourceLimitError`](warnings.md#resourcelimiterror) before any
  allocation; the dense system grows like $N^2$ in memory and work per
  excitation. The notice that a solve may take minutes comes from
  `Engine.solve` (not from `solve_mutual_*`, which enforce the limit but
  emit no notice) from `Engine.warn_segments` on (default `SOFT_LIMIT` =
  5 000); it is not checked against `max_segments`, so a notice size above
  the hard limit never shows.

    **Instead:** raise the limit per engine to solve a larger world on
    purpose.

- **Touching electrodes.** `diagnose` reports wires of different
  electrodes that are closer than five times the sum of their radii, but
  not two electrodes that meet at a common segment end point (for example
  a rod through a node of a separate grid).

    **Instead:** metal in contact is one galvanic cluster: bond such
    electrodes with a conductor.

- **Aliases.** `cim`, `bem` and `image_nlayer` run `image` and `mom`;
  comparing them with their targets adds no independent information.
