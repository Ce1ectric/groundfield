# Earth return and inductive coupling

This page explains the longitudinal impedance of the conductors of a
grounding study — PEN conductors, cable screens, measurement leads,
interconnections between earth electrodes — and in particular the
**earth-return impedance** that the current returning through the soil
adds to it. Read it before you give conductors an `inductance_model`, when
you choose `Engine.earth_inductive_model`, or when a reactance looks
suspicious.

At 50 Hz, and more so at the harmonics up to 1 kHz that matter for AP2,
conductors are not only resistors: their longitudinal impedance has an
inductive part, they couple magnetically to each other, and the soil
closes the loop. The galvanic part of the problem (leakage, potentials in
the soil) is not affected; it is explained on the [engine pages](engines/index.md).

## Mathematical and physical context

A conductor with `inductance_model="neumann"` is split into branches (one
per sub-segment of `discretize_segment_length`, or one branch for a lumped
conductor). Their longitudinal impedance matrix enters the network
equations of the [engine family](engines/index.md#network-equations):

$$
Z_b(\omega) = R + j\omega L + \Delta Z(\omega) ,
$$

the voltage drop along each branch per unit current in every branch, at
the angular frequency $\omega = 2\pi f$. Its three parts:

- $R$ — the diagonal of the DC resistances of the branches; with
  `resistance_model="skin"` a frequency-dependent internal-impedance
  correction is added (see [skin effect](#skin-effect)).
- $L$ — the partial-inductance matrix. Between two branches $i$ and $j$
  it is Neumann's double line integral
  $L_{ij} = \frac{\mu_0}{4\pi}\int_i\int_j \frac{\mathrm d\mathbf l_i\cdot\mathrm d\mathbf l_j}{r}$,
  the magnetic flux that a unit current in one branch links with the other
  ($\mu_0$ the permeability of free space, $r$ the distance of the two
  line elements). `groundfield` evaluates it in closed form for parallel,
  skew, touching and coaxial pairs (Grover 1946). The self terms are those
  of a thin wire with the DC internal inductance $\mu_0\ell/(8\pi)$ of a
  solid round conductor of length $\ell$ (none for a cable shield,
  `conductor_type="cable_shield"`, a thin-walled tube). To $L$ belongs the
  **image** of the earth model (below).
- $\Delta Z$ — the earth-return correction: how much the finite
  conductivity of the soil changes the magnetic field of that image.

The physical scale is the skin depth of the soil,

$$
\delta = \sqrt{\frac{2\rho}{\omega\mu_0}} \approx 503\,\sqrt{\frac{\rho}{f}}\ \text{m},
$$

the depth over which an alternating field in soil of resistivity $\rho$
decays by $1/e$. In the frequency range of the package it lies between
about 110 m (50 Ω·m, 1 kHz) and 1.6 km (500 Ω·m, 50 Hz): comparable to or
larger than the extent of a low-voltage network. The return current
therefore spreads deep and wide, and neither a perfect mirror nor a
perfectly conducting earth describes it.
`FieldResult.metadata["penetration_depth"]` reports $\delta$ for every
frequency of a solve with inductive conductors.

The soil-side galvanic coupling (leakage, potentials) is unaffected: the
field kernel stays real and quasi-static, the earth return acts on the
longitudinal impedance of conductors only.

??? note "Corrected statement"

    Earlier versions of this page gave the range of $\delta$ as "about 350 m
    (50 Ω·m, 1 kHz) and 5 km (500 Ω·m, 50 Hz)". The formula above gives
    112.5 m and 1591.5 m there (`groundfield.coupling.carson.skin_depth`
    agrees); the 159 m at 1 kHz in 100 Ω·m quoted below were right.

## The three earth models

`Engine.earth_inductive_model` selects the model. Each is the combination
of an image convention for $L$ and a correction $\Delta Z$ defined against
that very image — an earth-return correction is meaningless without the
baseline it was derived for:

| `earth_inductive_model` | Image in $L$ | Correction $\Delta Z$ | Use |
|---|---|---|---|
| `"sommerfeld"` (default since 0.13.0) | none (free space) | full Sommerfeld / Pollaczek kernel minus free space, every segment pair | quantitative $Z(f)$, any geometry, layered earth |
| `"carson_series"` | anti-parallel (perfectly conducting earth) | Carson per metre on near-parallel horizontal pairs, Sommerfeld otherwise ([below](#carson-on-its-own-baseline)) | long parallel lines over homogeneous earth; fast |
| `"perfect_mirror"` | additive (magnetic mirror) | none | diagnostic baseline only |

### Sommerfeld (Pollaczek kernel)

The vector potential of an alternating line current in or above a
conducting half-space is a Sommerfeld integral (Sommerfeld 1909;
Pollaczek 1926 for the buried conductor, Carson 1926 for the overhead
one). Take the time convention $e^{+j\omega t}$, the soil conductivity
$\sigma = 1/\rho$, the propagation constant $\gamma$ with
$\gamma^2 = j\omega\mu_0\sigma$, the radial wavenumber $\lambda$ of the
Hankel transform and $u = \sqrt{\lambda^2 + \gamma^2}$. For two points at
depths $z$ and $z'$ in homogeneous soil, at the horizontal distance $s$
and the distance $R$, the Green's function of the vector potential is

$$
G(s, z, z') = \frac{e^{-\gamma R}}{R}
+ \int_0^\infty \frac{\lambda}{u}\,\frac{u - \lambda}{u + \lambda}\,
e^{-u(z+z')}\,J_0(\lambda s)\,\mathrm d\lambda :
$$

the field of the source in an unbounded conducting medium, plus the field
reflected at the soil surface, written as a superposition of cylindrical
waves ($J_0$ the Bessel function of order zero). The factor
$(u - \lambda)/(u + \lambda)$ is the reflection coefficient of the
insulating air. The earth-return correction of two segments is the
change of their mutual impedance against free space,

$$
\Delta Z_{ij} = j\omega\frac{\mu_0}{4\pi}\int_i\int_j \Bigl(G - \frac{1}{R}\Bigr)\,
\mathrm d\mathbf l_i\cdot\mathrm d\mathbf l_j ,
$$

integrated over the actual geometry of each segment pair — any length,
orientation and depth — and added to an image-free Neumann matrix. For
layered earth the spectral amplitude carries the generalised reflection
coefficients of the stack, and overhead pairs keep Carson's air-side
geometry; the amplitudes are documented in
`groundfield.coupling.sommerfeld_inductance`. In the long-wire limit the
correction gives Pollaczek's per-metre impedance, including the positive
earth-return resistance of about $\omega\mu_0/8$ per metre; for
$\sigma \to 0$ it vanishes (the soil is magnetically transparent).

#### Numerics of the Sommerfeld correction

- **Image-free baseline.** The correction is built against free space, so
  no image has to be added and removed again. Conductors that cross the
  soil surface or lie in it are solved (the pair integrals are split at
  $z = 0$); with the image conventions of the other models such a segment
  is singular and raises.
- **Exact layered kernel.** For any number of layers the same-layer
  amplitude uses generalised reflection coefficients and all other pairs a
  batched interface solve; a three-layer earth whose lower two layers are
  equal equals the two-layer earth exactly.
- **Resolved $\lambda$ grid.** The logarithmic region of the quadrature in
  $\lambda$ ends before $J_0$ oscillates, the linear region carries 16
  nodes per $J_0$ period up to four times the largest physical scale, and
  an exhausted panel budget warns (`SommerfeldResolutionWarning`) instead
  of truncating silently.
- **Multi-frequency assembly.** Geometry, $\lambda$ grid and Bessel tables
  are built once per pair for the whole frequency set of a solve,
  bit-identical to single-frequency assemblies, with a bounded cache.
- **Long segments.** A single 16 × 16 rule over a segment cannot resolve
  $e^{-\gamma R}$ along a segment much longer than the skin depth. A pair
  of segments longer than a quarter skin depth of the most conductive
  layer gets a composite outer rule (16 nodes per piece, at most 512
  pieces, then a `SommerfeldResolutionWarning`) and a denser table of the
  spectral integral over the horizontal distance.
- **Self terms and close pairs** (user decision Q34). Along the line of
  closest approach of a segment with itself, or of two segments that come
  close (wires beside each other, a core in its shield), the integrand has
  a kink — the $\gamma^2R/2$ term of $(e^{-\gamma R}-1)/R$ and its
  counterpart in the spectral part, smoothed only over the decay length
  $z + z'$ — which a tensor-product rule cannot follow. The self term of
  every segment is therefore the exact one-dimensional reduction
  $2\int_0^L (L-u)\,g(u)\,\mathrm du$ of its double integral ($g(u)$ the
  integrand at the axial separation $u$), with 16-node panels graded from
  $z + z'$ up to a quarter skin depth, uniform up to 40 skin depths and
  doubling beyond. Close pairs (closest distance below half the longer
  segment) get a graded rule along their line of closest approach (a
  coinciding pair: the self rule). The self correction is within 1.1e-9 of
  a refined rule and of an independent reference from 1 m and 0.01 skin
  depths to 3 km at 50 Hz and 1 kHz (buried at 0.7 m in 100, 1000/30 and
  30/1000 Ω·m, overhead at 10 m over 10 Ω·m) and continuous in the length
  (steps below 3e-13); close pairs are within about 4e-8 (3.5e-8 for two
  1 m wires 1 mm apart in 30 over 1000 Ω·m at 1 kHz). Both rest on the
  256-node logarithmic region of the $\lambda$ grid.
- **Very long conductors.** Both rules cost at most linearly in the
  length: one lumped conductor of 2 km from a rod to a junction, 50 Hz and
  1 kHz, is solved in 1.1 s within 0.21 GB, 100 km in 10.6 s within
  0.24 GB. **Beyond about 33 km** such a conductor is not converged and
  the solve warns (`SommerfeldResolutionWarning`; the resolved $\lambda$
  grid reaches $2^{17}\cdot 2\pi\,(z + z')/30$, 38 km for a conductor at
  0.7 m).
- **Tabulated assembly** (the default since 2026.11,
  `Engine.sommerfeld_assembly="tabulated"`; user decision Q16).
  Sub-segments are grouped by depth, and one table of the spectral integral
  per pair of depth groups and frequency replaces the per-pair quadrature.
  It is much faster on large networks (a 30-house TN network 306 s →
  12 s) and agrees with a converged reference better than the per-pair
  rule (`sommerfeld_assembly="exact"`), whose 24-node interpolation in the
  horizontal distance dominates its error (2e-10 against 1.2e-8 of two
  refined references). Measured times are on the
  [performance page](performance.md).

??? note "Measurements and audit trail"

    What 2026.11 changed (ADR-0006 amendment):

    - **Image-free baseline** (findings 10.46, F52): the numbers equal the
      former additive pairing to round-off.
    - **Exact layered kernel** (10.35): a three-layer earth whose lower two
      layers are equal was +17…+25 % in $X'$ before.
    - **Resolved $\lambda$ grid** (10.36): distant pair entries were off by
      −24 % (200 m) and +274 % (400 m) before.
    - **Multi-frequency assembly** (10.60).
    - **Long segments** (10.24, audits of 2026.11): a lumped 3 km PEN read
      +13.4 % earth-return resistance at 1 kHz in 100 Ω·m.
    - **Self terms and close pairs** (user decision Q34): up to 2026.11.0
      the self correction of a buried segment was 4.8e-4 off at a quarter
      skin depth (growing like $L/\delta$ below it), stepped by 3.5e-4
      where the composite rule took over, and was within 5.9e-4 for longer
      segments; two wires 1 mm apart were 1.5e-3 off in 30 over
      1000 Ω·m. With the 64 nodes of earlier builds in the logarithmic
      region of the $\lambda$ grid, short segments were 1e-5 … 7.4e-5 off
      at 0.01 skin depths and up to 3e-4 at 1 m, about 1e-8 of a solved
      impedance (audit W3B-01).
    - **Very long conductors**: the 2 km conductor took 28 s and 1.6 GB,
      the 100 km one more than 7.8 GB (audit W2B-05); the limit at about
      33 km is audit W3B-10.

### Carson on its own baseline

Carson (1926) derived the earth-return impedance of a line over
homogeneous earth as a correction to the impedance of the line and its
**anti-parallel** image in a perfectly conducting plane. The correction is
$(\omega\mu_0/\pi)(P + jQ)$ per metre, where Carson's series $P$ and $Q$
depend on the dimensionless parameter $a = D\sqrt{\omega\mu_0\sigma}$ —
$D$ the distance between conductor and image, $2h$ for a self term at
height or depth $h$ — and on the angle of the image direction. It is a
per-metre result for infinitely long parallel horizontal wires, so
`carson_series` uses it only inside that validity domain:

- **Near-parallel horizontal pairs on one side of the surface** —
  $\lvert\cos\varphi\rvert > 0.99$ ($\varphi$ the angle between the two
  segments) and both segments inclined by less than 8.1° — get the
  per-metre correction at their horizontal axis distance, times the length
  over which their projections overlap; the self term of a horizontal
  segment is the per-metre value times its length. Pairs that do not
  overlap (staggered, end to end) and collinear pieces of one wire get
  none: the per-metre self value on the diagonal already contains the rest
  of the same wire. The tiling converges under refinement. Two conductors
  that overlap on one axis are wires lying side by side and get the
  correction at the distance of touching wires, $r_i + r_j$ (the sum of
  their radii).
- **Pairs across the surface** — an overhead wire parallel to a buried
  one — have no anti-parallel baseline (a perfect conductor screens the
  soil). They couple through the field transmitted into the soil,
  `carson_mixed_mutual_correction`: Carson's $J$ plus the bounded
  remainder of the buried wire's attenuation (Ametani et al. 2009).
- **Tilted near-parallel pairs.** Where a near-parallel pair comes close
  within its overlap (its smallest distance there is below 30 times the
  change of its offset), the per-metre correction is integrated over the
  overlap (Gauss–Legendre nodes) instead of being taken at the mid-point
  distance, so a lumped pair and its pieces add up: a core crossing its
  shield's axis at mid-length differs by at most 8.0e-6 of the loop
  reactance. Other pairs keep one entry at the mid-point distance, exact
  to second order there.
- **Everything else** — oblique pairs and every pair and self term of a
  segment inclined by more than 8.1° (risers, sloping leads) — gets the
  exact Sommerfeld correction of a homogeneous earth of the same
  conductivity, moved onto the anti-parallel baseline.

At the 8.1° boundary the result steps from the per-metre tiling to the
Sommerfeld value, by the error of the per-metre model there: 13 % for two
1 km leads 2 m apart at 1 kHz, more for shorter leads (42 % for two 300 m
leads at 50 Hz). That step is the price of keeping Carson's closed form
where it is accurate; for oblique or short conductors use `sommerfeld`.

On this baseline a 1 km isolated PEN reproduces $l\,X'_\text{Carson}$
within 1 %
(`tests/test_inductive_solver_integration.py::test_carson_series_reactance_matches_carson`).
Carson's small-$a$ series is evaluated to $a^{12}$ and the self-term
quadrature at $\theta = 0$ to round-off, so the regime boundaries are
continuous; the asymptotic expansion is used for $a > 15$.

`carson_series` remains an approximation for short, inclined or buried
conductors and for layered soil (an effective homogeneous conductivity
is used); it is the fast choice for long parallel lines over homogeneous
earth, not a replacement of `sommerfeld` in general.

**Networks of short cables.** Tiled over the pieces of service drops
that are shorter than the skin depth (30 … 60 m against 159 m at 1 kHz
in 100 Ω·m), Carson's infinite-line correction overstates the earth
return of an LV network. For the generated star, radial and grid
networks of 5 … 100 houses with an insulated Neumann PEN,
`carson_series` reads the input reactance 4 … 21 % and the resistance up
to 11 % high at 1 kHz, and the reactance 21 … 50 % and the resistance up
to 0.9 % high at 50 Hz; at 50 Hz it is then no closer than the
diagnostic `perfect_mirror` for four of the five networks measured
(note below). A change of such
a network is reproduced by `carson_series` only within that error,
possibly with the opposite sign. Use `sommerfeld` for LV networks.

??? note "Measurements and audit trail"

    - **Validity domain** of `carson_series`: since the second adversarial
      audit of 2026.11 (A2C-1). Up to then an oblique pair got
      $\sqrt{\ell_i\ell_j}\,\lvert\cos\varphi\rvert$ times the per-metre
      value once per segment pair, which grew with the number of segments
      (two 1 km leads at 9°: 20× the Sommerfeld coupling). Pairs across the
      surface: the solver used to treat the buried wire as an overhead wire
      at height $\lvert z\rvert$, which gave 36 % too little mutual
      reactance for a lead 1 m above a PEN at 0.7 m depth at 1 kHz.
    - **Tilted near-parallel pairs** (audit W1R3-07): before, a core
      crossing its shield's axis at mid-length differed by 4.3e-4 of the
      loop reactance at 2e-3 rad and 5.2e-2 at 4e-2 rad; generated networks
      are unchanged by the rule.
    - **0.15.0** (finding 10.20 / F32): the solver added Carson's
      correction to the **additive** image matrix of the perfect mirror,
      which roughly doubled the earth-return reactance: 1.66–2.25 times
      Carson's value for 100 m – 1 km lines, +41 % on $\lvert Z\rvert$ at
      1 kHz for a TN network with inductive PEN conductors. Since 2026.11
      the matrix is built on the anti-parallel image.
    - **Networks of short cables** (audits W3B-02, W3R2B-03). Soils:
      100 Ω·m and two-layer soils of 30 … 200 Ω·m. Reference: `sommerfeld`
      with a refined discretisation (PEN in 2.5 m pieces, 1 m segments); the
      default `sommerfeld` agrees with it to 1e-3 for the 5-house radial
      network and to 4e-5 for the 80- and 100-house ones. Per network, input
      reactance at 1 kHz / 50 Hz: the 5-house radial network +13 % / +31 %,
      the 10-house radial network +20 % / +41 %, the 30-house star
      +17 % / +44 %, the 80-house radial network +19 % / +50 %; the
      100-house radial network R +11 % / +0.9 %. At 50 Hz `carson_series`
      is no closer than `perfect_mirror` for the 10-house radial network,
      the 30-house star and the 80- and 100-house radial networks
      (`perfect_mirror` +40 / +37 / +41 / +41 % there), but closer for the
      5-house radial network (`perfect_mirror` +45 %). The error changes
      with the geometry: across the cabinet fix of audit W1B-01/02 the
      star's reactance moves by +1.18 % under `carson_series` and −3.54 %
      under `sommerfeld` (the refined reference: −3.54 %), between two house
      spacings around the cabinet threshold of the 100-house grid by
      +1.81 / −2.77 % (reference −2.77 %), and of the 50-house grid by
      −21.6 / −18.9 % (reference −18.9 %).

### Equivalent depth of layered soil

The complex-depth formulas of Dubanton and Deri replace a homogeneous
earth by a perfectly conducting plane at the complex depth
$p = \sqrt{\rho/(j\omega\mu_0)}$ and give the line impedance in closed
form. `gf.penetration_depth(soil, frequencies)` returns the $p$ (and the
equivalent resistivity $j\omega\mu_0 p^2$) for which these formulas
reproduce the earth return of any layer stack: the plane-wave surface
impedance of the stack over $j\omega\mu_0$, which matches the layered
surface reflection coefficient to second order in the horizontal
wavenumber. Over the 39 soils of the AP1 grid, 50 Hz … 1 kHz and 0.1 …
1000 m, the earth-return correction of two overhead or buried wires is
reproduced within a median of 2.2 % (the formula's own median error in
homogeneous soil: 2.4 %), at most 28 … 33 % where the separation
approaches $\lvert p\rvert$; the homogeneous depth of $\rho_1$ is off by up
to 2800 %. Method and numbers: [Soil](../reference/api/soil.md#penetration-depth).
The solver's models do not use it.

### Perfect mirror

`perfect_mirror` treats the earth as a perfect magnetic mirror (additive
image, no correction). It is cheap and frequency independent but far from
the physics of buried conductors below 1 kHz; it was the default before
0.13.0 and is kept as a baseline for comparisons.

Its image is applied to every pair of inductive segments, also to a pair
of which one segment lies above the soil surface ($z < 0$, an overhead
lead) and one below it (a buried PEN or cable). Such a pair has no
magnetic mirror between them in any physical sense, so the additive image
has no basis there. The numbers are kept as they are (a baseline), but the
solve emits one `UserWarning` that counts these pairs, names the
conductors on each side and points to `"sommerfeld"`; `carson_series` and
`sommerfeld` treat such pairs through the field transmitted into the soil
and do not warn. Segments in or across the surface plane are rejected
under this image convention.

??? note "Measurements and audit trail"

    The warning on pairs across the surface exists since 2026.11 (user
    decision Q27).

## Conductor details

- **Touching segments.** Chain neighbours and bends share an end point
  where the Neumann integrand is logarithmically singular; Grover's vertex
  formula and the coaxial gap formula replace the 16 × 16 Gauss–Legendre
  rule by default (`touching="exact"`), so a straight conductor keeps the
  inductance of the unsplit wire for every discretisation
  (`tests/test_inductive_solver_integration.py::test_inductance_is_refinement_invariant_for_a_straight_conductor`).
  `touching="quadrature"` reproduces 0.15.0.
- **Lumped and single-segment conductors.** A conductor with an
  `inductance_model` that the topology keeps as one branch is one
  inductive branch from start to end, coupled to all others.
- **Close pairs.** Pairs that miss the closed forms — nearly parallel
  wires close together, crossing and skew pairs, a branch starting on
  another segment — are integrated semi-analytically (inner integral in
  closed form, outer Gauss–Legendre panels graded towards the
  near-singular points). Measured against mpmath: 5e-11 for directions
  that differ by more than 1e-6 rad, 1.0e-9 for filaments crossing at
  1e-8 rad; filaments from one vertex to round-off on axis-parallel
  geometry (4e-10 in a general orientation).
- **Wires beside each other.** Where a point of one wire lies beside
  another wire — its foot on that wire's axis falls inside it — closer than
  the sum of the radii $r_i + r_j$, the filament integral is meaningless:
  it diverges for coincident axes (two coincident 50 m leads read 1.87 H)
  and exceeds the self inductance once the wires are closer than about
  their radius, which makes the inductance matrix indefinite (a network
  whose service drops ran back along the PEN main had a negative input
  reactance). There the distance from the other axis is taken as
  $r_i + r_j$ — two wires touching side by side, as two cables in one
  trench lie — pointwise inside the Neumann integral, both one-sided
  integrals averaged.

    *A core inside a shield.* Where the thinner wire runs inside the
    thicker one and the thicker one is a cable shield
    (`conductor_type="cable_shield"`, user decision Q32), the shield radius
    is used instead: a core inside a shield is a filament inside a thin
    tube and couples to it as at the tube radius, and since the shield
    carries no internal inductance (user decision Q33) the loop gives the
    coaxial inductance per metre

    $$
    L'_\text{coax} = \frac{\mu_0}{2\pi}\Bigl(\ln\frac{r_s}{r_c} + \frac14\Bigr),
    $$

    the flux between the core of radius $r_c$ and the shield of radius
    $r_s$ plus the internal inductance of the solid core: 26.97 µH for
    100 m with a 5 mm core and a 15 mm shield (31.97 µH with the internal
    inductance of a solid shield). Two unmarked conductors on one axis are
    separate cables and touch (43.47 µH for the same pair).

    *Pieces add up to the lumped pair.* A wire inside a shield couples at
    the shield radius at every axial offset, and for two other conductors
    lying beside each other a point lies beside the partner when its foot
    falls inside the partner's whole conductor: the sub-pieces of the two
    conductors then add up to the lumped pair (within 3e-12). That holds
    for pieces that are not parallel too, at any angle and whatever their
    length — a core tilted against its shield or crossing its axis, wires
    leaving a junction or crossing at a small angle: the stretches inside,
    beside and away from the partner are found exactly, and a core leaving
    the shield's tube along its length adds up as well. The value does not
    depend on the order or the direction of the segments, is continuous in
    the geometry, and leaves chain neighbours, bends turning by less than
    90° and pairs farther apart unchanged. For unequal radii it steps where
    the thinner axis leaves the thicker wire, from "inside" to "side by
    side": the two describe different cables. Equal radii never count as
    one wire inside the other.

    *At the low level*, `build_inductance_matrix(..., shields=[False, True],
    include_internal=[True, False], conductors=[0, 1])` does the same. The
    marker leaves the self terms alone, so a shield needs
    `include_internal=False` (else it keeps the 50 nH/m of a solid wire,
    31.97 µH), and `conductors` labels the pieces of each conductor
    (without it every segment is a conductor of its own: unmarked touching
    pieces of one cable then do not add up, 40.34 instead of 43.47 µH in
    0.25 m pieces, while a marked shield needs no labels).
- **Diagnostics.** `conductors.find_close_wires` and `diagnose()` report
  crossing, touching, interpenetrating and close wires; `diagnose()` also
  reports inductive conductors that lie beside each other closer than
  touching over a stretch (category `overlap`), the pairs the rule above
  evaluates as touching wires.

??? note "Measurements and audit trail"

    - **Touching segments, close pairs, wires beside each other:** finding
      10.46. With the 16 × 16 rule close pairs were +12 % at 10 cm and up to
      +250 % at 1 cm for two 5 m segments.
    - **Lumped and single-segment conductors** (finding 10.24): up to
      0.15.0 their inductance was silently dropped (a 300 m PEN read
      0.0564 Ω at 1 kHz instead of 0.210 + 3.78j Ω).
    - **Shields:** before user decision Q33 a shield carried the internal
      inductance of a solid wire (31.97 µH for the coax above).
    - **Pieces adding up to the lumped pair** (audit W1A-02; before, the
      coax in 1 m pieces read 2.2 % low). Audits W1R2-01, W1R3-01 and
      W1R3-02 extended the rule to tilted and crossing pieces: a core tilted
      by 4.5e-5 rad read 7.4 % low in 0.25 m pieces, with a step at the
      parallel tolerance; until W1R3 the rule acted only while the lateral
      distance changed by at most $r_i + r_j$ along a piece and for
      $\lvert\cos\varphi\rvert \ge 0.99$, and beyond either bound the rule
      of the single segments mixed in: a core crossing the shield's axis
      at mid-length read −8.6e-5 of the loop off the lumped pair in 10 m
      pieces, shield pieces of mixed length −1.6e-3, two wires leaving a
      junction stepped by 1.5e-3 at 8.1°. The `conductors` labels of
      `build_inductance_matrix`: audit W1R2-08.
    - **Diagnostics:** finding 10.44; the `overlap` category of
      `diagnose()` exists since 2026.11.

### Skin effect

With `Conductor.resistance_model="skin"` (opt-in) every branch of the
conductor adds $\ell_b\,\Delta Z'_\text{skin}(f)$ to its diagonal ($\ell_b$
the branch length): the exact internal impedance per metre of a solid
round conductor of radius $a$ and resistivity $\rho$,

$$
Z'_\text{int} = \frac{k\rho}{2\pi a}\,\frac{I_0(ka)}{I_1(ka)},\qquad
k = \sqrt{j\omega\mu_0/\rho},
$$

which describes the current crowding towards the surface of the conductor
($I_0$, $I_1$ the modified Bessel functions of the first kind), taken
relative to the DC convention ($R'_\text{dc}$ plus the internal inductance
$\mu_0/(8\pi)$ already in $L$). For 150 mm² aluminium the resistance rises
by the factors 1.056, 1.231 and 1.550 at 250, 550 and 1000 Hz and the
internal inductance falls to 0.740 of its DC value at 1 kHz
(`tests/test_skin_effect.py`); the IEC 60287-1-1 skin factor agrees to
about 1 %. A conductor without an `inductance_model` has no inductance
at all, so only the resistance part applies to it. The default
`resistance_model="dc"` keeps the numbers and the JSON of earlier
releases.

??? note "Measurements and audit trail"

    The skin-effect model is finding 10.37 of review pass 10.

## Validation

- **Carson:** the solver's 1 km PEN against $l\,X'_\text{Carson}$
  (test above); Carson's $P$ and $Q$ against an independent
  `mpmath` quadrature across all regimes (`tests/test_carson_pq.py`).
- **Sommerfeld against Pollaczek:** benchmark B13 takes the per-metre
  series impedance of an isolated line from the difference of a 1600 m
  and an 800 m line (which cancels the end effects) and compares it with
  the conductor-earth loop impedance of `references.pollaczek`; the
  generated numbers are on the [validation page](validation.md) (rows
  `pollaczek_x`, `pollaczek_r`). Distant pair entries against an
  independent quadrature: `tests/test_sommerfeld_accuracy.py`.
- **`references.pollaczek`** itself uses the exact two-layer amplitude
  and the $K_0$ head and agrees with an independent boundary-value
  solution to $10^{-7}$ (`tests/test_pollaczek_layered.py`).
- **Engine level:** `compare_engines` compares complex impedances at
  every frequency, so a reactance mismatch between earth models cannot
  pass as consistent (benchmark B13 row `image_vs_mom`).

??? note "Measurements and audit trail"

    The exact two-layer amplitude of `references.pollaczek` is finding
    10.45 (since 2026.11), the complex comparison of `compare_engines`
    finding 10.71.

## Literature

- Sommerfeld, A. (1909). Über die Ausbreitung der Wellen in der drahtlosen
  Telegraphie. *Annalen der Physik* 28(4), 665–736.
- Carson, J. R. (1926). Wave propagation in overhead wires with ground
  return. *Bell System Technical Journal* 5(4), 539–554.
- Pollaczek, F. (1926). Über das Feld einer unendlich langen
  wechselstromdurchflossenen Einfachleitung. *Elektrische
  Nachrichtentechnik* 3(9), 339–360.
- Schelkunoff, S. A. (1934). The electromagnetic theory of coaxial
  transmission lines and cylindrical shields. *Bell System Technical
  Journal* 13(4), 532–579 (internal impedance of round conductors).
- Grover, F. W. (1946). *Inductance Calculations: Working Formulas and
  Tables*. Van Nostrand; Dover reprint.
- Wait, J. R. (1972). *Electromagnetic Waves in Stratified Media*.
  Pergamon, Oxford (reflection coefficients of layered media).
- Dommel, H. W. (1986). *EMTP Theory Book*. Bonneville Power
  Administration, Portland (Carson's series coefficients and sign rule).
- Oeding, D., & Oswald, B. R. (2016). *Elektrische Kraftwerke und Netze*,
  8th ed. Springer, Berlin (earth-return conductors; inductance
  references of `references.oeding`).
- IEC 60287-1-1. *Electric cables — Calculation of the current rating —
  Part 1-1: Current rating equations (100 % load factor) and calculation
  of losses — General*.

Architecture decisions: [ADR-0003](../adr/0003-distributed-conductor-model.md)
(distributed conductors), [ADR-0004](../adr/0004-inductive-coupling.md)
(inductive coupling), [ADR-0005](../adr/0005-carson-earth-return.md)
(Carson), [ADR-0006](../adr/0006-sommerfeld-earth-return.md)
(Sommerfeld), each with its 2026.11 amendment.
