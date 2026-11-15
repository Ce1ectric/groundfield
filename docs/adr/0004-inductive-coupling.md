# ADR-0004: Inductive coupling between conductor segments

| | |
|---|---|
| **Status** | Accepted — amended 2026.11 (review pass 10: image conventions, touching pairs, lumped conductors, skin effect); see the amendment at the end of this document |
| **Date** | 2026-05-04 |
| **Deciders** | Project maintainers |
| **Scope** | `groundfield` |

## Context

ADR-0003 introduced the distributed-conductor model. Each conductor
is split into $n$ longitudinal sub-segments, and the resulting
nodal-analysis system carries one branch per segment with series
impedance

$$
Z_\text{long}^{(k)} \;=\; R^{(k)} \;+\; j\omega\, L^{(k)}.
$$

ADR-0003 deferred $L^{(k)}$ to a follow-up step. With
$L^{(k)} = 0$ the system is real and frequency-independent, and the
``engine.frequencies`` list returns the same DC solution for every
entry. This is fine for DC-near grounding-impedance studies but
**not** for analysing **inductive coupling between the measurement
lead and the current injection** — the very effect that forced the
distributed-conductor refactor in the first place.

This ADR settles how the inductive part is added on top of the
distributed model.

## Decision

### Physical model

Self- and mutual-inductance of conductor segments are computed from
**Neumann's double-line integral** (Grover 1946, Sunde 1968 ch. 7,
Paul 2010 ch. 5):

$$
M_{ij} \;=\; \frac{\mu_0}{4\pi}
\oint_{C_i} \oint_{C_j}
\frac{d\vec{\ell}_i \cdot d\vec{\ell}_j}{r_{ij}}.
$$

For two **straight segments** of equal length $\ell$ that share the
same axis direction at perpendicular distance $d$, the closed form
is

$$
M_\parallel(\ell, d) \;=\; \frac{\mu_0\,\ell}{2\pi}
\Bigl[\ln\!\Bigl(\frac{\ell + \sqrt{\ell^2 + d^2}}{d}\Bigr)
      - \frac{\sqrt{\ell^2 + d^2} - d}{\ell}\Bigr].
$$

For arbitrary 3-D orientations and unequal lengths the implementation
falls back to a **two-point Gauss–Legendre quadrature** of the
double integral on each segment pair — sufficient because
$1/r_{ij}$ varies slowly over a single sub-segment, and the
quadrature is cheap (constant cost per pair, total cost
$\mathcal{O}(M^2)$ for $M$ segments). For **the segment with
itself** we use the thin-wire approximation

$$
L_\text{self} \;\approx\; \frac{\mu_0\,\ell}{2\pi}
\Bigl[\ln\!\Bigl(\frac{2\ell}{a}\Bigr) - 1\Bigr],
$$

with $a$ the wire radius. This formula is bounded between
$\ell/a > 10$ (thin-wire regime), which holds comfortably for the
typical cable runs ($\ell$ ≈ 1–5 m, $a$ ≈ 4 mm).

### Earth return ("ideal earth")

In this ADR the earth is treated as a **perfect mirror** for the
magnetic field ($\sigma_\text{earth} \to \infty$): the image of a
segment at depth $z_s$ contributes its own Neumann integral against
all other segments' real and image positions, exactly as in the
electric image-charge sum used by the existing solvers. This is the
same approximation already used by Sunde 1968 ch. 7.4 for buried
power lines below 1 kHz; the **finite earth conductivity correction
(Carson)** is the subject of a follow-up ADR.

### Schema

`Conductor` gains one new field:

<!-- skip-doctest: schema sketch, not runnable code -->
```python
inductance_model: Literal[None, "neumann"]   # default None
```

- `None` (default) — backwards-compatible: $L^{(k)} = 0$, the
  longitudinal branch is purely resistive, and the system is real
  per frequency (existing behaviour).
- `"neumann"` — the segments contribute self- and mutual-inductance
  to the branch impedance via the formulas above. The longitudinal
  block of the augmented system becomes complex:

$$
Z_b(\omega) \;=\; R + j\omega\, L \in \mathbb{C}^{M \times M},
$$

with $L$ a dense, **symmetric, positive-definite** matrix over the
distributed-conductor branches. Mutual coupling exists between
**every pair** of segments — not only within the same conductor —
so that a measurement lead routed parallel to the current-injection
lead automatically picks up its mutual inductance.

### Linear system

The block structure of the augmented system is unchanged from
ADR-0003:

$$
\begin{bmatrix}
   Z_g & -C_s & 0 \\
   C_s^{\top} & 0 & B^{\top} \\
   0 & B & -Z_b(\omega)
\end{bmatrix}
\begin{bmatrix}
   \mathbf{I}_\text{leak} \\
   \boldsymbol{\varphi}_n \\
   \mathbf{I}_\text{long}
\end{bmatrix}
\;=\;
\begin{bmatrix}
   \mathbf{0} \\
   \mathbf{I}_\text{in} \\
   \mathbf{0}
\end{bmatrix}.
$$

The earth Green's function block $Z_g$ stays real and
frequency-independent (quasi-static, $f < 1\,\mathrm{kHz}$) — the
finite-frequency correction would have to come from a magnetic
Green's function on the layered soil, which is again Carson territory
(ADR-0005). The $Z_b(\omega)$ block carries all the
frequency-dependent physics introduced here.

### Frequency loop

When at least one conductor has `inductance_model == "neumann"`, the
solver:

1. Discretises the geometry **once** and assembles $Z_g$, the
   resistance vector $R$, and the inductance matrix $L$ — all
   frequency-independent.
2. Loops over `engine.frequencies`. For each $f_k$ the
   $\omega = 2\pi f_k$ is plugged into $Z_b(\omega)$ and the full
   complex linear system is solved with `numpy.linalg.solve` on the
   complex augmented matrix.
3. Returns one complex potential / current entry per frequency
   in `FieldResult.electrode_potentials[name][k]` etc. — the
   `FieldResult` shape is unchanged; entries that used to be real
   per frequency are now genuinely complex.

When **all** conductors have `inductance_model is None` (the
default), the solver keeps the historic real-only fast path: one
`linalg.solve` over the real augmented matrix, the result broadcast
across `engine.frequencies`. Quasi-static DC studies stay
bit-exact and at the same cost.

### Self-inductance and shielded conductors

The thin-wire self-inductance assumes a **bare** conductor. For a
cable with a conductor inside an outer sheath the relevant
self-inductance is geometrically smaller; this distinction matters
for shielded MV cables but not for the bare-copper / PEN typical study.
We keep the model simple: `inductance_model == "neumann"` always
uses the bare-conductor self formula; refinements come with the
cable-shield work later.

## Validation

- **Single isolated segment** — the segment's self-L matches the
  closed-form Grover formula above to within 0.5 % (we sweep
  $\ell/a$ from 10 to 1000).
- **Two parallel coaxial segments** — the Neumann quadrature
  reproduces $M_\parallel(\ell, d)$ to within 1 % for separations
  $d / \ell \in [0.1, 10]$.
- **Two perpendicular segments** crossing in the middle — the
  Neumann integral evaluates to zero by symmetry (test tolerance
  $10^{-9}$).
- **DC reproducibility** — at $\omega = 0$ the inductive system
  collapses bit-exact to the resistive system from ADR-0003
  (regression).
- **Cross-engine** at $f = 50\,\mathrm{Hz}$ — image, mom, cim, bem
  agree on the cluster impedance of a galvanic distributed
  conductor with the inductive model active to within 5 %.
- **Loop coupling** — two parallel galvanic conductors (one current
  injection, one open-circuit measurement lead) show the expected
  finite open-circuit voltage at 50 Hz that scales linearly with
  the source current and with $\omega$.

## Consequences

### Positive

- Covers inductive coupling between the measurement lead and the
  current injection. The model is now physically complete for
  $f < 1\,\mathrm{kHz}$ except for the Carson earth-return
  correction.
- The Neumann integral is **geometric only** — it works on top of
  any of the existing electric Green's-function backends without
  duplicating their kernel logic.
- Frequency-resolved `FieldResult`s without API change: the same
  `electrode_potentials[name][k]` shape now actually depends on
  $f_k$.

### Negative

- The system becomes complex when the inductive model is enabled,
  doubling the per-frequency LU factorisation cost. A frequency
  sweep with $N_f$ frequencies costs $N_f$ complex solves (vs. one
  real solve in the resistive-only case). For a typical sweep
  ($N_f \le 20$, $N \le 5\,000$ segments) this is still well
  inside the 32-GB / 12-core workstation budget.
- The mutual-inductance matrix is dense and symmetric; the same
  ACA/iterative roadmap that applies to the electric Z-matrix
  applies here. Both will be tackled together.

### Neutral

- Default behaviour (`inductance_model is None`) is unchanged.
  Existing notebooks and tests continue to produce DC results at
  identical cost.
- The earth is still treated as a perfect mirror for the magnetic
  field — Carson finite-conductivity corrections are the subject
  of ADR-0005.

## References

- Grover, F. W. (1946). *Inductance Calculations: Working Formulas
  and Tables*. Dover (reprint 2004).
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover, ch. 7.
- Paul, C. R. (2010). *Inductance: Loop and Partial*. Wiley.
- Carson, J. R. (1926). Wave propagation in overhead wires with
  ground return. *Bell Syst. Tech. J.* 5(4) — referenced for the
  follow-up ADR-0005.

---

## Amendment 2026.11 — image conventions, exact touching pairs, lumped conductors, skin effect (review pass 10)

Findings 10.20, 10.24, 10.37, 10.44, 10.46 and 10.60. The physical model
of this ADR (Neumann partial inductances, complex branch block) stands;
the following decisions refine it.

1. **The image is part of the earth model.** The perfect-mirror image of
   section *Earth return ("ideal earth")* is one of three image
   conventions of `build_inductance_matrix(image=...)`
   (`ImageConvention`: `"none"`, `"additive"`, `"antiparallel"`). The
   solver builds the inductance matrix with the image of the selected
   `earth_inductive_model`: `perfect_mirror` → additive (this ADR),
   `carson_series` → anti-parallel (ADR-0005), `sommerfeld` → none
   (ADR-0006). An earth-return correction is defined against one image
   and must not be added to another (10.20). `perfect_mirror` has not
   been the default since 0.13.0.
2. **Pair integrals.** Parallel, skew and perpendicular segment pairs use
   closed forms; pairs that share an end point (chain neighbours, bends)
   use Grover's vertex formula and the coaxial gap formula by default
   (`touching="exact"`), so a straight conductor keeps the inductance of
   the unsplit wire for every discretisation (10.46).
   `touching="quadrature"` reproduces 0.15.0.
3. **Internal inductance.** Every self term carries the DC internal
   inductance $\mu_0\ell/(8\pi)$ of a solid round conductor, for
   distributed and lumped branches alike (`include_internal=True`),
   except a cable shield (`conductor_type="cable_shield"`), a
   thin-walled tube without one (item 11, user decision Q33).
4. **Lumped and single-segment conductors.** A conductor with an
   `inductance_model` that the topology keeps as one branch is one
   inductive branch from start to end, coupled to all others; up to
   0.15.0 its inductance was dropped silently (10.24).
5. **Skin effect (opt-in).** `Conductor.resistance_model="skin"` adds the
   exact internal impedance of a solid round conductor relative to the DC
   convention (resistance $R_\text{ac}(f)$, internal inductance
   $L_\text{int}(f)$); the default `"dc"` keeps the numbers (10.37). The
   bare-conductor restriction of section *Self-inductance and shielded
   conductors* stands.
6. **Frequency loop.** The earth-return correction is assembled once per
   solve for the whole frequency set (bit-identical to single-frequency
   assemblies, bounded cache; 10.60); the real fast path of purely
   galvanic worlds is unchanged.
7. **Close wires.** Coincident conductors have no finite mutual
   inductance; `find_close_wires` and `diagnose()` report crossing,
   touching, interpenetrating and close wires (10.44).
8. **Wires closer than the sum of their radii (second adversarial audit).**
   Two near-parallel segments whose axes stay closer than $r_1 + r_2$ over
   a common length (coincident leads, a service drop along its PEN main)
   are evaluated as touching wires at the axis distance $r_1 + r_2$, in
   every image convention; before, the quadrature stopped at its 1e-6 m
   floor (two coincident 50 m leads: 1.87 H). Close pairs that miss the
   closed forms — nearly parallel, crossing, skew pairs and T-junctions —
   are integrated semi-analytically (inner integral in closed form, graded
   outer panels), with `touching="exact"`; `touching="quadrature"`
   keeps the 0.15.0 rule. The coaxial closed forms are evaluated in one
   vectorised call per segment (audit A2C-5). The "5e-11" stated here
   holds for directions that differ by more than 1e-6 rad; see item 9 for
   the measured worst cases.
9. **Wires beside each other, tiny angles (third adversarial audit,
   A3C-1, A3C-2, A3C-4, A3C-5).** The touching rule of item 8 switched a
   whole pair, and the decision depended on which segment was the row:
   a V pair of 1 m and 0.5 m at 1.5° read 0.362 or 0.503 µH, the
   generator's lumped in-line leads 0.134 or 0.127 Ω of measured
   reactance (50 Hz); tilted V and X pairs whose far ends lay just
   outside the cylinder kept a filament value above the self inductance
   (an indefinite matrix, a two-wire loop with a negative reactance). The
   rule now acts pointwise inside the Neumann integral: where a point of
   one wire lies beside the other wire (its foot on the other axis falls
   inside that wire) closer than $r_1 + r_2$, that distance is used —
   touching wires — and where the thinner wire runs inside the thicker
   one ($\rho + r_\text{thin} \le r_\text{thick}$, a core inside a
   cable shield) the thicker radius: a filament inside a thin tube
   couples to it as at the tube radius, which gives the coaxial loop
   inductance $\frac{\mu_0}{2\pi}(\ln\frac{r_s}{r_c} + \frac14)$ per
   metre when the shield carries no internal inductance (+36 % before;
   since user decision Q32 the reading needs the shield to be marked,
   item 11). Both one-sided integrals (inner integral in
   closed form) are averaged, on end points in a fixed order, so every
   order and direction of the segments gives the same value bit for bit;
   the value is continuous in the geometry (4e-15 against an
   independent quadrature), chain neighbours and bends turning by less
   than 90° keep their closed forms, and 2000 random close
   configurations gave no indefinite matrix (700 before). Equal radii
   never count as one wire inside the other; for unequal radii the model
   steps from "inside" to "side by side" where the thinner axis leaves
   the thicker wire (the two describe different cables). Filaments from
   one vertex below 0.81° use a cancellation-free form of Grover's vertex
   formula (16 % off at 1.5e-6 rad before), and the close-pair rule takes
   the crossing point of nearly parallel axes from a cross product
   (9.5e-4 at 1e-8 rad before). Measured against mpmath: closed forms to
   round-off, vertex pairs to 4e-10 in a general orientation, close pairs
   5e-11 above 1e-6 rad and 1.0e-9 at 1e-8 rad; pairs inside the parallel
   tolerance use the parallel closed form at the start offset (4.2e-9 for
   wires 1 cm apart tilted by 1e-10 rad, 1.7e-5 at 1 µm).
10. **Pairs across the surface (user decision Q27).** Section *Earth
    return ("ideal earth")* applies the image to every pair, also to one
    segment above the soil surface ($z < 0$, an overhead lead) and one
    below it (a buried PEN), for which a magnetic mirror between them has
    no physical basis. The numbers stay — `perfect_mirror` is a
    diagnostic baseline — but the solve emits one `UserWarning` per
    inductive assembly with the number of such pairs and the conductors
    on each side, pointing to `sommerfeld`; `carson_series` and
    `sommerfeld` couple such pairs through the field transmitted into the
    soil and do not warn. For two parallel 1 km wires, one 1 m above the
    surface and one 0.7 m deep and 1 m beside it, in 100 Ω·m, the
    additive image gives 2.28 times (50 Hz) and 2.77 times (1 kHz) the
    mutual reactance of `sommerfeld`; `carson_series` gives 1.12 and 1.03
    times.
11. **Concentric cables by an explicit marker (user decisions Q32 and
    Q33).** Item 9 read every thinner wire on the axis of a thicker one
    as a core inside a cable shield, so two separate cables of different
    radii drawn on one axis were one concentric cable. The reading now
    needs the thicker conductor to carry `conductor_type="cable_shield"`
    (the existing type tag, which selected only a plot colour before);
    two conductors of any other types on one axis are separate
    conductors and touch, as every pair lying side by side
    (`build_inductance_matrix(..., shields=...)` marks the segments at
    the low level). A shield is a thin-walled tube: its self terms carry
    no internal inductance of a solid wire, and
    `Conductor.internal_impedance_per_length` of a `"dc"` shield has no
    inductive part; the solid-round skin model (`resistance_model="skin"`)
    is rejected for a shield. A 100 m coaxial pair (core 5 mm, shield
    15 mm, 0.7 m deep) reads a loop inductance of 26.966 µH in a solve
    (`perfect_mirror` and `sommerfeld`; 26.970 µH with `carson_series`),
    whatever the sub-pieces of the two conductors (item 12), against the
    analytic 26.972 µH of the infinitely long coax, instead of 31.966 µH
    with the internal inductance of a solid shield; unmarked, the two
    cables read 43.472 µH (43.476 µH with `carson_series`), the 43.48 µH
    of two touching solid wires. A shield without a core loses $\mu_0\ell/(8\pi)$ = 50 nH/m
    of self inductance against the same conductor without the tag
    (`buried_lead()` of the measurement generator builds its lead as a
    shield: the two-pole reading of a three-house network with buried feed
    and probe leads −2.5 % in reactance at 50 Hz and 1 kHz, the
    three-pole reading unchanged to 1e-6).
12. **Pieces of conductors lying beside each other (wave-1 audit of the
    calendar release, W1A-02).** Item 9 decides per *segment* whether a
    point lies beside the partner, so a pair of sub-pieces of two
    conductors on one axis that do not overlap coupled as two filaments on
    the axis (distance 0) while the overlapping pairs coupled at the shield
    radius or at $r_1 + r_2$: the pieces did not add up to the lumped pair.
    The 100 m coax read 26.912 / 26.852 / 26.374 / 24.608 µH in 10 / 5 /
    1 / 0.25 m pieces (−0.2 … −8.7 %; every joint of the pieces cost
    about 6 nH), the unmarked pair 43.400 … 40.343 µH, the drops routed
    back along the PEN main of a golden case 3.9e-4 low in reactance at
    1 kHz. Two rules now act on parallel pairs closer than touching. A
    thinner wire inside a cable shield ($\delta + r_\text{thin} \le r_s$,
    $\delta$ the lateral distance of the axes) couples at the shield
    radius at every axial offset — overlapping, end to end and apart —
    which is exact for a filament on the axis of a thin tube (every point
    of the tube is $\sqrt{x^2 + r_s^2}$ away), so a shield made of several
    conductors, or a core and shield in sections, add up too. For two
    other conductors lying beside each other (the rule of item 9 acts on a
    parallel pair of their pieces), a point lies beside the partner when
    its foot falls inside the partner's whole *conductor*, not only inside
    the partner segment; the stretches have Grover's parallel-filament
    closed form. The solver passes one label per conductor
    (`build_inductance_matrix(..., conductors=...)`; without labels every
    segment counts as a conductor of its own, the rule of item 9 in closed
    form). The pieces now add up to the lumped loop to 3e-12 (coax, marked
    and unmarked, 0.25 … 20 m pieces, all three earth-return models) and
    the drops along the main to 3e-15; the value stays continuous in the
    geometry, the same for every order and direction of the segments, and
    pieces of one conductor or conductors that only meet end to end on one
    axis (a main split at its taps) keep the filament integral (every
    other golden case bit-identical). The assembly of a 1 km coax in 2 m
    pieces takes 6.4 s instead of 5.8 s. Remaining: two unmarked cables in
    sections on one axis (a section of one beside a section of the other
    that it does not overlap) couple as filaments on the axis between the
    sections that do not overlap.
13. **Nearly parallel pieces (second wave-1 audit of the calendar
    release, W1R2-01, W1R2-03).** The closed form of item 12 needs the
    pieces parallel within $|\cos\varphi| \ge 1 - 10^{-9}$, about
    4.5e-5 rad. A core tilted by more against its shield kept the rule of
    item 9 per segment, so the drift of item 12 came back in full, with a
    step at the tolerance: the 100 m coax read 26.966 µH in 0.25 m pieces
    at 4.4e-5 rad and 24.966 µH at 4.5e-5 rad (−7.4 %; −0.17 / −0.36 /
    −1.86 % in 10 / 5 / 1 m pieces). Pieces of two conductors that are
    *nearly parallel* — the lateral distance changes by at most the
    touching distance $r_1 + r_2$ along the longer piece, and
    $|\cos\varphi| \ge 0.99$; both bounds were dropped by item 14 — now
    get the two readings of item 12 against
    the partner's whole conductor with the stretches found exactly: each
    piece is split where the foot of its points passes an end of the
    partner's conductor and where the distance $\rho(s)$ from the partner's
    axis crosses $r_1 + r_2$ or $r_\text{thick} - r_\text{thin}$ (from
    the closest approach of the two lines). On the stretches inside the
    shield and beside the partner the effective distance is constant and
    the integral has a closed form in the foot coordinate
    ($du = \cos\varphi\,ds$); a side that is entirely free takes the
    filament value of the pair, and a side with free and clamped stretches
    (a piece where the core leaves the touching distance, or passes an end
    of the partner) is integrated by the quadrature of item 9 with the
    conductor extents. The pieces add up to the lumped pair for tilts
    0 … 1e-3 rad in 10 / 5 / 1 / 0.25 m pieces, in the matrix within
    6.4e-12 (marked) and 3.1e-10 (unmarked; the largest value is the
    parallel closed form just below its tolerance, at 4.4e-5 rad) and in
    the solver (`perfect_mirror` and `sommerfeld` within 3.2e-11 marked,
    3.1e-10 unmarked; under `carson_series` the Carson
    correction of one lumped pair, tiled at its mid-point distance, differs
    from the pieces by 1.7e-5 at 3e-4 rad and 1.3e-4 at 1e-3 rad,
    independent of the piece length — the correction's own tiling, not the
    rule). The literal proposal, the closed form at the mean lateral
    distance of each pair, missed by up to 1.35 % at 1.5e-4 … 7e-4 rad,
    where the core leaves the shield's tube inside a piece. Beyond the
    gate the rule of item 9 acted per segment. Where the core leaves the
    axis at the shield's start the touching stretch is then about one
    piece long, and the pieces differed from the lumped pair by 1.5e-7
    (10 m pieces, 2.01e-3 rad), 6.6e-7 (1 m, 2.01e-2 rad) and 1.4e-6
    (0.25 m, 8.01e-2 rad); in general the error was not small (item 14).
    The generator's overhead measurement
    leads are such a pair (the probe lead starts 10 mm beside the feed
    lead and ends on its axis, 2e-4 rad): their mutual in the default 5 m
    pieces was +1.0e-4 off the lumped one (1 m pieces +5.4e-4), now
    4e-16; the probe junction potential moves by 6.5e-5 … 1.1e-4, the
    feed potentials and every electrode result not at all. Without labels
    (`conductors=None`) unmarked touching pieces stay per segment, while
    the inside reading of a marked shield needs no labels, also tilted.
    The pairs of two bundled conductors are evaluated in chunks of 4096
    (W1R2-03; element-wise, so bit-identical): a 600 m pair in 1 m pieces
    peaked at 292 MiB with all pairs at once, now 30 MiB, the level
    before item 12.
14. **No length gate and no angular cap (third wave-1 audit of the
    calendar release, W1R3-01, W1R3-02).** The two bounds of item 13 mixed
    two readings within one pair of conductors, and between a lumped pair
    and its pieces. Where the core crosses the shield axis at mid-length
    the touching stretch is two pieces long: just beyond the length gate
    the loop in 10 / 5 / 1 / 0.25 m pieces read −8.6e-5 / −6.4e-5 /
    −4.0e-5 / −3.0e-5 off the lumped pair (solver: X −7.6e-5 / −5.6e-5 in
    10 / 5 m pieces under `perfect_mirror` and `sommerfeld`, homogeneous,
    two- and three-layer soil). Long pieces at small tilts read −1.7e-4
    (2 × 50 m, 5e-4 rad), shield pieces of 15 / 1 m at 2e-3 rad −1.6e-3,
    and a core running 5 m beyond its shield +2.9e-5, its lumped mutual
    4.4e-6 off the independent quadrature (the lumped loop stepped by
    −3.3e-5 at the gate). At the cap, two 2 m wires leaving one junction
    in 0.02 m pieces read 0 at 8.0° and −1.55e-3 at 8.2°. Decision of the
    coordinator: the rule acts on every pair of pieces of two conductors
    with a hit of the pointwise rule that is neither parallel within the
    closed form's tolerance nor orthogonal, at any angle and for pieces
    of any length; the classification of item 13 is exact at any angle.
    Every case above now adds up to the lumped pair within 2.2e-12, every
    lumped mutual equals the independent quadrature within 4.4e-16, the
    loop is smooth in the tilt (no step at the old gate, none at 8.1°;
    wires leaving a junction add up at 1° … 90° in 0.1 / 0.05 / 0.02 m
    pieces), and the matrices stay symmetric and positive definite. In the
    solver the pieces equal the lumped pair within 1.1e-15 under
    `perfect_mirror`; under `sommerfeld` within 1.6e-8 at 1 kHz (3.2e-7
    at 10 × the old gate in 5 m pieces), the correction's own piece
    dependence, the same inside the old gate; under `carson_series` the
    correction of a tilted pair is tiled at the mid-point distance of each
    pair and the pieces differ from the lumped pair by 4.2e-4 (10 m pieces,
    2e-3 rad), 1.4e-3 (5 m, 4e-3 rad), 1.9e-2 (10 m, 2e-2 rad) — the
    tiling, not this rule. Only pairs with a piece that comes within the
    touching distance of the other's axis are classified (the others keep
    the filament value either way; bit-identical): a 600 m coax crossing
    at 0.05 / 0.3 rad in 1 m pieces assembles in 3.95 / 3.82 s against
    3.73 / 3.66 s on the snapshot, where the gates skipped these pairs.
    Generated networks are bit-identical, with and without an inductive
    PEN.
