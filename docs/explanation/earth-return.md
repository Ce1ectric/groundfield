# Earth return and inductive coupling

Conductors in a grounding study — PEN conductors, cable screens,
measurement leads, interconnections between earth electrodes — are not
only resistors. At 50 Hz, and more so at the harmonics up to 1 kHz that
matter for AP2, their longitudinal impedance has an inductive part, they
couple magnetically to each other, and the current that returns through
the soil adds the **earth-return impedance**. This page explains how
`groundfield` assembles that impedance, which earth model to use, what
changed in 2026.11 and how the result is validated.

## Mathematical and physical context

A conductor with `inductance_model="neumann"` is split into branches (one
per sub-segment of `discretize_segment_length`, or one branch for a lumped
conductor). Their longitudinal impedance matrix enters the network
equations of the [engine family](engines/index.md#network-equations):

$$
Z_b(\omega) = R + j\omega L + \Delta Z(\omega) .
$$

- $R$ — the DC resistances of the branches; with
  `resistance_model="skin"` a frequency-dependent internal-impedance
  correction is added (see [skin effect](#skin-effect)).
- $L$ — the partial-inductance matrix: Neumann's double line integral
  $L_{ij} = \frac{\mu_0}{4\pi}\int_i\int_j \frac{\mathrm d\mathbf l_i\cdot\mathrm d\mathbf l_j}{r}$
  between every pair of branches, closed forms for parallel, skew,
  touching and coaxial pairs (Grover 1946), thin-wire self terms with the
  DC internal inductance $\mu_0\ell/(8\pi)$ of a solid round conductor —
  plus the **image** of the earth model (below).
- $\Delta Z$ — the earth-return correction: how much the finite
  conductivity of the soil changes the magnetic field of the image.

The physical scale is the skin depth of the soil,

$$
\delta = \sqrt{\frac{2\rho}{\omega\mu_0}} \approx 503\,\sqrt{\frac{\rho}{f}}\ \text{m},
$$

which lies between about 350 m (50 Ω·m, 1 kHz) and 5 km (500 Ω·m, 50 Hz) in
the frequency range of the package: comparable to or larger than the
extent of a low-voltage network. The return current therefore spreads deep
and wide, and neither a perfect mirror nor a perfectly conducting earth
describes it. `FieldResult.metadata["penetration_depth"]` reports $\delta$
for every frequency of a solve.

The soil-side galvanic coupling (leakage, potentials) is unaffected: the
field kernel stays real and quasi-static, the earth return acts on the
longitudinal impedance of conductors only.

## The three earth models

`Engine.earth_inductive_model` selects the model. Each is the combination
of an image convention for $L$ and a correction $\Delta Z$ defined against
that very image — an earth-return correction is meaningless without the
baseline it was derived for:

| `earth_inductive_model` | Image of $L$ | $\Delta Z$ | Use |
|---|---|---|---|
| `"sommerfeld"` (default since 0.13.0) | none (free space) | $\Delta G = G - 1/R$: the full Sommerfeld/Pollaczek kernel minus the free-space part, integrated over every segment pair | quantitative $Z(f)$, any geometry, layered earth |
| `"carson_series"` | anti-parallel (perfectly conducting earth) | Carson's per-metre correction $(\omega\mu_0/\pi)(P + jQ)$ times the pair length | long parallel lines over homogeneous earth; fast |
| `"perfect_mirror"` | additive (magnetic mirror) | none | diagnostic baseline only |

### Sommerfeld (Pollaczek kernel)

The vector potential of an alternating line current in or above a
conducting half-space is a Sommerfeld integral (Sommerfeld 1909;
Pollaczek 1926 for the buried conductor, Carson 1926 for the overhead
one). With the time convention $e^{+j\omega t}$, $\gamma^2 = j\omega\mu_0\sigma$
and $u = \sqrt{\lambda^2 + \gamma^2}$, the Green's function of two
points at depths $z, z'$ in homogeneous soil, horizontal distance $\rho$
and distance $R$, is

$$
G(\rho, z, z') = \frac{e^{-\gamma R}}{R}
+ \int_0^\infty \frac{\lambda}{u}\,\frac{u - \lambda}{u + \lambda}\,
e^{-u(z+z')}\,J_0(\lambda\rho)\,\mathrm d\lambda ,
$$

and the earth-return correction of two segments is
$\Delta Z_{ij} = j\omega\frac{\mu_0}{4\pi}\int_i\int_j (G - 1/R)\,
\mathrm d\mathbf l_i\cdot\mathrm d\mathbf l_j$, the change against free
space. For layered earth the spectral amplitude carries the generalised
reflection coefficients of the stack, overhead pairs keep Carson's
air-side geometry; the amplitudes are documented in
`groundfield.coupling.sommerfeld_inductance`. `groundfield` integrates
$\Delta G$ over the actual geometry of each segment pair — any length,
orientation and depth — and adds it to an image-free Neumann matrix. In
the long-wire limit it gives Pollaczek's per-metre impedance, including
the positive earth-return resistance of about $\omega\mu_0/8$ per metre;
for $\sigma \to 0$ it vanishes (the soil is magnetically transparent).

What 2026.11 changed (ADR-0006 amendment):

- **Image-free baseline** (findings 10.46, F52). The correction is built
  against free space, so no image has to be added and removed again; the
  numbers equal the former additive pairing to round-off. Conductors that
  cross the soil surface or lie in it are now solved (the pair integrals
  are split at $z = 0$); with the image conventions of the other models
  such a segment is singular and raises.
- **Exact layered kernel** (10.35). For any number of layers the
  same-layer amplitude uses generalised reflection coefficients and all
  other pairs a batched interface solve; a three-layer earth whose lower
  two layers are equal now equals the two-layer earth exactly (it was
  +17…+25 % in $X'$ before).
- **Resolved $\lambda$ grid** (10.36). The logarithmic region ends before
  $J_0$ oscillates, the linear region carries 16 nodes per $J_0$ period
  up to four times the largest physical scale, and an exhausted panel
  budget warns (`SommerfeldResolutionWarning`) instead of truncating
  silently; distant pair entries were off by −24 % (200 m) and +274 %
  (400 m) before.
- **Multi-frequency assembly** (10.60). Geometry, $\lambda$ grid and
  Bessel tables are built once per pair for the whole frequency set of a
  solve, bit-identical to single-frequency assemblies, with a bounded
  cache.
- **Tabulated assembly** (opt-in, `Engine.sommerfeld_assembly="tabulated"`).
  Sub-segments are grouped by depth and one table of the spectral integral
  per pair of depth groups and frequency replaces the per-pair
  quadrature. It is much faster on large networks and agrees with a
  converged reference better than the default per-pair rule, whose
  24-node interpolation in the horizontal distance dominates its error;
  the default stays `"exact"` in this release. Measured times are on the
  [performance page](performance.md).

### Carson on its own baseline

Carson (1926) derived the earth-return impedance of a line over
homogeneous earth as a correction to the impedance of the line and its
**anti-parallel** image in a perfectly conducting plane, with the
dimensionless parameter $a = D\sqrt{\omega\mu_0\sigma}$ ($D$ the distance
between conductor and image, $2h$ for a self term). `carson_series`
multiplies the per-metre correction by the geometric length of each
segment pair.

Up to 0.15.0 the solver added Carson's correction to the **additive**
image matrix of the perfect mirror, which roughly doubled the
earth-return reactance (finding 10.20 / F32): 1.66–2.25 times Carson's
value for 100 m – 1 km lines, +41 % on $|Z|$ at 1 kHz for a TN network
with inductive PEN conductors. Since 2026.11 the matrix is built on the
anti-parallel image; a 1 km isolated PEN now reproduces
$l\,X'_\text{Carson}$ within 1 %
(`tests/test_inductive_solver_integration.py::test_carson_series_reactance_matches_carson`).
Carson's small-$a$ series is evaluated to $a^{12}$ and the self-term
quadrature at $\theta = 0$ to round-off, so the regime boundaries are
continuous; the asymptotic expansion is used for $a > 15$.

`carson_series` remains an approximation for short, inclined or buried
conductors and for layered soil (an effective homogeneous conductivity
is used); it is the fast choice for long parallel lines over homogeneous
earth, not a replacement of `sommerfeld` in general.

### Perfect mirror

`perfect_mirror` treats the earth as a perfect magnetic mirror (additive
image, no correction). It is cheap and frequency independent but far from
the physics of buried conductors below 1 kHz; it was the default before
0.13.0 and is kept as a baseline for comparisons.

## Conductor details

- **Touching segments** (10.46). Chain neighbours and bends share an end
  point where the Neumann integrand is logarithmically singular; Grover's
  vertex formula and the coaxial gap formula replace the 16 × 16
  Gauss–Legendre rule by default (`touching="exact"`), so a straight
  conductor keeps the inductance of the unsplit wire for every
  discretisation
  (`tests/test_inductive_solver_integration.py::test_inductance_is_refinement_invariant_for_a_straight_conductor`).
  `touching="quadrature"` reproduces 0.15.0.
- **Lumped and single-segment conductors** (10.24). A conductor with an
  `inductance_model` that the topology keeps as one branch is one
  inductive branch from start to end, coupled to all others; up to 0.15.0
  its inductance was silently dropped (a 300 m PEN read 0.0564 Ω at 1 kHz
  instead of 0.210 + 3.78j Ω).
- **Close wires** (10.44). `conductors.find_close_wires` and `diagnose()`
  report crossing, touching, interpenetrating and close wires; two
  identical coincident conductors have no finite mutual inductance, which
  is why the measurement generator lays in-line leads touching, not
  coincident.

### Skin effect

With `Conductor.resistance_model="skin"` (opt-in, finding 10.37) every
branch of the conductor adds $\ell_b\,\Delta Z'_\text{skin}(f)$ to its
diagonal: the exact internal impedance of a solid round conductor,

$$
Z'_\text{int} = \frac{k\rho}{2\pi a}\,\frac{I_0(ka)}{I_1(ka)},\qquad
k = \sqrt{j\omega\mu_0/\rho},
$$

relative to the DC convention ($R'_\text{dc}$ plus the internal inductance
$\mu_0/(8\pi)$ already in $L$). For 150 mm² aluminium the resistance rises
by the factors 1.056, 1.231 and 1.550 at 250, 550 and 1000 Hz and the
internal inductance falls to 0.740 of its DC value at 1 kHz
(`tests/test_skin_effect.py`); the IEC 60287-1-1 skin factor agrees to
about 1 %. A conductor without an `inductance_model` has no inductance
at all, so only the resistance part applies to it. The default
`resistance_model="dc"` keeps the numbers and the JSON of earlier
releases.

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
  and the $K_0$ head since 2026.11 (finding 10.45) and agrees with an
  independent boundary-value solution to $10^{-7}$
  (`tests/test_pollaczek_layered.py`).
- **Engine level:** `compare_engines` compares complex impedances at
  every frequency, so a reactance mismatch between earth models can no
  longer pass as consistent (finding 10.71; benchmark B13 row
  `image_vs_mom`).

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
