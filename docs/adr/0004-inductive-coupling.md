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
   distributed and lumped branches alike (`include_internal=True`).
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
   a shield conductor in a solve still carries the internal inductance of
   a solid wire, item 3). Both one-sided integrals (inner integral in
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
