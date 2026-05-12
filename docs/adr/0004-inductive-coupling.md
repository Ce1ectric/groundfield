# ADR-0004: Inductive coupling between conductor segments

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-05-04 |
| **Deciders** | Christian Ehlert |
| **Scope** | `groundfield`, work package 1 of the dissertation |

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
**not** for the AP1 question on **inductive coupling between the
measurement lead and the current injection** — the very effect that
forced the distributed-conductor refactor in the first place.

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
AP1 cable runs ($\ell$ ≈ 1–5 m, $a$ ≈ 4 mm).

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
for shielded MV cables but not for the bare-copper / PEN AP1 study.
We keep the model simple: `inductance_model == "neumann"` always
uses the bare-conductor self formula; refinements come with the
cable-shield work later in the dissertation.

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

- Closes the AP1 question on inductive coupling between the
  measurement lead and the current injection. The model is now
  physically complete for $f < 1\,\mathrm{kHz}$ except for the
  Carson earth-return correction.
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
  real solve in the resistive-only case). For a typical AP1 sweep
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
