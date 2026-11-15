# ADR-0005: Carson correction for the earth-return path

| | |
|---|---|
| **Status** | Accepted — amended 2026.11 (review pass 10: Carson on its anti-parallel baseline, F32); see the amendment at the end of this document |
| **Date** | 2026-05-06 |
| **Deciders** | Project maintainers |
| **Scope** | `groundfield` |

## Context

ADR-0004 added the Neumann self- and mutual-inductance integrals
between distributed-conductor segments and assembled them into a
per-frequency branch-impedance block

$$
Z_b(\omega) \;=\; R \;+\; j\omega\, L_\text{Neumann}.
$$

For the magnetic image the earth was treated as a **perfect mirror**
($\sigma_\text{earth}\to\infty$). That is the same simplification
Sunde 1968, ch. 7.4 uses for buried filaments and is known to be
accurate when the relevant geometric dimension is small compared to
the electromagnetic skin depth in soil

$$
\delta(\omega) \;=\; \sqrt{\frac{2}{\omega\,\mu_0\,\sigma_\text{earth}}}
\;\approx\; 503\,\sqrt{\rho_\text{earth} \big/ f}\quad[\text{m}].
$$

For the quasi-static frequency window ($f \le 1\,\mathrm{kHz}$) and typical
soil resistivities ($\rho_\text{earth} \in [50, 5000]\,\Omega\,\mathrm{m}$)
the skin depth $\delta$ ranges from roughly **350 m** (50 Hz, 50 Ωm)
to **35 km** (50 Hz, 5000 Ωm). The relevant geometric dimensions of
an the TN low-voltage distribution network — house-to-station distance < 200 m, PEN-cable
length up to 1 km, separation between measurement-lead and current
injection 10–100 m — are *not always* small compared to $\delta$.
The perfect-mirror model therefore systematically **underestimates**
the resistive part of the earth-return impedance and slightly
**overestimates** the reactive part. The discrepancy is the very
quantity that the research question

> *"diffusion field and Carson relevance for earth currents"*

asks us to put a number on.

This ADR adds the Carson 1926 correction to close that gap. The
result is a frequency-dependent, complex-valued addition to the
branch-impedance block,

$$
Z_b(\omega) \;=\; \underbrace{R}_{\text{ohmic}}
   \;+\;\underbrace{j\omega\, L_\text{Neumann}^{\text{(perfect mirror)}}}_{\text{ADR-0004}}
   \;+\;\underbrace{\Delta Z_\text{Carson}(\omega, \sigma_\text{earth}, h_i, h_j, d_{ij})}_{\text{this ADR}},
$$

evaluated per frequency on the same per-branch endpoint geometry
that ADR-0004 already exposes.

## Decision

### Physical model

The starting point is John R. Carson, *"Wave propagation in
overhead wires with ground return"*, Bell Syst. Tech. J. 5(4)
(1926), pp. 539–554. Carson solves Maxwell's equations for a
straight horizontal wire above a homogeneous semi-infinite
conductive half-space and writes the ground-return self-impedance
correction as

$$
Z'_\text{Carson} \;=\; 4\omega\,J(2h\sqrt{\alpha},\,0),
\qquad
\alpha \;=\; 4\pi\lambda\omega \;\;[\text{CGS-emu}],
$$

with

$$
J(p, q) \;=\; \int_0^{\infty}\!\!\bigl(\sqrt{\mu^2 + j} \;-\; \mu\bigr)\,
e^{-p\mu}\cos(q\mu)\;d\mu
\quad\text{(Carson eq. 29).}
$$

The mutual-impedance correction between two parallel wires at
heights $h_1, h_2$ with horizontal separation $x$ is

$$
Z'_{12,\text{Carson}} \;=\; 4\omega\,J\bigl((h_1+h_2)\sqrt{\alpha},\,x\sqrt{\alpha}\bigr).
$$

In **SI units** this becomes (Carson eq. 30/31, converted by
$4\omega \to \omega\mu_0/\pi$ and $\alpha = 4\pi\lambda\omega \to
\omega\mu_0\sigma_\text{earth}$):

$$
\boxed{\;
\Delta Z_\text{Carson}(\omega) \;=\; \frac{\omega\mu_0}{\pi}\,
\bigl[P(a,\theta) \;+\; jQ(a,\theta)\bigr],
\;}
$$

with the dimensionless Carson parameters

$$
a \;=\; D\sqrt{\omega\mu_0\sigma_\text{earth}} \;=\; \frac{D\sqrt{2}}{\delta(\omega)},
\qquad
D \;=\; \begin{cases}
2h_i & \text{(self impedance, $\theta = 0$)} \\
\sqrt{(h_i+h_j)^2 + d_{ij}^2} & \text{(mutual impedance)}
\end{cases},
\qquad
\theta \;=\; \arctan\!\bigl(d_{ij}\big/(h_i+h_j)\bigr).
$$

The functions $P(a,\theta)$, $Q(a,\theta)$ are the real and
imaginary parts of the Carson integral $J$. Their evaluation is
split into three regimes following Carson's own discussion in
section III of the original paper.

### Three evaluation regimes

The functions $P, Q$ are evaluated by selecting one of three
expansions according to the magnitude of $a$:

#### Regime 1 — small $a$ ($a \le 0.25$, leading-term form)

Carson eqs. 34–35:

$$
P_\text{small}(a,\theta) \;=\; \frac{\pi}{8}
\;-\; \frac{a}{3\sqrt{2}}\cos\theta
\;+\; \frac{a^2}{16}\cos(2\theta)\,
   \bigl(0.6728 + \ln(2/a)\bigr)
\;+\; \frac{a^2}{16}\theta\,\sin(2\theta),
$$

$$
Q_\text{small}(a,\theta) \;=\; -0.0386
\;+\; \tfrac{1}{2}\,\ln(2/a)
\;+\; \frac{a}{3\sqrt{2}}\cos\theta.
$$

#### Regime 2 — intermediate $a$ ($0.25 < a \le 5$, direct quadrature)

*(Rewritten after the 2026-07-08 audit, WP-B2. The originally
planned $\sigma_i$-series implementation was never built; the code
evaluates Carson's integral $J(p, q)$ numerically.)*

Two quadrature paths, selected by the oscillation parameter
$q = a\sin\theta$:

- **$\theta \approx 0$ (self-correction geometry):** 64-node
  Gauss–Legendre on $[0, 30/p]$ — the integrand decays purely
  exponentially and the historic evaluation is machine-accurate
  (verified against adaptive references).
- **$q > 0$ (mutual geometries):** QUADPACK's Fourier integrator
  (``scipy.integrate.quad`` with ``weight='cos'``) on
  $[0, \infty)$. This replaced the historic fixed-node
  Gauss–Legendre on $[0, 30/p]$, which failed for
  $\theta \to \pi/2$ (small $p$, unresolved $\cos(q\mu)$
  oscillation over a huge interval): the audit measured errors up
  to $-36\,\%$ in $P$ and $+67\,\%$ in $Q$ at buried-pair mutual
  geometries ($a = 0.44$–$1.6$, $\theta \approx 89°$). The Fourier
  path also covers $p = 0$ ($\theta = \pi/2$) exactly, where the
  imaginary part converges only conditionally through the
  oscillation. Verified to $10^{-11}$ against an independent
  composite per-half-period Gauss rule with Euler acceleration
  (``tests/test_audit_quadratures.py``).

#### Regime 3 — large $a$ ($a > 5$, asymptotic expansion)

Carson eqs. 36–37:

$$
P_\text{large}(a,\theta) \;=\; \frac{1}{\sqrt{2}}\,\frac{\cos\theta}{a}
\;-\; \frac{\cos(2\theta)}{a^2}
\;+\; \frac{1}{\sqrt{2}}\,\frac{\cos(3\theta)}{a^3}
\;+\; \frac{3}{\sqrt{2}}\,\frac{\cos(5\theta)}{a^5} \;-\; \dots,
$$

$$
Q_\text{large}(a,\theta) \;=\; \frac{1}{\sqrt{2}}\,\frac{\cos\theta}{a}
\;-\; \frac{\cos(3\theta)}{a^3}
\;+\; \frac{3}{\sqrt{2}}\,\frac{\cos(5\theta)}{a^5} \;-\; \dots
$$

The truncation matches Carson's published curves (Fig. 2/3 of the
original paper) within plotting accuracy for $a \ge 5$; the
regime-2/regime-3 boundary at $a = 5$ carries the asymptote's
truncation error ($\lesssim 5\cdot10^{-4}$ absolute in $P, Q$ at
$\theta = 80°$, smaller for steeper geometries). Two audit
corrections (2026-07-08, WP-B2): the $a^{-3}$ term of $Q$ carries
the factor $1/\sqrt{2}$ (previously missing — $\sim 6\cdot10^{-4}$
absolute at $a = 8$), and for $\theta > 1.4$ rad ($\approx 80°$)
the expansion degrades and the dispatcher routes through the
regime-2 quadrature instead.

### Linear-system integration

The Carson correction enters the existing per-branch impedance
block as **one extra complex contribution** per branch pair
$(i, j)$:

$$
Z_b^{(i,j)}(\omega) \;=\; \delta_{ij}\,R^{(i)}
   \;+\; j\omega\, L_\text{Neumann}^{(i,j)}
   \;+\; \Delta Z_\text{Carson}^{(i,j)}(\omega).
$$

The matrix $\Delta Z_\text{Carson}$ is dense and symmetric (the
Carson kernel is reciprocal). Assembly cost is the same as the
Neumann matrix — one closed-form evaluation per $M(M-1)/2$ pairs —
but it must be **rebuilt at every frequency**, because the kernel
itself depends on $\omega$ through $a$. The Neumann part stays
frequency-independent and is built once, exactly as in ADR-0004.

The branch block consumed by `_solve_cluster_currents` is therefore
formed at run time as

```
Z_b(omega) = R_diag + 1j * omega * L_neumann + dZ_carson(omega)
```

with `R_diag`, `L_neumann` precomputed and `dZ_carson` recomputed
per frequency. The complex linear system

$$
\begin{pmatrix}
Z & -C & 0 \\
C^\top & 0 & B^\top \\
0 & B & -Z_b(\omega)
\end{pmatrix}
\begin{pmatrix} I_e \\ \varphi_n \\ I_b \end{pmatrix}
\;=\;
\begin{pmatrix} 0 \\ I_\text{in} \\ 0 \end{pmatrix}
$$

is solved per frequency, exactly as in ADR-0004.

### Earth-conductivity source

The earth conductivity $\sigma_\text{earth}$ is taken from the
soil model:

| Soil model | Source for $\sigma_\text{earth}$ |
|---|---|
| `HomogeneousSoil` | $\sigma = 1/\rho$ — exact |
| `TwoLayerSoil` | $\sigma = 1/\rho_1$ (upper layer) **with a runtime warning**; see "Layered earth" below |
| `MultiLayerSoil` | $\sigma = 1/\rho_1$ with a warning, identical strategy |

For the homogeneous case the Carson series is **exact** (within the
quasi-static / low-frequency assumption that Carson himself states:
$\omega \to 0$ and displacement currents in the dielectric
neglected). For layered soils the pure Carson series is
approximate; we keep it as a default for usability and document the
escape hatch (`mom_sommerfeld`) explicitly.

### Engine-side switch

A new field is added to `Engine`:

<!-- skip-doctest: schema sketch with a ``...`` body -->
```python
class Engine(BaseModel):
    ...
    earth_inductive_model: Literal["perfect_mirror", "carson_series"] = "perfect_mirror"
```

`Conductor.inductance_model = "neumann"` continues to mean *use
the Neumann self/mutual integrals* — no change. Whether the
**earth** behaves like a perfect mirror or like a finite-σ Carson
half-space is now a property of the **engine** rather than of each
conductor, because every distributed conductor in the same world
sits above the same earth. Default is `"perfect_mirror"` for
backward compatibility (every existing test and notebook keeps its
exact result).

`Conductor.inductance_model = None` is unchanged — the system stays
real and the DC fast path is preserved bit-exact, regardless of
`earth_inductive_model`.

### What about the electric Green's function?

The Carson correction in this ADR addresses the **magnetic** image
only — i.e. the mutual-impedance contribution to the longitudinal
branch block. The **electric** image (charge image used by the
existing image / image_2layer / mom kernels for the leakage
current to soil) is *not* affected. That part already uses the
appropriate static Green's function for the soil model
(homogeneous, two-layer, n-layer). The two pieces are physically
decoupled in the quasi-static regime: the conductive current in
the wires sees the earth via the magnetic Green's function (Carson
territory), the leakage current sees it via the electric Green's
function (already implemented). For full electromagnetic coupling
one would have to upgrade `mom_sommerfeld` to the Pollaczek
kernel; that is left as a follow-up — see ADR-0006 (deferred).

## Alternatives considered

### Deri/Semlyen complex depth

The Deri & Semlyen 1981 approximation replaces Carson's integral
with the closed-form expression

$$
\Delta Z_\text{Deri-Semlyen}(\omega) \;=\;
\frac{j\omega\mu_0}{2\pi}\,
\ln\!\Bigl(\frac{D' + p}{D}\Bigr),
\qquad
p \;=\; 1\Big/\sqrt{j\omega\mu_0\sigma_\text{earth}},
$$

where the image is moved by a **complex depth** $p$ below the soil
surface. It is appealingly cheap (no series), but its accuracy
degrades for short wires and for $a \in [0.5, 5]$. We keep it as
an internal sanity check in `coupling/carson.py`
(`deri_semlyen_correction`) and use it in tests as a second
opinion against the series implementation; we do **not** use it as
the production code path.

### Direct numerical Sommerfeld quadrature

Pollaczek 1926 generalises Carson to the layered half-space; the
resulting integral has to be evaluated numerically. This is
already implemented for the **electric** Green's function in
`solver/mom_sommerfeld.py`. The **magnetic** Pollaczek kernel
would slot into the same backend and would be the rigorous answer
for the layered case. Cost: every frequency requires a 2-D
Sommerfeld quadrature *per segment pair* — orders of magnitude
slower than the closed-form Carson series. We defer this to a
follow-up ADR (ADR-0006) and use `mom_sommerfeld` (electric only)
plus Carson series (magnetic) as the typical working configuration.

### Layered earth via effective resistivity

A pragmatic middle ground: when the soil is layered we replace
$\rho_\text{earth}$ in the Carson series by an **effective**
resistivity $\rho_\text{eff}(\omega)$ that interpolates between
$\rho_1$ (when $\delta(\omega) \ll h_1$, the first-layer regime)
and $\rho_2$ (when $\delta(\omega) \gg h_1$, the deep regime).
We keep this as a **future option** and currently emit a runtime
warning instead. The reason: the depth-blending is itself an
approximation whose validity range is the same as the Pollaczek
result we are trying to avoid. Better to be explicit about the
approximation than to hide it inside an interpolation rule.

## Validation

Every test below is implemented in `tests/test_carson_coupling.py`
unless noted otherwise.

### Unit tests against Carson 1926

1. **Wave-antenna example (Carson p. 552)**
   $h = 30\,\mathrm{ft}$, $f = 5 \cdot 10^4\,\mathrm{Hz}$,
   $\lambda = 10^{-12}$ emu → $a = 4.0$ → $J = 0.126 + j0.168$.
   Tolerance: $\le 0.5\,\%$ relative.
2. **Wave-antenna example, second case**
   Same geometry, $\lambda = 10^{-14}$ emu → $a = 0.4$ →
   $J = 0.323 + j0.871$. Tolerance: $\le 0.5\,\%$.
3. **Railway example (Carson p. 553)**
   $f = 25\,\mathrm{Hz}$, $h = 30\,\mathrm{ft}$,
   $x = 120\,\mathrm{ft}$, $\lambda = 10^{-12}$ emu →
   $J = 0.369 + j1.135$, $\theta \approx 63°30'$. Tolerance:
   $\le 0.5\,\%$.
4. **Tabulated curves Fig. 2/3**
   Spot checks at $a \in \{0.5, 1, 2, 4\}$ and
   $\theta \in \{0, \pi/4, \pi/2\}$. Tolerance: $\le 1\,\%$
   (Carson plotted on a small grid).

### Regime-boundary continuity

5. At $a = 0.25$ the small-$a$ form (regime 1) and the full series
   (regime 2) agree to $\le 10^{-9}$ in $P, Q$.
6. At $a = 5$ the full series (regime 2) and the asymptotic
   expansion (regime 3) agree to $\le 10^{-6}$ in $P, Q$.

### Limit checks

7. **$\sigma_\text{earth} \to \infty$** (perfect conductor) →
   $\Delta Z_\text{Carson} \to 0$. The full assembled branch
   block reproduces ADR-0004 bit-exact (regression test).
8. **$\omega \to 0$** → $\Delta Z_\text{Carson} \to 0$. The DC
   solution reproduces ADR-0003 bit-exact.
9. **Frequency monotonicity** — for fixed $\sigma_\text{earth}$
   and a 1 km PEN over earth the real part of the diagonal
   $Z_b(\omega)$ grows monotonically with $\omega$ in
   $[10\,\mathrm{Hz}, 1\,\mathrm{kHz}]$. Sanity check: ohmic
   earth-return contribution increases with frequency.

### Engineering benchmarks

10. **1 km PEN self impedance vs. Oeding + Carson lookup**.
    Build a horizontal PEN conductor at $h = 0.6\,\mathrm{m}$
    above a homogeneous earth ($\rho = 100\,\Omega\,\mathrm{m}$),
    1 km long, radius 4 mm, run all distributed-capable backends
    at $f \in \{50, 150, 500, 1000\}\,\mathrm{Hz}$.
    Verify $Z_\text{self,per km}$ within 5 % of the Oeding/Tleis
    closed-form Carson R/X tabulation. Implementation:
    `scripts/benchmarks/pen_1km_carson.py` plus a regression test.
11. **Loop coupling open-circuit voltage with vs. without Carson**.
    Two parallel galvanic conductors, 1 km long, separation 50 m.
    The injected current is 1 A on conductor 1, conductor 2 is
    open. The open-circuit voltage at $f = 50\,\mathrm{Hz}$ must
    increase by $\ge 5 \%$ (Carson on) compared to the
    perfect-mirror result. Test: `test_carson_coupling.py::
    test_loop_coupling_carson_increases_open_voltage`.
12. **Cross-engine consistency at 50 Hz with Carson active**.
    All distributed-capable backends (`image`, `image_2layer`,
    `mom`, `cim`, `bem`) must agree on the cluster impedance to
    within 5 %. `fem` continues to log a warning and fall back to
    the resistive solution.

### Layered-earth handling

13. **Two-layer warning**. Building an `Engine` with
    `earth_inductive_model = "carson_series"` against a
    `TwoLayerSoil` world emits a `UserWarning` containing the
    string `"Carson series uses upper-layer rho_1"`.
14. **Carson(approx layered) vs. mom_sommerfeld**. For a TwoLayerSoil
    with $\rho_1 = 100\,\Omega\,\mathrm{m}$, $\rho_2 = 1000\,\Omega\,\mathrm{m}$,
    $h_1 = 1\,\mathrm{m}$, the Carson approximation differs from
    the Pollaczek/Sommerfeld reference by less than the
    documented bound (≤ 20 % at 50 Hz, ≤ 10 % at 1 kHz). The
    bound itself is a documented test artefact, not a guarantee
    — validated by the project's internal Carson harness.

### Cross-references

15. **DC reproducibility** — `inductance_model = None` keeps the
    system real for any setting of `earth_inductive_model`
    (regression).
16. **Bit-exact perfect-mirror regression** — the full ADR-0004
    test suite continues to pass with `earth_inductive_model =
    "perfect_mirror"` (default).

## Consequences

### Positive

- Quantifies the relevance of the Carson correction for earth
  currents: "at which frequency / soil resistivity does the
  perfect-mirror approximation break down" can now be answered
  numerically — validated by the project's internal Carson harness.
- The closed-form Carson series adds essentially zero cost
  ($\mathcal{O}(M^2)$ scalar evaluations per frequency, vectorised
  via `numpy`). Frequency sweeps with $N_f \le 20$ on
  relevant geometries ($M \le 5\,000$) remain inside the
  workstation budget.
- The implementation is **physically transparent**: the same
  series Carson published in 1926, with the same regime split.
  Tests against Carson's own worked examples are the gold
  standard.
- The earth conductivity now enters the inductance assembly
  explicitly, which makes the **penetration depth** $\delta$ a
  first-class diagnostic. The new
  `FieldResult.metadata["penetration_depth"]` exposes
  $\delta(\omega)$ per frequency for every engine.

### Negative

- For layered soil the series is an approximation. We document
  the limit and provide a quantified comparison against
  `mom_sommerfeld` in the validation notebook. A user who needs
  rigor on layered soil must switch backends.
- The branch block is rebuilt at every frequency. With Carson on
  and $N_f$ frequencies, $N$ segments, the assembly cost is
  $\mathcal{O}(N_f \cdot M^2)$ closed-form evaluations on top of
  the existing $\mathcal{O}(N_f \cdot N^3)$ LU. For typical
  ($M \le 5\,000$, $N_f \le 20$) the assembly is still
  $\le 10\,\mathrm{s}$ in the worst case; the LU dominates.

### Neutral

- Default behaviour (`earth_inductive_model = "perfect_mirror"`)
  is unchanged. Existing notebooks, tests and benchmarks produce
  identical results to ADR-0004.
- The Pollaczek kernel for the magnetic Green's function on a
  layered soil is deferred to ADR-0006 (open). That ADR will
  add `solver/mom_sommerfeld.py` magnetic-kernel support and
  will reuse the same `engine.earth_inductive_model` switch with
  a third value `"sommerfeld"`.

## References

- **Carson, J. R.** (1926). Wave propagation in overhead wires
  with ground return. *Bell Syst. Tech. J.* **5**(4), 539–554.
  Primary source — equations 23, 25–37 are the production code
  path of this ADR.
- **Pollaczek, F.** (1926). Über das Feld einer unendlich langen
  wechselstromdurchflossenen Einfachleitung. *Elektrische
  Nachrichtentechnik* 3(9), 339–360. Layered-earth extension
  (deferred).
- **Deri, A.; Tevan, G.; Semlyen, A.; Castanheira, A.** (1981).
  The complex ground return plane: a simplified model for
  homogeneous and multi-layer earth return. *IEEE Trans. PAS*
  **100**(8), 3686–3693. Used as a sanity check.
- **Oeding, D. & Oswald, B. R.** (2016). *Elektrische Kraftwerke
  und Netze*, 8. Aufl., Springer. §9.4 "Erdrückleiter".
- **Tleis, N. D.** (2008). *Power Systems Modelling and Fault
  Analysis: Theory and Practice*, Newnes. Ch. 3 — modern Carson
  series in SI units.
- **Sunde, E. D.** (1968). *Earth Conduction Effects in
  Transmission Systems*, Dover. Ch. 7 — buried-wire equivalent.

---

## Amendment 2026.11 — Carson on its own baseline (review pass 10, F32)

Findings 10.20 (F32), 10.46 and 10.63.

1. **Baseline.** Carson's correction is the change of the impedance of a
   line and its **anti-parallel** image in a perfectly conducting earth.
   Section *Linear-system integration* added it to the Neumann matrix of
   ADR-0004 with its **additive** perfect-mirror image, which roughly
   doubled the earth-return reactance: 1.66–2.25 times Carson's value for
   100 m – 1 km lines and +41 % on $|Z|$ at 1 kHz for a TN network with
   inductive PEN conductors (review pass 10). Since 2026.11
   `carson_series` assembles the inductance matrix with
   `image="antiparallel"`; a 1 km isolated PEN reproduces
   $l\,X'_\text{Carson}$ within 1 %
   (`tests/test_inductive_solver_integration.py::test_carson_series_reactance_matches_carson`).
2. **Regimes.** The small-$a$ regime ($a \le 0.25$) evaluates Carson's
   series to $a^{12}$, with the sign rule read as the sign of each
   coefficient (the literal recursion is $10^{-9}$ off at $a = 0.25$); the
   self-term quadrature at $\theta = 0$ uses a graded composite rule and is
   at round-off; the asymptotic expansion is used only for $a > 15$ (the
   boundary at $a = 5$ jumped by up to $9.3 \cdot 10^{-4}$), with the sign
   of the $a^{-7}$ term of $Q$ corrected. Regime 2 therefore covers
   $0.25 < a \le 15$.
3. **Cost.** $P$ and $Q$ are memoised (bit-identical) and the corrections
   of all frequencies are built in one assembly.
4. **Positioning.** `carson_series` is the fast approximation for long
   parallel lines over homogeneous earth; it is not a general replacement
   of `sommerfeld`, and the documentation no longer calls the two
   "identical".
5. **Validity domain (second adversarial audit, A2C-1).** The per-metre
   tiling — the per-metre correction at the horizontal axis distance times
   the length over which the two segments overlap — is kept for
   near-parallel pairs ($|\cos\varphi| > 0.99$) of near-horizontal
   segments (inclination below 8.1°) on one side of the surface. A pair
   with one wire overhead and one buried has no anti-parallel baseline (a
   perfect conductor screens the soil) and couples through the field
   transmitted into the soil, `carson_mixed_mutual_correction` (Carson's
   $J$ and the remainder of the buried wire's attenuation; Ametani et al.
   2009). Against an independent quadrature it agrees to 4.2e-9 for
   x ≤ 300 m ($q = x\sqrt{\omega\mu_0\sigma} \le 8.4$ at 1 kHz in
   10 Ω·m; 1e-10 at 50 Hz) and to 1.2e-7 at x = 1 km ($q = 28$), where
   Carson's $J$ itself is evaluated by QUADPACK's Fourier rule (the "1e-9"
   first stated here was measured up to 300 m). Oblique pairs and every
   pair and self term of a segment inclined by more than 8.1° get the
   Sommerfeld correction of a homogeneous earth of the same conductivity;
   before, an oblique pair got the per-metre value once per segment pair,
   which grew with the number of segments (two 1 km leads at 9°: 20× the
   Sommerfeld coupling). At the 8.1° boundary the result steps by the
   error of the per-metre model there (12 % for two 1 km leads 2 m apart
   at 1 kHz; 12.8 % after item 6).
6. **Order of the segments (third adversarial audit).** Near-parallel
   pairs that are not exactly parallel were projected on the axis of the
   pair's first segment: reversing the segment order moved entries by up
   to 3.1 % of the largest one, and a pair could get an overlap in one
   order and none in the other. They are projected on the bisector of the
   two axes from the mean of the mid-points; every order and direction
   gives the same matrix. Exactly parallel pairs keep their projection.
   The step at 8.1° of item 5 becomes 12.8 % for two 1 km leads 2 m apart
   at 1 kHz (12.3 % with the old projection; 42 % for two 300 m leads at
   50 Hz).
7. **Tilted pairs per stretch (audit W1R3-07).** A near-parallel pair that
   is not parallel took the per-metre value at its mid-point distance for
   the whole overlap, so a lumped pair and its pieces differed: a core
   crossing its shield's axis at mid-length, loop reactance at 1 kHz,
   4.3e-4 at 2e-3 rad, 1.4e-3 / 4.4e-3 / 1.9e-2 / 5.2e-2 at 4e-3 / 8e-3 /
   2e-2 / 4e-2 rad. Where the pair comes close within its overlap — its
   smallest distance there is below 30 times the change of its horizontal
   offset along the overlap — the correction is integrated over the
   overlap with Gauss–Legendre nodes (4, plus 8 per change of the offset
   by $h_i + h_j$, at most 32; on one depth no closer than touching).
   Elsewhere the mid-point value is exact to second order (an error of
   about $(\text{change}/d)^2/24$ of the entry, below 5e-5) and the pair
   keeps its single entry, as do pairs parallel up to round-off (bit for
   bit; the golden cases do not move). Lumped pair and pieces now agree
   within 2.9e-6 in the matrix and 8.0e-6 of the solved loop. The
   100-house star has 3173 tilted entries among its 6710, none of which
   comes that close: it keeps every entry and its cost (integrating all of
   them cost 5.5 s of QUADPACK per frequency for 1.8e-6 of X).
8. **Accuracy for networks of short cables (stream w3perf, item 7).**
   Carson's correction belongs to infinitely long lines; tiled over the
   pieces of short service drops (30 … 60 m at a skin depth of 159 m,
   1 kHz, 100 Ω·m) it overstates the earth return of a generated LV
   network. Against `sommerfeld` with a refined discretisation (PEN in
   2.5 m pieces, 1 m segments; the default `sommerfeld` agrees with it to
   8e-5), the input impedance of the generated star (100 houses) and grid
   (100 and 50 houses) networks with an insulated Neumann PEN reads X
   +9.3 … +14.9 % and R +5.5 … +10.9 % high at 1 kHz under
   `carson_series`, and this error changes with the geometry. A change of
   the network is therefore reproduced by `carson_series` only to within
   that error: across the cabinet fix of audit W1B-01/02 the star's X
   moves by +1.18 % under `carson_series` and −3.54 % under `sommerfeld`
   (the refined reference: −3.54 %), across the cabinet threshold of the
   100-house grid by +1.81 / −2.77 % (reference −2.77 %), of the 50-house
   grid by −21.6 / −18.9 % (reference −18.9 %). For such networks use
   `sommerfeld`; `carson_series` is the fast choice for long parallel
   lines.

