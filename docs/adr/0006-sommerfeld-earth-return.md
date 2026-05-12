# ADR-0006: Geometric Sommerfeld earth-return Green function

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-05-07 |
| **Deciders** | Christian Ehlert |
| **Scope** | `groundfield`, work package 1 of the dissertation |

## Context

ADR-0005 added Carson 1926's earth-return correction
$\Delta Z_\text{Carson}(\omega)$ in the form

$$
\Delta Z_b^{(i,j)}(\omega) \;\approx\; z'_\text{Carson}(\omega, h_i, h_j, d_{ij})\,\cdot\,\ell_{ij},
$$

where $z'_\text{Carson}$ is Carson's per-meter closed form and
$\ell_{ij}$ is the projected geometric length of the segment pair.
This is the standard **transmission-line modelling** convention used
in EMTP/PSCAD/ATP. It is exact for **infinite parallel wires over a
homogeneous earth**, but it has two structural limitations that
matter for the AP1 dissertation work:

1. **Short conductors / arbitrary geometry.** Carson's per-meter
   formula assumes translation invariance along the wire axis. For
   wires whose length is comparable to the earth skin depth
   $\delta$, or for non-parallel / 3-D arrangements, the per-meter
   closed form does not capture end effects. The AP1 frequency
   range ($f \le 1\,\mathrm{kHz}$) and soil resistivities
   ($\rho_e \in [50, 5000]\,\Omega\,\mathrm{m}$) yield
   $\delta \in [350\,\mathrm{m}, 35\,\mathrm{km}]$ — comparable to
   or larger than typical TN-Ortsnetz distances. Many electrodes
   and short connection wires therefore live in the regime
   $L \lesssim \delta$ where Carson is the wrong asymptote.
2. **Layered earth.** Carson's series is rigorously derived for a
   homogeneous half-space. Layered configurations
   ($\rho_1, \rho_2, h_1$) appear in every AP1 study, and ADR-0005
   handles them only with a `UserWarning` and the upper-layer
   $\rho_1$ — a documented approximation, not a model.

This ADR establishes the **rigorous geometric formulation** of the
earth-return inductive coupling: the vector-potential Green's
function with the σ- (and layer-) dependent earth correction is
integrated over the actual segment-pair geometry. ADR-0005's Carson
series then becomes a **fast asymptotic option**, and ADR-0006
becomes the **default for AP1-relevant studies** where the
asymptotic assumptions break down.

## Decision

### Physical model

For a horizontal current source $I\,d\vec{l}'$ at position $\vec{r}'$
over (or in) a conducting half-space, the quasi-static
vector-potential Green's function is

$$
G_\text{mag}(\vec{r}, \vec{r}';\,\omega,\sigma_e) \;=\;
\frac{1}{R} \;+\; \int_0^{\infty}\!\frac{u_e-\lambda}{u_e+\lambda}\,
e^{-\lambda(z+z')}\,J_0(\lambda\rho)\,d\lambda,
\qquad u_e \;=\; \sqrt{\lambda^2 + j\omega\mu_0\sigma_e}\,,
$$

with $R = |\vec{r}-\vec{r}'|$ the direct distance, $\rho$ the
horizontal distance and $z, z'$ the depths (positive into the
soil). The integrand has the structure
"reflection coefficient × exponential decay × Bessel kernel" —
exactly the same family as the **electric** Sommerfeld kernel
already implemented in `solver/mom_sommerfeld.py`.

The partial mutual impedance between two finite-length 3-D segments
is then

$$
Z_{ij}(\omega) \;=\;
j\omega\,\frac{\mu_0}{4\pi}\!\int_{C_i}\!\!\int_{C_j}
(\hat{l}_i\cdot\hat{l}_j)\,G_\text{mag}(\vec{r}_i, \vec{r}_j;\,\omega,\sigma_e)\,
dl_i\,dl_j .
$$

This is the analogue of ADR-0004's Neumann integral, but with the
σ-dependent kernel replacing the perfect-mirror $1/R + 1/R'$.

### Limit checks (built into the test suite)

| Limit | Result | Validates |
|---|---|---|
| $\sigma_e \to \infty$ | $G_\text{mag}\to 1/R + 1/R'$ → ADR-0004 perfect mirror, **bit-exact regression** | ADR-0004 unchanged |
| $\sigma_e \to 0$ | $G_\text{mag}\to 1/R$ → free space, no image | $\omega \to 0$ same |
| Long parallel wires + homogeneous earth | $\int Z_{ij}$ collapses to Carson's per-m × $\ell$ | ADR-0005 recovered as asymptote |
| Short wires over homogeneous earth | $Z_{ij}$ deviates from Carson by $\mathcal{O}((L/\delta)^2)$ | New physics captured |
| Layered earth | $G_\text{mag}$ uses the **layered** reflection coefficient $\Gamma_\text{mag}^{(n)}(\lambda)$ | AP1 layered studies |

### Two pillars in one ADR

This ADR explicitly addresses both homogeneous and layered earth.
The implementation rolls out in two pillars, but the API and the
mathematical formulation are designed for both from the start so
that no second renaming is needed when the layered piece comes
online.

#### Pillar A — Homogeneous earth (this release)

The reflection coefficient is the Fresnel form

$$
\Gamma_\text{mag}^{(1)}(\lambda;\sigma_e) \;=\; \frac{u_e - \lambda}{u_e + \lambda}.
$$

Implementation:

```python
def earth_return_green_homogeneous(
    *, rho: float, z_i: float, z_j: float,
    omega: float, sigma_earth: float,
) -> complex:
    """G_mag(r, r') - 1/R, the σ-dependent earth correction."""
```

#### Pillar B — Layered earth (same ADR, planned for the AP1 sweep)

For an `n`-layer earth, the reflection coefficient is built from
the recursive Tagg/Sunde-style formula

$$
\Gamma_\text{mag}^{(n)}(\lambda;\rho_1,\dots,\rho_n,h_1,\dots,h_{n-1};\omega) \;=\;
\frac{u_1 - \lambda - (u_1+\lambda)\,\Gamma^{(n-1)}_2(\lambda)\,e^{-2u_1 h_1}}
     {u_1 + \lambda - (u_1-\lambda)\,\Gamma^{(n-1)}_2(\lambda)\,e^{-2u_1 h_1}},
$$

with $u_k = \sqrt{\lambda^2 + j\omega\mu_0\sigma_k}$. For $n=2$
(two-layer), this collapses to a closed form that the test suite
can compare against.

Implementation:

```python
def earth_return_green_layered(
    *, rho: float, z_i: float, z_j: float,
    omega: float, layers: list[SoilLayer],
) -> complex:
    ...
```

`HomogeneousSoil` is implemented as the `n=1` case of the layered
function. Both share the same Sommerfeld-quadrature backend —
only the reflection coefficient differs.

### Engine-side switch

`Engine.earth_inductive_model` gains a third value:

```python
EarthInductiveModel = Literal[
    "perfect_mirror",   # ADR-0004
    "carson_series",    # ADR-0005 (asymptotic per-m Carson)
    "sommerfeld",       # ADR-0006 (geometric Sommerfeld kernel)
]
```

Default remains `"perfect_mirror"` — every existing notebook and
test continues to produce its current bit-exact result. AP1
notebooks opt into `"sommerfeld"` for the dissertation-grade
runs; `"carson_series"` is preserved as the cheap asymptotic
diagnostic and is documented as such.

### Numerics — Sommerfeld quadrature

Each segment-pair contributes one σ-dependent matrix entry
$\Delta Z_{ij}^{Sommerfeld}$. The double integral splits into:

- **Outer (geometry)**: 16×16 Gauss–Legendre over the two segment
  parameterisations. Reuses the `_GL_NODES, _GL_WEIGHTS` set in
  `coupling/inductance.py`.
- **Inner (Sommerfeld)**: adaptive quadrature on $[0, \lambda_\max]$
  with $\lambda_\max = M / (z_i + z_j + \rho)$ chosen so that the
  exponential factor $e^{-\lambda_\max (z_i+z_j)} \le 10^{-13}$.
  64-point Gauss–Legendre is the production scheme; for very small
  $z_i + z_j$ (wires near the surface) we split
  $[0, \lambda_\max]$ into a logarithmic and a linear part to
  resolve the Bessel oscillations.

For horizontal segments at constant depth, the kernel
$K(\lambda) = \Gamma_\text{mag}(\lambda)\,e^{-\lambda(z_i+z_j)}$
factors out of the outer-integration loop and is computed once per
segment-pair. The total cost is

$$
T_{Sommerfeld} \;=\; \mathcal{O}\bigl(N_f \cdot M^2 \cdot N_\lambda\bigr)
$$

evaluations of $J_0$ per outer node, fully vectorised through
`numpy`. For AP1 ($M \le 5000$, $N_\lambda \le 200$, $N_f \le 20$)
this is ≲ 5 minutes total — tolerable for the planned parameter
sweeps.

### `mom_sommerfeld` reuse

The existing electric Sommerfeld backend already evaluates a
similar (but scalar-potential) Sommerfeld kernel. We **do not**
fold the magnetic kernel into `mom_sommerfeld`'s solver loop —
keeping the two separate makes the code path clearer and lets
ADR-0006 land without touching an existing backend. Both kernels
share helper code in a new `coupling/sommerfeld_inductance.py`
module.

## Validation

`tests/test_sommerfeld_inductance.py`:

1. **σ → ∞ collapses to ADR-0004**: with `sigma_earth = 1e9` and a
   buried PEN, the Sommerfeld solution agrees with the perfect-
   mirror solution to within $10^{-6}$ on the cluster impedance.
2. **σ → 0 collapses to free space**: with `sigma_earth = 1e-9`,
   the Sommerfeld correction is negligible compared to the
   self-inductance term.
3. **ω → 0 collapses to perfect mirror**: at DC the correction
   vanishes; the full DC test suite from ADR-0003 must continue
   to pass.
4. **Long-wire homogeneous limit recovers Carson**:
   for a 1 km PEN at 50 Hz, $\rho_e = 100\,\Omega\,\mathrm{m}$,
   `sommerfeld` and `carson_series` agree on the cluster
   impedance to $\le 2\,\%$. For $L = 10\,\mathrm{km}$ the bound
   sharpens to $\le 0.5\,\%$.
5. **Short-wire deviation**: for $L = 10\,\mathrm{m}$ the
   Sommerfeld result deviates from the per-m × $\ell$ Carson
   approximation by $\ge 5\,\%$. The deviation grows with
   $L/\delta \to 0$.
6. **Cross-engine consistency at 50 Hz with `"sommerfeld"`
   active**: `image`, `mom`, `cim`, `bem` agree on the cluster
   impedance to within 5 %.
7. **Two-layer regression**: a `TwoLayerSoil` world with
   $\rho_2/\rho_1 = 10$ produces results that *differ* from the
   homogeneous Sommerfeld baseline by a controlled, frequency-
   monotone amount — i.e. the layered correction kicks in as the
   skin depth in the upper layer crosses $h_1$.
8. **Two-layer textbook check**: at the limit $\rho_2/\rho_1 = 1$
   the layered Sommerfeld solution reproduces the homogeneous
   Sommerfeld solution to within $10^{-9}$.
9. **Field-point consistency**: the potentials reported by
   `result.potential(...)` are unchanged when only the inductive
   model switches — the electric Green's function is independent
   of `earth_inductive_model`.

## Consequences

### Positive

- **AP1 dissertation work runs on a physically rigorous model.**
  Layered earth, short wires, non-parallel geometries — all are
  treated correctly within the same numerical machinery.
- The Carson series stays available as a cheap asymptotic
  reference. Users get a **two-engine self-validation** for free:
  if the two disagree, the geometry is outside the asymptotic
  regime.
- The penetration-depth diagnostic in `FieldResult.metadata`
  becomes more meaningful — the actual physics of the layered
  earth is now in the kernel, so the diagnostic can be augmented
  with a per-pair Carson parameter $a_{ij}$ that diagnoses
  *which* pairs are in the long-wire regime.
- Layered-earth handling no longer needs a `UserWarning` —
  the Pollaczek/Sommerfeld kernel is the production path.

### Negative

- Computational cost grows by $\mathcal{O}(N_\lambda)$ per
  segment-pair per frequency. Modest for AP1 (5 min sweep) but
  substantial for very large worlds (>10 000 segments). For those
  cases the asymptotic Carson model remains available as the
  fallback.
- Three earth-inductive models in the engine schema increases the
  test matrix; we mitigate by parametrising the cross-engine
  tests over the model.

### Neutral

- Default behaviour unchanged. Existing notebooks and tests
  produce identical results.
- Pillar B (layered earth) ships in the same release as Pillar A;
  the API is layered-aware from day one.

## References

- **Sommerfeld, A.** (1909). Über die Ausbreitung der Wellen in
  der drahtlosen Telegraphie. *Ann. Phys.* **28**(4), 665–736.
- **Carson, J. R.** (1926). Wave propagation in overhead wires
  with ground return. *Bell Syst. Tech. J.* **5**(4), 539–554.
- **Pollaczek, F.** (1926). Über das Feld einer unendlich langen
  wechselstromdurchflossenen Einfachleitung. *Elektrische
  Nachrichtentechnik* 3(9), 339–360. Layered-earth extension.
- **Wait, J. R.** (1972). *Electromagnetic Waves in Stratified
  Media*, Pergamon. Ch. 3 — generalised reflection coefficients
  $\Gamma^{(n)}_\text{mag}$.
- **Tleis, N. D.** (2008). *Power Systems Modelling and Fault
  Analysis*, Newnes. Ch. 3 — modern Sommerfeld kernel for
  transmission-line modelling.
- **Stratton, J. A.** (1941). *Electromagnetic Theory*,
  McGraw-Hill. §9-10 — derivation of the half-space vector
  potential.
