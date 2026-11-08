# Coupling

The `coupling` package collects the coupling relations the solvers
consume: the Neumann partial inductances between conductor segments
(`inductance`, ADR-0004), the earth-return correction of that
inductance — Carson's per-metre series for long parallel wires over
homogeneous earth (`carson`, ADR-0005) and the geometric
Sommerfeld/Pollaczek integral for any geometry and any number of
layers (`sommerfeld_inductance`, ADR-0006) — and the scalar Green's
function of two-layer soil for electrodes through the layer interface
(`layered_green`, ADR-0007). There is no capacitive coupling: below
1 kHz it is negligible for buried conductors.

## Mathematical and physical context

Conductors with `inductance_model="neumann"` are split into segments
whose series impedance matrix is

$$
\mathbf{Z}_b(\omega) = \mathbf{R} + j\omega\,\mathbf{L} + \Delta\mathbf{Z}_\text{earth}(\omega),
\qquad
L_{ij} = \frac{\mu_0}{4\pi}\int_{\Gamma_i}\!\int_{\Gamma_j}
\frac{\mathrm{d}\mathbf{l}_i\cdot\mathrm{d}\mathbf{l}_j}{\lvert\mathbf{r}_i - \mathbf{r}_j\rvert},
$$

the Neumann double-line integral (closed forms for parallel and touching
segments, Gauss–Legendre otherwise) plus the internal inductance
$\mu_0\ell/(8\pi)$ on the diagonal. The earth enters twice: as an image
term in $\mathbf{L}$ and as the frequency-dependent correction
$\Delta\mathbf{Z}_\text{earth}$; the two are only meaningful together,
which is why every builder takes the same `image=` convention
(`"none"`: free space; `"additive"`: perfect magnetic mirror;
`"antiparallel"`: perfect electric conductor).

**Carson (1926).** For a long straight wire above homogeneous earth of
conductivity $\sigma$, against the antiparallel image,

$$
\Delta Z_\text{Carson} = \frac{\omega\mu_0}{\pi}\bigl[P(a,\theta) + jQ(a,\theta)\bigr],
\qquad a = D\sqrt{\omega\mu_0\sigma} = \frac{\sqrt{2}\,D}{\delta},
$$

with $D = 2h$ (self) or the image distance of a pair (mutual) and
$\delta = \sqrt{2/(\omega\mu_0\sigma)}$ the skin depth in the soil.
$P, Q$ are evaluated by Carson's series ($a \le 0.25$), quadrature
($0.25 < a \le 15$) and the asymptotic expansion ($a > 15$). The
correction is per metre and scaled by the segment-pair length, so it is
exact only for long parallel wires; for a layered soil the upper-layer
resistivity is used with a warning.

**Sommerfeld / Pollaczek.** The quasi-static vector-potential Green's
function of the layered half-space,

$$
G(\rho, z, z') = \delta_{kj}\,\frac{e^{-\gamma_k R}}{R}
+ \int_0^\infty \lambda\,F_{kj}(\lambda; z, z')\,J_0(\lambda\rho)\,\mathrm{d}\lambda,
\qquad \gamma_k^2 = j\omega\mu_0\sigma_k ,
$$

is integrated over every segment pair (time convention $e^{+j\omega t}$).
It handles short and non-parallel wires, buried and overhead conductors
and any number of layers, and is the default
(`Engine.earth_inductive_model="sommerfeld"`). In the long-wire limit it
gives the Pollaczek line impedance with the earth-return resistance
$R' \approx \omega\mu_0/8$.

## Choosing the earth-return model

| `earth_inductive_model` | Use for | Limits |
|---|---|---|
| `"sommerfeld"` (default) | quantitative $Z(f)$ of buried and overhead conductors, layered soil, short or bent conductors | cost grows with the number of segment pairs; `sommerfeld_assembly="tabulated"` for large networks |
| `"carson_series"` | fast scoping of long parallel conductors over homogeneous soil | per-metre approximation; layered soil uses $\rho_1$ (warning); rejects segments in the surface plane |
| `"perfect_mirror"` | diagnostic baseline only | frequency-independent image, far from the physics of buried conductors below 1 kHz |

The theory, the validation against Carson's and Pollaczek's line
impedances, and the history of these models are on the
[earth return](../../explanation/earth-return.md) page.

## Example

The per-metre impedance of a conductor buried 0.7 m deep (wire radius
5 mm, internal inductance included via $\mathrm{GMR} = a\,e^{-1/4}$),
from the closed-form references the solver is validated against:

```python
import math

from groundfield.references import earth_return, pollaczek

gmr = 0.005 * math.exp(-0.25)
for f in (50.0, 1000.0):
    z_carson = earth_return.carson_self_impedance(100.0, f, gmr=gmr)       # Ω/m
    z_buried = pollaczek.pollaczek_self_impedance(100.0, 100.0, 5.0, f, depth=0.7, gmr=gmr)
    z_layered = pollaczek.pollaczek_self_impedance(100.0, 1000.0, 5.0, f, depth=0.7, gmr=gmr)
    # homogeneous soil: the buried (Pollaczek) and Carson line impedances agree
    assert abs(z_buried - z_carson) / abs(z_carson) < 1e-3
    # earth-return resistance omega*mu0/8: 49.3 mOhm/km at 50 Hz
    assert abs(z_carson.real - earth_return.carson_earth_return_resistance(f)) < 1e-12
    # 1000 Ohm m below 5 m of 100 Ohm m soil: about 9 % more reactance
    assert 1.08 < z_layered.imag / z_buried.imag < 1.10
```

## Neumann inductance (ADR-0004)

::: groundfield.coupling.inductance
    options:
      heading_level: 3

## Carson earth return (ADR-0005)

::: groundfield.coupling.carson
    options:
      heading_level: 3

## Sommerfeld / Pollaczek earth return (ADR-0006)

::: groundfield.coupling.sommerfeld_inductance
    options:
      heading_level: 3

## Two-layer scalar Green's function (ADR-0007)

::: groundfield.coupling.layered_green
    options:
      heading_level: 3

## Earth-model resolvers

::: groundfield.coupling.resolve_earth_conductivity
    options:
      heading_level: 3

::: groundfield.coupling.resolve_earth_layers
    options:
      heading_level: 3
