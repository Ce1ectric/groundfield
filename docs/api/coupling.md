# Coupling

The `groundfield.coupling` package collects every conductor-to-conductor
and conductor-to-earth coupling relation that the solvers consume.
The structure mirrors the physical decomposition: galvanic coupling
through cluster nodes (handled by the solver builder, no module of
its own), inductive coupling between distributed-conductor segments
(`inductance.py`, ADR-0004), and the Carson earth-return correction
that adds finite-conductivity effects to the inductive matrix
(`carson.py`, ADR-0005).

## Inductance — Neumann self and mutual integrals (ADR-0004)

Implements the Neumann partial-inductance assembly under a
**perfect-mirror** earth assumption. Each pair of distributed-conductor
segments contributes a self- or mutual-inductance entry to the
branch-impedance block

$$
Z_b(\omega) \;=\; R \;+\; j\omega\, L_\text{Neumann}.
$$

The thin-wire self-inductance uses Grover 1946 (closed form, plus
optional internal-field $\mu_0/(8\pi)$ contribution for the DC
limit). Off-diagonal entries are evaluated by a closed-form
parallel-segments fast path or, for arbitrary 3-D geometries, by
a 16×16 Gauss–Legendre quadrature of the Neumann double-line
integral. The image of every segment in the soil surface
contributes one extra Neumann integral against the original — this
is the perfect-mirror assumption and the starting point for the
Carson refinement.

::: groundfield.coupling.inductance

## Carson earth-return correction (ADR-0005)

Adds Carson 1926's finite-conductivity correction
$\Delta Z_\text{Carson}(\omega)$ on top of the perfect-mirror
inductance. The branch-impedance block becomes

$$
Z_b(\omega) \;=\; R \;+\; j\omega\, L_\text{Neumann} \;+\;
\Delta Z_\text{Carson}(\omega, \sigma_\text{earth}, h_i, h_j, d_{ij}).
$$

The correction is evaluated as

$$
\Delta Z_\text{Carson}(\omega) \;=\; \frac{\omega\,\mu_0}{\pi}\,
\bigl[P(a, \theta) \,+\, j\,Q(a, \theta)\bigr],
\qquad
a \;=\; D\,\sqrt{\omega\,\mu_0\,\sigma_\text{earth}}
\;=\; \frac{D\sqrt{2}}{\delta(\omega)},
$$

with $D = 2h_i$ ($\theta = 0$) for the self contribution and
$D = \sqrt{(h_i+h_j)^2 + d_{ij}^2}$,
$\theta = \arctan(d_{ij}/(h_i+h_j))$ for the mutual contribution.
$\delta(\omega) = \sqrt{2 / (\omega\mu_0\sigma_\text{earth})}$ is the
electromagnetic skin depth in soil — the natural length scale at
which the perfect-mirror approximation starts to break down.

### Three regimes

Following Carson 1926 §III the implementation switches between
three numerical regimes:

| Regime | Range of $a$ | Method |
|---|---|---|
| Small | $a \le 0.25$ | Closed-form leading-term expansion (Carson eqs. 34/35) |
| Intermediate | $0.25 < a \le 5$ | 64-point Gauss–Legendre quadrature of Carson eq. 29 |
| Asymptotic | $a > 5$ | Inverse-power expansion (Carson eqs. 36/37) |

### Validity and limitations

- **Homogeneous soil** — the Carson series is exact (within the
  quasi-static / sub-kHz assumption Carson himself states) when
  $\sigma_\text{earth}$ is uniform.
- **Layered soil** — the implementation falls back to
  $\sigma = 1/\rho_1$ of the upper layer with a runtime
  `UserWarning`. For a rigorous result switch to
  `backend="mom_sommerfeld"`, which uses the full Pollaczek
  kernel.
- **Frequency** — derived for $\omega \ll 1 / \mu_0\sigma$; for
  AP1 (sub-kHz, $\rho_\text{earth} \in [50, 5000]\,\Omega\,\mathrm{m}$)
  the assumption is comfortably satisfied.
- **Geometry** — Carson's derivation assumes parallel wires above
  a plane homogeneous half-space. Non-parallel segment pairs are
  handled by projection onto the parallel component (orthogonal
  components contribute zero by Neumann symmetry).

::: groundfield.coupling.carson

## Sommerfeld geometric earth-return Green's function (ADR-0006)

The rigorous formulation of the earth-return inductive coupling.
Integrates the σ-dependent vector-potential Green's function

$$
G_\text{mag}(\vec{r}, \vec{r}';\,\omega,\sigma_e) \;=\;
\frac{1}{R} \;+\; \int_0^{\infty}\!
\Gamma_\text{mag}(\lambda)\,
e^{-\lambda(z+z')}\,J_0(\lambda\rho)\,d\lambda
$$

over the actual segment-pair geometry, where the reflection
coefficient $\Gamma_\text{mag}$ encodes the homogeneous or
layered-earth structure. Reduces to the perfect-mirror case at
$\sigma\to\infty$, to free space at $\sigma\to 0$, and converges
to ADR-0005's per-meter Carson asymptote for long parallel wires
over homogeneous earth. Unlike Carson, it correctly handles short
wires, non-parallel geometries, and **layered earth without
approximation** (Pollaczek/Wait kernel).

### When to use which

| Problem class | Recommended option |
|---|---|
| AP1 dissertation work, layered earth, mixed wire lengths | `"sommerfeld"` |
| Long parallel PEN over homogeneous earth, fast scoping | `"carson_series"` |
| Pure DC studies, perfect-mirror reference | `"perfect_mirror"` |

::: groundfield.coupling.sommerfeld_inductance

## Cross-layer Green's function (ADR-0007 Phase B)

`groundfield.coupling.layered_green` solves the two-layer matching
problem for the **electric** Green's function — used by `image_2layer`,
`mom_sommerfeld`, `cim`, and `bem` whenever a source / observer pair
straddles the upper-layer interface (driven rods, deep meshes,
foundation electrodes that cross $z = h_1$).

Two entry points:

- ``two_layer_spectral_kernel`` — the kernel
  $\widetilde{G}(\lambda; z, z_s)$ in spectral space,
- ``two_layer_real_space_kernel`` — its real-space counterpart
  $G(s, z, z_s)$ obtained by Sommerfeld inversion.

Together with the cross-layer-aware self-action factory
(``_two_layer_self_kernel_factory``) they lift the long-standing
``z_max < h_1`` precondition for ``n_layers == 2``. For
``n_layers >= 3`` the layered backends still emit a documented
``UserWarning`` — the n-layer extension is on the roadmap.

::: groundfield.coupling.layered_green

## Earth-conductivity / earth-layer resolvers

These helpers normalise any soil model to a homogeneous
$\sigma_\text{earth}$ (Carson) or a layered structure (Sommerfeld).
They are typically invoked by the engine builders, not by user code.

```python
from groundfield.coupling import (
    resolve_earth_conductivity, resolve_earth_layers,
)

sigma = resolve_earth_conductivity(world.soil)        # Carson
layers = resolve_earth_layers(world.soil)             # Sommerfeld
```

::: groundfield.coupling.resolve_earth_conductivity

::: groundfield.coupling.resolve_earth_layers
