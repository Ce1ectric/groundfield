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
  typical (sub-kHz, $\rho_\text{earth} \in [50, 5000]\,\Omega\,\mathrm{m}$)
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
| Reference computations, layered earth, mixed wire lengths | `"sommerfeld"` |
| Long parallel PEN over homogeneous earth, fast scoping | `"carson_series"` |
| Pure DC studies, perfect-mirror reference | `"perfect_mirror"` |

::: groundfield.coupling.sommerfeld_inductance

## Cross-layer Green's function (ADR-0007 Phase B)

`groundfield.coupling.layered_green` solves the two-layer matching
problem for the **electric** Green's function — used by `image_2layer`,
`mom_sommerfeld`, `cim`, and `bem` whenever a source / observer pair
straddles the upper-layer interface (driven rods, deep meshes,
foundation electrodes that cross $z = h_1$).

Entry points:

- ``two_layer_spectral_kernel`` — the kernel
  $\widetilde{G}(\lambda; z, z_s)$ in spectral space,
- ``two_layer_real_space_kernel`` — its real-space counterpart
  $G(s, z, z_s)$ obtained by Sommerfeld inversion,
- ``two_layer_layered_correction_real_space`` /
  ``two_layer_layered_correction_group`` — the layered *correction*
  $\Delta G = G_\text{2-layer} - G_\text{homog}$ for one distance or
  for a whole reaction block at one $(z, z_s)$ pair,
- ``two_layer_probe_matrix`` — the post-processing potential matrix
  $\varphi = G\,I$ for arbitrary probe/source layer combinations.

Together with the cross-layer-aware self-action factory
(``_two_layer_self_kernel_factory``) they lift the long-standing
``z_max < h_1`` precondition for ``n_layers == 2``. For
``n_layers >= 3`` the layered backends still emit a documented
``UserWarning`` — the n-layer extension is on the roadmap.

### Numerical strategy: analytic primary term, quadrature for the rest

The spectral kernel splits into a **primary** term — the particular
solution $\rho_k/(2\lambda)\,e^{-\lambda|z-z_s|}$ in the layer that
contains the source — and a **reflected** remainder generated by the
free surface at $z = 0$ and the interface at $z = h_1$. The two behave
very differently in $\lambda$ and are treated differently:

$$
G(s, z, z_s) \;=\;
\underbrace{\frac{\rho_k}{2\,\sqrt{s^2 + (z-z_s)^2}}
            \;+\; \sum_{d_j < d_\text{cut}} \frac{c_j}{2\sqrt{s^2+d_j^2}}}
           _{\text{closed form}}
\;+\;
\underbrace{\int_0^{\lambda_\text{max}}\!
            \Big[\Phi_\text{refl}(\lambda)
            - \!\!\sum_{d_j < d_\text{cut}}\!\!
              \tfrac{c_j}{2\lambda}e^{-\lambda d_j}\Big]
            J_0(\lambda s)\,\lambda\,d\lambda}
           _{\text{split-grid quadrature}} .
$$

The primary term is the only part that does **not** decay in
$\lambda$ at all, and its Hankel inverse is the exact $1/R$ field.
Since 0.15.0 it is evaluated in closed form.

The reflected remainder decays like $e^{-\lambda d_\text{img}}$ with
$d_\text{img}$ the smallest image distance — which is **not** by itself
a licence to truncate, because every image distance can degenerate:
$z + z_s \to 0$ for a pair at the air/soil surface,
$|2h_1 - z - z_s| \to 0$ for a pair at the interface, and
$z_l - z_u \to 0$ for a pair straddling it. Up to 0.15.0.dev
$\lambda_\text{max}$ was `lambda_max_factor` $/\min(h_1, s+z+z_s)$,
which has no relation to $d_\text{img}$, and those three regimes were
wrong by −2.7e-2, +4.1 % and **−39.5 %** respectively with no warning —
the last one *growing as the mesh was refined* towards $h_1$.

Since 0.15.0 the truncation is guaranteed instead of assumed. The
reflected kernel is written as its exact image superposition
$\sum_j \frac{c_j}{2\lambda}e^{-\lambda d_j}$ (the Sunde/Tagg series,
for all four layer combinations); every term with
$\lambda_\text{max} d_j < 30$ is subtracted from the integrand and
added back through its exact Hankel inverse $c_j/(2R_j)$, so the
discarded tail is bounded by
$\sum_j |c_j| e^{-\lambda_\text{max} d_j}/(2 d_j) \le e^{-30}
\sum_j |c_j| / (2 d_j)$ by construction. Measured against an
independently derived closed form, surface pairs, interface pairs,
pairs a micrometre below the interface, pairs straddling it and radii
from 0 to 20 km all agree to ~3e-10.

Up to 0.14.1 the primary term went through the quadrature as well,
which left an $O((\lambda_\text{max}R)^{-1/2})$ truncation residue
*oscillating* in $s$ with period $2\pi/\lambda_\text{max}$: −4.5 % at
$s = 1$ m and ±4 % swings around $s = 10$ m for an equal-depth pair in
$h_1 = 6.1$ m soil. The homogeneous limit
$\rho_2 = \rho_1 \Rightarrow \rho/2\,(1/r + 1/r')$ is now reproduced
to ~1e-10 instead of ~1e-2 (ADR-0007 validation item 1 asks for 1e-9).

At $s = 0$ with $z = z_s$ every $1/R$ pole is clamped at
$R = 1/\lambda_\text{clamp}$, $\lambda_\text{clamp} =$
`lambda_max_factor` $/\min(h_1, s+z+z_s)$ — so the clamp depends only
on the knob and the depths, never on the largest radius that happens to
share a group. The value stays finite and continuous but is not a
physical potential; the caller regularises at the wire radius.

Depths must satisfy $z, z_s \ge 0$ (positive into the soil). Negative
depths raise `ValueError`; up to 0.15.0.dev they returned `nan`, which
propagated silently into `grounding_impedance()`.

### Oscillation resolution and `SommerfeldResolutionWarning`

The linear region of the $\lambda$ grid places one 16-node Gauss panel
per $J_0(\lambda s)$ period, so the required node count grows with the
oscillation content
$n_\text{osc} = \lambda_\text{max}\,s / 2\pi$. The policy has three
tiers:

| Regime | Resolution |
|---|---|
| $16\,n_\text{osc} \le 2^{16}$ nodes | one panel per period (16 nodes / oscillation) |
| beyond the soft budget | reduced to the floor of 8 nodes / oscillation |
| $8\,n_\text{osc} > 2^{20}$ nodes | hard allocation budget — `SommerfeldResolutionWarning` |

Eight nodes per oscillation is the same adequacy criterion the
single-panel fast path uses at small radius. Before 0.15.0 the panel
count was capped at 4096 with no adequacy check, so raising
`lambda_max_factor` from 200 to 6000 turned a layered correction of
−0.2311 into **+0.0349** — a sign flip with no diagnostic.

### What `lambda_max_factor` does since 0.15.0

Because the truncated tail is bounded by the closed-form image split
above rather than by making $\lambda_\text{max}$ large,
$\lambda_\text{max}$ is now chosen by the module and
`lambda_max_factor` is only an *upper bound* on it plus the $1/R$
regularisation wavenumber. The choice is:

| Step | Rule |
|---|---|
| requested | $\lambda_\text{user} =$ `lambda_max_factor` $/\min(h_1, s_\text{min}+z+z_s)$ |
| cost cap | at most $1024 \cdot$ `lambda_max_factor`$/200$ panels, never more than 4096 (the historic allocation) |
| decay cap | $120 / d_\text{rest}$ — no point integrating past $e^{-120}$ |
| repair | raised back to $30/d_\text{rest}$ if the image-term budget ran out (then also warns) |

Consequences, both intended:

* **Raising the knob normally does nothing at all.** Measured: the full
  kernel of a surface pair is identical to ten digits for any factor
  from 30 to 30 000. That is the resolution of F14 — the knob can no
  longer make the answer worse, because the answer no longer depends on
  it.
* **Lowering it is cheap and equally exact**, because work moves into
  the closed-form image sum. A convergence check should therefore sweep
  *downwards* as well, and see nothing move.
* Because $\lambda_\text{max}$ no longer grows with the knob or with
  $s$, `SommerfeldResolutionWarning` is no longer reachable from a
  public entry point; it remains as a guard behind the image-term
  budget, and is tested through it.

The cost cap is also what fixes the multi-kilometre performance of the
remote-injection workload: at 0.15.0.dev a 4 km group cost 42x a 90 m
one (the node count grew like $\lambda_\text{max}\,s$ with nothing
bounding it). Measured for a 4000-pair group, $z = z_s = 0.7$ m,
$h_1 = 5$ m, $\rho = 1000/30$, default knob:

| $s_\text{max}$ | v0.14.1 | 0.15.0.dev | 0.15.0 |
|---|---|---|---|
| 90 m | 94 ms | 131 ms | 59 ms |
| 400 m | 186 ms | 418 ms | 49 ms |
| 1 km | 190 ms | 1 107 ms | 66 ms |
| 4 km | 199 ms | 11 216 ms | 71 ms |
| 20 km | 196 ms | 11 641 ms | 75 ms |

!!! note
    `coupling.sommerfeld_inductance._build_lambda_grid` (the magnetic
    counterpart, ADR-0006) still carries the historic unchecked
    4096-panel cap with 8-node panels.

### Large-group interpolation

Reaction blocks share one $(z, z_s)$ pair, so the kernel there depends
on the horizontal distance only, and what is left for the quadrature
after the closed-form split is analytic in $t = \ln(s + c)$ with
$c = d_\text{rest}$, the decay length of that remainder. Groups above
64 distance pairs are therefore sampled on a log-spaced $s$-grid —
sized from the group's actual dynamic range at ~12 nodes per e-fold —
and interpolated with a cubic spline, turning a block from
$O(n_\text{pairs} \cdot n_\lambda)$ into
$O(n_\text{nodes} \cdot n_\lambda)$. Interpolation is skipped whenever
it would need at least as many nodes as the group has pairs. The
closed-form primary and image terms are always evaluated per pair,
outside the interpolation.

Measured agreement with the exact per-pair contraction is ~1e-6
(worst case 1.1e-5, at a surface pair as $s \to 0$ where the *scalar*
branch is the less accurate of the two), against ~1e-2 for the fixed
48-node linear interpolation used up to 0.14.1; for
`two_layer_probe_matrix` 8e-7 against a documented 1e-4. Because the
node count no longer depends on a fixed constant, the *same* matrix
entry no longer changes with the number of pairs in its block — that
discontinuity was ~3 % at the 64-pair threshold and is now below 1.1e-5
for every group size from 60 to 129 pairs.

!!! warning "Do not restate this as ~1e-8"
    Source comments claiming "~1e-8 relative" and "four orders of
    margin" stood in `layered_green.py` at 0.15.0.dev; they were 2–3
    orders optimistic and contradicted this page. The numbers above are
    measured.

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
