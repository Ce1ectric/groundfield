# Engine family

`groundfield` computes the quasi-static potential field of current-injected
electrodes and conductors in a horizontally layered half-space. Several
numerical backends ("engines") solve this one physical problem. They share
the data model (`World`, electrodes, conductors, sources, junctions), the
solve pipeline and the network equations of bonded clusters, and they
differ in two places: **which Green's function** they evaluate (closed-form
image series or numerical Sommerfeld quadrature) and **which current model**
they solve for (one uniform current per straight wire of an electrode,
or one current per segment). This page explains the family, how to choose an engine and where
the deprecated names come from. The theory of each engine has its own page:

| Page | Engine | Soil | Green's function | Current model |
|---|---|---|---|---|
| [`image`](image.md) | `image` | homogeneous | point source + air image, closed form | uniform per electrode |
| [`image_2layer`](image_2layer.md) | `image_2layer` | two layers | Tagg / Sunde image series; Sommerfeld kernel across the interface | uniform per electrode and layer |
| [`mom`](mom.md) | `mom` | one or two layers | the kernels of the image family | one current per segment (point matching) |
| [`mom_sommerfeld`](mom_sommerfeld.md) | `mom_sommerfeld` | any number of layers | Sommerfeld integral by quadrature | one current per segment (point matching) |
| [`fem`](fem.md) | `fem` (experimental) | any number of layers | volume PDE of an equivalent hemisphere | none (one hemisphere per cluster) |

The shared ingredients have their own pages too:
[kernels](../kernels.md) (point source, exact line source, Galerkin self
term), [earth return](../earth-return.md) (inductive coupling of conductors),
[concrete shells](../concrete-shells.md) and
[sources and measurement](../sources-and-measurement.md). The measured
accuracy of every engine is on the generated
[validation page](../validation.md).

## Mathematical and physical context

Below 1 kHz the soil is a conductor whose displacement current is
negligible ($\omega\varepsilon\rho \ll 1$) and whose skin depth
$\delta = \sqrt{2\rho/(\omega\mu_0)}$ — about 500 m for 100 Ω·m at 1 kHz,
more at lower frequencies and higher resistivities — is large against the
extent of the electrodes. The galvanic part of the problem is therefore
**stationary current flow** for each frequency: with $\sigma = 1/\rho$
piecewise constant in depth,

$$
-\nabla\cdot\bigl(\sigma(z)\,\nabla\varphi\bigr) = q
\quad\text{in the soil } (z > 0),\qquad
\frac{\partial\varphi}{\partial z}\Big|_{z=0} = 0,\qquad
\varphi \to 0 \text{ for } |\mathbf r|\to\infty ,
$$

with continuity of $\varphi$ and of the normal current density
$\sigma\,\partial\varphi/\partial z$ at every layer interface ($z$ points
into the soil). Frequency enters through conductors only: their
longitudinal impedance (resistance, skin effect, partial inductances and
the earth-return correction, see [earth return](../earth-return.md)) makes
the network equations complex and frequency dependent, while the soil
kernel stays real. A galvanic world is therefore solved once and reused for
every frequency.

The Green's function of this problem, the potential of a unit point
current at depth $z_s$ observed at horizontal distance $s$ and depth $z$ in
the top layer, is the Sommerfeld integral

$$
G(s, z; z_s) = \frac{\rho_1}{4\pi}\int_0^\infty
\frac{e^{-\lambda|z-z_s|} + e^{-\lambda(z+z_s)}
 + \Gamma_1(\lambda)\bigl[e^{-\lambda(2h_1-|z-z_s|)} + e^{-\lambda(2h_1-z-z_s)}\bigr]}
{1-\Gamma_1(\lambda)\,e^{-2\lambda h_1}}\;J_0(\lambda s)\,\mathrm d\lambda ,
$$

where $\Gamma_1(\lambda)$ is the reflection coefficient of the stack below
the first interface, built bottom-up from the interface coefficients
$K_i = (\rho_{i+1}-\rho_i)/(\rho_{i+1}+\rho_i)$. For one layer
$\Gamma_1 = 0$ and $G$ is the point source plus its air image; for two
layers $\Gamma_1 = K_1$ is constant and the denominator expands into the
Tagg / Sunde image series; for three and more layers no closed form exists
and the integral is evaluated numerically. The engines differ in how they
evaluate $G$ and in how they discretise the electrodes.

### Two current models

All integral engines discretise every wire into straight segments of
length at most `Engine.segment_length` (and split them at layer
interfaces). They then differ in the unknowns:

- **Uniform current per straight wire** (`image`, `image_2layer`). Each
  straight wire of an electrode — a full wire line of a grid, an arm of a
  star, a straight run of a polyline (user decisions Q30 and Q44 of
  2026.11), and each electrode *per soil layer* it reaches — carries one
  leakage current, spread uniformly per unit length over its segments;
  the wires of an electrode are ideally bonded. The unknowns are a
  handful per electrode, the reaction matrix is reduced with length
  weights to a wire-by-wire multiport matrix, and the solve is cheap. The
  ansatz is exact for a ring by symmetry and good for impedances of
  compact electrodes (grids about 1 % above one current per segment); it
  misplaces current along each wire — towards the tips of a star and the
  ends of a strip — which matters for surface potentials near the
  electrode (touch and step voltages) more than for impedances.
- **One current per segment** (`mom`, `mom_sommerfeld`). Point matching
  (collocation): one pulse current per segment, the potential matched at
  every segment midpoint. This resolves how the current distributes along
  and between electrodes; it uses a point-source near field, which leaves
  it 1–2 % below the exact thin-wire solution and drifting under
  refinement (see [`mom`](mom.md)).

`fem` is not a third current model of the same problem: it replaces each
cluster by an equivalent hemisphere (experimental, see [`fem`](fem.md)).

### Network equations

Electrodes bonded by ideal conductors form a cluster with one potential;
finite conductors (lumped, distributed, inductive) are branches between
nodes; [junctions](../sources-and-measurement.md#junctions) are nodes
without soil contact. Every engine of the integral family assembles the
same augmented nodal system — multiport relation between leakage currents
and node potentials, Kirchhoff's current law at every node, and Ohm's law
along every branch:

$$
\begin{bmatrix} Z & -C & 0 \\ C^{\top} & 0 & B^{\top} \\ 0 & B & -Z_b(\omega) \end{bmatrix}
\begin{bmatrix} \mathbf I_e \\ \boldsymbol\varphi_n \\ \mathbf I_b \end{bmatrix}
=
\begin{bmatrix} \mathbf 0 \\ \mathbf I_\text{in} \\ \mathbf 0 \end{bmatrix},
$$

with $Z$ the soil reaction (multiport) matrix, $C$ the owner-to-node
incidence, $B$ the branch-to-node incidence and $Z_b$ the branch impedance
matrix. Sources inject $+I$ at `attached_to` and, if set, $-I$ at
`return_to` (two-terminal sources, see
[sources and measurement](../sources-and-measurement.md)). A node without a
source floats: it takes the potential the field imposes with zero net
current, which keeps transfer impedances reciprocal.

An impressed voltage source (`VoltageSource`, since 2026.11) is added by
modified nodal analysis: its current $I_v$ becomes one more unknown, which
enters Kirchhoff's current law at `attached_to` and `return_to`, and one
more row fixes $\varphi_a - \varphi_r + Z_q I_v = U$ ($\varphi_r = 0$
for remote earth; $Z_q = R + j\omega L$ the internal impedance of the
source, zero for an ideal source). With the terminal incidence $S$ and the
diagonal matrix $Z_q(\omega)$ of the internal impedances the system grows
by one block row and column,

$$
\begin{bmatrix} Z & -C & 0 & 0 \\ C^{\top} & 0 & B^{\top} & -S \\ 0 & B & -Z_b(\omega) & 0 \\ 0 & S^{\top} & 0 & Z_q(\omega) \end{bmatrix}
\begin{bmatrix} \mathbf I_e \\ \boldsymbol\varphi_n \\ \mathbf I_b \\ \mathbf I_v \end{bmatrix}
=
\begin{bmatrix} \mathbf 0 \\ \mathbf I_\text{in} \\ \mathbf 0 \\ \mathbf U \end{bmatrix},
$$

and the solution of a world without voltage sources is unchanged. A
reactive $Z_q$ makes the system complex at $f > 0$, so such a world is
solved per frequency also without inductive conductors. The `mom` family
borders its segment-level system the same way.

## The registered backends

Backends are registered in `groundfield.solver.registry` with their solve
function, declared capabilities and options model (ADR-0014). `Engine.solve`
resolves the requested name, checks the world against the capabilities
before any work is done (soil layer count, concrete shells, explicitly
requested kernel; `BackendCapabilityError` names a backend that can solve
the world), forwards `image` to `image_2layer` for a two-layer soil, and
writes the metadata core (backend, effective backend, kernel, version,
frequencies, segments, soil kind, earth model, solve time, series controls,
provenance) into every result. The table is generated from the registry:

<!-- generated:begin registry -->
| Name | Status | What it runs | Soil layers | Kernel (default first) | `return_to`, junctions, voltage sources | Concrete shells | Conductor models | Options |
|---|---|---|---|---|---|---|---|---|
| `image` | current | Homogeneous soil, closed-form image-charge sum; forwards two_layer → `image_2layer`, multi_layer → `image_nlayer` | 1, 2 | `line`, `point` | yes, yes, yes | yes | distributed, inductive, lumped, skin | — |
| `image_2layer` | current | Two-layer soil, Tagg / Sunde image series | 2 | `line`, `point` | yes, yes, yes | yes | distributed, inductive, lumped, skin | — |
| `image_nlayer` | deprecated name of `image` | Layered soil routed to image (n = 1) or image_2layer (n = 2) | 1, 2 | `line`, `point` | yes, yes, yes | yes | distributed, inductive, lumped, skin | — |
| `cim` | deprecated name of `image` | Deprecated name of the image auto-dispatch (10.76) | 1, 2 | `line`, `point` | yes, yes, yes | yes | distributed, inductive, lumped, skin | — |
| `mom` | current | Method of moments, point-matching collocation | 1, 2 | `point` | yes, yes, yes | yes | distributed, inductive, lumped, skin | — |
| `mom_sommerfeld` | current | Method of moments with Sommerfeld quadrature, any n | 1, 2, ≥ 3 | `point` | yes, yes, yes | yes | distributed, inductive, lumped, skin | `MomSommerfeldOptions` |
| `bem` | deprecated name of `mom` | Deprecated name of 'mom' (10.76) | 1, 2 | `point` | yes, yes, yes | yes | distributed, inductive, lumped, skin | — |
| `fem` | experimental | Conforming axisymmetric finite elements (rods and rings on one axis) | 1, 2, ≥ 3 | none (volume mesh) | yes, yes, yes | yes | distributed, inductive, lumped, skin | `FemOptions` |
<!-- generated:end registry -->

Backend-specific settings live in `Engine.options`, keyed by backend name
(`MomSommerfeldOptions`, `FemOptions`); the image-series controls
`image_max_terms` (default 300) and `image_series_tol` (default 1e-6) are
common `Engine` fields because every layered engine and the post-solve
field evaluation read them.

### Deprecated names

Three names of earlier releases are deprecated aliases since 2026.11; they
keep working, emit a `DeprecationWarning` naming the replacement, and will be
removed no earlier than 2027.5.

- **`cim`** ran the complex-image method: approximate $\Gamma_1(\lambda)$ by
  a sum of exponentials (matrix-pencil fit), which turns the Sommerfeld
  integral into a finite sum of complex images. For one and two layers the
  closed forms are exact and no fit was ever used on a solve path; the
  three-layer kernel was structurally incomplete and has raised since
  0.11.0. `cim` now runs the `image` auto-dispatch, which gives the numbers
  `cim` gave on every world it accepted. The fit itself,
  `groundfield.solver.cim.fit_complex_images`, remains available for
  studies of the spectral reflection coefficient; it reports convergence
  only when the maximum error on an independent grid meets the tolerance.
- **`bem`** was described as a boundary-element collocation but assembled
  the same reaction matrix as `mom` (to 4e-16) and now runs `mom`.
- **`image_nlayer`** routed a layered soil with one or two distinct layers
  to `image` / `image_2layer` and rejected three. `image` does the routing
  itself; a real image series for $n \ge 3$ (Stefanescu's construction)
  is not implemented (ADR-0002), use `mom_sommerfeld`.

Layer stacks are canonicalised before they reach a backend: zero-thickness
layers are dropped and adjacent layers of equal resistivity merged, so
`TwoLayerSoil(ρ, ρ, h)` and `MultiLayerSoil([ρ, ρ, ρ])` are solved as the
homogeneous soil they are (benchmark B9).

## Choosing an engine

| Question | Engine | Why |
|---|---|---|
| Impedance, potentials, rho-f samples in homogeneous or two-layer soil | `image` (default; forwards to `image_2layer`) | closed form, exact line kernel, cheapest |
| Three or more distinct layers | `mom_sommerfeld` | the only engine with that Green's function |
| Is the uniform current ansatz good enough here? (grids, stars, bonded clusters, rods through the interface, touch and step voltages) | `mom` next to `image` | same kernels, one current per segment |
| Is the image series right for this soil? | `mom_sommerfeld` next to `mom` | same discretisation, independent kernel |
| Quick plausibility of one compact electrode in one layer | `fem` (experimental) | volume PDE; equals Dwight's formulas in homogeneous soil |

`validation.compare_engines(world, engines)` runs a world through several
engines and compares complex cluster and input impedances at every
frequency. Two engines that run the same computation (an alias and its
target) are noted as such: their agreement is not evidence.

What a comparison checks depends on the pair. `image` against `mom` checks
the current model (same kernel), `mom` against `mom_sommerfeld` checks the
kernel (same discretisation). Neither pair checks the discretisation
itself; that is done against the analytic and independently computed
references of the benchmark suite (B1–B13, [validation](../validation.md)).

## Literature

- Sommerfeld, A. (1909). Über die Ausbreitung der Wellen in der drahtlosen
  Telegraphie. *Annalen der Physik* 28(4), 665–736.
- Stefanescu, S. S., & Schlumberger, C. (1930). Sur la distribution
  électrique potentielle autour d'une prise de terre ponctuelle dans un
  terrain à couches horizontales, homogènes et isotropes. *Journal de
  Physique et le Radium* 1.
- Tagg, G. F. (1964). *Earth Resistances*. George Newnes, London.
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission Systems*.
  Dover, New York (first edition Van Nostrand, 1949).
- Harrington, R. F. (1968). *Field Computation by Moment Methods*.
  Macmillan, New York.
- Li, Z.-X., Chen, W., Fan, J.-B., & Lu, J. (2006). A novel mathematical
  modeling of grounding system buried in multilayer earth. *IEEE
  Transactions on Power Delivery* 21(3) (quasi-static complex images, the
  method behind the deprecated `cim`).
- Hua, Y., & Sarkar, T. K. (1990). Matrix pencil method for estimating
  parameters of exponentially damped/undamped sinusoids in noise. *IEEE
  Transactions on Acoustics, Speech, and Signal Processing* 38(5) (the fit
  of `fit_complex_images`).
- IEEE Std 80-2013. *IEEE Guide for Safety in AC Substation Grounding*.

Architecture decisions: [ADR-0001](../../adr/0001-two-layer-method.md)
(two-layer method), [ADR-0002](../../adr/0002-engine-family.md) (engine
family, with its amendments), [ADR-0007](../../adr/0007-cross-layer-electrodes.md)
(cross-layer electrodes), [ADR-0014](../../adr/0014-backend-registry-and-pipeline.md)
(registry and shared pipeline), [ADR-0015](../../adr/0015-line-source-kernel-default.md)
(exact line-source kernel as default).
