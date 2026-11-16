# Engines

This page helps you choose the numerical engine (the *backend*) for a study
and tells you where each engine must not be used. Start with the decision
path; the tables below it give the accuracy and the cost behind each choice,
cited from the [validation page](../validation.md), the engine pages, the
[performance page](../performance.md) and the
[known limitations](../../reference/known-limitations.md). The second half
of the page explains the physics the engines share and where they differ.

All engines solve one physical problem: the quasi-static current field of
electrodes and conductors in horizontally layered soil, below 1 kHz. They
share the data model (`World`), the solve pipeline and the network
equations, so the same world can be solved by any of them and the results
compared. They differ in the soils they accept, in how they represent the
current on the wires, in accuracy and in cost.

## Choosing an engine

### Decision path

```text
How many distinct soil layers?
│
├── three or more ─────────────────────────────────────────▶ mom_sommerfeld
│                                                             (impedances only:
│                                                              no field quantities)
└── one or two
    │
    ├── touch, step or mesh voltages within about 5 m of a
    │   grid, a star, a strip end or a bonded group of
    │   electrodes; the current distribution along wires ─────▶ mom
    │
    └── everything else: impedances, networks, frequency
        sweeps, rho-f samples, potentials away from the wires ─▶ image (default;
                                                                 two layers run
                                                                 image_2layer)

Check a result once with a second method:
    image  next to  mom              same kernels, other current model
    mom    next to  mom_sommerfeld   same discretisation, other Green's function
    fem                              independent reference for one cluster
```

`image` is the right engine for most work: it is exact for its model, the
cheapest for large networks, and its model error is about 1 % on
impedances. That model, one uniform current per straight wire, misplaces
the current along a wire, so surface potentials next to the ends and
corners of extended electrodes need `mom`. Three or more layers need the
numerical Green's function of `mom_sommerfeld`. `fem` shares nothing with
the integral engines but the network solve; it is the independent check of
a single earthing system, not a tool for networks.

| Question | Engine | Why |
|---|---|---|
| Impedance, potentials, rho-f samples in homogeneous or two-layer soil | `image` (default; forwards to `image_2layer`) | closed form, exact line kernel, cheapest for large networks |
| Touch, step and mesh voltages near grids, stars, strip ends, bonded clusters | `mom` | one current per segment; the image family errs by up to tens of per cent there |
| Three or more distinct layers | `mom_sommerfeld` | the only integral engine with that Green's function |
| Is the uniform current ansatz good enough here? | `mom` next to `image` | same kernels, one current per segment |
| Is the image series right for this soil? | `mom_sommerfeld` next to `mom` | same discretisation, independent kernel |
| A reference that shares no kernel with the integral engines | `fem` (experimental) | volume PDE; about 1e-4 for coaxial rods and rings, about 2e-3 in 3-D |

### The engines side by side

| | [`image`](image.md), [`image_2layer`](image_2layer.md) | [`mom`](mom.md) | [`mom_sommerfeld`](mom_sommerfeld.md) | [`fem`](fem.md), axisymmetric | [`fem`](fem.md#the-three-dimensional-model), 3-D |
|---|---|---|---|---|---|
| Soil layers | 1 or 2 | 1 or 2 | any number | any number | any number; contrast up to about 300 |
| Electrodes | thin wires anywhere in the soil | thin wires anywhere in the soil | thin wires; with 3 or more layers in the top layer only | rods on one vertical axis, rings centred on it | any thin wires; at most 100 ports |
| Current model | one uniform current per straight wire (and soil layer) | one current per segment | one current per segment | solid electrodes; the current follows from the field | smoothed line sources; the current follows from the field |
| Green's function | closed form: air image, Tagg/Sunde series | that of the image family | Sommerfeld integral by quadrature | none (finite elements) | none (finite elements) |
| Source kernel | exact line source (default) or point | point | point | — | — |
| Impedance error | about 1 % (the ansatz); stars more | 0.6 … 1.9 % low | as `mom` | about 1e-4 | about 2e-3 |
| Touch and step voltages | impedance-grade near grids and stars (warns) | the reference | as `mom`; none for 3 or more layers | 1 or 2 layers | 1 or 2 layers |
| Status | default | current | current | experimental | experimental; `fem3d` extra |

The current models and Green's functions are explained
[below](#mathematical-and-physical-context); the numbers of the two error
rows are in the next section.

### Accuracy against references

Every engine is compared with references that do not use a groundfield
solver (closed forms, an independent thin-wire solver, high-precision
quadrature, the solid electrodes of `fem`) in the benchmark suite B1–B13 of
the [validation page](../validation.md). Typical deviations, signed where
the sign is known:

| Case and reference | `image`, line kernel | `mom` | `fem` |
|---|---|---|---|
| Rod 3 m, 100 Ω·m; thin-wire solution 33.1957 Ω (B2) | +0.92 % | −0.63 … −1.4 % (ds 0.5 … 0.125 m) | 7.3e-5 (solid rod) |
| Ring r = 2 m at 0.8 m; closed form 12.4816 Ω (B3) | −0.24 % (ds 0.5 m), −1.5e-4 (ds 0.125 m) | −1.2 % | 5.7e-5 (solid torus) |
| Ring with three bonded 4 m rods; 3-D `fem` (B10), homogeneous / two-layer | +0.81 % / +1.97 % (point kernel) | −0.6 % / −0.8 % | reference (estimate 3.9e-3 / 4.3e-3) |
| Rods and rings in the top layer of two-layer soil; `mom` (B6) | within 0.37 % (point kernel) | reference | +0.57 … +1.3 % |
| Grids 10 m 2 × 2 … 40 m 8 × 8; one current per segment | +0.36 … +0.70 % | — | — |
| Stars of 3 / 4 / 8 arms of 10 m; one current per segment | +1.8 / +3.5 / +10.5 % | — | — |
| Touch / step voltage at a grid corner, 20 m 4 × 4; `mom` | +6.4 % / −11 % | reference | — |
| Touch / step voltage at a star tip, 4 arms of 10 m; `mom` | +10 % / −23 % | reference | — |

How to read it:

- **`image` is exact for its own model.** For a straight wire the line
  kernel reproduces the closed-form resistance of a uniform line current to
  about 1e-6 at any segment length; a ring converges with the square of the
  segment length ([kernel of the image family](../validation.md#kernel-of-the-image-family)).
  What remains is the error of the ansatz: about 1 % on impedances of rods,
  rings, grids and rectangles, more on stars, and up to tens of per cent on
  surface potentials next to grid corners and star tips — with the step
  voltage *under*-estimated, the non-conservative direction
  ([field quantities](../validation.md#field-quantities-near-the-electrode)).
- **`mom` resolves the current distribution but not the absolute value.**
  Its point-source near field leaves impedances 0.6 … 1.9 % low and drifts
  as the segments get shorter. Its strength is the *shape* of the current
  and of the surface potential, which is why it is the reference for field
  quantities and for the current model of the image family.
- **`mom_sommerfeld` checks the kernel.** It reproduces the Tagg/Sunde
  series of `mom` to 1.2e-8 on the two-layer worlds of B6 and the
  three-layer kernel to 2.3e-13 of a 30-digit quadrature (B5); its tables
  for three layers move a grid's impedance by 1.5e-10 (B12). Its
  discretisation error is that of `mom`.
- **`fem` is the independent reference.** The axisymmetric model agrees
  with Weber's disc to 3.1e-4 and with the thin-wire rod and ring to
  better than 1e-4 (B1–B3); the 3-D model agrees with the axisymmetric one
  to about 2e-3 away from layer interfaces and other wires, and reports an
  error estimate of 3 … 5e-3.

### Cost

Wall-clock time of one `world.solve` at 50 Hz with 0.5 m segments and the
default options, after a warm-up solve; peak memory of the process where
it matters. Measured for this page with the code of this release on a
shared Apple-silicon machine (macOS arm64, numba installed, one BLAS thread,
load 8 … 15); the times vary with the machine's load, the ratios between
the engines less so. The networks are those of the how-to
[speed up large studies](../../how-to/speed-up-large-studies.md): a
station ring and *n* houses with 10 m × 10 m mesh foundations on a PEN
chain.

| World | Segments | `image` | `mom` | `mom_sommerfeld` | `fem` |
|---|---:|---|---|---|---|
| rod 3 m, 100 Ω·m | 6 | 1 ms | < 1 ms | < 1 ms | 0.64 s (axisymmetric) |
| ring r = 2 m, 100 Ω·m | 26 | 3 ms | < 1 ms | < 1 ms | 0.23 s (axisymmetric) |
| grid 20 m 4 × 4, 100 Ω·m | 400 | 0.03 s | 5 ms | 4 ms | — |
| the same grid, 300 / 100 Ω·m, $h_1$ = 2 m | 400 | 0.08 s | 0.03 s | 0.25 s | — |
| the same grid, 300 / 100 / 1000 Ω·m | 400 | — | — | 0.05 s | — |
| 30 houses, 100 Ω·m | 2426 | 0.19 s | 0.17 s | — | — |
| 30 houses, 300 / 100 Ω·m, $h_1$ = 2 m | 2426 | 0.71 s | 0.47 s | — | — |
| 80 houses, 100 Ω·m | 6426 | 1.5 s, 0.6 GiB | 2.0 s, 1.7 GiB | — | — |
| 120 houses, 100 Ω·m | 9626 | 2.6 s, 1.0 GiB | 5.8 s, 3.6 GiB | — | — |
| 30 houses, inductive PEN, 11 frequencies | 2426 | 4.9 s | 9.5 s | — | — |

What the table and the [performance page](../performance.md) show:

- **Up to a few thousand segments `image` and `mom` take a second or
  less.** The line kernel of the image family integrates near pairs
  exactly, which costs about what the dense LU of `mom` costs.
- **`mom` grows faster.** Its LU is $O(N^3)$ and it holds more of the
  dense system in memory (3.6 GiB against 1.0 GiB at 9 626 segments); with
  inductive conductors it factorises once per frequency (twice the time of
  `image` for 11 frequencies above).
- **`image_2layer` pays for contrast and thin top layers.** The image
  series needs 20 terms at $\lvert K\rvert$ = 0.5 and 277 at 0.94; a
  40-house network takes 3.2 s in 500 / 100 Ω·m with $h_1$ = 5 m and
  4.0 s in 1000 / 30 Ω·m with $h_1$ = 1.5 m, where
  `Engine.image_far_field = "point"` brings it to 2.0 s for catalogue
  sweeps.
- **`mom_sommerfeld` integrates every distinct pair.** In two-layer soil
  it costs several times `mom` (0.25 s against 0.03 s above); for three
  layers its depth-group tables make it fast (a 40 m grid at 1.25 m
  segments in 0.30 s instead of 66.6 s).
- **`fem` is a reference, not a production engine.** The axisymmetric
  model takes a fraction of a second for a rod or a ring and about 17 s for
  eight rings with a central rod (a million nodes). The 3-D model takes
  26 s and 1.8 GB for a ring of 12.6 m, 75 … 140 s and 4.3 GB for the AP1
  station (28 m of wire), and 2 … 3.5 minutes and up to 3.8 GB for the ring
  with three rods of B10.
- **Inductive conductors** add the earth-return assembly, the same for
  every engine and the dominant cost of inductive studies (12.2 s for a
  generated 30-house TN network with a Neumann PEN at four frequencies).

Every engine stops before allocating anything when a world has more than
`Engine.max_segments` segments (default 20 000, `ResourceLimitError`) and
notes from 5 000 segments on that the solve may take minutes.

### Where each engine must not be used

Each row names what goes wrong, how large the error is, what the code
tells you and what to use instead. The entries of the
[known limitations](../../reference/known-limitations.md) give the
measurements; [warnings and errors](../../reference/warnings.md) describes
every class. Three limits hold for every engine: the model is
quasi-static and validated below 1 kHz, the soil is linear and
horizontally layered, and potentials in the soil (touch and step voltages,
plots, field exports) are available for one and two layers only
(`NotImplementedError` for three or more).

**`image` and `image_2layer`**
([current ansatz](../../reference/known-limitations.md#current-ansatz-and-source-kernels),
[soil layers](../../reference/known-limitations.md#soil-layers)):

| Do not use it for | Error | Signal | Instead |
|---|---|---|---|
| touch, step and mesh voltages within 5 m of grids, stars, strip ends, open polylines, bonded clusters | grid corner +6.4 % / −11 %, star tip +10 % / −23 % (touch / step) | `UniformCurrentFieldWarning` | `mom` |
| impedances of stars with many arms | +3.5 % (4 arms), +10.5 % (8 arms) | none | `mom` |
| three or more distinct layers | not supported | `BackendCapabilityError` | `mom_sommerfeld` |
| a layer contrast with $\lvert K\rvert$ > 0.94 (2000 / 30 Ω·m) | series truncated at 300 terms | `SeriesTruncationWarning` | raise `image_max_terms` |
| rod ends within one wire radius of a layer interface | +22 % with the tip at the interface | `InterfaceContactWarning` | axisymmetric `fem` (solid rod) |
| horizontal wires within a few wire radii of an interface | +3.6 % at 2 radii over a layer 33 times more conductive | `InterfaceContactWarning` | axisymmetric `fem` for rings |

**`mom`**
([point kernel](../../reference/known-limitations.md#current-ansatz-and-source-kernels)):

| Do not use it for | Error | Signal | Instead |
|---|---|---|---|
| absolute impedances at the 1 % level | 0.6 … 1.9 % low, drifting with the segment length | none | `image` where the ansatz holds, `fem`, or a convergence study |
| horizontal wires within 0.6 segment lengths of a layer interface | a ring 5 cm below the interface −46 % at 0.5 m segments | `PointKernelInterfaceWarning`; within 0.387: `BackendCapabilityError` | shorter segments |
| segments longer than four times their depth | point self image biased high | `ShallowSegmentWarning` | shorter segments |
| three or more distinct layers | not supported | `BackendCapabilityError` | `mom_sommerfeld` |
| very large networks | 3.6 GiB at 9 626 segments, growing like $N^2$ | `ResourceLimitError` above `max_segments` | `image` |

Rod ends and horizontal wires at a layer interface have the thin-wire
limits of the image family (`InterfaceContactWarning`).

**`mom_sommerfeld`**
([soil layers](../../reference/known-limitations.md#soil-layers)):

| Do not use it for | Error | Signal | Instead |
|---|---|---|---|
| touch and step voltages, potential plots with three or more layers | not implemented | `NotImplementedError` | — (impedances only) |
| electrodes below the top layer with three or more layers | not supported | `ValueError` | — |
| two layers of extreme contrast or a top layer thinner than 0.2 m | the quadrature may not converge | `SommerfeldConvergenceWarning`, `metadata["converged"]` | `image_2layer` |
| segment midpoints within about 1 mm of an interface | interface term saturated | none | — |

The point-kernel limits of `mom` apply as well.

**`fem`**
([backends](../../reference/known-limitations.md#backends)):

| Do not use it for | Error | Signal | Instead |
|---|---|---|---|
| networks and many clusters | more than 100 ports refused; the AP1 station takes 1.5 … 3 minutes and 4.1 … 4.3 GB | `ResourceLimitError` | `image`, `mom` |
| any world but coaxial rods and rings without the extra | — | `ImportError` naming `fem3d` | install `fem3d` |
| 3-D model: long wires, such as the generators' 60 m house mesh | refused at the defaults (more than 2 million nodes); with `source_radius=0.1`, `growth=0.6` 1.5 minutes and 4.6 GB | `ResourceLimitError` | raise the options; integral engines |
| 3-D model: a layer contrast of 1000 | the conjugate gradients do not converge | `BackendCapabilityError` | axisymmetric model, integral engines |
| 3-D model: wires beside each other closer than $1.5\,R$, end to end closer than $5\,R$ ($R$ = `source_radius`, 6 cm) | a smaller $R$ and more nodes; refused below 3.3 wire radii | `BackendCapabilityError` | integral engines |
| 3-D model: differences below about 2e-3 | not resolved | `metadata["fem_error_estimate"]` (3 … 5e-3) | axisymmetric model, integral engines |
| results without a second method | experimental | `ExperimentalBackendWarning` | compare with an integral engine |

### Check your choice

Before a long run, `gf.diagnose(world, engine)` collects the checks that
need no solve: resolution, segment budget, thin wires, close and
interpenetrating wires, rod ends and wires at a layer interface, mixed
coordinates and overlapping cables
([read and act on warnings](../../how-to/warnings-and-diagnostics.md)).
After it, `gf.compare_engines` runs the world through a second engine and
`gf.convergence_study` refines the segment length:

```python
import groundfield as gf

world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=2.0))
gf.create_electrode(world, "grid_mesh", name="grid", corner=(-10.0, -10.0, 0.5), size=(20.0, 20.0), n_x=4, n_y=4)
gf.create_source(world, attached_to="grid", magnitude=1.0)

print(gf.diagnose(world, gf.create_engine()) or "no findings")
report = gf.compare_engines(world, engines={"image": gf.create_engine(), "mom": gf.create_engine(backend="mom")})
print(f"image against mom: {report.deviations['grid']:.2%}")   # 0.92%
study = gf.convergence_study(world, gf.create_engine(), segment_lengths=[1.0, 0.5, 0.25])
print(study[["segment_length_m", "abs_Z"]])                     # 4.8005 Ω at every length
```

<!-- doctest-check
assert not list(gf.diagnose(world, gf.create_engine()))
assert report.is_consistent and 0.005 < report.deviations["grid"] < 0.015
assert abs(study["abs_Z"].max() / study["abs_Z"].min() - 1.0) < 1e-5
-->

The grid's wires are straight, so the line kernel of `image` gives the same
impedance at every segment length; `mom` reads 0.9 % lower, the size of its
point-kernel bias. Which pair checks what:

- `image` against `mom` checks the current model (the same kernels).
- `mom` against `mom_sommerfeld` checks the Green's function (the same
  discretisation).
- Neither pair checks the discretisation itself; that is what the
  convergence study and the references of the benchmark suite do.
- Two names that run the same computation (a deprecated alias and its
  target) are reported as such by `compare_engines`: their agreement is not
  evidence.

## Mathematical and physical context

Below 1 kHz the soil is a conductor whose displacement current is
negligible ($\omega\varepsilon\rho \ll 1$) and whose skin depth
$\delta = \sqrt{2\rho/(\omega\mu_0)}$ — about 500 m for 100 Ω·m at 1 kHz,
more at lower frequencies and higher resistivities — is large against the
extent of the electrodes. The galvanic part of the problem is therefore
**stationary current flow** at each frequency. With the conductivity
$\sigma = 1/\rho$ piecewise constant in depth and a current density $q$
injected by the electrodes, the potential $\varphi$ satisfies

$$
-\nabla\cdot\bigl(\sigma(z)\,\nabla\varphi\bigr) = q
\quad\text{in the soil } (z > 0),\qquad
\frac{\partial\varphi}{\partial z}\Big|_{z=0} = 0,\qquad
\varphi \to 0 \text{ for } |\mathbf r|\to\infty .
$$

In words: current is conserved in the soil, no current crosses the soil
surface into the air, and the potential vanishes at remote earth.
$\varphi$ and the normal current density $\sigma\,\partial\varphi/\partial z$
are continuous at every layer interface ($z$ points into the soil).
Frequency enters through the conductors only: their longitudinal impedance
(resistance, skin effect, partial inductances and the earth-return
correction, see [earth return](../earth-return.md)) makes the network
equations complex and frequency dependent, while the soil kernel stays
real. A galvanic world is therefore solved once and reused for every
frequency.

The Green's function of this problem is the potential of a unit point
current at depth $z_s$, observed at horizontal distance $s$ and depth $z$
in the top layer. It is the Sommerfeld integral

$$
G(s, z; z_s) = \frac{\rho_1}{4\pi}\int_0^\infty
\frac{e^{-\lambda|z-z_s|} + e^{-\lambda(z+z_s)}
 + \Gamma_1(\lambda)\bigl[e^{-\lambda(2h_1-|z-z_s|)} + e^{-\lambda(2h_1-z-z_s)}\bigr]}
{1-\Gamma_1(\lambda)\,e^{-2\lambda h_1}}\;J_0(\lambda s)\,\mathrm d\lambda ,
$$

a superposition of cylindrical waves $J_0(\lambda s)$ over the horizontal
wavenumber $\lambda$. Here $\rho_1$ and $h_1$ are the resistivity and the
thickness of the top layer, and $\Gamma_1(\lambda)$ is the reflection
coefficient of the stack below the first interface, built bottom-up from
the interface coefficients $K_i = (\rho_{i+1}-\rho_i)/(\rho_{i+1}+\rho_i)$.
For one layer $\Gamma_1 = 0$ and $G$ is the point source plus its air
image; for two layers $\Gamma_1 = K_1$ is constant and the denominator
expands into the Tagg / Sunde image series; for three and more layers no
closed form exists and the integral is evaluated numerically. The engines
differ in how they evaluate $G$ and in how they discretise the electrodes.

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

`fem` is not a third current model of the same problem: it solves the
field in the soil by finite elements, the electrodes as the solid bodies
they are where the world is axisymmetric and as smoothed line sources in
three dimensions otherwise (experimental, see [`fem`](fem.md)).

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
matrix; $\mathbf I_e$ are the leakage currents, $\boldsymbol\varphi_n$ the
node potentials, $\mathbf I_b$ the branch currents and $\mathbf I_\text{in}$
the injected source currents. The first row says that the soil turns
leakage currents into potentials, the second that current is conserved at
every node, the third that each branch obeys Ohm's law. Sources inject $+I$
at `attached_to` and, if set, $-I$ at `return_to` (two-terminal sources,
see [sources and measurement](../sources-and-measurement.md)). A node
without a source floats: it takes the potential the field imposes with
zero net current, which keeps transfer impedances reciprocal.

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
| `fem` | experimental | Conforming finite elements: axisymmetric (rods and rings on one axis) or three-dimensional (any electrodes; fem3d extra) | 1, 2, ≥ 3 | none (volume mesh) | yes, yes, yes | yes | distributed, inductive, lumped, skin | `FemOptions` |
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
(exact line-source kernel as default), [ADR-0016](../../adr/0016-conforming-fem.md)
(conforming finite elements).
