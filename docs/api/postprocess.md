# Postprocess

The ``postprocess`` subpackage turns a raw :class:`FieldResult` into
quantities the user actually needs: potential / EPR plots, a fitted
``rho-f`` standard model, and a vector fit that produces a closed-form
SymPy expression compatible with
:class:`groundinsight.BusType.impedance_formula`.

## Plotting helpers

The standard plot suite (``plot_potential_contour``,
``plot_potential_profile``, ``plot_potential_radial``) is
re-exported at the package level for convenience and is documented
in [Quickstart](../quickstart.md). Below we focus on the **reduced
model** path that closes the field → network bridge.

## Vector fitting and SymPy export

### Physical background

For a passive, linear, time-invariant grounding cluster, the
driving-point impedance $Z(s)$ at the feed-in electrode is a
rational function of the Laplace variable $s = j\omega$. Under
the dissertation's $f \le 1\,\mathrm{kHz}$ assumption, the
function is well approximated by a low-order partial-fraction
expansion

$$
Z(s) \;\approx\; R_\infty \;+\; s\,L_\infty \;+\;
\sum_{k=1}^{N_p}\,\frac{r_k}{s - p_k}
$$

with poles $p_k$ on the negative real axis (damped RC modes) and
residues $r_k$. Complex-conjugate pole pairs are admissible and
produce damped resonant LC-like behaviour. The fit is constructed
by the **Vector Fitting** algorithm of Gustavsen & Semlyen 1999,
which iterates pole locations to a stable solution and is the
de-facto standard for transmission-line and grounding modelling.

The output is a SymPy expression in a single free symbol ``s``
with complex-conjugate pole pairs combined into real
second-order terms. With the symbolic substitution
$s \to j\,2\pi f$ the expression matches the
``groundinsight.BusType.impedance_formula`` parser, so a field-grade
``Z(s)`` can be turned into a network-grade ``BusType`` without any
manual reformatting.

### Example

```python
import numpy as np
import groundfield as gf
from groundfield.postprocess.vector_fitting import (
    vector_fit, fit_to_sympy, rho_f_from_field_result,
)

# 1. Run any groundfield engine and obtain Z(f) at the feed cluster.
soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
world = gf.create_world(soil=soil)
gf.create_electrode(
    world, "ring", name="g1",
    center=(0.0, 0.0, 0.8), radius=5.0, wire_radius=0.005,
)
gf.create_source(world, attached_to="g1", magnitude=1.0)

frequencies = np.array([10.0, 50.0, 150.0, 250.0, 500.0, 1000.0])
engine = gf.create_engine(backend="image", frequencies=frequencies.tolist())
result = world.solve(engine)

# 2. Fit a rational Z(s) directly from the FieldResult.
fit = rho_f_from_field_result(result, electrode_name="g1", n_poles=3)

# 3. Render as a SymPy expression and extract the formula string
#    that groundinsight.BusType.impedance_formula consumes.
expr = fit_to_sympy(fit, decimals=6)
print(expr)            # printable rational expression
formula_str = str(expr)
```

A pre-tabulated $Z(f)$ array can be fitted directly with
:func:`vector_fit(frequencies, Z_values, n_poles=3)`. The returned
:class:`VectorFitResult` carries poles, residues, and an
``evaluate(frequencies)`` method for re-evaluation.

For the export of the fit as a ``BusType`` (JSON file or live
``groundinsight`` instance) see the [IO reference](io.md) and
[ADR-0008](../adr/0008-groundinsight-bridge.md).

### API reference — vector fitting

::: groundfield.postprocess.vector_fitting

## Standard rho-f form

The dissertation's five-coefficient ansatz

$$
Z(\rho, f) \;=\; k_1 \rho \;+\; (k_2 + j k_3)\,f \;+\; (k_4 + j k_5)\,f\,\rho
$$

is fitted across a $\rho$-sweep and is already in the canonical
``(rho, f)`` symbol set used by
``groundinsight.BusType.impedance_formula``. See
:mod:`groundfield.postprocess.rho_f_standard` and the
[IO reference](io.md) for the export path.

::: groundfield.postprocess.rho_f_standard

## Touch and step voltages

The :mod:`groundfield.postprocess.safety` module turns a
:class:`FieldResult` into the engineering safety quantities used
in EN 50522:2010 / IEC 61936-1.

### Physical background

The earth potential rise of a touched grounding cluster is

$$
U_E \;=\; \varphi_\text{cluster}.
$$

Standing on the soil surface 1 m away from the touched part, the
person's feet sit at the surface potential
$\varphi(\mathbf r_\text{feet})$ and the voltage appearing across
the body is

$$
U_T \;=\; U_E - \varphi(\mathbf r_\text{feet}).
$$

The *step* voltage between two surface points at the typical
$d_\text{step} = 1\,\mathrm{m}$ separation is

$$
U_S \;=\; \varphi(\mathbf r_1) - \varphi(\mathbf r_1 + d_\text{step}\,\hat{\mathbf e}).
$$

Both quantities are returned as **complex phasors per frequency
index**, so that inductive- and resistive-coupling effects above
DC remain visible in AP1 studies.

### Example

```python
import groundfield as gf

soil = gf.HomogeneousSoil(resistivity=100.0)
world = gf.create_world(soil=soil)
gf.create_electrode(world, "rod", name="g1",
                    position=(0.0, 0.0, 0.5), length=1.5)
gf.create_source(world, attached_to="g1", magnitude=10.0)

result = gf.create_engine(backend="image", segment_length=0.05).solve(world)

# Touch voltage 1 m east of the rod:
U_T = gf.touch_voltage(result, world, electrode="g1", distance=1.0)

# Worst-case touch voltage on a circle around the rod:
angles, voltages = gf.touch_voltage_envelope(
    result, world, electrode="g1", distance=1.0, n_angles=24,
)
U_T_worst = float(abs(voltages).max())

# Step voltage 1 m east of the rod, walking radially outward:
U_S = gf.step_voltage(result, position=(1.0, 0.0, 0.0),
                      direction=(1.0, 0.0, 0.0), step=1.0)

# Reference: EN 50522 Table B.4 permissible touch voltage at t_F = 0.5 s.
U_TP = gf.permissible_touch_voltage_en50522(0.5)   # 225 V
```

### Validity envelope

* Frequency: dissertation envelope $f \le 1\,\mathrm{kHz}$,
  inherited from the underlying Green's function.
* Backends: every solver that populates ``point_sources`` (image,
  image_2layer, image_nlayer, mom, mom_sommerfeld, cim, bem).
  Stub backends raise inside :meth:`FieldResult.potential`.
* Coordinate convention: soil surface at $z = 0$, positive $z$
  points downwards into the soil. The default ``surface_z = 0.0``
  models bare-foot contact on the ground surface.
* `permissible_touch_voltage_en50522` reproduces EN 50522:2010
  **Table B.4** verbatim — eight anchors at $t_f \in \{0.05,
  0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00\}\,\mathrm{s}$ with
  $U_{TP} \in \{725, 655, 525, 225, 115, 95, 85, 85\}\,\mathrm{V}$,
  values rounded to 5 V in the standard. Log-log interpolation
  inside the grid; outside the grid the values are clamped to
  the table endpoints (no relaxation below 50 ms; the terminal
  85 V plateau between 5 s and 10 s is reproduced exactly).

### API reference — safety

::: groundfield.postprocess.safety

## Current sharing and split factor

The :mod:`groundfield.postprocess.current_balance` module turns
the per-electrode currents in :class:`FieldResult` into the
engineering quantities that answer the AP1 question *"where does
the injected source current actually return?"*.

### Physical background

For each galvanic cluster $c$ the **net soil leakage** is

$$
I_c \;=\; \sum_{e \in c} I_e,
$$

with $I_e$ the per-electrode soil-leakage current stored in
:attr:`FieldResult.electrode_currents` (positive in direction
*electrode → soil*). Because the cluster members share a
potential $U_c$, the **cluster impedance** is
$Z_c = U_c / I_c$, and the per-electrode share inside one cluster
is the complex ratio $s_{e \mid c} = I_e / I_c$.

The **split factor** of a current source is

$$
s \;=\; \frac{\sum_{e \in c_\text{src}} I_e}{I_\text{src}},
$$

with $c_\text{src}$ the cluster of the source's ``attached_to``
electrode. By construction, $s = 1 + 0\,j$ when the entire
injected current leaves the source cluster through the soil
(no metallic parallel path), and $s < 1$ when a metallic
conductor (PEN trunk, parallel measurement lead, cable shield)
carries part of the current as a parallel resistive path. This
is the **galvanic** current division across parallel paths.

#### Not the same as the *Reduktionsfaktor*

In the German EVU / Schirmtechnik literature (Oeding & Oswald
2016) the *Reduktionsfaktor* refers to the additional
**transformatorische / inductive coupling correction** between a
current-carrying conductor and a parallel grounding / shield
conductor. That quantity is angle-dependent: it vanishes when the
two conductors are perpendicular (no flux linkage) but is large
for collinear runs.

The split factor here is **purely galvanic** — present whenever
there are parallel resistive paths, irrespective of the
geometric angle between conductors. The proper Reduktionsfaktor
is on the roadmap; the inductance backends in
:mod:`groundfield.coupling` (Neumann, Carson, Sommerfeld) are
already in place and will be picked up by a dedicated future
helper.

### Example

```python
import groundfield as gf

soil = gf.HomogeneousSoil(resistivity=100.0)
world = gf.create_world(soil=soil)
g1 = gf.create_electrode(
    world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=1.5,
)
g_aux = gf.create_electrode(
    world, "rod", name="g_aux", position=(50.0, 0.0, 0.5), length=1.5,
)
# Finite-impedance metallic feed lead between source and aux cluster.
gf.create_conductor(
    world, name="feed_lead", start=g1, end=g_aux,
    conductor_type="bare_copper", cross_section=50e-6,
)
gf.create_source(
    world, name="src", attached_to="g1", return_to="g_aux", magnitude=10.0,
)
result = gf.create_engine(backend="image", segment_length=0.05).solve(world)

# Per-cluster summary, sorted by descending |ΣI|:
gf.cluster_current_balance(result)
# Per-electrode table (with kind / depth annotations):
gf.electrode_current_table(result, world=world)
# Split factor of the source — < 1 because the metallic feed lead
# carries part of I_src as a parallel resistive path:
s = gf.split_factor(result, world)
# Top-15 bar chart of |I| per electrode:
gf.plot_current_sharing(result, world=world, top_n=15)
```

### Validity envelope

* Frequency: dissertation envelope $f \le 1\,\mathrm{kHz}$.
* Backends: every solver that populates
  ``electrode_potentials`` / ``electrode_currents`` / ``clusters``
  (image, image_2layer, image_nlayer, mom, mom_sommerfeld, cim,
  bem). Stub backends produce empty / NaN cluster tables.
* The image backend treats ``Source.return_to`` as informational
  metadata; the injected current dissipates via the Dirichlet
  far-field boundary. The split factor still detects metallic
  parallel paths because the cluster's net soil leakage is
  reduced by whatever current is shunted into a finite-impedance
  branch.

### API reference — current sharing

::: groundfield.postprocess.current_balance

## Parameter sweeps and convergence studies

The :mod:`groundfield.postprocess.sweep` and
:mod:`groundfield.postprocess.convergence` modules turn the AP1
parameter axes into a single tabular response. Both produce
long-format :class:`pandas.DataFrame` objects that feed naturally
into the :math:`\rho`-:math:`f` fit and the vector-fitting
pipelines.

### `sweep` — Cartesian product over named axes

```python
import groundfield as gf

def world_factory(*, rho, h_1):
    soil = gf.TwoLayerSoil(rho_1=rho, rho_2=10*rho, h_1=h_1)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.5), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    return world

eng = gf.create_engine(backend="image_2layer", segment_length=0.1,
                       frequencies=[50.0, 200.0])
df = gf.sweep(
    world_factory,
    eng,
    axes={"rho": [50, 100, 500, 1000], "h_1": [1.0, 2.0, 5.0]},
)
gf.plot_sweep_lines(df, x="rho", y="abs_Z", color="h_1",
                    log_x=True, log_y=True)
gf.plot_sweep_heatmap(df, x="rho", y="h_1", response="abs_Z",
                      frequency_Hz=50.0)
```

The default response captures the cluster impedance and EPR at
the source's cluster per frequency (``Z_re``, ``Z_im``,
``abs_Z``, ``arg_Z_deg``, ``U_E_re``, ``U_E_im``, ``abs_U_E``,
``I_re``, ``I_im``, ``abs_I``). Pass ``response=...`` to extract
custom scalars.

### `convergence_study` — refinement over `segment_length`

```python
df = gf.convergence_study(
    world, eng, segment_lengths=[0.5, 0.2, 0.1, 0.05, 0.02],
)
gf.plot_convergence(df, response="abs_Z", reference=R_sunde)
```

The helper clones the engine via :meth:`Engine.model_copy`, so
the original engine is **not** mutated. The plot's x-axis is
inverted so finer ``segment_length`` sits on the right (the
convergence-asymptote direction). Pass an analytical reference
(Sunde, Dwight, IEEE 80) via ``reference=...`` for a clean
asymptote line.

### API reference — sweep

::: groundfield.postprocess.sweep

### API reference — convergence

::: groundfield.postprocess.convergence

## World-geometry plots (no solve required)

The :mod:`groundfield.postprocess.geometry_plot` module renders
the *physical* world — electrodes, conductors and current sources
— **before** the solver runs. AP1-grade networks with several
hundred electrodes benefit from a quick sanity check that catches
typos in coordinates, missing conductors, or sources attached to
the wrong electrode without paying for a full solve.

### Conductor colour scheme

The conductor colour follows
:data:`groundfield.conductors.ConductorType`:

| ``conductor_type`` | Colour                  |
|--------------------|-------------------------|
| ``pen``            | green (`#2c7a2c`)       |
| ``bare_copper``    | orange (`#d97300`)      |
| ``cable_shield``   | grey (`#888888`)        |
| ``overhead``       | steel blue (`#1f77b4`)  |
| ``generic``        | dark grey (`#444444`)   |

The line **style** flags the soil-coupling mode:
``coupling_to_soil="galvanic"`` is solid, ``"isolated"`` is
dashed.

### Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
g_rod = gf.create_electrode(
    world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=1.5,
)
g_ring = gf.create_electrode(
    world, "ring", name="g2", center=(10.0, 0.0, 0.8), radius=2.5,
)
gf.create_conductor(world, name="bond", start=g_rod, end=g_ring,
                    conductor_type="bare_copper")
gf.create_source(world, name="src", attached_to=g_rod, magnitude=10.0,
                 return_to=g_ring)

# Top-down 2-D view (default).
gf.plot_world(world, plane="xy")
# Vertical slice with the soil surface as a dotted grey line.
gf.plot_world(world, plane="xz")
# 3-D wireframe with inverted z (depth grows downwards on screen).
gf.plot_world_3d(world)

# Pure helper — bounding box of the geometry, useful for custom
# extents on the field plots.
x_min, x_max, y_min, y_max, z_min, z_max = gf.world_bounds_3d(world)
```

### API reference — geometry plots

::: groundfield.postprocess.geometry_plot

## Plotting API reference

::: groundfield.postprocess.plotting
