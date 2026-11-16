# Quickstart

This page takes you from an empty Python session to the numbers of a
typical earthing study: the earthing impedance of a rod, a ring and a grid
in homogeneous and in two-layer soil, the earth potential rise and the
touch voltage of a fault, a potential profile with a plot, and the
impedances of two earthing systems joined by a PEN conductor. Every code
block runs as it stands, in the order of the page (later blocks reuse
earlier objects); the comments show what it prints. Install the package
first ([installation](installation.md)).

## 1. A rod in homogeneous soil

A driven rod of 3 m in soil of 100 Ω·m, fed with 1 A:

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))                       # 100 Ω·m
gf.create_electrode(world, "rod", name="rod", position=(0.0, 0.0, 0.0), length=3.0)
gf.create_source(world, attached_to="rod", magnitude=1.0)                         # 1 A

result = world.solve(gf.create_engine(frequencies=[50.0]))
print(f"{result.input_impedance()[0].real:.2f} Ω")    # 35.99 Ω
```

<!-- doctest-check
assert abs(result.input_impedance()[0] - 35.9905) < 5e-4
-->

What each line does:

- **Soil.** `HomogeneousSoil(rho=100.0)` is a half-space of resistivity
  $\rho = 100\,\Omega\,\mathrm{m}$ below an insulating air space. All inputs
  are SI units (m, Ω·m, A, Hz).
- **World.** The `World` holds the physical model: soil, electrodes,
  conductors and sources.
- **Electrode.** Coordinates are $(x, y, z)$ in metres with $z$ the depth,
  positive downwards. A rod's `position` is its upper end; this one runs
  from the surface to 3 m. The wire radius defaults to 5 mm.
- **Source.** 1 A RMS injected at the rod. Without `return_to` the current
  returns through remote earth, the potential reference ($\varphi \to 0$
  far away).
- **Engine and solve.** The `Engine` holds the numerics. The default
  backend `image` solves the problem in closed form; the result holds one
  value per frequency, here one. With 1 A the impedance in ohms is also
  the earth potential rise in volts.

Dwight's formula for a rod, $R = \frac{\rho}{2\pi L}\bigl(\ln\frac{4L}{a} - 1\bigr)$
with length $L$ and wire radius $a$, is a quick plausibility check:

```python
from groundfield.references import dwight1936

print(f"{dwight1936.rod(rho=100.0, length=3.0, radius=0.005):.2f} Ω")   # 35.99 Ω
```

<!-- doctest-check
assert abs(dwight1936.rod(rho=100.0, length=3.0, radius=0.005) / result.input_impedance()[0].real - 1.0) < 3e-4
-->

The two agree to about 0.01 %: both assume a uniform current along the
rod. A real rod leaks more current near its ends, and its resistance lies
about 1 % lower; [choosing an engine](../explanation/engines/index.md)
explains when that matters.

## 2. A rod, a ring and a grid in two soils

A small helper builds one earthing system per call, so that the same
electrode can be solved in different soils:

```python
def earthing_system(kind, soil, current=1.0):
    """A world with one earth electrode named "earth", fed with `current` in A."""
    world = gf.create_world(soil=soil)
    if kind == "rod":
        gf.create_electrode(world, "rod", name="earth", position=(0.0, 0.0, 0.0), length=3.0)
    elif kind == "ring":
        gf.create_electrode(world, "ring", name="earth", center=(0.0, 0.0, 0.8), radius=3.0)
    else:  # a grid of 20 m × 20 m with 4 × 4 meshes, 0.5 m deep
        gf.create_electrode(
            world, "grid_mesh", name="earth", corner=(-10.0, -10.0, 0.5), size=(20.0, 20.0), n_x=4, n_y=4
        )
    gf.create_source(world, attached_to="earth", magnitude=current)
    return world


soils = {
    "homogeneous": gf.HomogeneousSoil(rho=100.0),
    "two-layer": gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=2.0),
}
engine = gf.create_engine(frequencies=[50.0])
impedances = {}
for kind in ("rod", "ring", "grid"):
    impedances[kind] = [earthing_system(kind, soil).solve(engine).input_impedance()[0].real for soil in soils.values()]
    print(f"{kind:5s} {impedances[kind][0]:6.2f} Ω {impedances[kind][1]:6.2f} Ω")
# rod    35.99 Ω  51.41 Ω
# ring    9.41 Ω  13.81 Ω
# grid    2.37 Ω   4.80 Ω
```

<!-- doctest-check
expected = {"rod": (35.9905, 51.41), "ring": (9.41, 13.81), "grid": (2.3735, 4.80)}
for kind, (z_h, z_2) in expected.items():
    assert abs(impedances[kind][0] - z_h) < 5e-3 and abs(impedances[kind][1] - z_2) < 5e-3, kind
-->

`TwoLayerSoil` is a top layer of resistivity $\rho_1 = 100\,\Omega\,\mathrm{m}$
and thickness $h_1 = 2\,\mathrm{m}$ over a half-space of
$\rho_2 = 300\,\Omega\,\mathrm{m}$. The more resistive lower layer raises
every impedance, and the more so the larger the electrode: the current of a
large electrode spreads deeper into the soil. The rod gains 43 %, the ring
47 %, the 20 m grid more than doubles. The engine needs no change: `image`
forwards a two-layer soil to `image_2layer`, the Tagg/Sunde image series.

## 3. A fault: earth potential rise and touch voltage

Feed the grid in the two-layer soil with an earth-fault current of 100 A.
The touch voltage next to a grid depends on how the current crowds to its
edges and corners, so solve it with `mom`, which computes one current per
wire segment (the default `image` assumes one uniform current per straight
wire and warns near grids):

```python
grid = earthing_system("grid", soils["two-layer"], current=100.0)   # 100 A
fault = grid.solve(gf.create_engine(backend="mom", frequencies=[50.0]))

epr = abs(fault.electrode_potentials["earth"][0])
_, u_touch = gf.touch_voltage_envelope(fault, grid, electrode="earth", distance=1.0)
u_step = abs(gf.step_voltage(fault, position=(10.0, 10.0, 0.0), direction=(1.0, 1.0, 0.0), world=grid))
u_tp = gf.permissible_touch_voltage_en50522(0.5)   # fault cleared after 0.5 s
print(f"EPR {epr:.0f} V, touch voltage up to {abs(u_touch).max():.0f} V, step voltage {u_step:.0f} V")
print(f"permissible touch voltage {u_tp:.0f} V")
# EPR 476 V, touch voltage up to 129 V, step voltage 54 V
# permissible touch voltage 225 V
```

<!-- doctest-check
assert abs(epr - 475.70) < 0.05
assert abs(abs(u_touch).max() - 128.94) < 0.05 and abs(u_step - 54.02) < 0.05
assert u_tp == 225.0
-->

- The **earth potential rise** (EPR) is the potential of the grid against
  remote earth: here $4.76\,\Omega \times 100\,\mathrm{A}$ (`mom` reads the
  impedance 0.9 % below `image`).
- The **touch voltage** $U_T = U_E - \varphi(\text{feet})$ is the EPR
  $U_E$ of the touched metal minus the surface potential where a person
  stands, 1 m away. `touch_voltage_envelope` puts the feet 1 m outside the
  grid's footprint, on 24 directions and on the arc around each corner; the
  largest value lies off the corners.
- The **step voltage** is the difference of the surface potential over
  1 m; here from the point above a corner outwards.
- The comparison with the permissible touch voltage of EN 50522 (Figure 8,
  Table B.4; 225 V for 0.5 s) leaves out body, footwear and surface-layer
  resistances, which is conservative;
  `permissible_prospective_touch_voltage_en50522` adds them.

With the default `image` engine the same envelope reads 9 % higher and the
call warns with `UniformCurrentFieldWarning`. For impedances the two engines
agree within 1 %; for field quantities near extended electrodes `mom` is
the reference ([choosing an engine](../explanation/engines/index.md)).
Every result scales linearly with the source current.

## 4. A potential profile and a plot

`FieldResult.potential` evaluates the potential at any point of the soil
or its surface. Along a line at $y$ = 2.5 m, through the middle of a row of
meshes:

```python
import numpy as np

x = np.array([0.0, 11.0, 20.0, 50.0])                       # m from the centre
points = np.column_stack([x, np.full_like(x, 2.5), np.zeros_like(x)])
for xi, phi in zip(x, fault.potential(points).real):
    print(f"x = {xi:4.0f} m: {phi:4.0f} V ({phi / epr:.0%} of the EPR)")
# x =    0 m:  460 V (97% of the EPR)
# x =   11 m:  400 V (84% of the EPR)
# x =   20 m:  233 V (49% of the EPR)
# x =   50 m:   95 V (20% of the EPR)
```

<!-- doctest-check
phi_check = fault.potential(points).real
assert abs(phi_check[0] - 460.20) < 0.05 and abs(phi_check[-1] - 95.16) < 0.05
-->

Inside the grid the surface stays close to the EPR, which keeps touch
voltages there small; outside it the potential falls off towards remote
earth, and the touch voltage grows with it. Two plotting functions show the
same as figures:

```python
import matplotlib.pyplot as plt

gf.plot_potential_profile(fault, start=(0.0, 2.5, 0.0), direction=(1.0, 0.0, 0.0), distance=50.0)
gf.plot_surface_potential(fault, grid)
plt.show()
```

The first draws the curve of the table above, the second a colour map of
the surface potential over the grid and its surroundings with the
electrode drawn on top. The [plotting guide](../how-to/plot-results.md)
shows the other plots (geometry, cross-sections, current sharing, sweeps).

## 5. Two earthing systems joined by a PEN conductor

A substation ring earth and the rod earth of a house 40 m away, joined by
a PEN conductor, in the two-layer soil; a current of 1 A is injected at
the substation and returns through remote earth.

```python
world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=2.0))
gf.create_electrode(world, "ring", name="substation", center=(0.0, 0.0, 0.8), radius=3.0)
gf.create_electrode(world, "rod", name="house", position=(40.0, 0.0, 0.0), length=3.0)
gf.create_conductor(world, name="pen", start="substation", end="house", cross_section=95e-6)
gf.create_source(world, attached_to="substation", magnitude=1.0)

result = world.solve(gf.create_engine(frequencies=[50.0]))
print(result.input_impedance()[0])                # (11.2695+0j) Ω
print(result.cluster_impedance("substation")[0])  # (14.1034+0j) Ω
```

The conductor is a 95 mm² copper PEN (`cross_section` in m²): a branch
with resistance $R = \rho_\text{Cu}\,\ell/A$ between the two electrodes,
with the copper resistivity $\rho_\text{Cu}$, the length $\ell$ and the
cross-section $A$. Without `cross_section` it would be an ideal connection
and both electrodes would form one galvanic cluster.

### Three impedances of one solve

```python
Z_in = result.input_impedance()[0]
Z_substation = result.cluster_impedance("substation")[0]
Z_house = result.cluster_impedance("house")[0]
share = result.electrode_currents["substation"][0]   # leakage current, A
print(f"{Z_in.real:.2f} Ω, {Z_substation.real:.2f} Ω, {share.real:.0%} at the substation")
# 11.27 Ω, 14.10 Ω, 80% at the substation
```

| Method | Definition | What it describes |
|---|---|---|
| `input_impedance(source)` | $(\varphi_a - \varphi_r)/I_s$ | what an earth tester or a fault sees at the feeding point: the whole connected system |
| `cluster_impedance(electrode)` | $\varphi_c / \sum_{e \in c} I_e$ | the local earthing system: one galvanic cluster and only the current it leaks itself |
| `grounding_impedance(electrode)` | $\varphi_c / I_e$ | the share of one electrode; equal to the cluster impedance for a cluster of one electrode |

$\varphi_a$, $\varphi_r$ are the potentials of the feeding and the
return node (zero for a return through remote earth), $I_s$ the source
current, $\varphi_c$ the length-weighted potential of a cluster and
$I_e$ the current an electrode leaks into the soil. Here 20 % of the
test current flows through the PEN to the house earth, so the substation
ring alone (14.10 Ω) is seen as 11.27 Ω at the feeding point. Because
the PEN resistance is only a few milliohms, the input impedance is the
parallel connection of the two cluster impedances:

```python
print(abs(1 / (1 / Z_substation + 1 / Z_house) - Z_in))   # 0.0003 Ω
```

Which one you need depends on the question: the input impedance for a
measurement or a fault loop (it is the default of `sweep`,
`convergence_study` and the `rho-f` fits), the cluster impedance to
characterise one earthing system on its own. The
[sources and measurement](../explanation/sources-and-measurement.md)
page explains the definitions in detail.

## 6. Check the result with a second method

Every number from a field model should be checked against an
independent computation. Two quick checks: solve the same world with a
second backend, and refine the discretisation.

```python
report = gf.compare_engines(
    world,
    engines={"image": gf.create_engine(), "mom": gf.create_engine(backend="mom")},
    rel_tolerance=0.05,
)
print(report.summary())     # largest deviation 1.1 % (house cluster): consistent

fine = world.solve(gf.create_engine(segment_length=0.25))
print(fine.input_impedance()[0])   # (11.2758+0j) Ω, +0.06 % at half the segment length
```

The two methods differ in the current model: `mom` solves for one current
per segment by point matching with midpoint point sources, `image_2layer`
assumes a uniform current per unit length on each straight wire and
integrates it with the exact line-source kernel. Agreement within about
1 % is expected (the point kernel reads 1–2 % low, see
[source kernels](../explanation/kernels.md)).

??? info "Values asserted by the test suite"

    The code blocks on this page are executed by the documentation tests;
    these assertions keep the numbers quoted in sections 5 and 6 in step
    with the code.

    ```python
    assert abs(Z_in - 11.2695) < 5e-4 and abs(Z_substation - 14.1034) < 5e-4
    assert abs(share.real - 0.799) < 1e-3
    assert abs(1 / (1 / Z_substation + 1 / Z_house) - Z_in) < 1e-3
    assert report.is_consistent and max(report.deviations.values()) < 0.012
    assert abs(fine.input_impedance()[0] - 11.2758) < 5e-4
    ```

## Where next

- [Tutorials](../tutorials/index.md): from the first impedance to layered
  soil, connected systems, frequency response, the `rho-f` export, TN
  networks, measurements and touch voltages, step by step.
- [Examples](../examples/index.md): three complete case studies — a
  secondary substation with its low-voltage network, a single-family house,
  and the mesh grid of a high-voltage substation in two-layer soil.
- [Choosing an engine](../explanation/engines/index.md): which backend for
  which question, and where each one must not be used.
- [How-to guides](../how-to/index.md): recipes for sweeps, plots, touch
  voltages, export and large studies.
- [Factories](../reference/api/factories.md): every `create_*` function and
  its parameters; [units and conventions](../reference/units-and-conventions.md):
  coordinates, phasors, signs.
