# Model the inductive coupling of cables and leads

Below 1 kHz the soil is quasi-static, but cables are not: a PEN conductor,
a cable shield or a measurement lead has a series impedance
$R' + j\omega L'$ per metre and couples magnetically to every other
conductor. This guide shows how to switch the inductive model on, how to
choose the earth-return model, how to check a model against the classical
per-metre formulas, and how to add the skin effect.

## Switch it on

Inductance is a property of a conductor with a cross section (a branch);
the earth-return model is a property of the engine:

```python
import math

import numpy as np

import groundfield as gf


def two_cables(gap, rho=100.0):
    # Cable A carries 1 A between its end rods; cable B, `gap` metres away, is passive.
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=rho))
    for name, y in (("A", 0.0), ("B", gap)):
        gf.create_electrode(world, "rod", name=f"{name}_start", position=(0.0, y, 0.6), length=1.5)
        gf.create_electrode(world, "rod", name=f"{name}_end", position=(100.0, y, 0.6), length=1.5)
        gf.create_conductor(
            world,
            name=f"cable_{name}",
            start=f"{name}_start",
            end=f"{name}_end",
            conductor_type="pen",
            resistivity=2.82e-8,  # aluminium, Ohm m
            cross_section=50e-6,  # m^2
            coupling_to_soil="isolated",  # insulated: a branch without leakage
            inductance_model="neumann",  # self and mutual inductance (Neumann integrals)
            discretize_segment_length=5.0,  # sub-branches of 5 m
        )
    gf.create_source(world, attached_to="A_start", return_to="A_end", magnitude=1.0)
    return world


frequencies = [50.0, 250.0, 1000.0]
engine = gf.create_engine(segment_length=0.5, frequencies=frequencies, sommerfeld_assembly="tabulated")
induced = {}
for gap in (0.5, 5.0, 50.0):
    result = engine.solve(two_cables(gap))
    potentials = result.electrode_potentials
    induced[gap] = np.array([potentials["B_start"][k] - potentials["B_end"][k] for k in range(len(frequencies))])
    print(f"gap {gap:4.1f} m: |U_B| = {np.round(np.abs(induced[gap]), 4)} V per A at {frequencies} Hz")
```

<!-- doctest-check
for gap, u in induced.items():
    assert np.all(np.diff(np.abs(u)) > 0)  # grows with frequency
    assert 85.0 < math.degrees(np.angle(u[0])) < 90.0  # inductive: about +90 deg at 50 Hz
assert np.all(np.abs(induced[0.5]) > np.abs(induced[5.0])) and np.all(np.abs(induced[5.0]) > np.abs(induced[50.0]))
-->

The voltage induced along the passive cable is almost purely imaginary
(about +90° against the current in cable A), grows with frequency and
falls with the distance between the cables.

## Choose the earth-return model

| `earth_inductive_model` | Physics | Use it for |
|---|---|---|
| `"sommerfeld"` (default) | Sommerfeld/Pollaczek integral of the conducting half-space or layered earth over the real segment geometry; conductors in or above the surface plane are handled | every quantitative study |
| `"carson_series"` | Carson's (1926) per-metre correction for infinitely long **parallel** wires over homogeneous earth, on the anti-parallel image: tiled per metre of overlap for near-parallel horizontal pairs ($\lvert\cos\varphi\rvert > 0.99$, inclination below 8.1°), across the surface through the field transmitted into the soil; every other pair and inclined segment gets the Sommerfeld correction of the same earth, so the result steps at 8.1° | long parallel cables in homogeneous soil; not for layered soil |
| `"perfect_mirror"` | the earth as a perfect magnetic mirror, frequency independent; with conductors on both sides of the soil surface the solve warns (its additive image has no basis for such pairs) | a fast diagnostic baseline only |

With `"sommerfeld"` the correction is assembled from shared tables
(`sommerfeld_assembly="tabulated"`, the default since 2026.11; the
examples on this page spell it out): it agrees with the per-pair assembly
(`"exact"`) to far below the model accuracy and is much faster on networks
with many inductive segments (see
[Speed up large studies](speed-up-large-studies.md)).

## Check against the per-metre formulas

For a long straight cable the series impedance per metre approaches the
conductor–earth loop impedance of an infinite line, given by Pollaczek's
integral (`groundfield.references.pollaczek`) and, for a line on the
surface, by Carson's formula (`groundfield.references.earth_return`). The
comparison needs lines much longer than the skin depth of the earth,
$\delta = \sqrt{2\rho/(\omega\mu_0)}$ (about 160 m at 1 kHz in
100 Ω·m), and the end effects of the finite line must cancel: we take the
difference of an 800 m and a 1600 m line. The line carries no leakage, so
its series impedance is $(\varphi_\mathrm{start} - \varphi_\mathrm{end}) /
I_\mathrm{end}$.

```python
from groundfield.references.pollaczek import loop_impedance_conductor_earth

RHO, F = 100.0, 1000.0


def line_impedance(length):
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=RHO))
    gf.create_electrode(world, "ring", name="start", center=(0.0, 0.0, 0.8), radius=2.0)
    gf.create_electrode(world, "rod", name="end", position=(length, 0.0, 0.5), length=3.0)
    gf.create_conductor(
        world,
        name="line",
        start="start",
        end="end",
        cross_section=95e-6,
        coupling_to_soil="isolated",
        inductance_model="neumann",
        discretize_segment_length=50.0,
    )
    gf.create_source(world, attached_to="start", magnitude=1.0)
    result = gf.create_engine(segment_length=2.0, frequencies=[F], sommerfeld_assembly="tabulated").solve(world)
    p = result.electrode_potentials
    return (p["start"][0] - p["end"][0]) / result.electrode_currents["end"][0], world.conductors[0]


(z_800, line), (z_1600, _) = line_impedance(800.0), line_impedance(1600.0)
z_per_m = (z_1600 - z_800) / 800.0
reference = loop_impedance_conductor_earth(
    RHO,
    RHO,
    10.0,  # homogeneous earth: equal layers, thickness irrelevant
    F,
    depth=0.5 * (line.start[2] + line.end[2]),
    gmr=line.wire_radius * math.exp(-0.25),  # geometric mean radius of a solid wire
    r_internal=line.resistivity / line.cross_section,
)
print(f"groundfield: z' = {z_per_m * 1e3:.5f} Ohm/km")
print(f"Pollaczek:   z' = {reference * 1e3:.5f} Ohm/km")
print(f"X' {z_per_m.imag / reference.imag - 1:+.1e}, R' {z_per_m.real / reference.real - 1:+.1e}")
```

<!-- doctest-check
assert abs(z_per_m.imag / reference.imag - 1.0) < 5e-3
assert abs(z_per_m.real / reference.real - 1.0) < 2e-2
-->

The reactance agrees closely; the earth-return resistance converges more
slowly with the line length, because the current in the earth spreads
over the skin depth beyond the ends of the line.

## Add the skin effect

By default a conductor uses its DC resistance and the DC internal
inductance. `resistance_model="skin"` uses the exact solution for a solid
round conductor instead (Bessel functions), so the resistance rises and
the internal inductance falls with frequency. For a check, at low
frequency the classical expansion $R_\mathrm{ac}/R_\mathrm{dc} \approx 1 +
\frac{1}{48}(r/\delta_c)^4$ with the skin depth of the conductor
$\delta_c = \sqrt{2\rho_c/(\omega\mu_0)}$ must hold:

```python
world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(world, "rod", name="a", position=(0.0, 0.0, 0.6), length=2.0)
gf.create_electrode(world, "rod", name="b", position=(200.0, 0.0, 0.6), length=2.0)
pen = gf.create_conductor(
    world,
    name="pen",
    start="a",
    end="b",
    conductor_type="pen",
    resistivity=2.82e-8,
    cross_section=150e-6,
    coupling_to_soil="isolated",
    inductance_model="neumann",
    resistance_model="skin",
    discretize_segment_length=10.0,
)
r_dc = pen.resistivity / pen.cross_section
radius = math.sqrt(pen.cross_section / math.pi)
for f in (50.0, 550.0, 1000.0):
    ratio = pen.internal_impedance_per_length(f).real / r_dc
    delta_c = math.sqrt(2.0 * pen.resistivity / (2.0 * math.pi * f * 4e-7 * math.pi))
    print(f"{f:6.0f} Hz: R_ac/R_dc = {ratio:.5f} (low-frequency expansion {1 + (radius / delta_c) ** 4 / 48:.5f})")
```

<!-- doctest-check
ratio_50 = pen.internal_impedance_per_length(50.0).real / r_dc
delta_50 = math.sqrt(2.0 * pen.resistivity / (2.0 * math.pi * 50.0 * 4e-7 * math.pi))
assert abs(ratio_50 - (1 + (radius / delta_50) ** 4 / 48)) < 1e-5
assert pen.internal_impedance_per_length(1000.0).real > pen.internal_impedance_per_length(550.0).real > r_dc
-->

The solver adds the difference to the DC convention to every branch of
the conductor, so `engine.solve(world)` now sees a frequency-dependent
resistance. At 50 Hz the effect is negligible for typical PEN cross
sections; it matters for harmonics up to 1 kHz.

## Things to know

- A conductor without `inductance_model` is purely resistive.
- Two conductors closer than the sum of their radii — cables in one
  trench drawn on one axis, a service drop along its main — couple as
  wires touching side by side; a thinner conductor inside a thicker one
  (a core on the axis of a cable shield) couples as a filament inside a
  tube, at the radius of the thicker one. Two separate cables of
  different radii drawn on one axis are therefore read as one concentric
  cable: draw them at least $r_1 + r_2$ apart.
- The leads of a measurement (tutorial
  [T07](../tutorials/T07_fall_of_potential_measurement.ipynb)) couple
  like cables; lay the probe lead at an angle to the feed lead.
- Conductors that cross or lie in the soil surface are solved with
  `"sommerfeld"`; with the other two models their image is singular and
  the solve raises.
- The theory of the three models is on the
  [earth-return page](../explanation/earth-return.md).
