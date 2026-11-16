# Assess touch voltages

This guide computes the touch voltages of an installation at its
accessible metal parts and compares them with the permissible values of
EN 50522. Read it when you verify an earthing system for an earth fault:
a station, a transferred potential at a house, an open-air grid.

EN 50522 assesses the **touch voltage** $U_T$ between an accessible
conductive part and the feet of a person who touches it, standing 1 m
away from the part, against the **permissible touch voltage**
$U_{Tp}(t_f)$ of the fault duration $t_f$. groundfield returns the
prospective (open-circuit) value

$$
U_T = U_\text{part} - \varphi(\text{feet}),
$$

the potential of the touched part less the soil potential at the feet, as
a phasor in V for the source current of the solve.

**Prerequisites:** the basic workflow of tutorial
[T01](../tutorials/T01_first_grounding_impedance.ipynb). The steps define
the touch points, evaluate a transferred potential, sample the meshes of
an open-air grid and compare the results with the permissible values.

## 1. Define the touch points

The touched parts are the metal parts above ground: the housing or the
door of a station, a mast, a fence post, a cable cabinet, a house
connection box. `gf.touch_voltage_at_part` takes one part — its
horizontal position and the electrode or junction it is bonded to — and
puts the feet on the circle of radius `distance` (1 m) around it.
`exclude` removes the foot positions inside outlines where nobody can
stand (the housing itself); `directions` keeps only the given sides,
for example the outside of a fence.

A compact substation: a housing of 3 m × 2.5 m inside a ring earth of
2.5 m radius, 100 A of earth-fault current into the station earth at
50 Hz, 100 Ω·m.

```python
import math

import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(world, "ring", name="station", center=(0.0, 0.0, 0.6), radius=2.5)
gf.create_source(world, attached_to="station", magnitude=100.0)
result = gf.create_engine(frequencies=[50.0]).solve(world)

housing = [(-1.5, -1.25), (1.5, -1.25), (1.5, 1.25), (-1.5, 1.25)]  # outline in m
angles, u_door, feet = gf.touch_voltage_at_part(
    result, world, position=(0.0, -1.25), bonded_to="station",  # door in the south wall
    exclude=[housing], return_points=True,
)
_, u_fence = gf.touch_voltage_at_part(
    result, world, position=(0.0, -4.0), bonded_to="station",  # fence post, bonded
    directions=[(0.0, -1.0)],  # people stand outside the fence
)
u_e = abs(result.electrode_potentials["station"][0])
print(f"earth potential rise {u_e:.0f} V")
print(f"door: {abs(u_door).max():.0f} V (largest of {len(u_door)} foot positions)")
print(f"fence post: {abs(u_fence[0]):.0f} V")
```

<!-- doctest-check
assert abs(u_e - 1121.0) < 0.5
assert len(u_door) == 13 and abs(abs(u_door).max() - 447.2) < 0.5
assert abs(abs(u_fence[0]) - 783.5) < 0.5
assert all(f[1] <= -1.25 + 1e-9 for f in feet)  # no foot inside the housing
-->

Of the 24 positions around the door, the 11 inside the housing are
dropped; the two on the wall line are kept. The door is inside the ring,
where the surface potential is high, so its touch voltage (447 V) is
lower than 1 m outside the ring; the bonded fence post outside the ring
has 784 V.

## 2. Evaluate a transferred potential over the PEN

A part bonded to a node far from the faulted earth carries that node's
potential into an area of low potential rise (EN 50522, 3.6). A house
40 m away without an earth electrode of its own is connected to the
station by the PEN conductor of the low-voltage cable, which ends on a
junction at the service connection box:

```python
gf.create_junction(world, name="service_box", position=(40.0, 0.0, 0.7))
gf.create_conductor(
    world, name="pen", start="station", end="service_box",
    resistivity=2.83e-8, cross_section=150e-6,  # aluminium, 150 mm²
)
result = gf.create_engine(frequencies=[50.0]).solve(world)

house = [(40.0, -4.0), (50.0, -4.0), (50.0, 4.0), (40.0, 4.0)]
_, u_box = gf.touch_voltage_at_part(
    result, world, position=(40.0, 0.0), bonded_to="service_box", exclude=[house]
)
print(f"service connection box: {abs(u_box).max():.0f} V of {u_e:.0f} V")
```

<!-- doctest-check
assert abs(abs(u_box).max() - 1081.2) < 0.5
assert abs(u_box).max() > 0.95 * u_e
-->

No current flows on the PEN, so the box carries the station potential,
while the soil at the house is almost at remote earth: 96 % of the earth
potential rise is a touch voltage there.

## 3. Sample the meshes of an open-air grid

`touch_voltage_envelope` puts the feet 1 m from the nearest wire of an
electrode, all around it — a proxy where the touched parts are not known.
By default it samples nothing inside the footprint, because there stands
the station, the house, the steel structure or the mast. Over an
open-air grid, such as a substation yard, people walk over the grid and
the surface potential sags inside the meshes: `interior="meshes"` adds
the centre of every mesh (the *mesh voltage* of IEEE Std 80), of a ring,
and of the cells of a foundation built from bonded wires;
`interior=True` also the inside of a closed polyline. Use the modes only
where the inside is accessible open ground.

Near grids the `image` backend's uniform current is not field-grade
(`UniformCurrentFieldWarning`), so the yard is solved with `mom`:

```python
yard = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(
    yard, "grid_mesh", name="grid", corner=(0.0, 0.0, 0.5), size=(20.0, 20.0), n_x=2, n_y=2
)
gf.create_source(yard, attached_to="grid", magnitude=100.0)
yard_result = gf.create_engine(backend="mom", segment_length=0.5, frequencies=[50.0]).solve(yard)

_, outside = gf.touch_voltage_envelope(yard_result, yard, electrode="grid")
_, everywhere = gf.touch_voltage_envelope(yard_result, yard, electrode="grid", interior="meshes")
print(f"1 m outside: {abs(outside).max():.1f} V; mesh centres: {abs(everywhere[24:]).max():.1f} V")
```

<!-- doctest-check
assert len(outside) == 24 and len(everywhere) == 24 + 4
assert abs(abs(outside).max() - 113.6) < 0.1 and abs(abs(everywhere[24:]).max() - 90.3) < 0.1
-->

The value to assess is the largest of both, here 1 m outside a corner.

The feet stand 1 m from the nearest wire of the whole foundation
(`anchor="footprint"`, the default). `anchor="connection_point"` puts them
on a circle round the electrode's connection point instead. That point is
the centre of a hand-built grid mesh, but a corner of a rectangular
foundation of the generators — the start of its local −y side, the
electrode named after the foundation — so the circle goes round that
corner: 100 A into a generated 20 m × 5 m ring at 0.8 m in 100 Ω·m give
229.71 V there against 161.76 V round its centre (default engine, 0.5 m
segments). Keep the default for generated foundations.

??? note "Background and audit trail"

    `anchor="connection_point"` keeps the rule of 0.15.0. The connection
    point of a generated foundation is a corner since 2026.11: for meshes
    since audit A2G-04, for rectangles without cross-braces since user
    decision Q38. With 0.15.0, round the centre of one grid mesh, the same
    case gave 156.07 V (audits D3A-03, R4B-09).

## 4. Compare with the permissible touch voltage

`permissible_touch_voltage_en50522(t_f)` follows Figure 8 / Table B.4 of
EN 50522:2022 up to 10 s and returns 80 V beyond, the value the standard
allows for current-flow durations considerably longer than 10 s. Pass
`math.inf` for an earth fault that is not cleared, as in a
resonant-earthed network. With footwear or a resistive surface layer,
`permissible_prospective_touch_voltage_en50522` adds the body current
times the additional resistances (Annex B.4):

```python
u_tp_cleared = gf.permissible_touch_voltage_en50522(0.5)  # cleared after 0.5 s
u_tp_uncleared = gf.permissible_touch_voltage_en50522(math.inf)
u_vtp = gf.permissible_prospective_touch_voltage_en50522(math.inf, R_F1=1000.0, rho_s=100.0)
for part, u in (("door", abs(u_door).max()), ("service connection box", abs(u_box).max())):
    print(f"{part}: {u:.0f} V against U_Tp = {u_tp_cleared:.0f} V (0.5 s), "
          f"{u_tp_uncleared:.0f} V (not cleared), U_vTp = {u_vtp:.1f} V (1000 Ω shoes, not cleared)")
```

<!-- doctest-check
assert u_tp_cleared == 225.0 and u_tp_uncleared == 80.0
assert abs(u_vtp - (80.0 + 0.050 * (1000.0 + 1.5 * 100.0))) < 1e-9
-->

**What you get:** each part's largest touch voltage against $U_{Tp}$ for a
cleared and an uncleared fault and against $U_{vTp}$ with footwear. Both
parts exceed the limits by far at 100 A: the earthing system of this
example needs more than a ring (EN 50522, 5.4.2 and Annex E, describe the
verification steps and the measures).

## See also

- Tutorial [T08](../tutorials/T08_touch_and_step_assessment.ipynb): a
  complete assessment of a substation grid.
- [`postprocess` reference](../reference/api/postprocess.md): the
  definitions, the validity of the image backend near grids and the
  measured deviations.
- [`image` engine](../explanation/engines/image.md#field-quantities-and-the-uniform-current-ansatz):
  why the uniform-current ansatz is impedance-grade, not field-grade.
- [Read and act on warnings](warnings-and-diagnostics.md):
  `UniformCurrentFieldWarning` and the other warnings of a field
  evaluation.
