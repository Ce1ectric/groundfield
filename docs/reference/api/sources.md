# Sources

A source impresses a current into the network. `CurrentSource` feeds a
node (`attached_to`: an electrode or a junction) and takes the current
back either at remote earth (`return_to=None`) or at a second node
(`return_to`). `VoltageSource` exists only as a data model so that
older world files load; solving a world that contains one raises
`NotImplementedError`.

## Mathematical and physical context

Sources are the right-hand side of the nodal system. A one-terminal
source injects the phasor $I = |I|\,e^{j\varphi_I}$ (RMS value,
`magnitude` in A, `phase_deg` in degrees) at `attached_to`; Kirchhoff's
current law closes the circuit through the soil to remote earth, the
potential reference. A two-terminal source injects $+I$ at
`attached_to` and $-I$ at `return_to`, so the loop closes through the
soil and the conductor network — the set-up of a fall-of-potential
measurement with its auxiliary electrode, or of a fault current that
returns to a neighbouring substation. The impedance the source sees is

$$
Z_\text{in} = \frac{\varphi_a - \varphi_r}{I},
$$

`FieldResult.input_impedance()`, with $\varphi_r = 0$ for a return
through remote earth. The problem is linear, so a voltage $U$ at the
feeding point corresponds to the current $I = U/Z_\text{in}$: solve with
1 A and scale.

A backend whose capabilities do not include two-terminal sources
returns the current through remote earth and warns with
`BackendCapabilityWarning` (see the [backend registry](registry.md));
the measurement physics is explained in
[sources and measurement](../../explanation/sources-and-measurement.md).

## Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(world, "rod", name="earth", position=(0.0, 0.0, 0.0), length=3.0)
gf.create_electrode(world, "rod", name="aux", position=(100.0, 0.0, 0.0), length=1.0)
gf.create_source(world, attached_to="earth", return_to="aux", magnitude=1.0)

result = world.solve(gf.create_engine(frequencies=[50.0]))
Z_loop = result.input_impedance()[0]               # 126.19 Ω
Z_earth = result.cluster_impedance("earth")[0]      # 35.83 Ω
Z_aux = result.cluster_impedance("aux")[0]          # 90.35 Ω

# +1 A leaves "earth" into the soil, -1 A at "aux" (the current returns)
I = result.electrode_currents
assert abs(I["earth"][0] - 1.0) < 1e-9 and abs(I["aux"][0] + 1.0) < 1e-9
# the loop is the series connection of both earths (mutual coupling included)
assert abs(Z_loop - (Z_earth + Z_aux)) < 1e-9
```

`Source` is a discriminated union over `kind` (`"current"`,
`"voltage"`); `SourceAdapter` validates a stand-alone source dictionary,
for example from a JSON file. Assignments are validated, so a source
edited between two solves (`src.magnitude = 20.0`) cannot take a
non-finite value.

## API reference

::: groundfield.sources
