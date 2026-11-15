# Sources

A source drives the network at a node (`attached_to`: an electrode or a
junction) against remote earth (`return_to=None`) or against a second
node (`return_to`). `CurrentSource` impresses a current;
`VoltageSource` impresses a voltage between its two terminals, and the
solve returns its current.

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
through remote earth.

A voltage source fixes $\varphi_a - \varphi_r = U = |U|\,e^{j\varphi_U}$
(`magnitude` in V, RMS) and leaves its current $I$ to the solve. The
engines add it by modified nodal analysis: one more unknown, the source
current, in Kirchhoff's current law at both terminals, and one more row
with the constraint (see the
[engine family](../../explanation/engines/index.md#network-equations)).
$I$ has the sign convention of a current source — it flows from the source
into `attached_to` — and the result reports it in
`result.sources[k].current` (the impressed voltage in
`result.sources[k].voltage`, both in `to_dataframe("sources")`);
`input_impedance()` is $U / I$. The problem is linear, so the solution
equals the 1 A solve of a current source at the same terminals scaled by
$U / Z_\text{in}$, to round-off, in every engine that solves voltage
sources (`image`, `image_2layer` including electrodes through the layer
interface, `mom`, `mom_sommerfeld`; inductive conductors and several
frequencies included). Several sources superpose; for one voltage source
alone the other voltage sources are shorted (0 V) and the current sources
open. The source is ideal: a series impedance is a junction at the
terminal and a finite conductor to the electrode, as in the example below.
A voltage source of 0 V is an ideal short whose current the solve returns.

`World.check` rejects, before any numerics run, a voltage source whose
terminals are bonded into one cluster by ideal conductors (short circuit)
and voltage sources that form a loop through ideal bonds or remote earth —
two of them from one cluster to remote earth, or a ring of two-terminal
ones: the loop fixes one potential difference twice. `fem`, which lacks the
registry capability `voltage_sources`, raises `BackendCapabilityError`.
Up to 0.15.0 every backend dropped voltage sources with a warning and the
world solved to zero potentials (finding 10.51).

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

A voltage source of 230 V behind a 10 Ω series resistance, from the
junction `T` to the same rod:

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(world, "rod", name="earth", position=(0.0, 0.0, 0.0), length=3.0)
gf.create_junction(world, name="T", position=(0.0, 0.0, 0.0))
gf.create_conductor(
    world, name="r_i", start="T", end="earth", cross_section=50e-6,
    lumped_series_resistance_ohm=10.0,
)
gf.create_source(world, attached_to="T", kind="voltage", magnitude=230.0)

result = world.solve(gf.create_engine(frequencies=[50.0]))
I = result.sources[0].current[0]                    # 5.001 A, solved
Z_in = result.input_impedance()[0]                  # 45.99 Ω = 230 V / I
U_E = result.electrode_potentials["earth"][0]       # 179.99 V = 230 V - 10 Ω · I

# the terminal sits at the impressed voltage; the rod alone is 35.99 Ω
assert abs(result.metadata["junction_potentials"]["T"][0] - 230.0) < 1e-9
assert abs(Z_in - (10.0 + 35.9905)) < 1e-3 and abs(U_E - (Z_in - 10.0) * I) < 1e-9
```

`Source` is a discriminated union over `kind` (`"current"`,
`"voltage"`); `SourceAdapter` validates a stand-alone source dictionary,
for example from a JSON file. Assignments are validated, so a source
edited between two solves (`src.magnitude = 20.0`) cannot take a
non-finite value.

## API reference

::: groundfield.sources
