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

A voltage source drives its terminals with $U = |U|\,e^{j\varphi_U}$
(`magnitude` in V, RMS) behind an internal series impedance
$Z_q(f) = R + j\omega L$ (`internal_resistance_ohm` in Ω,
`internal_inductance_h` in H, both ≥ 0 and 0 by default — the ideal
source),

$$
\varphi_a - \varphi_r + Z_q\,I = U ,
$$

and leaves its current $I$ to the solve. The engines add it by modified
nodal analysis: one more unknown, the source current, in Kirchhoff's
current law at both terminals, and one more row with the constraint,
$Z_q$ on its diagonal (see the
[engine family](../../explanation/engines/index.md#network-equations)).
$I$ has the sign convention of a current source — it flows from the source
into `attached_to` — and the result reports it in
`result.sources[k].current` (the impressed voltage in
`result.sources[k].voltage`, $R$ and $L$ in
`result.sources[k].internal_resistance_ohm` / `internal_inductance_h`, all
of them in `to_dataframe("sources")`). `input_impedance()` is the
impedance the source sees at its terminals,
$Z_\text{in} = (\varphi_a - \varphi_r)/I = U/I - Z_q$ — without the
source's own $Z_q$. The problem is linear: for a world with this one
source the solution equals the 1 A solve of a current source at the same
terminals scaled by $I = U / (Z_q + Z_\text{in})$, to round-off, in every
engine that solves voltage sources (`image`, `image_2layer` including
electrodes through the layer interface, `mom`, `mom_sommerfeld`;
inductive conductors and several frequencies included). With several
sources the solve is the superposition of all of them — for one voltage
source alone the other voltage sources are shorted (0 V behind their
$Z_q$) and the current sources open — and each `input_impedance()` is an
apparent one: two 230 V sources on two 3 m rods 5 m apart (100 Ω·m) each
drive 7.5 % less current than one alone and report an input impedance
8.1 % higher (`image`). A voltage source of 0 V is a short through $Z_q$
whose current the solve returns. The reactance $j\omega L$ is evaluated
at every solved frequency (it vanishes at $f = 0$), so a world whose
voltage source has an inductance is solved once per frequency even
without inductive conductors.

`World.check` rejects, before any numerics run, a voltage source whose
terminals are bonded into one cluster by ideal conductors (short circuit,
with or without $Z_q$), negative or non-finite $R$ / $L$, and voltage
sources without internal resistance that form a loop through ideal bonds
(conductors at or below 10⁻⁶ Ω) or remote earth — two of them from one
cluster to remote earth, or a ring of two-terminal ones: the loop fixes
one potential difference twice. One source of the loop with
`internal_resistance_ohm > 0` opens it; an inductance alone does not,
because its impedance vanishes at $f = 0$. `fem`, which lacks the registry
capability `voltage_sources`, raises `BackendCapabilityError`. Up to
0.15.0 every backend dropped voltage sources with a warning and the world
solved to zero potentials (finding 10.51).

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

A voltage source of 230 V behind a 10 Ω series resistance, modelled as
before 2026.11 with a junction `T` and a lumped conductor to the same rod
(the resistance is then part of the network, so `input_impedance()`
includes it):

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

The same source with its internal impedance on the source itself, here
$Z_q = 10\ \Omega + j\omega \cdot 5\ \text{mH}$ at 0, 50 and 1000 Hz
(the current is $U / (Z_q + Z_E)$, the input impedance the rod's
$Z_E$ alone):

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(world, "rod", name="earth", position=(0.0, 0.0, 0.0), length=3.0)
source = gf.create_source(
    world, attached_to="earth", kind="voltage", magnitude=230.0,
    internal_resistance_ohm=10.0, internal_inductance_h=5e-3,
)

result = world.solve(gf.create_engine(frequencies=[0.0, 50.0, 1000.0]))
I = result.sources[0].current                       # 5.001, 4.998, 4.130 A
Z_E = result.input_impedance()                      # 35.99 Ω at every frequency
U_E = result.electrode_potentials["earth"]          # 179.99, 179.88, 148.62 V

for k, f in enumerate(result.frequencies):
    Z_q = source.internal_impedance(f)              # 10 Ω + j·2π·f·5 mH
    assert abs(I[k] - 230.0 / (Z_q + Z_E[k])) < 1e-12 * abs(I[k])
    assert abs(U_E[k] - (230.0 - Z_q * I[k])) < 1e-9   # the terminal voltage
assert abs(Z_E[1] - 35.9905) < 1e-3 and abs(abs(I[2]) - 4.1295) < 1e-4
```

`Source` is a discriminated union over `kind` (`"current"`,
`"voltage"`); `SourceAdapter` validates a stand-alone source dictionary,
for example from a JSON file. Assignments are validated, so a source
edited between two solves (`src.magnitude = 20.0`) cannot take a
non-finite value.

## API reference

::: groundfield.sources
