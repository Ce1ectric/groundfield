# Sources, junctions and the measurement loop

A grounding system is never fed in isolation. A fault current enters at
one point and returns through a remote source; a measurement instrument
injects between the earth electrode and an auxiliary electrode and reads
the voltage against a probe; cable joints and instrument terminals are
nodes without soil contact. This page explains how `groundfield` models
these sources and nodes, which impedance a result reports, and how the
fall-of-potential measurement is wired since 2026.11.

## Mathematical and physical context

The engines solve a linear network: soil multiport, conductor branches
and nodes (see the [engine family](engines/index.md#network-equations)).
Current sources enter as impressed node currents, voltage sources as a
fixed potential difference between their terminals whose current is an
unknown (modified nodal analysis), so every quantity is linear in the
source values and results of several sources superpose. The
potential reference is **remote earth**: $\varphi \to 0$ far away from
all electrodes. A potential of an electrode is therefore its earth
potential rise (EPR) against remote earth, and a voltage between two
points is a difference of such potentials.

## Two-terminal sources

A current source has two terminals:

- `attached_to` — the node where the current $I$ enters
  (an electrode, a junction or, through its cluster, a bonded group);
- `return_to` — the node where it leaves again; `None` means the return
  through remote earth.

The solver injects $+I$ at `attached_to` and $-I$ at `return_to`
(`groundfield.solver._pipeline.inject`, shared by the image family and
the `mom` family). Up to 0.15.0 `return_to` was stored but ignored, so an
auxiliary electrode carried no current and the station potential did not
depend on its distance (finding 10.16, F01); a generator then compensated
with a manual counter-source, which now warns as a double count. The
current balance holds to round-off, and a two-terminal source equals the
superposition of two one-terminal injections
(`tests/test_two_terminal_sources.py`).

Sources are validated before any work is done (`World.check`, called by
every solve): unknown node names raise with a did-you-mean suggestion,
magnitudes must be finite, a current source must not be 0 A, a source whose
two terminals end in one cluster (a short circuit) raises, and so do voltage
sources without internal resistance that form a loop (finding 10.66; the
message names the sources of the loop).

## Voltage sources

A `VoltageSource` drives `attached_to` against `return_to` (or against
remote earth) with the voltage $U$ behind an internal impedance
$Z_q = R + j\omega L$ (`internal_resistance_ohm`, `internal_inductance_h`;
0 by default, the ideal source) — an instrument that drives a test
voltage, the source voltage of a fault loop behind the impedance of the
feeding transformer and line. The terminals then sit at
$\varphi_a - \varphi_r = U - Z_q I$. The current is solved for and
reported like the impressed current of a current source
(`result.sources[k].current`); `input_impedance()` $= U/I - Z_q$ is the
impedance the source sees, without its own $Z_q$. For one source the
answer is the 1 A current-source solve scaled by
$I = U/(Z_q + Z_\text{in})$ — the fault current $I_F = U_q/(Z_q + R_F + T)$
of a fault study, with $T$ the input impedance of the network — and a
voltage source saves that step; with several voltage-driven terminals in
one solve the scaling would need the transfer matrix of all of them, and
the result is their superposition (the input impedance of each is then an
apparent one). Before 2026.11 a series resistance was a junction at the
terminal, a lumped conductor to the electrode and the ideal source at the
junction; that pattern and `internal_resistance_ohm` give the same
solution to round-off. Up to 0.15.0 voltage sources were dropped with a
warning (finding 10.51); they are solved since 2026.11
([sources](../reference/api/sources.md)).

## Junctions

A junction (`create_junction(world, position=...)`) is a network node
without soil contact: conductors may start and end at it and sources may
attach to it, but it leaks no current. Junctions model cable joints of a
PEN conductor and the terminals of a measurement instrument. A junction
must reach the soil through its conductors; a dangling junction raises.
The solve reports its potential in `result.metadata["junction_potentials"]`.

Junctions replace the 5 cm pseudo-rods that the layout builder used for
PEN joints up to 0.15.0: those were small electrodes, and depending on the
penetration they leaked several per cent of the test current (finding
10.38).

## Three impedances

A result offers three impedances. They coincide for a single electrode
fed against remote earth and differ as soon as conductors connect the fed
cluster to other electrodes:

| Impedance | Definition | Accessor | Meaning |
|---|---|---|---|
| **input** | $Z_\text{in} = (\varphi_a - \varphi_r)/I_s$ | `result.input_impedance(source)` | what the feeding point sees: EPR per injected ampere for a return through remote earth; the driving-point impedance of the whole connected system |
| **cluster** | $Z_c = \varphi_c / I_{c,\text{leak}}$ | `result.cluster_impedance(name)` | the local spreading impedance of the fed cluster: its potential over the current it leaks itself |
| **electrode** | $Z_e = \varphi_e / I_{e,\text{leak}}$ | `result.grounding_impedance(name)` | one electrode of a cluster: its potential over its own leakage |

$\varphi_a$ and $\varphi_r$ are the potentials of the clusters the source
attaches to and returns at, $\varphi_c$ the length-weighted cluster
potential, $I_{c,\text{leak}}$ the sum of the leakage currents of the
cluster's electrodes. With the split factor $s = I_{c,\text{leak}}/I_s$,
the share of the injected current that leaks through the fed cluster,
$Z_\text{in} = s\,Z_c$ for a one-terminal source
(`tests/test_impedance_quantity.py::test_split_factor_links_input_and_cluster_impedance`).

In a TN network the station's cluster leaks only a part of the current —
the rest flows through the PEN conductors into the house foundations — so
the cluster impedance can exceed the input impedance by a large factor.
For an engineer the input impedance is the quantity of interest: it is
what a fall-of-potential measurement approaches and what a fault sees.
Since 2026.11 it is the default (`quantity="input"`) of `sweep`,
`convergence_study` and the rho-f exporters (`rho_f_from_field_result`,
`rho_f_standard_from_results`); `quantity="cluster"` and
`quantity="electrode"` reproduce the earlier semantics (`sweep` reported
the cluster impedance, the exporters the electrode impedance; findings
10.18, 10.19). Every fit records the quantity it represents.

## The fall-of-potential measurement

The classical measurement of a grounding impedance injects a current
between the earth electrode E and an auxiliary electrode H far away, and
reads the voltage between E and a probe S placed between them. In
homogeneous soil, for a compact electrode and a distant auxiliary
electrode, the reading equals the remote impedance when the probe sits at
61.8 % of the distance to H (Tagg 1964).
`tests/test_two_terminal_sources.py::test_fall_of_potential_follows_the_61_8_percent_rule`
checks that `groundfield` reproduces this rule.

The TN network generator builds the measurement as it is wired in the
field (finding 10.17):

- two junctions at the instrument, `meas_H` and `meas_S`, at the position
  of the substation and the height of the leads;
- the feed lead from `meas_H` to the auxiliary electrode, the probe lead
  from the probe electrode to `meas_S`;
- one two-terminal source `attached_to=<station>`, `return_to="meas_H"`;
- the reading `generators.measured_impedance(result)`
  $= (\varphi_E - \varphi_S)/I$.

Up to 0.15.0 the leads were wired to the station itself: the probe lead
short-circuited the instrument and the feed lead closed the loop through
the metal. With junctions, resistive leads do not change the reading
(`tests/test_measurement_loop.py::test_resistive_leads_do_not_change_the_reading`)
and inductive leads shift it by their coupling without shorting it.
Leads that run towards the same side are laid side by side (touching),
because two coincident conductors have no finite mutual inductance. The
leads run horizontally at `MeasurementLeadConfig.depth_m` below the
surface; a depth of 0 places them 0.1 m above ground.

`input_impedance` of the measurement source is the two-pole impedance
between E and H — the loop impedance including the auxiliary electrode
and the feed lead — not the reading; `measured_impedance` is the reading.

## Validity

- The source model is an impressed current (ideal) or an impressed voltage
  behind a lumped internal impedance $R + j\omega L$ (no frequency
  dependence of $R$ and $L$ themselves; a source with a frequency-dependent
  impedance needs one solve per frequency with its values).
- Two-terminal sources, junctions and voltage sources are supported by the
  image family and the `mom` family (registry capabilities
  `two_terminal_sources`, `junctions`, `voltage_sources`); `fem` ignores
  `return_to` (with a warning) and rejects junctions and voltage sources.
- Measured readings include the coupling of the leads only if the leads
  are modelled as inductive conductors (`inductance_model="neumann"`).

## Literature

- Tagg, G. F. (1964). *Earth Resistances*. George Newnes, London (the
  61.8 % rule of the fall-of-potential method).
- IEEE Std 81-2012. *IEEE Guide for Measuring Earth Resistivity, Ground
  Impedance, and Earth Surface Potentials of a Grounding System*.
- EN 50522:2022. *Earthing of power installations exceeding 1 kV a.c.*
  (earth potential rise and touch voltage as the quantities of the
  safety assessment).

Related: [kernels](kernels.md), [engine family](engines/index.md). API:
`create_source`, `create_junction`, `FieldResult.input_impedance`,
`generators.measured_impedance`.
