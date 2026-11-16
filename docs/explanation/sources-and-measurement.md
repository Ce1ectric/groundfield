# Sources, junctions and the measurement loop

This page explains how `groundfield` feeds a grounding system — current and
voltage sources with two terminals, nodes without soil contact
(junctions) — which of the three impedances of a result to read, and how
the fall-of-potential measurement is wired. Read it before you set up a
fault or a measurement study, or when two impedances of one result
disagree.

A grounding system is never fed in isolation. A fault current enters at
one point and returns through a remote source; a measurement instrument
injects between the earth electrode and an auxiliary electrode and reads
the voltage against a probe; cable joints and instrument terminals are
nodes without soil contact.

## Mathematical and physical context

The engines solve a linear network: the soil multiport of the electrodes,
the branches of the conductors and the nodes that connect them (see the
[engine family](engines/index.md#network-equations)). Current sources
enter as impressed node currents. Voltage sources enter as a fixed
potential difference between their terminals whose current is an
additional unknown (modified nodal analysis). Every quantity is therefore
linear in the source values, and results of several sources superpose.

The potential reference is **remote earth**: $\varphi \to 0$ far away from
all electrodes. The potential of an electrode is therefore its earth
potential rise (EPR) against remote earth, and a voltage between two
points — a touch voltage, a measured reading — is a difference of such
potentials.

## Two-terminal sources

A current source has two terminals:

- `attached_to` — the node where the current $I$ enters
  (an electrode, a junction or, through its cluster, a bonded group);
- `return_to` — the node where it leaves again; `None` means the return
  through remote earth.

The solver injects $+I$ at `attached_to` and $-I$ at `return_to`
(`groundfield.solver._pipeline.inject`, shared by every backend). The
current balance holds to round-off, and a two-terminal source equals the
superposition of two one-terminal injections
(`tests/test_two_terminal_sources.py`). A manual counter-source at the
return electrode, as generators used before, now warns as a double count.

Sources are validated before any work is done (`World.check`, called by
every solve): unknown node names raise with a did-you-mean suggestion,
magnitudes must be finite, a current source must not be 0 A, a source
whose two terminals end in one cluster (a short circuit) raises, and so do
voltage sources without internal resistance that form a loop (the message
names the sources of the loop).

??? note "Measurements and audit trail"

    Up to 0.15.0 `return_to` was stored but ignored, so an auxiliary
    electrode carried no current and the station potential did not depend
    on its distance (finding 10.16, F01). The validation of voltage-source
    loops is finding 10.66.

## Voltage sources

A `VoltageSource` drives `attached_to` against `return_to` (or against
remote earth) with the voltage $U$ behind an internal impedance
$Z_q = R + j\omega L$ (`internal_resistance_ohm`, `internal_inductance_h`;
0 by default, the ideal source) — an instrument that drives a test
voltage, or the source voltage of a fault loop behind the impedance of the
feeding transformer and line. With the potentials $\varphi_a$ and
$\varphi_r$ of its two terminals and its current $I$ the source imposes

$$
\varphi_a - \varphi_r = U - Z_q I ,
$$

the terminal voltage is the source voltage less the drop across the
internal impedance. The current is solved for and reported like the
impressed current of a current source (`result.sources[k].current`);
`input_impedance()` $= U/I - Z_q$ is the impedance the source sees,
without its own $Z_q$.

For one source the answer is the 1 A current-source solve scaled by

$$
I = \frac{U}{Z_q + Z_\text{in}} ,
$$

$Z_\text{in}$ the input impedance of the network at the source's
terminals. This is the fault current $I_F = U_q/(Z_q + R_F + T)$ of a
fault study, with $U_q$ the source voltage, $R_F$ a fault resistance and
$T$ the input impedance of the network, and a voltage source saves that
step. With several voltage-driven terminals in one solve the scaling would
need the transfer matrix of all of them; the result is their
superposition, and the input impedance of each is then an apparent one. A
series resistance written as a junction at the terminal, a lumped
conductor to the electrode and the ideal source at the junction gives the
same solution as `internal_resistance_ohm` to round-off
([sources](../reference/api/sources.md)).

??? note "Measurements and audit trail"

    Up to 0.15.0 voltage sources were dropped with a warning (finding
    10.51); they are solved since 2026.11. Before 2026.11 the junction
    pattern above was the only way to give a source a series resistance.

## Junctions

A junction (`create_junction(world, position=...)`) is a network node
without soil contact: conductors may start and end at it and sources may
attach to it, but it leaks no current. Junctions model cable joints of a
PEN conductor and the terminals of a measurement instrument. A junction
must reach the soil through its conductors; a dangling junction raises.
The solve reports its potential in `result.metadata["junction_potentials"]`.

??? note "Measurements and audit trail"

    Junctions replace the 5 cm pseudo-rods that the layout builder used for
    PEN joints up to 0.15.0: those were small electrodes, and depending on
    the penetration they leaked several per cent of the test current
    (finding 10.38).

## Three impedances

A result offers three impedances. They coincide for a single electrode
fed against remote earth and differ as soon as conductors connect the fed
cluster to other electrodes:

| Impedance | Definition | Accessor | What it is |
|---|---|---|---|
| **input** | $Z_\text{in} = (\varphi_a - \varphi_r)/I_s$ | `result.input_impedance(source)` | the driving-point impedance of the whole connected system |
| **cluster** | $Z_c = \varphi_c / I_{c,\text{leak}}$ | `result.cluster_impedance(name)` | the local spreading impedance of the fed cluster |
| **electrode** | $Z_e = \varphi_e / I_{e,\text{leak}}$ | `result.grounding_impedance(name)` | one electrode of a cluster over its own leakage |

Here $I_s$ is the source current, $\varphi_a$ and $\varphi_r$ are the
potentials of the clusters the source attaches to and returns at,
$\varphi_c$ is the length-weighted cluster potential and
$I_{c,\text{leak}}$ the sum of the leakage currents of the cluster's
electrodes; $\varphi_e$ and $I_{e,\text{leak}}$ are the potential and the
leakage of one electrode. For a return through remote earth the input
impedance is the EPR per injected ampere.

The **split factor** $s = I_{c,\text{leak}}/I_s$ is the share of the
injected current that leaks through the fed cluster itself. For a
one-terminal source $\varphi_a = \varphi_c$ and $\varphi_r = 0$, so

$$
Z_\text{in} = \frac{\varphi_c}{I_s} = \frac{I_{c,\text{leak}}}{I_s}\,
\frac{\varphi_c}{I_{c,\text{leak}}} = s\,Z_c
$$

(`tests/test_impedance_quantity.py::test_split_factor_links_input_and_cluster_impedance`).

In a TN network the station's cluster leaks only a part of the current —
the rest flows through the PEN conductors into the house foundations — so
$s < 1$ and the cluster impedance can exceed the input impedance by a
large factor. For an engineer the input impedance is the quantity of
interest: it is what a fall-of-potential measurement approaches and what a
fault sees. It is the default (`quantity="input"`) of `sweep`,
`convergence_study` and the rho-f exporters (`rho_f_from_field_result`,
`rho_f_standard_from_results`); `quantity="cluster"` and
`quantity="electrode"` give the other two. Every fit records the quantity
it represents.

??? note "Measurements and audit trail"

    The input impedance is the default since 2026.11 (findings 10.18,
    10.19). Before, `sweep` reported the cluster impedance and the
    exporters the electrode impedance; `quantity="cluster"` and
    `quantity="electrode"` reproduce those semantics.

## The fall-of-potential measurement

The classical measurement of a grounding impedance injects a current $I$
between the earth electrode E and an auxiliary electrode H at the
distance $D$, and reads the voltage between E and a probe S placed between
them at the distance $x$ from E. For a hemispherical electrode E of
radius $r_E$ in homogeneous soil, with H far enough away to act as a point
source, the reading is

$$
\frac{\varphi_E - \varphi_S}{I} = \frac{\rho}{2\pi}\Bigl[\frac{1}{r_E}
- \frac{1}{D} - \frac{1}{x} + \frac{1}{D - x}\Bigr] ,
$$

the true resistance $\rho/(2\pi r_E)$ plus the potential that the return
current at H and the injected current at E impose on the probe. The
correction vanishes when $1/(D - x) = 1/x + 1/D$, that is for
$(x/D)^2 + x/D - 1 = 0$ or $x/D = (\sqrt5 - 1)/2$: the reading equals the
remote impedance when the probe sits at 61.8 % of the distance to H
(Tagg 1964).
`tests/test_two_terminal_sources.py::test_fall_of_potential_follows_the_61_8_percent_rule`
checks that `groundfield` reproduces this rule. The finite size of a real
earthing system leaves a residual error at 61.8 %, and for an extended
system such as a whole LV network the electrical centre is not known; the
[fall-of-potential tutorial](../tutorials/T07_fall_of_potential_measurement.ipynb)
shows how far the auxiliary electrode must be.

The TN network generator builds the measurement as it is wired in the
field:

- two junctions at the instrument, `meas_H` and `meas_S`, at the position
  of the substation and the height of the leads;
- the feed lead from `meas_H` to the auxiliary electrode, the probe lead
  from the probe electrode to `meas_S`;
- one two-terminal source `attached_to=<station>`, `return_to="meas_H"`;
- the reading `generators.measured_impedance(result)`
  $= (\varphi_E - \varphi_S)/I$.

With junctions, resistive leads do not change the reading
(`tests/test_measurement_loop.py::test_resistive_leads_do_not_change_the_reading`)
and inductive leads shift it by their coupling without shorting it. Leads
that run towards the same side are laid side by side (touching), because
two coincident conductors have no finite mutual inductance. The leads run
horizontally at `MeasurementLeadConfig.depth_m` below the surface; a depth
of 0 places them 0.1 m above ground.

`input_impedance` of the measurement source is the two-pole impedance
between E and H — the loop impedance including the auxiliary electrode
and the feed lead — not the reading; `measured_impedance` is the reading.

??? note "Measurements and audit trail"

    Up to 0.15.0 the leads were wired to the station itself: the probe lead
    short-circuited the instrument and the feed lead closed the loop
    through the metal (finding 10.17).

## Validity

- The source model is an impressed current (ideal) or an impressed voltage
  behind a lumped internal impedance $R + j\omega L$ (no frequency
  dependence of $R$ and $L$ themselves; a source with a frequency-dependent
  impedance needs one solve per frequency with its values).
- Two-terminal sources, junctions and voltage sources are supported by
  every backend — the image family, the `mom` family and `fem` (registry
  capabilities `two_terminal_sources`, `junctions`, `voltage_sources`).
- Measured readings include the coupling of the leads only if the leads
  are modelled as inductive conductors (`inductance_model="neumann"`).

??? note "Corrected statement"

    Earlier versions of this page said that `fem` ignores `return_to` and
    rejects junctions and voltage sources. That held for the
    equivalent-hemisphere `fem` before user decision Q37; the conforming
    `fem` runs the shared network pipeline and its registry entry declares
    all three capabilities.

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
