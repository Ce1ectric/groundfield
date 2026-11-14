# Concepts

## Position in the software family

```
  groundmeas   ──▶   groundinsight   ◀──   groundfield
  (measurements)     (reduced network       (field model,
                      model)                 reference)
```

`groundfield` is the field-theoretical reference of the family. It
computes the potential field of grounding systems in layered soil from
their geometry and material data and compresses the solution into reduced
models — impedance functions $Z(\rho, f)$, multiport matrices — that
`groundinsight` uses as building blocks of a network model, and against
which `groundmeas` measurements can be interpreted. The field model is a
reference, not the end product: every solution exposes the quantities the
reduction needs.

## What is computed

Below 1 kHz the soil behaves as a conductor without displacement current
and with a skin depth large against a low-voltage network, so the
galvanic field is the stationary current field of the electrodes in a
horizontally layered half-space, solved once per world and reused for
every frequency. Frequency enters through the conductors: their
resistance (optionally with skin effect), partial inductances and the
earth-return impedance make the network equations complex. The
[engine family](engines/index.md) explains the equations, the
[earth-return page](earth-return.md) the conductor impedance.

A computation has four parts:

| Object | Role |
|---|---|
| `World` | the physical configuration: soil model, electrodes, conductors, junctions, sources |
| soil model | `HomogeneousSoil(rho)`, `TwoLayerSoil(rho_1, rho_2, h_1)`, `MultiLayerSoil(layers)`: horizontal layers of constant resistivity, $z$ positive into the soil |
| `Engine` | the numerical configuration: backend, frequencies, segment length, kernel, earth model, series controls, backend options |
| `FieldResult` | potentials, leakage currents and impedances per frequency, the segment sources for field evaluation, and the metadata (kernel, series, provenance) needed to reproduce it |

Electrodes are thin wires in contact with the soil (rods, rings, strips,
polylines, stars, meshes, grids), optionally encased in concrete
([concrete shells](concrete-shells.md)). Conductors connect nodes: ideal
bonds merge electrodes into one cluster; finite conductors are lumped or
distributed branches, galvanic (leaking along their length) or isolated,
optionally inductive. Junctions are nodes without soil contact. Sources
inject current at a node and take it back at another node or at remote
earth ([sources and measurement](sources-and-measurement.md)).

## Modelling assumptions

- **Quasi-static field** for $f \lesssim 1$ kHz: stationary current flow in
  the soil, no displacement current; the soil permittivity is inert.
- **Horizontally layered soil** with piecewise constant resistivity; layer
  stacks are canonicalised (equal adjacent layers merged).
- **Thin wires**: the wire radius is small against the segment length and
  the distance to other wires; the current flows on the axis, the
  potential is observed on the surface.
- **Current model**: uniform leakage per unit length per straight wire of
  an electrode (and layer) in the image family — impedance-grade, not
  field-grade near grids and stars ([`image`](engines/image.md#field-quantities-and-the-uniform-current-ansatz));
  one current per segment in the `mom` family.
- **Remote earth** as the potential reference.
- **Conductor impedance** from partial inductances and an earth-return
  correction; the Sommerfeld/Pollaczek kernel by default, Carson's series
  as a fast approximation for long parallel lines.

## The impedances of a result

A result reports three impedances — input, cluster and electrode
impedance — that coincide for one electrode and differ in a network:
the input impedance is what the feeding point sees (and what a
measurement approaches), the cluster impedance is the local spreading
impedance of the fed cluster. The input impedance is the default of the
sweeps and fits since 2026.11; the
[sources page](sources-and-measurement.md#three-impedances) has the
definitions.

## The reduced model

The objective of a field computation is not only a number but a model
that a network calculation can use. For one feeding point the reduced
model is the impedance as a function of soil resistivity and frequency.
The standard form exported to `groundinsight` is

$$
Z(\rho, f) = k_0 + k_1\rho + (k_2 + jk_3)\,f + (k_4 + jk_5)\,f\rho ,
$$

with the resistance $k_0$ that does not scale with $\rho$ (the share of a
lower soil layer, or the series resistance of conductors), the spreading
resistance $k_1\rho$, the soil-independent loop terms $k_2 f$, $k_3 f$
and Carson-type earth-return terms $k_4 f\rho$, $k_5 f\rho$. It is fitted
by weighted linear least squares to a family of solves that spans both
$\rho$ and $f$, with relative weights by default; `max_relative` reports
the worst pointwise error and a fit above 5 % warns. Where the six-term
form cannot follow the data (strongly layered soils), a vector fit of the
frequency response per $\rho$ is the alternative. For several feeding
points the reduced model is the multiport matrix of the grounding system
(`solve_mutual_matrix`), which together with the branch admittances of
the network forms the system $(Y_L + Y_G)\,u = i$ of the AP1 catalogue.

## Typical questions

`groundfield` is built for the dissertation questions on interconnected
grounding systems of medium- and low-voltage networks:

1. How strongly does a remote current injection influence a grounding
   measurement, and how is the measurement loop to be modelled?
2. How important are coupling and return-path effects below 1 kHz?
3. Which robust statements hold for typical distribution-network
   configurations, and how do they reduce to a small set of parameters?
4. Which touch voltages arise in the fault case, and over which spectrum?

The [tutorials](../tutorials/index.md) walk through these computations;
this explanation section gives the theory and the evidence behind them.
