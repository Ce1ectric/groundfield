# Concepts

This page explains, in plain words, the objects you build a computation
from, what happens when you call `solve`, and what a result contains. Read
it once to get the overall picture; the [quickstart](../getting-started/quickstart.md)
shows the same in code, and the [engines page](engines/index.md) explains
the numerical methods.

## Position in the software family

```text
  groundmeas   ──▶   groundinsight   ◀──   groundfield
  (measurements)     (reduced network       (field model,
                      model)                 reference)
```

`groundfield` is the field-theoretical reference of the family. It
computes the potential field of earthing systems in layered soil from their
geometry and material data and compresses the solution into reduced
models — impedance functions $Z(\rho, f)$, multiport matrices — that
`groundinsight` uses as building blocks of a network model, and against
which `groundmeas` measurements can be interpreted. The field model is a
reference, not the end product: every solution exposes the quantities the
reduction needs.

## The data model

A computation separates the **physics** (what is in the ground) from the
**numerics** (how it is solved). The physics lives in a `World`, the
numerics in an `Engine`; `world.solve(engine)` returns a `FieldResult`.

| Object | What it is | Created with |
|---|---|---|
| soil model | horizontal layers of constant resistivity: `HomogeneousSoil(rho)`, `TwoLayerSoil(rho_1, rho_2, h_1)`, `MultiLayerSoil(layers)`; depth $z$ positive into the soil | the class itself |
| `World` | the physical model: one soil, and the electrodes, junctions, conductors and sources in it | `create_world` |
| electrode | a thin wire in contact with the soil: rod, ring, strip, polyline, star, mesh, grid; optionally encased in concrete ([concrete shells](concrete-shells.md)) | `create_electrode` |
| junction | a node without soil contact: a cable joint, an instrument terminal | `create_junction` |
| conductor | a branch between two nodes: an ideal bond, or a cable with resistance, optionally leaking into the soil along its length and optionally inductive | `create_conductor` |
| source | a current or voltage source that drives a node against a second node or against remote earth ([sources and measurement](sources-and-measurement.md)) | `create_source` |
| `Engine` | the numerics: backend, frequencies, segment length, source kernel, earth-return model, backend options | `create_engine` |
| `FieldResult` | potentials, leakage currents and impedances per frequency, the sources for evaluating the field, and the metadata that reproduces the solve | `world.solve(engine)` |

Two ideas recur everywhere:

- **Cluster.** Electrodes joined by ideal conductors (a conductor without a
  `cross_section`) are one galvanic cluster with one potential. A cable
  with a cross-section is a branch with its own resistance between two
  clusters.
- **Node.** Electrodes and junctions are the nodes of the conductor
  network. Conductors connect nodes; sources inject current at a node.

## What happens in a solve

```text
world.solve(engine)
 1. check        the backend's capabilities (soil layers, kernel, concrete shells)
                 and the resource limit, before anything is allocated
 2. discretise   every wire into straight segments of at most segment_length,
                 split at layer interfaces
 3. soil         the reaction matrix: the potential every segment's current
                 causes at every other segment (Green's function of the soil)
 4. network      the nodal equations: soil matrix + Kirchhoff's current law at
                 every node + Ohm's law along every conductor + the sources
 5. result       one solution per frequency: potentials, currents, impedances,
                 point sources for the field, metadata
```

Below 1 kHz the soil behaves as a pure conductor: no displacement current,
and a skin depth large against a low-voltage network. Step 3 is therefore
the stationary current field of the electrodes in a horizontally layered
half-space; it does not depend on frequency and is computed once per world.
Frequency enters in step 4 through the conductors: their resistance
(optionally with skin effect), partial inductances and the earth-return
impedance make the network equations complex. A purely galvanic world is
solved once and reused for every frequency; a world with inductive
conductors is solved once per frequency, which is cheap against step 3.
The [engines page](engines/index.md#network-equations) gives the equations,
the [earth-return page](earth-return.md) the conductor impedance.

The engines differ in steps 2 and 3: how many unknown currents a wire
carries (one per straight wire, or one per segment) and how the Green's
function is evaluated (closed-form image series, numerical Sommerfeld
quadrature, or a finite-element volume mesh instead).

## What a result contains

| Access | Meaning | Unit |
|---|---|---|
| `result.frequencies` | the frequencies of the engine, in their order | Hz |
| `result.input_impedance(source)` | the voltage at a source over its current: what a fault or an earth tester sees | Ω |
| `result.cluster_impedance(electrode)` | the potential of a cluster over the current it leaks itself | Ω |
| `result.grounding_impedance(electrode)` | the potential of a cluster over one electrode's leakage current | Ω |
| `result.electrode_potentials[name]` | the potential of an electrode against remote earth (the EPR for the source current) | V |
| `result.electrode_currents[name]` | the current an electrode leaks into the soil | A |
| `result.potential(points)` | the potential at any points in or on the soil (homogeneous and two-layer soil) | V |
| `result.metadata` | backend, effective backend, kernel, version, segments, solve time, series controls, provenance | — |

Every quantity is a complex RMS phasor with one value per frequency
([units and conventions](../reference/units-and-conventions.md)). Touch and
step voltages, current sharing, plots, sweeps and reduced models are
functions that take a result: `touch_voltage`, `step_voltage`,
`cluster_current_balance`, `plot_surface_potential`, `sweep`,
`fit_rho_f_standard` and others ([postprocess](../reference/api/postprocess.md)).
`save_result` and `load_result` store a result with its provenance.

The three impedances coincide for a single electrode and differ in a
network: the input impedance is what the feeding point sees (and what a
measurement approaches), the cluster impedance is the local spreading
impedance of the fed cluster. The input impedance is the default of the
sweeps and fits since 2026.11; the
[sources page](sources-and-measurement.md#three-impedances) has the
definitions.

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
points the reduced model is the multiport matrix of the earthing systems
(`solve_mutual_matrix`): its entry $Z_{ij}$ is the potential of cluster $i$
per ampere leaking from cluster $j$. Together with the branch admittances
$\mathbf{Y}_L$ of the network and $\mathbf{Y}_G = \mathbf{Z}^{-1}$ it forms
the nodal system $(\mathbf{Y}_L + \mathbf{Y}_G)\,\mathbf{u} = \mathbf{i}$
of the AP1 catalogue, with the node voltages $\mathbf{u}$ and the injected
currents $\mathbf{i}$.

## Typical questions

`groundfield` is built for the dissertation questions on interconnected
earthing systems of medium- and low-voltage networks:

1. How strongly does a remote current injection influence an earthing
   measurement, and how is the measurement loop to be modelled?
2. How important are coupling and return-path effects below 1 kHz?
3. Which robust statements hold for typical distribution-network
   configurations, and how do they reduce to a small set of parameters?
4. Which touch voltages arise in the fault case, and over which spectrum?

The [tutorials](../tutorials/index.md) walk through these computations;
this explanation section gives the theory and the evidence behind them.
