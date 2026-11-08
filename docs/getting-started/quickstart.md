# Quickstart

A substation ring earth and the rod earth of a house 40 m away, joined
by a PEN conductor, in two-layer soil; a current of 1 A is injected at
the substation and returns through remote earth.

```python
import groundfield as gf

world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=2.0))
gf.create_electrode(world, "ring", name="substation", center=(0.0, 0.0, 0.8), radius=3.0)
gf.create_electrode(world, "rod", name="house", position=(40.0, 0.0, 0.0), length=3.0)
gf.create_conductor(world, name="pen", start="substation", end="house", cross_section=95e-6)
gf.create_source(world, attached_to="substation", magnitude=1.0)

result = world.solve(gf.create_engine(frequencies=[50.0]))
print(result.input_impedance()[0])                # (11.2699+0j) Ω
print(result.cluster_impedance("substation")[0])  # (14.1046+0j) Ω
```

Line by line:

- **Soil.** `TwoLayerSoil` is a top layer of resistivity
  $\rho_1 = 100\,\Omega\,\mathrm{m}$ and thickness $h_1 = 2\,\mathrm{m}$
  over a half-space of $\rho_2 = 300\,\Omega\,\mathrm{m}$. All inputs are
  SI units (m, Ω·m, A, Hz).
- **Electrodes.** Coordinates are $(x, y, z)$ in metres with $z$ the
  depth, positive downwards: the ring lies 0.8 m deep, the rod runs from
  the surface to 3 m. The wire radius defaults to 5 mm.
- **Conductor.** A 95 mm² copper PEN (`cross_section` in m²) is a branch
  with resistance $R = \rho_\text{Cu}\,\ell/A$ between the two
  electrodes. Without `cross_section` it would be an ideal connection
  and both electrodes would form one galvanic cluster.
- **Source.** 1 A RMS phasor at the substation; without `return_to` the
  current returns through remote earth (the potential reference,
  $\varphi \to 0$ far away).
- **Engine.** The numerics: here only the frequency list. The default
  backend `image` forwards a two-layer soil to `image_2layer` (the
  Tagg/Sunde image series) with the exact line-source kernel; segments
  are at most 0.5 m long.

## Three impedances of one solve

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
| `cluster_impedance(electrode)` | $\varphi_c / \sum_{e \in c} I_e$ | the local earthing system: one galvanic cluster (electrodes joined by ideal conductors) and only the current it leaks itself |
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

## Check the result with a second method

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
print(fine.input_impedance()[0])   # (11.2759+0j) Ω, +0.05 % at half the segment length
```

`mom` solves for the segment currents by point matching with midpoint
point sources, `image_2layer` imposes uniform line currents per segment
with the exact line-source kernel; agreement within about 1 % is
expected (the point kernel is known to read 1–2 % low,
see [source kernels](../explanation/kernels.md)).

??? info "Values asserted by the test suite"

    The code blocks on this page are executed by the documentation tests;
    these assertions keep the numbers quoted above in step with the code.

    ```python
    assert abs(Z_in - 11.2699) < 5e-4 and abs(Z_substation - 14.1046) < 5e-4
    assert abs(share.real - 0.799) < 1e-3
    assert abs(1 / (1 / Z_substation + 1 / Z_house) - Z_in) < 1e-3
    assert report.is_consistent and max(report.deviations.values()) < 0.012
    assert abs(fine.input_impedance()[0] - 11.2759) < 5e-4
    ```

## Where next

- [Tutorials](../tutorials/index.md): from this first impedance to
  layered soil, frequency response, `rho-f` export, TN networks,
  measurements and touch voltages.
- [Factories](../reference/api/factories.md): every `create_*` function
  and its parameters.
- [Units and conventions](../reference/units-and-conventions.md):
  coordinates, phasors, signs.
