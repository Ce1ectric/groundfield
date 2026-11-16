# groundfield

**Numerical field computation for grounding systems.**

`groundfield` is an open-source Python package that computes the electric
field of earthing systems in layered soil: electrodes of any shape, the
conductors that connect them (PEN conductors, cable screens, measurement
leads) with their inductive coupling and earth return, and the current
and potential distribution that follows from a fault or a test current.
This page says what the package computes, shows a first example and points
you to the part of the documentation you need.

## Who it is for

Engineers and researchers who plan, assess or measure earthing systems in
medium- and low-voltage networks and need numbers they can trace back to
physics: the earthing impedance of interconnected systems, the influence of
soil layering and of the measurement set-up, touch and step voltages, and
the frequency response below 1 kHz. You should know the engineering side
(EN 50522, IEEE Std 80); the documentation explains the code and the
numerical methods.

## What it computes

| Quantity | Typical use |
|---|---|
| Input, cluster and electrode impedance $Z(f)$ | earthing resistance, fault and test-current loops |
| Potential $\varphi(\mathbf{r}, f)$ in and on the soil | potential profiles, earth potential rise (EPR) |
| Touch and step voltage, EN 50522 limits | safety assessment, including fault-current spectra |
| Current sharing between electrodes and conductors | PEN and cable-screen currents, split factors |
| `rho-f` model $Z(\rho, f)$ | reduced models for `groundinsight` (`BusType`) |

The physics is the quasi-static conduction problem
$\nabla\cdot(\sigma\nabla\varphi) = 0$ in horizontally layered soil: current
is conserved in the soil of conductivity $\sigma$, and the potential
$\varphi$ follows from the currents the electrodes leak. It is coupled to
the conductor network by Kirchhoff's laws and, for conductors with
inductance, by the Neumann inductance with a Carson or Sommerfeld
earth-return correction. The frequency range is limited to
$f < 1\,\mathrm{kHz}$ on purpose: it keeps the model identifiable from
measurements.

The field solution is a **reference, not the end product**: every solve is
instrumented so that a reduced, interpretable equivalent model (the
`rho-f` model of an earthing system) can be derived from it and handed to
`groundinsight`.

## A first example

The earthing impedance and the earth potential rise of a 20 m × 20 m grid
in two-layer soil, for an earth-fault current of 100 A:

```python
import groundfield as gf

soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=2.0)        # Ω·m, Ω·m, m
world = gf.create_world(soil=soil)
gf.create_electrode(world, "grid_mesh", name="grid", corner=(-10.0, -10.0, 0.5), size=(20.0, 20.0), n_x=4, n_y=4)
gf.create_source(world, attached_to="grid", magnitude=100.0)       # A

result = world.solve(gf.create_engine(frequencies=[50.0]))
print(f"Z = {result.input_impedance()[0].real:.2f} Ω")            # Z = 4.80 Ω
print(f"EPR = {abs(result.electrode_potentials['grid'][0]):.0f} V")  # EPR = 480 V
```

<!-- doctest-check
assert abs(result.input_impedance()[0] - 4.8005) < 5e-4
assert abs(abs(result.electrode_potentials["grid"][0]) - 480.05) < 0.05
-->

A `World` holds the physics (soil, electrodes, conductors, sources), an
`Engine` the numerics (backend, frequencies, discretisation). Coordinates
are in metres with the depth $z$ positive downwards. The
[quickstart](getting-started/quickstart.md) continues this example with
touch voltages, a potential profile and a plot.

## Where to go next

| If you want to … | read |
|---|---|
| install the package and run a first solve | [Installation](getting-started/installation.md), [Quickstart](getting-started/quickstart.md) |
| learn the package step by step | [Tutorials](tutorials/index.md): ten executed notebooks, from a single rod to whole low-voltage networks |
| see complete engineering studies | [Examples](examples/index.md): a secondary substation with its low-voltage network, a single-family house, the mesh grid of a high-voltage substation |
| solve a specific task | [How-to guides](how-to/index.md): sweeps, plots, touch voltages, export, large studies, warnings |
| choose a numerical engine and know its limits | [Choosing an engine](explanation/engines/index.md) |
| understand the methods, their equations and their accuracy | [Explanation](explanation/index.md) |
| look up a function, a unit, a warning or a German term | [Reference](reference/index.md) |
| see what changed and which numbers moved | [Changelog](changelog.md) |

## How a computation runs

```text
soil model ──┐
geometry  ───┼──▶ World ──▶ Engine.solve ──▶ FieldResult ──┬──▶ potentials, impedances,
conductors ──┤                (backend,                     │    touch / step voltages
sources  ────┘                 frequencies,                 └──▶ rho-f fit ──▶ groundinsight
                               kernel)                               (BusType JSON)
```

The same world can be solved by several engines, which is how results are
cross-checked. The default engine `image` covers homogeneous and two-layer
soil; `mom`, `mom_sommerfeld` and the experimental `fem` cover what it
cannot ([choosing an engine](explanation/engines/index.md),
[concepts](explanation/concepts.md)).

## Software family

```text
groundmeas  ──▶  groundinsight  ◀──  groundfield
(measurements)   (reduced network    (field model,
                  model)              PDE reference)
```

`groundfield` is the field-model member of the family. It exports reduced
models as `groundinsight` `BusType` formulas (JSON file or live object, see
[IO and groundinsight](reference/api/io.md)), where they can be reconciled
with measurements from `groundmeas`.

## License and citation

`groundfield` is released under the MIT license. If you use it in
academic work, please cite it with the metadata in `CITATION.cff` of
the repository.
