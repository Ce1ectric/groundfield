# groundfield

**Numerical field computation for grounding systems.**

`groundfield` is an open-source Python package that computes the
electric field of grounding systems in layered soil: electrodes of any
shape, the conductors that connect them (PEN conductors, cable
screens, measurement leads) with their inductive coupling and earth
return, and the current and potential distribution that follows from a
fault or a test current. It is the field-model member of the software
family `groundmeas` (measurements), `groundinsight` (reduced network
model) and `groundfield`.

The field solution is a **reference, not the end product**: every
solve is instrumented so that a reduced, interpretable equivalent model
(the `rho-f` model of an earthing system) can be derived from it and
handed to `groundinsight`.

## Who it is for

Engineers and researchers who plan, assess or measure earthing systems
in medium- and low-voltage distribution networks and need numbers they
can trace back to physics: grounding impedance of interconnected
systems, the influence of soil layering and of the measurement set-up,
touch and step voltages, and the frequency response below 1 kHz.

## What it computes

| Quantity | Typical use |
|---|---|
| Input, cluster and electrode impedance $Z(f)$ | earthing resistance, fault and test-current loops |
| Potential $\varphi(\mathbf{r}, f)$ in and on the soil | potential profiles, earth potential rise (EPR) |
| Touch and step voltage, EN 50522 limits | safety assessment, including fault-current spectra |
| Current sharing between electrodes and conductors | reduction factors, PEN and cable-screen currents |
| `rho-f` model $Z(\rho, f)$ | reduced models for `groundinsight` (`BusType`) |

The physics is the quasi-static conduction problem
$\nabla\cdot(\sigma\nabla\varphi) = 0$ in horizontally layered soil,
coupled to the conductor network by Kirchhoff's laws and, for
conductors with inductance, by the Neumann inductance with a Carson or
Sommerfeld earth-return correction. The frequency range is limited to
$f < 1\,\mathrm{kHz}$ on purpose: it keeps the model identifiable from
measurements.

## Workflow

```text
soil model ──┐
geometry  ───┼──▶ World ──▶ Engine.solve ──▶ FieldResult ──┬──▶ potentials, impedances,
conductors ──┤                (backend,                     │    touch / step voltages
sources  ────┘                 frequencies,                 └──▶ rho-f fit ──▶ groundinsight
                               kernel)                               (BusType JSON)
```

A `World` holds the physics (soil, electrodes, conductors, sources), an
`Engine` the numerics (backend, frequencies, discretisation). The same
world can be solved by several engines, which is how results are
cross-checked.

## Where to start

| If you want to … | read |
|---|---|
| install the package and run a first solve | [Installation](getting-started/installation.md), [Quickstart](getting-started/quickstart.md) |
| learn the package step by step | [Tutorials](tutorials/index.md) (executed notebooks) |
| solve a specific task | [How-to guides](how-to/index.md) |
| understand the methods, their equations and validity | [Explanation](explanation/index.md) |
| look up a function, a unit, a warning or a German term | [Reference](reference/index.md) |
| see what changed and which numbers moved | [Changelog](changelog.md) |

## Software family

```text
groundmeas  ──▶  groundinsight  ◀──  groundfield
(measurements)   (reduced network    (field model,
                  model)              PDE reference)
```

`groundfield` exports reduced models as `groundinsight` `BusType`
formulas (JSON file or live object, see
[IO and groundinsight](reference/api/io.md)), where they can be
reconciled with measurements from `groundmeas`.

## License and citation

`groundfield` is released under the MIT license. If you use it in
academic work, please cite it with the metadata in `CITATION.cff` of
the repository.
