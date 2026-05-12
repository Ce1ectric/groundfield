# groundfield

**Numerical field computation for grounding systems.**

[![Python versions](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`groundfield` is an open-source Python package for the physical
reference modelling of networked grounding systems. Within the
`groundmeas` / `groundinsight` / `groundfield` software family,
`groundfield` covers the field-theoretical side: soil models,
electrode geometries, conductors and their couplings are formulated as
a 3-D problem in the soil and solved numerically. The results are
field profiles, potential curves, current distributions, and — most
importantly — reduced equivalent models (`rho-f` model) that, via
vector fitting and a SymPy-compatible export, can be handed over
directly to `groundinsight` as a `BusType`. A world-generator
framework (`groundfield.generators`, ADR-0009) parameterises the
AP1 TN-Ortsnetz study (5–200 single-family houses, two-layer soil,
stochastic electrode mixes) and produces fully-wired `World`
instances ready for any of the eight integral / FEM backends.

- **Documentation**: <https://ce1ectric.github.io/groundfield/>
- **Source code**: <https://github.com/Ce1ectric/groundfield>
- **Issue tracker**: <https://github.com/Ce1ectric/groundfield/issues>

## Position within the software family

```
  groundmeas   ──▶   groundinsight   ◀──   groundfield
  (measurement)      (reduced network            (field model,
                      model)                      PDE reference)
```

`groundfield` provides the physically grounded reference model from
which reduced impedance and multi-port representations are derived.
These travel into `groundinsight` as `BusType` / `BranchType`
formulas where they can be reconciled with measurement data from
`groundmeas`.

## Goals

`groundfield` is being developed as a tool for **work package 1** of
the dissertation on networked grounding systems. The investigations
include:

- layered soil (two-layer and multi-layer models),
- arbitrary electrode geometries (ring, strip, rod, foundation, mesh),
- conductors, cable shields, and PEN with mutual coupling,
- inductive coupling between distributed conductors (Neumann
  partial-inductance assembly, ADR-0004),
- finite-conductivity earth-return correction (Carson 1926, ADR-0005)
  as a fast asymptotic option, plus the rigorous geometric Sommerfeld
  Green's function (ADR-0006) with native layered-earth support — the
  combination directly answers the AP1 question on diffusion-field
  effects, layered-earth coupling, and short-wire end effects below
  1 kHz,
- driven rods (Tiefenerder), foundation electrodes and conductors
  that cross soil-layer interfaces (ADR-0007) — `image_2layer`
  automatically dispatches to a rigorous Sommerfeld kernel for
  cross-layer pairs and keeps the fast Tagg/Sunde image series for
  pure-upper-layer worlds,
- current and potential distribution in the soil,
- influence of the current injection and measurement geometry on the
  grounding-measurement result,
- derivation of reduced `rho-f` models for `groundinsight`.

## Installation

`groundfield` requires **Python 3.12 or newer**.

```bash
git clone https://github.com/Ce1ectric/groundfield.git
cd groundfield
poetry install
```

The documentation extras live in an optional Poetry group:

```bash
poetry install --with docs
```

## Quickstart

```python
import groundfield as gf

# 1. Soil model (e.g. two-layer model from the AP1 parameter space)
soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)

# 2. Build a world and add a ring electrode
world = gf.create_world(soil=soil)
gf.create_electrode(
    world, "ring", name="g1",
    center=(0.0, 0.0, 0.8), radius=5.0, wire_radius=0.005,
)
gf.create_source(world, attached_to="g1", magnitude=1.0)

# 3. Configure the engine and solve.
#    Backends (auto-dispatched on the soil model):
#       "image"          — homogeneous, image charges
#       "image_2layer"   — Tagg/Sunde series for 2-layer soils
#       "image_nlayer"   — n-layer image dispatcher
#       "cim"            — Complex Image Method
#       "mom"            — Galerkin Method-of-Moments
#       "mom_sommerfeld" — Galerkin MoM with direct Sommerfeld quadrature
#       "bem"            — boundary-element collocation
#       "fem"            — axisymmetric volume FEM (optional)
engine = gf.create_engine(backend="image",
                          frequencies=[50.0, 150.0, 250.0])
result = world.solve(engine)

# 4. Inspect
print(result.cluster_impedance("g1"))
gf.plot_potential_radial(result, around="g1", world=world,
                         depths=[0.0, 0.5, 1.0])
```

### Reduced-model export to groundinsight

Turn a field-grade ``Z(s)`` into a `BusType` consumable by
`groundinsight`:

```python
from groundfield.postprocess.vector_fitting import rho_f_from_field_result
from groundfield.io.groundinsight import save_bustype_json, to_bustype

fit = rho_f_from_field_result(result, electrode_name="g1", n_poles=3)

# JSON path (no groundinsight installed required):
save_bustype_json(fit, "bus_type_substation.json",
                  name="SubstationBus", system_type="Substation",
                  voltage_level=20)

# Python API path (pip install groundfield[groundinsight]):
bus_type = to_bustype(fit, name="SubstationBus",
                      system_type="Substation", voltage_level=20)
```

### TN-Ortsnetz generator (AP1)

Build a fully-wired AP1 reference world from a high-level config:

```python
cfg = gf.TnNetworkConfig(
    soil=gf.TwoLayerSoilSpec(rho_1=100.0, rho_2=500.0, h_1=2.0),
    building_counts={"residential": 30},
    source_magnitude_A=1.0,
)
world = gf.TnNetworkGenerator().build(cfg)
```

Numerical / categorical fields accept either fixed values or any
of `Constant`, `Uniform`, `Normal`, `LogNormal`, `Weibull`,
`Discrete`, `Categorical` for parameter sweeps and Monte Carlo runs.

## Guiding principles

- **The PDE / field model is a reference, not the end product.** The
  solver must be instrumented so that every solution can be reduced
  to an identification-friendly form.
- **Measurability before accuracy.** The relevant frequency range is
  < 1 kHz; this allows simplified soil models and fast solvers.
- **Grey-box, not black-box.** Geometric and material inputs stay
  visible; only the parts that are not physically prescribed are
  identified.

## Development

```bash
# Tests with coverage
poetry run pytest --cov=groundfield

# Formatting
poetry run black src tests scripts

# Local documentation
poetry install --with docs
poetry run mkdocs serve
```

Releases are triggered through the Poetry script. It updates the
version in `pyproject.toml`, `src/groundfield/__init__.py`, and
`CITATION.cff`, moves the `[Unreleased]` block of `CHANGELOG.md` into
a new section, and creates an annotated tag.

```bash
poetry run release patch
poetry run release minor
poetry run release major
poetry run release set 1.2.3
```

## Citing

If you use `groundfield` in academic work, please cite according to
the metadata in `CITATION.cff`.

## License

`groundfield` is released under the [MIT license](LICENSE).
