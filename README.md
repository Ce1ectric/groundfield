# groundfield

**Numerical field computation for grounding systems.**

[![Python versions](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`groundfield` is an open-source Python package for the physical
reference modelling of networked grounding systems. Within the
`groundmeas` / `groundinsight` / `groundfield` software family,
`groundfield` covers the field-theoretical side: soil models,
electrode geometries, conductors and their couplings are formulated
as a 3-D problem in the soil and solved numerically. Field profiles,
potential curves and current distributions are reduced to equivalent
`rho-f` models that can be handed over to `groundinsight` as a
`BusType`. See the [documentation](https://ce1ectric.github.io/groundfield/)
for full details.

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

## Scope

`groundfield` covers layered soil (two-layer and multi-layer models),
typical electrode geometries (ring, strip, rod, foundation, mesh),
conductors, cable shields and PEN with their mutual coupling, the
Carson 1926 and rigorous Sommerfeld earth-return corrections, cross-
layer electrodes, current and potential distribution in the soil, the
influence of the measurement geometry on the grounding-measurement
result, and the derivation of reduced `rho-f` models for
`groundinsight`. See the [scope and concepts page](https://ce1ectric.github.io/groundfield/concepts/)
for the full list with references to the underlying ADRs.

## New in 0.6.0

- **OSM-driven building footprints** ([ADR-0011](docs/adr/0011-osm-building-footprints.md)).
  Pull real building outlines from OpenStreetMap via the Overpass API,
  project them into a local ENU frame, and feed them straight into
  `TnNetworkGenerator`. Each house's foundation electrode inherits its
  size and orientation from the polygon's oriented minimum bounding
  rectangle; the only stochastic axis that survives the override is
  `presence_prob`. See the new optional [`groundfield.geo`](docs/api/geo.md)
  subpackage, [example 09](docs/examples/09_osm_pipeline.md), and
  notebook [`32_osm_footprints.ipynb`](notebooks/32_osm_footprints.ipynb).

- **Concrete encasement for foundation electrodes** ([ADR-0012](docs/adr/0012-foundation-concrete-encasement.md)).
  DIN-18014 foundation electrodes sit in a concrete shell that, depending
  on moisture, has a resistivity anywhere from 30 Ω·m (wet) to 50 000 Ω·m
  (dry) — materially different from the surrounding soil. New optional
  fields on `FoundationElectrodeSpec` (`concrete_rho_ohm_m`,
  `concrete_thickness_m`, `concrete_model`) expose the closed-form
  Sunde-shell model in two flavours: a *lumped* series resistance on
  the PEN service drop (V1, default, zero solver-side risk) and a
  *distributed* per-segment diagonal augmentation in the
  `image` / `image_2layer` backends (V2). Stochastic moisture maps onto
  `concrete_rho_ohm_m=Discrete(values=[50, 150, 500, 2000], weights=…)`
  for the four empirical bands. Notebook
  [`33_concrete_encasement.ipynb`](notebooks/33_concrete_encasement.ipynb)
  is the interactive parameter-variation workbench.

## Installation

`groundfield` requires **Python 3.12 or newer**.

```bash
git clone https://github.com/Ce1ectric/groundfield.git
cd groundfield
poetry install
```

For OSM-driven building footprints (ADR-0011), enable the optional
`geo` extra (pulls in `requests`, `shapely`, `pyproj`):

```bash
pip install groundfield[geo]
# or, from a Poetry checkout
poetry install --extras geo
```

The documentation extras live in an optional Poetry group:

```bash
poetry install --with docs
```

## Quickstart

```python
import groundfield as gf

soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
world = gf.create_world(soil=soil)
gf.create_electrode(
    world, "ring", name="g1",
    center=(0.0, 0.0, 0.8), radius=5.0, wire_radius=0.005,
)
gf.create_source(world, attached_to="g1", magnitude=1.0)

engine = gf.create_engine(backend="image",
                          frequencies=[50.0, 150.0, 250.0])
result = world.solve(engine)
print(result.cluster_impedance("g1"))
```

`backend="image"` auto-dispatches to the matching layered backend.
The full backend list, the `rho-f` export to `groundinsight` and
the `TnNetworkGenerator` are documented in
[Quickstart](https://ce1ectric.github.io/groundfield/quickstart/)
and the [examples gallery](https://ce1ectric.github.io/groundfield/examples/).

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
