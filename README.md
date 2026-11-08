# groundfield

**Numerical field computation for grounding systems.**

[![Python 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![CalVer YYYY.M.MICRO](https://img.shields.io/badge/calver-YYYY.M.MICRO-22bfda)](https://calver.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`groundfield` computes the electric field of grounding systems in
layered soil: electrodes of any shape, the PEN conductors, cable screens
and measurement leads that connect them (with inductive coupling and
earth return), and the resulting current and potential distribution
below 1 kHz. From a field solution it derives grounding impedances,
touch and step voltages (EN 50522) and reduced `rho-f` models that are
exported to the sister project
[`groundinsight`](https://github.com/Ce1ectric/groundinsight). Together
with `groundmeas` (measurements) the three packages model networked
grounding systems in medium- and low-voltage distribution networks.

- **Documentation:** <https://ce1ectric.github.io/groundfield/>
- **Release notes:** [changelog](https://ce1ectric.github.io/groundfield/latest/changelog/)
- **Source and issues:** <https://github.com/Ce1ectric/groundfield>

## Installation

```bash
pip install groundfield                    # Python 3.12 to 3.14
pip install "groundfield[geo,fast]"        # optional: OpenStreetMap footprints, compiled kernels
pip install "groundfield[groundinsight]"   # optional: live groundinsight BusType objects
```

For development: `git clone`, then `poetry install` (see the
[installation guide](https://ce1ectric.github.io/groundfield/latest/getting-started/installation/)).

## Example

A substation ring earth and a house rod 40 m away, joined by a PEN
conductor, in two-layer soil; 1 A injected at the substation:

```python
import groundfield as gf

world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=2.0))
gf.create_electrode(world, "ring", name="substation", center=(0.0, 0.0, 0.8), radius=3.0)
gf.create_electrode(world, "rod", name="house", position=(40.0, 0.0, 0.0), length=3.0)
gf.create_conductor(world, name="pen", start="substation", end="house", cross_section=95e-6)
gf.create_source(world, attached_to="substation", magnitude=1.0)

result = world.solve(gf.create_engine(frequencies=[50.0]))
print(result.input_impedance()[0])                # 11.27 Ω, seen at the feeding point
print(result.cluster_impedance("substation")[0])  # 14.10 Ω, the substation earth alone
```

Coordinates are in metres, with the depth `z` positive downwards. The
[quickstart](https://ce1ectric.github.io/groundfield/latest/getting-started/quickstart/)
explains the example and the [tutorials](https://ce1ectric.github.io/groundfield/latest/tutorials/)
continue from there.

## Development

```bash
poetry install --with docs
poetry run pytest -n auto                       # tests
poetry run ruff format . && poetry run ruff check --select F,E9,B,W605 .
poetry run mkdocs serve                         # documentation preview
python scripts/release.py calver --dry-run      # maintainers: release checks
```

Versions follow calendar versioning `YYYY.M.MICRO` (from `2026.11.0`;
up to `0.15.0` semantic versioning); deprecated names keep working for
at least two releases and six months.

## Citing

If you use `groundfield` in academic work, please cite it with the
metadata in [`CITATION.cff`](CITATION.cff).

## License

`groundfield` is released under the [MIT license](LICENSE).
