# Installation

`groundfield` requires **Python 3.12 or newer** and uses
[Poetry](https://python-poetry.org/) for dependency management.

## From the Git checkout

```bash
git clone https://github.com/Ce1ectric/groundfield.git
cd groundfield
poetry install
```

Dev dependencies (pytest, black, ipykernel) live in the ``dev`` group
and are installed by default.

## Documentation group

To build the documentation locally:

```bash
poetry install --with docs
poetry run mkdocs serve
```

## PyPI

Once the first release is published:

```bash
pip install groundfield
```

## Integration with the sister projects

`groundfield` is designed to work alongside `groundinsight` and
`groundmeas`. When the three projects share a Poetry environment they
can be wired up as path dependencies:

```toml
[tool.poetry.dependencies]
groundinsight = { path = "../groundinsight", develop = true }
groundmeas = { path = "../groundmeas", develop = true }
```

This step is optional and is usually only relevant for
integration-heavy development.
