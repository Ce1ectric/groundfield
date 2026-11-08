# Installation

`groundfield` runs on **Python 3.12, 3.13 and 3.14** on any operating
system. The core dependencies are numpy, scipy, pydantic, sympy, pandas
and matplotlib.

## From PyPI

```bash
pip install groundfield
```

Check the installation:

```bash
python -c "import groundfield; print(groundfield.__version__)"
```

### Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `geo` | `requests`, `shapely`, `pyproj` | building footprints from OpenStreetMap (Overpass), WGS84 → local coordinates, polygon geometry ([Geo / OSM](../reference/api/geo.md), ADR-0011) |
| `groundinsight` | `groundinsight` | live `groundinsight.BusType` objects and writing into the `groundinsight` SQLite store; the JSON export works without it ([IO and groundinsight](../reference/api/io.md), ADR-0008) |
| `fast` | `numba` | compiled kernels for the two-layer image series and the far field of the exact line-source kernel; same results, shorter solve times |

```bash
pip install "groundfield[geo,fast]"
```

Without the `geo` extra, `import groundfield.geo` works, and the
functions that need the extra raise an `ImportError` with the install
command. Without `fast`, every kernel runs in numpy.

## From a Git checkout (development)

The project uses [Poetry](https://python-poetry.org/) (2.x) and a
committed lock file:

```bash
git clone https://github.com/Ce1ectric/groundfield.git
cd groundfield
poetry install                     # package + dev group
poetry install --extras "geo fast" # optional extras
```

The `dev` group contains pytest (with pytest-cov and pytest-xdist),
ruff, mypy, ipykernel, and `groundinsight`, `requests`, `shapely` and
`pyproj`, so the bridge and OSM tests run end to end.

### Tests

```bash
poetry run pytest -n auto                  # full suite, parallel
poetry run pytest -m "not slow"            # skip long-running tests
poetry run pytest --run-benchmark tests/benchmarks   # engine benchmark suite
poetry run pytest --run-network            # tests that query Overpass
```

The markers `finding`, `slow`, `network`, `benchmark` and `golden` are
registered in `pyproject.toml`; `pytest --finding 10.28` selects the
tests of one review finding.

### Code standards

```bash
poetry run ruff format .
poetry run ruff check --select F,E9,B,W605 .   # blocking subset in CI
poetry run mypy                                # report only for now
```

The same checks run as [pre-commit](https://pre-commit.com/) hooks
(`.pre-commit-config.yaml`; install pre-commit with pipx, then
`pre-commit install`). Docstrings follow the NumPy convention.

### Documentation

```bash
poetry install --with docs
poetry run mkdocs serve           # live preview on http://127.0.0.1:8000
poetry run mkdocs build --strict  # what CI runs
```

The docs group contains MkDocs with Material, mkdocstrings,
mkdocs-jupyter (renders the tutorial notebooks with their stored
outputs) and mike (versioned deployment). Python code blocks in the
Markdown pages are executed by the test suite, so a snippet that no
longer matches the API fails CI; see the how-to guide on contributing
to the documentation for the rules.

## Versions and releases

Releases use calendar versioning `YYYY.M.MICRO` in its PEP 440 form:
`2026.11.0` is the first release of November 2026, `2026.11.1` the
second, `2027.1.0` the first of January 2027. Releases up to `0.15.0`
used semantic versioning. Deprecated names keep working for at least
two releases and six months; see [Deprecations](../reference/deprecations.md).

!!! note "For maintainers: cutting a release"

    The release tool lives in the repository and is not installed with
    the package:

    ```bash
    python scripts/release.py calver --dry-run   # all checks, prints the plan
    python scripts/release.py calver             # writes, commits, tags
    git push origin main
    git push origin vYYYY.M.MICRO
    ```

    The tool validates everything before it writes (clean tree, branch
    up to date with its upstream, the three version sources
    `pyproject.toml`, `src/groundfield/__init__.py` and `CITATION.cff`
    agree, non-empty `[Unreleased]` block, no unmerged
    `changelog.d/` fragments), then commits `chore(release): vX` and
    creates the annotated tag. It never pushes. The tag starts the
    release workflow (tests, tag/version check, build, PyPI trusted
    publishing) and the documentation workflow, which deploys the docs
    of that version with the alias `latest`. `python scripts/release.py
    set 2026.11.0rc1` sets an explicit version, e.g. a release candidate.

## Working with the sister projects

`groundfield`, `groundinsight` and `groundmeas` can share one Poetry
environment through path dependencies, which is useful when changing
the interface between them:

```toml
[tool.poetry.group.dev.dependencies]
groundinsight = { path = "../groundinsight", develop = true }
groundmeas = { path = "../groundmeas", develop = true }
```
