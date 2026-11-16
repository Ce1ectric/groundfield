"""The example notebooks (case studies) run, check their numbers and stay fresh.

``docs/examples/E01`` … ``E03`` are versioned Jupyter notebooks that the site
renders with their stored outputs, like the tutorials (see
``tests/test_tutorials.py``, finding 10.89). This module

* executes every example with nbclient from the repository root (marker
  ``slow``), so an example that raises or fails one of its ``assert`` cells
  fails the suite;
* checks without executing anything that the expected examples exist, that
  they are committed with outputs and without errors, that each one checks
  its numbers with ``assert``, and that none queries the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "docs" / "examples"

#: The case studies of the documentation, in order.
EXPECTED = (
    "E01_secondary_substation_and_lv_network",
    "E02_single_family_house",
    "E03_hv_substation_grid_two_layer_soil",
)

#: Time budget per notebook in s (the notebooks themselves aim at < 120 s).
TIMEOUT_S = 300

#: Stored notebooks above this size probably embed too many or too large
#: figures.
MAX_NOTEBOOK_BYTES = 1_500_000

NOTEBOOKS = sorted(EXAMPLES.glob("E*.ipynb"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def test_every_planned_example_exists() -> None:
    assert [p.stem for p in NOTEBOOKS] == list(EXPECTED)


def test_every_example_is_on_the_index_page() -> None:
    index = (EXAMPLES / "index.md").read_text(encoding="utf-8")
    for name in EXPECTED:
        assert f"({name}.ipynb)" in index, name


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_example_is_committed_with_outputs(path: Path) -> None:
    """The site renders stored outputs (mkdocs-jupyter, ``execute: false``)."""
    nb = _load(path)
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    cells = _code_cells(nb)
    assert cells, path.name
    for k, cell in enumerate(cells):
        assert cell.get("execution_count") is not None, (
            f"{path.name}: code cell {k} was not executed"
        )
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", (
                f"{path.name}: stored error in code cell {k}"
            )
    assert path.stat().st_size < MAX_NOTEBOOK_BYTES, path.name


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_example_checks_its_numbers(path: Path) -> None:
    """Every example asserts on its key numbers."""
    asserts = sum(_source(c).count("assert ") for c in _code_cells(_load(path)))
    assert asserts >= 3, f"{path.name}: {asserts} assert statements"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_example_needs_no_network(path: Path) -> None:
    """No example may query Overpass without a cache (CI has no network)."""
    text = "\n".join(_source(c) for c in _code_cells(_load(path)))
    for call in ("from_osm(", "query_and_project(", "query_buildings("):
        if call in text:
            assert "cache_dir=" in text, f"{path.name}: {call} without cache_dir"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_example_prints_no_local_path(path: Path) -> None:
    """Stored outputs carry no path of the machine that ran the notebook."""
    nb = _load(path)
    for k, cell in enumerate(_code_cells(nb)):
        for out in cell.get("outputs", []):
            text = json.dumps(out)
            for marker in ("/Users/", "/home/", "/var/folders/", "C:\\\\Users"):
                assert marker not in text, f"{path.name}: code cell {k} prints {marker}"


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_example_executes(path: Path) -> None:
    """Run the notebook top to bottom; its assert cells are the checks."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as exc:
        pytest.fail(f"{path.name} failed:\n{exc}")
