"""The tutorial notebooks run, check their numbers and stay fresh (10.89).

``docs/tutorials/T01`` … ``T10`` are versioned Jupyter notebooks that the
site renders with their stored outputs. Up to 2026.11 the example notebooks
lived outside version control and outside CI, and six of 46 no longer ran
(review pass 10, finding 10.89). This module

* executes every tutorial with nbclient from the repository root (marker
  ``slow``, 300 s per notebook), so a notebook that raises, fails one of
  its ``assert`` cells or uses a deprecated name fails the suite;
* checks without executing anything that the expected tutorials exist,
  that they are committed with outputs and without errors, that each one
  checks its numbers with ``assert``, and that the OpenStreetMap cache
  file of T06 still has the name the Overpass client looks up (otherwise
  T06 would silently try the network).

The notebooks' own ``assert`` cells carry the value checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = REPO_ROOT / "docs" / "tutorials"

#: The tutorials of the documentation plan, in order.
EXPECTED = (
    "T01_first_grounding_impedance",
    "T02_layered_soil",
    "T03_interconnected_systems_and_impedances",
    "T04_frequency_response_and_rho_f",
    "T05_export_to_groundinsight",
    "T06_tn_network_synthetic_and_osm",
    "T07_fall_of_potential_measurement",
    "T08_touch_and_step_assessment",
    "T09_choosing_an_engine_and_convergence",
    "T10_reproducible_studies",
)

#: Time budget per notebook in s (the notebooks themselves aim at < 120 s).
TIMEOUT_S = 300

#: Stored notebooks above this size probably embed too many or too large
#: figures.
MAX_NOTEBOOK_BYTES = 1_500_000

NOTEBOOKS = sorted(TUTORIALS.glob("T*.ipynb"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def test_every_planned_tutorial_exists() -> None:
    assert [p.stem for p in NOTEBOOKS] == list(EXPECTED)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_is_committed_with_outputs(path: Path) -> None:
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
def test_notebook_checks_its_numbers(path: Path) -> None:
    """Every tutorial asserts on its key numbers."""
    asserts = sum(_source(c).count("assert ") for c in _code_cells(_load(path)))
    assert asserts >= 3, f"{path.name}: {asserts} assert statements"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_needs_no_network(path: Path) -> None:
    """No tutorial may query Overpass without a cache (CI has no network)."""
    text = "\n".join(_source(c) for c in _code_cells(_load(path)))
    for call in ("from_osm(", "query_and_project(", "query_buildings("):
        if call in text:
            assert "cache_dir=" in text, f"{path.name}: {call} without cache_dir"


def test_osm_fixture_has_the_cache_key_of_its_query() -> None:
    """T06 reads ``data/osm/<sha256(query)>.json``; if the query text of the
    Overpass client changes, the fixture must be renamed (or T06 would try
    the network)."""
    pytest.importorskip("pyproj")
    from groundfield.geo import osm

    query = osm.build_query(51.9136, 10.8432, 150.0, timeout_s=osm.DEFAULT_TIMEOUT_S)
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()
    fixtures = sorted((TUTORIALS / "data" / "osm").glob("*.json"))
    assert [f.stem for f in fixtures] == [key]
    payload = json.loads(fixtures[0].read_text(encoding="utf-8"))
    assert "synthetic" in payload["generator"]  # not real map data


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_tutorial_executes(path: Path) -> None:
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
    for k, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                text = out.get("text", "")
                assert "DeprecationWarning" not in text, (
                    f"{path.name}, cell {k}: {text}"
                )
                assert "FutureWarning" not in text, f"{path.name}, cell {k}: {text}"
