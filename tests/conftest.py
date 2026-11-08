"""Shared pytest configuration.

Markers (registered in ``pyproject.toml``, enforced by
``--strict-markers``):

``finding(id, ...)``
    Links a test to one or more review findings, e.g.
    ``@pytest.mark.finding("10.15")``, or to a finding of an audit round
    of the release, e.g. ``@pytest.mark.finding("A1-01")``. Select with
    ``--finding 10.15``.
``slow``
    Long-running test. Deselect with ``-m "not slow"``.
``network``
    Needs network access (Overpass / OSM). Skipped unless
    ``--run-network`` is given.
``benchmark``
    Performance benchmark. Skipped unless ``--run-benchmark`` is given
    or the ``-m`` expression names ``benchmark``.
``golden``
    Golden-value regression test (``tests/golden``).

Options
-------
``--golden-report PATH``
    Write all golden-value comparisons (identical, within tolerance,
    moved) as JSON to ``PATH`` at the end of the session.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict

import pytest

_FINDING_ID = re.compile(r"^(\d+\.\d+|F\d+|S\d+|A\d+-\d+)$")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("groundfield")
    group.addoption(
        "--finding",
        action="append",
        default=[],
        metavar="ID",
        help="run only tests marked with this review finding (repeatable)",
    )
    group.addoption(
        "--run-network",
        action="store_true",
        help="run tests marked 'network'",
    )
    group.addoption(
        "--run-benchmark",
        action="store_true",
        help="run tests marked 'benchmark'",
    )
    group.addoption(
        "--golden-report",
        metavar="PATH",
        default=None,
        help="write the golden-value comparison report (JSON) to PATH",
    )


def _finding_ids(item: pytest.Item) -> list[str]:
    ids: list[str] = []
    for mark in item.iter_markers(name="finding"):
        if not mark.args:
            raise pytest.UsageError(f"{item.nodeid}: finding marker needs an id")
        for arg in mark.args:
            if not isinstance(arg, str) or not _FINDING_ID.match(arg):
                raise pytest.UsageError(
                    f"{item.nodeid}: invalid finding id {arg!r} "
                    "(expected e.g. '10.15', 'F02', 'S01' or 'A1-01')"
                )
            ids.append(arg)
    return ids


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    wanted = set(config.getoption("--finding"))
    markexpr = config.getoption("-m") or ""
    run_network = config.getoption("--run-network")
    run_benchmark = config.getoption("--run-benchmark") or "benchmark" in markexpr

    skip_network = pytest.mark.skip(reason="needs --run-network")
    skip_benchmark = pytest.mark.skip(reason="needs --run-benchmark")

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        ids = _finding_ids(item)  # validates every finding marker
        if wanted and not wanted.intersection(ids):
            deselected.append(item)
            continue
        if "network" in item.keywords and not run_network:
            item.add_marker(skip_network)
        if "benchmark" in item.keywords and not run_benchmark:
            item.add_marker(skip_benchmark)
        selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(scope="session")
def golden_report() -> list:
    """Collects golden-value comparisons for ``--golden-report``."""
    return _GOLDEN_ROWS


_GOLDEN_ROWS: list = []


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    path = session.config.getoption("--golden-report")
    if not path or not _GOLDEN_ROWS:
        return
    from tests.golden import harness

    rows = [asdict(r) for r in _GOLDEN_ROWS]
    payload = {
        "rtol": harness.rtol(),
        "strict": harness.strict(),
        "n_quantities": len(rows),
        "n_identical": sum(r["status"] == "identical" for r in rows),
        "n_within_tol": sum(r["status"] == "within_tol" for r in rows),
        "moved": [r for r in rows if r["status"] not in ("identical", "within_tol")],
        "within_tol": [r for r in rows if r["status"] == "within_tol"],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")
