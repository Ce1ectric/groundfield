"""Benchmark-suite helpers: measured values can be written to a JSON file.

Set ``GF_BENCHMARK_REPORT=<path>`` and run the suite with
``--run-benchmark``; every benchmark that calls the ``measured`` fixture
appends ``{"benchmark", "backend", "case", "value", "bound"}`` to the
file at the end of the session. The values feed the "measured" column of
``tolerances.py``.
"""

from __future__ import annotations

import json
import os

import pytest

_ROWS: list[dict] = []


@pytest.fixture
def measured():
    """Callable ``measured(benchmark, backend, case, value, bound)``."""

    def record(benchmark: str, backend: str, case: str, value: float, bound) -> None:
        _ROWS.append(
            {
                "benchmark": benchmark,
                "backend": backend,
                "case": case,
                "value": float(value),
                "bound": None if bound is None else float(bound),
            }
        )

    return record


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    path = os.environ.get("GF_BENCHMARK_REPORT")
    if path and _ROWS:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_ROWS, fh, indent=1)
            fh.write("\n")
