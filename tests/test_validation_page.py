"""The generated validation page matches the engine benchmarks (docs decision 6).

``docs/explanation/validation.md``, its figures and the validation tables
of the engine theory pages are written by
``scripts/generate_validation_page.py`` from ``tests/benchmarks/tolerances.py``
and the stored measurements ``docs/explanation/validation/measurements.json``.

* The fast tests check that the committed files equal what the generator
  renders from the stored measurements and the current tolerance table,
  that every bound has measurements and that every stored value meets its
  bound — so a changed bound, a stale page or a failing benchmark cannot
  reach the documentation unnoticed.
* The slow test (``--run-benchmark``) runs the benchmark suite again and
  compares the fresh values with the stored ones.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "generate_validation_page.py"


def _load_generator() -> ModuleType:
    """Import the generator script as a module (``scripts`` is not shipped)."""
    spec = importlib.util.spec_from_file_location("gf_validation_page", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_generator()


@pytest.fixture(scope="module")
def stored(generator: ModuleType) -> dict:
    return generator.load_measurements()


def test_committed_pages_match_the_stored_measurements(generator: ModuleType) -> None:
    problems = generator.check()
    assert problems == [], (
        "generated documentation is out of date: "
        + "; ".join(problems)
        + " (run: python scripts/generate_validation_page.py)"
    )


def test_every_bound_has_measurements(generator: ModuleType, stored: dict) -> None:
    from tests.benchmarks.tolerances import BENCHMARKS

    measured = {
        (r["benchmark"], r["backend"]) for r in stored["benchmarks"] if r["bound"]
    }
    missing = [
        (bench_id, key)
        for bench_id, bench in BENCHMARKS.items()
        for key, b in bench.bounds.items()
        if b.bound is not None and (bench_id, key) not in measured
    ]
    assert missing == [], f"bounds without a stored measurement: {missing}"


def test_stored_values_use_the_current_bounds(stored: dict) -> None:
    from tests.benchmarks.tolerances import bound

    stale = [
        (r["benchmark"], r["backend"], r["case"])
        for r in stored["benchmarks"]
        if r["bound"] is not None
        and bound(r["benchmark"], r["backend"]).bound != r["bound"]
    ]
    assert stale == [], f"measured against an old bound, re-measure: {stale}"


def test_every_stored_value_meets_its_bound(stored: dict) -> None:
    failing = [
        (r["benchmark"], r["backend"], r["case"], r["value"], r["bound"])
        for r in stored["benchmarks"]
        if r["bound"] is not None and not r["value"] <= r["bound"]
    ]
    assert failing == []
    assert stored["suite"]["exit_code"] == 0
    assert stored["suite"]["counts"].get("failed", 0) == 0


def test_line_kernel_is_exact_for_straight_wires(
    generator: ModuleType, stored: dict
) -> None:
    """The page's kernel claim: straight wires in homogeneous soil to 1e-5."""
    rows = [
        r
        for r in stored["kernel"]
        if r["kernel"] == "line" and r["case"] in ("rod", "strip")
    ]
    assert rows and all(abs(r["deviation"]) < 1e-5 for r in rows)


def test_closed_forms_of_the_kernel_check(generator: ModuleType) -> None:
    """The uniform-current references against independent evaluations."""
    from scipy.integrate import dblquad

    gen = generator

    def numeric(length: float, d: float) -> float:
        value, _ = dblquad(
            lambda u, v: 1.0 / ((u - v) ** 2 + d * d) ** 0.5,
            0.0,
            length,
            0.0,
            length,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        return value / length**2

    for length, d in ((3.0, 0.008), (10.0, 1.0), (6.0, 0.05)):
        assert gen.galerkin_self(length, d) == pytest.approx(
            numeric(length, d), rel=1e-8
        )
    # Dwight / Sunde: R = rho/(2 pi L) (ln(4L/a) - 1) for a << L.
    rod = gen.rod_uniform_resistance(100.0, 3.0, 0.008)
    sunde = 100.0 / (2 * math.pi * 3.0) * (math.log(4 * 3.0 / 0.008) - 1.0)
    assert rod == pytest.approx(sunde, rel=5e-4)
    # Equal resistivities: the two-layer strip is the homogeneous strip.
    hom = gen.strip_uniform_resistance(100.0, 10.0, 0.005, 0.5)
    two = gen.strip_uniform_resistance(100.0, 10.0, 0.005, 0.5, 100.0, 5.0)
    assert two == hom


@pytest.mark.slow
@pytest.mark.benchmark
def test_a_fresh_benchmark_run_reproduces_the_stored_values(
    generator: ModuleType, stored: dict
) -> None:
    fresh = generator.measure()
    assert fresh["suite"]["exit_code"] == 0, fresh["suite"]
    problems = generator.compare_measurements(fresh, stored)
    assert problems == [], "\n".join(problems)
