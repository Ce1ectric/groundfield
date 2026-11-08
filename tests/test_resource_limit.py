"""Per-engine resource limit ``Engine.max_segments`` (finding 10.06, user decision Q17).

The hard problem-size limit of the segment backends (the number of leakage
segments and the size of the dense nodal system) was the process-wide
module constant :data:`groundfield.diagnostics.HARD_LIMIT` (20 000): a
world above it could only be solved by changing the constant for every
engine of the process. The user decided (2026-11-08, question Q17) that
the limit is adjustable per engine: ``Engine(max_segments=...)`` raises or
lowers it for one engine, is validated, recorded in the result metadata,
and honoured by :func:`~groundfield.diagnostics.diagnose`,
:func:`~groundfield.diagnostics.check_segment_budget`,
:func:`~groundfield.diagnostics.expected_segments` and every solve path,
``solve_mutual_*`` included; the module constants stay the defaults.

Every test here fails on the base (``1e587cc``) because the field does
not exist (``extra="forbid"``).
"""

from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield import diagnostics
from groundfield.diagnostics import (
    ResourceLimitError,
    check_budget_ok,
    check_segment_budget,
    expected_segments,
)

pytestmark = pytest.mark.finding("10.06")

#: A 5 m ring at ds = 0.5 m: 63 segments.
N_RING = 63


def _ring_world(*, source: bool = True) -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "ring", name="g1", center=(0.0, 0.0, 0.7), radius=5.0)
    if source:
        gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _two_rings() -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=300.0, rho_2=60.0, h_1=4.0))
    for name, x in (("a", 0.0), ("b", 30.0)):
        gf.create_electrode(w, "ring", name=name, center=(x, 0.0, 0.7), radius=5.0)
    return w


def _engine(**kw: object) -> gf.Engine:
    return gf.Engine(segment_length=0.5, frequencies=[50.0], **kw)


def _solve(world: gf.World, engine: gf.Engine) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.solve(world)


# ---------------------------------------------------------------------
# The field
# ---------------------------------------------------------------------


def test_max_segments_is_a_validated_engine_field() -> None:
    """Default ``None`` (the module constant); positive integers only."""
    # user decision Q17
    assert gf.Engine().max_segments is None
    assert gf.Engine(max_segments=50_000).max_segments == 50_000
    assert gf.create_engine(max_segments=123).max_segments == 123
    for bad in (0, -5, 2.5, "many"):
        with pytest.raises(ValueError):
            gf.Engine(max_segments=bad)
    engine = gf.Engine(max_segments=1000)
    with pytest.raises(ValueError):
        engine.max_segments = 0  # validated on assignment too
    assert engine.max_segments == 1000
    # JSON round trip; engine JSON written before the field existed loads
    # with the default.
    assert gf.Engine.model_validate_json(engine.model_dump_json()) == engine
    old = {k: v for k, v in gf.Engine().model_dump().items() if k != "max_segments"}
    assert gf.Engine.model_validate(old).max_segments is None
    # copies keep it
    assert engine.with_frequencies(50.0, 1000.0).max_segments == 1000
    assert engine.model_copy(update={"segment_length": 1.0}).max_segments == 1000


# ---------------------------------------------------------------------
# Every solve path honours the engine's limit
# ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["image", "image_2layer", "mom", "mom_sommerfeld"])
def test_engine_limit_lowers_the_limit_for_one_engine(backend: str) -> None:
    """63 segments: refused by an engine with ``max_segments=50`` only."""
    # user decision Q17
    world = _ring_world()
    if backend in ("image_2layer", "mom_sommerfeld"):
        world.soil = gf.TwoLayerSoil(rho_1=300.0, rho_2=60.0, h_1=4.0)
    tracemalloc.start()
    try:
        with pytest.raises(ResourceLimitError, match=r"Engine\.max_segments = 50"):
            _solve(world, _engine(backend=backend, max_segments=50))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 20 * 2**20  # bytes: refused before the problem is built
    res = _solve(world, _engine(backend=backend, max_segments=N_RING))
    assert res.metadata["n_segments"] == N_RING
    assert len(_solve(world, _engine(backend=backend)).point_sources) == N_RING


def test_engine_limit_overrides_the_module_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module constant is the default only: an engine raises it."""
    # user decision Q17
    monkeypatch.setattr(diagnostics, "HARD_LIMIT", 50)
    world = _ring_world()
    with pytest.raises(ResourceLimitError, match=r"HARD_LIMIT = 50"):
        _solve(world, _engine())
    res = _solve(world, _engine(max_segments=100))
    assert len(res.point_sources) == N_RING


def test_the_limit_is_recorded_in_the_result_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``metadata["max_segments"]``: the limit the solve was checked against."""
    # user decision Q17
    world = _ring_world()
    assert _solve(world, _engine(max_segments=777)).metadata["max_segments"] == 777
    assert _solve(world, _engine()).metadata["max_segments"] == diagnostics.HARD_LIMIT
    monkeypatch.setattr(diagnostics, "HARD_LIMIT", 4321)
    res = _solve(world, _engine())
    assert res.metadata["max_segments"] == 4321
    # the engine block of the provenance keeps the request itself
    assert res.metadata["provenance"]["engine"]["max_segments"] is None
    assert gf.Engine.model_validate(res.metadata["provenance"]["engine"]) == _engine()


@pytest.mark.parametrize("which", ["matrix", "field"])
def test_mutual_paths_honour_the_engine_limit(
    monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """``solve_mutual_matrix`` / ``solve_mutual_field`` (2 × 63 segments)."""
    # user decision Q17
    world = _two_rings()

    def run(engine: gf.Engine) -> np.ndarray:
        if which == "matrix":
            return gf.solve_mutual_matrix(world, engine, ["a", "b"])
        points = np.array([[15.0, 0.0, 0.0]])
        return gf.solve_mutual_field(world, engine, ["a", "b"], points)

    with pytest.raises(ResourceLimitError, match=r"Engine\.max_segments = 100"):
        run(_engine(max_segments=100))
    monkeypatch.setattr(diagnostics, "HARD_LIMIT", 100)
    with pytest.raises(ResourceLimitError, match=r"HARD_LIMIT = 100"):
        run(_engine())
    assert np.all(np.isfinite(run(_engine(max_segments=2 * N_RING))))


def test_the_nodal_system_check_honours_the_engine_limit() -> None:
    """A 100-piece isolated chain: 199 + 4 unknowns, 12 segments."""
    # user decision Q17
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="a", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="b", position=(100.0, 0.0, 0.5), length=3.0)
    gf.create_conductor(
        w,
        start="a",
        end="b",
        name="pen",
        cross_section=50e-6,
        discretize_segment_length=1.0,
        coupling_to_soil="isolated",
    )
    gf.create_source(w, attached_to="a")
    with pytest.raises(ResourceLimitError, match=r"203 unknowns.*max_segments = 150"):
        _solve(w, _engine(max_segments=150))
    assert _solve(w, _engine(max_segments=203)).metadata["max_segments"] == 203


# ---------------------------------------------------------------------
# Diagnostics follow the engine
# ---------------------------------------------------------------------


def test_check_segment_budget_and_expected_segments_honour_the_engine_limit() -> None:
    # user decision Q17
    world = _ring_world()
    with pytest.raises(ResourceLimitError, match=r"Engine\.max_segments = 10\b"):
        check_segment_budget(world, _engine(max_segments=10))
    assert check_segment_budget(world, _engine(max_segments=N_RING)) == N_RING
    assert not check_budget_ok(world, _engine(max_segments=10))
    assert check_budget_ok(world, _engine())
    pred = expected_segments(world, _engine(max_segments=10))
    assert pred["limit"] == 10
    assert pred["total"] == N_RING
    assert expected_segments(world, _engine())["limit"] == diagnostics.HARD_LIMIT


def test_expected_segments_counts_by_the_bound_above_the_engine_limit() -> None:
    """A ring entered in mm (62 832 segments): beyond the engine's limit the
    closed-form bound is returned without discretising, below it the count is
    exact."""
    # user decision Q17
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "ring", name="g1", center=(0.0, 0.0, 800.0), radius=5000.0)
    tracemalloc.start()
    try:
        pred = expected_segments(w, _engine(max_segments=1000))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert pred["total"] >= 62_832 and pred["limit"] == 1000
    assert peak < 5 * 2**20
    exact = expected_segments(w, _engine(max_segments=100_000))
    assert exact["total"] == 62_832 and exact["limit"] == 100_000


def test_diagnose_honours_the_engine_limit() -> None:
    # user decision Q17
    world = _ring_world()
    report = diagnostics.diagnose(world, _engine(max_segments=50))
    budget = report.by_category("budget")
    assert len(budget) == 1 and budget[0].severity == "warning"
    assert budget[0].data["limit"] == 50
    assert "Engine.max_segments = 50" in budget[0].message
    assert report.context["max_segments"] == 50
    assert not diagnostics.diagnose(world, _engine()).by_category("budget")
    # the solve emits the same cause as one warning before it refuses
    with pytest.warns(UserWarning, match="budget: .*max_segments = 50"):
        with pytest.raises(ResourceLimitError):
            _engine(max_segments=50).solve(world)


def test_raising_the_engine_limit_silences_the_budget_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above the module default but within the engine's limit: no warning."""
    # user decision Q17
    monkeypatch.setattr(diagnostics, "HARD_LIMIT", 50)
    world = _ring_world()
    assert diagnostics.diagnose(world, _engine()).by_category("budget")
    assert not diagnostics.diagnose(world, _engine(max_segments=100)).by_category(
        "budget"
    )
