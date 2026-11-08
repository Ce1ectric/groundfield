"""Pre-solve diagnostics: one aggregated report, resource guard (10.70, 10.06).

Finding 10.70: the solve emitted one warning per electrode and per
cause (81 in docs example 07), polyline and star electrodes were never
checked (F53 / F54 were listed as fixed in 0.15.0, but ``diagnostics.py``
was unchanged since 0.13.0), and close or interpenetrating wires
(10.44) went unreported. Finding 10.06: ``HARD_LIMIT`` only warned, so a
ring entered in millimetres went on to solve a 62 832-segment problem.
"""

from __future__ import annotations

import math
import time
import tracemalloc
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield import diagnostics
from groundfield.diagnostics import check_segment_resolution, expected_segments


def _soil(rho: float = 100.0) -> gf.HomogeneousSoil:
    """Homogeneous soil; ``resistivity`` was the field name up to 0.15.0."""
    try:
        return gf.HomogeneousSoil(rho=rho)
    except Exception:  # noqa: BLE001 - v0.15.0 names the field ``resistivity``
        return gf.HomogeneousSoil(resistivity=rho)


def _engine(ds: float = 0.5) -> gf.Engine:
    return gf.create_engine(backend="image", segment_length=ds, frequencies=[50.0])


# ---------------------------------------------------------------------
# F53 / F54 — every electrode kind is checked
# ---------------------------------------------------------------------


@pytest.mark.finding("10.82", "10.79")
@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("star", {"center": (0.0, 0.0, 0.5), "n_arms": 4, "arm_length": 0.2}),
        ("polyline", {"vertices": [(0.0, 0.0, 0.5), (0.3, 0.0, 0.5)]}),
    ],
)
def test_resolution_check_covers_star_and_polyline(kind: str, params: dict) -> None:
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, kind, name="e", **params)
    msgs = check_segment_resolution(w, _engine(0.5))
    assert any(m.startswith("resolution:") and "'e'" in m for m in msgs), msgs


# ---------------------------------------------------------------------
# 10.70 — aggregation
# ---------------------------------------------------------------------


def _many_short_rods(n: int = 30) -> gf.World:
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "ring", name="st", center=(0.0, 0.0, 0.7), radius=3.0)
    for k in range(n):
        gf.create_electrode(
            w,
            "rod",
            name=f"joint_{k}",
            position=(10.0 + 2.0 * k, 0.0, 0.7),
            length=0.05,
        )
    gf.create_source(w, attached_to="st", magnitude=1.0)
    return w


@pytest.mark.finding("10.70")
def test_one_message_per_cause_not_per_electrode() -> None:
    msgs = check_segment_resolution(_many_short_rods(30), _engine(0.5))
    resolution = [m for m in msgs if m.startswith("resolution:")]
    assert len(resolution) == 1, msgs
    assert "30 electrodes" in resolution[0]
    assert "joint_0" in resolution[0]
    assert "and 25 more" in resolution[0]


@pytest.mark.finding("10.70")
def test_a_solve_emits_at_most_one_warning_per_cause() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _many_short_rods(30).solve(_engine(0.5))
    resolution = [c for c in caught if "resolution:" in str(c.message)]
    assert len(resolution) == 1, [str(c.message)[:80] for c in caught]


@pytest.mark.finding("10.70")
def test_report_is_structured_and_aggregated() -> None:
    w = _many_short_rods(4)
    gf.create_electrode(
        w, "rod", name="fat", position=(-20.0, 0.0, 0.5), length=3.0, wire_radius=0.2
    )
    report = diagnostics.diagnose(w, _engine(0.5))
    cats = report.categories()
    assert "resolution" in cats and "thin_wire" in cats
    res = report.by_category("resolution")
    assert sorted(d.subjects[0] for d in res) == [f"joint_{k}" for k in range(4)]
    assert all(d.severity == "warning" for d in res)
    assert res[0].data["min_feature_length_m"] == pytest.approx(0.05)
    text = str(report)
    assert text.count("resolution:") == 1 and text.count("thin-wire:") == 1
    with pytest.warns(diagnostics.DiagnosticsWarning) as rec:
        report.warn()
    assert len(rec) == 1


@pytest.mark.finding("10.70", "10.44")
@pytest.mark.parametrize(
    ("kernel", "severity"), [("point", "warning"), ("line", "info")]
)
def test_close_and_interpenetrating_wires_are_reported(
    kernel: str, severity: str
) -> None:
    """Point kernel: a warning (5.7x the one-rod impedance at 2 mm, e04);
    the exact line kernel integrates near pairs and stays physical."""
    w = gf.create_world(soil=_soil())
    gf.create_electrode(
        w, "rod", name="a", position=(0.0, 0.0, 0.5), length=3.0, wire_radius=0.008
    )
    gf.create_electrode(
        w, "rod", name="b", position=(0.002, 0.0, 0.5), length=3.0, wire_radius=0.008
    )
    gf.create_electrode(
        w, "rod", name="c", position=(0.05, 0.0, 0.5), length=3.0, wire_radius=0.008
    )
    gf.create_electrode(w, "rod", name="far", position=(5.0, 0.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="a", magnitude=1.0)
    eng = gf.create_engine(segment_length=0.5, kernel=kernel)
    report = diagnostics.diagnose(w, eng)
    prox = report.by_category("proximity")
    pairs = {tuple(sorted(d.subjects)): d.data["kind"] for d in prox}
    assert pairs[("a", "b")] == "interpenetrating"
    assert pairs[("a", "c")] == "close"
    assert not any("far" in p for p in pairs)
    assert all(d.severity == severity for d in prox)


@pytest.mark.finding("10.70")
def test_inert_configuration_is_reported() -> None:
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_conductor(
        w, name="loose", start="g1", end=(10.0, 0.0, 0.5), cross_section=50e-6
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    w.concrete_shell_corrections["g1"] = 0.5
    report = diagnostics.diagnose(w, _engine(0.5))
    inert = {d.subjects[0]: d for d in report.by_category("inert")}
    assert "loose" in inert and "not part of the network" in inert["loose"].message
    assert "concrete_shell_corrections" in inert


@pytest.mark.finding("10.70")
def test_series_truncation_is_reported_after_the_solve() -> None:
    # The rod stays in the upper layer: the Tagg/Sunde series path.
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=10.0, h_1=3.0))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=2.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(
        backend="image_2layer", segment_length=0.5, image_max_terms=5
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = w.solve(eng)
    report = diagnostics.diagnose(w, eng, result=res)
    series = report.by_category("series")
    assert len(series) == 1 and "image_max_terms" in series[0].message


@pytest.mark.finding("10.70")
def test_clean_world_has_an_empty_report() -> None:
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    report = diagnostics.diagnose(w, _engine(0.25))
    assert len(report) == 0 and not report
    assert str(report) == "no findings"


# ---------------------------------------------------------------------
# 10.06 — resource guard before any allocation
# ---------------------------------------------------------------------


def _mm_ring() -> gf.World:
    """A 5 m ring entered in millimetres: 62 832 segments at 0.5 m."""
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "ring", name="g1", center=(0.0, 0.0, 800.0), radius=5000.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


@pytest.mark.finding("10.06")
def test_world_beyond_the_hard_limit_is_rejected_before_any_allocation() -> None:
    w = _mm_ring()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(
                diagnostics.ResourceLimitError, match=r"62832.*HARD_LIMIT"
            ):
                w.solve(_engine(0.5))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert time.perf_counter() - t0 < 5.0
    assert peak < 20 * 2**20  # bytes: nothing of size N or N^2 was allocated


@pytest.mark.finding("10.06")
def test_hard_limit_can_be_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "ring", name="g1", center=(0.0, 0.0, 0.7), radius=5.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    monkeypatch.setattr(diagnostics, "HARD_LIMIT", 50)
    with pytest.raises(diagnostics.ResourceLimitError):
        w.solve(_engine(0.5))  # 63 segments
    monkeypatch.setattr(diagnostics, "HARD_LIMIT", 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = w.solve(_engine(0.5))
    assert len(res.point_sources) == 63


@pytest.mark.finding("10.06")
def test_mutual_matrix_path_is_guarded_too(monkeypatch: pytest.MonkeyPatch) -> None:
    from groundfield.solver.mutual import solve_mutual_matrix

    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "ring", name="g1", center=(0.0, 0.0, 0.7), radius=5.0)
    monkeypatch.setattr(diagnostics, "HARD_LIMIT", 50)
    with pytest.raises(diagnostics.ResourceLimitError):
        solve_mutual_matrix(w, _engine(0.5), ["g1"], np.array([[0.0, 0.0, 0.7]]))


def test_expected_segments_is_exact_for_rods_split_at_the_interface() -> None:
    """The prediction follows the solver's discretisation, layers included."""
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=1.3))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image_2layer", segment_length=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_solved = len(w.solve(eng).point_sources)
    assert expected_segments(w, eng)["total"] == n_solved == 7


def test_expected_segments_does_not_materialise_huge_electrodes() -> None:
    tracemalloc.start()
    try:
        pred = expected_segments(_mm_ring(), _engine(0.5))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert pred["total"] >= 62832
    assert peak < 5 * 2**20
    assert math.isfinite(pred["total"])


# ---------------------------------------------------------------------
# Audit round 2, A2S-02: conductor sub-pieces count against the limit
# ---------------------------------------------------------------------


def _chain_world(n_pieces: int, coupling: str, inductance_model=None) -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="a", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="b", position=(100.0, 0.0, 0.5), length=3.0)
    kw = {} if inductance_model is None else {"inductance_model": inductance_model}
    gf.create_conductor(
        w,
        start="a",
        end="b",
        name="pen",
        cross_section=50e-6,
        discretize_segment_length=100.0 / n_pieces,
        coupling_to_soil=coupling,
        **kw,
    )
    gf.create_source(w, attached_to="a")
    return w


@pytest.mark.finding("10.06")
@pytest.mark.parametrize("coupling", ["isolated", "galvanic"])
def test_conductor_chain_beyond_the_limit_is_rejected(
    monkeypatch: pytest.MonkeyPatch, coupling: str
) -> None:
    """1000 isolated sub-pieces = 1000 branches + 999 interior nodes.

    The base counted 12 segments for the isolated chain and solved it (the
    dense nodal system grows with the square of the piece count: 407 MiB
    for 2000 pieces); the galvanic chain was rejected through its leakage
    segments already.
    """
    # audit A2S-02
    monkeypatch.setattr(diagnostics, "HARD_LIMIT", 200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(diagnostics.ResourceLimitError, match="unknowns|segments"):
            _chain_world(1000, coupling).solve(_engine(0.5))


@pytest.mark.finding("10.06")
@pytest.mark.parametrize("inductance_model", [None, "neumann"])
def test_a_chain_in_millimetre_pieces_is_rejected_before_allocation(
    inductance_model,
) -> None:
    """A 100 m PEN with discretize_segment_length in mm: 100 000 pieces.

    Rejected at the default limit in closed form, quickly and without
    allocating the chain (the inductive one before the inductance
    assembly).
    """
    # audit A2S-02
    w = _chain_world(100_000, "isolated", inductance_model)
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(diagnostics.ResourceLimitError, match="200003 unknowns"):
                w.solve(_engine(0.5))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert time.perf_counter() - t0 < 5.0
    assert peak < 20 * 2**20  # bytes


@pytest.mark.finding("10.06")
@pytest.mark.parametrize(("coupling", "expected"), [("isolated", 23), ("galvanic", 35)])
def test_nodal_unknowns_are_the_size_of_the_solved_system(
    monkeypatch: pytest.MonkeyPatch, coupling: str, expected: int
) -> None:
    """Two rods and a chain of 10 pieces: the predicted and solved sizes agree."""
    # audit A2S-02
    from groundfield.solver import _topology

    sizes: list[int] = []
    original = _topology._augmented_matrix

    def recording(Z, system, cluster_id):  # noqa: ANN001, ANN202
        A = original(Z, system, cluster_id)
        sizes.append(A.shape[0])
        return A

    monkeypatch.setattr(_topology, "_augmented_matrix", recording)
    w = _chain_world(10, coupling)
    engine = _engine(0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.solve(engine)
    assert expected_segments(w, engine)["nodal_unknowns"] == expected == sizes[0]


@pytest.mark.finding("10.06")
def test_diagnose_reports_a_nodal_system_beyond_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # audit A2S-02
    monkeypatch.setattr(diagnostics, "HARD_LIMIT", 200)
    report = diagnostics.diagnose(_chain_world(1000, "isolated"), _engine(0.5))
    budget = report.by_category("budget")
    assert budget and budget[0].data["n_unknowns"] == 1999 + 4
    assert not diagnostics.check_budget_ok(_chain_world(1000, "isolated"), _engine(0.5))


@pytest.mark.finding("10.70", "10.32")
def test_diagnose_reports_a_rod_end_at_the_interface() -> None:
    """Thin-wire limit at a rod end (audit A3S-05), without an engine too."""
    # audit A3S-05
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0))
    gf.create_electrode(
        w, "rod", name="tip", position=(0, 0, 0.5), length=4.5, wire_radius=0.008
    )
    gf.create_electrode(
        w, "rod", name="far", position=(20, 0, 0.5), length=4.0, wire_radius=0.008
    )
    gf.create_source(w, attached_to="tip")
    for report in (diagnostics.diagnose(w), diagnostics.diagnose(w, _engine(0.5))):
        found = report.by_category("interface_contact")
        assert [d.subjects for d in found] == [("tip",)]
        assert found[0].severity == "warning"
        assert found[0].data["end"] == "tip"
        assert found[0].data["offset_m"] == pytest.approx(0.0, abs=1e-12)
        assert "interface contact: 1 rod end lies within one wire radius" in str(report)
