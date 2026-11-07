"""Tests for the direct-Sommerfeld MoM backend (``mom_sommerfeld``).

The Sommerfeld-quadrature kernel is the reference kernel within the
layered family (the discretisation is ``mom``'s point matching). These
tests check that on homogeneous soil it agrees with ``image``, that on
2-layer soil it agrees with the closed-form Tagg/Sunde series, that the
kernel collapses correctly to the homogeneous limit when the layer
contrasts vanish, and that non-convergence is never silent (10.47).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver._layered import LayerStack
from groundfield.solver.mom_sommerfeld import sommerfeld_kernel_value

SEG = 0.10  # coarser segments — Sommerfeld quadrature is expensive


def _world(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_kernel_homogeneous_closed_form() -> None:
    """The kernel on a 1-layer stack is the homogeneous closed form."""
    stack = LayerStack(rhos=np.array([100.0]), h=np.zeros(0))
    s = 1.0
    z = 0.5
    z_s = 1.0
    G = sommerfeld_kernel_value(stack, s=s, z=z, z_s=z_s)
    r = float(np.sqrt(s**2 + (z - z_s) ** 2))
    r_img = float(np.sqrt(s**2 + (z + z_s) ** 2))
    expected = 1.0 / r + 1.0 / r_img
    assert G == pytest.approx(expected, rel=1e-6)


def test_kernel_two_layer_K_zero() -> None:
    """For ρ₁ = ρ₂ the layered kernel must equal the homogeneous one."""
    rho = 100.0
    stack = LayerStack(rhos=np.array([rho, rho]), h=np.array([2.0]))
    s, z, z_s = 1.0, 0.5, 1.0
    G = sommerfeld_kernel_value(stack, s=s, z=z, z_s=z_s)
    r = float(np.sqrt(s**2 + (z - z_s) ** 2))
    r_img = float(np.sqrt(s**2 + (z + z_s) ** 2))
    expected = 1.0 / r + 1.0 / r_img
    assert G == pytest.approx(expected, rel=1e-4)


def test_mom_sommerfeld_homogeneous() -> None:
    """``mom_sommerfeld`` on homogeneous soil agrees with ``image`` ≤ 5 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng_image = gf.create_engine(backend="image", segment_length=SEG)
    eng_som = gf.create_engine(backend="mom_sommerfeld", segment_length=SEG)
    Z_image = eng_image.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_som = eng_som.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert abs(Z_som - Z_image) / Z_image < 0.05


@pytest.mark.parametrize("rho_2", [200.0, 500.0])
def test_mom_sommerfeld_two_layer_matches_image_2layer(rho_2: float) -> None:
    """``mom_sommerfeld`` agrees with ``image_2layer`` to ≤ 5 %."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=rho_2, h_1=2.0)
    eng_il2 = gf.create_engine(backend="image_2layer", segment_length=SEG)
    eng_som = gf.create_engine(backend="mom_sommerfeld", segment_length=SEG)
    Z_il2 = eng_il2.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_som = eng_som.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert abs(Z_som - Z_il2) / Z_il2 < 0.05


def test_mom_sommerfeld_metadata() -> None:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    eng = gf.create_engine(backend="mom_sommerfeld", segment_length=SEG)
    res = eng.solve(_world(soil))
    assert res.backend == "mom_sommerfeld"
    # Point matching, like mom and bem (10.31; "galerkin" up to 0.15.0).
    assert res.metadata["solver"] == "point_matching"
    assert "lambda_max_factor" in res.metadata


# ---------------------------------------------------------------------
# 10.47 — non-convergence is never silent
# ---------------------------------------------------------------------


def _exact_two_layer_series(K: float, h: float, s: float, z: float, zs: float) -> float:
    """Tagg/Sunde image series of the n = 2 top-layer kernel (independent)."""
    adz, p = abs(z - zs), z + zs
    m_max = int(np.ceil(np.log(1e-17) / np.log(abs(K)))) + 10
    m = np.arange(m_max + 1, dtype=float)

    def r(d):
        return np.maximum(np.hypot(s, d), 1e-3)

    terms = K**m * (
        1 / r(2 * m * h + adz)
        + K / r(2 * (m + 1) * h - adz)
        + 1 / r(2 * m * h + p)
        + K / r(2 * (m + 1) * h - p)
    )
    return float(np.sum(terms[::-1]))


#: (h_1, rho_2/rho_1, z, z_s, s): pairs 5 km apart over a 5 cm top layer.
#: The J_0 oscillation needs ~5.5e5 half-period panels against the
#: default budget of 2e5; on v0.15.0 the absolute floor epsabs = 1e-9
#: let the refinement ladder "converge" on the capped grid and the
#: kernel came back up to 3.1e-3 wrong without a warning (R2-07).
_BUDGET_BOUND = [
    (0.05, 0.0005, 0.045, 0.047, 5000.0),
    (0.05, 0.0005, 0.025, 0.025, 5000.0),
    (0.05, 0.0005, 0.048, 0.048, 5000.0),
    (0.05, 1 / 33.3, 0.045, 0.047, 5000.0),
    (0.05, 1 / 33.3, 0.025, 0.025, 5000.0),
]


@pytest.mark.finding("10.47")
@pytest.mark.parametrize(("h", "ratio", "z", "zs", "s"), _BUDGET_BOUND)
def test_kernel_warns_or_converges_when_the_oscillation_budget_binds(
    h, ratio, z, zs, s
) -> None:
    from groundfield.solver.mom_sommerfeld import SommerfeldConvergenceWarning

    stack = LayerStack(rhos=np.array([100.0, 100.0 * ratio]), h=np.array([h]))
    ref = _exact_two_layer_series(float(stack.K[0]), h, s, z, zs)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        g = sommerfeld_kernel_value(stack, s=s, z=z, z_s=zs)
    warned = [w for w in rec if issubclass(w.category, SommerfeldConvergenceWarning)]
    assert warned or abs(g - ref) / abs(ref) < 1e-6
    if warned:
        assert "max_osc_panels" in str(warned[0].message)


@pytest.mark.finding("10.47")
@pytest.mark.parametrize(
    ("h", "ratio", "z", "zs", "s"),
    [
        (0.2, 1 / 33.3, 0.1, 0.1, 5000.0),
        (1.0, 1 / 33.3, 0.5, 0.5, 5000.0),
        (5.0, 1 / 33.3, 0.8, 0.8, 400.0),  # AP1-like contrast and span
    ],
)
def test_kernel_stays_silent_when_the_oscillation_is_resolved(
    h, ratio, z, zs, s
) -> None:
    """No false alarm: full coverage up to the last partial half period."""
    from groundfield.solver.mom_sommerfeld import SommerfeldConvergenceWarning

    stack = LayerStack(rhos=np.array([100.0, 100.0 * ratio]), h=np.array([h]))
    ref = _exact_two_layer_series(float(stack.K[0]), h, s, z, zs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SommerfeldConvergenceWarning)
        g = sommerfeld_kernel_value(stack, s=s, z=z, z_s=zs)
    assert g == pytest.approx(ref, rel=1e-9)


def _ring_world(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "ring", name="g1", center=(0.0, 0.0, 0.5), radius=2.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


@pytest.mark.finding("10.47")
def test_solve_reports_convergence_in_the_metadata() -> None:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=1.0)
    eng = gf.create_engine(
        backend="mom_sommerfeld", segment_length=1.0, frequencies=[50.0, 1000.0]
    )
    res = eng.solve(_ring_world(soil))
    assert res.metadata["converged"] is True
    assert res.metadata["converged_per_frequency"] == [True, True]
    assert res.metadata["n_unconverged_entries"] == 0


@pytest.mark.finding("10.47")
def test_solve_flags_a_capped_oscillation_budget() -> None:
    """A tiny panel budget leaves the J_0 oscillation unresolved for the
    farther ring segments: one aggregated warning, converged False."""
    from groundfield.solver.mom_sommerfeld import SommerfeldConvergenceWarning

    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=1.0)
    eng = gf.Engine(
        backend="mom_sommerfeld",
        segment_length=1.0,
        frequencies=[50.0, 1000.0],
        sommerfeld_max_osc_panels=3,
    )
    with pytest.warns(SommerfeldConvergenceWarning, match="max_osc_panels"):
        res = eng.solve(_ring_world(soil))
    assert res.metadata["converged"] is False
    assert res.metadata["converged_per_frequency"] == [False, False]
    assert res.metadata["n_unconverged_entries"] > 0


@pytest.mark.finding("10.47")
def test_solve_flags_a_truncated_two_layer_diagonal() -> None:
    """The n = 2 diagonal is the Tagg/Sunde series: a truncated series is
    reported like in ``mom`` (R2-09: +0.02 % and silent in v0.15.0)."""
    from groundfield.solver.image_2layer import SeriesTruncationWarning

    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "ring", name="g1", center=(0.0, 0.0, 0.8), radius=2.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.Engine(backend="mom_sommerfeld", segment_length=1.0, image_max_terms=20)
    with pytest.warns(SeriesTruncationWarning):
        res = eng.solve(w)
    assert res.metadata["converged"] is False


# ---------------------------------------------------------------------
# 10.61 — one quadrature per distinct pair argument; depth-group tables
# ---------------------------------------------------------------------

_THREE_LAYERS = gf.MultiLayerSoil(
    layers=[
        gf.SoilLayer(rho=1000.0, h=5.0),
        gf.SoilLayer(rho=30.0, h=10.0),
        gf.SoilLayer(rho=200.0, h=None),
    ]
)


def _grid_arrays(size: float, n_mesh: int, ds: float):
    """Segment arrays of a square grid mesh at 0.8 m (the B11/B12 layout)."""
    from groundfield.solver._pipeline import SegmentArrays, discretise

    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    half = 0.5 * size
    gf.create_electrode(
        w,
        "grid_mesh",
        name="g1",
        corner=(-half, -half, 0.8),
        size=(size, size),
        n_x=n_mesh,
        n_y=n_mesh,
    )
    disc = discretise(w, gf.create_engine(segment_length=ds), None)
    return SegmentArrays.from_segments(disc.segments)


_QUAD = dict(lambda_max_factor=200.0, epsabs=1e-9, epsrel=1e-7, max_osc_panels=200_000)


@pytest.mark.finding("10.61")
def test_one_quadrature_per_distinct_pair_argument(monkeypatch) -> None:
    """A ring at one depth needs one quadrature per distinct distance.

    v0.15.0 integrated every ordered pair (N(N-1) remainder quadratures,
    about 0.6 ms each, R2-06); the kernel of a pair depends only on the
    distance and the two depths, and is symmetric for two top-layer
    points.
    """
    from groundfield.solver import mom_sommerfeld as ms

    calls = {"n": 0}
    original = ms._integrate_panels

    def count(func, edges):
        calls["n"] += 1
        return original(func, edges)

    monkeypatch.setattr(ms, "_integrate_panels", count)
    w = gf.create_world(soil=_THREE_LAYERS)
    gf.create_electrode(w, "ring", name="g1", center=(0.0, 0.0, 0.8), radius=2.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res = gf.Engine(backend="mom_sommerfeld", segment_length=0.25).solve(w)
    n = res.metadata["n_segments"]
    assert n >= 48
    # Every quadrature runs at least two refinement rungs, so the pair loop
    # of v0.15.0 integrated at least 2 N (N - 1) times (5100 here). Ring
    # distances repeat up to round-off: 176 integrations remain.
    assert calls["n"] <= 0.1 * 2 * n * (n - 1), (calls["n"], n)


@pytest.mark.finding("10.61")
def test_tabulated_reaction_matrix_is_within_1e_6_of_the_quadrature() -> None:
    """Depth-group tables reproduce the pointwise matrix to 1e-6 (B12 criterion)."""
    from groundfield.solver._layered import as_layer_stack
    from groundfield.solver.mom_sommerfeld import _build_Z_sommerfeld

    a = _grid_arrays(20.0, 2, 0.5)
    stack = as_layer_stack(_THREE_LAYERS)
    report: dict = {}
    Z_tab = _build_Z_sommerfeld(
        a.points, a.lengths, a.radii, stack, tabulate=True, report=report, **_QUAD
    )
    Z_ref = _build_Z_sommerfeld(
        a.points, a.lengths, a.radii, stack, tabulate=False, **_QUAD
    )
    assert report["groups_tabulated"] == 1
    assert 0.0 < report["max_check_error"] <= 1e-7
    assert np.max(np.abs(Z_tab / Z_ref - 1.0)) <= 1e-6
    # The diagonal never comes from a table.
    assert np.array_equal(np.diag(Z_tab), np.diag(Z_ref))


@pytest.mark.finding("10.61")
def test_a_table_that_fails_its_check_is_not_used(monkeypatch) -> None:
    """A group whose table misses the check tolerance is evaluated pointwise."""
    from groundfield.solver import mom_sommerfeld as ms
    from groundfield.solver._layered import as_layer_stack

    monkeypatch.setattr(ms, "_TAB_TOL", 0.0)
    a = _grid_arrays(20.0, 2, 0.5)
    stack = as_layer_stack(_THREE_LAYERS)
    report: dict = {}
    Z = ms._build_Z_sommerfeld(
        a.points, a.lengths, a.radii, stack, tabulate=True, report=report, **_QUAD
    )
    Z_ref = ms._build_Z_sommerfeld(
        a.points, a.lengths, a.radii, stack, tabulate=False, **_QUAD
    )
    assert report["groups_tabulated"] == 0
    assert np.array_equal(Z, Z_ref)


@pytest.mark.finding("10.61")
def test_solve_reports_the_kernel_tabulation() -> None:
    w = gf.create_world(soil=_THREE_LAYERS)
    gf.create_electrode(
        w,
        "grid_mesh",
        name="g1",
        corner=(-10.0, -10.0, 0.8),
        size=(20.0, 20.0),
        n_x=2,
        n_y=2,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res = gf.Engine(backend="mom_sommerfeld", segment_length=0.5).solve(w)
    tab = res.metadata["kernel_tabulation"]
    assert tab["enabled"] is True
    assert tab["groups_tabulated"] == 1
    assert 0.0 < tab["max_check_error"] <= 1e-7
    # Two layers: the kernel check of the series is never tabulated.
    res_2 = gf.Engine(backend="mom_sommerfeld", segment_length=1.0).solve(
        _ring_world(gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=5.0))
    )
    assert res_2.metadata["kernel_tabulation"]["enabled"] is False


@pytest.mark.finding("10.61")
def test_engine_option_switches_the_tabulation_off() -> None:
    """``options={"mom_sommerfeld": {"tabulate": False}}`` reaches the solver."""
    w = gf.create_world(soil=_THREE_LAYERS)
    gf.create_electrode(
        w,
        "grid_mesh",
        name="g1",
        corner=(-10.0, -10.0, 0.8),
        size=(20.0, 20.0),
        n_x=2,
        n_y=2,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.Engine(
        backend="mom_sommerfeld",
        segment_length=0.5,
        options={"mom_sommerfeld": {"tabulate": False}},
    )
    tab = eng.solve(w).metadata["kernel_tabulation"]
    assert tab["enabled"] is False and tab["groups_tabulated"] == 0
