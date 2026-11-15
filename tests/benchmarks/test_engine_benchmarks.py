"""Engine benchmark suite B1–B13 (review pass 10, R2 §3.3).

Reference worlds with analytic or independently computed reference
values; every bound comes from :mod:`tests.benchmarks.tolerances`, which
the documentation quotes. Skipped by default; run with::

    pytest tests/benchmarks --run-benchmark
    GF_BENCHMARK_REPORT=measured.json pytest tests/benchmarks --run-benchmark

The slowest benchmarks (B4: ~100 s, B11: ~30 s) are additionally marked
``slow``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver._layered import LayerStack
from tests.benchmarks import references as ref
from tests.benchmarks.tolerances import bound

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

H100 = gf.HomogeneousSoil(resistivity=100.0)


def _solve(backend: str, world: gf.World, ds: float, **kw) -> gf.FieldResult:
    # Engine cross-checks compare the solvers' current models with the
    # point kernel of the mom family; the image family runs the exact line
    # kernel by default since 10.28 (about 1-2 % above point-matching).
    kw.setdefault("kernel", "point")
    return gf.create_engine(backend=backend, segment_length=ds, **kw).solve(world)


def _z(backend: str, world: gf.World, ds: float, name: str = "g1") -> float:
    return _solve(backend, world, ds).cluster_impedance(name)[0].real


def _rod(soil, *, top: float = 0.0, length: float = 3.0, radius: float | None = None):
    w = gf.create_world(soil=soil)
    kw = {} if radius is None else {"wire_radius": radius}
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, top), length=length, **kw)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _ring(soil, *, r0: float = 2.0, z: float = 0.8, radius: float | None = None):
    w = gf.create_world(soil=soil)
    kw = {} if radius is None else {"wire_radius": radius}
    gf.create_electrode(w, "ring", name="g1", center=(0, 0, z), radius=r0, **kw)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _check(measured, benchmark, backend, case, deviation, tol=None):
    b = bound(benchmark, backend) if tol is None else tol
    limit = b.bound if hasattr(b, "bound") else b
    measured(benchmark, backend, case, deviation, limit)
    assert deviation <= limit, (
        f"{benchmark} {backend} {case}: {deviation:.3e} > {limit:.1e}"
    )


# ---------------------------------------------------------------------
# B1 — surface disc (fem discretisation alone; user decision Q37)
# ---------------------------------------------------------------------


def test_b1_fem_surface_disc(measured) -> None:
    from groundfield.solver.fem._assembly import solve_ports
    from groundfield.solver.fem._mesh import MeshParameters, RodGeometry, build_mesh

    r_exact = 100.0 / 4.0  # Weber: rho / (4 b), b = 1 m

    def resistance(level: int) -> float:
        coin = RodGeometry(radius=1.0, z_top=0.0, z_bottom=1e-5)
        mesh = build_mesh([coin], [], [], 3000.0, MeshParameters(), level)
        return 1.0 / solve_ports(mesh, np.array([100.0]), [0.0]).conductance[0, 0]

    ladder = [resistance(level) for level in (0, 1, 2)]
    d = np.diff(ladder)
    assert (d > 0).all(), ladder  # monotone from below
    extrapolated = ladder[1] + d[0] / 3.0  # the default: levels 0 and 1
    _check(measured, "B1", "fem", "levels 0/1", abs(extrapolated / r_exact - 1.0))
    order = math.log2(d[0] / d[1])
    measured("B1", "fem", "observed order", order, None)
    assert 1.4 < order < 2.4, order


# ---------------------------------------------------------------------
# B2 / B3 — rod and ring in homogeneous soil
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def rod_limit() -> float:
    return ref.rod_thin_wire_limit(3.0, 0.008, 100.0)


@pytest.mark.parametrize("ds", [0.5, 0.25, 0.125])
@pytest.mark.parametrize("backend", ["image", "mom", "fem"])
def test_b2_rod_homogeneous(backend, ds, rod_limit, measured) -> None:
    z = _z(backend, _rod(H100, radius=0.008), ds)
    _check(measured, "B2", backend, f"ds={ds}", abs(z / rod_limit - 1.0))


@pytest.mark.parametrize("ds", [0.5, 0.25, 0.125])
@pytest.mark.parametrize("backend", ["image", "mom", "fem"])
def test_b3_ring_homogeneous(backend, ds, measured) -> None:
    r_ref = ref.ring_thin_wire_resistance(100.0, 2.0, 0.008, 0.8)
    z = _z(backend, _ring(H100, radius=0.008), ds)
    _check(measured, "B3", backend, f"ds={ds}", abs(z / r_ref - 1.0))


# ---------------------------------------------------------------------
# B4 / B5 — Sommerfeld kernel against series and mpmath references
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_b4_two_layer_kernel_is_accurate_or_warns(measured) -> None:
    from groundfield.solver.mom_sommerfeld import (
        SommerfeldConvergenceWarning,
        sommerfeld_kernel_value,
    )

    tol = bound("B4", "mom_sommerfeld").bound
    worst_silent, warned_in_envelope = 0.0, []
    for h in (0.05, 0.2, 1.0, 5.0, 30.0):
        for ratio in (1 / 2000, 1 / 33.3, 1 / 3, 3, 33.3, 2000):
            stack = LayerStack(rhos=np.array([100.0, 100.0 * ratio]), h=np.array([h]))
            k = float(stack.K[0])
            depths = [
                (0.01 * h, 0.01 * h),
                (0.5 * h, 0.5 * h),
                (h - 2e-3, h - 2e-3),
                (0.6 * h, 0.6 * h + min(0.02, 0.1 * h)),
                (0.9 * h, 0.9 * h + 0.002),
            ]
            for z, zs in depths:
                for s in (0.0, 0.01, 0.5, 5.0, 50.0, 500.0, 5000.0):
                    g_ref = ref.two_layer_series_kernel(k, h, s, z, zs)
                    with warnings.catch_warnings(record=True) as rec:
                        warnings.simplefilter("always")
                        g = sommerfeld_kernel_value(stack, s=s, z=z, z_s=zs)
                    warned = any(
                        issubclass(w.category, SommerfeldConvergenceWarning)
                        for w in rec
                    )
                    if warned:
                        if h >= 0.2:
                            warned_in_envelope.append((h, ratio, s, z, zs))
                        continue
                    worst_silent = max(worst_silent, abs(g - g_ref) / abs(g_ref))
    measured("B4", "mom_sommerfeld", "max silent error", worst_silent, tol)
    assert worst_silent <= tol
    assert warned_in_envelope == []


def test_b5_three_layer_kernel(measured) -> None:
    from groundfield.solver.mom_sommerfeld import sommerfeld_kernel_value

    tol = bound("B5", "mom_sommerfeld").bound
    worst = 0.0
    for h1 in (0.5, 2.0, 5.0):
        for h2 in (0.01, 0.3, 3.0):
            for r1, r2 in ((100.0, 1000.0), (1000.0, 30.0), (100.0, 10.0)):
                k = (r2 - r1) / (r2 + r1)
                dup = LayerStack(rhos=np.array([r1, r2, r2]), h=np.array([h1, h2]))
                top = LayerStack(rhos=np.array([r1, r1, r2]), h=np.array([h1, h2]))
                for z, zs in (
                    (0.3 * h1, 0.3 * h1),
                    (0.9 * h1, 0.91 * h1),
                    (0.05, 0.05),
                ):
                    for s in (0.0, 0.3, 3.0, 30.0, 300.0):
                        g = sommerfeld_kernel_value(dup, s=s, z=z, z_s=zs)
                        g_ref = ref.two_layer_series_kernel(k, h1, s, z, zs)
                        worst = max(worst, abs(g - g_ref) / abs(g_ref))
                        g = sommerfeld_kernel_value(top, s=s, z=z, z_s=zs)
                        g_ref = ref.two_layer_series_kernel(k, h1 + h2, s, z, zs)
                        worst = max(worst, abs(g - g_ref) / abs(g_ref))
    measured("B5", "mom_sommerfeld", "duplicated stacks", worst, tol)
    assert worst <= tol
    genuine = LayerStack(rhos=np.array([100.0, 1000.0, 50.0]), h=np.array([2.0, 1.0]))
    worst = 0.0
    for s, z, zs in (
        (0.0, 0.8, 0.8),
        (3.0, 0.8, 0.8),
        (30.0, 1.9, 1.9),
        (30.0, 0.5, 1.2),
    ):
        g = sommerfeld_kernel_value(genuine, s=s, z=z, z_s=zs)
        g_ref = ref.three_layer_kernel_mpmath(
            (100.0, 1000.0, 50.0), (2.0, 1.0), s, z, zs
        )
        worst = max(worst, abs(g - g_ref) / abs(g_ref))
    measured("B5", "mom_sommerfeld", "genuine three layers", worst, tol)
    assert worst <= tol


# ---------------------------------------------------------------------
# B6 / B7 — two-layer soil, top layer and across the interface
# ---------------------------------------------------------------------

_B6_SOILS = [
    (1000.0, 30.0, 5.0),
    (1000.0, 30.0, 10.0),
    (1000.0, 30.0, 30.0),
    (30.0, 200.0, 5.0),
    (100.0, 400.0, 2.0),
]


@pytest.mark.parametrize("electrode", ["ring", "rod"])
@pytest.mark.parametrize(("rho_1", "rho_2", "h_1"), _B6_SOILS)
def test_b6_top_layer_electrodes(rho_1, rho_2, h_1, electrode, measured) -> None:
    soil = gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=h_1)

    def world():
        return _ring(soil) if electrode == "ring" else _rod(soil, length=1.5)

    z_mom = _z("mom", world(), 0.25)
    case = f"{electrode} {rho_1:g}/{rho_2:g}/h={h_1:g}"
    for backend in ("image_2layer", "mom_sommerfeld", "fem"):
        z = _z(backend, world(), 0.25)
        _check(measured, "B6", backend, case, abs(z / z_mom - 1.0))


@pytest.mark.parametrize(
    ("rho_1", "rho_2", "z_ref"), [(1000.0, 30.0, 25.6913), (100.0, 1000.0, 64.0749)]
)
def test_b7_rod_through_interface_mom_family(rho_1, rho_2, z_ref, measured) -> None:
    soil = gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=2.0)
    z_mom = _z("mom", _rod(soil), 0.25)
    assert z_mom == pytest.approx(z_ref, abs=5e-5)
    z = _z("mom_sommerfeld", _rod(soil), 0.25)
    _check(
        measured, "B7", "mom_sommerfeld", f"{rho_1:g}/{rho_2:g}", abs(z / z_mom - 1.0)
    )


@pytest.mark.parametrize(("rho_1", "rho_2"), [(1000.0, 30.0), (100.0, 1000.0)])
@pytest.mark.parametrize("backend", ["image_2layer", "fem"])
def test_b7_rod_through_interface_image_2layer(backend, rho_1, rho_2, measured) -> None:
    soil = gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=2.0)
    z_mom = _z("mom", _rod(soil), 0.25)
    z = _z(backend, _rod(soil), 0.25)
    _check(measured, "B7", backend, f"{rho_1:g}/{rho_2:g}", abs(z / z_mom - 1.0))


# ---------------------------------------------------------------------
# B8 — reciprocity of transfer impedances
# ---------------------------------------------------------------------

_B8_SOILS = {
    "homogeneous": H100,
    "top_layer": gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0),
    "cross_layer": gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0),
}


def _two_port(soil, source: str) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="A", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "ring", name="B", center=(4.0, 0.0, 0.8), radius=1.5)
    gf.create_source(w, attached_to=source, magnitude=1.0)
    return w


@pytest.mark.parametrize("soil_label", list(_B8_SOILS))
@pytest.mark.parametrize(
    "backend", ["image", "image_2layer", "cim", "mom", "bem", "mom_sommerfeld"]
)
def test_b8_two_port_reciprocity(backend, soil_label, measured) -> None:
    soil = _B8_SOILS[soil_label]
    if (backend == "image") != (soil_label == "homogeneous"):
        pytest.skip("image is the homogeneous backend, image_2layer the layered one")
    # The backend's default kernel (the line kernel for the image family).
    r_a = _solve(backend, _two_port(soil, "A"), 0.5, kernel=None)
    r_b = _solve(backend, _two_port(soil, "B"), 0.5, kernel=None)
    z_ba = r_a.electrode_potentials["B"][0] / r_a.electrode_currents["A"][0]
    z_ab = r_b.electrode_potentials["A"][0] / r_b.electrode_currents["B"][0]
    _check(measured, "B8", backend, soil_label, abs(z_ab / z_ba - 1.0))
    if backend in ("mom", "bem", "mom_sommerfeld") and soil_label != "homogeneous":
        expected = {"top_layer": 17.28197, "cross_layer": 1.43970}[soil_label]
        assert z_ba.real == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------
# B9 — limits of the layered kernels
# ---------------------------------------------------------------------


def _multi(rho: float) -> gf.MultiLayerSoil:
    return gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=rho, thickness=0.5),
            gf.SoilLayer(resistivity=rho, thickness=2.0),
            gf.SoilLayer(resistivity=rho, thickness=None),
        ]
    )


@pytest.mark.parametrize(
    "backend", ["image_2layer", "cim", "mom", "bem", "mom_sommerfeld", "fem"]
)
def test_b9_equal_resistivity_is_homogeneous(backend, measured) -> None:
    homogeneous_backend = "image" if backend == "image_2layer" else backend
    z_hom = _z(homogeneous_backend, _ring(H100), 0.5)
    key = "equal_rho_fem" if backend == "fem" else "equal_rho_integral"
    for label, soil in (
        ("TwoLayer(100,100,h=0.5)", gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=0.5)),
        ("TwoLayer(100,100,h=30)", gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=30.0)),
        ("MultiLayer([100]*3)", _multi(100.0)),
    ):
        if backend == "image_2layer" and isinstance(soil, gf.MultiLayerSoil):
            continue  # image_2layer takes TwoLayerSoil only
        z = _z(backend, _ring(soil), 0.5)
        _check(measured, "B9", key, f"{backend} {label}", abs(z / z_hom - 1.0))


@pytest.mark.parametrize("backend", ["image_2layer", "mom", "mom_sommerfeld", "fem"])
@pytest.mark.parametrize(("h_1", "rho_limit"), [(4e-3, 30.0), (4e3, 1000.0)])
def test_b9_layer_thickness_limits(backend, h_1, rho_limit, measured) -> None:
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=h_1)
    z = _z(backend, _ring(soil), 0.5)
    homogeneous_backend = "image" if backend == "image_2layer" else backend
    z_lim = _z(
        homogeneous_backend, _ring(gf.HomogeneousSoil(resistivity=rho_limit)), 0.5
    )
    key = "layer_limit_fem" if backend == "fem" else "layer_limit_integral"
    _check(measured, "B9", key, f"{backend} h1={h_1:g}", abs(z / z_lim - 1.0))


# ---------------------------------------------------------------------
# B10 — ring with three bonded rods
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "soil",
    [H100, gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)],
    ids=["homogeneous", "2L-1000-30-5"],
)
def test_b10_ring_with_bonded_rods(soil, measured) -> None:
    def world():
        w = gf.create_world(soil=soil)
        gf.create_electrode(w, "ring", name="g1", center=(0, 0, 0.5), radius=2.0)
        for k, ang in enumerate((0.0, 2.1, 4.2)):
            gf.create_electrode(
                w,
                "rod",
                name=f"r{k}",
                position=(2 * np.cos(ang), 2 * np.sin(ang), 0.5),
                length=4.0,
            )
            gf.create_conductor(w, start="g1", end=f"r{k}")
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    z_mom = _z("mom", world(), 0.25)
    z = _z("image", world(), 0.25)
    _check(measured, "B10", "image", type(soil).__name__, abs(z / z_mom - 1.0))


# ---------------------------------------------------------------------
# B11 / B12 — grids
# ---------------------------------------------------------------------


def _grid(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w,
        "grid_mesh",
        name="g1",
        corner=(-20, -20, 0.8),
        size=(40.0, 40.0),
        n_x=4,
        n_y=4,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["image_2layer", "mom"])
def test_b11_grid_refinement(backend, measured) -> None:
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    zs = [_z(backend, _grid(soil), ds) for ds in (5.0, 2.5, 1.25)]
    for coarse, fine, label in ((zs[0], zs[1], "5->2.5"), (zs[1], zs[2], "2.5->1.25")):
        _check(measured, "B11", backend, label, abs(coarse / fine - 1.0))


@pytest.mark.slow
def test_b11_grid_mom_sommerfeld_equals_mom(measured) -> None:
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    z_mom = _z("mom", _grid(soil), 5.0)
    z = _z("mom_sommerfeld", _grid(soil), 5.0)
    _check(measured, "B11", "mom_sommerfeld", "ds=5", abs(z / z_mom - 1.0))


@pytest.mark.parametrize("ds", [2.5, 1.25])
def test_b12_three_layer_grid_interpolation(ds, measured) -> None:
    """Tabulated against quadrature-only ``mom_sommerfeld`` (10.61)."""
    from groundfield.solver._layered import as_layer_stack
    from groundfield.solver._pipeline import SegmentArrays, discretise
    from groundfield.solver.mom_sommerfeld import (
        _build_Z_sommerfeld,
        solve_mom_sommerfeld,
    )

    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=1000.0, h=5.0),
            gf.SoilLayer(rho=30.0, h=10.0),
            gf.SoilLayer(rho=200.0, h=None),
        ]
    )
    engine = gf.create_engine(backend="mom_sommerfeld", segment_length=ds)
    disc = discretise(_grid(soil), engine, (5.0,))
    a = SegmentArrays.from_segments(disc.segments)
    quad = dict(
        lambda_max_factor=200.0, epsabs=1e-9, epsrel=1e-7, max_osc_panels=200_000
    )
    stack = as_layer_stack(soil)
    Z_tab = _build_Z_sommerfeld(
        a.points, a.lengths, a.radii, stack, tabulate=True, **quad
    )
    Z_ref = _build_Z_sommerfeld(
        a.points, a.lengths, a.radii, stack, tabulate=False, **quad
    )
    entry = float(np.max(np.abs(Z_tab / Z_ref - 1.0)))
    _check(measured, "B12", "mom_sommerfeld", f"matrix ds={ds}", entry)
    z_tab = solve_mom_sommerfeld(_grid(soil), engine, tabulate=True)
    z_ref = solve_mom_sommerfeld(_grid(soil), engine, tabulate=False)
    dev = abs(z_tab.cluster_impedance("g1")[0] / z_ref.cluster_impedance("g1")[0] - 1.0)
    _check(measured, "B12", "mom_sommerfeld", f"Z ds={ds}", dev)


# ---------------------------------------------------------------------
# B13 — inductive line, complex comparison at every frequency
# ---------------------------------------------------------------------


def test_b13_inductive_line_image_vs_mom(measured) -> None:
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, "ring", name="g1", center=(0, 0, 0.8), radius=2.0)
    gf.create_electrode(w, "rod", name="far", position=(400.0, 0.0, 0.5), length=3.0)
    gf.create_conductor(
        w,
        start="g1",
        end="far",
        discretize_segment_length=50.0,
        coupling_to_soil="isolated",
        inductance_model="neumann",
        cross_section=95e-6,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    engines = {
        b: gf.create_engine(
            backend=b,
            segment_length=1.0,
            frequencies=[50.0, 1000.0],
            earth_inductive_model="sommerfeld",
        )
        for b in ("image", "mom")
    }
    tol = bound("B13", "image_vs_mom").bound
    rep = gf.compare_engines(w, engines, rel_tolerance=tol)
    worst = max(r.deviation for r in rep.rows)
    measured("B13", "image_vs_mom", "worst row", worst, tol)
    covered = {(r.frequency, r.component) for r in rep.rows}
    assert covered == {(50.0, "Re"), (50.0, "Im"), (1000.0, "Re"), (1000.0, "Im")}
    assert rep.is_consistent, rep.summary()


def _b13_line_world(length: float) -> gf.World:
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, "ring", name="g1", center=(0, 0, 0.8), radius=2.0)
    gf.create_electrode(w, "rod", name="far", position=(length, 0.0, 0.5), length=3.0)
    gf.create_conductor(
        w,
        name="line",
        start="g1",
        end="far",
        discretize_segment_length=50.0,
        coupling_to_soil="isolated",
        inductance_model="neumann",
        cross_section=95e-6,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _line_impedance(length: float, frequency: float) -> complex:
    """Series impedance of the isolated line: (phi_g1 - phi_far) / I_far."""
    engine = gf.create_engine(
        backend="image",
        segment_length=1.0,
        frequencies=[frequency],
        earth_inductive_model="sommerfeld",
    )
    res = engine.solve(_b13_line_world(length))
    v1 = res.electrode_potentials["g1"][0]
    v2 = res.electrode_potentials["far"][0]
    return (v1 - v2) / res.electrode_currents["far"][0]


@pytest.mark.slow
def test_b13_line_impedance_against_pollaczek(measured) -> None:
    """Per-metre series impedance of the B13 line against Pollaczek (1 kHz).

    The line carries no leakage, so (phi_g1 - phi_far) / I_far is its
    series impedance. The difference of a 1600 m and an 800 m line
    cancels the end effects of a finite line; the reference is the
    conductor-earth loop impedance R'_dc + Z'_P of an infinite buried
    line at the mean depth (references.pollaczek).
    """
    from groundfield.references.pollaczek import loop_impedance_conductor_earth

    frequency = 1000.0
    z_800, z_1600 = (_line_impedance(L, frequency) for L in (800.0, 1600.0))
    z_per_m = (z_1600 - z_800) / 800.0
    line = _b13_line_world(800.0).conductors[0]
    ref = loop_impedance_conductor_earth(
        100.0,
        100.0,
        10.0,
        frequency,
        depth=0.5 * (line.start[2] + line.end[2]),
        gmr=line.wire_radius * math.exp(-0.25),
        r_internal=line.resistivity / line.cross_section,
    )
    _check(measured, "B13", "pollaczek_x", "X' 1 kHz", abs(z_per_m.imag / ref.imag - 1))
    _check(measured, "B13", "pollaczek_r", "R' 1 kHz", abs(z_per_m.real / ref.real - 1))
