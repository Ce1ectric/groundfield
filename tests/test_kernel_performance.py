"""Memory and evaluation-count gates of the image-family kernels.

Review pass 10, findings 10.56–10.59, 10.62 and 10.65. Every gate is
deterministic (tracemalloc peaks, call counts, bitwise comparisons);
wall-clock benchmarks are marked ``benchmark`` and skipped by default.
"""

from __future__ import annotations

import time
import tracemalloc
import warnings

import numpy as np
import pytest

import groundfield as gf

# The private modules are imported lazily inside the tests, so that the
# behavioural gates (memory peak, identity product) also run — and fail —
# on v0.15.0, which has neither module.


def _k():
    from groundfield.solver import _kernels

    return _kernels


def _series(K: float, h_1: float, max_terms: int, tol: float):
    from groundfield.solver._series import ImageSeries

    return ImageSeries.from_coefficients(K, h_1, max_terms, tol)


# ---------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------


def _houses_world(soil, n_side: int = 4, *, pen: bool = False) -> gf.World:
    """``n_side²`` 10 × 10 m house grids (2 × 2 meshes), 25 m apart.

    At ``segment_length = 0.5`` every house has 120 segments, so
    ``n_side = 4`` gives N = 1920.
    """
    w = gf.create_world(soil=soil)
    names = []
    for i in range(n_side):
        for j in range(n_side):
            name = f"h{i}{j}"
            gf.create_electrode(
                w,
                "grid_mesh",
                name=name,
                corner=(25.0 * i, 25.0 * j, 0.8),
                size=(10.0, 10.0),
                n_x=2,
                n_y=2,
            )
            names.append(name)
    for k, name in enumerate(names[1:], start=1):
        gf.create_conductor(
            w,
            name=f"c{k}",
            start=names[k - 1],
            end=name,
            conductor_type="pen",
            cross_section=50e-6,
            coupling_to_soil="isolated",
            **(
                {"discretize_segment_length": 30.0, "inductance_model": "neumann"}
                if pen
                else {}
            ),
        )
    gf.create_source(w, attached_to=names[0], magnitude=1.0)
    return w


def _solve(world: gf.World, **engine_kw) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The bit-identity and performance gates are those of the point
        # kernel (the default up to 0.15.0, 10.28).
        engine_kw.setdefault("kernel", "point")
        return world.solve(gf.Engine(segment_length=0.5, **engine_kw))


def _peak_mb(fn) -> float:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 2**20
    finally:
        tracemalloc.stop()


# ---------------------------------------------------------------------
# 10.56 — no (N, N, 3) tensors
# ---------------------------------------------------------------------


@pytest.mark.finding("10.56")
def test_solve_memory_peak_is_bounded() -> None:
    """A 1920-segment solve stays far below 96 bytes per segment pair.

    v0.15.0 peaked at 338.5 MiB here (96 B·N² = 337.5 MiB); the reaction
    matrix alone is 8 B·N² = 28.1 MiB, the gate 16 B·N² = 56.25 MiB.
    """
    w = _houses_world(gf.HomogeneousSoil(resistivity=100.0))
    _solve(w)  # warm-up (imports, caches)
    n = 16 * 120
    peak = _peak_mb(lambda: _solve(w))
    assert peak < 16 * n * n / 2**20, f"peak {peak:.1f} MB"


@pytest.mark.finding("10.56")
def test_cross_layer_solve_memory_peak_is_bounded() -> None:
    """The ADR-0007 cross-layer path (house grids below h_1) stays below
    48 B·N²: the reaction matrix, the layered correction and the
    distance grid of one depth group, no ``(N, N, k)`` tensors."""
    soil = gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=0.5)
    w = _houses_world(soil)
    _solve(w)  # warm-up
    n = 16 * 120
    peak = _peak_mb(lambda: _solve(w))
    assert peak < 48 * n * n / 2**20, f"peak {peak:.1f} MB"


@pytest.mark.finding("10.56")
def test_block_mode_beyond_the_budget(monkeypatch) -> None:
    """Above ``MATRIX_BUDGET_BYTES`` the kernels never hold an N × N matrix.

    The block evaluation agrees with the materialised one to BLAS
    round-off.
    """
    w = _houses_world(gf.HomogeneousSoil(resistivity=100.0))
    ref = _solve(w)
    monkeypatch.setattr(_k(), "MATRIX_BUDGET_BYTES", 0)
    monkeypatch.setattr(_k(), "BLOCK_BYTES", 1 << 20)
    n = 16 * 120
    res = _solve(w)
    peak = _peak_mb(lambda: _solve(w))
    assert peak < 8 * n * n / 2**20, f"peak {peak:.1f} MB"
    for name, u in ref.electrode_potentials.items():
        np.testing.assert_allclose(res.electrode_potentials[name], u, rtol=1e-13)
        np.testing.assert_allclose(
            res.electrode_currents[name],
            ref.electrode_currents[name],
            rtol=1e-12,
            atol=1e-15,
        )


@pytest.mark.finding("10.56")
def test_reaction_matrix_entries_are_bit_identical_to_the_tensor_form() -> None:
    """Component-wise blocks reproduce ``np.linalg.norm`` over (N, N, 3)."""
    rng = np.random.default_rng(11)
    pts = rng.uniform(-20.0, 20.0, size=(300, 3))
    pts[:, 2] = rng.uniform(0.3, 3.0, size=300)
    img = pts.copy()
    img[:, 2] *= -1.0
    r = np.maximum(np.linalg.norm(pts[:, None] - pts[None], axis=2), 1e-3)
    ri = np.maximum(np.linalg.norm(pts[:, None] - img[None], axis=2), 1e-3)
    reference = (1.0 / r) + (1.0 / ri)
    block = _k()._point_field_kernel(pts, pts, 1e-3)
    np.testing.assert_array_equal(block, reference)


# ---------------------------------------------------------------------
# 10.57 — one reaction-matrix evaluation per solve
# ---------------------------------------------------------------------


@pytest.mark.finding("10.57")
@pytest.mark.parametrize("pen", [False, True], ids=["galvanic", "inductive_3f"])
def test_one_reaction_matrix_assembly_per_solve(monkeypatch, pen: bool) -> None:
    calls = []
    original = _k()._half_space_matrix

    def spy(*args, **kwargs):
        calls.append(args[0].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(_k(), "_half_space_matrix", spy)
    w = _houses_world(gf.HomogeneousSoil(resistivity=100.0), n_side=2, pen=pen)
    freqs = [50.0, 250.0, 1000.0] if pen else [50.0]
    _solve(w, frequencies=freqs, earth_inductive_model="perfect_mirror")
    assert len(calls) == 1


@pytest.mark.finding("10.57")
def test_one_series_assembly_per_two_layer_solve(monkeypatch) -> None:
    calls = []
    original = _k()._series_matrix

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_k(), "_series_matrix", spy)
    w = _houses_world(gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0), n_side=2)
    _solve(w)
    assert len(calls) == 1


# ---------------------------------------------------------------------
# 10.58 — two-layer series accumulated before the product
# ---------------------------------------------------------------------


def _per_image_reference(points: np.ndarray, currents: np.ndarray, series):
    """The v0.15.0 loop: one ``(1/r) @ currents`` product per image."""
    diff_xy = points[:, None, 0:2] - points[None, :, 0:2]
    delta_sq = np.einsum("mnk,mnk->mn", diff_xy, diff_xy)
    z_field = points[:, 2:3]
    z_src = points[None, :, 2]
    extra = np.zeros((points.shape[0],) + currents.shape[1:])
    for quad in series.iter_layer_terms():
        for term in quad:
            z_img = term.offset + term.sign * z_src
            r = np.sqrt(delta_sq + (z_field - z_img) ** 2)
            np.maximum(r, 1e-3, out=r)
            extra += term.weight * ((1.0 / r) @ currents)
    return extra


@pytest.mark.finding("10.58")
def test_series_matrix_equals_the_per_image_loop() -> None:
    """``S`` is the per-image loop applied to the identity, bit for bit;
    applied to currents it agrees with the loop to round-off."""
    rng = np.random.default_rng(3)
    pts = rng.uniform(-15.0, 15.0, size=(250, 3))
    pts[:, 2] = rng.uniform(0.4, 3.0, size=250)
    series = _series(-0.67, 5.0, 300, 1e-6)
    S = _k()._series_matrix(pts, series)
    np.testing.assert_array_equal(S, _per_image_reference(pts, np.eye(250), series))
    cur = rng.normal(size=(250, 3))
    ref = _per_image_reference(pts, cur, series)
    np.testing.assert_allclose(S @ cur, ref, rtol=0, atol=1e-13 * np.abs(ref).max())


@pytest.mark.finding("10.58")
def test_numba_series_kernel_is_bit_identical(monkeypatch) -> None:
    """The optional numba kernel gives exactly the numpy matrix."""
    pytest.importorskip("numba")
    rng = np.random.default_rng(5)
    pts = rng.uniform(-30.0, 30.0, size=(300, 3))
    pts[:, 2] = rng.uniform(0.4, 3.0, size=300)
    series = _series(-0.94, 5.0, 300, 1e-6)
    monkeypatch.setattr(_k(), "NUMBA_MIN_WORK", 0)
    with_numba = _k()._series_matrix(pts, series)
    monkeypatch.setattr(_k(), "USE_NUMBA", False)
    without = _k()._series_matrix(pts, series)
    np.testing.assert_array_equal(with_numba, without)


# ---------------------------------------------------------------------
# 10.59 — direct assembly in solve_mutual_matrix
# ---------------------------------------------------------------------


@pytest.mark.finding("10.59")
@pytest.mark.parametrize(
    "soil",
    [
        gf.HomogeneousSoil(resistivity=100.0),
        gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0),
    ],
    ids=["homogeneous", "two_layer"],
)
def test_mutual_matrix_without_identity_product(monkeypatch, soil) -> None:
    """No ``N × N`` identity is built (v0.15.0: ``kernel(eye(N))``, O(N³))."""
    w = gf.create_world(soil=soil)
    for k in range(3):
        gf.create_electrode(
            w, "rod", name=f"r{k}", position=(12.0 * k, 0.0, 0.5), length=3.0
        )
    n_segments = 3 * 6
    sizes = []
    original = np.eye

    def spy(n, *args, **kwargs):
        sizes.append(n)
        return original(n, *args, **kwargs)

    monkeypatch.setattr(np, "eye", spy)
    probes = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [24.0, 0.0, 0.0]])
    Z = gf.solve_mutual_matrix(
        w, gf.Engine(segment_length=0.5), ["r0", "r1", "r2"], probes
    )
    assert n_segments not in sizes
    assert np.all(np.isfinite(Z))


@pytest.mark.finding("10.59")
def test_direct_matrix_equals_the_identity_product() -> None:
    """``matrix()`` is bit-identical to the kernel applied to ``eye(N)``."""
    from groundfield.solver.image_2layer import _two_layer_self_kernel_factory

    rng = np.random.default_rng(2)
    pts = rng.uniform(-10.0, 10.0, size=(120, 3))
    pts[:, 2] = rng.uniform(0.5, 2.0, size=120)
    lengths = np.full(120, 0.5)
    radii = np.full(120, 0.005)
    eye = np.eye(120)
    hs = _k().HalfSpaceKernel(100.0)
    np.testing.assert_array_equal(
        hs.matrix(pts, lengths, radii),
        _k().HalfSpaceKernel(100.0)(pts, lengths, radii, eye),
    )
    soil = gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0)
    k2 = _two_layer_self_kernel_factory(soil, 300, 1e-6)
    np.testing.assert_array_equal(
        k2.matrix(pts, lengths, radii),
        _two_layer_self_kernel_factory(soil, 300, 1e-6)(pts, lengths, radii, eye),
    )


# ---------------------------------------------------------------------
# 10.62 — chunked FieldResult.potential
# ---------------------------------------------------------------------


@pytest.mark.finding("10.62")
def test_potential_on_a_large_grid_is_memory_bounded() -> None:
    """10 000 points × 1 920 sources: v0.15.0 peaked at 1757.9 MiB here
    (96 B per point-source pair)."""
    res = _solve(_houses_world(gf.HomogeneousSoil(resistivity=100.0)))
    g = np.linspace(-20.0, 100.0, 100)
    X, Y = np.meshgrid(g, g)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    res.potential(pts[:10])  # warm-up of the point-source cache
    peak = _peak_mb(lambda: res.potential(pts))
    assert peak < 80.0, f"peak {peak:.1f} MB"


@pytest.mark.finding("10.62")
@pytest.mark.parametrize(
    "soil",
    [
        gf.HomogeneousSoil(resistivity=100.0),
        gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0),
    ],
    ids=["homogeneous", "two_layer"],
)
def test_blocked_potential_agrees_with_one_piece(monkeypatch, soil) -> None:
    res = _solve(_houses_world(soil, n_side=2))
    rng = np.random.default_rng(9)
    pts = np.column_stack(
        [rng.uniform(-10.0, 45.0, 700), rng.uniform(-10.0, 45.0, 700), np.zeros(700)]
    )
    one_piece = res.potential(pts)
    monkeypatch.setattr(_k(), "FIELD_PIECE_BYTES", 0)
    monkeypatch.setattr(_k(), "BLOCK_BYTES", 8 * 64 * 480)
    blocked = res.potential(pts)
    np.testing.assert_allclose(blocked, one_piece, rtol=1e-13)


@pytest.mark.finding("10.62")
def test_series_field_shares_blocks_and_skips_a_zero_imaginary_part() -> None:
    """Sharing the 1/r blocks between Re and Im changes nothing, bit for bit."""
    rng = np.random.default_rng(4)
    src = rng.uniform(-10.0, 10.0, size=(200, 3))
    src[:, 2] = rng.uniform(0.5, 2.0, size=200)
    pts = np.column_stack(
        [rng.uniform(-30, 30, 50), rng.uniform(-30, 30, 50), np.zeros(50)]
    )
    cur = rng.normal(size=200) + 1j * rng.normal(size=200)
    series = _series(0.5, 4.0, 100, 1e-6)
    both = _k()._series_field(pts, src, [cur.real, cur.imag], series, 1e-3)
    re = _k()._series_field(pts, src, [cur.real], series, 1e-3)
    im = _k()._series_field(pts, src, [cur.imag], series, 1e-3)
    np.testing.assert_array_equal(both[0], re[0])
    np.testing.assert_array_equal(both[1], im[0])


# ---------------------------------------------------------------------
# 10.65 — point-source arrays cached on the result
# ---------------------------------------------------------------------


@pytest.mark.finding("10.65")
def test_point_source_arrays_are_cached_and_invalidated() -> None:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = _solve(w)
    a1 = res._point_source_arrays()
    a2 = res._point_source_arrays()
    assert a1[0] is a2[0] and a1[1] is a2[1]
    phi = res.potential(np.array([[3.0, 0.0, 0.0]]))
    res.point_sources = list(res.point_sources[:-1])  # new list -> rebuilt
    assert res._point_source_arrays()[0].shape[0] == len(res.point_sources)
    assert res.potential(np.array([[3.0, 0.0, 0.0]])) != phi
    dumped = res.model_dump()
    assert "_ps_cache" not in dumped


# ---------------------------------------------------------------------
# Benchmarks (skipped unless --run-benchmark)
# ---------------------------------------------------------------------


def _calibration_seconds() -> float:
    """Wall time of a fixed numpy workload (the machine's speed unit)."""
    rng = np.random.default_rng(0)
    a = rng.random((600, 600))
    t0 = time.perf_counter()
    for _ in range(20):
        np.sqrt(a @ a)
    return time.perf_counter() - t0


@pytest.mark.benchmark
@pytest.mark.finding("10.56", "10.57")
def test_benchmark_homogeneous_solve() -> None:
    """1920-segment homogeneous solve.

    Measured back to back on 2 shared cores: this branch 0.50–0.56
    calibration units, v0.15.0 3.6–4.3.
    """
    w = _houses_world(gf.HomogeneousSoil(resistivity=100.0))
    _solve(w)
    unit = _calibration_seconds()
    t0 = time.perf_counter()
    _solve(w)
    ratio = (time.perf_counter() - t0) / unit
    print(f"homogeneous solve: {ratio:.2f} calibration units")
    assert ratio < 1.0


@pytest.mark.benchmark
@pytest.mark.finding("10.58")
def test_benchmark_two_layer_solve() -> None:
    """480-segment two-layer solve with 144 images (K = -0.67).

    Measured back to back on 2 shared cores: this branch 1.15–1.44
    calibration units, v0.15.0 2.45–2.68.
    """
    w = _houses_world(gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0), n_side=2)
    _solve(w, image_max_terms=500)
    unit = _calibration_seconds()
    t0 = time.perf_counter()
    _solve(w, image_max_terms=500)
    ratio = (time.perf_counter() - t0) / unit
    print(f"two-layer solve: {ratio:.2f} calibration units")
    assert ratio < 2.0
