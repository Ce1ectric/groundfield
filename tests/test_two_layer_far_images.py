"""Far Tagg/Sunde images of the line kernel carry the quadrupole term (10.28).

Wave 3 of the 2026.11 work (stream w3xl, item 1). The images ``n >= 1`` of
the two-layer series were point sources beyond six segment lengths, so a
straight wire in two-layer soil was not reproduced exactly: a 10 m strip at
0.5 m in 1000 over 50 Ohm m with h_1 = 1 m deviated from its uniform-current
resistance by up to +5.1e-4, not monotonically in the segment length, and
the surface potential above it by up to 0.94 % (the alternating series of a
conductive lower layer amplifies the error of its far images). Every far
image now gets the far form of direct pairs — point, quadrupole and the
radius term of the reduced kernel — tapered off between 60 and 120 segment
lengths of vertical distance, where it falls below the remainder of the
switch; images beyond every observer are skipped. The cost was 2.2 ... 3.9
times the solve without the term on all images, 1.2 ... 1.5 times with the
taper (16 house grids, evidence ``item1_gain.py``).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

RHO1, RHO2, H1 = 1000.0, 50.0, 1.0
L, Z, A = 10.0, 0.5, 0.005
K = (RHO2 - RHO1) / (RHO2 + RHO1)


def _images(n_max: int):
    """(weight, vertical image depth) of the series, surface pair first."""
    out = [(1.0, Z), (1.0, -Z)]
    for n in range(1, n_max + 1):
        q = 2.0 * n * H1
        out += [(K**n, q + Z), (K**n, q - Z), (K**n, -q + Z), (K**n, -q - Z)]
    return out


N_MAX = next(n for n in range(1, 10_000) if abs(K) ** n < 1e-18)


def _double_integral(d: float) -> float:
    """∫∫ dx dy / sqrt((x - y)² + d²) over [0, L]²."""
    return 2.0 * (L * math.asinh(L / d) - math.sqrt(L * L + d * d) + d)


def _resistance_reference() -> float:
    """Uniform-current resistance of the strip, the image sum in closed form.

    Direct term with the reduced kernel (d = a), the images at their
    vertical distance to the strip (``_images``): 90.614143 Ohm; a 2-D
    Gauss rule of the same sum agrees to 1.2e-9 (evidence
    ``item1_strip_reference.py``).
    """
    terms = [(1.0, A)] + [(w, abs(Z - zk)) for w, zk in _images(N_MAX)[1:]]
    total = math.fsum(w * _double_integral(d) for w, d in terms)
    return RHO1 / (4.0 * math.pi * L * L) * total


def _strip_world() -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=RHO1, rho_2=RHO2, h_1=H1))
    gf.create_electrode(
        w, "strip", name="s", start=(0, 0, Z), end=(L, 0, Z), wire_radius=A
    )
    gf.create_source(w, attached_to="s", magnitude=1.0)
    return w


def _solve(world, ds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gf.create_engine(segment_length=ds, frequencies=[50.0]).solve(world)


@pytest.mark.finding("10.28")
def test_a_straight_strip_in_two_layer_soil_is_exact_at_any_segment_length() -> None:
    """Within 4.5e-6 of the closed form on a fine grid of ds = 2 ... 0.125 m.

    Measured (stream fw3xl, after audit W3A-02): -1.8e-6, -1.7e-6, -2.4e-6,
    -1.4e-6, -2.9e-6, -3.6e-6, -3.3e-6, -3.1e-6, -3.1e-6, -3.3e-6 at ds = 2,
    1.5, 1, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.125 m, not monotone (audit
    W3A-03: the former check of monotonicity on five points did not hold on
    this grid). The closed form takes the images axis to axis; about 2e-6 of
    the rest is the reduced kernel the soil-surface image keeps, the rest
    the quadrupole remainder of the far form. Before wave 3: +8.2e-5,
    -2.8e-6, +5.1e-4, +1.4e-4, -3.2e-5 at 2, 1, 0.5, 0.25, 0.125 m; between
    wave 3 and W3A-02 within 1.2e-6, the reduced kernel on the interface
    images (+2.5e-6 here) offsetting the remainder.
    """
    ref = _resistance_reference()
    world = _strip_world()
    dev = [
        _solve(world, ds).cluster_impedance("s")[0].real / ref - 1.0
        for ds in (2.0, 1.5, 1.0, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.125)
    ]
    assert max(abs(d) for d in dev) < 4.5e-6, dev


def _line_potential(point, current: float) -> float:
    """Potential of the uniform line current and its images at a point."""
    x, y, z = point
    total = 0.0
    for w, zk in _images(N_MAX):
        rho = math.hypot(y, z - zk)
        total += w * (math.asinh((L - x) / rho) + math.asinh(x / rho))
    return RHO1 * current / (4.0 * math.pi * L) * total


@pytest.mark.finding("10.28")
def test_the_surface_potential_above_a_strip_matches_the_image_sum() -> None:
    """The field evaluation carries the far images' quadrupole term as well.

    Against the closed-form line potential of the image series: 8e-6 at
    0.5 m segments (0.94 % before; 2.7e-6 / 3.7e-6 at 1 / 0.25 m).
    """
    pts = np.array(
        [
            [5.0, 1.0, 0.0],
            [5.0, 3.0, 0.0],
            [-2.0, 0.5, 0.0],
            [12.0, -1.0, 0.0],
            [5.0, 0.3, 0.0],
        ]
    )
    res = _solve(_strip_world(), 0.5)
    current = float(np.real(res.electrode_currents["s"][0]))
    ref = np.array([_line_potential(p, current) for p in pts])
    assert np.max(np.abs(res.potential(pts).real / ref - 1.0)) < 2e-5


@pytest.mark.finding("10.28")
def test_the_compiled_image_kernel_is_bit_identical(monkeypatch) -> None:
    """numba (extra ``fast``) and numpy give the same far-image blocks."""
    pytest.importorskip("numba")
    from groundfield.solver import _kernels
    from groundfield.solver import _line_kernel as lk

    rng = np.random.default_rng(3)
    n = 300
    mid = rng.uniform(-20.0, 20.0, size=(n, 3))
    mid[:, 2] = rng.uniform(0.3, 3.0, size=n)
    tangent = rng.normal(size=(n, 3))
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    segs = lk.LineSegments(
        mid, rng.uniform(0.2, 1.0, n), tangent, rng.uniform(0.002, 0.01, n)
    )
    images = [
        (0.7**k, s * 2.0 * k * 3.5, sg)
        for k in range(1, 30)
        for s in (1, -1)
        for sg in (1, -1)
    ]
    pts = np.column_stack(
        [rng.uniform(-25, 25, 200), rng.uniform(-25, 25, 200), np.zeros(200)]
    )
    monkeypatch.setattr(lk, "LINE_NUMBA_MIN_PAIRS", 0)

    def blocks():
        return (
            lk.image_quadrupole_rows(
                segs.mid,
                segs.tangent,
                segs.length,
                segs,
                images,
                obs_radius=segs.radius,
            ),
            lk.image_quadrupole_rows(pts, None, np.zeros(200), segs, images),
        )

    compiled = blocks()
    monkeypatch.setattr(_kernels, "USE_NUMBA", False)
    plain = blocks()
    np.testing.assert_array_equal(compiled[0], plain[0])
    np.testing.assert_array_equal(compiled[1], plain[1])
    assert np.any(compiled[0]) and np.any(compiled[1])


@pytest.mark.finding("10.28")
def test_the_far_image_term_belongs_to_the_pair() -> None:
    """An electrode far away does not change the other electrode's entries.

    The taper of the far-image term depends on the pair's own vertical
    distance and segment length only (a taper by the depth range of the whole
    world changed the field of a rod by 1.2e-8 when another rod was added).
    """
    from groundfield.solver._kernels import TwoLayerSeriesKernel
    from groundfield.solver._pipeline import SegmentArrays, discretise
    from groundfield.solver._series import ImageSeries

    soil = gf.TwoLayerSoil(rho_1=RHO1, rho_2=RHO2, h_1=H1)
    series = ImageSeries.for_soil(soil, 2000, 1e-14)

    def series_matrix(extra: bool):
        w = _strip_world()
        if extra:  # a 0.2 m deep strip 500 m away, with 2 m segments
            gf.create_electrode(
                w, "strip", name="far", start=(500, 0, 0.2), end=(520, 0, 0.2)
            )
        eng = gf.Engine(segment_length=0.5)
        disc = discretise(w, eng)
        a = SegmentArrays.from_segments(disc.segments)
        line = a.line_segments()
        kernel = TwoLayerSeriesKernel(soil.rho_1, series, line=line)
        return kernel.series_matrix(a.points)

    alone = series_matrix(False)
    n = alone.shape[0]
    with_far = series_matrix(True)[:n, :n]
    np.testing.assert_allclose(with_far, alone, rtol=1e-13, atol=0.0)


@pytest.mark.finding("10.28")
def test_far_image_terms_below_the_tolerance_are_left_out(monkeypatch) -> None:
    """Below a thin top layer the images are dense in depth and light.

    A house grid at 0.8 m below h_1 = 0.5 m (cross-layer path): of the 63
    deep images of the layered correction 58 lie within the reach of the
    taper and 26 of those carry a quadrupole term that can exceed 1e-9 of a
    unit point term; leaving out the rest moves Z by less than 1e-10 (1000
    times below the truncation of the series itself).
    """
    from groundfield.solver import _line_kernel as lk
    from groundfield.solver import image_2layer
    from groundfield.solver._pipeline import SegmentArrays, discretise

    soil = gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=0.5)

    def solve_and_count():
        w = gf.create_world(soil=soil)
        gf.create_electrode(
            w,
            "grid_mesh",
            name="g",
            corner=(0, 0, 0.8),
            size=(10.0, 10.0),
            n_x=2,
            n_y=2,
        )
        gf.create_source(w, attached_to="g", magnitude=1.0)
        a = SegmentArrays.from_segments(
            discretise(
                w, gf.Engine(segment_length=0.5), layer_interfaces=(0.5,)
            ).segments
        )
        fam = image_2layer._LayeredImageFamilies(
            a.line_segments(), a.points[:, 2] >= 0.5, soil
        )
        z = _solve(w, 0.5).cluster_impedance("g")[0].real
        return z, max(len(f) for f in fam.far), max(len(g[0]) for g in fam.groups)

    z_tol, n_tol, n_all = solve_and_count()
    monkeypatch.setattr(lk, "IMAGE_QUADRUPOLE_TOL", 0.0)
    z_all, n_none, _ = solve_and_count()
    assert (n_tol, n_none, n_all) == (26, 58, 63)
    assert abs(z_tol / z_all - 1.0) < 1e-10


# ---------------------------------------------------------------------
# audit W3A-01: the image work follows the pair, not the longest segment
# ---------------------------------------------------------------------


def _houses(soil, *, long_piece: bool = False) -> gf.World:
    """16 house grids (10 x 10 m, 2 x 2 meshes) at 0.8 m, 25 m apart."""
    w = gf.create_world(soil=soil)
    names = []
    for i in range(4):
        for j in range(4):
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
        )
    if long_piece:
        # one bare galvanic copper conductor between two houses, 10 m pieces
        gf.create_conductor(
            w,
            name="bare",
            start="h30",
            end="h33",
            conductor_type="bare_copper",
            cross_section=50e-6,
            coupling_to_soil="galvanic",
            discretize_segment_length=10.0,
        )
    gf.create_source(w, attached_to="h00", magnitude=1.0)
    return w


def _image_work(world, monkeypatch) -> dict[str, int]:
    """Far-image evaluations and Galerkin image pairs of one solve.

    The spy of auditor W3A: every call of ``image_quadrupole_rows`` counts
    ``rows × columns × images``, every Galerkin call its pairs, split by
    whether both segments of the pair are short (at most 1 m). Keyword
    arguments pass through (the kernel accumulates into ``out``).
    """
    from groundfield.solver import _kernels as K
    from groundfield.solver import _line_kernel as lk
    from groundfield.solver import image_2layer as i2

    work = {"far": 0, "galerkin_short": 0, "galerkin_long": 0}
    far, gal = lk.image_quadrupole_rows, lk.galerkin_pairs
    depth = [0]

    def far_spy(obs_mid, obs_t, obs_len, segs, images, *args, **kwargs):
        work["far"] += obs_mid.shape[0] * segs.mid.shape[0] * len(images)
        return far(obs_mid, obs_t, obs_len, segs, images, *args, **kwargs)

    def gal_spy(obs, i, src, j, *args, **kwargs):
        # count the outer call only (the kernel calls itself for chunks)
        if depth[0] == 0:
            short = (obs.length[i] <= 1.0) & (src.length[j] <= 1.0)
            work["galerkin_short"] += int(short.sum())
            work["galerkin_long"] += int((~short).sum())
        depth[0] += 1
        try:
            return gal(obs, i, src, j, *args, **kwargs)
        finally:
            depth[0] -= 1

    with monkeypatch.context() as mp:
        for mod in (lk, K, i2):
            if hasattr(mod, "image_quadrupole_rows"):
                mp.setattr(mod, "image_quadrupole_rows", far_spy)
            if hasattr(mod, "galerkin_pairs"):
                mp.setattr(mod, "galerkin_pairs", gal_spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gf.Engine(segment_length=0.5, frequencies=[50.0]).solve(world)
    return work


@pytest.mark.finding("10.28")
@pytest.mark.parametrize("h_1", [5.0, 0.5], ids=["series", "cross_layer"])
def test_w3a_01_a_few_long_pieces_do_not_multiply_the_image_work(
    monkeypatch, h_1
) -> None:
    """Eight 9.4 m pieces among 1920 segments of 0.5 m (auditor W3A).

    The reach of the far-image term (120 segment lengths) and the image
    count followed the longest segment of the whole world: the far-image
    work grew from 77.4 M to 431 M evaluations (h_1 = 5 m) and from 103 M to
    152 M (h_1 = 0.5 m), and every far image of the long pieces added a
    Galerkin self image of every short segment. Now the pairs of the short
    segments keep their own reach: the far work grows by less than 50 %, and
    the long pieces add no Galerkin pair between two short segments.

    The auditor's second gate, "fewer than 1.5 times the Galerkin image
    pairs", counts the near image pairs of the long pieces themselves: a pair
    within 7.5 lengths of the longer segment is integrated exactly, which is
    the near-pair criterion of every pair (the base 34ad269 grew from 59 k to
    324 k pairs on the series path for the same reason).
    """
    soil = gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=h_1)
    plain = _image_work(_houses(soil), monkeypatch)
    extra = _image_work(_houses(soil, long_piece=True), monkeypatch)
    assert extra["far"] < 1.5 * plain["far"], (plain, extra)
    assert extra["galerkin_short"] == plain["galerkin_short"], (plain, extra)
    assert plain["galerkin_long"] == 0
