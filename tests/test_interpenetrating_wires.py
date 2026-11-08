"""Wires lying beside each other closer than the sum of their radii.

Two segments of different conductors whose axes come closer than the sum
of their radii — coincident leads, a service drop routed back along its
PEN main, two cables in one trench at a too small offset, wires leaving a
junction or crossing at a small angle — have no meaningful filament mutual
inductance. Coincident filaments diverge: the 16x16 Gauss–Legendre rule
only stopped at its 1e-6 m floor (two coincident 50 m leads +587j Ohm at
50 Hz, stream GEN-2; a TN network whose drops ran back along the PEN main
had a negative input reactance at 1 kHz, -0.633 Ohm, stream DOC-B). Audit
round 2 of the calendar release evaluated near-parallel pairs whose partner
lay inside a cylinder of radius r_a + r_b around the row segment as
touching wires; audit round 3 found that this whole-pair switch depended on
which segment was the row (a V pair of 1 m and 0.5 m at 1.5 deg: 0.362 or
0.503 uH; the generator's in-line reading 0.134 or 0.127 Ohm), and that
tilted V and X pairs just outside the cylinder kept a filament value above
the self inductance (an indefinite matrix, a two-wire loop with a negative
reactance), A3C-1 / A3C-2. A wire inside another (a core on the axis of a
cable shield) was treated as two wires touching on the outside, A3C-5.

The rule now acts pointwise inside the Neumann integral: along a stretch
where a point of one wire lies beside the other wire (its foot on the other
axis falls inside the other wire) closer than r_a + r_b, the distance is
taken as r_a + r_b (touching wires), or as the radius of the thicker wire
where the thinner one runs inside it (a filament inside a thin tube); both
one-sided integrals are averaged, on end points in a fixed order. The
Carson tiling gives two conductors overlapping on one axis the earth-return
mutual of touching wires.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.conductors import Conductor
from groundfield.coupling.carson import carson_mutual_correction
from groundfield.coupling.inductance import (
    _build_inductance_matrix_loop,
    build_carson_correction_matrices,
    build_inductance_matrix,
    parallel_segments_mutual,
)

pytestmark = pytest.mark.finding("10.46")


def _chain(a, b, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pts = a + np.linspace(0.0, 1.0, n + 1)[:, None] * (b - a)
    return np.stack([pts[:-1], pts[1:]], axis=1)


# stream GEN-2 request / DOC-B
def test_coincident_leads_have_the_mutual_of_touching_wires() -> None:
    """Two coincident 50 m leads (5 m segments, r = 5 mm): the mutual
    inductance of two 50 m wires touching side by side (base: 1.9 H, the
    quadrature floor)."""
    lead = _chain((0.0, 0.0, -0.1), (50.0, 0.0, -0.1), 10)
    seg = np.concatenate([lead, lead])
    L = build_inductance_matrix(seg, np.full(20, 0.005), image="none")
    mutual = L[:10, 10:].sum()
    # Segments of the two leads that meet end to end keep the exact coaxial
    # closed form (as chain neighbours do), O(1e-9 H) each from the 1 cm
    # offset of touching wires: 2e-4 of the total.
    assert mutual == pytest.approx(
        parallel_segments_mutual(50.0, 0.01), rel=1e-3, abs=0
    )


# stream DOC-B
@pytest.mark.parametrize("image", ["none", "additive", "antiparallel"])
def test_drops_along_the_pen_main_keep_the_matrix_positive_definite(image) -> None:
    """A PEN main and two service drops routed back along it on the same
    axis and depth (the DOC-B network): positive definite in every image
    convention."""
    seg = np.concatenate(
        [
            _chain((0.0, 0.0, 0.6), (-200.0, 0.0, 0.6), 20),
            _chain((-200.0, 0.0, 0.6), (-20.0, 0.0, 0.6), 18),
            _chain((-200.0, 0.0, 0.6), (20.0, 0.0, 0.6), 22),
        ]
    )
    L = build_inductance_matrix(seg, np.full(len(seg), 0.0069), image=image)
    assert np.linalg.eigvalsh(L).min() > 0.0


def _overlap_world() -> gf.World:
    """Substation rod, a PEN main to a cabinet 200 m away and two service
    drops from the cabinet back along the main (10 m chains on one axis
    and depth, as the TN generator built them for two houses)."""
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    for name, x, y in (("s", 0.0, 0.0), ("kvs", -200.0, 0.0), ("h0", -20.0, 30.0)):
        gf.create_electrode(w, "rod", name=name, position=(x, y, 0.0), length=2.0)
    gf.create_electrode(w, "rod", name="h1", position=(20.0, 30.0, 0.0), length=2.0)
    area = 150e-6
    for name, a, b, xa, xb in (
        ("pen_main", "s", "kvs", 0.0, -200.0),
        ("drop_0", "kvs", "h0", -200.0, -20.0),
        ("drop_1", "kvs", "h1", -200.0, 20.0),
    ):
        w.add_conductor(
            Conductor(
                name=name,
                start=(xa, 0.0, 0.6),
                end=(xb, 0.0, 0.6),
                start_electrode=a,
                end_electrode=b,
                wire_radius=math.sqrt(area / math.pi),
                resistivity=2.82e-8,
                cross_section=area,
                discretize_segment_length=10.0,
                coupling_to_soil="isolated",
                inductance_model="neumann",
            )
        )
    gf.create_source(w, attached_to="s", magnitude=1.0)
    return w


# stream DOC-B
@pytest.mark.parametrize("model", ["perfect_mirror", "sommerfeld", "carson_series"])
def test_overlapping_drops_give_a_passive_input_impedance(model: str) -> None:
    """Base: Im Z_in(1 kHz) < 0 with perfect_mirror and sommerfeld."""
    eng = gf.create_engine(
        backend="image",
        segment_length=1.0,
        frequencies=[50.0, 1000.0],
        earth_inductive_model=model,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = np.asarray(eng.solve(_overlap_world()).input_impedance())
    assert (z.imag > 0.0).all()


# stream DOC-B
def test_carson_tiling_couples_overlapping_coaxial_conductors() -> None:
    """Two conductors overlapping on one axis over 6 m get the Carson
    mutual of touching wires per metre of overlap (base: zero)."""
    seg = np.array(
        [
            [[0.0, 0.0, 0.7], [10.0, 0.0, 0.7]],
            [[4.0, 0.0, 0.7], [20.0, 0.0, 0.7]],
        ]
    )
    om = 2.0 * math.pi * 1000.0
    dZ = build_carson_correction_matrices(
        seg, np.full(2, 0.005), omegas=[om], sigma_earth=0.01
    )[0]
    expected = 6.0 * carson_mutual_correction(om, 0.7, 0.7, 0.01, 0.01)
    assert dZ[0, 1] == pytest.approx(expected, rel=1e-12)


def test_wires_at_the_touching_distance_are_unchanged() -> None:
    """At an axis distance of exactly r_a + r_b the closed form of the
    parallel pair is kept bit for bit (the guard is continuous)."""
    a = _chain((0.0, 0.0, 0.7), (20.0, 0.0, 0.7), 4)
    b = _chain((0.0, 0.01, 0.7), (20.0, 0.01, 0.7), 4)
    seg = np.concatenate([a, b])
    with_guard = build_inductance_matrix(seg, np.full(8, 0.005), image="none")
    no_guard = build_inductance_matrix(seg, np.full(8, 1e-9), image="none")
    assert np.array_equal(with_guard[:4, 4:], no_guard[:4, 4:])


def test_loop_reference_applies_the_same_guard() -> None:
    lead = _chain((0.0, 0.0, 0.5), (12.0, 0.0, 0.5), 3)
    other = _chain((3.0, 0.002, 0.5), (9.0, 0.002, 0.5), 2)  # 2 mm off axis
    seg = np.concatenate([lead, other, _chain((0, 3, 0.5), (12, 3, 0.5), 2)])
    radii = np.full(len(seg), 0.004)
    for image in ("none", "additive", "antiparallel"):
        L_vec = build_inductance_matrix(seg, radii, image=image)
        L_loop = _build_inductance_matrix_loop(seg, radii, image=image)
        np.testing.assert_allclose(L_vec, L_loop, rtol=1e-12, atol=1e-20)
    # audit A3C-1: the pair 2 mm apart (overlap 2 m of 4 m and 3 m) gets the
    # beside rule; moving the whole of the second wire to the touching
    # distance (8 mm, the whole-pair rule of audit round 2) differs from it
    # by +2.3e-4 (the stretches beyond the overlap keep their 2 mm).
    from groundfield.coupling.inductance import _parallel_filaments_mutual

    L = build_inductance_matrix(seg, radii, image="none")
    assert L[1, 3] == pytest.approx(
        _beside_reference(seg[1, 0], seg[1, 1], 0.004, seg[3, 0], seg[3, 1], 0.004),
        rel=1e-12,
        abs=0,
    )
    moved = seg[3].copy()
    moved[:, 1] = 0.008
    displaced = _parallel_filaments_mutual(seg[1, 0], seg[1, 1], moved[0], moved[1])
    assert L[1, 3] / displaced - 1.0 == pytest.approx(2.29e-4, abs=0.02e-4)


# ---------------------------------------------------------------------
# audit round 3: the beside rule (A3C-1, A3C-2, A3C-5)
# ---------------------------------------------------------------------


def _one_sided_reference(a1, a2, b1, b2, d, r_thin, r_thick) -> float:
    """Independent quadrature of one side of the beside rule (m).

    Break points of the integrand's state (beside closer than touching,
    inside the thicker wire, foot before / after the partner) by a dense
    scan and bisection; scipy's adaptive quadrature in between.
    """
    from scipy.integrate import quad

    a1, a2, b1, b2 = (np.asarray(p, dtype=float) for p in (a1, a2, b1, b2))
    la, lb = np.linalg.norm(a2 - a1), np.linalg.norm(b2 - b1)
    ua, ub = (a2 - a1) / la, (b2 - b1) / lb

    def state(s: float):
        v = a1 + s * ua - b1
        t0 = float(v @ ub)
        rho = float(np.linalg.norm(v - t0 * ub))
        st = 0
        if 0.0 <= t0 <= lb and rho < d:
            st = 2 if (r_thin < r_thick and rho + r_thin <= r_thick) else 1
        return st, t0, rho

    def key(s: float):
        st, t0, _ = state(s)
        return st, t0 < 0.0, t0 > lb

    grid = np.linspace(0.0, la, 2001)
    keys = [key(x) for x in grid]
    breaks = {0.0, float(la)}
    for k in range(len(grid) - 1):
        if keys[k] != keys[k + 1]:
            lo, hi = grid[k], grid[k + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if key(mid) == keys[k] else (lo, mid)
            breaks.add(0.5 * (lo + hi))

    def g(s: float) -> float:
        st, t0, rho = state(s)
        rho_e = max(rho if st == 0 else (d if st == 1 else r_thick), 1e-300)
        return math.asinh((lb - t0) / rho_e) + math.asinh(t0 / rho_e)

    pts = sorted(breaks)
    return sum(
        quad(g, x0, x1, limit=500, epsabs=0.0, epsrel=1e-13)[0]
        for x0, x1 in zip(pts[:-1], pts[1:], strict=True)
        if x1 > x0
    )


def _beside_reference(p1a, p2a, ra, p1b, p2b, rb) -> float:
    """mu0/(4 pi) cos(phi) (I_ab + I_ba) / 2 with the independent quadrature."""
    d, r_thin, r_thick = ra + rb, min(ra, rb), max(ra, rb)
    i_ab = _one_sided_reference(p1a, p2a, p1b, p2b, d, r_thin, r_thick)
    i_ba = _one_sided_reference(p1b, p2b, p1a, p2a, d, r_thin, r_thick)
    ua = (np.asarray(p2a, float) - p1a) / np.linalg.norm(np.asarray(p2a, float) - p1a)
    ub = (np.asarray(p2b, float) - p1b) / np.linalg.norm(np.asarray(p2b, float) - p1b)
    return 1e-7 * float(ua @ ub) * 0.5 * (i_ab + i_ba)


R = 0.01  # wire radius in m of the V and X pairs


def _v_pair(l_a: float, l_b: float, eps: float) -> np.ndarray:
    a = [[0.0, 0.0, 0.7], [l_a, 0.0, 0.7]]
    b = [[0.0, 0.0, 0.7], [l_b * math.cos(eps), l_b * math.sin(eps), 0.7]]
    return np.array([a, b], dtype=float)


def _x_pair(length: float, eps: float) -> np.ndarray:
    h = 0.5 * length
    a = [[-h, 0.0, 0.7], [h, 0.0, 0.7]]
    b = [
        [-h * math.cos(eps), -h * math.sin(eps), 0.7],
        [h * math.cos(eps), h * math.sin(eps), 0.7],
    ]
    return np.array([a, b], dtype=float)


#: Two-segment configurations the rule acts on: (segments, radii).
BESIDE_CASES = {
    # V pair whose short wire ends inside, the long one outside the other's
    # cylinder: the whole-pair switch gave 0.362 or 0.503 uH (A3C-1).
    "v_1m_0.5m_1.5deg": (_v_pair(1.0, 0.5, math.radians(1.5)), (R, R)),
    # far ends 1.001 (r_a + r_b) apart: M > L with the filament (A3C-2)
    "v_far_end_1.001": (_v_pair(1.0, 1.0, math.asin(1.001 * 2 * R)), (R, R)),
    "x_far_end_1.03": (_x_pair(1.0, math.asin(1.03 * 2 * R / 0.5)), (R, R)),
    "v_30deg": (_v_pair(1.0, 1.0, math.radians(30.0)), (R, R)),
    "t_junction_10deg": (
        np.array(
            [
                [[0.0, 0.0, 0.7], [10.0, 0.0, 0.7]],
                [
                    [4.0, 0.0, 0.7],
                    [4.0 + 5.0 * math.cos(0.17), 5.0 * math.sin(0.17), 0.7],
                ],
            ]
        ),
        (0.005, 0.008),
    ),
    "partial_overlap_2mm": (
        np.array(
            [[[4.0, 0.0, 0.5], [8.0, 0.0, 0.5]], [[3.0, 0.002, 0.5], [6.0, 0.002, 0.5]]]
        ),
        (0.004, 0.004),
    ),
    "coaxial_overlap": (
        np.array(
            [[[0.0, 0.0, 0.7], [10.0, 0.0, 0.7]], [[4.0, 0.0, 0.7], [20.0, 0.0, 0.7]]]
        ),
        (0.005, 0.005),
    ),
    # the generator's lumped in-line leads: the probe lead starts the
    # touching distance beside the feed lead and ends on its axis
    "converging_lead": (
        np.array(
            [
                [[0.0, 0.0, -0.1], [200.0, 0.0, -0.1]],
                [[0.0, 0.01, -0.1], [123.6, 0.0, -0.1]],
            ]
        ),
        (0.005, 0.005),
    ),
    "core_in_shield": (
        np.array(
            [[[0.0, 0.0, 0.7], [100.0, 0.0, 0.7]], [[0.0, 0.0, 0.7], [100.0, 0.0, 0.7]]]
        ),
        (0.005, 0.015),
    ),
    "core_off_axis": (
        np.array(
            [
                [[0.0, 0.0, 0.7], [100.0, 0.0, 0.7]],
                [[0.0, 0.008, 0.7], [100.0, 0.012, 0.7]],
            ]
        ),
        (0.005, 0.015),
    ),
}


# audit A3C-1
@pytest.mark.parametrize("name", sorted(BESIDE_CASES))
def test_beside_rule_matches_an_independent_quadrature(name: str) -> None:
    from groundfield.coupling.inductance import _beside_mutual

    seg, (ra, rb) = BESIDE_CASES[name]
    got = _beside_mutual(seg[0, 0], seg[0, 1], ra, seg[1, 0], seg[1, 1], rb)
    ref = _beside_reference(seg[0, 0], seg[0, 1], ra, seg[1, 0], seg[1, 1], rb)
    assert got == pytest.approx(ref, rel=1e-11, abs=0)


# audit A3C-1
@pytest.mark.parametrize("name", sorted(BESIDE_CASES))
@pytest.mark.parametrize("image", ["none", "additive"])
def test_beside_rule_does_not_depend_on_the_order_or_direction(name, image) -> None:
    """Every order of the two segments and direction of each gives the
    same mutual inductance: bit for bit (up to the orientation sign) in
    free space, to round-off with the additive image (its image term is an
    ordinary pair evaluated from the row segment)."""
    seg, radii = BESIDE_CASES[name]
    radii = np.asarray(radii, dtype=float)
    ref = build_inductance_matrix(seg, radii, image=image)[0, 1]
    for swap in (False, True):
        for flip_a in (False, True):
            for flip_b in (False, True):
                s = seg.copy()
                if flip_a:
                    s[0] = s[0, ::-1]
                if flip_b:
                    s[1] = s[1, ::-1]
                r = radii
                if swap:
                    s, r = s[::-1].copy(), radii[::-1].copy()
                m = build_inductance_matrix(s, r, image=image)[0, 1]
                expected = (-1.0) ** (flip_a + flip_b) * ref
                if image == "none":
                    assert m == expected
                else:
                    assert m == pytest.approx(expected, rel=1e-14, abs=0)


# audit A3C-1 (the auditor's test)
def test_touching_wire_clamp_does_not_depend_on_the_segment_order() -> None:
    """V pair, 1 m and 0.5 m, 1.5 deg: the short wire's far end is 13.1 mm
    from the long wire's axis (inside 20 mm), the long wire's far end
    26.2 mm from the short wire's axis (outside). Audit round 3 measured
    0.36237 uH in one order, 0.50274 uH in the other (+38.7 %)."""
    seg = _v_pair(1.0, 0.5, math.radians(1.5))
    radii = np.full(2, R)
    m_ab = build_inductance_matrix(seg, radii, image="none")[0, 1]
    m_ba = build_inductance_matrix(seg[::-1].copy(), radii, image="none")[0, 1]
    assert m_ab == pytest.approx(m_ba, rel=1e-12, abs=0)


# audit A3C-1 (the auditor's test)
@pytest.mark.parametrize("image", ["none", "additive", "antiparallel"])
def test_generator_lead_pair_mutual_does_not_depend_on_the_order(image) -> None:
    """The lumped in-line leads as TnNetworkGenerator builds them (d_aux =
    200 m, probe at 61.8 %, overhead_lead(segment_length_m=None)). Audit
    round 3 measured (image "none") 263.35 uH with the feed lead first,
    238.63 uH with the probe lead first."""
    seg = BESIDE_CASES["converging_lead"][0]
    radii = np.array([0.005, 0.005])
    m_feed_first = build_inductance_matrix(seg, radii, image=image)[0, 1]
    m_probe_first = build_inductance_matrix(seg[::-1].copy(), radii, image=image)[0, 1]
    assert m_feed_first == pytest.approx(m_probe_first, rel=1e-12, abs=0)


def _inline_world() -> gf.World:
    from groundfield.generators import (
        HomogeneousSoilSpec,
        MeasurementInjectionConfig,
        MeasurementProbeConfig,
        MeasurementSetupConfig,
        TnNetworkConfig,
        TnNetworkGenerator,
        overhead_lead,
    )

    cfg = TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=100.0),
        building_counts={"residential": 3},
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(200.0, 0.0),
                feed_lead=overhead_lead(segment_length_m=None),
            ),
            probe=MeasurementProbeConfig(
                position_xy=(0.618 * 200.0, 0.0),
                lead=overhead_lead(segment_length_m=None),
            ),
        ),
    )
    return TnNetworkGenerator(cfg, seed=1).build()


# audit A3C-1 (the auditor's test, solver level)
def test_inline_reading_does_not_depend_on_the_order_of_the_leads() -> None:
    """Swapping meas_feed_lead and meas_probe_lead in world.conductors does
    not change the reading. Audit round 3 measured at 50 Hz,
    perfect_mirror: 1.582769+0.134442j Ohm as built, 1.582769+0.126676j Ohm
    swapped (= the value with 5 m lead segments); now 0.126676 Ohm in both
    orders."""
    import copy

    from groundfield.generators.measurement import measured_impedance

    world = _inline_world()
    swapped = copy.deepcopy(world)
    names = [c.name for c in swapped.conductors]
    i, j = names.index("meas_feed_lead"), names.index("meas_probe_lead")
    swapped.conductors[i], swapped.conductors[j] = (
        swapped.conductors[j],
        swapped.conductors[i],
    )
    eng = gf.create_engine(
        backend="image",
        segment_length=2.0,
        kernel="point",
        frequencies=[50.0],
        earth_inductive_model="perfect_mirror",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_built = measured_impedance(eng.solve(world))[0]
        z_swapped = measured_impedance(eng.solve(swapped))[0]
    assert z_built.imag == pytest.approx(z_swapped.imag, rel=1e-9, abs=0)
    assert z_built.imag == pytest.approx(0.126676, abs=5e-7)


def _network_segments() -> tuple[np.ndarray, np.ndarray, list[slice]]:
    """Four conductors (chains) that touch, cross and lie beside each other."""
    feed = _chain((0.0, 0.0, 0.7), (60.0, 0.0, 0.7), 12)
    probe = _chain((0.0, 0.01, 0.7), (37.0, 0.0, 0.7), 8)  # converges
    branch = _chain((20.0, 0.0, 0.7), (40.0, 2.0, 0.7), 4)  # T at 5.7 deg
    cross = _chain((30.0, -3.0, 0.7), (34.0, 3.0, 0.7), 3)  # crossing
    parts = [feed, probe, branch, cross]
    radii = [0.005, 0.004, 0.006, 0.005]
    seg = np.concatenate(parts)
    r = np.concatenate([np.full(len(p), q) for p, q in zip(parts, radii, strict=True)])
    bounds = np.cumsum([0] + [len(p) for p in parts])
    return seg, r, [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


# audit A3C-1
@pytest.mark.parametrize("image", ["none", "additive", "antiparallel"])
def test_matrix_does_not_depend_on_the_order_of_the_conductors(image) -> None:
    """All 24 orders of four conductors, each with reversed segments in
    turn: the re-indexed matrix agrees to round-off (the beside entries
    bit for bit in the direct block)."""
    import itertools

    seg, radii, parts = _network_segments()
    ref = build_inductance_matrix(seg, radii, image=image)
    scale = np.abs(ref).max()
    for n, order in enumerate(itertools.permutations(range(4))):
        flip = order[n % 4]  # reverse the chain of one conductor
        index, pieces = [], []
        for k in order:
            idx = np.arange(parts[k].start, parts[k].stop)
            p = seg[idx]
            if k == flip:
                idx, p = idx[::-1], p[::-1, ::-1]
            index.append(idx)
            pieces.append(p)
        index = np.concatenate(index)
        L = build_inductance_matrix(np.concatenate(pieces), radii[index], image=image)
        sign = np.where(
            np.isin(index, np.arange(parts[flip].start, parts[flip].stop)), -1.0, 1.0
        )
        back = sign[:, None] * sign[None, :] * L
        assert np.abs(back - ref[np.ix_(index, index)]).max() <= 1e-13 * scale


# ---------------------------------------------------------------- A3C-2


# audit A3C-2 (the auditor's test)
@pytest.mark.parametrize("kind", ["V", "X"])
@pytest.mark.parametrize("ratio", [1.001, 1.03])
@pytest.mark.parametrize("internal", [True, False])
def test_partial_inductance_matrix_stays_positive_definite(
    kind, ratio, internal
) -> None:
    """Far-end separation D = ratio (r_a + r_b), just outside the cylinder
    of the whole-pair rule: audit round 3 measured a minimum eigenvalue of
    -1.3e-8 H (V, 1.001) and -7.3e-9 H (V, 1.03), likewise X."""
    D = ratio * 2 * R
    seg = (
        _v_pair(1.0, 1.0, math.asin(D))
        if kind == "V"
        else _x_pair(1.0, math.asin(D / 0.5))
    )
    L = build_inductance_matrix(
        seg, np.full(2, R), image="none", include_internal=internal
    )
    assert np.linalg.eigvalsh(L).min() > 0.0


def _random_close_set(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Three to six wires, each leaving a vertex of, crossing, branching off
    or lying beside an earlier one; the angle is drawn so that the far ends
    come 0.3 ... 3 touching distances apart (the band in which the
    whole-pair rule of audit round 2 kept M > L)."""
    radii = [float(rng.uniform(0.002, 0.012))]
    p = rng.uniform(-1.0, 1.0, 3)
    p[2] = 0.7
    u = rng.normal(size=3)
    u[2] = 0.0
    u /= np.linalg.norm(u)
    seg = [[p, p + rng.uniform(0.3, 4.0) * u]]
    for _ in range(int(rng.integers(2, 6))):
        k = int(rng.integers(len(seg)))
        a1, a2 = seg[k]
        la = float(np.linalg.norm(a2 - a1))
        ua = (a2 - a1) / la
        r = float(rng.uniform(0.002, 0.012))
        d = radii[k] + r
        lb = float(rng.uniform(0.3, 4.0))
        kind = str(rng.choice(["v", "x", "t", "beside"]))
        reach = 0.5 * lb if kind == "x" else lb
        ang = min(rng.uniform(0.3, 3.0) * d / reach, 1.0) * rng.choice([-1.0, 1.0])
        rot = np.array(
            [
                [math.cos(ang), -math.sin(ang), 0.0],
                [math.sin(ang), math.cos(ang), 0.0],
                [0, 0, 1],
            ]
        )
        ub = rot @ ua
        if kind == "v":
            start, away = (a1, ub) if rng.random() < 0.5 else (a2, -ub)
            q1, q2 = start, start + lb * away
        elif kind == "x":
            c = a1 + rng.uniform(0.2, 0.8) * la * ua
            q1, q2 = c - 0.5 * lb * ub, c + 0.5 * lb * ub
        elif kind == "t":
            q1 = a1 + rng.uniform(0.1, 0.9) * la * ua
            q2 = q1 + lb * ub
        else:
            normal = np.array([-ua[1], ua[0], 0.0])
            q1 = (
                a1
                + rng.uniform(-0.3, 0.7) * la * ua
                + rng.uniform(0.0, 2.0) * d * normal
            )
            q2 = q1 + lb * ub
        seg.append([np.asarray(q1, float), np.asarray(q2, float)])
        radii.append(r)
    return np.array(seg, dtype=float), np.array(radii)


# audit A3C-2
@pytest.mark.parametrize("seed", range(60))
def test_random_close_configurations_are_positive_definite(seed: int) -> None:
    """Random V, X, T and side-by-side configurations, with and without
    the internal inductance, in every image convention."""
    seg, radii = _random_close_set(np.random.default_rng(seed))
    for image in ("none", "additive", "antiparallel"):
        for internal in (True, False):
            L = build_inductance_matrix(
                seg, radii, image=image, include_internal=internal
            )
            assert np.linalg.eigvalsh(L).min() > 0.0, (image, internal)


def _v_loop_world(D: float, resistance_model: str) -> gf.World:
    """Out along w1, back along w2 (two 1 m wires from one junction whose
    far ends are D apart), the junction tied to a rod."""
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="rodA", position=(-1.0, 0.0, 0.0), length=2.0)
    eps = math.asin(D / 1.0)
    gf.create_junction(w, name="A", position=(0.0, 0.0, 0.7))
    gf.create_junction(w, name="B", position=(1.0, 0.0, 0.7))
    gf.create_junction(w, name="C", position=(math.cos(eps), math.sin(eps), 0.7))
    kw = dict(
        wire_radius=R,
        resistivity=1.72e-8,
        cross_section=math.pi * R * R,
        coupling_to_soil="isolated",
        inductance_model="neumann",
        discretize_segment_length=None,
        resistance_model=resistance_model,
    )
    for name, p, q, a, b in (
        ("ground", (-1.0, 0.0, 0.0), (0.0, 0.0, 0.7), "rodA", "A"),
        ("w1", (0.0, 0.0, 0.7), (1.0, 0.0, 0.7), "A", "B"),
        ("w2", (0.0, 0.0, 0.7), (math.cos(eps), math.sin(eps), 0.7), "A", "C"),
    ):
        w.add_conductor(
            Conductor(
                name=name, start=p, end=q, start_electrode=a, end_electrode=b, **kw
            )
        )
    gf.create_source(w, attached_to="B", return_to="C", magnitude=1.0)
    return w


# audit A3C-2 (the auditor's test, extended)
@pytest.mark.parametrize("model", ["perfect_mirror", "sommerfeld"])
@pytest.mark.parametrize("resistance_model", ["dc", "skin"])
def test_two_wire_loop_has_a_positive_reactance(model, resistance_model) -> None:
    """Loop out along w1, back along w2 with the far ends 0.5 ... 3
    (r_a + r_b) apart: audit round 3 measured Im Z_in(1 kHz) = -1.40e-4
    Ohm at 1.01 (both models), -5.1e-4 Ohm with the skin model, and with
    the skin model still negative at 1.2."""
    eng = gf.create_engine(
        backend="image",
        segment_length=1.0,
        kernel="point",
        frequencies=[50.0, 1000.0],
        earth_inductive_model=model,
    )
    for ratio in (0.5, 0.99, 1.01, 1.04, 1.2, 1.5, 3.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = eng.solve(
                _v_loop_world(ratio * 2 * R, resistance_model)
            ).input_impedance()
        assert (np.asarray(z).imag > 0.0).all(), ratio


# audit A3C-2
@pytest.mark.parametrize(
    "geometry",
    ["v_far_end", "x_far_end", "parallel_offset", "staggered_overlap"],
)
def test_mutual_is_continuous_across_the_touching_distance(geometry: str) -> None:
    """The rule acts pointwise, so the mutual inductance has no step where
    a far end, the lateral offset or the overlap passes the touching
    distance (the whole-pair rule jumped by up to 27 %)."""

    def seg(x: float) -> np.ndarray:
        if geometry == "v_far_end":
            return _v_pair(1.0, 1.0, math.asin(x * 2 * R))
        if geometry == "x_far_end":
            return _x_pair(1.0, math.asin(x * 2 * R / 0.5))
        if geometry == "parallel_offset":
            return np.array(
                [
                    [[0.0, 0.0, 0.7], [5.0, 0.0, 0.7]],
                    [[1.0, x * 2 * R, 0.7], [7.0, x * 2 * R, 0.7]],
                ]
            )
        # lateral offset 1 cm, the partner's start moves across the end of a
        return np.array(
            [
                [[0.0, 0.0, 0.7], [5.0, 0.0, 0.7]],
                [[5.0 + (1.0 - x), 0.01, 0.7], [9.0, 0.01, 0.7]],
            ]
        )

    radii = np.full(2, R)
    lo = build_inductance_matrix(seg(1.0 - 1e-7), radii, image="none")[0, 1]
    hi = build_inductance_matrix(seg(1.0 + 1e-7), radii, image="none")[0, 1]
    assert abs(hi / lo - 1.0) < 1e-6


# audit A3C-1: the rule leaves alone what it does not act on
def test_chain_neighbours_and_bends_keep_their_closed_forms() -> None:
    """Chain neighbours, bends turning by less than 90 deg and pairs
    farther apart than the touching distance are bit-identical with the
    filament values (the rule only acts beside the partner)."""
    pts = np.array(
        [[0, 0, 0.7], [3, 0, 0.7], [6, 1, 0.7], [7, 4, 0.7], [7, 9, 0.7], [4, 11, 0.7]],
        dtype=float,
    )
    seg = np.concatenate(
        [
            np.stack([pts[:-1], pts[1:]], axis=1),
            _chain((0.0, 0.05, 0.7), (3.0, 0.05, 0.7), 2),
        ]
    )
    thick = build_inductance_matrix(seg, np.full(len(seg), 0.01), image="none")
    thin = build_inductance_matrix(seg, np.full(len(seg), 1e-9), image="none")
    off = ~np.eye(len(seg), dtype=bool)
    assert np.array_equal(thick[off], thin[off])


# ---------------------------------------------------------------- A3C-5


# audit A3C-5
def test_wire_inside_another_couples_at_the_radius_of_the_thicker_wire() -> None:
    """A core (r = 5 mm) on the axis of a shield (r = 15 mm), 100 m: the
    mutual of a filament inside a thin tube, at the tube radius (audit
    round 3: at r_c + r_s, loop inductance +36 %). With the internal
    inductance on the core only (a thin-walled shield has none) the loop
    is the analytic coax mu0/(2 pi) (ln(r_s/r_c) + 1/4) per metre, up to
    the finite length (r_s / l)."""
    length, rc, rs = 100.0, 0.005, 0.015
    seg = np.array(
        [[[0, 0, 0.7], [length, 0, 0.7]], [[0, 0, 0.7], [length, 0, 0.7]]], float
    )
    L = build_inductance_matrix(
        seg, np.array([rc, rs]), image="none", include_internal=np.array([True, False])
    )
    assert L[0, 1] == pytest.approx(
        parallel_segments_mutual(length, rs), rel=1e-12, abs=0
    )
    loop = L[0, 0] + L[1, 1] - 2.0 * L[0, 1]
    coax = 2e-7 * length * (math.log(rs / rc) + 0.25)
    assert loop == pytest.approx(coax, rel=5e-4)


# audit A3C-5 (the auditor's test)
def test_wire_inside_another_uses_the_geometric_mean_distance() -> None:
    length, rc, rs = 100.0, 0.005, 0.015
    seg = np.array(
        [[[0, 0, 0.7], [length, 0, 0.7]], [[0, 0, 0.7], [length, 0, 0.7]]], float
    )
    L = build_inductance_matrix(seg, np.array([rc, rs]), image="none")
    loop = L[0, 0] + L[1, 1] - 2 * L[0, 1]
    loop_gmd = L[0, 0] + L[1, 1] - 2 * parallel_segments_mutual(length, rs)
    assert loop == pytest.approx(loop_gmd, rel=0.05)


# audit A3C-5: equal radii never count as one inside the other
def test_coincident_wires_of_equal_radius_stay_touching_wires() -> None:
    """Two coincident wires of one radius (cables in one trench, the TN
    generator's convention) couple as touching wires, never as a wire
    inside a tube: positive definite even without internal inductance."""
    seg = np.array([[[0, 0, 0.7], [10, 0, 0.7]], [[10, 0, 0.7], [0, 0, 0.7]]], float)
    L = build_inductance_matrix(
        seg, np.full(2, 0.005), image="none", include_internal=False
    )
    assert L[0, 1] == pytest.approx(
        -parallel_segments_mutual(10.0, 0.01), rel=1e-12, abs=0
    )
    assert np.linalg.eigvalsh(L).min() > 0.0
