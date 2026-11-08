"""Neumann mutual inductance of close, non-parallel segment pairs.

The 16x16 Gauss–Legendre rule resolves 1/r only when the segments are
far apart compared with their length. Pairs that miss the closed forms —
nearly parallel wires close together whose directions differ by more than
the parallel tolerance (1e-9 in cos), crossing pairs, skew pairs passing
close by — were integrated with it. Two 5 m segments with aligned ends and
1e-5 ... 1e-3 rad between them: +12 % at 10 cm, +200 ... +250 % at 1 cm;
with offset ends -19 % at 1 mm; crossing pairs about -1 %, skew pairs 1 cm
apart -2.3 ... +0.5 %. Found in audit round 2 of the calendar release
while attacking the guard for wires lying side by side (the generator's
in-line probe lead starts 1 cm beside the feed lead and ends on its axis).
With ``touching="exact"`` (the default) such pairs are integrated
semi-analytically: inner integral in closed form, outer Gauss–Legendre
panels graded towards the near-singular points.

Reference: an adaptive quadrature of the same inner closed form, written
here independently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from groundfield.coupling.inductance import (
    MU_0,
    _build_inductance_matrix_loop,
    build_inductance_matrix,
    neumann_mutual,
    parallel_segments_mutual,
)

pytestmark = pytest.mark.finding("10.46")


def _reference(p1a, p2a, p1b, p2b) -> float:
    la = np.linalg.norm(p2a - p1a)
    lb = np.linalg.norm(p2b - p1b)
    ua = (p2a - p1a) / la
    ub = (p2b - p1b) / lb

    def inner(s: float) -> float:
        w = p1a + s * ua - p1b
        t0 = float(w @ ub)
        d = max(math.sqrt(max(float(w @ w) - t0 * t0, 0.0)), 1e-300)
        return math.asinh((lb - t0) / d) + math.asinh(t0 / d)

    cuts = sorted(
        {0.0, la}
        | {float(x) for x in ((p1b - p1a) @ ua, (p2b - p1a) @ ua) if 0.0 < x < la}
    )
    val = sum(
        quad(inner, a, b, limit=500, epsabs=0.0, epsrel=1e-12)[0]
        for a, b in zip(cuts[:-1], cuts[1:], strict=True)
    )
    return MU_0 / (4.0 * math.pi) * float(ua @ ub) * val


def _offset_tilt(d: float, phi: float):
    a = (np.array([0.0, 0.0, 0.7]), np.array([5.0, 0.0, 0.7]))
    b1 = np.array([0.5, d, 0.7])
    b2 = b1 + 5.0 * np.array([math.cos(phi), math.sin(phi), 0.0])
    return (*a, b1, b2)


def _crossing(angle_deg: float, dz: float = 0.0):
    th = math.radians(angle_deg)
    a = (np.array([0.0, 0.0, 0.7]), np.array([5.0, 0.0, 0.7]))
    b1 = np.array([2.5 - 2.0 * math.cos(th), -2.0 * math.sin(th), 0.7 + dz])
    b2 = np.array([2.5 + 2.0 * math.cos(th), 2.0 * math.sin(th), 0.7 + dz])
    return (*a, b1, b2)


def _t_junction(angle_deg: float):
    """Segment b starts on the interior of a (no shared end point)."""
    th = math.radians(angle_deg)
    a = (np.array([0.0, 0.0, 0.7]), np.array([10.0, 0.0, 0.7]))
    b1 = np.array([4.0, 0.0, 0.7])
    return (*a, b1, b1 + 5.0 * np.array([math.cos(th), math.sin(th), 0.0]))


CASES = {
    "1mm_1e-4rad": _offset_tilt(0.001, 1e-4),  # base -19 %
    "1cm_1e-2rad": _offset_tilt(0.01, 1e-2),  # base +10.5 %
    "10cm_1e-4rad": _offset_tilt(0.1, 1e-4),  # base +4.9 %
    "50cm_0.3rad": _offset_tilt(0.5, 0.3),  # base -2.3e-7
    "crossing_5deg": _crossing(5.0),  # base -1.1 %
    "crossing_30deg": _crossing(30.0),  # base -0.9 %
    "skew_1cm_5deg": _crossing(5.0, 0.01),  # base +0.5 %
    "skew_1cm_45deg": _crossing(45.0, 0.01),  # base -2.3 %
    "t_junction_45deg": _t_junction(45.0),  # base -3.5 %
    "t_junction_1deg": _t_junction(1.0),  # base -13 %
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_close_pairs_match_an_adaptive_reference(name: str) -> None:
    pts = CASES[name]
    assert neumann_mutual(*pts) == pytest.approx(_reference(*pts), rel=1e-9, abs=0)


def test_the_matrix_assembly_uses_the_same_rule() -> None:
    """Filament-thin wires (1e-12 m): the crossing and T-junction pairs
    touch, and with physical radii they get the rule for wires lying
    beside each other closer than touching (tests/test_interpenetrating_wires.py;
    audit round 3, A3C-1)."""
    for name in sorted(CASES):
        p1a, p2a, p1b, p2b = CASES[name]
        seg = np.array([[p1a, p2a], [p1b, p2b]])
        L = build_inductance_matrix(seg, np.full(2, 1e-12), image="none")
        assert L[0, 1] == pytest.approx(neumann_mutual(*CASES[name]), rel=1e-13, abs=0)
    seg = np.concatenate(
        [np.array([[c[0], c[1]], [c[2], c[3]]]) for c in CASES.values()]
    )
    radii = np.full(len(seg), 1e-4)
    for image in ("none", "additive", "antiparallel"):
        np.testing.assert_allclose(
            build_inductance_matrix(seg, radii, image=image),
            _build_inductance_matrix_loop(seg, radii, image=image),
            rtol=1e-12,
            atol=1e-20,
        )


def test_the_quadrature_rule_keeps_the_0_15_0_numbers() -> None:
    """``touching="quadrature"`` reproduces 0.15.0, also for close pairs."""
    pts = CASES["1cm_1e-2rad"]
    old = neumann_mutual(*pts, touching="quadrature")
    assert old / _reference(*pts) - 1.0 > 0.05


def test_in_line_probe_lead_couples_as_touching_wires() -> None:
    """The generator's in-line probe lead (50 m, from 1 cm beside the feed
    lead to its axis) against a 150 m feed lead, 5 m segments: touching
    wires over the 50 m overlap plus the end-to-end coupling with the rest
    of the feed lead (base: 92.8 uH from quadrature errors)."""

    def chain(a, b, n):
        a, b = np.asarray(a, float), np.asarray(b, float)
        pts = a + np.linspace(0.0, 1.0, n + 1)[:, None] * (b - a)
        return np.stack([pts[:-1], pts[1:]], axis=1)

    feed = chain((0.0, 0.0, -0.1), (150.0, 0.0, -0.1), 30)
    probe = chain((0.0, 0.01, -0.1), (50.0, 0.0, -0.1), 10)
    seg = np.concatenate([feed, probe])
    L = build_inductance_matrix(seg, np.full(len(seg), 0.005), image="none")

    def F(x: float) -> float:
        return x * math.log(x)

    end_to_end = MU_0 / (4.0 * math.pi) * (F(150.0) - F(50.0) - F(100.0))
    expected = parallel_segments_mutual(50.0, 0.01) + end_to_end
    assert L[:30, 30:].sum() == pytest.approx(expected, rel=2e-3, abs=0)


# ---------------------------------------------------------------------
# audit round 3: tiny angles (A3C-4)
# ---------------------------------------------------------------------


def _mp_reference(p1a, p2a, p1b, p2b) -> float:
    """mu0/(4 pi) (ua.ub) int_a [asinh((lb - t0)/rho) + asinh(t0/rho)] ds in
    mpmath (30 digits), tanh-sinh on the pieces between the closest
    approach of the axes, the passages of b's ends and the ends of a (the
    near-singular points sit at the ends of the pieces). Agrees with the
    auditor's graded variant to 7e-12."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    A1, A2, B1, B2 = (
        mp.matrix([mp.mpf(repr(float(c))) for c in p]) for p in (p1a, p2a, p1b, p2b)
    )
    da, db = A2 - A1, B2 - B1
    la, lb = mp.norm(da), mp.norm(db)
    ua, ub = da / la, db / lb
    dot = sum(ua[i] * ub[i] for i in range(3))
    v0 = A1 - B1
    c0 = sum(v0[i] * ub[i] for i in range(3))

    def g(s):
        v = v0 + s * ua
        t0 = c0 + s * dot
        w = v - t0 * ub
        rho = mp.sqrt(sum(w[i] ** 2 for i in range(3))) or mp.mpf(10) ** -60
        return mp.asinh((lb - t0) / rho) + mp.asinh(t0 / rho)

    marks = {mp.mpf(0), la}
    for target in (mp.mpf(0), lb):
        x = (target - c0) / dot
        if 0 < x < la:
            marks.add(x)
    x = (dot * c0 - sum(v0[i] * ua[i] for i in range(3))) / (1 - dot * dot)
    if 0 < x < la:
        marks.add(x)
    return float(mp.mpf(1e-7) * dot * mp.quad(g, sorted(marks)))


def _rotated(points, yaw: float = 0.3, pitch: float = 0.7):
    """Rotate about z by ``yaw``, about y by ``pitch``, then shift."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    out = []
    for p in points:
        x, y, z = cy * p[0] - sy * p[1], sy * p[0] + cy * p[1], p[2]
        out.append(np.array([cp * x - sp * z + 3.0, y - 2.0, sp * x + cp * z + 0.7]))
    return out


# audit A3C-4 (the auditor's test, extended)
@pytest.mark.parametrize(
    ("phi", "where", "frame"),
    [
        (1e-8, 2.5, "aligned"),
        (1e-8, 0.05, "rotated"),
        (7e-8, 2.5, "aligned"),
        (7e-8, 0.05, "rotated"),
        (1e-6, 2.5, "rotated"),
        (1e-4, 0.05, "aligned"),
    ],
)
def test_crossing_filaments_at_tiny_angles(
    phi: float, where: float, frame: str
) -> None:
    """Two 5 m filaments crossing at ``where`` along a at the angle ``phi``.
    Audit round 3: the closest-approach mark came from 1 - cos^2, which
    cancels and fell below its 1e-14 cut-off: 9.5e-4 at 1e-8 rad. The mark
    now comes from a cross product; measured worst case 1.0e-9 at 1e-8 ...
    3e-8 rad, 3.5e-10 at 7e-8 rad (the crossing point of two axes 1e-8 rad
    apart is only known to about 1e-16/1e-8 of the length), 8.6e-11 at
    1e-7 rad, below 5e-11 from 1e-6 rad on."""
    u = np.array([math.cos(phi), math.sin(phi), 0.0])
    c = np.array([where, 0.0, 0.0])
    pts = [np.zeros(3), np.array([5.0, 0.0, 0.0]), c - 2.0 * u, c + 3.0 * u]
    if frame == "rotated":
        pts = _rotated(pts)
    bound = 2e-9 if phi < 1e-7 else (1e-10 if phi < 1e-5 else 5e-12)
    assert neumann_mutual(*pts) == pytest.approx(_mp_reference(*pts), rel=bound, abs=0)
