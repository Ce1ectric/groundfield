"""Neumann partial inductances of touching and coaxial segments (10.46).

Chain neighbours and bends share an end point; the Neumann integrand is
log-singular there and the 16x16 Gauss–Legendre rule is only accurate to
-0.3 % ... +0.8 % (review R3-12). ``touching="exact"`` uses the closed
forms (Grover 1946) and is the default since wave 2 of the release;
``"quadrature"`` reproduces the numbers of 0.15.0.

References: a semi-analytic double integral (inner integral in closed
form, outer adaptive quadrature), written here independently of the
package.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from groundfield.coupling.inductance import (
    MU_0,
    build_inductance_matrix,
    neumann_mutual,
    thin_wire_self_inductance,
)

pytestmark = pytest.mark.finding("10.46")


def _reference_mutual(p1a, p2a, p1b, p2b) -> float:
    p1a, p2a, p1b, p2b = (np.asarray(v, float) for v in (p1a, p2a, p1b, p2b))
    la = np.linalg.norm(p2a - p1a)
    lb = np.linalg.norm(p2b - p1b)
    ua = (p2a - p1a) / la
    ub = (p2b - p1b) / lb

    def inner(s: float) -> float:
        w = p1a + s * ua - p1b
        t0 = w @ ub
        d = math.sqrt(max(w @ w - t0 * t0, 0.0))
        if d < 1e-300:
            return abs(math.log(abs(lb - t0) / abs(t0)))
        return math.asinh((lb - t0) / d) - math.asinh(-t0 / d)

    pts = sorted(
        p for p in (1e-9 * la, 1e-6 * la, 1e-3 * la, la - 1e-6 * la) if 0 < p < la
    )
    val = quad(inner, 0.0, la, limit=800, epsabs=1e-15, epsrel=1e-13, points=pts)[0]
    return MU_0 / (4.0 * math.pi) * float(ua @ ub) * val


@pytest.mark.parametrize("angle", [10.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
def test_bend_matches_reference(angle: float) -> None:
    th = math.radians(angle)
    a = (np.array([0.0, 0.0, 1.0]), np.array([2.0, 0.0, 1.0]))
    b = (
        np.array([2.0, 0.0, 1.0]),
        np.array([2.0 + 3.0 * math.cos(th), 3.0 * math.sin(th), 1.0]),
    )
    ref = _reference_mutual(*a, *b)
    got = neumann_mutual(*a, *b, touching="exact")
    if angle == 90.0:
        assert got == 0.0
    else:
        assert got == pytest.approx(ref, rel=1e-11)


@pytest.mark.parametrize(
    "l1, l2, gap", [(1.0, 1.0, 0.0), (2.0, 5.0, 0.0), (1.0, 1.0, 1e-3), (3.0, 0.5, 2.0)]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_coaxial_pair_matches_reference(l1, l2, gap, reverse) -> None:
    a = (np.array([0.0, 0.0, 1.0]), np.array([l1, 0.0, 1.0]))
    b = (np.array([l1 + gap, 0.0, 1.0]), np.array([l1 + gap + l2, 0.0, 1.0]))
    if reverse:
        b = (b[1], b[0])
    if gap == 0.0:
        F = lambda x: x * math.log(x)  # noqa: E731
        ref = MU_0 / (4 * math.pi) * (F(l1 + l2) - F(l1) - F(l2))
        ref = -ref if reverse else ref
    else:
        ref = _reference_mutual(*a, *b)
    assert neumann_mutual(*a, *b, touching="exact") == pytest.approx(ref, rel=1e-12)


def test_quadrature_rule_is_inaccurate_for_neighbours() -> None:
    """The historic 16x16 rule (opt-in) is -0.21 % off for neighbours."""
    a = (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]))
    b = (np.array([1.0, 0.0, 1.0]), np.array([2.0, 0.0, 1.0]))
    exact = neumann_mutual(*a, *b, touching="exact")
    quadrature = neumann_mutual(*a, *b, touching="quadrature")
    assert quadrature / exact - 1.0 == pytest.approx(-2.13e-3, abs=2e-5)


def test_exact_touching_is_the_default() -> None:
    """Closed forms by default (wave 2 of review pass 10; 0.15.0: quadrature)."""
    a = (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]))
    b = (np.array([1.0, 0.0, 1.0]), np.array([1.5, 0.8, 1.0]))
    assert neumann_mutual(*a, *b) == neumann_mutual(*a, *b, touching="exact")
    eps = np.array([[[0.0, 0, 1.0], [5.0, 0, 1.0]], [[5.0, 0, 1.0], [10.0, 0, 1.0]]])
    radii = np.full(2, 0.004)
    for image in ("none", "additive", "antiparallel"):
        default = build_inductance_matrix(eps, radii, image=image)
        exact = build_inductance_matrix(eps, radii, image=image, touching="exact")
        assert np.array_equal(default, exact)


@pytest.mark.parametrize("n", [2, 4, 10, 40, 200])
def test_chain_sum_is_refinement_invariant(n: int) -> None:
    """A straight wire split into n segments has the self inductance of
    the unsplit wire (v0.15.0 quadrature: 0.99983 ... 0.99965)."""
    x = np.linspace(0.0, 20.0, n + 1)
    eps = np.array([[[x[k], 0, 1.0], [x[k + 1], 0, 1.0]] for k in range(n)])
    L = build_inductance_matrix(eps, np.full(n, 0.004), image="none", touching="exact")
    assert L.sum() == pytest.approx(thin_wire_self_inductance(20.0, 0.004), rel=1e-12)


def test_vectorised_exact_matches_loop_reference() -> None:
    from groundfield.coupling.inductance import _build_inductance_matrix_loop

    rng = np.random.default_rng(3)
    pts = np.cumsum(rng.uniform(-3.0, 3.0, size=(13, 3)), axis=0)
    pts[:, 2] = np.abs(pts[:, 2]) + 0.5
    eps = np.stack([pts[:-1], pts[1:]], axis=1)  # a polyline: all neighbours touch
    radii = np.full(len(eps), 0.005)
    for image in ("none", "additive", "antiparallel"):
        L_vec = build_inductance_matrix(eps, radii, image=image, touching="exact")
        L_loop = _build_inductance_matrix_loop(
            eps, radii, image=image, touching="exact"
        )
        np.testing.assert_allclose(L_vec, L_loop, rtol=1e-12, atol=1e-20)


# ---------------------------------------------------------------------
# Cost of the exact rule (audit round 2, A2C-5): no scalar call per pair
# ---------------------------------------------------------------------


def _straight_chain(m: int, step: float, direction=(1.0, 0.0, 0.0), z: float = 0.7):
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    t = step * np.arange(m + 1)
    pts = t[:, None] * d[None, :] + np.array([0.0, 0.0, z])
    return np.stack([pts[:-1], pts[1:]], axis=1)


# audit A2C-5
@pytest.mark.parametrize(
    "direction, image",
    [
        ((1.0, 0.0, 0.0), "none"),
        ((1.0, 0.0, 0.0), "antiparallel"),
        ((0.956, 0.292, 0.0), "none"),  # oblique: the axis is not a coordinate axis
        ((0.0, 0.0, 1.0), "additive"),  # vertical: coaxial with its own image too
    ],
)
def test_scalar_closed_form_calls_grow_linearly(monkeypatch, direction, image) -> None:
    """A straight conductor of M sub-segments has M(M-1)/2 coaxial pairs;
    the exact rule evaluates them vectorised (base: one Python call per
    pair, 19 900 calls for M = 200 and 2-4x the assembly time of the
    quadrature). Scalar calls are left for the diagonal image terms only."""
    import groundfield.coupling.inductance as ind

    calls = {"n": 0}
    original = ind._touching_mutual

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ind, "_touching_mutual", counting)
    m = 200
    seg = _straight_chain(m, 5.0 if direction[2] == 0.0 else 0.1, direction)
    ind.build_inductance_matrix(seg, np.full(m, 0.007), image=image)
    assert calls["n"] <= 2 * m, f"{calls['n']} scalar calls for {m} segments"


# audit A2C-5
def test_touching_batch_is_bit_identical_with_the_scalar_closed_forms() -> None:
    """The vectorised closed forms equal the scalar ones bit for bit, for
    coaxial pairs (gaps, touching neighbours, reversed, overlapping),
    bends and unrelated partners, on and off the coordinate axes."""
    from groundfield.coupling.inductance import (
        _touching_mutual,
        _touching_mutual_batch,
    )

    rng = np.random.default_rng(2026)
    n_closed = 0
    for trial in range(60):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        if trial % 3 == 0:
            d = np.array([0.0, 1.0, 0.0])
        origin = rng.normal(size=3) * 10.0
        t = np.sort(rng.uniform(-50.0, 50.0, 10))
        pts = origin + t[:, None] * d
        p1, p2 = pts[:-1].copy(), pts[1:].copy()
        flip = rng.random(p1.shape[0]) < 0.3
        p1[flip], p2[flip] = p2[flip].copy(), p1[flip].copy()
        extra_1 = [pts[3], rng.normal(size=3) * 20.0, origin + (t[2] + 0.3) * d]
        extra_2 = [pts[3] + rng.normal(size=3), rng.normal(size=3) * 20.0]
        extra_2.append(origin + (t[4] + 0.1) * d)  # overlaps: no closed form
        p1 = np.vstack([p1, *extra_1])
        p2 = np.vstack([p2, *extra_2])
        for i in range(p1.shape[0]):
            q1, q2 = np.delete(p1, i, 0), np.delete(p2, i, 0)
            values, valid = _touching_mutual_batch(p1[i], p2[i], q1, q2)
            for k in range(q1.shape[0]):
                ref = _touching_mutual(p1[i], p2[i], q1[k], q2[k])
                assert valid[k] == (ref is not None)
                if ref is not None:
                    assert values[k] == ref  # bit for bit
                    n_closed += 1
    assert n_closed > 3000


# audit A2C-5
@pytest.mark.benchmark
def test_exact_rule_on_a_straight_chain_is_not_slower_than_twice_the_quadrature() -> (
    None
):
    """Wall-time guard (generous: the CI machine is shared). Base: 2.4-4.2x
    for M = 200-800; now about 1.0-1.3x."""
    import time

    seg = _straight_chain(300, 5.0)
    radii = np.full(300, 0.007)

    def best(rule: str) -> float:
        out = []
        for _ in range(3):
            t0 = time.perf_counter()
            build_inductance_matrix(seg, radii, image="none", touching=rule)
            out.append(time.perf_counter() - t0)
        return min(out)

    assert best("exact") < 2.0 * best("quadrature")
