"""Validate the Neumann inductance helpers against the standard
Oeding/Oswald (2016) loop-inductance formulas.

Two reference results from chapter 9 of *Elektrische Kraftwerke und
Netze* are checked here:

- **Eq. (9.13c)** — self-inductance per unit length of a two-wire
  loop with parallel round conductors of radius $r$ at distance
  $d$:
  $$L'_S = (\\mu_0 / \\pi)\\,[\\ln(d/r) + 1/4].$$
- **Eq. (9.8)** — mutual-inductance per unit length between two
  two-wire loops:
  $$L'_{c,\\mathrm{I,II}} = (\\mu_0 / 2\\pi)\\,
    \\ln(s_{14}\\,s_{23} / (s_{13}\\,s_{24})).$$

The Neumann implementation in :mod:`groundfield.coupling.inductance`
treats finite-length straight wires; in the limit $L \\to \\infty$
the partial-inductance combination

$$
L_\\text{loop}(L) = 2\\,L_\\text{self}(L) - 2\\,M_\\parallel(L, d)
$$

must approach the Oeding result $L \\cdot L'_S$ with end-effect
error of order $1/L$. We test that:

1. the relative error at $L = 100\\,\\mathrm{m}$ stays below 0.5 %
   (well-engineered AP1 PEN strands), and
2. the error decreases at least like $1/L$ as the wire gets longer
   (sanity for the asymptotic regime).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from groundfield.coupling.inductance import (
    MU_0,
    neumann_mutual,
    parallel_segments_mutual,
    thin_wire_self_inductance,
)
from groundfield.references import oeding


# ---------------------------------------------------------------------
# 1. Reference formulas — basic algebraic checks
# ---------------------------------------------------------------------


def test_oeding_loop_self_default_includes_internal() -> None:
    """``loop_self_inductance_per_length`` with the default flag
    matches Eq. (9.13c) including the +1/4 internal-field term."""
    d = 0.5
    r = 0.005
    L = oeding.loop_self_inductance_per_length(d, r)
    expected = (MU_0 / math.pi) * (math.log(d / r) + 0.25)
    assert L == pytest.approx(expected, rel=1e-12)


def test_oeding_loop_self_external_only() -> None:
    """Without the internal-field contribution we recover the pure
    external-field expression $(\\mu_0 / \\pi)\\ln(d/r)$."""
    d = 0.5
    r = 0.005
    L = oeding.loop_self_inductance_per_length(d, r, include_internal=False)
    expected = (MU_0 / math.pi) * math.log(d / r)
    assert L == pytest.approx(expected, rel=1e-12)


def test_oeding_two_wire_radius_helper() -> None:
    """``two_wire_loop_radius`` returns $r \\cdot e^{-1/4}$ (the
    mean geometric distance of a circle from itself)."""
    r = 0.01
    r_eff = oeding.two_wire_loop_radius(r)
    assert r_eff == pytest.approx(r * math.exp(-0.25), rel=1e-12)


def test_oeding_internal_inductance_value() -> None:
    """Internal-field inductance per unit length is
    $\\mu_0 / (8\\pi)$ — a fixed material constant."""
    L_int = oeding.internal_inductance_per_length()
    assert L_int == pytest.approx(MU_0 / (8.0 * math.pi), rel=1e-12)


def test_oeding_mutual_validates_inputs() -> None:
    """All four distance arguments must be positive."""
    with pytest.raises(ValueError):
        oeding.loop_mutual_inductance_per_length(s13=0.0, s14=1.0, s23=1.0, s24=1.0)


# ---------------------------------------------------------------------
# 2. Self-inductance: Neumann implementation vs Oeding 9.13c
# ---------------------------------------------------------------------


def _neumann_loop_self_per_length(L: float, d: float, r: float) -> float:
    """Combine partial inductances of two parallel wires into the
    loop self-inductance per unit length.

    Loop convention: wire 1 carries +I, wire 2 carries −I; both
    have the same axis direction. The loop-current contribution
    is $L_\\text{loop} = L_{11} + L_{22} - 2 M_{12}$.
    """
    L_self = thin_wire_self_inductance(L, r, include_internal=True)
    M_par = parallel_segments_mutual(L, d)
    return (2.0 * L_self - 2.0 * M_par) / L


@pytest.mark.parametrize(
    "wire_length,distance,radius,tolerance",
    [
        (100.0, 0.5, 0.005, 0.005),     # 30 m AP1 PEN run, ≤ 0.5 %
        (1000.0, 0.5, 0.005, 0.001),    # very long wire, ≤ 0.1 %
        (500.0, 1.0, 0.01, 0.001),      # MV cable separation, ≤ 0.1 %
        (100.0, 0.2, 0.0075, 0.005),    # close pair, finite length
    ],
)
def test_neumann_loop_self_matches_oeding(
    wire_length: float, distance: float, radius: float, tolerance: float,
) -> None:
    """The Neumann partial-inductance combination converges to the
    Oeding 9.13c per-unit-length result at long-wire lengths."""
    L_oeding = oeding.loop_self_inductance_per_length(distance, radius)
    L_num = _neumann_loop_self_per_length(wire_length, distance, radius)
    rel = abs(L_num - L_oeding) / L_oeding
    assert rel < tolerance, (
        f"L={wire_length}, d={distance}, r={radius}: "
        f"Oeding={L_oeding*1e6:.4f} µH/m, Neumann={L_num*1e6:.4f} µH/m, "
        f"rel={rel*100:.3f}%"
    )


def test_neumann_loop_self_converges_at_one_over_L() -> None:
    """End-effect error decreases like 1/L as the wire is stretched."""
    d, r = 0.5, 0.005
    L_oeding = oeding.loop_self_inductance_per_length(d, r)
    errors = []
    lengths = [10.0, 100.0, 1000.0, 10000.0]
    for L in lengths:
        L_num = _neumann_loop_self_per_length(L, d, r)
        errors.append(abs(L_num - L_oeding))
    # Each tenfold increase in L should reduce the error by roughly
    # the same factor 10 (or better).
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert ratio > 9.5, (
            f"convergence too slow: L={lengths[i]}→{lengths[i+1]}, "
            f"ratio={ratio:.2f}"
        )


# ---------------------------------------------------------------------
# 3. Mutual inductance: Neumann implementation vs Oeding 9.8
# ---------------------------------------------------------------------


def _neumann_loop_mutual_per_length(
    L: float,
    pos1: tuple[float, float], pos2: tuple[float, float],
    pos3: tuple[float, float], pos4: tuple[float, float],
) -> float:
    """Loop-loop mutual inductance per unit length, evaluated by
    summing partial mutuals between every wire pair.

    All four wires run along the z-axis from 0 to L; ``pos*`` are
    the (x, y) cross-sectional coordinates.
    """
    def M(p, q):
        a1 = np.array([p[0], p[1], 0.0])
        a2 = np.array([p[0], p[1], L])
        b1 = np.array([q[0], q[1], 0.0])
        b2 = np.array([q[0], q[1], L])
        return neumann_mutual(a1, a2, b1, b2)
    M13 = M(pos1, pos3)
    M14 = M(pos1, pos4)
    M23 = M(pos2, pos3)
    M24 = M(pos2, pos4)
    return (M13 - M14 - M23 + M24) / L


def test_neumann_loop_mutual_matches_oeding() -> None:
    """Two parallel two-wire loops: the Neumann partial-mutuals
    reproduce Oeding Eq. (9.8) for $L = 100\\,\\mathrm{m}$ to
    within 0.1 %."""
    pos1 = (0.0, 0.0)
    pos2 = (0.5, 0.0)
    pos3 = (3.0, 0.0)
    pos4 = (3.5, 0.0)
    s13 = math.dist(pos1, pos3)
    s14 = math.dist(pos1, pos4)
    s23 = math.dist(pos2, pos3)
    s24 = math.dist(pos2, pos4)
    M_oeding = oeding.loop_mutual_inductance_per_length(
        s13=s13, s14=s14, s23=s23, s24=s24,
    )
    M_num = _neumann_loop_mutual_per_length(
        100.0, pos1, pos2, pos3, pos4,
    )
    rel = abs(M_num - M_oeding) / abs(M_oeding)
    assert rel < 0.001, (
        f"Oeding={M_oeding*1e6:.4f} µH/m, Neumann={M_num*1e6:.4f} µH/m, "
        f"rel={rel*100:.3f}%"
    )


def test_neumann_loop_mutual_offset_geometry() -> None:
    """Two-loop geometry with a vertical offset between the loops:
    $s_{ij}$ are no longer all in a line, but Eq. 9.8 still applies.

    Note: the loop-loop mutual is here much smaller in magnitude
    than in the in-line case (the four partial mutuals cancel more
    completely), so the constant end-effect contribution makes the
    *relative* error converge slower with $L$. We use a longer
    wire ($L = 5\\,\\mathrm{km}$) to bring the relative error
    safely below 0.1 %.
    """
    pos1 = (0.0, 0.0)
    pos2 = (0.5, 0.0)
    pos3 = (2.0, 1.5)
    pos4 = (2.5, 1.5)
    s13 = math.dist(pos1, pos3)
    s14 = math.dist(pos1, pos4)
    s23 = math.dist(pos2, pos3)
    s24 = math.dist(pos2, pos4)
    M_oeding = oeding.loop_mutual_inductance_per_length(
        s13=s13, s14=s14, s23=s23, s24=s24,
    )
    M_num = _neumann_loop_mutual_per_length(
        5000.0, pos1, pos2, pos3, pos4,
    )
    rel = abs(M_num - M_oeding) / abs(M_oeding)
    assert rel < 0.001, (
        f"Oeding={M_oeding*1e6:.4f} µH/m, Neumann={M_num*1e6:.4f} µH/m, "
        f"rel={rel*100:.3f}%"
    )


def test_neumann_loop_mutual_offset_converges() -> None:
    """Even when the loop-loop mutual is small in magnitude, the
    end-effect error scales like $1/L$. We sample two lengths and
    require the longer one to halve the error of the shorter."""
    pos1 = (0.0, 0.0)
    pos2 = (0.5, 0.0)
    pos3 = (2.0, 1.5)
    pos4 = (2.5, 1.5)
    s13 = math.dist(pos1, pos3)
    s14 = math.dist(pos1, pos4)
    s23 = math.dist(pos2, pos3)
    s24 = math.dist(pos2, pos4)
    M_oeding = oeding.loop_mutual_inductance_per_length(
        s13=s13, s14=s14, s23=s23, s24=s24,
    )
    M_500 = _neumann_loop_mutual_per_length(500.0, pos1, pos2, pos3, pos4)
    M_5000 = _neumann_loop_mutual_per_length(5000.0, pos1, pos2, pos3, pos4)
    err_500 = abs(M_500 - M_oeding)
    err_5000 = abs(M_5000 - M_oeding)
    # Ten-fold L should reduce the error by roughly 10× (1/L scaling)
    assert err_500 / err_5000 > 8.0, (err_500, err_5000)


def test_oeding_mutual_zero_when_loops_far_aligned() -> None:
    """When two loops sit symmetrically (s13·s24 == s14·s23), the
    mutual cancels exactly — Oeding 9.8 becomes ln(1) = 0."""
    # 1↔3 and 2↔4 are symmetric: place 3 and 4 in equal-distance
    # arrangement so that s13·s24 = s14·s23.
    M_oeding = oeding.loop_mutual_inductance_per_length(
        s13=2.0, s14=2.0, s23=2.0, s24=2.0,
    )
    assert M_oeding == pytest.approx(0.0, abs=1e-15)
