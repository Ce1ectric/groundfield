"""Exact two-layer Pollaczek reference (review pass 10, finding 10.45).

The reference below is written here independently of the module: the
two-interface boundary-value problem for the vector potential of a line
current is solved numerically per lambda (4x4 linear system), the
spectral integral by adaptive quadrature, and the direct term is
K0(gamma GMR).
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from groundfield.references.pollaczek import (
    pollaczek_mutual_impedance,
    pollaczek_self_impedance,
)

pytestmark = pytest.mark.finding("10.45")

MU_0 = 4e-7 * math.pi
GMR = 0.007


def _amplitude(lam, omega, rho_1, rho_2, d, z_obs, z_src) -> complex:
    """Secondary field at z_obs of a unit spectral primary at z_src.

    Unknowns: c (air, e^{lam z}), a1 / b1 (layer 1, e^{-u1 z},
    e^{-u1 (d - z)}), a2 (layer 2, e^{-u2 (z - d)}).
    """
    u0 = lam
    u1 = cmath.sqrt(lam * lam + 1j * omega * MU_0 / rho_1)
    u2 = cmath.sqrt(lam * lam + 1j * omega * MU_0 / rho_2)
    E1 = cmath.exp(-u1 * d)
    A = np.array(
        [
            [1.0, -1.0, -E1, 0.0],
            [u0, u1, -u1 * E1, 0.0],
            [0.0, E1, 1.0, -1.0],
            [0.0, -u1 * E1, u1, u2],
        ],
        dtype=complex,
    )
    b = np.zeros(4, dtype=complex)
    if z_src < d:  # source in layer 1
        e_top = cmath.exp(-u1 * z_src)
        e_bot = cmath.exp(-u1 * (d - z_src))
        b[0] += e_top / u1
        b[1] += e_top
        b[2] -= e_bot / u1
        b[3] -= -e_bot
        uj = u1
    else:  # source in layer 2
        e_top = cmath.exp(-u2 * (z_src - d))
        b[2] += e_top / u2
        b[3] += e_top
        uj = u2
    c, a1, b1, a2 = np.linalg.solve(A, b)
    if z_obs < d:
        return a1 * cmath.exp(-u1 * z_obs) + b1 * cmath.exp(-u1 * (d - z_obs))
    del uj
    return a2 * cmath.exp(-u2 * (z_obs - d))


def _reference(rho_1, rho_2, d, f, h_i, h_j, x, gmr=None) -> complex:
    omega = 2.0 * math.pi * f

    def part(lam, k):
        v = _amplitude(lam, omega, rho_1, rho_2, d, h_i, h_j) * math.cos(lam * x)
        return v.real if k == 0 else v.imag

    kw = dict(limit=1000, epsabs=1e-15, epsrel=1e-12)
    split = 60.0 / max(min(h_i + h_j, abs(h_i - h_j) or 1e9, 2 * d), 1e-3)
    # breakpoints at the skin wavenumbers and geometric scales
    scales = [math.sqrt(omega * MU_0 / r) for r in (rho_1, rho_2)]
    scales += [1.0 / (2 * d), 1.0 / (h_i + h_j)]
    pts = sorted({c * s for s in scales for c in (0.1, 1.0, 10.0) if 0 < c * s < split})
    re = quad(part, 0, split, args=(0,), points=pts, **kw)[0]
    re += quad(part, split, np.inf, args=(0,), **kw)[0]
    im = quad(part, 0, split, args=(1,), points=pts, **kw)[0]
    im += quad(part, split, np.inf, args=(1,), **kw)[0]
    same = (h_i < d) == (h_j < d)
    head = 0j
    if same:
        rho_k = rho_1 if h_i < d else rho_2
        dist = gmr if gmr is not None else math.hypot(x, h_i - h_j)
        head = complex(kv(0, cmath.sqrt(1j * omega * MU_0 / rho_k) * dist))
    return 1j * omega * MU_0 / (2 * math.pi) * (head + re + 1j * im)


@pytest.mark.parametrize(
    "rho_1, rho_2, h_1, depth, f",
    [
        (1000.0, 30.0, 5.0, 0.7, 50.0),
        (100.0, 10.0, 1.0, 0.9, 1000.0),  # module was -0.52 % in R'
        (160.776, 34.074, 1.848, 1.2, 1e6),  # Case V, module was -28.5 %
        (100.0, 10.0, 1.0, 3.0, 1000.0),  # conductor in layer 2
    ],
)
def test_self_impedance_matches_independent_bvp(rho_1, rho_2, h_1, depth, f) -> None:
    z = pollaczek_self_impedance(rho_1, rho_2, h_1, f, depth=depth, gmr=GMR)
    ref = _reference(rho_1, rho_2, h_1, f, depth, depth, 0.0, gmr=GMR)
    assert z.real == pytest.approx(ref.real, rel=1e-7)
    assert z.imag == pytest.approx(ref.imag, rel=1e-7)


@pytest.mark.parametrize("f, depth", [(1000.0, 1.0), (1000.0, 3.0), (10_000.0, 3.0)])
def test_homogeneous_self_uses_the_exact_k0_head(f, depth) -> None:
    """ln(2h/GMR) head: R' -0.15 % (1 m) / -0.99 % (3 m) at 30 Ohm m, 1 kHz."""
    z = pollaczek_self_impedance(30.0, 30.0, 10.0, f, depth=depth, gmr=GMR)
    ref = _reference(30.0, 30.0, 10.0, f, depth, depth, 0.0, gmr=GMR)
    assert z.real == pytest.approx(ref.real, rel=1e-7)
    assert z.imag == pytest.approx(ref.imag, rel=1e-7)


@pytest.mark.parametrize(
    "h_i, h_j, x",
    [(0.7, 0.7, 4.0), (0.6, 0.9, 0.5), (0.7, 3.0, 4.0), (2.5, 3.5, 1.0)],
    ids=["layer1", "layer1_close", "cross_layer", "layer2"],
)
def test_mutual_impedance_matches_independent_bvp(h_i, h_j, x) -> None:
    z = pollaczek_mutual_impedance(
        100.0, 10.0, 2.0, 1000.0, depth_i=h_i, depth_j=h_j, separation=x
    )
    ref = _reference(100.0, 10.0, 2.0, 1000.0, h_i, h_j, x)
    assert z.real == pytest.approx(ref.real, rel=1e-6)
    assert z.imag == pytest.approx(ref.imag, rel=1e-6)
