"""Speed of the ADR-0007 cross-layer Sommerfeld path (roadmap row
*Vectorise the ADR-0007 cross-layer Sommerfeld path*).

Wave 3 of the 2026.11 work (stream w3xl, item 4). The path is grouped by
depth since the 2026-07-08 audit; what remained per group were one LAPACK
solve of a 3 × 3 system per quadrature node and twice per group (layered
and baseline amplitudes) — half the time of a cross-layer solve (0.40 of
0.85 s at the AP1 station with 10 % penetration, h_1 = 5 m, rods through
the interface). Batching those solves into one LAPACK call gains nothing
(0.208 against 0.216 s for the 463 376 systems of that solve, measured):
LAPACK spends its time per system. The system is now eliminated in closed
form, vectorised over the nodes (30 times faster than LAPACK on its own,
the solve 1.9 times faster). Results move by round-off: the reaction
matrices of the AP1 station by at most 4e-17 of their largest entry, its
impedance by 4.3e-16, six golden quantities by at most 2.5e-16.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf

SOILS = [
    # (z_s, rho_1, rho_2, h_1): source in either layer, mild to extreme contrasts
    (0.8, 500.0, 100.0, 5.0),
    (5.5, 500.0, 100.0, 5.0),
    (0.8, 100.0, 100.0, 5.0),
    (4.99, 1000.0, 30.0, 5.0),
    (5.01, 30.0, 1000.0, 5.0),
    (0.3, 1e4, 1.0, 0.5),
    (0.6, 1.0, 1e4, 0.5),
]


def _lapack_amplitudes(lambdas, z_s, rho_1, rho_2, h_1):
    """The stable 3 × 3 system of ``_spectral_amplitudes``, solved by LAPACK."""
    e = np.exp(-lambdas * h_1)
    s1, s2 = 1.0 / rho_1, 1.0 / rho_2
    zero = np.zeros_like(lambdas)
    if z_s <= h_1:
        d0 = (rho_1 / 2.0) * np.exp(-lambdas * z_s)
        ph = (rho_1 / (2.0 * lambdas)) * np.exp(-lambdas * (h_1 - z_s))
        dph = -(rho_1 / 2.0) * np.exp(-lambdas * (h_1 - z_s))
        p2, dp2 = zero, zero
    else:
        d0, ph, dph = zero, zero, zero
        p2 = (rho_2 / (2.0 * lambdas)) * np.exp(-lambdas * (z_s - h_1))
        dp2 = (rho_2 / 2.0) * np.exp(-lambdas * (z_s - h_1))
    M = np.zeros((lambdas.size, 3, 3))
    M[:, 0, 0], M[:, 0, 1] = lambdas * e, -lambdas
    M[:, 1, 0], M[:, 1, 1], M[:, 1, 2] = 1.0, e, -1.0
    M[:, 2, 0], M[:, 2, 1], M[:, 2, 2] = s1 * lambdas, -s1 * lambdas * e, s2 * lambdas
    rhs = np.stack([-d0, p2 - ph, s2 * dp2 - s1 * dph], axis=-1)
    sol = np.linalg.solve(M, rhs[..., None])[..., 0]
    return sol[:, 0], sol[:, 1], sol[:, 2]


@pytest.mark.parametrize(("z_s", "rho_1", "rho_2", "h_1"), SOILS)
def test_closed_form_amplitudes_equal_the_linear_solve(z_s, rho_1, rho_2, h_1) -> None:
    """The elimination reproduces LAPACK to round-off.

    1.1e-14 of the amplitude scale measured, 2.4e-12 for the contrast 1e4
    with the source in the lower layer, where LAPACK itself is 1.9e-12 off
    the 50-digit reference and the elimination 5.3e-13 (next test).
    """
    from groundfield.coupling.layered_green import _spectral_amplitudes

    lam = np.geomspace(1e-12, 2e3, 4001)
    A, B, _, C = _spectral_amplitudes(lam, z_s, rho_1, rho_2, h_1)
    Ar, Br, Cr = _lapack_amplitudes(lam, z_s, rho_1, rho_2, h_1)
    scale = np.abs(Ar) + np.abs(Br) + np.abs(Cr)
    for got, ref in ((A, Ar), (B, Br), (C, Cr)):
        assert np.all(np.abs(got - ref) <= 5e-12 * scale + 1e-300)


@pytest.mark.parametrize(("z_s", "rho_1", "rho_2", "h_1"), SOILS)
def test_closed_form_amplitudes_against_a_50_digit_reference(
    z_s, rho_1, rho_2, h_1
) -> None:
    """Against mpmath (50 digits) the elimination is at least as good as LAPACK.

    Worst case over these soils and λ = 1e-7 … 2e3 1/m: LAPACK 1.9e-12, the
    elimination 5.3e-13 (contrast 1e4); 2.7e-14 elsewhere for both.
    """
    mp = pytest.importorskip("mpmath")
    from groundfield.coupling.layered_green import _spectral_amplitudes

    mp.mp.dps = 50
    lam = np.geomspace(1e-7, 2e3, 25)
    A, B, _, C = _spectral_amplitudes(lam, z_s, rho_1, rho_2, h_1)
    Ar, Br, Cr = _lapack_amplitudes(lam, z_s, rho_1, rho_2, h_1)
    worst_closed = worst_lapack = 0.0
    for k, lk in enumerate(lam):
        lm, zs, r1, r2, h = (mp.mpf(float(v)) for v in (lk, z_s, rho_1, rho_2, h_1))
        e = mp.e ** (-lm * h)
        s1, s2 = 1 / r1, 1 / r2
        if z_s <= h_1:
            d0 = (r1 / 2) * mp.e ** (-lm * zs)
            ph = (r1 / (2 * lm)) * mp.e ** (-lm * (h - zs))
            dph = -(r1 / 2) * mp.e ** (-lm * (h - zs))
            p2 = dp2 = mp.mpf(0)
        else:
            d0 = ph = dph = mp.mpf(0)
            p2 = (r2 / (2 * lm)) * mp.e ** (-lm * (zs - h))
            dp2 = (r2 / 2) * mp.e ** (-lm * (zs - h))
        M = mp.matrix([[lm * e, -lm, 0], [1, e, -1], [s1 * lm, -s1 * lm * e, s2 * lm]])
        x = mp.lu_solve(M, mp.matrix([-d0, p2 - ph, s2 * dp2 - s1 * dph]))
        ref = [float(x[0]), float(x[1]), float(x[2])]
        scale = sum(abs(v) for v in ref) + 1e-300
        worst_closed = max(
            worst_closed,
            max(abs(A[k] - ref[0]), abs(B[k] - ref[1]), abs(C[k] - ref[2])) / scale,
        )
        worst_lapack = max(
            worst_lapack,
            max(abs(Ar[k] - ref[0]), abs(Br[k] - ref[1]), abs(Cr[k] - ref[2])) / scale,
        )
    assert worst_closed < 2e-12
    assert worst_closed <= max(worst_lapack, 1e-13)


def _rod_through_the_interface() -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0))
    gf.create_electrode(w, "ring", name="ring", center=(0.0, 0.0, 0.6), radius=2.0)
    gf.create_electrode(w, "rod", name="rod", position=(2.0, 0.0, 0.6), length=6.0)
    gf.create_conductor(w, name="b", start="ring", end="rod", cross_section=50e-6)
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    return w


@pytest.mark.parametrize("kernel", ["point", "line"])
def test_cross_layer_solve_runs_no_linear_solve_per_node(monkeypatch, kernel) -> None:
    """Deterministic cost gate: no LAPACK solve in the spectral path.

    Before: two ``np.linalg.solve`` calls per depth group (layered and
    baseline amplitudes) plus the diagonal — 481 calls for the AP1 station
    at 10 % penetration.
    """
    w = _rod_through_the_interface()
    calls = []
    real_solve = np.linalg.solve

    def counting_solve(a, b):
        calls.append(np.shape(a))
        return real_solve(a, b)

    monkeypatch.setattr(np.linalg, "solve", counting_solve)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gf.Engine(segment_length=0.5, kernel=kernel).solve(w)
    assert res.metadata["series_used"] is False  # the cross-layer path ran
    assert not [s for s in calls if len(s) == 3 and s[-2:] == (3, 3)]
