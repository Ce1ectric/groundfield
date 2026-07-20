"""Regression tests for the layered Pollaczek earth-return reference.

Drop into ``tests/test_pollaczek_reference.py``. Pins the layered
per-unit-length belab against the homogeneous Carson reference already in
the repo (:func:`groundfield.references.earth_return.carson_self_impedance`),
the two-layer limiting cases, and the input guards.
"""
import math

import pytest

from groundfield.references.earth_return import (
    MU_0,
    carson_mutual_impedance,
    carson_self_impedance,
)
from groundfield.references.pollaczek import (
    layered_effective_wavenumber,
    loop_impedance_conductor_earth,
    pollaczek_mutual_impedance,
    pollaczek_self_impedance,
)

# A representative buried PEN conductor (150 mm^2, 0.7 m deep).
GMR = 0.7788 * math.sqrt(150e-6 / math.pi)
DEPTH = 0.7


# ---------------------------------------------------------------------------
# Homogeneous limit: Pollaczek(rho, rho) == Carson
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rho", [30.0, 100.0, 300.0, 1000.0])
@pytest.mark.parametrize("freq", [50.0, 250.0, 550.0, 1000.0])
def test_homogeneous_reactance_matches_carson(rho, freq):
    """Reactance of the homogeneous Pollaczek limit matches Carson < 0.5 %."""
    z_poll = pollaczek_self_impedance(rho, rho, 10.0, freq, depth=DEPTH, gmr=GMR)
    z_cars = carson_self_impedance(rho, freq, gmr=GMR)
    assert z_poll.imag == pytest.approx(z_cars.imag, rel=5e-3)


@pytest.mark.parametrize("rho", [30.0, 100.0, 1000.0])
@pytest.mark.parametrize("freq", [50.0, 550.0, 1000.0])
def test_homogeneous_resistance_approaches_carson_from_below(rho, freq):
    """Earth-return resistance approaches omega*mu0/8 from below (Pollaczek is exact)."""
    z_poll = pollaczek_self_impedance(rho, rho, 10.0, freq, depth=DEPTH, gmr=GMR)
    r_carson = 2.0 * math.pi * freq * MU_0 / 8.0
    assert z_poll.real <= r_carson * (1.0 + 1e-6)       # never above the Carson constant
    assert z_poll.real == pytest.approx(r_carson, rel=2e-2)  # within 2 %


# ---------------------------------------------------------------------------
# Two-layer limiting cases and bracketing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rho_1,rho_2", [(100.0, 1000.0), (1000.0, 30.0)])
def test_thick_upper_layer_recovers_upper_resistivity(rho_1, rho_2):
    """h_1 -> infinity: the two-layer belab collapses onto the upper layer."""
    z_2l = pollaczek_self_impedance(rho_1, rho_2, 1e6, 50.0, depth=DEPTH, gmr=GMR)
    z_up = pollaczek_self_impedance(rho_1, rho_1, 10.0, 50.0, depth=DEPTH, gmr=GMR)
    assert z_2l.imag == pytest.approx(z_up.imag, rel=1e-3)


@pytest.mark.parametrize("rho_1,rho_2", [(100.0, 1000.0), (1000.0, 30.0)])
def test_vanishing_upper_layer_recovers_lower_resistivity(rho_1, rho_2):
    """h_1 -> 0: the two-layer belab collapses onto the lower layer."""
    z_2l = pollaczek_self_impedance(rho_1, rho_2, 1e-3, 50.0, depth=DEPTH, gmr=GMR)
    z_lo = pollaczek_self_impedance(rho_2, rho_2, 10.0, 50.0, depth=DEPTH, gmr=GMR)
    assert z_2l.imag == pytest.approx(z_lo.imag, rel=1e-3)


@pytest.mark.parametrize("rho_1,rho_2,h_1", [
    (1000.0, 30.0, 5.0), (30.0, 1000.0, 5.0), (100.0, 200.0, 10.0),
    (1000.0, 30.0, 30.0), (30.0, 1000.0, 30.0),
])
@pytest.mark.parametrize("freq", [50.0, 250.0, 1000.0])
def test_two_layer_reactance_within_carson_bracket(rho_1, rho_2, h_1, freq):
    """The layered reactance lies within [Carson(rho_1), Carson(rho_2)]."""
    x = pollaczek_self_impedance(rho_1, rho_2, h_1, freq, depth=DEPTH, gmr=GMR).imag
    lo, hi = sorted((carson_self_impedance(rho_1, freq, gmr=GMR).imag,
                     carson_self_impedance(rho_2, freq, gmr=GMR).imag))
    assert lo - 1e-12 <= x <= hi + 1e-12


def test_low_frequency_is_deep_layer_dominated():
    """Thin resistive topsoil over a conductive layer: the LF return is deep-layer dominated."""
    # rho_1=1000 (thin, 3 m) over rho_2=30: at 50 Hz the reactance sits near the
    # conductive lower layer, not the resistive top.
    x = pollaczek_self_impedance(1000.0, 30.0, 3.0, 50.0, depth=DEPTH, gmr=GMR).imag
    x_lo = carson_self_impedance(30.0, 50.0, gmr=GMR).imag
    x_up = carson_self_impedance(1000.0, 50.0, gmr=GMR).imag
    assert abs(x - x_lo) < abs(x - x_up)


# ---------------------------------------------------------------------------
# Effective wavenumber recursion
# ---------------------------------------------------------------------------
def test_effective_wavenumber_homogeneous_is_u1():
    omega = 2 * math.pi * 50.0
    ue = layered_effective_wavenumber(0.3, 200.0, 200.0, 5.0, omega)
    u1 = ((0.3**2) + 1j * omega * MU_0 / 200.0) ** 0.5
    assert ue == pytest.approx(u1)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(rho_1=-1.0, rho_2=100.0, h_1=5.0, frequency=50.0),
    dict(rho_1=100.0, rho_2=0.0, h_1=5.0, frequency=50.0),
    dict(rho_1=100.0, rho_2=100.0, h_1=5.0, frequency=0.0),
])
def test_rejects_nonpositive_inputs(kwargs):
    with pytest.raises(ValueError):
        pollaczek_self_impedance(**kwargs, depth=DEPTH, gmr=GMR)


def test_rejects_nonpositive_depth_and_gmr():
    with pytest.raises(ValueError):
        pollaczek_self_impedance(100.0, 100.0, 5.0, 50.0, depth=-0.1, gmr=GMR)
    with pytest.raises(ValueError):
        pollaczek_self_impedance(100.0, 100.0, 5.0, 50.0, depth=DEPTH, gmr=0.0)


# ---------------------------------------------------------------------------
# Conductor-earth loop impedance (internal R' + earth-return, ideal 0-Ohm
# termination) for real earthing conductors.
# ---------------------------------------------------------------------------
# IEC 60228 DC resistances (20 degC).
R_PEN = 0.206e-3    # NAYY-J PEN, 150 mm^2 Al -> 0.206 Ohm/km
GMR_PEN = GMR       # 0.7788*sqrt(A/pi): internal inductance folded into GMR
R_SCR = 0.727e-3    # NA2XS2Y Cu wire screen, 25 mm^2 -> 0.727 Ohm/km
GMR_SCR = 13.5e-3   # screen radius over the insulation (thin -> no internal L)
DEPTH_SCR = 1.0


@pytest.mark.parametrize("rho_1,rho_2,h_1", [(100.0, 100.0, 10.0), (1000.0, 30.0, 3.0)])
@pytest.mark.parametrize("freq", [50.0, 550.0, 1000.0])
def test_loop_is_internal_resistance_plus_earth_return(rho_1, rho_2, h_1, freq):
    """z'_loop = r_internal + Z'_earth-return exactly: the added part is a
    pure real series resistance, the reactance is the earth-return reactance."""
    ze = pollaczek_self_impedance(rho_1, rho_2, h_1, freq, depth=DEPTH, gmr=GMR_PEN)
    zl = loop_impedance_conductor_earth(
        rho_1, rho_2, h_1, freq, depth=DEPTH, gmr=GMR_PEN, r_internal=R_PEN
    )
    assert zl.real == pytest.approx(ze.real + R_PEN, rel=1e-12)
    assert zl.imag == pytest.approx(ze.imag, rel=1e-12)


def test_loop_real_conductors_at_power_frequency():
    """NAYY PEN and NA2XS2Y screen at 50 Hz / 100 Ohm.m: the loop resistance
    is the conductor's own R' plus the small (~0.049 Ohm/km) earth term and
    is internal-resistance dominated at power frequency; the reactance comes
    from the earth return; and the lower-resistance PEN loop is the more
    inductive (larger phase angle)."""
    r_earth = 2.0 * math.pi * 50.0 * MU_0 / 8.0     # ~4.93e-5 Ohm/m
    z_pen = loop_impedance_conductor_earth(
        100.0, 100.0, 10.0, 50.0, depth=DEPTH, gmr=GMR_PEN, r_internal=R_PEN
    )
    z_scr = loop_impedance_conductor_earth(
        100.0, 100.0, 10.0, 50.0, depth=DEPTH_SCR, gmr=GMR_SCR, r_internal=R_SCR
    )
    # loop resistance = internal R' + earth-return R' (~omega mu0/8)
    assert z_pen.real == pytest.approx(R_PEN + r_earth, rel=2e-3)
    assert z_scr.real == pytest.approx(R_SCR + r_earth, rel=2e-3)
    # at 50 Hz the conductor's own resistance dominates the real part
    assert R_PEN > 3.0 * r_earth and R_SCR > 10.0 * r_earth
    # both inductive; the low-R PEN loop has the larger phase angle
    assert z_pen.imag > 0.0 and z_scr.imag > 0.0
    assert math.atan2(z_pen.imag, z_pen.real) > math.atan2(z_scr.imag, z_scr.real)


def test_loop_two_layer_shifts_reactance_deep_layer_dominated():
    """A two-layer earth moves the loop *reactance* (the earth-return part,
    within the Carson bracket, deep-layer dominated at 50 Hz) while the
    internal-resistance part of R'_loop is unchanged."""
    kw = dict(depth=DEPTH, gmr=GMR_PEN, r_internal=R_PEN)
    z_hom = loop_impedance_conductor_earth(100.0, 100.0, 10.0, 50.0, **kw)
    z_res_over_cond = loop_impedance_conductor_earth(1000.0, 30.0, 3.0, 50.0, **kw)
    z_cond_over_res = loop_impedance_conductor_earth(30.0, 1000.0, 3.0, 50.0, **kw)
    # thin resistive topsoil over a conductive layer -> reactance below the
    # homogeneous case (deep conductive layer dominates); reverse case above.
    assert z_res_over_cond.imag < z_hom.imag < z_cond_over_res.imag
    # the internal resistance dominates and barely moves with the soil (only
    # the tiny earth-return resistance is soil-dependent).
    for z in (z_hom, z_res_over_cond, z_cond_over_res):
        assert z.real == pytest.approx(R_PEN, rel=0.3)


def test_loop_rejects_negative_internal_resistance():
    with pytest.raises(ValueError):
        loop_impedance_conductor_earth(
            100.0, 100.0, 5.0, 50.0, depth=DEPTH, gmr=GMR, r_internal=-1e-6
        )


# ---------------------------------------------------------------------------
# Layered mutual (coupling) impedance between two conductor-earth loops.
# ---------------------------------------------------------------------------
def _mutual_k0_homogeneous(rho, freq, x, h1, h2):
    """Exact homogeneous Pollaczek mutual via the K0 form — the independent
    reference for the layered Sunde mutual in the ``rho_1 == rho_2`` limit."""
    import numpy as np
    from scipy.integrate import quad as _quad
    from scipy.special import kv

    omega = 2 * math.pi * freq
    g2 = 1j * omega * MU_0 / rho
    g = np.sqrt(g2)
    d = math.hypot(x, h1 - h2)
    D = math.hypot(x, h1 + h2)

    def _f(lam, part):
        u = np.sqrt(lam * lam + g2)
        v = np.exp(-(h1 + h2) * u) / (lam + u)
        return v.real if part == "re" else v.imag

    J = (
        2.0 * _quad(lambda t: _f(t, "re") * math.cos(t * x), 0, np.inf, limit=400)[0]
        + 2j * _quad(lambda t: _f(t, "im") * math.cos(t * x), 0, np.inf, limit=400)[0]
    )
    return 1j * omega * MU_0 / (2 * math.pi) * (kv(0, g * d) - kv(0, g * D) + J)


@pytest.mark.parametrize(
    "rho,freq,x", [(100.0, 50.0, 10.0), (30.0, 550.0, 4.0), (1000.0, 50.0, 40.0)]
)
def test_mutual_matches_k0_reference_homogeneous(rho, freq, x):
    """The layered Sunde mutual reduces to the exact homogeneous K0-form
    Pollaczek mutual (reactance to < 1 % in the <= 1 kHz regime)."""
    z = pollaczek_mutual_impedance(
        rho, rho, 10.0, freq, depth_i=0.7, depth_j=0.7, separation=x
    )
    zk = _mutual_k0_homogeneous(rho, freq, x, 0.7, 0.7)
    assert z.imag == pytest.approx(zk.imag, rel=1e-2)


@pytest.mark.parametrize("rho", [30.0, 100.0, 1000.0])
@pytest.mark.parametrize("freq", [50.0, 550.0, 1000.0])
def test_mutual_matches_carson_mutual(rho, freq):
    """Homogeneous mutual reactance matches the Carson mutual reference."""
    z = pollaczek_mutual_impedance(
        rho, rho, 10.0, freq, depth_i=0.7, depth_j=0.7, separation=10.0
    )
    zc = carson_mutual_impedance(rho, freq, separation=10.0)
    assert z.imag == pytest.approx(zc.imag, rel=1e-2)


def test_mutual_reduces_to_self_as_separation_goes_to_gmr():
    """With ``depth_i == depth_j`` and ``separation -> GMR`` the mutual
    recovers the self impedance (the coincidence limit of the coupling)."""
    z_self = pollaczek_self_impedance(100.0, 100.0, 10.0, 50.0, depth=DEPTH, gmr=GMR)
    z_mut = pollaczek_mutual_impedance(
        100.0, 100.0, 10.0, 50.0, depth_i=DEPTH, depth_j=DEPTH, separation=GMR
    )
    assert z_mut.imag == pytest.approx(z_self.imag, rel=1e-6)


def test_mutual_below_self_and_decays_with_separation():
    """The mutual reactance is below the self and decreases monotonically
    with the horizontal separation."""
    z_self = pollaczek_self_impedance(100.0, 100.0, 10.0, 50.0, depth=DEPTH, gmr=GMR)
    xs = [2.0, 5.0, 10.0, 30.0, 100.0]
    xm = [
        pollaczek_mutual_impedance(
            100.0, 100.0, 10.0, 50.0, depth_i=DEPTH, depth_j=DEPTH, separation=x
        ).imag
        for x in xs
    ]
    assert all(m < z_self.imag for m in xm)
    assert all(a > b for a, b in zip(xm, xm[1:]))


@pytest.mark.parametrize("freq", [50.0, 1000.0])
def test_mutual_two_layer_within_bracket(freq):
    """Two-layer mutual reactance lies within the two homogeneous bounds,
    deep-layer dominated at low frequency."""
    kw = dict(depth_i=DEPTH, depth_j=DEPTH, separation=10.0)
    xt = pollaczek_mutual_impedance(1000.0, 30.0, 3.0, freq, **kw).imag
    x1 = pollaczek_mutual_impedance(1000.0, 1000.0, 3.0, freq, **kw).imag
    x2 = pollaczek_mutual_impedance(30.0, 30.0, 3.0, freq, **kw).imag
    lo, hi = sorted((x1, x2))
    assert lo - 1e-12 <= xt <= hi + 1e-12
    if freq <= 100.0:  # deep conductive layer dominates -> near the rho2 bound
        assert abs(xt - x2) < abs(xt - x1)


def test_mutual_conductor_above_conductor_separation_zero():
    """``separation == 0`` with different depths (one conductor above the
    other) is valid and gives a finite coupling."""
    z = pollaczek_mutual_impedance(
        100.0, 100.0, 10.0, 50.0, depth_i=0.7, depth_j=1.5, separation=0.0
    )
    assert z.imag > 0.0


def test_mutual_rejects_bad_inputs():
    with pytest.raises(ValueError):  # negative separation
        pollaczek_mutual_impedance(
            100.0, 100.0, 5.0, 50.0, depth_i=DEPTH, depth_j=DEPTH, separation=-1.0
        )
    with pytest.raises(ValueError):  # coincident conductors -> self case
        pollaczek_mutual_impedance(
            100.0, 100.0, 5.0, 50.0, depth_i=0.7, depth_j=0.7, separation=0.0
        )
    with pytest.raises(ValueError):  # non-positive resistivity
        pollaczek_mutual_impedance(
            -100.0, 100.0, 5.0, 50.0, depth_i=DEPTH, depth_j=DEPTH, separation=5.0
        )


def test_two_loop_impedance_matrix_two_pens():
    """Full 2x2 loop matrix of two parallel NAYY PENs ~4 m apart: self loop
    (with internal R') on the diagonal, earth-return coupling off-diagonal.
    The coupling is purely earth-return (real part ~ omega*mu0/8, no internal
    resistance) and is a sizeable fraction of the self reactance."""
    z11 = loop_impedance_conductor_earth(
        100.0, 100.0, 10.0, 50.0, depth=DEPTH, gmr=GMR, r_internal=R_PEN
    )
    z12 = pollaczek_mutual_impedance(
        100.0, 100.0, 10.0, 50.0, depth_i=DEPTH, depth_j=DEPTH, separation=4.0
    )
    r_earth = 2 * math.pi * 50.0 * MU_0 / 8.0
    # the coupling carries the earth-return resistance but no internal R'
    assert z12.real == pytest.approx(r_earth, rel=5e-2)
    assert z12.real < z11.real  # self has the extra internal resistance
    # coupling reactance is a real fraction of the self reactance at 4 m
    assert 0.2 < z12.imag / z11.imag < 0.9
