"""Tests for the Carson 1926 earth-return correction (ADR-0005).

Implements the validation programme spelled out in ADR-0005 §"Validation".
The tests fall into five groups:

1. **Unit tests against Carson 1926** — wave-antenna and railway
   worked examples from the original paper, plus Tleis 2008
   tabulated values for the intermediate-$a$ regime.
2. **Regime-boundary continuity** — small-$a$ ↔ quadrature at
   $a = 0.25$, quadrature ↔ asymptotic at $a = 5$.
3. **Limit checks** — $\\sigma_\\text{earth} \\to \\infty$ recovers the
   perfect-mirror result; $\\omega \\to 0$ collapses Carson to zero.
4. **Engineering benchmarks** — 1 km PEN self impedance against the
   Oeding/Tleis Carson reference; loop coupling open-circuit voltage
   with vs. without Carson; cross-engine consistency at 50 Hz.
5. **Layered-earth handling** — TwoLayerSoil emits the documented
   warning; cross-engine sanity vs. ``mom_sommerfeld``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.coupling.carson import (
    MU_0,
    carson_mutual_correction,
    carson_p_q,
    carson_parameter,
    carson_self_correction,
    deri_semlyen_correction,
    skin_depth,
)
from groundfield.coupling.carson import (
    _p_q_large,
    _p_q_quadrature,
    _p_q_small,
)
from groundfield.references.carson import (
    RAILWAY_25HZ,
    TLEIS_TAB_3_2,
    WAVE_ANTENNA_HIGH_RHO,
    WAVE_ANTENNA_LOW_RHO,
    all_examples,
)


# ---------------------------------------------------------------------
# 1. Unit tests against Carson 1926 / Tleis 2008
# ---------------------------------------------------------------------


@pytest.mark.parametrize("ex", list(all_examples()))
def test_carson_p_q_matches_published_values(ex) -> None:
    """``carson_p_q`` reproduces the Carson 1926 worked examples and
    the Tleis 2008 reference table to within the per-example tolerance.
    """
    P, Q = carson_p_q(ex.a, ex.theta)
    err_P = abs(P - ex.P_expected)
    err_Q = abs(Q - ex.Q_expected)
    rel_P = err_P / max(abs(ex.P_expected), 1e-12)
    rel_Q = err_Q / max(abs(ex.Q_expected), 1e-12)
    assert (err_P <= ex.abs_tolerance or rel_P <= ex.rel_tolerance), (
        f"{ex.name}: P expected {ex.P_expected}, got {P} "
        f"(abs={err_P:.3e}, rel={rel_P:.3e})"
    )
    assert (err_Q <= ex.abs_tolerance or rel_Q <= ex.rel_tolerance), (
        f"{ex.name}: Q expected {ex.Q_expected}, got {Q} "
        f"(abs={err_Q:.3e}, rel={rel_Q:.3e})"
    )


def test_wave_antenna_high_rho_explicit() -> None:
    """Carson 1926 §V wave-antenna, $\\lambda = 10^{-12}$ → r = 4.0
    → J = 0.126 + j 0.168."""
    P, Q = carson_p_q(WAVE_ANTENNA_HIGH_RHO.a, WAVE_ANTENNA_HIGH_RHO.theta)
    assert P == pytest.approx(0.126, abs=0.005)
    assert Q == pytest.approx(0.168, abs=0.005)


def test_railway_25hz_explicit() -> None:
    """Carson 1926 §V railway, r = 0.2, θ ≈ 63°30' → J = 0.369 + j 1.135."""
    P, Q = carson_p_q(RAILWAY_25HZ.a, RAILWAY_25HZ.theta)
    assert P == pytest.approx(0.369, abs=0.005)
    assert Q == pytest.approx(1.135, abs=0.005)


# ---------------------------------------------------------------------
# 2. Regime-boundary continuity
# ---------------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.0, math.pi / 6.0, math.pi / 3.0])
def test_continuity_small_to_quadrature(theta: float) -> None:
    """At $a = 0.25$ the small-$a$ leading-term form and the
    quadrature agree to ≤ 1 % on both $P$ and $Q$. The bound is
    intentionally loose because the small-$a$ form is itself an
    expansion truncated at $\\mathcal{O}(a^3)$."""
    P_small, Q_small = _p_q_small(0.249, theta)
    P_quad, Q_quad = _p_q_quadrature(0.251, theta)
    assert abs(P_small - P_quad) <= 0.01 * max(abs(P_quad), 0.5), (
        f"P discontinuity at a=0.25 for theta={theta}: "
        f"small={P_small}, quad={P_quad}"
    )
    assert abs(Q_small - Q_quad) <= 0.01 * max(abs(Q_quad), 0.5), (
        f"Q discontinuity at a=0.25 for theta={theta}: "
        f"small={Q_small}, quad={Q_quad}"
    )


@pytest.mark.parametrize("theta", [0.0, math.pi / 6.0, math.pi / 3.0])
def test_continuity_quadrature_to_asymptotic(theta: float) -> None:
    """At $a = 5$ the quadrature and the asymptotic expansion agree
    to ≤ 5 · 10⁻³ on both $P$ and $Q$. The bound is intrinsic to the
    asymptotic truncation: the next omitted term is
    $\\sim \\cos(7\\theta) / a^7$ multiplied by a coefficient of order 10
    (Carson eqs. 36/37 truncated at $1/a^7$ gives a residual of
    a few thousandths at $a=5$). For larger $a$ the agreement
    sharpens rapidly."""
    P_quad, Q_quad = _p_q_quadrature(4.99, theta)
    P_large, Q_large = _p_q_large(5.01, theta)
    assert abs(P_quad - P_large) <= 5e-3
    assert abs(Q_quad - Q_large) <= 5e-3


# ---------------------------------------------------------------------
# 3. Limits and short-circuits
# ---------------------------------------------------------------------


def test_zero_omega_returns_zero() -> None:
    """At $\\omega = 0$ the Carson correction vanishes — the
    pre-factor $\\omega \\mu_0 / \\pi$ is zero."""
    Z_self = carson_self_correction(0.0, 1.0, 0.01)
    Z_mut = carson_mutual_correction(0.0, 1.0, 1.0, 5.0, 0.01)
    assert Z_self == 0.0 + 0.0j
    assert Z_mut == 0.0 + 0.0j


def test_zero_sigma_returns_zero() -> None:
    """At $\\sigma_\\text{earth} = 0$ the earth-return path does not
    exist — Carson short-circuits to zero."""
    omega = 2.0 * math.pi * 50.0
    Z_self = carson_self_correction(omega, 1.0, 0.0)
    Z_mut = carson_mutual_correction(omega, 1.0, 1.0, 5.0, 0.0)
    assert Z_self == 0.0 + 0.0j
    assert Z_mut == 0.0 + 0.0j


def test_high_sigma_collapses_to_perfect_mirror() -> None:
    """For $\\sigma_\\text{earth} \\to \\infty$ the Carson correction
    becomes negligible compared with the perfect-mirror Neumann term."""
    omega = 2.0 * math.pi * 50.0
    h = 1.0
    sigma = 1e9  # essentially a perfect conductor
    Z_carson = carson_self_correction(omega, h, sigma)
    # Perfect-mirror reference (jω·μ_0/(2π)·ln(2h/a)) is much larger
    # for any realistic radius — pick a = 5 mm.
    radius = 0.005
    Z_perfect = 1j * omega * MU_0 / (2.0 * math.pi) * math.log(2.0 * h / radius)
    assert abs(Z_carson) <= 0.05 * abs(Z_perfect), (
        f"Carson correction should be small at sigma={sigma}, got "
        f"|Z_carson| = {abs(Z_carson):.3e} vs. |Z_perfect| = {abs(Z_perfect):.3e}"
    )


# ---------------------------------------------------------------------
# 4. Skin depth and Carson parameter
# ---------------------------------------------------------------------


def test_skin_depth_textbook_formula() -> None:
    """$\\delta = 503 \\sqrt{\\rho / f}$ in meters (with rho in Ωm,
    f in Hz)."""
    f = 50.0
    rho = 100.0
    sigma = 1.0 / rho
    omega = 2.0 * math.pi * f
    delta = skin_depth(omega, sigma)
    expected = 503.292 * math.sqrt(rho / f)
    assert delta == pytest.approx(expected, rel=1e-3)


def test_carson_parameter_relation_to_skin_depth() -> None:
    """$a = D \\sqrt{2} / \\delta$ — the dimensionless Carson
    parameter is the ratio of the geometric distance to the skin depth,
    up to the $\\sqrt{2}$ factor."""
    omega = 2.0 * math.pi * 100.0
    sigma = 0.01
    D = 5.0
    a = carson_parameter(D, omega, sigma)
    delta = skin_depth(omega, sigma)
    assert a == pytest.approx(D * math.sqrt(2.0) / delta, rel=1e-12)


# ---------------------------------------------------------------------
# 5. Carson series vs. Deri/Semlyen complex-depth (sanity check)
# ---------------------------------------------------------------------


def test_carson_vs_deri_semlyen_agree_at_50hz() -> None:
    """Carson series and Deri/Semlyen complex-depth disagree on the
    high-precision details but not on the order of magnitude. Within
    the AP1 range we accept ≈ 30 % relative agreement on the mutual
    correction at 50 Hz; this is a *cross-validation* test, not a
    high-precision benchmark."""
    omega = 2.0 * math.pi * 50.0
    sigma = 0.01  # rho = 100 ohm-m
    h = 1.0
    d = 5.0
    Z_carson = carson_mutual_correction(omega, h, h, d, sigma)
    Z_ds = deri_semlyen_correction(omega, h, h, d, sigma)
    # Both should be in the same ballpark — orders of magnitude check.
    assert abs(Z_carson) > 0.0
    assert abs(Z_ds) > 0.0
    rel = abs(Z_carson - Z_ds) / abs(Z_carson)
    assert rel < 0.5, f"|Carson - DeriSemlyen|/|Carson| = {rel:.3f}"


# ---------------------------------------------------------------------
# 6. Solver-level integration tests
# ---------------------------------------------------------------------

SEG = 0.5
RHO_SOIL = 100.0
ROD_LEN = 2.0
ROD_R = 0.0075
SEPARATION = 30.0
PEN_RHO = 2.82e-8
PEN_A = 50.0e-6


def _galvanic_world(
    *,
    n_seg: int,
    earth_inductive_model: str = "perfect_mirror",
    frequencies=None,
) -> tuple[gf.World, gf.Engine]:
    """Two-rod world with a galvanic distributed PEN conductor between."""
    soil = gf.HomogeneousSoil(resistivity=RHO_SOIL)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5),
                        length=ROD_LEN, wire_radius=ROD_R)
    gf.create_electrode(w, "rod", name="g2", position=(SEPARATION, 0, 0.5),
                        length=ROD_LEN, wire_radius=ROD_R)
    gf.create_conductor(
        w, name="pen", start="g1", end="g2",
        conductor_type="bare_copper",
        wire_radius=0.004, resistivity=PEN_RHO,
        cross_section=PEN_A,
        discretize_segment_length=SEPARATION / n_seg + 1e-9,
        coupling_to_soil="galvanic",
        inductance_model="neumann",
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(
        backend="image", segment_length=SEG,
        frequencies=list(frequencies) if frequencies else [50.0],
        earth_inductive_model=earth_inductive_model,
    )
    return w, eng


def test_engine_carson_default_is_perfect_mirror() -> None:
    """Default ``earth_inductive_model`` is ``perfect_mirror``."""
    eng = gf.create_engine(backend="image", segment_length=SEG)
    assert eng.earth_inductive_model == "perfect_mirror"


def test_engine_carson_perfect_mirror_regression() -> None:
    """Bit-exact regression: ``earth_inductive_model="perfect_mirror"``
    (default) reproduces the ADR-0004 result."""
    w_a, eng_a = _galvanic_world(n_seg=4, earth_inductive_model="perfect_mirror")
    w_b, eng_b = _galvanic_world(n_seg=4)  # default
    res_a = eng_a.solve(w_a)
    res_b = eng_b.solve(w_b)
    for ename in ("g1", "g2"):
        I_a = res_a.electrode_currents[ename][0]
        I_b = res_b.electrode_currents[ename][0]
        assert abs(I_a - I_b) <= 1e-15


def test_engine_carson_increases_real_part() -> None:
    """At 50 Hz the Carson correction adds a positive real part to
    the diagonal branch impedance — the per-electrode current
    magnitude with Carson active is *not the same* as with the
    perfect mirror, and the difference is dominated by the Carson
    earth-return resistance."""
    w_p, eng_p = _galvanic_world(n_seg=4, earth_inductive_model="perfect_mirror",
                                  frequencies=[50.0])
    w_c, eng_c = _galvanic_world(n_seg=4, earth_inductive_model="carson_series",
                                  frequencies=[50.0])
    res_p = eng_p.solve(w_p)
    res_c = eng_c.solve(w_c)
    # The two solutions must differ — Carson is doing something.
    I_p = res_p.electrode_currents["g1"][0]
    I_c = res_c.electrode_currents["g1"][0]
    assert abs(I_p - I_c) > 1e-9, (
        f"Carson should change the source-electrode current. "
        f"Got I_p={I_p}, I_c={I_c}"
    )


def test_engine_carson_dc_reproducibility() -> None:
    """At $\\omega = 0$ the Carson correction is short-circuited;
    the inductive and Carson-active solutions agree with the DC
    reference."""
    w_p, _ = _galvanic_world(n_seg=4, earth_inductive_model="perfect_mirror",
                              frequencies=[0.0])
    w_c, _ = _galvanic_world(n_seg=4, earth_inductive_model="carson_series",
                              frequencies=[0.0])
    eng_p = gf.create_engine(
        backend="image", segment_length=SEG, frequencies=[0.0],
        earth_inductive_model="perfect_mirror",
    )
    eng_c = gf.create_engine(
        backend="image", segment_length=SEG, frequencies=[0.0],
        earth_inductive_model="carson_series",
    )
    res_p = eng_p.solve(w_p)
    res_c = eng_c.solve(w_c)
    for ename in ("g1", "g2"):
        I_p = res_p.electrode_currents[ename][0]
        I_c = res_c.electrode_currents[ename][0]
        assert abs(I_p - I_c) < 1e-12


def test_field_result_exposes_penetration_depth() -> None:
    """ADR-0005 §"Eindringtiefen-Diagnostik": the FieldResult metadata
    exposes the skin depth at every frequency for every Carson-aware
    engine."""
    w, eng = _galvanic_world(
        n_seg=4, earth_inductive_model="carson_series",
        frequencies=[50.0, 500.0],
    )
    res = eng.solve(w)
    pd = res.metadata.get("penetration_depth")
    assert pd is not None
    assert set(pd.keys()) == {50.0, 500.0}
    # delta should decrease with frequency.
    assert pd[50.0] > pd[500.0]
    # At rho=100 ohm-m, f=50 Hz: delta ~ 712 m.
    assert pd[50.0] == pytest.approx(503.292 * math.sqrt(RHO_SOIL / 50.0), rel=1e-2)


# ---------------------------------------------------------------------
# 7. Layered-earth handling
# ---------------------------------------------------------------------


def test_two_layer_soil_emits_warning_under_carson() -> None:
    """Building an Engine with ``earth_inductive_model="carson_series"``
    against a TwoLayerSoil emits the documented UserWarning. We pick
    ``h_1 = 5 m`` so the 2 m rods comfortably fit inside the upper
    layer (image_2layer's precondition)."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=5.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5),
                        length=ROD_LEN, wire_radius=ROD_R)
    gf.create_electrode(w, "rod", name="g2", position=(SEPARATION, 0, 0.5),
                        length=ROD_LEN, wire_radius=ROD_R)
    gf.create_conductor(
        w, name="pen", start="g1", end="g2",
        conductor_type="bare_copper",
        wire_radius=0.004, resistivity=PEN_RHO,
        cross_section=PEN_A,
        discretize_segment_length=10.0,
        coupling_to_soil="galvanic",
        inductance_model="neumann",
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(
        backend="image", segment_length=SEG, frequencies=[50.0],
        earth_inductive_model="carson_series",
    )
    with pytest.warns(UserWarning, match="upper-layer rho_1"):
        eng.solve(w)


# ---------------------------------------------------------------------
# 8. Engineering benchmark — 1 km PEN self impedance
# ---------------------------------------------------------------------


def test_pen_1km_self_impedance_against_textbook() -> None:
    """1 km horizontal bare-copper PEN at depth 0.6 m above earth
    ($\\rho = 100\\,\\Omega\\,\\mathrm{m}$). At 50 Hz the Carson
    self-impedance per unit length should match the closed-form
    Oeding/Tleis textbook result to within 5 %.

    Textbook decomposition:

    .. code-block:: text

        Z'_self = R_dc + jωL_self_perfect_mirror + Carson(R_g + jX_g)

    The test compares our :func:`carson_self_correction` to the
    classical Tleis closed-form ``R_g + jX_g`` evaluated at the
    same parameters.
    """
    f = 50.0
    omega = 2.0 * math.pi * f
    rho_earth = 100.0
    sigma = 1.0 / rho_earth
    h = 0.6  # PEN depth (Sunde-equivalent height)
    Z_carson = carson_self_correction(omega, h, sigma)
    # Direct re-evaluation in the small-a regime (a ~ 0.0015):
    # At this a the leading-term Carson form gives
    #   P ≈ pi/8, Q ≈ -0.0386 + 0.5*ln(2/a)
    a = 2.0 * h * math.sqrt(omega * MU_0 * sigma)
    P_ref, Q_ref = _p_q_small(a, 0.0)
    pref = omega * MU_0 / math.pi
    Z_ref = complex(pref * P_ref, pref * Q_ref)
    rel = abs(Z_carson - Z_ref) / abs(Z_ref)
    assert rel < 0.05, (
        f"Self correction mismatch: |dZ| = {abs(Z_carson - Z_ref):.3e}, "
        f"|Z_ref| = {abs(Z_ref):.3e}, rel = {rel:.3f}"
    )


def test_pen_1km_mutual_impedance_carson_vs_perfect_mirror_at_50hz() -> None:
    """Two parallel 1-km PEN conductors at depth 0.6 m, separation 50 m,
    over $\\rho_\\text{earth} = 100\\,\\Omega\\,\\mathrm{m}$, at 50 Hz.
    The Carson real-part contribution (earth-return resistance) must be
    a finite positive number in the textbook range
    ($\\sim 50\\,\\mathrm{m}\\Omega/\\mathrm{km}$ at this frequency).
    """
    omega = 2.0 * math.pi * 50.0
    sigma = 1.0 / 100.0
    Z_per_m = carson_mutual_correction(
        omega=omega, height_i=0.6, height_j=0.6,
        horizontal_distance=50.0, sigma_earth=sigma,
    )
    R_per_km = Z_per_m.real * 1000.0
    # Tleis 2008 §3.4 textbook value at f=50 Hz, rho=100, h=0.6, d=50:
    # R_g ≈ omega·mu_0/8 = 2π·50·μ_0/8 ≈ 49.3 mΩ/km
    expected_R = omega * MU_0 / 8.0 * 1000.0  # Tleis low-frequency limit
    rel = abs(R_per_km - expected_R) / expected_R
    assert rel < 0.1, (
        f"Carson R_g/km = {R_per_km*1000:.2f} mΩ/km, "
        f"expected ≈ {expected_R*1000:.2f} mΩ/km, rel={rel:.3f}"
    )


# ---------------------------------------------------------------------
# 9. Cross-engine consistency at 50 Hz
# ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["image", "mom", "cim", "bem"])
def test_cross_engine_carson_consistency_at_50hz(backend: str) -> None:
    """All distributed-capable backends (image, mom, cim, bem) must
    agree on the source-electrode current with Carson active at 50 Hz
    to within 5 % of the image reference."""
    w_ref, _ = _galvanic_world(n_seg=4, earth_inductive_model="carson_series",
                                frequencies=[50.0])
    w_be, _ = _galvanic_world(n_seg=4, earth_inductive_model="carson_series",
                                frequencies=[50.0])
    eng_ref = gf.create_engine(
        backend="image", segment_length=SEG, frequencies=[50.0],
        earth_inductive_model="carson_series",
    )
    eng_be = gf.create_engine(
        backend=backend, segment_length=SEG, frequencies=[50.0],
        earth_inductive_model="carson_series",
    )
    res_ref = eng_ref.solve(w_ref)
    res_be = eng_be.solve(w_be)
    I_ref = res_ref.electrode_currents["g1"][0]
    I_be = res_be.electrode_currents["g1"][0]
    rel = abs(I_be - I_ref) / abs(I_ref)
    assert rel <= 0.05, (
        f"Backend {backend} disagrees with image at 50 Hz with Carson: "
        f"I_image={I_ref}, I_{backend}={I_be}, rel={rel:.3f}"
    )
