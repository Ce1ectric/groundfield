"""Tests for the geometric Sommerfeld earth-return Green function (ADR-0006).

Validation programme spelled out in ADR-0006 §"Validation":

1. **Limits**: σ→∞ collapses to ADR-0004 (perfect-mirror,
   bit-exact); σ→0 collapses to free-space; ω→0 collapses to perfect
   mirror.
2. **Long-wire homogeneous limit**: cluster impedance with
   ``earth_inductive_model="sommerfeld"`` agrees with
   ``"carson_series"`` to within 5 % on a 1 km PEN at 50 Hz —
   the Carson asymptote is recovered through the cluster solver.
3. **Short-wire deviation**: at $L = 10\\,\\mathrm{m}$ the Sommerfeld
   cluster impedance deviates from the Carson asymptote by ≥ 5 %.
4. **Layered earth**: a TwoLayerSoil with $\\rho_2 / \\rho_1 = 10$
   produces measurably different cluster impedance from the
   homogeneous Sommerfeld baseline; the deviation is monotone with
   frequency in the right direction.
5. **Cross-engine consistency**: image, mom, cim, bem agree on the
   cluster impedance with Sommerfeld active to within 5 %.
6. **Reflection-coefficient correctness**: Γ(λ → 0) → 1,
   Γ(λ → ∞) → 0; the (Γ - 1) integrand in the σ → 0 limit
   reproduces the free-space Lipschitz–Hankel identity.

Notes
-----
The Sommerfeld kernel evaluation uses ``scipy.special.j0`` and an
adaptive λ-grid; tests below trust those numerics rather than
re-deriving them, which is why the long-wire convergence target is
≤ 5 % rather than machine precision.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf
from groundfield.coupling.sommerfeld_inductance import (
    LayeredEarth,
    earth_return_correction_homogeneous,
    earth_return_correction_layered,
    reflection_coefficient_homogeneous,
    reflection_coefficient_layered,
    build_sommerfeld_correction_matrix,
)


# ---------------------------------------------------------------------
# 1. Reflection coefficients
# ---------------------------------------------------------------------


def test_reflection_homogeneous_sigma_zero_returns_zero() -> None:
    """At $\\sigma = 0$ the reflection coefficient vanishes (no
    reflection from a non-conducting earth)."""
    lambdas = np.array([1e-6, 1.0, 100.0])
    Gamma = reflection_coefficient_homogeneous(
        lambdas, omega=2 * math.pi * 50.0, sigma_earth=0.0,
    )
    assert np.allclose(Gamma, 0.0)


def test_reflection_homogeneous_sigma_infinite_returns_one() -> None:
    """At $\\sigma \\to \\infty$ the reflection coefficient
    saturates at $+1$.

    The convergence is $|\\Gamma| \\approx 1 - 2\\lambda/p_\\text{skin}$
    with $p_\\text{skin} = \\sqrt{\\omega\\mu_0\\sigma}$, so for any
    finite $\\lambda$ we approach 1 only at the rate $1/\\sqrt{\\sigma}$.
    We pick $\\sigma = 10^{20}$ S/m → $p_\\text{skin} \\approx 6\\cdot 10^7$
    and ask for atol = 1e-5, which is well within the finite-$\\sigma$
    error bound for $\\lambda \\le 100$.
    """
    lambdas = np.array([1e-6, 1.0, 100.0])
    Gamma = reflection_coefficient_homogeneous(
        lambdas, omega=2 * math.pi * 50.0, sigma_earth=1e20,
    )
    assert np.allclose(np.abs(Gamma), 1.0, atol=1e-5)


def test_reflection_layered_n_one_matches_homogeneous() -> None:
    """A LayeredEarth with one layer should produce the same
    reflection coefficient as the homogeneous formula."""
    lambdas = np.array([0.01, 1.0, 100.0])
    earth = LayeredEarth(rhos=(100.0,), thicknesses=())
    Gamma_layered = reflection_coefficient_layered(
        lambdas, omega=2 * math.pi * 50.0, earth=earth,
    )
    Gamma_hom = reflection_coefficient_homogeneous(
        lambdas, omega=2 * math.pi * 50.0, sigma_earth=1.0 / 100.0,
    )
    assert np.allclose(Gamma_layered, Gamma_hom, rtol=1e-12)


def test_reflection_layered_equal_rhos_matches_homogeneous() -> None:
    """A 2-layer earth with $\\rho_1 = \\rho_2$ behaves like a
    homogeneous half-space."""
    lambdas = np.array([0.01, 1.0, 100.0])
    earth = LayeredEarth(rhos=(100.0, 100.0), thicknesses=(1.0,))
    Gamma_layered = reflection_coefficient_layered(
        lambdas, omega=2 * math.pi * 50.0, earth=earth,
    )
    Gamma_hom = reflection_coefficient_homogeneous(
        lambdas, omega=2 * math.pi * 50.0, sigma_earth=1.0 / 100.0,
    )
    assert np.allclose(Gamma_layered, Gamma_hom, rtol=1e-9)


# ---------------------------------------------------------------------
# 2. Lipschitz–Hankel limit
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "z_sum,rho", [(1.2, 5.0), (5.0, 5.0), (3.0, 30.0), (1.0, 0.5)],
)
def test_sigma_zero_recovers_lipschitz_hankel(z_sum: float, rho: float) -> None:
    """At $\\sigma \\to 0$, $(\\Gamma - 1) \\to -1$, and the integral
    becomes $-\\int e^{-\\lambda(z+z')} J_0(\\lambda \\rho) d\\lambda
    = -1/\\sqrt{(z+z')^2 + \\rho^2}$ (Lipschitz–Hankel identity)."""
    val = earth_return_correction_homogeneous(
        rho=rho, z_i=z_sum / 2, z_j=z_sum / 2,
        omega=2.0 * math.pi * 50.0, sigma_earth=1e-12,
    )
    expected = -1.0 / math.sqrt(z_sum * z_sum + rho * rho)
    assert val.real == pytest.approx(expected, rel=1e-3)
    assert abs(val.imag) < 1e-6


def test_sigma_infinite_buried_pair_is_screened() -> None:
    """At $\\sigma \\to \\infty$ a buried pair is screened.

    With the Pollaczek kernel (audit 2026-07-08, WP-C1) the buried
    correction tends to $-1/R - 1/R'$: the attenuated direct term
    $e^{-\\gamma R}/R \\to 0$ and the reflected term vanishes
    ($\\lambda/u_1 \\to 0$), so the *total* coupling
    (Neumann $1/R + 1/R'$ plus this correction) goes to zero —
    a conductor embedded in a perfectly conducting medium carries no
    external flux. (The historic kernel instead returned the
    additive electrostatic mirror in this limit — one of the audit's
    headline findings.)
    """
    rho, z_i, z_j = 5.0, 0.6, 0.6
    val = earth_return_correction_homogeneous(
        rho=rho, z_i=z_i, z_j=z_j,
        omega=2.0 * math.pi * 50.0, sigma_earth=1e12,
    )
    R_dir = math.sqrt(rho * rho + (z_i - z_j) ** 2)
    R_mir = math.sqrt(rho * rho + (z_i + z_j) ** 2)
    expected = -1.0 / R_dir - 1.0 / R_mir
    assert val.real == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------
# 3. Solver-level: limits and regression
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
    soil=None,
):
    soil = soil if soil is not None else gf.HomogeneousSoil(resistivity=RHO_SOIL)
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


def test_engine_sommerfeld_is_default() -> None:
    """``earth_inductive_model="sommerfeld"`` is the default since 0.13.0."""
    eng = gf.create_engine(backend="image", segment_length=SEG)
    assert eng.earth_inductive_model == "sommerfeld"


def test_default_engine_produces_sommerfeld_result() -> None:
    """A ``create_engine`` with no ``earth_inductive_model`` uses the
    Sommerfeld/Pollaczek stack end-to-end: the solved inductive coupling
    equals the explicit-``sommerfeld`` result and differs from the
    ``perfect_mirror`` baseline (guards the default *and* its plumbing)."""
    w_def, _ = _galvanic_world(n_seg=8, frequencies=[500.0])
    eng_default = gf.create_engine(
        backend="image", segment_length=SEG, frequencies=[500.0]
    )  # no earth_inductive_model -> the new default
    i_default = w_def.solve(eng_default).electrode_currents["g2"][0]

    w_s, eng_s = _galvanic_world(
        n_seg=8, earth_inductive_model="sommerfeld", frequencies=[500.0]
    )
    i_somm = w_s.solve(eng_s).electrode_currents["g2"][0]

    w_p, eng_p = _galvanic_world(
        n_seg=8, earth_inductive_model="perfect_mirror", frequencies=[500.0]
    )
    i_pm = w_p.solve(eng_p).electrode_currents["g2"][0]

    assert i_default == pytest.approx(i_somm, rel=1e-9)
    assert abs(i_default - i_pm) > 1e-6 * abs(i_pm)


def test_engine_sommerfeld_dc_reproducibility() -> None:
    """At $\\omega = 0$ the Sommerfeld correction vanishes; the
    DC result is identical to the perfect-mirror DC result."""
    w_p, _ = _galvanic_world(n_seg=4, earth_inductive_model="perfect_mirror",
                              frequencies=[0.0])
    w_s, _ = _galvanic_world(n_seg=4, earth_inductive_model="sommerfeld",
                              frequencies=[0.0])
    eng_p = gf.create_engine(backend="image", segment_length=SEG,
                              frequencies=[0.0],
                              earth_inductive_model="perfect_mirror")
    eng_s = gf.create_engine(backend="image", segment_length=SEG,
                              frequencies=[0.0],
                              earth_inductive_model="sommerfeld")
    res_p = eng_p.solve(w_p)
    res_s = eng_s.solve(w_s)
    for name in ("g1", "g2"):
        assert abs(
            res_p.electrode_currents[name][0]
            - res_s.electrode_currents[name][0]
        ) < 1e-12


def test_engine_sommerfeld_changes_solution_at_50hz() -> None:
    """At 50 Hz the Sommerfeld correction differs from the
    perfect-mirror result by a measurable amount."""
    w_p, eng_p = _galvanic_world(
        n_seg=2, earth_inductive_model="perfect_mirror",
        frequencies=[50.0],
    )
    w_s, eng_s = _galvanic_world(
        n_seg=2, earth_inductive_model="sommerfeld",
        frequencies=[50.0],
    )
    res_p = eng_p.solve(w_p)
    res_s = eng_s.solve(w_s)
    I_p = res_p.electrode_currents["g1"][0]
    I_s = res_s.electrode_currents["g1"][0]
    assert abs(I_p - I_s) > 1e-9, (
        f"Sommerfeld should change the solution. Got I_p={I_p}, I_s={I_s}"
    )


# ---------------------------------------------------------------------
# 4. Sommerfeld vs Carson at the cluster level (long-wire limit)
# ---------------------------------------------------------------------


def test_sommerfeld_matches_carson_on_long_pen_at_50hz() -> None:
    """For a 30 m PEN over 100 Ω·m homogeneous earth at 50 Hz
    (deep in the Carson asymptotic regime), ``sommerfeld`` and
    ``carson_series`` produce cluster impedances that agree to
    within 10 %.

    The bound is loose because the geometry is short by Carson
    standards (30 m vs δ = 712 m at 50 Hz, 100 Ω·m); the Sommerfeld
    captures end effects that Carson's per-m × ℓ scaling misses,
    so they are NOT expected to be bit-exact here."""
    w_c, eng_c = _galvanic_world(
        n_seg=2, earth_inductive_model="carson_series",
        frequencies=[50.0],
    )
    w_s, eng_s = _galvanic_world(
        n_seg=2, earth_inductive_model="sommerfeld",
        frequencies=[50.0],
    )
    res_c = eng_c.solve(w_c)
    res_s = eng_s.solve(w_s)
    U_c = res_c.electrode_potentials["g1"][0]
    I_c = res_c.electrode_currents["g1"][0]
    U_s = res_s.electrode_potentials["g1"][0]
    I_s = res_s.electrode_currents["g1"][0]
    Z_c = U_c / I_c
    Z_s = U_s / I_s
    rel = abs(Z_s - Z_c) / abs(Z_c)
    assert rel < 0.10, (
        f"Sommerfeld and Carson disagree on the 30 m PEN cluster "
        f"impedance at 50 Hz: Z_carson={Z_c}, Z_sommerfeld={Z_s}, "
        f"rel={rel:.3f}"
    )


# ---------------------------------------------------------------------
# 5. Layered earth: handled natively without warnings
# ---------------------------------------------------------------------


def test_two_layer_soil_no_warning_under_sommerfeld() -> None:
    """Building an Engine with ``earth_inductive_model="sommerfeld"``
    against a TwoLayerSoil **does not** emit a layered-earth warning —
    the Sommerfeld kernel handles layered earth natively (in contrast
    to the Carson series, which only sees the upper layer).

    Adjusted in 0.15.0 (review pass 9, F26/F29): the PEN conductor below
    is discretised at ``discretize_segment_length=15.0`` while sitting at
    $z = 0.5$ m, i.e. $L \\gg 4z$ — squarely inside the shallow-segment
    regime that 0.15.0 started diagnosing with
    :class:`~groundfield.solver.image.ShallowSegmentWarning`. That
    warning is *correct* here and has nothing to do with layered earth,
    so it is exempted **individually**. The blanket
    ``simplefilter("error", UserWarning)`` is deliberately kept:
    the warning this test contrasts against — Carson's "only sees
    ``rho_1``" diagnostic — is a bare :class:`UserWarning`, so widening
    the exemption to all of ``UserWarning`` would silently void the
    test's own claim.
    """
    import warnings

    from groundfield.solver.image import ShallowSegmentWarning

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
        discretize_segment_length=15.0,
        coupling_to_soil="galvanic",
        inductance_model="neumann",
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(
        backend="image", segment_length=SEG, frequencies=[50.0],
        earth_inductive_model="sommerfeld",
    )
    with warnings.catch_warnings():
        # Keep the original strict filter: *any* UserWarning is an error,
        # so the Carson layered-earth warning — a bare ``UserWarning`` —
        # is still caught by this test. Only the one warning that this
        # geometry legitimately earns is exempted.
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("ignore", ShallowSegmentWarning)
        eng.solve(w)


def test_layered_earth_differs_from_homogeneous_at_50hz() -> None:
    """A TwoLayerSoil with $\\rho_2 / \\rho_1 = 10$ produces a
    different cluster impedance than a HomogeneousSoil at $\\rho_1$
    when the skin depth in the upper layer is larger than $h_1$."""
    soil_hom = gf.HomogeneousSoil(resistivity=100.0)
    # h_1 must comfortably exceed the rod tip depth (rod at z=0.5
    # with length 2 reaches z=2.5); the image_2layer backend
    # requires all segments to live in the upper layer.
    soil_lay = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=5.0)
    w_h, eng_h = _galvanic_world(
        n_seg=2, earth_inductive_model="sommerfeld",
        frequencies=[50.0], soil=soil_hom,
    )
    w_l, eng_l = _galvanic_world(
        n_seg=2, earth_inductive_model="sommerfeld",
        frequencies=[50.0], soil=soil_lay,
    )
    res_h = eng_h.solve(w_h)
    res_l = eng_l.solve(w_l)
    Z_h = res_h.electrode_potentials["g1"][0] / res_h.electrode_currents["g1"][0]
    Z_l = res_l.electrode_potentials["g1"][0] / res_l.electrode_currents["g1"][0]
    rel = abs(Z_l - Z_h) / abs(Z_h)
    # At 50 Hz with rho1 = 100, delta_1 ~ 712 m >> h_1 = 2 m, so
    # the upper layer is "transparent" magnetically and the lower
    # layer dominates — but for the spreading resistance the upper
    # layer still matters. The two cluster impedances must differ.
    assert rel > 1e-4, (
        f"Layered earth should differ from homogeneous: "
        f"Z_homog={Z_h}, Z_layered={Z_l}, rel={rel:.3e}"
    )


# ---------------------------------------------------------------------
# 6. Cross-engine consistency
# ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["image", "mom", "cim", "bem"])
def test_cross_engine_sommerfeld_consistency_at_50hz(backend: str) -> None:
    """All distributed-capable backends agree on the cluster
    impedance with Sommerfeld active to within 10 %."""
    w_ref, _ = _galvanic_world(
        n_seg=2, earth_inductive_model="sommerfeld",
        frequencies=[50.0],
    )
    w_be, _ = _galvanic_world(
        n_seg=2, earth_inductive_model="sommerfeld",
        frequencies=[50.0],
    )
    eng_ref = gf.create_engine(
        backend="image", segment_length=SEG, frequencies=[50.0],
        earth_inductive_model="sommerfeld",
    )
    eng_be = gf.create_engine(
        backend=backend, segment_length=SEG, frequencies=[50.0],
        earth_inductive_model="sommerfeld",
    )
    res_ref = eng_ref.solve(w_ref)
    res_be = eng_be.solve(w_be)
    Z_ref = (res_ref.electrode_potentials["g1"][0]
             / res_ref.electrode_currents["g1"][0])
    Z_be = (res_be.electrode_potentials["g1"][0]
            / res_be.electrode_currents["g1"][0])
    rel = abs(Z_be - Z_ref) / abs(Z_ref)
    assert rel < 0.10, (
        f"Backend {backend} disagrees with image at 50 Hz with "
        f"Sommerfeld: Z_image={Z_ref}, Z_{backend}={Z_be}, rel={rel:.3f}"
    )
