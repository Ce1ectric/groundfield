"""Tests for the standard-form rho-f fitter.

The fit form is

    Z(rho, f) = k1·rho + (k2 + j·k3)·f + (k4 + j·k5)·f·rho

— five real unknowns, linear LSQ in the real and imaginary
halves separately.
"""

from __future__ import annotations


import numpy as np
import pytest

from groundfield.postprocess.rho_f_standard import (
    RhoFStandardFit,
    fit_rho_f_standard,
    fit_to_sympy_standard,
    rho_f_standard_from_results,
)


def _synthetic_Z(rho, f, k1, k2, k3, k4, k5):
    rho = np.asarray(rho, dtype=float)
    f = np.asarray(f, dtype=float)
    return k1 * rho + (k2 + 1j * k3) * f + (k4 + 1j * k5) * f * rho


def test_recovers_exact_coefficients() -> None:
    """For noise-free synthetic data, all five coefficients are
    recovered to machine precision."""
    rhos = [50.0, 100.0, 200.0, 500.0, 1000.0]
    freqs = np.geomspace(1.0, 1000.0, 12)
    k_true = dict(k1=0.05, k2=1e-4, k3=2e-3, k4=2e-7, k5=5e-6)
    rho_arr, f_arr, Z_arr = [], [], []
    for r in rhos:
        for ff in freqs:
            rho_arr.append(r)
            f_arr.append(float(ff))
            Z_arr.append(_synthetic_Z(r, ff, **k_true))
    fit = fit_rho_f_standard(rho_arr, f_arr, Z_arr)
    for key, true_val in k_true.items():
        got = getattr(fit, key)
        assert abs(got - true_val) <= 1e-9 * (abs(true_val) + 1e-12), (
            f"{key}: expected {true_val}, got {got}"
        )
    assert fit.rms_error < 1e-9


def test_rejects_single_rho() -> None:
    """The LSQ is under-determined when all samples share a single
    rho value (k1 and k4 cannot be separated)."""
    freqs = np.geomspace(1.0, 1000.0, 10)
    rhos = [100.0] * len(freqs)
    Z = [_synthetic_Z(100.0, f, 0.05, 0.0, 0.0, 0.0, 1e-6) for f in freqs]
    with pytest.raises(ValueError, match="distinct rho"):
        fit_rho_f_standard(rhos, freqs, Z)


def test_rejects_single_frequency() -> None:
    """Same for a single-frequency set (k2 and k4 collapse)."""
    rhos = np.linspace(50, 1000, 10)
    freqs = [50.0] * len(rhos)
    Z = [_synthetic_Z(r, 50.0, 0.05, 0.0, 0.0, 0.0, 1e-6) for r in rhos]
    with pytest.raises(ValueError, match="distinct f"):
        fit_rho_f_standard(rhos, freqs, Z)


def test_evaluate_at_arbitrary_points() -> None:
    """``RhoFStandardFit.evaluate`` reproduces the closed form."""
    fit = RhoFStandardFit(
        k1=0.05,
        k2=1e-4,
        k3=2e-3,
        k4=2e-7,
        k5=5e-6,
        rms_error=0.0,
        rms_relative=0.0,
        sample_rho=np.array([100.0]),
        sample_f=np.array([50.0]),
        sample_Z=np.array([0.0 + 0.0j]),
    )
    Z = fit.evaluate(rho=200.0, f=100.0)
    Z_expected = 0.05 * 200 + (1e-4 + 2e-3j) * 100 + (2e-7 + 5e-6j) * 100 * 200
    assert abs(Z - Z_expected) < 1e-12


def test_fit_to_sympy_standard_round_trip() -> None:
    """The exported SymPy expression evaluates to the same Z(rho, f)
    as the fit."""
    sp = pytest.importorskip("sympy")
    rhos = [50.0, 100.0, 500.0]
    freqs = [1.0, 10.0, 100.0, 1000.0]
    rho_arr, f_arr, Z_arr = [], [], []
    for r in rhos:
        for ff in freqs:
            rho_arr.append(r)
            f_arr.append(ff)
            Z_arr.append(_synthetic_Z(r, ff, 0.04, 5e-5, 1e-3, 1e-7, 3e-6))
    fit = fit_rho_f_standard(rho_arr, f_arr, Z_arr)
    expr = fit_to_sympy_standard(fit)
    rho_sym = sp.Symbol("rho", real=True, positive=True)
    f_sym = sp.Symbol("f", real=True, positive=True)
    # Spot check at one (rho, f)
    val = complex(expr.subs({rho_sym: 200.0, f_sym: 50.0}).evalf())
    val_native = fit.evaluate(rho=200.0, f=50.0)
    assert abs(val - val_native) / abs(val_native) < 1e-5


def test_rho_f_standard_from_results_synthetic() -> None:
    """End-to-end: build minimal FieldResult-like objects with a
    synthetic Z(ρ, f) and verify that the wrapper recovers the
    true coefficients."""
    from groundfield.solver.result import FieldResult

    rhos = [100.0, 300.0, 1000.0]
    freqs = list(np.geomspace(1.0, 1000.0, 10))
    k_true = dict(k1=0.06, k2=1e-4, k3=2e-3, k4=1.5e-7, k5=4e-6)
    results = []
    for r in rhos:
        Z = _synthetic_Z(r, np.array(freqs), **k_true)
        I = np.ones_like(Z, dtype=complex)
        results.append(
            FieldResult(
                backend="image",
                frequencies=freqs,
                electrode_potentials={"g1": list(Z)},
                electrode_currents={"g1": list(I)},
                soil_resistivity=r,
            )
        )
    fit = rho_f_standard_from_results(results, rhos, "g1", quantity="electrode")
    for key, true_val in k_true.items():
        got = getattr(fit, key)
        assert abs(got - true_val) <= 1e-9 * (abs(true_val) + 1e-12)


# ---------------------------------------------------------------------
# Review pass 10, 10.27: constant term, relative weights, max_relative
# ---------------------------------------------------------------------


def _offset_samples():
    """Two-layer-like data: R(rho -> 0) = 1.88 Ohm stays finite (R5-06)."""
    rho = np.repeat([30.0, 100.0, 200.0, 500.0, 1000.0], 3)
    f = np.tile([0.0, 50.0, 1000.0], 5)
    return rho, f, 1.88 + 0.109 * rho + 0j


def _worst(fit, rho, f, Z) -> float:
    return float(np.max(np.abs(fit.evaluate(rho, f) - Z) / np.abs(Z)))


@pytest.mark.finding("10.27")
def test_constant_term_captures_a_finite_intercept() -> None:
    rho, f, Z = _offset_samples()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a good fit must not warn
        fit = fit_rho_f_standard(rho, f, Z)
    assert _worst(fit, rho, f, Z) < 1e-12
    assert fit.k0 == pytest.approx(1.88, rel=1e-12)
    assert fit.k1 == pytest.approx(0.109, rel=1e-12)
    assert fit.max_relative < 1e-12
    assert fit.evaluate(0.0, 0.0) == pytest.approx(1.88, rel=1e-12)


@pytest.mark.finding("10.27")
def test_two_layer_ring_sweep_is_fitted_within_five_percent() -> None:
    """R5-06: ring (r = 2 m, 0.8 m) in 2-layer soil rho_2 = 100, h_1 = 5 m,
    rho_1 = 30...1000: v0.15.0 was 27 % off at rho_1 = 30 while
    rms_relative read 2.6 %; constant term + relative weights: 3.3 %."""
    import warnings

    import groundfield as gf

    rhos = [30.0, 100.0, 200.0, 500.0, 1000.0]
    results = []
    for rho_1 in rhos:
        w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho_1, rho_2=100.0, h_1=5.0))
        gf.create_electrode(w, "ring", name="st", center=(0, 0, 0.8), radius=2.0)
        gf.create_source(w, attached_to="st", magnitude=1.0)
        eng = gf.create_engine(
            backend="image_2layer", segment_length=0.5, frequencies=[0.0, 50.0, 1000.0]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append(w.solve(eng))
    fit = rho_f_standard_from_results(results, rhos, "st")
    worst = _worst(fit, fit.sample_rho, fit.sample_f, fit.sample_Z)
    assert worst < 0.05
    assert fit.max_relative == pytest.approx(worst, rel=1e-12)


@pytest.mark.finding("10.27")
def test_large_pointwise_error_is_reported_and_warned() -> None:
    from groundfield.postprocess.rho_f_standard import RhoFStandardQualityWarning

    rho, f, Z = _offset_samples()
    with pytest.warns(RhoFStandardQualityWarning, match="max_relative"):
        fit = fit_rho_f_standard(rho, f, Z, k0=False, weights="uniform")
    assert fit.max_relative == pytest.approx(_worst(fit, rho, f, Z), rel=1e-12)
    assert fit.max_relative > 0.2  # 0.26 at rho = 30 (v0.15.0 form)
    assert fit.rms_relative < 0.05  # ... while the rms figure looks fine


@pytest.mark.finding("10.27")
def test_v0150_form_is_reproduced_on_request() -> None:
    """k0=False, weights='uniform' is the v0.15.0 fit, bit for bit."""
    rhos = [50.0, 100.0, 500.0]
    freqs = [1.0, 10.0, 100.0, 1000.0]
    rho = np.repeat(rhos, len(freqs))
    f = np.tile(freqs, len(rhos))
    Z = _synthetic_Z(rho, f, 0.04, 5e-5, 1e-3, 1e-7, 3e-6) * (
        1 + 1e-3 * np.cos(np.arange(rho.size))
    )
    fit = fit_rho_f_standard(rho, f, Z, k0=False, weights="uniform")
    sol_r, *_ = np.linalg.lstsq(np.column_stack([rho, f, f * rho]), Z.real, rcond=None)
    sol_i, *_ = np.linalg.lstsq(np.column_stack([f, f * rho]), Z.imag, rcond=None)
    assert (fit.k1, fit.k2, fit.k4) == tuple(float(x) for x in sol_r)
    assert (fit.k3, fit.k5) == tuple(float(x) for x in sol_i)
    assert fit.k0 == 0.0


@pytest.mark.finding("10.27")
def test_sympy_export_contains_the_constant_term() -> None:
    sp = pytest.importorskip("sympy")
    rho, f, Z = _offset_samples()
    Z = Z + (2e-4 + 3e-3j) * f + (1e-7 + 4e-6j) * f * rho
    fit = fit_rho_f_standard(rho, f, Z)
    expr = fit_to_sympy_standard(fit, decimals=15)
    rho_sym = sp.Symbol("rho", real=True, positive=True)
    f_sym = sp.Symbol("f", real=True, positive=True)
    for r, ff in ((30.0, 0.0), (250.0, 400.0)):
        val = complex(expr.subs({rho_sym: r, f_sym: ff}).evalf(20))
        assert val == pytest.approx(complex(fit.evaluate(r, ff)), rel=1e-12)
    assert fit.coefficients["k0"] == pytest.approx(1.88, rel=1e-9)


@pytest.mark.finding("10.27")
def test_fit_options_are_validated() -> None:
    rho, f, Z = _offset_samples()
    with pytest.raises(ValueError, match="weights"):
        fit_rho_f_standard(rho, f, Z, weights="inverse")
    Z0 = Z.copy()
    Z0[0] = 0.0
    with pytest.raises(ValueError, match="relative"):
        fit_rho_f_standard(rho, f, Z0)
    # Three distinct points cannot determine four real unknowns.
    with pytest.raises(ValueError, match="rank"):
        fit_rho_f_standard(
            [30.0, 30.0, 100.0, 100.0], [0.0, 50.0, 0.0, 0.0], [4.0, 4.0, 11.0, 11.0]
        )
