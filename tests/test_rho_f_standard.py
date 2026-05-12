"""Tests for the standard-form rho-f fitter.

The fit form is

    Z(rho, f) = k1·rho + (k2 + j·k3)·f + (k4 + j·k5)·f·rho

— five real unknowns, linear LSQ in the real and imaginary
halves separately.
"""

from __future__ import annotations

import math

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
        k1=0.05, k2=1e-4, k3=2e-3, k4=2e-7, k5=5e-6,
        rms_error=0.0, rms_relative=0.0,
        sample_rho=np.array([100.0]), sample_f=np.array([50.0]),
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
        results.append(FieldResult(
            backend="image",
            frequencies=freqs,
            electrode_potentials={"g1": list(Z)},
            electrode_currents={"g1": list(I)},
            soil_resistivity=r,
        ))
    fit = rho_f_standard_from_results(results, rhos, "g1")
    for key, true_val in k_true.items():
        got = getattr(fit, key)
        assert abs(got - true_val) <= 1e-9 * (abs(true_val) + 1e-12)
