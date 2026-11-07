"""Tests for vector fitting and SymPy export.

Validates the implementation against synthetic ground-truth
impedances. The tests cover:

1. Recovery of a known rational $Z(s)$ to within a tight tolerance.
2. Real-pole and complex-pole fits.
3. SymPy export round-trip: evaluating the SymPy expression at the
   sample frequencies reproduces the fitted values.
4. ``rho_f_from_field_result`` extracts $U/I$ correctly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf
from groundfield.postprocess.vector_fitting import (
    fit_to_sympy,
    rho_f_from_field_result,
    vector_fit,
)


# ---------------------------------------------------------------------
# 1. Synthetic ground-truth recovery
# ---------------------------------------------------------------------


def _eval_known_Z(s, R_inf, L_inf, poles, residues):
    """Evaluate Z(s) = R_inf + L_inf*s + sum r_k/(s-p_k)."""
    Z = np.full_like(s, complex(R_inf), dtype=complex) + L_inf * s
    for p, r in zip(poles, residues, strict=True):
        Z = Z + r / (s - p)
    return Z


def test_vector_fit_recovers_single_real_pole() -> None:
    """A single real-pole RC response: Z(s) = R + r/(s+a)."""
    R_inf = 30.0
    poles = [-2.0 * math.pi * 100.0]  # corner at 100 Hz
    residues = [-2.0 * math.pi * 100.0 * 50.0]  # 50 Ω contribution at DC
    freqs = np.geomspace(1.0, 1000.0, 30)
    s = 2j * math.pi * freqs
    Z = _eval_known_Z(s, R_inf, 0.0, poles, residues)
    fit = vector_fit(
        freqs,
        Z,
        n_poles=1,
        n_iter=10,
        include_R_inf=True,
        include_L_inf=False,
        complex_poles=False,
    )
    assert fit.rms_error < 1e-6, f"RMS too large: {fit.rms_error}"
    assert abs(fit.R_inf - R_inf) / R_inf < 1e-3
    assert abs(fit.poles[0].real - poles[0]) / abs(poles[0]) < 1e-3


def test_vector_fit_recovers_two_real_poles() -> None:
    """Two-pole low-pass response."""
    R_inf = 10.0
    poles = [-2.0 * math.pi * 50.0, -2.0 * math.pi * 500.0]
    residues = [-100.0, -50.0]
    freqs = np.geomspace(1.0, 5000.0, 50)
    s = 2j * math.pi * freqs
    Z = _eval_known_Z(s, R_inf, 0.0, poles, residues)
    fit = vector_fit(
        freqs,
        Z,
        n_poles=2,
        n_iter=12,
        include_R_inf=True,
        complex_poles=False,
    )
    assert fit.rms_error < 1e-3, f"RMS too large: {fit.rms_error}"


def test_vector_fit_complex_pole_pair() -> None:
    """A damped resonant mode (complex-conjugate pole pair)."""
    R_inf = 20.0
    p = complex(-2.0 * math.pi * 30.0, 2.0 * math.pi * 200.0)
    r = complex(-1.0 * math.pi * 200.0, 0.0)
    poles = [p, p.conjugate()]
    residues = [r, r.conjugate()]
    freqs = np.geomspace(1.0, 1000.0, 60)
    s = 2j * math.pi * freqs
    Z = _eval_known_Z(s, R_inf, 0.0, poles, residues)
    fit = vector_fit(
        freqs,
        Z,
        n_poles=2,
        n_iter=15,
        include_R_inf=True,
        complex_poles=True,
    )
    # Allow looser bound for complex poles (harder to recover)
    assert fit.rms_error / np.mean(np.abs(Z)) < 0.05, (
        f"Relative RMS too large: {fit.rms_error / np.mean(np.abs(Z))}"
    )


# ---------------------------------------------------------------------
# 2. evaluate() round-trip
# ---------------------------------------------------------------------


def test_evaluate_reproduces_input_within_rms() -> None:
    """``VectorFitResult.evaluate`` at the fit frequencies returns
    values consistent with ``rms_error``."""
    R_inf = 30.0
    poles = [-2.0 * math.pi * 100.0]
    residues = [-2.0 * math.pi * 100.0 * 50.0]
    freqs = np.geomspace(1.0, 1000.0, 30)
    s = 2j * math.pi * freqs
    Z = _eval_known_Z(s, R_inf, 0.0, poles, residues)
    fit = vector_fit(
        freqs,
        Z,
        n_poles=1,
        n_iter=10,
        include_R_inf=True,
        complex_poles=False,
    )
    Z_eval = fit.evaluate(freqs)
    assert np.max(np.abs(Z_eval - Z)) < 10.0 * fit.rms_error


# ---------------------------------------------------------------------
# 3. SymPy export
# ---------------------------------------------------------------------


def test_fit_to_sympy_real_pole_round_trip() -> None:
    """Evaluating the SymPy expression at the fit frequencies gives
    the same values as ``VectorFitResult.evaluate``."""
    sp = pytest.importorskip("sympy")
    R_inf = 30.0
    poles = [-2.0 * math.pi * 100.0]
    residues = [-2.0 * math.pi * 100.0 * 50.0]
    freqs = np.geomspace(1.0, 1000.0, 20)
    s = 2j * math.pi * freqs
    Z = _eval_known_Z(s, R_inf, 0.0, poles, residues)
    fit = vector_fit(
        freqs,
        Z,
        n_poles=1,
        n_iter=10,
        include_R_inf=True,
        complex_poles=False,
    )
    expr = fit_to_sympy(fit)
    s_sym = sp.Symbol("s", complex=True)
    Z_eval_sympy = np.array(
        [complex(expr.subs(s_sym, complex(2j * math.pi * f)).evalf()) for f in freqs]
    )
    Z_eval_native = fit.evaluate(freqs)
    rel_err = np.max(np.abs(Z_eval_sympy - Z_eval_native) / np.abs(Z_eval_native))
    assert rel_err < 1e-4, f"SymPy/native round-trip mismatch: rel={rel_err}"


def test_fit_to_sympy_complex_pole_pair_real_expression() -> None:
    """For a complex-conjugate pole pair, ``fit_to_sympy`` must
    produce a *real-valued* expression (the imaginary parts cancel
    exactly)."""
    sp = pytest.importorskip("sympy")
    p = complex(-2.0 * math.pi * 30.0, 2.0 * math.pi * 200.0)
    r = complex(-50.0, 25.0)
    poles = [p, p.conjugate()]
    residues = [r, r.conjugate()]
    freqs = np.geomspace(1.0, 1000.0, 40)
    s = 2j * math.pi * freqs
    Z = _eval_known_Z(s, 0.0, 0.0, poles, residues)
    fit = vector_fit(
        freqs,
        Z,
        n_poles=2,
        n_iter=15,
        include_R_inf=True,
        complex_poles=True,
    )
    expr = fit_to_sympy(fit)
    # Substituting a real s should give a (numerically) real value.
    # The symmetrisation in fit_to_sympy averages the pair so the
    # expression is rigorously real-valued, but residual numerics
    # leave a tiny imag at evaluation; we accept ≤ 1 % relative.
    s_sym = sp.Symbol("s", complex=True)
    val = complex(expr.subs(s_sym, sp.Float(100.0)).evalf())
    assert abs(val.imag) < 0.01 * (abs(val.real) + 1e-9)


# ---------------------------------------------------------------------
# 4. rho_f_from_field_result wrapper
# ---------------------------------------------------------------------


def test_rho_f_from_field_result_extracts_correctly() -> None:
    """End-to-end: build a synthetic FieldResult-like object and
    verify that ``rho_f_from_field_result`` extracts U/I correctly."""
    from groundfield.solver.result import FieldResult

    freqs = list(np.geomspace(1.0, 1000.0, 20))
    s = 2j * math.pi * np.array(freqs)
    R_inf = 25.0
    poles = [-2.0 * math.pi * 80.0]
    residues = [-2.0 * math.pi * 80.0 * 30.0]
    U = _eval_known_Z(s, R_inf, 0.0, poles, residues)
    I = np.ones_like(s, dtype=complex)
    res = FieldResult(
        backend="image",
        frequencies=freqs,
        electrode_potentials={"g1": list(U)},
        electrode_currents={"g1": list(I)},
        soil_resistivity=100.0,
    )
    fit = rho_f_from_field_result(
        res,
        "g1",
        quantity="electrode",  # hand-built result: no source snapshot (10.19)
        n_poles=1,
        n_iter=10,
        include_R_inf=True,
        complex_poles=False,
    )
    assert fit.rms_error < 1e-3
    assert abs(fit.R_inf - R_inf) / R_inf < 1e-3


def test_rho_f_from_field_result_unknown_electrode_raises() -> None:
    """Unknown electrode name → KeyError."""
    from groundfield.solver.result import FieldResult

    res = FieldResult(
        backend="image",
        frequencies=[50.0, 100.0],
        electrode_potentials={"g1": [complex(1.0), complex(1.0)]},
        electrode_currents={"g1": [complex(1.0), complex(1.0)]},
        soil_resistivity=100.0,
    )
    with pytest.raises(KeyError, match="not in result"):
        rho_f_from_field_result(res, "nonexistent", quantity="electrode")


# ---------------------------------------------------------------------
# Audit 2026-07-08, WP-B4 — conjugate-pair real basis
# ---------------------------------------------------------------------


def test_complex_residues_are_recovered() -> None:
    """A synthetic resonant Z with genuinely complex residues must be
    reproduced to high accuracy.

    The historic real-stacked LS forced all residues real, silently
    removing the Im(r) degree of freedom of complex pole pairs — this
    synthetic target was unfittable then (audit finding N2/WP-B4).
    """
    p = complex(-80.0, 2 * np.pi * 300.0)
    r = complex(3.0, 40.0)  # deliberately large Im(r)
    p2 = -2 * np.pi * 40.0  # one real pole
    r2 = 25.0
    freqs = np.linspace(10.0, 1000.0, 60)
    s = 2j * np.pi * freqs
    Z = 5.0 + r / (s - p) + np.conj(r) / (s - np.conj(p)) + r2 / (s - p2)

    fit = vector_fit(freqs, Z, n_poles=3, n_iter=30)
    rel = fit.rms_error / float(np.mean(np.abs(Z)))
    assert rel < 1e-6
    # The fitted pair must carry a genuinely complex residue.
    pair_res = [
        x for x, q in zip(fit.residues, fit.poles, strict=True) if abs(q.imag) > 1.0
    ]
    assert pair_res and max(abs(x.imag) for x in pair_res) > 10.0


def test_convergence_diagnostics_exposed() -> None:
    """The relocation loop reports n_iter_used / pole_shift and
    breaks early once the poles settle."""
    freqs = np.linspace(10.0, 1000.0, 40)
    s = 2j * np.pi * freqs
    Z = 10.0 + (2000.0) / (s + 500.0)
    fit = vector_fit(freqs, Z, n_poles=1, n_iter=25)
    assert fit.n_iter_used >= 1
    assert fit.pole_shift < 1e-8  # converged (early break)
    assert fit.n_iter_used < 25  # did not run blind to the cap
    assert fit.poles[0].real == pytest.approx(-500.0, rel=1e-6)


def test_inverse_magnitude_weighting_redistributes_the_error() -> None:
    """Weighting trades absolute for relative accuracy — testable
    only with a genuine model error.

    The target carries two poles but is fitted with one, so the fit
    *must* compromise; the weighting then decides where the residual
    lands. Defining property (checked on this deliberately
    under-parameterised fit): ``inverse_magnitude`` minimises the
    *relative* error, ``uniform`` the *absolute* one. (An earlier
    version of this test compared two numerically perfect fits of an
    exactly representable target — a meaningless round-off race that
    flipped between BLAS builds.)
    """
    freqs = np.geomspace(1.0, 1000.0, 50)
    s = 2j * np.pi * freqs
    Z = 0.05 + 500.0 / (s + 30.0) + 800.0 / (s + 2.0 * np.pi * 800.0)
    fit_u = vector_fit(freqs, Z, n_poles=1, n_iter=25)
    fit_w = vector_fit(freqs, Z, n_poles=1, n_iter=25, weighting="inverse_magnitude")
    d_u = fit_u.evaluate(freqs) - Z
    d_w = fit_w.evaluate(freqs) - Z
    rel_u = float(np.sqrt(np.mean((np.abs(d_u) / np.abs(Z)) ** 2)))
    rel_w = float(np.sqrt(np.mean((np.abs(d_w) / np.abs(Z)) ** 2)))
    abs_u = float(np.sqrt(np.mean(np.abs(d_u) ** 2)))
    abs_w = float(np.sqrt(np.mean(np.abs(d_w) ** 2)))
    # Weighted fit wins on the relative metric (sandbox reference:
    # 0.130 vs 0.155), uniform wins on the absolute one (0.040 vs
    # 0.271) — generous factors keep the test BLAS-independent.
    assert rel_w < rel_u * 0.95
    assert abs_u < abs_w * 0.5


def test_dead_frequency_masked_with_warning() -> None:
    """Zero-current frequencies are masked (not fitted as Z = 0)."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        w, "rod", name="g1", position=(0, 0, 0.0), length=3.0, wire_radius=0.005
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(
        backend="image",
        segment_length=0.5,
        frequencies=[10.0, 50.0, 100.0, 500.0, 800.0, 1000.0],
    )
    res = eng.solve(w)
    # Forge one dead frequency in a copy of the result arrays.
    res.electrode_currents["g1"][2] = 0j
    # The forged dead column is an electrode current; the input impedance
    # (default since 10.19) divides by the source current instead.
    with pytest.warns(UserWarning, match="no current"):
        fit = rho_f_from_field_result(res, "g1", quantity="electrode", n_poles=1)
    assert fit.fit_frequencies.size == 5  # dead sample excluded


# ---------------------------------------------------------------------
# Review pass 10, 10.41: 0 Hz sample, quality warning, passivity check
# ---------------------------------------------------------------------

_F11 = np.array([0.0, 50, 150, 250, 350, 450, 550, 650, 750, 850, 1000.0])


def _rl_ladder(seed: int, noise: float) -> np.ndarray:
    """Passive RL-ladder data at the 11 AP1 frequencies (R5-04 family)."""
    rng = np.random.default_rng(seed)
    a = 2 * math.pi * np.geomspace(20.0, 5000.0, 4) * rng.uniform(0.5, 2.0, 4)
    rk = rng.uniform(0.05, 0.5, 4)
    s = 2j * math.pi * _F11
    z = 2.0 + sum(r * s / (s + ak) for r, ak in zip(rk, a, strict=True))
    return z * (1 + noise * (rng.standard_normal(11) + 1j * rng.standard_normal(11)))


@pytest.mark.finding("10.41")
@pytest.mark.parametrize("n_poles", [2, 3, 4])
def test_dc_sample_with_imaginary_part_does_not_collapse_the_fit(n_poles: int) -> None:
    """v0.15.0: 20/20 fits collapsed to Z ~ 0 (relative rms 0.99) or raised
    LinAlgError when Im Z(0) carried 1e-3 noise."""
    import warnings

    for seed in range(20):
        z = _rl_ladder(seed, 1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = vector_fit(_F11, z, n_poles=n_poles, n_iter=30)
        rel = fit.rms_error / np.abs(z).mean()
        assert rel < 0.02, (seed, rel)
        # The model is real at 0 Hz and reproduces Re Z(0).
        z0 = fit.evaluate([0.0])[0]
        assert z0.imag == pytest.approx(0.0, abs=1e-9)
        assert z0.real == pytest.approx(z[0].real, rel=0.02)


@pytest.mark.finding("10.41")
def test_tiny_imaginary_dc_part_is_projected_silently() -> None:
    """R5-04 regression: clean data, only Im Z(0) = 1e-6 Ohm."""
    import warnings

    z = _rl_ladder(3, 0.0)
    z[0] += 1e-6j
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = vector_fit(_F11, z, n_poles=4, n_iter=8)
    assert fit.rms_error / np.abs(z).mean() < 1e-3


@pytest.mark.finding("10.41")
def test_large_imaginary_dc_part_is_reported() -> None:
    from groundfield.postprocess.vector_fitting import VectorFitQualityWarning

    z = _rl_ladder(3, 0.0)
    z[0] += 0.05j  # 2.5 % of |Z(0)|: not a DC value
    with pytest.warns(VectorFitQualityWarning, match="0 Hz"):
        vector_fit(_F11, z, n_poles=4, n_iter=8)


@pytest.mark.finding("10.41")
def test_poor_fit_warns_and_reports_its_relative_rms() -> None:
    from groundfield.postprocess.vector_fitting import VectorFitQualityWarning

    f = np.geomspace(1.0, 1e4, 40)
    s = 2j * math.pi * f
    # Three lightly damped resonances cannot be fitted with one real pole.
    z = 1.0 + sum(
        100.0 * s / ((s + 30.0) ** 2 + (2 * math.pi * fr) ** 2)
        for fr in (50, 500, 5000)
    )
    with pytest.warns(VectorFitQualityWarning, match="relative rms"):
        fit = vector_fit(f, z, n_poles=1, n_iter=20, complex_poles=False)
    assert fit.rms_relative == pytest.approx(fit.rms_error / np.abs(z).mean())
    assert fit.rms_relative > 0.01


@pytest.mark.finding("10.41")
def test_invalid_samples_raise_value_error_not_linalg_error() -> None:
    z = _rl_ladder(0, 0.0)
    bad = z.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        vector_fit(_F11, bad, n_poles=2)
    with pytest.raises(ValueError, match="f > 0"):
        vector_fit([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], n_poles=1)


def _nonpassive_fit():
    """R5-05: seed 17, 0.1 % noise, Z(0) real, 4 poles -> Re Z < 0 at 5.5 Hz."""
    import warnings

    rng = np.random.default_rng(17)
    a = 2 * math.pi * np.geomspace(20.0, 5000.0, 4) * rng.uniform(0.5, 2.0, 4)
    rk = rng.uniform(0.05, 0.5, 4)
    s = 2j * math.pi * _F11
    z = 2.0 + sum(r * s / (s + ak) for r, ak in zip(rk, a, strict=True))
    noise = 1e-3 * (rng.standard_normal(11) + 1j * rng.standard_normal(11))
    noise[0] = noise[0].real
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vector_fit(_F11, z * (1 + noise), n_poles=4, n_iter=8)


@pytest.mark.finding("10.41")
def test_check_passivity_finds_an_in_band_violation_between_samples() -> None:
    from groundfield.postprocess.vector_fitting import check_passivity

    fit = _nonpassive_fit()
    report = check_passivity(fit, 1000.0)
    assert not report.passed
    assert report.min_re_Z < -0.01  # -0.019 Ohm in the review
    assert report.f_at_min == pytest.approx(5.508, abs=0.05)
    dense = np.linspace(0.0, 1000.0, 400001)
    assert report.min_re_Z <= fit.evaluate(dense).real.min() + 1e-9
    d = report.as_dict()
    assert d["passed"] is False and set(d) >= {"min_re_Z_Ohm", "f_at_min_Hz"}


@pytest.mark.finding("10.41")
def test_check_passivity_passes_a_passive_fit_and_checks_standard_forms() -> None:
    from groundfield.postprocess.rho_f_standard import fit_rho_f_standard
    from groundfield.postprocess.vector_fitting import check_passivity

    good = vector_fit(_F11, _rl_ladder(1, 0.0), n_poles=4, n_iter=8)
    rep = check_passivity(good)  # f_max defaults to the highest sample
    assert rep.passed and rep.min_re_Z > 1.9 and rep.f_max == 1000.0
    # Standard form with k4 < 0: Re Z = rho (0.05 - 2e-6 f) < 0 above 25 kHz.
    rho = np.repeat([30.0, 100.0, 300.0, 1000.0], 4)
    f = np.tile([0.0, 50.0, 500.0, 1000.0], 4)
    std = fit_rho_f_standard(rho, f, 0.05 * rho + 2e-3j * f + (-2e-6 + 3e-6j) * f * rho)
    assert check_passivity(std, 1000.0).passed
    bad = check_passivity(std, 30e3)
    assert not bad.passed and bad.f_at_min == 30e3 and bad.rho_at_min == 1000.0
