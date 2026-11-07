"""One passivity check for rho-f fits (review pass 10, 10.40).

A grounding impedance is the driving-point impedance of a passive
one-port, so ``Re Z(j 2 pi f) >= 0`` at every frequency. Before this
release the package carried two checks with two definitions of the band:
``check_passivity`` (0 Hz to the highest fitted frequency, uniform grid)
and the BusType export (0 Hz to twice that, pole-adapted grid, between
the samples). The export now calls ``check_passivity``'s implementation,
evaluated on the formula it writes, and both use one band: 0 Hz to
2 x the highest fitted frequency, plus the midpoints between samples and
points around every pole.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from groundfield.io.groundinsight import (
    NonPassiveFitError,
    NonPassiveFitWarning,
    to_bustype_dict,
)
from groundfield.postprocess import vector_fitting
from groundfield.postprocess.rho_f_standard import RhoFStandardFit, fit_rho_f_standard
from groundfield.postprocess.vector_fitting import (
    VectorFitResult,
    check_passivity,
    vector_fit,
)

_F_AP1 = np.array([0.0, 50, 150, 250, 350, 450, 550, 650, 750, 850, 1000.0])
_EXPORT_KEYS = {
    "passed",
    "min_re_Z_Ohm",
    "f_at_min_Hz",
    "rho_at_min_Ohm_m",
    "f_checked_Hz",
    "rho_checked_Ohm_m",
    "n_points",
}


def _vf(poles, residues, *, R_inf: float, freqs=_F_AP1) -> VectorFitResult:
    return VectorFitResult(
        poles=np.asarray(poles, dtype=complex),
        residues=np.asarray(residues, dtype=complex),
        R_inf=R_inf,
        L_inf=0.0,
        rms_error=0.0,
        fit_frequencies=np.asarray(freqs, dtype=float),
        fit_values=np.zeros(len(freqs), dtype=complex),
    )


def _resonance_above_the_samples() -> VectorFitResult:
    """Passive at every sample (<= 1 kHz), Re Z = -1 Ohm at 1.5 kHz.

    A lightly damped pole pair at 1.5 kHz with a negative real residue
    pulls Re Z from 2 Ohm to about -1 Ohm at the resonance; at the
    samples (>= 500 Hz away) it contributes less than 0.12 Ohm.
    """
    alpha, beta = 2 * math.pi * 20.0, 2 * math.pi * 1500.0
    p = complex(-alpha, beta)
    r = complex(-3.0 * alpha, 0.0)
    return _vf([p, p.conjugate()], [r, r.conjugate()], R_inf=2.0)


def _seed17() -> VectorFitResult:
    """Reviewer R5's seed-17 4-pole fit: Re Z = -0.0194 Ohm at 5.507 Hz."""
    pole = complex(-2.218572357914484, 34.821424546873004)
    residue = complex(-4.552790445343534, 0.9021603844867948)
    return _vf(
        [pole, pole.conjugate(), -8414.047102128188, -524.2350826056921],
        [residue, residue.conjugate(), -3108.4886146284425, -159.2253268178532],
        R_inf=2.739623036917992,
    )


def _passive_ladder() -> VectorFitResult:
    rng = np.random.default_rng(1)
    a = 2 * math.pi * np.geomspace(20.0, 5000.0, 4) * rng.uniform(0.5, 2.0, 4)
    rk = rng.uniform(0.05, 0.5, 4)
    s = 2j * math.pi * _F_AP1
    z = 2.0 + sum(r * s / (s + ak) for r, ak in zip(rk, a, strict=True))
    return vector_fit(_F_AP1, z, n_poles=4, n_iter=8)


# ---------------------------------------------------------------------
# One band
# ---------------------------------------------------------------------


@pytest.mark.finding("10.40")
def test_default_band_is_twice_the_highest_fitted_frequency() -> None:
    assert check_passivity(_passive_ladder()).f_max == 2000.0
    rho = np.repeat([30.0, 100.0, 300.0, 1000.0], 4)
    f = np.tile([0.0, 50.0, 500.0, 1000.0], 4)
    std = fit_rho_f_standard(rho, f, 0.05 * rho + 2e-3j * f)
    report = check_passivity(std)
    assert report.f_max == 2000.0
    assert report.rho_range == (30.0, 1000.0)


@pytest.mark.finding("10.40")
def test_violation_between_the_highest_sample_and_twice_it_is_found() -> None:
    """The base's ``check_passivity`` stopped at the highest sample (1 kHz)
    and passed this fit; the export flagged it. Both now agree."""
    fit = _resonance_above_the_samples()
    assert min(fit.evaluate(_F_AP1).real) > 1.8  # passive at every sample
    report = check_passivity(fit)
    assert not report.passed
    assert report.min_re_Z == pytest.approx(-1.0, abs=0.02)
    assert report.f_at_min == pytest.approx(1500.0, rel=1e-3)
    dense = np.linspace(0.0, 2000.0, 400001)
    assert report.min_re_Z <= fit.evaluate(dense).real.min() + 1e-9
    with pytest.warns(NonPassiveFitWarning, match="above the fitted band 0 to 1000 Hz"):
        payload = to_bustype_dict(
            fit, name="res", system_type="TN", voltage_level=0.4, rho_at_fit=100.0
        )
    exported = payload["metadata"]["passivity"]
    assert exported["min_re_Z_Ohm"] == pytest.approx(report.min_re_Z, abs=1e-9)
    assert exported["f_at_min_Hz"] == pytest.approx(report.f_at_min, rel=1e-9)


def test_explicit_band_is_honoured() -> None:
    fit = _resonance_above_the_samples()
    assert check_passivity(fit, 1000.0).passed
    assert check_passivity(fit, 1000.0).f_max == 1000.0
    with pytest.raises(ValueError, match="f_max"):
        check_passivity(fit, 0.0)
    with pytest.raises(ValueError, match="n must be"):
        check_passivity(fit, n=1)


# ---------------------------------------------------------------------
# One implementation, one report
# ---------------------------------------------------------------------


@pytest.mark.finding("10.40")
def test_export_runs_the_check_passivity_implementation(monkeypatch) -> None:
    calls: list[dict] = []
    real = vector_fitting._check_passivity

    def spy(fit, **kwargs):
        calls.append(kwargs)
        return real(fit, **kwargs)

    monkeypatch.setattr(vector_fitting, "_check_passivity", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NonPassiveFitWarning)
        to_bustype_dict(
            _seed17(), name="s17", system_type="TN", voltage_level=0.4, rho_at_fit=100.0
        )
    assert len(calls) == 1
    assert callable(calls[0]["evaluate"])  # the exported formula is checked
    assert calls[0]["rho_at_fit"] == 100.0


@pytest.mark.finding("10.40")
def test_report_dict_is_the_export_metadata_and_strict_json() -> None:
    fit = _seed17()
    report = check_passivity(fit)
    d = report.as_dict()
    assert set(d) == _EXPORT_KEYS
    assert d["f_checked_Hz"] == [0.0, 2000.0]
    assert d["rho_at_min_Ohm_m"] is None and d["rho_checked_Ohm_m"] is None
    assert d["n_points"] > 2000
    json.dumps(d, allow_nan=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NonPassiveFitWarning)
        exported = to_bustype_dict(
            fit, name="s17", system_type="TN", voltage_level=0.4, rho_at_fit=100.0
        )["metadata"]["passivity"]
    assert set(exported) == _EXPORT_KEYS
    assert exported["passed"] is False and report.passed is False
    assert exported["min_re_Z_Ohm"] == pytest.approx(report.min_re_Z, abs=1e-9)
    assert exported["f_at_min_Hz"] == pytest.approx(report.f_at_min, rel=1e-6)
    assert exported["rho_checked_Ohm_m"] == [100.0, 100.0]
    assert exported["n_points"] == d["n_points"]


@pytest.mark.finding("10.40")
def test_non_finite_values_fail_with_a_strict_json_report() -> None:
    """A pole at s = 0 makes Z(0) infinite: not passive, and the report
    carries ``None`` instead of NaN (strict JSON)."""
    fit = _vf([0.0, -100.0], [5.0, 30.0], R_inf=1.0)
    report = check_passivity(fit)
    assert not report.passed
    assert math.isnan(report.min_re_Z)
    assert report.f_at_min == 0.0
    d = report.as_dict()
    assert d["min_re_Z_Ohm"] is None
    json.dumps(d, allow_nan=False)
    with pytest.raises(NonPassiveFitError, match="not finite at f = 0 Hz"):
        to_bustype_dict(
            fit,
            name="dc_pole",
            system_type="TN",
            voltage_level=0.4,
            rho_at_fit=100.0,
            require_passive=True,
        )


# ---------------------------------------------------------------------
# Regression guards (pass on the base as well)
# ---------------------------------------------------------------------


def test_seed17_dip_between_the_first_samples_is_found_by_both_paths() -> None:
    fit = _seed17()
    report = check_passivity(fit)
    assert report.min_re_Z == pytest.approx(-0.01937, abs=2e-4)
    assert report.f_at_min == pytest.approx(5.507, abs=5e-3)
    assert check_passivity(_passive_ladder()).passed


def test_the_rounded_formula_is_checked_not_the_fit() -> None:
    """Re Z = rho (k1 + k4 f) is +0.0002 rho at 2 kHz with the exact
    coefficients and -0.001 rho with two significant digits
    (0.0214 -> 0.021, -1.06e-5 -> -1.1e-5): the export checks what it
    writes."""
    fit = RhoFStandardFit(
        k1=0.0214,
        k2=0.0,
        k3=1e-3,
        k4=-1.06e-5,
        k5=0.0,
        rms_error=0.0,
        rms_relative=0.0,
        sample_rho=np.array([30.0, 1000.0, 30.0, 1000.0]),
        sample_f=np.array([0.0, 0.0, 1000.0, 1000.0]),
        sample_Z=np.ones(4, dtype=complex),
    )
    assert check_passivity(fit).passed
    with warnings.catch_warnings():
        warnings.simplefilter("error", NonPassiveFitWarning)
        to_bustype_dict(fit, name="ok", system_type="TN", voltage_level=0.4)
    with pytest.warns(NonPassiveFitWarning, match="rho = 1000 Ohm m"):
        payload = to_bustype_dict(
            fit, name="r2", system_type="TN", voltage_level=0.4, decimals=2
        )
    report = payload["metadata"]["passivity"]
    assert report["min_re_Z_Ohm"] == pytest.approx(1000.0 * (0.021 - 0.022))
    assert report["f_at_min_Hz"] == 2000.0 and report["rho_at_min_Ohm_m"] == 1000.0
