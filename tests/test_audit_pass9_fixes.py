"""Regression tests for the ninth 2026-07-28 review pass (0.14.1).

Each test maps 1:1 to one of the user-visible bullets in the
``Fixed (Review pass 9 — implemented 2026-07-28)`` CHANGELOG block.
Eleven of the fourteen tests fail on ``0.14.0`` (commit ``428442d``)
and pass afterwards; the remaining three are deliberate *controls* over
behaviour that was already correct and must not regress
(:func:`test_fit_to_sympy_preserves_positive_residue_imag`,
:func:`test_with_frequencies_preserve_order_stays_silent`,
:func:`test_with_frequencies_accepts_dc_and_ascending`).

* :func:`test_fit_to_sympy_preserves_negative_residue_imag`
* :func:`test_fit_to_sympy_preserves_positive_residue_imag`
* :func:`test_fit_to_sympy_matches_fit_over_a_frequency_sweep`
* :func:`test_bustype_spec_roundtrip_with_negative_residue_imag`
* :func:`test_plot_potential_contour_without_extent`
* :func:`test_plot_potential_contour_xz_without_extent`
* :func:`test_with_frequencies_rejects_empty`
* :func:`test_with_frequencies_rejects_negative`
* :func:`test_with_frequencies_rejects_nan`
* :func:`test_with_frequencies_rejects_inf`
* :func:`test_with_frequencies_warns_on_non_monotonic`
* :func:`test_with_frequencies_preserve_order_stays_silent`
* :func:`test_with_frequencies_accepts_dc_and_ascending`
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import sympy as sp

import groundfield as gf
from groundfield.io.groundinsight import (
    BusTypeSpec,
    evaluate_spec,
    to_bustype_dict,
)
from groundfield.postprocess.vector_fitting import (
    VectorFitResult,
    fit_to_sympy,
)
from groundfield.solver.engine import Engine, EngineFrequencyOrderWarning
from groundfield.soil.models import HomogeneousSoil


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _pair_fit(pole: complex, residue: complex, *, R_inf: float = 0.0):
    """Build a one-conjugate-pair :class:`VectorFitResult`."""
    return VectorFitResult(
        poles=np.array([pole, np.conj(pole)]),
        residues=np.array([residue, np.conj(residue)]),
        R_inf=R_inf,
        L_inf=0.0,
        rms_error=0.0,
        fit_frequencies=np.array([50.0]),
        fit_values=np.array([0j]),
    )


def _pair_truth(pole: complex, residue: complex, f: float, R_inf: float = 0.0):
    """Analytic value of ``R_inf + r/(s-p) + r*/(s-p*)`` at ``f``."""
    s = 2j * np.pi * f
    return R_inf + residue / (s - pole) + np.conj(residue) / (s - np.conj(pole))


def _eval_sympy(expr, f: float) -> complex:
    fn = sp.lambdify(sp.Symbol("s"), expr, "numpy")
    return complex(fn(2j * np.pi * f))


# ---------------------------------------------------------------------
# fit_to_sympy — sign of the canonical residue imaginary part
# ---------------------------------------------------------------------


def test_fit_to_sympy_preserves_negative_residue_imag():
    """A pair with ``Im(r) < 0`` must export the residue sign unchanged.

    The pre-0.14.1 implementation symmetrised the pair with
    ``0.5 * (abs(r.imag) + abs(r2.imag))``, forcing ``Im(r) >= 0``. For
    a canonical residue with a negative imaginary part that flips the
    sign of the ``-2*Im(r)*Im(p)`` numerator term, so the exported
    expression is a *different* rational function from the fit.
    """
    pole, residue = complex(-100.0, 300.0), complex(2.0, -5.0)
    expr = fit_to_sympy(_pair_fit(pole, residue), decimals=15)
    for f in (1.0, 50.0, 500.0):
        assert _eval_sympy(expr, f) == pytest.approx(
            _pair_truth(pole, residue, f), rel=1e-9
        )


def test_fit_to_sympy_preserves_positive_residue_imag():
    """The ``Im(r) > 0`` branch was already correct and must stay so."""
    pole, residue = complex(-100.0, 300.0), complex(2.0, 5.0)
    expr = fit_to_sympy(_pair_fit(pole, residue), decimals=15)
    for f in (1.0, 50.0, 500.0):
        assert _eval_sympy(expr, f) == pytest.approx(
            _pair_truth(pole, residue, f), rel=1e-9
        )


def test_fit_to_sympy_matches_fit_over_a_frequency_sweep():
    """Two pairs with mixed residue signs, checked against the fit itself."""
    poles = np.array([-100 + 900j, -100 - 900j, -500 + 3000j, -500 - 3000j])
    residues = np.array([50 - 300j, 50 + 300j, -20 + 100j, -20 - 100j])
    freqs = np.array([1.0, 16.7, 50.0, 150.0, 1000.0])
    fit = VectorFitResult(
        poles=poles,
        residues=residues,
        R_inf=5.0,
        L_inf=0.0,
        rms_error=0.0,
        fit_frequencies=freqs,
        fit_values=np.zeros(freqs.size, dtype=complex),
    )
    expr = fit_to_sympy(fit, decimals=15)
    for f in freqs:
        s = 2j * np.pi * f
        truth = 5.0 + np.sum(residues / (s - poles))
        assert _eval_sympy(expr, float(f)) == pytest.approx(truth, rel=1e-9)


def test_bustype_spec_roundtrip_with_negative_residue_imag():
    """The full groundinsight bridge must reproduce the fit.

    This is the path that matters in practice: ``to_bustype_dict`` ->
    ``BusTypeSpec`` -> ``evaluate_spec``. Before 0.14.1 the exported
    ``impedance_formula`` disagreed with the fit by tens of per cent for
    residues with a negative imaginary part.
    """
    pole, residue = complex(-34.0, 383.0), complex(391.0, -451.0)
    freqs = [1.0, 16.7, 50.0, 400.0]
    fit = VectorFitResult(
        poles=np.array([pole, np.conj(pole)]),
        residues=np.array([residue, np.conj(residue)]),
        R_inf=12.0,
        L_inf=0.0,
        rms_error=0.0,
        fit_frequencies=np.array(freqs),
        fit_values=np.zeros(len(freqs), dtype=complex),
    )
    payload = to_bustype_dict(
        fit,
        name="pass9",
        system_type="LV",
        voltage_level=0.4,
        decimals=15,
        rho_at_fit=100.0,
    )
    got = evaluate_spec(BusTypeSpec.from_dict(payload), freqs, rho=100.0)
    truth = np.array([_pair_truth(pole, residue, f, R_inf=12.0) for f in freqs])
    assert got == pytest.approx(truth, rel=1e-6)


# ---------------------------------------------------------------------
# plot_potential_contour — NumPy >= 2 compatibility
# ---------------------------------------------------------------------


def _solved_world():
    world = gf.create_world(soil=HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world, "rod", name="g1", position=(0.0, 0.0, 0.5),
        length=3.0, wire_radius=0.01,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    return world, gf.create_engine(backend="image").solve(world)


@pytest.mark.parametrize("plane", ["xy", "xz"])
def test_plot_potential_contour_without_extent(plane):
    """``extent=None`` derives the window itself — the documented default.

    ``ndarray.ptp()`` was removed in NumPy 2.0 while the project pins
    ``numpy = "^2.1.0"``, so this branch raised ``AttributeError`` on
    every supported NumPy before 0.14.1.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    world, result = _solved_world()
    fig = gf.plot_potential_contour(result, world=world, plane=plane, n=12)
    assert fig is not None


def test_plot_potential_contour_xz_without_extent():
    """Explicit companion to the parametrised case, kept for the 1:1 map."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    world, result = _solved_world()
    assert gf.plot_potential_contour(
        result, world=world, plane="xz", y=0.0, n=12
    ) is not None


# ---------------------------------------------------------------------
# Engine.with_frequencies — validation on the default path
# ---------------------------------------------------------------------


def test_with_frequencies_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        Engine(backend="image").with_frequencies()


def test_with_frequencies_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        Engine(backend="image").with_frequencies(-50.0)


def test_with_frequencies_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        Engine(backend="image").with_frequencies(float("nan"))


def test_with_frequencies_rejects_inf():
    with pytest.raises(ValueError, match="inf"):
        Engine(backend="image").with_frequencies(float("inf"))


def test_with_frequencies_warns_on_non_monotonic():
    """The default path documents that ``_validate_frequencies`` runs.

    ``model_copy(update=...)`` skips every validator under pydantic v2,
    so before 0.14.1 the warning never fired on this path.
    """
    with pytest.warns(EngineFrequencyOrderWarning):
        eng = Engine(backend="image").with_frequencies(5000.0, 50.0)
    assert eng.frequencies == [5000.0, 50.0]


def test_with_frequencies_preserve_order_stays_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", EngineFrequencyOrderWarning)
        eng = Engine(backend="image").with_frequencies(
            5000.0, 50.0, preserve_order=True
        )
    assert eng.frequencies == [5000.0, 50.0]


def test_with_frequencies_accepts_dc_and_ascending():
    """DC is a legitimate quasi-static operating point; order is kept."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", EngineFrequencyOrderWarning)
        eng = Engine(backend="image").with_frequencies(0.0, 50.0, 150.0)
    assert eng.frequencies == [0.0, 50.0, 150.0]
