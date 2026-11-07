"""One formula renderer for rho-f fits (review pass 10, 10.73).

``fit_to_sympy`` (vector fits), ``fit_to_sympy_standard`` and the
groundinsight exporter build their expressions with one renderer: the
partial fractions of a vector fit (one term per real pole or conjugate
pole pair), the standard form term by term, and every floating-point
coefficient rounded **once** to ``decimals`` significant digits.

Up to the base of this release ``fit_to_sympy`` passed its sum through
``sympy.simplify``, which merged the partial fractions into one rational
function of degree four over three (a conditioning risk, R5-17), and the
exporter duplicated the pole-pairing rule in its own builder.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

sp = pytest.importorskip("sympy")

from groundfield.io._formula import (  # noqa: E402
    format_impedance_formula,
    parse_impedance_formula,
)
from groundfield.io.groundinsight import BusTypeSpec, evaluate_spec, to_bustype_dict  # noqa: E402
from groundfield.postprocess import vector_fitting  # noqa: E402
from groundfield.postprocess.rho_f_standard import (  # noqa: E402
    RhoFStandardFit,
    fit_rho_f_standard,
    fit_to_sympy_standard,
)
from groundfield.postprocess.vector_fitting import (  # noqa: E402
    VectorFitResult,
    fit_to_sympy,
)

S = sp.Symbol("s", complex=True)


def _fit(poles, residues, *, R_inf=12.0, L_inf=2.5e-4) -> VectorFitResult:
    freqs = np.array([0.0, 16.7, 50.0, 150.0, 400.0, 1000.0])
    return VectorFitResult(
        poles=np.asarray(poles, dtype=complex),
        residues=np.asarray(residues, dtype=complex),
        R_inf=R_inf,
        L_inf=L_inf,
        rms_error=0.0,
        fit_frequencies=freqs,
        fit_values=np.zeros(freqs.size, dtype=complex),
    )


def _pair_and_real_pole() -> VectorFitResult:
    """Complex pole pair (pass-9 F03 case, Im r < 0), a real pole and L_inf."""
    pole, residue = complex(-34.0, 383.0), complex(391.0, -451.0)
    return _fit(
        [pole, pole.conjugate(), -2.0 * math.pi * 700.0],
        [residue, residue.conjugate(), -2.0 * math.pi * 700.0 * 3.0],
    )


def _evaluate_in_s(expr, freqs: np.ndarray) -> np.ndarray:
    func = sp.lambdify(S, expr, modules="numpy")
    return np.asarray(func(2j * np.pi * freqs), dtype=complex) * np.ones(freqs.size)


def _significant_digits(text: str) -> int:
    mantissa = text.lower().split("e")[0].lstrip("+-").replace(".", "")
    return max(len(mantissa.lstrip("0").rstrip("0")), 1)


GRID = np.concatenate([[0.0], np.geomspace(0.1, 2000.0, 300)])


# ---------------------------------------------------------------------
# fit_to_sympy: partial fractions, single rounding
# ---------------------------------------------------------------------


@pytest.mark.finding("10.73")
def test_fit_to_sympy_keeps_one_term_per_pole_or_pole_pair() -> None:
    """R_inf + L_inf s + pair term + real-pole term: four terms, no s**3.

    The base returned ``(0.00025*s**4 + 13.1166*s**3 + ...)/(1.0*s**3 + ...)``.
    """
    expr = fit_to_sympy(_pair_and_real_pole())
    assert isinstance(expr, sp.Add)
    assert len(expr.args) == 4, expr
    text = str(expr)
    assert "s**3" not in text and "s**4" not in text, text
    # Every term is at most quadratic in s in its denominator.
    for term in expr.args:
        assert sp.degree(sp.denom(term), S) <= 2, term
    assert expr.free_symbols == {S}


@pytest.mark.parametrize("decimals, bound", [(3, 2e-2), (6, 2e-5), (12, 1e-10)])
def test_fit_to_sympy_rounds_every_coefficient_once(
    decimals: int, bound: float
) -> None:
    fit = _pair_and_real_pole()
    expr = fit_to_sympy(fit, decimals=decimals)
    floats = expr.atoms(sp.Float)
    assert floats
    for atom in floats:
        assert _significant_digits(str(atom)) <= decimals, (atom, expr)
    got = _evaluate_in_s(expr, GRID)
    ref = fit.evaluate(GRID)
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < bound


def test_full_precision_rendering_is_exact_partial_fractions() -> None:
    """At 17 digits the rendered expression is ``fit.evaluate`` to round-off."""
    fit = _pair_and_real_pole()
    got = _evaluate_in_s(fit_to_sympy(fit, decimals=17), GRID)
    ref = fit.evaluate(GRID)
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-14


@pytest.mark.finding("10.73")
@pytest.mark.parametrize(
    "bad", [0, -2, 2.5, True, float("inf"), float("nan"), "6", None]
)
def test_fit_to_sympy_rejects_invalid_decimals(bad) -> None:
    with pytest.raises(ValueError, match="significant digits"):
        fit_to_sympy(_pair_and_real_pole(), decimals=bad)


@pytest.mark.finding("10.73")
@pytest.mark.parametrize("digits", [np.int64(6), np.int32(6), 6.0, np.float64(6.0)])
def test_fit_to_sympy_accepts_integral_numpy_and_float_decimals(digits) -> None:
    """A digit count read from an array (``np.int64``) or a float that is an
    integer counts like the ``int``; the self-check found ``np.int64(6)``
    rejected while ``np.float64(6.0)`` was accepted."""
    fit = _pair_and_real_pole()
    assert fit_to_sympy(fit, decimals=digits) == fit_to_sympy(fit, decimals=6)


def test_near_conjugate_poles_render_a_real_pair_term() -> None:
    """Poles within the 1e-3 pairing tolerance are symmetrised into one
    real second-order term (pairing rule shared with the exporter)."""
    p = complex(-2.0 * math.pi * 30.0, 2.0 * math.pi * 200.0)
    r = complex(-50.0, 25.0)
    q = complex(p.real * (1 + 2e-4), -p.imag * (1 - 3e-4))
    fit = _fit([p, q], [r, r.conjugate()], R_inf=1.0, L_inf=0.0)
    expr = fit_to_sympy(fit, decimals=12)
    assert len(expr.args) == 2  # R_inf and one pair term
    value = complex(expr.subs(S, sp.Float(100.0)).evalf())
    assert value.imag == 0.0
    # A lone complex pole (no partner within tolerance) keeps a complex term.
    lone = _fit([p], [r], R_inf=1.0, L_inf=0.0)
    expr = fit_to_sympy(lone, decimals=12)
    assert expr.has(sp.I)
    np.testing.assert_allclose(
        _evaluate_in_s(expr, GRID), lone.evaluate(GRID), rtol=1e-10
    )


# ---------------------------------------------------------------------
# One renderer shared with the exporter
# ---------------------------------------------------------------------


@pytest.mark.finding("10.73")
def test_exporter_and_fit_to_sympy_share_the_renderer(monkeypatch) -> None:
    calls: list = []
    real = vector_fitting._partial_fractions

    def spy(fit, s):
        calls.append(s)
        return real(fit, s)

    monkeypatch.setattr(vector_fitting, "_partial_fractions", spy)
    fit = _pair_and_real_pole()
    fit_to_sympy(fit)
    to_bustype_dict(
        fit, name="vf", system_type="LV", voltage_level=0.4, rho_at_fit=100.0
    )
    assert len(calls) == 2
    assert calls[0] == S
    assert calls[1].free_symbols == {sp.Symbol("f"), sp.Symbol("j")}


@pytest.mark.finding("10.73")
@pytest.mark.parametrize("decimals", [6, 12, 15])
def test_exported_formula_is_the_fit_to_sympy_expression_in_f(decimals: int) -> None:
    """The exported text has the partial-fraction terms of ``fit_to_sympy``,
    each coefficient correctly rounded once, and evaluates like it."""
    fit = _pair_and_real_pole()
    formula = to_bustype_dict(
        fit,
        name="vf",
        system_type="LV",
        voltage_level=0.4,
        rho_at_fit=100.0,
        decimals=decimals,
    )["impedance_formula"]
    expected = format_impedance_formula(
        vector_fitting._partial_fractions(
            fit, 2 * sp.pi * sp.Symbol("j") * sp.Symbol("f")
        ),
        significant_digits=decimals,
    )
    assert formula == expected
    parsed = parse_impedance_formula(formula)
    assert len(parsed.args) == len(fit_to_sympy(fit, decimals=decimals).args)
    spec = BusTypeSpec.from_dict(
        to_bustype_dict(
            fit,
            name="vf",
            system_type="LV",
            voltage_level=0.4,
            rho_at_fit=100.0,
            decimals=decimals,
        )
    )
    got = evaluate_spec(spec, GRID, 100.0)
    via_sympy = _evaluate_in_s(fit_to_sympy(fit, decimals=decimals), GRID)
    scale = np.max(np.abs(via_sympy))
    assert np.max(np.abs(got - via_sympy)) / scale < 5 * 10.0 ** (1 - decimals)


# ---------------------------------------------------------------------
# Standard form
# ---------------------------------------------------------------------


def _standard_fit() -> RhoFStandardFit:
    rho = np.repeat([30.0, 100.0, 300.0, 1000.0], 4)
    f = np.tile([0.0, 50.0, 500.0, 1000.0], 4)
    z = 2.0 + 0.05 * rho + 2e-3j * f + (2e-6 + 3e-6j) * f * rho
    return fit_rho_f_standard(rho, f, z)


def _rounded_text(value: float, digits: int) -> str:
    return format(float(value), f".{digits}g")


@pytest.mark.parametrize("decimals", [4, 6, 12])
def test_standard_form_is_rendered_term_by_term_and_rounded_once(decimals: int) -> None:
    """Each coefficient k0 ... k5 appears once, correctly rounded once."""
    fit = _standard_fit()
    expr = fit_to_sympy_standard(fit, decimals=decimals)
    printed = sorted(_rounded_text(a, decimals) for a in expr.atoms(sp.Float))
    expected = sorted(_rounded_text(v, decimals) for v in fit.coefficients.values())
    assert printed == expected, expr
    for atom in expr.atoms(sp.Float):
        assert _significant_digits(str(atom)) <= decimals
    assert expr.free_symbols == {sp.Symbol("rho", real=True, positive=True)} | {
        sp.Symbol("f", real=True, positive=True)
    }
    rho = np.array([30.0, 137.0, 1000.0])[:, None]
    func = sp.lambdify(
        sorted(expr.free_symbols, key=str), expr, modules="numpy"
    )  # (f, rho)
    got = np.asarray(func(GRID[None, :], rho), dtype=complex)
    ref = fit.evaluate(rho, GRID[None, :])
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 10.0 ** (1 - decimals)


def test_standard_export_prints_each_coefficient_once() -> None:
    fit = _standard_fit()
    formula = to_bustype_dict(
        fit, name="std", system_type="TN", voltage_level=0.4, decimals=8
    )["impedance_formula"]
    literals = re.findall(r"\d+(?:\.\d+)?(?:e-?\d+)?", formula)
    for value in fit.coefficients.values():
        text = format(abs(value), ".8g")
        mantissa, sep, exponent = text.partition("e")
        text = f"{mantissa}e{int(exponent)}" if sep else text
        assert text in literals, (text, formula)


# ---------------------------------------------------------------------
# round_coefficients
# ---------------------------------------------------------------------


@pytest.mark.finding("10.73")
def test_round_coefficients_rounds_each_float_once_and_is_idempotent() -> None:
    from groundfield.io._formula import round_coefficients

    f = sp.Symbol("f")
    expr = sp.Float(1.23456789012345) * f + sp.Float(-9.87654321e-7) + sp.Float(25.0)
    out = round_coefficients(expr, 6)
    assert {str(a) for a in out.atoms(sp.Float)} >= {"1.23457"}
    assert round_coefficients(out, 6) == out
    assert format_impedance_formula(out, significant_digits=6) == (
        format_impedance_formula(expr, significant_digits=6)
    )
    with pytest.raises(ValueError, match="significant digits"):
        round_coefficients(expr, 0)


@pytest.mark.finding("10.73")
@pytest.mark.parametrize("digits", list(range(1, 16)))
def test_printing_a_rounded_expression_is_idempotent(digits: int) -> None:
    """``format_impedance_formula`` of a pre-rounded expression gives the
    same text as printing the full-precision one (single rounding).

    Up to 15 digits every decimal survives the float64 conversion of the
    printer; 16-digit decimals do not always (DBL_DIG = 15).
    """
    from groundfield.io._formula import round_coefficients

    rng = np.random.default_rng(digits)
    f = sp.Symbol("f")
    values = rng.standard_normal(40) * 10.0 ** rng.integers(-12, 12, 40)
    expr = sp.Add(*[sp.Float(float(v)) * f**k for k, v in enumerate(values[:8])])
    for v in values:
        e = sp.Float(float(v)) * f
        assert format_impedance_formula(
            round_coefficients(e, digits), significant_digits=digits
        ) == format_impedance_formula(e, significant_digits=digits), (v, digits)
    assert format_impedance_formula(
        round_coefficients(expr, digits), significant_digits=digits
    ) == format_impedance_formula(expr, significant_digits=digits)
