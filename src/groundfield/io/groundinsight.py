"""Export of reduced ``rho-f`` fits to ``groundinsight``.

This module is the canonical bridge between the field-grade
reference computation in ``groundfield`` and the reduced equivalent
network model in ``groundinsight``.

Two equally supported transports are provided:

* **JSON file** — neutral, language-agnostic schema versioned via
  ``schema_version``. Produce with :func:`to_bustype_dict` /
  :func:`save_bustype_json`; read back with
  :func:`load_bustype_json`.
* **Python API** — :func:`to_bustype` returns a live
  :class:`groundinsight.BusType` Pydantic instance via a lazy
  import. ``groundinsight`` is therefore an *optional* dependency
  of ``groundfield`` (extras group ``[groundinsight]``). The JSON
  path does not require ``groundinsight`` at all.

Mathematical background
-----------------------
``groundinsight.BusType.impedance_formula`` is parsed with two free
symbols, ``f`` (frequency in Hz) and ``rho``
(``Bus.specific_earth_resistance``), and the imaginary unit ``j``.
Both fit families exported here are projected onto that symbol set:

* :class:`~groundfield.postprocess.rho_f_standard.RhoFStandardFit`
  is already in the canonical ``(rho, f)`` form

  $$
  Z(\\rho, f) \\;=\\; k_1\\rho \\;+\\; (k_2 + j k_3)\\,f
                    \\;+\\; (k_4 + j k_5)\\,f\\,\\rho.
  $$

* :class:`~groundfield.postprocess.vector_fitting.VectorFitResult`
  is a rational function of the Laplace variable
  $s = j\\,2\\pi f$. The export substitutes
  $s\\to j\\,2\\pi f$ symbolically so the resulting expression is
  in ``f`` only (independent of ``rho``); the soil resistivity the
  fit belongs to is recorded in the samples and metadata blocks.

The exporter does not care which impedance was fitted (input
impedance, cluster impedance or the impedance of one electrode); it
exports the function it is given.


Formula text
------------
The formula keeps the structure of the fit (partial fractions for a
vector fit) and prints every floating-point coefficient once, rounded
to ``decimals`` significant digits with Python's correctly rounded
float formatting, so the text is identical on every platform. A
coefficient therefore carries a relative rounding error of at most
$\\tfrac12\\,10^{1-N}$ for $N$ = ``decimals``; with the default
$N = 12$ the formula reproduces ``fit.evaluate`` to about
$10^{-11}$ relative.

Formula safety
--------------
A formula read from a file is untrusted text. :func:`evaluate_spec`,
:meth:`BusTypeSpec.from_dict` and :func:`load_bustype_json` parse it
with an allow-list parser that builds the SymPy expression from the
Python syntax tree and never evaluates the text: numbers, ``f``,
``rho``, ``j`` / ``I``, ``pi``, ``E``, ``+ - * /``, ``**`` with a
literal exponent, and ``sqrt`` / ``exp`` / ``log``, within size and
magnitude bounds. Every formula written by the exporter satisfies this
grammar, including those of ``groundfield`` 0.15.0 and earlier.

Passivity
---------
A grounding impedance is the driving-point impedance of a passive
one-port, so its real part is non-negative at every frequency,

$$
\\operatorname{Re} Z(j\\,2\\pi f) \\;\\ge\\; 0 \\qquad (f \\ge 0).
$$

A fit can violate this between its samples (a spurious resonance) or
beyond them. Before a ``BusType`` is built the exported formula is
evaluated on a dense grid from 0 to twice the highest fitted frequency
(linear and logarithmic points, the samples and the midpoints between
them, and for a vector fit points around every pole), for the standard
form on the fitted range of ``rho``, and the grid minima are refined
with a bounded scalar minimisation. The result is stored in
``metadata["passivity"]``; a violation emits
:class:`NonPassiveFitWarning`, or raises :class:`NonPassiveFitError`
with ``require_passive=True``.

See also
--------
``docs/adr/0008-groundinsight-bridge.md`` — full design
rationale, JSON schema, and validation programme.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as _importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np

from groundfield.postprocess.rho_f_standard import (
    RhoFStandardFit,
    fit_to_sympy_standard,
)
from groundfield.postprocess.vector_fitting import VectorFitResult


def _resolve_groundfield_version() -> str:
    """Resolve the installed ``groundfield`` version without triggering
    the package ``__init__`` (and its heavy Pydantic imports). Falls
    back to the literal stored in ``groundfield/__init__.py`` when the
    package is not installed (e.g. running from a source checkout
    without ``pip install -e .``).
    """
    try:
        return _importlib_metadata.version("groundfield")
    except _importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


_GROUNDFIELD_VERSION = _resolve_groundfield_version()

if TYPE_CHECKING:  # pragma: no cover - typing only
    import sympy

__all__ = [
    "BusTypeSpec",
    "to_bustype_dict",
    "to_bustype",
    "save_bustype_json",
    "load_bustype_json",
    "save_bustype_to_db",
    "evaluate_spec",
    "fit_quality_summary",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "EvaluateSpecError",
    "NonPassiveFitWarning",
    "NonPassiveFitError",
]


class EvaluateSpecError(ValueError):
    """Raised when a :class:`BusTypeSpec` or its formula is rejected.

    Raised by :func:`evaluate_spec` and, for a formula outside the
    allowed grammar, already by :meth:`BusTypeSpec.from_dict` and
    :func:`load_bustype_json`. Subclass of :class:`ValueError` so legacy
    ``except ValueError`` blocks keep working unchanged; downstream
    consumers (notably the ``groundinsight`` consumer of the bridge
    format) can catch the typed exception instead of matching against
    ``str(exc)`` substrings.
    """


class NonPassiveFitWarning(UserWarning):
    """The exported impedance formula has ``Re Z < 0`` somewhere.

    A passive grounding impedance satisfies $\\operatorname{Re} Z \\ge 0$
    at every frequency. The warning names the frequency (and ``rho``)
    of the most negative real part; ``metadata["passivity"]`` of the
    exported document carries the same numbers. Refit, or export with
    ``require_passive=True`` to turn the warning into
    :class:`NonPassiveFitError`.
    """


class NonPassiveFitError(ValueError):
    """Raised instead of :class:`NonPassiveFitWarning` with ``require_passive=True``."""


SCHEMA_NAME = "groundfield.bustype"
SCHEMA_VERSION = 1
"""Schema name and version of the JSON document produced by this module."""

_FULL_PRECISION_DIGITS = 17
"""Digits used inside SymPy before the single rounding at print time."""

_PASSIVITY_BAND_FACTOR = 2.0
"""The passivity check covers 0 Hz to this factor times the highest fitted frequency."""

_PASSIVITY_REL_TOL = 1e-9
"""``Re Z >= -tol * max|Z|`` counts as passive (rounding of the formula)."""


_IO_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


# ---------------------------------------------------------------------
# Spec dataclass — neutral, in-memory representation of the JSON file
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BusTypeSpec:
    """Neutral, in-memory representation of an exported ``BusType``.

    The spec is what :func:`to_bustype_dict` returns under the hood and
    what :func:`load_bustype_json` reads back. It carries everything
    necessary to either build a ``groundinsight.BusType`` (via
    :func:`to_bustype`) or to write the on-disk JSON document.

    Attributes
    ----------
    name, description, system_type, voltage_level
        The four scalar fields that the ``groundinsight.BusType``
        Pydantic model carries.
    impedance_formula
        The fitted ``Z(rho, f)`` expression as text. Free symbols:
        ``f`` (Hz) and ``rho`` ($\\Omega\\,\\mathrm{m}$); imaginary unit
        ``j``. Only the grammar described in the module docstring is
        accepted when the spec is loaded or evaluated.
    samples
        Dict with the parallel tabular representation. Keys
        ``"frequency_Hz"``, ``"rho_Ohm_m"``, ``"Z_real_Ohm"``,
        ``"Z_imag_Ohm"`` each map to a list of floats of equal
        length. Provided so a future tabular ingest path on the
        ``groundinsight`` side can be served immediately.
    metadata
        Free-form metadata dict; populated by the exporters with
        the fit method, fit quality, fit-method-specific details
        (poles/residues for vector fits, $k_1\\dots k_5$ for the
        standard form), the passivity check (``"passivity"``), the
        source ``groundfield`` version and a UTC timestamp.
    """

    name: str
    description: Optional[str]
    system_type: str
    voltage_level: float
    impedance_formula: str
    samples: dict[str, list[float]]
    metadata: dict[str, Any]

    # --- serialisation ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready ``dict`` matching schema v1."""
        return {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "system_type": self.system_type,
            "voltage_level": float(self.voltage_level),
            "impedance_formula": self.impedance_formula,
            "samples": {
                "frequency_Hz": [float(x) for x in self.samples["frequency_Hz"]],
                "rho_Ohm_m": [float(x) for x in self.samples["rho_Ohm_m"]],
                "Z_real_Ohm": [float(x) for x in self.samples["Z_real_Ohm"]],
                "Z_imag_Ohm": [float(x) for x in self.samples["Z_imag_Ohm"]],
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BusTypeSpec":
        """Build a :class:`BusTypeSpec` from a JSON-loaded ``dict``.

        Parameters
        ----------
        payload : dict
            Schema-v1 document as produced by :func:`to_bustype_dict`.

        Returns
        -------
        BusTypeSpec

        Raises
        ------
        ValueError
            If the schema name or version does not match.
        EvaluateSpecError
            If ``impedance_formula`` is missing or outside the allowed
            formula grammar. The formula is parsed, never evaluated as
            code; the message names the offending element.
        """
        if payload.get("schema") != SCHEMA_NAME:
            raise ValueError(
                f"unexpected schema name: got {payload.get('schema')!r}, "
                f"expected {SCHEMA_NAME!r}"
            )
        version = int(payload.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version}; this groundfield "
                f"build only reads v{SCHEMA_VERSION}"
            )
        name = str(payload["name"])
        formula = payload.get("impedance_formula")
        if formula is None:
            raise EvaluateSpecError(
                f"BusType document {name!r} has no 'impedance_formula'."
            )
        formula = formula if isinstance(formula, str) else str(formula)
        _parse_formula(formula, context=f"BusType document {name!r}")
        samples = payload.get("samples") or {}
        return cls(
            name=name,
            description=payload.get("description"),
            system_type=str(payload["system_type"]),
            voltage_level=float(payload["voltage_level"]),
            impedance_formula=formula,
            samples={
                "frequency_Hz": list(samples.get("frequency_Hz", [])),
                "rho_Ohm_m": list(samples.get("rho_Ohm_m", [])),
                "Z_real_Ohm": list(samples.get("Z_real_Ohm", [])),
                "Z_imag_Ohm": list(samples.get("Z_imag_Ohm", [])),
            },
            metadata=dict(payload.get("metadata", {})),
        )


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


_GRAMMAR_HINT = (
    "An impedance formula is an expression in f (Hz) and rho (Ohm m) built "
    "from numbers, j (or I), pi, E, + - * /, ** with a literal exponent, "
    "sqrt, exp and log; groundfield parses it and never runs it as code. "
    "Formulas written by groundfield's exporter always satisfy this grammar."
)


def _parse_formula(text: str, *, context: str) -> "sympy.Expr":
    """Parse ``text`` with the allow-list parser or raise :class:`EvaluateSpecError`."""
    from groundfield.io._formula import FormulaError, parse_impedance_formula

    try:
        return parse_impedance_formula(text, symbols=("f", "rho"))
    except FormulaError as exc:
        raise EvaluateSpecError(
            f"{context}: impedance_formula is not an allowed impedance formula: "
            f"{exc}. {_GRAMMAR_HINT}"
        ) from exc


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _render(expr: "sympy.Expr", *, decimals: int) -> str:
    """Print ``expr`` with ``decimals`` significant digits per coefficient."""
    from groundfield.io._formula import format_impedance_formula

    return format_impedance_formula(expr, significant_digits=decimals)


_POLE_PAIR_REL_TOL = 1e-3
"""Relative tolerance for treating a pole as real or two poles as a pair.

Same rule as :func:`groundfield.postprocess.vector_fitting.fit_to_sympy`.
"""


def _vector_fit_expression(fit: VectorFitResult) -> "sympy.Expr":
    """Partial-fraction SymPy expression of a vector fit in ``f``.

    With $s = 2\\pi j f$ (``j`` a symbol printed as the imaginary unit)

    $$
    Z = R_\\infty + L_\\infty s + \\sum_\\text{real} \\frac{r}{s - p}
        + \\sum_\\text{pairs}
          \\frac{2\\Re r\\,(s - \\Re p) - 2\\Im r\\,\\Im p}
               {(s - \\Re p)^2 + (\\Im p)^2},
    $$

    one term per real pole or conjugate pole pair, matched and
    symmetrised as :func:`fit_to_sympy` does (the residue at the pole
    with positive imaginary part fixes the sign; a complex pole without
    partner keeps a complex term). Unlike :func:`fit_to_sympy` the sum
    is not passed through ``sympy.simplify``, which can merge the terms
    into one high-order rational function (e.g. ``pi**4*f**4*j**4``):
    the partial fractions stay readable and well conditioned.
    """
    import sympy as sp

    s = 2 * sp.pi * sp.Symbol("j") * sp.Symbol("f")
    poles = np.asarray(fit.poles, dtype=complex).ravel()
    residues = np.asarray(fit.residues, dtype=complex).ravel()
    tol = _POLE_PAIR_REL_TOL
    terms = [sp.Float(float(fit.R_inf))]
    if float(fit.L_inf) != 0.0:
        terms.append(sp.Float(float(fit.L_inf)) * s)
    used = np.zeros(poles.size, dtype=bool)
    for k, (p, r) in enumerate(zip(poles, residues, strict=True)):
        if used[k]:
            continue
        used[k] = True
        if abs(p.imag) < tol * max(abs(p.real), 1.0):
            terms.append(sp.Float(r.real) / (s - sp.Float(p.real)))
            continue
        partner, best = -1, math.inf
        for k2 in range(k + 1, poles.size):
            if used[k2]:
                continue
            d_real = abs(poles[k2].real - p.real) / max(abs(p.real), 1.0)
            d_imag = abs(poles[k2].imag + p.imag) / max(abs(p.imag), 1.0)
            if d_real < tol and d_imag < tol and d_real + d_imag < best:
                partner, best = k2, d_real + d_imag
        if partner < 0:  # complex pole without a conjugate partner
            r_c = sp.Float(r.real) + sp.I * sp.Float(r.imag)
            p_c = sp.Float(p.real) + sp.I * sp.Float(p.imag)
            terms.append(r_c / (s - p_c))
            continue
        used[partner] = True
        p2, r2 = poles[partner], residues[partner]
        r_plus, r_minus = (r, r2) if p.imag > 0.0 else (r2, r)
        re_p = sp.Float(0.5 * (p.real + p2.real))
        im_p = sp.Float(0.5 * (abs(p.imag) + abs(p2.imag)))
        re_r = sp.Float(0.5 * (r.real + r2.real))
        im_r = sp.Float(0.5 * (r_plus.imag - r_minus.imag))
        numerator = 2 * re_r * (s - re_p) - 2 * im_r * im_p
        terms.append(numerator / ((s - re_p) ** 2 + im_p**2))
    return sp.Add(*terms)


def _vector_fit_to_formula_in_f(fit: VectorFitResult, *, decimals: int) -> str:
    """Convert a ``VectorFitResult`` to an ``impedance_formula`` string in ``f``.

    The vector fit is in the Laplace variable $s$; ``groundinsight``
    expects the formula in $f$ (Hz) with the imaginary unit ``j``.
    :func:`_vector_fit_expression` builds the partial fractions with
    $s = 2\\pi j f$ at full float64 precision, and every coefficient is
    rounded once, to ``decimals`` significant digits, when the text is
    printed. No rationalisation takes place, so the coefficients stay
    readable floats.
    """
    expr = _vector_fit_expression(fit)
    text = _render(expr, decimals=decimals)  # raises on non-finite values
    if not expr.free_symbols:
        warnings.warn(
            "the exported vector fit does not depend on the frequency f "
            f"(Z = {float(fit.R_inf):g} Ohm): it has no poles and L_inf = 0.",
            UserWarning,
            skip_file_prefixes=(_IO_DIR,),
        )
    return text


def _rho_f_standard_to_formula(fit: RhoFStandardFit, *, decimals: int) -> str:
    """Convert a ``RhoFStandardFit`` to an ``impedance_formula`` string.

    The standard form already has the canonical ``(rho, f)`` symbol
    set used by ``groundinsight``; the imaginary unit is printed as
    ``j`` and every coefficient is rounded once, to ``decimals``
    significant digits.
    """
    expr = fit_to_sympy_standard(fit, decimals=_FULL_PRECISION_DIGITS)
    return _render(expr, decimals=decimals)


def _samples_from_vector_fit(
    fit: VectorFitResult,
    *,
    rho_at_fit: float,
) -> dict[str, list[float]]:
    """Build the ``samples`` block from a vector fit (single rho)."""
    f = np.asarray(fit.fit_frequencies, dtype=float)
    Z = np.asarray(fit.fit_values, dtype=complex)
    n = f.size
    return {
        "frequency_Hz": f.tolist(),
        "rho_Ohm_m": [float(rho_at_fit)] * n,
        "Z_real_Ohm": Z.real.tolist(),
        "Z_imag_Ohm": Z.imag.tolist(),
    }


def _samples_from_rho_f_standard(
    fit: RhoFStandardFit,
) -> dict[str, list[float]]:
    """Build the ``samples`` block from a standard-form rho-f fit."""
    return {
        "frequency_Hz": fit.sample_f.astype(float).tolist(),
        "rho_Ohm_m": fit.sample_rho.astype(float).tolist(),
        "Z_real_Ohm": fit.sample_Z.real.astype(float).tolist(),
        "Z_imag_Ohm": fit.sample_Z.imag.astype(float).tolist(),
    }


def _metadata_from_vector_fit(
    fit: VectorFitResult,
    *,
    rho_at_fit: float,
    electrode_name: Optional[str],
    soil_summary: Optional[str],
) -> dict[str, Any]:
    """Metadata block for a vector-fit-based export."""
    return {
        "fit_method": "vector_fit",
        "fit_quality": {
            "rms_error_Ohm": float(fit.rms_error),
        },
        "n_poles": int(fit.poles.size),
        "poles": [{"real": float(p.real), "imag": float(p.imag)} for p in fit.poles],
        "residues": [
            {"real": float(r.real), "imag": float(r.imag)} for r in fit.residues
        ],
        "R_inf": float(fit.R_inf),
        "L_inf": float(fit.L_inf),
        "rho_at_fit_Ohm_m": float(rho_at_fit),
        "electrode_name": electrode_name,
        "soil_summary": soil_summary,
        "source": f"groundfield {_GROUNDFIELD_VERSION}",
        "created_at_utc": _utc_now_iso(),
    }


def _metadata_from_rho_f_standard(
    fit: RhoFStandardFit,
    *,
    electrode_name: Optional[str],
    soil_summary: Optional[str],
) -> dict[str, Any]:
    """Metadata block for a standard-form rho-f export."""
    return {
        "fit_method": "rho_f_standard",
        "fit_quality": {
            "rms_error_Ohm": float(fit.rms_error),
            "rms_relative": float(fit.rms_relative),
        },
        "coefficients": {
            "k1": float(fit.k1),
            "k2": float(fit.k2),
            "k3": float(fit.k3),
            "k4": float(fit.k4),
            "k5": float(fit.k5),
        },
        "electrode_name": electrode_name,
        "soil_summary": soil_summary,
        "source": f"groundfield {_GROUNDFIELD_VERSION}",
        "created_at_utc": _utc_now_iso(),
    }


# ---------------------------------------------------------------------
# Public conversion API
# ---------------------------------------------------------------------


def _passivity_frequency_grid(
    f_samples: np.ndarray, f_hi: float, poles: np.ndarray
) -> np.ndarray:
    """Frequencies (Hz) at which the passivity check evaluates ``Re Z``.

    Linear and logarithmic points over ``[0, f_hi]``, the samples and
    the midpoints between them, and for every pole $p = -\\alpha \\pm
    j\\beta$ points $(\\beta + t\\,\\alpha)/2\\pi$, $|t| \\le 8$, which
    resolve resonances narrower than any fixed grid.
    """
    parts = [np.array([0.0, f_hi])]
    if f_hi > 0.0:
        parts += [np.linspace(0.0, f_hi, 2001), np.geomspace(f_hi * 1e-6, f_hi, 601)]
    samples = np.unique(f_samples)
    parts += [samples, 0.5 * (samples[1:] + samples[:-1])]
    t = np.linspace(-8.0, 8.0, 161)
    for p in np.asarray(poles, dtype=complex).ravel():
        width = max(abs(p.real), 1e-9 * max(abs(p), 1.0))
        parts.append((abs(p.imag) + width * t) / (2.0 * math.pi))
    grid = np.concatenate(parts)
    grid = grid[np.isfinite(grid) & (grid >= 0.0) & (grid <= f_hi)]
    return np.unique(grid)


def _passivity_report(
    expr: "sympy.Expr",
    fit: Union[RhoFStandardFit, VectorFitResult],
    *,
    rho_at_fit: Optional[float],
) -> dict[str, Any]:
    """Check ``Re Z >= 0`` of the exported formula; see the module docstring."""
    import sympy as sp
    from scipy.optimize import minimize_scalar

    f_sym, rho_sym = sp.Symbol("f"), sp.Symbol("rho")
    func = sp.lambdify((f_sym, rho_sym), expr, modules=["numpy"])

    if isinstance(fit, RhoFStandardFit):
        f_samples = np.asarray(fit.sample_f, dtype=float)
        rho_samples = np.asarray(fit.sample_rho, dtype=float)
        poles = np.zeros(0, dtype=complex)
    else:
        f_samples = np.asarray(fit.fit_frequencies, dtype=float)
        rho_samples = np.array([float(rho_at_fit)])
        poles = np.asarray(fit.poles, dtype=complex)
    f_samples = f_samples[np.isfinite(f_samples) & (f_samples >= 0.0)]
    f_fit_max = float(f_samples.max()) if f_samples.size else 0.0
    f_hi = _PASSIVITY_BAND_FACTOR * f_fit_max
    f_grid = _passivity_frequency_grid(f_samples, f_hi, poles)

    rho_samples = rho_samples[np.isfinite(rho_samples)]
    rho_lo, rho_hi = float(rho_samples.min()), float(rho_samples.max())
    if rho_sym in expr.free_symbols and rho_hi > rho_lo:
        inner = (
            np.geomspace(rho_lo, rho_hi, 9)
            if rho_lo > 0.0
            else np.linspace(rho_lo, rho_hi, 9)
        )
        rho_grid = np.unique(np.concatenate([inner, rho_samples]))
    else:
        rho_grid = np.array([rho_lo])

    def evaluate(f: Any, rho: Any) -> np.ndarray:
        shape = np.broadcast(np.asarray(f), np.asarray(rho)).shape
        with np.errstate(all="ignore"):
            out = np.asarray(func(f, rho), dtype=complex)
        return np.broadcast_to(out, shape)

    Z = evaluate(f_grid[None, :], rho_grid[:, None])
    report: dict[str, Any] = {
        "passed": True,
        "min_re_Z_Ohm": None,
        "f_at_min_Hz": None,
        "rho_at_min_Ohm_m": None,
        "f_checked_Hz": [0.0, f_hi],
        "rho_checked_Ohm_m": [float(rho_grid.min()), float(rho_grid.max())],
        "n_points": int(Z.size),
    }
    finite = np.isfinite(Z)
    if not finite.all():
        i_r, i_f = np.argwhere(~finite)[0]
        report.update(
            passed=False,
            f_at_min_Hz=float(f_grid[i_f]),
            rho_at_min_Ohm_m=float(rho_grid[i_r]),
        )
        return report

    re = Z.real
    i_r, i_f = np.unravel_index(int(np.argmin(re)), re.shape)
    best_re, best_f, rho_worst = float(re[i_r, i_f]), float(f_grid[i_f]), rho_grid[i_r]
    row = re[i_r]
    interior = (row[1:-1] <= row[:-2]) & (row[1:-1] <= row[2:])
    minima = list(np.flatnonzero(interior) + 1) + [0, row.size - 1]
    for i in sorted(set(minima), key=lambda k: row[k])[:10]:
        lo, hi = f_grid[max(i - 1, 0)], f_grid[min(i + 1, row.size - 1)]
        if hi <= lo:
            continue
        res = minimize_scalar(
            lambda x: float(evaluate(x, rho_worst).real),
            bounds=(float(lo), float(hi)),
            method="bounded",
            options={"xatol": 1e-9 * max(float(hi), 1.0)},
        )
        if res.fun < best_re:
            best_re, best_f = float(res.fun), float(res.x)
    scale = float(np.max(np.abs(Z))) if Z.size else 0.0
    report.update(
        passed=bool(best_re >= -_PASSIVITY_REL_TOL * scale),
        min_re_Z_Ohm=best_re,
        f_at_min_Hz=best_f,
        rho_at_min_Ohm_m=float(rho_worst),
    )
    return report


def _passivity_message(name: str, report: dict[str, Any], f_fit_max: float) -> str:
    """Actionable text for :class:`NonPassiveFitWarning` / :class:`NonPassiveFitError`."""
    f_at = report["f_at_min_Hz"]
    where = "inside" if f_at <= f_fit_max else "above"
    rho_text = (
        f", rho = {report['rho_at_min_Ohm_m']:.6g} Ohm m"
        if report["rho_checked_Ohm_m"][0] != report["rho_checked_Ohm_m"][1]
        else ""
    )
    if report["min_re_Z_Ohm"] is None:
        what = f"is not finite at f = {f_at:.6g} Hz{rho_text}"
    else:
        what = (
            f"has Re Z = {report['min_re_Z_Ohm']:.4g} Ohm at f = {f_at:.6g} Hz"
            f"{rho_text}"
        )
    return (
        f"BusType {name!r}: the exported impedance formula is not passive: it "
        f"{what}, {where} the fitted band 0 to {f_fit_max:g} Hz (checked from "
        f"0 to {report['f_checked_Hz'][1]:g} Hz). A passive grounding impedance "
        "has Re Z >= 0 at every frequency. Refit (for a vector fit try another "
        "n_poles or more frequency samples), or pass require_passive=True to "
        "refuse such exports."
    )


FitLike = Union[RhoFStandardFit, VectorFitResult]


def _build_spec(
    fit: FitLike,
    *,
    name: str,
    system_type: str,
    voltage_level: float,
    description: Optional[str],
    decimals: int,
    rho_at_fit: Optional[float],
    electrode_name: Optional[str],
    soil_summary: Optional[str],
    require_passive: bool,
    caller: str,
) -> BusTypeSpec:
    """Assemble a :class:`BusTypeSpec` from any supported fit object."""
    if not isinstance(fit, (RhoFStandardFit, VectorFitResult)):
        raise TypeError(
            f"unsupported fit type {type(fit).__name__}; expected "
            "RhoFStandardFit or VectorFitResult"
        )
    if isinstance(fit, VectorFitResult) and rho_at_fit is None:
        raise ValueError(
            "rho_at_fit is required for VectorFitResult exports — "
            "the resulting BusType is bound to a specific soil "
            "resistivity, and that value goes into samples and metadata."
        )
    if isinstance(decimals, bool) or int(decimals) != decimals or decimals < 1:
        raise ValueError(
            f"{caller}: decimals is the number of significant digits per "
            f"coefficient and must be a positive integer, got {decimals!r}."
        )
    from groundfield.io._formula import parse_impedance_formula

    try:
        if isinstance(fit, RhoFStandardFit):
            formula = _rho_f_standard_to_formula(fit, decimals=int(decimals))
        else:
            formula = _vector_fit_to_formula_in_f(fit, decimals=int(decimals))
        # Self-check: the written formula must load with the safe parser.
        expr = parse_impedance_formula(formula, symbols=("f", "rho"))
    except ValueError as exc:  # non-finite or out-of-range coefficients
        raise ValueError(f"{caller}: cannot export the fit: {exc}") from exc

    if isinstance(fit, RhoFStandardFit):
        samples = _samples_from_rho_f_standard(fit)
        metadata = _metadata_from_rho_f_standard(
            fit,
            electrode_name=electrode_name,
            soil_summary=soil_summary,
        )
        f_fit_max = float(np.max(fit.sample_f)) if np.size(fit.sample_f) else 0.0
    else:
        samples = _samples_from_vector_fit(fit, rho_at_fit=rho_at_fit)
        metadata = _metadata_from_vector_fit(
            fit,
            rho_at_fit=rho_at_fit,
            electrode_name=electrode_name,
            soil_summary=soil_summary,
        )
        f_fit_max = (
            float(np.max(fit.fit_frequencies)) if np.size(fit.fit_frequencies) else 0.0
        )

    passivity = _passivity_report(expr, fit, rho_at_fit=rho_at_fit)
    metadata["passivity"] = passivity
    if not passivity["passed"]:
        message = _passivity_message(str(name), passivity, f_fit_max)
        if require_passive:
            raise NonPassiveFitError(message)
        warnings.warn(message, NonPassiveFitWarning, skip_file_prefixes=(_IO_DIR,))
    return BusTypeSpec(
        name=str(name),
        description=description,
        system_type=str(system_type),
        voltage_level=float(voltage_level),
        impedance_formula=formula,
        samples=samples,
        metadata=metadata,
    )


def to_bustype_dict(
    fit: FitLike,
    *,
    name: str,
    system_type: str,
    voltage_level: float,
    description: Optional[str] = None,
    decimals: int = 12,
    rho_at_fit: Optional[float] = None,
    electrode_name: Optional[str] = None,
    soil_summary: Optional[str] = None,
    require_passive: bool = False,
) -> dict[str, Any]:
    """Convert a fit into the JSON-ready schema-v1 ``dict``.

    Parameters
    ----------
    fit : RhoFStandardFit or VectorFitResult
        The fitted rho-f model.
    name : str
        ``BusType.name``.
    system_type : str
        ``BusType.system_type`` (e.g. ``"TN"``, ``"LV"``).
    voltage_level : float
        ``BusType.voltage_level`` in the unit used by ``groundinsight``.
    description : str, optional
        ``BusType.description``.
    decimals : int, optional
        Significant digits of every floating-point coefficient of the
        formula (1 to 17; larger values are treated as 17). Default 12:
        the formula then reproduces ``fit.evaluate`` to about
        $10^{-11}$ relative (ADR-0008 requires $10^{-9}$); 6 gives a
        short formula with about $10^{-6}$ relative deviation.
    rho_at_fit : float, optional
        Soil resistivity in $\\Omega\\,\\mathrm{m}$ at which a
        :class:`VectorFitResult` was computed. Required for vector
        fits (it goes into ``samples["rho_Ohm_m"]`` and
        ``metadata["rho_at_fit_Ohm_m"]``); ignored for standard-form
        fits. For layered soil pass the value the ``groundinsight``
        ``Bus`` will carry and describe the layers in ``soil_summary``.
    electrode_name, soil_summary : str, optional
        Free-form metadata entries; recommended for traceability.
    require_passive : bool, optional
        If True, raise :class:`NonPassiveFitError` instead of warning
        when the formula has $\\operatorname{Re} Z < 0$ somewhere in
        the checked band. Default False.

    Returns
    -------
    dict
        JSON-ready ``dict`` matching ``schema_version = 1`` of
        ``groundfield.bustype``, including ``metadata["passivity"]``
        with the keys ``passed``, ``min_re_Z_Ohm``, ``f_at_min_Hz``,
        ``rho_at_min_Ohm_m``, ``f_checked_Hz``, ``rho_checked_Ohm_m``
        and ``n_points``.

    Raises
    ------
    ValueError
        If ``rho_at_fit`` is missing for a vector fit, or if the fit has
        non-finite coefficients.
    NonPassiveFitError
        With ``require_passive=True``, if the passivity check fails.
    TypeError
        If ``fit`` is not a supported fit object.

    Warns
    -----
    NonPassiveFitWarning
        If the passivity check fails and ``require_passive`` is False.
    """
    spec = _build_spec(
        fit,
        name=name,
        system_type=system_type,
        voltage_level=voltage_level,
        description=description,
        decimals=decimals,
        rho_at_fit=rho_at_fit,
        electrode_name=electrode_name,
        soil_summary=soil_summary,
        require_passive=require_passive,
        caller="to_bustype_dict",
    )
    return spec.to_dict()


def save_bustype_json(
    fit: FitLike,
    path: Union[str, Path],
    *,
    name: str,
    system_type: str,
    voltage_level: float,
    description: Optional[str] = None,
    decimals: int = 12,
    rho_at_fit: Optional[float] = None,
    electrode_name: Optional[str] = None,
    soil_summary: Optional[str] = None,
    indent: int = 2,
    require_passive: bool = False,
) -> Path:
    """Write the schema-v1 JSON file to disk.

    Performs the conversion and checks of :func:`to_bustype_dict`.
    Nothing is written when the export is refused (missing
    ``rho_at_fit`` for a vector fit, ``require_passive=True`` with a
    failed passivity check).

    Parameters
    ----------
    fit : RhoFStandardFit or VectorFitResult
        The fitted rho-f model.
    path : str or pathlib.Path
        Target file; parent directories are created.
    name, system_type, voltage_level, description
        The ``BusType`` fields, see :func:`to_bustype_dict`.
    decimals : int, optional
        Significant digits per coefficient (default 12), see
        :func:`to_bustype_dict`.
    rho_at_fit : float, optional
        Soil resistivity in $\\Omega\\,\\mathrm{m}$ of a
        :class:`VectorFitResult`; required for vector fits, ignored for
        standard-form fits.
    electrode_name, soil_summary : str, optional
        Free-form metadata entries.
    indent : int, optional
        JSON indentation. Default 2.
    require_passive : bool, optional
        Raise :class:`NonPassiveFitError` instead of warning when the
        passivity check fails. Default False.

    Returns
    -------
    pathlib.Path
        The path the file was actually written to, for chaining.

    Raises
    ------
    ValueError
        If ``rho_at_fit`` is missing for a vector fit.
    NonPassiveFitError
        With ``require_passive=True``, if the passivity check fails.

    Warns
    -----
    NonPassiveFitWarning
        If the passivity check fails and ``require_passive`` is False.
    """
    payload = _build_spec(
        fit,
        name=name,
        system_type=system_type,
        voltage_level=voltage_level,
        description=description,
        decimals=decimals,
        rho_at_fit=rho_at_fit,
        electrode_name=electrode_name,
        soil_summary=soil_summary,
        require_passive=require_passive,
        caller="save_bustype_json",
    ).to_dict()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, sort_keys=False)
        fh.write("\n")
    return p


def load_bustype_json(path: Union[str, Path]) -> BusTypeSpec:
    """Load a ``BusType`` JSON document into a :class:`BusTypeSpec`.

    Validates the schema name and version and parses the formula with
    the allow-list parser (it is never evaluated as code).
    Forward-compatibility for future schema versions is the explicit
    responsibility of this function: when ``schema_version`` differs
    from :data:`SCHEMA_VERSION`, dispatch to the appropriate loader.
    Today only v1 exists.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON file written by :func:`save_bustype_json`.

    Returns
    -------
    BusTypeSpec

    Raises
    ------
    ValueError
        If the schema name or version does not match.
    EvaluateSpecError
        If ``impedance_formula`` is missing or outside the allowed
        formula grammar.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return BusTypeSpec.from_dict(payload)


# ---------------------------------------------------------------------
# Live groundinsight.BusType — Python API path
# ---------------------------------------------------------------------


_GROUNDINSIGHT_IMPORT_HINT = (
    "groundinsight is not installed. Install it with "
    "'pip install groundfield[groundinsight]' or "
    "'pip install groundinsight'."
)


def _import_groundinsight():
    """Lazy import for the ``groundinsight`` package.

    Centralised so the import error message is consistent across the
    Python-API entry points.
    """
    try:
        import groundinsight  # noqa: F401
        from groundinsight.models.core_models import BusType
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(_GROUNDINSIGHT_IMPORT_HINT) from exc
    return BusType


def _to_live_bustype(spec: BusTypeSpec):
    """Build the ``groundinsight.BusType`` for an already validated spec."""
    BusType = _import_groundinsight()
    return BusType(
        name=spec.name,
        description=spec.description,
        system_type=spec.system_type,
        voltage_level=spec.voltage_level,
        impedance_formula=spec.impedance_formula,
    )


def to_bustype(
    fit: FitLike,
    *,
    name: str,
    system_type: str,
    voltage_level: float,
    description: Optional[str] = None,
    decimals: int = 12,
    rho_at_fit: Optional[float] = None,
    electrode_name: Optional[str] = None,
    soil_summary: Optional[str] = None,
    require_passive: bool = False,
):
    """Build a live :class:`groundinsight.BusType` from a fit.

    Performs the same conversion and checks as :func:`to_bustype_dict`,
    but returns a fully-validated ``BusType`` Pydantic instance instead
    of a JSON-ready ``dict``. The arguments are validated before
    ``groundinsight`` is imported lazily; a missing install raises an
    :class:`ImportError` with a clear pointer to the optional install.

    The metadata block (including the passivity check) is **not**
    carried over into the returned ``BusType`` (the ``BusType`` schema
    has no metadata field) — use :func:`to_bustype_dict` or
    :func:`save_bustype_json` if metadata must be persisted.

    Parameters
    ----------
    fit : RhoFStandardFit or VectorFitResult
        The fitted rho-f model.
    name, system_type, voltage_level, description
        The ``BusType`` fields, see :func:`to_bustype_dict`.
    decimals : int, optional
        Significant digits per coefficient (default 12).
    rho_at_fit : float, optional
        Soil resistivity in $\\Omega\\,\\mathrm{m}$ of a
        :class:`VectorFitResult`; required for vector fits, ignored for
        standard-form fits.
    electrode_name, soil_summary : str, optional
        Free-form metadata entries (not part of the live ``BusType``).
    require_passive : bool, optional
        Raise :class:`NonPassiveFitError` instead of warning when the
        passivity check fails. Default False.

    Returns
    -------
    groundinsight.models.core_models.BusType

    Raises
    ------
    ValueError
        If ``rho_at_fit`` is missing for a vector fit.
    NonPassiveFitError
        With ``require_passive=True``, if the passivity check fails.
    ImportError
        If ``groundinsight`` is not installed.

    Warns
    -----
    NonPassiveFitWarning
        If the passivity check fails and ``require_passive`` is False.
    """
    spec = _build_spec(
        fit,
        name=name,
        system_type=system_type,
        voltage_level=voltage_level,
        description=description,
        decimals=decimals,
        rho_at_fit=rho_at_fit,
        electrode_name=electrode_name,
        soil_summary=soil_summary,
        require_passive=require_passive,
        caller="to_bustype",
    )
    return _to_live_bustype(spec)


def save_bustype_to_db(
    fit: FitLike,
    *,
    name: str,
    system_type: str,
    voltage_level: float,
    description: Optional[str] = None,
    decimals: int = 12,
    rho_at_fit: Optional[float] = None,
    electrode_name: Optional[str] = None,
    soil_summary: Optional[str] = None,
    overwrite: bool = False,
    require_passive: bool = False,
) -> None:
    """Build a BusType and save it to the ``groundinsight`` database.

    Writes into the SQLite store opened by ``gi.start_dbsession(path)``;
    requires ``groundinsight`` to be installed *and* a session to be
    active. Performs the same conversion and checks as
    :func:`to_bustype_dict`.

    Parameters
    ----------
    fit : RhoFStandardFit or VectorFitResult
        The fitted rho-f model.
    name, system_type, voltage_level, description
        The ``BusType`` fields, see :func:`to_bustype_dict`.
    decimals : int, optional
        Significant digits per coefficient (default 12).
    rho_at_fit : float, optional
        Soil resistivity in $\\Omega\\,\\mathrm{m}$ of a
        :class:`VectorFitResult`; required for vector fits, ignored for
        standard-form fits.
    electrode_name, soil_summary : str, optional
        Free-form metadata entries (not stored in the database).
    overwrite : bool, optional
        Replace an existing ``BusType`` of the same name. Default False.
    require_passive : bool, optional
        Raise :class:`NonPassiveFitError` instead of warning when the
        passivity check fails. Default False.

    Raises
    ------
    ValueError
        If ``rho_at_fit`` is missing for a vector fit.
    NonPassiveFitError
        With ``require_passive=True``, if the passivity check fails.
    ImportError
        If ``groundinsight`` is not installed.
    RuntimeError
        From ``groundinsight`` if no database session is open.

    Warns
    -----
    NonPassiveFitWarning
        If the passivity check fails and ``require_passive`` is False.
    """
    spec = _build_spec(
        fit,
        name=name,
        system_type=system_type,
        voltage_level=voltage_level,
        description=description,
        decimals=decimals,
        rho_at_fit=rho_at_fit,
        electrode_name=electrode_name,
        soil_summary=soil_summary,
        require_passive=require_passive,
        caller="save_bustype_to_db",
    )
    bustype = _to_live_bustype(spec)
    import groundinsight as gi  # already verified by _to_live_bustype

    gi.save_bustype_to_db(bustype, overwrite=overwrite)


# ---------------------------------------------------------------------
# Diagnostics — keep the math/phys context close to the API
# ---------------------------------------------------------------------


def evaluate_spec(
    spec: BusTypeSpec,
    frequencies: Sequence[float],
    rho: float,
) -> np.ndarray:
    """Evaluate a :class:`BusTypeSpec` at arbitrary frequencies.

    Parses the stored ``impedance_formula`` with the allow-list parser
    (see the module docstring; the text is never evaluated as code) and
    evaluates the resulting expression with NumPy at the provided
    ``rho`` and ``f`` values. Useful for cheap sanity checks that do
    not require ``groundinsight`` to be installed.

    Parameters
    ----------
    spec : BusTypeSpec
        Loaded spec (e.g. from :func:`load_bustype_json`).
    frequencies : sequence of float
        Frequencies in Hz.
    rho : float
        Soil resistivity in $\\Omega\\,\\mathrm{m}$. Ignored if the
        formula does not depend on ``rho`` (vector-fit case).

    Returns
    -------
    np.ndarray
        Complex-valued $Z$ in $\\Omega$ at every requested frequency.

    Raises
    ------
    EvaluateSpecError
        Subclass of :class:`ValueError`; raised when ``spec`` is not a
        :class:`BusTypeSpec` instance, ``impedance_formula`` is empty
        / missing on the spec, or the formula is outside the allowed
        grammar or its size and magnitude bounds (including unknown
        free symbols — anything other than ``f``, ``rho``, ``j`` /
        ``I``, ``pi`` and ``E``). The exception is typed so downstream
        consumers (notably ``groundinsight``) can catch it without
        substring matching on the message. Existing ``except
        ValueError`` blocks keep working because
        :class:`EvaluateSpecError` inherits from :class:`ValueError`.
    """
    import sympy as sp

    # Defensive entry-point validation. Hand-rolled BusTypeSpec-like
    # dicts (e.g. constructed in a CLI helper from a YAML campaign
    # config) routinely miss the impedance_formula; surface the real
    # problem here instead of an opaque error deep in SymPy.
    if not isinstance(spec, BusTypeSpec):
        raise EvaluateSpecError(
            "evaluate_spec: expected a BusTypeSpec instance, got "
            f"{type(spec).__name__}. Build one via load_bustype_json "
            "or BusTypeSpec.from_dict before calling evaluate_spec."
        )
    formula = getattr(spec, "impedance_formula", None)
    if formula is None or not str(formula).strip():
        raise EvaluateSpecError(
            "evaluate_spec: spec.impedance_formula is missing or "
            "empty. A BusTypeSpec must carry a non-empty "
            "impedance_formula string with free symbols in "
            "{f, rho, j}."
        )
    expr = _parse_formula(
        formula if isinstance(formula, str) else str(formula),
        context="evaluate_spec: spec.impedance_formula",
    )
    func = sp.lambdify((sp.Symbol("f"), sp.Symbol("rho")), expr, modules=["numpy"])
    f_arr = np.asarray(frequencies, dtype=float)
    out = np.asarray(func(f_arr, float(rho)), dtype=complex)
    return np.broadcast_to(out, f_arr.shape).copy()


def fit_quality_summary(spec: BusTypeSpec) -> str:
    """One-line human-readable summary of fit quality / kind."""
    md = spec.metadata
    method = md.get("fit_method", "unknown")
    fq = md.get("fit_quality", {})
    rms = fq.get("rms_error_Ohm", math.nan)
    rel = fq.get("rms_relative")
    if rel is not None:
        return (
            f"BusTypeSpec(name={spec.name!r}, method={method}, "
            f"rms={rms:.3e} Ω, rms_rel={rel:.2%})"
        )
    return f"BusTypeSpec(name={spec.name!r}, method={method}, rms={rms:.3e} Ω)"
