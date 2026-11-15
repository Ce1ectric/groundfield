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
  fit belongs to (``rho_at_fit``) is recorded in the samples and
  metadata blocks.

A ``groundinsight`` ``Bus`` carries one ``specific_earth_resistance``,
the ``rho`` of the formula: $\\rho$ for homogeneous soil and, for
layered soil, the **effective resistivity**

$$
\\rho_\\text{eff} = \\rho_1^{\\,1-w}\\,\\rho_2^{\\,w},
\\qquad w = w_0\\,(h_1 / 10\\ \\mathrm{m})^{-\\beta},
$$

with the weights $(w_0, \\beta)$ of the electrode class: the AP1
calibration $w_0 = 0.13$, $\\beta = 0.8$ by default, or a preset of
:data:`~groundfield.soil.EFFECTIVE_RESISTIVITY_PRESETS` such as
``"station"`` or ``"lv_group_2_p10"``
(:func:`~groundfield.soil.effective_resistivity`; ADR-0008), the value
the AP2 network model gives a bus. A vector fit therefore passes
``rho_at_fit=effective_resistivity(soil, preset=...)``, and a
standard-form fit over layered soils is fitted over the effective
resistivities of the solved soils, ``rho_f_standard_from_results(results,
[effective_resistivity(r.soil, preset=...) for r in results])``; both kinds
of BusType then pair with a ``Bus`` whose resistivity uses the same
preset. Every document records this convention as
``metadata["rho_definition"] = "effective_resistivity"`` (documents of
0.15.0 have no such entry) and the preset the caller names,
``metadata["rho_preset"]`` (``rho_preset=...`` of the exporters, ``None``
when no preset is named; user decision Q39; documents of earlier versions
have no such entry and load with ``None``). The exporter cannot check the
number it is given against the preset. :func:`evaluate_spec` and
:func:`bustype_from_spec` take the preset of the bus they serve and emit
:class:`RhoPresetMismatchWarning` when the document names another one (a
preset holds for its electrode class only: the AP1 station paired with
``"lv_group_4_p40"`` misses the field solution by 39.9 % rms). The layers
themselves, and explicit weights $w_0$, $\\beta$, go into
``soil_summary``.

The exporter does not care which impedance was fitted (input
impedance, cluster impedance or the impedance of one electrode); it
exports the function it is given.

Formula text
------------
The formula keeps the structure of the fit (partial fractions for a
vector fit, the standard form term by term) and prints every
floating-point coefficient once, rounded to ``decimals`` significant
digits with Python's correctly rounded float formatting, so the text is
identical on every platform. The expression comes from the same
renderer as :func:`~groundfield.postprocess.vector_fitting.fit_to_sympy`
and :func:`~groundfield.postprocess.rho_f_standard.fit_to_sympy_standard`
(one pole-pairing rule, no ``sympy.simplify``). A coefficient therefore
carries a relative rounding error of at most $\\tfrac12\\,10^{1-N}$ for
$N$ = ``decimals``; with the default $N = 12$ the formula reproduces
``fit.evaluate`` to about $10^{-11}$ relative.

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
beyond them. Before a ``BusType`` is built the exported formula (what
``groundinsight`` will evaluate, coefficients rounded) goes through
:func:`~groundfield.postprocess.vector_fitting.check_passivity`, the one
passivity check of the package, over its default band from 0 Hz to
twice the highest fitted frequency: for a vector fit a dense grid
(linear and logarithmic points, the samples and the midpoints between
them, points around every pole) whose lowest minima are refined by a
bounded scalar minimisation; for the standard form, whose real part is
bilinear in $(\\rho, f)$, the four corners of the fitted range of
``rho`` times the band, which is exact. The report
(:meth:`~groundfield.postprocess.vector_fitting.PassivityReport.as_dict`)
is stored in ``metadata["passivity"]``; a violation emits
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
import numbers
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np

from groundfield.postprocess.rho_f_standard import RhoFStandardFit
from groundfield.postprocess.vector_fitting import PassivityReport, VectorFitResult

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
    "MAX_METADATA_DEPTH",
    "EvaluateSpecError",
    "NonPassiveFitWarning",
    "NonPassiveFitError",
    "RhoPresetMismatchWarning",
    "bustype_from_spec",
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


class RhoPresetMismatchWarning(UserWarning):
    """The BusType was fitted over another effective-resistivity preset.

    Emitted by :func:`evaluate_spec` and :func:`bustype_from_spec` when the
    caller names the preset of its bus (``rho_preset=...``) and the document
    records a different one in ``metadata["rho_preset"]`` (user decision
    Q39): the ``rho`` of the bus then means another resistivity than the
    ``rho`` of the formula.
    """


class NonPassiveFitError(ValueError):
    """Raised instead of :class:`NonPassiveFitWarning` with ``require_passive=True``."""


SCHEMA_NAME = "groundfield.bustype"
SCHEMA_VERSION = 1

#: ``metadata["rho_definition"]`` of every exported document: the ``rho`` of
#: the formula and of the samples is the effective resistivity of the soil
#: (:func:`~groundfield.soil.effective_resistivity`; the resistivity itself
#: for homogeneous soil). Documents of groundfield 0.15.0 have no such entry.
_RHO_DEFINITION = "effective_resistivity"

MAX_METADATA_DEPTH = 32
"""Deepest nesting of objects and lists in the metadata of a BusType document.

The exporters write at most four levels. A deeper document is refused on
load (:class:`ValueError`); the JSON decoder reads thousands of levels, but
every consumer that walks the metadata recursively would fail with
:class:`RecursionError`.
"""
"""Schema name and version of the JSON document produced by this module."""

_IO_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _groundfield_version() -> str:
    """Return ``groundfield.__version__``, the canonical version source.

    Read at export time rather than from the installed distribution
    metadata, which is missing when the package runs from a source
    checkout and stale under an editable install after a release.
    """
    try:
        import groundfield

        return str(groundfield.__version__)
    except (ImportError, AttributeError):  # pragma: no cover - defensive
        return "unknown"


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
        length (``null`` in the JSON document for a value that is not
        finite). Provided so a future tabular ingest path on the
        ``groundinsight`` side can be served immediately.
    metadata
        Free-form metadata dict; populated by the exporters with
        the fit method, fit quality (``rms_error_Ohm``,
        ``rms_relative``, for the standard form ``max_relative``;
        ``null`` when unknown), the impedance the fit represents
        (``"quantity"``: ``"input"``, ``"cluster"``, ``"electrode"``
        or ``null`` for raw samples), how ``rho`` is defined
        (``"rho_definition"``: ``"effective_resistivity"``, see the module
        docstring), fit-method-specific details (poles/residues for
        vector fits, $k_0\\dots k_5$ and the ``"weights"`` for the
        standard form), the passivity check (``"passivity"``), the source
        ``groundfield`` version and a UTC timestamp.
    """

    name: str
    description: Optional[str]
    system_type: str
    voltage_level: float
    impedance_formula: str
    samples: dict[str, list[float]]
    metadata: dict[str, Any]

    @property
    def rho_preset(self) -> Optional[str]:
        """Effective-resistivity preset of the ``rho`` of the formula.

        ``metadata["rho_preset"]`` (user decision Q39): the preset name the
        exporter was given, ``None`` when none was named and for documents
        of earlier versions, which have no such entry.
        """
        value = (
            self.metadata.get("rho_preset") if isinstance(self.metadata, dict) else None
        )
        return value if isinstance(value, str) else None

    # --- serialisation ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready ``dict`` matching schema v1.

        The result is strict JSON: every number that is not finite (a
        sample, a fit-quality figure of a hand-built fit, a description
        that is not a string, ...) is written as ``None`` (JSON ``null``).

        Raises
        ------
        ValueError
            If the metadata nest deeper than :data:`MAX_METADATA_DEPTH`.
        """
        return {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "description": _strict_json(self.description),
            "system_type": self.system_type,
            "voltage_level": float(self.voltage_level),
            "impedance_formula": self.impedance_formula,
            "samples": {
                key: [_finite_or_none(x) for x in self.samples.get(key, [])]
                for key in ("frequency_Hz", "rho_Ohm_m", "Z_real_Ohm", "Z_imag_Ohm")
            },
            "metadata": _strict_json(dict(self.metadata)),
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
            If the payload is not a schema-v1 BusType document: not a
            ``dict``, a wrong schema name or version, a missing ``name``,
            ``system_type`` or ``voltage_level``, a ``name`` or
            ``system_type`` that is an object or a list, a ``description``
            that is neither a string nor ``null``, a ``voltage_level`` that
            is not a finite number, ``samples`` / ``metadata`` that are not
            objects (sample lists hold numbers or ``None``), or metadata
            nested deeper than :data:`MAX_METADATA_DEPTH`.
        EvaluateSpecError
            If ``impedance_formula`` is missing, not a string or a number,
            or outside the allowed formula grammar. The formula is parsed,
            never evaluated as code; the message names the offending
            element.
        """
        if not isinstance(payload, dict):
            raise ValueError(
                f"a BusType document is a JSON object, got {type(payload).__name__}"
            )
        if payload.get("schema") != SCHEMA_NAME:
            raise ValueError(
                f"unexpected schema name: got {payload.get('schema')!r}, "
                f"expected {SCHEMA_NAME!r}"
            )
        try:
            version = int(payload.get("schema_version", 0))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "unsupported schema_version "
                f"{str(payload.get('schema_version'))[:40]!r}; this groundfield "
                f"build only reads v{SCHEMA_VERSION}"
            ) from None
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version}; this groundfield "
                f"build only reads v{SCHEMA_VERSION}"
            )
        for key in ("name", "system_type", "voltage_level"):
            if key not in payload:
                raise ValueError(f"BusType document has no {key!r}")
        for key in ("name", "system_type"):
            if isinstance(payload[key], (dict, list)):
                raise ValueError(
                    f"BusType document: {key!r} must be a string, got a JSON "
                    f"{'object' if isinstance(payload[key], dict) else 'array'}"
                )
        name = str(payload["name"])
        description = payload.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(
                f"BusType document {name!r}: description must be a string or "
                f"null, got {type(description).__name__}"
            )
        formula = payload.get("impedance_formula")
        if formula is None:
            raise EvaluateSpecError(
                f"BusType document {name!r} has no 'impedance_formula'."
            )
        if isinstance(formula, (dict, list)):
            raise EvaluateSpecError(
                f"BusType document {name!r}: impedance_formula must be a string."
            )
        formula = formula if isinstance(formula, str) else str(formula)
        _parse_formula(formula, context=f"BusType document {name!r}")
        samples = payload.get("samples") or {}
        try:
            voltage_level = float(payload["voltage_level"])
        except (TypeError, ValueError, OverflowError):
            voltage_level = math.nan
        if not math.isfinite(voltage_level):
            raise ValueError(
                f"BusType document {name!r}: voltage_level must be a finite "
                f"number, got {str(payload['voltage_level'])[:40]!r}"
            )
        metadata = payload.get("metadata") or {}
        if not isinstance(samples, dict) or not isinstance(metadata, dict):
            raise ValueError(
                f"BusType document {name!r}: samples and metadata must be JSON objects"
            )
        _strict_json(metadata, context=f"BusType document {name!r}: metadata")
        preset = metadata.get("rho_preset")
        if preset is not None and not isinstance(preset, str):
            raise ValueError(
                f"BusType document {name!r}: metadata['rho_preset'] must be a "
                f"preset name or null, got {str(preset)[:40]!r}"
            )
        sample_lists: dict[str, list[Any]] = {}
        for key in ("frequency_Hz", "rho_Ohm_m", "Z_real_Ohm", "Z_imag_Ohm"):
            values = samples.get(key, [])
            if isinstance(values, np.ndarray):
                values = values.tolist()
            if not isinstance(values, (list, tuple)) or not all(
                value is None
                or (isinstance(value, numbers.Real) and not isinstance(value, bool))
                for value in values
            ):
                raise ValueError(
                    f"BusType document {name!r}: samples[{key!r}] must be a list "
                    "of numbers (null for a value that is not finite)"
                )
            sample_lists[key] = list(values)
        return cls(
            name=name,
            description=description,
            system_type=str(payload["system_type"]),
            voltage_level=voltage_level,
            impedance_formula=formula,
            samples=sample_lists,
            metadata=dict(metadata),
        )


# ---------------------------------------------------------------------
# Internal helpers — formula text
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


def _vector_fit_to_formula_in_f(fit: VectorFitResult, *, decimals: int) -> str:
    """Convert a ``VectorFitResult`` to an ``impedance_formula`` string in ``f``.

    The vector fit is in the Laplace variable $s$; ``groundinsight``
    expects the formula in $f$ (Hz) with the imaginary unit ``j``. The
    partial fractions come from the renderer that
    :func:`~groundfield.postprocess.vector_fitting.fit_to_sympy` uses,
    with $s = 2\\pi j f$ at full float64 precision; every coefficient is
    rounded once, to ``decimals`` significant digits, when the text is
    printed. No rationalisation takes place, so the coefficients stay
    readable floats.
    """
    import sympy as sp

    from groundfield.postprocess.vector_fitting import _partial_fractions

    s = 2 * sp.pi * sp.Symbol("j") * sp.Symbol("f")
    expr = _partial_fractions(fit, s)
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
    set used by ``groundinsight``. The expression comes from the renderer
    that :func:`~groundfield.postprocess.rho_f_standard.fit_to_sympy_standard`
    uses; the imaginary unit is printed as ``j`` and every coefficient is
    rounded once, to ``decimals`` significant digits.
    """
    import sympy as sp

    from groundfield.postprocess.rho_f_standard import _standard_expression

    expr = _standard_expression(fit, sp.Symbol("rho"), sp.Symbol("f"))
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


def _finite_or_none(value: Any) -> Optional[float]:
    """``float(value)``, or ``None`` when missing or not finite (strict JSON)."""
    if value is None:
        return None
    try:
        value = float(value)
    except OverflowError:  # an integer beyond the float range
        return None
    return value if math.isfinite(value) else None


def _strict_json(value: Any, *, context: str = "BusTypeSpec metadata") -> Any:
    """Copy of a JSON-like structure with every non-finite float as ``None``.

    Iterative, so the depth is bounded by :data:`MAX_METADATA_DEPTH` and not
    by Python's recursion limit (audit A3SG-04: metadata nested 3 000 levels,
    which the JSON decoder reads, raised :class:`RecursionError` here).

    Raises
    ------
    ValueError
        If ``value`` nests objects and lists deeper than
        :data:`MAX_METADATA_DEPTH`.
    """

    def scalar(item: Any) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    if not isinstance(value, (dict, list, tuple)):
        return scalar(value)
    root: Any = {} if isinstance(value, dict) else []
    stack: list[tuple[Any, Any, int]] = [(value, root, 1)]
    while stack:
        source, target, depth = stack.pop()
        if depth > MAX_METADATA_DEPTH:
            raise ValueError(
                f"{context} is nested deeper than {MAX_METADATA_DEPTH} levels"
            )
        items = source.items() if isinstance(source, dict) else enumerate(source)
        for key, item in items:
            if isinstance(item, (dict, list, tuple)):
                child: Any = {} if isinstance(item, dict) else []
                stack.append((item, child, depth + 1))
            else:
                child = scalar(item)
            if isinstance(target, dict):
                target[key] = child
            else:
                target.append(child)
    return root


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
            "rms_error_Ohm": _finite_or_none(fit.rms_error),
            "rms_relative": _finite_or_none(fit.rms_relative),
        },
        "quantity": fit.quantity,
        "n_poles": int(fit.poles.size),
        "poles": [{"real": float(p.real), "imag": float(p.imag)} for p in fit.poles],
        "residues": [
            {"real": float(r.real), "imag": float(r.imag)} for r in fit.residues
        ],
        "R_inf": float(fit.R_inf),
        "L_inf": float(fit.L_inf),
        "rho_at_fit_Ohm_m": float(rho_at_fit),
        "rho_definition": _RHO_DEFINITION,
        "electrode_name": electrode_name,
        "soil_summary": soil_summary,
        "source": f"groundfield {_groundfield_version()}",
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
            "rms_error_Ohm": _finite_or_none(fit.rms_error),
            "rms_relative": _finite_or_none(fit.rms_relative),
            "max_relative": _finite_or_none(fit.max_relative),
        },
        "coefficients": {name: float(v) for name, v in fit.coefficients.items()},
        "weights": fit.weights,
        "quantity": fit.quantity,
        "rho_definition": _RHO_DEFINITION,
        "electrode_name": electrode_name,
        "soil_summary": soil_summary,
        "source": f"groundfield {_groundfield_version()}",
        "created_at_utc": _utc_now_iso(),
    }


# ---------------------------------------------------------------------
# Internal helpers — passivity
# ---------------------------------------------------------------------


def _formula_evaluator(expr: "sympy.Expr"):
    """NumPy evaluator ``(f, rho) -> Z`` of a parsed impedance formula.

    Built by :func:`~groundfield.io._formula.formula_function` from the
    expression tree, without SymPy's code printer (``lambdify``), which
    evaluates sub-expressions of the unevaluated tree.
    """
    from groundfield.io._formula import formula_function

    return formula_function(expr, ("f", "rho"))


def _passivity_report(
    expr: "sympy.Expr", fit: FitLike, *, rho_at_fit: Optional[float]
) -> PassivityReport:
    """Run :func:`~groundfield.postprocess.vector_fitting.check_passivity` on the formula.

    The one passivity check of the package, evaluated on the exported
    formula (what groundinsight computes) instead of the fit object, over
    its default band (see the module docstring).
    """
    from groundfield.postprocess import vector_fitting

    return vector_fitting._check_passivity(
        fit,
        evaluate=_formula_evaluator(expr),
        rho_at_fit=None if isinstance(fit, RhoFStandardFit) else rho_at_fit,
    )


def _passivity_message(name: str, report: PassivityReport, f_fit_max: float) -> str:
    """Actionable text for :class:`NonPassiveFitWarning` / :class:`NonPassiveFitError`."""
    f_at = report.f_at_min
    where = "inside" if f_at <= f_fit_max else "above"
    rho_text = (
        f", rho = {report.rho_at_min:.6g} Ohm m"
        if report.rho_range is not None and report.rho_range[0] != report.rho_range[1]
        else ""
    )
    if not math.isfinite(report.min_re_Z):
        what = f"is not finite at f = {f_at:.6g} Hz{rho_text}"
    else:
        what = f"has Re Z = {report.min_re_Z:.4g} Ohm at f = {f_at:.6g} Hz{rho_text}"
    return (
        f"BusType {name!r}: the exported impedance formula is not passive: it "
        f"{what}, {where} the fitted band 0 to {f_fit_max:g} Hz (checked from "
        f"0 to {report.f_max:g} Hz). A passive grounding impedance "
        "has Re Z >= 0 at every frequency. Refit (for a vector fit try another "
        "n_poles or more frequency samples), or pass require_passive=True to "
        "refuse such exports."
    )


# ---------------------------------------------------------------------
# Public conversion API
# ---------------------------------------------------------------------


FitLike = Union[RhoFStandardFit, VectorFitResult]


def _check_rho_at_fit(fit: FitLike, rho_at_fit: Optional[float], caller: str) -> None:
    """Validate ``rho_at_fit`` of a vector fit (finding 10.84).

    Standard-form fits ignore ``rho_at_fit`` (documented behaviour).
    """
    if not isinstance(fit, VectorFitResult):
        return
    if rho_at_fit is None:
        raise ValueError(
            f"{caller}: rho_at_fit is required for a VectorFitResult. A vector "
            "fit Z(f) holds for the one soil it was computed on; the BusType "
            "records that resistivity (samples['rho_Ohm_m'], "
            "metadata['rho_at_fit_Ohm_m']) so that it is paired with a "
            "groundinsight Bus of the same specific_earth_resistance. Pass it "
            f"explicitly, e.g. {caller}(fit, ..., rho_at_fit=100.0). For "
            "layered soil the convention is the effective resistivity of the "
            "soil, rho_at_fit=groundfield.effective_resistivity(soil) (the rho "
            "that the groundinsight Bus carries), and the layers are described "
            "with soil_summary=...; standard-form fits (RhoFStandardFit) do not "
            "need it."
        )
    try:
        value = float(rho_at_fit)
    except (TypeError, ValueError):
        value = math.nan
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(
            f"{caller}: rho_at_fit must be a positive, finite soil resistivity "
            f"in Ohm m, got {rho_at_fit!r}."
        )


def _check_rho_preset(rho_preset: Any, caller: str) -> Optional[str]:
    """``rho_preset`` as a preset name or ``None``; ValueError otherwise."""
    if rho_preset is None:
        return None
    from groundfield.soil.effective import EFFECTIVE_RESISTIVITY_PRESETS
    from groundfield.utils.suggest import did_you_mean

    if isinstance(rho_preset, str) and rho_preset in EFFECTIVE_RESISTIVITY_PRESETS:
        return rho_preset
    hint = (
        did_you_mean(rho_preset, EFFECTIVE_RESISTIVITY_PRESETS, n=3)
        if isinstance(rho_preset, str)
        else ""
    )
    raise ValueError(
        f"{caller}: rho_preset must be a preset name of "
        "groundfield.soil.EFFECTIVE_RESISTIVITY_PRESETS (the preset passed to "
        "effective_resistivity(soil, preset=...)) or None, got "
        f"{rho_preset!r}.{hint}"
    )


def _warn_on_preset_mismatch(spec: "BusTypeSpec", rho_preset: Any, caller: str) -> None:
    """Warn when the caller's preset differs from the document's (Q39)."""
    wanted = _check_rho_preset(rho_preset, caller)
    recorded = spec.rho_preset
    if wanted is None or recorded is None or wanted == recorded:
        return
    warnings.warn(
        f"{caller}: BusType {spec.name!r} was fitted over "
        f"effective_resistivity(soil, preset={recorded!r}), but its bus uses "
        f"preset {wanted!r}. The rho of the bus then means another "
        "resistivity than the rho of the formula, and a preset holds for its "
        "electrode class only (the AP1 station with 'lv_group_4_p40': 39.9 % "
        f"rms). Give the bus the effective resistivity of preset {recorded!r}, "
        "or refit the BusType with the preset of the bus.",
        RhoPresetMismatchWarning,
        skip_file_prefixes=(_IO_DIR,),
    )


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
    rho_preset: Optional[str] = None,
) -> BusTypeSpec:
    """Assemble a :class:`BusTypeSpec` from any supported fit object."""
    if not isinstance(fit, (RhoFStandardFit, VectorFitResult)):
        raise TypeError(
            f"unsupported fit type {type(fit).__name__}; expected "
            "RhoFStandardFit or VectorFitResult"
        )
    _check_rho_at_fit(fit, rho_at_fit, caller)
    rho_preset = _check_rho_preset(rho_preset, caller)  # user decision Q39
    try:
        level = float(voltage_level)
    except (TypeError, ValueError, OverflowError):
        level = math.nan
    if not math.isfinite(level):
        raise ValueError(
            f"{caller}: voltage_level must be a finite number, got {voltage_level!r}."
        )
    from groundfield.io._formula import _significant_digits, parse_impedance_formula

    try:
        digits = _significant_digits(decimals)
    except ValueError:
        raise ValueError(
            f"{caller}: decimals is the number of significant digits per "
            f"coefficient and must be a positive integer, got {decimals!r}."
        ) from None
    try:
        if isinstance(fit, RhoFStandardFit):
            formula = _rho_f_standard_to_formula(fit, decimals=digits)
        else:
            formula = _vector_fit_to_formula_in_f(fit, decimals=digits)
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

    # user decision Q39: the preset next to the rho definition
    items = list(metadata.items())
    at = [key for key, _ in items].index("rho_definition") + 1
    metadata = dict(items[:at] + [("rho_preset", rho_preset)] + items[at:])
    passivity = _passivity_report(expr, fit, rho_at_fit=rho_at_fit)
    metadata["passivity"] = passivity.as_dict()
    if not passivity.passed:
        message = _passivity_message(str(name), passivity, f_fit_max)
        if require_passive:
            raise NonPassiveFitError(message)
        warnings.warn(message, NonPassiveFitWarning, skip_file_prefixes=(_IO_DIR,))
    return BusTypeSpec(
        name=str(name),
        description=description,
        system_type=str(system_type),
        voltage_level=level,
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
    rho_preset: Optional[str] = None,
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
        fits. For layered soil the convention is the
        effective resistivity of the soil with the weight preset of the
        electrode class, ``effective_resistivity(soil, preset=...)``
        (:func:`~groundfield.soil.effective_resistivity`), the value a
        ``groundinsight`` ``Bus`` with the same preset carries as
        ``specific_earth_resistance`` (see the module docstring); describe
        the layers and the preset in ``soil_summary``.
    electrode_name, soil_summary : str, optional
        Free-form metadata entries; recommended for traceability.
    require_passive : bool, optional
        If True, raise :class:`NonPassiveFitError` instead of warning
        when the formula has $\\operatorname{Re} Z < 0$ somewhere in
        the checked band. Default False.
    rho_preset : str, optional
        The preset of :func:`~groundfield.soil.effective_resistivity` that
        gave the ``rho`` of the fit (``rho_at_fit`` of a vector fit, the
        ``rho`` samples of a standard form), a key of
        :data:`~groundfield.soil.EFFECTIVE_RESISTIVITY_PRESETS`, recorded as
        ``metadata["rho_preset"]``. Default ``None``: no preset named
        (homogeneous soil, explicit weights, or not stated). A bus that uses
        another preset is then reported by :func:`evaluate_spec` and
        :func:`bustype_from_spec` (user decision Q39).

    Returns
    -------
    dict
        JSON-ready ``dict`` matching ``schema_version = 1`` of
        ``groundfield.bustype`` (strict JSON: a number that is not
        finite, such as the unknown rms error of a hand-built fit, is
        ``None``), including ``metadata["passivity"]``
        with the keys ``passed``, ``min_re_Z_Ohm``, ``f_at_min_Hz``,
        ``rho_at_min_Ohm_m``, ``f_checked_Hz``, ``rho_checked_Ohm_m``
        and ``n_points``.

    Raises
    ------
    ValueError
        If ``rho_at_fit`` is missing for a vector fit or not a positive
        finite number, if ``voltage_level`` is not finite, if the fit
        has non-finite coefficients, or if ``rho_preset`` is not a preset
        name.
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
        rho_preset=rho_preset,
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
    rho_preset: Optional[str] = None,
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
        standard-form fits. For layered soil the convention is the
        effective resistivity ``effective_resistivity(soil, preset=...)``
        (see :func:`to_bustype_dict`).
    electrode_name, soil_summary : str, optional
        Free-form metadata entries.
    indent : int, optional
        JSON indentation. Default 2.
    require_passive : bool, optional
        Raise :class:`NonPassiveFitError` instead of warning when the
        passivity check fails. Default False.
    rho_preset : str, optional
        The preset of :func:`~groundfield.soil.effective_resistivity` behind
        the ``rho`` of the fit, recorded as ``metadata["rho_preset"]``
        (default ``None``: none named), see :func:`to_bustype_dict`.

    Returns
    -------
    pathlib.Path
        The path the file was actually written to, for chaining.

    Raises
    ------
    ValueError
        If ``rho_at_fit`` is missing for a vector fit or invalid.
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
        rho_preset=rho_preset,
    ).to_dict()
    text = json.dumps(payload, indent=indent, sort_keys=False, allow_nan=False)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(text + "\n")
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
        If the file is not valid JSON (also: nested too deeply to be
        read) or not a schema-v1 BusType document, see
        :meth:`BusTypeSpec.from_dict`.
    EvaluateSpecError
        If ``impedance_formula`` is missing or outside the allowed
        formula grammar.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except RecursionError:
            raise ValueError(
                f"BusType document {p.name!r} is nested too deeply to be read"
            ) from None
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
    rho_preset: Optional[str] = None,
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
        standard-form fits. For layered soil the convention is the
        effective resistivity ``effective_resistivity(soil, preset=...)``
        (see :func:`to_bustype_dict`).
    electrode_name, soil_summary : str, optional
        Free-form metadata entries (not part of the live ``BusType``).
    require_passive : bool, optional
        Raise :class:`NonPassiveFitError` instead of warning when the
        passivity check fails. Default False.
    rho_preset : str, optional
        The preset of :func:`~groundfield.soil.effective_resistivity` behind
        the ``rho`` of the fit, recorded as ``metadata["rho_preset"]``
        (default ``None``: none named), see :func:`to_bustype_dict`.

    Returns
    -------
    groundinsight.models.core_models.BusType

    Raises
    ------
    ValueError
        If ``rho_at_fit`` is missing for a vector fit or invalid.
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
        rho_preset=rho_preset,
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
    rho_preset: Optional[str] = None,
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
        standard-form fits. For layered soil the convention is the
        effective resistivity ``effective_resistivity(soil, preset=...)``
        (see :func:`to_bustype_dict`).
    electrode_name, soil_summary : str, optional
        Free-form metadata entries (not stored in the database).
    overwrite : bool, optional
        Replace an existing ``BusType`` of the same name. Default False.
    require_passive : bool, optional
        Raise :class:`NonPassiveFitError` instead of warning when the
        passivity check fails. Default False.
    rho_preset : str, optional
        The preset of :func:`~groundfield.soil.effective_resistivity` behind
        the ``rho`` of the fit, recorded as ``metadata["rho_preset"]``
        (default ``None``: none named), see :func:`to_bustype_dict`.

    Raises
    ------
    ValueError
        If ``rho_at_fit`` is missing for a vector fit or invalid.
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
        rho_preset=rho_preset,
    )
    bustype = _to_live_bustype(spec)
    import groundinsight as gi  # already verified by _to_live_bustype

    gi.save_bustype_to_db(bustype, overwrite=overwrite)


def bustype_from_spec(spec: BusTypeSpec, *, rho_preset: Optional[str] = None):
    """Build a live :class:`groundinsight.BusType` from a loaded document.

    The ``groundinsight`` model of a spec built by the exporters or read
    with :func:`load_bustype_json` (its four scalar fields and the
    formula; the metadata is not carried over, as with :func:`to_bustype`).

    Parameters
    ----------
    spec : BusTypeSpec
        The document.
    rho_preset : str, optional
        The preset of :func:`~groundfield.soil.effective_resistivity` that
        the bus of this BusType uses for its ``specific_earth_resistance``.
        A different preset recorded in the document emits
        :class:`RhoPresetMismatchWarning` (user decision Q39); ``None``
        (default) checks nothing.

    Returns
    -------
    groundinsight.models.core_models.BusType

    Raises
    ------
    TypeError
        If ``spec`` is not a :class:`BusTypeSpec`.
    ValueError
        If ``rho_preset`` is not a preset name.
    ImportError
        If ``groundinsight`` is not installed.

    Warns
    -----
    RhoPresetMismatchWarning
        If ``rho_preset`` differs from the preset the document records.
    """
    if not isinstance(spec, BusTypeSpec):
        raise TypeError(
            f"bustype_from_spec: expected a BusTypeSpec, got {type(spec).__name__}"
        )
    _warn_on_preset_mismatch(spec, rho_preset, "bustype_from_spec")
    return _to_live_bustype(spec)


# ---------------------------------------------------------------------
# Diagnostics — keep the math/phys context close to the API
# ---------------------------------------------------------------------


def evaluate_spec(
    spec: BusTypeSpec,
    frequencies: Sequence[float],
    rho: float,
    *,
    rho_preset: Optional[str] = None,
) -> np.ndarray:
    """Evaluate a :class:`BusTypeSpec` at arbitrary frequencies.

    Parses the stored ``impedance_formula`` with the allow-list parser
    (see the module docstring; the text is never evaluated as code) and
    evaluates the resulting expression with NumPy at the provided
    ``rho`` and ``f`` values, node by node through
    :func:`~groundfield.io._formula.formula_function` (complex
    arithmetic, principal branch for roots and logarithms; SymPy's code
    generation is not used). Useful for cheap sanity checks that do not
    require ``groundinsight`` to be installed.

    Parameters
    ----------
    spec : BusTypeSpec
        Loaded spec (e.g. from :func:`load_bustype_json`).
    frequencies : sequence of float
        Frequencies in Hz.
    rho : float
        Soil resistivity in $\\Omega\\,\\mathrm{m}$. Ignored if the
        formula does not depend on ``rho`` (vector-fit case).
    rho_preset : str, optional
        The preset of :func:`~groundfield.soil.effective_resistivity` that
        gave ``rho`` (the bus's preset). When the document records another
        one (:attr:`BusTypeSpec.rho_preset`), a
        :class:`RhoPresetMismatchWarning` is emitted (user decision Q39);
        ``None`` (default), or a document without a recorded preset, checks
        nothing.

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
        A ``rho_preset`` that is not a preset name raises
        :class:`ValueError`.

    Warns
    -----
    RhoPresetMismatchWarning
        If ``rho_preset`` differs from the preset the document records.
    """
    from groundfield.io._formula import formula_function

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
    _warn_on_preset_mismatch(spec, rho_preset, "evaluate_spec")
    func = formula_function(expr, ("f", "rho"))
    f_arr = np.asarray(frequencies, dtype=float)
    out = func(f_arr, float(rho))
    return np.broadcast_to(out, f_arr.shape).copy()


def fit_quality_summary(spec: BusTypeSpec) -> str:
    """One-line human-readable summary of fit quality / kind.

    Names the fit method, the rms error, the relative rms and worst
    pointwise relative error when the document records them, and the
    impedance the fit represents (``quantity``: ``input``,
    ``cluster`` or ``electrode``) when known.
    """
    md = spec.metadata if isinstance(spec.metadata, dict) else {}
    fq = md.get("fit_quality")
    if not isinstance(fq, dict):
        fq = {}
    method = _summary_label(md.get("fit_method")) or "unknown"
    rms = _summary_figure(fq.get("rms_error_Ohm"))
    rms_text = "unknown" if rms is None else f"{rms:.3e} Ω"
    parts = [f"name={spec.name!r}", f"method={method}", f"rms={rms_text}"]
    rms_relative = _summary_figure(fq.get("rms_relative"))
    if rms_relative is not None:
        parts.append(f"rms_rel={rms_relative:.2%}")
    max_relative = _summary_figure(fq.get("max_relative"))
    if max_relative is not None:
        parts.append(f"max_rel={max_relative:.2%}")
    quantity = _summary_label(md.get("quantity"))
    if quantity is not None:
        parts.append(f"quantity={quantity}")
    preset = spec.rho_preset if isinstance(spec, BusTypeSpec) else None
    if preset is not None:
        parts.append(f"rho_preset={preset}")
    return f"BusTypeSpec({', '.join(parts)})"


def _summary_figure(value: Any) -> Optional[float]:
    """A finite real number of the metadata as ``float``, else ``None``.

    Loaded documents may hold anything there (audit A3SG-04: ``null``, a
    list, an object or a 400-digit integer crashed the summary).
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return _finite_or_none(value)


def _summary_label(value: Any) -> Optional[str]:
    """A string or a number of the metadata as text; ``None`` for anything else."""
    if isinstance(value, str):
        return value
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return str(value)
    return None
