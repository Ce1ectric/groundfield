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
  in ``f`` only (independent of ``rho``); the underlying soil is
  recorded in the metadata block.

See also
--------
``docs/adr/0008-groundinsight-bridge.md`` — full design
rationale, JSON schema, and validation programme.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as _importlib_metadata
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from groundfield.postprocess.rho_f_standard import (
    RhoFStandardFit,
    fit_to_sympy_standard,
)
from groundfield.postprocess.vector_fitting import (
    VectorFitResult,
    fit_to_sympy,
)


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

__all__ = [
    "BusTypeSpec",
    "to_bustype_dict",
    "to_bustype",
    "save_bustype_json",
    "load_bustype_json",
    "save_bustype_to_db",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
]


SCHEMA_NAME = "groundfield.bustype"
SCHEMA_VERSION = 1
"""Schema name and version of the JSON document produced by this module."""


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
        The fitted ``Z(rho, f)`` expression as a SymPy-compatible
        string. Free symbols: ``f`` (Hz), ``rho``
        ($\\Omega\\,\\mathrm{m}$), and the imaginary unit ``j``.
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
        standard form), the source ``groundfield`` version and a
        UTC timestamp.
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
        """Build a :class:`BusTypeSpec` from a JSON-loaded ``dict``."""
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
        samples = payload.get("samples") or {}
        return cls(
            name=str(payload["name"]),
            description=payload.get("description"),
            system_type=str(payload["system_type"]),
            voltage_level=float(payload["voltage_level"]),
            impedance_formula=str(payload["impedance_formula"]),
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


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _vector_fit_to_formula_in_f(
    fit: VectorFitResult, *, decimals: int
) -> str:
    """Convert a ``VectorFitResult`` to an ``impedance_formula`` string in ``f``.

    The vector fit is in the Laplace variable $s$; ``groundinsight``
    expects the formula in $f$ (Hz) with the imaginary unit ``j``.
    The substitution $s \\to j\\,2\\pi f$ is done symbolically by
    SymPy after :func:`fit_to_sympy` has produced the expression.

    The output is rounded to ``decimals`` digits per coefficient (via
    :func:`fit_to_sympy`) so the printed formula stays compact.
    """
    import sympy as sp

    expr_s = fit_to_sympy(fit, decimals=decimals)
    s = sp.Symbol("s", complex=True)
    f = sp.Symbol("f", real=True, positive=True)
    j = sp.Symbol("j")  # ``groundinsight`` parses ``j`` as the imag. unit
    # Symbolic substitution s -> j * 2 * pi * f. The resulting expression
    # is real on the real-f axis only after groundinsight maps j to I.
    expr_f = expr_s.subs(s, j * 2 * sp.pi * f)
    # Light simplification — keeps the structure but combines obvious
    # constants. ``simplify`` would expand and collapse rational
    # second-order terms produced by ``fit_to_sympy``; we keep the
    # cheaper ``nsimplify`` + ``expand_complex``-free form.
    expr_f = sp.nsimplify(expr_f, rational=False)
    return str(expr_f)


def _rho_f_standard_to_formula(
    fit: RhoFStandardFit, *, decimals: int
) -> str:
    """Convert a ``RhoFStandardFit`` to an ``impedance_formula`` string.

    The standard form already has the canonical ``(rho, f)`` symbol
    set used by ``groundinsight``; only the imaginary unit must be
    spelled ``j`` (not ``I``) for the formula to round-trip through
    ``validate_impedance_formula_value``.
    """
    import sympy as sp

    expr = fit_to_sympy_standard(fit, decimals=decimals)
    # ``fit_to_sympy_standard`` uses ``sp.I``; ``groundinsight`` accepts
    # both ``I`` and ``j``, but the more common convention in the
    # consuming code base is ``j``.
    j = sp.Symbol("j")
    expr = expr.subs(sp.I, j)
    return str(expr)


def _samples_from_vector_fit(
    fit: VectorFitResult, *, rho_at_fit: float,
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
        "poles": [
            {"real": float(p.real), "imag": float(p.imag)} for p in fit.poles
        ],
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
) -> BusTypeSpec:
    """Assemble a :class:`BusTypeSpec` from any supported fit object."""
    if isinstance(fit, RhoFStandardFit):
        formula = _rho_f_standard_to_formula(fit, decimals=decimals)
        samples = _samples_from_rho_f_standard(fit)
        metadata = _metadata_from_rho_f_standard(
            fit,
            electrode_name=electrode_name,
            soil_summary=soil_summary,
        )
    elif isinstance(fit, VectorFitResult):
        if rho_at_fit is None:
            raise ValueError(
                "rho_at_fit is required for VectorFitResult exports — "
                "the resulting BusType is bound to a specific soil "
                "resistivity, and that value goes into samples and metadata."
            )
        formula = _vector_fit_to_formula_in_f(fit, decimals=decimals)
        samples = _samples_from_vector_fit(fit, rho_at_fit=rho_at_fit)
        metadata = _metadata_from_vector_fit(
            fit,
            rho_at_fit=rho_at_fit,
            electrode_name=electrode_name,
            soil_summary=soil_summary,
        )
    else:
        raise TypeError(
            f"unsupported fit type {type(fit).__name__}; expected "
            "RhoFStandardFit or VectorFitResult"
        )
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
) -> dict[str, Any]:
    """Convert a fit into the JSON-ready schema-v1 ``dict``.

    Parameters
    ----------
    fit
        A :class:`~groundfield.postprocess.rho_f_standard.RhoFStandardFit`
        or :class:`~groundfield.postprocess.vector_fitting.VectorFitResult`.
    name, system_type, voltage_level, description
        Match the four scalar fields of ``groundinsight.BusType``.
    decimals
        Number of significant digits to keep per coefficient when
        rendering the SymPy formula. Default 12 — chosen so that the
        round-trip through ``groundinsight.compute_impedance``
        reproduces ``fit.evaluate(...)`` to better than $10^{-9}$
        relative on AP1-typical impedance magnitudes. Reduce to 6 if
        you want a shorter, human-readable formula at the cost of
        ~$10^{-7}$ round-trip drift.
    rho_at_fit
        Soil resistivity at which a :class:`VectorFitResult` was
        produced, in $\\Omega\\,\\mathrm{m}$. Required for vector fits;
        ignored for standard-form fits.
    electrode_name, soil_summary
        Free-form metadata entries; recommended for traceability.

    Returns
    -------
    dict
        JSON-ready ``dict`` matching ``schema_version = 1`` of
        ``groundfield.bustype``.
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
) -> Path:
    """Write the schema-v1 JSON file to disk.

    Returns
    -------
    pathlib.Path
        The path the file was actually written to (as a
        :class:`pathlib.Path`), for chaining.
    """
    payload = to_bustype_dict(
        fit,
        name=name,
        system_type=system_type,
        voltage_level=voltage_level,
        description=description,
        decimals=decimals,
        rho_at_fit=rho_at_fit,
        electrode_name=electrode_name,
        soil_summary=soil_summary,
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, sort_keys=False)
        fh.write("\n")
    return p


def load_bustype_json(path: Union[str, Path]) -> BusTypeSpec:
    """Load a ``BusType`` JSON document into a :class:`BusTypeSpec`.

    Validates the schema name and version. Forward-compatibility for
    future schema versions is the explicit responsibility of this
    function: when ``schema_version`` differs from
    :data:`SCHEMA_VERSION`, dispatch to the appropriate loader. Today
    only v1 exists.
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
):
    """Build a live :class:`groundinsight.BusType` from a fit.

    Performs the same conversion as :func:`to_bustype_dict`, but
    returns a fully-validated ``BusType`` Pydantic instance instead
    of a JSON-ready ``dict``. ``groundinsight`` is imported lazily;
    a missing install raises an :class:`ImportError` with a clear
    pointer to the optional install.

    The metadata block is **not** carried over into the returned
    ``BusType`` (the ``BusType`` schema has no metadata field) — use
    :func:`to_bustype_dict` or :func:`save_bustype_json` if metadata
    must be persisted.
    """
    BusType = _import_groundinsight()
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
    )
    return BusType(
        name=spec.name,
        description=spec.description,
        system_type=spec.system_type,
        voltage_level=spec.voltage_level,
        impedance_formula=spec.impedance_formula,
    )


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
) -> None:
    """Convenience wrapper: build a BusType and save it to the
    ``groundinsight`` SQLite store opened by ``gi.start_dbsession()``.

    Requires ``groundinsight`` to be installed *and* a session to be
    active. Raises ``ImportError`` if the package is missing and a
    plain ``RuntimeError`` (from ``groundinsight``) if no session
    is open.
    """
    bustype = to_bustype(
        fit,
        name=name,
        system_type=system_type,
        voltage_level=voltage_level,
        description=description,
        decimals=decimals,
        rho_at_fit=rho_at_fit,
        electrode_name=electrode_name,
        soil_summary=soil_summary,
    )
    import groundinsight as gi  # already verified by to_bustype above

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

    Re-evaluates the stored ``impedance_formula`` with the provided
    ``rho`` and ``f`` values, using the same SymPy-based path
    ``groundinsight.utils.impedance_calculator`` follows. Useful for
    cheap sanity checks that do not require ``groundinsight`` to be
    installed.

    Parameters
    ----------
    spec
        Loaded spec (e.g. from :func:`load_bustype_json`).
    frequencies
        Frequencies in Hz.
    rho
        Soil resistivity in $\\Omega\\,\\mathrm{m}$. Ignored if the
        formula does not depend on ``rho`` (vector-fit case).

    Returns
    -------
    np.ndarray
        Complex-valued $Z$ at every requested frequency.
    """
    import sympy as sp

    f_sym = sp.Symbol("f", real=True, positive=True)
    rho_sym = sp.Symbol("rho", real=True, positive=True)
    j_sym = sp.Symbol("j")
    expr = sp.sympify(spec.impedance_formula).subs(j_sym, sp.I)
    func = sp.lambdify((f_sym, rho_sym), expr, modules=["numpy"])
    f_arr = np.asarray(frequencies, dtype=float)
    out = func(f_arr, float(rho))
    out = np.asarray(out, dtype=complex)
    if out.ndim == 0:
        out = np.full(f_arr.shape, complex(out), dtype=complex)
    return out


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
    return (
        f"BusTypeSpec(name={spec.name!r}, method={method}, "
        f"rms={rms:.3e} Ω)"
    )
