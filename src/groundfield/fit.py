"""Reduced rho-f models: the curated namespace ``gf.fit``.

Fits that turn solved impedances ``Z(f)`` (and ``Z(rho, f)``) into the
reduced models exported to groundinsight:

- vector fitting of ``Z(f)`` to a rational model with poles and
  residues (:func:`vector_fit`, :func:`rho_f_from_field_result`,
  :func:`fit_to_sympy`);
- the research standard form ``Z(rho, f)``
  (:func:`fit_rho_f_standard`, :func:`rho_f_standard_from_results`,
  :func:`fit_to_sympy_standard`).

The objects are the same as the flat top-level names
(``gf.fit.vector_fit is gf.vector_fit``). The export itself lives in
``gf.io`` (``to_bustype``, ``save_bustype_json``, …).
"""

from groundfield.postprocess.rho_f_standard import (
    RhoFStandardFit,
    fit_rho_f_standard,
    fit_to_sympy_standard,
    rho_f_standard_from_results,
)
from groundfield.postprocess.vector_fitting import (
    VectorFitConvergenceWarning,
    VectorFitResult,
    VectorFitUnderdeterminedWarning,
    fit_to_sympy,
    rho_f_from_field_result,
    vector_fit,
)

__all__ = [
    # vector fitting Z(f)
    "VectorFitResult",
    "vector_fit",
    "fit_to_sympy",
    "rho_f_from_field_result",
    "VectorFitConvergenceWarning",
    "VectorFitUnderdeterminedWarning",
    # standard form Z(rho, f)
    "RhoFStandardFit",
    "fit_rho_f_standard",
    "fit_to_sympy_standard",
    "rho_f_standard_from_results",
]
