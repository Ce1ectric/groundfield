"""Standard-form rho-f model for the ``groundinsight`` bridge.

The reduced grounding-cluster impedance uses the
**physically-motivated 5-coefficient form**

$$
Z(\\rho, f) \\;=\\; k_1 \\rho \\;+\\; (k_2 + j k_3)\\,f
                  \\;+\\; (k_4 + j k_5)\\,f\\,\\rho,
$$

with $\\rho$ a soil-resistivity parameter (typically the upper-layer
$\\rho_1$ in a fixed-structure 2-layer setup) and $f$ the
frequency. The five real coefficients have direct physical
interpretation:

- $k_1\\,\\rho$       — DC spreading resistance (Dwight-class
  scaling with the dominant local soil resistivity).
- $k_2\\,f$           — purely-inductive coupling that does not
  depend on the soil (e.g. a metallic-cable loop-inductance term).
- $k_3\\,f$           — purely-resistive frequency-dependent term
  (negligible in most quasi-static typical studies).
- $k_4\\,f\\,\\rho$   — Carson-type earth-return resistance: scales
  with both frequency and soil resistivity.
- $k_5\\,f\\,\\rho$   — Carson-type earth-return reactance.

This is **not** a general rational function — it is a fixed
parametric ansatz that captures the leading orders for production-grade
grounding-cluster impedances. It is fitted from a *parametric
family* of `groundfield` runs that span both $\\rho$ and $f$, and
exported as a SymPy expression with two free symbols $\\rho$
(``rho``) and $f$ (``f``) — the ``BusType.impedance_formula``
convention used by ``groundinsight``.

Mathematically the fit is a **linear least-squares** problem in
the five real unknowns:

- Real part: $\\Re Z = k_1\\rho + k_2 f + k_4 f\\rho$
  → 3-feature regression in ($\\rho$, $f$, $f\\rho$).
- Imaginary part: $\\Im Z = k_3 f + k_5 f\\rho$
  → 2-feature regression in ($f$, $f\\rho$).

Both halves are decoupled in the coefficients, so the fit is
unique whenever the sample set spans at least two distinct
$\\rho$ values and at least two distinct frequencies.

References
----------
- The rho-f model is the reduced grey-box representation handed to
  `groundinsight`.
- `groundinsight.BusType.impedance_formula`: the consumer of
  the SymPy expression returned by :func:`fit_to_sympy_standard`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "RhoFStandardFit",
    "fit_rho_f_standard",
    "rho_f_standard_from_results",
    "fit_to_sympy_standard",
]


@dataclass(frozen=True)
class RhoFStandardFit:
    """Result of a standard-form rho-f fit.

    Attributes
    ----------
    k1, k2, k3, k4, k5
        The five real coefficients of the formula
        $Z = k_1\\rho + (k_2 + j k_3)f + (k_4 + j k_5)f\\rho$.
    rms_error
        Root-mean-square residual error in $\\Omega$ over the input
        samples.
    rms_relative
        ``rms_error`` divided by the mean of $|Z|$ over the samples
        — a dimensionless quality figure.
    sample_rho, sample_f, sample_Z
        Original samples used for the fit (1-D arrays of length $N$),
        kept for diagnostics.
    """

    k1: float
    k2: float
    k3: float
    k4: float
    k5: float
    rms_error: float
    rms_relative: float
    sample_rho: np.ndarray
    sample_f: np.ndarray
    sample_Z: np.ndarray

    @property
    def coefficients(self) -> dict[str, float]:
        """Return the five coefficients as a dict."""
        return {
            "k1": self.k1, "k2": self.k2, "k3": self.k3,
            "k4": self.k4, "k5": self.k5,
        }

    def evaluate(
        self, rho: float | np.ndarray, f: float | np.ndarray,
    ) -> complex | np.ndarray:
        """Evaluate the fitted $Z(\\rho, f)$ at arbitrary points."""
        rho = np.asarray(rho, dtype=float)
        f = np.asarray(f, dtype=float)
        return (
            self.k1 * rho
            + (self.k2 + 1j * self.k3) * f
            + (self.k4 + 1j * self.k5) * f * rho
        )


def fit_rho_f_standard(
    rho_samples: Sequence[float],
    f_samples: Sequence[float],
    Z_samples: Sequence[complex],
) -> RhoFStandardFit:
    """Fit the 5-coefficient standard rho-f form via linear LSQ.

    Parameters
    ----------
    rho_samples
        Soil-resistivity values $\\rho$ in $\\Omega\\,\\mathrm{m}$
        for each sample (length $N$).
    f_samples
        Frequencies $f$ in Hz for each sample (length $N$).
    Z_samples
        Complex driving-point impedances $Z$ in $\\Omega$ at each
        $(\\rho, f)$ sample (length $N$).

    Returns
    -------
    RhoFStandardFit

    Raises
    ------
    ValueError
        If sample arrays have inconsistent lengths, or if there are
        fewer than two distinct $\\rho$ values *or* fewer than two
        distinct $f$ values (the LSQ is then under-determined).
    """
    rho = np.asarray(rho_samples, dtype=float)
    f = np.asarray(f_samples, dtype=float)
    Z = np.asarray(Z_samples, dtype=complex)
    if rho.shape != f.shape or rho.shape != Z.shape:
        raise ValueError(
            "rho_samples, f_samples and Z_samples must have identical shape; "
            f"got rho={rho.shape}, f={f.shape}, Z={Z.shape}."
        )
    if rho.size < 4:
        raise ValueError(
            "fit_rho_f_standard needs at least 4 samples; got "
            f"{rho.size}."
        )
    if np.unique(rho).size < 2:
        raise ValueError(
            "fit_rho_f_standard needs at least two distinct rho values "
            "to identify k1 and k4 separately."
        )
    if np.unique(f).size < 2:
        raise ValueError(
            "fit_rho_f_standard needs at least two distinct f values "
            "to identify k2 and k4 separately."
        )

    # Real part: Re(Z) = k1·ρ + k2·f + k4·f·ρ
    A_real = np.column_stack([rho, f, f * rho])
    sol_real, *_ = np.linalg.lstsq(A_real, Z.real, rcond=None)
    k1, k2, k4 = float(sol_real[0]), float(sol_real[1]), float(sol_real[2])

    # Imag part: Im(Z) = k3·f + k5·f·ρ
    A_imag = np.column_stack([f, f * rho])
    sol_imag, *_ = np.linalg.lstsq(A_imag, Z.imag, rcond=None)
    k3, k5 = float(sol_imag[0]), float(sol_imag[1])

    Z_fit = (
        k1 * rho
        + (k2 + 1j * k3) * f
        + (k4 + 1j * k5) * f * rho
    )
    rms = float(np.sqrt(np.mean(np.abs(Z_fit - Z) ** 2)))
    rms_rel = rms / max(float(np.mean(np.abs(Z))), 1e-12)

    return RhoFStandardFit(
        k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
        rms_error=rms, rms_relative=rms_rel,
        sample_rho=rho, sample_f=f, sample_Z=Z,
    )


def rho_f_standard_from_results(
    results: Sequence,
    rhos: Sequence[float],
    electrode_name: str,
) -> RhoFStandardFit:
    """Build the (ρ, f, Z) sample table from a list of FieldResults.

    Use case: parametric soil-resistivity sweep. Run one
    ``Engine.solve`` per soil resistivity, collect the FieldResults
    along with the driving $\\rho$, and pass the lot to this
    function.

    Parameters
    ----------
    results
        List of :class:`groundfield.FieldResult` (one per
        $\\rho$ value).
    rhos
        Soil-resistivity parameter $\\rho$ corresponding to each
        FieldResult (same length).
    electrode_name
        Name of the electrode to extract.

    Returns
    -------
    RhoFStandardFit
    """
    if len(results) != len(rhos):
        raise ValueError(
            f"results and rhos must have same length; "
            f"got {len(results)} vs {len(rhos)}."
        )
    rho_arr: list[float] = []
    f_arr: list[float] = []
    Z_arr: list[complex] = []
    for res, rho_val in zip(results, rhos):
        if electrode_name not in res.electrode_potentials:
            raise KeyError(
                f"electrode '{electrode_name}' not in FieldResult; "
                f"available: {list(res.electrode_potentials)}"
            )
        U = np.asarray(res.electrode_potentials[electrode_name], dtype=complex)
        I = np.asarray(res.electrode_currents[electrode_name], dtype=complex)
        f_local = np.asarray(res.frequencies, dtype=float)
        Z = np.where(np.abs(I) > 0.0, U / np.where(I != 0, I, 1.0), 0.0 + 0.0j)
        rho_arr.extend([float(rho_val)] * len(f_local))
        f_arr.extend(f_local.tolist())
        Z_arr.extend(Z.tolist())
    return fit_rho_f_standard(rho_arr, f_arr, Z_arr)


def fit_to_sympy_standard(fit: RhoFStandardFit, *, decimals: int = 6):
    """Convert a :class:`RhoFStandardFit` to a SymPy expression.

    Returns a :class:`sympy.Expr` in two free symbols ``rho`` and
    ``f`` (both real), in the canonical typical form

    $$
    Z(\\rho, f) \\;=\\; k_1\\rho \\;+\\; (k_2 + j k_3)\\,f
                      \\;+\\; (k_4 + j k_5)\\,f\\,\\rho.
    $$

    The expression is suitable for direct insertion into
    ``groundinsight.BusType.impedance_formula``.
    """
    import sympy as sp

    rho = sp.Symbol("rho", real=True, positive=True)
    f = sp.Symbol("f", real=True, positive=True)
    j = sp.I
    k1 = sp.Float(fit.k1, decimals)
    k2 = sp.Float(fit.k2, decimals)
    k3 = sp.Float(fit.k3, decimals)
    k4 = sp.Float(fit.k4, decimals)
    k5 = sp.Float(fit.k5, decimals)
    expr = k1 * rho + (k2 + j * k3) * f + (k4 + j * k5) * f * rho
    return sp.simplify(expr)
