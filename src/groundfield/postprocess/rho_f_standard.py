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
- $k_2\\,f$           — soil-independent **resistive** frequency
  dependence in $\\Omega/\\mathrm{Hz}$ (skin/proximity losses of the
  metallic parts; negligible in most quasi-static typical studies).
- $k_3\\,f$           — soil-independent **reactive** coupling in
  $\\Omega/\\mathrm{Hz}$, i.e. the loop-inductance term of a
  metallic path; the equivalent inductance is
  $L = k_3 / (2\\pi)$ in H.
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

So $k_2$ and $k_4$ live in the **real** (resistive) half and
$k_3$ and $k_5$ in the **imaginary** (reactive) half — the pairing
implied by the complex factors $(k_2 + j k_3)$ and
$(k_4 + j k_5)$ above.

Both halves are decoupled in the coefficients, so the fit is
unique whenever the sample set spans at least two distinct
$\\rho$ values and at least two distinct frequencies. The two
conditions remove two different collinearities of the design
matrix:

- A single $\\rho = R$ makes the columns $f$ and $f\\rho = R f$
  proportional, so only $k_2 + R\\,k_4$ (resp. $k_3 + R\\,k_5$)
  is identifiable.
- A single $f = F$ makes the columns $\\rho$ and $f\\rho = F\\rho$
  proportional, so only $k_1 + F\\,k_4$ is identifiable.

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
        ``k1`` is the DC spreading-resistance slope in
        $\\Omega/(\\Omega\\,\\mathrm{m})$; ``k2`` (real) and ``k3``
        (imaginary) are the soil-independent resistive resp.
        reactive frequency slopes in $\\Omega/\\mathrm{Hz}$; ``k4``
        (real) and ``k5`` (imaginary) are the Carson-type
        earth-return resistance resp. reactance slopes in
        $\\Omega/(\\mathrm{Hz}\\,\\Omega\\,\\mathrm{m})$.
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
    # Identifiability: a constant rho makes the columns f and f*rho
    # proportional (k2/k4 and k3/k5 confounded); a constant f makes
    # the columns rho and f*rho proportional (k1/k4 confounded).
    if np.unique(rho).size < 2:
        raise ValueError(
            "fit_rho_f_standard needs at least two distinct rho values "
            "to identify k2 and k4 (and k3 and k5) separately."
        )
    if np.unique(f).size < 2:
        raise ValueError(
            "fit_rho_f_standard needs at least two distinct f values "
            "to identify k1 and k4 separately."
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

    Raises
    ------
    ValueError
        If ``results`` and ``rhos`` differ in length, if the
        per-electrode arrays of a result do not match its
        ``frequencies``, or if the electrode carries no current at
        *any* $(\\rho, f)$ sample (then $Z = U/I$ is nowhere
        defined).
    KeyError
        If ``electrode_name`` is absent from a result's
        ``electrode_potentials`` or ``electrode_currents``.

    Notes
    -----
    Samples at which the electrode carries **no current** are
    *masked out* rather than entered as $Z = 0$: a vanishing current
    means the driving-point impedance is undefined, not zero, and a
    single injected zero row pulls the least-squares plane through
    the origin (a 1-in-7 dead DC column biases $k_1$ by tens of
    percent). Masking mirrors
    :func:`groundfield.postprocess.vector_fitting.rho_f_from_field_result`;
    a :class:`UserWarning` reports how many samples were dropped.
    """
    if len(results) != len(rhos):
        raise ValueError(
            f"results and rhos must have same length; "
            f"got {len(results)} vs {len(rhos)}."
        )
    rho_arr: list[float] = []
    f_arr: list[float] = []
    Z_arr: list[complex] = []
    n_dead = 0
    n_total = 0
    for res, rho_val in zip(results, rhos):
        if electrode_name not in res.electrode_potentials:
            raise KeyError(
                f"electrode '{electrode_name}' not in FieldResult "
                f"potentials; available: {list(res.electrode_potentials)}"
            )
        if electrode_name not in res.electrode_currents:
            raise KeyError(
                f"electrode '{electrode_name}' not in FieldResult "
                f"currents; available: {list(res.electrode_currents)}"
            )
        U = np.asarray(res.electrode_potentials[electrode_name], dtype=complex)
        I = np.asarray(res.electrode_currents[electrode_name], dtype=complex)
        f_local = np.asarray(res.frequencies, dtype=float)
        if U.shape != f_local.shape or I.shape != f_local.shape:
            raise ValueError(
                "rho_f_standard_from_results: inconsistent shapes for "
                f"electrode '{electrode_name}': frequencies={f_local.shape}, "
                f"potentials={U.shape}, currents={I.shape}."
            )
        # Review pass 9 (finding F41), mirroring the earlier fix N23 in
        # vector_fitting.rho_f_from_field_result: frequencies at which
        # the electrode carries no current are MASKED, not injected as
        # Z = 0 samples, which used to drag the fit toward the origin.
        alive = np.abs(I) > 0.0
        n_total += int(alive.size)
        n_dead += int((~alive).sum())
        rho_arr.extend([float(rho_val)] * int(alive.sum()))
        f_arr.extend(f_local[alive].tolist())
        Z_arr.extend((U[alive] / I[alive]).tolist())

    if not Z_arr:
        raise ValueError(
            f"rho_f_standard_from_results: electrode '{electrode_name}' "
            f"carries no current at any of the {n_total} (rho, f) samples "
            "— Z = U/I is undefined. Fit the driven electrode instead."
        )
    if n_dead:
        import warnings as _warnings

        _warnings.warn(
            f"rho_f_standard_from_results: electrode '{electrode_name}' "
            f"carries no current at {n_dead} of {n_total} (rho, f) "
            "samples; those samples are excluded from the fit "
            "(historic behaviour injected Z = 0 there).",
            UserWarning,
            stacklevel=2,
        )
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
