"""Standard-form rho-f model for the ``groundinsight`` bridge.

The reduced grounding-cluster impedance uses the
**physically-motivated standard form**

$$
Z(\\rho, f) \\;=\\; k_0 \\;+\\; k_1 \\rho \\;+\\; (k_2 + j k_3)\\,f
                  \\;+\\; (k_4 + j k_5)\\,f\\,\\rho,
$$

with $\\rho$ a soil-resistivity parameter (typically the upper-layer
$\\rho_1$ in a fixed-structure 2-layer setup) and $f$ the
frequency. The real coefficients have direct physical
interpretation:

- $k_0$             — resistance that does not scale with $\\rho$
  in $\\Omega$: the share of the lower layer in a $\\rho_1$ sweep
  ($R(\\rho_1 \\to 0)$ stays finite, set by $\\rho_2$), or the
  series resistance of conductors in a network. Optional (``k0=True``
  by default since review pass 10, 10.27); real, so that $Z(f = 0)$
  is real.
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

Mathematically the fit is a **weighted linear least-squares** problem:

- Real part: $\\Re Z = k_0 + k_1\\rho + k_2 f + k_4 f\\rho$
  → regression in ($1$, $\\rho$, $f$, $f\\rho$).
- Imaginary part: $\\Im Z = k_3 f + k_5 f\\rho$
  → regression in ($f$, $f\\rho$).

With ``weights="relative"`` (default since review pass 10) every
sample is weighted by $1/|Z_n|$, i.e. the fit minimises the sum of
squared *relative* errors; with ``"uniform"`` it minimises absolute
errors, which concentrates the error on the low-impedance samples
(low $\\rho$), where the high-$\\rho$ samples dominate the residual.

So $k_2$ and $k_4$ live in the **real** (resistive) half and
$k_3$ and $k_5$ in the **imaginary** (reactive) half — the pairing
implied by the complex factors $(k_2 + j k_3)$ and
$(k_4 + j k_5)$ above.

Both halves are decoupled in the coefficients, so the fit is
unique whenever the sample set spans at least two distinct
$\\rho$ values and at least two distinct frequencies (and, with
$k_0$, the real design matrix has full rank). The two conditions
remove two different collinearities of the design matrix:

- A single $\\rho = R$ makes the columns $f$ and $f\\rho = R f$
  proportional, so only $k_2 + R\\,k_4$ (resp. $k_3 + R\\,k_5$)
  is identifiable.
- A single $f = F$ makes the columns $\\rho$ and $f\\rho = F\\rho$
  proportional, so only $k_1 + F\\,k_4$ is identifiable.

Fit quality
-----------
``rms_relative`` (rms error over the mean $|Z|$) is dominated by the
high-impedance samples and hides large relative errors at low
$\\rho$. ``max_relative`` = $\\max_n |Z_\\text{fit} - Z_n| / |Z_n|$ is
the figure to check; above ``max_relative_tol`` (5 %) the fit warns
with :class:`RhoFStandardQualityWarning`. Review pass 10 measured on a
$\\rho_1$ sweep (30…1000 Ω·m) of a 2 m ring in two-layer soil
($\\rho_2$ = 100 Ω·m, $h_1$ = 5 m): v0.15.0 form 27 % worst error at
$\\rho_1$ = 30 (rms_relative 2.6 %), with $k_0$ and relative weights
3.3 %; an AP1-like station (4 m × 4 m loop with three 4 m rods) 43 %
against 6.2 %.

References
----------
- The rho-f model is the reduced grey-box representation handed to
  `groundinsight`.
- `groundinsight.BusType.impedance_formula`: the consumer of
  the SymPy expression returned by :func:`fit_to_sympy_standard`.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from groundfield.postprocess._impedance import Quantity, impedance_series

__all__ = [
    "RhoFStandardFit",
    "RhoFStandardQualityWarning",
    "fit_rho_f_standard",
    "rho_f_standard_from_results",
    "fit_to_sympy_standard",
]

Weights = Literal["uniform", "relative"]


class RhoFStandardQualityWarning(UserWarning):
    """The standard-form fit misses a sample by more than ``max_relative_tol``.

    The standard form is a fixed ansatz; a large pointwise error means it
    cannot represent the sampled :math:`Z(\\rho, f)` (review pass 10,
    10.27). Check ``RhoFStandardFit.max_relative``, restrict the
    :math:`\\rho` range, or fit per :math:`\\rho` with vector fitting.
    """


@dataclass(frozen=True)
class RhoFStandardFit:
    """Result of a standard-form rho-f fit.

    Attributes
    ----------
    k1, k2, k3, k4, k5
        Real coefficients of
        $Z = k_0 + k_1\\rho + (k_2 + j k_3)f + (k_4 + j k_5)f\\rho$.
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
        — a dimensionless figure dominated by the large samples.
    sample_rho, sample_f, sample_Z
        Original samples used for the fit (1-D arrays of length $N$),
        kept for diagnostics.
    k0
        Constant term in $\\Omega$ (0 when fitted with ``k0=False``).
    max_relative
        Worst pointwise relative error
        $\\max_n |Z_\\text{fit}(\\rho_n, f_n) - Z_n| / |Z_n|$ over the
        samples with $Z_n \\ne 0$ (dimensionless); ``nan`` for
        hand-built instances.
    weights
        Weighting used in the fit, ``"uniform"`` or ``"relative"``.
    quantity
        Impedance the samples represent when the fit was built by
        :func:`rho_f_standard_from_results` (``"input"``, ``"cluster"``
        or ``"electrode"``); ``None`` for raw samples.
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
    k0: float = 0.0
    max_relative: float = math.nan
    weights: str = "uniform"
    quantity: str | None = None

    @property
    def coefficients(self) -> dict[str, float]:
        """Return the coefficients ``k0`` … ``k5`` as a dict."""
        return {
            "k0": self.k0,
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "k4": self.k4,
            "k5": self.k5,
        }

    def evaluate(
        self,
        rho: float | np.ndarray,
        f: float | np.ndarray,
    ) -> complex | np.ndarray:
        """Evaluate the fitted $Z(\\rho, f)$ in Ω at arbitrary points.

        Parameters
        ----------
        rho
            Soil resistivity in Ω·m (scalar or array).
        f
            Frequency in Hz (scalar or array, broadcast with ``rho``).
        """
        rho = np.asarray(rho, dtype=float)
        f = np.asarray(f, dtype=float)
        return (
            self.k0
            + self.k1 * rho
            + (self.k2 + 1j * self.k3) * f
            + (self.k4 + 1j * self.k5) * f * rho
        )


def _lstsq(A: np.ndarray, b: np.ndarray, label: str) -> np.ndarray:
    sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < A.shape[1]:
        raise ValueError(
            f"fit_rho_f_standard: the {label} design matrix has rank {rank} < "
            f"{A.shape[1]} unknowns; the samples do not determine the "
            "coefficients (add distinct (rho, f) samples or pass k0=False)."
        )
    return sol


def fit_rho_f_standard(
    rho_samples: Sequence[float],
    f_samples: Sequence[float],
    Z_samples: Sequence[complex],
    *,
    k0: bool = True,
    weights: Weights = "relative",
    max_relative_tol: float = 0.05,
) -> RhoFStandardFit:
    """Fit the standard rho-f form via weighted linear least squares.

    Parameters
    ----------
    rho_samples
        Soil-resistivity values $\\rho$ in $\\Omega\\,\\mathrm{m}$
        for each sample (length $N$).
    f_samples
        Frequencies $f$ in Hz for each sample (length $N$).
    Z_samples
        Complex impedances $Z$ in $\\Omega$ at each $(\\rho, f)$ sample
        (length $N$).
    k0
        Fit the constant term $k_0$ (default ``True`` since review pass
        10, 10.27). ``False`` gives the five-coefficient form of v0.15.0.
    weights
        ``"relative"`` (default since review pass 10): weight each
        sample by $1/|Z_n|$ — minimise relative errors. ``"uniform"``:
        minimise absolute errors (v0.15.0).
    max_relative_tol
        Warn with :class:`RhoFStandardQualityWarning` when
        ``max_relative`` exceeds this value (dimensionless, default
        0.05).

    Returns
    -------
    RhoFStandardFit

    Raises
    ------
    ValueError
        If sample arrays have inconsistent lengths, there are fewer than
        four samples, fewer than two distinct $\\rho$ values *or* fewer
        than two distinct $f$ values, the design matrix is rank
        deficient, ``weights`` is unknown, or ``weights="relative"`` meets
        a sample with $Z = 0$.

    Warns
    -----
    RhoFStandardQualityWarning
        If the worst pointwise relative error exceeds ``max_relative_tol``.

    Notes
    -----
    ``k0=False, weights="uniform"`` reproduces the v0.15.0 coefficients
    bit for bit.
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
            f"fit_rho_f_standard needs at least 4 samples; got {rho.size}."
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
    if weights not in ("uniform", "relative"):
        raise ValueError(f"weights must be 'uniform' or 'relative', got {weights!r}.")
    absZ = np.abs(Z)
    if weights == "relative":
        if np.any(absZ == 0.0):
            raise ValueError(
                "weights='relative' needs |Z| > 0 at every sample; use "
                "weights='uniform' or drop the zero samples."
            )
        w = 1.0 / absZ
    else:
        w = None

    # Real part: Re(Z) = k0 + k1·ρ + k2·f + k4·f·ρ
    cols = [rho, f, f * rho]
    if k0:
        cols.insert(0, np.ones_like(rho))
    A_real = np.column_stack(cols)
    A_imag = np.column_stack([f, f * rho])
    b_real, b_imag = Z.real, Z.imag
    if w is not None:
        A_real, b_real = A_real * w[:, None], b_real * w
        A_imag, b_imag = A_imag * w[:, None], b_imag * w
    sol_real = _lstsq(A_real, b_real, "real-part")
    c0 = float(sol_real[0]) if k0 else 0.0
    k1, k2, k4 = (float(x) for x in sol_real[-3:])

    # Imag part: Im(Z) = k3·f + k5·f·ρ
    sol_imag = _lstsq(A_imag, b_imag, "imaginary-part")
    k3, k5 = float(sol_imag[0]), float(sol_imag[1])

    Z_fit = c0 + k1 * rho + (k2 + 1j * k3) * f + (k4 + 1j * k5) * f * rho
    err = np.abs(Z_fit - Z)
    rms = float(np.sqrt(np.mean(err**2)))
    rms_rel = rms / max(float(np.mean(absZ)), 1e-12)
    nonzero = absZ > 0.0
    max_rel = float(np.max(err[nonzero] / absZ[nonzero])) if nonzero.any() else math.nan
    if max_rel > max_relative_tol:
        worst = int(np.flatnonzero(nonzero)[np.argmax(err[nonzero] / absZ[nonzero])])
        warnings.warn(
            f"fit_rho_f_standard: max_relative = {max_rel:.1%} exceeds "
            f"{max_relative_tol:.1%} (worst sample rho = {rho[worst]:g} Ohm m, "
            f"f = {f[worst]:g} Hz; rms_relative = {rms_rel:.1%}). The standard "
            "form does not represent these samples well.",
            RhoFStandardQualityWarning,
            stacklevel=2,
        )

    return RhoFStandardFit(
        k1=k1,
        k2=k2,
        k3=k3,
        k4=k4,
        k5=k5,
        rms_error=rms,
        rms_relative=rms_rel,
        sample_rho=rho,
        sample_f=f,
        sample_Z=Z,
        k0=c0,
        max_relative=max_rel,
        weights=weights,
    )


def rho_f_standard_from_results(
    results: Sequence,
    rhos: Sequence[float],
    electrode_name: str | None = None,
    *,
    quantity: Quantity = "input",
    source: str | None = None,
    k0: bool = True,
    weights: Weights = "relative",
    max_relative_tol: float = 0.05,
) -> RhoFStandardFit:
    """Build the (ρ, f, Z) sample table from a list of FieldResults and fit.

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
        Soil-resistivity parameter $\\rho$ in Ω·m corresponding to each
        FieldResult (same length).
    electrode_name
        Electrode whose impedance is fitted; required for
        ``quantity="cluster"`` and ``"electrode"``, optional for
        ``"input"`` (see
        :func:`groundfield.postprocess.vector_fitting.rho_f_from_field_result`).
    quantity
        ``"input"`` (default since review pass 10, 10.19): the
        driving-point impedance of the current source.
        ``"cluster"``: the cluster impedance of ``electrode_name``.
        ``"electrode"``: :math:`U_e/I_e` of the named electrode, the
        quantity v0.15.0 fitted.
    source
        Current source for ``quantity="input"``; may be omitted with a
        single current source.
    k0, weights, max_relative_tol
        Forwarded to :func:`fit_rho_f_standard`.

    Returns
    -------
    RhoFStandardFit

    Raises
    ------
    ValueError
        If ``results`` and ``rhos`` differ in length, if the
        per-electrode arrays of a result do not match its
        ``frequencies``, if the denominator current is zero at *every*
        $(\\rho, f)$ sample (then $Z = U/I$ is nowhere defined), or for
        an invalid ``quantity`` / ``electrode_name`` combination.
    KeyError
        If ``electrode_name`` is absent from a result's
        ``electrode_potentials`` or ``electrode_currents``.

    Notes
    -----
    Samples at which the denominator current is **zero** are
    *masked out* rather than entered as $Z = 0$: a vanishing current
    means the impedance is undefined, not zero, and a single injected
    zero row pulls the least-squares plane through the origin (a 1-in-7
    dead DC column biases $k_1$ by tens of percent). Masking mirrors
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
    for res, rho_val in zip(results, rhos, strict=True):
        z_all, i_all = impedance_series(
            res,
            quantity,
            electrode_name=electrode_name,
            source=source,
            caller="rho_f_standard_from_results",
        )
        f_local = np.asarray(res.frequencies, dtype=float)
        # Review pass 9 (finding F41), mirroring the earlier fix N23 in
        # vector_fitting.rho_f_from_field_result: frequencies at which
        # the denominator current vanishes are MASKED, not injected as
        # Z = 0 samples, which used to drag the fit toward the origin.
        alive = np.abs(i_all) > 0.0
        n_total += int(alive.size)
        n_dead += int((~alive).sum())
        rho_arr.extend([float(rho_val)] * int(alive.sum()))
        f_arr.extend(f_local[alive].tolist())
        Z_arr.extend(z_all[alive].tolist())

    label = f"electrode '{electrode_name}'" if quantity != "input" else "the source"
    if not Z_arr:
        raise ValueError(
            f"rho_f_standard_from_results: {label} carries no current at any "
            f"of the {n_total} (rho, f) samples — Z = U/I is undefined. Fit "
            "the driven electrode instead."
        )
    if n_dead:
        import warnings as _warnings

        _warnings.warn(
            f"rho_f_standard_from_results: {label} carries no current at "
            f"{n_dead} of {n_total} (rho, f) samples; those samples are "
            "excluded from the fit (historic behaviour injected Z = 0 there).",
            UserWarning,
            stacklevel=2,
        )
    fit = fit_rho_f_standard(
        rho_arr,
        f_arr,
        Z_arr,
        k0=k0,
        weights=weights,
        max_relative_tol=max_relative_tol,
    )
    return dataclasses.replace(fit, quantity=quantity)


def _standard_expression(fit: RhoFStandardFit, rho, f):
    """Standard-form SymPy expression at full precision (one renderer, 10.73).

    Builds :math:`k_0 + k_1\\rho + (k_2 + j k_3) f + (k_4 + j k_5) f\\rho`
    term by term with float64-precision coefficients (``j`` is
    :data:`sympy.I`), the constant term only when it is non-zero. Used by
    :func:`fit_to_sympy_standard` and by the ``groundinsight`` exporter,
    which round every coefficient once.
    """
    import sympy as sp

    def num(value: float) -> sp.Float:
        return sp.Float(float(value))

    expr = (
        num(fit.k1) * rho
        + (num(fit.k2) + sp.I * num(fit.k3)) * f
        + (num(fit.k4) + sp.I * num(fit.k5)) * f * rho
    )
    if fit.k0 != 0.0:
        expr = num(fit.k0) + expr
    return expr


def fit_to_sympy_standard(fit: RhoFStandardFit, *, decimals: int = 6):
    """Convert a :class:`RhoFStandardFit` to a SymPy expression.

    Returns a :class:`sympy.Expr` in two free symbols ``rho`` and
    ``f`` (both real), in the standard form

    $$
    Z(\\rho, f) \\;=\\; k_0 \\;+\\; k_1\\rho \\;+\\; (k_2 + j k_3)\\,f
                      \\;+\\; (k_4 + j k_5)\\,f\\,\\rho,
    $$

    the constant term only when it is non-zero. The expression is built
    term by term by the renderer the ``groundinsight`` exporter uses
    (review pass 10, 10.73) and is suitable for direct insertion into
    ``groundinsight.BusType.impedance_formula``.

    Parameters
    ----------
    fit
        Standard-form fit.
    decimals
        Significant digits of every coefficient (1 to 17; larger values
        count as 17). Each coefficient is correctly rounded **once**, so
        its relative error is at most :math:`\\tfrac12\\,10^{1-N}` for
        :math:`N` = ``decimals``.

    Returns
    -------
    sympy.Expr

    Raises
    ------
    ValueError
        If ``decimals`` is not a positive integer.
    """
    import sympy as sp

    from groundfield.io._formula import round_coefficients

    rho = sp.Symbol("rho", real=True, positive=True)
    f = sp.Symbol("f", real=True, positive=True)
    return round_coefficients(_standard_expression(fit, rho, f), decimals)
