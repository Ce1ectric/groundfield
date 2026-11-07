"""Vector fitting and SymPy export for the ``rho-f`` reduced model.

This module turns a frequency response computed by ``groundfield``
into a closed-form rational impedance expression that
:mod:`groundinsight` can consume as ``BusType.impedance_formula``.
It is the bridge between the field-grade reference computation
(``groundfield``) and the reduced equivalent-network model
(``groundinsight``).

Mathematical background
-----------------------
For a passive, linear, time-invariant grounding cluster, the
driving-point impedance $Z(s)$ at the feed-in electrode is a
rational function of the Laplace variable $s = j\\omega$. Under
the $f \\le 1\\,\\mathrm{kHz}$ assumption the
function is well approximated by a low-order partial-fraction
expansion

$$
Z(s) \\;\\approx\\; R_\\infty \\;+\\; s\\,L_\\infty \\;+\\;
\\sum_{k=1}^{N_p}\\,\\frac{r_k}{s - p_k},
$$

with poles $p_k$ on the negative real axis (damped RC modes) and
residues $r_k$. Complex-conjugate pole pairs are admissible too —
they produce damped resonant LC-like behaviour. The fit is
constructed by the **Vector Fitting** algorithm of Gustavsen &
Semlyen 1999, which iterates pole locations to a stable solution
and is the de-facto standard for transmission-line and grounding
modelling.

Implementation
--------------
This module implements a clean, dependency-free version of vector
fitting in NumPy:

1. Initial pole guess: linearly spaced on the negative real axis
   between the smallest and largest sampled frequencies.
2. Pole relocation via Sanathanan-Koerner-style least squares with
   a shared denominator (the canonical Gustavsen iteration).
3. Residue solve given the converged poles.
4. Stability enforcement: any pole ending up in the right half
   plane is reflected.

The exported SymPy expression follows the
``groundinsight.BusType.impedance_formula`` convention: a single
free symbol ``s`` and arbitrary numeric constants.

References
----------
- Gustavsen, B. & Semlyen, A. (1999). Rational approximation of
  frequency domain responses by Vector Fitting. *IEEE Trans. Power
  Delivery* **14**(3), 1052–1061.
- Gustavsen, B. (2006). Improving the pole relocating properties
  of vector fitting. *IEEE Trans. Power Delivery* **21**(3),
  1587–1592.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from groundfield.postprocess._impedance import Quantity, impedance_series

__all__ = [
    "VectorFitResult",
    "VectorFitUnderdeterminedWarning",
    "VectorFitConvergenceWarning",
    "VectorFitQualityWarning",
    "PassivityReport",
    "vector_fit",
    "check_passivity",
    "fit_to_sympy",
    "rho_f_from_field_result",
]


class VectorFitQualityWarning(UserWarning):
    """The vector fit is poor, or a sample had to be corrected.

    Emitted (review pass 10, 10.41) when the relative rms error
    ``rms_error / mean|Z|`` exceeds ``rms_tol`` (default 1 %), and when
    the imaginary part of a 0 Hz sample exceeds 0.1 % of its magnitude
    (it is discarded: a rational model with real coefficients is real at
    :math:`s = 0`). Inspect ``VectorFitResult.rms_relative``; try other
    ``n_poles``, more samples, or ``weighting="inverse_magnitude"``.
    """


class VectorFitConvergenceWarning(UserWarning):
    """The pole-relocation loop did not converge.

    Emitted when the relative pole displacement of the last
    relocation step is still above 1 % after ``n_iter`` iterations
    (audit 2026-07-08, WP-B4 — the historic implementation ran a
    fixed number of blind iterations with no convergence check).
    Inspect ``VectorFitResult.pole_shift`` / ``n_iter_used`` and
    consider more iterations, a different ``n_poles``, or
    ``weighting="inverse_magnitude"`` for wide-dynamic-range data.
    """


class VectorFitUnderdeterminedWarning(UserWarning):
    """Vector fit is under- or exactly-determined.

    A vector fit with ``n_poles`` poles has ``2 * n_poles + extras``
    real unknowns (residues plus optional ``R_inf`` / ``L_inf``); the
    real-imag stacking of an input of length ``N`` gives ``2 N``
    equations. ``n_poles >= N`` therefore leads to an under- or
    exactly-determined least-squares problem. The fit *can* succeed
    numerically but is effectively an identity interpolation that
    carries no information beyond the input samples.

    A dedicated warning category lets users opt into / out of the
    diagnostic with ``warnings.simplefilter("once",
    VectorFitUnderdeterminedWarning)``.
    """


@dataclass(frozen=True)
class VectorFitResult:
    """Result of a vector fit.

    Attributes
    ----------
    poles
        Complex poles $p_k$ (1-D array, length $N_p$). Stable poles
        have ``poles.real <= 0``.
    residues
        Complex residues $r_k$ (same shape as ``poles``).
    R_inf
        Real constant offset $R_\\infty$ (DC residual).
    L_inf
        Real proportional-to-$s$ term $L_\\infty$ (high-frequency
        slope). 0 if the fit was performed without it.
    rms_error
        Root-mean-square residual fitting error in $\\Omega$, taken
        over the input frequency points.
    fit_frequencies
        Frequencies (Hz) used for the fit, kept for diagnostics.
    fit_values
        $Z(s)$ values used for the fit (0 Hz samples projected onto
        their real part).
    n_iter_used, pole_shift
        Iterations of the pole relocation and the last relative pole
        displacement.
    rms_relative
        ``rms_error`` divided by the mean of $|Z|$ (dimensionless).
    quantity
        Impedance the samples represent when the fit was built by
        :func:`rho_f_from_field_result` (``"input"``, ``"cluster"`` or
        ``"electrode"``); ``None`` for raw samples.
    """

    poles: np.ndarray
    residues: np.ndarray
    R_inf: float
    L_inf: float
    rms_error: float
    fit_frequencies: np.ndarray
    fit_values: np.ndarray
    # Convergence diagnostics of the pole-relocation loop
    # (audit 2026-07-08, WP-B4). Defaults keep older constructors
    # (e.g. the groundinsight JSON bridge) source-compatible.
    n_iter_used: int = 0
    pole_shift: float = 0.0
    # Relative rms error rms_error / mean|Z| (review pass 10, 10.41);
    # nan for instances built by hand or by the groundinsight bridge.
    rms_relative: float = math.nan
    # Impedance the samples represent when built by
    # rho_f_from_field_result ("input" / "cluster" / "electrode").
    quantity: str | None = None

    def evaluate(self, frequencies: Sequence[float]) -> np.ndarray:
        """Evaluate the fitted $Z(s)$ at arbitrary frequencies."""
        s = 2j * math.pi * np.asarray(frequencies, dtype=float)
        Z = np.full_like(s, self.R_inf, dtype=complex)
        Z = Z + self.L_inf * s
        for p, r in zip(self.poles, self.residues, strict=True):
            Z = Z + r / (s - p)
        return Z


# ---------------------------------------------------------------------
# Vector fitting algorithm
# ---------------------------------------------------------------------


def _initial_poles(
    frequencies: np.ndarray,
    n_poles: int,
    *,
    complex_conj: bool = True,
) -> np.ndarray:
    """Initial pole placement for vector fitting.

    Linearly distributed real poles on $[-\\omega_\\max, -\\omega_\\min]$
    (``complex_conj=False``), or complex-conjugate pairs with real
    parts spread the same way and imaginary parts geometrically
    distributed (``complex_conj=True``).
    """
    omega = 2.0 * math.pi * frequencies
    omega_min = max(float(np.min(omega)), 1e-3)
    omega_max = max(float(np.max(omega)), omega_min * 10.0)
    if complex_conj and n_poles >= 2:
        n_pairs = n_poles // 2
        n_extra = n_poles - 2 * n_pairs
        # Imaginary parts ~ geomspace; real parts ~ -alpha * |im|
        omega_pairs = np.geomspace(omega_min, omega_max, n_pairs)
        alpha = 0.01  # mild damping
        poles_pairs = -alpha * omega_pairs + 1j * omega_pairs
        poles = np.concatenate([poles_pairs, poles_pairs.conj()])
        if n_extra:
            poles = np.append(poles, [-omega_min])
        return poles
    # Pure real placement
    return -np.linspace(omega_min, omega_max, n_poles)


def _enforce_stability(poles: np.ndarray) -> np.ndarray:
    """Reflect any unstable poles to the left half plane."""
    out = poles.copy()
    unstable = out.real > 0
    out[unstable] = -out[unstable].real + 1j * out[unstable].imag
    return out


def _enforce_conjugate_symmetry(poles: np.ndarray) -> np.ndarray:
    """Force the pole list to consist of pure-real or exact-conjugate pairs.

    The eigenvalue solve in vector fitting does not guarantee exactly
    mirrored imaginary parts: numerical noise can leave (p, p*) with
    a sub-percent asymmetry that breaks downstream code that expects
    real-valued time-domain models. We pair off poles by closest
    "approximate conjugate" and replace each pair with two exactly
    conjugate poles (averaging real parts and the |imag| of the pair).
    Poles whose ``|imag| < tol * max(|real|, 1)`` are treated as real
    and snapped to ``imag = 0``.
    """
    out = np.asarray(poles, dtype=complex).copy()
    used = np.zeros(out.size, dtype=bool)
    rel_tol = 1e-2  # poles within 1 % count as approximately conjugate
    for i in range(out.size):
        if used[i]:
            continue
        if abs(out[i].imag) < rel_tol * max(abs(out[i].real), 1.0):
            out[i] = complex(out[i].real, 0.0)
            used[i] = True
            continue
        # Find best approximate conjugate among the remaining poles.
        best_j = -1
        best_score = float("inf")
        for j in range(i + 1, out.size):
            if used[j]:
                continue
            d_real = abs(out[j].real - out[i].real) / max(abs(out[i].real), 1.0)
            d_imag = abs(out[j].imag + out[i].imag) / max(abs(out[i].imag), 1.0)
            score = d_real + d_imag
            if score < best_score:
                best_score = score
                best_j = j
        if best_j < 0:
            # Lone complex pole — make it real (drop imag) for safety.
            out[i] = complex(out[i].real, 0.0)
            used[i] = True
            continue
        # Symmetrise the pair: same real part, exactly opposite imag.
        re_avg = 0.5 * (out[i].real + out[best_j].real)
        im_avg = 0.5 * (abs(out[i].imag) + abs(out[best_j].imag))
        out[i] = complex(re_avg, im_avg)
        out[best_j] = complex(re_avg, -im_avg)
        used[i] = True
        used[best_j] = True
    return out


def _pole_pair_structure(poles: np.ndarray) -> list[tuple]:
    """Classify a conjugate-symmetric pole list into real poles and
    conjugate pairs.

    Returns a list of ``("real", i)`` and ``("pair", i, j)`` entries,
    where entry ``i`` carries ``imag > 0`` and ``j`` its exact
    conjugate. Assumes :func:`_enforce_conjugate_symmetry` ran first.
    """
    structure: list[tuple] = []
    used = np.zeros(poles.size, dtype=bool)
    for i in range(poles.size):
        if used[i]:
            continue
        if abs(poles[i].imag) < 1e-12:
            structure.append(("real", i))
            used[i] = True
            continue
        partner = -1
        for j in range(i + 1, poles.size):
            if used[j]:
                continue
            if (
                abs(poles[j].real - poles[i].real) < 1e-9
                and abs(poles[j].imag + poles[i].imag) < 1e-9
            ):
                partner = j
                break
        if partner < 0:  # defensive — symmetrisation guarantees a partner
            structure.append(("real", i))
            used[i] = True
            continue
        if poles[i].imag > 0:
            structure.append(("pair", i, partner))
        else:
            structure.append(("pair", partner, i))
        used[i] = True
        used[partner] = True
    return structure


def _pair_basis(
    s: np.ndarray,
    poles: np.ndarray,
    structure: list[tuple],
) -> np.ndarray:
    """Real-coefficient partial-fraction basis (Gustavsen pair form).

    Audit 2026-07-08, WP-B4: the historic implementation solved the
    real-stacked LS with one **real** unknown per pole, which forced
    every residue (and every σ-residue steering the pole relocation)
    to be real — silently removing the ``Im r`` degree of freedom of
    complex pole pairs. The standard remedy is the pair-transformed
    real basis: a conjugate pair ``(p, p̄)`` contributes the two
    columns

    .. math::

        \\phi_1 = \\frac{1}{s-p} + \\frac{1}{s-\\bar p},
        \\qquad
        \\phi_2 = \\frac{j}{s-p} - \\frac{j}{s-\\bar p},

    whose **real** coefficients ``(c_1, c_2)`` reconstruct the
    complex residue ``r = c_1 + j c_2`` (and ``\\bar r`` at the
    conjugate pole). Real poles keep their single ``1/(s-p)``
    column.

    Returns the complex ``(N, n_poles)`` matrix whose columns are
    ordered by ``structure`` (pairs occupy two adjacent columns).
    """
    A = np.zeros((s.size, poles.size), dtype=complex)
    col = 0
    for entry in structure:
        if entry[0] == "real":
            A[:, col] = 1.0 / (s - poles[entry[1]])
            col += 1
        else:
            _, i, j = entry
            A[:, col] = 1.0 / (s - poles[i]) + 1.0 / (s - poles[j])
            A[:, col + 1] = 1j / (s - poles[i]) - 1j / (s - poles[j])
            col += 2
    return A


def _coeffs_to_residues(
    coeffs: np.ndarray,
    poles: np.ndarray,
    structure: list[tuple],
) -> np.ndarray:
    """Map the real pair-basis coefficients back to complex residues."""
    residues = np.zeros(poles.size, dtype=complex)
    col = 0
    for entry in structure:
        if entry[0] == "real":
            residues[entry[1]] = complex(coeffs[col], 0.0)
            col += 1
        else:
            _, i, j = entry
            residues[i] = complex(coeffs[col], coeffs[col + 1])
            residues[j] = residues[i].conjugate()
            col += 2
    return residues


def _solve_lstsq(M: np.ndarray, rhs: np.ndarray, poles: np.ndarray) -> np.ndarray:
    """``np.linalg.lstsq`` with a clear ``ValueError`` instead of ``LinAlgError``."""
    try:
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"vector_fit: least-squares solve failed ({exc}) with poles "
            f"{np.array2string(poles, precision=4)}; reduce n_poles or check "
            "the samples."
        ) from exc
    return sol


def vector_fit(
    frequencies: Sequence[float],
    Z_values: Sequence[complex],
    *,
    n_poles: int = 4,
    n_iter: int = 8,
    include_R_inf: bool = True,
    include_L_inf: bool = False,
    complex_poles: bool = True,
    weighting: str = "uniform",
    rms_tol: float = 0.01,
) -> VectorFitResult:
    """Fit a rational $Z(s)$ approximation to a sampled frequency response.

    Implements the Vector Fitting algorithm of Gustavsen & Semlyen
    1999 in its single-output, equal-weight form. The denominator
    polynomial is shared between numerator and denominator
    estimation (Sanathanan-Koerner with iterative pole relocation);
    after ``n_iter`` outer iterations the poles are taken as fixed
    and the residues are solved from a final linear system.

    Parameters
    ----------
    frequencies
        Frequency samples in Hz, length $N$.
    Z_values
        Complex impedance samples $Z(j2\\pi f_k)$ in $\\Omega$,
        length $N$.
    n_poles
        Target number of poles. Complex-conjugate pairs count as
        2 poles each.
    n_iter
        Number of pole-relocation iterations.
    include_R_inf
        Whether to fit a constant offset $R_\\infty$ (DC residual).
    include_L_inf
        Whether to fit a $s\\cdot L_\\infty$ proportional term.
    complex_poles
        ``True`` (default) initialises with complex-conjugate pole
        pairs, suitable for resonant behaviour. Set ``False`` for
        purely-resistive / monotonic responses.
    weighting
        ``"uniform"`` (default, historic behaviour) or
        ``"inverse_magnitude"``: weights every sample by
        ``1/|Z_n|``, the standard remedy when ``|Z|`` spans several
        decades and the unweighted LS would fit only the largest
        magnitudes (audit 2026-07-08, WP-B4).
    rms_tol
        Warn with :class:`VectorFitQualityWarning` when the relative rms
        error ``rms_error / mean|Z|`` exceeds this value (default 0.01).

    Raises
    ------
    ValueError
        If the samples are not finite, no sample has ``f > 0``,
        ``n_poles < 1``, ``weighting`` is unknown, or the least-squares
        problem cannot be solved (instead of numpy's ``LinAlgError``).

    Warns
    -----
    VectorFitQualityWarning
        Relative rms above ``rms_tol``; imaginary part of a 0 Hz sample
        above 0.1 % of its magnitude (discarded).
    VectorFitConvergenceWarning, VectorFitUnderdeterminedWarning
        See the warning classes.

    Notes
    -----
    **0 Hz samples** (review pass 10, 10.41). The model has real
    coefficients, so it is real at :math:`s = 0`. A 0 Hz sample with an
    imaginary part (measured or averaged data) cannot be represented;
    up to v0.15.0 the pole relocation then moved a pole onto the origin
    and the fit collapsed to :math:`Z \\approx 0` at every other
    frequency (relative rms 0.99, 20 of 20 noisy data sets) or numpy
    raised ``LinAlgError``. The 0 Hz samples are now projected onto
    their real part before fitting, which is bit-for-bit neutral for
    real 0 Hz samples (all solver output).

    Since the 2026-07-08 audit (WP-B4) the least-squares systems use
    the **conjugate-pair real basis** (see :func:`_pair_basis`):
    complex pole pairs carry genuinely complex residues, matching
    the cited Gustavsen & Semlyen algorithm. The pole-relocation
    loop additionally monitors the relative pole displacement,
    breaks early below ``1e-8`` and warns
    (:class:`VectorFitConvergenceWarning`) when it is still above
    1 % after ``n_iter`` iterations; the diagnostics are exposed as
    ``VectorFitResult.n_iter_used`` / ``pole_shift``.

    Returns
    -------
    VectorFitResult
    """
    frequencies = np.asarray(frequencies, dtype=float)
    Z = np.asarray(Z_values, dtype=complex)
    if frequencies.shape != Z.shape:
        raise ValueError("frequencies and Z_values must have the same shape")
    if not (np.all(np.isfinite(frequencies)) and np.all(np.isfinite(Z))):
        raise ValueError("vector_fit: frequencies and Z_values must be finite.")
    if not np.any(frequencies > 0.0):
        raise ValueError(
            "vector_fit: needs at least one sample with f > 0; a pure DC "
            "data set has no frequency dependence to fit."
        )
    dc = frequencies == 0.0
    if np.any(dc) and np.any(Z[dc].imag != 0.0):
        # A real-coefficient rational model is real at s = 0: project
        # the 0 Hz samples onto their real part (10.41). Bit-neutral for
        # real DC samples.
        worst = float(np.max(np.abs(Z[dc].imag) / np.maximum(np.abs(Z[dc]), 1e-300)))
        if worst > 1e-3:
            import warnings as _warnings

            _warnings.warn(
                "vector_fit: the 0 Hz sample has an imaginary part of "
                f"{worst:.2%} of its magnitude; it is discarded (the rational "
                "model is real at f = 0). Check the DC value of the data.",
                VectorFitQualityWarning,
                stacklevel=2,
            )
        Z = Z.copy()
        Z[dc] = Z[dc].real
    if n_poles < 1:
        raise ValueError(
            "vector_fit: n_poles must be at least 1, got "
            f"n_poles={n_poles}. n_poles == 0 produces an s-free fit "
            "whose SymPy conversion yields a constant Z(rho, f) "
            "expression without frequency dependence — that is never "
            "the intended behaviour, so this case is rejected at the "
            "API boundary."
        )
    # The previous trigger
    # ``2 * n_poles >= N`` counted 4 real DOFs per pole regardless of
    # whether the search was constrained to conjugate-symmetric pairs.
    # Under ``complex_poles=True`` (the default) the conjugate-symmetry
    # constraint halves the actual free parameters: a pair carries
    # 4 real DOFs (real pole, imag pole, real residue, imag residue),
    # and a lone unpaired real pole carries 2. The corrected formula
    # counts ``n_independent_poles = n_poles // 2 + (n_poles % 2)`` for
    # the pair-constrained search and ``n_poles`` for the all-real one.
    # The strict ``>`` in the conjugate branch keeps a uniquely
    # determined fit (e.g. ``n_poles=2, N=2``) from emitting a false
    # positive.
    if complex_poles:
        n_independent_poles = n_poles // 2 + (n_poles % 2)
        is_underdetermined = 2 * n_independent_poles > frequencies.size
    else:
        is_underdetermined = 2 * n_poles >= frequencies.size
    if is_underdetermined:
        import warnings as _warnings

        if complex_poles:
            ratio_explanation = (
                "2 * n_independent_poles > len(frequencies) is the "
                "critical ratio under complex-conjugate-pair fitting"
            )
        else:
            ratio_explanation = (
                "2 * n_poles >= len(frequencies) is the critical "
                "ratio under all-real-pole fitting"
            )
        _warnings.warn(
            "vector_fit: n_poles="
            f"{n_poles} with len(frequencies)={frequencies.size} "
            "leads to an under- or exactly-determined least-squares "
            "fit (each pole contributes residue + pole-location "
            f"DOFs; {ratio_explanation}). The numerical solve may "
            "succeed but produces an identity interpolation of the "
            "input samples that carries no information beyond the "
            "raw frequency response. Reduce n_poles to "
            f"{max(1, frequencies.size // 2 - 1)} or fewer, or supply "
            "more frequency samples.",
            VectorFitUnderdeterminedWarning,
            stacklevel=2,
        )
    if weighting not in ("uniform", "inverse_magnitude"):
        raise ValueError(
            "vector_fit: weighting must be 'uniform' or "
            f"'inverse_magnitude', got {weighting!r}."
        )
    s = 2j * math.pi * frequencies
    if weighting == "inverse_magnitude":
        w = 1.0 / np.maximum(np.abs(Z), 1e-12 * float(np.max(np.abs(Z))))
    else:
        w = np.ones(s.size)

    poles = _initial_poles(frequencies, n_poles, complex_conj=complex_poles)
    poles = _enforce_conjugate_symmetry(poles)

    # Outer loop: pole relocation. The system is, schematically,
    #   sum_k r_k / (s_n - p_k) + d + s_n*h - sum_k r'_k Z_n / (s_n - p_k)
    #     = Z_n
    # where {r_k, d, h, r'_k} are the unknowns and the new poles are
    # the zeros of the denominator polynomial 1 + sum r'_k/(s-p_k).
    # Both the numerator residues r_k and the sigma residues r'_k use
    # the conjugate-pair real basis (audit 2026-07-08, WP-B4), so
    # complex pole pairs carry genuinely complex residues.
    extras = (1 if include_R_inf else 0) + (1 if include_L_inf else 0)
    n_iter_used = 0
    pole_shift = float("inf")
    for _ in range(max(1, n_iter)):
        structure = _pole_pair_structure(poles)
        basis = _pair_basis(s, poles, structure)  # (N, n_poles)
        A = np.zeros((s.size, n_poles + extras + n_poles), dtype=complex)
        A[:, :n_poles] = basis
        col = n_poles
        if include_R_inf:
            A[:, col] = 1.0
            col += 1
        if include_L_inf:
            A[:, col] = s
            col += 1
        A[:, col : col + n_poles] = -Z[:, None] * basis

        # Solve with real-imag stacking (least-squares), optionally
        # row-weighted.
        Aw = A * w[:, None]
        M = np.vstack([Aw.real, Aw.imag])
        rhs = np.concatenate([(w * Z).real, (w * Z).imag])
        sol = _solve_lstsq(M, rhs, poles)

        # Reconstruct the complex sigma residues from the pair basis.
        sigma_residues = _coeffs_to_residues(
            sol[n_poles + extras :],
            poles,
            structure,
        )
        # New poles: eigenvalues of A_p - b·c^T where A_p = diag(poles),
        # b = ones, c = sigma_residues. (Standard Gustavsen formulation.)
        Ap = np.diag(poles)
        b = np.ones(n_poles)
        try:
            new_poles = np.linalg.eigvals(Ap - np.outer(b, sigma_residues))
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"vector_fit: pole relocation failed ({exc}); reduce n_poles or "
                "check the samples."
            ) from exc
        new_poles = _enforce_stability(new_poles)
        new_poles = _enforce_conjugate_symmetry(new_poles)
        # Sort for deterministic order (deepest |imag| first).
        sorted_idx = np.argsort(-np.abs(new_poles.imag))
        new_poles = new_poles[sorted_idx]
        n_iter_used += 1
        scale = float(np.max(np.abs(poles))) or 1.0
        pole_shift = (
            float(np.max(np.abs(np.sort_complex(new_poles) - np.sort_complex(poles))))
            / scale
        )
        poles = new_poles
        if pole_shift < 1e-8:
            break
    if pole_shift > 1e-2:
        import warnings as _warnings

        _warnings.warn(
            f"vector_fit: pole relocation did not converge within "
            f"n_iter={n_iter} iterations (last relative pole shift "
            f"{pole_shift:.2e}). The returned fit may sit in a limit "
            "cycle; inspect rms_error, increase n_iter, or adjust "
            "n_poles / weighting.",
            VectorFitConvergenceWarning,
            stacklevel=2,
        )

    # Final residue solve with the converged poles (pair basis —
    # residues at conjugate poles are exact conjugates by
    # construction, no post-hoc repair required).
    structure = _pole_pair_structure(poles)
    basis = _pair_basis(s, poles, structure)
    A = np.zeros((s.size, n_poles + extras), dtype=complex)
    A[:, :n_poles] = basis
    col = n_poles
    if include_R_inf:
        A[:, col] = 1.0
        col += 1
    if include_L_inf:
        A[:, col] = s
        col += 1
    Aw = A * w[:, None]
    M = np.vstack([Aw.real, Aw.imag])
    rhs = np.concatenate([(w * Z).real, (w * Z).imag])
    sol = _solve_lstsq(M, rhs, poles)
    residues = _coeffs_to_residues(sol[:n_poles], poles, structure)
    R_inf = float(sol[n_poles]) if include_R_inf else 0.0
    L_inf = float(sol[n_poles + (1 if include_R_inf else 0)]) if include_L_inf else 0.0

    Z_fit = (
        np.full_like(s, R_inf, dtype=complex)
        + L_inf * s
        + np.sum(residues[:, None] / (s[None, :] - poles[:, None]), axis=0)
    )
    rms = float(np.sqrt(np.mean(np.abs(Z_fit - Z) ** 2)))
    rms_rel = rms / max(float(np.mean(np.abs(Z))), 1e-300)
    if rms_rel > rms_tol:
        import warnings as _warnings

        _warnings.warn(
            f"vector_fit: relative rms error {rms_rel:.2%} exceeds "
            f"{rms_tol:.2%} (n_poles={n_poles}). The rational model does not "
            "represent the samples; try another n_poles, more samples or "
            "weighting='inverse_magnitude'.",
            VectorFitQualityWarning,
            stacklevel=2,
        )

    return VectorFitResult(
        rms_relative=rms_rel,
        n_iter_used=n_iter_used,
        pole_shift=pole_shift,
        poles=poles,
        residues=residues,
        R_inf=R_inf,
        L_inf=L_inf,
        rms_error=rms,
        fit_frequencies=frequencies,
        fit_values=Z,
    )


# ---------------------------------------------------------------------
# Passivity check
# ---------------------------------------------------------------------


_PASSIVITY_BAND_FACTOR = 2.0
"""Default band of :func:`check_passivity`: 0 Hz to this factor times the
highest fitted frequency."""

_PASSIVITY_REL_TOL = 1e-9
"""``Re Z >= -tol * max|Z|`` counts as passive (round-off and coefficient
rounding of an exported formula)."""

_PASSIVITY_N_REFINE = 10
"""Number of the lowest local grid minima refined by scalar minimisation."""


@dataclass(frozen=True)
class PassivityReport:
    """Outcome of :func:`check_passivity`.

    Attributes
    ----------
    passed
        ``True`` when :math:`\\Re Z \\ge -10^{-9} \\max|Z|` everywhere on the
        checked band and every value is finite.
    min_re_Z
        Smallest :math:`\\Re Z` found, in Ω; ``nan`` if :math:`Z` is not
        finite somewhere (``f_at_min`` then names the frequency).
    f_at_min
        Frequency of ``min_re_Z`` in Hz.
    f_max
        Upper end of the checked band in Hz (the band starts at 0 Hz).
    rho_at_min
        Soil resistivity of ``min_re_Z`` in Ω·m: the corner of the checked
        :math:`\\rho` range for a standard-form fit, the resistivity the
        fit belongs to for an exported vector fit, else ``None``.
    rho_range
        Checked resistivity range ``(min, max)`` in Ω·m, or ``None``.
    n_points
        Number of grid points evaluated (before the refinement).
    """

    passed: bool
    min_re_Z: float
    f_at_min: float
    f_max: float
    rho_at_min: float | None = None
    rho_range: tuple[float, float] | None = None
    n_points: int = 0

    def as_dict(self) -> dict[str, object]:
        """Strict-JSON summary; the ``metadata["passivity"]`` block of an export.

        Keys: ``passed``, ``min_re_Z_Ohm`` (``None`` when not finite),
        ``f_at_min_Hz``, ``rho_at_min_Ohm_m``, ``f_checked_Hz``
        (``[0, f_max]``), ``rho_checked_Ohm_m`` (``[min, max]`` or
        ``None``) and ``n_points``.
        """
        finite = math.isfinite(self.min_re_Z)
        return {
            "passed": bool(self.passed),
            "min_re_Z_Ohm": float(self.min_re_Z) if finite else None,
            "f_at_min_Hz": float(self.f_at_min),
            "rho_at_min_Ohm_m": (
                None if self.rho_at_min is None else float(self.rho_at_min)
            ),
            "f_checked_Hz": [0.0, float(self.f_max)],
            "rho_checked_Ohm_m": (
                None if self.rho_range is None else [float(x) for x in self.rho_range]
            ),
            "n_points": int(self.n_points),
        }


def check_passivity(
    fit,
    f_max: float | None = None,
    *,
    n: int = 2001,
    rho_range: tuple[float, float] | None = None,
) -> PassivityReport:
    """Check :math:`\\Re Z(j 2\\pi f) \\ge 0` of a fitted rho-f model on a band.

    A grounding impedance is the driving-point impedance of a passive
    one-port, so its real part is non-negative at every frequency,

    .. math::

        \\operatorname{Re} Z(j\\,2\\pi f) \\;\\ge\\; 0 \\qquad (f \\ge 0).

    A rational fit can violate this *between* or *beyond* its samples:
    review pass 10 found a 4-pole fit of passive data (rms 4.4 mΩ, no
    warning) with :math:`\\Re Z = -0.019\\,\\Omega` at 5.5 Hz, between the
    0 Hz and 50 Hz samples (10.40 / 10.41). This is the one passivity
    check of the package: the groundinsight exporter runs the same
    implementation on the formula it writes.

    Parameters
    ----------
    fit
        :class:`VectorFitResult` or
        :class:`~groundfield.postprocess.rho_f_standard.RhoFStandardFit`.
    f_max
        Upper end of the band in Hz (the band starts at 0 Hz). Default:
        twice the highest fitted frequency, the band of the export check.
    n
        Number of points of the linear frequency grid of a vector fit
        (``>= 2``). The grid also holds ``3 n // 10`` logarithmically
        spaced points from :math:`10^{-6} f_\\max`, the samples and the
        midpoints between them, and 161 points within eight half-widths
        :math:`|\\Re p|` of every pole frequency :math:`|\\Im p|/2\\pi`,
        which resolve resonances narrower than any fixed grid. The ten
        lowest local minima of the grid are refined by a bounded scalar
        minimisation.
    rho_range
        ``(rho_min, rho_max)`` in Ω·m for a standard-form fit; default
        the sampled range. :math:`\\Re Z = k_0 + k_1\\rho + k_2 f +
        k_4 f\\rho` is bilinear, so its minimum over the box
        :math:`[\\rho_\\min, \\rho_\\max] \\times [0, f_\\max]` lies at a
        corner and the check is exact.

    Returns
    -------
    PassivityReport
        ``passed`` is ``True`` when every evaluated :math:`Z` is finite and
        :math:`\\min \\Re Z \\ge -10^{-9} \\max |Z|` (the tolerance absorbs
        round-off and the coefficient rounding of an exported formula).

    Raises
    ------
    TypeError
        For an unsupported fit type.
    ValueError
        For ``f_max <= 0``, ``n < 2`` or an invalid ``rho_range``.
    """
    return _check_passivity(fit, f_max=f_max, n=n, rho_range=rho_range)


def _passivity_frequency_grid(
    f_samples: np.ndarray, f_hi: float, poles: np.ndarray, n: int
) -> np.ndarray:
    """Frequencies in Hz at which :func:`check_passivity` evaluates ``Re Z``."""
    parts = [np.array([0.0, f_hi])]
    if f_hi > 0.0:
        parts += [
            np.linspace(0.0, f_hi, n),
            np.geomspace(f_hi * 1e-6, f_hi, max(2, (3 * n) // 10)),
        ]
    samples = np.unique(f_samples)
    parts += [samples, 0.5 * (samples[1:] + samples[:-1])]
    t = np.linspace(-8.0, 8.0, 161)
    for p in np.asarray(poles, dtype=complex).ravel():
        width = max(abs(p.real), 1e-9 * max(abs(p), 1.0))
        parts.append((abs(p.imag) + width * t) / (2.0 * math.pi))
    grid = np.concatenate(parts)
    grid = grid[np.isfinite(grid) & (grid >= 0.0) & (grid <= f_hi)]
    return np.unique(grid)


def _check_passivity(
    fit,
    *,
    f_max: float | None = None,
    n: int = 2001,
    rho_range: tuple[float, float] | None = None,
    evaluate=None,
    rho_at_fit: float | None = None,
) -> PassivityReport:
    """Implementation of :func:`check_passivity`, shared with the exporter.

    ``evaluate(f, rho)`` replaces the fit's own evaluation (the exporter
    passes the rounded formula it writes); ``rho_at_fit`` is the
    resistivity in Ω·m a vector fit belongs to and only labels the report.
    """
    from groundfield.postprocess.rho_f_standard import RhoFStandardFit

    if isinstance(fit, RhoFStandardFit):
        bilinear = True
        f_samples = np.asarray(fit.sample_f, dtype=float)
        poles = np.zeros(0, dtype=complex)
        if rho_range is None:
            rho_range = (float(np.min(fit.sample_rho)), float(np.max(fit.sample_rho)))
        r_lo, r_hi = (float(x) for x in rho_range)
        if not (0.0 <= r_lo <= r_hi) or not math.isfinite(r_hi):
            raise ValueError(
                f"rho_range must satisfy 0 <= min <= max, got {rho_range!r}."
            )
        rho_range = (r_lo, r_hi)
        if evaluate is None:

            def evaluate(f, rho):
                return fit.evaluate(rho, f)

    elif isinstance(fit, VectorFitResult):
        bilinear = False
        f_samples = np.asarray(fit.fit_frequencies, dtype=float)
        poles = np.asarray(fit.poles, dtype=complex)
        rho_range = None if rho_at_fit is None else (rho_at_fit, rho_at_fit)
        if evaluate is None:

            def evaluate(f, rho):
                return fit.evaluate(f)

    else:
        raise TypeError(
            "check_passivity expects a VectorFitResult or a RhoFStandardFit, "
            f"got {type(fit).__name__}."
        )
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n!r}.")
    f_samples = f_samples[np.isfinite(f_samples) & (f_samples >= 0.0)]
    if f_max is None:
        f_top = _PASSIVITY_BAND_FACTOR * (
            float(f_samples.max()) if f_samples.size else 0.0
        )
    else:
        f_top = float(f_max)
        if not (f_top > 0.0) or not math.isfinite(f_top):
            raise ValueError(f"f_max must be finite and > 0, got {f_max!r}.")

    def z_at(f, rho):
        with np.errstate(all="ignore"):
            out = np.asarray(evaluate(f, rho), dtype=complex)
        return np.broadcast_to(out, np.broadcast(np.asarray(f), np.asarray(rho)).shape)

    if bilinear:
        # Re Z is bilinear in (rho, f): its minimum lies at a corner.
        rho_pts = np.array([rho_range[0], rho_range[1]])
        f_pts = np.unique([0.0, f_top])
        rho_grid, f_grid = np.meshgrid(rho_pts, f_pts, indexing="ij")
        rho_grid, f_grid = rho_grid.ravel(), f_grid.ravel()
        Z = z_at(f_grid, rho_grid)
    else:
        f_grid = _passivity_frequency_grid(f_samples, f_top, poles, int(n))
        rho_value = np.nan if rho_range is None else rho_range[0]
        rho_grid = np.full(f_grid.size, rho_value)
        Z = z_at(f_grid, rho_value)

    def label(i: int) -> float | None:
        return None if rho_range is None else float(rho_grid[i])

    finite = np.isfinite(Z)
    if not finite.all():
        i = int(np.flatnonzero(~finite)[0])
        return PassivityReport(
            passed=False,
            min_re_Z=math.nan,
            f_at_min=float(f_grid[i]),
            f_max=f_top,
            rho_at_min=label(i),
            rho_range=rho_range,
            n_points=int(Z.size),
        )
    re = Z.real
    i_min = int(np.argmin(re))
    best_re, best_f = float(re[i_min]), float(f_grid[i_min])
    if not bilinear and re.size > 2:
        from scipy.optimize import minimize_scalar

        interior = list(
            np.flatnonzero((re[1:-1] <= re[:-2]) & (re[1:-1] <= re[2:])) + 1
        )
        candidates = sorted(set(interior) | {0, re.size - 1}, key=lambda k: re[k])
        for i in candidates[:_PASSIVITY_N_REFINE]:
            lo, hi = f_grid[max(i - 1, 0)], f_grid[min(i + 1, re.size - 1)]
            if hi <= lo:
                continue
            res = minimize_scalar(
                lambda x: float(z_at(np.array([x]), rho_value)[0].real),
                bounds=(float(lo), float(hi)),
                method="bounded",
                options={"xatol": 1e-9 * max(float(hi), 1.0)},
            )
            if res.fun < best_re:
                best_re, best_f = float(res.fun), float(res.x)
    scale = float(np.max(np.abs(Z))) if Z.size else 0.0
    return PassivityReport(
        passed=bool(best_re >= -_PASSIVITY_REL_TOL * scale),
        min_re_Z=best_re,
        f_at_min=best_f,
        f_max=f_top,
        rho_at_min=label(i_min),
        rho_range=rho_range,
        n_points=int(Z.size),
    )


# ---------------------------------------------------------------------
# SymPy export
# ---------------------------------------------------------------------


_POLE_PAIR_REL_TOL = 1e-3
"""Relative tolerance of the renderer's pole pairing.

A pole with ``|Im p| < tol * max(|Re p|, 1)`` is rendered as a real pole;
two poles whose real parts and negated imaginary parts agree within
``tol`` (relative to ``max(|.|, 1)``) are rendered as one conjugate pair.
"""


def _partial_fractions(fit: VectorFitResult, s):
    """Partial-fraction SymPy expression of a vector fit, at full precision.

    The one renderer of vector fits (review pass 10, 10.73), used by
    :func:`fit_to_sympy` (``s`` a symbol) and by the ``groundinsight``
    exporter (``s = 2*pi*j*f``):

    .. math::

        Z = R_\\infty + L_\\infty s + \\sum_\\text{real} \\frac{r}{s - p}
            + \\sum_\\text{pairs}
              \\frac{2\\Re r\\,(s - \\Re p) - 2\\Im r\\,\\Im p}
                   {(s - \\Re p)^2 + (\\Im p)^2},

    one term per real pole or conjugate pole pair. Poles are paired with
    :data:`_POLE_PAIR_REL_TOL`; a matched pair is symmetrised (averaged
    real parts and ``|Im|``, the residue at the pole with positive
    imaginary part fixes the sign of ``Im r``), so the pair term is real
    for real ``s``. A complex pole without partner keeps a complex term.
    The sum is not simplified: ``sympy.simplify`` would merge the terms
    into one rational function of high degree (``s**4`` over ``s**3`` for
    a pair, a real pole and :math:`L_\\infty`), which is hard to read and
    badly conditioned.

    Parameters
    ----------
    fit
        The vector fit.
    s
        SymPy symbol or expression substituted for the Laplace variable.

    Returns
    -------
    sympy.Expr
        Sum of the terms; every number is a float64-precision
        :class:`sympy.Float` (round with
        :func:`groundfield.io._formula.round_coefficients`).
    """
    import sympy as sp

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
        # The pair is written with Im p > 0; the residue at that pole
        # fixes the sign of Im r (an abs()-based average lost it before
        # 0.14.1, see CHANGELOG 0.14.1).
        r_plus, r_minus = (r, r2) if p.imag > 0.0 else (r2, r)
        re_p = sp.Float(0.5 * (p.real + p2.real))
        im_p = sp.Float(0.5 * (abs(p.imag) + abs(p2.imag)))
        re_r = sp.Float(0.5 * (r.real + r2.real))
        im_r = sp.Float(0.5 * (r_plus.imag - r_minus.imag))
        numerator = 2 * re_r * (s - re_p) - 2 * im_r * im_p
        terms.append(numerator / ((s - re_p) ** 2 + im_p**2))
    return sp.Add(*terms)


def fit_to_sympy(fit: VectorFitResult, *, decimals: int = 6):
    """Convert a :class:`VectorFitResult` to a SymPy expression in ``s``.

    The expression is the partial-fraction form of the fit,

    $$
    Z(s) \\;=\\; R_\\infty \\;+\\; L_\\infty \\cdot s \\;+\\;
    \\sum_k\\,\\frac{r_k}{s - p_k},
    $$

    with every complex-conjugate pole/residue pair combined into one
    real second-order term,

    $$
    \\frac{r}{s - p} + \\frac{r^*}{s - p^*}
    \\;=\\; \\frac{2\\,\\Re(r)\\,(s - \\Re(p)) -
                  2\\,\\Im(r)\\,\\Im(p)}
                 {(s - \\Re(p))^2 + \\Im(p)^2},
    $$

    so the expression is real for real ``s``. It is built by the renderer
    the ``groundinsight`` exporter uses (one term per real pole or pole
    pair; review pass 10, 10.73) and is not simplified: up to the
    calendar release ``sympy.simplify`` merged the terms into one rational
    function of degree four over three.

    Parameters
    ----------
    fit
        The vector fit.
    decimals
        Significant digits of every floating-point coefficient (1 to 17;
        larger values count as 17). Each coefficient is the full-precision
        value correctly rounded **once** to ``decimals`` digits, so its
        relative error is at most :math:`\\tfrac12\\,10^{1-N}` for
        :math:`N` = ``decimals``. Evaluated with :func:`sympy.lambdify`
        on reviewer R5's 98 vector fits (0 to 2 kHz), the worst deviation
        from :meth:`VectorFitResult.evaluate`, relative to
        :math:`\\max|Z|`, is :math:`4.0\\cdot10^{-6}` at the default
        6 digits and :math:`4.8\\cdot10^{-12}` at 12.

    Returns
    -------
    sympy.Expr
        Expression in the free symbol ``s`` (``sympy.Symbol("s",
        complex=True)``).

    Raises
    ------
    ValueError
        If ``decimals`` is not a positive integer.

    Warns
    -----
    UserWarning
        If the expression does not depend on ``s`` (a fit without poles
        and with :math:`L_\\infty = 0`).
    """
    import sympy as sp

    from groundfield.io._formula import round_coefficients

    s = sp.Symbol("s", complex=True)
    expr = round_coefficients(_partial_fractions(fit, s), decimals)

    # Defensive guard: every rho-f formula must depend on ``s``. A
    # constant Z(rho, f) silently breaks the groundinsight BusType
    # extraction, so we surface that as a ``UserWarning`` here. This
    # cannot trigger under the current ``vector_fit`` because
    # ``n_poles >= 1`` is enforced — the guard catches programmatically
    # constructed ``VectorFitResult`` objects with zero poles and zero
    # ``L_inf``.
    if s not in expr.free_symbols:
        import warnings

        warnings.warn(
            "fit_to_sympy: the resulting expression has no dependency "
            f"on the frequency-domain variable s ({expr!r}). Downstream "
            "groundinsight BusType.impedance_formula consumers expect "
            "a frequency-dependent rho-f model. Re-run vector_fit with "
            "n_poles >= 1 and at least one non-zero residue, or pass "
            "include_L_inf=True if you genuinely want a constant + "
            "inductive shoulder.",
            UserWarning,
            stacklevel=2,
        )

    return expr


# ---------------------------------------------------------------------
# Convenience wrapper for FieldResult
# ---------------------------------------------------------------------


def rho_f_from_field_result(
    result,
    electrode_name: str | None = None,
    *,
    quantity: Quantity = "input",
    source: str | None = None,
    n_poles: int = 4,
    n_iter: int = 8,
    include_R_inf: bool = True,
    include_L_inf: bool = False,
    complex_poles: bool = True,
) -> VectorFitResult:
    """Fit $Z(s)$ to the impedance of a :class:`FieldResult`.

    Reads the impedance selected by ``quantity`` at every frequency in
    ``result.frequencies`` and runs :func:`vector_fit` on the series.

    Parameters
    ----------
    result
        :class:`groundfield.FieldResult` produced by an
        ``Engine.solve`` call with at least 4–8 frequency samples.
    electrode_name
        Electrode whose impedance is fitted. Required for
        ``quantity="cluster"`` and ``"electrode"``. With ``"input"`` it
        is optional; when given it must belong to the cluster the source
        feeds (and selects among several sources).
    quantity
        ``"input"`` (default since review pass 10, 10.19): the
        driving-point impedance :math:`(\\varphi_a - \\varphi_r)/I_s`
        of the current source (:meth:`FieldResult.input_impedance`) —
        the impedance a groundinsight BusType stands for.
        ``"cluster"``: :meth:`FieldResult.cluster_impedance` of
        ``electrode_name``. ``"electrode"``: :math:`U_e/I_e` of the
        named electrode alone, the quantity v0.15.0 fitted (17 % above
        the cluster impedance for a ring with two bonded rods).
    source
        Current source for ``quantity="input"``; may be omitted when the
        result carries exactly one.
    n_poles, n_iter, include_R_inf, include_L_inf, complex_poles
        Forwarded to :func:`vector_fit`.

    Returns
    -------
    VectorFitResult

    Raises
    ------
    ValueError
        For an unknown ``quantity``; for ``"input"`` on a result without
        source snapshot or with an ``electrode_name`` the source does not
        feed; for ``"cluster"`` / ``"electrode"`` without
        ``electrode_name``; if the denominator current is zero at every
        frequency.
    KeyError
        If ``electrode_name`` or ``source`` is unknown.

    Notes
    -----
    Frequencies at which the denominator current is zero (a passive
    electrode, a dead column) are masked with a :class:`UserWarning`
    instead of being entered as :math:`Z = 0`.
    """
    z_all, i_all = impedance_series(
        result,
        quantity,
        electrode_name=electrode_name,
        source=source,
        caller="rho_f_from_field_result",
    )
    freqs = np.asarray(result.frequencies, dtype=float)
    if i_all.shape != freqs.shape:
        raise ValueError(
            "Inconsistent shapes between frequencies and electrode currents"
        )
    label = f"electrode '{electrode_name}'" if quantity != "input" else "the source"
    # Audit 2026-07-08, WP-B4 (finding N23): frequencies where the
    # denominator current vanishes (passive electrode / dead column)
    # are MASKED instead of being injected as Z = 0 samples — a
    # single dead frequency used to pull the whole fit through the
    # origin silently.
    alive = np.abs(i_all) > 0.0
    if not alive.any():
        raise ValueError(
            f"rho_f_from_field_result: {label} carries no current at any "
            "frequency — Z = U/I is undefined. Fit the driven electrode "
            "instead."
        )
    if not alive.all():
        import warnings as _warnings

        _warnings.warn(
            f"rho_f_from_field_result: {label} carries no current at "
            f"{int((~alive).sum())} of {alive.size} frequencies; those "
            "samples are excluded from the fit (historic behaviour injected "
            "Z = 0 there).",
            UserWarning,
            stacklevel=2,
        )
    fit = vector_fit(
        freqs[alive],
        z_all[alive],
        n_poles=n_poles,
        n_iter=n_iter,
        include_R_inf=include_R_inf,
        include_L_inf=include_L_inf,
        complex_poles=complex_poles,
    )
    return dataclasses.replace(fit, quantity=quantity)
