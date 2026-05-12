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
the dissertation's $f \\le 1\\,\\mathrm{kHz}$ assumption the
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

import cmath
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "VectorFitResult",
    "vector_fit",
    "fit_to_sympy",
    "rho_f_from_field_result",
]


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
        Original $Z(s)$ values used for the fit.
    """

    poles: np.ndarray
    residues: np.ndarray
    R_inf: float
    L_inf: float
    rms_error: float
    fit_frequencies: np.ndarray
    fit_values: np.ndarray

    def evaluate(self, frequencies: Sequence[float]) -> np.ndarray:
        """Evaluate the fitted $Z(s)$ at arbitrary frequencies."""
        s = 2j * math.pi * np.asarray(frequencies, dtype=float)
        Z = np.full_like(s, self.R_inf, dtype=complex)
        Z = Z + self.L_inf * s
        for p, r in zip(self.poles, self.residues):
            Z = Z + r / (s - p)
        return Z


# ---------------------------------------------------------------------
# Vector fitting algorithm
# ---------------------------------------------------------------------


def _initial_poles(
    frequencies: np.ndarray, n_poles: int, *, complex_conj: bool = True,
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


def vector_fit(
    frequencies: Sequence[float],
    Z_values: Sequence[complex],
    *,
    n_poles: int = 4,
    n_iter: int = 8,
    include_R_inf: bool = True,
    include_L_inf: bool = False,
    complex_poles: bool = True,
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

    Returns
    -------
    VectorFitResult
    """
    frequencies = np.asarray(frequencies, dtype=float)
    Z = np.asarray(Z_values, dtype=complex)
    if frequencies.shape != Z.shape:
        raise ValueError(
            "frequencies and Z_values must have the same shape"
        )
    if n_poles < 1:
        raise ValueError("n_poles must be at least 1")
    s = 2j * math.pi * frequencies

    poles = _initial_poles(frequencies, n_poles, complex_conj=complex_poles)

    # Outer loop: pole relocation. The system is, schematically,
    #   sum_k r_k / (s_n - p_k) + d + s_n*h - sum_k r'_k Z_n / (s_n - p_k)
    #     = Z_n
    # where {r_k, d, h, r'_k} are the unknowns and the new poles are
    # the zeros of the denominator polynomial 1 + sum r'_k/(s-p_k).
    extras = (1 if include_R_inf else 0) + (1 if include_L_inf else 0)
    for _ in range(max(1, n_iter)):
        n_unknowns = n_poles + extras + n_poles  # numer poles + d/h + denom poles
        A = np.zeros((s.size, n_unknowns), dtype=complex)
        for k, p in enumerate(poles):
            A[:, k] = 1.0 / (s - p)
        col = n_poles
        if include_R_inf:
            A[:, col] = 1.0
            col += 1
        if include_L_inf:
            A[:, col] = s
            col += 1
        for k, p in enumerate(poles):
            A[:, col + k] = -Z / (s - p)

        # Solve with real-imag stacking (least-squares)
        M = np.vstack([A.real, A.imag])
        rhs = np.concatenate([Z.real, Z.imag])
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)

        # Extract residues of the denominator polynomial
        sigma_residues = sol[n_poles + extras:]
        # New poles: eigenvalues of A_p - b·c^T where A_p = diag(poles),
        # b = ones, c = sigma_residues. (Standard Gustavsen formulation.)
        Ap = np.diag(poles)
        b = np.ones(n_poles)
        new_poles = np.linalg.eigvals(Ap - np.outer(b, sigma_residues))
        new_poles = _enforce_stability(new_poles)
        new_poles = _enforce_conjugate_symmetry(new_poles)
        # Sort for deterministic order (deepest |imag| first).
        sorted_idx = np.argsort(-np.abs(new_poles.imag))
        new_poles = new_poles[sorted_idx]
        poles = new_poles

    # Final residue solve with the converged poles
    n_unknowns = n_poles + extras
    A = np.zeros((s.size, n_unknowns), dtype=complex)
    for k, p in enumerate(poles):
        A[:, k] = 1.0 / (s - p)
    col = n_poles
    if include_R_inf:
        A[:, col] = 1.0
        col += 1
    if include_L_inf:
        A[:, col] = s
        col += 1
    M = np.vstack([A.real, A.imag])
    rhs = np.concatenate([Z.real, Z.imag])
    sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    residues = sol[:n_poles].astype(complex)
    R_inf = float(sol[n_poles]) if include_R_inf else 0.0
    L_inf = float(sol[n_poles + (1 if include_R_inf else 0)]) if include_L_inf else 0.0

    # Enforce that residues at conjugate poles are themselves
    # conjugates (same numerical-precision argument as for the
    # poles). Pairs are identified by exact equality after
    # _enforce_conjugate_symmetry.
    used = np.zeros(n_poles, dtype=bool)
    for i in range(n_poles):
        if used[i]:
            continue
        if abs(poles[i].imag) < 1e-12:
            residues[i] = complex(residues[i].real, 0.0)
            used[i] = True
            continue
        # Find the conjugate partner
        for j in range(i + 1, n_poles):
            if used[j]:
                continue
            if (
                abs(poles[j].real - poles[i].real) < 1e-9
                and abs(poles[j].imag + poles[i].imag) < 1e-9
            ):
                re_avg = 0.5 * (residues[i].real + residues[j].real)
                im_diff = 0.5 * (residues[i].imag - residues[j].imag)
                # If pole_i has positive imag, the canonical residue
                # also carries positive imag; flip sign for j.
                if poles[i].imag > 0:
                    residues[i] = complex(re_avg, im_diff)
                    residues[j] = complex(re_avg, -im_diff)
                else:
                    residues[i] = complex(re_avg, -im_diff)
                    residues[j] = complex(re_avg, im_diff)
                used[i] = True
                used[j] = True
                break
        else:
            # Lone complex pole (shouldn't happen after symmetrisation)
            residues[i] = complex(residues[i].real, 0.0)
            used[i] = True

    Z_fit = (
        np.full_like(s, R_inf, dtype=complex)
        + L_inf * s
        + np.sum(residues[:, None] / (s[None, :] - poles[:, None]), axis=0)
    )
    rms = float(np.sqrt(np.mean(np.abs(Z_fit - Z) ** 2)))

    return VectorFitResult(
        poles=poles,
        residues=residues,
        R_inf=R_inf,
        L_inf=L_inf,
        rms_error=rms,
        fit_frequencies=frequencies,
        fit_values=Z,
    )


# ---------------------------------------------------------------------
# SymPy export
# ---------------------------------------------------------------------


def fit_to_sympy(fit: VectorFitResult, *, decimals: int = 6):
    """Convert a :class:`VectorFitResult` to a SymPy expression.

    The resulting expression is

    $$
    Z(s) \\;=\\; R_\\infty \\;+\\; L_\\infty \\cdot s \\;+\\;
    \\sum_k\\,\\frac{r_k}{s - p_k}.
    $$

    Complex-conjugate pole/residue pairs are combined into a single
    real second-order term to keep the formula compact and
    interpretable for ``groundinsight``:

    $$
    \\frac{r}{s - p} + \\frac{r^*}{s - p^*}
    \\;=\\; \\frac{2\\,\\Re(r)\\,(s - \\Re(p)) -
                  2\\,\\Im(r)\\,\\Im(p)}
                 {(s - \\Re(p))^2 + \\Im(p)^2}.
    $$

    The coefficients are rounded to ``decimals`` digits to keep
    the printed formula readable.

    Returns a :class:`sympy.Expr`. Free symbol: ``s``.
    """
    import sympy as sp

    s = sp.Symbol("s", complex=True)
    expr = sp.Float(fit.R_inf, decimals)
    if fit.L_inf != 0.0:
        expr = expr + sp.Float(fit.L_inf, decimals) * s

    # Bucket poles into real and complex-conjugate pairs. Vector
    # fitting's eigenvalue solve does not return exactly mirrored
    # imaginary parts; we use a loose relative tolerance and
    # **symmetrise** the matched pair (taking the average of |Im|)
    # so the resulting expression is rigorously real-valued for
    # real s.
    used = np.zeros(len(fit.poles), dtype=bool)
    rel_tol = 1e-3
    for k, (p, r) in enumerate(zip(fit.poles, fit.residues)):
        if used[k]:
            continue
        if abs(p.imag) < rel_tol * max(abs(p.real), 1.0):
            # Effectively real pole
            expr = expr + sp.Float(r.real, decimals) / (
                s - sp.Float(p.real, decimals)
            )
            used[k] = True
            continue
        # Look for the conjugate (loose tolerance, opposite-sign imag)
        conj_idx = -1
        best_score = float("inf")
        for k2 in range(k + 1, len(fit.poles)):
            if used[k2]:
                continue
            d_real = abs(fit.poles[k2].real - p.real) / max(abs(p.real), 1.0)
            d_imag = abs(fit.poles[k2].imag + p.imag) / max(abs(p.imag), 1.0)
            score = d_real + d_imag
            if d_real < rel_tol and d_imag < rel_tol and score < best_score:
                conj_idx = k2
                best_score = score
        if conj_idx < 0:
            # No conjugate found — fall back to a single complex term
            expr = expr + sp.nsimplify(r, rational=False) / (
                s - sp.nsimplify(p, rational=False)
            )
            used[k] = True
            continue
        # Symmetrise: take averages so the pair is exactly conjugate.
        used[k] = True
        used[conj_idx] = True
        p2 = fit.poles[conj_idx]
        r2 = fit.residues[conj_idx]
        Re_p_avg = 0.5 * (p.real + p2.real)
        Im_p_avg = 0.5 * (abs(p.imag) + abs(p2.imag))  # use positive imag for the canonical pair
        Re_r_avg = 0.5 * (r.real + r2.real)
        # The residue conjugate carries opposite imag; canonical pair
        # has +Im_r corresponding to +Im_p. If p.imag > 0, r.imag is
        # the "canonical" imag; otherwise flip.
        Im_r_avg = 0.5 * (abs(r.imag) + abs(r2.imag))
        if p.imag < 0:
            Im_r_avg *= -1.0  # match sign convention
        # Real second-order term:
        # 2·Re(r)·(s - Re(p)) - 2·Im(r)·Im(p)  /  (s - Re(p))² + Im(p)²
        Re_p_sym = sp.Float(Re_p_avg, decimals)
        Im_p_sym = sp.Float(Im_p_avg, decimals)
        Re_r_sym = sp.Float(Re_r_avg, decimals)
        Im_r_sym = sp.Float(Im_r_avg, decimals)
        num = 2 * Re_r_sym * (s - Re_p_sym) - 2 * Im_r_sym * Im_p_sym
        den = (s - Re_p_sym) ** 2 + Im_p_sym ** 2
        expr = expr + num / den

    return sp.simplify(expr)


# ---------------------------------------------------------------------
# Convenience wrapper for FieldResult
# ---------------------------------------------------------------------


def rho_f_from_field_result(
    result,
    electrode_name: str,
    *,
    n_poles: int = 4,
    n_iter: int = 8,
    include_R_inf: bool = True,
    include_L_inf: bool = False,
    complex_poles: bool = True,
) -> VectorFitResult:
    """Fit $Z(s)$ for one electrode of a :class:`FieldResult`.

    Computes
    $Z_k = U_k / I_k$
    at every frequency in ``result.frequencies`` for
    ``result.electrode_potentials[electrode_name]`` and
    ``result.electrode_currents[electrode_name]``, then runs
    :func:`vector_fit` on the resulting series.

    Parameters
    ----------
    result
        :class:`groundfield.FieldResult` produced by an
        ``Engine.solve`` call with at least 4–8 frequency samples.
    electrode_name
        Name of the electrode to extract.
    n_poles, n_iter, include_R_inf, include_L_inf, complex_poles
        Forwarded to :func:`vector_fit`.

    Returns
    -------
    VectorFitResult
    """
    if electrode_name not in result.electrode_potentials:
        raise KeyError(
            f"electrode '{electrode_name}' not in result; "
            f"available: {list(result.electrode_potentials)}"
        )
    freqs = np.asarray(result.frequencies, dtype=float)
    U = np.asarray(result.electrode_potentials[electrode_name], dtype=complex)
    I = np.asarray(result.electrode_currents[electrode_name], dtype=complex)
    if I.shape != freqs.shape:
        raise ValueError(
            "Inconsistent shapes between frequencies and electrode currents"
        )
    Z = np.where(np.abs(I) > 0.0, U / np.where(I != 0, I, 1.0), 0.0 + 0.0j)
    return vector_fit(
        freqs, Z,
        n_poles=n_poles, n_iter=n_iter,
        include_R_inf=include_R_inf, include_L_inf=include_L_inf,
        complex_poles=complex_poles,
    )
