"""Complex Image Method backend (``cim``).

Mathematical / physical model
-----------------------------
The :mod:`image_nlayer` backend expands the upward-looking reflection
$\\Gamma_1(\\lambda)$ as a power series in the per-layer
attenuation factors $e^{-2\\lambda h_i}$ — a representation that
converges fast for two layers but generates many terms for
$n \\ge 3$ and high contrasts.

The **Complex Image Method (CIM)** instead approximates
$\\Gamma_1(\\lambda)$ directly by a finite sum of complex
exponentials,
$$
\\Gamma_1(\\lambda) \\;\\approx\\; \\sum_{k=1}^{P} a_k\\,
e^{-2\\lambda \\beta_k},
$$
with complex coefficients $a_k \\in \\mathbb{C}$ and complex
"image depths" $\\beta_k \\in \\mathbb{C}$ (with
$\\Re\\{\\beta_k\\} > 0$ to keep the integrals convergent).
Substituting this approximation into the Sommerfeld integral and
using the closed-form
$$
\\int_0^{\\infty} e^{-\\lambda d} J_0(\\lambda s)\\, d\\lambda
\\;=\\; \\frac{1}{\\sqrt{s^2 + d^2}}
$$
immediately gives a **closed-form** spatial Green's function of the
same shape as the homogeneous image-charge sum, but with **complex
image positions**. A kernel built from such a fit would read
$$
\\varphi(s, z) \\;=\\; \\frac{\\rho_1\\, I}{4\\pi}\\,
\\Bigl(\\frac{1}{r} + \\frac{1}{r_{\\text{air}}}
     + \\sum_{k=1}^{P} a_k
       \\Bigl(\\frac{1}{\\sqrt{s^2 + (z + z_s + 2\\beta_k)^2}}
            + \\frac{1}{\\sqrt{s^2 + (z - z_s + 2\\beta_k)^2}}\\Bigr)
     \\Bigr).
$$
The cost of one potential evaluation would therefore be the same as
the homogeneous backend, multiplied by ``2 * P`` — independent of the
layer count. *No backend evaluates this kernel today*; see "Solver
status of the fit" below, and note that the form above still needs the
$2 h_1$ image families and the multiple-reflection denominator of
:mod:`groundfield.solver._layered` before it represents the layered
Green's function completely.

Numerical fit strategy
----------------------
We use the **matrix-pencil method** (a numerically stable variant of
Prony's algorithm) to fit a sample of $\\Gamma_1(\\lambda)$ on a
uniform grid in $\\lambda$ to ``P`` complex exponentials.
This is a faithful Python re-implementation of the segmented-sampling
least-squares idea of Dan et al. 2021 (without the segmentation
heuristic, which is needed mainly for very many layers; for the
typical two-/three-layer use cases a single segment with a moderately
oversampled grid is enough).

One structural detail matters: a pure sum of *decaying* exponentials
vanishes for $\\lambda \\to \\infty$, whereas
$$
\\lim_{\\lambda \\to \\infty} \\Gamma_1(\\lambda) \\;=\\; K_1
\\;=\\; \\frac{\\rho_2 - \\rho_1}{\\rho_2 + \\rho_1}
\\;\\neq\\; 0
$$
(every $e^{-2 \\lambda h_i}$ in the recursion dies, leaving the
top-interface Fresnel coefficient). The asymptote is therefore
**split off analytically** — it is carried by the single
$\\beta = 0$ image, i.e. an image at the air-mirror position
$z = -z_s$ with weight $K_1$ — and only the decaying remainder
$\\Gamma_1(\\lambda) - K_1$ is handed to the matrix pencil.
Fitting $\\Gamma_1$ itself (as this module did up to 0.14.1) leaves a
residual of order $|K_1|$ for *every* stack, because the pole
filters discard precisely the constant term that would carry the
asymptote (review pass 9, F39).

Solver status of the fit (0.15.0)
---------------------------------
:func:`fit_complex_images` is a **standalone spectral helper**: no
solver path evaluates it any more. ``cim`` (and ``bem``, which shares
these kernels) rejects $n \\ge 3$ soils, and for $n \\le 2$ both
engines use *exact closed-form* self-kernels — the homogeneous
image-charge sum for $n = 1$ and the Tagg/Sunde series of
:mod:`groundfield.solver.image_2layer` for $n = 2$ — which do not
involve a fit at all. Up to 0.14.1 the fit was nevertheless computed
on every solve and its (failed) diagnostics were reported as
``metadata['cim_n_images']`` / ``['cim_rms']``, which invited readers
to judge an approximation that had no influence on the answer
(review pass 9, F34). The call is gone; the metadata now carries
``cim_fit_used: False`` and a ``reduces_to`` key naming the backend
this result is bit-identical to.

Validity
--------
- Quasi-static, $f < 1\\,\\mathrm{kHz}$.
- ``n_layers == 1`` → homogeneous image-charge sum, bit-identical to
  ``image`` ($\\Gamma_1 \\equiv 0$).
- ``n_layers == 2`` → exact Tagg/Sunde series, bit-identical to
  ``image_2layer``. ``cim`` is therefore **not** an independent
  cross-check of ``image_2layer`` in this regime (ADR-0002
  amendment 2026-07-09).
- ``n_layers ≥ 3`` is **rejected** (audit 2026-07-08, WP-E): the
  historic complex-image expansion approximated an *incomplete*
  Green's function (single image family, no multiple reflections —
  see :mod:`groundfield.solver._layered`) and systematically
  underestimated the layered correction. Use ``mom_sommerfeld`` or
  ``fem`` for three and more layers.

References
----------
- Sarkar, T. K. & Pereira, O. (1995). Using the matrix pencil method
  to estimate the parameters of a sum of complex exponentials, IEEE
  Antennas & Propagation Magazine 37(1).
- Li, Z.-X. et al. (2006). A novel mathematical modeling of grounding
  system buried in multilayer earth, IEEE PWRD 21(3).
- Dan, Y. et al. (2021). Segmented sampling least squares algorithm
  for Green's function of arbitrary layered soil, IEEE PWRD 36(3).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.solver._layered import (
    LayerStack,
    as_layer_stack,
    reflection_gamma,
)
from groundfield.solver.image import (
    _assemble_inductance_matrix,
    _build_clusters,
    _build_distributed_topology,
    _build_finite_branches,
    _discretize_electrode,
    _reject_concrete_shells,
    _self_corrected_kernel,
    _Segment,
    _solve_cluster_currents,
    _warn_ignored_sources,
)
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = [
    "ComplexImageFit",
    "ComplexImageFitWarning",
    "fit_complex_images",
    "solve_cim",
]

_log = get_logger(__name__)

#: Samples per structure scale $1/(2 h_{\\max})$ of the decaying
#: remainder $\\Gamma_1 - K_1$. The matrix pencil needs a *uniform*
#: grid, so the step is capped at this resolution; 8 samples per
#: e-folding of the slowest exponential is ample for a pencil of
#: order 8–12.
_SAMPLES_PER_STRUCTURE_SCALE = 8.0

#: Below this the λ-dependence of Γ_1 (or Γ_1 itself) counts as
#: numerically absent — the exact closed form is returned instead of
#: running an ill-conditioned pencil on noise.
_FLAT_TOL = 1e-12

#: Default residual above which :func:`fit_complex_images` warns.
#: Since $|\\Gamma_1| \\le 1$ by construction, an absolute RMS is a
#: meaningful accuracy measure.
_DEFAULT_RMS_TOL = 1e-3

#: Size of the independent log-spaced verification grid on which the
#: fit is re-evaluated (catches window truncation and overfitting,
#: which a residual measured on the fitting grid alone cannot see).
_N_VERIFY = 512


class ComplexImageFitWarning(UserWarning):
    """The complex-image fit of $\\Gamma_1(\\lambda)$ is unreliable.

    Emitted by :func:`fit_complex_images` when the RMS residual on the
    sample grid or on the independent verification grid exceeds
    ``rms_tol``, or when the matrix pencil returned no usable pole for
    a remainder that is *not* numerically flat. All three situations
    mean the returned images do not represent $\\Gamma_1$ and must
    not be fed into a Green's function.

    Up to 0.14.1 the same situations were silent: the pole filters
    (``|p| < 0.999``, ``Re β > 0``) discarded the constant term that
    carries the $\\lambda \\to \\infty$ asymptote $K_1$, so the
    residual was of order $|K_1|$ for every stack, and a two-layer
    stack (where that constant is the *whole* function) returned
    ``P = 0`` with ``rms = nan`` (review pass 9, F39).

    A dedicated category lets callers opt in / out with
    ``warnings.simplefilter("error", ComplexImageFitWarning)``.
    """


@dataclass(frozen=True)
class ComplexImageFit:
    """Result of :func:`fit_complex_images`.

    The fit represents
    $\\Gamma_1(\\lambda) \\approx \\sum_k a_k e^{-2 \\lambda \\beta_k}$.
    Index 0 is the analytic asymptote image ($\\beta_0 = 0$,
    $a_0 \\approx K_1$) whenever the stack has one; the remaining
    entries are the decaying images recovered by the matrix pencil.

    Attributes
    ----------
    a : np.ndarray, shape (P,), complex
        Image weights $a_k$.
    beta : np.ndarray, shape (P,), complex
        Image depths $\\beta_k$ (units of metres).
    rms : float
        RMS of the residual $|\\text{fit} - \\Gamma_1|$ on the sample
        grid. Always finite: a failed fit reports the residual it
        actually achieved (and warns) instead of ``nan``.
    k_inf : float
        The $\\lambda \\to \\infty$ asymptote
        $K_1 = (\\rho_2 - \\rho_1)/(\\rho_2 + \\rho_1)$ that the
        $\\beta = 0$ image carries. ``0.0`` for a homogeneous stack.
    rms_extrapolated : float
        RMS of the same residual on an **independent** log-spaced
        verification grid covering the full requested
        $\\lambda$ range. Larger than ``rms`` when the uniform
        fitting window did not reach the decay band of the thinnest
        layer (large thickness spread) or when the pencil overfitted.
    converged : bool
        Both residuals are ``<= rms_tol`` *and* the decaying remainder
        was either numerically flat or successfully fitted. ``False``
        means a :class:`ComplexImageFitWarning` was emitted.
    """

    a: np.ndarray
    beta: np.ndarray
    rms: float
    k_inf: float = 0.0
    rms_extrapolated: float = 0.0
    converged: bool = True


def _empty_fit() -> ComplexImageFit:
    """Fit with no images at all ($\\Gamma_1 \\equiv 0$, exact)."""
    return ComplexImageFit(
        a=np.zeros(0, dtype=complex),
        beta=np.zeros(0, dtype=complex),
        rms=0.0,
        k_inf=0.0,
        converged=True,
    )


def _matrix_pencil_beta(
    g: np.ndarray, delta: float, n_images: int
) -> np.ndarray:
    """Recover the decay rates $\\beta_k$ of a sum of exponentials.

    Matrix-pencil method (Sarkar & Pereira 1995) on a uniformly
    sampled signal $g_j \\approx \\sum_k c_k e^{-2 \\Delta \\beta_k j}$:
    the poles $p_k = e^{-2 \\Delta \\beta_k}$ are the eigenvalues of
    the rank-reduced Hankel pencil $[Y_0, Y_1]$.

    Poles are filtered to $|p_k| < 0.999$ (a pole on or outside the
    unit circle is non-decaying, hence not integrable against the
    Sommerfeld identity) and to $\\Re\\{\\beta_k\\} > 0$. The
    coefficients are **not** solved here — the caller re-solves them
    by least squares on the surviving set, which is what makes the
    filtered model self-consistent.

    Parameters
    ----------
    g : np.ndarray, shape (N,), complex
        Uniformly sampled, decaying signal.
    delta : float
        Grid step $\\Delta$ of the sample grid.
    n_images : int
        Target pencil order.

    Returns
    -------
    beta : np.ndarray, shape (P,), complex
        Surviving decay rates, ``P <= n_images``. Empty if the pencil
        is degenerate or every pole was filtered out.
    """
    n_samples = int(g.size)
    empty = np.zeros(0, dtype=complex)

    L = max(n_images + 1, n_samples // 3)
    L = min(L, n_samples - n_images - 1)
    if L < n_images + 1:
        # Not enough samples — fall back to fewer images.
        n_images = max(1, L - 1)
    if L < 2 or n_images < 1:
        return empty

    rows = n_samples - L
    Y = np.zeros((rows, L + 1), dtype=complex)
    for i in range(rows):
        Y[i, :] = g[i:i + L + 1]
    Y0 = Y[:, :L]
    Y1 = Y[:, 1:L + 1]

    # SVD-based filtering: project both blocks onto the dominant
    # singular subspace of order n_images. Filter near-zero singular
    # values to avoid 1/0.
    U, S, Vh = np.linalg.svd(Y0, full_matrices=False)
    s_max = float(S.max()) if S.size else 0.0
    if s_max == 0.0:
        return empty
    P = int(min(n_images, np.sum(S > s_max * 1e-10)))
    if P < 1:
        return empty
    Up = U[:, :P]
    Sp = S[:P]
    Vp = Vh[:P, :]

    # Y1 ≈ Up · diag(Sp) · Vp · Z, with Z holding the poles in its
    # eigenvalues. Equivalently the poles solve
    #   eig( pinv(Y0) · Y1 ) ≈ pinv(Sp) · Up^H · Y1 · Vp^H.
    Z = np.diag(1.0 / Sp) @ Up.conj().T @ Y1 @ Vp.conj().T
    poles = np.linalg.eigvals(Z)

    # Map poles back to the β coefficients: p_k = exp(-2 Δ β_k)
    #   → β_k = -log(p_k) / (2 Δ).
    poles = poles[np.abs(poles) < 0.999]
    if poles.size == 0:
        return empty
    beta = -np.log(poles) / (2.0 * delta)
    return beta[beta.real > 0.0]


def fit_complex_images(
    stack: LayerStack,
    *,
    n_images: int = 8,
    n_samples: int = 64,
    lambda_min_factor: float = 1e-3,
    lambda_max_factor: float = 50.0,
    rms_tol: float = _DEFAULT_RMS_TOL,
    warn: bool = True,
) -> ComplexImageFit:
    """Fit Γ_1(λ) of an n-layer stack by ``n_images`` complex exponentials.

    Physical / mathematical context
    ------------------------------
    The upward-looking reflection coefficient of a stratified
    half-space (see :func:`groundfield.solver._layered.reflection_gamma`)
    is a rational function of the interface attenuations
    $e^{-2 \\lambda h_i}$. It is **not** a decaying function: it
    tends to the top-interface Fresnel coefficient

    $$
    \\lim_{\\lambda \\to \\infty} \\Gamma_1(\\lambda)
    \\;=\\; K_1 \\;=\\; \\frac{\\rho_2 - \\rho_1}{\\rho_2 + \\rho_1},
    $$

    while a sum $\\sum_k a_k e^{-2\\lambda\\beta_k}$ with
    $\\Re\\{\\beta_k\\} > 0$ tends to 0. The asymptote is hence
    split off analytically and carried by an image at
    $\\beta_0 = 0$ (weight $K_1$; in the spatial domain an image
    at the air-mirror position $z = -z_s$, integrable through the
    Sommerfeld identity because $d = z + z_s > 0$), and the matrix
    pencil only sees the genuinely decaying remainder
    $\\Gamma_1(\\lambda) - K_1$.

    Steps:

    1. Build a uniform sampling grid
       $\\lambda_j = \\lambda_{\\min} + j \\Delta$. Only the
       thicknesses $h_2, \\dots, h_{n-1}$ enter $\\Gamma_1$
       (the recursion never uses $h_1$), so the *structure* scale is
       $h_{\\max} = \\max_i h_{i\\ge2}$ and the *decay* scale is
       $h_{\\min} = \\min_i h_{i\\ge2}$: the step is capped at
       $\\Delta \\le 1/(8 h_{\\max})$ so the slowest exponential is
       resolved, and the window reaches up to
       $\\lambda_{\\max} = \\lambda_{\\max,\\text{factor}}/h_{\\min}$
       where the fastest one has died.
    2. Sample $g_j = \\Gamma_1(\\lambda_j)$ and subtract $K_1$.
    3. Apply the matrix-pencil method to the remainder to recover the
       poles $p_k = e^{-2 \\Delta \\beta_k}$, filtered to
       $|p_k| < 0.999$ and $\\Re\\{\\beta_k\\} > 0$.
    4. Re-solve the least-squares coefficients $a_k$ for the
       *surviving* $\\{\\beta_k\\} \\cup \\{0\\}$ on the original
       samples of $\\Gamma_1$ — the filtered model is therefore
       self-consistent (up to 0.14.1 the coefficients were solved
       before the filter and then merely sliced, so the reported
       weights belonged to a different model).

    Special cases:

    - ``stack.n_layers <= 1`` → empty fit ($\\Gamma_1 \\equiv 0$).
    - ``stack.n_layers == 2`` → $\\Gamma_1 \\equiv K_1$ exactly;
      the single $\\beta = 0$ image reproduces it with ``rms = 0``
      (up to 0.14.1 this returned ``P = 0`` and ``rms = nan``).
    - $\\rho_1 = \\dots = \\rho_n$ → empty fit.

    Notes
    -----
    No solver path calls this function: ``cim``/``bem`` reject
    $n \\ge 3$ and use exact closed-form kernels for $n \\le 2$
    (see the module docstring). It is kept public as the spectral
    building block for a future complete $n \\ge 3$ complex-image
    kernel, and it now reports failure instead of hiding it.

    Parameters
    ----------
    stack
        Layer stack to fit.
    n_images
        Target number of *decaying* complex images. Sensible range
        4–12; the returned $P$ is one larger when the asymptote
        image is present, and may be smaller when the SVD truncates
        poles.
    n_samples
        Number of samples drawn from Γ_1(λ).
    lambda_min_factor
        Lower grid bound as a multiple of $1/h_{\\max}$.
    lambda_max_factor
        Upper grid bound as a multiple of $1/h_{\\min}$ (an upper
        bound only — the step cap of item 1 above may stop the grid
        earlier).
    rms_tol
        Residual (on the fitting *and* on the verification grid) above
        which the fit counts as failed (``converged=False`` plus a
        :class:`ComplexImageFitWarning`).
    warn
        Set to ``False`` to suppress the warning in bulk sweeps;
        ``converged`` / ``rms`` / ``rms_extrapolated`` still report the
        failure.

    Returns
    -------
    ComplexImageFit

    Warns
    -----
    ComplexImageFitWarning
        If ``rms`` or ``rms_extrapolated`` exceeds ``rms_tol``, or the
        pencil produced no usable pole for a non-flat remainder.
    """
    if stack.n_layers <= 1:
        return _empty_fit()

    k_inf = float(stack.K[0])

    # Only h_2 … h_{n-1} enter Γ_1 — the recursion in
    # `reflection_gamma` never touches h_1.
    h_rel = np.asarray(stack.h[1:], dtype=float)

    if h_rel.size == 0:
        # n == 2: Γ_1 ≡ K_1, a constant. Exactly one image, at β = 0.
        if abs(k_inf) < _FLAT_TOL:
            return _empty_fit()
        return ComplexImageFit(
            a=np.array([k_inf + 0j]),
            beta=np.zeros(1, dtype=complex),
            rms=0.0,
            k_inf=k_inf,
            converged=True,
        )

    h_struct = float(np.max(h_rel))
    h_decay = float(np.min(h_rel))
    lam_min = lambda_min_factor / h_struct
    lam_max = lambda_max_factor / h_decay

    # Uniform spacing required by the matrix-pencil method; capped so
    # the slowest exponential (scale 1/(2 h_struct)) is resolved.
    delta = min(
        (lam_max - lam_min) / (n_samples - 1),
        1.0 / (_SAMPLES_PER_STRUCTURE_SCALE * h_struct),
    )
    lam = lam_min + delta * np.arange(n_samples)
    g = reflection_gamma(stack, lam).astype(complex)

    # Degenerate case ρ_1 = ρ_2 = … = ρ_n: nothing to fit.
    if np.max(np.abs(g)) < _FLAT_TOL:
        return _empty_fit()

    resid = g - k_inf
    if np.max(np.abs(resid)) < _FLAT_TOL:
        # Γ_1 is constant on the whole window (e.g. ρ_2 = … = ρ_n).
        return ComplexImageFit(
            a=np.array([k_inf + 0j]),
            beta=np.zeros(1, dtype=complex),
            rms=float(np.sqrt(np.mean(np.abs(resid) ** 2))),
            k_inf=k_inf,
            converged=True,
        )

    beta_decay = _matrix_pencil_beta(resid, delta, n_images)
    pencil_failed = beta_decay.size == 0

    # β = 0 (the analytic asymptote) plus the decaying images.
    beta = np.concatenate([np.zeros(1, dtype=complex), beta_decay])

    # Coefficients a_k by linear least squares on the original samples
    # of Γ_1 — solved *after* all pole filtering.
    A = np.exp(-2.0 * lam[:, None] * beta[None, :])  # (n_samples, P)
    a, *_ = np.linalg.lstsq(A, g, rcond=None)
    rms = float(np.sqrt(np.mean(np.abs(A @ a - g) ** 2)))

    # Independent verification on a log-spaced grid over the *full*
    # requested λ range: the uniform fitting window may stop well
    # below λ_max when the thickness spread is large, and a residual
    # measured on the fitting grid alone cannot see that.
    lam_v = np.logspace(np.log10(lam_min), np.log10(lam_max), _N_VERIFY)
    g_v = reflection_gamma(stack, lam_v).astype(complex)
    model_v = np.exp(-2.0 * lam_v[:, None] * beta[None, :]) @ a
    rms_wide = float(np.sqrt(np.mean(np.abs(model_v - g_v) ** 2)))

    converged = (
        bool(rms <= rms_tol)
        and bool(rms_wide <= rms_tol)
        and not pencil_failed
    )

    _log.info(
        "cim.fit: n_images_used=%d (P_decay=%d), rms=%.2e, "
        "rms_extrapolated=%.2e, K_1=%.4f, n_layers=%d, "
        "lam=[%.3g, %.3g], delta=%.3g",
        beta.size, beta_decay.size, rms, rms_wide, k_inf,
        stack.n_layers, lam[0], lam[-1], delta,
    )
    if not converged and warn:
        if pencil_failed:
            reason = (
                "the matrix pencil returned no usable pole for a "
                "non-flat remainder Γ_1 - K_1"
            )
        elif rms > rms_tol:
            reason = f"RMS residual {rms:.3e} > rms_tol {rms_tol:.3e}"
        else:
            reason = (
                f"RMS residual on the verification grid "
                f"{rms_wide:.3e} > rms_tol {rms_tol:.3e}"
            )
        warnings.warn(
            f"fit_complex_images: unreliable fit for an "
            f"{stack.n_layers}-layer stack ({reason}). Grid: "
            f"lambda in [{lam[0]:.3g}, {lam[-1]:.3g}], delta="
            f"{delta:.3g}, structure scale 1/(2*h_max)="
            f"{1.0 / (2.0 * h_struct):.3g}, decay scale "
            f"1/(2*h_min)={1.0 / (2.0 * h_decay):.3g}. Raise "
            "n_samples / n_images, or widen lambda_max_factor when the "
            "thickness spread h_max/h_min is large. Do not use these "
            "images in a Green's function.",
            ComplexImageFitWarning,
            stacklevel=2,
        )
    return ComplexImageFit(
        a=a,
        beta=beta,
        rms=rms,
        k_inf=k_inf,
        rms_extrapolated=rms_wide,
        converged=converged,
    )


# ---------------------------------------------------------------------
# Self-action kernel
# ---------------------------------------------------------------------


def _cim_self_kernel_factory(
    stack: LayerStack,
    *,
    max_terms: int = 200,
    tol: float = 1e-6,
):
    """Build a self-action closure for the reachable ``cim`` regimes.

    Strategy:

    - ``n_layers == 1`` → homogeneous self-kernel ($\\Gamma_1 \\equiv 0$,
      no extra images); bit-identical to ``image``.
    - ``n_layers == 2`` → exact Tagg/Sunde self-kernel (closed form,
      bit-identical to ``image_2layer``), created with
      ``allow_cross_layer=True`` so a rod through the interface
      dispatches to the rigorous cross-layer path (ADR-0007) instead
      of silently applying the upper-layer series. $\\Gamma_1 = K_1$
      is constant in $\\lambda$ here, so the geometric image series
      *is* the complex-image representation (a single image at
      $\\beta = 0$) — evaluating it in closed form is both exact and
      cheaper than any fit.
    - ``n_layers >= 3`` → :class:`NotImplementedError`. The historic
      complex-image kernel for this regime implemented an incomplete
      Green's function (audit 2026-07-08, WP-E); :func:`solve_cim`
      rejects such soils before reaching this factory, and this branch
      keeps the module from growing a silent wrong-physics path again.
    """
    rho_1 = float(stack.rhos[0])

    if stack.n_layers <= 1:
        def _hom(seg_points, seg_lengths, wire_radii, currents):
            return _self_corrected_kernel(
                seg_points, seg_lengths, wire_radii, currents, rho_1
            )
        return _hom

    if stack.n_layers == 2:
        from groundfield.soil.models import TwoLayerSoil
        from groundfield.solver.image_2layer import (
            _two_layer_self_kernel_factory,
        )
        soil = TwoLayerSoil(
            rho_1=float(stack.rhos[0]),
            rho_2=float(stack.rhos[1]),
            h_1=float(stack.h[0]),
        )
        return _two_layer_self_kernel_factory(
            soil, max_terms=max_terms, tol=tol, allow_cross_layer=True,
        )

    raise NotImplementedError(
        f"cim: no self-kernel for n_layers = {stack.n_layers} >= 3 — "
        "the historic complex-image kernel was structurally incomplete "
        "(audit 2026-07-08). Use backend='mom_sommerfeld' or 'fem'."
    )


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_cim(world: "World", engine: "Engine") -> FieldResult:
    """Complex-Image-Method solver for layered soil.

    Accepts :class:`HomogeneousSoil` and :class:`TwoLayerSoil`;
    :class:`MultiLayerSoil` is accepted only while it reduces to
    $n \\le 2$ (see below).

    What actually runs
    ------------------
    - $n = 1$ → homogeneous image-charge self-kernel. Bit-identical
      to ``image``.
    - $n = 2$ → exact Tagg/Sunde series self-kernel. Bit-identical
      to ``image_2layer`` (which is why ``cim`` is *not* an
      independent cross-check of it — ADR-0002 amendment 2026-07-09).
    - $n \\ge 3$ → :class:`NotImplementedError` (audit 2026-07-08,
      WP-E). Use ``mom_sommerfeld`` or ``fem``.

    **No complex-image fit is evaluated.** Up to 0.14.1 this solver
    called :func:`fit_complex_images` on every solve and published its
    diagnostics even though neither reachable path consumed them
    (review pass 9, F34); the metadata now says so explicitly with
    ``cim_fit_used = False`` instead of reporting a NaN residual.

    Parameters
    ----------
    world
        World to evaluate.
    engine
        Engine configuration; ``engine.segment_length`` controls the
        discretisation, ``engine.image_max_terms`` /
        ``engine.image_series_tol`` the Tagg/Sunde truncation at
        $n = 2$.

    Returns
    -------
    FieldResult
        ``metadata['cim_fit_used'] = False`` and
        ``metadata['reduces_to']`` name the backend this result is
        bit-identical to.
    """
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'cim' supports HomogeneousSoil, TwoLayerSoil, "
            f"and MultiLayerSoil. Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")
    _reject_concrete_shells(world, "cim")
    _warn_ignored_sources(world, "cim")

    stack = as_layer_stack(world.soil)
    if stack.n_layers >= 3:
        # Audit 2026-07-08, WP-E: the historic n>=3 CIM kernel
        # implemented an incomplete Green's function (single
        # (z+z_s)-type image family; missing the 2*h_1 families and
        # the surface-interface multiple-reflection denominator — see
        # solver/_layered.py). Reject loudly until a complete kernel
        # exists.
        raise NotImplementedError(
            f"cim: n_layers = {stack.n_layers} >= 3 is not supported — "
            "the historic complex-image kernel was structurally "
            "incomplete (audit 2026-07-08). Use "
            "backend='mom_sommerfeld' (full layered Green's function) "
            "or 'fem' for n >= 3 soils."
        )
    ds = engine.segment_length
    reduces_to = "image_2layer" if stack.n_layers == 2 else "image"

    _log.info(
        "cim: n_layers=%d, closed-form self-kernel (no complex-image "
        "fit), bit-identical to backend '%s'",
        stack.n_layers, reduces_to,
    )

    # 1) Discretisation.
    all_segments: list[_Segment] = []
    elec_to_segidx: dict[str, list[int]] = {}
    interfaces = (
        (float(stack.h[0]),) if stack.n_layers >= 2 else None
    )
    for e in world.electrodes:
        segs = _discretize_electrode(e, ds, layer_interfaces=interfaces)
        elec_to_segidx[e.name] = list(
            range(len(all_segments), len(all_segments) + len(segs))
        )
        all_segments.extend(segs)

    # 2) Per-electrode input currents.
    elec_input_current: dict[str, complex] = {
        e.name: 0j for e in world.electrodes
    }
    for src in world.sources:
        if src.kind != "current":
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in elec_input_current:
            elec_input_current[src.attached_to] += i_complex

    # 3) Cluster building (ideal conductors only) and finite-impedance
    #    branch list (passed into the nodal-analysis solver).
    cluster_id = _build_clusters(world.electrodes, world.conductors)
    finite_branches = _build_finite_branches(world.conductors, cluster_id)

    # 3b) Distributed-conductor topology (ADR-0003) + ADR-0004
    #     inductive coupling assembly.
    cond_segs, distributed_branches_objs, interior_nodes = _build_distributed_topology(
        world.conductors, cluster_id
    )
    pseudo_owners: list[str] = []
    for s in cond_segs:
        pn = s.electrode_name
        elec_to_segidx[pn] = [len(all_segments)]
        all_segments.append(s)
        cluster_id[pn] = pn
        pseudo_owners.append(pn)
    for n_ in interior_nodes:
        if n_ not in cluster_id:
            cluster_id[n_] = n_
            pseudo_owners.append(n_)
            elec_to_segidx[n_] = []
    n_lumped_branches = len(finite_branches)
    distributed_branch_tuples = [
        (db.node_a, db.node_b, db.R) for db in distributed_branches_objs
    ]
    finite_branches = list(finite_branches) + distributed_branch_tuples
    earth_inductive_model = getattr(
        engine, "earth_inductive_model", "perfect_mirror"
    )
    sigma_earth_for_carson: float | None = None
    layered_earth_for_sommerfeld: object = None
    if earth_inductive_model == "carson_series":
        from groundfield.coupling import resolve_earth_conductivity

        sigma_earth_for_carson = resolve_earth_conductivity(world.soil)
    elif earth_inductive_model == "sommerfeld":
        from groundfield.coupling import resolve_earth_layers

        layered_earth_for_sommerfeld = resolve_earth_layers(world.soil)
    inductance_matrix_full, has_inductance, carson_builder = _assemble_inductance_matrix(
        distributed_branches_objs,
        n_lumped_branches=n_lumped_branches,
        n_total_branches=len(finite_branches),
        earth_model=earth_inductive_model,
        sigma_earth=sigma_earth_for_carson,
        layered_earth=layered_earth_for_sommerfeld,
    )

    seg_points = np.array([s.midpoint for s in all_segments])
    seg_lengths = np.array([s.length for s in all_segments])
    wire_radii = np.array([s.wire_radius for s in all_segments])

    # 4) Self-kernel + frequency loop. The Tagg/Sunde truncation
    #    knobs come from the engine (consistent with image_2layer).
    #    n>=3 cross-layer geometries needed a guard here while the
    #    incomplete n>=3 kernel existed; the n>=3 rejection above makes
    #    that guard unreachable, so it was removed in 0.15.0 (n=2
    #    cross-layer is handled rigorously by
    #    _two_layer_self_kernel_factory(allow_cross_layer=True),
    #    ADR-0007).
    self_kernel = _cim_self_kernel_factory(
        stack,
        max_terms=engine.image_max_terms, tol=engine.image_series_tol,
    )
    n_segments = len(all_segments)
    n_freq = len(engine.frequencies)
    omegas = [2.0 * np.pi * float(f) for f in engine.frequencies]
    real_electrode_names = {e.name for e in world.electrodes}

    # ADR-0010 Tier 1 (WP-F): the multi-port grounding matrix Z is
    # frequency-independent — share it across the per-frequency calls.
    _mp_cache: dict = {}

    def _solve_at(omega: float) -> tuple[dict[str, complex], np.ndarray]:
        carson_dz = (
            carson_builder(omega) if (has_inductance and carson_builder is not None)
            else None
        )
        elec_total = _solve_cluster_currents(
            electrodes=world.electrodes,
            elec_input_current=elec_input_current,
            cluster_id=cluster_id,
            seg_points=seg_points,
            seg_lengths=seg_lengths,
            wire_radii=wire_radii,
            elec_to_segidx=elec_to_segidx,
            self_kernel=self_kernel,
            finite_branches=finite_branches,
            pseudo_owners=pseudo_owners,
            omega=omega if has_inductance else 0.0,
            inductance_matrix=inductance_matrix_full if has_inductance else None,
            carson_correction=carson_dz,
            multiport_cache=_mp_cache,
        )
        sc = np.zeros(n_segments, dtype=complex)
        for ename, idxs in elec_to_segidx.items():
            if not idxs:
                continue
            I_total = elec_total.get(ename, 0j)
            if I_total == 0j:
                continue
            L_total = seg_lengths[idxs].sum()
            sc[idxs] = I_total * seg_lengths[idxs] / L_total
        return elec_total, sc

    def _phi_batch(sc_list: list[np.ndarray]) -> list[np.ndarray]:
        """Batched segment-potential evaluation (ADR-0010 Tier 1)."""
        k = len(sc_list)
        stacked = np.zeros((n_segments, 2 * k))
        for m, sc in enumerate(sc_list):
            stacked[:, m] = sc.real
            stacked[:, k + m] = sc.imag
        if not stacked.any():
            return [np.zeros(n_segments, dtype=complex)] * k
        phi = self_kernel(seg_points, seg_lengths, wire_radii, stacked)
        return [phi[:, m] + 1j * phi[:, k + m] for m in range(k)]

    elec_per_freq: list[dict[str, complex]] = []
    sc_per_freq: list[np.ndarray] = []
    phi_per_freq: list[np.ndarray] = []
    if has_inductance:
        for omega in omegas:
            et, sc = _solve_at(omega)
            elec_per_freq.append(et)
            sc_per_freq.append(sc)
        phi_per_freq = _phi_batch(sc_per_freq)
    else:
        et, sc = _solve_at(0.0)
        ph = _phi_batch([sc])[0]
        elec_per_freq = [et] * n_freq
        sc_per_freq = [sc] * n_freq
        phi_per_freq = [ph] * n_freq

    electrode_potentials: dict[str, list[complex]] = {}
    electrode_currents: dict[str, list[complex]] = {}
    conductor_currents: dict[str, list[complex]] = {}
    conductor_potentials: dict[str, list[complex]] = {}
    for ename, idxs in elec_to_segidx.items():
        if not idxs:
            continue
        u_list = [
            complex(np.mean(phi_per_freq[k][idxs])) for k in range(n_freq)
        ]
        i_list = [elec_per_freq[k][ename] for k in range(n_freq)]
        if ename in real_electrode_names:
            electrode_potentials[ename] = u_list
            electrode_currents[ename] = i_list
        else:
            conductor_potentials[ename] = u_list
            conductor_currents[ename] = i_list

    point_sources = [
        PointSource(
            position=tuple(seg_points[i].tolist()),
            current=[complex(sc_per_freq[k][i]) for k in range(n_freq)],
            electrode_name=all_segments[i].electrode_name,
            length=float(seg_lengths[i]),
        )
        for i in range(n_segments)
    ]
    cluster_members: dict[str, list[str]] = {}
    for ename in real_electrode_names:
        cluster_members[ename] = sorted(
            n for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_electrode_names
        )

    metadata = {
        "world_name": world.name,
        "n_segments": n_segments,
        "segment_length": ds,
        "n_layers": int(stack.n_layers),
        "rhos": stack.rhos.tolist(),
        "h": stack.h.tolist(),
        # Honesty flags (review pass 9, F34): no complex-image fit is
        # evaluated on the reachable n <= 2 paths, so there is no fit
        # quality to report — and the numbers below are bit-identical
        # to `reduces_to`, i.e. cim is not an independent cross-check.
        "cim_fit_used": False,
        "cim_n_images": 0,
        "cim_rms": None,
        "reduces_to": reduces_to,
        "stub": False,
        "earth_inductive_model": earth_inductive_model,
    }
    if has_inductance:
        from groundfield.coupling.carson import skin_depth

        sigma_ref = (
            sigma_earth_for_carson
            if sigma_earth_for_carson is not None
            else 1.0 / float(stack.rhos[0])
        )
        metadata["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma_ref)
            for f in engine.frequencies
        }
    if conductor_currents:
        metadata["conductor_node_currents"] = conductor_currents
        metadata["conductor_node_potentials"] = conductor_potentials

    return FieldResult(
        backend="cim",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=float(stack.rhos[0]),
        soil=world.soil,
        clusters=cluster_members,
        metadata=metadata,
    )
