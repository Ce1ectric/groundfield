"""Self- and mutual-inductance helpers for distributed conductors.

This module implements the inductance physics described in ADR-0004.
It provides three public functions:

- :func:`thin_wire_self_inductance` — closed-form self-inductance of
  a straight thin-wire segment of length $\\ell$ and radius $a$
  (Grover 1946):
  $L_\\text{self} = (\\mu_0\\,\\ell)/(2\\pi)
   \\bigl[\\ln(2\\ell/a) - 1\\bigr]$.
- :func:`parallel_segments_mutual` — closed-form mutual inductance
  of two parallel coaxial segments of equal length $\\ell$ at
  perpendicular distance $d$.
- :func:`neumann_mutual` — generic Neumann double-line integral via
  two-point Gauss–Legendre quadrature, valid for any 3-D
  segment-pair geometry.

Earth-image contributions are handled by the caller: for a
perfect-mirror earth the caller adds the integral against the
mirror image of the source segment (see
:func:`perfect_mirror_self_pair_inductance` and the assembly logic
in :func:`build_inductance_matrix`).

References
----------
- Grover, F. W. (1946). *Inductance Calculations: Working Formulas
  and Tables*. Dover (reprint 2004).
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*. Dover, ch. 7.
- Paul, C. R. (2010). *Inductance: Loop and Partial*. Wiley.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "MU_0",
    "thin_wire_self_inductance",
    "parallel_segments_mutual",
    "neumann_mutual",
    "perfect_mirror_self_pair_inductance",
    "build_inductance_matrix",
    "_build_inductance_matrix_loop",  # Regression-test reference (ADR-0010)
    "build_carson_correction_matrix",
]

# Vacuum permeability in H/m (CODATA 2018).
MU_0 = 4.0e-7 * math.pi


# ---------------------------------------------------------------------
# Closed-form helpers (used as analytical references and fast paths)
# ---------------------------------------------------------------------


def thin_wire_self_inductance(
    length: float,
    wire_radius: float,
    *,
    include_internal: bool = True,
) -> float:
    """Closed-form self-inductance of a straight thin-wire segment.

    Decomposed into the **external** Grover 1946 thin-wire term and
    the **internal** contribution from the magnetic field inside the
    conductor (uniform DC current distribution, non-magnetic
    material):

    $$
    L_\\text{ext} \\;=\\; \\frac{\\mu_0\\,\\ell}{2\\pi}
    \\Bigl[\\ln\\!\\Bigl(\\frac{2\\ell}{a}\\Bigr) - 1\\Bigr],
    \\qquad
    L_\\text{int} \\;=\\; \\frac{\\mu_0\\,\\ell}{8\\pi}.
    $$

    Adding the internal term reproduces the standard Oeding/Oswald
    loop-inductance formula (Gl. 9.13c) when used as the diagonal
    contribution of a two-wire loop. For high-frequency studies
    where the current flows on the surface only (skin effect at
    $f \\gg f_\\text{skin}$), pass ``include_internal=False`` and
    the external term alone is returned.

    Parameters
    ----------
    length
        Segment length $\\ell$ in metres.
    wire_radius
        Wire radius $a$ in metres.
    include_internal
        ``True`` (default) — add the internal-field contribution
        $\\mu_0 \\ell / (8\\pi)$. Appropriate for the
        $f < 1\\,\\mathrm{kHz}$ regime with mostly uniform
        current distribution. Set ``False`` for the pure
        external-field thin-wire formula.

    Returns
    -------
    L : float
        Partial self-inductance in henries.
    """
    if length <= 0.0 or wire_radius <= 0.0:
        raise ValueError("length and wire_radius must be positive")
    L_ext = (MU_0 * length) / (2.0 * math.pi) * (
        math.log(2.0 * length / wire_radius) - 1.0
    )
    if not include_internal:
        return L_ext
    L_int = (MU_0 * length) / (8.0 * math.pi)
    return L_ext + L_int


def parallel_segments_mutual(length: float, distance: float) -> float:
    """Closed-form mutual inductance of two parallel coaxial segments.

    For two segments of equal length $\\ell$ that share the same
    axis direction and are placed at perpendicular distance $d$,

    $$
    M_\\parallel(\\ell, d) \\;=\\; \\frac{\\mu_0\\,\\ell}{2\\pi}
    \\Bigl[\\ln\\!\\Bigl(\\frac{\\ell + \\sqrt{\\ell^2 + d^2}}{d}\\Bigr)
          - \\frac{\\sqrt{\\ell^2 + d^2} - d}{\\ell}\\Bigr].
    $$

    Used both as an internal closed-form fast path (when the segment
    pair geometry matches the assumption) and as a reference in the
    test suite.

    Parameters
    ----------
    length
        Segment length $\\ell$ in m (both segments have the same).
    distance
        Perpendicular distance $d$ between the two parallel axes
        in m. Must be > 0.

    Returns
    -------
    M : float
        Partial mutual inductance in H.
    """
    if length <= 0.0 or distance <= 0.0:
        raise ValueError("length and distance must be positive")
    s = math.hypot(length, distance)
    return (MU_0 * length) / (2.0 * math.pi) * (
        math.log((length + s) / distance) - (s - distance) / length
    )


# ---------------------------------------------------------------------
# Neumann quadrature for arbitrary segment pairs
# ---------------------------------------------------------------------


# 16-point Gauss–Legendre nodes and weights on [-1, 1].
#
# Chosen empirically: across the relevant geometry range
# (sub-segment lengths 0.5–5 m, segment-pair distances 0.1–10 m,
# i.e. ℓ/d ≲ 10), 16-point Gauss–Legendre keeps the Neumann
# integral within ≲ 0.05 % of the closed-form parallel-segments
# reference. 8-point would also be acceptable for ℓ/d ≤ 4 but
# breaks above (≈ 1 % at ℓ/d = 10), so we go with 16-point as a
# safe default. Per-pair cost is 16 × 16 = 256 1/r evaluations,
# fully vectorised — fast enough for the dense O(M²) assembly.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _parallel_filaments_mutual(
    p1_a: np.ndarray, p2_a: np.ndarray,
    p1_b: np.ndarray, p2_b: np.ndarray,
    *,
    parallel_tol: float = 1e-9,
) -> float | None:
    """Closed-form mutual inductance for two parallel straight filaments.

    Returns ``None`` when the two segments are not parallel (and the
    caller should fall back to the Neumann quadrature). When they
    are parallel — possibly with a longitudinal offset and possibly
    anti-parallel — the integral is the standard Grover formula

    $$
    M \\;=\\; \\frac{\\mu_0\\,\\sigma}{4\\pi}\\,
    \\bigl[F(\\beta_2) - F(\\beta_2 - \\ell_a) - F(\\beta_1) + F(\\beta_1 - \\ell_a)\\bigr],
    $$

    with $F(x) = x\\,\\sinh^{-1}(x/d) - \\sqrt{x^2 + d^2}$,
    $\\sigma = \\mathrm{sign}(\\hat{u}_a \\cdot \\hat{u}_b)$,
    $\\beta_1 = (p_1^b - p_1^a) \\cdot \\hat{u}_a$,
    $\\beta_2 = (p_2^b - p_1^a) \\cdot \\hat{u}_a$, and $d$ the
    constant perpendicular distance between the two parallel axes.

    Cheap, exact (no quadrature error), and the typical case in
    typical — every PEN strand and every pair of consecutive
    sub-segments along the same conductor falls into it.
    """
    da = p2_a - p1_a
    db = p2_b - p1_b
    la = float(np.linalg.norm(da))
    lb = float(np.linalg.norm(db))
    if la <= 0.0 or lb <= 0.0:
        return None
    ua = da / la
    ub = db / lb
    dot = float(ua @ ub)
    # Need parallel or anti-parallel within tolerance.
    if abs(abs(dot) - 1.0) > parallel_tol:
        return None
    # Perpendicular component of the offset must be the same at both
    # ends of segment b (otherwise the axes are not truly parallel).
    rel1 = p1_b - p1_a
    rel2 = p2_b - p1_a
    perp1 = rel1 - (rel1 @ ua) * ua
    perp2 = rel2 - (rel2 @ ua) * ua
    if np.linalg.norm(perp2 - perp1) > parallel_tol:
        return None
    d = float(np.linalg.norm(perp1))
    if d == 0.0:
        # Coaxial — the integral diverges, fall back to the
        # quadrature, which clamps at ``min_distance``.
        return None
    # Project segment b onto a's axis, in the direction of ua.
    # For an anti-parallel b (dot ≈ −1) the projection is
    # decreasing in s_b; we encode the anti-parallel case via a
    # sign and a swapped pair of beta endpoints.
    s_b1 = float(rel1 @ ua)
    s_b2 = float(rel2 @ ua)
    sigma = 1.0 if dot > 0 else -1.0
    if sigma < 0:
        # b runs against ua: swap the projection endpoints so that
        # beta_1 < beta_2 holds for both orientations.
        s_b1, s_b2 = s_b2, s_b1

    def F(x: float, d: float) -> float:
        return x * math.asinh(x / d) - math.hypot(x, d)

    M = (MU_0 / (4.0 * math.pi)) * sigma * (
        F(s_b2, d) - F(s_b2 - la, d) - F(s_b1, d) + F(s_b1 - la, d)
    )
    return M


def neumann_mutual(
    p1_a: np.ndarray, p2_a: np.ndarray,
    p1_b: np.ndarray, p2_b: np.ndarray,
    *,
    min_distance: float = 1e-6,
    parallel_tol: float = 1e-9,
) -> float:
    """Neumann mutual-inductance integral between two straight segments.

    Hybrid implementation:

    1. If the two segments are **parallel** (or anti-parallel) within
       ``parallel_tol``, the closed-form Grover expression in
       :func:`_parallel_filaments_mutual` is used — exact, no
       quadrature error. This fast path covers the bulk of the typical
       inductance assembly (parallel PEN strands, consecutive
       sub-segments along the same conductor).
    2. Otherwise the Neumann double integral is evaluated by 16×16
       Gauss–Legendre quadrature. Empirically accurate to ≲ 0.05 %
       across the typical geometry range; see ADR-0004 for the
       calibration data.

    The kernel $1/r$ is clamped at ``min_distance`` to suppress the
    integrable singularity for segments that touch — physically the
    diagonal uses :func:`thin_wire_self_inductance` instead, so
    this only affects pathological inputs.

    Parameters
    ----------
    p1_a, p2_a
        Endpoints of segment *a* as ``(x, y, z)`` arrays in metres.
    p1_b, p2_b
        Endpoints of segment *b*.
    min_distance
        Numerical floor on $|r_a - r_b|$ in metres.
    parallel_tol
        Tolerance on $||\\hat{u}_a\\cdot\\hat{u}_b| - 1|$ that
        triggers the closed-form fast path.

    Returns
    -------
    M : float
        Partial mutual inductance in H.
    """
    p1_a = np.asarray(p1_a, dtype=float)
    p2_a = np.asarray(p2_a, dtype=float)
    p1_b = np.asarray(p1_b, dtype=float)
    p2_b = np.asarray(p2_b, dtype=float)

    # Fast path: parallel filaments with closed form.
    M_par = _parallel_filaments_mutual(
        p1_a, p2_a, p1_b, p2_b, parallel_tol=parallel_tol,
    )
    if M_par is not None:
        return M_par

    # Generic 3-D case: Gauss–Legendre quadrature.
    da = p2_a - p1_a
    db = p2_b - p1_b
    la = float(np.linalg.norm(da))
    lb = float(np.linalg.norm(db))
    if la <= 0.0 or lb <= 0.0:
        raise ValueError("segment lengths must be positive")
    ua = da / la
    ub = db / lb
    dot = float(ua @ ub)
    if dot == 0.0:
        return 0.0  # perpendicular — orthogonality kills the integral
    s_nodes = 0.5 * (_GL_NODES + 1.0)
    w_nodes = 0.5 * _GL_WEIGHTS
    pts_a = p1_a[None, :] + s_nodes[:, None] * da[None, :]
    pts_b = p1_b[None, :] + s_nodes[:, None] * db[None, :]
    diff = pts_a[:, None, :] - pts_b[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.maximum(r, min_distance, out=r)
    inv_r = 1.0 / r
    integral = float(np.einsum("i,j,ij->", w_nodes, w_nodes, inv_r))
    return MU_0 / (4.0 * math.pi) * dot * la * lb * integral


# ---------------------------------------------------------------------
# Perfect-mirror earth: image contributions
# ---------------------------------------------------------------------


def _mirror(p: np.ndarray) -> np.ndarray:
    """Reflect a point at the soil surface (z → -z).

    Inside ``groundfield`` the z-axis points *into* the soil. The
    mirror image of a buried source therefore sits above the
    surface at $z' = -z$.
    """
    out = p.copy()
    out[2] = -out[2]
    return out


def perfect_mirror_self_pair_inductance(
    p1: np.ndarray, p2: np.ndarray, wire_radius: float
) -> float:
    """Self-inductance contribution of a segment plus its earth image.

    For a single segment the *partial* self-inductance against itself
    uses the thin-wire formula. The image segment (mirrored at the
    soil surface) is treated as an external segment whose Neumann
    integral with the original is added to the total.

    Parameters
    ----------
    p1, p2
        Endpoints in metres (with $z > 0$ pointing into the soil).
    wire_radius
        Wire radius in m (only used for the thin-wire self term).

    Returns
    -------
    L : float
        Self-inductance plus image contribution, in henries.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    L_self = thin_wire_self_inductance(
        float(np.linalg.norm(p2 - p1)), wire_radius,
    )
    p1_img = _mirror(p1)
    p2_img = _mirror(p2)
    # Image segment runs in the opposite direction (current image of a
    # vertical filament is anti-parallel; for a horizontal filament it
    # is parallel). The general rule is: image of (p1 → p2) is
    # (p1' → p2'), with the *same* tangent direction up to the
    # sign of ẑ. For Neumann the dot product ǔ_a·ǔ_b takes care of
    # the sign automatically — we just feed the mirrored endpoints.
    L_image = neumann_mutual(p1, p2, p1_img, p2_img)
    return L_self + L_image


def _build_inductance_matrix_loop(
    seg_endpoints: np.ndarray,
    wire_radii: np.ndarray,
    *,
    use_image: bool = True,
) -> np.ndarray:
    """Reference (loop-based) implementation of :func:`build_inductance_matrix`.

    Kept verbatim from the pre-vectorisation code path. Used by the
    Tier-0 regression tests as the oracle the vectorised assembly
    must reproduce bit-exactly. New callers should use
    :func:`build_inductance_matrix`, which dispatches to the
    vectorised path described in ADR-0010.
    """
    M = seg_endpoints.shape[0]
    L = np.zeros((M, M), dtype=float)
    for i in range(M):
        p1_i = seg_endpoints[i, 0]
        p2_i = seg_endpoints[i, 1]
        # Diagonal — self plus image
        if use_image:
            L[i, i] = perfect_mirror_self_pair_inductance(
                p1_i, p2_i, wire_radii[i],
            )
        else:
            li = float(np.linalg.norm(p2_i - p1_i))
            L[i, i] = thin_wire_self_inductance(li, wire_radii[i])
        # Off-diagonal — Neumann + image of partner
        for j in range(i + 1, M):
            p1_j = seg_endpoints[j, 0]
            p2_j = seg_endpoints[j, 1]
            m = neumann_mutual(p1_i, p2_i, p1_j, p2_j)
            if use_image:
                p1_j_img = _mirror(p1_j)
                p2_j_img = _mirror(p2_j)
                m += neumann_mutual(p1_i, p2_i, p1_j_img, p2_j_img)
            L[i, j] = L[j, i] = m
    return L


# ---------------------------------------------------------------------
# ADR-0010 (Tier 0b): vectorised partial-inductance assembly
# ---------------------------------------------------------------------


def _parallel_filaments_mutual_batch(
    p1_a: np.ndarray, ua_a: np.ndarray, la_a: float,
    p1_b: np.ndarray, ub_b: np.ndarray, lb_b: np.ndarray,
    dots: np.ndarray,
    *,
    parallel_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Grover closed-form for K parallel pairs.

    One reference segment *a* vs. K candidate segments *b*. Returns
    ``(M, mask)``: ``M[k]`` is the closed-form mutual inductance
    when ``mask[k]`` is true, undefined otherwise. The caller falls
    back to the Neumann quadrature for the non-masked entries.

    Parameters
    ----------
    p1_a, ua_a, la_a
        Reference segment: start point, unit vector, length.
    p1_b, ub_b, lb_b
        K candidate segments: start points (K, 3), unit vectors
        (K, 3), lengths (K,).
    dots
        Pre-computed dot products ``ua_a @ ub_b.T``, shape (K,).
    """
    K = p1_b.shape[0]
    M_out = np.zeros(K, dtype=float)
    valid = np.zeros(K, dtype=bool)
    if K == 0:
        return M_out, valid

    # Parallel-or-anti-parallel mask
    near_parallel = np.abs(np.abs(dots) - 1.0) < parallel_tol

    # The perpendicular component of the offset must be the same at
    # both ends of segment b (otherwise the axes are not truly
    # parallel — numerical noise on dots can pass the test even when
    # the segments are skew). We re-check via the perp-component test
    # used in the loop version.
    rel1 = p1_b - p1_a[None, :]                                  # (K, 3)
    rel2 = (p1_b + ub_b * lb_b[:, None]) - p1_a[None, :]          # (K, 3) — this is p2_b - p1_a
    # Project onto ua_a; the "perpendicular component" is rel - (rel·ua)·ua.
    s1 = rel1 @ ua_a                                              # (K,)
    s2 = rel2 @ ua_a                                              # (K,)
    perp1 = rel1 - s1[:, None] * ua_a[None, :]                    # (K, 3)
    perp2 = rel2 - s2[:, None] * ua_a[None, :]                    # (K, 3)
    delta_perp = np.linalg.norm(perp2 - perp1, axis=1)            # (K,)
    skew_mask = delta_perp > parallel_tol
    parallel_mask = near_parallel & ~skew_mask

    if not parallel_mask.any():
        return M_out, valid

    # Distance d (constant along the parallel pair). When d == 0 the
    # closed form diverges and the caller must use the quadrature.
    d = np.linalg.norm(perp1, axis=1)                              # (K,)
    coaxial = d == 0.0
    parallel_mask &= ~coaxial
    if not parallel_mask.any():
        return M_out, valid

    # For anti-parallel b we swap (s1, s2) so that beta_1 < beta_2
    # holds for both orientations.
    sigma = np.where(dots > 0.0, 1.0, -1.0)                        # (K,)
    s_b1 = np.where(sigma > 0.0, s1, s2)                           # (K,)
    s_b2 = np.where(sigma > 0.0, s2, s1)                           # (K,)

    # F(x, d) = x * arcsinh(x / d) - sqrt(x^2 + d^2)
    def F(x: np.ndarray, d_loc: np.ndarray) -> np.ndarray:
        return x * np.arcsinh(x / d_loc) - np.hypot(x, d_loc)

    # Only evaluate where parallel_mask is True; use a safe d to
    # avoid division warnings on the masked-out entries.
    d_safe = np.where(parallel_mask, d, 1.0)
    F_b2 = F(s_b2, d_safe)
    F_b2_minus_la = F(s_b2 - la_a, d_safe)
    F_b1 = F(s_b1, d_safe)
    F_b1_minus_la = F(s_b1 - la_a, d_safe)
    M_par = (MU_0 / (4.0 * math.pi)) * sigma * (
        F_b2 - F_b2_minus_la - F_b1 + F_b1_minus_la
    )
    M_out[parallel_mask] = M_par[parallel_mask]
    valid[parallel_mask] = True
    return M_out, valid


def _neumann_quadrature_batch(
    p1_a: np.ndarray, da_a: np.ndarray, la_a: float, ua_a: np.ndarray,
    p1_b: np.ndarray, db_b: np.ndarray, lb_b: np.ndarray, ub_b: np.ndarray,
    dots: np.ndarray,
    *,
    min_distance: float = 1e-6,
) -> np.ndarray:
    """Vectorised 16×16 Gauss–Legendre Neumann integral for K pairs.

    One reference segment *a* against K candidate segments *b*,
    evaluated in a single batched NumPy call. The batched diff
    array has shape ``(K, 16, 16, 3)`` — peak memory
    $K \\cdot 256 \\cdot 24$ bytes.
    """
    K = p1_b.shape[0]
    if K == 0:
        return np.zeros(0, dtype=float)

    s_nodes = 0.5 * (_GL_NODES + 1.0)         # (16,)
    w_nodes = 0.5 * _GL_WEIGHTS                # (16,)

    # Quadrature points along a and along each b
    pts_a = p1_a[None, :] + s_nodes[:, None] * da_a[None, :]   # (16, 3)
    pts_b = (p1_b[:, None, :]
             + s_nodes[None, :, None] * db_b[:, None, :])      # (K, 16, 3)

    # diff[k, i, j, :] = pts_a[i, :] - pts_b[k, j, :]
    diff = pts_a[None, :, None, :] - pts_b[:, None, :, :]      # (K, 16, 16, 3)
    r = np.linalg.norm(diff, axis=-1)                          # (K, 16, 16)
    np.maximum(r, min_distance, out=r)
    inv_r = 1.0 / r
    # integral[k] = sum_{i,j} w_i * w_j * inv_r[k, i, j]
    integrals = np.einsum("i,j,kij->k", w_nodes, w_nodes, inv_r)   # (K,)

    M = (MU_0 / (4.0 * math.pi)) * dots * la_a * lb_b * integrals
    # Pairs with dot == 0 contribute nothing (handled by ``dots`` factor).
    return M


def build_inductance_matrix(
    seg_endpoints: np.ndarray,        # shape (M, 2, 3): each branch's [start, end]
    wire_radii: np.ndarray,           # shape (M,)
    *,
    use_image: bool = True,
) -> np.ndarray:
    """Assemble the dense partial-inductance matrix over branches.

    Vectorised implementation per ADR-0010 Tier 0b. Reproduces
    :func:`_build_inductance_matrix_loop` bit-exactly to floating-
    point precision but evaluates the off-diagonal entries one row
    at a time in batched NumPy calls — typical networks
    (~1000 segments) speed up by 1–2 orders of magnitude.

    Parameters
    ----------
    seg_endpoints
        Array of shape ``(M, 2, 3)`` with the start- and end-points
        of every distributed-conductor longitudinal-branch segment.
    wire_radii
        Per-branch wire radii in metres, shape ``(M,)``.
    use_image
        When ``True`` (default) the earth is treated as a perfect
        magnetic mirror: every branch's image (z → -z) is summed
        into both the diagonal (via
        :func:`perfect_mirror_self_pair_inductance`) and the
        off-diagonal (one extra Neumann integral against the
        mirrored partner).

    Returns
    -------
    L : np.ndarray, shape (M, M)
        Symmetric partial-inductance matrix in H. Entry ``L[i, j]``
        is the partial mutual inductance between branch *i* and
        branch *j* (and the diagonal is the self-inductance + image).

    Notes
    -----
    The legacy loop-based implementation is kept as
    :func:`_build_inductance_matrix_loop` for regression testing.
    See :file:`tests/test_inductance_vectorised.py` for the
    bit-exact regression suite.
    """
    M = seg_endpoints.shape[0]
    if M == 0:
        return np.zeros((0, 0), dtype=float)

    L = np.zeros((M, M), dtype=float)

    # Pre-compute per-segment quantities that the loop would
    # otherwise rebuild on every call.
    p1 = seg_endpoints[:, 0, :].astype(float, copy=False)        # (M, 3)
    p2 = seg_endpoints[:, 1, :].astype(float, copy=False)        # (M, 3)
    da = p2 - p1                                                  # (M, 3)
    la = np.linalg.norm(da, axis=1)                               # (M,)
    if (la <= 0.0).any():
        raise ValueError("segment lengths must be positive")
    ua = da / la[:, None]                                         # (M, 3)

    # Mirrored endpoints (for the image contribution). Mirror by
    # flipping z; same shape and tangent direction up to a sign — the
    # Neumann integrand handles the sign via the dot product.
    if use_image:
        p1_img = p1.copy(); p1_img[:, 2] = -p1_img[:, 2]
        p2_img = p2.copy(); p2_img[:, 2] = -p2_img[:, 2]
        da_img = p2_img - p1_img
        la_img = np.linalg.norm(da_img, axis=1)
        ua_img = da_img / la_img[:, None]

    # Diagonal — self plus own-image. Cheap (M operations); leave it
    # as a Python loop so we can call the existing scalar helpers.
    for i in range(M):
        if use_image:
            L[i, i] = perfect_mirror_self_pair_inductance(
                p1[i], p2[i], wire_radii[i],
            )
        else:
            L[i, i] = thin_wire_self_inductance(la[i], wire_radii[i])

    # Off-diagonal — for each row i, compute L[i, i+1:M] as a batch.
    for i in range(M - 1):
        slc = slice(i + 1, M)
        K = M - i - 1

        ua_a = ua[i]                                              # (3,)
        la_a = float(la[i])
        da_a = da[i]                                              # (3,)
        p1_a = p1[i]                                              # (3,)

        ua_b = ua[slc]                                            # (K, 3)
        ub_b = ua_b   # alias for clarity below
        la_b = la[slc]                                            # (K,)
        db_b = da[slc]                                            # (K, 3)
        p1_b = p1[slc]                                            # (K, 3)

        # Pairwise dots between segment i and each j > i
        dots = ub_b @ ua_a                                        # (K,)

        # ---- Primary contribution: a vs. b ----
        # Closed form for parallel pairs, quadrature for the rest.
        M_par, par_mask = _parallel_filaments_mutual_batch(
            p1_a, ua_a, la_a, p1_b, ub_b, la_b, dots,
        )
        M_pri = np.where(par_mask, M_par, 0.0)
        npar_mask = ~par_mask
        if npar_mask.any():
            ks = np.where(npar_mask)[0]
            M_quad = _neumann_quadrature_batch(
                p1_a, da_a, la_a, ua_a,
                p1_b[ks], db_b[ks], la_b[ks], ub_b[ks],
                dots[ks],
            )
            M_pri[ks] = M_quad

        # ---- Image contribution: a vs. mirror(b) ----
        if use_image:
            ua_b_img = ua_img[slc]                                # (K, 3)
            la_b_img = la_img[slc]                                # (K,)
            db_b_img = (p2_img[slc] - p1_img[slc])                # (K, 3)
            p1_b_img = p1_img[slc]                                # (K, 3)
            dots_img = ua_b_img @ ua_a                            # (K,)

            M_par_img, par_mask_img = _parallel_filaments_mutual_batch(
                p1_a, ua_a, la_a,
                p1_b_img, ua_b_img, la_b_img, dots_img,
            )
            M_img = np.where(par_mask_img, M_par_img, 0.0)
            npar_mask_img = ~par_mask_img
            if npar_mask_img.any():
                ks = np.where(npar_mask_img)[0]
                M_quad_img = _neumann_quadrature_batch(
                    p1_a, da_a, la_a, ua_a,
                    p1_b_img[ks], db_b_img[ks], la_b_img[ks], ua_b_img[ks],
                    dots_img[ks],
                )
                M_img[ks] = M_quad_img
            M_pri = M_pri + M_img

        L[i, slc] = M_pri
        L[slc, i] = M_pri

    return L


# ---------------------------------------------------------------------
# ADR-0005: Carson earth-return correction matrix
# ---------------------------------------------------------------------


def build_carson_correction_matrix(
    seg_endpoints: np.ndarray,        # shape (M, 2, 3)
    wire_radii: np.ndarray,           # shape (M,)
    *,
    omega: float,
    sigma_earth: float,
) -> np.ndarray:
    """Assemble the dense Carson earth-return correction matrix.

    Adds the per-segment-pair impedance correction
    $\\Delta Z^{(i,j)}_\\text{Carson}(\\omega)$ described in ADR-0005
    on top of the perfect-mirror Neumann inductance matrix. The
    output has the same shape as :func:`build_inductance_matrix`
    but is **complex** and **frequency-dependent** — it is therefore
    rebuilt at every frequency by the solver.

    Each entry is the Carson per-unit-length correction integrated
    over the segment length(s) by the **midpoint rule** in the
    longitudinal direction. For two parallel segments of equal length
    $\\ell$, the midpoint rule recovers the per-unit-length result
    exactly because the Carson correction is uniform along the wire
    (translation invariance of the homogeneous half-space). For
    non-parallel segments we project onto the parallel component of
    the segment-pair geometry — Carson's original derivation only
    covers parallel wires, and any orthogonal component contributes
    zero by symmetry (cf. ADR-0005 §"Decision/Earth-conductivity
    source").

    Earth-conductivity sign convention
    ----------------------------------
    In ``groundfield`` the $z$-axis points *into* the soil. A wire
    above ground has $z < 0$ and ``height = -z > 0``. A wire just
    below the surface has $z > 0$ and we use ``height = z`` as the
    Sunde-equivalent depth (Carson's $h$). For the PEN cable at
    $z = 0.6\\,\\mathrm{m}$ this produces the textbook
    "1 m below surface" Carson result.

    Parameters
    ----------
    seg_endpoints
        Array of shape ``(M, 2, 3)`` — same layout as
        :func:`build_inductance_matrix`.
    wire_radii
        Per-branch wire radii in metres (currently only used as a
        regulariser when two segments would coincide; the Carson
        kernel itself does not depend on the radius — that is in
        the perfect-mirror $\\ln(2h/a)$ piece, which stays where
        it was in :func:`build_inductance_matrix`).
    omega
        Angular frequency in rad/s.
    sigma_earth
        Earth conductivity in S/m.

    Returns
    -------
    dZ : np.ndarray, shape (M, M), dtype complex
        Symmetric Carson correction matrix in $\\Omega$. Entry
        ``dZ[i, j]`` is the *integrated* Carson contribution to
        the $(i, j)$ branch-impedance entry (per-unit-length
        Carson value times the geometric mean of the two segment
        lengths in the parallel-projection sense).
    """
    from groundfield.coupling.carson import carson_mutual_correction

    if omega <= 0.0 or sigma_earth <= 0.0:
        return np.zeros(
            (seg_endpoints.shape[0], seg_endpoints.shape[0]),
            dtype=complex,
        )

    M = seg_endpoints.shape[0]
    dZ = np.zeros((M, M), dtype=complex)

    # Pre-compute mid-points, axis directions, lengths, heights.
    midpoints = 0.5 * (seg_endpoints[:, 0, :] + seg_endpoints[:, 1, :])
    axes = seg_endpoints[:, 1, :] - seg_endpoints[:, 0, :]
    lengths = np.linalg.norm(axes, axis=1)
    # Avoid divide-by-zero — caller is expected to filter degenerate
    # branches, but be defensive.
    safe_lengths = np.where(lengths > 0.0, lengths, 1.0)
    unit_axes = axes / safe_lengths[:, None]
    # Carson "height" h is the absolute value of the depth coordinate
    # (Sunde-equivalent). For a wire at the surface (z = 0) the
    # asymptote is logarithmic, regularised by the wire radius.
    heights = np.abs(midpoints[:, 2])
    heights = np.where(heights > 0.0, heights, np.maximum(wire_radii, 1e-3))

    for i in range(M):
        # Diagonal: self correction (parallel-to-self ⇒ θ = 0).
        a_self = 2.0 * heights[i] * math.sqrt(omega * MU_0 * sigma_earth)
        # Use the mutual function with d = 0 to share one code path
        # (it dispatches to θ = 0 internally). Per-unit-length value
        # times this branch's length.
        dz_per_m = carson_mutual_correction(
            omega=omega,
            height_i=heights[i],
            height_j=heights[i],
            horizontal_distance=0.0,
            sigma_earth=sigma_earth,
        )
        dZ[i, i] = dz_per_m * lengths[i]
        for j in range(i + 1, M):
            # Project segment j onto segment i's axis to get the
            # "parallel" length. Orthogonal components contribute
            # zero per the Neumann projection theorem (Carson's
            # derivation is for parallel filaments).
            dot = float(unit_axes[i] @ unit_axes[j])
            if abs(dot) < 1e-9:
                continue  # purely orthogonal
            # Horizontal distance between the two midpoints, in the
            # plane perpendicular to the soil normal.
            dx = midpoints[i, 0] - midpoints[j, 0]
            dy = midpoints[i, 1] - midpoints[j, 1]
            d_horiz = math.hypot(dx, dy)
            dz_per_m = carson_mutual_correction(
                omega=omega,
                height_i=heights[i],
                height_j=heights[j],
                horizontal_distance=d_horiz,
                sigma_earth=sigma_earth,
            )
            # Effective length: geometric mean projected by the
            # cosine of the segment-pair angle.
            ell = math.sqrt(lengths[i] * lengths[j]) * abs(dot)
            sign = 1.0 if dot > 0.0 else -1.0
            dZ[i, j] = dZ[j, i] = sign * dz_per_m * ell
    return dZ
