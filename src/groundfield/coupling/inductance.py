r"""Self- and mutual-inductance helpers for distributed conductors.

This module implements the inductance physics described in ADR-0004.
It provides three public functions:

- :func:`thin_wire_self_inductance` — closed-form self-inductance of
  a straight thin-wire segment of length $\ell$ and radius $a$
  (Grover 1946):
  $L_\text{self} = (\mu_0\,\ell)/(2\pi)
   \bigl[\ln(2\ell/a) - 1\bigr]$.
- :func:`parallel_segments_mutual` — closed-form mutual inductance
  of two parallel coaxial segments of equal length $\ell$ at
  perpendicular distance $d$.
- :func:`neumann_mutual` — generic Neumann double-line integral via
  two-point Gauss–Legendre quadrature, valid for any 3-D
  segment-pair geometry.

Earth-image contributions are handled by the caller: for a
perfect-mirror earth the caller adds the integral against the
mirror image of the source segment (see
:func:`perfect_mirror_self_pair_inductance` and the assembly logic
in :func:`build_inductance_matrix`).

Conventions
-----------
**Image convention** (``image=``). The earth image of a segment
$(p_1 \to p_2)$ is the segment $(p_1' \to p_2')$ with
$z \to -z$ (the $z$-axis points into the soil). Its Neumann
integral against a partner carries the orientation through the dot
product, so the mirrored-endpoint image is the image in a
**perfect magnetic conductor** (horizontal currents co-directed,
vertical currents reversed). Three conventions are offered:

- ``"none"`` — free space, $L = L_\text{dir}$. The exact
  quasi-static baseline of the Sommerfeld earth-return correction
  (:mod:`groundfield.coupling.sommerfeld_inductance`): at DC the
  soil is magnetically transparent.
- ``"additive"`` — $L = L_\text{dir} + L_\text{img}$, the
  ADR-0004 "perfect mirror" (historic default, ``use_image=True``).
- ``"antiparallel"`` — $L = L_\text{dir} - L_\text{img}$, the
  perfect-electric-conductor image (horizontal currents reversed).
  This is the baseline against which Carson's correction
  $\Delta Z = (\omega\mu_0/\pi)(P + jQ)$ is defined
  (:mod:`groundfield.coupling.carson`). A perfect conductor screens
  the soil: a pair on opposite sides of the surface has $L = 0$ (since
  audit round 2 of the calendar release; the mirrored-endpoint image of
  a buried partner lands above ground and gave $\ln(D_\text{mirror}/D)
  < 0$), and its whole coupling is in the earth-return correction.

An earth-return correction matrix is only meaningful together with
the image convention of the inductance matrix it is added to; the
correction builders therefore take the same ``image=`` argument.

**Internal inductance** (``include_internal=``). The diagonal
(partial self) term is the external thin-wire inductance plus, by
default, the DC internal inductance of a solid round conductor,
$\mu_0 \ell/(8\pi)$. Every function of this package that returns
a self term uses this one convention, controlled by
``include_internal`` (default ``True``). A per-unit-length
reference that includes the internal flux uses the geometric mean
radius $\mathrm{GMR} = a\,e^{-1/4}$; with ``include_internal=False``
the matching reference uses $\mathrm{GMR} = a$. The frequency
dependence of the internal impedance (skin effect) is provided by
:meth:`groundfield.conductors.Conductor.internal_impedance_per_length`
as a correction relative to this DC convention.

**Wires lying side by side.** Two near-parallel segments whose axes
stay closer than the sum of their radii over a common length (two
conductors in one trench, a service drop routed back along its PEN
main, coincident leads) are evaluated as touching wires at the axis
distance $r_i + r_j$ in :func:`build_inductance_matrix`; the filament
integral of such a pair diverges (two coincident 50 m leads read
+587j Ω at 50 Hz, a TN network with drops along its PEN main had a
negative input reactance).

**Close pairs.** Nearly parallel wires close together whose directions
differ by more than the closed form's tolerance, crossing and skew pairs
are integrated semi-analytically with ``touching="exact"``; the 16x16
rule was +12 % at 10 cm and up to +250 % at 1 cm for two 5 m segments
(audit round 2 of the calendar release).

**Touching segments** (``touching=``). The Neumann integrand is
log-singular for segments that share an end point (bends, chain
neighbours) or lie on one axis. ``"exact"`` (the default since the
calendar release 2026.11) uses the closed forms (Grover 1946: two
filaments meeting at a point; coaxial filaments with a gap), accurate
to machine precision; a straight wire split into any number of
segments keeps the self inductance of the unsplit wire.
``"quadrature"`` (the default up to 0.15.0, kept to reproduce old
numbers) integrates such pairs with the 16x16 Gauss–Legendre rule,
which is accurate to about -0.3 % ... +0.8 % for them (review pass
10, finding 10.46). The closed forms of all coaxial pairs of a segment
are evaluated in one vectorised call — a straight conductor of $M$
sub-segments has $M(M-1)/2$ of them — bit-identical with the scalar
routine (audit round 2 of the calendar release, A2C-5).

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
from typing import Literal

import numpy as np

__all__ = [
    "MU_0",
    "ImageConvention",
    "TouchingRule",
    "thin_wire_self_inductance",
    "parallel_segments_mutual",
    "neumann_mutual",
    "perfect_mirror_self_pair_inductance",
    "build_inductance_matrix",
    "_build_inductance_matrix_loop",  # Regression-test reference (ADR-0010)
    "build_carson_correction_matrix",
    "build_carson_correction_matrices",
]

# Vacuum permeability in H/m. The conventional value 4*pi*1e-7 (exact
# in the pre-2019 SI; the CODATA 2018 value 1.25663706212e-6 differs
# by 5.5e-10 relative, which is irrelevant here).
MU_0 = 4.0e-7 * math.pi

#: Image convention of an inductance matrix: ``"none"`` (free space),
#: ``"additive"`` (perfect magnetic mirror, ADR-0004) or
#: ``"antiparallel"`` (perfect electric mirror, Carson's baseline).
ImageConvention = Literal["none", "additive", "antiparallel"]

#: Treatment of touching / coaxial segment pairs in the Neumann
#: integral: ``"exact"`` (closed forms, the default) or
#: ``"quadrature"`` (16x16 Gauss–Legendre, the default up to 0.15.0).
TouchingRule = Literal["quadrature", "exact"]

_IMAGE_CONVENTIONS: tuple[str, ...] = ("none", "additive", "antiparallel")
_TOUCHING_RULES: tuple[str, ...] = ("quadrature", "exact")

# Two end points closer than this fraction of the longer segment are
# treated as one vertex (touching pair). Discretised chains share their
# end points bit for bit, so the tolerance only has to absorb rounding.
_TOUCH_TOL_REL = 1e-9


def _resolve_image(image: str | None, use_image: bool | None) -> str:
    """Resolve the ``image`` / legacy ``use_image`` pair to one convention."""
    if image is not None and use_image is not None:
        raise TypeError("pass either 'image' or the legacy 'use_image', not both")
    if image is None:
        return "none" if use_image is False else "additive"
    if image not in _IMAGE_CONVENTIONS:
        raise ValueError(f"image must be one of {_IMAGE_CONVENTIONS}, got {image!r}")
    return image


def _check_touching_rule(touching: str) -> None:
    if touching not in _TOUCHING_RULES:
        raise ValueError(f"touching must be one of {_TOUCHING_RULES}, got {touching!r}")


# ---------------------------------------------------------------------
# Closed-form helpers (used as analytical references and fast paths)
# ---------------------------------------------------------------------


def thin_wire_self_inductance(
    length: float,
    wire_radius: float,
    *,
    include_internal: bool = True,
) -> float:
    r"""Closed-form self-inductance of a straight thin-wire segment.

    Decomposed into the **external** Grover 1946 thin-wire term and
    the **internal** contribution from the magnetic field inside the
    conductor (uniform DC current distribution, non-magnetic
    material):

    $$
    L_\text{ext} \;=\; \frac{\mu_0\,\ell}{2\pi}
    \Bigl[\ln\!\Bigl(\frac{2\ell}{a}\Bigr) - 1\Bigr],
    \qquad
    L_\text{int} \;=\; \frac{\mu_0\,\ell}{8\pi}.
    $$

    Adding the internal term reproduces the standard Oeding/Oswald
    loop-inductance formula (Gl. 9.13c) when used as the diagonal
    contribution of a two-wire loop. This is the package-wide
    internal-inductance convention (see the module docstring): a
    per-unit-length reference consistent with
    ``include_internal=True`` uses $\mathrm{GMR} = a\,e^{-1/4}$.
    For the frequency-dependent internal impedance (skin effect) see
    :meth:`groundfield.conductors.Conductor.internal_impedance_per_length`.

    Parameters
    ----------
    length
        Segment length $\ell$ in metres.
    wire_radius
        Wire radius $a$ in metres.
    include_internal
        ``True`` (default) — add the internal-field contribution
        $\mu_0 \ell / (8\pi)$ (uniform current, the DC limit).
        ``False`` — the pure external-field thin-wire formula.

    Returns
    -------
    L : float
        Partial self-inductance in henries.
    """
    if length <= 0.0 or wire_radius <= 0.0:
        raise ValueError("length and wire_radius must be positive")
    L_ext = (
        (MU_0 * length) / (2.0 * math.pi) * (math.log(2.0 * length / wire_radius) - 1.0)
    )
    if not include_internal:
        return L_ext
    L_int = (MU_0 * length) / (8.0 * math.pi)
    return L_ext + L_int


def parallel_segments_mutual(length: float, distance: float) -> float:
    r"""Closed-form mutual inductance of two parallel coaxial segments.

    For two segments of equal length $\ell$ that share the same
    axis direction and are placed at perpendicular distance $d$,

    $$
    M_\parallel(\ell, d) \;=\; \frac{\mu_0\,\ell}{2\pi}
    \Bigl[\ln\!\Bigl(\frac{\ell + \sqrt{\ell^2 + d^2}}{d}\Bigr)
          - \frac{\sqrt{\ell^2 + d^2} - d}{\ell}\Bigr].
    $$

    Used both as an internal closed-form fast path (when the segment
    pair geometry matches the assumption) and as a reference in the
    test suite.

    Parameters
    ----------
    length
        Segment length $\ell$ in m (both segments have the same).
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
    return (
        (MU_0 * length)
        / (2.0 * math.pi)
        * (math.log((length + s) / distance) - (s - distance) / length)
    )


def _vertex_integral(l_a: float, l_b: float, cos_eps: float) -> float:
    """$\\int_0^{l_a}\\!\\int_0^{l_b} ds\\,dt/r$ for filaments from one vertex.

    Grover (1946), "two straight filaments meeting at a point": with
    the filaments leaving the common vertex at the angle $\\varepsilon$
    and $R$ the distance between their far ends,

    $$
    I = 2\\bigl[l_a \\operatorname{artanh}\\tfrac{l_b}{l_a + R}
          + l_b \\operatorname{artanh}\\tfrac{l_a}{l_b + R}\\bigr].
    $$

    Diverges for $\\varepsilon \\to 0$ (the filaments fold onto each
    other); the caller only uses it for non-overlapping pairs.
    """
    R = math.sqrt(max(l_a * l_a + l_b * l_b - 2.0 * l_a * l_b * cos_eps, 0.0))
    return 2.0 * (l_a * math.atanh(l_b / (l_a + R)) + l_b * math.atanh(l_a / (l_b + R)))


def _coaxial_gap_integral(l_a: float, l_b: float, gap: float) -> float:
    """$\\int\\!\\int ds\\,dt/r$ for coaxial filaments separated by ``gap``.

    $I = F(l_a + l_b + g) - F(l_a + g) - F(l_b + g) + F(g)$ with
    $F(x) = x\\ln x$ and $F(0) = 0$ (``gap`` $\\ge 0$, touching for 0).
    """

    def F(x: float) -> float:
        return x * math.log(x) if x > 0.0 else 0.0

    return F(l_a + l_b + gap) - F(l_a + gap) - F(l_b + gap) + F(gap)


def _touching_mutual(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
) -> float | None:
    """Exact Neumann mutual of a touching or coaxial segment pair.

    Returns the closed-form partial mutual inductance in H when the two
    straight filaments (i) lie on one axis without overlapping (any
    gap, including zero), or (ii) share an end point at a non-zero
    angle. Returns ``None`` for every other pair (and for overlapping
    coaxial pairs, whose filament integral diverges), so that the
    caller falls back to its general rule.
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
    tol = _TOUCH_TOL_REL * max(la, lb)
    if abs(abs(dot) - 1.0) <= 1e-9:
        rel1 = p1_b - p1_a
        perp = rel1 - float(rel1 @ ua) * ua
        if float(np.linalg.norm(perp)) > tol:
            return None  # parallel, distinct axes: the Grover parallel form
        s1 = float(rel1 @ ua)
        s2 = float((p2_b - p1_a) @ ua)
        lo, hi = min(s1, s2), max(s1, s2)
        if lo >= la - tol:
            gap = max(lo - la, 0.0)
        elif hi <= tol:
            gap = max(-hi, 0.0)
        else:
            return None  # overlapping coaxial filaments: divergent
        sign = 1.0 if dot > 0.0 else -1.0
        return MU_0 / (4.0 * math.pi) * sign * _coaxial_gap_integral(la, lb, gap)
    for vertex_a, away_a in ((p1_a, ua), (p2_a, -ua)):
        for vertex_b, away_b in ((p1_b, ub), (p2_b, -ub)):
            if float(np.linalg.norm(vertex_a - vertex_b)) <= tol:
                cos_eps = float(away_a @ away_b)
                if cos_eps >= 1.0 - 1e-12:
                    return None  # folded onto each other: divergent
                return MU_0 / (4.0 * math.pi) * dot * _vertex_integral(la, lb, cos_eps)
    return None


def _row_dots(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot products of the rows of ``a`` (K, 3) with ``b`` ((K, 3) or (3,)).

    Evaluated as a stack of (1, 3) @ (3, 1) products: NumPy computes each
    of them with the same BLAS ``dot`` routine as the 1-D ``x @ y`` of
    :func:`_touching_mutual` (a chain of fused multiply-adds on x86-64),
    so every entry equals the scalar result bit for bit — which
    ``a @ b`` (BLAS ``gemv``) and an element-wise sum do not.
    """
    rhs = b[:, :, None] if b.ndim == 2 else b[:, None]
    return np.matmul(a[:, None, :], rhs)[:, 0, 0]


def _touching_mutual_batch(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_touching_mutual` of one segment against K partners at once.

    Returns ``(values, valid)``: ``values[k]`` is the closed-form mutual
    inductance in H of segment *a* and partner *k* where ``valid[k]`` is
    true (``_touching_mutual`` would return a number), undefined
    otherwise. Coaxial pairs — on a straight conductor of M sub-segments
    all M(M-1)/2 pairs are coaxial — are evaluated vectorised (review
    pass 10, audit A2C-5: one scalar call per pair made the exact rule
    2-4x slower than the quadrature it replaces); pairs that only share
    an end point (bends, a handful per segment) keep the scalar call.

    Every entry equals the scalar ``_touching_mutual`` bit for bit: the
    same operations in the same order, the 3-element dot products and
    norms through :func:`_row_dots`, the logarithms with :func:`math.log`.
    """
    K = p1_b.shape[0]
    values = np.zeros(K, dtype=float)
    valid = np.zeros(K, dtype=bool)
    if K == 0:
        return values, valid
    da = p2_a - p1_a
    la = float(np.linalg.norm(da))
    if la <= 0.0:
        return values, valid
    db = p2_b - p1_b
    lb = np.sqrt(_row_dots(db, db))
    usable = lb > 0.0
    ua = da / la
    ub = db / np.where(usable, lb, 1.0)[:, None]
    dot = _row_dots(ub, ua)
    tol = _TOUCH_TOL_REL * np.maximum(la, lb)
    coaxial = usable & (np.abs(np.abs(dot) - 1.0) <= 1e-9)

    ks = np.flatnonzero(coaxial)
    if ks.size:
        rel1 = p1_b[ks] - p1_a
        s1 = _row_dots(rel1, ua)
        perp = rel1 - s1[:, None] * ua
        on_axis = np.sqrt(_row_dots(perp, perp)) <= tol[ks]
        s2 = _row_dots(p2_b[ks] - p1_a, ua)
        lo = np.minimum(s1, s2)
        hi = np.maximum(s1, s2)
        after = lo >= la - tol[ks]
        before = ~after & (hi <= tol[ks])
        gap = np.where(after, np.maximum(lo - la, 0.0), np.maximum(-hi, 0.0))
        keep = on_axis & (after | before)  # otherwise overlapping: divergent
        sign = np.where(dot[ks] > 0.0, 1.0, -1.0)
        lb_k = lb[ks]
        f_all, f_a, f_b, f_gap = _x_log_x((la + lb_k) + gap, la + gap, lb_k + gap, gap)
        integral = f_all - f_a - f_b + f_gap
        coax_values = MU_0 / (4.0 * math.pi) * sign * integral
        values[ks[keep]] = coax_values[keep]
        valid[ks[keep]] = True
    # Pairs that only share an end point (bends): the scalar closed form,
    # for the partners that have an end point within ``tol`` of one of
    # segment a's (the scalar routine returns None for all others).
    rest = np.flatnonzero(usable & ~coaxial)
    if rest.size:
        near = np.zeros(rest.size, dtype=bool)
        for vertex_a in (p1_a, p2_a):
            for ends in (p1_b, p2_b):
                gap_vec = vertex_a - ends[rest]
                near |= np.sqrt(_row_dots(gap_vec, gap_vec)) <= tol[rest]
        for k in rest[near]:
            val = _touching_mutual(p1_a, p2_a, p1_b[k], p2_b[k])
            if val is not None:
                values[k] = val
                valid[k] = True
    return values, valid


def _x_log_x(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """$F(x) = x \\ln x$ with $F(0) = 0$, element-wise, for several arrays.

    The logarithm is :func:`math.log`, once per distinct argument, instead
    of :func:`numpy.log`: that keeps the coaxial closed form bit-identical
    with :func:`_coaxial_gap_integral` (NumPy's SIMD logarithm differs
    from the C library in the last bit for some arguments). The
    arguments of a chain of equal sub-segments repeat, so the number of
    logarithms grows with the number of distinct lengths, not of pairs.
    """
    sizes = [a.size for a in arrays]
    x = np.concatenate(arrays)
    out = np.zeros(x.shape, dtype=float)
    pos = x > 0.0
    if pos.any():
        unique, inverse = np.unique(x[pos], return_inverse=True)
        logs = np.fromiter(map(math.log, unique.tolist()), float, unique.size)
        out[pos] = x[pos] * logs[inverse]
    return tuple(np.split(out, np.cumsum(sizes)[:-1]))


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
# Pairs that *touch* (shared end point, coaxial neighbours) have a
# log-singular integrand; there the rule is only accurate to
# -0.3 % ... +0.8 % (review pass 10, R3-12) — use ``touching="exact"``.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _parallel_filaments_mutual(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
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

    M = (
        (MU_0 / (4.0 * math.pi))
        * sigma
        * (F(s_b2, d) - F(s_b2 - la, d) - F(s_b1, d) + F(s_b1 - la, d))
    )
    return M


#: A non-parallel pair whose closest quadrature nodes are nearer than this
#: fraction of the longer segment is integrated semi-analytically with
#: ``touching="exact"``. Measured against an adaptive reference, the 16x16
#: rule is at round-off from half a segment length on (2.6e-7 at 0.2,
#: 1.4e-4 at 0.1, 19 % at 2e-4 for two 5 m segments 1 mm apart and
#: 1e-4 rad inclined; crossing pairs 1-4.5 %, skew pairs 1 cm apart
#: 0.5-2.3 %).
_CLOSE_PAIR_FRACTION = 0.5


def _close_pair_mutual(
    p1_a: np.ndarray, p2_a: np.ndarray, p1_b: np.ndarray, p2_b: np.ndarray
) -> float:
    """Neumann mutual of two close straight filaments, semi-analytic (H).

    The inner integral over segment *b* is exact,
    $\\int_0^{\\ell_b} dt/|P - B(t)| = \\operatorname{asinh}((\\ell_b -
    t_0)/\\rho) + \\operatorname{asinh}(t_0/\\rho)$ with $t_0$ the projection
    of the point $P(s)$ of segment *a* on *b* and $\\rho$ its distance from
    *b*'s line; the outer integral over *a* uses 16-point Gauss–Legendre
    panels graded geometrically, down to the local distance $\\rho$,
    towards the points where $P(s)$ passes an end of *b* or comes
    closest to *b*'s line, and towards the ends of *a*. Accurate to
    round-off for pairs that the 16x16 rule cannot resolve (nearly
    parallel wires close together, crossing and skew pairs); not for
    overlapping coaxial filaments, whose integral diverges.
    """
    da = p2_a - p1_a
    db = p2_b - p1_b
    la = float(np.linalg.norm(da))
    lb = float(np.linalg.norm(db))
    ua = da / la
    ub = db / lb
    dot = float(ua @ ub)
    if dot == 0.0:
        return 0.0
    v0 = p1_a - p1_b
    c0 = float(v0 @ ub)  # t0(s) = c0 + s dot
    floor = 1e-30 * max(la, lb)

    def rho_at(x: float) -> float:
        v = v0 + x * ua
        return max(float(np.linalg.norm(v - (c0 + x * dot) * ub)), floor)

    points = {0.0, la}
    for target in (0.0, lb):
        x = (target - c0) / dot
        if 0.0 < x < la:
            points.add(x)
    den = 1.0 - dot * dot
    if den > 1e-14:
        x = (dot * c0 - float(v0 @ ua)) / den  # closest approach of the lines
        if 0.0 < x < la:
            points.add(x)
    marks = sorted(points)
    edges = [0.0]
    for lo, hi in zip(marks[:-1], marks[1:], strict=True):
        if hi <= lo:
            continue
        mid = 0.5 * (lo + hi)
        left, w = [lo], rho_at(lo)
        while left[-1] + w < mid:
            left.append(left[-1] + w)
            w *= 2.0
        right, w = [hi], rho_at(hi)
        while right[-1] - w > mid:
            right.append(right[-1] - w)
            w *= 2.0
        edges += left[1:] + [mid] + right[::-1]
    e = np.asarray(edges)
    lo = e[:-1, None]
    half = 0.5 * (e[1:, None] - lo)
    x = (lo + half * (_GL_NODES[None, :] + 1.0)).ravel()
    w = (half * _GL_WEIGHTS[None, :]).ravel()
    v = v0[None, :] + x[:, None] * ua[None, :]
    t0 = c0 + x * dot
    rho = np.linalg.norm(v - t0[:, None] * ub[None, :], axis=1)
    rho = np.maximum(rho, floor)
    g = np.arcsinh((lb - t0) / rho) + np.arcsinh(t0 / rho)
    return MU_0 / (4.0 * math.pi) * dot * float(w @ g)


def _needs_close_rule(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    r_min: float,
) -> bool:
    """Whether a quadrature pair gets :func:`_close_pair_mutual`.

    Close (nearest quadrature nodes within :data:`_CLOSE_PAIR_FRACTION`
    of the longer segment), not sharing an end point (closed forms) and
    not coaxial (overlapping coaxial filaments diverge; non-overlapping
    ones have a closed form).
    """
    da = p2_a - p1_a
    db = p2_b - p1_b
    la = float(np.linalg.norm(da))
    lb = float(np.linalg.norm(db))
    longer = max(la, lb)
    if r_min >= _CLOSE_PAIR_FRACTION * longer:
        return False
    tol = _TOUCH_TOL_REL * longer
    for va in (p1_a, p2_a):
        for vb in (p1_b, p2_b):
            if float(np.linalg.norm(va - vb)) <= tol:
                return False
    ua = da / la
    perp = [
        float(np.linalg.norm(q - p1_a - float((q - p1_a) @ ua) * ua))
        for q in (p1_b, p2_b)
    ]
    return max(perp) > tol


def neumann_mutual(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    *,
    min_distance: float = 1e-6,
    parallel_tol: float = 1e-9,
    touching: TouchingRule = "exact",
) -> float:
    """Neumann mutual-inductance integral between two straight segments.

    Hybrid implementation:

    1. If the two segments are **parallel** (or anti-parallel) within
       ``parallel_tol`` on distinct axes, the closed-form Grover
       expression in :func:`_parallel_filaments_mutual` is used —
       exact, no quadrature error. This fast path covers the bulk of
       the typical inductance assembly (parallel PEN strands).
    2. With ``touching="exact"``, pairs that share an end point (chain
       neighbours, bends) or lie on one axis without overlapping use
       their closed forms (Grover 1946; see the module docstring).
    3. Otherwise the Neumann double integral is evaluated by 16×16
       Gauss–Legendre quadrature: exact to round-off for pairs
       separated by more than about half their length, but only
       to -0.3 % ... +0.8 % for touching pairs (review pass 10,
       finding 10.46), and up to +250 % for nearly parallel wires 1 cm
       apart (audit round 2 of the calendar release). With
       ``touching="exact"`` such close pairs — nearest quadrature nodes
       within half the longer segment, no shared end point, not coaxial
       — are integrated semi-analytically instead (inner integral in
       closed form, outer panels graded towards the near-singular
       points; :func:`_close_pair_mutual`).

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
    touching
        ``"exact"`` (default) or ``"quadrature"`` (the 16x16 rule, the
        default up to 0.15.0) — the treatment of touching and coaxial
        pairs.

    Returns
    -------
    M : float
        Partial mutual inductance in H.
    """
    _check_touching_rule(touching)
    p1_a = np.asarray(p1_a, dtype=float)
    p2_a = np.asarray(p2_a, dtype=float)
    p1_b = np.asarray(p1_b, dtype=float)
    p2_b = np.asarray(p2_b, dtype=float)

    if touching == "exact":
        M_touch = _touching_mutual(p1_a, p2_a, p1_b, p2_b)
        if M_touch is not None:
            return M_touch

    # Fast path: parallel filaments with closed form.
    M_par = _parallel_filaments_mutual(
        p1_a,
        p2_a,
        p1_b,
        p2_b,
        parallel_tol=parallel_tol,
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
    if touching == "exact" and _needs_close_rule(
        p1_a, p2_a, p1_b, p2_b, float(r.min())
    ):
        return _close_pair_mutual(p1_a, p2_a, p1_b, p2_b)
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
    p1: np.ndarray,
    p2: np.ndarray,
    wire_radius: float,
    *,
    include_internal: bool = True,
    touching: TouchingRule = "exact",
) -> float:
    """Self-inductance contribution of a segment plus its earth image.

    For a single segment the *partial* self-inductance against itself
    uses the thin-wire formula. The image segment (mirrored at the
    soil surface) is treated as an external segment whose Neumann
    integral with the original is added to the total (the
    ``"additive"`` image convention of ADR-0004).

    Parameters
    ----------
    p1, p2
        Endpoints in metres (with $z > 0$ pointing into the soil).
    wire_radius
        Wire radius in m (only used for the thin-wire self term).
    include_internal
        Add the DC internal inductance $\\mu_0\\ell/(8\\pi)$ (default,
        see :func:`thin_wire_self_inductance`).
    touching
        Rule for the segment–image pair when the segment touches the
        surface with one end point (see :func:`neumann_mutual`).

    Returns
    -------
    L : float
        Self-inductance plus image contribution, in henries.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    L_self = thin_wire_self_inductance(
        float(np.linalg.norm(p2 - p1)),
        wire_radius,
        include_internal=include_internal,
    )
    p1_img = _mirror(p1)
    p2_img = _mirror(p2)
    # Image segment runs in the opposite direction (current image of a
    # vertical filament is anti-parallel; for a horizontal filament it
    # is parallel). The general rule is: image of (p1 → p2) is
    # (p1' → p2'), with the *same* tangent direction up to the
    # sign of ẑ. For Neumann the dot product ǔ_a·ǔ_b takes care of
    # the sign automatically — we just feed the mirrored endpoints.
    L_image = neumann_mutual(p1, p2, p1_img, p2_img, touching=touching)
    return L_self + L_image


def _include_internal_array(include_internal, M: int) -> np.ndarray:
    """Broadcast ``include_internal`` (bool or per-segment array) to (M,)."""
    arr = np.asarray(include_internal, dtype=bool)
    if arr.ndim == 0:
        return np.full(M, bool(arr))
    if arr.shape != (M,):
        raise ValueError(
            f"include_internal must be a bool or have shape ({M},), got {arr.shape}"
        )
    return arr


def _check_image_geometry(
    p1: np.ndarray, p2: np.ndarray, wire_radii: np.ndarray, *, where: str
) -> None:
    """Reject segments whose mirror image is singular (image conventions).

    Two cases make the image term ill-defined:

    - the segment **crosses** the soil surface ($z$ changes sign
      between its end points): its mirror image intersects it at the
      crossing point (a vertical riser even overlaps its image), so
      the image mutual is singular — review pass 10, finding 10.46
      (pass-9 F52). The message says so instead of the historic
      "lies in the soil-surface plane".
    - the segment lies (almost) **in** the plane $z = 0$: it coincides
      with its own image and the partial self-inductance diverges
      logarithmically as $2|z| \\to a$.
    """
    z1 = p1[:, 2]
    z2 = p2[:, 2]
    lengths = np.linalg.norm(p2 - p1, axis=1)
    tol = 1e-9 * np.maximum(lengths, 1e-12)
    crossing = ((z1 < -tol) & (z2 > tol)) | ((z1 > tol) & (z2 < -tol))
    if crossing.any():
        k = int(np.argmax(crossing))
        raise ValueError(
            f"{where}: segment {k} crosses the soil surface (z from "
            f"{z1[k]:.4g} m to {z2[k]:.4g} m). Its mirror image intersects "
            "the segment at the crossing point, so the image term of the "
            "requested image convention is singular. Split the conductor "
            "at z = 0 so that every segment lies on one side of the "
            "surface, or use image='none' (the exact Sommerfeld baseline)."
        )
    # Guard against the degenerate mirror geometry: a segment lying
    # (almost) in the soil-surface plane z = 0 coincides with its own
    # image, and the partial self-inductance diverges logarithmically
    # as the mirror distance 2|z| approaches the wire radius. Without
    # this check the assembly silently produces self-inductances that
    # are orders of magnitude too large (observed: ~4.7 H instead of
    # ~55 µH for a 25 m segment). Conductors strung between electrode
    # heads at z = 0 are the typical trap.
    z_mid = 0.5 * np.abs(z1 + z2)
    degenerate = 2.0 * z_mid <= wire_radii
    if degenerate.any():
        k = int(np.argmax(degenerate))
        raise ValueError(
            f"{where}: segment {k} lies in the "
            f"soil-surface plane (|z_mid| = {z_mid[k]:.4g} m <= "
            f"wire_radius/2 = {0.5 * wire_radii[k]:.4g} m). Its "
            "perfect-mirror image coincides with the segment and "
            "the partial self-inductance diverges. Bury the "
            "conductor (e.g. z = 0.7 m) or disable the image "
            "term with image='none'."
        )


def _build_inductance_matrix_loop(
    seg_endpoints: np.ndarray,
    wire_radii: np.ndarray,
    *,
    use_image: bool | None = None,
    image: ImageConvention | None = None,
    include_internal: bool | np.ndarray = True,
    touching: TouchingRule = "exact",
) -> np.ndarray:
    """Reference (loop-based) implementation of :func:`build_inductance_matrix`.

    Kept from the pre-vectorisation code path. Used by the regression
    tests as the oracle the vectorised assembly must reproduce
    bit-exactly (for every ``image`` / ``include_internal`` /
    ``touching`` combination). New callers should use
    :func:`build_inductance_matrix`, which dispatches to the
    vectorised path described in ADR-0010.
    """
    mode = _resolve_image(image, use_image)
    _check_touching_rule(touching)
    M = seg_endpoints.shape[0]
    internal = _include_internal_array(include_internal, M)
    L = np.zeros((M, M), dtype=float)
    sign = -1.0 if mode == "antiparallel" else 1.0
    opposite = _opposite_sides(
        np.asarray(seg_endpoints[:, 0, :], dtype=float),
        np.asarray(seg_endpoints[:, 1, :], dtype=float),
    )
    for i in range(M):
        p1_i = seg_endpoints[i, 0]
        p2_i = seg_endpoints[i, 1]
        # Diagonal — self plus image
        if mode == "additive":
            L[i, i] = perfect_mirror_self_pair_inductance(
                p1_i,
                p2_i,
                wire_radii[i],
                include_internal=bool(internal[i]),
                touching=touching,
            )
        elif mode == "antiparallel":
            p1f = np.asarray(p1_i, dtype=float)
            p2f = np.asarray(p2_i, dtype=float)
            L_self = thin_wire_self_inductance(
                float(np.linalg.norm(p2f - p1f)),
                wire_radii[i],
                include_internal=bool(internal[i]),
            )
            L_img = neumann_mutual(
                p1f, p2f, _mirror(p1f), _mirror(p2f), touching=touching
            )
            L[i, i] = L_self - L_img
        else:
            li = float(np.linalg.norm(p2_i - p1_i))
            L[i, i] = thin_wire_self_inductance(
                li, wire_radii[i], include_internal=bool(internal[i])
            )
        # Off-diagonal — Neumann + image of partner
        for j in range(i + 1, M):
            p1_j = seg_endpoints[j, 0]
            p2_j = seg_endpoints[j, 1]
            if mode == "antiparallel" and opposite[i, j]:
                L[i, j] = L[j, i] = 0.0  # screened by the PEC (A2C-1)
                continue
            m = neumann_mutual(p1_i, p2_i, p1_j, p2_j, touching=touching)
            m = _touching_wires(m, p1_i, p2_i, wire_radii[i], p1_j, p2_j, wire_radii[j])
            if mode != "none":
                p1_j_img = _mirror(p1_j)
                p2_j_img = _mirror(p2_j)
                m_img = neumann_mutual(
                    p1_i, p2_i, p1_j_img, p2_j_img, touching=touching
                )
                m_img = _touching_wires(
                    m_img, p1_i, p2_i, wire_radii[i], p1_j_img, p2_j_img, wire_radii[j]
                )
                m = m + sign * m_img
            L[i, j] = L[j, i] = m
    return L


def _touching_wires(
    m: float,
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    r_a: float,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    r_b: float,
) -> float:
    """Scalar form of :func:`_interpenetrating_batch` for the loop oracle."""
    vals, inside = _interpenetrating_batch(
        np.asarray(p1_a, dtype=float),
        np.asarray(p2_a, dtype=float),
        float(r_a),
        np.asarray(p1_b, dtype=float)[None, :],
        np.asarray(p2_b, dtype=float)[None, :],
        np.array([float(r_b)]),
    )
    return float(vals[0]) if inside[0] else m


# ---------------------------------------------------------------------
# ADR-0010 (Tier 0b): vectorised partial-inductance assembly
# ---------------------------------------------------------------------


def _parallel_filaments_mutual_batch(
    p1_a: np.ndarray,
    ua_a: np.ndarray,
    la_a: float,
    p1_b: np.ndarray,
    ub_b: np.ndarray,
    lb_b: np.ndarray,
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
    rel1 = p1_b - p1_a[None, :]  # (K, 3)
    rel2 = (p1_b + ub_b * lb_b[:, None]) - p1_a[None, :]  # (K, 3) — this is p2_b - p1_a
    # Project onto ua_a; the "perpendicular component" is rel - (rel·ua)·ua.
    s1 = rel1 @ ua_a  # (K,)
    s2 = rel2 @ ua_a  # (K,)
    perp1 = rel1 - s1[:, None] * ua_a[None, :]  # (K, 3)
    perp2 = rel2 - s2[:, None] * ua_a[None, :]  # (K, 3)
    delta_perp = np.linalg.norm(perp2 - perp1, axis=1)  # (K,)
    skew_mask = delta_perp > parallel_tol
    parallel_mask = near_parallel & ~skew_mask

    if not parallel_mask.any():
        return M_out, valid

    # Distance d (constant along the parallel pair). When d == 0 the
    # closed form diverges and the caller must use the quadrature.
    d = np.linalg.norm(perp1, axis=1)  # (K,)
    coaxial = d == 0.0
    parallel_mask &= ~coaxial
    if not parallel_mask.any():
        return M_out, valid

    # For anti-parallel b we swap (s1, s2) so that beta_1 < beta_2
    # holds for both orientations.
    sigma = np.where(dots > 0.0, 1.0, -1.0)  # (K,)
    s_b1 = np.where(sigma > 0.0, s1, s2)  # (K,)
    s_b2 = np.where(sigma > 0.0, s2, s1)  # (K,)

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
    M_par = (
        (MU_0 / (4.0 * math.pi)) * sigma * (F_b2 - F_b2_minus_la - F_b1 + F_b1_minus_la)
    )
    M_out[parallel_mask] = M_par[parallel_mask]
    valid[parallel_mask] = True
    return M_out, valid


def _neumann_quadrature_batch(
    p1_a: np.ndarray,
    da_a: np.ndarray,
    la_a: float,
    ua_a: np.ndarray,
    p1_b: np.ndarray,
    db_b: np.ndarray,
    lb_b: np.ndarray,
    ub_b: np.ndarray,
    dots: np.ndarray,
    *,
    min_distance: float = 1e-6,
    r_min_out: np.ndarray | None = None,
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

    s_nodes = 0.5 * (_GL_NODES + 1.0)  # (16,)
    w_nodes = 0.5 * _GL_WEIGHTS  # (16,)

    # Quadrature points along a and along each b
    pts_a = p1_a[None, :] + s_nodes[:, None] * da_a[None, :]  # (16, 3)
    pts_b = p1_b[:, None, :] + s_nodes[None, :, None] * db_b[:, None, :]  # (K, 16, 3)

    # diff[k, i, j, :] = pts_a[i, :] - pts_b[k, j, :]
    diff = pts_a[None, :, None, :] - pts_b[:, None, :, :]  # (K, 16, 16, 3)
    r = np.linalg.norm(diff, axis=-1)  # (K, 16, 16)
    if r_min_out is not None:
        r_min_out[:] = r.min(axis=(1, 2))
    np.maximum(r, min_distance, out=r)
    inv_r = 1.0 / r
    # integral[k] = sum_{i,j} w_i * w_j * inv_r[k, i, j]
    integrals = np.einsum("i,j,kij->k", w_nodes, w_nodes, inv_r)  # (K,)

    M = (MU_0 / (4.0 * math.pi)) * dots * la_a * lb_b * integrals
    # Pairs with dot == 0 contribute nothing (handled by ``dots`` factor).
    return M


def _offdiag_row(
    p1_a: np.ndarray,
    ua_a: np.ndarray,
    la_a: float,
    da_a: np.ndarray,
    p1_b: np.ndarray,
    ub_b: np.ndarray,
    lb_b: np.ndarray,
    db_b: np.ndarray,
    close_rule: bool = False,
) -> np.ndarray:
    """Neumann mutuals of one segment against K partners (vectorised).

    Closed form for parallel pairs on distinct axes, 16x16
    Gauss–Legendre for the rest — exactly the historic per-row
    assembly of :func:`build_inductance_matrix`. With ``close_rule``
    (``touching="exact"``), close non-parallel pairs are integrated
    semi-analytically (:func:`_close_pair_mutual`).
    """
    dots = ub_b @ ua_a  # (K,)
    M_par, par_mask = _parallel_filaments_mutual_batch(
        p1_a,
        ua_a,
        la_a,
        p1_b,
        ub_b,
        lb_b,
        dots,
    )
    M_row = np.where(par_mask, M_par, 0.0)
    npar_mask = ~par_mask
    if npar_mask.any():
        ks = np.where(npar_mask)[0]
        r_min = np.empty(ks.size) if close_rule else None
        M_row[ks] = _neumann_quadrature_batch(
            p1_a,
            da_a,
            la_a,
            ua_a,
            p1_b[ks],
            db_b[ks],
            lb_b[ks],
            ub_b[ks],
            dots[ks],
            r_min_out=r_min,
        )
        if close_rule:
            near = r_min < _CLOSE_PAIR_FRACTION * np.maximum(la_a, lb_b[ks])
            p2_a = p1_a + da_a
            for n in np.flatnonzero(near & (dots[ks] != 0.0)):
                k = ks[n]
                p2_b = p1_b[k] + db_b[k]
                if _needs_close_rule(p1_a, p2_a, p1_b[k], p2_b, float(r_min[n])):
                    M_row[k] = _close_pair_mutual(p1_a, p2_a, p1_b[k], p2_b)
    return M_row


def _apply_touching_exact(
    M_row: np.ndarray,
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    la_a: float,
    lb_b: np.ndarray,
    ua_a: np.ndarray,
    ub_b: np.ndarray,
) -> None:
    """Overwrite touching / coaxial pairs of one row by their closed forms.

    The candidates (a shared end point, or a coaxial partner) are
    evaluated in one vectorised call (:func:`_touching_mutual_batch`,
    bit-identical with the scalar closed forms; audit A2C-5).
    """
    if p1_b.shape[0] == 0:
        return
    tol = _TOUCH_TOL_REL * np.maximum(lb_b, la_a)
    d_end = np.minimum(
        np.minimum(
            np.linalg.norm(p1_b - p1_a, axis=1), np.linalg.norm(p2_b - p1_a, axis=1)
        ),
        np.minimum(
            np.linalg.norm(p1_b - p2_a, axis=1), np.linalg.norm(p2_b - p2_a, axis=1)
        ),
    )
    # Coaxial candidates: (anti-)parallel with a vanishing perpendicular
    # offset of the start point of b from a's axis.
    dots = ub_b @ ua_a
    rel = p1_b - p1_a[None, :]
    perp = rel - (rel @ ua_a)[:, None] * ua_a[None, :]
    coax = (np.abs(np.abs(dots) - 1.0) <= 1e-9) & (np.linalg.norm(perp, axis=1) <= tol)
    candidates = np.flatnonzero((d_end <= tol) | coax)
    if candidates.size == 0:
        return
    values, valid = _touching_mutual_batch(
        p1_a, p2_a, p1_b[candidates], p2_b[candidates]
    )
    M_row[candidates[valid]] = values[valid]


#: ``|cos phi|`` above which two segments count as near-parallel (8.1 deg):
#: the interpenetration guard below and the Carson tiling share it.
_PARALLEL_COS_MIN = 0.99


def _interpenetrating_batch(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    r_a: float,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    r_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual inductance of interpenetrating parallel wires (vectorised).

    Two near-parallel segments (``|cos phi| > 0.99``) whose projections
    overlap and whose axes stay closer than the sum of their radii
    $r_a + r_b$ over the overlap are wires lying side by side (two
    conductors in one trench, a service drop routed back along its PEN
    main, two leads laid on top of each other). Their filament integral
    grows like $\\ln(1/d)$ as the axis distance $d \\to 0$ and diverges
    for coincident or overlapping coaxial filaments, where the 16x16
    rule only stops at its $10^{-6}$ m floor: two coincident 50 m leads
    read +587j Ω at 50 Hz, and a TN network whose drops ran back along
    the PEN main had a negative input reactance (a non-positive-definite
    inductance matrix). Such pairs are evaluated as parallel filaments
    at the axis distance $r_a + r_b$ (touching wires) over the projected
    overlap — the closed form of :func:`parallel_segments_mutual` with
    offsets — which keeps the matrix positive definite.

    Returns ``(values, mask)`` over the K partners *b*: ``values[k]`` in
    H where ``mask[k]``.
    """
    K = p1_b.shape[0]
    values = np.zeros(K, dtype=float)
    mask = np.zeros(K, dtype=bool)
    if K == 0:
        return values, mask
    da = p2_a - p1_a
    la = float(np.linalg.norm(da))
    db = p2_b - p1_b
    lb = np.linalg.norm(db, axis=1)
    if la <= 0.0:
        return values, mask
    ua = da / la
    dots = (db @ ua) / np.where(lb > 0.0, lb, 1.0)
    rel1 = p1_b - p1_a
    rel2 = p2_b - p1_a
    s1 = rel1 @ ua
    s2 = rel2 @ ua
    perp1 = np.linalg.norm(rel1 - s1[:, None] * ua, axis=1)
    perp2 = np.linalg.norm(rel2 - s2[:, None] * ua, axis=1)
    lo = np.minimum(s1, s2)
    hi = np.maximum(s1, s2)
    overlap = np.minimum(hi, la) - np.maximum(lo, 0.0)
    d_min = r_a + np.asarray(r_b, dtype=float)
    mask = (
        (lb > 0.0)
        & (np.abs(dots) > _PARALLEL_COS_MIN)
        & (overlap > _TOUCH_TOL_REL * np.maximum(la, lb))
        & (np.maximum(perp1, perp2) < (1.0 - 1e-9) * d_min)
    )
    if not mask.any():
        return values, mask
    d = d_min[mask]
    b1, b2 = lo[mask], hi[mask]

    def F(x: np.ndarray) -> np.ndarray:
        return x * np.arcsinh(x / d) - np.hypot(x, d)

    sign = np.where(dots[mask] > 0.0, 1.0, -1.0)
    values[mask] = (
        MU_0 / (4.0 * math.pi) * sign * (F(b2) - F(b2 - la) - F(b1) + F(b1 - la))
    )
    return values, mask


def _overhead(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Side of the soil surface of every segment: ``True`` above it.

    A segment lies on one side (image conventions reject segments that
    cross the surface); its mid-point decides — a segment that only
    touches the surface with one end belongs to the side of the other.
    """
    return 0.5 * (p1[:, 2] + p2[:, 2]) < 0.0


def _opposite_sides(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """``(M, M)`` mask of segment pairs on opposite sides of the surface."""
    above = _overhead(p1, p2)
    return above[:, None] != above[None, :]


def _inductance_blocks(
    p1: np.ndarray,
    p2: np.ndarray,
    wire_radii: np.ndarray | None,
    *,
    want_direct: bool,
    want_image: bool,
    self_from_rows: bool,
    include_internal: np.ndarray,
    touching: str,
    diagonal: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Direct and image blocks of the partial-inductance matrix.

    Returns ``(L_dir, L_img)`` with ``L_dir`` the free-space partial
    inductances (thin-wire self terms on the diagonal) and ``L_img``
    the Neumann integrals against the mirrored partners (the
    mirrored-endpoint, perfect-magnetic-conductor image; its own image
    on the diagonal). Every image convention is a signed sum of the
    two, so one code path serves all of them and the image block that
    :mod:`groundfield.coupling.sommerfeld_inductance` removes is the
    very block that :func:`build_inductance_matrix` added.

    ``self_from_rows`` selects the historic expression of the segment
    length in the self term (``np.linalg.norm`` along the rows when
    ``True``, per segment otherwise); the two differ in the last bit
    for some inputs, and keeping both is what makes every legacy call
    of :func:`build_inductance_matrix` bit-identical.

    With ``wire_radii``, interpenetrating parallel pairs (axes closer
    than the sum of the radii over a common length, direct or against
    a mirrored partner) are evaluated as touching wires
    (:func:`_interpenetrating_batch`); without radii the filament
    integral is kept. ``diagonal=False`` leaves the diagonal of the
    direct block at zero (off-diagonal mutuals only, no radii needed).
    """
    M = p1.shape[0]
    radii = None if wire_radii is None else np.asarray(wire_radii, dtype=float)
    da = p2 - p1  # (M, 3)
    la = np.linalg.norm(da, axis=1)  # (M,)
    if (la <= 0.0).any():
        raise ValueError("segment lengths must be positive")
    ua = da / la[:, None]  # (M, 3)

    L_dir = np.zeros((M, M), dtype=float) if want_direct else None
    L_img = np.zeros((M, M), dtype=float) if want_image else None

    if want_image:
        p1_img = p1.copy()
        p1_img[:, 2] = -p1_img[:, 2]
        p2_img = p2.copy()
        p2_img[:, 2] = -p2_img[:, 2]
        da_img = p2_img - p1_img
        la_img = np.linalg.norm(da_img, axis=1)
        ua_img = da_img / la_img[:, None]

    # Diagonal — cheap (M operations); scalar helpers.
    for i in range(M):
        if want_direct and diagonal:
            if self_from_rows:
                length_i = la[i]
            else:
                length_i = float(np.linalg.norm(p2[i] - p1[i]))
            L_dir[i, i] = thin_wire_self_inductance(
                length_i, radii[i], include_internal=bool(include_internal[i])
            )
        if want_image:
            L_img[i, i] = neumann_mutual(
                p1[i], p2[i], _mirror(p1[i]), _mirror(p2[i]), touching=touching
            )

    # Off-diagonal — for each row i, L[i, i+1:M] as a batch.
    for i in range(M - 1):
        slc = slice(i + 1, M)
        if want_direct:
            M_pri = _offdiag_row(
                p1[i],
                ua[i],
                float(la[i]),
                da[i],
                p1[slc],
                ua[slc],
                la[slc],
                da[slc],
                close_rule=touching == "exact",
            )
            if touching == "exact":
                _apply_touching_exact(
                    M_pri,
                    p1[i],
                    p2[i],
                    p1[slc],
                    p2[slc],
                    float(la[i]),
                    la[slc],
                    ua[i],
                    ua[slc],
                )
            if radii is not None:
                vals, inside = _interpenetrating_batch(
                    p1[i], p2[i], radii[i], p1[slc], p2[slc], radii[slc]
                )
                M_pri[inside] = vals[inside]
            L_dir[i, slc] = M_pri
            L_dir[slc, i] = M_pri
        if want_image:
            M_img = _offdiag_row(
                p1[i],
                ua[i],
                float(la[i]),
                da[i],
                p1_img[slc],
                ua_img[slc],
                la_img[slc],
                p2_img[slc] - p1_img[slc],
                close_rule=touching == "exact",
            )
            if touching == "exact":
                _apply_touching_exact(
                    M_img,
                    p1[i],
                    p2[i],
                    p1_img[slc],
                    p2_img[slc],
                    float(la[i]),
                    la_img[slc],
                    ua[i],
                    ua_img[slc],
                )
            if radii is not None:
                vals, inside = _interpenetrating_batch(
                    p1[i], p2[i], radii[i], p1_img[slc], p2_img[slc], radii[slc]
                )
                M_img[inside] = vals[inside]
            L_img[i, slc] = M_img
            L_img[slc, i] = M_img
    return L_dir, L_img


def build_inductance_matrix(
    seg_endpoints: np.ndarray,  # shape (M, 2, 3): each branch's [start, end]
    wire_radii: np.ndarray,  # shape (M,)
    *,
    use_image: bool | None = None,
    image: ImageConvention | None = None,
    include_internal: bool | np.ndarray = True,
    touching: TouchingRule = "exact",
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
        (in m, $z$ positive into the soil) of every
        distributed-conductor longitudinal-branch segment.
    wire_radii
        Per-branch wire radii in metres, shape ``(M,)``.
    use_image
        Legacy switch: ``True`` is ``image="additive"``, ``False`` is
        ``image="none"``. Pass either this or ``image``, not both.
    image
        Image convention of the returned matrix (see the module
        docstring): ``"none"`` (free space), ``"additive"`` (perfect
        magnetic mirror, the historic default) or ``"antiparallel"``
        (perfect electric mirror — Carson's baseline). The image term
        is the Neumann integral against the mirrored partner (and the
        segment's own image on the diagonal). Defaults to
        ``"additive"``.
    include_internal
        Add the DC internal inductance $\\mu_0\\ell/(8\\pi)$ to the
        self terms (default ``True``). A boolean applies to every
        segment; an array of shape ``(M,)`` selects it per segment
        (e.g. ``False`` for conductors whose internal impedance is
        supplied separately by a skin-effect model).
    touching
        ``"exact"`` (default): closed forms for pairs that share an end
        point or lie on one axis; ``"quadrature"``: the 16x16 rule of
        0.15.0 (see :func:`neumann_mutual`).

    Returns
    -------
    L : np.ndarray, shape (M, M)
        Symmetric partial-inductance matrix in H. Entry ``L[i, j]``
        is the partial mutual inductance between branch *i* and
        branch *j* (and the diagonal is the self-inductance + image).

    Raises
    ------
    ValueError
        With an image convention other than ``"none"``: a segment
        that crosses the soil surface (its image intersects it) or
        lies in the surface plane (its image coincides with it). A
        segment of zero length.

    Notes
    -----
    The legacy loop-based implementation is kept as
    :func:`_build_inductance_matrix_loop` for regression testing.
    See :file:`tests/test_inductance_vectorised.py` for the
    bit-exact regression suite.
    """
    mode = _resolve_image(image, use_image)
    _check_touching_rule(touching)
    M = seg_endpoints.shape[0]
    if M == 0:
        return np.zeros((0, 0), dtype=float)

    p1 = seg_endpoints[:, 0, :].astype(float, copy=False)  # (M, 3)
    p2 = seg_endpoints[:, 1, :].astype(float, copy=False)  # (M, 3)
    radii = np.asarray(wire_radii, dtype=float)
    internal = _include_internal_array(include_internal, M)
    if mode != "none":
        _check_image_geometry(p1, p2, radii, where="build_inductance_matrix")
    L_dir, L_img = _inductance_blocks(
        p1,
        p2,
        radii,
        want_direct=True,
        want_image=mode != "none",
        self_from_rows=mode == "none",
        include_internal=internal,
        touching=touching,
    )
    if mode == "none":
        return L_dir
    if mode == "additive":
        return L_dir + L_img
    L = L_dir - L_img
    # A perfect conductor screens the soil: a pair on opposite sides of
    # the surface has no anti-parallel baseline (audit A2C-1).
    L[_opposite_sides(p1, p2)] = 0.0
    return L


def _image_inductance_matrix(
    seg_endpoints: np.ndarray,
    *,
    touching: TouchingRule = "exact",
    wire_radii: np.ndarray | None = None,
) -> np.ndarray:
    """Image block $L_\\text{img}$ (mirrored-endpoint image), shape (M, M).

    The block that :func:`build_inductance_matrix` adds for
    ``image="additive"`` and subtracts for ``image="antiparallel"`` —
    computed by the same code path, so that an earth-return correction
    can move between conventions exactly (with ``wire_radii``, a mirrored
    partner that interpenetrates a segment is evaluated as a touching
    wire, as in :func:`build_inductance_matrix`). Raises for segments
    that cross or lie in the soil surface (singular image).
    """
    _check_touching_rule(touching)
    M = seg_endpoints.shape[0]
    if M == 0:
        return np.zeros((0, 0), dtype=float)
    p1 = seg_endpoints[:, 0, :].astype(float, copy=False)
    p2 = seg_endpoints[:, 1, :].astype(float, copy=False)
    # The in-plane check needs a radius; a zero radius keeps only the
    # crossing check and the exactly-in-plane case.
    _check_image_geometry(p1, p2, np.zeros(M), where="earth-return correction")
    _, L_img = _inductance_blocks(
        p1,
        p2,
        wire_radii,
        want_direct=False,
        want_image=True,
        self_from_rows=False,
        include_internal=np.ones(M, dtype=bool),
        touching=touching,
    )
    return L_img


# ---------------------------------------------------------------------
# ADR-0005: Carson earth-return correction matrix
# ---------------------------------------------------------------------


def _carson_pair_plan(
    seg_endpoints: np.ndarray, wire_radii: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, int, float, float, float]],
    list[tuple[int, int]],
]:
    """Frequency-independent geometry of the Carson tiling.

    Carson's correction is a per-metre result for long parallel
    horizontal wires. It is tiled over the segment pairs inside that
    validity domain — near-parallel (``|cos phi| > 0.99``) pairs of
    near-horizontal segments (inclination below 8.1 deg) — and every
    other non-orthogonal pair, and the self term of every inclined or
    vertical segment, is handed to the exact Sommerfeld kernel (audit
    round 2 of the calendar release, A2C-1; up to then oblique pairs got
    ``sqrt(l_i l_j) |cos phi|`` times the per-metre value, once per
    segment pair, which over-counts long oblique conductors by roughly
    their number of segments: a probe lead turning 8.1 deg against the
    feed lead changed the measured reactance 17.6-fold).

    Returns
    -------
    heights
        Carson height ``|z|`` of every segment mid-point in m
        (regularised for a segment in the surface plane).
    lengths
        Segment lengths in m.
    horizontal
        Whether a segment is near-horizontal (Carson self term).
    entries
        Tiled pairs ``(i, j, horizontal_distance, factor, sign)``: a
        mutual correction ``sign * dz'(d) * factor`` with the per-metre
        value at the horizontal axis distance ``d`` and the axial overlap
        ``factor`` (m).
    fallback
        Pairs ``(i, j)``, ``i <= j``, for the Sommerfeld kernel (``i == j``:
        the self term of a non-horizontal segment).
    """
    M = seg_endpoints.shape[0]
    radii = np.asarray(wire_radii, dtype=float)
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
    heights = np.where(heights > 0.0, heights, np.maximum(radii, 1e-3))
    horizontal = np.hypot(unit_axes[:, 0], unit_axes[:, 1]) > _PARALLEL_COS_MIN

    entries: list[tuple[int, int, float, float, float]] = []
    fallback: list[tuple[int, int]] = [
        (i, i) for i in range(M) if lengths[i] > 0.0 and not horizontal[i]
    ]
    for i in range(M):
        for j in range(i + 1, M):
            dot = float(unit_axes[i] @ unit_axes[j])
            if abs(dot) < 1e-9:
                continue  # purely orthogonal — zero by symmetry
            if not (horizontal[i] and horizontal[j] and abs(dot) > _PARALLEL_COS_MIN):
                # Oblique, inclined or vertical: outside Carson's domain.
                fallback.append((i, j))
                continue
            sign = 1.0 if dot > 0.0 else -1.0
            d_vec = midpoints[j] - midpoints[i]
            # Near-parallel: transmission-line tiling (WP-C3).
            u = unit_axes[i]
            axial = float(d_vec @ u)
            perp = d_vec - axial * u
            d_perp_horiz = math.hypot(perp[0], perp[1])
            d_perp_vert = abs(perp[2])
            same_axis = (
                d_perp_horiz < max(radii[i], radii[j], 1e-3) and d_perp_vert < 1e-6
            )
            # Axial overlap of the two segment intervals along u.
            t_i0 = float((seg_endpoints[i, 0] - midpoints[i]) @ u)
            t_i1 = float((seg_endpoints[i, 1] - midpoints[i]) @ u)
            lo_i, hi_i = min(t_i0, t_i1), max(t_i0, t_i1)
            t_j0 = float((seg_endpoints[j, 0] - midpoints[i]) @ u)
            t_j1 = float((seg_endpoints[j, 1] - midpoints[i]) @ u)
            lo_j, hi_j = min(t_j0, t_j1), max(t_j0, t_j1)
            overlap = min(hi_i, hi_j) - max(lo_i, lo_j)
            if same_axis:
                # Collinear pieces of one wire never overlap (their earth
                # term is carried by the per-length diagonals); two
                # conductors that do overlap on one axis are wires lying
                # side by side: the mutual of touching wires (finding of
                # stream DOC-B / GEN-2; up to then such pairs got none).
                if overlap <= _TOUCH_TOL_REL * max(lengths[i], lengths[j]):
                    continue
                d_perp_horiz = max(d_perp_horiz, float(radii[i] + radii[j]))
            elif overlap <= 0.0:
                continue  # staggered/end-to-end: no local tiling
            entries.append((i, j, d_perp_horiz, overlap, sign))
    return heights, lengths, horizontal, entries, fallback


def _direct_mutual_matrix(
    seg_endpoints: np.ndarray, wire_radii: np.ndarray | None = None
) -> np.ndarray:
    """Off-diagonal free-space Neumann mutuals $L_\\text{dir}$ (zero diagonal).

    The direct block of :func:`build_inductance_matrix` (same code path,
    exact touching pairs, interpenetrating pairs as touching wires when
    ``wire_radii`` is given), used to move a correction of a pair on
    opposite sides of the surface between image conventions.
    """
    M = seg_endpoints.shape[0]
    p1 = seg_endpoints[:, 0, :].astype(float, copy=False)
    p2 = seg_endpoints[:, 1, :].astype(float, copy=False)
    L_dir, _ = _inductance_blocks(
        p1,
        p2,
        wire_radii,
        want_direct=True,
        want_image=False,
        self_from_rows=False,
        include_internal=np.ones(M, dtype=bool),
        touching="exact",
        diagonal=False,
    )
    return L_dir


def _carson_fallback_values(
    seg_endpoints: np.ndarray,
    wire_radii: np.ndarray,
    fallback: list[tuple[int, int]],
    omegas: list[float],
    sigma_earth: float,
) -> dict[tuple[int, int], np.ndarray]:
    """Sommerfeld corrections of the pairs outside Carson's domain.

    Exact quasi-static correction of a homogeneous earth of conductivity
    ``sigma_earth`` (:mod:`groundfield.coupling.sommerfeld_inductance`,
    image-free, tabulated assembly over the segments that take part:
    one $S(\\rho)$ table per pair of depth groups — 1.2 s instead of
    64 s for two 1 km leads of 50 segments at an angle, and closer to a
    converged reference than the per-pair rule), moved onto the
    anti-parallel baseline of the Carson matrix: $\\Delta Z =
    \\Delta Z_\\text{none} + j\\omega (L_\\text{none} -
    L_\\text{antiparallel})$, i.e. plus the image mutual for a pair on
    one side of the surface and plus the direct mutual for a pair on
    opposite sides (whose anti-parallel baseline is zero). Returns
    ``(n_f,)`` complex values in Ω per pair.
    """
    from groundfield.coupling.sommerfeld_inductance import (
        LayeredEarth,
        _tabulated_values_none,
    )

    p1 = seg_endpoints[:, 0, :].astype(float, copy=False)
    p2 = seg_endpoints[:, 1, :].astype(float, copy=False)
    _check_image_geometry(
        p1, p2, np.zeros(p1.shape[0]), where="earth-return correction"
    )
    L_img = _image_inductance_matrix(seg_endpoints, wire_radii=wire_radii)
    opposite = _opposite_sides(p1, p2)
    L_dir = (
        _direct_mutual_matrix(seg_endpoints, wire_radii)
        if any(opposite[i, j] for i, j in fallback)
        else None
    )
    earth = LayeredEarth(rhos=(1.0 / sigma_earth,), thicknesses=())
    members = sorted({k for pair in fallback for k in pair})
    where = {k: n for n, k in enumerate(members)}
    table = _tabulated_values_none(
        np.asarray(seg_endpoints, dtype=float)[members], omegas, earth
    )
    om = np.asarray(omegas, dtype=float)
    out: dict[tuple[int, int], np.ndarray] = {}
    for i, j in fallback:
        vals = table[:, where[i], where[j]]
        shift = L_dir[i, j] if opposite[i, j] else L_img[i, j]
        out[(i, j)] = vals + np.where(om > 0.0, 1j * om * shift, 0.0)
    return out


def build_carson_correction_matrices(
    seg_endpoints: np.ndarray,  # shape (M, 2, 3)
    wire_radii: np.ndarray,  # shape (M,)
    *,
    omegas,
    sigma_earth: float,
    image: ImageConvention = "antiparallel",
) -> np.ndarray:
    """Carson earth-return correction matrices for several frequencies.

    Multi-frequency form of :func:`build_carson_correction_matrix`: the
    frequency-independent tiling geometry (pair classification, axial
    overlaps, Carson heights) is built once and every frequency only
    evaluates the per-metre Carson values. Entry ``k`` equals
    ``build_carson_correction_matrix(..., omega=omegas[k])`` bit for
    bit.

    Parameters
    ----------
    seg_endpoints
        Array of shape ``(M, 2, 3)`` (m), as in
        :func:`build_inductance_matrix`.
    wire_radii
        Per-branch wire radii in m, shape ``(M,)``.
    omegas
        Angular frequencies in rad/s (sequence of floats); entries
        ``<= 0`` give zero matrices.
    sigma_earth
        Earth conductivity in S/m.
    image
        Image convention of the inductance matrix this correction is
        added to (see :func:`build_carson_correction_matrix`).

    Returns
    -------
    dZ : np.ndarray, shape (n_f, M, M), dtype complex
        Correction matrices in Ω.
    """
    mode = _resolve_image(image, None)
    omegas = [float(om) for om in np.atleast_1d(np.asarray(omegas, dtype=float))]
    from groundfield.coupling.carson import (
        carson_mixed_mutual_correction,
        carson_mutual_correction,
    )

    M = seg_endpoints.shape[0]
    out = np.zeros((len(omegas), M, M), dtype=complex)
    if M == 0 or sigma_earth <= 0.0 or all(om <= 0.0 for om in omegas):
        return out
    radii = np.asarray(wire_radii, dtype=float)
    heights, lengths, horizontal, entries, fallback = _carson_pair_plan(
        seg_endpoints, radii
    )
    p1 = seg_endpoints[:, 0, :].astype(float, copy=False)
    p2 = seg_endpoints[:, 1, :].astype(float, copy=False)
    above = _overhead(p1, p2)
    opposite = above[:, None] != above[None, :]
    fallback_values = (
        _carson_fallback_values(seg_endpoints, radii, fallback, omegas, sigma_earth)
        if fallback
        else {}
    )
    M_conv = None
    if mode != "antiparallel":
        # Carson's correction is defined against the anti-parallel
        # image: L_anti = L_dir - L_img on one side of the surface, 0
        # across it. Moving it onto an inductance matrix with another
        # image convention, dZ -> dZ - scale jw M_conv:
        #   additive: L_add - L_anti = 2 L_img  |  L_dir + L_img (across)
        #   none:     L_dir - L_anti =   L_img  |  L_dir         (across)
        M_conv = _image_inductance_matrix(seg_endpoints, wire_radii=radii)
        if opposite.any():
            L_dir = _direct_mutual_matrix(seg_endpoints, radii)
            if mode == "additive":
                M_conv[opposite] = 0.5 * (L_dir[opposite] + M_conv[opposite])
            else:
                M_conv[opposite] = L_dir[opposite]
    for k, omega in enumerate(omegas):
        if omega <= 0.0:
            continue
        dZ = out[k]
        for i in range(M):
            if not horizontal[i]:
                continue  # inclined / vertical: Sommerfeld (fallback)
            # Diagonal: self correction (parallel-to-self ⇒ θ = 0),
            # per-unit-length value times this branch's length.
            dz_per_m = carson_mutual_correction(
                omega=omega,
                height_i=heights[i],
                height_j=heights[i],
                horizontal_distance=0.0,
                sigma_earth=sigma_earth,
            )
            dZ[i, i] = dz_per_m * lengths[i]
        for i, j, d_h, factor, sign in entries:
            if opposite[i, j]:
                # Overhead wire parallel to a buried one: the field
                # transmitted into the soil (no PEC baseline across the
                # surface; audit A2C-1).
                h_air, d_soil = (
                    (heights[i], heights[j]) if above[i] else (heights[j], heights[i])
                )
                dz_per_m = carson_mixed_mutual_correction(
                    omega, h_air, d_soil, d_h, sigma_earth
                )
            else:
                dz_per_m = carson_mutual_correction(
                    omega=omega,
                    height_i=heights[i],
                    height_j=heights[j],
                    horizontal_distance=d_h,
                    sigma_earth=sigma_earth,
                )
            dZ[i, j] = dZ[j, i] = sign * dz_per_m * factor
        for (i, j), vals in fallback_values.items():
            dZ[i, j] = dZ[j, i] = vals[k]
        if M_conv is not None:
            scale = 2.0 if mode == "additive" else 1.0
            dZ -= scale * 1j * omega * M_conv
    return out


def build_carson_correction_matrix(
    seg_endpoints: np.ndarray,  # shape (M, 2, 3)
    wire_radii: np.ndarray,  # shape (M,)
    *,
    omega: float,
    sigma_earth: float,
    image: ImageConvention = "antiparallel",
) -> np.ndarray:
    """Assemble the dense Carson earth-return correction matrix.

    Returns the per-segment-pair impedance correction
    $\\Delta Z^{(i,j)}_\\text{Carson}(\\omega)$ described in ADR-0005.
    The output has the same shape as :func:`build_inductance_matrix`
    but is **complex** and **frequency-dependent** — it is therefore
    rebuilt at every frequency by the solver (or built for all
    frequencies at once with :func:`build_carson_correction_matrices`).

    Image baseline (review pass 10, findings 10.20 / F32)
    -----------------------------------------------------
    Carson's correction $(\\omega\\mu_0/\\pi)(P + jQ)$ is defined
    against the **anti-parallel** (perfect-electric-conductor) image:
    the complete series impedance is
    $Z = j\\omega L_\\text{antiparallel} + \\Delta Z_\\text{Carson}$
    with $L_\\text{antiparallel}$ from
    ``build_inductance_matrix(image="antiparallel")``. The ``image``
    argument names the image convention of the inductance matrix the
    returned correction is **added to**; for ``"additive"`` /
    ``"none"`` the matrix is shifted by $-2j\\omega L_\\text{img}$ /
    $-j\\omega L_\\text{img}$ so that the sum is the same physical
    impedance. The default ``"antiparallel"`` returns the plain Carson
    correction (bit-identical to earlier releases). The docstring of
    earlier releases said the correction goes "on top of the
    perfect-mirror (additive) Neumann matrix" — that pairing
    double-counts the image and gives 1.66–2.25× the earth-return
    reactance on 100 m – 1 km lines.

    Tiling convention (audit 2026-07-08, WP-C3)
    -------------------------------------------
    Carson's correction is an *infinite-wire, per-unit-length*
    concept. To make the assembled matrix consistent — and, crucially,
    **convergent under mesh refinement** — the per-pair entries tile
    the per-unit-length values without double counting:

    - **Diagonal** ``dZ[i, i]``: self correction per metre × segment
      length. Summed over a refined straight wire this reproduces the
      full ``z'_self · ℓ`` exactly, independent of the segment count.
    - **Collinear pairs** (same axis, negligible perpendicular
      offset): **zero.** The infinite-wire self value on the diagonal
      already contains the interaction with the rest of the same
      wire.
    - **Near-parallel pairs on distinct axes** (``|cosφ| > 0.99``) of
      near-horizontal segments: mutual correction per metre at the
      **perpendicular** horizontal axis separation, weighted by the
      **axial overlap length** of the two segment intervals.
      Non-overlapping (e.g. staggered or end-to-end) pairs contribute
      zero. Two different conductors that overlap on one axis are
      wires lying side by side: the mutual at the distance of touching
      wires $r_i + r_j$ (they used to get none).
    - **Pairs on opposite sides of the surface** (an overhead wire
      parallel to a buried one): the field transmitted into the soil,
      :func:`~groundfield.coupling.carson.carson_mixed_mutual_correction`
      — the anti-parallel baseline of such a pair is zero (audit round 2
      of the calendar release, A2C-1; the buried wire used to be treated
      as an overhead wire at height $|z|$ on the mirrored-endpoint image:
      -36 % mutual reactance for a lead 1 m above a PEN at 0.7 m, 1 kHz).
    - **Outside Carson's domain** — oblique pairs
      (``1e-9 < |cosφ| ≤ 0.99``) and every pair and self term of a
      segment inclined by more than 8.1 deg: the exact quasi-static
      Sommerfeld correction of a homogeneous earth of the same
      conductivity (tabulated assembly), moved onto the anti-parallel
      baseline. Up to audit round 2 oblique pairs got the projected
      geometric-mean heuristic $\\sqrt{\\ell_i\\ell_j}\\,|\\cos\\varphi|$ per
      segment pair, which grows with the number of segments (a probe lead
      turning 8.1 deg against the feed lead changed the reading
      17.6-fold). At the domain boundary the result steps from the
      per-metre tiling to the Sommerfeld value, by the model error of
      the tiling there (two 1 km leads 2 m apart, 1 kHz: 12 %).
      Orthogonal pairs contribute zero by symmetry.

    Earth-conductivity sign convention
    ----------------------------------
    In ``groundfield`` the $z$-axis points *into* the soil. A wire
    above ground has $z < 0$ and ``height = -z > 0``. A wire just
    below the surface has $z > 0$ and we use ``height = z`` as the
    Sunde-equivalent depth (Carson's $h$).

    Parameters
    ----------
    seg_endpoints
        Array of shape ``(M, 2, 3)`` in m — same layout as
        :func:`build_inductance_matrix`.
    wire_radii
        Per-branch wire radii in metres (only used as a regulariser
        for a wire exactly at the surface; the Carson kernel itself
        does not depend on the radius).
    omega
        Angular frequency in rad/s.
    sigma_earth
        Earth conductivity in S/m.
    image
        Image convention of the inductance matrix the correction is
        added to: ``"antiparallel"`` (default, Carson's own baseline),
        ``"additive"`` or ``"none"``.

    Returns
    -------
    dZ : np.ndarray, shape (M, M), dtype complex
        Symmetric Carson correction matrix in $\\Omega$.
    """
    if omega <= 0.0 or sigma_earth <= 0.0:
        return np.zeros(
            (seg_endpoints.shape[0], seg_endpoints.shape[0]),
            dtype=complex,
        )
    return build_carson_correction_matrices(
        seg_endpoints,
        wire_radii,
        omegas=[omega],
        sigma_earth=sigma_earth,
        image=image,
    )[0]
