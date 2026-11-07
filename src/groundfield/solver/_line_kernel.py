"""Exact line-source kernel of the image family (private), ``kernel="line"``.

Review pass 10, 10.28. The point kernel places each segment's current
at its midpoint; its row sums carry an error of about ``-0.23/L``
independent of the segment length ``L``, so the grounding resistance
converges to a value 0.9–1.9 % below the exact uniform-current
solution. The line kernel treats every segment as a uniform line
current.

Kernel
------
The potential of a uniform line current :math:`I` on a segment of
length :math:`L` (end points :math:`A`, :math:`B`) at a point
:math:`X` is

.. math::
    \\varphi(X) = \\frac{\\rho I}{4\\pi L}
    \\ln\\frac{R_1 + R_2 + L}{R_1 + R_2 - L},
    \\qquad R_1 = |X - A|,\\ R_2 = |X - B|,

valid for any orientation; images use the mirrored end points. The
reaction matrix is its **Galerkin** form, the segment average over the
observing segment,

.. math::
    K_{ij} = \\frac{1}{L_i L_j}\\int_{i}\\int_{j}
    \\frac{\\mathrm{d}l\\,\\mathrm{d}l'}{\\sqrt{|x - y|^2 + a^2}},

which matches the solver's length-weighted row reduction: for a
straight wire the sum over segment pairs is the exact double integral
of the whole wire, at any segment length. The reduced kernel
(:math:`a` = wire radius) regularises the self term,

.. math::
    K_{ii} = \\frac{2}{L}\\Bigl[\\operatorname{arsinh}\\frac{L}{a}
    - \\sqrt{1 + (a/L)^2} + \\frac{a}{L}\\Bigr].

Evaluation
----------
Exact Galerkin integrals are needed only for pairs closer than
:data:`NEAR_FACTOR` times the longer segment: parallel pairs have a
closed form, other pairs use a Gauss–Legendre rule on the observing
segment, graded towards the point of closest approach, with the exact
inner line potential. Farther pairs use the point kernel plus the
leading (quadrupole) term of the segment average,

.. math::
    \\frac{1}{R} + \\frac{L_i^2 (3c_i^2 - 1) + L_j^2 (3c_j^2 - 1)}
    {24 R^3},\\qquad c_k = \\frac{(P_i - P_j)\\cdot t_k}{R},

whose remainder is of order :math:`(L/R)^4`. Field points (after the
solve) see the exact line potential near a segment and the point
kernel plus :math:`L_j^2 (3c_j^2 - 1)/(24R^3)` farther away. The
Tagg/Sunde images of a two-layer soil (``n >= 1``) are at least
:math:`2(h_1 - z)` away; they use the point form, with the exact
Galerkin value for the rare image pairs closer than
:data:`NEAR_FACTOR` segment lengths (wires close to the interface).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = ["NEAR_FACTOR", "SELF_TERM_CROSSOVER", "LineSegments"]

#: Pairs closer than ``NEAR_FACTOR · max(L_i, L_j)`` (midpoint distance)
#: are integrated exactly; farther pairs use the quadrupole-corrected
#: point form.
NEAR_FACTOR: float = 6.0

#: Tolerance of ``|t_i · t_j|`` for the closed-form parallel integral.
_PARALLEL_TOL = 1e-10


@dataclass(frozen=True)
class LineSegments:
    """Straight segments as uniform line sources.

    Attributes
    ----------
    mid
        Midpoints in m, shape ``(N, 3)``.
    length
        Lengths in m, shape ``(N,)``.
    tangent
        Unit tangents, shape ``(N, 3)``.
    radius
        Wire radii in m, shape ``(N,)`` (regularisation of the kernel).
    """

    mid: np.ndarray
    length: np.ndarray
    tangent: np.ndarray
    radius: np.ndarray

    @property
    def start(self) -> np.ndarray:
        """Start points in m, shape ``(N, 3)``."""
        return self.mid - 0.5 * self.length[:, None] * self.tangent

    def subset(self, mask: np.ndarray) -> LineSegments:
        """The segments selected by a boolean or index array."""
        return LineSegments(
            self.mid[mask], self.length[mask], self.tangent[mask], self.radius[mask]
        )

    def image(self, offset: float = 0.0, sign: float = -1.0) -> LineSegments:
        """Image segments at depth ``offset + sign · z``.

        ``offset = 0, sign = -1`` is the mirror image in the soil
        surface; the Tagg/Sunde images use ``offset = ±2 n h_1``.
        """
        mid = self.mid.copy()
        mid[:, 2] = offset + sign * mid[:, 2]
        tangent = self.tangent.copy()
        tangent[:, 2] *= sign
        return LineSegments(mid, self.length, tangent, self.radius)


# ---------------------------------------------------------------------
# Line potential
# ---------------------------------------------------------------------


def line_potential_pairs(
    X: np.ndarray, A: np.ndarray, t: np.ndarray, L: np.ndarray, a2: np.ndarray
) -> np.ndarray:
    """Reduced line potential ``(1/L) ∫_0^L dl / sqrt(|X - A - l t|² + a²)``.

    Evaluated pairwise and without cancellation: with ``s`` the
    projection of ``X - A`` on ``t`` and ``ρ² = |X - A|² - s² + a²``,

    * ``0 <= s <= L``: ``arsinh((L - s)/ρ) + arsinh(s/ρ)``;
    * ``s < 0``: ``log1p(L (1 + (L - 2s)/(R_1 + R_2)) / (R_1 - s))``;
    * ``s > L``: the mirror image of the previous case.

    Parameters
    ----------
    X
        Observation points in m, shape ``(P, 3)`` (or broadcastable).
    A, t, L
        Segment start points (m), unit tangents and lengths (m).
    a2
        Squared regularisation radius in m², shape broadcastable to
        ``(P,)``.

    Returns
    -------
    numpy.ndarray
        Line potential per unit current density in 1/m, shape ``(P,)``.
    """
    W = X - A
    s = np.einsum("...k,...k->...", W, t)
    w2 = np.einsum("...k,...k->...", W, W)
    L = np.broadcast_to(L, s.shape)
    rho2 = np.maximum(w2 - s * s, 0.0) + np.broadcast_to(a2, s.shape)
    u = L - s
    R1 = np.sqrt(s * s + rho2)
    R2 = np.sqrt(u * u + rho2)
    out = np.empty(s.shape)
    inside = (s >= 0.0) & (u >= 0.0)
    before = s < 0.0
    after = u < 0.0
    rho = np.sqrt(rho2[inside])
    out[inside] = np.arcsinh(u[inside] / rho) + np.arcsinh(s[inside] / rho)
    if before.any():
        Lb, sb, R1b, R2b = L[before], s[before], R1[before], R2[before]
        out[before] = np.log1p(Lb * (1.0 + (Lb - 2.0 * sb) / (R1b + R2b)) / (R1b - sb))
    if after.any():
        La, ua, R1a, R2a = L[after], u[after], R1[after], R2[after]
        out[after] = np.log1p(La * (1.0 + (La - 2.0 * ua) / (R1a + R2a)) / (R2a - ua))
    return out / L


# ---------------------------------------------------------------------
# Galerkin pair integrals
# ---------------------------------------------------------------------


def galerkin_self(L: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Self term ``(2/L)[arsinh(L/a) - sqrt(1 + (a/L)²) + a/L]`` in 1/m."""
    return 2.0 * (np.arcsinh(L / a) - np.sqrt(1.0 + (a / L) ** 2) + a / L) / L


#: Ratio :math:`L/a` at which the collocation self term :math:`2\ln(L/a)/L`
#: and the Galerkin self term :func:`galerkin_self` coincide (2.990).
SELF_TERM_CROSSOVER: float = 2.990091615374695


def point_self_term(L: np.ndarray, a: np.ndarray) -> np.ndarray:
    r"""Self term of the point kernel in 1/m.

    The point kernel uses the collocation form :math:`2\ln(L/a)/L`, the
    long-segment limit of the segment's potential on its own surface.
    It collapses for short segments (zero at :math:`L = a`, so a
    sliver at a layer interface acts as a short: ``mom`` gave a rod whose
    tip is 9 mm below :math:`h_1` 27 % less resistance than the same rod
    1 mm above it, review pass 10, 10.32). Below
    :data:`SELF_TERM_CROSSOVER` the Galerkin self term
    :func:`galerkin_self`, valid for :math:`L \sim a`, takes over:

    .. math::
        K_{ii} = \max\Bigl(\frac{2\ln(L/a)}{L},\;
        \frac{2}{L}\Bigl[\operatorname{arsinh}\frac{L}{a}
        - \sqrt{1 + (a/L)^2} + \frac{a}{L}\Bigr]\Bigr).

    The two terms are equal at :math:`L/a = 2.990`, so the switch is
    continuous, and every segment with :math:`L/a \ge 2.990` keeps the
    collocation value bit for bit.

    Parameters
    ----------
    L
        Segment lengths in m.
    a
        Wire radii in m.

    Returns
    -------
    numpy.ndarray
        Self terms in 1/m (the potential is :math:`\rho/(4\pi)` times
        the term times the segment current).
    """
    collocation = 2.0 * np.log(L / a) / L
    short = L < SELF_TERM_CROSSOVER * a
    if not np.any(short):
        return collocation
    return np.where(short, np.maximum(collocation, galerkin_self(L, a)), collocation)


def _g2(u: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Second antiderivative of ``1/sqrt(u² + d²)``."""
    return u * np.arcsinh(u / d) - np.sqrt(u * u + d * d)


def _parallel_pairs(Ai, ti, Li, Aj, tj, Lj, a2) -> np.ndarray:
    """Closed-form Galerkin integral of (anti-)parallel segment pairs."""
    w = Aj - Ai
    ya = np.einsum("pk,pk->p", w, ti)
    yb = ya + Lj * np.einsum("pk,pk->p", tj, ti)
    y1 = np.minimum(ya, yb)
    y2 = np.maximum(ya, yb)
    d = np.sqrt(np.maximum(np.einsum("pk,pk->p", w, w) - ya * ya, 0.0) + a2)
    val = _g2(Li - y1, d) - _g2(-y1, d) - _g2(Li - y2, d) + _g2(-y2, d)
    return val / (Li * Lj)


def _graded_rule(n_nodes: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre rule on ``[0, 1]`` graded towards ``0``."""
    x, w = leggauss(n_nodes)
    breaks = np.array([0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 1.0])
    nodes, weights = [], []
    for lo, hi in zip(breaks[:-1], breaks[1:], strict=True):
        nodes.append(lo + (hi - lo) * 0.5 * (x + 1.0))
        weights.append((hi - lo) * 0.5 * w)
    return np.concatenate(nodes), np.concatenate(weights)


_GRADED_NODES, _GRADED_WEIGHTS = _graded_rule()
_PLAIN_NODES = 0.5 * (leggauss(8)[0] + 1.0)  # plain 8-point rule on [0, 1]
_PLAIN_WEIGHTS = 0.5 * leggauss(8)[1]

#: A skew pair whose segments stay at least this many observer lengths
#: apart is integrated with one plain 8-point Gauss–Legendre panel per
#: side instead of the graded rule: the integrand's nearest singularity
#: is then at least one panel length away, and the rule is accurate to
#: about 1e-10 relative (checked against the graded rule).
_PLAIN_RULE_SEPARATION = 1.0


def _closest_parameter(Ai, di, Aj, dj) -> np.ndarray:
    """Parameter ``s ∈ [0, 1]`` on ``Ai + s di`` closest to segment ``j``."""
    r = Ai - Aj
    a = np.einsum("pk,pk->p", di, di)
    e = np.einsum("pk,pk->p", dj, dj)
    f = np.einsum("pk,pk->p", dj, r)
    c = np.einsum("pk,pk->p", di, r)
    b = np.einsum("pk,pk->p", di, dj)
    denom = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(
            denom > 1e-14 * a * e, np.clip((b * f - c * e) / denom, 0.0, 1.0), 0.0
        )
        t = (b * s + f) / e
        s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), s)
        s = np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s)
    return s


def _skew_pairs(Ai, ti, Li, Aj, tj, Lj, a2) -> np.ndarray:
    """Galerkin integral of non-parallel pairs by graded quadrature.

    The outer integral over segment ``i`` is split at the point of
    closest approach to segment ``j`` and each part is integrated with
    a Gauss–Legendre rule graded towards that point (touching and
    crossing wires have a peak of width ``a`` there); the inner
    integral is the exact line potential.
    """
    s_star = _closest_parameter(Ai, Li[:, None] * ti, Aj, Lj[:, None] * tj)
    x_star = s_star * Li
    # Separation at the closest approach: distance of the closest point
    # of segment i from segment j. Well separated pairs (most near
    # pairs) need no grading.
    X_star = Ai + x_star[:, None] * ti
    W = X_star - Aj
    along = np.clip(np.einsum("pk,pk->p", W, tj), 0.0, Lj)
    gap = W - along[:, None] * tj
    plain = np.einsum("pk,pk->p", gap, gap) >= (_PLAIN_RULE_SEPARATION * Li) ** 2
    total = np.zeros(Ai.shape[0])
    for mask, nodes, weights in (
        (~plain, _GRADED_NODES, _GRADED_WEIGHTS),
        (plain, _PLAIN_NODES, _PLAIN_WEIGHTS),
    ):
        if not mask.any():
            continue
        xs, Lm = x_star[mask], Li[mask]
        A_i, t_i = Ai[mask], ti[mask]
        A_j, t_j, L_j, a2_j = Aj[mask], tj[mask], Lj[mask], a2[mask]
        part = np.zeros(xs.shape[0])
        for side, extent in ((-1.0, xs), (+1.0, Lm - xs)):
            # nodes x = x* + side·extent·u, u in [0, 1] (graded towards 0)
            x = xs[:, None] + side * extent[:, None] * nodes[None, :]
            X = A_i[:, None, :] + x[:, :, None] * t_i[:, None, :]
            phi = line_potential_pairs(
                X, A_j[:, None, :], t_j[:, None, :], L_j[:, None], a2_j[:, None]
            )
            part += extent * (phi @ weights)
        total[mask] = part
    return total / Li


def galerkin_pairs(
    obs: LineSegments, i: np.ndarray, src: LineSegments, j: np.ndarray
) -> np.ndarray:
    """Galerkin integral ``K_ij`` of segment pairs in 1/m.

    Parameters
    ----------
    obs, src
        Observing and source segments (``src`` may be an image set).
    i, j
        Pair indices into ``obs`` and ``src``.

    Returns
    -------
    numpy.ndarray
        ``(1/(L_i L_j)) ∬ dl dl' / sqrt(|x - y|² + a²)`` per pair, with
        ``a² = (a_i² + a_j²)/2``.
    """
    Ai, ti, Li = obs.start[i], obs.tangent[i], obs.length[i]
    Aj, tj, Lj = src.start[j], src.tangent[j], src.length[j]
    a2 = 0.5 * (obs.radius[i] ** 2 + src.radius[j] ** 2)
    out = np.empty(i.shape[0])
    par = np.abs(np.einsum("pk,pk->p", ti, tj)) > 1.0 - _PARALLEL_TOL
    if par.any():
        out[par] = _parallel_pairs(
            Ai[par], ti[par], Li[par], Aj[par], tj[par], Lj[par], a2[par]
        )
    skew = ~par
    if skew.any():
        out[skew] = _skew_pairs(
            Ai[skew], ti[skew], Li[skew], Aj[skew], tj[skew], Lj[skew], a2[skew]
        )
    return out


# ---------------------------------------------------------------------
# Pair search and far-field correction
# ---------------------------------------------------------------------


def near_pairs(
    obs_mid: np.ndarray,
    obs_len: np.ndarray,
    src_mid: np.ndarray,
    src_len: np.ndarray,
    factor: float = NEAR_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs with ``|P_i - Q_j| < factor · max(L_i, L_j)``.

    ``obs_len`` may be zero (field points). The distance is computed as
    in :func:`quadrupole_block`, so a pair is either *near* here or
    corrected there, never both.

    The search uses each segment's own radius: the pairs within
    ``factor · L_j`` of every source ``j`` and those within
    ``factor · L_i`` of every observer ``i``. Up to the audit of the
    2026.11 work it searched every observer with the radius of the
    longest segment of the whole set, so one conductor discretised at
    50 m in a 16-house world turned a 0.8 s line solve into 29 s and
    1.7 GiB (audit round 1, A1-numerics F02). The pair set is the same,
    so the results are too.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Observer and source indices of the near pairs, sorted by
        observer, then source.
    """
    from scipy.spatial import cKDTree

    empty = np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    n_obs, n_src = obs_mid.shape[0], src_mid.shape[0]
    if n_obs == 0 or n_src == 0:
        return empty
    pad = 1.0 + 1e-9
    found_i: list[np.ndarray] = []
    found_j: list[np.ndarray] = []

    def _collect(hits, query_index: np.ndarray, *, query_is_source: bool) -> None:
        counts = np.fromiter((len(h) for h in hits), dtype=int, count=len(hits))
        if counts.sum() == 0:
            return
        q = np.repeat(query_index, counts)
        other = np.concatenate([np.asarray(h, dtype=int) for h in hits if len(h)])
        found_i.append(other if query_is_source else q)
        found_j.append(q if query_is_source else other)

    # Pairs near a source: the observers within factor · L_j of source j.
    _collect(
        cKDTree(obs_mid).query_ball_point(src_mid, r=factor * src_len * pad + 1e-12),
        np.arange(n_src),
        query_is_source=True,
    )
    # Pairs near an observer segment: the sources within factor · L_i.
    with_length = np.flatnonzero(obs_len > 0.0)
    if with_length.size:
        _collect(
            cKDTree(src_mid).query_ball_point(
                obs_mid[with_length], r=factor * obs_len[with_length] * pad + 1e-12
            ),
            with_length,
            query_is_source=False,
        )
    if not found_i:
        return empty
    key = np.unique(np.concatenate(found_i) * n_src + np.concatenate(found_j))
    i, j = key // n_src, key % n_src
    d = obs_mid[i] - src_mid[j]
    r2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]
    lim = factor * np.maximum(obs_len[i], src_len[j])
    keep = r2 < lim * lim
    return i[keep], j[keep]


def quadrupole_block(
    obs_mid: np.ndarray,
    obs_t: np.ndarray | None,
    obs_len: np.ndarray,
    src: LineSegments,
    factor: float = NEAR_FACTOR,
) -> np.ndarray:
    """Leading correction of the segment average to the point kernel.

    ``[L_i²(3c_i² - 1) + L_j²(3c_j² - 1)] / (24 R³)`` for every pair with
    ``R >= factor · max(L_i, L_j)``, zero for nearer pairs (they are
    integrated exactly). For field points pass ``obs_t=None`` and zero
    lengths.

    Returns
    -------
    numpy.ndarray
        Correction in 1/m, shape ``(len(obs_mid), len(src.mid))``.
    """
    dx = np.subtract.outer(obs_mid[:, 0], src.mid[:, 0])
    dy = np.subtract.outer(obs_mid[:, 1], src.mid[:, 1])
    dz = np.subtract.outer(obs_mid[:, 2], src.mid[:, 2])
    r2 = (dx * dx + dy * dy) + dz * dz
    lim = factor * np.maximum.outer(obs_len, src.length)
    far = r2 >= lim * lim
    tj = src.tangent
    uj = dx * tj[None, :, 0] + dy * tj[None, :, 1] + dz * tj[None, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (src.length[None, :] ** 2) * (3.0 * uj * uj / r2 - 1.0)
        if obs_t is not None:
            ui = dx * obs_t[:, 0:1] + dy * obs_t[:, 1:2] + dz * obs_t[:, 2:3]
            q += (obs_len[:, None] ** 2) * (3.0 * ui * ui / r2 - 1.0)
        q /= 24.0 * r2 * np.sqrt(r2)
    return np.where(far, q, 0.0)


# ---------------------------------------------------------------------
# Sparse corrections to the point kernel
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SparseCorrection:
    """Additive corrections ``K[rows, cols] += values``, sorted by row."""

    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray

    @classmethod
    def concatenate(cls, parts: list[SparseCorrection]) -> SparseCorrection:
        """Join several corrections and sort them by row."""
        rows = np.concatenate([p.rows for p in parts]) if parts else np.zeros(0, int)
        cols = np.concatenate([p.cols for p in parts]) if parts else np.zeros(0, int)
        vals = np.concatenate([p.values for p in parts]) if parts else np.zeros(0)
        order = np.argsort(rows, kind="stable")
        return cls(rows[order], cols[order], vals[order])

    def add_to_rows(self, block: np.ndarray, i0: int, scale: float = 1.0) -> None:
        """Add the corrections of rows ``i0 … i0 + len(block)`` to ``block``."""
        lo, hi = np.searchsorted(self.rows, [i0, i0 + block.shape[0]])
        if hi > lo:
            np.add.at(
                block,
                (self.rows[lo:hi] - i0, self.cols[lo:hi]),
                scale * self.values[lo:hi],
            )


def _point_values(obs_mid, src_mid, i, j, min_distance) -> np.ndarray:
    """``1/max(r, d)`` of index pairs, with the point kernel's arithmetic."""
    d = obs_mid[i] - src_mid[j]
    r = np.sqrt((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2])
    return 1.0 / np.maximum(r, min_distance)


def reaction_corrections(segs: LineSegments, min_distance: float) -> SparseCorrection:
    """Exact minus point value of the half-space reaction matrix.

    Covers the self terms (Galerkin self term instead of
    ``2 ln(L/a)/L``) and every direct or surface-image pair nearer than
    :data:`NEAR_FACTOR` segment lengths (Galerkin integral instead of
    ``1/r``; for the self image ``1/(2z)``).
    """
    n = segs.length.shape[0]
    parts = []
    diag = np.arange(n)
    parts.append(
        SparseCorrection(
            diag,
            diag,
            galerkin_self(segs.length, segs.radius)
            - point_self_term(segs.length, segs.radius),
        )
    )
    i, j = near_pairs(segs.mid, segs.length, segs.mid, segs.length)
    off = i != j
    i, j = i[off], j[off]
    if i.size:
        exact = galerkin_pairs(segs, i, segs, j)
        parts.append(
            SparseCorrection(
                i, j, exact - _point_values(segs.mid, segs.mid, i, j, min_distance)
            )
        )
    img = segs.image()
    i, j = near_pairs(segs.mid, segs.length, img.mid, img.length)
    if i.size:
        exact = galerkin_pairs(segs, i, img, j)
        point = _point_values(segs.mid, img.mid, i, j, min_distance)
        self_image = i == j
        point[self_image] = 1.0 / (2.0 * np.abs(segs.mid[i[self_image], 2]))
        parts.append(SparseCorrection(i, j, exact - point))
    return SparseCorrection.concatenate(parts)


#: Pair count from which the optional numba kernel evaluates the dense
#: far-field term (below it the numpy path is as fast). Both paths give
#: bit-identical blocks; ``groundfield.solver._kernels.USE_NUMBA``
#: switches numba off for the line kernel as well.
LINE_NUMBA_MIN_PAIRS: int = 200_000


def _numba_quadrupole_rows():
    """The compiled kernel, or ``None`` (numba missing or switched off)."""
    from groundfield.solver import _kernels

    if not _kernels.USE_NUMBA:
        return None
    try:
        from groundfield.solver._line_numba import quadrupole_rows
    except ImportError:  # numba is an optional extra
        return None
    return quadrupole_rows


def _quadrupole_with_image(
    obs_mid: np.ndarray,
    obs_t: np.ndarray | None,
    obs_len: np.ndarray,
    segs: LineSegments,
) -> np.ndarray:
    """``quadrupole_block`` of the segments plus that of their surface images."""
    img = segs.image()
    kernel = None
    if obs_mid.shape[0] * segs.mid.shape[0] >= LINE_NUMBA_MIN_PAIRS:
        kernel = _numba_quadrupole_rows()
    if kernel is None:
        return quadrupole_block(obs_mid, obs_t, obs_len, segs) + quadrupole_block(
            obs_mid, obs_t, obs_len, img
        )
    out = np.empty((obs_mid.shape[0], segs.mid.shape[0]))
    t = obs_t if obs_t is not None else np.zeros_like(obs_mid)
    c = np.ascontiguousarray
    kernel(
        c(obs_mid[:, 0]), c(obs_mid[:, 1]), c(obs_mid[:, 2]),
        c(t[:, 0]), c(t[:, 1]), c(t[:, 2]), c(obs_len, dtype=float), obs_t is not None,
        c(segs.mid[:, 0]), c(segs.mid[:, 1]), c(segs.mid[:, 2]), c(img.mid[:, 2]),
        c(segs.tangent[:, 0]), c(segs.tangent[:, 1]), c(segs.tangent[:, 2]),
        c(img.tangent[:, 2]), c(segs.length, dtype=float), float(NEAR_FACTOR), out,
    )  # fmt: skip
    return out


def reaction_quadrupole_rows(segs: LineSegments, i0: int, i1: int) -> np.ndarray:
    """Dense far-field correction of rows ``i0:i1`` (direct + surface image)."""
    sl = slice(i0, i1)
    return _quadrupole_with_image(segs.mid[sl], segs.tangent[sl], segs.length[sl], segs)


def _depth_gap(z_obs: np.ndarray, z_img_lo: float, z_img_hi: float) -> float:
    """Smallest vertical distance between observer depths and an image band."""
    lo, hi = float(z_obs.min()), float(z_obs.max())
    return max(0.0, z_img_lo - hi, lo - z_img_hi)


def series_corrections(
    segs: LineSegments, images: list[tuple[float, float, float]], min_distance: float
) -> SparseCorrection:
    """Exact minus point value of near Tagg/Sunde image pairs.

    Parameters
    ----------
    segs
        The segments (observers and sources).
    images
        ``(weight, offset, sign)`` of the series images ``n >= 1``.
    min_distance
        Clamp of the point kernel in m.

    Returns
    -------
    SparseCorrection
        Weighted corrections ``K^n (exact − 1/r)`` of the image pairs
        nearer than :data:`NEAR_FACTOR` segment lengths; usually empty
        (images are at least ``2(h_1 - z)`` away).
    """
    z = segs.mid[:, 2]
    reach = NEAR_FACTOR * float(segs.length.max())
    parts = []
    for weight, offset, sign in images:
        band = sorted((offset + sign * float(z.min()), offset + sign * float(z.max())))
        if _depth_gap(z, band[0], band[1]) >= reach:
            continue
        img = segs.image(offset, sign)
        i, j = near_pairs(segs.mid, segs.length, img.mid, img.length)
        if i.size:
            exact = galerkin_pairs(segs, i, img, j)
            point = _point_values(segs.mid, img.mid, i, j, min_distance)
            parts.append(SparseCorrection(i, j, weight * (exact - point)))
    return SparseCorrection.concatenate(parts)


def field_corrections(
    points: np.ndarray, segs: LineSegments, min_distance: float
) -> SparseCorrection:
    """Exact line potential minus point value at near field points.

    Direct and surface-image sources; the line potential is regularised
    with ``min_distance`` (the point kernel's clamp).
    """
    zero = np.zeros(points.shape[0])
    parts = []
    a2 = min_distance * min_distance
    for src in (segs, segs.image()):
        i, j = near_pairs(points, zero, src.mid, src.length)
        if i.size:
            exact = line_potential_pairs(
                points[i],
                src.start[j],
                src.tangent[j],
                src.length[j],
                np.full(i.size, a2),
            )
            parts.append(
                SparseCorrection(
                    i, j, exact - _point_values(points, src.mid, i, j, min_distance)
                )
            )
    return SparseCorrection.concatenate(parts)


def field_quadrupole_rows(points: np.ndarray, segs: LineSegments) -> np.ndarray:
    """Dense far-field correction at field points (direct + surface image)."""
    return _quadrupole_with_image(points, None, np.zeros(points.shape[0]), segs)


def series_field_corrections(
    points: np.ndarray,
    segs: LineSegments,
    images: list[tuple[float, float, float]],
    min_distance: float,
) -> SparseCorrection:
    """Weighted exact-minus-point corrections of near series images at field points."""
    zero = np.zeros(points.shape[0])
    a2 = min_distance * min_distance
    z_pts = points[:, 2]
    z = segs.mid[:, 2]
    reach = NEAR_FACTOR * float(segs.length.max())
    parts = []
    for weight, offset, sign in images:
        band = sorted((offset + sign * float(z.min()), offset + sign * float(z.max())))
        if _depth_gap(z_pts, band[0], band[1]) >= reach:
            continue
        img = segs.image(offset, sign)
        i, j = near_pairs(points, zero, img.mid, img.length)
        if i.size:
            exact = line_potential_pairs(
                points[i],
                img.start[j],
                img.tangent[j],
                img.length[j],
                np.full(i.size, a2),
            )
            point = _point_values(points, img.mid, i, j, min_distance)
            parts.append(SparseCorrection(i, j, weight * (exact - point)))
    return SparseCorrection.concatenate(parts)
