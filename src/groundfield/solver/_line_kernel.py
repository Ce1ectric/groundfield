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
:data:`SWITCH_FACTOR` times the longer segment: parallel pairs have a
closed form, other pairs use a Gauss–Legendre rule on the observing
segment, graded towards the point of closest approach, with the exact
inner line potential. Farther pairs use the point kernel plus the
leading (quadrupole) term of the segment average and the radius term of
the reduced kernel,

.. math::
    \\frac{1}{R} + \\frac{L_i^2 (3c_i^2 - 1) + L_j^2 (3c_j^2 - 1) - 12 a^2}
    {24 R^3},\\qquad c_k = \\frac{(P_i - P_j)\\cdot t_k}{R},

whose remainder is of order :math:`(L/R)^4`; between :data:`NEAR_FACTOR`
(6) and :data:`SWITCH_FACTOR` (7.5) segment lengths the exact and the far
value are blended with a C¹ weight (user decision D2). Field points (after
the solve) see the exact line potential near a segment and the point
kernel plus :math:`L_j^2 (3c_j^2 - 1)/(24R^3)` farther away.

Images
------
The soil-surface image is treated like a direct pair. The Tagg/Sunde
images of a two-layer soil (``n >= 1``) and the images of the cross-layer
correction (``image_2layer._LayeredImageFamilies``) are at least
:math:`2(h_1 - z)` away from a segment. Near image pairs are Galerkin
integrals in which the line potential of the image at a point of the
observer uses the transverse distance :math:`\\max(\\rho_\\perp, a)` of the
point from the image's axis (:func:`galerkin_pairs`, audit W3A-02): the
mean value of a line potential over the wire's circumference, exact for a
parallel image at any distance, and the reduced kernel for a collinear one
(:math:`\\rho_\\perp = 0`, a rod against its own image); an image within the
wire radius of its own segment (a wire within :math:`a/2` of the interface)
blends into the reduced kernel of the direct pair it then nearly coincides
with, so that a wire on the interface is continuous between the series and
the cross-layer path (:func:`image_shift`). Far images carry
the quadrupole term without the radius term (:func:`image_quadrupole_rows`,
tapered off between 60 and 120 segment lengths of vertical distance,
:data:`IMAGE_QUADRUPOLE_REACH`, and left out below
:data:`IMAGE_QUADRUPOLE_TOL`), or stay point sources with
``Engine.image_far_field = "point"``. The far-image work follows the pair
(:class:`FarImages`: an image is evaluated on the pairs whose longer segment
reaches it, audit W3A-01) and, in the reaction matrix, which is symmetric,
on its upper half (:func:`add_symmetric`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = [
    "FAR_FIELD_FORMS",
    "FarImages",
    "add_symmetric",
    "IMAGE_QUADRUPOLE_REACH",
    "IMAGE_QUADRUPOLE_TOL",
    "NEAR_FACTOR",
    "SELF_TERM_CROSSOVER",
    "SWITCH_FACTOR",
    "LineSegments",
]

#: Pairs closer than ``NEAR_FACTOR · max(L_i, L_j)`` (midpoint distance)
#: are integrated exactly; farther pairs use the quadrupole-corrected
#: point form, blended with the exact value up to :data:`SWITCH_FACTOR`.
NEAR_FACTOR: float = 6.0

#: Pairs farther than ``SWITCH_FACTOR · max(L_i, L_j)`` use the far form
#: alone. Between :data:`NEAR_FACTOR` and ``SWITCH_FACTOR`` segment lengths
#: the value is ``w · exact + (1 − w) · far`` with a weight ``w`` that falls
#: smoothly (C¹) from 1 to 0 in the squared distance (user decision D2 of
#: 2026-11-15, audit W2B-01/-02). Before, the form switched at six segment
#: lengths in one step; the pairs six segments apart on a straight wire and
#: every segment's own Tagg/Sunde image at a distance of six segment lengths
#: sat exactly on it, so the round-off of the coordinates decided their form
#: and results of the same world at different positions spread by up to
#: 4.4e-7 (direct pairs) and 2.6e-5 (images) in the impedance.
SWITCH_FACTOR: float = 7.5

#: Forms of the far images (``Engine.image_far_field``): far Tagg/Sunde
#: images (``n >= 1``) and far images of the cross-layer correction carry the
#: quadrupole term of the segment average like direct pairs (wave 3 of the
#: 2026.11 work, 10.28), or the point form of the releases before
#: (``"point"``, the author's decision of 2026-11-15; up to then a private
#: flag ``FAR_IMAGE_QUADRUPOLE``).
FAR_FIELD_FORMS: tuple[str, ...] = ("quadrupole", "point")

#: ``(SWITCH_FACTOR / NEAR_FACTOR)²``: end of the blend in ``r² / lim²``.
_SWITCH_U: float = (SWITCH_FACTOR / NEAR_FACTOR) ** 2

#: Vertical distances between an observer and an image, in units of the
#: longer segment of the pair, over which the quadrupole term of far images
#: is tapered off (full below the first, none beyond the second, a C¹ blend
#: between; a property of the pair alone). The term is below ``1/(8 G²)`` of
#: the image's point value at ``G`` segment lengths — 3.5e-5 at 60, the level
#: of the quadrupole remainder at the switch (2e-5) — while every image costs
#: the same: the series of 1000 over 30 Ohm m keeps 277 images per sign
#: pair, of which those within 120 segment lengths of 0.5 m at h_1 = 1.5 m
#: are 20, and images farther from every observer are skipped. Measured on 16
#: house grids: the impedance moves by at most 1e-6 against the quadrupole
#: on every image, the solve takes 1.2 instead of 3.9 times the time without
#: it (``item1_gain.py`` of stream w3xl).
IMAGE_QUADRUPOLE_REACH: tuple[float, float] = (60.0, 120.0)

#: A far image's quadrupole term is left out for a pair when its largest
#: possible size, ``|w| (L² + 3a²) / (6 dz²)`` relative to a unit point term
#: (``w`` the image weight, ``dz`` the vertical distance of observer and
#: image, ``L`` and ``a`` the longer segment and the radius of the pair), is
#: below this value — a thousandth of the tail the Tagg/Sunde series itself
#: leaves out (``Engine.image_series_tol``, default 1e-6). Decided per pair;
#: an image is skipped where no pair can reach it. Below a thin top layer the
#: images are dense in depth and their weights small: of the 58 images per
#: family within the reach of a house grid below h_1 = 0.5 m, 26 remain
#: (Z moves by less than 1e-10).
IMAGE_QUADRUPOLE_TOL: float = 1e-9

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
    X: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
    L: np.ndarray,
    a2: np.ndarray,
    transverse: bool = False,
    blend: np.ndarray | float = 0.0,
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
    transverse
        Use ``ρ² = max(ρ_⊥², a²) + blend · min(ρ_⊥², a²)``, ``ρ_⊥`` the
        transverse distance of ``X`` from the line, instead of
        ``ρ_⊥² + a²``: the potential of an image line averaged over the
        circumference of radius ``a`` around ``X`` (``blend = 0``) or the
        reduced kernel (``blend = 1``; audit W3A-02, see
        :func:`galerkin_pairs`).
    blend
        Weight of the reduced kernel with ``transverse``, broadcastable to
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
    if transverse:
        perp2 = np.maximum(w2 - s * s, 0.0)
        a2b = np.broadcast_to(a2, s.shape)
        rho2 = np.maximum(perp2, a2b) + np.broadcast_to(blend, s.shape) * np.minimum(
            perp2, a2b
        )
    else:
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


#: Below this ratio :math:`L/a` :func:`galerkin_self` uses the form without
#: cancellation. Only a part cut off at a layer interface can be that short
#: (the regular discretisation requires :math:`L > a`), so every other
#: segment keeps the historic arithmetic bit for bit.
_GALERKIN_SHORT_RATIO = 0.25


def galerkin_self(L: np.ndarray, a: np.ndarray) -> np.ndarray:
    r"""Self term ``(2/L)[arsinh(L/a) - sqrt(1 + (a/L)²) + a/L]`` in 1/m.

    The potential of a uniform line current of length :math:`L` averaged
    over its own surface (radius :math:`a`), per unit current density and
    without the factor :math:`\rho/(4\pi)`. It tends to :math:`1/a` (a
    sphere of radius :math:`a`) for :math:`L \to 0`. Below
    :math:`L/a = 0.25` it is evaluated as
    :math:`(2/L)[\operatorname{arsinh} x - x/(1 + \sqrt{1 + x^2})]`,
    :math:`x = L/a`, which avoids the cancellation of
    :math:`\sqrt{1 + (a/L)^2} - a/L` (that form loses all digits for
    :math:`L/a \lesssim 10^{-8}`; a part cut off at a layer interface can
    be that short, audit round 2, A2S-01).

    Parameters
    ----------
    L
        Segment lengths in m, ``> 0``.
    a
        Wire radii in m, ``> 0``.

    Returns
    -------
    numpy.ndarray
        Self terms in 1/m.
    """
    out = 2.0 * (np.arcsinh(L / a) - np.sqrt(1.0 + (a / L) ** 2) + a / L) / L
    x = np.asarray(L / a)
    short = x < _GALERKIN_SHORT_RATIO
    if np.any(short):
        Ls = np.broadcast_to(L, x.shape)[short]
        xs = x[short]
        out = np.array(out, dtype=float, copy=True)
        out[short] = 2.0 * (np.arcsinh(xs) - xs / (1.0 + np.sqrt(1.0 + xs * xs))) / Ls
    return out


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


#: A segment shorter than this fraction of the axis distance ``d`` of a
#: parallel pair is integrated numerically in :func:`_parallel_pairs`: the
#: closed form subtracts second antiderivatives of size ``d`` to obtain a
#: value of size ``L_i L_j / d`` and loses every digit when both segments
#: are much shorter than ``d`` (two parts of 1e-11 m cut off at a layer
#: interface, 8 mm apart: 0 instead of 125 1/m; audit round 2, A2S-01).
#: Regular segments are longer than the wire radius and pairs farther
#: apart than ``NEAR_FACTOR`` lengths are no near pairs, so this never
#: touches them.
_SHORT_PARALLEL_RATIO = 1e-3
_SHORT_NODES = 0.5 * (leggauss(6)[0] + 1.0)  # 6-point rule on [0, 1]
_SHORT_WEIGHTS = 0.5 * leggauss(6)[1]


def _parallel_pairs(
    Ai, ti, Li, Aj, tj, Lj, a2, transverse=False, blend=0.0
) -> np.ndarray:
    """Galerkin integral of (anti-)parallel segment pairs.

    Closed form, except for pairs with a segment shorter than
    :data:`_SHORT_PARALLEL_RATIO` times the axis distance: that segment is
    integrated with a 6-point Gauss–Legendre rule (exact to round-off,
    the integrand is smooth on its scale) and the other one in closed
    form, or both numerically when both are short.
    """
    w = Aj - Ai
    ya = np.einsum("pk,pk->p", w, ti)
    yb = ya + Lj * np.einsum("pk,pk->p", tj, ti)
    y1 = np.minimum(ya, yb)
    y2 = np.maximum(ya, yb)
    d_perp2 = np.maximum(np.einsum("pk,pk->p", w, w) - ya * ya, 0.0)
    if transverse:
        d = np.sqrt(np.maximum(d_perp2, a2) + blend * np.minimum(d_perp2, a2))
    else:
        d = np.sqrt(d_perp2 + a2)
    val = _g2(Li - y1, d) - _g2(-y1, d) - _g2(Li - y2, d) + _g2(-y2, d)
    val = val / (Li * Lj)
    short_i = Li < _SHORT_PARALLEL_RATIO * d
    short_j = Lj < _SHORT_PARALLEL_RATIO * d
    if np.any(short_i | short_j):
        val[short_i | short_j] = _short_parallel_pairs(
            Li, Lj, y1, y2, d, short_i, short_j
        )[short_i | short_j]
    return val


def _short_parallel_pairs(Li, Lj, y1, y2, d, short_i, short_j) -> np.ndarray:
    """Parallel-pair integral when a segment is much shorter than ``d``.

    Segment ``i`` covers ``[0, L_i]`` and segment ``j`` covers
    ``[y_1, y_2]`` of the axis of ``i``. Returns the mean of
    ``1/sqrt((y - s)² + d²)`` over both segments (in 1/m) for every pair;
    only the entries of the masked pairs are meaningful.
    """
    x, w = _SHORT_NODES, _SHORT_WEIGHTS
    out = np.zeros(Li.shape[0])
    both = short_i & short_j
    if np.any(both):
        s = Li[both, None, None] * x[None, :, None]
        y = y1[both, None, None] + (y2 - y1)[both, None, None] * x[None, None, :]
        f = 1.0 / np.sqrt((y - s) ** 2 + d[both, None, None] ** 2)
        out[both] = np.einsum("pmn,m,n->p", f, w, w)
    only_i = short_i & ~short_j
    if np.any(only_i):
        s = Li[only_i, None] * x[None, :]
        dd = d[only_i, None]
        inner = np.arcsinh((y2[only_i, None] - s) / dd) - np.arcsinh(
            (y1[only_i, None] - s) / dd
        )
        out[only_i] = (inner @ w) / Lj[only_i]
    only_j = short_j & ~short_i
    if np.any(only_j):
        y = y1[only_j, None] + (y2 - y1)[only_j, None] * x[None, :]
        dd = d[only_j, None]
        inner = np.arcsinh((Li[only_j, None] - y) / dd) + np.arcsinh(y / dd)
        out[only_j] = (inner @ w) / Li[only_j]
    return out


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


def _skew_pairs(Ai, ti, Li, Aj, tj, Lj, a2, transverse=False, blend=None) -> np.ndarray:
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
        b_j = 0.0 if blend is None else blend[mask][:, None]
        part = np.zeros(xs.shape[0])
        for side, extent in ((-1.0, xs), (+1.0, Lm - xs)):
            # nodes x = x* + side·extent·u, u in [0, 1] (graded towards 0)
            x = xs[:, None] + side * extent[:, None] * nodes[None, :]
            X = A_i[:, None, :] + x[:, :, None] * t_i[:, None, :]
            phi = line_potential_pairs(
                X,
                A_j[:, None, :],
                t_j[:, None, :],
                L_j[:, None],
                a2_j[:, None],
                transverse,
                b_j,
            )
            part += extent * (phi @ weights)
        total[mask] = part
    return total / Li


#: Pairs per call of the quadrature in :func:`galerkin_pairs`.
_GALERKIN_CHUNK: int = 1 << 14


def galerkin_pairs(
    obs: LineSegments,
    i: np.ndarray,
    src: LineSegments,
    j: np.ndarray,
    *,
    transverse: bool = False,
    shift: np.ndarray | None = None,
) -> np.ndarray:
    """Galerkin integral ``K_ij`` of segment pairs in 1/m.

    Parameters
    ----------
    obs, src
        Observing and source segments (``src`` may be an image set).
    i, j
        Pair indices into ``obs`` and ``src``.
    transverse
        For an image set ``src``: the inner line potential at a point of the
        observer uses the transverse distance ``max(ρ_⊥, a)`` of the point
        from the image's axis instead of ``sqrt(ρ_⊥² + a²)`` (audit W3A-02).
        The potential of a line averaged over a circle of radius ``a`` is
        ``ln max(ρ, a)`` in two dimensions (the mean-value property), so an
        image that passes the wire at ``ρ_⊥ >= a`` acts with its axis value
        (the reduced kernel left out ``a²/(2ρ²)`` of it: 8 % of the
        interface image of a wire 1 cm above the interface), and a collinear
        image (``ρ_⊥ = 0``, a rod against its own image) keeps the reduced
        kernel, which is exact there. Direct pairs and the soil-surface image
        keep the reduced kernel.
    shift
        With ``transverse``: per source of ``src`` the distance in m of the
        image from its own segment, across the segment's axis. An image within
        the wire radius of its segment (a wire within ``a/2`` of the
        interface) nearly coincides with it and takes the reduced kernel of
        the direct pair, blended with the weight
        ``max(0, 1 - shift²/a²)``: with the transverse distance alone the
        series path and the cross-layer path of the same wire on the
        interface differed by ``(ρ_1 - ρ_2)`` times the difference of the
        two kernels (0.14 % for a ring on a 1000/30 interface).

    Returns
    -------
    numpy.ndarray
        ``(1/(L_i L_j)) ∬ dl dl' / sqrt(|x - y|² + a²)`` per pair, with
        ``a² = (a_i² + a_j²)/2`` (``transverse``: as described above).

    Notes
    -----
    Pairs are evaluated in chunks of :data:`_GALERKIN_CHUNK`, so that the
    quadrature temporaries (up to 112 nodes × 3 coordinates per skew pair)
    stay bounded however many near pairs a world has.
    """
    if i.shape[0] > _GALERKIN_CHUNK:
        return np.concatenate(
            [
                galerkin_pairs(
                    obs,
                    i[k : k + _GALERKIN_CHUNK],
                    src,
                    j[k : k + _GALERKIN_CHUNK],
                    transverse=transverse,
                    shift=shift,
                )
                for k in range(0, i.shape[0], _GALERKIN_CHUNK)
            ]
        )
    Ai, ti, Li = obs.start[i], obs.tangent[i], obs.length[i]
    Aj, tj, Lj = src.start[j], src.tangent[j], src.length[j]
    a2 = 0.5 * (obs.radius[i] ** 2 + src.radius[j] ** 2)
    blend = np.zeros(i.shape[0])
    if transverse and shift is not None:
        blend = np.maximum(1.0 - shift[j] * shift[j] / a2, 0.0)
    out = np.empty(i.shape[0])
    par = np.abs(np.einsum("pk,pk->p", ti, tj)) > 1.0 - _PARALLEL_TOL
    if par.any():
        out[par] = _parallel_pairs(
            Ai[par],
            ti[par],
            Li[par],
            Aj[par],
            tj[par],
            Lj[par],
            a2[par],
            transverse,
            blend[par],
        )
    skew = ~par
    if skew.any():
        out[skew] = _skew_pairs(
            Ai[skew],
            ti[skew],
            Li[skew],
            Aj[skew],
            tj[skew],
            Lj[skew],
            a2[skew],
            transverse,
            blend[skew],
        )
    return out


def image_shift(segs: LineSegments, img: LineSegments) -> np.ndarray:
    """Distance of each image segment from its own segment across its axis, in m.

    ``img`` is :meth:`LineSegments.image` of ``segs``: the image is displaced
    vertically by ``Δz``, of which ``|Δz| sqrt(1 - t_z²)`` is across the
    segment's axis (zero for a vertical rod, whose image is collinear).
    """
    dz = img.mid[:, 2] - segs.mid[:, 2]
    tz = segs.tangent[:, 2]
    return np.abs(dz) * np.sqrt(np.maximum(1.0 - tz * tz, 0.0))


# ---------------------------------------------------------------------
# Pair search and far-field correction
# ---------------------------------------------------------------------


def near_pairs(
    obs_mid: np.ndarray,
    obs_len: np.ndarray,
    src_mid: np.ndarray,
    src_len: np.ndarray,
    factor: float = NEAR_FACTOR,
    gap: float = 0.0,
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

    ``gap`` is a lower bound of every pair distance in m (the depth gap of
    the observers and an image band): only the segments whose radius
    ``factor · L`` reaches it are queried (audit W3A-01; the pair set is
    the same).

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
    r_src = factor * src_len * pad + 1e-12
    reach = np.flatnonzero(r_src >= gap)
    if reach.size:
        _collect(
            cKDTree(obs_mid).query_ball_point(src_mid[reach], r=r_src[reach]),
            reach,
            query_is_source=True,
        )
    # Pairs near an observer segment: the sources within factor · L_i.
    r_obs = factor * obs_len * pad + 1e-12
    with_length = np.flatnonzero((obs_len > 0.0) & (r_obs >= gap))
    if with_length.size:
        _collect(
            cKDTree(src_mid).query_ball_point(
                obs_mid[with_length], r=r_obs[with_length]
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
    obs_radius: np.ndarray | None = None,
) -> np.ndarray:
    """Leading correction of the segment average to the point kernel.

    ``[L_i²(3c_i² - 1) + L_j²(3c_j² - 1) - 12 a²] / (24 R³)`` for every pair
    with ``R >= factor · max(L_i, L_j)``, zero for nearer pairs (they are
    integrated exactly). ``a² = (a_i² + a_j²)/2`` is the radius of the
    reduced kernel ``1/sqrt(r² + a²)`` of the exact pairs, whose expansion
    the term completes (``1/sqrt(R² + a²) = 1/R - a²/(2R³) + …``; wave 3 of
    the 2026.11 work: the far form used ``1/R``, ``a²/(2R²)`` = 1.4e-6 off at
    six segment lengths of 0.5 m). For field points pass ``obs_t=None`` and
    zero lengths: their far form keeps the plain point value (the 1 mm of
    their exact line potential only regularises points on the axis).

    Parameters
    ----------
    obs_mid, obs_t, obs_len
        Observer midpoints (m), unit tangents (``None`` for field points)
        and lengths (m).
    src
        Source segments.
    factor
        Near-pair factor.
    obs_radius
        Wire radii of the observing segments in m (with ``obs_t``); ``None``
        omits the radius term.

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
            if obs_radius is not None:
                q -= 12.0 * (0.5 * np.add.outer(obs_radius**2, src.radius**2))
        q /= 24.0 * r2 * np.sqrt(r2)
    return np.where(far, q, 0.0)


# ---------------------------------------------------------------------
# Continuous near/far switch (user decision D2)
# ---------------------------------------------------------------------


def _switch(r2: np.ndarray, lim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weights of the blended switch for pairs with ``r² < (SWITCH · L)²``.

    ``far`` is ``r² >= lim²`` with ``lim = NEAR_FACTOR · max(L_i, L_j)`` —
    the comparison of :func:`quadrupole_block`, on the same numbers, so a
    pair is ``far`` exactly where the dense far form was added. ``w`` is the
    weight of the exact value: 1 for near pairs, then
    ``1 − t²(3 − 2t)`` with ``t = (r²/lim² − 1)/(SWITCH²/NEAR² − 1)``,
    which reaches 0 with zero slope at ``SWITCH_FACTOR`` segment lengths.
    """
    lim2 = lim * lim
    far = r2 >= lim2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (r2 / lim2 - 1.0) / (_SWITCH_U - 1.0)
    t = np.clip(t, 0.0, 1.0)
    w = 1.0 - t * t * (3.0 - 2.0 * t)
    return np.where(far, w, 1.0), far


def switch_pairs(
    obs_mid: np.ndarray,
    obs_len: np.ndarray,
    src_mid: np.ndarray,
    src_len: np.ndarray,
    gap: float = 0.0,
    image: tuple[np.ndarray, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pairs that need the exact value, with their blending weights.

    Every pair with ``|P_i − Q_j| < SWITCH_FACTOR · max(L_i, L_j)``; ``gap``
    as in :func:`near_pairs`.

    ``image = (z_src, offset, sign)`` marks the sources as the image
    ``offset + sign · z_src`` of segments at depths ``z_src`` (Tagg/Sunde
    and cross-layer images): the distance of the switch is then formed with
    the vertical difference ``(z_i - sign · z_j) - offset`` of
    :func:`image_quadrupole_rows`, bit for bit, so that the dense far form
    and the blend decide a pair on the same number (audit W3A-01; a pair at
    six segment lengths exactly, such as two 0.4 m rod segments against an
    image 2.4 m away, otherwise got the far term twice or not at all).

    Returns
    -------
    tuple[numpy.ndarray, ...]
        Observer indices, source indices, the weight ``w`` of the exact
        value and ``far`` (the dense far form of :func:`quadrupole_block` /
        :func:`image_quadrupole_rows` is added for the pair).
    """
    i, j = near_pairs(obs_mid, obs_len, src_mid, src_len, SWITCH_FACTOR, gap)
    d = obs_mid[i] - src_mid[j]
    if image is not None:
        z_src, offset, sign = image
        d[:, 2] = (obs_mid[i, 2] - sign * z_src[j]) - offset
    r2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]
    w, far = _switch(r2, NEAR_FACTOR * np.maximum(obs_len[i], src_len[j]))
    return i, j, w, far


def pair_quadrupole(
    obs_mid: np.ndarray,
    obs_t: np.ndarray | None,
    obs_len: np.ndarray,
    src: LineSegments,
    i: np.ndarray,
    j: np.ndarray,
    obs_radius: np.ndarray | None = None,
) -> np.ndarray:
    """The far-field term of :func:`quadrupole_block` for index pairs, in 1/m.

    Without the near/far mask (callers pass far pairs only). ``obs_t=None``
    and zero lengths for field points.
    """
    d = obs_mid[i] - src.mid[j]
    r2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]
    tj = src.tangent[j]
    uj = d[:, 0] * tj[:, 0] + d[:, 1] * tj[:, 1] + d[:, 2] * tj[:, 2]
    q = (src.length[j] ** 2) * (3.0 * uj * uj / r2 - 1.0)
    if obs_t is not None:
        ti = obs_t[i]
        ui = d[:, 0] * ti[:, 0] + d[:, 1] * ti[:, 1] + d[:, 2] * ti[:, 2]
        q += (obs_len[i] ** 2) * (3.0 * ui * ui / r2 - 1.0)
        if obs_radius is not None:
            q -= 12.0 * (0.5 * (obs_radius[i] ** 2 + src.radius[j] ** 2))
    q /= 24.0 * r2 * np.sqrt(r2)
    return q


def _blended(
    exact: np.ndarray,
    point: np.ndarray,
    w: np.ndarray,
    far: np.ndarray,
    q: np.ndarray | None,
) -> np.ndarray:
    """Sparse correction to the dense ``point (+ q · far)`` of each pair.

    Near pairs: ``exact − point`` (as before the blend, bit for bit). Pairs
    in the band: ``w · (exact − point − q)``, so that the total is
    ``w · exact + (1 − w) · (point + q)``; ``q = None`` where the dense far
    form is the point form.
    """
    out = exact - point
    if np.any(far):
        rest = out[far] if q is None else out[far] - q[far]
        out[far] = w[far] * rest
    return out


def _corrections(
    obs: LineSegments | None,
    obs_mid: np.ndarray,
    obs_len: np.ndarray,
    src: LineSegments,
    i: np.ndarray,
    j: np.ndarray,
    w: np.ndarray,
    far: np.ndarray,
    point: np.ndarray,
    min_distance: float,
    *,
    with_quadrupole: bool,
    image: bool = False,
) -> np.ndarray:
    """Blended exact-minus-point values of segment or field-point pairs.

    ``obs`` are the observing segments (Galerkin integrals), or ``None`` for
    field points (exact line potential regularised with ``min_distance``).
    ``image`` marks ``src`` as Tagg/Sunde or cross-layer images: their
    Galerkin integrals use the transverse distance and their far form has no
    radius term (audit W3A-02, :func:`galerkin_pairs`).
    """
    if obs is not None:
        # An image set ``src`` is the image of the observing segments.
        shift = image_shift(obs, src) if image else None
        exact = galerkin_pairs(obs, i, src, j, transverse=image, shift=shift)
    else:
        exact = line_potential_pairs(
            obs_mid[i],
            src.start[j],
            src.tangent[j],
            src.length[j],
            np.full(i.size, min_distance * min_distance),
        )
    q = None
    if with_quadrupole and np.any(far):
        q = np.zeros(i.size)
        q[far] = pair_quadrupole(
            obs_mid,
            None if obs is None else obs.tangent,
            obs_len,
            src,
            i[far],
            j[far],
            obs_radius=None if (obs is None or image) else obs.radius,
        )
    return _blended(exact, point, w, far, q)


@dataclass(frozen=True)
class FarImages:
    """The far images some pair needs, with the shortest pair that needs each.

    Attributes
    ----------
    images
        ``(weight, offset, sign)`` of the images, in the order of the series.
    shortest
        Per image the shortest longer segment of a pair, in m, for which the
        image's far-field term can be nonzero: a pair with
        ``max(L_i, L_j) < shortest[k]`` lies beyond the taper
        (:data:`IMAGE_QUADRUPOLE_REACH`) or below the tolerance
        (:data:`IMAGE_QUADRUPOLE_TOL`) of image ``k``, whatever its position
        (its vertical distance to the image is at least the depth gap of the
        observers and the image band). Audit W3A-01: up to it every image any
        pair needed was evaluated on every pair, so eight 9.4 m pieces among
        1920 segments of 0.5 m multiplied the far-image work by 5.6.
    """

    images: tuple[tuple[float, float, float], ...] = ()
    shortest: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return len(self.images)

    def __bool__(self) -> bool:
        return bool(self.images)

    def __iter__(self):
        return iter(self.images)


#: Relative margin of :attr:`FarImages.shortest`: a pair is evaluated when its
#: longer segment is at least ``(1 - margin)`` times the bound, so the
#: round-off of the taper and tolerance tests cannot drop a nonzero term.
_SHORTEST_MARGIN = 1e-9


def far_images(
    z_obs: np.ndarray, segs: LineSegments, images: list[tuple[float, float, float]]
) -> FarImages:
    """The images that carry the far-field quadrupole term for some pair.

    For every image the depth gap ``g`` between the observers (depths
    ``z_obs``) and the image band bounds the vertical distance of every pair
    from below, so the image's term vanishes for a pair whose longer segment
    ``L`` satisfies ``L <= g / 120`` (the end of the taper,
    :data:`IMAGE_QUADRUPOLE_REACH`) or ``|w| (L² + 3 a_max²) / (6 g²) <``
    :data:`IMAGE_QUADRUPOLE_TOL`. The smaller of the two lengths is
    :attr:`FarImages.shortest`; an image that no segment reaches is left out
    (its taper or tolerance is zero for every pair, so leaving it out changes
    no number and saves its cost).
    """
    if not images or not z_obs.size or not segs.mid.shape[0]:
        return FarImages()
    z = segs.mid[:, 2]
    z_lo, z_hi = float(z.min()), float(z.max())
    l_max = float(segs.length.max())
    hi = IMAGE_QUADRUPOLE_REACH[1]
    reach = hi * l_max
    # The largest (L² + 3a²) of any pair: a field point has a = 0, a segment
    # pair a² = (a_i² + a_j²)/2 <= max a².
    a2_max = float(segs.radius.max()) ** 2
    size = (l_max * l_max + 3.0 * a2_max) / (6.0 * _qtol())
    kept, shortest = [], []
    for weight, offset, sign in images:
        band = sorted((offset + sign * z_lo, offset + sign * z_hi))
        gap = _depth_gap(z_obs, band[0], band[1])
        wa = abs(weight)
        if not (gap < reach and wa * size >= gap * gap):
            continue
        by_tol = np.sqrt(max(6.0 * _qtol() * gap * gap / wa - 3.0 * a2_max, 0.0))
        kept.append((weight, offset, sign))
        shortest.append(max(gap / hi, by_tol) * (1.0 - _SHORTEST_MARGIN))
    return FarImages(tuple(kept), np.array(shortest, dtype=float))


def far_image_rows(
    obs_mid: np.ndarray,
    obs_t: np.ndarray | None,
    obs_len: np.ndarray,
    segs: LineSegments,
    far: FarImages,
    obs_radius: np.ndarray | None = None,
) -> np.ndarray:
    """:func:`image_quadrupole_rows` of the far images, pair class by pair class.

    Every image is evaluated on the pairs whose longer segment reaches
    :attr:`FarImages.shortest` only: the rows of such observers against all
    sources and the other rows against such sources (audit W3A-01). The
    images are taken in the order of the series and every entry accumulates
    its terms in that order, so the block is bit-identical to one pass of
    :func:`image_quadrupole_rows` over all pairs and images (the terms left
    out are exact zeros). Arguments as :func:`image_quadrupole_rows`.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(obs_mid), len(segs.mid))``, in 1/m.
    """
    n_obs, n_src = obs_mid.shape[0], segs.mid.shape[0]
    out = np.zeros((n_obs, n_src))
    if not far or not n_obs or not n_src:
        return out
    obs_len = np.asarray(obs_len, dtype=float)
    distinct = np.unique(np.concatenate([obs_len, segs.length]))
    # Level of an image: the number of distinct lengths below its bound;
    # level 0 needs every pair.
    level = np.searchsorted(distinct, far.shortest, side="left")
    images = far.images
    k = 0
    while k < len(images):
        end = k + 1
        while end < len(images) and level[end] == level[k]:
            end += 1
        run = list(images[k:end])
        lev = int(level[k])
        k = end
        if lev == 0:
            image_quadrupole_rows(
                obs_mid, obs_t, obs_len, segs, run, obs_radius=obs_radius, out=out
            )
            continue
        if lev >= distinct.size:
            continue  # no pair reaches the image
        need = distinct[lev]
        rows = obs_len >= need
        cols = np.flatnonzero(segs.length >= need)
        if rows.any():
            r = np.flatnonzero(rows)
            blk = out[r]
            image_quadrupole_rows(
                obs_mid[r],
                None if obs_t is None else obs_t[r],
                obs_len[r],
                segs,
                run,
                obs_radius=None if obs_radius is None else obs_radius[r],
                out=blk,
            )
            out[r] = blk
        if cols.size and not rows.all():
            r = np.flatnonzero(~rows)
            ix = np.ix_(r, cols)
            blk = out[ix]
            image_quadrupole_rows(
                obs_mid[r],
                None if obs_t is None else obs_t[r],
                obs_len[r],
                segs.subset(cols),
                run,
                obs_radius=None if obs_radius is None else obs_radius[r],
                out=blk,
            )
            out[ix] = blk
    return out


def _qtol() -> float:
    """:data:`IMAGE_QUADRUPOLE_TOL`, floored at 1e-300 (0 keeps every term)."""
    return max(float(IMAGE_QUADRUPOLE_TOL), 1e-300)


def image_taper(dz: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Taper of the far-image quadrupole term of pairs (:data:`IMAGE_QUADRUPOLE_REACH`).

    ``dz`` is the vertical distance of observer and image in m, ``length``
    the longer segment of the pair in m.
    """
    lo, hi = IMAGE_QUADRUPOLE_REACH
    t = np.clip((np.abs(dz) * (1.0 / length) - lo) * (1.0 / (hi - lo)), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def image_quadrupole_rows(
    obs_mid: np.ndarray,
    obs_t: np.ndarray | None,
    obs_len: np.ndarray,
    segs: LineSegments,
    images: list[tuple[float, float, float]],
    obs_radius: np.ndarray | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Dense far-field term of far images, ``Σ_k w_k · τ_k · Q_k``, in 1/m.

    ``Q_k`` is the far-field term of :func:`quadrupole_block` of the
    observers against the image ``(offset_k, sign_k)`` of the segments
    (:meth:`LineSegments.image`), nonzero where
    ``r² >= (NEAR_FACTOR · max(L_i, L_j))²`` and the term can exceed
    :data:`IMAGE_QUADRUPOLE_TOL`; ``w_k`` the image weight and ``τ_k`` the
    taper :func:`image_taper` of the pair. Item 1 of wave 3
    (10.28): up to it the Tagg/Sunde images were point sources beyond six
    segment lengths. With numba (extra ``fast``) and enough work the
    compiled kernel runs; both paths give bit-identical blocks.

    Evaluation (audit W3A-01): the vertical distance is formed as
    ``(z_i - z_j) - o`` (``sign = +1``) or ``(z_i + z_j) - o``
    (``sign = -1``), so that the term of a pair against an image equals,
    bit for bit, the term of the swapped pair against the partner image
    (the same image for ``sign = -1``, offset ``-o`` for ``sign = +1``);
    ``Q`` is the polynomial ``[(P_0 + Δz (P_1 + P_2 Δz)) / R² - B] / (24 R³)``
    of the pair's constants, with one division and one square root per
    pair and image; the taper and the tolerance test are left out for an
    image where they hold for every pair of the block (they would give
    ``τ = 1`` and ``keep``, the same numbers).

    Parameters
    ----------
    obs_mid, obs_t, obs_len
        Observer midpoints (m), unit tangents (``None`` for field points)
        and lengths (m; zeros for field points).
    segs
        The source segments.
    images
        ``(weight, offset, sign)`` of every image.
    obs_radius
        Wire radii of the observing segments in m (the radius term of
        :func:`quadrupole_block`); ``None`` for field points.
    out
        C-contiguous float array of shape ``(len(obs_mid), len(segs.mid))``
        to add the terms to, image by image; ``None`` starts from zeros.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(obs_mid), len(segs.mid))`` (``out`` when given).
    """
    n_obs, n_src = obs_mid.shape[0], segs.mid.shape[0]
    if out is None:
        out = np.zeros((n_obs, n_src))
    if not images or not n_obs or not n_src:
        return out
    kernel = None
    if n_obs * n_src * len(images) >= LINE_NUMBA_MIN_PAIRS:
        kernel = _numba_image_quadrupole_rows()
    has_t = obs_t is not None
    t = obs_t if has_t else np.zeros_like(obs_mid)
    obs_len = np.asarray(obs_len, dtype=float)
    ro = obs_radius if (has_t and obs_radius is not None) else np.zeros(n_obs)
    plan = _image_plan(obs_mid[:, 2], obs_len, ro, segs, images)
    lo, hi = IMAGE_QUADRUPOLE_REACH
    if kernel is not None:
        c = np.ascontiguousarray
        kernel(
            c(obs_mid[:, 0]), c(obs_mid[:, 1]), c(obs_mid[:, 2]),
            c(t[:, 0]), c(t[:, 1]), c(t[:, 2]), c(obs_len), c(ro, dtype=float),
            c(segs.mid[:, 0]), c(segs.mid[:, 1]), c(segs.mid[:, 2]),
            c(segs.tangent[:, 0]), c(segs.tangent[:, 1]), c(segs.tangent[:, 2]),
            c(segs.length, dtype=float), c(segs.radius, dtype=float),
            *plan, float(NEAR_FACTOR), float(lo), float(hi), _qtol(), out,
        )  # fmt: skip
        return out
    # numpy: the operations of the compiled kernel, in sub-blocks of rows
    # (audit W3A-05: one block of the caller's row loop held about twenty
    # temporaries of its size, 161 instead of 80 MiB without numba).
    step = max(1, _NUMPY_SUB_BLOCK // max(n_src, 1))
    for r0 in range(0, n_obs, step):
        sl = slice(r0, min(n_obs, r0 + step))
        _image_rows_numpy(
            obs_mid[sl], t[sl], obs_len[sl], ro[sl], segs, plan, out[sl]
        )  # fmt: skip
    return out


#: Pair entries per temporary of the numpy path of
#: :func:`image_quadrupole_rows` (256 KiB): its sub-blocks hold about twenty
#: such arrays, a few MiB whatever the size of the caller's block.
_NUMPY_SUB_BLOCK: int = 1 << 15

#: Relative margin of the tests that leave the taper or the near/tolerance
#: test of an image out for a whole block (:func:`_image_plan`).
_BLOCK_MARGIN = 1e-6

#: Variants of the image loop: the near/tolerance test holds for every pair
#: (``KEEP``), the taper is one for every pair (``FULL``).
_KEEP, _FULL = 1, 2


def _image_plan(
    z_obs: np.ndarray,
    obs_len: np.ndarray,
    obs_radius: np.ndarray,
    segs: LineSegments,
    images: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Offsets, signs, weights / 24, ``|weights|`` and loop variants of the images.

    For each image the range of the vertical distance over the block's pairs
    (from the depth ranges of observers and sources) decides whether the
    near/tolerance test holds for every pair (``r² >= lim²`` with the
    longest pair, ``|w| qtol >= Δz²`` with the shortest) and whether the
    taper is one for every pair (``|Δz| <= 60 L`` with the shortest pair);
    with a margin of :data:`_BLOCK_MARGIN`. The kernel then skips the test
    or the taper; the numbers are those of the full test.
    """
    lo, _ = IMAGE_QUADRUPOLE_REACH
    z = segs.mid[:, 2]
    zo_lo, zo_hi = float(z_obs.min()), float(z_obs.max())
    zs_lo, zs_hi = float(z.min()), float(z.max())
    l_long = max(float(obs_len.max()), float(segs.length.max()))
    l_short = max(float(obs_len.min()), float(segs.length.min()))
    a2_min = 0.5 * (float(obs_radius.min()) ** 2 + float(segs.radius.min()) ** 2)
    lim_long = NEAR_FACTOR * l_long
    qtol_min = (l_short * l_short + 3.0 * a2_min) / (6.0 * _qtol())
    m = len(images)
    offsets = np.empty(m)
    signs = np.empty(m)
    weights = np.empty(m)
    wabs = np.empty(m)
    variant = np.zeros(m, dtype=np.int64)
    for k, (weight, offset, sign) in enumerate(images):
        if sign > 0:
            d_lo, d_hi = zo_lo - zs_hi, zo_hi - zs_lo
        else:
            d_lo, d_hi = zo_lo + zs_lo, zo_hi + zs_hi
        e_lo, e_hi = d_lo - offset, d_hi - offset
        adz_max = max(abs(e_lo), abs(e_hi))
        adz_min = 0.0 if e_lo <= 0.0 <= e_hi else min(abs(e_lo), abs(e_hi))
        flags = 0
        if adz_min * adz_min >= lim_long * lim_long * (1.0 + _BLOCK_MARGIN) and abs(
            weight
        ) * qtol_min >= adz_max * adz_max * (1.0 + _BLOCK_MARGIN):
            flags |= _KEEP
        if adz_max <= lo * l_short * (1.0 - _BLOCK_MARGIN):
            flags |= _FULL
        offsets[k] = offset
        signs[k] = 1.0 if sign > 0 else -1.0
        weights[k] = weight * (1.0 / 24.0)
        wabs[k] = abs(weight)
        variant[k] = flags
    return offsets, signs, weights, wabs, variant


def _image_rows_numpy(
    obs_mid: np.ndarray,
    t: np.ndarray,
    obs_len: np.ndarray,
    ro: np.ndarray,
    segs: LineSegments,
    plan: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    out: np.ndarray,
) -> None:
    """The far-image term of a few rows, added to ``out`` (numpy path).

    The operations of the compiled kernel
    (:func:`groundfield.solver._line_numba.image_quadrupole_rows`) in the
    same order.
    """
    lo, hi = IMAGE_QUADRUPOLE_REACH
    dx = np.subtract.outer(obs_mid[:, 0], segs.mid[:, 0])
    dy = np.subtract.outer(obs_mid[:, 1], segs.mid[:, 1])
    dxy2 = dx * dx + dy * dy
    hx = dx * segs.tangent[None, :, 0] + dy * segs.tangent[None, :, 1]
    gx = dx * t[:, 0:1] + dy * t[:, 1:2]
    del dx, dy
    ls2 = (segs.length * segs.length)[None, :]
    lo2 = (obs_len * obs_len)[:, None]
    tsz = segs.tangent[None, :, 2]
    tzo = t[:, 2:3]
    a = ls2 * hx
    bq = lo2 * gx
    p0 = 3.0 * (a * hx + bq * gx)
    a_t = a * tsz
    bq_t = bq * tzo
    p1p = 6.0 * (a_t + bq_t)
    p1m = 6.0 * (bq_t - a_t)
    p2 = 3.0 * (ls2 * tsz * tsz + lo2 * tzo * tzo)
    del a, bq, a_t, bq_t, hx, gx
    a2 = 0.5 * np.add.outer(ro * ro, segs.radius * segs.radius)
    b = ls2 + lo2
    lmax = np.maximum.outer(obs_len, segs.length)
    lim = NEAR_FACTOR * lmax
    lim2 = lim * lim
    inv_l = 1.0 / lmax
    inv_span = 1.0 / (hi - lo)
    qtol = (lmax * lmax + 3.0 * a2) * (1.0 / (6.0 * _qtol()))
    del lim, lmax, a2
    dp = np.subtract.outer(obs_mid[:, 2], segs.mid[:, 2])
    dm = np.add.outer(obs_mid[:, 2], segs.mid[:, 2])
    offsets, signs, weights, wabs, variant = plan
    for k in range(offsets.shape[0]):
        plus = signs[k] > 0.0
        dz = (dp if plus else dm) - offsets[k]
        p1 = p1p if plus else p1m
        r2 = dxy2 + dz * dz
        inv2 = 1.0 / np.maximum(r2, 1e-100)
        i3 = inv2 * np.sqrt(inv2)
        q = ((p2 * dz + p1) * dz + p0) * inv2 - b
        wk = weights[k]
        if variant[k] & _FULL:
            coef = wk * i3
        else:
            s = np.minimum(np.maximum((np.abs(dz) * inv_l - lo) * inv_span, 0.0), 1.0)
            tau = 1.0 - s * s * (3.0 - 2.0 * s)
            coef = (wk * tau) * i3
        term = coef * q
        if variant[k] & _KEEP:
            out += term
        else:
            keep = (r2 >= lim2) & (wabs[k] * qtol >= dz * dz)
            out += np.where(keep, term, 0.0)


def add_symmetric(matrix: np.ndarray, upper_rows, rows: int) -> None:
    """``matrix += F`` for a symmetric ``F`` given by its upper row blocks.

    ``upper_rows(i0, i1)`` returns ``F[i0:i1, i0:]`` (the row block from the
    diagonal on); the part below the diagonal block is added as the
    transpose of the rest (audit W3A-01: the far-image terms of the reaction
    matrix are symmetric — a pair against an image equals the swapped pair
    against the partner image — so half of them are evaluated). Each entry
    receives exactly one contribution.

    Parameters
    ----------
    matrix
        Square array, modified in place.
    upper_rows
        Callable ``(i0, i1) -> F[i0:i1, i0:]``.
    rows
        Rows per block.
    """
    n = matrix.shape[0]
    for i0 in range(0, n, rows):
        i1 = min(n, i0 + rows)
        block = upper_rows(i0, i1)
        matrix[i0:i1, i0:] += block
        if i1 < n:
            matrix[i1:, i0:i1] += block[:, i1 - i0 :].T


def _numba_image_quadrupole_rows():
    """The compiled image kernel, or ``None`` (numba missing or switched off)."""
    from groundfield.solver import _kernels

    if not _kernels.USE_NUMBA:
        return None
    try:
        from groundfield.solver._line_numba import image_quadrupole_rows
    except ImportError:  # numba is an optional extra
        return None
    return image_quadrupole_rows


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

    def symmetrized(self) -> SparseCorrection:
        """``(C + Cᵀ) / 2``, for a correction whose exact values are symmetric.

        The image corrections of the reaction matrix are symmetric
        (reciprocity: a pair against an image equals the swapped pair against
        the partner image), but the Galerkin integral of a skew image pair
        with the transverse distance (:func:`galerkin_pairs`) is evaluated
        from the observing segment and differs between the two orders by
        ``O(a²/ρ²)`` near the wire: up to 2.3e-8 of the largest entry of a
        rod and a ring through the interface (audit W3A-02). Each entry is
        replaced by the mean of the two orders (both are in the set, the
        near-pair criterion is symmetric); parallel and collinear pairs are
        symmetric already.
        """
        half = 0.5 * self.values
        return SparseCorrection.concatenate(
            [
                SparseCorrection(self.rows, self.cols, half),
                SparseCorrection(self.cols, self.rows, half),
            ]
        )

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
    :data:`SWITCH_FACTOR` segment lengths: the Galerkin integral instead of
    ``1/r`` (for the self image ``1/(2z)``) below :data:`NEAR_FACTOR`, the
    blend towards the far form of :func:`reaction_quadrupole_rows` beyond.
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
    i, j, w, far = switch_pairs(segs.mid, segs.length, segs.mid, segs.length)
    off = i != j
    i, j, w, far = i[off], j[off], w[off], far[off]
    if i.size:
        point = _point_values(segs.mid, segs.mid, i, j, min_distance)
        values = _corrections(
            segs, segs.mid, segs.length, segs, i, j, w, far, point, min_distance,
            with_quadrupole=True,
        )  # fmt: skip
        parts.append(SparseCorrection(i, j, values))
    img = segs.image()
    i, j, w, far = switch_pairs(segs.mid, segs.length, img.mid, img.length)
    if i.size:
        point = _point_values(segs.mid, img.mid, i, j, min_distance)
        self_image = i == j
        point[self_image] = 1.0 / (2.0 * np.abs(segs.mid[i[self_image], 2]))
        values = _corrections(
            segs, segs.mid, segs.length, img, i, j, w, far, point, min_distance,
            with_quadrupole=True,
        )  # fmt: skip
        parts.append(SparseCorrection(i, j, values))
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
    obs_radius: np.ndarray | None = None,
) -> np.ndarray:
    """``quadrupole_block`` of the segments plus that of their surface images."""
    img = segs.image()
    with_radius = obs_t is not None and obs_radius is not None
    kernel = None
    if obs_mid.shape[0] * segs.mid.shape[0] >= LINE_NUMBA_MIN_PAIRS:
        kernel = _numba_quadrupole_rows()
    if kernel is None:
        ro = obs_radius if with_radius else None
        return quadrupole_block(
            obs_mid, obs_t, obs_len, segs, obs_radius=ro
        ) + quadrupole_block(obs_mid, obs_t, obs_len, img, obs_radius=ro)
    out = np.empty((obs_mid.shape[0], segs.mid.shape[0]))
    t = obs_t if obs_t is not None else np.zeros_like(obs_mid)
    ro = obs_radius if with_radius else np.zeros(obs_mid.shape[0])
    c = np.ascontiguousarray
    kernel(
        c(obs_mid[:, 0]), c(obs_mid[:, 1]), c(obs_mid[:, 2]),
        c(t[:, 0]), c(t[:, 1]), c(t[:, 2]), c(obs_len, dtype=float), obs_t is not None,
        c(ro, dtype=float), with_radius,
        c(segs.mid[:, 0]), c(segs.mid[:, 1]), c(segs.mid[:, 2]), c(img.mid[:, 2]),
        c(segs.tangent[:, 0]), c(segs.tangent[:, 1]), c(segs.tangent[:, 2]),
        c(img.tangent[:, 2]), c(segs.length, dtype=float), c(segs.radius, dtype=float),
        float(NEAR_FACTOR), out,
    )  # fmt: skip
    return out


def reaction_quadrupole_rows(segs: LineSegments, i0: int, i1: int) -> np.ndarray:
    """Dense far-field correction of rows ``i0:i1`` (direct + surface image)."""
    sl = slice(i0, i1)
    return _quadrupole_with_image(
        segs.mid[sl], segs.tangent[sl], segs.length[sl], segs, segs.radius[sl]
    )


def _depth_gap(z_obs: np.ndarray, z_img_lo: float, z_img_hi: float) -> float:
    """Smallest vertical distance between observer depths and an image band."""
    lo, hi = float(z_obs.min()), float(z_obs.max())
    return max(0.0, z_img_lo - hi, lo - z_img_hi)


def series_corrections(
    segs: LineSegments,
    images: list[tuple[float, float, float]],
    min_distance: float,
    quadrupole: bool = True,
) -> SparseCorrection:
    """Exact minus far-form value of near Tagg/Sunde image pairs.

    Parameters
    ----------
    segs
        The segments (observers and sources).
    images
        ``(weight, offset, sign)`` of the series images ``n >= 1``.
    min_distance
        Clamp of the point kernel in m.
    quadrupole
        The far form of the images carries the quadrupole term (``True``,
        ``Engine.image_far_field = "quadrupole"``) or is the point form.

    Returns
    -------
    SparseCorrection
        Weighted corrections of the image pairs nearer than
        :data:`SWITCH_FACTOR` segment lengths: ``K^n (exact − 1/r)`` below
        :data:`NEAR_FACTOR`, the blend towards the far form (point plus
        :func:`image_quadrupole_rows`) beyond; usually few (images are at
        least ``2(h_1 - z)`` away). An image whose depth band is at least
        :data:`SWITCH_FACTOR` longest segment lengths from every observer is
        skipped: no pair of it can be nearer, and the blend is zero there.
    """
    z = segs.mid[:, 2]
    reach = SWITCH_FACTOR * float(segs.length.max())
    parts = []
    for weight, offset, sign in images:
        band = sorted((offset + sign * float(z.min()), offset + sign * float(z.max())))
        gap = _depth_gap(z, band[0], band[1])
        if gap >= reach:
            continue
        img = segs.image(offset, sign)
        i, j, w, far = switch_pairs(
            segs.mid, segs.length, img.mid, img.length, gap, image=(z, offset, sign)
        )
        if i.size:
            point = _point_values(segs.mid, img.mid, i, j, min_distance)
            values = _corrections(
                segs, segs.mid, segs.length, img, i, j, w, far, point, min_distance,
                with_quadrupole=quadrupole, image=True,
            )  # fmt: skip
            parts.append(SparseCorrection(i, j, weight * values))
    return SparseCorrection.concatenate(parts)


def field_corrections(
    points: np.ndarray, segs: LineSegments, min_distance: float
) -> SparseCorrection:
    """Exact line potential minus point value at near field points.

    Direct and surface-image sources nearer than :data:`SWITCH_FACTOR`
    segment lengths, blended towards the far form beyond
    :data:`NEAR_FACTOR`; the line potential is regularised with
    ``min_distance`` (the point kernel's clamp).
    """
    zero = np.zeros(points.shape[0])
    parts = []
    for src in (segs, segs.image()):
        i, j, w, far = switch_pairs(points, zero, src.mid, src.length)
        if i.size:
            point = _point_values(points, src.mid, i, j, min_distance)
            values = _corrections(
                None, points, zero, src, i, j, w, far, point, min_distance,
                with_quadrupole=True,
            )  # fmt: skip
            parts.append(SparseCorrection(i, j, values))
    return SparseCorrection.concatenate(parts)


def field_quadrupole_rows(points: np.ndarray, segs: LineSegments) -> np.ndarray:
    """Dense far-field correction at field points (direct + surface image)."""
    return _quadrupole_with_image(points, None, np.zeros(points.shape[0]), segs)


def series_field_corrections(
    points: np.ndarray,
    segs: LineSegments,
    images: list[tuple[float, float, float]],
    min_distance: float,
    quadrupole: bool = True,
) -> SparseCorrection:
    """Weighted exact-minus-far-form corrections of near series images at field points.

    ``quadrupole`` as in :func:`series_corrections`.
    """
    zero = np.zeros(points.shape[0])
    z_pts = points[:, 2]
    z = segs.mid[:, 2]
    reach = SWITCH_FACTOR * float(segs.length.max())
    parts = []
    for weight, offset, sign in images:
        band = sorted((offset + sign * float(z.min()), offset + sign * float(z.max())))
        gap = _depth_gap(z_pts, band[0], band[1])
        if gap >= reach:
            continue
        img = segs.image(offset, sign)
        i, j, w, far = switch_pairs(
            points, zero, img.mid, img.length, gap, image=(z, offset, sign)
        )
        if i.size:
            point = _point_values(points, img.mid, i, j, min_distance)
            values = _corrections(
                None, points, zero, img, i, j, w, far, point, min_distance,
                with_quadrupole=quadrupole,
            )  # fmt: skip
            parts.append(SparseCorrection(i, j, weight * values))
    return SparseCorrection.concatenate(parts)
