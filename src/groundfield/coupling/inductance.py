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
``include_internal`` (default ``True``; the solver passes ``False`` for
the conductors of ``conductor_type="cable_shield"``, thin-walled tubes,
user decision Q33). A per-unit-length
reference that includes the internal flux uses the geometric mean
radius $\mathrm{GMR} = a\,e^{-1/4}$; with ``include_internal=False``
the matching reference uses $\mathrm{GMR} = a$. The frequency
dependence of the internal impedance (skin effect) is provided by
:meth:`groundfield.conductors.Conductor.internal_impedance_per_length`
as a correction relative to this DC convention.

**Wires lying beside each other.** Where a point of one wire lies
beside another wire (its foot on that wire's axis falls inside the
wire) closer than the sum of the radii $r_i + r_j$ — two conductors in
one trench, a service drop routed back along its PEN main, coincident
leads, wires leaving a junction or crossing at a small angle — the
distance from the other axis is taken as $r_i + r_j$ (touching wires)
inside the Neumann integral of :func:`build_inductance_matrix`, or as
the radius of the thicker wire where the thinner one runs inside it and
the thicker one is marked as a cable shield (``shields=``; a core inside
a cable shield couples as a filament inside a thin tube). The marker is
explicit since the user decision Q32 of 2026.11: two unmarked wires on
one axis are separate conductors that touch. For parallel wires the rule
refers to whole conductors (``conductors=``, one label per conductor, as
the solver passes them; audit W1A-02 of the calendar release): a thinner
wire inside a cable shield couples at the shield radius at every axial
offset (exact for a filament on the axis of a thin tube), and a point lies
beside another conductor when its foot falls inside that conductor, so
the sub-pieces of two conductors lying beside each other add up to the
lumped pair (a 100 m coax in 1 m pieces read 2.2 % low before). Pieces
tilted against each other by more than the closed form's tolerance of
about 4.5e-5 rad, at any angle and whatever their length, get the same
two readings with the stretches inside, beside and away from the partner
found exactly (audits W1R2-01, W1R3-01, W1R3-02; a core tilted by 4.5e-5
rad read 7.4 % low in 0.25 m pieces, with a step at the tolerance; until
W1R3 the rule stopped where the lateral distance changed by more than the
touching distance along a piece, or at $|\cos\varphi| = 0.99$, and a core
crossing its shield's axis read -8.6e-5 of the loop off the lumped pair
in 10 m pieces, two wires leaving a junction stepped by 1.5e-3 at 8.1°).
The filament integral diverges for coincident axes (two coincident 50 m
leads read +587j Ω at 50 Hz) and exceeds the self inductance once the
wires are closer than about their radius (an indefinite matrix, a TN
network with drops along its PEN main had a negative input reactance).
The rule acts pointwise and on both one-sided integrals, so the value
is continuous in the geometry and independent of the order and
direction of the segments (audit round 3 of the calendar release; the
whole-pair switch of audit round 2 depended on which segment was the
row). Chain neighbours and bends turning by less than 90° keep their
closed forms.

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

MU_0 = 4.0e-7 * math.pi
"""Vacuum permeability μ0 = 4π · 10⁻⁷ H/m (the conventional value).

Exact in the pre-2019 SI; the CODATA 2018 value 1.25663706212e-6 differs
by 5.5 · 10⁻¹⁰ relative, which is irrelevant here.
"""

ImageConvention = Literal["none", "additive", "antiparallel"]
"""Image convention of an inductance matrix.

- ``"none"``: free space, no image;
- ``"additive"``: perfect magnetic mirror at the soil surface (ADR-0004);
- ``"antiparallel"``: perfect electric mirror, Carson's baseline.
"""

TouchingRule = Literal["quadrature", "exact"]
"""Treatment of touching and coaxial segment pairs in the Neumann integral.

- ``"exact"``: closed forms (the default);
- ``"quadrature"``: the 16 x 16 Gauss–Legendre rule (the default up to
  0.15.0).
"""

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

    Raises
    ------
    ValueError
        If ``length`` or ``wire_radius`` is not positive.
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

    Raises
    ------
    ValueError
        If ``length`` or ``distance`` is not positive.
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


#: Filaments from one vertex at an angle below 0.81 deg (``cos > 1 - 1e-4``)
#: use the cancellation-free form of :func:`_folded_vertex_mutual`; the
#: direct form of :func:`_vertex_integral` loses about ``1e-16 / (1 - cos)``
#: (2.7e-9 just above the coaxial tolerance of 4.5e-5 rad).
_VERTEX_STABLE_COS = 1.0 - 1e-4


def _folded_vertex_mutual(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    la: float,
    lb: float,
    ua: np.ndarray,
    ub: np.ndarray,
    dot: float,
    tol: float,
) -> float | None:
    """Filaments from one vertex at a small angle (H), or ``None``.

    Used below 0.81 deg (:data:`_VERTEX_STABLE_COS`), including pairs that
    pass the coaxial test of :func:`_touching_mutual`
    ($|\\,|\\cos\\varphi| - 1| \\le 10^{-9}$, an angle below 4.5e-5 rad) but
    fold out of the common axis: the vertex formula of
    :func:`_vertex_integral` holds, but its direct evaluation cancels.
    With $\\sigma = \\sin(\\varepsilon/2)$ from the chord between the two
    directions away from the vertex, $q = 4\\ell_a\\ell_b\\sigma^2$,
    $R = \\sqrt{(\\ell_a - \\ell_b)^2 + q}$ the far-end distance and
    $S = \\ell_a + \\ell_b + R$,

    $$
    I = \\ell_a \\ln\\frac{S}{\\ell_a + R - \\ell_b}
      + \\ell_b \\ln\\frac{S}{\\ell_b + R - \\ell_a},
    $$

    where the smaller denominator is $q/(R + |\\ell_a - \\ell_b|)$ —
    accurate down to any non-zero angle. Up to audit round 3 such pairs
    went to the 16x16 rule (16 % off at 1.5e-6 rad, 3.8 % at 1e-5 rad).
    Returns ``None`` when the pair shares no end point or the far end of
    *b* lies on *a*'s axis (overlapping coaxial filaments: divergent).
    """
    for vertex_a, away_a in ((p1_a, ua), (p2_a, -ua)):
        for vertex_b, far_b, away_b in ((p1_b, p2_b, ub), (p2_b, p1_b, -ub)):
            if float(np.linalg.norm(vertex_a - vertex_b)) > tol:
                continue
            rel = far_b - vertex_a
            if float(np.linalg.norm(rel - float(rel @ ua) * ua)) <= tol:
                return None  # folded onto the axis: divergent
            chord = away_a - away_b
            q = la * lb * float(chord @ chord)  # 4 la lb sin^2(eps/2)
            if q <= 0.0:
                return None
            diff = abs(la - lb)
            R = math.sqrt(diff * diff + q)
            S = la + lb + R
            big, small = R + diff, q / (R + diff)
            d_a, d_b = (big, small) if la >= lb else (small, big)
            integral = la * math.log(S / d_a) + lb * math.log(S / d_b)
            return MU_0 / (4.0 * math.pi) * dot * integral
    return None


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
    angle (also below the coaxial tolerance, :func:`_folded_vertex_mutual`).
    Returns ``None`` for every other pair (and for overlapping coaxial
    pairs, whose filament integral diverges), so that the caller falls
    back to its general rule.
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
            # parallel on distinct axes (the Grover parallel form), or two
            # filaments from one vertex at a tiny angle (audit A3C-4)
            return _folded_vertex_mutual(
                p1_a, p2_a, p1_b, p2_b, la, lb, ua, ub, dot, tol
            )
        s1 = float(rel1 @ ua)
        s2 = float((p2_b - p1_a) @ ua)
        lo, hi = min(s1, s2), max(s1, s2)
        if lo >= la - tol:
            gap = max(lo - la, 0.0)
        elif hi <= tol:
            gap = max(-hi, 0.0)
        else:
            # overlapping coaxial filaments diverge; filaments from one
            # vertex at a tiny angle do not (audit A3C-4)
            return _folded_vertex_mutual(
                p1_a, p2_a, p1_b, p2_b, la, lb, ua, ub, dot, tol
            )
        sign = 1.0 if dot > 0.0 else -1.0
        return MU_0 / (4.0 * math.pi) * sign * _coaxial_gap_integral(la, lb, gap)
    for vertex_a, away_a in ((p1_a, ua), (p2_a, -ua)):
        for vertex_b, away_b in ((p1_b, ub), (p2_b, -ub)):
            if float(np.linalg.norm(vertex_a - vertex_b)) <= tol:
                cos_eps = float(away_a @ away_b)
                if cos_eps > _VERTEX_STABLE_COS:
                    # small angle: the direct form cancels (audit A3C-4)
                    return _folded_vertex_mutual(
                        p1_a, p2_a, p1_b, p2_b, la, lb, ua, ub, dot, tol
                    )
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
        # Filaments from one vertex at a tiny angle (A3C-4): the scalar
        # routine, for the few that share an end point with segment a.
        for k in ks[~keep]:
            if (
                min(
                    float(np.linalg.norm(va - vb))
                    for va in (p1_a, p2_a)
                    for vb in (p1_b[k], p2_b[k])
                )
                <= tol[k]
            ):
                val = _touching_mutual(p1_a, p2_a, p1_b[k], p2_b[k])
                if val is not None:
                    values[k] = val
                    valid[k] = True
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

#: ``1 - cos^2`` below which :func:`_close_pair_mutual` takes the closest
#: approach of the axes from a cross product (angles below about 1e-3 rad);
#: above it the direct form loses less than about 1e-16 / (1 - cos^2).
_CLOSE_PAIR_ROBUST_DEN = 1e-6


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
    closest to *b*'s line, and towards the ends of *a*. Measured against
    an adaptive mpmath reference: at most 5e-11 for directions differing
    by more than 1e-6 rad (nearly parallel wires close together, crossing
    and skew pairs, branches starting on a segment), 8.6e-11 at 1e-7 rad,
    1.0e-9 for filaments crossing at 1e-8 rad — there the crossing point
    of the two axes is only known to about 1e-16/1e-8 of the length
    (audit round 3; before, the closest approach came from
    $1 - \\cos^2\\varphi$, which cancels: 9.5e-4 at 1e-8 rad). Not for
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
    den = 1.0 - dot * dot

    def rho_at(x: float) -> float:
        v = v0 + x * ua
        return max(float(np.linalg.norm(v - (c0 + x * dot) * ub)), floor)

    points = {0.0, la}
    for target in (0.0, lb):
        x = (target - c0) / dot
        if 0.0 < x < la:
            points.add(x)
    if den > _CLOSE_PAIR_ROBUST_DEN:
        x = (dot * c0 - float(v0 @ ua)) / den  # closest approach of the lines
        if 0.0 < x < la:
            points.add(x)
    else:
        # Nearly parallel axes (below about 1e-3 rad): 1 - dot^2 cancels, the
        # closest approach comes from the cross product n = ua x ub instead,
        # s* = ((p1_b - p1_a) x ub) . n / |n|^2 (audit round 3, A3C-4:
        # crossing filaments at 1e-8 rad were 9.5e-4 off, the mark of the
        # crossing point lost below 1 - dot^2 = 1e-14).
        n = np.cross(ua, ub)
        sin2 = float(n @ n)
        if sin2 > 1e-300:
            x = float(np.cross(p1_b - p1_a, ub) @ n) / sin2
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

    Accuracy (measured against adaptive mpmath references, audit round 3
    of the calendar release): closed forms to round-off; filaments from
    one vertex to round-off on axis-parallel geometry and to 4e-10 in a
    general orientation down to 3e-9 rad (the angle itself carries about
    1e-16/angle); close pairs as stated in :func:`_close_pair_mutual`
    (5e-11 above 1e-6 rad, 1.0e-9 at 1e-8 rad). Pairs inside the
    parallel tolerance — directions within ``parallel_tol`` in cos (4.5e-5
    rad) and a lateral offset that varies by at most ``parallel_tol`` m
    along the segment — use the parallel closed form at the offset of the
    start point, off by about the relative variation of the offset
    (4.2e-9 for wires 1 cm apart tilted by 1e-10 rad over 5 m, 1.7e-5 at
    1 µm).

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

    Raises
    ------
    ValueError
        If ``touching`` is not ``"exact"`` or ``"quadrature"``, or if a
        segment of the generic (non-parallel, non-touching) case has
        zero length.
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


def _shield_array(shields, M: int) -> np.ndarray | None:
    """Broadcast ``shields`` (``None``, bool or per-segment array) to (M,).

    ``None`` and ``False`` mark no segment (returns ``None``, the
    assembly of the 2026.11 development line before user decision Q32 for
    every pair that is not a core inside a shield).
    """
    if shields is None:
        return None
    arr = np.asarray(shields, dtype=bool)
    if arr.ndim == 0:
        return np.full(M, bool(arr)) if bool(arr) else None
    if arr.shape != (M,):
        raise ValueError(
            f"shields must be a bool or have shape ({M},), got {arr.shape}"
        )
    return arr if arr.any() else None


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
    shields: bool | np.ndarray | None = None,
    conductors: np.ndarray | None = None,
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
    shield = _shield_array(shields, M)
    flags = np.zeros(M, dtype=bool) if shield is None else shield
    labels = _conductor_labels(conductors, M)
    L = np.zeros((M, M), dtype=float)
    L_dir = np.zeros((M, M), dtype=float)
    hits: list[tuple[int, int]] = []
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
            beside = _beside_mutual(
                np.asarray(p1_i, dtype=float),
                np.asarray(p2_i, dtype=float),
                float(wire_radii[i]),
                np.asarray(p1_j, dtype=float),
                np.asarray(p2_j, dtype=float),
                float(wire_radii[j]),
                shield_a=bool(flags[i]),
                shield_b=bool(flags[j]),
            )
            if beside is not None:
                m = beside
                hits.append((i, j))
            L_dir[i, j] = L_dir[j, i] = m
            if mode != "none":
                p1_j_img = _mirror(p1_j)
                p2_j_img = _mirror(p2_j)
                m_img = neumann_mutual(
                    p1_i, p2_i, p1_j_img, p2_j_img, touching=touching
                )
                m_img = _touching_wires(
                    m_img,
                    p1_i,
                    p2_i,
                    wire_radii[i],
                    p1_j_img,
                    p2_j_img,
                    wire_radii[j],
                    shields=(bool(flags[i]), bool(flags[j])),
                )
                L[i, j] = L[j, i] = sign * m_img
    # conductors lying beside each other along one axis (audit W1A-02)
    p1_all = np.asarray(seg_endpoints[:, 0, :], dtype=float)
    p2_all = np.asarray(seg_endpoints[:, 1, :], dtype=float)
    _apply_bundles(
        L_dir, p1_all, p2_all, np.asarray(wire_radii, dtype=float), shield, labels, hits
    )
    off = ~np.eye(M, dtype=bool)
    L[off] = L_dir[off] if mode == "none" else L_dir[off] + L[off]
    return L


def _touching_wires(
    m: float,
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    r_a: float,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    r_b: float,
    shields: tuple[bool, bool] = (False, False),
) -> float:
    """The beside rule (:func:`_beside_mutual`) for the loop oracle.

    Returns the mutual of wires lying beside each other closer than
    touching, or ``m`` (the filament value) where the rule does not act.
    """
    val = _beside_mutual(
        np.asarray(p1_a, dtype=float),
        np.asarray(p2_a, dtype=float),
        float(r_a),
        np.asarray(p1_b, dtype=float),
        np.asarray(p2_b, dtype=float),
        float(r_b),
        shield_a=shields[0],
        shield_b=shields[1],
    )
    return m if val is None else val


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


#: ``|cos phi|`` above which two segments count as near-parallel (8.1 deg)
#: in the Carson tiling (its validity domain, audit round 2, A2C-1).
_PARALLEL_COS_MIN = 0.99

# ---------------------------------------------------------------------
# Wires lying beside each other closer than touching (audit round 3)
# ---------------------------------------------------------------------

#: Pre-selection of the pairs checked by :func:`_beside_mutual`: closest
#: points of the two segments within this factor of the touching distance.
#: Generous on purpose — the decision itself is taken by the exact,
#: order-free test of :func:`_beside_mutual`.
_BESIDE_PREFILTER = 1.0 + 1e-6

#: Relative margin below the touching distance $r_a + r_b$: two wires at
#: exactly that axis distance (side by side, touching) keep their filament
#: value bit for bit.
_BESIDE_MARGIN = 1e-9

_Point = tuple[float, float, float]


def _canonical_segment(p1: np.ndarray, p2: np.ndarray) -> tuple[_Point, _Point]:
    """End points of a segment in a fixed (lexicographic) order.

    The beside rule is evaluated on these, so that its value does not
    depend on the direction of either segment or on which of the two is
    the row of the assembly (bit for bit).
    """
    a = (float(p1[0]), float(p1[1]), float(p1[2]))
    b = (float(p2[0]), float(p2[1]), float(p2[2]))
    return (a, b) if a <= b else (b, a)


def _axial_span(
    b1: _Point, b2: _Point, ext: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float]:
    """Interval on the axis of segment ``b1 -> b2`` (from ``b1``, in m)
    between the feet of the two conductor end points ``ext``."""
    lb = math.hypot(b2[0] - b1[0], b2[1] - b1[1], b2[2] - b1[2])
    ub = ((b2[0] - b1[0]) / lb, (b2[1] - b1[1]) / lb, (b2[2] - b1[2]) / lb)
    t = [
        (float(e[0]) - b1[0]) * ub[0]
        + (float(e[1]) - b1[1]) * ub[1]
        + (float(e[2]) - b1[2]) * ub[2]
        for e in ext
    ]
    return min(t), max(t)


def _beside_plan(
    a1: _Point,
    a2: _Point,
    b1: _Point,
    b2: _Point,
    d_touch: float,
    r_thin: float,
    r_thick: float,
    inside_rule: bool,
    extent: tuple[float, float] | None = None,
) -> tuple[bool, tuple]:
    """Pieces of segment *a* for the one-sided integral of the beside rule.

    Splits *a* at the points where it passes an end of *b* (its foot on
    *b*'s axis leaves *b*), comes closest to *b*'s axis, or where its
    distance $\\rho$ from that axis crosses the touching distance
    $d = r_a + r_b$ or $r_\\text{thick} - r_\\text{thin}$. On each piece
    the effective distance is constant in kind: $\\rho$ itself (free), $d$
    (the stretch lies beside *b* closer than touching: wires lying side by
    side), or, with ``inside_rule`` (the thicker wire is a cable shield),
    $r_\\text{thick}$ (the thinner wire runs inside the shield: a core
    inside a cable shield couples as a filament inside a thin tube).
    Returns ``(clamped, plan)``: whether a stretch longer than the vertex
    tolerance is clamped, and the data :func:`_beside_integral` needs. The
    axis geometry is computed from cross products, which stay accurate for
    nearly parallel axes (no $1 - \\cos^2$ cancellation).

    ``extent`` (the bundle rule of conductors that are not parallel, audits
    W1R2-01 and W1R3 of the calendar release): the interval $[t_\\text{lo}, t_\\text{hi}]$ on
    *b*'s axis (measured from ``b1``) that the conductor of *b* occupies. A
    point then lies beside *b* when its foot falls inside that interval, and
    the inside reading holds at every axial offset, as for parallel pieces
    (:func:`_bundle_mutual_batch`); the inner integral stays over *b*. With
    ``None`` (the default) the arithmetic is the one of the pointwise rule.
    """
    ax, ay, az = a2[0] - a1[0], a2[1] - a1[1], a2[2] - a1[2]
    bx, by, bz = b2[0] - b1[0], b2[1] - b1[1], b2[2] - b1[2]
    la = math.hypot(ax, ay, az)
    lb = math.hypot(bx, by, bz)
    ua = (ax / la, ay / la, az / la)
    ub = (bx / lb, by / lb, bz / lb)
    dot = ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2]
    w = (a1[0] - b1[0], a1[1] - b1[1], a1[2] - b1[2])
    c0 = w[0] * ub[0] + w[1] * ub[1] + w[2] * ub[2]  # foot t0(s) = c0 + s dot

    def rho_at(x: float) -> float:
        t0 = c0 + x * dot
        return math.hypot(
            w[0] + x * ua[0] - t0 * ub[0],
            w[1] + x * ua[1] - t0 * ub[1],
            w[2] + x * ua[2] - t0 * ub[2],
        )

    marks = {0.0, la}
    if dot != 0.0:
        for target in (0.0, lb) if extent is None else (0.0, lb, *extent):
            x = (target - c0) / dot
            if 0.0 < x < la:
                marks.add(x)
    n = (
        ua[1] * ub[2] - ua[2] * ub[1],
        ua[2] * ub[0] - ua[0] * ub[2],
        ua[0] * ub[1] - ua[1] * ub[0],
    )
    sin2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
    if sin2 > 1e-300:
        # closest approach of the axes, s* = ((b1 - a1) x ub) . n / |n|^2,
        # and their distance |(b1 - a1) . n| / |n|; rho(s)^2 = delta^2 +
        # (s - s*)^2 |n|^2 on the line.
        e = (-w[0], -w[1], -w[2])
        exu = (
            e[1] * ub[2] - e[2] * ub[1],
            e[2] * ub[0] - e[0] * ub[2],
            e[0] * ub[1] - e[1] * ub[0],
        )
        s_star = (exu[0] * n[0] + exu[1] * n[1] + exu[2] * n[2]) / sin2
        if 0.0 < s_star < la:
            marks.add(s_star)
        delta = abs(e[0] * n[0] + e[1] * n[1] + e[2] * n[2]) / math.sqrt(sin2)
        for d in (d_touch, r_thick - r_thin):
            if d > delta:
                half = math.sqrt((d - delta) * (d + delta) / sin2)
                for x in (s_star - half, s_star + half):
                    if 0.0 < x < la:
                        marks.add(x)
    edges = sorted(marks)
    tol = _TOUCH_TOL_REL * max(la, lb)
    d_clamp = d_touch * (1.0 - _BESIDE_MARGIN)
    inside_rule = inside_rule and r_thin < r_thick
    pieces: list[tuple[float, float, float]] = []  # (lo, hi, rho_e or nan)
    clamped = False
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        mid = 0.5 * (lo + hi)
        t0 = c0 + mid * dot
        rho_e = math.nan
        if extent is not None:
            # bundle rule: beside the partner's conductor, inside at every
            # axial offset (audit W1R2-01)
            rho = rho_at(mid)
            if inside_rule and rho + r_thin <= r_thick:
                rho_e = r_thick
            elif extent[0] <= t0 <= extent[1] and rho < d_clamp:
                rho_e = d_touch
            if not math.isnan(rho_e):
                clamped = clamped or hi - lo > tol
        elif 0.0 <= t0 <= lb:
            rho = rho_at(mid)
            if rho < d_clamp:
                inside = inside_rule and rho + r_thin <= r_thick
                rho_e = r_thick if inside else d_touch
                clamped = clamped or hi - lo > tol
        pieces.append((lo, hi, rho_e))
    return clamped, (la, lb, ua, ub, dot, w, c0, rho_at, pieces)


def _beside_integral(plan: tuple) -> float:
    """One-sided integral of the beside rule over segment *a*, in m.

    $\\int_0^{\\ell_a} [\\operatorname{asinh}((\\ell_b - t_0)/\\rho_e) +
    \\operatorname{asinh}(t_0/\\rho_e)]\\,ds$: the inner integral over *b*
    in closed form at the effective distance $\\rho_e$ of
    :func:`_beside_plan`, the outer one with 16-point Gauss–Legendre panels
    graded geometrically towards the ends of every piece, starting at the
    local length scale of the integrand there ($\\rho_e$ on a clamped
    piece, $\\rho$ on a free one; as in :func:`_close_pair_mutual`).
    """
    la, lb, ua, ub, dot, w, c0, rho_at, pieces = plan
    floor = 1e-30 * max(la, lb)
    p_lo: list[float] = []
    p_hi: list[float] = []
    p_rho: list[float] = []
    for lo, hi, rho_e in pieces:
        free = math.isnan(rho_e)
        mid = 0.5 * (lo + hi)
        left, step = [lo], rho_e if not free else max(rho_at(lo), floor)
        while left[-1] + step < mid:
            left.append(left[-1] + step)
            step *= 2.0
        right, step = [hi], rho_e if not free else max(rho_at(hi), floor)
        while right[-1] - step > mid:
            right.append(right[-1] - step)
            step *= 2.0
        cuts = left + [mid] + right[::-1]
        p_lo += cuts[:-1]
        p_hi += cuts[1:]
        p_rho += [rho_e] * (len(cuts) - 1)
    lo_arr = np.asarray(p_lo)[:, None]
    half = 0.5 * (np.asarray(p_hi)[:, None] - lo_arr)
    x = (lo_arr + half * (_GL_NODES[None, :] + 1.0)).ravel()
    wts = (half * _GL_WEIGHTS[None, :]).ravel()
    t0 = c0 + x * dot
    rho = np.sqrt(
        (w[0] + x * ua[0] - t0 * ub[0]) ** 2
        + (w[1] + x * ua[1] - t0 * ub[1]) ** 2
        + (w[2] + x * ua[2] - t0 * ub[2]) ** 2
    )
    rho_e = np.repeat(np.asarray(p_rho), _GL_NODES.size)
    free = np.isnan(rho_e)
    rho_e[free] = np.maximum(rho[free], floor)
    g = np.arcsinh((lb - t0) / rho_e) + np.arcsinh(t0 / rho_e)
    return float(wts @ g)


def _beside_mutual(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    r_a: float,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    r_b: float,
    *,
    shield_a: bool = False,
    shield_b: bool = False,
    ext_a: tuple[np.ndarray, np.ndarray] | None = None,
    ext_b: tuple[np.ndarray, np.ndarray] | None = None,
) -> float | None:
    """Mutual inductance of two wires lying beside each other, in H.

    Two wires whose axes come closer than the sum of their radii
    $r_a + r_b$ along a stretch where they lie beside each other (the
    foot of a point of one wire on the other wire's axis falls inside the
    other wire) — coincident or overlapping leads, two cables in one
    trench, a service drop routed back along its PEN main, wires leaving a
    junction or crossing at a small angle — have no meaningful filament
    mutual inductance: it diverges for coincident axes and exceeds the
    self inductance of the wires once they are closer than about their
    radius (an indefinite inductance matrix, a negative loop reactance).
    Along such a stretch the distance from the other wire's axis is taken
    as the distance of touching wires, $r_a + r_b$, pointwise inside the
    Neumann integral; where the thinner wire runs inside the thicker one
    ($\\rho + r_\\text{thin} \\le r_\\text{thick}$) and the thicker one is
    marked as a cable shield (``shield_a`` / ``shield_b``; the solver marks
    the conductors with ``conductor_type="cable_shield"``, user decision
    Q32 of 2026.11), as the shield's radius — a core inside a cable shield
    couples as a filament inside a thin tube, at the tube radius (the
    geometric mean distance of a point inside a circle from the circle).
    Two unmarked wires on one axis are two separate conductors and touch
    (from the 2026.11 development line up to Q32 every thinner wire inside
    a thicker one was read as a core inside a shield). Elsewhere the
    integrand is the filament one, so the value is continuous in the
    geometry, and pairs that only meet end to end (chain neighbours, bends
    of less than 90°) are not affected.

    The one-sided integral over *a* (inner integral over *b* in closed
    form) and the one over *b* differ where the rule acts; their mean is
    used, evaluated on the end points in a fixed order — the value is the
    same, bit for bit, for either order of the two segments and either
    direction of each (up to the sign of the orientation factor). Returns
    ``None`` when the rule does not act (no stretch beside the partner
    closer than touching), so that the caller keeps its closed form or
    quadrature value.

    ``ext_a`` and ``ext_b`` (both or neither) are the end points of the
    straight conductors the two segments belong to: the bundle rule of
    conductors that are not parallel (:func:`_apply_tilted_bundles`, audits
    W1R2-01, W1R3),
    in which a point lies beside the partner's whole conductor and a wire
    inside a cable shield couples at the shield radius at every axial
    offset, as for parallel pieces.
    """
    d_touch = float(r_a) + float(r_b)
    if not d_touch > 0.0:
        return None
    da = (
        float(p2_a[0]) - float(p1_a[0]),
        float(p2_a[1]) - float(p1_a[1]),
        float(p2_a[2]) - float(p1_a[2]),
    )
    db = (
        float(p2_b[0]) - float(p1_b[0]),
        float(p2_b[1]) - float(p1_b[1]),
        float(p2_b[2]) - float(p1_b[2]),
    )
    la = math.hypot(*da)
    lb = math.hypot(*db)
    dot = (
        da[0] / la * (db[0] / lb)
        + da[1] / la * (db[1] / lb)
        + da[2] / la * (db[2] / lb)
    )
    if dot == 0.0:
        return None  # orthogonal: zero whatever the distance
    a1, a2 = _canonical_segment(p1_a, p2_a)
    b1, b2 = _canonical_segment(p1_b, p2_b)
    r_thin, r_thick = min(float(r_a), float(r_b)), max(float(r_a), float(r_b))
    # user decision Q32: the inside reading needs the thicker wire to be a
    # cable shield (equal radii never count as one inside the other).
    inside = bool(shield_a) if float(r_a) > float(r_b) else bool(shield_b)
    span_b = span_a = None
    if ext_a is not None and ext_b is not None:
        span_b = _axial_span(b1, b2, ext_b)
        span_a = _axial_span(a1, a2, ext_a)
    clamped_a, plan_a = _beside_plan(
        a1, a2, b1, b2, d_touch, r_thin, r_thick, inside, span_b
    )
    clamped_b, plan_b = _beside_plan(
        b1, b2, a1, a2, d_touch, r_thin, r_thick, inside, span_a
    )
    if not (clamped_a or clamped_b):
        return None
    total = _beside_integral(plan_a) + _beside_integral(plan_b)
    return MU_0 / (4.0 * math.pi) * dot * (0.5 * total)


def _beside_batch(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    r_a: float,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    r_b: np.ndarray,
    shield_a: bool = False,
    shield_b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_beside_mutual` of one segment against K partners.

    The partners whose closest points come within the touching distance
    (pre-selected with the clamped two-segment distance,
    :func:`groundfield.conductors.proximity.segment_distance`) are
    evaluated by the scalar rule; returns ``(values, mask)``, ``values[k]``
    in H where ``mask[k]``. ``shield_a`` and ``shield_b`` (one flag per
    partner, ``None`` for none) mark cable shields (user decision Q32).
    """
    K = p1_b.shape[0]
    values = np.zeros(K, dtype=float)
    mask = np.zeros(K, dtype=bool)
    if K == 0:
        return values, mask
    from groundfield.conductors.proximity import segment_distance

    r_b = np.asarray(r_b, dtype=float)
    dist, _, _ = segment_distance(
        np.broadcast_to(p1_a, (K, 3)), np.broadcast_to(p2_a, (K, 3)), p1_b, p2_b
    )
    near = dist <= (float(r_a) + r_b) * _BESIDE_PREFILTER + 1e-12
    for k in np.flatnonzero(near):
        val = _beside_mutual(
            p1_a,
            p2_a,
            r_a,
            p1_b[k],
            p2_b[k],
            float(r_b[k]),
            shield_a=bool(shield_a),
            shield_b=False if shield_b is None else bool(shield_b[k]),
        )
        if val is not None:
            values[k] = val
            mask[k] = True
    return values, mask


# ---------------------------------------------------------------------
# Conductors lying beside each other along one axis (audit W1A-02)
# ---------------------------------------------------------------------


def _lexicographic_ends(
    p1: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """End points of K segments in lexicographic order, shapes ``(K, 3)``."""
    swap = (p1[:, 0] > p2[:, 0]) | (
        (p1[:, 0] == p2[:, 0])
        & ((p1[:, 1] > p2[:, 1]) | ((p1[:, 1] == p2[:, 1]) & (p1[:, 2] > p2[:, 2])))
    )
    lo = np.where(swap[:, None], p2, p1)
    hi = np.where(swap[:, None], p1, p2)
    return lo, hi


def _distance_to_line(q: np.ndarray, p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Distance in m of the points ``q`` from the lines ``p + t u``, (K,)."""
    w = q - p
    r = w - _row_dots(w, u)[:, None] * u
    return np.sqrt(_row_dots(r, r))


def _x_log_abs_x(x: np.ndarray) -> np.ndarray:
    """$|x| \\ln|x|$ with the value 0 at 0 (the on-axis limit of $G_d$)."""
    ax = np.abs(x)
    out = np.zeros_like(ax)
    pos = ax > 0.0
    out[pos] = ax[pos] * np.log(ax[pos])
    return out


def _stretch_integral(
    s1: np.ndarray,
    s2: np.ndarray,
    beta_lo: np.ndarray,
    beta_hi: np.ndarray,
    d: np.ndarray,
    on_axis: np.ndarray,
) -> np.ndarray:
    r"""$\int_{s_1}^{s_2}\!\int_{\beta_\text{lo}}^{\beta_\text{hi}}
    dt\,ds/\sqrt{(t - s)^2 + d^2}$ for parallel filaments, in m.

    $= G(\beta_\text{hi} - s_1) - G(\beta_\text{hi} - s_2)
    - G(\beta_\text{lo} - s_1) + G(\beta_\text{lo} - s_2)$ with
    $G(x) = x \operatorname{asinh}(x/d) - \sqrt{x^2 + d^2}$; for filaments on
    one axis (``on_axis``, $d = 0$, the stretch outside the partner) its limit
    $G(x) = |x| \ln|x|$. Zero for an empty stretch ($s_2 \le s_1$).
    """
    d_safe = np.where(on_axis, 1.0, d)

    def G(x: np.ndarray) -> np.ndarray:
        off = x * np.arcsinh(x / d_safe) - np.hypot(x, d_safe)
        return np.where(on_axis, _x_log_abs_x(x), off)

    J = (G(beta_hi - s1) - G(beta_hi - s2)) - (G(beta_lo - s1) - G(beta_lo - s2))
    return np.where(s2 > s1, J, 0.0)


def _one_sided_beside(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    d_e: np.ndarray,
    delta: np.ndarray,
    on_axis: np.ndarray,
    unbounded: np.ndarray,
) -> np.ndarray:
    """One-sided integral of the beside rule over *a* for parallel pairs, in m.

    The inner integral over *b* is taken at the effective distance ``d_e``
    on the stretch of *a* beside the partner's conductor (end points
    ``e1``, ``e2`` on *b*'s axis; the whole of *a* where ``unbounded``), at
    the lateral distance ``delta`` elsewhere. All points and the results
    have K rows.
    """
    la = np.sqrt(_row_dots(a2 - a1, a2 - a1))
    u = (a2 - a1) / la[:, None]
    pb1 = _row_dots(b1 - a1, u)
    pb2 = _row_dots(b2 - a1, u)
    lo, hi = np.minimum(pb1, pb2), np.maximum(pb1, pb2)
    pe1 = _row_dots(e1 - a1, u)
    pe2 = _row_dots(e2 - a1, u)
    e_lo, e_hi = np.minimum(pe1, pe2), np.maximum(pe1, pe2)
    zero = np.zeros_like(la)
    e_lo = np.where(unbounded, zero, e_lo)
    e_hi = np.where(unbounded, la, e_hi)
    in_lo = np.maximum(zero, e_lo)
    in_hi = np.minimum(la, e_hi)
    no = np.zeros_like(on_axis)
    inside = _stretch_integral(in_lo, in_hi, lo, hi, d_e, no)
    before = _stretch_integral(zero, np.minimum(la, e_lo), lo, hi, delta, on_axis)
    after = _stretch_integral(np.maximum(zero, e_hi), la, lo, hi, delta, on_axis)
    return inside + (before + after)


def _bundle_mutual_batch(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    r_a: np.ndarray,
    shield_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    r_b: np.ndarray,
    shield_b: np.ndarray,
    ext_a: tuple[np.ndarray, np.ndarray] | None = None,
    ext_b: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""The beside rule for K pairs of parallel wires, in closed form (H).

    For two parallel segments whose axes are $\delta$ apart, below the
    touching distance $r_a + r_b$, the beside rule (:func:`_beside_mutual`)
    takes the inner integral over one segment at the effective distance $d$
    (the shield radius for a thinner wire inside a cable shield, else
    $r_a + r_b$; :func:`_beside_plan`) for the points of the other segment
    that lie beside it, and at $\delta$ elsewhere; the mutual is
    $\mu_0/(4\pi)\,\operatorname{sign}(\hat u_a\cdot\hat u_b)$ times the mean
    of the two one-sided integrals. For parallel filaments each stretch has
    the closed form of :func:`_stretch_integral` (Grover's parallel filaments
    with an axial offset).

    ``ext_a`` and ``ext_b`` are the end points of the conductors the two
    segments belong to (on their axes; default: the segments themselves). A
    point lies beside the *conductor* of the partner when its foot falls
    inside that conductor (audit W1A-02): the pieces of two conductors lying
    beside each other along one axis — a core inside a cable shield, two
    cables drawn on one axis — then couple at $d$ on the whole stretch they
    share, also between pieces that do not overlap, and the pieces add up
    to the lumped pair (with the segment's own extent, a 100 m coax in 1 m
    pieces read 2.2 % low). A thinner wire inside a cable shield
    ($\delta + r_\text{thin} \le r_s$, the *inside* reading) couples at
    the shield radius at every axial offset, wherever the two conductors
    end: for a filament on the axis of a thin tube this is exact, every
    point of the tube is $\sqrt{x^2 + r_s^2}$ away. The value is continuous
    in the axial geometry (the stretch beside the partner grows from zero)
    and the same, bit for bit, for either order of the two segments and
    either direction of each (up to the orientation sign).

    Returns
    -------
    values, valid, inside : np.ndarray
        ``values[k]`` in H where ``valid[k]`` (parallel within the coaxial
        tolerance and closer than touching), undefined otherwise;
        ``inside[k]`` marks the inside reading.
    """
    da = p2_a - p1_a
    db = p2_b - p1_b
    la = np.sqrt(_row_dots(da, da))
    lb = np.sqrt(_row_dots(db, db))
    dot = _row_dots(da / la[:, None], db / lb[:, None])
    parallel = np.abs(np.abs(dot) - 1.0) <= 1e-9
    a1, a2 = _lexicographic_ends(p1_a, p2_a)
    b1, b2 = _lexicographic_ends(p1_b, p2_b)
    ea1, ea2 = (a1, a2) if ext_a is None else _lexicographic_ends(*ext_a)
    eb1, eb2 = (b1, b2) if ext_b is None else _lexicographic_ends(*ext_b)
    ua = (a2 - a1) / la[:, None]
    ub = (b2 - b1) / lb[:, None]
    # lateral distance: the four point-line distances, summed order-free
    delta = 0.25 * (
        (_distance_to_line(b1, a1, ua) + _distance_to_line(b2, a1, ua))
        + (_distance_to_line(a1, b1, ub) + _distance_to_line(a2, b1, ub))
    )
    r_a = np.asarray(r_a, dtype=float)
    r_b = np.asarray(r_b, dtype=float)
    d_touch = r_a + r_b
    valid = parallel & (delta < d_touch * (1.0 - _BESIDE_MARGIN))
    r_thin = np.minimum(r_a, r_b)
    r_thick = np.maximum(r_a, r_b)
    thick_is_shield = np.where(r_a > r_b, shield_a, shield_b)
    inside = thick_is_shield & (r_thin < r_thick) & (delta + r_thin <= r_thick)
    d_e = np.where(inside, r_thick, d_touch)
    on_axis = delta <= _TOUCH_TOL_REL * np.maximum(la, lb)
    i_ab = _one_sided_beside(a1, a2, b1, b2, eb1, eb2, d_e, delta, on_axis, inside)
    i_ba = _one_sided_beside(b1, b2, a1, a2, ea1, ea2, d_e, delta, on_axis, inside)
    sign = np.where(dot > 0.0, 1.0, -1.0)
    values = (MU_0 / (4.0 * math.pi)) * sign * (0.5 * (i_ab + i_ba))
    return values, valid & parallel, valid & inside


def _conductor_labels(conductors, M: int) -> np.ndarray:
    """Integer conductor label per segment (``None``: one per segment)."""
    if conductors is None:
        return np.arange(M)
    arr = np.asarray(conductors)
    if arr.shape != (M,):
        raise ValueError(f"conductors must have shape ({M},), got {arr.shape}")
    _, labels = np.unique(arr, return_inverse=True)
    return labels.reshape(M)


def _conductor_extents(
    p1: np.ndarray, p2: np.ndarray, members: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """End points of the conductor made of the segments ``members``.

    The segments of one conductor label that lie on one straight line form
    one straight conductor: its extreme end points along that line, for
    every member. Otherwise every member keeps its own end points.
    """
    q1, q2 = p1[members], p2[members]
    if members.size == 1:
        return q1, q2
    o = q1[0]
    u = (q2[0] - o) / float(np.linalg.norm(q2[0] - o))
    ends = np.concatenate([q1, q2])
    t = (ends - o) @ u
    perp = ends - o - t[:, None] * u
    span = float(t.max() - t.min())
    if float(np.sqrt((perp * perp).sum(axis=1)).max()) > _TOUCH_TOL_REL * span:
        return q1, q2
    e1 = np.broadcast_to(ends[int(np.argmin(t))], q1.shape)
    e2 = np.broadcast_to(ends[int(np.argmax(t))], q1.shape)
    return e1, e2


def _apply_bundles(
    L_dir: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    radii: np.ndarray,
    shields: np.ndarray | None,
    labels: np.ndarray,
    hits: list[tuple[int, int]],
) -> None:
    """Couple the pieces of conductors lying beside each other (in place).

    ``hits`` are the pairs on which the beside rule acted. Two conductors
    (``labels``) with a pair of pieces that is parallel and closer than
    touching along a stretch — a core inside a cable shield, two cables
    drawn on one axis — lie beside each other: every parallel pair of their
    pieces closer than touching gets the beside rule against the partner's
    whole conductor, in closed form (:func:`_bundle_mutual_batch`), so the
    pieces add up to the lumped pair (audit W1A-02). Pieces of one
    conductor, and conductors that only meet end to end (a main split at
    its taps), keep their values. A thinner wire parallel inside a cable
    shield (``shields``) couples at the shield radius at every axial offset
    (:func:`_apply_inside_shields`), so a shield made of several
    conductors, or a core and shield in sections, add up as well. Pieces
    tilted against each other by more than the closed form's tolerance, at
    any angle, get the same two readings, with the exact pointwise
    classification (:func:`_apply_tilted_bundles`, audits W1R2-01,
    W1R3-01, W1R3-02). Writes the direct block ``L_dir`` symmetrically;
    does nothing without hits and shields.
    """
    flags = np.zeros(len(radii), dtype=bool) if shields is None else shields
    if hits:
        _apply_touching_bundles(L_dir, p1, p2, radii, flags, labels, hits)
    if flags.any():
        _apply_inside_shields(L_dir, p1, p2, radii, flags)
    if hits or flags.any():
        _apply_tilted_bundles(L_dir, p1, p2, radii, flags, labels, hits)


def _apply_inside_shields(
    L_dir: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    radii: np.ndarray,
    flags: np.ndarray,
) -> None:
    """Every thinner wire parallel inside a cable shield couples at its radius.

    The inside reading of :func:`_bundle_mutual_batch` for every pair of a
    shield segment and a thinner parallel segment inside it, at every axial
    offset (overlapping, end to end, apart; user decision Q32, audit
    W1A-02). Without shields nothing changes.
    """
    da = p2 - p1
    la = np.sqrt(_row_dots(da, da))
    u = da / la[:, None]
    for i in np.flatnonzero(flags):
        dots = _row_dots(u, u[i])
        lateral = np.maximum(
            _distance_to_line(
                p1, np.broadcast_to(p1[i], p1.shape), np.broadcast_to(u[i], u.shape)
            ),
            _distance_to_line(
                p2, np.broadcast_to(p1[i], p1.shape), np.broadcast_to(u[i], u.shape)
            ),
        )
        cand = np.flatnonzero(
            (radii < radii[i])
            & (np.abs(np.abs(dots) - 1.0) <= 1e-6)
            & (lateral <= radii[i] * (1.0 + 1e-6))
        )
        if cand.size == 0:
            continue
        K = cand.size
        values, _, inside = _bundle_mutual_batch(
            np.broadcast_to(p1[i], (K, 3)),
            np.broadcast_to(p2[i], (K, 3)),
            np.full(K, radii[i]),
            np.ones(K, dtype=bool),
            p1[cand],
            p2[cand],
            radii[cand],
            flags[cand],
        )
        L_dir[i, cand[inside]] = values[inside]
        L_dir[cand[inside], i] = values[inside]


def _apply_touching_bundles(
    L_dir: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    radii: np.ndarray,
    flags: np.ndarray,
    labels: np.ndarray,
    hits: list[tuple[int, int]],
) -> None:
    """The beside rule against the partner's conductor for bundled pairs.

    See :func:`_apply_bundles`; the inside reading is applied afterwards
    by :func:`_apply_inside_shields`.
    """
    I = np.array([i for i, _ in hits])
    J = np.array([j for _, j in hits])
    keep = labels[I] != labels[J]
    I, J = I[keep], J[keep]
    if I.size == 0:
        return
    _, close, inside = _bundle_mutual_batch(
        p1[I], p2[I], radii[I], flags[I], p1[J], p2[J], radii[J], flags[J]
    )
    close &= ~inside  # the inside reading is applied by _apply_inside_shields
    bundles = sorted(
        {
            (min(a, b), max(a, b))
            for a, b in zip(
                labels[I[close]].tolist(), labels[J[close]].tolist(), strict=True
            )
        }
    )
    extents: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for c in sorted({c for pair in bundles for c in pair}):
        members = np.flatnonzero(labels == c)
        extents[c] = (members, *_conductor_extents(p1, p2, members))
    for ca, cb in bundles:
        ia, ea1, ea2 = extents[ca]
        ib, eb1, eb2 = extents[cb]
        # rows of the first conductor in chunks: the temporaries of the
        # closed form stay bounded (audit W1R2-03; element-wise, so the
        # values are the same bit for bit as for all pairs at once)
        rows = max(1, _BUNDLE_CHUNK // ib.size)
        for start in range(0, ia.size, rows):
            sl = slice(start, start + rows)
            na = ia[sl].size
            II = np.repeat(ia[sl], ib.size)
            JJ = np.tile(ib, na)
            values, valid, inside = _bundle_mutual_batch(
                p1[II],
                p2[II],
                radii[II],
                flags[II],
                p1[JJ],
                p2[JJ],
                radii[JJ],
                flags[JJ],
                ext_a=(
                    np.repeat(ea1[sl], ib.size, axis=0),
                    np.repeat(ea2[sl], ib.size, axis=0),
                ),
                ext_b=(np.tile(eb1, (na, 1)), np.tile(eb2, (na, 1))),
            )
            valid &= ~inside
            L_dir[II[valid], JJ[valid]] = values[valid]
            L_dir[JJ[valid], II[valid]] = values[valid]


# ---------------------------------------------------------------------
# Tilted conductors lying beside each other (audits W1R2-01, W1R3)
# ---------------------------------------------------------------------

#: Pairs of pieces evaluated at once by the bundle rules: the temporaries
#: stay a few MiB whatever the length of the conductors (audit W1R2-03; the
#: 600 m pair of 1 m pieces peaked at 292 MiB with all pairs at once).
_BUNDLE_CHUNK = 4096


def _tilted_gate(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
) -> np.ndarray:
    """Mask of the pairs among K pairs of segments that the tilted rule takes.

    Every pair that is neither parallel within the tolerance of the closed
    form (:func:`_bundle_mutual_batch`, $|\\cos\\varphi| \\ge 1 - 10^{-9}$)
    nor orthogonal within the same tolerance ($|\\cos\\varphi| \\le
    10^{-9}$: the mutual of such a pair is at most 1e-9 of its parallel
    value, so the rule would change it by round-off; a T-junction drawn at
    90 deg has $|\\cos\\varphi|$ of a few 1e-16, and on such junctions of
    a generated 80-house network with an inductive PEN the rule changed
    1086 entries by at most 3.3e-23 H). Up to audit W1R3-01 / W1R3-02 of
    the calendar release the rule also needed the lateral distance to
    change by at most $r_a + r_b$ along the longer piece and
    $|\\cos\\varphi| \\ge 0.99$; beyond either
    bound the rule of the single segments acted, so the pieces of one pair
    of conductors could mix two readings (a core crossing the shield axis
    read -8.6e-5 of the loop off the lumped pair in 10 m pieces, two wires
    leaving a junction stepped by 1.5e-3 at 8.1 deg). The classification
    of :func:`_tilted_one_sided` is exact at any angle, so neither bound
    is needed.
    """
    da = p2_a - p1_a
    db = p2_b - p1_b
    la = np.sqrt(_row_dots(da, da))
    lb = np.sqrt(_row_dots(db, db))
    dot = _row_dots(da / la[:, None], db / lb[:, None])
    return (np.abs(np.abs(dot) - 1.0) > 1e-9) & (np.abs(dot) > 1e-9)


def _tilted_one_sided(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    r_thin: np.ndarray,
    r_thick: np.ndarray,
    d_touch: np.ndarray,
    inside_ok: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""One-sided integral of the bundle rule over *a*, K pairs that are tilted.

    *a* is split where the foot of its points on *b*'s axis passes an end
    of the partner's conductor (``e1``, ``e2``) and where the distance
    $\rho(s)$ from that axis crosses $r_a + r_b$ or $r_\text{thick} -
    r_\text{thin}$ (exactly, from the closest approach of the two lines,
    as :func:`_beside_plan`). A stretch is *inside* (``inside_ok`` and
    $\rho + r_\text{thin} \le r_\text{thick}$, at every axial offset),
    *beside* (the foot inside the partner's conductor and $\rho < r_a +
    r_b$) or free. On the inside and beside stretches the effective distance
    $\rho_e$ is constant, and with $u = t_0(s)$ (the foot on *b*'s axis,
    $du = \cos\varphi\,ds$) the integral is exact in closed form:
    $\int_{s_1}^{s_2} [\operatorname{asinh}((\ell_b - t_0)/\rho_e) +
    \operatorname{asinh}(t_0/\rho_e)]\,ds = J(u_1, u_2; 0, \ell_b; \rho_e)
    /\cos\varphi$ (:func:`_stretch_integral`). *b* must point along *a*
    ($\hat u_a \cdot \hat u_b > 0$).

    Returns
    -------
    J, clamped, free : np.ndarray
        ``J`` in m: the integral over the inside and beside stretches (and
        over free stretches shorter than the vertex tolerance, at their
        own distance); ``clamped``: a stretch longer than the tolerance is
        inside or beside; ``free``: a free stretch longer than the
        tolerance exists (its integral is not in ``J``).
    """
    da = a2 - a1
    la = np.sqrt(_row_dots(da, da))
    ua = da / la[:, None]
    db = b2 - b1
    lb = np.sqrt(_row_dots(db, db))
    ub = db / lb[:, None]
    dot = _row_dots(ua, ub)
    w = a1 - b1
    c0 = _row_dots(w, ub)  # foot t0(s) = c0 + s dot
    n = np.cross(ua, ub)
    sin2 = _row_dots(n, n)
    # closest approach of the axes and their distance (as _beside_plan)
    s_star = _row_dots(np.cross(-w, ub), n) / sin2
    delta = np.abs(_row_dots(w, n)) / np.sqrt(sin2)
    te1 = _row_dots(e1 - b1, ub)
    te2 = _row_dots(e2 - b1, ub)
    t_lo = np.minimum(te1, te2)
    t_hi = np.maximum(te1, te2)
    marks = [np.zeros_like(la), la, (t_lo - c0) / dot, (t_hi - c0) / dot]
    for d in (d_touch, r_thick - r_thin):
        crosses = d > delta
        half = np.sqrt(np.where(crosses, (d - delta) * (d + delta), 0.0) / sin2)
        marks += [
            np.where(crosses, s_star - half, 0.0),
            np.where(crosses, s_star + half, 0.0),
        ]
    edges = np.sort(np.clip(np.stack(marks, axis=1), 0.0, la[:, None]), axis=1)
    lo, hi = edges[:, :-1], edges[:, 1:]
    mid = 0.5 * (lo + hi)
    t_mid = c0[:, None] + mid * dot[:, None]
    v = (
        w[:, None, :]
        + mid[:, :, None] * ua[:, None, :]
        - t_mid[:, :, None] * ub[:, None, :]
    )
    rho = np.sqrt(np.einsum("kpi,kpi->kp", v, v))
    inside = inside_ok[:, None] & (rho + r_thin[:, None] <= r_thick[:, None])
    beside = (
        (t_lo[:, None] <= t_mid)
        & (t_mid <= t_hi[:, None])
        & (rho < (d_touch * (1.0 - _BESIDE_MARGIN))[:, None])
    )
    clamped = inside | beside
    longest = np.maximum(la, lb)[:, None]
    long_enough = (hi - lo) > _TOUCH_TOL_REL * longest
    rho_e = np.where(
        inside,
        r_thick[:, None],
        np.where(clamped, d_touch[:, None], np.maximum(rho, 1e-30 * longest)),
    )
    free = ~clamped & long_enough
    J = _stretch_integral(
        c0[:, None] + lo * dot[:, None],
        c0[:, None] + hi * dot[:, None],
        0.0,
        lb[:, None],
        rho_e,
        np.zeros(rho_e.shape, dtype=bool),
    )
    J = np.where(free, 0.0, J).sum(axis=1) / dot
    return J, (clamped & long_enough).any(axis=1), free.any(axis=1)


def _tilted_bundle_batch(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    r_a: np.ndarray,
    shield_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    r_b: np.ndarray,
    shield_b: np.ndarray,
    ext_a: tuple[np.ndarray, np.ndarray],
    ext_b: tuple[np.ndarray, np.ndarray],
    m_fil: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""The bundle rule for K pairs of pieces that are tilted, in H.

    The mutual is $\mu_0/(4\pi)\cos\varphi$ times the mean of the two
    one-sided integrals of the pointwise rule against the partner's whole
    conductor (``ext_a`` / ``ext_b``: the end points of the conductors).
    A side whose stretches are all inside or beside the partner is
    integrated in closed form (:func:`_tilted_one_sided`); a side that
    is entirely free takes the filament value, $J = m_\text{fil} /
    (\mu_0/(4\pi)\cos\varphi)$ with ``m_fil`` the filament mutual of the
    pair (the matrix entry before the rule). A side with both kinds of
    stretch is marked ``exact``: the caller evaluates such pairs with
    :func:`_beside_mutual` and the conductor extents. The value is the same,
    bit for bit, for either order of the two pieces and either direction of
    each (up to the orientation sign).

    Returns
    -------
    values, valid, exact : np.ndarray
        ``values[k]`` in H where ``valid[k]`` (a stretch of either side is
        inside or beside) and not ``exact[k]``.
    """
    da = p2_a - p1_a
    db = p2_b - p1_b
    la = np.sqrt(_row_dots(da, da))
    lb = np.sqrt(_row_dots(db, db))
    dot = _row_dots(da / la[:, None], db / lb[:, None])
    a1, a2 = _lexicographic_ends(p1_a, p2_a)
    b1, b2 = _lexicographic_ends(p1_b, p2_b)
    flip = (_row_dots(a2 - a1, b2 - b1) < 0.0)[:, None]
    r_a = np.asarray(r_a, dtype=float)
    r_b = np.asarray(r_b, dtype=float)
    d_touch = r_a + r_b
    r_thin = np.minimum(r_a, r_b)
    r_thick = np.maximum(r_a, r_b)
    inside_ok = np.where(r_a > r_b, shield_a, shield_b) & (r_thin < r_thick)
    j_a, cl_a, fr_a = _tilted_one_sided(
        a1,
        a2,
        np.where(flip, b2, b1),
        np.where(flip, b1, b2),
        ext_b[0],
        ext_b[1],
        r_thin,
        r_thick,
        d_touch,
        inside_ok,
    )
    j_b, cl_b, fr_b = _tilted_one_sided(
        b1,
        b2,
        np.where(flip, a2, a1),
        np.where(flip, a1, a2),
        ext_a[0],
        ext_a[1],
        r_thin,
        r_thick,
        d_touch,
        inside_ok,
    )
    k = (MU_0 / (4.0 * math.pi)) * dot
    filament = m_fil / k
    j_a = np.where(cl_a, j_a, filament)
    j_b = np.where(cl_b, j_b, filament)
    valid = cl_a | cl_b
    exact = valid & ((cl_a & fr_a) | (cl_b & fr_b))
    return k * (0.5 * (j_a + j_b)), valid, exact


def _tilted_pairs(
    L_dir: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    radii: np.ndarray,
    flags: np.ndarray,
    II: np.ndarray,
    JJ: np.ndarray,
    ext_a: tuple[np.ndarray, np.ndarray],
    ext_b: tuple[np.ndarray, np.ndarray],
    hit_keys: np.ndarray,
) -> None:
    """Evaluate and write the bundle rule for the tilted pairs
    ``(II[k], JJ[k])`` (one chunk; see :func:`_apply_tilted_bundles`).

    Only pairs with a piece that comes within the touching distance of the
    other piece's axis can have a stretch inside or beside the partner;
    the others are skipped before the classification (their entries keep
    the filament value either way, and the evaluation is element-wise, so
    the values are the same bit for bit; audit W1R3-01: without the old
    gates a long pair crossing at a small angle evaluates every pair).
    """
    M = len(radii)
    keys = np.minimum(II, JJ) * M + np.maximum(II, JJ)
    da = p2[II] - p1[II]
    db = p2[JJ] - p1[JJ]
    la = np.sqrt(_row_dots(da, da))
    lb = np.sqrt(_row_dots(db, db))
    reach = (radii[II] + radii[JJ]) * (1.0 + 1e-6) + 1e-9 * np.maximum(la, lb)
    near = (
        _segment_to_lines_distance(p1[II], p2[II], p1[JJ], db / lb[:, None]) <= reach
    ) | (_segment_to_lines_distance(p1[JJ], p2[JJ], p1[II], da / la[:, None]) <= reach)
    near |= np.isin(keys, hit_keys)
    sel = np.flatnonzero(near & _tilted_gate(p1[II], p2[II], p1[JJ], p2[JJ]))
    if sel.size == 0:
        return
    II, JJ = II[sel], JJ[sel]
    ea = (ext_a[0][sel], ext_a[1][sel])
    eb = (ext_b[0][sel], ext_b[1][sel])
    values, valid, exact = _tilted_bundle_batch(
        p1[II],
        p2[II],
        radii[II],
        flags[II],
        p1[JJ],
        p2[JJ],
        radii[JJ],
        flags[JJ],
        ea,
        eb,
        L_dir[II, JJ],
    )
    # pairs on which the pointwise rule acted hold its value, not the
    # filament: they, and sides with free and clamped stretches, are
    # integrated by the pointwise rule against the conductors
    exact |= np.isin(keys[sel], hit_keys)
    for k in np.flatnonzero(exact):
        i, j = int(II[k]), int(JJ[k])
        val = _beside_mutual(
            p1[i],
            p2[i],
            float(radii[i]),
            p1[j],
            p2[j],
            float(radii[j]),
            shield_a=bool(flags[i]),
            shield_b=bool(flags[j]),
            ext_a=(ea[0][k], ea[1][k]),
            ext_b=(eb[0][k], eb[1][k]),
        )
        valid[k] = val is not None
        if val is not None:
            values[k] = val
    L_dir[II[valid], JJ[valid]] = values[valid]
    L_dir[JJ[valid], II[valid]] = values[valid]


def _all_conductor_extents(
    p1: np.ndarray, p2: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """End points of the conductor of every segment, shapes ``(M, 3)``.

    :func:`_conductor_extents` for every label (a segment of a conductor
    that is not straight keeps its own end points).
    """
    e1 = p1.copy()
    e2 = p2.copy()
    order = np.argsort(labels, kind="stable")
    for members in np.split(order, np.flatnonzero(np.diff(labels[order])) + 1):
        if members.size > 1:
            q1, q2 = _conductor_extents(p1, p2, members)
            e1[members] = q1
            e2[members] = q2
    return e1, e2


def _segment_line_distance(
    q1: np.ndarray, q2: np.ndarray, p: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Smallest distance in m of the segments ``q1 -> q2`` (K, 3) from the
    line through ``p`` along the unit vector ``u`` (3,)."""
    r1 = q1 - p
    r1 = r1 - (r1 @ u)[:, None] * u
    r2 = q2 - p
    r2 = r2 - (r2 @ u)[:, None] * u
    dr = r2 - r1
    den = _row_dots(dr, dr)
    t = np.clip(
        np.where(den > 0.0, -_row_dots(r1, dr) / np.where(den > 0.0, den, 1.0), 0.0),
        0.0,
        1.0,
    )
    r = r1 + t[:, None] * dr
    return np.sqrt(_row_dots(r, r))


def _segment_to_lines_distance(
    q1: np.ndarray, q2: np.ndarray, p: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Smallest distance in m of the segments ``q1 -> q2`` from the lines
    through ``p`` along the unit vectors ``u``, one line per segment (all
    shapes ``(K, 3)``)."""
    r1 = q1 - p
    r1 = r1 - _row_dots(r1, u)[:, None] * u
    r2 = q2 - p
    r2 = r2 - _row_dots(r2, u)[:, None] * u
    dr = r2 - r1
    den = _row_dots(dr, dr)
    t = np.clip(
        np.where(den > 0.0, -_row_dots(r1, dr) / np.where(den > 0.0, den, 1.0), 0.0),
        0.0,
        1.0,
    )
    r = r1 + t[:, None] * dr
    return np.sqrt(_row_dots(r, r))


def _apply_tilted_bundles(
    L_dir: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    radii: np.ndarray,
    flags: np.ndarray,
    labels: np.ndarray,
    hits: list[tuple[int, int]],
) -> None:
    """The bundle rule for conductors that are not parallel (audits W1R2-01,
    W1R3-01, W1R3-02).

    The parallel closed form (:func:`_bundle_mutual_batch`) needs the
    pieces parallel within $|\\cos\\varphi| \\ge 1 - 10^{-9}$ (4.5e-5 rad);
    a core tilted by more against its shield kept the per-piece rule, and
    the 100 m coax in 0.25 m pieces read 7.4 % low at 4.5e-5 rad, with a
    step at the tolerance. Here every pair of pieces of two conductors
    lying beside each other that is not parallel (:func:`_tilted_gate`)
    gets the two readings against the partner's whole conductor: the
    inside reading at every axial offset, the beside reading where the
    foot falls inside the partner's conductor. The conductors are those of
    a pair of the pointwise rule (``hits``, other labels, not parallel;
    every pair of their pieces that comes within the touching distance is
    evaluated) and, as for parallel pieces without labels, a cable shield
    with every thinner wire that comes inside its tube (the inside reading
    needs no conductor identity). The classification is exact
    (:func:`_tilted_one_sided`), so the pieces add up to the lumped pair
    whatever the tilt and the length of the pieces: a core crossing its
    shield's axis, wires leaving a junction or crossing at any angle (up to
    audit W1R3 the rule stopped at $\\ell\\sin\\varphi = r_a + r_b$ and at
    $|\\cos\\varphi| = 0.99$, where the per-segment rule took over with
    steps of up to 1.6e-3 of the loop). Pieces of one conductor are not
    touched.
    """
    M = len(radii)
    I = np.array([i for i, _ in hits], dtype=int)
    J = np.array([j for _, j in hits], dtype=int)
    bundles: list[tuple[int, int]] = []
    if I.size:
        keep = (labels[I] != labels[J]) & _tilted_gate(p1[I], p2[I], p1[J], p2[J])
        bundles = sorted(
            {
                (min(a, b), max(a, b))
                for a, b in zip(
                    labels[I[keep]].tolist(), labels[J[keep]].tolist(), strict=True
                )
            }
        )
    if not bundles and not flags.any():
        return
    hit_keys = np.unique(I * M + J)
    e1, e2 = _all_conductor_extents(p1, p2, labels)
    members = {
        c: np.flatnonzero(labels == c) for c in sorted({c for b in bundles for c in b})
    }
    for ca, cb in bundles:
        ia, ib = members[ca], members[cb]
        rows = max(1, _BUNDLE_CHUNK // ib.size)
        for start in range(0, ia.size, rows):
            II = np.repeat(ia[start : start + rows], ib.size)
            JJ = np.tile(ib, II.size // ib.size)
            _tilted_pairs(
                L_dir,
                p1,
                p2,
                radii,
                flags,
                II,
                JJ,
                (e1[II], e2[II]),
                (e1[JJ], e2[JJ]),
                hit_keys,
            )
    if not flags.any():
        return
    # the inside reading of a cable shield needs no conductor identity:
    # every thinner tilted wire that comes inside the tube
    n_labels = int(labels.max()) + 1
    bundle_keys = np.array([a * n_labels + b for a, b in bundles], dtype=int)
    da = p2 - p1
    u = da / np.sqrt(_row_dots(da, da))[:, None]
    for i in np.flatnonzero(flags):
        near = _segment_line_distance(p1, p2, p1[i], u[i])
        cand = np.flatnonzero(
            (radii < radii[i]) & (labels != labels[i]) & (near + radii <= radii[i])
        )
        if cand.size == 0:
            continue
        lo = np.minimum(labels[cand], labels[i])
        hi = np.maximum(labels[cand], labels[i])
        cand = cand[~np.isin(lo * n_labels + hi, bundle_keys)]
        if cand.size == 0:
            continue
        II = np.full(cand.size, i)
        _tilted_pairs(
            L_dir,
            p1,
            p2,
            radii,
            flags,
            II,
            cand,
            (e1[II], e2[II]),
            (e1[cand], e2[cand]),
            hit_keys,
        )


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
    shields: np.ndarray | None = None,
    beside_hits: list[tuple[int, int]] | None = None,
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

    With ``wire_radii``, pairs that lie beside each other closer than the
    sum of their radii (direct or against a mirrored partner) get the
    beside rule (:func:`_beside_mutual`: touching wires, or a wire inside
    a cable shield, ``shields``: one flag per segment, ``None`` for none);
    without radii the filament integral is kept. ``diagonal=False`` leaves
    the diagonal of the direct block at zero (off-diagonal mutuals only, no
    radii needed). ``beside_hits``, if given, collects the pairs
    ``(i, j)``, ``i < j``, of the direct block on which the beside rule
    acted (for :func:`_apply_bundles`).
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
                vals, beside = _beside_batch(
                    p1[i],
                    p2[i],
                    radii[i],
                    p1[slc],
                    p2[slc],
                    radii[slc],
                    shield_a=False if shields is None else bool(shields[i]),
                    shield_b=None if shields is None else shields[slc],
                )
                M_pri[beside] = vals[beside]
                if beside_hits is not None:
                    beside_hits += [(i, i + 1 + int(k)) for k in np.flatnonzero(beside)]
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
                vals, beside = _beside_batch(
                    p1[i],
                    p2[i],
                    radii[i],
                    p1_img[slc],
                    p2_img[slc],
                    radii[slc],
                    shield_a=False if shields is None else bool(shields[i]),
                    shield_b=None if shields is None else shields[slc],
                )
                M_img[beside] = vals[beside]
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
    shields: bool | np.ndarray | None = None,
    conductors: np.ndarray | None = None,
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
        Per-branch wire radii in metres, shape ``(M,)``: the thin-wire
        self terms, and the distance of touching wires $r_i + r_j$ (or
        the shield radius for a wire inside a cable shield, see
        ``shields``) taken where two wires lie beside each other closer
        than that (module docstring, "Wires lying beside each other").
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
    shields
        Segments of cable shields (a bool for all, or an array of shape
        ``(M,)``; ``None``, the default, for none). A thinner wire
        parallel inside a shield ($\\delta + r_\\text{thin} \\le r_s$)
        couples to it at the shield radius, at every axial offset (a core
        inside a cable shield; user decision Q32 of 2026.11); two unmarked
        wires on one axis touch. **The self terms do not depend on it**:
        pass ``include_internal=~shields`` to drop the internal inductance
        of a solid wire from the shields, as the solver does for
        ``conductor_type="cable_shield"`` (Q33). Without it a shield keeps
        $\\mu_0\\ell/(8\\pi)$ = 50 nH/m: the 100 m coax with 5 mm core and
        15 mm shield reads 31.97 µH instead of 26.97 µH.
    conductors
        Conductor label of every segment, shape ``(M,)`` (any values that
        can be compared for equality, e.g. conductor names; ``None``, the
        default: every segment is a conductor of its own). For two
        conductors lying beside each other (a pair of their segments closer
        than touching: on one axis, tilted against each other, crossing,
        leaving one junction), a point lies beside the partner when its
        foot falls inside the partner's whole conductor — the straight run
        of its segments with that label — so the sub-pieces of the two
        conductors add up to the lumped pair (audit W1A-02 of the calendar
        release). For segments that are not parallel the stretches inside,
        beside and away from the partner are found exactly, at any angle and
        for pieces of any length (audits W1R2-01, W1R3-01, W1R3-02: a
        slightly tilted core, a core crossing its shield's axis, two wires
        leaving a junction add up within 3e-12; up to W1R3 the rule stopped
        where the lateral distance changed by more than $r_i + r_j$ along a
        segment and at $|\\cos\\varphi| = 0.99$, and the pieces read up to
        1.6e-3 off the lumped pair). Pieces of one
        conductor and conductors that only meet end to end keep the
        filament integral. **Without labels**, unmarked touching pieces of
        one cable are not recognised as one conductor: they keep the rule
        of the single segments and do not add up (the 100 m pair of 5 and
        15 mm wires reads 40.34 µH in 0.25 m pieces against 43.47 µH
        lumped); the inside reading of a marked shield (``shields``) needs
        no labels and adds up either way (audit W1R2-08). The solver
        passes one label per conductor.

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
    shield = _shield_array(shields, M)
    labels = _conductor_labels(conductors, M)
    hits: list[tuple[int, int]] = []
    L_dir, L_img = _inductance_blocks(
        p1,
        p2,
        radii,
        want_direct=True,
        want_image=mode != "none",
        self_from_rows=mode == "none",
        include_internal=internal,
        touching=touching,
        shields=shield,
        beside_hits=hits,
    )
    _apply_bundles(L_dir, p1, p2, radii, shield, labels, hits)
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
    shields: bool | np.ndarray | None = None,
) -> np.ndarray:
    """Image block $L_\\text{img}$ (mirrored-endpoint image), shape (M, M).

    The block that :func:`build_inductance_matrix` adds for
    ``image="additive"`` and subtracts for ``image="antiparallel"`` —
    computed by the same code path, so that an earth-return correction
    can move between conventions exactly (with ``wire_radii``, a mirrored
    partner lying beside a segment closer than touching gets the beside
    rule, as in :func:`build_inductance_matrix`, with the same
    ``shields``). Raises for segments
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
        shields=_shield_array(shields, M),
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
            if np.array_equal(unit_axes[j], sign * unit_axes[i]):
                u, ref = unit_axes[i], midpoints[i]
            else:
                # Not exactly parallel: project on the bisector of the two
                # axes from the mean of the mid-points, so that the overlap
                # and the distance do not depend on which segment comes
                # first (audit round 3: on the axis of segment i, reversing
                # the segment order moved entries by up to 3.1 % of the
                # largest one, and a pair could get an overlap in one order
                # and none in the other).
                w = unit_axes[i] + sign * unit_axes[j]
                u = w / np.linalg.norm(w)
                ref = 0.5 * (midpoints[i] + midpoints[j])
            axial = float(d_vec @ u)
            perp = d_vec - axial * u
            d_perp_horiz = math.hypot(perp[0], perp[1])
            d_perp_vert = abs(perp[2])
            same_axis = (
                d_perp_horiz < max(radii[i], radii[j], 1e-3) and d_perp_vert < 1e-6
            )
            # Axial overlap of the two segment intervals along u.
            t_i0 = float((seg_endpoints[i, 0] - ref) @ u)
            t_i1 = float((seg_endpoints[i, 1] - ref) @ u)
            lo_i, hi_i = min(t_i0, t_i1), max(t_i0, t_i1)
            t_j0 = float((seg_endpoints[j, 0] - ref) @ u)
            t_j1 = float((seg_endpoints[j, 1] - ref) @ u)
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
            if np.array_equal(unit_axes[j], sign * unit_axes[i]):
                entries.append((i, j, d_perp_horiz, overlap, sign))
                continue
            # A tilted pair: tiled per stretch (audit W1R3-07). Up to 2026.11.0
            # the per-metre value at the mid-point distance was taken for the
            # whole overlap, so a lumped pair and its pieces differed (core
            # crossing its shield's axis, 1 kHz: 4.3e-4 of X at 2e-3 rad,
            # 5.2e-2 at 4e-2 rad). Where the distance changes along the overlap
            # the correction is integrated over it (Gauss-Legendre nodes);
            # pairs parallel up to round-off keep the single entry.
            entries.extend(
                _tilted_stretch_entries(
                    seg_endpoints,
                    unit_axes,
                    midpoints,
                    radii,
                    heights,
                    i,
                    j,
                    u,
                    ref,
                    sign,
                    max(lo_i, lo_j),
                    min(hi_i, hi_j),
                    d_perp_horiz,
                )
            )
    return heights, lengths, horizontal, entries, fallback


#: Relative change of the axis distance along the overlap (of the overlap
#: length) below which a near-parallel pair keeps one Carson entry at its
#: mid-point distance: pairs parallel up to round-off stay bit-identical.
_STRETCH_TOL = 1e-9
#: Gauss-Legendre nodes of a tilted stretch: _STRETCH_MIN_NODES plus
#: _STRETCH_NODES_PER_SCALE per change of the offset by the image distance
#: h_i + h_j, at most _STRETCH_MAX_NODES.
#: A tilted pair whose smallest distance along the overlap exceeds this
#: multiple of the change of its offset keeps the mid-point entry: the error
#: of the mid-point value is then about (change / d)^2 / 24 of the entry,
#: below 5e-5 for a ratio of 30. Measured (stream w3perf): with 30 the
#: generated 100-house star keeps every entry (the stretch rule acts on no
#: pair there; for every pair it would cost 5.5 s of QUADPACK per frequency
#: for 1.8e-6 of X), and a core crossing its shield's axis adds up in pieces
#: within 8.0e-6 of the loop reactance (ratio 10: 1.7e-5, 3: 8.7e-5).
_STRETCH_MIDPOINT_RATIO = 30.0
_STRETCH_MIN_NODES = 4
_STRETCH_NODES_PER_SCALE = 8.0
_STRETCH_MAX_NODES = 32


def _tilted_stretch_entries(
    seg_endpoints,
    unit_axes,
    midpoints,
    radii,
    heights,
    i,
    j,
    u,
    ref,
    sign,
    lo,
    hi,
    d_mid,
) -> list[tuple[int, int, float, float, float]]:
    """Carson entries of a tilted near-parallel pair, tiled per stretch.

    The horizontal axis distance $d(t)$ of the two segments at the position
    $t$ along the bisector ``u`` changes linearly along the overlap
    $[t_\\text{lo}, t_\\text{hi}]$; the per-metre correction is integrated
    over it with Gauss–Legendre nodes, $\\sum_k w_k\\,\\Delta Z'(d(t_k))$
    (4 nodes plus 8 per change of the offset by the image distance
    $h_i + h_j$, at most 32). A pair whose offset changes by less than
    :data:`_STRETCH_TOL` of the overlap, or whose smallest distance along the
    overlap exceeds :data:`_STRETCH_MIDPOINT_RATIO` times that change (the
    mid-point value is then exact to second order), keeps the single entry
    at the mid-point distance ``d_mid``. On one depth two wires are no closer than
    touching, $r_i + r_j$, as for collinear conductors.
    """
    overlap = hi - lo
    e_i, e_j = unit_axes[i], unit_axes[j]
    t_mid_i = float((midpoints[i] - ref) @ u)
    t_mid_j = float((midpoints[j] - ref) @ u)
    c_i, c_j = float(e_i @ u), float(e_j @ u)

    def offset(t: np.ndarray) -> np.ndarray:
        p_i = midpoints[i][None, :] + ((t - t_mid_i) / c_i)[:, None] * e_i[None, :]
        p_j = midpoints[j][None, :] + ((t - t_mid_j) / c_j)[:, None] * e_j[None, :]
        v = p_j - p_i
        return v - (v @ u)[:, None] * u[None, :]

    # change of the horizontal offset vector along the overlap (a pair crossing
    # at mid-length has equal distances at both ends)
    ends = offset(np.array([lo, hi]))
    a, b = ends[0, :2], ends[1, :2] - ends[0, :2]
    change = math.hypot(*b)
    if change <= _STRETCH_TOL * max(overlap, d_mid):
        return [(i, j, d_mid, overlap, sign)]
    # Where the distance stays well above its change along the overlap, d(t)
    # is nearly linear and the mid-point value is exact to second order (a
    # relative error of about (change / d)^2 / 24 of the entry); only pairs
    # that come close to each other within the overlap (a core crossing its
    # shield's axis, wires leaving a junction) need the integral.
    s_min = min(1.0, max(0.0, -float(a @ b) / float(b @ b)))
    if math.hypot(*(a + s_min * b)) > _STRETCH_MIDPOINT_RATIO * change:
        return [(i, j, d_mid, overlap, sign)]
    scale = max(float(heights[i] + heights[j]), 1e-3)
    n = int(
        min(
            _STRETCH_MAX_NODES,
            _STRETCH_MIN_NODES + math.ceil(_STRETCH_NODES_PER_SCALE * change / scale),
        )
    )
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (lo + hi) + 0.5 * overlap * x
    v = offset(t)
    d = np.hypot(v[:, 0], v[:, 1])
    # wires on one depth are no closer than touching (continuous along the
    # stretch, unlike the threshold of the single entry)
    side = np.abs(v[:, 2]) < 1e-6
    d = np.where(side, np.maximum(d, float(radii[i] + radii[j])), d)
    return [
        (i, j, float(d_k), float(0.5 * overlap * w_k), sign)
        for d_k, w_k in zip(d, w, strict=True)
    ]


def _direct_mutual_matrix(
    seg_endpoints: np.ndarray, wire_radii: np.ndarray | None = None
) -> np.ndarray:
    """Off-diagonal free-space Neumann mutuals $L_\\text{dir}$ (zero diagonal).

    The direct block of :func:`build_inductance_matrix` (same code path,
    exact touching pairs, the beside rule for wires closer than touching
    when ``wire_radii`` is given), used to move a correction of a pair on
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
            # += : a tilted pair has one entry per node of its stretch
            dZ[i, j] += sign * dz_per_m * factor
            dZ[j, i] = dZ[i, j]
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
      **axial overlap length** of the two segment intervals — measured
      along the bisector of the two axes for a pair that is not exactly
      parallel, so that the order of the segments does not matter
      (audit round 3 of the calendar release; on the first segment's axis
      the entries moved by up to 3.1 % of the largest one).
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
      the tiling there (two 1 km leads 2 m apart, 1 kHz: 13 %).
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
