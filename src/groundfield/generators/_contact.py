r"""Where generated conductors meet a site (private; user decisions Q41, Q43).

Since user decision Q41 of 2026.11 every conductor that the generators route
to a foundation — a PEN service drop, a cable that leaves a substation or a
cabinet — ends at the point of that foundation nearest to the conductor: the
connection lug on the side facing the cable. Since user decision Q43 the same
holds for every horizontal earth electrode of a site: a circular ring
(:class:`~groundfield.generators.electrode_specs.RingElectrodeSpec`, e.g. the
default substations of both generators) and a strip; rods (vertical) keep
their head. This module computes that point in plan (the horizontal
projection) for the wires of a site:

* :func:`nearest_point` — the point of the wires nearest to a point;
* :func:`closest_pair` — the closest pair between straight segments (the
  sections of a cable, or the wires of another foundation) and the wires:
  the segment, the parameter on it and the point on the wires.

Circles
-------
A circular wire is searched as an inscribed polygon whose sagitta
:math:`r\,(1 - \cos(\pi/n))` is at most :data:`CIRCLE_SAGITTA_M` (vertices
at the four axis points, so its bounding box is the circle's), and the point
found is refined to the exact closest point of the circle: for a point
:math:`P` the radial point :math:`C + r\,(P - C)/|P - C|`; for a straight
segment the first intersection with the circle (smallest arc position) if it
crosses or touches it, else the radial point through the segment point
nearest to the centre (outside) or through the end point farthest from it
(inside), over every segment of the search. The choice between two wires of
one site that lie within the sagitta of the same distance is made on the
polygon.

Geometry
--------
For two straight segments :math:`P(s) = A + s\,d_1` and
:math:`Q(u) = B + u\,d_2` (:math:`s, u \in [0, 1]`) the plan distance
:math:`f(u) = \min_s |P(s) - Q(u)|` is convex in :math:`u`. If the segments
cross or touch, its minimum is 0 at the crossing (for collinear, overlapping
segments at the first point of the overlap). Otherwise the minimum is reached
with an end point of one of the segments involved, so it lies among
:math:`u \in \{0,\ 1,\ u(A),\ u(A + d_1)\}`, the last two being the clamped
projections of the end points of :math:`P` onto :math:`Q`; that set also
contains the first point of a range of equal distance (parallel segments).

Ties
----
Points within :data:`TIE_TOLERANCE_M` of the smallest distance tie; among
them the point on the wire with the lowest index wins, then the one with the
smallest arc position along that wire (measured from the wire's connection
point, the first vertex), then the first segment. So a cable parallel to a
side of a rectangle meets it at the start of the range of equal distance —
for a side that the cable passes along its whole length, the corner at which
the wire starts. The result is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CIRCLE_SAGITTA_M",
    "TIE_TOLERANCE_M",
    "Contact",
    "FoundationWire",
    "Outline",
    "PreparedWires",
    "circle_polygon",
    "closest_pair",
    "convex_outline",
    "inside_outlines",
    "join",
    "nearest_point",
    "wire_pieces",
]

TIE_TOLERANCE_M = 1e-9
"""Distances (m) closer than this to the smallest one tie."""

CIRCLE_SAGITTA_M = 1e-3
"""Largest sagitta (m) of the polygon a circular wire is searched on."""

_PARALLEL = 1e-12
"""Relative cross product below which two segments count as parallel."""


@dataclass(frozen=True)
class FoundationWire:
    """One horizontal wire of a site, in plan.

    The wires of a site's foundations (user decision Q41) and, since user
    decision Q43, of its other horizontal earth electrodes: circular rings
    and strips.

    Attributes
    ----------
    electrode
        Name of the electrode (wire) in the world.
    index
        Position of the wire among the horizontal wires of the site (world
        order); the lower index wins a tie.
    xy
        Plan vertices ``(x, y)`` in m from the wire's connection point (its
        first vertex), closed loops with the first vertex repeated at the end.
        For a circle the inscribed polygon of :func:`circle_polygon`.
    z
        Depth of the wire in m (positive into the soil).
    circle
        ``(x, y, r)`` in m — centre and radius of a circular wire, whose
        points are refined to the exact circle (module docstring); ``None``
        for a straight or polygonal wire.
    """

    electrode: str
    index: int
    xy: tuple[tuple[float, float], ...]
    z: float
    circle: tuple[float, float, float] | None = None


def circle_polygon(
    cx: float, cy: float, radius: float
) -> tuple[tuple[float, float], ...]:
    """Inscribed polygon of a circle for the closest-point search.

    Parameters
    ----------
    cx, cy
        Centre in m.
    radius
        Radius in m.

    Returns
    -------
    tuple of (x, y)
        ``n + 1`` vertices in m from the circle's connection point
        :math:`(c_x + r, c_y)` counter-clockwise, the first repeated at the
        end; ``n`` a multiple of 4, at least 64, with a sagitta of at most
        :data:`CIRCLE_SAGITTA_M`. The four axis points are exact vertices.
    """
    r = float(radius)
    cos_step = 1.0 - CIRCLE_SAGITTA_M / r
    n = 64 if cos_step <= -1.0 else math.ceil(math.pi / math.acos(cos_step))
    n = max(64, 4 * math.ceil(n / 4))
    quarter = n // 4
    exact = {0: (1.0, 0.0), quarter: (0.0, 1.0), 2 * quarter: (-1.0, 0.0)}
    exact[3 * quarter] = (0.0, -1.0)
    out = []
    for k in range(n + 1):
        c, s = exact.get(k % n) or (
            math.cos(2.0 * math.pi * k / n),
            math.sin(2.0 * math.pi * k / n),
        )
        out.append((cx + r * c, cy + r * s))
    return tuple(out)


def _arc_on_circle(
    xy: tuple[float, float], circle: tuple[float, float, float]
) -> float:
    """Arc position in m of a point of a circle, counter-clockwise from +x."""
    cx, cy, r = circle
    angle = math.atan2(xy[1] - cy, xy[0] - cx)
    if angle < 0.0:
        angle += 2.0 * math.pi
    return r * angle


def _radial(
    point: Sequence[float], circle: tuple[float, float, float]
) -> tuple[float, float]:
    """Point of the circle on the ray from its centre through ``point``
    (the connection point when ``point`` is the centre)."""
    cx, cy, r = circle
    dx, dy = float(point[0]) - cx, float(point[1]) - cy
    d = math.hypot(dx, dy)
    if d == 0.0:
        return (cx + r, cy)
    return (cx + r * (dx / d), cy + r * (dy / d))


def _segment_circle(
    a: Sequence[float], b: Sequence[float], circle: tuple[float, float, float]
) -> tuple[float, float, tuple[float, float]]:
    """Exact closest pair of the segment ``a -> b`` and a circle.

    Returns ``(distance, t, xy)``: the plan distance in m, the parameter of
    the pair's point on the segment and the pair's point on the circle. A
    segment that crosses or touches the circle meets it at the intersection
    with the smallest arc position (distance 0); one inside it at the radial
    point through its end farther from the centre; one outside it at the
    radial point through its point nearest to the centre.
    """
    cx, cy, r = circle
    ax, ay = float(a[0]) - cx, float(a[1]) - cy
    dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    dd = dx * dx + dy * dy
    s0 = 0.0 if dd == 0.0 else min(max(-(ax * dx + ay * dy) / dd, 0.0), 1.0)
    g0 = math.hypot(ax + s0 * dx, ay + s0 * dy)
    ga, gb = math.hypot(ax, ay), math.hypot(ax + dx, ay + dy)
    gmax = max(ga, gb)
    if g0 > r:
        xy = _radial((cx + ax + s0 * dx, cy + ay + s0 * dy), circle)
        return g0 - r, s0, xy
    if gmax < r:
        # Inside: the end point farther from the centre (ties: smaller arc).
        ends = []
        for dist, t in ((r - ga, 0.0), (r - gb, 1.0)):
            xy = _radial((cx + ax + t * dx, cy + ay + t * dy), circle)
            ends.append((dist, _arc_on_circle(xy, circle), t, xy))
        low = min(e[0] for e in ends)
        ends = [e for e in ends if e[0] <= low + TIE_TOLERANCE_M]
        ends.sort(key=lambda e: (e[1], e[2]))
        dist, _, t, xy = ends[0]
        return dist, t, xy
    # Crossing or touching: the roots of |w + s d| = r within [0, 1].
    wd = ax * dx + ay * dy
    ww = ax * ax + ay * ay
    disc = max(wd * wd - dd * (ww - r * r), 0.0)
    root = math.sqrt(disc)
    roots = [] if dd == 0.0 else [(-wd - root) / dd, (-wd + root) / dd]
    cand = []
    for s in roots:
        s = min(max(s, 0.0), 1.0)
        xy = _radial((cx + ax + s * dx, cy + ay + s * dy), circle)
        gap = math.hypot(cx + ax + s * dx - xy[0], cy + ay + s * dy - xy[1])
        cand.append((gap, _arc_on_circle(xy, circle), s, xy))
    if not cand:  # a point segment on the circle
        xy = _radial((cx + ax, cy + ay), circle)
        return abs(ga - r), 0.0, xy
    low = min(c[0] for c in cand)
    cand = [c for c in cand if c[0] <= low + TIE_TOLERANCE_M]
    cand.sort(key=lambda c: (c[1], c[2]))
    gap, _, s, xy = cand[0]
    return gap, s, xy


@dataclass(frozen=True)
class Contact:
    """Point of a foundation where a conductor ends.

    Attributes
    ----------
    xy
        Plan position in m.
    electrode
        Name of the wire that holds the point.
    distance
        Plan distance in m from the point to the conductor's other geometry
        (the point or segment it was paired with).
    z
        Depth of the wire in m.
    """

    xy: tuple[float, float]
    electrode: str
    distance: float
    z: float


class PreparedWires(tuple):
    """The straight pieces of a foundation's wires, computed once.

    :func:`_wire_segments` of ``wires`` (starts, directions, wire indices,
    arc offsets, lengths, owners) and the plan bounding box of the wires;
    :func:`closest_pair` and :func:`nearest_point` accept it in place of
    the wires (the PEN router prepares every building once, audit R4A-06).
    """

    __slots__ = ()

    @classmethod
    def of(cls, wires: Sequence[FoundationWire]) -> "PreparedWires":
        """Prepare ``wires`` (a :class:`PreparedWires` stays as it is)."""
        if isinstance(wires, PreparedWires):
            return wires
        parts = _wire_segments(wires)
        a, d = parts[0], parts[1]
        ends = np.concatenate([a, a + d])
        box = (*ends.min(axis=0), *ends.max(axis=0))
        return cls((*parts, tuple(float(v) for v in box), tuple(wires)))

    @property
    def wires(self) -> tuple[FoundationWire, ...]:
        """The wires themselves."""
        return self[7]


def _wire_segments(
    wires: Sequence[FoundationWire],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[FoundationWire]
]:
    """Segments of every wire: start, direction, wire index, arc offset, length."""
    if isinstance(wires, PreparedWires):
        return wires[:6]  # type: ignore[return-value]
    starts, dirs, index, arc0, length, owner = [], [], [], [], [], []
    for wire in wires:
        pts = np.asarray(wire.xy, dtype=float).reshape(-1, 2)
        arc = 0.0
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            d = b - a
            seg = float(math.hypot(d[0], d[1]))
            if seg == 0.0:
                continue
            starts.append(a)
            dirs.append(d)
            index.append(wire.index)
            arc0.append(arc)
            length.append(seg)
            owner.append(wire)
            arc += seg
    if not starts:
        raise ValueError("A foundation without wires has no point to connect to.")
    return (
        np.asarray(starts),
        np.asarray(dirs),
        np.asarray(index, dtype=float),
        np.asarray(arc0),
        np.asarray(length),
        owner,
    )


def _point_segment_distance(
    p: np.ndarray, a: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from points ``p`` to segments ``a + s d`` and the clamped ``s``.

    All arrays broadcast; the last axis holds ``(x, y)``.
    """
    dd = np.sum(d * d, axis=-1)
    safe = np.where(dd > 0.0, dd, 1.0)
    s = np.where(dd > 0.0, np.sum((p - a) * d, axis=-1) / safe, 0.0)
    s = np.clip(s, 0.0, 1.0)
    q = a + s[..., None] * d
    return np.sqrt(np.sum((p - q) ** 2, axis=-1)), s


def _pick(
    distance: np.ndarray, index: np.ndarray, arc: np.ndarray, order: np.ndarray
) -> int:
    """Flat position of the winner: smallest distance, then index, arc, order."""
    best = float(np.min(distance))
    tied = np.flatnonzero(distance <= best + TIE_TOLERANCE_M)
    keys = np.lexsort((order[tied], arc[tied], index[tied]))
    return int(tied[keys[0]])


def nearest_point(point: Sequence[float], wires: Sequence[FoundationWire]) -> Contact:
    """Point of ``wires`` nearest to ``point`` in plan (ties: see the module).

    Parameters
    ----------
    point
        Plan position ``(x, y)`` in m.
    wires
        The wires of a foundation.

    Returns
    -------
    Contact
        The nearest point, its wire and the plan distance in m.
    """
    a, d, index, arc0, length, owner = _wire_segments(wires)
    p = np.asarray(point, dtype=float)[:2]
    dist, u = _point_segment_distance(p[None, :], a, d)
    k = _pick(dist, index, arc0 + u * length, np.arange(len(dist), dtype=float))
    circle = owner[k].circle
    if circle is not None:
        # The exact point of the circle (user decision Q43).
        cx, cy, r = circle
        return Contact(
            xy=_radial(p, circle),
            electrode=owner[k].electrode,
            distance=abs(math.hypot(float(p[0]) - cx, float(p[1]) - cy) - r),
            z=float(owner[k].z),
        )
    xy = a[k] + u[k] * d[k]
    return Contact(
        xy=(float(xy[0]), float(xy[1])),
        electrode=owner[k].electrode,
        distance=float(dist[k]),
        z=float(owner[k].z),
    )


def closest_pair(
    segments: Sequence[tuple[Sequence[float], Sequence[float]]],
    wires: Sequence[FoundationWire],
) -> tuple[int, float, Contact]:
    """Closest pair in plan between straight ``segments`` and ``wires``.

    Parameters
    ----------
    segments
        Straight segments ``((x0, y0), (x1, y1))`` in m — the sections of a
        cable path, in path order.
    wires
        The wires of a foundation.

    Returns
    -------
    index : int
        Position in ``segments`` of the segment that holds the pair.
    t : float
        Parameter of the pair's point on that segment (0 at its start, 1 at
        its end).
    contact : Contact
        The pair's point on the wires, with the plan distance in m between
        the two points of the pair.

    Notes
    -----
    Exact for straight segments (see the module docstring); ties go to the
    lowest wire index, then to the smallest arc position along the wire,
    then to the first segment. The point on ``segments`` is the point of the
    chosen segment nearest to the contact.
    """
    if not segments:
        raise ValueError("closest_pair needs at least one segment.")
    prepared = PreparedWires.of(wires)
    a, d, index, arc0, length, owner = _wire_segments(prepared)
    seg_all = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    keep = _candidate_segments(seg_all, prepared)
    seg = seg_all[keep]
    pa = seg[:, 0, :][:, None, :]  # (n, 1, 2) cable segment start
    pd = (seg[:, 1, :] - seg[:, 0, :])[:, None, :]  # (n, 1, 2) direction
    wa = a[None, :, :]  # (1, m, 2) wire segment start
    wd = d[None, :, :]
    n, m = seg.shape[0], a.shape[0]

    # Candidate parameters u on each wire segment (module docstring).
    dd = np.sum(wd * wd, axis=-1)
    u_a = np.clip(np.sum((pa - wa) * wd, axis=-1) / dd, 0.0, 1.0)
    u_b = np.clip(np.sum((pa + pd - wa) * wd, axis=-1) / dd, 0.0, 1.0)
    cands = [np.zeros((n, m)), np.ones((n, m)), u_a, u_b]
    # Crossing (or touching) segments: distance 0 at the crossing.
    cross = pd[..., 0] * wd[..., 1] - pd[..., 1] * wd[..., 0]
    scale = np.sqrt(np.sum(pd * pd, axis=-1) * dd)
    rel = wa - pa
    proper = np.abs(cross) > _PARALLEL * np.maximum(scale, 1e-300)
    safe = np.where(proper, cross, 1.0)
    u_x = (rel[..., 0] * pd[..., 1] - rel[..., 1] * pd[..., 0]) / safe
    # Where the lines cross, Q(u_x) lies on the line of P; it is a point of
    # P (distance 0) when the segments cross, and otherwise just one more
    # point of the wire whose distance is evaluated like the others.
    cands.append(np.where(proper, np.clip(u_x, 0.0, 1.0), 0.0))
    u = np.stack(cands, axis=-1)  # (n, m, 5)
    q = wa[..., None, :] + u[..., None] * wd[..., None, :]  # (n, m, 5, 2)
    dist, _ = _point_segment_distance(q, pa[..., None, :], pd[..., None, :])
    # Collinear overlap: the u range of the overlap starts at a candidate
    # already (0, u_a or u_b); its distance is 0 up to round-off.
    arc = arc0[None, :, None] + u * length[None, :, None]
    idx = np.broadcast_to(index[None, :, None], u.shape)
    order = np.broadcast_to(keep.astype(float)[:, None, None], u.shape)
    flat = _pick(dist.ravel(), idx.ravel(), arc.ravel(), order.ravel())
    i, j, c = np.unravel_index(flat, u.shape)
    circle = owner[j].circle
    if circle is not None:
        # The exact closest pair of the circle and the segments (user
        # decision Q43): the polygon chose the wire; every segment of the
        # search is paired with the circle itself. Ties: smallest arc
        # position on the circle, then the first segment.
        pairs = []
        for k in range(seg.shape[0]):
            gap, t_k, xy_k = _segment_circle(seg[k, 0], seg[k, 1], circle)
            pairs.append((gap, _arc_on_circle(xy_k, circle), k, t_k, xy_k))
        low = min(p[0] for p in pairs)
        gap, _, k, t_k, xy_k = min(
            (p for p in pairs if p[0] <= low + TIE_TOLERANCE_M),
            key=lambda p: (p[1], p[2]),
        )
        contact = Contact(
            xy=(float(xy_k[0]), float(xy_k[1])),
            electrode=owner[j].electrode,
            distance=float(gap),
            z=float(owner[j].z),
        )
        return int(keep[k]), float(t_k), contact
    xy = q[i, j, c]
    # The point of segment i nearest to the contact.
    _, t = _point_segment_distance(
        xy[None, :], seg[i, 0][None, :], (seg[i, 1] - seg[i, 0])[None, :]
    )
    contact = Contact(
        xy=(float(xy[0]), float(xy[1])),
        electrode=owner[j].electrode,
        distance=float(dist[i, j, c]),
        z=float(owner[j].z),
    )
    return int(keep[i]), float(t[0]), contact


def _candidate_segments(seg: np.ndarray, wires: "PreparedWires") -> np.ndarray:
    """Indices of the segments that can hold the closest pair (or a tie).

    A segment whose bounding box is farther from the wires' bounding box
    than an upper bound of the closest distance (the distance from the
    wires' first vertex to the nearest segment), plus the tie tolerance,
    holds no candidate of :func:`closest_pair`'s pick: leaving it out
    changes nothing, and the candidates evaluated keep their values and
    their order keys (audit R4A-06).
    """
    n = seg.shape[0]
    if n <= 8:
        return np.arange(n)
    x0, y0, x1, y1 = wires[6]
    first = wires[0][0]
    dist, _ = _point_segment_distance(
        first[None, :], seg[:, 0, :], seg[:, 1, :] - seg[:, 0, :]
    )
    bound = float(np.min(dist)) + TIE_TOLERANCE_M
    lo = np.minimum(seg[:, 0, :], seg[:, 1, :])
    hi = np.maximum(seg[:, 0, :], seg[:, 1, :])
    gx = np.maximum(0.0, np.maximum(lo[:, 0] - x1, x0 - hi[:, 0]))
    gy = np.maximum(0.0, np.maximum(lo[:, 1] - y1, y0 - hi[:, 1]))
    # Margin: round-off of the gap and of the distances (relative 1e-12).
    scale = 1e-12 * (1.0 + float(np.max(np.abs(seg))) + abs(x0) + abs(y1))
    return np.flatnonzero(np.hypot(gx, gy) <= bound + scale)


def wire_pieces(
    wires: Sequence[FoundationWire],
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[str]]:
    """Straight pieces of ``wires`` in wire order, with the wire of each."""
    segments, owners = _pieces(wires)
    return segments, [wire.electrode for wire in owners]


def _pieces(
    wires: Sequence[FoundationWire],
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[FoundationWire]]:
    """Straight pieces of ``wires`` in wire order, with the wire object of each."""
    if isinstance(wires, PreparedWires):
        wires = wires.wires
    segments, owners = [], []
    for wire in wires:
        for a, b in zip(wire.xy[:-1], wire.xy[1:], strict=True):
            if a != b:
                segments.append((a, b))
                owners.append(wire)
    return segments, owners


def join(
    a_xy: Sequence[float],
    a_wires: Sequence[FoundationWire] | None,
    b_xy: Sequence[float],
    b_wires: Sequence[FoundationWire] | None,
) -> tuple[
    tuple[tuple[float, float], str | None], tuple[tuple[float, float], str | None]
]:
    """End points in plan of a straight conductor between two nodes.

    Parameters
    ----------
    a_xy, b_xy
        Plan connection points ``(x, y)`` in m of the two nodes.
    a_wires, b_wires
        Foundation wires of the nodes, or ``None`` (empty) for a node
        without a foundation.

    Returns
    -------
    tuple
        ``((xy_a, wire_a), (xy_b, wire_b))``: a node without horizontal
        wires is met at its connection point (wire ``None``), a node with
        them at the point of its wires nearest to the other end, two such
        nodes at their closest pair (user decisions Q41, Q43; circles
        exactly, two disjoint circles on the line of their centres).
    """
    pa = (float(a_xy[0]), float(a_xy[1]))
    pb = (float(b_xy[0]), float(b_xy[1]))
    if not a_wires and not b_wires:
        return (pa, None), (pb, None)
    if not a_wires:
        cb = nearest_point(pa, b_wires)  # type: ignore[arg-type]
        return (pa, None), (cb.xy, cb.electrode)
    if not b_wires:
        ca = nearest_point(pb, a_wires)
        return (ca.xy, ca.electrode), (pb, None)
    segments, owners = _pieces(a_wires)
    i, t, cb = closest_pair(segments, b_wires)
    (x0, y0), (x1, y1) = segments[i]
    xy = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    a_wire = owners[i]
    if a_wire.circle is None:
        return (xy, a_wire.electrode), (cb.xy, cb.electrode)
    # A circle at end a (user decision Q43): pair the circle itself with the
    # wire of b that the polygon search chose.
    b_list = b_wires.wires if isinstance(b_wires, PreparedWires) else b_wires
    b_wire = next(w for w in b_list if w.electrode == cb.electrode)
    if b_wire.circle is not None:
        (xa, ya, ra), (xb, yb, rb) = a_wire.circle, b_wire.circle
        gap = math.hypot(xb - xa, yb - ya)
        if gap > ra + rb:
            ux, uy = (xb - xa) / gap, (yb - ya) / gap
            pa = (xa + ra * ux, ya + ra * uy)
            pb = (xb - rb * ux, yb - rb * uy)
        else:  # nested or overlapping circles: the polygon pair, on the circles
            pa, pb = _radial(xy, a_wire.circle), _radial(cb.xy, b_wire.circle)
        return (pa, a_wire.electrode), (pb, b_wire.electrode)
    b_segments, _ = _pieces([b_wire])
    k, tb, ca = closest_pair(b_segments, [a_wire])
    (x0, y0), (x1, y1) = b_segments[k]
    xb = (x0 + tb * (x1 - x0), y0 + tb * (y1 - y0))
    return (ca.xy, a_wire.electrode), (xb, b_wire.electrode)


@dataclass(frozen=True)
class Outline:
    """Closed plan outline of a horizontal electrode: the area it encloses.

    Attributes
    ----------
    xy
        Closed polygon ``(x, y)`` in m, the first vertex repeated at the end
        (a rectangle's four corners, a building outline, the polygon of
        :func:`circle_polygon` for a circle).
    circle
        ``(x, y, r)`` in m of a circle, tested exactly; ``None`` for a
        polygon.
    """

    xy: tuple[tuple[float, float], ...]
    circle: tuple[float, float, float] | None = None


def inside_outlines(
    point: Sequence[float], outlines: Sequence[Outline], tol: float = 1e-9
) -> bool:
    """Whether ``point`` lies strictly inside one of ``outlines`` (plan).

    A point within ``tol`` (m) of an outline lies on it, not inside. Circles
    are tested exactly, polygons by the even-odd rule.
    """
    x, y = float(point[0]), float(point[1])
    for outline in outlines:
        if outline.circle is not None:
            cx, cy, r = outline.circle
            if math.hypot(x - cx, y - cy) < r - tol:
                return True
            continue
        pts = outline.xy
        on_edge = False
        flag = False
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:], strict=True):
            dx, dy = bx - ax, by - ay
            dd = dx * dx + dy * dy
            t = 0.0 if dd == 0.0 else ((x - ax) * dx + (y - ay) * dy) / dd
            t = min(max(t, 0.0), 1.0)
            if math.hypot(x - ax - t * dx, y - ay - t * dy) <= tol:
                on_edge = True
                break
            if (ay > y) != (by > y) and x < ax + (y - ay) * dx / dy:
                flag = not flag
        if flag and not on_edge:
            return True
    return False


def convex_outline(
    points: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    """Closed convex hull (counter-clockwise, monotone chain) of plan points.

    The outline of a rectangular foundation from the vertices of its wires
    (every vertex lies on the rectangle or inside it).
    """
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) < 3:
        return tuple(pts)

    def half(seq: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in seq:
            while (
                len(out) >= 2
                and (
                    (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                    - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])
                )
                <= 0.0
            ):
                out.pop()
            out.append(p)
        return out[:-1]

    hull = half(pts) + half(pts[::-1])
    return tuple(hull + hull[:1])
