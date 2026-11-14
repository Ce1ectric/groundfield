r"""Where generated conductors meet a foundation (private; user decision Q41).

Since user decision Q41 of 2026.11 every conductor that the generators route
to a foundation — a PEN service drop, a cable that leaves a substation or a
cabinet — ends at the point of that foundation nearest to the conductor: the
connection lug on the side facing the cable. This module computes that point
in plan (the horizontal projection) for the wires of a foundation:

* :func:`nearest_point` — the point of the wires nearest to a point;
* :func:`closest_pair` — the closest pair between straight segments (the
  sections of a cable, or the wires of another foundation) and the wires:
  the segment, the parameter on it and the point on the wires.

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
    "TIE_TOLERANCE_M",
    "Contact",
    "FoundationWire",
    "closest_pair",
    "join",
    "nearest_point",
    "wire_pieces",
]

TIE_TOLERANCE_M = 1e-9
"""Distances (m) closer than this to the smallest one tie."""

_PARALLEL = 1e-12
"""Relative cross product below which two segments count as parallel."""


@dataclass(frozen=True)
class FoundationWire:
    """One wire of a foundation, in plan.

    Attributes
    ----------
    electrode
        Name of the electrode (wire) in the world.
    index
        Position of the wire among the wires of the site's foundations (world
        order); the lower index wins a tie.
    xy
        Plan vertices ``(x, y)`` in m from the wire's connection point (its
        first vertex), closed loops with the first vertex repeated at the end.
    z
        Depth of the wire in m (positive into the soil).
    """

    electrode: str
    index: int
    xy: tuple[tuple[float, float], ...]
    z: float


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


def _wire_segments(
    wires: Sequence[FoundationWire],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[FoundationWire]
]:
    """Segments of every wire: start, direction, wire index, arc offset, length."""
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
    a, d, index, arc0, length, owner = _wire_segments(wires)
    seg = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
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
    order = np.broadcast_to(np.arange(n, dtype=float)[:, None, None], u.shape)
    flat = _pick(dist.ravel(), idx.ravel(), arc.ravel(), order.ravel())
    i, j, c = np.unravel_index(flat, u.shape)
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
    return int(i), float(t[0]), contact


def wire_pieces(
    wires: Sequence[FoundationWire],
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[str]]:
    """Straight pieces of ``wires`` in wire order, with the wire of each."""
    segments, owners = [], []
    for wire in wires:
        for a, b in zip(wire.xy[:-1], wire.xy[1:], strict=True):
            if a != b:
                segments.append((a, b))
                owners.append(wire.electrode)
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
        ``((xy_a, wire_a), (xy_b, wire_b))``: a node without a foundation is
        met at its connection point (wire ``None``), a node with one at the
        point of its wires nearest to the other end, two foundations at
        their closest pair (user decision Q41).
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
    segments, owners = wire_pieces(a_wires)
    i, t, cb = closest_pair(segments, b_wires)
    (x0, y0), (x1, y1) = segments[i]
    xy = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return (xy, owners[i]), (cb.xy, cb.electrode)
