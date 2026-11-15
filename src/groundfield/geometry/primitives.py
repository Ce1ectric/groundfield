"""Geometric primitives for grounding systems.

Straight wire segments are the common currency between the electrode
models of :mod:`groundfield.geometry.electrodes` and the solvers: every
electrode discretises itself into a :class:`WireSegments` container
(``Electrode.segments(ds)``), and every geometric query that used to be
an ``isinstance`` ladder over the electrode kinds (wire length, segment
count, bounding box, smallest feature, drawable polylines) is a method
of the electrode model (review pass 10, 10.79).

The discretisation conventions live here and in the electrode methods
only, so the solver, the pre-flight diagnostics and the plots can no
longer disagree about them:

* every straight wire of length :math:`\\ell` is cut into
  :math:`n = \\max(1, \\lceil \\ell / \\Delta s - 10^{-9} \\rceil)` equal
  segments (the tolerance keeps floating-point noise from adding one);
* a wire whose depth range crosses a soil-layer interface is first split
  at the interface (ADR-0007 step 1), so no segment straddles it;
* grid wires use a segment count that is a multiple of the number of
  mesh spans (:func:`_segments_per_wire`), so wire crossings fall on
  segment end points rather than on the midpoint point sources;
* a ring is cut into :math:`\\max(8, \\lceil 2\\pi r / \\Delta s \\rceil)`
  equal arc segments represented by their chords' midpoints on the arc;
* every segment records the straight wire it lies on
  (:attr:`WireSegments.wires`): a full wire line of a grid, an arm of a
  star, a straight run of a polyline (its edges between two bends; a
  vertex where the polyline continues straight on does not start a new
  wire); a rod, a strip and a ring are one wire. The image family solves
  one uniform current per wire (user decision Q30 of 2026.11).

Coordinates are in metres with the :math:`z` axis pointing down into
the soil (depth positive).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

__all__ = ["WireSegments"]


@dataclass(frozen=True)
class WireSegments:
    """Straight segments of a discretised wire geometry.

    Attributes
    ----------
    midpoints
        Segment midpoints ``(x, y, z)`` in m, shape ``(N, 3)``. The
        solvers place the point source (point kernel) or the centre of
        the line source (line kernel) here.
    lengths
        Segment lengths in m, shape ``(N,)``. For a ring this is the arc
        length represented by the segment.
    directions
        Unit tangents of the segment axes, shape ``(N, 3)``.
    wires
        Index of the straight wire every segment lies on, shape ``(N,)``
        (integers from 0), or ``None`` for one wire (all segments on wire
        0). A grid numbers its full wire lines (longitudinal ones first),
        a star its arms, a polyline its straight runs (consecutive edges
        that continue in the same direction are one wire); a rod, a strip
        and a ring are one wire. The image family gives
        every wire of an electrode its own uniform current (user decision
        Q30 of 2026.11), see :attr:`wire_index`.

    Notes
    -----
    The end points follow as ``midpoints ∓ lengths/2 · directions``
    (:attr:`starts`, :attr:`ends`). For a ring they are the end points
    of the tangent chord through the arc midpoint, not points on the
    circle.
    """

    midpoints: np.ndarray
    lengths: np.ndarray
    directions: np.ndarray
    wires: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def wire_index(self) -> np.ndarray:
        """Wire index per segment, shape ``(N,)`` (zeros for one wire)."""
        if self.wires is None:
            return np.zeros(len(self), dtype=int)
        return np.asarray(self.wires, dtype=int)

    @property
    def n_wires(self) -> int:
        """Number of distinct straight wires (0 without segments)."""
        if not len(self):
            return 0
        return int(np.unique(self.wire_index).size)

    @property
    def starts(self) -> np.ndarray:
        """Segment start points in m, shape ``(N, 3)``."""
        return self.midpoints - 0.5 * self.lengths[:, None] * self.directions

    @property
    def ends(self) -> np.ndarray:
        """Segment end points in m, shape ``(N, 3)``."""
        return self.midpoints + 0.5 * self.lengths[:, None] * self.directions

    @property
    def total_length(self) -> float:
        """Sum of the segment lengths in m."""
        return float(self.lengths.sum())

    @classmethod
    def empty(cls) -> WireSegments:
        """Container without segments."""
        return cls(np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)))

    @classmethod
    def concatenate(cls, parts: Iterable[WireSegments]) -> WireSegments:
        """Join several containers in order.

        The wire indices are kept as they are (no offset): the parts of
        one wire cut at a layer interface stay one wire. A caller that
        joins different wires numbers them itself (:func:`_edge_wires`).
        """
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        wires = None
        if any(p.wires is not None for p in parts):
            wires = np.concatenate([p.wire_index for p in parts])
        return cls(
            np.concatenate([p.midpoints for p in parts], axis=0),
            np.concatenate([p.lengths for p in parts]),
            np.concatenate([p.directions for p in parts], axis=0),
            wires,
        )


#: Tolerance of the segment count on ``length / ds``: a wire that exceeds a
#: whole multiple of ``ds`` only through floating-point noise (an edge of
#: 3.0000000000000004 m at ds = 1.5 m) gets no extra segment. GEN-1 found 52
#: instead of 50 segments on footprint polygons (review pass 10, 10.32).
#: For a length computed from coordinates the tolerance grows with their
#: round-off (:func:`_segment_count`, ``scale``).
SEGMENT_COUNT_TOL = 1e-9

#: Origin ``(x0, y0, 0)`` in m of the local frame a solver discretises in
#: (:mod:`groundfield.solver._frame`), or ``None``: the round-off tolerances
#: of this module are evaluated with the coordinates of the user's frame,
#: ``p + origin``, so a world shifted into its local frame is discretised
#: with the same decisions as the original.
_FRAME_ORIGIN: ContextVar[np.ndarray | None] = ContextVar(
    "groundfield_frame_origin", default=None
)


@contextmanager
def _frame_origin(origin: np.ndarray) -> Iterator[None]:
    """Evaluate the round-off tolerances as if ``origin`` (m) were added back."""
    token = _FRAME_ORIGIN.set(np.asarray(origin, dtype=float))
    try:
        yield
    finally:
        _FRAME_ORIGIN.reset(token)


def _coordinate_scale(*points: Sequence[float] | np.ndarray) -> float:
    """Largest coordinate magnitude in m of ``points`` in the user's frame.

    The magnitude that sets the round-off of the points: ``max |p_i|``, or
    ``max |p_i + origin|`` while a local frame is active
    (:data:`_FRAME_ORIGIN`).
    """
    origin = _FRAME_ORIGIN.get()
    if origin is None:
        return max(float(np.max(np.abs(p))) for p in points)
    return max(
        float(np.max(np.abs(np.asarray(p, dtype=float) + origin))) for p in points
    )


def _segment_count(
    length: float, ds: float, minimum: int = 1, scale: float = 0.0
) -> int:
    """Segments of a wire: ``max(minimum, ceil(length / ds - tol))``.

    ``tol`` is :data:`SEGMENT_COUNT_TOL`, or the round-off of a length
    computed from coordinates of magnitude ``scale`` if that is larger:
    ``COLLINEAR_ROUNDOFF_ULPS * eps * scale / ds`` (the allowance of the
    straight-run test, :func:`_continues_straight`). At local coordinates
    (``scale`` below 70 km per metre of ``ds``) the tolerance is
    :data:`SEGMENT_COUNT_TOL`; at a UTM easting of 32.6e6 m an edge may
    exceed a whole number of 0.5 m segments by 0.46 um without getting
    another one. Up to the calendar release the tolerance was 1e-9 of
    ``ds`` everywhere, which the round-off of projected coordinates
    exceeded: a generated 10 m mesh foundation got 508 instead of 440
    segments at the UTM origin (audit fw1sol).

    Parameters
    ----------
    length
        Wire length in m.
    ds
        Target (maximum) segment length in m.
    minimum
        Smallest count (8 for a ring).
    scale
        Magnitude in m of the coordinates ``length`` was computed from
        (:func:`_coordinate_scale`); ``0`` for a length given as a
        parameter (rod, ring, grid span).
    """
    tol = SEGMENT_COUNT_TOL
    if scale > 0.0:
        roundoff = COLLINEAR_ROUNDOFF_ULPS * _EPS * scale / ds
        if roundoff > tol:
            tol = roundoff
    return max(minimum, int(np.ceil(length / ds - tol)))


def _split_interval_at_interfaces(
    t0: float,
    t1: float,
    interfaces: Iterable[float] | None,
) -> list[tuple[float, float]]:
    """Partition ``[t0, t1]`` at every interior interface value.

    ADR-0007 step 1 (audit 2026-07-08, WP-D2): segments must not
    straddle a soil-layer interface, otherwise the discretiser
    assigns the whole segment to the layer of its midpoint and the
    kernel evaluates up to ``ds/2`` of wire length with the wrong
    layer resistivity.
    """
    cuts = sorted(t for t in (interfaces or ()) if t0 + 1e-12 < t < t1 - 1e-12)
    bounds = [t0, *cuts, t1]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def _segments_per_wire(span: float, ds: float, n_spans: int) -> int:
    """Segments per grid wire, aligned to the crossing pattern.

    A wire of length ``span`` is crossed at ``n_spans + 1`` points (the
    perpendicular wires), dividing it into ``n_spans`` equal mesh spans.
    The requested target is ``ceil(span / ds)`` segments; this rounds it
    **up to the next multiple of** ``n_spans`` so every span holds a whole
    number of segments and the crossings coincide with segment endpoints,
    never with the midpoint point sources. With one interior span or none
    (``n_spans <= 1``) there is nothing to align and the plain count is
    returned. The result is never coarser than requested.
    """
    n = _segment_count(span, ds)
    if n_spans <= 1:
        return n
    return int(np.ceil(n / n_spans)) * n_spans


def _straight_wire(
    p0: np.ndarray, direction: np.ndarray, t_lo: float, seg_len: float, n: int
) -> WireSegments:
    """``n`` segments of length ``seg_len`` along ``p0 + t·direction``.

    Segment ``k`` has its midpoint at ``t = t_lo + (k + 1/2)·seg_len``;
    the arithmetic is exactly that of the historic per-segment loop.
    """
    t = t_lo + (np.arange(n) + 0.5) * seg_len
    mids = p0[None, :] + t[:, None] * direction[None, :]
    return WireSegments(
        mids, np.full(n, seg_len), np.broadcast_to(direction, (n, 3)).copy()
    )


#: Machine epsilon of float64.
_EPS = float(np.finfo(float).eps)

#: Round-off allowance of the straight-run test of :func:`_straight_runs`,
#: in units of the machine epsilon times the largest coordinate magnitude of
#: the three vertices (audit W1A-01 of the calendar release). A vertex
#: continues a straight wire when its perpendicular distance from the chord
#: of its neighbours is at most ``COLLINEAR_ROUNDOFF_ULPS * eps * max|x| +
#: 1e-12`` m: 2.4e-12 m at local coordinates of 100 m, 0.46 um at a projected
#: (UTM) easting of 32.6e6 m. Generated mesh wires, which have a vertex at
#: every crossing, stay one wire wherever they stand (their crossing vertices
#: are off the chord by round-off, at most 1.9e-9 m at the UTM easting, 1/250
#: of the tolerance), while
#: any visible bend of a building outline starts a new one (user decision
#: Q30). Up to the audit the test was an absolute sine of 1e-9 between the
#: edge directions, which the round-off at UTM coordinates exceeded.
COLLINEAR_ROUNDOFF_ULPS = 64.0

#: Floor of the straight-run tolerance in m (coordinates near the origin).
_COLLINEAR_FLOOR_M = 1e-12


def _continues_straight(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Whether the edge ``b -> c`` continues the edge ``a -> b`` straight on.

    The middle vertex ``b`` must lie on the chord ``a -> c`` up to the
    round-off of the coordinates (:data:`COLLINEAR_ROUNDOFF_ULPS`), and the
    two edges must point the same way (a vertex where the wire turns back
    starts a new wire).

    Parameters
    ----------
    a, b, c
        Consecutive vertices in m, shape ``(3,)``.

    Returns
    -------
    bool
        ``True`` for a straight continuation.
    """
    ab = b - a
    if float(np.dot(ab, c - b)) <= 0.0:
        return False
    ac = c - a
    height = float(np.linalg.norm(np.cross(ab, ac))) / float(np.linalg.norm(ac))
    scale = _coordinate_scale(a, b, c)
    tol = COLLINEAR_ROUNDOFF_ULPS * float(np.finfo(float).eps) * scale
    return height <= tol + _COLLINEAR_FLOOR_M


def _straight_runs(
    edges: Sequence[tuple[Sequence[float], Sequence[float]]], *, closed: bool
) -> list[int]:
    """Wire index of every edge of a polyline: one per straight run.

    Consecutive edges that continue in the same direction (their common
    vertex on the chord of the two outer ones up to round-off,
    :func:`_continues_straight`) belong to one straight wire; a closed
    polyline also joins its last edge to its first. The wires are numbered
    in the order of their first edge, so the first edge is on wire 0.

    Parameters
    ----------
    edges
        ``(start, end)`` points in m of the consecutive edges.
    closed
        Whether the last edge ends where the first one starts.

    Returns
    -------
    list[int]
        Wire index per edge.
    """
    pts = [
        (np.asarray(start, dtype=float), np.asarray(end, dtype=float))
        for start, end in edges
    ]

    def continues(prev: tuple, nxt: tuple) -> bool:
        return _continues_straight(prev[0], nxt[0], nxt[1])

    wire = [0] * len(pts)
    for k in range(1, len(pts)):
        wire[k] = wire[k - 1] if continues(pts[k - 1], pts[k]) else wire[k - 1] + 1
    if closed and len(pts) > 2 and wire[-1] != 0 and continues(pts[-1], pts[0]):
        last = wire[-1]
        wire = [0 if w == last else w for w in wire]
    return wire


def _edge_wires(
    edges: Sequence[tuple[Sequence[float], Sequence[float]]],
    ds: float,
    *,
    skip_degenerate: bool,
    edge_wires: Sequence[int] | None = None,
) -> WireSegments:
    """Discretise a chain of straight edges (polyline / star arms).

    Every edge is a wire of its own (:attr:`WireSegments.wires`: the
    position of the edge among the edges kept), unless ``edge_wires``
    gives the wire index per edge (a polyline's straight runs,
    :func:`_straight_runs`).
    """
    parts: list[WireSegments] = []
    for k_edge, (start, end) in enumerate(edges):
        p0 = np.array(start, dtype=float)
        p1 = np.array(end, dtype=float)
        L = float(np.linalg.norm(p1 - p0))
        if skip_degenerate and L <= 1e-12:
            continue
        direction = (p1 - p0) / L
        n = _segment_count(L, ds, scale=_coordinate_scale(p0, p1))
        seg_len = L / n
        # Historic per-segment arithmetic: p0 + ((k + 0.5) * seg_len) * dir.
        mids = p0[None, :] + ((np.arange(n) + 0.5) * seg_len)[:, None] * direction
        wire = len(parts) if edge_wires is None else int(edge_wires[k_edge])
        parts.append(
            WireSegments(
                mids,
                np.full(n, seg_len),
                np.broadcast_to(direction, (n, 3)).copy(),
                np.full(n, wire, dtype=int),
            )
        )
    return WireSegments.concatenate(parts)


def _grid_wires(
    *,
    cx: float,
    cy: float,
    cz: float,
    dx: float,
    dy: float,
    nx_wires: int,
    ny_wires: int,
    ds: float,
) -> WireSegments:
    """Segments of a rectangular wire grid (mesh / grid-mesh family).

    Parameters
    ----------
    cx, cy, cz
        Corner of the rectangle in m (min-x, min-y, depth).
    dx, dy
        Extent of the rectangle in m.
    nx_wires
        Number of longitudinal wires (each running from ``cx`` to
        ``cx + dx`` at one of ``nx_wires`` distinct ``y`` values).
    ny_wires
        Number of transverse wires (each running from ``cy`` to
        ``cy + dy`` at one of ``ny_wires`` distinct ``x`` values).
    ds
        Target segment length in m.
    """
    xs = np.linspace(cx, cx + dx, ny_wires)
    ys = np.linspace(cy, cy + dy, nx_wires)

    # Segments per wire, aligned so grid crossings fall on segment
    # *endpoints* rather than midpoints. Longitudinal wires (span dx) are
    # crossed by the ``ny_wires`` transverse wires, i.e. ``ny_wires - 1``
    # mesh spans; making the segment count a multiple of that span count
    # places one or more whole segments in each span, so no segment
    # midpoint (the point-source location) ever coincides with a crossing.
    # Coincident midpoints of a longitudinal and a transverse segment
    # would otherwise land within the singularity clamp and inflate the
    # grid resistance by ~2x for the uniform-current ``image`` backend (a
    # silent grid-discretisation trap; the current-solving backends are
    # far less sensitive but pay the same clamp warning).
    n_long = _segments_per_wire(dx, ds, ny_wires - 1)
    n_trans = _segments_per_wire(dy, ds, nx_wires - 1)

    # Longitudinal wires (along x for each y)
    seg_len_x = dx / n_long
    xm = cx + (np.arange(n_long) + 0.5) * seg_len_x
    long_mids = np.empty((len(ys) * n_long, 3))
    long_mids[:, 0] = np.tile(xm, len(ys))
    long_mids[:, 1] = np.repeat(ys, n_long)
    long_mids[:, 2] = cz
    # Transverse wires (along y for each x)
    seg_len_y = dy / n_trans
    ym = cy + (np.arange(n_trans) + 0.5) * seg_len_y
    trans_mids = np.empty((len(xs) * n_trans, 3))
    trans_mids[:, 0] = np.repeat(xs, n_trans)
    trans_mids[:, 1] = np.tile(ym, len(xs))
    trans_mids[:, 2] = cz
    n_l, n_t = long_mids.shape[0], trans_mids.shape[0]
    directions = np.zeros((n_l + n_t, 3))
    directions[:n_l, 0] = 1.0
    directions[n_l:, 1] = 1.0
    # One wire per full wire line: the longitudinal lines 0 .. nx_wires - 1,
    # then the transverse ones (user decision Q30).
    wires = np.concatenate(
        [
            np.repeat(np.arange(len(ys)), n_long),
            len(ys) + np.repeat(np.arange(len(xs)), n_trans),
        ]
    )
    return WireSegments(
        np.concatenate([long_mids, trans_mids], axis=0),
        np.concatenate([np.full(n_l, seg_len_x), np.full(n_t, seg_len_y)]),
        directions,
        wires,
    )


def _grid_wire_polylines(
    *,
    cx: float,
    cy: float,
    cz: float,
    dx: float,
    dy: float,
    nx_wires: int,
    ny_wires: int,
) -> list[np.ndarray]:
    """End points of every wire of a rectangular grid (see :func:`_grid_wires`)."""
    xs = np.linspace(cx, cx + dx, ny_wires)
    ys = np.linspace(cy, cy + dy, nx_wires)
    out = [np.array([[cx, y, cz], [cx + dx, y, cz]], dtype=float) for y in ys]
    out += [np.array([[x, cy, cz], [x, cy + dy, cz]], dtype=float) for x in xs]
    return out
