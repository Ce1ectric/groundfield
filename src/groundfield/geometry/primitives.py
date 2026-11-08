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
  equal arc segments represented by their chords' midpoints on the arc.

Coordinates are in metres with the :math:`z` axis pointing down into
the soil (depth positive).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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

    def __len__(self) -> int:
        return int(self.lengths.shape[0])

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
        """Join several containers in order."""
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        return cls(
            np.concatenate([p.midpoints for p in parts], axis=0),
            np.concatenate([p.lengths for p in parts]),
            np.concatenate([p.directions for p in parts], axis=0),
        )


#: Tolerance of the segment count on ``length / ds``: a wire that exceeds a
#: whole multiple of ``ds`` only through floating-point noise (an edge of
#: 3.0000000000000004 m at ds = 1.5 m) gets no extra segment. GEN-1 found 52
#: instead of 50 segments on footprint polygons (review pass 10, 10.32).
SEGMENT_COUNT_TOL = 1e-9


def _segment_count(length: float, ds: float, minimum: int = 1) -> int:
    """Segments of a wire: ``max(minimum, ceil(length / ds - SEGMENT_COUNT_TOL))``.

    Parameters
    ----------
    length
        Wire length in m.
    ds
        Target (maximum) segment length in m.
    minimum
        Smallest count (8 for a ring).
    """
    return max(minimum, int(np.ceil(length / ds - SEGMENT_COUNT_TOL)))


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


def _edge_wires(
    edges: Sequence[tuple[Sequence[float], Sequence[float]]],
    ds: float,
    *,
    skip_degenerate: bool,
) -> WireSegments:
    """Discretise a chain of straight edges (polyline / star arms)."""
    parts: list[WireSegments] = []
    for start, end in edges:
        p0 = np.array(start, dtype=float)
        p1 = np.array(end, dtype=float)
        L = float(np.linalg.norm(p1 - p0))
        if skip_degenerate and L <= 1e-12:
            continue
        direction = (p1 - p0) / L
        n = _segment_count(L, ds)
        seg_len = L / n
        # Historic per-segment arithmetic: p0 + ((k + 0.5) * seg_len) * dir.
        mids = p0[None, :] + ((np.arange(n) + 0.5) * seg_len)[:, None] * direction
        parts.append(
            WireSegments(
                mids, np.full(n, seg_len), np.broadcast_to(direction, (n, 3)).copy()
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
    return WireSegments(
        np.concatenate([long_mids, trans_mids], axis=0),
        np.concatenate([np.full(n_l, seg_len_x), np.full(n_t, seg_len_y)]),
        directions,
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
