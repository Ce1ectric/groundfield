r"""Detection of close, interpenetrating and crossing wires.

The thin-wire kernels of groundfield (point and line sources for the
galvanic reaction matrix, Neumann integrals for the inductance matrix)
assume that two distinct wires are separated by much more than their
radii. Review pass 10 (finding 10.44) measured what happens when they
are not: two 3 m rods (radius 8 mm) 2 mm apart — their surfaces
interpenetrate — give about ten times the physical impedance, 2 cm
apart still +79 %, 5 cm +24 %; foundations 0.2 m apart at a 2 m
segment length +3.1 %. None of this raised a diagnostic.

This module provides the geometric part of the guard as a public
helper: given straight wire segments with radii and owner names it
returns every pair of segments from *different* owners whose axes come
closer than a threshold, classified as

``"crossing"``
    the axes intersect inside both segments (two conductors crossing at
    one depth);
``"touching"``
    the end point of one segment lies on the interior of the other (a
    T-contact; the model does not connect conductors there);
``"interpenetrating"``
    the axis distance is below the sum of the radii, $d < a_i + a_j$, by
    more than $10^{-9}$ of it or 16 units in the last place of the largest
    coordinate of the pair, whichever is larger (wires drawn at the
    touching distance are ``"close"``, also at projected coordinates; audits
    W3B-03, W3R2B-02);
``"close"``
    $d$ is below the threshold
    $\max(\texttt{rel\_threshold}\cdot(a_i + a_j), \texttt{abs\_threshold})$;
``"junction"``
    the two segments only share an end point (reported only with
    ``include_junctions=True``).

Pairs that only meet at a shared end point (a conductor ending on an
electrode or another conductor — a junction by design) are skipped by
default. The diagnostics layer decides what to warn or raise; the
helper itself never warns.

The minimum distance between two segments is computed exactly
(closest points of two line segments, clamped to the segment ends);
candidate pairs are pre-selected with a k-d tree on the segment
mid-points, so the cost is close to linear in the number of segments.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from groundfield.conductors.conductor import Conductor

__all__ = [
    "ProximityKind",
    "WireProximity",
    "segment_distance",
    "find_close_wires",
    "find_close_conductors",
]

ProximityKind = Literal["crossing", "touching", "interpenetrating", "close", "junction"]
"""Kind of a close wire pair (``WireProximity.kind``).

- ``"crossing"``: the axes intersect inside both segments;
- ``"touching"``: the end point of one segment lies on the interior of
  the other (a T-contact the model does not connect);
- ``"interpenetrating"``: the axis distance is below the sum of the radii;
- ``"close"``: the axis distance is below the reporting threshold;
- ``"junction"``: the two segments only share an end point.

The exact criteria are in the module docstring.
"""

# End points closer than this fraction of the longer segment count as
# shared (junction) and two axes closer than it as intersecting.
_TOUCH_REL = 1e-9
# Two wires interpenetrate when their axes are closer than the sum of their
# radii by more than this fraction of it: wires drawn at the touching
# distance lie closer by round-off (1e-14 m for 5 mm wires, the service drops
# of a row since decision D9) and are close, not interpenetrating (the
# tolerance of the overlap check of diagnose(), audit W3B-03).
_CONTACT_REL_TOL = 1e-9
# ... or by more than this many units in the last place of the largest
# coordinate of the pair, whichever is larger (audit W3R2B-02): at projected
# coordinates (5.8e6 m, UTM) float64 is spaced 9.3e-10 m, more than 1e-9 of a
# touching distance of 1 cm, and a row drawn at the origin and translated
# there lies closer than touching by round-off (16 ulps: 2.1e-8 m at
# 5.8e6 m). Up to about 2.8 km from the origin (for 1 cm touching) the
# relative term is the larger one, so local worlds keep their findings.
_CONTACT_ULPS = 16.0
_EPS = float(np.finfo(float).eps)


def _closer_than_touching(touch, scale):
    """Axis distance below which two wires lie closer than touching.

    ``touch`` is the sum of the radii in m, ``scale`` the largest absolute
    coordinate of the pair in m (scalars or arrays): the smaller of
    ``touch * (1 - 1e-9)`` and ``touch - 16 * eps * scale`` (audits W3B-03,
    W3R2B-02). Shared by the kinds of :func:`find_close_wires` and the
    overlap check of :func:`groundfield.diagnostics.diagnose`.
    """
    return np.minimum(
        np.multiply(touch, 1.0 - _CONTACT_REL_TOL),
        np.subtract(touch, _CONTACT_ULPS * _EPS * np.asarray(scale, dtype=float)),
    )


@dataclass(frozen=True)
class WireProximity:
    """One pair of wire segments that are too close.

    Attributes
    ----------
    kind
        ``"crossing"``, ``"touching"``, ``"interpenetrating"``,
        ``"close"`` or ``"junction"`` (see the module docstring).
    name_a
        Owner name of the first segment (e.g. a conductor name).
    name_b
        Owner name of the second segment.
    index_a
        Index of the first segment in the input arrays.
    index_b
        Index of the second segment in the input arrays.
    distance
        Minimum axis-to-axis distance $d$ in m.
    clearance
        Surface-to-surface distance $d - (a_a + a_b)$ in m; negative
        when the wires interpenetrate.
    point_a
        Closest point on the axis of the first segment, ``(x, y, z)`` in m.
    point_b
        Closest point on the axis of the second segment, ``(x, y, z)`` in m.
    """

    kind: ProximityKind
    name_a: str
    name_b: str
    index_a: int
    index_b: int
    distance: float
    clearance: float
    point_a: tuple[float, float, float]
    point_b: tuple[float, float, float]


def segment_distance(
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Minimum distance between straight segments (vectorised).

    Closest points $p_a(s) = p_{1,a} + s\,d_a$ and
    $p_b(t) = p_{1,b} + t\,d_b$ with $s, t \in [0, 1]$, by the standard
    clamped two-segment algorithm (Ericson 2005, sec. 5.1.9). Parallel
    segments and zero-length segments (points) are handled explicitly:
    the distance of a point to a segment is that of its orthogonal
    projection, clamped to the segment (audit of review pass 10, F04:
    a point-like second segment was measured from the start of the
    first one).

    Parameters
    ----------
    p1_a, p2_a, p1_b, p2_b
        End points, arrays of shape ``(K, 3)`` (or ``(3,)``) in m.

    Returns
    -------
    distance, s, t : np.ndarray
        Distance in m and the segment parameters of the closest points,
        each of shape ``(K,)``.
    """
    p1_a = np.atleast_2d(np.asarray(p1_a, dtype=float))
    p2_a = np.atleast_2d(np.asarray(p2_a, dtype=float))
    p1_b = np.atleast_2d(np.asarray(p1_b, dtype=float))
    p2_b = np.atleast_2d(np.asarray(p2_b, dtype=float))
    d1 = p2_a - p1_a
    d2 = p2_b - p1_b
    r = p1_a - p1_b
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(
            denom > 1e-14 * np.maximum(a * e, 1e-300),
            np.clip((b * f - c * e) / denom, 0.0, 1.0),
            0.0,
        )
        t = (b * s + f) / np.where(e > 0.0, e, 1.0)
        # clamp t and recompute s where needed
        t_lo = t < 0.0
        t_hi = t > 1.0
        t = np.clip(t, 0.0, 1.0)
        s = np.where(t_lo, np.clip(-c / np.where(a > 0.0, a, 1.0), 0.0, 1.0), s)
        s = np.where(t_hi, np.clip((b - c) / np.where(a > 0.0, a, 1.0), 0.0, 1.0), s)
        # Degenerate segments: a point against a segment is the clamped
        # projection; two points are their distance.
        scale = np.maximum(np.maximum(a, e), 1e-300)
        point_a = a <= 1e-24 * scale
        point_b = e <= 1e-24 * scale
        s = np.where(point_b, np.clip(-c / np.where(a > 0.0, a, 1.0), 0.0, 1.0), s)
        t = np.where(point_b, 0.0, t)
        s = np.where(point_a, 0.0, s)
        t = np.where(
            point_a & ~point_b, np.clip(f / np.where(e > 0.0, e, 1.0), 0.0, 1.0), t
        )
    ca = p1_a + s[:, None] * d1
    cb = p1_b + t[:, None] * d2
    dist = np.linalg.norm(ca - cb, axis=1)
    return dist, s, t


def find_close_wires(
    endpoints: np.ndarray,
    radii: Sequence[float] | np.ndarray,
    names: Sequence[str],
    *,
    rel_threshold: float = 5.0,
    abs_threshold: float = 0.0,
    include_junctions: bool = False,
) -> list[WireProximity]:
    r"""Find segment pairs of different owners that are too close.

    Parameters
    ----------
    endpoints
        Segment end points, shape ``(K, 2, 3)``, in m.
    radii
        Wire radius of every segment in m, shape ``(K,)``.
    names
        Owner name of every segment (conductor or electrode name),
        length ``K``. Segments of one owner are never paired.
    rel_threshold
        A pair is reported when its axis distance is below
        ``rel_threshold * (a_i + a_j)`` (default 5: e.g. 8 cm for two
        8 mm rods, where the point-source kernel is already +24 % off).
    abs_threshold
        Additional absolute threshold in m (default 0); the larger of
        the two applies.
    include_junctions
        Also report pairs that only meet at a shared end point (default
        ``False``: such junctions are connections by design).

    Returns
    -------
    list of WireProximity
        Sorted by ascending clearance (worst first).

    Raises
    ------
    ValueError
        If ``endpoints`` does not have shape ``(K, 2, 3)``, if ``radii`` or
        ``names`` does not have one entry per segment, if an end point or
        radius is not finite or a radius is negative, or if a threshold is
        negative.
    """
    ep = np.asarray(endpoints, dtype=float)
    if ep.ndim != 3 or ep.shape[1:] != (2, 3):
        raise ValueError(f"endpoints must have shape (K, 2, 3), got {ep.shape}")
    K = ep.shape[0]
    rad = np.asarray(radii, dtype=float).reshape(-1)
    if rad.shape != (K,) or len(names) != K:
        raise ValueError("radii and names must have one entry per segment")
    if not np.all(np.isfinite(ep)) or not np.all(np.isfinite(rad)) or np.any(rad < 0):
        raise ValueError("endpoints and radii must be finite, radii >= 0")
    if rel_threshold < 0.0 or abs_threshold < 0.0:
        raise ValueError("thresholds must be non-negative")
    if K < 2:
        return []
    p1 = ep[:, 0]
    p2 = ep[:, 1]
    mid = 0.5 * (p1 + p2)
    half = 0.5 * np.linalg.norm(p2 - p1, axis=1)
    thr = np.maximum(rel_threshold * 2.0 * rad.max(), abs_threshold)
    reach = 2.0 * half.max() + thr
    from scipy.spatial import cKDTree  # lazy: keep `import groundfield` light

    tree = cKDTree(mid)
    pairs = tree.query_pairs(r=float(reach), output_type="ndarray")
    if pairs.size == 0:
        return []
    names_arr = np.asarray(list(names), dtype=object)
    keep = names_arr[pairs[:, 0]] != names_arr[pairs[:, 1]]
    pairs = pairs[keep]
    if pairs.size == 0:
        return []
    i, j = pairs[:, 0], pairs[:, 1]
    # cheap pre-filter: midpoint distance minus both half lengths
    gap = np.linalg.norm(mid[i] - mid[j], axis=1) - half[i] - half[j]
    pair_thr = np.maximum(rel_threshold * (rad[i] + rad[j]), abs_threshold)
    sel = gap <= pair_thr
    i, j, pair_thr = i[sel], j[sel], pair_thr[sel]
    if i.size == 0:
        return []
    dist, s, t = segment_distance(p1[i], p2[i], p1[j], p2[j])
    hit = dist < pair_thr
    scale = np.maximum(np.abs(ep[i]).max(axis=(1, 2)), np.abs(ep[j]).max(axis=(1, 2)))
    inside = dist < _closer_than_touching(rad[i] + rad[j], scale)
    out: list[WireProximity] = []
    for k in np.nonzero(hit)[0]:
        ia, ib = int(i[k]), int(j[k])
        length = max(2.0 * half[ia], 2.0 * half[ib], 1e-12)
        tol = _TOUCH_REL * length
        at_end_a = s[k] <= 1e-9 or s[k] >= 1.0 - 1e-9
        at_end_b = t[k] <= 1e-9 or t[k] >= 1.0 - 1e-9
        if dist[k] <= tol and at_end_a and at_end_b:
            if not include_junctions:
                continue  # shared end point: a junction by design
            kind: ProximityKind = "junction"
        elif dist[k] <= tol:
            kind = "touching" if (at_end_a or at_end_b) else "crossing"
        elif inside[k]:
            kind = "interpenetrating"
        else:
            kind = "close"
        pa = p1[ia] + s[k] * (p2[ia] - p1[ia])
        pb = p1[ib] + t[k] * (p2[ib] - p1[ib])
        out.append(
            WireProximity(
                kind=kind,
                name_a=str(names_arr[ia]),
                name_b=str(names_arr[ib]),
                index_a=ia,
                index_b=ib,
                distance=float(dist[k]),
                clearance=float(dist[k] - rad[ia] - rad[ib]),
                point_a=(float(pa[0]), float(pa[1]), float(pa[2])),
                point_b=(float(pb[0]), float(pb[1]), float(pb[2])),
            )
        )
    out.sort(key=lambda p: p.clearance)
    return out


def find_close_conductors(
    conductors: Iterable[Conductor],
    *,
    rel_threshold: float = 5.0,
    abs_threshold: float = 0.0,
    include_junctions: bool = False,
) -> list[WireProximity]:
    """:func:`find_close_wires` for :class:`~groundfield.conductors.Conductor` objects.

    Every conductor is one straight wire from ``start`` to ``end`` with
    its ``wire_radius``; pairs are reported by conductor name.

    Parameters
    ----------
    conductors
        Conductors, e.g. ``world.conductors``.
    rel_threshold, abs_threshold, include_junctions
        See :func:`find_close_wires`.

    Returns
    -------
    list of WireProximity
        Close / interpenetrating / crossing conductor pairs, worst
        first.
    """
    conds = list(conductors)
    if len(conds) < 2:
        return []
    ep = np.array([[c.start, c.end] for c in conds], dtype=float)
    radii = np.array([c.wire_radius for c in conds], dtype=float)
    names = [c.name for c in conds]
    return find_close_wires(
        ep,
        radii,
        names,
        rel_threshold=rel_threshold,
        abs_threshold=abs_threshold,
        include_junctions=include_junctions,
    )
