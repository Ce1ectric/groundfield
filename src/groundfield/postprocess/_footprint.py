"""Horizontal footprint of an electrode's wires (private helper).

The safety helpers need the electrode geometry in the horizontal
projection: where does a person stand who is ``d`` metres away from the
nearest wire, on a given side of the electrode (review pass 10, 10.21),
and how far is a surface point from the wires (10.22)?

Every electrode is reduced to straight wires (``segments``, a vertical
rod becomes a point) and circles (rings). The wires are the ones the
electrode reports itself through ``wire_polylines()`` (the geometry
methods of :mod:`groundfield.geometry.electrodes`, review pass 10,
10.79), bit for bit the wire positions of the solver's discretisation;
only a ring keeps its exact circle, which ``wire_polylines`` samples by
a polygon.

Mathematical content
--------------------
For a wire set :math:`W` and a distance :math:`d > 0` the
:math:`d`-neighbourhood :math:`N_d = \\{p : \\operatorname{dist}(p, W)
\\le d\\}` is the union of one capsule (stadium) per straight wire and
one annulus per circle. The foot position in direction
:math:`\\hat{\\mathbf e}` seen from the reference point
:math:`\\mathbf c` is the exit point of the ray
:math:`\\mathbf c + t\\,\\hat{\\mathbf e}` from :math:`N_d`,

.. math::

    t^\\ast = \\sup\\{t \\ge 0 : \\mathbf c + t\\,\\hat{\\mathbf e} \\in N_d\\},

so the feet are exactly :math:`d` from the nearest wire. Each capsule
is convex, so its intersection with the ray is one interval and
:math:`t^\\ast` is the largest upper end over the capsules (computed in
closed form for the rectangle and the two end disks). When the ray from
:math:`\\mathbf c` never comes within :math:`d` of the wires (possible
for a non-convex footprint whose centroid lies off the wires), the feet
are placed at :math:`\\mathbf s + d\\,\\hat{\\mathbf e}` with
:math:`\\mathbf s` the support point of the wires in direction
:math:`\\hat{\\mathbf e}`; every wire point then lies at least
:math:`d` away, the support point exactly :math:`d`.

The reference point :math:`\\mathbf c` is the length-weighted centroid
of the wires (the centre of a ring, grid or star, the midpoint of a
strip, the head of a rod).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__: list[str] = []

# Relative tolerance below which a ray parameter counts as zero (the
# reference point sitting exactly at distance d from a wire).
_T_TOL = 1e-9


@dataclass(frozen=True)
class Footprint:
    """Wires of one electrode in the horizontal projection.

    Attributes
    ----------
    segments
        Straight wires, array of shape ``(K, 2, 2)`` with start and end
        point ``(x, y)`` in m. A vertical rod is a degenerate segment
        (start == end).
    circles
        Circular wires, array of shape ``(C, 3)`` with centre ``x``,
        ``y`` and radius in m.
    """

    segments: np.ndarray
    circles: np.ndarray

    # ------------------------------------------------------------------
    @property
    def reference_point(self) -> np.ndarray:
        """Length-weighted centroid of the wires, shape ``(2,)``, in m."""
        a = self.segments[:, 0, :]
        b = self.segments[:, 1, :]
        lengths = np.linalg.norm(b - a, axis=1)
        weights = list(lengths)
        mids = list(0.5 * (a + b))
        for cx, cy, r in self.circles:
            weights.append(2.0 * math.pi * r)
            mids.append(np.array([cx, cy]))
        w = np.asarray(weights, dtype=float)
        m = np.asarray(mids, dtype=float).reshape(-1, 2)
        if w.sum() > 0.0:
            return (w[:, None] * m).sum(axis=0) / w.sum()
        # Only points (rods): plain mean of the positions.
        return m.mean(axis=0)

    # ------------------------------------------------------------------
    def distance(self, points: np.ndarray) -> np.ndarray:
        """Horizontal distance in m from ``points`` ``(M, 2)`` to the wires."""
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        best = np.full(p.shape[0], np.inf)
        if self.segments.size:
            a = self.segments[:, 0, :][None, :, :]
            ab = (self.segments[:, 1, :] - self.segments[:, 0, :])[None, :, :]
            ap = p[:, None, :] - a
            l2 = np.einsum("mkd,mkd->mk", ab, ab)
            t = np.where(
                l2 > 0.0,
                np.einsum("mkd,mkd->mk", ap, ab) / np.where(l2 > 0.0, l2, 1.0),
                0.0,
            )
            t = np.clip(t, 0.0, 1.0)
            diff = ap - t[..., None] * ab
            best = np.minimum(
                best, np.sqrt(np.einsum("mkd,mkd->mk", diff, diff)).min(1)
            )
        for cx, cy, r in self.circles:
            rho = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
            best = np.minimum(best, np.abs(rho - r))
        return best

    # ------------------------------------------------------------------
    def _exit_parameter(self, c: np.ndarray, e: np.ndarray, d: float) -> float:
        """Largest ``t`` with ``dist(c + t e, wires) <= d``, ``-inf`` if none."""
        best = -np.inf
        if self.segments.size:
            a = self.segments[:, 0, :]
            b = self.segments[:, 1, :]
            for end in (a, b):  # end disks of every capsule
                w = c[None, :] - end
                bb = w @ e
                disc = bb**2 - (np.einsum("kd,kd->k", w, w) - d * d)
                ok = disc >= 0.0
                if ok.any():
                    best = max(best, float((-bb[ok] + np.sqrt(disc[ok])).max()))
            lengths = np.linalg.norm(b - a, axis=1)
            for k in np.flatnonzero(lengths > 0.0):  # capsule rectangles
                u = (b[k] - a[k]) / lengths[k]
                n = np.array([-u[1], u[0]])
                w = c - a[k]
                lo, hi = -np.inf, np.inf
                for alpha, beta, v_lo, v_hi in (
                    (float(w @ u), float(e @ u), 0.0, float(lengths[k])),
                    (float(w @ n), float(e @ n), -d, d),
                ):
                    if beta == 0.0:
                        if not (v_lo <= alpha <= v_hi):
                            lo, hi = np.inf, -np.inf
                        continue
                    t1, t2 = (v_lo - alpha) / beta, (v_hi - alpha) / beta
                    lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
                if lo <= hi:
                    best = max(best, hi)
        for cx, cy, r in self.circles:  # annulus: exit through radius r + d
            w = c - np.array([cx, cy])
            bb = float(w @ e)
            disc = bb**2 - (float(w @ w) - (r + d) ** 2)
            if disc >= 0.0:
                best = max(best, -bb + math.sqrt(disc))
        return best

    def _support_point(self, c: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Wire point farthest along ``e``; ties broken towards the ray."""
        e_perp = np.array([-e[1], e[0]])
        cands: list[np.ndarray] = []
        values: list[float] = []
        for a, b in self.segments:
            va, vb = float(a @ e), float(b @ e)
            if abs(va - vb) <= 1e-12 * max(1.0, abs(va), abs(vb)):
                # Whole segment is a support candidate: take its point
                # closest to the line through c along e.
                ab = b - a
                q = float(ab @ e_perp)
                s = (
                    0.0
                    if q == 0.0
                    else float(np.clip(-((a - c) @ e_perp) / q, 0.0, 1.0))
                )
                cands.append(a + s * ab)
                values.append(max(va, vb))
            else:
                cands.append(a if va > vb else b)
                values.append(max(va, vb))
        for cx, cy, r in self.circles:
            p = np.array([cx, cy]) + r * e
            cands.append(p)
            values.append(float(p @ e))
        v = np.asarray(values)
        top = v.max()
        tol = 1e-12 * max(1.0, abs(top))
        idx = [i for i in np.flatnonzero(v >= top - tol)]
        off = [abs(float((cands[i] - c) @ e_perp)) for i in idx]
        return cands[idx[int(np.argmin(off))]]

    def feet(self, directions: np.ndarray, d: float) -> np.ndarray:
        """Foot positions ``(n, 2)`` at distance ``d`` from the wires.

        Parameters
        ----------
        directions
            Unit vectors ``(n, 2)`` in the horizontal plane.
        d
            Horizontal distance to the nearest wire in m, ``> 0``.

        Returns
        -------
        numpy.ndarray
            Foot positions ``(x, y)`` in m, one row per direction.
        """
        c = self.reference_point
        out = np.empty((len(directions), 2))
        for i, e in enumerate(np.asarray(directions, dtype=float)):
            t = self._exit_parameter(c, e, d)
            if t >= -_T_TOL * max(1.0, d):
                out[i] = c + max(t, 0.0) * e
            else:
                out[i] = self._support_point(c, e) + d * e
        return out


# ---------------------------------------------------------------------
# electrode -> footprint
# ---------------------------------------------------------------------


def _wire_runs(electrode) -> list[np.ndarray]:
    """``electrode.wire_polylines()``, or a ``TypeError`` naming the fallback."""
    try:
        return electrode.wire_polylines()
    except (AttributeError, NotImplementedError):
        raise TypeError(
            "No horizontal footprint for electrode kind "
            f"{getattr(electrode, 'kind', None)!r}; use anchor='connection_point'."
        ) from None


def footprint(electrode) -> Footprint:
    """Horizontal wire footprint of an electrode.

    Every straight wire of ``electrode.wire_polylines()`` (consecutive
    vertices of every run) projected onto the surface; a ring is its exact
    circle.

    Parameters
    ----------
    electrode
        Any electrode of :mod:`groundfield.geometry.electrodes`.

    Returns
    -------
    Footprint

    Raises
    ------
    TypeError
        For an object without wire geometry.
    """
    if getattr(electrode, "kind", None) == "ring":
        cx, cy, _ = electrode.center
        return Footprint(np.zeros((0, 2, 2)), np.array([[cx, cy, electrode.radius]]))
    pairs = [
        run[k : k + 2, :2] for run in _wire_runs(electrode) for k in range(len(run) - 1)
    ]
    segments = np.array(pairs, dtype=float) if pairs else np.zeros((0, 2, 2))
    return Footprint(segments, np.zeros((0, 3)))


def _grid_lines(electrode) -> tuple[np.ndarray, np.ndarray]:
    """``x`` of the wires along ``y`` and ``y`` of the wires along ``x``.

    Read from the wires of a mesh / grid electrode, so the positions are
    the discretiser's.
    """
    xs: set[float] = set()
    ys: set[float] = set()
    for run in _wire_runs(electrode):
        (x0, y0), (x1, y1) = run[0, :2], run[-1, :2]
        if x0 == x1:
            xs.add(float(x0))
        if y0 == y1:
            ys.add(float(y0))
    return np.array(sorted(xs)), np.array(sorted(ys))


def mesh_centres(electrode) -> np.ndarray:
    """Centres ``(M, 2)`` of the meshes of a mesh / grid electrode, in m.

    The midpoints between neighbouring wires, the classical position of
    the IEEE 80 mesh voltage. Empty for every other electrode kind.
    """
    if getattr(electrode, "kind", None) not in ("grid_mesh", "mesh"):
        return np.zeros((0, 2))
    xs, ys = _grid_lines(electrode)
    mx = 0.5 * (xs[:-1] + xs[1:])
    my = 0.5 * (ys[:-1] + ys[1:])
    gx, gy = np.meshgrid(mx, my, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])


# ---------------------------------------------------------------------
# interior of closed loops (envelope, 10.21)
# ---------------------------------------------------------------------

#: Candidate grid (per axis) of the interior search of a closed polyline.
_INTERIOR_GRID = 33


def _inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd test of ``points`` ``(M, 2)`` against ``polygon`` ``(K, 2)``."""
    x = points[:, 0][:, None]
    y = points[:, 1][:, None]
    x0, y0 = polygon[:, 0][None, :], polygon[:, 1][None, :]
    x1, y1 = np.roll(polygon[:, 0], -1)[None, :], np.roll(polygon[:, 1], -1)[None, :]
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cut = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (crosses & (x < x_cut)).sum(axis=1) % 2 == 1


def _area_centroid(polygon: np.ndarray) -> np.ndarray:
    """Area centroid of a simple polygon (vertex mean if it has no area).

    Computed relative to the first vertex, so UTM-scale coordinates do not
    cancel in the shoelace products.
    """
    origin = polygon[0]
    local = polygon - origin
    x, y = local[:, 0], local[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = 0.5 * cross.sum()
    if abs(area) <= 1e-12 * max(1.0, float(np.ptp(local, axis=0).max()) ** 2):
        return polygon.mean(axis=0)
    return origin + np.array(
        [
            ((x + xn) * cross).sum() / (6.0 * area),
            ((y + yn) * cross).sum() / (6.0 * area),
        ]
    )


def _loop_centre(electrode) -> np.ndarray:
    """Point inside a closed polyline that lies farthest from its wires.

    The candidates are a regular grid over the bounding box of the wires
    (:meth:`bounds`) and the area centroid; among the candidates inside
    the polygon the one with the largest distance to the wires wins, ties
    going to the candidate closest to the centroid (for a rectangle its
    centre). Shape ``(1, 2)``, or ``(0, 2)`` if no candidate lies inside.
    """
    polygon = np.asarray(electrode.vertices, dtype=float)[:, :2]
    lower, upper = electrode.bounds()
    gx = np.linspace(lower[0], upper[0], _INTERIOR_GRID)
    gy = np.linspace(lower[1], upper[1], _INTERIOR_GRID)
    centroid = _area_centroid(polygon)
    grid = np.column_stack([a.ravel() for a in np.meshgrid(gx, gy, indexing="ij")])
    candidates = np.vstack([centroid[None, :], grid])
    candidates = candidates[_inside_polygon(candidates, polygon)]
    if not len(candidates):
        return np.zeros((0, 2))
    d = footprint(electrode).distance(candidates)
    extent = float(max(upper[0] - lower[0], upper[1] - lower[1], 1.0))
    best = candidates[d >= d.max() - 1e-9 * extent]
    k = int(np.argmin(np.hypot(*(best - centroid).T)))
    return best[k : k + 1]


def interior_points(electrode, *, outlines: bool) -> np.ndarray:
    """Centre of every closed mesh of an electrode, ``(M, 2)`` in m.

    The interior positions of the touch-voltage envelope (review pass 10,
    10.21), the analogue of the IEEE 80 mesh voltage: the mesh centres of a
    ``mesh`` / ``grid_mesh`` electrode, the centre of a ``ring``, and,
    with ``outlines=True``, the point of a closed ``polyline`` farthest
    from its wires (:func:`_loop_centre`). Empty for every other shape.

    Parameters
    ----------
    electrode
        Any electrode of :mod:`groundfield.geometry.electrodes`.
    outlines
        Include closed polylines (building outlines) as well.
    """
    kind = getattr(electrode, "kind", None)
    if kind in ("grid_mesh", "mesh"):
        return mesh_centres(electrode)
    if kind == "ring":
        return np.array([electrode.center[:2]], dtype=float)
    if (
        outlines
        and kind == "polyline"
        and electrode.closed
        and len(electrode.vertices) > 2
    ):
        return _loop_centre(electrode)
    return np.zeros((0, 2))
