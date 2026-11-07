"""Horizontal footprint of an electrode's wires (private helper).

The safety helpers need the electrode geometry in the horizontal
projection: where does a person stand who is ``d`` metres away from the
nearest wire, on a given side of the electrode (review pass 10, 10.21),
and how far is a surface point from the wires (10.22)?

Every electrode is reduced to straight wires (``segments``, a vertical
rod becomes a point) and circles (rings). This is a local
implementation; the electrode classes are expected to gain geometry
methods (``wire_polylines`` and similar) in a later refactoring, at which
point this module should delegate to them.

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


def _segments(pairs: list[tuple[tuple, tuple]]) -> np.ndarray:
    if not pairs:
        return np.zeros((0, 2, 2))
    return np.array([[a[:2], b[:2]] for a, b in pairs], dtype=float)


def _mesh_lines(electrode) -> tuple[np.ndarray, np.ndarray]:
    """x positions of the transverse and y positions of the longitudinal
    wires of a mesh / grid electrode (mirrors the discretiser)."""
    cx, cy, _ = electrode.corner
    dx, dy = electrode.size
    if electrode.kind == "grid_mesh":
        xs = cx + dx * np.arange(electrode.n_x + 1) / electrode.n_x
        ys = cy + dy * np.arange(electrode.n_y + 1) / electrode.n_y
    else:  # "mesh": uniform spacing, as in solver.image._discretize_mesh
        nx = max(2, int(np.round(dx / electrode.spacing)) + 1)
        ny = max(2, int(np.round(dy / electrode.spacing)) + 1)
        xs = np.linspace(cx, cx + dx, nx)
        ys = np.linspace(cy, cy + dy, ny)
    return xs, ys


def footprint(electrode) -> Footprint:
    """Horizontal wire footprint of an electrode.

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
        For an electrode kind this helper does not know.
    """
    kind = getattr(electrode, "kind", None)
    none = np.zeros((0, 3))
    if kind == "rod":
        x, y, _ = electrode.position
        return Footprint(_segments([((x, y), (x, y))]), none)
    if kind == "ring":
        cx, cy, _ = electrode.center
        return Footprint(np.zeros((0, 2, 2)), np.array([[cx, cy, electrode.radius]]))
    if kind == "strip":
        return Footprint(_segments([(electrode.start, electrode.end)]), none)
    if kind in ("polyline", "star"):
        return Footprint(_segments(list(electrode.edges)), none)
    if kind in ("grid_mesh", "mesh"):
        xs, ys = _mesh_lines(electrode)
        y0, y1 = ys[0], ys[-1]
        x0, x1 = xs[0], xs[-1]
        pairs = [((x, y0), (x, y1)) for x in xs] + [((x0, y), (x1, y)) for y in ys]
        return Footprint(_segments(pairs), none)
    raise TypeError(
        f"No horizontal footprint for electrode kind {kind!r}; use "
        "anchor='connection_point'."
    )


def mesh_centres(electrode) -> np.ndarray:
    """Centres ``(M, 2)`` of the meshes of a mesh / grid electrode, in m.

    Empty for every other electrode kind.
    """
    if getattr(electrode, "kind", None) not in ("grid_mesh", "mesh"):
        return np.zeros((0, 2))
    xs, ys = _mesh_lines(electrode)
    mx = 0.5 * (xs[:-1] + xs[1:])
    my = 0.5 * (ys[:-1] + ys[1:])
    gx, gy = np.meshgrid(mx, my, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])
