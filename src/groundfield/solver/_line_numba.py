"""Optional numba kernel of the line-kernel far-field term (private).

Imported only by :mod:`groundfield.solver._line_kernel`, and only when
``numba`` is installed (extra ``fast``). The kernel evaluates the
quadrupole correction of a segment pair and of its surface image in
one pass, with the same IEEE operations in the same order as the numpy
path (:func:`groundfield.solver._line_kernel.quadrupole_block`, called
once for the segments and once for their images, then summed) — no
fast-math flags, so no fused multiply-add and no reassociation: both
paths give bit-identical blocks. numba only removes the temporaries of
the numpy path (about 30 full-size arrays per block).
"""

from __future__ import annotations

import numba
import numpy as np

__all__ = ["quadrupole_rows"]


@numba.njit(nogil=True, boundscheck=False, error_model="numpy", cache=True)
def quadrupole_rows(
    xo,
    yo,
    zo,
    tox,
    toy,
    toz,
    lo,
    has_t,
    xs,
    ys,
    zs,
    zi,
    tsx,
    tsy,
    tsz,
    tiz,
    ls,
    factor,
    out,
):  # pragma: no cover
    """Far-field correction of segments plus surface images, in 1/m.

    ``out[r, c] = Q(obs r, src c) + Q(obs r, image of src c)`` with
    ``Q = [L_c²(3u_c²/R² - 1) + L_r²(3u_r²/R² - 1)] / (24 R³)`` for
    ``R >= factor · max(L_r, L_c)`` and ``Q = 0`` otherwise.

    Parameters
    ----------
    xo, yo, zo
        Observer midpoints in m, shape ``(b,)``.
    tox, toy, toz
        Observer unit tangents, shape ``(b,)`` (ignored without ``has_t``).
    lo
        Observer lengths in m (zeros for field points), shape ``(b,)``.
    has_t
        Whether the observers are segments (``True``) or points.
    xs, ys, zs, zi
        Source midpoints in m and image depths ``-z`` in m, shape ``(N,)``.
    tsx, tsy, tsz, tiz
        Source unit tangents and the image tangent ``-t_z``, shape ``(N,)``.
    ls
        Source lengths in m, shape ``(N,)``.
    factor
        Near-pair factor (pairs nearer than ``factor · max(L)`` get 0).
    out
        Output block, shape ``(b, N)``, overwritten.
    """
    for r in range(out.shape[0]):
        lo2 = lo[r] * lo[r]
        for c in range(out.shape[1]):
            dx = xo[r] - xs[c]
            dy = yo[r] - ys[c]
            lim = factor * max(lo[r], ls[c])
            lim2 = lim * lim
            dxy2 = dx * dx + dy * dy
            ls2 = ls[c] * ls[c]
            # direct pair
            dz = zo[r] - zs[c]
            r2 = dxy2 + dz * dz
            qd = 0.0
            if r2 >= lim2:
                u = dx * tsx[c] + dy * tsy[c] + dz * tsz[c]
                q = ls2 * (3.0 * u * u / r2 - 1.0)
                if has_t:
                    v = dx * tox[r] + dy * toy[r] + dz * toz[r]
                    q += lo2 * (3.0 * v * v / r2 - 1.0)
                q /= 24.0 * r2 * np.sqrt(r2)
                qd = q
            # surface image
            dz = zo[r] - zi[c]
            r2 = dxy2 + dz * dz
            qi = 0.0
            if r2 >= lim2:
                u = dx * tsx[c] + dy * tsy[c] + dz * tiz[c]
                q = ls2 * (3.0 * u * u / r2 - 1.0)
                if has_t:
                    v = dx * tox[r] + dy * toy[r] + dz * toz[r]
                    q += lo2 * (3.0 * v * v / r2 - 1.0)
                q /= 24.0 * r2 * np.sqrt(r2)
                qi = q
            out[r, c] = qd + qi
