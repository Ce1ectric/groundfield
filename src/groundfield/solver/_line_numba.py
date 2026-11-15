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

__all__ = ["image_quadrupole_rows", "quadrupole_rows"]


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
    ro,
    has_r,
    xs,
    ys,
    zs,
    zi,
    tsx,
    tsy,
    tsz,
    tiz,
    ls,
    rs,
    factor,
    out,
):  # pragma: no cover
    """Far-field correction of segments plus surface images, in 1/m.

    ``out[r, c] = Q(obs r, src c) + Q(obs r, image of src c)`` with
    ``Q = [L_c²(3u_c²/R² - 1) + L_r²(3u_r²/R² - 1) - 12 a²] / (24 R³)``,
    ``a² = (a_r² + a_c²)/2`` (with ``has_r``), for
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
    ro, has_r
        Observer wire radii in m and whether the radius term applies.
    xs, ys, zs, zi
        Source midpoints in m and image depths ``-z`` in m, shape ``(N,)``.
    tsx, tsy, tsz, tiz
        Source unit tangents and the image tangent ``-t_z``, shape ``(N,)``.
    ls, rs
        Source lengths and wire radii in m, shape ``(N,)``.
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
                    if has_r:
                        q -= 12.0 * (0.5 * (ro[r] * ro[r] + rs[c] * rs[c]))
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
                    if has_r:
                        q -= 12.0 * (0.5 * (ro[r] * ro[r] + rs[c] * rs[c]))
                q /= 24.0 * r2 * np.sqrt(r2)
                qi = q
            out[r, c] = qd + qi


#: The flags of the series kernel (:mod:`groundfield.solver._series_numba`):
#: no NaN, infinity or signed-zero semantics, which lets the compiler
#: vectorise the selection of far pairs; no reassociation, no contraction
#: (fused multiply-add) and no approximate reciprocal, so every result for
#: finite inputs is the IEEE result of the numpy path.
_FLAGS = {"nnan", "ninf", "nsz"}


@numba.njit(
    fastmath=_FLAGS, nogil=True, boundscheck=False, error_model="numpy", cache=True
)
def image_quadrupole_rows(
    xo,
    yo,
    zo,
    tox,
    toy,
    toz,
    lo,
    has_t,
    ro,
    has_r,
    xs,
    ys,
    tsx,
    tsy,
    ls,
    rs,
    z_img,
    tz_img,
    weights,
    factor,
    reach_lo,
    reach_hi,
    tol,
    out,
):  # pragma: no cover
    """Far-field term of far images, ``Σ_k w_k · τ_k · Q_k``, in 1/m.

    The numpy path is :func:`groundfield.solver._line_kernel.image_quadrupole_rows`;
    the operations and their order are the same, so both paths give
    bit-identical blocks. The inner loop is branch-free (``Q`` is formed for
    every pair and selected for the far ones; the distance is floored at
    1e-100 m² for the near ones only, which are discarded). ``τ_k`` is the
    taper of the pair's vertical distance in units of its longer segment,
    from 1 at ``reach_lo`` to 0 at ``reach_hi``.

    Parameters
    ----------
    xo, yo, zo, tox, toy, toz, lo, has_t, ro, has_r
        Observers as in :func:`quadrupole_rows`.
    xs, ys, tsx, tsy, ls, rs
        Horizontal source coordinates (m), horizontal tangent components,
        lengths (m) and wire radii (m), shape ``(N,)``.
    z_img, tz_img
        Image depths (m) and vertical image tangents, shape ``(m, N)``.
    weights
        Image weights, shape ``(m,)``.
    factor
        Near-pair factor (pairs nearer than ``factor · max(L)`` get 0).
    reach_lo, reach_hi
        The taper of the quadrupole term, in segment lengths.
    tol
        The term is left out for a pair where
        ``|w| (L² + 3a²) / (6 dz²) < tol``.
    out
        Output block, shape ``(b, N)``, overwritten.
    """
    n = out.shape[1]
    dxy2 = np.empty(n)
    hx = np.empty(n)
    gx = np.empty(n)
    ls2 = np.empty(n)
    a2t = np.empty(n)
    lim2 = np.empty(n)
    inv_l = np.empty(n)
    qtol = np.empty(n)
    inv_span = 1.0 / (reach_hi - reach_lo)
    inv_tol = 1.0 / (6.0 * tol)
    for r in range(out.shape[0]):
        lo2 = lo[r] * lo[r]
        tzo = toz[r]
        for c in range(n):
            dx = xo[r] - xs[c]
            dy = yo[r] - ys[c]
            dxy2[c] = dx * dx + dy * dy
            hx[c] = dx * tsx[c] + dy * tsy[c]
            gx[c] = dx * tox[r] + dy * toy[r]
            ls2[c] = ls[c] * ls[c]
            a2t[c] = 12.0 * (0.5 * (ro[r] * ro[r] + rs[c] * rs[c]))
            lmax = max(lo[r], ls[c])
            lim = factor * lmax
            lim2[c] = lim * lim
            inv_l[c] = 1.0 / lmax
            qtol[c] = (
                lmax * lmax + 3.0 * (0.5 * (ro[r] * ro[r] + rs[c] * rs[c]))
            ) * inv_tol
            out[r, c] = 0.0
        row = out[r]
        zr = zo[r]
        for k in range(z_img.shape[0]):
            wk = weights[k]
            wa = abs(wk)
            zk = z_img[k]
            tk = tz_img[k]
            # One loop per observer kind, so that the inner loop has no
            # branch and vectorises.
            if has_t and has_r:
                for c in range(n):
                    dz = zr - zk[c]
                    r2 = dxy2[c] + dz * dz
                    rq = max(r2, 1e-100)
                    u = hx[c] + dz * tk[c]
                    v = gx[c] + dz * tzo
                    q = ls2[c] * (3.0 * u * u / rq - 1.0)
                    q += lo2 * (3.0 * v * v / rq - 1.0)
                    q -= a2t[c]
                    q /= 24.0 * rq * np.sqrt(rq)
                    s = min(max((abs(dz) * inv_l[c] - reach_lo) * inv_span, 0.0), 1.0)
                    tau = 1.0 - s * s * (3.0 - 2.0 * s)
                    keep = (r2 >= lim2[c]) & (wa * qtol[c] >= dz * dz)
                    row[c] += wk * (q * tau) if keep else 0.0
            elif has_t:
                for c in range(n):
                    dz = zr - zk[c]
                    r2 = dxy2[c] + dz * dz
                    rq = max(r2, 1e-100)
                    u = hx[c] + dz * tk[c]
                    v = gx[c] + dz * tzo
                    q = ls2[c] * (3.0 * u * u / rq - 1.0)
                    q += lo2 * (3.0 * v * v / rq - 1.0)
                    q /= 24.0 * rq * np.sqrt(rq)
                    s = min(max((abs(dz) * inv_l[c] - reach_lo) * inv_span, 0.0), 1.0)
                    tau = 1.0 - s * s * (3.0 - 2.0 * s)
                    keep = (r2 >= lim2[c]) & (wa * qtol[c] >= dz * dz)
                    row[c] += wk * (q * tau) if keep else 0.0
            else:
                for c in range(n):
                    dz = zr - zk[c]
                    r2 = dxy2[c] + dz * dz
                    rq = max(r2, 1e-100)
                    u = hx[c] + dz * tk[c]
                    q = ls2[c] * (3.0 * u * u / rq - 1.0)
                    q /= 24.0 * rq * np.sqrt(rq)
                    s = min(max((abs(dz) * inv_l[c] - reach_lo) * inv_span, 0.0), 1.0)
                    tau = 1.0 - s * s * (3.0 - 2.0 * s)
                    keep = (r2 >= lim2[c]) & (wa * qtol[c] >= dz * dz)
                    row[c] += wk * (q * tau) if keep else 0.0
    return out
