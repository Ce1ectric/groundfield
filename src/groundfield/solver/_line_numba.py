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
    ro,
    xs,
    ys,
    zs,
    tsx,
    tsy,
    tsz,
    ls,
    rs,
    offsets,
    signs,
    weights,
    wabs,
    variant,
    factor,
    reach_lo,
    reach_hi,
    tol,
    out,
):  # pragma: no cover
    """Far-field term of far images, ``Σ_k w_k · τ_k · Q_k``, in 1/m.

    The numpy path is :func:`groundfield.solver._line_kernel.image_quadrupole_rows`;
    the operations and their order are the same, so both paths give
    bit-identical blocks. Per pair the constants of
    ``Q = [(P_0 + Δz (P_1 + P_2 Δz)) / R² - B] / (24 R³)`` are formed once;
    per image the vertical distance ``Δz = (z_r ∓ z_c) - o`` (sign ±1),
    one division and one square root (audit W3A-01). The distance is
    floored at 1e-100 m² for the near pairs only, which are discarded.
    ``τ_k`` is the taper of the pair's vertical distance in units of its
    longer segment, from 1 at ``reach_lo`` to 0 at ``reach_hi``.

    Parameters
    ----------
    xo, yo, zo, tox, toy, toz, lo
        Observer midpoints (m), unit tangents (zeros for field points) and
        lengths (m; zeros for field points), shape ``(b,)``.
    ro
        Observer wire radii in m (zeros for field points), for the tolerance
        test; the far form of images has no radius term (audit W3A-02).
    xs, ys, zs, tsx, tsy, tsz, ls, rs
        Source midpoints (m), unit tangents, lengths (m) and wire radii (m),
        shape ``(N,)``.
    offsets, signs
        Image offsets (m) and signs (±1), shape ``(m,)``.
    weights, wabs
        Image weights divided by 24 and the absolute weights, shape ``(m,)``.
    variant
        Per image: bit 1 set when the near/tolerance test holds for every
        pair of the block, bit 2 when the taper is one for every pair.
    factor
        Near-pair factor (pairs nearer than ``factor · max(L)`` get 0).
    reach_lo, reach_hi
        The taper of the quadrupole term, in segment lengths.
    tol
        The term is left out for a pair where
        ``|w| (L² + 3a²) / (6 dz²) < tol``.
    out
        Output block, shape ``(b, N)``; the terms are added to it, image by
        image in the given order.
    """
    n = out.shape[1]
    # One buffer for the pair constants of a row: eleven separate arrays of
    # the same size start at the same offset in their pages and evict each
    # other from the L1 cache (55 instead of 5 ns per pair on arm64). The
    # padding keeps the rows off a multiple of 4 KiB.
    pre = np.empty((11, n + 8))
    dxy2 = pre[0]
    dp = pre[1]
    dm = pre[2]
    p0 = pre[3]
    p1p = pre[4]
    p1m = pre[5]
    p2 = pre[6]
    bb = pre[7]
    lim2 = pre[8]
    inv_l = pre[9]
    qtol = pre[10]
    inv_span = 1.0 / (reach_hi - reach_lo)
    inv_tol = 1.0 / (6.0 * tol)
    for r in range(out.shape[0]):
        lo2 = lo[r] * lo[r]
        tzo = toz[r]
        for c in range(n):
            dx = xo[r] - xs[c]
            dy = yo[r] - ys[c]
            dxy2[c] = dx * dx + dy * dy
            hx = dx * tsx[c] + dy * tsy[c]
            gx = dx * tox[r] + dy * toy[r]
            ls2 = ls[c] * ls[c]
            a = ls2 * hx
            bq = lo2 * gx
            p0[c] = 3.0 * (a * hx + bq * gx)
            a_t = a * tsz[c]
            bq_t = bq * tzo
            p1p[c] = 6.0 * (a_t + bq_t)
            p1m[c] = 6.0 * (bq_t - a_t)
            p2[c] = 3.0 * (ls2 * tsz[c] * tsz[c] + lo2 * tzo * tzo)
            a2 = 0.5 * (ro[r] * ro[r] + rs[c] * rs[c])
            bb[c] = ls2 + lo2
            lmax = max(lo[r], ls[c])
            lim = factor * lmax
            lim2[c] = lim * lim
            inv_l[c] = 1.0 / lmax
            qtol[c] = (lmax * lmax + 3.0 * a2) * inv_tol
            dp[c] = zo[r] - zs[c]
            dm[c] = zo[r] + zs[c]
        row = out[r]
        for k in range(offsets.shape[0]):
            off = offsets[k]
            wk = weights[k]
            wa = wabs[k]
            keep_all = (variant[k] & 1) != 0
            full = (variant[k] & 2) != 0
            if signs[k] > 0.0:
                d = dp
                p1 = p1p
            else:
                d = dm
                p1 = p1m
            # One loop per variant, so that the inner loops have no branch.
            if keep_all and full:
                for c in range(n):
                    dz = d[c] - off
                    r2 = dxy2[c] + dz * dz
                    inv2 = 1.0 / max(r2, 1e-100)
                    i3 = inv2 * np.sqrt(inv2)
                    q = ((p2[c] * dz + p1[c]) * dz + p0[c]) * inv2 - bb[c]
                    row[c] += (wk * i3) * q
            elif keep_all:
                for c in range(n):
                    dz = d[c] - off
                    r2 = dxy2[c] + dz * dz
                    inv2 = 1.0 / max(r2, 1e-100)
                    i3 = inv2 * np.sqrt(inv2)
                    q = ((p2[c] * dz + p1[c]) * dz + p0[c]) * inv2 - bb[c]
                    s = min(max((abs(dz) * inv_l[c] - reach_lo) * inv_span, 0.0), 1.0)
                    tau = 1.0 - s * s * (3.0 - 2.0 * s)
                    row[c] += ((wk * tau) * i3) * q
            elif full:
                for c in range(n):
                    dz = d[c] - off
                    r2 = dxy2[c] + dz * dz
                    inv2 = 1.0 / max(r2, 1e-100)
                    i3 = inv2 * np.sqrt(inv2)
                    q = ((p2[c] * dz + p1[c]) * dz + p0[c]) * inv2 - bb[c]
                    keep = (r2 >= lim2[c]) & (wa * qtol[c] >= dz * dz)
                    row[c] += (wk * i3) * q if keep else 0.0
            else:
                for c in range(n):
                    dz = d[c] - off
                    r2 = dxy2[c] + dz * dz
                    inv2 = 1.0 / max(r2, 1e-100)
                    i3 = inv2 * np.sqrt(inv2)
                    q = ((p2[c] * dz + p1[c]) * dz + p0[c]) * inv2 - bb[c]
                    s = min(max((abs(dz) * inv_l[c] - reach_lo) * inv_span, 0.0), 1.0)
                    tau = 1.0 - s * s * (3.0 - 2.0 * s)
                    keep = (r2 >= lim2[c]) & (wa * qtol[c] >= dz * dz)
                    row[c] += ((wk * tau) * i3) * q if keep else 0.0
    return out
