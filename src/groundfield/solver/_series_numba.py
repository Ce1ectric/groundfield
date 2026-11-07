"""Optional numba kernel of the two-layer image series (private).

Imported only by :mod:`groundfield.solver._kernels`, and only when
``numba`` is installed (extra ``fast``). The kernel accumulates the
same image terms in the same order and with the same IEEE operations
as the numpy block :func:`groundfield.solver._kernels._series_block_numpy`
— no fused multiply-add, no reciprocal approximation — so both paths
give bit-identical matrices; numba only removes the temporaries and
vectorises the loop over sources (about 3× faster than the numpy
block on one core for 1 200 segments and 148 images, review pass 10,
10.58).

The fast-math flags ``nnan``, ``ninf`` and ``nsz`` let the compiler
vectorise the clamp ``max(r, d)``; they do not change any result for
finite inputs (distances are positive, weights finite).
"""

from __future__ import annotations

import numba
import numpy as np

__all__ = ["series_rows"]

_FLAGS = {"nnan", "ninf", "nsz"}


@numba.njit(
    fastmath=_FLAGS, nogil=True, boundscheck=False, error_model="numpy", cache=True
)
def _accumulate_image(dxy, z_obs, z_img, weight, min_distance, acc):  # pragma: no cover
    """``acc += weight · 1/max(sqrt(dxy + (z_obs - z_img)²), d)`` (one row)."""
    for j in range(acc.shape[0]):
        d = z_obs - z_img[j]
        r = np.sqrt(dxy[j] + d * d)
        r = max(r, min_distance)
        acc[j] += weight * (1.0 / r)


@numba.njit(
    fastmath=_FLAGS, nogil=True, boundscheck=False, error_model="numpy", cache=True
)
def series_rows(
    xo, yo, zo, xs, ys, z_images, weights, min_distance, out
):  # pragma: no cover
    """Series block ``out[i, j] = Σ_k w_k / max(r_ijk, d)``.

    Parameters
    ----------
    xo, yo, zo
        Observer coordinates in m, shape ``(b,)``.
    xs, ys
        Horizontal source coordinates in m, shape ``(N,)``.
    z_images
        Image depths in m, shape ``(m, N)``, canonical image order.
    weights
        Image weights ``K^n``, shape ``(m,)``.
    min_distance
        Distance clamp in m.
    out
        Output block, shape ``(b, N)``; overwritten.
    """
    n = xs.shape[0]
    dxy = np.empty(n)
    for i in range(out.shape[0]):
        xi = xo[i]
        yi = yo[i]
        for j in range(n):
            a = xi - xs[j]
            c = yi - ys[j]
            dxy[j] = a * a + c * c
        row = out[i]
        row[:] = 0.0
        for k in range(z_images.shape[0]):
            _accumulate_image(dxy, zo[i], z_images[k], weights[k], min_distance, row)
    return out
