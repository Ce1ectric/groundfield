"""Reaction-matrix and field kernels of the image family (private).

One module holds every Green's-function kernel of the closed-form
family (review pass 10, 10.75): the half-space point kernel
:math:`1/r + 1/r'` with its analytical self-term, the Tagg/Sunde series
of :class:`~groundfield.solver._series.ImageSeries`, and the field
evaluations used after the solve.

Memory and bit-identity (10.56, 10.57, 10.62)
--------------------------------------------
Up to v0.15.0 every kernel built ``(N, N, 3)`` float64 difference
tensors — a peak of 96 bytes per segment pair (1.25 GB for an AP1
30-house network, ``MemoryError`` at 80 houses). The kernels here are
built component-wise, row block by row block, with the same
floating-point operations in the same order, so every matrix entry is
bit-identical to v0.15.0.

The products with the matrix are a different matter: OpenBLAS sums a
row of ``A @ X`` in an order that depends on the block it belongs to,
so a row-chunked product is *not* bit-identical to the full one. The
operators therefore materialise the reaction matrix once per solve
whenever it fits :data:`MATRIX_BUDGET_BYTES` and apply it with exactly
the BLAS calls of v0.15.0 — 8 instead of 96 bytes per pair, one
assembly per solve (the second solve-side evaluation of v0.15.0 reuses
the stored matrix) and bit-identical results. Beyond the budget they
switch to row-block evaluation with ``O(N · block)`` memory; results
then differ from the materialised evaluation by BLAS round-off only
(about ``1e-15`` relative). Field evaluations follow the same rule
with :data:`FIELD_PIECE_BYTES`.

Units
-----
Kernels are returned *unscaled*, in 1/m; a potential in V follows as
:math:`\\rho/(4\\pi)` times the kernel applied to currents in A.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from groundfield.solver._guards import _MIN_DISTANCE, _require_image_separation
from groundfield.solver._line_kernel import (
    FarImages,
    LineSegments,
    SparseCorrection,
    add_symmetric,
    far_image_rows,
    far_images,
    field_corrections,
    field_quadrupole_rows,
    point_self_term,
    reaction_corrections,
    reaction_quadrupole_rows,
    series_corrections,
    series_field_corrections,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from groundfield.solver._series import ImageSeries

__all__ = [
    "BLOCK_BYTES",
    "DEFAULT_KERNEL",
    "FIELD_PIECE_BYTES",
    "MATRIX_BUDGET_BYTES",
    "NUMBA_MIN_WORK",
    "USE_NUMBA",
]

#: Reaction matrices up to this size in bytes are materialised once per
#: solve and applied with the BLAS calls of v0.15.0 (bit-identical
#: results). Larger problems are evaluated row block by row block.
MATRIX_BUDGET_BYTES: int = 1 << 30

#: Size in bytes of one ``(b, N)`` float64 work array of the block loops.
BLOCK_BYTES: int = 1 << 22

#: Largest field kernel ``(M, N)`` in bytes that a field evaluation
#: builds in one piece (bit-identical to v0.15.0); larger evaluations
#: run in row blocks of :data:`BLOCK_BYTES`.
FIELD_PIECE_BYTES: int = 1 << 24

#: Use the optional numba kernel of the two-layer series (extra ``fast``)
#: when ``numba`` is importable. Both paths give bit-identical matrices.
USE_NUMBA: bool = True

#: Smallest series workload (rows × sources × images) handed to numba;
#: below it the one-off compilation (about 1 s) would not pay off.
NUMBA_MIN_WORK: int = 50_000_000

#: Kernel an engine requests when it leaves the choice to the backend
#: (``Engine.kernel = None``, the default since 2026.11, review pass 10,
#: 10.28): the exact line-source kernel wherever a backend implements it
#: (the image family), the backend's own kernel elsewhere.
DEFAULT_KERNEL = "line"


def _block_rows(n_cols: int) -> int:
    """Rows per block so that one ``(rows, n_cols)`` array fits :data:`BLOCK_BYTES`."""
    rows = BLOCK_BYTES // (8 * max(int(n_cols), 1))
    return max(8, rows - rows % 8)


def _xyz(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous coordinate columns of an ``(N, 3)`` point array."""
    return (
        np.ascontiguousarray(points[:, 0]),
        np.ascontiguousarray(points[:, 1]),
        np.ascontiguousarray(points[:, 2]),
    )


# ---------------------------------------------------------------------
# Half-space point kernel
# ---------------------------------------------------------------------


def _half_space_block(
    obs: tuple[np.ndarray, np.ndarray, np.ndarray],
    src: tuple[np.ndarray, np.ndarray, np.ndarray],
    min_distance: float,
    *,
    out: np.ndarray | None = None,
    count_close: bool = False,
) -> tuple[np.ndarray, int]:
    """Block ``1/max(r, d) + 1/max(r', d)`` between observers and sources.

    ``r`` is the distance to the source, ``r'`` the distance to its
    mirror image in the soil surface (depth ``-z``). The distances are
    formed component-wise as ``sqrt((dx² + dy²) + dz²)`` — the operation
    order of ``np.linalg.norm`` over an ``(..., 3)`` difference tensor —
    so every entry is bit-identical to the v0.15.0 kernels.

    Parameters
    ----------
    obs, src
        Coordinate columns ``(x, y, z)`` in m of the observers
        (``b`` each) and the sources (``N`` each).
    min_distance
        Clamp of both distances in m.
    out
        Optional ``(b, N)`` output array.
    count_close
        Also count the pairs with ``r < min_distance`` (before clamping).

    Returns
    -------
    tuple[numpy.ndarray, int]
        The ``(b, N)`` block in 1/m and the number of close pairs.
    """
    xo, yo, zo = obs
    xs, ys, zs = src
    dxy = np.subtract.outer(xo, xs)
    np.multiply(dxy, dxy, out=dxy)
    t = np.subtract.outer(yo, ys) if out is None else np.subtract.outer(yo, ys, out=out)
    np.multiply(t, t, out=t)
    np.add(dxy, t, out=dxy)  # dx² + dy²
    np.subtract.outer(zo, zs, out=t)
    np.multiply(t, t, out=t)
    np.add(dxy, t, out=t)
    np.sqrt(t, out=t)  # r
    n_close = int(np.count_nonzero(t < min_distance)) if count_close else 0
    np.maximum(t, min_distance, out=t)
    np.divide(1.0, t, out=t)  # 1/r
    zz = np.add.outer(zo, zs)  # z_o - (-z_s): image in the surface
    np.multiply(zz, zz, out=zz)
    np.add(dxy, zz, out=dxy)
    np.sqrt(dxy, out=dxy)  # r'
    np.maximum(dxy, min_distance, out=dxy)
    np.divide(1.0, dxy, out=dxy)  # 1/r'
    np.add(t, dxy, out=t)
    return t, n_close


def _warn_clamped_pairs(n_clamped: int, stacklevel: int) -> None:
    """Warn about distinct segments within the singularity clamp."""
    if n_clamped > 0:
        warnings.warn(
            f"_self_corrected_kernel: {n_clamped} off-diagonal segment "
            f"pair(s) are closer than the {_MIN_DISTANCE * 1e3:.0f} mm "
            "singularity clamp — distinct electrodes/segments overlap "
            "and their mutual coupling is saturated at the clamp. "
            "Check the geometry for coincident conductors.",
            UserWarning,
            stacklevel=stacklevel + 1,
        )


def _half_space_diagonal(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    *,
    owner_names: Sequence[str] | None,
) -> np.ndarray:
    """Self-terms of the half-space reaction matrix in 1/m.

    The direct part is the analytical line self-potential
    ``2 ln(L/a)/L``, replaced by the Galerkin self term for segments
    shorter than 2.99 wire radii
    (:func:`~groundfield.solver._line_kernel.point_self_term`, review
    pass 10, 10.32); the self-image part is the point image at distance
    ``2z`` (the ``L << 4z`` limit of the image line, see
    :class:`~groundfield.solver._guards.ShallowSegmentWarning`).

    Raises
    ------
    ValueError
        Through :func:`_require_image_separation` for segments at or
        above the surface or with a touching mirror image.
    """
    # phi_self_direct = rho · I / (2π·L) · ln(L/a)
    #   ⇒ K_ii (direct part) = 2·ln(L/a) / L  (Galerkin form for L < 2.99 a)
    diag_direct = point_self_term(seg_lengths, wire_radii)
    # Guard (audit 2026-07-28, F26/F29; z <= 0 since 0.15.0): the
    # point-image form diverges as z -> 0 and used to be clamped at
    # ``_MIN_DISTANCE``, which returned a mesh-dependent, far too large
    # impedance. A segment whose mirror image touches the conductor, or
    # that sits in the air, is a hard error for every backend.
    _require_image_separation(
        seg_points,
        wire_radii,
        where="_self_corrected_kernel",
        owner_names=owner_names,
    )
    diag_image = 1.0 / (2.0 * np.abs(seg_points[:, 2]))
    return diag_direct + diag_image


def _half_space_matrix(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    *,
    owner_names: Sequence[str] | None = None,
    stacklevel: int = 2,
) -> np.ndarray:
    """Unscaled half-space reaction matrix ``K0`` (``N × N``, 1/m).

    Off-diagonal: point source plus surface image, clamped at
    ``_MIN_DISTANCE``; diagonal: :func:`_half_space_diagonal`. Built in
    row blocks without ``(N, N, 3)`` temporaries; entries bit-identical
    to v0.15.0.
    """
    n = seg_points.shape[0]
    K = np.empty((n, n))
    xyz = _xyz(seg_points)
    rows = _block_rows(n)
    n_close = 0
    for i0 in range(0, n, rows):
        i1 = min(n, i0 + rows)
        obs = (xyz[0][i0:i1], xyz[1][i0:i1], xyz[2][i0:i1])
        _, nc = _half_space_block(
            obs, xyz, _MIN_DISTANCE, out=K[i0:i1], count_close=True
        )
        n_close += nc
    # The 1 mm clamp is meant for *field* evaluations; inside the
    # reaction matrix a clamped off-diagonal pair means two distinct
    # segments (nearly) coincide — surface that instead of hiding it.
    _warn_clamped_pairs((n_close - n) // 2, stacklevel + 1)  # diag is 0
    np.fill_diagonal(
        K,
        _half_space_diagonal(
            seg_points, seg_lengths, wire_radii, owner_names=owner_names
        ),
    )
    return K


class HalfSpaceKernel:
    """Reaction operator of the homogeneous half-space (point kernel).

    Calling the operator returns the segment-midpoint potentials
    ``rho/(4π) · K0 @ currents`` in V for segment currents in A
    (``currents`` of shape ``(N,)`` or ``(N, k)``). Within
    :data:`MATRIX_BUDGET_BYTES` the matrix ``K0`` is assembled once per
    geometry and reused by later calls with the same segment arrays
    (one assembly per solve, 10.57); beyond the budget it is evaluated
    in row blocks.

    Parameters
    ----------
    rho
        Soil resistivity in Ω·m.
    owner_names
        Optional per-segment owner labels for the depth-guard message.
    line
        Segment geometry for the exact line-source kernel
        (``Engine(kernel="line")``, :mod:`groundfield.solver._line_kernel`);
        ``None`` (default) keeps the point kernel.

    Attributes
    ----------
    assemblies
        Number of full matrix evaluations so far.
    """

    def __init__(
        self,
        rho: float,
        *,
        owner_names: Sequence[str] | None = None,
        line: LineSegments | None = None,
    ):
        self.rho = rho
        self.pref = rho / (4.0 * np.pi)
        self.owner_names = owner_names
        self.line = line
        self.assemblies = 0
        self._geometry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._K0: np.ndarray | None = None
        self._line_corrections: SparseCorrection | None = None

    def _corrections(self) -> SparseCorrection:
        """Sparse line-kernel corrections (built once)."""
        if self._line_corrections is None:
            self._line_corrections = reaction_corrections(self.line, _MIN_DISTANCE)
        return self._line_corrections

    def _apply_line(self, rows: np.ndarray, i0: int) -> None:
        """Turn point-kernel rows ``i0 …`` into line-kernel rows (in place)."""
        rows += reaction_quadrupole_rows(self.line, i0, i0 + rows.shape[0])
        self._corrections().add_to_rows(rows, i0)

    @staticmethod
    def fits(n_segments: int) -> bool:
        """Whether an ``N × N`` matrix fits :data:`MATRIX_BUDGET_BYTES`."""
        return 8 * n_segments * n_segments <= MATRIX_BUDGET_BYTES

    @property
    def materialised(self) -> bool:
        """Whether the operator holds its matrix (repeat calls are cheap)."""
        return self._K0 is not None

    def _same_geometry(self, seg_points, seg_lengths, wire_radii) -> bool:
        g = self._geometry
        return (
            g is not None
            and g[0] is seg_points
            and g[1] is seg_lengths
            and g[2] is wire_radii
        )

    def unscaled_matrix(
        self, seg_points: np.ndarray, seg_lengths: np.ndarray, wire_radii: np.ndarray
    ) -> np.ndarray:
        """``K0`` in 1/m, assembled once per geometry and kept."""
        if self._K0 is None or not self._same_geometry(
            seg_points, seg_lengths, wire_radii
        ):
            K0 = _half_space_matrix(
                seg_points,
                seg_lengths,
                wire_radii,
                owner_names=self.owner_names,
                stacklevel=4,
            )
            if self.line is not None:
                rows = _block_rows(K0.shape[0])
                for i0 in range(0, K0.shape[0], rows):
                    self._apply_line(K0[i0 : i0 + rows], i0)
            self._K0 = K0
            self._geometry = (seg_points, seg_lengths, wire_radii)
            self.assemblies += 1
        return self._K0

    def matrix(
        self, seg_points: np.ndarray, seg_lengths: np.ndarray, wire_radii: np.ndarray
    ) -> np.ndarray:
        """Reaction matrix ``rho/(4π) · K0`` in Ω (new array).

        Equal, bit for bit, to applying the v0.15.0 kernel to the
        identity matrix, without the ``O(N³)`` product (10.59).
        """
        return self.pref * self.unscaled_matrix(seg_points, seg_lengths, wire_radii)

    def __call__(
        self,
        seg_points: np.ndarray,
        seg_lengths: np.ndarray,
        wire_radii: np.ndarray,
        currents: np.ndarray,
    ) -> np.ndarray:
        n = seg_points.shape[0]
        if self.fits(n):
            K0 = self.unscaled_matrix(seg_points, seg_lengths, wire_radii)
            return self.pref * (K0 @ currents)
        return self.pref * self._blocked_product(
            seg_points, seg_lengths, wire_radii, currents
        )

    def _blocked_product(
        self, seg_points, seg_lengths, wire_radii, currents
    ) -> np.ndarray:
        """``K0 @ currents`` row block by row block (beyond the budget)."""
        n = seg_points.shape[0]
        diag = _half_space_diagonal(
            seg_points, seg_lengths, wire_radii, owner_names=self.owner_names
        )
        cur = np.asarray(currents)
        out = np.empty((n,) + cur.shape[1:], dtype=np.result_type(cur, float))
        xyz = _xyz(seg_points)
        rows = _block_rows(n)
        n_close = 0
        for i0 in range(0, n, rows):
            i1 = min(n, i0 + rows)
            obs = (xyz[0][i0:i1], xyz[1][i0:i1], xyz[2][i0:i1])
            blk, nc = _half_space_block(obs, xyz, _MIN_DISTANCE, count_close=True)
            n_close += nc
            idx = np.arange(i0, i1)
            blk[idx - i0, idx] = diag[i0:i1]
            if self.line is not None:
                self._apply_line(blk, i0)
            out[i0:i1] = blk @ cur
        _warn_clamped_pairs((n_close - n) // 2, 3)
        self.assemblies += 1
        return out


def _self_corrected_kernel(
    seg_points: np.ndarray,  # (N, 3)
    seg_lengths: np.ndarray,  # (N,)
    wire_radii: np.ndarray,  # (N,)
    currents: np.ndarray,  # (N,) or (N, k)
    rho: float,
    *,
    owner_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Evaluation **at the segment midpoints** with proper self-action.

    The diagonal (segment onto itself) uses the analytical line
    self-potential; off-diagonal entries fall back to the point-source
    approximation 1/r + 1/r_image. The diagonal *image* term is the
    point form $1/(2z)$, which is the $L \\ll 4 z$ limit of the exact
    image-line term $(2/L)\\,\\operatorname{arsinh}(L/(4z))$ — see
    :func:`_check_segment_depths` for the shallow-electrode regime
    where that distinction matters.

    This is the reaction-matrix kernel shared by the whole closed-form
    family: ``image``, ``image_2layer``, ``mom``, ``bem``, ``cim``,
    ``mom_sommerfeld`` and ``mutual`` all route their homogeneous
    (direct + free-surface-image) part through it (since review pass 10
    through :class:`HalfSpaceKernel`), which is why both depth guards of
    :func:`_require_image_separation` are enforced here rather than in
    the individual ``solve_*`` entry points.

    Parameters
    ----------
    seg_points, seg_lengths, wire_radii
        Segment midpoints (m), lengths (m) and wire radii (m).
    currents
        Segment currents in A, shape ``(N,)`` or ``(N, k)``.
    rho
        Resistivity in Ω·m.
    owner_names
        Optional per-segment owner labels (already formatted, e.g.
        ``"electrode 'g1'"``) used only to make the guard's error
        message name what the user wrote. Passing them never changes a
        returned number.

    Returns
    -------
    phi : np.ndarray, shape (N,) or (N, k)
        Potential at the segment midpoints in V.

    Raises
    ------
    ValueError
        If any segment midpoint is at or above the soil surface
        (``z <= 0``) — the kernel mirrors ``|z|``, so a conductor in
        air would otherwise be solved as if it were buried — or lies
        in the soil-surface plane (``2*|z| <= wire_radius``), where
        source and mirror image coincide and the self-image term
        diverges (:func:`_require_image_separation`).
    """
    return HalfSpaceKernel(rho, owner_names=owner_names)(
        seg_points, seg_lengths, wire_radii, currents
    )


# ---------------------------------------------------------------------
# Two-layer image series (n >= 1)
# ---------------------------------------------------------------------


def _series_images(series: ImageSeries) -> list[tuple[float, float, float]]:
    """``(weight, offset, sign)`` of every image ``n >= 1``, canonical order."""
    return [
        (term.weight, term.offset, term.sign)
        for quad in series.iter_layer_terms()
        for term in quad
    ]


def _image_depths(
    series: ImageSeries, z_src: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    """``(weight, image depths)`` of every ``n >= 1`` image, canonical order."""
    return [
        (term.weight, term.offset + term.sign * z_src)
        for quad in series.iter_layer_terms()
        for term in quad
    ]


def _numba_series_rows():
    """The compiled numba series kernel, or ``None`` without numba."""
    if not USE_NUMBA:
        return None
    try:
        from groundfield.solver._series_numba import series_rows
    except ImportError:  # numba is an optional extra
        return None
    return series_rows


def _series_block(
    obs: tuple[np.ndarray, np.ndarray, np.ndarray],
    src_xy: tuple[np.ndarray, np.ndarray],
    images: list[tuple[float, np.ndarray]],
    min_distance: float,
    out: np.ndarray,
    *,
    use_numba: bool,
) -> np.ndarray:
    """Series block (numba when requested and available, else numpy)."""
    if use_numba and images:
        rows_fn = _numba_series_rows()
        if rows_fn is not None:
            weights = np.array([w for w, _ in images], dtype=float)
            depths = np.ascontiguousarray(np.array([z for _, z in images], dtype=float))
            return rows_fn(
                obs[0],
                obs[1],
                obs[2],
                src_xy[0],
                src_xy[1],
                depths,
                weights,
                float(min_distance),
                out,
            )
    return _series_block_numpy(obs, src_xy, images, min_distance, out)


def _want_numba(n_rows: int, n_cols: int, n_images: int) -> bool:
    """Whether a series workload is large enough for the numba path."""
    return USE_NUMBA and n_rows * n_cols * n_images >= NUMBA_MIN_WORK


def _series_block_numpy(
    obs: tuple[np.ndarray, np.ndarray, np.ndarray],
    src_xy: tuple[np.ndarray, np.ndarray],
    images: list[tuple[float, np.ndarray]],
    min_distance: float,
    out: np.ndarray,
) -> np.ndarray:
    """``out = Σ_images K^n · (1/max(r, d))`` for one row block (numpy).

    The images are accumulated entry by entry in the canonical order of
    :class:`~groundfield.solver._series.ImageSeries`, with the operations
    of the v0.15.0 per-image loop — the block equals that loop applied to
    the identity matrix, bit for bit.
    """
    xo, yo, zo = obs
    xs, ys = src_xy
    delta_sq = np.subtract.outer(xo, xs)
    np.multiply(delta_sq, delta_sq, out=delta_sq)
    t = np.subtract.outer(yo, ys)
    np.multiply(t, t, out=t)
    np.add(delta_sq, t, out=delta_sq)  # dx² + dy²
    out[...] = 0.0
    zo_col = zo[:, None]
    for weight, z_img in images:
        np.subtract(zo_col, z_img[None, :], out=t)
        np.multiply(t, t, out=t)
        np.add(delta_sq, t, out=t)
        np.sqrt(t, out=t)
        np.maximum(t, min_distance, out=t)
        np.divide(1.0, t, out=t)
        np.multiply(weight, t, out=t)
        np.add(out, t, out=out)
    return out


def _series_matrix(
    seg_points: np.ndarray, series: ImageSeries, *, out: np.ndarray | None = None
) -> np.ndarray:
    """Unscaled series reaction matrix ``S`` (``N × N``, 1/m).

    ``S[i, j] = Σ_{n ≥ 1} K^n Σ_4 1/max(r_n, d)`` between segment ``i``
    and the images of segment ``j`` (point sources). Equal, bit for bit,
    to the v0.15.0 per-image loop applied to the identity matrix.
    """
    n = seg_points.shape[0]
    S = np.empty((n, n)) if out is None else out
    x, y, z = _xyz(seg_points)
    images = _image_depths(series, z)
    use_numba = _want_numba(n, n, len(images))
    rows = _block_rows(n)
    for i0 in range(0, n, rows):
        i1 = min(n, i0 + rows)
        _series_block(
            (x[i0:i1], y[i0:i1], z[i0:i1]),
            (x, y),
            images,
            _MIN_DISTANCE,
            S[i0:i1],
            use_numba=use_numba,
        )
    return S


class TwoLayerSeriesKernel:
    """Reaction operator of the two-layer soil inside the upper layer.

    ``phi = rho_1/(4π) · (K0 @ I) + (S @ I) · rho_1/(4π)``: the half-space
    part :class:`HalfSpaceKernel` with :math:`\\rho_1`, plus the Tagg/Sunde
    images (``n >= 1``), at least :math:`2(h_1 - z)` away from every
    segment. With the point kernel they are point sources; with the line
    kernel the near image pairs are Galerkin integrals (transverse distance,
    audit W3A-02) and the far ones carry the quadrupole term of the segment
    average (``far_field = "quadrupole"``, 10.28) or stay point sources
    (``"point"``).

    The series matrix ``S`` is accumulated image by image *before* the
    product with the currents (10.58; v0.15.0 multiplied every image
    separately), which changes the result by BLAS round-off only. Within
    :data:`MATRIX_BUDGET_BYTES` (``16 N²`` bytes for ``K0`` and ``S``)
    both matrices are kept for the solve; beyond it they are evaluated
    in row blocks.

    Parameters
    ----------
    rho_1
        Upper-layer resistivity in Ω·m.
    series
        The image series.
    line
        Segment geometry for the exact line-source kernel; ``None``
        (default) keeps the point kernel.
    far_field
        Form of the far images with the line kernel
        (``Engine.image_far_field``): ``"quadrupole"`` (default) or
        ``"point"``.
    """

    def __init__(
        self,
        rho_1: float,
        series: ImageSeries,
        *,
        line: LineSegments | None = None,
        far_field: str = "quadrupole",
    ):
        self.rho_1 = rho_1
        self.quadrupole = far_field == "quadrupole"
        self.pref = rho_1 / (4.0 * np.pi)
        self.series = series
        self.line = line
        self.half_space = HalfSpaceKernel(rho_1, line=line)
        self._S: np.ndarray | None = None
        self._S_points: np.ndarray | None = None
        self._series_corrections: SparseCorrection | None = None
        self._far: FarImages | None = None

    def _images(self) -> list[tuple[float, float, float]]:
        """``(weight, offset, sign)`` of every image ``n >= 1``, canonical order."""
        return _series_images(self.series)

    def _line_series(self) -> SparseCorrection:
        """Exact-minus-far-form corrections of near series images (line kernel)."""
        if self._series_corrections is None:
            self._series_corrections = series_corrections(
                self.line, self._images(), _MIN_DISTANCE, quadrupole=self.quadrupole
            )
        return self._series_corrections

    def _apply_line(self, rows: np.ndarray, i0: int) -> None:
        """Point-image rows ``i0 …`` to line-kernel rows, in place.

        The far images get the quadrupole term of the segment average
        (:func:`~groundfield.solver._line_kernel.image_quadrupole_rows`, wave 3
        of the 2026.11 work, 10.28), the near ones the sparse corrections.
        """
        if self.quadrupole:
            rows += self._far_rows(i0, i0 + rows.shape[0])
        self._line_series().add_to_rows(rows, i0)

    def _far_rows(self, i0: int, i1: int, c0: int = 0) -> np.ndarray:
        """Far-image terms of rows ``i0:i1`` against the segments ``c0 …``."""
        line = self.line
        sl = slice(i0, i1)
        return far_image_rows(
            line.mid[sl],
            line.tangent[sl],
            line.length[sl],
            line if c0 == 0 else line.subset(slice(c0, None)),
            self._far_images(),
            obs_radius=line.radius[sl],
        )

    def _far_images(self) -> FarImages:
        """Far images with the pairs that need them (built once)."""
        if self._far is None:
            self._far = far_images(self.line.mid[:, 2], self.line, self._images())
        return self._far

    def _series(self, seg_points: np.ndarray) -> np.ndarray:
        """``S`` (point images plus the line-kernel terms of the images)."""
        S = _series_matrix(seg_points, self.series)
        if self.line is not None:
            if self.quadrupole and self._far_images():
                # The far-image terms are symmetric: the upper half, mirrored
                # (audit W3A-01).
                add_symmetric(
                    S,
                    lambda i0, i1: self._far_rows(i0, i1, c0=i0),
                    _block_rows(S.shape[0]),
                )
            self._line_series().add_to_rows(S, 0)
        return S

    @property
    def assemblies(self) -> int:
        """Number of full matrix evaluations so far."""
        return self.half_space.assemblies

    @property
    def materialised(self) -> bool:
        """Whether the operator holds its matrices."""
        return self.half_space.materialised and self._S is not None

    @staticmethod
    def fits(n_segments: int) -> bool:
        """Whether ``K0`` and ``S`` together fit :data:`MATRIX_BUDGET_BYTES`."""
        return 16 * n_segments * n_segments <= MATRIX_BUDGET_BYTES

    def series_matrix(self, seg_points: np.ndarray) -> np.ndarray:
        """``S`` in 1/m, assembled once per geometry and kept."""
        if self._S is None or self._S_points is not seg_points:
            self._S = self._series(seg_points)
            self._S_points = seg_points
        return self._S

    def matrix(
        self, seg_points: np.ndarray, seg_lengths: np.ndarray, wire_radii: np.ndarray
    ) -> np.ndarray:
        """Reaction matrix in Ω (new array), bit-identical to v0.15.0's ``kernel(eye)``.

        Within :data:`MATRIX_BUDGET_BYTES` the series matrix is kept for
        later calls of the operator on the same geometry (``mom`` assembles
        the matrix and then applies the operator: the series was built
        twice per solve before).
        """
        M = self.half_space.matrix(seg_points, seg_lengths, wire_radii)
        if not self.fits(seg_points.shape[0]):
            S = self._series(seg_points)
            S *= self.pref
            M += S
            return M
        S = self.series_matrix(seg_points)
        rows = _block_rows(M.shape[0])
        for i0 in range(0, M.shape[0], rows):
            M[i0 : i0 + rows] += S[i0 : i0 + rows] * self.pref
        return M

    def __call__(
        self,
        seg_points: np.ndarray,
        seg_lengths: np.ndarray,
        wire_radii: np.ndarray,
        currents: np.ndarray,
    ) -> np.ndarray:
        n = seg_points.shape[0]
        if self.fits(n):
            phi = self.half_space(seg_points, seg_lengths, wire_radii, currents)
            extra = self.series_matrix(seg_points) @ currents
        else:
            phi = self.half_space.pref * self.half_space._blocked_product(
                seg_points, seg_lengths, wire_radii, currents
            )
            extra = self._blocked_series_product(seg_points, currents)
        extra *= self.pref
        return phi + extra

    def _blocked_series_product(self, seg_points, currents) -> np.ndarray:
        """``S @ currents`` row block by row block (beyond the budget)."""
        n = seg_points.shape[0]
        cur = np.asarray(currents)
        out = np.empty((n,) + cur.shape[1:], dtype=np.result_type(cur, float))
        x, y, z = _xyz(seg_points)
        images = _image_depths(self.series, z)
        use_numba = _want_numba(n, n, len(images))
        rows = _block_rows(n)
        blk = np.empty((min(rows, n), n))
        for i0 in range(0, n, rows):
            i1 = min(n, i0 + rows)
            b = blk[: i1 - i0]
            _series_block(
                (x[i0:i1], y[i0:i1], z[i0:i1]),
                (x, y),
                images,
                _MIN_DISTANCE,
                b,
                use_numba=use_numba,
            )
            if self.line is not None:
                self._apply_line(b, i0)
            out[i0:i1] = b @ cur
        return out


# ---------------------------------------------------------------------
# Field kernels (potential at arbitrary points, post-solve)
# ---------------------------------------------------------------------


def _point_field_kernel(
    points: np.ndarray, sources: np.ndarray, min_distance: float
) -> np.ndarray:
    """Unscaled half-space point kernel ``1/r + 1/r'`` at field points.

    Parameters
    ----------
    points
        Field points in m, shape ``(M, 3)``.
    sources
        Point-source positions in m, shape ``(N, 3)``, depth ``z > 0``.
    min_distance
        Clamp for both distances in m (field evaluations only).

    Returns
    -------
    numpy.ndarray
        Kernel in 1/m, shape ``(M, N)``; the potential is
        ``rho/(4π) · kernel @ I``.
    """
    block, _ = _half_space_block(_xyz(points), _xyz(sources), min_distance)
    return block


def _field_row_blocks(
    n_points: int, n_sources: int, *, force_blocks: bool = False
) -> list[slice]:
    """Row slices of a field evaluation.

    One piece up to :data:`FIELD_PIECE_BYTES` (bit-identical to v0.15.0)
    unless ``force_blocks``; row blocks of :data:`BLOCK_BYTES` beyond.
    """
    if not force_blocks and 8 * n_points * n_sources <= FIELD_PIECE_BYTES:
        return [slice(0, n_points)]
    rows = _block_rows(n_sources)
    return [slice(i0, min(n_points, i0 + rows)) for i0 in range(0, n_points, rows)]


def _point_field(
    points: np.ndarray,
    sources: np.ndarray,
    currents: np.ndarray,
    rho: float,
    min_distance: float,
    line: LineSegments | None = None,
) -> np.ndarray:
    """Half-space potential in V of point sources at field points.

    ``rho/(4π) · (K @ Re I + j · K @ Im I)`` with ``K`` from
    :func:`_point_field_kernel`, in one piece up to
    :data:`FIELD_PIECE_BYTES` (bit-identical to v0.15.0) and in row
    blocks of :data:`BLOCK_BYTES` beyond (10.62).

    Parameters
    ----------
    points
        Field points in m, shape ``(M, 3)``.
    sources
        Point-source positions in m, shape ``(N, 3)``.
    currents
        Complex source currents in A, shape ``(N,)``.
    rho
        Resistivity in Ω·m.
    min_distance
        Distance clamp in m.
    line
        Source segments for the exact line-source kernel; ``None``
        (default) evaluates point sources.
    """
    src = _xyz(sources)
    obs_all = _xyz(points)
    out = np.empty(points.shape[0], dtype=complex)
    for sl in _field_row_blocks(points.shape[0], sources.shape[0]):
        kernel, _ = _half_space_block(tuple(c[sl] for c in obs_all), src, min_distance)
        out[sl] = (rho / (4.0 * np.pi)) * (
            kernel @ currents.real + 1j * (kernel @ currents.imag)
        )
    if line is not None:
        re, im = _line_field_correction(
            points, line, [currents.real, currents.imag], min_distance
        )
        out += (rho / (4.0 * np.pi)) * (re + 1j * im)
    return out


def _inverse_distance(
    delta_sq: np.ndarray, z_field: np.ndarray, z_img: np.ndarray, min_distance: float
) -> np.ndarray:
    """``1/max(sqrt(s² + (z - z_img)²), d)`` for one image (field series)."""
    r = np.sqrt(delta_sq + (z_field - z_img[None, :]) ** 2)
    np.maximum(r, min_distance, out=r)
    return 1.0 / r


def _series_field(
    points: np.ndarray,
    sources: np.ndarray,
    real_parts: Sequence[np.ndarray],
    series: ImageSeries,
    min_distance: float,
    line: LineSegments | None = None,
    far_field: str = "quadrupole",
) -> list[np.ndarray]:
    """Unscaled Tagg/Sunde sums of real current arrays at field points.

    For every real current array ``I`` the sum accumulates, per series
    index ``n``, the four images of :meth:`ImageSeries.layer_terms`
    before weighting them with :math:`K^n`
    (``phi += K^n · Σ_4 (1/r) @ I``), after the ``n = 0`` pair. All
    arrays share each ``1/r`` block (v0.15.0 rebuilt the geometry per
    array), and large evaluations run in row blocks (10.62, see
    :func:`_field_row_blocks`). Valid for field points and sources in
    the upper layer.

    Parameters
    ----------
    points
        Field points in m, shape ``(M, 3)``.
    sources
        Point-source positions in m, shape ``(N, 3)``.
    real_parts
        Real current arrays in A, each of shape ``(N,)`` or ``(N, J)``
        (typically the real and the non-zero imaginary part).
    series
        The image series.
    min_distance
        Distance clamp in m.
    line
        Source segments for the exact line-source kernel (images per
        series term); ``None`` (default) evaluates point sources.
    far_field
        Form of the far images with the line kernel
        (``Engine.image_far_field``, recorded in ``metadata["series"]``).

    Returns
    -------
    list[numpy.ndarray]
        One sum per current array in A/m, shape ``(M,)`` or ``(M, J)``;
        the potential is ``rho_1/(4π)`` times this.
    """
    xs, ys, zs = _xyz(sources)
    xo_all, yo_all, zo_all = _xyz(points)
    surface = [term.sign * zs for term in series.surface_terms()]
    layered = [
        (quad[0].weight, [term.offset + term.sign * zs for term in quad])
        for quad in series.iter_layer_terms()
    ]
    outs = [np.empty((points.shape[0],) + c.shape[1:]) for c in real_parts]
    for sl in _field_row_blocks(points.shape[0], sources.shape[0]):
        xo, yo, zo = xo_all[sl], yo_all[sl], zo_all[sl]
        diff_x = np.subtract.outer(xo, xs)
        diff_y = np.subtract.outer(yo, ys)
        delta_sq = diff_x * diff_x + diff_y * diff_y
        del diff_x, diff_y
        z_field = zo[:, None]

        phis = [np.zeros((xo.shape[0],) + c.shape[1:]) for c in real_parts]
        # n = 0 (two images, weight 1)
        for z_img in surface:
            inv_r = _inverse_distance(delta_sq, z_field, z_img, min_distance)
            for phi, c in zip(phis, real_parts, strict=True):
                phi += inv_r @ c
        # n = 1, 2, ...
        for weight, quad in layered:
            accs = [np.zeros_like(phi) for phi in phis]
            for z_img in quad:
                inv_r = _inverse_distance(delta_sq, z_field, z_img, min_distance)
                for acc, c in zip(accs, real_parts, strict=True):
                    acc += inv_r @ c
            for phi, acc in zip(phis, accs, strict=True):
                phi += weight * acc
        for out, phi in zip(outs, phis, strict=True):
            out[sl] = phi
    if line is not None:
        images = _series_images(series)
        for out, corr in zip(
            outs,
            _line_field_correction(
                points, line, real_parts, min_distance, images, far_field
            ),
            strict=True,
        ):
            out += corr
    return outs


def _line_field_correction(
    points: np.ndarray,
    line: LineSegments,
    real_parts: Sequence[np.ndarray],
    min_distance: float,
    images: list[tuple[float, float, float]] | None = None,
    far_field: str = "quadrupole",
) -> list[np.ndarray]:
    """Line-kernel correction of point-kernel field sums (unscaled, A/m).

    Adds, per real current array, the dense far-field (quadrupole) term
    of the direct and surface-image sources and the sparse exact-minus-
    point corrections of near sources (and of near Tagg/Sunde images
    when ``images`` are given, their far quadrupole terms with
    ``far_field = "quadrupole"``) — see :mod:`groundfield.solver._line_kernel`.
    """
    quadrupole = far_field == "quadrupole"
    from scipy.sparse import csr_matrix

    m, n = points.shape[0], line.mid.shape[0]
    outs = [np.zeros((m,) + c.shape[1:]) for c in real_parts]
    far = FarImages()
    if images and quadrupole:
        far = far_images(points[:, 2], line, images)
    for sl in _field_row_blocks(m, n, force_blocks=True):
        Q = field_quadrupole_rows(points[sl], line)
        if far:
            # Far Tagg/Sunde images with the quadrupole term (wave 3, 10.28).
            pts = points[sl]
            Q += far_image_rows(pts, None, np.zeros(pts.shape[0]), line, far)
        for out, c in zip(outs, real_parts, strict=True):
            out[sl] += Q @ c
        # Sparse near corrections block by block as well (audit round 1,
        # A1-numerics F02): their memory then stays bounded by the block.
        pts = points[sl]
        corr = [field_corrections(pts, line, min_distance)]
        if images:
            corr.append(
                series_field_corrections(
                    pts, line, images, min_distance, quadrupole=quadrupole
                )
            )
        sparse = SparseCorrection.concatenate(corr)
        if sparse.rows.size:
            C = csr_matrix(
                (sparse.values, (sparse.rows, sparse.cols)), shape=(pts.shape[0], n)
            )
            for out, c in zip(outs, real_parts, strict=True):
                out[sl] += C @ c
    return outs
