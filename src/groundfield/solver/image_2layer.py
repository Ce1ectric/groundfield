"""Image-charge backend for **2-layer soil** (Tagg / Sunde).

Computes the potential field in a layered half-space (upper layer
$\\rho_1$ of thickness $h_1$, semi-infinite lower layer
$\\rho_2$) using the classical **image-charge series**.

Theory (short)
--------------
With the reflection coefficient at the layer interface
$$
K \\;=\\; \\frac{\\rho_2 - \\rho_1}{\\rho_2 + \\rho_1}
$$
a point current source $I$ at $(x_s, y_s, z_s)$ with
$0 < z_s < h_1$ produces, at any field point in the upper
layer, the potential
$$
\\varphi(x, y, z) \\;=\\;
\\frac{\\rho_1\\, I}{4\\pi}\\,\\Bigl[
    \\tfrac{1}{r_0^+} + \\tfrac{1}{r_0^-}
    + \\sum_{n=1}^{\\infty} K^n
      \\Bigl(\\tfrac{1}{r_n^{++}} + \\tfrac{1}{r_n^{+-}}
           + \\tfrac{1}{r_n^{-+}} + \\tfrac{1}{r_n^{--}}\\Bigr)\\Bigr],
$$
with the image-source distances
$$
r_n^{\\sigma\\tau} \\;=\\; \\sqrt{(x-x_s)^2 + (y-y_s)^2 +
                               (z - \\sigma\\,2 n h_1 - \\tau z_s)^2},
\\qquad \\sigma, \\tau \\in \\{+1,-1\\}.
$$
The $n = 0$ term (two images, weight 1) is exactly the
homogeneous image-charge backend. The 2-layer backend therefore
reduces to the homogeneous result for $\\rho_2 = \\rho_1$ (i.e.
$K = 0$) — usable as a consistency test.

Convergence
-----------
Series terms decay as $K^n$ (geometric). The series is owned by
:class:`~groundfield.solver._series.ImageSeries`: it stops after the
first $n$ whose geometric tail bound $|K|^{n+1}/(1-|K|)$ falls below
``tol``, or at ``max_terms``; ``FieldResult.metadata`` reports
``n_terms_used`` and ``converged``, and a
:class:`SeriesTruncationWarning` is emitted when the cap is reached.

Geometry
--------
- Electrodes inside the **upper layer** ($z_\\text{seg} < h_1$) use
  the image series above.
- As soon as a segment reaches the interface, every source/observer
  pair is evaluated with the rigorous Sommerfeld kernel of ADR-0007
  (segments are split at $h_1$ first).
- For $|K| \\to 1$ (extreme contrasts, e.g. wet soil over rock)
  convergence slows down significantly; raise
  ``Engine.image_max_terms``.

The solve flow itself is the shared image-family pipeline of
:mod:`groundfield.solver._pipeline`; this module supplies the
two-layer Green's function.

References
----------
- Tagg, G. F. (1964). *Earth Resistances*. Pitman, ch. 5.
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*. Dover, sect. 3.5.
- ADR-0001 ``docs/adr/0001-two-layer-method.md``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from groundfield.soil.models import HomogeneousSoil, MultiLayerSoil, TwoLayerSoil
from groundfield.solver._guards import (
    _MIN_DISTANCE,
    _check_point_interface_diagonal,
    _require_image_separation,
)
from groundfield.solver._kernels import (
    DEFAULT_KERNEL,
    TwoLayerSeriesKernel,
    _block_rows,
    _half_space_block,
    _xyz,
)
from groundfield.solver._line_kernel import (
    FarImages,
    LineSegments,
    SparseCorrection,
    _point_values,
    add_symmetric,
    far_image_rows,
    far_images,
    galerkin_pairs,
    image_taper,
    point_self_term,
    reaction_corrections,
    reaction_quadrupole_rows,
)
from groundfield.solver._pipeline import solve_image_family
from groundfield.solver._series import ImageSeries
from groundfield.solver._series import (
    SeriesTruncationWarning as SeriesTruncationWarning,
)
from groundfield.solver._series import _series_tail_bound as _series_tail_bound
from groundfield.solver.result import FieldResult
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_image_2layer", "SeriesTruncationWarning"]

_log = get_logger(__name__)


# ---------------------------------------------------------------------
# 2-layer kernel
# ---------------------------------------------------------------------


def _two_layer_image_offsets(
    K: float, h_1: float, max_terms: int, tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Generate the series coefficients for the image-charge sum.

    Thin wrapper around :meth:`ImageSeries.offsets` (kept for the
    sibling backends that import it).

    Returns
    -------
    z_offsets : np.ndarray, shape (M_img,)
        Offsets used as ``z_image = z_offsets[k] + sign_zs * z_src``.
    sign_zs_arr : np.ndarray, shape (M_img,)
        ±1 — sign factor applied to ``z_src`` per image term.
    weights : np.ndarray, shape (M_img,)
        $K^n$ weights per image term.
    n_terms_used : int
        Last evaluated $n$. ``n_terms_used == max_terms`` means
        the tolerance was not reached.
    """
    series = ImageSeries.from_coefficients(K, h_1, max_terms, tol)
    z_offsets, sign_zs_arr, weights = series.offsets()
    return z_offsets, sign_zs_arr, weights, series.n_terms


class _TwoLayerKernel:
    """Reaction operator of the two-layer soil (callable kernel).

    Inside the upper layer the operator is
    :class:`~groundfield.solver._kernels.TwoLayerSeriesKernel` (half-space
    part with :math:`\\rho_1` plus the Tagg/Sunde images, assembled once
    per geometry). With ``allow_cross_layer`` and a segment at or below
    the interface it switches to the ADR-0007 Sommerfeld path for every
    pair (:func:`_layered_green_kernel`, whose matrices are cached too).
    Called with ``(seg_points, seg_lengths, wire_radii, currents)`` it
    returns the segment-midpoint potentials in V.
    """

    def __init__(
        self,
        soil: TwoLayerSoil,
        max_terms: int,
        tol: float,
        *,
        allow_cross_layer: bool = False,
        line: LineSegments | None = None,
        far_field: str = "quadrupole",
    ) -> None:
        self.soil = soil
        self.allow_cross_layer = allow_cross_layer
        self.line = line
        self.far_field = far_field
        self.series = ImageSeries.for_soil(soil, max_terms, tol)
        self.in_layer = TwoLayerSeriesKernel(
            soil.rho_1, self.series, line=line, far_field=far_field
        )
        # Cache for the (current-independent) cross-layer matrices: the
        # per-frequency and per-excitation kernel calls within one solve
        # share the identical geometry, so phi_hom_matrix and delta_Z
        # are assembled once (audit 2026-07-08, WP-D performance note).
        self._cross_cache: dict = {}

    def _cross_layer(self, seg_points: np.ndarray) -> bool:
        # ADR-0007: when the geometry crosses the layer interface the
        # Sommerfeld kernel from coupling.layered_green is used for every
        # source/observer pair; pure-upper-layer geometries keep the
        # Tagg/Sunde image series.
        return self.allow_cross_layer and float(seg_points[:, 2].max()) >= self.soil.h_1

    @property
    def materialised(self) -> bool:
        """Whether repeated calls reuse stored matrices."""
        return "phi_hom_matrix" in self._cross_cache or self.in_layer.materialised

    @property
    def assemblies(self) -> int:
        """Number of full reaction-matrix evaluations so far."""
        return self.in_layer.assemblies + int(self._cross_cache.get("assemblies", 0))

    def matrix(
        self, seg_points: np.ndarray, seg_lengths: np.ndarray, wire_radii: np.ndarray
    ) -> np.ndarray:
        """Reaction matrix in Ω, bit-identical to v0.15.0's ``kernel(eye)``."""
        if self._cross_layer(seg_points):
            if not self.in_layer.fits(seg_points.shape[0]):
                # Beyond the matrix budget nothing is kept (10.56): the two
                # matrices are summed in place and returned.
                phi_hom_matrix, delta_Z = _CrossLayerParts(
                    seg_points,
                    seg_lengths,
                    wire_radii,
                    soil=self.soil,
                    line=self.line,
                    far_field=self.far_field,
                ).matrices()
                phi_hom_matrix += delta_Z
                return phi_hom_matrix
            phi_hom_matrix, delta_Z = _layered_green_matrices(
                seg_points,
                seg_lengths,
                wire_radii,
                soil=self.soil,
                cache=self._cross_cache,
                line=self.line,
                far_field=self.far_field,
            )
            return phi_hom_matrix + delta_Z
        return self.in_layer.matrix(seg_points, seg_lengths, wire_radii)

    def __call__(
        self,
        seg_points: np.ndarray,
        seg_lengths: np.ndarray,
        wire_radii: np.ndarray,
        currents: np.ndarray,
    ) -> np.ndarray:
        if self._cross_layer(seg_points):
            if self.in_layer.fits(seg_points.shape[0]):
                return _layered_green_kernel(
                    seg_points,
                    seg_lengths,
                    wire_radii,
                    currents,
                    soil=self.soil,
                    cache=self._cross_cache,
                    line=self.line,
                    far_field=self.far_field,
                )
            # Beyond the matrix budget (16 N² bytes, as the series path):
            # row blocks with O(N · block) memory (10.56).
            parts = self._cross_parts(seg_points, seg_lengths, wire_radii)
            self._cross_cache["assemblies"] = (
                int(self._cross_cache.get("assemblies", 0)) + 1
            )
            return parts.product(currents)
        return self.in_layer(seg_points, seg_lengths, wire_radii, currents)

    def _cross_parts(
        self, seg_points: np.ndarray, seg_lengths: np.ndarray, wire_radii: np.ndarray
    ) -> _CrossLayerParts:
        """The cross-layer pieces of this geometry (kept across calls)."""
        key = (
            seg_points.shape[0],
            hash(seg_points.tobytes()),
            hash(seg_lengths.tobytes()),
            self.line is not None,
        )
        if self._cross_cache.get("parts_key") != key:
            self._cross_cache["parts"] = _CrossLayerParts(
                seg_points,
                seg_lengths,
                wire_radii,
                soil=self.soil,
                line=self.line,
                far_field=self.far_field,
            )
            self._cross_cache["parts_key"] = key
        return self._cross_cache["parts"]


def _two_layer_self_kernel_factory(
    soil: TwoLayerSoil,
    max_terms: int,
    tol: float,
    *,
    allow_cross_layer: bool = False,
    line: LineSegments | None = None,
    far_field: str = "quadrupole",
) -> _TwoLayerKernel:
    """Build the self-action operator for the 2-layer case.

    The $n = 0$ direct contribution of a segment onto itself is
    handled by the analytical line-self correction (same as the
    homogeneous backend, but with $\\rho_1$ as prefactor). The
    image terms with $n \\ge 1$ are at least $2(h_1 - z)$ away
    from the segments, so the point-source approximation is used for
    them.

    Parameters
    ----------
    soil, max_terms, tol
        Two-layer soil model and Tagg/Sunde series-truncation
        parameters.
    allow_cross_layer
        When ``True`` (ADR-0007) the kernel transparently dispatches
        to the rigorous Sommerfeld-quadrature path in
        :mod:`groundfield.coupling.layered_green` as soon as a segment
        reaches the interface. When ``False`` (default) the kernel uses
        only the image series and the caller must enforce the
        precondition ``z_max < h_1``.
    line
        Segment geometry for the exact line-source kernel
        (``Engine(kernel="line")``); ``None`` keeps the point kernel.
    far_field
        Form of the far images with the line kernel
        (``Engine.image_far_field``): ``"quadrupole"`` (default) or
        ``"point"``.

    Returns
    -------
    _TwoLayerKernel
        Callable ``(seg_points, seg_lengths, wire_radii, currents) ->
        phi`` with a ``matrix(...)`` method for the full reaction
        matrix.
    """
    return _TwoLayerKernel(
        soil,
        max_terms,
        tol,
        allow_cross_layer=allow_cross_layer,
        line=line,
        far_field=far_field,
    )


def _layered_green_matrices(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    *,
    soil: TwoLayerSoil,
    cache: dict | None = None,
    line: LineSegments | None = None,
    far_field: str = "quadrupole",
) -> tuple[np.ndarray, np.ndarray]:
    """ADR-0007 cross-layer fallback using the rigorous Sommerfeld kernel.

    Computes ``phi`` at every segment midpoint by decomposing into

    .. code-block:: text

        phi = phi_homog(rho_1)  +  delta_phi_layered

    where ``phi_homog(rho_1)`` is the standard homogeneous-soil
    potential (direct point source + free-surface image, computed
    via the existing :func:`_self_corrected_kernel` with the upper-
    layer resistivity), and ``delta_phi_layered`` is the additional
    contribution from the lower layer.

    The decomposition has two key advantages:

    1. The diagonal (segment-self) of ``phi_homog`` uses the
       analytical line-self formula, so the wire-radius
       regularisation that ADR-0007's bare Sommerfeld kernel would
       otherwise need is inherited "for free" from the historic
       homogeneous code.
    2. In the homogeneous limit $\\rho_2 = \\rho_1$ the
       layered-correction term vanishes identically, and the
       result reduces to the ordinary homogeneous-soil potential
       — bit-exact regression on the limit case.

    The layered correction $\\Delta G = G_{\\text{2-layer}} - G_{\\text{homog}}$
    is built point-source / point-observer (no line-self
    correction needed because the singular direct term is in
    $\\phi_{\\text{homog}}$); a wire-radius regularisation is still
    applied to the diagonal in case the spectral quadrature has
    residual numerical noise at $s \\to 0$.

    For pure-upper-layer worlds (`z_max < h_1`) the calling factory
    short-circuits to the fast Tagg/Sunde image series instead, so
    this path is only invoked when at least one segment crosses the
    interface.
    """
    n = seg_points.shape[0]
    # Both matrices below are independent of ``currents`` — reuse
    # them across the per-excitation / per-frequency kernel calls of
    # one solve (keyed by the geometry buffer).
    cache_key = None
    if cache is not None:
        cache_key = (
            n,
            hash(seg_points.tobytes()),
            hash(seg_lengths.tobytes()),
            line is not None,
            far_field,
        )
        if cache.get("key") == cache_key:
            return cache["phi_hom_matrix"], cache["delta_Z"]

    parts = _CrossLayerParts(
        seg_points, seg_lengths, wire_radii, soil=soil, line=line, far_field=far_field
    )
    phi_hom_matrix, delta_Z = parts.matrices()
    if cache is not None:
        cache["key"] = cache_key
        cache["phi_hom_matrix"] = phi_hom_matrix
        cache["delta_Z"] = delta_Z
        cache["assemblies"] = int(cache.get("assemblies", 0)) + 1
    return phi_hom_matrix, delta_Z


class _CrossLayerParts:
    """The pieces of the ADR-0007 cross-layer reaction matrix of one geometry.

    ``phi = Φ_hom + ΔZ`` (see :func:`_layered_green_matrices`). The object
    serves the two evaluation modes of the image family (review pass 10,
    10.56):

    * :meth:`matrices` assembles both ``N × N`` matrices once per solve —
      within :data:`~groundfield.solver._kernels.MATRIX_BUDGET_BYTES` — with
      the arithmetic of the historic assembly (bit-identical);
    * :meth:`product` applies them row block by row block beyond the budget,
      with ``O(N · block)`` memory; the depth groups of ``ΔZ`` are planned
      once from their whole range of distances and kept, so a block costs
      the evaluation of its entries only. Results differ from the
      materialised evaluation by BLAS round-off only, as on the within-layer
      path.

    Parameters
    ----------
    seg_points, seg_lengths, wire_radii
        Segment midpoints (m), lengths (m) and wire radii (m).
    soil
        The two-layer soil.
    line
        Segment geometry of the exact line-source kernel, or ``None``
        (point kernel).
    far_field
        Form of the far images with the line kernel
        (``Engine.image_far_field``).
    """

    def __init__(
        self,
        seg_points: np.ndarray,
        seg_lengths: np.ndarray,
        wire_radii: np.ndarray,
        *,
        soil: TwoLayerSoil,
        line: LineSegments | None,
        far_field: str = "quadrupole",
    ) -> None:
        self.seg_points = seg_points
        self.seg_lengths = seg_lengths
        self.wire_radii = wire_radii
        self.soil = soil
        self.line = line
        h_1 = soil.h_1
        # ADR-0007 Phase A.1: per-segment baseline rho. Each source
        # segment j gets its layer-local rho as the homogeneous-soil
        # baseline; the layered correction is then the *smooth*
        # deviation from that baseline (no residual 1/r divergence
        # from a rho mismatch). For pure-upper-layer worlds this
        # collapses to the historic rho_1 baseline.
        self.rho_per_segment = np.where(seg_points[:, 2] < h_1, soil.rho_1, soil.rho_2)
        self.lower = seg_points[:, 2] >= h_1
        self.groups = _depth_groups(seg_points)
        self._phi_rows: _PhiHomRows | None = None
        self._diagonal: np.ndarray | None = None
        self._sparse: SparseCorrection | None = None
        self._plans: dict = {}
        # Line form of the layered correction (wave 3 of the 2026.11 work,
        # 10.28, ADR-0007): its image families, dense far terms and sparse
        # near terms.
        self.families = (
            _LayeredImageFamilies(line, self.lower, soil, far_field)
            if line is not None
            else None
        )

    # ------------------------------------------------------------------
    # Shared pieces
    # ------------------------------------------------------------------

    def phi_rows(self) -> _PhiHomRows:
        """Row provider of ``Φ_hom`` (built once)."""
        if self._phi_rows is None:
            soil = self.soil
            # The homogeneous part uses rho_at_source(j) for each column
            # (source segment); this generalises _self_corrected_kernel:
            # the diagonal still uses the line-self formula with
            # rho_local(i), and off-diagonal entries use the point-source
            # 1/r + 1/r_image formula with rho_local(j).
            self._phi_rows = _PhiHomRows(
                self.seg_points,
                self.seg_lengths,
                self.wire_radii,
                self.rho_per_segment,
                line=self.line,
                transmitted_rho=2.0
                * soil.rho_1
                * soil.rho_2
                / (soil.rho_1 + soil.rho_2),
                lower=self.lower,
            )
        return self._phi_rows

    def diagonal(self) -> np.ndarray:
        """Diagonal of ``ΔZ`` in Ω (built once)."""
        if self._diagonal is None:
            self._diagonal = _layered_diagonal(
                self.seg_points,
                self.seg_lengths,
                self.wire_radii,
                self.rho_per_segment,
                self.lower,
                self.soil,
                self.line,
                families=self.families,
            )
        return self._diagonal

    def sparse(self) -> SparseCorrection:
        """Off-diagonal near-pair corrections of ``ΔZ`` (line kernel, built once)."""
        if self._sparse is None:
            if self.families is None:
                empty = np.zeros(0, dtype=int)
                self._sparse = SparseCorrection(empty, empty, np.zeros(0))
            else:
                self._sparse = self.families.sparse()
        return self._sparse

    def add_far_rows(self, block: np.ndarray, i0: int) -> None:
        """Add the far-field terms of the layered images to rows ``i0 …``."""
        if self.families is not None and self.families.quadrupole:
            block += self.families.rows(i0, i0 + block.shape[0])

    # ------------------------------------------------------------------
    # Materialised mode
    # ------------------------------------------------------------------

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """``(Φ_hom, ΔZ)`` as ``N × N`` arrays in Ω."""
        n = self.seg_points.shape[0]
        phi_hom_matrix = self.phi_rows().matrix()
        # Layered correction. The baseline rho per source is
        # rho_at_source(j). Off-diagonal assembly is grouped by the
        # (z_i, z_j, baseline) combination and evaluated with ONE
        # spectral-amplitude solve per group (audit 2026-07-08, WP-D
        # performance note): the historic per-entry scalar quadrature made
        # the cross-layer reaction matrix O(n²) BVP solves.
        delta_Z = np.zeros((n, n), dtype=float)
        _fill_layered_correction(
            delta_Z, self.seg_points, self.rho_per_segment, self.soil
        )
        # Exact line sources (10.28): the layered correction in line form —
        # the far-field terms of its images over all pairs and the Galerkin
        # integrals of the near image pairs.
        if self.families is not None and self.families.quadrupole:
            # Symmetric (reciprocity): the upper half, mirrored (W3A-01).
            add_symmetric(
                delta_Z,
                lambda i0, i1: self.families.rows(i0, i1, c0=i0),
                _block_rows(n),
            )
        self.sparse().add_to_rows(delta_Z, 0)
        diag = np.arange(n)
        delta_Z[diag, diag] = self.diagonal()
        return phi_hom_matrix, delta_Z

    # ------------------------------------------------------------------
    # Blocked mode
    # ------------------------------------------------------------------

    def _plan(self, gi: int, gj: int):
        """Plan of depth group ``(gi, gj)`` from its whole range (kept)."""
        key = (gi, gj)
        if key not in self._plans:
            from groundfield.coupling.layered_green import layered_correction_group_plan

            rows, cols = self.groups[gi], self.groups[gj]
            chunk = _block_rows(cols.size)
            s_min, s_max = np.inf, -np.inf
            for r0 in range(0, rows.size, chunk):
                s = _horizontal_distances(self.seg_points, rows[r0 : r0 + chunk], cols)
                s_min = min(s_min, float(s.min()))
                s_max = max(s_max, float(s.max()))
            soil = self.soil
            self._plans[key] = layered_correction_group_plan(
                float(self.seg_points[rows[0], 2]),
                float(self.seg_points[cols[0], 2]),
                rho_1=soil.rho_1,
                rho_2=soil.rho_2,
                h_1=soil.h_1,
                rho_baseline=float(self.rho_per_segment[cols[0]]),
                s_min=s_min,
                s_max=s_max,
                n_pairs=rows.size * cols.size,
                min_distance=_MIN_DISTANCE,
            )
        return self._plans[key]

    def delta_rows(self, i0: int, i1: int) -> np.ndarray:
        """Rows ``i0:i1`` of ``ΔZ`` in Ω."""
        n = self.seg_points.shape[0]
        out = np.zeros((i1 - i0, n), dtype=float)
        for gi, rows in enumerate(self.groups):
            lo, hi = np.searchsorted(rows, [i0, i1])
            if hi <= lo:
                continue
            sub = rows[lo:hi]
            for gj, cols in enumerate(self.groups):
                plan = self._plan(gi, gj)
                if plan is None:
                    continue
                s = _horizontal_distances(self.seg_points, sub, cols)
                out[np.ix_(sub - i0, cols)] = plan(s.ravel()).reshape(s.shape) / (
                    2.0 * np.pi
                )
        self.add_far_rows(out, i0)
        self.sparse().add_to_rows(out, i0)
        idx = np.arange(i0, i1)
        out[idx - i0, idx] = self.diagonal()[i0:i1]
        return out

    def product(self, currents: np.ndarray) -> np.ndarray:
        """``(Φ_hom + ΔZ) @ currents`` row block by row block."""
        n = self.seg_points.shape[0]
        cur = np.asarray(currents)
        out = np.empty((n,) + cur.shape[1:], dtype=np.result_type(cur, float))
        rows = _block_rows(n)
        phi = self.phi_rows()
        for i0 in range(0, n, rows):
            i1 = min(n, i0 + rows)
            out[i0:i1] = phi.rows(i0, i1) @ cur + self.delta_rows(i0, i1) @ cur
        return out


def _horizontal_distances(seg_points: np.ndarray, rows, cols) -> np.ndarray:
    """``rows × cols`` horizontal distances in m, component-wise.

    Bit-identical to the former ``(rows, cols, 2)`` difference tensor and
    ``einsum`` (10.56), with two instead of four ``rows × cols`` arrays.
    """
    s_grid = np.subtract.outer(seg_points[rows, 0], seg_points[cols, 0])
    s_grid *= s_grid
    dy2 = np.subtract.outer(seg_points[rows, 1], seg_points[cols, 1])
    dy2 *= dy2
    s_grid += dy2
    del dy2
    np.sqrt(s_grid, out=s_grid)
    return s_grid


def _depth_groups(seg_points: np.ndarray) -> list[np.ndarray]:
    """Segment indices per depth (rounded to 1 nm), in increasing depth order."""
    z_keys = np.round(seg_points[:, 2] / 1e-9).astype(np.int64)
    return [np.where(z_keys == key)[0] for key in np.unique(z_keys)]


def _fill_layered_correction(
    delta_Z: np.ndarray,
    seg_points: np.ndarray,
    rho_per_segment: np.ndarray,
    soil: TwoLayerSoil,
) -> None:
    """Off-diagonal layered correction ``ΔZ`` (point form, Ω), in place.

    One spectral evaluation per ``(z_i, z_j)`` depth group (audit 2026-07-08,
    WP-D performance note; the historic per-entry quadrature made the matrix
    O(n²) boundary-value solves). A group larger than one block of
    :data:`~groundfield.solver._kernels.BLOCK_BYTES` is planned once from the
    range of its distances and then evaluated row chunk by row chunk
    (review pass 10, 10.56): its temporaries — the distances, their
    logarithm, the interpolated values and the closed-form image terms —
    then take O(block) memory instead of several ``rows × cols`` arrays, and
    the entries are those of the one-piece evaluation (the interpolation is
    evaluated point by point; groups contracted pair by pair are small and
    always evaluated in one piece).
    """
    from groundfield.coupling.layered_green import (
        layered_correction_group_plan,
        two_layer_layered_correction_group,
    )

    rho_1, rho_2, h_1 = soil.rho_1, soil.rho_2, soil.h_1
    groups = _depth_groups(seg_points)
    for rows in groups:
        z_i = float(seg_points[rows[0], 2])
        for cols in groups:
            z_j = float(seg_points[cols[0], 2])
            rho_baseline_j = float(rho_per_segment[cols[0]])
            chunk = _block_rows(cols.size)
            if rows.size <= chunk:
                s_grid = _horizontal_distances(seg_points, rows, cols)
                # The closed-form 1/R terms regularised at the 1 mm of the
                # point forms they combine with (audit A3S-01), not at the
                # h_1/200 of the truncation plan.
                vals = two_layer_layered_correction_group(
                    s_grid.ravel(),
                    z_i,
                    z_j,
                    rho_1=rho_1,
                    rho_2=rho_2,
                    h_1=h_1,
                    rho_baseline=rho_baseline_j,
                    min_distance=_MIN_DISTANCE,
                )
                delta_Z[np.ix_(rows, cols)] = vals.reshape(s_grid.shape) / (2.0 * np.pi)
                continue
            # Large group: the range of its distances first, chunk by chunk.
            s_min, s_max = np.inf, -np.inf
            for r0 in range(0, rows.size, chunk):
                s_grid = _horizontal_distances(seg_points, rows[r0 : r0 + chunk], cols)
                s_min = min(s_min, float(s_grid.min()))
                s_max = max(s_max, float(s_grid.max()))
            group = layered_correction_group_plan(
                z_i,
                z_j,
                rho_1=rho_1,
                rho_2=rho_2,
                h_1=h_1,
                rho_baseline=rho_baseline_j,
                s_min=s_min,
                s_max=s_max,
                n_pairs=rows.size * cols.size,
                min_distance=_MIN_DISTANCE,
            )
            if group is None:
                continue  # delta_Z is zero there already
            for r0 in range(0, rows.size, chunk):
                sub = rows[r0 : r0 + chunk]
                s_grid = _horizontal_distances(seg_points, sub, cols)
                vals = group(s_grid.ravel())
                delta_Z[np.ix_(sub, cols)] = vals.reshape(s_grid.shape) / (2.0 * np.pi)


def _layered_green_kernel(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    currents,
    *,
    soil: TwoLayerSoil,
    cache: dict | None = None,
    line: LineSegments | None = None,
    far_field: str = "quadrupole",
):
    """ADR-0007 cross-layer reaction kernel: ``(Φ_hom + ΔZ) · currents``.

    See :func:`_layered_green_matrices` for the decomposition; the two
    matrices are taken from ``cache`` when the geometry is unchanged.
    """
    phi_hom_matrix, delta_Z = _layered_green_matrices(
        seg_points,
        seg_lengths,
        wire_radii,
        soil=soil,
        cache=cache,
        line=line,
        far_field=far_field,
    )
    return phi_hom_matrix @ currents + delta_Z @ currents


class _LayeredImageFamilies:
    r"""The layered correction ``ΔZ`` of the cross-layer path as image families.

    In two-layer soil the Green's function of every layer combination is an
    exact superposition of images (Sunde; the generations of
    :func:`groundfield.coupling.layered_green._image_generation`), so the
    layered correction relative to the baseline of the source layer is, with
    :math:`K = (\rho_2 - \rho_1)/(\rho_2 + \rho_1)`,
    :math:`c = 2\rho_1\rho_2/(\rho_1 + \rho_2)` and :math:`q = 2h_1`,

    * observer and source in layer 1: :math:`\rho_1 K^m` at the four
      Tagg/Sunde images :math:`z_s \pm mq`, :math:`\pm mq - z_s` (``m >= 1``);
    * both in layer 2: :math:`-\rho_2 K` at the interface image
      :math:`q - z_s`, :math:`-\rho_2 K^2` at the surface image
      :math:`-z_s` and :math:`\rho_2 (1 - K^2) K^m` at :math:`-z_s - mq`;
    * in different layers: :math:`c\,K^m` at :math:`z_s + mq` (source in
      layer 2) or :math:`z_s - mq` (source in layer 1) and at
      :math:`-z_s - mq` (``m >= 1``; the ``m = 0`` pair is the transmitted
      resistivity of ``Φ_hom``).

    The spectral quadrature evaluates ``ΔZ`` between segment midpoints. This
    object adds, per image, ``coefficient · (line − point)/(4π)``: the
    far-field term of :func:`~groundfield.solver._line_kernel.image_quadrupole_rows`
    over all pairs (images beyond
    :data:`~groundfield.solver._line_kernel.IMAGE_QUADRUPOLE_REACH` segment
    lengths tapered off) and, for the pairs within
    :data:`~groundfield.solver._line_kernel.SWITCH_FACTOR` segment lengths,
    the blended Galerkin integral minus the far form — the treatment of the
    Tagg/Sunde images of the within-layer path (wave 3 of the 2026.11 work,
    10.28; before, only the interface image of near pairs in the same layer
    was a Galerkin integral and everything else stayed in point form). The
    diagonal (a segment and its own images) is the tapered Galerkin integral
    at every distance.

    The families are grouped by image geometry ``(offset, sign)``; each group
    carries one 2 × 2 table of coefficients indexed by the layers of observer
    and source.

    With ``far_field = "point"`` (``Engine.image_far_field``, the author's
    decision of 2026-11-15) the line form is that of the base of wave 3:
    only the image in the interface of a pair in one layer (``ρ_1 K`` at
    ``q - z_s`` for the upper layer, ``-ρ_2 K`` for the lower) is a Galerkin
    integral, for the pairs within
    :data:`~groundfield.solver._line_kernel.SWITCH_FACTOR` segment lengths
    (blended, user decision D2) and on the diagonal at every distance; every
    other image keeps the point form of the spectral quadrature.
    """

    def __init__(
        self,
        line: LineSegments,
        lower: np.ndarray,
        soil: TwoLayerSoil,
        far_field: str = "quadrupole",
    ):
        from groundfield.solver._line_kernel import IMAGE_QUADRUPOLE_REACH

        self.quadrupole = far_field == "quadrupole"
        self.line = line
        self.layer = lower.astype(np.intp)
        rho_1, rho_2, h_1 = soil.rho_1, soil.rho_2, soil.h_1
        K = (rho_2 - rho_1) / (rho_2 + rho_1)
        c = 2.0 * rho_1 * rho_2 / (rho_1 + rho_2)
        q = 2.0 * h_1
        has = (bool((~lower).any()), bool(lower.any()))
        # Images up to the depth where their quadrupole term is tapered off
        # (and their weight above the floor of the spectral module).
        reach = IMAGE_QUADRUPOLE_REACH[1] * float(line.length.max())
        z_max = float(line.mid[:, 2].max())
        m_max = int(np.ceil((reach + 2.0 * z_max) / q)) + 1
        weights = []
        for m in range(1, m_max + 1):
            if abs(K) ** m < 1e-17:
                break
            weights.append((m, K**m))

        def table(uu, ul, lu, ll):
            # [observer layer][source layer]; layer 0 = upper, 1 = lower.
            tab = np.array([[uu, ul], [lu, ll]], dtype=float) / (4.0 * np.pi)
            tab[0, :] *= has[0]
            tab[1, :] *= has[1]
            tab[:, 0] *= has[0]
            tab[:, 1] *= has[1]
            return tab

        groups = [
            ([(w, m * q, 1.0) for m, w in weights], table(rho_1, c, 0.0, 0.0)),
            ([(w, -m * q, 1.0) for m, w in weights], table(rho_1, 0.0, c, 0.0)),
            (
                [(w, -m * q, -1.0) for m, w in weights],
                table(rho_1, c, c, rho_2 * (1.0 - K * K)),
            ),
            ([(w, m * q, -1.0) for m, w in weights], table(rho_1, 0.0, 0.0, 0.0)),
            ([(1.0, q, -1.0)], table(0.0, 0.0, 0.0, -rho_2 * K)),
            ([(1.0, 0.0, -1.0)], table(0.0, 0.0, 0.0, -rho_2 * K * K)),
        ]
        self.groups = [(images, tab) for images, tab in groups if images and tab.any()]
        if not self.quadrupole:
            # The interface image of pairs in one layer only (base of wave 3).
            self.groups = [
                ([(w, off, sgn) for w, off, sgn in images if off == q and sgn < 0], tab)
                for images, tab in self.groups
            ]
            self.groups = [(images, tab) for images, tab in self.groups if images]
        z = line.mid[:, 2]
        self.far = [
            far_images(z, line, images) if self.quadrupole else FarImages()
            for images, _ in self.groups
        ]

    def rows(self, i0: int, i1: int, c0: int = 0) -> np.ndarray:
        """Far-field terms of rows ``i0:i1`` against the segments ``c0 …``, in Ω.

        Shape ``(i1 - i0, N - c0)``.
        """
        line = self.line
        out = np.zeros((i1 - i0, line.mid.shape[0] - c0))
        if not self.quadrupole:
            return out
        sl = slice(i0, i1)
        lay = self.layer
        src = line if c0 == 0 else line.subset(slice(c0, None))
        for (_, tab), far in zip(self.groups, self.far, strict=True):
            if not far:
                continue
            Q = far_image_rows(
                line.mid[sl],
                line.tangent[sl],
                line.length[sl],
                src,
                far,
                obs_radius=line.radius[sl],
            )
            out += tab[lay[sl][:, None], lay[None, c0:]] * Q
        return out

    def sparse(self) -> SparseCorrection:
        """Blended Galerkin corrections of the near off-diagonal image pairs, in Ω."""
        from groundfield.solver._line_kernel import (
            SWITCH_FACTOR,
            _corrections,
            _depth_gap,
            switch_pairs,
        )

        line = self.line
        z = line.mid[:, 2]
        z_lo, z_hi = float(z.min()), float(z.max())
        reach = SWITCH_FACTOR * float(line.length.max())
        lay = self.layer
        parts = []
        for images, tab in self.groups:
            for weight, offset, sign in images:
                band = sorted((offset + sign * z_lo, offset + sign * z_hi))
                gap = _depth_gap(z, band[0], band[1])
                if gap >= reach:
                    continue
                img = line.image(offset, sign)
                i, j, w, far = switch_pairs(
                    line.mid,
                    line.length,
                    img.mid,
                    img.length,
                    gap,
                    image=(z, offset, sign),
                )
                coef = tab[lay[i], lay[j]]
                keep = (i != j) & (coef != 0.0)
                if not keep.any():
                    continue
                i, j, w, far, coef = i[keep], j[keep], w[keep], far[keep], coef[keep]
                point = _point_values(line.mid, img.mid, i, j, _MIN_DISTANCE)
                values = _corrections(
                    line,
                    line.mid,
                    line.length,
                    img,
                    i,
                    j,
                    w,
                    far,
                    point,
                    _MIN_DISTANCE,
                    with_quadrupole=self.quadrupole,
                    image=True,
                )
                parts.append(SparseCorrection(i, j, (weight * coef) * values))
        return SparseCorrection.concatenate(parts)

    def diagonal(self) -> np.ndarray:
        """Line-minus-point value of every segment against its own images, in Ω.

        Per image the Galerkin integral minus the point value at ``s = a``
        (``1/sqrt(a² + d²)``, the form of the spectral diagonal), tapered
        like the far-field term (:func:`~groundfield.solver._line_kernel.image_taper`
        of the distance of the segment to its image). The diagonal of ``ΔZ`` is assigned after
        the far-field rows (:meth:`rows`) are added, so it carries none of
        them.
        """
        line = self.line
        n = line.mid.shape[0]
        a = line.radius
        out = np.zeros(n)
        if not self.quadrupole:
            # The interface image at every distance, untapered (base of wave 3).
            for images, tab in self.groups:
                coef_diag = tab[self.layer, self.layer]
                for weight, offset, sign in images:
                    img = line.image(offset, sign)
                    dz = line.mid[:, 2] - img.mid[:, 2]
                    idx = np.arange(n)
                    exact = galerkin_pairs(line, idx, img, idx, transverse=True)
                    value = exact - 1.0 / np.sqrt(a * a + dz * dz)
                    out += (weight * coef_diag) * value
            return out
        for (_, tab), far in zip(self.groups, self.far, strict=True):
            coef_diag = tab[self.layer, self.layer]
            for (weight, offset, sign), shortest in zip(
                far.images, far.shortest, strict=True
            ):
                img = line.image(offset, sign)
                dz = line.mid[:, 2] - img.mid[:, 2]
                taper = image_taper(dz, line.length)
                # A segment's own image follows the rule of every pair: only
                # segments that reach the image (audit W3A-01; before, every
                # far image of a long segment gave every short one a Galerkin
                # self image, and its diagonal changed when a long piece was
                # added to the world) and are within its taper.
                idx = np.flatnonzero((line.length >= shortest) & (taper != 0.0))
                if not idx.size:
                    continue
                exact = galerkin_pairs(line, idx, img, idx, transverse=True)
                value = exact - 1.0 / np.sqrt(a[idx] * a[idx] + dz[idx] * dz[idx])
                out[idx] += (weight * coef_diag[idx]) * (taper[idx] * value)
        return out


def _layered_diagonal(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    rho_per_segment: np.ndarray,
    lower: np.ndarray,
    soil: TwoLayerSoil,
    line: LineSegments | None,
    *,
    families: _LayeredImageFamilies | None = None,
) -> np.ndarray:
    """Diagonal of the layered correction ``ΔZ`` in Ω.

    Point kernel: 3×3 Gauss-Legendre along the segment axis (line-self
    peak regularised at the wire radius: evaluated at s = a, whose 1/R
    terms need no further clamp — audit A3S-01, the h_1/200 clamp of the
    truncation plan took 68-84 % of the interface image of a part touching
    h_1). Line kernel: the point value of ``ΔG`` at ``s = a`` plus the
    exchange of its interface-image term for the Galerkin self image (the
    self image of a segment touching the interface is a touching collinear
    pair). Identical segments share one evaluation.
    """
    from groundfield.coupling.layered_green import (
        two_layer_layered_correction_real_space,
    )

    rho_1, rho_2, h_1 = soil.rho_1, soil.rho_2, soil.h_1
    n = seg_points.shape[0]
    out = np.empty(n)
    if line is not None:
        if families is None:
            families = _LayeredImageFamilies(line, lower, soil)
        images = families.diagonal()
        a = line.radius
        cache: dict = {}
        for k in range(n):
            key = (
                round(float(line.mid[k, 2]), 9),
                round(float(a[k]), 9),
                bool(lower[k]),
            )
            if key not in cache:
                # Evaluated at s = a without the h_1/200 clamp of the
                # truncation plan (``min_distance = a``), so that it contains
                # exactly the point image ``coeff / sqrt(a² + d²)`` exchanged
                # below (audit A3S-01: the clamp took 68-84 % of the layered
                # correction of every part within h_1/400 of the interface —
                # a ring at h_1 = 5 m read -5.29 Ohm).
                cache[key] = two_layer_layered_correction_real_space(
                    s=float(a[k]),
                    z=float(line.mid[k, 2]),
                    z_s=float(line.mid[k, 2]),
                    rho_1=rho_1,
                    rho_2=rho_2,
                    h_1=h_1,
                    rho_baseline=float(rho_per_segment[k]),
                    min_distance=float(a[k]),
                ) / (2.0 * np.pi)
            out[k] = cache[key] + images[k]
        return out

    gl3_nodes, gl3_weights = np.polynomial.legendre.leggauss(3)
    gl3_nodes = 0.5 * (gl3_nodes + 1.0)
    gl3_weights = 0.5 * gl3_weights
    diag_cache: dict = {}
    for i in range(n):
        z_i = float(seg_points[i, 2])
        L_i = float(seg_lengths[i])
        a_i = float(wire_radii[i])
        rho_baseline_i = float(rho_per_segment[i])
        key = (round(z_i, 9), round(L_i, 9), round(a_i, 9), round(rho_baseline_i, 9))
        if key in diag_cache:
            out[i] = diag_cache[key]
            continue
        z_lo = z_i - 0.5 * L_i
        delta_G = 0.0
        for a, w_a in zip(gl3_nodes, gl3_weights, strict=True):
            z_a = z_lo + a * L_i
            for b, w_b in zip(gl3_nodes, gl3_weights, strict=True):
                z_b = z_lo + b * L_i
                delta_G += (
                    w_a
                    * w_b
                    * two_layer_layered_correction_real_space(
                        s=a_i,
                        z=z_a,
                        z_s=z_b,
                        rho_1=rho_1,
                        rho_2=rho_2,
                        h_1=h_1,
                        rho_baseline=rho_baseline_i,
                        min_distance=a_i,
                    )
                )
        diag_cache[key] = delta_G / (2.0 * np.pi)
        out[i] = diag_cache[key]
    return out


class _PhiHomRows:
    r"""Rows of the reaction matrix for homog-with-per-source-rho soil.

    ``M[i, j]`` is the potential at segment midpoint *i* due to a
    uniform-line-current ``I_j`` distributed along segment *j*, computed in a
    homogeneous half-space whose resistivity equals ``rho_per_segment[j]``.

    This generalises :func:`_self_corrected_kernel`, which uses a single
    global ``rho``. The line-self formula on the diagonal uses
    ``rho_per_segment[i]`` (which equals ``[j]`` when ``i=j``); off-diagonal
    entries use ``rho_per_segment[j]`` — i.e. the resistivity of the source
    segment's layer — which gives the correct ``rho_local·1/r`` near-source
    behaviour in each column.

    With the exact line-source kernel (``line``), the difference between the
    segment average and the point form of the direct and surface-image terms
    is scaled, for a pair in different layers (``lower[i] != lower[j]``),
    with ``transmitted_rho`` :math:`= 2\rho_1\rho_2/(\rho_1 + \rho_2)` instead
    of the source resistivity: that is the coefficient of the transmitted
    direct and surface-image terms of the two-layer Green's function, whose
    point form the layered correction completes. Scaling it with the source
    resistivity made the reaction matrix of a rod through the interface
    3.8 % asymmetric and a two-port with it non-reciprocal by up to 4.7 %
    (finding of stream eng2 on the 2026.11 work).

    Rows are built component-wise (10.56): entries bit-identical to the
    former ``(N, N, 3)`` difference tensors, 8 instead of 96 bytes per
    segment pair.
    """

    def __init__(
        self,
        seg_points: np.ndarray,
        seg_lengths: np.ndarray,
        wire_radii: np.ndarray,
        rho_per_segment: np.ndarray,
        *,
        line: LineSegments | None = None,
        transmitted_rho: float | None = None,
        lower: np.ndarray | None = None,
    ) -> None:
        self.n = seg_points.shape[0]
        self.xyz = _xyz(seg_points)
        # Diagonal: line-self + image-point. The image term is guarded
        # instead of clamped (audit 2026-07-28, F26/F29): a segment whose
        # mirror image touches the conductor has no finite self-image
        # term, and the historic ``_MIN_DISTANCE`` clamp turned that into
        # a silently mesh-dependent, far too large impedance.
        diag_direct = point_self_term(seg_lengths, wire_radii)
        _require_image_separation(
            seg_points, wire_radii, where="_build_phi_hom_per_source_rho"
        )
        diag_image = 1.0 / (2.0 * np.abs(seg_points[:, 2]))
        self.diag = diag_direct + diag_image
        self.rho_per_segment = rho_per_segment
        self.scale = rho_per_segment[None, :] / (4.0 * np.pi)
        self.line = line
        self.transmitted_rho = transmitted_rho
        self.lower = lower
        # Exact line-source kernel (10.28): near-pair corrections of the
        # direct and surface-image terms (sparse, once).
        self.corrections = (
            reaction_corrections(line, _MIN_DISTANCE) if line is not None else None
        )

    def rows(self, i0: int, i1: int, out: np.ndarray | None = None) -> np.ndarray:
        """Rows ``i0:i1`` in Ω (into ``out`` when given)."""
        xyz = self.xyz
        obs = (xyz[0][i0:i1], xyz[1][i0:i1], xyz[2][i0:i1])
        blk, _ = _half_space_block(obs, xyz, _MIN_DISTANCE, out=out)
        idx = np.arange(i0, i1)
        blk[idx - i0, idx] = self.diag[i0:i1]
        if self.line is None:
            # Apply rho_per_segment per column (source segment), in place:
            # M[i, j] = rho_per_segment[j] / (4π) · kernel[i, j]
            blk *= self.scale
            return blk
        # Far-field and near-pair corrections of the direct and surface-image
        # terms, with the transmitted resistivity for pairs in different
        # layers.
        corr = reaction_quadrupole_rows(self.line, i0, i1)
        self.corrections.add_to_rows(corr, i0)
        blk += corr
        blk *= self.scale
        if self.transmitted_rho is not None and self.lower is not None:
            cross = self.lower[i0:i1, None] != self.lower[None, :]
            blk += np.where(
                cross,
                corr
                * (
                    (self.transmitted_rho - self.rho_per_segment[None, :])
                    / (4.0 * np.pi)
                ),
                0.0,
            )
        return blk

    def matrix(self) -> np.ndarray:
        """The full ``N × N`` matrix in Ω."""
        kernel = np.empty((self.n, self.n))
        rows = _block_rows(self.n)
        for i0 in range(0, self.n, rows):
            i1 = min(self.n, i0 + rows)
            self.rows(i0, i1, out=kernel[i0:i1])
        return kernel


def _build_phi_hom_per_source_rho(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    rho_per_segment: np.ndarray,
    *,
    line: LineSegments | None = None,
    transmitted_rho: float | None = None,
    lower: np.ndarray | None = None,
) -> np.ndarray:
    """Reaction matrix for homog-with-per-source-rho soil (see :class:`_PhiHomRows`)."""
    return _PhiHomRows(
        seg_points,
        seg_lengths,
        wire_radii,
        rho_per_segment,
        line=line,
        transmitted_rho=transmitted_rho,
        lower=lower,
    ).matrix()


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def _as_two_layer(soil: MultiLayerSoil) -> TwoLayerSoil:
    """The :class:`TwoLayerSoil` of a stack with at most two distinct layers.

    Adjacent layers of equal resistivity are merged first (the canonical
    stack, review pass 10, 10.48); a stack that merges to one layer becomes
    ``TwoLayerSoil(rho, rho, h)`` with the declared first thickness, which
    reproduces the homogeneous result.

    Raises
    ------
    BackendCapabilityError
        If the stack has three or more distinct layers.
    """
    from groundfield.solver._layered import as_layer_stack
    from groundfield.solver.registry import BackendCapabilityError

    stack = as_layer_stack(soil)
    if stack.n_layers == 2:
        return TwoLayerSoil(
            rho_1=float(stack.rhos[0]),
            rho_2=float(stack.rhos[1]),
            h_1=float(stack.h[0]),
        )
    if stack.n_layers == 1:
        h_1 = soil.layers[0].h if soil.layers[0].h else 1.0
        rho = float(stack.rhos[0])
        return TwoLayerSoil(rho_1=rho, rho_2=rho, h_1=float(h_1))
    raise BackendCapabilityError(
        f"Backend 'image_2layer' solves at most two distinct layers; this "
        f"MultiLayerSoil has {stack.n_layers}. Use backend='mom_sommerfeld'."
    )


def _owner_interfaces(soil: TwoLayerSoil) -> tuple[float, ...]:
    """Interfaces at which electrodes get one owner per layer (10.15).

    Only an interface where the resistivity changes
    (:func:`~groundfield.solver._layered.interface_depths`):
    ``TwoLayerSoil(rho, rho, h)`` is homogeneous soil and keeps one owner
    per electrode, so it reproduces the homogeneous backend exactly.
    """
    from groundfield.solver._layered import interface_depths

    return interface_depths(soil)


def _crosses_interface(world: "World", layer_interfaces) -> bool:
    """Whether any electrode's depth range strictly contains an interface."""
    for e in world.electrodes:
        lower, upper = e.bounds()
        if any(lower[2] < h < upper[2] for h in layer_interfaces):
            return True
    return False


class _TwoLayerGreen:
    """Green's function of the two-layer soil (pipeline object).

    Inside the upper layer the reaction kernel is the Tagg/Sunde image
    series (:class:`~groundfield.solver._series.ImageSeries`); as soon
    as a segment reaches the interface the kernel switches to the
    rigorous Sommerfeld path of ADR-0007 for every pair.
    """

    backend = "image_2layer"
    where = "solve_image_2layer"

    def __init__(
        self,
        soil: TwoLayerSoil,
        max_terms: int,
        tol: float,
        ds: float,
        kernel: str = "point",
        far_field: str = "quadrupole",
    ):
        self.soil = soil
        self.max_terms = max_terms
        self.tol = tol
        self.ds = ds
        self.kernel = kernel
        self.far_field = far_field
        # Wires are cut where the resistivity changes only, like in every
        # other backend: TwoLayerSoil(rho, rho, h) is cut nowhere, as
        # homogeneous soil (audit round 2, A2S-07).
        self.layer_interfaces = _owner_interfaces(soil) or None
        self.soil_resistivity = float(soil.rho_1)
        self.reference_resistivity = float(soil.rho_1)
        self.series = ImageSeries.for_soil(soil, max_terms, tol)
        # Set by ``prepare``: whether the reaction kernel runs on the
        # ADR-0007 Sommerfeld path (then the image series is not used).
        self.cross_layer = False

    def prepare(self, segments, arrays, *, stacklevel: int):  # noqa: ANN001
        """Detect cross-layer geometry and build the reaction kernel."""
        h_1 = self.soil.h_1
        ds = self.ds
        # ADR-0007: when all segments live inside the upper layer the
        # historic Tagg/Sunde image series is used (bit-exact
        # regression). When any segment is at or below the interface,
        # the kernel dispatches to the rigorous Sommerfeld path.
        z_max = arrays.points[:, 2].max()
        cross_layer = bool(z_max >= h_1)
        self.cross_layer = cross_layer
        if cross_layer:
            _log.info(
                "image_2layer: cross-layer geometry detected "
                "(z_max = %.3f m, h_1 = %.3f m). Using ADR-0007 "
                "Sommerfeld fallback for all source/observer pairs.",
                z_max,
                h_1,
            )
        elif self.kernel != "line" and h_1 - z_max < 2.0 * ds:
            # Audit 2026-07-08 WP-D2: within ~one segment of the
            # interface the point-source treatment of the K-weighted
            # nearest image (distance 2(h_1 − z)) degrades. Surface the
            # margin so the user can refine ds or accept the bias. The
            # line kernel integrates that image (a strip 0.1 m above the
            # interface is ds-independent to 4e-6, a ring one wire radius
            # above within 4.1e-5 of its exact value) and does not warn
            # (audit W3A-06).
            warnings.warn(
                f"image_2layer: deepest segment ends {h_1 - z_max:.2f} m "
                f"above the layer interface (h_1 = {h_1:.2f} m), which is "
                f"less than 2×segment_length ({2.0 * ds:.2f} m). The "
                "image-series self-terms near the interface carry an "
                "elevated discretisation error there — consider a finer "
                "segment_length.",
                UserWarning,
                stacklevel=stacklevel,
            )
        if self.kernel != "line" and self.layer_interfaces:
            # The point form of the diagonal cannot represent a horizontal
            # wire next to h_1 on either side of it: the 3-point depth rule
            # of the cross-layer path (audit round 2), the point image of the
            # interface on the series path (audit A3S-02). Without a physical
            # interface (rho_1 == rho_2) nothing is to be checked.
            _check_point_interface_diagonal(
                segments,
                arrays.points,
                arrays.lengths,
                arrays.directions,
                h_1,
                backend="'image_2layer' (kernel='point')",
                remedy="Use the default kernel (kernel=None or 'line'), which "
                "integrates these segments exactly, or a finer segment_length",
                cross_layer=cross_layer,
            )
        # Tag every segment with its layer index (0 = upper, 1 = lower).
        for s in segments:
            s.layer_index = 0 if s.midpoint[2] < h_1 else 1
        line = arrays.line_segments() if self.kernel == "line" else None
        return _two_layer_self_kernel_factory(
            self.soil,
            self.max_terms,
            self.tol,
            allow_cross_layer=cross_layer,
            line=line,
            far_field=self.far_field,
        )

    def metadata(self, solution, *, stacklevel: int) -> dict:  # noqa: ANN001
        """Soil parameters and the series convergence report.

        The report describes the series the solve actually evaluated.
        On the ADR-0007 cross-layer path every pair runs through the
        Sommerfeld kernel and the image series is not used at all:
        ``series_used`` is ``False``, ``n_terms_used`` is ``0`` and no
        :class:`SeriesTruncationWarning` is emitted (review pass 10,
        10.50 / F43; before, such solves reported the term count of a
        series that never ran and warned about its truncation).
        ``series`` records the controls for the post-solve field
        evaluation (:meth:`FieldResult.potential`): ``max_terms``, ``tol``
        and ``far_field`` (``Engine.image_far_field``).
        """
        soil = self.soil
        K = soil.reflection_coefficient
        series_used = not self.cross_layer and any(
            np.any(sc) for sc in solution.segment_currents
        )
        n_terms_used = self.series.n_terms if series_used else 0
        converged = (
            _series_tail_bound(abs(K), n_terms_used) < self.tol
            if n_terms_used
            else True
        )
        if not converged:
            # One channel only (10.70): the warning, no duplicate log record.
            self.series.warn_if_truncated("image_2layer", stacklevel=stacklevel)
        return {
            "K": float(K),
            "rho_1": float(soil.rho_1),
            "rho_2": float(soil.rho_2),
            "h_1": float(soil.h_1),
            "n_terms_used": n_terms_used,
            "converged": converged,
            "series_used": series_used,
            "series": {
                "max_terms": int(self.max_terms),
                "tol": float(self.tol),
                "far_field": str(self.far_field),
            },
        }


def solve_image_2layer(
    world: "World",
    engine: "Engine",
    *,
    max_terms: int | None = None,
    tol: float | None = None,
    far_field: str | None = None,
) -> FieldResult:
    """Image-charge solver for 2-layer soil (Tagg / Sunde).

    Runs the shared image-family pipeline
    (:mod:`groundfield.solver._pipeline`) with the two-layer Green's
    function.

    Parameters
    ----------
    world
        World whose ``soil`` is a :class:`TwoLayerSoil`. Electrodes in
        the upper layer use the Tagg/Sunde series; geometries reaching
        the interface use the ADR-0007 Sommerfeld path.
    engine
        Engine configuration. ``engine.segment_length`` controls the
        discretisation as in the homogeneous backend.
    max_terms
        Maximum number of series terms; ``None`` (default) takes
        ``engine.image_max_terms``.
    tol
        Series truncation: stop as soon as the geometric tail bound
        $|K|^{n+1}/(1-|K|)$ falls below ``tol`` (dimensionless);
        ``None`` (default) takes ``engine.image_series_tol``.
    far_field
        Form of the far images with the line kernel, ``"quadrupole"`` or
        ``"point"``; ``None`` (default) takes ``engine.image_far_field``.

    Returns
    -------
    FieldResult
        Result object. :attr:`FieldResult.metadata` exposes diagnostic
        fields (``K``, ``rho_1``, ``rho_2``, ``h_1``, ``n_terms_used``,
        ``converged``, ``series_used``, ``series``).

    Raises
    ------
    BackendCapabilityError
        If the soil is a :class:`HomogeneousSoil` or a
        :class:`~groundfield.soil.models.MultiLayerSoil` with three or
        more distinct layers (a stack with at most two is solved as the
        equivalent two-layer soil). Before 2026.11 both raised
        :class:`TypeError`.
    TypeError
        If ``world.soil`` is no soil model.
    ValueError
        If the world has no electrodes, or the geometry fails a
        discretisation guard (thin wire, burial depth).

    Warns
    -----
    SeriesTruncationWarning
        If the series is used and reaches ``max_terms`` before ``tol``.
    """
    if isinstance(world.soil, MultiLayerSoil):
        # Audit round 1, A1-numerics F09: the registry accepts a layer
        # stack that merges to at most two layers; solve it as the
        # equivalent TwoLayerSoil instead of raising a bare TypeError.
        world = world.model_copy(update={"soil": _as_two_layer(world.soil)})
    if isinstance(world.soil, HomogeneousSoil):
        from groundfield.solver.registry import BackendCapabilityError

        raise BackendCapabilityError(
            "Backend 'image_2layer' solves two-layer soil; got a "
            "HomogeneousSoil. Use backend='image' (Engine(backend='image')"
            ".solve dispatches by soil model)."
        )
    if not isinstance(world.soil, TwoLayerSoil):
        raise TypeError(
            "Backend 'image_2layer' requires a TwoLayerSoil. "
            f"Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    soil: TwoLayerSoil = world.soil
    _log.info(
        "image_2layer: rho_1=%.1f, rho_2=%.1f, h_1=%.2f, K=%.4f",
        soil.rho_1,
        soil.rho_2,
        soil.h_1,
        soil.reflection_coefficient,
    )
    if max_terms is None:
        max_terms = engine.image_max_terms
    if tol is None:
        tol = engine.image_series_tol
    if far_field is None:
        far_field = engine.image_far_field
    if engine.kernel is None:
        # The backend's preferred kernel (Engine.kernel = None, 10.28).
        engine = engine._with(kernel=DEFAULT_KERNEL)
    green = _TwoLayerGreen(
        soil, max_terms, tol, engine.segment_length, engine.kernel, far_field
    )
    owner_interfaces = None
    if _crosses_interface(world, _owner_interfaces(soil)):
        # One owner per (electrode, layer) for rods through h_1 (10.15), on
        # top of the one owner per straight wire of every world (Q30).
        owner_interfaces = _owner_interfaces(soil)
    return solve_image_family(
        world, engine, green, stacklevel=3, owner_interfaces=owner_interfaces
    )
