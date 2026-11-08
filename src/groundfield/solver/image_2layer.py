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
from groundfield.solver._discretize import _LayerOwners
from groundfield.solver._guards import (
    _MIN_DISTANCE,
    _check_segment_depths,
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
    LineSegments,
    _point_values,
    galerkin_pairs,
    near_pairs,
    point_self_term,
    reaction_corrections,
    reaction_quadrupole_rows,
)
from groundfield.solver._pipeline import (
    SegmentArrays,
    assemble_branches,
    build_result,
    build_topology,
    discretise,
    inject,
    solve,
    solve_image_family,
)
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
    ) -> None:
        self.soil = soil
        self.allow_cross_layer = allow_cross_layer
        self.line = line
        self.series = ImageSeries.for_soil(soil, max_terms, tol)
        self.in_layer = TwoLayerSeriesKernel(soil.rho_1, self.series, line=line)
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
        return bool(self._cross_cache) or self.in_layer.materialised

    @property
    def assemblies(self) -> int:
        """Number of full reaction-matrix evaluations so far."""
        return self.in_layer.assemblies + int(self._cross_cache.get("assemblies", 0))

    def matrix(
        self, seg_points: np.ndarray, seg_lengths: np.ndarray, wire_radii: np.ndarray
    ) -> np.ndarray:
        """Reaction matrix in Ω, bit-identical to v0.15.0's ``kernel(eye)``."""
        if self._cross_layer(seg_points):
            phi_hom_matrix, delta_Z = _layered_green_matrices(
                seg_points,
                seg_lengths,
                wire_radii,
                soil=self.soil,
                cache=self._cross_cache,
                line=self.line,
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
            return _layered_green_kernel(
                seg_points,
                seg_lengths,
                wire_radii,
                currents,
                soil=self.soil,
                cache=self._cross_cache,
                line=self.line,
            )
        return self.in_layer(seg_points, seg_lengths, wire_radii, currents)


def _two_layer_self_kernel_factory(
    soil: TwoLayerSoil,
    max_terms: int,
    tol: float,
    *,
    allow_cross_layer: bool = False,
    line: LineSegments | None = None,
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

    Returns
    -------
    _TwoLayerKernel
        Callable ``(seg_points, seg_lengths, wire_radii, currents) ->
        phi`` with a ``matrix(...)`` method for the full reaction
        matrix.
    """
    return _TwoLayerKernel(
        soil, max_terms, tol, allow_cross_layer=allow_cross_layer, line=line
    )


def _layered_green_matrices(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    *,
    soil: TwoLayerSoil,
    cache: dict | None = None,
    line: LineSegments | None = None,
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
    from groundfield.coupling.layered_green import (
        two_layer_layered_correction_real_space,
    )

    rho_1 = soil.rho_1
    rho_2 = soil.rho_2
    h_1 = soil.h_1

    n = seg_points.shape[0]
    # ADR-0007 Phase A.1: per-segment baseline rho. Each source
    # segment j gets its layer-local rho as the homogeneous-soil
    # baseline; the layered correction is then the *smooth*
    # deviation from that baseline (no residual 1/r divergence
    # from a rho mismatch). For pure-upper-layer worlds this
    # collapses to the historic rho_1 baseline.
    rho_per_segment = np.where(
        seg_points[:, 2] < h_1,
        rho_1,
        rho_2,
    )

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
        )
        if cache.get("key") == cache_key:
            return cache["phi_hom_matrix"], cache["delta_Z"]

    # Step 1: build phi_hom matrix that uses rho_at_source(j) for
    # each column (source segment). This generalises
    # _self_corrected_kernel: the diagonal still uses the line-
    # self formula with rho_local(i), and off-diagonal entries use
    # the point-source 1/r + 1/r_image formula with rho_local(j).
    phi_hom_matrix = _build_phi_hom_per_source_rho(
        seg_points,
        seg_lengths,
        wire_radii,
        rho_per_segment,
        line=line,
        transmitted_rho=2.0 * rho_1 * rho_2 / (rho_1 + rho_2),
        lower=seg_points[:, 2] >= h_1,
    )

    # Step 2: layered correction. The baseline rho per source is
    # rho_at_source(j). For the diagonal (i = j) we still need to
    # handle the residual line-self peak by 3×3 Gauss-Legendre
    # averaging along the segment axis — this is small but
    # important for short segments in highly contrasting soils.
    #
    # Off-diagonal assembly is grouped by the (z_i, z_j, baseline)
    # combination and evaluated with ONE spectral-amplitude solve per
    # group (audit 2026-07-08, WP-D performance note): the historic
    # per-entry scalar quadrature made the cross-layer reaction
    # matrix O(n²) BVP solves, which dominated multi-electrode
    # cross-layer worlds after the WP-B1 oscillation-resolved grids.
    from groundfield.coupling.layered_green import (
        two_layer_layered_correction_group,
    )

    delta_Z = np.zeros((n, n), dtype=float)
    z_keys = np.round(seg_points[:, 2] / 1e-9).astype(np.int64)
    for zi_key in np.unique(z_keys):
        rows = np.where(z_keys == zi_key)[0]
        z_i = float(seg_points[rows[0], 2])
        for zj_key in np.unique(z_keys):
            cols = np.where(z_keys == zj_key)[0]
            z_j = float(seg_points[cols[0], 2])
            rho_baseline_j = float(rho_per_segment[cols[0]])
            # Horizontal distances component-wise (10.56): bit-identical
            # to the former (rows, cols, 2) difference tensor and
            # ``einsum``, two instead of four ``rows × cols`` arrays.
            s_grid = np.subtract.outer(seg_points[rows, 0], seg_points[cols, 0])
            s_grid *= s_grid
            dy2 = np.subtract.outer(seg_points[rows, 1], seg_points[cols, 1])
            dy2 *= dy2
            s_grid += dy2
            del dy2
            np.sqrt(s_grid, out=s_grid)
            vals = two_layer_layered_correction_group(
                s_grid.ravel(),
                z_i,
                z_j,
                rho_1=rho_1,
                rho_2=rho_2,
                h_1=h_1,
                rho_baseline=rho_baseline_j,
            )
            delta_Z[np.ix_(rows, cols)] = vals.reshape(s_grid.shape) / (2.0 * np.pi)

    if line is not None:
        # Exact line sources (10.28): the near-singular interface image
        # of the layered correction as a Galerkin integral, the rest of
        # it in point form (diagonal included).
        _add_line_interface_images(
            delta_Z,
            line,
            rho_per_segment,
            seg_points[:, 2] >= h_1,
            soil,
        )
        if cache is not None:
            cache["key"] = cache_key
            cache["phi_hom_matrix"] = phi_hom_matrix
            cache["delta_Z"] = delta_Z
            cache["assemblies"] = int(cache.get("assemblies", 0)) + 1
        return phi_hom_matrix, delta_Z

    # Diagonal: 3×3 Gauss-Legendre along the segment axis (line-self
    # peak regularised at the wire radius). Identical segments (same
    # depth, length, radius, baseline) share one evaluation.
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
            delta_Z[i, i] = diag_cache[key]
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
                    )
                )
        diag_cache[key] = delta_G / (2.0 * np.pi)
        delta_Z[i, i] = diag_cache[key]

    if cache is not None:
        cache["key"] = cache_key
        cache["phi_hom_matrix"] = phi_hom_matrix
        cache["delta_Z"] = delta_Z
        cache["assemblies"] = int(cache.get("assemblies", 0)) + 1
    return phi_hom_matrix, delta_Z


def _layered_green_kernel(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    currents,
    *,
    soil: TwoLayerSoil,
    cache: dict | None = None,
    line: LineSegments | None = None,
):
    """ADR-0007 cross-layer reaction kernel: ``(Φ_hom + ΔZ) · currents``.

    See :func:`_layered_green_matrices` for the decomposition; the two
    matrices are taken from ``cache`` when the geometry is unchanged.
    """
    phi_hom_matrix, delta_Z = _layered_green_matrices(
        seg_points, seg_lengths, wire_radii, soil=soil, cache=cache, line=line
    )
    return phi_hom_matrix @ currents + delta_Z @ currents


def _add_line_interface_images(
    delta_Z: np.ndarray,
    line: LineSegments,
    rho_per_segment: np.ndarray,
    lower: np.ndarray,
    soil: TwoLayerSoil,
) -> None:
    r"""Line-exact interface image of the layered correction, in place.

    For two points in the same layer the layered correction
    :math:`\Delta G` contains the image of the source mirrored in the
    interface, :math:`c_j / (4\pi R')` with :math:`c_j = \rho_1 K`
    (source in layer 1) or :math:`-\rho_2 K` (layer 2),
    :math:`K = (\rho_2 - \rho_1)/(\rho_2 + \rho_1)`, whose distance
    :math:`R'` vanishes as both points approach the interface. Near the
    interface its point form misses the segment average by far more than
    the rest of :math:`\Delta G`; here it is replaced by the Galerkin
    integral of the segment and the mirrored source segment for every
    pair nearer than :data:`~groundfield.solver._line_kernel.NEAR_FACTOR`
    segment lengths. The diagonal becomes the point value of
    :math:`\Delta G` at :math:`s = a` plus the same exchange of its
    interface-image term (the self image of a segment touching the
    interface is a touching collinear pair).

    The image term keeps the reaction matrix symmetric: the integral of
    segment :math:`i` against the image of :math:`j` equals that of
    :math:`j` against the image of :math:`i`.

    Parameters
    ----------
    delta_Z
        Layered correction in Ω with point-form off-diagonal entries,
        shape ``(N, N)``; modified in place (the diagonal is replaced).
    line
        Segment geometry.
    rho_per_segment
        Resistivity of the layer of each segment in Ω·m (the baseline).
    lower
        Whether each segment lies in the lower layer.
    soil
        The two-layer soil.
    """
    from groundfield.coupling.layered_green import (
        two_layer_layered_correction_real_space,
    )

    rho_1, rho_2, h_1 = soil.rho_1, soil.rho_2, soil.h_1
    K = (rho_2 - rho_1) / (rho_2 + rho_1)
    coeff = np.where(lower, -rho_2 * K, rho_1 * K) / (4.0 * np.pi)
    image = line.image(2.0 * h_1, -1.0)

    i, j = near_pairs(line.mid, line.length, image.mid, image.length)
    keep = (lower[i] == lower[j]) & (i != j)
    i, j = i[keep], j[keep]
    if i.size:
        exact = galerkin_pairs(line, i, image, j)
        point = _point_values(line.mid, image.mid, i, j, _MIN_DISTANCE)
        np.add.at(delta_Z, (i, j), coeff[j] * (exact - point))

    n = line.length.shape[0]
    diag = np.arange(n)
    self_image = galerkin_pairs(line, diag, image, diag)
    a = line.radius
    d = 2.0 * np.abs(h_1 - line.mid[:, 2])
    point_image = 1.0 / np.sqrt(a * a + d * d)
    cache: dict = {}
    for k in range(n):
        key = (round(float(line.mid[k, 2]), 9), round(float(a[k]), 9), bool(lower[k]))
        if key not in cache:
            cache[key] = two_layer_layered_correction_real_space(
                s=float(a[k]),
                z=float(line.mid[k, 2]),
                z_s=float(line.mid[k, 2]),
                rho_1=rho_1,
                rho_2=rho_2,
                h_1=h_1,
                rho_baseline=float(rho_per_segment[k]),
            ) / (2.0 * np.pi)
        delta_Z[k, k] = cache[key] + coeff[k] * (self_image[k] - point_image[k])


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
    """Reaction matrix for homog-with-per-source-rho soil.

    Returns a real ``(n, n)`` matrix ``M`` such that
    ``phi[i] = sum_j M[i, j] · I_j`` is the potential at segment
    midpoint *i* due to a uniform-line-current ``I_j`` distributed
    along segment *j*, computed in a homogeneous half-space whose
    resistivity equals ``rho_per_segment[j]``.

    This generalises :func:`_self_corrected_kernel`, which uses a
    single global ``rho``. The line-self formula on the diagonal
    uses ``rho_per_segment[i]`` (which equals ``[j]`` when ``i=j``);
    off-diagonal entries use ``rho_per_segment[j]`` — i.e. the
    resistivity of the source segment's layer — which gives the
    correct ``rho_local·1/r`` near-source behaviour in each
    column.

    With the exact line-source kernel (``line``), the difference between
    the segment average and the point form of the direct and
    surface-image terms is scaled, for a pair in different layers
    (``lower[i] != lower[j]``), with ``transmitted_rho``
    :math:`= 2\\rho_1\\rho_2/(\\rho_1 + \\rho_2)` instead of the source
    resistivity: that is the coefficient of the transmitted direct and
    surface-image terms of the two-layer Green's function, whose point
    form the layered correction completes. Scaling it with the source
    resistivity made the reaction matrix of a rod through the interface
    3.8 % asymmetric and a two-port with it non-reciprocal by up to 4.7 %
    (finding of stream eng2 on the 2026.11 work).
    """
    # Built component-wise in row blocks (10.56): entries bit-identical
    # to the former (N, N, 3) difference tensors, 8 instead of 96 bytes
    # per segment pair. Off-diagonal: 1/r + 1/r_image.
    kernel = np.empty((seg_points.shape[0], seg_points.shape[0]))
    xyz = _xyz(seg_points)
    rows = _block_rows(seg_points.shape[0])
    for i0 in range(0, seg_points.shape[0], rows):
        i1 = min(seg_points.shape[0], i0 + rows)
        obs = (xyz[0][i0:i1], xyz[1][i0:i1], xyz[2][i0:i1])
        _half_space_block(obs, xyz, _MIN_DISTANCE, out=kernel[i0:i1])

    # Diagonal: line-self + image-point. The image term is guarded
    # instead of clamped (audit 2026-07-28, F26/F29): a segment whose
    # mirror image touches the conductor has no finite self-image
    # term, and the historic ``_MIN_DISTANCE`` clamp turned that into
    # a silently mesh-dependent, far too large impedance.
    diag_direct = point_self_term(seg_lengths, wire_radii)
    _require_image_separation(
        seg_points, wire_radii, where="_build_phi_hom_per_source_rho"
    )
    z_mid = seg_points[:, 2]
    diag_image = 1.0 / (2.0 * np.abs(z_mid))
    np.fill_diagonal(kernel, diag_direct + diag_image)
    scale = rho_per_segment[None, :] / (4.0 * np.pi)
    if line is not None:
        # Exact line-source kernel (10.28): far-field and near-pair
        # corrections of the direct and surface-image terms, with the
        # transmitted resistivity for pairs in different layers.
        corrections = reaction_corrections(line, _MIN_DISTANCE)
        for i0 in range(0, kernel.shape[0], rows):
            blk = kernel[i0 : i0 + rows]
            corr = reaction_quadrupole_rows(line, i0, i0 + blk.shape[0])
            corrections.add_to_rows(corr, i0)
            blk += corr
            blk *= scale
            if transmitted_rho is not None and lower is not None:
                cross = lower[i0 : i0 + blk.shape[0], None] != lower[None, :]
                blk += np.where(
                    cross,
                    corr
                    * ((transmitted_rho - rho_per_segment[None, :]) / (4.0 * np.pi)),
                    0.0,
                )
        return kernel

    # Apply rho_per_segment per column (source segment), in place
    # (10.56; the product is commutative, so bit-identical):
    # M[i, j] = rho_per_segment[j] / (4π) · kernel[i, j]
    kernel *= scale
    return kernel


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

    Only an interface where the resistivity changes: ``TwoLayerSoil(rho,
    rho, h)`` is homogeneous soil and keeps one owner per electrode, so it
    reproduces the homogeneous backend exactly.
    """
    return (float(soil.h_1),) if soil.rho_1 != soil.rho_2 else ()


def _crosses_interface(world: "World", layer_interfaces) -> bool:
    """Whether any electrode's depth range strictly contains an interface."""
    for e in world.electrodes:
        lower, upper = e.bounds()
        if any(lower[2] < h < upper[2] for h in layer_interfaces):
            return True
    return False


def _solve_with_layer_owners(
    world: "World", engine: "Engine", green: "_TwoLayerGreen", *, stacklevel: int
) -> FieldResult:
    """The image-family pipeline with one owner per (electrode, layer) (10.15).

    The steps of :func:`~groundfield.solver._pipeline.solve_image_family`,
    with the owner split of
    :meth:`~groundfield.solver._discretize._LayerOwners.split` between the
    topology and the solve and its merge before the result is built. Used
    only for worlds with an electrode through an interface; every other
    world runs through :func:`solve_image_family` itself.

    Merge note: the pipeline step functions are shared, so changes inside
    them reach this path as well; a step added to ``solve_image_family``
    must be added here too (or the split / merge become a hook of
    ``solve_image_family``).
    """
    disc = discretise(world, engine, green.layer_interfaces)
    node_currents = inject(world, where=green.where, stacklevel=stacklevel)
    topo = build_topology(world, disc)
    owners = _LayerOwners.split(
        disc, topo, [e.name for e in world.electrodes], _owner_interfaces(green.soil)
    )
    branches = assemble_branches(world, engine, topo)
    _check_segment_depths(disc.segments, where=green.where, stacklevel=stacklevel + 1)
    arrays = SegmentArrays.from_segments(disc.segments)
    kernel = green.prepare(disc.segments, arrays, stacklevel=stacklevel + 1)
    solution = solve(
        world,
        engine,
        disc=disc,
        node_currents=node_currents,
        topo=topo,
        branches=branches,
        arrays=arrays,
        kernel=kernel,
    )
    owners.merge(disc, topo, solution.owner_currents)
    metadata = {
        "world_name": world.name,
        "n_segments": len(disc.segments),
        "segment_length": engine.segment_length,
        **green.metadata(solution, stacklevel=stacklevel + 1),
        "stub": False,
        "earth_inductive_model": branches.earth_inductive_model,
        "kernel": engine.kernel,
        "layer_owners": {
            name: len(subs) + 1 for name, subs in owners.sub_owners.items()
        },
    }
    return build_result(
        world,
        engine,
        green=green,
        disc=disc,
        branches=branches,
        arrays=arrays,
        solution=solution,
        cluster_id=topo.cluster_id,
        metadata=metadata,
    )


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
    ):
        self.soil = soil
        self.max_terms = max_terms
        self.tol = tol
        self.ds = ds
        self.kernel = kernel
        self.layer_interfaces = (soil.h_1,)
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
        elif h_1 - z_max < 2.0 * ds:
            # Audit 2026-07-08 WP-D2: within ~one segment of the
            # interface the point-source treatment of the K-weighted
            # nearest image (distance 2(h_1 − z)) degrades. Surface the
            # margin so the user can refine ds or accept the bias.
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
        evaluation (:meth:`FieldResult.potential`).
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
            "series": {"max_terms": int(self.max_terms), "tol": float(self.tol)},
        }


def solve_image_2layer(
    world: "World",
    engine: "Engine",
    *,
    max_terms: int | None = None,
    tol: float | None = None,
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
    if engine.kernel is None:
        # The backend's preferred kernel (Engine.kernel = None, 10.28).
        engine = engine.model_copy(update={"kernel": DEFAULT_KERNEL})
    green = _TwoLayerGreen(soil, max_terms, tol, engine.segment_length, engine.kernel)
    if _crosses_interface(world, _owner_interfaces(soil)):
        # One owner per (electrode, layer) for rods through h_1 (10.15).
        return _solve_with_layer_owners(world, engine, green, stacklevel=3)
    return solve_image_family(world, engine, green, stacklevel=3)
