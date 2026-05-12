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
Series terms decay as $K^n$ (geometric). For $|K| < 1$,
typically $n \\lesssim 50$ is enough for a relative accuracy of
$10^{-6}$. The implementation truncates after ``max_terms``
terms or as soon as $|K|^n < \\text{tol}$. If ``max_terms`` is
hit without reaching the tolerance, a warning is recorded in
``FieldResult.metadata``.

Preconditions
-------------
- All electrodes must lie completely in the **upper layer**
  ($z_\\text{seg} < h_1$); otherwise the backend raises a clear
  ``ValueError``.
- For $|K| \\to 1$ (extreme contrasts, e.g. wet soil over rock)
  convergence slows down significantly. The solver issues a log
  warning in that case.

References
----------
- Tagg, G. F. (1964). *Earth Resistances*. Pitman, ch. 5.
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*. Dover, sect. 3.5.
- ADR-0001 ``docs/adr/0001-two-layer-method.md``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from groundfield.soil.models import HomogeneousSoil, TwoLayerSoil
from groundfield.solver.image import (
    _MIN_DISTANCE,
    _assemble_inductance_matrix,
    _build_clusters,
    _build_distributed_topology,
    _build_finite_branches,
    _discretize_electrode,
    _self_corrected_kernel,
    _Segment,
    _solve_cluster_currents,
)
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_image_2layer"]

_log = get_logger(__name__)


# ---------------------------------------------------------------------
# 2-layer kernel
# ---------------------------------------------------------------------


def _two_layer_image_offsets(
    K: float, h_1: float, max_terms: int, tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Generate the series coefficients for the image-charge sum.

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
    pairs: list[tuple[float, int, float]] = [
        (0.0, +1, 1.0),  # z = +z_src
        (0.0, -1, 1.0),  # z = -z_src
    ]

    abs_K = abs(K)
    n_terms_used = 0
    for n in range(1, max_terms + 1):
        K_n = K ** n
        # Four images per n
        for sign_n in (+1, -1):
            for sign_zs in (+1, -1):
                pairs.append((sign_n * 2.0 * n * h_1, sign_zs, K_n))
        n_terms_used = n
        if abs_K ** n < tol:
            break

    z_offsets = np.array([p[0] for p in pairs], dtype=float)
    sign_zs_arr = np.array([p[1] for p in pairs], dtype=float)
    weights_arr = np.array([p[2] for p in pairs], dtype=float)
    return z_offsets, sign_zs_arr, weights_arr, n_terms_used


def _two_layer_potential_kernel(
    field_points: np.ndarray,
    source_points: np.ndarray,
    currents: np.ndarray,
    soil: TwoLayerSoil,
    max_terms: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    """Vectorised 2-layer image-charge sum at field points.

    Parameters
    ----------
    field_points
        Evaluation points, shape ``(M, 3)``.
    source_points
        Source midpoints, shape ``(N, 3)``.
    currents
        Real-valued current vector ``(N,)``. Complex currents must be
        split externally and the kernel called twice.
    soil
        :class:`TwoLayerSoil`. ``soil.h_1`` is the upper-layer depth.
    max_terms, tol
        Series truncation parameters.

    Returns
    -------
    phi : np.ndarray, shape (M,)
        Evaluation result.
    n_terms_used : int
        Last evaluated $n$ (used by the caller to flag
        convergence issues).
    """
    K = soil.reflection_coefficient
    h_1 = soil.h_1
    rho_1 = soil.rho_1

    z_offsets, sign_zs_arr, weights, n_used = _two_layer_image_offsets(
        K, h_1, max_terms, tol
    )

    # 2-D distance in (x, y) — shared across all images.
    diff_xy = field_points[:, None, 0:2] - source_points[None, :, 0:2]
    delta_sq = np.einsum("mnk,mnk->mn", diff_xy, diff_xy)  # (M, N)

    z_field = field_points[:, 2:3]    # (M, 1)
    z_src = source_points[None, :, 2]  # (1, N)

    phi = np.zeros(field_points.shape[0], dtype=float)
    for k in range(z_offsets.size):
        z_img = z_offsets[k] + sign_zs_arr[k] * z_src        # (1, N)
        r_sq = delta_sq + (z_field - z_img) ** 2              # (M, N)
        r = np.sqrt(r_sq)
        np.maximum(r, _MIN_DISTANCE, out=r)
        phi += weights[k] * ((1.0 / r) @ currents)

    phi *= rho_1 / (4.0 * np.pi)
    return phi, n_used


def _two_layer_self_kernel_factory(
    soil: TwoLayerSoil, max_terms: int, tol: float,
    *, allow_cross_layer: bool = False,
):
    """Build a self-action closure for the 2-layer case.

    The $n = 0$ direct contribution of a segment onto itself is
    handled by the analytical line-self correction (same as the
    homogeneous backend, but with $\\rho_1$ as prefactor). The
    image terms with $n \\ge 1$ are at least $2 h_1$ away
    from the segments, so the point-source approximation is good
    enough.

    Parameters
    ----------
    soil, max_terms, tol
        Two-layer soil model and Tagg/Sunde series-truncation
        parameters.
    allow_cross_layer
        When ``True`` (ADR-0007) the kernel transparently dispatches
        to the rigorous Sommerfeld-quadrature path in
        :mod:`groundfield.coupling.layered_green` for any pair of
        source/observer points that crosses the layer interface.
        Pairs that stay in the upper layer continue to use the fast
        Tagg/Sunde image series. When ``False`` (default, kept for
        bit-exact regression) the kernel uses only the image series
        and the caller must enforce the precondition
        ``z_max < h_1``.
    """
    rho_1 = soil.rho_1
    K = soil.reflection_coefficient
    h_1 = soil.h_1
    abs_K = abs(K)

    def _self_kernel(seg_points, seg_lengths, wire_radii, currents):
        # ADR-0007: when the geometry crosses the layer interface we
        # dispatch the Sommerfeld kernel from coupling.layered_green
        # for every source/observer pair. Pure-upper-layer geometries
        # keep the historic fast image-series path bit-exact.
        z_max = float(seg_points[:, 2].max())
        if allow_cross_layer and z_max >= h_1:
            return _layered_green_kernel(
                seg_points, seg_lengths, wire_radii, currents,
                soil=soil,
            )

        # n = 0: identical to the homogeneous self-kernel with rho = rho_1
        phi = _self_corrected_kernel(
            seg_points, seg_lengths, wire_radii, currents, rho_1
        )

        # n >= 1: image terms are far from the segments — point-source
        # approximation suffices.
        diff_xy = seg_points[:, None, 0:2] - seg_points[None, :, 0:2]
        delta_sq = np.einsum("mnk,mnk->mn", diff_xy, diff_xy)
        z_field = seg_points[:, 2:3]
        z_src = seg_points[None, :, 2]

        # ``extra`` must match the shape of ``phi``, which depends on
        # whether ``currents`` is a 1-D vector (N,) or a 2-D matrix
        # (N, M) — Engine B's matrix assembly passes an identity
        # matrix to obtain the full reaction matrix in one call.
        extra = np.zeros_like(phi)
        for n in range(1, max_terms + 1):
            K_n = K ** n
            for sign_n in (+1, -1):
                for sign_zs in (+1, -1):
                    z_img = sign_n * 2.0 * n * h_1 + sign_zs * z_src
                    r = np.sqrt(delta_sq + (z_field - z_img) ** 2)
                    np.maximum(r, _MIN_DISTANCE, out=r)
                    extra += K_n * ((1.0 / r) @ currents)
            if abs_K ** n < tol:
                break

        extra *= rho_1 / (4.0 * np.pi)
        return phi + extra

    return _self_kernel


def _layered_green_kernel(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    currents,
    *,
    soil: TwoLayerSoil,
):
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
        seg_points[:, 2] < h_1, rho_1, rho_2,
    )

    # Step 1: build phi_hom matrix that uses rho_at_source(j) for
    # each column (source segment). This generalises
    # _self_corrected_kernel: the diagonal still uses the line-
    # self formula with rho_local(i), and off-diagonal entries use
    # the point-source 1/r + 1/r_image formula with rho_local(j).
    phi_hom_matrix = _build_phi_hom_per_source_rho(
        seg_points, seg_lengths, wire_radii, rho_per_segment,
    )
    phi_hom = phi_hom_matrix @ currents

    # Step 2: layered correction. The baseline rho per source is
    # rho_at_source(j). For the diagonal (i = j) we still need to
    # handle the residual line-self peak by 3×3 Gauss-Legendre
    # averaging along the segment axis — this is small but
    # important for short segments in highly contrasting soils.
    delta_Z = np.zeros((n, n), dtype=float)
    gl3_nodes, gl3_weights = np.polynomial.legendre.leggauss(3)
    gl3_nodes = 0.5 * (gl3_nodes + 1.0)
    gl3_weights = 0.5 * gl3_weights
    for i in range(n):
        x_i, y_i, z_i = seg_points[i]
        L_i = float(seg_lengths[i])
        for j in range(n):
            x_j, y_j, z_j = seg_points[j]
            s_horiz = float(np.hypot(x_i - x_j, y_i - y_j))
            rho_baseline_j = float(rho_per_segment[j])
            if i == j:
                z_lo = float(z_i) - 0.5 * L_i
                delta_G = 0.0
                for a, w_a in zip(gl3_nodes, gl3_weights):
                    z_a = z_lo + a * L_i
                    for b, w_b in zip(gl3_nodes, gl3_weights):
                        z_b = z_lo + b * L_i
                        s_eval = max(s_horiz, float(wire_radii[i]))
                        delta_G += w_a * w_b * two_layer_layered_correction_real_space(
                            s=s_eval, z=z_a, z_s=z_b,
                            rho_1=rho_1, rho_2=rho_2, h_1=h_1,
                            rho_baseline=rho_baseline_j,
                        )
                delta_Z[i, i] = delta_G / (2.0 * np.pi)
            else:
                delta_G = two_layer_layered_correction_real_space(
                    s=s_horiz, z=float(z_i), z_s=float(z_j),
                    rho_1=rho_1, rho_2=rho_2, h_1=h_1,
                    rho_baseline=rho_baseline_j,
                )
                delta_Z[i, j] = delta_G / (2.0 * np.pi)
    return phi_hom + delta_Z @ currents


def _build_phi_hom_per_source_rho(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    rho_per_segment: np.ndarray,
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
    """
    n = seg_points.shape[0]
    image_points = seg_points.copy()
    image_points[:, 2] = -image_points[:, 2]

    diff_real = seg_points[:, None, :] - seg_points[None, :, :]
    diff_image = seg_points[:, None, :] - image_points[None, :, :]
    r_real = np.linalg.norm(diff_real, axis=2)
    r_image = np.linalg.norm(diff_image, axis=2)
    np.maximum(r_real, _MIN_DISTANCE, out=r_real)
    np.maximum(r_image, _MIN_DISTANCE, out=r_image)

    # Off-diagonal kernel: 1/r + 1/r_image.
    kernel = (1.0 / r_real) + (1.0 / r_image)

    # Diagonal: line-self + image-point.
    diag_direct = 2.0 * np.log(seg_lengths / wire_radii) / seg_lengths
    z_mid = seg_points[:, 2]
    diag_image = 1.0 / np.maximum(2.0 * np.abs(z_mid), _MIN_DISTANCE)
    np.fill_diagonal(kernel, diag_direct + diag_image)

    # Apply rho_per_segment per column (source segment).
    # M[i, j] = rho_per_segment[j] / (4π) · kernel[i, j]
    M = (rho_per_segment[None, :] / (4.0 * np.pi)) * kernel
    return M


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_image_2layer(
    world: "World",
    engine: "Engine",
    *,
    max_terms: int = 100,
    tol: float = 1e-6,
) -> FieldResult:
    """Image-charge solver for 2-layer soil (Tagg / Sunde).

    Parameters
    ----------
    world
        World whose ``soil`` is a :class:`TwoLayerSoil`. All electrodes
        must lie inside the upper layer.
    engine
        Engine configuration. ``engine.segment_length`` controls the
        discretisation as in the homogeneous backend.
    max_terms
        Maximum number of series terms.
    tol
        Series truncation: stop as soon as $|K|^n < \\text{tol}$.

    Returns
    -------
    FieldResult
        Result object. :attr:`FieldResult.metadata` exposes diagnostic
        fields (``backend``, ``K``, ``n_terms_used``, ``converged``).
    """
    if not isinstance(world.soil, TwoLayerSoil):
        raise TypeError(
            "Backend 'image_2layer' requires TwoLayerSoil. "
            f"Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    soil: TwoLayerSoil = world.soil
    h_1 = soil.h_1
    K = soil.reflection_coefficient
    ds = engine.segment_length

    _log.info(
        "image_2layer: rho_1=%.1f, rho_2=%.1f, h_1=%.2f, K=%.4f",
        soil.rho_1, soil.rho_2, h_1, K,
    )

    # 1) Discretisation — identical to the homogeneous backend
    all_segments: list[_Segment] = []
    elec_to_segidx: dict[str, list[int]] = {}
    for e in world.electrodes:
        segs = _discretize_electrode(e, ds)
        elec_to_segidx[e.name] = list(range(len(all_segments),
                                            len(all_segments) + len(segs)))
        all_segments.extend(segs)

    # 2) Per-electrode input currents from sources
    elec_input_current: dict[str, complex] = {
        e.name: 0j for e in world.electrodes
    }
    for src in world.sources:
        if src.kind != "current":
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in elec_input_current:
            elec_input_current[src.attached_to] += i_complex

    # 3) Cluster building (ideal conductors only) and finite-impedance
    #    branch list (passed into the nodal-analysis solver).
    cluster_id = _build_clusters(world.electrodes, world.conductors)
    finite_branches = _build_finite_branches(world.conductors, cluster_id)

    # 3b) Distributed-conductor topology (ADR-0003) + ADR-0004
    #     inductive coupling assembly.
    cond_segs, distributed_branches_objs, interior_nodes = _build_distributed_topology(
        world.conductors, cluster_id
    )
    pseudo_owners: list[str] = []
    for s in cond_segs:
        pn = s.electrode_name
        elec_to_segidx[pn] = [len(all_segments)]
        all_segments.append(s)
        cluster_id[pn] = pn
        pseudo_owners.append(pn)
    for n_ in interior_nodes:
        if n_ not in cluster_id:
            cluster_id[n_] = n_
            pseudo_owners.append(n_)
            elec_to_segidx[n_] = []
    n_lumped_branches = len(finite_branches)
    distributed_branch_tuples = [
        (db.node_a, db.node_b, db.R) for db in distributed_branches_objs
    ]
    finite_branches = list(finite_branches) + distributed_branch_tuples
    earth_inductive_model = getattr(
        engine, "earth_inductive_model", "perfect_mirror"
    )
    sigma_earth_for_carson: float | None = None
    layered_earth_for_sommerfeld: object = None
    if earth_inductive_model == "carson_series":
        from groundfield.coupling import resolve_earth_conductivity

        sigma_earth_for_carson = resolve_earth_conductivity(world.soil)
    elif earth_inductive_model == "sommerfeld":
        from groundfield.coupling import resolve_earth_layers

        layered_earth_for_sommerfeld = resolve_earth_layers(world.soil)
    inductance_matrix_full, has_inductance, carson_builder = _assemble_inductance_matrix(
        distributed_branches_objs,
        n_lumped_branches=n_lumped_branches,
        n_total_branches=len(finite_branches),
        earth_model=earth_inductive_model,
        sigma_earth=sigma_earth_for_carson,
        layered_earth=layered_earth_for_sommerfeld,
    )

    seg_points = np.array([s.midpoint for s in all_segments])
    seg_lengths = np.array([s.length for s in all_segments])
    wire_radii = np.array([s.wire_radius for s in all_segments])

    # ADR-0007: detect cross-layer geometry. When all segments live
    # inside the upper layer, the historic Tagg/Sunde image series
    # is used (bit-exact regression). When any segment is at or
    # below the interface, the kernel automatically dispatches to
    # the rigorous Sommerfeld path.
    z_max = seg_points[:, 2].max()
    cross_layer = bool(z_max >= h_1)
    if cross_layer:
        _log.info(
            "image_2layer: cross-layer geometry detected "
            "(z_max = %.3f m, h_1 = %.3f m). Using ADR-0007 "
            "Sommerfeld fallback for all source/observer pairs.",
            z_max, h_1,
        )
    # Tag every segment with its layer index (0 = upper, 1 = lower).
    for s in all_segments:
        s.layer_index = 0 if s.midpoint[2] < h_1 else 1

    # 4) Self-kernel closure and frequency loop
    self_kernel = _two_layer_self_kernel_factory(
        soil, max_terms, tol, allow_cross_layer=cross_layer,
    )
    n_segments = len(all_segments)
    n_freq = len(engine.frequencies)
    omegas = [2.0 * np.pi * float(f) for f in engine.frequencies]
    real_electrode_names = {e.name for e in world.electrodes}

    def _solve_at(omega: float) -> tuple[
        dict[str, complex], np.ndarray, np.ndarray
    ]:
        carson_dz = (
            carson_builder(omega) if (has_inductance and carson_builder is not None)
            else None
        )
        elec_total = _solve_cluster_currents(
            electrodes=world.electrodes,
            elec_input_current=elec_input_current,
            cluster_id=cluster_id,
            seg_points=seg_points,
            seg_lengths=seg_lengths,
            wire_radii=wire_radii,
            elec_to_segidx=elec_to_segidx,
            self_kernel=self_kernel,
            finite_branches=finite_branches,
            pseudo_owners=pseudo_owners,
            omega=omega if has_inductance else 0.0,
            inductance_matrix=inductance_matrix_full if has_inductance else None,
            carson_correction=carson_dz,
        )
        sc = np.zeros(n_segments, dtype=complex)
        for ename, idxs in elec_to_segidx.items():
            if not idxs:
                continue
            I_total = elec_total.get(ename, 0j)
            if I_total == 0j:
                continue
            L_total = seg_lengths[idxs].sum()
            sc[idxs] = I_total * seg_lengths[idxs] / L_total
        ph = np.zeros(n_segments, dtype=complex)
        if sc.any():
            phi_re = self_kernel(
                seg_points, seg_lengths, wire_radii, sc.real,
            )
            phi_im = self_kernel(
                seg_points, seg_lengths, wire_radii, sc.imag,
            )
            ph = phi_re + 1j * phi_im
        return elec_total, sc, ph

    elec_per_freq: list[dict[str, complex]] = []
    sc_per_freq: list[np.ndarray] = []
    phi_per_freq: list[np.ndarray] = []
    if has_inductance:
        for omega in omegas:
            et, sc, ph = _solve_at(omega)
            elec_per_freq.append(et)
            sc_per_freq.append(sc)
            phi_per_freq.append(ph)
    else:
        et, sc, ph = _solve_at(0.0)
        elec_per_freq = [et] * n_freq
        sc_per_freq = [sc] * n_freq
        phi_per_freq = [ph] * n_freq

    n_terms_used_self = 0
    if any(np.any(sc) for sc in sc_per_freq):
        _, _, _, n_terms_used_self = _two_layer_image_offsets(
            K, h_1, max_terms, tol
        )

    electrode_potentials: dict[str, list[complex]] = {}
    electrode_currents: dict[str, list[complex]] = {}
    conductor_currents: dict[str, list[complex]] = {}
    conductor_potentials: dict[str, list[complex]] = {}
    for ename, idxs in elec_to_segidx.items():
        if not idxs:
            continue
        u_list = [
            complex(np.mean(phi_per_freq[k][idxs])) for k in range(n_freq)
        ]
        i_list = [elec_per_freq[k][ename] for k in range(n_freq)]
        if ename in real_electrode_names:
            electrode_potentials[ename] = u_list
            electrode_currents[ename] = i_list
        else:
            conductor_potentials[ename] = u_list
            conductor_currents[ename] = i_list

    # 7) Point-source list for post-processing
    point_sources = [
        PointSource(
            position=tuple(seg_points[i].tolist()),
            current=[complex(sc_per_freq[k][i]) for k in range(n_freq)],
            electrode_name=all_segments[i].electrode_name,
            length=float(seg_lengths[i]),
        )
        for i in range(n_segments)
    ]

    cluster_members: dict[str, list[str]] = {}
    for ename in real_electrode_names:
        cluster_members[ename] = sorted(
            n for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_electrode_names
        )

    converged = (abs(K) ** n_terms_used_self < tol) if n_terms_used_self else True
    if not converged:
        _log.warning(
            "image_2layer: max_terms=%d reached, |K|^n = %.2e > tol=%.2e. "
            "Result may be inaccurate.",
            max_terms, abs(K) ** n_terms_used_self, tol,
        )

    metadata = {
        "world_name": world.name,
        "n_segments": n_segments,
        "segment_length": ds,
        "K": float(K),
        "rho_1": float(soil.rho_1),
        "rho_2": float(soil.rho_2),
        "h_1": float(h_1),
        "n_terms_used": n_terms_used_self,
        "converged": converged,
        "stub": False,
        "earth_inductive_model": earth_inductive_model,
    }
    if has_inductance and sigma_earth_for_carson is not None:
        from groundfield.coupling.carson import skin_depth

        metadata["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma_earth_for_carson)
            for f in engine.frequencies
        }
    elif has_inductance:
        from groundfield.coupling.carson import skin_depth

        sigma_ref = 1.0 / float(soil.rho_1)
        metadata["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma_ref)
            for f in engine.frequencies
        }
    if conductor_currents:
        metadata["conductor_node_currents"] = conductor_currents
        metadata["conductor_node_potentials"] = conductor_potentials

    return FieldResult(
        backend="image_2layer",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=float(soil.rho_1),
        soil=soil,
        clusters=cluster_members,
        metadata=metadata,
    )
