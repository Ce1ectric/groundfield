"""Full mutual cluster-grounding-impedance matrix in one assembly.

This module adds a **single additive entry point**
:func:`solve_mutual_matrix` that computes the full ``nG × nG`` mutual
grounding-impedance matrix ``Z_G`` of a set of grounding clusters
("gnodes") in **one** matrix assembly, instead of the historic pattern
of one independent ``Engine.solve`` per excitation.

Motivation
----------
The catalogue step in ``ap1_pipeline.py`` builds ``Z_G`` via the loop
(see ``_compute_ZG``)::

    for j in range(nG):
        world = create_world(soil); build all nG anchors
        create_source(world, attached_to=anchors[j], magnitude=1.0)
        res = engine.solve(world)
        Z[j, j] = res.cluster_impedance(anchors[j])[0]          # diagonal
        for i != j:
            Z[i, j] = res.potential([[x_i, y_i, probe_depth]])[0]   # off-diag
    Z = 0.5 * (Z + Z.T)

Because the **world and the soil are identical across all ``j``** and
only the *source* moves, this re-assembles the ``N × N`` segment
reaction matrix ``nG`` times and re-evaluates the ``nG²`` Tagg/Sunde
field-series for the off-diagonal entries. :func:`solve_mutual_matrix`
removes both redundancies:

1. The segment reaction matrix is assembled **once** (via the existing
   ``self_kernel`` closures, including the ADR-0012 V2 concrete shell).
2. The off-diagonal probe-point kernel is built **once** and reused for
   every excitation column through a single ``Phi = K @ T`` matmul.

Bit-exactness
-------------
The implementation reuses the **existing** kernel/solve primitives
(:func:`~groundfield.solver.image._solve_cluster_currents`'s assembly
and solve logic, :func:`~groundfield.solver.image._self_corrected_kernel`,
the 2-layer self-kernel factory and the
:meth:`FieldResult._potential_two_layer` / ``_potential_homogeneous``
series) unchanged, with the **same operation order** wherever possible.
The only structural change is that linear operators are applied to a
matrix of stacked excitation columns rather than to one vector at a
time. See ``solve_mutual_matrix`` for the column-by-column reduction of
the diagonal and the off-diagonal blocks.

The companion contract test (``/tmp/test_mutual_matrix.py`` during
development) checks ``max|Z_new − Z_old| < 1e-9`` against the historic
per-excitation method for homogeneous-limit, mid-contrast and
cross-layer two-layer soils at 50 Hz and 1000 Hz.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.solver.image import (
    _MIN_DISTANCE,
    _assemble_inductance_matrix,
    _build_clusters,
    _build_distributed_topology,
    _build_finite_branches,
    _discretize_electrode,
    _Segment,
    _self_corrected_kernel,
    _solve_cluster_currents,
)
from groundfield.solver.image_2layer import _two_layer_self_kernel_factory

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_mutual_matrix", "solve_mutual_field"]


# ---------------------------------------------------------------------
# Probe-point field kernels (matrix-valued, reused over all columns)
# ---------------------------------------------------------------------


def _probe_kernel_homogeneous(
    probe_points: np.ndarray,   # (M, 3)
    source_points: np.ndarray,  # (N, 3)
    rho: float,
    min_distance: float,
) -> np.ndarray:
    """Real ``(M, N)`` homogeneous image-charge probe kernel.

    ``Phi[m] = sum_n K[m, n] · I_n`` reproduces
    :meth:`FieldResult._potential_homogeneous` exactly: the same
    ``1/r + 1/r_image`` form, the same ``rho/(4π)`` prefactor and the
    same ``min_distance`` clamp. Building the matrix once and applying
    it to every excitation column is bit-identical to evaluating the
    potential per column, because there is only a single (non-series)
    term — no intermediate summation order to preserve.
    """
    image_sources = source_points.copy()
    image_sources[:, 2] = -image_sources[:, 2]
    diff_real = probe_points[:, None, :] - source_points[None, :, :]
    diff_image = probe_points[:, None, :] - image_sources[None, :, :]
    r_real = np.linalg.norm(diff_real, axis=2)
    r_image = np.linalg.norm(diff_image, axis=2)
    np.maximum(r_real, min_distance, out=r_real)
    np.maximum(r_image, min_distance, out=r_image)
    kernel = (1.0 / r_real) + (1.0 / r_image)
    return (rho / (4.0 * np.pi)) * kernel


def _probe_potential_two_layer(
    probe_points: np.ndarray,    # (M, 3)
    source_points: np.ndarray,   # (N, 3)
    currents: np.ndarray,        # (N, J) complex or real
    soil: TwoLayerSoil,
    min_distance: float,
    max_terms: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Matrix form of :meth:`FieldResult._potential_two_layer`.

    Evaluates the Tagg/Sunde image series at every probe point for a
    **stack** of current columns simultaneously. The series geometry
    (``delta_sq``, the per-image ``1/r`` block) is built once and
    applied to all columns through a single ``(1/r) @ currents`` matmul
    per image term — preserving the exact term-by-term accumulation
    order (``phi += K_n · acc``) of the scalar reference, so that
    column ``j`` of the returned array equals
    ``FieldResult._potential_two_layer(probe_points, source_points,
    currents[:, j], soil, min_distance, max_terms, tol)`` up to BLAS
    round-off.

    Parameters
    ----------
    currents
        ``(N, J)`` array (J excitation columns). May be real or
        complex; the real and imaginary parts are run through the
        identical real-valued series, exactly like the scalar method.

    Returns
    -------
    phi : np.ndarray, shape (M, J)
        Complex potential per probe point and excitation column.
    """
    K = soil.reflection_coefficient
    h_1 = soil.h_1
    rho_1 = soil.rho_1

    M = probe_points.shape[0]
    currents = np.atleast_2d(currents)
    if currents.shape[0] != source_points.shape[0]:
        # accept a single column passed as shape (N,) -> (1, N)
        currents = currents.T
    J = currents.shape[1]

    # ------------------------------------------------------------------
    # Layer dispatch (audit 2026-07-08, WP-D1). The Tagg/Sunde image
    # series below is derived for source AND observer in the upper
    # layer; applying it to layer-2 points is wrong (measured +7 % at
    # z = 7 m, +25 % at z = 12 m for K = +0.818, h_1 = 5 m). Pure
    # upper-layer evaluations keep the historic fast path bit-exact;
    # any pair involving layer 2 goes through the rigorous spectral
    # kernel of coupling.layered_green.
    # ------------------------------------------------------------------
    probe_l1 = probe_points[:, 2] < h_1
    src_l1 = source_points[:, 2] < h_1
    if not (probe_l1.all() and src_l1.all()):
        from groundfield.coupling.layered_green import two_layer_probe_matrix

        rho_2 = soil.rho_2
        phi = np.zeros((M, J), dtype=complex)
        # uu block — fast image series on the sub-arrays.
        if probe_l1.any() and src_l1.any():
            phi[probe_l1] += _probe_potential_two_layer(
                probe_points[probe_l1], source_points[src_l1],
                currents[src_l1], soil, min_distance, max_terms, tol,
            )
        # Remaining blocks via the spectral kernel: layer-2 sources to
        # all probes, layer-1 sources to layer-2 probes.
        if (~src_l1).any():
            G = two_layer_probe_matrix(
                probe_points, source_points[~src_l1],
                rho_1=rho_1, rho_2=rho_2, h_1=h_1,
                min_distance=min_distance,
            )
            phi += G @ currents[~src_l1]
        if (~probe_l1).any() and src_l1.any():
            G = two_layer_probe_matrix(
                probe_points[~probe_l1], source_points[src_l1],
                rho_1=rho_1, rho_2=rho_2, h_1=h_1,
                min_distance=min_distance,
            )
            phi[~probe_l1] += G @ currents[src_l1]
        return phi

    diff_xy = probe_points[:, None, 0:2] - source_points[None, :, 0:2]
    delta_sq = np.einsum("mnk,mnk->mn", diff_xy, diff_xy)  # (M, N)
    z_field = probe_points[:, 2:3]      # (M, 1)
    z_src = source_points[None, :, 2]   # (1, N)

    def _series_for(real_currents: np.ndarray) -> np.ndarray:
        """Core series for a real-valued ``(N, J)`` current stack."""
        phi = np.zeros((M, J), dtype=float)
        # n = 0 (two images, weight 1) — mirrors the scalar method's
        # ``for sign_zs in (+1, -1): phi += (1/r) @ real_currents``.
        for sign_zs in (+1, -1):
            z_img = sign_zs * z_src
            r = np.sqrt(delta_sq + (z_field - z_img) ** 2)
            np.maximum(r, min_distance, out=r)
            phi += (1.0 / r) @ real_currents
        # n = 1, 2, ... — accumulate per term (``phi += K_n · acc``).
        abs_K = abs(K)
        for n in range(1, max_terms + 1):
            K_n = K ** n
            acc = np.zeros((M, J), dtype=float)
            for sign_n in (+1, -1):
                for sign_zs in (+1, -1):
                    z_img = sign_n * 2.0 * n * h_1 + sign_zs * z_src
                    r = np.sqrt(delta_sq + (z_field - z_img) ** 2)
                    np.maximum(r, min_distance, out=r)
                    acc += (1.0 / r) @ real_currents
            phi += K_n * acc
            # Geometric tail bound |K|^(n+1)/(1-|K|) — consistent
            # with the solver-side criterion in image_2layer.
            if abs_K < 1.0 and abs_K ** (n + 1) / (1.0 - abs_K) < tol:
                break
        return phi

    phi_re = _series_for(currents.real)
    phi_im = _series_for(currents.imag)
    return (rho_1 / (4.0 * np.pi)) * (phi_re + 1j * phi_im)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_mutual_matrix(
    world: "World",
    engine: "Engine",
    anchors: list[str],
    probe_points: np.ndarray,
    *,
    frequency_index: int = 0,
    symmetrize: bool = True,
    matrix_min_distance: float = 1e-3,
    matrix_max_terms: int = 100,
    matrix_tol: float = 1e-6,
) -> np.ndarray:
    """Full mutual grounding-impedance matrix ``Z_G`` in one assembly.

    Reproduces the historic per-excitation construction (one
    ``Engine.solve`` per gnode; diagonal from
    :meth:`FieldResult.cluster_impedance`, off-diagonal from
    :meth:`FieldResult.potential` at the probe points) **bit-for-bit**,
    but assembles the segment reaction matrix and the probe-point field
    kernel only once.

    ``world`` must already carry **all** ``nG`` grounding clusters
    (every anchor in ``anchors`` must be a real electrode in
    ``world``). It must **not** carry any current source — the
    excitations are applied internally, one cluster at a time, exactly
    as the historic loop did. Finite-impedance / distributed conductors
    *between distinct clusters* are out of scope for the catalogue use
    case (each gnode is an isolated galvanic cluster) and are handled
    via the standard :func:`_solve_cluster_currents` machinery only in
    so far as they remain confined to a single excited cluster.

    Parameters
    ----------
    world
        Assembled world holding all ``nG`` grounding clusters and the
        soil model. Sources, if any, are ignored.
    engine
        Engine configuration. ``segment_length`` controls the
        discretisation; ``frequencies`` selects the solved frequency
        via ``frequency_index``; ``image_max_terms`` /
        ``image_series_tol`` feed the 2-layer **self**-kernel (the
        reaction-matrix assembly), exactly as ``Engine.solve`` would.
    anchors
        Length-``nG`` list of electrode names, one per gnode. Anchor
        ``j`` receives the unit current of excitation ``j`` and its
        cluster supplies the diagonal entry ``Z[j, j]``.
    probe_points
        ``(nG, 3)`` array of off-diagonal probe coordinates: row ``i``
        is the ``(x, y, depth)`` point at which the potential of every
        other excitation is sampled to fill column ``i`` of the
        off-diagonal. Mirrors ``res.potential([[x_i, y_i, depth]])``.
    frequency_index
        Index into ``engine.frequencies``. Default 0.
    symmetrize
        If ``True`` (default) return ``0.5 · (Z + Zᵀ)`` — the historic
        post-processing.
    matrix_min_distance, matrix_max_terms, matrix_tol
        Truncation / clamp parameters of the **off-diagonal** field
        evaluation. The defaults ``1e-3 / 100 / 1e-6`` reproduce the
        hard-coded defaults of :meth:`FieldResult.potential` /
        ``_potential_two_layer`` exactly. (The reaction-matrix
        assembly uses ``engine.image_max_terms`` / ``image_series_tol``
        independently, just like the live solver.)

    Returns
    -------
    Z : np.ndarray, shape (nG, nG), complex
        The mutual grounding-impedance matrix.
    """
    from groundfield.solver.engine import Engine  # noqa: F401  (typing only)

    if world.soil is None:
        raise ValueError("World has no soil model.")
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    soil = world.soil
    nG = len(anchors)
    probe_points = np.asarray(probe_points, dtype=float)
    if probe_points.shape != (nG, 3):
        raise ValueError(
            f"probe_points must have shape ({nG}, 3), got "
            f"{probe_points.shape}."
        )

    # ------------------------------------------------------------------
    # 1) Discretisation + topology — identical to the live backends.
    # ------------------------------------------------------------------
    ds = engine.segment_length
    all_segments: list[_Segment] = []
    elec_to_segidx: dict[str, list[int]] = {}
    interfaces = (
        (world.soil.h_1,) if isinstance(world.soil, TwoLayerSoil) else None
    )
    for e in world.electrodes:
        segs = _discretize_electrode(e, ds, layer_interfaces=interfaces)
        elec_to_segidx[e.name] = list(
            range(len(all_segments), len(all_segments) + len(segs))
        )
        all_segments.extend(segs)

    cluster_id = _build_clusters(world.electrodes, world.conductors)
    finite_branches = _build_finite_branches(world.conductors, cluster_id)

    cond_segs, distributed_branches_objs, interior_nodes = (
        _build_distributed_topology(world.conductors, cluster_id)
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

    # Inductive coupling assembly (kept for parity; the catalogue probe
    # case has no distributed conductors, so ``has_inductance`` is False
    # and the system stays real/frequency-independent — matching the
    # historic ``image`` / ``image_2layer`` no-inductance fast path).
    earth_inductive_model = getattr(
        engine, "earth_inductive_model", "perfect_mirror"
    )
    sigma_earth_for_carson: float | None = None
    layered_earth_for_sommerfeld: object = None
    if earth_inductive_model == "carson_series":
        from groundfield.coupling import resolve_earth_conductivity

        sigma_earth_for_carson = resolve_earth_conductivity(soil)
    elif earth_inductive_model == "sommerfeld":
        from groundfield.coupling import resolve_earth_layers

        layered_earth_for_sommerfeld = resolve_earth_layers(soil)
    inductance_matrix_full, has_inductance, carson_builder = (
        _assemble_inductance_matrix(
            distributed_branches_objs,
            n_lumped_branches=n_lumped_branches,
            n_total_branches=len(finite_branches),
            earth_model=earth_inductive_model,
            sigma_earth=sigma_earth_for_carson,
            layered_earth=layered_earth_for_sommerfeld,
        )
    )

    n_segments = len(all_segments)
    seg_points = np.array([s.midpoint for s in all_segments])   # (N, 3)
    seg_lengths = np.array([s.length for s in all_segments])     # (N,)
    wire_radii = np.array([s.wire_radius for s in all_segments])  # (N,)
    seg_shell_coeffs = np.array(
        [s.concrete_shell_coefficient_ohm_m for s in all_segments],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 2) Soil-specific self-kernel closure — same dispatch as
    #    ``Engine.solve`` (image -> image_2layer for TwoLayerSoil).
    # ------------------------------------------------------------------
    rho_homog: float | None = None
    soil_two_layer: TwoLayerSoil | None = None
    if isinstance(soil, TwoLayerSoil):
        soil_two_layer = soil
    elif isinstance(soil, MultiLayerSoil):
        n_layers = len(soil.layers)
        if n_layers == 1:
            rho_homog = float(soil.layers[0].resistivity)
        elif n_layers == 2:
            soil_two_layer = TwoLayerSoil(
                rho_1=float(soil.layers[0].resistivity),
                rho_2=float(soil.layers[1].resistivity),
                h_1=float(soil.layers[0].thickness),
            )
        else:
            raise NotImplementedError(
                "solve_mutual_matrix supports homogeneous and 2-layer "
                f"soils; got MultiLayerSoil with n={n_layers} layers."
            )
    elif isinstance(soil, HomogeneousSoil):
        rho_homog = float(soil.resistivity)
    else:
        raise TypeError(f"Unsupported soil type {type(soil).__name__}.")

    if soil_two_layer is not None:
        # 2-layer reaction-matrix self-kernel: feed the engine's
        # image-series truncation, exactly like ``solve_image_2layer``.
        self_max_terms = int(getattr(engine, "image_max_terms", 100))
        self_tol = float(getattr(engine, "image_series_tol", 1e-6))
        z_max = float(seg_points[:, 2].max())
        cross_layer = bool(z_max >= soil_two_layer.h_1)
        self_kernel = _two_layer_self_kernel_factory(
            soil_two_layer, self_max_terms, self_tol,
            allow_cross_layer=cross_layer,
        )
    else:
        rho_for_kernel = float(rho_homog)

        def self_kernel(seg_pts, seg_lens, wr, currents):  # noqa: ANN001
            return _self_corrected_kernel(
                seg_pts, seg_lens, wr, currents, rho_for_kernel
            )

    # ------------------------------------------------------------------
    # 3) Reaction matrix ONCE.  ``Z_seg_full = self_kernel(eye)`` is the
    #    N×N segment reaction matrix used by ``mom._build_Z_*``; column
    #    k is ``self_kernel(e_k)``. Reusing it for every excitation makes
    #    the diagonal ``ph`` evaluation and the cluster-current assembly
    #    a sequence of matvecs instead of nG full kernel calls.
    # ------------------------------------------------------------------
    eye = np.eye(n_segments)
    Z_seg_full = self_kernel(seg_points, seg_lengths, wire_radii, eye)  # (N, N)

    # ADR-0012 V2 shell augmentation (added to the *diagonal* of the
    # reaction matrix exactly as in ``_solve_cluster_currents`` and the
    # post-solve ``ph`` step).
    shell_diag: np.ndarray | None = None
    if np.any(seg_shell_coeffs > 0.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            shell_diag = np.where(
                seg_lengths > 0.0,
                seg_shell_coeffs / seg_lengths,
                0.0,
            )

    # A linear ``self_kernel``-equivalent backed by the precomputed
    # matrix: ``Z_seg_full @ currents`` reproduces ``self_kernel(...)``
    # for the unit-current assembly inside ``_solve_cluster_currents``.
    # The shell augmentation is applied on top (the assembly inside
    # ``_solve_cluster_currents`` does the same: bulk kernel then
    # ``phi += shell_diag · currents``), so we deliberately pass a
    # *bulk-only* kernel here and let ``shell_coefficients`` re-add the
    # shell — keeping the exact same operation order as the live solver.
    def _cached_bulk_self_kernel(seg_pts, seg_lens, wr, currents):  # noqa: ANN001
        return Z_seg_full @ currents

    omega = (
        2.0 * np.pi * float(engine.frequencies[frequency_index])
        if has_inductance
        else 0.0
    )
    carson_dz = (
        carson_builder(omega)
        if (has_inductance and carson_builder is not None)
        else None
    )

    # ------------------------------------------------------------------
    # 4) Per-excitation leakage distribution.  Each gnode is an isolated
    #    galvanic cluster, so injecting 1 A into cluster ``j`` activates
    #    only that cluster; the resulting per-segment leakage column
    #    ``sc_j`` is non-zero on cluster-``j`` segments only — exactly
    #    what the historic single-source solve produced.
    # ------------------------------------------------------------------
    real_electrode_names = {e.name for e in world.electrodes}
    # cluster members per real electrode (== FieldResult.clusters), used
    # for the diagonal ``cluster_impedance`` average and current sum.
    cluster_members: dict[str, list[str]] = {}
    for ename in real_electrode_names:
        cluster_members[ename] = sorted(
            n
            for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_electrode_names
        )

    T = np.zeros((n_segments, nG), dtype=complex)  # column j = sc_j
    Z = np.zeros((nG, nG), dtype=complex)

    for j, anchor_j in enumerate(anchors):
        if anchor_j not in elec_to_segidx:
            raise KeyError(f"Anchor '{anchor_j}' is not an electrode in world.")
        elec_input_current = {e.name: 0j for e in world.electrodes}
        elec_input_current[anchor_j] = 1.0 + 0j

        elec_total = _solve_cluster_currents(
            electrodes=world.electrodes,
            elec_input_current=elec_input_current,
            cluster_id=cluster_id,
            seg_points=seg_points,
            seg_lengths=seg_lengths,
            wire_radii=wire_radii,
            elec_to_segidx=elec_to_segidx,
            self_kernel=_cached_bulk_self_kernel,
            finite_branches=finite_branches,
            pseudo_owners=pseudo_owners,
            omega=omega if has_inductance else 0.0,
            inductance_matrix=inductance_matrix_full if has_inductance else None,
            carson_correction=carson_dz,
            shell_coefficients=seg_shell_coeffs,
        )

        # Reconstruct the per-segment leakage column (image.py:1455-1463).
        sc = np.zeros(n_segments, dtype=complex)
        for ename, idxs in elec_to_segidx.items():
            if not idxs:
                continue
            I_total = elec_total.get(ename, 0j)
            if I_total == 0j:
                continue
            L_total = seg_lengths[idxs].sum()
            sc[idxs] = I_total * seg_lengths[idxs] / L_total
        T[:, j] = sc

        # -- Diagonal Z[j, j] == cluster_impedance(anchor_j) -----------
        # Build ``ph`` (potential at every segment midpoint) exactly as
        # the post-solve step of the live solver: bulk self-kernel on
        # the real and imaginary leakage, then the shell augmentation.
        members = cluster_members.get(anchor_j, [anchor_j])
        i_sum = sum(elec_total.get(m, 0j) for m in members)
        member0_idxs = elec_to_segidx[members[0]]
        if sc.any():
            phi_re = Z_seg_full @ sc.real
            phi_im = Z_seg_full @ sc.imag
            if shell_diag is not None:
                phi_re = phi_re + shell_diag * sc.real
                phi_im = phi_im + shell_diag * sc.imag
            ph_member0 = (phi_re + 1j * phi_im)[member0_idxs]
            u_cluster = complex(np.mean(ph_member0))
        else:
            u_cluster = 0j
        Z[j, j] = (u_cluster / i_sum) if i_sum != 0 else complex("nan")

    # ------------------------------------------------------------------
    # 5) Off-diagonal block.  ``Z[i, j] = potential at probe_i due to the
    #    leakage column sc_j`` — the probe kernel is built ONCE and
    #    applied to all excitation columns at once.
    # ------------------------------------------------------------------
    if soil_two_layer is not None:
        Phi = _probe_potential_two_layer(
            probe_points,
            seg_points,
            T,
            soil_two_layer,
            matrix_min_distance,
            max_terms=matrix_max_terms,
            tol=matrix_tol,
        )  # (nG, nG): row i = probe i, col j = excitation j
    else:
        K_probe = _probe_kernel_homogeneous(
            probe_points, seg_points, float(rho_homog), matrix_min_distance
        )  # (nG, N)
        Phi = K_probe @ T.real + 1j * (K_probe @ T.imag)

    for j in range(nG):
        for i in range(nG):
            if i == j:
                continue
            Z[i, j] = Phi[i, j]

    if symmetrize:
        return 0.5 * (Z + Z.T)
    return Z


def solve_mutual_field(
    world: "World",
    engine: "Engine",
    anchors: list[str],
    field_points: np.ndarray,
    *,
    frequency_index: int = 0,
    matrix_min_distance: float = 1e-3,
    matrix_max_terms: int = 100,
    matrix_tol: float = 1e-6,
) -> np.ndarray:
    """Galvanic Green-function matrix: potential at arbitrary field points per node.

    Returns ``R`` of shape ``(M, nG)`` with ``R[i, j]`` = potential at
    ``field_points[i]`` (``x, y, depth``) under unit current (1 A) injected
    into ``anchors[j]`` while all other clusters stay floating. This is the
    off-diagonal field evaluation of :func:`solve_mutual_matrix`, generalised
    to an **arbitrary** number of field points (decoupled from ``nG``) and
    without the self/diagonal term — the building block for the
    frequency-dependent surface-potential distribution by superposition::

        phi(field_points, f) = R @ I_leak(f)

    with ``I_leak(f) = Y_G @ u(f)`` from the reduced network
    ``(Y_L(f) + Y_G) u = i``. Uses the same one-shot assembly as
    :func:`solve_mutual_matrix` (reaction matrix and per-excitation leakage
    columns built once); only the field-point evaluation is generalised. The
    expensive 3D part is therefore frequency-independent and computed once, so
    a frequency sweep over ``I_leak(f)`` is cheap.

    ``world`` must already carry **all** ``nG`` grounding clusters (every
    anchor in ``anchors`` must be a real electrode in ``world``) and the soil
    model. Any current source is ignored — the excitations are applied
    internally, one cluster at a time, exactly as the historic per-excitation
    loop did.

    Parameters
    ----------
    world
        Assembled world holding all ``nG`` grounding clusters and the soil
        model. Sources, if any, are ignored.
    engine
        Engine configuration. ``segment_length`` controls the
        discretisation; ``frequencies`` selects the solved frequency via
        ``frequency_index``; ``image_max_terms`` / ``image_series_tol`` feed
        the 2-layer **self**-kernel (the reaction-matrix assembly), exactly
        as ``Engine.solve`` would.
    anchors
        Length-``nG`` list of electrode names, one per node. Anchor ``j``
        receives the unit current of excitation ``j`` and fills column ``j``
        of ``R``.
    field_points
        ``(M, 3)`` array of ``(x, y, depth)`` evaluation points, decoupled
        from ``nG``: ``M`` may differ from ``nG`` and the points are
        arbitrary (surface or buried). Row ``i`` becomes row ``i`` of ``R``.
    frequency_index
        Index into ``engine.frequencies``. Default 0. Only relevant when
        distributed conductors make the assembled system
        frequency-dependent.
    matrix_min_distance, matrix_max_terms, matrix_tol
        Truncation / clamp parameters of the field evaluation. The defaults
        ``1e-3 / 100 / 1e-6`` reproduce the hard-coded defaults of
        :meth:`FieldResult.potential` / ``_potential_two_layer`` exactly.
        (The reaction-matrix assembly uses ``engine.image_max_terms`` /
        ``image_series_tol`` independently, just like the live solver.)

    Returns
    -------
    R : np.ndarray, shape (M, nG), complex
        Galvanic Green-function matrix; ``R[i, j]`` is the potential at
        ``field_points[i]`` under unit excitation of cluster ``j``.

    Raises
    ------
    ValueError
        If ``world`` has no soil model, holds no electrodes, or
        ``field_points`` does not have shape ``(M, 3)``.
    KeyError
        If an entry of ``anchors`` is not an electrode in ``world``.
    NotImplementedError
        If the soil is a ``MultiLayerSoil`` with three or more layers
        (only homogeneous and 2-layer soils are supported).
    TypeError
        If the soil model type is not supported.

    See Also
    --------
    solve_mutual_matrix : Full ``nG × nG`` mutual grounding-impedance matrix.
    """
    if world.soil is None:
        raise ValueError("World has no soil model.")
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    soil = world.soil
    nG = len(anchors)
    field_points = np.asarray(field_points, dtype=float)
    if field_points.ndim != 2 or field_points.shape[1] != 3:
        raise ValueError(f"field_points must have shape (M, 3), got {field_points.shape}.")

    # 1) Discretisation + topology — identical to solve_mutual_matrix.
    ds = engine.segment_length
    all_segments: list[_Segment] = []
    elec_to_segidx: dict[str, list[int]] = {}
    interfaces = (
        (world.soil.h_1,) if isinstance(world.soil, TwoLayerSoil) else None
    )
    for e in world.electrodes:
        segs = _discretize_electrode(e, ds, layer_interfaces=interfaces)
        elec_to_segidx[e.name] = list(range(len(all_segments), len(all_segments) + len(segs)))
        all_segments.extend(segs)

    cluster_id = _build_clusters(world.electrodes, world.conductors)
    finite_branches = _build_finite_branches(world.conductors, cluster_id)
    cond_segs, distributed_branches_objs, interior_nodes = (
        _build_distributed_topology(world.conductors, cluster_id)
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
    distributed_branch_tuples = [(db.node_a, db.node_b, db.R) for db in distributed_branches_objs]
    finite_branches = list(finite_branches) + distributed_branch_tuples

    earth_inductive_model = getattr(engine, "earth_inductive_model", "perfect_mirror")
    sigma_earth_for_carson: float | None = None
    layered_earth_for_sommerfeld: object = None
    if earth_inductive_model == "carson_series":
        from groundfield.coupling import resolve_earth_conductivity
        sigma_earth_for_carson = resolve_earth_conductivity(soil)
    elif earth_inductive_model == "sommerfeld":
        from groundfield.coupling import resolve_earth_layers
        layered_earth_for_sommerfeld = resolve_earth_layers(soil)
    inductance_matrix_full, has_inductance, carson_builder = _assemble_inductance_matrix(
        distributed_branches_objs,
        n_lumped_branches=n_lumped_branches,
        n_total_branches=len(finite_branches),
        earth_model=earth_inductive_model,
        sigma_earth=sigma_earth_for_carson,
        layered_earth=layered_earth_for_sommerfeld,
    )

    n_segments = len(all_segments)
    seg_points = np.array([s.midpoint for s in all_segments])
    seg_lengths = np.array([s.length for s in all_segments])
    wire_radii = np.array([s.wire_radius for s in all_segments])
    seg_shell_coeffs = np.array(
        [s.concrete_shell_coefficient_ohm_m for s in all_segments], dtype=float
    )

    # 2) Soil self-kernel — same case split as solve_mutual_matrix.
    rho_homog: float | None = None
    soil_two_layer: TwoLayerSoil | None = None
    if isinstance(soil, TwoLayerSoil):
        soil_two_layer = soil
    elif isinstance(soil, MultiLayerSoil):
        n_layers = len(soil.layers)
        if n_layers == 1:
            rho_homog = float(soil.layers[0].resistivity)
        elif n_layers == 2:
            soil_two_layer = TwoLayerSoil(
                rho_1=float(soil.layers[0].resistivity),
                rho_2=float(soil.layers[1].resistivity),
                h_1=float(soil.layers[0].thickness),
            )
        else:
            raise NotImplementedError(
                "solve_mutual_field supports homogeneous and 2-layer soils; "
                f"got MultiLayerSoil with n={n_layers} layers."
            )
    elif isinstance(soil, HomogeneousSoil):
        rho_homog = float(soil.resistivity)
    else:
        raise TypeError(f"Unsupported soil type {type(soil).__name__}.")

    if soil_two_layer is not None:
        self_max_terms = int(getattr(engine, "image_max_terms", 100))
        self_tol = float(getattr(engine, "image_series_tol", 1e-6))
        z_max = float(seg_points[:, 2].max())
        cross_layer = bool(z_max >= soil_two_layer.h_1)
        self_kernel = _two_layer_self_kernel_factory(
            soil_two_layer, self_max_terms, self_tol, allow_cross_layer=cross_layer
        )
    else:
        rho_for_kernel = float(rho_homog)

        def self_kernel(seg_pts, seg_lens, wr, currents):  # noqa: ANN001
            return _self_corrected_kernel(seg_pts, seg_lens, wr, currents, rho_for_kernel)

    eye = np.eye(n_segments)
    Z_seg_full = self_kernel(seg_points, seg_lengths, wire_radii, eye)
    omega = (
        2.0 * np.pi * float(engine.frequencies[frequency_index]) if has_inductance else 0.0
    )
    carson_dz = (
        carson_builder(omega) if (has_inductance and carson_builder is not None) else None
    )

    def _cached_bulk_self_kernel(seg_pts, seg_lens, wr, currents):  # noqa: ANN001
        return Z_seg_full @ currents

    # 3) Per-excitation leakage columns T (no diagonal/self term needed).
    T = np.zeros((n_segments, nG), dtype=complex)
    for j, anchor_j in enumerate(anchors):
        if anchor_j not in elec_to_segidx:
            raise KeyError(f"Anchor '{anchor_j}' is not an electrode in world.")
        elec_input_current = {e.name: 0j for e in world.electrodes}
        elec_input_current[anchor_j] = 1.0 + 0j
        elec_total = _solve_cluster_currents(
            electrodes=world.electrodes,
            elec_input_current=elec_input_current,
            cluster_id=cluster_id,
            seg_points=seg_points,
            seg_lengths=seg_lengths,
            wire_radii=wire_radii,
            elec_to_segidx=elec_to_segidx,
            self_kernel=_cached_bulk_self_kernel,
            finite_branches=finite_branches,
            pseudo_owners=pseudo_owners,
            omega=omega if has_inductance else 0.0,
            inductance_matrix=inductance_matrix_full if has_inductance else None,
            carson_correction=carson_dz,
            shell_coefficients=seg_shell_coeffs,
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
        T[:, j] = sc

    # 4) Potential at the M field points per excitation (nG) — one kernel pass.
    if soil_two_layer is not None:
        R = _probe_potential_two_layer(
            field_points, seg_points, T, soil_two_layer,
            matrix_min_distance, max_terms=matrix_max_terms, tol=matrix_tol,
        )
    else:
        K_probe = _probe_kernel_homogeneous(
            field_points, seg_points, float(rho_homog), matrix_min_distance
        )
        R = K_probe @ T.real + 1j * (K_probe @ T.imag)
    return R
