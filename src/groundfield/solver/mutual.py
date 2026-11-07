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

1. The segment reaction matrix is assembled **once**, directly (the
   kernel operators' ``matrix`` method; v0.15.0 applied the kernel to an
   ``N × N`` identity, one ``O(N³)`` product per image term — review
   pass 10, 10.59), including the ADR-0012 V2 concrete shell.
2. The off-diagonal probe-point kernel is built **once** and reused for
   every excitation column through a single ``Phi = K @ T`` matmul.

Discretisation, topology and the branch block come from the shared
image-family pipeline (:mod:`groundfield.solver._pipeline`).

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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.solver._kernels import (
    HalfSpaceKernel,
    _point_field_kernel,
    _series_field,
)
from groundfield.solver._pipeline import (
    BranchModel,
    Discretisation,
    SegmentArrays,
    Topology,
    _segment_currents,
    assemble_branches,
    build_topology,
    discretise,
)
from groundfield.solver._series import ImageSeries
from groundfield.solver._topology import _solve_cluster_currents
from groundfield.solver.image_2layer import _two_layer_self_kernel_factory

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_mutual_matrix", "solve_mutual_field"]


# ---------------------------------------------------------------------
# Probe-point field kernels (matrix-valued, reused over all columns)
# ---------------------------------------------------------------------


def _probe_kernel_homogeneous(
    probe_points: np.ndarray,  # (M, 3)
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
    return (rho / (4.0 * np.pi)) * _point_field_kernel(
        probe_points, source_points, min_distance
    )


def _probe_potential_two_layer(
    probe_points: np.ndarray,  # (M, 3)
    source_points: np.ndarray,  # (N, 3)
    currents: np.ndarray,  # (N, J) complex or real
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
                probe_points[probe_l1],
                source_points[src_l1],
                currents[src_l1],
                soil,
                min_distance,
                max_terms,
                tol,
            )
        # Remaining blocks via the spectral kernel: layer-2 sources to
        # all probes, layer-1 sources to layer-2 probes.
        if (~src_l1).any():
            G = two_layer_probe_matrix(
                probe_points,
                source_points[~src_l1],
                rho_1=rho_1,
                rho_2=rho_2,
                h_1=h_1,
                min_distance=min_distance,
            )
            phi += G @ currents[~src_l1]
        if (~probe_l1).any() and src_l1.any():
            G = two_layer_probe_matrix(
                probe_points[~probe_l1],
                source_points[src_l1],
                rho_1=rho_1,
                rho_2=rho_2,
                h_1=h_1,
                min_distance=min_distance,
            )
            phi[~probe_l1] += G @ currents[src_l1]
        return phi

    series = ImageSeries.for_soil(soil, max_terms, tol)
    # An all-zero imaginary part sums to exactly zero and is skipped.
    parts = [currents.real]
    if np.any(currents.imag):
        parts.append(currents.imag)
    sums = _series_field(probe_points, source_points, parts, series, min_distance)
    phi_re = sums[0]
    phi_im = sums[1] if len(sums) > 1 else np.zeros_like(phi_re)
    return (rho_1 / (4.0 * np.pi)) * (phi_re + 1j * phi_im)


# ---------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------


@dataclass
class _MutualProblem:
    """Discretised world, reaction matrix and branch block of one call."""

    disc: Discretisation
    topo: Topology
    branches: BranchModel
    arrays: SegmentArrays
    soil_two_layer: TwoLayerSoil | None
    rho_homog: float | None
    Z_seg_full: np.ndarray
    omega: float
    carson_dz: np.ndarray | None


def _prepare(
    world: "World", engine: "Engine", frequency_index: int, *, where: str
) -> _MutualProblem:
    """Discretise, build the topology and assemble the reaction matrix.

    Steps 1, 3 and 4 of the image-family pipeline, then the soil
    dispatch of ``Engine.solve`` (``image`` → ``image_2layer`` for a
    two-layer soil) and the full segment reaction matrix in Ω.
    """
    soil = world.soil
    # 1) Discretisation + topology — identical to the live backends.
    interfaces = (soil.h_1,) if isinstance(soil, TwoLayerSoil) else None
    disc = discretise(world, engine, interfaces)
    topo = build_topology(world, disc)
    # Inductive coupling assembly (kept for parity; the catalogue probe
    # case has no distributed conductors, so the system stays
    # real/frequency-independent — the image no-inductance fast path).
    branches = assemble_branches(world, engine, topo)
    arrays = SegmentArrays.from_segments(disc.segments)

    # 2) Soil-specific reaction operator — same dispatch as
    #    ``Engine.solve`` (image -> image_2layer for TwoLayerSoil).
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
                f"{where} supports homogeneous and 2-layer soils; "
                f"got MultiLayerSoil with n={n_layers} layers."
            )
    elif isinstance(soil, HomogeneousSoil):
        rho_homog = float(soil.resistivity)
    else:
        raise TypeError(f"Unsupported soil type {type(soil).__name__}.")

    if soil_two_layer is not None:
        # 2-layer reaction operator: feed the engine's image-series
        # truncation, exactly like ``solve_image_2layer``.
        z_max = float(arrays.points[:, 2].max())
        kernel = _two_layer_self_kernel_factory(
            soil_two_layer,
            int(getattr(engine, "image_max_terms", 100)),
            float(getattr(engine, "image_series_tol", 1e-6)),
            allow_cross_layer=bool(z_max >= soil_two_layer.h_1),
        )
    else:
        kernel = HalfSpaceKernel(float(rho_homog))

    # 3) Reaction matrix ONCE, assembled directly (10.59): the N×N
    #    segment reaction matrix, column k is the potential of 1 A in
    #    segment k. Reusing it for every excitation makes the diagonal
    #    ``ph`` evaluation and the cluster-current assembly a sequence of
    #    matvecs instead of nG full kernel calls.
    Z_seg_full = kernel.matrix(arrays.points, arrays.lengths, arrays.radii)

    omega = (
        2.0 * np.pi * float(engine.frequencies[frequency_index])
        if branches.has_inductance
        else 0.0
    )
    carson_dz = (
        branches.earth_correction(omega)
        if branches.earth_correction is not None
        else None
    )
    return _MutualProblem(
        disc=disc,
        topo=topo,
        branches=branches,
        arrays=arrays,
        soil_two_layer=soil_two_layer,
        rho_homog=rho_homog,
        Z_seg_full=Z_seg_full,
        omega=omega,
        carson_dz=carson_dz,
    )


def _unit_excitation(
    world: "World", problem: _MutualProblem, anchor: str
) -> tuple[dict[str, complex], np.ndarray]:
    """Owner currents and segment currents for 1 A into ``anchor``.

    Each gnode is an isolated galvanic cluster, so injecting 1 A into
    cluster ``j`` activates only that cluster; the per-segment leakage
    column is non-zero on cluster-``j`` segments only — exactly what
    the historic single-source solve produced.

    Raises
    ------
    KeyError
        If ``anchor`` is not an electrode of ``world``.
    """
    disc, topo, arrays = problem.disc, problem.topo, problem.arrays
    if anchor not in disc.owner_segments:
        raise KeyError(f"Anchor '{anchor}' is not an electrode in world.")
    elec_input_current = {e.name: 0j for e in world.electrodes}
    elec_input_current[anchor] = 1.0 + 0j
    Z_seg_full = problem.Z_seg_full

    # A linear ``self_kernel``-equivalent backed by the precomputed
    # matrix. The shell augmentation is applied on top by
    # ``_solve_cluster_currents`` (bulk kernel, then
    # ``phi += shell_diag · currents``), so the kernel is *bulk-only* —
    # the same operation order as the live solver.
    def _cached_bulk_self_kernel(seg_pts, seg_lens, wr, currents):  # noqa: ANN001
        return Z_seg_full @ currents

    has_inductance = problem.branches.has_inductance
    elec_total = _solve_cluster_currents(
        electrodes=world.electrodes,
        elec_input_current=elec_input_current,
        cluster_id=topo.cluster_id,
        seg_points=arrays.points,
        seg_lengths=arrays.lengths,
        wire_radii=arrays.radii,
        elec_to_segidx=disc.owner_segments,
        self_kernel=_cached_bulk_self_kernel,
        finite_branches=topo.finite_branches,
        pseudo_owners=topo.pseudo_owners,
        omega=problem.omega if has_inductance else 0.0,
        inductance_matrix=problem.branches.inductance_matrix,
        carson_correction=problem.carson_dz,
        shell_coefficients=arrays.shell_coefficients,
    )
    sc = _segment_currents(elec_total, disc.owner_segments, arrays.lengths)
    return elec_total, sc


def _field_of_columns(
    problem: _MutualProblem,
    field_points: np.ndarray,
    T: np.ndarray,
    *,
    min_distance: float,
    max_terms: int,
    tol: float,
) -> np.ndarray:
    """Potential at ``field_points`` of every leakage column of ``T``."""
    seg_points = problem.arrays.points
    if problem.soil_two_layer is not None:
        return _probe_potential_two_layer(
            field_points,
            seg_points,
            T,
            problem.soil_two_layer,
            min_distance,
            max_terms=max_terms,
            tol=tol,
        )
    K_probe = _probe_kernel_homogeneous(
        field_points, seg_points, float(problem.rho_homog), min_distance
    )
    return K_probe @ T.real + 1j * (K_probe @ T.imag)


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
    if world.soil is None:
        raise ValueError("World has no soil model.")
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    nG = len(anchors)
    probe_points = np.asarray(probe_points, dtype=float)
    if probe_points.shape != (nG, 3):
        raise ValueError(
            f"probe_points must have shape ({nG}, 3), got {probe_points.shape}."
        )

    problem = _prepare(world, engine, frequency_index, where="solve_mutual_matrix")
    disc, topo, arrays = problem.disc, problem.topo, problem.arrays
    n_segments = len(disc.segments)
    seg_lengths = arrays.lengths
    Z_seg_full = problem.Z_seg_full

    # ADR-0012 V2 shell augmentation (added to the *diagonal* of the
    # reaction matrix exactly as in ``_solve_cluster_currents`` and the
    # post-solve ``ph`` step).
    shell_diag: np.ndarray | None = None
    if np.any(arrays.shell_coefficients > 0.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            shell_diag = np.where(
                seg_lengths > 0.0,
                arrays.shell_coefficients / seg_lengths,
                0.0,
            )

    # ------------------------------------------------------------------
    # 4) Per-excitation leakage distribution (one isolated gnode each).
    # ------------------------------------------------------------------
    real_electrode_names = [e.name for e in world.electrodes]
    real_set = set(real_electrode_names)
    # cluster members per real electrode (== FieldResult.clusters), used
    # for the diagonal ``cluster_impedance`` average and current sum.
    cluster_id = topo.cluster_id
    cluster_members: dict[str, list[str]] = {}
    for ename in real_electrode_names:
        cluster_members[ename] = sorted(
            n
            for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_set
        )

    T = np.zeros((n_segments, nG), dtype=complex)  # column j = sc_j
    Z = np.zeros((nG, nG), dtype=complex)

    for j, anchor_j in enumerate(anchors):
        elec_total, sc = _unit_excitation(world, problem, anchor_j)
        T[:, j] = sc

        # -- Diagonal Z[j, j] == cluster_impedance(anchor_j) -----------
        # Build ``ph`` (potential at every segment midpoint) exactly as
        # the post-solve step of the live solver: bulk reaction matrix on
        # the real and imaginary leakage, then the shell augmentation.
        members = cluster_members.get(anchor_j, [anchor_j])
        i_sum = sum(elec_total.get(m, 0j) for m in members)
        member0_idxs = disc.owner_segments[members[0]]
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
    Phi = _field_of_columns(
        problem,
        probe_points,
        T,
        min_distance=matrix_min_distance,
        max_terms=matrix_max_terms,
        tol=matrix_tol,
    )  # (nG, nG): row i = probe i, col j = excitation j

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

    nG = len(anchors)
    field_points = np.asarray(field_points, dtype=float)
    if field_points.ndim != 2 or field_points.shape[1] != 3:
        raise ValueError(
            f"field_points must have shape (M, 3), got {field_points.shape}."
        )

    problem = _prepare(world, engine, frequency_index, where="solve_mutual_field")
    n_segments = len(problem.disc.segments)

    # Per-excitation leakage columns T (no diagonal/self term needed).
    T = np.zeros((n_segments, nG), dtype=complex)
    for j, anchor_j in enumerate(anchors):
        _, T[:, j] = _unit_excitation(world, problem, anchor_j)

    # Potential at the M field points per excitation (nG) — one kernel pass.
    return _field_of_columns(
        problem,
        field_points,
        T,
        min_distance=matrix_min_distance,
        max_terms=matrix_max_terms,
        tol=matrix_tol,
    )
