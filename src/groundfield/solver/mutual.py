"""Mutual grounding-impedance matrix and field matrix in one assembly.

:func:`solve_mutual_matrix` computes the full ``nG × nG`` mutual
grounding-impedance matrix ``Z_G`` of a set of grounding clusters
("gnodes"), :func:`solve_mutual_field` the potential at arbitrary points
per unit excitation, each in **one** matrix assembly instead of one
``Engine.solve`` per excitation.

Motivation
----------
The AP1 catalogue builds ``Z_G`` for every site and soil. The historic
loop (one world per excitation, ``cluster_impedance`` on the diagonal,
``FieldResult.potential`` at a probe point off the diagonal) assembled
the ``N × N`` segment reaction matrix ``nG`` times. Here:

1. The segment reaction matrix is assembled **once**, directly (the
   kernel operators' ``matrix`` method; v0.15.0 applied the kernel to an
   ``N × N`` identity, one ``O(N³)`` product per image term — review
   pass 10, 10.59), including the ADR-0012 V2 concrete shell.
2. The leakage distribution of every excitation is a solve of the same
   augmented nodal system as :meth:`Engine.solve`, with one owner per
   (electrode, soil layer) for electrodes through the layer interface
   (10.15).
3. The off-diagonal entries are either the reaction of the leakage
   columns (``offdiag="galerkin"``, no further kernel evaluation) or the
   probe-point potentials (``offdiag="probe"``, one probe kernel for all
   columns).

Discretisation, topology and the branch block come from the shared
image-family pipeline (:mod:`groundfield.solver._pipeline`); only the
image family is accepted (10.34).

Consistency with ``Engine.solve``
---------------------------------
The diagonal equals :meth:`FieldResult.cluster_impedance` of a
per-excitation solve with the same engine to round-off (10.26), probe
off-diagonals equal :meth:`FieldResult.potential` at the probe point,
and the columns of :func:`solve_mutual_field` equal
``FieldResult.potential`` at the field points
(``tests/test_mutual_matrix.py``, ``tests/test_mutual_field.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.solver._discretize import _LayerOwners
from groundfield.solver._kernels import (
    DEFAULT_KERNEL,
    HalfSpaceKernel,
    _line_field_correction,
    _point_field_kernel,
    _series_field,
)
from groundfield.solver._line_kernel import LineSegments
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

__all__ = ["OffDiagonal", "solve_mutual_matrix", "solve_mutual_field"]

#: Off-diagonal definition of :func:`solve_mutual_matrix` (10.34).
OffDiagonal = Literal["probe", "galerkin"]


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
    max_terms: int,
    tol: float,
    line: LineSegments | None = None,
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
                line=None if line is None else line.subset(src_l1),
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
    series.warn_if_truncated("solve_mutual_matrix / solve_mutual_field")
    # An all-zero imaginary part sums to exactly zero and is skipped.
    parts = [currents.real]
    if np.any(currents.imag):
        parts.append(currents.imag)
    sums = _series_field(
        probe_points, source_points, parts, series, min_distance, line=line
    )
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
    line: LineSegments | None
    max_terms: int
    tol: float
    layer_owners: _LayerOwners


def _layers_differ(soil: TwoLayerSoil | None) -> bool:
    """Whether a two-layer soil has two resistivities (one owner per layer, 10.15)."""
    return soil is not None and soil.rho_1 != soil.rho_2


#: Backends whose model the multi-port assembly of this module reproduces:
#: uniform current per owner, the image-family kernels (10.34).
_IMAGE_FAMILY = frozenset({"image", "image_2layer", "image_nlayer"})


def _check_backend(world: "World", engine: "Engine", where: str) -> str:
    """Check the engine's backend and return the kernel it runs (10.34).

    ``solve_mutual_*`` assemble the image-family multi-port. Up to 2026.11
    they ignored ``Engine.backend``: an engine for ``mom`` returned image
    numbers (an 8-arm star 7.854 Ω instead of mom's 7.128 Ω). The backend
    is resolved and checked like :meth:`Engine.solve` does (aliases, soil
    forwarding, deprecation warnings, capability check); the sources of
    ``world`` are ignored here as in the assembly.

    Returns
    -------
    str
        Kernel of the run: ``engine.kernel``, or the backend's preferred
        kernel if the engine leaves it to the backend (``None``).

    Raises
    ------
    BackendCapabilityError
        If the resolved backend is not ``image``, ``image_2layer`` or
        ``image_nlayer``, or cannot represent ``world``.
    """
    from groundfield.solver import registry

    resolved = registry.resolve_backend(engine.backend, world)
    registry.emit_resolution_warnings(resolved)
    if resolved.spec.name not in _IMAGE_FAMILY:
        raise registry.BackendCapabilityError(
            f"{where} assembles the multi-port of the image family (uniform "
            f"current per electrode); backend '{engine.backend}' (running "
            f"'{resolved.spec.name}') solves a different model. Use an "
            "engine with backend='image' (forwards to image_2layer / "
            "image_nlayer by soil), or solve one world per excitation with "
            "Engine.solve for the other backends."
        )
    requested = engine.kernel
    registry.check_capabilities(
        resolved,
        world.model_copy(update={"sources": []}),
        kernel=requested,
        kernel_explicit=requested is not None,
    )
    kernel = registry.effective_kernel(
        resolved.spec, requested if requested is not None else DEFAULT_KERNEL
    )
    return str(kernel)


def _prepare(
    world: "World", engine: "Engine", frequency_index: int, *, where: str
) -> _MutualProblem:
    """Discretise, build the topology and assemble the reaction matrix.

    Steps 1, 3 and 4 of the image-family pipeline, then the soil
    dispatch of ``Engine.solve`` (``image`` → ``image_2layer`` for a
    two-layer soil) and the full segment reaction matrix in Ω.
    """
    kernel_name = _check_backend(world, engine, where)
    soil = world.soil
    # Soil dispatch of ``Engine.solve`` (image -> image_2layer /
    # image_nlayer): a layer stack is solved through its canonical form,
    # equal adjacent layers merged (10.48).
    rho_homog: float | None = None
    soil_two_layer: TwoLayerSoil | None = None
    if isinstance(soil, TwoLayerSoil):
        soil_two_layer = soil
    elif isinstance(soil, MultiLayerSoil):
        from groundfield.solver._layered import as_layer_stack

        stack = as_layer_stack(soil)
        if stack.n_layers == 1:
            rho_homog = float(stack.rhos[0])
        elif stack.n_layers == 2:
            soil_two_layer = TwoLayerSoil(
                rho_1=float(stack.rhos[0]),
                rho_2=float(stack.rhos[1]),
                h_1=float(stack.h[0]),
            )
        else:
            from groundfield.solver import registry

            raise registry.BackendCapabilityError(
                f"{where} supports homogeneous and 2-layer soils; "
                f"got MultiLayerSoil with n={stack.n_layers} layers."
            )
    elif isinstance(soil, HomogeneousSoil):
        rho_homog = float(soil.rho)
    else:
        raise TypeError(f"Unsupported soil type {type(soil).__name__}.")

    # 1) Discretisation + topology — identical to the live backends,
    #    including one owner per (electrode, layer) for electrodes
    #    through the layer interface (10.15).
    interfaces = (soil_two_layer.h_1,) if soil_two_layer is not None else None
    disc = discretise(world, engine, interfaces)
    topo = build_topology(world, disc)
    layer_owners = _LayerOwners.split(
        disc,
        topo,
        [e.name for e in world.electrodes],
        interfaces if _layers_differ(soil_two_layer) else None,
    )
    # Inductive coupling assembly (kept for parity; the catalogue probe
    # case has no distributed conductors, so the system stays
    # real/frequency-independent — the image no-inductance fast path).
    # One call solves one frequency: the branch correction would otherwise
    # be assembled for the engine's whole frequency set on first use
    # (10.60). An index out of range only matters for an inductive world,
    # where it raises below as before.
    n_freq = len(engine.frequencies)
    if -n_freq <= frequency_index < n_freq:
        engine = engine._with(frequencies=[engine.frequencies[frequency_index]])
        frequency_index = 0
    branches = assemble_branches(world, engine, topo)
    arrays = SegmentArrays.from_segments(disc.segments)

    # Kernel of the run (10.28): the engine's, or the backend's preferred
    # one (exact line sources) — for the reaction matrix and the field
    # evaluation alike.
    line = arrays.line_segments() if kernel_name == "line" else None
    if soil_two_layer is not None:
        # 2-layer reaction operator: feed the engine's image-series
        # truncation, exactly like ``solve_image_2layer``.
        z_max = float(arrays.points[:, 2].max())
        kernel = _two_layer_self_kernel_factory(
            soil_two_layer,
            int(engine.image_max_terms),
            float(engine.image_series_tol),
            allow_cross_layer=bool(z_max >= soil_two_layer.h_1),
            line=line,
        )
    else:
        kernel = HalfSpaceKernel(float(rho_homog), line=line)

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
        line=line,
        max_terms=int(engine.image_max_terms),
        tol=float(engine.image_series_tol),
        layer_owners=layer_owners,
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
        # The other clusters float (10.29), as in Engine.solve: zero net
        # current, equalising currents inside bonded groups and between
        # the layer owners of a rod through the interface.
        floating=True,
    )
    sc = _segment_currents(elec_total, disc.owner_segments, arrays.lengths)
    return elec_total, sc


def _field_of_columns(
    problem: _MutualProblem,
    field_points: np.ndarray,
    T: np.ndarray,
    *,
    min_distance: float,
    max_terms: int | None,
    tol: float | None,
) -> np.ndarray:
    """Potential at ``field_points`` of every leakage column of ``T``.

    ``max_terms`` / ``tol`` of ``None`` take the engine's series controls
    (review pass 10, 10.25: they were 100 / 1e-6 whatever the engine said).
    """
    seg_points = problem.arrays.points
    if problem.soil_two_layer is not None:
        return _probe_potential_two_layer(
            field_points,
            seg_points,
            T,
            problem.soil_two_layer,
            min_distance,
            max_terms=problem.max_terms if max_terms is None else int(max_terms),
            tol=problem.tol if tol is None else float(tol),
            line=problem.line,
        )
    K_probe = _probe_kernel_homogeneous(
        field_points, seg_points, float(problem.rho_homog), min_distance
    )
    Phi = K_probe @ T.real + 1j * (K_probe @ T.imag)
    if problem.line is not None:
        # Line sources (10.28): far-field and near-source corrections.
        re, im = _line_field_correction(
            field_points, problem.line, [T.real, T.imag], min_distance
        )
        Phi = Phi + (float(problem.rho_homog) / (4.0 * np.pi)) * (re + 1j * im)
    return Phi


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_mutual_matrix(
    world: "World",
    engine: "Engine",
    anchors: list[str],
    probe_points: np.ndarray | None = None,
    *,
    frequency_index: int = 0,
    symmetrize: bool = True,
    offdiag: OffDiagonal | None = None,
    matrix_min_distance: float = 1e-3,
    matrix_max_terms: int | None = None,
    matrix_tol: float | None = None,
) -> np.ndarray:
    """Full mutual grounding-impedance matrix ``Z_G`` in one assembly.

    Column ``j`` belongs to 1 A injected into ``anchors[j]``; the other
    clusters float (zero net current, review pass 10, 10.29), exactly as
    in :meth:`Engine.solve` with one source. The segment reaction matrix and the
    leakage distribution of every excitation are computed once and
    reused for all entries.

    * **Diagonal.** ``Z[j, j]`` is the cluster impedance of ``anchors[j]``,
      the length-weighted (Galerkin) potential average over every
      member of its cluster divided by the cluster current — the same
      reduction as :meth:`FieldResult.cluster_impedance` after a
      per-excitation :meth:`Engine.solve` (equal to round-off). Up to
      2026.11 the diagonal was the unweighted mean over the
      alphabetically first member: +0.150 % for an L-shaped footprint
      at ds = 2 m, -3.8 … -9.1 % for a rod split at the layer interface
      (review pass 10, 10.26).
    * **Off-diagonal**, ``offdiag="galerkin"``: the reaction
      ``Z[i, j] = T_iᵀ (K + S) T_j`` of the leakage columns ``T`` with
      the reaction matrix ``K`` and the concrete-shell diagonal ``S``.
      It is the current-weighted potential of cluster ``i`` under
      excitation ``j`` — the off-diagonal counterpart of the diagonal,
      reciprocal to round-off, and the matrix that reproduces a solve
      of the clusters joined by branches (see Notes).
    * **Off-diagonal**, ``offdiag="probe"``: the potential at
      ``probe_points[i]`` under excitation ``j``, as
      :meth:`FieldResult.potential` returns it after a per-excitation
      solve. A point potential is not the potential of cluster ``i``:
      the matrix is not reciprocal (``Z[i, j]`` and ``Z[j, i]`` differ by
      up to 35 % in two-layer soil, review pass 10, 10.34), and
      ``symmetrize=True`` averages that asymmetry away.

    Parameters
    ----------
    world
        Assembled world holding all ``nG`` grounding clusters and the
        soil model (homogeneous or two-layer, also as a
        :class:`~groundfield.soil.MultiLayerSoil` of at most two distinct
        layers). Sources, if any, are ignored.
    engine
        Engine of the image family (``backend="image"`` and its
        aliases); ``segment_length``, ``kernel``, ``image_max_terms`` and
        ``image_series_tol`` act exactly as in :meth:`Engine.solve`.
        ``frequencies[frequency_index]`` is the solved frequency.
    anchors
        Length-``nG`` list of electrode names, one per cluster ("gnode").
    probe_points
        ``(nG, 3)`` array of ``(x, y, depth)`` points in m, one per
        anchor: required for ``offdiag="probe"``, rejected for
        ``offdiag="galerkin"``.
    frequency_index
        Index into ``engine.frequencies``. Default 0.
    symmetrize
        If ``True`` (default) return ``0.5 · (Z + Zᵀ)``.
    offdiag
        ``"galerkin"`` or ``"probe"``, see above. ``None`` (default)
        selects ``"probe"`` when ``probe_points`` is given (the call of
        releases up to 0.15.0) and ``"galerkin"`` otherwise.
    matrix_min_distance, matrix_max_terms, matrix_tol
        Clamp distance in m and series controls of the probe-point
        evaluation (``offdiag="probe"`` only). ``None`` (default) for the
        series controls takes ``engine.image_max_terms`` /
        ``engine.image_series_tol``, the controls
        :meth:`FieldResult.potential` uses after a solve with the same
        engine (before 2026.11 they were hard-coded to 100 / ``1e-6``,
        review pass 10, 10.25).

    Returns
    -------
    Z : np.ndarray, shape (nG, nG), complex
        Mutual grounding-impedance matrix in Ω.

    Raises
    ------
    ValueError
        If ``world`` has no soil model or no electrodes, if
        ``probe_points`` does not fit ``offdiag``, or if ``offdiag`` is
        unknown.
    BackendCapabilityError
        If the engine's backend is not the image family (a subclass of
        ``ValueError``; up to 2026.11 the backend was ignored, 10.34).
    KeyError
        If an entry of ``anchors`` is not an electrode in ``world``.

    Notes
    -----
    With the uniform current per owner of the image family, the
    potential of a cluster is the current-weighted average of its
    segment potentials. With every other cluster floating,
    ``T_iᵀ (K + S) T_j`` therefore equals the potential of cluster ``i``
    under excitation ``j`` exactly (reciprocity of the reaction matrix),
    bonded groups and rods through the layer interface included, and
    ``(Y_L + Z_G⁻¹) u = i`` reproduces the solve of the clusters joined
    by the branches ``Y_L``. With probe off-diagonals it does not
    (evidence in the 2026.11 changelog).
    """
    if world.soil is None:
        raise ValueError("World has no soil model.")
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")
    if offdiag is None:
        offdiag = "probe" if probe_points is not None else "galerkin"
    if offdiag not in ("probe", "galerkin"):
        raise ValueError(f"offdiag must be 'probe' or 'galerkin', got {offdiag!r}.")

    nG = len(anchors)
    if offdiag == "probe":
        if probe_points is None:
            raise ValueError("offdiag='probe' needs probe_points of shape (nG, 3).")
        probe_points = np.asarray(probe_points, dtype=float)
        if probe_points.shape != (nG, 3):
            raise ValueError(
                f"probe_points must have shape ({nG}, 3), got {probe_points.shape}."
            )
    elif probe_points is not None:
        raise ValueError(
            "probe_points are not used with offdiag='galerkin'; pass "
            "offdiag='probe' to sample the off-diagonal at points."
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

    # Cluster members per real electrode (== FieldResult.clusters).
    real_set = {e.name for e in world.electrodes}
    cluster_id = topo.cluster_id
    cluster_members: dict[str, list[str]] = {}
    for ename in (e.name for e in world.electrodes):
        cluster_members[ename] = sorted(
            n
            for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_set
        )

    T = np.zeros((n_segments, nG), dtype=complex)  # leakage column per excitation
    PH = np.zeros((n_segments, nG), dtype=complex)  # segment potentials per excitation
    Z = np.zeros((nG, nG), dtype=complex)
    for j, anchor_j in enumerate(anchors):
        elec_total, sc = _unit_excitation(world, problem, anchor_j)
        T[:, j] = sc
        if sc.any():
            # Segment potentials as in the post-solve step of the live
            # solver: bulk reaction matrix on the real and imaginary
            # leakage, then the shell augmentation.
            phi_re = Z_seg_full @ sc.real
            phi_im = Z_seg_full @ sc.imag
            if shell_diag is not None:
                phi_re = phi_re + shell_diag * sc.real
                phi_im = phi_im + shell_diag * sc.imag
            PH[:, j] = phi_re + 1j * phi_im
        members = cluster_members.get(anchor_j, [anchor_j])
        Z[j, j] = _cluster_impedance(problem, members, elec_total, PH[:, j])

    if offdiag == "galerkin":
        # Reaction of every pair of leakage columns (bilinear, no
        # conjugation: reciprocity holds for the complex system too).
        Phi = T.T @ PH
    else:
        Phi = _field_of_columns(
            problem,
            probe_points,
            T,
            min_distance=matrix_min_distance,
            max_terms=matrix_max_terms,
            tol=matrix_tol,
        )  # (nG, nG): row i = probe i, column j = excitation j
    off = ~np.eye(nG, dtype=bool)
    Z[off] = Phi[off]

    if symmetrize:
        return 0.5 * (Z + Z.T)
    return Z


def _cluster_impedance(
    problem: _MutualProblem,
    members: list[str],
    elec_total: dict[str, complex],
    ph: np.ndarray,
) -> complex:
    """Cluster impedance of one excitation, reduced like ``FieldResult``.

    Mirrors :meth:`FieldResult.cluster_impedance` step by step (10.26):
    the electrode current is the sum over its owners (one per soil layer
    for electrodes through an interface, 10.15), the electrode potential
    the length-weighted average over all its segments, the cluster
    potential the electrode-length-weighted average over all members.

    Parameters
    ----------
    problem
        Prepared problem (discretisation and layer owners).
    members
        Sorted member names of the cluster.
    elec_total
        Owner currents of the excitation in A.
    ph
        Segment potentials of the excitation in V, shape ``(n_segments,)``.

    Returns
    -------
    complex
        Cluster impedance in Ω; ``nan`` if the cluster current is zero.
    """
    owners = problem.layer_owners
    lengths = problem.arrays.lengths
    weights: list[float] = []
    potentials: list[complex] = []
    i_sum: complex = 0
    for m in members:
        current = elec_total.get(m, 0j)
        for sub in owners.sub_owners.get(m, []):
            current += elec_total.get(sub, 0j)
        i_sum = i_sum + current
        idxs = owners.segments_of(m, problem.disc.owner_segments)
        w = np.asarray(lengths, dtype=float)[idxs]
        potentials.append(complex((w @ ph[idxs]) / float(w.sum())))
        length = 0.0
        for k in idxs:
            length = length + float(lengths[k])
        weights.append(length)
    w_members = np.array(weights, dtype=float)
    u_cluster = complex(
        sum(float(wm) * u for wm, u in zip(w_members, potentials, strict=True))
        / float(w_members.sum())
    )
    return (u_cluster / i_sum) if i_sum != 0 else complex("nan")


def solve_mutual_field(
    world: "World",
    engine: "Engine",
    anchors: list[str],
    field_points: np.ndarray,
    *,
    frequency_index: int = 0,
    matrix_min_distance: float = 1e-3,
    matrix_max_terms: int | None = None,
    matrix_tol: float | None = None,
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
        Clamp distance in m and series controls of the field evaluation.
        ``None`` (default) for the series controls takes
        ``engine.image_max_terms`` / ``engine.image_series_tol``, the
        controls :meth:`FieldResult.potential` uses after a solve with the
        same engine (before 2026.11 they were hard-coded to 100 /
        ``1e-6``, review pass 10, 10.25).

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
