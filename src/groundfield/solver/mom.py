"""Point-matching method-of-moments backend (``mom``).

Resolution-scheme counterpart of the image backends (ADR-0001). The
``image`` backends approximate the current along an electrode by a
*uniform per unit length* profile; this backend solves for one current
per segment, so it resolves how the current distributes along and
between the electrodes of a cluster.

Method
------
1. Discretise every electrode into wire segments (same routine as the
   image backends; for two-layer soil the segments are split at the
   layer interface).
2. Assemble the $N \\times N$ segment reaction matrix $Z$ from the
   Green's-function kernel of the matching image backend (homogeneous:
   point source plus air image; two-layer: Tagg/Sunde series; geometries
   reaching below $h_1$: the cross-layer kernel of
   :mod:`groundfield.coupling.layered_green`):

   - off-diagonal $Z_{ij}$: potential at the midpoint of segment $i$
     of a unit **point** source at the midpoint of segment $j$;
   - diagonal $Z_{ii}$: analytic potential of the uniformly loaded
     segment at its own surface (line self-potential), plus images.

   The matrix is assembled directly, entry by entry (review pass 10,
   10.59); up to 0.15.0 it was extracted by applying the kernel to the
   identity matrix, an $O(N^3)$ product per image term. The entries
   are bit-identical.

   The basis is a pulse (one constant current per segment) and the test
   function a delta at the segment midpoint. This is **point matching**
   (collocation) with a point-source near field. It is *not* a Galerkin
   scheme, which would test with the basis functions, i.e. average the
   potential over each segment and integrate the source along each
   segment (review pass 10, 10.31).
3. Solve the bordered system

   $$
   \\begin{bmatrix} Z & -C \\\\ C^{\\top} & 0 \\end{bmatrix}
   \\begin{bmatrix} I_\\text{seg} \\\\ \\varphi_c \\end{bmatrix}
   = \\begin{bmatrix} 0 \\\\ I_\\text{in} \\end{bmatrix}
   $$

   (plus one row and column per finite conductor branch), where $C$ is
   the segment-to-node membership matrix: every segment midpoint of a
   node sits at the node potential, and the segment currents of a node
   add up to its injected current. $Z$ is symmetric; the bordered
   matrix as assembled is not, and it is solved with a dense general LU
   (:func:`numpy.linalg.solve`).
4. Every node with galvanic contact to the soil is an unknown, also
   when it carries no source: a source-free cluster floats with zero
   net current (review pass 10, 10.29), which keeps transfer impedances
   reciprocal.
5. Report the node potentials (the segment mean; point matching makes
   every midpoint of a node equipotential) and the segment currents as
   point sources, in the same :class:`FieldResult` layout as the image
   backends.

Solve flow (review pass 10, 10.75)
----------------------------------
The point-matching backends (``mom``, ``mom_sommerfeld`` and the
deprecated alias ``bem``) run through the shared solve pipeline of the
image family, :mod:`groundfield.solver._pipeline`:
:func:`~groundfield.solver._pipeline.discretise`,
:func:`~groundfield.solver._pipeline.inject` (impressed source
currents), :func:`~groundfield.solver._pipeline.build_topology`
(clusters, finite branches, distributed conductors),
:func:`~groundfield.solver._pipeline.assemble_branches` (inductive
branch block) and the burial-depth guard of the image family. Only the
solve step (one unknown current per segment instead of per electrode)
and the read-back of the node potentials are specific to the family;
they live in :func:`solve_point_matching`. Each backend supplies a
Green's-function object implementing the
:class:`~groundfield.solver._pipeline.ImageGreen` protocol plus the
reaction matrix and the segment-potential evaluation
(:class:`_PointMatchingGreen`). Up to 0.15.0 each backend carried its own
copy of the whole pipeline.

Accuracy
--------
- The point-source near field underestimates the coupling of adjacent
  segments (the exact potential of a collinear neighbour at the
  midpoint is $\\ln 3 / \\Delta s$ per unit current, the point source
  gives $1 / \\Delta s$, i.e. $-9\\,\\%$), and the self-image of a
  shallow segment is a point image. Both errors stay of the same order
  as the near-field terms under refinement, so ``mom`` does not
  converge to the thin-wire limit. On a 3 m rod (radius 8 mm,
  100 Ω·m) it lies 1.0 % ($\\Delta s$ = 0.5 m) to 1.7 %
  ($\\Delta s$ = 0.0625 m) below the exact thin-wire Galerkin solution
  at the same segment length and drifts by −0.96 % over that
  refinement ladder; the uniform-current ``image`` result is closer to
  the exact solution at every $\\Delta s$.
- ``mom`` is therefore a check of the uniform-current *ansatz*
  (current sharing along a cluster, across the layer interface, between
  passive and driven electrodes), not an accuracy reference for
  impedances at the 1 % level.
- For $\\rho_2 = \\rho_1$ the two-layer kernel collapses onto the
  homogeneous one bit-exactly; a :class:`MultiLayerSoil` is solved on
  its canonical stack (equal adjacent layers merged, 10.48) and must
  reduce to at most two layers.

Limitations
-----------
- Dense $N \\times N$ matrix and one dense LU per inductive frequency.
- $n \\ge 3$ distinct soil layers: use ``mom_sommerfeld``, which
  shares this discretisation and evaluates the layered kernel by
  quadrature.
- Point source kernel only (``Engine(kernel="point")``).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from groundfield.soil.models import HomogeneousSoil, MultiLayerSoil, TwoLayerSoil
from groundfield.solver import _pipeline
from groundfield.solver._guards import (
    _check_point_cross_layer_diagonal,
    _check_segment_depths,
)
from groundfield.solver._kernels import HalfSpaceKernel
from groundfield.solver._layered import canonical_soil
from groundfield.solver._pipeline import (
    SegmentArrays,
    Solution,
    _reject_concrete_shells,
)
from groundfield.solver._series import (
    ImageSeries,
    SeriesTruncationWarning,
    _series_tail_bound,
)
from groundfield.solver.image_2layer import _two_layer_self_kernel_factory
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from groundfield.soil.models import SoilModel
    from groundfield.solver._discretize import _Segment
    from groundfield.solver._pipeline import (
        BranchModel,
        Discretisation,
        SelfKernel,
        Topology,
    )
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_mom"]

_log = get_logger(__name__)


# ---------------------------------------------------------------------
# Z-matrix assembly
# ---------------------------------------------------------------------


def _build_Z_homogeneous(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Full N×N reaction matrix for homogeneous soil, in Ω.

    Assembled directly by :meth:`HalfSpaceKernel.matrix` (10.59):
    $\\rho/(4\\pi)$ times the point kernel $1/r + 1/r'$ off the
    diagonal and the line self-potential on it. Bit-identical to the
    historic extraction ``_self_corrected_kernel(..., eye(N), rho)``
    without its $O(N^3)$ product.

    Parameters
    ----------
    seg_points, seg_lengths, wire_radii
        Segment midpoints (m), lengths (m) and wire radii (m).
    rho
        Soil resistivity in Ω·m.
    """
    return HalfSpaceKernel(rho).matrix(seg_points, seg_lengths, wire_radii)


def _build_Z_two_layer(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    soil: TwoLayerSoil,
    max_terms: int,
    tol: float,
) -> np.ndarray:
    """Full N×N reaction matrix for 2-layer soil, in Ω.

    Assembled directly by the Tagg/Sunde reaction operator of
    :func:`groundfield.solver.image_2layer._two_layer_self_kernel_factory`
    (``matrix()``, 10.59; the cross-layer path of ADR-0007 when a
    segment reaches the interface). Bit-identical to the historic
    ``kernel(..., eye(N))`` extraction, which cost one $N \\times N$ by
    $N \\times N$ product per image term.

    Parameters
    ----------
    seg_points, seg_lengths, wire_radii
        Segment midpoints (m), lengths (m) and wire radii (m).
    soil
        Two-layer soil.
    max_terms, tol
        Tagg/Sunde truncation (term count, dimensionless tail bound).
    """
    kernel = _two_layer_self_kernel_factory(
        soil,
        max_terms,
        tol,
        allow_cross_layer=True,
    )
    return kernel.matrix(seg_points, seg_lengths, wire_radii)


# ---------------------------------------------------------------------
# Point-matching solve
# ---------------------------------------------------------------------


def _solvable_nodes(
    cluster_id: dict[str, str],
    elec_to_segidx: dict[str, list[int]],
    finite_branches: list[tuple[str, str, float]],
) -> set[str]:
    """Nodes that enter the linear system of the point-matching solve.

    A node is a galvanic cluster root, a conductor pseudo-node or an
    interior node of a distributed conductor. Nodes connected through
    finite branches form components; every node of a component that
    owns at least one leakage segment is an unknown. A component
    without any segment (no galvanic contact to the soil anywhere)
    would have an undetermined potential and is left out.

    Parameters
    ----------
    cluster_id
        Node name -> cluster root, including pseudo- and interior nodes.
    elec_to_segidx
        Node name -> indices of its leakage segments.
    finite_branches
        ``(node_a, node_b, R)`` branch list (R in Ω, unused here).

    Returns
    -------
    set[str]
        Cluster roots / node names of the unknown potentials.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nodes = set(cluster_id.values())
    for n in nodes:
        find(n)
    for a, b, _R in finite_branches:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    grounded = {find(cluster_id[name]) for name, idxs in elec_to_segidx.items() if idxs}
    return {n for n in parent if find(n) in grounded}


def _floating_clusters(
    elec_input_current: dict[str, complex],
    cluster_id: dict[str, str],
    finite_branches: list[tuple[str, str, float]],
    electrode_names: set[str],
) -> set[str]:
    """Electrode clusters with neither a source nor a finite branch.

    Their net leakage current is zero by Kirchhoff's current law; the
    reported per-electrode currents are made to sum to exactly zero (see
    :func:`_node_results`), so :meth:`FieldResult.cluster_impedance`
    returns ``nan`` for them instead of a ratio of round-off.
    """
    sourced = {cluster_id[e] for e, ic in elec_input_current.items() if ic != 0j}
    linked = {a for a, _b, _R in finite_branches} | {b for _a, b, _R in finite_branches}
    roots = {cluster_id[e] for e in electrode_names}
    return roots - sourced - linked


def _node_results(
    *,
    elec_to_segidx: dict[str, list[int]],
    phi_per_freq: list[np.ndarray],
    sc_per_freq: list[np.ndarray],
    cluster_id: dict[str, str],
    electrode_names: set[str],
    floating: set[str],
    n_freq: int,
) -> tuple[
    dict[str, list[complex]],
    dict[str, list[complex]],
    dict[str, list[complex]],
    dict[str, list[complex]],
]:
    """Per-node potentials (V) and leakage currents (A) of a solve.

    Point matching makes every segment midpoint of a node an
    equipotential, so the plain segment mean is the node potential (to
    round-off). Members of a floating cluster report currents that sum
    to exactly zero in the member order of :attr:`FieldResult.clusters`
    (sorted names): a single member reports ``0``; with several members
    the last one absorbs the round-off residual of the solve.

    Returns
    -------
    electrode_potentials, electrode_currents, conductor_potentials, conductor_currents
        Mappings ``name -> value per frequency``.
    """
    electrode_potentials: dict[str, list[complex]] = {}
    electrode_currents: dict[str, list[complex]] = {}
    conductor_potentials: dict[str, list[complex]] = {}
    conductor_currents: dict[str, list[complex]] = {}
    for ename, idxs in elec_to_segidx.items():
        if not idxs:
            continue
        u_list = [complex(np.mean(phi_per_freq[k][idxs])) for k in range(n_freq)]
        i_list = [complex(sc_per_freq[k][idxs].sum()) for k in range(n_freq)]
        if ename in electrode_names:
            electrode_potentials[ename] = u_list
            electrode_currents[ename] = i_list
        else:
            conductor_potentials[ename] = u_list
            conductor_currents[ename] = i_list
    for root in floating:
        members = sorted(
            n
            for n in electrode_currents
            if n in electrode_names and cluster_id[n] == root
        )
        if not members:
            continue
        for k in range(n_freq):
            if len(members) == 1:
                electrode_currents[members[0]][k] = 0j
                continue
            # Same left-to-right summation as FieldResult.cluster_impedance,
            # so the cluster total is s + (-s) == 0 exactly.
            rest = sum(electrode_currents[m][k] for m in members[:-1])
            electrode_currents[members[-1]][k] = -complex(rest)
    return (
        electrode_potentials,
        electrode_currents,
        conductor_potentials,
        conductor_currents,
    )


def _point_matching_solve(
    Z: np.ndarray,
    elec_input_current: dict[str, complex],
    cluster_id: dict[str, str],
    elec_to_segidx: dict[str, list[int]],
    n_segments: int,
    finite_branches: list[tuple[str, str, float]] | None = None,
    omega: float = 0.0,
    inductance_matrix: np.ndarray | None = None,
    carson_correction: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, complex]]:
    """Solve the point-matching linear system on segment level.

    In addition to the per-segment leakage currents and the per-node
    potentials, the system carries one branch current per
    finite-impedance conductor that connects two nodes. (Up to 0.15.0
    this function was called ``_galerkin_solve``; the scheme is point
    matching, see the module docstring. The old name is kept as an
    alias.)

    Mathematically the linear system is

    $$
    \\begin{bmatrix}
        Z_\\text{seg} & -C_s & 0 \\\\
        C_s^{\\top} & 0 & B^{\\top} \\\\
        0 & B & R_b
    \\end{bmatrix}
    \\begin{bmatrix}
        \\mathbf{I}_\\text{seg} \\\\
        \\boldsymbol{\\varphi}_n \\\\
        \\mathbf{I}_b
    \\end{bmatrix} =
    \\begin{bmatrix}
        \\mathbf{0} \\\\ \\mathbf{I}_\\text{in} \\\\ \\mathbf{0}
    \\end{bmatrix},
    $$

    with $C_s$ the segment-to-cluster incidence matrix and
    $B$ the branch-to-cluster incidence matrix (``+1`` at the
    branch start, ``−1`` at its end). For
    ``finite_branches in (None, [])`` the system collapses exactly to
    the bordered (N_a + K_a) × (N_a + K_a) system of the module docstring.

    Unknown set (review pass 10, 10.29): *every* node of a component
    (nodes joined by finite branches) that has galvanic contact to the
    soil, also when the component carries no source. A source-free
    component is a floating conductor system — its KCL rows have a zero
    right-hand side, so its net leakage current vanishes while its
    segment currents redistribute until its surface is an equipotential
    at the potential the field of the other clusters imposes. Because
    the segment matrix is symmetric, transfer impedances obtained from
    two solves are then reciprocal. Up to 0.15.0 source-free components
    were dropped (zero current) and their potential was read back as the
    incident field, non-reciprocal by up to 146 %.

    Returns
    -------
    seg_currents : np.ndarray, shape (N,)
        Per-segment leakage current in A (complex).
    cluster_potential : dict[cluster_root, complex]
        Common potential per node in V (also solved as part of the
        linear system).
    """
    if finite_branches is None:
        finite_branches = []

    # Aggregate input current per cluster.
    cluster_input: dict[str, complex] = {}
    for ename, ic in elec_input_current.items():
        if ic == 0j:
            continue
        cluster_input.setdefault(cluster_id[ename], 0j)
        cluster_input[cluster_id[ename]] += ic

    seg_currents = np.zeros(n_segments, dtype=complex)
    cluster_potential: dict[str, complex] = {}
    if not cluster_input:
        # No source anywhere: the (non-singular) system has the zero
        # solution.
        return seg_currents, cluster_potential

    # Unknowns (review pass 10, 10.29): every node of every connected
    # component that owns at least one leakage segment — not only the
    # components that carry a source. A component without a source is
    # a floating conductor system: zero net current, potential set by
    # the field of the others, current distribution free. Up to 0.15.0
    # such clusters were dropped and their potential read back as the
    # mean of the incident field, which broke the reciprocity of
    # transfer impedances (1.1 % to 146 %). For worlds in which every
    # component carries a source the unknown set, its ordering and
    # hence the solution are bit-identical to the historic scheme.
    active_set = _solvable_nodes(cluster_id, elec_to_segidx, finite_branches)
    if not active_set:
        return seg_currents, cluster_potential

    active_clusters = sorted(active_set)
    K_a = len(active_clusters)
    cluster_idx = {c: k for k, c in enumerate(active_clusters)}

    # Active segments: those whose owning node lies in the solvable set
    # (in practice every segment).
    active_seg_indices: list[int] = []
    seg_to_active_cluster: list[int] = []
    for ename, idxs in elec_to_segidx.items():
        c = cluster_id[ename]
        if c not in active_set:
            continue
        for s in idxs:
            active_seg_indices.append(s)
            seg_to_active_cluster.append(cluster_idx[c])
    active_seg_indices_arr = np.array(active_seg_indices, dtype=int)
    N_a = active_seg_indices_arr.size

    # Active branches: those entirely inside the active set, plus
    # their original ordinal so that an inductance matrix passed in
    # with the same ordering as ``finite_branches`` can be restricted
    # to the active subset.
    active_branch_indices: list[int] = []
    active_branches: list[tuple[str, str, float]] = []
    for idx, (a, b, R) in enumerate(finite_branches):
        if a in active_set and b in active_set:
            active_branch_indices.append(idx)
            active_branches.append((a, b, R))
    M_a = len(active_branches)

    # Reduced reaction matrix on the active segments.
    Z_active = Z[np.ix_(active_seg_indices_arr, active_seg_indices_arr)]

    # Build the augmented system A · x = b.
    n_unknowns = N_a + K_a + M_a
    A = np.zeros((n_unknowns, n_unknowns))
    # Block 1: Z_seg · I_seg − C_s · phi_n = 0
    A[:N_a, :N_a] = Z_active
    for i, k in enumerate(seg_to_active_cluster):
        A[i, N_a + k] = -1.0
    # Block 2: KCL per cluster:  C_s^T · I_seg + B^T · I_b = I_in
    for k in range(K_a):
        for i, kk in enumerate(seg_to_active_cluster):
            if kk == k:
                A[N_a + k, i] = 1.0
    for m, (a, b, _R) in enumerate(active_branches):
        ka = cluster_idx[a]
        kb = cluster_idx[b]
        A[N_a + ka, N_a + K_a + m] = +1.0
        A[N_a + kb, N_a + K_a + m] = -1.0
    # Block 3: Branch Ohm's law:  phi_a − phi_b = R · I_b
    #   ⇔  +phi_a − phi_b − R · I_b = 0
    for m, (a, b, R) in enumerate(active_branches):
        ka = cluster_idx[a]
        kb = cluster_idx[b]
        A[N_a + K_a + m, N_a + ka] = +1.0
        A[N_a + K_a + m, N_a + kb] = -1.0
        A[N_a + K_a + m, N_a + K_a + m] = -R

    # Right-hand side.
    b_re = np.zeros(n_unknowns)
    b_im = np.zeros(n_unknowns)
    for k, c in enumerate(active_clusters):
        ic = cluster_input.get(c, 0j)
        b_re[N_a + k] = ic.real
        b_im[N_a + k] = ic.imag

    use_inductive = inductance_matrix is not None and omega != 0.0 and M_a > 0

    if not use_inductive:
        # DC / no-inductive fast path: A is real; one LU factorisation
        # for both right-hand sides (multi-RHS, ADR-0010 Tier 1).
        sol = np.linalg.solve(A, np.column_stack([b_re, b_im]))
        sol_re, sol_im = sol[:, 0], sol[:, 1]
        I_active = sol_re[:N_a] + 1j * sol_im[:N_a]
        phi_active = sol_re[N_a : N_a + K_a] + 1j * sol_im[N_a : N_a + K_a]
    else:
        # Inductive coupling: the branch block carries jω·L, with
        # ``inductance_matrix`` covering the full ``finite_branches``
        # ordering; restrict to the active subset.
        n_total_branches = inductance_matrix.shape[0]
        if inductance_matrix.shape != (
            n_total_branches,
            n_total_branches,
        ) or n_total_branches != len(finite_branches):
            raise ValueError(
                f"inductance_matrix shape {inductance_matrix.shape} does "
                f"not match number of finite_branches "
                f"({len(finite_branches)})."
            )
        L_active = inductance_matrix[
            np.ix_(active_branch_indices, active_branch_indices)
        ]
        # ADR-0005 optional Carson correction.
        dZ_carson_active: np.ndarray | None = None
        if carson_correction is not None:
            if carson_correction.shape != inductance_matrix.shape:
                raise ValueError(
                    f"carson_correction shape {carson_correction.shape} "
                    f"must match inductance_matrix {inductance_matrix.shape}."
                )
            dZ_carson_active = carson_correction[
                np.ix_(active_branch_indices, active_branch_indices)
            ]
        A_c = A.astype(complex)
        for m in range(M_a):
            row = N_a + K_a + m
            for m_prime in range(M_a):
                contrib = 1j * omega * L_active[m, m_prime]
                if dZ_carson_active is not None:
                    contrib += dZ_carson_active[m, m_prime]
                if m_prime == m:
                    A_c[row, N_a + K_a + m_prime] -= contrib
                else:
                    A_c[row, N_a + K_a + m_prime] = -contrib
        b_c = b_re + 1j * b_im
        sol = np.linalg.solve(A_c, b_c)
        I_active = sol[:N_a]
        phi_active = sol[N_a : N_a + K_a]

    seg_currents[active_seg_indices_arr] = I_active
    for k, c in enumerate(active_clusters):
        cluster_potential[c] = complex(phi_active[k])

    return seg_currents, cluster_potential


#: Historic name (the scheme is point matching, not Galerkin; 10.31).
_galerkin_solve = _point_matching_solve


# ---------------------------------------------------------------------
# Shared point-matching pipeline (10.75)
# ---------------------------------------------------------------------


class _PointMatchingGreen:
    """Soil-specific part of the point-matching backends.

    Implements the :class:`~groundfield.solver._pipeline.ImageGreen`
    protocol of the shared pipeline — ``backend``, ``where``,
    ``layer_interfaces``, ``soil_resistivity``,
    ``reference_resistivity``, :meth:`prepare` and :meth:`metadata` —
    and adds what the per-segment solve needs on top: the full reaction
    matrix (:meth:`reaction_matrix`) and the segment-midpoint potentials
    of a solved current distribution (:meth:`segment_potentials`).

    Subclasses set the attributes in ``__init__`` and implement
    :meth:`prepare`.

    Attributes
    ----------
    backend
        Backend label stored in :attr:`FieldResult.backend`.
    where
        Caller label used in warning and error messages.
    soil
        Canonical soil model (10.48) stored in :attr:`FieldResult.soil`.
    layer_interfaces
        Interface depths in m at which the discretisation splits wires,
        or ``None``.
    soil_resistivity
        Upper-layer resistivity in Ω·m (:attr:`FieldResult.soil_resistivity`).
    reference_resistivity
        Resistivity in Ω·m of the reference skin depth reported for
        inductive worlds without an earth-return conductivity.
    """

    backend: str
    where: str
    soil: SoilModel
    layer_interfaces: tuple[float, ...] | None
    soil_resistivity: float
    reference_resistivity: float

    def prepare(
        self, segments: list[_Segment], arrays: SegmentArrays, *, stacklevel: int
    ) -> SelfKernel:
        """Reaction operator of the discretised geometry (``matrix()`` and call)."""
        raise NotImplementedError

    def reaction_matrix(self, kernel: SelfKernel, arrays: SegmentArrays) -> np.ndarray:
        """Full ``N × N`` reaction matrix in Ω, assembled directly (10.59)."""
        return kernel.matrix(arrays.points, arrays.lengths, arrays.radii)

    def segment_potentials(
        self,
        kernel: SelfKernel,
        Z: np.ndarray,
        arrays: SegmentArrays,
        seg_currents: np.ndarray,
    ) -> np.ndarray:
        """Segment-midpoint potentials (V) of the segment currents ``seg_currents`` (A).

        The default is the product with the assembled matrix,
        ``Z @ Re(I) + j Z @ Im(I)``.
        """
        return Z @ seg_currents.real + 1j * (Z @ seg_currents.imag)

    def metadata(self, solution: Solution, *, stacklevel: int) -> dict:
        """Backend-specific metadata (may warn about the solve)."""
        return {}


class _MomGreen(_PointMatchingGreen):
    """Green's function of ``mom``: the closed-form image-family kernels.

    Homogeneous soil uses the half-space point kernel
    (:class:`~groundfield.solver._kernels.HalfSpaceKernel`), two-layer
    soil the Tagg/Sunde series, and a geometry that reaches the layer
    interface the cross-layer kernel of ADR-0007 for every pair.

    Parameters
    ----------
    soil
        Canonical soil: :class:`HomogeneousSoil` or :class:`TwoLayerSoil`.
    max_terms, tol
        Tagg/Sunde truncation (term count, dimensionless tail bound).
    backend
        Label of the result (``"mom"``).
    """

    where = "mom"

    def __init__(
        self,
        soil: HomogeneousSoil | TwoLayerSoil,
        max_terms: int,
        tol: float,
        *,
        backend: str = "mom",
    ) -> None:
        self.backend = backend
        self.soil = soil
        self.max_terms = max_terms
        self.tol = tol
        if isinstance(soil, TwoLayerSoil):
            self.layer_interfaces = (soil.h_1,)
            self.soil_resistivity = float(soil.rho_1)
        else:
            self.layer_interfaces = None
            self.soil_resistivity = float(soil.rho)
        self.reference_resistivity = self.soil_resistivity

    def prepare(
        self, segments: list[_Segment], arrays: SegmentArrays, *, stacklevel: int
    ) -> SelfKernel:
        """Reaction operator; the cross-layer path when a segment reaches ``h_1``."""
        soil = self.soil
        if isinstance(soil, HomogeneousSoil):
            return HalfSpaceKernel(soil.rho)
        # Cross-layer support (audit 2026-07-08, WP-D3): the shared
        # self-kernel factory dispatches to the rigorous ADR-0007
        # Sommerfeld path when any segment sits at or below h_1. mom is
        # the engine that *solves* the per-segment current distribution —
        # exactly the tool to quantify the image family's uniform-leakage
        # bias across the interface.
        z_max = arrays.points[:, 2].max()
        if z_max >= soil.h_1:
            _log.info(
                "mom: cross-layer geometry (z_max = %.3f m >= h_1 = "
                "%.3f m) — using the ADR-0007 Sommerfeld kernel.",
                z_max,
                soil.h_1,
            )
            _check_point_cross_layer_diagonal(
                segments,
                arrays.points,
                arrays.lengths,
                arrays.directions,
                float(soil.h_1),
                backend=repr(self.backend),
                remedy="Use backend='image_2layer' with its default line kernel, "
                "which integrates these segments exactly, or a finer "
                "segment_length",
            )
        return _two_layer_self_kernel_factory(
            soil,
            self.max_terms,
            self.tol,
            allow_cross_layer=True,
        )

    def segment_potentials(
        self,
        kernel: SelfKernel,
        Z: np.ndarray,
        arrays: SegmentArrays,
        seg_currents: np.ndarray,
    ) -> np.ndarray:
        """Potentials from the reaction operator itself.

        The operator keeps the matrices it assembled for :meth:`reaction_matrix`
        (one assembly per solve, 10.57), so this costs two products; the
        values are bit-identical to ``mom`` of 0.15.0, which applied a
        freshly built kernel to the real and imaginary parts.
        """
        points, lengths, radii = arrays.points, arrays.lengths, arrays.radii
        return kernel(points, lengths, radii, seg_currents.real) + 1j * kernel(
            points, lengths, radii, seg_currents.imag
        )

    def metadata(self, solution: Solution, *, stacklevel: int) -> dict:
        """Soil parameters and the series convergence report (two layers)."""
        soil = self.soil
        if not isinstance(soil, TwoLayerSoil):
            return {}
        K = soil.reflection_coefficient
        n_terms_used = ImageSeries.from_coefficients(
            K, soil.h_1, self.max_terms, self.tol
        ).n_terms
        converged = (
            _series_tail_bound(abs(K), n_terms_used) < self.tol
            if n_terms_used
            else True
        )
        if not converged:
            warnings.warn(
                f"mom: the Tagg/Sunde image series was truncated at "
                f"max_terms={self.max_terms} with tail bound > "
                f"tol={self.tol:.2e} (|K| = {abs(K):.3f}). Raise "
                "Engine.image_max_terms for high layer contrasts.",
                SeriesTruncationWarning,
                stacklevel=stacklevel,
            )
        return {
            "K": float(K),
            "rho_1": float(soil.rho_1),
            "rho_2": float(soil.rho_2),
            "h_1": float(soil.h_1),
            "n_terms_used": n_terms_used,
            "converged": bool(converged),
        }


def _point_matching_step(
    engine: Engine,
    *,
    disc: Discretisation,
    node_currents: dict[str, complex],
    topo: Topology,
    branches: BranchModel,
    Z: np.ndarray,
    potentials: Callable[[np.ndarray], np.ndarray],
) -> tuple[Solution, list[dict[str, complex]]]:
    """Solve step of the point-matching family (replaces pipeline step 5).

    One unknown current per segment and one potential per node
    (:func:`_point_matching_solve`), per frequency when the branch block
    is inductive, once otherwise.

    Parameters
    ----------
    engine
        Engine; ``frequencies`` in Hz.
    disc, node_currents, topo, branches
        Results of the pipeline steps 1–4.
    Z
        Reaction matrix in Ω, ``N × N``.
    potentials
        ``seg_currents -> segment-midpoint potentials`` of the Green's
        function.

    Returns
    -------
    solution
        Segment currents (A) and potentials (V) per frequency, and the
        leakage current per owner (A).
    node_potentials
        Potential in V of every solved node per frequency (also nodes
        without leakage segments).
    """
    n_freq = len(engine.frequencies)
    n_segments = len(disc.segments)

    def _solve_at(omega: float) -> tuple[np.ndarray, np.ndarray, dict[str, complex]]:
        correction = (
            branches.earth_correction(omega)
            if branches.earth_correction is not None
            else None
        )
        sc, node_phi = _point_matching_solve(
            Z=Z,
            elec_input_current=node_currents,
            cluster_id=topo.cluster_id,
            elec_to_segidx=disc.owner_segments,
            n_segments=n_segments,
            finite_branches=topo.finite_branches,
            omega=omega if branches.has_inductance else 0.0,
            inductance_matrix=branches.inductance_matrix,
            carson_correction=correction,
        )
        ph = np.zeros(n_segments, dtype=complex)
        if sc.any():
            ph = potentials(sc)
        return sc, ph, node_phi

    if branches.has_inductance:
        sc_per_freq: list[np.ndarray] = []
        phi_per_freq: list[np.ndarray] = []
        node_phi_per_freq: list[dict[str, complex]] = []
        for f in engine.frequencies:
            sc, ph, node_phi = _solve_at(2.0 * np.pi * float(f))
            sc_per_freq.append(sc)
            phi_per_freq.append(ph)
            node_phi_per_freq.append(node_phi)
    else:
        sc, ph, node_phi = _solve_at(0.0)
        sc_per_freq = [sc] * n_freq
        phi_per_freq = [ph] * n_freq
        node_phi_per_freq = [node_phi] * n_freq
    owner_currents = [
        {
            name: complex(sc[idxs].sum())
            for name, idxs in disc.owner_segments.items()
            if idxs
        }
        for sc in sc_per_freq
    ]
    return (
        Solution(
            owner_currents=owner_currents,
            segment_currents=sc_per_freq,
            segment_potentials=phi_per_freq,
        ),
        node_phi_per_freq,
    )


def _point_matching_result(
    world: World,
    engine: Engine,
    *,
    green: _PointMatchingGreen,
    disc: Discretisation,
    node_currents: dict[str, complex],
    topo: Topology,
    branches: BranchModel,
    arrays: SegmentArrays,
    solution: Solution,
    node_potentials: list[dict[str, complex]],
    metadata: dict,
) -> FieldResult:
    """Build the result of the point-matching family (pipeline step 6).

    Same layout as :func:`groundfield.solver._pipeline.build_result`; the
    node potential is the plain segment mean (point matching enforces
    it at every midpoint), and floating clusters report an exactly zero
    net current (10.29). Junctions (leakage-free nodes, 10.38) report
    the potential of their node in the solved system in
    ``metadata["junction_potentials"]`` (V per frequency), as the image
    family does.
    """
    n_freq = len(engine.frequencies)
    real_names = [e.name for e in world.electrodes]
    real_set = set(real_names)
    (
        electrode_potentials,
        electrode_currents,
        conductor_potentials,
        conductor_currents,
    ) = _node_results(
        elec_to_segidx=disc.owner_segments,
        phi_per_freq=solution.segment_potentials,
        sc_per_freq=solution.segment_currents,
        cluster_id=topo.cluster_id,
        electrode_names=real_set,
        floating=_floating_clusters(
            node_currents, topo.cluster_id, topo.finite_branches, real_set
        ),
        n_freq=n_freq,
    )
    sc_per_freq = solution.segment_currents
    point_sources = [
        PointSource(
            position=tuple(arrays.points[i].tolist()),
            current=[complex(sc_per_freq[k][i]) for k in range(n_freq)],
            electrode_name=disc.segments[i].electrode_name,
            length=float(arrays.lengths[i]),
        )
        for i in range(len(disc.segments))
    ]
    cluster_id = topo.cluster_id
    clusters = {
        ename: sorted(
            n
            for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_set
        )
        for ename in real_names
    }
    metadata = dict(metadata)
    if branches.has_inductance:
        from groundfield.coupling.carson import skin_depth

        sigma = (
            branches.sigma_earth
            if branches.sigma_earth is not None
            else 1.0 / float(green.reference_resistivity)
        )
        metadata["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma) for f in engine.frequencies
        }
    if conductor_currents:
        metadata["conductor_node_currents"] = conductor_currents
        metadata["conductor_node_potentials"] = conductor_potentials
    # Leakage-free junction nodes (10.38): their potentials follow from the
    # conductor network (KCL) and are reported like the image family's.
    junctions = world.junctions
    if junctions:
        metadata["junction_potentials"] = {
            j.name: [
                complex(node_potentials[k].get(cluster_id[j.name], complex("nan")))
                for k in range(n_freq)
            ]
            for j in junctions
        }
    return FieldResult(
        backend=green.backend,
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=float(green.soil_resistivity),
        soil=green.soil,
        clusters=clusters,
        metadata=metadata,
    )


def solve_point_matching(
    world: World,
    engine: Engine,
    green: _PointMatchingGreen,
    *,
    stacklevel: int = 3,
) -> FieldResult:
    """Run the point-matching family through the shared solve pipeline.

    Steps 1–4 of :mod:`groundfield.solver._pipeline` (discretisation,
    source injection, topology, inductive branch block) and the
    burial-depth guard are those of the image family; the solve and the
    read-back are :func:`_point_matching_step` and
    :func:`_point_matching_result` (review pass 10, 10.75).

    Parameters
    ----------
    world
        World with at least one electrode and a soil ``green`` accepts.
    engine
        Engine configuration (``segment_length`` in m, ``frequencies``
        in Hz, ``earth_inductive_model``).
    green
        Soil-specific part (:class:`_PointMatchingGreen`).
    stacklevel
        ``stacklevel`` that makes the warnings point at the caller of
        the backend entry point (``solve_mom`` / ``solve_mom_sommerfeld``).

    Returns
    -------
    FieldResult
        Node potentials (V) and leakage currents (A) per frequency, the
        segment currents as point sources, and metadata.
    """
    # Steps 1-4 of the shared pipeline, called through the module so that
    # every extension of a step (two-terminal sources and junction nodes
    # in ``inject`` / ``build_topology``, inductive fixes in
    # ``assemble_branches``) reaches this family unchanged.
    disc = _pipeline.discretise(world, engine, green.layer_interfaces)
    node_currents = _pipeline.inject(world, where=green.where, stacklevel=stacklevel)
    topo = _pipeline.build_topology(world, disc)
    branches = _pipeline.assemble_branches(world, engine, topo)
    # Burial-depth validation of the galvanic image construction, as in
    # the image family (F26/F29): segments partly above the surface are
    # rejected, shallow ones warn (ShallowSegmentWarning).
    _check_segment_depths(disc.segments, where=green.where, stacklevel=stacklevel + 1)
    arrays = SegmentArrays.from_segments(disc.segments)
    kernel = green.prepare(disc.segments, arrays, stacklevel=stacklevel + 1)
    Z = green.reaction_matrix(kernel, arrays)
    solution, node_potentials = _point_matching_step(
        engine,
        disc=disc,
        node_currents=node_currents,
        topo=topo,
        branches=branches,
        Z=Z,
        potentials=lambda sc: green.segment_potentials(kernel, Z, arrays, sc),
    )
    metadata = {
        "world_name": world.name,
        "n_segments": len(disc.segments),
        "segment_length": engine.segment_length,
        "solver": "point_matching",
        "stub": False,
        "earth_inductive_model": branches.earth_inductive_model,
        **green.metadata(solution, stacklevel=stacklevel + 1),
    }
    return _point_matching_result(
        world,
        engine,
        green=green,
        disc=disc,
        node_currents=node_currents,
        topo=topo,
        branches=branches,
        arrays=arrays,
        solution=solution,
        node_potentials=node_potentials,
        metadata=metadata,
    )


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_mom(
    world: World,
    engine: Engine,
    *,
    two_layer_max_terms: int | None = None,
    two_layer_tol: float | None = None,
) -> FieldResult:
    """Point-matching method-of-moments backend.

    Supports :class:`HomogeneousSoil`, :class:`TwoLayerSoil` and a
    :class:`MultiLayerSoil` whose canonical stack has at most two layers
    (10.48). The Green's-function kernel is taken from the matching
    image backend; only the resolution scheme (one current per segment,
    point matching at the segment midpoints, instead of one uniform
    current per electrode) differs. Clusters without a source float
    (zero net current, 10.29). See the module docstring for the
    accuracy of the scheme and the shared solve flow.

    Parameters
    ----------
    world
        World to evaluate.
    engine
        Engine configuration; ``engine.segment_length`` (m) controls the
        discretisation.
    two_layer_max_terms, two_layer_tol
        Truncation parameters for the Tagg/Sunde series (term count,
        dimensionless tail bound), only used for two-layer soil.
        ``None`` (default) takes ``engine.image_max_terms`` /
        ``engine.image_series_tol``, so the engine knob reaches this
        backend consistently with ``image_2layer``.

    Returns
    -------
    FieldResult
        Result object compatible with :class:`FieldResult` from the
        image backends. ``metadata['solver']`` is ``'point_matching'``
        (``'galerkin'`` up to 0.15.0, although the scheme was the
        same).

    Raises
    ------
    ValueError
        If the world has no electrodes, or a segment lies partly above
        the soil surface.
    BackendCapabilityError
        For a :class:`MultiLayerSoil` with three or more distinct layers
        (a :class:`TypeError` before 2026.11), or for electrodes with a
        concrete-shell coefficient (ADR-0012).
    TypeError
        If ``world.soil`` is no soil model.
    """
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'mom' supports HomogeneousSoil, TwoLayerSoil and "
            "MultiLayerSoil stacks with at most two distinct layers. "
            f"Got: {type(world.soil).__name__}."
        )
    # Review pass 10, 10.48: solve the canonical soil — a MultiLayerSoil
    # whose adjacent layers share a resistivity is a two-layer (or
    # homogeneous) soil, and a TwoLayerSoil without contrast is
    # homogeneous.
    soil = canonical_soil(world.soil)
    if isinstance(soil, MultiLayerSoil):
        from groundfield.solver.registry import BackendCapabilityError

        raise BackendCapabilityError(
            "Backend 'mom' supports at most two distinct soil layers; this "
            f"MultiLayerSoil has {len(soil.layers)}. Use "
            "backend='mom_sommerfeld' for n >= 3 soils."
        )
    _reject_concrete_shells(world, "mom")
    if two_layer_max_terms is None:
        two_layer_max_terms = engine.image_max_terms
    if two_layer_tol is None:
        two_layer_tol = engine.image_series_tol
    _log.info(
        "mom: soil=%s, segment_length=%.3f",
        type(soil).__name__,
        engine.segment_length,
    )
    green = _MomGreen(soil, two_layer_max_terms, two_layer_tol)
    return solve_point_matching(world, engine, green, stacklevel=3)
