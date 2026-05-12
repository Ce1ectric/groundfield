"""Method-of-Moments backend (``mom``).

Independent second engine used to cross-validate the image backends
(see ADR-0001). The ``image`` backends approximate the current
distribution along an electrode by a *uniform per unit length*
profile; this backend solves for the actual non-uniform per-segment
current distribution through a Galerkin scheme.

Method
------
1. Discretise every electrode into wire segments (same routine as the
   image backends).
2. Assemble the full $N \\times N$ segment-level reaction matrix
   $Z$ using the same Green's-function kernel as the
   corresponding image backend (so for a ``HomogeneousSoil`` world
   ``mom`` shares its physics with ``image``; for a ``TwoLayerSoil``
   world it shares it with ``image_2layer``). The diagonal of
   $Z$ carries the analytical line-self-potential.
3. Solve the linear system
   $$
   \\begin{bmatrix} Z & -C \\\\ C^{\\top} & 0 \\end{bmatrix}
   \\begin{bmatrix} I_\\text{seg} \\\\ \\varphi_c \\end{bmatrix}
   = \\begin{bmatrix} 0 \\\\ I_\\text{in} \\end{bmatrix},
   $$
   where $C$ is the segment-to-cluster membership matrix.

   The constraint enforces equal segment potential within each
   galvanic cluster; the additional row sums the segment currents to
   the input current per cluster. The system is symmetric and dense;
   :func:`numpy.linalg.solve` handles it directly.

4. Compute the average cluster potential and produce a
   :class:`FieldResult` exactly like the image backends, so that
   :func:`groundfield.compare_engines` can compare the two side by
   side.

What this means in practice
---------------------------
- For homogeneous soil, ``mom`` agrees with ``image`` to within a few
  per cent and lies closer to the Sunde reference (the average-
  potential method of ``image`` carries a residual ~5 % bias from
  the uniform-current assumption, which the Galerkin solve removes).
- For 2-layer soil, ``mom`` and ``image_2layer`` agree within the
  same envelope. In the limit $\\rho_2 = \\rho_1$ both engines
  collapse onto the homogeneous solution.

Limitations
-----------
- Performance is $O(N^3)$ for the LU solve plus $O(N^2)$
  for the matrix build. For AP1 geometries ``N`` stays well below
  1 000, so the runtime is acceptable. Larger meshes will need a
  preconditioned iterative solver later.
- This backend deliberately re-uses the existing Green's-function
  kernels rather than introducing a Sommerfeld quadrature. A truly
  independent layered Green's function (option B in ADR-0001) would
  give a stronger cross-check; this lighter version is the
  pragmatic first step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from groundfield.soil.models import HomogeneousSoil, TwoLayerSoil
from groundfield.solver.image import (
    _assemble_inductance_matrix,
    _build_clusters,
    _build_distributed_topology,
    _build_finite_branches,
    _discretize_electrode,
    _Segment,
    _self_corrected_kernel,
)
from groundfield.solver.image_2layer import _two_layer_self_kernel_factory
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
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
    """Full N×N reaction matrix for homogeneous soil.

    Built by feeding the identity matrix as the "currents" argument
    of :func:`_self_corrected_kernel`: the resulting ``phi`` matrix is
    exactly $Z$ (with the $\\rho/(4\\pi)$ prefactor and
    the line-self-potential on the diagonal).
    """
    n = seg_points.shape[0]
    eye = np.eye(n)
    return _self_corrected_kernel(seg_points, seg_lengths, wire_radii, eye, rho)


def _build_Z_two_layer(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    soil: TwoLayerSoil,
    max_terms: int,
    tol: float,
) -> np.ndarray:
    """Full N×N reaction matrix for 2-layer soil.

    Uses the Tagg/Sunde self-kernel closure produced by
    :func:`groundfield.solver.image_2layer._two_layer_self_kernel_factory`.
    """
    n = seg_points.shape[0]
    self_kernel = _two_layer_self_kernel_factory(soil, max_terms, tol)
    eye = np.eye(n)
    return self_kernel(seg_points, seg_lengths, wire_radii, eye)


# ---------------------------------------------------------------------
# Galerkin solve
# ---------------------------------------------------------------------


def _galerkin_solve(
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
    """Solve the Galerkin linear system on segment level.

    Augmented variant of the historic Galerkin scheme: in addition to
    the per-segment leakage currents and the per-cluster potentials,
    the system optionally carries one branch current per
    finite-impedance conductor that connects two clusters.

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
    the previous (N_a + K_a) × (N_a + K_a) Galerkin system.

    Returns
    -------
    seg_currents : np.ndarray, shape (N,)
        Per-segment current distribution (complex).
    cluster_potential : dict[cluster_root, complex]
        Common potential per cluster (also solved as part of the
        linear system).
    """
    if finite_branches is None:
        finite_branches = []

    # Aggregate input current per cluster, identify the active ones.
    cluster_input: dict[str, complex] = {}
    for ename, ic in elec_input_current.items():
        if ic == 0j:
            continue
        cluster_input.setdefault(cluster_id[ename], 0j)
        cluster_input[cluster_id[ename]] += ic

    seg_currents = np.zeros(n_segments, dtype=complex)
    cluster_potential: dict[str, complex] = {}
    if not cluster_input and not finite_branches:
        return seg_currents, cluster_potential

    # Active set: source clusters plus every cluster transitively
    # reachable through a finite branch.
    active_set: set[str] = set(cluster_input.keys())
    if finite_branches:
        changed = True
        while changed:
            changed = False
            for a, b, _R in finite_branches:
                if a in active_set and b not in active_set:
                    active_set.add(b)
                    changed = True
                elif b in active_set and a not in active_set:
                    active_set.add(a)
                    changed = True

    if not active_set:
        return seg_currents, cluster_potential

    active_clusters = sorted(active_set)
    K_a = len(active_clusters)
    cluster_idx = {c: k for k, c in enumerate(active_clusters)}

    # Active segments: those whose owning electrode lies in an active
    # cluster. Passive (no input current and no branch link) clusters
    # are excluded from the unknowns.
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

    use_inductive = (
        inductance_matrix is not None
        and omega != 0.0
        and M_a > 0
    )

    if not use_inductive:
        # DC / no-inductive fast path: A is real, solve real and imag
        # right-hand sides independently.
        sol_re = np.linalg.solve(A, b_re)
        sol_im = np.linalg.solve(A, b_im)
        I_active = sol_re[:N_a] + 1j * sol_im[:N_a]
        phi_active = sol_re[N_a:N_a + K_a] + 1j * sol_im[N_a:N_a + K_a]
    else:
        # Inductive coupling: the branch block carries jω·L, with
        # ``inductance_matrix`` covering the full ``finite_branches``
        # ordering; restrict to the active subset.
        n_total_branches = inductance_matrix.shape[0]
        if (
            inductance_matrix.shape != (n_total_branches, n_total_branches)
            or n_total_branches != len(finite_branches)
        ):
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
        phi_active = sol[N_a:N_a + K_a]

    seg_currents[active_seg_indices_arr] = I_active
    for k, c in enumerate(active_clusters):
        cluster_potential[c] = complex(phi_active[k])

    return seg_currents, cluster_potential


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_mom(
    world: "World",
    engine: "Engine",
    *,
    two_layer_max_terms: int = 100,
    two_layer_tol: float = 1e-6,
) -> FieldResult:
    """Galerkin Method-of-Moments backend.

    Supports :class:`HomogeneousSoil` and :class:`TwoLayerSoil`. The
    Green's-function kernel is taken from the matching image backend;
    only the resolution scheme (per-segment Galerkin instead of
    per-electrode average potential) differs.

    Parameters
    ----------
    world
        World to evaluate.
    engine
        Engine configuration; ``engine.segment_length`` controls the
        discretisation.
    two_layer_max_terms, two_layer_tol
        Truncation parameters for the Tagg/Sunde series, only used
        when ``world.soil`` is a :class:`TwoLayerSoil`.

    Returns
    -------
    FieldResult
        Result object compatible with :class:`FieldResult` from the
        image backends. ``metadata['solver']`` is set to ``'galerkin'``.
    """
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil)):
        raise TypeError(
            "Backend 'mom' currently supports HomogeneousSoil and "
            f"TwoLayerSoil only. Got: {type(world.soil).__name__}."
        )

    ds = engine.segment_length
    _log.info(
        "mom: soil=%s, segment_length=%.3f",
        type(world.soil).__name__, ds,
    )

    # 1) Discretisation — identical to the image backends.
    all_segments: list[_Segment] = []
    elec_to_segidx: dict[str, list[int]] = {}
    for e in world.electrodes:
        segs = _discretize_electrode(e, ds)
        elec_to_segidx[e.name] = list(range(len(all_segments),
                                            len(all_segments) + len(segs)))
        all_segments.extend(segs)

    # 2) Per-electrode input currents from the configured sources.
    elec_input_current: dict[str, complex] = {
        e.name: 0j for e in world.electrodes
    }
    for src in world.sources:
        if src.kind != "current":
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in elec_input_current:
            elec_input_current[src.attached_to] += i_complex

    # 3) Cluster building (ideal conductors only), finite-impedance
    #    branch list, plus the distributed-conductor topology
    #    (ADR-0003) — all fed into the augmented Galerkin system.
    cluster_id = _build_clusters(world.electrodes, world.conductors)
    finite_branches = _build_finite_branches(world.conductors, cluster_id)
    cond_segs, distributed_branches_objs, interior_nodes = _build_distributed_topology(
        world.conductors, cluster_id
    )
    for s in cond_segs:
        pn = s.electrode_name
        elec_to_segidx[pn] = [len(all_segments)]
        all_segments.append(s)
        cluster_id[pn] = pn
    for n_ in interior_nodes:
        if n_ not in cluster_id:
            cluster_id[n_] = n_
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

    n_segments = len(all_segments)
    seg_points = np.array([s.midpoint for s in all_segments])
    seg_lengths = np.array([s.length for s in all_segments])
    wire_radii = np.array([s.wire_radius for s in all_segments])

    # 4) Validate 2-layer precondition (after all segments collected).
    if isinstance(world.soil, TwoLayerSoil):
        z_max = seg_points[:, 2].max()
        if z_max >= world.soil.h_1:
            raise ValueError(
                "mom: a segment lies below the layer interface "
                f"(z_max = {z_max:.3f} m, h_1 = {world.soil.h_1:.3f} m). "
                "All electrodes must sit in the upper layer."
            )

    # 5) Assemble the reaction matrix Z (kernel depends on soil model).
    n_terms_used = 0
    if isinstance(world.soil, HomogeneousSoil):
        Z = _build_Z_homogeneous(
            seg_points, seg_lengths, wire_radii, world.soil.resistivity
        )
    else:
        # TwoLayerSoil
        Z = _build_Z_two_layer(
            seg_points, seg_lengths, wire_radii,
            world.soil, two_layer_max_terms, two_layer_tol,
        )
        # Approximate the term count from the same offset generator.
        from groundfield.solver.image_2layer import _two_layer_image_offsets

        _, _, _, n_terms_used = _two_layer_image_offsets(
            world.soil.reflection_coefficient, world.soil.h_1,
            two_layer_max_terms, two_layer_tol,
        )

    # 6) Frequency loop. With no inductive coupling, solve once and
    #    replicate across frequencies; otherwise loop.
    n_freq = len(engine.frequencies)
    omegas = [2.0 * np.pi * float(f) for f in engine.frequencies]
    real_electrode_names = {e.name for e in world.electrodes}

    if isinstance(world.soil, HomogeneousSoil):
        rho_self = world.soil.resistivity
        phi_kernel = lambda sc_real: _self_corrected_kernel(
            seg_points, seg_lengths, wire_radii, sc_real, rho_self,
        )
    else:
        self_kernel = _two_layer_self_kernel_factory(
            world.soil, two_layer_max_terms, two_layer_tol,
        )
        phi_kernel = lambda sc_real: self_kernel(
            seg_points, seg_lengths, wire_radii, sc_real,
        )

    def _solve_at(omega: float) -> tuple[np.ndarray, np.ndarray]:
        carson_dz = (
            carson_builder(omega) if (has_inductance and carson_builder is not None)
            else None
        )
        sc, _ = _galerkin_solve(
            Z=Z,
            elec_input_current=elec_input_current,
            cluster_id=cluster_id,
            elec_to_segidx=elec_to_segidx,
            n_segments=n_segments,
            finite_branches=finite_branches,
            omega=omega if has_inductance else 0.0,
            inductance_matrix=inductance_matrix_full if has_inductance else None,
            carson_correction=carson_dz,
        )
        ph = np.zeros(n_segments, dtype=complex)
        if sc.any():
            ph = phi_kernel(sc.real) + 1j * phi_kernel(sc.imag)
        return sc, ph

    sc_per_freq: list[np.ndarray] = []
    phi_per_freq: list[np.ndarray] = []
    if has_inductance:
        for omega in omegas:
            sc, ph = _solve_at(omega)
            sc_per_freq.append(sc)
            phi_per_freq.append(ph)
    else:
        sc, ph = _solve_at(0.0)
        sc_per_freq = [sc] * n_freq
        phi_per_freq = [ph] * n_freq

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
        i_list = [
            complex(sc_per_freq[k][idxs].sum()) for k in range(n_freq)
        ]
        if ename in real_electrode_names:
            electrode_potentials[ename] = u_list
            electrode_currents[ename] = i_list
        else:
            conductor_potentials[ename] = u_list
            conductor_currents[ename] = i_list

    # 8) Point-source list and cluster map for post-processing.
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

    # 9) Soil-specific metadata.
    if isinstance(world.soil, HomogeneousSoil):
        soil_resistivity = float(world.soil.resistivity)
        soil_meta: dict[str, float | int | bool] = {}
    else:
        K = world.soil.reflection_coefficient
        soil_resistivity = float(world.soil.rho_1)
        soil_meta = {
            "K": float(K),
            "rho_1": float(world.soil.rho_1),
            "rho_2": float(world.soil.rho_2),
            "h_1": float(world.soil.h_1),
            "n_terms_used": n_terms_used,
            "converged": bool(abs(K) ** n_terms_used < two_layer_tol)
            if n_terms_used else True,
        }

    return FieldResult(
        backend="mom",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=soil_resistivity,
        soil=world.soil,
        clusters=cluster_members,
        metadata={
            "world_name": world.name,
            "n_segments": n_segments,
            "segment_length": ds,
            "solver": "galerkin",
            "stub": False,
            "earth_inductive_model": earth_inductive_model,
            **soil_meta,
            **_carson_skin_metadata(
                has_inductance, sigma_earth_for_carson, world.soil,
                engine.frequencies,
            ),
            **(
                {
                    "conductor_node_currents": conductor_currents,
                    "conductor_node_potentials": conductor_potentials,
                }
                if conductor_currents else {}
            ),
        },
    )


def _carson_skin_metadata(
    has_inductance: bool,
    sigma_for_carson: float | None,
    soil,
    frequencies,
) -> dict:
    """Build the ``penetration_depth`` metadata block."""
    if not has_inductance:
        return {}
    from groundfield.coupling.carson import skin_depth
    from groundfield.soil.models import HomogeneousSoil, TwoLayerSoil

    if sigma_for_carson is not None:
        sigma_ref = sigma_for_carson
    elif isinstance(soil, HomogeneousSoil):
        sigma_ref = 1.0 / float(soil.resistivity)
    elif isinstance(soil, TwoLayerSoil):
        sigma_ref = 1.0 / float(soil.rho_1)
    else:
        return {}
    return {
        "penetration_depth": {
            float(f): skin_depth(2.0 * np.pi * f, sigma_ref)
            for f in frequencies
        }
    }
