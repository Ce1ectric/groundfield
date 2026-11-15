"""Network topology and the augmented nodal system (private).

Clusters of ideally bonded electrodes, finite-impedance and
distributed-conductor branches, and the solve of the augmented nodal
system that couples the multi-port grounding matrix with the branch
network (:func:`_solve_cluster_currents`).

Moved out of :mod:`groundfield.solver.image` (review pass 10, 10.75 /
10.79); every name stays importable from there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from groundfield.solver._discretize import (
    _discretize_conductor,
    _DistributedBranch,
    _is_conductor_chain,
    _Segment,
)

__all__: list[str] = []


def _build_distributed_topology(
    conductors,
    cluster_id: dict[str, str],
) -> tuple[list[_Segment], list[_DistributedBranch], set[str]]:
    """Aggregate the per-conductor discretisations into one set.

    Parameters
    ----------
    conductors
        Iterable of :class:`Conductor` instances.
    cluster_id
        Cluster-root mapping from :func:`_build_clusters`. End-cluster
        identifiers in the resulting branch list are translated to
        their cluster roots so that ideal-galvanic shortcuts of the
        end electrodes are honoured automatically.

    Returns
    -------
    seg_list : list[_Segment]
        Conductor pseudo-electrode segments (galvanic case only).
    branches : list[_DistributedBranch]
        Longitudinal branches with cluster-root endpoints, plus the
        geometric data needed for the inductive-coupling assembly
        (per-branch endpoints in 3-D and wire radius). The
        topology-only triple ``(node_a, node_b, R)`` is recovered
        as ``(b.node_a, b.node_b, b.R)``.
    interior_nodes : set[str]
        Pseudo-node identifiers introduced by the distributed
        discretisation (midpoint nodes + isolated interior nodes).
        Used by the solver to know which nodes are *not* derived
        from real electrodes.
    """
    seg_list: list[_Segment] = []
    branches: list[_DistributedBranch] = []
    interior_nodes: set[str] = set()
    for c in conductors:
        if not getattr(c, "is_distributed", False):
            continue
        if c.start_electrode is None or c.end_electrode is None:
            # Floating endpoints — defer until the measurement-lead
            # work; for now the discretiser would still work, but the
            # solver needs an extra cluster row, which we leave out
            # of the first distributed-conductor release.
            raise ValueError(
                f"Distributed conductor '{c.name}' must have both "
                "start_electrode and end_electrode set; floating "
                "endpoints are not yet supported."
            )
        segs, brs = _discretize_conductor(c)
        for s in segs:
            seg_list.append(s)
            interior_nodes.add(s.electrode_name)
        # Translate end-cluster identifiers via cluster_id; interior
        # midpoint / isolated-node identifiers are passed through
        # unchanged.
        for db in brs:
            ra = cluster_id.get(db.node_a, db.node_a)
            rb = cluster_id.get(db.node_b, db.node_b)
            branches.append(
                _DistributedBranch(
                    node_a=ra,
                    node_b=rb,
                    R=db.R,
                    p_a=db.p_a,
                    p_b=db.p_b,
                    wire_radius=db.wire_radius,
                    inductive=db.inductive,
                    shield=db.shield,
                    conductor=db.conductor,
                )
            )
            if db.node_a not in cluster_id:
                interior_nodes.add(db.node_a)
            if db.node_b not in cluster_id:
                interior_nodes.add(db.node_b)
    return seg_list, branches, interior_nodes


# ---------------------------------------------------------------------
# Cluster building and current sharing
# ---------------------------------------------------------------------


def _build_clusters(electrodes, conductors, junctions=()) -> dict[str, str]:
    """Union-find on the nodes: returns the cluster root per name.

    The nodes are the electrodes and the leakage-free junctions
    (``junctions``, default none). Only **ideal** conductors
    (``Conductor.is_ideal()`` returns ``True``, i.e. ``cross_section
    is None`` or the resulting series resistance is below the
    ideal-resistance threshold) are treated as galvanic shorts and
    merge their two end nodes into one cluster.

    Finite-impedance conductors are *not* used for clustering — they
    enter the solver later as branches in the nodal-analysis system
    (see :func:`_solve_cluster_currents` and the module docstring).

    Conductors with purely geometric end-points (no
    ``start_electrode`` / ``end_electrode``) are ignored.
    """
    parent = {e.name: e.name for e in electrodes}
    for j in junctions:
        parent[j.name] = j.name

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for c in conductors:
        a = getattr(c, "start_electrode", None)
        b = getattr(c, "end_electrode", None)
        if a is None or b is None:
            continue
        if a not in parent or b not in parent:
            continue
        # Only ideal conductors fuse clusters; finite-impedance
        # branches stay separate so that the nodal-analysis solver
        # sees them as proper branches.
        is_ideal = getattr(c, "is_ideal", None)
        if callable(is_ideal) and not is_ideal():
            continue
        union(a, b)

    return {n: find(n) for n in parent}


def _check_soil_contact(
    junction_names: list[str],
    cluster_id: dict[str, str],
    finite_branches: list[tuple[str, str, float]],
    owner_segments: dict[str, list[int]],
) -> None:
    """Reject junctions whose subnetwork has no soil contact.

    Nodes joined by finite branches form components; a component has
    soil contact if one of its nodes owns a leakage segment (an
    electrode or a galvanic conductor segment). Without junctions every
    component contains an electrode, so only junctions can end up in a
    component without soil contact — a floating subnetwork whose
    potential is undefined (the augmented system would be singular).

    Raises
    ------
    ValueError
        Naming the junctions without soil contact.
    """
    if not junction_names:
        return
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _R in finite_branches:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    grounded = {find(cluster_id[name]) for name, idxs in owner_segments.items() if idxs}
    floating = [j for j in junction_names if find(cluster_id[j]) not in grounded]
    if floating:
        raise ValueError(
            f"Junction(s) {floating} have no soil contact: no conductor path "
            "connects them to an electrode or to a galvanically coupled "
            "conductor, so their potential is undefined (a floating "
            "subnetwork). Connect them to the grounded network or remove "
            "them."
        )


def _build_finite_branches(
    conductors,
    cluster_id: dict[str, str],
    *,
    distributed_as_lumped: bool = False,
) -> list[tuple[str, str, float]]:
    """Collect lumped finite-impedance branches as ``(node_a, node_b, R)``.

    Each finite, non-distributed conductor between two electrodes
    whose cluster roots differ becomes one branch in the
    nodal-analysis system. The branch resistance is
    :attr:`Conductor.series_resistance`.

    Conductors that are skipped here:

    - **Ideal** conductors — already used by :func:`_build_clusters`
      to merge clusters.
    - **Distributed** conductors (``is_distributed == True``) —
      normally handled by :func:`_build_distributed_topology`, which
      produces a per-segment branch chain. Backends that cannot
      consume the distributed topology (currently the FEM
      equivalent-hemisphere backend) can pass
      ``distributed_as_lumped=True`` to fall back to a single
      lumped branch with the conductor's full series resistance —
      the discretisation is then *ignored*.
    - Conductors whose two end electrodes already share a cluster
      (an ideal short loops over them) — the finite branch would be
      shorted out and carry no net current.
    """
    branches: list[tuple[str, str, float]] = []
    for c in conductors:
        a = getattr(c, "start_electrode", None)
        b = getattr(c, "end_electrode", None)
        if a is None or b is None:
            continue
        if a not in cluster_id or b not in cluster_id:
            continue
        is_ideal = getattr(c, "is_ideal", None)
        if not callable(is_ideal) or is_ideal():
            continue
        if _is_conductor_chain(c) and not distributed_as_lumped:
            # Routed through _build_distributed_topology instead. An
            # isolated distributed conductor that degenerates to a single
            # sub-piece (n_segments == 1, i.e. discretize_segment_length >=
            # conductor length) is no chain — _discretize_conductor returns
            # an empty topology for it — so it falls through to the lumped
            # branch here (it used to vanish: open circuit). A galvanic one
            # is a chain with one leaking midpoint (10.24).
            continue
        node_a = cluster_id[a]
        node_b = cluster_id[b]
        if node_a == node_b:
            # Both endpoints already share an ideal cluster — the
            # finite branch is shorted by the ideal connection and
            # carries no net current. Skip it.
            continue
        R = float(getattr(c, "series_resistance", 0.0))
        if R <= 0.0:
            continue
        branches.append((node_a, node_b, R))
    return branches


@dataclass
class _ActiveSystem:
    """Active part of the augmented nodal system of one solve.

    Attributes
    ----------
    cluster_input
        Impressed current per cluster root in A (non-zero entries only).
    active_clusters
        Sorted cluster roots that take part in the system (source nodes
        plus every node reachable from one through finite branches).
    cluster_idx
        Position of each active cluster root in ``active_clusters``.
    active_elecs
        Owners (electrodes, then conductor pseudo-nodes) with leakage
        segments in an active cluster — the rows/columns of the
        multi-port matrix ``Z``.
    active_branch_indices
        Positions of the active branches in the full ``finite_branches``
        list (used to restrict the inductance matrix).
    active_branches
        Active ``(node_a, node_b, R)`` branch triples.
    """

    cluster_input: dict[str, complex]
    active_clusters: list[str]
    cluster_idx: dict[str, int]
    active_elecs: list[str]
    active_branch_indices: list[int]
    active_branches: list[tuple[str, str, float]]

    @property
    def n_owners(self) -> int:
        """Number of active owners ``N_a`` (multi-port size)."""
        return len(self.active_elecs)

    @property
    def n_nodes(self) -> int:
        """Number of active nodes ``K_a``."""
        return len(self.active_clusters)

    @property
    def n_branches(self) -> int:
        """Number of active finite branches ``M_a``."""
        return len(self.active_branches)


def _cluster_input_currents(
    elec_input_current: dict[str, complex], cluster_id: dict[str, str]
) -> dict[str, complex]:
    """Sum the impressed currents per cluster root (non-zero only)."""
    cluster_input: dict[str, complex] = {}
    for ename, ic in elec_input_current.items():
        if ic == 0j:
            continue
        cluster_input.setdefault(cluster_id[ename], 0j)
        cluster_input[cluster_id[ename]] += ic
    return cluster_input


def _active_cluster_set(
    cluster_input: dict[str, complex],
    finite_branches: list[tuple[str, str, float]],
) -> set[str]:
    """Source clusters plus every cluster reachable through finite branches."""
    active_clusters_set: set[str] = set(cluster_input.keys())
    if finite_branches:
        # Union-find over branch endpoints to propagate activity.
        parent: dict[str, str] = {}

        def find_b(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union_b(a: str, b: str) -> None:
            parent[find_b(a)] = find_b(b)

        for a, b, _R in finite_branches:
            union_b(a, b)
        active_roots = {find_b(c) for c in active_clusters_set}
        for a, b, _R in finite_branches:
            if find_b(a) in active_roots or find_b(b) in active_roots:
                active_clusters_set.add(a)
                active_clusters_set.add(b)
    return active_clusters_set


def _floating_cluster_set(
    cluster_input: dict[str, complex],
    finite_branches: list[tuple[str, str, float]],
    cluster_id: dict[str, str],
    owner_roots: list[str],
) -> set[str]:
    """Active nodes when source-free clusters float (finding 10.29).

    Nodes (cluster roots) joined by finite branches form components. A
    component enters the system if it is driven (carries an impressed
    current) or if it is a *non-trivial* passive component with soil
    contact: several leakage owners (a bonded group) or at least one
    branch (a passive network). Such a component floats — its net
    leakage is zero by Kirchhoff's current law, but its members exchange
    current to stay equipotential and its branches carry their
    equalising currents.

    A passive component of a single owner without branches is left out:
    under the uniform-current ansatz its one owner carries zero current
    whether it is solved or not, and its potential is the incident field
    either way. Leaving it out keeps the results of every world without
    bonded passive groups or passive networks bit-identical.

    Components without any leakage owner (no soil contact) are rejected
    earlier, by the topology step.

    Parameters
    ----------
    cluster_input
        Impressed current per cluster root in A (non-zero entries only).
    finite_branches
        ``(node_a, node_b, R)`` per finite branch, endpoints are cluster
        roots.
    cluster_id
        Cluster root per node name.
    owner_roots
        Cluster root of every owner with leakage segments (one entry per
        owner).

    Returns
    -------
    set[str]
        Cluster roots of the active nodes.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for root in set(cluster_id.values()):
        find(root)
    for a, b, _R in finite_branches:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    n_owners: dict[str, int] = {}
    for root in owner_roots:
        comp = find(root)
        n_owners[comp] = n_owners.get(comp, 0) + 1
    with_branch = {find(a) for a, _b, _R in finite_branches}
    driven = {find(c) for c in cluster_input}
    active_components = {
        comp
        for comp in set(n_owners) | driven
        if comp in driven or n_owners.get(comp, 0) > 1 or comp in with_branch
    }
    return {node for node in parent if find(node) in active_components}


def _passive_isolated_roots(
    system: _ActiveSystem, finite_branches: list[tuple[str, str, float]]
) -> set[str]:
    """Active cluster roots with neither an impressed current nor a branch.

    Their net leakage is zero by Kirchhoff's current law (floating
    bonded groups, finding 10.29).
    """
    linked = {a for a, _b, _R in finite_branches} | {b for _a, b, _R in finite_branches}
    return set(system.active_clusters) - set(system.cluster_input) - linked


def _zero_net_current(
    elec_total: dict[str, complex],
    system: _ActiveSystem,
    cluster_id: dict[str, str],
    finite_branches: list[tuple[str, str, float]],
) -> None:
    """Make the members of an isolated floating cluster sum to exactly 0.

    The solve enforces Kirchhoff's current law to round-off; summed in
    the member order of :attr:`FieldResult.clusters` (sorted names, the
    left-to-right sum of :meth:`FieldResult.cluster_impedance`) the
    last member absorbs that residue, so the cluster impedance of a
    floating cluster is ``nan`` instead of a ratio of round-off (the
    convention of the ``mom`` family, 10.29).
    """
    for root in _passive_isolated_roots(system, finite_branches):
        members = sorted(n for n in system.active_elecs if cluster_id[n] == root)
        if len(members) < 2:
            continue
        rest = sum(elec_total[m] for m in members[:-1])
        elec_total[members[-1]] = -complex(rest)


def _active_system(
    *,
    electrodes,
    elec_input_current: dict[str, complex],
    cluster_id: dict[str, str],
    elec_to_segidx: dict[str, list[int]],
    finite_branches: list[tuple[str, str, float]],
    pseudo_owners: list[str],
    floating: bool = False,
) -> _ActiveSystem | None:
    """Identify the active nodes, owners and branches of one solve.

    Returns ``None`` when nothing is driven (no impressed current), in
    which case every owner carries zero current. With ``floating`` the
    non-trivial source-free components float
    (:func:`_floating_cluster_set`); without it they are left out, as up
    to 0.15.0.
    """
    cluster_input = _cluster_input_currents(elec_input_current, cluster_id)
    if floating:
        all_owners = [e.name for e in electrodes] + list(pseudo_owners)
        owner_roots = [
            cluster_id[n] for n in all_owners if len(elec_to_segidx.get(n, [])) > 0
        ]
        active_clusters_set = _floating_cluster_set(
            cluster_input, finite_branches, cluster_id, owner_roots
        )
    else:
        if not cluster_input and not finite_branches:
            return None
        active_clusters_set = _active_cluster_set(cluster_input, finite_branches)
    if not active_clusters_set:
        return None

    active_clusters = sorted(active_clusters_set)
    cluster_idx = {c: k for k, c in enumerate(active_clusters)}

    # Owners: real electrodes + conductor pseudo-nodes. Real electrodes
    # come first (so the historic ordering is preserved when there are
    # no pseudo-owners), pseudo-owners are appended in the order in
    # which the discretiser produced them.
    #
    # An owner only enters the Z-matrix block if it has at least one
    # segment in ``elec_to_segidx`` — anonymous interior nodes from an
    # *isolated* distributed conductor have no leakage and contribute
    # to the system only via KCL (Block 2), not via Block 1.
    all_owners = [e.name for e in electrodes] + list(pseudo_owners)
    active_elecs = [
        n
        for n in all_owners
        if cluster_id[n] in active_clusters_set and len(elec_to_segidx.get(n, [])) > 0
    ]

    # Active branches: only those entirely inside the active node set
    # (always true here, but kept explicit for safety). We also keep
    # the original index of each active branch so that the optional
    # inductance matrix — built in the same order as ``finite_branches``
    # — can be restricted to the active subset below.
    active_branch_indices: list[int] = []
    active_branches: list[tuple[str, str, float]] = []
    for idx, (a, b, R) in enumerate(finite_branches):
        if a in active_clusters_set and b in active_clusters_set:
            active_branch_indices.append(idx)
            active_branches.append((a, b, R))
    return _ActiveSystem(
        cluster_input=cluster_input,
        active_clusters=active_clusters,
        cluster_idx=cluster_idx,
        active_elecs=active_elecs,
        active_branch_indices=active_branch_indices,
        active_branches=active_branches,
    )


def _shell_diagonal(
    shell_coefficients: np.ndarray | None, seg_lengths: np.ndarray
) -> np.ndarray | None:
    """Per-segment radial shell resistance ``C_k / L_k`` (ADR-0012 V2).

    ``None`` when no segment carries a concrete shell — the historic
    behaviour is then preserved bit-exact.
    """
    if shell_coefficients is None or not np.any(shell_coefficients > 0.0):
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            seg_lengths > 0.0,
            np.asarray(shell_coefficients, dtype=float) / seg_lengths,
            0.0,
        )


def _uniform_excitation(
    active_elecs: list[str],
    elec_to_segidx: dict[str, list[int]],
    seg_lengths: np.ndarray,
    n_segments: int,
) -> np.ndarray:
    """Excitation matrix ``E``: 1 A per owner, uniform per unit length.

    Column ``j`` distributes 1 A over the segments of owner ``j`` in
    proportion to their lengths, shape ``(n_segments, N_a)``.
    """
    excitation = np.zeros((n_segments, len(active_elecs)))
    for j, name_j in enumerate(active_elecs):
        idxs_j = elec_to_segidx[name_j]
        L_j = seg_lengths[idxs_j].sum()
        # uniform per unit length, Σ = 1 A
        excitation[idxs_j, j] = seg_lengths[idxs_j] / L_j
    return excitation


def _reduce_rows(
    active_elecs: list[str],
    elec_to_segidx: dict[str, list[int]],
    seg_lengths: np.ndarray,
    phi_all: np.ndarray,
) -> np.ndarray:
    """Multi-port matrix ``Z`` from the segment potentials ``K·E``.

    Row reduction: **length-weighted** average of the segment
    potentials (audit 2026-07-08, WP-B3). This is the Galerkin pairing
    consistent with the length-weighted excitation columns and restores
    exact reciprocity ``Z[i, j] == Z[j, i]`` at the discrete level. The
    historic unweighted mean broke symmetry whenever an electrode
    carried segments of unequal length (mesh electrodes with
    ``dx/nx != dy/ny``, rods split at the layer interface); for uniform
    per-electrode segment lengths — every rod/ring/strip — the two
    averages are identical, so historic results are unchanged there.
    """
    N_a = len(active_elecs)
    # Square for the multi-port matrix; rectangular for the coupling of
    # the floating core to outside anchors (_solve_unit_excitations).
    Z = np.empty((N_a, phi_all.shape[1]))
    for i, name_i in enumerate(active_elecs):
        idxs_i = elec_to_segidx[name_i]
        w_i = seg_lengths[idxs_i]
        Z[i, :] = (w_i @ phi_all[idxs_i, :]) / w_i.sum()
    return Z


def _multiport_matrix(
    system: _ActiveSystem,
    *,
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    elec_to_segidx: dict[str, list[int]],
    self_kernel,
    shell_diag: np.ndarray | None,
    multiport_cache: dict | None,
) -> np.ndarray:
    """Multi-port grounding matrix ``Z`` of the active owners.

    ``Z[i, j]`` is the average potential at owner ``i`` for 1 A
    injected (uniform per unit length) at owner ``j``.

    ADR-0010 Tier 1 (audit 2026-07-08, WP-F): all ``N_a`` excitation
    columns are passed to the kernel in ONE call as an
    ``(n_segments, N_a)`` matrix, so the kernel's O(N²) geometry work
    is done once instead of once per column. With ``multiport_cache``
    the assembled ``Z`` is reused across the per-frequency calls (``Z``
    is frequency-independent; only the branch block of the augmented
    system changes with ω). The cache also keeps the excitation ``E``
    and the segment potentials ``K·E`` of the active owners (keys
    ``"excitation"`` and ``"phi_all"``).
    """
    active_elecs = system.active_elecs
    if (
        multiport_cache is not None
        and multiport_cache.get("active_elecs") == active_elecs
    ):
        return multiport_cache["Z"]
    n_segments = seg_points.shape[0]
    excitation = _uniform_excitation(
        active_elecs, elec_to_segidx, seg_lengths, n_segments
    )
    phi_all = self_kernel(seg_points, seg_lengths, wire_radii, excitation)
    if shell_diag is not None:
        phi_all = phi_all + shell_diag[:, None] * excitation
    Z = _reduce_rows(active_elecs, elec_to_segidx, seg_lengths, phi_all)
    if multiport_cache is not None:
        multiport_cache["active_elecs"] = list(active_elecs)
        multiport_cache["Z"] = Z
        multiport_cache["excitation"] = excitation
        multiport_cache["phi_all"] = phi_all
    return Z


def _augmented_matrix(
    Z: np.ndarray, system: _ActiveSystem, cluster_id: dict[str, str]
) -> np.ndarray:
    """Real augmented matrix of the nodal system (see ``_solve_cluster_currents``).

    ::

        [ Z      -C       0    ] [ I_e    ]   [ 0       ]
        [ C^T    0        B^T  ] [ phi_n  ] = [ I_in    ]
        [ 0      B        R_b  ] [ I_b    ]   [ 0       ]
    """
    N_a, K_a, M_a = system.n_owners, system.n_nodes, system.n_branches
    cluster_idx = system.cluster_idx
    n_unknowns = N_a + K_a + M_a
    A = np.zeros((n_unknowns, n_unknowns))

    # Block 1: Z · I_e − C · phi_n = 0
    A[:N_a, :N_a] = Z
    elec_cluster = np.array(
        [cluster_idx[cluster_id[name]] for name in system.active_elecs], dtype=int
    )
    owners = np.arange(N_a)
    A[owners, N_a + elec_cluster] = -1.0  # −C

    # Block 2: KCL per node:  C^T · I_e  +  B^T · I_b  =  I_in
    A[N_a + elec_cluster, owners] = 1.0  # C^T
    for m, (a, b, _R) in enumerate(system.active_branches):
        ka = cluster_idx[a]
        kb = cluster_idx[b]
        # Branch convention: positive direction a → b. KCL at node a:
        # the branch leaves a with +I_b → contributes +1 to row a.
        # KCL at node b: enters with +I_b → −1 (re-arranged so that
        # +B^T appears on the LHS of  C^T I_e + B^T I_b = I_in).
        A[N_a + ka, N_a + K_a + m] = +1.0
        A[N_a + kb, N_a + K_a + m] = -1.0

    # Block 3: Ohm's law per branch:  phi_a − phi_b = R · I_b
    #   ⇔  +1·phi_a − 1·phi_b − R · I_b = 0
    # (positive branch direction is a → b, so a positive I_b means
    # current leaves node a and enters node b — consistent with the
    # KCL signs above.)
    for m, (a, b, R) in enumerate(system.active_branches):
        ka = cluster_idx[a]
        kb = cluster_idx[b]
        A[N_a + K_a + m, N_a + ka] = +1.0
        A[N_a + K_a + m, N_a + kb] = -1.0
        A[N_a + K_a + m, N_a + K_a + m] = -R
    return A


def _augmented_rhs(system: _ActiveSystem) -> tuple[np.ndarray, np.ndarray]:
    """Real and imaginary right-hand side (impressed node currents)."""
    N_a, K_a, M_a = system.n_owners, system.n_nodes, system.n_branches
    n_unknowns = N_a + K_a + M_a
    b_re = np.zeros(n_unknowns)
    b_im = np.zeros(n_unknowns)
    for k, c in enumerate(system.active_clusters):
        ic = system.cluster_input.get(c, 0j)
        b_re[N_a + k] = ic.real
        b_im[N_a + k] = ic.imag
    return b_re, b_im


def _branch_impedance_matrix(
    A: np.ndarray,
    system: _ActiveSystem,
    *,
    omega: float,
    inductance_matrix: np.ndarray,
    carson_correction: np.ndarray | None,
    n_finite_branches: int,
) -> np.ndarray:
    """Complex augmented matrix with the inductive branch block.

    Augments Block 3 with ``jωL`` (ADR-0004) and the optional
    frequency-dependent branch correction ``ΔZ``: the earth-return
    correction (ADR-0005 / ADR-0006, built against the image convention
    of ``L``) plus the skin-effect correction of
    ``resistance_model="skin"`` conductors on the diagonal (review pass
    10, 10.37; see :mod:`groundfield.solver._inductive`). The diagonal
    already carries ``−R``; ``−(jωL_mm + ΔZ_mm)`` is added to it, and
    ``−(jωL_mm' + ΔZ_mm')`` is written to the off-diagonal positions of
    branch row ``m``, branch-current column ``m'``. ``ΔZ`` is already the
    *full* per-branch impedance contribution (per-unit-length values
    times lengths), so it adds with sign −1.

    Raises
    ------
    ValueError
        If the inductance or correction matrix does not match the
        finite-branch list.
    """
    # Restrict the (full-finite-branches) inductance matrix to the
    # active subset; both ``finite_branches`` and
    # ``inductance_matrix`` must use the same ordering.
    n_total_branches = inductance_matrix.shape[0]
    if inductance_matrix.shape != (n_total_branches, n_total_branches):
        raise ValueError(
            f"inductance_matrix must be square, got {inductance_matrix.shape}."
        )
    if n_total_branches != n_finite_branches:
        raise ValueError(
            f"inductance_matrix size {n_total_branches} does not "
            f"match number of finite_branches ({n_finite_branches})."
        )
    idx = system.active_branch_indices
    L_active = inductance_matrix[np.ix_(idx, idx)]
    # ADR-0005: optional Carson correction. Pre-restricted to
    # the active subset by the same index permutation as L.
    dZ_carson_active: np.ndarray | None = None
    if carson_correction is not None:
        if carson_correction.shape != inductance_matrix.shape:
            raise ValueError(
                f"carson_correction shape {carson_correction.shape} "
                f"must match inductance_matrix {inductance_matrix.shape}."
            )
        dZ_carson_active = carson_correction[np.ix_(idx, idx)]
    N_a, K_a, M_a = system.n_owners, system.n_nodes, system.n_branches
    A_c = A.astype(complex)
    contrib = 1j * omega * L_active
    if dZ_carson_active is not None:
        contrib = contrib + dZ_carson_active
    r0 = N_a + K_a
    diag = np.arange(r0, r0 + M_a)
    block_diag = A_c[diag, diag] - np.diagonal(contrib)
    A_c[r0 : r0 + M_a, r0 : r0 + M_a] = -contrib
    A_c[diag, diag] = block_diag
    return A_c


def _augmented_solution(
    A: np.ndarray,
    b_re: np.ndarray,
    b_im: np.ndarray,
    system: _ActiveSystem,
    *,
    omega: float,
    inductance_matrix: np.ndarray | None,
    carson_correction: np.ndarray | None,
    n_finite_branches: int,
) -> np.ndarray:
    """Solve the augmented system; returns ``[I_e, phi_n, I_b]`` (complex).

    Two paths:

    (a) DC / no inductive coupling — ``Z`` and ``R_b`` are real, so the
        real and imaginary parts of a complex source are solved with one
        factorisation and two right-hand sides (ADR-0010 Tier 1).
    (b) Inductive coupling active (``inductance_matrix is not None``
        and ``omega != 0``) — the branch block becomes complex
        (``Z_b = R + jωL``), so one complex system is solved.
    """
    use_inductive = (
        inductance_matrix is not None and omega != 0.0 and system.n_branches > 0
    )
    if not use_inductive:
        sol = np.linalg.solve(A, np.column_stack([b_re, b_im]))
        return sol[:, 0] + 1j * sol[:, 1]
    A_c = _branch_impedance_matrix(
        A,
        system,
        omega=omega,
        inductance_matrix=inductance_matrix,
        carson_correction=carson_correction,
        n_finite_branches=n_finite_branches,
    )
    b_c = b_re + 1j * b_im
    return np.linalg.solve(A_c, b_c)


def _owner_currents(
    A: np.ndarray,
    b_re: np.ndarray,
    b_im: np.ndarray,
    system: _ActiveSystem,
    *,
    omega: float,
    inductance_matrix: np.ndarray | None,
    carson_correction: np.ndarray | None,
    n_finite_branches: int,
) -> np.ndarray:
    """Leakage currents of the active owners (see :func:`_augmented_solution`)."""
    sol = _augmented_solution(
        A,
        b_re,
        b_im,
        system,
        omega=omega,
        inductance_matrix=inductance_matrix,
        carson_correction=carson_correction,
        n_finite_branches=n_finite_branches,
    )
    return sol[: system.n_owners]


def _solve_cluster_currents(
    *,
    electrodes,
    elec_input_current: dict[str, complex],
    cluster_id: dict[str, str],
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    elec_to_segidx: dict[str, list[int]],
    self_kernel,
    finite_branches: list[tuple[str, str, float]] | None = None,
    pseudo_owners: list[str] | None = None,
    omega: float = 0.0,
    inductance_matrix: np.ndarray | None = None,
    carson_correction: np.ndarray | None = None,
    shell_coefficients: np.ndarray | None = None,
    multiport_cache: dict | None = None,
    floating: bool = False,
    node_potentials: dict[str, complex] | None = None,
) -> dict[str, complex]:
    """Distribute input currents through grounding and finite-conductor branches.

    Solves the augmented nodal-analysis system that couples the
    multi-port grounding matrix $Z$ with the resistive branches
    introduced by finite-impedance conductors.

    Notation
    --------
    Let:

    - $\\mathbf{I}_e \\in \\mathbb{C}^{N_a}$
      — total current leaked into the soil per active electrode,
    - $\\boldsymbol{\\varphi}_n \\in \\mathbb{C}^{K_a}$
      — common potential per active node (= cluster of electrodes
      fused by ideal conductors),
    - $\\mathbf{I}_b \\in \\mathbb{C}^{M_a}$
      — current through each active finite-impedance branch,
    - $C \\in \\{0,1\\}^{N_a \\times K_a}$ the electrode-to-node
      incidence matrix,
    - $B \\in \\{-1,0,+1\\}^{M_a \\times K_a}$ the branch-to-node
      incidence matrix (``+1`` at the branch start, ``-1`` at its
      end),
    - $R_b \\in \\mathbb{R}^{M_a \\times M_a}$ the diagonal matrix
      of branch resistances.

    The system

    $$
    \\begin{bmatrix} Z & -C & 0 \\\\
                    C^{\\top} & 0 & B^{\\top} \\\\
                    0 & B & -R_b \\end{bmatrix}
    \\begin{bmatrix} \\mathbf{I}_e \\\\
                    \\boldsymbol{\\varphi}_n \\\\
                    \\mathbf{I}_b \\end{bmatrix}
    \\;=\\;
    \\begin{bmatrix} \\mathbf{0} \\\\
                    \\mathbf{I}_{\\text{in}} \\\\
                    \\mathbf{0} \\end{bmatrix}
    $$

    expresses, in turn, (i) the multi-port relation
    $Z\\mathbf{I}_e = C\\boldsymbol{\\varphi}_n$ between leakage
    currents and node potentials, (ii) Kirchhoff's current law per
    node $C^{\\top}\\mathbf{I}_e + B^{\\top}\\mathbf{I}_b
    = \\mathbf{I}_{\\text{in}}$, and (iii) Ohm's law along each branch
    $B\\boldsymbol{\\varphi}_n - R_b\\mathbf{I}_b = \\mathbf{0}$
    (i.e. $\\varphi_a - \\varphi_b = R_b\\, I_b$, with positive
    branch direction $a \\to b$). For $M_a = 0$ (no finite
    branches) the system collapses to the classical cluster
    constraint with one common potential per cluster.

    Active set
    ----------
    The function automatically identifies the *active set* of nodes:
    every node that carries an input current, plus every node that is
    transitively connected to an active node by at least one finite
    branch. Passive nodes (with no input current and no finite branch
    to the active region) carry zero current and are excluded from
    the linear system to keep its size minimal.

    With ``floating=True`` (the image-family pipeline, finding 10.29)
    source-free components float instead: a bonded group of passive
    electrodes or a passive network joined by finite branches enters
    the system with zero net impressed current, so its members share
    one potential per cluster and exchange their equalising currents.
    A single passive electrode without branches is still left out —
    under the uniform-current ansatz it carries no current either way
    (:func:`_floating_cluster_set`). The members of an isolated floating
    cluster report currents that sum to exactly zero.

    Parameters
    ----------
    self_kernel
        Callable with signature
        ``(seg_points, seg_lengths, wire_radii, currents) -> phi``;
        captures the soil-specific self-action (homogeneous, 2-layer,
        n-layer via CIM, ...).
    finite_branches
        List of ``(node_a, node_b, R)`` triples, where ``node_a`` and
        ``node_b`` are *cluster roots* and ``R`` the finite branch
        resistance in Ω. Default ``None`` reproduces the historic
        ideal-cluster behaviour exactly.
    pseudo_owners
        Names of conductor pseudo-nodes (introduced by the
        distributed-conductor model, ADR-0003) that participate in
        the linear system in addition to the real electrodes. Each
        pseudo-owner is treated as its own one-segment cluster:
        ``cluster_id[name] == name`` is expected, and
        ``elec_to_segidx[name]`` must already point at its conductor
        midpoint segment.
    multiport_cache
        Optional dict used to reuse the multi-port grounding matrix
        ``Z`` across calls with identical geometry (ADR-0010 Tier 0a:
        the frequency loop only changes the branch block, never
        ``Z``). Pass one empty dict per solve and share it across the
        per-frequency calls; the first call fills it, subsequent
        calls skip the kernel assembly entirely. ``None`` (default)
        disables caching.
    floating
        Let source-free components float (see *Active set*). Default
        ``False`` keeps the behaviour of 0.15.0 for the other callers
        (``cim``, ``solve_mutual_*``).
    node_potentials
        Optional output mapping, filled with the potential in V of every
        active node (cluster root, junction, conductor interior node)
        from the solution of the augmented system.
    """
    if finite_branches is None:
        finite_branches = []
    if pseudo_owners is None:
        pseudo_owners = []

    elec_total = {e.name: 0j for e in electrodes}
    for pn in pseudo_owners:
        elec_total[pn] = 0j
    system = _active_system(
        electrodes=electrodes,
        elec_input_current=elec_input_current,
        cluster_id=cluster_id,
        elec_to_segidx=elec_to_segidx,
        finite_branches=finite_branches,
        pseudo_owners=pseudo_owners,
        floating=floating,
    )
    if system is None:
        return elec_total

    shell_diag = _shell_diagonal(shell_coefficients, seg_lengths)
    Z = _multiport_matrix(
        system,
        seg_points=seg_points,
        seg_lengths=seg_lengths,
        wire_radii=wire_radii,
        elec_to_segidx=elec_to_segidx,
        self_kernel=self_kernel,
        shell_diag=shell_diag,
        multiport_cache=multiport_cache,
    )
    A = _augmented_matrix(Z, system, cluster_id)
    b_re, b_im = _augmented_rhs(system)
    sol = _augmented_solution(
        A,
        b_re,
        b_im,
        system,
        omega=omega,
        inductance_matrix=inductance_matrix,
        carson_correction=carson_correction,
        n_finite_branches=len(finite_branches),
    )
    N_a = system.n_owners
    for i, name in enumerate(system.active_elecs):
        elec_total[name] = complex(sol[i])
    if floating:
        _zero_net_current(elec_total, system, cluster_id, finite_branches)
    if node_potentials is not None:
        for k, root in enumerate(system.active_clusters):
            node_potentials[root] = complex(sol[N_a + k])
    return elec_total


# ---------------------------------------------------------------------
# Unit excitations of one geometry (solve_mutual_*, audit A2S-03)
# ---------------------------------------------------------------------


def _augmented_solutions(
    A: np.ndarray,
    B_re: np.ndarray,
    B_im: np.ndarray,
    system: _ActiveSystem,
    *,
    omega: float,
    inductance_matrix: np.ndarray | None,
    carson_correction: np.ndarray | None,
    n_finite_branches: int,
) -> np.ndarray:
    """:func:`_augmented_solution` for several right-hand sides at once.

    ``B_re`` and ``B_im`` have one column per excitation; returns the
    complex solutions column by column, from one factorisation of the
    augmented matrix. One column reproduces :func:`_augmented_solution`
    (the same ``numpy.linalg.solve`` call).
    """
    use_inductive = (
        inductance_matrix is not None and omega != 0.0 and system.n_branches > 0
    )
    n_rhs = B_re.shape[1]
    if not use_inductive:
        sol = np.linalg.solve(A, np.hstack([B_re, B_im]))
        return sol[:, :n_rhs] + 1j * sol[:, n_rhs:]
    A_c = _branch_impedance_matrix(
        A,
        system,
        omega=omega,
        inductance_matrix=inductance_matrix,
        carson_correction=carson_correction,
        n_finite_branches=n_finite_branches,
    )
    return np.linalg.solve(A_c, B_re + 1j * B_im)


def _solve_unit_excitations(
    *,
    electrodes,
    anchors: list[str],
    cluster_id: dict[str, str],
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    elec_to_segidx: dict[str, list[int]],
    self_kernel,
    finite_branches: list[tuple[str, str, float]],
    pseudo_owners: list[str],
    omega: float = 0.0,
    inductance_matrix: np.ndarray | None = None,
    carson_correction: np.ndarray | None = None,
    shell_coefficients: np.ndarray | None = None,
) -> list[dict[str, complex]]:
    """Owner currents for 1 A into each anchor, the other clusters floating.

    The result equals one :func:`_solve_cluster_currents` call with
    ``floating=True`` per anchor (finding 10.29), at the cost of about
    one: every source-free component that floats (a bonded group or a
    network with branches, :func:`_floating_cluster_set`) is part of the
    system of *every* excitation, so the multi-port matrix of this
    *floating core*, the augmented matrix and its factorisation are
    built once and every excitation is one right-hand side.

    * An anchor in the core changes only the right-hand side (1 A at its
      node).
    * Any other anchor is a single owner without branches: KCL fixes its
      current to exactly 1 A, and its coupling to the core, the column
      ``Z_{core, anchor}`` of the multi-port matrix (all such columns in
      one kernel call), moves to the right-hand side. Its own row only
      returns its potential, which the callers evaluate from the
      segment currents anyway.
    * Without a floating core every system is one anchor alone and the
      per-anchor solve is kept (linear in the number of anchors, results
      unchanged).

    Up to audit round 2 (A2S-03) every excitation rebuilt the core's
    multi-port matrix with all floating owners as columns and factorised
    the full augmented system again: the cost grew with the square of
    the number of bonded grounding nodes (256 ring-and-rod nodes: 24.3 s
    without floating, 199.9 s with it).

    Parameters
    ----------
    anchors
        Owner (electrode) names, one excitation each; every one must have
        leakage segments.

    The other parameters are those of :func:`_solve_cluster_currents`.

    Returns
    -------
    list[dict[str, complex]]
        Owner currents in A per anchor (every owner, zero where inactive).
    """
    template = {e.name: 0j for e in electrodes}
    for pn in pseudo_owners:
        template[pn] = 0j
    common = dict(
        electrodes=electrodes,
        cluster_id=cluster_id,
        elec_to_segidx=elec_to_segidx,
        finite_branches=finite_branches,
        pseudo_owners=pseudo_owners,
        floating=True,
    )
    core = _active_system(elec_input_current={}, **common)
    if core is None:
        return [
            _solve_cluster_currents(
                elec_input_current={
                    **{e.name: 0j for e in electrodes},
                    anchor: 1.0 + 0j,
                },
                seg_points=seg_points,
                seg_lengths=seg_lengths,
                wire_radii=wire_radii,
                self_kernel=self_kernel,
                omega=omega,
                inductance_matrix=inductance_matrix,
                carson_correction=carson_correction,
                shell_coefficients=shell_coefficients,
                **common,
            )
            for anchor in anchors
        ]

    shell_diag = _shell_diagonal(shell_coefficients, seg_lengths)
    Z = _multiport_matrix(
        core,
        seg_points=seg_points,
        seg_lengths=seg_lengths,
        wire_radii=wire_radii,
        elec_to_segidx=elec_to_segidx,
        self_kernel=self_kernel,
        shell_diag=shell_diag,
        multiport_cache=None,
    )
    A = _augmented_matrix(Z, core, cluster_id)
    N_a = core.n_owners
    n = A.shape[0]
    core_nodes = set(core.active_clusters)
    outside = [a for a in dict.fromkeys(anchors) if cluster_id[a] not in core_nodes]
    coupling: dict[str, np.ndarray] = {}
    if outside:
        # Z[core owner, outside anchor] for every outside anchor at once.
        excitation = _uniform_excitation(
            outside, elec_to_segidx, seg_lengths, seg_points.shape[0]
        )
        phi = self_kernel(seg_points, seg_lengths, wire_radii, excitation)
        if shell_diag is not None:
            phi = phi + shell_diag[:, None] * excitation
        Z_out = _reduce_rows(core.active_elecs, elec_to_segidx, seg_lengths, phi)
        coupling = {a: Z_out[:, k] for k, a in enumerate(outside)}
    B_re = np.zeros((n, len(anchors)))
    B_im = np.zeros((n, len(anchors)))
    for k, anchor in enumerate(anchors):
        if anchor in coupling:
            B_re[:N_a, k] = -coupling[anchor]
        else:
            B_re[N_a + core.cluster_idx[cluster_id[anchor]], k] = 1.0
    sol = _augmented_solutions(
        A,
        B_re,
        B_im,
        core,
        omega=omega,
        inductance_matrix=inductance_matrix,
        carson_correction=carson_correction,
        n_finite_branches=len(finite_branches),
    )
    out: list[dict[str, complex]] = []
    for k, anchor in enumerate(anchors):
        elec_total = dict(template)
        for i, name in enumerate(core.active_elecs):
            elec_total[name] = complex(sol[i, k])
        if anchor in coupling:
            elec_total[anchor] = 1.0 + 0j
            system = core
        else:
            system = _ActiveSystem(
                cluster_input={cluster_id[anchor]: 1.0 + 0j},
                active_clusters=core.active_clusters,
                cluster_idx=core.cluster_idx,
                active_elecs=core.active_elecs,
                active_branch_indices=core.active_branch_indices,
                active_branches=core.active_branches,
            )
        _zero_net_current(elec_total, system, cluster_id, finite_branches)
        out.append(elec_total)
    return out
