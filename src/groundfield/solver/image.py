"""Image-charge backend for **homogeneous** soil.

Computes the potential field of an arbitrary grounding system in a
homogeneous half-space (resistivity $\\rho$, soil surface at
$z = 0$, $z$ axis pointing into the soil) using the
classical image-charge method.

Notes
-----
A point current source $I$ at $r_s = (x_s, y_s, z_s)$ with
$z_s > 0$ (inside the soil) produces, in a homogeneous half-space
with an insulating soil surface, the potential
$$
\\varphi(r) \\;=\\; \\frac{\\rho\\, I}{4\\pi}\\,
\\Big(\\frac{1}{|r - r_s|} + \\frac{1}{|r - r_s'|}\\Big),
$$
with the image $r_s' = (x_s, y_s, -z_s)$ mirrored at the soil
surface. An extended electrode is discretised into $N$ segments;
each segment carries one point current source at its midpoint. The
total current $I_e$ of an electrode is distributed **uniformly
per unit length** across its segments — a surprisingly good
approximation for wire electrodes at low frequencies (cf. Sunde 1968,
Tagg 1964).

The input impedance of an electrode is computed as the average of the
potential on its own segment midpoints (average-potential method). For
a single driven rod the backend reproduces the Sunde formula within a
few per cent.

Further properties of this backend:

* Frequency-independent: in the quasi-static range
  $f < 1\\,\\mathrm{kHz}$ the backend returns the same real
  solution per frequency. Complex extensions (Carson,
  frequency-dependent soil) come in later backends.
* Multiple electrodes: each electrode has its own total current
  (sum of the current sources attached to it). An electrode without a
  source carries zero current and acts purely as a passive observer.

References
----------
.. [1] E. D. Sunde, *Earth Conduction Effects in Transmission
       Systems*, Dover, 1968.
.. [2] G. F. Tagg, *Earth Resistances*, Pitman, 1964.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.geometry.electrodes import _ElectrodeBase
    from groundfield.solver.engine import Engine
    from groundfield.world import World

from groundfield.geometry.electrodes import (
    GridMeshElectrode,
    MeshElectrode,
    RingElectrode,
    RodElectrode,
    StripElectrode,
)
from groundfield.soil.models import HomogeneousSoil
from groundfield.solver.result import FieldResult, PointSource

__all__ = ["solve_image"]

# Numerical cutoff: no field point may be closer to a source than
# ``_MIN_DISTANCE`` (in metres). Distances below the cutoff are clamped
# to it to suppress the 1/r singularity during visual evaluations.
_MIN_DISTANCE = 1e-3


@dataclass
class _Segment:
    """Discretisation segment of an electrode or distributed conductor.

    Attributes
    ----------
    midpoint
        Cartesian coordinates ``(x, y, z)`` of the segment midpoint
        in metres. Used as the point-source location in the
        Green's-function evaluation.
    length
        Segment length in metres.
    electrode_name
        Name of the *node* the segment leaks current into. For an
        electrode segment this is the electrode's name (which is
        also the cluster root for cluster-building). For a
        galvanic-conductor segment in the distributed-conductor
        model (ADR-0003) this is the pseudo-node name
        ``f"__cond_{conductor.name}__seg_{k}"`` and the segment
        forms its own cluster.
    wire_radius
        Wire radius in metres (for the analytical line self-action
        on the Z-matrix diagonal).
    conductor_name
        For conductor segments: the owning conductor's name. ``None``
        for electrode segments.
    """

    midpoint: np.ndarray  # shape (3,)
    length: float
    electrode_name: str
    wire_radius: float
    conductor_name: str | None = None
    # ADR-0007: which soil layer the segment lives in (0 = upper,
    # 1 = next, ..., n-1 = bottom semi-infinite). Set by the
    # discretiser when ``layer_interfaces`` is supplied; left at 0
    # for homogeneous-soil callers.
    layer_index: int = 0


# ---------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------


def _discretize_rod(electrode: RodElectrode, ds: float) -> list[_Segment]:
    """Vertical driven rod into N equally long segments."""
    n = max(1, int(np.ceil(electrode.length / ds)))
    seg_len = electrode.length / n
    x0, y0, z0 = electrode.position
    segs: list[_Segment] = []
    for k in range(n):
        zc = z0 + (k + 0.5) * seg_len
        segs.append(
            _Segment(
                midpoint=np.array([x0, y0, zc], dtype=float),
                length=seg_len,
                electrode_name=electrode.name,
                wire_radius=electrode.wire_radius,
            )
        )
    return segs


def _discretize_ring(electrode: RingElectrode, ds: float) -> list[_Segment]:
    """Horizontal ring into N equally long arc segments."""
    perimeter = 2.0 * np.pi * electrode.radius
    n = max(8, int(np.ceil(perimeter / ds)))
    seg_len = perimeter / n
    cx, cy, cz = electrode.center
    segs: list[_Segment] = []
    for k in range(n):
        phi = 2.0 * np.pi * (k + 0.5) / n
        segs.append(
            _Segment(
                midpoint=np.array(
                    [cx + electrode.radius * np.cos(phi),
                     cy + electrode.radius * np.sin(phi),
                     cz],
                    dtype=float,
                ),
                length=seg_len,
                electrode_name=electrode.name,
                wire_radius=electrode.wire_radius,
            )
        )
    return segs


def _discretize_strip(electrode: StripElectrode, ds: float) -> list[_Segment]:
    """Horizontal straight strip into N equally long segments along its axis."""
    L = electrode.length
    n = max(1, int(np.ceil(L / ds)))
    seg_len = L / n
    p0 = np.array(electrode.start, dtype=float)
    p1 = np.array(electrode.end, dtype=float)
    direction = (p1 - p0) / L
    segs: list[_Segment] = []
    for k in range(n):
        midpoint = p0 + (k + 0.5) * seg_len * direction
        segs.append(
            _Segment(
                midpoint=midpoint,
                length=seg_len,
                electrode_name=electrode.name,
                wire_radius=electrode.wire_radius,
            )
        )
    return segs


def _grid_segments(
    *,
    cx: float,
    cy: float,
    cz: float,
    dx: float,
    dy: float,
    nx_wires: int,
    ny_wires: int,
    ds: float,
    electrode_name: str,
    wire_radius: float,
) -> list[_Segment]:
    """Common segment builder for the rectangular mesh / grid family.

    Parameters
    ----------
    nx_wires
        Number of longitudinal wires (each running from ``cx`` to
        ``cx + dx`` at one of ``nx_wires`` distinct ``y`` values).
    ny_wires
        Number of transverse wires (each running from ``cy`` to
        ``cy + dy`` at one of ``ny_wires`` distinct ``x`` values).
    """
    xs = np.linspace(cx, cx + dx, ny_wires)
    ys = np.linspace(cy, cy + dy, nx_wires)

    segs: list[_Segment] = []
    # Longitudinal wires (along x for each y)
    for y in ys:
        n = max(1, int(np.ceil(dx / ds)))
        seg_len = dx / n
        for k in range(n):
            xm = cx + (k + 0.5) * seg_len
            segs.append(
                _Segment(
                    midpoint=np.array([xm, y, cz], dtype=float),
                    length=seg_len,
                    electrode_name=electrode_name,
                    wire_radius=wire_radius,
                )
            )
    # Transverse wires (along y for each x)
    for x in xs:
        n = max(1, int(np.ceil(dy / ds)))
        seg_len = dy / n
        for k in range(n):
            ym = cy + (k + 0.5) * seg_len
            segs.append(
                _Segment(
                    midpoint=np.array([x, ym, cz], dtype=float),
                    length=seg_len,
                    electrode_name=electrode_name,
                    wire_radius=wire_radius,
                )
            )
    return segs


def _discretize_mesh(electrode: MeshElectrode, ds: float) -> list[_Segment]:
    """Mesh earth electrode (uniform spacing) as a grid of wires."""
    cx, cy, cz = electrode.corner
    dx, dy = electrode.size
    spacing = electrode.spacing
    nx = max(2, int(np.round(dx / spacing)) + 1)
    ny = max(2, int(np.round(dy / spacing)) + 1)
    return _grid_segments(
        cx=cx, cy=cy, cz=cz, dx=dx, dy=dy,
        nx_wires=ny, ny_wires=nx,
        ds=ds, electrode_name=electrode.name,
        wire_radius=electrode.wire_radius,
    )


def _discretize_grid_mesh(
    electrode: GridMeshElectrode, ds: float
) -> list[_Segment]:
    """Mesh earth electrode with explicit n_x × n_y meshes.

    The grid has ``n_x + 1`` transverse and ``n_y + 1`` longitudinal
    wires (one wire per cell boundary plus the outer perimeter).
    """
    cx, cy, cz = electrode.corner
    dx, dy = electrode.size
    return _grid_segments(
        cx=cx, cy=cy, cz=cz, dx=dx, dy=dy,
        nx_wires=electrode.n_y + 1,
        ny_wires=electrode.n_x + 1,
        ds=ds, electrode_name=electrode.name,
        wire_radius=electrode.wire_radius,
    )


def _discretize_electrode(
    electrode: "_ElectrodeBase", ds: float
) -> list[_Segment]:
    if isinstance(electrode, RodElectrode):
        return _discretize_rod(electrode, ds)
    if isinstance(electrode, RingElectrode):
        return _discretize_ring(electrode, ds)
    if isinstance(electrode, StripElectrode):
        return _discretize_strip(electrode, ds)
    if isinstance(electrode, GridMeshElectrode):
        return _discretize_grid_mesh(electrode, ds)
    if isinstance(electrode, MeshElectrode):
        return _discretize_mesh(electrode, ds)
    raise TypeError(
        f"Image backend does not know {type(electrode).__name__}."
    )


# ---------------------------------------------------------------------
# Conductor discretisation (ADR-0003: distributed conductor model)
# ---------------------------------------------------------------------


@dataclass
class _DistributedBranch:
    """One longitudinal-segment branch of a distributed conductor.

    Carries the topology used by the nodal-analysis system
    (``node_a``, ``node_b``, ``R``) plus the geometric information
    required by the inductive-coupling assembly (ADR-0004): the
    branch endpoints in 3-D and the wire radius.
    """

    node_a: str
    node_b: str
    R: float
    p_a: np.ndarray  # shape (3,)
    p_b: np.ndarray  # shape (3,)
    wire_radius: float
    inductive: bool  # True if the parent conductor enables Neumann coupling


def _conductor_node_name(conductor_name: str, k: int) -> str:
    """Pseudo-node identifier for the *k*-th midpoint segment of a
    distributed conductor.

    The naming convention `__cond_{name}__seg_{k}` is reserved — the
    leading double underscore guarantees no collision with user
    electrode names.
    """
    return f"__cond_{conductor_name}__seg_{k}"


def _discretize_conductor(conductor) -> tuple[list[_Segment], list[_DistributedBranch]]:
    """Discretise a conductor into midpoint segments + longitudinal branches.

    Implements the distributed-conductor model documented in ADR-0003.
    The conductor is split into ``n = conductor.n_segments`` collinear
    sub-pieces of equal length. Each sub-piece produces:

    - one pseudo-electrode segment at its midpoint, contributing to
      the multi-port grounding matrix only when
      ``coupling_to_soil == "galvanic"``;
    - one longitudinal branch whose endpoints are the pseudo-node
      midpoints (or the conductor's start / end electrode cluster at
      the conductor ends) and whose resistance is
      $R^{(k)} = \\rho_\\text{mat}\\, \\ell_k / A$.

    Topology
    --------
    For a conductor with ``n`` segments and ``coupling_to_soil ==
    "galvanic"`` the longitudinal-branch chain reads

    ::

        start_cluster ──[R/2]── M_0 ──[R]── M_1 ──[R]── ... ──[R]── M_{n-1} ──[R/2]── end_cluster

    where each ``M_k`` is a pseudo-electrode node (single segment at
    the midpoint of sub-piece *k*). The half-resistance stubs at the
    two ends preserve the total series resistance
    $\\sum R^{(k)} = R_\\text{total}$. For
    ``coupling_to_soil == "isolated"`` the pseudo-electrode list is
    empty; the longitudinal chain becomes a series of
    finite-resistance branches between the two end clusters with
    interior nodes that carry no leakage.

    Parameters
    ----------
    conductor
        :class:`Conductor` instance with finite
        ``discretize_segment_length`` (i.e. ``is_distributed``).

    Returns
    -------
    segments : list[_Segment]
        Pseudo-electrode segments at the conductor midpoints. Empty
        for ``coupling_to_soil == "isolated"``.
    branches : list[(str, str, float)]
        Longitudinal-branch list ``(node_in, node_out, R)``. For an
        isolated conductor the chain runs through anonymous interior
        nodes ``f"__cond_{name}__node_{k}"`` (k=1..n-1).
    """
    n = conductor.n_segments
    L = conductor.length
    if not conductor.is_distributed or n <= 1:
        # Lumped fallback — caller handles via _build_finite_branches.
        return [], []

    seg_len = L / n
    R_total = conductor.series_resistance
    R_seg = R_total / n if R_total > 0.0 else 0.0
    p0 = np.array(conductor.start, dtype=float)
    p1 = np.array(conductor.end, dtype=float)
    direction = (p1 - p0) / L

    # End-cluster identifiers (must exist in the world by construction)
    start_node = conductor.start_electrode
    end_node = conductor.end_electrode
    if start_node is None or end_node is None:
        # Floating endpoints: bind them to anonymous "endpoint" nodes
        # that participate in KCL but not in any cluster.
        if start_node is None:
            start_node = f"__cond_{conductor.name}__endpoint_start"
        if end_node is None:
            end_node = f"__cond_{conductor.name}__endpoint_end"

    segments: list[_Segment] = []
    branches: list[_DistributedBranch] = []
    inductive = getattr(conductor, "inductance_model", None) is not None

    if conductor.coupling_to_soil == "galvanic":
        # Each sub-piece becomes a one-segment pseudo-electrode at its
        # midpoint; longitudinal branches link the consecutive nodes
        # with R/2 stubs at the two ends.
        midpoint_nodes: list[str] = []
        midpoints: list[np.ndarray] = []
        for k in range(n):
            midpoint = p0 + (k + 0.5) * seg_len * direction
            mid_node = _conductor_node_name(conductor.name, k)
            midpoint_nodes.append(mid_node)
            midpoints.append(midpoint)
            segments.append(
                _Segment(
                    midpoint=midpoint,
                    length=seg_len,
                    electrode_name=mid_node,
                    wire_radius=conductor.wire_radius,
                    conductor_name=conductor.name,
                )
            )
        # Branch chain: start_cluster -[R/2]- M_0 -[R]- M_1 -...- M_{n-1} -[R/2]- end_cluster
        # Geometric endpoints of each branch (used for inductance
        # assembly): start at conductor.start, then run between
        # successive midpoints, then to conductor.end.
        branch_endpoints = [(p0, midpoints[0])]
        for k in range(n - 1):
            branch_endpoints.append((midpoints[k], midpoints[k + 1]))
        branch_endpoints.append((midpoints[-1], p1))
        Rs = [R_seg / 2.0] + [R_seg] * (n - 1) + [R_seg / 2.0]
        chain_nodes = [start_node] + midpoint_nodes + [end_node]
        for k, (pa, pb) in enumerate(branch_endpoints):
            branches.append(_DistributedBranch(
                node_a=chain_nodes[k],
                node_b=chain_nodes[k + 1],
                R=Rs[k],
                p_a=pa,
                p_b=pb,
                wire_radius=conductor.wire_radius,
                inductive=inductive,
            ))
    else:
        # ``isolated``: no leakage along the wire. Interior nodes are
        # anonymous; the chain is a series of n branches between the
        # two end clusters. The interior nodes do not appear in the
        # Z-matrix but participate in KCL (every interior node carries
        # zero leakage by construction).
        interior_nodes = [
            f"__cond_{conductor.name}__node_{k}" for k in range(1, n)
        ]
        nodes_chain = [start_node, *interior_nodes, end_node]
        # Geometric endpoints follow the conductor axis at the
        # sub-segment boundaries.
        node_positions = [p0 + (k * seg_len) * direction for k in range(n + 1)]
        for k in range(n):
            branches.append(_DistributedBranch(
                node_a=nodes_chain[k],
                node_b=nodes_chain[k + 1],
                R=R_seg,
                p_a=node_positions[k],
                p_b=node_positions[k + 1],
                wire_radius=conductor.wire_radius,
                inductive=inductive,
            ))

    return segments, branches


def _assemble_inductance_matrix(
    distributed_branches: list[_DistributedBranch],
    *,
    n_lumped_branches: int,
    n_total_branches: int,
    earth_model: str = "perfect_mirror",
    sigma_earth: float | None = None,
    layered_earth: object = None,
) -> tuple[np.ndarray | None, bool, "_CarsonBuilder | None"]:
    """Build the partial-inductance matrix over all finite branches.

    The output covers *all* finite branches (lumped + distributed) in
    the same ordering used by the solver's ``finite_branches`` list:
    lumped finite-impedance branches come first (no inductance, all
    zeros), distributed branches come after. Inductive entries are
    only populated when at least one distributed conductor was
    created with ``inductance_model == "neumann"``; the rest of the
    matrix is left zero.

    The earth is treated as a perfect magnetic mirror by default
    (ADR-0004). When ``earth_model == "carson_series"`` (ADR-0005)
    the function additionally returns a closure that, given an
    angular frequency $\\omega$, produces the Carson correction
    matrix $\\Delta Z_\\text{Carson}(\\omega)$ over the same
    branch indices. The solver evaluates that closure once per
    frequency and adds the result to the per-branch impedance block:

    $$
    Z_b(\\omega) \\;=\\; R \\;+\\; j\\omega\\,L
        \\;+\\; \\Delta Z_\\text{Carson}(\\omega).
    $$

    Cross-conductor mutual inductance is included automatically
    because the assembly iterates over every active distributed
    branch.

    Parameters
    ----------
    distributed_branches
        Branches produced by :func:`_build_distributed_topology`. The
        ``inductive`` flag selects whether each branch contributes
        to the matrix.
    n_lumped_branches
        Number of lumped finite-impedance branches that come before
        the distributed ones in ``finite_branches``.
    n_total_branches
        Total number of finite branches in ``finite_branches``.
    earth_model
        - ``"perfect_mirror"`` (default, ADR-0004): system stays
          real, third return value is ``None``.
        - ``"carson_series"`` (ADR-0005): per-meter Carson
          correction × length, third return is a closure
          ``omega -> dZ_carson(omega)``.
        - ``"sommerfeld"`` (ADR-0006): geometric Sommerfeld kernel
          integration over the segment-pair geometry, with
          layered-earth support, third return is a closure
          ``omega -> dZ_sommerfeld(omega)``.
    sigma_earth
        Earth conductivity in S/m. Required when
        ``earth_model == "carson_series"``.
    layered_earth
        :class:`LayeredEarth` configuration for the Sommerfeld
        kernel. Required when ``earth_model == "sommerfeld"``;
        ignored otherwise.

    Returns
    -------
    L : np.ndarray | None
        ``(n_total_branches, n_total_branches)`` partial-inductance
        matrix in henries, or ``None`` when no distributed conductor
        carries an inductive model.
    has_inductance : bool
        Convenience flag matching ``L is not None``.
    carson_builder : _CarsonBuilder | None
        Closure that evaluates the Carson correction matrix at a
        given $\\omega$. ``None`` when ``earth_model ==
        "perfect_mirror"`` or no inductive branches are present.
    """
    from groundfield.coupling.inductance import (
        build_carson_correction_matrix,
        build_inductance_matrix,
    )

    inductive_branches = [db for db in distributed_branches if db.inductive]
    if not inductive_branches:
        return None, False, None

    # Map each inductive branch back to its offset inside
    # ``finite_branches`` — the lumped block sits at indices
    # [0, n_lumped_branches), so the distributed-conductor branches
    # start at ``n_lumped_branches`` in the order they were appended.
    inductive_offsets: list[int] = []
    inductive_endpoints: list[np.ndarray] = []
    inductive_radii: list[float] = []
    for offset, db in enumerate(distributed_branches):
        if not db.inductive:
            continue
        inductive_offsets.append(n_lumped_branches + offset)
        inductive_endpoints.append(np.stack([db.p_a, db.p_b], axis=0))
        inductive_radii.append(db.wire_radius)

    if not inductive_offsets:
        return None, False, None

    seg_endpoints = np.stack(inductive_endpoints, axis=0)
    radii = np.array(inductive_radii)
    L_sub = build_inductance_matrix(seg_endpoints, radii, use_image=True)

    L_full = np.zeros((n_total_branches, n_total_branches))
    idx = np.array(inductive_offsets, dtype=int)
    L_full[np.ix_(idx, idx)] = L_sub

    carson_builder: "_CarsonBuilder | None" = None
    if earth_model == "carson_series":
        if sigma_earth is None or sigma_earth <= 0.0:
            raise ValueError(
                "earth_model='carson_series' requires sigma_earth > 0; "
                f"got sigma_earth={sigma_earth!r}"
            )
        seg_endpoints_snapshot = seg_endpoints.copy()
        radii_snapshot = radii.copy()
        idx_snapshot = idx.copy()
        sigma_snapshot = float(sigma_earth)
        n_total_snapshot = n_total_branches

        def _carson_at(omega: float) -> np.ndarray:
            dZ_full = np.zeros(
                (n_total_snapshot, n_total_snapshot), dtype=complex,
            )
            if omega <= 0.0:
                return dZ_full
            dZ_sub = build_carson_correction_matrix(
                seg_endpoints_snapshot,
                radii_snapshot,
                omega=omega,
                sigma_earth=sigma_snapshot,
            )
            dZ_full[np.ix_(idx_snapshot, idx_snapshot)] = dZ_sub
            return dZ_full

        carson_builder = _carson_at
    elif earth_model == "sommerfeld":
        if layered_earth is None:
            raise ValueError(
                "earth_model='sommerfeld' requires layered_earth "
                "(LayeredEarth instance); got None."
            )
        from groundfield.coupling.sommerfeld_inductance import (
            build_sommerfeld_correction_matrix,
        )

        seg_endpoints_snapshot = seg_endpoints.copy()
        radii_snapshot = radii.copy()
        idx_snapshot = idx.copy()
        earth_snapshot = layered_earth
        n_total_snapshot = n_total_branches

        def _sommerfeld_at(omega: float) -> np.ndarray:
            dZ_full = np.zeros(
                (n_total_snapshot, n_total_snapshot), dtype=complex,
            )
            if omega <= 0.0:
                return dZ_full
            dZ_sub = build_sommerfeld_correction_matrix(
                seg_endpoints_snapshot,
                radii_snapshot,
                omega=omega,
                earth=earth_snapshot,
            )
            dZ_full[np.ix_(idx_snapshot, idx_snapshot)] = dZ_sub
            return dZ_full

        carson_builder = _sommerfeld_at

    return L_full, True, carson_builder


# Type alias used by callers (no runtime cost; just documentation).
_CarsonBuilder = "callable[[float], np.ndarray]"


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
            # Floating endpoints — defer until the AP1 measurement-lead
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
            branches.append(_DistributedBranch(
                node_a=ra, node_b=rb,
                R=db.R, p_a=db.p_a, p_b=db.p_b,
                wire_radius=db.wire_radius, inductive=db.inductive,
            ))
            if db.node_a not in cluster_id:
                interior_nodes.add(db.node_a)
            if db.node_b not in cluster_id:
                interior_nodes.add(db.node_b)
    return seg_list, branches, interior_nodes


# ---------------------------------------------------------------------
# Potential evaluation
# ---------------------------------------------------------------------


def _potential_kernel(
    field_points: np.ndarray,  # (M, 3)
    source_points: np.ndarray,  # (N, 3)
    currents: np.ndarray,       # (N,)
    rho: float,
) -> np.ndarray:
    """Vectorised image-charge sum — pure point-source evaluation.

    Used for **field points away from the sources** (plots, profiles).
    Singularities are clamped at ``_MIN_DISTANCE``.

    Returns
    -------
    phi : np.ndarray, shape (M,)
        Potential at every field point.
    """
    image_points = source_points.copy()
    image_points[:, 2] = -image_points[:, 2]

    diff_real = field_points[:, None, :] - source_points[None, :, :]
    diff_image = field_points[:, None, :] - image_points[None, :, :]
    r_real = np.linalg.norm(diff_real, axis=2)
    r_image = np.linalg.norm(diff_image, axis=2)
    np.maximum(r_real, _MIN_DISTANCE, out=r_real)
    np.maximum(r_image, _MIN_DISTANCE, out=r_image)

    kernel = (1.0 / r_real) + (1.0 / r_image)
    phi = (rho / (4.0 * np.pi)) * (kernel @ currents)
    return phi


def _self_corrected_kernel(
    seg_points: np.ndarray,    # (N, 3)
    seg_lengths: np.ndarray,   # (N,)
    wire_radii: np.ndarray,    # (N,)
    currents: np.ndarray,      # (N,)
    rho: float,
) -> np.ndarray:
    """Evaluation **at the segment midpoints** with proper self-action.

    The diagonal (segment onto itself) uses the analytical line
    self-potential; off-diagonal entries fall back to the point-source
    approximation 1/r + 1/r_image.

    Returns
    -------
    phi : np.ndarray, shape (N,)
        Potential at the segment midpoints.
    """
    n = seg_points.shape[0]
    image_points = seg_points.copy()
    image_points[:, 2] = -image_points[:, 2]

    diff_real = seg_points[:, None, :] - seg_points[None, :, :]
    diff_image = seg_points[:, None, :] - image_points[None, :, :]
    r_real = np.linalg.norm(diff_real, axis=2)
    r_image = np.linalg.norm(diff_image, axis=2)

    # Off-diagonal: point source, safely clamped.
    np.maximum(r_real, _MIN_DISTANCE, out=r_real)
    np.maximum(r_image, _MIN_DISTANCE, out=r_image)
    kernel = (1.0 / r_real) + (1.0 / r_image)

    # Replace the diagonal with the line self-potential of the direct
    # contribution.
    # phi_self_direct = rho · I / (2π·L) · ln(L/a)
    #   ⇒ K_ii (direct part) = 2·ln(L/a) / L
    diag_direct = 2.0 * np.log(seg_lengths / wire_radii) / seg_lengths

    # Self-image contribution: point source at (x, y, -z), distance 2·z.
    z_mid = seg_points[:, 2]
    # If the segment lies exactly at z=0, the image contribution falls
    # into the _MIN_DISTANCE cutoff via the point-source formula. This
    # is acceptable because shallow electrodes are an edge case anyway.
    diag_image = 1.0 / np.maximum(2.0 * np.abs(z_mid), _MIN_DISTANCE)

    np.fill_diagonal(kernel, diag_direct + diag_image)

    phi = (rho / (4.0 * np.pi)) * (kernel @ currents)
    return phi


# ---------------------------------------------------------------------
# Cluster building and current sharing
# ---------------------------------------------------------------------


def _build_clusters(electrodes, conductors) -> dict[str, str]:
    """Union-find on electrodes: returns the cluster root per name.

    Only **ideal** conductors (``Conductor.is_ideal()`` returns
    ``True``, i.e. ``cross_section is None`` or the resulting series
    resistance is below the ideal-resistance threshold) are treated
    as galvanic shorts and merge their two end electrodes into one
    cluster.

    Finite-impedance conductors are *not* used for clustering — they
    enter the solver later as branches in the nodal-analysis system
    (see :func:`_solve_cluster_currents` and the module docstring).

    Conductors with purely geometric end-points (no
    ``start_electrode`` / ``end_electrode``) are ignored.
    """
    parent = {e.name: e.name for e in electrodes}

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
        if getattr(c, "is_distributed", False) and not distributed_as_lumped:
            # Routed through _build_distributed_topology instead.
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
    """
    if finite_branches is None:
        finite_branches = []
    if pseudo_owners is None:
        pseudo_owners = []

    # Cluster-level input currents (sum over electrodes of the cluster).
    cluster_input: dict[str, complex] = {}
    for ename, ic in elec_input_current.items():
        if ic == 0j:
            continue
        cluster_input.setdefault(cluster_id[ename], 0j)
        cluster_input[cluster_id[ename]] += ic

    elec_total = {e.name: 0j for e in electrodes}
    for pn in pseudo_owners:
        elec_total[pn] = 0j
    if not cluster_input and not finite_branches:
        return elec_total

    # ------------------------------------------------------------------
    # Active node set: source nodes plus every node transitively
    # reachable through finite branches.
    # ------------------------------------------------------------------
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

    if not active_clusters_set:
        return elec_total

    active_clusters = sorted(active_clusters_set)
    cluster_idx = {c: k for k, c in enumerate(active_clusters)}
    K_a = len(active_clusters)

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
        n for n in all_owners
        if cluster_id[n] in active_clusters_set
        and len(elec_to_segidx.get(n, [])) > 0
    ]
    N_a = len(active_elecs)

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
    M_a = len(active_branches)

    # ------------------------------------------------------------------
    # Multi-port grounding matrix Z[i, j]: average potential at
    # electrode i for 1 A injected (uniform per unit length) at
    # electrode j. Built one column at a time.
    # ------------------------------------------------------------------
    Z = np.zeros((N_a, N_a))
    n_segments = seg_points.shape[0]
    for j, name_j in enumerate(active_elecs):
        idxs_j = elec_to_segidx[name_j]
        test_currents = np.zeros(n_segments)
        L_j = seg_lengths[idxs_j].sum()
        test_currents[idxs_j] = seg_lengths[idxs_j] / L_j  # uniform, Σ = 1 A
        phi_test = self_kernel(
            seg_points, seg_lengths, wire_radii, test_currents
        )
        for i, name_i in enumerate(active_elecs):
            idxs_i = elec_to_segidx[name_i]
            Z[i, j] = float(phi_test[idxs_i].mean())

    # ------------------------------------------------------------------
    # Augmented linear system
    #   [ Z      -C       0    ] [ I_e    ]   [ 0       ]
    #   [ C^T    0        B^T  ] [ phi_n  ] = [ I_in    ]
    #   [ 0      B        R_b  ] [ I_b    ]   [ 0       ]
    # ------------------------------------------------------------------
    n_unknowns = N_a + K_a + M_a
    A = np.zeros((n_unknowns, n_unknowns))

    # Block 1: Z · I_e − C · phi_n = 0
    A[:N_a, :N_a] = Z
    elec_cluster = [cluster_idx[cluster_id[name]] for name in active_elecs]
    for i, k in enumerate(elec_cluster):
        A[i, N_a + k] = -1.0  # −C

    # Block 2: KCL per node:  C^T · I_e  +  B^T · I_b  =  I_in
    for k, c in enumerate(active_clusters):
        for i, name in enumerate(active_elecs):
            if cluster_id[name] == c:
                A[N_a + k, i] = 1.0  # C^T
    for m, (a, b, _R) in enumerate(active_branches):
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
    for m, (a, b, R) in enumerate(active_branches):
        ka = cluster_idx[a]
        kb = cluster_idx[b]
        A[N_a + K_a + m, N_a + ka] = +1.0
        A[N_a + K_a + m, N_a + kb] = -1.0
        A[N_a + K_a + m, N_a + K_a + m] = -R

    # ------------------------------------------------------------------
    # Right-hand side and solve.
    #
    # Two paths:
    #   (a) DC / no inductive coupling — Z and R_b are real, so we
    #       solve the real and imaginary parts of a complex source
    #       independently with the same factorisation. This keeps
    #       the historic fast path unchanged.
    #   (b) Inductive coupling active (``inductance_matrix is not
    #       None`` and ``omega != 0``) — the branch block becomes
    #       complex (Z_b = R + jωL), so we solve one complex linear
    #       system. The off-diagonal mutual-inductance entries M_{ij}
    #       go into the same Block 3 row that already carries the
    #       diagonal R + jωL_self contribution.
    # ------------------------------------------------------------------
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
        sol_re = np.linalg.solve(A, b_re)
        sol_im = np.linalg.solve(A, b_im)
        I_active = sol_re[:N_a] + 1j * sol_im[:N_a]
    else:
        # Restrict the (full-finite-branches) inductance matrix to the
        # active subset; both ``finite_branches`` and
        # ``inductance_matrix`` must use the same ordering.
        n_total_branches = inductance_matrix.shape[0]
        if inductance_matrix.shape != (n_total_branches, n_total_branches):
            raise ValueError(
                f"inductance_matrix must be square, got "
                f"{inductance_matrix.shape}."
            )
        if n_total_branches != len(finite_branches):
            raise ValueError(
                f"inductance_matrix size {n_total_branches} does not "
                f"match number of finite_branches ({len(finite_branches)})."
            )
        L_active = inductance_matrix[
            np.ix_(active_branch_indices, active_branch_indices)
        ]
        # ADR-0005: optional Carson correction. Pre-restricted to
        # the active subset by the same index permutation as L.
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
        # Augment Block 3 with the inductive contribution. The
        # diagonal already carries −R; we add −jω·L_self to it and
        # plug −jω·L_{m,m'} into the off-diagonal positions of
        # Block 3 row m, branch-current column m'. With Carson on,
        # the Carson dZ matrix is already the *full* per-pair
        # impedance contribution (per-unit-length value times the
        # parallel-projection length), so it adds with sign −1.
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

    for i, name in enumerate(active_elecs):
        elec_total[name] = complex(I_active[i])

    return elec_total


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_image(world: "World", engine: "Engine") -> FieldResult:
    """Image-charge solver for homogeneous soil.

    Parameters
    ----------
    world
        World whose ``soil`` must be a :class:`HomogeneousSoil`.
    engine
        Engine configuration; relevant fields are ``segment_length``
        and ``frequencies``.
    """
    if not isinstance(world.soil, HomogeneousSoil):
        raise TypeError(
            "Backend 'image' requires HomogeneousSoil. "
            f"Got: {type(world.soil).__name__}. "
            "For layered models pick backend='image_2layer' "
            "(Tagg/Sunde) or backend='mom' (planned)."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    rho = world.soil.resistivity
    ds = engine.segment_length

    # 1) Discretisation of the electrodes
    all_segments: list[_Segment] = []
    elec_to_segidx: dict[str, list[int]] = {}
    for e in world.electrodes:
        segs = _discretize_electrode(e, ds)
        elec_to_segidx[e.name] = list(range(len(all_segments),
                                            len(all_segments) + len(segs)))
        all_segments.extend(segs)

    # 2) Per-electrode input currents from the configured sources
    elec_input_current: dict[str, complex] = {
        e.name: 0j for e in world.electrodes
    }
    for src in world.sources:
        if src.kind != "current":
            # Voltage sources are ignored in this simple image model.
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in elec_input_current:
            elec_input_current[src.attached_to] += i_complex

    # 3) Cluster building: electrodes joined by an *ideal* conductor
    #    share a common potential. Conductors with a finite series
    #    resistance enter the linear system as branches of the
    #    nodal-analysis solver instead.
    cluster_id = _build_clusters(world.electrodes, world.conductors)
    finite_branches = _build_finite_branches(world.conductors, cluster_id)

    # 3b) Distributed-conductor topology (ADR-0003).
    #     Conductors with a finite ``discretize_segment_length`` are
    #     split into sub-pieces; ``coupling_to_soil="galvanic"`` adds
    #     midpoint pseudo-electrode segments to the Z-matrix, while
    #     the longitudinal-segment chain is appended to the branch
    #     list. Pseudo-nodes get one-element entries in
    #     ``elec_to_segidx`` and are flagged as their own clusters.
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
    # Anonymous interior nodes from isolated distributed conductors
    # (no leakage segment, only KCL participation) are also flagged
    # as standalone clusters.
    for n in interior_nodes:
        if n not in cluster_id:
            cluster_id[n] = n
            pseudo_owners.append(n)
            elec_to_segidx[n] = []  # no segment, no leakage
    n_lumped_branches = len(finite_branches)
    distributed_branch_tuples = [
        (db.node_a, db.node_b, db.R) for db in distributed_branches_objs
    ]
    finite_branches = list(finite_branches) + distributed_branch_tuples

    # ADR-0004 + ADR-0005: assemble the partial-inductance matrix
    # for the active distributed-conductor branches (lumped branches
    # stay purely resistive and contribute zero entries). When
    # ``engine.earth_inductive_model == "carson_series"`` we also
    # receive a closure that builds dZ_carson(omega) per frequency.
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
    seg_points = np.array([s.midpoint for s in all_segments])  # (N, 3)
    seg_lengths = np.array([s.length for s in all_segments])    # (N,)

    # 4) Current sharing within clusters via the multi-port matrix
    wire_radii = np.array([s.wire_radius for s in all_segments])

    def _homogeneous_self(seg_pts, seg_lens, wr, currents):
        """Closure: homogeneous self-action with fixed rho."""
        return _self_corrected_kernel(seg_pts, seg_lens, wr, currents, rho)

    n_freq = len(engine.frequencies)
    omegas = [2.0 * np.pi * float(f) for f in engine.frequencies]
    real_electrode_names = {e.name for e in world.electrodes}

    def _solve_at(omega: float) -> tuple[
        dict[str, complex], np.ndarray, np.ndarray
    ]:
        """Solve once at a given angular frequency.

        Returns
        -------
        elec_total : dict
            Per-owner total leakage current.
        seg_currents : np.ndarray
            Per-segment current distribution (uniform per unit length).
        phi_at_segments : np.ndarray
            Potential at every segment midpoint.
        """
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
            self_kernel=_homogeneous_self,
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
            phi_re = _self_corrected_kernel(
                seg_points, seg_lengths, wire_radii, sc.real, rho,
            )
            phi_im = _self_corrected_kernel(
                seg_points, seg_lengths, wire_radii, sc.imag, rho,
            )
            ph = phi_re + 1j * phi_im
        return elec_total, sc, ph

    # Frequency loop. With no inductive coupling the system is
    # frequency-independent, so we solve once and replicate.
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

    # Build the FieldResult mappings.
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

    # 7) Point-source list for post-processing (plots, profiles)
    point_sources = [
        PointSource(
            position=tuple(seg_points[i].tolist()),
            current=[complex(sc_per_freq[k][i]) for k in range(n_freq)],
            electrode_name=all_segments[i].electrode_name,
            length=float(seg_lengths[i]),
        )
        for i in range(n_segments)
    ]

    # Cluster map: electrode_name -> sorted list of cluster members
    # (only real electrodes are surfaced).
    cluster_members: dict[str, list[str]] = {}
    for ename in real_electrode_names:
        cluster_members[ename] = sorted(
            n for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_electrode_names
        )

    metadata: dict = {
        "world_name": world.name,
        "n_segments": n_segments,
        "segment_length": ds,
        "stub": False,
        "earth_inductive_model": earth_inductive_model,
    }
    # ADR-0005 §"Eindringtiefen-Diagnostik": expose the
    # electromagnetic skin depth in soil at every solved frequency,
    # so notebooks and benchmarks can answer "is my geometry small
    # or large compared to delta(omega)?" without re-deriving the
    # formula. Only active for engines that ran a frequency loop.
    if has_inductance and sigma_earth_for_carson is not None:
        from groundfield.coupling.carson import skin_depth

        metadata["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma_earth_for_carson)
            for f in engine.frequencies
        }
    elif has_inductance and isinstance(world.soil, HomogeneousSoil):
        # No Carson active, but homogeneous soil — still useful as a
        # *reference* skin depth, even though the perfect-mirror
        # solver does not actually use it.
        from groundfield.coupling.carson import skin_depth

        sigma_ref = 1.0 / float(world.soil.resistivity)
        metadata["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma_ref)
            for f in engine.frequencies
        }
    if conductor_currents:
        metadata["conductor_node_currents"] = conductor_currents
        metadata["conductor_node_potentials"] = conductor_potentials

    return FieldResult(
        backend="image",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=float(rho),
        soil=world.soil,
        clusters=cluster_members,
        metadata=metadata,
    )
