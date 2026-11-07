"""Discretisation of electrodes and distributed conductors (private).

Turns the geometric electrode models of :mod:`groundfield.geometry`
and the distributed conductors (ADR-0003) into the leakage segments
and longitudinal branches the image-family solvers assemble.

Moved out of :mod:`groundfield.solver.image` (review pass 10, 10.75 /
10.79); every name stays importable from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from groundfield.geometry.electrodes import (
    GridMeshElectrode,
    MeshElectrode,
    PolylineElectrode,
    RingElectrode,
    RodElectrode,
    StarElectrode,
    StripElectrode,
)
from groundfield.solver._guards import _require_thin_wire

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.geometry.electrodes import _ElectrodeBase

__all__: list[str] = []


# Frequently reused unit tangents (vertical rods, axis-aligned grid
# wires). Kept immutable so the shared instances cannot be mutated
# through a ``_Segment``.
def _frozen(v: "list[float]") -> np.ndarray:
    arr = np.array(v, dtype=float)
    arr.flags.writeable = False
    return arr


_TANGENT_Z = _frozen([0.0, 0.0, 1.0])
_TANGENT_X = _frozen([1.0, 0.0, 0.0])
_TANGENT_Y = _frozen([0.0, 1.0, 0.0])


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
    direction
        Unit tangent ``(dx, dy, dz)`` of the segment axis, shape
        ``(3,)``. Only ``|dz|`` is consumed so far, by
        :func:`_check_segment_depths`, to reconstruct the segment's
        vertical *extent* ``z_mid ± L·|dz|/2`` — a midpoint-only depth
        check accepts a rod that pokes out of the ground (audit
        2026-07-29). ``None`` means "orientation not recorded"; the
        depth check then degrades to the midpoint rather than assuming
        one. Every discretiser in this module sets it.
    """

    midpoint: np.ndarray  # shape (3,)
    length: float
    electrode_name: str
    wire_radius: float
    conductor_name: str | None = None
    direction: np.ndarray | None = None
    # ADR-0007: which soil layer the segment lives in (0 = upper,
    # 1 = next, ..., n-1 = bottom semi-infinite). Set by the
    # discretiser when ``layer_interfaces`` is supplied; left at 0
    # for homogeneous-soil callers.
    layer_index: int = 0
    # ADR-0012 V2: per-segment radial shell coefficient
    # $C = \rho_c/(2\pi)\,\ln(r_b/r_a)$ in Ω·m. The MoM diagonal
    # entry for this segment is augmented by $C / \text{length}$
    # (the radial voltage drop through the concrete shell). Left
    # at ``0.0`` for segments that are not part of a concrete-
    # encased foundation electrode.
    concrete_shell_coefficient_ohm_m: float = 0.0


# ---------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------


def _split_interval_at_interfaces(
    t0: float,
    t1: float,
    interfaces,
) -> list[tuple[float, float]]:
    """Partition ``[t0, t1]`` at every interior interface value.

    ADR-0007 step 1 (audit 2026-07-08, WP-D2): segments must not
    straddle a soil-layer interface, otherwise the discretiser
    assigns the whole segment to the layer of its midpoint and the
    kernel evaluates up to ``ds/2`` of wire length with the wrong
    layer resistivity.
    """
    cuts = sorted(t for t in (interfaces or ()) if t0 + 1e-12 < t < t1 - 1e-12)
    bounds = [t0, *cuts, t1]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def _discretize_rod(
    electrode: RodElectrode,
    ds: float,
    layer_interfaces=None,
) -> list[_Segment]:
    """Vertical driven rod into N equally long segments per layer.

    With ``layer_interfaces`` given, the rod's z-interval is first
    split at every interface it crosses; each part is then
    discretised independently, so no segment straddles an interface.
    """
    x0, y0, z0 = electrode.position
    segs: list[_Segment] = []
    parts = _split_interval_at_interfaces(
        z0,
        z0 + electrode.length,
        layer_interfaces,
    )
    for z_lo, z_hi in parts:
        part_len = z_hi - z_lo
        n = max(1, int(np.ceil(part_len / ds)))
        seg_len = part_len / n
        _require_thin_wire(seg_len, electrode.wire_radius, electrode.name)
        for k in range(n):
            zc = z_lo + (k + 0.5) * seg_len
            segs.append(
                _Segment(
                    midpoint=np.array([x0, y0, zc], dtype=float),
                    length=seg_len,
                    electrode_name=electrode.name,
                    wire_radius=electrode.wire_radius,
                    direction=_TANGENT_Z,
                )
            )
    return segs


def _discretize_ring(electrode: RingElectrode, ds: float) -> list[_Segment]:
    """Horizontal ring into N equally long arc segments."""
    perimeter = 2.0 * np.pi * electrode.radius
    n = max(8, int(np.ceil(perimeter / ds)))
    seg_len = perimeter / n
    _require_thin_wire(seg_len, electrode.wire_radius, electrode.name)
    cx, cy, cz = electrode.center
    segs: list[_Segment] = []
    for k in range(n):
        phi = 2.0 * np.pi * (k + 0.5) / n
        segs.append(
            _Segment(
                midpoint=np.array(
                    [
                        cx + electrode.radius * np.cos(phi),
                        cy + electrode.radius * np.sin(phi),
                        cz,
                    ],
                    dtype=float,
                ),
                length=seg_len,
                electrode_name=electrode.name,
                wire_radius=electrode.wire_radius,
                # Horizontal arc chord: |e_z| = 0.
                direction=np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=float),
            )
        )
    return segs


def _discretize_strip(
    electrode: StripElectrode,
    ds: float,
    layer_interfaces=None,
) -> list[_Segment]:
    """Straight strip into N equally long segments along its axis.

    A sloped strip whose z-range crosses a layer interface is split
    there first (ADR-0007 step 1) — horizontal strips are unaffected.
    """
    L = electrode.length
    p0 = np.array(electrode.start, dtype=float)
    p1 = np.array(electrode.end, dtype=float)
    direction = (p1 - p0) / L
    # Map interface depths to axis parameters t in [0, L].
    t_cuts: list[float] = []
    dz = p1[2] - p0[2]
    if layer_interfaces and abs(dz) > 1e-12:
        for h in layer_interfaces:
            t = (h - p0[2]) / dz * L
            t_cuts.append(t)
    segs: list[_Segment] = []
    # ADR-0012 V2: read the concrete-shell coefficient from the
    # electrode and propagate to every segment. ``0.0`` (default)
    # leaves the diagonal untouched.
    shell_coeff = float(electrode.concrete_shell_coefficient_ohm_m)
    for t_lo, t_hi in _split_interval_at_interfaces(0.0, L, t_cuts):
        part_len = t_hi - t_lo
        n = max(1, int(np.ceil(part_len / ds)))
        seg_len = part_len / n
        _require_thin_wire(seg_len, electrode.wire_radius, electrode.name)
        for k in range(n):
            midpoint = p0 + (t_lo + (k + 0.5) * seg_len) * direction
            segs.append(
                _Segment(
                    midpoint=midpoint,
                    length=seg_len,
                    electrode_name=electrode.name,
                    wire_radius=electrode.wire_radius,
                    concrete_shell_coefficient_ohm_m=shell_coeff,
                    direction=direction,
                )
            )
    return segs


def _discretize_polyline(
    electrode: "PolylineElectrode",
    ds: float,
    layer_interfaces=None,
) -> list[_Segment]:
    """Polyline (open or closed) — one strip-style chain per edge.

    Every edge is discretised exactly like :func:`_discretize_strip`
    (same segment-length convention, same thin-wire guard, same
    per-segment concrete-shell coefficient), so a closed rectangular
    polyline reproduces the four perimeter wires of the equivalent
    ``grid_mesh`` / strip-chain foundation.

    Degenerate edges (zero length, e.g. from duplicated polygon
    vertices) are skipped rather than raising — OSM footprints
    routinely carry repeated points.
    """
    shell_coeff = float(electrode.concrete_shell_coefficient_ohm_m)
    segs: list[_Segment] = []
    for start, end in electrode.edges:
        p0 = np.array(start, dtype=float)
        p1 = np.array(end, dtype=float)
        L = float(np.linalg.norm(p1 - p0))
        if L <= 1e-12:
            continue
        direction = (p1 - p0) / L
        n = max(1, int(np.ceil(L / ds)))
        seg_len = L / n
        _require_thin_wire(seg_len, electrode.wire_radius, electrode.name)
        for k in range(n):
            segs.append(
                _Segment(
                    midpoint=p0 + (k + 0.5) * seg_len * direction,
                    length=seg_len,
                    electrode_name=electrode.name,
                    wire_radius=electrode.wire_radius,
                    concrete_shell_coefficient_ohm_m=shell_coeff,
                    direction=direction,
                )
            )
    if not segs:
        raise ValueError(
            f"PolylineElectrode {electrode.name!r} discretised to zero "
            "segments — all edges are degenerate."
        )
    return segs


def _discretize_star(
    electrode: "StarElectrode",
    ds: float,
    layer_interfaces=None,
) -> list[_Segment]:
    """n-arm star (*Sternerder*) — one strip-style chain per radial arm.

    Every arm runs from the shared centre node to its tip and is
    discretised exactly like :func:`_discretize_strip` (same
    segment-length convention, thin-wire guard, per-segment concrete-
    shell coefficient). All arms belong to one electrode, so the shared
    centre makes them a single galvanic cluster.
    """
    shell_coeff = float(electrode.concrete_shell_coefficient_ohm_m)
    segs: list[_Segment] = []
    for start, end in electrode.edges:
        p0 = np.array(start, dtype=float)
        p1 = np.array(end, dtype=float)
        L = float(np.linalg.norm(p1 - p0))
        direction = (p1 - p0) / L
        n = max(1, int(np.ceil(L / ds)))
        seg_len = L / n
        _require_thin_wire(seg_len, electrode.wire_radius, electrode.name)
        for k in range(n):
            segs.append(
                _Segment(
                    midpoint=p0 + (k + 0.5) * seg_len * direction,
                    length=seg_len,
                    electrode_name=electrode.name,
                    wire_radius=electrode.wire_radius,
                    concrete_shell_coefficient_ohm_m=shell_coeff,
                    direction=direction,
                )
            )
    return segs


def _segments_per_wire(span: float, ds: float, n_spans: int) -> int:
    """Segments per grid wire, aligned to the crossing pattern.

    A wire of length ``span`` is crossed at ``n_spans + 1`` points (the
    perpendicular wires), dividing it into ``n_spans`` equal mesh spans.
    The requested target is ``ceil(span / ds)`` segments; this rounds it
    **up to the next multiple of** ``n_spans`` so every span holds a whole
    number of segments and the crossings coincide with segment endpoints,
    never with the midpoint point sources. With one interior span or none
    (``n_spans <= 1``) there is nothing to align and the plain count is
    returned. The result is never coarser than requested.
    """
    n = max(1, int(np.ceil(span / ds)))
    if n_spans <= 1:
        return n
    return int(np.ceil(n / n_spans)) * n_spans


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

    # Segments per wire, aligned so grid crossings fall on segment
    # *endpoints* rather than midpoints. Longitudinal wires (span dx) are
    # crossed by the ``ny_wires`` transverse wires, i.e. ``ny_wires - 1``
    # mesh spans; making the segment count a multiple of that span count
    # places one or more whole segments in each span, so no segment
    # midpoint (the point-source location) ever coincides with a crossing.
    # Coincident midpoints of a longitudinal and a transverse segment
    # would otherwise land within the singularity clamp and inflate the
    # grid resistance by ~2x for the uniform-current ``image`` backend (a
    # silent grid-discretisation trap; the current-solving backends are
    # far less sensitive but pay the same clamp warning).
    n_long = _segments_per_wire(dx, ds, ny_wires - 1)
    n_trans = _segments_per_wire(dy, ds, nx_wires - 1)

    segs: list[_Segment] = []
    # Longitudinal wires (along x for each y)
    seg_len = dx / n_long
    _require_thin_wire(seg_len, wire_radius, electrode_name)
    for y in ys:
        for k in range(n_long):
            xm = cx + (k + 0.5) * seg_len
            segs.append(
                _Segment(
                    midpoint=np.array([xm, y, cz], dtype=float),
                    length=seg_len,
                    electrode_name=electrode_name,
                    wire_radius=wire_radius,
                    direction=_TANGENT_X,
                )
            )
    # Transverse wires (along y for each x)
    seg_len = dy / n_trans
    _require_thin_wire(seg_len, wire_radius, electrode_name)
    for x in xs:
        for k in range(n_trans):
            ym = cy + (k + 0.5) * seg_len
            segs.append(
                _Segment(
                    midpoint=np.array([x, ym, cz], dtype=float),
                    length=seg_len,
                    electrode_name=electrode_name,
                    wire_radius=wire_radius,
                    direction=_TANGENT_Y,
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
        cx=cx,
        cy=cy,
        cz=cz,
        dx=dx,
        dy=dy,
        nx_wires=ny,
        ny_wires=nx,
        ds=ds,
        electrode_name=electrode.name,
        wire_radius=electrode.wire_radius,
    )


def _discretize_grid_mesh(electrode: GridMeshElectrode, ds: float) -> list[_Segment]:
    """Mesh earth electrode with explicit n_x × n_y meshes.

    The grid has ``n_x + 1`` transverse and ``n_y + 1`` longitudinal
    wires (one wire per cell boundary plus the outer perimeter).
    """
    cx, cy, cz = electrode.corner
    dx, dy = electrode.size
    return _grid_segments(
        cx=cx,
        cy=cy,
        cz=cz,
        dx=dx,
        dy=dy,
        nx_wires=electrode.n_y + 1,
        ny_wires=electrode.n_x + 1,
        ds=ds,
        electrode_name=electrode.name,
        wire_radius=electrode.wire_radius,
    )


def _discretize_electrode(
    electrode: "_ElectrodeBase",
    ds: float,
    layer_interfaces=None,
) -> list[_Segment]:
    """Dispatch to the per-primitive discretiser.

    Parameters
    ----------
    layer_interfaces
        Optional sequence of soil-layer interface depths in m
        (e.g. ``(h_1,)`` for a two-layer soil). Primitives whose
        geometry can cross an interface (rods, sloped strips) split
        their segments there so no segment straddles a layer boundary
        (ADR-0007 step 1, audit 2026-07-08 WP-D2). Horizontal
        primitives (rings, meshes) are unaffected.
    """
    if isinstance(electrode, RodElectrode):
        return _discretize_rod(electrode, ds, layer_interfaces)
    if isinstance(electrode, RingElectrode):
        return _discretize_ring(electrode, ds)
    if isinstance(electrode, StripElectrode):
        return _discretize_strip(electrode, ds, layer_interfaces)
    if isinstance(electrode, PolylineElectrode):
        return _discretize_polyline(electrode, ds, layer_interfaces)
    if isinstance(electrode, StarElectrode):
        return _discretize_star(electrode, ds, layer_interfaces)
    if isinstance(electrode, GridMeshElectrode):
        return _discretize_grid_mesh(electrode, ds)
    if isinstance(electrode, MeshElectrode):
        return _discretize_mesh(electrode, ds)
    raise TypeError(f"Image backend does not know {type(electrode).__name__}.")


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
                    direction=direction,
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
            branches.append(
                _DistributedBranch(
                    node_a=chain_nodes[k],
                    node_b=chain_nodes[k + 1],
                    R=Rs[k],
                    p_a=pa,
                    p_b=pb,
                    wire_radius=conductor.wire_radius,
                    inductive=inductive,
                )
            )
    else:
        # ``isolated``: no leakage along the wire. Interior nodes are
        # anonymous; the chain is a series of n branches between the
        # two end clusters. The interior nodes do not appear in the
        # Z-matrix but participate in KCL (every interior node carries
        # zero leakage by construction).
        interior_nodes = [f"__cond_{conductor.name}__node_{k}" for k in range(1, n)]
        nodes_chain = [start_node, *interior_nodes, end_node]
        # Geometric endpoints follow the conductor axis at the
        # sub-segment boundaries.
        node_positions = [p0 + (k * seg_len) * direction for k in range(n + 1)]
        for k in range(n):
            branches.append(
                _DistributedBranch(
                    node_a=nodes_chain[k],
                    node_b=nodes_chain[k + 1],
                    R=R_seg,
                    p_a=node_positions[k],
                    p_b=node_positions[k + 1],
                    wire_radius=conductor.wire_radius,
                    inductive=inductive,
                )
            )

    return segments, branches
