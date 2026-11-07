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

from groundfield.geometry.electrodes import PolylineElectrode
from groundfield.geometry.primitives import (
    _segments_per_wire as _segments_per_wire,
)
from groundfield.geometry.primitives import (
    _split_interval_at_interfaces as _split_interval_at_interfaces,
)
from groundfield.solver._guards import _require_thin_wire

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.geometry.electrodes import _ElectrodeBase

__all__: list[str] = []


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


def _discretize_electrode(
    electrode: "_ElectrodeBase",
    ds: float,
    layer_interfaces=None,
) -> list[_Segment]:
    """Leakage segments of one electrode.

    The geometry comes from the electrode model itself
    (:meth:`~groundfield.geometry.electrodes.RodElectrode.segments` and
    siblings, see :mod:`groundfield.geometry.primitives` for the
    conventions); this function attaches the solver attributes (owner,
    wire radius, concrete-shell coefficient) and enforces the thin-wire
    guard.

    Parameters
    ----------
    electrode
        Electrode model.
    ds
        Target segment length in m.
    layer_interfaces
        Optional sequence of soil-layer interface depths in m
        (e.g. ``(h_1,)`` for a two-layer soil). Primitives whose
        geometry can cross an interface (rods, sloped strips) split
        their segments there so no segment straddles a layer boundary
        (ADR-0007 step 1, audit 2026-07-08 WP-D2). Horizontal
        primitives (rings, meshes) are unaffected.

    Raises
    ------
    TypeError
        If the electrode model does not provide a discretisation.
    ValueError
        If a segment is not longer than the wire radius
        (:func:`_require_thin_wire`), or a polyline has no
        non-degenerate edge.
    """
    try:
        geo = electrode.segments(ds, layer_interfaces)
    except NotImplementedError:
        raise TypeError(
            f"Image backend does not know {type(electrode).__name__}."
        ) from None
    if isinstance(electrode, PolylineElectrode) and len(geo) == 0:
        raise ValueError(
            f"PolylineElectrode {electrode.name!r} discretised to zero "
            "segments — all edges are degenerate."
        )
    wire_radius = electrode.wire_radius
    thin = geo.lengths <= wire_radius
    if thin.any():
        _require_thin_wire(
            float(geo.lengths[int(np.argmax(thin))]), wire_radius, electrode.name
        )
    # ADR-0012 V2: strips, polylines and stars carry a concrete-shell
    # coefficient that every segment inherits; ``0.0`` leaves the
    # reaction-matrix diagonal untouched.
    shell_coeff = float(getattr(electrode, "concrete_shell_coefficient_ohm_m", 0.0))
    return [
        _Segment(
            midpoint=geo.midpoints[k],
            length=float(geo.lengths[k]),
            electrode_name=electrode.name,
            wire_radius=wire_radius,
            direction=geo.directions[k],
            concrete_shell_coefficient_ohm_m=shell_coeff,
        )
        for k in range(len(geo))
    ]


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
