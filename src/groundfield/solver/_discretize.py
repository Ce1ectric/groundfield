"""Discretisation of electrodes and distributed conductors (private).

Turns the geometric electrode models of :mod:`groundfield.geometry`
and the distributed conductors (ADR-0003) into the leakage segments
and longitudinal branches the image-family solvers assemble.

Moved out of :mod:`groundfield.solver.image` (review pass 10, 10.75 /
10.79); every name stays importable from there.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    # Straight wire of the electrode the segment lies on
    # (:attr:`~groundfield.geometry.primitives.WireSegments.wires`): a wire
    # line of a grid, an arm of a star, an edge of a polyline. The image
    # family gives every wire its own owner (user decision Q30).
    wire: int = 0


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
        primitives (rings, meshes) are unaffected. A part beyond an
        interface keeps its own segment however short it is (audit
        round 2, A2S-01).

    Raises
    ------
    TypeError
        If the electrode model does not provide a discretisation.
    ValueError
        If a segment of the discretisation without the interface cuts
        is not longer than the wire radius (:func:`_require_thin_wire`),
        or a polyline has no non-degenerate edge.
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
    if thin.any() and layer_interfaces:
        # A part cut off at a layer interface keeps its own segment however
        # short it is: the wire touches that layer, and the Galerkin self
        # term of every segment-based kernel is valid for any length (audit
        # round 2, A2S-01). The guard concerns the chosen segment length,
        # i.e. the discretisation without the interface cuts.
        regular = electrode.segments(ds).lengths
        thin, geo_lengths = regular <= wire_radius, regular
    else:
        geo_lengths = geo.lengths
    if thin.any():
        _require_thin_wire(
            float(geo_lengths[int(np.argmax(thin))]), wire_radius, electrode.name
        )
    # ADR-0012 V2: strips, polylines and stars carry a concrete-shell
    # coefficient that every segment inherits; ``0.0`` leaves the
    # reaction-matrix diagonal untouched.
    shell_coeff = float(getattr(electrode, "concrete_shell_coefficient_ohm_m", 0.0))
    wires = geo.wire_index
    return [
        _Segment(
            midpoint=geo.midpoints[k],
            length=float(geo.lengths[k]),
            electrode_name=electrode.name,
            wire_radius=wire_radius,
            direction=geo.directions[k],
            concrete_shell_coefficient_ohm_m=shell_coeff,
            wire=int(wires[k]),
        )
        for k in range(len(geo))
    ]


# ---------------------------------------------------------------------
# Owners of an electrode: one per (straight wire, soil layer) — review
# pass 10, 10.15 (F02); user decisions Q30 and Q44 of 2026.11
# ---------------------------------------------------------------------

#: Name prefix of every sub-owner of an electrode (a part in one soil layer
#: or one straight wire). Reserved like the ``__cond_`` prefix of the
#: conductor pseudo-nodes (:data:`groundfield.world.RESERVED_NAME_PREFIXES`),
#: so a sub-owner cannot collide with a user name.
_SUB_OWNER_PREFIX = "__layer_"

#: Whether the image family gives every straight wire of an electrode its own
#: owner (user decision Q30). Private: :func:`_one_owner_per_wire_off`
#: switches it off to rebuild the one-owner grids, stars and polylines of the
#: releases before 2026.11 for comparisons (tests and evidence only).
_SPLIT_WIRES = True


@contextmanager
def _one_owner_per_wire_off() -> Iterator[None]:
    """Solve grids, stars and polylines as one owner, as before Q30.

    A private comparison tool (tests and evidence of the numbers that the
    user decision Q30 changed); not thread-safe.
    """
    global _SPLIT_WIRES
    previous = _SPLIT_WIRES
    _SPLIT_WIRES = False
    try:
        yield
    finally:
        _SPLIT_WIRES = previous


def _layer_owner_name(electrode_name: str, layer: int) -> str:
    """Name of the sub-owner of ``electrode_name`` in soil layer ``layer``.

    The ``__layer_`` prefix is reserved like the ``__cond_`` prefix of
    the conductor pseudo-nodes, so it cannot collide with a user name.
    """
    return f"{_SUB_OWNER_PREFIX}{electrode_name}__L{layer}"


def _wire_owner_name(electrode_name: str, wire: int, layer: int | None = None) -> str:
    """Name of the sub-owner of ``electrode_name`` on its straight wire ``wire``.

    Carries the reserved prefix of the electrode parts
    (:data:`_SUB_OWNER_PREFIX`); ``layer`` is appended for an electrode
    whose wires reach several soil layers.
    """
    suffix = "" if layer is None else f"__L{layer}"
    return f"{_SUB_OWNER_PREFIX}{electrode_name}__W{wire}{suffix}"


@dataclass
class _OwnerSplit:
    """Owners per (electrode, straight wire, soil layer) of the image family.

    The image family carries one uniform current density per owner. Two
    splits make the owners finer than the electrode:

    * **Straight wires** (user decisions Q30 and Q44 of 2026.11). A grid
      (:class:`~groundfield.geometry.electrodes.GridMeshElectrode`,
      :class:`~groundfield.geometry.electrodes.MeshElectrode`) gets one
      owner per full wire line, a star one per arm, a polyline one per
      straight run (one per side of a building outline; a vertex where
      the polyline continues straight on, such as the crossings of a
      generated mesh wire, starts no new owner); rods, strips and rings
      stay one owner. One density over all wires of a
      grid or star made it 3.9 % (20 m × 20 m grid, 4 × 4 meshes), 5.3 %
      (40 m × 40 m, 8 × 8) and 4.3 % (4-arm star) higher than the
      per-segment ``mom`` solution, because the current crowds to the
      outer wires and the tips; with one density per wire the current
      may differ between the wires, as the generated mesh foundations
      (one electrode per wire line since 2026.11) already did.
    * **Soil layers** (review pass 10, 10.15 / F02). An electrode that
      crosses a soil-layer interface (a rod through :math:`h_1`) gets one
      owner per layer it reaches: the leakage density scales roughly with
      :math:`1/\\rho` of the layer. Up to 0.15.0 the whole rod carried one
      density, which put a 12 m rod through :math:`h_1 = 5` m in
      1000/30 Ω·m soil 566 % above the per-segment ``mom`` solution.

    All owners of an electrode belong to its cluster (ideally bonded: one
    potential). The electrode keeps its name for the owner of its first
    segment; every other (wire, layer) group forms a sub-owner
    (:func:`_wire_owner_name`, :func:`_layer_owner_name`), a pseudo-owner
    of the nodal system attached to the electrode's cluster. After the
    solve, :meth:`merge` restores one owner per electrode: its current is
    the sum over its owners, its segment list the original one, so the
    reported potential is the length-weighted average over the whole
    electrode (equal to the cluster potential). The split is internal:
    every result is reported per electrode.

    Attributes
    ----------
    sub_owners
        Sub-owner names per split electrode (the electrode itself is not
        included).
    segments
        Original segment indices per split electrode.
    n_layers
        Number of soil layers per electrode split by layer (the
        ``layer_owners`` metadata of the result).
    n_wires
        Number of straight wires per electrode split by wire (the
        ``wire_owners`` metadata of the result).
    """

    sub_owners: dict[str, list[str]]
    segments: dict[str, list[int]]
    n_layers: dict[str, int] = field(default_factory=dict)
    n_wires: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.sub_owners)

    def owners_of(self, electrode_name: str) -> list[str]:
        """The electrode and its sub-owners."""
        return [electrode_name, *self.sub_owners.get(electrode_name, [])]

    def segments_of(self, electrode_name: str, owner_segments: dict) -> list[int]:
        """All segment indices of an electrode (split or not)."""
        if electrode_name in self.segments:
            return self.segments[electrode_name]
        return owner_segments[electrode_name]

    def merge(
        self,
        disc,
        topo,
        owner_currents: list[dict[str, complex]],
        node_currents: dict[str, complex] | None = None,
    ) -> None:
        """Restore one owner per electrode after the solve (in place).

        Parameters
        ----------
        disc, topo
            The discretisation and topology passed to :meth:`split`.
        owner_currents
            Owner currents per frequency; the same mapping may appear
            several times (frequency-independent solves) and is merged
            once.
        node_currents
            Impressed current per node in A. With it, a source-free
            cluster without branches that holds a split electrode reports
            electrode currents that sum to exactly zero (in the member
            order of :attr:`FieldResult.clusters`), as every other floating
            cluster does (:func:`~groundfield.solver._topology._zero_net_current`):
            the sum over the owners of an electrode reorders the
            additions, so the residue of the solve is moved to the last
            member again.
        """
        for name, subs in self.sub_owners.items():
            disc.owner_segments[name] = self.segments[name]
            for sub in subs:
                del disc.owner_segments[sub]
                topo.cluster_id.pop(sub, None)
                topo.pseudo_owners.remove(sub)
        floating = self._floating_members(disc, topo, node_currents)
        seen: set[int] = set()
        for currents in owner_currents:
            if id(currents) in seen:
                continue
            seen.add(id(currents))
            for name, subs in self.sub_owners.items():
                total = currents.get(name, 0j)
                for sub in subs:
                    total += currents.pop(sub, 0j)
                currents[name] = total
            for members in floating:
                if len(members) == 1:
                    # a cluster of its own: exactly zero (not the -0 of the
                    # negated empty sum; audit W1A-07)
                    currents[members[0]] = 0j
                    continue
                rest = sum(currents.get(m, 0j) for m in members[:-1])
                currents[members[-1]] = -complex(rest)

    def _floating_members(
        self, disc, topo, node_currents: dict[str, complex] | None
    ) -> list[list[str]]:
        """Sorted members of the source-free clusters without branches
        that contain a split electrode (see :meth:`merge`)."""
        if node_currents is None or not self.sub_owners:
            return []
        cluster_id = topo.cluster_id
        driven = {cluster_id[n] for n, ic in node_currents.items() if ic != 0j}
        linked = {a for a, _b, _R in topo.finite_branches}
        linked |= {b for _a, b, _R in topo.finite_branches}
        roots = {cluster_id[name] for name in self.sub_owners}
        out: list[list[str]] = []
        for root in sorted(roots - driven - linked):
            out.append(
                sorted(
                    n
                    for n, idxs in disc.owner_segments.items()
                    if idxs and cluster_id.get(n) == root
                )
            )
        return out

    @classmethod
    def split(
        cls,
        disc,
        topo,
        electrode_names,
        layer_interfaces=None,
        *,
        wires: bool | None = None,
    ) -> _OwnerSplit:
        """Give every electrode one owner per (straight wire, soil layer).

        Modifies ``disc.owner_segments``, ``topo.cluster_id`` and
        ``topo.pseudo_owners`` in place; electrodes of one wire inside one
        layer (rods, strips and rings in one layer) are left untouched
        (bit-identical results).

        Parameters
        ----------
        disc
            Discretisation with ``segments`` and ``owner_segments``
            (:class:`~groundfield.solver._pipeline.Discretisation`).
        topo
            Topology with ``cluster_id`` and ``pseudo_owners``
            (:class:`~groundfield.solver._pipeline.Topology`).
        electrode_names
            Names of the real electrodes.
        layer_interfaces
            Interface depths in m at which electrodes get one owner per
            layer; ``None`` or empty splits by wire only.
        wires
            Split by straight wire (user decision Q30); ``None`` (default)
            takes :data:`_SPLIT_WIRES` (``True``).

        Returns
        -------
        _OwnerSplit
            The split, for :meth:`merge` after the solve.

        Raises
        ------
        ValueError
            If a sub-owner name is already taken (cannot happen for user
            names, which may not carry the reserved prefix).
        """
        owners = cls(sub_owners={}, segments={})
        if wires is None:
            wires = _SPLIT_WIRES
        bounds = None
        if layer_interfaces:
            bounds = np.sort(np.asarray(layer_interfaces, dtype=float))
        if bounds is None and not wires:
            return owners
        for name in electrode_names:
            idxs = disc.owner_segments.get(name, [])
            if len(idxs) < 2:
                continue
            n = len(idxs)
            if wires:
                wire = [disc.segments[i].wire for i in idxs]
            else:
                wire = [0] * n
            if bounds is not None:
                z = np.array([disc.segments[i].midpoint[2] for i in idxs], dtype=float)
                # Segments are split at the interfaces, so the midpoint names
                # the layer of the whole segment (z == h belongs to the layer
                # below, as in the kernels).
                layer = np.searchsorted(bounds, z, side="right").tolist()
            else:
                layer = [0] * n
            keys = list(zip(wire, layer, strict=True))
            groups = sorted(set(keys))
            if len(groups) == 1:
                continue
            first = keys[0]
            owners.segments[name] = list(idxs)
            disc.owner_segments[name] = [
                i for i, key in zip(idxs, keys, strict=True) if key == first
            ]
            several_layers = len(set(layer)) > 1
            subs = []
            for key in groups:
                if key == first:
                    continue
                w, lv = key
                if w == first[0]:
                    sub = _layer_owner_name(name, lv)
                else:
                    sub = _wire_owner_name(name, w, lv if several_layers else None)
                if sub in disc.owner_segments or sub in topo.cluster_id:
                    raise ValueError(
                        f"Internal owner name {sub!r} of electrode {name!r} is "
                        "already taken."
                    )
                disc.owner_segments[sub] = [
                    i for i, kj in zip(idxs, keys, strict=True) if kj == key
                ]
                topo.cluster_id[sub] = topo.cluster_id[name]
                topo.pseudo_owners.append(sub)
                subs.append(sub)
            owners.sub_owners[name] = subs
            if several_layers:
                owners.n_layers[name] = len(set(layer))
            if len(set(wire)) > 1:
                owners.n_wires[name] = len(set(wire))
        return owners


#: Name of 2026.11 (10.15); the split is by wire and by layer since Q30.
_LayerOwners = _OwnerSplit


# ---------------------------------------------------------------------
# Conductor discretisation (ADR-0003: distributed conductor model)
# ---------------------------------------------------------------------


@dataclass
class _DistributedBranch:
    """One longitudinal-segment branch of a distributed conductor.

    Carries the topology used by the nodal-analysis system
    (``node_a``, ``node_b``, ``R``) plus the geometric information
    required by the inductive-coupling assembly (ADR-0004): the
    branch endpoints in 3-D, the wire radius and whether the conductor is
    a cable shield (``conductor_type="cable_shield"``: a thinner wire on
    its axis is a core inside it, and its self term has no internal
    inductance of a solid wire; user decisions Q32 and Q33), and the name
    of the conductor (the pieces of two conductors lying beside each other
    along one axis couple at the on-axis distance; audit W1A-02).
    """

    node_a: str
    node_b: str
    R: float
    p_a: np.ndarray  # shape (3,)
    p_b: np.ndarray  # shape (3,)
    wire_radius: float
    inductive: bool  # True if the parent conductor enables Neumann coupling
    shield: bool = False  # True for conductor_type == "cable_shield"
    conductor: str | None = None  # name of the parent conductor (audit W1A-02)


def _conductor_node_name(conductor_name: str, k: int) -> str:
    """Pseudo-node identifier for the *k*-th midpoint segment of a
    distributed conductor.

    The naming convention `__cond_{name}__seg_{k}` is reserved — the
    leading double underscore guarantees no collision with user
    electrode names.
    """
    return f"__cond_{conductor_name}__seg_{k}"


def _is_conductor_chain(conductor) -> bool:
    """Whether the solver represents ``conductor`` as a chain of sub-pieces.

    A distributed conductor (finite ``discretize_segment_length``) is a
    chain when it has more than one sub-piece, or when it is galvanic: a
    single leaking sub-piece is the coarsest discretisation of the same
    leaking wire (one midpoint pseudo-node, two half-resistance stubs),
    not a leakage-free lumped branch (review pass 10, 10.24, galvanic
    half). An isolated single sub-piece is exactly one series branch and
    stays lumped. The one predicate of discretisation, topology, the
    inductive assembly and the diagnostics.
    """
    if not getattr(conductor, "is_distributed", False):
        return False
    return int(conductor.n_segments) > 1 or (
        getattr(conductor, "coupling_to_soil", "isolated") == "galvanic"
    )


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

    A galvanic conductor with a single sub-piece (``n == 1``) gives one
    pseudo-node at the conductor midpoint between two ``R/2`` stubs; an
    isolated one stays a lumped branch (see :func:`_is_conductor_chain`).

    Parameters
    ----------
    conductor
        :class:`Conductor` instance with finite
        ``discretize_segment_length`` (i.e. ``is_distributed``).

    Returns
    -------
    segments : list[_Segment]
        Pseudo-electrode segments at the conductor midpoints. Empty
        for ``coupling_to_soil == "isolated"`` and for conductors that
        are no chain (lumped fallback, handled by the caller).
    branches : list[(str, str, float)]
        Longitudinal-branch list ``(node_in, node_out, R)``. For an
        isolated conductor the chain runs through anonymous interior
        nodes ``f"__cond_{name}__node_{k}"`` (k=1..n-1).
    """
    n = conductor.n_segments
    L = conductor.length
    if not _is_conductor_chain(conductor):
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
    shield = bool(getattr(conductor, "is_shield", False))

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
                    shield=shield,
                    conductor=conductor.name,
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
                    shield=shield,
                    conductor=conductor.name,
                )
            )

    return segments, branches
