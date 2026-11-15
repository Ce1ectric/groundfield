"""Shared solve pipeline of the image family (private).

``solve_image`` (homogeneous soil) and ``solve_image_2layer`` (two-layer
soil, Tagg/Sunde series) run through one pipeline that differs only in
the Green's-function object handed to :func:`solve_image_family`
(review pass 10, 10.75). The steps, in order:

1. :func:`discretise` — electrodes into leakage segments
   (:meth:`~groundfield.geometry.electrodes.RodElectrode.segments` and
   siblings), split at the soil-layer interfaces the Green's function
   asks for. The predicted segment count is checked against
   :data:`groundfield.diagnostics.HARD_LIMIT` first (10.06).
2. :func:`inject` — the one shared injection step (10.16, 10.66): the
   world is validated (:meth:`World.check
   <groundfield.world.World.check>`), then every current source adds
   $+I$ at ``attached_to`` and, with ``return_to``, $-I$ at
   ``return_to``. Returns the impressed current per node name —
   electrodes and junctions — which the topology step attaches to the
   clusters. :func:`impressed_voltages` lists the voltage sources
   (2026.11), which step 5 adds to the nodal system by modified nodal
   analysis: one unknown (the source current) and one constraint row
   $\\varphi_a - \\varphi_r = U$ per source.
3. :func:`build_topology` — clusters of ideally bonded electrodes and
   junctions, lumped finite-impedance branches and the
   distributed-conductor chains (ADR-0003), whose galvanic segments
   join the leakage segments as one-segment pseudo-owners. Junctions
   are nodes without leakage segments; a junction subnetwork without
   soil contact is rejected here.

   3b. :func:`split_owners` — one uniform-current owner per straight
   wire of every electrode (a wire line of a grid, an arm of a star, a
   straight run of a polyline; user decisions Q30 and Q44 of 2026.11)
   and, for electrodes through a layer interface, per soil layer
   (10.15). The owners of an electrode are ideally bonded and merged
   back into one per electrode before step 6.
4. :func:`assemble_branches` — the frequency-dependent branch block:
   partial inductances (ADR-0004) and the earth-return correction
   closure (Carson series ADR-0005, Sommerfeld ADR-0006).
   **Extension point (COUP):** solver-side inductive fixes (image
   baselines, Neumann inductance of lumped / single-segment
   conductors, skin effect) belong here.
5. :func:`solve` — per frequency, the augmented nodal system of
   :mod:`groundfield.solver._topology` with the multi-port matrix built
   from the Green's function's reaction kernel and the rows of the
   voltage sources; source-free components
   float with zero net current (10.29); then the segment currents
   (uniform per unit length per owner), the segment potentials and the
   potential of every active node (:attr:`Solution.node_potentials`).
6. :func:`build_result` — node potentials (length-weighted, F10/F11),
   point sources, clusters and metadata as a
   :class:`~groundfield.solver.result.FieldResult`; junction potentials
   in ``metadata["junction_potentials"]``.

The Green's-function object (:class:`ImageGreen`) supplies the backend
name, the layer interfaces, the reaction kernel and the
backend-specific metadata. **Extension point (ENG):** the ``mom``
family can be ported by providing a Green's-function object whose
``solve`` replaces the uniform-current owner model with per-segment
currents; steps 1–4 and 6 are shared as they are. A backend that runs
steps 2 and 3 honours ``return_to`` and junctions and should declare
``two_terminal_sources`` and ``junctions`` in its registry
capabilities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from groundfield.solver._discretize import _discretize_electrode, _OwnerSplit
from groundfield.solver._guards import _check_segment_depths
from groundfield.solver._inductive import _assemble_inductance_matrix
from groundfield.solver._line_kernel import LineSegments
from groundfield.solver._topology import (
    VoltageTerminals,
    _build_clusters,
    _build_distributed_topology,
    _build_finite_branches,
    _check_soil_contact,
    _solve_cluster_currents,
)
from groundfield.solver.result import FieldResult, PointSource

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from groundfield.solver._discretize import _DistributedBranch, _Segment
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__: list[str] = []

#: Reaction kernel ``(seg_points, seg_lengths, wire_radii, currents) -> phi``:
#: segment-midpoint potentials in V for segment currents in A (``currents``
#: of shape ``(N,)`` or ``(N, k)``).
SelfKernel = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


class ImageGreen(Protocol):
    """Soil-specific part of the image-family pipeline.

    Attributes
    ----------
    backend
        Backend label stored in :attr:`FieldResult.backend`.
    where
        Caller label used in warning and error messages.
    layer_interfaces
        Interface depths in m at which the discretisation splits wires,
        or ``None``.
    soil_resistivity
        Value stored in :attr:`FieldResult.soil_resistivity` in Ω·m.
    reference_resistivity
        Resistivity in Ω·m of the reference skin depth reported for
        inductive worlds without an earth-return conductivity.
    """

    backend: str
    where: str
    layer_interfaces: tuple[float, ...] | None
    soil_resistivity: float
    reference_resistivity: float

    def prepare(
        self, segments: list[_Segment], arrays: SegmentArrays, *, stacklevel: int
    ) -> SelfKernel:
        """Return the reaction kernel for the discretised geometry."""
        ...

    def metadata(self, solution: Solution, *, stacklevel: int) -> dict:
        """Backend-specific metadata (may warn about the solve)."""
        ...


# ---------------------------------------------------------------------
# Step containers
# ---------------------------------------------------------------------


@dataclass
class Discretisation:
    """Leakage segments and their owners (steps 1 and 3).

    Attributes
    ----------
    segments
        Leakage segments: electrode segments first (world order), then
        the galvanic segments of distributed conductors.
    owner_segments
        Segment indices per owner: electrodes, conductor pseudo-nodes
        (one segment each) and leakage-free interior nodes (empty).
    """

    segments: list[_Segment]
    owner_segments: dict[str, list[int]]


@dataclass
class Topology:
    """Network topology (step 3).

    Attributes
    ----------
    cluster_id
        Cluster root per node name.
    finite_branches
        ``(node_a, node_b, R)`` per finite branch: the lumped conductors
        first, then the distributed-conductor chains.
    distributed_branches
        Geometry of the distributed-conductor branches (inductive
        assembly).
    n_lumped_branches
        Number of lumped branches at the head of ``finite_branches``.
    pseudo_owners
        Conductor pseudo-nodes taking part in the nodal system.
    """

    cluster_id: dict[str, str]
    finite_branches: list[tuple[str, str, float]]
    distributed_branches: list[_DistributedBranch]
    n_lumped_branches: int
    pseudo_owners: list[str]


@dataclass
class BranchModel:
    """Frequency-dependent branch block (step 4).

    Attributes
    ----------
    earth_inductive_model
        Engine setting the block was assembled for.
    inductance_matrix
        Partial-inductance matrix over all finite branches in H, or
        ``None`` when the branch block is frequency independent. It is
        all zero when only the skin effect makes the block frequency
        dependent (conductors with ``resistance_model="skin"`` and no
        ``inductance_model``).
    earth_correction
        ``omega -> ΔZ(omega)`` over all finite branches in Ω: the
        earth-return correction of the inductive branches
        (``carson_series``, ``sommerfeld``) plus the skin-effect
        correction on the diagonal of ``resistance_model="skin"``
        branches (review pass 10, 10.37), or ``None`` without either.
    sigma_earth
        Earth conductivity in S/m used by ``carson_series``, else
        ``None``.
    """

    earth_inductive_model: str
    inductance_matrix: np.ndarray | None
    earth_correction: Callable[[float], np.ndarray] | None
    sigma_earth: float | None

    @property
    def has_inductance(self) -> bool:
        """Whether the branch block is frequency dependent.

        True with an inductive conductor and also with a skin-effect
        conductor alone (all-zero inductance matrix); the name predates
        the skin term (10.37).
        """
        return self.inductance_matrix is not None


@dataclass
class SegmentArrays:
    """Per-segment arrays consumed by the kernels.

    Attributes
    ----------
    points
        Segment midpoints in m, shape ``(N, 3)``.
    lengths
        Segment lengths in m, shape ``(N,)``.
    radii
        Wire radii in m, shape ``(N,)``.
    shell_coefficients
        Concrete-shell coefficients in Ω·m, shape ``(N,)`` (ADR-0012 V2).
    directions
        Unit tangents, shape ``(N, 3)``; ``None`` if a segment has no
        recorded orientation.
    """

    points: np.ndarray
    lengths: np.ndarray
    radii: np.ndarray
    shell_coefficients: np.ndarray
    directions: np.ndarray | None = None

    @classmethod
    def from_segments(cls, segments: list[_Segment]) -> SegmentArrays:
        """Stack the per-segment attributes."""
        directions = None
        if segments and all(s.direction is not None for s in segments):
            directions = np.array([s.direction for s in segments], dtype=float)
        return cls(
            points=np.array([s.midpoint for s in segments]),
            lengths=np.array([s.length for s in segments]),
            radii=np.array([s.wire_radius for s in segments]),
            shell_coefficients=np.array(
                [s.concrete_shell_coefficient_ohm_m for s in segments], dtype=float
            ),
            directions=directions,
        )

    def line_segments(self) -> LineSegments:
        """The segments as uniform line sources (``kernel="line"``).

        Raises
        ------
        ValueError
            If a segment has no recorded orientation.
        """
        if self.directions is None:
            raise ValueError(
                "kernel='line' needs the orientation of every segment; a "
                "segment without a direction was discretised."
            )
        return LineSegments(self.points, self.lengths, self.directions, self.radii)


@dataclass
class Solution:
    """Per-frequency result of step 5.

    Attributes
    ----------
    owner_currents
        Leakage current per owner in A, one mapping per frequency.
    segment_currents
        Segment currents in A, one ``(N,)`` vector per frequency.
    segment_potentials
        Segment-midpoint potentials in V, one ``(N,)`` vector per
        frequency.
    node_potentials
        Potential in V of every active node of the augmented system
        (cluster roots, junctions, interior conductor nodes), one
        mapping per frequency. Empty mappings for backends that do not
        provide them.
    voltage_source_currents
        Current in A of every voltage source (name -> current from the
        source into ``attached_to``), one mapping per frequency; empty
        for a world without voltage sources.
    """

    owner_currents: list[dict[str, complex]]
    segment_currents: list[np.ndarray]
    segment_potentials: list[np.ndarray] = field(default_factory=list)
    node_potentials: list[dict[str, complex]] = field(default_factory=list)
    voltage_source_currents: list[dict[str, complex]] = field(default_factory=list)


@dataclass(frozen=True)
class ImpressedVoltage:
    """One voltage source of the world, as the solve step reads it.

    Attributes
    ----------
    name
        Source name.
    attached_to
        Node (electrode or junction) at the potential $U$ above
        ``return_to``.
    return_to
        Second terminal, or ``None`` for remote earth.
    voltage
        Voltage phasor $U = |U|\\,e^{j\\varphi}$ in V.
    """

    name: str
    attached_to: str
    return_to: str | None
    voltage: complex

    def terminals(self, cluster_id: dict[str, str]) -> VoltageTerminals:
        """``(root_a, root_r, U)``: the cluster roots of the two terminals."""
        root_r = None if self.return_to is None else cluster_id[self.return_to]
        return (cluster_id[self.attached_to], root_r, self.voltage)


# ---------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------


def discretise(
    world: World,
    engine: Engine,
    layer_interfaces: Sequence[float] | None = None,
) -> Discretisation:
    """Step 1: discretise every electrode of ``world``.

    The predicted segment count is checked against
    :data:`groundfield.diagnostics.HARD_LIMIT` first, from the geometry
    alone (finding 10.06), so a unit error fails before anything of
    the problem size is allocated.

    Parameters
    ----------
    world
        World with at least one electrode.
    engine
        Engine; ``segment_length`` is the target segment length in m.
    layer_interfaces
        Interface depths in m at which wires are split.

    Raises
    ------
    groundfield.diagnostics.ResourceLimitError
        If the predicted segment count exceeds the hard limit.

    Warns
    -----
    InterfaceContactWarning
        If a rod head or tip lies within one wire radius of one of the
        ``layer_interfaces`` (a limit of the thin-wire model, audit
        A3S-05).
    """
    from groundfield.diagnostics import check_segment_budget
    from groundfield.solver._guards import _warn_rod_ends_near_interfaces

    check_segment_budget(world, engine, layer_interfaces)
    _warn_rod_ends_near_interfaces(world.electrodes, layer_interfaces)
    ds = engine.segment_length
    segments: list[_Segment] = []
    owner_segments: dict[str, list[int]] = {}
    for e in world.electrodes:
        if layer_interfaces is None:
            segs = _discretize_electrode(e, ds)
        else:
            segs = _discretize_electrode(e, ds, layer_interfaces=layer_interfaces)
        owner_segments[e.name] = list(range(len(segments), len(segments) + len(segs)))
        segments.extend(segs)
    return Discretisation(segments=segments, owner_segments=owner_segments)


def inject(world: World, *, where: str, stacklevel: int) -> dict[str, complex]:
    """Step 2: impressed current per node in A (complex phasor).

    The one shared injection step of every backend on the pipeline
    (review findings 10.16 and 10.66; up to 0.15.0 seven backends
    carried a copy of this loop, none of which read ``return_to``).

    Contract
    --------
    * The world is validated first (:meth:`World.check
      <groundfield.world.World.check>`): every source terminal names an
      electrode or junction, the world has at least one source of
      non-zero magnitude, no current source has magnitude 0, no
      two-terminal source is short-circuited by an ideal bond and the
      voltage sources form no loop. Errors are raised before any
      numerics run.
    * Voltage sources impress no current; :func:`impressed_voltages`
      lists them for the solve step.
    * A current source with phasor
      $I = |I|\\,e^{j\\varphi}$ adds $+I$ to ``attached_to`` and, if
      ``return_to`` is set, $-I$ to ``return_to``. The sum over all
      nodes is therefore the net current returning through remote
      earth: zero for a world of two-terminal sources only, so
      Kirchhoff's current law closes every measurement loop.
    * The mapping has one entry per electrode and junction (world
      order, electrodes first); nodes without a source carry ``0j``. Contributions are accumulated in
      source order, so a world without ``return_to`` gets exactly the
      values of 0.15.0.

    Parameters
    ----------
    world
        The world to be solved.
    where
        Caller label for messages (kept for the step signature).
    stacklevel
        ``stacklevel`` of warnings emitted here (kept for the step
        signature).

    Returns
    -------
    dict[str, complex]
        Impressed current per node name in A.

    Raises
    ------
    KeyError, ValueError, NotImplementedError
        See :meth:`World.check <groundfield.world.World.check>`.
    """
    world.check()
    _warn_manual_counter_sources(world)
    node_currents: dict[str, complex] = {e.name: 0j for e in world.electrodes}
    for junction in world.junctions:
        node_currents[junction.name] = 0j
    for src in world.sources:
        if src.kind != "current":
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        node_currents[src.attached_to] += i_complex
        if src.return_to is not None:
            node_currents[src.return_to] -= i_complex
    return node_currents


def impressed_voltages(world: World) -> list[ImpressedVoltage]:
    """Step 2b: the voltage sources of ``world``, in source order.

    Each becomes one unknown (its current) and one constraint row
    $\\varphi_a - \\varphi_r = U$ of the nodal system of step 5
    (modified nodal analysis, 2026.11). Called after :func:`inject`,
    which has validated the world.

    Parameters
    ----------
    world
        The world to be solved.

    Returns
    -------
    list[ImpressedVoltage]
        One entry per :class:`~groundfield.sources.VoltageSource`; empty
        for a world of current sources (the solve is then that of the
        releases before 2026.11, bit for bit).
    """
    return [
        ImpressedVoltage(
            name=src.name,
            attached_to=src.attached_to,
            return_to=src.return_to,
            voltage=complex(src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))),
        )
        for src in world.sources
        if src.kind == "voltage"
    ]


def voltage_terminal_nodes(voltages: Sequence[ImpressedVoltage]) -> set[str]:
    """Node names of the terminals of ``voltages`` (driven nodes)."""
    nodes = {v.attached_to for v in voltages}
    nodes |= {v.return_to for v in voltages if v.return_to is not None}
    return nodes


def _warn_manual_counter_sources(world: World) -> None:
    """Warn about the 0.15.0 workaround for the inert ``return_to``.

    Before ``return_to`` was honoured (finding 10.16), a measurement
    loop was closed by hand with a second source of the same magnitude
    and a phase shifted by 180° at the return electrode. Together with
    an honoured ``return_to`` that source counts the return current
    twice (``-2 I`` at the auxiliary electrode). The pattern — a
    one-terminal source at the ``return_to`` node of a two-terminal
    source with equal magnitude and opposite phase — is reported once
    per solve.
    """
    from groundfield.solver.registry import warn_external

    two_terminal = [s for s in world.sources if s.return_to is not None]
    for src in two_terminal:
        for other in world.sources:
            if other is src or other.return_to is not None:
                continue
            if other.attached_to != src.return_to:
                continue
            if not np.isclose(abs(other.magnitude), abs(src.magnitude), rtol=1e-9):
                continue
            i_src = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
            i_other = other.magnitude * np.exp(1j * np.deg2rad(other.phase_deg))
            if abs(i_src + i_other) > 1e-9 * abs(i_src):
                continue
            warn_external(
                f"Source {other.name!r} at {other.attached_to!r} looks like the "
                f"manual counter-source of source {src.name!r} (same magnitude, "
                "opposite phase). Since 2026.11 Source.return_to is honoured, so "
                f"{src.name!r} already takes its current back at "
                f"{src.return_to!r} and the return current is now counted "
                "twice. Remove the counter-source.",
                UserWarning,
            )


def _warn_concrete_shell_registry(world: World) -> None:
    """Deprecation of ``World.concrete_shell_corrections`` (ADR-0012 V1).

    The solver consumes concrete shells only as the distributed
    per-length coefficient of the electrodes; the lumped registry is
    read by no backend and by no ``solve_mutual_*`` call (finding 10.12).
    """
    if not world.concrete_shell_corrections:
        return
    from groundfield.solver.registry import DEPRECATION_REMOVAL, warn_external

    warn_external(
        "World.concrete_shell_corrections (the lumped ADR-0012 V1 registry, "
        f"{len(world.concrete_shell_corrections)} entries) is deprecated and "
        "is not read by the solver: concrete shells enter the solve only "
        "through the per-length coefficient concrete_shell_coefficient_ohm_m "
        "of the electrodes (ADR-0012 V2). Set that coefficient on the encased "
        "electrodes and clear the registry (world.reset_concrete_corrections()). "
        f"The field will be removed no earlier than {DEPRECATION_REMOVAL}.",
        DeprecationWarning,
    )


def build_topology(world: World, disc: Discretisation) -> Topology:
    """Step 3: clusters, finite branches and distributed conductors.

    Electrodes and junctions joined by an *ideal* conductor share one
    potential (cluster); conductors with a finite series resistance
    enter the nodal system as branches. Distributed conductors
    (ADR-0003) add a chain of branches; with
    ``coupling_to_soil="galvanic"`` every sub-piece also leaks through a
    one-segment pseudo-owner, which is appended to ``disc`` (its own
    cluster). Leakage-free interior nodes of isolated chains become
    clusters without segments.

    Junctions (:class:`~groundfield.geometry.junction.Junction`) are
    nodes without leakage segments: they appear in ``cluster_id`` and
    in the branch list, take part in Kirchhoff's current law, and never
    become owners of the multi-port matrix. A junction whose
    subnetwork has no soil contact raises :class:`ValueError` here,
    before any numerics run.
    """
    cluster_id = _build_clusters(
        world.electrodes, world.conductors, junctions=world.junctions
    )
    finite_branches = _build_finite_branches(world.conductors, cluster_id)
    cond_segs, distributed, interior_nodes = _build_distributed_topology(
        world.conductors, cluster_id
    )
    pseudo_owners: list[str] = []
    for s in cond_segs:
        pn = s.electrode_name
        disc.owner_segments[pn] = [len(disc.segments)]
        disc.segments.append(s)
        cluster_id[pn] = pn
        pseudo_owners.append(pn)
    # Anonymous interior nodes from isolated distributed conductors
    # (no leakage segment, only KCL participation) are also flagged
    # as standalone clusters.
    for n in interior_nodes:
        if n not in cluster_id:
            cluster_id[n] = n
            pseudo_owners.append(n)
            disc.owner_segments[n] = []  # no segment, no leakage
    n_lumped = len(finite_branches)
    finite_branches = list(finite_branches) + [
        (db.node_a, db.node_b, db.R) for db in distributed
    ]
    _check_soil_contact(
        [j.name for j in world.junctions],
        cluster_id,
        finite_branches,
        disc.owner_segments,
    )
    _warn_concrete_shell_registry(world)
    return Topology(
        cluster_id=cluster_id,
        finite_branches=finite_branches,
        distributed_branches=distributed,
        n_lumped_branches=n_lumped,
        pseudo_owners=pseudo_owners,
    )


def assemble_branches(world: World, engine: Engine, topo: Topology) -> BranchModel:
    """Step 4: inductance matrix and frequency-dependent branch correction.

    ADR-0004 + ADR-0005 + ADR-0006: the partial-inductance matrix covers
    every branch of a conductor with ``inductance_model == "neumann"`` —
    the distributed chains and, since review pass 10 (finding 10.24),
    lumped and single-segment conductors as one branch from start to
    end — built on the image baseline of the earth-return model (see
    :mod:`groundfield.solver._inductive`). The returned correction
    ``omega -> ΔZ(omega)`` holds the earth-return correction
    (``earth_inductive_model`` ``"carson_series"`` or ``"sommerfeld"``)
    and the skin-effect correction of conductors with
    ``resistance_model="skin"`` (finding 10.37; such a world is
    frequency dependent even without an inductance model). It is
    assembled for all frequencies of ``engine`` at once on first use
    (finding 10.60) and returns the same matrix for each frequency as a
    single-frequency assembly, bit for bit.
    """
    from groundfield.solver._inductive import branch_records

    model = engine.earth_inductive_model
    sigma_earth: float | None = None
    layered_earth: object = None
    if model == "carson_series":
        from groundfield.coupling import resolve_earth_conductivity

        sigma_earth = resolve_earth_conductivity(world.soil)
    elif model == "sommerfeld":
        from groundfield.coupling import resolve_earth_layers

        layered_earth = resolve_earth_layers(world.soil)
    lumped, skin = branch_records(
        world.conductors,
        cluster_id=topo.cluster_id,
        finite_branches=topo.finite_branches,
        distributed_branches=topo.distributed_branches,
        n_lumped_branches=topo.n_lumped_branches,
    )
    L_full, has_inductance, correction = _assemble_inductance_matrix(
        topo.distributed_branches,
        n_lumped_branches=topo.n_lumped_branches,
        n_total_branches=len(topo.finite_branches),
        earth_model=model,
        sigma_earth=sigma_earth,
        layered_earth=layered_earth,
        lumped_branches=lumped,
        skin_terms=skin,
        # The angular frequencies exactly as the solve step forms them.
        omegas=[2.0 * np.pi * float(f) for f in engine.frequencies],
        # Exact or tabulated Sommerfeld assembly (10.60).
        sommerfeld_method=engine.sommerfeld_assembly,
    )
    return BranchModel(
        earth_inductive_model=model,
        inductance_matrix=L_full if has_inductance else None,
        earth_correction=correction if has_inductance else None,
        sigma_earth=sigma_earth,
    )


def _segment_currents(
    owner_currents: dict[str, complex],
    owner_segments: dict[str, list[int]],
    seg_lengths: np.ndarray,
) -> np.ndarray:
    """Segment currents in A: each owner's current, uniform per unit length."""
    sc = np.zeros(seg_lengths.shape[0], dtype=complex)
    for ename, idxs in owner_segments.items():
        if not idxs:
            continue
        I_total = owner_currents.get(ename, 0j)
        if I_total == 0j:
            continue
        L_total = seg_lengths[idxs].sum()
        sc[idxs] = I_total * seg_lengths[idxs] / L_total
    return sc


def _segment_potentials(
    kernel: SelfKernel, arrays: SegmentArrays, sc_list: list[np.ndarray]
) -> list[np.ndarray]:
    """Segment-midpoint potentials in V for a list of segment-current vectors.

    All real and imaginary parts are stacked into one
    ``(n_segments, 2·len(sc_list))`` excitation so the kernel runs once
    for the whole frequency set (ADR-0010 Tier 1). The concrete-shell
    drop (ADR-0012 V2) is added exactly as in the multi-port assembly,
    so node potentials and cluster impedances reflect it.
    """
    n_segments = arrays.lengths.shape[0]
    k = len(sc_list)
    stacked = np.zeros((n_segments, 2 * k))
    for m, sc in enumerate(sc_list):
        stacked[:, m] = sc.real
        stacked[:, k + m] = sc.imag
    if not stacked.any():
        return [np.zeros(n_segments, dtype=complex)] * k
    phi = kernel(arrays.points, arrays.lengths, arrays.radii, stacked)
    shell_coeffs = arrays.shell_coefficients
    if np.any(shell_coeffs > 0.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            shell_diag = np.where(
                arrays.lengths > 0.0,
                shell_coeffs / arrays.lengths,
                0.0,
            )
        phi = phi + shell_diag[:, None] * stacked
    return [phi[:, m] + 1j * phi[:, k + m] for m in range(k)]


def _reused_potentials(
    mp_cache: dict, owner_list: list[dict[str, complex]], n_segments: int
) -> list[np.ndarray]:
    """Segment potentials from the stored ``K·E`` of the multi-port assembly.

    Every segment-current vector of the solve is ``E @ I_active`` (uniform
    per unit length per owner), so ``K @ sc = (K·E) @ I_active`` — the
    concrete-shell term is part of ``K·E`` already. Used when the
    reaction matrix was too large to keep (10.57); the result differs
    from a second kernel evaluation by BLAS round-off only.
    """
    phi_all = mp_cache["phi_all"]
    names = mp_cache["active_elecs"]
    out = []
    for owner_currents in owner_list:
        i_active = np.array([owner_currents.get(n, 0j) for n in names], dtype=complex)
        if not i_active.any():
            out.append(np.zeros(n_segments, dtype=complex))
            continue
        out.append(phi_all @ i_active.real + 1j * (phi_all @ i_active.imag))
    return out


def solve(
    world: World,
    engine: Engine,
    *,
    disc: Discretisation,
    node_currents: dict[str, complex],
    topo: Topology,
    branches: BranchModel,
    arrays: SegmentArrays,
    kernel: SelfKernel,
    voltages: Sequence[ImpressedVoltage] = (),
) -> Solution:
    """Step 5: owner currents, segment currents and segment potentials.

    The multi-port matrix ``Z`` is frequency-independent and shared by
    the per-frequency solves (ADR-0010 Tier 1). Without inductive
    branches the whole system is frequency-independent: it is solved
    once and replicated.

    Every voltage source of ``voltages`` adds one unknown, its current,
    and one constraint row $\\varphi_a - \\varphi_r = U$ (modified nodal
    analysis, :func:`~groundfield.solver._topology._voltage_source_entries`);
    the currents are kept in :attr:`Solution.voltage_source_currents`.

    Source-free components float (finding 10.29): a bonded group of
    passive electrodes is one equipotential with zero net current, and
    a passive network carries its equalising currents. The potential of
    every active node (including junctions) is kept in
    :attr:`Solution.node_potentials`.
    """
    n_freq = len(engine.frequencies)
    mp_cache: dict = {}
    terminals = [v.terminals(topo.cluster_id) for v in voltages]
    v_currents: list[dict[str, complex]] = []

    def _solve_at(
        omega: float,
    ) -> tuple[dict[str, complex], np.ndarray, dict[str, complex]]:
        correction = (
            branches.earth_correction(omega)
            if branches.earth_correction is not None
            else None
        )
        node_pots: dict[str, complex] = {}
        source_currents: list[complex] = []
        owner_currents = _solve_cluster_currents(
            electrodes=world.electrodes,
            elec_input_current=node_currents,
            cluster_id=topo.cluster_id,
            seg_points=arrays.points,
            seg_lengths=arrays.lengths,
            wire_radii=arrays.radii,
            elec_to_segidx=disc.owner_segments,
            self_kernel=kernel,
            finite_branches=topo.finite_branches,
            pseudo_owners=topo.pseudo_owners,
            omega=omega if branches.has_inductance else 0.0,
            inductance_matrix=branches.inductance_matrix,
            carson_correction=correction,
            shell_coefficients=arrays.shell_coefficients,
            multiport_cache=mp_cache,
            floating=True,
            node_potentials=node_pots,
            voltage_sources=terminals or None,
            source_currents=source_currents if terminals else None,
        )
        if terminals:
            v_currents.append(
                {v.name: i for v, i in zip(voltages, source_currents, strict=True)}
            )
        sc = _segment_currents(owner_currents, disc.owner_segments, arrays.lengths)
        return owner_currents, sc, node_pots

    def _potentials(
        owner_list: list[dict[str, complex]], sc_list: list[np.ndarray]
    ) -> list[np.ndarray]:
        # One reaction-matrix evaluation per solve (10.57): an operator
        # that keeps its matrix is simply applied again (the products
        # of v0.15.0, bit-identical); beyond the matrix budget the
        # segment potentials follow from the stored K·E of the
        # multi-port assembly instead of a second evaluation.
        if getattr(kernel, "materialised", True) or "phi_all" not in mp_cache:
            return _segment_potentials(kernel, arrays, sc_list)
        return _reused_potentials(mp_cache, owner_list, arrays.lengths.shape[0])

    if branches.has_inductance:
        owner_per_freq: list[dict[str, complex]] = []
        sc_per_freq: list[np.ndarray] = []
        nodes_per_freq: list[dict[str, complex]] = []
        for f in engine.frequencies:
            et, sc, nodes = _solve_at(2.0 * np.pi * float(f))
            owner_per_freq.append(et)
            sc_per_freq.append(sc)
            nodes_per_freq.append(nodes)
        phi_per_freq = _potentials(owner_per_freq, sc_per_freq)
    else:
        et, sc, nodes = _solve_at(0.0)
        ph = _potentials([et], [sc])[0]
        owner_per_freq = [et] * n_freq
        sc_per_freq = [sc] * n_freq
        phi_per_freq = [ph] * n_freq
        nodes_per_freq = [nodes] * n_freq
        v_currents = v_currents * n_freq
    return Solution(
        owner_currents=owner_per_freq,
        segment_currents=sc_per_freq,
        segment_potentials=phi_per_freq,
        node_potentials=nodes_per_freq,
        voltage_source_currents=v_currents,
    )


def build_result(
    world: World,
    engine: Engine,
    *,
    green: ImageGreen,
    disc: Discretisation,
    branches: BranchModel,
    arrays: SegmentArrays,
    solution: Solution,
    cluster_id: dict[str, str],
    metadata: dict,
) -> FieldResult:
    """Step 6: assemble the :class:`FieldResult`.

    Node potentials are the length-weighted (Galerkin) averages of the
    segment potentials — the reduction the multi-port matrix enforces
    (F10/F11). Conductor pseudo-nodes are reported in
    ``metadata["conductor_node_currents"]`` /
    ``["conductor_node_potentials"]``, junction potentials in
    ``metadata["junction_potentials"]`` (V per frequency, see
    :func:`_junction_potentials`).
    """
    n_freq = len(engine.frequencies)
    real_names = [e.name for e in world.electrodes]
    real_set = set(real_names)
    phi_per_freq = solution.segment_potentials
    electrode_potentials: dict[str, list[complex]] = {}
    electrode_currents: dict[str, list[complex]] = {}
    conductor_currents: dict[str, list[complex]] = {}
    conductor_potentials: dict[str, list[complex]] = {}
    for ename, idxs in disc.owner_segments.items():
        if not idxs:
            continue
        u_list = _weighted_node_potential(phi_per_freq, idxs, arrays.lengths, n_freq)
        i_list = [solution.owner_currents[k][ename] for k in range(n_freq)]
        if ename in real_set:
            electrode_potentials[ename] = u_list
            electrode_currents[ename] = i_list
        else:
            conductor_potentials[ename] = u_list
            conductor_currents[ename] = i_list

    sc_per_freq = solution.segment_currents
    directions = arrays.directions
    point_sources = [
        PointSource(
            position=tuple(arrays.points[i].tolist()),
            current=[complex(sc_per_freq[k][i]) for k in range(n_freq)],
            electrode_name=disc.segments[i].electrode_name,
            length=float(arrays.lengths[i]),
            direction=None if directions is None else tuple(directions[i].tolist()),
        )
        for i in range(len(disc.segments))
    ]

    # Cluster map: electrode name -> sorted cluster members (real
    # electrodes only), in world order.
    clusters: dict[str, list[str]] = {}
    for ename in real_names:
        clusters[ename] = sorted(
            n
            for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_set
        )

    metadata = dict(metadata)
    # ADR-0005 penetration-depth diagnostic: the electromagnetic skin
    # depth in soil at every solved frequency (inductive worlds only).
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
    if world.junctions:
        metadata["junction_potentials"] = _junction_potentials(
            world, disc, arrays, solution, cluster_id, n_freq
        )

    return FieldResult(
        backend=green.backend,
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=float(green.soil_resistivity),
        soil=world.soil,
        clusters=clusters,
        metadata=metadata,
        sources=solved_source_records(world, n_freq, solution),
    )


def solved_source_records(world: World, n_freq: int, solution: Solution) -> list:
    """Source snapshot with the solved currents of the voltage sources.

    A world with voltage sources reports the current of every voltage
    source from the solve (:attr:`Solution.voltage_source_currents`), so
    the result is self-contained also when a backend is called directly.
    A world of current sources returns ``[]``: :meth:`Engine.solve
    <groundfield.solver.engine.Engine.solve>` fills the snapshot it took
    before the dispatch, as before 2026.11.
    """
    if not solution.voltage_source_currents:
        return []
    from groundfield.solver.result import _source_records

    return _source_records(world.sources, n_freq, solution.voltage_source_currents)


def split_owners(
    world: World,
    disc: Discretisation,
    topo: Topology,
    owner_interfaces: Sequence[float] | None = None,
) -> _OwnerSplit:
    """Step 3b: one owner per straight wire and soil layer of every electrode.

    A grid gets one uniform-current owner per full wire line, a star one
    per arm, a polyline one per straight run (user decisions Q30 and Q44
    of 2026.11); with ``owner_interfaces`` an electrode through a layer
    interface gets one per layer (10.15). The owners of an electrode are
    ideally bonded (its cluster); :meth:`_OwnerSplit.merge` restores one
    owner per electrode after the solve. See
    :class:`~groundfield.solver._discretize._OwnerSplit`.
    """
    return _OwnerSplit.split(
        disc, topo, [e.name for e in world.electrodes], owner_interfaces
    )


def solve_image_family(
    world: World,
    engine: Engine,
    green: ImageGreen,
    *,
    stacklevel: int = 3,
    owner_interfaces: Sequence[float] | None = None,
) -> FieldResult:
    """Run the image-family pipeline for one Green's-function object.

    Parameters
    ----------
    world
        World with at least one electrode and a soil the Green's
        function accepts.
    engine
        Engine configuration (``segment_length``, ``frequencies``,
        ``earth_inductive_model``).
    green
        Soil-specific part (:class:`ImageGreen`).
    stacklevel
        ``stacklevel`` that makes the pipeline's warnings point at the
        caller of the backend entry point (``solve_image`` /
        ``solve_image_2layer``).
    owner_interfaces
        Interface depths in m at which an electrode gets one owner per
        soil layer (10.15; ``image_2layer`` passes them for worlds with an
        electrode through the interface, which then report
        ``metadata["layer_owners"]``). ``None``: one owner per straight
        wire only (:func:`split_owners`).

    Returns
    -------
    FieldResult
        Node potentials and currents per frequency, point sources and
        metadata.
    """
    disc = discretise(world, engine, green.layer_interfaces)
    node_currents = inject(world, where=green.where, stacklevel=stacklevel)
    voltages = impressed_voltages(world)
    topo = build_topology(world, disc)
    owners = split_owners(world, disc, topo, owner_interfaces)
    branches = assemble_branches(world, engine, topo)
    # Burial-depth validation (audit 2026-07-28, F26/F29): the galvanic
    # image construction needs a finite image separation 2 z >> a.
    # Reject surface-laid / airborne conductors and warn for the shallow
    # regime where the point-image stand-in biases the diagonal (point
    # kernel only, F08).
    _check_segment_depths(
        disc.segments,
        where=green.where,
        stacklevel=stacklevel + 1,
        kernel=engine.kernel,
    )
    arrays = SegmentArrays.from_segments(disc.segments)
    kernel = green.prepare(disc.segments, arrays, stacklevel=stacklevel + 1)
    solution = solve(
        world,
        engine,
        disc=disc,
        node_currents=node_currents,
        topo=topo,
        branches=branches,
        arrays=arrays,
        kernel=kernel,
        voltages=voltages,
    )
    if owners:
        owners.merge(
            disc,
            topo,
            solution.owner_currents,
            node_currents,
            driven_nodes=voltage_terminal_nodes(voltages),
        )
    metadata = {
        "world_name": world.name,
        "n_segments": len(disc.segments),
        "segment_length": engine.segment_length,
        **green.metadata(solution, stacklevel=stacklevel + 1),
        "stub": False,
        "earth_inductive_model": branches.earth_inductive_model,
        "kernel": engine.kernel,
    }
    if owner_interfaces is not None:
        metadata["layer_owners"] = dict(owners.n_layers)
    if owners.n_wires:
        # Owners per electrode with more than one straight wire (Q30).
        metadata["wire_owners"] = dict(owners.n_wires)
    return build_result(
        world,
        engine,
        green=green,
        disc=disc,
        branches=branches,
        arrays=arrays,
        solution=solution,
        cluster_id=topo.cluster_id,
        metadata=metadata,
    )


def _junction_potentials(
    world: World,
    disc: Discretisation,
    arrays: SegmentArrays,
    solution: Solution,
    cluster_id: dict[str, str],
    n_freq: int,
) -> dict[str, list[complex]]:
    """Potential in V of every junction, one value per frequency.

    A junction in an active node of the augmented system takes the
    node potential of the solution. A junction bonded by ideal
    conductors to a passive electrode that stays out of the system
    (a lone passive electrode, see
    :func:`~groundfield.solver._topology._floating_cluster_set`) shares
    that electrode's potential: the length-weighted average of the
    cluster's segment potentials.
    """
    out: dict[str, list[complex]] = {}
    for junction in world.junctions:
        root = cluster_id[junction.name]
        nodes = solution.node_potentials
        if nodes and all(root in nodes[k] for k in range(n_freq)):
            out[junction.name] = [complex(nodes[k][root]) for k in range(n_freq)]
            continue
        idxs = [
            i
            for name, seg in disc.owner_segments.items()
            if seg and cluster_id.get(name) == root
            for i in seg
        ]
        if not idxs:  # pragma: no cover - build_topology rejects this
            out[junction.name] = [complex("nan")] * n_freq
            continue
        out[junction.name] = _weighted_node_potential(
            solution.segment_potentials, idxs, arrays.lengths, n_freq
        )
    return out


# ---------------------------------------------------------------------
# Shared helpers (also imported by the sibling backends)
# ---------------------------------------------------------------------


def _weighted_node_potential(
    phi_per_freq: "Sequence[np.ndarray]",
    idxs: "Sequence[int]",
    seg_lengths: np.ndarray,
    n_freq: int,
) -> list[complex]:
    """Node potential of one owner as a length-weighted segment average.

    Physics / consistency
    ---------------------
    A galvanic node (electrode or conductor pseudo-node) is a single
    equipotential in the model. The discrete stand-in for that single
    potential is the **Galerkin (length-weighted) average** of the
    segment-midpoint potentials,

    .. math::
        \\varphi_e \\;=\\; \\frac{\\sum_{k \\in e} L_k\\,\\varphi_k}
                                {\\sum_{k \\in e} L_k},

    which is the pairing dual to the uniform-per-unit-length current
    ansatz $I_k = I_e L_k / \\sum_j L_j$: it is the average potential
    over the electrode's *surface*, not over its segment *list*, and
    it is exactly the reduction the multi-port matrix
    :math:`Z_{ij}` uses inside
    :func:`_solve_cluster_currents` (audit 2026-07-08, WP-B3).

    Reporting the plain ``np.mean`` here — the behaviour up to
    0.14.1 — meant the impedance the user reads back was **not** the
    impedance the solver enforced whenever an electrode carried
    segments of unequal length (mesh electrodes with
    ``dx/nx != dy/ny``, rods split at a soil-layer interface):
    measured -10.9 % on a 4.509 m rod split at ``h_1 = 5 m``
    (weighted 45.686 Ω vs. reported 40.701 Ω), and ideally bonded
    electrodes reported *different* potentials although the solve
    constrains them to one (audit 2026-07-28, F10/F11). For uniform
    per-electrode segment lengths — every plain rod, ring or strip —
    the two averages are identical.

    Parameters
    ----------
    phi_per_freq
        One segment-potential vector per frequency, each of shape
        ``(n_segments,)``.
    idxs
        Segment indices belonging to the owner.
    seg_lengths
        Per-segment length in m, shape ``(n_segments,)``.
    n_freq
        Number of frequencies to reduce.

    Returns
    -------
    list[complex]
        Length-weighted node potential per frequency, in V.
    """
    w = np.asarray(seg_lengths, dtype=float)[idxs]
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        # Defensive only: segment lengths are positive by construction.
        return [complex(np.mean(phi_per_freq[k][idxs])) for k in range(n_freq)]
    return [complex((w @ phi_per_freq[k][idxs]) / w_sum) for k in range(n_freq)]


def _reject_voltage_sources(world, backend: str) -> None:
    """Reject voltage sources in a backend that solves current sources only.

    Since 2026.11 the backends on the shared pipeline solve impressed
    voltages (registry capability ``voltage_sources``); a backend without
    it is called directly here (``Engine.solve`` rejects the world in the
    capability check already). Up to 0.15.0 such a backend dropped the
    voltage sources with a warning and solved to zero.

    Raises
    ------
    BackendCapabilityError
        If ``world`` contains a :class:`~groundfield.sources.VoltageSource`.
    """
    names = [s.name for s in world.sources if s.kind == "voltage"]
    if names:
        from groundfield.solver.registry import (
            BackendCapabilityError,
            _voltage_capable_backends,
        )

        raise BackendCapabilityError(
            f"Backend '{backend}' does not implement impressed voltage sources "
            f"(voltage source(s) {names}). Backends that can: "
            f"{_voltage_capable_backends()}."
        )


#: Name of :func:`_reject_voltage_sources` up to 2026.11 (it warned and
#: dropped the voltage sources); kept for the re-export of ``solver.image``.
_warn_ignored_sources = _reject_voltage_sources


def _reject_concrete_shells(world, backend: str) -> None:
    """Reject worlds carrying concrete-shell corrections (ADR-0012).

    Only the image family consumes the per-segment shell coefficient
    on the reaction-matrix diagonal. Every other backend would
    silently solve the bare-metal problem, which the electrode
    documentation explicitly promises will fail loudly instead.

    Raises
    ------
    BackendCapabilityError
        If any electrode carries a non-zero
        ``concrete_shell_coefficient_ohm_m`` (a ``NotImplementedError``,
        as raised here before 2026.11).
    """
    shelled = [
        e.name
        for e in world.electrodes
        if float(getattr(e, "concrete_shell_coefficient_ohm_m", 0.0) or 0.0) != 0.0
    ]
    if shelled:
        from groundfield.solver.registry import BackendCapabilityError

        raise BackendCapabilityError(
            f"Backend '{backend}' does not implement the concrete-shell "
            f"correction (ADR-0012) carried by electrode(s) {shelled}. "
            "Use backend='image' / 'image_2layer', or set "
            "concrete_shell_coefficient_ohm_m = 0."
        )
