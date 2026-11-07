"""Shared solve pipeline of the image family (private).

``solve_image`` (homogeneous soil) and ``solve_image_2layer`` (two-layer
soil, Tagg/Sunde series) run through one pipeline that differs only in
the Green's-function object handed to :func:`solve_image_family`
(review pass 10, 10.75). The steps, in order:

1. :func:`discretise` — electrodes into leakage segments
   (:meth:`~groundfield.geometry.electrodes.RodElectrode.segments` and
   siblings), split at the soil-layer interfaces the Green's function
   asks for.
2. :func:`inject` — impressed source currents per electrode.
   **Extension point (CORE-2):** two-terminal sources (``return_to``)
   and leakage-free junction nodes are added here; the step returns the
   impressed current per node, which the topology step attaches to the
   clusters.
3. :func:`build_topology` — clusters of ideally bonded electrodes,
   lumped finite-impedance branches and the distributed-conductor
   chains (ADR-0003), whose galvanic segments join the leakage
   segments as one-segment pseudo-owners.
4. :func:`assemble_branches` — the frequency-dependent branch block:
   partial inductances (ADR-0004) and the earth-return correction
   closure (Carson series ADR-0005, Sommerfeld ADR-0006).
   **Extension point (COUP):** solver-side inductive fixes (image
   baselines, Neumann inductance of lumped / single-segment
   conductors, skin effect) belong here.
5. :func:`solve` — per frequency, the augmented nodal system of
   :mod:`groundfield.solver._topology` with the multi-port matrix built
   from the Green's function's reaction kernel; then the segment
   currents (uniform per unit length per owner) and the segment
   potentials.
6. :func:`build_result` — node potentials (length-weighted, F10/F11),
   point sources, clusters and metadata as a
   :class:`~groundfield.solver.result.FieldResult`.

The Green's-function object (:class:`ImageGreen`) supplies the backend
name, the layer interfaces, the reaction kernel and the
backend-specific metadata. **Extension point (ENG):** the ``mom``
family can be ported by providing a Green's-function object whose
``solve`` replaces the uniform-current owner model with per-segment
currents; steps 1–4 and 6 are shared as they are.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from groundfield.solver._discretize import _discretize_electrode
from groundfield.solver._guards import _check_segment_depths
from groundfield.solver._inductive import _assemble_inductance_matrix
from groundfield.solver._topology import (
    _build_clusters,
    _build_distributed_topology,
    _build_finite_branches,
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
        ``None`` without inductive conductors.
    earth_correction
        ``omega -> ΔZ(omega)`` earth-return correction over all finite
        branches in Ω, or ``None``.
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
        """Whether any finite branch carries an inductance."""
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
    """

    points: np.ndarray
    lengths: np.ndarray
    radii: np.ndarray
    shell_coefficients: np.ndarray

    @classmethod
    def from_segments(cls, segments: list[_Segment]) -> SegmentArrays:
        """Stack the per-segment attributes."""
        return cls(
            points=np.array([s.midpoint for s in segments]),
            lengths=np.array([s.length for s in segments]),
            radii=np.array([s.wire_radius for s in segments]),
            shell_coefficients=np.array(
                [s.concrete_shell_coefficient_ohm_m for s in segments], dtype=float
            ),
        )


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
    """

    owner_currents: list[dict[str, complex]]
    segment_currents: list[np.ndarray]
    segment_potentials: list[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------


def discretise(
    world: World,
    engine: Engine,
    layer_interfaces: Sequence[float] | None = None,
) -> Discretisation:
    """Step 1: discretise every electrode of ``world``.

    Parameters
    ----------
    world
        World with at least one electrode.
    engine
        Engine; ``segment_length`` is the target segment length in m.
    layer_interfaces
        Interface depths in m at which wires are split.
    """
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
    """Step 2: impressed current per electrode in A (complex phasor).

    Current sources add ``magnitude·exp(j·phase)`` to the electrode they
    are attached to; other source kinds are skipped with a warning.
    """
    node_currents: dict[str, complex] = {e.name: 0j for e in world.electrodes}
    for src in world.sources:
        if src.kind != "current":
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in node_currents:
            node_currents[src.attached_to] += i_complex
    _warn_ignored_sources(world, where, stacklevel=stacklevel + 1)
    return node_currents


def build_topology(world: World, disc: Discretisation) -> Topology:
    """Step 3: clusters, finite branches and distributed conductors.

    Electrodes joined by an *ideal* conductor share one potential
    (cluster); conductors with a finite series resistance enter the
    nodal system as branches. Distributed conductors (ADR-0003) add a
    chain of branches; with ``coupling_to_soil="galvanic"`` every
    sub-piece also leaks through a one-segment pseudo-owner, which is
    appended to ``disc`` (its own cluster). Leakage-free interior nodes
    of isolated chains become clusters without segments.
    """
    cluster_id = _build_clusters(world.electrodes, world.conductors)
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
    return Topology(
        cluster_id=cluster_id,
        finite_branches=finite_branches,
        distributed_branches=distributed,
        n_lumped_branches=n_lumped,
        pseudo_owners=pseudo_owners,
    )


def assemble_branches(world: World, engine: Engine, topo: Topology) -> BranchModel:
    """Step 4: inductance matrix and earth-return correction of the branches.

    ADR-0004 + ADR-0005 + ADR-0006: the partial-inductance matrix covers
    the distributed-conductor branches with ``inductance_model ==
    "neumann"`` (lumped branches stay purely resistive). With
    ``earth_inductive_model`` ``"carson_series"`` or ``"sommerfeld"`` a
    closure builds the per-frequency earth-return correction.
    """
    model = getattr(engine, "earth_inductive_model", "perfect_mirror")
    sigma_earth: float | None = None
    layered_earth: object = None
    if model == "carson_series":
        from groundfield.coupling import resolve_earth_conductivity

        sigma_earth = resolve_earth_conductivity(world.soil)
    elif model == "sommerfeld":
        from groundfield.coupling import resolve_earth_layers

        layered_earth = resolve_earth_layers(world.soil)
    L_full, has_inductance, correction = _assemble_inductance_matrix(
        topo.distributed_branches,
        n_lumped_branches=topo.n_lumped_branches,
        n_total_branches=len(topo.finite_branches),
        earth_model=model,
        sigma_earth=sigma_earth,
        layered_earth=layered_earth,
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
) -> Solution:
    """Step 5: owner currents, segment currents and segment potentials.

    The multi-port matrix ``Z`` is frequency-independent and shared by
    the per-frequency solves (ADR-0010 Tier 1). Without inductive
    branches the whole system is frequency-independent: it is solved
    once and replicated.
    """
    n_freq = len(engine.frequencies)
    mp_cache: dict = {}

    def _solve_at(omega: float) -> tuple[dict[str, complex], np.ndarray]:
        correction = (
            branches.earth_correction(omega)
            if branches.earth_correction is not None
            else None
        )
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
        )
        sc = _segment_currents(owner_currents, disc.owner_segments, arrays.lengths)
        return owner_currents, sc

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
        for f in engine.frequencies:
            et, sc = _solve_at(2.0 * np.pi * float(f))
            owner_per_freq.append(et)
            sc_per_freq.append(sc)
        phi_per_freq = _potentials(owner_per_freq, sc_per_freq)
    else:
        et, sc = _solve_at(0.0)
        ph = _potentials([et], [sc])[0]
        owner_per_freq = [et] * n_freq
        sc_per_freq = [sc] * n_freq
        phi_per_freq = [ph] * n_freq
    return Solution(
        owner_currents=owner_per_freq,
        segment_currents=sc_per_freq,
        segment_potentials=phi_per_freq,
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
    ``["conductor_node_potentials"]``.
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
    point_sources = [
        PointSource(
            position=tuple(arrays.points[i].tolist()),
            current=[complex(sc_per_freq[k][i]) for k in range(n_freq)],
            electrode_name=disc.segments[i].electrode_name,
            length=float(arrays.lengths[i]),
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
    )


def solve_image_family(
    world: World, engine: Engine, green: ImageGreen, *, stacklevel: int = 3
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

    Returns
    -------
    FieldResult
        Node potentials and currents per frequency, point sources and
        metadata.
    """
    disc = discretise(world, engine, green.layer_interfaces)
    node_currents = inject(world, where=green.where, stacklevel=stacklevel)
    topo = build_topology(world, disc)
    branches = assemble_branches(world, engine, topo)
    # Burial-depth validation (audit 2026-07-28, F26/F29): the galvanic
    # image construction needs a finite image separation 2 z >> a.
    # Reject surface-laid / airborne conductors and warn for the shallow
    # regime where the point-image stand-in biases the diagonal.
    _check_segment_depths(disc.segments, where=green.where, stacklevel=stacklevel + 1)
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
    )
    metadata = {
        "world_name": world.name,
        "n_segments": len(disc.segments),
        "segment_length": engine.segment_length,
        **green.metadata(solution, stacklevel=stacklevel + 1),
        "stub": False,
        "earth_inductive_model": branches.earth_inductive_model,
    }
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


def _warn_ignored_sources(world, backend: str, *, stacklevel: int = 3) -> None:
    """Warn when a backend drops non-current sources.

    Every backend supports impressed current sources only; other
    source kinds are skipped during the injection assembly. Without
    this warning a world driven solely by voltage sources solves to
    all-zero with no diagnostic.
    """
    n_ignored = sum(1 for s in world.sources if s.kind != "current")
    if n_ignored:
        warnings.warn(
            f"{backend}: {n_ignored} non-current source(s) ignored — "
            "only CurrentSource is supported. A world driven solely "
            "by non-current sources solves to all-zero.",
            UserWarning,
            stacklevel=stacklevel,
        )


def _reject_concrete_shells(world, backend: str) -> None:
    """Reject worlds carrying concrete-shell corrections (ADR-0012).

    Only the image family consumes the per-segment shell coefficient
    on the reaction-matrix diagonal. Every other backend would
    silently solve the bare-metal problem, which the electrode
    documentation explicitly promises will fail loudly instead.

    Raises
    ------
    NotImplementedError
        If any electrode carries a non-zero
        ``concrete_shell_coefficient_ohm_m``.
    """
    shelled = [
        e.name
        for e in world.electrodes
        if float(getattr(e, "concrete_shell_coefficient_ohm_m", 0.0) or 0.0) != 0.0
    ]
    if shelled:
        raise NotImplementedError(
            f"Backend '{backend}' does not implement the concrete-shell "
            f"correction (ADR-0012) carried by electrode(s) {shelled}. "
            "Use backend='image' / 'image_2layer', or set "
            "concrete_shell_coefficient_ohm_m = 0."
        )
