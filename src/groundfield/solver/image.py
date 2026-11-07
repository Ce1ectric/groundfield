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

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

# Re-exports: the implementation moved into private modules (review
# pass 10, 10.75 / 10.79). Sibling backends, tests and user notebooks
# import these names from ``groundfield.solver.image``; keep them here.
from groundfield.solver._discretize import _TANGENT_X as _TANGENT_X
from groundfield.solver._discretize import _TANGENT_Y as _TANGENT_Y
from groundfield.solver._discretize import _TANGENT_Z as _TANGENT_Z
from groundfield.solver._discretize import (
    _conductor_node_name as _conductor_node_name,
)
from groundfield.solver._discretize import (
    _discretize_conductor as _discretize_conductor,
)
from groundfield.solver._discretize import (
    _discretize_electrode as _discretize_electrode,
)
from groundfield.solver._discretize import (
    _discretize_grid_mesh as _discretize_grid_mesh,
)
from groundfield.solver._discretize import _discretize_mesh as _discretize_mesh
from groundfield.solver._discretize import (
    _discretize_polyline as _discretize_polyline,
)
from groundfield.solver._discretize import _discretize_ring as _discretize_ring
from groundfield.solver._discretize import _discretize_rod as _discretize_rod
from groundfield.solver._discretize import _discretize_star as _discretize_star
from groundfield.solver._discretize import _discretize_strip as _discretize_strip
from groundfield.solver._discretize import _DistributedBranch as _DistributedBranch
from groundfield.solver._discretize import _frozen as _frozen
from groundfield.solver._discretize import _grid_segments as _grid_segments
from groundfield.solver._discretize import _Segment as _Segment
from groundfield.solver._discretize import _segments_per_wire as _segments_per_wire
from groundfield.solver._discretize import (
    _split_interval_at_interfaces as _split_interval_at_interfaces,
)
from groundfield.solver._guards import (
    _MIN_IMAGE_SEPARATION_RATIO as _MIN_IMAGE_SEPARATION_RATIO,
)
from groundfield.solver._guards import _MIN_DISTANCE as _MIN_DISTANCE
from groundfield.solver._guards import _SURFACE_TOL as _SURFACE_TOL
from groundfield.solver._guards import ShallowSegmentWarning as ShallowSegmentWarning
from groundfield.solver._guards import (
    _check_segment_depths as _check_segment_depths,
)
from groundfield.solver._guards import (
    _require_image_separation as _require_image_separation,
)
from groundfield.solver._guards import _require_thin_wire as _require_thin_wire
from groundfield.solver._guards import (
    _segment_owner_label as _segment_owner_label,
)
from groundfield.solver._guards import _segment_z_extent as _segment_z_extent
from groundfield.solver._inductive import _CarsonBuilder as _CarsonBuilder
from groundfield.solver._inductive import (
    _assemble_inductance_matrix as _assemble_inductance_matrix,
)
from groundfield.solver._kernels import _potential_kernel as _potential_kernel
from groundfield.solver._kernels import (
    _self_corrected_kernel as _self_corrected_kernel,
)
from groundfield.solver._pipeline import (
    _reject_concrete_shells as _reject_concrete_shells,
)
from groundfield.solver._pipeline import (
    _warn_ignored_sources as _warn_ignored_sources,
)
from groundfield.solver._pipeline import (
    _weighted_node_potential as _weighted_node_potential,
)
from groundfield.solver._topology import (
    _build_clusters as _build_clusters,
)
from groundfield.solver._topology import (
    _build_distributed_topology as _build_distributed_topology,
)
from groundfield.solver._topology import (
    _build_finite_branches as _build_finite_branches,
)
from groundfield.solver._topology import (
    _solve_cluster_currents as _solve_cluster_currents,
)
from groundfield.soil.models import HomogeneousSoil
from groundfield.solver.result import FieldResult, PointSource

__all__ = ["solve_image", "ShallowSegmentWarning"]


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
        elec_to_segidx[e.name] = list(
            range(len(all_segments), len(all_segments) + len(segs))
        )
        all_segments.extend(segs)

    # 2) Per-electrode input currents from the configured sources
    elec_input_current: dict[str, complex] = {e.name: 0j for e in world.electrodes}
    n_ignored_sources = 0
    for src in world.sources:
        if src.kind != "current":
            # Voltage sources are not supported by this backend.
            n_ignored_sources += 1
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in elec_input_current:
            elec_input_current[src.attached_to] += i_complex
    if n_ignored_sources:
        warnings.warn(
            f"solve_image: {n_ignored_sources} non-current source(s) "
            "ignored — only CurrentSource is supported. A world driven "
            "solely by voltage sources solves to all-zero.",
            UserWarning,
            stacklevel=2,
        )

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
    earth_inductive_model = getattr(engine, "earth_inductive_model", "perfect_mirror")
    sigma_earth_for_carson: float | None = None
    layered_earth_for_sommerfeld: object = None
    if earth_inductive_model == "carson_series":
        from groundfield.coupling import resolve_earth_conductivity

        sigma_earth_for_carson = resolve_earth_conductivity(world.soil)
    elif earth_inductive_model == "sommerfeld":
        from groundfield.coupling import resolve_earth_layers

        layered_earth_for_sommerfeld = resolve_earth_layers(world.soil)
    inductance_matrix_full, has_inductance, carson_builder = (
        _assemble_inductance_matrix(
            distributed_branches_objs,
            n_lumped_branches=n_lumped_branches,
            n_total_branches=len(finite_branches),
            earth_model=earth_inductive_model,
            sigma_earth=sigma_earth_for_carson,
            layered_earth=layered_earth_for_sommerfeld,
        )
    )

    n_segments = len(all_segments)
    seg_points = np.array([s.midpoint for s in all_segments])  # (N, 3)
    seg_lengths = np.array([s.length for s in all_segments])  # (N,)

    # 3c) Burial-depth validation (audit 2026-07-28, F26/F29): the
    #     galvanic image construction needs a finite image separation
    #     2 z >> a. Reject surface-laid / airborne conductors and warn
    #     for the shallow regime where the point-image stand-in biases
    #     the diagonal.
    _check_segment_depths(all_segments, where="solve_image")

    # 4) Current sharing within clusters via the multi-port matrix
    wire_radii = np.array([s.wire_radius for s in all_segments])
    # ADR-0012 V2: per-segment concrete-shell coefficient (zero for
    # every segment that does not belong to a concrete-encased
    # foundation electrode — historic case).
    seg_shell_coeffs = np.array(
        [s.concrete_shell_coefficient_ohm_m for s in all_segments],
        dtype=float,
    )

    def _homogeneous_self(seg_pts, seg_lens, wr, currents):
        """Closure: homogeneous self-action with fixed rho."""
        return _self_corrected_kernel(seg_pts, seg_lens, wr, currents, rho)

    n_freq = len(engine.frequencies)
    omegas = [2.0 * np.pi * float(f) for f in engine.frequencies]
    real_electrode_names = {e.name for e in world.electrodes}

    # ADR-0010 Tier 1 (WP-F): the multi-port grounding matrix Z is
    # frequency-independent — share it across the per-frequency calls.
    _mp_cache: dict = {}

    def _solve_at(omega: float) -> tuple[dict[str, complex], np.ndarray]:
        """Solve once at a given angular frequency.

        Returns
        -------
        elec_total : dict
            Per-owner total leakage current.
        seg_currents : np.ndarray
            Per-segment current distribution (uniform per unit length).
        """
        carson_dz = (
            carson_builder(omega)
            if (has_inductance and carson_builder is not None)
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
            shell_coefficients=seg_shell_coeffs,
            multiport_cache=_mp_cache,
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
        return elec_total, sc

    def _phi_batch(sc_list: list[np.ndarray]) -> list[np.ndarray]:
        """Segment-midpoint potentials for a list of current vectors.

        All real/imaginary parts are stacked into one
        ``(n_segments, 2·len(sc_list))`` excitation matrix so the
        kernel's O(N²) geometry tensors are built exactly once for
        the whole frequency set (ADR-0010 Tier 1).
        """
        k = len(sc_list)
        stacked = np.zeros((n_segments, 2 * k))
        for m, sc in enumerate(sc_list):
            stacked[:, m] = sc.real
            stacked[:, k + m] = sc.imag
        if not stacked.any():
            return [np.zeros(n_segments, dtype=complex)] * k
        phi = _self_corrected_kernel(
            seg_points,
            seg_lengths,
            wire_radii,
            stacked,
            rho,
        )
        # ADR-0012 V2: same shell augmentation as in
        # _solve_cluster_currents so electrode_potentials /
        # cluster_impedance reflect the concrete-shell drop.
        if np.any(seg_shell_coeffs > 0.0):
            with np.errstate(divide="ignore", invalid="ignore"):
                shell_diag = np.where(
                    seg_lengths > 0.0,
                    seg_shell_coeffs / seg_lengths,
                    0.0,
                )
            phi = phi + shell_diag[:, None] * stacked
        return [phi[:, m] + 1j * phi[:, k + m] for m in range(k)]

    # Frequency loop. With no inductive coupling the system is
    # frequency-independent, so we solve once and replicate.
    elec_per_freq: list[dict[str, complex]] = []
    sc_per_freq: list[np.ndarray] = []
    phi_per_freq: list[np.ndarray] = []
    if has_inductance:
        for omega in omegas:
            et, sc = _solve_at(omega)
            elec_per_freq.append(et)
            sc_per_freq.append(sc)
        phi_per_freq = _phi_batch(sc_per_freq)
    else:
        et, sc = _solve_at(0.0)
        ph = _phi_batch([sc])[0]
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
        u_list = _weighted_node_potential(phi_per_freq, idxs, seg_lengths, n_freq)
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
            n
            for n in cluster_id
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
            float(f): skin_depth(2.0 * np.pi * f, sigma_ref) for f in engine.frequencies
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
