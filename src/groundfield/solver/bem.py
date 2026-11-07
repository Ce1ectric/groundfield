"""Boundary-element entry point of the point-matching family (``bem``).

What this backend computes
--------------------------
``bem`` assembles **the same** point-matching system as
:mod:`groundfield.solver.mom` and solves it with the same routine:
one pulse current per wire segment, the reaction matrix

$$
Z_{ij} \\;=\\; \\frac{\\rho_1}{4\\pi}\\, G(\\mathbf{r}_i, \\mathbf{r}_j)
\\quad (i \\ne j),
$$

i.e. the potential at the midpoint of segment $i$ of a unit *point*
source at the midpoint of segment $j$, the analytic line
self-potential on the diagonal, and the constraints

$$
\\sum_{j=1}^{N} Z_{ij}\\, I_j \\;=\\; \\varphi_c
\\quad \\forall\\, i \\in c,
\\qquad \\sum_{j \\in c} I_j \\;=\\; I_{c,\\text{in}},
$$

over the galvanic clusters $c$ (a source-free cluster floats with
$I_{c,\\text{in}} = 0$, review pass 10, 10.29). $G$ is the closed-form
kernel of the image family: $1/r + 1/r_{\\text{air}}$ for homogeneous
soil and the Tagg/Sunde series of
:mod:`groundfield.solver.image_2layer` for two layers. The measured
difference to ``mom`` is 0.0 to 2.3e-16 relative: ``bem`` and ``mom``
are one computation under two names, and a cross-validation table must
not count them as two engines.

Up to 0.15.0 this module described its matrix as a *collocation* of
$G$ that differed from a *Galerkin* average in ``mom``. Neither half
was true: both backends point-match at the segment midpoints with a
point-source near field (10.31). A genuine collocation boundary-element
method in the sense of Colominas, Navarrina & Casteleiro (2007, 2012)
integrates the source exactly along each segment; that would raise the
near-field coupling by about 1.6 % to 1.9 % on a 3 m rod and is not
implemented.

Validity
--------
- Quasi-static, $f < 1\\,\\mathrm{kHz}$.
- ``HomogeneousSoil``, ``TwoLayerSoil`` and a ``MultiLayerSoil`` whose
  canonical stack (equal adjacent layers merged, 10.48) has at most two
  layers; $n \\ge 3$ distinct layers raise ``NotImplementedError``
  (use ``mom_sommerfeld``). No complex-image fit is computed (F34).
- Thin-wire approximation (wire radius small against the segment
  length), as in every segment-based backend.

References
----------
- Colominas, I., Navarrina, F., & Casteleiro, M. (2007). Numerical
  simulation of transferred potentials in earthing grids considering
  layered soil models. *IEEE PWRD* 22(3).
- Colominas, I., París, J., Navarrina, F., & Casteleiro, M. (2012).
  Improvement of computer methods for grounding analysis in layered
  soils by using high-efficient convergence acceleration techniques.
  *Adv. Eng. Soft.* 44.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.solver._layered import LayerStack, as_layer_stack, canonical_soil
from groundfield.solver.image import (
    _assemble_inductance_matrix,
    _build_clusters,
    _build_distributed_topology,
    _build_finite_branches,
    _discretize_electrode,
    _reject_concrete_shells,
    _self_corrected_kernel,
    _Segment,
    _warn_ignored_sources,
)
from groundfield.solver.mom import (  # shared point-matching solve
    _floating_clusters,
    _point_matching_solve,
    _node_results,
)
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_bem"]

_log = get_logger(__name__)


# ---------------------------------------------------------------------
# Z-matrix assembly (point matching, closed-form kernels)
# ---------------------------------------------------------------------


def _build_Z_collocation(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    stack: LayerStack,
    *,
    max_terms: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    """Point-matching N×N reaction matrix (identical to ``mom``'s).

    Strategy:

    - ``n_layers <= 1`` → homogeneous matrix from the closed-form
      ``image`` self-kernel (line self-action on the diagonal,
      point-source ``1/r + 1/r_{\\text{air}}`` off-diagonal).
    - ``n_layers == 2`` → exact Tagg/Sunde matrix from the
      ``image_2layer`` self-kernel (``allow_cross_layer=True``, so
      interface-crossing geometries dispatch to the rigorous
      cross-layer path instead of silently applying the upper-layer
      series).
    - ``n_layers >= 3`` → :class:`NotImplementedError`. The historic
      complex-image contribution for this regime came from an
      incomplete Green's function (audit 2026-07-08, WP-E);
      :func:`solve_bem` rejects such soils before reaching this
      builder, and this branch keeps the module from growing a silent
      wrong-physics path again.

    Parameters
    ----------
    max_terms, tol
        Tagg/Sunde series truncation, forwarded from the engine
        (``Engine.image_max_terms`` / ``Engine.image_series_tol``).
    """
    n = seg_points.shape[0]
    rho_1 = float(stack.rhos[0])
    eye = np.eye(n)

    if stack.n_layers <= 1:
        return _self_corrected_kernel(seg_points, seg_lengths, wire_radii, eye, rho_1)

    if stack.n_layers == 2:
        from groundfield.soil.models import TwoLayerSoil
        from groundfield.solver.image_2layer import (
            _two_layer_self_kernel_factory,
        )

        soil = TwoLayerSoil(
            rho_1=float(stack.rhos[0]),
            rho_2=float(stack.rhos[1]),
            h_1=float(stack.h[0]),
        )
        kern = _two_layer_self_kernel_factory(
            soil,
            max_terms=max_terms,
            tol=tol,
            allow_cross_layer=True,
        )
        return kern(seg_points, seg_lengths, wire_radii, eye)

    raise NotImplementedError(
        f"bem: no reaction matrix for n_layers = {stack.n_layers} "
        ">= 3 — the shared complex-image kernel was structurally "
        "incomplete (audit 2026-07-08). Use "
        "backend='mom_sommerfeld' or 'fem'."
    )


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_bem(world: "World", engine: "Engine") -> FieldResult:
    """Point-matching method-of-moments solver under the name ``bem``.

    What actually runs
    ------------------
    - $n = 1$ → homogeneous image-charge kernel.
    - $n = 2$ → exact Tagg/Sunde series kernel.
    - $n \\ge 3$ distinct layers → :class:`NotImplementedError`
      (audit 2026-07-08, WP-E). Use ``mom_sommerfeld``.

    In both reachable regimes the assembled reaction matrix and the
    constraint solve are those of ``mom`` (relative difference 0.0 to
    2.3e-16), so this result is **not** an independent check of
    ``mom`` — see the module docstring. No complex-image fit is
    computed (review pass 9, F34).

    Parameters
    ----------
    world
        World to evaluate.
    engine
        Engine configuration; ``engine.segment_length`` controls the
        discretisation, ``engine.image_max_terms`` /
        ``engine.image_series_tol`` the Tagg/Sunde truncation at
        $n = 2$.

    Returns
    -------
    FieldResult
        ``metadata['cim_fit_used'] = False`` and
        ``metadata['reduces_to'] = 'mom'`` record both facts above;
        ``metadata['solver']`` is ``'point_matching'`` (``'collocation'``
        up to 0.15.0).
    """
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'bem' supports HomogeneousSoil, TwoLayerSoil, "
            f"and MultiLayerSoil. Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")
    _reject_concrete_shells(world, "bem")
    _warn_ignored_sources(world, "bem")

    stack = as_layer_stack(world.soil)
    if stack.n_layers >= 3:
        # Audit 2026-07-08, WP-E: bem shares the historic incomplete
        # n>=3 CIM kernel — reject loudly until a complete kernel
        # exists.
        raise NotImplementedError(
            f"bem: n_layers = {stack.n_layers} >= 3 is not supported — "
            "the shared complex-image kernel was structurally "
            "incomplete (audit 2026-07-08). Use "
            "backend='mom_sommerfeld' (full layered Green's function) "
            "or 'fem' for n >= 3 soils."
        )
    ds = engine.segment_length

    _log.info(
        "bem: n_layers=%d, segment_length=%.3f, closed-form kernel "
        "(no complex-image fit); identical reaction matrix to 'mom'",
        stack.n_layers,
        ds,
    )

    # 1) Discretisation.
    all_segments: list[_Segment] = []
    elec_to_segidx: dict[str, list[int]] = {}
    interfaces = (float(stack.h[0]),) if stack.n_layers >= 2 else None
    for e in world.electrodes:
        segs = _discretize_electrode(e, ds, layer_interfaces=interfaces)
        elec_to_segidx[e.name] = list(
            range(len(all_segments), len(all_segments) + len(segs))
        )
        all_segments.extend(segs)

    # 2) Per-electrode input currents.
    elec_input_current: dict[str, complex] = {e.name: 0j for e in world.electrodes}
    for src in world.sources:
        if src.kind != "current":
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in elec_input_current:
            elec_input_current[src.attached_to] += i_complex

    cluster_id = _build_clusters(world.electrodes, world.conductors)
    finite_branches = _build_finite_branches(world.conductors, cluster_id)

    # 2b) Distributed-conductor topology (ADR-0003) + ADR-0004
    #     inductive coupling.
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
    seg_points = np.array([s.midpoint for s in all_segments])
    seg_lengths = np.array([s.length for s in all_segments])
    wire_radii = np.array([s.wire_radius for s in all_segments])

    # 3) Reaction matrix via collocation (series knobs from engine).
    #    The n>=3 cross-layer guard that used to sit here was
    #    unreachable behind the n>=3 rejection above and was removed in
    #    0.15.0; n=2 cross-layer geometries are handled rigorously by
    #    _two_layer_self_kernel_factory(allow_cross_layer=True)
    #    (ADR-0007).
    Z = _build_Z_collocation(
        seg_points,
        seg_lengths,
        wire_radii,
        stack,
        max_terms=engine.image_max_terms,
        tol=engine.image_series_tol,
    )

    # 4) Frequency loop (point-matching solve + Z · I_seg for phi).
    n_freq = len(engine.frequencies)
    omegas = [2.0 * np.pi * float(f) for f in engine.frequencies]
    real_electrode_names = {e.name for e in world.electrodes}

    def _solve_at(omega: float) -> tuple[np.ndarray, np.ndarray]:
        carson_dz = (
            carson_builder(omega)
            if (has_inductance and carson_builder is not None)
            else None
        )
        sc, _ = _point_matching_solve(
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
            ph = Z @ sc.real + 1j * (Z @ sc.imag)
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

    # Node potentials and leakage currents; floating clusters (no
    # source, no branch) report an exactly zero net current (10.29).
    (
        electrode_potentials,
        electrode_currents,
        conductor_potentials,
        conductor_currents,
    ) = _node_results(
        elec_to_segidx=elec_to_segidx,
        phi_per_freq=phi_per_freq,
        sc_per_freq=sc_per_freq,
        cluster_id=cluster_id,
        electrode_names=real_electrode_names,
        floating=_floating_clusters(
            elec_input_current, cluster_id, finite_branches, real_electrode_names
        ),
        n_freq=n_freq,
    )

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
            n
            for n in cluster_id
            if cluster_id[n] == cluster_id[ename] and n in real_electrode_names
        )

    metadata = {
        "world_name": world.name,
        "n_segments": n_segments,
        "segment_length": ds,
        "n_layers": int(stack.n_layers),
        "rhos": stack.rhos.tolist(),
        "h": stack.h.tolist(),
        # Honesty flags (review pass 9, F34): the reachable n <= 2
        # paths use exact closed-form kernels, so there is no
        # complex-image fit to report — and the reaction matrix is the
        # one `mom` assembles, so this is not an independent check.
        "cim_fit_used": False,
        "cim_n_images": 0,
        "cim_rms": None,
        "reduces_to": "mom",
        "solver": "point_matching",
        "stub": False,
        "earth_inductive_model": earth_inductive_model,
    }
    if has_inductance:
        from groundfield.coupling.carson import skin_depth

        sigma_ref = (
            sigma_earth_for_carson
            if sigma_earth_for_carson is not None
            else 1.0 / float(stack.rhos[0])
        )
        metadata["penetration_depth"] = {
            float(f): skin_depth(2.0 * np.pi * f, sigma_ref) for f in engine.frequencies
        }
    if conductor_currents:
        metadata["conductor_node_currents"] = conductor_currents
        metadata["conductor_node_potentials"] = conductor_potentials

    return FieldResult(
        backend="bem",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=float(stack.rhos[0]),
        soil=canonical_soil(world.soil),
        clusters=cluster_members,
        metadata=metadata,
    )
