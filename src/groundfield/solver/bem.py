"""Boundary-Element-Method backend (``bem``).

Mathematical / physical model
-----------------------------
Following Colominas, Navarrina & Casteleiro (2007, 2012) we treat the
grounding system as a **boundary** problem: the unknown is the
leakage-current density along the wire surfaces. Discretising every
electrode into $N$ line segments turns the boundary integral
equation into a dense linear system
$$
\\sum_{j=1}^{N}\\, Z_{ij}\\, I_j \\;=\\; \\varphi_c
\\qquad \\forall\\,i \\in c,
\\qquad \\sum_{j \\in c} I_j \\;=\\; I_{c,\\text{in}},
$$
where $c$ ranges over the galvanic clusters,
$\\varphi_c$ is the (unknown) shared cluster potential, and the
**reaction matrix entries**
$$
Z_{ij} \\;=\\;
\\frac{1}{4\\pi}
\\int_{\\Sigma_i} G(\\mathbf{r}_i, \\mathbf{r}'_j)\\, dS_j
$$
are obtained by **collocation** of $G$ at the centre of segment
$i$ (single test point per segment), instead of the Galerkin
average used in :mod:`groundfield.solver.mom`. Collocation is the
historically dominant flavour of BEM in the grounding literature
because it preserves the same accuracy on smooth electrodes while
being roughly half the cost of the Galerkin scheme.

The Green's function $G$ is the **layered Sommerfeld kernel**
itself; we evaluate it through the closed-form complex-image fit
provided by :mod:`groundfield.solver.cim`, so the BEM and the CIM
engines share the same physics for layered soils. For homogeneous
soils $G(r) = 1/r + 1/r_{\\text{air-img}}$.

Differences from ``mom``
------------------------
- ``mom`` uses Galerkin (average potential) on a layered Green's
  function expressed via the real Tagg/Sunde series (closed form for
  ``n = 2``); ``bem`` uses collocation with the CIM kernel (closed
  form for any ``n``).
- The two engines therefore disagree at the ~1 % level on identical
  inputs — they sample different aspects of the same continuum
  problem. Their agreement is one of the cross-validation criteria.

Validity
--------
- Quasi-static, $f < 1\\,\\mathrm{kHz}$.
- Wire radius small compared to segment length; thin-wire
  approximation in the line self-correction (same as the other
  segment-based engines).

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
from groundfield.solver._layered import LayerStack, as_layer_stack
from groundfield.solver.cim import (
    ComplexImageFit,
    fit_complex_images,
)
from groundfield.solver.image import (
    _MIN_DISTANCE,
    _assemble_inductance_matrix,
    _build_clusters,
    _build_distributed_topology,
    _build_finite_branches,
    _discretize_electrode,
    _self_corrected_kernel,
    _Segment,
)
from groundfield.solver.mom import _galerkin_solve  # used as constraint solver
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_bem"]

_log = get_logger(__name__)


# ---------------------------------------------------------------------
# Z-matrix assembly via collocation
# ---------------------------------------------------------------------


def _build_Z_collocation(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    stack: LayerStack,
    fit: ComplexImageFit,
) -> np.ndarray:
    """Collocation N×N reaction matrix.

    Strategy:

    - ``n_layers <= 1`` → homogeneous matrix from the closed-form
      ``image`` self-kernel (line self-action on the diagonal,
      point-source ``1/r + 1/r_{\\text{air}}`` off-diagonal).
    - ``n_layers == 2`` → exact Tagg/Sunde matrix from the
      ``image_2layer`` self-kernel.
    - ``n_layers >= 3`` → homogeneous matrix plus the complex-image
      contribution $\\sum_k a_k / r_k$ from the matrix-pencil
      fit (single image per pole at $z = -(z_s + 2 \\beta_k)$).
    """
    n = seg_points.shape[0]
    rho_1 = float(stack.rhos[0])
    eye = np.eye(n)

    if stack.n_layers <= 1:
        return _self_corrected_kernel(
            seg_points, seg_lengths, wire_radii, eye, rho_1
        )

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
        kern = _two_layer_self_kernel_factory(soil, max_terms=200, tol=1e-6)
        return kern(seg_points, seg_lengths, wire_radii, eye)

    # n >= 3: homogeneous matrix + complex-image contribution.
    Z = _self_corrected_kernel(
        seg_points, seg_lengths, wire_radii, eye, rho_1
    )
    if fit.a.size == 0:
        return Z

    diff_xy = seg_points[:, None, 0:2] - seg_points[None, :, 0:2]
    delta_sq = np.einsum("mnk,mnk->mn", diff_xy, diff_xy)
    z_field = seg_points[:, 2:3]
    z_src = seg_points[None, :, 2]
    Z_complex = np.zeros_like(Z, dtype=complex)
    for a, b in zip(fit.a, fit.beta):
        d = z_field + z_src + 2.0 * b
        r = np.sqrt(delta_sq + d ** 2)
        r_abs = np.abs(r)
        tiny = r_abs < _MIN_DISTANCE
        if np.any(tiny):
            r = np.where(tiny, _MIN_DISTANCE + 0j, r)
        Z_complex += a * (1.0 / r)
    Z_complex *= rho_1 / (4.0 * np.pi)
    return Z + Z_complex.real


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_bem(
    world: "World",
    engine: "Engine",
    *,
    n_images: int = 8,
    n_samples: int = 64,
) -> FieldResult:
    """Boundary-Element-Method solver (collocation, layered CIM kernel).

    Parameters
    ----------
    world
        World to evaluate.
    engine
        Engine configuration.
    n_images, n_samples
        Forwarded to :func:`groundfield.solver.cim.fit_complex_images`
        when the soil is layered. Ignored otherwise.

    Returns
    -------
    FieldResult
    """
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'bem' supports HomogeneousSoil, TwoLayerSoil, "
            f"and MultiLayerSoil. Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    stack = as_layer_stack(world.soil)
    fit = fit_complex_images(stack, n_images=n_images, n_samples=n_samples)
    ds = engine.segment_length

    _log.info(
        "bem: n_layers=%d, n_images=%d, segment_length=%.3f",
        stack.n_layers, fit.a.size, ds,
    )

    # 1) Discretisation.
    all_segments: list[_Segment] = []
    elec_to_segidx: dict[str, list[int]] = {}
    for e in world.electrodes:
        segs = _discretize_electrode(e, ds)
        elec_to_segidx[e.name] = list(
            range(len(all_segments), len(all_segments) + len(segs))
        )
        all_segments.extend(segs)

    # 2) Per-electrode input currents.
    elec_input_current: dict[str, complex] = {
        e.name: 0j for e in world.electrodes
    }
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

    if stack.n_layers >= 3:
        z_max = seg_points[:, 2].max()
        h_1 = float(stack.h[0])
        if z_max >= h_1:
            # n=2 cross-layer is handled via the shared
            # _two_layer_self_kernel_factory dispatcher (Phase A).
            import warnings as _w

            _w.warn(
                f"bem: cross-layer geometry on n_layers="
                f"{stack.n_layers} not yet supported. "
                "Use backend='image_2layer' for n=2; for n>=3 "
                "thicken the upper layer.",
                UserWarning,
                stacklevel=2,
            )

    # 3) Reaction matrix via collocation.
    Z = _build_Z_collocation(seg_points, seg_lengths, wire_radii, stack, fit)

    # 4) Frequency loop (Galerkin solve + Z · I_seg for phi).
    n_freq = len(engine.frequencies)
    omegas = [2.0 * np.pi * float(f) for f in engine.frequencies]
    real_electrode_names = {e.name for e in world.electrodes}

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

    metadata = {
        "world_name": world.name,
        "n_segments": n_segments,
        "segment_length": ds,
        "n_layers": int(stack.n_layers),
        "rhos": stack.rhos.tolist(),
        "h": stack.h.tolist(),
        "cim_n_images": int(fit.a.size),
        "cim_rms": float(fit.rms),
        "solver": "collocation",
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
            float(f): skin_depth(2.0 * np.pi * f, sigma_ref)
            for f in engine.frequencies
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
        soil=world.soil,
        clusters=cluster_members,
        metadata=metadata,
    )
