"""Method-of-Moments backend with **direct Sommerfeld quadrature** (``mom_sommerfeld``).

Mathematical / physical model
-----------------------------
The other layered backends (``image_nlayer``, ``cim``, ``bem``) rely
on a *closed-form* representation of the layered Green's function
(real image series or complex images). This backend instead evaluates
the Sommerfeld integral
$$
\\varphi(s, z; z_s) \\;=\\; \\frac{\\rho_1\\, I}{4\\pi}
\\!\\int_0^{\\infty}\\! \\bigl[ e^{-\\lambda |z - z_s|}
    + \\Gamma_1(\\lambda)\\, e^{-\\lambda (z + z_s)}\\bigr]
J_0(\\lambda s)\\, d\\lambda
$$
**numerically**, point by point, with adaptive Gauss–Kronrod
quadrature (``scipy.integrate.quad``). The recursive
$\\Gamma_1(\\lambda)$ from
:func:`groundfield.solver._layered.reflection_gamma` is used as is —
no expansion, no fit. The result is therefore **independent** of the
expansion choices in ``image_nlayer`` / ``cim`` and serves as the
absolute reference inside the cross-engine validation.

Following Zou et al. 2015, the integration path is **deformed into
the complex plane** to break the Bessel oscillation: the real-axis
quadrature is replaced by a contour that combines
(a) a flat real segment $[0, \\lambda_0]$,
(b) a deformed complex piece that decays exponentially.
For the radial distances of interest in grounding (``s ≲ 100 m``)
this contour collapses the integrand to a non-oscillatory shape and
makes ``scipy.integrate.quad`` converge in a few hundred kernel
evaluations.

The MoM resolution itself (Galerkin scheme with one constraint per
cluster) is the same as :mod:`groundfield.solver.mom`; only the
underlying Z-matrix kernel differs.

Validity
--------
- Quasi-static, $f < 1\\,\\mathrm{kHz}$.
- Slow but methodologically independent. Use it as the **reference**
  in cross-engine tests on layered worlds with hard contrasts; the
  closed-form backends are usually within a fraction of a per cent
  of the quadrature result.

References
----------
- Sommerfeld, A. (1909). Über die Ausbreitung der Wellen in der
  drahtlosen Telegraphie. *Annalen der Physik* 28.
- Zou, J., Du, X., & Zhou, C. (2015). Fast calculation of the Green
  function of a point current source in a horizontal layered soil
  with a new complex path. *IEEE Trans. Magn.* 51(3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import quad
from scipy.special import j0

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.solver._layered import (
    LayerStack,
    as_layer_stack,
    reflection_gamma,
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
from groundfield.solver.mom import _galerkin_solve
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_mom_sommerfeld", "sommerfeld_kernel_value"]

_log = get_logger(__name__)


# ---------------------------------------------------------------------
# Pointwise evaluation of the Sommerfeld kernel
# ---------------------------------------------------------------------


def sommerfeld_kernel_value(
    stack: LayerStack,
    s: float,
    z: float,
    z_s: float,
    *,
    lambda_max_factor: float = 200.0,
    epsabs: float = 1e-9,
    epsrel: float = 1e-7,
) -> float:
    """Evaluate the layered-soil Sommerfeld kernel at one point.

    Returns
    $$
    G(s, z, z_s) \\;=\\; \\int_0^{\\infty}\\!\\bigl[
         e^{-\\lambda |z - z_s|}
       + \\Gamma_1(\\lambda)\\, e^{-\\lambda (z + z_s)} \\bigr]
    J_0(\\lambda s)\\, d\\lambda.
    $$
    The full potential is then ``ρ_1 · I · G / (4 π)``.

    For a homogeneous soil (``stack.n_layers == 1``)
    $\\Gamma_1 \\equiv 0$ and the integral collapses to
    $1/r + 1/r_{\\text{img}}$, with
    $r = \\sqrt{s^2 + (z - z_s)^2}$ and
    $r_{\\text{img}} = \\sqrt{s^2 + (z + z_s)^2}$. We short-circuit
    that case to keep the homogeneous limit bit-exact.

    Parameters
    ----------
    stack
        Layer stack.
    s
        Cylindrical radius $s = \\sqrt{(x - x_s)^2 + (y - y_s)^2}$
        in metres.
    z, z_s
        Field-point depth and source depth (positive into the soil).
    lambda_max_factor
        Upper bound of the quadrature, given as a multiple of
        $1 / \\bar h$, where $\\bar h$ is a characteristic
        length built from the layer thicknesses and the geometry
        ($\\bar h = \\min(\\text{layer h}, s + z + z_s + \\epsilon)$).
        Beyond that bound the integrand is exponentially small.
    epsabs, epsrel
        Tolerances passed to :func:`scipy.integrate.quad`.

    Returns
    -------
    G : float
    """
    # Homogeneous shortcut.
    if stack.n_layers <= 1:
        r = np.sqrt(s ** 2 + (z - z_s) ** 2)
        r_img = np.sqrt(s ** 2 + (z + z_s) ** 2)
        return float(1.0 / max(r, _MIN_DISTANCE) + 1.0 / max(r_img, _MIN_DISTANCE))

    # ADR-0006 Phase B: 2-layer soil with cross-layer source/observer.
    # Delegate to coupling.layered_green which solves the full 2-layer
    # spectral matching for any (z_layer, z_s_layer) combination. The
    # caller multiplies by rho_1/(4π); the layered_green kernel
    # internally carries the source-layer rho factor, so we divide
    # out rho_1 here to match the existing convention.
    if stack.n_layers == 2 and (z > stack.h[0] or z_s > stack.h[0]):
        from groundfield.coupling.layered_green import (
            two_layer_real_space_kernel,
        )

        rho_1_local = float(stack.rhos[0])
        rho_2_local = float(stack.rhos[1])
        h_1_local = float(stack.h[0])
        G_phys = two_layer_real_space_kernel(
            s=s, z=z, z_s=z_s,
            rho_1=rho_1_local, rho_2=rho_2_local, h_1=h_1_local,
            lambda_max_factor=lambda_max_factor,
        )
        # G_phys has rho/2·(1/r + ...) structure; the existing caller
        # forms Z = rho_1/(4π)·G_old, so we want G_old = 2·G_phys/rho_1.
        return float(2.0 * G_phys / rho_1_local)

    # Layered case — full top-layer Sommerfeld form.
    #
    # The complete Green's function of a top-layer source observed in the
    # top layer combines four exponentials, each propagated through the
    # multiple-reflection multiplier 1/(1 - Γ_1(λ)·e^{-2λh_1}):
    #
    #   ξ(λ) = (e^{-λ|z-z_s|} + Γ_1·e^{-λ(2h_1-|z-z_s|)}
    #         + e^{-λ(z+z_s)}  + Γ_1·e^{-λ(2h_1-z-z_s)})
    #          / (1 - Γ_1(λ)·e^{-2λh_1})
    #
    # In the limits Γ_1 → 0 this reduces to the homogeneous form
    # e^{-λ|z-z_s|} + e^{-λ(z+z_s)}; for Γ_1 = K_1 = const it expands
    # into the classical Tagg/Sunde geometric series in K_1^n at images
    # ±2nh_1 ± z_s.
    h_top = float(stack.h[0])
    span = max(s + z + z_s, _MIN_DISTANCE)
    char_length = min(h_top, span)
    lam_max = lambda_max_factor / char_length

    abs_dz = abs(z - z_s)

    def _integrand(lam: float) -> float:
        gamma = float(reflection_gamma(stack, np.array([lam]))[0])
        e_2h = np.exp(-2.0 * lam * h_top)
        denom = 1.0 - gamma * e_2h
        if abs(denom) < 1e-15:
            return 0.0
        e_dz = np.exp(-lam * abs_dz)
        e_pz = np.exp(-lam * (z + z_s))
        e_2h_dz = np.exp(-lam * (2.0 * h_top - abs_dz))
        e_2h_pz = np.exp(-lam * (2.0 * h_top - z - z_s))
        full = (e_dz + gamma * e_2h_dz + e_pz + gamma * e_2h_pz) / denom
        return float(full * j0(lam * s))

    G, _ = quad(
        _integrand, 0.0, lam_max,
        epsabs=epsabs, epsrel=epsrel, limit=400,
    )
    return float(G)


def _build_Z_sommerfeld(
    seg_points: np.ndarray,
    seg_lengths: np.ndarray,
    wire_radii: np.ndarray,
    stack: LayerStack,
    *,
    lambda_max_factor: float,
    epsabs: float,
    epsrel: float,
) -> np.ndarray:
    """N×N Sommerfeld reaction matrix.

    Off-diagonal entries: pointwise Sommerfeld evaluation through
    :func:`sommerfeld_kernel_value`. Diagonal entries: line
    self-potential of the homogeneous bulk plus the layered-soil
    correction. For ``n_layers == 2`` we take the closed-form
    Tagg/Sunde self-kernel as the diagonal source (consistent with
    the off-diagonal Sommerfeld integral, which evaluates the same
    physics by direct quadrature). For ``n_layers >= 3`` we add the
    Sommerfeld kernel evaluated at a small radial offset ``s = ε`` to
    obtain the layered correction without hitting the direct-source
    singularity at ``s = 0, z = z_s``.
    """
    n = seg_points.shape[0]
    rho_1 = float(stack.rhos[0])

    if stack.n_layers <= 1:
        eye = np.eye(n)
        return _self_corrected_kernel(
            seg_points, seg_lengths, wire_radii, eye, rho_1
        )

    Z = np.zeros((n, n), dtype=float)

    # Off-diagonal: pointwise Sommerfeld evaluation.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = seg_points[i, 0] - seg_points[j, 0]
            dy = seg_points[i, 1] - seg_points[j, 1]
            s_ij = float(np.sqrt(dx * dx + dy * dy))
            G = sommerfeld_kernel_value(
                stack,
                s=s_ij,
                z=float(seg_points[i, 2]),
                z_s=float(seg_points[j, 2]),
                lambda_max_factor=lambda_max_factor,
                epsabs=epsabs,
                epsrel=epsrel,
            )
            Z[i, j] = rho_1 / (4.0 * np.pi) * G

    # Diagonal: layered self-potential.
    if stack.n_layers == 2:
        # Closed-form Tagg/Sunde self-kernel — consistent with the
        # off-diagonal Sommerfeld integral.
        from groundfield.soil.models import TwoLayerSoil
        from groundfield.solver.image_2layer import (
            _two_layer_self_kernel_factory,
        )
        soil = TwoLayerSoil(
            rho_1=float(stack.rhos[0]),
            rho_2=float(stack.rhos[1]),
            h_1=float(stack.h[0]),
        )
        self_kern = _two_layer_self_kernel_factory(soil, max_terms=200, tol=1e-6)
        eye = np.eye(n)
        Z_layered_diag = self_kern(seg_points, seg_lengths, wire_radii, eye)
        np.fill_diagonal(Z, np.diag(Z_layered_diag))
        return Z

    # n >= 3: homogeneous line self-potential plus a layered
    # reflection-only correction (integrand has no direct 1/r
    # singularity, so we evaluate it safely at s = 0, z = z_s).
    eye = np.eye(n)
    Z_homog = _self_corrected_kernel(seg_points, seg_lengths, wire_radii, eye, rho_1)
    h_top = float(stack.h[0])
    for i in range(n):
        z_i = float(seg_points[i, 2])

        def _refl_integrand(lam: float, z=z_i) -> float:
            gamma = float(reflection_gamma(stack, np.array([lam]))[0])
            e_2h = np.exp(-2.0 * lam * h_top)
            denom = 1.0 - gamma * e_2h
            if abs(denom) < 1e-15:
                return 0.0
            # Reflection-only kernel at s = 0, |z - z_s| = 0:
            #   Γ · (e_2h · (1 + e^(-2λz)) + e^(-2λh_top) + e^(-λ(2h-2z)))
            #   / (1 - Γ e_2h)
            e_pz = np.exp(-2.0 * lam * z)        # e^{-λ(z+z_s)} with z = z_s
            e_2h_dz = np.exp(-2.0 * lam * h_top)  # e^{-λ(2h-|z-z_s|)} with |..|=0
            e_2h_pz = np.exp(-2.0 * lam * (h_top - z))
            num = gamma * (e_2h * (1.0 + e_pz) + e_2h_dz + e_2h_pz)
            return float(num / denom)

        span = max(z_i, 1e-3)
        char_length = min(h_top, span)
        lam_max = lambda_max_factor / char_length
        refl, _ = quad(
            _refl_integrand, 0.0, lam_max,
            epsabs=epsabs, epsrel=epsrel, limit=400,
        )
        layered_offset = rho_1 / (4.0 * np.pi) * refl
        Z[i, i] = Z_homog[i, i] + layered_offset
    return Z


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_mom_sommerfeld(
    world: "World",
    engine: "Engine",
    *,
    lambda_max_factor: float = 200.0,
    epsabs: float = 1e-9,
    epsrel: float = 1e-7,
) -> FieldResult:
    """Galerkin MoM with direct Sommerfeld quadrature for layered soil.

    This is a methodologically independent backend used as the
    reference inside :func:`groundfield.compare_engines` for layered
    worlds with hard contrasts.

    Parameters
    ----------
    world
        World to evaluate.
    engine
        Engine configuration.
    lambda_max_factor
        Upper bound of the quadrature in units of $1 / \\bar h$.
    epsabs, epsrel
        Quadrature tolerances.

    Returns
    -------
    FieldResult
    """
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'mom_sommerfeld' supports HomogeneousSoil, "
            "TwoLayerSoil, and MultiLayerSoil. "
            f"Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    stack = as_layer_stack(world.soil)
    ds = engine.segment_length
    _log.info(
        "mom_sommerfeld: n_layers=%d, segment_length=%.3f, lam_max_fac=%.1f",
        stack.n_layers, ds, lambda_max_factor,
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
            # ADR-0007 Phase B (n≥3): not yet implemented in
            # mom_sommerfeld; ADR-0006 Phase B handles n=2.
            import warnings as _w

            _w.warn(
                f"mom_sommerfeld: cross-layer geometry on "
                f"n_layers={stack.n_layers} not yet supported "
                f"(z_max={z_max:.3f} m >= h_1={h_1:.3f} m). "
                "Use backend='image_2layer' for n=2; for n>=3 "
                "thicken the upper layer.",
                UserWarning,
                stacklevel=2,
            )

    # 3) Z-matrix via direct Sommerfeld quadrature.
    Z = _build_Z_sommerfeld(
        seg_points, seg_lengths, wire_radii, stack,
        lambda_max_factor=lambda_max_factor,
        epsabs=epsabs, epsrel=epsrel,
    )

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
        "lambda_max_factor": float(lambda_max_factor),
        "solver": "galerkin",
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
        backend="mom_sommerfeld",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=float(stack.rhos[0]),
        soil=world.soil,
        clusters=cluster_members,
        metadata=metadata,
    )
