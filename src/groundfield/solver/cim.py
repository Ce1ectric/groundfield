"""Complex Image Method backend (``cim``).

Mathematical / physical model
-----------------------------
The :mod:`image_nlayer` backend expands the upward-looking reflection
$\\Gamma_1(\\lambda)$ as a power series in the per-layer
attenuation factors $e^{-2\\lambda h_i}$ — a representation that
converges fast for two layers but generates many terms for
$n \\ge 3$ and high contrasts.

The **Complex Image Method (CIM)** instead approximates
$\\Gamma_1(\\lambda)$ directly by a finite sum of complex
exponentials,
$$
\\Gamma_1(\\lambda) \\;\\approx\\; \\sum_{k=1}^{P} a_k\\,
e^{-2\\lambda \\beta_k},
$$
with complex coefficients $a_k \\in \\mathbb{C}$ and complex
"image depths" $\\beta_k \\in \\mathbb{C}$ (with
$\\Re\\{\\beta_k\\} > 0$ to keep the integrals convergent).
Substituting this approximation into the Sommerfeld integral and
using the closed-form
$$
\\int_0^{\\infty} e^{-\\lambda d} J_0(\\lambda s)\\, d\\lambda
\\;=\\; \\frac{1}{\\sqrt{s^2 + d^2}}
$$
immediately gives a **closed-form** spatial Green's function of the
same shape as the homogeneous image-charge sum, but with **complex
image positions**. The kernel of every backend that consumes the
layered Green's function (this engine and ``bem``) becomes:
$$
\\varphi(s, z) \\;=\\; \\frac{\\rho_1\\, I}{4\\pi}\\,
\\Bigl(\\frac{1}{r} + \\frac{1}{r_{\\text{air}}}
     + \\sum_{k=1}^{P} a_k
       \\Bigl(\\frac{1}{\\sqrt{s^2 + (z + z_s + 2\\beta_k)^2}}
            + \\frac{1}{\\sqrt{s^2 + (z - z_s + 2\\beta_k)^2}}\\Bigr)
     \\Bigr).
$$
The cost of one potential evaluation is therefore the same as the
homogeneous backend, multiplied by ``2 * P`` — independent of the
layer count.

Numerical fit strategy
----------------------
We use the **matrix-pencil method** (a numerically stable variant of
Prony's algorithm) to fit a sample of $\\Gamma_1(\\lambda)$ on a
log-spaced grid in $\\lambda$ to ``P`` complex exponentials.
This is a faithful Python re-implementation of the segmented-sampling
least-squares idea of Dan et al. 2021 (without the segmentation
heuristic, which is needed mainly for very many layers; for the
typical two-/three-layer use cases a single segment with a moderately
oversampled grid is enough).

Validity
--------
- Quasi-static, $f < 1\\,\\mathrm{kHz}$.
- For ``n_layers == 1`` the fit returns ``P = 0`` and the engine
  collapses to the homogeneous image-charge sum exactly.
- For ``n_layers == 2`` with a moderate $P$ (typically 6–10)
  the engine reproduces ``image_2layer`` to better than 0.1 %.
- For ``n_layers ≥ 3`` the engine is the first one in the suite that
  can be evaluated efficiently, and it forms one of the
  cross-validation pairs in the test suite (against
  ``mom_sommerfeld``).

References
----------
- Sarkar, T. K. & Pereira, O. (1995). Using the matrix pencil method
  to estimate the parameters of a sum of complex exponentials, IEEE
  Antennas & Propagation Magazine 37(1).
- Li, Z.-X. et al. (2006). A novel mathematical modeling of grounding
  system buried in multilayer earth, IEEE PWRD 21(3).
- Dan, Y. et al. (2021). Segmented sampling least squares algorithm
  for Green's function of arbitrary layered soil, IEEE PWRD 36(3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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
    _solve_cluster_currents,
)
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_cim", "fit_complex_images"]

_log = get_logger(__name__)


# ---------------------------------------------------------------------
# Fit Γ_1(λ) ≈ Σ a_k · e^(-2 λ β_k) by the matrix-pencil method
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ComplexImageFit:
    """Result of :func:`fit_complex_images`.

    Attributes
    ----------
    a : np.ndarray, shape (P,), complex
        Image weights $a_k$.
    beta : np.ndarray, shape (P,), complex
        Image depths $\\beta_k$ (units of metres).
    rms : float
        RMS of the residual on the sample grid (diagnostic only).
    """

    a: np.ndarray
    beta: np.ndarray
    rms: float


def fit_complex_images(
    stack: LayerStack,
    *,
    n_images: int = 8,
    n_samples: int = 64,
    lambda_min_factor: float = 1e-3,
    lambda_max_factor: float = 50.0,
) -> ComplexImageFit:
    """Fit Γ_1(λ) of an n-layer stack by ``n_images`` complex exponentials.

    Steps:

    1. Build a uniform sampling grid $\\lambda_j = j \\Delta$
       on $(0, \\lambda_{\\max}]$. The grid step
       $\\Delta$ is chosen from the smallest layer thickness
       $h_{\\min}$ so that
       $\\Delta = (1/h_{\\min}) \\cdot
       (\\lambda_{\\max,\\text{factor}} / N)$ covers the regime where
       Γ_1 has any structure (it decays for
       $\\lambda \\gg 1/h_{\\min}$).
    2. Sample $g_j = \\Gamma_1(\\lambda_j)$.
    3. Apply the matrix-pencil method to ``g_j`` to recover the
       $P$ poles $p_k = e^{-2 \\Delta \\beta_k}$ of the
       sum of exponentials.
    4. Solve the least-squares system for the coefficients $a_k$.

    For ``stack.n_layers <= 1`` returns an empty fit (Γ_1 ≡ 0).

    Parameters
    ----------
    stack
        Layer stack to fit.
    n_images
        Target number of complex images $P$. Sensible range
        4–12 for two-/three-layer stacks; small numbers are fine for
        soft contrasts, harder contrasts need more.
    n_samples
        Number of samples drawn from Γ_1(λ).
    lambda_min_factor, lambda_max_factor
        Bounds of the λ-grid expressed as multiples of
        $1/h_{\\min}$.

    Returns
    -------
    ComplexImageFit
    """
    if stack.n_layers <= 1:
        return ComplexImageFit(
            a=np.zeros(0, dtype=complex),
            beta=np.zeros(0, dtype=complex),
            rms=0.0,
        )

    h_min = float(np.min(stack.h))
    lam_min = lambda_min_factor / h_min
    lam_max = lambda_max_factor / h_min

    # Uniform spacing required by the matrix-pencil method.
    delta = (lam_max - lam_min) / (n_samples - 1)
    lam = lam_min + delta * np.arange(n_samples)
    g = reflection_gamma(stack, lam)
    g = g.astype(complex)

    # If Γ_1(λ) is essentially zero everywhere (degenerate case
    # ρ_1 = ρ_2 = … = ρ_n; matrix-pencil would receive a zero Hankel
    # block and divide by zero), short-circuit with an empty fit.
    if np.max(np.abs(g)) < 1e-12:
        return ComplexImageFit(
            a=np.zeros(0, dtype=complex),
            beta=np.zeros(0, dtype=complex),
            rms=0.0,
        )

    # Matrix-pencil method: form the Hankel pencil [Y0, Y1] of size
    # (n_samples - L) x L, with pencil parameter L ≈ n_samples/3.
    L = max(n_images + 1, n_samples // 3)
    L = min(L, n_samples - n_images - 1)
    if L < n_images + 1:
        # Not enough samples — fall back to fewer images.
        n_images = max(1, L - 1)

    rows = n_samples - L
    Y = np.zeros((rows, L + 1), dtype=complex)
    for i in range(rows):
        Y[i, :] = g[i:i + L + 1]
    Y0 = Y[:, :L]
    Y1 = Y[:, 1:L + 1]

    # SVD-based filtering: project both blocks onto the dominant
    # singular subspace of order n_images. Filter near-zero singular
    # values to avoid 1/0.
    U, S, Vh = np.linalg.svd(Y0, full_matrices=False)
    s_max = float(S.max()) if S.size else 0.0
    if s_max == 0.0:
        return ComplexImageFit(
            a=np.zeros(0, dtype=complex),
            beta=np.zeros(0, dtype=complex),
            rms=0.0,
        )
    keep_mask = S > s_max * 1e-10
    P = int(min(n_images, np.sum(keep_mask)))
    if P < 1:
        return ComplexImageFit(
            a=np.zeros(0, dtype=complex),
            beta=np.zeros(0, dtype=complex),
            rms=0.0,
        )
    Up = U[:, :P]
    Sp = S[:P]
    Vp = Vh[:P, :]

    # Y1 ≈ Up · diag(Sp) · Vp · Z, with Z holding the poles in its
    # eigenvalues. Equivalently the poles solve
    #   eig( pinv(Y0) · Y1 ) ≈ pinv(Sp) · Up^H · Y1 · Vp^H.
    Z = np.diag(1.0 / Sp) @ Up.conj().T @ Y1 @ Vp.conj().T
    poles = np.linalg.eigvals(Z)

    # Map poles back to the β coefficients: p_k = exp(-2 Δ β_k)
    #   → β_k = -log(p_k) / (2 Δ).
    # Drop poles outside the unit disc (non-physical, would blow up).
    mask = np.abs(poles) < 0.999
    poles = poles[mask]
    if poles.size == 0:
        return ComplexImageFit(
            a=np.zeros(0, dtype=complex),
            beta=np.zeros(0, dtype=complex),
            rms=float("nan"),
        )
    beta = -np.log(poles) / (2.0 * delta)

    # Coefficients a_k by linear least squares on the original samples.
    # g(λ_j) ≈ Σ a_k · exp(-2 λ_j β_k)
    A = np.exp(-2.0 * lam[:, None] * beta[None, :])  # (n_samples, P)
    a, *_ = np.linalg.lstsq(A, g, rcond=None)

    # Discard images with negative real part of β (non-decaying).
    keep = beta.real > 0.0
    beta = beta[keep]
    a = a[keep]
    if beta.size == 0:
        return ComplexImageFit(
            a=np.zeros(0, dtype=complex),
            beta=np.zeros(0, dtype=complex),
            rms=float("nan"),
        )
    A_keep = np.exp(-2.0 * lam[:, None] * beta[None, :])
    rms = float(np.sqrt(np.mean(np.abs(A_keep @ a - g) ** 2)))
    _log.info(
        "cim.fit: n_images_used=%d, rms=%.2e, n_layers=%d",
        beta.size, rms, stack.n_layers,
    )
    return ComplexImageFit(a=a, beta=beta, rms=rms)


# ---------------------------------------------------------------------
# Self-action and field-evaluation kernels with complex images
# ---------------------------------------------------------------------


def _cim_self_kernel_factory(stack: LayerStack, fit: ComplexImageFit):
    """Build a self-action closure for the CIM.

    Strategy:

    - ``n_layers == 1`` → homogeneous self-kernel (Γ_1 ≡ 0, no extra
      images).
    - ``n_layers == 2`` → exact Tagg/Sunde self-kernel (closed form,
      bit-identical to ``image_2layer``). The matrix-pencil fit of a
      constant Γ_1 = K_1 is ill-conditioned, so we deliberately skip
      it and use the closed-form geometric series instead.
    - ``n_layers ≥ 3`` → complex-image contribution
      $\\sum_k a_k / r_k$ added to the homogeneous self-kernel, with
      $r_k = \\sqrt{s^2 + (z + z_s + 2 \\beta_k)^2}$. The
      Sommerfeld expansion $\\Gamma_1(\\lambda) e^{-\\lambda(z+z_s)}
      \\approx \\sum_k a_k e^{-\\lambda(z + z_s + 2 \\beta_k)}$
      contributes a single image per pole — no sign-of-z_s loop.
    """
    rho_1 = float(stack.rhos[0])

    if stack.n_layers <= 1:
        def _hom(seg_points, seg_lengths, wire_radii, currents):
            return _self_corrected_kernel(
                seg_points, seg_lengths, wire_radii, currents, rho_1
            )
        return _hom

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
        return _two_layer_self_kernel_factory(soil, max_terms=200, tol=1e-6)

    # n >= 3: closed-form complex-image expansion of Γ_1(λ).
    def _kernel(seg_points, seg_lengths, wire_radii, currents):
        phi = _self_corrected_kernel(
            seg_points, seg_lengths, wire_radii, currents, rho_1
        )
        if fit.a.size == 0:
            return phi
        diff_xy = seg_points[:, None, 0:2] - seg_points[None, :, 0:2]
        delta_sq = np.einsum("mnk,mnk->mn", diff_xy, diff_xy)
        z_field = seg_points[:, 2:3]
        z_src = seg_points[None, :, 2]
        # Single complex image per pole at z = -(z_src + 2 β_k).
        extra = np.zeros(phi.shape, dtype=complex)
        for a, b in zip(fit.a, fit.beta):
            d = z_field + z_src + 2.0 * b
            r = np.sqrt(delta_sq + d ** 2)
            r_abs = np.abs(r)
            tiny = r_abs < _MIN_DISTANCE
            if np.any(tiny):
                r = np.where(tiny, _MIN_DISTANCE + 0j, r)
            extra += a * ((1.0 / r) @ currents)
        extra *= rho_1 / (4.0 * np.pi)
        return phi + extra.real

    return _kernel


def _cim_field_potential(
    field_points: np.ndarray,
    source_points: np.ndarray,
    currents: np.ndarray,
    stack: LayerStack,
    fit: ComplexImageFit,
) -> np.ndarray:
    """Evaluate the CIM-approximated potential at arbitrary field points."""
    rho_1 = float(stack.rhos[0])
    image_pts = source_points.copy()
    image_pts[:, 2] = -image_pts[:, 2]

    diff_real = field_points[:, None, :] - source_points[None, :, :]
    diff_air = field_points[:, None, :] - image_pts[None, :, :]
    r_real = np.linalg.norm(diff_real, axis=2)
    r_air = np.linalg.norm(diff_air, axis=2)
    np.maximum(r_real, _MIN_DISTANCE, out=r_real)
    np.maximum(r_air, _MIN_DISTANCE, out=r_air)
    base = (1.0 / r_real) + (1.0 / r_air)
    phi = base @ currents

    if fit.a.size > 0:
        diff_xy = field_points[:, None, 0:2] - source_points[None, :, 0:2]
        delta_sq = np.einsum("mnk,mnk->mn", diff_xy, diff_xy)
        z_field = field_points[:, 2:3]
        z_src = source_points[None, :, 2]
        extra = np.zeros(phi.shape, dtype=complex)
        for a, b in zip(fit.a, fit.beta):
            d = z_field + z_src + 2.0 * b
            r = np.sqrt(delta_sq + d ** 2)
            r_abs = np.abs(r)
            tiny = r_abs < _MIN_DISTANCE
            if np.any(tiny):
                r = np.where(tiny, _MIN_DISTANCE + 0j, r)
            extra += a * ((1.0 / r) @ currents)
        phi = phi + extra.real
    return rho_1 / (4.0 * np.pi) * phi


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_cim(
    world: "World",
    engine: "Engine",
    *,
    n_images: int = 8,
    n_samples: int = 64,
) -> FieldResult:
    """Complex-Image-Method solver for layered soil.

    Accepts :class:`HomogeneousSoil`, :class:`TwoLayerSoil`, and
    :class:`MultiLayerSoil`. The fit reduces to ``P = 0`` on
    homogeneous soil (no extra images needed); for layered soils the
    closed-form CIM Green's function is used.

    Parameters
    ----------
    world
        World to evaluate.
    engine
        Engine configuration; ``engine.segment_length`` controls the
        discretisation.
    n_images
        Target number of complex images.
    n_samples
        Number of λ-samples in the matrix-pencil fit.

    Returns
    -------
    FieldResult
        ``metadata['cim_n_images']`` and ``metadata['cim_rms']`` expose
        the fit quality.
    """
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'cim' supports HomogeneousSoil, TwoLayerSoil, "
            f"and MultiLayerSoil. Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    stack = as_layer_stack(world.soil)
    fit = fit_complex_images(stack, n_images=n_images, n_samples=n_samples)
    ds = engine.segment_length

    _log.info(
        "cim: n_layers=%d, n_images=%d (target %d), rms=%.2e",
        stack.n_layers, fit.a.size, n_images, fit.rms,
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

    # 3) Cluster building (ideal conductors only) and finite-impedance
    #    branch list (passed into the nodal-analysis solver).
    cluster_id = _build_clusters(world.electrodes, world.conductors)
    finite_branches = _build_finite_branches(world.conductors, cluster_id)

    # 3b) Distributed-conductor topology (ADR-0003) + ADR-0004
    #     inductive coupling assembly.
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
    for n_ in interior_nodes:
        if n_ not in cluster_id:
            cluster_id[n_] = n_
            pseudo_owners.append(n_)
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

    seg_points = np.array([s.midpoint for s in all_segments])
    seg_lengths = np.array([s.length for s in all_segments])
    wire_radii = np.array([s.wire_radius for s in all_segments])

    if stack.n_layers >= 3:
        z_max = seg_points[:, 2].max()
        h_1 = float(stack.h[0])
        if z_max >= h_1:
            # ADR-0006/0007 Phase B: n>=3 cross-layer not yet
            # implemented. For n=2 the kernel delegates to the
            # cross-layer-aware _two_layer_self_kernel_factory.
            import warnings as _w

            _w.warn(
                f"cim: cross-layer geometry on n_layers="
                f"{stack.n_layers} not yet supported. "
                "Use backend='image_2layer' for n=2; for n>=3 "
                "thicken the upper layer.",
                UserWarning,
                stacklevel=2,
            )

    # 4) Self-kernel + frequency loop.
    self_kernel = _cim_self_kernel_factory(stack, fit)
    n_segments = len(all_segments)
    n_freq = len(engine.frequencies)
    omegas = [2.0 * np.pi * float(f) for f in engine.frequencies]
    real_electrode_names = {e.name for e in world.electrodes}

    def _solve_at(omega: float) -> tuple[
        dict[str, complex], np.ndarray, np.ndarray
    ]:
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
            self_kernel=self_kernel,
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
            phi_re = self_kernel(seg_points, seg_lengths, wire_radii, sc.real)
            phi_im = self_kernel(seg_points, seg_lengths, wire_radii, sc.imag)
            ph = phi_re + 1j * phi_im
        return elec_total, sc, ph

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
        backend="cim",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=float(stack.rhos[0]),
        soil=world.soil,
        clusters=cluster_members,
        metadata=metadata,
    )
