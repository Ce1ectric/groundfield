"""Finite-Element-Method backend (``fem``).

Mathematical / physical model
-----------------------------
The other engines in the family solve the **integral form** of the
quasi-static current-flow problem (image charges, BEM, MoM with a
layered Green's function). This backend instead discretises the
**volume PDE** directly,
$$
-\\nabla \\cdot (\\sigma(\\mathbf{r})\\, \\nabla \\varphi) \\;=\\; q,
\\qquad \\sigma(\\mathbf{r}) = 1/\\rho(\\mathbf{r}),
$$
with Neumann boundary at the soil surface
($\\partial \\varphi / \\partial z = 0$ at $z = 0$,
electrically insulating air) and a Dirichlet far-field
($\\varphi \\to 0$ as $|\\mathbf{r}| \\to \\infty$,
truncated to a finite outer radius $R_{\\text{far}}$).
$q$ is the current source density.

For work package 1, where most reference electrodes are essentially
**axisymmetric** around their connection point (a single rod, a ring,
a hemisphere), we exploit the symmetry and discretise the problem on
a 2-D $(s, z)$ mesh with cylindrical coordinates. The PDE
becomes
$$
-\\frac{1}{s} \\frac{\\partial}{\\partial s}\\!
\\left(s\\, \\sigma\\, \\frac{\\partial \\varphi}{\\partial s}\\right)
- \\frac{\\partial}{\\partial z}\\!
  \\left(\\sigma\\, \\frac{\\partial \\varphi}{\\partial z}\\right)
\\;=\\; q,
$$
solved on a triangular finite-element mesh covering
$(s, z) \\in [0, R_{\\text{far}}] \\times [0, Z_{\\text{far}}]$
with linear hat functions. The weak form is assembled with a sparse
COO-builder; the linear system is solved with
``scipy.sparse.linalg.spsolve``.

Layer model
-----------
Layer boundaries enter through the piecewise-constant conductivity
$\\sigma(z)$: each element gets the conductivity of the layer
its centroid sits in. The PDE handles arbitrary horizontally
stratified soils (any ``n``).

Scope
-----
- **Geometry coverage.** The axisymmetric formulation captures
  :class:`RodElectrode` (vertical rod, ``s = 0``) and
  :class:`RingElectrode` and :class:`MeshElectrode` *as effective
  hemispheres* — the equivalent-hemisphere radius is computed from
  the electrode's geometric parameters before the FEM run. This is
  the standard reduction used in dissertation-level reference
  comparisons (see Sunde 1968 ch. 2.1, Dwight 1936): a ring or mesh
  electrode of effective area $A$ and effective length
  $L$ is replaced by the hemisphere of radius
  $a_{\\text{eq}}$ that produces the same DC resistance in
  homogeneous soil. The replacement is exact only for hemispheres,
  good (better than 5 %) for rings and shallow meshes, and
  documented as a known approximation.
- **Multi-electrode.** Multiple electrodes are aggregated into one
  effective hemisphere centred at the centroid of the cluster — the
  ``fem`` backend therefore reports cluster-level results rather than
  per-electrode currents. For a single cluster (the typical AP1
  case) the approximation is appropriate.
- **Frequency.** Quasi-static, frequency-independent.

The FEM backend's purpose in the engine family is to provide a
**volume-PDE cross-check** that does not share any code path with
the integral-equation engines. Where it disagrees with the others on
simple geometries by more than a few per cent, the source is
typically the equivalent-hemisphere reduction described above (and
documented in the result metadata).

References
----------
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover, ch. 2.1.
- Dwight, H. B. (1936). Calculation of resistances to ground.
- Güemes, J. A., & Hernando, F. E. (2004). Method for calculating
  the ground resistance of grounding grids using FEM. *IEEE PWRD*
  19(2).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from groundfield.geometry.electrodes import (
    GridMeshElectrode,
    MeshElectrode,
    RingElectrode,
    RodElectrode,
    StripElectrode,
)
from groundfield.references import dwight1936 as dw
from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.solver._layered import LayerStack, as_layer_stack
from groundfield.solver.image import (
    _build_clusters,
    _build_finite_branches,
)
from groundfield.solver.result import FieldResult, PointSource
from groundfield.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.geometry.electrodes import _ElectrodeBase
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = ["solve_fem", "equivalent_hemisphere_radius"]

_log = get_logger(__name__)


# ---------------------------------------------------------------------
# Equivalent-hemisphere reduction
# ---------------------------------------------------------------------


def equivalent_hemisphere_radius(
    electrode: "_ElectrodeBase", rho_top: float
) -> float:
    """Equivalent-hemisphere radius giving the same homogeneous-soil
    resistance as ``electrode``.

    Uses the closed-form Dwight 1936 formulas through
    :mod:`groundfield.references.dwight1936`. The hemisphere radius is
    $$
    a_{\\text{eq}} \\;=\\; \\frac{\\rho}{2 \\pi R_{\\text{Dwight}}}.
    $$
    Parameters
    ----------
    electrode
        Single electrode primitive.
    rho_top
        Resistivity used inside the Dwight formula. For layered soil
        the top-layer resistivity is the natural choice — the FEM
        then re-solves the actual layered problem on the equivalent
        hemisphere.
    """
    if isinstance(electrode, RodElectrode):
        R = dw.rod(rho=rho_top, length=electrode.length, radius=electrode.wire_radius)
    elif isinstance(electrode, RingElectrode):
        R = dw.buried_ring(
            rho=rho_top,
            ring_diameter=2.0 * electrode.radius,
            wire_diameter=2.0 * electrode.wire_radius,
            depth=max(electrode.center[2], 1e-3),
        )
    elif isinstance(electrode, StripElectrode):
        # Straight horizontal wire — Dwight's classic ``horizontal_wire``
        # formula expects the *half-length* L (total = 2 L).
        L = electrode.length
        depth = max(electrode.start[2], 1e-3)
        R = dw.horizontal_wire(
            rho=rho_top, length=L / 2.0,
            radius=electrode.wire_radius, depth=depth,
        )
    elif isinstance(electrode, GridMeshElectrode):
        # Schwarz / Sverak / IEEE Std 80 (Sverak 1981) formula for a
        # buried rectangular meshed grid:
        #
        #     R ≈ ρ / L_C + ρ / sqrt(20 A) · (1 + 1 / (1 + h sqrt(20/A)))
        #
        # with L_C the total buried wire length, A the grid footprint
        # area and h the burial depth. Captures the dependence on the
        # inner mesh density, which the simple strip-along-diagonal
        # approximation used for the legacy ``MeshElectrode`` misses.
        dx, dy = electrode.size
        depth = max(electrode.corner[2], 1e-3)
        A = dx * dy
        n_long = electrode.n_y + 1   # longitudinal wires (one per y-row)
        n_tran = electrode.n_x + 1   # transverse wires (one per x-column)
        L_C = n_long * dx + n_tran * dy
        R = rho_top / L_C + (rho_top / math.sqrt(20.0 * A)) * (
            1.0 + 1.0 / (1.0 + depth * math.sqrt(20.0 / A))
        )
    elif isinstance(electrode, MeshElectrode):
        # Use the strip approximation along the diagonal as a rough
        # proxy for a mesh ground electrode. The FEM result for a
        # mesh is dominated by its overall extent, not the inner
        # spacing, so this is acceptable for the cross-check role.
        dx, dy = electrode.size
        diag = float(np.hypot(dx, dy))
        # horizontal_strip expects half-length L (total length 2*L),
        # a strip cross-section width and thickness. We model the
        # mesh as one equivalent strip of width ≈ wire_radius and
        # thickness ≈ wire_radius / 9 (so the b<a/8 guard passes).
        a = max(2.0 * electrode.wire_radius, 0.01)
        b = a / 9.0
        R = dw.horizontal_strip(
            rho=rho_top,
            length=diag / 2.0,
            width=a,
            thickness=b,
            depth=max(electrode.corner[2], 1e-3),
        )
    else:
        raise TypeError(
            f"FEM backend cannot reduce {type(electrode).__name__} to a "
            "hemisphere — extend equivalent_hemisphere_radius()."
        )
    return float(rho_top / (2.0 * np.pi * R))


# ---------------------------------------------------------------------
# Triangular axisymmetric mesh + assembly
# ---------------------------------------------------------------------


def _build_axisymmetric_mesh(
    a_eq: float,
    h_layers: list[float],
    *,
    r_far_factor: float = 30.0,
    z_far_factor: float = 20.0,
    n_radial: int = 60,
    n_axial: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a triangular mesh on $[0, R] \\times [0, Z]$.

    Parameters
    ----------
    a_eq
        Equivalent hemisphere radius (used to set the inner boundary).
    h_layers
        Finite layer thicknesses (used to fix mesh lines at the layer
        interfaces — guarantees the conductivity step is mesh-aligned).
    r_far_factor, z_far_factor
        Outer truncation given as multiples of ``a_eq + max(h_layers)``.
    n_radial, n_axial
        Initial vertex counts in $s$ and $z$.

    Returns
    -------
    nodes : np.ndarray, shape (Nv, 2)
        Vertex coordinates (s, z).
    triangles : np.ndarray, shape (Nt, 3), int
        Per-triangle vertex indices.
    z_layer_idx : np.ndarray, shape (Nt,), int
        Layer index of every triangle (0-based).
    """
    base_length = a_eq + (sum(h_layers) if h_layers else 1.0)
    R_far = r_far_factor * base_length
    Z_far = z_far_factor * base_length

    # Radial nodes: log-spaced from a_eq to R_far + dense cluster near the source.
    s_nodes = np.geomspace(a_eq * 0.05, R_far, n_radial)
    s_nodes = np.concatenate(([0.0], s_nodes))
    s_nodes = np.unique(np.round(s_nodes, 6))

    # Axial nodes: layer interfaces plus geometric refinement near the surface.
    z_axis = list(np.linspace(0.0, max(Z_far, base_length), n_axial))
    if h_layers:
        cum = 0.0
        for h in h_layers:
            cum += h
            z_axis.append(cum)
    z_axis = np.array(sorted(set(np.round(z_axis, 6))))
    z_axis = z_axis[(z_axis >= 0.0) & (z_axis <= Z_far)]

    # Cartesian product → vertices.
    S, Z = np.meshgrid(s_nodes, z_axis, indexing="xy")
    nodes = np.stack([S.ravel(), Z.ravel()], axis=1)

    # Triangulate each grid cell into two right triangles.
    nx = s_nodes.size
    ny = z_axis.size
    tris: list[tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = j * nx + (i + 1)
            n01 = (j + 1) * nx + i
            n11 = (j + 1) * nx + (i + 1)
            tris.append((n00, n10, n11))
            tris.append((n00, n11, n01))
    triangles = np.array(tris, dtype=int)

    # Assign each triangle to a layer.
    if h_layers:
        z_centroid = nodes[triangles, 1].mean(axis=1)
        layer_idx = np.zeros(triangles.shape[0], dtype=int)
        cum = 0.0
        for k, h in enumerate(h_layers):
            cum_next = cum + h
            mask = (z_centroid >= cum) & (z_centroid < cum_next)
            layer_idx[mask] = k
            cum = cum_next
        layer_idx[z_centroid >= cum] = len(h_layers)
    else:
        layer_idx = np.zeros(triangles.shape[0], dtype=int)

    return nodes, triangles, layer_idx


def _assemble_stiffness_axisymmetric(
    nodes: np.ndarray,
    triangles: np.ndarray,
    sigma_per_triangle: np.ndarray,
) -> coo_matrix:
    """Assemble the axisymmetric weak-form stiffness matrix.

    The bilinear form for the cylindrical Laplacian with linear hat
    functions on a 2-D triangle $T$ becomes
    $$
    a_{ij}^T \\;=\\; 2\\pi\\, \\sigma_T \\bar s_T\\,
                     \\bigl(\\nabla \\phi_i \\cdot \\nabla \\phi_j\\bigr)
                     \\, |T|,
    $$
    with $\\bar s_T$ the centroid radius and $|T|$ the
    area of the triangle in the $(s, z)$ plane.
    """
    Nv = nodes.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for t_idx, t in enumerate(triangles):
        v = nodes[t]                                  # (3, 2)
        x = v[:, 0]
        y = v[:, 1]
        # Triangle area (2-D, planar).
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area <= 0.0:
            continue
        s_bar = float(np.mean(x))
        # Hat-function gradients: rows of  (1/(2 area)) · adj.
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2.0 * area)
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2.0 * area)
        sigma = sigma_per_triangle[t_idx]
        # 3x3 element matrix.
        K_e = sigma * (np.outer(b, b) + np.outer(c, c)) * area * 2.0 * np.pi * s_bar
        for i_local in range(3):
            for j_local in range(3):
                rows.append(int(t[i_local]))
                cols.append(int(t[j_local]))
                vals.append(float(K_e[i_local, j_local]))
    K = coo_matrix((vals, (rows, cols)), shape=(Nv, Nv)).tocsr()
    return K


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def solve_fem(
    world: "World",
    engine: "Engine",
    *,
    n_radial: int = 60,
    n_axial: int = 40,
    r_far_factor: float = 30.0,
    z_far_factor: float = 20.0,
) -> FieldResult:
    """Axisymmetric Finite-Element solver for grounding systems.

    Reduces every cluster to its **equivalent hemisphere** at the
    cluster centroid, then solves the volume PDE on a 2-D
    axisymmetric triangular mesh. This is the only volume-PDE engine
    in the suite and forms the third independent cross-check (next to
    the closed-form ``image_*`` family and the integral ``mom``/``bem``
    family).

    Parameters
    ----------
    world
        World to evaluate. Must currently contain a single galvanic
        cluster.
    engine
        Engine configuration.
    n_radial, n_axial
        Mesh resolution.
    r_far_factor, z_far_factor
        Far-field truncation (multiples of the characteristic length
        $a_{\\text{eq}} + \\sum h_i$).

    Returns
    -------
    FieldResult
        ``metadata['equivalent_hemisphere_radius']`` reports the
        reduction used on the cluster.
    """
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'fem' supports HomogeneousSoil, TwoLayerSoil, "
            f"and MultiLayerSoil. Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")

    stack = as_layer_stack(world.soil)
    rho_1 = float(stack.rhos[0])
    # The mesh follows the soil description (one z-line per layer
    # boundary). We do *not* collapse equal-ρ stacks down to a
    # 1-layer mesh — keeping the topology consistent across a ρ₂
    # sweep is a stronger guarantee than reproducing the homogeneous
    # discretisation bit-exactly. The (small) discretisation bias at
    # ρ₂ = ρ₁ is documented and bounded by the FEM K_zero test
    # tolerance.
    h_layers = stack.h.tolist()

    # 1) Cluster the electrodes; build per-cluster equivalent hemispheres.
    cluster_id = _build_clusters(world.electrodes, world.conductors)
    clusters: dict[str, list[str]] = {}
    for ename, root in cluster_id.items():
        clusters.setdefault(root, []).append(ename)
    # FEM cannot consume the distributed-conductor topology because
    # the equivalent-hemisphere reduction is not defined for the tiny
    # midpoint pseudo-electrodes a distributed conductor would
    # produce. Fall back to lumped branches and warn the user — for
    # quantitative distributed-conductor work pick one of the
    # integral-equation backends (image, mom, cim, bem).
    has_distributed = any(
        getattr(c, "is_distributed", False) for c in world.conductors
    )
    if has_distributed:
        _log.warning(
            "fem: distributed conductors detected — the FEM backend "
            "treats every conductor as lumped (single branch with the "
            "full series resistance). Use 'image', 'image_2layer', "
            "'mom', 'cim' or 'bem' for distributed-conductor results."
        )
    has_inductance = any(
        getattr(c, "inductance_model", None) is not None
        for c in world.conductors
    )
    if has_inductance:
        _log.warning(
            "fem: inductance_model is not supported by the FEM backend "
            "(equivalent-hemisphere reduction is DC only). The "
            "computation falls back to the resistive solution; switch "
            "to image / mom / cim / bem for inductive coupling."
        )
    if (
        getattr(engine, "earth_inductive_model", "perfect_mirror")
        == "carson_series"
    ):
        _log.warning(
            "fem: earth_inductive_model='carson_series' is ignored — "
            "the FEM backend has no inductive branch model. "
            "Switch to image / image_2layer / mom / cim / bem for "
            "Carson-corrected results (ADR-0005)."
        )
    finite_branches = _build_finite_branches(
        world.conductors, cluster_id,
        distributed_as_lumped=True,
    )

    # Per-cluster active current.
    elec_input_current: dict[str, complex] = {
        e.name: 0j for e in world.electrodes
    }
    for src in world.sources:
        if src.kind != "current":
            continue
        i_complex = src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg))
        if src.attached_to in elec_input_current:
            elec_input_current[src.attached_to] += i_complex
    cluster_current: dict[str, complex] = {root: 0j for root in clusters}
    for ename, ic in elec_input_current.items():
        cluster_current[cluster_id[ename]] += ic

    # Active set: every cluster with a non-zero source plus every
    # cluster transitively reachable through a finite branch.
    active_set: set[str] = {r for r, ic in cluster_current.items() if ic != 0j}
    if finite_branches:
        # Seed-and-propagate: any cluster connected to an active one
        # via a chain of finite branches is itself active (current
        # flows through the branches).
        changed = True
        while changed:
            changed = False
            for a, b, _R in finite_branches:
                if a in active_set and b not in active_set:
                    active_set.add(b)
                    changed = True
                elif b in active_set and a not in active_set:
                    active_set.add(a)
                    changed = True
    active_clusters = [r for r in clusters if r in active_set]

    n_freq = len(engine.frequencies)
    electrode_potentials: dict[str, list[complex]] = {
        e.name: [0j] * n_freq for e in world.electrodes
    }
    electrode_currents: dict[str, list[complex]] = {
        e.name: [0j] * n_freq for e in world.electrodes
    }

    a_eq_per_cluster: dict[str, float] = {}
    Z_per_cluster: dict[str, float] = {}

    for root in active_clusters:
        # Build a single equivalent hemisphere from the parallel
        # combination of the per-electrode hemispheres. The
        # hemisphere DC resistance is R = ρ / (2 π a), so two
        # hemispheres in parallel give 1/R_par = 2π(a_1 + a_2)/ρ
        # — i.e. the *radii* add. (Inverting them, as a parallel
        # resistor formula would suggest, is wrong here because the
        # geometric factor sits in the numerator of the conductance.)
        a_per_electrode = [
            equivalent_hemisphere_radius(world.get_electrode(ename), rho_top=rho_1)
            for ename in clusters[root]
        ]
        a_eq = float(sum(a_per_electrode))
        a_eq_per_cluster[root] = a_eq

        # 2) Build the axisymmetric mesh.
        nodes, triangles, layer_idx = _build_axisymmetric_mesh(
            a_eq, h_layers,
            r_far_factor=r_far_factor,
            z_far_factor=z_far_factor,
            n_radial=n_radial,
            n_axial=n_axial,
        )
        sigma_per_layer = 1.0 / stack.rhos
        sigma_per_triangle = sigma_per_layer[layer_idx]

        K = _assemble_stiffness_axisymmetric(nodes, triangles, sigma_per_triangle)

        # 3) Boundary conditions.
        # Dirichlet inner boundary: φ = 1 on every node with s² + z² ≤ a_eq²
        # AND z ≥ 0. Plus φ = 0 on the outer boundary.
        s = nodes[:, 0]
        z = nodes[:, 1]
        inner = (s * s + z * z) <= (a_eq + 1e-9) ** 2
        # Always include the on-axis surface point (s=0, z=0).
        inner |= (s < 1e-9) & (z < 1e-9)
        outer = (s >= s.max() - 1e-9) | (z >= z.max() - 1e-9)
        dirichlet = inner | outer

        rhs = np.zeros(nodes.shape[0])
        rhs[inner] = 1.0  # unit potential on the hemisphere boundary

        # Eliminate Dirichlet rows.
        free = ~dirichlet
        K_ff = K[free][:, free]
        K_fd = K[free][:, dirichlet]
        b = -K_fd @ rhs[dirichlet]
        phi = np.zeros(nodes.shape[0])
        phi[free] = spsolve(K_ff, b)
        phi[dirichlet] = rhs[dirichlet]

        # 4) Resistance via energy: 1/R = 2π ∫_Ω σ |∇φ|² s ds dz.
        #    With unit boundary potential this gives 1/R directly.
        # We integrate per-triangle.
        inv_R = 0.0
        for t_idx, t in enumerate(triangles):
            v = nodes[t]
            x_v = v[:, 0]
            y_v = v[:, 1]
            area = 0.5 * abs(
                (x_v[1] - x_v[0]) * (y_v[2] - y_v[0])
                - (x_v[2] - x_v[0]) * (y_v[1] - y_v[0])
            )
            if area <= 0.0:
                continue
            s_bar = float(np.mean(x_v))
            b_v = np.array([y_v[1] - y_v[2], y_v[2] - y_v[0], y_v[0] - y_v[1]]) / (
                2.0 * area
            )
            c_v = np.array([x_v[2] - x_v[1], x_v[0] - x_v[2], x_v[1] - x_v[0]]) / (
                2.0 * area
            )
            phi_t = phi[t]
            grad_s = float(b_v @ phi_t)
            grad_z = float(c_v @ phi_t)
            inv_R += (
                sigma_per_triangle[t_idx]
                * (grad_s * grad_s + grad_z * grad_z)
                * area
                * 2.0
                * np.pi
                * s_bar
            )
        R_cluster = 1.0 / inv_R if inv_R > 0.0 else float("inf")
        Z_per_cluster[root] = R_cluster
        # Hemisphere-radius distribution per electrode is reused below
        # in step 5 once the leakage current per cluster is known.

    # ------------------------------------------------------------------
    # 5) Nodal analysis on the cluster level.
    #    With per-cluster self-resistance R_c (FEM) plus optional
    #    finite-impedance branches between clusters, solve
    #
    #        diag(1/R_c) · phi_n   +   B^T · I_b   =   I_in
    #        B · phi_n - R_b · I_b                  =   0
    #
    #    For finite_branches == [] this collapses to
    #        phi_n = R_c · I_in
    #    — i.e. the historic single-cluster behaviour.
    # ------------------------------------------------------------------
    cluster_idx_map = {root: k for k, root in enumerate(active_clusters)}
    K_a = len(active_clusters)
    active_branches = [
        (a, b, R) for (a, b, R) in finite_branches
        if a in cluster_idx_map and b in cluster_idx_map
    ]
    M_a = len(active_branches)
    n_unk = K_a + M_a
    A_mat = np.zeros((n_unk, n_unk))
    rhs_re = np.zeros(n_unk)
    rhs_im = np.zeros(n_unk)
    for k, root in enumerate(active_clusters):
        R_c = Z_per_cluster[root]
        A_mat[k, k] = 1.0 / R_c if np.isfinite(R_c) and R_c > 0.0 else 0.0
        ic = cluster_current[root]
        rhs_re[k] = ic.real
        rhs_im[k] = ic.imag
    for m, (a, b, R) in enumerate(active_branches):
        ka = cluster_idx_map[a]
        kb = cluster_idx_map[b]
        # KCL contributions of branch m at nodes a and b
        A_mat[ka, K_a + m] = +1.0
        A_mat[kb, K_a + m] = -1.0
        # Branch Ohm's law:  phi_a - phi_b = R · I_b
        # ⇔  +phi_a - phi_b - R · I_b = 0
        A_mat[K_a + m, ka] = +1.0
        A_mat[K_a + m, kb] = -1.0
        A_mat[K_a + m, K_a + m] = -R
    if n_unk > 0:
        sol_re = np.linalg.solve(A_mat, rhs_re)
        sol_im = np.linalg.solve(A_mat, rhs_im)
    else:
        sol_re = np.zeros(0)
        sol_im = np.zeros(0)
    phi_node = sol_re[:K_a] + 1j * sol_im[:K_a]
    branch_current = sol_re[K_a:] + 1j * sol_im[K_a:]

    # 6) Per-electrode currents and potentials.
    #    Within a cluster the leakage current is split proportionally
    #    to each member's hemisphere conductance (G ∝ a), exactly as
    #    in the historic single-cluster code path. The leakage of a
    #    cluster is its KCL balance: I_leak = I_in - Σ I_branch_out.
    for k, root in enumerate(active_clusters):
        I_in_c = cluster_current[root]
        I_branch_out = 0j
        for m, (a, b, _R) in enumerate(active_branches):
            if a == root:
                I_branch_out += branch_current[m]
            elif b == root:
                I_branch_out -= branch_current[m]
        I_leak = I_in_c - I_branch_out
        u_cluster = complex(phi_node[k])
        a_per_electrode_c = [
            equivalent_hemisphere_radius(world.get_electrode(ename), rho_top=rho_1)
            for ename in clusters[root]
        ]
        a_total = sum(a_per_electrode_c)
        for ename, a_e in zip(clusters[root], a_per_electrode_c):
            share = a_e / a_total if a_total > 0.0 else 0.0
            electrode_currents[ename] = [I_leak * share] * n_freq
            electrode_potentials[ename] = [u_cluster] * n_freq

    # Inactive clusters: zero current, zero potential.
    for root, members in clusters.items():
        if root in active_clusters:
            continue
        for ename in members:
            electrode_currents[ename] = [0j] * n_freq
            electrode_potentials[ename] = [0j] * n_freq

    point_sources: list[PointSource] = []
    cluster_members_map: dict[str, list[str]] = {
        ename: sorted(clusters[root])
        for root, members in clusters.items()
        for ename in members
    }

    return FieldResult(
        backend="fem",
        frequencies=list(engine.frequencies),
        electrode_potentials=electrode_potentials,
        electrode_currents=electrode_currents,
        point_sources=point_sources,
        soil_resistivity=rho_1,
        soil=world.soil,
        clusters=cluster_members_map,
        metadata={
            "world_name": world.name,
            "n_layers": int(stack.n_layers),
            "rhos": stack.rhos.tolist(),
            "h": h_layers,
            "n_radial": int(n_radial),
            "n_axial": int(n_axial),
            "equivalent_hemisphere_radius": {
                k: float(v) for k, v in a_eq_per_cluster.items()
            },
            "fem_cluster_resistance": {
                k: float(v) for k, v in Z_per_cluster.items()
            },
            "stub": False,
            "approximation": "equivalent-hemisphere reduction per cluster",
        },
    )
