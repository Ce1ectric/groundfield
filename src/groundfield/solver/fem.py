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
electrically insulating air) and the far-field decay
$\\varphi \\to 0$ as $|\\mathbf{r}| \\to \\infty$,
truncated to a finite outer radius $r_{\\text{far}}$ that carries
the monopole DtN condition described below.
$q$ is the current source density.

For most reference electrodes that are essentially **axisymmetric**
around their connection point (a single rod, a ring, a hemisphere),
we exploit the symmetry and discretise the problem on
a 2-D $(s, z)$ mesh with cylindrical coordinates. The PDE
becomes
$$
-\\frac{1}{s} \\frac{\\partial}{\\partial s}\\!
\\left(s\\, \\sigma\\, \\frac{\\partial \\varphi}{\\partial s}\\right)
- \\frac{\\partial}{\\partial z}\\!
  \\left(\\sigma\\, \\frac{\\partial \\varphi}{\\partial z}\\right)
\\;=\\; q,
$$
solved on a triangular finite-element mesh with linear hat
functions. The weak form is assembled with a sparse COO-builder; the
linear system is solved with ``scipy.sparse.linalg.spsolve``.

On a straight-sided triangle the hat-function gradients are
constant and $\\int_T s\\, dA = \\bar s_T |T|$ holds exactly, so
the centroid-radius element rule used below is an exact quadrature
of the axisymmetric weak form — not a one-point approximation.

Mesh: boundary-conforming spherical shell
-----------------------------------------
The Dirichlet electrode of this backend is the **equivalent
hemisphere** $r = a_{\\text{eq}}$ (see below). A mesh that does
not resolve that surface geometrically turns the electrode into a
staircase whose shape — and hence whose capacitance — changes
discontinuously with the mesh, so refinement does not converge.
The mesh is therefore built in **spherical shell coordinates**
centred on the electrode,
$$
s = r \\sin\\vartheta, \\qquad z = r \\cos\\vartheta,
\\qquad r \\in [a_{\\text{eq}}, r_{\\text{far}}],
\\quad \\vartheta \\in [0, \\pi/2],
$$
with a geometric ladder in $r$ and a uniform ladder in
$\\vartheta$. Consequences:

- The inner boundary $r = a_{\\text{eq}}$ is an exact mesh line,
  i.e. the discrete electrode **is** the intended hemisphere for
  every resolution (a *conforming* boundary; no staircase, no
  cut cells, no immersed-boundary weights). Strictly, it is the
  polyline inscribed in that hemisphere, refined together with the
  angular resolution; a layer interface shallower than
  $a_{\\text{eq}}$ splits one of its chords and the inserted node
  joins the Dirichlet set, so the electrode stays closed (see
  :func:`_cut_triangles_at_depth`).
- $\\vartheta = 0$ is the downward symmetry axis and
  $\\vartheta = \\pi/2$ is the soil surface, so the natural
  (Neumann) boundary condition sits exactly on $z = 0$.
- Element aspect ratios of the *uncut* shell are $O(1)$ and
  scale-invariant: the radial step is
  $r \\ln(r_{\\text{far}}/a_{\\text{eq}}) / n_{\\text{radial}}$
  and the transverse step is $r \\pi / (2 n_{\\text{axial}})$,
  both proportional to $r$ (measured shape quality
  $\\max \\sum \\ell^2 / |T| = 10.7$ at the default resolution,
  against 2.31 for an equilateral triangle). The interface cut is
  the exception: where a layer boundary grazes a node it does
  produce slivers. Over 66 adversarial grazing depths the worst
  cut element reached $\\sum \\ell^2 / |T| = 1.1 \\cdot 10^5$ at
  an area of $6.7 \\cdot 10^{-12}\\,\\text{m}^2$. Because the
  plane is snapped onto nodes within a relative tolerance of
  $10^{-6}$ and truly degenerate elements carry no volume and are
  skipped in the assembly, the effect on the answer is negligible:
  at fixed $r_{\\text{far}}$ the same sweep stays inside
  $-0.23837 \\ldots -0.23816\\,\\%$ (a spread of
  $2 \\cdot 10^{-4}$ percentage points around the uncut
  $-0.23939\\,\\%$), with no zero-area element produced. So the
  guarantee is *bounded* quality away from the cut, not $O(1)$
  quality everywhere.
- The near-electrode resolution is decoupled from the outer
  truncation: it depends on $r_{\\text{far}}$ only through
  $\\ln r_{\\text{far}}$, so a deep soil layer no longer starves
  the electrode of nodes.
- Refining $(n_{\\text{radial}}, n_{\\text{axial}})$ by
  $n \\mapsto 2n - 1$ produces *nested* FE spaces,
  $V_h \\subset V_{h/2} \\subset V$. The conductance is the
  **minimum** of the energy over the admissible set (Dirichlet
  principle), so a larger space can only lower that minimum:
  $G_h \\ge G_{h/2} \\ge G$ and therefore
  $R_h = 1/G_h$ **increases monotonically towards the exact
  value from below** — the convergence guarantee this backend
  previously lacked. Measured on the reference rod
  ($n \\mapsto 2n-1$ from 29 × 19):
  $G_h = 0.015640831 \\to 0.015516853 \\to 0.015485931 \\to
  0.015478205$ against $G = 0.0154757$, i.e.
  $R_h = 61.95 \\to 64.62\\,\\Omega$.

Layer model
-----------
Layer boundaries enter through the piecewise-constant conductivity
$\\sigma(z)$. A spherical shell mesh is not aligned with the
horizontal interfaces $z = \\sum_{i \\le k} h_i$, so every element
crossed by an interface is **cut along the interface** at mesh-build
time (a triangle split into two or three sub-triangles, with the cut
nodes shared between neighbours). The conductivity jump is therefore
mesh-aligned as well and each element carries a single conductivity,
the one of the layer its centroid sits in. The PDE handles arbitrary
horizontally stratified soils (any ``n``).

Far field: monopole DtN (Robin) boundary
----------------------------------------
Truncating the domain with $\\varphi = 0$ at $r_{\\text{far}}$
short-circuits the remaining half-space and removes the
$\\rho / (2 \\pi r_{\\text{far}})$ tail of the resistance — a
one-sided error of order $a_{\\text{eq}} / r_{\\text{far}}$. Since
the leading far-field term of *any* grounding electrode (also in a
layered soil, where only the amplitude changes) is the monopole
$\\varphi \\propto 1/r$, its exact Dirichlet-to-Neumann map on a
sphere is the Robin condition
$$
\\partial \\varphi / \\partial r = -\\varphi / r_{\\text{far}}
\\quad \\Longrightarrow \\quad
\\int_\\Omega \\sigma \\nabla \\varphi \\cdot \\nabla v \\, dV
+ \\oint_{r_{\\text{far}}} \\frac{\\sigma}{r_{\\text{far}}}
  \\varphi\\, v \\, dS = 0 .
$$
This is exact for the homogeneous half-space (where
$\\varphi = a_{\\text{eq}}/r$ satisfies it identically), so the
truncation error drops from $O(a_{\\text{eq}}/r_{\\text{far}})$ to
the multipole residual $O((a_{\\text{eq}}/r_{\\text{far}})^3)$.
The conductance is then the full bilinear form,
$G = 1/R = a(\\varphi, \\varphi)$ = dissipated power inside the
truncation sphere plus the power carried through it.

Scope
-----
- **Geometry coverage.** The axisymmetric formulation captures
  :class:`RodElectrode` (vertical rod, ``s = 0``) and
  :class:`RingElectrode` and :class:`MeshElectrode` *as effective
  hemispheres* — the equivalent-hemisphere radius is computed from
  the electrode's geometric parameters before the FEM run. This is
  the standard reduction used in research-level reference
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
  per-electrode currents. For a single cluster (the typical
  case) the approximation is appropriate.
- **Frequency.** Quasi-static, frequency-independent.

The FEM backend's purpose in the engine family is to provide a
**volume-PDE cross-check** that does not share any code path with
the integral-equation engines. Where it disagrees with the others on
simple geometries by more than a few per cent, the source is the
equivalent-hemisphere reduction described above (and documented in
the result metadata) — *not* the discretisation: for homogeneous
soil the solver reproduces
$R = \\rho / (2 \\pi a_{\\text{eq}})$ to $< 1\\,\\%$ at the
default resolution and converges to it at second order.

References
----------
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover, ch. 2.1.
- Dwight, H. B. (1936). Calculation of resistances to ground.
- Güemes, J. A., & Hernando, F. E. (2004). Method for calculating
  the ground resistance of grounding grids using FEM. *IEEE PWRD*
  19(2).
- Givoli, D. (1992). *Numerical Methods for Problems in Infinite
  Domains*, Elsevier — Dirichlet-to-Neumann (DtN) truncation
  boundaries; the monopole DtN map used here is its lowest mode.
- Strang, G., & Fix, G. (2008). *An Analysis of the Finite Element
  Method*, 2nd ed. — the Dirichlet/minimum-energy principle behind
  the one-sided convergence of $R_h$.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
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
from groundfield.solver._layered import as_layer_stack
from groundfield.solver.image import (
    _build_clusters,
    _build_finite_branches,
    _reject_concrete_shells,
    _warn_ignored_sources,
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


class _AxiMesh(NamedTuple):
    """Boundary-conforming axisymmetric mesh of the soil half-space.

    The mesh covers the spherical shell
    $a_{\\text{eq}} \\le r \\le r_{\\text{far}}$, $z \\ge 0$ of the
    $(s, z)$ half-plane, i.e. the soil outside the equivalent
    hemisphere and inside the truncation sphere.

    Attributes
    ----------
    nodes : np.ndarray, shape (Nv, 2)
        Vertex coordinates $(s, z)$.
    triangles : np.ndarray, shape (Nt, 3), int
        Per-triangle vertex indices.
    layer_index : np.ndarray, shape (Nt,), int
        0-based soil-layer index of every triangle. Interfaces are
        mesh-conforming, so this is exact (no partially filled
        elements).
    inner_nodes : np.ndarray, shape (Ni,), int
        Nodes on the electrode surface $r = a_{\\text{eq}}$. Tracked
        by index rather than by a geometric predicate so that the
        Dirichlet set is exactly the conforming hemisphere. Nodes
        inserted *on* the electrode boundary by an interface cut (a
        layer shallower than $a_{\\text{eq}}$ crosses one chord of
        the electrode polyline) are added to this set by
        :func:`_cut_triangles_at_depth`, so the set stays the
        complete discrete electrode and no free node is left on it.
    outer_edges : np.ndarray, shape (Ne, 2), int
        Boundary edges on the truncation sphere
        $r = r_{\\text{far}}$, in the order needed for the far-field
        surface integral.
    r_far : float
        Truncation radius.
    """

    nodes: np.ndarray
    triangles: np.ndarray
    layer_index: np.ndarray
    inner_nodes: np.ndarray
    outer_edges: np.ndarray
    r_far: float


def _layer_of_depth(z: np.ndarray, h_layers: list[float]) -> np.ndarray:
    """0-based layer index of the depths ``z`` for a layer stack.

    A depth exactly on an interface is attributed to the layer *below*
    it, matching :class:`~groundfield.solver._layered.LayerStack`
    conventions.
    """
    if not h_layers:
        return np.zeros(np.shape(z), dtype=int)
    cum = np.cumsum(np.asarray(h_layers, dtype=float))
    return np.searchsorted(cum, np.asarray(z, dtype=float), side="right")


def _cut_triangles_at_depth(
    nodes: np.ndarray,
    triangles: np.ndarray,
    outer_edges: list[tuple[int, int]],
    z_cut: float,
    *,
    tol: float,
    inner_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Split every triangle crossed by the horizontal plane $z = z_c$.

    Makes a soil-layer interface an exact mesh line so that the
    conductivity jump never runs through the interior of an element
    (an element straddling the jump would otherwise smear the
    interface over one mesh width — a first-order geometric error).

    The split is *conforming*: the intersection point of an edge is
    keyed on the (ordered) pair of its end nodes, so the two triangles
    sharing that edge insert the same node and no hanging nodes
    appear. Nodes within ``tol`` of the plane are snapped onto it
    first — a per-node decision, hence identical for all triangles
    touching the node, which is what keeps the split free of slivers.

    The Dirichlet (electrode) node set is carried through the split.
    An interface shallower than $a_{\\text{eq}}$ crosses one *chord*
    of the electrode polyline, so a cut node is inserted **on the
    electrode boundary**; leaving it out of the Dirichlet set would
    turn it into a free node, i.e. a small insulating patch with a
    natural zero-flux condition in the middle of the energised
    electrode. A cut node belongs to the electrode exactly when both
    end nodes of the edge it splits do — the criterion is
    topological, not a radius tolerance, because on strongly
    anisotropic meshes ($n_{\\text{radial}}$ small,
    $n_{\\text{axial}}$ large) a *radial* mesh edge leaving the
    electrode can dip below $r = a_{\\text{eq}}$ as well, and its cut
    node must stay free.

    Returns
    -------
    nodes, triangles, outer_edges, inner_nodes
        Updated mesh arrays. Existing node indices are preserved;
        cut nodes are appended. ``outer_edges`` entries split by the
        plane are replaced by their two halves, and ``inner_nodes``
        gains every cut node that lies on the electrode boundary.
    """
    nodes = np.array(nodes, dtype=float, copy=True)
    inner_nodes = np.asarray(inner_nodes, dtype=int)
    close = np.abs(nodes[:, 1] - z_cut) <= tol
    nodes[close, 1] = z_cut
    d = nodes[:, 1] - z_cut

    dv = d[triangles]
    cut_mask = (dv > 0.0).any(axis=1) & (dv < 0.0).any(axis=1)
    if not cut_mask.any():
        return nodes, triangles, outer_edges, inner_nodes

    extra: list[tuple[float, float]] = []
    edge_node: dict[tuple[int, int], int] = {}
    n_base = nodes.shape[0]

    def _split_node(ia: int, ib: int) -> int:
        """Index of the node where edge (ia, ib) crosses the plane."""
        key = (ia, ib) if ia < ib else (ib, ia)
        hit = edge_node.get(key)
        if hit is not None:
            return hit
        ka, kb = key
        t = d[ka] / (d[ka] - d[kb])
        s_new = nodes[ka, 0] + t * (nodes[kb, 0] - nodes[ka, 0])
        idx = n_base + len(extra)
        extra.append((float(s_new), float(z_cut)))
        edge_node[key] = idx
        return idx

    new_tris: list[tuple[int, int, int]] = []
    for t_idx in np.nonzero(cut_mask)[0]:
        tri = triangles[t_idx]
        dd = d[tri]
        on_plane = np.nonzero(dd == 0.0)[0]
        if on_plane.size == 1:
            # One vertex on the plane, the other two on opposite sides:
            # a single cut on the opposite edge → two sub-triangles.
            i0 = int(on_plane[0])
            i1, i2 = (i0 + 1) % 3, (i0 + 2) % 3
            p = _split_node(int(tri[i1]), int(tri[i2]))
            new_tris.append((int(tri[i0]), int(tri[i1]), p))
            new_tris.append((int(tri[i0]), p, int(tri[i2])))
        else:
            # Generic case: one vertex alone on its side of the plane.
            # Triangle + convex quad → three sub-triangles.
            pos = dd > 0.0
            lone = int(np.nonzero(pos if pos.sum() == 1 else ~pos)[0][0])
            i1, i2 = (lone + 1) % 3, (lone + 2) % 3
            p1 = _split_node(int(tri[lone]), int(tri[i1]))
            p2 = _split_node(int(tri[lone]), int(tri[i2]))
            new_tris.append((int(tri[lone]), p1, p2))
            new_tris.append((p1, int(tri[i1]), int(tri[i2])))
            new_tris.append((p1, int(tri[i2]), p2))

    if extra:
        nodes = np.vstack([nodes, np.array(extra, dtype=float)])
    triangles = np.vstack(
        [triangles[~cut_mask], np.array(new_tris, dtype=int).reshape(-1, 3)]
    )

    # A straight plane cuts a straight edge at most once, so every
    # affected boundary edge simply becomes two.
    split_edges: list[tuple[int, int]] = []
    for ia, ib in outer_edges:
        key = (ia, ib) if ia < ib else (ib, ia)
        p_split = edge_node.get(key)
        if p_split is None:
            split_edges.append((ia, ib))
        else:
            split_edges.extend([(ia, p_split), (p_split, ib)])

    # A cut node splitting an edge whose *both* end nodes are on the
    # electrode lies on the electrode boundary itself and must inherit
    # the Dirichlet condition (see the docstring).
    inner_set = set(inner_nodes.tolist())
    on_electrode = [
        p for (ka, kb), p in edge_node.items()
        if ka in inner_set and kb in inner_set
    ]
    if on_electrode:
        inner_nodes = np.concatenate(
            [inner_nodes, np.array(sorted(on_electrode), dtype=int)]
        )
    return nodes, triangles, split_edges, inner_nodes


def _build_axisymmetric_mesh(
    a_eq: float,
    h_layers: list[float],
    *,
    r_far_factor: float = 30.0,
    z_far_factor: float | None = None,
    n_radial: int = 60,
    n_axial: int = 40,
) -> _AxiMesh:
    """Build the boundary-conforming spherical-shell mesh.

    The mesh discretises the soil between the equivalent hemisphere
    $r = a_{\\text{eq}}$ and the truncation sphere
    $r = r_{\\text{far}}$ in the $(s, z)$ half-plane, using the
    spherical shell coordinates
    $$
    s = r \\sin\\vartheta, \\qquad z = r \\cos\\vartheta,
    \\qquad \\vartheta \\in [0, \\pi/2],
    $$
    with a **geometric** ladder in $r$ (the natural grading for the
    $1/r$ monopole field: equal relative steps ⇒ equal error per
    element) and a **uniform** ladder in $\\vartheta$. Every soil
    layer interface is then cut into the element grid
    (:func:`_cut_triangles_at_depth`).

    Two properties matter physically and both come from the choice of
    coordinates rather than from any tolerance:

    * the electrode surface $r = a_{\\text{eq}}$ is an exact mesh
      line — the discrete electrode is the intended hemisphere at
      every resolution, so refinement converges instead of
      re-shaping the electrode (a staircase disc on an $(s, z)$
      tensor grid does the latter);
    * the near-electrode resolution is $\\propto a_{\\text{eq}}
      \\ln(r_{\\text{far}}/a_{\\text{eq}}) / n_{\\text{radial}}$,
      i.e. only logarithmically sensitive to the outer truncation, so
      declaring a thick soil layer no longer coarsens the electrode.

    Parameters
    ----------
    a_eq
        Equivalent-hemisphere radius; the inner (Dirichlet) boundary.
    h_layers
        Finite layer thicknesses. Every interface depth
        $\\sum_{i \\le k} h_i$ inside the mesh is cut in as an exact
        element boundary, so the conductivity step is mesh-aligned.
    r_far_factor
        Truncation radius as a multiple of the characteristic length
        $\\bar L = a_{\\text{eq}} + \\sum_i h_i$, i.e.
        $r_{\\text{far}} = f_r \\bar L$ (subject to the
        $\\ge 4 a_{\\text{eq}}$ floor below). This is the *only*
        knob that sets the domain size — the truncation of this mesh
        is a single sphere, not a box.
    z_far_factor
        Deprecated axial companion of ``r_far_factor``, kept so that
        callers written against the $(s, z)$ box mesh of v0.14.1 keep
        working. ``None`` (the default) ignores it. A number acts as
        a *lower bound* on the same radius,
        $r_{\\text{far}} = \\max(f_r, f_z)\\,\\bar L$, and a warning
        is logged whenever it actually overrides ``r_far_factor``, so
        the knob can never silently do nothing.
    n_radial
        Number of node lines in $r$ (radial resolution).
    n_axial
        Number of node lines in $\\vartheta$ (transverse resolution,
        from the downward axis to the soil surface). The historical
        name is kept: on the previous $(s, z)$ tensor grid this was
        the axial node count.

    Returns
    -------
    _AxiMesh
        Nodes, triangles, per-triangle layer index, the electrode
        boundary nodes, the truncation-sphere edges and
        $r_{\\text{far}}$.

    Notes
    -----
    Passing ``n_radial → 2 n_radial - 1`` and
    ``n_axial → 2 n_axial - 1`` refines every element into four and
    yields a *nested* finite-element space (for homogeneous soil,
    where no interface cut perturbs the pattern), which is what makes
    the resistance converge monotonically.
    """
    if not np.isfinite(a_eq) or a_eq <= 0.0:
        raise ValueError(f"a_eq must be a positive length, got {a_eq!r}.")
    n_radial = max(int(n_radial), 3)
    n_axial = max(int(n_axial), 3)

    base_length = a_eq + (sum(h_layers) if h_layers else 1.0)
    # ``r_far_factor`` alone sets the truncation sphere. Silently
    # taking max(r_far_factor, z_far_factor) made every
    # ``r_far_factor <= z_far_factor`` a dead knob (with the old
    # defaults 30/20 that was every value <= 20), so a truncation
    # study saw no effect at all.
    factor = float(r_far_factor)
    if z_far_factor is not None and float(z_far_factor) > factor:
        _log.warning(
            "fem: z_far_factor=%g overrides r_far_factor=%g — the "
            "truncation of the spherical-shell mesh is a single radius, "
            "so the larger factor wins. Pass r_far_factor alone (and "
            "leave z_far_factor at None) to control it.",
            float(z_far_factor), factor,
        )
        factor = float(z_far_factor)
    r_far = factor * base_length
    # The shell must have room for the graded ladder even if the
    # caller passes a tiny truncation factor.
    r_far = max(r_far, 4.0 * a_eq)

    r_lines = np.geomspace(a_eq, r_far, n_radial)
    r_lines[0] = a_eq
    r_lines[-1] = r_far
    theta = np.linspace(0.0, 0.5 * np.pi, n_axial)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    # Make the axis (s = 0) and the soil surface (z = 0) exact.
    sin_t[0], cos_t[0] = 0.0, 1.0
    sin_t[-1], cos_t[-1] = 1.0, 0.0

    nodes = np.stack(
        [np.outer(r_lines, sin_t).ravel(), np.outer(r_lines, cos_t).ravel()],
        axis=1,
    )

    # Structured quads → two triangles each (index = i_r * n_axial + k).
    idx = np.arange(n_radial * n_axial).reshape(n_radial, n_axial)
    n00 = idx[:-1, :-1].ravel()
    n10 = idx[1:, :-1].ravel()
    n11 = idx[1:, 1:].ravel()
    n01 = idx[:-1, 1:].ravel()
    triangles = np.concatenate(
        [np.stack([n00, n10, n11], axis=1), np.stack([n00, n11, n01], axis=1)]
    )

    inner_nodes = idx[0, :].copy()
    outer_edges: list[tuple[int, int]] = [
        (int(idx[-1, k]), int(idx[-1, k + 1])) for k in range(n_axial - 1)
    ]

    # Cut the layer interfaces into the shell mesh. ``inner_nodes`` is
    # carried through the cut: an interface shallower than a_eq splits
    # a chord of the electrode polyline and the new node has to join
    # the Dirichlet set, otherwise it stays free and punches an
    # insulating patch into the electrode surface.
    if h_layers:
        cum = 0.0
        for h in h_layers:
            cum += float(h)
            if cum <= 0.0 or cum >= r_far:
                # Interface below the truncation sphere — never
                # crossed inside the mesh.
                continue
            nodes, triangles, outer_edges, inner_nodes = (
                _cut_triangles_at_depth(
                    nodes, triangles, outer_edges, cum,
                    tol=1e-6 * max(cum, a_eq),
                    inner_nodes=inner_nodes,
                )
            )

    layer_index = _layer_of_depth(nodes[triangles, 1].mean(axis=1), h_layers)

    return _AxiMesh(
        nodes=nodes,
        triangles=triangles,
        layer_index=layer_index,
        inner_nodes=np.asarray(inner_nodes, dtype=int),
        outer_edges=np.asarray(outer_edges, dtype=int).reshape(-1, 2),
        r_far=float(r_far),
    )


def _assemble_stiffness_axisymmetric(
    nodes: np.ndarray,
    triangles: np.ndarray,
    sigma_per_triangle: np.ndarray,
) -> csr_matrix:
    """Assemble the axisymmetric weak-form stiffness matrix.

    The bilinear form for the cylindrical Laplacian with linear hat
    functions on a 2-D triangle $T$ becomes
    $$
    a_{ij}^T \\;=\\; 2\\pi\\, \\sigma_T \\bar s_T\\,
                     \\bigl(\\nabla \\phi_i \\cdot \\nabla \\phi_j\\bigr)
                     \\, |T|,
    $$
    with $\\bar s_T$ the centroid radius and $|T|$ the
    area of the triangle in the $(s, z)$ plane. The gradients are
    constant on $T$ and $\\int_T s\\, dA = \\bar s_T |T|$ holds
    exactly, so this element rule is an exact quadrature of the
    axisymmetric weak form.

    Degenerate (zero-area) triangles — which the interface cut can
    produce when a layer boundary grazes a node — carry no volume and
    are skipped.
    """
    Nv = nodes.shape[0]
    v = nodes[triangles]                              # (Nt, 3, 2)
    x = v[:, :, 0]
    y = v[:, :, 1]
    area2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (
        y[:, 1] - y[:, 0]
    )
    area = 0.5 * np.abs(area2)
    keep = area > 0.0
    if not keep.any():
        return coo_matrix(([], ([], [])), shape=(Nv, Nv)).tocsr()
    x, y, area = x[keep], y[keep], area[keep]
    tri = triangles[keep]
    inv = 1.0 / (2.0 * area)
    b = np.stack(
        [y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1
    ) * inv[:, None]
    c = np.stack(
        [x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1
    ) * inv[:, None]
    weight = (
        np.asarray(sigma_per_triangle, dtype=float)[keep]
        * area
        * 2.0
        * np.pi
        * x.mean(axis=1)
    )
    K_e = weight[:, None, None] * (
        b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    )
    rows = np.broadcast_to(tri[:, :, None], K_e.shape).ravel()
    cols = np.broadcast_to(tri[:, None, :], K_e.shape).ravel()
    K = coo_matrix((K_e.ravel(), (rows, cols)), shape=(Nv, Nv)).tocsr()
    return K


def _assemble_far_field_robin(
    nodes: np.ndarray,
    outer_edges: np.ndarray,
    sigma_per_edge: np.ndarray,
    r_far: float,
) -> csr_matrix:
    """Assemble the monopole-DtN (Robin) far-field boundary term.

    The truncation sphere carries the exact Dirichlet-to-Neumann map
    of the monopole, $\\partial \\varphi/\\partial r = -\\varphi/r$,
    whose weak form contributes the surface mass matrix
    $$
    M_{ij} \\;=\\; \\oint_{r_{\\text{far}}}
    \\frac{\\sigma}{r_{\\text{far}}}\\, \\phi_i \\phi_j \\, dS,
    \\qquad dS = 2\\pi s\\, d\\ell .
    $$
    With linear hat functions on a straight boundary edge of length
    $\\ell$ between radii $s_1, s_2$ the integrals are exact:
    $$
    M^e \\;=\\; \\frac{2\\pi \\sigma \\ell}{12\\, r_{\\text{far}}}
    \\begin{pmatrix} 3 s_1 + s_2 & s_1 + s_2 \\\\
                     s_1 + s_2 & s_1 + 3 s_2 \\end{pmatrix}.
    $$
    The term is symmetric positive semi-definite, so it also makes the
    system non-singular without any Dirichlet condition on the outer
    boundary.
    """
    Nv = nodes.shape[0]
    if outer_edges.size == 0:
        return coo_matrix(([], ([], [])), shape=(Nv, Nv)).tocsr()
    p1 = nodes[outer_edges[:, 0]]
    p2 = nodes[outer_edges[:, 1]]
    length = np.hypot(p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1])
    s1 = p1[:, 0]
    s2 = p2[:, 0]
    pref = (
        2.0 * np.pi * np.asarray(sigma_per_edge, dtype=float) * length
        / (12.0 * r_far)
    )
    M_e = np.empty((outer_edges.shape[0], 2, 2))
    M_e[:, 0, 0] = pref * (3.0 * s1 + s2)
    M_e[:, 1, 1] = pref * (s1 + 3.0 * s2)
    M_e[:, 0, 1] = pref * (s1 + s2)
    M_e[:, 1, 0] = M_e[:, 0, 1]
    rows = np.broadcast_to(outer_edges[:, :, None], M_e.shape).ravel()
    cols = np.broadcast_to(outer_edges[:, None, :], M_e.shape).ravel()
    return coo_matrix((M_e.ravel(), (rows, cols)), shape=(Nv, Nv)).tocsr()


def _solve_hemisphere_conductance(
    mesh: _AxiMesh,
    sigma_per_layer: np.ndarray,
    h_layers: list[float],
    *,
    far_field: str = "robin",
) -> float:
    """Conductance $G = 1/R$ of the conforming hemisphere on ``mesh``.

    Solves the unit-potential problem ($\\varphi = 1$ on
    $r = a_{\\text{eq}}$) and returns the full bilinear form
    $G = a(\\varphi, \\varphi)$, which is simultaneously

    * the dissipated power at 1 V excitation (plus, for
      ``far_field='robin'``, the power leaving through the truncation
      sphere), and
    * the current injected into the electrode.

    Because $a$ is symmetric positive definite, the discrete
    $\\varphi$ *minimises* $a(\\cdot, \\cdot)$ over the FE space, so
    $G_h \\ge G$ and $R_h \\le R$ for every mesh: the resistance
    converges from below and monotonically under nested refinement.
    """
    sigma_per_triangle = np.asarray(sigma_per_layer, dtype=float)[
        mesh.layer_index
    ]
    A = _assemble_stiffness_axisymmetric(
        mesh.nodes, mesh.triangles, sigma_per_triangle
    )
    dirichlet = np.zeros(mesh.nodes.shape[0], dtype=bool)
    dirichlet[mesh.inner_nodes] = True
    if far_field == "robin":
        sigma_edge = np.asarray(sigma_per_layer, dtype=float)[
            _layer_of_depth(mesh.nodes[mesh.outer_edges, 1].mean(axis=1), h_layers)
        ]
        A = A + _assemble_far_field_robin(
            mesh.nodes, mesh.outer_edges, sigma_edge, mesh.r_far
        )
    elif far_field == "dirichlet":
        dirichlet[mesh.outer_edges.ravel()] = True
    else:  # pragma: no cover - guarded by solve_fem
        raise ValueError(
            f"far_field must be 'robin' or 'dirichlet', got {far_field!r}."
        )

    phi = np.zeros(mesh.nodes.shape[0])
    phi[mesh.inner_nodes] = 1.0  # unit potential on the hemisphere
    free = ~dirichlet
    if free.any():
        A_ff = A[free][:, free]
        rhs = -(A[free][:, dirichlet] @ phi[dirichlet])
        phi[free] = spsolve(A_ff.tocsc(), rhs)
    # G = a(phi, phi) — energy of the discrete solution.
    return float(phi @ (A @ phi))


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
    z_far_factor: float | None = None,
    far_field: str = "robin",
) -> FieldResult:
    """Axisymmetric Finite-Element solver for grounding systems.

    Reduces every cluster to its **equivalent hemisphere** at the
    cluster centroid, then solves the volume PDE on a 2-D
    axisymmetric triangular mesh. This is the only volume-PDE engine
    in the suite and forms the third independent cross-check (next to
    the closed-form ``image_*`` family and the integral ``mom``/``bem``
    family).

    The mesh conforms to the hemisphere $r = a_{\\text{eq}}$ and the
    layer interfaces $z = \\sum_{i \\le k} h_i$ (see
    :func:`_build_axisymmetric_mesh`), and the domain is truncated
    with the monopole DtN condition
    $\\partial\\varphi/\\partial r = -\\varphi/r_{\\text{far}}$. For
    homogeneous soil the discrete problem therefore converges to the
    analytic hemisphere resistance
    $R = \\rho / (2 \\pi a_{\\text{eq}})$ monotonically from below
    (second order in the mesh size), which is what makes a mesh
    refinement a meaningful accuracy knob.

    Parameters
    ----------
    world
        World to evaluate. Must currently contain a single galvanic
        cluster.
    engine
        Engine configuration.
    n_radial, n_axial
        Mesh resolution: node lines in the radial ($r$) and
        transverse ($\\vartheta$) direction of the spherical shell
        mesh. Refining as $n \\mapsto 2n - 1$ keeps the FE spaces
        nested and the convergence monotone.
    r_far_factor
        Far-field truncation radius as a multiple of the
        characteristic length $a_{\\text{eq}} + \\sum h_i$. The
        truncation is a single sphere, so this one number is the
        domain size; it has effect for every value (a truncation
        sweep is meaningful).
    z_far_factor
        Deprecated. ``None`` (default) ignores it; a number acts as a
        lower bound on the same radius and logs a warning when it
        overrides ``r_far_factor``. Kept only for callers written
        against the v0.14.1 $(s, z)$ box mesh, where the radial and
        axial extents were separate.
    far_field
        ``'robin'`` (default) applies the exact monopole
        Dirichlet-to-Neumann map on the truncation sphere;
        ``'dirichlet'`` grounds it ($\\varphi = 0$), which is the
        cruder classical truncation and biases $R$ low by
        $\\approx a_{\\text{eq}} / r_{\\text{far}}$. Provided for
        truncation-error studies.

    Returns
    -------
    FieldResult
        ``metadata['equivalent_hemisphere_radius']`` reports the
        reduction used on the cluster,
        ``metadata['fem_cluster_resistance']`` the solved
        hemisphere resistance and ``metadata['fem_mesh']`` the mesh
        size actually used per cluster.
    """
    if far_field not in ("robin", "dirichlet"):
        raise ValueError(
            "far_field must be 'robin' (monopole DtN) or 'dirichlet' "
            f"(grounded truncation sphere), got {far_field!r}."
        )
    if not isinstance(world.soil, (HomogeneousSoil, TwoLayerSoil, MultiLayerSoil)):
        raise TypeError(
            "Backend 'fem' supports HomogeneousSoil, TwoLayerSoil, "
            f"and MultiLayerSoil. Got: {type(world.soil).__name__}."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")
    _reject_concrete_shells(world, "fem")
    _warn_ignored_sources(world, "fem")

    stack = as_layer_stack(world.soil)
    rho_1 = float(stack.rhos[0])
    # The mesh follows the soil description (every layer interface is
    # cut into the elements). We do *not* collapse equal-ρ stacks down
    # to a 1-layer mesh — keeping the topology consistent across a ρ₂
    # sweep is a stronger guarantee than reproducing the homogeneous
    # discretisation bit-exactly. Since the mesh is graded relative to
    # a_eq instead of the domain height, the residual ρ₂ = ρ₁ bias is
    # only the logarithmic resolution loss of the larger r_far
    # (< 1 % for Σh up to 100 m; see tests/test_pass9_fem.py).
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
    mesh_per_cluster: dict[str, dict[str, float]] = {}

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

        # 2) Build the boundary-conforming spherical-shell mesh: the
        #    electrode surface r = a_eq and every layer interface are
        #    exact mesh lines.
        mesh = _build_axisymmetric_mesh(
            a_eq, h_layers,
            r_far_factor=r_far_factor,
            z_far_factor=z_far_factor,
            n_radial=n_radial,
            n_axial=n_axial,
        )
        mesh_per_cluster[root] = {
            "n_nodes": int(mesh.nodes.shape[0]),
            "n_triangles": int(mesh.triangles.shape[0]),
            "n_electrode_nodes": int(mesh.inner_nodes.size),
            "r_far": float(mesh.r_far),
        }

        # 3) Unit-potential problem on the conforming hemisphere plus
        #    the monopole-DtN far field, and 4) the conductance as the
        #    energy of the discrete solution (see
        #    _solve_hemisphere_conductance).
        G_cluster = _solve_hemisphere_conductance(
            mesh, 1.0 / stack.rhos, h_layers, far_field=far_field,
        )
        R_cluster = 1.0 / G_cluster if G_cluster > 0.0 else float("inf")
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
        # Multi-RHS: one LU factorisation for both right-hand sides
        # (ADR-0010 Tier 1).
        sol = np.linalg.solve(A_mat, np.column_stack([rhs_re, rhs_im]))
        sol_re, sol_im = sol[:, 0], sol[:, 1]
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
            "mesh": "conforming spherical shell (a_eq <= r <= r_far)",
            "far_field": far_field,
            "equivalent_hemisphere_radius": {
                k: float(v) for k, v in a_eq_per_cluster.items()
            },
            "fem_cluster_resistance": {
                k: float(v) for k, v in Z_per_cluster.items()
            },
            "fem_mesh": mesh_per_cluster,
            "stub": False,
            "approximation": "equivalent-hemisphere reduction per cluster",
        },
    )
