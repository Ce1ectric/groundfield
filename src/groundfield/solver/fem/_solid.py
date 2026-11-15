"""Three-dimensional conforming model of arbitrary electrodes (private).

Part B of user decision Q37 (scope of 2026-11-15): every world that the
axisymmetric model (:mod:`groundfield.solver.fem._axisymmetric`) cannot
hold — strips, polylines, stars, meshes, grids, rods and rings side by
side, several clusters, galvanic distributed conductors — is solved in
three dimensions. The soil is meshed with linear tetrahedra
(:mod:`groundfield.solver.fem._mesh3d`; every layer interface a mesh
plane), the electrodes enter as smoothed line sources along their wire
axes with an analytic radius correction
(:mod:`groundfield.solver.fem._wires3d`), a concrete shell either as its
per-length coefficient (each wire its own shell, ADR-0012) or — with
``concrete_resistivity`` — resolved as a volume of concrete around the
wire (user decision Q40), and the port conductance matrix comes from one
conjugate-gradient solve per port (:mod:`groundfield.solver.fem._assembly3d`).
Every electrode is one port, every leaking piece of a galvanic distributed
conductor one more; the network around them is the shared pipeline's
(:mod:`groundfield.solver.fem._network`).

Two nested meshes (the second the uniform refinement of the first) give
the port conductance matrix twice; it is extrapolated to zero mesh size
with the second order of linear elements, and the relative difference
between the extrapolated and the finer level is the error estimate
(``metadata["fem_error_estimate"]``), as in the axisymmetric model.

The model needs the optional ``fem3d`` extra (``scikit-fem`` and
``pyamg``); without it the solve raises :class:`ImportError` naming the
extra.
"""

from __future__ import annotations

import math
import resource
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from groundfield.solver._layered import as_layer_stack
from groundfield.solver.fem._ports import EarthPorts
from groundfield.solver.fem._wires3d import (
    BUMP_ENERGY_RADIUS,
    WireModel,
    axial_quadrature,
    frames,
    radius_correction,
    world_pieces,
)

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__: list[str] = []

#: Import hint of the optional extra.
FEM3D_IMPORT_HINT = (
    "Backend 'fem' solves this world with its three-dimensional model, which "
    "requires the optional 'fem3d' extra (scikit-fem, pyamg). Install with "
    "'pip install groundfield[fem3d]' or 'poetry install --extras fem3d'."
)

#: Default radius in m of the smoothed line sources.
DEFAULT_SOURCE_RADIUS = 0.06

#: Default growth of the 3-D mesh size per metre away from the wires.
DEFAULT_GROWTH = 0.5

#: An element that a layer interface cuts is no larger than this fraction
#: of its distance from the nearest wire.
INTERFACE_RATIO = 0.3

#: An element that a concrete prism face cuts is no larger than the
#: prism side divided by this.
PRISM_DIVISIONS = 8.0

#: Side of a square concrete prism per metre of conformal radius: a square
#: of side $s$ has the logarithmic capacity (conformal radius) $0.59017\\,s$
#: (Pólya & Szegő 1951), so a circular shell of outer radius $r_b$ becomes
#: the square prism of side $r_b/0.59017$ with the same resistance to the
#: soil.
SQUARE_CONFORMAL_RADIUS = 0.5901702


def require_extra() -> None:
    """Import the optional dependencies or raise the hint."""
    try:
        import pyamg  # noqa: F401
        import skfem  # noqa: F401
    except ImportError as exc:
        raise ImportError(FEM3D_IMPORT_HINT) from exc


def _refuse(world: "World", engine: "Engine", why: str) -> None:
    from groundfield.solver.fem._guards import remedy
    from groundfield.solver.registry import BackendCapabilityError

    raise BackendCapabilityError(
        "Backend 'fem' cannot solve this world: " + why + " " + remedy(world, engine)
    )


def _peak_memory_gb() -> float:
    """Peak resident memory of the process in GB (``ru_maxrss``)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak) / (2**30 if sys.platform == "darwin" else 2**20)


def source_radius(
    world: "World",
    engine: "Engine",
    model: WireModel,
    interfaces: list[float],
    requested: float | None,
    prisms_side: float | None,
) -> float:
    """Radius $R$ of the smoothed line sources, or a refusal.

    The default is :data:`DEFAULT_SOURCE_RADIUS`, at least 2.2 bare wire
    radii (the energy radius $0.544\\,R$ must exceed the wire radius). The
    disc of a piece must stay in the soil and in its layer: $R\\,f$ with
    $f = \\sqrt{1 - d_z^2}$ (the vertical reach of a disc perpendicular to
    the piece) is at most 0.8 times the depth of the piece and its gap to
    every interface it does not cross.
    """
    bare = [p.radius for p in model.pieces if p.shell == 0.0]
    a_bare = max(bare) if bare else 0.0
    smallest = 1.2 * a_bare / BUMP_ENERGY_RADIUS
    if requested is not None and float(requested) < smallest:
        raise ValueError(
            f"FemOptions.source_radius = {float(requested):.4g} m is too small for "
            f"the bare wires of radius {a_bare:.4g} m: the smoothed source must "
            f"reach beyond the wire, source_radius >= {smallest:.4g} m."
        )
    r = (
        float(requested)
        if requested is not None
        else max(DEFAULT_SOURCE_RADIUS, 2.2 * a_bare)
    )
    limit = math.inf
    for p in model.pieces:
        f = math.sqrt(max(0.0, 1.0 - float(p.direction[2]) ** 2))
        if f < 1e-12:
            continue
        z_lo = min(float(p.start[2]), float(p.end[2]))
        z_hi = max(float(p.start[2]), float(p.end[2]))
        gaps = [z_lo]
        for h in interfaces:
            if h <= z_lo:
                gaps.append(z_lo - h)
            elif h >= z_hi:
                gaps.append(h - z_hi)
        limit = min(limit, 0.8 * min(gaps) / f)
    if prisms_side is not None:
        limit = min(limit, 0.45 * prisms_side)
    r = min(r, limit)
    if not BUMP_ENERGY_RADIUS * r > 1.2 * a_bare:
        _refuse(
            world,
            engine,
            "its three-dimensional model spreads each wire over a disc that must "
            "lie in the soil and in one layer, and a wire lies too close to the "
            f"soil surface or to a layer interface for its radius ({a_bare:.3g} m): "
            f"the disc would be {r:.3g} m. Move the wire, or use an axisymmetric "
            "world (rods on one axis, rings around it).",
        )
    return r


def _prisms(world, engine, model: WireModel, rho_c: float):
    """Concrete prisms of the encased pieces (Q40), or a refusal."""
    from groundfield.solver.fem._mesh3d import Prism

    prisms = []
    side = None
    for p in model.pieces:
        if p.shell <= 0.0:
            continue
        d = np.abs(p.direction)
        if not np.isclose(d.max(), 1.0, rtol=0.0, atol=1e-9):
            _refuse(
                world,
                engine,
                "resolving the concrete as a volume (FemOptions.concrete_resistivity) "
                "needs straight encased wires along x, y or z; electrode "
                f"'{model.names[p.port]}' has an encased wire along "
                f"({p.direction[0]:.3g}, {p.direction[1]:.3g}, {p.direction[2]:.3g}). "
                "Leave concrete_resistivity at None (each wire its own shell).",
            )
        s = p.radius / SQUARE_CONFORMAL_RADIUS
        side = s if side is None else min(side, s)
        lo = np.minimum(p.start, p.end) - 0.5 * s
        hi = np.maximum(p.start, p.end) + 0.5 * s
        if lo[2] <= 0.0:
            _refuse(
                world,
                engine,
                f"the concrete around electrode '{model.names[p.port]}' would reach "
                "the soil surface; bury it deeper.",
            )
        prisms.append(Prism(lower=lo, upper=hi))
    del rho_c
    return tuple(prisms), side


def _display_assignment(model: WireModel, segs, owner: np.ndarray) -> np.ndarray:
    """Display segment of every axial point (the nearest of its port)."""
    out = np.zeros(model.points.shape[0], dtype=int)
    port = model.port
    mids = np.array([s.midpoint for s in segs], dtype=float)
    dirs = np.array([s.direction for s in segs], dtype=float)
    half = 0.5 * np.array([s.length for s in segs], dtype=float)
    for k in np.unique(port):
        rows = np.nonzero(owner == k)[0]
        pts = np.nonzero(port == k)[0]
        rel = model.points[pts][:, None, :] - mids[rows][None, :, :]
        t = np.clip(np.einsum("ijk,jk->ij", rel, dirs[rows]), -half[rows], half[rows])
        d = np.linalg.norm(rel - t[:, :, None] * dirs[rows][None, :, :], axis=2)
        out[pts] = rows[np.argmin(d, axis=1)]
    return out


def solid_earth(
    world: "World",
    engine: "Engine",
    *,
    refinement: int = 0,
    extrapolate: bool = True,
    r_far_factor: float = 1000.0,
    far_field: str = "robin",
    max_nodes: int = 4_000_000,
    source_radius_m: float | None = None,
    growth: float = DEFAULT_GROWTH,
    concrete_resistivity: float | None = None,
    extra_segments=(),
    concrete_mesh: float | None = None,
) -> EarthPorts:
    """Port model of any world in three dimensions (see the module docstring).

    Parameters
    ----------
    world, engine
        World in its local frame and the engine (``segment_length`` sets
        the display segments of the result).
    refinement
        Uniform refinements of the coarser mesh (each one halves every
        element, about eight times the nodes).
    extrapolate
        Also solve the uniform refinement of the coarser mesh and
        extrapolate the port conductance matrix (second order).
    r_far_factor
        Half-width of the truncation box in characteristic lengths, as in
        the axisymmetric model.
    far_field
        ``"robin"`` (monopole Dirichlet-to-Neumann map) or ``"dirichlet"``.
    max_nodes
        Resource limit: the node count of the finer mesh.
    source_radius_m
        Radius $R$ in m of the smoothed line sources; ``None`` takes the
        default (:func:`source_radius`).
    growth
        Growth of the mesh size per metre away from the wires.
    concrete_resistivity
        ``None``: every encased wire carries its shell coefficient $C$
        (ADR-0012, each wire its own shell). A value in Ω·m resolves the
        concrete as square prisms of that resistivity around the encased
        wires, the metal radius following from
        $C = \\frac{\\rho_c}{2\\pi}\\ln(r_b/r_a)$ (user decision Q40).
    extra_segments
        Leaking pieces of galvanic distributed conductors (one port each).
    concrete_mesh
        Private, for the overlap study of user decision Q40: build the mesh
        (prism refinement, source radius) as for resolved concrete of this
        resistivity in Ω·m but keep every wire's own shell coefficient, so
        both concrete models are compared on the same mesh.

    Raises
    ------
    ImportError
        Without the ``fem3d`` extra.
    BackendCapabilityError
        For a wire too close to the surface or an interface, and for
        resolved concrete around wires that are not axis-aligned.
    groundfield.diagnostics.ResourceLimitError
        If the finer mesh would exceed ``max_nodes`` nodes.
    """
    require_extra()
    from groundfield.diagnostics import ResourceLimitError
    from groundfield.solver.fem import _assembly3d as asm3
    from groundfield.solver.fem import _mesh3d as mesh3
    from groundfield.solver.fem._axisymmetric import _display_segments

    t_start = time.perf_counter()
    stack = as_layer_stack(world.soil)
    rhos = np.asarray(stack.rhos, dtype=float)
    depths = [float(d) for d in np.cumsum(stack.h)]
    model = world_pieces(world, extra_segments)
    n_ports = len(model.names)

    prisms: tuple = ()
    side = None
    if concrete_resistivity is not None or concrete_mesh is not None:
        prisms, side = _prisms(world, engine, model, 0.0)
    mesh_prisms = prisms
    if concrete_resistivity is None:
        prisms = ()  # concrete_mesh: refine at the prism faces only
    r_src = source_radius(world, engine, model, depths, source_radius_m, side)
    h_w = 0.5 * r_src
    n_levels = 2 if extrapolate else 1
    h_fine = h_w / 2 ** (refinement + n_levels - 1)

    all_pts = np.concatenate([np.stack([p.start, p.end]) for p in model.pieces])
    lower, upper = all_pts.min(axis=0), all_pts.max(axis=0)
    extent = max(1.0, float(np.max(upper - lower)), float(upper[2]))
    contrast = float(rhos.max() / rhos.min())
    length = max(extent, (depths[-1] if depths else 0.0) * contrast)
    r_far = float(r_far_factor) * length
    center = np.array([0.5 * (lower[0] + upper[0]), 0.5 * (lower[1] + upper[1]), 0.0])
    inside = [d for d in depths if d < r_far]

    predicted = mesh3.predicted_nodes(model.total_length, h_w, growth, r_src)
    predicted_fine = predicted * 8 ** (refinement + n_levels - 1)
    if predicted_fine > max_nodes:
        raise ResourceLimitError(
            f"Backend 'fem': the three-dimensional mesh would have about "
            f"{predicted_fine} nodes at its finer level, above the limit "
            f"max_nodes = {max_nodes} (FemOptions). Coarsen it "
            "(FemOptions.source_radius, growth, extrapolate=False), or raise "
            "max_nodes to solve it on purpose."
        )

    # Coarse mesh: isotropic background refined towards the wire axes,
    # then cut by every interface (and concrete prism face) plane.
    t0 = time.perf_counter()
    samples = np.concatenate(
        [
            np.linspace(p.start, p.end, max(2, math.ceil(p.length / (0.5 * h_w)) + 1))
            for p in model.pieces
        ]
    )
    bis = mesh3.background(center, r_far)
    prism_size = None if not mesh_prisms else side / PRISM_DIVISIONS
    mesh3.refine_towards(
        bis,
        samples,
        h_w,
        growth,
        r_src,
        interfaces=inside,
        interface_ratio=INTERFACE_RATIO,
        prisms=mesh_prisms,
        prism_size=prism_size,
    )
    from skfem import MeshTet

    mesh = MeshTet(bis.p, bis.t.astype(np.int32))
    for _ in range(refinement):
        mesh = mesh.refined()
    n_coarse = mesh.p.shape[1]
    if extrapolate:
        n_fine = n_coarse + mesh3.edge_count(mesh.t, n_coarse)
    else:
        n_fine = n_coarse
    if n_fine > max_nodes:
        raise ResourceLimitError(
            f"Backend 'fem': the three-dimensional mesh has {n_fine} nodes at its "
            f"finer level, above the limit max_nodes = {max_nodes} (FemOptions). "
            "Coarsen it (FemOptions.source_radius, growth, extrapolate=False), or "
            "raise max_nodes to solve it on purpose."
        )
    t_mesh = time.perf_counter() - t0

    # Wires: axial quadrature, local resistivity, radius correction.
    axial_quadrature(model, h_fine)
    zq = model.points[:, 2]
    rho_q = rhos[np.searchsorted(np.asarray(depths), zq, side="right")]
    encased = np.array([p.shell > 0.0 for p in model.pieces])[model.piece]
    shell_q = np.array([p.shell for p in model.pieces])[model.piece]
    r_eta = BUMP_ENERGY_RADIUS * r_src
    if prisms:
        rho_c = float(concrete_resistivity)
        rb = np.array([p.radius for p in model.pieces])
        ra = rb * np.exp(
            -2.0 * math.pi * np.array([p.shell for p in model.pieces]) / rho_c
        )
        metal = [
            p if p.shell == 0.0 else type(p)(p.port, p.start, p.end, float(ra[i]), 0.0)
            for i, p in enumerate(model.pieces)
        ]
        corrected = WireModel(
            model.names, metal, model.points, model.weights, model.piece
        )
        rho_q = np.where(encased, rho_c, rho_q)
        coefficient = radius_correction(corrected, r_eta, rho_q, 20.0 * r_src)
    else:
        coefficient = shell_q + radius_correction(model, r_eta, rho_q, 20.0 * r_src)
    if not np.all(coefficient > 0.0):  # pragma: no cover - guarded by source_radius
        raise ValueError("fem: a non-positive radial resistance of a wire.")
    tangent = np.array([p.direction for p in model.pieces])[model.piece]
    e1, e2 = frames(tangent)
    disc, disc_w = asm3.bump_disc(r_src, 0.5 * h_fine)

    planes = [
        (0, center[0] - r_far),
        (0, center[0] + r_far),
        (1, center[1] - r_far),
        (1, center[1] + r_far),
        (2, r_far),
    ]
    sigma_layers = 1.0 / rhos
    bounds = np.concatenate([[0.0], inside, [np.inf]])
    thickness = np.diff(bounds)
    material_planes = [
        (2, float(h), 0.25 * float(min(thickness[k], thickness[k + 1])))
        for k, h in enumerate(inside)
    ]
    for q in mesh_prisms:
        for axis in range(3):
            material_planes += [
                (axis, float(q.lower[axis]), 0.25 * side),
                (axis, float(q.upper[axis]), 0.25 * side),
            ]

    def box_fixed(nodes: np.ndarray) -> list[np.ndarray]:
        return [
            (nodes[0] == center[0] - r_far) | (nodes[0] == center[0] + r_far),
            (nodes[1] == center[1] - r_far) | (nodes[1] == center[1] + r_far),
            (nodes[2] == 0.0) | (nodes[2] == r_far),
        ]

    solutions, info = [], []
    for level in range(n_levels):
        t1 = time.perf_counter()
        if level:
            mesh = mesh.refined()
        t = mesh.t
        p = mesh3.snap_to_planes(mesh.p, t, material_planes, box_fixed(mesh.p))
        # The finer mesh refines the snapped coarser one (nested spaces).
        mesh = MeshTet(p, t)
        stiff = asm3.stiffness_matrix(
            p,
            t,
            np.asarray(inside, dtype=float),
            sigma_layers,
            prisms,
            None if not prisms else 1.0 / rho_c,
        )
        faces = mesh3.box_faces(p, t, planes)
        if far_field == "robin":
            zf = p[2, faces].mean(axis=0)
            sig_f = sigma_layers[np.searchsorted(np.asarray(inside), zf, side="right")]
            stiff = stiff + asm3.robin(p, faces, sig_f, center)
        else:
            stiff = _ground(stiff, np.unique(faces))
        from scipy.spatial import cKDTree

        tree = cKDTree(samples)
        dist, _ = tree.query(p[:, t].mean(axis=1).T)
        hmax = mesh3.longest_edges(p, t)
        candidates = np.nonzero(dist < r_src + hmax + h_w)[0]
        average = asm3.disc_average_matrix(
            p, t, model.points, e1, e2, disc, disc_w, candidates
        )
        t2 = time.perf_counter()
        sol = asm3.solve_ports_3d(
            stiff, average, model.weights, coefficient, model.port, n_ports
        )
        solutions.append(sol)
        info.append(
            {
                "nodes": int(p.shape[1]),
                "tetrahedra": int(t.shape[1]),
                "assembly_time_s": t2 - t1,
                "solve_time_s": time.perf_counter() - t2,
                "cg_iterations": sol.iterations,
            }
        )
    g_levels = [s.conductance for s in solutions]
    g_fine = g_levels[-1]
    conductance = g_fine + (g_fine - g_levels[0]) / 3.0 if extrapolate else g_fine
    conductance = 0.5 * (conductance + conductance.T)

    # Display segments (electrodes as the integral backends cut them, a
    # conductor piece as itself) and their leakage per port volt.
    segs, owner = _display_segments(world, engine, depths)
    segs = list(segs) + list(extra_segments)
    owner = np.concatenate(
        [owner, np.arange(len(world.electrodes), n_ports, dtype=int)]
    )
    assign = _display_assignment(model, segs, owner)
    leakage = np.zeros((len(segs), n_ports))
    np.add.at(leakage, assign, solutions[-1].current)

    r_levels = [np.linalg.inv(g) for g in g_levels]
    r_ext = np.linalg.inv(conductance)
    diag = np.abs(np.diag(r_ext))
    estimate = float(np.max(np.abs(np.diag(r_ext) - np.diag(r_levels[-1])) / diag))
    metadata = {
        "fem_model": "3d",
        "fem_ports": list(model.names),
        "fem_port_resistance": r_ext.tolist(),
        "fem_port_resistance_levels": [r.tolist() for r in r_levels],
        "fem_refinement": int(refinement),
        "fem_extrapolated": bool(extrapolate),
        "fem_error_estimate": estimate,
        "fem_mesh": info,
        "fem_mesh_time_s": t_mesh,
        "fem_r_far": r_far,
        "fem_source_radius": r_src,
        "fem_growth": float(growth),
        "fem_concrete": "shell coefficient per wire"
        if not prisms
        else f"resolved, {float(concrete_resistivity):g} Ohm m",
        "fem_peak_memory_gb": _peak_memory_gb(),
        "fem_time_s": time.perf_counter() - t_start,
        "far_field": far_field,
    }
    return EarthPorts(
        names=tuple(model.names),
        conductance=conductance,
        segment_points=np.array([s.midpoint for s in segs], dtype=float),
        segment_lengths=np.array([s.length for s in segs], dtype=float),
        segment_directions=np.array([s.direction for s in segs], dtype=float),
        segment_owner=owner,
        segment_leakage=leakage,
        metadata=metadata,
    )


def _ground(stiff, nodes: np.ndarray):
    """Dirichlet $\\varphi = 0$ on ``nodes`` (rows and columns to the identity)."""
    from scipy.sparse import diags

    keep = np.ones(stiff.shape[0])
    keep[nodes] = 0.0
    d = diags(keep)
    return (d @ stiff @ d + diags(1.0 - keep)).tocsr()
