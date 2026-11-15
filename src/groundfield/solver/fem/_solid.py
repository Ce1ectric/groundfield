"""Three-dimensional conforming model of arbitrary electrodes (private).

Part B of user decision Q37 (scope of 2026-11-15): every world that the
axisymmetric model (:mod:`groundfield.solver.fem._axisymmetric`) cannot
hold — strips, polylines, stars, meshes, grids, rods and rings side by
side, several clusters, galvanic distributed conductors — is solved in
three dimensions. The soil is meshed with linear tetrahedra
(:mod:`groundfield.solver.fem._mesh3d`: an isotropic bisection mesh, no
interface is a mesh plane; the elements a layer interface cuts are
partially penalised immersed elements), the electrodes enter as smoothed
line sources along their wire axes with an analytic radius correction
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
(``metadata["fem_error_estimate"]``), as in the axisymmetric model. With
the ratio $r$ of the two levels' errors the extrapolated value errs by
$e_1(4 - r)/3$ and the estimate is $e_1(r - 1)/3$: a bound while
$r \\ge 2.5$ (rods and rings run at 3 … 5), no bound for resolved concrete,
which converges at first order (audit W4B-02).

Before anything is meshed the solve checks the display segments against
``Engine.max_segments`` and the number of ports against
:data:`MAX_PORTS_3D` (audit W4B-08).

The model needs the optional ``fem3d`` extra (``scikit-fem`` and
``pyamg``); without it the solve raises :class:`ImportError` naming the
extra.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from groundfield.solver._layered import as_layer_stack
from groundfield.solver.fem._ports import EarthPorts
from groundfield.solver.fem._wires3d import (
    BUMP_ENERGY_RADIUS,
    NEAR_FACTOR,
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
#: of side $s$ has the logarithmic capacity (outer conformal radius)
#: $0.59017\\,s$ (Pólya & Szegő 1951), so a circular shell of outer radius
#: $r_b$ becomes the square prism of side $r_b/0.59017$ with the same
#: resistance to the soil. The concrete between the bar and a nearly
#: equipotential prism surface ($\\rho_c \\gg \\rho$) is set by the square's
#: inner conformal radius at its centre, $0.539353\\,s = 0.9139\\,r_b$: in
#: that limit the prism's concrete is 3.8 % less resistive than the circular
#: shell (5 mm bar, 5 cm cover; audit W4B-04, evidence ``f03`` of stream
#: fw4fem3d). One side cannot match both radii; the prism keeps the outer one.
SQUARE_CONFORMAL_RADIUS = 0.5901702

#: The 3-D model refuses worlds with more ports than this before meshing
#: (audit W4B-08). Every electrode and every leaking piece of a galvanic
#: distributed conductor is a port, and every port costs one
#: conjugate-gradient solve per mesh level: 0.93 … 0.96 s per port and level
#: at 583 000 nodes of the finer mesh (a galvanic chain of 18 and 34 ports;
#: auditor W4B and evidence ``f14`` of stream fw4fem3d), growing with the
#: mesh. 100 ports are about 3 minutes of port solves at that size and about
#: 11 minutes at ``max_nodes_3d``.
MAX_PORTS_3D = 100

#: Port-solve time in s per port and per node of the finer mesh, both levels
#: together (0.93 s per port and level at 583 073 nodes, auditor W4B): the
#: estimate the port refusal names.
PORT_SECONDS_PER_NODE = 2.0 * 0.93 / 583_073


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
    """Peak resident memory of the process so far in GB (``ru_maxrss``).

    ``nan`` where the :mod:`resource` module does not exist (Windows).
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return math.nan
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


def _display_shares(model: WireModel, segs, owner: np.ndarray):
    """Share of every axial point's current per display segment.

    Every axial point stands for its stretch of the wire axis (half an axial
    cell, ``model.intervals``). The stretch is split among the display
    segments of its port in proportion to its overlap with each, measured
    along the segment, so a display segment carries the integral of the
    leakage over its own length. (The nearest segment per point would alias
    the axial cells against the display segments: ±4 % per segment on a 1 m
    ring at 0.1 m segments.) Only the segments nearest to the stretch's
    midpoint take part (a ring's chords, the pieces of a polyline at a
    corner); a stretch that overlaps none of them goes to the nearest one.

    Returns
    -------
    point, segment, share
        Triplets: the current of axial point ``point`` times ``share`` goes
        to display segment ``segment``; the shares of a point add up to 1.
    """
    port = model.port
    mids = np.array([s.midpoint for s in segs], dtype=float)
    dirs = np.array([s.direction for s in segs], dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    half = 0.5 * np.array([s.length for s in segs], dtype=float)
    start = np.array([p.start for p in model.pieces])[model.piece]
    tangent = np.array([p.direction for p in model.pieces])[model.piece]
    xa = start + model.intervals[:, :1] * tangent
    xb = start + model.intervals[:, 1:] * tangent
    xm = 0.5 * (xa + xb)
    out_p, out_s, out_w = [], [], []
    for k in np.unique(port):
        rows = np.nonzero(owner == k)[0]
        pts = np.nonzero(port == k)[0]
        n_chunks = max(1, (pts.size * rows.size) // 1_000_000 + 1)
        for chunk in np.array_split(pts, n_chunks):
            rel = xm[chunk][:, None, :] - mids[rows][None, :, :]
            t = np.clip(
                np.einsum("ijk,jk->ij", rel, dirs[rows]), -half[rows], half[rows]
            )
            dist = np.linalg.norm(rel - t[:, :, None] * dirs[rows][None, :, :], axis=2)
            nearest = np.argmin(dist, axis=1)
            dmin = dist[np.arange(chunk.size), nearest]
            ta = np.einsum(
                "ijk,jk->ij", xa[chunk][:, None, :] - mids[rows][None], dirs[rows]
            )
            tb = np.einsum(
                "ijk,jk->ij", xb[chunk][:, None, :] - mids[rows][None], dirs[rows]
            )
            lo = np.maximum(np.minimum(ta, tb), -half[rows])
            hi = np.minimum(np.maximum(ta, tb), half[rows])
            overlap = np.maximum(hi - lo, 0.0)
            close = dist <= np.minimum(
                0.5 * half[rows][None, :], dmin[:, None] + 0.1 * half[rows][None, :]
            )
            overlap = np.where(close, overlap, 0.0)
            total = overlap.sum(axis=1)
            lone = total <= 0.0
            overlap[lone, nearest[lone]] = 1.0
            total[lone] = 1.0
            i, j = np.nonzero(overlap)
            out_p.append(chunk[i])
            out_s.append(rows[j])
            out_w.append(overlap[i, j] / total[i])
    return np.concatenate(out_p), np.concatenate(out_s), np.concatenate(out_w)


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
    max_ports: int = MAX_PORTS_3D,
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
    max_ports
        Resource limit: the number of ports (electrodes and leaking
        conductor pieces), checked before meshing (:data:`MAX_PORTS_3D`).

    Raises
    ------
    ImportError
        Without the ``fem3d`` extra.
    BackendCapabilityError
        For a wire too close to the surface or an interface, for resolved
        concrete around wires that are not axis-aligned, and when the
        conjugate gradients do not converge (high layer contrast; tested up
        to about 300).
    groundfield.diagnostics.ResourceLimitError
        Before meshing: if the display segments exceed
        ``Engine.max_segments``, the ports ``max_ports`` or the predicted
        finer mesh ``max_nodes`` nodes; after the coarser mesh is built: if
        its refinement would exceed ``max_nodes``.
    """
    require_extra()
    from groundfield.diagnostics import ResourceLimitError
    from groundfield.solver.fem import _assembly3d as asm3
    from groundfield.solver.fem import _mesh3d as mesh3
    from groundfield.solver.fem._axisymmetric import _display_segments

    t_start = time.perf_counter()
    peak_before = _peak_memory_gb()
    stack = as_layer_stack(world.soil)
    rhos = np.asarray(stack.rhos, dtype=float)
    depths = [float(d) for d in np.cumsum(stack.h)]
    # Display segments (electrodes as the integral backends cut them, a
    # conductor piece as itself): their count is checked against
    # Engine.max_segments before anything is meshed (audit W4B-08).
    segs, owner = _display_segments(world, engine, depths)
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
    if n_ports > max_ports:
        minutes = n_ports * PORT_SECONDS_PER_NODE * predicted_fine / 60.0
        raise ResourceLimitError(
            f"Backend 'fem': the three-dimensional model would solve {n_ports} "
            f"ports, above its limit of {max_ports} ports (every electrode and "
            "every leaking piece of a galvanic distributed conductor is a port, "
            "and every port one conjugate-gradient solve per mesh level: about "
            f"{minutes:.0f} minutes at the predicted {predicted_fine} nodes of the "
            "finer mesh). Coarsen the galvanic conductors "
            "(discretize_segment_length), or use an integral backend "
            "(image, image_2layer, mom)."
        )
    if predicted_fine > max_nodes:
        raise ResourceLimitError(
            f"Backend 'fem': the three-dimensional mesh would have about "
            f"{predicted_fine} nodes at its finer level, above the limit "
            f"max_nodes_3d = {max_nodes} (FemOptions). Coarsen it "
            "(FemOptions.source_radius, growth, extrapolate=False), or raise "
            "max_nodes_3d to solve it on purpose."
        )

    # Coarse mesh: isotropic background refined towards the wire axes (and
    # the interfaces and concrete prism faces near them).
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
            f"finer level, above the limit max_nodes_3d = {max_nodes} (FemOptions). "
            "Coarsen it (FemOptions.source_radius, growth, extrapolate=False), or "
            "raise max_nodes_3d to solve it on purpose."
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
            p if p.shell == 0.0 else replace(p, radius=float(ra[i]), shell=0.0)
            for i, p in enumerate(model.pieces)
        ]
        corrected = WireModel(
            model.names, metal, model.points, model.weights, model.piece
        )
        rho_q = np.where(encased, rho_c, rho_q)
        coefficient = radius_correction(corrected, r_eta, rho_q, NEAR_FACTOR * r_src)
    else:
        coefficient = shell_q + radius_correction(
            model, r_eta, rho_q, NEAR_FACTOR * r_src
        )
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
        candidates = mesh3.near_elements(p, t, samples, r_src + h_w)
        average = asm3.disc_average_matrix(
            p, t, model.points, e1, e2, disc, disc_w, candidates
        )
        t2 = time.perf_counter()
        try:
            sol = asm3.solve_ports_3d(
                stiff, average, model.weights, coefficient, model.port, n_ports
            )
        except asm3.ConvergenceError as exc:
            _refuse(
                world,
                engine,
                "the conjugate gradients of its three-dimensional model did not "
                f"converge for port '{model.names[exc.port]}' in "
                f"{asm3.CG_MAX_ITERATIONS} iterations at a layer contrast of "
                f"{contrast:.4g} (max rho / min rho; the model is tested up to a "
                "contrast of about 300). Its axisymmetric model solves rods on "
                "one vertical axis and rings around it at any contrast "
                "(FemOptions.model='auto' or model='axisymmetric').",
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

    # Leakage of the display segments per port volt.
    segs = list(segs) + list(extra_segments)
    owner = np.concatenate(
        [owner, np.arange(len(world.electrodes), n_ports, dtype=int)]
    )
    point, segment, share = _display_shares(model, segs, owner)
    leakage = np.zeros((len(segs), n_ports))
    np.add.at(leakage, segment, share[:, None] * solutions[-1].current[point])

    r_levels = [np.linalg.inv(g) for g in g_levels]
    r_ext = np.linalg.inv(conductance)
    diag = np.abs(np.diag(r_ext))
    estimate = float(np.max(np.abs(np.diag(r_ext) - np.diag(r_levels[-1])) / diag))
    # Memory (audit W4B-09): what the solve added to the peak resident memory
    # of the process, and that peak itself.
    peak_after = _peak_memory_gb()
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
        "fem_peak_memory_gb": max(0.0, peak_after - peak_before),
        "fem_process_peak_memory_gb": peak_after,
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
