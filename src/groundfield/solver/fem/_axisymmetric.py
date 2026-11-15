"""Axisymmetric conforming model of coaxial rods and rings (private).

The world must be a body of revolution about one vertical axis: every
electrode is a :class:`~groundfield.geometry.electrodes.RodElectrode` on
the axis or a :class:`~groundfield.geometry.electrodes.RingElectrode`
centred on it, in any horizontally layered soil. Every other world is
refused (:class:`~groundfield.solver.registry.BackendCapabilityError`,
naming the backends that solve it). The electrodes are modelled as they
are: the rod as a solid cylinder with flat end faces, the ring as a solid
torus of circular cross section, both perfect conductors, an encased
electrode with its concrete shell (:mod:`groundfield.solver.fem._assembly`).

Every electrode is one port. The mesh (:mod:`groundfield.solver.fem._mesh`)
is solved at the refinement levels ``refinement`` and ``refinement + 1``;
with ``extrapolate`` the port conductance matrix is extrapolated to zero
mesh size with the observed second order (Richardson,
$G = G_{\\ell+1} + (G_{\\ell+1} - G_\\ell)/3$), and the largest relative
difference of a port resistance between the extrapolated and the finer
matrix is reported as the error estimate (``metadata["fem_error_estimate"]``;
``NaN`` for a single level, audit W4A-02). It bounds the error of the
extrapolated value while the error of the finer level is at most 1/2.5 of
that of the coarser (observed: 1/3.9 away from interfaces); against the
finer level itself it is 0.88 ... 0.99 of its error in 36 cases of audit
W4A, 0.80 ... 1.0 next to an interface cut into a ring's block.

The classification (:func:`classify`) is cheap and raises
:class:`NotAxisymmetric` without the remedy text, so ``model="auto"`` falls
back to the three-dimensional model at no cost; it includes the check that
every ring has room for its mesh block (audit W4A-03).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from groundfield.geometry.electrodes import RingElectrode, RodElectrode
from groundfield.solver._layered import as_layer_stack
from groundfield.solver.fem._assembly import solve_ports
from groundfield.solver.fem._mesh import (
    MeshParameters,
    RingGeometry,
    RodGeometry,
    build_mesh,
    mesh_lines,
    predicted_nodes,
)
from groundfield.solver.fem._ports import EarthPorts

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__: list[str] = []

#: Tolerance of the common axis (horizontal offset of a rod or a ring centre
#: from the axis of the first electrode), relative to the smallest length the
#: offset acts on: the smallest wire radius and the smallest surface gap
#: between two electrodes (audit W4A-08). The resistance matrix is even in the
#: offset (the world offset by -delta is the mirror image of the one offset by
#: +delta), so the axisymmetric reading of an offset world is off by
#: kappa (delta / L)^2: below 1e-6 at delta <= 1e-3 L for kappa <= 1. Measured
#: with ``image`` and ``mom`` on identical discretisations (a ring around a
#: rod, two rods 0.1 m apart on one axis, two rings 34 mm apart): kappa <= 6e-4
#: for L the wire radius, <= 1.1e-2 for L the gap. Up to audit W4A-08 the
#: tolerance was 1e-9 of the world's size (3 nm for a 3 m rod), and a rod
#: 1 um off a ring's axis went to the 3-D model (40.7 s, 2.35 GB).
AXIS_TOL = 1e-3

#: Floor of the axis tolerance relative to the larger of 1 m and the world's
#: size: the round-off of the coordinates.
_AXIS_ROUND_OFF = 1e-12

#: A ring wire may come no closer than this fraction of its radius to the
#: soil surface, the axis or another electrode (the solid torus must fit).
_CLEARANCE = 0.05

#: An interface farther than ``1 + _CLEARANCE`` wire radii from a ring's
#: axis is kept outside its O-grid block; a nearer one is cut into it.
_SHARE = 0.45


class NotAxisymmetric(Exception):
    """Why the axisymmetric model does not take a world (internal).

    :func:`classify` raises it without building the remedy text, so the
    dispatch of ``model="auto"`` stays cheap; :func:`coaxial_electrodes`
    turns it into the user-facing
    :class:`~groundfield.solver.registry.BackendCapabilityError`.
    """


class CoaxialElectrode(NamedTuple):
    """One electrode of an axisymmetric world."""

    #: Index in ``world.electrodes`` (the port index).
    index: int
    name: str
    #: ``RodGeometry`` or ``RingGeometry`` (meridian plane, m).
    geometry: object
    #: Concrete-shell coefficient $C$ in Ω·m (ADR-0012).
    shell: float


def _refuse(world: "World", engine: "Engine", why: str) -> None:
    from groundfield.solver.fem._guards import remedy
    from groundfield.solver.registry import BackendCapabilityError

    raise BackendCapabilityError(
        "Backend 'fem' cannot solve this world: " + why + " " + remedy(world, engine)
    )


_THREE_D = (
    "The three-dimensional model solves it (FemOptions.model='auto', the "
    "default, or '3d'; the fem3d extra)."
)


def coaxial_electrodes(
    world: "World", engine: "Engine"
) -> tuple[tuple[float, float], list[CoaxialElectrode]]:
    """The electrodes of an axisymmetric world, or a refusal.

    :func:`classify` with the refusal turned into a
    :class:`~groundfield.solver.registry.BackendCapabilityError` that names
    the backends that solve the world.

    Raises
    ------
    BackendCapabilityError
        For every reason of :class:`NotAxisymmetric`.
    """
    try:
        return classify(world)
    except NotAxisymmetric as exc:
        _refuse(world, engine, str(exc))
        raise  # pragma: no cover - _refuse raises


def classify(world: "World") -> tuple[tuple[float, float], list[CoaxialElectrode]]:
    """The electrodes of an axisymmetric world.

    Returns
    -------
    axis
        ``(x0, y0)`` in m of the common axis.
    electrodes
        One :class:`CoaxialElectrode` per electrode, in world order. Ring
        geometries carry a provisional block half-width; see
        :func:`block_half_widths`.

    Raises
    ------
    NotAxisymmetric
        For an electrode that is neither a rod nor a ring, a rod or ring
        off the common axis (by more than :data:`AXIS_TOL` of the smallest
        wire radius or surface gap), touching or overlapping electrodes, a
        ring wire that reaches within 5 % of its radius to the soil
        surface, and a ring without room for its mesh block (within about a
        wire radius, diagonally, of a rod's end edge; audit W4A-03).
    """
    kinds = sorted(
        {
            type(e).__name__
            for e in world.electrodes
            if not isinstance(e, (RodElectrode, RingElectrode))
        }
    )
    if kinds:
        names = [
            e.name
            for e in world.electrodes
            if not isinstance(e, (RodElectrode, RingElectrode))
        ]
        shown = ", ".join(repr(n) for n in names[:5])
        more = f" and {len(names) - 5} more" if len(names) > 5 else ""
        raise NotAxisymmetric(
            "its axisymmetric model represents rods on one vertical axis and "
            "horizontal rings centred on that axis, and electrode(s) "
            f"{shown}{more} ({', '.join(kinds)}) are not axisymmetric. The "
            "three-dimensional model solves them (FemOptions.model='auto', the "
            "default, or '3d'; the fem3d extra)."
        )
    out: list[CoaxialElectrode] = []
    for k, e in enumerate(world.electrodes):
        shell = float(getattr(e, "concrete_shell_coefficient_ohm_m", 0.0) or 0.0)
        if isinstance(e, RodElectrode):
            z0 = max(float(e.position[2]), 0.0)  # a head at round-off above 0
            geo: object = RodGeometry(
                radius=float(e.wire_radius),
                z_top=z0,
                z_bottom=float(e.position[2]) + float(e.length),
            )
        else:
            geo = RingGeometry(
                radius=float(e.radius),
                depth=float(e.center[2]),
                wire_radius=float(e.wire_radius),
                half_width=0.0,
            )
        out.append(CoaxialElectrode(k, e.name, geo, shell))
    points = []
    for e in world.electrodes:
        p = e.position if isinstance(e, RodElectrode) else e.center
        points.append((float(p[0]), float(p[1])))
    xy = np.array(points)
    x0, y0 = float(xy[0, 0]), float(xy[0, 1])
    scale = max(1.0, float(np.max(np.abs(xy))))
    for e in world.electrodes:
        extent = e.length if isinstance(e, RodElectrode) else e.radius
        scale = max(scale, float(extent))
    tol = max(AXIS_TOL * _offset_scale(out), _AXIS_ROUND_OFF * scale)
    offset = np.hypot(xy[:, 0] - x0, xy[:, 1] - y0)
    off = offset > tol
    if off.any():
        k = int(np.argmax(off))
        e = world.electrodes[k]
        what = "rod head" if isinstance(e, RodElectrode) else "ring centre"
        raise NotAxisymmetric(
            "its axisymmetric model needs every rod on one vertical axis and "
            f"every ring centred on it, and the {what} of electrode '{e.name}' "
            f"lies at (x, y) = ({xy[k, 0]:.6g}, {xy[k, 1]:.6g}) m, "
            f"{float(offset[k]):.3g} m off the axis of "
            f"'{world.electrodes[0].name}' at ({x0:.6g}, {y0:.6g}) m (tolerance "
            f"{tol:.3g} m). " + _THREE_D
        )
    _check_overlaps(out)
    stack = as_layer_stack(world.soil)
    block_half_widths(out, [float(d) for d in np.cumsum(stack.h)], MeshParameters())
    return (x0, y0), out


def _rect_gap(
    s: float, z: float, s_hi: float, z_lo: float, z_hi: float
) -> tuple[float, float]:
    """Euclidean and Chebyshev distance of ``(s, z)`` from ``[0, s_hi] x [z_lo, z_hi]``."""
    ds = max(s - s_hi, 0.0)
    dz = max(z_lo - z, 0.0, z - z_hi)
    return math.hypot(ds, dz), max(ds, dz)


def _offset_scale(electrodes: list[CoaxialElectrode]) -> float:
    """Smallest wire radius and positive surface gap in m (the axis tolerance).

    The gaps are those of the meridian plane: between rods on the axis
    (head of the lower minus tip of the upper), between a ring wire and a
    rod, and between two ring wires.
    """
    rods = [e.geometry for e in electrodes if isinstance(e.geometry, RodGeometry)]
    rings = [e.geometry for e in electrodes if isinstance(e.geometry, RingGeometry)]
    lengths = [r.radius for r in rods] + [g.wire_radius for g in rings]
    by_top = sorted(rods, key=lambda r: r.z_top)
    gaps = [
        lo.z_top - up.z_bottom for up, lo in zip(by_top[:-1], by_top[1:], strict=True)
    ]
    for g in rings:
        for r in rods:
            euclid, _ = _rect_gap(g.radius, g.depth, r.radius, r.z_top, r.z_bottom)
            gaps.append(euclid - g.wire_radius)
    for i, g1 in enumerate(rings):
        for g2 in rings[i + 1 :]:
            centres = math.hypot(g1.radius - g2.radius, g1.depth - g2.depth)
            gaps.append(centres - g1.wire_radius - g2.wire_radius)
    return min(lengths + [d for d in gaps if d > 0.0])


def _check_overlaps(electrodes: list[CoaxialElectrode]) -> None:
    rods = [e for e in electrodes if isinstance(e.geometry, RodGeometry)]
    rings = [e for e in electrodes if isinstance(e.geometry, RingGeometry)]
    by_top = sorted(rods, key=lambda e: e.geometry.z_top)
    for upper, lower in zip(by_top[:-1], by_top[1:], strict=True):
        if lower.geometry.z_top <= upper.geometry.z_bottom:
            raise NotAxisymmetric(
                f"rods '{upper.name}' and '{lower.name}' on the same axis touch or "
                f"overlap (z = {upper.geometry.z_top:.4g} ... "
                f"{upper.geometry.z_bottom:.4g} m and {lower.geometry.z_top:.4g} ... "
                f"{lower.geometry.z_bottom:.4g} m); bond them or move one."
            )
    for ring in rings:
        g = ring.geometry
        a = g.wire_radius
        if g.depth - a <= _CLEARANCE * a:
            raise NotAxisymmetric(
                f"the wire of ring '{ring.name}' (radius {a:.4g} m at z = "
                f"{g.depth:.4g} m) reaches within {_CLEARANCE:.0%} of its radius "
                "to the soil surface or above it; fem models the wire as the "
                "solid torus it is. Bury it deeper."
            )
        for rod in rods:
            r = rod.geometry
            euclid, _ = _rect_gap(g.radius, g.depth, r.radius, r.z_top, r.z_bottom)
            if euclid <= (1.0 + _CLEARANCE) * a:
                raise NotAxisymmetric(
                    f"ring '{ring.name}' touches or nearly touches rod '{rod.name}' "
                    f"(wire surface {euclid - a:.3g} m from the rod); fem models "
                    "both as the solid bodies they are."
                )
    for i, first in enumerate(rings):
        for second in rings[i + 1 :]:
            g1, g2 = first.geometry, second.geometry
            gap = math.hypot(g1.radius - g2.radius, g1.depth - g2.depth)
            if gap <= (1.0 + _CLEARANCE) * (g1.wire_radius + g2.wire_radius):
                raise NotAxisymmetric(
                    f"rings '{first.name}' and '{second.name}' touch or nearly "
                    "touch; bond them or move one."
                )


def block_half_widths(
    electrodes: list[CoaxialElectrode],
    interfaces: list[float],
    params: MeshParameters,
) -> list[CoaxialElectrode]:
    """Electrodes with the half-width of every ring's O-grid block set.

    The block is a square of half-width $c$ around the wire axis, at most
    ``params.box_half_width`` wire radii, and stays clear of the soil
    surface, the axis, the rods, the other rings' blocks and every
    interface farther than $1.05\\,a$ from the wire axis (it takes at most
    45 % of each gap beyond the wire); a nearer interface is cut into it,
    and the ring is marked ``near_interface`` (its block's lines along $s$
    are refined, audit W4A-01).

    Raises
    ------
    NotAxisymmetric
        If a block cannot contain its wire (a ring within a wire radius,
        diagonally, of a rod's end edge, audit W4A-03); :func:`classify`
        runs this check, so the dispatch takes the 3-D model instead.
    """
    rods = [e.geometry for e in electrodes if isinstance(e.geometry, RodGeometry)]
    rings = [e for e in electrodes if isinstance(e.geometry, RingGeometry)]
    limit: dict[int, tuple[float, bool]] = {}
    for ring in rings:
        g = ring.geometry
        a = g.wire_radius
        c = params.box_half_width * a
        c = min(c, a + _SHARE * (g.depth - a), a + _SHARE * (g.radius - a))
        for r in rods:
            _, cheb = _rect_gap(g.radius, g.depth, r.radius, r.z_top, r.z_bottom)
            c = min(c, a + _SHARE * (cheb - a))
        near = False
        for h in interfaces:
            d = abs(g.depth - h)
            if d > (1.0 + _CLEARANCE) * a:
                c = min(c, a + _SHARE * (d - a))
            else:
                near = True
        for other in rings:
            if other is ring:
                continue
            o = other.geometry
            cheb = max(abs(g.radius - o.radius), abs(g.depth - o.depth))
            c = min(c, a + _SHARE * (cheb - a - o.wire_radius))
        if not c > a:
            raise NotAxisymmetric(
                f"there is no room for the mesh block around the wire of ring "
                f"'{ring.name}' (half-width {c:.3g} m <= wire radius {a:.3g} m): "
                "the wire lies within about its radius, diagonally, of a rod's "
                "end edge, another ring or the soil surface, where the "
                "axisymmetric mesh cannot resolve the gap. " + _THREE_D
            )
        limit[ring.index] = (c, near)
    out = []
    for e in electrodes:
        if e.index in limit:
            g = e.geometry
            c, near = limit[e.index]
            e = e._replace(
                geometry=RingGeometry(
                    radius=g.radius,
                    depth=g.depth,
                    wire_radius=g.wire_radius,
                    half_width=c,
                    near_interface=near,
                )
            )
        out.append(e)
    return out


def _hat_fractions(z_nodes: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Share of each node's hat function (along $z$) in each interval.

    Parameters
    ----------
    z_nodes
        Sorted depths in m of the nodes of a straight boundary line.
    bounds
        Sorted interval ends in m, covering the nodes.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_nodes, n_intervals)``; every row sums to 1.
    """
    lo = np.concatenate([[z_nodes[0]], z_nodes[:-1]])
    hi = np.concatenate([z_nodes[1:], [z_nodes[-1]]])
    z = z_nodes[:, None]
    rise_len = (z_nodes - lo)[:, None]
    fall_len = (hi - z_nodes)[:, None]
    b = bounds[None, :]
    u = np.clip(b, lo[:, None], z)
    v = np.clip(b, z, hi[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        rise = np.where(rise_len > 0.0, (u - lo[:, None]) ** 2 / (2.0 * rise_len), 0.0)
        fall = np.where(fall_len > 0.0, (v - z) - (v - z) ** 2 / (2.0 * fall_len), 0.0)
    cdf = rise + fall
    total = 0.5 * (rise_len + fall_len)[:, 0]
    return np.diff(cdf, axis=1) / total[:, None]


def _display_segments(world: "World", engine: "Engine", interfaces: list[float]):
    """Leakage segments of every electrode as the integral backends cut them.

    They carry the point sources of the result. Their count is checked
    against the engine's resource limit first, like the segments of every
    backend (:func:`groundfield.diagnostics.check_segment_budget`), so a unit
    error of ``segment_length`` fails before anything is allocated.
    """
    from groundfield.diagnostics import check_segment_budget
    from groundfield.solver._discretize import _discretize_electrode

    check_segment_budget(world, engine, interfaces or None)

    segs, owner = [], []
    for k, e in enumerate(world.electrodes):
        for s in _discretize_electrode(e, engine.segment_length, interfaces or None):
            segs.append(s)
            owner.append(k)
    return segs, np.array(owner, dtype=int)


def _leakage_matrix(
    world: "World",
    electrodes: list[CoaxialElectrode],
    port_of_mesh: list[int],
    mesh,
    sol,
    segs,
    owner: np.ndarray,
) -> np.ndarray:
    """Leakage per display segment for 1 V per port, shape ``(M, P)``."""
    n_ports = len(electrodes)
    q = np.zeros((len(segs), n_ports))
    lengths = np.array([s.length for s in segs])
    for m, k in enumerate(port_of_mesh):
        e = electrodes[k]
        rows = np.nonzero(owner == k)[0]
        nodes = sol.surface_nodes[m]
        leak = sol.leakage[m]  # (n_k, P_mesh)
        leak = leak[:, np.argsort(port_of_mesh)]  # columns in world order
        if isinstance(e.geometry, RingGeometry):
            total = leak.sum(axis=0)
            share = lengths[rows] / lengths[rows].sum()
            q[rows] = share[:, None] * total[None, :]
            continue
        g = e.geometry
        s = mesh.nodes[nodes, 0]
        z = mesh.nodes[nodes, 1]
        lateral = s == g.radius
        top = (z == g.z_top) & ~lateral
        bottom = (z == g.z_bottom) & ~lateral
        mids = np.array([float(segs[i].midpoint[2]) for i in rows])
        order = np.argsort(mids)
        rows = rows[order]
        half = 0.5 * lengths[rows]
        bounds = np.concatenate([[mids[order][0] - half[0]], mids[order] + half])
        bounds[0], bounds[-1] = g.z_top, g.z_bottom
        lat = np.nonzero(lateral)[0]
        lat = lat[np.argsort(z[lat])]
        frac = _hat_fractions(z[lat], bounds)
        q[rows] += frac.T @ leak[lat]
        q[rows[0]] += leak[top].sum(axis=0)
        q[rows[-1]] += leak[bottom].sum(axis=0)
    return q


def axisymmetric_earth(
    world: "World",
    engine: "Engine",
    *,
    refinement: int = 0,
    extrapolate: bool = True,
    r_far_factor: float = 1000.0,
    far_field: str = "robin",
    max_nodes: int = 2_000_000,
    params: MeshParameters | None = None,
) -> EarthPorts:
    """Port model of an axisymmetric world (see the module docstring).

    Parameters
    ----------
    world, engine
        The world (in its local frame) and the engine; the engine's
        ``segment_length`` sets the display segments.
    refinement
        Refinement level of the coarser mesh (bisections of the level-0
        lines).
    extrapolate
        Solve also at ``refinement + 1`` and extrapolate the port
        conductance matrix (Richardson, second order). ``False``: the level
        ``refinement`` alone, with ``fem_error_estimate`` = ``NaN``.
    r_far_factor
        Size of the truncation box as a multiple of the characteristic
        length $L$: the largest of the electrodes' depth and radius and the
        depth of the deepest interface times the layer contrast
        $\\max\\rho / \\min\\rho$ (a resistive bottom keeps the current in
        the layers above it over that distance).
    far_field
        ``"robin"`` or ``"dirichlet"`` (:func:`~._assembly.solve_ports`).
    max_nodes
        Resource limit: the predicted node count of the finer mesh
        (:func:`~._mesh.predicted_nodes`, an estimate that errs high). The
        solve needs about 4 GB per million nodes of the finer mesh.

    Raises
    ------
    BackendCapabilityError
        For a world that is not axisymmetric (:func:`coaxial_electrodes`).
    groundfield.diagnostics.ResourceLimitError
        If the finer mesh would exceed ``max_nodes`` nodes.
    """
    from groundfield.diagnostics import ResourceLimitError

    params = params or MeshParameters()
    _axis, electrodes = coaxial_electrodes(world, engine)
    stack = as_layer_stack(world.soil)
    depths = [float(d) for d in np.cumsum(stack.h)]
    rhos = np.asarray(stack.rhos, dtype=float)
    try:
        electrodes = block_half_widths(electrodes, depths, params)
    except NotAxisymmetric as exc:  # only with a coarser params.box_half_width
        _refuse(world, engine, str(exc))

    extent = 1.0
    for e in electrodes:
        g = e.geometry
        if isinstance(g, RodGeometry):
            extent = max(extent, g.z_bottom)
        else:
            extent = max(extent, g.radius + g.wire_radius, g.depth + g.wire_radius)
    contrast = float(rhos.max() / rhos.min())
    length = max(extent, (depths[-1] if depths else 0.0) * contrast)
    r_far = float(r_far_factor) * length
    inside = [d for d in depths if d < r_far]

    rods = [e for e in electrodes if isinstance(e.geometry, RodGeometry)]
    rings = [e for e in electrodes if isinstance(e.geometry, RingGeometry)]
    port_of_mesh = [e.index for e in rods] + [e.index for e in rings]
    rod_geo = [e.geometry for e in rods]
    ring_geo = [e.geometry for e in rings]
    shells = []
    for k in port_of_mesh:
        e = electrodes[k]
        radius = (
            e.geometry.radius
            if isinstance(e.geometry, RodGeometry)
            else e.geometry.wire_radius
        )
        shells.append(1.0 / (2.0 * math.pi * radius * e.shell) if e.shell > 0 else 0.0)

    segs, owner = _display_segments(world, engine, depths)
    levels = [refinement, refinement + 1] if extrapolate else [refinement]
    lines = [
        mesh_lines(rod_geo, ring_geo, inside, r_far, params, level) for level in levels
    ]
    predicted = predicted_nodes(*lines[-1], ring_geo, levels[-1], inside)
    if predicted > max_nodes:
        raise ResourceLimitError(
            f"Backend 'fem': the mesh at refinement level {levels[-1]} would have "
            f"about {predicted} nodes, above the limit max_nodes = {max_nodes} "
            "(FemOptions). Lower FemOptions.refinement, or raise max_nodes to "
            "solve it on purpose."
        )
    solutions, meshes, info = [], [], []
    for level, ln in zip(levels, lines, strict=True):
        t0 = time.perf_counter()
        mesh = build_mesh(rod_geo, ring_geo, inside, r_far, params, level, lines=ln)
        sol = solve_ports(mesh, rhos, shells, far_field_model=far_field)
        meshes.append(mesh)
        solutions.append(sol)
        info.append({**mesh.stats, "solve_time_s": time.perf_counter() - t0})

    perm = np.argsort(port_of_mesh)  # mesh port order -> world order
    g_levels = [s.conductance[np.ix_(perm, perm)] for s in solutions]
    g_fine = g_levels[-1]
    if extrapolate:
        conductance = g_fine + (g_fine - g_levels[0]) / 3.0
    else:
        conductance = g_fine
    conductance = 0.5 * (conductance + conductance.T)

    leakage = _leakage_matrix(
        world, electrodes, port_of_mesh, meshes[-1], solutions[-1], segs, owner
    )
    r_levels = [np.linalg.inv(g) for g in g_levels]
    r_ext = np.linalg.inv(conductance)
    diag = np.abs(np.diag(r_ext))
    if extrapolate:
        estimate = float(np.max(np.abs(np.diag(r_ext) - np.diag(r_levels[-1])) / diag))
    else:  # one level has no estimate (audit W4A-02: it read 0.0)
        estimate = math.nan
    metadata = {
        "fem_model": "axisymmetric",
        "fem_ports": [world.electrodes[k].name for k in range(len(electrodes))],
        "fem_port_resistance": r_ext.tolist(),
        "fem_port_resistance_levels": [r.tolist() for r in r_levels],
        "fem_refinement": int(refinement),
        "fem_extrapolated": bool(extrapolate),
        "fem_error_estimate": estimate,
        "fem_mesh": info,
        "fem_r_far": r_far,
        "far_field": far_field,
    }
    return EarthPorts(
        names=tuple(e.name for e in world.electrodes),
        conductance=conductance,
        segment_points=np.array([s.midpoint for s in segs], dtype=float),
        segment_lengths=np.array([s.length for s in segs], dtype=float),
        segment_directions=np.array([s.direction for s in segs], dtype=float),
        segment_owner=owner,
        segment_leakage=leakage,
        metadata=metadata,
    )
