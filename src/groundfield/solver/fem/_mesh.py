"""Conforming axisymmetric mesh of the soil around coaxial electrodes (private).

The meridian half-plane $(s, z)$, $s \\ge 0$ the distance from the
common vertical axis and $z \\ge 0$ the depth, is meshed with a
**graded tensor grid**: one set of vertical lines $s = s_i$ and one set of
horizontal lines $z = z_j$, each graded geometrically away from a list of
feature coordinates. Every feature is an exact mesh line:

* the surface $s = a$ and the end faces $z = z_0$, $z = z_1$ of every rod
  on the axis (radius $a$, flat ends);
* every soil-layer interface $z = h_k$;
* the soil surface $z = 0$, the axis $s = 0$ and the truncation box
  $s = R$, $z = R$.

The cells inside a rod are removed, so the rod is a rectangle cut out of
the grid and its boundary is a closed mesh polyline. The circular cross
section of a ring wire (radius $a$, centre $(r_0, t)$) cannot be a tensor
line; around it the grid leaves a square block of half-width $c$ out, and
an **O-grid** fills the block: the nodes of the block boundary are
projected radially onto the circle of the wire and joined by layers
spaced uniformly in $\\ln \\rho$, so the circle is inscribed exactly and
the block conforms to the tensor grid around it. A layer interface that
passes through such a block (a ring within about its wire radius of the
interface) is cut into the O-grid elements at its own mesh line
(:func:`cut_at_depth`), so every element lies in one layer; features
closer than :data:`_SAME_LINE` are one line (audit W4R3A-01), and a
triangle of zero area is refused as an internal error. Next to such
an interface the field varies on the scale of the gap between the wire
and the interface (and of the narrow wedge where an interface cuts the
wire near its edge), so that block has four times the lines
(``MeshParameters.near_interface_intervals``; audit W4A-01), and it is
singular where the interface cuts the wire or touches it: the O-grid of
that block is graded towards those points, its rays geometrically in
angle and its first layers geometrically in radius from cells of
``MeshParameters.contact_size`` wire radii (audit W4R2A-02).

Grading. The spacing at a feature $x_f$ with size $h_f$ grows linearly
with the distance, $h(x) = \\min_f\\,(h_f + g\\,|x - x_f|)$ with the growth
$g$ (geometric ratio $1 + g$), which is the natural grading for the
logarithmic field next to a wire and for the $1/r$ far field: the
truncation box can be placed thousands of electrode lengths away for a
few dozen extra lines. The nodes between two features follow the size
function exactly (:func:`graded_lines`).

Refinement. Level $\\ell$ bisects every interval of the level-0 lines
$\\ell$ times (and doubles the angular and radial resolution of every
O-grid; the graded layers of a block at an interface are bisected like
the lines), so the tensor part of the finite-element spaces is nested.
The O-grid is not: its layers move with the level and the polygon
inscribed in the wire grows towards the circle. The discrete conductance therefore
converges at second order but is not guaranteed to fall monotonically
(it does in every case tested away from interfaces; next to an interface
cut into a block it did not before that block was refined, audit
W4A-01).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__: list[str] = []


# ---------------------------------------------------------------------
# One-dimensional graded lines
# ---------------------------------------------------------------------


#: Coordinates of mesh features closer than this, relative to the larger of
#: 1 m and the coordinate, are one line (the tolerance of the line look-up
#: :func:`_index` too). Two features computed two ways differ by round-off
#: (5.6e-17 m for 0.1 + 0.2 and 0.3); a ring axis or a rod end a few
#: picometres off an interface is the same world physically. Up to audit
#: W4R3A-01 the tolerance was 1e-12, the size of the interface cut's snap
#: (:func:`cut_at_depth`): a feature just beyond it, or a line that bisection
#: put between it and the interface, was snapped onto the plane and left a
#: row of zero-area cells (+35 ... +44 %, a singular factor, or -2.7 % at
#: 2 m, silently). Distinct lines are now at least 1e-9 / 2**level of
#: max(1 m, x) apart, above the snap tolerance 1e-12 up to level 9 (the
#: finest level of ``FemOptions.refinement`` = 6 is 7). A merge moves a mesh
#: line (of a ring's block, its contacts) onto the physical feature of its run
#: (``keep`` of :func:`graded_lines`); the geometry moves only where two
#: physical features are this close, by at most 1e-9 of max(1 m, x): a rod
#: end and an interface (a rod tip on 1000 / 30 Ohm m moves by 1.9e-6 per
#: nanometre), a ring axis and an interface (1e-9 per nanometre), two
#: interfaces (the layer between them vanishes). No merge moves an interface
#: relative to a ring wire, which can be that sensitive: a wire touching a
#: 1000 times more conductive layer moves by 2.6e-3 per nanometre of the
#: interface (the fem, confirmed by the boundary elements at 10 um, evidence
#: ``fw4r3a/e04``).
_SAME_LINE = 1e-9

#: Snap tolerance of the interface cut, relative to the larger of 1 m and the
#: interface's depth: round-off of nodes that lie on the cut line.
_CUT_ROUND_OFF = 1e-12


def graded_lines(
    features: list[tuple[float, float]],
    growth: float,
    keep: "set[float] | frozenset[float]" = frozenset(),
) -> np.ndarray:
    """Node coordinates through every feature, graded away from them.

    Parameters
    ----------
    features
        ``(x, h)`` pairs: a coordinate in m that must be a node and the
        node spacing in m wanted there (``math.inf`` for a line without
        its own size, e.g. a domain boundary). At least one size must be
        finite. Duplicates keep the smallest size; so do coordinates within
        :data:`_SAME_LINE` of the line of a run, which become that line.
    growth
        Growth $g > 0$ of the spacing per unit distance: the spacing at
        $x$ is $h(x) = \\min_f (h_f + g\\,|x - x_f|)$.
    keep
        Coordinates of physical features (interfaces, rod ends and radii,
        the soil surface, the truncation box): a run that contains one has
        its line there (the first of them), so a merge moves a mesh-only
        feature (a block or contact line) and not the geometry (audit
        W4R3A-01: a ring wire touching a 1000 times more conductive layer
        moves by 2.6e-3 per nanometre of the interface). Without one the
        line is the first (smallest) coordinate of the run.

    Returns
    -------
    numpy.ndarray
        Sorted node coordinates in m. Every feature coordinate is a node
        (bit for bit, or within :data:`_SAME_LINE` for one merged into its
        neighbour); between two neighbouring features the nodes follow
        $h(x)$, with at least one interval.
    """
    if growth <= 0.0:
        raise ValueError(f"growth must be positive, got {growth!r}.")
    merged: dict[float, float] = {}
    for x, h in features:
        x, h = float(x), float(h)
        merged[x] = min(merged.get(x, math.inf), h)
    # One line for coordinates computed two ways (a block's centre line at
    # 0.30000000000000004 m and an interface at 0.3 m; a rod tip at 0.1 + 0.2)
    # and for features a few picometres apart: two lines that close were a
    # row of cells of zero height, which made the stiffness matrix singular
    # or the resistance wrong by a factor of ten (audit W4R3A-01).
    xs_all = sorted(merged)
    xs_list: list[float] = []
    hs_list: list[float] = []
    pinned: list[bool] = []
    for x in xs_all:
        if xs_list and x - xs_list[-1] <= _SAME_LINE * max(1.0, abs(xs_list[-1])):
            hs_list[-1] = min(hs_list[-1], merged[x])
            if x in keep and not pinned[-1]:
                xs_list[-1], pinned[-1] = x, True
            continue
        xs_list.append(x)
        hs_list.append(merged[x])
        pinned.append(x in keep)
    xs = np.array(xs_list)
    hs = np.array(hs_list)
    finite = np.isfinite(hs)
    if not finite.any():
        raise ValueError("graded_lines needs at least one feature with a finite size.")
    # The size at every feature after the other features' growth: then
    # within an interval only its two end features decide.
    h_at = np.array(
        [float(np.min(hs[finite] + growth * np.abs(xs[finite] - x))) for x in xs]
    )
    g = growth
    out = [float(xs[0])]
    for i in range(len(xs) - 1):
        a, b = float(xs[i]), float(xs[i + 1])
        ha, hb = float(h_at[i]), float(h_at[i + 1])
        # Crossover of the two linear branches of h(x) on [a, b].
        x_star = min(max((hb - ha + g * (a + b)) / (2.0 * g), a), b)
        n_a = math.log((ha + g * (x_star - a)) / ha) / g
        n_b = math.log((hb + g * (b - x_star)) / hb) / g
        total = n_a + n_b
        n = max(1, math.ceil(total - 1e-9))
        for k in range(1, n):
            xi = k * total / n
            if xi <= n_a:
                x = a + ha * math.expm1(g * xi) / g
            else:
                x = b - hb * math.expm1(g * (total - xi)) / g
            out.append(x)
        out.append(b)
    return np.array(out)


def bisect(lines: np.ndarray, levels: int) -> np.ndarray:
    """Insert the midpoint of every interval, ``levels`` times (nested)."""
    for _ in range(int(levels)):
        mid = 0.5 * (lines[:-1] + lines[1:])
        refined = np.empty(2 * lines.size - 1)
        refined[0::2] = lines
        refined[1::2] = mid
        lines = refined
    return lines


# ---------------------------------------------------------------------
# Mesh description
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RodGeometry:
    """A rod on the axis in the meridian plane.

    Attributes
    ----------
    radius
        Wire radius $a$ in m (outer shell radius for an encased rod).
    z_top, z_bottom
        Depth of the head and of the tip in m.
    """

    radius: float
    z_top: float
    z_bottom: float


@dataclass(frozen=True)
class RingGeometry:
    """A ring around the axis in the meridian plane.

    Attributes
    ----------
    radius
        Ring radius $r_0$ in m.
    depth
        Depth $t$ of the wire axis in m.
    wire_radius
        Wire radius $a$ in m (outer shell radius for an encased ring).
    half_width
        Half-width $c$ in m of the square O-grid block around the wire
        (chosen by the caller so that the block stays clear of the soil
        surface, the axis, the other electrodes and of every interface
        farther than $1.05\\,a$ from the wire axis).
    near_interface
        Whether a layer interface lies within $1.05\\,a$ of the wire axis
        (it is cut into the block); the block then has
        ``MeshParameters.near_interface_intervals`` per side instead of
        ``box_intervals``.
    contacts
        Angles in rad, $\\operatorname{atan2}(z - t, s - r_0)$, of the
        points of the wire's circle where a near interface cuts it at a
        narrow angle (two per interface, 0.7 a or farther from the axis) or
        comes closest to it (one, at 1 ... 1.05 a); the O-grid is graded
        towards them (audit W4R2A-02).
    contact_size
        Arc length in m of the O-grid's finest cells at the contacts, at
        level 0 (each level halves it); ``0.0``: no grading.
    """

    radius: float
    depth: float
    wire_radius: float
    half_width: float
    near_interface: bool = False
    contacts: tuple[float, ...] = ()
    contact_size: float = 0.0


@dataclass(frozen=True)
class MeshParameters:
    """Resolution of the level-0 mesh (dimensionless).

    Attributes
    ----------
    growth
        Growth $g$ of the line spacing per unit distance (ratio $1 + g$).
    end_size
        Line spacing at the surface and the end faces of a rod, in units of
        its radius. Small, because the edges of the flat end faces are
        singular (a re-entrant corner of the conductor): with spacing
        $0.002\\,a$ the observed convergence stays of second order even for
        a rod tip lying on a layer interface.
    box_half_width
        Default half-width of the O-grid block of a ring, in wire radii.
    box_intervals
        Intervals of the tensor grid per side of an O-grid block at level
        0 (even, so that the horizontal diameter of the wire is a line).
    near_interface_intervals
        Intervals of the tensor grid per side of the block of a ring wire
        within $1.05\\,a$ of a layer interface, at level 0. The gap between
        the wire and the interface, and the narrow wedge where the
        interface cuts the wire near its edge, make the field vary over
        $\\sqrt{2a\\,|d - a|}$ along the wire's cross section ($d$: distance
        of the interface from the wire axis; 0.14 a at $d = 1.01\\,a$). With
        ``box_intervals`` the levels converged at first order and not
        monotonically there (up to 6e-3 off, audit W4A-01); 64 intervals
        fixed that up to a contrast of 33. With the grading towards the
        contacts (``contact_size``) 32 are as accurate as 64 and the mesh has
        fewer nodes than the 64 intervals without it (audit W4R2A-02,
        evidence ``fw4r2a/e01``).
    contact_size
        Arc length of the finest O-grid cells of such a block at the points
        where the interface cuts the wire at a narrow angle (0.7 a or farther
        from its axis, at 45.6 deg or less) or comes closest to it, at
        level 0, in wire radii; the first radial layers start with the same
        thickness. Where an interface cuts the wire the field is singular
        (an edge of the conductor on a material interface, strongly so in
        a narrow wedge), and a wire touching a much more conductive layer
        passes its current through a contact zone that shrinks with the
        contrast; without this grading the levels converged at about first
        order beyond a contrast of 33 and the estimate fell below the error
        (audit W4R2A-02: −1.6e-2 with an estimate of 1.1e-2 for a ring
        touching a 1000 times more conductive layer). The angular and the
        radial grading are both needed (each alone: −1.4e-2 and −3.4e-2
        there; evidence ``fw4r2a/e01``).
    contact_growth
        Growth of the arc length and of the layer thickness away from a
        contact (ratio $1 + g$ between neighbours), up to the block's
        uniform size.
    """

    growth: float = 0.2
    end_size: float = 0.002
    box_half_width: float = 3.0
    box_intervals: int = 8
    near_interface_intervals: int = 32
    contact_size: float = 1e-3
    contact_growth: float = 0.3


@dataclass
class AxiMesh:
    """Triangular mesh of the meridian half-plane.

    Attributes
    ----------
    nodes
        Node coordinates $(s, z)$ in m, shape ``(N, 2)``.
    triangles
        Node indices per triangle, shape ``(T, 3)``.
    layer
        Soil layer of every triangle (0-based), shape ``(T,)``.
    port
        Port (electrode) index of every node on an electrode surface,
        ``-1`` elsewhere, shape ``(N,)``.
    port_edges
        Boundary edges of every port's surface in the soil, one
        ``(E_k, 2)`` array per port (the head face of a rod at the soil
        surface is not one: it borders the air).
    outer_edges
        Edges on the truncation box ($s = R$ or $z = R$), shape ``(E, 2)``.
    s_lines, z_lines
        The tensor lines in m.
    r_far
        Size $R$ of the truncation box in m.
    interfaces
        The mesh lines in m of the layer interfaces inside the box, each
        within :data:`_SAME_LINE` of its interface's depth (the line it was
        merged into and cut at; ``layer`` counts the lines above a
        triangle's centroid).
    stats
        Mesh size (nodes, triangles, lines, level).
    """

    nodes: np.ndarray
    triangles: np.ndarray
    layer: np.ndarray
    port: np.ndarray
    port_edges: list[np.ndarray]
    outer_edges: np.ndarray
    s_lines: np.ndarray
    z_lines: np.ndarray
    r_far: float
    interfaces: tuple[float, ...] = ()
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# Line sets
# ---------------------------------------------------------------------


def _box_points(centre: float, c: float, n: int) -> list[float]:
    """The ``n + 1`` equidistant tensor lines of an O-grid block side."""
    step = 2.0 * c / n
    return [centre - c + k * step for k in range(n)] + [centre + c]


def block_intervals(ring: RingGeometry, params: MeshParameters) -> int:
    """Level-0 intervals of the tensor grid per side of a ring's block."""
    if ring.near_interface:
        return params.near_interface_intervals
    return params.box_intervals


def _contact_angles(ring: RingGeometry, n: int, growth: float) -> list[float]:
    """Ray angles of a ring's O-grid graded towards its contacts (level 0).

    From every contact angle the arc between neighbouring rays starts at
    ``ring.contact_size`` and grows by ``1 + growth`` until it reaches the
    block's uniform arc, $2\\pi a / (4n)$ for $n$ intervals per side (or a
    quarter turn is covered).
    """
    if not (ring.contact_size > 0.0 and ring.contacts):
        return []
    a = ring.wire_radius
    arc_uniform = 2.0 * math.pi * a / (4 * n)
    out: list[float] = []
    for phi in ring.contacts:
        out.append(phi)
        for sign in (1.0, -1.0):
            theta, arc = 0.0, ring.contact_size
            while arc < arc_uniform and theta < 0.25 * math.pi:
                theta += arc / a
                out.append(phi + sign * theta)
                arc *= 1.0 + growth
    return out


def _contact_lines(
    ring: RingGeometry, n: int, growth: float
) -> tuple[list[float], list[float]]:
    """Tensor lines through the block boundary where the graded rays meet it.

    The O-grid's rays run from the wire axis to the nodes of the block
    boundary, which are the crossings of the tensor lines; a ray at angle
    $\\phi$ meets the top or bottom side at an $s$ line and the left or
    right side at a $z$ line. Returns ``(s_lines, z_lines)`` in m.
    """
    s_out: list[float] = []
    z_out: list[float] = []
    c = ring.half_width
    for ang in _contact_angles(ring, n, growth):
        us, uz = math.cos(ang), math.sin(ang)
        lam = c / max(abs(us), abs(uz))
        if abs(uz) >= abs(us):
            s_out.append(ring.radius + lam * us)
        else:
            z_out.append(ring.depth + lam * uz)
    return s_out, z_out


def _add_contact_lines(
    features: list[tuple[float, float]], lines: list[float], tol: float
) -> None:
    """Append ``lines`` as features without a size of their own.

    A line within ``tol`` of a line already present is dropped, so that no
    sliver interval appears between them.
    """
    if not lines:
        return
    present = np.array(sorted(x for x, _ in features))
    for x in sorted(lines):
        k = int(np.searchsorted(present, x))
        near = [present[j] for j in (k - 1, k) if 0 <= j < present.size]
        if any(abs(x - y) <= tol for y in near):
            continue
        features.append((x, math.inf))
        present = np.insert(present, k, x)


def _layer_fractions(
    ring: RingGeometry,
    params: "MeshParameters",
    level: int,
    perimeter: int,
    psi_max: float,
) -> np.ndarray:
    """Radial positions $u \\in [0, 1]$ of a ring's O-grid layers.

    A layer at $u$ lies at $\\rho = a\\,(d/a)^u$ on the ray to a block
    boundary node at distance $d$ ($u = 0$: the wire, $u = 1$: the block
    boundary; ``psi_max`` $= \\ln(d_{\\max}/a)$). Without contacts the layers
    are uniform in $u$, as many as make the cells about square at the block
    corners for ``perimeter`` rays. With contacts the level-0 layers start
    with the thickness ``ring.contact_size`` at the wire, grow by
    ``1 + params.contact_growth`` to the uniform spacing of the block's
    $4n$ uniform rays and stay uniform beyond; level $\\ell$ bisects them
    $\\ell$ times, like the tensor lines (audit W4R2A-02: layers recomputed
    per level, not bisected, kept the touching ring at contrast 1000 below
    second order).
    """
    if not (ring.contact_size > 0.0 and ring.contacts):
        n_layers = max(1, math.ceil(psi_max / (2.0 * math.pi / perimeter)))
        return np.arange(n_layers + 1) / n_layers
    n_uniform = 4 * block_intervals(ring, params)
    n_u = max(1, math.ceil(psi_max / (2.0 * math.pi / n_uniform)))
    du_max = 1.0 / n_u
    du = ring.contact_size / (ring.wire_radius * psi_max)
    u = [0.0]
    while du < du_max and u[-1] + du < 1.0:
        u.append(u[-1] + du)
        du *= 1.0 + params.contact_growth
    rest = 1.0 - u[-1]
    n_rest = max(1, math.ceil(rest / du_max - 1e-9))
    u += [u[-1] + rest * k / n_rest for k in range(1, n_rest + 1)]
    u[-1] = 1.0
    return bisect(np.asarray(u), level)


def mesh_lines(
    rods: list[RodGeometry],
    rings: list[RingGeometry],
    interfaces: list[float],
    r_far: float,
    params: MeshParameters,
    level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The tensor lines ``(s_lines, z_lines)`` in m of one refinement level."""
    inf = math.inf
    g = params.growth
    s_feat: list[tuple[float, float]] = [(0.0, inf), (r_far, inf)]
    z_feat: list[tuple[float, float]] = [(0.0, inf), (r_far, inf)]
    # (z range, size) of every electrode for the size of nearby interfaces
    bodies: list[tuple[float, float, float]] = []
    for rod in rods:
        h = params.end_size * rod.radius
        s_feat.append((rod.radius, h))
        z_feat += [(rod.z_top, h), (rod.z_bottom, h)]
        bodies.append((rod.z_top, rod.z_bottom, h))
    for ring in rings:
        n = block_intervals(ring, params)
        step = 2.0 * ring.half_width / n
        s_feat += [(x, step) for x in _box_points(ring.radius, ring.half_width, n)]
        z_feat += [(x, step) for x in _box_points(ring.depth, ring.half_width, n)]
        s_extra, z_extra = _contact_lines(ring, n, params.contact_growth)
        tol = 0.25 * ring.contact_size * ring.half_width / ring.wire_radius
        _add_contact_lines(s_feat, s_extra, tol)
        _add_contact_lines(z_feat, z_extra, tol)
        bodies.append(
            (ring.depth - ring.half_width, ring.depth + ring.half_width, step)
        )
    for h in interfaces:
        sizes = [size + g * max(0.0, lo - h, h - hi) for lo, hi, size in bodies]
        z_feat.append((h, min(sizes) if sizes else inf))
    # physical features keep their place in a merge (audit W4R3A-01)
    s_keep = {0.0, float(r_far)} | {float(rod.radius) for rod in rods}
    z_keep = {0.0, float(r_far)} | {float(h) for h in interfaces}
    z_keep |= {float(z) for rod in rods for z in (rod.z_top, rod.z_bottom)}
    s_lines = bisect(graded_lines(s_feat, g, s_keep), level)
    z_lines = bisect(graded_lines(z_feat, g, z_keep), level)
    return s_lines, z_lines


def _index(lines: np.ndarray, value: float) -> int:
    """Index of the line at ``value`` (it is a feature, hence exact)."""
    k = int(np.argmin(np.abs(lines - value)))
    if abs(float(lines[k]) - value) > _SAME_LINE * max(1.0, abs(value)):
        raise AssertionError(f"mesh line {value!r} missing")  # pragma: no cover
    return k


# ---------------------------------------------------------------------
# Interface cut of the O-grid elements
# ---------------------------------------------------------------------


def cut_at_depth(
    nodes: np.ndarray,
    triangles: np.ndarray,
    port: np.ndarray,
    z_cut: float,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split every triangle crossed by the plane $z = z_c$.

    The split is conforming (an edge crossed by the plane gets one node,
    shared by the two triangles of the edge); nodes within ``tol`` of the
    plane are moved onto it first. A node inserted on an edge whose two
    end nodes belong to the same port lies on that electrode's surface
    and joins the port.

    Parameters
    ----------
    nodes, triangles, port
        The mesh arrays (see :class:`AxiMesh`).
    z_cut
        Depth of the plane in m: a tensor line (:func:`build_mesh`), so the
        tensor nodes on it lie on it exactly and the others are at least
        a line spacing away.
    tol
        Snap tolerance in m (round-off).

    Returns
    -------
    nodes, triangles, port
        The updated arrays; existing node indices are kept, new nodes are
        appended.
    """
    nodes = np.array(nodes, dtype=float, copy=True)
    close = np.abs(nodes[:, 1] - z_cut) <= tol
    nodes[close, 1] = z_cut
    d = nodes[:, 1] - z_cut
    dv = d[triangles]
    crossed = (dv > 0.0).any(axis=1) & (dv < 0.0).any(axis=1)
    if not crossed.any():
        return nodes, triangles, port
    n_base = nodes.shape[0]
    extra: list[tuple[float, float]] = []
    edge_node: dict[tuple[int, int], int] = {}

    def split(ia: int, ib: int) -> int:
        key = (ia, ib) if ia < ib else (ib, ia)
        hit = edge_node.get(key)
        if hit is not None:
            return hit
        ka, kb = key
        frac = d[ka] / (d[ka] - d[kb])
        s_new = nodes[ka, 0] + frac * (nodes[kb, 0] - nodes[ka, 0])
        index = n_base + len(extra)
        extra.append((float(s_new), float(z_cut)))
        edge_node[key] = index
        return index

    new: list[tuple[int, int, int]] = []
    for t_idx in np.nonzero(crossed)[0]:
        tri = [int(v) for v in triangles[t_idx]]
        dd = d[tri]
        on = np.nonzero(dd == 0.0)[0]
        if on.size == 1:
            i0 = int(on[0])
            i1, i2 = (i0 + 1) % 3, (i0 + 2) % 3
            p = split(tri[i1], tri[i2])
            new += [(tri[i0], tri[i1], p), (tri[i0], p, tri[i2])]
        else:
            positive = dd > 0.0
            lone = int(np.nonzero(positive if positive.sum() == 1 else ~positive)[0][0])
            i1, i2 = (lone + 1) % 3, (lone + 2) % 3
            p1 = split(tri[lone], tri[i1])
            p2 = split(tri[lone], tri[i2])
            new += [(tri[lone], p1, p2), (p1, tri[i1], tri[i2]), (p1, tri[i2], p2)]
    nodes = np.vstack([nodes, np.array(extra, dtype=float)])
    triangles = np.vstack([triangles[~crossed], np.array(new, dtype=int)])
    added = -np.ones(len(extra), dtype=int)
    for (ka, kb), index in edge_node.items():
        if port[ka] >= 0 and port[ka] == port[kb]:
            added[index - n_base] = port[ka]
    return nodes, triangles, np.concatenate([port, added])


# ---------------------------------------------------------------------
# Mesh assembly
# ---------------------------------------------------------------------


def _boundary_edges(triangles: np.ndarray) -> np.ndarray:
    """Edges that belong to exactly one triangle, shape ``(E, 2)``."""
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _o_grid_size(
    s_lines: np.ndarray,
    z_lines: np.ndarray,
    ring: RingGeometry,
    params: "MeshParameters | None" = None,
    level: int = 0,
) -> tuple[int, int]:
    """Perimeter nodes $P$ and radial layers of a ring's O-grid (as built)."""
    c = ring.half_width
    n_s = _index(s_lines, ring.radius + c) - _index(s_lines, ring.radius - c)
    n_z = _index(z_lines, ring.depth + c) - _index(z_lines, ring.depth - c)
    perimeter = 2 * (n_s + n_z)
    # build_mesh: ceil(ln(max distance / a) / (2 pi / P)), the largest
    # distance being the block's corner; the tiny offset keeps the count
    # an upper bound if round-off of the node coordinates crosses an integer.
    if ring.contact_size > 0.0 and ring.contacts:
        psi = math.log(math.sqrt(2.0) * c / ring.wire_radius) * (1.0 + 1e-12)
        u = _layer_fractions(ring, params or MeshParameters(), level, perimeter, psi)
        return perimeter, int(u.size - 1) + 1
    x = math.log(math.sqrt(2.0) * c / ring.wire_radius) / (2.0 * math.pi / perimeter)
    return perimeter, max(1, math.ceil(x + 1e-9))


def _o_grid_centre_z(ring: RingGeometry, cut_lines: list[float]) -> float:
    """Depth in m of the centre of a ring's O-grid.

    The wire's axis depth, or the mesh line of an interface within
    :data:`_SAME_LINE` of it: then the axis is merged with the interface
    like two features (it moves by at most that much), and the O-grid's
    rays at the axis height lie on the cut line. Tilted by the merge they
    put their nodes up to :data:`_SAME_LINE` off the plane, partly within
    the cut's snap tolerance, and the cut split them into slivers (audit
    W4R3A-01).
    """
    t = ring.depth
    tol = _SAME_LINE * max(1.0, abs(t)) * (1.0 + 1e-6)  # the box line's round-off
    near = [line for line in cut_lines if 0.0 < abs(line - t) <= tol]
    return min(near, key=lambda line: abs(line - t)) if near else t


def predicted_nodes(
    s_lines: np.ndarray,
    z_lines: np.ndarray,
    rings: list[RingGeometry],
    level: int,
    interfaces: list[float] | tuple[float, ...] = (),
    params: "MeshParameters | None" = None,
) -> int:
    """Node count of :func:`build_mesh` estimated before it is built.

    An estimate that errs high: the tensor nodes (including those inside
    the rods and the O-grid blocks, which the mesh drops), the O-grid
    nodes of every ring as :func:`build_mesh` lays them out, and for every
    interface cut into a block the most nodes a straight cut can add
    ($2P + 4n + 2$ for $P$ perimeter nodes and $n$ layers: each layer
    polygon is convex and crossed at most twice, each ray at most once,
    and every quadrilateral passed adds at most one diagonal). Up to audit
    W4A-10 the O-grid count grew like $2^\\ell$ instead of $4^\\ell$, and
    the estimate was 1.5 ... 2.3 % low for a single ring.

    Parameters
    ----------
    s_lines, z_lines
        The tensor lines of the level (:func:`mesh_lines`).
    rings
        The rings, with their block half-widths.
    level
        Refinement level (kept for the signature; the lines carry it).
    interfaces
        Depths in m of the layer interfaces inside the box.
    """
    n = s_lines.size * z_lines.size
    for ring in rings:
        perimeter, layers = _o_grid_size(s_lines, z_lines, ring, params, level)
        n += perimeter * layers
        cuts = sum(1 for h in interfaces if abs(ring.depth - h) < ring.half_width)
        n += cuts * (2 * perimeter + 4 * layers + 2)
    return int(n)


def build_mesh(
    rods: list[RodGeometry],
    rings: list[RingGeometry],
    interfaces: list[float],
    r_far: float,
    params: MeshParameters,
    level: int,
    lines: tuple[np.ndarray, np.ndarray] | None = None,
) -> AxiMesh:
    """Mesh the soil around coaxial rods and rings (see the module docstring).

    Ports are numbered rods first, then rings, in the order given.

    Parameters
    ----------
    rods, rings
        The electrodes; they must not overlap (checked by the caller).
    interfaces
        Depths in m of the layer interfaces (canonical, inside the box).
    r_far
        Size of the truncation box in m.
    params, level
        Level-0 resolution and refinement level.
    lines
        Tensor lines from :func:`mesh_lines`, if already computed.
    """
    if lines is None:
        lines = mesh_lines(rods, rings, interfaces, r_far, params, level)
    s_lines, z_lines = lines
    ns, nz = s_lines.size, z_lines.size
    keep = np.ones((ns - 1, nz - 1), dtype=bool)
    port = -np.ones(ns * nz, dtype=int)

    def nid(i: int, j: int) -> int:
        return i * nz + j

    for k, rod in enumerate(rods):
        ia = _index(s_lines, rod.radius)
        j0, j1 = _index(z_lines, rod.z_top), _index(z_lines, rod.z_bottom)
        keep[:ia, j0:j1] = False
        for i in range(ia + 1):
            port[nid(i, j0)] = k
            port[nid(i, j1)] = k
        for j in range(j0, j1 + 1):
            port[nid(ia, j)] = k
    # Every interface is cut at its own mesh line: the feature line it was
    # merged into (within _SAME_LINE), on which the tensor nodes lie exactly.
    # Cut at the depth itself, a line within the snap tolerance was moved
    # onto the plane and left zero-area cells (audit W4R3A-01).
    cut_lines = [float(z_lines[_index(z_lines, h)]) for h in interfaces]
    blocks = []
    for ring in rings:
        c = ring.half_width
        i0, i1 = _index(s_lines, ring.radius - c), _index(s_lines, ring.radius + c)
        j0, j1 = _index(z_lines, ring.depth - c), _index(z_lines, ring.depth + c)
        keep[i0:i1, j0:j1] = False
        blocks.append((i0, i1, j0, j1))

    ii, jj = np.nonzero(keep)
    n00 = ii * nz + jj
    n10 = (ii + 1) * nz + jj
    n11 = (ii + 1) * nz + jj + 1
    n01 = ii * nz + jj + 1
    tri_parts = [np.stack([n00, n10, n11], axis=1), np.stack([n00, n11, n01], axis=1)]
    nodes = np.stack(np.meshgrid(s_lines, z_lines, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    extra: list[np.ndarray] = []
    n_next = nodes.shape[0]
    port_parts = [port]
    for k, (ring, (i0, i1, j0, j1)) in enumerate(zip(rings, blocks, strict=True)):
        perimeter = (
            [nid(i, j0) for i in range(i0, i1)]
            + [nid(i1, j) for j in range(j0, j1)]
            + [nid(i, j1) for i in range(i1, i0, -1)]
            + [nid(i0, j) for j in range(j1, j0, -1)]
        )
        perimeter = np.array(perimeter, dtype=int)
        centre = np.array([ring.radius, _o_grid_centre_z(ring, cut_lines)])
        offset = nodes[perimeter] - centre
        dist = np.hypot(offset[:, 0], offset[:, 1])
        unit = offset / dist[:, None]
        a = ring.wire_radius
        # Radial layers uniform in ln(rho), about square elements.
        u = _layer_fractions(
            ring, params, level, perimeter.size, math.log(float(dist.max()) / a)
        )
        n_layers = u.size - 1
        layers = [perimeter]
        for m in range(n_layers - 1, -1, -1):
            rho = a * (dist / a) ** u[m]
            ids = np.arange(n_next, n_next + perimeter.size)
            extra.append(centre + rho[:, None] * unit)
            n_next += perimeter.size
            layers.append(ids)
            port_parts.append(
                np.full(perimeter.size, len(rods) + k if m == 0 else -1, dtype=int)
            )
        for outer, inner in zip(layers[:-1], layers[1:], strict=True):
            o1 = np.roll(outer, -1)
            i1n = np.roll(inner, -1)
            tri_parts.append(np.stack([outer, o1, i1n], axis=1))
            tri_parts.append(np.stack([outer, i1n, inner], axis=1))
    if extra:
        nodes = np.vstack([nodes, *extra])
    port = np.concatenate(port_parts)
    triangles = np.concatenate(tri_parts)
    for h, line in zip(interfaces, cut_lines, strict=True):
        if any(abs(r.depth - h) < r.half_width for r in rings):
            nodes, triangles, port = cut_at_depth(
                nodes,
                triangles,
                port,
                line,
                tol=_CUT_ROUND_OFF * max(1.0, line),
            )

    used = np.zeros(nodes.shape[0], dtype=bool)
    used[triangles.ravel()] = True
    renumber = -np.ones(nodes.shape[0], dtype=int)
    renumber[used] = np.arange(int(used.sum()))
    nodes = nodes[used]
    triangles = renumber[triangles]
    port = port[used]

    # Every triangle is built with positive area in the (s, z) plane (the
    # tensor cells, the O-grid layers traversed counter-clockwise, the cut
    # keeping its parent's orientation). A triangle of zero area made the
    # factor singular or the resistance wrong without notice (audits
    # W4R2A, W4R3A-01), a negative one would overlap its neighbours: both
    # are a defect of this mesher, never a result.
    v = nodes[triangles]
    area2 = (v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1]) - (
        v[:, 2, 0] - v[:, 0, 0]
    ) * (v[:, 1, 1] - v[:, 0, 1])
    bad = ~(area2 > 0.0)
    if bad.any():
        s_bad, z_bad = v[np.argmax(bad)].mean(axis=0)
        raise RuntimeError(
            "Backend 'fem': internal error of the axisymmetric mesher: "
            f"{int(bad.sum())} triangles have zero or negative area (the first "
            f"at s = {s_bad:.9g} m, z = {z_bad:.12g} m), which would make the "
            "solve singular or its result wrong without notice. This is a "
            "defect of groundfield; please report the world."
        )

    boundary = _boundary_edges(triangles)
    pa, pb = port[boundary[:, 0]], port[boundary[:, 1]]
    za, zb = nodes[boundary[:, 0], 1], nodes[boundary[:, 1], 1]
    n_ports = len(rods) + len(rings)
    port_edges = [
        boundary[(pa == k) & (pb == k) & ~((za == 0.0) & (zb == 0.0))]
        for k in range(n_ports)
    ]
    sa, sb = nodes[boundary[:, 0], 0], nodes[boundary[:, 1], 0]
    outer = boundary[((sa == r_far) & (sb == r_far)) | ((za == r_far) & (zb == r_far))]

    # The layers are counted from the cut lines: a centroid between an
    # interface's depth and its line (at most _SAME_LINE apart) is not in
    # a layer of its own.
    bounds = np.asarray(sorted(cut_lines), dtype=float)
    centroid_z = nodes[triangles, 1].mean(axis=1)
    layer = np.searchsorted(bounds, centroid_z, side="right")
    return AxiMesh(
        nodes=nodes,
        triangles=triangles,
        layer=layer,
        port=port,
        port_edges=port_edges,
        outer_edges=outer,
        s_lines=s_lines,
        z_lines=z_lines,
        r_far=float(r_far),
        interfaces=tuple(float(h) for h in bounds),
        stats={
            "level": int(level),
            "n_nodes": int(nodes.shape[0]),
            "n_triangles": int(triangles.shape[0]),
            "n_s_lines": int(ns),
            "n_z_lines": int(nz),
            "r_far": float(r_far),
        },
    )
