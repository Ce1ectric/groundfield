"""Tetrahedral mesh of the soil for the three-dimensional ``fem`` model (private).

The mesh starts from a coarse **isotropic** grid: $8 \\times 8 \\times 4$
cubes on the truncation box $[c_x - R, c_x + R] \\times [c_y - R, c_y + R]
\\times [0, R]$, each split into six Kuhn tetrahedra (``scikit-fem``'s
``MeshTet.init_tensor``). It is refined by a deterministic conforming
**longest-edge bisection** (:class:`Bisector`, Rivara) until every
tetrahedron is no longer than $1.5\\,h(x)$,

$$
h(x) = h_\\mathrm{w} + g\\,\\max\\bigl(0,\\ d(x) - R_\\eta\\bigr),
$$

$d$ the distance from the nearest wire axis, $R_\\eta$ the radius of the
smoothed line sources (:mod:`groundfield.solver.fem._wires3d`),
$h_\\mathrm{w} = R_\\eta/2$ and the growth $g$; an element that a layer
interface or a concrete prism face cuts is refined further near the
electrodes. Longest-edge bisection of Kuhn tetrahedra keeps the shapes in a
few similarity classes (aspect ratio at most 2 here), so the multigrid
solver converges in a few dozen iterations; a graded *tensor* grid instead
carries its fine lines (the layer interfaces, the electrode planes) to the
box as slabs thousands of times wider than thick, on which it stalls (345
conjugate-gradient iterations for a 5 m slab, no convergence for 0.3 m;
evidence of stream w4fem3d).

No interface is a mesh plane: the elements an interface cuts are immersed
elements (:mod:`groundfield.solver.fem._assembly3d`). A node very close to
an interface is moved onto it (:func:`snap_to_planes`) so no cut leaves a
sliver. The finer mesh of the extrapolation is the uniform (red)
refinement of the snapped coarser one, so the two spaces are nested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__: list[str] = []

#: Index pairs of the six edges of a tetrahedron.
TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

#: A tetrahedron is refined while its longest edge exceeds this multiple
#: of the target size.
SIZE_FACTOR = 1.5

#: A node closer to a material plane than this fraction of its shortest
#: edge is moved onto the plane (:func:`snap_to_planes`).
SNAP_RATIO = 0.3


@dataclass(frozen=True)
class Prism:
    """An axis-aligned box of concrete around an encased wire (Q40).

    Attributes
    ----------
    lower, upper
        Opposite corners in m, shape ``(3,)``.
    """

    lower: np.ndarray
    upper: np.ndarray


class Bisector:
    """Conforming longest-edge bisection of a tetrahedral mesh (Rivara).

    A marked tetrahedron is cut at the midpoint of its longest edge into
    two; every tetrahedron that then has a cut edge (a hanging node) is
    bisected in turn, at its own longest edge, until the mesh is conforming
    again. Ties between equally long edges are broken by the global node
    numbers of the edge, so neighbouring tetrahedra always agree on the
    edge to cut and the result is deterministic. Bisection never moves a
    node, so a plane that is a union of faces stays one.

    Parameters
    ----------
    p
        Node coordinates in m, shape ``(3, N)``.
    t
        Tetrahedra, shape ``(4, T)``.
    """

    _BIG = np.int64(1) << 32

    def __init__(self, p: np.ndarray, t: np.ndarray) -> None:
        self.p = np.ascontiguousarray(p, dtype=float)
        self.t = np.ascontiguousarray(t, dtype=np.int64)
        self._keys = np.zeros(0, dtype=np.int64)
        self._mids = np.zeros(0, dtype=np.int64)

    def _edge_key(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b) * self._BIG + np.maximum(a, b)

    def _midpoint(self, keys: np.ndarray) -> np.ndarray:
        if self._keys.size == 0:
            return np.full(keys.shape, -1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self._keys, keys), self._keys.size - 1)
        return np.where(self._keys[pos] == keys, self._mids[pos], -1)

    def _refinement_edge(self, tt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        best_len = np.full(tt.shape[1], -1.0)
        best_key = np.zeros(tt.shape[1], dtype=np.int64)
        best = np.zeros(tt.shape[1], dtype=np.int64)
        for k, (a, b) in enumerate(TET_EDGES):
            d = self.p[:, tt[a]] - self.p[:, tt[b]]
            length2 = np.einsum("ij,ij->j", d, d)
            key = self._edge_key(tt[a], tt[b])
            better = (length2 > best_len) | ((length2 == best_len) & (key > best_key))
            best_len = np.where(better, length2, best_len)
            best_key = np.where(better, key, best_key)
            best = np.where(better, k, best)
        return best, best_key

    def refine(self, marked: np.ndarray) -> "Bisector":
        """Bisect the ``marked`` tetrahedra (indices) and restore conformity."""
        t = self.t
        mask = np.zeros(t.shape[1], dtype=bool)
        mask[marked] = True
        while mask.any():
            idx = np.nonzero(mask)[0]
            tt = t[:, idx]
            edge, keys = self._refinement_edge(tt)
            mids = self._midpoint(keys)
            new = mids < 0
            if new.any():
                unique, inverse = np.unique(keys[new], return_inverse=True)
                first = np.zeros(unique.size, dtype=np.int64)
                first[inverse[::-1]] = np.nonzero(new)[0][::-1]
                a = tt[np.array(TET_EDGES)[edge[first], 0], first]
                b = tt[np.array(TET_EDGES)[edge[first], 1], first]
                n0 = self.p.shape[1]
                self.p = np.concatenate(
                    [self.p, 0.5 * (self.p[:, a] + self.p[:, b])], axis=1
                )
                values = n0 + np.arange(unique.size, dtype=np.int64)
                mids[new] = values[inverse]
                keys_all = np.concatenate([self._keys, unique])
                order = np.argsort(keys_all, kind="stable")
                self._keys = keys_all[order]
                self._mids = np.concatenate([self._mids, values])[order]
            first_child, second_child = tt.copy(), tt.copy()
            for k, (a, b) in enumerate(TET_EDGES):
                sel = edge == k
                first_child[b, sel] = mids[sel]
                second_child[a, sel] = mids[sel]
            t[:, idx] = first_child
            t = np.concatenate([t, second_child], axis=1)
            mask = np.zeros(t.shape[1], dtype=bool)
            for a, b in TET_EDGES:
                mask |= self._midpoint(self._edge_key(t[a], t[b])) >= 0
        self.t = t
        return self


def background(center: np.ndarray, r_far: float, cells: int = 4):
    """Coarse isotropic grid of Kuhn tetrahedra on the truncation box.

    ``2 cells x 2 cells x cells`` cubes of side ``r_far / cells`` on
    $[c_x - R, c_x + R] \\times [c_y - R, c_y + R] \\times [0, R]$, every cube
    split into six tetrahedra (``scikit-fem``'s ``MeshTet.init_tensor``).
    No interface is a mesh plane (layers enter as immersed elements,
    :mod:`~groundfield.solver.fem._assembly3d`), so the mesh stays isotropic: a
    tensor grid through the interfaces would carry slabs of the layer
    thickness to the box, thousands of times wider than thick, on which the
    multigrid solver stalls (stream w4fem3d: 345 conjugate-gradient
    iterations for a 5 m slab, no convergence for a 0.3 m one).
    """
    from skfem import MeshTet

    x = np.linspace(center[0] - r_far, center[0] + r_far, 2 * cells + 1)
    y = np.linspace(center[1] - r_far, center[1] + r_far, 2 * cells + 1)
    z = np.linspace(0.0, r_far, cells + 1)
    mesh = MeshTet.init_tensor(x, y, z)
    return Bisector(mesh.p, mesh.t)


def longest_edges(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Longest edge in m of every tetrahedron."""
    e2 = np.zeros(t.shape[1])
    for a, b in TET_EDGES:
        e2 = np.maximum(e2, np.sum((p[:, t[a]] - p[:, t[b]]) ** 2, axis=0))
    return np.sqrt(e2)


def refine_towards(
    mesh: Bisector,
    samples: np.ndarray,
    h_w: float,
    growth: float,
    r_core: float,
    interfaces: list[float] = (),
    interface_ratio: float = 0.1,
    prisms: tuple["Prism", ...] = (),
    prism_size: float | None = None,
) -> Bisector:
    """Bisect until every tetrahedron meets the target size $h(x)$.

    Parameters
    ----------
    mesh
        The mesh (:class:`Bisector`), refined in place.
    samples
        Points in m along every wire axis, no farther apart than ``h_w``.
    h_w
        Target size in m within ``r_core`` of a wire axis.
    growth
        Growth $g$ of the target size per metre beyond ``r_core``.
    r_core
        Radius in m of the zone of size ``h_w`` (the source radius).
    interfaces
        Layer interface depths in m: a tetrahedron that a layer interface
        cuts is no larger than ``interface_ratio`` times its distance from
        the nearest wire (and not below ``h_w``), so the laminate
        conductivity of the cut elements (:func:`layer_fractions`) is
        confined to a thin band around the interface near the electrodes.
    prisms, prism_size
        Concrete prisms (user decision Q40): a tetrahedron that a prism
        face cuts is no larger than ``prism_size``.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(samples)
    for _ in range(400):
        p, t = mesh.p, mesh.t
        hmax = longest_edges(p, t)
        d, _ = tree.query(p[:, t].mean(axis=1).T)
        target = h_w + growth * np.maximum(d - 0.5 * hmax - r_core, 0.0)
        if len(interfaces):
            z = p[2, t]
            z_lo, z_hi = z.min(axis=0), z.max(axis=0)
            band = np.maximum(h_w, interface_ratio * d)
            for h in interfaces:
                cut = (z_lo < h) & (z_hi > h)
                target = np.where(cut, np.minimum(target, band), target)
        if prisms and prism_size is not None:
            cut = prism_cut(p, t, prisms)
            target = np.where(cut, np.minimum(target, prism_size), target)
        marked = np.nonzero(hmax > SIZE_FACTOR * target)[0]
        if marked.size == 0:
            return mesh
        mesh.refine(marked)
    raise RuntimeError(
        "fem: the 3-D mesh refinement did not converge"
    )  # pragma: no cover


def prism_cut(p: np.ndarray, t: np.ndarray, prisms: tuple["Prism", ...]) -> np.ndarray:
    """Whether a face of a concrete prism cuts each tetrahedron."""
    lo, hi = p[:, t].min(axis=1), p[:, t].max(axis=1)
    cut = np.zeros(t.shape[1], dtype=bool)
    for q in prisms:
        overlap = np.all((hi > q.lower[:, None]) & (lo < q.upper[:, None]), axis=0)
        inside = np.all((lo >= q.lower[:, None]) & (hi <= q.upper[:, None]), axis=0)
        cut |= overlap & ~inside
    return cut


def tet_volumes(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Signed volumes in m³ of the tetrahedra."""
    a = p[:, t[1]] - p[:, t[0]]
    b = p[:, t[2]] - p[:, t[0]]
    c = p[:, t[3]] - p[:, t[0]]
    return np.einsum("ij,ij->j", a, np.cross(b.T, c.T).T) / 6.0


def snap_to_planes(
    p: np.ndarray,
    t: np.ndarray,
    planes: list[tuple[int, float, float]],
    fixed: list[np.ndarray],
    ratio: float = SNAP_RATIO,
) -> np.ndarray:
    """Move nodes that lie very close to a material plane onto it.

    A node nearer to a plane (a layer interface, a concrete prism face)
    than ``ratio`` times its shortest edge is moved onto it along the
    plane's axis, unless it is ``fixed`` for that axis (on the soil
    surface, the box or another plane across the same axis) or the move
    would invert an element. Afterwards every element the plane cuts has a
    substantial part on either side: the immersed elements
    (:func:`~._assembly3d.ife_matrices`) lose their accuracy when one part
    is a sliver (evidence of stream w4fem3d: a ring above a conductive
    layer moved between +6e-4 and -5e-3 with the box size before).

    Parameters
    ----------
    p
        Node coordinates in m ``(3, N)``; a modified copy is returned.
    t
        Tetrahedra.
    planes
        ``(axis, value, reach)`` triples: a node moves only if it is also
        nearer than ``reach`` (a quarter of the thinner adjacent layer), so
        the large elements far from the electrodes are not distorted.
    fixed
        Per axis a boolean mask of nodes that must not move along it.
    ratio
        Snap distance in units of each node's shortest edge.
    """
    if not planes:
        return p
    p = p.copy()
    shortest = shortest_edges_at_nodes(p, t)
    fixed = [f.copy() for f in fixed]
    for axis, value, reach in planes:
        gap = p[axis] - value
        near = np.abs(gap) < np.minimum(ratio * shortest, reach)
        move = near & (gap != 0.0) & ~fixed[axis]
        if not move.any():
            fixed[axis] |= gap == 0.0
            continue
        old = p[axis].copy()
        touched = t[:, np.any(move[t], axis=0)]
        v_old = tet_volumes(p, touched)
        p[axis, move] = value
        v_new = tet_volumes(p, touched)
        bad = (np.sign(v_new) != np.sign(v_old)) | (
            np.abs(v_new) < 1e-3 * np.abs(v_old)
        )
        while bad.any():
            nodes = np.unique(touched[:, bad].ravel())
            back = nodes[move[nodes]]
            if back.size == 0:  # pragma: no cover - only moved nodes change volumes
                break
            p[axis, back] = old[back]
            move[back] = False
            v_new = tet_volumes(p, touched)
            bad = (np.sign(v_new) != np.sign(v_old)) | (
                np.abs(v_new) < 1e-3 * np.abs(v_old)
            )
        fixed[axis] |= p[axis] == value
    return p


def below_fraction(z: np.ndarray, h: float) -> np.ndarray:
    """Volume fraction of each tetrahedron above depth ``h`` (where $z < h$).

    Exact for any vertex depths, ties included (the fraction depends on the
    four depths only): one vertex $a$ above the plane gives
    $\\prod_{j\\ne a}(h - z_a)/(z_j - z_a)$, three give one minus the
    corresponding product of the vertex below, and two-two the volume of the
    prism between the plane and the upper edge, split into three
    tetrahedra on a canonical embedding of the four depths. (A closed-form
    B-spline expression cancels catastrophically for equal depths, which
    tetrahedra of a tensor grid always have; stream w4fem3d.)

    Parameters
    ----------
    z
        Vertex depths in m, shape ``(4, ne)``.
    h
        Depth in m.
    """
    z = np.asarray(z, dtype=float)
    ne = z.shape[1]
    h_all = np.broadcast_to(np.asarray(h, dtype=float), (ne,))
    zs = np.sort(z, axis=0)
    upper = zs < h_all[None, :]  # strictly above the plane (smaller depth)
    n_up = upper.sum(axis=0)
    f = np.where(n_up == 4, 1.0, 0.0)
    one = n_up == 1
    if one.any():
        za, hh = zs[0, one], h_all[one]
        f[one] = np.prod((hh - za)[None, :] / (zs[1:, one] - za[None, :]), axis=0)
    three = n_up == 3
    if three.any():
        zb, hh = zs[3, three], h_all[three]
        part = np.prod((zb - hh)[None, :] / (zb[None, :] - zs[:3, three]), axis=0)
        f[three] = np.where(zb == hh, 1.0, 1.0 - part)
    two = np.nonzero(n_up == 2)[0]
    if two.size:
        h = h_all[two]
        za, zb, zc, zd = zs[0, two], zs[1, two], zs[2, two], zs[3, two]
        m = two.size
        a = np.stack([np.zeros(m), np.zeros(m), za], axis=1)
        b = np.stack([np.ones(m), np.zeros(m), zb], axis=1)
        c = np.stack([np.zeros(m), np.ones(m), zc], axis=1)
        d = np.stack([np.zeros(m), np.zeros(m), zd], axis=1)

        def at(u, w, zu, zw):
            frac = (h - zu) / (zw - zu)
            return u + frac[:, None] * (w - u)

        pac, pad = at(a, c, za, zc), at(a, d, za, zd)
        pbc, pbd = at(b, c, zb, zc), at(b, d, zb, zd)

        def vol(p0, p1, p2, p3):
            return (
                np.abs(np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
            )

        wedge = vol(a, pac, pad, pbd) + vol(a, pac, pbd, pbc) + vol(a, pbc, pbd, b)
        f[two] = wedge / vol(a, b, c, d)
    return np.clip(f, 0.0, 1.0)


def edge_count(t: np.ndarray, n_nodes: int) -> int:
    """Number of distinct edges of the tetrahedra ``t`` (4, ne)."""
    pairs = np.concatenate([np.sort(t[[a, b]], axis=0) for a, b in TET_EDGES], axis=1)
    key = pairs[0].astype(np.int64) * int(n_nodes) + pairs[1]
    return int(np.unique(key).size)


def near_elements(
    p: np.ndarray,
    t: np.ndarray,
    samples: np.ndarray,
    reach: float,
    chunk: int = 1_000_000,
) -> np.ndarray:
    """Indices of the tetrahedra within ``reach`` (plus their size) of a wire.

    Computed in chunks, so no array of the size of all elements times their
    vertices is formed.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(samples)
    found = []
    for s0 in range(0, t.shape[1], chunk):
        tt = t[:, s0 : s0 + chunk]
        centroid = (p[:, tt[0]] + p[:, tt[1]] + p[:, tt[2]] + p[:, tt[3]]).T / 4.0
        d, _ = tree.query(centroid)
        found.append(s0 + np.nonzero(d < reach + longest_edges(p, tt))[0])
    return np.concatenate(found)


def shortest_edges_at_nodes(
    p: np.ndarray, t: np.ndarray, chunk: int = 1_000_000
) -> np.ndarray:
    """Length in m of the shortest edge at every node (chunked)."""
    shortest = np.full(p.shape[1], np.inf)
    for s0 in range(0, t.shape[1], chunk):
        tt = t[:, s0 : s0 + chunk]
        for a, b in TET_EDGES:
            length = np.sqrt(np.sum((p[:, tt[a]] - p[:, tt[b]]) ** 2, axis=0))
            np.minimum.at(shortest, tt[a], length)
            np.minimum.at(shortest, tt[b], length)
    return shortest


def box_faces(
    p: np.ndarray, t: np.ndarray, planes: list[tuple[int, float]]
) -> np.ndarray:
    """Tetrahedron faces on the truncation box, node triples ``(3, nf)``.

    A face lies on a box plane when its three nodes have the plane's
    coordinate bit for bit (the grid planes of :func:`background` are
    never moved by bisection).
    """
    found = []
    for drop in range(4):
        tri = t[[k for k in range(4) if k != drop]]
        for axis, value in planes:
            on = np.all(p[axis, tri] == value, axis=0)
            if on.any():
                found.append(tri[:, on])
    if not found:  # pragma: no cover - the box always has faces
        return np.zeros((3, 0), dtype=t.dtype)
    return np.concatenate(found, axis=1)


def predicted_nodes(
    total_length: float, h_w: float, growth: float, r_core: float
) -> int:
    """Node count of the coarser 3-D mesh before it is built (resource guard).

    $N \\approx 12\\,000 + 5\\,L\\,[\\pi r_c^2/h_\\mathrm{w}^3 + 2\\pi/(g^2 h_\\mathrm{w})]$ for
    the wire length $L$: the background plus the tube of size $h_\\mathrm{w}$
    within $r_c$ of the wires and the graded zone around it. Fitted to
    measured meshes (a 3 m rod, a 12.6 m ring, a 28 m station: 3.3 … 5.4
    times the bracket per metre over a background of about 10 000 nodes;
    stream w4fem3d); the exact count is checked again before the finer
    level is built.
    """
    per_length = math.pi * r_core**2 / h_w**3 + 2.0 * math.pi / (growth**2 * h_w)
    return int(12_000 + 5.0 * per_length * total_length)
