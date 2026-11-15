"""Assembly and port solve of the three-dimensional ``fem`` model (private).

Linear tetrahedra. The weak form of $-\\nabla\\cdot(\\sigma\\nabla\\varphi) = f$
with the wires as smoothed line sources
(:mod:`groundfield.solver.fem._wires3d`) is the minimum of

$$
\\tfrac12\\int\\sigma|\\nabla\\varphi|^2\\,\\mathrm dV
+ \\tfrac12\\oint_{\\Gamma_\\infty}\\sigma\\frac{\\mathbf n\\cdot\\mathbf r}{r^2}\\varphi^2\\,\\mathrm dA
+ \\tfrac12\\sum_k\\int_{\\Lambda_k}\\frac{(\\Pi(s) - U_k)^2}{C_\\mathrm{w}(s)}\\,\\mathrm ds ,
$$

$\\Pi(s)$ the bump-weighted disc average of $\\varphi$ at the axial position
$s$ of port $k$, $C_\\mathrm{w}$ the radial resistance per unit length
between the metal and that average. Its stationarity gives the current
density $j = (U_k - \\Pi)/C_\\mathrm{w}$ that each wire leaks and the
port conductance matrix $G$ (the Schur complement onto the port
potentials), symmetric positive definite.

* **Stiffness.** Assembled from the six edge weights of each tetrahedron
  ($K_{ij} = \\sigma|T|\\nabla\\lambda_i\\cdot\\nabla\\lambda_j$, the diagonal
  the negative row sum), in chunks, so the memory is that of the matrix.
* **Far field.** The monopole Dirichlet-to-Neumann (Robin) condition on
  the box faces, as in the axisymmetric model, three-point rule per face.
* **Sources.** The disc averages are point quadratures (a Gauss–Jacobi
  rule in $r^2$ times a uniform rule in angle, points no farther apart
  than ``delta``), axial Gauss points two per cell; ``delta`` is half
  the size of the finer mesh's elements at the wire, so the finite
  elements never resolve the individual points.
* **Solve.** One conjugate-gradient solve per port with a smoothed-
  aggregation algebraic multigrid preconditioner (``pyamg``, set up under a
  fixed seed of numpy's global random state, which is restored afterwards);
  $G$ is a quadratic functional of the solution, so its error is the square
  of the solver's energy error. The iterations grow with the layer
  contrast (16 … 115 up to 33, about 1 260 at 333 at the finer level); at
  1000 they do not converge (audit W4B-03).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from groundfield.solver.fem._mesh3d import TET_EDGES

__all__: list[str] = []

#: Relative residual of the conjugate-gradient port solves.
CG_TOLERANCE = 1e-8

#: Iteration limit of the conjugate-gradient port solves. A rod through an
#: interface of contrast 333 needs about 1 260 at the finer level; at 1000
#: the solves do not converge (audit W4B-03).
CG_MAX_ITERATIONS = 3000

#: Seed of numpy's global random state while the multigrid hierarchy is set
#: up: pyamg draws the start vectors of its spectral-radius estimates from
#: ``np.random``. The user's state is saved and restored around it, so a
#: result is the same whatever the user's seed and the user's stream is not
#: advanced (audit W4B-07).
AMG_SEED = 20261115


class ConvergenceError(RuntimeError):
    """The conjugate gradients of a port solve did not converge.

    Attributes
    ----------
    port
        Index of the port whose solve failed.
    """

    def __init__(self, port: int) -> None:
        super().__init__(
            f"fem: the conjugate gradients did not converge for port {port} in "
            f"{CG_MAX_ITERATIONS} iterations (relative residual {CG_TOLERANCE:g})."
        )
        self.port = port


#: An element that one interface cuts is an immersed element when its
#: extent is at most this multiple of the thinner adjacent layer, a
#: laminate otherwise (far from the electrodes).
IFE_SIZE_RATIO = 4.0


def tet_gradients(p: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Volumes ``(ne,)`` and barycentric gradients ``(ne, 4, 3)``."""
    p0 = p[:, t[0]].T
    jac = np.stack([p[:, t[1]].T - p0, p[:, t[2]].T - p0, p[:, t[3]].T - p0], axis=2)
    vol = np.abs(np.linalg.det(jac)) / 6.0
    inv = np.linalg.inv(jac)
    return vol, np.concatenate([-inv.sum(axis=1, keepdims=True), inv], axis=1)


def ife_matrices(
    x: np.ndarray,
    axis: np.ndarray,
    value: np.ndarray,
    sigma_plus: np.ndarray,
    sigma_minus: np.ndarray,
    vol: np.ndarray,
    return_maps: bool = False,
):
    """Immersed-interface element matrices for one axis-aligned plane each.

    Li, Lin & Wu (2003); Kafafy et al. (2005). On the side where the
    coordinate ``axis`` exceeds ``value`` the shape functions have the
    gradient $\\mathbf g$, on the other side $M\\mathbf g$ with
    $M = I + (\\sigma_+/\\sigma_- - 1)\\,\\mathbf e\\mathbf e^\\mathsf T$ ($\\mathbf e$ the
    plane normal): continuous across the plane, with continuous normal
    flux. $(\\varphi_0, \\mathbf g)$ follow from the four nodal values; the
    energy is $V_+\\sigma_+|\\mathbf g|^2 + V_-\\sigma_-|M\\mathbf g|^2$ with the exact
    volumes on either side.

    Parameters
    ----------
    x
        Vertex coordinates in m, shape ``(4, m, 3)``.
    axis, value
        Plane of every element: axis index ``(m,)`` and coordinate in m.
    sigma_plus, sigma_minus
        Conductivity in S/m on the side above and below ``value``.
    vol
        Element volumes in m³.
    """
    from groundfield.solver.fem._mesh3d import below_fraction

    m = x.shape[1]
    e = np.zeros((m, 3))
    e[np.arange(m), axis] = 1.0
    c = sigma_plus / sigma_minus - 1.0
    coord = x[:, np.arange(m), axis]  # (4, m)
    rel = x.copy()
    rel[:, np.arange(m), axis] -= value[None, :]
    plus = coord > value[None, :]
    normal = rel[:, np.arange(m), axis]
    scaled = rel.copy()
    scaled[:, np.arange(m), axis] = np.where(plus, normal, (1.0 + c)[None, :] * normal)
    a = np.moveaxis(np.concatenate([np.ones((4, m, 1)), scaled], axis=2), 0, 1)
    inverse = np.linalg.inv(a)  # (m, 4, 4): nodal values -> (phi_0, g)
    b = inverse[:, 1:, :]
    f_minus = below_fraction(coord, value)
    v_minus, v_plus = vol * f_minus, vol * (1.0 - f_minus)
    eye = np.broadcast_to(np.eye(3), (m, 3, 3))
    ee = e[:, :, None] * e[:, None, :]
    msq = eye + ((1.0 + c) ** 2 - 1.0)[:, None, None] * ee
    d = (v_plus * sigma_plus)[:, None, None] * eye + (v_minus * sigma_minus)[
        :, None, None
    ] * msq
    k_e = np.einsum("mai,mab,mbj->mij", b, d, b)
    if return_maps:
        return k_e, inverse
    return k_e


#: Penalty of the partially penalised immersed elements on the faces the
#: interface crosses, in units of $\\max\\sigma / h_F$ (Lin, Lin & Zhang 2015).
PENALTY = 10.0


def ife_face_terms(
    p: np.ndarray,
    t: np.ndarray,
    inverse: np.ndarray,
    axis: np.ndarray,
    value: np.ndarray,
    sigma_plus: np.ndarray,
    sigma_minus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face terms of the partially penalised immersed elements (COO triplets).

    The immersed shape functions are linear on either side of the plane
    within each element, but the kink of one element need not match its
    neighbour's on a face the plane crosses. The symmetric partially
    penalised form (Lin, Lin & Zhang 2015) adds on every such interior face

    $$
    -\\int_F \\{\\sigma\\partial_n u\\}[v] - \\int_F \\{\\sigma\\partial_n v\\}[u]
    + \\frac{\\gamma\\,\\max\\sigma}{h_F}\\int_F [u][v],
    $$

    which keeps the method consistent (a piecewise linear solution with the
    right flux jump is reproduced exactly) and restores the second order;
    without it the discrete energy can fall below the physical one (stream
    w4fem3d: -2 % on a layered cube).

    Parameters
    ----------
    p
        Node coordinates in m.
    t
        Tetrahedra of the immersed elements ``(4, m)``.
    inverse
        Their parameter maps (nodal values -> ``(phi_0, g)``), ``(m, 4, 4)``.
    axis, value, sigma_plus, sigma_minus
        Their planes and conductivities (as in :func:`ife_matrices`).

    Returns
    -------
    tuple
        ``(rows, cols, values)`` of the face terms.
    """
    m = t.shape[1]
    if m == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    faces, owner, opposite = [], [], []
    for drop in range(4):
        keep = [k for k in range(4) if k != drop]
        faces.append(np.sort(t[keep], axis=0))
        owner.append(np.arange(m))
        opposite.append(t[drop])
    faces = np.concatenate(faces, axis=1)
    owner = np.concatenate(owner)
    opposite = np.concatenate(opposite)
    # Pair the faces by their sorted node triples (lexicographic, stable).
    # A packed integer key of the three node numbers aliased above 2**21
    # nodes and coupled unrelated elements (audit W4B-06).
    order = np.lexsort((faces[2], faces[1], faces[0]))
    sorted_faces = faces[:, order]
    pair = np.nonzero(np.all(sorted_faces[:, 1:] == sorted_faces[:, :-1], axis=0))[0]
    e1, e2 = owner[order[pair]], owner[order[pair + 1]]
    f = faces[:, order[pair]]
    same = (axis[e1] == axis[e2]) & (value[e1] == value[e2])
    ax = axis[e1]
    v0 = value[e1]
    coord = p[ax[None, :], f]  # (3, nf)
    crossed = same & (coord.min(axis=0) < v0) & (coord.max(axis=0) > v0)
    e1, e2, f, ax, v0, coord = (
        e1[crossed],
        e2[crossed],
        f[:, crossed],
        ax[crossed],
        v0[crossed],
        coord[:, crossed],
    )
    nf = f.shape[1]
    if nf == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    opp1 = opposite[order[pair]][crossed]
    verts = np.moveaxis(p[:, f], 0, 2)  # (3, nf, 3)
    normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= (2.0 * area)[:, None]
    toward = np.einsum("ij,ij->i", normal, p[:, opp1].T - verts[0])
    normal *= -np.sign(toward)[:, None]  # outward of the first element
    # Split the face at the plane: the vertex alone on its side, then the two.
    side = coord > v0[None, :]
    alone = np.where(
        side.sum(axis=0) == 1, np.argmax(side, axis=0), np.argmin(side, axis=0)
    )
    rest = np.array([[1, 2], [0, 2], [0, 1]])[alone].T
    a_ = verts[alone, np.arange(nf)]
    b_ = verts[rest[0], np.arange(nf)]
    c_ = verts[rest[1], np.arange(nf)]

    def cut(u, w):
        frac = (v0 - u[np.arange(nf), ax]) / (
            w[np.arange(nf), ax] - u[np.arange(nf), ax]
        )
        return u + frac[:, None] * (w - u)

    m1, m2 = cut(a_, b_), cut(a_, c_)
    triangles = [(a_, m1, m2), (b_, c_, m2), (b_, m2, m1)]
    # local node list: the union of both elements' nodes (at most five)
    nodes = np.concatenate([t[:, e1], t[:, e2]], axis=0)  # (8, nf)
    rule = np.array(
        [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
    )
    eye3 = np.eye(3)
    sigmax = np.maximum(sigma_plus[e1], sigma_minus[e1])
    gamma = PENALTY * sigmax / np.sqrt(2.0 * area)
    k_f = np.zeros((nf, 8, 8))
    for tri in triangles:
        ta = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]), axis=1)
        for bc in rule:
            x = bc[0] * tri[0] + bc[1] * tri[1] + bc[2] * tri[2]
            rows_val, rows_flux = [], []
            for e in (e1, e2):
                rel = x.copy()
                rel[np.arange(nf), ax] -= v0
                plus = x[np.arange(nf), ax] > v0
                scale = np.where(plus, 1.0, sigma_plus[e] / sigma_minus[e])
                y = rel.copy()
                y[np.arange(nf), ax] *= scale
                val = np.einsum(
                    "fk,fkn->fn",
                    np.concatenate([np.ones((nf, 1)), y], axis=1),
                    inverse[e],
                )
                w = normal.copy()
                w[np.arange(nf), ax] *= scale
                sig = np.where(plus, sigma_plus[e], sigma_minus[e])
                flux = sig[:, None] * np.einsum("fk,fkn->fn", w, inverse[e][:, 1:, :])
                rows_val.append(val)
                rows_flux.append(flux)
            jump = np.concatenate([rows_val[0], -rows_val[1]], axis=1)  # (nf, 8)
            mean = 0.5 * np.concatenate([rows_flux[0], rows_flux[1]], axis=1)
            weight = (ta / 3.0)[:, None, None]
            k_f += weight * (
                -mean[:, :, None] * jump[:, None, :]
                - jump[:, :, None] * mean[:, None, :]
                + gamma[:, None, None] * jump[:, :, None] * jump[:, None, :]
            )
    del eye3
    rows = np.broadcast_to(nodes.T[:, :, None], k_f.shape).ravel()
    cols = np.broadcast_to(nodes.T[:, None, :], k_f.shape).ravel()
    return rows, cols, k_f.ravel()


def classify_elements(
    p: np.ndarray,
    t: np.ndarray,
    interfaces: np.ndarray,
    sigma_layers: np.ndarray,
    prisms: tuple = (),
    sigma_concrete: float | None = None,
    chunk: int = 500_000,
) -> dict:
    """Material model of every element (see :func:`stiffness_matrix`).

    Returns a dict of per-element arrays: ``kind`` (0: scalar conductivity
    ``sigma``; 1: immersed element with ``axis``, ``value``, ``sigma_plus``,
    ``sigma_minus``; 2: laminate with ``sigma_h``, ``sigma_v``).
    """
    from groundfield.solver.fem._mesh3d import below_fraction

    ne = t.shape[1]
    out = {
        "kind": np.zeros(ne, dtype=np.int8),
        "sigma": np.zeros(ne),
        "axis": np.full(ne, -1, dtype=np.int8),
        "value": np.zeros(ne, dtype=np.float64),
        "sigma_plus": np.zeros(ne),
        "sigma_minus": np.zeros(ne),
        "sigma_h": np.zeros(ne),
        "sigma_v": np.zeros(ne),
    }
    bounds = np.concatenate([[0.0], interfaces, [np.inf]])
    thick = np.diff(bounds)
    for s0 in range(0, ne, chunk):
        sl = slice(s0, min(ne, s0 + chunk))
        x = np.moveaxis(p[:, t[:, sl]], 0, 2)  # (4, m, 3)
        m = x.shape[1]
        z = x[:, :, 2]
        z_lo, z_hi = z.min(axis=0), z.max(axis=0)
        n_cut = np.zeros(m, dtype=int)
        which = np.full(m, -1)
        for k, h in enumerate(interfaces):
            cut = (z_lo < h) & (z_hi > h)
            n_cut += cut
            which = np.where(cut & (which < 0), k, which)
        sigma = sigma_layers[
            np.searchsorted(interfaces, x.mean(axis=0)[:, 2], side="right")
        ]
        kind = np.zeros(m, dtype=np.int8)
        axis = np.full(m, -1, dtype=np.int64)
        value = np.zeros(m)
        sp = np.zeros(m)
        sm = np.zeros(m)
        if interfaces.size:
            k = np.clip(which, 0, interfaces.size - 1)
            thin = np.minimum(thick[k], thick[k + 1])
            size = np.max(np.ptp(x, axis=0), axis=1)
            ife = (n_cut == 1) & (size <= IFE_SIZE_RATIO * thin)
            kind[ife] = 1
            axis[ife] = 2
            value[ife] = interfaces[k[ife]]
            sp[ife] = sigma_layers[k[ife] + 1]
            sm[ife] = sigma_layers[k[ife]]
            lam = (n_cut > 1) | ((n_cut == 1) & ~ife)
            if lam.any():
                cum = [np.zeros(lam.sum())]
                cum += [below_fraction(z[:, lam], h) for h in interfaces]
                cum.append(np.ones(lam.sum()))
                frac = np.diff(np.array(cum), axis=0)
                out["sigma_h"][sl][lam] = frac.T @ sigma_layers
                out["sigma_v"][sl][lam] = 1.0 / (frac.T @ (1.0 / sigma_layers))
                kind[lam] = 2
        if prisms:
            concrete, plane, mixed = _concrete_elements(x, prisms)
            kind[concrete | mixed] = 0
            sigma = np.where(concrete, sigma_concrete, sigma)
            single = mixed & (plane[0] >= 0)
            inside_plus = plane[2] > 0
            kind[single] = 1
            axis[single] = plane[0][single]
            value[single] = plane[1][single]
            sp[single] = np.where(inside_plus, sigma_concrete, sigma)[single]
            sm[single] = np.where(inside_plus, sigma, sigma_concrete)[single]
            rest = np.nonzero(mixed & (plane[0] < 0))[0]
            if rest.size:
                frac = _inside_fraction(x[:, rest], prisms)
                sigma[rest] = frac * sigma_concrete + (1.0 - frac) * sigma[rest]
        out["kind"][sl] = kind
        out["sigma"][sl] = sigma
        out["axis"][sl] = axis
        out["value"][sl] = value
        out["sigma_plus"][sl] = sp
        out["sigma_minus"][sl] = sm
    return out


def _inside(points: np.ndarray, prisms: tuple) -> np.ndarray:
    """Whether points ``(..., 3)`` lie inside the union of the prisms."""
    out = np.zeros(points.shape[:-1], dtype=bool)
    for q in prisms:
        out |= np.all((points >= q.lower) & (points <= q.upper), axis=-1)
    return out


def _concrete_elements(x: np.ndarray, prisms: tuple):
    """Elements inside the concrete, and the plane of each mixed element.

    Returns ``(inside (ne,), (axis, value, side) per element, mixed (ne,))``;
    ``axis = -1`` where no single prism face separates the vertices inside
    the concrete from those outside (edges and corners of the concrete).
    """
    ne = x.shape[1]
    flags = _inside(x, prisms)  # (4, ne)
    n_in = flags.sum(axis=0)
    inside = n_in == 4
    mixed = (n_in > 0) & (n_in < 4)
    axis = np.full(ne, -1)
    value = np.zeros(ne)
    side = np.zeros(ne, dtype=int)
    idx = np.nonzero(mixed)[0]
    found = np.zeros(idx.size, dtype=int)
    for q in prisms:
        for a in range(3):
            for v, inner in ((q.lower[a], 1), (q.upper[a], -1)):
                coord = x[:, idx, a]
                pos = coord > v
                # the plane separates exactly the inside vertices
                agree = np.all(
                    pos == (flags[:, idx] if inner > 0 else ~flags[:, idx]), axis=0
                )
                straddle = (coord.min(axis=0) < v) & (coord.max(axis=0) > v)
                hit = agree & straddle
                new = hit & (found == 0)
                axis[idx[new]] = a
                value[idx[new]] = v
                side[idx[new]] = inner
                found += hit
    ambiguous = idx[found != 1]
    axis[ambiguous] = -1
    return inside, (axis, value, side), mixed


def _inside_fraction(x: np.ndarray, prisms: tuple, n: int = 6) -> np.ndarray:
    """Volume fraction inside the concrete by a barycentric lattice rule."""
    pts = [
        (i, j, k, n - i - j - k)
        for i in range(n + 1)
        for j in range(n + 1 - i)
        for k in range(n + 1 - i - j)
    ]
    lam = (np.array(pts, dtype=float) + 0.25) / (n + 1.0)
    lam /= lam.sum(axis=1, keepdims=True)
    q = np.einsum("pv,vmc->pmc", lam, x)
    return _inside(q, prisms).mean(axis=0)


def stiffness_matrix(
    p: np.ndarray,
    t: np.ndarray,
    interfaces: np.ndarray,
    sigma_layers: np.ndarray,
    prisms: tuple = (),
    sigma_concrete: float | None = None,
    chunk: int = 200_000,
) -> csr_matrix:
    """Global stiffness matrix (CSR) of the soil and the resolved concrete.

    * An element within one material: the P1 matrix
      $\\sigma|T|\\nabla\\lambda_i\\cdot\\nabla\\lambda_j$.
    * An element that one layer interface $z = h$ cuts, or one face of a
      concrete prism (user decision Q40): the partially penalised immersed
      element (:func:`ife_matrices`, :func:`ife_face_terms`), second order
      with no mesh plane at the interface, so the mesh stays isotropic (a
      laminate mean converges only at first order, a mesh cut along the
      planes degenerates where a thin layer crosses the large far elements;
      evidence of stream w4fem3d).
    * An element much larger than the layers it cuts (far from the
      electrodes): the laminate, arithmetic along the layers, harmonic
      across them. An element at an edge or corner of the concrete (a band
      of volume $O(h^2)$, which keeps the second order): the volume mean.

    Assembled from the off-diagonal entries; the diagonal is the negative
    row sum (every contribution annihilates constants).
    """
    n = p.shape[1]
    model = classify_elements(p, t, interfaces, sigma_layers, prisms, sigma_concrete)
    kind = model["kind"]
    plain = np.nonzero(kind != 1)[0]
    rows = np.empty(6 * plain.size, dtype=np.int32)
    cols = np.empty(6 * plain.size, dtype=np.int32)
    vals = np.empty(6 * plain.size)
    for s0 in range(0, plain.size, chunk):
        idx = plain[s0 : s0 + chunk]
        tt = t[:, idx]
        vol, grad = tet_gradients(p, tt)
        lam = kind[idx] == 2
        tensor = np.repeat(model["sigma"][idx][:, None], 3, axis=1)
        tensor[lam, 0] = tensor[lam, 1] = model["sigma_h"][idx][lam]
        tensor[lam, 2] = model["sigma_v"][idx][lam]
        for k, (a, b) in enumerate(TET_EDGES):
            sl = slice(6 * s0 + k * idx.size, 6 * s0 + (k + 1) * idx.size)
            vals[sl] = vol * np.einsum("ij,ij,ij->i", grad[:, a], tensor, grad[:, b])
            rows[sl] = np.minimum(tt[a], tt[b])
            cols[sl] = np.maximum(tt[a], tt[b])
    upper = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    del rows, cols, vals
    off = upper + upper.T
    del upper
    ife = np.nonzero(kind == 1)[0]
    if ife.size:
        tt = t[:, ife]
        x = np.moveaxis(p[:, tt], 0, 2)
        vol, _ = tet_gradients(p, tt)
        ax, val = model["axis"][ife].astype(np.int64), model["value"][ife]
        sp, sm = model["sigma_plus"][ife], model["sigma_minus"][ife]
        k_e, inverse = ife_matrices(x, ax, val, sp, sm, vol, return_maps=True)
        rows = np.broadcast_to(tt.T[:, :, None], k_e.shape).ravel()
        cols = np.broadcast_to(tt.T[:, None, :], k_e.shape).ravel()
        fr, fc, fv = ife_face_terms(p, tt, inverse, ax, val, sp, sm)
        rows = np.concatenate([rows, fr])
        cols = np.concatenate([cols, fc])
        vals = np.concatenate([k_e.ravel(), fv])
        keep = rows != cols
        off = (
            off
            + coo_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n)).tocsr()
        )
    return (off - diags(np.asarray(off.sum(axis=1)).ravel())).tocsr()


def robin(
    p: np.ndarray, faces: np.ndarray, sigma_faces: np.ndarray, center: np.ndarray
):
    """Monopole Robin term $\\sigma(\\mathbf n\\cdot\\mathbf r/r^2)\\,u\\,v$ on box faces."""
    n = p.shape[1]
    corners = [p[:, faces[k]].T for k in range(3)]
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= (2.0 * area)[:, None]
    middle = (corners[0] + corners[1] + corners[2]) / 3.0
    normal *= np.sign(np.einsum("ij,ij->i", normal, middle - center))[:, None]
    rule = np.array(
        [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
    )
    m_e = np.zeros((faces.shape[1], 3, 3))
    for b in rule:
        x = b[0] * corners[0] + b[1] * corners[1] + b[2] * corners[2]
        r = x - center
        coef = np.einsum("ij,ij->i", normal, r) / np.einsum("ij,ij->i", r, r)
        m_e += (area / 3.0 * coef * sigma_faces)[:, None, None] * np.outer(b, b)[None]
    rows = np.broadcast_to(faces.T[:, :, None], m_e.shape).ravel()
    cols = np.broadcast_to(faces.T[:, None, :], m_e.shape).ravel()
    return coo_matrix((m_e.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def locate(p: np.ndarray, t: np.ndarray, x: np.ndarray, candidates: np.ndarray):
    """Containing tetrahedron and barycentric coordinates of points ``x``.

    Parameters
    ----------
    p, t
        Mesh arrays.
    x
        Points in m, shape ``(k, 3)``.
    candidates
        Indices of the tetrahedra that may contain them.

    Returns
    -------
    tuple
        ``(element (k,), barycentric (k, 4))``.

    Raises
    ------
    RuntimeError
        If a point lies in none of the candidates.
    """
    from scipy.spatial import cKDTree

    tc = t[:, candidates]
    tree = cKDTree(p[:, tc].mean(axis=1).T)
    p0 = p[:, tc[0]].T
    jac = np.stack([p[:, tc[1]].T - p0, p[:, tc[2]].T - p0, p[:, tc[3]].T - p0], axis=2)
    inv = np.linalg.inv(jac)
    element = -np.ones(len(x), dtype=np.int64)
    bary = np.zeros((len(x), 4))
    todo = np.arange(len(x))
    for k in (8, 32, 128, 512):
        if todo.size == 0:
            break
        _, idx = tree.query(x[todo], k=min(k, len(candidates)))
        idx = idx.reshape(todo.size, -1)
        best = np.full(todo.size, -np.inf)
        for j in range(idx.shape[1]):
            c = idx[:, j]
            lam = np.einsum("kij,kj->ki", inv[c], x[todo] - p0[c])
            worst = np.minimum(lam.min(axis=1), 1.0 - lam.sum(axis=1))
            better = worst > best
            sel = todo[better]
            element[sel] = candidates[c[better]]
            bary[sel, 0] = 1.0 - lam[better].sum(axis=1)
            bary[sel, 1:] = lam[better]
            best = np.where(better, worst, best)
        todo = todo[best < -1e-9]
    if todo.size:  # pragma: no cover - the candidates cover the source discs
        raise RuntimeError(f"fem: {todo.size} source points lie outside the mesh")
    return element, bary


def bump_disc(radius: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Points ``(m, 2)`` and weights ``(m,)`` of the bump density on its disc.

    Gauss–Jacobi in $u = r^2/R^2$ for the weight $3(1-u)^2$, a uniform
    rule in angle with about ``delta`` between points; weights sum to 1.
    """
    from scipy.special import roots_jacobi

    n_r = max(3, math.ceil(radius / delta))
    xr, wr = roots_jacobi(n_r, 2, 0)
    u = 0.5 * (xr + 1.0)
    wu = wr * 0.5**3 * 3.0
    pts, wts = [], []
    for k, (rk, wk) in enumerate(zip(radius * np.sqrt(u), wu, strict=True)):
        n_t = max(6, math.ceil(2.0 * math.pi * rk / delta))
        theta = 2.0 * math.pi * (np.arange(n_t) + 0.5 * (k % 2)) / n_t
        pts.append(np.column_stack([rk * np.cos(theta), rk * np.sin(theta)]))
        wts.append(np.full(n_t, wk / n_t))
    return np.concatenate(pts), np.concatenate(wts)


def disc_average_matrix(
    p: np.ndarray,
    t: np.ndarray,
    points: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    disc: np.ndarray,
    disc_w: np.ndarray,
    candidates: np.ndarray,
    chunk: int = 2000,
) -> csr_matrix:
    """Rows: the bump-weighted disc average of a P1 function at each axial point."""
    n = p.shape[1]
    rows, cols, vals = [], [], []
    m = disc.shape[0]
    for s in range(0, points.shape[0], chunk):
        xs = points[s : s + chunk]
        q = (
            xs[:, None, :]
            + disc[None, :, 0, None] * e1[s : s + chunk, None, :]
            + disc[None, :, 1, None] * e2[s : s + chunk, None, :]
        ).reshape(-1, 3)
        element, bary = locate(p, t, q, candidates)
        w = np.tile(disc_w, xs.shape[0])
        r = np.repeat(np.arange(s, s + xs.shape[0]), m)
        rows.append(np.repeat(r, 4))
        cols.append(t[:, element].T.ravel())
        vals.append((bary * w[:, None]).ravel())
    a = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(points.shape[0], n),
    ).tocsr()
    a.sum_duplicates()
    return a


@dataclass
class PortSolution3D:
    """Port conductance and leakage of one 3-D mesh.

    Attributes
    ----------
    conductance
        Port conductance matrix $G$ in S, shape ``(P, P)``.
    current
        Current in A leaving at every axial point for 1 V on port ``j``
        (0 V on the others), shape ``(n_axial, P)``.
    iterations
        Conjugate-gradient iterations per port.
    """

    conductance: np.ndarray
    current: np.ndarray
    iterations: list[int]


def solve_ports_3d(
    stiffness: csr_matrix,
    average: csr_matrix,
    weights: np.ndarray,
    coefficient: np.ndarray,
    port: np.ndarray,
    n_ports: int,
) -> PortSolution3D:
    """Port conductance matrix of the coupled system (see the module docstring).

    The system matrix $K + A^\\mathsf T D A$ ($A$ the disc averages, $D$ the
    wire conductances per axial point) is never formed: the coupling is
    dense around the wires (every axial point joins all the nodes of its
    disc) and would multiply the memory of the multigrid hierarchy. The
    conjugate gradients run on the operator, preconditioned by the
    smoothed-aggregation multigrid of the soil stiffness $K$ alone; the
    coupling is a positive semidefinite update local to the wires.

    Parameters
    ----------
    stiffness
        Soil stiffness with the far-field term, shape ``(N, N)``.
    average
        Disc-average rows of the axial points, shape ``(n_axial, N)``.
    weights
        Axial quadrature weights in m.
    coefficient
        $C_\\mathrm{w}$ in Ω·m at every axial point (positive).
    port
        Port index of every axial point.
    n_ports
        Number of ports.

    Raises
    ------
    ConvergenceError
        If a port solve does not reach the relative residual
        :data:`CG_TOLERANCE` in :data:`CG_MAX_ITERATIONS` iterations (high
        layer contrast; the caller turns it into a refusal).
    """
    import pyamg
    from scipy.sparse.linalg import LinearOperator, cg

    wc = weights / coefficient
    average_t = average.T.tocsr()
    n = stiffness.shape[0]

    def apply(v: np.ndarray) -> np.ndarray:
        return stiffness @ v + average_t @ (wc * (average @ v))

    operator = LinearOperator((n, n), matvec=apply, dtype=float)
    state = np.random.get_state()
    np.random.seed(AMG_SEED)
    try:
        hierarchy = pyamg.smoothed_aggregation_solver(
            stiffness, symmetry="symmetric", max_coarse=2000
        )
    finally:
        np.random.set_state(state)
    preconditioner = hierarchy.aspreconditioner(cycle="V")
    g = np.zeros((n_ports, n_ports))
    current = np.zeros((weights.size, n_ports))
    iterations: list[int] = []
    for k in range(n_ports):
        mask = (port == k).astype(float)
        rhs = np.asarray(average_t @ (wc * mask)).ravel()
        count = [0]

        def tick(_x: np.ndarray, count: list[int] = count) -> None:
            count[0] += 1

        x, info = cg(
            operator,
            rhs,
            rtol=CG_TOLERANCE,
            maxiter=CG_MAX_ITERATIONS,
            M=preconditioner,
            callback=tick,
        )
        if info != 0:
            raise ConvergenceError(k)
        iterations.append(count[0])
        pi = np.asarray(average @ x).ravel()
        current[:, k] = wc * (mask - pi)
        for j in range(n_ports):
            g[j, k] = float(current[port == j, k].sum())
    g = 0.5 * (g + g.T)
    return PortSolution3D(conductance=g, current=current, iterations=iterations)
