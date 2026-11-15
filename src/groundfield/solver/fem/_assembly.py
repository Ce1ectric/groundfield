"""Axisymmetric P1 assembly and the electrode-port reduction (private).

Weak form of $-\\nabla\\cdot(\\sigma\\nabla\\varphi) = 0$ on the meridian
half-plane, with the volume element $\\mathrm dV = 2\\pi s\\,\\mathrm ds\\,
\\mathrm dz$:

$$
a(\\varphi, v) = \\int 2\\pi s\\,\\sigma\\,\\nabla\\varphi\\cdot\\nabla v
\\,\\mathrm ds\\,\\mathrm dz
+ \\oint_{\\Gamma_\\infty} 2\\pi s\\,\\sigma\\,\\frac{\\mathbf n\\cdot\\mathbf r}
{|\\mathbf r|^2}\\,\\varphi v\\,\\mathrm dl
+ \\sum_k \\oint_{\\Gamma_k} 2\\pi s\\, g_k\\,(\\varphi - U_k)(v - V_k)\\,\\mathrm dl .
$$

* Linear triangles: the gradients are constant and
  $\\int_T s\\,\\mathrm dA = \\bar s_T |T|$ holds exactly, so the volume
  term is integrated exactly.
* The soil surface $z = 0$ and the axis $s = 0$ carry the natural
  condition (no current into the air; symmetry).
* $\\Gamma_\\infty$, the box $s = R$ or $z = R$: the Robin condition
  $\\partial_n\\varphi = -(\\mathbf n\\cdot\\mathbf r/|\\mathbf r|^2)\\,\\varphi$,
  $\\mathbf r$ from the point of the axis on the soil surface. It is the
  exact Dirichlet-to-Neumann map of the monopole $\\varphi \\propto 1/r$
  on any boundary, and the leading far field of every grounding system in
  horizontally layered soil is that monopole (its amplitude set by the
  bottom layer), so the truncation error is the multipole remainder,
  $O((L/R)^2)$ for an electrode of size $L$. Integrated with four-point
  Gauss per edge.
* A bare electrode $k$ is an equipotential: its surface nodes are one
  unknown $U_k$ (a *port*). An encased electrode (concrete shell,
  ADR-0012) keeps its surface nodes free and couples them to $U_k$
  through the shell's radial resistance per unit length $C_k$: the
  surface conductance $g_k = 1/(2\\pi r_b C_k)$ in S/m², $r_b$ the outer
  shell radius (the electrode's ``wire_radius``), so that a radial
  current $I'$ per unit length drops $C_k I'$ across the shell — the
  per-length coefficient of ADR-0012, locally at every point of the
  surface (Sunde's cylindrical shell, exact for a shell thin against the
  scale on which the leakage current varies).

Port reduction. With the interior unknowns $\\boldsymbol\\varphi_I$ and
the port potentials $\\mathbf U$, the system
$\\begin{pmatrix} K_{II} & K_{IP} \\\\ K_{PI} & K_{PP}\\end{pmatrix}$
gives the port conductance matrix (Schur complement)
$G = K_{PP} - K_{PI} K_{II}^{-1} K_{IP}$, symmetric positive definite:
$\\mathbf I = G\\,\\mathbf U$ are the currents the electrodes leak into the
soil. One sparse LU factorisation of $K_{II}$ serves all ports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

from groundfield.solver.fem._mesh import AxiMesh

__all__: list[str] = []

_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(4)


def stiffness(nodes: np.ndarray, triangles: np.ndarray, sigma: np.ndarray, n: int):
    """Volume term $2\\pi\\sigma\\bar s_T|T|\\,\\nabla\\phi_i\\cdot\\nabla\\phi_j$.

    Parameters
    ----------
    nodes, triangles
        Mesh arrays (m).
    sigma
        Conductivity per triangle in S/m.
    n
        Size of the returned square matrix (``>= len(nodes)``).
    """
    v = nodes[triangles]
    x, y = v[:, :, 0], v[:, :, 1]
    area = 0.5 * np.abs(
        (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    )
    keep = area > 0.0
    x, y, area = x[keep], y[keep], area[keep]
    tri = triangles[keep]
    inv = 1.0 / (2.0 * area)
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
    b = b * inv[:, None]
    c = c * inv[:, None]
    weight = np.asarray(sigma, dtype=float)[keep] * area * 2.0 * np.pi * x.mean(axis=1)
    k_e = weight[:, None, None] * (
        b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    )
    rows = np.broadcast_to(tri[:, :, None], k_e.shape).ravel()
    cols = np.broadcast_to(tri[:, None, :], k_e.shape).ravel()
    return coo_matrix((k_e.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def _edge_quadrature(nodes: np.ndarray, edges: np.ndarray):
    """Gauss points, weights (incl. the length) and shape values per edge."""
    p1 = nodes[edges[:, 0]]
    p2 = nodes[edges[:, 1]]
    d = p2 - p1
    length = np.hypot(d[:, 0], d[:, 1])
    for xg, wg in zip(_GAUSS_X, _GAUSS_W, strict=True):
        t = 0.5 * (xg + 1.0)
        yield p1 + t * d, 0.5 * wg * length, np.array([1.0 - t, t]), d


def far_field(nodes: np.ndarray, edges: np.ndarray, sigma: np.ndarray, n: int):
    """Monopole Robin term on the truncation box (see the module docstring)."""
    if edges.size == 0:
        return csr_matrix((n, n))
    m_e = np.zeros((edges.shape[0], 2, 2))
    for p, w, shape, d in _edge_quadrature(nodes, edges):
        vertical = np.abs(d[:, 0]) <= np.abs(d[:, 1])  # an edge of s = R
        n_dot_r = np.where(vertical, p[:, 0], p[:, 1])
        coef = n_dot_r / (p[:, 0] ** 2 + p[:, 1] ** 2)
        weight = w * 2.0 * np.pi * sigma * coef * p[:, 0]
        m_e += weight[:, None, None] * (shape[:, None] * shape[None, :])[None]
    rows = np.broadcast_to(edges[:, :, None], m_e.shape).ravel()
    cols = np.broadcast_to(edges[:, None, :], m_e.shape).ravel()
    return coo_matrix((m_e.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def shell_coupling(
    nodes: np.ndarray, edges: np.ndarray, g: float, virtual: int, n: int
):
    """Surface-conductance term $2\\pi s g (\\varphi - U)(v - V)$ of one port.

    ``virtual`` is the row/column of the port potential $U$.
    """
    m_e = np.zeros((edges.shape[0], 2, 2))
    c_e = np.zeros((edges.shape[0], 2))
    u_e = np.zeros(edges.shape[0])
    for p, w, shape, _d in _edge_quadrature(nodes, edges):
        weight = w * 2.0 * np.pi * g * p[:, 0]
        m_e += weight[:, None, None] * (shape[:, None] * shape[None, :])[None]
        c_e -= weight[:, None] * shape[None, :]
        u_e += weight
    rows = [np.broadcast_to(edges[:, :, None], m_e.shape).ravel()]
    cols = [np.broadcast_to(edges[:, None, :], m_e.shape).ravel()]
    vals = [m_e.ravel()]
    rows += [edges.ravel(), np.full(edges.size, virtual)]
    cols += [np.full(edges.size, virtual), edges.ravel()]
    vals += [c_e.ravel(), c_e.ravel()]
    rows.append(np.array([virtual]))
    cols.append(np.array([virtual]))
    vals.append(np.array([u_e.sum()]))
    return coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()


@dataclass
class PortSolution:
    """Port conductance and surface leakage of one mesh.

    Attributes
    ----------
    conductance
        Port conductance matrix $G$ in S, shape ``(P, P)``: the currents in
        A that the electrodes leak into the soil for port potentials in V.
    surface_nodes
        Node indices on the electrode surfaces, one array per port.
    leakage
        Current in A leaving the soil-side surface of each port through
        each of its nodes (the consistent nodal flux), for 1 V on port
        ``j`` and 0 V on the others: one ``(n_k, P)`` array per port.
    potentials
        Node potentials in V for 1 V on each port, shape ``(N, P)``.
    """

    conductance: np.ndarray
    surface_nodes: list[np.ndarray]
    leakage: list[np.ndarray]
    potentials: np.ndarray


def solve_ports(
    mesh: AxiMesh,
    rhos: np.ndarray,
    shell_conductance: list[float],
    *,
    far_field_model: str = "robin",
) -> PortSolution:
    """Port conductance matrix of ``mesh`` (see the module docstring).

    Parameters
    ----------
    mesh
        The mesh; ``mesh.layer`` indexes ``rhos``.
    rhos
        Resistivity per layer in Ω·m.
    shell_conductance
        Surface conductance $g_k = 1/(2\\pi r_b C_k)$ of the concrete shell
        in S/m² per port; ``0`` for a bare electrode.
    far_field_model
        ``"robin"`` (monopole Dirichlet-to-Neumann map) or ``"dirichlet"``
        (the box grounded, $\\varphi = 0$).
    """
    nodes, triangles = mesh.nodes, mesh.triangles
    n_nodes = nodes.shape[0]
    n_ports = len(mesh.port_edges)
    encased = [k for k in range(n_ports) if shell_conductance[k] > 0.0]
    virtual = {k: n_nodes + m for m, k in enumerate(encased)}
    n_ext = n_nodes + len(encased)
    sigma = 1.0 / np.asarray(rhos, dtype=float)
    k_soil = stiffness(nodes, triangles, sigma[mesh.layer], n_ext)
    outer = mesh.outer_edges
    grounded = np.zeros(n_ext, dtype=bool)
    if far_field_model == "robin":
        bounds = np.asarray(mesh.interfaces, dtype=float)
        sig_edge = sigma[
            np.searchsorted(bounds, nodes[outer, 1].mean(axis=1), side="right")
        ]
        k_soil = k_soil + far_field(nodes, outer, sig_edge, n_ext)
    elif far_field_model == "dirichlet":
        grounded[np.unique(outer.ravel())] = True
    else:  # pragma: no cover - validated by FemOptions
        raise ValueError(f"unknown far-field model {far_field_model!r}")
    k_total = k_soil
    for k, vk in virtual.items():
        k_total = k_total + shell_coupling(
            nodes, mesh.port_edges[k], float(shell_conductance[k]), vk, n_ext
        )

    # Degrees of freedom: the interior unknowns first, then one per port
    # (all surface nodes of a bare port, the shell potential of an encased
    # one, whose surface nodes stay free).
    port_of = np.concatenate([mesh.port, -np.ones(len(encased), dtype=int)])
    for k, vk in virtual.items():
        port_of[:n_nodes][mesh.port == k] = -1
        port_of[vk] = k
    interior = np.nonzero((port_of < 0) & ~grounded)[0]
    n_int = interior.size
    dof = -np.ones(n_ext, dtype=int)
    dof[interior] = np.arange(n_int)
    bound = port_of >= 0
    dof[bound] = n_int + port_of[bound]
    rows = np.nonzero(dof >= 0)[0]
    p_map = coo_matrix(
        (np.ones(rows.size), (rows, dof[rows])), shape=(n_ext, n_int + n_ports)
    ).tocsr()
    k_dof = (p_map.T @ k_total @ p_map).tocsc()
    k_ii = k_dof[:n_int, :n_int].tocsc()
    k_ip = k_dof[:n_int, n_int:].toarray()
    k_pp = k_dof[n_int:, n_int:].toarray()
    x = splu(k_ii).solve(k_ip)
    conductance = k_pp - k_ip.T @ x
    conductance = 0.5 * (conductance + conductance.T)

    full = np.zeros((n_int + n_ports, n_ports))
    full[:n_int] = -x
    full[n_int:] = np.eye(n_ports)
    phi_ext = np.asarray(p_map @ full)  # (n_ext, P); grounded nodes stay 0
    # The current leaving each surface node into the soil is the
    # soil-side nodal flux (the shell term is the electrode side).
    flux = np.asarray(k_soil @ phi_ext)
    surface_nodes, leakage = [], []
    for k in range(n_ports):
        idx = np.nonzero(mesh.port == k)[0]
        surface_nodes.append(idx)
        leakage.append(flux[idx])
    return PortSolution(
        conductance=conductance,
        surface_nodes=surface_nodes,
        leakage=leakage,
        potentials=phi_ext[:n_nodes],
    )
