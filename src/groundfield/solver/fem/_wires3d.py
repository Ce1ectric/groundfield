"""Thin-wire electrodes of the three-dimensional ``fem`` model (private).

The 3-D model (:mod:`groundfield.solver.fem._solid`) represents every
electrode and every leaking piece of a galvanic distributed conductor by
the axis of its wire: a chain of straight *pieces* (a ring by a fine
polygon). Along a piece the wire leaks the current density $j(s)$ in A/m,
and the soil sees it as a **smoothed line source**: $j(s)$ spread over
the disc of radius $R$ perpendicular to the axis with the bump density

$$
\\eta(r) = \\frac{3}{\\pi R^2}\\Bigl(1 - \\frac{r^2}{R^2}\\Bigr)^2,
\\qquad \\int \\eta\\,\\mathrm dA = 1,
$$

and the potential the wire "sees" is the $\\eta$-weighted average $\\Pi(s)$
over the same disc. The field of the smoothed source is smooth, so the
finite elements need no resolution of the wire radius $a$; what lies
inside the disc is added analytically as a radial resistance per unit
length between the metal and the disc average,

$$
U - \\Pi(s) = C_\\mathrm{w}(s)\\,j(s), \\qquad
C_\\mathrm{w} = C + \\Delta C(s),
$$

with the concrete-shell coefficient $C$ of an encased electrode (ADR-0012)
and the *radius correction* $\\Delta C$: the difference between the
potential of the thin wire of radius $a$ (reduced kernel
$1/\\sqrt{d^2 + a^2}$, source on the axis, potential on the surface) and
that of the smoothed source (energy radius $r_\\eta = 0.5442\\,R$ of the
bump), in homogeneous soil of the local resistivity, for a current density
that is uniform over the few $R$ around the point:

$$
\\Delta C(s) = \\frac{\\rho}{4\\pi}\\sum_k \\Bigl[\\Phi_k(s; a) - \\Phi_k(s; r_\\eta)\\Bigr],
\\qquad
\\Phi_k(s; c) = \\int_{\\text{piece }k} \\frac{\\mathrm ds'}{\\sqrt{|x(s) - x(s')|^2 + c^2}},
$$

summed over the pieces within reach (the own piece, its neighbours at a
corner or junction, and their images in the soil surface). Far inside a
straight wire this is Sunde's $\\frac{\\rho}{2\\pi}\\ln(r_\\eta/a)$; near a free
end, a corner or a junction it carries the three-dimensional geometry.
The smoothed source differs from the reduced kernel only in how its end
region averages (the bump's pair distances instead of $r_\\eta$); that
difference integrates to $0.086\\,R$ along the wire per free end
(evidence of stream w4fem3d), about $10^{-4}$ of a 3 m rod at $R = 4$ cm,
and is neglected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.world import World

__all__: list[str] = []

#: Energy radius of the bump density in units of its radius $R$:
#: $\\ln r_\\eta = \\iint \\eta(u)\\eta(v)\\ln|u - v|$ (computed once by radial
#: quadrature; reproduced by a 2-D pair quadrature to 1.4e-4).
BUMP_ENERGY_RADIUS = 0.5442413183379312

#: A piece of a ring polygon is no longer than this fraction of the ring
#: radius (the polygon's length is then short of the circle by 1.6e-5).
RING_CHORD_ANGLE = 2.0 * math.pi / 360.0


@dataclass(frozen=True)
class WirePiece:
    """One straight piece of a wire axis.

    Attributes
    ----------
    port
        Port index (world electrodes first, then the leaking pieces of
        galvanic distributed conductors).
    start, end
        End points in m, shape ``(3,)``.
    radius
        Wire radius in m the soil sees (the outer shell radius $r_b$ of an
        encased electrode).
    shell
        Concrete-shell coefficient $C$ in Ω·m (``0`` for a bare wire).
    """

    port: int
    start: np.ndarray
    end: np.ndarray
    radius: float
    shell: float

    @property
    def length(self) -> float:
        """Length in m."""
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        """Unit tangent."""
        return (self.end - self.start) / self.length


@dataclass
class WireModel:
    """The thin wires of one world.

    Attributes
    ----------
    names
        Port names: the electrodes in world order, then the pseudo-owners
        of the leaking conductor pieces in the order of the pipeline.
    pieces
        Straight pieces of every port.
    points, weights, piece
        Axial quadrature: points in m ``(n, 3)``, weights in m ``(n,)``
        (Gauss, two per axial cell), index of the piece of each point.
    """

    names: tuple[str, ...]
    pieces: list[WirePiece]
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    piece: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def port(self) -> np.ndarray:
        """Port index of every axial point."""
        return np.array([p.port for p in self.pieces], dtype=int)[self.piece]

    @property
    def total_length(self) -> float:
        """Total wire length in m."""
        return float(sum(p.length for p in self.pieces))


def _polylines(electrode) -> list[np.ndarray]:
    """Wire axes of one electrode as polylines (rings as fine polygons)."""
    from groundfield.geometry.electrodes import RingElectrode

    if isinstance(electrode, RingElectrode):
        n = max(64, math.ceil(2.0 * math.pi / RING_CHORD_ANGLE)) + 1
        return electrode.wire_polylines(n_arc=n)
    return electrode.wire_polylines()


def world_pieces(world: "World", extra_segments=()) -> WireModel:
    """Straight pieces of every electrode and leaking conductor piece.

    Parameters
    ----------
    world
        World in the local frame of the solve.
    extra_segments
        Leaking pieces of galvanic distributed conductors as the pipeline
        discretises them (:class:`~groundfield.solver._discretize._Segment`
        with midpoint, length, direction and wire radius); each is a port
        of its own.
    """
    pieces: list[WirePiece] = []
    names = [e.name for e in world.electrodes]
    for k, e in enumerate(world.electrodes):
        shell = float(getattr(e, "concrete_shell_coefficient_ohm_m", 0.0) or 0.0)
        for line in _polylines(e):
            line = np.asarray(line, dtype=float)
            for a, b in zip(line[:-1], line[1:], strict=True):
                if np.linalg.norm(b - a) <= 1e-12:
                    continue
                pieces.append(
                    WirePiece(k, a.copy(), b.copy(), float(e.wire_radius), shell)
                )
    for seg in extra_segments:
        k = len(names)
        names.append(seg.electrode_name)
        mid = np.asarray(seg.midpoint, dtype=float)
        d = np.asarray(seg.direction, dtype=float)
        half = 0.5 * float(seg.length) * d / np.linalg.norm(d)
        pieces.append(WirePiece(k, mid - half, mid + half, float(seg.wire_radius), 0.0))
    return WireModel(names=tuple(names), pieces=pieces)


def axial_quadrature(model: WireModel, spacing: float) -> WireModel:
    """Two Gauss points per axial cell of at most ``spacing`` metres."""
    xg, wg = np.polynomial.legendre.leggauss(2)
    xs, ws, ks = [], [], []
    for k, p in enumerate(model.pieces):
        length = p.length
        n = max(1, math.ceil(length / spacing - 1e-9))
        s = ((np.arange(n)[:, None] + 0.5 * (xg[None, :] + 1.0)) / n).ravel()
        xs.append(p.start[None, :] + s[:, None] * (p.end - p.start)[None, :])
        ws.append(np.tile(0.5 * wg * length / n, n))
        ks.append(np.full(s.size, k, dtype=int))
    model.points = np.concatenate(xs)
    model.weights = np.concatenate(ws)
    model.piece = np.concatenate(ks)
    return model


def frames(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to each direction, shape ``(n, 3)`` each."""
    d = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    helper = np.where(
        (np.abs(d[:, 0]) < 0.9)[:, None],
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(d, e1)


def _line_integral(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """$\\int_a^b \\mathrm ds' / \\sqrt{|x - y(s')|^2 + c^2}$ for points ``x`` (n, 3)."""
    length = float(np.linalg.norm(b - a))
    e = (b - a) / length
    rel = x - a
    t = rel @ e
    p2 = np.maximum(np.einsum("ij,ij->i", rel, rel) - t * t, 0.0)
    s = np.sqrt(p2 + c * c)
    return np.arcsinh((length - t) / s) + np.arcsinh(t / s)


def radius_correction(
    model: WireModel, r_eta: float, rho_local: np.ndarray, reach: float
) -> np.ndarray:
    """$\\Delta C$ in Ω·m at every axial point (see the module docstring).

    Parameters
    ----------
    model
        Wires with their axial quadrature.
    r_eta
        Energy radius in m of the smoothed source.
    rho_local
        Resistivity in Ω·m around every axial point.
    reach
        Pieces farther than this from a point (and surface images farther
        than this) are left out; their two terms differ by
        $O((r_\\eta/d)^2)$.
    """
    from scipy.spatial import cKDTree

    x = model.points
    a_obs = np.array([p.radius for p in model.pieces])[model.piece]
    tree = cKDTree(x)
    total = np.zeros(x.shape[0])
    mirror = np.array([1.0, 1.0, -1.0])
    for p in model.pieces:
        ends = [(p.start, p.end)]
        if min(p.start[2], p.end[2]) < reach:
            ends.append((p.start * mirror, p.end * mirror))
        for a, b in ends:
            mid = 0.5 * (a + b)
            near = tree.query_ball_point(
                mid, 0.5 * float(np.linalg.norm(b - a)) + reach
            )
            if not near:
                continue
            idx = np.asarray(near, dtype=int)
            xi = x[idx]
            total[idx] += _line_integral(xi, a, b, a_obs[idx]) - _line_integral(
                xi, a, b, np.full(idx.size, r_eta)
            )
    return rho_local / (4.0 * math.pi) * total
