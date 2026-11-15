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

summed over what the mesh does not already resolve. The observer's own
wire counts whole: its polyline through every corner (a ring's polygon, a
strip, a closed loop) and every piece on the same axis line (the next piece
of a galvanic conductor, the surface image of a rod whose head reaches the
surface) — along its own axis the mesh sees the smoothed source, not the
thin wire, and only the whole line integral reproduces the 2-D radial
resistance. A piece of the own polyline that runs parallel beside the
point (an arm folded back, the opposite side of a narrow loop) counts like
another wire. Of every other piece, and of every other surface image, only
the part within $1.5\\,R$ of the point counts (:data:`NEAR_FACTOR`): farther
away the disc average of a smoothed line is its line field (the 2-D
mean-value property of the radial bump), so the mesh gives the exact field
and the reduced kernels would only add $\\frac{\\rho}{4\\pi}
\\ln\\frac{D^2 + r_\\eta^2}{D^2 + a^2}$ per metre with the observer's current.
For two parallel lines the exact correction is below $10^{-3}$ of the self
term from $D = 1.5\\,R$ on (evidence ``f01`` of stream fw4fem3d); the
surface image of a horizontal wire lies $2z \\ge 2.5\\,R$ away and never
counts (audit W4B-01: up to 2026.11 the surface images and every wire within
$20\\,R$ counted, +1.9e-3 for a ring 6 cm deep, +4.6e-3 for two rings 10 cm
apart). Interface images are left out for the same reason. Far inside a
straight wire this is Sunde's $\\frac{\\rho}{2\\pi}\\ln(r_\\eta/a)$; near a free
end, a corner or a junction it carries the three-dimensional geometry.
The smoothed source differs from the reduced kernel only in how its end
region averages (the bump's pair distances instead of $r_\\eta$); that
difference integrates to $0.084\\,R$ along the wire per free end
(evidence ``e01`` of stream w4fem3d), about $10^{-4}$ of a 3 m rod at
$R = 4$ cm, and is neglected.
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

#: The radius correction counts a piece of another wire, and a surface image
#: that does not continue the observer's own axis, only where it lies within
#: this many source radii $R$ of the observer (audit W4B-01). Beyond it the
#: mesh gives the exact field: for two parallel lines the exact correction is
#: below 1e-3 of the self term from $1.5\\,R$ on, while a cut at $2\\,R$ would
#: still keep 2 % of it (evidence ``f01`` of stream fw4fem3d).
NEAR_FACTOR = 1.5


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
    chain
        Index of the wire the piece belongs to (one polyline of one
        electrode, a ring's polygon); ``None``: a wire of its own. The radius
        correction counts the observer's own wire whole.
    """

    port: int
    start: np.ndarray
    end: np.ndarray
    radius: float
    shell: float
    chain: int | None = None

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
    intervals
        The stretch of its piece each point stands for (half an axial cell),
        as distances in m from the piece's start, shape ``(n, 2)``.
    """

    names: tuple[str, ...]
    pieces: list[WirePiece]
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    piece: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    intervals: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

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
    chain = 0
    for k, e in enumerate(world.electrodes):
        shell = float(getattr(e, "concrete_shell_coefficient_ohm_m", 0.0) or 0.0)
        for line in _polylines(e):
            line = np.asarray(line, dtype=float)
            for a, b in zip(line[:-1], line[1:], strict=True):
                if np.linalg.norm(b - a) <= 1e-12:
                    continue
                pieces.append(
                    WirePiece(k, a.copy(), b.copy(), float(e.wire_radius), shell, chain)
                )
            chain += 1
    for seg in extra_segments:
        k = len(names)
        names.append(seg.electrode_name)
        mid = np.asarray(seg.midpoint, dtype=float)
        d = np.asarray(seg.direction, dtype=float)
        half = 0.5 * float(seg.length) * d / np.linalg.norm(d)
        pieces.append(
            WirePiece(k, mid - half, mid + half, float(seg.wire_radius), 0.0, chain)
        )
        chain += 1
    return WireModel(names=tuple(names), pieces=pieces)


def axial_quadrature(model: WireModel, spacing: float) -> WireModel:
    """Two Gauss points per axial cell of at most ``spacing`` metres."""
    xg, wg = np.polynomial.legendre.leggauss(2)
    xs, ws, ks, iv = [], [], [], []
    for k, p in enumerate(model.pieces):
        length = p.length
        n = max(1, math.ceil(length / spacing - 1e-9))
        s = ((np.arange(n)[:, None] + 0.5 * (xg[None, :] + 1.0)) / n).ravel()
        xs.append(p.start[None, :] + s[:, None] * (p.end - p.start)[None, :])
        ws.append(np.tile(0.5 * wg * length / n, n))
        ks.append(np.full(s.size, k, dtype=int))
        edges = np.arange(2 * n + 1) * (0.5 * length / n)
        iv.append(np.stack([edges[:-1], edges[1:]], axis=1))
    model.points = np.concatenate(xs)
    model.weights = np.concatenate(ws)
    model.piece = np.concatenate(ks)
    model.intervals = np.concatenate(iv)
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
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    lo: np.ndarray | float = 0.0,
    hi: np.ndarray | float | None = None,
) -> np.ndarray:
    """$\\int \\mathrm ds' / \\sqrt{|x - y(s')|^2 + c^2}$ over $s' \\in [lo, hi]$.

    ``y(s') = a + s' (b - a)/|b - a|`` for points ``x`` (n, 3); ``lo`` and
    ``hi`` in m along the piece (default: the whole piece).
    """
    length = float(np.linalg.norm(b - a))
    e = (b - a) / length
    rel = x - a
    t = rel @ e
    p2 = np.maximum(np.einsum("ij,ij->i", rel, rel) - t * t, 0.0)
    s = np.sqrt(p2 + c * c)
    top = length if hi is None else hi
    return np.arcsinh((top - t) / s) - np.arcsinh((lo - t) / s)


def radius_correction(
    model: WireModel, r_eta: float, rho_local: np.ndarray, near: float
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
    near
        Distance in m (:data:`NEAR_FACTOR` times the source radius): of a
        piece that is not the observer's own wire, and of a surface image
        that does not continue the observer's axis, only the part within
        ``near`` of the observer counts. The observer's own wire (its chain
        and every piece on its axis line) counts whole, except a piece of
        the chain parallel to the observer's piece and off its axis (an arm
        folded back beside it, a loop's opposite side), which counts like
        another wire.
    """
    x = model.points
    n = x.shape[0]
    pieces = model.pieces
    a_obs = np.array([p.radius for p in pieces])[model.piece]
    chains = np.array(
        [-(k + 1) if p.chain is None else p.chain for k, p in enumerate(pieces)]
    )
    chain_obs = chains[model.piece]
    d_obs = np.array([p.direction for p in pieces])[model.piece]
    on_axis = 1e-6 * near  # m; collinear pieces share the axis to round-off
    total = np.zeros(n)
    mirror = np.array([1.0, 1.0, -1.0])
    for k, p in enumerate(pieces):
        for image in (False, True):
            a, b = (p.start * mirror, p.end * mirror) if image else (p.start, p.end)
            length = float(np.linalg.norm(b - a))
            e = (b - a) / length
            rel = x - a
            t = rel @ e
            p2 = np.maximum(np.einsum("ij,ij->i", rel, rel) - t * t, 0.0)
            parallel = np.linalg.norm(np.cross(d_obs, e), axis=1) < 1e-6
            collinear = parallel & (p2 < on_axis**2)
            if image:
                whole = collinear
            else:  # own wire, but not an arm of it folded back beside the point
                whole = collinear | ((chain_obs == chains[k]) & ~parallel)
            half = np.sqrt(np.maximum(near * near - p2, 0.0))
            lo = np.where(whole, 0.0, np.clip(t - half, 0.0, length))
            hi = np.where(whole, length, np.clip(t + half, 0.0, length))
            idx = np.nonzero(hi > lo)[0]
            if idx.size == 0:
                continue
            xi = x[idx]
            total[idx] += _line_integral(
                xi, a, b, a_obs[idx], lo[idx], hi[idx]
            ) - _line_integral(xi, a, b, np.full(idx.size, r_eta), lo[idx], hi[idx])
    return rho_local / (4.0 * math.pi) * total
