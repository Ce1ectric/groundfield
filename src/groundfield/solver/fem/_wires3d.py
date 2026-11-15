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
wire counts whole: its polyline through every corner where it has turned by
less than 120° (a ring's polygon, a strip, the corners of a loop) and every
piece on the same axis line (the next piece of a galvanic conductor, the
surface image of a rod whose head reaches the surface) — along its own axis
the mesh sees the smoothed source, not the thin wire, and only the whole
line integral reproduces the 2-D radial resistance. A piece of the own
polyline that runs back towards the point (turned by 120° or more: an arm
folded back, the opposite side of a narrow loop or a small ring, either arm
of a narrow V) or runs parallel beside it counts like a wire of the same
electrode (audit W4R2B-02: up to 2026.11 only an exactly parallel arm did,
so tilting a folded arm by 1e-6 rad added the whole line term). Of every
other piece of the same electrode, and of every surface image that does not
continue the observer's axis, only the part within $1.5\\,R$ of the point
counts (:data:`NEAR_FACTOR`): farther away the disc average of a smoothed
line is its line field (the 2-D mean-value property of the radial bump), so
the mesh gives the exact field and the reduced kernels would only add
$\\frac{\\rho}{4\\pi}\\ln\\frac{D^2 + r_\\eta^2}{D^2 + a^2}$ per metre with the
observer's current. A piece of **another port** counts only where it lies on
the observer's axis line (whole) or belongs to a wire that meets the
observer's wire (:func:`wire_systems`: a rod on a ring, a T-joint, a
crossing, the bonded pieces of a galvanic conductor at a bend — wires that
touch are one conductor and carry about the same current there); elsewhere
it carries its own current, which the observer's correction cannot know
(audit W4R2B-03: two coaxial rings 7.5 cm apart read +4.35e-3 in a port
entry with the window term). The 3-D model keeps such wires at least
$1.5\\,R$ apart (:func:`beside_wires`), so none of them reaches the window.
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
from typing import TYPE_CHECKING, NamedTuple

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
#: still keep 2 % of it (evidence ``f01`` of stream fw4fem3d). Wires that run
#: beside each other are kept this many source radii apart (audit W4R2B-03,
#: :func:`beside_wires`), so the window never reaches the side of another
#: wire.
NEAR_FACTOR = 1.5

#: The observer's own polyline counts whole only where it has turned by less
#: than 120° (``d_obs · e > FOLD_COS``); a piece turned farther runs back
#: towards the point and counts like a wire of the same electrode (audit
#: W4R2B-02). A rule that clipped the own chain by its distance from the
#: observer's axis line, continuous in the angle, also clipped the curved
#: continuation of a ring: −1.19e-3 against the solid torus for a ring of
#: r = 0.3 m (−4.9e-4 with this rule), −2.0e-3 for r = 1 m (evidence ``g02`` of
#: stream fw4r2b).
FOLD_COS = -0.5

#: Two wires run *beside each other* where the nearest point of one to a
#: point of the other lies on its side (not at an end) and the two run within
#: this angle (sine) of parallel: 30°. Wires that meet or cross at a larger
#: angle are close only over a few source radii (:func:`beside_wires`).
BESIDE_SINE = 0.5


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
        correction counts the observer's own wire whole where it has turned
        by less than 120° (:data:`FOLD_COS`).
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
    s = np.sqrt(_axis_distance2(rel, e) + c * c)
    top = length if hi is None else hi
    return np.arcsinh((top - t) / s) - np.arcsinh((lo - t) / s)


def _axis_distance2(rel: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Squared distance in m² of points ``rel`` (n, 3) from the axis line ``e``.

    From the cross product, $|\\mathbf r \\times \\mathbf e|^2$: the
    difference $|\\mathbf r|^2 - (\\mathbf r\\cdot\\mathbf e)^2$ cancels to
    about $|\\mathbf r|^2\\varepsilon$ on the axis, which exceeded the
    collinear tolerance $(10^{-6}\\cdot 1.5\\,R)^2$ from about 7 m along a piece
    that is not axis-aligned and clipped the observer's own piece (audit
    W4R2B-01: −1.63 % of the correction at such points).
    """
    c = np.cross(rel, e)
    return np.einsum("ij,ij->i", c, c)


def _segment_distances(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray
) -> np.ndarray:
    """Smallest distance in m between segments ``p0 p1`` and ``q0 q1`` (rows).

    Closed form of the clamped closest points (Ericson 2005, §5.1.9) for
    arrays of shape ``(n, 3)``; every segment has a positive length.
    """
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(
            denom > 1e-12 * a * e, np.clip((b * f - c * e) / denom, 0.0, 1.0), 0.0
        )
        t = (b * s + f) / e
        low, high = t < 0.0, t > 1.0
        s = np.where(low, np.clip(-c / a, 0.0, 1.0), s)
        s = np.where(high, np.clip((b - c) / a, 0.0, 1.0), s)
    t = np.clip(t, 0.0, 1.0)
    gap = p0 + s[:, None] * d1 - (q0 + t[:, None] * d2)
    return np.sqrt(np.einsum("ij,ij->i", gap, gap))


def _chains(model: "WireModel") -> np.ndarray:
    """Chain index of every piece (a piece without a chain its own)."""
    return np.array(
        [-(k + 1) if p.chain is None else p.chain for k, p in enumerate(model.pieces)]
    )


def wire_systems(model: "WireModel") -> np.ndarray:
    """Label of the connected wire system of every piece.

    Two chains (wires) belong to one system where their axes come within the
    smaller of their wire radii of each other — they meet: a rod on a ring,
    a T-joint, a crossing in one plane, a galvanic conductor and the
    electrode it is attached to, the pieces of a galvanic conductor — and
    the relation is closed transitively. Wires that touch are one conductor;
    their currents agree where they meet, so the radius correction may count
    a piece of another port of the same system with the observer's current
    (:func:`radius_correction`).
    """
    pieces = model.pieces
    m = len(pieces)
    chains = _chains(model)
    keys = {c: i for i, c in enumerate(dict.fromkeys(chains.tolist()))}
    parent = list(range(len(keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if m > 1:
        start = np.array([p.start for p in pieces])
        end = np.array([p.end for p in pieces])
        radius = np.array([p.radius for p in pieces])
        lo = np.minimum(start, end) - radius[:, None]
        hi = np.maximum(start, end) + radius[:, None]
        for i in range(m - 1):
            j = np.arange(i + 1, m)
            j = j[(chains[j] != chains[i])]
            if j.size == 0:
                continue
            j = j[np.all((lo[j] <= hi[i]) & (hi[j] >= lo[i]), axis=1)]
            if j.size == 0:
                continue
            d = _segment_distances(
                np.broadcast_to(start[i], (j.size, 3)),
                np.broadcast_to(end[i], (j.size, 3)),
                start[j],
                end[j],
            )
            for k in j[d <= np.minimum(radius[i], radius[j]) + 1e-12]:
                a, b = find(keys[int(chains[i])]), find(keys[int(chains[k])])
                if a != b:
                    parent[b] = a
    return np.array([find(keys[int(c)]) for c in chains], dtype=int)


class Beside(NamedTuple):
    """The closest pair of wires that run beside each other (:func:`beside_wires`)."""

    #: Distance in m of the two wire axes.
    distance: float
    #: Port index of the wire whose point lies beside the other wire.
    port: int
    #: Port index of the other wire (the same for a folded arm).
    other: int
    #: ``"beside"`` (side by side within 30° of parallel), ``"folded"`` (an
    #: arm of the same polyline) or ``"around"`` (the whole wire lies within
    #: the reach of another wire it does not meet).
    kind: str


def beside_wires(model: WireModel, reach: float) -> Beside | None:
    """The closest wires that run beside each other within ``reach`` m.

    The 3-D model keeps such wires at least :data:`NEAR_FACTOR` source radii
    apart (:func:`groundfield.solver.fem._solid.source_radius`; audits
    W4R2B-03, W4R2A-01): closer, the discs of their smoothed sources reach
    the other wire, the mesh no longer gives the other wire's exact field
    and the radius correction has no term for it. Two wires run beside each
    other where

    * a point of one lies within ``reach`` of the *side* of a piece of the
      other — the nearest point of that piece is not one of its ends — and
      the two pieces run within 30° of parallel (:data:`BESIDE_SINE`): wires
      side by side, concentric rings, an arm of the same polyline folded back
      beside the point (kind ``"folded"``), the two arms of a V narrower than
      30°; or
    * the whole of one electrode's wire lies within ``reach`` of another wire
      that it does not meet (:func:`wire_systems`; kind ``"around"``): a
      small ring around a rod's head.

    Left alone: pieces on one axis line (the collinear pieces of a galvanic
    conductor, stacked rods), wires that meet or cross at more than 30° (a
    rod on a ring, a T-joint, a crossing, a corner, the attachment of a
    galvanic conductor) — near them a point is close to the other wire's end
    or crosses it within a few source radii — and the consecutive chords of a
    ring.

    Returns
    -------
    Beside or None
        The configuration with the smallest axis distance below ``reach``
        (the distance for ``"around"`` is the farthest point of the wire from
        the other wire, so the whole wire no longer lies within 1.5 radii of
        the source radius it leads to), ``None`` if there is none.
    """
    pieces = model.pieces
    m = len(pieces)
    if m < 2 or not reach > 0.0:
        return None
    start = np.array([p.start for p in pieces])
    end = np.array([p.end for p in pieces])
    length = np.array([p.length for p in pieces])
    direction = (end - start) / length[:, None]
    ports = np.array([p.port for p in pieces], dtype=int)
    chains = _chains(model)
    systems = wire_systems(model)
    # sample points inside every piece, spacing reach / 12
    spacing = reach / 12.0
    pts, owner = [], []
    for k in range(m):
        n = max(4, math.ceil(length[k] / spacing))
        s = (np.arange(n) + 0.5) / n
        pts.append(start[k] + s[:, None] * (end[k] - start[k]))
        owner.append(np.full(n, k, dtype=int))
    x_all = np.concatenate(pts)
    own_all = np.concatenate(owner)
    lo = np.minimum(start, end) - reach
    hi = np.maximum(start, end) + reach
    best: Beside | None = None
    chunk = max(1, 400_000 // m)
    # the farthest point of every chain from each other chain (row: chain)
    labels = np.unique(chains)
    col = np.searchsorted(labels, chains)
    farthest = np.zeros((labels.size, labels.size))
    for i0 in range(0, x_all.shape[0], chunk):
        x = x_all[i0 : i0 + chunk]
        own = own_all[i0 : i0 + chunk]
        box_lo, box_hi = x.min(axis=0), x.max(axis=0)
        cand = np.nonzero(np.all((lo <= box_hi) & (hi >= box_lo), axis=1))[0]
        if cand.size == 0:
            continue
        rel = x[:, None, :] - start[cand][None, :, :]
        tau = np.einsum("ijk,jk->ij", rel, direction[cand])
        foot = np.clip(tau, 0.0, length[cand][None, :])
        gap = rel - foot[:, :, None] * direction[cand][None, :, :]
        dist = np.linalg.norm(gap, axis=2)
        sine = np.linalg.norm(
            np.cross(direction[own][:, None, :], direction[cand][None, :, :]), axis=2
        )
        to_line = np.linalg.norm(
            rel - tau[:, :, None] * direction[cand][None, :, :], axis=2
        )
        collinear = (sine < 1e-6) & (to_line < 1e-6 * reach)
        other = (cand[None, :] != own[:, None]) & ~collinear
        tol = 1e-9 * length[cand][None, :]
        side = (tau > tol) & (tau < length[cand][None, :] - tol)
        hit = other & side & (sine < BESIDE_SINE) & (dist < reach)
        if np.any(hit):
            d = np.where(hit, dist, np.inf)
            i, j = np.unravel_index(int(np.argmin(d)), d.shape)
            if best is None or d[i, j] < best.distance:
                p_, q_ = int(own[i]), int(cand[j])
                kind = "folded" if chains[p_] == chains[q_] else "beside"
                best = Beside(float(d[i, j]), int(ports[p_]), int(ports[q_]), kind)
        # the distance of every point from each chain (inf: beyond reach)
        order = np.argsort(col[cand], kind="stable")
        uniq, first_col = np.unique(col[cand][order], return_index=True)
        to_chain = np.full((x.shape[0], labels.size), np.inf)
        to_chain[:, uniq] = np.minimum.reduceat(
            np.where(other, dist, np.inf)[:, order], first_col, axis=1
        )
        c_own = col[own]
        for c1 in np.unique(c_own):
            far = to_chain[c_own == c1].max(axis=0)
            farthest[c1] = np.maximum(farthest[c1], far)
    first = {int(c): k for k, c in reversed(list(enumerate(col)))}
    np.fill_diagonal(farthest, np.inf)
    for c1, c2 in zip(*np.nonzero(farthest < reach), strict=True):
        k1, k2 = first[int(c1)], first[int(c2)]
        far = float(farthest[c1, c2])
        if systems[k1] == systems[k2]:
            continue
        if best is None or far < best.distance:
            best = Beside(far, int(ports[k1]), int(ports[k2]), "around")
    return best


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
        Distance in m (:data:`NEAR_FACTOR` times the source radius). Every
        piece on the observer's axis line counts whole (the next piece of a
        galvanic conductor, a rod's surface image). The observer's own
        polyline counts whole where it has turned by less than 120°
        (:data:`FOLD_COS`) and is not parallel beside the point; of the rest
        of its electrode, of a piece of another port whose wire meets the
        observer's wire (:func:`wire_systems`) and of the surface images only
        the part within ``near`` of the point counts. Every other piece of
        another port counts nothing (audit W4R2B-03).
    """
    x = model.points
    n = x.shape[0]
    pieces = model.pieces
    a_obs = np.array([p.radius for p in pieces])[model.piece]
    chains = _chains(model)
    ports = np.array([p.port for p in pieces], dtype=int)
    systems = wire_systems(model)
    chain_obs = chains[model.piece]
    port_obs = ports[model.piece]
    system_obs = systems[model.piece]
    d_obs = np.array([p.direction for p in pieces])[model.piece]
    on_axis = 1e-6 * near  # m; collinear pieces share the axis to round-off
    total = np.zeros(n)
    mirror = np.array([1.0, 1.0, -1.0])
    for k, p in enumerate(pieces):
        # the window counts for the observer's electrode and for wires that
        # meet the observer's wire (one conductor), not for other ports
        window = (port_obs == ports[k]) | (system_obs == systems[k])
        for image in (False, True):
            a, b = (p.start * mirror, p.end * mirror) if image else (p.start, p.end)
            length = float(np.linalg.norm(b - a))
            e = (b - a) / length
            rel = x - a
            t = rel @ e
            p2 = _axis_distance2(rel, e)
            parallel = np.linalg.norm(np.cross(d_obs, e), axis=1) < 1e-6
            collinear = parallel & (p2 < on_axis**2)
            if image:
                whole = collinear
            else:  # own wire, not where it runs back or beside the point
                whole = collinear | (
                    (chain_obs == chains[k]) & ~parallel & ((d_obs @ e) > FOLD_COS)
                )
            half = np.sqrt(np.maximum(near * near - p2, 0.0))
            lo = np.where(whole, 0.0, np.clip(t - half, 0.0, length))
            hi = np.where(whole, length, np.clip(t + half, 0.0, length))
            hi = np.where(whole | window, hi, lo)
            idx = np.nonzero(hi > lo)[0]
            if idx.size == 0:
                continue
            xi = x[idx]
            total[idx] += _line_integral(
                xi, a, b, a_obs[idx], lo[idx], hi[idx]
            ) - _line_integral(xi, a, b, np.full(idx.size, r_eta), lo[idx], hi[idx])
    return rho_local / (4.0 * math.pi) * total
