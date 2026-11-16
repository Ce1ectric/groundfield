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
wire counts whole: its polyline as far as it has turned by at most 90°
along the chain (a ring's polygon, a strip, the corners of a loop;
:data:`TURN_WHOLE`) and every piece on the same axis line that belongs to
the observer's port or wire system (the next piece of a galvanic conductor,
the surface image of a rod whose head reaches the surface) — along its own
axis the mesh sees the smoothed source, not the thin wire, and only the
whole line integral reproduces the 2-D radial resistance. Where the own
polyline has turned farther it runs back towards the point (an arm folded
back, the opposite side of a narrow loop or a small ring, either arm of a
narrow V) or alongside it after a full turn (the next arm of a meander, the
next winding of a spiral): from a turn of 150° on it counts like a wire of
the same electrode, and between 90° and 150° by a weight linear in the turn
(audits W4R2B-02, W4R3B-03: up to 2026.11 a direction within 120° of the
observer's counted whole, so the meander's third arm added +6.8e-3 and a V
jumped at 120°). Of every other piece of the same electrode, and of every
surface image that does not continue the observer's axis, only the part
within $1.5\\,R$ of the point counts (:data:`NEAR_FACTOR`): farther away the
disc average of a smoothed line is its line field (the 2-D mean-value
property of the radial bump), so the mesh gives the exact field and the
reduced kernels would only add
$\\frac{\\rho}{4\\pi}\\ln\\frac{D^2 + r_\\eta^2}{D^2 + a^2}$ per metre with the
observer's current. A piece of **another port** counts only where it
belongs to a wire that meets the observer's wire (:func:`wire_systems`:
their surfaces touch — a rod on a ring, a T-joint, a crossing, the pieces of
a galvanic conductor — wires that touch are one conductor and carry about
the same current there), whole on the observer's axis line, by its window
elsewhere, and by a window fading out with the surface gap where the two
have just stopped touching (:data:`TAPER`, audit W4R3B-04); every other
piece carries its own current, which the observer's correction cannot know
(audit W4R2B-03: two coaxial rings 7.5 cm apart read +4.35e-3 in a port
entry with the window term; audit W4R3B-01: a collinear rod of another
electrode 1 cm below counted whole). The 3-D model keeps wires that do not
meet but run beside each other at least $1.5\\,R$ apart, and wires end to end
at least $3\\,R$ (:func:`beside_wires`), so none of them reaches the window;
wires that meet at less than 10° count as not meeting there. An encased wire
meets others by its outer shell radius $r_b$ (what the soil sees): two encased
wires of different ports whose shells touch are one conductor in the
correction, and their separate port entries are not resolved.
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

#: The observer's own polyline counts whole up to the turn ``TURN_WHOLE``
#: (90°) accumulated along the chain from the observer's piece, like another
#: wire of the same electrode (its window only) from ``TURN_NONE`` (150°) on,
#: and in between by the weight $(150° - T)/60°$ of whole and the rest by its
#: window (audit W4R3B-03). The turn is the sum of the angles at the corners
#: passed, along a closed chain the shorter way round: an S-meander's third arm
#: or the next winding of a spiral has turned by 360° although it runs the
#: same way. (Up to audit W4R3B-03 a piece counted whole where its direction
#: was within 120° of the observer's, ``d_obs · e > −1/2``: the meander's
#: third arm, 0.20 m away, added +6.8e-3 of the correction; and the switch at
#: 120° made a V of 0.5 m arms jump by −1.26e-3 in $Z$.) A rule that clipped
#: the own chain by its distance from the observer's axis line also clipped
#: the curved continuation of a ring (−1.19e-3 against the solid torus for
#: r = 0.3 m; evidence ``g02`` of stream fw4r2b).
TURN_WHOLE = 0.5 * math.pi
#: See :data:`TURN_WHOLE`.
TURN_NONE = 5.0 * math.pi / 6.0

#: Two wires meet where their axes come within the sum of their radii (their
#: surfaces touch), or, end to end in line, where the gap between their ends
#: is zero to this round-off tolerance in m (audits W4R3B-01, W4R3B-04).
TOUCH_TOL = 1e-9

#: The window term of a wire of another port that does not meet the
#: observer's fades out linearly from an axis distance of $a_1 + a_2$ (where
#: the surfaces stop touching, weight 1) to ``TAPER`` $(a_1 + a_2)$ (weight 0),
#: so the correction is continuous where two crossing wires stop meeting
#: (audit W4R3B-04: a hard switch at the smaller radius made a 2 m cross jump
#: by −7.5e-4 in $Z$, `mom` −1.6e-5).
TAPER = 2.0

#: Two wires run *beside each other* where the nearest point of one to a
#: point of the other lies on its side (not at an end) and the two run within
#: this angle (sine) of parallel: 30°. Wires that cross at a larger angle are
#: close only over a few source radii, wires that meet at any angle are one
#: conductor there (:func:`beside_wires`). Two pieces lie *in line* where the
#: gap between their nearest ends runs within this angle of both axes.
BESIDE_SINE = 0.5

#: Two wires end to end in line keep their gap at least this many source
#: radii (the coordinator's decision on audit W4R3B-01): at a gap of 1.5 R
#: the transfer resistance of stacked rods read 1.5 … 2.0 times outside its
#: estimate, from a gap of 3 R on inside it in every measured case (wire
#: radii of 2, 4 and 8 mm; evidence ``h09`` of stream fw4r3b).
IN_LINE_FACTOR = 3.0

#: Two wires that meet are left alone near where they meet only where they
#: meet at 10° or more (this sine; the coordinator's decision on audit
#: W4R3B-02). Below, nearly parallel or running together at the touching
#: distance, they run beside each other as wires that do not meet: two
#: strips touching along their length are not one conductor.
MEET_SINE = math.sin(math.radians(10.0))


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
        electrode, a ring's polygon; its pieces in order along it);
        ``None``: a wire of its own. The radius correction counts the
        observer's own wire whole up to a turn of 90° along it
        (:data:`TURN_WHOLE`).
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
    return _closest_points(p0, p1, q0, q1)[0]


def _closest_points(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distance, parameters and gap vector of the closest points of segments.

    Returns
    -------
    distance, s, t, gap
        The smallest distance in m, the parameters ``s`` of ``p0 p1`` and
        ``t`` of ``q0 q1`` (0 … 1, exactly 0 or 1 at an end) of the closest
        points, and the vector in m from the first to the second.
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
    gap = q0 + t[:, None] * d2 - (p0 + s[:, None] * d1)
    return np.sqrt(np.einsum("ij,ij->i", gap, gap)), s, t, gap


def _chains(model: "WireModel") -> np.ndarray:
    """Chain index of every piece (a piece without a chain its own)."""
    return np.array(
        [-(k + 1) if p.chain is None else p.chain for k, p in enumerate(model.pieces)]
    )


class _Contacts(NamedTuple):
    """Pairs of pieces ``i < j`` close to each other (:func:`_contacts`)."""

    #: Piece indices of every pair.
    i: np.ndarray
    j: np.ndarray
    #: Distance in m of the two axes (segments).
    distance: np.ndarray
    #: Whether the two lie in line: the gap between an end of each runs within
    #: 30° of both axes (:data:`BESIDE_SINE`) and is not zero.
    in_line: np.ndarray
    #: Gap in m between the two wire surfaces (:func:`_surface_gap`).
    surface_gap: np.ndarray
    #: 1 where the two meet (the surface gap is zero to :data:`TOUCH_TOL`),
    #: falling linearly to 0 at a surface gap of $a_i + a_j$
    #: (:data:`TAPER`); 0 in line with a gap.
    weight: np.ndarray


def _reduce(gap: np.ndarray, e: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``gap`` (n, 3) less up to ``a`` of its part perpendicular to ``e``."""
    axial = np.einsum("ij,ij->i", gap, e)[:, None] * e
    lateral = gap - axial
    size = np.linalg.norm(lateral, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        keep = np.where(size > a, (size - a) / size, 0.0)
    return axial + keep[:, None] * lateral


def _surface_gap(
    gap: np.ndarray, e_i: np.ndarray, a_i: np.ndarray, e_j: np.ndarray, a_j: np.ndarray
) -> np.ndarray:
    """Gap in m between two wire surfaces, from the gap of their axes.

    ``gap`` runs from the closest point of one axis to that of the other.
    Each wire reaches its radius from its axis in the plane perpendicular to
    it, up to its flat end: the gap less the disc of each wire — exactly
    $d - a_i - a_j$ where two wires cross (the gap is perpendicular to both),
    the end gap where two wires lie end to end on one axis, and the gap of a
    flat end face to the other wire's surface where a wire ends beside
    another (a rod's head below a small ring: 0.49 mm for auditor W4A's ring
    0.75 radii above the head, whose axes are 15.2 mm apart, within
    $a_i + a_j$ = 16 mm). Taken in both orders, the smaller.
    """
    one = _reduce(_reduce(gap, e_i, a_i), e_j, a_j)
    two = _reduce(_reduce(gap, e_j, a_j), e_i, a_i)
    return np.minimum(np.linalg.norm(one, axis=1), np.linalg.norm(two, axis=1))


def _contacts(model: "WireModel", reach: float = 0.0) -> _Contacts:
    """Every pair of pieces within ``reach`` m or :data:`TAPER` (a_i + a_j).

    The pieces of one chain are included (a polyline's consecutive pieces
    meet at their common vertex). Two pieces meet where their surfaces
    touch (:func:`_surface_gap`; audit W4R3B-04): where they cross, their
    axes come within $a_i + a_j$ (a crossing, a strip on a strip); where one
    ends at the other, its flat end reaches the other's surface (a rod on a
    ring, a T-joint); where they lie end to end on one axis (stacked rods,
    the pieces of a straight galvanic conductor), the gap between their ends
    is zero to :data:`TOUCH_TOL` (audit W4R3B-01: up to then any gap within
    the smaller radius met, so the other rod counted whole with the
    observer's current).
    """
    pieces = model.pieces
    m = len(pieces)
    empty = np.zeros(0)
    if m < 2:
        return _Contacts(
            empty.astype(int), empty.astype(int), empty, empty > 0, empty, empty
        )
    start = np.array([p.start for p in pieces])
    end = np.array([p.end for p in pieces])
    radius = np.array([p.radius for p in pieces])
    margin = np.maximum(TAPER * (radius + radius.max()), float(reach))
    lo = np.minimum(start, end) - margin[:, None]
    hi = np.maximum(start, end) + margin[:, None]
    lo_bare = np.minimum(start, end)
    hi_bare = np.maximum(start, end)
    out_i, out_j = [], []
    for i in range(m - 1):
        j = np.arange(i + 1, m)
        j = j[np.all((lo_bare[j] <= hi[i]) & (hi_bare[j] >= lo[i]), axis=1)]
        if j.size:
            out_i.append(np.full(j.size, i, dtype=int))
            out_j.append(j)
    if not out_i:
        return _Contacts(
            empty.astype(int), empty.astype(int), empty, empty > 0, empty, empty
        )
    i = np.concatenate(out_i)
    j = np.concatenate(out_j)
    d, s, t, gap = _closest_points(start[i], end[i], start[j], end[j])
    total = radius[i] + radius[j]
    keep = d <= np.maximum(TAPER * total, float(reach))
    i, j, d, s, t, gap = i[keep], j[keep], d[keep], s[keep], t[keep], gap[keep]
    total = total[keep]
    e_i = (end[i] - start[i]) / np.linalg.norm(end[i] - start[i], axis=1)[:, None]
    e_j = (end[j] - start[j]) / np.linalg.norm(end[j] - start[j], axis=1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        g = gap / d[:, None]
    ends = ((s <= 0.0) | (s >= 1.0)) & ((t <= 0.0) | (t >= 1.0))
    in_line = (
        (d > TOUCH_TOL)
        & ends
        & (np.linalg.norm(np.cross(g, e_i), axis=1) < BESIDE_SINE)
        & (np.linalg.norm(np.cross(g, e_j), axis=1) < BESIDE_SINE)
    )
    surface = _surface_gap(gap, e_i, radius[i], e_j, radius[j])
    weight = np.clip(1.0 - surface / ((TAPER - 1.0) * total), 0.0, 1.0)
    # in line with a gap: no window at all (beside_wires keeps such wires
    # 1.5 R apart, so it would never reach them)
    weight = np.where(in_line, 0.0, weight)
    weight = np.where(surface <= TOUCH_TOL, 1.0, weight)
    return _Contacts(i, j, d, in_line, surface, weight)


def _union(n: int, pairs_i: np.ndarray, pairs_j: np.ndarray) -> np.ndarray:
    """Connected components of ``n`` nodes under the given pairs (labels)."""
    parent = list(range(n))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a, b in zip(pairs_i.tolist(), pairs_j.tolist(), strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(k) for k in range(n)], dtype=int)


def wire_systems(model: "WireModel", contacts: _Contacts | None = None) -> np.ndarray:
    """Label of the connected wire system of every piece.

    Two chains (wires) belong to one system where two of their pieces meet
    (:func:`_contacts`: their surfaces touch — crossing wires $a_1 + a_2$
    axis to axis, a flat end at the other's surface, end to end on one axis
    only without a gap) — a rod on a ring, a T-joint, a crossing, a
    galvanic conductor and the electrode it is attached to, the pieces of a
    galvanic conductor — and the relation is closed transitively. Wires
    that touch are one conductor; their currents agree where they meet, so
    the radius correction may count a piece of another port of the same
    system with the observer's current (:func:`radius_correction`). (Up to
    audits W4R3B-01 and W4R3B-04 the axes had to come within the smaller
    radius: two crossing strips whose surfaces still overlapped stopped
    meeting, and two stacked rods with an end gap below the radius met.)
    """
    chains = _chains(model)
    keys = {c: i for i, c in enumerate(dict.fromkeys(chains.tolist()))}
    col = np.array([keys[int(c)] for c in chains], dtype=int)
    if contacts is None:
        contacts = _contacts(model)
    meet = contacts.weight >= 1.0
    labels = _union(len(keys), col[contacts.i[meet]], col[contacts.j[meet]])
    return labels[col]


def _system_weights(model: "WireModel", contacts: _Contacts, systems: np.ndarray):
    """Window weight between wire systems (audit W4R3B-04).

    1 within a system; between two systems the widest path of the tapered
    contacts (:func:`_contacts`, weights between 0 and 1): two wires that
    just stopped meeting keep their window term with a weight that fades out
    with their distance, continuous in the geometry. Returns the compact
    index of every system (``-1``: no tapered contact) and the weight matrix
    over the indexed systems.
    """
    n_sys = int(systems.max()) + 1 if systems.size else 0
    index = np.full(n_sys, -1, dtype=int)
    w = contacts.weight
    taper = (w > 0.0) & (w < 1.0)
    si, sj = systems[contacts.i[taper]], systems[contacts.j[taper]]
    between = si != sj
    si, sj, wt = si[between], sj[between], w[taper][between]
    if wt.size == 0:
        return index, np.zeros((0, 0))
    involved = np.unique(np.concatenate([si, sj]))
    index[involved] = np.arange(involved.size)
    a, b = index[si], index[sj]
    weights = np.eye(involved.size)
    members = {k: [k] for k in range(involved.size)}
    owner = list(range(involved.size))
    # Kruskal in descending weight: the edge that first joins two groups is
    # the bottleneck of the widest path between any of their members.
    for k in np.argsort(-wt, kind="stable"):
        ga, gb = owner[a[k]], owner[b[k]]
        if ga == gb:
            continue
        ma, mb = members[ga], members.pop(gb)
        weights[np.ix_(ma, mb)] = wt[k]
        weights[np.ix_(mb, ma)] = wt[k]
        for x in mb:
            owner[x] = ga
        ma.extend(mb)
    return index, weights


def _turns(model: "WireModel") -> tuple[np.ndarray, np.ndarray]:
    """Turn accumulated along every chain, and the total of closed chains.

    Returns
    -------
    turn, total
        Radians, per piece: the sum of the angles at the corners from the
        chain's first piece to this one, and the whole turn of a closed
        chain (its last piece ends where its first starts, the closing corner
        included; ``inf`` for an open chain).
    """
    pieces = model.pieces
    m = len(pieces)
    chains = _chains(model)
    direction = np.array([p.direction for p in pieces]).reshape(m, 3)
    order = np.argsort(chains, kind="stable")
    ch = chains[order]
    d = direction[order]
    same = ch[1:] == ch[:-1]
    cos = np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]), -1.0, 1.0)
    angle = np.where(same, np.arccos(cos), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(angle)])
    first = np.concatenate([[True], ~same])
    group = np.cumsum(first) - 1
    cum = cum - cum[first][group]
    last = np.concatenate([~same, [True]])
    turn = np.empty(m)
    turn[order] = cum
    total = np.full(m, np.inf)
    for g0, g1 in zip(np.nonzero(first)[0], np.nonzero(last)[0], strict=True):
        if g1 == g0:
            continue
        a, b = pieces[order[g0]], pieces[order[g1]]
        if np.linalg.norm(a.start - b.end) <= TOUCH_TOL:
            closing = math.acos(max(-1.0, min(1.0, float(b.direction @ a.direction))))
            total[order[g0 : g1 + 1]] = cum[g1] + closing
    return turn, total


class Beside(NamedTuple):
    """The closest pair of wires that run beside each other (:func:`beside_wires`)."""

    #: Distance in m of the two wire axes.
    distance: float
    #: Port index of the wire whose point lies beside the other wire.
    port: int
    #: Port index of the other wire (the same for a folded arm).
    other: int
    #: ``"beside"`` (side by side within 30° of parallel), ``"folded"`` (an
    #: arm of the same polyline), ``"around"`` (the whole wire lies within
    #: the reach of another wire it does not meet) or ``"in line"`` (end to
    #: end with a gap, audit W4R3B-01).
    kind: str
    #: Angle in degrees at which the two wires meet, where they touch but meet
    #: at less than 10° (:data:`MEET_SINE`); ``None`` otherwise.
    angle: float | None = None

    @property
    def limit(self) -> float:
        """The source radius in m these wires allow.

        Their distance over :data:`IN_LINE_FACTOR` for wires end to end,
        over :data:`NEAR_FACTOR` otherwise.
        """
        factor = IN_LINE_FACTOR if self.kind == "in line" else NEAR_FACTOR
        return self.distance / factor


def _runs(model: "WireModel", contacts: _Contacts) -> np.ndarray:
    """Label of the straight run of every piece.

    A run is a chain of pieces on one axis line that meet end to end (the
    pieces of a straight galvanic conductor, a polyline's vertex on a
    straight stretch); every other piece is a run of its own.
    """
    pieces = model.pieces
    direction = np.array([p.direction for p in pieces]).reshape(len(pieces), 3)
    i, j = contacts.i, contacts.j
    parallel = np.linalg.norm(np.cross(direction[i], direction[j]), axis=1) < 1e-6
    keep = (contacts.distance <= TOUCH_TOL) & parallel
    return _union(len(pieces), i[keep], j[keep])


def beside_wires(model: WireModel, reach: float) -> Beside | None:
    """The closest wires that run beside each other within ``reach`` m.

    ``reach`` is :data:`NEAR_FACTOR` source radii. The 3-D model keeps such
    wires at least that far apart (:func:`groundfield.solver.fem._solid.source_radius`;
    audits W4R2B-03, W4R2A-01): closer, the discs of their smoothed sources
    reach the other wire, the mesh no longer gives the other wire's exact
    field and the radius correction has no term for it; wires end to end in
    line it keeps :data:`IN_LINE_FACTOR` source radii apart (audit W4R3B-01).
    Two wires run beside each other where

    * a point of one lies within ``reach`` of the *side* of a piece of the
      other — the nearest point of that piece is not one of its ends — and
      the two pieces run within 30° of parallel (:data:`BESIDE_SINE`): wires
      side by side, concentric rings, an arm of the same polyline folded back
      beside the point (kind ``"folded"``), a galvanic conductor in the
      trench of an earthing strip; or
    * the whole of one electrode's wire lies within ``reach`` of another wire
      that it does not meet (:func:`wire_systems`; kind ``"around"``): a
      small ring around a rod's head; or
    * two wires that do not meet lie in line, end to end with a gap below
      :data:`IN_LINE_FACTOR` source radii (kind ``"in line"``: stacked rods
      of two electrodes, audit W4R3B-01; up to then they were left alone,
      and rods 1 cm apart read +1.6e-2 in a port entry and −7.3e-2 in the
      transfer entry).

    Left alone: pieces on one axis line that meet (the collinear pieces of a
    galvanic conductor, a rod cut into touching pieces), wires that meet at
    10° or more (:data:`MEET_SINE`; a rod on a ring, a T-joint, a crossing,
    a corner, the attachment of a galvanic conductor, a V or a branch of 10
    … 30°) — two pieces whose straight runs meet at an angle θ come within
    ``reach`` of each other only within ``reach`` / sin θ of where they
    meet, where the window term counts the other wire, whose current agrees
    there (audit W4R3B-02: up to then a V narrower than 30° was refused with
    the distance of the sample point nearest to the apex) — wires that cross
    at more than 30°, and the consecutive chords of a ring. Wires that meet
    at less than 10°, nearly parallel or touching along their length, run
    beside each other like wires that do not meet; the result names their
    angle.

    Returns
    -------
    Beside or None
        The configuration that allows the smallest source radius
        (:attr:`Beside.limit`): for ``"beside"``, ``"folded"`` and ``"in
        line"`` the distance of the two pieces' axes, independent of the
        sampling; for ``"around"`` the farthest point of the wire from the
        other wire, so the whole wire no longer lies within 1.5 radii of the
        source radius it leads to. ``None`` if there is none.
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
    reach_line = reach * IN_LINE_FACTOR / NEAR_FACTOR
    contacts = _contacts(model, reach_line)
    systems = wire_systems(model, contacts)
    runs = _runs(model, contacts)
    meet = contacts.weight >= 1.0
    n_runs = int(runs.max()) + 1
    touching = np.unique(
        np.concatenate(
            [
                runs[contacts.i[meet]] * n_runs + runs[contacts.j[meet]],
                runs[contacts.j[meet]] * n_runs + runs[contacts.i[meet]],
            ]
        )
    )
    best: Beside | None = None
    hits_p: list[np.ndarray] = []
    hits_q: list[np.ndarray] = []
    hits_d: list[np.ndarray] = []
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
        # pieces whose straight runs meet at 10° or more are left alone
        # (audit W4R3B-02 and the coordinator's minimum angle)
        met = np.isin(runs[own][:, None] * n_runs + runs[cand][None, :], touching)
        met &= sine >= MEET_SINE * (1.0 - 1e-9)
        hit = other & side & ~met & (sine < BESIDE_SINE) & (dist < reach)
        if np.any(hit):
            i, j = np.nonzero(hit)
            hits_p.append(own[i])
            hits_q.append(cand[j])
            hits_d.append(dist[i, j])
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
    if hits_p:
        # the distance of the two pieces' axes, not of the nearest sample
        # (audit W4R3B-02); among equal distances the pair of the nearest
        # sample names the wires
        pairs, where = np.unique(
            np.stack([np.concatenate(hits_p), np.concatenate(hits_q)], axis=1),
            axis=0,
            return_inverse=True,
        )
        sampled = np.full(pairs.shape[0], np.inf)
        np.minimum.at(sampled, where.ravel(), np.concatenate(hits_d))
        d = _segment_distances(
            start[pairs[:, 0]], end[pairs[:, 0]], start[pairs[:, 1]], end[pairs[:, 1]]
        )
        tie = d <= d.min() * (1.0 + 1e-9)
        k = int(np.argmin(np.where(tie, sampled, np.inf)))
        p_, q_ = int(pairs[k, 0]), int(pairs[k, 1])
        kind = "folded" if chains[p_] == chains[q_] else "beside"
        angle = None
        if np.isin(runs[p_] * n_runs + runs[q_], touching):
            sin_pq = float(np.linalg.norm(np.cross(direction[p_], direction[q_])))
            angle = math.degrees(math.asin(min(1.0, sin_pq)))
        best = Beside(float(d[k]), int(ports[p_]), int(ports[q_]), kind, angle)
    # wires in line, end to end, that do not meet (audit W4R3B-01)
    line = (
        contacts.in_line
        & (contacts.distance < reach_line)
        & (systems[contacts.i] != systems[contacts.j])
    )
    if np.any(line):
        k = int(np.argmin(np.where(line, contacts.distance, np.inf)))
        i, j = int(contacts.i[k]), int(contacts.j[k])
        found = Beside(
            float(contacts.distance[k]), int(ports[i]), int(ports[j]), "in line"
        )
        if best is None or found.limit < best.limit:
            best = found
    first = {int(c): k for k, c in reversed(list(enumerate(col)))}
    np.fill_diagonal(farthest, np.inf)
    for c1, c2 in zip(*np.nonzero(farthest < reach), strict=True):
        k1, k2 = first[int(c1)], first[int(c2)]
        far = float(farthest[c1, c2])
        if systems[k1] == systems[k2]:
            continue
        found = Beside(far, int(ports[k1]), int(ports[k2]), "around")
        if best is None or found.limit < best.limit:
            best = found
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
        Distance in m (:data:`NEAR_FACTOR` times the source radius). A piece
        on the observer's axis line counts whole where it belongs to the
        observer's port or wire system (the next piece of a galvanic
        conductor, a rod's surface image; audit W4R3B-01). The observer's own
        polyline counts whole up to a turn of 90° along it, by a weight
        falling linearly to 0 at 150° (:data:`TURN_WHOLE`, audit W4R3B-03),
        and by its part within ``near`` of the point for the rest of the
        weight; of the rest of its electrode, of a piece of another port
        whose wire meets the observer's wire (:func:`wire_systems`) and of
        the surface images only the part within ``near`` counts, of a piece
        of another port that has just stopped meeting it that part by a
        weight fading out to :data:`TAPER` times the touching distance
        (audit W4R3B-04). Every other piece of another port counts nothing
        (audit W4R2B-03).
    """
    x = model.points
    n = x.shape[0]
    pieces = model.pieces
    a_obs = np.array([p.radius for p in pieces])[model.piece]
    chains = _chains(model)
    ports = np.array([p.port for p in pieces], dtype=int)
    contacts = _contacts(model)
    systems = wire_systems(model, contacts)
    sys_index, sys_weights = _system_weights(model, contacts, systems)
    turn, total_turn = _turns(model)
    chain_obs = chains[model.piece]
    port_obs = ports[model.piece]
    system_obs = systems[model.piece]
    index_obs = sys_index[system_obs]
    turn_obs = turn[model.piece]
    d_obs = np.array([p.direction for p in pieces])[model.piece]
    on_axis = 1e-6 * near  # m; collinear pieces share the axis to round-off
    span = TURN_NONE - TURN_WHOLE
    total = np.zeros(n)
    mirror = np.array([1.0, 1.0, -1.0])
    for k, p in enumerate(pieces):
        # the window counts for the observer's electrode and for wires that
        # meet the observer's wire (one conductor), fading out for wires
        # that have just stopped meeting it; not for other ports
        if sys_index[systems[k]] >= 0:
            fade = np.where(
                index_obs >= 0, sys_weights[index_obs, sys_index[systems[k]]], 0.0
            )
        else:
            fade = np.zeros(n)
        window = np.where(
            (port_obs == ports[k]) | (system_obs == systems[k]), 1.0, fade
        )
        own = chain_obs == chains[k]
        steps = np.abs(turn[k] - turn_obs)
        if np.isfinite(total_turn[k]):
            steps = np.minimum(steps, total_turn[k] - steps)
        weight = np.where(own, np.clip((TURN_NONE - steps) / span, 0.0, 1.0), 0.0)
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
                f_whole = np.where(collinear, window, 0.0)
                f_window = np.where(collinear, 0.0, window)
            else:  # the own wire as far as it has turned by less than 150°
                f_whole = np.where(collinear, window, weight)
                f_window = np.where(collinear, 0.0, np.where(own, 1.0 - weight, window))
            idx = np.nonzero(f_whole > 0.0)[0]
            if idx.size:
                xi = x[idx]
                total[idx] += f_whole[idx] * (
                    _line_integral(xi, a, b, a_obs[idx], 0.0, length)
                    - _line_integral(xi, a, b, np.full(idx.size, r_eta), 0.0, length)
                )
            half = np.sqrt(np.maximum(near * near - p2, 0.0))
            lo = np.clip(t - half, 0.0, length)
            hi = np.clip(t + half, 0.0, length)
            idx = np.nonzero((f_window > 0.0) & (hi > lo))[0]
            if idx.size:
                xi = x[idx]
                total[idx] += f_window[idx] * (
                    _line_integral(xi, a, b, a_obs[idx], lo[idx], hi[idx])
                    - _line_integral(
                        xi, a, b, np.full(idx.size, r_eta), lo[idx], hi[idx]
                    )
                )
    return rho_local / (4.0 * math.pi) * total
