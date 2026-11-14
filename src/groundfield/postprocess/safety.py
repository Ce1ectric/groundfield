"""Touch and step voltages for grounding-system safety assessment.

This module turns a :class:`groundfield.solver.result.FieldResult` into
the engineering quantities used in the grounding-system safety
verification according to EN 50522:2022 (DIN EN 50522 (VDE 0101-2):
2023-10) / IEC 61936-1: the **touch voltage** :math:`U_T` and the
**step voltage** :math:`U_S`. It also provides the **permissible**
values :math:`U_{Tp}(t_f)` of EN 50522:2022, Table B.4 (the curve of
Figure 8), the permissible **prospective** (open-circuit) touch voltage
:math:`U_{vTp}` of Annex B.4, and the true-RMS touch voltage over a
fault-current spectrum.

Mathematical / physical content
-------------------------------
A grounding cluster at earth potential rise

.. math::

    U_E \\;=\\; \\varphi_\\text{cluster},

is connected — through any galvanically bonded metallic part —
to whatever a person can touch. Standing on the soil surface 1 m away
from the touched part (the conventional position of EN 50522), the
person's feet sit at the surface potential
:math:`\\varphi(\\mathbf{r}_\\text{feet})`. The prospective touch
voltage is

.. math::

    U_T \\;=\\; U_E \\;-\\; \\varphi(\\mathbf{r}_\\text{feet}).

The touched parts are the accessible metal parts above ground: a
station housing or its door, a mast, a fence, a service connection box.
:func:`touch_voltage_at_part` takes such a part as a touch point — its
horizontal position and the electrode or junction it is bonded to — and
puts the feet on the circle of radius ``distance`` around it, optionally
only in given directions or outside the outlines where nobody can stand
(the housing or building itself). A part bonded to a node far from the
faulted earth, for example to a PEN junction at a house, carries a
transferred potential.

Where the parts are not known, :func:`touch_voltage` and
:func:`touch_voltage_envelope` use the electrode itself: since review
pass 10 (10.21) the metre is measured from its **footprint** — the
horizontal projection of its wires — not from its connection point; the
feet stand ``distance`` metres from the nearest wire, on the side given
by ``direction`` (seen from the length-weighted centroid of the wires).
The envelope does not sample the inside of the footprint by default,
because there stands the station, the house, the steel structure or the
mast. Over an open-air grid such as a substation yard people do walk
inside the loops, where the surface potential sags:
``interior="meshes"`` adds the mesh centres of grids and the centre of a
ring, the analogue of the IEEE Std 80 *mesh voltage*, and
``interior=True`` also the inside of a closed polyline.

For a *step* voltage the body bridges two surface points at the
typical step distance :math:`d_\\text{step} = 1\\,\\mathrm{m}`:

.. math::

    U_S \\;=\\; \\varphi(\\mathbf{r}_1) \\;-\\;
              \\varphi(\\mathbf{r}_1 + d_\\text{step}\\,\\hat{\\mathbf e}).

Both quantities are returned as complex phasors per frequency
index — consistent with the rest of ``groundfield`` — so that
inductive- and resistive-coupling effects above DC remain
visible.

Validity envelope
-----------------
* Frequency: quasi-static envelope :math:`f \\le 1\\,\\mathrm{kHz}`,
  inherited from the underlying Green's function.
* Current ansatz: the ``image`` family (``image``, ``image_2layer``,
  ``image_nlayer``, ``cim``) solves **one uniform current per
  electrode**. Impedances are fine, but near grids, meshes, stars, strip
  ends and bonded multi-electrode clusters the surface potential is
  impedance-grade, not field-grade. Measured against the per-segment
  ``mom`` backend (homogeneous 100 Ω·m, ``segment_length`` 0.5 m) with
  the default line kernel, touch voltage with the feet 1 m outside the
  footprint and step voltage from 1 m to 2 m outside it: 20 m × 20 m
  grid with 4 × 4 meshes at the corner, touch +20 %, step −21 %
  (non-conservative; the step from the point above the corner 1 m
  outwards is −28 %); 4-arm star (5 m arms) beyond a tip touch +8 %,
  step −9 %; 10 m strip beyond its end step −6 %; rings, rods and closed
  rectangular loops within 2 %. The point kernel, which ``mom`` shares,
  isolates the ansatz: grid +19 % / −21 % (−28 % from above the corner),
  star +7 % / −10 %, strip −7 %, rings, rods and loops within 1 %. The
  helpers emit
  :class:`UniformCurrentFieldWarning` once per call when a uniform-current
  result is evaluated within :data:`NEAR_FIELD_DISTANCE_M` of such an
  electrode; re-solve with ``backend="mom"`` for field-grade values.
* Coordinate convention: the soil surface is at :math:`z = 0`,
  positive :math:`z` points downwards into the soil (see
  :mod:`groundfield.geometry.electrodes`). The default
  ``surface_z = 0.0`` therefore models bare-foot contact at the
  ground surface.
* :func:`permissible_touch_voltage_en50522` interpolates
  EN 50522:2022, Table B.4 (the curve of Figure 8) for fault durations
  :math:`50\\,\\mathrm{ms} \\le t_f \\le 10\\,\\mathrm{s}`, clamps to
  725 V below 50 ms and returns 80 V for every :math:`t_f > 10\\,\\mathrm{s}`
  including an uncleared fault (the value the standard allows for
  current-flow durations considerably longer than 10 s).

References
----------
- EN 50522:2022 — Earthing of power installations exceeding 1 kV a.c.,
  Annex B (Table B.1 body current, Table B.4 permissible touch voltage,
  B.4 prospective touch voltage). German edition DIN EN 50522
  (VDE 0101-2):2023-10.
- IEC 60479-1:2018 — Effects of current on human beings and livestock
  (curve c2, body impedance).
- IEC 61936-1:2021 — Power installations exceeding 1 kV a.c.
- IEEE Std 80-2013 — Guide for safety in AC substation grounding.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from groundfield.postprocess._footprint import (
    Footprint,
    _inside_polygon,
    footprint,
    footprint_of,
    foundation_wires,
    interior_points_of,
)

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = [
    "touch_voltage",
    "step_voltage",
    "touch_voltage_envelope",
    "touch_voltage_at_part",
    "touch_voltage_spectrum",
    "permissible_touch_voltage_en50522",
    "permissible_prospective_touch_voltage_en50522",
    "UniformCurrentFieldWarning",
    "NEAR_FIELD_DISTANCE_M",
]

Anchor = Literal["footprint", "connection_point"]

#: Interior sampling of :func:`touch_voltage_envelope`: ``False`` (none),
#: ``"meshes"`` (grids, rings, bonded-wire grids) or ``True`` (also the
#: inside of closed polylines).
Interior = bool | Literal["meshes"]


class UniformCurrentFieldWarning(UserWarning):
    """Field quantity from a uniform-current backend near a crowding geometry.

    The ``image`` family assigns one uniform current density to every
    electrode. Near grids, meshes, stars, strip ends and bonded
    multi-electrode clusters the true current crowds to edges, corners and
    tips, so touch and step voltages computed there are impedance-grade
    (review pass 10, 10.22; see the module docstring for measured
    deviations). Re-solve with ``backend="mom"`` for field-grade values,
    or silence the category with
    ``warnings.simplefilter("ignore", UniformCurrentFieldWarning)``.
    """


#: Backends that solve one uniform current per electrode.
_UNIFORM_CURRENT_BACKENDS = frozenset({"image", "image_2layer", "image_nlayer", "cim"})

#: Horizontal distance in m from a crowding electrode within which the
#: safety helpers warn about uniform-current results (10.22). At 5 m from
#: the footprint the measured touch / step deviations of the uniform
#: ansatz against ``mom`` are at most 4 % for strips, bonded clusters and
#: the default house grid, 6.5 % for a 4-arm star and 8 % for a 20 m grid
#: (default line kernel; with the point kernel 5 % for the star).
NEAR_FIELD_DISTANCE_M: float = 5.0


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _normalised(v: tuple[float, float, float]) -> np.ndarray:
    """Return ``v`` normalised to unit length, raise on the zero vector."""
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 0.0:
        raise ValueError(f"direction must be a non-zero 3-vector, got {tuple(v)}.")
    return arr / n


def _horizontal(direction: tuple[float, float, float]) -> np.ndarray:
    """Unit horizontal direction ``(ex, ey, 0)`` of ``direction``."""
    e_hat = _normalised(direction)
    e_h = np.array([e_hat[0], e_hat[1], 0.0])
    if float(np.linalg.norm(e_h)) <= 0.0:
        raise ValueError(f"direction has no horizontal component: {tuple(direction)}.")
    return e_h / float(np.linalg.norm(e_h))


def _check_distance(distance: float) -> None:
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"distance must be > 0, got {distance!r}.")


def _check_anchor(anchor: str) -> None:
    if anchor not in ("footprint", "connection_point"):
        raise ValueError(
            f"anchor must be 'footprint' or 'connection_point', got {anchor!r}."
        )


def _cluster_potential(
    result: "FieldResult", electrode: str, frequency_index: int
) -> complex:
    """Return the cluster potential the user is touching."""
    if electrode not in result.electrode_potentials:
        raise KeyError(
            f"Electrode '{electrode}' not found in result.electrode_potentials. "
            f"Known: {sorted(result.electrode_potentials.keys())}."
        )
    try:
        return complex(result.electrode_potentials[electrode][frequency_index])
    except IndexError as e:  # pragma: no cover — defensive
        raise IndexError(
            f"frequency_index {frequency_index} out of range "
            f"[0, {len(result.electrode_potentials[electrode])})."
        ) from e


def _node_potential(result: "FieldResult", name: str, frequency_index: int) -> complex:
    """Potential in V of the electrode or junction ``name`` at one frequency."""
    junctions = result.metadata.get("junction_potentials") or {}
    if name in junctions:
        values = junctions[name]
    elif name in result.electrode_potentials:
        values = result.electrode_potentials[name]
    else:
        known = sorted(set(result.electrode_potentials) | set(junctions))
        shown = ", ".join(repr(k) for k in known[:12])
        more = " ..." if len(known) > 12 else ""
        raise KeyError(
            f"{name!r} is neither an electrode nor a junction with a potential "
            f"in this result (known: {shown}{more})."
        )
    try:
        return complex(values[frequency_index])
    except IndexError as e:
        raise IndexError(
            f"frequency_index {frequency_index} out of range [0, {len(values)})."
        ) from e


def _check_interior(interior: object) -> "Interior":
    """``interior`` as ``False``, ``"meshes"`` or ``True`` (numpy booleans too)."""
    if isinstance(interior, (bool, np.bool_)):
        return bool(interior)
    if isinstance(interior, str) and interior == "meshes":
        return "meshes"
    hint = (
        " (interior=None was the 'meshes' sampling of the development "
        "snapshots; pass interior='meshes')"
        if interior is None
        else ""
    )
    raise ValueError(
        f"interior must be False, 'meshes' or True, got {interior!r}{hint}."
    )


def _horizontal_xy(values, what: str) -> np.ndarray:
    """``(x, y)`` of a 2- or 3-vector of finite numbers (``z`` ignored)."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size not in (2, 3) or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{what} must be (x, y) or (x, y, z) with finite coordinates, "
            f"got {values!r}."
        )
    return arr[:2].copy()


def _part_directions(n_angles: int | None, directions) -> tuple[np.ndarray, np.ndarray]:
    """Azimuths ``(n,)`` in rad and horizontal unit vectors ``(n, 2)``."""
    if directions is None:
        n = 24 if n_angles is None else int(n_angles)
        if n < 3:
            raise ValueError(
                f"n_angles must be >= 3, got {n_angles}; pass directions= for "
                "fewer foot positions."
            )
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return angles, np.column_stack([np.cos(angles), np.sin(angles)])
    if n_angles is not None:
        raise ValueError("pass either n_angles or directions, not both.")
    rows = [np.asarray(d, dtype=float).ravel() for d in directions]
    if not rows:
        raise ValueError("directions must contain at least one direction.")
    units = []
    for d in rows:
        if d.size not in (2, 3):
            raise ValueError(
                "directions must be a sequence of 2- or 3-vectors, got an "
                f"entry {tuple(d.tolist())}."
            )
        units.append(_horizontal((d[0], d[1], d[2] if d.size == 3 else 0.0))[:2])
    unit = np.asarray(units)
    angles = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * np.pi)
    return angles, unit


def _accessible(
    local: np.ndarray, outlines, origin: np.ndarray, distance: float
) -> np.ndarray:
    """Mask of the foot positions not strictly inside any of ``outlines``.

    ``local`` are the foot positions relative to ``origin`` (the touched
    part); the outlines are shifted to the same origin, so map coordinates
    such as UTM do not cost the precision of the boundary test. A foot on
    an outline (within ``1e-9`` of ``max(1, distance)``) stands against
    the wall and is kept.
    """
    keep = np.ones(len(local), dtype=bool)
    if outlines is None:
        return keep
    tol = 1e-9 * max(1.0, distance)
    for k, outline in enumerate(outlines):
        poly = np.asarray(outline, dtype=float)
        if (
            poly.ndim != 2
            or poly.shape[0] < 3
            or poly.shape[1] not in (2, 3)
            or not np.all(np.isfinite(poly))
        ):
            raise ValueError(
                "exclude must be a sequence of outlines, each a polygon of at "
                "least 3 vertices (x, y) or (x, y, z) with finite coordinates "
                f"in m; exclude[{k}] is not."
            )
        poly = poly[:, :2] - origin
        edges = np.stack([poly, np.roll(poly, -1, axis=0)], axis=1)
        on_wall = Footprint(edges, np.zeros((0, 3))).distance(local) <= tol
        keep &= ~(_inside_polygon(local, poly) & ~on_wall)
    return keep


def _feet_xy(
    world: "World",
    electrode: str,
    directions: np.ndarray,
    distance: float,
    anchor: str,
    result: "FieldResult",
) -> np.ndarray:
    """Horizontal foot positions ``(n, 2)`` for unit ``directions`` ``(n, 2)``.

    With ``anchor="footprint"`` the footprint is that of the whole
    foundation the electrode belongs to (:func:`foundation_wires`): a
    rotated foundation is several bonded wires (audit 1, H4).
    """
    e = world.get_electrode(electrode)
    if anchor == "connection_point":
        cp = e.connection_point
        return np.column_stack(
            [cp[0] + distance * directions[:, 0], cp[1] + distance * directions[:, 1]]
        )
    wires = foundation_wires(world, electrode, getattr(result, "clusters", None))
    return footprint_of(wires).feet(directions, distance)


def _crowding(electrode, result: "FieldResult") -> bool:
    """Whether the uniform-current ansatz misrepresents the field near it."""
    kind = getattr(electrode, "kind", None)
    if kind in ("grid_mesh", "mesh", "star", "strip"):
        return True
    if kind == "polyline" and not (electrode.closed and len(electrode.vertices) > 2):
        return True
    return len(result.clusters.get(electrode.name, [electrode.name])) > 1


def _uniform_current_caveat(
    result: "FieldResult",
    world: "World | None",
    points: np.ndarray,
    *,
    caller: str,
) -> None:
    """Warn once if a uniform-current result is evaluated near crowding wires."""
    if result.backend not in _UNIFORM_CURRENT_BACKENDS:
        return
    xy = np.asarray(points, dtype=float)[:, :2]
    near = False
    if world is None:
        # Without the world the electrode shapes are unknown: use the
        # point sources as the wire positions (conservative).
        if result.point_sources:
            src = np.array([ps.position[:2] for ps in result.point_sources])
            half = np.array([0.5 * ps.length for ps in result.point_sources])
            d = np.sqrt(((xy[:, None, :] - src[None, :, :]) ** 2).sum(-1)) - half
            near = bool(d.min() <= NEAR_FIELD_DISTANCE_M)
    else:
        for e in world.electrodes:
            if e.name not in result.electrode_potentials or not _crowding(e, result):
                continue
            try:
                fp = footprint(e)
            except TypeError:  # unknown kind: assume crowding everywhere
                near = True
                break
            if float(fp.distance(xy).min()) <= NEAR_FIELD_DISTANCE_M:
                near = True
                break
    if near:
        warnings.warn(
            f"{caller}: uniform-current backend '{result.backend}' within "
            f"{NEAR_FIELD_DISTANCE_M:g} m of a grid, mesh, star, strip, open "
            "polyline or bonded multi-electrode cluster: the surface potential "
            "is impedance-grade (20 m grid corner against 'mom': touch 1 m "
            "outside +20 %, step from 1 m to 2 m outside -21 %, step from the "
            "point above the corner 1 m outwards -28 %). Use backend='mom' for "
            "field-grade touch and step voltages.",
            UniformCurrentFieldWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------
# Touch voltage
# ---------------------------------------------------------------------


def touch_voltage(
    result: "FieldResult",
    world: "World",
    *,
    electrode: str,
    distance: float = 1.0,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    surface_z: float = 0.0,
    frequency_index: int = 0,
    anchor: Anchor = "footprint",
) -> complex:
    """Touch voltage at a single point on the soil surface.

    Evaluates :math:`U_T = U_E - \\varphi(\\mathbf{r}_\\text{feet})`,
    where :math:`U_E` is the cluster potential of the touched electrode.
    With the default ``anchor="footprint"`` the feet stand ``distance``
    metres (horizontally) from the nearest wire of the electrode, on the
    ray that leaves the electrode's footprint in the horizontal
    projection of ``direction``: 1 m outside the edge of a grid, 1 m
    beyond the tip of a star arm, 1 m outside a ring on either side.
    This is a proxy for an unknown touched part; for a known accessible
    metal part use :func:`touch_voltage_at_part`.

    Parameters
    ----------
    result
        Solver output from :meth:`Engine.solve`.
    world
        Companion world used to resolve the electrode geometry.
    electrode
        Name of the touched electrode. Its cluster potential is the
        EPR seen by the hand.
    distance
        Horizontal distance between the touched part and the feet,
        in m. EN 50522 uses ``1.0``.
    direction
        Side of the electrode on which the person stands, in world
        coordinates, seen from the length-weighted centroid of the wires.
        Only the horizontal projection is used (the feet stay on the
        surface). Defaults to ``+x``.
    surface_z
        :math:`z` coordinate of the soil surface in m. Default ``0.0``
        (groundfield convention; positive ``z`` is below ground).
    frequency_index
        Index into :attr:`FieldResult.frequencies`. Default 0.
    anchor
        ``"footprint"`` (default since review pass 10, 10.21) measures
        ``distance`` from the nearest wire; ``"connection_point"``
        measures it from :attr:`connection_point` of ``electrode``, the
        rule of v0.15.0. For a hand-built grid mesh or star that is its
        centre, as in v0.15.0. A rectangular foundation of the generators
        is since 2026.11 several bonded wires — one per mesh line (audit
        A2G-04), one per side without cross-braces (user decision Q38) —
        and the electrode named after it is its local :math:`-y` side,
        whose connection point is a corner of the rectangle (in 0.15.0 one
        grid mesh connected at its centre): the feet then stand around that
        corner. For 100 A into a generated ring at 0.8 m in 100 Ω·m the
        largest value on the circle (default engine, the same at 0.5 m and
        0.25 m segments) is 270.89 V round the corner of a 10 m × 10 m ring
        against 245.22 V round its centre, 229.71 against 159.61 V for
        20 m × 5 m, 91.10 against 111.38 V for 30 m × 30 m and, in 5 cm of
        150 Ω·m concrete, 134.24 against 119.32 V for 40 m × 10 m (audit
        D3A-03). Use the default ``"footprint"`` for generated foundations;
        it takes the whole foundation and does not depend on the
        connection point.

    Returns
    -------
    complex
        Phasor :math:`U_T(f) = U_E(f) - \\varphi(\\mathbf{r}_\\text{feet}, f)`
        in V.

    Raises
    ------
    KeyError
        If ``electrode`` is unknown to ``world`` or ``result``.
    ValueError
        If ``distance`` is not strictly positive, ``direction`` has no
        horizontal component, or ``anchor`` is unknown.

    Warns
    -----
    UniformCurrentFieldWarning
        Once per call, for a uniform-current result evaluated near a
        crowding geometry (see the module docstring).

    Notes
    -----
    Up to v0.15.0 the metre was measured from the connection point, so
    for a 20 m × 20 m grid the feet stood above the grid and the helper
    reported 9.92 V where 1 m outside the edge has 69.6 V (100 A,
    100 Ω·m; F13 / 10.21).
    """
    _check_distance(distance)
    _check_anchor(anchor)
    e_h = _horizontal(direction)
    feet_xy = _feet_xy(world, electrode, e_h[None, :2], distance, anchor, result)[0]
    feet = np.array([[feet_xy[0], feet_xy[1], surface_z]])
    _uniform_current_caveat(result, world, feet, caller="touch_voltage")
    phi_feet = complex(result.potential(feet, frequency_index=frequency_index)[0])
    U_E = _cluster_potential(result, electrode, frequency_index)
    return U_E - phi_feet


def touch_voltage_envelope(
    result: "FieldResult",
    world: "World",
    *,
    electrode: str,
    distance: float = 1.0,
    n_angles: int = 24,
    surface_z: float = 0.0,
    frequency_index: int = 0,
    anchor: Anchor = "footprint",
    interior: Interior = False,
    return_points: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Touch voltages around an electrode, on request also inside its loops.

    Walks ``n_angles`` equally spaced directions and places the feet as
    in :func:`touch_voltage`: with ``anchor="footprint"`` this traces
    the curve at ``distance`` from the electrode's wires (rounded around
    corners and arm tips). The maximum of ``|U_T|`` is the conservative
    envelope used in safety verification when the touched parts are not
    known; where they are, :func:`touch_voltage_at_part` puts the feet
    around each part.

    By default nothing inside the footprint is sampled: there stands the
    station, the house, the steel structure or the mast. Over an open-air
    grid such as a substation yard people walk inside the loops, and the
    surface potential sags there, so the touch voltage can exceed every
    value on the offset curve (a 20 m × 20 m perimeter loop at 0.5 m
    depth, 100 Ω·m, 100 A: 158 V in the centre against at most 141 V
    1 m outside). For such grids ``interior="meshes"`` also evaluates the
    centre of every closed mesh, the analogue of the *mesh voltage* of
    IEEE Std 80 (the worst touch voltage inside a mesh of a grid, near
    the mesh centre).

    Parameters
    ----------
    result, world
        See :func:`touch_voltage`.
    electrode
        Name of the touched electrode.
    distance
        Horizontal distance from the wires in m (1.0 by EN 50522).
    n_angles
        Number of equally spaced directions
        :math:`\\theta_k = 2\\pi k / n` with :math:`k = 0, \\dots, n - 1`.
        Must be ``>= 3``.
    surface_z, frequency_index, anchor
        See :func:`touch_voltage`.
    interior
        Which closed loops are also sampled inside (review pass 10, 10.21).
        ``False`` (default): none, the offset curve only, for every
        electrode kind. ``"meshes"``: the centre of every mesh of a
        ``mesh`` / ``grid_mesh`` electrode, the centre of a ``ring``, and
        the centre of every cell of a foundation built from bonded wires
        (:func:`~groundfield.postprocess._footprint.foundation_wires`; the
        generators build every mesh foundation as one open polyline per
        wire line and every rectangle without cross-braces as one wire per
        side, user decision Q38), but not the inside of a closed
        ``polyline``, the
        building-outline foundation of the generators. Use it where the
        inside of the loops is accessible open ground: a substation yard
        or another open-air grid, a ring around an open area. ``True``:
        also a closed polyline, at the point inside it farthest from its
        wires. The modes cannot see buildings: a mesh foundation's cells
        lie under the house, a ring may surround a station housing or a
        mast foot.
    return_points
        Also return the foot positions.

    Returns
    -------
    angles : np.ndarray, shape (n,)
        Azimuth in radians of each foot position, measured from ``+x``.
        The first ``n_angles`` entries are :math:`\\theta_k`; appended
        interior points carry their azimuth seen from the centroid of
        the wires (0 for a point that coincides with it).
    voltages : np.ndarray, shape (n,), dtype complex
        Touch-voltage phasor :math:`U_T` in V at every foot position.
    points : np.ndarray, shape (n, 3)
        Foot positions in m; only with ``return_points=True``.

    Raises
    ------
    ValueError
        For ``n_angles < 3``, a non-positive ``distance``, an unknown
        ``anchor`` or an ``interior`` other than ``False``, ``"meshes"``
        and ``True``.

    Warns
    -----
    UniformCurrentFieldWarning
        Once per call, for a uniform-current result evaluated near a
        crowding geometry (see the module docstring).

    Notes
    -----
    ``n = n_angles`` plus the number of interior points. The rule of
    v0.15.0 (circle around the connection point) is
    ``anchor="connection_point"``; for a generated rectangular foundation
    that connection point is a corner since 2026.11, no longer the centre
    (see ``anchor`` in :func:`touch_voltage`). During the 2026.11 development the
    mesh sampling was the default (``interior=None``); it is now the
    explicit ``interior="meshes"``.
    """
    if n_angles < 3:
        raise ValueError(f"n_angles must be >= 3, got {n_angles}.")
    _check_distance(distance)
    _check_anchor(anchor)
    # user decision Q3: no interior sampling unless asked for.
    interior = _check_interior(interior)

    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    feet_xy = _feet_xy(world, electrode, dirs, distance, anchor, result)
    if interior is not False:
        wires = foundation_wires(world, electrode, getattr(result, "clusters", None))
        inner = interior_points_of(wires, outlines=interior is True)
        if len(inner):
            ref = footprint_of(wires).reference_point
            az = np.mod(
                np.arctan2(inner[:, 1] - ref[1], inner[:, 0] - ref[0]), 2 * np.pi
            )
            az[np.hypot(*(inner - ref).T) <= 1e-12] = 0.0
            angles = np.concatenate([angles, az])
            feet_xy = np.vstack([feet_xy, inner])
    feet = np.column_stack([feet_xy, np.full(len(feet_xy), surface_z, dtype=float)])
    _uniform_current_caveat(result, world, feet, caller="touch_voltage_envelope")
    phi_feet = result.potential(feet, frequency_index=frequency_index)
    U_E = _cluster_potential(result, electrode, frequency_index)
    voltages = U_E - phi_feet
    if return_points:
        return angles, voltages, feet
    return angles, voltages


def touch_voltage_at_part(
    result: "FieldResult",
    world: "World | None" = None,
    *,
    position: Sequence[float],
    bonded_to: str,
    distance: float = 1.0,
    n_angles: int | None = None,
    directions: "Sequence[Sequence[float]] | None" = None,
    exclude: "Sequence[Sequence[Sequence[float]]] | None" = None,
    surface_z: float = 0.0,
    frequency_index: int = 0,
    return_points: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Touch voltage of an accessible metal part, feet ``distance`` around it.

    EN 50522 defines the touch voltage between a conductive part that a
    person touches and the person's feet, standing 1 m away from the
    part. The touch points of an installation are its accessible metal
    parts above ground — a station housing or its door, a mast, a fence
    post, a cable cabinet, a service connection box — not the buried
    electrode. This helper takes one such part: its horizontal position
    :math:`\\mathbf p` and the electrode or junction it is bonded to,
    whose potential :math:`U_\\text{part}` it carries. The feet stand on
    the circle of radius :math:`d` = ``distance`` around the part,

    .. math::

        \\mathbf r_k = \\mathbf p + d\\,(\\cos\\theta_k, \\sin\\theta_k),
        \\qquad
        U_{T,k} = U_\\text{part} - \\varphi(\\mathbf r_k),

    at ``n_angles`` equally spaced azimuths
    :math:`\\theta_k = 2\\pi k / n`, or only in the given ``directions``;
    foot positions inside an outline of ``exclude`` (the housing or the
    building itself, where nobody can stand) are dropped. A part bonded
    to a node far from the faulted earth — a service connection box on the
    PEN junction of a house, a cable screen led out of a station —
    carries a transferred potential; ``U_T`` is then most of the earth
    potential rise.

    Parameters
    ----------
    result
        Solver output from :meth:`Engine.solve`.
    world
        Optional companion world. Only used to decide whether to emit
        :class:`UniformCurrentFieldWarning`, as in :func:`step_voltage`.
    position
        Horizontal position ``(x, y)`` of the touched part in m. A third
        coordinate is accepted and ignored: the part stands above ground,
        the feet on the surface.
    bonded_to
        Name of the electrode or junction (:func:`~groundfield.create_junction`)
        whose potential the part carries: the station earth for its
        housing, the PEN junction of a house for its connection box.
    distance
        Horizontal distance between the part and the feet in m, ``> 0``
        (1.0 by EN 50522).
    n_angles
        Number of equally spaced foot positions on the circle, ``>= 3``;
        24 when neither ``n_angles`` nor ``directions`` is given.
    directions
        Only these horizontal directions (2- or 3-vectors, the ``z``
        component is ignored), seen from the part: for example the outside
        of a fence. Not together with ``n_angles``.
    exclude
        Outlines where nobody can stand, each a polygon of vertices
        ``(x, y)`` or ``(x, y, z)`` in m (``z`` ignored), e.g. the housing
        of a station or a building. Foot positions strictly inside an
        outline are dropped; positions on it (against the wall) are kept.
    surface_z
        :math:`z` coordinate of the soil surface in m. Default ``0.0``.
    frequency_index
        Index into :attr:`FieldResult.frequencies`. Default 0.
    return_points
        Also return the foot positions.

    Returns
    -------
    angles : np.ndarray, shape (n,)
        Azimuth in radians in :math:`[0, 2\\pi)` of each remaining foot
        position, seen from the part and measured from ``+x``.
    voltages : np.ndarray, shape (n,), dtype complex
        Touch-voltage phasor :math:`U_T` in V at every foot position.
    points : np.ndarray, shape (n, 3)
        Foot positions in m; only with ``return_points=True``.

    Raises
    ------
    KeyError
        If ``bonded_to`` is neither an electrode nor a junction with a
        potential in ``result``.
    IndexError
        If ``frequency_index`` is out of range.
    ValueError
        If ``distance`` is not a positive finite number, ``position`` is
        not a finite 2- or 3-vector, ``n_angles < 3``, both ``n_angles``
        and ``directions`` are given, a direction has no horizontal
        component, an outline is not a polygon, or every foot position
        lies inside an outline.

    Warns
    -----
    UniformCurrentFieldWarning
        Once per call, for a uniform-current result evaluated near a
        crowding geometry (see the module docstring).

    Notes
    -----
    For a part on a wire of the electrode with the feet pointing away
    from it, the foot position is the one of :func:`touch_voltage` and so
    is the value. The voltage is the prospective (open-circuit) touch
    voltage; compare its magnitude with
    :func:`permissible_touch_voltage_en50522`, or with
    :func:`permissible_prospective_touch_voltage_en50522` when footwear or
    a surface layer adds resistance.
    """
    # user decision Q3: touch points are the accessible metal parts.
    _check_distance(distance)
    origin = _horizontal_xy(position, "position")
    angles, unit = _part_directions(n_angles, directions)
    keep = _accessible(distance * unit, exclude, origin, distance)
    if not keep.any():
        raise ValueError(
            "every foot position around the part lies inside an outline of "
            "exclude; there is no accessible standing position."
        )
    angles, feet_xy = angles[keep], origin + distance * unit[keep]
    u_part = _node_potential(result, bonded_to, frequency_index)
    feet = np.column_stack([feet_xy, np.full(len(feet_xy), surface_z, dtype=float)])
    _uniform_current_caveat(result, world, feet, caller="touch_voltage_at_part")
    voltages = u_part - result.potential(feet, frequency_index=frequency_index)
    if return_points:
        return angles, voltages, feet
    return angles, voltages


# ---------------------------------------------------------------------
# Step voltage
# ---------------------------------------------------------------------


def step_voltage(
    result: "FieldResult",
    *,
    position: tuple[float, float, float],
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    step: float = 1.0,
    surface_z: float | None = None,
    frequency_index: int = 0,
    world: "World | None" = None,
) -> complex:
    """Step voltage between two surface points.

    Evaluates :math:`U_S = \\varphi(\\mathbf{r}_1) -
    \\varphi(\\mathbf{r}_1 + d_\\text{step}\\,\\hat{\\mathbf{e}})`
    on the soil surface.

    Parameters
    ----------
    result
        Solver output.
    position
        First foot position :math:`(x, y, z)` in m. The :math:`z`
        coordinate is overridden by ``surface_z`` if the latter is
        given; the default keeps ``z`` as supplied so callers can use
        the helper for buried-step studies if desired.
    direction
        Step direction in world coordinates. The horizontal
        projection is used (the :math:`z` component is ignored —
        both feet stay on the same surface).
    step
        Step length :math:`d_\\text{step}` in m. EN 50522 uses ``1.0``.
    surface_z
        Optional explicit surface :math:`z` in m. ``None`` keeps the
        :math:`z` of ``position`` (typical for bare-foot contact
        at the ground surface, ``position = (x, y, 0.0)``).
    frequency_index
        Index into :attr:`FieldResult.frequencies`.
    world
        Optional companion world. Only used to decide whether to emit
        :class:`UniformCurrentFieldWarning`: with the world the warning
        is restricted to the electrode shapes with measured ansatz
        errors; without it any wire within
        :data:`NEAR_FIELD_DISTANCE_M` of a uniform-current result
        triggers it.

    Returns
    -------
    complex
        Phasor :math:`U_S(f)` in V.

    Raises
    ------
    ValueError
        If ``step`` is not strictly positive or ``direction`` has
        no horizontal component.

    Warns
    -----
    UniformCurrentFieldWarning
        Once per call, see the module docstring. At the corner of a
        20 m × 20 m grid the uniform-current ansatz under-estimates the
        step voltage by 21 % (feet 1 m and 2 m outside the corner) to 28 %
        (feet above the corner and 1 m outside), with either kernel.
    """
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"step must be > 0, got {step!r}.")

    e_h = _horizontal(direction)
    z0 = float(position[2]) if surface_z is None else float(surface_z)
    p1 = np.array([float(position[0]), float(position[1]), z0])
    p2 = p1 + step * e_h
    pts = np.stack([p1, p2])
    _uniform_current_caveat(result, world, pts, caller="step_voltage")

    phi = result.potential(pts, frequency_index=frequency_index)
    return complex(phi[0] - phi[1])


# ---------------------------------------------------------------------
# Touch voltage over a fault spectrum
# ---------------------------------------------------------------------


def _source_current(result: "FieldResult", source: str | None, k: int) -> complex:
    """Impressed current of the selected current source at index ``k``."""
    if not result.sources:
        raise ValueError(
            "touch_voltage_spectrum needs the source snapshot of the solve "
            "(FieldResult.sources); this result carries none."
        )
    current = [s for s in result.sources if s.kind == "current"]
    if source is None:
        if len(current) != 1:
            names = ", ".join(s.name for s in current) or "none"
            raise ValueError(
                f"touch_voltage_spectrum: {len(current)} current sources "
                f"({names}); pass source=<name>."
            )
        record = current[0]
    else:
        matches = [s for s in current if s.name == source]
        if not matches:
            raise KeyError(f"No current source {source!r} in the result.")
        record = matches[0]
    i_src = complex(record.current[k])
    if i_src == 0:
        raise ValueError(
            f"Source {record.name!r} carries no current at frequency index {k}."
        )
    return i_src


def touch_voltage_spectrum(
    result: "FieldResult | Sequence[FieldResult]",
    world: "World",
    *,
    electrode: str,
    currents: "Sequence[complex] | Mapping[float, complex]",
    distance: float = 1.0,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    surface_z: float = 0.0,
    anchor: Anchor = "footprint",
    source: str | None = None,
) -> float:
    """True-RMS touch voltage of a fault-current spectrum at one position.

    The field problem is linear, so the touch voltage of harmonic
    :math:`k` scales with the fault current :math:`I_k` of that
    harmonic. With :math:`U_T(f_k)` from a solve with source current
    :math:`I_\\text{src}(f_k)`, the touch voltage at the fixed foot
    position of :func:`touch_voltage` is

    .. math::

        U_{T,\\text{rms}} \\;=\\; \\sqrt{\\sum_k
        \\left|U_T(f_k)\\,\\frac{I_k}{I_\\text{src}(f_k)}\\right|^2},

    the RMS value of the superposed harmonics (they are orthogonal over
    a period).

    Parameters
    ----------
    result
        One multi-frequency result, or a sequence of results whose
        frequencies are taken in order (for example one result per
        harmonic).
    world
        Companion world (geometry of the touched electrode).
    electrode
        Name of the touched electrode.
    currents
        Fault current per frequency as RMS phasors (or RMS magnitudes)
        in A: either a sequence aligned with the frequencies of
        ``result``, or a mapping ``{frequency_Hz: current}`` that covers
        every frequency exactly once.
    distance, direction, surface_z, anchor
        Foot position, see :func:`touch_voltage`; the same position is
        used for every frequency.
    source
        Name of the current source the results were solved with; may be
        omitted when there is exactly one.

    Returns
    -------
    float
        RMS touch voltage :math:`U_{T,\\text{rms}}` in V.

    Raises
    ------
    ValueError
        If the currents do not match the frequencies, are not finite, a
        frequency appears in more than one result (it would be counted
        twice), a result carries no source snapshot, or the source current
        is zero at some frequency.

    Notes
    -----
    EN 50522's permissible touch voltage is a 50 Hz curve; comparing a
    spectrum RMS with :math:`U_{Tp}` ignores the frequency factors of
    IEC 60479-2 (the body tolerates more current at higher frequencies),
    which is conservative.
    """
    results = [result] if hasattr(result, "frequencies") else list(result)
    pairs = [
        (res, k, float(f)) for res in results for k, f in enumerate(res.frequencies)
    ]
    freqs = [f for _, _, f in pairs]
    # The harmonics of a spectrum are orthogonal only at distinct
    # frequencies; a frequency carried by two results was counted twice
    # (audit 1, L4).
    ordered = sorted(freqs)
    for lo, hi in zip(ordered, ordered[1:], strict=False):
        if math.isclose(lo, hi, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"touch_voltage_spectrum: {lo:g} Hz appears more than once among "
                "the results; give every harmonic once."
            )
    if isinstance(currents, Mapping):
        keys = [float(f) for f in currents]
        values = []
        used: set[int] = set()
        for f in freqs:
            hit = [
                i
                for i, key in enumerate(keys)
                if math.isclose(key, f, rel_tol=1e-9, abs_tol=1e-9)
            ]
            if len(hit) != 1:
                raise ValueError(
                    f"currents must give exactly one value for {f:g} Hz; "
                    f"keys: {sorted(keys)}."
                )
            used.add(hit[0])
            values.append(list(currents.values())[hit[0]])
        extra = [keys[i] for i in range(len(keys)) if i not in used]
        if extra:
            raise ValueError(f"currents has no result for frequencies {extra} Hz.")
    else:
        values = list(currents)
        if len(values) != len(pairs):
            raise ValueError(
                f"currents has {len(values)} entries but the results carry "
                f"{len(pairs)} frequencies."
            )
    i_k = np.asarray(values, dtype=complex)
    if not np.all(np.isfinite(i_k)):
        raise ValueError("currents must be finite.")

    _check_distance(distance)
    _check_anchor(anchor)
    e_h = _horizontal(direction)
    feet_xy = _feet_xy(world, electrode, e_h[None, :2], distance, anchor, results[0])[0]
    feet = np.array([[feet_xy[0], feet_xy[1], surface_z]])
    warned = set()
    total = 0.0
    for (res, k, _), i_fault in zip(pairs, i_k, strict=True):
        if res.backend not in warned:
            _uniform_current_caveat(res, world, feet, caller="touch_voltage_spectrum")
            warned.add(res.backend)
        u_t = _cluster_potential(res, electrode, k) - complex(
            res.potential(feet, frequency_index=k)[0]
        )
        total += abs(u_t * i_fault / _source_current(res, source, k)) ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------
# Permissible touch voltage U_Tp per EN 50522:2022, Table B.4
# ---------------------------------------------------------------------

# EN 50522:2022, Table B.4 (DIN EN 50522 (VDE 0101-2):2023-10): fault
# duration t_f in s -> permissible touch voltage U_Tp in V, the
# tabulated form of the curve in Figure 8 (values rounded to 5 V in the
# standard). Verified against the standard text in review pass 10
# (10.42) and again for user decision Q4. The plateau of 85 V at 5 s and
# 10 s is explicit; the curve ends at 10 s.
_EN50522_TP_GRID: tuple[tuple[float, float], ...] = (
    (0.05, 725.0),
    (0.10, 655.0),
    (0.20, 525.0),
    (0.50, 225.0),
    (1.00, 115.0),
    (2.00, 95.0),
    (5.00, 85.0),
    (10.00, 85.0),
)

# EN 50522:2022, Table B.1: fault duration t_f in s -> permissible body
# current I_B in A (curve c2 of IEC 60479-1:2018, probability of
# ventricular fibrillation below 5 %, left hand to both feet).
_EN50522_IB_GRID: tuple[tuple[float, float], ...] = (
    (0.05, 0.900),
    (0.10, 0.800),
    (0.20, 0.600),
    (0.50, 0.200),
    (1.00, 0.080),
    (2.00, 0.060),
    (5.00, 0.051),
    (10.00, 0.050),
)

#: Permissible touch voltage in V for fault durations beyond the 10 s end
#: of Figure 8 / Table B.4: EN 50522:2022 allows 80 V for current-flow
#: durations considerably longer than 10 s (note to Figure 8 in 5.4.1; B.3
#: after Table B.4). User decision Q4: used for every t_f > 10 s.
_EN50522_TP_LONG_DURATION = 80.0

#: End of the curve of Figure 8 / Table B.4 in s.
_EN50522_CURVE_END_S = 10.0


def _loglog(
    t: float, grid: tuple[tuple[float, float], ...], *, exact_anchors: bool = False
) -> float:
    """Piecewise log-log interpolation, clamped to the grid ends.

    ``exact_anchors`` returns the tabulated value itself at a tabulated
    abscissa (``exp(log(v))`` may differ from ``v`` by one ulp). It is off
    for :math:`U_{Tp}`, whose arithmetic is pinned by the golden values of
    v0.15.0.
    """
    ts = np.array([x for x, _ in grid], dtype=float)
    vs = np.array([v for _, v in grid], dtype=float)
    if t <= ts[0]:
        return float(vs[0])
    if t >= ts[-1]:
        return float(vs[-1])
    if exact_anchors and t in ts:
        return float(vs[int(np.flatnonzero(ts == t)[0])])
    return float(math.exp(np.interp(math.log(t), np.log(ts), np.log(vs))))


def permissible_touch_voltage_en50522(t_clear_s: float) -> float:
    """Permissible touch voltage :math:`U_{Tp}(t_f)` per EN 50522:2022.

    Returns the maximum admissible touch voltage as a function of the
    fault duration :math:`t_f`. Up to 10 s it follows the curve of
    Figure 8 in its tabulated form, Table B.4 (DIN EN 50522
    (VDE 0101-2):2023-10),

    .. math::

        \\{(t_k, U_{Tp,k})\\} \\;=\\;
        \\{(0.05, 725),\\ (0.10, 655),\\ (0.20, 525),\\
        (0.50, 225),\\ (1.00, 115),\\ (2.00, 95),\\
        (5.00, 85),\\ (10.0, 85)\\},

    in s and V (rounded to 5 V in the standard), interpolated linearly in
    :math:`\\log t`–:math:`\\log U` between the anchors and clamped to
    725 V below 50 ms. Beyond the end of the curve,

    .. math::

        U_{Tp}(t_f) \\;=\\; 80\\,\\mathrm{V}
        \\qquad \\text{for } t_f > 10\\,\\mathrm{s},

    including an earth fault that is not cleared (``math.inf``).

    Parameters
    ----------
    t_clear_s
        Fault duration :math:`t_f` in s, ``> 0``; ``math.inf`` for an
        earth fault that is not cleared.

    Returns
    -------
    float
        Permissible touch voltage :math:`U_{Tp}` in V.

    Raises
    ------
    ValueError
        If ``t_clear_s`` is NaN or not strictly positive.

    Notes
    -----
    Where the values come from in EN 50522:2022:

    * 5.4.1, Figure 8: :math:`U_{Tp}` as a function of the fault
      duration; Annex B, B.3, Table B.4: the values of that curve from
      0.05 s to 10 s. The curve is not defined beyond 10 s.
    * The note to Figure 8 (5.4.1) and the paragraph after Table B.4
      (B.3): for a current-flow duration considerably longer than 10 s,
      80 V may be used as the permissible touch voltage.
    * 5.4.1 assumes that every earth fault is cleared, automatically or
      by hand, so that no touch voltage of very long or unlimited
      duration occurs. A resonant-earthed network can carry its earth
      fault for hours, which is where the 80 V value matters.

    The helper uses 80 V for every :math:`t_f > 10\\,\\mathrm{s}` instead
    of continuing the 85 V end value of Table B.4: 80 V is below 85 V,
    on the safe side where the curve gives no value. Hence the step from
    85 V at 10 s to 80 V just above. Up to the 2026.11 development
    snapshots an opt-in keyword selected the 80 V and the default
    returned 85 V; v0.15.0 returned 85 V and rejected ``math.inf``.

    Review pass 10 (10.42) checked the anchors against the standard
    text: they are the published values. A secondary reproduction quoted
    in the review (716 / 654 / 537 / 220 / 117 / 96 / 86 / 85 V) does not
    match EN 50522:2022. Compare the value with :func:`touch_voltage`,
    :func:`touch_voltage_at_part` or :func:`touch_voltage_envelope`
    (or, with additional resistances, use
    :func:`permissible_prospective_touch_voltage_en50522`).
    """
    # math.isnan rejects text (TypeError) before float() could parse it.
    if math.isnan(t_clear_s) or t_clear_s <= 0.0:
        raise ValueError(f"t_clear_s must be > 0 (s), got {t_clear_s!r}.")
    t = float(t_clear_s)
    # user decision Q4: 80 V beyond the end of Figure 8 / Table B.4.
    if t > _EN50522_CURVE_END_S:
        return _EN50522_TP_LONG_DURATION
    return _loglog(t, _EN50522_TP_GRID)


def permissible_prospective_touch_voltage_en50522(
    t_f: float,
    *,
    R_H: float = 0.0,
    R_F1: float = 0.0,
    rho_s: float | None = None,
    HF: float = 1.0,
) -> float:
    """Permissible prospective touch voltage :math:`U_{vTp}` (EN 50522:2022, B.4).

    The prospective (open-circuit) touch voltage may exceed the
    permissible touch voltage across the body by the drop of the body
    current over the additional resistances (equivalent circuit of
    Figure B.2, Equation (A.3) of the standard):

    .. math::

        U_{vTp}(t_f) \\;=\\; U_{Tp}(t_f) \\;+\\;
        \\frac{I_B(t_f)}{HF}\\,R_A,
        \\qquad
        R_A = R_H + R_{F1} + R_{F2},
        \\qquad
        R_{F2} = 1.5\\,\\mathrm{m}^{-1}\\cdot\\rho_s,

    with :math:`U_{Tp}` from :func:`permissible_touch_voltage_en50522`
    (Table B.4 up to 10 s, 80 V beyond) and the permissible body current
    :math:`I_B(t_f)` of Table B.1 (0.05 s: 900 mA, 0.1 s: 800 mA,
    0.2 s: 600 mA, 0.5 s: 200 mA, 1 s: 80 mA, 2 s: 60 mA, 5 s: 51 mA,
    10 s: 50 mA; interpolated log-log, 900 mA below 50 ms, and held at
    50 mA beyond 10 s).

    Parameters
    ----------
    t_f
        Fault duration in s, ``> 0``; ``math.inf`` for an earth fault
        that is not cleared.
    R_H
        Hand resistance (e.g. gloves) in Ω, ``>= 0``.
    R_F1
        Resistance of the footwear in Ω, ``>= 0`` (EN 50522 quotes
        1000 Ω as an average for worn, damp shoes).
    rho_s
        Resistivity of the surface layer at the standing position in
        Ω·m; ``None`` (default) omits the standing-position resistance
        :math:`R_{F2}`.
    HF
        Heart-current factor of the current path, ``> 0``; 1 for left
        hand to feet, the path the standard uses here.

    Returns
    -------
    float
        Permissible prospective touch voltage :math:`U_{vTp}` in V.

    Raises
    ------
    ValueError
        If a resistance or ``rho_s`` is negative or not finite, ``HF`` is
        not positive, or ``t_f`` is invalid (see
        :func:`permissible_touch_voltage_en50522`).

    Examples
    --------
    Figure B.3 of the standard, 0.5 s, no footwear, 500 Ω·m gravel
    (:math:`R_F = 750\\,\\Omega`): 225 V + 0.2 A · 750 Ω = 375 V.

    >>> permissible_prospective_touch_voltage_en50522(0.5, rho_s=500.0)
    375.0

    An earth fault that is not cleared, 1000 Ω shoes:
    80 V + 0.05 A · 1000 Ω = 130 V.

    >>> permissible_prospective_touch_voltage_en50522(math.inf, R_F1=1000.0)
    130.0
    """
    for name, value in (("R_H", R_H), ("R_F1", R_F1)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0 Ohm, got {value!r}.")
    if rho_s is not None and (not math.isfinite(rho_s) or rho_s < 0.0):
        raise ValueError(f"rho_s must be finite and >= 0 Ohm m, got {rho_s!r}.")
    if not math.isfinite(HF) or HF <= 0.0:
        raise ValueError(f"HF must be finite and > 0, got {HF!r}.")
    u_tp = permissible_touch_voltage_en50522(t_f)
    i_b = _loglog(
        min(float(t_f), _EN50522_CURVE_END_S), _EN50522_IB_GRID, exact_anchors=True
    )
    r_a = R_H + R_F1 + (0.0 if rho_s is None else 1.5 * float(rho_s))
    return u_tp + i_b / HF * r_a
