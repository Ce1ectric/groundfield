"""Local frame of a solve at projected coordinates (private).

Worlds built from survey or GIS data carry projected coordinates: a UTM
easting of 32 604 321 m (with the zone prefix) and a northing of
5 812 345 m, where the spacing of the doubles is 3.7e-9 m and 9.3e-10 m.
Every quantity the solvers compute is a function of coordinate
*differences*, but the discretisation (segment midpoints and end points)
and the line kernel (quadrature nodes on the segments) form new points at
the magnitude of the coordinates, whose round-off then reaches the
results: up to the calendar release a strip and a star moved by 1.4e-7,
the mutual matrix of three building outlines by 2e-7, between (0, 0) and
the UTM origin (fw1sol, open point 1; evidence of stream w2mom).

The solvers therefore run in a local frame whenever a horizontal
coordinate of the world exceeds :data:`LOCAL_FRAME_THRESHOLD_M`:

* the origin :math:`o` is the centre of the world's horizontal bounding
  box rounded to whole metres (depths are never shifted: the soil surface
  and the layer interfaces stay where they are);
* the world is copied with every coordinate shifted by :math:`-o`. The
  difference :math:`p - o` is exact when :math:`p` lies within
  :math:`[o/2, 2o]` (Sterbenz's lemma), which every point of a world given
  wholly in projected coordinates does (its extent is far below its
  distance from the origin); for a point outside that interval it is
  rounded to the nearest double, at most half a unit in the last place of
  the difference: 1.9e-9 m for coordinates below 3.36e7 m. Measured
  (evidence ``w2b08_frame_shift.py`` of stream fw2mom, exact rational and
  60-digit decimal references, audit W2B-08): 0 for a UTM world,
  9.1e-13 m for a world spanning 5 ... 25 km, 8.2e-10 m for the local
  nodes of a world that mixes UTM and local coordinates. Harmless: the
  discretisation decides in the user's coordinates (next item), and the
  input coordinates themselves are rounded to 3.7e-9 m at an easting of
  3.3e7 m;
* the round-off tolerances of the discretisation (straight runs and
  segment counts, :mod:`groundfield.geometry.primitives`) are evaluated
  with the coordinates of the user's frame, :math:`p + o`, so the copy is
  discretised with the same decisions as the original;
* the result is reported in the user's coordinates (point-source
  positions shifted back by :math:`+o`), and
  :meth:`~groundfield.solver.result.FieldResult.potential` evaluates
  fields in a local frame of the sources in the same way.

A world within the threshold of the origin is not shifted (the round-off
of coordinates below 10 km is below 1.8e-12 m). Its electrodes are
discretised as before; the count rule of a distributed conductor allows
the round-off of its length at any position since 2026.11, so a conductor
whose length exceeds a whole number of pieces only by round-off gets that
whole number near the origin too (a 60 m counterpoise in 10 m pieces:
7 → 6 pieces in 876 of 3600 directions, audit W2B-03).

Mixed worlds (user decision D5)
-------------------------------
The frame cannot help a world that mixes projected and local
coordinates: electrodes at UTM coordinates and a node or electrode given
near (0, 0), typically by mistake (a return electrode, a cabinet or a
cable end entered in a local system). Such a world is solved as given:
its parts lie some 33 000 km apart and a conductor between them is that
long (W2B-04, a 12 m × 7 m outline and a 3 m rod at UTM coordinates: a
return rod at local (80, 15) m raises Z_in by 0.43 %; a
PEN cable to a cabinet at local coordinates is 33 118 km long, Z_in
7494 + j25396 Ω instead of 5.39 + j0.19 Ω at 50 Hz with Carson's earth
return, and the Sommerfeld earth return does not finish). Its round-off is that of the projected world
(1.5e-11 against 1.0e-11, point kernel). :func:`coordinate_span`
classifies the points of a world: a point within :data:`LOCAL_RADIUS_M`
of the origin is local, one beyond :data:`PROJECTED_RADIUS_M` projected
(UTM northings outside the tropics, every easting with its zone prefix,
Gauss-Krüger coordinates). A world with both is *mixed*: every solve in a
local frame (:meth:`LocalFrame.of`) warns with
:class:`~groundfield.diagnostics.MixedCoordinatesWarning`, once, before
anything is discretised, and :func:`~groundfield.diagnostics.diagnose`
reports it as a warning (category ``coordinates``), together with a world
whose horizontal extent exceeds :data:`MAX_EXTENT_M` (as information).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__: list[str] = []

#: Horizontal coordinate magnitude in m beyond which a solve runs in a local
#: frame. Below it the coordinates are local in practice (a generated
#: network, an OSM layout around its centre) and solved unchanged; projected
#: coordinates (UTM, Gauss-Krueger) lie far above it.
LOCAL_FRAME_THRESHOLD_M = 1.0e4

#: Horizontal distance from the origin in m up to which (inclusive) the
#: coordinate check takes a point as given in local coordinates (user
#: decision D5).
LOCAL_RADIUS_M = 1.0e4

#: Horizontal distance from the origin in m beyond which the coordinate
#: check takes a point as given in projected coordinates (user decision D5).
PROJECTED_RADIUS_M = 1.0e6

#: Horizontal extent of a world in m (the diagonal of its bounding box)
#: above which :func:`~groundfield.diagnostics.diagnose` reports it (user
#: decision D5).
MAX_EXTENT_M = 1.0e5

#: Point fields of the electrode, conductor and junction models.
_POINT_FIELDS = ("position", "center", "start", "end", "corner")

#: Number of names a message lists per group.
_MAX_NAMES = 5


def _model_points(model: Any) -> list[tuple[float, float, float]]:
    """Every point of an electrode, conductor or junction model, in m."""
    fields = type(model).model_fields
    points = [getattr(model, name) for name in _POINT_FIELDS if name in fields]
    if "vertices" in fields:
        points.extend(model.vertices)
    return [p for p in points if p is not None]


def _models(world: World) -> list[Any]:
    """The electrodes, conductors and junctions of ``world``, in that order."""
    return [*world.electrodes, *world.conductors, *world.junctions]


def _world_xy(world: World) -> np.ndarray:
    """Horizontal coordinates ``(N, 2)`` in m of every point of ``world``."""
    points = [p for model in _models(world) for p in _model_points(model)]
    if not points:
        return np.zeros((0, 2))
    return np.asarray(points, dtype=float)[:, :2]


@dataclass(frozen=True)
class CoordinateSpan:
    """Where the points of a world lie (the coordinate check, user decision D5).

    Every electrode, junction and conductor is classified by its own points
    (horizontal distance from the origin): *near* with a point within
    :data:`LOCAL_RADIUS_M` (inclusive), *far* with a point beyond
    :data:`PROJECTED_RADIUS_M`, *spanning* with both (a conductor, or a
    polyline, between local and projected points).

    Attributes
    ----------
    near
        Names of the elements with near points only, in world order.
    far
        Names of the elements with far points only.
    spanning
        Names of the elements with near and far points.
    spanning_lengths_m
        Length in m of each spanning element that is a conductor, ``None``
        for an electrode (same order as ``spanning``).
    extent_m
        Diagonal of the horizontal bounding box of every point, in m.
    bounds_xy
        ``(x_min, x_max, y_min, y_max)`` of that box, in m.
    """

    near: tuple[str, ...]
    far: tuple[str, ...]
    spanning: tuple[str, ...]
    spanning_lengths_m: tuple[float | None, ...]
    extent_m: float
    bounds_xy: tuple[float, float, float, float]

    @property
    def mixed(self) -> bool:
        """Whether points lie both within the local and beyond the projected radius."""
        return bool(self.spanning) or (bool(self.near) and bool(self.far))

    def message(self) -> str:
        """One sentence on a mixed world (the warning and the diagnostics finding)."""
        groups = []
        if self.near:
            groups.append(
                f"{_listed([f'{n!r}' for n in self.near])} within "
                f"{LOCAL_RADIUS_M / 1e3:g} km of the origin"
            )
        if self.far:
            groups.append(
                f"{_listed([f'{n!r}' for n in self.far])} beyond "
                f"{PROJECTED_RADIUS_M / 1e3:g} km from it"
            )
        if self.spanning:
            items = [
                repr(n) if length is None else f"{n!r}, {_km(length)} long"
                for n, length in zip(
                    self.spanning, self.spanning_lengths_m, strict=True
                )
            ]
            groups.append(f"{_listed(items)} with points in both")
        return (
            "the world mixes local and projected coordinates: "
            + ", ".join(groups)
            + f"; horizontal extent {_km(self.extent_m)}. The solve takes every "
            "point as given, so the two groups lie that far apart and a conductor "
            "between them is that long; a node or electrode given in the wrong "
            "coordinate system is the usual cause (a return electrode, a cabinet or "
            "a cable end entered in local coordinates in a UTM world). Give every "
            "point in one system."
        )


def _km(metres: float) -> str:
    """``metres`` as whole kilometres, a space as the thousands separator."""
    return f"{metres / 1e3:,.0f} km".replace(",", " ")


def _listed(items: list[str]) -> str:
    """``"1 element ('a')"`` / ``"7 elements ('a', ..., 'e' and 2 more)"``."""
    shown = ", ".join(items[:_MAX_NAMES])
    if len(items) > _MAX_NAMES:
        shown += f" and {len(items) - _MAX_NAMES} more"
    return f"{len(items)} element{'s' if len(items) != 1 else ''} ({shown})"


def coordinate_span(world: World) -> CoordinateSpan | None:
    """Classify the points of ``world`` as local or projected (user decision D5).

    Parameters
    ----------
    world
        World whose electrodes, conductors and junctions are checked.

    Returns
    -------
    CoordinateSpan or None
        The elements with points within :data:`LOCAL_RADIUS_M`, beyond
        :data:`PROJECTED_RADIUS_M` of the origin (horizontal distance) or
        both, and the horizontal extent of the world; ``None`` for a world
        without points.
    """
    from groundfield.conductors.conductor import Conductor

    xy = _world_xy(world)
    if xy.size == 0:
        return None
    near: list[str] = []
    far: list[str] = []
    spanning: list[str] = []
    lengths: list[float | None] = []
    for model in _models(world):
        pts = _model_points(model)
        if not pts:
            continue
        r = np.hypot(*np.asarray(pts, dtype=float)[:, :2].T)
        is_near = bool(np.any(r <= LOCAL_RADIUS_M))
        is_far = bool(np.any(r > PROJECTED_RADIUS_M))
        if is_near and is_far:
            spanning.append(model.name)
            lengths.append(
                float(model.length) if isinstance(model, Conductor) else None
            )
        elif is_near:
            near.append(model.name)
        elif is_far:
            far.append(model.name)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    return CoordinateSpan(
        near=tuple(near),
        far=tuple(far),
        spanning=tuple(spanning),
        spanning_lengths_m=tuple(lengths),
        extent_m=float(math.hypot(*(hi - lo))),
        bounds_xy=(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])),
    )


def _warn_if_mixed(world: World, xy: np.ndarray) -> None:
    """Emit :class:`MixedCoordinatesWarning` once for a mixed world (D5)."""
    if xy.size == 0:
        return
    r = np.hypot(xy[:, 0], xy[:, 1])
    if not (float(r.min()) <= LOCAL_RADIUS_M and float(r.max()) > PROJECTED_RADIUS_M):
        return
    span = coordinate_span(world)
    if span is None or not span.mixed:  # pragma: no cover - same rule as above
        return
    from groundfield.diagnostics import MixedCoordinatesWarning
    from groundfield.solver.registry import warn_external

    message = span.message()
    warn_external(
        message[0].upper() + message[1:] + " (diagnose() reports it under "
        "'coordinates'.)",
        MixedCoordinatesWarning,
    )


def _moved(model: Any, shift: np.ndarray) -> Any:
    """Copy of ``model`` with every point moved horizontally by ``shift`` (m)."""
    fields = type(model).model_fields
    dx, dy = float(shift[0]), float(shift[1])

    def move(p):  # noqa: ANN001, ANN202
        return (p[0] + dx, p[1] + dy, p[2])

    update: dict[str, Any] = {
        name: move(getattr(model, name))
        for name in _POINT_FIELDS
        if name in fields and getattr(model, name) is not None
    }
    if "vertices" in fields:
        update["vertices"] = [move(v) for v in model.vertices]
    return model.model_copy(update=update) if update else model


def _origin_of(xy: np.ndarray) -> np.ndarray | None:
    """Frame origin ``(x0, y0, 0)`` in m for horizontal points ``xy``, or ``None``."""
    if xy.size == 0 or float(np.max(np.abs(xy))) <= LOCAL_FRAME_THRESHOLD_M:
        return None
    centre = np.round(0.5 * (xy.min(axis=0) + xy.max(axis=0)))
    return np.array([float(centre[0]), float(centre[1]), 0.0])


def local_origin(world: World) -> np.ndarray | None:
    """Origin of the local frame of ``world``, or ``None`` near the origin.

    Parameters
    ----------
    world
        World whose electrodes, conductors and junctions span the frame.

    Returns
    -------
    numpy.ndarray or None
        ``(x0, y0, 0)`` in m, the centre of the horizontal bounding box of
        every point of the world rounded to whole metres, if a horizontal
        coordinate exceeds :data:`LOCAL_FRAME_THRESHOLD_M`; ``None``
        otherwise.
    """
    xy = _world_xy(world)
    if xy.size == 0:
        return None
    return _origin_of(xy)


def points_origin(points: np.ndarray) -> np.ndarray | None:
    """Origin of a local frame for point sources ``points`` ``(N, 3)`` in m.

    The rule of :func:`local_origin` applied to the points themselves.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        return None
    return _origin_of(points[:, :2])


@dataclass(frozen=True)
class LocalFrame:
    """A world in its local frame.

    Attributes
    ----------
    world
        The world shifted by ``-origin`` (the caller's world itself when
        ``origin`` is ``None``).
    origin
        Frame origin ``(x0, y0, 0)`` in m, or ``None``.
    """

    world: World
    origin: np.ndarray | None

    @classmethod
    def of(cls, world: World) -> LocalFrame:
        """The local frame of ``world`` (:func:`local_origin`).

        Called once by every solve in a local frame (the image family, the
        ``mom`` family, ``solve_mutual_*``) before anything is discretised.

        Warns
        -----
        MixedCoordinatesWarning
            If points of ``world`` lie both within :data:`LOCAL_RADIUS_M`
            and beyond :data:`PROJECTED_RADIUS_M` of the origin (a world
            that mixes local and projected coordinates, user decision D5);
            the world is solved as given.
        """
        xy = _world_xy(world)
        _warn_if_mixed(world, xy)
        origin = _origin_of(xy) if xy.size else None
        if origin is None:
            return cls(world=world, origin=None)
        shift = -origin
        local = world.model_copy(
            update={
                "electrodes": [_moved(e, shift) for e in world.electrodes],
                "conductors": [_moved(c, shift) for c in world.conductors],
                "junctions": [_moved(j, shift) for j in world.junctions],
            }
        )
        return cls(world=local, origin=origin)

    @contextmanager
    def active(self) -> Iterator[None]:
        """Evaluate the discretisation tolerances in the user's frame."""
        if self.origin is None:
            yield
            return
        from groundfield.geometry.primitives import _frame_origin

        with _frame_origin(self.origin):
            yield

    def to_user(self, result: FieldResult) -> FieldResult:
        """``result`` with its point-source positions in the user's frame."""
        if self.origin is None:
            return result
        ox, oy = float(self.origin[0]), float(self.origin[1])
        result.point_sources = [
            ps.model_copy(
                update={
                    "position": (
                        ps.position[0] + ox,
                        ps.position[1] + oy,
                        ps.position[2],
                    )
                }
            )
            for ps in result.point_sources
        ]
        return result

    def local_points(self, points: np.ndarray) -> np.ndarray:
        """``points`` ``(M, 3)`` in m of the user's frame, in the local frame."""
        if self.origin is None:
            return points
        return np.asarray(points, dtype=float) - self.origin
