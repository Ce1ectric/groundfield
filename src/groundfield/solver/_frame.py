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
  subtraction of two nearby doubles is exact (Sterbenz), so the copy
  describes exactly the same geometry;
* the round-off tolerances of the discretisation (straight runs and
  segment counts, :mod:`groundfield.geometry.primitives`) are evaluated
  with the coordinates of the user's frame, :math:`p + o`, so the copy is
  discretised with the same decisions as the original;
* the result is reported in the user's coordinates (point-source
  positions shifted back by :math:`+o`), and
  :meth:`~groundfield.solver.result.FieldResult.potential` evaluates
  fields in a local frame of the sources in the same way.

A world within the threshold of the origin is solved exactly as before
(bit for bit): the round-off of coordinates below 10 km is below
1.8e-12 m.
"""

from __future__ import annotations

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

#: Point fields of the electrode, conductor and junction models.
_POINT_FIELDS = ("position", "center", "start", "end", "corner")


def _model_points(model: Any) -> list[tuple[float, float, float]]:
    """Every point of an electrode, conductor or junction model, in m."""
    fields = type(model).model_fields
    points = [getattr(model, name) for name in _POINT_FIELDS if name in fields]
    if "vertices" in fields:
        points.extend(model.vertices)
    return [p for p in points if p is not None]


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
    points = [
        p
        for group in (world.electrodes, world.conductors, world.junctions)
        for model in group
        for p in _model_points(model)
    ]
    if not points:
        return None
    return _origin_of(np.asarray(points, dtype=float)[:, :2])


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
        """The local frame of ``world`` (:func:`local_origin`)."""
        origin = local_origin(world)
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
