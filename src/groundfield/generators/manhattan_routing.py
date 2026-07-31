r"""Manhattan-routed pathfinding for the PEN cable layout.

The :func:`route_manhattan` helper finds an axis-aligned (no
diagonals) polyline from a start to an end point on a coarse
grid that avoids a list of obstacle rectangles. It powers the
PEN-cable router in
:class:`~groundfield.generators.ortsnetz_builder.OrtsnetzLayout`
where buildings (their axis-aligned bounding rectangles) act as
hard obstacles -- a real LV cable cannot pass *through* a
foundation.

Algorithm
---------
* A 4-connected A\* search on a regular Manhattan grid of cell
  size :math:`g` (the user-configurable ``min_segment_length_m``).
* Heuristic: Manhattan distance to the goal cell.
* Cell ``(i, j)`` is blocked iff its central rectangle
  :math:`[c_x - g/2,\,c_x + g/2] \times [c_y - g/2,\,c_y + g/2]`
  overlaps any obstacle (after inflating the obstacle by an
  optional ``clearance_m`` safety margin). The start and end
  cells are exempt so that a substation or KVS placed close to
  -- or even on -- a building footprint can still be reached.
* After A\* returns, consecutive collinear cells are merged so
  that every returned waypoint marks an actual *corner* in the
  polyline. The grid origin is locked to ``start`` so the path's
  first waypoint is exactly the requested ``start``; an
  axis-aligned bridge from the last grid cell to the requested
  ``end`` is appended at the end (one or two extra segments,
  always axis-aligned). The bridge obeys the same
  ``escape_radius_m`` blind-spot rule as the grid cells, so an
  anchor that lands *inside* a foundation polygon is routable
  instead of raising.

The implementation is intentionally pure-Python and obstacle-
free at import time: :mod:`shapely` is never required even
though obstacle inputs may originate from
:meth:`BuildingFootprint.axis_aligned_bounding_rectangle`.
"""

from __future__ import annotations

import heapq
import math
from typing import Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ObstacleBox",
    "route_manhattan",
    "segment_intersects_box",
    "merge_collinear_waypoints",
]


# ---------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------


class ObstacleBox(BaseModel):
    """Axis-aligned bounding box used as a routing obstacle.

    Attributes
    ----------
    x_min, y_min, x_max, y_max
        Lower-left and upper-right corner in metres.
    name
        Optional label (typically the OSM building id or a human-
        readable house identifier). Pure metadata; the router
        does not consume it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    name: Optional[str] = Field(
        default=None,
        description="Optional label for diagnostics / plots.",
    )

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def inflated(self, padding_m: float) -> "ObstacleBox":
        """Return a copy enlarged by ``padding_m`` on every side."""
        if padding_m <= 0.0:
            return self
        return ObstacleBox(
            x_min=self.x_min - padding_m,
            y_min=self.y_min - padding_m,
            x_max=self.x_max + padding_m,
            y_max=self.y_max + padding_m,
            name=self.name,
        )

    def contains_point(self, x: float, y: float, *, tol: float = 0.0) -> bool:
        return (
            self.x_min - tol <= x <= self.x_max + tol
            and self.y_min - tol <= y <= self.y_max + tol
        )

    def rectangle_overlaps(
        self, cx: float, cy: float, half: float,
    ) -> bool:
        """Does the square ``[cx-half, cx+half]^2`` overlap the box?"""
        return (
            cx + half > self.x_min
            and cx - half < self.x_max
            and cy + half > self.y_min
            and cy - half < self.y_max
        )


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------


def segment_intersects_box(
    p0: tuple[float, float],
    p1: tuple[float, float],
    box: ObstacleBox,
    *,
    tol: float = 1e-9,
) -> bool:
    r"""True if the *axis-aligned* segment $p_0 \to p_1$ enters ``box``.

    The segment is assumed axis-aligned (one of ``x`` or ``y``
    coincides on both endpoints). Touching the box on the open
    boundary is *not* considered an intersection -- that is what
    makes the router happy when the cable runs along a street
    that is exactly tangent to a foundation.

    Parameters
    ----------
    p0, p1
        Segment endpoints.
    box
        Obstacle box.
    tol
        Small slack so a segment running on the boundary of the
        box does not count as intersecting.
    """
    x0, y0 = p0
    x1, y1 = p1
    if math.isclose(y0, y1, abs_tol=tol):
        # Horizontal segment at y = y0
        if y0 <= box.y_min + tol or y0 >= box.y_max - tol:
            return False
        x_low, x_high = (x0, x1) if x0 <= x1 else (x1, x0)
        return x_low < box.x_max - tol and x_high > box.x_min + tol
    if math.isclose(x0, x1, abs_tol=tol):
        # Vertical segment at x = x0
        if x0 <= box.x_min + tol or x0 >= box.x_max - tol:
            return False
        y_low, y_high = (y0, y1) if y0 <= y1 else (y1, y0)
        return y_low < box.y_max - tol and y_high > box.y_min + tol
    raise ValueError(
        f"segment_intersects_box: segment ({p0}, {p1}) is not "
        "axis-aligned."
    )


def merge_collinear_waypoints(
    waypoints: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Drop interior waypoints that lie on the line of their neighbours.

    The result keeps the polyline's geometric shape (every actual
    corner) but discards the artificial mid-cell stops that the
    A\\* search produced. Endpoints are always kept.
    """
    if len(waypoints) < 3:
        return list(waypoints)
    out: list[tuple[float, float]] = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        prev = out[-1]
        cur = waypoints[i]
        nxt = waypoints[i + 1]
        dx1, dy1 = cur[0] - prev[0], cur[1] - prev[1]
        dx2, dy2 = nxt[0] - cur[0], nxt[1] - cur[1]
        # Both segments must be along the same axis with the same sign
        # to be collinear and absorbable.
        same_x = math.isclose(dx1, 0.0) and math.isclose(dx2, 0.0)
        same_y = math.isclose(dy1, 0.0) and math.isclose(dy2, 0.0)
        if same_x and dy1 * dy2 > 0.0:
            continue
        if same_y and dx1 * dx2 > 0.0:
            continue
        out.append(cur)
    out.append(waypoints[-1])
    return out


# ---------------------------------------------------------------------
# Manhattan A*
# ---------------------------------------------------------------------


def route_manhattan(
    start: tuple[float, float],
    end: tuple[float, float],
    obstacles: Iterable[ObstacleBox] = (),
    *,
    grid_size: float = 10.0,
    clearance_m: float = 0.5,
    escape_radius_m: Optional[float] = None,
    max_iterations: int = 200_000,
) -> list[tuple[float, float]]:
    r"""Route a Manhattan polyline from ``start`` to ``end``.

    Parameters
    ----------
    start, end
        Endpoint coordinates in metres. The grid origin is locked
        to ``start`` so the first waypoint is *exactly* ``start``;
        an axis-aligned bridge from the last grid cell to ``end``
        is appended.
    obstacles
        Building obstacle boxes the cable must avoid.
    grid_size
        Cell size in metres. The minimum segment length of the
        returned polyline (after merging collinear segments) is at
        least :func:`grid_size` for any leg interior to the path;
        the trailing bridge may be shorter when ``end`` does not
        align with the grid.
    clearance_m
        Additional padding around each obstacle. Defaults to
        ``0.5`` m so the cable keeps a small safety distance from
        every foundation wall.
    escape_radius_m
        Inside a disk of this radius around ``start`` *and* around
        ``end``, obstacles are *not* enforced. This is the "release
        valve" for real-OSM layouts where the substation or a KVS
        lat/lon lands inside or very near a foundation polygon and
        every neighbouring grid cell would otherwise be blocked --
        the cable physically has to exit the substation cubicle
        somehow, so allowing a short blind-spot around each anchor
        models reality and prevents spurious
        :class:`RuntimeError` from
        ``A* exhausted``. ``None`` (default) maps to
        ``grid_size`` (i.e. exactly one cell of slack around each
        anchor). Pass ``0.0`` to restore the strict pre-v0.7
        behaviour, or a larger value when an anchor lands inside a
        dense city block.

        The escape zone applies to the A\* grid cells *and* to the
        trailing bridge from the last cell centre to ``end`` (since
        0.15.0): a bridge segment whose two endpoints both lie
        inside one anchor's escape disk is exempt from obstacle
        testing, and any obstacle that *contains* an anchor is
        ignored by the bridge altogether — a cable whose endpoint
        is inside a foundation polygon cannot avoid that polygon,
        so enforcing it there would only turn a solvable route into
        a hard error. Before 0.15.0 the bridge ignored
        ``escape_radius_m`` entirely, so an ``end`` anchor inside a
        footprint raised regardless of how large the escape radius
        was set.
    max_iterations
        Upper bound on the A\* loop body. Increase for very large
        layouts (hundreds of obstacles spread across kilometres).

    Returns
    -------
    list of (float, float)
        Polyline waypoints. ``len(out) >= 2``; the first entry is
        ``start`` and the last entry is ``end``. Consecutive
        entries differ in exactly one coordinate (Manhattan).

    Raises
    ------
    RuntimeError
        When no path exists (obstacle field encloses the
        substation or KVS), or when neither candidate bridge from
        the last grid cell to ``end`` clears the obstacles that the
        escape zone does not exempt.
    """
    if grid_size <= 0.0:
        raise ValueError(
            f"route_manhattan: grid_size must be positive, got {grid_size}."
        )
    if escape_radius_m is None:
        escape_radius_m = grid_size
    if escape_radius_m < 0.0:
        raise ValueError(
            "route_manhattan: escape_radius_m must be >= 0, got "
            f"{escape_radius_m}."
        )

    obstacles = [o.inflated(clearance_m) for o in obstacles]

    # Grid origin locked to start so cell (0, 0) is exactly ``start``.
    ox, oy = start
    half = grid_size / 2.0

    # Bounding cells: include end and every obstacle, with margin.
    margin_cells = 3
    xs = [start[0], end[0]] + [o.x_min for o in obstacles] + [o.x_max for o in obstacles]
    ys = [start[1], end[1]] + [o.y_min for o in obstacles] + [o.y_max for o in obstacles]
    i_min = int(math.floor((min(xs) - ox) / grid_size)) - margin_cells
    i_max = int(math.ceil((max(xs) - ox) / grid_size)) + margin_cells
    j_min = int(math.floor((min(ys) - oy) / grid_size)) - margin_cells
    j_max = int(math.ceil((max(ys) - oy) / grid_size)) + margin_cells

    def cell_center(i: int, j: int) -> tuple[float, float]:
        return (ox + i * grid_size, oy + j * grid_size)

    def world_to_cell(x: float, y: float) -> tuple[int, int]:
        return (
            int(round((x - ox) / grid_size)),
            int(round((y - oy) / grid_size)),
        )

    start_cell = (0, 0)
    end_cell = world_to_cell(*end)

    # Snap end into the valid range (paranoia).
    end_cell = (
        max(i_min, min(i_max, end_cell[0])),
        max(j_min, min(j_max, end_cell[1])),
    )

    # Pre-compute the "escape" cell sets: every cell whose centre
    # is within ``escape_radius_m`` of the start (or end) endpoint
    # is exempt from obstacle blocking, no matter how dense the
    # surrounding building cluster is.
    escape_r2 = escape_radius_m * escape_radius_m

    def _near_anchor(
        p: tuple[float, float], anchor: tuple[float, float],
    ) -> bool:
        """Is ``p`` inside ``anchor``'s escape disk?"""
        dx = p[0] - anchor[0]
        dy = p[1] - anchor[1]
        return dx * dx + dy * dy <= escape_r2

    def _in_escape_zone(i: int, j: int) -> bool:
        if escape_radius_m <= 0.0:
            return False
        center = cell_center(i, j)
        return any(_near_anchor(center, a) for a in (start, end))

    def is_blocked(i: int, j: int) -> bool:
        if (i, j) == start_cell or (i, j) == end_cell:
            return False
        if _in_escape_zone(i, j):
            return False
        cx, cy = cell_center(i, j)
        for o in obstacles:
            if o.rectangle_overlaps(cx, cy, half):
                return True
        return False

    # A*
    open_heap: list[tuple[int, int, tuple[int, int]]] = []
    counter = 0  # tie-breaker for the heap (stable order, no compare on tuples)
    heapq.heappush(
        open_heap,
        (abs(end_cell[0]) + abs(end_cell[1]), counter, start_cell),
    )
    g_score: dict[tuple[int, int], int] = {start_cell: 0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    iterations = 0
    found = False
    current = start_cell
    while open_heap:
        if iterations > max_iterations:
            raise RuntimeError(
                "route_manhattan: A* exceeded "
                f"{max_iterations} iterations before reaching end "
                f"= {end}. Increase max_iterations or grid_size, or "
                "relax clearance_m."
            )
        iterations += 1
        _, _, current = heapq.heappop(open_heap)
        if current == end_cell:
            found = True
            break
        i0, j0 = current
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            i, j = i0 + di, j0 + dj
            if not (i_min <= i <= i_max and j_min <= j <= j_max):
                continue
            if is_blocked(i, j):
                continue
            tentative = g_score[current] + 1
            if tentative < g_score.get((i, j), 10**18):
                g_score[(i, j)] = tentative
                came_from[(i, j)] = current
                f = tentative + abs(end_cell[0] - i) + abs(end_cell[1] - j)
                counter += 1
                heapq.heappush(open_heap, (f, counter, (i, j)))

    if not found:
        # Diagnose which side is enclosed so the caller gets an
        # actionable suggestion (move that anchor, widen the
        # escape_radius_m, or drop clearance_m).
        start_neighbours_open = sum(
            not is_blocked(start_cell[0] + di, start_cell[1] + dj)
            and i_min <= start_cell[0] + di <= i_max
            and j_min <= start_cell[1] + dj <= j_max
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        end_neighbours_open = sum(
            not is_blocked(end_cell[0] + di, end_cell[1] + dj)
            and i_min <= end_cell[0] + di <= i_max
            and j_min <= end_cell[1] + dj <= j_max
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        offender = "start" if start_neighbours_open == 0 else (
            "end" if end_neighbours_open == 0 else "both endpoints"
        )
        raise RuntimeError(
            "route_manhattan: no Manhattan path from "
            f"{start} to {end}. The {offender} anchor is enclosed "
            f"by obstacles (start has {start_neighbours_open}/4 free "
            f"neighbours, end has {end_neighbours_open}/4). "
            "Suggested fixes (in order of safety): "
            "(1) increase ``escape_radius_m`` (currently "
            f"{escape_radius_m:.1f} m -- raise to e.g. "
            f"{2 * escape_radius_m:.1f} m); "
            "(2) reduce ``clearance_m`` (currently "
            f"{clearance_m:.2f} m); "
            "(3) move the offending anchor by a few metres so it "
            "lands clearly outside any foundation footprint."
        )

    # Reconstruct path of cell centres.
    path_cells: list[tuple[int, int]] = [current]
    while current in came_from:
        current = came_from[current]
        path_cells.append(current)
    path_cells.reverse()
    waypoints = [cell_center(*c) for c in path_cells]

    # Anchor first waypoint at the *actual* start (cell (0,0) == start).
    waypoints[0] = start

    # Bridge from last cell centre to ``end`` (axis-aligned).
    last = waypoints[-1]
    if last != end:
        dx = end[0] - last[0]
        dy = end[1] - last[1]
        if math.isclose(dx, 0.0) or math.isclose(dy, 0.0):
            waypoints.append(end)
        else:
            # The trailing bridge honours the same escape zone as
            # ``is_blocked`` does inside the A* loop. Two exemptions,
            # both active only for ``escape_radius_m > 0``:
            #
            # 1. An obstacle that *contains* an anchor is dropped for
            #    the whole bridge. The cable ends inside that
            #    foundation polygon, so no axis-aligned bridge can
            #    avoid it -- enforcing it would turn the escape valve
            #    into a guaranteed failure (the situation the valve
            #    exists for).
            # 2. A bridge segment whose *both* endpoints lie inside
            #    one anchor's escape disk is exempt from every
            #    obstacle, mirroring the per-cell blind spot.
            if escape_radius_m > 0.0:
                bridge_obstacles = [
                    o
                    for o in obstacles
                    if not (
                        o.contains_point(*start) or o.contains_point(*end)
                    )
                ]
            else:
                bridge_obstacles = obstacles

            def _bridge_crosses(
                a: tuple[float, float], b: tuple[float, float],
            ) -> bool:
                """Does bridge segment ``a -> b`` hit an enforced box?"""
                if escape_radius_m > 0.0 and any(
                    _near_anchor(a, anc) and _near_anchor(b, anc)
                    for anc in (start, end)
                ):
                    return False
                return any(
                    segment_intersects_box(a, b, o)
                    for o in bridge_obstacles
                )

            # Choose the bridge corner that does not cross an
            # obstacle. Two options: go-x-first or go-y-first.
            corner_x_first = (end[0], last[1])
            corner_y_first = (last[0], end[1])
            for corner in (corner_x_first, corner_y_first):
                if not (
                    _bridge_crosses(last, corner)
                    or _bridge_crosses(corner, end)
                ):
                    waypoints.append(corner)
                    waypoints.append(end)
                    break
            else:
                # Neither bridge works without an obstacle -- give up.
                raise RuntimeError(
                    "route_manhattan: cannot bridge final cell "
                    f"{last} to end {end} without crossing an "
                    f"obstacle (escape_radius_m = {escape_radius_m:.1f} m). "
                    "Suggested fixes: reduce grid_size (currently "
                    f"{grid_size:.1f} m), increase escape_radius_m so the "
                    "bridge fits inside the anchor's blind spot, or reduce "
                    f"clearance_m (currently {clearance_m:.2f} m)."
                )

    return merge_collinear_waypoints(waypoints)
