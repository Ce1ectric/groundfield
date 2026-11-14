r"""Manhattan routing of PEN cables and service drops (layout builder).

Part of :mod:`groundfield.generators.layout`. The router lays PEN cables
as axis-aligned polylines on a grid of ``min_segment_length_m`` cells
(4-connected A\* in :func:`~groundfield.generators.manhattan_routing.route_manhattan`)
around the axis-aligned bounding boxes of the building footprints, and
taps every house to the nearest cable with a short one- or two-segment
stub that does not cross another footprint.

The routes are plain data (:class:`PenCable`, :class:`BuildingConnection`);
:meth:`LvNetworkLayout.plan <groundfield.generators.layout.LvNetworkLayout.plan>`
turns them into the junctions and branches of a
:class:`~groundfield.generators.plan.NetworkPlan`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple, Optional

from pydantic import BaseModel, ConfigDict

from groundfield.generators.manhattan_routing import (
    ObstacleBox,
    merge_collinear_waypoints,
    route_manhattan,
    segment_intersects_box,
)
from groundfield.geo.footprint import BuildingFootprint

if TYPE_CHECKING:  # pragma: no cover - typing only
    from groundfield.generators.layout.model import CabinetPlacement

__all__ = [
    "BuildingConnection",
    "LateralCoverageResult",
    "PenCable",
]


# ---------------------------------------------------------------------
# Route snapshots
# ---------------------------------------------------------------------


class PenCable(BaseModel):
    """One PEN cable as a polyline of axis-aligned waypoints.

    Attributes
    ----------
    name
        Unique cable id used for diagnostics, ``to_world`` naming
        and plot legends.
    start_anchor
        Name of the anchor (substation or cable cabinet) the cable roots at.
    waypoints
        Ordered list of $(x, y)$ corners. The first entry equals
        the start anchor's position and consecutive entries differ
        in exactly one coordinate (Manhattan).
    end_anchor
        Optional name of the anchor (cable cabinet) the cable terminates at.
        ``None`` when the cable runs to free coordinates (a
        :emphasis:`stub end` used as a tap for nearby houses but
        not for further trunking).
    grid_size_m
        The grid size used when this cable was routed -- kept on
        the model so :meth:`LvNetworkLayout.connect_buildings` can
        align stubs with the trunk.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    start_anchor: str
    waypoints: list[tuple[float, float]]
    end_anchor: Optional[str] = None
    grid_size_m: float = 10.0

    def segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return list(zip(self.waypoints[:-1], self.waypoints[1:], strict=True))

    def total_length_m(self) -> float:
        return sum(
            abs(p1[0] - p0[0]) + abs(p1[1] - p0[1]) for p0, p1 in self.segments()
        )


class BuildingConnection(BaseModel):
    """Stub from one house's foundation centre to its serving cable.

    Attributes
    ----------
    house_idx
        Index into :attr:`LvNetworkLayout.footprints`.
    cable_name
        Name of the :class:`PenCable` the stub taps into.
    stub_waypoints
        Polyline from the house centroid to the tap point on the
        cable. ``[centroid, ..., tap_point]``. The interior
        waypoints (if any) are the Manhattan corners.
    tap_point_xy
        The tap point on the cable polyline nearest to the house centroid
        (Manhattan), the end of ``stub_waypoints``. The materialised drop
        of a house with a foundation, ring or strip taps the cable at the
        closest pair of cable and those wires instead (user decisions Q41,
        Q43, see :meth:`LvNetworkLayout.to_world
        <groundfield.generators.layout.LvNetworkLayout.to_world>`); a house
        with rods only is tapped here, or at the nearest point where the
        cable is built when this point lies inside a site foundation (audit
        R6A-01).
    """

    model_config = ConfigDict(extra="forbid")

    house_idx: int
    cable_name: str
    stub_waypoints: list[tuple[float, float]]
    tap_point_xy: tuple[float, float]

    def total_length_m(self) -> float:
        return sum(
            abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
            for p0, p1 in zip(
                self.stub_waypoints[:-1],
                self.stub_waypoints[1:],
                strict=True,
            )
        )


class LateralCoverageResult(NamedTuple):
    """Outcome of :meth:`LvNetworkLayout.connect_all_buildings`.

    Attributes
    ----------
    laterals
        Names of the PEN lateral cables that were added to reach
        otherwise-unconnected houses, in the order they were created.
    islands
        Indices of footprints that still could not be connected -- houses
        enclosed by other footprints on every side, for which no
        obstacle-free service stub exists even after adding a lateral. An
        empty list means full coverage.
    connections
        Snapshot of :attr:`LvNetworkLayout.connections` after the pass (the
        tap from every connected house to its serving cable).
    """

    laterals: list[str]
    islands: list[int]
    connections: list["BuildingConnection"]


# ---------------------------------------------------------------------
# Routing methods of the layout builder
# ---------------------------------------------------------------------


class _RoutingMixin:
    """PEN routing methods of :class:`~groundfield.generators.layout.LvNetworkLayout`."""

    if TYPE_CHECKING:  # pragma: no cover - attributes of the layout model
        footprints: list[BuildingFootprint]
        substation_name: str
        substation_xy: tuple[float, float]
        cabinet_placements: list["CabinetPlacement"]
        pen_cables: list[PenCable]
        connections: list[BuildingConnection]
        obstacle_clearance_m: float

        def _anchor_position(self, name: str) -> tuple[float, float]: ...

        def lat_lon_to_xy(
            self, lat_deg: float, lon_deg: float
        ) -> tuple[float, float]: ...

    def add_pen_cable(
        self,
        *,
        start: str,
        end_xy: Optional[tuple[float, float]] = None,
        end_lat_lon: Optional[tuple[float, float]] = None,
        end: Optional[str] = None,
        name: Optional[str] = None,
        min_segment_length_m: float = 10.0,
        clearance_m: Optional[float] = None,
        escape_radius_m: Optional[float] = None,
    ) -> PenCable:
        """Add a PEN cable routed Manhattan-style.

        Exactly one of ``end``, ``end_xy`` or ``end_lat_lon`` must
        be set. When ``end`` is the name of another anchor (a cabinet)
        the cable terminates there and the receiving cabinet is
        recorded as :attr:`PenCable.end_anchor`. When ``end_xy`` or
        ``end_lat_lon`` is set, the cable's last waypoint is the
        free coordinate and houses near it can still tap into the
        trunk. ``end_lat_lon`` is projected through the layout's
        :attr:`frame_origin_lat_lon` (raises if no frame origin is
        set).

        Parameters
        ----------
        start
            Name of an anchor present in this layout
            (``substation_name`` or the name of a cabinet).
        end_xy
            Free $(x, y)$ endpoint coordinate in metres.
        end
            Name of the destination anchor. Mutually exclusive
            with ``end_xy``.
        name
            Optional cable id. Auto-generated when ``None``.
        min_segment_length_m
            Grid size of the Manhattan A\\* router. Defaults to
            10 m -- a reasonable street-segment length and an
            order of magnitude longer than a typical foundation
            footprint.
        clearance_m
            Per-call override for :attr:`obstacle_clearance_m`.
        escape_radius_m
            Radius around the start *and* end anchor inside which
            obstacles are not enforced. Defaults to
            ``min_segment_length_m`` (one cell of slack) which is
            usually enough to handle a substation that landed inside
            or right next to a foundation polygon in a real OSM
            extract. Raise this when an anchor is wedged into a
            dense city block; pass ``0.0`` for the strict pre-v0.7
            behaviour.

        Returns
        -------
        PenCable
            The freshly-routed cable. Already registered in
            :attr:`pen_cables`.
        """
        # Exactly one of {end, end_xy, end_lat_lon} must be provided.
        provided = sum(arg is not None for arg in (end, end_xy, end_lat_lon))
        if provided != 1:
            raise ValueError(
                "LvNetworkLayout.add_pen_cable: exactly one of "
                "``end``, ``end_xy`` or ``end_lat_lon`` must be set."
            )
        if name is None:
            name = f"pen_{len(self.pen_cables)}"
        if any(c.name == name for c in self.pen_cables):
            raise ValueError(f"LvNetworkLayout.add_pen_cable: name {name!r} is taken.")

        start_xy = self._anchor_position(start)
        if end is not None:
            end_xy_resolved = self._anchor_position(end)
        elif end_lat_lon is not None:
            end_xy_resolved = self.lat_lon_to_xy(*end_lat_lon)
        else:
            assert end_xy is not None  # narrowing for type checkers
            end_xy_resolved = tuple(end_xy)

        clearance = self.obstacle_clearance_m if clearance_m is None else clearance_m
        obstacles = self._obstacle_boxes()
        waypoints = route_manhattan(
            start=start_xy,
            end=end_xy_resolved,
            obstacles=obstacles,
            grid_size=min_segment_length_m,
            clearance_m=clearance,
            escape_radius_m=escape_radius_m,
        )
        cable = PenCable(
            name=name,
            start_anchor=start,
            waypoints=waypoints,
            end_anchor=end,
            grid_size_m=min_segment_length_m,
        )
        self.pen_cables.append(cable)
        return cable

    def connect_buildings(
        self,
        *,
        only_unconnected: bool = True,
        clearance_m: Optional[float] = None,
    ) -> list[BuildingConnection]:
        """Attach every house to the closest PEN cable.

        For each footprint not yet listed in :attr:`connections`
        (when ``only_unconnected=True``, the default), this method:

        1. Computes the Manhattan-shortest tap point on every
           cable.
        2. Picks the cable whose tap point is closest to the
           house *centroid* and whose stub does not cross another
           foundation (after the same ``clearance_m`` inflation
           used for cable routing).
        3. Builds the stub polyline (at most two segments) and
           records a :class:`BuildingConnection`.

        Houses that cannot be connected to *any* cable (e.g.
        surrounded by foundations on all sides) are reported via
        a :class:`UserWarning` and skipped; they remain
        unconnected and can be revisited after the user adds
        another cable.
        """
        import warnings

        if not self.pen_cables:
            raise RuntimeError(
                "LvNetworkLayout.connect_buildings: no PEN cables "
                "in the layout. Call ``add_pen_cable`` first."
            )
        clearance = self.obstacle_clearance_m if clearance_m is None else clearance_m

        connected_idx = (
            {c.house_idx for c in self.connections} if only_unconnected else set()
        )

        new_connections: list[BuildingConnection] = []
        for i, fp in enumerate(self.footprints):
            if i in connected_idx:
                continue
            centroid = fp.centroid_xy_m()
            # Treat every *other* footprint as a stub obstacle.
            other_obstacles = [
                self._footprint_to_box(self.footprints[j], j).inflated(clearance)
                for j in range(len(self.footprints))
                if j != i
            ]
            best: Optional[BuildingConnection] = None
            best_length = math.inf
            for cable in self.pen_cables:
                tap, stub = self._stub_to_cable(
                    centroid,
                    cable,
                    other_obstacles,
                )
                if stub is None:
                    continue
                length = sum(
                    abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
                    for p0, p1 in zip(stub[:-1], stub[1:], strict=True)
                )
                if length < best_length:
                    best_length = length
                    best = BuildingConnection(
                        house_idx=i,
                        cable_name=cable.name,
                        stub_waypoints=stub,
                        tap_point_xy=tap,
                    )
            if best is None:
                warnings.warn(
                    f"LvNetworkLayout.connect_buildings: house {i} "
                    f"(centroid {centroid}) cannot be tapped to any "
                    "existing cable without crossing another "
                    "footprint. Add another PEN cable or reduce "
                    "``clearance_m``.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            new_connections.append(best)
        self.connections.extend(new_connections)
        return new_connections

    def connect_all_buildings(
        self,
        *,
        source_anchors: Optional[list[str]] = None,
        max_passes: int = 4,
        margin_m: float = 3.0,
        lateral_clearance_m: Optional[float] = None,
        min_segment_length_m: float = 10.0,
        escape_radius_m: Optional[float] = None,
        clearance_m: Optional[float] = None,
    ) -> "LateralCoverageResult":
        r"""Connect *every* house to the LV network, adding laterals as needed.

        :meth:`connect_buildings` only taps each house to the nearest
        existing cable with a short (at most two-segment) stub and skips
        houses whose stub would cross another footprint -- typically houses
        with no cable in reach. This method closes that gap so the layout
        models a real LV network in which every customer is served.

        Strategy ("laterals from the nearest node" model):

        1. Tap whatever is already reachable via :meth:`connect_buildings`.
        2. For every still-unconnected house (farthest from any source node
           first), route an obstacle-avoiding PEN *lateral*
           (:meth:`add_pen_cable`, full Manhattan A\*) from the nearest
           connected source node (substation or a cabinet) to a free point just
           outside the house, then re-tap. Re-tapping frequently connects
           neighbouring houses to the same new lateral for free.
        3. Repeat until no house remains or no further progress is made.

        Houses enclosed by other footprints on every side stay unconnected
        and are reported in :attr:`LateralCoverageResult.islands` rather
        than raising.

        Parameters
        ----------
        source_anchors
            Names of the nodes laterals may originate from. ``None`` (the
            default) uses the substation plus every cabinet galvanically
            connected to it through the existing PEN-cable graph (see
            :meth:`_anchors_connected_to_substation`), so a stranded cabinet is
            never used as a source and cannot create an isolated
            sub-network.
        max_passes
            Maximum number of tap/lateral rounds. Each round adds at most
            one lateral per still-unconnected house.
        margin_m
            Extra distance, beyond the house's bounding-box half-diagonal,
            at which the lateral end point is placed outside the footprint.
        lateral_clearance_m
            Obstacle inflation used when searching for a free lateral end
            point. Defaults to :attr:`obstacle_clearance_m`.
        min_segment_length_m, escape_radius_m, clearance_m
            Forwarded to :meth:`add_pen_cable` for the lateral routing
            (with one finer-grid retry on failure). ``clearance_m`` also
            overrides the tap clearance in :meth:`connect_buildings`.

        Returns
        -------
        LateralCoverageResult
            ``laterals`` (added cable names), ``islands`` (still-unconnected
            footprint indices), and ``connections`` (snapshot after the
            pass).

        Raises
        ------
        RuntimeError
            If ``source_anchors`` resolves to an empty list (e.g. passed
            explicitly as ``[]``).
        """
        import warnings

        if not self.footprints:
            return LateralCoverageResult([], [], list(self.connections))

        clearance = self.obstacle_clearance_m if clearance_m is None else clearance_m
        lat_clear = clearance if lateral_clearance_m is None else lateral_clearance_m

        if source_anchors is None:
            source_anchors = self._anchors_connected_to_substation()
        nodes = [(a, self._anchor_position(a)) for a in source_anchors]
        if not nodes:
            raise RuntimeError(
                "LvNetworkLayout.connect_all_buildings: no source anchor to "
                "originate laterals from. Pass ``source_anchors`` or route "
                "a PEN cable from the substation first."
            )

        def _tap() -> None:
            # connect_buildings warns per skipped house; islands are reported
            # at the end, so silence the per-house noise here. It also raises
            # when there are no cables yet -- skip the tap in that case so the
            # very first lateral can still be laid from the substation.
            if not self.pen_cables:
                return
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.connect_buildings(
                    only_unconnected=True,
                    clearance_m=clearance,
                )

        def _nearest_distance(i: int) -> float:
            cx, cy = self.footprints[i].centroid_xy_m()
            return min(math.hypot(cx - px, cy - py) for _, (px, py) in nodes)

        laterals: list[str] = []
        for _ in range(max_passes):
            _tap()
            done = {c.house_idx for c in self.connections}
            orphans = [i for i in range(len(self.footprints)) if i not in done]
            if not orphans:
                break
            progressed = False
            for i in sorted(orphans, key=_nearest_distance, reverse=True):
                if i in {c.house_idx for c in self.connections}:
                    continue  # already tapped via a neighbour's lateral
                fp = self.footprints[i]
                cx, cy = fp.centroid_xy_m()
                src, _src_xy = min(
                    nodes,
                    key=lambda nd: math.hypot(cx - nd[1][0], cy - nd[1][1]),
                )
                end_xy = self._free_point_near_house(
                    fp,
                    self._anchor_position(src),
                    clearance=lat_clear,
                    margin_m=margin_m,
                )
                if end_xy is None:
                    continue
                name = f"lateral_house_{i:04d}"
                if any(c.name == name for c in self.pen_cables):
                    continue
                routed = False
                eff_escape = (
                    escape_radius_m
                    if escape_radius_m is not None
                    else min_segment_length_m
                )
                # Grid ladder: configured -> half (tighter final bridges) ->
                # fifth (maze escape on long winding routes, where a fine grid
                # would otherwise exhaust the A* iteration budget).
                for grid, esc in (
                    (min_segment_length_m, escape_radius_m),
                    (max(1.0, min_segment_length_m / 2.0), 2.0 * eff_escape),
                    (max(1.0, min_segment_length_m / 5.0), 2.0 * eff_escape),
                ):
                    try:
                        self.add_pen_cable(
                            start=src,
                            end_xy=end_xy,
                            name=name,
                            min_segment_length_m=grid,
                            clearance_m=clearance,
                            escape_radius_m=esc,
                        )
                        routed = True
                        break
                    except RuntimeError:
                        continue
                if not routed:
                    continue
                laterals.append(name)
                progressed = True
                _tap()  # tap house i (and any neighbours now in reach)
            if not progressed:
                break

        done = {c.house_idx for c in self.connections}
        islands = [i for i in range(len(self.footprints)) if i not in done]
        return LateralCoverageResult(laterals, islands, list(self.connections))

    def _footprint_to_box(
        self,
        fp: BuildingFootprint,
        idx: int,
    ) -> ObstacleBox:
        (cx, cy), (dx, dy) = fp.axis_aligned_bounding_rectangle()
        return ObstacleBox(
            x_min=cx - dx / 2.0,
            y_min=cy - dy / 2.0,
            x_max=cx + dx / 2.0,
            y_max=cy + dy / 2.0,
            name=f"house_{idx}",
        )

    def _obstacle_boxes(self) -> list[ObstacleBox]:
        return [self._footprint_to_box(fp, i) for i, fp in enumerate(self.footprints)]

    def _anchors_connected_to_substation(self) -> list[str]:
        """Source anchors galvanically connected to the substation.

        Walks the node graph induced by PEN cables whose *both* endpoints
        are named anchors (``start_anchor`` / ``end_anchor``) and returns
        the connected component containing the substation, ordered
        substation-first then cabinets in placement order. A cabinet reachable only
        through a free-ended cable (e.g. a trunk tail) is *not* included.
        """
        adjacency: dict[str, set[str]] = {self.substation_name: set()}
        for k in self.cabinet_placements:
            adjacency.setdefault(k.name, set())
        for cable in self.pen_cables:
            a, b = cable.start_anchor, cable.end_anchor
            if a in adjacency and b is not None and b in adjacency:
                adjacency[a].add(b)
                adjacency[b].add(a)
        seen = {self.substation_name}
        stack = [self.substation_name]
        while stack:
            current = stack.pop()
            for neighbour in adjacency.get(current, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return [self.substation_name] + [
            k.name for k in self.cabinet_placements if k.name in seen
        ]

    def _free_point_near_house(
        self,
        fp: BuildingFootprint,
        node_xy: tuple[float, float],
        *,
        clearance: float,
        margin_m: float = 3.0,
    ) -> Optional[tuple[float, float]]:
        r"""Free $(x, y)$ just outside ``fp``, biased towards ``node_xy``.

        Returns the first candidate -- placed at growing distances and a
        fan of bearings around the house→node direction -- that lies
        outside every footprint box inflated by ``clearance``. Returns
        ``None`` if the house is enclosed on all tried bearings.
        """
        (cx, cy), (dx, dy) = fp.axis_aligned_bounding_rectangle()
        half_diag = 0.5 * math.hypot(dx, dy)
        ux, uy = node_xy[0] - cx, node_xy[1] - cy
        norm = math.hypot(ux, uy)
        if norm < 1e-9:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = ux / norm, uy / norm
        boxes = [
            self._footprint_to_box(f, j).inflated(clearance)
            for j, f in enumerate(self.footprints)
        ]
        for extra in (0.0, 3.0, 6.0, 10.0, 16.0):
            dist = half_diag + margin_m + extra
            for ang_deg in (
                0,
                20,
                -20,
                45,
                -45,
                70,
                -70,
                110,
                -110,
                160,
                -160,
                180,
            ):
                theta = math.radians(ang_deg)
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                vx = cos_t * ux - sin_t * uy
                vy = sin_t * ux + cos_t * uy
                px, py = cx + dist * vx, cy + dist * vy
                if not any(b.contains_point(px, py) for b in boxes):
                    return (px, py)
        return None

    def _stub_to_cable(
        self,
        centroid: tuple[float, float],
        cable: PenCable,
        obstacles: list[ObstacleBox],
    ) -> tuple[
        Optional[tuple[float, float]],
        Optional[list[tuple[float, float]]],
    ]:
        """Find the Manhattan-closest tap point on ``cable``.

        Returns ``(tap_point, stub_waypoints)`` or ``(None, None)``
        when no obstacle-free stub exists.
        """
        best: Optional[tuple[tuple[float, float], list[tuple[float, float]], float]] = (
            None
        )
        for seg_start, seg_end in cable.segments():
            tap = _project_onto_axis_segment(centroid, seg_start, seg_end)
            stub = _manhattan_stub(centroid, tap, obstacles)
            if stub is None:
                continue
            length = sum(
                abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
                for p0, p1 in zip(stub[:-1], stub[1:], strict=True)
            )
            if best is None or length < best[2]:
                best = (tap, stub, length)
        if best is None:
            return (None, None)
        return (best[0], best[1])


# ---------------------------------------------------------------------
# Free-floating helpers
# ---------------------------------------------------------------------


def _project_onto_axis_segment(
    point: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> tuple[float, float]:
    """Closest point on an axis-aligned segment to ``point``."""
    sx, sy = seg_start
    ex, ey = seg_end
    if math.isclose(sy, ey, abs_tol=1e-9):
        # Horizontal segment.
        x = max(min(sx, ex), min(point[0], max(sx, ex)))
        return (x, sy)
    if math.isclose(sx, ex, abs_tol=1e-9):
        # Vertical segment.
        y = max(min(sy, ey), min(point[1], max(sy, ey)))
        return (sx, y)
    raise ValueError(
        f"_project_onto_axis_segment: segment {(seg_start, seg_end)} "
        "is not axis-aligned."
    )


def _manhattan_stub(
    centroid: tuple[float, float],
    tap: tuple[float, float],
    obstacles: list[ObstacleBox],
) -> Optional[list[tuple[float, float]]]:
    """Build a Manhattan polyline from ``centroid`` to ``tap``.

    Returns ``None`` when no axis-aligned 1- or 2-segment polyline
    can connect ``centroid`` to ``tap`` without crossing any
    obstacle. Prefers a direct (single-segment) connection when
    one of the coordinates already matches.
    """
    cx, cy = centroid
    tx, ty = tap
    if math.isclose(cx, tx) or math.isclose(cy, ty):
        if not any(segment_intersects_box(centroid, tap, o) for o in obstacles):
            return [centroid, tap]
        # Single straight stub is blocked -- fall through to the
        # corner attempts.
    # Two corner options.
    for corner in ((tx, cy), (cx, ty)):
        seg1 = (centroid, corner)
        seg2 = (corner, tap)
        # Skip degenerate corners (collinear).
        if math.isclose(corner[0], centroid[0]) and math.isclose(
            corner[1], centroid[1]
        ):
            continue
        if math.isclose(corner[0], tap[0]) and math.isclose(corner[1], tap[1]):
            continue
        if any(
            segment_intersects_box(seg1[0], seg1[1], o)
            or segment_intersects_box(seg2[0], seg2[1], o)
            for o in obstacles
        ):
            continue
        return merge_collinear_waypoints([centroid, corner, tap])
    return None
