r"""Routing of the PEN cables of generated TN networks (private).

The PEN network of :class:`~groundfield.generators.tn_network.TnNetworkGenerator`
is built as a *nearest-point tree*: every cable — the cable of a cable
cabinet and the service drop of a building — starts at the point of the
network already built that is nearest to its far end. Where that point lies
inside a cable, the cable is split there by a leakage-free tap junction
(:class:`~groundfield.generators.plan.PlannedJunction`).

Why: a cable that started at its feeding node regardless of the geometry
could run back along the cable that feeds that node. In the default star
network the single cabinet sat 180 m beyond two houses next to the
substation; both service drops ran from the cabinet back along the PEN main.
Coaxial overlapping conductors have no finite Neumann mutual inductance and
made an inductive PEN network non-passive (:math:`\operatorname{Im} Z_\text{in}
= -0.633\,\Omega` at 1 kHz, audit item of stream DOC-B). The shortest
connection from a point to a set of straight cables never runs along one of
them, so no two cables of a nearest-point tree overlap: a new cable either
starts at a node or leaves a cable at its nearest point (perpendicular to
it).

Which cables a node may connect to keeps the topology of the generator:

* a cabinet's cable (star topology) starts at the nearest point of the
  cables built before it, cabinets in the order of their distance from the
  substation — collinear cabinets are chained instead of fed by coincident
  cables;
* a building's service drop starts at the nearest point of the cable path
  from the substation to its feeding node (its nearest cabinet, or the tap
  of the radial trunk that the slot budget assigns), or of a drop already
  routed from that path for the same feeding node, or at the feeding node
  itself;
* a building whose connection point lies on a cable (closer than
  :data:`MIN_DROP_LENGTH_M`) splits that cable at its anchor and gets no
  drop.

Coordinates are the horizontal positions of the anchors' connection
points, the end points of the materialised conductors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from groundfield.generators.plan import (
    ConductorParameters,
    PlannedBranch,
    PlannedJunction,
)

__all__ = ["MIN_DROP_LENGTH_M", "PenRouter"]

MIN_DROP_LENGTH_M = 0.01
"""Shortest service drop in m; a building closer to a cable sits on it."""

_ENDPOINT_TOL = 1e-9
"""Relative position along a cable below which a point is its end node."""


@dataclass
class _Cable:
    """One straight PEN cable section between two nodes."""

    name: str
    start: str
    end: str
    role: str
    group: str | None
    """Feeding node of a service drop (``None`` for the mains)."""
    root: str = ""
    """Name of the cable this section was split from (its own name if none)."""

    def __post_init__(self) -> None:
        if not self.root:
            self.root = self.name


class PenRouter:
    """Nearest-point tree of the PEN cables of one network.

    Parameters
    ----------
    root
        Name of the substation node.
    node_xy
        Horizontal position in m of every node (substation, cabinets,
        buildings): the connection point of its anchor electrode.
    depth_m
        Depth in m of every cable and tap junction.
    conductor
        Conductor parameters of every cable.
    """

    def __init__(
        self,
        root: str,
        node_xy: dict[str, tuple[float, float]],
        *,
        depth_m: float,
        conductor: ConductorParameters,
    ) -> None:
        self.root = root
        self.xy = {name: (float(x), float(y)) for name, (x, y) in node_xy.items()}
        self.depth_m = float(depth_m)
        self.conductor = conductor
        self.cables: list[_Cable] = []
        self.junctions: list[PlannedJunction] = []
        # Upstream cable of every node of the mains (towards the root).
        self._parent: dict[str, str] = {}
        self._sections: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Building the tree
    # ------------------------------------------------------------------

    def add_main(self, start: str, end: str, *, name: str) -> None:
        """Add a cable of the mains between two given nodes (no routing)."""
        self.cables.append(_Cable(name, start, end, "pen", None))
        self._parent.setdefault(end, name)

    def connect_main(self, node: str, *, name: str) -> None:
        """Connect ``node`` by a cable from the nearest point of the mains."""
        mains = [c for c in self.cables if c.group is None]
        start = self._attach(self.xy[node], mains, extra_nodes=[self.root])
        self.cables.append(_Cable(name, start, node, "pen", None))
        self._parent[node] = name

    def connect_drop(self, building: str, feeder: str, *, name: str) -> None:
        """Connect ``building`` from the nearest point of its feeding path.

        Candidates: the mains on the path from the root to ``feeder``,
        the drops already routed for ``feeder`` and ``feeder`` itself.
        """
        path = set(self._path(feeder))
        candidates = [
            c
            for c in self.cables
            if (c.group is None and c.name in path) or c.group == feeder
        ]
        target = self.xy[building]
        cable, point, distance = self._nearest(target, candidates, [feeder])
        if cable is not None and distance < MIN_DROP_LENGTH_M:
            # The building's connection point lies on the cable: the
            # cable runs through its anchor.
            self._split(cable, building)
            return
        start = point if cable is None else self._split_at(cable, point)
        self.cables.append(_Cable(name, start, building, "service_drop", feeder))

    def drop_order(self, buildings: list[str], feeder: str) -> list[str]:
        """``buildings`` sorted by their distance to the path of ``feeder``.

        Closer buildings are routed first, so a farther one can hang off
        their drops; ties keep the given order.
        """
        path = set(self._path(feeder))
        mains = [c for c in self.cables if c.group is None and c.name in path]

        def key(item: tuple[int, str]) -> tuple[float, int]:
            k, name = item
            return self._nearest(self.xy[name], mains, [feeder])[2], k

        return [name for _, name in sorted(enumerate(buildings), key=key)]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def branches(self) -> list[PlannedBranch]:
        """The cable sections as plan branches, in creation order."""
        return [
            PlannedBranch(
                name=c.name,
                role=c.role,
                start=c.start,
                end=c.end,
                conductor=self.conductor,
                axis_z_m=self.depth_m,
            )
            for c in self.cables
        ]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _path(self, node: str) -> list[str]:
        """Names of the mains from ``node`` up to the root."""
        names: list[str] = []
        by_name = {c.name: c for c in self.cables}
        seen: set[str] = set()
        while node in self._parent and node not in seen:
            seen.add(node)
            cable = by_name[self._parent[node]]
            names.append(cable.name)
            node = cable.start
        return names

    def _nearest(
        self,
        target: tuple[float, float],
        cables: list[_Cable],
        extra_nodes: list[str],
    ) -> tuple[_Cable | None, object, float]:
        """Nearest candidate point to ``target``.

        Returns ``(cable, t, distance)`` for an interior point at the
        fraction ``t`` of ``cable``, or ``(None, node_name, distance)``
        for a node (a cable end or one of ``extra_nodes``).
        """
        best: tuple[float, int, _Cable | None, object] | None = None
        tx, ty = target
        for k, node in enumerate(extra_nodes):
            x, y = self.xy[node]
            d = math.hypot(tx - x, ty - y)
            if best is None or d < best[0]:
                best = (d, -1 - k, None, node)
        for k, cable in enumerate(cables):
            (ax, ay), (bx, by) = self.xy[cable.start], self.xy[cable.end]
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 == 0.0 else ((tx - ax) * dx + (ty - ay) * dy) / length2
            t = min(max(t, 0.0), 1.0)
            px, py = ax + t * dx, ay + t * dy
            d = math.hypot(tx - px, ty - py)
            if t <= _ENDPOINT_TOL:
                candidate: tuple[_Cable | None, object] = (None, cable.start)
            elif t >= 1.0 - _ENDPOINT_TOL:
                candidate = (None, cable.end)
            else:
                candidate = (cable, t)
            if best is None or d < best[0] - 1e-12:
                best = (d, k, *candidate)
        assert best is not None  # extra_nodes is never empty
        return best[2], best[3], best[0]

    def _attach(
        self,
        target: tuple[float, float],
        cables: list[_Cable],
        *,
        extra_nodes: list[str],
    ) -> str:
        """Node at the nearest point of ``cables`` (a tap junction inside one)."""
        cable, point, _ = self._nearest(target, cables, extra_nodes)
        if cable is None:
            return str(point)
        return self._split_at(cable, point)

    def _split_at(self, cable: _Cable, t: object) -> str:
        """Insert a tap junction at the fraction ``t`` of ``cable``."""
        (ax, ay), (bx, by) = self.xy[cable.start], self.xy[cable.end]
        t = float(t)  # type: ignore[arg-type]
        name = f"pen_tap_{len(self.junctions)}"
        xy = (ax + t * (bx - ax), ay + t * (by - ay))
        self.xy[name] = xy
        self.junctions.append(
            PlannedJunction(name=name, position_xy=xy, depth_m=self.depth_m)
        )
        self._split(cable, name)
        return name

    def _split(self, cable: _Cable, node: str) -> None:
        """Split ``cable`` at ``node``: ``start -> node`` and ``node -> end``.

        The first part keeps the cable's name; the others are called
        ``<name>_s<k>`` (``k`` = 1, 2, ... per original cable).
        """
        k = self._sections.get(cable.root, 0) + 1
        self._sections[cable.root] = k
        tail = _Cable(
            f"{cable.root}_s{k}", node, cable.end, cable.role, cable.group, cable.root
        )
        old_end = cable.end
        cable.end = node
        self.cables.insert(self.cables.index(cable) + 1, tail)
        if cable.group is None:
            self._parent[node] = cable.name
            if self._parent.get(old_end) == cable.name:
                self._parent[old_end] = tail.name
