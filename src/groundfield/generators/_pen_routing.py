r"""Routing of the PEN cables of generated TN networks (private).

Rules of :class:`~groundfield.generators.tn_network.TnNetworkGenerator`
(since 2026.11; this module states exactly what the router guarantees):

0. **Where a cable meets a site** (user decision Q41 of 2026.11): a site
   with a foundation (the wires of its
   :class:`~groundfield.generators.electrode_specs.FoundationElectrodeSpec`
   electrodes, :func:`~groundfield.generators.materialise.site_foundation_wires`)
   is met at the point of the foundation nearest to the cable — a cable
   leaving it at the point nearest to the cable's other end, a drop at the
   closest pair below — and the conductor is attached to the wire that
   holds the point (:class:`~groundfield.generators.plan.PlannedContact`);
   ties go to the lowest wire index, then to the smallest arc position along
   the wire (:mod:`groundfield.generators._contact`). Any other site (rods,
   rings, strips) is met at the connection point of its anchor electrode.
1. **Every cabinet has its own cable**, as its topology documents: in the
   star topology a straight cable from the substation to the cabinet
   (``pen_main_<anchor>``), in the radial-trunk topology the trunk
   segment from the previous tap source of the feeder
   (``pen_trunk_f<i>_<k>``). Cables are never chained through another
   cabinet; two cables with a common route (collinear cabinets) are two
   separate cables in one trench.
2. **A service drop taps the cable path that feeds its own building** —
   the cable of its cabinet (star) or the trunk from the substation to its
   tap source (radial) — at the point of that path nearest to the
   building: the closest pair in plan between the sections of the path and
   the wires of the building's foundation, or the point nearest to the
   connection point of a building without a foundation. An interior point
   becomes a T-joint: a leakage-free tap junction ``pen_tap_<k>``
   (:class:`~groundfield.generators.plan.PlannedJunction`) that splits
   the cable section; the parts after the split are named
   ``<cable>_s<k>``. When the nearest point is an end of the path the
   drop starts at that node: the substation or the cabinet, at its point
   nearest to the building, or — at a building the cable runs through
   (rule 4) — the joint on the cable (audit R4A-01). A building fed from
   the substation itself (a substation slot of the radial trunk) has no
   feeding cable: its drop starts at the substation. A drop from a node
   that would run along a trunk of the radial topology starts where it
   leaves the trunk, at the trunk's node at the end of the shared stretch
   (audit R4B-01). A drop never starts on another building's drop.
3. **No section shorter than** :attr:`PenRouter.min_length`: a tap
   closer than that to an existing node of the path (along the path)
   snaps to the node, so every cable section and every drop is at least
   :attr:`PenRouter.min_length` long (:func:`min_section_length`: at
   least :data:`MIN_DROP_LENGTH_M`, and long enough that the section's
   series resistance stays at least twice the threshold below which the
   :class:`~groundfield.conductors.conductor.Conductor` model treats a
   conductor as an ideal short and refuses it as a distributed
   conductor).
4. **A building that a cable crosses is joined to it and fed by it** — a
   cable of the building's feeding path passes its connection point, or a
   wire of its foundation, closer than :attr:`PenRouter.min_length` in
   plan, away from the path's nodes (it crosses or touches the
   foundation): the path is split there, the cable is joined to the
   anchor (a building without a foundation) or to the wire it crosses or
   touches, and the building gets no drop. For this rule every trunk of
   the radial topology counts as a cable of the feeding path (a building
   in a substation slot has none of its own; a trunk beyond the
   building's tap source runs on past it), and the star topology feeds a
   building that a main crosses from that main's cabinet (the nearest of
   the cabinets whose mains cross it, ties to the lower cabinet index)
   instead of its nearest cabinet (audit R4A-02). A main that crosses a
   building it does not feed stays unjoined (no cross-tie between two
   cabinets through a house). Where the foundation touches the path at a
   node (a cabinet inside the outline) the joint moves along the path to
   :attr:`PenRouter.min_length` from the node. Two nodes closer than
   :attr:`PenRouter.min_length` cannot be joined by a cable: the router
   raises :class:`ValueError` naming both (a site on top of another).

What follows: a drop never runs along a cable — it leaves an interior
tap perpendicularly, an end node at an angle of at least 90° to the cable
it taps, a trunk at the end of the stretch it would share with it, or a
snapped node towards a point at least :attr:`PenRouter.min_length` off the
cable (sharing at most that length with it: a foundation that touches the
end node of the path from beyond is tapped one minimum section before the
node) — so the DOC-B defect (two service drops from a cabinet 180 m out
running back along the main, coaxial with it) cannot recur; since user
decision Q41 a drop neither runs along a side of its own foundation nor
crosses it (audit D3A-02), and since audits R4A-02 / R4B-01 no drop runs
along a trunk or another cabinet's main (the drops of the substation slots
did, through the houses the trunk crosses). A drop *may* run along another
building's drop, or pass over another building's foundation on its way
(a house behind a house that the cable crosses). Overlapping conductors
lie in one trench; the inductive coupling evaluates them as touching wires
(finite mutual inductance, passive network).

Coordinates are horizontal positions: the points where the materialised
conductors end (they run at the PEN depth, ``axis_z_m``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from groundfield.conductors.conductor import _IDEAL_RESISTANCE_THRESHOLD, Conductor
from groundfield.generators._contact import (
    FoundationWire,
    PreparedWires,
    closest_pair,
    join,
    nearest_point,
)
from groundfield.generators.plan import (
    ConductorParameters,
    PlannedBranch,
    PlannedContact,
    PlannedJunction,
)

__all__ = ["MIN_DROP_LENGTH_M", "PenRouter", "min_section_length"]

MIN_DROP_LENGTH_M = 0.01
"""Shortest cable section or service drop in m (for thin conductors).

A building closer than this to its feeding cable sits on it; a tap closer
than this to a node of the cable is that node."""


def min_section_length(conductor: ConductorParameters) -> float:
    """Shortest cable section or drop in m that the router creates.

    Parameters
    ----------
    conductor
        Parameters of the cable (wire radius in m, cross section in m²).

    Returns
    -------
    float
        ``max(MIN_DROP_LENGTH_M, 2 * L_ideal)`` in m, where ``L_ideal`` is
        the length at which the series resistance
        :math:`R = \\rho_\\text{mat} L / A` of the conductor (copper, the
        :class:`~groundfield.conductors.conductor.Conductor` default)
        reaches the threshold (1e-6 Ω) below which the conductor model
        treats a conductor as an ideal short — and refuses it as a
        distributed conductor. 1 cm for wire radii up to 7.3 mm
        (``L_ideal`` = 4.7 mm at the default radius of 5 mm), 2.9 cm for
        240 mm². An ideal conductor (``cross_section=None``) gives
        :data:`MIN_DROP_LENGTH_M`.
    """
    if conductor.cross_section is None:
        return MIN_DROP_LENGTH_M
    if conductor.cross_section == "from_radius":
        area = math.pi * conductor.wire_radius_m**2
    else:
        area = float(conductor.cross_section)
    resistivity = float(Conductor.model_fields["resistivity"].default)
    ideal_length = _IDEAL_RESISTANCE_THRESHOLD * area / resistivity
    return max(MIN_DROP_LENGTH_M, 2.0 * ideal_length)


@dataclass(frozen=True)
class _End:
    """One end of a cable section or drop: a node and the point it is at."""

    node: str
    """Plan node (site or junction)."""
    xy: tuple[float, float]
    """Plan position in m of the end point."""
    contact: str | None = None
    """Wire of the site's foundation that holds the point (user decision
    Q41), or ``None`` for the connection point of the node."""


@dataclass
class _Section:
    """One straight PEN cable section between two nodes."""

    name: str
    start: _End
    end: _End
    role: str
    cable: str
    """Name of the cable the section belongs to (its first section's name)."""


@dataclass
class _Nearest:
    """Nearest point of a feeding path to a building."""

    distance: float
    """Plan distance in m between the pair's two points (between the drop's
    two ends once snapped)."""
    section: _Section | None
    """Section that holds the point (``None``: the path is empty)."""
    t: float
    """Fraction along ``section``."""
    node: str | None
    """The node the point is (an end of ``section``, or the origin of an
    empty path), or ``None`` for an interior point of ``section``."""
    building: _End | None = None
    """Where the building is met (its foundation point, or its connection
    point)."""
    start: _End | None = None
    """Where the drop starts once snapped to ``node`` (``None`` for an
    interior point: the drop starts at a new tap junction)."""


class PenRouter:
    """PEN cables of one network: cabinet cables and service drops.

    See the module docstring for the rules.

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
    wires
        Foundation wires (plan) of the nodes that have a foundation
        (:func:`~groundfield.generators.materialise.site_foundation_wires`);
        a cable or drop ends at the point of these wires nearest to it (user
        decision Q41). Nodes without an entry end at ``node_xy``.

    Attributes
    ----------
    min_length
        Shortest cable section or drop in m (:func:`min_section_length`).
    junctions
        The tap junctions, in creation order.
    """

    def __init__(
        self,
        root: str,
        node_xy: dict[str, tuple[float, float]],
        *,
        depth_m: float,
        conductor: ConductorParameters,
        wires: Mapping[str, Sequence[FoundationWire]] | None = None,
    ) -> None:
        self.root = root
        self.xy = {name: (float(x), float(y)) for name, (x, y) in node_xy.items()}
        # Prepared once per node: the closest-pair search runs twice per
        # building (drop order and drop) and on every cable it may cross
        # (audit R4A-06).
        self.wires = {
            name: PreparedWires.of(list(w)) for name, w in (wires or {}).items() if w
        }
        self.depth_m = float(depth_m)
        self.conductor = conductor
        self.min_length = min_section_length(conductor)
        self.junctions: list[PlannedJunction] = []
        # Sections of every cable in order along the cable, cables in
        # creation order; drops are never split.
        self._cables: dict[str, list[_Section]] = {}
        self._drops: list[_Section] = []
        self._splits: dict[str, int] = {}
        # Buildings a cable runs through, joined to it by rule 4.
        self._joined: set[str] = set()
        self._boxes: dict[str, tuple[float, float, float, float]] = {}

    # ------------------------------------------------------------------
    # Building the network
    # ------------------------------------------------------------------

    def add_cable(self, start: str, end: str, *, name: str) -> None:
        """Add a cable of the mains (a cabinet's cable or a trunk segment).

        The cable runs straight between the two nodes; at a node with a
        foundation it ends at the point of the foundation nearest to the
        other end (user decision Q41).

        Raises
        ------
        ValueError
            If the two nodes are closer than :attr:`min_length`.
        """
        a, b, distance = self._join(start, end)
        if distance < self.min_length:
            raise self._coincide_error(end, start, distance)
        self._cables[name] = [_Section(name, a, b, "pen", name)]
        # Plan bounding box of the cable: its sections stay on the straight
        # line (a joint at a building's connection point lies within
        # min_length of it, hence the margin).
        m = self.min_length
        self._boxes[name] = (
            min(a.xy[0], b.xy[0]) - m,
            min(a.xy[1], b.xy[1]) - m,
            max(a.xy[0], b.xy[0]) + m,
            max(a.xy[1], b.xy[1]) + m,
        )

    def connect_drop(
        self,
        building: str,
        cables: Sequence[str],
        origin: str,
        *,
        name: str,
        crossing: Sequence[str] = (),
        avoid: Sequence[str] | None = None,
    ) -> None:
        """Connect ``building`` to the nearest point of its feeding path.

        Parameters
        ----------
        building
            Node name of the building.
        cables
            Names of the cables that feed the building (added with
            :meth:`add_cable`); may be empty.
        origin
            The feeding node (cabinet or substation): the start of the drop
            when ``cables`` is empty.
        name
            Name of the drop.
        crossing
            Cables that count as feeding cables for rule 4 only: one of
            them that crosses the building (or passes closer than
            :attr:`min_length`) is joined to it there and the building gets
            no drop, unless ``cables`` already passes through it. The drop
            of any other building still taps ``cables`` (or starts at
            ``origin``). The radial topology passes all its trunks: for a
            building in a substation slot they are its feeding path, and a
            trunk beyond the building's tap source that crosses it is joined
            too (audit R4A-02).
        avoid
            Cables a drop from a node must not run along (default
            ``crossing``): such a drop starts where it leaves the cable
            (audit R4B-01). The star topology passes all mains.

        Raises
        ------
        ValueError
            If the building is closer than :attr:`min_length` to the node
            its drop would start at.
        """
        near = self._nearest(building, cables, origin, crossing)
        if near.node is not None:
            assert near.start is not None and near.building is not None
            start, end = near.start, near.building
            avoid = crossing if avoid is None else avoid
            if avoid:
                start, end = self._leave_cables(start, end, building, avoid)
            distance = self._distance(start.xy, end.xy)
            if distance < self.min_length:
                raise self._coincide_error(building, start.node, distance)
        else:
            assert near.section is not None and near.building is not None
            if near.distance < self.min_length:
                # The cable passes through the building's connection: its
                # anchor's connection point, or a wire of its foundation
                # that it crosses or touches. The cable is joined to it
                # there (at the cable's point, which lies within min_length
                # of the wire; for a crossing it is the crossing).
                meet = near.building
                if meet.contact is not None:
                    (ax, ay), (bx, by) = near.section.start.xy, near.section.end.xy
                    meet = _End(
                        building,
                        (ax + near.t * (bx - ax), ay + near.t * (by - ay)),
                        meet.contact,
                    )
                self._split(near.section, meet)
                self._joined.add(building)
                return
            start, end = self._split_at(near.section, near.t), near.building
        self._drops.append(_Section(name, start, end, "service_drop", name))

    def drop_order(
        self,
        buildings: Sequence[str],
        cables: Sequence[str],
        origin: str,
        *,
        crossing: Sequence[str] = (),
    ) -> list[str]:
        """``buildings`` sorted by their distance to their feeding path.

        Buildings on the path come first (they split it before the taps of
        the others are placed); ties keep the given order. ``crossing``: see
        :meth:`connect_drop`.
        """

        def key(item: tuple[int, str]) -> tuple[float, int]:
            k, name = item
            return self._nearest(name, cables, origin, crossing).distance, k

        return [name for _, name in sorted(enumerate(buildings), key=key)]

    def crossing(self, building: str, cables: Sequence[str]) -> list[str]:
        """The cables of ``cables`` that pass through ``building`` (rule 4).

        A cable passes through a building when the closest pair of its
        sections and the building's foundation wires (or its connection
        point) is closer than :attr:`min_length`, away from the cable's
        nodes — where it would be joined to the building without a drop.
        The star topology feeds such a building from the cabinet of one of
        these cables (audit R4A-02).

        Returns
        -------
        list of str
            The crossing cables, in the order of ``cables``.
        """
        return [
            cable
            for cable in self._near(building, cables)
            if self._is_joint(self._nearest_on_path(building, [cable], self.root))
        ]

    def _near(self, building: str, cables: Sequence[str]) -> list[str]:
        """The cables of ``cables`` whose bounding box comes within
        :attr:`min_length` of the building's: the others cannot pass through
        it (a cheap filter, audit R4A-06)."""
        wires = self.wires.get(building)
        if wires is None:
            bx0, by0 = bx1, by1 = self.xy[building]
        else:
            bx0, by0, bx1, by1 = wires[6]
        pad = self.min_length * (1.0 + 1e-6) + 1e-9
        out = []
        for cable in cables:
            x0, y0, x1, y1 = self._boxes[cable]
            if not (
                x1 < bx0 - pad or x0 > bx1 + pad or y1 < by0 - pad or y0 > by1 + pad
            ):
                out.append(cable)
        return out

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def branches(self) -> list[PlannedBranch]:
        """The cable sections (cable by cable, in order along each cable)
        and then the service drops, as plan branches."""
        sections = [s for cable in self._cables.values() for s in cable]
        return [
            PlannedBranch(
                name=s.name,
                role=s.role,
                start=s.start.node,
                end=s.end.node,
                conductor=self.conductor,
                axis_z_m=self.depth_m,
                start_contact=_contact(s.start),
                end_contact=_contact(s.end),
            )
            for s in [*sections, *self._drops]
        ]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _join(self, a: str, b: str) -> tuple[_End, _End, float]:
        """Ends of a straight conductor between the nodes ``a`` and ``b``.

        A node without a foundation is met at its connection point; a node
        with one at the point of its wires nearest to the other end (user
        decision Q41): two foundations at their closest pair.
        """
        (xa, wa), (xb, wb) = join(
            self.xy[a], self.wires.get(a), self.xy[b], self.wires.get(b)
        )
        ea, eb = _End(a, xa, wa), _End(b, xb, wb)
        return ea, eb, self._distance(ea.xy, eb.xy)

    def _meet(self, building: str, xy: tuple[float, float]) -> _End:
        """Where a conductor from the plan point ``xy`` meets ``building``:
        the point of its foundation nearest to ``xy``, or its connection
        point (a building without a foundation)."""
        wires = self.wires.get(building)
        if wires is None:
            return _End(building, self.xy[building])
        contact = nearest_point(xy, wires)
        return _End(building, contact.xy, contact.electrode)

    def _snap(
        self, section: _Section, at_start: bool, building: str
    ) -> tuple[_End, _End, float]:
        """Ends of a drop from the end node of ``section`` to ``building``.

        A building the cable runs through (joined by rule 4) is left at the
        joint, the section's end point on the cable (audit R4A-01): the
        drop taps the cable, it does not start on that building's
        foundation. The substation, a cabinet and a tap junction are left
        at their point nearest to the building (:meth:`_join`).
        """
        end = section.start if at_start else section.end
        if end.node in self._joined:
            meet = self._meet(building, end.xy)
            return end, meet, self._distance(end.xy, meet.xy)
        return self._join(end.node, building)

    def _leave_cables(
        self, start: _End, end: _End, building: str, cables: Sequence[str]
    ) -> tuple[_End, _End]:
        """Ends of a drop from a node that does not run along ``cables``.

        A drop from a node that would run along a section of ``cables`` —
        both ends of the section within :attr:`min_length` of the drop's
        line, sharing more than :attr:`min_length` with it (audit R5A-01: an
        exact collinearity test let a drop 2 µm beside a trunk run along it
        for 60 m) — starts where it leaves the cable instead: at the
        section's end node farther along the drop (repeated until it runs
        along none). In the radial topology
        the drop of a building beyond a later cabinet on the feeder's axis,
        or of a substation slot beyond the trunk's end, leaves the trunk at
        its last node instead of running along it from the tap source
        (audit R4B-01). Cables that start or end at the drop's node run
        away from it at an angle and share no length with it; the section
        a drop taps is never among the shared ones (rule 2).
        """
        sections = [sec for cable in cables for sec in self._cables[cable]]
        for _ in range(len(sections) + 1):
            (sx, sy), (ex, ey) = start.xy, end.xy
            length = math.hypot(ex - sx, ey - sy)
            if length <= self.min_length:
                break
            ux, uy = (ex - sx) / length, (ey - sy) / length
            best: tuple[float, _End] | None = None
            for sec in sections:
                along = []
                for point in (sec.start.xy, sec.end.xy):
                    px, py = point[0] - sx, point[1] - sy
                    if abs(-uy * px + ux * py) > self.min_length:
                        break
                    along.append(ux * px + uy * py)
                else:
                    (a0, a1), ends = along, (sec.start, sec.end)
                    lo, hi = min(a0, a1), max(a0, a1)
                    far = ends[0] if a0 > a1 else ends[1]
                    shared = min(hi, length) - max(lo, 0.0)
                    if (
                        shared > self.min_length
                        and hi < length
                        and (best is None or hi > best[0])
                    ):
                        best = (hi, far)
            if best is None:
                break
            node = best[1]
            if node.node in self._joined:
                start = node
                end = self._meet(building, node.xy)
            else:
                start, end, _ = self._join(node.node, building)
        return start, end

    def _nearest(
        self,
        building: str,
        cables: Sequence[str],
        origin: str,
        crossing: Sequence[str] = (),
    ) -> _Nearest:
        """Nearest point of the sections of ``cables`` to ``building``.

        The building is its connection point, or the wires of its
        foundation (closest pair, user decision Q41). A point closer than
        :attr:`min_length` (along its section) to an end of the section is
        that end node, unless the building's foundation touches the cable
        there (closer than :attr:`min_length` to the node): then the point
        moves along the section to :attr:`min_length` from the node, so the
        cable can be joined to the foundation without a section shorter
        than that. An empty path is its ``origin``. A cable of
        ``crossing`` that passes through the building (rule 4) wins over a
        drop to the path.
        """
        near = self._nearest_on_path(building, cables, origin)
        if crossing and not self._is_joint(near):
            # Rule 4 for cables that do not feed the building (the trunks of
            # the radial topology): one that crosses it is joined there. A
            # set: the list membership test cost O(trunks x path) per
            # building (audit R5A-04).
            own = set(cables)
            others = self._near(
                building, [cable for cable in crossing if cable not in own]
            )
            if others:
                cross = self._nearest_on_path(building, others, origin)
                if self._is_joint(cross):
                    return cross
        return near

    def _is_joint(self, near: _Nearest) -> bool:
        """Whether ``near`` is a point where the cable passes through the
        building (rule 4: the cable is joined to it, no drop)."""
        return near.node is None and near.distance < self.min_length

    def _nearest_on_path(
        self, building: str, cables: Sequence[str], origin: str
    ) -> _Nearest:
        """:meth:`_nearest` without the ``crossing`` cables."""
        sections = [s for cable in cables for s in self._cables[cable]]
        if not sections:
            start, meet, distance = self._join(origin, building)
            return _Nearest(distance, None, 0.0, origin, meet, start)
        wires = self.wires.get(building)
        if wires is None:
            target = self.xy[building]
            tx, ty = target
            best: tuple[float, _Section, float] | None = None
            for section in sections:
                (ax, ay), (bx, by) = section.start.xy, section.end.xy
                dx, dy = bx - ax, by - ay
                length2 = dx * dx + dy * dy
                t = (
                    0.0
                    if length2 == 0.0
                    else ((tx - ax) * dx + (ty - ay) * dy) / length2
                )
                t = min(max(t, 0.0), 1.0)
                d = math.hypot(tx - (ax + t * dx), ty - (ay + t * dy))
                if best is None or d < best[0] - 1e-12:
                    best = (d, section, t)
            assert best is not None
            distance, section, t = best
            meet = _End(building, target)
        else:
            i, t, contact = closest_pair(
                [(sec.start.xy, sec.end.xy) for sec in sections], wires
            )
            section, distance = sections[i], contact.distance
            meet = _End(building, contact.xy, contact.electrode)
        length = self._distance(section.start.xy, section.end.xy)
        at_start = t * length <= (1.0 - t) * length
        node = section.start.node if at_start else section.end.node
        along = t * length if at_start else (1.0 - t) * length
        if along >= self.min_length:
            return _Nearest(distance, section, t, None, meet)
        start, end, snapped = self._snap(section, at_start, building)
        if wires is None or snapped >= self.min_length:
            return _Nearest(snapped, section, t, node, end, start)
        if length >= 2 * self.min_length:
            # The foundation touches the cable at the node (a cabinet or a
            # tap inside the outline): meet it min_length along the section
            # (a hair more, so that round-off cannot leave a shorter
            # section).
            step = self.min_length * (1.0 + 1e-9) / length
            t = step if at_start else 1.0 - step
            (ax, ay), (bx, by) = section.start.xy, section.end.xy
            point = (ax + t * (bx - ax), ay + t * (by - ay))
            contact = nearest_point(point, wires)
            near = _Nearest(
                contact.distance,
                section,
                t,
                None,
                _End(building, contact.xy, contact.electrode),
            )
            if self._is_joint(near):
                return near
        else:
            near = _Nearest(snapped, section, t, node, end, start)
        # The cable only touches the foundation at the node and leaves it
        # there (a cabinet on a wire): where it crosses the foundation
        # elsewhere it is joined there (audit R5A-03: the node won the tie
        # of the two zero distances, the house got a 1 cm drop and the
        # crossing stayed unjoined).
        inner = self._interior_joint(building, sections, wires)
        return near if inner is None else inner

    def _interior_joint(
        self,
        building: str,
        sections: Sequence[_Section],
        wires: PreparedWires,
    ) -> _Nearest | None:
        """Where ``sections`` pass through the building's foundation away
        from their nodes (rule 4), or ``None``.

        The closest pair of the foundation wires and the sections without
        :attr:`min_length` (a hair more) at either end; a pair closer than
        :attr:`min_length` is a joint, at least :attr:`min_length` from
        every node of the path (rule 3).
        """
        trim = self.min_length * (1.0 + 1e-9)
        inner: list[tuple[_Section, float, float]] = []
        pieces = []
        for section in sections:
            length = self._distance(section.start.xy, section.end.xy)
            if length <= 2.0 * trim:
                continue
            lo, hi = trim / length, 1.0 - trim / length
            (ax, ay), (bx, by) = section.start.xy, section.end.xy
            inner.append((section, lo, hi))
            pieces.append(
                (
                    (ax + lo * (bx - ax), ay + lo * (by - ay)),
                    (ax + hi * (bx - ax), ay + hi * (by - ay)),
                )
            )
        if not pieces:
            return None
        i, u, contact = closest_pair(pieces, wires)
        if not contact.distance < self.min_length:
            return None
        section, lo, hi = inner[i]
        return _Nearest(
            contact.distance,
            section,
            lo + u * (hi - lo),
            None,
            _End(building, contact.xy, contact.electrode),
        )

    def _coincide_error(self, node: str, other: str, distance: float) -> ValueError:
        """Two nodes closer than :attr:`min_length` cannot be joined."""
        return ValueError(
            f"PEN routing: the connection of {node!r} coincides with "
            f"that of {other!r} ({distance:.3g} m apart, less than the "
            f"shortest cable section of {self.min_length:g} m); a cable "
            "between them would have no length. Move one of the two sites."
        )

    def _split_at(self, section: _Section, t: float) -> _End:
        """Insert a tap junction at the fraction ``t`` of ``section``."""
        (ax, ay), (bx, by) = section.start.xy, section.end.xy
        name = f"pen_tap_{len(self.junctions)}"
        xy = (ax + t * (bx - ax), ay + t * (by - ay))
        self.xy[name] = xy
        self.junctions.append(
            PlannedJunction(name=name, position_xy=xy, depth_m=self.depth_m)
        )
        end = _End(name, xy)
        self._split(section, end)
        return end

    def _split(self, section: _Section, node: _End) -> None:
        """Split ``section`` at ``node``: ``start -> node`` and ``node -> end``.

        The first part keeps the section's name; the second is called
        ``<cable>_s<k>`` (``k`` = 1, 2, ... per cable, in split order).
        """
        k = self._splits.get(section.cable, 0) + 1
        self._splits[section.cable] = k
        tail = _Section(
            f"{section.cable}_s{k}",
            node,
            section.end,
            section.role,
            section.cable,
        )
        section.end = node
        sections = self._cables[section.cable]
        index = next(i for i, s in enumerate(sections) if s is section)
        sections.insert(index + 1, tail)


def _contact(end: _End) -> PlannedContact | None:
    """The plan contact of a section end (``None``: a connection point)."""
    if end.contact is None:
        return None
    return PlannedContact(electrode=end.contact, position_xy=end.xy)
