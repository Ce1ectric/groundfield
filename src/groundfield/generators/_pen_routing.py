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
   drop starts at that node (the substation or the cabinet, or a building
   the cable runs through), at the node's point nearest to the building. A
   building fed from the substation itself (a substation slot of the
   radial trunk) has no feeding cable: its drop starts at the substation.
   A drop never starts on another building's drop.
3. **No section shorter than** :attr:`PenRouter.min_length`: a tap
   closer than that to an existing node of the path (along the path)
   snaps to the node, so every cable section and every drop is at least
   :attr:`PenRouter.min_length` long (:func:`min_section_length`: at
   least :data:`MIN_DROP_LENGTH_M`, and long enough that the section's
   series resistance stays at least twice the threshold below which the
   :class:`~groundfield.conductors.conductor.Conductor` model treats a
   conductor as an ideal short and refuses it as a distributed
   conductor).
4. **A building on its feeding path** — its connection point, or a wire of
   its foundation, closer than :attr:`PenRouter.min_length` to the path in
   plan, away from the path's nodes; a cable that crosses the foundation —
   splits the path there: the cable is joined to the anchor (a building
   without a foundation) or to the wire it crosses or touches, and the
   building gets no drop (the drop shrinks to nothing at the crossing).
   Where the foundation touches the path at a node (a cabinet inside the
   outline) the joint moves along the path to :attr:`PenRouter.min_length`
   from the node. Two nodes closer than :attr:`PenRouter.min_length`
   cannot be joined by a cable: the router raises :class:`ValueError`
   naming both (a site on top of another).

What follows: a drop never runs along the cable it taps — it leaves an
interior tap perpendicularly, or an end node at an angle of at least
90° to the cable, or a snapped node towards a point at least
:attr:`PenRouter.min_length` off the cable (sharing at most that length
with it) — so the DOC-B defect (two service drops from a cabinet 180 m out
running back along the main, coaxial with it) cannot recur; and since user
decision Q41 a drop neither runs along a side of its own foundation nor
crosses it (audit D3A-02). A drop *may* run along another cable: a
cable of another cabinet, a trunk of the radial topology (the drops of
the substation slots), another building's drop. Such conductors lie in
one trench; the inductive coupling evaluates overlapping conductors as
touching wires (finite mutual inductance, passive network).

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
    """Plan distance in m between the pair's two points (to ``node``'s
    connection once snapped)."""
    section: _Section | None
    """Section that holds the point (``None``: the path is empty)."""
    t: float
    """Fraction along ``section``."""
    node: str | None
    """The node the point is (an end of ``section``, or the origin of an
    empty path), or ``None`` for an interior point of ``section``."""
    building: _End | None = None
    """Where the building is met (its foundation point, or its connection
    point); ``None`` once snapped to ``node``."""


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
        self.wires = {name: list(w) for name, w in (wires or {}).items() if w}
        self.depth_m = float(depth_m)
        self.conductor = conductor
        self.min_length = min_section_length(conductor)
        self.junctions: list[PlannedJunction] = []
        # Sections of every cable in order along the cable, cables in
        # creation order; drops are never split.
        self._cables: dict[str, list[_Section]] = {}
        self._drops: list[_Section] = []
        self._splits: dict[str, int] = {}

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

    def connect_drop(
        self,
        building: str,
        cables: Sequence[str],
        origin: str,
        *,
        name: str,
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

        Raises
        ------
        ValueError
            If the building is closer than :attr:`min_length` to the node
            its drop would start at.
        """
        near = self._nearest(building, cables, origin)
        if near.node is not None:
            start, end, distance = self._join(near.node, building)
            if distance < self.min_length:
                raise self._coincide_error(building, near.node, distance)
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
                return
            start, end = self._split_at(near.section, near.t), near.building
        self._drops.append(_Section(name, start, end, "service_drop", name))

    def drop_order(
        self, buildings: Sequence[str], cables: Sequence[str], origin: str
    ) -> list[str]:
        """``buildings`` sorted by their distance to their feeding path.

        Buildings on the path come first (they split it before the taps of
        the others are placed); ties keep the given order.
        """

        def key(item: tuple[int, str]) -> tuple[float, int]:
            k, name = item
            return self._nearest(name, cables, origin).distance, k

        return [name for _, name in sorted(enumerate(buildings), key=key)]

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

    def _nearest(self, building: str, cables: Sequence[str], origin: str) -> _Nearest:
        """Nearest point of the sections of ``cables`` to ``building``.

        The building is its connection point, or the wires of its
        foundation (closest pair, user decision Q41). A point closer than
        :attr:`min_length` (along its section) to an end of the section is
        that end node, unless the building's foundation touches the cable
        there (closer than :attr:`min_length` to the node): then the point
        moves along the section to :attr:`min_length` from the node, so the
        cable can be joined to the foundation without a section shorter
        than that.
        """
        sections = [s for cable in cables for s in self._cables[cable]]
        if not sections:
            return _Nearest(self._join(origin, building)[2], None, 0.0, origin)
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
        snapped = self._join(node, building)[2]
        if wires is None or snapped >= self.min_length or length < 2 * self.min_length:
            return _Nearest(snapped, section, t, node)
        # The foundation touches the cable at the node (a cabinet or a tap
        # inside the outline): meet it min_length along the section (a hair
        # more, so that round-off cannot leave a shorter section).
        step = self.min_length * (1.0 + 1e-9) / length
        t = step if at_start else 1.0 - step
        (ax, ay), (bx, by) = section.start.xy, section.end.xy
        point = (ax + t * (bx - ax), ay + t * (by - ay))
        contact = nearest_point(point, wires)
        return _Nearest(
            contact.distance,
            section,
            t,
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
