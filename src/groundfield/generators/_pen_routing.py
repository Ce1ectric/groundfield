r"""Routing of the PEN cables of generated TN networks (private).

Rules of :class:`~groundfield.generators.tn_network.TnNetworkGenerator`
(since 2026.11; this module states exactly what the router guarantees):

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
   building's connection point. An interior point becomes a T-joint: a
   leakage-free tap junction ``pen_tap_<k>``
   (:class:`~groundfield.generators.plan.PlannedJunction`) that splits
   the cable section; the parts after the split are named
   ``<cable>_s<k>``. When the nearest point is an end of the path the
   drop starts at that node (the substation or the cabinet). A building
   fed from the substation itself (a substation slot of the radial
   trunk) has no feeding cable: its drop starts at the substation. A drop
   never starts on another building's drop.
3. **No section shorter than** :attr:`PenRouter.min_length`: a tap
   closer than that to an existing node of the path (along the path)
   snaps to the node, so every cable section and every drop is at least
   :attr:`PenRouter.min_length` long (:func:`min_section_length`: at
   least :data:`MIN_DROP_LENGTH_M`, and long enough that the section's
   series resistance stays at least twice the threshold below which the
   :class:`~groundfield.conductors.conductor.Conductor` model treats a
   conductor as an ideal short and refuses it as a distributed
   conductor).
4. A building whose connection point lies on its feeding path (closer
   than :attr:`PenRouter.min_length`, away from the path's nodes) splits
   the path at its anchor: the cable runs through the anchor and the
   building gets no drop. Two nodes closer than
   :attr:`PenRouter.min_length` cannot be joined by a cable: the router
   raises :class:`ValueError` naming both (a site on top of another).

What follows: a drop never runs along the cable it taps — it leaves an
interior tap perpendicularly, or an end node at an angle of at least
90° to the cable, or a snapped node towards a point at least
:attr:`PenRouter.min_length` off the cable — so the DOC-B defect (two
service drops from a cabinet 180 m out running back along the main,
coaxial with it) cannot recur. A drop *may* run along another cable: a
cable of another cabinet, a trunk of the radial topology (the drops of
the substation slots), another building's drop. Such conductors lie in
one trench; the inductive coupling evaluates overlapping conductors as
touching wires (finite mutual inductance, passive network).

Coordinates are the horizontal positions of the anchors' connection
points, the end points of the materialised conductors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from groundfield.conductors.conductor import _IDEAL_RESISTANCE_THRESHOLD, Conductor
from groundfield.generators.plan import (
    ConductorParameters,
    PlannedBranch,
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


@dataclass
class _Section:
    """One straight PEN cable section between two nodes."""

    name: str
    start: str
    end: str
    role: str
    cable: str
    """Name of the cable the section belongs to (its first section's name)."""


@dataclass
class _Nearest:
    """Nearest point of a feeding path to a building's connection point."""

    distance: float
    """Distance in m to the point (to ``node`` once snapped)."""
    section: _Section | None
    """Section that holds the point (``None``: the path is empty)."""
    t: float
    """Fraction along ``section``."""
    node: str | None
    """The node the point is (an end of ``section``, or the origin of an
    empty path), or ``None`` for an interior point of ``section``."""


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
    ) -> None:
        self.root = root
        self.xy = {name: (float(x), float(y)) for name, (x, y) in node_xy.items()}
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

        Raises
        ------
        ValueError
            If the two nodes are closer than :attr:`min_length`.
        """
        distance = self._distance(self.xy[start], self.xy[end])
        if distance < self.min_length:
            raise self._coincide_error(end, start, distance)
        self._cables[name] = [_Section(name, start, end, "pen", name)]

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
            If the building's connection point is closer than
            :attr:`min_length` to the node its drop would start at.
        """
        target = self.xy[building]
        near = self._nearest(target, cables, origin)
        if near.node is not None:
            if near.distance < self.min_length:
                raise self._coincide_error(building, near.node, near.distance)
            start = near.node
        else:
            assert near.section is not None
            if near.distance < self.min_length:
                # The building's connection point lies on the cable: the
                # cable runs through its anchor.
                self._split(near.section, building)
                return
            start = self._split_at(near.section, near.t)
        self._drops.append(_Section(name, start, building, "service_drop", name))

    def drop_order(
        self, buildings: Sequence[str], cables: Sequence[str], origin: str
    ) -> list[str]:
        """``buildings`` sorted by their distance to their feeding path.

        Buildings on the path come first (they split it before the taps of
        the others are placed); ties keep the given order.
        """

        def key(item: tuple[int, str]) -> tuple[float, int]:
            k, name = item
            return self._nearest(self.xy[name], cables, origin).distance, k

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
                start=s.start,
                end=s.end,
                conductor=self.conductor,
                axis_z_m=self.depth_m,
            )
            for s in [*sections, *self._drops]
        ]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _nearest(
        self, target: tuple[float, float], cables: Sequence[str], origin: str
    ) -> _Nearest:
        """Nearest point of the sections of ``cables`` to ``target``.

        A point closer than :attr:`min_length` (along its section) to an
        end of the section is that end node.
        """
        sections = [s for cable in cables for s in self._cables[cable]]
        if not sections:
            return _Nearest(self._distance(target, self.xy[origin]), None, 0.0, origin)
        tx, ty = target
        best: tuple[float, _Section, float] | None = None
        for section in sections:
            (ax, ay), (bx, by) = self.xy[section.start], self.xy[section.end]
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 == 0.0 else ((tx - ax) * dx + (ty - ay) * dy) / length2
            t = min(max(t, 0.0), 1.0)
            d = math.hypot(tx - (ax + t * dx), ty - (ay + t * dy))
            if best is None or d < best[0] - 1e-12:
                best = (d, section, t)
        assert best is not None
        distance, section, t = best
        length = self._distance(self.xy[section.start], self.xy[section.end])
        if t * length <= (1.0 - t) * length:
            node, along = section.start, t * length
        else:
            node, along = section.end, (1.0 - t) * length
        if along < self.min_length:
            return _Nearest(self._distance(target, self.xy[node]), section, t, node)
        return _Nearest(distance, section, t, None)

    def _coincide_error(self, node: str, other: str, distance: float) -> ValueError:
        """Two nodes closer than :attr:`min_length` cannot be joined."""
        return ValueError(
            f"PEN routing: the connection point of {node!r} coincides with "
            f"that of {other!r} ({distance:.3g} m apart, less than the "
            f"shortest cable section of {self.min_length:g} m); a cable "
            "between them would have no length. Move one of the two sites."
        )

    def _split_at(self, section: _Section, t: float) -> str:
        """Insert a tap junction at the fraction ``t`` of ``section``."""
        (ax, ay), (bx, by) = self.xy[section.start], self.xy[section.end]
        name = f"pen_tap_{len(self.junctions)}"
        xy = (ax + t * (bx - ax), ay + t * (by - ay))
        self.xy[name] = xy
        self.junctions.append(
            PlannedJunction(name=name, position_xy=xy, depth_m=self.depth_m)
        )
        self._split(section, name)
        return name

    def _split(self, section: _Section, node: str) -> None:
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
