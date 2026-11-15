"""Pre-solve world diagnostics — counts, mesh budget, resolution, proximity.

This module collects **structural** diagnostics that work directly
on a :class:`~groundfield.world.World` (and optionally an
:class:`~groundfield.solver.engine.Engine` and a solved
:class:`~groundfield.solver.result.FieldResult`). It is the typical
counterpart to :mod:`groundfield.validation` (a *post-solve*
cross-engine check):

* :func:`diagnose` — one structured, aggregated report
  (:class:`DiagnosticsReport`) of everything that limits the accuracy
  or the meaning of a solve: coordinates that mix local and projected
  systems, discretisation quality, segment budget, close or
  interpenetrating wires, inert configuration and, after the solve,
  series truncation. One line per cause, not per electrode (review
  finding 10.70).
* :func:`world_statistics` — aggregate counts and lengths,
  bounding box, footprint area, conductor-length statistics.
* :func:`expected_segments` — predicts the number of point-source
  segments that the image-family discretiser will produce per
  electrode kind, plus a total. Useful for budgeting wall-clock
  cost before kicking off a 200-EFH run.
* :func:`check_segment_resolution` — the discretisation part of the
  report as plain strings (thin-wire ratio, electrode size vs. segment
  length, segment budget), one string per cause. The solve emits each
  of them as one warning.
* :func:`check_segment_budget` — the hard resource guard of the solver
  pipeline: a world whose predicted segment count, or whose dense nodal
  system (every conductor sub-piece included), exceeds the engine's limit
  raises :class:`ResourceLimitError` before anything of that size is
  allocated (finding 10.06; audit round 2, A2S-02). The limit is
  ``Engine.max_segments``, or :data:`HARD_LIMIT` for an engine that
  leaves it at ``None`` (user decision Q17, 2026-11-08).

Validity envelope
-----------------
* The segment counts follow the electrodes' own discretisation
  (:meth:`~groundfield.geometry.electrodes.RodElectrode.segments` and
  siblings) including the split at the soil-layer interfaces, i.e. the
  image-family discretiser that ``image``, ``image_2layer``,
  ``image_nlayer``, ``mom``, ``mom_sommerfeld``, ``cim`` and ``bem``
  share. FEM does not use it; for FEM the segment count is not
  predictive of cost.
* Wire-length / segment computations follow the *physical*
  geometry. Sub-segment overheads from cluster-bonding, KCL
  pseudo-nodes and lumped-conductor branches are not
  double-counted.

Mathematical background of the checks
-------------------------------------
The image family represents every wire segment of length $L$ and
radius $a$ by a line (or point) source with uniform current and
evaluates the potential on its axis (thin-wire approximation). The
self term of the reaction matrix behaves like
$2\\ln(L/a)/L$; it loses dominance when $L$ approaches $a$, so the
thin-wire check asks for $L \\ge 5a$ (:data:`MIN_THINWIRE_RATIO`). An
electrode feature shorter than one segment is represented by the
discretiser's lower floor only (one segment per wire, eight per ring),
which the resolution check reports. Two different wires closer than a
few radii violate the assumption of the point-source kernel that a
source sees another wire only through its midpoint: two bonded 3 m rods
of radius 8 mm, 2 mm apart, solve to 5.7 times the impedance of one rod
with the point kernel and to 0.998 times with the exact line kernel
(review finding 10.44); the proximity check uses
:func:`groundfield.conductors.find_close_wires` on the segments the
solver assembles. The dense reaction matrix of $N$ segments has
$N^2$ entries (8 bytes each); above
:data:`~groundfield.solver._kernels.MATRIX_BUDGET_BYTES` it is applied
in row blocks, so memory stays bounded but the work keeps growing like
$N^2$ per excitation — the reason for the hard segment limit. The dense
augmented nodal system of $n$ unknowns (current owners, nodes and
branches) is allocated whole, $16 n^2$ bytes complex.

Resource limit
--------------
The limit is set per engine with ``Engine(max_segments=...)``, a
positive integer that bounds both sizes, the leakage segments and the
unknowns of the dense nodal system; ``None`` (the default) takes
:data:`HARD_LIMIT` (20 000), read at call time, so a changed module
constant still acts as the default of every engine. Every function here
that takes an engine (:func:`check_segment_budget`,
:func:`expected_segments`, :func:`check_segment_resolution`,
:func:`diagnose`) and every segment backend, ``solve_mutual_matrix`` and
``solve_mutual_field`` included, check against the same limit;
``FieldResult.metadata["max_segments"]`` records it.

The soft notice is set per engine as well, ``Engine(warn_segments=...)``,
counted like ``max_segments`` (user decision Q31, 2026-11-08); ``None``
(the default) takes :data:`SOFT_LIMIT` (5 000), read at call time. At or
above this size every :meth:`Engine.solve
<groundfield.solver.engine.Engine.solve>`, with any backend, warns that it
may take from seconds to minutes on a laptop, and :func:`diagnose` reports
it as an *info* finding; ``solve_mutual_matrix`` and ``solve_mutual_field``
check the hard limit but emit no notice. For ``fem`` the segment count is
not its problem size (the mesh of each driven cluster is), and ``fem`` does
not refuse a world above the limit either. The two sizes are not checked
against each other: a notice above the engine's hard limit never appears,
because such a world is refused first.
:func:`check_segment_resolution` and :func:`diagnose` (``report.context
["warn_segments"]``) use the engine's value.

References
----------
- ADR-0003 (`docs/adr/0003-distributed-conductor-model.md`) —
  distributed-conductor topology used by :func:`expected_segments`
  for conductors with finite ``discretize_segment_length``.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = [
    "world_statistics",
    "expected_segments",
    "check_segment_resolution",
    "check_segment_budget",
    "diagnose",
    "Diagnostic",
    "DiagnosticsReport",
    "DiagnosticsWarning",
    "MixedCoordinatesWarning",
    "ResourceLimitError",
    "MIN_THINWIRE_RATIO",
    "SOFT_LIMIT",
    "HARD_LIMIT",
    "OVERLAP_REL_LENGTH",
    "OVERLAP_FRACTION",
]


# ---------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------

# Heuristic thresholds. Tuned for the typical envelope (image-family
# backends, 50 Hz, ρ in [50, 1000] Ω·m, electrode dimensions
# ~ 0.5 ... 30 m). The module-level public constants below are the
# canonical handles for tests, notebooks, and external callers — the
# leading-underscore aliases are kept for backwards compatibility.

MIN_THINWIRE_RATIO: float = 5.0
"""Minimum recommended ratio of segment_length to wire_radius.

Below this, the thin-wire approximation that underpins the
average-potential method becomes increasingly biased: the
self-action integral is computed under the assumption
``segment_length >> wire_radius``.
"""

SOFT_LIMIT: int = 5_000
"""Problem size above which the user should be aware of cost.

Compared with the segment count and with the size of the dense nodal
system (:func:`expected_segments`, key ``nodal_unknowns``).

The reaction matrix scales as :math:`O(N^2)` in memory (bounded by row
blocks above the kernel budget) and the multi-port assembly as
:math:`O(N^2)` per excitation, so ~ 5 000 segments is roughly where
solve time becomes minutes rather than seconds on a typical laptop.

The default of every engine that leaves ``Engine.warn_segments`` at
``None`` (user decision Q31), read at call time; an engine moves the
notice with ``Engine(warn_segments=...)``.
"""

HARD_LIMIT: int = 20_000
"""Default problem size above which the solver refuses to start.

The image-family pipeline, the ``mom`` family and ``solve_mutual_*``
raise :class:`ResourceLimitError` for a world whose predicted segment
count, or whose dense nodal system (current owners, nodes and branches,
every sub-piece of a distributed conductor included), exceeds the
engine's limit, before anything of that size is allocated (finding
10.06: a ring entered in millimetres predicts 62 832 segments; audit
round 2, A2S-02: an isolated conductor in mm-long sub-pieces). The limit
of one engine is ``Engine(max_segments=...)``; this constant is the
default of every engine that leaves ``max_segments`` at ``None`` and is
read at solve time, so setting ``groundfield.diagnostics.HARD_LIMIT``
changes the default for the whole process.
"""

# Backwards-compatible private aliases (not read by this module any
# more: the public constants are, at call time).
_MIN_THINWIRE_RATIO = MIN_THINWIRE_RATIO
_BUDGET_WARN_THRESHOLD = SOFT_LIMIT
_BUDGET_HARD_THRESHOLD = HARD_LIMIT

#: Relative proximity threshold handed to ``find_close_wires``: a pair of
#: wires closer than this multiple of the sum of their radii is reported.
PROXIMITY_REL_THRESHOLD: float = 5.0

#: Overlap of inductive conductors (check ``overlap`` of :func:`diagnose`):
#: two conductors closer than touching (axis distance below the sum of
#: their radii, d_t) lie beside each other when that stretch is longer than
#: OVERLAP_REL_LENGTH * d_t and OVERLAP_FRACTION of the shorter conductor.
#: Two conductors leaving a common point at an angle phi stay closer than
#: touching over about d_t / phi; the touching rule changes their mutual
#: inductance over that stretch by about (mu0 / 2 pi) d_t / phi (the mean
#: of ln(d_t / d) along a linear divergence is 1), against a self
#: inductance of about (mu0 / 2 pi) l (ln(2 l / r) - 1), ten times
#: (mu0 / 2 pi) l for the cables of an LV network. A stretch of 5 % of the
#: shorter conductor therefore moves the coupling by about 0.5 % of a self
#: inductance; a branch is reported only below about d_t / (0.05 l): 0.46
#: deg for 12 mm touching and a 30 m drop, 0.1 rad at most.
OVERLAP_REL_LENGTH: float = 10.0
OVERLAP_FRACTION: float = 0.05
# Relative tolerance at the touching distance (audit W3B-03): two cables
# drawn exactly at d_t lie closer by round-off (1e-14 m for 5 mm cables),
# and the inductance matrix evaluates them the same either way (it clamps
# the distance at d_t), so a pair counts as closer than touching only below
# d_t (1 - _OVERLAP_TOUCH_TOL).
_OVERLAP_TOUCH_TOL = 1e-9

#: Number of names listed per cause in the aggregated text.
_MAX_NAMES = 5


class ResourceLimitError(ValueError):
    """The predicted problem size exceeds the configured limit.

    Raised before any array of the problem size is allocated. Check the
    units of the geometry (metres) and ``Engine.segment_length``; to
    solve a large world on purpose, raise the limit of the engine,
    ``Engine(max_segments=...)`` (default
    ``groundfield.diagnostics.HARD_LIMIT``).
    """


class DiagnosticsWarning(UserWarning):
    """One aggregated warning with the findings of :func:`diagnose`."""


class MixedCoordinatesWarning(UserWarning):
    """The world mixes local and projected coordinates (user decision D5).

    Emitted once per solve by every solve in a local frame (the image
    family, ``mom``, ``mom_sommerfeld`` and ``solve_mutual_*``), before
    anything is discretised, when points of the world lie both within
    10 km of the origin and beyond 1000 km from it: electrodes at UTM or
    Gauss-Krüger coordinates and a node or electrode near (0, 0), almost
    always given in the wrong system by mistake. The world is solved as
    given, its two groups some 33 000 km apart (audit W2B-04, measured with
    a 12 m × 7 m outline and a 3 m rod at UTM coordinates: a return rod given
    at local (80, 15) m raises Z_in by 0.43 %, a PEN cable to a cabinet at local coordinates
    becomes a 33 118 km conductor, Z_in 7494 + j25396 Ω instead of
    5.39 + j0.19 Ω at 50 Hz with Carson's earth return, and the Sommerfeld
    earth return does not finish).
    :func:`diagnose` reports the same world under ``coordinates``.

    Remedy: give every point in one coordinate system.
    """


Severity = Literal["info", "warning"]


@dataclass(frozen=True)
class Diagnostic:
    """One finding of :func:`diagnose`.

    Attributes
    ----------
    category
        Cause: ``"coordinates"``, ``"thin_wire"``, ``"resolution"``,
        ``"budget"``, ``"proximity"``, ``"overlap"``,
        ``"interface_contact"``, ``"inert"`` or ``"series"``.
    severity
        ``"warning"`` (the result is likely biased or the setting has no
        effect) or ``"info"`` (worth knowing, usually harmless).
    subjects
        Names the finding is about (electrodes, conductors, the
        world, a setting).
    message
        One sentence for this subject.
    data
        Machine-readable values (units in the key names).
    """

    category: str
    severity: Severity
    subjects: tuple[str, ...]
    message: str
    data: dict[str, Any] = field(default_factory=dict)


# Text per cause of the aggregated report:
# prefix, (singular, plural) subject, predicate, explanation. Causes
# without a subject text (budget, series) list their messages only.
_CAUSES: dict[str, tuple[str, tuple[str, str], str, str]] = {
    "thin_wire": (
        "thin-wire",
        ("wire has", "wires have"),
        "segment_length / wire_radius below {ratio:g}",
        "the thin-wire self-action assumption may bias the result; use "
        "longer segments or the true (smaller) radius",
    ),
    "resolution": (
        "resolution",
        ("electrode has", "electrodes have"),
        "a feature shorter than segment_length {ds:g} m",
        "the discretiser falls back to its lower floor (one segment per "
        "wire, eight per ring) and the result is coarse there; refine "
        "segment_length",
    ),
    "proximity": (
        "proximity",
        (
            "pair of leakage wires of different owners is",
            "pairs of leakage wires of different owners are",
        ),
        "closer than {rel:g}x the sum of their radii",
        "the point-source kernel assumes separated wires (two bonded 3 m "
        "rods 2 mm apart solve to 5.7 times the impedance of one rod, "
        "finding 10.44); use Engine(kernel='line'), whose exact near-pair "
        "integrals stay physical (0.998 times), or merge such wires into "
        "one electrode",
    ),
    "overlap": (
        "overlap",
        ("pair of inductive conductors lies", "pairs of inductive conductors lie"),
        "beside each other closer than touching (the sum of their radii) "
        "over more than {rel_length:g} times that distance and {fraction:g} % "
        "of the shorter one",
        "the inductance matrix evaluates them as touching wires, so their "
        "coupling over that stretch follows the radii, not the drawn spacing "
        "(a service drop drawn on its PEN main, two cables on one axis); draw "
        "them at their real distance, or model a concentric cable with "
        "conductor_type='cable_shield' around its core",
    ),
    "inert": (
        "inert",
        ("setting has", "settings have"),
        "no effect on the solve",
        "",
    ),
    "interface_contact": (
        "interface contact",
        ("rod end lies", "rod ends lie"),
        "within one wire radius of a layer interface",
        "the thin-wire model has no end face, so the contact with the layer "
        "beyond is all or nothing there (a solid 4.5 m rod, a = 8 mm, in "
        "1000/30 Ohm m: +21 % with the tip at the interface, -4.6 % 1 mm past "
        "it, audit A3S-05); keep rod ends a few wire radii from an interface "
        "or bracket the result by moving the end by +-2a",
    ),
}


@dataclass
class DiagnosticsReport:
    """Aggregated result of :func:`diagnose`.

    Attributes
    ----------
    findings
        All findings, in the order the checks ran.
    context
        Values the aggregated text refers to (``segment_length_m``,
        ``min_thinwire_ratio``, ...).
    """

    findings: list[Diagnostic] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.findings)

    def __bool__(self) -> bool:
        return bool(self.findings)

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self.findings)

    def categories(self) -> list[str]:
        """Causes present in the report, in check order."""
        seen: list[str] = []
        for d in self.findings:
            if d.category not in seen:
                seen.append(d.category)
        return seen

    def by_category(self, category: str) -> list[Diagnostic]:
        """Findings of one cause."""
        return [d for d in self.findings if d.category == category]

    def lines(self, *, min_severity: Severity = "info") -> list[str]:
        """One aggregated line per cause.

        Parameters
        ----------
        min_severity
            ``"info"`` (default) lists every cause, ``"warning"`` only the
            causes with at least one warning.
        """
        out = []
        for cat in self.categories():
            items = self.by_category(cat)
            if min_severity == "warning" and not any(
                d.severity == "warning" for d in items
            ):
                continue
            out.append(self._line(cat, items))
        return out

    def _line(self, category: str, items: list[Diagnostic]) -> str:
        shown = "; ".join(d.message for d in items[:_MAX_NAMES])
        if len(items) > _MAX_NAMES:
            shown += f"; and {len(items) - _MAX_NAMES} more"
        if category not in _CAUSES:
            return f"{category}: {shown}"
        prefix, subject, predicate, why = _CAUSES[category]
        predicate = predicate.format(
            ratio=self.context.get("min_thinwire_ratio", MIN_THINWIRE_RATIO),
            ds=self.context.get("segment_length_m", float("nan")),
            rel=self.context.get("proximity_rel_threshold", PROXIMITY_REL_THRESHOLD),
            rel_length=self.context.get("overlap_rel_length", OVERLAP_REL_LENGTH),
            fraction=100.0 * self.context.get("overlap_fraction", OVERLAP_FRACTION),
        )
        n = len(items)
        tail = f" — {why}." if why else ""
        return f"{prefix}: {n} {subject[n > 1]} {predicate}: {shown}{tail}"

    def summary(self) -> str:
        """The aggregated text (one paragraph per cause)."""
        return str(self)

    def __str__(self) -> str:
        return "\n".join(self.lines()) if self.findings else "no findings"

    def warn(self, *, min_severity: Severity = "warning", stacklevel: int = 2) -> None:
        """Emit the report as one :class:`DiagnosticsWarning` (if not empty).

        Parameters
        ----------
        min_severity
            Lowest severity that is included (default ``"warning"``).
        stacklevel
            Passed to :func:`warnings.warn`.
        """
        text = self.lines(min_severity=min_severity)
        if text:
            warnings.warn(
                "groundfield diagnostics:\n" + "\n".join(text),
                DiagnosticsWarning,
                stacklevel=stacklevel,
            )

    def to_records(self) -> list[dict[str, Any]]:
        """The findings as plain dictionaries (for tables and JSON)."""
        return [
            {
                "category": d.category,
                "severity": d.severity,
                "subjects": list(d.subjects),
                "message": d.message,
                **d.data,
            }
            for d in self.findings
        ]


# ---------------------------------------------------------------------
# Wire-length and segment-count primitives
# ---------------------------------------------------------------------


def _electrode_wire_length(e) -> float:
    """Total wire length of an electrode in metres (``e.total_length()``)."""
    return float(e.total_length())


def _layer_interfaces(soil) -> tuple[float, ...] | None:
    """Interface depths in m at which the segment-based backends cut wires.

    The interfaces of the canonical layer stack, i.e. only where the
    resistivity changes (:func:`groundfield.solver._layered.interface_depths`,
    the rule of every backend). Up to audit round 2 every declared
    interface was counted, so a stack with two layers of one resistivity
    predicted one segment more than the solve (A2S-07).
    """
    from groundfield.solver._layered import interface_depths

    return interface_depths(soil) or None


def _segment_lower_bound(e, ds: float) -> int:
    """``ceil(total_length / ds)``: a lower bound of the segment count.

    Every wire of length $L_w$ is split into at least
    $\\lceil L_w/ds \\rceil$ segments and
    $\\sum_w \\lceil L_w/ds \\rceil \\ge \\lceil \\sum_w L_w/ds \\rceil$;
    computed in closed form, without discretising.
    """
    return max(1, int(math.ceil(float(e.total_length()) / ds)))


def _hard_limit(engine: "Engine | None" = None) -> int:
    """The problem-size limit of ``engine`` (user decision Q17).

    ``engine.max_segments`` if the engine sets one, else :data:`HARD_LIMIT`
    read now (so a changed module constant acts as the default).
    """
    value = None if engine is None else getattr(engine, "max_segments", None)
    return int(HARD_LIMIT) if value is None else int(value)


def _limit_label(engine: "Engine | None", limit: int) -> str:
    """Name and value of the limit in a message: the knob the user turns."""
    if engine is not None and getattr(engine, "max_segments", None) is not None:
        return f"Engine.max_segments = {limit}"
    return f"HARD_LIMIT = {limit} (the default of Engine.max_segments)"


def _soft_limit(engine: "Engine | None" = None) -> int:
    """The size of the soft notice of ``engine`` (user decision Q31).

    ``engine.warn_segments`` if the engine sets one, else :data:`SOFT_LIMIT`
    read now (so a changed module constant acts as the default).
    """
    value = None if engine is None else getattr(engine, "warn_segments", None)
    return int(SOFT_LIMIT) if value is None else int(value)


def _soft_label(engine: "Engine | None", limit: int) -> str:
    """Name and value of the notice size in a message: the knob to turn."""
    if engine is not None and getattr(engine, "warn_segments", None) is not None:
        return f"Engine.warn_segments = {limit}"
    return f"SOFT_LIMIT = {limit} (the default of Engine.warn_segments)"


#: How a message tells the user to solve a large world on purpose.
_RAISE_THE_LIMIT = (
    "to solve a large world on purpose, raise the limit of the engine, "
    "Engine(max_segments=...)"
)


def _electrode_segment_count(
    e, ds: float, layer_interfaces=None, limit: int | None = None
) -> int:
    """Segment count of one electrode, as the solver discretises it.

    Exact (:meth:`segment_count` of the electrode) unless the closed-form
    lower bound already exceeds ``limit`` (default :data:`HARD_LIMIT`);
    then that bound is returned so a unit error cannot allocate the
    segments of a huge electrode just to count them.
    """
    lower = _segment_lower_bound(e, ds)
    if lower > (_hard_limit() if limit is None else limit):
        return lower
    return int(e.segment_count(ds, layer_interfaces))


def _conductor_segment_count(c) -> int:
    """Point-source segments contributed by a conductor.

    Lumped conductors (``discretize_segment_length is None``)
    contribute zero point-source segments — they are pure
    nodal-analysis branches.

    Distributed conductors contribute one midpoint per sub-piece
    (``conductor.n_segments``, also a single one) **only** when
    ``coupling_to_soil == "galvanic"``. With
    ``coupling_to_soil == "isolated"`` (typical jacketed cable —
    PEN inside an NAYY, etc.) the longitudinal-branch chain is
    represented by interior KCL nodes that do **not** appear in
    :attr:`FieldResult.point_sources` because they cannot leak
    current to the soil.
    """
    if not c.is_distributed:
        return 0
    if c.coupling_to_soil != "galvanic":
        return 0
    return int(c.n_segments)


def _conductor_unknowns(c) -> int:
    """Unknowns a conductor adds to the dense nodal system (upper bound).

    A lumped (or single isolated) conductor is one branch. A conductor
    chain (:func:`groundfield.solver._discretize._is_conductor_chain`) of
    $n$ sub-pieces adds, when isolated, $n$ branches and $n - 1$ interior
    nodes ($2n - 1$) and, when galvanic, $n$ leaking owners with their $n$
    midpoint nodes and $n + 1$ branches ($3n + 1$). Closed form: nothing
    is discretised (a sub-piece length in mm instead of m must not
    allocate the chain just to count it).
    """
    from groundfield.solver._discretize import _is_conductor_chain

    if not _is_conductor_chain(c):
        return 1
    n = int(c.n_segments)
    if c.coupling_to_soil == "galvanic":
        return 3 * n + 1
    return 2 * n - 1


def _straight_wire_count(electrode) -> int:
    """Straight wires of an electrode: its uniform-current owners per layer.

    The image family gives every straight wire its own owner (user
    decisions Q30 and Q44 of 2026.11): a full wire line of a grid, an arm
    of a star, a straight run of a polyline; rods, strips and rings are one
    wire. Counted from the geometry, without discretising.
    """
    kind = getattr(electrode, "kind", None)
    if kind == "grid_mesh":
        return int(electrode.n_x + 1 + electrode.n_y + 1)
    if kind == "mesh":
        return int(sum(electrode._wire_counts()))
    if kind == "star":
        return int(electrode.n_arms)
    if kind == "polyline":
        from groundfield.geometry.primitives import _straight_runs

        return 1 + max(_straight_runs(electrode.edges, closed=electrode.closed))
    return 1


def _nodal_unknowns(world: "World", layer_interfaces=None) -> int:
    """Size of the dense augmented nodal system of the solver (upper bound).

    The image and ``mom`` families solve one dense system with one unknown
    per current owner (a straight wire of an electrode — a wire line of a
    grid, an arm of a star, a straight run of a polyline, user decisions
    Q30 and Q44 —, its part in every further soil layer, every leaking
    conductor sub-piece), per node (one per electrode cluster, junction and
    interior conductor node), per branch (lumped conductors and every
    chain sub-piece) and per voltage source (its current, 2026.11). Its matrix has $n^2$ entries and is allocated whole,
    so a long conductor chain of fine sub-pieces dominates the memory long
    before the leakage segments do (audit round 2, A2S-02: a 100 m PEN in
    2000 isolated pieces, 12 segments and about 4000 unknowns, took
    407 MiB). The count is exact for unbonded electrodes that carry
    current; bonded electrodes share one node, and a source-free electrode
    of one owner that is a cluster of its own is left out of the solve
    (up to audit W1A-06 an electrode counted as one owner: a 100 m
    20 x 20 grid with 50 L-shaped building outlines read 102 unknowns
    instead of 393).

    Parameters
    ----------
    world
        World to inspect.
    layer_interfaces
        Interface depths in m at which electrodes get one owner per
        layer.

    Returns
    -------
    int
        Owners plus nodes plus branches plus one unknown (the source
        current) per voltage source; an electrode counts as one owner
        per straight wire and soil layer and one node (every electrode its
        own cluster).
    """
    interfaces = tuple(layer_interfaces or ())
    n = len(world.junctions)
    for e in world.electrodes:
        crossings = 0
        if interfaces:
            lower, upper = e.bounds()
            crossings = sum(1 for h in interfaces if lower[2] < h < upper[2])
        n += 1 + _straight_wire_count(e) * (1 + crossings)
    n += sum(_conductor_unknowns(c) for c in world.conductors)
    # Modified nodal analysis: one current unknown per voltage source.
    n += sum(1 for s in world.sources if s.kind == "voltage")
    return n


def _check_ds(engine: "Engine") -> float:
    ds = float(engine.segment_length)
    if not math.isfinite(ds) or ds <= 0.0:
        raise ValueError(
            f"engine.segment_length must be > 0, got {engine.segment_length!r}."
        )
    return ds


# ---------------------------------------------------------------------
# world_statistics
# ---------------------------------------------------------------------


def world_statistics(world: "World") -> dict[str, Any]:
    """Return a structural snapshot of a world.

    Aggregates counts, lengths and the bounding box; complements
    :meth:`World.summary` (one-line text) with a richer
    machine-readable dictionary that scales to production-grade networks.

    Parameters
    ----------
    world
        World to inspect.

    Returns
    -------
    dict
        Keys:

        * ``n_electrodes`` (int),
        * ``n_electrodes_by_kind`` (``dict[str, int]``: rod / ring
          / strip / polyline / star / mesh / grid_mesh),
        * ``n_junctions`` (int),
        * ``n_conductors`` (int),
        * ``n_conductors_by_type`` (``dict[str, int]``),
        * ``n_distributed_conductors`` / ``n_lumped_conductors``,
        * ``n_galvanic_conductors`` / ``n_isolated_conductors``,
        * ``n_sources`` (int),
        * ``total_electrode_wire_length_m`` (float),
        * ``total_conductor_length_m`` (float),
        * ``conductor_length_stats``: ``{min, median, max, mean}``
          in metres (empty dict when no conductors),
        * ``bounds_3d``: ``(x_min, x_max, y_min, y_max, z_min,
          z_max)`` in metres,
        * ``footprint_xy``: ``(x_min, x_max, y_min, y_max)``,
        * ``footprint_area_m2`` (float),
        * ``depth_range_m``: ``(z_min, z_max)``,
        * ``has_layered_soil`` (bool — diagnostic flag set when
          the soil model is :class:`TwoLayerSoil` or
          :class:`MultiLayerSoil`).
    """
    from groundfield.postprocess.geometry_plot import world_bounds_3d
    from groundfield.soil.models import MultiLayerSoil, TwoLayerSoil

    by_kind: Counter[str] = Counter()
    total_e_wire = 0.0
    for e in world.electrodes:
        by_kind[str(e.kind)] += 1
        total_e_wire += _electrode_wire_length(e)

    by_type: Counter[str] = Counter(c.conductor_type for c in world.conductors)
    n_distributed = sum(1 for c in world.conductors if c.is_distributed)
    n_galvanic = sum(1 for c in world.conductors if c.coupling_to_soil == "galvanic")
    cond_lengths = [float(c.length) for c in world.conductors]
    total_c_length = float(sum(cond_lengths))
    if cond_lengths:
        cond_stats = {
            "min": float(min(cond_lengths)),
            "median": float(median(cond_lengths)),
            "max": float(max(cond_lengths)),
            "mean": float(sum(cond_lengths) / len(cond_lengths)),
        }
    else:
        cond_stats = {}

    x_min, x_max, y_min, y_max, z_min, z_max = world_bounds_3d(world)
    footprint = (x_min, x_max, y_min, y_max)
    area = (x_max - x_min) * (y_max - y_min)

    has_layered = isinstance(world.soil, (TwoLayerSoil, MultiLayerSoil))

    return {
        "n_electrodes": len(world.electrodes),
        "n_electrodes_by_kind": dict(by_kind),
        "n_junctions": len(world.junctions),
        "n_conductors": len(world.conductors),
        "n_conductors_by_type": dict(by_type),
        "n_distributed_conductors": n_distributed,
        "n_lumped_conductors": len(world.conductors) - n_distributed,
        "n_galvanic_conductors": n_galvanic,
        "n_isolated_conductors": len(world.conductors) - n_galvanic,
        "n_sources": len(world.sources),
        "total_electrode_wire_length_m": float(total_e_wire),
        "total_conductor_length_m": total_c_length,
        "conductor_length_stats": cond_stats,
        "bounds_3d": (x_min, x_max, y_min, y_max, z_min, z_max),
        "footprint_xy": footprint,
        "footprint_area_m2": float(area),
        "depth_range_m": (z_min, z_max),
        "has_layered_soil": has_layered,
    }


# ---------------------------------------------------------------------
# expected_segments and the resource guard
# ---------------------------------------------------------------------


def expected_segments(
    world: "World",
    engine: "Engine",
    *,
    layer_interfaces: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Predict the number of point-source segments after discretisation.

    Uses the electrodes' own discretisation
    (:meth:`~groundfield.geometry.electrodes.RodElectrode.segment_count`
    and siblings), so the prediction is *exact* for the image family,
    including the split of wires at the soil-layer interfaces. An
    electrode whose closed-form lower bound
    $\\lceil L_\\text{total}/ds \\rceil$ already exceeds the engine's
    limit (``Engine.max_segments``, default :data:`HARD_LIMIT`) is
    counted by that bound instead, so that a unit error does not
    allocate a huge discretisation just to count it. The FEM backend
    (:mod:`groundfield.solver.fem`) uses an axisymmetric volume mesh that
    is unrelated to ``segment_length``; for FEM this prediction is not
    informative.

    Parameters
    ----------
    world
        World to inspect.
    engine
        Engine carrying ``segment_length`` (in metres) and the limit
        ``max_segments``.
    layer_interfaces
        Interface depths in m at which wires are split. ``None``
        (default) takes them from ``world.soil`` (``h_1`` of a two-layer
        soil, the layer boundaries of a layered soil).

    Returns
    -------
    dict
        Keys:

        * ``per_electrode``: ``dict[name, int]`` with the segment
          count of every named electrode,
        * ``per_kind``: ``dict[kind, int]`` aggregated counts per
          electrode kind,
        * ``electrode_total`` (int),
        * ``per_conductor``: ``dict[name, int]`` (only listed for
          distributed conductors; lumped conductors contribute 0
          and are omitted),
        * ``conductor_total`` (int),
        * ``total`` (int) — sum of electrode and conductor segments,
        * ``nodal_unknowns`` (int) — size of the dense augmented nodal
          system (owners, nodes and branches, including every conductor
          sub-piece), an upper bound; checked against the limit like
          ``total``,
        * ``limit`` (int) — the limit the solve checks both sizes
          against: ``engine.max_segments``, or :data:`HARD_LIMIT` if the
          engine leaves it at ``None``.

    Raises
    ------
    ValueError
        If ``engine.segment_length`` is not strictly positive.
    """
    ds = _check_ds(engine)
    limit = _hard_limit(engine)
    if layer_interfaces is None:
        layer_interfaces = _layer_interfaces(world.soil)

    per_electrode: dict[str, int] = {}
    per_kind: Counter[str] = Counter()
    e_total = 0
    for e in world.electrodes:
        n = _electrode_segment_count(e, ds, layer_interfaces, limit)
        per_electrode[e.name] = n
        per_kind[e.kind] += n
        e_total += n

    per_conductor: dict[str, int] = {}
    c_total = 0
    for c in world.conductors:
        n = _conductor_segment_count(c)
        if n > 0:
            per_conductor[c.name] = n
        c_total += n

    return {
        "per_electrode": per_electrode,
        "per_kind": dict(per_kind),
        "electrode_total": e_total,
        "per_conductor": per_conductor,
        "conductor_total": c_total,
        "total": e_total + c_total,
        "nodal_unknowns": _nodal_unknowns(world, layer_interfaces),
        "limit": limit,
    }


def check_segment_budget(
    world: "World",
    engine: "Engine",
    layer_interfaces: Sequence[float] | None = None,
) -> int:
    """Hard resource guard of the solver pipeline (finding 10.06).

    Predicts the problem size from the geometry and raises before
    anything of that size is allocated. Two sizes are checked against
    the engine's limit, ``engine.max_segments`` (user decision Q17), or
    :data:`HARD_LIMIT` (read at call time) if the engine leaves it at
    ``None``:

    * the leakage segments (reaction matrix): first the closed-form
      lower bound $\\sum_e \\lceil L_e/ds \\rceil$, then, if that is
      within the limit, the exact count;
    * the unknowns of the dense augmented nodal system (owners, nodes
      and branches, including every sub-piece and interior node of a
      conductor chain), in closed form (audit round 2, A2S-02: up to
      then an isolated conductor of 2000 sub-pieces counted as zero).

    Parameters
    ----------
    world
        World to be solved.
    engine
        Engine carrying ``segment_length`` (m) and ``max_segments``.
    layer_interfaces
        Interface depths in m at which the backend splits wires.

    Returns
    -------
    int
        The predicted segment count (electrodes plus galvanic
        conductor segments).

    Raises
    ------
    ResourceLimitError
        If either size exceeds the limit.
    """
    ds = _check_ds(engine)
    limit = _hard_limit(engine)
    label = _limit_label(engine, limit)
    unknowns = _nodal_unknowns(world, layer_interfaces)
    if unknowns > limit:
        raise ResourceLimitError(
            f"World {world.name!r} needs a nodal system of {unknowns} unknowns "
            "(current owners, nodes and branches, among them every sub-piece "
            "and interior node of the distributed conductors), more than "
            f"{label}: its dense matrix has {unknowns * unknowns:.3g} "
            f"entries ({16.0 * unknowns * unknowns / 2**30:.3g} GiB complex). "
            "Check discretize_segment_length of the conductors (metres); "
            f"{_RAISE_THE_LIMIT}."
        )
    conductors = sum(_conductor_segment_count(c) for c in world.conductors)
    lower = conductors + sum(_segment_lower_bound(e, ds) for e in world.electrodes)
    if lower > limit:
        n, exact = lower, False
    else:
        n = conductors + sum(
            int(e.segment_count(ds, layer_interfaces)) for e in world.electrodes
        )
        exact = True
    if n > limit:
        what = "" if exact else "at least "
        raise ResourceLimitError(
            f"World {world.name!r} discretises into {what}{n} segments at "
            f"segment_length = {ds:g} m, more than {label} "
            f"(a dense reaction matrix of {n} segments has {n * n:.3g} "
            f"entries, {8.0 * n * n / 2**30:.3g} GiB). Check the units of the "
            f"geometry (metres) and Engine.segment_length; {_RAISE_THE_LIMIT}."
        )
    return n


# ---------------------------------------------------------------------
# The checks of the report
# ---------------------------------------------------------------------


def _coordinate_findings(world: "World") -> list[Diagnostic]:
    """Mixed local / projected coordinates and very wide worlds (decision D5)."""
    from groundfield.solver._frame import (
        LOCAL_RADIUS_M,
        MAX_EXTENT_M,
        PROJECTED_RADIUS_M,
        _km,
        coordinate_span,
    )

    span = coordinate_span(world)
    if span is None:
        return []
    data: dict[str, Any] = {
        "extent_m": span.extent_m,
        "bounds_xy_m": span.bounds_xy,
    }
    if span.mixed:
        data.update(
            near=list(span.near),
            far=list(span.far),
            spanning=list(span.spanning),
            spanning_lengths_m=list(span.spanning_lengths_m),
            local_radius_m=LOCAL_RADIUS_M,
            projected_radius_m=PROJECTED_RADIUS_M,
        )
        return [
            Diagnostic(
                "coordinates",
                "warning",
                (world.name,),
                span.message() + " The solve warns with MixedCoordinatesWarning.",
                data,
            )
        ]
    if span.extent_m > MAX_EXTENT_M:
        data["max_extent_m"] = MAX_EXTENT_M
        return [
            Diagnostic(
                "coordinates",
                "info",
                (world.name,),
                f"the horizontal extent of the world is {_km(span.extent_m)} "
                "(the diagonal of its bounding box), more than "
                f"{MAX_EXTENT_M / 1e3:g} km: check that every point is given in "
                "metres and in one coordinate system (two UTM zones, or the "
                "zone prefix on some eastings only).",
                data,
            )
        ]
    return []


def _resolution_findings(world: "World", engine: "Engine") -> list[Diagnostic]:
    ds = _check_ds(engine)
    out: list[Diagnostic] = []
    for e in world.electrodes:
        ratio = ds / float(e.wire_radius)
        if ratio < MIN_THINWIRE_RATIO:
            out.append(
                Diagnostic(
                    "thin_wire",
                    "warning",
                    (e.name,),
                    f"electrode '{e.name}' ({e.kind}) {ratio:.1f}",
                    {"ratio": ratio, "wire_radius_m": float(e.wire_radius)},
                )
            )
    for c in world.conductors:
        if not c.is_distributed:
            continue
        ratio = float(c.discretize_segment_length) / float(c.wire_radius)
        if ratio < MIN_THINWIRE_RATIO:
            out.append(
                Diagnostic(
                    "thin_wire",
                    "warning",
                    (c.name,),
                    f"distributed conductor '{c.name}' {ratio:.1f} "
                    "(discretize_segment_length / wire_radius)",
                    {"ratio": ratio, "wire_radius_m": float(c.wire_radius)},
                )
            )
    for e in world.electrodes:
        feature = float(e.min_feature_length())
        if feature < ds:
            out.append(
                Diagnostic(
                    "resolution",
                    "warning",
                    (e.name,),
                    f"'{e.name}' ({e.kind}, {feature:.3g} m)",
                    {"min_feature_length_m": feature, "segment_length_m": ds},
                )
            )
    est = expected_segments(world, engine)
    n_total = int(est["total"])
    n_nodal = int(est["nodal_unknowns"])
    limit = int(est["limit"])
    label = _limit_label(engine, limit)
    if n_total > limit:
        out.append(
            Diagnostic(
                "budget",
                "warning",
                (world.name,),
                f"predicted total segment count {n_total} exceeds {label}: the "
                "segment backends refuse such a world (ResourceLimitError) "
                "unless Engine.max_segments is raised. Consider a coarser "
                "segment_length or a sparser distributed-conductor "
                "discretisation.",
                {"n_segments": n_total, "limit": limit},
            )
        )
    elif n_nodal > limit:
        out.append(
            Diagnostic(
                "budget",
                "warning",
                (world.name,),
                f"predicted nodal system of {n_nodal} unknowns (owners, nodes and "
                f"branches, among them every conductor sub-piece) exceeds {label}: "
                "the solve refuses such a world (ResourceLimitError) unless "
                "Engine.max_segments is raised. Consider a sparser "
                "distributed-conductor discretisation (discretize_segment_length "
                "in metres).",
                {"n_unknowns": n_nodal, "limit": limit},
            )
        )
    elif max(n_total, n_nodal) >= (notice := _soft_limit(engine)):
        # user decision Q31: the size of the notice is the engine's.
        what = (
            f"predicted total segment count {n_total}"
            if n_total >= n_nodal
            else f"predicted nodal system of {n_nodal} unknowns"
        )
        out.append(
            Diagnostic(
                "budget",
                "info",
                (world.name,),
                f"{what} is at or above {_soft_label(engine, notice)} — solve "
                "time may run from seconds to minutes on a typical laptop.",
                {"n_segments": n_total, "n_unknowns": n_nodal, "limit": notice},
            )
        )
    return out


_BOND_NAME = re.compile(r"^(?P<root>.+)_bond_[0-9]+$")
"""Name of a conductor that bonds a wire to its foundation's anchor."""


def _foundation_roots(world: "World") -> dict[str, str]:
    """Name of the foundation every bonded wire of a generated foundation
    belongs to.

    A generated rectangular foundation is several straight wires (the anchor
    ``<name>`` and ``<name>_w<j>``, see
    :func:`groundfield.postprocess._footprint.foundation_wires`), each
    bonded to the anchor by a conductor ``<name>_bond_<k>``. Only a wire
    bonded so maps to its anchor (audit R4A-04: names alone made unbonded
    strips ``earth`` / ``earth_w1`` one owner); every other electrode maps
    to itself.
    """
    from groundfield.postprocess._footprint import _WIRE_NAME

    names = {e.name for e in world.electrodes}
    out = {n: n for n in names}
    for c in world.conductors:
        bond = _BOND_NAME.match(c.name)
        if bond is None:
            continue
        root = bond.group("root")
        wire = _WIRE_NAME.match(c.end_electrode or "")
        if (
            c.start_electrode == root
            and root in names
            and wire is not None
            and wire.group("root") == root
            and c.end_electrode in names
        ):
            out[c.end_electrode] = root
    return out


def _shared_vertices(world: "World", owner: dict[str, str]):
    """Vertices that two bonded wires of one foundation share.

    Returns a map from a sorted pair of wire names to the array of their
    common vertices (m, shape ``(k, 3)``): the corners where two sides meet
    and the crossings of a mesh (its wires have a vertex at every crossing).
    """
    import numpy as np

    groups: dict[str, list] = {}
    for e in world.electrodes:
        groups.setdefault(owner.get(e.name, e.name), []).append(e)
    out: dict[tuple[str, str], Any] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        where: dict[tuple[float, float, float], set[str]] = {}
        for e in members:
            for run in e.wire_polylines():
                for v in np.asarray(run, dtype=float):
                    key = tuple(float(x) for x in np.round(v, 6))
                    where.setdefault(key, set()).add(e.name)
        for key, wires in where.items():
            if len(wires) < 2:
                continue
            ordered = sorted(wires)
            for i, a in enumerate(ordered):
                for b in ordered[i + 1 :]:
                    out.setdefault((a, b), []).append(key)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _leakage_wires(world: "World", engine: "Engine | None"):
    """End points, radii and names of every leakage wire segment.

    Electrode segments carry the electrode's own name; the grouping of the
    bonded wires of one foundation is applied to the hits
    (:func:`_proximity_findings`).
    """
    import numpy as np

    ends, radii, names = [], [], []
    interfaces = _layer_interfaces(world.soil)
    for e in world.electrodes:
        if engine is not None:
            geo = e.segments(float(engine.segment_length), interfaces)
            half = 0.5 * geo.lengths[:, None] * geo.directions
            seg = np.stack([geo.midpoints - half, geo.midpoints + half], axis=1)
        else:
            parts = [
                np.stack([p[:-1], p[1:]], axis=1)
                for p in e.wire_polylines()
                if len(p) > 1
            ]
            seg = np.concatenate(parts, axis=0)
        ends.append(seg)
        radii.extend([float(e.wire_radius)] * len(seg))
        names.extend([e.name] * len(seg))
    for c in world.conductors:
        n = _conductor_segment_count(c)
        if n == 0:
            continue
        p0, p1 = np.asarray(c.start, dtype=float), np.asarray(c.end, dtype=float)
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        pts = p0 + t * (p1 - p0)
        ends.append(np.stack([pts[:-1], pts[1:]], axis=1))
        radii.extend([float(c.wire_radius)] * n)
        names.extend([c.name] * n)
    if not ends:
        return np.zeros((0, 2, 3)), np.zeros(0), []
    return np.concatenate(ends, axis=0), np.asarray(radii), names


def _proximity_findings(world: "World", engine: "Engine | None") -> list[Diagnostic]:
    import numpy as np

    from groundfield.conductors.proximity import find_close_wires

    ends, radii, names = _leakage_wires(world, engine)
    if len(names) < 2:
        return []
    hits = find_close_wires(ends, radii, names, rel_threshold=PROXIMITY_REL_THRESHOLD)
    owner = _foundation_roots(world)
    shared = (
        _shared_vertices(world, owner) if any(k != v for k, v in owner.items()) else {}
    )
    worst: dict[tuple[str, str], Any] = {}
    for h in hits:  # sorted worst first
        root_a = owner.get(h.name_a, h.name_a)
        root_b = owner.get(h.name_b, h.name_b)
        if root_a == root_b:
            # Two bonded wires of one foundation: they meet at its corners
            # and crossings by design (audit D3A-04). A pair whose closest
            # points both lie within the threshold of a vertex the two
            # wires share is that meeting; any other pair (the opposite
            # sides of a narrow ring, the parallel wires of a fine mesh) is
            # real proximity (audit R4A-04).
            pair = tuple(sorted((h.name_a, h.name_b)))
            corners = shared.get(pair)  # type: ignore[arg-type]
            if corners is not None:
                reach = PROXIMITY_REL_THRESHOLD * (radii[h.index_a] + radii[h.index_b])
                da = np.linalg.norm(corners - np.asarray(h.point_a), axis=1)
                db = np.linalg.norm(corners - np.asarray(h.point_b), axis=1)
                if np.any(np.maximum(da, db) <= reach * (1.0 + 1e-9)):
                    continue
            key = pair
        else:
            key = tuple(sorted((root_a, root_b)))
        worst.setdefault(key, h)  # type: ignore[arg-type]
    # The exact line kernel integrates near pairs (finding 10.28) and
    # stays physical for close parallel wires; the point kernel does not.
    point_kernel = engine is None or engine.kernel == "point"
    out = []
    for (a, b), h in worst.items():
        severity: Severity = (
            "warning"
            if point_kernel and h.kind in ("interpenetrating", "close")
            else "info"
        )
        out.append(
            Diagnostic(
                "proximity",
                severity,
                (a, b),
                f"'{a}'–'{b}' {h.kind}, axis distance {h.distance * 1e3:.3g} mm "
                f"(clearance {h.clearance * 1e3:.3g} mm)",
                {
                    "kind": h.kind,
                    "distance_m": h.distance,
                    "clearance_m": h.clearance,
                    "point_a": h.point_a,
                    "point_b": h.point_b,
                },
            )
        )
    return out


def _overlap_stretch(p1a, p2a, p1b, p2b, s0, reach) -> tuple[float, float, float]:
    """Stretch of segment a closer than ``reach`` to segment b.

    The distance from a point moving along a to segment b is convex in its
    position, so the stretch is one interval around the closest point
    ``s0`` (fraction of a); its ends are found by bisection. Returns
    ``(s_lo, s_hi, length_m)``.
    """
    import numpy as np

    from groundfield.conductors.proximity import segment_distance

    def dist(s: float) -> float:
        q = p1a + s * (p2a - p1a)
        d, _, _ = segment_distance(q[None, :], q[None, :], p1b[None, :], p2b[None, :])
        return float(d[0])

    def edge(inside: float, outside: float) -> float:
        if dist(outside) < reach:
            return outside
        for _ in range(60):
            mid = 0.5 * (inside + outside)
            if dist(mid) < reach:
                inside = mid
            else:
                outside = mid
        return inside

    lo, hi = edge(s0, 0.0), edge(s0, 1.0)
    return lo, hi, (hi - lo) * float(np.linalg.norm(p2a - p1a))


def _overlap_findings(world: "World") -> list[Diagnostic]:
    """Inductive conductors lying beside each other closer than touching.

    Since 2026.11 the inductance matrix evaluates such pairs as touching
    wires (their mutual at the distance ``r_a + r_b``), except a thinner
    conductor inside a ``"cable_shield"`` conductor, its concentric core
    (user decision Q32), which is not reported.
    """
    import numpy as np

    from groundfield.conductors.proximity import find_close_conductors

    conds = [
        c
        for c in world.conductors
        if c.inductance_model is not None
        and math.dist(tuple(c.start), tuple(c.end)) > 0.0
    ]
    if len(conds) < 2:
        return []
    by_name = {c.name: c for c in conds}
    # pairs that share an end are kept: a drop drawn back along its main
    # from the tap shares the tap
    hits = find_close_conductors(
        conds, rel_threshold=1.0 - _OVERLAP_TOUCH_TOL, include_junctions=True
    )
    out: list[Diagnostic] = []
    seen: set[tuple[str, str]] = set()
    for h in hits:  # worst first; one finding per pair
        key = tuple(sorted((h.name_a, h.name_b)))
        if key in seen:
            continue
        a, b = by_name[h.name_a], by_name[h.name_b]
        touch = float(a.wire_radius + b.wire_radius)
        if touch <= 0.0:
            continue
        p1a, p2a = np.asarray(a.start, float), np.asarray(a.end, float)
        p1b, p2b = np.asarray(b.start, float), np.asarray(b.end, float)
        la = float(np.linalg.norm(p2a - p1a))
        s0 = float(np.dot(np.asarray(h.point_a) - p1a, p2a - p1a) / (la * la))
        lo, hi, length = _overlap_stretch(
            p1a, p2a, p1b, p2b, s0, touch * (1.0 - _OVERLAP_TOUCH_TOL)
        )
        shorter = min(la, float(np.linalg.norm(p2b - p1b)))
        if length <= max(OVERLAP_REL_LENGTH * touch, OVERLAP_FRACTION * shorter):
            continue  # a branch or a crossing, not wires along each other
        shield, core = (a, b) if a.wire_radius >= b.wire_radius else (b, a)
        if (
            shield.conductor_type == "cable_shield"
            and core.wire_radius < shield.wire_radius
            and h.distance <= shield.wire_radius
        ):
            continue  # a concentric cable (user decision Q32)
        seen.add(key)
        out.append(
            Diagnostic(
                "overlap",
                "warning",
                key,  # type: ignore[arg-type]
                f"'{key[0]}'–'{key[1]}' beside each other over {length:.3g} m, "
                f"axis distance {h.distance * 1e3:.3g} mm (touching "
                f"{touch * 1e3:.3g} mm)",
                {
                    "overlap_m": length,
                    "distance_m": h.distance,
                    "touching_m": touch,
                    "start": tuple(float(v) for v in p1a + lo * (p2a - p1a)),
                    "end": tuple(float(v) for v in p1a + hi * (p2a - p1a)),
                },
            )
        )
    return out


def _interface_contact_findings(world: "World") -> list[Diagnostic]:
    """Rod heads and tips within one wire radius of a layer interface (A3S-05)."""
    from groundfield.solver._guards import _rod_ends_near_interfaces

    if world.soil is None:
        return []
    hits = _rod_ends_near_interfaces(world.electrodes, _layer_interfaces(world.soil))
    return [
        Diagnostic(
            "interface_contact",
            "warning",
            (name,),
            f"'{name}' {which} at z = {z_end:.6g} m, interface at {h:.6g} m "
            f"({(z_end - h) * 1e3:+.3g} mm, wire radius {a * 1e3:.3g} mm)",
            {
                "end": which,
                "z_end_m": z_end,
                "interface_m": h,
                "offset_m": z_end - h,
                "wire_radius_m": a,
            },
        )
        for name, which, z_end, h, a in hits
    ]


def _inert_findings(world: "World") -> list[Diagnostic]:
    from groundfield.soil.models import _EPS_R_DEFAULT

    out: list[Diagnostic] = []
    for c in world.conductors:
        ends = (c.start_electrode, c.end_electrode)
        if all(n is not None for n in ends):
            continue
        out.append(
            Diagnostic(
                "inert",
                "warning",
                (c.name,),
                f"conductor '{c.name}' is not part of the network: "
                f"{'neither end is' if ends == (None, None) else 'one end is not'} "
                "attached to an electrode or junction (start_electrode / "
                "end_electrode), so it carries no current.",
                {"start_electrode": ends[0], "end_electrode": ends[1]},
            )
        )
    if world.concrete_shell_corrections:
        out.append(
            Diagnostic(
                "inert",
                "warning",
                ("concrete_shell_corrections",),
                "World.concrete_shell_corrections (the lumped ADR-0012 V1 "
                f"registry, {len(world.concrete_shell_corrections)} entries) is "
                "deprecated and not read by the solver; concrete shells enter "
                "the solve only as concrete_shell_coefficient_ohm_m on the "
                "electrodes.",
                {"entries": dict(world.concrete_shell_corrections)},
            )
        )
    defaults = {"far_field": "dirichlet", "surface": "neumann", "reference_node": None}
    for key, default in defaults.items():
        value = getattr(world.boundary, key)
        if value != default:
            out.append(
                Diagnostic(
                    "inert",
                    "info",
                    (f"boundary.{key}",),
                    f"boundary.{key} = {value!r} is stored but no backend reads it "
                    f"(every backend implies {default!r}).",
                    {"value": value},
                )
            )
    eps = getattr(world.soil, "relative_permittivity", None)
    if eps is not None and float(eps) != float(_EPS_R_DEFAULT):
        out.append(
            Diagnostic(
                "inert",
                "info",
                ("soil.relative_permittivity",),
                f"soil.relative_permittivity = {float(eps):g} is stored but the "
                "quasi-static formulation neglects displacement currents.",
                {"value": float(eps)},
            )
        )
    return out


def _series_findings(result: "FieldResult") -> list[Diagnostic]:
    md = result.metadata
    if md.get("converged", True) is not False:
        return []
    series = md.get("series") or {}
    backend = md.get("effective_backend", result.backend)
    return [
        Diagnostic(
            "series",
            "warning",
            (str(backend),),
            f"backend '{backend}' reports converged=False (series truncated at "
            f"{md.get('n_terms_used', '?')} of max_terms "
            f"{series.get('max_terms', '?')}, tol {series.get('tol', '?')}): "
            "raise Engine.image_max_terms for high layer contrasts.",
            {
                "n_terms_used": md.get("n_terms_used"),
                "max_terms": series.get("max_terms"),
                "K": md.get("K"),
            },
        )
    ]


def diagnose(
    world: "World",
    engine: "Engine | None" = None,
    *,
    result: "FieldResult | None" = None,
    proximity: bool = True,
) -> DiagnosticsReport:
    """Aggregated diagnostics of a world (review finding 10.70).

    Parameters
    ----------
    world
        World to inspect.
    engine
        Engine of the solve. Needed for the discretisation checks
        (thin wire, resolution, budget against the engine's limit
        ``max_segments`` and its notice size ``warn_segments``); the
        proximity check then runs on the segments
        the solver assembles. Without an engine only the geometry-only
        checks run.
    result
        Solved result; adds the post-solve checks (series truncation).
    proximity
        Run the close / interpenetrating wire check
        (:func:`groundfield.conductors.find_close_wires`) on the leakage
        wires of different owners (default ``True``).

    Returns
    -------
    DiagnosticsReport
        Findings with category, severity, subjects and data; ``str()``
        gives one aggregated line per cause, :meth:`DiagnosticsReport.warn`
        emits them as one :class:`DiagnosticsWarning`.

    Notes
    -----
    Categories:

    ``coordinates``
        The world mixes local and projected coordinates: points lie both
        within 10 km of the origin and beyond 1000 km from it (horizontal
        distance; warning, user decision D5, audit W2B-04). The world is
        solved as given, its two groups some 33 000 km apart, and the solve
        warns with :class:`MixedCoordinatesWarning`. A world that does not
        mix but whose horizontal extent (the diagonal of its bounding box)
        exceeds 100 km is reported as information (two UTM zones, a zone
        prefix on some eastings only, a geometry in millimetres). The data
        name the elements of each group (``near``, ``far``), the conductors
        between them (``spanning``, ``spanning_lengths_m``) and the extent
        (``extent_m``, ``bounds_xy_m``).
    ``thin_wire``
        ``segment_length / wire_radius`` below :data:`MIN_THINWIRE_RATIO`
        (electrodes; ``discretize_segment_length`` for distributed
        conductors).
    ``resolution``
        An electrode feature shorter than ``segment_length``
        (:meth:`~groundfield.geometry.electrodes.RodElectrode.min_feature_length`:
        rod and strip length, ring circumference, shortest polyline
        edge, star arm, mesh span).
    ``budget``
        Predicted segment count or nodal system at or above the engine's
        notice size, ``Engine.warn_segments`` or by default
        :data:`SOFT_LIMIT` (info: the solve may take minutes), or above the
        engine's limit, ``Engine.max_segments`` or by default
        :data:`HARD_LIMIT` (warning: the solve refuses).
        ``report.context["max_segments"]`` and
        ``report.context["warn_segments"]`` hold the two sizes.
    ``proximity``
        Leakage wires of different owners closer than
        :data:`PROXIMITY_REL_THRESHOLD` times the sum of their radii. Two
        wires of one generated foundation (``<name>`` and ``<name>_w<j>``,
        bonded by ``<name>_bond_<k>``) are reported only away from the
        vertices they share: they meet at its corners and crossings by
        design (audit D3A-04), but the opposite sides of a narrow encased
        ring or the parallel wires of a fine mesh are close wires like any
        other (audit R4A-04); findings between foundations name their
        anchors:
        ``interpenetrating`` / ``close`` are warnings with the point
        kernel and information with the exact line kernel (which
        integrates near pairs), ``crossing`` / ``touching`` (wires that
        meet without a bond in the model) are information.
    ``overlap``
        Two conductors with an ``inductance_model`` (insulated or bare) that
        lie beside each other closer than touching — axis distance below
        the sum of their radii by more than 1e-9 of it (round-off of a
        drawing at the touching distance is not an overlap, audit W3B-03) —
        over more than :data:`OVERLAP_REL_LENGTH`
        times that distance and :data:`OVERLAP_FRACTION` of the shorter
        conductor (warning): a service drop drawn on its PEN main, two
        cables on one axis. The inductance matrix evaluates them
        as touching wires (since 2026.11), so their coupling over the
        stretch follows their radii instead of the drawn spacing. A
        thinner conductor inside a ``"cable_shield"`` conductor is its
        concentric core (user decision Q32) and not reported; conductors
        that meet at a point (a crossing, a T-contact, a branch: two drops
        leaving a cabinet 1.7 deg apart run within touching for 0.4 m of
        their 180 m) are not overlaps (thresholds: comment of
        :data:`OVERLAP_FRACTION`). The data give the length of the
        stretch (``overlap_m``), its ends, the smallest axis distance and
        the touching distance.
    ``interface_contact``
        A rod head or tip within one wire radius of a physical layer
        interface, where the thin-wire model (no end face) deviates from
        the solid rod by up to +21 % (audit A3S-05); the solve warns with
        :class:`~groundfield.solver.InterfaceContactWarning` too.
    ``inert``
        Settings the solve ignores: conductors not attached at both ends,
        the deprecated concrete-shell registry, non-default boundary
        conditions, a non-default soil permittivity.
    ``series``
        After the solve: a backend that reports ``converged=False``.
    """
    report = DiagnosticsReport(
        context={
            "min_thinwire_ratio": MIN_THINWIRE_RATIO,
            "proximity_rel_threshold": PROXIMITY_REL_THRESHOLD,
            "overlap_rel_length": OVERLAP_REL_LENGTH,
            "overlap_fraction": OVERLAP_FRACTION,
        }
    )
    report.findings.extend(_coordinate_findings(world))
    if engine is not None:
        report.context["segment_length_m"] = _check_ds(engine)
        report.context["max_segments"] = _hard_limit(engine)
        report.context["warn_segments"] = _soft_limit(engine)
        report.findings.extend(_resolution_findings(world, engine))
    if proximity and (engine is None or check_budget_ok(world, engine)):
        report.findings.extend(_proximity_findings(world, engine))
    report.findings.extend(_overlap_findings(world))
    report.findings.extend(_interface_contact_findings(world))
    report.findings.extend(_inert_findings(world))
    if result is not None:
        report.findings.extend(_series_findings(result))
    return report


def check_budget_ok(world: "World", engine: "Engine") -> bool:
    """Whether the world is within the engine's limit (no exception).

    The limit is ``engine.max_segments``, or :data:`HARD_LIMIT` if the
    engine leaves it at ``None`` (see :func:`check_segment_budget`).
    """
    try:
        check_segment_budget(world, engine, _layer_interfaces(world.soil))
    except ResourceLimitError:
        return False
    return True


# ---------------------------------------------------------------------
# check_segment_resolution
# ---------------------------------------------------------------------


def check_segment_resolution(world: "World", engine: "Engine") -> list[str]:
    """Heuristic discretisation-quality check, one string per cause.

    Inspects the world / engine pair for common modelling pitfalls and
    returns one human-readable string per cause (review finding 10.70:
    up to 0.15.0 one string per electrode, so a layout with 81 joint
    rods emitted 81 warnings per solve). The empty list means *no
    concerns detected*. Causes:

    1. **Thin-wire ratio** (``"thin-wire: ..."``). Each electrode must
       satisfy ``segment_length >= MIN_THINWIRE_RATIO * wire_radius``
       so that the thin-wire average-potential self-action remains
       valid; a distributed conductor's ``discretize_segment_length``
       likewise.
    2. **Electrode feature shorter than one segment**
       (``"resolution: ..."``), using each electrode's
       :meth:`min_feature_length` — every kind, including polyline and
       star electrodes (F53 / F54, listed as fixed in 0.15.0 but
       skipped until 2026.11).
    3. **Total segment budget** (``"budget: ..."``): the segment count or
       the size of the dense nodal system (every conductor sub-piece
       included) at or above the engine's notice size
       ``Engine.warn_segments`` (default :data:`SOFT_LIMIT`), or above the
       engine's limit ``Engine.max_segments`` (default :data:`HARD_LIMIT`;
       the solve then refuses with :class:`ResourceLimitError`).

    The function never raises (except for a non-positive
    ``segment_length``) — it only reports. The full report with
    proximity and inert-setting checks is :func:`diagnose`.

    Parameters
    ----------
    world
        World to inspect.
    engine
        Engine carrying ``segment_length``, ``max_segments`` and
        ``warn_segments``.

    Returns
    -------
    list[str]
        Zero to three diagnostic strings, in the order thin-wire,
        resolution, budget; each names up to five electrodes or
        conductors and the number of further ones.
    """
    report = DiagnosticsReport(
        findings=_resolution_findings(world, engine),
        context={
            "min_thinwire_ratio": MIN_THINWIRE_RATIO,
            "segment_length_m": _check_ds(engine),
        },
    )
    return report.lines()
