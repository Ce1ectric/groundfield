"""Pre-solve world diagnostics — counts, mesh budget, resolution, proximity.

This module collects **structural** diagnostics that work directly
on a :class:`~groundfield.world.World` (and optionally an
:class:`~groundfield.solver.engine.Engine` and a solved
:class:`~groundfield.solver.result.FieldResult`). It is the typical
counterpart to :mod:`groundfield.validation` (a *post-solve*
cross-engine check):

* :func:`diagnose` — one structured, aggregated report
  (:class:`DiagnosticsReport`) of everything that limits the accuracy
  or the meaning of a solve: discretisation quality, segment budget,
  close or interpenetrating wires, inert configuration and, after the
  solve, series truncation. One line per cause, not per electrode
  (review finding 10.70).
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
``FieldResult.metadata["max_segments"]`` records it. :data:`SOFT_LIMIT`
(the size above which a solve takes minutes on a laptop, reported as
information) stays a module constant.

References
----------
- ADR-0003 (`docs/adr/0003-distributed-conductor-model.md`) —
  distributed-conductor topology used by :func:`expected_segments`
  for conductors with finite ``discretize_segment_length``.
"""

from __future__ import annotations

import math
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
    "ResourceLimitError",
    "MIN_THINWIRE_RATIO",
    "SOFT_LIMIT",
    "HARD_LIMIT",
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


Severity = Literal["info", "warning"]


@dataclass(frozen=True)
class Diagnostic:
    """One finding of :func:`diagnose`.

    Attributes
    ----------
    category
        Cause: ``"thin_wire"``, ``"resolution"``, ``"budget"``,
        ``"proximity"``, ``"interface_contact"``, ``"inert"`` or
        ``"series"``.
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


def _nodal_unknowns(world: "World", layer_interfaces=None) -> int:
    """Size of the dense augmented nodal system of the solver (upper bound).

    The image and ``mom`` families solve one dense system with one unknown
    per current owner (an electrode, its part in every further soil
    layer, every leaking conductor sub-piece), per node (one per
    electrode cluster, junction and interior conductor node) and per
    branch (lumped conductors and every chain sub-piece). Its matrix has
    $n^2$ entries and is allocated whole, so a long conductor chain of
    fine sub-pieces dominates the memory long before the leakage
    segments do (audit round 2, A2S-02: a 100 m PEN in 2000 isolated
    pieces, 12 segments and about 4000 unknowns, took 407 MiB).

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
        Owners plus nodes plus branches; an electrode counts as one
        owner and one node (every electrode its own cluster).
    """
    interfaces = tuple(layer_interfaces or ())
    n = len(world.junctions)
    for e in world.electrodes:
        n += 2
        if interfaces:
            lower, upper = e.bounds()
            n += sum(1 for h in interfaces if lower[2] < h < upper[2])
    n += sum(_conductor_unknowns(c) for c in world.conductors)
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
    elif max(n_total, n_nodal) >= SOFT_LIMIT:
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
                f"{what} exceeds {SOFT_LIMIT} — "
                "solve time may run from seconds to minutes on a typical laptop.",
                {"n_segments": n_total, "n_unknowns": n_nodal, "limit": SOFT_LIMIT},
            )
        )
    return out


def _leakage_wires(world: "World", engine: "Engine | None"):
    """End points, radii and owner names of every leakage wire segment."""
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
    from groundfield.conductors.proximity import find_close_wires

    ends, radii, names = _leakage_wires(world, engine)
    if len(names) < 2:
        return []
    hits = find_close_wires(ends, radii, names, rel_threshold=PROXIMITY_REL_THRESHOLD)
    worst: dict[tuple[str, str], Any] = {}
    for h in hits:  # sorted worst first
        key = tuple(sorted((h.name_a, h.name_b)))
        worst.setdefault(key, h)
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
        ``max_segments``); the proximity check then runs on the segments
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
        Predicted segment count or nodal system at or above
        :data:`SOFT_LIMIT` (info) or above the engine's limit,
        ``Engine.max_segments`` or by default :data:`HARD_LIMIT` (warning:
        the solve refuses). ``report.context["max_segments"]`` holds the
        limit.
    ``proximity``
        Leakage wires of different owners closer than
        :data:`PROXIMITY_REL_THRESHOLD` times the sum of their radii:
        ``interpenetrating`` / ``close`` are warnings with the point
        kernel and information with the exact line kernel (which
        integrates near pairs), ``crossing`` / ``touching`` (wires that
        meet without a bond in the model) are information.
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
        }
    )
    if engine is not None:
        report.context["segment_length_m"] = _check_ds(engine)
        report.context["max_segments"] = _hard_limit(engine)
        report.findings.extend(_resolution_findings(world, engine))
    if proximity and (engine is None or check_budget_ok(world, engine)):
        report.findings.extend(_proximity_findings(world, engine))
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
       included) at or above :data:`SOFT_LIMIT`, or above the engine's
       limit ``Engine.max_segments`` (default :data:`HARD_LIMIT`; the
       solve then refuses with :class:`ResourceLimitError`).

    The function never raises (except for a non-positive
    ``segment_length``) — it only reports. The full report with
    proximity and inert-setting checks is :func:`diagnose`.

    Parameters
    ----------
    world
        World to inspect.
    engine
        Engine carrying ``segment_length`` and ``max_segments``.

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
