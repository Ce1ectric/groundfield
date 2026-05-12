"""Pre-solve world diagnostics — counts, mesh budget, resolution checks.

This module collects **structural** diagnostics that work directly
on a :class:`~groundfield.world.World` (and optionally an
:class:`~groundfield.solver.engine.Engine`), without invoking the
solver. It is the AP1 counterpart to
:mod:`groundfield.validation` (which is a *post-solve* cross-engine
check):

* :func:`world_statistics` — aggregate counts and lengths,
  bounding box, footprint area, conductor-length statistics.
* :func:`expected_segments` — predicts the number of point-source
  segments that the image-family discretiser will produce per
  electrode kind, plus a total. Useful for budgeting wall-clock
  cost before kicking off a 200-EFH run.
* :func:`check_segment_resolution` — heuristic warnings about
  the discretisation quality (thin-wire ratio, electrode size
  vs. segment length, segment-count budget). Returns a list of
  human-readable strings, empty when everything looks healthy.

Validity envelope
-----------------
* The segment counts in :func:`expected_segments` mirror the
  conventions of :mod:`groundfield.solver.image` (the image-family
  discretiser used by the ``image``, ``image_2layer``,
  ``image_nlayer``, ``mom``, ``mom_sommerfeld``, ``cim`` and
  ``bem`` backends). FEM does not use this discretiser; for FEM
  the segment count is not predictive of cost.
* Wire-length / segment computations follow the *physical*
  geometry. Sub-segment overheads from cluster-bonding, KCL
  pseudo-nodes and lumped-conductor branches are not
  double-counted.

References
----------
- ADR-0003 (`docs/adr/0003-distributed-conductor-model.md`) —
  distributed-conductor topology used by :func:`expected_segments`
  for conductors with finite ``discretize_segment_length``.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import median
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

__all__ = [
    "world_statistics",
    "expected_segments",
    "check_segment_resolution",
]


# ---------------------------------------------------------------------
# Wire-length and segment-count primitives
# ---------------------------------------------------------------------


def _electrode_wire_length(e) -> float:
    """Total wire length of an electrode in metres.

    Sums the geometric length of every wire that makes up the
    electrode. For mesh / grid-mesh electrodes this includes both
    longitudinal and transverse wires.
    """
    from groundfield.geometry.electrodes import (
        GridMeshElectrode,
        MeshElectrode,
        RingElectrode,
        RodElectrode,
        StripElectrode,
    )

    if isinstance(e, RodElectrode):
        return float(e.length)
    if isinstance(e, RingElectrode):
        return float(2.0 * math.pi * e.radius)
    if isinstance(e, StripElectrode):
        return float(e.length)
    if isinstance(e, MeshElectrode):
        dx, dy = e.size
        # Match the discretiser convention exactly: nx = number of
        # x-positioned (parallel-to-y) wires, ny = number of
        # y-positioned (parallel-to-x) wires.
        nx_pos = max(2, int(round(dx / e.spacing)) + 1)
        ny_pos = max(2, int(round(dy / e.spacing)) + 1)
        return float(ny_pos * dx + nx_pos * dy)
    if isinstance(e, GridMeshElectrode):
        dx, dy = e.size
        return float((e.n_y + 1) * dx + (e.n_x + 1) * dy)
    return 0.0  # pragma: no cover - defensive


def _electrode_segment_count(e, ds: float) -> int:
    """Predicted number of point-source segments for an electrode.

    Mirrors :mod:`groundfield.solver.image`'s discretiser exactly:

    - rod: ``max(1, ceil(length / ds))``,
    - ring: ``max(8, ceil(2 pi r / ds))``,
    - strip: ``max(1, ceil(length / ds))``,
    - mesh / grid_mesh: per-wire ``max(1, ceil(d_axis / ds))`` summed
      over both wire directions.
    """
    from groundfield.geometry.electrodes import (
        GridMeshElectrode,
        MeshElectrode,
        RingElectrode,
        RodElectrode,
        StripElectrode,
    )

    if isinstance(e, RodElectrode):
        return max(1, int(math.ceil(e.length / ds)))
    if isinstance(e, RingElectrode):
        return max(8, int(math.ceil(2.0 * math.pi * e.radius / ds)))
    if isinstance(e, StripElectrode):
        return max(1, int(math.ceil(e.length / ds)))
    if isinstance(e, (MeshElectrode, GridMeshElectrode)):
        dx, dy = e.size
        if isinstance(e, GridMeshElectrode):
            nx_pos = e.n_x + 1
            ny_pos = e.n_y + 1
        else:
            nx_pos = max(2, int(round(dx / e.spacing)) + 1)
            ny_pos = max(2, int(round(dy / e.spacing)) + 1)
        seg_along_x = max(1, int(math.ceil(dx / ds)))
        seg_along_y = max(1, int(math.ceil(dy / ds)))
        return ny_pos * seg_along_x + nx_pos * seg_along_y
    return 0  # pragma: no cover - defensive


def _conductor_segment_count(c) -> int:
    """Point-source segments contributed by a conductor.

    Lumped conductors (``discretize_segment_length is None``)
    contribute zero point-source segments — they are pure
    nodal-analysis branches.

    Distributed conductors contribute one midpoint per sub-piece
    (``conductor.n_segments``) **only** when
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


# ---------------------------------------------------------------------
# world_statistics
# ---------------------------------------------------------------------


def world_statistics(world: "World") -> dict[str, Any]:
    """Return a structural snapshot of a world.

    Aggregates counts, lengths and the bounding box; complements
    :meth:`World.summary` (one-line text) with a richer
    machine-readable dictionary that scales to AP1-grade networks.

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
          / strip / mesh / grid_mesh),
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
    from groundfield.geometry.electrodes import (
        GridMeshElectrode,
        MeshElectrode,
        RingElectrode,
        RodElectrode,
        StripElectrode,
    )
    from groundfield.postprocess.geometry_plot import world_bounds_3d
    from groundfield.soil.models import MultiLayerSoil, TwoLayerSoil

    kind_map: dict[type, str] = {
        RodElectrode: "rod",
        RingElectrode: "ring",
        StripElectrode: "strip",
        MeshElectrode: "mesh",
        GridMeshElectrode: "grid_mesh",
    }
    by_kind: Counter[str] = Counter()
    total_e_wire = 0.0
    for e in world.electrodes:
        by_kind[kind_map.get(type(e), "unknown")] += 1
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
# expected_segments
# ---------------------------------------------------------------------


def expected_segments(world: "World", engine: "Engine") -> dict[str, Any]:
    """Predict the number of point-source segments after discretisation.

    Mirrors the image-family discretiser
    (:mod:`groundfield.solver.image`) so the prediction is
    *exact* for the ``image`` / ``image_2layer`` / ``image_nlayer``
    / ``mom`` / ``mom_sommerfeld`` / ``cim`` / ``bem`` backends.
    The FEM backend (:mod:`groundfield.solver.fem`) uses an
    axisymmetric volume mesh that is unrelated to ``segment_length``;
    for FEM this prediction is not informative.

    Parameters
    ----------
    world
        World to inspect.
    engine
        Engine carrying ``segment_length`` (in metres).

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
        * ``total`` (int) — sum of electrode and conductor segments.

    Raises
    ------
    ValueError
        If ``engine.segment_length`` is not strictly positive.
    """
    ds = float(engine.segment_length)
    if not math.isfinite(ds) or ds <= 0.0:
        raise ValueError(
            f"engine.segment_length must be > 0, got {engine.segment_length!r}."
        )

    per_electrode: dict[str, int] = {}
    per_kind: Counter[str] = Counter()
    e_total = 0
    for e in world.electrodes:
        n = _electrode_segment_count(e, ds)
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
    }


# ---------------------------------------------------------------------
# check_segment_resolution
# ---------------------------------------------------------------------

# Heuristic thresholds. Tuned for the AP1 envelope (image-family
# backends, 50 Hz, ρ in [50, 1000] Ω·m, electrode dimensions
# ~ 0.5 ... 30 m).
_MIN_THINWIRE_RATIO = 5.0
"""Minimum recommended ratio of segment_length to wire_radius.

Below this, the thin-wire approximation that underpins the
average-potential method becomes increasingly biased: the
self-action integral is computed under the assumption
``segment_length >> wire_radius``.
"""

_BUDGET_WARN_THRESHOLD = 5_000
"""Total segment count above which the user should be aware of cost.

The dense Z-matrix scales as :math:`O(N^2)` in memory and the
solve as :math:`O(N^3)` for the LU + multi-port factorisation, so
~ 5 000 segments is roughly where solve time becomes minutes
rather than seconds on a typical laptop.
"""

_BUDGET_HARD_THRESHOLD = 20_000
"""Total segment count above which the budget warning becomes urgent."""


def check_segment_resolution(world: "World", engine: "Engine") -> list[str]:
    """Heuristic discretisation-quality check.

    Inspects the world / engine pair for common AP1 modelling
    pitfalls and returns one human-readable string per finding.
    The empty list means *no concerns detected*. Categories:

    1. **Thin-wire ratio.** Each electrode must satisfy
       ``segment_length >= 5 * wire_radius`` so that the
       thin-wire average-potential self-action remains valid.
    2. **Electrode smaller than one segment.** An electrode whose
       *smallest geometric dimension* is below ``segment_length``
       is either degenerately discretised (1 segment) or, for
       rings, falls back to the discretiser's ``max(8, ...)`` floor
       — in either case the user probably wants a finer
       ``segment_length``.
    3. **Distributed-conductor ratio.** A finite
       ``discretize_segment_length`` must also stay above
       ``5 * conductor.wire_radius``.
    4. **Total segment budget.** Warns when the predicted total
       crosses :data:`_BUDGET_WARN_THRESHOLD` and again at
       :data:`_BUDGET_HARD_THRESHOLD`.

    The function never raises — it only reports. Use the returned
    list to inform the user before kicking off a long solve.

    Parameters
    ----------
    world
        World to inspect.
    engine
        Engine carrying ``segment_length``.

    Returns
    -------
    list[str]
        Zero or more diagnostic strings. The list is sorted by
        severity (budget warnings last, electrode-specific
        warnings first).
    """
    from groundfield.geometry.electrodes import (
        GridMeshElectrode,
        MeshElectrode,
        RingElectrode,
        RodElectrode,
        StripElectrode,
    )

    ds = float(engine.segment_length)
    msgs: list[str] = []

    # 1 + 2: per-electrode checks
    for e in world.electrodes:
        ratio = ds / e.wire_radius
        if ratio < _MIN_THINWIRE_RATIO:
            msgs.append(
                f"thin-wire: electrode '{e.name}' ({e.kind}) has "
                f"segment_length / wire_radius = {ratio:.1f} "
                f"(< {_MIN_THINWIRE_RATIO:.0f}); "
                f"the thin-wire self-action assumption may bias the result."
            )

        # Smallest characteristic geometric dimension.
        if isinstance(e, RodElectrode):
            char = e.length
        elif isinstance(e, RingElectrode):
            char = 2.0 * math.pi * e.radius
        elif isinstance(e, StripElectrode):
            char = e.length
        elif isinstance(e, GridMeshElectrode):
            char = min(e.size[0] / max(1, e.n_x), e.size[1] / max(1, e.n_y))
        elif isinstance(e, MeshElectrode):
            char = min(e.size[0], e.size[1], e.spacing)
        else:  # pragma: no cover - defensive
            char = float("inf")

        if char < ds:
            msgs.append(
                f"resolution: electrode '{e.name}' ({e.kind}) has its "
                f"smallest dimension {char:.3f} m below the engine's "
                f"segment_length {ds:.3f} m — the discretiser will fall "
                f"back to its lower floor and the result will be coarse."
            )

    # 3: distributed-conductor wire / segment ratio
    for c in world.conductors:
        if not c.is_distributed:
            continue
        ratio = float(c.discretize_segment_length) / c.wire_radius
        if ratio < _MIN_THINWIRE_RATIO:
            msgs.append(
                f"thin-wire: distributed conductor '{c.name}' has "
                f"discretize_segment_length / wire_radius = {ratio:.1f} "
                f"(< {_MIN_THINWIRE_RATIO:.0f}); the longitudinal "
                f"self-inductance may be biased."
            )

    # 4: total segment-count budget
    est = expected_segments(world, engine)
    n_total = est["total"]
    if n_total >= _BUDGET_HARD_THRESHOLD:
        msgs.append(
            f"budget: predicted total segment count {n_total} exceeds "
            f"{_BUDGET_HARD_THRESHOLD} — expect a heavy memory and "
            f"solve-time cost (dense system: O(N²) memory, O(N³) solve). "
            f"Consider using a coarser segment_length or a sparser "
            f"distributed-conductor discretisation."
        )
    elif n_total >= _BUDGET_WARN_THRESHOLD:
        msgs.append(
            f"budget: predicted total segment count {n_total} exceeds "
            f"{_BUDGET_WARN_THRESHOLD} — solve time may run from "
            f"seconds to minutes on a typical laptop."
        )

    return msgs
