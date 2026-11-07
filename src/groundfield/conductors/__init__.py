"""Conductors, PEN, and cable shields.

Subpackage for all current-carrying elements above and inside the soil
that are not electrodes themselves but still take part in the current
distribution:

- overhead lines (phase conductor, earth wire),
- cables (phase + shield, concentric or three-core),
- PEN conductors in TN low-voltage networks,
- auxiliary electrodes for grounding measurements.

Contents
--------
Conductor
    Single conductor segment with end-point coordinates and a type tag
    (``"pen"``, ``"cable_shield"``, ``"bare_copper"``, ``"overhead"``,
    ``"generic"``), its series resistance and — with
    ``resistance_model="skin"`` — its frequency-dependent internal
    impedance.
ConductorType, ResistanceModel
    Literal unions of the supported type tags and resistance models.
CrossSectionUnitWarning
    Raised for cross sections that look like mm² entered as m².
solid_round_internal_impedance
    Exact internal impedance per unit length of a solid round wire
    (skin effect).
find_close_wires, find_close_conductors, WireProximity
    Detection of close, interpenetrating and crossing wire pairs.

Notes
-----
Carson corrections for the earth-return path are not computed here;
they enter via :mod:`groundfield.coupling`. This keeps the conductor
model purely geometric and material-based, so it can be combined with
different earth-return models.
"""

from __future__ import annotations

from groundfield.conductors.conductor import (
    MAX_PLAUSIBLE_CROSS_SECTION,
    Conductor,
    ConductorType,
    CrossSectionUnitWarning,
    ResistanceModel,
)
from groundfield.conductors.proximity import (
    WireProximity,
    find_close_conductors,
    find_close_wires,
    segment_distance,
)
from groundfield.conductors.skin import solid_round_internal_impedance

__all__ = [
    "Conductor",
    "ConductorType",
    "ResistanceModel",
    "CrossSectionUnitWarning",
    "MAX_PLAUSIBLE_CROSS_SECTION",
    "solid_round_internal_impedance",
    "WireProximity",
    "find_close_wires",
    "find_close_conductors",
    "segment_distance",
]
