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
    ``"generic"``).
ConductorType
    Literal union of the supported type tags.

Notes
-----
Carson corrections for the earth-return path are not computed here;
they enter via :mod:`groundfield.coupling`. This keeps the conductor
model purely geometric and material-based, so it can be combined with
different earth-return models.
"""

from __future__ import annotations

from groundfield.conductors.conductor import Conductor, ConductorType

__all__ = ["Conductor", "ConductorType"]
