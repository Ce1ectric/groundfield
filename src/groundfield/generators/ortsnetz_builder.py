"""Deprecated module path of the LV network layout builder.

The builder moved to the :mod:`groundfield.generators.layout` package
(model, routing, measurement, penetration, plot) and ``OrtsnetzLayout``
was renamed to :class:`~groundfield.generators.layout.LvNetworkLayout`.
Importing this module emits a :class:`DeprecationWarning`; it re-exports
the former names (removal no earlier than groundfield 2027.5).
"""

from __future__ import annotations

from groundfield.generators._compat import warn_renamed
from groundfield.generators.layout import (
    AuxiliaryElectrodePlacement,
    BuildingConnection,
    CabinetPlacement,
    KvsPlacement,
    LateralCoverageResult,
    LvNetworkLayout,
    OrtsnetzLayout,
    PenCable,
    VoltageProbePlacement,
    triangle_rod_grounding,
)

warn_renamed(
    "groundfield.generators.ortsnetz_builder",
    "groundfield.generators.layout",
    stacklevel=2,
)

__all__ = [
    "AuxiliaryElectrodePlacement",
    "BuildingConnection",
    "CabinetPlacement",
    "KvsPlacement",
    "LateralCoverageResult",
    "LvNetworkLayout",
    "OrtsnetzLayout",
    "PenCable",
    "VoltageProbePlacement",
    "triangle_rod_grounding",
]
