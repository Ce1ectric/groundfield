"""LV network layout builder: real footprints, routed PEN cables, one plan.

The package splits the layout builder by concern:

* :mod:`~groundfield.generators.layout.model` — the layout snapshot, its
  construction (from footprints or OpenStreetMap) and the export to a
  :class:`~groundfield.generators.plan.NetworkPlan` / world;
* :mod:`~groundfield.generators.layout.routing` — Manhattan routing of
  PEN cables and service drops;
* :mod:`~groundfield.generators.layout.measurement` — the simulated
  fall-of-potential measurement (auxiliary electrode, voltage probe);
* :mod:`~groundfield.generators.layout.penetration` — reproducible
  foundation-electrode penetration masks;
* :mod:`~groundfield.generators.layout.plot` — plan view and surface
  potential plots.
"""

from __future__ import annotations

from groundfield.generators.layout.measurement import (
    AuxiliaryElectrodePlacement,
    VoltageProbePlacement,
    triangle_rod_grounding,
)
from groundfield.generators.layout.model import (
    CabinetPlacement,
    KvsPlacement,
    LvNetworkLayout,
    OrtsnetzLayout,
)
from groundfield.generators.layout.routing import (
    BuildingConnection,
    LateralCoverageResult,
    PenCable,
)

__all__ = [
    "AuxiliaryElectrodePlacement",
    "BuildingConnection",
    "CabinetPlacement",
    "LateralCoverageResult",
    "LvNetworkLayout",
    "PenCable",
    "VoltageProbePlacement",
    "triangle_rod_grounding",
    # Deprecated aliases (removal no earlier than 2027.5)
    "KvsPlacement",
    "OrtsnetzLayout",
]
