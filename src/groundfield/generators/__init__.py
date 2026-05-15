"""``World`` generators for reference networks.

This subpackage hosts the **factory layer** that turns high-level
parameter sets — for example "N single-family houses on a two-layer
soil" — into a fully populated :class:`groundfield.World` that the
solver can consume.

The framework is organised in five composable spec layers:

* :mod:`~groundfield.generators.distributions` — Constant, Uniform,
  Normal, LogNormal, Weibull, Discrete, Categorical. Each numerical
  / categorical field of any spec class accepts a fixed value or a
  :class:`Distribution`.
* :mod:`~groundfield.generators.electrode_specs` — single-electrode
  specs (rod / ring / strip / foundation) with ``presence_prob`` and
  ``offset_xy_m``.
* :mod:`~groundfield.generators.grounding` — :class:`GroundingSystemSpec`
  composes a list of :data:`ElectrodeSpec` into one cluster.
* :mod:`~groundfield.generators.placement` — Manhattan-grid and
  explicit-coordinate placement strategies.
* :mod:`~groundfield.generators.soil_specs` — homogeneous /
  two-layer / multi-layer soil specs that resolve to the
  :mod:`groundfield.soil.models` types.
* :mod:`~groundfield.generators.building` — :class:`BuildingTypeSpec`
  bundles a name with a grounding system and an optional plot size.

The first concrete generator,
:class:`~groundfield.generators.tn_network.TnNetworkGenerator`,
composes all of the above into a TN low-voltage network.

See ``docs/adr/0009-world-generators.md`` for the design rationale
and the validation programme.
"""

from __future__ import annotations

from groundfield.generators.base import (
    GeneratorConfig,
    WorldGenerator,
    resolve_value,
)
from groundfield.generators.building import (
    BuildingTypeSpec,
    default_building_catalog,
)
from groundfield.generators.distributions import (
    AnyDistribution,
    Categorical,
    Constant,
    Discrete,
    Distribution,
    LogNormal,
    Normal,
    Uniform,
    Weibull,
)
from groundfield.generators.electrode_specs import (
    ElectrodeSpec,
    FoundationElectrodeSpec,
    RingElectrodeSpec,
    RodElectrodeSpec,
    StripElectrodeSpec,
    rod_circle,
)
from groundfield.generators.grounding import GroundingSystemSpec
from groundfield.generators.measurement import (
    MeasurementInjectionConfig,
    MeasurementLeadConfig,
    MeasurementProbeConfig,
    MeasurementSetupConfig,
    buried_lead,
    neighbour_substation_grounding,
    overhead_lead,
    single_rod_grounding,
)
from groundfield.generators.placement import (
    ExplicitPlacement,
    ManhattanGridPlacement,
    OsmBuildingPlacement,
    PlacementSpec,
)
from groundfield.generators.soil_specs import (
    HomogeneousSoilSpec,
    MultiLayerSoilSpec,
    SoilLayerSpec,
    SoilSpec,
    TwoLayerSoilSpec,
    materialise_soil,
)
from groundfield.generators.tn_network import (
    KvsConfig,
    PenConfig,
    SubstationConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
)

__all__ = [
    # Base
    "GeneratorConfig",
    "WorldGenerator",
    "resolve_value",
    # Distributions
    "Distribution",
    "Constant",
    "Uniform",
    "Normal",
    "LogNormal",
    "Weibull",
    "Discrete",
    "Categorical",
    "AnyDistribution",
    # Electrode specs
    "RodElectrodeSpec",
    "RingElectrodeSpec",
    "StripElectrodeSpec",
    "FoundationElectrodeSpec",
    "ElectrodeSpec",
    "rod_circle",
    # Grounding system
    "GroundingSystemSpec",
    # Measurement setup
    "MeasurementSetupConfig",
    "MeasurementInjectionConfig",
    "MeasurementProbeConfig",
    "MeasurementLeadConfig",
    "overhead_lead",
    "buried_lead",
    "single_rod_grounding",
    "neighbour_substation_grounding",
    # Placement
    "ManhattanGridPlacement",
    "ExplicitPlacement",
    "OsmBuildingPlacement",
    "PlacementSpec",
    # Soil specs
    "HomogeneousSoilSpec",
    "TwoLayerSoilSpec",
    "MultiLayerSoilSpec",
    "SoilLayerSpec",
    "SoilSpec",
    "materialise_soil",
    # Building type
    "BuildingTypeSpec",
    "default_building_catalog",
    # TN-network generator
    "TnNetworkConfig",
    "TnNetworkGenerator",
    "SubstationConfig",
    "KvsConfig",
    "PenConfig",
]
