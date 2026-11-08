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

Two front-ends compose these layers into a TN low-voltage network:
:class:`~groundfield.generators.tn_network.TnNetworkGenerator`
(stochastic population parameters) and
:class:`~groundfield.generators.layout.LvNetworkLayout` (real footprints
and routed PEN cables). Both produce a pure-data
:class:`~groundfield.generators.plan.NetworkPlan`; one materialiser
(:func:`~groundfield.generators.materialise.materialise_plan`) turns
every plan into a :class:`groundfield.World`.

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
    MEASUREMENT_H_TERMINAL,
    MEASUREMENT_S_TERMINAL,
    MeasurementInjectionConfig,
    MeasurementLeadConfig,
    MeasurementProbeConfig,
    MeasurementSetupConfig,
    buried_lead,
    measured_impedance,
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
from groundfield.generators.manhattan_routing import (
    ObstacleBox,
    merge_collinear_waypoints,
    route_manhattan,
    segment_intersects_box,
)
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
from groundfield.generators.materialise import materialise_plan
from groundfield.generators.pen_topology import (
    PenTopology,
    RadialTrunkTopology,
    StarCabinetTopology,
    StarKvsTopology,
)
from groundfield.generators.plan import (
    ConductorParameters,
    NetworkPlan,
    PlannedBranch,
    PlannedElectrode,
    PlannedJunction,
    PlannedMeasurement,
    PlannedSite,
    PlannedSource,
)
from groundfield.generators.tn_network import (
    CabinetConfig,
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
    "measured_impedance",
    "MEASUREMENT_H_TERMINAL",
    "MEASUREMENT_S_TERMINAL",
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
    "CabinetConfig",
    "PenConfig",
    # PEN topologies
    "PenTopology",
    "StarCabinetTopology",
    "RadialTrunkTopology",
    # Manhattan routing + LV network layout builder
    "ObstacleBox",
    "route_manhattan",
    "segment_intersects_box",
    "merge_collinear_waypoints",
    "LvNetworkLayout",
    "PenCable",
    "CabinetPlacement",
    "AuxiliaryElectrodePlacement",
    "VoltageProbePlacement",
    "BuildingConnection",
    "LateralCoverageResult",
    "triangle_rod_grounding",
    # Network plan + materialiser
    "NetworkPlan",
    "PlannedSite",
    "PlannedElectrode",
    "PlannedJunction",
    "PlannedBranch",
    "PlannedMeasurement",
    "PlannedSource",
    "ConductorParameters",
    "materialise_plan",
    # Deprecated aliases (removal no earlier than 2027.5)
    "KvsConfig",
    "StarKvsTopology",
    "OrtsnetzLayout",
    "KvsPlacement",
]
