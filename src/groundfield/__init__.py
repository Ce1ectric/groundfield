"""groundfield — numerical field computation for grounding systems.

``groundfield`` is an open-source Python package for the physical
reference modelling of networked grounding systems. Within the
``groundmeas`` / ``groundinsight`` / ``groundfield`` software family it
covers the field-theoretical side: soil models, electrode geometries,
conductors and their couplings are formulated as a 3-D problem in the
soil and solved numerically. The results are field profiles, potential
curves, current distributions, and reduced equivalent models
(``rho-f``) for further use in ``groundinsight``.

The package is deliberately designed to support reference
computations for networked grounding systems such as TN distribution
networks with a substation, house connections, cable cabinets, and
layered soil.

Namespaces
----------
The data model, the factories (``create_*``) and the engine are
top-level names. The post-processing API is also grouped in curated
namespaces; every function stays available under its flat top-level
name as well:

``gf.plot``
    Potential maps and profiles, world geometry, current sharing,
    sweeps and convergence plots.
``gf.safety``
    Touch and step voltages, EN 50522 permissible touch voltage.
``gf.fit``
    Reduced rho-f models (vector fitting, research standard form).
``gf.io``
    Export to groundinsight (BusType JSON), CSV and VTK.

``import groundfield`` loads only the data model, the factories and the
engine; post-processing, IO, generators, the geo ingest, references,
cross-engine validation and diagnostics are imported on first use
(PEP 562), so pandas, matplotlib, sympy and most of scipy stay unloaded
until a function needs them.

Subpackages
-----------
soil
    Soil models (homogeneous, two-layer, multi-layer).
geometry
    Electrode geometries.
conductors
    Conductors, PEN, cable shields.
solver
    Numerical field solver (image method, MoM, …) in the frequency
    domain.
coupling
    Inductive and galvanic coupling between conductors and soil.
postprocess
    Potentials, touch and step voltages, current sharing, sweeps, fits,
    plots.
io
    File I/O and exchange formats, in particular the export of reduced
    ``rho-f`` models for ``groundinsight``.
generators, geo
    Synthetic and OpenStreetMap-based world generators.
utils
    Logging, name suggestions, warnings of the public API.

Examples
--------
>>> import groundfield as gf
>>> soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
>>> world = gf.create_world(soil=soil)
>>> gf.create_electrode(world, "rod", name="g1",
...                     position=(0.0, 0.0, 0.0), length=1.5)
>>> gf.create_source(world, attached_to="g1", magnitude=1.0)
>>> result = gf.create_engine(backend="image", segment_length=0.05).solve(world)
>>> result.cluster_impedance("g1")[0].real  # doctest: +SKIP
"""

import importlib as _importlib
import sys as _sys
import typing as _typing

# Version
__version__ = "0.15.0"

# ---------------------------------------------------------------------
# Eager imports: data model, engine, factories (light; no pandas,
# matplotlib or sympy)
# ---------------------------------------------------------------------
from groundfield.boundary import BoundaryConditions
from groundfield.conductors.conductor import Conductor, ConductorType
from groundfield.geometry.electrodes import (
    Electrode,
    GridMeshElectrode,
    MeshElectrode,
    PolylineElectrode,
    RingElectrode,
    RodElectrode,
    StarElectrode,
    StripElectrode,
)
from groundfield.geometry.junction import Junction
from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    SoilLayer,
    SoilModel,
    TwoLayerSoil,
)
from groundfield.soil.effective import effective_resistivity
from groundfield.solver.engine import Backend, Engine
from groundfield.solver.mutual import solve_mutual_field, solve_mutual_matrix
from groundfield.solver.result import FieldResult
from groundfield.sources import (
    CurrentSource,
    Source,
    SourceAdapter,
    VoltageSource,
)
from groundfield.world import World
from groundfield.api import (
    create_conductor,
    create_electrode,
    create_engine,
    create_junction,
    create_source,
    create_world,
    run_simulation,
)

# ---------------------------------------------------------------------
# Lazy imports (PEP 562): module -> names, resolved on first access.
# Keep this table and the TYPE_CHECKING block below in sync (tested).
# ---------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, ...]] = {
    "groundfield.postprocess.plotting": (
        "plot_potential_contour",
        "plot_potential_profile",
        "plot_potential_radial",
        "plot_surface_potential",
        "world_bounds_xy",
    ),
    "groundfield.postprocess.safety": (
        "UniformCurrentFieldWarning",
        "permissible_prospective_touch_voltage_en50522",
        "permissible_touch_voltage_en50522",
        "step_voltage",
        "touch_voltage",
        "touch_voltage_at_part",
        "touch_voltage_envelope",
        "touch_voltage_spectrum",
    ),
    "groundfield.postprocess.current_balance": (
        "cluster_current_balance",
        "electrode_current_table",
        "plot_current_sharing",
        "split_factor",
    ),
    "groundfield.postprocess.geometry_plot": (
        "plot_world",
        "plot_world_3d",
        "world_bounds_3d",
    ),
    "groundfield.postprocess.sweep": (
        "plot_sweep_heatmap",
        "plot_sweep_lines",
        "sweep",
    ),
    "groundfield.postprocess.convergence": (
        "convergence_study",
        "plot_convergence",
    ),
    "groundfield.postprocess.vector_fitting": (
        "PassivityReport",
        "VectorFitQualityWarning",
        "VectorFitResult",
        "check_passivity",
        "fit_to_sympy",
        "rho_f_from_field_result",
        "vector_fit",
    ),
    "groundfield.postprocess.rho_f_standard": (
        "RhoFStandardFit",
        "RhoFStandardQualityWarning",
        "fit_rho_f_standard",
        "fit_to_sympy_standard",
        "rho_f_standard_from_results",
    ),
    "groundfield.io.csv": (
        "read_csv_with_provenance",
        "save_cluster_impedances_csv",
        "save_electrode_table_csv",
        "save_potential_path_csv",
    ),
    "groundfield.io.groundinsight": (
        "BusTypeSpec",
        "EvaluateSpecError",
        "NonPassiveFitError",
        "NonPassiveFitWarning",
        "evaluate_spec",
        "fit_quality_summary",
        "load_bustype_json",
        "save_bustype_json",
        "to_bustype",
        "to_bustype_dict",
    ),
    "groundfield.coupling.sommerfeld_inductance": ("LayeredEarth",),
    "groundfield.io.results": (
        "ResultFormatError",
        "load_result",
        "save_result",
    ),
    "groundfield.solver.result": ("world_fingerprint",),
    "groundfield.solver.registry": (
        "BackendCapabilityError",
        "ExperimentalBackendWarning",
        "available_backends",
    ),
    "groundfield.io.vtk": (
        "export_field_vtk",
        "export_geometry_vtk",
    ),
    "groundfield.generators": (
        "BuildingTypeSpec",
        "Categorical",
        "Constant",
        "Discrete",
        "Distribution",
        "ElectrodeSpec",
        "ExplicitPlacement",
        "FoundationElectrodeSpec",
        "GeneratorConfig",
        "GroundingSystemSpec",
        "HomogeneousSoilSpec",
        "LogNormal",
        "ManhattanGridPlacement",
        "MeasurementInjectionConfig",
        "MeasurementLeadConfig",
        "MeasurementProbeConfig",
        "MeasurementSetupConfig",
        "MultiLayerSoilSpec",
        "Normal",
        "ObstacleBox",
        "LvNetworkLayout",
        "NetworkPlan",
        "OrtsnetzLayout",
        "OsmBuildingPlacement",
        "PenCable",
        "PenTopology",
        "RadialTrunkTopology",
        "RingElectrodeSpec",
        "RodElectrodeSpec",
        "StarCabinetTopology",
        "StarKvsTopology",
        "StripElectrodeSpec",
        "TnNetworkConfig",
        "TnNetworkGenerator",
        "TwoLayerSoilSpec",
        "Uniform",
        "Weibull",
        "WorldGenerator",
        "buried_lead",
        "default_building_catalog",
        "materialise_plan",
        "measured_impedance",
        "neighbour_substation_grounding",
        "overhead_lead",
        "rod_circle",
        "single_rod_grounding",
    ),
    # The geo data class BuildingFootprint has no optional dependency;
    # the active functions raise an ImportError with an install hint
    # when the ``geo`` extra is missing (ADR-0011).
    "groundfield.geo": (
        "BuildingFootprint",
        "OverpassError",
        "Projector",
        "query_and_project",
        "query_buildings",
    ),
    "groundfield.references": ("dwight1936",),
    "groundfield.validation": (
        "ComparisonRow",
        "EngineComparison",
        "compare_engines",
    ),
    "groundfield.solver.fem": ("FemReductionWarning",),
    "groundfield.geometry.electrodes": (
        "concrete_shell_coefficient",
        "shell_coefficient_from_resistance",
        "shell_resistance_from_coefficient",
    ),
    "groundfield.solver.mutual": ("OffDiagonal", "ProbePointsIgnoredWarning"),
    "groundfield.diagnostics": (
        "Diagnostic",
        "DiagnosticsReport",
        "DiagnosticsWarning",
        "ResourceLimitError",
        "check_segment_resolution",
        "diagnose",
        "expected_segments",
        "world_statistics",
    ),
}

if _typing.TYPE_CHECKING:
    from groundfield.postprocess.plotting import (
        plot_potential_contour,
        plot_potential_profile,
        plot_potential_radial,
        plot_surface_potential,
        world_bounds_xy,
    )
    from groundfield.postprocess.safety import (
        UniformCurrentFieldWarning,
        permissible_prospective_touch_voltage_en50522,
        permissible_touch_voltage_en50522,
        step_voltage,
        touch_voltage,
        touch_voltage_at_part,
        touch_voltage_envelope,
        touch_voltage_spectrum,
    )
    from groundfield.postprocess.current_balance import (
        cluster_current_balance,
        electrode_current_table,
        plot_current_sharing,
        split_factor,
    )
    from groundfield.postprocess.geometry_plot import (
        plot_world,
        plot_world_3d,
        world_bounds_3d,
    )
    from groundfield.postprocess.sweep import (
        plot_sweep_heatmap,
        plot_sweep_lines,
        sweep,
    )
    from groundfield.postprocess.convergence import (
        convergence_study,
        plot_convergence,
    )
    from groundfield.postprocess.vector_fitting import (
        PassivityReport,
        VectorFitQualityWarning,
        VectorFitResult,
        check_passivity,
        fit_to_sympy,
        rho_f_from_field_result,
        vector_fit,
    )
    from groundfield.io.results import (
        ResultFormatError,
        load_result,
        save_result,
    )
    from groundfield.solver.result import world_fingerprint
    from groundfield.solver.registry import (
        BackendCapabilityError,
        ExperimentalBackendWarning,
        available_backends,
    )
    from groundfield.postprocess.rho_f_standard import (
        RhoFStandardFit,
        RhoFStandardQualityWarning,
        fit_rho_f_standard,
        fit_to_sympy_standard,
        rho_f_standard_from_results,
    )
    from groundfield.io.csv import (
        read_csv_with_provenance,
        save_cluster_impedances_csv,
        save_electrode_table_csv,
        save_potential_path_csv,
    )
    from groundfield.io.groundinsight import (
        BusTypeSpec,
        EvaluateSpecError,
        NonPassiveFitError,
        NonPassiveFitWarning,
        evaluate_spec,
        fit_quality_summary,
        load_bustype_json,
        save_bustype_json,
        to_bustype,
        to_bustype_dict,
    )
    from groundfield.coupling.sommerfeld_inductance import LayeredEarth
    from groundfield.io.vtk import (
        export_field_vtk,
        export_geometry_vtk,
    )
    from groundfield.generators import (
        BuildingTypeSpec,
        Categorical,
        Constant,
        Discrete,
        Distribution,
        ElectrodeSpec,
        ExplicitPlacement,
        FoundationElectrodeSpec,
        GeneratorConfig,
        GroundingSystemSpec,
        HomogeneousSoilSpec,
        LogNormal,
        ManhattanGridPlacement,
        MeasurementInjectionConfig,
        MeasurementLeadConfig,
        MeasurementProbeConfig,
        MeasurementSetupConfig,
        MultiLayerSoilSpec,
        Normal,
        ObstacleBox,
        LvNetworkLayout,
        NetworkPlan,
        OrtsnetzLayout,
        OsmBuildingPlacement,
        PenCable,
        PenTopology,
        RadialTrunkTopology,
        RingElectrodeSpec,
        RodElectrodeSpec,
        StarCabinetTopology,
        StarKvsTopology,
        StripElectrodeSpec,
        TnNetworkConfig,
        TnNetworkGenerator,
        TwoLayerSoilSpec,
        Uniform,
        Weibull,
        WorldGenerator,
        buried_lead,
        default_building_catalog,
        materialise_plan,
        measured_impedance,
        neighbour_substation_grounding,
        overhead_lead,
        rod_circle,
        single_rod_grounding,
    )
    from groundfield.geo import (
        BuildingFootprint,
        OverpassError,
        Projector,
        query_and_project,
        query_buildings,
    )
    from groundfield.references import dwight1936
    from groundfield.validation import (
        ComparisonRow,
        EngineComparison,
        compare_engines,
    )
    from groundfield.solver.fem import FemReductionWarning
    from groundfield.geometry.electrodes import (
        concrete_shell_coefficient,
        shell_coefficient_from_resistance,
        shell_resistance_from_coefficient,
    )
    from groundfield.solver.mutual import OffDiagonal, ProbePointsIgnoredWarning
    from groundfield.diagnostics import (
        Diagnostic,
        DiagnosticsReport,
        DiagnosticsWarning,
        ResourceLimitError,
        check_segment_resolution,
        diagnose,
        expected_segments,
        world_statistics,
    )

#: Curated namespaces (modules), listed by ``dir(groundfield)``.
_NAMESPACES = ("fit", "io", "plot", "safety")

#: Subpackages and modules reachable as attributes before their first
#: explicit import (``gf.generators``, ``gf.postprocess``, …).
_SUBMODULES = frozenset(
    {
        "api",
        "boundary",
        "conductors",
        "coupling",
        "diagnostics",
        "generators",
        "geo",
        "geometry",
        "postprocess",
        "references",
        "soil",
        "solver",
        "sources",
        "utils",
        "validation",
        "world",
        *_NAMESPACES,
    }
)

_LAZY_NAMES: dict[str, str] = {
    name: module for module, names in _LAZY_IMPORTS.items() for name in names
}

__all__ = [
    "__version__",
    # Data model
    "World",
    "BoundaryConditions",
    "HomogeneousSoil",
    "TwoLayerSoil",
    "MultiLayerSoil",
    "SoilLayer",
    "SoilModel",
    "effective_resistivity",
    "Electrode",
    "RodElectrode",
    "RingElectrode",
    "StripElectrode",
    "PolylineElectrode",
    "StarElectrode",
    "MeshElectrode",
    "GridMeshElectrode",
    "Junction",
    "Conductor",
    "ConductorType",
    "Source",
    "SourceAdapter",
    "CurrentSource",
    "VoltageSource",
    # Numerics
    "Engine",
    "Backend",
    "FieldResult",
    "solve_mutual_matrix",
    "solve_mutual_field",
    "OffDiagonal",
    "ProbePointsIgnoredWarning",
    "concrete_shell_coefficient",
    "shell_coefficient_from_resistance",
    "shell_resistance_from_coefficient",
    "available_backends",
    "BackendCapabilityError",
    "ExperimentalBackendWarning",
    # Factories
    "create_world",
    "create_electrode",
    "create_conductor",
    "create_source",
    "create_junction",
    "create_engine",
    "run_simulation",
    # Plots
    "plot_potential_contour",
    "plot_potential_profile",
    "plot_potential_radial",
    "plot_surface_potential",
    "world_bounds_xy",
    # Safety (touch / step voltage)
    "touch_voltage",
    "touch_voltage_envelope",
    "touch_voltage_at_part",
    "touch_voltage_spectrum",
    "step_voltage",
    "permissible_touch_voltage_en50522",
    "permissible_prospective_touch_voltage_en50522",
    "UniformCurrentFieldWarning",
    # Current sharing
    "cluster_current_balance",
    "electrode_current_table",
    "split_factor",
    "plot_current_sharing",
    # World geometry plots (no solve required)
    "plot_world",
    "plot_world_3d",
    "world_bounds_3d",
    # Parameter-sweep helpers
    "sweep",
    "plot_sweep_lines",
    "plot_sweep_heatmap",
    "convergence_study",
    "plot_convergence",
    # rho-f fits (vector fitting + research standard form)
    "VectorFitResult",
    "vector_fit",
    "fit_to_sympy",
    "rho_f_from_field_result",
    "check_passivity",
    "PassivityReport",
    "VectorFitQualityWarning",
    "RhoFStandardFit",
    "fit_rho_f_standard",
    "fit_to_sympy_standard",
    "rho_f_standard_from_results",
    "RhoFStandardQualityWarning",
    # IO / results (versioned JSON, 10.77)
    "save_result",
    "load_result",
    "world_fingerprint",
    "ResultFormatError",
    # IO / groundinsight bridge
    "BusTypeSpec",
    "to_bustype_dict",
    "to_bustype",
    "save_bustype_json",
    "load_bustype_json",
    "evaluate_spec",
    "EvaluateSpecError",
    "NonPassiveFitWarning",
    "NonPassiveFitError",
    "fit_quality_summary",
    # Coupling / layered earth
    "LayeredEarth",
    # IO / CSV
    "save_potential_path_csv",
    "save_electrode_table_csv",
    "save_cluster_impedances_csv",
    "read_csv_with_provenance",
    # IO / VTK
    "export_geometry_vtk",
    "export_field_vtk",
    # Generators / distributions
    "GeneratorConfig",
    "WorldGenerator",
    "Distribution",
    "Constant",
    "Uniform",
    "Normal",
    "LogNormal",
    "Weibull",
    "Discrete",
    "Categorical",
    "TnNetworkConfig",
    "TnNetworkGenerator",
    "PenTopology",
    "StarCabinetTopology",
    "StarKvsTopology",
    "RadialTrunkTopology",
    "LvNetworkLayout",
    "OrtsnetzLayout",
    "NetworkPlan",
    "materialise_plan",
    "measured_impedance",
    "PenCable",
    "ObstacleBox",
    # Spec layer
    "RodElectrodeSpec",
    "RingElectrodeSpec",
    "StripElectrodeSpec",
    "FoundationElectrodeSpec",
    "ElectrodeSpec",
    "GroundingSystemSpec",
    "BuildingTypeSpec",
    "HomogeneousSoilSpec",
    "TwoLayerSoilSpec",
    "MultiLayerSoilSpec",
    "ManhattanGridPlacement",
    "ExplicitPlacement",
    "rod_circle",
    "default_building_catalog",
    # Measurement setup
    "MeasurementSetupConfig",
    "MeasurementInjectionConfig",
    "MeasurementProbeConfig",
    "MeasurementLeadConfig",
    "overhead_lead",
    "buried_lead",
    "single_rod_grounding",
    "neighbour_substation_grounding",
    # Geo / OSM (optional ``geo`` extra; ADR-0011)
    "BuildingFootprint",
    "OsmBuildingPlacement",
    "Projector",
    "OverpassError",
    "query_buildings",
    "query_and_project",
    # Reference formulas
    "dwight1936",
    # Cross-engine
    "compare_engines",
    "ComparisonRow",
    "FemReductionWarning",
    "EngineComparison",
    # Pre-solve diagnostics
    "world_statistics",
    "expected_segments",
    "check_segment_resolution",
    "diagnose",
    "DiagnosticsReport",
    "Diagnostic",
    "DiagnosticsWarning",
    "ResourceLimitError",
]


def __getattr__(name: str) -> object:
    """Resolve lazy names and subpackages on first access (PEP 562)."""
    module_name = _LAZY_NAMES.get(name)
    if module_name is not None:
        module = _importlib.import_module(module_name)
        try:
            value = getattr(module, name)
        except AttributeError:  # a submodule, e.g. references.dwight1936
            value = _importlib.import_module(f"{module_name}.{name}")
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}",
        name=name,
        obj=_sys.modules[__name__],
    )


def __dir__() -> list[str]:
    """Public names: ``__all__``, the curated namespaces and loaded globals."""
    return sorted(set(globals()) | set(__all__) | set(_NAMESPACES))
