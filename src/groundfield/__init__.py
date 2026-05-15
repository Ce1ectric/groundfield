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

Subpackages
-----------
soil
    Soil models (homogeneous, multi-layer, frequency-dependent).
geometry
    Electrode and conductor geometries, mesh generation.
conductors
    Conductors, PEN, cable shields, and their self/mutual impedances.
solver
    Numerical field solver (image method, MoM) in the frequency domain.
coupling
    Inductive and galvanic coupling between conductors, shields, and
    soil.
postprocess
    Evaluation: potential curves, touch and step voltages, current
    densities, plots.
io
    File I/O and exchange formats, in particular the export of
    reduced ``rho-f`` models for ``groundinsight``.
utils
    Helpers for coordinates, units, logging, and validation.

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

from __future__ import annotations

# Version
__version__ = "0.5.0"

# Re-exports — data model
from groundfield.boundary import BoundaryConditions
from groundfield.conductors.conductor import Conductor, ConductorType
from groundfield.geometry.electrodes import (
    Electrode,
    GridMeshElectrode,
    MeshElectrode,
    RingElectrode,
    RodElectrode,
    StripElectrode,
)
from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    SoilLayer,
    SoilModel,
    TwoLayerSoil,
)
from groundfield.solver.engine import Backend, Engine
from groundfield.solver.result import FieldResult
from groundfield.sources import (
    CurrentSource,
    Source,
    SourceAdapter,
    VoltageSource,
)
from groundfield.world import World

# Re-exports — top-level factories
from groundfield.api import (
    create_conductor,
    create_electrode,
    create_engine,
    create_source,
    create_world,
    run_simulation,
)

# Re-exports — postprocess / plots
from groundfield.postprocess.plotting import (
    plot_potential_contour,
    plot_potential_profile,
    plot_potential_radial,
    plot_surface_potential,
    world_bounds_xy,
)
from groundfield.postprocess.safety import (
    permissible_touch_voltage_en50522,
    step_voltage,
    touch_voltage,
    touch_voltage_envelope,
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
    VectorFitResult,
    fit_to_sympy,
    rho_f_from_field_result,
    vector_fit,
)
from groundfield.postprocess.rho_f_standard import (
    RhoFStandardFit,
    fit_rho_f_standard,
    fit_to_sympy_standard,
    rho_f_standard_from_results,
)

# Re-exports — io / sister-project bridge
from groundfield.io.csv import (
    save_cluster_impedances_csv,
    save_electrode_table_csv,
    save_potential_path_csv,
)
from groundfield.io.groundinsight import (
    BusTypeSpec,
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

# Re-exports — world generators (typical + future)
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
    OsmBuildingPlacement,
    RingElectrodeSpec,
    RodElectrodeSpec,
    StripElectrodeSpec,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
    Uniform,
    Weibull,
    WorldGenerator,
    buried_lead,
    default_building_catalog,
    neighbour_substation_grounding,
    overhead_lead,
    rod_circle,
    single_rod_grounding,
)

# Re-exports — geo subpackage (ADR-0011). The data class
# :class:`BuildingFootprint` has zero optional dependencies and is
# always importable; the active functions (:class:`Projector`,
# :func:`query_buildings`, :func:`query_and_project`) raise a clear
# :class:`ImportError` with an install hint when the optional
# ``geo`` extra (``pip install groundfield[geo]``) is absent.
from groundfield.geo import (
    BuildingFootprint,
    OverpassError,
    Projector,
    query_and_project,
    query_buildings,
)

# Analytical reference formulas (used in plausibility tests)
from groundfield.references import dwight1936

# Cross-engine validation (see ADR-0001)
from groundfield.validation import EngineComparison, compare_engines

# Pre-solve world diagnostics
from groundfield.diagnostics import (
    check_segment_resolution,
    expected_segments,
    world_statistics,
)

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
    "Electrode",
    "RodElectrode",
    "RingElectrode",
    "StripElectrode",
    "MeshElectrode",
    "GridMeshElectrode",
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
    # Factories
    "create_world",
    "create_electrode",
    "create_conductor",
    "create_source",
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
    "step_voltage",
    "permissible_touch_voltage_en50522",
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
    "RhoFStandardFit",
    "fit_rho_f_standard",
    "fit_to_sympy_standard",
    "rho_f_standard_from_results",
    # IO / groundinsight bridge
    "BusTypeSpec",
    "to_bustype_dict",
    "to_bustype",
    "save_bustype_json",
    "load_bustype_json",
    "evaluate_spec",
    "fit_quality_summary",
    # Coupling / layered earth
    "LayeredEarth",
    # IO / CSV
    "save_potential_path_csv",
    "save_electrode_table_csv",
    "save_cluster_impedances_csv",
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
    "EngineComparison",
    # Pre-solve diagnostics
    "world_statistics",
    "expected_segments",
    "check_segment_resolution",
]
