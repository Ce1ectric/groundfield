"""Field-solution post-processing.

Extracts the engineering quantities of interest from the computed
potential distribution:

- earth potential rise ``U_E`` at the feed-in point,
- touch and step voltages (footprint-anchored), the EN 50522:2022
  permissible touch voltage and prospective touch voltage, and the
  true-RMS touch voltage over a fault spectrum (:mod:`.safety`),
- potential profiles along defined paths (measurement traverses),
- current sharing onto return paths and the galvanic split factor per
  source (:mod:`.current_balance`; the inductive *Reduktionsfaktor* is
  on the roadmap),
- parameter sweeps and convergence studies (:mod:`.sweep`,
  :mod:`.convergence`),
- reduced rho-f models: vector fitting and the standard form
  (:mod:`.vector_fitting`, :mod:`.rho_f_standard`),
- 2-D / 3-D field and geometry plots (:mod:`.plotting`,
  :mod:`.geometry_plot`).
"""

from __future__ import annotations

from groundfield.postprocess.current_balance import (
    cluster_current_balance,
    electrode_current_table,
    plot_current_sharing,
    split_factor,
)
from groundfield.postprocess.convergence import (
    convergence_study,
    plot_convergence,
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
from groundfield.postprocess.plotting import (
    plot_potential_contour,
    plot_potential_profile,
    plot_potential_radial,
    plot_surface_potential,
    world_bounds_xy,
)
from groundfield.postprocess.rho_f_standard import (
    RhoFStandardFit,
    RhoFStandardQualityWarning,
    fit_rho_f_standard,
    fit_to_sympy_standard,
    rho_f_standard_from_results,
)
from groundfield.postprocess.safety import (
    NEAR_FIELD_DISTANCE_M,
    UniformCurrentFieldWarning,
    permissible_prospective_touch_voltage_en50522,
    permissible_touch_voltage_en50522,
    step_voltage,
    touch_voltage,
    touch_voltage_envelope,
    touch_voltage_spectrum,
)
from groundfield.postprocess.vector_fitting import (
    VectorFitResult,
    fit_to_sympy,
    rho_f_from_field_result,
    vector_fit,
)

__all__ = [
    "plot_potential_contour",
    "plot_potential_profile",
    "plot_potential_radial",
    "plot_surface_potential",
    "world_bounds_xy",
    "VectorFitResult",
    "vector_fit",
    "fit_to_sympy",
    "rho_f_from_field_result",
    "RhoFStandardFit",
    "RhoFStandardQualityWarning",
    "fit_rho_f_standard",
    "fit_to_sympy_standard",
    "rho_f_standard_from_results",
    "touch_voltage",
    "touch_voltage_envelope",
    "step_voltage",
    "touch_voltage_spectrum",
    "permissible_touch_voltage_en50522",
    "permissible_prospective_touch_voltage_en50522",
    "UniformCurrentFieldWarning",
    "NEAR_FIELD_DISTANCE_M",
    "cluster_current_balance",
    "electrode_current_table",
    "split_factor",
    "plot_current_sharing",
    "plot_world",
    "plot_world_3d",
    "world_bounds_3d",
    "sweep",
    "plot_sweep_lines",
    "plot_sweep_heatmap",
    "convergence_study",
    "plot_convergence",
]
