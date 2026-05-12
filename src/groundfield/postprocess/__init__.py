"""Field-solution post-processing.

Extracts the engineering quantities of interest from the computed
potential distribution:

- earth potential rise ``U_E`` at the feed-in point,
- touch and step voltages at arbitrary measurement points,
- potential profiles along defined paths (measurement traverses),
- current density in the soil and current sharing onto return paths,
- galvanic split factor per source (resistive division across
  parallel paths; the inductive *Reduktionsfaktor* is on the
  roadmap),
- 2-D / 3-D field plots.

Contents
--------
plotting
    Matplotlib helpers: contour plots in xy / xz planes, line
    profiles, radial trumpet plots.
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
    fit_rho_f_standard,
    fit_to_sympy_standard,
    rho_f_standard_from_results,
)
from groundfield.postprocess.safety import (
    permissible_touch_voltage_en50522,
    step_voltage,
    touch_voltage,
    touch_voltage_envelope,
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
    "fit_rho_f_standard",
    "fit_to_sympy_standard",
    "rho_f_standard_from_results",
    "touch_voltage",
    "touch_voltage_envelope",
    "step_voltage",
    "permissible_touch_voltage_en50522",
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
