"""Plotting: the curated namespace ``gf.plot``.

Every plotting helper of groundfield in one place — potential maps and
profiles of a solved :class:`~groundfield.FieldResult`, the world
geometry (no solve needed), current sharing, parameter sweeps and
convergence studies. The functions are the same objects as the flat
top-level names (``gf.plot.plot_world is gf.plot_world``). matplotlib is
imported on the first plot call.

Plotted potentials are complex phasors; see each function for the
component it shows.
"""

from groundfield.postprocess.convergence import plot_convergence
from groundfield.postprocess.current_balance import plot_current_sharing
from groundfield.postprocess.geometry_plot import (
    plot_world,
    plot_world_3d,
    world_bounds_3d,
)
from groundfield.postprocess.plotting import (
    plot_potential_contour,
    plot_potential_profile,
    plot_potential_radial,
    plot_surface_potential,
    world_bounds_xy,
)
from groundfield.postprocess.sweep import plot_sweep_heatmap, plot_sweep_lines

__all__ = [
    # potential of a solved result
    "plot_potential_contour",
    "plot_potential_profile",
    "plot_potential_radial",
    "plot_surface_potential",
    "world_bounds_xy",
    # world geometry (no solve required)
    "plot_world",
    "plot_world_3d",
    "world_bounds_3d",
    # current sharing, sweeps, convergence
    "plot_current_sharing",
    "plot_sweep_lines",
    "plot_sweep_heatmap",
    "plot_convergence",
]
