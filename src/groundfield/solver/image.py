"""Image-charge backend for **homogeneous** soil.

Computes the potential field of an arbitrary grounding system in a
homogeneous half-space (resistivity $\\rho$, soil surface at
$z = 0$, $z$ axis pointing into the soil) using the
classical image-charge method.

Notes
-----
A point current source $I$ at $r_s = (x_s, y_s, z_s)$ with
$z_s > 0$ (inside the soil) produces, in a homogeneous half-space
with an insulating soil surface, the potential
$$
\\varphi(r) \\;=\\; \\frac{\\rho\\, I}{4\\pi}\\,
\\Big(\\frac{1}{|r - r_s|} + \\frac{1}{|r - r_s'|}\\Big),
$$
with the image $r_s' = (x_s, y_s, -z_s)$ mirrored at the soil
surface. An extended electrode is discretised into $N$ segments;
each segment carries one point current source at its midpoint. The
total current $I_e$ of an electrode is distributed **uniformly
per unit length** across its segments — a surprisingly good
approximation for wire electrodes at low frequencies (cf. Sunde 1968,
Tagg 1964).

The input impedance of an electrode is computed as the average of the
potential on its own segment midpoints (average-potential method). For
a single driven rod the backend reproduces the Sunde formula within a
few per cent.

Further properties of this backend:

* Frequency-independent: in the quasi-static range
  $f < 1\\,\\mathrm{kHz}$ the backend returns the same real
  solution per frequency. Complex extensions (Carson,
  frequency-dependent soil) come in later backends.
* Multiple electrodes: each electrode has its own total current.
  Sources inject at the node they are attached to and, with
  ``return_to``, take the current back at a second node (two-terminal
  sources). Electrodes bonded by ideal conductors share one potential.
  A source-free cluster floats with zero net current: a lone passive
  electrode carries no current (uniform-current ansatz) and reports the
  incident potential, a bonded group of passive electrodes exchanges
  current to stay one equipotential.

References
----------
.. [1] E. D. Sunde, *Earth Conduction Effects in Transmission
       Systems*, Dover, 1968.
.. [2] G. F. Tagg, *Earth Resistances*, Pitman, 1964.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.world import World

# Re-exports: the implementation moved into private modules (review
# pass 10, 10.75 / 10.79). Sibling backends, tests and user notebooks
# import these names from ``groundfield.solver.image``; keep them here.
from groundfield.solver._discretize import (
    _conductor_node_name as _conductor_node_name,
)
from groundfield.solver._discretize import (
    _discretize_conductor as _discretize_conductor,
)
from groundfield.solver._discretize import (
    _discretize_electrode as _discretize_electrode,
)
from groundfield.solver._discretize import _DistributedBranch as _DistributedBranch
from groundfield.solver._discretize import _Segment as _Segment
from groundfield.solver._discretize import _segments_per_wire as _segments_per_wire
from groundfield.solver._discretize import (
    _split_interval_at_interfaces as _split_interval_at_interfaces,
)
from groundfield.solver._guards import (
    _MIN_IMAGE_SEPARATION_RATIO as _MIN_IMAGE_SEPARATION_RATIO,
)
from groundfield.solver._guards import _MIN_DISTANCE as _MIN_DISTANCE
from groundfield.solver._guards import _SURFACE_TOL as _SURFACE_TOL
from groundfield.solver._guards import ShallowSegmentWarning as ShallowSegmentWarning
from groundfield.solver._guards import (
    _check_segment_depths as _check_segment_depths,
)
from groundfield.solver._guards import (
    _require_image_separation as _require_image_separation,
)
from groundfield.solver._guards import _require_thin_wire as _require_thin_wire
from groundfield.solver._guards import (
    _segment_owner_label as _segment_owner_label,
)
from groundfield.solver._inductive import _CarsonBuilder as _CarsonBuilder
from groundfield.solver._inductive import (
    _assemble_inductance_matrix as _assemble_inductance_matrix,
)
from groundfield.solver._kernels import HalfSpaceKernel
from groundfield.solver._kernels import (
    _self_corrected_kernel as _self_corrected_kernel,
)
from groundfield.solver._pipeline import (
    _reject_concrete_shells as _reject_concrete_shells,
)
from groundfield.solver._pipeline import solve_image_family
from groundfield.solver._pipeline import (
    _warn_ignored_sources as _warn_ignored_sources,
)
from groundfield.solver._pipeline import (
    _weighted_node_potential as _weighted_node_potential,
)
from groundfield.solver._topology import (
    _build_clusters as _build_clusters,
)
from groundfield.solver._topology import (
    _build_distributed_topology as _build_distributed_topology,
)
from groundfield.solver._topology import (
    _build_finite_branches as _build_finite_branches,
)
from groundfield.solver._topology import (
    _solve_cluster_currents as _solve_cluster_currents,
)
from groundfield.soil.models import HomogeneousSoil
from groundfield.solver.result import FieldResult

__all__ = ["solve_image", "ShallowSegmentWarning"]


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


class _HalfSpaceGreen:
    """Green's function of the homogeneous half-space (pipeline object).

    The reaction kernel is the point source plus its mirror image in
    the insulating soil surface, :math:`\\rho/(4\\pi)\\,(1/r + 1/r')`,
    with the analytical line self-term on the diagonal
    (:func:`_self_corrected_kernel`).
    """

    backend = "image"
    where = "solve_image"
    layer_interfaces = None

    def __init__(self, rho: float, kernel: str = "point") -> None:
        self.rho = rho
        self.kernel = kernel
        self.soil_resistivity = float(rho)
        self.reference_resistivity = float(rho)

    def prepare(self, segments, arrays, *, stacklevel: int):  # noqa: ANN001
        """Reaction operator with the fixed resistivity ``rho``.

        The operator assembles its matrix once and reuses it for the
        segment potentials after the solve (10.57). ``kernel="line"``
        selects the exact line-source kernel (10.28).
        """
        line = arrays.line_segments() if self.kernel == "line" else None
        return HalfSpaceKernel(self.rho, line=line)

    def metadata(self, solution, *, stacklevel: int) -> dict:  # noqa: ANN001
        """No backend-specific metadata."""
        return {}


def solve_image(world: "World", engine: "Engine") -> FieldResult:
    """Image-charge solver for homogeneous soil.

    Runs the shared image-family pipeline
    (:mod:`groundfield.solver._pipeline`) with the half-space Green's
    function.

    Parameters
    ----------
    world
        World whose ``soil`` must be a :class:`HomogeneousSoil`.
    engine
        Engine configuration; relevant fields are ``segment_length``,
        ``frequencies`` and ``earth_inductive_model``.

    Returns
    -------
    FieldResult
        Node potentials and leakage currents per frequency, point
        sources and metadata.

    Raises
    ------
    TypeError
        If the soil is not a :class:`HomogeneousSoil`.
    ValueError
        If the world has no electrodes, or the geometry fails a
        discretisation guard (thin wire, burial depth).
    """
    if not isinstance(world.soil, HomogeneousSoil):
        raise TypeError(
            "Backend 'image' requires HomogeneousSoil. "
            f"Got: {type(world.soil).__name__}. "
            "For layered models pick backend='image_2layer' "
            "(Tagg/Sunde) or backend='mom' (planned)."
        )
    if not world.electrodes:
        raise ValueError("World contains no electrodes.")
    green = _HalfSpaceGreen(world.soil.rho, engine.kernel)
    return solve_image_family(world, engine, green, stacklevel=3)
