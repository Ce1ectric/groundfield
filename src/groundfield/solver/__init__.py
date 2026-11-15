"""Numerical field solver.

This subpackage forms the computational core of ``groundfield``. The
key quantity is the complex potential ``phi(r, f)`` in the soil and on
the conductor surfaces, evaluated per frequency in the phasor domain.
The backends are registered in :mod:`groundfield.solver.registry`, the
authority on names, capabilities and options
(:func:`available_backends`): the image family (``image``, the
closed-form image-charge sum for homogeneous soil, which forwards
layered soils to ``image_2layer``, the Tagg/Sunde image series), the
point-matching method of moments (``mom`` with the same kernels,
``mom_sommerfeld`` with Sommerfeld quadrature of the layered Green's
function for any number of layers) and the experimental finite-element
backend ``fem`` (axisymmetric for coaxial rods and rings, three-dimensional
otherwise). ``cim``, ``bem`` and ``image_nlayer``
are deprecated names of ``image`` and ``mom``.

Contents
--------
Engine
    Top-level configuration of the numerical kernel: backend choice,
    frequency list, mesh resolution, tolerances. ``solve(world)`` runs
    the simulation.
Backend
    Literal type of the built-in backend names (for type hints; the
    registry also accepts registered third-party names).
FieldResult, PointSource, SourceRecord
    Result objects.
solve_image, solve_image_2layer, solve_mom
    Backend entry points (usually called via ``Engine.solve``).
solve_mutual_matrix, solve_mutual_field
    Multi-port grounding matrices of the image family.
register_backend, get_backend, available_backends, BackendCapabilities
    The backend registry, with its error, warning and options classes.
InterfaceContactWarning
    Diagnostic warning category: a rod head or tip lies within one wire
    radius of a layer interface, where the thin-wire model (no end face)
    deviates from the solid rod by up to +21 % (audit A3S-05).
PointKernelInterfaceWarning
    Diagnostic warning category: the point form of the reaction matrix
    (``kernel="point"`` and the ``mom`` family) meets a horizontal wire
    within 0.6 segment lengths of a layer interface, on either side of it
    (-3 % ... -17 % against the exact line kernel; closer still the solve
    raises).
ShallowSegmentWarning
    Diagnostic warning category: a leakage segment is longer than four
    times its burial depth, so the point-image stand-in for the
    segment's image line biases the reaction-matrix diagonal high (point
    kernel and the ``mom`` family; the exact line kernel has no such
    bias). Re-exported here so callers can write
    ``warnings.simplefilter("ignore", ShallowSegmentWarning)`` without
    importing from the backend module.

Guiding principle
-----------------
The PDE / field model is a reference, not the end product. Every
solution must expose the quantities required by ``groundinsight``
(input impedance, transfer impedances, ``rho-f`` curve) for the
reduction step.
"""

from __future__ import annotations

from groundfield.solver._guards import (
    InterfaceContactWarning,
    PointKernelInterfaceWarning,
)
from groundfield.solver.engine import CORE_METADATA_KEYS, Backend, Engine
from groundfield.solver.image import ShallowSegmentWarning, solve_image
from groundfield.solver.image_2layer import solve_image_2layer
from groundfield.solver.mom import solve_mom
from groundfield.solver.mutual import (
    ProbePointsIgnoredWarning,
    solve_mutual_field,
    solve_mutual_matrix,
)
from groundfield.solver.registry import (
    BackendCapabilities,
    BackendCapabilityError,
    BackendCapabilityWarning,
    BackendOptions,
    BackendOptionsWarning,
    ExperimentalBackendWarning,
    FemOptions,
    MomSommerfeldOptions,
    available_backends,
    get_backend,
    register_backend,
)
from groundfield.solver.result import FieldResult, PointSource, SourceRecord

__all__ = [
    "Engine",
    "Backend",
    "FieldResult",
    "PointSource",
    "SourceRecord",
    # backend registry (10.74)
    "register_backend",
    "get_backend",
    "available_backends",
    "BackendCapabilities",
    "BackendCapabilityError",
    "BackendCapabilityWarning",
    "BackendOptions",
    "BackendOptionsWarning",
    "ExperimentalBackendWarning",
    "MomSommerfeldOptions",
    "FemOptions",
    "CORE_METADATA_KEYS",
    "InterfaceContactWarning",
    "PointKernelInterfaceWarning",
    "ShallowSegmentWarning",
    "solve_image",
    "solve_image_2layer",
    "solve_mom",
    "solve_mutual_matrix",
    "solve_mutual_field",
    "ProbePointsIgnoredWarning",
]
