"""Numerical field solver.

This subpackage forms the computational core of ``groundfield``. The
key quantity is the complex potential ``phi(r, f)`` in the soil and on
the conductor surfaces, evaluated per frequency in the phasor domain.
The default solution method for homogeneous soil is the closed-form
image-charge sum; for layered soil the Tagg/Sunde image series. A
Method-of-Moments backend with the layered Green's function and a
finite-element backend (``scikit-fem``) are reserved.

Contents
--------
Engine
    Top-level configuration of the numerical kernel: backend choice,
    frequency list, mesh resolution, tolerances. ``solve(world)`` runs
    the simulation.
Backend
    Literal type listing the available backends
    (``"image"``, ``"image_2layer"``, ``"mom"``, ``"fem"``).
FieldResult, PointSource
    Result objects.
solve_image, solve_image_2layer
    Backend entry points (usually called via ``Engine.solve``).

Guiding principle
-----------------
The PDE / field model is a reference, not the end product. Every
solution must expose the quantities required by ``groundinsight``
(input impedance, transfer impedances, ``rho-f`` curve) for the
reduction step.
"""

from __future__ import annotations

from groundfield.solver.engine import Backend, Engine
from groundfield.solver.image import solve_image
from groundfield.solver.image_2layer import solve_image_2layer
from groundfield.solver.mom import solve_mom
from groundfield.solver.result import FieldResult, PointSource

__all__ = [
    "Engine",
    "Backend",
    "FieldResult",
    "PointSource",
    "solve_image",
    "solve_image_2layer",
    "solve_mom",
]
