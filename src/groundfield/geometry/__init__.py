"""Geometries of electrodes and grounding systems.

This subpackage provides the 3-D geometry of the grounding system. An
electrode is built from a set of wire segments (:class:`Segment`) with
start and end points, wire radius, material, and a mapping to one or
more electric nodes. The subpackage builds a mesh for the field solver
and exposes the standard parametric geometries.

Contents
--------
RodElectrode, RingElectrode, StripElectrode, MeshElectrode,
GridMeshElectrode
    Parametric standard electrode geometries.
Electrode
    Discriminated union over all electrode types.

Notes
-----
The geometry layer is deliberately decoupled from the solver, so the
same geometry can be fed into the image-charge backend, MoM, or FEM.
"""

from __future__ import annotations

from groundfield.geometry.electrodes import (
    Electrode,
    GridMeshElectrode,
    MeshElectrode,
    RingElectrode,
    RodElectrode,
    StripElectrode,
)

__all__ = [
    "Electrode",
    "RodElectrode",
    "RingElectrode",
    "StripElectrode",
    "MeshElectrode",
    "GridMeshElectrode",
]
