"""Geometries of electrodes and grounding systems.

This subpackage provides the 3-D geometry of the grounding system:
the standard parametric electrode geometries and their discretisation
into straight wire segments (:class:`WireSegments`), which every
electrode produces itself through ``segments(ds)``.

Contents
--------
RodElectrode, RingElectrode, StripElectrode, PolylineElectrode,
StarElectrode, MeshElectrode, GridMeshElectrode
    Parametric standard electrode geometries. Each answers the
    geometry queries ``segments(ds)``, ``segment_count(ds)``,
    ``wire_polylines()``, ``bounds()``, ``total_length()`` and
    ``min_feature_length()``.
Electrode
    Discriminated union over all electrode types.
WireSegments
    Midpoints, lengths and unit tangents of a discretised wire
    geometry.

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
    PolylineElectrode,
    RingElectrode,
    RodElectrode,
    StarElectrode,
    StripElectrode,
)
from groundfield.geometry.junction import Junction
from groundfield.geometry.primitives import WireSegments

__all__ = [
    "Electrode",
    "RodElectrode",
    "RingElectrode",
    "StripElectrode",
    "PolylineElectrode",
    "StarElectrode",
    "MeshElectrode",
    "GridMeshElectrode",
    "Junction",
    "WireSegments",
]
