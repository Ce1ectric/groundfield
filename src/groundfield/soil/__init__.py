"""Soil models for ``groundfield``.

This subpackage describes the propagation medium "soil". Supported
models include homogeneous, layered (two- and multi-layer) and
frequency-dependent variants in the spirit of Visacro / Alipio. The
models supply the effective electromagnetic parameters
$\\rho(f)$, $\\varepsilon_r(f)$ required by the field
solver, together with the corresponding Green's functions or
image-charge coefficients.

Contents
--------
HomogeneousSoil
    Uniform half-space with a single resistivity.
TwoLayerSoil
    Upper layer of finite thickness over a semi-infinite lower layer.
MultiLayerSoil
    Generic multi-layer model with arbitrarily many horizontal layers.
SoilLayer
    Single layer used by ``MultiLayerSoil``.
SoilModel
    Discriminated union of all supported soil types.

Notes
-----
For work package 1 the two-layer model is the primary case (parameter
space: varying resistivities and layer thicknesses). The other classes
are extension points for later work.
"""

from __future__ import annotations

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    SoilLayer,
    SoilModel,
    TwoLayerSoil,
)

__all__ = [
    "HomogeneousSoil",
    "TwoLayerSoil",
    "MultiLayerSoil",
    "SoilLayer",
    "SoilModel",
]
