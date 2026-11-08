"""Soil models for ``groundfield``.

This subpackage describes the propagation medium "soil": a horizontally
stratified half-space below a non-conducting air half-space, described
by the layer resistivities $\\rho_i$ (Ω·m) and thicknesses $h_i$ (m).
Below 1 kHz the formulation is quasi-static (conduction currents only),
so the resistivities and thicknesses are the only parameters any
backend reads. The solver turns them into Green's functions or
image-charge coefficients.

Contents
--------
HomogeneousSoil
    Uniform half-space with a single resistivity ``rho``.
TwoLayerSoil
    Upper layer (``rho_1``, thickness ``h_1``) over a semi-infinite
    lower layer (``rho_2``).
MultiLayerSoil
    Generic multi-layer model with arbitrarily many horizontal layers.
SoilLayer
    Single layer (``rho``, thickness ``h``; ``h=None`` for the bottom
    half-space) used by ``MultiLayerSoil``.
SoilModel
    Discriminated union of all supported soil types (``kind``).
effective_resistivity
    Effective resistivity $\\rho_\\text{eff} = \\rho_1^{1-w}\\rho_2^{w}$ of a
    homogeneous or two-layer soil (AP1 calibration): the ``rho`` of a
    groundinsight bus and of a BusType fitted in layered soil.
EFFECTIVE_RESISTIVITY_PRESETS
    Read-only mapping of named weight presets ``(w0, beta)`` of
    ``effective_resistivity`` per electrode class (``"lv_network"``, the
    default; ``"station"``; ``"lv_group_<g>_p<pp>"`` of the AP2 hand-over).

Notes
-----
The two-layer model is the primary case (parameter space: varying
resistivities and layer thicknesses). Names follow the literature
symbols; the pre-2026.11 names ``resistivity`` and ``thickness`` are
deprecated aliases (see :mod:`groundfield.soil.models`).
"""

from groundfield.soil.effective import (
    EFFECTIVE_RESISTIVITY_PRESETS,
    effective_resistivity,
)
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
    "effective_resistivity",
    "EFFECTIVE_RESISTIVITY_PRESETS",
]
