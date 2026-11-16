"""Coupling between conductors, shields, and soil.

This subpackage provides the coupling relations that follow from the
conductor geometry and the material properties:

- galvanic coupling via shared grounding nodes (layered Green's
  functions, :mod:`~groundfield.coupling.layered_green`),
- inductive coupling between conductor segments (Neumann partial
  inductances, :mod:`~groundfield.coupling.inductance`),
- the earth-return correction of the inductive coupling: Carson's
  per-metre correction (:mod:`~groundfield.coupling.carson`, homogeneous
  earth, long parallel horizontal wires — on one side of the surface, or
  one above and one below it through the transmitted field; the matrix
  builder hands every other pair to the Sommerfeld kernel) and the
  geometric Sommerfeld/Pollaczek kernel
  (:mod:`~groundfield.coupling.sommerfeld_inductance`, exact quasi-static
  for any number of layers).

A correction matrix is only meaningful together with the image
convention of the inductance matrix it is added to (``image=``
argument: ``"none"``, ``"additive"``, ``"antiparallel"``; see
:mod:`groundfield.coupling.inductance`). There is no capacitive
coupling: below 1 kHz it is negligible for buried conductors.

Guiding question
----------------
How strong are coupling and return-path effects in the low-frequency
range, and when does the diffusion field / Carson model actually
matter? The routines here provide the numerical basis for that
assessment.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groundfield.coupling.sommerfeld_inductance import LayeredEarth

__all__: list[str] = ["resolve_earth_conductivity", "resolve_earth_layers"]


def resolve_earth_conductivity(soil) -> float:
    """Resolve earth conductivity $\\sigma_\\text{earth}$ from a soil model.

    Used by the Carson earth-return correction (ADR-0005) to get a
    single scalar conductivity from the (potentially layered) soil
    model. The mapping is:

    - :class:`~groundfield.soil.models.HomogeneousSoil` →
      $\\sigma = 1/\\rho$, exact.
    - :class:`~groundfield.soil.models.TwoLayerSoil` →
      $\\sigma = 1/\\rho_1$ (upper layer), with a runtime warning.
    - :class:`~groundfield.soil.models.MultiLayerSoil` →
      $\\sigma = 1/\\rho_1$ (top layer), with a runtime warning.

    The layered cases are an *approximation*; the Pollaczek/Sommerfeld
    kernel (``earth_inductive_model="sommerfeld"``) is the rigorous
    alternative.

    Parameters
    ----------
    soil
        Any concrete soil model from
        :mod:`groundfield.soil.models`.

    Returns
    -------
    sigma_earth : float
        Earth conductivity in S/m.

    Raises
    ------
    TypeError
        If the soil model has no resistivity field that can be
        interpreted as an upper-layer value.
    """
    from groundfield.soil.models import (
        HomogeneousSoil,
        MultiLayerSoil,
        TwoLayerSoil,
    )

    if isinstance(soil, HomogeneousSoil):
        return 1.0 / float(soil.resistivity)
    if isinstance(soil, TwoLayerSoil):
        warnings.warn(
            "Carson series uses upper-layer rho_1 for a TwoLayerSoil — "
            "this is an approximation. For a rigorous result on layered "
            "soils use the Pollaczek/Sommerfeld kernel via "
            "earth_inductive_model='sommerfeld'. See ADR-0005 §'Layered earth'.",
            UserWarning,
            stacklevel=2,
        )
        return 1.0 / float(soil.rho_1)
    if isinstance(soil, MultiLayerSoil):
        warnings.warn(
            "Carson series uses top-layer rho for a MultiLayerSoil — "
            "this is an approximation. For a rigorous result on layered "
            "soils use the Pollaczek/Sommerfeld kernel via "
            "earth_inductive_model='sommerfeld'. See ADR-0005 §'Layered earth'.",
            UserWarning,
            stacklevel=2,
        )
        return 1.0 / float(soil.layers[0].resistivity)
    raise TypeError(f"Unsupported soil type for Carson series: {type(soil).__name__}")


def resolve_earth_layers(soil) -> "LayeredEarth":
    """Resolve a soil model into a :class:`LayeredEarth` configuration.

    Used by the Sommerfeld earth-return kernel (ADR-0006). Unlike
    :func:`resolve_earth_conductivity`, this preserves the full
    layered structure — no warning is emitted: the kernel is exact for
    any number of layers (generalised reflection coefficients, review
    pass 10, finding 10.35; before that, ``n >= 3`` used a single
    air-image approximation that was 17–25 % off in the earth-return
    reactance).

    Parameters
    ----------
    soil
        Any concrete soil model from
        :mod:`groundfield.soil.models`.

    Returns
    -------
    LayeredEarth
        Frozen layered-earth configuration (resistivities and layer
        thicknesses) consumable by
        :func:`groundfield.coupling.sommerfeld_inductance.build_sommerfeld_correction_matrix`.

    Raises
    ------
    TypeError
        If ``soil`` is not a :class:`~groundfield.soil.models.HomogeneousSoil`,
        :class:`~groundfield.soil.models.TwoLayerSoil` or
        :class:`~groundfield.soil.models.MultiLayerSoil`.
    """
    from groundfield.coupling.sommerfeld_inductance import LayeredEarth
    from groundfield.soil.models import (
        HomogeneousSoil,
        MultiLayerSoil,
        TwoLayerSoil,
    )

    if isinstance(soil, HomogeneousSoil):
        return LayeredEarth(
            rhos=(float(soil.resistivity),),
            thicknesses=(),
        )
    if isinstance(soil, TwoLayerSoil):
        return LayeredEarth(
            rhos=(float(soil.rho_1), float(soil.rho_2)),
            thicknesses=(float(soil.h_1),),
        )
    if isinstance(soil, MultiLayerSoil):
        rhos = tuple(float(layer.resistivity) for layer in soil.layers)
        thicknesses = tuple(float(layer.thickness) for layer in soil.layers[:-1])
        return LayeredEarth(rhos=rhos, thicknesses=thicknesses)
    raise TypeError(
        f"Unsupported soil type for Sommerfeld kernel: {type(soil).__name__}"
    )
