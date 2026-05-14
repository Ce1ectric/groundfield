"""Coupling between conductors, shields, and soil.

This subpackage provides the coupling relations that follow from the
conductor geometry and the material properties:

- galvanic coupling via shared grounding nodes,
- inductive coupling between parallel conductors (Neumann integrals),
- Carson correction for the earth-return path at frequencies below a
  few kHz,
- capacitive coupling between conductor and soil (optional).

Guiding question
----------------
How strong are coupling and return-path effects in the low-frequency
range, and when does the diffusion field / Carson model actually
matter? The routines here provide the numerical basis for that
assessment.
"""

from __future__ import annotations

import warnings

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

    The layered cases are an *approximation*; ADR-0005 references the
    Pollaczek/Sommerfeld kernel (via ``mom_sommerfeld``) as the
    rigorous alternative.

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
            "backend='mom_sommerfeld'. See ADR-0005 §'Layered earth'.",
            UserWarning,
            stacklevel=2,
        )
        return 1.0 / float(soil.rho_1)
    if isinstance(soil, MultiLayerSoil):
        warnings.warn(
            "Carson series uses top-layer rho for a MultiLayerSoil — "
            "this is an approximation. For a rigorous result on layered "
            "soils use the Pollaczek/Sommerfeld kernel via "
            "backend='mom_sommerfeld'. See ADR-0005 §'Layered earth'.",
            UserWarning,
            stacklevel=2,
        )
        return 1.0 / float(soil.layers[0].resistivity)
    raise TypeError(
        f"Unsupported soil type for Carson series: {type(soil).__name__}"
    )


def resolve_earth_layers(soil) -> "LayeredEarth":
    """Resolve a soil model into a :class:`LayeredEarth` configuration.

    Used by the Sommerfeld earth-return kernel (ADR-0006). Unlike
    :func:`resolve_earth_conductivity`, this preserves the full
    layered structure — no warning is emitted, and layered soils
    are handled natively by the Pollaczek/Wait kernel.

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
    """
    from groundfield.coupling.sommerfeld_inductance import LayeredEarth
    from groundfield.soil.models import (
        HomogeneousSoil,
        MultiLayerSoil,
        TwoLayerSoil,
    )

    if isinstance(soil, HomogeneousSoil):
        return LayeredEarth(
            rhos=(float(soil.resistivity),), thicknesses=(),
        )
    if isinstance(soil, TwoLayerSoil):
        return LayeredEarth(
            rhos=(float(soil.rho_1), float(soil.rho_2)),
            thicknesses=(float(soil.h_1),),
        )
    if isinstance(soil, MultiLayerSoil):
        rhos = tuple(float(layer.resistivity) for layer in soil.layers)
        thicknesses = tuple(
            float(layer.thickness) for layer in soil.layers[:-1]
        )
        return LayeredEarth(rhos=rhos, thicknesses=thicknesses)
    raise TypeError(
        f"Unsupported soil type for Sommerfeld kernel: "
        f"{type(soil).__name__}"
    )
