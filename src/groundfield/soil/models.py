"""Data classes for soil models.

The classes defined here are pure data containers (Pydantic v2). They
describe the geometry and electromagnetic parameters of the soil. The
actual evaluation (Green's function, image-charge sum, reflection
coefficient) happens inside :mod:`groundfield.solver` and consumes the
data exposed here.

Notes
-----
In the frequency range below 1 kHz (work package 1 of the
dissertation) the quasi-static soil models implemented here are
sufficient. Frequency-dependent extensions (Visacro, Alipio, Messier)
will be added later as specialised subclasses without breaking this
interface.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "SoilModel",
    "HomogeneousSoil",
    "TwoLayerSoil",
    "MultiLayerSoil",
    "SoilLayer",
]


class _SoilBase(BaseModel):
    """Common base for all soil models.

    Sets up the Pydantic configuration and defines a discriminator
    field ``kind``. Concrete soil models inherit and override it with
    a literal value.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    kind: str = Field(..., description="Discriminator for the soil model.")


class HomogeneousSoil(_SoilBase):
    """Homogeneous soil with a uniform resistivity.

    Attributes
    ----------
    resistivity
        Resistivity $\\rho$ in $\\Omega\\,\\mathrm{m}$.
    relative_permittivity
        Relative permittivity $\\varepsilon_r$. Default 10
        (mid-range value for soil, IEEE Std 80).
    """

    kind: Literal["homogeneous"] = "homogeneous"
    resistivity: float = Field(..., gt=0.0, description="ρ in Ω·m.")
    relative_permittivity: float = Field(default=10.0, gt=0.0)


class TwoLayerSoil(_SoilBase):
    """Two-layer model (finite-thickness upper layer over a semi-infinite lower layer).

    Attributes
    ----------
    rho_1
        Resistivity of the upper layer $\\rho_1$ in
        $\\Omega\\,\\mathrm{m}$.
    rho_2
        Resistivity of the lower (semi-infinite) layer $\\rho_2$
        in $\\Omega\\,\\mathrm{m}$.
    h_1
        Thickness of the upper layer $h_1$ in metres.
    relative_permittivity
        Common relative permittivity for both layers (simplified).
    """

    kind: Literal["two_layer"] = "two_layer"
    rho_1: float = Field(..., gt=0.0)
    rho_2: float = Field(..., gt=0.0)
    h_1: float = Field(..., gt=0.0)
    relative_permittivity: float = Field(default=10.0, gt=0.0)

    @property
    def reflection_coefficient(self) -> float:
        """Reflection coefficient $K = (\\rho_2 - \\rho_1) / (\\rho_2 + \\rho_1)$."""
        return (self.rho_2 - self.rho_1) / (self.rho_2 + self.rho_1)


class SoilLayer(BaseModel):
    """A single layer used by :class:`MultiLayerSoil`."""

    model_config = ConfigDict(extra="forbid")

    resistivity: float = Field(..., gt=0.0, description="ρ in Ω·m.")
    thickness: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Layer thickness in m. ``None`` for the semi-infinite "
            "bottom layer."
        ),
    )
    relative_permittivity: float = Field(default=10.0, gt=0.0)


class MultiLayerSoil(_SoilBase):
    """Multi-layer, horizontally stratified soil.

    The last entry in ``layers`` must have ``thickness=None``
    (semi-infinite). All other layers must have a finite thickness.
    """

    kind: Literal["multi_layer"] = "multi_layer"
    layers: list[SoilLayer] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_layers(self) -> "MultiLayerSoil":
        for layer in self.layers[:-1]:
            if layer.thickness is None:
                raise ValueError(
                    "Only the last layer may be semi-infinite "
                    "(thickness=None)."
                )
        if self.layers[-1].thickness is not None:
            raise ValueError(
                "The last layer must be semi-infinite (thickness=None)."
            )
        return self


# Discriminated union used inside ``World``.
SoilModel = Union[HomogeneousSoil, TwoLayerSoil, MultiLayerSoil]
