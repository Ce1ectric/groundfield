"""Soil-model specifications for generator-built worlds.

The soil model parameters in :mod:`groundfield.soil.models` are
plain numbers (Pydantic v2). For generator runs we need a *spec*
layer that carries the same fields but additionally allows
:class:`Distribution` instances so the soil can be sampled
stochastically. ``Spec.to_soil(rng)`` returns a fully numeric
:class:`SoilModel` that the solver consumes.

Three concrete spec classes mirror the three first-class soil
models:

* :class:`HomogeneousSoilSpec` — single-resistivity, half-space.
* :class:`TwoLayerSoilSpec` — finite upper layer over a
  semi-infinite lower layer. The default for AP1.
* :class:`MultiLayerSoilSpec` — arbitrary stratification, the last
  layer semi-infinite.

All three carry ``kind`` literals and form a discriminated union
:data:`SoilSpec` for use in generator configs and JSON
serialisation.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundfield.generators.distributions import AnyDistribution, Distribution
from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    SoilLayer,
    SoilModel,
    TwoLayerSoil,
)

__all__ = [
    "HomogeneousSoilSpec",
    "TwoLayerSoilSpec",
    "MultiLayerSoilSpec",
    "SoilLayerSpec",
    "SoilSpec",
]


def _to_float(value: Union[float, Distribution], rng: np.random.Generator) -> float:
    if isinstance(value, Distribution):
        return float(value.sample(rng))
    return float(value)


# ---------------------------------------------------------------------
# Concrete specs
# ---------------------------------------------------------------------


class HomogeneousSoilSpec(BaseModel):
    """Homogeneous soil with a single resistivity $\\rho$.

    Use as a sanity check (every layered backend collapses to this
    when $K = 0$) or for very simple studies.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["homogeneous"] = "homogeneous"
    resistivity: Union[float, AnyDistribution] = Field(
        default=100.0,
        description="ρ in Ω·m.",
    )
    relative_permittivity: float = Field(default=10.0, gt=0.0)

    def to_soil(self, rng: np.random.Generator) -> HomogeneousSoil:
        return HomogeneousSoil(
            resistivity=_to_float(self.resistivity, rng),
            relative_permittivity=self.relative_permittivity,
        )


class TwoLayerSoilSpec(BaseModel):
    """Two-layer model — the AP1 default.

    Upper layer of resistivity $\\rho_1$ and thickness $h_1$ over a
    semi-infinite lower layer of resistivity $\\rho_2$.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["two_layer"] = "two_layer"
    rho_1: Union[float, AnyDistribution] = Field(default=100.0, description="Upper-layer ρ in Ω·m.")
    rho_2: Union[float, AnyDistribution] = Field(default=100.0, description="Lower-layer ρ in Ω·m.")
    h_1: Union[float, AnyDistribution] = Field(default=5.0, description="Upper-layer thickness in m.")
    relative_permittivity: float = Field(default=10.0, gt=0.0)

    def to_soil(self, rng: np.random.Generator) -> TwoLayerSoil:
        return TwoLayerSoil(
            rho_1=_to_float(self.rho_1, rng),
            rho_2=_to_float(self.rho_2, rng),
            h_1=_to_float(self.h_1, rng),
            relative_permittivity=self.relative_permittivity,
        )


class SoilLayerSpec(BaseModel):
    """A single layer used by :class:`MultiLayerSoilSpec`.

    ``thickness_m=None`` marks the semi-infinite bottom layer (only
    valid as the *last* entry in :class:`MultiLayerSoilSpec.layers`).
    """

    model_config = ConfigDict(extra="forbid")

    resistivity: Union[float, AnyDistribution] = Field(default=100.0, description="ρ in Ω·m.")
    thickness_m: Optional[Union[float, AnyDistribution]] = Field(
        default=5.0,
        description="Layer thickness in m, or None for the semi-infinite bottom layer.",
    )
    relative_permittivity: float = Field(default=10.0, gt=0.0)


class MultiLayerSoilSpec(BaseModel):
    """Multi-layer horizontally stratified soil.

    The list ``layers`` is consumed top-down: the first entry is the
    surface layer, the last entry must have ``thickness_m=None``
    (semi-infinite). Each layer's resistivity (and finite-layer
    thickness) may carry a :class:`Distribution`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["multi_layer"] = "multi_layer"
    layers: list[SoilLayerSpec] = Field(default_factory=list, min_length=1)

    @model_validator(mode="after")
    def _validate_layers(self) -> "MultiLayerSoilSpec":
        for layer in self.layers[:-1]:
            if layer.thickness_m is None:
                raise ValueError(
                    "MultiLayerSoilSpec: only the last layer may have "
                    "thickness_m=None (semi-infinite)."
                )
        if self.layers[-1].thickness_m is not None:
            raise ValueError(
                "MultiLayerSoilSpec: the last layer must have "
                "thickness_m=None (semi-infinite)."
            )
        return self

    def to_soil(self, rng: np.random.Generator) -> MultiLayerSoil:
        layers: list[SoilLayer] = []
        for spec in self.layers:
            thickness: Optional[float]
            if spec.thickness_m is None:
                thickness = None
            else:
                thickness = _to_float(spec.thickness_m, rng)
            layers.append(
                SoilLayer(
                    resistivity=_to_float(spec.resistivity, rng),
                    thickness=thickness,
                    relative_permittivity=spec.relative_permittivity,
                )
            )
        return MultiLayerSoil(layers=layers)


# ---------------------------------------------------------------------
# Discriminated union + helper
# ---------------------------------------------------------------------


SoilSpec = Annotated[
    Union[HomogeneousSoilSpec, TwoLayerSoilSpec, MultiLayerSoilSpec],
    Field(discriminator="kind"),
]
"""JSON-serialisable union of soil specs (homogeneous / two-layer / multi-layer)."""


def materialise_soil(
    spec: Union[HomogeneousSoilSpec, TwoLayerSoilSpec, MultiLayerSoilSpec],
    rng: np.random.Generator,
) -> SoilModel:
    """Convenience dispatcher: ``spec.to_soil(rng)``.

    Provided so callers can write ``materialise_soil(cfg.soil, rng)``
    without having to discriminate by ``isinstance`` themselves.
    """
    return spec.to_soil(rng)
