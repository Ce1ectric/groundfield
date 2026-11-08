r"""Soil-model specifications for generator-built worlds.

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
  semi-infinite lower layer. The default for typical cases.
* :class:`MultiLayerSoilSpec` — arbitrary stratification, the last
  layer semi-infinite.

All three carry ``kind`` literals and form a discriminated union
:data:`SoilSpec` for use in generator configs and JSON
serialisation.

Naming
------
Spec fields of the generators carry lower-case unit suffixes (``depth_m``,
``concrete_rho_ohm_m``, ``source_magnitude_a``). The soil specs are the
one deliberate exception: a soil spec *is* its soil model with
distributions allowed, so its fields are the model's literature symbols
— resistivity :math:`\rho` (``rho``, ``rho_1``, ``rho_2``) in Ω·m and
layer thickness :math:`h` (``h_1``, ``h``) in m, as on
:class:`~groundfield.soil.models.TwoLayerSoil` since its first release —
and a spec and its model read the same (decision of the 2026.11 naming
review; ``TwoLayerSoilSpec(rho_1=..., rho_2=..., h_1=...)`` keeps its
0.15.0 spelling). The 0.15.0 names ``resistivity`` and ``thickness_m``
remain accepted as deprecated aliases (removal in 2027.5 at the
earliest).
"""

from __future__ import annotations

import sys
import warnings
from typing import Annotated, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundfield.generators._compat import rename_input_keys, warn_renamed
from groundfield.generators.distributions import AnyDistribution, Distribution
from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    SoilLayer,
    SoilModel,
    TwoLayerSoil,
)
from groundfield.utils.deprecation import InertConfigurationWarning

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


_EPS_R_DEFAULT = 10.0
"""Default relative permittivity of the specs (that of the soil models)."""


def _user_stacklevel() -> int:
    """``stacklevel`` of the first caller outside groundfield and pydantic.

    Counted from the function that calls :func:`warnings.warn` (level 1),
    i.e. the caller of this helper.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        module = str(frame.f_globals.get("__name__", ""))
        internal = module == "groundfield" or module.startswith(
            ("groundfield.", "pydantic", "pydantic_core")
        )
        if not internal:
            return level
        frame = frame.f_back
        level += 1
    return 2


def _warn_inert_permittivity(owner: str, values: list[float]) -> None:
    """Warn that a spec's non-default ``relative_permittivity`` is inert.

    The soil models warn when a user constructs them with such a value, but
    stay silent when groundfield's own code builds them; here the setting
    comes from the user's spec, so the warning goes to the user's line
    (the first caller outside groundfield) in every case (audit A1-10).
    """
    odd = [v for v in values if v != _EPS_R_DEFAULT]
    if not odd:
        return
    shown = ", ".join(f"{v:g}" for v in odd)
    warnings.warn(
        f"{owner}(relative_permittivity={shown}) is stored, but no backend "
        "reads it: every backend solves the quasi-static conduction problem "
        "below 1 kHz, where the displacement current is negligible "
        "(omega*eps0*eps_r*rho = 4.5e-3 at 1 kHz for rho = 1000 Ohm*m and "
        "eps_r = 81). The value does not change any result.",
        InertConfigurationWarning,
        stacklevel=_user_stacklevel(),
    )


# Soil-model field names before the literature-symbol convention of the
# model layer (``rho``, ``h``). ``_soil_model`` passes whichever name the
# installed model class declares, so the specs work with both.
_LEGACY_MODEL_NAMES = {"rho": "resistivity", "h": "thickness"}


def _soil_model(model_cls: type, **values: object) -> object:
    """Construct ``model_cls`` from canonical field names (``rho``, ``h``)."""
    fields = model_cls.model_fields  # type: ignore[attr-defined]
    kwargs = {
        (name if name in fields else _LEGACY_MODEL_NAMES.get(name, name)): value
        for name, value in values.items()
    }
    return model_cls(**kwargs)


def _homogeneous_soil(
    rho: float, relative_permittivity: float = 10.0
) -> HomogeneousSoil:
    """Return a :class:`~groundfield.soil.models.HomogeneousSoil`.

    Parameters
    ----------
    rho
        Resistivity in Ω·m.
    relative_permittivity
        Relative permittivity (dimensionless).
    """
    return _soil_model(  # type: ignore[return-value]
        HomogeneousSoil, rho=rho, relative_permittivity=relative_permittivity
    )


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
    rho: Union[float, AnyDistribution] = Field(
        default=100.0,
        description="Resistivity ρ in Ω·m.",
    )
    relative_permittivity: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def _renamed_fields(cls, data: object) -> object:
        return rename_input_keys(data, {"resistivity": "rho"}, "HomogeneousSoilSpec")

    @property
    def resistivity(self) -> Union[float, Distribution]:
        """Deprecated alias of :attr:`rho`."""
        warn_renamed("HomogeneousSoilSpec.resistivity", "HomogeneousSoilSpec.rho")
        return self.rho

    def to_soil(self, rng: np.random.Generator) -> HomogeneousSoil:
        """Draw the distributions and return the soil model.

        A non-default ``relative_permittivity`` warns with
        :class:`~groundfield.utils.InertConfigurationWarning` (no backend
        reads it).
        """
        _warn_inert_permittivity(type(self).__name__, [self.relative_permittivity])
        return _homogeneous_soil(
            _to_float(self.rho, rng), relative_permittivity=self.relative_permittivity
        )


class TwoLayerSoilSpec(BaseModel):
    """Two-layer model — the default.

    Upper layer of resistivity $\\rho_1$ and thickness $h_1$ over a
    semi-infinite lower layer of resistivity $\\rho_2$.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["two_layer"] = "two_layer"
    rho_1: Union[float, AnyDistribution] = Field(
        default=100.0, description="Upper-layer ρ in Ω·m."
    )
    rho_2: Union[float, AnyDistribution] = Field(
        default=100.0, description="Lower-layer ρ in Ω·m."
    )
    h_1: Union[float, AnyDistribution] = Field(
        default=5.0, description="Upper-layer thickness in m."
    )
    relative_permittivity: float = Field(default=10.0, gt=0.0)

    def to_soil(self, rng: np.random.Generator) -> TwoLayerSoil:
        """Draw the distributions and return the soil model.

        A non-default ``relative_permittivity`` warns with
        :class:`~groundfield.utils.InertConfigurationWarning` (no backend
        reads it).
        """
        _warn_inert_permittivity(type(self).__name__, [self.relative_permittivity])
        return TwoLayerSoil(
            rho_1=_to_float(self.rho_1, rng),
            rho_2=_to_float(self.rho_2, rng),
            h_1=_to_float(self.h_1, rng),
            relative_permittivity=self.relative_permittivity,
        )


class SoilLayerSpec(BaseModel):
    """A single layer used by :class:`MultiLayerSoilSpec`.

    ``h=None`` marks the semi-infinite bottom layer (only valid as the
    *last* entry in :class:`MultiLayerSoilSpec.layers`).
    """

    model_config = ConfigDict(extra="forbid")

    rho: Union[float, AnyDistribution] = Field(
        default=100.0, description="Resistivity ρ in Ω·m."
    )
    h: Optional[Union[float, AnyDistribution]] = Field(
        default=5.0,
        description="Layer thickness in m, or None for the semi-infinite bottom layer.",
    )
    relative_permittivity: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def _renamed_fields(cls, data: object) -> object:
        return rename_input_keys(
            data, {"resistivity": "rho", "thickness_m": "h"}, "SoilLayerSpec"
        )

    @property
    def resistivity(self) -> Union[float, Distribution]:
        """Deprecated alias of :attr:`rho`."""
        warn_renamed("SoilLayerSpec.resistivity", "SoilLayerSpec.rho")
        return self.rho

    @property
    def thickness_m(self) -> Optional[Union[float, Distribution]]:
        """Deprecated alias of :attr:`h`."""
        warn_renamed("SoilLayerSpec.thickness_m", "SoilLayerSpec.h")
        return self.h


class MultiLayerSoilSpec(BaseModel):
    """Multi-layer horizontally stratified soil.

    The list ``layers`` is consumed top-down: the first entry is the
    surface layer, the last entry must have ``h=None``
    (semi-infinite). Each layer's resistivity (and finite-layer
    thickness) may carry a :class:`Distribution`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["multi_layer"] = "multi_layer"
    layers: list[SoilLayerSpec] = Field(default_factory=list, min_length=1)

    @model_validator(mode="after")
    def _validate_layers(self) -> "MultiLayerSoilSpec":
        for layer in self.layers[:-1]:
            if layer.h is None:
                raise ValueError(
                    "MultiLayerSoilSpec: only the last layer may have "
                    "h=None (semi-infinite)."
                )
        if self.layers[-1].h is not None:
            raise ValueError(
                "MultiLayerSoilSpec: the last layer must have h=None (semi-infinite)."
            )
        return self

    def to_soil(self, rng: np.random.Generator) -> MultiLayerSoil:
        """Draw the distributions and return the soil model.

        Per layer the thickness is drawn before the resistivity. A
        non-default ``relative_permittivity`` of any layer warns with
        :class:`~groundfield.utils.InertConfigurationWarning` (no backend
        reads it).
        """
        _warn_inert_permittivity(
            type(self).__name__, [layer.relative_permittivity for layer in self.layers]
        )
        layers: list[SoilLayer] = []
        for spec in self.layers:
            thickness: Optional[float]
            if spec.h is None:
                thickness = None
            else:
                thickness = _to_float(spec.h, rng)
            layers.append(
                _soil_model(  # type: ignore[arg-type]
                    SoilLayer,
                    rho=_to_float(spec.rho, rng),
                    h=thickness,
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
