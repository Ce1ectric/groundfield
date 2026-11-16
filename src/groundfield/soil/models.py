"""Data classes for soil models.

The classes defined here are pure data containers (Pydantic v2). They
describe the geometry and electromagnetic parameters of the soil. The
actual evaluation (Green's function, image-charge sum, reflection
coefficient) happens inside :mod:`groundfield.solver` and consumes the
data exposed here.

Physical model
--------------
The soil is a horizontally stratified half-space ``z > 0`` (``z`` points
downwards) below a non-conducting air half-space. Each layer is
characterised by its resistivity $\\rho$ in $\\Omega\\,\\mathrm{m}$; the
potential obeys $\\nabla \\cdot (\\rho^{-1} \\nabla \\varphi) = 0$ in the
soil with $\\partial_z \\varphi = 0$ at the surface and continuity of
$\\varphi$ and of the normal current density at every interface. For a
two-layer soil the image-series coefficient is the reflection
coefficient $K = (\\rho_2 - \\rho_1) / (\\rho_2 + \\rho_1)$.

Below 1 kHz the formulation is quasi-static: the displacement current
$\\omega \\varepsilon_0 \\varepsilon_r$ is neglected against the
conduction current $1/\\rho$. Their ratio
$\\omega \\varepsilon_0 \\varepsilon_r \\rho$ is $4.5 \\cdot 10^{-3}$ at
1 kHz for $\\rho = 1000\\ \\Omega\\,\\mathrm{m}$ and
$\\varepsilon_r = 81$; only very resistive rock
($\\rho = 10\\ \\mathrm{k\\Omega\\,m}$) reaches $4.5 \\cdot 10^{-2}$ at
1 kHz, a limit of the quasi-static model itself. No backend reads
``relative_permittivity``; setting a non-default value therefore warns
with :class:`~groundfield.utils.InertConfigurationWarning`.

Parameter names
---------------
The model layer uses bare SI units and the literature symbols:
``rho`` (Ω·m) for a resistivity, ``rho_1`` / ``rho_2`` / ``h_1`` for the
two-layer model and ``h`` (m) for the thickness of a :class:`SoilLayer`
(``None`` = semi-infinite bottom layer). The names used up to 0.15.0,
``resistivity`` and ``thickness``, are deprecated aliases: they are
accepted as keywords, as JSON keys (worlds written by 0.15.0 load
unchanged), as attributes and in ``model_copy(update=...)``, and warn
with :class:`DeprecationWarning`. They will be removed in groundfield
2027.5 at the earliest.

Validation
----------
Resistivities, thicknesses and the relative permittivity must be finite
and positive; ``inf`` and ``nan`` are rejected with the parameter name
(an infinite resistivity used to fail much later inside the solver).
Assignments are validated too (``validate_assignment=True``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    ModelWrapValidatorHandler,
    Tag,
    ValidationError,
    field_validator,
    model_validator,
)

from groundfield.utils.deprecation import warn_deprecated, warn_inert

if TYPE_CHECKING:  # pragma: no cover - type-hint imports only
    from pydantic_core import InitErrorDetails
from groundfield.utils.suggest import did_you_mean

__all__ = [
    "SoilModel",
    "HomogeneousSoil",
    "TwoLayerSoil",
    "MultiLayerSoil",
    "SoilLayer",
]

#: Default relative permittivity of every soil model (mid-range value for
#: soil, IEEE Std 80). Not used by any backend, see the module docstring.
_EPS_R_DEFAULT = 10.0


def _inert_permittivity(owner: str, value: float) -> float:
    """Warn when ``relative_permittivity`` is set to a non-default value."""
    if value != _EPS_R_DEFAULT:
        warn_inert(
            f"{owner}(relative_permittivity={value:g}) is stored, but no "
            "backend reads it: every backend solves the quasi-static "
            "conduction problem below 1 kHz, where the displacement "
            "current is negligible (omega*eps0*eps_r*rho = 4.5e-3 at "
            "1 kHz for rho = 1000 Ohm*m and eps_r = 81). The value does "
            "not change any result."
        )
    return value


class _ModelLayerBase(BaseModel):
    """Shared configuration and deprecated-name handling of the soil models.

    Subclasses list their renamed fields in ``_renamed`` (old name →
    canonical name). This base class translates an old name used as a
    keyword or JSON key to the canonical field with a
    :class:`DeprecationWarning`, rejects a call that passes both names,
    and translates old names in :meth:`model_copy` (which pydantic does
    not validate).

    The translation happens in a model validator, not through
    ``validation_alias=AliasChoices(...)``: with an alias choice the
    pydantic mypy plugin types ``__init__`` as ``**kwargs`` and a
    misspelled keyword is no longer a type error (audit A1-09). Type
    checkers therefore report the deprecated keywords as well. A value
    given under a deprecated name that fails validation is reported
    under that name, as before.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    _renamed: ClassVar[dict[str, str]] = {}

    @model_validator(mode="wrap")
    @classmethod
    def _accept_deprecated_names(
        cls, data: Any, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        if not isinstance(data, Mapping):
            return handler(data)
        found = cls._check_deprecated_names(data)
        if not found:
            return handler(data)
        given = {cls._renamed[old]: old for old in found}  # canonical -> given
        data = {cls._renamed.get(k, k): v for k, v in data.items()}
        try:
            return handler(data)
        except ValidationError as exc:
            errors: list[InitErrorDetails] = []
            for e in exc.errors():
                loc = e["loc"]
                if loc and loc[0] in given:
                    loc = (given[str(loc[0])], *loc[1:])
                detail: InitErrorDetails = {
                    "type": e["type"],
                    "loc": loc,
                    "input": e["input"],
                }
                if "ctx" in e:
                    detail["ctx"] = e["ctx"]
                errors.append(detail)
            try:
                renamed = ValidationError.from_exception_data(exc.title, errors)
            except Exception:  # noqa: BLE001 - keep the original on any doubt
                raise exc from None
            raise renamed from None

    @classmethod
    def _check_deprecated_names(cls, data: Mapping[str, Any]) -> list[str]:
        """Warn for old names in ``data``; return the old names found."""
        found = []
        for old, new in cls._renamed.items():
            if old not in data:
                continue
            if new in data:
                raise ValueError(
                    f"{cls.__name__}: got both '{new}' and its deprecated "
                    f"alias '{old}'; pass only '{new}'."
                )
            warn_deprecated(f"{cls.__name__}: '{old}'", f"'{new}'")
            found.append(old)
        return found

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy the model; old field names in ``update`` are translated.

        Parameters
        ----------
        update
            Field values to change in the copy (not validated, as in
            pydantic). Deprecated names such as ``resistivity`` are
            mapped to their canonical field with a
            :class:`DeprecationWarning`.
        deep
            Deep copy instead of a shallow one.

        Returns
        -------
        Self
            The copy.

        Raises
        ------
        ValueError
            A key of ``update`` is not a field of the model (pydantic
            would silently store it next to the fields, so the copy
            would keep the old value).
        """
        if update:
            data = dict(update)
            for old in self._check_deprecated_names(data):
                data[self._renamed[old]] = data.pop(old)
            fields = type(self).model_fields
            unknown = [key for key in data if key not in fields]
            if unknown:
                key = unknown[0]
                raise ValueError(
                    f"{type(self).__name__}.model_copy: {key!r} is not a "
                    f"field.{did_you_mean(key, fields)} Fields: "
                    f"{', '.join(fields)}."
                )
            update = data
        return super().model_copy(update=update, deep=deep)


class _SoilBase(_ModelLayerBase):
    """Common base for all soil models.

    Defines the discriminator field ``kind``; concrete soil models
    override it with a literal value.
    """

    kind: str = Field(..., description="Discriminator for the soil model.")


class HomogeneousSoil(_SoilBase):
    """Homogeneous soil: a half-space of uniform resistivity.

    The potential of a point current source ``I`` at depth ``d`` is the
    source term plus its image above the surface,
    $\\varphi = \\frac{\\rho I}{4\\pi}\\left(\\frac{1}{r} +
    \\frac{1}{r'}\\right)$.

    Attributes
    ----------
    rho
        Resistivity $\\rho$ in $\\Omega\\,\\mathrm{m}$; finite and
        positive. Deprecated alias: ``resistivity``.
    relative_permittivity
        Relative permittivity $\\varepsilon_r$ (dimensionless). Default
        10. Not read by any backend (quasi-static formulation); a
        non-default value warns.
    """

    _renamed: ClassVar[dict[str, str]] = {"resistivity": "rho"}

    kind: Literal["homogeneous"] = "homogeneous"
    rho: float = Field(
        ...,
        gt=0.0,
        allow_inf_nan=False,
        description="Resistivity ρ in Ω·m.",
    )
    relative_permittivity: float = Field(
        default=_EPS_R_DEFAULT, gt=0.0, allow_inf_nan=False
    )

    @field_validator("relative_permittivity")
    @classmethod
    def _warn_permittivity(cls, value: float) -> float:
        return _inert_permittivity(cls.__name__, value)

    @property
    def resistivity(self) -> float:
        """Deprecated alias of :attr:`rho` (Ω·m); removed in 2027.5 at the earliest."""
        warn_deprecated("HomogeneousSoil: 'resistivity'", "'rho'")
        return self.rho

    @resistivity.setter
    def resistivity(self, value: float) -> None:
        warn_deprecated("HomogeneousSoil: 'resistivity'", "'rho'")
        self.rho = value


class TwoLayerSoil(_SoilBase):
    """Two-layer soil: an upper layer of thickness ``h_1`` over a half-space.

    The Tagg / Sunde image series of the two-layer model is driven by the
    reflection coefficient
    $K = (\\rho_2 - \\rho_1) / (\\rho_2 + \\rho_1) \\in (-1, 1)$.

    Attributes
    ----------
    rho_1
        Resistivity of the upper layer $\\rho_1$ in
        $\\Omega\\,\\mathrm{m}$; finite and positive.
    rho_2
        Resistivity of the lower (semi-infinite) layer $\\rho_2$
        in $\\Omega\\,\\mathrm{m}$; finite and positive.
    h_1
        Thickness of the upper layer $h_1$ in m; finite and positive.
        (For $h_1 \\to \\infty$ use :class:`HomogeneousSoil` with
        ``rho=rho_1``.)
    relative_permittivity
        Common relative permittivity of both layers (dimensionless).
        Not read by any backend; a non-default value warns.
    """

    kind: Literal["two_layer"] = "two_layer"
    rho_1: float = Field(..., gt=0.0, allow_inf_nan=False)
    rho_2: float = Field(..., gt=0.0, allow_inf_nan=False)
    h_1: float = Field(..., gt=0.0, allow_inf_nan=False)
    relative_permittivity: float = Field(
        default=_EPS_R_DEFAULT, gt=0.0, allow_inf_nan=False
    )

    @field_validator("relative_permittivity")
    @classmethod
    def _warn_permittivity(cls, value: float) -> float:
        return _inert_permittivity(cls.__name__, value)

    @property
    def reflection_coefficient(self) -> float:
        """Reflection coefficient $K = (\\rho_2 - \\rho_1) / (\\rho_2 + \\rho_1)$."""
        return (self.rho_2 - self.rho_1) / (self.rho_2 + self.rho_1)


class SoilLayer(_ModelLayerBase):
    """A single horizontal layer of a :class:`MultiLayerSoil`.

    Attributes
    ----------
    rho
        Resistivity $\\rho$ of the layer in $\\Omega\\,\\mathrm{m}$;
        finite and positive. Deprecated alias: ``resistivity``.
    h
        Layer thickness in m, finite and positive; ``None`` (default)
        marks the semi-infinite bottom layer. Deprecated alias:
        ``thickness``.
    relative_permittivity
        Relative permittivity (dimensionless). Not read by any backend;
        a non-default value warns.
    """

    _renamed: ClassVar[dict[str, str]] = {"resistivity": "rho", "thickness": "h"}

    rho: float = Field(
        ...,
        gt=0.0,
        allow_inf_nan=False,
        description="Resistivity ρ in Ω·m.",
    )
    h: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description="Layer thickness in m; None for the semi-infinite bottom layer.",
    )
    relative_permittivity: float = Field(
        default=_EPS_R_DEFAULT, gt=0.0, allow_inf_nan=False
    )

    @field_validator("relative_permittivity")
    @classmethod
    def _warn_permittivity(cls, value: float) -> float:
        return _inert_permittivity(cls.__name__, value)

    @property
    def resistivity(self) -> float:
        """Deprecated alias of :attr:`rho` (Ω·m); removed in 2027.5 at the earliest."""
        warn_deprecated("SoilLayer: 'resistivity'", "'rho'")
        return self.rho

    @resistivity.setter
    def resistivity(self, value: float) -> None:
        warn_deprecated("SoilLayer: 'resistivity'", "'rho'")
        self.rho = value

    @property
    def thickness(self) -> float | None:
        """Deprecated alias of :attr:`h` (m); removed in 2027.5 at the earliest."""
        warn_deprecated("SoilLayer: 'thickness'", "'h'")
        return self.h

    @thickness.setter
    def thickness(self, value: float | None) -> None:
        warn_deprecated("SoilLayer: 'thickness'", "'h'")
        self.h = value


class MultiLayerSoil(_SoilBase):
    """Horizontally stratified soil with an arbitrary number of layers.

    ``layers`` runs from the surface downwards. Every layer but the last
    has a finite thickness ``h``; the last one is the semi-infinite
    bottom half-space (``h=None``). A single-layer stack is a valid,
    if roundabout, homogeneous half-space.

    Attributes
    ----------
    layers
        The layers, top to bottom (at least one).
    """

    kind: Literal["multi_layer"] = "multi_layer"
    layers: list[SoilLayer] = Field(...)

    @model_validator(mode="after")
    def _validate_layers(self) -> Self:
        n = len(self.layers)
        if n == 0:
            raise ValueError(
                "MultiLayerSoil needs at least one layer: the bottom "
                "half-space SoilLayer(rho=..., h=None)."
            )
        if n == 1 and self.layers[0].h is not None:
            raise ValueError(
                "MultiLayerSoil with a single layer describes a homogeneous "
                "half-space, so its only layer must have h=None (got "
                f"h={self.layers[0].h:g} m). For a homogeneous soil use "
                f"HomogeneousSoil(rho={self.layers[0].rho:g})."
            )
        for i, layer in enumerate(self.layers[:-1], start=1):
            if layer.h is None:
                raise ValueError(
                    f"MultiLayerSoil: layer {i} of {n} has h=None, but only "
                    "the last layer may be the semi-infinite half-space; give "
                    "it a finite thickness h in m."
                )
        last = self.layers[-1]
        if last.h is not None:
            raise ValueError(
                f"MultiLayerSoil: the last layer (layer {n} of {n}) must be "
                f"the bottom half-space with h=None, got h={last.h:g} m."
            )
        return self


def _soil_kind(value: Any) -> str | None:
    """Discriminator of :data:`SoilModel`; infers ``kind`` when it is missing."""
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind is not None:
            return str(kind)
        if "layers" in value:
            return "multi_layer"
        if any(k in value for k in ("rho_1", "rho_2", "h_1")):
            return "two_layer"
        return "homogeneous"
    return getattr(value, "kind", None)


SoilModel = Annotated[
    Union[
        Annotated[HomogeneousSoil, Tag("homogeneous")],
        Annotated[TwoLayerSoil, Tag("two_layer")],
        Annotated[MultiLayerSoil, Tag("multi_layer")],
    ],
    Discriminator(_soil_kind),
]
"""Union of the soil models, discriminated by ``kind``.

:class:`HomogeneousSoil` (``"homogeneous"``), :class:`TwoLayerSoil`
(``"two_layer"``) and :class:`MultiLayerSoil` (``"multi_layer"``). The
kind is inferred from the keys when a plain mapping omits it. Used by
``World.soil``.
"""
