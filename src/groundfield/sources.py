"""Source models for current injection.

Grounding studies are dominated by **current sources**: ground-fault
currents, test currents in loop measurements, lightning partial
currents. Voltage sources are rare and are provided here as an
optional subclass for completeness.

A source is attached to an electrode (or a conductor). The actual
distribution of the injected current onto the discretised segments is
done by the solver.

Notes
-----
:data:`Source` is a *discriminated* union — Pydantic uses the
``kind`` field to pick the correct sub-class on validation and JSON
round-trip. This means error messages for malformed source dicts point
at the specific sub-class (``CurrentSource`` / ``VoltageSource``)
selected by ``kind``, instead of dumping the entire union's validator
chain.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter

__all__ = [
    "Source",
    "CurrentSource",
    "VoltageSource",
    "SourceAdapter",
]


class _SourceBase(BaseModel):
    """Common base class for all source models."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique name within the ``World``.")
    kind: str = Field(..., description="Discriminator for the source type.")
    attached_to: str = Field(
        ...,
        description=(
            "Name of the electrode or conductor where the source feeds in."
        ),
    )
    return_to: str | None = Field(
        default=None,
        description=(
            "Optional: name of the return-path / auxiliary electrode. "
            "``None`` means return through the remote earth (classical "
            "grounding measurement)."
        ),
    )


class CurrentSource(_SourceBase):
    """Impressed current source.

    Parameters
    ----------
    magnitude
        Current amplitude in A (RMS in the frequency domain).
    phase_deg
        Phase angle in degrees. Default 0.
    """

    kind: Literal["current"] = "current"
    magnitude: float = Field(..., description="|I| in A.")
    phase_deg: float = Field(default=0.0, description="Phase angle in degrees.")


class VoltageSource(_SourceBase):
    """Impressed voltage source (special case).

    Rarely used, kept here for full multi-port tests.
    """

    kind: Literal["voltage"] = "voltage"
    magnitude: float = Field(..., description="|U| in V.")
    phase_deg: float = Field(default=0.0)


#: Discriminated union over the concrete source subclasses. The
#: ``kind`` field selects the validator at validation time, which keeps
#: Pydantic error messages on the actual sub-class instead of the union.
Source = Annotated[
    Union[CurrentSource, VoltageSource],
    Discriminator("kind"),
]

#: Convenience :class:`TypeAdapter` for the discriminated union, used by
#: code paths that need to validate a stand-alone source dict (e.g. CLI
#: imports, JSON round-trips). ``World.sources`` uses the annotation
#: directly via Pydantic field generation.
SourceAdapter: TypeAdapter[Source] = TypeAdapter(Source)
