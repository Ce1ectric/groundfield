"""Boundary conditions for the field computation.

The boundary configuration bundles the assumptions imposed at the
boundary of the computational domain:

- ``far_field``: behaviour at infinity (Dirichlet
  $\\varphi = 0$, Neumann $\\partial_n \\varphi = 0$). For
  grounding systems, Dirichlet at infinity is the standard assumption
  ("remote earth").
- ``surface``: condition at the soil surface ($z = 0$). Default
  is Neumann (no current flow into the air).
- ``reference_node``: optional reference node that fixes the zero
  point of the potential.

Implementation status
---------------------
**No backend reads these fields** (checked for ``image``,
``image_2layer``, ``image_nlayer``, ``cim``, ``mom``,
``mom_sommerfeld``, ``bem`` and ``fem``: a world with every field set
to a non-default value solves bit-identically, see
``tests/test_boundary_conditions.py``). Every backend imposes the
defaults implicitly:

- ``far_field = "dirichlet"`` through the $1/r$ decay of the Green's
  function (the potential vanishes at remote earth);
- ``surface = "neumann"`` through the air-side image charge at
  $z \\to -z$ (the ``fem`` backend through its natural boundary
  condition);
- ``reference_node`` is never consulted; all potentials are reported
  relative to remote earth ($\\varphi \\to 0$ at infinity).

The fields are accepted and round-trip through serialisation for
forward compatibility. Setting a non-default value warns with
:class:`~groundfield.utils.InertConfigurationWarning` — when the model
is constructed (directly, via :func:`groundfield.create_world` or from a
world JSON) and in :meth:`groundfield.world.World.set_boundary_conditions`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from groundfield.utils.deprecation import warn_inert

__all__ = ["BoundaryConditions"]

#: Values every backend implements implicitly.
_IMPLEMENTED: dict[str, Any] = {
    "far_field": "dirichlet",
    "surface": "neumann",
    "reference_node": None,
}


class BoundaryConditions(BaseModel):
    """Boundary-condition configuration of a :class:`World`.

    Attributes
    ----------
    far_field
        Condition at infinity. ``"dirichlet"`` (default): φ → 0 at
        remote earth. Not read by any backend.
    surface
        Condition at the soil surface ``z = 0``. ``"neumann"``
        (default): no current flows into the air. Not read by any
        backend.
    reference_node
        Name of an electrode whose potential would define zero. Not
        read by any backend: potentials are always relative to remote
        earth.

    Warns
    -----
    InertConfigurationWarning
        When a field is set to a value other than the implemented
        default (see the module docstring).
    """

    model_config = ConfigDict(extra="forbid")

    far_field: Literal["dirichlet", "neumann"] = Field(
        default="dirichlet",
        description=(
            "Behaviour at infinity. Default: φ → 0 (remote earth), "
            "imposed implicitly by every backend; other values are "
            "stored but not read."
        ),
    )
    surface: Literal["neumann", "dirichlet"] = Field(
        default="neumann",
        description=(
            "Boundary condition at the soil surface z = 0. Default: no "
            "current flow into air, imposed implicitly by every backend; "
            "other values are stored but not read."
        ),
    )
    reference_node: str | None = Field(
        default=None,
        description=(
            "Name of the reference electrode for the zero potential. "
            "Not read by any backend — all potentials are reported "
            "relative to remote earth (φ → 0 at infinity)."
        ),
    )

    @field_validator("far_field", "surface", "reference_node")
    @classmethod
    def _warn_inert(cls, value: Any, info: ValidationInfo) -> Any:
        name = info.field_name
        if name is not None and value != _IMPLEMENTED[name]:
            warn_inert(
                f"BoundaryConditions({name}={value!r}) is stored, but no "
                "backend reads it: every backend imposes phi -> 0 at "
                "remote earth, no current into the air at z = 0, and "
                "reports potentials relative to remote earth. The value "
                "does not change any result."
            )
        return value
