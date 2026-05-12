"""Boundary conditions for the field computation.

The boundary configuration bundles all assumptions imposed at the
boundary of the computational domain:

- ``far_field``: behaviour at infinity (Dirichlet
  $\\varphi = 0$, Neumann $\\partial_n \\varphi = 0$). For
  grounding systems, Dirichlet at infinity is the standard assumption
  ("remote earth").
- ``surface``: condition at the soil surface ($z = 0$). Default
  is Neumann (no current flow into the air).
- ``reference_node``: optional reference node that fixes the zero
  point of the potential.

Implementation status (v0.2.0)
------------------------------
The boundary fields are accepted on every :class:`World` and round-trip
through serialisation, but the integral / image-charge backends in
``groundfield.solver`` enforce the **defaults implicitly**:

- ``far_field = "dirichlet"`` is hard-wired through the
  $1/r$ Green's-function decay.
- ``surface = "neumann"`` is hard-wired through the
  air-side image charge at $z \\to -z$.
- ``reference_node`` is **not** consulted; all potentials are reported
  relative to remote earth ($\\varphi \\to 0$ at infinity).

Setting any field to a non-default value via
:meth:`groundfield.world.World.set_boundary_conditions` therefore emits
a :class:`UserWarning`. The fields remain reserved for the future
:mod:`groundfield.solver.fem` backend, which will resolve them
explicitly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["BoundaryConditions"]


class BoundaryConditions(BaseModel):
    """Boundary-condition configuration of a :class:`World`.

    Notes
    -----
    The integral / image-charge backends in ``groundfield.solver`` use
    the defaults implicitly (``far_field="dirichlet"``,
    ``surface="neumann"``, ``reference_node=None``) and ignore any
    other value. The fields are reserved for the upcoming FEM backend.
    See the module docstring for a longer discussion.
    """

    model_config = ConfigDict(extra="forbid")

    far_field: Literal["dirichlet", "neumann"] = Field(
        default="dirichlet",
        description=(
            "Behaviour at infinity. Default: φ → 0 (remote earth). "
            "Hard-wired in the v0.2.0 integral backends; non-default "
            "values are accepted but not consumed."
        ),
    )
    surface: Literal["neumann", "dirichlet"] = Field(
        default="neumann",
        description=(
            "Boundary condition at the soil surface z = 0. Default: "
            "no current flow into air. Hard-wired in the v0.2.0 "
            "integral backends; non-default values are accepted but "
            "not consumed."
        ),
    )
    reference_node: str | None = Field(
        default=None,
        description=(
            "Name of the reference electrode for the zero potential. "
            "Not consulted by the v0.2.0 backends — all potentials "
            "are reported relative to remote earth (φ → 0 at "
            "infinity)."
        ),
    )
