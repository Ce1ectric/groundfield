"""Leakage-free junction nodes of the conductor network.

A :class:`Junction` is a named point at which conductors meet and to
which sources can be attached, **without** contact to the soil: it
injects no current into the ground, so in the nodal system of the
solver it is a node with a Kirchhoff current-law row but no leakage
owner,

$$
\\sum_{b \\ni J} \\pm I_b \\;=\\; I_{\\text{in},J},
\\qquad
\\varphi_a - \\varphi_b = Z_b\\, I_b \\ \\text{along every branch } b,
$$

and its potential $\\varphi_J$ follows from the branch equations. It
has no geometry of its own besides its position, which the conductors
ending there use as their end point.

Typical uses:

* a **cable joint** of a PEN network (replaces the 5 cm pseudo-rods the
  generators used up to 0.15.0, which leak 3–7 % of a test current,
  finding 10.38);
* the **terminals of a measuring instrument**: the feed lead of a
  fall-of-potential measurement runs from the auxiliary electrode to a
  junction at the instrument and the source drives the current between
  the earthing system and that junction; the probe lead ends on a
  second junction whose potential is what the voltmeter reads
  (finding 10.17 — up to 0.15.0 the leads had to end on an electrode,
  which short-circuited the instrument);
* the far end of a conductor that only exchanges current through other
  conductors.

Every junction must be connected through conductors to something with
soil contact (an electrode, or a galvanically coupled distributed
conductor). A subnetwork of junctions without soil contact has an
undefined potential and is rejected by the solver.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["Junction"]

Point3D = tuple[float, float, float]


class Junction(BaseModel):
    """Named leakage-free network node.

    Attributes
    ----------
    name
        Unique node name within the ``World`` (shared namespace with the
        electrodes).
    kind
        Always ``"junction"``.
    position
        Location ``(x, y, z)`` in m, ``z`` positive downwards. Above the
        surface (``z < 0``) is allowed: a junction has no soil contact.
        Conductors ending on the junction use it as their end point.

    Notes
    -----
    Conductors refer to a junction by name through
    ``Conductor.start_electrode`` / ``end_electrode``, sources through
    ``attached_to`` / ``return_to``. Results report the junction
    potential per frequency in ``FieldResult.metadata
    ["junction_potentials"][name]`` (V).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique node name within the ``World``.")
    kind: Literal["junction"] = "junction"
    position: Point3D = Field(..., description="Location (x, y, z) in m.")

    @field_validator("position")
    @classmethod
    def _finite_position(cls, value: Point3D) -> Point3D:
        if not all(math.isfinite(float(c)) for c in value):
            raise ValueError(
                f"position must have finite coordinates (m), got {value!r}"
            )
        return value

    @property
    def connection_point(self) -> Point3D:
        """End point used by conductors that start or end here (m)."""
        return self.position
