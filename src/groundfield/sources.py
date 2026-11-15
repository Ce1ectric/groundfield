"""Source models: impressed currents and impressed voltages.

Grounding studies are dominated by **current sources**: ground-fault
currents, test currents in loop measurements, lightning partial
currents. A source feeds a *node* of the network — an electrode or a
leakage-free junction (:class:`~groundfield.geometry.junction.Junction`)
— and, optionally, takes its current back at a second node. A
**voltage source** impresses a potential difference between the same
two terminals instead and the solve returns its current.

One-terminal and two-terminal sources
-------------------------------------
With ``return_to=None`` the impressed current $I$ enters the soil at
``attached_to`` and returns through remote earth (the classical
grounding-impedance set-up). With ``return_to`` set, the source is a
**two-terminal** source: it injects $+I$ at ``attached_to`` and $-I$
at ``return_to``, so Kirchhoff's current law closes the loop through
the soil and the conductor network (a fall-of-potential measurement
with its auxiliary electrode, a fault current returning to a
neighbouring substation). The driving-point impedance the source sees
is $(\\varphi_a - \\varphi_r)/I$
(:meth:`~groundfield.solver.result.FieldResult.input_impedance`).
Every backend whose capabilities declare ``two_terminal_sources``
honours ``return_to``; the others warn and return the current through
remote earth.

Voltage sources
---------------
A :class:`VoltageSource` is an ideal voltage $U = |U|\\,e^{j\\varphi}$
(RMS phasor, ``magnitude`` in V) between ``attached_to`` and
``return_to``, or between ``attached_to`` and remote earth (the
potential reference) for ``return_to=None``:

$$
\\varphi_a - \\varphi_r = U .
$$

Its current $I$ is an unknown of the solve (modified nodal analysis:
one constraint row and one current unknown per voltage source in the
nodal system of electrodes, junctions and conductor branches) and
follows the sign convention of a current source: it flows from the
source into ``attached_to`` and back out of ``return_to``. The result
reports it in :attr:`~groundfield.solver.result.SourceRecord.current`,
the impressed voltage in :attr:`~groundfield.solver.result.SourceRecord.voltage`,
and ``input_impedance()`` is $U / I$. The problem is linear: a voltage
source gives the solution of a 1 A current source at the same terminals
scaled by $U / Z_\\text{in}$, also with several sources, conductors,
inductive coupling and layered soil (to round-off,
``tests/test_voltage_sources.py``).

* Several voltage and current sources superpose; for one voltage
  source alone the others are shorted (0 V), the current sources open.
* A voltage source of 0 V is an ideal short between its terminals (or
  to remote earth); the solve returns the current through it.
* The source is ideal (no internal impedance). A series impedance is a
  junction at the source terminal and a finite conductor to the
  electrode (``lumped_series_resistance_ohm``, or an inductive
  conductor model).
* Rejected before any numerics run (:meth:`World.check
  <groundfield.world.World.check>`): a voltage source whose terminals
  are bonded into one cluster by ideal conductors (short circuit), and
  voltage sources that form a loop through ideal bonds or remote earth
  (two of them from one cluster to remote earth, a ring of two-terminal
  ones) — the loop fixes one potential difference twice.
* Backends without the registry capability ``voltage_sources`` (the
  experimental ``fem``) raise
  :class:`~groundfield.solver.registry.BackendCapabilityError`.

Up to 0.15.0 every backend dropped voltage sources with a warning, so a
world driven by one solved to zero potentials (review finding 10.51).

Notes
-----
:data:`Source` is a *discriminated* union — Pydantic uses the
``kind`` field to pick the correct sub-class on validation and JSON
round-trip. This means error messages for malformed source dicts point
at the specific sub-class (``CurrentSource`` / ``VoltageSource``)
selected by ``kind``, instead of dumping the entire union's validator
chain.

Assignments are validated (``validate_assignment``), so a source
handle edited between two solves (``src.magnitude = 20.0``) cannot
take a non-finite value.
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

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(..., description="Unique name within the ``World``.")
    kind: str = Field(..., description="Discriminator for the source type.")
    attached_to: str = Field(
        ...,
        description=(
            "Name of the node the source feeds: an electrode or a junction "
            "of the ``World``. A conductor is not a node; put a junction at "
            "its end instead."
        ),
    )
    return_to: str | None = Field(
        default=None,
        description=(
            "Optional second terminal: name of the electrode or junction "
            "that takes the current back (auxiliary electrode, return "
            "conductor end). ``None`` means return through remote earth "
            "(classical grounding measurement)."
        ),
    )


class CurrentSource(_SourceBase):
    """Impressed current source.

    Attributes
    ----------
    magnitude
        Current amplitude in A (RMS in the frequency domain); finite. A
        world is solved only if at least one source has a non-zero
        magnitude, and a source of magnitude 0 is rejected when solved
        (it injects nothing; remove it instead).
    phase_deg
        Phase angle in degrees; finite. Default 0.
    """

    kind: Literal["current"] = "current"
    magnitude: float = Field(..., allow_inf_nan=False, description="|I| in A.")
    phase_deg: float = Field(
        default=0.0, allow_inf_nan=False, description="Phase angle in degrees."
    )


class VoltageSource(_SourceBase):
    """Impressed (ideal) voltage source.

    Fixes the potential of ``attached_to`` at $U$ above ``return_to``,
    or above remote earth for ``return_to=None``:
    $\\varphi_a - \\varphi_r = U = |U|\\,e^{j\\varphi}$. The source
    current $I$ (from the source into ``attached_to``) is solved for
    and reported in :attr:`~groundfield.solver.result.SourceRecord.current`;
    :meth:`~groundfield.solver.result.FieldResult.input_impedance` is
    $U / I$. See the module docstring for the semantics, the checks and
    the backends that solve it.

    Attributes
    ----------
    magnitude
        Voltage amplitude in V (RMS in the frequency domain); finite.
        0 V is an ideal short (its current is still solved for); a world
        needs at least one source of non-zero magnitude.
    phase_deg
        Phase angle in degrees; finite. Default 0.
    """

    kind: Literal["voltage"] = "voltage"
    magnitude: float = Field(..., allow_inf_nan=False, description="|U| in V.")
    phase_deg: float = Field(
        default=0.0, allow_inf_nan=False, description="Phase angle in degrees."
    )


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
