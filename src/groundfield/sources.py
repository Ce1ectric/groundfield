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
A :class:`VoltageSource` is a voltage $U = |U|\\,e^{j\\varphi}$ (RMS
phasor, ``magnitude`` in V) behind an internal series impedance
$Z_q(f) = R + j\\omega L$ (``internal_resistance_ohm``,
``internal_inductance_h``; both 0 by default, the ideal source), between
``attached_to`` and ``return_to``, or between ``attached_to`` and remote
earth (the potential reference) for ``return_to=None``:

$$
\\varphi_a - \\varphi_r + Z_q\\,I = U .
$$

Its current $I$ is an unknown of the solve (modified nodal analysis:
one constraint row and one current unknown per voltage source in the
nodal system of electrodes, junctions and conductor branches; $Z_q$ is
the diagonal entry of the row) and follows the sign convention of a
current source: it flows from the source into ``attached_to`` and back
out of ``return_to``. The result reports it in
:attr:`~groundfield.solver.result.SourceRecord.current`, the impressed
voltage in :attr:`~groundfield.solver.result.SourceRecord.voltage` and
$R$, $L$ in the record's two fields of the same names;
``input_impedance()`` is the impedance the source sees at its terminals,
$Z_\\text{in} = (\\varphi_a - \\varphi_r) / I = U / I - Z_q$ ($Z_q$
excluded). The problem is linear: a world with **one** source, a
voltage source, gives the solution of a 1 A current source at the same
terminals scaled by $I = U / (Z_q + Z_\\text{in})$, with conductors,
inductive coupling and layered soil (to round-off,
``tests/test_voltage_sources.py``, ``tests/test_voltage_source_impedance.py``).

* Several voltage and current sources superpose: for one voltage source
  alone the other voltage sources are shorted (0 V, their $Z_q$ kept),
  the current sources open. The solve is then the superposition of all
  of them, not the scaled 1 A solve of one source, and the input
  impedance of each is an apparent one: two 230 V sources on two 3 m
  rods 5 m apart (100 Ω·m) each drive 7.5 % less current than one of
  them alone ($U / Z_\\text{in}$ of the 1 A solve), and
  ``input_impedance()`` reports 8.1 % more (``image``; ``mom``: 7.5 %
  and 8.2 %).
* A voltage source of 0 V is a short through $Z_q$ between its
  terminals (or to remote earth); the solve returns the current through
  it. With $Z_q \\ne 0$ it is a passive impedance.
* $Z_q$ is evaluated at every solved frequency; $j\\omega L$ makes the
  system complex, so a galvanic world whose sources carry an inductance
  is solved once per frequency (otherwise once for all frequencies).
  Before 2026.11 a series impedance was modelled as a junction at the
  source terminal and a finite conductor to the electrode
  (``lumped_series_resistance_ohm``); the two give the same solution to
  round-off.
* Rejected before any numerics run (:meth:`World.check
  <groundfield.world.World.check>`): a voltage source whose terminals
  are bonded into one cluster by ideal conductors (short circuit, with
  or without $Z_q$), and voltage sources without internal resistance
  that form a loop through ideal bonds or remote earth (two of them from
  one cluster to remote earth, a ring of two-terminal ones) — the loop
  fixes one potential difference twice. A source with
  ``internal_resistance_ohm > 0`` opens such a loop; an inductance alone
  does not, because $Z_q$ vanishes at $f = 0$.
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

import math
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
    """Impressed voltage source, ideal or behind an internal impedance.

    Drives ``attached_to`` against ``return_to``, or against remote earth
    for ``return_to=None``, with the voltage $U = |U|\\,e^{j\\varphi}$
    behind the internal series impedance
    $Z_q(f) = R + j\\omega L$ (``internal_resistance_ohm``,
    ``internal_inductance_h``; both 0 by default, the ideal source):

    $$
    \\varphi_a - \\varphi_r = U - Z_q\\,I .
    $$

    The source current $I$ (from the source into ``attached_to``) is
    solved for and reported in
    :attr:`~groundfield.solver.result.SourceRecord.current`;
    :meth:`~groundfield.solver.result.FieldResult.input_impedance` is
    the impedance the source sees at its terminals,
    $(\\varphi_a - \\varphi_r) / I = U / I - Z_q$, without $Z_q$. See
    the module docstring for the semantics, the checks and the backends
    that solve it.

    Attributes
    ----------
    magnitude
        Voltage amplitude in V (RMS in the frequency domain); finite.
        0 V is a short through $Z_q$ (its current is still solved for);
        a world needs at least one source of non-zero magnitude.
    phase_deg
        Phase angle in degrees; finite. Default 0.
    internal_resistance_ohm
        Internal series resistance $R$ in Ω; finite, ≥ 0. Default 0.
    internal_inductance_h
        Internal series inductance $L$ in H; finite, ≥ 0. Default 0.
        It enters as $j\\omega L$ at each solved frequency and vanishes
        at $f = 0$.
    """

    kind: Literal["voltage"] = "voltage"
    magnitude: float = Field(..., allow_inf_nan=False, description="|U| in V.")
    phase_deg: float = Field(
        default=0.0, allow_inf_nan=False, description="Phase angle in degrees."
    )
    internal_resistance_ohm: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "Internal series resistance R of the source in Ω (≥ 0); "
            "0 (default) for an ideal source."
        ),
    )
    internal_inductance_h: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "Internal series inductance L of the source in H (≥ 0); "
            "0 (default) for an ideal source. Enters as jωL."
        ),
    )

    def internal_impedance(self, frequency: float) -> complex:
        """Internal impedance $Z_q = R + j\\omega L$ at ``frequency``.

        Parameters
        ----------
        frequency
            Frequency in Hz.

        Returns
        -------
        complex
            $Z_q$ in Ω, with $\\omega = 2\\pi f$; $R$ at $f = 0$ and
            ``0j`` for an ideal source.
        """
        return _internal_impedance(
            self.internal_resistance_ohm,
            self.internal_inductance_h,
            2.0 * math.pi * float(frequency),
        )


def _internal_impedance(resistance: float, inductance: float, omega: float) -> complex:
    """Series impedance $R + j\\omega L$ of a voltage source (private helper).

    One formula for the solve (per angular frequency) and the read-back
    of the result (:meth:`FieldResult.input_impedance
    <groundfield.solver.result.FieldResult.input_impedance>`), so both
    use the same floating-point value.

    Parameters
    ----------
    resistance
        $R$ in Ω.
    inductance
        $L$ in H.
    omega
        Angular frequency $\\omega = 2\\pi f$ in rad/s.

    Returns
    -------
    complex
        $Z_q$ in Ω.
    """
    return complex(float(resistance), float(omega) * float(inductance))


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
