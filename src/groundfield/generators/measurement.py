"""Measurement-setup specifications for grounding-resistance studies.

A :class:`MeasurementSetupConfig` adds the *measurement
infrastructure* of a fall-of-potential (or 4-wire) grounding-
resistance measurement to a generator-built world:

* an **auxiliary current electrode** (Hilfserder) at a configurable
  remote position, with its own grounding system (a single
  *Erdungsspieß* / driven rod, a neighbour substation's ring +
  rods, or any combination via :class:`GroundingSystemSpec`),
* a **voltage probe** (Spannungssonde) at a configurable
  intermediate position, again with its own grounding system,
* optional **metallic measurement leads** that carry the current
  from the source back to the auxiliary electrode
  (Stromeinspeiseleitung) and from the voltage probe to the
  measurement device near the substation (Spannungs-Messleitung).
  Each lead is modelled as a :class:`groundfield.Conductor` with
  configurable depth (overhead at the surface vs. buried cable),
  conductor type, and inductance model. With
  ``inductance_model="neumann"`` (default), the leads couple
  inductively to every other conductor in the world — PEN, MV
  cable shields, the parallel measurement lead — via the Neumann
  double-line integral (ADR-0004) plus the chosen earth-return
  correction (Carson / Sommerfeld, ADR-0005 / ADR-0006).

This is the configuration layer needed for the **galvanic
fall-of-potential** study (vary aux electrode position) and for
the **inductive coupling** between feed line and parallel
measurement / PEN / cable shield.

Mathematical / physical content
-------------------------------
Galvanic part: with ``cfg.measurement`` set, the
:class:`CurrentSource` injects ``I`` at the substation cluster and
draws ``-I`` at the auxiliary electrode cluster. The current flows
through the soil (and through any metallic leads, if present) back
to the auxiliary electrode — exactly the closed loop of a real
measurement. The voltage probe is a passive electrode at which the
post-processing evaluates the surface potential.

Inductive part: the metallic leads are :class:`Conductor` instances
with finite cross-section and ``inductance_model="neumann"``. Their
mutual inductance to every other conductor (parallel lead, PEN
trunk, cable shield) is computed exactly. Earth-return effects are
added on top via the engine's ``earth_inductive_model``.

Validity envelope
-----------------
* Frequency: $f \\le 1\\,\\mathrm{kHz}$ (quasi-static).
* Lead routing: each lead is modelled as a *single straight wire*
  between its two end-points (substation anchor ↔ aux/probe
  anchor). For a richer routing (multiple bends, parallel-then-
  diverging), assemble several short leads manually after
  ``build()``. The straight-line model is adequate for the typical
  questions ("how strong is the inductive coupling between the
  feed lead and the parallel measurement lead?").
* The leads' depth is the wire's $z$-coordinate at both ends; the
  wire interpolates linearly (so a 200 m long lead from
  ``z = 0`` to ``z = 0.6 m`` is in fact a slight diagonal — for
  typical distances this is below the geometry resolution).
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import Field

from groundfield.generators.base import GeneratorConfig
from groundfield.generators.distributions import AnyDistribution
from groundfield.generators.electrode_specs import RodElectrodeSpec
from groundfield.generators.grounding import GroundingSystemSpec

__all__ = [
    "MeasurementLeadConfig",
    "MeasurementInjectionConfig",
    "MeasurementProbeConfig",
    "MeasurementSetupConfig",
    "overhead_lead",
    "buried_lead",
    "single_rod_grounding",
    "neighbour_substation_grounding",
]


# ---------------------------------------------------------------------
# MeasurementLeadConfig — physical wire from one site to another
# ---------------------------------------------------------------------


class MeasurementLeadConfig(GeneratorConfig):
    """One physical measurement lead (current feed or voltage probe).

    A measurement lead is a finite-impedance :class:`Conductor` that
    connects two anchor electrodes (typically the substation cluster
    and the auxiliary or probe cluster). Two routing variants are
    common in typical studies and selectable via :attr:`depth_m`:

    * ``depth_m = 0.0`` — overhead at the surface. The classical
      default case for the Stromeinspeiseleitung of a
      fall-of-potential measurement.
    * ``depth_m > 0.0`` (e.g. 0.6 m) — buried cable. Used when the
      study models a permanently installed measurement infrastructure
      or a measurement that uses an existing buried PEN cable as
      the feed.

    The default settings (``conductor_type="bare_copper"``,
    ``coupling_to_soil="isolated"``, ``inductance_model="neumann"``)
    match a bare overhead measurement wire that does not leak
    current to the soil along its length but does generate a magnetic
    field that couples to every parallel conductor — exactly the
    inductive-coupling problem.
    """

    depth_m: Union[float, AnyDistribution] = Field(
        default=0.0,
        description=(
            "Lead depth in m. ``0`` = overhead at the surface; "
            "positive values bury the lead like a cable."
        ),
    )
    wire_radius_m: float = Field(default=0.005, description="Wire radius in m.")
    conductor_type: str = Field(
        default="bare_copper",
        description=(
            "Conductor type for the engine. 'bare_copper' for an "
            "overhead measurement lead, 'cable_shield' or 'pen' for "
            "a buried cable that doubles as a measurement lead."
        ),
    )
    coupling_to_soil: str = Field(
        default="isolated",
        description=(
            "'isolated' (no current leaks along the lead) is the "
            "default for both overhead and buried cables. Set "
            "'galvanic' only if you want to model an exposed bare "
            "overhead lead that physically touches the ground."
        ),
    )
    segment_length_m: Optional[float] = Field(
        default=5.0,
        description=(
            "Discretisation segment length in m. ``None`` keeps the "
            "lead lumped (one segment for the whole length)."
        ),
    )
    inductance_model: Optional[str] = Field(
        default="neumann",
        description=(
            "ADR-0004 inductance model: 'neumann' enables mutual "
            "inductance to every other conductor in the world. "
            "Set to ``None`` for galvanic-only studies — that "
            "disables every inductive effect."
        ),
    )


# ---------------------------------------------------------------------
# Helpers — common lead presets
# ---------------------------------------------------------------------


def overhead_lead(
    *,
    wire_radius_m: float = 0.005,
    inductance_model: Optional[str] = "neumann",
    segment_length_m: Optional[float] = 5.0,
) -> MeasurementLeadConfig:
    """Convenience factory for an overhead bare-copper measurement lead.

    Surface routing (``depth_m=0``), bare copper, isolated soil
    coupling, Neumann inductance enabled.
    """
    return MeasurementLeadConfig(
        depth_m=0.0,
        wire_radius_m=wire_radius_m,
        conductor_type="bare_copper",
        coupling_to_soil="isolated",
        segment_length_m=segment_length_m,
        inductance_model=inductance_model,
    )


def buried_lead(
    *,
    depth_m: float = 0.6,
    wire_radius_m: float = 0.005,
    inductance_model: Optional[str] = "neumann",
    segment_length_m: Optional[float] = 5.0,
) -> MeasurementLeadConfig:
    """Convenience factory for a buried-cable measurement lead.

    Default depth 0.6 m (typical NS cable trench). Insulated cable
    (no soil leakage along the lead), Neumann inductance enabled.
    """
    return MeasurementLeadConfig(
        depth_m=depth_m,
        wire_radius_m=wire_radius_m,
        conductor_type="cable_shield",
        coupling_to_soil="isolated",
        segment_length_m=segment_length_m,
        inductance_model=inductance_model,
    )


def single_rod_grounding(
    *,
    length_m: Union[float, AnyDistribution] = 1.5,
    depth_m: Union[float, AnyDistribution] = 0.0,
) -> GroundingSystemSpec:
    """One driven rod — a typical *Erdungsspieß* used as an aux electrode."""
    return GroundingSystemSpec(
        electrodes=[
            RodElectrodeSpec(length_m=length_m, depth_m=depth_m),
        ],
    )


def neighbour_substation_grounding() -> GroundingSystemSpec:
    """A neighbour substation's typical grounding (ring + 4 rods).

    Used as the auxiliary electrode in measurements where the
    current returns through a remote, well-grounded substation
    instead of a single driven rod.
    """
    # Imported here to avoid a top-level circular import:
    # tn_network depends on this module.
    from groundfield.generators.electrode_specs import (
        RingElectrodeSpec,
        rod_circle,
    )
    return GroundingSystemSpec(
        electrodes=[
            RingElectrodeSpec(radius_m=4.0, depth_m=0.6),
            *rod_circle(n=4, radius_m=2.0, length_m=2.5),
        ],
    )


# ---------------------------------------------------------------------
# Aux + probe configs
# ---------------------------------------------------------------------


class MeasurementInjectionConfig(GeneratorConfig):
    """Auxiliary current electrode + optional metallic feed lead.

    The auxiliary electrode (*Hilfserder*) is the remote electrode
    through which the test current returns to the source. Default:
    a single driven rod at $(200, 0)$ m. For studies that model the
    measurement against a neighbouring substation, point ``grounding``
    at :func:`neighbour_substation_grounding`.

    ``feed_lead`` is the metallic *Stromeinspeiseleitung* between
    the substation cluster and the aux electrode. ``None`` (default)
    means *galvanic-only*: the source's ``return_to`` is set to the
    aux electrode but no metallic wire closes the loop — the return
    current flows entirely through the soil. Setting ``feed_lead``
    to a :class:`MeasurementLeadConfig` (often via
    :func:`overhead_lead` or :func:`buried_lead`) adds the physical
    wire and enables the inductive coupling.
    """

    position_xy: tuple[float, float] = Field(
        default=(200.0, 0.0),
        description="Auxiliary electrode position (x, y) in m.",
    )
    grounding: GroundingSystemSpec = Field(
        default_factory=single_rod_grounding,
        description="Auxiliary electrode grounding system.",
    )
    feed_lead: Optional[MeasurementLeadConfig] = Field(
        default=None,
        description=(
            "Optional metallic feed lead from the substation to the "
            "auxiliary electrode. ``None`` = no wire (galvanic only); "
            "a :class:`MeasurementLeadConfig` adds the wire and "
            "enables inductive coupling."
        ),
    )


class MeasurementProbeConfig(GeneratorConfig):
    """Voltage probe + optional metallic measurement lead.

    The voltage probe (*Spannungssonde*) is the second remote
    electrode; the post-processing evaluates the potential at its
    location. Default: a short (0.5 m) driven rod at $(60, 0)$ m —
    the classical 62 % point in fall-of-potential measurements.

    ``lead`` is the metallic measurement wire from the probe back
    to the measurement device near the substation. As with
    :class:`MeasurementInjectionConfig.feed_lead`, ``None`` is
    galvanic-only; setting it enables inductive coupling.
    """

    position_xy: tuple[float, float] = Field(
        default=(60.0, 0.0),
        description="Voltage probe position (x, y) in m.",
    )
    grounding: GroundingSystemSpec = Field(
        default_factory=lambda: single_rod_grounding(length_m=0.5),
        description="Voltage probe grounding system (typically a short rod).",
    )
    lead: Optional[MeasurementLeadConfig] = Field(
        default=None,
        description=(
            "Optional metallic measurement lead from the probe to "
            "the substation-side device. ``None`` for galvanic-only."
        ),
    )


# ---------------------------------------------------------------------
# Top-level measurement-setup config
# ---------------------------------------------------------------------


class MeasurementSetupConfig(GeneratorConfig):
    """Earth-resistance measurement setup.

    When ``TnNetworkConfig.measurement`` is set, the generator:

    1. builds the auxiliary electrode at
       :attr:`injection.position_xy` via
       ``injection.grounding.build_at(...)``;
    2. builds the voltage probe at :attr:`probe.position_xy` via
       ``probe.grounding.build_at(...)``;
    3. if :attr:`injection.feed_lead` is set, adds a
       :class:`Conductor` from the substation anchor to the aux
       anchor with the configured depth, conductor type, and
       inductance model;
    4. if :attr:`probe.lead` is set, adds a similar conductor from
       the substation anchor to the probe anchor;
    5. attaches the source to the substation anchor with
       ``return_to`` pointing at the aux anchor — the test current
       physically returns through the auxiliary electrode (and,
       if the metallic feed lead is present, mostly through it).

    The default factory leaves both leads as ``None`` (galvanic
    only). To enable inductive coupling, set, e.g.,

    >>> measurement = MeasurementSetupConfig(
    ...     injection=MeasurementInjectionConfig(
    ...         position_xy=(200.0, 0.0),
    ...         feed_lead=overhead_lead(),
    ...     ),
    ...     probe=MeasurementProbeConfig(
    ...         position_xy=(60.0, 0.0),
    ...         lead=overhead_lead(),
    ...     ),
    ... )
    """

    injection: MeasurementInjectionConfig = Field(
        default_factory=MeasurementInjectionConfig,
        description="Auxiliary current electrode block.",
    )
    probe: MeasurementProbeConfig = Field(
        default_factory=MeasurementProbeConfig,
        description="Voltage probe block.",
    )
