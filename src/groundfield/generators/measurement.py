r"""Measurement-setup specifications for grounding-resistance studies.

A :class:`MeasurementSetupConfig` adds the *measurement
infrastructure* of a fall-of-potential (or 4-wire) grounding-
resistance measurement to a generator-built world:

* an **auxiliary current electrode** (Hilfserder) at a configurable
  remote position, with its own grounding system (a single
  *Erdungsspieß* / driven rod, a neighbour substation's ring +
  rods, or any combination via :class:`GroundingSystemSpec`),
* a **voltage probe** (Spannungssonde) at a configurable
  intermediate position, again with its own grounding system,
* optional **metallic measurement leads**: the current feed lead
  (Stromeinspeiseleitung) from the instrument to the auxiliary
  electrode and the probe lead (Spannungs-Messleitung) from the
  instrument to the voltage probe. Each lead is modelled as a
  :class:`groundfield.Conductor` with configurable depth (overhead at
  the surface vs. buried cable), conductor type, and inductance model.
  With ``inductance_model="neumann"`` (default), the leads couple
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
The instrument (an earth tester with terminals E, H and S) stands at
the substation position. Its current source drives the test current
:math:`I` from terminal H to terminal E: E is the substation cluster
(the source's ``attached_to``), H is the end of the current loop (the
source's ``return_to``) — the junction :data:`MEASUREMENT_H_TERMINAL`
at which the feed lead ends, or the auxiliary electrode itself when no
feed lead is modelled. The current flows through the soil from the
substation earthing system to the auxiliary electrode and back through
the feed lead, exactly the closed loop of a real measurement. The
voltmeter reads

.. math::

   Z_\text{meas} = \frac{\varphi_E - \varphi_S}{I},

with S the junction :data:`MEASUREMENT_S_TERMINAL` at which the probe
lead ends (or the probe electrode itself without a probe lead); see
:func:`measured_impedance`. The terminals are leakage-free
:class:`~groundfield.geometry.junction.Junction` nodes, so neither lead
bonds its remote electrode to the substation (up to 0.15.0 both leads
started on the substation electrode, which short-circuited the
voltmeter, review finding 10.17).

Inductive part: the metallic leads are :class:`Conductor` instances
with finite cross-section and ``inductance_model="neumann"``. Their
mutual inductance to every other conductor (parallel lead, PEN
trunk, cable shield) is computed exactly. Earth-return effects are
added on top via the engine's ``earth_inductive_model``. The voltage
induced in the probe lead appears in the potential of terminal S and
therefore in the reading.

Validity envelope
-----------------
* Frequency: $f \le 1\,\mathrm{kHz}$ (quasi-static).
* Lead routing: each lead is modelled as a *single straight wire*
  from its instrument terminal to the remote anchor. For a richer
  routing (multiple bends, parallel-then-diverging), assemble several
  short leads manually after ``build()``. The straight-line model is
  adequate for the typical questions ("how strong is the inductive
  coupling between the feed lead and the parallel measurement lead?").
  With both leads present, terminal S lies the touching distance (sum
  of the two wire radii) to the left of terminal H, seen towards the
  probe: an in-line probe lead then runs beside the feed lead instead
  of through it (coincident filaments have no finite mutual
  inductance). Place the probe off the auxiliary axis to model a wider
  lead separation.
* Lead height: :attr:`MeasurementLeadConfig.depth_m` is the
  :math:`z` coordinate of the lead axis (positive into the soil,
  negative above ground). The lead runs horizontally at that depth
  from its terminal (at the same height) to the horizontal position of
  the remote anchor electrode and is electrically connected to it; the
  short vertical riser to the electrode head is not modelled.
  ``depth_m = 0`` ("on the surface") would put the wire into the
  soil-surface plane, where source and image coincide; such a lead is
  placed :data:`SURFACE_LEAD_HEIGHT_M` above the ground (isolated lead)
  or below it (galvanic lead, which must touch the soil).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

from pydantic import Field, model_validator

from groundfield.generators.base import GeneratorConfig
from groundfield.generators.distributions import AnyDistribution
from groundfield.generators.electrode_specs import RodElectrodeSpec
from groundfield.generators.grounding import GroundingSystemSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from groundfield.solver.result import FieldResult, SourceRecord

__all__ = [
    "MEASUREMENT_H_TERMINAL",
    "MEASUREMENT_S_TERMINAL",
    "SURFACE_LEAD_HEIGHT_M",
    "lead_axis_z",
    "measured_impedance",
    "MeasurementLeadConfig",
    "MeasurementInjectionConfig",
    "MeasurementProbeConfig",
    "MeasurementSetupConfig",
    "overhead_lead",
    "buried_lead",
    "single_rod_grounding",
    "neighbour_substation_grounding",
]


MEASUREMENT_H_TERMINAL = "meas_H"
"""Name of the instrument's current terminal (a junction).

Created when :attr:`MeasurementInjectionConfig.feed_lead` is set: the feed
lead runs from this junction to the auxiliary electrode and the current
source of the measurement returns its current here."""

MEASUREMENT_S_TERMINAL = "meas_S"
"""Name of the instrument's voltage terminal (a junction).

Created when :attr:`MeasurementProbeConfig.lead` is set: the probe lead
runs from this junction to the voltage probe; the voltmeter reads the
potential difference between the earthing system and this junction."""

#: Source kinds of a measurement: the current source, or since 2026.11 a
#: voltage source with its solved current (audit W2A-10).
_SOURCE_KINDS = ("current", "voltage")

_PROBE_ELECTRODE = re.compile(r"^probe_(rod|ring|strip|foundation)_\d+$")
"""Electrode names of the voltage-probe site of a generated network."""


# ---------------------------------------------------------------------
# MeasurementLeadConfig — physical wire from one site to another
# ---------------------------------------------------------------------


SURFACE_LEAD_HEIGHT_M = 0.1
"""Distance in m of a lead "on the surface" (``depth_m = 0``) from the ground plane.

A wire exactly in the soil-surface plane is singular for the image
kernels (source and image coincide), and the Sommerfeld earth-return
kernel degrades for :math:`|z|` below about ten wire radii. 0.1 m keeps a
5 mm lead clear of both; the earth-return impedance of a surface lead is
insensitive to the exact height while it is small against the skin depth
(hundreds of metres below 1 kHz)."""


def lead_axis_z(depth_m: float, coupling_to_soil: str = "isolated") -> float:
    """:math:`z` coordinate in m of a measurement lead's axis.

    Parameters
    ----------
    depth_m
        Configured depth in m: positive into the soil, negative above
        ground, ``0`` on the surface.
    coupling_to_soil
        ``"isolated"`` or ``"galvanic"``.

    Returns
    -------
    float
        ``depth_m``; for ``depth_m == 0`` the lead lies
        :data:`SURFACE_LEAD_HEIGHT_M` above the ground (isolated) or
        below it (galvanic: the bare wire touches the soil).
    """
    if depth_m != 0.0:
        return float(depth_m)
    if coupling_to_soil == "galvanic":
        return SURFACE_LEAD_HEIGHT_M
    return -SURFACE_LEAD_HEIGHT_M


# ---------------------------------------------------------------------
# Reading of the simulated instrument
# ---------------------------------------------------------------------


def _measurement_record(records: Sequence["SourceRecord"]) -> "SourceRecord":
    """The source of the measurement when ``source=`` is omitted.

    The single current source of the result, else its single voltage
    source (audit W2R2-05); the rule of the releases before 2026.11, which
    read the single current source whatever voltage sources the result
    carried, extended to voltage-driven measurements (audit W2A-10).

    Raises
    ------
    ValueError
        If the result has several current sources, or no current source
        and not exactly one voltage source.
    """
    by_kind = {kind: [r for r in records if r.kind == kind] for kind in _SOURCE_KINDS}
    for kind in _SOURCE_KINDS:  # current first
        if len(by_kind[kind]) == 1:
            return by_kind[kind][0]
        if by_kind[kind]:
            break
    counts = "; ".join(
        f"{len(found)} {kind} source{'s' if len(found) != 1 else ''}"
        + (f" ({', '.join(r.name for r in found)})" if found else "")
        for kind, found in by_kind.items()
    )
    raise ValueError(
        f"The result carries {counts}; without source= the reading takes the "
        "single current source, or the single voltage source of a result "
        "without current sources. Pass source=<name>."
    )


def measured_impedance(
    result: "FieldResult",
    probe: Optional[str] = None,
    *,
    source: Optional[str] = None,
) -> list[complex]:
    r"""Reading of a simulated fall-of-potential measurement.

    .. math::

       Z_\text{meas}(f) = \frac{\varphi_E(f) - \varphi_S(f)}{I(f)}

    with :math:`\varphi_E` the potential of the earthing system the test
    current is fed into (the galvanic cluster of the source's
    ``attached_to`` node, length-weighted as in
    :meth:`~groundfield.solver.result.FieldResult.input_impedance`),
    :math:`\varphi_S` the potential of the voltage terminal and :math:`I`
    the source current. This is the three-pole reading of an earth
    tester. The two-pole reading, the voltage across the current source
    (terminals E and H) over :math:`I`, is
    :meth:`~groundfield.solver.result.FieldResult.input_impedance`.

    The source is the current source of the measurement or, since
    2026.11, a voltage source (audit W2A-10): :math:`I` is then its solved
    current, and the reading is the same as with a current source at the
    same terminals (the problem is linear; an internal impedance of the
    source changes :math:`I`, not the reading).

    Parameters
    ----------
    result
        Solved result of a world with a current or voltage source.
    probe
        Node the voltmeter reads against the earthing system: the name of
        a junction or of an electrode (an electrode stands for its
        galvanic cluster). Default: the instrument terminal
        :data:`MEASUREMENT_S_TERMINAL` when the result has it (a network
        built with a probe lead), else the voltage-probe electrode
        (``probe_<kind>_<i>``) of a
        :class:`~groundfield.generators.tn_network.TnNetworkGenerator`
        network.
    source
        Name of the measurement's source. Omitted, the reading takes the
        result's single current source, or, when it has no current
        source, its single voltage source (audit W2R2-05: a result with
        one current source and voltage sources besides reads the current
        source, as up to 2026.11); several current sources, or several
        voltage sources and no current source, need the name.

    Returns
    -------
    list of complex
        :math:`Z_\text{meas}` in Ω per frequency (``nan`` where the source
        current is zero).

    Raises
    ------
    ValueError
        If the result carries no source snapshot (hand-built results), no
        current or voltage source, or, with ``source`` omitted, several
        current sources or no current and several voltage sources;
        if the record of the source carries no current per frequency (a
        voltage record without its solved current); if no probe is given
        and none can be found.
    KeyError
        If ``source`` or ``probe`` is unknown.

    Notes
    -----
    With metallic leads (:attr:`MeasurementInjectionConfig.feed_lead`,
    :attr:`MeasurementProbeConfig.lead`) the voltage terminal carries the
    voltage induced in the probe lead by the feed loop and the other
    conductors (inductive coupling, ADR-0004); without a probe lead the
    reading is the galvanic one. A non-inductive lead does not change the
    reading: the feed lead is in series with an ideal current source and
    the probe lead carries no current.
    """
    if not result.sources:
        raise ValueError(
            "measured_impedance needs the source snapshot of the solve; this "
            "result carries none (hand-built FieldResult?)."
        )
    if source is None:
        record = _measurement_record(result.sources)
    else:
        by_name = {s.name: s for s in result.sources}
        if source not in by_name:
            known = ", ".join(sorted(by_name)) or "none"
            raise KeyError(f"Unknown source {source!r} (known: {known}).")
        record = by_name[source]
        if record.kind not in _SOURCE_KINDS:
            raise ValueError(
                f"Source {source!r} is a {record.kind!r} source; the reading is "
                "defined for the current or voltage source of the measurement."
            )
    n_freq = len(result.frequencies)
    if len(record.current) != n_freq:
        # A voltage record without its solved current (audit W2A-10).
        raise ValueError(
            f"{record.kind.capitalize()} source {record.name!r}: the result "
            "carries no solved source current (the backend did not report "
            f"it), or not one per frequency ({len(record.current)} currents, "
            f"{n_freq} frequencies)."
        )
    if probe is None:
        probe = _default_probe(result)
    u_e = _node_potential(result, record.attached_to)
    u_s = _node_potential(result, probe)
    return [
        (ue - us) / ik if ik != 0 else complex("nan")
        for ue, us, ik in zip(u_e, u_s, record.current, strict=False)
    ]


def _default_probe(result: "FieldResult") -> str:
    """Voltage terminal of a generated measurement set-up."""
    junctions = result.metadata.get("junction_potentials") or {}
    if MEASUREMENT_S_TERMINAL in junctions:
        return MEASUREMENT_S_TERMINAL
    for name in result.electrode_potentials:
        if _PROBE_ELECTRODE.match(name):
            return name
    raise ValueError(
        f"No voltage terminal {MEASUREMENT_S_TERMINAL!r} and no probe electrode "
        "(probe_<kind>_<i>) in the result; pass probe=<junction or electrode>."
    )


def _node_potential(result: "FieldResult", name: str) -> list[complex]:
    """Potential per frequency (V) of a junction or an electrode's cluster."""
    junctions = result.metadata.get("junction_potentials") or {}
    if name in junctions:
        return [complex(v) for v in junctions[name]]
    if name not in result.electrode_potentials:
        known = sorted(set(result.electrode_potentials) | set(junctions))
        shown = ", ".join(known[:8]) + (" ..." if len(known) > 8 else "")
        raise KeyError(
            f"{name!r} is neither an electrode nor a junction of this result "
            f"(known: {shown})."
        )
    # Length-weighted cluster potential, the definition of input_impedance.
    return result._cluster_potential(name)[1]


class MeasurementLeadConfig(GeneratorConfig):
    """One physical measurement lead (current feed or voltage probe).

    A measurement lead is a finite-impedance :class:`Conductor` that
    connects two anchor electrodes (typically the substation cluster
    and the auxiliary or probe cluster). It runs horizontally at
    :math:`z` = :attr:`depth_m` (see :func:`lead_axis_z`) between the
    horizontal positions of the two anchors. Typical variants:

    * ``depth_m = 0.0`` — laid out on the ground (the classical feed
      lead of a fall-of-potential measurement); modelled
      :data:`SURFACE_LEAD_HEIGHT_M` above the surface.
    * ``depth_m < 0.0`` — strung above ground at height ``-depth_m``.
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
            "z of the lead axis in m: positive values bury the lead like "
            "a cable, negative values string it above ground, 0 lays it "
            "on the surface (see lead_axis_z)."
        ),
    )
    wire_radius_m: float = Field(default=0.005, description="Wire radius in m.")
    conductor_type: str = Field(
        default="bare_copper",
        description=(
            "Conductor type for the engine. 'bare_copper' for an "
            "overhead measurement lead, 'cable_shield' or 'pen' for "
            "a buried cable that doubles as a measurement lead. "
            "'cable_shield' makes the lead a thin-walled tube: no internal "
            "inductance of a solid wire (50 nH/m less) and a thinner "
            "conductor on its axis is read as its core (user decisions "
            "Q32, Q33)."
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

    @model_validator(mode="after")
    def _galvanic_lead_in_soil(self) -> "MeasurementLeadConfig":
        if (
            self.coupling_to_soil == "galvanic"
            and isinstance(self.depth_m, (int, float))
            and self.depth_m < 0.0
        ):
            raise ValueError(
                "MeasurementLeadConfig: a galvanic lead leaks current into the "
                f"soil and cannot hang in the air (depth_m={self.depth_m}); use "
                "depth_m >= 0 or coupling_to_soil='isolated'."
            )
        return self


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

    Laid out on the ground (``depth_m=0``, modelled
    :data:`SURFACE_LEAD_HEIGHT_M` above the surface), bare copper,
    isolated soil coupling, Neumann inductance enabled.
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

    The lead is built as a cable shield (``conductor_type="cable_shield"``,
    user decisions Q32 and Q33 of 2026.11): a thin-walled tube, so its self
    inductance carries no internal inductance of a solid wire,
    μ₀/(8π) = 50 nH/m less than the same lead as a solid
    wire (a 150 m lead: 7.5 µH, 2.4 mΩ of reactance at 50 Hz), and a
    thinner conductor drawn on its axis is read as its core, coupled at the
    lead radius (a filament inside a tube). Its skin effect is not modelled
    (``resistance_model="skin"`` is rejected for a shield). For a lead that
    is a solid wire, build a :class:`MeasurementLeadConfig` with
    ``conductor_type="bare_copper"``.

    Parameters
    ----------
    depth_m
        Depth of the lead axis in m (positive into the soil).
    wire_radius_m
        Radius of the lead in m.
    inductance_model
        ``"neumann"`` (default) or ``None`` for a resistive lead.
    segment_length_m
        Sub-piece length in m; ``None`` keeps the lead lumped.

    Returns
    -------
    MeasurementLeadConfig
        The lead configuration.
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

    ``feed_lead`` is the metallic *Stromeinspeiseleitung* from the
    instrument's current terminal :data:`MEASUREMENT_H_TERMINAL` to the
    aux electrode; the source drives the test current between the
    substation and that terminal. ``None`` (default) means
    *galvanic-only*: the source's ``return_to`` is the aux electrode
    itself, i.e. an ideal return path without impedance or inductive
    coupling. Setting ``feed_lead`` to a :class:`MeasurementLeadConfig`
    (often via :func:`overhead_lead` or :func:`buried_lead`) adds the
    physical wire and enables the inductive coupling.
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

    ``lead`` is the metallic measurement wire from the instrument's
    voltage terminal :data:`MEASUREMENT_S_TERMINAL` to the probe; it
    carries no current (ideal voltmeter), so its resistance does not
    enter the reading, but the voltage induced in it does. As with
    :class:`MeasurementInjectionConfig.feed_lead`, ``None`` is
    galvanic-only (the voltmeter reads the probe electrode directly);
    setting it enables inductive coupling.
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
    3. if :attr:`injection.feed_lead` is set, adds the instrument's
       current terminal, the junction :data:`MEASUREMENT_H_TERMINAL` at
       the substation position, and a :class:`Conductor` from it to the
       aux anchor with the configured depth, conductor type, and
       inductance model;
    4. if :attr:`probe.lead` is set, adds the voltage terminal, the
       junction :data:`MEASUREMENT_S_TERMINAL`, and a similar conductor
       from it to the probe anchor;
    5. attaches the source to the substation anchor with
       ``return_to`` pointing at the end of the current loop
       (:data:`MEASUREMENT_H_TERMINAL` with a feed lead, else the aux
       anchor): the test current flows through the soil from the
       substation to the auxiliary electrode and back through the feed
       lead.

    The reading of the simulated instrument is
    :func:`measured_impedance` (three-pole, E-S) and
    ``result.input_impedance()`` (two-pole, E-H).

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
