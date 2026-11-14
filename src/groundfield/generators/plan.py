r"""Pure-data network plan shared by every generator front-end (10.78).

A :class:`NetworkPlan` is the complete, concrete description of one
generated grounding network *before* it becomes a
:class:`groundfield.World`:

* the soil model;
* the **sites** — groups of bonded electrodes that form one galvanic
  cluster (substation, cable cabinets, buildings), each with its
  resolved electrode specifications and, for buildings, the footprint
  the foundation was derived from;
* the **junctions** — interior nodes of PEN cables (corners and tap
  points of a routed cable);
* the **branches** — conductors between two nodes (a node is a site or a
  junction): PEN cable sections and service drops;
* the optional **measurement** set-up (auxiliary electrode, voltage
  probe, measurement leads and the instrument terminals they end on);
* the **sources**.

Both front-ends produce a plan:
:meth:`groundfield.generators.TnNetworkGenerator.plan` (stochastic
population parameters) and
:meth:`groundfield.generators.LvNetworkLayout.plan` (an engineer's
layout of real footprints and routed cables). A single materialiser,
:func:`groundfield.generators.materialise.materialise_plan`, turns any
plan into a world; it owns the construction of foundations and their
concrete shells, PEN junctions, the measurement loop and the sources.

Every value in a plan is concrete: random draws (presence Bernoullis,
:class:`~groundfield.generators.distributions.Distribution` fields,
placement jitter) were made by the front-end. Materialising a plan
consumes no random numbers, so a plan persisted as JSON
(``plan.model_dump_json()``) reproduces its world exactly.

Physical content
----------------
The plan carries geometry and topology only; the field problem it
describes is the usual quasi-static one (frequency below 1 kHz):
electrodes leak current into the layered soil, branches are
finite-impedance conductors in the nodal analysis (ADR-0003), and
optional Neumann inductance couples them (ADR-0004).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundfield.generators.distributions import Distribution
from groundfield.generators.electrode_specs import ElectrodeSpec
from groundfield.geo.footprint import BuildingFootprint
from groundfield.soil.models import SoilModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from groundfield.world import World

__all__ = [
    "BranchRole",
    "ConductorParameters",
    "NetworkPlan",
    "PlannedBranch",
    "PlannedContact",
    "PlannedElectrode",
    "PlannedJunction",
    "PlannedMeasurement",
    "PlannedSite",
    "PlannedSource",
    "SiteRole",
]

SiteRole = Literal["substation", "cabinet", "building", "auxiliary", "probe"]
"""Role of a :class:`PlannedSite` in the network."""

BranchRole = Literal["pen", "service_drop", "measurement_lead"]
"""Role of a :class:`PlannedBranch` in the network."""


class PlannedElectrode(BaseModel):
    """One present electrode of a site, with every value resolved.

    Attributes
    ----------
    index
        Position of the electrode in the
        :class:`~groundfield.generators.grounding.GroundingSystemSpec`
        it was drawn from. It is part of the electrode name
        (``f"{site}_{kind}_{index}"``), so names stay stable when an
        earlier electrode of the spec was absent in this realisation.
    spec
        The electrode specification with concrete values only (no
        :class:`~groundfield.generators.distributions.Distribution`,
        ``presence_prob == 1``).
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, description="Position in the originating spec list.")
    spec: ElectrodeSpec

    @model_validator(mode="after")
    def _check_concrete(self) -> "PlannedElectrode":
        for field_name, value in self.spec.__dict__.items():
            if isinstance(value, Distribution):
                raise ValueError(
                    f"PlannedElectrode: field {field_name!r} of the "
                    f"{self.spec.kind} spec still holds a distribution; a "
                    "plan must be concrete (resolve the spec first)."
                )
        if float(self.spec.presence_prob) != 1.0:
            raise ValueError(
                "PlannedElectrode: presence_prob must be 1 -- absent "
                "electrodes are left out of the plan."
            )
        return self


class PlannedSite(BaseModel):
    """A grounding site: bonded electrodes forming one galvanic cluster.

    Attributes
    ----------
    name
        Site name; every electrode is named ``f"{name}_{kind}_{index}"``
        and the bonds ``f"{name}_bond_{k}"``.
    role
        ``"substation"``, ``"cabinet"``, ``"building"``,
        ``"auxiliary"`` (current return electrode of a measurement) or
        ``"probe"`` (voltage probe electrode).
    position_xy
        Site reference point ``(x, y)`` in m; electrode offsets are
        relative to it.
    electrodes
        Present electrodes in spec order. Empty when every electrode of
        the spec was absent in this realisation (the site then has no
        anchor and is not wired).
    bond_conductor_type
        Conductor type of the bonds between the anchor and the other
        electrodes of the site.
    footprint
        Building outline the foundation geometry was derived from
        (buildings with a footprint only; informational).
    building_type
        Name of the building type the site was drawn from (buildings of
        :class:`~groundfield.generators.tn_network.TnNetworkGenerator`).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    role: SiteRole
    position_xy: tuple[float, float]
    electrodes: list[PlannedElectrode] = Field(default_factory=list)
    bond_conductor_type: str = "bare_copper"
    footprint: Optional[BuildingFootprint] = None
    building_type: Optional[str] = None

    def electrode_name(self, electrode: PlannedElectrode) -> str:
        """Return the world name of one of the site's electrodes."""
        return f"{self.name}_{electrode.spec.kind}_{electrode.index}"

    @property
    def anchor(self) -> Optional[str]:
        """Name of the anchor electrode (the first present one), or ``None``."""
        if not self.electrodes:
            return None
        return self.electrode_name(self.electrodes[0])


class PlannedJunction(BaseModel):
    """Interior node of a PEN cable (a corner or a tap point).

    Attributes
    ----------
    name
        Node name, unique among sites and junctions.
    position_xy
        Horizontal position in m.
    depth_m
        Burial depth in m (positive into the soil).

    Notes
    -----
    A junction is materialised as a leakage-free
    :class:`~groundfield.geometry.junction.Junction` node at
    ``(x, y, depth_m)`` (see
    :func:`groundfield.generators.materialise.materialise_junction`); it
    injects no current into the soil. Up to 0.15.0 it was a 5 cm rod
    electrode that leaked part of the cable current (review finding
    10.38).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    position_xy: tuple[float, float]
    depth_m: float = Field(default=0.6, description="Burial depth in m.")


class ConductorParameters(BaseModel):
    """Electrical and discretisation parameters of a branch conductor.

    Attributes
    ----------
    conductor_type
        Conductor type tag (``"pen"``, ``"bare_copper"``,
        ``"cable_shield"``, ...).
    wire_radius_m
        Wire radius in m.
    cross_section
        Cross section in m², ``"from_radius"`` (π r²) or ``None`` (ideal
        galvanic short).
    coupling_to_soil
        ``"isolated"`` (no leakage along the wire) or ``"galvanic"``.
    segment_length_m
        Discretisation length in m of the distributed conductor model;
        ``None`` keeps the conductor lumped.
    inductance_model
        ``"neumann"`` or ``None`` (resistive branch).
    """

    model_config = ConfigDict(extra="forbid")

    conductor_type: str = "pen"
    wire_radius_m: float = Field(default=0.005, gt=0.0, description="Wire radius in m.")
    cross_section: Union[float, Literal["from_radius"], None] = "from_radius"
    coupling_to_soil: str = "isolated"
    segment_length_m: Optional[float] = Field(default=None, gt=0.0)
    inductance_model: Optional[str] = None


class PlannedContact(BaseModel):
    """Where a branch meets a foundation of a site (user decision Q41).

    Attributes
    ----------
    electrode
        Name of the electrode (a wire of the site's foundation) that holds
        the point; the branch is attached to it.
    position_xy
        Plan position ``(x, y)`` in m where the branch ends: the point of
        the foundation nearest to the branch, on that wire — or, where the
        generator joins a cable to a foundation wire that it crosses or
        touches (and where a drop starts at such a joint), the cable's
        point, which lies at most one minimum cable section off the wire
        (:func:`~groundfield.generators._pen_routing.min_section_length`:
        1 cm for wire radii up to 7.3 mm).

    Notes
    -----
    New in 2026.11. The generators set it on every branch end at a site with
    a foundation (:class:`~groundfield.generators.electrode_specs.FoundationElectrodeSpec`):
    the branch ends at the point of the foundation nearest to it instead of
    the connection point of the site's anchor electrode (a corner of a
    rectangular foundation, the first vertex of a building outline).
    :func:`~groundfield.generators.materialise.materialise_branch` rejects a
    contact that names no foundation wire of the site or lies farther than
    that from it (audit R4A-05).
    """

    model_config = ConfigDict(extra="forbid")

    electrode: str
    position_xy: tuple[float, float]


class PlannedBranch(BaseModel):
    """A conductor between two nodes of the plan.

    Attributes
    ----------
    name
        Conductor name in the world.
    role
        ``"pen"`` (cable section), ``"service_drop"`` or
        ``"measurement_lead"``.
    start, end
        Node names: a site (resolved to its anchor electrode) or a
        junction. A name that is neither is used verbatim as an
        electrode name.
    conductor
        Conductor parameters.
    cable
        Name of the routed cable the section belongs to (informational).
    axis_z_m
        ``None`` (default): the conductor runs straight between its two
        end points (see ``start_contact``). A value: the conductor runs
        horizontally at depth ``z = axis_z_m`` (m, positive into the soil,
        negative above ground) between the horizontal positions of those
        points; it stays electrically connected to the two end nodes.
    start_contact, end_contact
        ``None`` (default): the end point is the connection point of the
        node (a site's anchor electrode, or a junction). A
        :class:`PlannedContact` (only at a site): the branch ends at that
        point of a foundation wire of the site, at the wire's depth (or at
        ``axis_z_m``), and is attached to that wire (user decision Q41).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    role: BranchRole
    start: str
    end: str
    conductor: ConductorParameters
    cable: Optional[str] = None
    axis_z_m: Optional[float] = None
    start_contact: Optional[PlannedContact] = None
    end_contact: Optional[PlannedContact] = None


class PlannedMeasurement(BaseModel):
    """Fall-of-potential measurement set-up.

    Attributes
    ----------
    auxiliary
        Auxiliary current electrode (role ``"auxiliary"``).
    probe
        Voltage probe electrode (role ``"probe"``), or ``None`` when the
        probe is a pure sampling point without an electrode.
    feed_lead, probe_lead
        Optional metallic leads (role ``"measurement_lead"``). Each runs
        from an instrument terminal (see :attr:`terminals`) to the
        remote electrode.
    terminals
        Leakage-free junctions at the instrument at which the leads end:
        the current terminal ``H`` (feed lead) and the voltage terminal
        ``S`` (probe lead). The current source of the measurement drives
        its current between the earthing system and ``H``; the voltmeter
        reads the potential difference between the earthing system and
        ``S``. Empty when no lead is configured.
    """

    model_config = ConfigDict(extra="forbid")

    auxiliary: PlannedSite
    probe: Optional[PlannedSite] = None
    feed_lead: Optional[PlannedBranch] = None
    probe_lead: Optional[PlannedBranch] = None
    terminals: list[PlannedJunction] = Field(default_factory=list)


class PlannedSource(BaseModel):
    """A current or voltage source.

    Attributes
    ----------
    name
        Source name; ``None`` lets :func:`groundfield.create_source`
        number it.
    attached_to, return_to
        Node names (sites resolve to their anchor electrode); a name that
        is neither a site nor a junction is used verbatim as an
        electrode name. ``return_to=None`` returns through remote earth.
    kind
        ``"current"`` or ``"voltage"``.
    magnitude
        Amplitude in A (current source) or V (voltage source).
    phase_deg
        Phase angle in degrees.
    """

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    attached_to: str
    return_to: Optional[str] = None
    kind: Literal["current", "voltage"] = "current"
    magnitude: float
    phase_deg: float = 0.0


class NetworkPlan(BaseModel):
    """Concrete, serialisable description of a generated network.

    Attributes
    ----------
    name
        World name.
    soil
        Concrete soil model.
    sites
        Substation, cable cabinets and buildings, in materialisation
        order.
    junctions
        PEN cable junctions, in materialisation order.
    branches
        PEN cable sections and service drops, in materialisation order.
    measurement
        Optional measurement set-up.
    sources
        Sources, in materialisation order.

    Notes
    -----
    :func:`~groundfield.generators.materialise.materialise_plan` builds
    the world in a fixed order: sites, measurement sites, junctions,
    instrument terminals, branches, measurement leads, sources.
    Electrodes, junctions and conductors therefore appear in the world
    in that order.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    soil: SoilModel
    sites: list[PlannedSite] = Field(default_factory=list)
    junctions: list[PlannedJunction] = Field(default_factory=list)
    branches: list[PlannedBranch] = Field(default_factory=list)
    measurement: Optional[PlannedMeasurement] = None
    sources: list[PlannedSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_names(self) -> "NetworkPlan":
        nodes: set[str] = set()
        names = [s.name for s in self.all_sites] + [j.name for j in self.all_junctions]
        for name in names:
            if name in nodes:
                raise ValueError(f"NetworkPlan: node name {name!r} is used twice.")
            nodes.add(name)
        sites = {s.name for s in self.all_sites}
        for branch in self.all_branches:
            for end, contact in (
                (branch.start, branch.start_contact),
                (branch.end, branch.end_contact),
            ):
                if end not in nodes:
                    raise ValueError(
                        f"NetworkPlan: branch {branch.name!r} ends at unknown "
                        f"node {end!r}."
                    )
                if contact is None:
                    continue
                if end not in sites or not contact.electrode.startswith(f"{end}_"):
                    raise ValueError(
                        f"NetworkPlan: branch {branch.name!r} meets "
                        f"{contact.electrode!r} at node {end!r}; a contact "
                        "names an electrode of the site the branch ends at."
                    )
        return self

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    @property
    def all_sites(self) -> list[PlannedSite]:
        """Sites including the measurement electrodes, in build order."""
        out = list(self.sites)
        if self.measurement is not None:
            out.append(self.measurement.auxiliary)
            if self.measurement.probe is not None:
                out.append(self.measurement.probe)
        return out

    @property
    def all_junctions(self) -> list[PlannedJunction]:
        """PEN junctions and the instrument terminals, in build order."""
        out = list(self.junctions)
        if self.measurement is not None:
            out.extend(self.measurement.terminals)
        return out

    @property
    def all_branches(self) -> list[PlannedBranch]:
        """Branches including the measurement leads, in build order."""
        out = list(self.branches)
        if self.measurement is not None:
            for lead in (self.measurement.feed_lead, self.measurement.probe_lead):
                if lead is not None:
                    out.append(lead)
        return out

    def site(self, name: str) -> PlannedSite:
        """Return the site called ``name`` (measurement sites included)."""
        for s in self.all_sites:
            if s.name == name:
                return s
        raise KeyError(f"NetworkPlan has no site {name!r}.")

    def sites_with_role(self, role: SiteRole) -> list[PlannedSite]:
        """Return every site of the given role, in build order."""
        return [s for s in self.all_sites if s.role == role]

    def to_world(self) -> "World":
        """Materialise the plan (see :func:`materialise_plan`)."""
        from groundfield.generators.materialise import materialise_plan

        return materialise_plan(self)
