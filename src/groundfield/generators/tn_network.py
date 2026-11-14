"""TN low-voltage distribution network generator (reference cases).

The generator composes the four reusable spec layers
(:mod:`~groundfield.generators.electrode_specs`,
:mod:`~groundfield.generators.grounding`,
:mod:`~groundfield.generators.placement`,
:mod:`~groundfield.generators.soil_specs`) plus the
per-:mod:`~groundfield.generators.building` type catalog into a
fully populated :class:`groundfield.World` representing a typical
TN low-voltage distribution network (substation, house connections,
cable cabinets).

Compared with the v1 prototype this generator now supports:

* **Heterogeneous substation grounding** — ring AND/OR rods AND/OR
  strip AND/OR foundation, in any combination, each with its own
  presence probability and stochastic geometry.
* **Heterogeneous cable-cabinet grounding** — same flexibility for
  cable cabinets; v1 forced a single rod.
* **Building-type catalog** — multiple :class:`BuildingTypeSpec`
  entries with independent grounding systems. The default catalog
  ships ``residential`` / ``small_industry`` / ``medium_industry``
  / ``large_industry`` and is fully customisable.
* **Pluggable placement** — Manhattan grid (with optional jitter) or
  explicit caller-supplied coordinates. Future strategies plug into
  the same :data:`PlacementSpec` discriminated union.
* **Pluggable soil model** — homogeneous, two-layer, or arbitrary
  multi-layer.

Lazy-resolution semantics
-------------------------
:class:`TnNetworkGenerator` keeps the v1 lazy-mode build:

* ``gen.build(cfg)`` resolves distributions opportunistically. Each
  *site* (substation, every cable cabinet, every building) draws its
  ``presence_prob`` Bernoulli and samples its geometric
  distributions independently. This is what produces a real
  per-site mix when the user supplies population-level
  distributions (e.g. ``presence_prob=0.7`` for an "additional
  rod" electrode that 70 % of houses receive).
* ``gen.sample_world(rng)`` calls ``cfg.sample(rng)`` first, which
  collapses every :class:`Distribution` of the config tree to one
  constant per realisation — also those inside the electrode specs of
  a building type, so all buildings of that type share the drawn value
  (since 2026.11; v0.15.0 left those to per-site draws, 10.53). Plain
  population parameters stay per site: a ``presence_prob`` of ``0.7``
  or a placement ``jitter_m`` still draws per site in the build, from
  the same ``rng``. Useful for persistable Monte-Carlo realisations.

Geometric layout
----------------
* Substation at ``substation.position``. Its grounding system is
  built via :meth:`GroundingSystemSpec.build_at`.
* Buildings are placed via the configured :data:`PlacementSpec`.
  When ``placement`` is :class:`ManhattanGridPlacement`, buildings
  are generated in the order ``[type_0]*n_0 + [type_1]*n_1 + …`` —
  i.e. all buildings of the first type fill the grid first. This
  keeps the layout deterministic and reproducible. Use
  :class:`ExplicitPlacement` if you need a specific spatial mix.
* Cable cabinets are placed via their own
  :class:`PlacementSpec` (defaults to a Manhattan grid along the
  substation row). Their grounding follows :attr:`CabinetConfig.grounding`.
* PEN backbone (star topology): each cable cabinet has its own cable
  (main) from the substation. A building that a main crosses is joined
  to it there, without a drop, and fed by that main's cabinet (the
  nearest of the cabinets whose mains cross it, ties to the lower
  cabinet index; audits R4A-02, R4B-01); every other building is fed by
  the *nearest* cable cabinet by Manhattan distance, through a service
  drop that taps that cabinet's cable at the point nearest to the
  building (a T-joint, or the substation / the cabinet; rules in
  :mod:`groundfield.generators._pen_routing`). Every conductor routed to a
  foundation ends at the point of the foundation nearest to it (user
  decision Q41): a drop is the closest pair of its cable and the
  foundation wires, a cable leaves a substation with a foundation at its
  point nearest to the cabinet. In the radial topology every trunk counts
  for the crossing rule. No drop runs along a cable farther than one
  minimum section (1 cm); a drop may pass over other buildings'
  foundations, 0.2 m above their wires.

Validity envelope
-----------------
* Frequency: $f \\le 1\\,\\mathrm{kHz}$ (quasi-static).
* Soil: any of the spec'd models. default is two-layer.
* Topology: radial-with-trunk via cable cabinets — the
  open-building-map / real-street layout is roadmap.
"""

from __future__ import annotations

import math
import warnings
from typing import ClassVar, Literal, Optional, Union

import numpy as np
from pydantic import Field, model_validator

from groundfield.generators._compat import rename_input_keys, warn_renamed
from groundfield.generators.base import (
    GeneratorConfig,
    WorldGenerator,
    _coerce_rng,
)
from groundfield.generators.building import (
    BuildingTypeSpec,
    default_building_catalog,
)
from groundfield.generators.distributions import AnyDistribution, Distribution
from groundfield.generators.electrode_specs import (
    RingElectrodeSpec,
    RodElectrodeSpec,
    rod_circle,
)
from groundfield.generators.grounding import GroundingSystemSpec
from groundfield.generators._pen_routing import PenRouter
from groundfield.generators.materialise import _site_geometry, materialise_plan
from groundfield.generators.measurement import (
    MEASUREMENT_H_TERMINAL,
    MEASUREMENT_S_TERMINAL,
    MeasurementLeadConfig,
    MeasurementSetupConfig,
    lead_axis_z,
)
from groundfield.generators.pen_topology import (
    PenTopology,
    RadialTrunkTopology,
    StarCabinetTopology,
    StarKvsTopology,
)
from groundfield.generators.plan import (
    ConductorParameters,
    NetworkPlan,
    PlannedBranch,
    PlannedJunction,
    PlannedMeasurement,
    PlannedSite,
    PlannedSource,
)
from groundfield.generators.placement import (
    ManhattanGridPlacement,
    PlacementSpec,
)
from groundfield.generators.soil_specs import (
    SoilSpec,
    TwoLayerSoilSpec,
    materialise_soil,
)
from groundfield.world import World

__all__ = [
    "PenConfig",
    "SubstationConfig",
    "CabinetConfig",
    "KvsConfig",
    "TnNetworkConfig",
    "TnNetworkGenerator",
    "PenTopology",
    "RadialTrunkTopology",
    "StarCabinetTopology",
    "StarKvsTopology",
]


# ---------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------


def _to_float(value: Union[float, Distribution], rng: np.random.Generator) -> float:
    if isinstance(value, Distribution):
        return float(value.sample(rng))
    return float(value)


def _to_int(value: Union[int, Distribution], rng: np.random.Generator) -> int:
    if isinstance(value, Distribution):
        return int(round(float(value.sample(rng))))
    return int(round(float(value)))


def _beside(
    origin_xy: tuple[float, float], toward_xy: tuple[float, float], distance_m: float
) -> tuple[float, float]:
    """Point ``distance_m`` (m) to the left of ``origin_xy``, facing ``toward_xy``.

    Returns ``origin_xy`` unchanged when the two points coincide.
    """
    dx = float(toward_xy[0]) - float(origin_xy[0])
    dy = float(toward_xy[1]) - float(origin_xy[1])
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        return (float(origin_xy[0]), float(origin_xy[1]))
    return (
        float(origin_xy[0]) - distance_m * dy / norm,
        float(origin_xy[1]) + distance_m * dx / norm,
    )


def _override_foundation_from_footprint(
    grounding: GroundingSystemSpec,
    footprint,  # ``BuildingFootprint`` — typed loosely to avoid a hard import
    site_xy: Optional[tuple[float, float]] = None,
) -> GroundingSystemSpec:
    """Shape the foundations of ``grounding`` after a footprint's OMBR.

    Kept for callers of the pre-2026.11 private helper; equivalent to
    ``grounding.with_footprint(footprint, site_xy=site_xy, outline="ombr")``
    (see :meth:`GroundingSystemSpec.with_footprint`).
    """
    return grounding.with_footprint(footprint, site_xy=site_xy, outline="ombr")


# ---------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------


def _default_substation_grounding() -> GroundingSystemSpec:
    """default substation grounding: ring + 4 rods on a 2 m circle."""
    return GroundingSystemSpec(
        electrodes=[
            RingElectrodeSpec(radius_m=4.0, depth_m=0.6),
            *rod_circle(n=4, radius_m=2.0, length_m=2.5),
        ],
    )


def _default_cabinet_grounding() -> GroundingSystemSpec:
    """Default cable-cabinet grounding: a single 1.5 m driven rod."""
    return GroundingSystemSpec(
        electrodes=[
            RodElectrodeSpec(length_m=1.5, depth_m=0.0),
        ],
    )


def _default_building_placement() -> PlacementSpec:
    """default building placement: Manhattan grid 25 × 30 m, 10 per row."""
    return ManhattanGridPlacement(
        spacing_x_m=25.0,
        spacing_y_m=30.0,
        n_per_row=10,
        centre_xy=(0.0, 0.0),
    )


def _default_cabinet_placement() -> PlacementSpec:
    """Default cabinet placement: a wide Manhattan strip along the substation row."""
    return ManhattanGridPlacement(
        spacing_x_m=40.0,
        spacing_y_m=1.0,
        n_per_row=10,
        centre_xy=(0.0, 0.0),
    )


class SubstationConfig(GeneratorConfig):
    """Substation block — position + grounding system."""

    position: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Substation centre (x, y) in m.",
    )
    grounding: GroundingSystemSpec = Field(
        default_factory=_default_substation_grounding,
        description="Substation grounding system.",
    )


class CabinetConfig(GeneratorConfig):
    """Cable-cabinet block (German *KVS*) — placement, count, grounding.

    Attributes
    ----------
    placement
        Where to place the cabinets (star topology only).
    count_per_100_buildings
        Number of cabinets per 100 buildings; the count is
        ``ceil(count_per_100_buildings * n_buildings / 100)``, at least 1.
    fixed_count
        Optional fixed cabinet count (overrides the rate).
    grounding
        Grounding system of every cabinet.
    """

    placement: PlacementSpec = Field(
        default_factory=_default_cabinet_placement,
        description="Where to place the cable cabinets.",
    )
    count_per_100_buildings: Union[float, AnyDistribution] = Field(
        default=5.0,
        description=(
            "Number of cable cabinets per 100 buildings. The actual cabinet "
            "count is ceil(count_per_100_buildings * n_buildings / 100), "
            "with a floor of 1."
        ),
    )
    fixed_count: Optional[Union[int, AnyDistribution]] = Field(
        default=None,
        description=(
            "Optional: pin the number of cabinets to this value, ignoring "
            "``count_per_100_buildings``. Useful for explicit-placement runs."
        ),
    )
    grounding: GroundingSystemSpec = Field(
        default_factory=_default_cabinet_grounding,
        description="Per-cabinet grounding system.",
    )

    _field_renames: ClassVar[dict[str, str]] = {
        "quote_per_100_buildings": "count_per_100_buildings"
    }

    @model_validator(mode="before")
    @classmethod
    def _renamed_fields(cls, data: object) -> object:
        return rename_input_keys(data, cls._field_renames, "CabinetConfig")

    @property
    def quote_per_100_buildings(self) -> Union[float, Distribution]:
        """Deprecated alias of :attr:`count_per_100_buildings`."""
        warn_renamed(
            "CabinetConfig.quote_per_100_buildings",
            "CabinetConfig.count_per_100_buildings",
        )
        return self.count_per_100_buildings


class KvsConfig(CabinetConfig):
    """Deprecated alias of :class:`CabinetConfig`."""

    _unpickle_as: ClassVar[type] = CabinetConfig

    def model_post_init(self, context: object, /) -> None:
        warn_renamed("KvsConfig", "CabinetConfig", stacklevel=5)
        super().model_post_init(context)


class PenConfig(GeneratorConfig):
    """PEN cable backbone (distributed conductor, ADR-0003).

    Every PEN section (substation to cabinet, trunk) and service drop runs
    horizontally at ``z = depth_m`` between the horizontal positions of
    the connection points of its two end electrodes, i.e. in the cable
    trench; the short vertical risers to the electrode heads are not
    modelled. Up to 0.15.0 ``depth_m`` was ignored and the sections ran
    straight between the connection points (e.g. from the substation ring
    at 0.6 m to a cabinet rod head at 0 m).
    """

    wire_radius_m: float = Field(default=0.005, description="PEN wire radius in m.")
    depth_m: float = Field(
        default=0.6,
        gt=0.0,
        description="PEN burial depth in m (z of the cable axis, > 0).",
    )
    coupling_to_soil: str = Field(
        default="isolated",
        description=(
            "'isolated' for typical insulated cable, 'galvanic' for bare "
            "copper / exposed shield."
        ),
    )
    segment_length_m: Optional[float] = Field(
        default=5.0,
        description=(
            "Discretisation segment length in m for the distributed "
            "conductor model. None keeps the conductor lumped."
        ),
    )
    inductance_model: Optional[str] = Field(
        default=None,
        description="ADR-0004 inductance model: 'neumann' or None.",
    )


# ---------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------


def _default_building_catalog() -> list[BuildingTypeSpec]:
    return default_building_catalog()


def _default_building_counts() -> dict[str, Union[int, AnyDistribution]]:
    """30 residential single-family houses, no commercial."""
    return {"residential": 30}


def _default_soil() -> SoilSpec:
    return TwoLayerSoilSpec()


class TnNetworkConfig(GeneratorConfig):
    """Top-level configuration for a TN low-voltage distribution network.

    Attributes
    ----------
    name
        World name.
    soil
        Soil-model spec — homogeneous, two-layer, or multi-layer
        (see :data:`~groundfield.generators.soil_specs.SoilSpec`).
    substation
        Substation block (position + grounding system).
    cabinets
        Cable-cabinet block (placement, count, grounding).
    placement
        Building placement strategy. Default: Manhattan grid.
    building_types
        Catalog of building types available in this run. Each type
        carries its own grounding system. Defaults to a four-type
        catalog (residential, small/medium/large industry).
    building_counts
        Number of buildings per type. Keys must match a
        ``building_types[*].name``. Values may be fixed or
        :class:`Distribution`. Buildings are placed in catalog order.
    pen
        PEN backbone configuration.
    source_magnitude_a
        Driving current at the substation cluster, in A.
    """

    name: str = Field(default="tn_network", description="World name.")

    soil: SoilSpec = Field(
        default_factory=_default_soil,
        description="Soil model spec.",
    )

    substation: SubstationConfig = Field(
        default_factory=SubstationConfig,
        description="Substation block.",
    )

    cabinets: CabinetConfig = Field(
        default_factory=CabinetConfig,
        description="Cable cabinet block.",
    )

    placement: PlacementSpec = Field(
        default_factory=_default_building_placement,
        description="Building placement strategy.",
    )

    building_types: list[BuildingTypeSpec] = Field(
        default_factory=_default_building_catalog,
        description="Catalog of building types available in this run.",
    )

    building_counts: dict[str, Union[int, AnyDistribution]] = Field(
        default_factory=_default_building_counts,
        description=(
            "Number of buildings per type. Keys must match building_types[*].name."
        ),
    )

    pen: PenConfig = Field(
        default_factory=PenConfig,
        description="PEN backbone block.",
    )

    pen_topology: PenTopology = Field(
        default_factory=StarCabinetTopology,
        description=(
            "PEN backbone topology. ``StarCabinetTopology`` (default): "
            "every cabinet is wired directly to the substation by a "
            "cable of its own; a building that a cabinet's cable "
            "crosses is joined to it without a service drop and fed by "
            "that cabinet (the nearest of the cabinets whose cables "
            "cross it), every other building is fed by its nearest "
            "cabinet by Manhattan distance, its service drop tapping "
            "that cabinet's cable at the point nearest to the building "
            "(the closest pair of cable and foundation, user decision "
            "Q41). "
            "``RadialTrunkTopology`` lays out the substation as a "
            "radial set of LV feeders (Manhattan-aligned) with a "
            "finite slot budget per source — once a feeder's slots "
            "are exhausted a cabinet is inserted along the trunk axis "
            "to host the next batch of building taps. When "
            "``RadialTrunkTopology`` is selected the parent "
            ":attr:`CabinetConfig.placement` / ``fixed_count`` / "
            "``count_per_100_buildings`` fields are ignored — the "
            "topology decides cabinet positions and counts — but "
            ":attr:`CabinetConfig.grounding` is still honoured for the "
            "earthing of every inserted cabinet."
        ),
    )

    source_magnitude_a: Union[float, AnyDistribution] = Field(
        default=1.0,
        description=(
            "Driving current at the substation cluster, in A (RMS). "
            "Interpretation depends on the study: for a fall-of-"
            "potential measurement this is the *test current* fed in "
            "by the measurement device (typically 1 A … 25 A); for a "
            "fault simulation this is the *single-phase ground-fault "
            "current* (typically a few hundred A to several kA). "
            "The solver is linear, so for normalised studies setting "
            "this to 1.0 lets every potential / cluster impedance be "
            "read directly as Ω. Accepts a :class:`Distribution` for "
            "stochastic sweeps (e.g. ``Discrete(values=[1.0, 5000.0])`` "
            "to compare measurement vs. fault on the same world)."
        ),
    )

    measurement: Optional[MeasurementSetupConfig] = Field(
        default=None,
        description=(
            "Optional grounding-resistance measurement setup. "
            "When set, the generator adds "
            "the auxiliary current electrode, the voltage probe, "
            "and (optionally) the metallic feed / probe leads with "
            "their instrument terminals, and makes the source return "
            "its current through the auxiliary electrode (via the "
            "feed lead when one is configured). ``None`` keeps the "
            "historic behaviour: source returns to remote earth, no "
            "probe or aux electrode."
        ),
    )

    source_return_to: Optional[str] = Field(
        default=None,
        description=(
            "Optional **explicit** override for the source's "
            "``return_to`` electrode name. ``None`` (default) "
            "preserves the historic behaviour: ``return_to`` is "
            "derived automatically from :attr:`measurement` (the "
            "end of the current loop: the feed lead's terminal "
            "``meas_H``, else the auxiliary current electrode) or left "
            "as ``None`` for "
            "remote-earth return. Setting this field takes "
            "precedence over the measurement-setup auto-routing — "
            ":meth:`TnNetworkGenerator.build` emits a "
            ":class:`UserWarning` when both are set so the "
            "precedence is explicit rather than silent."
        ),
    )

    source_kind: Literal["current", "voltage"] = Field(
        default="current",
        description=(
            'Source type at the substation. Only ``"current"`` '
            '(default) and ``"voltage"`` are accepted — typos like '
            '``"voltage_"`` are rejected at validation time with a '
            "Pydantic ``ValidationError`` rather than silently falling "
            "through to the default ``CurrentSource`` factory. "
            '``"current"`` is the right '
            "choice for the typical fall-of-potential measurement and "
            'for fault simulations; ``"voltage"`` is reserved for '
            "the rare multi-port tests."
        ),
    )

    _field_renames: ClassVar[dict[str, str]] = {
        "kvs": "cabinets",
        "source_magnitude_A": "source_magnitude_a",
    }

    @model_validator(mode="before")
    @classmethod
    def _renamed_fields(cls, data: object) -> object:
        return rename_input_keys(data, cls._field_renames, "TnNetworkConfig")

    @property
    def kvs(self) -> CabinetConfig:
        """Deprecated alias of :attr:`cabinets`."""
        warn_renamed("TnNetworkConfig.kvs", "TnNetworkConfig.cabinets")
        return self.cabinets

    @property
    def source_magnitude_A(self) -> Union[float, Distribution]:  # noqa: N802
        """Deprecated alias of :attr:`source_magnitude_a`."""
        warn_renamed(
            "TnNetworkConfig.source_magnitude_A", "TnNetworkConfig.source_magnitude_a"
        )
        return self.source_magnitude_a


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------


def _router(cfg: "TnNetworkConfig", sites: list[PlannedSite]) -> PenRouter:
    """PEN router over ``sites`` (the first is the substation).

    A site with horizontal earth electrodes (foundation, ring, strip) is met
    at the point of their wires nearest to each cable (user decisions Q41,
    Q43); a site with rods only at the connection point of its anchor.
    """
    node_xy = {}
    wires = {}
    for site in sites:
        point, horizontal, _ = _site_geometry(site)
        if point is not None:
            node_xy[site.name] = (point[0], point[1])
            wires[site.name] = horizontal
    return PenRouter(
        sites[0].name,
        node_xy,
        depth_m=float(cfg.pen.depth_m),
        conductor=_pen_conductor(cfg.pen),
        wires=wires,
    )


def _pen_conductor(pen: PenConfig) -> ConductorParameters:
    """Conductor parameters shared by every PEN branch of the network."""
    return ConductorParameters(
        conductor_type="pen",
        wire_radius_m=pen.wire_radius_m,
        cross_section="from_radius",
        coupling_to_soil=pen.coupling_to_soil,
        segment_length_m=pen.segment_length_m,
        inductance_model=pen.inductance_model,
    )


def _lead_conductor(lead: MeasurementLeadConfig) -> ConductorParameters:
    """Conductor parameters of one measurement lead."""
    return ConductorParameters(
        conductor_type=lead.conductor_type,
        wire_radius_m=lead.wire_radius_m,
        cross_section="from_radius",
        coupling_to_soil=lead.coupling_to_soil,
        segment_length_m=lead.segment_length_m,
        inductance_model=lead.inductance_model,
    )


class TnNetworkGenerator(WorldGenerator[TnNetworkConfig]):
    """Generator for TN low-voltage distribution network reference worlds.

    See the module docstring for the full pipeline. In one sentence:
    resolve soil → substation grounding → place buildings (one per
    type-and-count) → per-building grounding → cable cabinets → wire the
    PEN backbone → measurement set-up → source, as a
    :class:`~groundfield.generators.plan.NetworkPlan`
    (:meth:`plan`), then materialise the plan (:meth:`build`).
    """

    def build(
        self,
        cfg: Optional[TnNetworkConfig] = None,
        *,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> World:
        """Materialise one realisation of the configured TN network.

        Equivalent to ``materialise_plan(self.plan(cfg, rng=rng))``.

        Parameters
        ----------
        cfg : TnNetworkConfig, optional
            Configuration overriding :attr:`cfg`. When ``None``
            (default) the generator's own ``cfg`` is used.
        rng : int or numpy.random.Generator, optional
            Draw the realisation from this seed / generator instead of
            the generator's own RNG.

        Returns
        -------
        World
            The assembled world with soil, electrodes, conductors and
            the configured source.

        Raises
        ------
        ValueError
            If the substation grounding resolves to zero electrodes,
            if the total building count is zero, or if a configured
            placement returns fewer positions than buildings.
        """
        return materialise_plan(self.plan(cfg, rng=rng))

    def plan(
        self,
        cfg: Optional[TnNetworkConfig] = None,
        *,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> NetworkPlan:
        """Resolve one realisation into a :class:`NetworkPlan`.

        Every random decision is made here, against ``rng`` (default:
        the generator's RNG), in a fixed order: soil, substation grounding, building
        counts, building positions, per-building grounding, cable
        cabinets (count, positions, groundings), measurement electrodes
        and leads, source magnitude. Materialising the returned plan
        draws no further random numbers.

        Parameters
        ----------
        cfg : TnNetworkConfig, optional
            Configuration overriding :attr:`cfg`.
        rng : int or numpy.random.Generator, optional
            Seed / generator of the random draws (default: the
            generator's own RNG).

        Returns
        -------
        NetworkPlan
            The concrete network.

        Raises
        ------
        ValueError
            See :meth:`build`.
        """
        cfg = cfg or self.cfg
        rng = self._rng if rng is None else _coerce_rng(rng)

        soil = materialise_soil(cfg.soil, rng)

        substation = cfg.substation.grounding.plan_site(
            "trafo", role="substation", position_xy=cfg.substation.position, rng=rng
        )
        if substation.anchor is None:
            raise ValueError(
                "TnNetworkGenerator: substation grounding produced zero "
                "electrodes. At least one electrode must have presence_prob > 0."
            )
        sites: list[PlannedSite] = [substation]

        # --- Buildings ---
        building_assignments = self._resolve_building_counts(cfg, rng)
        n_buildings = sum(count for _, count in building_assignments)
        if n_buildings < 1:
            raise ValueError(
                "TnNetworkGenerator: total building count is 0. Populate "
                "building_counts with at least one type."
            )
        positions = cfg.placement.generate(n_buildings, rng)
        if len(positions) != n_buildings:
            raise RuntimeError(
                f"placement.generate returned {len(positions)} positions, "
                f"expected {n_buildings}."
            )
        # Footprint-aware placement (ADR-0011): a placement with a
        # ``footprint_at(i)`` hook (currently :class:`OsmBuildingPlacement`)
        # shapes every foundation of the building after the footprint
        # (``GroundingSystemSpec.with_footprint``, OMBR rule).
        has_footprints = hasattr(cfg.placement, "footprint_at")
        buildings: list[PlannedSite] = []
        idx = 0
        for btype, count in building_assignments:
            for k in range(count):
                site_xy = positions[idx]
                grounding = btype.grounding
                footprint = cfg.placement.footprint_at(idx) if has_footprints else None
                if footprint is not None:
                    grounding = grounding.with_footprint(
                        footprint, site_xy=site_xy, outline="ombr"
                    )
                buildings.append(
                    grounding.plan_site(
                        f"{btype.name}_{k}",
                        role="building",
                        position_xy=site_xy,
                        rng=rng,
                        footprint=footprint,
                        building_type=btype.name,
                    )
                )
                idx += 1
        sites.extend(buildings)
        if not any(b.anchor is not None for b in buildings):
            raise ValueError(
                "TnNetworkGenerator: every building's grounding had zero "
                "present electrodes. Increase presence_prob."
            )

        # --- Cable cabinets + PEN backbone ---
        branches: list[PlannedBranch] = []
        junctions: list[PlannedJunction] = []
        if isinstance(cfg.pen_topology, RadialTrunkTopology):
            self._plan_radial_trunk(
                cfg, substation, buildings, sites, branches, junctions, rng
            )
        else:
            n_cabinets = self._resolve_cabinet_count(cfg, n_buildings, rng)
            cabinet_positions = cfg.cabinets.placement.generate(n_cabinets, rng)
            cabinets = [
                cfg.cabinets.grounding.plan_site(
                    f"kvs_{k}", role="cabinet", position_xy=site_xy, rng=rng
                )
                for k, site_xy in enumerate(cabinet_positions)
            ]
            sites.extend(cabinets)
            if not any(c.anchor is not None for c in cabinets):
                raise ValueError(
                    "TnNetworkGenerator: every cable-cabinet grounding had zero "
                    "present electrodes. Increase presence_prob or set "
                    "fixed_count > 0."
                )
            self._plan_star_backbone(
                cfg, substation, cabinets, buildings, branches, junctions
            )

        # --- Optional measurement set-up ---
        measurement = self._plan_measurement(cfg, substation, rng)

        # --- Source ---
        # With a measurement set-up the source is the earth tester: it
        # drives the test current between the substation (terminal E)
        # and the end of the current loop — the terminal H of the feed
        # lead, or the auxiliary electrode itself without a feed lead.
        # ``cfg.source_return_to`` is the explicit override of the
        # source's ``return_to``; warn when it shadows that terminal.
        return_to: Optional[str] = None
        loop_end: Optional[str] = None
        if measurement is not None:
            if measurement.feed_lead is not None:
                return_to = loop_end = MEASUREMENT_H_TERMINAL
            else:
                return_to = measurement.auxiliary.name
                loop_end = measurement.auxiliary.anchor
        if cfg.source_return_to is not None:
            if loop_end is not None and cfg.source_return_to != loop_end:
                warnings.warn(
                    "TnNetworkConfig.source_return_to="
                    f"{cfg.source_return_to!r} takes precedence over "
                    "the measurement-setup return terminal "
                    f"({loop_end!r}). The aux electrode and any "
                    "metallic feed/probe leads remain in the world, "
                    "but the source's return current is routed to "
                    "the user-supplied electrode instead.",
                    UserWarning,
                    stacklevel=3,
                )
            return_to = cfg.source_return_to
        source = PlannedSource(
            attached_to=substation.name,
            return_to=return_to,
            kind=cfg.source_kind,
            magnitude=_to_float(cfg.source_magnitude_a, rng),
        )
        return NetworkPlan(
            name=cfg.name,
            soil=soil,
            sites=sites,
            junctions=junctions,
            branches=branches,
            measurement=measurement,
            sources=[source],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_building_counts(
        self,
        cfg: TnNetworkConfig,
        rng: np.random.Generator,
    ) -> list[tuple[BuildingTypeSpec, int]]:
        """Resolve the (type → count) mapping into an ordered list.

        Reproducibility
        ---------------
        Each entry of ``cfg.building_counts`` whose value is a
        :class:`~groundfield.generators.distributions.Distribution`
        consumes draws from ``rng``, so the *order* in which the
        types are sampled decides which draw lands on which type.
        The order used here is the **catalog order** of
        ``cfg.building_types`` — a declared list, hence canonical —
        and never the insertion order of the ``building_counts``
        mapping. A config re-serialised with sorted keys, round-
        tripped through YAML, or hand-edited by moving a line
        therefore yields the *bit-identical* world for the same
        seed (F33).

        Returns
        -------
        list of (BuildingTypeSpec, int)
            One entry per building type present in
            ``cfg.building_counts`` with a resolved count > 0, in
            catalog order. Types missing from the catalog raise
            :class:`KeyError`.
        """
        catalog_by_name = {t.name: t for t in cfg.building_types}
        # Validate first, in sorted key order, so the error message
        # does not depend on the mapping's insertion order either.
        for name in sorted(cfg.building_counts):
            if name not in catalog_by_name:
                raise KeyError(
                    f"building_counts['{name}'] has no matching entry in "
                    f"building_types (available: {list(catalog_by_name)})."
                )
        out: list[tuple[BuildingTypeSpec, int]] = []
        seen: set[str] = set()
        for btype in cfg.building_types:
            if btype.name in seen or btype.name not in cfg.building_counts:
                continue
            seen.add(btype.name)
            count = _to_int(cfg.building_counts[btype.name], rng)
            if count > 0:
                out.append((catalog_by_name[btype.name], count))
        return out

    def _resolve_cabinet_count(
        self,
        cfg: TnNetworkConfig,
        n_buildings: int,
        rng: np.random.Generator,
    ) -> int:
        """Resolve the actual cable-cabinet count.

        Uses ``cfg.cabinets.fixed_count`` if provided; otherwise computes
        ``ceil(rate * n_buildings / 100)`` with a floor of 1.
        """
        if cfg.cabinets.fixed_count is not None:
            return max(1, _to_int(cfg.cabinets.fixed_count, rng))
        rate = _to_float(cfg.cabinets.count_per_100_buildings, rng)
        return max(1, int(math.ceil(rate * n_buildings / 100.0)))

    def _plan_measurement(
        self, cfg: TnNetworkConfig, substation: PlannedSite, rng: np.random.Generator
    ) -> Optional[PlannedMeasurement]:
        """Resolve the optional measurement set-up.

        Builds the auxiliary current electrode and the voltage probe
        sites and the optional metallic feed / probe leads, each a
        finite-impedance conductor from an instrument terminal (a
        junction at the substation position) to the remote electrode.
        Returns ``None`` without a measurement set-up.
        """
        meas = cfg.measurement
        if meas is None:
            return None
        aux = meas.injection.grounding.plan_site(
            "aux", role="auxiliary", position_xy=meas.injection.position_xy, rng=rng
        )
        if aux.anchor is None:
            raise ValueError(
                "TnNetworkGenerator: measurement.injection.grounding "
                "produced zero electrodes. Increase presence_prob."
            )
        probe = meas.probe.grounding.plan_site(
            "probe", role="probe", position_xy=meas.probe.position_xy, rng=rng
        )
        if probe.anchor is None:
            raise ValueError(
                "TnNetworkGenerator: measurement.probe.grounding "
                "produced zero electrodes. Increase presence_prob."
            )
        # The instrument stands at the substation position; each lead
        # ends on a leakage-free terminal junction there (H: current,
        # S: voltage), so it neither bonds the remote electrode to the
        # substation nor short-circuits the voltmeter (10.17).
        feed_lead = probe_lead = None
        terminals: list[PlannedJunction] = []
        if meas.injection.feed_lead is not None:
            terminal, feed_lead = self._plan_lead(
                meas.injection.feed_lead,
                "meas_feed_lead",
                MEASUREMENT_H_TERMINAL,
                substation.position_xy,
                aux.name,
                rng,
            )
            terminals.append(terminal)
        if meas.probe.lead is not None:
            s_xy = substation.position_xy
            if meas.injection.feed_lead is not None:
                # Two leads towards one side (the in-line probe of the
                # 61.8 % rule) lie side by side, touching, instead of
                # through each other: coincident filaments have no finite
                # mutual inductance.
                s_xy = _beside(
                    s_xy,
                    meas.probe.position_xy,
                    meas.injection.feed_lead.wire_radius_m
                    + meas.probe.lead.wire_radius_m,
                )
            terminal, probe_lead = self._plan_lead(
                meas.probe.lead,
                "meas_probe_lead",
                MEASUREMENT_S_TERMINAL,
                s_xy,
                probe.name,
                rng,
            )
            terminals.append(terminal)
        return PlannedMeasurement(
            auxiliary=aux,
            probe=probe,
            feed_lead=feed_lead,
            probe_lead=probe_lead,
            terminals=terminals,
        )

    def _plan_lead(
        self,
        lead: MeasurementLeadConfig,
        name: str,
        terminal: str,
        instrument_xy: tuple[float, float],
        end: str,
        rng: np.random.Generator,
    ) -> tuple[PlannedJunction, PlannedBranch]:
        """Resolve one measurement lead and its instrument terminal.

        The lead runs horizontally at :func:`lead_axis_z` of its
        (resolved) ``depth_m`` (F48) from the terminal junction at the
        instrument to the remote site's anchor; the terminal sits at the
        same height, so the lead has no riser.
        """
        depth = _to_float(lead.depth_m, rng)
        z = lead_axis_z(depth, lead.coupling_to_soil)
        junction = PlannedJunction(name=terminal, position_xy=instrument_xy, depth_m=z)
        branch = PlannedBranch(
            name=name,
            role="measurement_lead",
            start=terminal,
            end=end,
            conductor=_lead_conductor(lead),
            axis_z_m=z,
        )
        return junction, branch

    def _plan_star_backbone(
        self,
        cfg: TnNetworkConfig,
        substation: PlannedSite,
        cabinets: list[PlannedSite],
        buildings: list[PlannedSite],
        branches: list[PlannedBranch],
        junctions: list[PlannedJunction],
    ) -> None:
        """Star backbone: substation → each cabinet → its buildings.

        Every cabinet has its own straight cable from the substation
        (``pen_main_<anchor>``; collinear cabinets get separate cables in
        one trench). A building that a main crosses is joined to it and fed
        by it, without a drop: by the nearest of the cabinets whose mains
        cross it, ties to the lower cabinet index (audit R4A-02). Every
        other building is fed by its nearest cabinet (Manhattan metric
        between the site positions); its service drop taps that cabinet's
        cable at the point nearest to the building — the closest pair of
        cable and foundation wires (user decision Q41), or the point
        nearest to the anchor of a building without a foundation — at a
        leakage-free tap junction ``pen_tap_<k>``, or the substation / the
        cabinet when the nearest point is an end of the cable, never
        another building's drop, and a drop from the cabinet leaves another
        main it would run along at that main's node (audit R4B-01). The
        full rules are in :mod:`._pen_routing`.
        """
        wired = [c for c in cabinets if c.anchor is not None]
        router = _router(cfg, [substation, *wired, *buildings])
        mains: dict[str, str] = {}
        for cab in wired:
            mains[cab.name] = f"pen_main_{cab.anchor}"
            router.add_cable(substation.name, cab.name, name=mains[cab.name])
        fed: dict[str, list[PlannedSite]] = {c.name: [] for c in wired}
        cabinet_of = {mains[c.name]: c for c in wired}
        for b in buildings:
            if b.anchor is None:
                continue
            bx, by = b.position_xy
            # A building that a main crosses is joined to it and fed by it
            # (no drop); several: the nearest of their cabinets, ties to the
            # lower cabinet index. Any other building is fed by its nearest
            # cabinet (audit R4A-02).
            crossed = [
                cabinet_of[m]
                for m in router.crossing(b.name, [mains[c.name] for c in wired])
            ]
            cab = min(
                crossed or wired,
                key=lambda c: abs(c.position_xy[0] - bx) + abs(c.position_xy[1] - by),
            )
            fed[cab.name].append(b)
        every_main = list(mains.values())
        for cab in wired:
            by_name = {b.name: b for b in fed[cab.name]}
            cables = [mains[cab.name]]
            for name in router.drop_order(list(by_name), cables, cab.name):
                # A drop from the cabinet does not run along another main
                # (audit R4B-01); a main that crosses a building it does not
                # feed is not joined to it.
                router.connect_drop(
                    name,
                    cables,
                    cab.name,
                    name=f"pen_service_{by_name[name].anchor}",
                    avoid=every_main,
                )
        branches.extend(router.branches())
        junctions.extend(router.junctions)

    def _plan_radial_trunk(
        self,
        cfg: TnNetworkConfig,
        substation: PlannedSite,
        buildings: list[PlannedSite],
        sites: list[PlannedSite],
        branches: list[PlannedBranch],
        junctions: list[PlannedJunction],
        rng: np.random.Generator,
    ) -> None:
        r"""Radial-trunk backbone (:class:`RadialTrunkTopology`).

        1. Group every building by the angularly closest feeder out
           of the substation.
        2. Per feeder, sort the assigned buildings by their axial
           projection onto the feeder direction.
        3. Walk the sorted list, taking up to
           :attr:`RadialTrunkTopology.slots_per_substation` taps at the
           substation; when that budget is exhausted, place a cabinet
           along the feeder axis (grounded per :attr:`CabinetConfig.grounding`)
           and continue with :attr:`RadialTrunkTopology.slots_per_cabinet`
           taps per cabinet.
        4. Wire the trunk (substation → cabinet$_1$ → cabinet$_2$ → …,
           one cable per segment) and the service drops as straight PEN
           conductors. The drop of a building fed by a cabinet taps the
           trunk from the substation to that cabinet at the point nearest
           to the building (the closest pair of trunk and foundation
           wires, user decision Q41; a leakage-free tap junction, or a
           cabinet), so it never runs back along the trunk; the drop of a
           building in a substation slot starts at the substation. A
           building that a trunk crosses (any trunk: a substation slot has
           no cable of its own, a later trunk runs on past the building's
           tap source) is joined to it there without a drop (audit
           R4A-02), and a drop from a node that would run along a trunk
           (both ends of a trunk section within one minimum section of its
           line, audit R5A-01) starts where it leaves it (audit R4B-01). A
           drop never starts on another building's drop
           (:mod:`._pen_routing`).

        Raises
        ------
        ValueError
            If the topology wires no building at all.
        """
        topology = cfg.pen_topology
        assert isinstance(topology, RadialTrunkTopology)  # pragma: no cover
        trunk: list[tuple[str, str, str]] = []
        fed: dict[str, list[PlannedSite]] = {}
        # Trunk cables from the substation to every tap source.
        paths: dict[str, list[str]] = {substation.name: []}
        wired = [b for b in buildings if b.anchor is not None]
        groups = topology.assign_buildings_to_feeders(
            substation_xy=cfg.substation.position,
            building_positions=[b.position_xy for b in wired],
        )
        if sum(len(g) for g in groups) == 0:
            raise ValueError(
                "TnNetworkGenerator: RadialTrunkTopology dropped every "
                "building (all were behind the closest feeder axis or "
                "beyond max_feeder_length_m). Check feeder directions "
                "and the building layout."
            )
        any_service_drop = False
        for feeder_idx, building_indices in enumerate(groups):
            if not building_indices:
                continue
            cabinet_positions = topology.plan_feeder_cabinet_positions(
                substation_xy=cfg.substation.position,
                feeder_idx=feeder_idx,
                n_buildings_on_feeder=len(building_indices),
            )
            taps: list[PlannedSite] = [substation]
            previous = substation
            path: list[str] = []
            for k, cabinet_xy in enumerate(cabinet_positions):
                site_name = f"{topology.name_prefix}_f{feeder_idx}_{k}"
                cab = cfg.cabinets.grounding.plan_site(
                    site_name, role="cabinet", position_xy=cabinet_xy, rng=rng
                )
                sites.append(cab)
                if cab.anchor is None:
                    warnings.warn(
                        f"RadialTrunkTopology: cabinet {site_name!r} "
                        "grounding produced zero electrodes "
                        "(presence_prob=0?); the next batch of "
                        "buildings falls back to the previous tap "
                        "source.",
                        UserWarning,
                        stacklevel=3,
                    )
                    continue
                trunk_name = f"pen_trunk_f{feeder_idx}_{k}"
                trunk.append((trunk_name, previous.name, cab.name))
                path = [*path, trunk_name]
                paths[cab.name] = path
                taps.append(cab)
                previous = cab
            budgets = [topology.slots_per_substation] + [
                topology.slots_per_cabinet for _ in range(len(taps) - 1)
            ]
            tap_idx = 0
            for building_idx in building_indices:
                while tap_idx < len(taps) - 1 and budgets[tap_idx] <= 0:
                    tap_idx += 1
                if budgets[tap_idx] <= 0:
                    warnings.warn(
                        "RadialTrunkTopology: feeder "
                        f"{feeder_idx} exhausted every slot "
                        f"(have {len(taps)} tap sources, "
                        f"need ≥ {len(building_indices)}); "
                        "wiring remaining buildings to the last "
                        "tap anchor. Increase slots_per_cabinet or "
                        "n_feeders.",
                        UserWarning,
                        stacklevel=3,
                    )
                budgets[tap_idx] = max(0, budgets[tap_idx] - 1)
                fed.setdefault(taps[tap_idx].name, []).append(wired[building_idx])
                any_service_drop = True
        if not any_service_drop:
            raise ValueError(
                "TnNetworkGenerator: RadialTrunkTopology produced "
                "zero PEN service drops. Check feeder directions, "
                "cabinet grounding presence_prob, and slot budgets."
            )
        router = _router(cfg, [s for s in sites if s.anchor is not None])
        for name, start, end in trunk:
            router.add_cable(start, end, name=name)
        trunks = [name for name, _, _ in trunk]
        for tap, fed_buildings in fed.items():
            by_name = {b.name: b for b in fed_buildings}
            cables = paths[tap]
            # For the crossing rule every trunk counts as a feeding cable: a
            # trunk that crosses a building is joined there and the building
            # gets no drop -- a building in a substation slot has no feeding
            # cable of its own, and a trunk beyond a building's tap source
            # runs on past it (audit R4A-02).
            for name in router.drop_order(list(by_name), cables, tap, crossing=trunks):
                router.connect_drop(
                    name,
                    cables,
                    tap,
                    name=f"pen_service_{by_name[name].anchor}",
                    crossing=trunks,
                )
        branches.extend(router.branches())
        junctions.extend(router.junctions)
