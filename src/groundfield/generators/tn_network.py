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
* Cable cabinets (star topology) stand per building group by default
  (user decisions Q22 and Q35 of 2026.11): the buildings are cut into one
  group per cabinet — runs of consecutive bearings around the
  substation, starting after the widest angular gap, of sizes that differ
  by at most one
  (:meth:`~groundfield.generators.pen_topology.StarCabinetTopology.group_buildings`)
  — and each group's cabinet stands at its far end: on the ray from the
  substation through the group's centroid,
  :attr:`CabinetConfig.end_clearance_m` (1 m) beyond the group's earth
  electrodes along that ray (horizontal wires and rods) plus the plan
  extent of the cabinet's own grounding, when it stands there within
  :attr:`CabinetConfig.max_far_end_gap_m` (20 m) of its houses, otherwise
  on the ray through the group's far-end building, 1 m beyond that
  building (audits W1B-01, W1B-02); in the default grid of 30 houses at
  (-118.5, 0) and (118.5, 0) m. The threshold is a step (user decision
  D7): a group that crosses it moves its cabinet by up to the width of the
  group; the plan records per cabinet which rule placed it and its
  distance on the centroid ray (``PlannedSite.cabinet_rule``,
  ``PlannedSite.centroid_ray_gap_m``, audit W1R2-02). A cabinet
  whose electrodes would come closer than ``end_clearance_m`` to those of
  another site raises a :class:`ValueError` that names both.
  A :attr:`CabinetConfig.placement` places them as before (the former
  default was a Manhattan strip along the substation row, which put both
  cabinets of the 30-house grid west of the substation). Their grounding
  follows :attr:`CabinetConfig.grounding`.
* PEN backbone (star topology): each cable cabinet has its own cable
  (main) from the substation. A building that a main crosses is joined
  to it there, without a drop, and fed by that main's cabinet (the
  nearest of the cabinets whose mains cross it, ties to the lower
  cabinet index; audits R4A-02, R4B-01); every other building is fed by
  its group's cabinet (with a given placement: the *nearest* cable
  cabinet by Manhattan distance), through a service drop that taps that
  cabinet's cable at the point nearest to the building (a T-joint, or the
  substation / the cabinet; rules in
  :mod:`groundfield.generators._pen_routing`). Every conductor routed to a
  site with a horizontal earth electrode ends at the point of that
  electrode nearest to it (foundations, user decision Q41; circular rings
  and strips, user decision Q43): a drop is the closest pair of its cable
  and the building's horizontal wires, a cable leaves the substation (the
  default ring) at its point nearest to the cabinet. In the radial
  topology every trunk counts for the crossing rule, and a building in a
  substation slot taps its feeder's trunk (Q22, Q35). No drop runs along a
  cable farther than one minimum section (1 cm); a drop may pass over
  other buildings' foundations, 0.2 m above their wires. Drops that leave
  one tap in one direction (houses behind each other) lie side by side at
  the touching distance of the PEN, sorted by length and centred on their
  common axis, so that only neighbours touch (decision D9 of 2026.11,
  :attr:`~groundfield.generators.plan.PlannedBranch.axis_offset_m`); with
  placement jitter along one common normal, so that no generated network
  reports an overlap (audit W3R2B-01, rule 5 of
  :mod:`groundfield.generators._pen_routing`).

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
from typing import Annotated, ClassVar, Literal, NamedTuple, Optional, Union

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
from groundfield.generators._contact import Outline, inside_outlines
from groundfield.generators._pen_routing import PenRouter
from groundfield.generators.materialise import (
    SiteGeometry,
    _site_geometry,
    materialise_plan,
)
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
    CabinetRule,
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
    """The cabinet placement of the default configuration up to 2026.11.

    A wide Manhattan strip along the substation row; since user decisions
    Q22 and Q35 the default places one cabinet per building group instead
    (:attr:`CabinetConfig.placement` ``None``).
    """
    return ManhattanGridPlacement(
        spacing_x_m=40.0,
        spacing_y_m=1.0,
        n_per_row=10,
        centre_xy=(0.0, 0.0),
    )


_PositiveDistance = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
"""A positive, finite distance in m (``CabinetConfig.end_clearance_m``, audit
W1B-05: a negative clearance put the cabinets inside the outer houses, 0 on a
foundation wire, NaN failed late in the rod constructor)."""


def _clearance_not_below_gap(prefix: str, clearance_m: float, max_gap_m: float) -> str:
    """Message for ``end_clearance_m >= max_far_end_gap_m`` (audit W1R2-09)."""
    return (
        f"{prefix}CabinetConfig.end_clearance_m ({clearance_m:g} m) must be "
        f"smaller than max_far_end_gap_m ({max_gap_m:g} m): a cabinet on the ray "
        "through its group's centroid stands at least end_clearance_m from its "
        "group, so every group would take the ray through its far-end building "
        "(audit W1R2-09)."
    )


class SubstationConfig(GeneratorConfig):
    """Substation block — position + grounding system.

    Attributes
    ----------
    position
        Substation centre (x, y) in m, in the local coordinates in which
        the generator builds the network. It moves the substation only:
        the buildings, cable cabinets and the measurement set-up are
        placed around (0, 0) whatever ``position`` is. A position at
        projected coordinates (UTM, Gauss-Krüger) therefore builds a world
        that mixes projected and local coordinates — with 10 houses at the
        UTM origin a PEN cable 33 118 km long — reported by
        :func:`~groundfield.diagnostics.diagnose` and warned about at the
        solve with
        :class:`~groundfield.diagnostics.MixedCoordinatesWarning` (user
        decision D5), and ``RadialTrunkTopology`` drops every building and
        fails to build (audit W2R2-07). To place a network at projected
        coordinates, build it around the origin and translate the whole
        built world (every electrode, conductor and junction by the same
        offset); groundfield has no public helper for that. The solve does
        not need it: results depend on coordinate differences only.
    grounding
        Substation grounding system.
    """

    position: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description=(
            "Substation centre (x, y) in m, local coordinates: it moves the "
            "substation only, the buildings stay around (0, 0). To place a "
            "network at projected coordinates, translate the whole built world."
        ),
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
        Where to place the cabinets (star topology only). ``None`` (the
        default since 2026.11, user decisions Q22 and Q35): one cabinet per
        building group, at the far end of the houses it feeds — the
        buildings are grouped by
        :meth:`~groundfield.generators.pen_topology.StarCabinetTopology.group_buildings`
        (consecutive bearings around the substation, sizes differing by at
        most one), and the cabinet of a group stands on the ray from the
        substation through the group's centroid, ``end_clearance_m`` beyond
        the largest projection on that ray of the group's earth electrodes
        (horizontal wires and, since audit W1B-02, rods) plus the plan extent
        of the cabinet's own grounding — when it stands there within
        ``max_far_end_gap_m`` of the nearest electrode of its group; otherwise
        (a wide group, whose centroid ray leads past its houses) on the ray
        through the group's far-end building, the largest projection of its
        position on the direction of the centroid, ``end_clearance_m`` beyond
        that building's electrodes (audit W1B-01). It feeds its group. A
        cabinet whose electrodes would come closer than
        ``end_clearance_m`` to those of another site (a building of its or
        another group, the substation, another cabinet) raises a
        :class:`ValueError` naming both. A placement places the cabinets as up to
        2026.11 (the strip along the substation row of the former default,
        ``ManhattanGridPlacement(spacing_x_m=40, spacing_y_m=1,
        n_per_row=10)``, reproduces that world bit for bit), and every
        building is fed by its nearest cabinet (Manhattan metric). In both
        cases a building that a cabinet's cable crosses is fed by that cable
        (user decision Q41). A configuration stored before 2026.11 carries
        the former default placement and keeps its cabinets.
    count_per_100_buildings
        Number of cabinets per 100 buildings; the count is
        ``ceil(count_per_100_buildings * n_buildings / 100)``, at least 1
        (with building groups at most one per building).
    fixed_count
        Optional fixed cabinet count (overrides the rate).
    grounding
        Grounding system of every cabinet.
    end_clearance_m
        Smallest plan distance in m between the earth electrodes of a
        cabinet and those of any other site (building groups only),
        positive, finite and smaller than ``max_far_end_gap_m`` — a cabinet
        on the centroid ray stands at least this far from its group, so a
        larger value sent every group to the far-end ray, silently (audit
        W1R2-09); a distribution is checked where it is drawn (audit W1B-05).
    max_far_end_gap_m
        Largest plan distance in m (default 20, positive, finite, larger than
        ``end_clearance_m``) between a cabinet on the ray through its group's
        centroid and the nearest electrode of its group; a cabinet farther
        from the houses it feeds stands on the ray through the group's
        far-end building instead (building groups only; e.g. 10 houses and
        one small commercial building: 50.4 m on the centroid ray, 1.6 m on
        the far-end ray). The plan records per cabinet which rule placed it
        and that distance
        (:attr:`~groundfield.generators.plan.PlannedSite.cabinet_rule`,
        :attr:`~groundfield.generators.plan.PlannedSite.centroid_ray_gap_m`).
        The threshold is a step, kept for 2026.11 (user decision D7; audit
        W1R2-02): a group whose distance crosses it moves its cabinet by up
        to the width of the group, and the results jump — the 50-house grid
        with its house columns 32.0 / 32.2 m apart (southern group 19.99 →
        20.22 m) moves one cabinet by 141.8 m, its drops by −19 %, the input
        resistance at 50 Hz by −0.14 % beyond the smooth trend and, with an
        inductive PEN, the reactance at 1 kHz by −22 / −19 %
        (``carson_series`` / ``sommerfeld``). Groups lie on both sides of
        it: on the centroid ray the groups of the default 50-house grid
        stand 8.8 … 12.8 m from their houses, those of the 100-house grid
        10.05 … 23.01 m (two beyond 20 m; 15 / 25 m put four / none of its
        groups on the far-end ray), and over the 80 AP1 mixes the groups
        beyond it 20.21 … 53.56 m, the largest one kept 18.84 m.
    """

    placement: Optional[PlacementSpec] = Field(
        default=None,
        description=(
            "Where to place the cable cabinets (star topology). None: one "
            "cabinet per building group, at the far end of the houses it "
            "feeds (user decisions Q22, Q35); a placement: the given "
            "positions, every building fed by its nearest cabinet."
        ),
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
    end_clearance_m: Union[_PositiveDistance, AnyDistribution] = Field(
        default=1.0,
        description=(
            "Distance in m (positive, finite, smaller than max_far_end_gap_m) "
            "beyond the far end of its building group at which a cabinet's "
            "earth electrodes stand (placement=None only); a distribution is "
            "checked where it is drawn."
        ),
    )
    max_far_end_gap_m: Union[_PositiveDistance, AnyDistribution] = Field(
        default=20.0,
        description=(
            "Largest distance in m (positive, finite, larger than "
            "end_clearance_m) between a cabinet on the ray through its group's "
            "centroid and the nearest electrode of its group; farther, the "
            "cabinet stands on the ray through the group's far-end building "
            "instead (placement=None only; a distribution is checked where it "
            "is drawn). A step: a group that crosses it moves its cabinet; the "
            "plan records the rule per cabinet (PlannedSite.cabinet_rule)."
        ),
    )

    _field_renames: ClassVar[dict[str, str]] = {
        "quote_per_100_buildings": "count_per_100_buildings"
    }

    @model_validator(mode="before")
    @classmethod
    def _renamed_fields(cls, data: object) -> object:
        return rename_input_keys(data, cls._field_renames, "CabinetConfig")

    @model_validator(mode="after")
    def _clearance_below_far_end_gap(self) -> "CabinetConfig":
        # audit W1R2-09: with end_clearance_m >= max_far_end_gap_m every
        # group took the far-end ray, silently. Distributions are checked
        # where they are drawn (TnNetworkGenerator._group_cabinets).
        clearance, max_gap = self.end_clearance_m, self.max_far_end_gap_m
        if (
            isinstance(clearance, float)
            and isinstance(max_gap, float)
            and clearance >= max_gap
        ):
            raise ValueError(_clearance_not_below_gap("", clearance, max_gap))
        return self

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
            "cable of its own; by default the cabinets stand per "
            "building group, at the far end of the houses they feed "
            "(user decisions Q22, Q35; ``CabinetConfig.placement``); a "
            "building that a cabinet's cable crosses is joined to it "
            "without a service drop and fed by that cabinet (the nearest "
            "of the cabinets whose cables cross it), every other building "
            "is fed by its group's cabinet (with a given placement: its "
            "nearest cabinet by Manhattan distance), its service drop "
            "tapping that cabinet's cable at the point nearest to the "
            "building (the closest pair of cable and foundation, user "
            "decision Q41). "
            "``RadialTrunkTopology`` lays out the substation as a "
            "radial set of LV feeders (Manhattan-aligned) with a "
            "finite slot budget per source — once a feeder's slots "
            "are exhausted a cabinet is inserted along the trunk axis "
            "to host the next batch of building taps; a building in a "
            "substation slot taps its feeder's trunk (Q22, Q35). When "
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
            '``"current"`` impresses ``source_magnitude_a`` in A, the '
            "usual choice for a fall-of-potential measurement and for "
            'fault simulations; ``"voltage"`` impresses the same number '
            "in V (an ideal voltage source; set its internal_resistance_ohm "
            "/ internal_inductance_h on the built world) and the solve "
            "returns its current. Both kinds give the same readings: "
            "``measured_impedance`` and ``input_impedance`` divide by the "
            "impressed or solved source current."
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


def _rod_points(site: PlannedSite) -> list[tuple[float, float]]:
    """Plan positions in m of the vertical electrodes (rods) of ``site``.

    A rod stands at the site position plus its ``offset_xy_m``, the point
    :func:`~groundfield.generators.materialise.materialise_electrode` builds
    it at.
    """
    cx, cy = site.position_xy
    return [
        (cx + planned.spec.offset_xy_m[0], cy + planned.spec.offset_xy_m[1])
        for planned in site.electrodes
        if isinstance(planned.spec, RodElectrodeSpec)
    ]


def _plan_extent(
    site: PlannedSite, geometry: SiteGeometry
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]]]:
    """Points and circles in m that bound the electrodes of ``site`` in plan.

    The vertices of its straight horizontal wires and the heads of its rods
    (points), and its ``RingElectrodeSpec`` circles ``(x, y, r)``. Every
    linear function (a projection) and every distance from a point takes its
    largest value over the electrodes at one of them.
    """
    points: list[tuple[float, float]] = []
    circles: list[tuple[float, float, float]] = []
    for wire in geometry.wires:
        if wire.circle is not None:
            circles.append(wire.circle)
        else:
            points.extend(wire.xy)
    points.extend(_rod_points(site))
    if not points and not circles and geometry.point is not None:
        points.append((geometry.point[0], geometry.point[1]))
    return points, circles


def _plan_radius(site: PlannedSite, geometry: SiteGeometry) -> float:
    """Largest plan distance in m of the electrodes of ``site`` from its
    position (0 for one rod at the position, the default cabinet)."""
    px, py = site.position_xy
    points, circles = _plan_extent(site, geometry)
    radius = 0.0
    for x, y in points:
        radius = max(radius, math.hypot(x - px, y - py))
    for x, y, r in circles:
        radius = max(radius, math.hypot(x - px, y - py) + r)
    return radius


def _group_direction(
    substation_xy: tuple[float, float], group: list[PlannedSite]
) -> tuple[float, float]:
    """Unit vector from the substation towards a building group.

    Through the centroid of the group's building positions; through the
    group's building farthest from the substation when the centroid is the
    substation (ties to the first of the group); :math:`+x` when every
    building stands on it.
    """
    sx, sy = substation_xy
    n = len(group)
    ux = sum(float(b.position_xy[0]) for b in group) / n - sx
    uy = sum(float(b.position_xy[1]) for b in group) / n - sy
    norm = math.hypot(ux, uy)
    spread = max(
        math.hypot(float(b.position_xy[0]) - sx, float(b.position_xy[1]) - sy)
        for b in group
    )
    if norm <= 1e-9 * max(1.0, spread):
        far = max(
            enumerate(group),
            key=lambda kb: (
                math.hypot(kb[1].position_xy[0] - sx, kb[1].position_xy[1] - sy),
                -kb[0],
            ),
        )[1]
        ux, uy = float(far.position_xy[0]) - sx, float(far.position_xy[1]) - sy
        norm = math.hypot(ux, uy)
        if norm == 0.0:
            ux, uy, norm = 1.0, 0.0, 1.0
    return ux / norm, uy / norm


def _far_end(
    substation_xy: tuple[float, float],
    group: list[PlannedSite],
    geometry: dict[str, SiteGeometry],
    clearance_m: float,
    radius_m: float = 0.0,
) -> tuple[tuple[float, float], PlannedSite]:
    """Position in m of the cabinet of a building group, and its far-end building.

    User decisions Q22 and Q35, audits W1B-01 and W1B-02. The group's
    direction :math:`u` is that of its centroid (:func:`_group_direction`).
    The far-end building has the largest projection of its position on
    :math:`u` (ties within :math:`10^{-9}` of the group's extent: the one
    nearest to the ray along :math:`u`, then the first of the group). The
    cabinet stands on the ray from the substation through the far-end
    building (along :math:`u` itself when that building lies on it),
    ``clearance_m + radius_m`` beyond the largest projection on the ray of
    the far-end building's electrodes — horizontal wires (a circle by its
    centre and radius) and rods. ``radius_m`` is the plan extent of the
    cabinet's own grounding around its position (:func:`_plan_radius`), so
    no electrode of the cabinet comes closer than ``clearance_m`` to one of
    the far-end building: their projections on the ray differ by at least
    that much. The clearance from every other electrode is checked by
    :func:`_check_cabinet_clearance`. Used for a group whose cabinet would
    stand too far from its houses on the ray through the centroid
    (:func:`_cabinet_position`): for a group of about 180 degrees that ray
    leads past its houses (audit W1B-01). Up to then the reach counted the
    horizontal wires only, so a cabinet rod could stand on a house rod
    (audit W1B-02).
    """
    sx, sy = substation_xy
    ux, uy = _group_direction(substation_xy, group)
    rel = [(float(b.position_xy[0]) - sx, float(b.position_xy[1]) - sy) for b in group]
    proj = [x * ux + y * uy for x, y in rel]
    spread = max(1.0, max(math.hypot(x, y) for x, y in rel))
    tol = 1e-9 * spread
    top = max(proj)
    k_far = min(
        (k for k in range(len(group)) if proj[k] >= top - tol),
        key=lambda k: (abs(rel[k][1] * ux - rel[k][0] * uy), k),
    )
    fx, fy = rel[k_far]
    off_ray = abs(fy * ux - fx * uy)
    norm = math.hypot(fx, fy)
    if off_ray > tol and norm > 0.0:
        ux, uy = fx / norm, fy / norm
    far = group[k_far]
    points, circles = _plan_extent(far, geometry[far.name])
    reach = -math.inf
    for cx, cy, r in circles:
        reach = max(reach, (cx - sx) * ux + (cy - sy) * uy + r)
    for x, y in points:
        reach = max(reach, (x - sx) * ux + (y - sy) * uy)
    distance = reach + float(clearance_m) + float(radius_m)
    return (sx + distance * ux, sy + distance * uy), far


def _centroid_end(
    substation_xy: tuple[float, float],
    group: list[PlannedSite],
    geometry: dict[str, SiteGeometry],
    clearance_m: float,
    radius_m: float = 0.0,
) -> tuple[tuple[float, float], PlannedSite]:
    """Cabinet position in m on the ray through the group's centroid (Q22, Q35).

    On the ray from the substation along the group's direction :math:`u`
    (:func:`_group_direction`), ``clearance_m + radius_m`` beyond the largest
    projection on :math:`u` of every electrode of the group — horizontal
    wires (a circle by its centre and radius) and, since audit W1B-02, rods —
    so no electrode of the cabinet comes closer than ``clearance_m`` to one
    of the group. Returns the building that holds that electrode as well.
    """
    sx, sy = substation_xy
    ux, uy = _group_direction(substation_xy, group)
    reach, holder = -math.inf, group[0]
    for b in group:
        points, circles = _plan_extent(b, geometry[b.name])
        for cx, cy, r in circles:
            value = (cx - sx) * ux + (cy - sy) * uy + r
            if value > reach:
                reach, holder = value, b
        for x, y in points:
            value = (x - sx) * ux + (y - sy) * uy
            if value > reach:
                reach, holder = value, b
    distance = reach + float(clearance_m) + float(radius_m)
    return (sx + distance * ux, sy + distance * uy), holder


def _group_gap(
    xy: tuple[float, float],
    group: list[PlannedSite],
    geometry: dict[str, SiteGeometry],
    radius_m: float = 0.0,
) -> float:
    """Plan distance in m from a cabinet at ``xy`` (electrodes within
    ``radius_m`` of it) to the nearest electrode of its group, at least 0."""
    gap = math.inf
    for b in group:
        shapes = _plan_shapes(b, geometry[b.name])
        if shapes is not None:
            gap = min(gap, _gap_to(xy, shapes))
    return max(0.0, gap - float(radius_m))


class _CabinetPlace(NamedTuple):
    """Where :func:`_cabinet_position` puts the cabinet of a building group."""

    xy: tuple[float, float]
    """Cabinet position in m."""
    holder: PlannedSite
    """The building named in a clearance error (the one it stands beyond)."""
    rule: CabinetRule
    """``"centroid_ray"`` or ``"far_end_ray"`` (audit W1R2-02)."""
    gap_m: float
    """Gap in m between the centroid-ray position and the group, the value
    compared with ``max_far_end_gap_m``."""


def _cabinet_position(
    substation_xy: tuple[float, float],
    group: list[PlannedSite],
    geometry: dict[str, SiteGeometry],
    clearance_m: float,
    radius_m: float,
    max_gap_m: float,
) -> _CabinetPlace:
    """Where the cabinet of a building group stands (Q22, Q35; W1B-01, W1B-02).

    On the ray through the group's centroid (:func:`_centroid_end`, the rule
    of user decisions Q22 and Q35) when that cabinet stands within
    ``max_gap_m`` (:attr:`CabinetConfig.max_far_end_gap_m`) of the nearest
    electrode of its group — a cabinet stands next to the houses it feeds —
    otherwise on the ray through the group's far-end building
    (:func:`_far_end`, audit W1B-01). The centroid ray of a group of about
    180 degrees passes the houses next to the substation but its cabinet
    lands far from every house (AP1: 10 houses and one small commercial
    building, 50.4 m); the groups of the default 50-house grid stand 8.8 …
    12.8 m from their houses on it and keep it (with the far-end building,
    six houses of the southern group were fed straight from the
    substation).

    The threshold is a step, not a separation of two populations (audit
    W1R2-02; user decision D7 keeps it for 2026.11): over the 80 AP1 mixes
    the groups beyond it stand 20.21 … 53.56 m from their houses on the
    centroid ray, the largest gap kept is 18.84 m, and a group that crosses
    it moves its cabinet by up to the width of the group (50-house grid with
    its columns 32.0 / 32.2 m apart: 141.8 m). Returns the position, the
    building named in a clearance error, the rule taken and the gap on the
    centroid ray, which the plan records per cabinet
    (:attr:`~groundfield.generators.plan.PlannedSite.cabinet_rule`,
    :attr:`~groundfield.generators.plan.PlannedSite.centroid_ray_gap_m`).
    """
    xy, holder = _centroid_end(substation_xy, group, geometry, clearance_m, radius_m)
    gap = _group_gap(xy, group, geometry, radius_m)
    if gap <= max_gap_m:
        return _CabinetPlace(xy, holder, "centroid_ray", gap)
    xy, holder = _far_end(substation_xy, group, geometry, clearance_m, radius_m)
    return _CabinetPlace(xy, holder, "far_end_ray", gap)


class _PlanShapes(NamedTuple):
    """Electrodes of one site in plan, for the clearance check (audit W1B-02)."""

    name: str
    segments: np.ndarray
    """Straight wire pieces, shape ``(n, 2, 2)`` in m."""
    circles: list[tuple[float, float, float]]
    """``RingElectrodeSpec`` circles ``(x, y, r)`` in m."""
    points: np.ndarray
    """Rod heads, shape ``(n, 2)`` in m."""
    outlines: list[Outline]
    """Areas enclosed by its closed horizontal electrodes."""
    box: tuple[float, float, float, float]
    """Bounding box ``(x_min, y_min, x_max, y_max)`` in m."""


def _plan_shapes(site: PlannedSite, geometry: SiteGeometry) -> Optional[_PlanShapes]:
    """The electrodes of ``site`` in plan (``None`` without electrodes)."""
    if not site.electrodes:
        return None
    pieces = [
        (a, b)
        for wire in geometry.wires
        if wire.circle is None
        for a, b in zip(wire.xy[:-1], wire.xy[1:], strict=True)
    ]
    circles = [wire.circle for wire in geometry.wires if wire.circle is not None]
    rods = _rod_points(site)
    points, _ = _plan_extent(site, geometry)
    xs = [x for x, _ in points] + [x - r for x, _, r in circles]
    xs += [x + r for x, _, r in circles]
    ys = [y for _, y in points] + [y - r for _, y, r in circles]
    ys += [y + r for _, y, r in circles]
    return _PlanShapes(
        site.name,
        np.asarray(pieces, dtype=float).reshape(-1, 2, 2),
        circles,
        np.asarray(rods, dtype=float).reshape(-1, 2),
        list(geometry.outlines),
        (min(xs), min(ys), max(xs), max(ys)),
    )


def _gap_to(point: tuple[float, float], shapes: _PlanShapes) -> float:
    """Plan distance in m from ``point`` to the electrodes of a site (0 when
    it stands inside one of their closed outlines)."""
    if inside_outlines(point, shapes.outlines):
        return 0.0
    p = np.asarray(point, dtype=float)
    gap = math.inf
    if len(shapes.segments):
        a, b = shapes.segments[:, 0], shapes.segments[:, 1]
        d = b - a
        dd = np.einsum("ij,ij->i", d, d)
        t = np.einsum("ij,ij->i", p - a, d) / np.where(dd > 0.0, dd, 1.0)
        foot = a + np.clip(t, 0.0, 1.0)[:, None] * d
        gap = float(np.min(np.hypot(*(p - foot).T)))
    if len(shapes.points):
        gap = min(gap, float(np.min(np.hypot(*(p - shapes.points).T))))
    for x, y, r in shapes.circles:
        gap = min(gap, abs(math.hypot(p[0] - x, p[1] - y) - r))
    return gap


def _check_cabinet_clearance(
    cabinets: list[PlannedSite],
    radii: list[float],
    far_ends: list[PlannedSite],
    others: list[PlannedSite],
    geometry: dict[str, SiteGeometry],
    clearance_m: float,
) -> None:
    """Raise when a cabinet's electrodes come closer than ``clearance_m`` to
    those of another site (audit W1B-02).

    A cabinet's electrodes lie within its plan radius of its position; they
    keep the clearance from its far-end building by construction
    (:func:`_far_end`), but another building (of its group or of another
    one), the substation or another cabinet may stand near the ray beyond
    it. The check counts every electrode of those sites — horizontal wires,
    circles, rods — and a cabinet that stands inside one of their closed
    outlines.
    """
    shapes = [
        s
        for s in (
            _plan_shapes(site, geometry[site.name])
            for site in [*others, *cabinets]
            if site.name in geometry
        )
        if s is not None
    ]
    if not shapes:
        return
    boxes = np.asarray([s.box for s in shapes], dtype=float)
    for cab, radius, far in zip(cabinets, radii, far_ends, strict=True):
        if not cab.electrodes:
            continue
        cx, cy = cab.position_xy
        need = clearance_m + radius
        tol = 1e-9 * max(1.0, abs(cx), abs(cy), need)
        gx = np.maximum.reduce(
            [boxes[:, 0] - cx, np.zeros(len(boxes)), cx - boxes[:, 2]]
        )
        gy = np.maximum.reduce(
            [boxes[:, 1] - cy, np.zeros(len(boxes)), cy - boxes[:, 3]]
        )
        near = np.flatnonzero(np.hypot(gx, gy) < need - tol)
        blocked = []
        for k in near:
            other = shapes[int(k)]
            if other.name == cab.name:
                continue
            gap = _gap_to((cx, cy), other)
            if gap < need - tol:
                blocked.append(
                    f"{other.name!r} (inside its electrodes)"
                    if gap == 0.0
                    else f"{other.name!r} ({max(0.0, gap - radius):.3g} m)"
                )
        if blocked:
            raise ValueError(
                f"TnNetworkGenerator: the cable cabinet {cab.name!r} of the "
                f"building group of {far.name!r} (the building it stands beyond) "
                "stands at "
                f"({cx:.3f}, {cy:.3f}) m, CabinetConfig.end_clearance_m = "
                f"{clearance_m:g} m beyond that building, but "
                + (
                    f"its earth electrodes (within {radius:g} m of that point) come"
                    if radius > 0.0
                    else "its earth electrode there comes"
                )
                + f" closer than {clearance_m:g} m to those of "
                f"{', '.join(blocked)}. Give the "
                "cabinets a CabinetConfig.placement, or change end_clearance_m "
                "or the cabinet count (audit W1B-02)."
            )


def _router(
    cfg: "TnNetworkConfig",
    sites: list[PlannedSite],
    geometry: Optional[dict[str, SiteGeometry]] = None,
) -> PenRouter:
    """PEN router over ``sites`` (the first is the substation).

    A site with horizontal earth electrodes (foundation, ring, strip) is met
    at the point of their wires nearest to each cable (user decisions Q41,
    Q43); a site with rods only at the connection point of its anchor.
    ``geometry`` holds the sites already materialised for it.
    """
    node_xy = {}
    wires = {}
    geometry = geometry or {}
    for site in sites:
        known = geometry.get(site.name)
        point, horizontal, _ = known if known is not None else _site_geometry(site)
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

        Equivalent to ``materialise_plan(self.plan(cfg, rng=rng))``, with
        one exception: right after :meth:`plan` on this generator — without
        ``rng`` in either call, nothing drawn from the generator's RNG since
        and the same configuration — it materialises the plan that
        :meth:`plan` returned (once; the next call draws the next
        realisation). Before, it drew a second realisation, which differs
        from the plan with placement jitter or other per-site randomness
        (audit W3R2B).

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
        planned = self._take_last_plan(cfg) if rng is None else None
        if planned is None:
            planned = self.plan(cfg, rng=rng)
            if rng is None:
                # a plan nobody has seen: the next build() draws anew
                self._last_plan = None
        return materialise_plan(planned)

    def _take_last_plan(self, cfg: Optional[TnNetworkConfig]) -> Optional[NetworkPlan]:
        """The plan :meth:`plan` returned last, if :meth:`build` may use it.

        It may when the generator's RNG is the one that drew it and has not
        drawn since (its state is unchanged) and the configuration equals
        the one it was drawn for; the plan is handed out once.
        """
        last = getattr(self, "_last_plan", None)  # pickles of 0.15.0 have none
        self._last_plan = None
        if last is None:
            return None
        rng, state, drawn_for, planned = last
        if self._rng is not rng or self._rng.bit_generator.state != state:
            return None
        if (cfg or self.cfg) != drawn_for:
            return None
        return planned

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
        draws no further random numbers. A plan drawn from the generator's
        RNG is remembered (a copy): :meth:`build` right after it gives its
        world (audit W3R2B).

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
        own_stream = rng is None
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
            groups: Optional[list[list[PlannedSite]]] = None
            geometry: dict[str, SiteGeometry] = {}
            cabinets: list[PlannedSite]
            if cfg.cabinets.placement is not None:
                cabinets = [
                    cfg.cabinets.grounding.plan_site(
                        f"kvs_{k}", role="cabinet", position_xy=site_xy, rng=rng
                    )
                    for k, site_xy in enumerate(
                        cfg.cabinets.placement.generate(n_cabinets, rng)
                    )
                ]
            else:
                # One cabinet per building group, at the far end of the
                # houses it feeds (user decisions Q22, Q35).
                geometry = {
                    site.name: _site_geometry(site) for site in [substation, *buildings]
                }
                groups, cabinets = self._group_cabinets(
                    cfg, substation, buildings, n_cabinets, geometry, rng
                )
            sites.extend(cabinets)
            if not any(c.anchor is not None for c in cabinets):
                raise ValueError(
                    "TnNetworkGenerator: every cable-cabinet grounding had zero "
                    "present electrodes. Increase presence_prob or set "
                    "fixed_count > 0."
                )
            self._plan_star_backbone(
                cfg,
                substation,
                cabinets,
                buildings,
                branches,
                junctions,
                groups=groups,
                geometry=geometry,
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
        plan = NetworkPlan(
            name=cfg.name,
            soil=soil,
            sites=sites,
            junctions=junctions,
            branches=branches,
            measurement=measurement,
            sources=[source],
        )
        if own_stream:
            # build() right after this call materialises this plan instead
            # of drawing another realisation (audit W3R2B); copies, so that
            # changing the returned plan or the configuration in place
            # cannot change what build() gives
            self._last_plan = (
                rng,
                rng.bit_generator.state,
                cfg.model_copy(deep=True),
                plan.model_copy(deep=True),
            )
        return plan

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

    def _group_cabinets(
        self,
        cfg: TnNetworkConfig,
        substation: PlannedSite,
        buildings: list[PlannedSite],
        n_cabinets: int,
        geometry: dict[str, SiteGeometry],
        rng: np.random.Generator,
    ) -> tuple[list[list[PlannedSite]], list[PlannedSite]]:
        """Building groups and their cabinets (Q22, Q35; audits W1B-01, W1B-02).

        The wired buildings are grouped by
        :meth:`~groundfield.generators.pen_topology.StarCabinetTopology.group_buildings`;
        group ``k`` is fed by cabinet ``kvs_<k>``, which stands where
        :func:`_cabinet_position` puts it, its own grounding's plan extent
        included. With more cabinets than buildings every building is a
        group of its own and the remaining cabinets are not placed (a
        :class:`UserWarning`). The plan geometry of every cabinet is added to
        ``geometry``.

        The plan of each cabinet records the rule that placed it and the
        gap on the centroid ray (``cabinet_rule``, ``centroid_ray_gap_m``,
        audit W1R2-02).

        Raises
        ------
        ValueError
            If the drawn ``end_clearance_m`` or ``max_far_end_gap_m`` is not
            positive and finite, if the drawn ``end_clearance_m`` is not
            smaller than the drawn ``max_far_end_gap_m`` (audit W1R2-09), or
            if a cabinet's electrodes come closer than ``end_clearance_m``
            to those of another site (:func:`_check_cabinet_clearance`).
        """
        wired = [b for b in buildings if b.anchor is not None]
        if n_cabinets > len(wired):
            warnings.warn(
                f"TnNetworkGenerator: {n_cabinets} cable cabinets for "
                f"{len(wired)} buildings; one cabinet per building group places "
                f"{len(wired)} of them.",
                UserWarning,
                stacklevel=4,
            )
        origin = (float(substation.position_xy[0]), float(substation.position_xy[1]))
        index_groups = StarCabinetTopology.group_buildings(
            origin, [b.position_xy for b in wired], n_cabinets
        )
        clearance = _to_float(cfg.cabinets.end_clearance_m, rng)
        if not (math.isfinite(clearance) and clearance > 0.0):
            raise ValueError(
                "TnNetworkGenerator: CabinetConfig.end_clearance_m must be a "
                f"positive, finite distance in m; drew {clearance!r} (audit W1B-05)."
            )
        max_gap = _to_float(cfg.cabinets.max_far_end_gap_m, rng)
        if not (math.isfinite(max_gap) and max_gap > 0.0):
            raise ValueError(
                "TnNetworkGenerator: CabinetConfig.max_far_end_gap_m must be a "
                f"positive, finite distance in m; drew {max_gap!r}."
            )
        if clearance >= max_gap:
            raise ValueError(
                _clearance_not_below_gap("TnNetworkGenerator: ", clearance, max_gap)
            )
        groups = [[wired[i] for i in group] for group in index_groups]
        cabinets: list[PlannedSite] = []
        radii: list[float] = []
        far_ends: list[PlannedSite] = []
        for k, group in enumerate(groups):
            # The cabinet's electrodes are drawn first (in the order of the
            # former placement), at its position relative to itself, so the
            # plan extent of its grounding is known before it is placed.
            local = cfg.cabinets.grounding.plan_site(
                f"kvs_{k}", role="cabinet", position_xy=(0.0, 0.0), rng=rng
            )
            radius = _plan_radius(local, _site_geometry(local))
            place = _cabinet_position(
                origin, group, geometry, clearance, radius, max_gap
            )
            # The plan records which rule placed the cabinet and the gap it
            # was decided on (audit W1R2-02, user decision D7).
            cabinet = local.model_copy(
                update={
                    "position_xy": place.xy,
                    "cabinet_rule": place.rule,
                    "centroid_ray_gap_m": place.gap_m,
                }
            )
            geometry[cabinet.name] = _site_geometry(cabinet)
            cabinets.append(cabinet)
            radii.append(radius)
            far_ends.append(place.holder)
        _check_cabinet_clearance(
            cabinets, radii, far_ends, [substation, *buildings], geometry, clearance
        )
        return groups, cabinets

    def _plan_star_backbone(
        self,
        cfg: TnNetworkConfig,
        substation: PlannedSite,
        cabinets: list[PlannedSite],
        buildings: list[PlannedSite],
        branches: list[PlannedBranch],
        junctions: list[PlannedJunction],
        *,
        groups: Optional[list[list[PlannedSite]]] = None,
        geometry: Optional[dict[str, SiteGeometry]] = None,
    ) -> None:
        """Star backbone: substation → each cabinet → its buildings.

        Every cabinet has its own straight cable from the substation
        (``pen_main_<anchor>``; collinear cabinets get separate cables in
        one trench). A building that a main crosses is joined to it and fed
        by it, without a drop: by the nearest of the cabinets whose mains
        cross it, ties to the lower cabinet index (audit R4A-02). Every
        other building is fed by the cabinet of its building group
        (``groups[k]`` by ``cabinets[k]``, user decisions Q22, Q35), or —
        with a given cabinet placement, or when its group's cabinet has no
        electrodes — by its nearest cabinet (Manhattan metric between the
        site positions); its service drop taps that cabinet's cable at the
        point nearest to the building — the closest pair of cable and the
        building's horizontal wires (user decisions Q41, Q43), or the point
        nearest to the anchor of a building with rods only — at a
        leakage-free tap junction ``pen_tap_<k>``, or the substation / the
        cabinet when the nearest point is an end of the cable, never
        another building's drop, and a drop from the cabinet leaves another
        main it would run along at that main's node (audit R4B-01). The
        full rules are in :mod:`._pen_routing`.
        """
        wired = [c for c in cabinets if c.anchor is not None]
        router = _router(cfg, [substation, *wired, *buildings], geometry)
        mains: dict[str, str] = {}
        for cab in wired:
            mains[cab.name] = f"pen_main_{cab.anchor}"
            router.add_cable(substation.name, cab.name, name=mains[cab.name])
        fed: dict[str, list[PlannedSite]] = {c.name: [] for c in wired}
        cabinet_of = {mains[c.name]: c for c in wired}
        group_cabinet: dict[str, PlannedSite] = {}
        for group, cab in zip(groups or [], cabinets, strict=False):
            if cab.anchor is not None:
                group_cabinet.update((b.name, cab) for b in group)
        for b in buildings:
            if b.anchor is None:
                continue
            bx, by = b.position_xy
            # A building that a main crosses is joined to it and fed by it
            # (no drop); several: the nearest of their cabinets, ties to the
            # lower cabinet index. Any other building is fed by the cabinet
            # of its group, else by its nearest cabinet (audit R4A-02).
            crossed = [
                cabinet_of[m]
                for m in router.crossing(b.name, [mains[c.name] for c in wired])
            ]
            if not crossed and b.name in group_cabinet:
                fed[group_cabinet[b.name].name].append(b)
                continue
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
           building in a substation slot taps its feeder's whole trunk the
           same way (user decisions Q22, Q35), and starts at the
           substation only on a feeder without a trunk. A
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
        # The whole trunk of the feeder of every building in a substation
        # slot, which it taps (user decisions Q22, Q35).
        slot_paths: dict[str, list[str]] = {}
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
                if tap_idx == 0:
                    slot_paths[wired[building_idx].name] = path
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
            # A building in a substation slot taps its feeder's trunk (user
            # decisions Q22, Q35; up to then its drop started at the
            # substation); without a trunk (no cabinet on the feeder) it is
            # still fed from the substation. Any other building taps the
            # trunk from the substation to its tap source.
            own = {
                name: (slot_paths[name] if tap == substation.name else paths[tap])
                for name in by_name
            }
            # For the crossing rule every trunk counts as a feeding cable: a
            # trunk that crosses a building is joined there and the building
            # gets no drop -- a trunk beyond a building's tap source runs on
            # past it (audit R4A-02).
            for name in router.drop_order(
                list(by_name), paths[tap], tap, crossing=trunks, paths=own
            ):
                router.connect_drop(
                    name,
                    own[name],
                    tap,
                    name=f"pen_service_{by_name[name].anchor}",
                    crossing=trunks,
                )
        branches.extend(router.branches())
        junctions.extend(router.junctions)
