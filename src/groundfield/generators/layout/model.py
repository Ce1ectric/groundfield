r"""The LV network layout builder (front-end of :mod:`groundfield.generators.layout`).

Where :class:`~groundfield.generators.tn_network.TnNetworkGenerator`
generates *stochastic* reference networks from population parameters,
the layout builder composes a *single* deterministic network in the order
an engineer draws it on a plan:

1. ingest building footprints (:mod:`groundfield.geo.osm` or any other
   :class:`~groundfield.geo.footprint.BuildingFootprint` source); they are
   house positions and hard obstacles for the PEN router at once;
2. place the substation and, optionally, cable cabinets;
3. lay PEN cables (:meth:`add_pen_cable`, Manhattan A\* around the
   footprints) and tap every house to the nearest cable
   (:meth:`connect_buildings`, :meth:`connect_all_buildings`);
4. optionally add the measurement set-up (auxiliary electrode and voltage
   probe);
5. export: :meth:`plan` returns the
   :class:`~groundfield.generators.plan.NetworkPlan`, :meth:`to_world`
   materialises it through the shared materialiser
   (:func:`~groundfield.generators.materialise.materialise_plan`).

The layout itself is a Pydantic snapshot (JSON-serialisable). The
routing, measurement, penetration-mask and plotting methods live in the
sibling modules of this package.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundfield.generators._compat import (
    rename_input_keys,
    take_renamed_kwargs,
    warn_renamed,
)
from groundfield.generators.electrode_specs import (
    FoundationElectrodeSpec,
    RingElectrodeSpec,
    RodElectrodeSpec,
    rod_circle,
)
from groundfield.generators.grounding import FoundationOutline, GroundingSystemSpec
from groundfield.generators.layout.measurement import (
    AuxiliaryElectrodePlacement,
    VoltageProbePlacement,
    _MeasurementMixin,
)
from groundfield.generators.layout.penetration import _PenetrationMixin
from groundfield.generators.layout.plot import _PlotMixin
from groundfield.generators.layout.routing import (
    BuildingConnection,
    PenCable,
    _RoutingMixin,
)
from groundfield.generators.materialise import materialise_plan
from groundfield.generators.plan import (
    ConductorParameters,
    NetworkPlan,
    PlannedBranch,
    PlannedJunction,
    PlannedMeasurement,
    PlannedSite,
    PlannedSource,
)
from groundfield.geo.footprint import BuildingFootprint
from groundfield.geo.projection import Projector
from groundfield.soil.models import HomogeneousSoil, SoilModel
from groundfield.world import World

__all__ = [
    "CabinetPlacement",
    "LvNetworkLayout",
]


# ---------------------------------------------------------------------
# Default grounding factories (kept in sync with the TnNetwork
# generator defaults so the two pipelines compose with each other).
# ---------------------------------------------------------------------


def _default_substation_grounding() -> GroundingSystemSpec:
    """Ring (4 m) + 4 driven rods on a 2 m inner circle."""
    return GroundingSystemSpec(
        electrodes=[
            RingElectrodeSpec(radius_m=4.0, depth_m=0.6),
            *rod_circle(n=4, radius_m=2.0, length_m=2.5),
        ],
    )


def _default_cabinet_grounding() -> GroundingSystemSpec:
    """Single 1.5 m driven rod."""
    return GroundingSystemSpec(
        electrodes=[RodElectrodeSpec(length_m=1.5, depth_m=0.0)],
    )


def _default_house_grounding() -> GroundingSystemSpec:
    """10 m foundation electrode with a 2 x 2 mesh."""
    return GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                size_m=10.0,
                depth_m=0.8,
                n_x=2,
                n_y=2,
            ),
        ],
    )


# ---------------------------------------------------------------------
# Snapshot data classes
# ---------------------------------------------------------------------


class CabinetPlacement(BaseModel):
    """One user-placed cable cabinet (German *KVS*).

    Attributes
    ----------
    name
        Cabinet name; also the name of its grounding site.
    position_xy
        Cabinet position ``(x, y)`` in m.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    position_xy: tuple[float, float]


class KvsPlacement(CabinetPlacement):
    """Deprecated alias of :class:`CabinetPlacement`."""

    def model_post_init(self, context: object, /) -> None:
        warn_renamed("KvsPlacement", "CabinetPlacement", stacklevel=5)
        super().model_post_init(context)


# ---------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------


class LvNetworkLayout(
    _RoutingMixin, _MeasurementMixin, _PenetrationMixin, _PlotMixin, BaseModel
):
    r"""Imperative LV network layout builder.

    The builder is constructed empty (no buildings, no substation,
    no cables) and filled up by calling its methods in the order
    a planner would naturally take. Every state-changing method
    returns the new object (or its key) so the calls can be
    chained or stored:

    >>> layout = LvNetworkLayout.from_footprints(footprints,
    ...                                         substation_xy=(0, 0))
    >>> cabinet_1 = layout.add_cabinet((120, 0))
    >>> cable = layout.add_pen_cable(start="substation",
    ...                              end_xy=(250, 0))
    >>> layout.connect_buildings()
    >>> ax = layout.plot()

    Attributes
    ----------
    footprints
        Building footprints (already projected into the local
        ENU frame). Their axis-aligned bounding rectangles serve
        both as house positions *and* as PEN-routing obstacles.
    substation_xy
        Substation centre in metres.
    substation_name
        Substation anchor id used everywhere else (default
        ``"substation"``).
    cabinet_placements
        Optional pre-populated cabinet list. Most callers leave this
        empty and use :meth:`add_cabinet`.
    pen_cables
        Optional pre-populated PEN cables. Most callers leave
        this empty and use :meth:`add_pen_cable` /
        :meth:`add_pen_trunk`.
    connections
        Optional pre-populated building connections. Most callers
        leave this empty and use :meth:`connect_buildings`.
    obstacle_clearance_m
        Inflation applied to every footprint AABB before routing.
        Defaults to 1 m so cables keep a small visible distance
        from the foundation walls.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    footprints: list[BuildingFootprint]
    substation_xy: tuple[float, float]
    substation_name: str = "substation"
    cabinet_placements: list[CabinetPlacement] = Field(default_factory=list)
    pen_cables: list[PenCable] = Field(default_factory=list)
    connections: list[BuildingConnection] = Field(default_factory=list)
    auxiliary_electrode: Optional[AuxiliaryElectrodePlacement] = Field(
        default=None,
        description=(
            "Optional auxiliary current electrode (Hilfserder) used "
            "for the simulated fall-of-potential measurement. Set "
            "via :meth:`add_auxiliary_electrode`; when present, "
            ":meth:`to_world` wires the substation-side current "
            "source to return through this electrode instead of the "
            "remote-earth boundary."
        ),
    )
    voltage_probe: Optional[VoltageProbePlacement] = Field(
        default=None,
        description=(
            "Optional voltage probe (Spannungssonde) used as the "
            "ground-potential reference in the simulated measurement. "
            "Configured via :meth:`add_voltage_probe`; consumed by "
            ":meth:`measured_grounding_impedance` to compute "
            ":math:`Z_\\text{meas} = (\\varphi_\\text{sub} - "
            "\\varphi_\\text{probe}) / I_\\text{src}`. The probe is a "
            "pure sampling point (no electrode in the world) so the "
            "ideal-voltmeter assumption is preserved."
        ),
    )
    obstacle_clearance_m: float = Field(default=1.0, ge=0.0)
    frame_origin_lat_lon: Optional[tuple[float, float]] = Field(
        default=None,
        description=(
            "WGS84 origin (lat, lon) of the local ENU projection frame "
            "that the footprints and the substation are expressed in. "
            "Set automatically by :meth:`from_osm`; pass it explicitly "
            "when constructing the layout from already-projected data "
            "if you want to add cabinets or PEN endpoints via lat/lon later."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _renamed_fields(cls, data: object) -> object:
        return rename_input_keys(
            data, {"kvs_placements": "cabinet_placements"}, "LvNetworkLayout"
        )

    @property
    def kvs_placements(self) -> list[CabinetPlacement]:
        """Deprecated alias of :attr:`cabinet_placements`."""
        warn_renamed(
            "LvNetworkLayout.kvs_placements", "LvNetworkLayout.cabinet_placements"
        )
        return self.cabinet_placements

    @classmethod
    def from_footprints(
        cls,
        footprints: list[BuildingFootprint],
        *,
        substation_xy: tuple[float, float],
        substation_name: str = "substation",
        obstacle_clearance_m: float = 1.0,
        frame_origin_lat_lon: Optional[tuple[float, float]] = None,
    ) -> "LvNetworkLayout":
        """Build an empty layout from a list of footprints.

        ``frame_origin_lat_lon`` is optional; pass it when the
        footprints were projected by an external pipeline and you
        still want to add cabinets / PEN endpoints by lat / lon later.
        """
        return cls(
            footprints=list(footprints),
            substation_xy=substation_xy,
            substation_name=substation_name,
            obstacle_clearance_m=obstacle_clearance_m,
            frame_origin_lat_lon=frame_origin_lat_lon,
        )

    @classmethod
    def from_osm(
        cls,
        *,
        center_lat_deg: float,
        center_lon_deg: float,
        radius_m: float,
        substation_lat_lon: Optional[tuple[float, float]] = None,
        substation_xy: Optional[tuple[float, float]] = None,
        cabinet_lat_lons: Optional[list[tuple[float, float]]] = None,
        cabinet_xys: Optional[list[tuple[float, float]]] = None,
        substation_name: str = "substation",
        obstacle_clearance_m: float = 1.0,
        min_area_m2: float = 16.0,
        cache_dir=None,
        force_refresh: bool = False,
        endpoint: Optional[str] = None,
        timeout_s: Optional[int] = None,
        max_retries: int = 1,
        exclude_building_tags: Optional[Collection[str]] = None,
        **deprecated: object,
    ) -> "LvNetworkLayout":
        """One-shot constructor: query OSM and pre-place anchors.

        Parameters
        ----------
        center_lat_deg, center_lon_deg
            Centre of the buildings-in-radius Overpass query. Also
            the WGS84 origin of the local ENU frame
            :attr:`frame_origin_lat_lon` so subsequent
            ``position_lat_lon`` arguments stay consistent.
        radius_m
            Search radius in metres.
        substation_lat_lon
            Optional substation location as WGS84 ``(lat, lon)``.
            Mutually exclusive with ``substation_xy``. When both
            are omitted the substation defaults to the local
            origin ``(0, 0)``.
        substation_xy
            Substation location in already-projected metres.
            Mutually exclusive with ``substation_lat_lon``.
        cabinet_lat_lons
            Optional list of cabinet locations in WGS84 ``(lat, lon)``.
            Each entry is projected and appended to
            :attr:`cabinet_placements`. Names are auto-generated.
        cabinet_xys
            Optional list of cabinet locations in already-projected
            metres. Mutually exclusive with ``cabinet_lat_lons``.
        substation_name
            Anchor id for the substation. Default ``"substation"``.
        obstacle_clearance_m, min_area_m2
            Forwarded to the layout / OSM query respectively.
        cache_dir, force_refresh, endpoint, timeout_s, max_retries
            Forwarded to :func:`groundfield.geo.osm.query_and_project`.
        exclude_building_tags
            ``building=*`` values to drop, e.g.
            :data:`groundfield.geo.osm.NON_SERVICE_BUILDING_TAGS`
            (garages, sheds, carports, ...). ``None`` (default) keeps
            every element with a ``building`` tag.

        Returns
        -------
        LvNetworkLayout
            Layout with footprints loaded, projector seeded, and
            substation / cabinets pre-placed. Routing and connections
            are still up to the caller.

        Examples
        --------
        Build the layout for Mulmke (Heudeber / Nordharz) in a
        300 m radius around the village centre, with the
        substation at a chosen lat / lon and two cabinets along the
        main road:

        >>> layout = LvNetworkLayout.from_osm(
        ...     center_lat_deg=51.9136, center_lon_deg=10.8432,
        ...     radius_m=300.0,
        ...     substation_lat_lon=(51.9138, 10.8418),
        ...     cabinet_lat_lons=[
        ...         (51.9133, 10.8445),
        ...         (51.9140, 10.8460),
        ...     ],
        ... )
        """
        from groundfield.geo.osm import query_and_project

        renamed = take_renamed_kwargs(
            "LvNetworkLayout.from_osm",
            deprecated,
            {"kvs_lat_lons": "cabinet_lat_lons", "kvs_xys": "cabinet_xys"},
        )
        cabinet_lat_lons = renamed.get("cabinet_lat_lons", cabinet_lat_lons)  # type: ignore[assignment]
        cabinet_xys = renamed.get("cabinet_xys", cabinet_xys)  # type: ignore[assignment]
        if substation_lat_lon is not None and substation_xy is not None:
            raise ValueError(
                "LvNetworkLayout.from_osm: substation_lat_lon and "
                "substation_xy are mutually exclusive."
            )
        if cabinet_lat_lons is not None and cabinet_xys is not None:
            raise ValueError(
                "LvNetworkLayout.from_osm: cabinet_lat_lons and cabinet_xys "
                "are mutually exclusive."
            )

        query_kwargs: dict = {
            "min_area_m2": min_area_m2,
            "force_refresh": force_refresh,
            "max_retries": max_retries,
        }
        if exclude_building_tags is not None:
            query_kwargs["exclude_building_tags"] = exclude_building_tags
        if cache_dir is not None:
            query_kwargs["cache_dir"] = cache_dir
        if endpoint is not None:
            query_kwargs["endpoint"] = endpoint
        if timeout_s is not None:
            query_kwargs["timeout_s"] = timeout_s

        footprints, projector = query_and_project(
            lat0_deg=center_lat_deg,
            lon0_deg=center_lon_deg,
            radius_m=radius_m,
            **query_kwargs,
        )

        if substation_lat_lon is not None:
            sub_xy = projector.to_xy_m(*substation_lat_lon)
        elif substation_xy is not None:
            sub_xy = tuple(substation_xy)
        else:
            sub_xy = (0.0, 0.0)

        layout = cls(
            footprints=footprints,
            substation_xy=sub_xy,
            substation_name=substation_name,
            obstacle_clearance_m=obstacle_clearance_m,
            frame_origin_lat_lon=(center_lat_deg, center_lon_deg),
        )

        if cabinet_lat_lons:
            for lat, lon in cabinet_lat_lons:
                layout.add_cabinet(position_lat_lon=(lat, lon))
        elif cabinet_xys:
            for xy in cabinet_xys:
                layout.add_cabinet(position_xy=xy)

        return layout

    @property
    def projector(self) -> Optional[Projector]:
        """Lazy :class:`Projector` for the layout's ENU frame.

        Returns ``None`` when :attr:`frame_origin_lat_lon` is not
        set (the layout is in an anonymous local frame). The
        projector is reconstructed on every access; pyproj caches
        the underlying CRS objects so this stays cheap.
        """
        if self.frame_origin_lat_lon is None:
            return None
        lat, lon = self.frame_origin_lat_lon
        return Projector(lat0_deg=lat, lon0_deg=lon)

    def lat_lon_to_xy(
        self,
        lat_deg: float,
        lon_deg: float,
    ) -> tuple[float, float]:
        """Project a WGS84 ``(lat, lon)`` to the layout's local frame.

        Raises :class:`RuntimeError` when no
        :attr:`frame_origin_lat_lon` has been set (i.e. the layout
        was not built via :meth:`from_osm`).
        """
        projector = self.projector
        if projector is None:
            raise RuntimeError(
                "LvNetworkLayout.lat_lon_to_xy: layout has no "
                "frame_origin_lat_lon. Build via ``from_osm`` or "
                "pass ``frame_origin_lat_lon`` to ``from_footprints`` "
                "to enable lat / lon based placement."
            )
        return projector.to_xy_m(lat_deg, lon_deg)

    def xy_to_lat_lon(
        self,
        x_m: float,
        y_m: float,
    ) -> tuple[float, float]:
        """Inverse projection: local frame ``(x, y)`` → WGS84.

        Mirrors :meth:`lat_lon_to_xy`.
        """
        projector = self.projector
        if projector is None:
            raise RuntimeError(
                "LvNetworkLayout.xy_to_lat_lon: layout has no "
                "frame_origin_lat_lon. Build via ``from_osm`` first."
            )
        return projector.to_lat_lon(x_m, y_m)

    def add_cabinet(
        self,
        position_xy: Optional[tuple[float, float]] = None,
        *,
        position_lat_lon: Optional[tuple[float, float]] = None,
        name: Optional[str] = None,
    ) -> str:
        """Place a cable cabinet at ``position_xy`` or ``position_lat_lon``.

        Exactly one of ``position_xy`` (local ENU metres) or
        ``position_lat_lon`` (WGS84 degrees) must be set. The lat /
        lon variant projects through the layout's
        :attr:`frame_origin_lat_lon`; if no frame origin is set, a
        :class:`RuntimeError` is raised.

        Returns the assigned cabinet name (auto-generated when ``name``
        is ``None``). The returned name is what callers pass to
        :meth:`add_pen_cable` as ``start=...``.
        """
        if (position_xy is None) == (position_lat_lon is None):
            raise ValueError(
                "LvNetworkLayout.add_cabinet: exactly one of "
                "``position_xy`` and ``position_lat_lon`` must be set."
            )
        if position_lat_lon is not None:
            position_xy = self.lat_lon_to_xy(*position_lat_lon)
        if name is None:
            name = f"kvs_{len(self.cabinet_placements)}"
        if any(k.name == name for k in self.cabinet_placements):
            raise ValueError(
                f"LvNetworkLayout.add_cabinet: name {name!r} is already taken."
            )
        if name == self.substation_name:
            raise ValueError(
                f"LvNetworkLayout.add_cabinet: name {name!r} clashes with "
                "the substation name."
            )
        assert position_xy is not None
        self.cabinet_placements.append(
            CabinetPlacement(name=name, position_xy=tuple(position_xy)),
        )
        return name

    def add_kvs(
        self,
        position_xy: Optional[tuple[float, float]] = None,
        *,
        position_lat_lon: Optional[tuple[float, float]] = None,
        name: Optional[str] = None,
    ) -> str:
        """Deprecated alias of :meth:`add_cabinet`."""
        warn_renamed("LvNetworkLayout.add_kvs", "LvNetworkLayout.add_cabinet")
        return self.add_cabinet(
            position_xy, position_lat_lon=position_lat_lon, name=name
        )

    def _anchor_position(self, name: str) -> tuple[float, float]:
        if name == self.substation_name:
            return tuple(self.substation_xy)
        for k in self.cabinet_placements:
            if k.name == name:
                return tuple(k.position_xy)
        raise KeyError(
            f"LvNetworkLayout: anchor {name!r} is unknown. "
            f"Known anchors: [{self.substation_name!r}] + "
            f"{[k.name for k in self.cabinet_placements]!r}."
        )

    # ------------------------------------------------------------------
    # Export: network plan and world
    # ------------------------------------------------------------------

    def plan(
        self,
        *,
        name: str = "ortsnetz_layout",
        soil: Optional[SoilModel] = None,
        substation_grounding: Optional[GroundingSystemSpec] = None,
        cabinet_grounding: Optional[GroundingSystemSpec] = None,
        house_grounding: Optional[GroundingSystemSpec] = None,
        pen_wire_radius_m: float = 0.005,
        pen_segment_length_m: Optional[float] = 5.0,
        include_source: bool = True,
        source_magnitude_a: float = 1.0,
        seed: int = 0,
        foundation_mask: Optional[list[bool]] = None,
        foundation_outline: FoundationOutline = "ombr",
        **deprecated: object,
    ) -> NetworkPlan:
        """Resolve the layout into a :class:`~groundfield.generators.plan.NetworkPlan`.

        Parameters are those of :meth:`to_world`. Random draws (presence
        probabilities and distributions in the grounding specs) use
        ``numpy.random.default_rng(seed)`` in the order substation,
        cabinets, houses, auxiliary electrode. Each house's foundation
        electrodes are shaped after its footprint once its draws are made
        (see ``foundation_outline`` in :meth:`to_world`), so the outline
        does not change which optional electrodes a seed produces.

        Returns
        -------
        NetworkPlan
            The concrete network. Every PEN cable becomes a chain of
            branches between junctions at its corners and tap points;
            cable ends at the substation or a cabinet attach to that
            site.

        Raises
        ------
        ValueError
            If ``foundation_mask`` has the wrong length.
        RuntimeError
            If the substation, a cabinet or the auxiliary electrode
            grounding produces zero electrodes.
        """
        renamed = take_renamed_kwargs(
            "LvNetworkLayout.plan",
            deprecated,
            {
                "kvs_grounding": "cabinet_grounding",
                "source_magnitude_A": "source_magnitude_a",
            },
        )
        cabinet_grounding = renamed.get("cabinet_grounding", cabinet_grounding)  # type: ignore[assignment]
        source_magnitude_a = renamed.get("source_magnitude_a", source_magnitude_a)  # type: ignore[assignment]
        if foundation_mask is not None and len(foundation_mask) != len(
            self.footprints,
        ):
            raise ValueError(
                "LvNetworkLayout.to_world: foundation_mask length "
                f"{len(foundation_mask)} does not match footprints "
                f"count {len(self.footprints)}."
            )
        rng = np.random.default_rng(seed)

        # 1. Substation grounding.
        substation_spec = substation_grounding or _default_substation_grounding()
        substation = substation_spec.plan_site(
            self.substation_name,
            role="substation",
            position_xy=self.substation_xy,
            rng=rng,
        )
        if substation.anchor is None:
            raise RuntimeError(
                "LvNetworkLayout.to_world: substation grounding "
                "produced zero electrodes."
            )
        sites: list[PlannedSite] = [substation]

        # 2. Cable-cabinet groundings.
        cabinet_spec = cabinet_grounding or _default_cabinet_grounding()
        for k in self.cabinet_placements:
            cabinet = cabinet_spec.plan_site(
                k.name, role="cabinet", position_xy=k.position_xy, rng=rng
            )
            if cabinet.anchor is None:
                raise RuntimeError(
                    "LvNetworkLayout.to_world: cabinet "
                    f"{k.name!r} grounding produced zero electrodes."
                )
            sites.append(cabinet)

        # 3. House groundings, shaped after the footprint
        #    (``GroundingSystemSpec.with_footprint``). Houses outside
        #    ``foundation_mask`` stay in the layout (plots) but not in the
        #    network.
        house_spec = house_grounding or _default_house_grounding()
        houses: dict[int, PlannedSite] = {}
        for i, fp in enumerate(self.footprints):
            if foundation_mask is not None and not foundation_mask[i]:
                continue
            centroid = fp.centroid_xy_m()
            house = house_spec.plan_site(
                f"house_{i:03d}",
                role="building",
                position_xy=centroid,
                rng=rng,
                footprint=fp,
                outline=foundation_outline,
            )
            sites.append(house)
            houses[i] = house

        # 4. Auxiliary current electrode, if configured.
        measurement: Optional[PlannedMeasurement] = None
        if self.auxiliary_electrode is not None:
            aux = self.auxiliary_electrode
            aux_site = aux.grounding.plan_site(
                aux.name, role="auxiliary", position_xy=aux.position_xy, rng=rng
            )
            if aux_site.anchor is None:
                raise RuntimeError(
                    "LvNetworkLayout.to_world: auxiliary electrode "
                    "grounding produced zero electrodes "
                    f"({aux.name!r}). Increase presence_prob."
                )
            measurement = PlannedMeasurement(auxiliary=aux_site)

        # 5. PEN cables: a junction at every interior corner and tap
        #    point, one branch between consecutive nodes.
        conductor = ConductorParameters(
            conductor_type="pen",
            wire_radius_m=pen_wire_radius_m,
            cross_section="from_radius",
            coupling_to_soil="isolated",
            segment_length_m=pen_segment_length_m,
        )
        junctions: list[PlannedJunction] = []
        branches: list[PlannedBranch] = []
        cable_nodes: dict[str, list[tuple[str, tuple[float, float]]]] = {}
        for cable in self.pen_cables:
            nodes = self._cable_nodes(cable, junctions)
            cable_nodes[cable.name] = nodes
            for j, ((a0, _p0), (a1, _p1)) in enumerate(
                zip(nodes[:-1], nodes[1:], strict=True)
            ):
                if a0 == a1:
                    continue
                branches.append(
                    PlannedBranch(
                        name=f"{cable.name}_seg_{j}",
                        role="pen",
                        start=a0,
                        end=a1,
                        conductor=conductor,
                        cable=cable.name,
                    )
                )

        # 6. Service drops: tap node on the cable -> house.
        for c in self.connections:
            house = houses.get(c.house_idx)
            if house is None or house.anchor is None:
                continue
            tap = self._tap_anchor_for(c, cable_nodes)
            if tap is None:
                continue
            branches.append(
                PlannedBranch(
                    name=f"service_{c.house_idx:03d}",
                    role="service_drop",
                    start=tap,
                    end=house.name,
                    conductor=conductor,
                    cable=c.cable_name,
                )
            )

        # 7. Sources. With an auxiliary electrode the source takes its
        #    current back there (``Source.return_to``, honoured by the
        #    solver since 2026.11, finding 10.16).
        sources: list[PlannedSource] = []
        if include_source:
            sources.append(
                PlannedSource(
                    attached_to=substation.name,
                    return_to=None
                    if measurement is None
                    else measurement.auxiliary.name,
                    magnitude=source_magnitude_a,
                )
            )

        return NetworkPlan(
            name=name,
            soil=soil or HomogeneousSoil(resistivity=100.0),
            sites=sites,
            junctions=junctions,
            branches=branches,
            measurement=measurement,
            sources=sources,
        )

    def to_world(
        self,
        *,
        name: str = "ortsnetz_layout",
        soil: Optional[SoilModel] = None,
        substation_grounding: Optional[GroundingSystemSpec] = None,
        cabinet_grounding: Optional[GroundingSystemSpec] = None,
        house_grounding: Optional[GroundingSystemSpec] = None,
        pen_wire_radius_m: float = 0.005,
        pen_segment_length_m: Optional[float] = 5.0,
        include_source: bool = True,
        source_magnitude_a: float = 1.0,
        seed: int = 0,
        foundation_mask: Optional[list[bool]] = None,
        foundation_outline: FoundationOutline = "ombr",
        **deprecated: object,
    ) -> World:
        """Materialise the layout into a :class:`groundfield.World`.

        Equivalent to ``materialise_plan(self.plan(...))``. Each
        :class:`PenCable` becomes a chain of straight conductors between
        junctions at its corners and tap points, so the cable follows its
        Manhattan polyline; each :class:`BuildingConnection` becomes one
        conductor from the tap junction to the house foundation.

        Parameters
        ----------
        name
            World name.
        soil
            Soil model; default homogeneous 100 Ω·m.
        substation_grounding, cabinet_grounding, house_grounding
            Grounding specs of the substation, every cabinet and every
            house (defaults: ring + 4 rods, one 1.5 m rod, 2 × 2
            foundation mesh shaped after the footprint).
        pen_wire_radius_m
            PEN wire radius in m.
        pen_segment_length_m
            Discretisation length in m of the PEN conductors (``None``:
            lumped).
        include_source
            Add the current source at the substation. With an
            auxiliary electrode it is a two-terminal source that
            returns its current through that electrode
            (``Source.return_to``; up to 0.15.0 a second, 180° source on
            the auxiliary electrode stood in for it).
        source_magnitude_a
            Source current in A.
        seed
            Seed of the random draws in the grounding specs.
        foundation_mask
            Optional per-house bool list. Only houses with ``True``
            receive a foundation electrode (and a service drop); the
            others stay in the layout for plotting. Combine with
            :meth:`foundation_mask` for a reproducible penetration sweep.
        foundation_outline
            How each footprint shapes the foundation electrodes of
            ``house_grounding`` (see
            :meth:`GroundingSystemSpec.with_footprint
            <groundfield.generators.grounding.GroundingSystemSpec.with_footprint>`):
            ``"ombr"`` (default) — the oriented minimum bounding
            rectangle of the footprint (size, orientation and centre),
            the same rule as the OSM placement of
            :class:`~groundfield.generators.tn_network.TnNetworkGenerator`;
            ``"polygon"`` — the real outline as a closed ring
            (``style="ring"`` foundations only; DIN 18014); ``"spec"`` —
            the spec geometry at the footprint centroid (every house got
            the same foundation before 2026.11).

        Returns
        -------
        World
            The materialised network.
        """
        renamed = take_renamed_kwargs(
            "LvNetworkLayout.to_world",
            deprecated,
            {
                "kvs_grounding": "cabinet_grounding",
                "source_magnitude_A": "source_magnitude_a",
            },
        )
        cabinet_grounding = renamed.get("cabinet_grounding", cabinet_grounding)  # type: ignore[assignment]
        source_magnitude_a = renamed.get("source_magnitude_a", source_magnitude_a)  # type: ignore[assignment]
        return materialise_plan(
            self.plan(
                name=name,
                soil=soil,
                substation_grounding=substation_grounding,
                cabinet_grounding=cabinet_grounding,
                house_grounding=house_grounding,
                pen_wire_radius_m=pen_wire_radius_m,
                pen_segment_length_m=pen_segment_length_m,
                include_source=include_source,
                source_magnitude_a=source_magnitude_a,
                seed=seed,
                foundation_mask=foundation_mask,
                foundation_outline=foundation_outline,
            )
        )

    # ------------------------------------------------------------------
    # Cable decomposition (plan helpers)
    # ------------------------------------------------------------------

    def _cable_nodes(
        self, cable: PenCable, junctions: list[PlannedJunction]
    ) -> list[tuple[str, tuple[float, float]]]:
        """Ordered nodes of a cable; new junctions are appended to ``junctions``.

        The first waypoint is the start site; the last one is the end
        site when the cable ends at a cabinet; every other corner and
        every tap point on the cable is a junction
        (``f"{cable}_node_{k}"`` / ``f"{cable}_tap_{n}"``).
        """
        taps_for_cable = sorted(
            {
                tuple(c.tap_point_xy): c.house_idx
                for c in self.connections
                if c.cable_name == cable.name
            }.items(),
            key=lambda kv: (kv[0][0], kv[0][1]),
        )
        tap_lookup = {pos: house_idx for pos, house_idx in taps_for_cable}
        nodes: list[tuple[str, tuple[float, float]]] = []
        for k, wp in enumerate(cable.waypoints):
            wp_t = tuple(wp)
            if k == 0:
                nodes.append((self._resolve_anchor(cable.start_anchor), wp_t))
                continue
            if k == len(cable.waypoints) - 1 and cable.end_anchor is not None:
                nodes.append((self._resolve_anchor(cable.end_anchor), wp_t))
                continue
            node_name = f"{cable.name}_node_{k}"
            junctions.append(PlannedJunction(name=node_name, position_xy=wp_t))
            nodes.append((node_name, wp_t))
        return self._insert_tap_nodes_on_cable(
            cable=cable, nodes=nodes, tap_lookup=tap_lookup, junctions=junctions
        )

    def _resolve_anchor(self, name: str) -> str:
        """Validate that ``name`` is the substation or a cabinet (a plan site)."""
        if name == self.substation_name or any(
            k.name == name for k in self.cabinet_placements
        ):
            return name
        raise KeyError(f"LvNetworkLayout._resolve_anchor: unknown anchor {name!r}.")

    def _insert_tap_nodes_on_cable(
        self,
        *,
        cable: PenCable,
        nodes: list[tuple[str, tuple[float, float]]],
        tap_lookup: dict[tuple[float, float], int],
        junctions: list[PlannedJunction],
    ) -> list[tuple[str, tuple[float, float]]]:
        """Splice tap-point junctions into the corner-node list.

        Walks the cable in segment order and inserts every tap that
        falls inside the segment (sorted along the segment
        direction so adjacent stubs on a long bus segment stay in
        physical order).
        """
        if not tap_lookup:
            return nodes
        taps_remaining = set(tap_lookup.keys())
        out: list[tuple[str, tuple[float, float]]] = []
        for j in range(len(nodes) - 1):
            a0, p0 = nodes[j]
            a1, p1 = nodes[j + 1]
            out.append((a0, p0))
            on_segment: list[tuple[float, float]] = []
            for tap_pos in list(taps_remaining):
                if _approximately_equal(tap_pos, p0) or _approximately_equal(
                    tap_pos, p1
                ):
                    taps_remaining.discard(tap_pos)
                    continue
                if _point_on_axis_segment(tap_pos, p0, p1):
                    on_segment.append(tap_pos)
                    taps_remaining.discard(tap_pos)
            if not on_segment:
                continue
            # Sort along the segment direction so the inserted
            # order matches the physical walk from p0 to p1.
            if math.isclose(p0[1], p1[1], abs_tol=1e-9):
                forward = p1[0] >= p0[0]
                on_segment.sort(key=lambda t: t[0] if forward else -t[0])
            else:
                forward = p1[1] >= p0[1]
                on_segment.sort(key=lambda t: t[1] if forward else -t[1])
            for tap_pos in on_segment:
                tap_name = f"{cable.name}_tap_{len(out)}"
                junctions.append(PlannedJunction(name=tap_name, position_xy=tap_pos))
                out.append((tap_name, tap_pos))
        out.append(nodes[-1])
        return out

    def _tap_anchor_for(
        self,
        connection: BuildingConnection,
        cable_node_map: dict[str, list[tuple[str, tuple[float, float]]]],
    ) -> Optional[str]:
        """Find the cable-node anchor closest to a connection's
        tap point.
        """
        nodes = cable_node_map.get(connection.cable_name, [])
        if not nodes:
            return None
        tap = connection.tap_point_xy
        # Exact match first.
        for anchor, (x, y) in nodes:
            if math.isclose(x, tap[0], abs_tol=1e-6) and math.isclose(
                y, tap[1], abs_tol=1e-6
            ):
                return anchor
        # Fall back to nearest node (Manhattan).
        best_anchor: Optional[str] = None
        best_dist = math.inf
        for anchor, (x, y) in nodes:
            d = abs(x - tap[0]) + abs(y - tap[1])
            if d < best_dist:
                best_dist = d
                best_anchor = anchor
        return best_anchor


class OrtsnetzLayout(LvNetworkLayout):
    """Deprecated alias of :class:`LvNetworkLayout`."""

    def model_post_init(self, context: object, /) -> None:
        warn_renamed("OrtsnetzLayout", "LvNetworkLayout", stacklevel=5)
        super().model_post_init(context)


# ---------------------------------------------------------------------
# Free-floating helpers
# ---------------------------------------------------------------------


def _approximately_equal(
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    tol: float = 1e-6,
) -> bool:
    return math.isclose(a[0], b[0], abs_tol=tol) and math.isclose(
        a[1], b[1], abs_tol=tol
    )


def _point_on_axis_segment(
    point: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
    *,
    tol: float = 1e-6,
) -> bool:
    """Whether ``point`` lies on the axis-aligned segment."""
    sx, sy = seg_start
    ex, ey = seg_end
    if math.isclose(sy, ey, abs_tol=tol):
        return (
            math.isclose(point[1], sy, abs_tol=tol)
            and min(sx, ex) - tol <= point[0] <= max(sx, ex) + tol
        )
    if math.isclose(sx, ex, abs_tol=tol):
        return (
            math.isclose(point[0], sx, abs_tol=tol)
            and min(sy, ey) - tol <= point[1] <= max(sy, ey) + tol
        )
    return False
