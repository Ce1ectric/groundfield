r"""Imperative TN-Ortsnetz layout builder.

Where :class:`~groundfield.generators.tn_network.TnNetworkGenerator`
generates *stochastic* AP1 reference worlds from population-level
parameters (counts, distributions), the
:class:`OrtsnetzLayout` builder of this module composes a *single*
deterministic network in the order an engineer would draw it on a
plan:

1. Ingest building footprints from
   :mod:`groundfield.geo.osm` (or any other :class:`BuildingFootprint`
   producer). Footprints become both house positions *and* hard
   obstacles for the PEN cable router.
2. Place the substation at user-given ``(x, y)`` coordinates.
3. *Optionally* add one or more KVS (cable cabinets) at user-given
   ``(x, y)`` coordinates.
4. *Optionally* add PEN backbone segments substation → KVS or
   KVS → KVS by simply naming the two anchors.
5. Run :meth:`OrtsnetzLayout.add_pen_cable` to lay out a PEN cable
   from any anchor to a user-supplied ``(x, y)`` endpoint. The
   route is computed by :func:`manhattan_routing.route_manhattan`
   -- 4-connected A\* on a grid of cells :attr:`min_segment_length_m`
   wide -- and the cable never crosses a foundation polygon. The
   user controls clearance, grid size, and may add as many cables
   as needed.
6. :meth:`OrtsnetzLayout.connect_buildings` attaches every (still
   unconnected) house to the closest PEN cable via a short
   axis-aligned stub (one or two segments). When only one cable
   is in the layout, every house ends up on that cable. Stubs
   that would cross *another* building are routed around it via
   the same Manhattan A\* engine.

The builder is independent of the engine / solver layer; what it
produces is a *layout snapshot* (Pydantic-serialisable) plus two
exits:

* :meth:`OrtsnetzLayout.to_world` materialises the layout into a
  full :class:`groundfield.World` (substation grounding, KVS
  grounding, per-house foundation electrodes, PEN conductors and
  -- optionally -- a current source at the substation cluster).
* :meth:`OrtsnetzLayout.plot` renders the layout with matplotlib
  (foundation polygons, anchor markers, PEN trunk + stubs)
  without any solver work.

This module is the recommended entry point when the network is
known piece by piece (a real OSM extract plus a hand-placed
substation) and the AP1 statistical sweep approach
(:class:`TnNetworkGenerator`) is too rigid.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from groundfield.api import create_conductor, create_source, create_world
from groundfield.generators.electrode_specs import (
    FoundationElectrodeSpec,
    RingElectrodeSpec,
    RodElectrodeSpec,
    rod_circle,
)
from groundfield.generators.grounding import GroundingSystemSpec
from groundfield.generators.manhattan_routing import (
    ObstacleBox,
    merge_collinear_waypoints,
    route_manhattan,
    segment_intersects_box,
)
from groundfield.geo.footprint import BuildingFootprint
from groundfield.geo.projection import Projector
from groundfield.soil.models import HomogeneousSoil, SoilModel
from groundfield.world import World

if TYPE_CHECKING:  # pragma: no cover - typing only
    import matplotlib.axes as _mpl_axes

__all__ = [
    "KvsPlacement",
    "PenCable",
    "BuildingConnection",
    "OrtsnetzLayout",
]


# ---------------------------------------------------------------------
# Default grounding factories (kept in sync with the AP1 TnNetwork
# generator defaults so the two pipelines compose with each other).
# ---------------------------------------------------------------------


def _default_substation_grounding() -> GroundingSystemSpec:
    """Ring (4 m) + 4 rods on a 2 m inner circle (`Tiefenerder`)."""
    return GroundingSystemSpec(
        electrodes=[
            RingElectrodeSpec(radius_m=4.0, depth_m=0.6),
            *rod_circle(n=4, radius_m=2.0, length_m=2.5),
        ],
    )


def _default_kvs_grounding() -> GroundingSystemSpec:
    """Single 1.5 m driven rod."""
    return GroundingSystemSpec(
        electrodes=[RodElectrodeSpec(length_m=1.5, depth_m=0.0)],
    )


def _default_house_grounding() -> GroundingSystemSpec:
    """10 m foundation electrode with a 2x2 mesh (`Fundamenterder`)."""
    return GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                size_m=10.0, depth_m=0.8, n_x=2, n_y=2,
            ),
        ],
    )


# ---------------------------------------------------------------------
# Snapshot data classes
# ---------------------------------------------------------------------


class KvsPlacement(BaseModel):
    """One user-placed cable cabinet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    position_xy: tuple[float, float]


class PenCable(BaseModel):
    """One PEN cable as a polyline of axis-aligned waypoints.

    Attributes
    ----------
    name
        Unique cable id used for diagnostics, ``to_world`` naming
        and plot legends.
    start_anchor
        Name of the anchor (substation or KVS) the cable roots at.
    waypoints
        Ordered list of $(x, y)$ corners. The first entry equals
        the start anchor's position and consecutive entries differ
        in exactly one coordinate (Manhattan).
    end_anchor
        Optional name of the anchor (KVS) the cable terminates at.
        ``None`` when the cable runs to free coordinates (a
        :emphasis:`stub end` used as a tap for nearby houses but
        not for further trunking).
    grid_size_m
        The grid size used when this cable was routed -- kept on
        the model so :meth:`OrtsnetzLayout.connect_buildings` can
        align stubs with the trunk.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    start_anchor: str
    waypoints: list[tuple[float, float]]
    end_anchor: Optional[str] = None
    grid_size_m: float = 10.0

    def segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return list(zip(self.waypoints[:-1], self.waypoints[1:]))

    def total_length_m(self) -> float:
        return sum(
            abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
            for p0, p1 in self.segments()
        )


class BuildingConnection(BaseModel):
    """Stub from one house's foundation centre to its serving cable.

    Attributes
    ----------
    house_idx
        Index into :attr:`OrtsnetzLayout.footprints`.
    cable_name
        Name of the :class:`PenCable` the stub taps into.
    stub_waypoints
        Polyline from the house centroid to the tap point on the
        cable. ``[centroid, ..., tap_point]``. The interior
        waypoints (if any) are the Manhattan corners.
    tap_point_xy
        The actual tap point on the cable polyline.
    """

    model_config = ConfigDict(extra="forbid")

    house_idx: int
    cable_name: str
    stub_waypoints: list[tuple[float, float]]
    tap_point_xy: tuple[float, float]

    def total_length_m(self) -> float:
        return sum(
            abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
            for p0, p1 in zip(
                self.stub_waypoints[:-1], self.stub_waypoints[1:],
            )
        )


# ---------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------


class OrtsnetzLayout(BaseModel):
    r"""Imperative TN-Ortsnetz layout builder.

    The builder is constructed empty (no buildings, no substation,
    no cables) and filled up by calling its methods in the order
    a planner would naturally take. Every state-changing method
    returns the new object (or its key) so the calls can be
    chained or stored:

    >>> layout = OrtsnetzLayout.from_footprints(footprints,
    ...                                         substation_xy=(0, 0))
    >>> kvs_1 = layout.add_kvs((120, 0))
    >>> cable = layout.add_pen_cable(start="substation",
    ...                              end_xy=(250, 0))
    >>> layout.connect_buildings()
    >>> ax = layout.plot()

    Parameters
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
    kvs_placements
        Optional pre-populated KVS list. Most callers leave this
        empty and use :meth:`add_kvs`.
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
    kvs_placements: list[KvsPlacement] = Field(default_factory=list)
    pen_cables: list[PenCable] = Field(default_factory=list)
    connections: list[BuildingConnection] = Field(default_factory=list)
    obstacle_clearance_m: float = Field(default=1.0, ge=0.0)
    frame_origin_lat_lon: Optional[tuple[float, float]] = Field(
        default=None,
        description=(
            "WGS84 origin (lat, lon) of the local ENU projection frame "
            "that the footprints and the substation are expressed in. "
            "Set automatically by :meth:`from_osm`; pass it explicitly "
            "when constructing the layout from already-projected data "
            "if you want to add KVS or PEN endpoints via lat/lon later."
        ),
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_footprints(
        cls,
        footprints: list[BuildingFootprint],
        *,
        substation_xy: tuple[float, float],
        substation_name: str = "substation",
        obstacle_clearance_m: float = 1.0,
        frame_origin_lat_lon: Optional[tuple[float, float]] = None,
    ) -> "OrtsnetzLayout":
        """Build an empty layout from a list of footprints.

        ``frame_origin_lat_lon`` is optional; pass it when the
        footprints were projected by an external pipeline and you
        still want to add KVS / PEN endpoints by lat / lon later.
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
        kvs_lat_lons: Optional[list[tuple[float, float]]] = None,
        kvs_xys: Optional[list[tuple[float, float]]] = None,
        substation_name: str = "substation",
        obstacle_clearance_m: float = 1.0,
        min_area_m2: float = 16.0,
        cache_dir=None,
        force_refresh: bool = False,
        endpoint: Optional[str] = None,
        timeout_s: Optional[int] = None,
        max_retries: int = 1,
    ) -> "OrtsnetzLayout":
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
        kvs_lat_lons
            Optional list of KVS locations in WGS84 ``(lat, lon)``.
            Each entry is projected and appended to
            :attr:`kvs_placements`. Names are auto-generated.
        kvs_xys
            Optional list of KVS locations in already-projected
            metres. Mutually exclusive with ``kvs_lat_lons``.
        substation_name
            Anchor id for the substation. Default ``"substation"``.
        obstacle_clearance_m, min_area_m2
            Forwarded to the layout / OSM query respectively.
        cache_dir, force_refresh, endpoint, timeout_s, max_retries
            Forwarded to :func:`groundfield.geo.osm.query_and_project`.

        Returns
        -------
        OrtsnetzLayout
            Layout with footprints loaded, projector seeded, and
            substation / KVS pre-placed. Routing and connections
            are still up to the caller.

        Examples
        --------
        Build the layout for Mulmke (Heudeber / Nordharz) in a
        300 m radius around the village centre, with the
        substation at a chosen lat / lon and two KVS along the
        main road:

        >>> layout = OrtsnetzLayout.from_osm(
        ...     center_lat_deg=51.9136, center_lon_deg=10.8432,
        ...     radius_m=300.0,
        ...     substation_lat_lon=(51.9138, 10.8418),
        ...     kvs_lat_lons=[
        ...         (51.9133, 10.8445),
        ...         (51.9140, 10.8460),
        ...     ],
        ... )
        """
        from groundfield.geo.osm import query_and_project

        if substation_lat_lon is not None and substation_xy is not None:
            raise ValueError(
                "OrtsnetzLayout.from_osm: substation_lat_lon and "
                "substation_xy are mutually exclusive."
            )
        if kvs_lat_lons is not None and kvs_xys is not None:
            raise ValueError(
                "OrtsnetzLayout.from_osm: kvs_lat_lons and kvs_xys "
                "are mutually exclusive."
            )

        query_kwargs: dict = {
            "min_area_m2": min_area_m2,
            "force_refresh": force_refresh,
            "max_retries": max_retries,
        }
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

        if kvs_lat_lons:
            for lat, lon in kvs_lat_lons:
                layout.add_kvs(position_lat_lon=(lat, lon))
        elif kvs_xys:
            for xy in kvs_xys:
                layout.add_kvs(position_xy=xy)

        return layout

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

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
        self, lat_deg: float, lon_deg: float,
    ) -> tuple[float, float]:
        """Project a WGS84 ``(lat, lon)`` to the layout's local frame.

        Raises :class:`RuntimeError` when no
        :attr:`frame_origin_lat_lon` has been set (i.e. the layout
        was not built via :meth:`from_osm`).
        """
        projector = self.projector
        if projector is None:
            raise RuntimeError(
                "OrtsnetzLayout.lat_lon_to_xy: layout has no "
                "frame_origin_lat_lon. Build via ``from_osm`` or "
                "pass ``frame_origin_lat_lon`` to ``from_footprints`` "
                "to enable lat / lon based placement."
            )
        return projector.to_xy_m(lat_deg, lon_deg)

    def xy_to_lat_lon(
        self, x_m: float, y_m: float,
    ) -> tuple[float, float]:
        """Inverse projection: local frame ``(x, y)`` → WGS84.

        Mirrors :meth:`lat_lon_to_xy`.
        """
        projector = self.projector
        if projector is None:
            raise RuntimeError(
                "OrtsnetzLayout.xy_to_lat_lon: layout has no "
                "frame_origin_lat_lon. Build via ``from_osm`` first."
            )
        return projector.to_lat_lon(x_m, y_m)

    # ------------------------------------------------------------------
    # KVS placement
    # ------------------------------------------------------------------

    def add_kvs(
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

        Returns the assigned KVS name (auto-generated when ``name``
        is ``None``). The returned name is what callers pass to
        :meth:`add_pen_cable` as ``start=...``.
        """
        if (position_xy is None) == (position_lat_lon is None):
            raise ValueError(
                "OrtsnetzLayout.add_kvs: exactly one of "
                "``position_xy`` and ``position_lat_lon`` must be set."
            )
        if position_lat_lon is not None:
            position_xy = self.lat_lon_to_xy(*position_lat_lon)
        if name is None:
            name = f"kvs_{len(self.kvs_placements)}"
        if any(k.name == name for k in self.kvs_placements):
            raise ValueError(
                f"OrtsnetzLayout.add_kvs: name {name!r} is already taken."
            )
        if name == self.substation_name:
            raise ValueError(
                f"OrtsnetzLayout.add_kvs: name {name!r} clashes with "
                "the substation name."
            )
        assert position_xy is not None
        self.kvs_placements.append(
            KvsPlacement(name=name, position_xy=tuple(position_xy)),
        )
        return name

    # ------------------------------------------------------------------
    # PEN cabling
    # ------------------------------------------------------------------

    def add_pen_cable(
        self,
        *,
        start: str,
        end_xy: Optional[tuple[float, float]] = None,
        end_lat_lon: Optional[tuple[float, float]] = None,
        end: Optional[str] = None,
        name: Optional[str] = None,
        min_segment_length_m: float = 10.0,
        clearance_m: Optional[float] = None,
    ) -> PenCable:
        """Add a PEN cable routed Manhattan-style.

        Exactly one of ``end``, ``end_xy`` or ``end_lat_lon`` must
        be set. When ``end`` is the name of another anchor (KVS)
        the cable terminates there and the receiving cabinet is
        recorded as :attr:`PenCable.end_anchor`. When ``end_xy`` or
        ``end_lat_lon`` is set, the cable's last waypoint is the
        free coordinate and houses near it can still tap into the
        trunk. ``end_lat_lon`` is projected through the layout's
        :attr:`frame_origin_lat_lon` (raises if no frame origin is
        set).

        Parameters
        ----------
        start
            Name of an anchor present in this layout
            (``substation_name`` or the name of a KVS).
        end_xy
            Free $(x, y)$ endpoint coordinate in metres.
        end
            Name of the destination anchor. Mutually exclusive
            with ``end_xy``.
        name
            Optional cable id. Auto-generated when ``None``.
        min_segment_length_m
            Grid size of the Manhattan A\\* router. Defaults to
            10 m -- a reasonable street-segment length and an
            order of magnitude longer than a typical foundation
            footprint.
        clearance_m
            Per-call override for :attr:`obstacle_clearance_m`.

        Returns
        -------
        PenCable
            The freshly-routed cable. Already registered in
            :attr:`pen_cables`.
        """
        # Exactly one of {end, end_xy, end_lat_lon} must be provided.
        provided = sum(
            arg is not None for arg in (end, end_xy, end_lat_lon)
        )
        if provided != 1:
            raise ValueError(
                "OrtsnetzLayout.add_pen_cable: exactly one of "
                "``end``, ``end_xy`` or ``end_lat_lon`` must be set."
            )
        if name is None:
            name = f"pen_{len(self.pen_cables)}"
        if any(c.name == name for c in self.pen_cables):
            raise ValueError(
                f"OrtsnetzLayout.add_pen_cable: name {name!r} is taken."
            )

        start_xy = self._anchor_position(start)
        if end is not None:
            end_xy_resolved = self._anchor_position(end)
        elif end_lat_lon is not None:
            end_xy_resolved = self.lat_lon_to_xy(*end_lat_lon)
        else:
            assert end_xy is not None  # narrowing for type checkers
            end_xy_resolved = tuple(end_xy)

        clearance = (
            self.obstacle_clearance_m if clearance_m is None else clearance_m
        )
        obstacles = self._obstacle_boxes()
        waypoints = route_manhattan(
            start=start_xy,
            end=end_xy_resolved,
            obstacles=obstacles,
            grid_size=min_segment_length_m,
            clearance_m=clearance,
        )
        cable = PenCable(
            name=name,
            start_anchor=start,
            waypoints=waypoints,
            end_anchor=end,
            grid_size_m=min_segment_length_m,
        )
        self.pen_cables.append(cable)
        return cable

    # ------------------------------------------------------------------
    # Building → cable connection
    # ------------------------------------------------------------------

    def connect_buildings(
        self,
        *,
        only_unconnected: bool = True,
        clearance_m: Optional[float] = None,
    ) -> list[BuildingConnection]:
        """Attach every house to the closest PEN cable.

        For each footprint not yet listed in :attr:`connections`
        (when ``only_unconnected=True``, the default), this method:

        1. Computes the Manhattan-shortest tap point on every
           cable.
        2. Picks the cable whose tap point is closest to the
           house *centroid* and whose stub does not cross another
           foundation (after the same ``clearance_m`` inflation
           used for cable routing).
        3. Builds the stub polyline (at most two segments) and
           records a :class:`BuildingConnection`.

        Houses that cannot be connected to *any* cable (e.g.
        surrounded by foundations on all sides) are reported via
        a :class:`UserWarning` and skipped; they remain
        unconnected and can be revisited after the user adds
        another cable.
        """
        import warnings

        if not self.pen_cables:
            raise RuntimeError(
                "OrtsnetzLayout.connect_buildings: no PEN cables "
                "in the layout. Call ``add_pen_cable`` first."
            )
        clearance = (
            self.obstacle_clearance_m if clearance_m is None else clearance_m
        )

        connected_idx = (
            {c.house_idx for c in self.connections} if only_unconnected
            else set()
        )

        new_connections: list[BuildingConnection] = []
        for i, fp in enumerate(self.footprints):
            if i in connected_idx:
                continue
            centroid = fp.centroid_xy_m()
            # Treat every *other* footprint as a stub obstacle.
            other_obstacles = [
                self._footprint_to_box(self.footprints[j], j).inflated(clearance)
                for j in range(len(self.footprints)) if j != i
            ]
            best: Optional[BuildingConnection] = None
            best_length = math.inf
            for cable in self.pen_cables:
                tap, stub = self._stub_to_cable(
                    centroid, cable, other_obstacles,
                )
                if stub is None:
                    continue
                length = sum(
                    abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
                    for p0, p1 in zip(stub[:-1], stub[1:])
                )
                if length < best_length:
                    best_length = length
                    best = BuildingConnection(
                        house_idx=i,
                        cable_name=cable.name,
                        stub_waypoints=stub,
                        tap_point_xy=tap,
                    )
            if best is None:
                warnings.warn(
                    f"OrtsnetzLayout.connect_buildings: house {i} "
                    f"(centroid {centroid}) cannot be tapped to any "
                    "existing cable without crossing another "
                    "footprint. Add another PEN cable or reduce "
                    "``clearance_m``.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            new_connections.append(best)
        self.connections.extend(new_connections)
        return new_connections

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def to_world(
        self,
        *,
        name: str = "ortsnetz_layout",
        soil: Optional[SoilModel] = None,
        substation_grounding: Optional[GroundingSystemSpec] = None,
        kvs_grounding: Optional[GroundingSystemSpec] = None,
        house_grounding: Optional[GroundingSystemSpec] = None,
        pen_wire_radius_m: float = 0.005,
        pen_segment_length_m: Optional[float] = 5.0,
        include_source: bool = True,
        source_magnitude_A: float = 1.0,
        seed: int = 0,
    ) -> World:
        """Materialise the layout into a :class:`groundfield.World`.

        Each :class:`PenCable` becomes a chain of straight-line
        :class:`Conductor` segments between *PEN junction*
        electrodes (tiny rods at every corner and tap point) so
        the resulting cable physically follows the Manhattan
        polyline. Each :class:`BuildingConnection` becomes one
        additional conductor from the house foundation anchor to
        the nearest PEN junction on the serving cable.
        """
        rng = np.random.default_rng(seed)
        world = create_world(
            name=name,
            soil=soil or HomogeneousSoil(resistivity=100.0),
        )

        # 1. Substation grounding.
        substation_spec = substation_grounding or _default_substation_grounding()
        substation_anchor = substation_spec.build_at(
            world,
            site_xy=self.substation_xy,
            name_prefix=self.substation_name,
            rng=rng,
        )
        if substation_anchor is None:
            raise RuntimeError(
                "OrtsnetzLayout.to_world: substation grounding "
                "produced zero electrodes."
            )

        # 2. KVS groundings.
        kvs_spec = kvs_grounding or _default_kvs_grounding()
        kvs_anchors: dict[str, str] = {}
        for k in self.kvs_placements:
            anchor = kvs_spec.build_at(
                world,
                site_xy=k.position_xy,
                name_prefix=k.name,
                rng=rng,
            )
            if anchor is None:
                raise RuntimeError(
                    "OrtsnetzLayout.to_world: KVS "
                    f"{k.name!r} grounding produced zero electrodes."
                )
            kvs_anchors[k.name] = anchor

        # 3. House groundings (foundations).
        house_spec = house_grounding or _default_house_grounding()
        house_anchors: dict[int, str] = {}
        for i, fp in enumerate(self.footprints):
            centroid = fp.centroid_xy_m()
            anchor = house_spec.build_at(
                world,
                site_xy=centroid,
                name_prefix=f"house_{i:03d}",
                rng=rng,
            )
            if anchor is not None:
                house_anchors[i] = anchor

        # 4. PEN cabling -- one junction electrode per waypoint and
        #    tap point, conductors between consecutive junctions.
        common_pen_kwargs: dict = dict(
            conductor_type="pen",
            wire_radius=pen_wire_radius_m,
            cross_section="from_radius",
            coupling_to_soil="isolated",
        )
        if pen_segment_length_m is not None:
            common_pen_kwargs["discretize_segment_length"] = pen_segment_length_m

        # Build a lookup from cable.name -> ordered list of
        # ``(anchor_name, (x, y))`` along the polyline. Endpoints
        # reuse existing anchors (substation / KVS); internal
        # waypoints and tap points get fresh "pen_node" rods.
        cable_node_map: dict[str, list[tuple[str, tuple[float, float]]]] = {}
        for cable in self.pen_cables:
            nodes: list[tuple[str, tuple[float, float]]] = []
            taps_for_cable = sorted(
                {
                    tuple(c.tap_point_xy): c.house_idx
                    for c in self.connections
                    if c.cable_name == cable.name
                }.items(),
                key=lambda kv: (kv[0][0], kv[0][1]),
            )
            tap_lookup = {pos: house_idx for pos, house_idx in taps_for_cable}

            for k, wp in enumerate(cable.waypoints):
                wp_t = tuple(wp)
                if k == 0:
                    nodes.append((self._resolve_anchor(
                        cable.start_anchor, substation_anchor, kvs_anchors,
                    ), wp_t))
                    continue
                if k == len(cable.waypoints) - 1 and cable.end_anchor is not None:
                    nodes.append((self._resolve_anchor(
                        cable.end_anchor, substation_anchor, kvs_anchors,
                    ), wp_t))
                    continue
                # Internal corner -- create a tiny rod electrode.
                node_name = f"{cable.name}_node_{k}"
                _create_pen_node(world, node_name, wp_t)
                nodes.append((node_name, wp_t))
            # Insert tap nodes between corner nodes where the tap
            # falls on the cable segment.
            nodes = self._insert_tap_nodes_on_cable(
                cable=cable,
                nodes=nodes,
                tap_lookup=tap_lookup,
                world=world,
            )
            cable_node_map[cable.name] = nodes
            # Trunk conductors between consecutive nodes.
            for j, ((a0, _p0), (a1, _p1)) in enumerate(
                zip(nodes[:-1], nodes[1:])
            ):
                if a0 == a1:
                    continue
                create_conductor(
                    world,
                    name=f"{cable.name}_seg_{j}",
                    start=a0,
                    end=a1,
                    **common_pen_kwargs,
                )

        # 5. Service drops -- foundation anchor → tap node on cable.
        for c in self.connections:
            if c.house_idx not in house_anchors:
                continue
            tap_anchor = self._tap_anchor_for(c, cable_node_map)
            if tap_anchor is None:
                continue
            create_conductor(
                world,
                name=f"service_{c.house_idx:03d}",
                start=tap_anchor,
                end=house_anchors[c.house_idx],
                **common_pen_kwargs,
            )

        # 6. Source at the substation.
        if include_source:
            create_source(
                world,
                attached_to=substation_anchor,
                magnitude=source_magnitude_A,
            )

        return world

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        ax: Optional["_mpl_axes.Axes"] = None,
        show_unconnected: bool = True,
        annotate_houses: bool = False,
        figsize: tuple[float, float] = (12.0, 8.0),
    ) -> "_mpl_axes.Axes":
        """Render the layout with matplotlib.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the layout was drawn on. The figure can be
            retrieved via ``ax.figure``.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        connected_idx = {c.house_idx for c in self.connections}

        # Building footprints.
        for i, fp in enumerate(self.footprints):
            xs = [p[0] for p in fp.polygon_xy_m] + [fp.polygon_xy_m[0][0]]
            ys = [p[1] for p in fp.polygon_xy_m] + [fp.polygon_xy_m[0][1]]
            colour = "0.92" if i in connected_idx else "#fde0dc"
            ax.fill(xs, ys, color=colour, edgecolor="0.4",
                    linewidth=0.6, zorder=1)
            if annotate_houses:
                cx, cy = fp.centroid_xy_m()
                ax.text(cx, cy, str(i), ha="center", va="center",
                        fontsize=7, color="0.3", zorder=2)

        # PEN cables.
        cable_colours = {}
        palette = plt.get_cmap("tab10").colors
        for k, cable in enumerate(self.pen_cables):
            colour = palette[k % len(palette)]
            cable_colours[cable.name] = colour
            xs = [p[0] for p in cable.waypoints]
            ys = [p[1] for p in cable.waypoints]
            ax.plot(xs, ys, color=colour, lw=2.2, zorder=4,
                    label=f"PEN {cable.name}")
            # Mark every corner.
            for x, y in cable.waypoints:
                ax.plot(x, y, marker="o", color=colour, markersize=3,
                        zorder=5)

        # Service drops.
        drew_stub_label = False
        for c in self.connections:
            colour = cable_colours.get(c.cable_name, "saddlebrown")
            xs = [p[0] for p in c.stub_waypoints]
            ys = [p[1] for p in c.stub_waypoints]
            ax.plot(xs, ys, color=colour, lw=1.0, ls="--", zorder=3,
                    label=("PEN service drop" if not drew_stub_label
                           else None))
            drew_stub_label = True
            ax.plot(c.tap_point_xy[0], c.tap_point_xy[1],
                    marker="x", color=colour, markersize=6, zorder=6)

        # Substation marker.
        ax.plot(*self.substation_xy, marker="D",
                color="tab:orange", markersize=14, zorder=7,
                label="Substation")

        # KVS markers.
        for k in self.kvs_placements:
            ax.plot(*k.position_xy, marker="s",
                    color="tab:blue", markersize=11, zorder=7)
        if self.kvs_placements:
            ax.plot([], [], marker="s", color="tab:blue",
                    markersize=11, linestyle="None", label="KVS")
        if show_unconnected and any(
            i not in connected_idx for i in range(len(self.footprints))
        ):
            ax.plot([], [], marker="s", color="#fde0dc",
                    markersize=11, linestyle="None",
                    label="Unconnected house")

        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_title("TN-Ortsnetz layout (Manhattan PEN routing)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _anchor_position(self, name: str) -> tuple[float, float]:
        if name == self.substation_name:
            return tuple(self.substation_xy)
        for k in self.kvs_placements:
            if k.name == name:
                return tuple(k.position_xy)
        raise KeyError(
            f"OrtsnetzLayout: anchor {name!r} is unknown. "
            f"Known anchors: [{self.substation_name!r}] + "
            f"{[k.name for k in self.kvs_placements]!r}."
        )

    def _footprint_to_box(
        self, fp: BuildingFootprint, idx: int,
    ) -> ObstacleBox:
        (cx, cy), (dx, dy) = fp.axis_aligned_bounding_rectangle()
        return ObstacleBox(
            x_min=cx - dx / 2.0,
            y_min=cy - dy / 2.0,
            x_max=cx + dx / 2.0,
            y_max=cy + dy / 2.0,
            name=f"house_{idx}",
        )

    def _obstacle_boxes(self) -> list[ObstacleBox]:
        return [
            self._footprint_to_box(fp, i)
            for i, fp in enumerate(self.footprints)
        ]

    def _stub_to_cable(
        self,
        centroid: tuple[float, float],
        cable: PenCable,
        obstacles: list[ObstacleBox],
    ) -> tuple[
        Optional[tuple[float, float]],
        Optional[list[tuple[float, float]]],
    ]:
        """Find the Manhattan-closest tap point on ``cable``.

        Returns ``(tap_point, stub_waypoints)`` or ``(None, None)``
        when no obstacle-free stub exists.
        """
        best: Optional[
            tuple[tuple[float, float], list[tuple[float, float]], float]
        ] = None
        for seg_start, seg_end in cable.segments():
            tap = _project_onto_axis_segment(centroid, seg_start, seg_end)
            stub = _manhattan_stub(centroid, tap, obstacles)
            if stub is None:
                continue
            length = sum(
                abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
                for p0, p1 in zip(stub[:-1], stub[1:])
            )
            if best is None or length < best[2]:
                best = (tap, stub, length)
        if best is None:
            return (None, None)
        return (best[0], best[1])

    def _resolve_anchor(
        self,
        name: str,
        substation_anchor: str,
        kvs_anchors: dict[str, str],
    ) -> str:
        if name == self.substation_name:
            return substation_anchor
        if name in kvs_anchors:
            return kvs_anchors[name]
        raise KeyError(
            f"OrtsnetzLayout._resolve_anchor: unknown anchor {name!r}."
        )

    def _insert_tap_nodes_on_cable(
        self,
        *,
        cable: PenCable,
        nodes: list[tuple[str, tuple[float, float]]],
        tap_lookup: dict[tuple[float, float], int],
        world: World,
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
                tap_name = (
                    f"{cable.name}_tap_{len(out)}"
                )
                _create_pen_node(world, tap_name, tap_pos)
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
            if math.isclose(x, tap[0], abs_tol=1e-6) and \
                    math.isclose(y, tap[1], abs_tol=1e-6):
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


# ---------------------------------------------------------------------
# Free-floating helpers
# ---------------------------------------------------------------------


def _project_onto_axis_segment(
    point: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> tuple[float, float]:
    """Closest point on an axis-aligned segment to ``point``."""
    sx, sy = seg_start
    ex, ey = seg_end
    if math.isclose(sy, ey, abs_tol=1e-9):
        # Horizontal segment.
        x = max(min(sx, ex), min(point[0], max(sx, ex)))
        return (x, sy)
    if math.isclose(sx, ex, abs_tol=1e-9):
        # Vertical segment.
        y = max(min(sy, ey), min(point[1], max(sy, ey)))
        return (sx, y)
    raise ValueError(
        f"_project_onto_axis_segment: segment {(seg_start, seg_end)} "
        "is not axis-aligned."
    )


def _manhattan_stub(
    centroid: tuple[float, float],
    tap: tuple[float, float],
    obstacles: list[ObstacleBox],
) -> Optional[list[tuple[float, float]]]:
    """Build a Manhattan polyline from ``centroid`` to ``tap``.

    Returns ``None`` when no axis-aligned 1- or 2-segment polyline
    can connect ``centroid`` to ``tap`` without crossing any
    obstacle. Prefers a direct (single-segment) connection when
    one of the coordinates already matches.
    """
    cx, cy = centroid
    tx, ty = tap
    if math.isclose(cx, tx) or math.isclose(cy, ty):
        if not any(
            segment_intersects_box(centroid, tap, o)
            for o in obstacles
        ):
            return [centroid, tap]
        # Single straight stub is blocked -- fall through to the
        # corner attempts.
    # Two corner options.
    for corner in ((tx, cy), (cx, ty)):
        seg1 = (centroid, corner)
        seg2 = (corner, tap)
        # Skip degenerate corners (collinear).
        if math.isclose(corner[0], centroid[0]) and math.isclose(
            corner[1], centroid[1]
        ):
            continue
        if math.isclose(corner[0], tap[0]) and math.isclose(
            corner[1], tap[1]
        ):
            continue
        if any(
            segment_intersects_box(seg1[0], seg1[1], o)
            or segment_intersects_box(seg2[0], seg2[1], o)
            for o in obstacles
        ):
            continue
        return merge_collinear_waypoints([centroid, corner, tap])
    return None


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


def _create_pen_node(
    world: World,
    name: str,
    position_xy: tuple[float, float],
    *,
    depth_m: float = 0.6,
) -> str:
    """Insert a tiny rod electrode acting as a PEN cable junction.

    The rod is 5 cm long and sits at typical cable burial depth.
    Its self-resistance to earth is several orders of magnitude
    above the substation grounding so the junction is
    electrically transparent to the soil.
    """
    from groundfield.api import create_electrode

    create_electrode(
        world,
        "rod",
        name=name,
        position=(position_xy[0], position_xy[1], depth_m),
        length=0.05,
    )
    return name
