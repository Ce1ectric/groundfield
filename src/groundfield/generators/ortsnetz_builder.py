r"""Imperative TN-Ortsnetz layout builder.

Where :class:`~groundfield.generators.tn_network.TnNetworkGenerator`
generates *stochastic* reference worlds from population-level
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
substation) and the statistical sweep approach of
:class:`TnNetworkGenerator` is too rigid.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, NamedTuple, Optional

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
    "AuxiliaryElectrodePlacement",
    "VoltageProbePlacement",
    "PenCable",
    "BuildingConnection",
    "OrtsnetzLayout",
    "triangle_rod_grounding",
]


# ---------------------------------------------------------------------
# Default grounding factories (kept in sync with the TnNetwork
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


def triangle_rod_grounding(
    rod_length_m: float = 0.5,
    triangle_side_m: float = 0.5,
) -> GroundingSystemSpec:
    r"""Default Hilfserder geometry: three parallel vertical rods in
    an equilateral triangle.

    Real-world fall-of-potential test sets use a *bundle* of rods
    rather than a single Tiefenerder so the auxiliary's own
    spreading resistance stays well below the substation's. Three
    0.5 m rods at the corners of a 0.5 m equilateral triangle give
    roughly a third of the single-rod impedance.
    """
    half = triangle_side_m / 2.0
    # Equilateral triangle, centroid at the origin: corners are at
    # angles 90 deg, 210 deg, 330 deg from the centre at a radius
    # ``triangle_side_m / sqrt(3)``.
    r = triangle_side_m / math.sqrt(3.0)
    offsets = [
        (r * math.cos(math.radians(a)),
         r * math.sin(math.radians(a)))
        for a in (90.0, 210.0, 330.0)
    ]
    return GroundingSystemSpec(
        electrodes=[
            RodElectrodeSpec(
                length_m=rod_length_m, depth_m=0.0,
                offset_xy_m=off,
            )
            for off in offsets
        ],
    )


def _default_aux_grounding() -> GroundingSystemSpec:
    """Project default for the Hilfserder: 3 × 0.5 m rods in a
    0.5 m equilateral triangle (parallel-bonded)."""
    return triangle_rod_grounding(rod_length_m=0.5, triangle_side_m=0.5)


class AuxiliaryElectrodePlacement(BaseModel):
    r"""Auxiliary current electrode (Hilfserder) for a fall-of-
    potential measurement of the substation grounding.

    The auxiliary electrode closes the test-current loop. It is
    materialised in :meth:`OrtsnetzLayout.to_world` via
    :meth:`GroundingSystemSpec.build_at`, giving the user full
    control over its geometry (single rod, ring, mesh, foundation,
    or any combination). The default is a *bundle of three 0.5 m
    rods in a 0.5 m equilateral triangle, parallel-bonded* —
    matching real-world Hilfserder practice and yielding a
    spreading resistance roughly a third of a single rod.

    During :meth:`OrtsnetzLayout.to_world` the layout adds a
    second :class:`CurrentSource` with ``phase_deg=180`` at the
    anchor electrode so the engine's potential field develops the
    expected dipole pattern (positive trumpet at the substation,
    negative trumpet at the Hilfserder).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "aux"
    position_xy: tuple[float, float]
    grounding: GroundingSystemSpec = Field(
        default_factory=_default_aux_grounding,
        description=(
            "Per-Hilfserder grounding system spec. Default is a "
            "3 × 0.5 m rod triangle (0.5 m side); override for "
            "single-rod or other custom geometries."
        ),
    )


class VoltageProbePlacement(BaseModel):
    r"""Voltage probe (Spannungssonde) used during the simulated
    fall-of-potential measurement.

    Unlike the :class:`AuxiliaryElectrodePlacement`, the probe is
    modelled as a pure *sampling point* in 3-D space: an ideal
    voltmeter has infinite input impedance, so a physical probe
    rod would carry zero current and therefore must not perturb
    the engine's potential field. In :meth:`OrtsnetzLayout.to_world`
    no rod electrode is created for the probe; in
    :meth:`OrtsnetzLayout.measured_grounding_impedance` the field
    is sampled at :attr:`position_xy` via
    :meth:`FieldResult.potential`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "probe"
    position_xy: tuple[float, float]
    depth_m: float = Field(
        default=0.0,
        description=(
            "Depth in m at which the field is sampled. 0 = ground "
            "surface (typical: a probe rod tip just below grass)."
        ),
    )


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


class LateralCoverageResult(NamedTuple):
    """Outcome of :meth:`OrtsnetzLayout.connect_all_buildings`.

    Attributes
    ----------
    laterals
        Names of the PEN lateral cables that were added to reach
        otherwise-unconnected houses, in the order they were created.
    islands
        Indices of footprints that still could not be connected -- houses
        enclosed by other footprints on every side, for which no
        obstacle-free service stub exists even after adding a lateral. An
        empty list means full coverage.
    connections
        Snapshot of :attr:`OrtsnetzLayout.connections` after the pass (the
        tap from every connected house to its serving cable).
    """

    laterals: list[str]
    islands: list[int]
    connections: list["BuildingConnection"]


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
    # Foundation-electrode penetration mask
    # ------------------------------------------------------------------

    def foundation_mask(
        self,
        penetration: float,
        *,
        salt: int = 0,
    ) -> list[bool]:
        r"""Deterministic per-house Bernoulli mask for foundation
        electrodes (`Fundamenterder`).

        For each footprint a stable pseudo-random number
        :math:`r_i \in [0, 1)` is derived from the footprint's
        ``osm_id`` (or its index as a fallback) and the user-
        supplied ``salt``. The mask entry is ``True`` iff
        :math:`r_i < p` where :math:`p =` ``penetration``. Two
        properties hold:

        * **Reproducible.** Calling
          ``layout.foundation_mask(0.15)`` always returns the same
          mask for the same set of footprints, regardless of when
          or in which process you call it.
        * **Nested in p.** For a fixed ``salt``, every house with
          a foundation at :math:`p_1` also has one at
          :math:`p_2 > p_1` — increasing the penetration only
          *grows* the foundation-equipped subset. This is the
          natural property for penetration-rate sweep studies.

        Parameters
        ----------
        penetration
            Penetration rate :math:`p \in [0, 1]`. ``0`` returns
            an all-``False`` mask, ``1`` an all-``True`` mask.
        salt
            Optional integer that shifts the mask deterministically
            into a different realisation. Useful for Monte-Carlo
            studies where the user wants several independent runs
            at the *same* penetration rate.

        Returns
        -------
        list of bool
            One entry per footprint, in the layout's declared
            order. Length equals ``len(self.footprints)``.

        Examples
        --------
        >>> mask_15 = layout.foundation_mask(0.15)
        >>> mask_30 = layout.foundation_mask(0.30)
        >>> all(b1 <= b2 for b1, b2 in zip(mask_15, mask_30))   # nested
        True
        >>> layout.foundation_mask(0.15) == layout.foundation_mask(0.15)
        True
        """
        if not 0.0 <= penetration <= 1.0:
            raise ValueError(
                "OrtsnetzLayout.foundation_mask: penetration must be "
                f"in [0, 1], got {penetration}."
            )
        out: list[bool] = []
        for i, fp in enumerate(self.footprints):
            key = fp.osm_id if fp.osm_id is not None else i
            r = _stable_uniform(salt, key)
            out.append(r < penetration)
        return out

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

    # ------------------------------------------------------------------
    # Auxiliary current electrode (Hilfserder)
    # ------------------------------------------------------------------

    def add_auxiliary_electrode(
        self,
        *,
        distance_m: Optional[float] = None,
        direction_deg: float = 0.0,
        position_xy: Optional[tuple[float, float]] = None,
        position_lat_lon: Optional[tuple[float, float]] = None,
        name: str = "aux",
        grounding: Optional[GroundingSystemSpec] = None,
        length_m: Optional[float] = None,
        depth_m: float = 0.0,
    ) -> str:
        r"""Place the auxiliary current electrode (Hilfserder).

        Exactly one of the three position kinds must be set:

        * ``distance_m`` (with optional ``direction_deg``) places
          the electrode at
          :math:`\mathbf{r}_\text{aux} = \mathbf{r}_\text{sub} +
          d\,(\cos\theta,\,\sin\theta)`,
          with :math:`\theta` measured CCW from the local +x axis.
          ``direction_deg = 0`` corresponds to east, ``90`` to
          north. This is the convenient form for a controlled
          sensitivity sweep over distance, mimicking the
          fall-of-potential test geometry.
        * ``position_xy`` sets the electrode at an explicit
          ENU coordinate.
        * ``position_lat_lon`` projects through the layout's
          :attr:`frame_origin_lat_lon` (raises if no frame origin
          is set).

        Parameters
        ----------
        distance_m
            Manhattan-radius from the substation in metres. Must
            be positive.
        direction_deg
            Direction in degrees CCW from +x (east). Default 0.
        position_xy, position_lat_lon
            See above; mutually exclusive with ``distance_m``.
        name
            Anchor id of the auxiliary electrode (defaults to
            ``"aux"``). Must be unique across the layout. The
            materialised electrodes inside :meth:`to_world` are
            named ``f"{name}_{kind}_{i}"`` -- e.g. for the default
            triangle: ``aux_rod_0`` / ``aux_rod_1`` / ``aux_rod_2``.
        grounding
            :class:`GroundingSystemSpec` to use for the
            Hilfserder. ``None`` (default) selects the project
            default (3 × 0.5 m rods in a 0.5 m equilateral
            triangle, parallel-bonded), which matches typical
            field-deployment practice.
        length_m
            Convenience override: when ``grounding`` is ``None`` and
            ``length_m`` is given, the Hilfserder collapses to a
            *single* rod of that length (legacy v0.7 behaviour).
            Ignored when ``grounding`` is set explicitly.
        depth_m
            Depth of the rod head when ``length_m`` is used to
            build a single-rod spec. Default 0.

        Returns
        -------
        str
            The auxiliary anchor's name, identical to ``name``.

        Raises
        ------
        ValueError
            If no positioning kind is given, if more than one is
            given, or if ``name`` clashes with the substation or
            an existing KVS.
        RuntimeError
            From :meth:`lat_lon_to_xy` when
            ``position_lat_lon`` is set but the layout has no
            frame origin.

        Notes
        -----
        The placement *replaces* any previously configured
        auxiliary electrode (there is only one per layout in v1).
        Call this method again with different parameters to
        sweep the auxiliary-electrode distance without rebuilding
        the entire layout.
        """
        provided = sum(
            arg is not None
            for arg in (distance_m, position_xy, position_lat_lon)
        )
        if provided != 1:
            raise ValueError(
                "OrtsnetzLayout.add_auxiliary_electrode: exactly one "
                "of ``distance_m``, ``position_xy`` or "
                "``position_lat_lon`` must be set."
            )

        if distance_m is not None:
            if distance_m <= 0.0:
                raise ValueError(
                    "OrtsnetzLayout.add_auxiliary_electrode: "
                    f"distance_m must be > 0, got {distance_m}."
                )
            theta = math.radians(direction_deg)
            sx, sy = self.substation_xy
            xy = (sx + distance_m * math.cos(theta),
                  sy + distance_m * math.sin(theta))
        elif position_lat_lon is not None:
            xy = self.lat_lon_to_xy(*position_lat_lon)
        else:
            assert position_xy is not None
            xy = tuple(position_xy)

        if name == self.substation_name:
            raise ValueError(
                f"OrtsnetzLayout.add_auxiliary_electrode: name "
                f"{name!r} clashes with the substation name."
            )
        if any(k.name == name for k in self.kvs_placements):
            raise ValueError(
                f"OrtsnetzLayout.add_auxiliary_electrode: name "
                f"{name!r} clashes with an existing KVS."
            )

        # Resolve the grounding spec. Priority:
        #   1. Explicit ``grounding`` argument wins.
        #   2. ``length_m`` collapses to a single-rod spec (legacy).
        #   3. Otherwise use the project default (3 x 0.5 m triangle).
        if grounding is not None:
            grounding_spec = grounding
        elif length_m is not None:
            grounding_spec = GroundingSystemSpec(
                electrodes=[
                    RodElectrodeSpec(
                        length_m=float(length_m), depth_m=depth_m,
                    ),
                ],
            )
        else:
            grounding_spec = _default_aux_grounding()

        self.auxiliary_electrode = AuxiliaryElectrodePlacement(
            name=name,
            position_xy=xy,
            grounding=grounding_spec,
        )
        return name

    # ------------------------------------------------------------------
    # Voltage probe (Spannungssonde)
    # ------------------------------------------------------------------

    def add_voltage_probe(
        self,
        *,
        inline_fraction: Optional[float] = None,
        perpendicular_fraction: Optional[float] = None,
        perpendicular_side: str = "left",
        position_xy: Optional[tuple[float, float]] = None,
        position_lat_lon: Optional[tuple[float, float]] = None,
        name: str = "probe",
        depth_m: float = 0.0,
    ) -> str:
        r"""Place the voltage probe (Spannungssonde).

        The voltage probe replaces the "remote earth" reference of
        a simple ``U_sub / I`` measurement with the *actual*
        ground potential at a chosen point — exactly what a
        field-deployed grounding-impedance meter does.

        Four positioning kinds are accepted:

        * ``inline_fraction`` — along the substation → aux axis,
          at a fraction :math:`f` of the aux distance
          (:math:`f = 0` is the substation, :math:`f = 1` is the
          Hilfserder, :math:`f = 0.5` is the midpoint, which is
          the standard "50 % rule"). Requires
          :meth:`add_auxiliary_electrode` to have been called.
        * ``perpendicular_fraction`` (with ``perpendicular_side``
          ``"left"`` or ``"right"``) — 90° rotated from the
          substation → aux axis (left = CCW = north when aux runs
          east, right = CW = south). The probe distance from the
          substation is :math:`f \cdot D_\text{aux}`. This matches
          the *senkrecht zur Hilfserdertrasse* setup commonly
          used to suppress inductive coupling between the
          current-feed loop and the voltage-probe lead.
        * ``position_xy`` — explicit ENU metres.
        * ``position_lat_lon`` — WGS84 (projects through
          :attr:`frame_origin_lat_lon`).

        Examples
        --------
        Classic 50 % inline probe between substation and a 200 m
        east Hilfserder:

        >>> layout.add_auxiliary_electrode(distance_m=200.0,
        ...                                direction_deg=0.0)
        >>> layout.add_voltage_probe(inline_fraction=0.5)
        # probe ends up at (100, 0) relative to the substation

        90° perpendicular probe at the same fraction, on the
        "left" (= north) side:

        >>> layout.add_voltage_probe(perpendicular_fraction=0.5,
        ...                          perpendicular_side="left")
        # probe ends up at (0, 100) relative to the substation
        """
        provided = sum(
            arg is not None
            for arg in (
                inline_fraction,
                perpendicular_fraction,
                position_xy,
                position_lat_lon,
            )
        )
        if provided != 1:
            raise ValueError(
                "OrtsnetzLayout.add_voltage_probe: exactly one of "
                "``inline_fraction``, ``perpendicular_fraction``, "
                "``position_xy`` or ``position_lat_lon`` must be set."
            )
        if perpendicular_side not in ("left", "right"):
            raise ValueError(
                "OrtsnetzLayout.add_voltage_probe: perpendicular_side "
                f"must be 'left' or 'right', got {perpendicular_side!r}."
            )

        if inline_fraction is not None or perpendicular_fraction is not None:
            if self.auxiliary_electrode is None:
                raise RuntimeError(
                    "OrtsnetzLayout.add_voltage_probe: the inline / "
                    "perpendicular fractions need an auxiliary "
                    "electrode -- call ``add_auxiliary_electrode`` "
                    "first."
                )
            sx, sy = self.substation_xy
            ax, ay = self.auxiliary_electrode.position_xy
            d_vec = (ax - sx, ay - sy)
            d_aux = math.hypot(*d_vec)
            if d_aux <= 0.0:
                raise RuntimeError(
                    "OrtsnetzLayout.add_voltage_probe: the auxiliary "
                    "electrode coincides with the substation -- the "
                    "fractional placement is undefined."
                )
            u_inline = (d_vec[0] / d_aux, d_vec[1] / d_aux)
            if inline_fraction is not None:
                f = float(inline_fraction)
                if not 0.0 <= f <= 1.0:
                    raise ValueError(
                        "OrtsnetzLayout.add_voltage_probe: "
                        "inline_fraction must be in [0, 1], got "
                        f"{f}."
                    )
                xy = (sx + f * d_aux * u_inline[0],
                      sy + f * d_aux * u_inline[1])
            else:
                f = float(perpendicular_fraction)
                if not 0.0 <= f <= 1.0:
                    raise ValueError(
                        "OrtsnetzLayout.add_voltage_probe: "
                        "perpendicular_fraction must be in [0, 1], "
                        f"got {f}."
                    )
                # 90° rotation: left = CCW, right = CW.
                if perpendicular_side == "left":
                    u_perp = (-u_inline[1], u_inline[0])
                else:
                    u_perp = (u_inline[1], -u_inline[0])
                xy = (sx + f * d_aux * u_perp[0],
                      sy + f * d_aux * u_perp[1])
        elif position_lat_lon is not None:
            xy = self.lat_lon_to_xy(*position_lat_lon)
        else:
            assert position_xy is not None
            xy = tuple(position_xy)

        if name == self.substation_name:
            raise ValueError(
                f"OrtsnetzLayout.add_voltage_probe: name {name!r} "
                "clashes with the substation name."
            )
        if any(k.name == name for k in self.kvs_placements):
            raise ValueError(
                f"OrtsnetzLayout.add_voltage_probe: name {name!r} "
                "clashes with an existing KVS."
            )
        if (
            self.auxiliary_electrode is not None
            and name == self.auxiliary_electrode.name
        ):
            raise ValueError(
                f"OrtsnetzLayout.add_voltage_probe: name {name!r} "
                "clashes with the auxiliary electrode."
            )

        self.voltage_probe = VoltageProbePlacement(
            name=name,
            position_xy=xy,
            depth_m=depth_m,
        )
        return name

    # ------------------------------------------------------------------
    # Measured grounding impedance
    # ------------------------------------------------------------------

    def measured_grounding_impedance(
        self,
        result: object,
        *,
        frequency_index: int = 0,
        source_magnitude_A: float = 1.0,
        probe_xy: Optional[tuple[float, float]] = None,
        probe_depth_m: float = 0.0,
    ) -> complex:
        r"""Compute the simulated fall-of-potential measurement reading.

        Returns
        -------
        complex
            :math:`Z_\text{meas} =
            (\varphi_\text{sub} - \varphi_\text{probe}) / I_\text{src}`
            at the requested frequency.

        When neither ``probe_xy`` nor :attr:`voltage_probe` is set,
        the reference falls back to remote earth, i.e. the method
        returns :math:`\varphi_\text{sub} / I_\text{src}`.

        Parameters
        ----------
        result
            :class:`FieldResult` from :meth:`Engine.solve`.
        frequency_index, source_magnitude_A
            Pick the frequency and the test-current normalisation.
        probe_xy
            Optional explicit probe position in local ENU metres
            (overrides :attr:`voltage_probe` for this call only).
            Lets the caller compare several probe geometries from
            a single solve -- useful for the
            "0° vs 90° probe at the same distance" comparison
            without having to call ``add_voltage_probe`` /
            ``to_world`` again.
        probe_depth_m
            Sampling depth for the override (default 0 = surface).
            Ignored when ``probe_xy`` is ``None``.

        Raises
        ------
        RuntimeError
            If no auxiliary electrode is configured (the
            measurement loop is then physically open).
        """
        if self.auxiliary_electrode is None:
            raise RuntimeError(
                "OrtsnetzLayout.measured_grounding_impedance: no "
                "auxiliary electrode is configured. Call "
                "``add_auxiliary_electrode`` first."
            )

        # Find the substation cluster's anchor in the result.
        electrode_potentials = getattr(result, "electrode_potentials", None)
        if electrode_potentials is None:
            raise TypeError(
                "OrtsnetzLayout.measured_grounding_impedance: "
                "``result`` does not look like a FieldResult."
            )
        sub_anchor: Optional[str] = None
        prefix = f"{self.substation_name}_"
        for ename in electrode_potentials:
            if ename.startswith(prefix):
                sub_anchor = ename
                break
        if sub_anchor is None:
            raise RuntimeError(
                "OrtsnetzLayout.measured_grounding_impedance: no "
                "electrode with prefix "
                f"{self.substation_name!r} found in the result. "
                "Did you re-run ``to_world`` before solving?"
            )

        phi_sub = electrode_potentials[sub_anchor][frequency_index]

        # Pick the probe location: explicit override > layout's
        # voltage_probe > remote earth fall-back.
        if probe_xy is not None:
            sample_xy = tuple(probe_xy)
            sample_depth = probe_depth_m
        elif self.voltage_probe is not None:
            sample_xy = self.voltage_probe.position_xy
            sample_depth = self.voltage_probe.depth_m
        else:
            sample_xy = None

        if sample_xy is None:
            phi_ref: complex = 0.0 + 0.0j
        else:
            point = np.asarray(
                [[sample_xy[0], sample_xy[1], sample_depth]],
                dtype=float,
            )
            phi_ref = complex(
                result.potential(
                    point, frequency_index=frequency_index,
                )[0]
            )

        return (phi_sub - phi_ref) / source_magnitude_A

    # ------------------------------------------------------------------
    # Current-balance plausibility check
    # ------------------------------------------------------------------

    def verify_current_balance(
        self,
        result: object,
        *,
        frequency_index: int = 0,
        source_magnitude_A: float = 1.0,
        tolerance: float = 0.02,
    ) -> dict:
        r"""Plausibility check: sum of leakage currents must close the
        measurement loop.

        With the v0.7 closed-loop wiring the engine sees:

        * a positive ``+I`` injection at the substation cluster, which
          distributes itself across the substation grounding *and* the
          PEN-bonded foundations (because the PEN cables are
          finite-impedance branches that share the substation
          potential);
        * a negative ``-I`` injection at the Hilfserder cluster.

        Per Kirchhoff the *total* leakage current at every other
        electrode must vanish:

        .. math::
            \sum_e I_e^\text{(positive side)} \approx +I_\text{src}
            ,\qquad
            \sum_e I_e^\text{(aux side)} \approx -I_\text{src}

        Returns a dict with the two sums and a boolean ``ok`` flag
        indicating that both sums match :math:`\pm I_\text{src}`
        within ``tolerance`` (default 2 % of the test current).

        The "positive side" is every electrode whose name does *not*
        start with the auxiliary's name prefix; the "aux side" is
        every electrode whose name *does* start with that prefix.
        """
        if self.auxiliary_electrode is None:
            raise RuntimeError(
                "OrtsnetzLayout.verify_current_balance: no auxiliary "
                "electrode is configured. The measurement loop is "
                "physically open; current balance is undefined."
            )
        aux_prefix = f"{self.auxiliary_electrode.name}_"
        currents = result.electrode_currents
        pos_sum: complex = 0j
        neg_sum: complex = 0j
        for ename, i_list in currents.items():
            i = i_list[frequency_index]
            if ename.startswith(aux_prefix) or ename == self.auxiliary_electrode.name:
                neg_sum += i
            else:
                pos_sum += i
        expected_pos = +source_magnitude_A
        expected_neg = -source_magnitude_A
        err_pos = abs(pos_sum - expected_pos)
        err_neg = abs(neg_sum - expected_neg)
        tol_abs = tolerance * source_magnitude_A
        ok = (err_pos <= tol_abs) and (err_neg <= tol_abs)
        return {
            "positive_side_A": pos_sum,
            "auxiliary_side_A": neg_sum,
            "expected_positive_A": expected_pos,
            "expected_auxiliary_A": expected_neg,
            "abs_error_positive_A": err_pos,
            "abs_error_auxiliary_A": err_neg,
            "tolerance_A": tol_abs,
            "ok": ok,
        }

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
        escape_radius_m: Optional[float] = None,
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
        escape_radius_m
            Radius around the start *and* end anchor inside which
            obstacles are not enforced. Defaults to
            ``min_segment_length_m`` (one cell of slack) which is
            usually enough to handle a substation that landed inside
            or right next to a foundation polygon in a real OSM
            extract. Raise this when an anchor is wedged into a
            dense city block; pass ``0.0`` for the strict pre-v0.7
            behaviour.

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
            escape_radius_m=escape_radius_m,
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

    def connect_all_buildings(
        self,
        *,
        source_anchors: Optional[list[str]] = None,
        max_passes: int = 4,
        margin_m: float = 3.0,
        lateral_clearance_m: Optional[float] = None,
        min_segment_length_m: float = 10.0,
        escape_radius_m: Optional[float] = None,
        clearance_m: Optional[float] = None,
    ) -> "LateralCoverageResult":
        r"""Connect *every* house to the LV network, adding laterals as needed.

        :meth:`connect_buildings` only taps each house to the nearest
        existing cable with a short (at most two-segment) stub and skips
        houses whose stub would cross another footprint -- typically houses
        with no cable in reach. This method closes that gap so the layout
        models a real LV network in which every customer is served.

        Strategy ("laterals from the nearest node" model):

        1. Tap whatever is already reachable via :meth:`connect_buildings`.
        2. For every still-unconnected house (farthest from any source node
           first), route an obstacle-avoiding PEN *lateral*
           (:meth:`add_pen_cable`, full Manhattan A\*) from the nearest
           connected source node (substation or a KVS) to a free point just
           outside the house, then re-tap. Re-tapping frequently connects
           neighbouring houses to the same new lateral for free.
        3. Repeat until no house remains or no further progress is made.

        Houses enclosed by other footprints on every side stay unconnected
        and are reported in :attr:`LateralCoverageResult.islands` rather
        than raising.

        Parameters
        ----------
        source_anchors
            Names of the nodes laterals may originate from. ``None`` (the
            default) uses the substation plus every KVS galvanically
            connected to it through the existing PEN-cable graph (see
            :meth:`_anchors_connected_to_substation`), so a stranded KVS is
            never used as a source and cannot create an isolated
            sub-network.
        max_passes
            Maximum number of tap/lateral rounds. Each round adds at most
            one lateral per still-unconnected house.
        margin_m
            Extra distance, beyond the house's bounding-box half-diagonal,
            at which the lateral end point is placed outside the footprint.
        lateral_clearance_m
            Obstacle inflation used when searching for a free lateral end
            point. Defaults to :attr:`obstacle_clearance_m`.
        min_segment_length_m, escape_radius_m, clearance_m
            Forwarded to :meth:`add_pen_cable` for the lateral routing
            (with one finer-grid retry on failure). ``clearance_m`` also
            overrides the tap clearance in :meth:`connect_buildings`.

        Returns
        -------
        LateralCoverageResult
            ``laterals`` (added cable names), ``islands`` (still-unconnected
            footprint indices), and ``connections`` (snapshot after the
            pass).

        Raises
        ------
        RuntimeError
            If ``source_anchors`` resolves to an empty list (e.g. passed
            explicitly as ``[]``).
        """
        import warnings

        if not self.footprints:
            return LateralCoverageResult([], [], list(self.connections))

        clearance = (
            self.obstacle_clearance_m if clearance_m is None else clearance_m
        )
        lat_clear = (
            clearance if lateral_clearance_m is None else lateral_clearance_m
        )

        if source_anchors is None:
            source_anchors = self._anchors_connected_to_substation()
        nodes = [(a, self._anchor_position(a)) for a in source_anchors]
        if not nodes:
            raise RuntimeError(
                "OrtsnetzLayout.connect_all_buildings: no source anchor to "
                "originate laterals from. Pass ``source_anchors`` or route "
                "a PEN cable from the substation first."
            )

        def _tap() -> None:
            # connect_buildings warns per skipped house; islands are reported
            # at the end, so silence the per-house noise here. It also raises
            # when there are no cables yet -- skip the tap in that case so the
            # very first lateral can still be laid from the substation.
            if not self.pen_cables:
                return
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.connect_buildings(
                    only_unconnected=True, clearance_m=clearance,
                )

        def _nearest_distance(i: int) -> float:
            cx, cy = self.footprints[i].centroid_xy_m()
            return min(
                math.hypot(cx - px, cy - py) for _, (px, py) in nodes
            )

        laterals: list[str] = []
        for _ in range(max_passes):
            _tap()
            done = {c.house_idx for c in self.connections}
            orphans = [
                i for i in range(len(self.footprints)) if i not in done
            ]
            if not orphans:
                break
            progressed = False
            for i in sorted(orphans, key=_nearest_distance, reverse=True):
                if i in {c.house_idx for c in self.connections}:
                    continue  # already tapped via a neighbour's lateral
                fp = self.footprints[i]
                cx, cy = fp.centroid_xy_m()
                src, _src_xy = min(
                    nodes,
                    key=lambda nd: math.hypot(cx - nd[1][0], cy - nd[1][1]),
                )
                end_xy = self._free_point_near_house(
                    fp, self._anchor_position(src),
                    clearance=lat_clear, margin_m=margin_m,
                )
                if end_xy is None:
                    continue
                name = f"lateral_house_{i:04d}"
                if any(c.name == name for c in self.pen_cables):
                    continue
                routed = False
                eff_escape = (
                    escape_radius_m
                    if escape_radius_m is not None
                    else min_segment_length_m
                )
                # Grid ladder: configured -> half (tighter final bridges) ->
                # fifth (maze escape on long winding routes, where a fine grid
                # would otherwise exhaust the A* iteration budget).
                for grid, esc in (
                    (min_segment_length_m, escape_radius_m),
                    (max(1.0, min_segment_length_m / 2.0), 2.0 * eff_escape),
                    (max(1.0, min_segment_length_m / 5.0), 2.0 * eff_escape),
                ):
                    try:
                        self.add_pen_cable(
                            start=src, end_xy=end_xy, name=name,
                            min_segment_length_m=grid,
                            clearance_m=clearance, escape_radius_m=esc,
                        )
                        routed = True
                        break
                    except RuntimeError:
                        continue
                if not routed:
                    continue
                laterals.append(name)
                progressed = True
                _tap()  # tap house i (and any neighbours now in reach)
            if not progressed:
                break

        done = {c.house_idx for c in self.connections}
        islands = [i for i in range(len(self.footprints)) if i not in done]
        return LateralCoverageResult(laterals, islands, list(self.connections))

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
        foundation_mask: Optional[list[bool]] = None,
    ) -> World:
        """Materialise the layout into a :class:`groundfield.World`.

        Each :class:`PenCable` becomes a chain of straight-line
        :class:`Conductor` segments between *PEN junction*
        electrodes (tiny rods at every corner and tap point) so
        the resulting cable physically follows the Manhattan
        polyline. Each :class:`BuildingConnection` becomes one
        additional conductor from the house foundation anchor to
        the nearest PEN junction on the serving cable.

        Parameters
        ----------
        foundation_mask
            Optional per-house bool list. When given, only houses
            with a ``True`` entry receive a foundation electrode
            (and therefore a PEN service drop); the remaining
            houses are kept in the geometry for visualisation but
            are *not* materialised in the world. Combine with
            :meth:`foundation_mask` for a reproducible penetration
            sweep.

        Notes
        -----
        Other parameters follow the v0.7 :meth:`to_world` defaults.
        When :attr:`auxiliary_electrode` is set (i.e.
        :meth:`add_auxiliary_electrode` was called), the source's
        ``return_to`` is wired to the auxiliary anchor so the
        engine treats it as the test-current return path. Without
        an auxiliary electrode the source returns through the
        remote-earth boundary (= the historical pre-v0.7
        behaviour).
        """
        if foundation_mask is not None and len(foundation_mask) != len(
            self.footprints,
        ):
            raise ValueError(
                "OrtsnetzLayout.to_world: foundation_mask length "
                f"{len(foundation_mask)} does not match footprints "
                f"count {len(self.footprints)}."
            )
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

        # 3. House groundings (foundations). When a foundation_mask is
        #    supplied, only houses with mask[i] == True receive an
        #    electrode -- the remaining ones stay in the geometry
        #    (still rendered by ``plot``) but contribute nothing to
        #    the solver world.
        house_spec = house_grounding or _default_house_grounding()
        house_anchors: dict[int, str] = {}
        for i, fp in enumerate(self.footprints):
            if foundation_mask is not None and not foundation_mask[i]:
                continue
            centroid = fp.centroid_xy_m()
            anchor = house_spec.build_at(
                world,
                site_xy=centroid,
                name_prefix=f"house_{i:03d}",
                rng=rng,
            )
            if anchor is not None:
                house_anchors[i] = anchor

        # 3a. Auxiliary current electrode (Hilfserder), if configured.
        # Uses the same GroundingSystemSpec.build_at path as the
        # substation / KVS so any electrode topology is supported.
        # The returned anchor is the first present electrode (e.g.
        # ``aux_rod_0`` for the default 3-rod triangle); the
        # remaining electrodes are galvanically bonded inside
        # build_at, so the whole bundle forms one cluster.
        aux_anchor: Optional[str] = None
        if self.auxiliary_electrode is not None:
            aux = self.auxiliary_electrode
            aux_anchor = aux.grounding.build_at(
                world,
                site_xy=aux.position_xy,
                name_prefix=aux.name,
                rng=rng,
            )
            if aux_anchor is None:
                raise RuntimeError(
                    "OrtsnetzLayout.to_world: auxiliary electrode "
                    "grounding produced zero electrodes "
                    f"({aux.name!r}). Increase presence_prob."
                )

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

        # 6. Source at the substation. When an auxiliary electrode is
        # configured (fall-of-potential measurement setup) the test
        # current physically returns through that electrode. The v0.7
        # image solver consumes :attr:`Source.attached_to` only --
        # :attr:`Source.return_to` is recorded for documentation /
        # future solver upgrades but is not honoured by the engine.
        # We therefore *explicitly* close the measurement loop by
        # adding a second :class:`CurrentSource` at the auxiliary
        # anchor whose magnitude has the opposite sign
        # (``phase_deg=180``). Net injected charge is then zero, the
        # potential field develops the expected dipole pattern with
        # a positive trumpet at the substation and a negative one at
        # the Hilfserder -- exactly what a real measurement loop
        # produces.
        if include_source:
            create_source(
                world,
                attached_to=substation_anchor,
                return_to=aux_anchor,
                magnitude=source_magnitude_A,
            )
            if aux_anchor is not None:
                create_source(
                    world,
                    name=f"{aux_anchor}_return",
                    attached_to=aux_anchor,
                    magnitude=source_magnitude_A,
                    phase_deg=180.0,
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
        foundation_mask: Optional[list[bool]] = None,
        figsize: tuple[float, float] = (12.0, 8.0),
    ) -> "_mpl_axes.Axes":
        """Render the layout with matplotlib.

        Parameters
        ----------
        foundation_mask
            Optional per-house bool list (one entry per footprint).
            When supplied, houses with ``True`` get a saturated
            fill colour (foundation electrode present) and the
            remaining houses get a faded fill (no foundation).
            Combine with :meth:`foundation_mask` to visualise
            different penetration scenarios.

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

        if foundation_mask is not None and len(foundation_mask) != len(
            self.footprints,
        ):
            raise ValueError(
                "OrtsnetzLayout.plot: foundation_mask length "
                f"{len(foundation_mask)} does not match footprints "
                f"count {len(self.footprints)}."
            )

        # Building footprints.
        for i, fp in enumerate(self.footprints):
            xs = [p[0] for p in fp.polygon_xy_m] + [fp.polygon_xy_m[0][0]]
            ys = [p[1] for p in fp.polygon_xy_m] + [fp.polygon_xy_m[0][1]]
            if foundation_mask is not None:
                # With-foundation: saturated green-grey; without: faded.
                if foundation_mask[i]:
                    colour = "#7fbf7f"   # has Fundamenterder
                    edge = "#2a6f2a"
                else:
                    colour = "#f4f4f4"   # no Fundamenterder
                    edge = "0.7"
            else:
                colour = "0.92" if i in connected_idx else "#fde0dc"
                edge = "0.4"
            ax.fill(xs, ys, color=colour, edgecolor=edge,
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

        # Auxiliary current electrode (Hilfserder).
        if self.auxiliary_electrode is not None:
            aux = self.auxiliary_electrode
            # Dotted return-current line from substation to aux.
            ax.plot(
                [self.substation_xy[0], aux.position_xy[0]],
                [self.substation_xy[1], aux.position_xy[1]],
                color="purple", lw=0.8, ls=":", zorder=2,
                label="Measurement feed (return)",
            )
            ax.plot(*aux.position_xy, marker="v",
                    color="purple", markersize=13, zorder=7,
                    label=f"Auxiliary {aux.name!s} (Hilfserder)")

        # Voltage probe (Spannungssonde).
        if self.voltage_probe is not None:
            probe = self.voltage_probe
            ax.plot(*probe.position_xy, marker="*",
                    color="lime", markersize=18, zorder=8,
                    markeredgecolor="darkgreen", markeredgewidth=1.2,
                    label=f"Probe {probe.name!s} (Spannungssonde)")
        if show_unconnected and foundation_mask is None and any(
            i not in connected_idx for i in range(len(self.footprints))
        ):
            ax.plot([], [], marker="s", color="#fde0dc",
                    markersize=11, linestyle="None",
                    label="Unconnected house")
        if foundation_mask is not None:
            ax.plot([], [], marker="s", color="#7fbf7f",
                    markersize=11, linestyle="None",
                    label="House with Fundamenterder")
            ax.plot([], [], marker="s", color="#f4f4f4",
                    markeredgecolor="0.7",
                    markersize=11, linestyle="None",
                    label="House without Fundamenterder")

        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_title("TN-Ortsnetz layout (Manhattan PEN routing)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------------------------------------------------
    # Surface-potential plot with measurement markers
    # ------------------------------------------------------------------

    def plot_surface_potential(
        self,
        result: object,
        world: World,
        *,
        frequency_index: int = 0,
        padding_m: float = 50.0,
        n: int = 150,
        levels: int = 41,
        log: bool = False,
        symmetric: bool = False,
        two_slope: bool = True,
        cmap: str = "RdBu_r",
        show_electrodes: bool = True,
        figsize: tuple[float, float] = (11.0, 8.5),
        title: Optional[str] = None,
    ):
        r"""Surface-potential pseudo-colour plot with measurement
        markers (substation + Hilfserder + Spannungssonde).

        Renders $\varphi(x, y, z=0)$ on a regular grid sized to the
        world bounding box plus ``padding_m`` and overlays the
        substation, Hilfserder and Spannungssonde markers plus a
        dotted return-current line. Three colour-axis modes are
        supported:

        * ``two_slope=True`` (default) -- uses
          :class:`matplotlib.colors.TwoSlopeNorm` centred at
          $\varphi = 0$ so the colour resolution above and below
          zero is balanced regardless of the actual range. Ideal
          for the fall-of-potential setup, where the Hilfserder
          sits at a strongly negative potential (e.g. -10 V) and
          the substation + foundations together rise to only a
          few volts -- on a single linear scale the positive
          trumpet collapses into a single colour. The colourbar
          uses non-uniform contour levels so each half of the
          scale carries the same number of fills.
        * ``symmetric=True`` -- the classic
          :math:`[-|\varphi|_\text{max}, +|\varphi|_\text{max}]`
          range. Equivalent to ``TwoSlopeNorm`` only when the
          extremes are symmetric; otherwise large negative
          potentials clip the positive side. Setting
          ``symmetric=True`` *implicitly* turns off
          ``two_slope``.
        * ``log=True`` -- log-scale on $|\varphi|$. Useful when the
          potential decays across several decades toward remote
          earth.

        Parameters
        ----------
        result
            :class:`FieldResult` from :meth:`Engine.solve`.
        world
            Companion world used to derive the plot extent and
            (when ``show_electrodes=True``) to overlay the
            electrode geometry.
        frequency_index
            Index into :attr:`FieldResult.frequencies`.
        padding_m
            Extra space around the world bounding box in m.
        n
            Grid resolution per axis (``n × n`` evaluation points).
        levels
            Number of contour-fill levels. Split half-and-half
            between the negative and positive ranges when
            ``two_slope=True``.
        log, symmetric, two_slope
            Mutually-influencing scaling modes (see above).
        cmap
            Matplotlib colormap (diverging recommended).
        show_electrodes
            Overlay the electrode geometry (small markers + lines).
        figsize, title
            Figure size and optional title override.

        Returns
        -------
        matplotlib.figure.Figure
            The figure the plot was drawn on. Use ``fig.axes[0]``
            for further annotations.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

        from groundfield.postprocess.plotting import (
            _draw_electrodes, _make_grid, world_bounds_xy,
        )

        # Mutual exclusion of the scaling modes -- log dominates;
        # symmetric overrides two_slope. Default is two_slope.
        if log:
            two_slope = False
            symmetric = False
        elif symmetric:
            two_slope = False

        x_min, x_max, y_min, y_max = world_bounds_xy(world)
        extent = (
            x_min - padding_m, x_max + padding_m,
            y_min - padding_m, y_max + padding_m,
        )
        A, B, flat = _make_grid("xy", extent, 0.0, n)
        phi = result.potential(flat, frequency_index=frequency_index).real
        phi = phi.reshape(A.shape)

        fig, ax = plt.subplots(figsize=figsize)

        if log:
            phi_abs = np.abs(phi)
            positive = phi_abs[phi_abs > 0]
            v_min = float(positive.min()) if positive.size else 1e-9
            v_max = float(phi_abs.max()) if phi_abs.size else 1.0
            norm = LogNorm(vmin=max(v_min, 1e-9),
                           vmax=max(v_max, 10 * v_min))
            cs = ax.contourf(A, B, phi_abs, levels=levels,
                             cmap=cmap, norm=norm)
            cbar_label = "|φ| in V (log scale)"
        elif symmetric:
            v = float(np.max(np.abs(phi)))
            norm = Normalize(vmin=-v, vmax=v)
            cs = ax.contourf(A, B, phi, levels=levels,
                             cmap=cmap, norm=norm)
            cbar_label = "Potential φ in V"
        elif two_slope:
            v_neg = float(min(phi.min(), -1e-9))
            v_pos = float(max(phi.max(), +1e-9))
            norm = TwoSlopeNorm(vmin=v_neg, vcenter=0.0, vmax=v_pos)
            # Split contour levels half-and-half so each side of
            # zero carries the same visual resolution.
            n_per_side = max(2, levels // 2)
            neg_levels = np.linspace(v_neg, 0.0, n_per_side + 1)[:-1]
            pos_levels = np.linspace(0.0, v_pos, n_per_side + 1)
            level_array = np.concatenate([neg_levels, pos_levels])
            cs = ax.contourf(A, B, phi, levels=level_array,
                             cmap=cmap, norm=norm, extend="both")
            cbar_label = (
                "Potential φ in V (TwoSlopeNorm: equal colour "
                "resolution above / below 0)"
            )
        else:
            cs = ax.contourf(A, B, phi, levels=levels, cmap=cmap)
            cbar_label = "Potential φ in V"

        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label(cbar_label)

        ax.set_xlabel("x in m")
        ax.set_ylabel("y in m")
        ax.set_aspect("equal")
        if title is None:
            f_hz = result.frequencies[frequency_index]
            title = (
                f"Surface potential φ(x, y, z = 0 m), "
                f"f = {f_hz:g} Hz"
            )
        ax.set_title(title)

        if show_electrodes:
            _draw_electrodes(ax, world, "xy")

        # Substation marker.
        ax.plot(*self.substation_xy, marker="D",
                color="tab:orange", markersize=14, zorder=10,
                markeredgecolor="black", markeredgewidth=0.8,
                label="Substation")
        # Auxiliary electrode (Hilfserder).
        if self.auxiliary_electrode is not None:
            aux = self.auxiliary_electrode
            ax.plot(
                [self.substation_xy[0], aux.position_xy[0]],
                [self.substation_xy[1], aux.position_xy[1]],
                color="purple", lw=0.8, ls=":", zorder=9,
            )
            ax.plot(*aux.position_xy, marker="v",
                    color="purple", markersize=13, zorder=10,
                    markeredgecolor="black", markeredgewidth=0.8,
                    label="Hilfserder")
        # Voltage probe (Spannungssonde).
        if self.voltage_probe is not None:
            probe = self.voltage_probe
            ax.plot(*probe.position_xy, marker="*",
                    color="lime", markersize=18, zorder=11,
                    markeredgecolor="darkgreen", markeredgewidth=1.2,
                    label="Spannungssonde")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        return fig

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

    def _anchors_connected_to_substation(self) -> list[str]:
        """Source anchors galvanically connected to the substation.

        Walks the node graph induced by PEN cables whose *both* endpoints
        are named anchors (``start_anchor`` / ``end_anchor``) and returns
        the connected component containing the substation, ordered
        substation-first then KVS in placement order. A KVS reachable only
        through a free-ended cable (e.g. a trunk tail) is *not* included.
        """
        adjacency: dict[str, set[str]] = {self.substation_name: set()}
        for k in self.kvs_placements:
            adjacency.setdefault(k.name, set())
        for cable in self.pen_cables:
            a, b = cable.start_anchor, cable.end_anchor
            if a in adjacency and b is not None and b in adjacency:
                adjacency[a].add(b)
                adjacency[b].add(a)
        seen = {self.substation_name}
        stack = [self.substation_name]
        while stack:
            current = stack.pop()
            for neighbour in adjacency.get(current, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return [self.substation_name] + [
            k.name for k in self.kvs_placements if k.name in seen
        ]

    def _free_point_near_house(
        self,
        fp: BuildingFootprint,
        node_xy: tuple[float, float],
        *,
        clearance: float,
        margin_m: float = 3.0,
    ) -> Optional[tuple[float, float]]:
        r"""Free $(x, y)$ just outside ``fp``, biased towards ``node_xy``.

        Returns the first candidate -- placed at growing distances and a
        fan of bearings around the house→node direction -- that lies
        outside every footprint box inflated by ``clearance``. Returns
        ``None`` if the house is enclosed on all tried bearings.
        """
        (cx, cy), (dx, dy) = fp.axis_aligned_bounding_rectangle()
        half_diag = 0.5 * math.hypot(dx, dy)
        ux, uy = node_xy[0] - cx, node_xy[1] - cy
        norm = math.hypot(ux, uy)
        if norm < 1e-9:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = ux / norm, uy / norm
        boxes = [
            self._footprint_to_box(f, j).inflated(clearance)
            for j, f in enumerate(self.footprints)
        ]
        for extra in (0.0, 3.0, 6.0, 10.0, 16.0):
            dist = half_diag + margin_m + extra
            for ang_deg in (
                0, 20, -20, 45, -45, 70, -70, 110, -110, 160, -160, 180,
            ):
                theta = math.radians(ang_deg)
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                vx = cos_t * ux - sin_t * uy
                vy = sin_t * ux + cos_t * uy
                px, py = cx + dist * vx, cy + dist * vy
                if not any(b.contains_point(px, py) for b in boxes):
                    return (px, py)
        return None

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


def _stable_uniform(salt: int, key: object) -> float:
    r"""Deterministic uniform :math:`[0, 1)` draw from a salt + key.

    Built on MD5 (chosen for stable cross-Python-version hashing,
    not for cryptographic strength) so the value depends only on
    the textual representation of the inputs and is reproducible
    across runs, processes and Python versions.
    """
    blob = f"{salt}|{key}".encode("utf-8")
    digest = hashlib.md5(blob).digest()[:8]
    n = int.from_bytes(digest, "big")
    return n / float(1 << 64)


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
