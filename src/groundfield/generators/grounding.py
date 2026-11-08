r"""Composable grounding-system specifications.

A :class:`GroundingSystemSpec` represents *one* electrically connected
grounding installation: a (possibly heterogeneous) set of
:class:`~groundfield.generators.electrode_specs.ElectrodeSpec` instances
that together form a single galvanic cluster. The class is shared between

* the substation grounding (typically ring + several rods, optionally a
  strip and/or a foundation electrode),
* every cable cabinet grounding (typically a single rod, but any
  combination is allowed),
* the grounding of a house or another building (foundation, rod, mesh,
  or any combination).

Two steps turn a spec into electrodes:

1. :meth:`GroundingSystemSpec.resolve_electrodes` makes every random
   decision — one Bernoulli draw on ``presence_prob`` per electrode and
   one draw per :class:`~groundfield.generators.distributions.Distribution`
   field — and returns the present electrodes with concrete values
   (:class:`~groundfield.generators.plan.PlannedElectrode`).
2. :func:`groundfield.generators.materialise.materialise_site` creates
   them in a world and bonds every electrode to the first present one
   (the *anchor*) with an ideal conductor, so the site is one cluster.

:meth:`GroundingSystemSpec.build_at` does both in one call.

Building footprints
-------------------
:meth:`GroundingSystemSpec.with_footprint` derives the foundation
geometry from a building outline. It is the one rule used by every
front-end (``TnNetworkGenerator`` with an OSM placement and
``LvNetworkLayout``):

* ``"ombr"`` — the oriented minimum bounding rectangle (OMBR) of the
  outline: size, orientation and centre;
* ``"polygon"`` — the outline itself as a closed ring (DIN 18014: the
  strip foundation runs along the building perimeter; ring-style
  foundations only);
* ``"spec"`` — the spec geometry unchanged.

In both outline rules an outline narrower than
:data:`MIN_FOUNDATION_SIDE_M` (a mapped wall or fence) is widened to that
side with a :class:`UserWarning`.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from pydantic import Field

from groundfield.generators.base import GeneratorConfig
from groundfield.generators.distributions import Distribution
from groundfield.generators.electrode_specs import (
    ElectrodeSpec,
    FoundationElectrodeSpec,
    RingElectrodeSpec,
    RodElectrodeSpec,
    StripElectrodeSpec,
)
from groundfield.generators.materialise import (
    MIN_POLYGON_EDGE_M,
    clean_polygon,
    concrete_shell,
)
from groundfield.generators.plan import PlannedElectrode, PlannedSite
from groundfield.world import World

if TYPE_CHECKING:  # pragma: no cover - typing only
    from groundfield.geo.footprint import BuildingFootprint

__all__ = [
    "MIN_FOUNDATION_SIDE_M",
    "FoundationOutline",
    "GroundingSystemSpec",
]

MIN_FOUNDATION_SIDE_M = 1.0
"""Shortest side (m) of a foundation rectangle taken from a footprint's OMBR.

A narrower outline — a mapped wall or fence, e.g. 10 m x 5 mm — is widened
to this side with a :class:`UserWarning`; as a 5 mm wide mesh it could not
even be built (the mesh span is shorter than the wire radius, audit L5).
The AP1 pipeline applied the same 1 m floor to its OMBR foundations."""

FoundationOutline = Literal["spec", "ombr", "polygon"]
"""How a building footprint shapes a foundation electrode.

- ``"ombr"`` — oriented minimum bounding rectangle of the outline (size,
  orientation and centre).
- ``"polygon"`` — the outline itself (closed ring; ``style="ring"``
  foundations only).
- ``"spec"`` — ignore the footprint; keep the spec geometry.
"""


def _to_float(value: Union[float, Distribution], rng: np.random.Generator) -> float:
    """Resolve a ``float | Distribution`` to ``float`` (sample once)."""
    if isinstance(value, Distribution):
        return float(value.sample(rng))
    return float(value)


def _widened_ombr(
    footprint: "BuildingFootprint",
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """OMBR of ``footprint`` with sides of at least :data:`MIN_FOUNDATION_SIDE_M`.

    Returns ``((cx, cy), (dx, dy), orientation_deg)`` in m and degrees; a
    narrower side is widened with a :class:`UserWarning` (audit L5,
    A2G-06).
    """
    (cx, cy), (dx, dy), orientation_deg = footprint.oriented_bounding_rectangle()
    if min(dx, dy) < MIN_FOUNDATION_SIDE_M:
        label = "" if footprint.osm_id is None else f" {footprint.osm_id}"
        warnings.warn(
            f"Footprint{label} is {dx:.3g} m x {dy:.3g} m; its "
            f"foundation is widened to {MIN_FOUNDATION_SIDE_M:g} m "
            "(MIN_FOUNDATION_SIDE_M). Drop such outlines (walls, "
            "fences) from the footprints if they carry no foundation.",
            UserWarning,
            stacklevel=3,
        )
        dx = max(dx, MIN_FOUNDATION_SIDE_M)
        dy = max(dy, MIN_FOUNDATION_SIDE_M)
    return (cx, cy), (dx, dy), orientation_deg


def _rectangle_ring(
    centre: tuple[float, float], size: tuple[float, float], orientation_deg: float
) -> list[tuple[float, float]]:
    """Corners (m) of a ``size`` rectangle at ``centre``, rotated by the angle."""
    cx, cy = centre
    half_x, half_y = 0.5 * size[0], 0.5 * size[1]
    c = math.cos(math.radians(orientation_deg))
    s = math.sin(math.radians(orientation_deg))
    return [
        (cx + u * c - v * s, cy + u * s + v * c)
        for u, v in (
            (-half_x, -half_y),
            (half_x, -half_y),
            (half_x, half_y),
            (-half_x, half_y),
        )
    ]


# Kept for callers of the pre-2026.11 private names.
_MIN_POLY_EDGE_M = MIN_POLYGON_EDGE_M


def _clean_polygon(
    polygon_xy: "tuple[tuple[float, float], ...]",
    *,
    min_edge_m: float = MIN_POLYGON_EDGE_M,
) -> list[tuple[float, float]]:
    """Alias of :func:`groundfield.generators.materialise.clean_polygon`."""
    return clean_polygon(polygon_xy, min_edge_m=min_edge_m)


def _maybe_concrete_shell(
    spec: FoundationElectrodeSpec,
    *,
    dx: float,
    dy: float,
    rng: np.random.Generator,
    perimeter_m: Optional[float] = None,
) -> tuple[float, float, float]:
    """Resolve the shell fields of ``spec`` and evaluate the shell.

    Kept for callers of the pre-2026.11 private helper; see
    :func:`groundfield.generators.materialise.concrete_shell`. The shell
    spans every encased wire, as in the materialiser (user decision Q20):
    the perimeter ``2 (dx + dy)`` of a rectangle without cross-braces, the
    perimeter plus the braces, ``(n_y + 1) dx + (n_x + 1) dy``, of a
    ``style="mesh"`` foundation (audit D2B-09; it used the perimeter for
    meshes too). ``perimeter_m``, when given, is the encased length as is
    (the outline of a building).
    """
    if spec.concrete_rho_ohm_m is not None:
        spec = spec.model_copy(
            update={
                "concrete_rho_ohm_m": _to_float(spec.concrete_rho_ohm_m, rng),
                "concrete_thickness_m": _to_float(spec.concrete_thickness_m, rng),
            }
        )
    if perimeter_m is not None:
        length = float(perimeter_m)
    elif spec.style == "mesh":
        length = (spec.n_y + 1) * float(dx) + (spec.n_x + 1) * float(dy)
    else:
        length = 2.0 * (float(dx) + float(dy))
    return concrete_shell(spec, encased_length_m=length)


def _resolve_spec(spec: ElectrodeSpec, rng: np.random.Generator) -> ElectrodeSpec:
    """Draw every distribution of a present electrode spec.

    The draw order per kind is part of the reproducibility contract
    (seeded worlds of earlier releases stay identical): rod — depth,
    length; ring — depth, radius; strip — length, orientation, depth;
    foundation — depth, size (square foundations without polygon or
    ``size_xy_m``), concrete resistivity and thickness (only when a
    concrete shell is configured). Fields the realisation does not use
    (``size_m`` of a polygon or ``size_xy_m`` foundation, the shell
    thickness without a shell) are reset to their defaults when they
    hold a distribution, so the result is concrete.
    """
    update: dict[str, object] = {"presence_prob": 1.0}
    if isinstance(spec, RodElectrodeSpec):
        update["depth_m"] = _to_float(spec.depth_m, rng)
        update["length_m"] = _to_float(spec.length_m, rng)
    elif isinstance(spec, RingElectrodeSpec):
        update["depth_m"] = _to_float(spec.depth_m, rng)
        update["radius_m"] = _to_float(spec.radius_m, rng)
    elif isinstance(spec, StripElectrodeSpec):
        update["length_m"] = _to_float(spec.length_m, rng)
        update["orientation_deg"] = _to_float(spec.orientation_deg, rng)
        update["depth_m"] = _to_float(spec.depth_m, rng)
    elif isinstance(spec, FoundationElectrodeSpec):
        update["depth_m"] = _to_float(spec.depth_m, rng)
        if spec.polygon_xy_m is None and spec.size_xy_m is None:
            update["size_m"] = _to_float(spec.size_m, rng)
        elif isinstance(spec.size_m, Distribution):
            update["size_m"] = FoundationElectrodeSpec.model_fields["size_m"].default
        if spec.concrete_rho_ohm_m is not None:
            update["concrete_rho_ohm_m"] = _to_float(spec.concrete_rho_ohm_m, rng)
            update["concrete_thickness_m"] = _to_float(spec.concrete_thickness_m, rng)
        elif isinstance(spec.concrete_thickness_m, Distribution):
            update["concrete_thickness_m"] = FoundationElectrodeSpec.model_fields[
                "concrete_thickness_m"
            ].default
    else:  # pragma: no cover - the union is closed
        raise TypeError(f"Unknown electrode spec {type(spec).__name__}.")
    return spec.model_copy(update=update)


# ---------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------


class GroundingSystemSpec(GeneratorConfig):
    """A complete grounding installation.

    Attributes
    ----------
    electrodes
        Ordered list of electrode specs. Order matters only for the
        choice of anchor (first *present* electrode) and for the
        electrode names. All present electrodes are bonded into one
        cluster.
    bond_conductor_type
        Conductor type used to bond the electrodes together. Default
        ``"bare_copper"`` matches the typical bonding strap inside a
        substation pit; the bond is an ideal galvanic short either way.
    """

    electrodes: list[ElectrodeSpec] = Field(default_factory=list, min_length=0)
    bond_conductor_type: str = Field(
        default="bare_copper",
        description="Conductor type used to bond the electrodes together.",
    )

    # ------------------------------------------------------------------
    # Random decisions
    # ------------------------------------------------------------------

    def resolve_electrodes(self, rng: np.random.Generator) -> list[PlannedElectrode]:
        """Make every random decision of this grounding system.

        For each electrode in list order: one draw for a distributed
        ``presence_prob``, one Bernoulli draw when the probability is
        below one, and — for a present electrode — one draw per
        distribution field (see :func:`_resolve_spec` for the order).

        Parameters
        ----------
        rng
            Random generator.

        Returns
        -------
        list of PlannedElectrode
            The present electrodes with concrete values, each tagged with
            its index in :attr:`electrodes`.
        """
        out: list[PlannedElectrode] = []
        for i, spec in enumerate(self.electrodes):
            p = _to_float(spec.presence_prob, rng)
            if p < 1.0 and rng.random() >= p:
                continue
            out.append(PlannedElectrode(index=i, spec=_resolve_spec(spec, rng)))
        return out

    def plan_site(
        self,
        name: str,
        *,
        role: str,
        position_xy: tuple[float, float],
        rng: np.random.Generator,
        footprint: "Optional[BuildingFootprint]" = None,
        building_type: Optional[str] = None,
        outline: FoundationOutline = "spec",
    ) -> PlannedSite:
        """Resolve the system into a :class:`~groundfield.generators.plan.PlannedSite`.

        Parameters
        ----------
        name
            Site name (electrode-name prefix).
        role
            Site role, see :data:`groundfield.generators.plan.SiteRole`.
        position_xy
            Site reference point ``(x, y)`` in m.
        rng
            Random generator (see :meth:`resolve_electrodes`).
        footprint, building_type
            Recorded on the site for information.
        outline
            ``"spec"`` (default): the foundations keep the spec geometry.
            ``"ombr"`` / ``"polygon"``: after the random draws, the
            resolved foundations are shaped after ``footprint`` (see
            :meth:`with_footprint`). Drawing first keeps the random
            stream independent of the outline, so the same seed puts
            optional electrodes on the same sites whatever the outline
            (audit M2).

        Returns
        -------
        PlannedSite
            The concrete site.
        """
        electrodes = self.resolve_electrodes(rng)
        if footprint is not None and outline != "spec" and electrodes:
            shaped = GroundingSystemSpec(
                electrodes=[planned.spec for planned in electrodes],
                bond_conductor_type=self.bond_conductor_type,
            ).with_footprint(footprint, site_xy=position_xy, outline=outline)
            electrodes = [
                PlannedElectrode(index=planned.index, spec=spec)
                for planned, spec in zip(electrodes, shaped.electrodes, strict=True)
            ]
        return PlannedSite(
            name=name,
            role=role,  # type: ignore[arg-type]
            position_xy=(float(position_xy[0]), float(position_xy[1])),
            electrodes=electrodes,
            bond_conductor_type=self.bond_conductor_type,
            footprint=footprint,
            building_type=building_type,
        )

    # ------------------------------------------------------------------
    # Footprint rule (shared by every front-end)
    # ------------------------------------------------------------------

    def with_footprint(
        self,
        footprint: "BuildingFootprint",
        *,
        site_xy: Optional[tuple[float, float]] = None,
        outline: FoundationOutline = "ombr",
    ) -> "GroundingSystemSpec":
        r"""Return a copy whose foundation electrodes follow ``footprint``.

        Parameters
        ----------
        footprint
            Building outline in the world's local frame (m).
        site_xy
            Site position the spec will be built at, i.e. the origin of
            every ``offset_xy_m``. ``None`` uses the polygon centroid.
        outline
            ``"ombr"`` (default): every
            :class:`~groundfield.generators.electrode_specs.FoundationElectrodeSpec`
            takes the size :math:`(d_x, d_y)`, orientation
            :math:`\varphi` and centre :math:`(c_x, c_y)` of the
            outline's oriented minimum bounding rectangle; the centre
            enters as an extra offset :math:`(c_x, c_y) - \text{site}`
            on top of the spec's own ``offset_xy_m``. A side shorter than
            :data:`MIN_FOUNDATION_SIDE_M` is widened to it (with a
            :class:`UserWarning`). ``"polygon"``: every
            foundation follows the outline itself (``polygon_xy_m`` =
            exterior ring relative to ``site_xy``); only defined for
            ``style="ring"``. An outline whose OMBR is narrower than
            :data:`MIN_FOUNDATION_SIDE_M` is replaced by that OMBR widened
            to it, as a four-cornered outline (with a :class:`UserWarning`,
            as for ``"ombr"``); a polygon foundation along it would run
            along both faces of a wall, and an outline narrower than
            :data:`~groundfield.generators.materialise.MIN_POLYGON_EDGE_M`
            could not be built at all. ``"spec"``: return ``self``
            unchanged.

        Returns
        -------
        GroundingSystemSpec
            A new spec (``self`` when nothing changes). Presence
            probabilities, depths, styles, mesh counts, wire radii and
            concrete parameters are kept.

        Raises
        ------
        ValueError
            For ``outline="polygon"`` with a ``style="mesh"`` foundation
            (cross-braces are undefined on a general polygon) or an
            unknown ``outline``.
        ImportError
            ``outline="ombr"``, and ``outline="polygon"`` for an outline
            narrower than :data:`MIN_FOUNDATION_SIDE_M` on average, need
            :mod:`shapely` (``geo`` extra).
        """
        if outline not in ("spec", "ombr", "polygon"):
            raise ValueError(
                f"outline must be 'spec', 'ombr' or 'polygon', got {outline!r}."
            )
        if outline == "spec" or not any(
            isinstance(spec, FoundationElectrodeSpec) for spec in self.electrodes
        ):
            # Nothing to shape (no foundation): no OMBR, no warning.
            return self
        if site_xy is None:
            site_xy = footprint.centroid_xy_m()
        ring = [(float(x), float(y)) for x, y in footprint.polygon_xy_m]
        if outline == "ombr":
            (cx, cy), (dx, dy), orientation_deg = _widened_ombr(footprint)
            # Recentre on the OMBR centre (F25): the electrode sits at
            # ``site_xy + offset_xy_m``.
            recentre = (cx - site_xy[0], cy - site_xy[1])
        elif footprint.mean_width_m() < MIN_FOUNDATION_SIDE_M:
            # Only thin outlines can have an OMBR side below the minimum
            # (the mean width of a rectangle is below its short side), so
            # ordinary buildings need no shapely here.
            (cx, cy), (dx, dy), orientation_deg = (
                footprint.oriented_bounding_rectangle()
            )
            if min(dx, dy) < MIN_FOUNDATION_SIDE_M:
                ring = _rectangle_ring(*_widened_ombr(footprint))
        rewritten: list[ElectrodeSpec] = []
        changed = False
        for spec in self.electrodes:
            if not isinstance(spec, FoundationElectrodeSpec):
                rewritten.append(spec)
                continue
            if outline == "ombr":
                update: dict[str, object] = {
                    "size_xy_m": (dx, dy),
                    "orientation_deg": orientation_deg,
                    "offset_xy_m": (
                        spec.offset_xy_m[0] + recentre[0],
                        spec.offset_xy_m[1] + recentre[1],
                    ),
                }
            else:
                if spec.style != "ring":
                    raise ValueError(
                        "outline='polygon' needs style='ring' foundations — the "
                        "cross-braces of style='mesh' are undefined on a general "
                        "polygon. Use outline='ombr' or a ring-style spec."
                    )
                sx, sy = site_xy
                update = {"polygon_xy_m": tuple((x - sx, y - sy) for x, y in ring)}
            rewritten.append(spec.model_copy(update=update))
            changed = True
        if not changed:
            return self
        return self.model_copy(update={"electrodes": rewritten})

    # ------------------------------------------------------------------
    # One-call build
    # ------------------------------------------------------------------

    def build_at(
        self,
        world: World,
        site_xy: tuple[float, float],
        name_prefix: str,
        rng: np.random.Generator,
    ) -> Optional[str]:
        """Materialise the grounding system inside ``world``.

        Equivalent to :meth:`plan_site` followed by
        :func:`groundfield.generators.materialise.materialise_site`.

        Parameters
        ----------
        world
            The :class:`World` to populate.
        site_xy
            Centre position ``(x, y)`` of this site (substation centre,
            cabinet centre, building centre, …) in m.
        name_prefix
            Prefix of every created electrode name. The actual name is
            ``f"{name_prefix}_{kind}_{i}"`` with ``i`` the electrode's
            index in :attr:`electrodes`. Should be unique within the
            world.
        rng
            Random generator used for ``presence_prob`` and every
            distribution field.

        Returns
        -------
        str | None
            The name of the *anchor* electrode (the first present one).
            ``None`` if no electrode ended up being placed (every
            Bernoulli draw failed) — the caller may want to skip this
            site or raise.
        """
        from groundfield.generators.materialise import materialise_site

        site = PlannedSite(
            name=name_prefix,
            role="building",
            position_xy=(site_xy[0], site_xy[1]),
            electrodes=self.resolve_electrodes(rng),
            bond_conductor_type=self.bond_conductor_type,
        )
        return materialise_site(world, site)
