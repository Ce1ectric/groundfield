"""Building footprints as projected polygons.

A :class:`BuildingFootprint` is the data class that bridges the
external GIS world (currently :mod:`groundfield.geo.osm`; later
also image-recognition based ingest) and the
:mod:`groundfield.generators` stack. It carries a polygon already
projected to the local ENU frame in metres, together with the
attribute metadata that is useful for placing a foundation
electrode (number of levels, building use, original OSM tags).

The model is intentionally lightweight:

* Pure Pydantic v2; no :mod:`shapely` dependency at construction
  or serialisation time. Downstream consumers
  (``OsmBuildingPlacement``, the generator) lift the polygon into
  Shapely when they actually need geometric operations.
* Coordinates are metres in a *user-supplied* local ENU frame.
  The frame definition (origin lat / lon) is recorded once on the
  caller side, not on every footprint, to keep the JSON compact
  and avoid floating-point round-trip drift on the origin.
* The model is JSON-roundtrip safe and can therefore be embedded
  inside a :class:`generators.placement.OsmBuildingPlacement`
  field and persisted alongside a generator configuration.

Notes
-----
The polygon is stored as an explicit list of vertices rather than a
WKT / GeoJSON string. This keeps validation cheap (Pydantic checks
each tuple) and avoids pulling Shapely into the import path of
plain model deserialisation.

For an :class:`BuildingFootprint`'s exterior ring the convention
is **counter-clockwise** (CCW) so that the signed area is
positive; for any interior holes the convention is **clockwise**
(CW). :func:`ensure_orientation` enforces this. The convention
matches GeoJSON RFC 7946 §3.1.6 and is what Shapely expects on
construction.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Point2D = tuple[float, float]
"""Type alias for a 2-D point in the local ENU frame, ``(x_m, y_m)``."""

Ring = list[Point2D]
"""Type alias for a closed ring; the first and last vertex are equal
(or :func:`ensure_orientation` will close it)."""


__all__ = [
    "Point2D",
    "Ring",
    "BuildingFootprint",
    "signed_area",
    "ensure_orientation",
]


# ---------------------------------------------------------------------
# Polygon helpers (no shapely dependency)
# ---------------------------------------------------------------------


def signed_area(ring: Ring) -> float:
    r"""Signed area of a polygon ring via the shoelace formula.

    Parameters
    ----------
    ring
        Vertex list. The ring may be open (last vertex != first); the
        shoelace formula closes it implicitly.

    Returns
    -------
    float
        Signed area in m². Positive for counter-clockwise rings,
        negative for clockwise rings, zero for degenerate rings.

    Notes
    -----
    Implements

    .. math::

       A \;=\; \tfrac{1}{2}\sum_{i=0}^{n-1}\bigl(x_i\,y_{i+1}\;-\;
                                                x_{i+1}\,y_i\bigr)

    with indices taken modulo *n*. The orientation convention used
    here (CCW positive) matches the right-hand rule with a
    z-axis pointing *out of the ground* in the local ENU frame.
    """
    n = len(ring)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def ensure_orientation(ring: Ring, *, ccw: bool = True) -> Ring:
    """Return ``ring`` reordered so its signed area has the requested sign.

    Parameters
    ----------
    ring
        Input ring (open or closed).
    ccw
        If ``True`` (default), the returned ring is counter-clockwise
        (positive signed area); if ``False``, clockwise.

    Returns
    -------
    list[Point2D]
        A *new* list of vertices in the requested orientation. The
        ring is returned open (no duplicate final vertex) so that
        the caller can close it explicitly if needed.
    """
    cleaned: Ring = list(ring)
    if cleaned and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    a = signed_area(cleaned)
    if (a > 0) != ccw:
        cleaned.reverse()
    return cleaned


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------


class BuildingFootprint(BaseModel):
    """A single building's footprint, projected to the local ENU frame.

    Parameters
    ----------
    polygon_xy_m
        Vertices of the *exterior* ring in metres, CCW. May be
        passed open or closed; on validation the ring is normalised
        to CCW and stored open (no duplicate final vertex).
    holes_xy_m
        Interior rings (holes) in metres, CW. Each hole is
        normalised on validation. Empty list means no holes (the
        typical residential case).
    levels
        ``building:levels`` tag if present (often missing in
        OSM). ``None`` denotes a missing value; downstream
        consumers default to a heuristic such as
        ``levels_default=2`` for residential.
    building_use
        Value of the ``building=*`` tag (e.g. ``"residential"``,
        ``"apartments"``, ``"industrial"``, ``"yes"``). Kept as the
        raw OSM string so the consumer can map it to a
        :class:`generators.building.BuildingTypeSpec` at will.
    osm_id
        Original OSM way / relation id. Useful for tracing back to
        the source feature for debugging.
    osm_tags
        Untouched OSM key/value tag dictionary. Lets consumers
        extract additional information (e.g. ``addr:housenumber``,
        ``building:material``) without going back to the source.

    Notes
    -----
    The instance is **immutable** (``frozen=True``) so that a
    :class:`generators.placement.OsmBuildingPlacement` can rely on
    the polygon not changing while the generator iterates over
    sites.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    polygon_xy_m: Ring = Field(..., description="Exterior ring (CCW), metres.")
    holes_xy_m: list[Ring] = Field(
        default_factory=list,
        description="Interior holes (each CW), metres.",
    )
    levels: Optional[float] = Field(
        default=None,
        description="``building:levels`` tag if present, else ``None``.",
    )
    building_use: Optional[str] = Field(
        default=None,
        description="Raw ``building=*`` tag value.",
    )
    osm_id: Optional[int] = Field(
        default=None,
        description="Original OSM way / relation id.",
    )
    osm_tags: dict[str, str] = Field(
        default_factory=dict,
        description="Untouched OSM tag dict.",
    )

    # -----------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------

    @field_validator("polygon_xy_m")
    @classmethod
    def _check_exterior(cls, value: Ring) -> Ring:
        if len(value) < 3:
            raise ValueError(
                "BuildingFootprint.polygon_xy_m: a polygon needs at "
                f"least 3 vertices, got {len(value)}."
            )
        return ensure_orientation(value, ccw=True)

    @field_validator("holes_xy_m")
    @classmethod
    def _check_holes(cls, value: list[Ring]) -> list[Ring]:
        out: list[Ring] = []
        for i, ring in enumerate(value):
            if len(ring) < 3:
                raise ValueError(
                    "BuildingFootprint.holes_xy_m: hole "
                    f"{i} has only {len(ring)} vertices."
                )
            out.append(ensure_orientation(ring, ccw=False))
        return out

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------

    def area_m2(self) -> float:
        """Signed-area-based polygon area in m² (holes subtracted).

        Returns
        -------
        float
            ``area(exterior) - sum(area(holes))``, always non-negative
            because the rings have been normalised to CCW / CW on
            validation.
        """
        outer = signed_area(self.polygon_xy_m)
        inner = sum(-signed_area(h) for h in self.holes_xy_m)
        return outer - inner

    def centroid_xy_m(self) -> Point2D:
        r"""Geometric centroid of the *exterior* ring in metres.

        Notes
        -----
        Holes are ignored — for the residential / commercial
        building shapes we care about, the moment shift caused by
        a hole is at most a few centimetres and irrelevant to the
        soil-spreading calculation. Implements

        .. math::

           C_x = \frac{1}{6A}\sum_{i=0}^{n-1}
                 (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i),

        analogously for :math:`C_y`.
        """
        ring = self.polygon_xy_m
        a = signed_area(ring)
        if a == 0.0:
            # Degenerate ring; fall back to the vertex mean.
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        cx = cy = 0.0
        n = len(ring)
        for i in range(n):
            x0, y0 = ring[i]
            x1, y1 = ring[(i + 1) % n]
            cross = x0 * y1 - x1 * y0
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        denom = 6.0 * a
        return (cx / denom, cy / denom)

    # -----------------------------------------------------------------
    # Bounding rectangles
    # -----------------------------------------------------------------

    def axis_aligned_bounding_rectangle(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Axis-aligned bounding rectangle of the exterior ring.

        Returns
        -------
        tuple
            ``((cx, cy), (dx, dy))`` — centre of the bounding box and
            its side lengths in metres. ``cx``, ``cy`` are the means
            of the min/max coordinates; ``dx``, ``dy`` are the
            absolute extents.

        Notes
        -----
        Pure-Python implementation; no :mod:`shapely` dependency.
        Used as a fall-back when the optional ``geo`` extra is not
        installed; for a fully rotated foundation electrode use
        :meth:`oriented_bounding_rectangle` instead.
        """
        xs = [p[0] for p in self.polygon_xy_m]
        ys = [p[1] for p in self.polygon_xy_m]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return (
            (0.5 * (x_min + x_max), 0.5 * (y_min + y_max)),
            (x_max - x_min, y_max - y_min),
        )

    def oriented_bounding_rectangle(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        r"""Oriented minimum bounding rectangle (OMBR) of the exterior
        ring.

        Returns
        -------
        tuple
            ``((cx, cy), (dx, dy), orientation_deg)`` — centre,
            side lengths (``dx`` along the local long axis, ``dy``
            perpendicular), and rotation of the long axis with
            respect to the world ``+x`` direction in degrees.
            ``orientation_deg`` lies in ``[-90, 90]``.

        Notes
        -----
        Uses :func:`shapely.geometry.Polygon.minimum_rotated_rectangle`,
        which implements the rotating-calipers algorithm. Requires
        the optional ``geo`` extra; raises :class:`ImportError`
        with the install hint otherwise.

        Physical interpretation
        -----------------------
        The OMBR is the canonical projection of an arbitrary
        building outline onto a *Streifenfundament* (DIN 18014):
        the foundation ring follows the OMBR's perimeter, so its
        side lengths fix the conductor length that contributes to
        the galvanic spreading admittance, and the orientation
        fixes the angle relative to the PEN trunk for the
        inductive-coupling calculation. For a typical residential
        L-shape the OMBR captures the dominant edge alignment
        within a few degrees; full polygonal Strip chains are
        deferred to Phase B (ADR-0011).
        """
        try:
            from shapely.geometry import Polygon  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised in tests
            from groundfield.geo.projection import GEO_IMPORT_HINT

            raise ImportError(GEO_IMPORT_HINT) from exc
        from shapely.geometry import Polygon as _Polygon

        poly = _Polygon(self.polygon_xy_m, holes=self.holes_xy_m)
        if not poly.is_valid:
            # Buffer(0) is the Shapely idiom for cleaning up
            # self-intersections that occasionally show up in OSM data.
            poly = poly.buffer(0)
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        # ``minimum_rotated_rectangle`` always returns five coords
        # (closed ring). The four sides connect coords[i] -> coords[i+1].
        # Pick the longer side direction as the OMBR long axis.
        import math as _math

        x0, y0 = coords[0]
        x1, y1 = coords[1]
        x2, y2 = coords[2]
        len_a = _math.hypot(x1 - x0, y1 - y0)
        len_b = _math.hypot(x2 - x1, y2 - y1)
        if len_a >= len_b:
            long_dx, long_dy = x1 - x0, y1 - y0
            dx, dy = len_a, len_b
        else:
            long_dx, long_dy = x2 - x1, y2 - y1
            dx, dy = len_b, len_a
        angle = _math.degrees(_math.atan2(long_dy, long_dx))
        # Map onto [-90, 90] so the orientation is canonical
        # (a 180° rotation flips the rectangle onto itself).
        while angle > 90.0:
            angle -= 180.0
        while angle <= -90.0:
            angle += 180.0
        cx = sum(c[0] for c in coords[:4]) / 4.0
        cy = sum(c[1] for c in coords[:4]) / 4.0
        return ((cx, cy), (dx, dy), angle)
