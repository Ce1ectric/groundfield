"""WGS84 -> local ENU projection for OSM-derived geometries.

A single tangent-plane projection (azimuthal equidistant centred on a
user-supplied origin) is used. For the radii relevant to a single
TN-Ortsnetz (typically a few hundred metres, at most a few
kilometres) the residual distortion is well below the soil-resistivity
uncertainty and the OSM polygon quantisation, so we deliberately
avoid the UTM-zone bookkeeping recommended for larger areas. The
choice is justified in ADR-0011.

Lazy import of :mod:`pyproj`: ``import projection`` is free as long
as the user does not call :class:`Projector`. The first call raises
:class:`ImportError` with the installation hint when the optional
``geo`` dependency group is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyproj

__all__ = [
    "Projector",
]


_GEO_IMPORT_HINT = (
    "groundfield.geo requires the optional 'geo' extra. Install with "
    "'pip install groundfield[geo]' or "
    "'poetry install --extras geo'."
)


def _import_pyproj() -> "pyproj":
    """Lazy import of :mod:`pyproj` with a friendly error message."""
    try:
        import pyproj  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(_GEO_IMPORT_HINT) from exc
    return pyproj


class Projector:
    r"""WGS84 (longitude, latitude) <-> local ENU (x_m, y_m) projection.

    Parameters
    ----------
    lat0_deg, lon0_deg
        Latitude and longitude of the projection origin in decimal
        degrees, WGS84. The origin is the point that maps to
        ``(x_m, y_m) = (0, 0)``. The user picks it explicitly — it
        is **never inferred** from the data (see ADR-0011) so two
        runs over the same area at different dates produce
        identical metric coordinates regardless of which features
        appear in the underlying query.
    ellps
        Ellipsoid name accepted by :mod:`pyproj`. Defaults to
        ``"WGS84"``; passed straight through.

    Notes
    -----
    The projection is azimuthal equidistant
    (``+proj=aeqd``) centred on ``(lat0, lon0)`` with the WGS84
    ellipsoid. For distances :math:`r \lesssim 5 \, \mathrm{km}`
    around the origin the residual distortion is below
    :math:`10^{-5}` (a few centimetres on a kilometre baseline),
    which is two orders of magnitude below typical OSM polygon
    quantisation. Conformality is **not** preserved exactly — for
    our use case (placing electrodes and measuring distances
    between them) equal distance to the origin is the relevant
    invariant, not angle preservation.

    The projection is *not* thread-safe: each :class:`Projector`
    holds two :class:`pyproj.Transformer` instances internally;
    instantiate one per worker if you parallelise.
    """

    def __init__(
        self,
        lat0_deg: float,
        lon0_deg: float,
        *,
        ellps: str = "WGS84",
    ) -> None:
        if not -90.0 <= lat0_deg <= 90.0:
            raise ValueError(
                f"Projector: lat0_deg must be in [-90, 90], got {lat0_deg}."
            )
        if not -180.0 <= lon0_deg <= 180.0:
            raise ValueError(
                f"Projector: lon0_deg must be in [-180, 180], got {lon0_deg}."
            )
        pyproj = _import_pyproj()
        self.lat0_deg = float(lat0_deg)
        self.lon0_deg = float(lon0_deg)
        self.ellps = ellps
        proj4 = (
            f"+proj=aeqd +lat_0={self.lat0_deg} +lon_0={self.lon0_deg} "
            f"+ellps={self.ellps} +units=m +no_defs"
        )
        wgs84 = "EPSG:4326"
        # ``always_xy=True`` keeps (lon, lat) tuple order, the more
        # common convention in GIS code; the public API of this class
        # still exposes (lat, lon) in argument names.
        self._fwd = pyproj.Transformer.from_crs(
            wgs84, proj4, always_xy=True
        )
        self._inv = pyproj.Transformer.from_crs(
            proj4, wgs84, always_xy=True
        )

    # -----------------------------------------------------------------
    # Forward / inverse on single points
    # -----------------------------------------------------------------

    def to_xy_m(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        """Project a single point WGS84 -> local ENU.

        Parameters
        ----------
        lat_deg, lon_deg
            WGS84 coordinates of the point in decimal degrees.

        Returns
        -------
        tuple[float, float]
            ``(x_m, y_m)`` in the local ENU frame. ``x`` points
            east, ``y`` points north, ``z`` (not returned) points up.
        """
        x, y = self._fwd.transform(lon_deg, lat_deg)
        return float(x), float(y)

    def to_lat_lon(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Inverse of :meth:`to_xy_m`: local ENU -> WGS84.

        Parameters
        ----------
        x_m, y_m
            Local ENU coordinates in metres.

        Returns
        -------
        tuple[float, float]
            ``(lat_deg, lon_deg)`` in decimal degrees.
        """
        lon, lat = self._inv.transform(x_m, y_m)
        return float(lat), float(lon)

    # -----------------------------------------------------------------
    # Vectorised helpers for polygon rings
    # -----------------------------------------------------------------

    def ring_to_xy_m(
        self, lat_lon: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Vectorised forward projection of a list of ``(lat, lon)``.

        Parameters
        ----------
        lat_lon
            List of WGS84 ``(lat_deg, lon_deg)`` tuples.

        Returns
        -------
        list[tuple[float, float]]
            Same length, each tuple in local ENU metres.
        """
        if not lat_lon:
            return []
        lats = [p[0] for p in lat_lon]
        lons = [p[1] for p in lat_lon]
        xs, ys = self._fwd.transform(lons, lats)
        # pyproj returns numpy arrays for vector input, but plain
        # lists/tuples for scalar input. Normalise to a list of tuples
        # so the downstream Pydantic model accepts the result without
        # numpy-vs-tuple ambiguity.
        return [(float(xs[i]), float(ys[i])) for i in range(len(lats))]

    # -----------------------------------------------------------------
    # Identity helpers
    # -----------------------------------------------------------------

    def origin(self) -> tuple[float, float]:
        """Return the origin as ``(lat_deg, lon_deg)``."""
        return (self.lat0_deg, self.lon0_deg)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Projector(lat0_deg={self.lat0_deg!r}, "
            f"lon0_deg={self.lon0_deg!r}, ellps={self.ellps!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Projector):
            return NotImplemented
        return (
            self.lat0_deg == other.lat0_deg
            and self.lon0_deg == other.lon0_deg
            and self.ellps == other.ellps
        )

    def __hash__(self) -> int:
        return hash((self.lat0_deg, self.lon0_deg, self.ellps))


# Re-export the install hint so :mod:`groundfield.geo.osm` and
# :mod:`groundfield.geo.placement` raise the same message.
GEO_IMPORT_HINT: Optional[str] = _GEO_IMPORT_HINT
