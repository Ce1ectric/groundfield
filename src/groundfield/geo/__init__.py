"""Georeferenced data ingest for :mod:`groundfield`.

This subpackage bridges the external GIS world (OpenStreetMap via the
Overpass API, in the future also image-based footprint recognition)
and the :mod:`groundfield.generators` stack. The end product is a
list of :class:`BuildingFootprint` instances in a local ENU frame,
which a :class:`OsmBuildingPlacement` then feeds into a
:class:`TnNetworkGenerator` so that synthetic worlds inherit real
street layouts and real building outlines for the foundation
electrodes.

Optional dependencies
---------------------
:mod:`groundfield.geo.osm` and :mod:`groundfield.geo.projection`
import :mod:`requests`, :mod:`pyproj`, and :mod:`shapely`. These
ship in the ``geo`` Poetry / pip extra and are *not* part of the
core ``groundfield`` install. Calling any of the public APIs
without the extra raises a clear :class:`ImportError` with the
install hint::

    pip install groundfield[geo]
    poetry install --extras geo

The Pydantic data class :class:`BuildingFootprint` does **not**
depend on any of the optional libraries; it can be imported and
serialised on a core install. Only the *active* operations
(querying Overpass, projecting WGS84 to ENU, computing oriented
bounding rectangles) need the extra.

Design rationale
----------------
See ``docs/adr/0011-osm-building-footprints.md`` for the full
decision record (subpackage layout, projection choice, caching
strategy, integration with the generator stack, deferred
Phase B for non-rectangular foundations).
"""

from __future__ import annotations

from groundfield.geo.footprint import (
    BuildingFootprint,
    Point2D,
    Ring,
    ensure_orientation,
    signed_area,
)
from groundfield.geo.osm import (
    DEFAULT_ENDPOINT,
    OverpassError,
    build_query,
    default_cache_dir,
    parse_overpass_payload,
    query_and_project,
    query_buildings,
)
from groundfield.geo.placement import OsmBuildingPlacement
from groundfield.geo.projection import Projector

__all__ = [
    # Data model
    "BuildingFootprint",
    "Point2D",
    "Ring",
    "signed_area",
    "ensure_orientation",
    # Projection
    "Projector",
    # Overpass / OSM
    "DEFAULT_ENDPOINT",
    "OverpassError",
    "build_query",
    "default_cache_dir",
    "parse_overpass_payload",
    "query_buildings",
    "query_and_project",
    # Placement
    "OsmBuildingPlacement",
]
