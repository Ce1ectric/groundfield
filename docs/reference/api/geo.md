# Geo / OSM

The `geo` package brings real building layouts into the generators: it
queries building outlines from OpenStreetMap through the Overpass API
(data © OpenStreetMap contributors, ODbL), projects them from WGS84 into
a local east-north frame in metres, and hands them to the generators as
`BuildingFootprint` objects (through `OsmBuildingPlacement` in
`TnNetworkGenerator`, or `LvNetworkLayout.from_osm`). Design and
reasoning: ADR-0011.

The active functions need the optional `geo` extra
(`pip install "groundfield[geo]"`: `requests`, `pyproj`, `shapely`);
without it they raise an `ImportError` with the install command. The
data class `BuildingFootprint` has no optional dependency.

## Mathematical and physical context

The geo layer is topological: it decides which buildings exist, where
they are and how their foundation earths are oriented. Three effects
depend on that geometry rather than on the number of houses: the
spreading resistance of a foundation earth grows with its buried
length (the perimeter of the outline), the inductive coupling of a
foundation to a PEN cable along the street depends on its orientation,
and the mutual resistance between the earths depends on the real
spacing of the houses, which a regular grid does not reproduce.

**Projection.** Azimuthal equidistant projection (`+proj=aeqd`) centred
on a user-supplied origin: one tangent plane, distortion below
$10^{-5}$ within about 5 km of the origin, no UTM zone bookkeeping. The
origin is never inferred from the data, so two queries of the same area
at different dates give identical coordinates.

**Foundation outline.** A foundation earth (DIN 18014) is a closed
conductor in the strip foundation. The generators reduce an outline
either to its oriented minimum bounding rectangle (OMBR, the default:
smallest rectangle containing the polygon, rotating calipers) or follow
the polygon itself (`foundation_outline="polygon"` of the layout
builder). Outlines smaller than 1 m are widened to
`MIN_FOUNDATION_SIDE_M` with a warning.

**Requests.** Overpass queries are bounded (radius, timeout, response
size), validated before they are cached, and retried with back-off; a
malformed answer raises `OverpassError` and is not cached. Every
`building=*` value is a footprint by default; `exclude_building_tags`
(for example `NON_SERVICE_BUILDING_TAGS`: garages, sheds, carports, …)
drops buildings without a service connection.

## Example (offline)

```python
import math

import groundfield as gf

# a 12 m x 9 m house turned by 30 degrees, in local metres (east, north)
angle = math.radians(30.0)
corners = [(0.0, 0.0), (12.0, 0.0), (12.0, 9.0), (0.0, 9.0)]
outline = [
    (math.cos(angle) * x - math.sin(angle) * y, math.sin(angle) * x + math.cos(angle) * y)
    for x, y in corners
]
house = gf.BuildingFootprint(polygon_xy_m=outline, osm_id=1)

centre, size, orientation_deg = house.oriented_bounding_rectangle()   # needs the geo extra
assert abs(house.area_m2() - 108.0) < 1e-9
assert abs(size[0] - 12.0) < 1e-9 and abs(size[1] - 9.0) < 1e-9
assert abs(orientation_deg - 30.0) < 1e-9
```

The tutorial on TN networks (T06) builds a complete network from a
stored Overpass answer, without network access; the how-to guide on
OSM offline use shows how to cache and replay queries.

## API reference

::: groundfield.geo
