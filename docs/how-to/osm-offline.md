# Work with OpenStreetMap offline

This guide builds and uses an on-disk cache of OpenStreetMap building
footprints, so that a study runs without network access and reads the same
map every time. Read it before you build a network from real buildings, in
CI, behind a firewall or on the road.

`groundfield.geo` reads building footprints from OpenStreetMap through the
Overpass API. Every answer is cached on disk, and a cached answer is used
without any network access. That makes studies reproducible: the map
changes, your cache does not.

**Prerequisites:** the `geo` extra (requests, shapely, pyproj; see
[Install the optional extras](install-extras.md)). Only step 1 needs the
network.

## How the cache works

`query_buildings` (and everything built on it: `query_and_project`,
`LvNetworkLayout.from_osm`, the OSM placement of `TnNetworkGenerator`)
builds the Overpass query text from centre, radius and timeout, and looks
for the file `<sha256 of the query>.json` in `cache_dir` (default
`~/.cache/groundfield/osm`). If the file exists and is a valid Overpass
answer, it is used; otherwise the query is sent and the answer is written
there atomically. `force_refresh=True` sends the query again.

## 1. Prepare a cache while you are online

Run the query once with an explicit cache directory and keep the
directory with your study (commit it, or copy it to the offline machine):

<!-- skip-doctest: needs the Overpass API (network) -->
```python
from pathlib import Path

import groundfield as gf

footprints, projector = gf.query_and_project(51.9136, 10.8432, 150.0, cache_dir=Path("osm_cache"))
```

Every later call with the same centre, radius and timeout reads
`osm_cache/` and never touches the network.

## 2. Or put an Overpass answer you got elsewhere into the cache

An answer saved from another tool (for example the Overpass web
front-end, with the same query) can be put into the cache under the name
`groundfield` expects. The query text comes from the public
`groundfield.geo.osm.build_query`. Here a tiny two-building answer is
written by hand to keep the example self-contained:

```python
import hashlib
import json
from pathlib import Path

import groundfield as gf
from groundfield.geo import osm

LAT, LON, RADIUS = 51.9136, 10.8432, 150.0
payload = {
    "version": 0.6,
    "elements": [
        {
            "type": "way",
            "id": 1,
            "tags": {"building": "house"},
            "geometry": [
                {"lat": 51.91370, "lon": 10.84320},
                {"lat": 51.91370, "lon": 10.84335},
                {"lat": 51.91379, "lon": 10.84335},
                {"lat": 51.91379, "lon": 10.84320},
                {"lat": 51.91370, "lon": 10.84320},
            ],
        },
        {
            "type": "way",
            "id": 2,
            "tags": {"building": "garage"},
            "geometry": [
                {"lat": 51.91350, "lon": 10.84300},
                {"lat": 51.91350, "lon": 10.84308},
                {"lat": 51.91353, "lon": 10.84308},
                {"lat": 51.91353, "lon": 10.84300},
                {"lat": 51.91350, "lon": 10.84300},
            ],
        },
    ],
}
cache = Path("osm_cache")
cache.mkdir(exist_ok=True)
query = osm.build_query(LAT, LON, RADIUS, timeout_s=osm.DEFAULT_TIMEOUT_S)
(cache / f"{hashlib.sha256(query.encode('utf-8')).hexdigest()}.json").write_text(json.dumps(payload))

footprints, projector = gf.query_and_project(LAT, LON, RADIUS, cache_dir=cache)
for fp in footprints:
    print(fp.osm_id, fp.building_use, f"{fp.area_m2():.1f} m2")
```

<!-- doctest-check
assert [fp.osm_id for fp in footprints] == [1, 2]
assert 90.0 < footprints[0].area_m2() < 120.0  # about 10 m x 10 m
-->

**What you get:** the two footprints of the cached answer, with their
OSM ids, building use and area, read without any network access. The
cache key contains the timeout (`timeout_s`) and, for other endpoints than
the default, the endpoint URL: use the same values offline as online.

## 3. Filter the buildings

Every `building=*` way is a footprint by default, garages and sheds
included. Buildings without electrical service usually have no
foundation earth electrode connected to the PEN; drop them with
`exclude_building_tags` (the documented set
`gf.geo.NON_SERVICE_BUILDING_TAGS` contains garage, shed, carport, hut,
ruins and similar) and very small outlines with `min_area_m2`:

```python
from groundfield.geo import NON_SERVICE_BUILDING_TAGS

kept, _ = gf.query_and_project(LAT, LON, RADIUS, cache_dir=cache, exclude_building_tags=NON_SERVICE_BUILDING_TAGS)
print([fp.building_use for fp in kept])
```

<!-- doctest-check
assert [fp.building_use for fp in kept] == ["house"]
-->

## 4. Convert coordinates

Footprints are projected into local east/north metres around the query
centre. The returned `Projector` converts in both directions, so you can
place a substation or a probe from GPS coordinates:

```python
x, y = projector.to_xy_m(51.9137, 10.8433)
lat, lon = projector.to_lat_lon(x, y)
print(f"x = {x:.2f} m, y = {y:.2f} m -> back: {lat:.6f}, {lon:.6f}")
```

<!-- doctest-check
assert abs(lat - 51.9137) < 1e-9 and abs(lon - 10.8433) < 1e-9
-->

`LvNetworkLayout.from_osm(..., cache_dir=...)` accepts the same cache and
offers `lat_lon_to_xy` / `xy_to_lat_lon`; tutorial
[T06](../tutorials/T06_tn_network_synthetic_and_osm.ipynb) builds a whole
network from a cached answer.

## Limits and safety

Requests are bounded: the radius is at most 10 km, the timeout an
integer of at most 900 s, the response is capped in size, and malformed
answers or Overpass error remarks raise `OverpassError` and are never
cached. A cache file that cannot be read is fetched again with a warning.

## See also

- Tutorial [T06](../tutorials/T06_tn_network_synthetic_and_osm.ipynb): a
  whole TN network from a cached answer.
- [ADR-0011](../adr/0011-osm-building-footprints.md): the footprint data
  model, the projection, the Overpass query and its cache.
- API: [Geo / OSM](../reference/api/geo.md) and the
  [generators](../reference/api/generators.md).
