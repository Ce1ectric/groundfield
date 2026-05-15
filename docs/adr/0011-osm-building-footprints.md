# ADR-0011: OSM building footprints and footprint-driven foundation electrodes

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-05-14 |
| **Deciders** | Project maintainers |
| **Scope** | `groundfield.geo` (new subpackage); `generators.placement`,
              `generators.electrode_specs`, `generators.tn_network`
              (additive extensions); optional Poetry dependency group
              `geo` (`requests`, `shapely`, `pyproj`) |

## Context

ADR-0009 introduced the generator stack and explicitly deferred a
*follow-up generator that will read from open building-map data*
as the next fidelity step beyond `ManhattanGridPlacement`. The
existing `placement.py` module docstring repeats that statement.
This ADR delivers exactly that step for `TnNetworkGenerator`.

The motivating AP1 question is whether the synthetic Manhattan grid
ever masks effects that a real street layout would expose. Three
mechanisms are sensitive to the geometry, not only to the count
of houses:

1. **Galvanic dissipation per building.** A foundation electrode's
   spreading admittance scales with the *perimeter* of the
   foundation, not with a generic side length. Real residential
   footprints are L-shaped or rectangular with non-square aspect
   ratios; the bounding-rectangle approximation already captures
   the perimeter and the principal axes within $\sim 10\,\%$,
   which is below the AP1 soil-resistivity uncertainty.
2. **Inductive coupling to PEN and measurement leads.** The
   *orientation* of a foundation ring matters for the mutual
   inductance to a parallel PEN trunk along the street, which is
   one of the AP1 effects (Carson coupling below 1 kHz).
3. **Spatial distribution of houses.** Bunched-up Reihenhäuser at
   one end of a Ortsnetz behave differently from a uniform grid
   even at the same total count — the cluster impedance of the
   PEN backbone is not separable from layout.

OpenStreetMap (OSM, ODbL) is the only open data source that
covers all three with metric accuracy and is queryable
programmatically through the **Overpass API**. Coverage in
Germany for `building=*` is essentially complete in built-up areas;
`building:levels` is sparser but populated for newer surveys.

The user also requested **two implementation variants**:

* **Variant 1 (Overpass-driven):** lat/lon + radius → polygons.
  Adopted as the primary path. Reproducible, scriptable, exact.
* **Variant 2 (image recognition):** rasterised maps via CV/ML.
  Deferred. The principal obstacle is metric scale recovery, not
  segmentation. When implemented, it slots in behind the same
  `BuildingFootprint` data class and is otherwise invisible to the
  generator.

## Decision

### Subpackage layout

A new subpackage `groundfield.geo` is added, intentionally **next
to** `groundfield.geometry/`, not inside it. `geometry/` holds
electrode *primitives* (`RodElectrode`, `StripElectrode`, …); `geo/`
holds *georeferencing and external-data ingest*. Keeping the two
distinct prevents conflating "physical conductor shape" with
"GIS-derived input metadata".

```
src/groundfield/geo/
├── __init__.py
├── osm.py          # Overpass query, cache, parser
├── projection.py   # WGS84 → local ENU via pyproj
├── footprint.py    # BuildingFootprint Pydantic model
└── placement.py    # OsmBuildingPlacement(PlacementSpec)
```

`osm.py`, `projection.py`, and the `placement.OsmBuildingPlacement`
class **import** `shapely`, `pyproj`, and `requests`. These three
ship in a new optional Poetry dependency group:

```toml
[tool.poetry.group.geo]
optional = true
[tool.poetry.group.geo.dependencies]
requests = "^2.32"
shapely  = "^2.0"
pyproj   = "^3.6"
```

The top-level re-export shim in `groundfield/__init__.py` lazy-loads
`geo.*` and raises an instructive `ImportError` ("install with
`poetry install --with geo`") on first access — matching the
pattern already established by the optional `groundinsight`
dependency.

### Data model

```python
class BuildingFootprint(BaseModel):
    """A single OSM building, projected to local ENU metres."""
    polygon_xy_m: list[tuple[float, float]]     # closed ring, CCW
    holes_xy_m:   list[list[tuple[float, float]]] = []
    levels:       float | None = None           # building:levels (may be NaN)
    building_use: str | None = None             # building=residential|...
    osm_id:       int | None = None
    osm_tags:     dict[str, str] = {}
```

Coordinates are metric in the local ENU frame defined by the user
(see *Projection*). The model is Pydantic v2, JSON-roundtripable,
and frozen — every cache hit deserialises to a bit-identical
instance.

### Projection

WGS84 → local ENU via `pyproj.Proj` with a user-supplied origin
`(lat0_deg, lon0_deg)`. The origin is **never inferred from data**
(rejected during the *Coordinate-System* clarification) so two
runs against the same query at different dates produce identical
metric coordinates regardless of which buildings appear in the
result.

For radii up to a few kilometres at mid-latitudes the residual
projection distortion is $< 10\,\text{ppm}$, two orders of
magnitude below soil-resistivity uncertainty and the OSM polygon
quantisation. This justifies a single tangent-plane projection
for the whole query — no UTM zone bookkeeping, no zone-edge
discontinuities. The exact projection string used is
`+proj=aeqd +lat_0=… +lon_0=… +ellps=WGS84` (azimuthal
equidistant). The choice and its rationale are documented in
`projection.py`.

### Overpass query and caching

`osm.query_buildings(lat0, lon0, radius_m, *, cache_dir=…)` builds
a deterministic Overpass-QL query of the form

```
[out:json][timeout:30];
(
  way["building"](around:{r},{lat},{lon});
  relation["building"](around:{r},{lat},{lon});
);
out body geom tags;
```

and POSTs it to the configured Overpass endpoint. The response is
written verbatim to `{cache_dir}/{sha256_of_query}.json` and is
read from disk on every subsequent call — Overpass is hit at most
once per `(lat0, lon0, radius)` tuple. The cache directory
defaults to `~/.cache/groundfield/osm/` (XDG-aware via
`appdirs`-style logic, implemented locally to avoid one more
dependency). Disk format is the raw Overpass JSON, not the
projected `BuildingFootprint`s, so users can swap projections
without re-downloading.

A user-agent header `groundfield/<__version__>
(https://github.com/…)` is set per Overpass usage policy, and a
single retry with exponential backoff is performed on `429` /
`504`.

The cache and the network call are *only* in `osm.py`; the
generator layer never sees `requests`. This keeps the test
surface small (mock at the `osm.query_buildings` boundary).

### Generator integration — `OsmBuildingPlacement`

A new placement variant is added to the existing
`generators.placement.PlacementSpec` discriminated union:

```python
class OsmBuildingPlacement(BaseModel):
    kind: Literal["osm"] = "osm"
    footprints: list[BuildingFootprint]      # already projected
    selection: Literal["all", "first_n"] = "first_n"

    def generate(self, n, rng) -> list[tuple[float, float]]: ...
    def footprint_at(self, i: int) -> BuildingFootprint | None: ...
```

`generate(n, rng)` returns the polygon centroids in metric ENU,
exactly mirroring the interface of `ManhattanGridPlacement` and
`ExplicitPlacement` so no caller needs to special-case it.
`footprint_at(i)` is the *additive* hook the generator uses to
ask "is there a polygon I can use for building $i$?" — present
only on this variant, looked up via `hasattr` in the generator.

`OsmBuildingPlacement` does **not** call Overpass itself. It is
constructed from an explicit `footprints` list. The standard
workflow is:

```python
footprints = geo.query_and_project(
    lat0=…, lon0=…, radius_m=…,
    cache_dir=…, projection_origin=(lat0, lon0),
)
placement = OsmBuildingPlacement(footprints=footprints)
```

Separating the network step from the placement keeps the
generator config JSON-serialisable and bit-exactly reproducible
without an internet round-trip.

### Footprint-driven foundation electrodes — Phase A

`FoundationElectrodeSpec.size_xy_m` is currently `Optional`; the
existing semantics ("when set, overrides `size_m`") are preserved.
`TnNetworkGenerator.build` is extended at exactly one point:
before sampling a building's `GroundingSystemSpec`, if the
placement is an `OsmBuildingPlacement` and the per-building
spec contains a `FoundationElectrodeSpec`, the generator derives
three fields from the polygon and *overrides* the spec values
for this realisation:

* **`size_xy_m`** — sides of the oriented minimum bounding
  rectangle (Shapely's `minimum_rotated_rectangle`).
* **`offset_xy_m`** — polygon centroid minus the site (x, y);
  zero by construction in v1 (site = centroid).
* **footprint orientation** — angle of the OMBR principal axis.
  This is **not** currently a field on `FoundationElectrodeSpec`;
  Phase A adds it as `orientation_deg: float | None = None`,
  defaulting to `None` (= +x as today). When set, the
  underlying `GridMeshElectrode` is rotated about its centre by
  this angle. This is also useful outside the OSM path (e.g.
  hand-placed houses on a street that doesn't run E–W).

`presence_prob` is *unchanged*. This is the only stochastic axis
the user requested — Bernoulli per building, defaulting to
$p_\text{install} = 1.0$. Per-realisation diversity comes from
the `presence_prob` draw, not from sampling geometric
distributions (the polygon fixes those).

Buildings whose polygon has fewer than four vertices, or whose
OMBR area is below a configurable threshold (default 16 m²,
matches a typical Gartenhaus), are skipped with a debug log
entry to keep `TnNetworkGenerator` robust to OSM mapper noise.

### Footprint-driven foundation electrodes — Phase B (deferred)

Phase A approximates every building by its bounding rectangle.
Real L-shaped houses are covered correctly in their perimeter (by
construction OMBR ≥ polygon area; perimeter typically matches to
within 5 %) but the *interior* of an L-shape is not modelled.
Phase B introduces a new electrode primitive
`PolygonalStripElectrode(loop_xy_m, depth_m, wire_radius_m)` and
a sibling spec `PolygonalFoundationElectrodeSpec(polygon, depth,
inset_m, segment_max_m)` that materialises a closed Strip-chain
along `polygon.buffer(-inset_m).exterior` with each long segment
subdivided to `segment_max_m`. Phase B becomes essential once
non-rectangular Reihenhauszeilen or Gewerbehallen with complex
footprints enter AP1.

Phase B has no impact on the cache, projection, or placement
layers and can ship as an additive change.

## Validation programme

`tests/test_geo_*.py`:

1. **Overpass query construction** — given `(lat0, lon0, radius_m)`
   the QL string is byte-identical across runs, and the cache
   filename is the sha256 of that string.
2. **Cache round-trip** — `query_buildings` against a mocked
   Overpass response writes the cache file on first call and does
   **not** open a socket on the second call (verified with a
   `requests`-monkeypatch sentinel).
3. **Projection** — round-trip `(lat, lon) → ENU → (lat, lon)`
   for ten points within a 5 km radius returns an absolute error
   below $10^{-6}$ degrees ($\approx 0.1\,\text{m}$).
4. **Polygon parsing** — a synthetic Overpass payload with one
   simple way, one way with an inner ring, and one multipolygon
   relation parses to the expected `BuildingFootprint` count and
   ring orientation (CCW exteriors, CW holes).
5. **`OsmBuildingPlacement.generate`** — for $n$ ≤
   `len(footprints)`, returns the first $n$ polygon centroids in
   declared order. For $n$ > available, raises with a clear
   message.
6. **OMBR override** — for an L-shaped polygon, the override
   produces a `FoundationElectrodeSpec` with `size_xy_m` matching
   the OMBR sides to within $10^{-9}$ m and `orientation_deg`
   matching the OMBR axis to within $10^{-9}$ deg.
7. **Reproducibility under `presence_prob`** — fixed RNG seed
   produces a bit-identical sequence of "house has / hasn't a
   foundation electrode" decisions across two `TnNetworkGenerator.build`
   calls on the same config.
8. **End-to-end smoke** — a small (~20 houses) cached fixture
   under `tests/data/osm_sample.json` builds a `World` that
   solves successfully with `image_2layer` and produces a
   non-trivial PEN current distribution. Cluster impedance is
   compared to a `ManhattanGridPlacement` reference at the same
   house count and soil — values are expected to *differ* (the
   point of the new path) but to remain within an order of
   magnitude.
9. **Optional-dependency gate** — importing
   `groundfield.geo.placement` without the `geo` group installed
   raises `ImportError` with the install hint and does **not**
   break any non-`geo` test.

## Consequences

### Positive

- Realistic AP1 layouts become a one-liner: pick a TN-Ortsnetz on
  the map, run the generator, the houses sit where they sit and
  their foundation rings face the way the real buildings face.
- The Manhattan grid stays as the bit-reproducible *reference*
  case; the OSM path is the *realism* case. Cross-validating
  the two is itself an AP1 result.
- The cache + explicit-origin projection make a Monte-Carlo run
  over OSM-driven worlds bit-exactly replayable without internet
  access after the first run.
- The data class `BuildingFootprint` and the deferred Variant 2
  (image recognition) share an interface — when CV-based polygon
  recovery becomes interesting, it slots in behind the same
  `OsmBuildingPlacement(footprints=…)` constructor.

### Negative

- Three new optional dependencies (`requests`, `shapely`,
  `pyproj`). All ODbL/MIT-friendly and widely available; opt-in
  via the Poetry group keeps the core install lean.
- OSM data quality varies. `building:levels` is sparse outside
  city centres; the generator must remain robust when the tag
  is missing (handled by defaulting to `None`).
- Overpass rate limits (typically 10 000 queries/day; one query
  per Ortsnetz is well within this). The on-disk cache makes the
  effective rate one query per (origin, radius) tuple per
  developer lifetime.

### Neutral

- Existing `ManhattanGridPlacement` / `ExplicitPlacement` users
  are unaffected; the new placement is purely additive.
- `FoundationElectrodeSpec` gains one optional field
  (`orientation_deg`), defaulting to `None`. JSON round-trips of
  configs written before this ADR continue to load unchanged.
- ADR-0007 cross-layer-electrode preconditions cover the
  foundation depth range used here (0.5–1.5 m); no new
  numerical-kernel work is required.

## References

- **OpenStreetMap contributors** (2026). `building=*` tag wiki.
  https://wiki.openstreetmap.org/wiki/Key:building
- **OpenStreetMap Foundation.** *Open Database Licence (ODbL)
  v1.0.* https://opendatacommons.org/licenses/odbl/1-0/
- **Overpass API** project documentation.
  https://wiki.openstreetmap.org/wiki/Overpass_API
- **DIN 18014:2023-04.** *Fundamenterder — Planung, Ausführung
  und Dokumentation.* DIN, Berlin. Defines the geometric and
  electrical requirements that the `FoundationElectrodeSpec`
  (perimeter strip at 0.5–1 m depth, ring or mesh) reflects.
- **ADR-0009** — generator architecture; this ADR is the
  deferred OSM follow-up named there.
- **ADR-0007** — cross-layer electrodes; ensures a foundation
  ring at 0.8 m straddling a thin top layer is solved correctly.
