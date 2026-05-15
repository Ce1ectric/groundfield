# Geo / OSM

The ``geo`` subpackage is the **bridge between the GIS world and the
generator stack**. It ingests building outlines from OpenStreetMap
(via the Overpass API, ODbL) — and in the future from rasterised
maps via image recognition — projects them into a local
right-handed ENU frame in metres, and feeds them into
[`TnNetworkGenerator`][groundfield.generators.tn_network.TnNetworkGenerator]
through the [`OsmBuildingPlacement`][groundfield.geo.placement.OsmBuildingPlacement]
member of the [`PlacementSpec`][groundfield.generators.placement.PlacementSpec]
discriminated union.

## Mathematical / physical context

A foundation electrode according to DIN 18014 is a closed
conductor inside the building's *Streifenfundament*. Within the
quasi-static regime ($f \le 1\,\mathrm{kHz}$), the relevant field
problem is the same Laplace equation that drives the rest of
``groundfield`` (see [Generators](generators.md) for the full
PDE statement); the role of the geo layer is purely topological:
**which buildings exist, where are they, how are their foundation
rings oriented?**

Three mechanisms are sensitive to the *geometry*, not only to the
*count* of houses:

1. **Galvanic dissipation per building.** A foundation
   electrode's spreading admittance scales with the conductor
   length actually buried — for a closed perimeter ring that is
   the polygon's *perimeter*. The oriented minimum bounding
   rectangle (OMBR, see below) conserves the perimeter to within
   a few percent on the residential / commercial shapes we care
   about, so the OMBR-based reduction captures the dominant
   contribution to the cluster impedance.
2. **Inductive coupling to PEN and measurement leads.** The
   *orientation* of the foundation ring matters for the mutual
   inductance to a parallel PEN trunk along the street. Below
   1 kHz this is in the Carson / Sommerfeld regime
   (ADRs 0005 / 0006); the orientation enters the Neumann
   integral via the direction cosine on the wire elements.
3. **Spatial layout of the houses.** Real Reihenhauszeilen do
   not sit on a Manhattan grid. Cluster impedance of the PEN
   backbone is not separable from layout — the Manhattan grid
   used as a reference case in [ADR-0009](../adr/0009-world-generators.md)
   is the *upper-bound symmetric* idealisation; the OSM-driven
   layout is the *realism* case.

The OMBR projection is the **Phase-A** reduction defined in
[ADR-0011](../adr/0011-osm-building-footprints.md). It maps an
arbitrary building outline to a rectangle by

$$
\mathrm{OMBR}(P) \;=\; \mathop{\mathrm{arg\,min}}_{R \in \mathcal{R}}
  \big\{ \mathrm{area}(R) : P \subseteq R,\;
       R \text{ is a rectangle} \big\},
$$

implemented via the rotating-calipers algorithm in
[`shapely.geometry.Polygon.minimum_rotated_rectangle`][]. For an
L-shape this contains the polygon (so the perimeter approximation
is an upper bound) and aligns with the dominant edge; for a
clean rectangle it is the rectangle itself. Phase B (deferred)
replaces the rectangle by a polygonal Strip chain along
``polygon.buffer(-inset).exterior`` — this is required only when
non-rectangular foundation interiors become electrically relevant
(typically Gewerbehallen with complex outlines).

## Validity envelope

* **Frequency** — $f \le 1\,\mathrm{kHz}$ (same as the rest of
  the generator stack).
* **Projection** — azimuthal equidistant (``+proj=aeqd``) centred
  on a user-supplied origin. Residual distortion $< 10^{-5}$
  within ~5 km of the origin, well below soil-resistivity
  uncertainty and OSM polygon quantisation. **Single tangent
  plane**, no UTM-zone bookkeeping.
* **Origin** — **never inferred from data**. Two runs over the
  same area at different dates produce identical metric
  coordinates regardless of which features the underlying query
  returns. This is the reproducibility guarantee the AP1
  Monte-Carlo phase relies on.
* **Stochasticity** — the only *geometric* stochastic axis
  preserved across the OMBR override is
  [`FoundationElectrodeSpec.presence_prob`][groundfield.generators.electrode_specs.FoundationElectrodeSpec].
  Bernoulli per building, fully reproducible under a seeded RNG.
  All geometric distributions (``size_m``, ``size_xy_m``,
  ``orientation_deg``) are *overridden* by the polygon and
  therefore deterministic per footprint. The *non-geometric*
  fields that govern the foundation's electrical environment —
  in particular the concrete-shell parameters
  ``concrete_rho_ohm_m`` (moisture state) and
  ``concrete_thickness_m`` (typical 30–200 mm) from
  [ADR-0012](../adr/0012-foundation-concrete-encasement.md) —
  continue to accept :class:`AnyDistribution`, so an OSM-driven
  Monte-Carlo run still varies independently across the moisture
  classes per realisation.

## Installation

The geo layer is gated behind an optional dependency group named
``geo`` (see [`pyproject.toml`][pyproject_toml]):

```bash
pip install groundfield[geo]
# or
poetry install --extras geo
```

The optional dependencies are :mod:`requests` (HTTP to the
Overpass endpoint), :mod:`pyproj` (the projection), and
:mod:`shapely` (the OMBR and polygon hygiene). The
[`BuildingFootprint`][groundfield.geo.footprint.BuildingFootprint]
data class itself has **zero optional dependencies** and is
importable on a core install — only the active functions
(querying Overpass, building a [`Projector`][groundfield.geo.projection.Projector],
computing the OMBR) need the extra. The
:class:`ImportError` raised on first use carries the install
hint verbatim.

[pyproject_toml]: https://github.com/Ce1ectric/groundfield/blob/main/pyproject.toml

## Subpackage layout

```text
src/groundfield/geo/
├── footprint.py    # BuildingFootprint Pydantic model
├── projection.py   # Projector (WGS84 -> local ENU)
├── osm.py          # Overpass-API client + on-disk cache
└── placement.py    # OsmBuildingPlacement (PlacementSpec member)
```

## API reference

::: groundfield.geo.footprint.BuildingFootprint
    options:
      show_root_heading: true
      show_source: false

::: groundfield.geo.projection.Projector
    options:
      show_root_heading: true
      show_source: false

::: groundfield.geo.osm.build_query
    options:
      show_root_heading: true
      show_source: false

::: groundfield.geo.osm.query_buildings
    options:
      show_root_heading: true
      show_source: false

::: groundfield.geo.osm.parse_overpass_payload
    options:
      show_root_heading: true
      show_source: false

::: groundfield.geo.osm.query_and_project
    options:
      show_root_heading: true
      show_source: false

::: groundfield.geo.osm.OverpassError
    options:
      show_root_heading: true
      show_source: false

::: groundfield.geo.placement.OsmBuildingPlacement
    options:
      show_root_heading: true
      show_source: false

## Worked example

A full end-to-end demo lives in
[`notebooks/32_osm_footprints.ipynb`](https://github.com/Ce1ectric/groundfield/blob/main/notebooks/32_osm_footprints.ipynb).
It synthesises six rotated rectangles (so the notebook needs no
internet), feeds them into ``TnNetworkGenerator``, compares the
resulting cluster impedance against a Manhattan-grid reference
with the same building count, and finishes with an optional
*live Overpass query* (cached locally, gracefully skipped when
the ``geo`` extra is absent or no network is available).
