"""OpenStreetMap building ingest via the Overpass API.

Public API
----------
The two top-level entry points used by callers are:

* :func:`query_buildings` — given an origin ``(lat0, lon0)`` and a
  radius, return the raw Overpass-JSON response (cached on disk).
  Lower-level, allows custom post-processing.
* :func:`query_and_project` — convenience wrapper that runs the
  query, projects every building polygon into a local ENU frame
  via :class:`groundfield.geo.projection.Projector`, and returns
  a list of :class:`groundfield.geo.footprint.BuildingFootprint`
  instances ready to feed into a
  :class:`groundfield.geo.placement.OsmBuildingPlacement`.

Caching
-------
Every call to :func:`query_buildings` is keyed on a SHA-256 hash of
the *exact* Overpass-QL query string. The hash is used as the file
name inside ``cache_dir`` (default
``~/.cache/groundfield/osm/<sha256>.json``). After the first call
the on-disk JSON is read directly; no network round-trip is
performed. The cache stores the raw Overpass payload (not the
projected footprints) so the user can swap projection origins
without re-downloading.

Compliance
----------
A ``User-Agent`` of the form ``groundfield/<version>
(+https://github.com/Ce1ectric/groundfield)`` is sent with every
request per the Overpass API usage policy. The default endpoint is
the canonical ``https://overpass-api.de/api/interpreter`` mirror;
users with their own instance can pass ``endpoint=…``.

A single retry with exponential backoff is performed on HTTP
``429 Too Many Requests`` or ``504 Gateway Timeout``; everything
else raises immediately.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from groundfield.geo.footprint import BuildingFootprint, Ring
from groundfield.geo.projection import Projector

if TYPE_CHECKING:  # pragma: no cover - typing only
    import requests as _requests_module


def _groundfield_version() -> str:
    """Best-effort lookup of the installed ``groundfield`` version.

    Uses :mod:`importlib.metadata` so importing :mod:`groundfield.geo.osm`
    does *not* trigger the eager load of the full :mod:`groundfield`
    top-level package (which pulls in scipy, matplotlib, etc.). Falls
    back to ``"0.0.0+unknown"`` on a source checkout where the package
    has not been installed via Poetry / pip.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("groundfield")
        except PackageNotFoundError:
            return "0.0.0+unknown"
    except Exception:  # pragma: no cover - defensive
        return "0.0.0+unknown"

__all__ = [
    "DEFAULT_ENDPOINT",
    "OverpassError",
    "build_query",
    "query_buildings",
    "parse_overpass_payload",
    "query_and_project",
    "default_cache_dir",
]


DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
"""Default Overpass endpoint. Override with ``endpoint=…``."""

DEFAULT_TIMEOUT_S = 30
"""Per-request timeout passed both to Overpass (via ``[timeout:…]``)
and to ``requests`` as a network-side timeout."""

def _user_agent() -> str:
    """User-Agent string sent with every Overpass request.

    Built lazily so a source-only checkout (no installed dist) still
    works during tests; production installs reflect the real version.
    """
    return (
        f"groundfield/{_groundfield_version()} "
        "(+https://github.com/Ce1ectric/groundfield)"
    )

_GEO_IMPORT_HINT = (
    "groundfield.geo requires the optional 'geo' extra. Install with "
    "'pip install groundfield[geo]' or "
    "'poetry install --extras geo'."
)


class OverpassError(RuntimeError):
    """Raised on a non-success Overpass response.

    Attributes
    ----------
    status_code
        HTTP status code if available, else ``None``.
    body
        First 1 kB of the response body for diagnostic purposes.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


# ---------------------------------------------------------------------
# Lazy import for ``requests``
# ---------------------------------------------------------------------


def _import_requests() -> "_requests_module":
    try:
        import requests  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(_GEO_IMPORT_HINT) from exc
    return requests


# ---------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------


def default_cache_dir() -> Path:
    """Return the default on-disk Overpass cache directory.

    Honours ``XDG_CACHE_HOME`` when set; otherwise falls back to
    ``~/.cache/groundfield/osm`` on Linux/macOS and the equivalent
    on Windows via :class:`pathlib.Path.home`.

    The directory is *not* created here; the caller (e.g.
    :func:`query_buildings`) creates it on first write.
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "groundfield" / "osm"
    return Path.home() / ".cache" / "groundfield" / "osm"


# ---------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------


def build_query(
    lat0_deg: float,
    lon0_deg: float,
    radius_m: float,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> str:
    """Construct the Overpass-QL query string for a buildings-in-radius
    lookup.

    Parameters
    ----------
    lat0_deg, lon0_deg
        Centre of the lookup in WGS84 decimal degrees.
    radius_m
        Search radius in metres. Must be positive.
    timeout_s
        Query timeout passed to Overpass via ``[timeout:…]``.

    Returns
    -------
    str
        A deterministic Overpass-QL query. The string is
        byte-identical across runs given the same inputs, so its
        SHA-256 hash is a stable cache key.
    """
    if radius_m <= 0:
        raise ValueError(f"radius_m must be > 0, got {radius_m}.")
    # Float formatting kept fixed so the query string is stable
    # across platforms / locales — six decimals on lat/lon are
    # roughly 0.1 m at the equator; one decimal on radius_m is
    # 0.1 m, well below typical OSM polygon quantisation.
    lat_s = f"{lat0_deg:.6f}"
    lon_s = f"{lon0_deg:.6f}"
    r_s = f"{radius_m:.1f}"
    return (
        f"[out:json][timeout:{timeout_s}];\n"
        f"(\n"
        f'  way["building"](around:{r_s},{lat_s},{lon_s});\n'
        f'  relation["building"](around:{r_s},{lat_s},{lon_s});\n'
        f");\n"
        f"out body geom tags;"
    )


def _hash_query(query: str) -> str:
    """SHA-256 of the query string, used as the cache filename stem."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------


def _post_overpass(
    query: str,
    *,
    endpoint: str,
    timeout_s: int,
    max_retries: int = 1,
    _sleep: Any = time.sleep,
) -> dict[str, Any]:
    """POST ``query`` to ``endpoint`` and return the decoded JSON.

    Retries once on ``429`` / ``504`` with exponential backoff. All
    other non-2xx responses raise :class:`OverpassError`.

    The ``_sleep`` hook lets the test suite replace
    :func:`time.sleep` without touching the global module state.
    """
    requests = _import_requests()
    headers = {
        "User-Agent": _user_agent(),
        "Accept": "application/json",
    }
    for attempt in range(max_retries + 1):
        response = requests.post(
            endpoint,
            data={"data": query},
            headers=headers,
            timeout=timeout_s + 5,  # client side a little more lenient
        )
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise OverpassError(
                    "Overpass returned a 200 but the body is not "
                    "valid JSON.",
                    status_code=response.status_code,
                    body=response.text[:1024],
                ) from exc
        if response.status_code in (429, 504) and attempt < max_retries:
            _sleep(2 ** attempt)
            continue
        raise OverpassError(
            f"Overpass returned HTTP {response.status_code}.",
            status_code=response.status_code,
            body=response.text[:1024],
        )
    # Unreachable: the loop either returns or raises.
    raise OverpassError(  # pragma: no cover - defensive
        "Overpass request exhausted retries without a definitive outcome."
    )


# ---------------------------------------------------------------------
# High-level query with on-disk cache
# ---------------------------------------------------------------------


def query_buildings(
    lat0_deg: float,
    lon0_deg: float,
    radius_m: float,
    *,
    cache_dir: Optional[Path] = None,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    force_refresh: bool = False,
    _post: Any = None,
) -> dict[str, Any]:
    """Fetch (or load from cache) the raw Overpass response.

    Parameters
    ----------
    lat0_deg, lon0_deg
        Centre of the buildings-in-radius lookup, WGS84 degrees.
    radius_m
        Search radius in metres.
    cache_dir
        Directory in which the response is cached. Defaults to
        :func:`default_cache_dir`. Created on first write.
    endpoint
        Overpass endpoint URL. Defaults to the canonical mirror.
    timeout_s
        Per-query timeout passed to Overpass.
    force_refresh
        If ``True``, ignore any cached payload and re-query the
        endpoint, overwriting the cache file. Useful when the user
        knows the upstream data changed.
    _post
        Internal hook for tests: a callable with the same signature
        as :func:`_post_overpass`. Production calls use the default.

    Returns
    -------
    dict[str, Any]
        The decoded Overpass JSON payload. The shape is
        ``{"version": …, "generator": …, "elements": [...]}``;
        only ``elements`` is required downstream.

    Notes
    -----
    The cache filename is ``<sha256(query)>.json``; ergo two distinct
    queries that happen to round-trip to the same string share a
    cache file, and two semantically identical queries with
    different float formatting do *not* — see :func:`build_query`
    for the fixed formatting convention that guarantees stability.
    """
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
    query = build_query(
        lat0_deg, lon0_deg, radius_m, timeout_s=timeout_s
    )
    digest = _hash_query(query)
    path = Path(cache_dir) / f"{digest}.json"
    if path.exists() and not force_refresh:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    post = _post if _post is not None else _post_overpass
    payload = post(
        query,
        endpoint=endpoint,
        timeout_s=timeout_s,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return payload


# ---------------------------------------------------------------------
# Overpass payload -> BuildingFootprint
# ---------------------------------------------------------------------


def _element_to_lat_lon_ring(
    element: dict[str, Any],
) -> Optional[list[tuple[float, float]]]:
    """Extract the geometry of a single ``way`` element as a
    ``(lat, lon)`` ring, or ``None`` if the element has no inline
    geometry.

    Overpass with ``out body geom tags`` annotates each ``way`` with
    a ``geometry`` list of ``{lat, lon}`` dicts in vertex order.
    """
    geom = element.get("geometry")
    if not geom:
        return None
    ring: list[tuple[float, float]] = []
    for pt in geom:
        try:
            ring.append((float(pt["lat"]), float(pt["lon"])))
        except (KeyError, TypeError, ValueError):
            return None
    return ring


def parse_overpass_payload(
    payload: dict[str, Any],
    projector: Projector,
    *,
    min_area_m2: float = 0.0,
) -> list[BuildingFootprint]:
    """Convert an Overpass JSON payload into projected
    :class:`BuildingFootprint` instances.

    Parameters
    ----------
    payload
        Overpass JSON as returned by :func:`query_buildings`.
    projector
        Projection from WGS84 to the local ENU frame. The same
        projector should be reused across all consumers of one
        payload to keep the metric coordinates consistent.
    min_area_m2
        Skip features whose footprint area is strictly below this
        threshold. ``0.0`` (the default) keeps every parseable
        polygon; the placement layer may apply its own (higher)
        threshold via :class:`OsmBuildingPlacement.min_area_m2`.

    Returns
    -------
    list[BuildingFootprint]
        Order matches the Overpass element order, which Overpass
        keeps deterministic for the same query.

    Notes
    -----
    *Multipolygon relations* (``type=multipolygon``) are handled by
    walking the relation's ``members`` and treating each ``outer``
    way as an independent footprint with the ``inner`` ways of the
    *same* relation attached as holes. Relations without an
    ``outer`` member are skipped.
    """
    elements = payload.get("elements", []) or []
    by_id: dict[int, dict[str, Any]] = {
        e["id"]: e for e in elements if e.get("type") == "way" and "id" in e
    }
    out: list[BuildingFootprint] = []

    for el in elements:
        kind = el.get("type")
        if kind == "way":
            # Skip ways that belong to a multipolygon relation —
            # they would otherwise be emitted twice. Overpass does
            # not mark this directly, so for v1 we accept the
            # duplication; downstream consumers can de-duplicate
            # by ``osm_id`` if needed. The duplicate cost is small
            # because residential ways in OSM are usually standalone.
            ring_ll = _element_to_lat_lon_ring(el)
            if ring_ll is None or len(ring_ll) < 3:
                continue
            try:
                ring_xy: Ring = projector.ring_to_xy_m(ring_ll)
                fp = BuildingFootprint(
                    polygon_xy_m=ring_xy,
                    holes_xy_m=[],
                    levels=_levels_from_tags(el.get("tags") or {}),
                    building_use=(el.get("tags") or {}).get("building"),
                    osm_id=int(el["id"]) if "id" in el else None,
                    osm_tags=dict(el.get("tags") or {}),
                )
            except ValueError:
                continue
            if fp.area_m2() < min_area_m2:
                continue
            out.append(fp)
        elif kind == "relation":
            tags = el.get("tags") or {}
            if tags.get("type") != "multipolygon":
                continue
            members = el.get("members") or []
            outers: list[list[tuple[float, float]]] = []
            inners: list[list[tuple[float, float]]] = []
            for m in members:
                if m.get("type") != "way":
                    continue
                role = m.get("role")
                ring_ll = _element_to_lat_lon_ring(m)
                if ring_ll is None:
                    # Try the referenced way from the same payload.
                    ref = by_id.get(m.get("ref"))
                    if ref is not None:
                        ring_ll = _element_to_lat_lon_ring(ref)
                if ring_ll is None or len(ring_ll) < 3:
                    continue
                if role == "outer":
                    outers.append(ring_ll)
                elif role == "inner":
                    inners.append(ring_ll)
            if not outers:
                continue
            # v1: emit each ``outer`` ring as a separate footprint
            # carrying the *full* list of inner holes. Hole-to-outer
            # assignment by winding-rule is deferred (rare in
            # residential OSM data; multipolygon buildings are
            # almost always Gewerbe-/Industriebauten with at most
            # one outer).
            for outer_ll in outers:
                try:
                    outer_xy = projector.ring_to_xy_m(outer_ll)
                    holes_xy = [projector.ring_to_xy_m(h) for h in inners]
                    fp = BuildingFootprint(
                        polygon_xy_m=outer_xy,
                        holes_xy_m=holes_xy,
                        levels=_levels_from_tags(tags),
                        building_use=tags.get("building"),
                        osm_id=int(el["id"]) if "id" in el else None,
                        osm_tags=dict(tags),
                    )
                except ValueError:
                    continue
                if fp.area_m2() < min_area_m2:
                    continue
                out.append(fp)
        # Nodes are silently ignored: a building cannot be a node.
    return out


def _levels_from_tags(tags: dict[str, Any]) -> Optional[float]:
    """Best-effort parser for the ``building:levels`` tag.

    OSM convention is an integer but mappers often submit decimals
    (e.g. ``"1.5"`` for an attic). Returns ``None`` on any parse
    failure so the consumer can fall back to a default heuristic.
    """
    raw = tags.get("building:levels")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------
# Convenience: query + parse + project in one call
# ---------------------------------------------------------------------


def query_and_project(
    lat0_deg: float,
    lon0_deg: float,
    radius_m: float,
    *,
    projector: Optional[Projector] = None,
    cache_dir: Optional[Path] = None,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    min_area_m2: float = 0.0,
    force_refresh: bool = False,
) -> tuple[list[BuildingFootprint], Projector]:
    """Run :func:`query_buildings` and :func:`parse_overpass_payload`
    end-to-end.

    Parameters
    ----------
    lat0_deg, lon0_deg, radius_m
        Buildings-in-radius parameters.
    projector
        If ``None``, a new :class:`Projector` centred at
        ``(lat0_deg, lon0_deg)`` is created. Pass an existing
        projector to keep the metric frame consistent across
        multiple calls (useful when a study spans several
        overlapping radii).
    cache_dir, endpoint, timeout_s, force_refresh
        Forwarded to :func:`query_buildings`.
    min_area_m2
        Skip features below this footprint area. Defaults to
        ``0.0`` (keep everything; the placement layer applies its
        own threshold).

    Returns
    -------
    tuple[list[BuildingFootprint], Projector]
        Projected footprints and the projector used. The projector
        is returned so the caller can re-use it for further
        coordinate transforms (e.g. plotting a substation marker
        at a known lat/lon).
    """
    payload = query_buildings(
        lat0_deg,
        lon0_deg,
        radius_m,
        cache_dir=cache_dir,
        endpoint=endpoint,
        timeout_s=timeout_s,
        force_refresh=force_refresh,
    )
    if projector is None:
        projector = Projector(lat0_deg, lon0_deg)
    footprints = parse_overpass_payload(
        payload, projector, min_area_m2=min_area_m2
    )
    return footprints, projector
