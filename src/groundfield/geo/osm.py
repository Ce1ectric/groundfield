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

Retries are performed on HTTP ``429 Too Many Requests``, ``503`` and
``504 Gateway Timeout``, honouring a ``Retry-After`` header and capped
at :data:`MAX_RETRY_DELAY_S` per wait; the count is controlled by the
``max_retries`` keyword on :func:`query_buildings` and
:func:`query_and_project` (default ``1``, at most :data:`MAX_RETRIES`).
Everything else raises immediately.

Safety
------
Requests are bounded and validated (review finding 10.05):

* coordinates must be finite WGS84 values, the radius finite and at most
  :data:`MAX_RADIUS_M`, ``timeout_s`` an integer in
  ``[1, MAX_TIMEOUT_S]`` (it is interpolated into the query text);
* endpoints must use ``https`` (plain ``http`` only for a local
  instance on ``localhost``);
* the response is streamed and capped at ``max_response_bytes``
  (default :data:`DEFAULT_MAX_RESPONSE_BYTES`);
* an HTTP-200 payload whose ``remark`` reports an Overpass runtime
  error, or that is not a JSON object with an ``elements`` list, raises
  :class:`OverpassError` and is not cached;
* cache files are written atomically (temporary file + rename); an
  unreadable cache file is re-fetched with a warning;
* the cache key of a non-default endpoint includes the endpoint (the
  default endpoint keeps the historic key, so existing caches stay
  valid).

Building tags
-------------
Every element with a ``building=*`` tag is a footprint unless the
caller excludes tag values via ``exclude_building_tags`` (see
:data:`NON_SERVICE_BUILDING_TAGS`); by default nothing is excluded.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import numbers
import os
import time
import uuid
import warnings
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlsplit

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
    "DEFAULT_MAX_RESPONSE_BYTES",
    "MAX_RADIUS_M",
    "MAX_RETRIES",
    "MAX_RETRY_DELAY_S",
    "MAX_TIMEOUT_S",
    "NON_SERVICE_BUILDING_TAGS",
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

MAX_TIMEOUT_S = 900
"""Largest accepted ``timeout_s`` in s."""

MAX_RADIUS_M = 10_000.0
"""Largest accepted search radius in m (one LV / MV network area)."""

DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
"""Default cap on the size of an Overpass response (64 MiB)."""

MAX_RETRIES = 5
"""Largest accepted ``max_retries``."""

MAX_RETRY_DELAY_S = 60.0
"""Longest wait in s between two retries (also caps ``Retry-After``)."""

_RETRY_STATUSES = (429, 503, 504)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

NON_SERVICE_BUILDING_TAGS: frozenset[str] = frozenset(
    {
        "carport",
        "container",
        "garage",
        "garages",
        "greenhouse",
        "hut",
        "no",
        "roof",
        "ruins",
        "shed",
        "shelter",
        "tent",
    }
)
"""``building=*`` values that usually have neither an LV service
connection nor a foundation earth electrode (garages, sheds, roofs,
ruins, ``building=no``, ...). Pass it as ``exclude_building_tags`` to
drop such footprints; it is not applied by default."""


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

    Raises
    ------
    ValueError
        For non-finite or out-of-range coordinates (latitude in
        ``[-90, 90]``, longitude in ``[-180, 180]``), a radius outside
        ``(0, MAX_RADIUS_M]`` or a ``timeout_s`` outside
        ``[1, MAX_TIMEOUT_S]``.
    TypeError
        If ``timeout_s`` is not an integer (it is interpolated into the
        query text, so anything else could inject Overpass QL).
    """
    lat0_deg, lon0_deg, radius_m = _validate_area(lat0_deg, lon0_deg, radius_m)
    timeout_s = _validate_timeout(timeout_s)
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


def _validate_area(
    lat0_deg: float, lon0_deg: float, radius_m: float
) -> tuple[float, float, float]:
    """Check the search area; return it as floats."""
    try:
        lat, lon, radius = float(lat0_deg), float(lon0_deg), float(radius_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("lat0_deg, lon0_deg and radius_m must be numbers.") from exc
    if not (math.isfinite(lat) and -90.0 <= lat <= 90.0):
        raise ValueError(
            f"lat0_deg must be a finite latitude in [-90, 90], got {lat0_deg}."
        )
    if not (math.isfinite(lon) and -180.0 <= lon <= 180.0):
        raise ValueError(
            f"lon0_deg must be a finite longitude in [-180, 180], got {lon0_deg}."
        )
    if not (math.isfinite(radius) and 0.0 < radius <= MAX_RADIUS_M):
        raise ValueError(
            f"radius_m must be > 0 and <= {MAX_RADIUS_M:g} m, got {radius_m}."
        )
    return lat, lon, radius


def _validate_timeout(timeout_s: object) -> int:
    """Check ``timeout_s`` (an integer that ends up in the query text)."""
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, numbers.Integral):
        raise TypeError(f"timeout_s must be an integer, got {timeout_s!r}.")
    timeout_s = int(timeout_s)
    if not 1 <= timeout_s <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be in [1, {MAX_TIMEOUT_S}], got {timeout_s}.")
    return timeout_s


def _validate_max_response_bytes(max_response_bytes: object) -> int:
    if isinstance(max_response_bytes, bool) or not isinstance(
        max_response_bytes, numbers.Integral
    ):
        raise TypeError(
            f"max_response_bytes must be an integer, got {max_response_bytes!r}."
        )
    if max_response_bytes < 1:
        raise ValueError(f"max_response_bytes must be >= 1, got {max_response_bytes}.")
    return int(max_response_bytes)


def _validate_endpoint(endpoint: str) -> str:
    """Require ``https`` (``http`` only for a local instance)."""
    parts = urlsplit(str(endpoint))
    if parts.scheme == "https" and parts.hostname:
        return str(endpoint)
    if parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS:
        return str(endpoint)
    raise ValueError(
        f"endpoint must be an https URL (http only for localhost), got {endpoint!r}."
    )


def _validate_max_retries(max_retries: object) -> int:
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise TypeError(f"max_retries must be an integer, got {max_retries!r}.")
    if not 0 <= max_retries <= MAX_RETRIES:
        raise ValueError(
            f"max_retries must be in [0, {MAX_RETRIES}], got {max_retries}."
        )
    return max_retries


def _hash_query(query: str) -> str:
    """SHA-256 of the query string, used as the cache filename stem."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _cache_key(query: str, endpoint: str) -> str:
    """Cache file stem: the query hash, plus the endpoint when not the default."""
    if endpoint == DEFAULT_ENDPOINT:
        return _hash_query(query)
    return hashlib.sha256(f"{endpoint}\n{query}".encode("utf-8")).hexdigest()


def _check_payload(payload: Any) -> dict[str, Any]:
    """Raise :class:`OverpassError` unless ``payload`` is a usable result."""
    if not isinstance(payload, dict):
        raise OverpassError(
            f"Overpass payload must be a JSON object, got {type(payload).__name__}."
        )
    elements = payload.get("elements", [])
    if elements is not None and not isinstance(elements, list):
        raise OverpassError("Overpass payload: 'elements' must be a list.")
    remark = payload.get("remark")
    if isinstance(remark, str) and "error" in remark.lower():
        raise OverpassError(f"Overpass reported an error: {remark[:500]}")
    return payload


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` via a temporary file and an atomic rename.

    The temporary file lives in the same directory (same file system, so
    :func:`os.replace` is atomic) and is created with the default
    permissions of the process, like the cache file before 2026.11.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------


def _read_capped(response: Any, max_bytes: int) -> bytes:
    """Read a streamed body, at most ``max_bytes`` bytes."""
    length = (getattr(response, "headers", None) or {}).get("Content-Length")
    if length is not None:
        try:
            declared = int(length)
        except (TypeError, ValueError):
            declared = None
        if declared is not None and declared > max_bytes:
            raise OverpassError(
                f"Overpass response of {declared} bytes exceeds the cap of "
                f"{max_bytes} bytes (max_response_bytes).",
                status_code=getattr(response, "status_code", None),
            )
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=65536):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise OverpassError(
                f"Overpass response exceeds the cap of {max_bytes} bytes "
                "(max_response_bytes).",
                status_code=getattr(response, "status_code", None),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _peek_body(response: Any, limit: int = 1024) -> str:
    """First ``limit`` bytes of an error body, decoded leniently."""
    data = b""
    try:
        for chunk in response.iter_content(chunk_size=limit):
            data += chunk
            if len(data) >= limit:
                break
    except Exception:  # pragma: no cover - diagnostics only
        pass
    return data[:limit].decode("utf-8", errors="replace")


def _retry_delay(response: Any, attempt: int) -> float:
    """Wait before the next attempt: ``Retry-After`` or ``2**attempt``, capped."""
    value = (getattr(response, "headers", None) or {}).get("Retry-After")
    if value is not None:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            delay = math.nan  # HTTP-date form: use the exponential backoff
        if math.isfinite(delay):
            return min(max(delay, 0.0), MAX_RETRY_DELAY_S)
        if delay == math.inf:
            return MAX_RETRY_DELAY_S
    return min(float(2**attempt), MAX_RETRY_DELAY_S)


def _post_overpass(
    query: str,
    *,
    endpoint: str,
    timeout_s: int,
    max_retries: int = 1,
    _sleep: Any = time.sleep,
    _transport: Any = None,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> dict[str, Any]:
    """POST ``query`` to ``endpoint`` and return the decoded JSON.

    Retries up to ``max_retries`` times on ``429`` / ``503`` / ``504``,
    waiting ``Retry-After`` seconds when the server sends it, else
    ``2**attempt`` seconds, never longer than :data:`MAX_RETRY_DELAY_S`.
    All other non-2xx responses raise :class:`OverpassError` at once. The
    body is streamed and capped at ``max_response_bytes``.

    The ``_sleep`` and ``_transport`` hooks let tests replace
    :func:`time.sleep` and :func:`requests.post`.
    """
    _validate_endpoint(endpoint)
    _validate_max_retries(max_retries)
    max_response_bytes = _validate_max_response_bytes(max_response_bytes)
    transport = _transport if _transport is not None else _import_requests().post
    headers = {
        "User-Agent": _user_agent(),
        "Accept": "application/json",
    }
    for attempt in range(max_retries + 1):
        response = transport(
            endpoint,
            data={"data": query},
            headers=headers,
            timeout=timeout_s + 5,  # client side a little more lenient
            stream=True,
        )
        try:
            status = response.status_code
            if status == 200:
                body = _read_capped(response, max_response_bytes)
                try:
                    return json.loads(body)
                except ValueError as exc:
                    raise OverpassError(
                        "Overpass returned a 200 but the body is not valid JSON.",
                        status_code=status,
                        body=body[:1024].decode("utf-8", errors="replace"),
                    ) from exc
            if status in _RETRY_STATUSES and attempt < max_retries:
                _sleep(_retry_delay(response, attempt))
                continue
            raise OverpassError(
                f"Overpass returned HTTP {status}.",
                status_code=status,
                body=_peek_body(response),
            )
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
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
    max_retries: int = 1,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    _post: Any = None,
) -> dict[str, Any]:
    """Fetch (or load from cache) the raw Overpass response.

    Parameters
    ----------
    lat0_deg, lon0_deg
        Centre of the buildings-in-radius lookup, WGS84 degrees.
    radius_m
        Search radius in metres, at most :data:`MAX_RADIUS_M`.
    cache_dir
        Directory in which the response is cached. Defaults to
        :func:`default_cache_dir`. Created on first write.
    endpoint
        Overpass endpoint URL (``https``; ``http`` only for
        ``localhost``). Defaults to the canonical mirror.
    timeout_s
        Per-query timeout in s passed to Overpass, an integer in
        ``[1, MAX_TIMEOUT_S]``.
    force_refresh
        If ``True``, ignore any cached payload and re-query the
        endpoint, overwriting the cache file. Useful when the user
        knows the upstream data changed.
    max_retries
        Number of retry attempts on ``429`` / ``503`` / ``504``
        responses, in ``[0, MAX_RETRIES]``. Defaults to ``1`` (one
        retry, matching the historic behaviour).
    max_response_bytes
        Cap on the response size in bytes (default
        :data:`DEFAULT_MAX_RESPONSE_BYTES`).
    _post
        Internal hook for tests: a callable with the signature
        ``(query, *, endpoint, timeout_s, max_retries)``. Production
        calls use :func:`_post_overpass`.

    Returns
    -------
    dict[str, Any]
        The decoded Overpass JSON payload. The shape is
        ``{"version": …, "generator": …, "elements": [...]}``;
        only ``elements`` is required downstream.

    Raises
    ------
    OverpassError
        On HTTP errors, oversized or malformed responses and payloads
        whose ``remark`` reports an Overpass runtime error (such a
        payload is not cached).
    ValueError, TypeError
        For invalid arguments (see :func:`build_query`).

    Notes
    -----
    The cache filename is ``<sha256(query)>.json`` for the default
    endpoint and ``<sha256(endpoint + newline + query)>.json`` for any
    other endpoint. Files are written atomically; a cache file that
    cannot be read back is re-fetched (with a :class:`UserWarning`).
    """
    _validate_max_retries(max_retries)
    max_response_bytes = _validate_max_response_bytes(max_response_bytes)
    endpoint = _validate_endpoint(endpoint)
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
    query = build_query(lat0_deg, lon0_deg, radius_m, timeout_s=timeout_s)
    path = Path(cache_dir) / f"{_cache_key(query, endpoint)}.json"
    if path.exists() and not force_refresh:
        try:
            with path.open("r", encoding="utf-8") as fh:
                return _check_payload(json.load(fh))
        except (ValueError, OverpassError) as exc:
            warnings.warn(
                f"Overpass cache file {path.name} is unreadable ({exc}); re-fetching.",
                UserWarning,
                stacklevel=2,
            )
    if _post is not None:
        post = _post
    else:
        post = functools.partial(_post_overpass, max_response_bytes=max_response_bytes)
    payload = post(
        query,
        endpoint=endpoint,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    _check_payload(payload)
    _atomic_write_json(path, payload)
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
    exclude_building_tags: Optional[Collection[str]] = None,
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
    exclude_building_tags
        ``building=*`` values to drop (e.g.
        :data:`NON_SERVICE_BUILDING_TAGS`). ``None`` (default) keeps
        every element with a ``building`` tag — including garages,
        sheds and ``building=no``.

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
    _check_payload(payload)
    elements = payload.get("elements", []) or []
    if not all(isinstance(e, dict) for e in elements):
        raise OverpassError("Overpass payload: every element must be a JSON object.")
    if isinstance(exclude_building_tags, str):
        exclude_building_tags = (exclude_building_tags,)
    excluded = frozenset(exclude_building_tags or ())
    by_id: dict[int, dict[str, Any]] = {
        e["id"]: e for e in elements if e.get("type") == "way" and "id" in e
    }
    out: list[BuildingFootprint] = []

    for el in elements:
        if (el.get("tags") or {}).get("building") in excluded:
            continue
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
    max_retries: int = 1,
    exclude_building_tags: Optional[Collection[str]] = None,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    _post: Any = None,
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
    max_retries
        Number of retry attempts on ``429`` / ``503`` / ``504``
        responses. Forwarded to :func:`query_buildings`. Defaults to
        ``1``.
    exclude_building_tags
        Forwarded to :func:`parse_overpass_payload` (default: keep every
        building tag).
    max_response_bytes
        Forwarded to :func:`query_buildings`.
    _post
        Test hook, forwarded to :func:`query_buildings`.

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
        max_retries=max_retries,
        max_response_bytes=max_response_bytes,
        _post=_post,
    )
    if projector is None:
        projector = Projector(lat0_deg, lon0_deg)
    footprints = parse_overpass_payload(
        payload,
        projector,
        min_area_m2=min_area_m2,
        exclude_building_tags=exclude_building_tags,
    )
    return footprints, projector
