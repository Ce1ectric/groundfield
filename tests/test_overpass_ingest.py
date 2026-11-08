"""Bounded, validated Overpass requests with atomic caching (10.05).

All tests run against a mocked transport; the single test against the real
Overpass API is marked ``network`` and skipped by default.

Covered: query injection through ``timeout_s``, NaN / out-of-range
coordinates and radii, response-size cap (``Content-Length`` and streamed
body), HTTP-200 payloads carrying an Overpass runtime-error ``remark`` (not
cached), malformed payloads (``OverpassError`` instead of
``AttributeError``), atomic cache writes and recovery from a truncated cache
file, ``https`` endpoints, the endpoint in the cache key, bounded retries with
``Retry-After``; plus the explicit building-tag filter.
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from groundfield.geo import osm
from groundfield.geo.projection import Projector

pytestmark = pytest.mark.finding("10.05")

LAT, LON = 51.9136, 10.8432


def _way(i: int, building: str = "house") -> dict:
    lat, lon = LAT + 1e-4 * i, LON
    return {
        "type": "way",
        "id": 100 + i,
        "tags": {"building": building},
        "geometry": [
            {"lat": lat, "lon": lon},
            {"lat": lat + 5e-5, "lon": lon},
            {"lat": lat + 5e-5, "lon": lon + 8e-5},
            {"lat": lat, "lon": lon + 8e-5},
        ],
    }


PAYLOAD = {"version": 0.6, "elements": [_way(0), _way(1, "garage"), _way(2, "yes")]}


class FakeResponse:
    """Minimal stand-in for a streamed ``requests.Response``."""

    def __init__(self, status: int, body: bytes, headers: dict | None = None) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size: int = 1):
        for k in range(0, len(self._body), chunk_size):
            yield self._body[k : k + chunk_size]

    @property
    def text(self) -> str:  # pragma: no cover - must not be used (unbounded)
        raise AssertionError("the body must be read through iter_content")

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def _ok(payload: dict = PAYLOAD) -> FakeResponse:
    return FakeResponse(200, json.dumps(payload).encode())


# ---------------------------------------------------------------------
# Validated request parameters
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "timeout_s",
    ["25][maxsize:4000000000];node(1);out;//", 25.5, True, 0, -5, 100000],
)
def test_timeout_must_be_a_bounded_integer(timeout_s: object) -> None:
    with pytest.raises((TypeError, ValueError), match="timeout_s"):
        osm.build_query(LAT, LON, 200.0, timeout_s=timeout_s)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("lat", "lon", "radius"),
    [
        (math.nan, LON, 200.0),
        (LAT, math.inf, 200.0),
        (123.0, LON, 200.0),
        (LAT, 400.0, 200.0),
        (LAT, LON, math.inf),
        (LAT, LON, 5.0e5),
        (LAT, LON, 0.0),
    ],
)
def test_coordinates_and_radius_are_validated(
    lat: float, lon: float, radius: float
) -> None:
    with pytest.raises(ValueError):
        osm.build_query(lat, lon, radius)


def test_valid_query_text_is_unchanged() -> None:
    """Same text as before, so existing cache files keep their keys."""
    assert osm.build_query(LAT, LON, 250.0, timeout_s=42) == (
        "[out:json][timeout:42];\n(\n"
        '  way["building"](around:250.0,51.913600,10.843200);\n'
        '  relation["building"](around:250.0,51.913600,10.843200);\n'
        ");\nout body geom tags;"
    )


@pytest.mark.parametrize(
    "endpoint", ["http://overpass.example.org/api/interpreter", "ftp://x", "overpass"]
)
def test_insecure_or_invalid_endpoints_are_rejected(endpoint: str, tmp_path) -> None:
    with pytest.raises(ValueError, match="endpoint"):
        osm.query_buildings(LAT, LON, 100.0, endpoint=endpoint, cache_dir=tmp_path)


def test_local_http_endpoint_is_allowed(tmp_path) -> None:
    transport = FakeTransport([_ok()])
    payload = osm._post_overpass(
        osm.build_query(LAT, LON, 100.0),
        endpoint="http://localhost:12345/api/interpreter",
        timeout_s=30,
        _transport=transport,
    )
    assert payload == PAYLOAD


# ---------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------


def test_response_size_is_capped_by_content_length() -> None:
    big = FakeResponse(200, b"{}", headers={"Content-Length": str(10**9)})
    with pytest.raises(osm.OverpassError, match="exceeds"):
        osm._post_overpass(
            "q",
            endpoint=osm.DEFAULT_ENDPOINT,
            timeout_s=30,
            _transport=FakeTransport([big]),
        )
    assert big.closed


def test_response_size_is_capped_while_streaming() -> None:
    body = json.dumps({"elements": [_way(i) for i in range(200)]}).encode()
    transport = FakeTransport([FakeResponse(200, body)])
    with pytest.raises(osm.OverpassError, match="exceeds"):
        osm._post_overpass(
            "q",
            endpoint=osm.DEFAULT_ENDPOINT,
            timeout_s=30,
            _transport=transport,
            max_response_bytes=1000,
        )
    assert transport.calls[0]["stream"] is True


def test_remark_runtime_error_raises_and_is_not_cached(tmp_path) -> None:
    partial = dict(
        PAYLOAD, remark='runtime error: Query timed out in "query" at line 3.'
    )
    calls = []

    def fake_post(query, **kw):
        calls.append(query)
        return partial

    for _ in range(2):
        with pytest.raises(osm.OverpassError, match="runtime error"):
            osm.query_buildings(LAT, LON, 200.0, cache_dir=tmp_path, _post=fake_post)
    assert len(calls) == 2
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize(
    "payload", [["not", "a", "dict"], {"elements": ["x"]}, {"elements": 3}]
)
def test_malformed_payload_raises_overpass_error(payload: object) -> None:
    with pytest.raises(osm.OverpassError):
        osm.parse_overpass_payload(payload, Projector(LAT, LON))  # type: ignore[arg-type]


_SQUARE = _way(0)["geometry"]

MALFORMED_ELEMENTS = {
    "tags is a list": {
        "type": "way",
        "id": 1,
        "tags": ["building"],
        "geometry": _SQUARE,
    },
    "id is a list": {"type": "way", "id": [1], "tags": {"building": "yes"}},
    "id is a bool": {"type": "way", "id": True, "tags": {"building": "yes"}},
    "type is a number": {"type": 3, "id": 1},
    "member is a string": {
        "type": "relation",
        "id": 2,
        "tags": {"type": "multipolygon", "building": "yes"},
        "members": ["x"],
    },
    "member ref is a string": {
        "type": "relation",
        "id": 2,
        "tags": {"type": "multipolygon", "building": "yes"},
        "members": [{"type": "way", "ref": "7", "role": "outer"}],
    },
    "building value is a list": {
        "type": "way",
        "id": 6,
        "tags": {"building": ["yes"]},
        "geometry": _SQUARE,
    },
    "geometry is an object": {"type": "way", "id": 7, "geometry": {"lat": 1.0}},
    "vertex lat is a string": {
        "type": "way",
        "id": 8,
        "geometry": [{"lat": "52.0", "lon": 10.0}] + _SQUARE[1:],
    },
}


@pytest.mark.parametrize("label", sorted(MALFORMED_ELEMENTS))
def test_malformed_elements_raise_overpass_error(label: str) -> None:
    """Audit M1: such payloads crashed the parser with AttributeError /
    TypeError instead of OverpassError."""
    payload = {"elements": [MALFORMED_ELEMENTS[label]]}
    with pytest.raises(osm.OverpassError, match="elements\\[0\\]"):
        osm.parse_overpass_payload(payload, Projector(LAT, LON))


@pytest.mark.parametrize(
    ("lat", "lon"), [(math.nan, LON), (1e308, LON), (95.0, LON), (LAT, 200.0)]
)
def test_non_finite_or_out_of_range_vertices_are_rejected(lat, lon) -> None:
    """Audit M1: NaN / 1e308 / lat 95 vertices gave NaN-area footprints that
    passed ``min_area_m2`` (NaN < x is False)."""
    bad = dict(_way(0))
    bad["geometry"] = [{"lat": lat, "lon": lon}] + _way(0)["geometry"][1:]
    with pytest.raises(osm.OverpassError, match="vertex"):
        osm.parse_overpass_payload(
            {"elements": [bad]}, Projector(LAT, LON), min_area_m2=10.0
        )


def test_null_vertices_skip_the_element() -> None:
    """A ``null`` vertex (a node Overpass could not place) is not malformed:
    the element is skipped, the others are kept."""
    partial = dict(_way(1))
    partial["geometry"] = [None] + _way(1)["geometry"][1:]
    fps = osm.parse_overpass_payload(
        {"elements": [_way(0), partial]}, Projector(LAT, LON)
    )
    assert [fp.osm_id for fp in fps] == [100]


def test_malformed_payload_is_neither_cached_nor_parsed(tmp_path) -> None:
    """Audit M1: the payload was cached before it was parsed, so every later
    call crashed from the poisoned cache without re-fetching."""
    calls: list[int] = []

    def fake_post(query, *, endpoint, timeout_s, max_retries):
        calls.append(1)
        return {"elements": [MALFORMED_ELEMENTS["tags is a list"]]}

    for _ in range(2):
        with pytest.raises(osm.OverpassError):
            osm.query_and_project(LAT, LON, 100.0, cache_dir=tmp_path, _post=fake_post)
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_malformed_cache_file_is_refetched(tmp_path) -> None:
    query = osm.build_query(LAT, LON, 100.0)
    path = tmp_path / f"{osm._hash_query(query)}.json"
    path.write_text(
        json.dumps({"elements": [MALFORMED_ELEMENTS["member is a string"]]}),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="cache"):
        fps, _ = osm.query_and_project(
            LAT, LON, 100.0, cache_dir=tmp_path, _post=lambda q, **kw: PAYLOAD
        )
    assert len(fps) == 3
    assert json.loads(path.read_text(encoding="utf-8")) == PAYLOAD


def test_default_response_cap_bounds_the_decoded_size() -> None:
    """Audit L7: 64 MiB of JSON decode to up to 1.6 GiB (25 x for a body of
    empty objects). The default cap keeps the worst case below 0.5 GiB and
    is 5 x the largest payload of the AP1 catalogue (3.0 MiB)."""
    assert 3 * 2**20 * 5 <= osm.DEFAULT_MAX_RESPONSE_BYTES
    assert 25 * osm.DEFAULT_MAX_RESPONSE_BYTES <= 2**29
    response = FakeResponse(200, b"x" * (osm.DEFAULT_MAX_RESPONSE_BYTES + 1))
    with pytest.raises(osm.OverpassError, match="cap"):
        osm._post_overpass(
            "q",
            endpoint=osm.DEFAULT_ENDPOINT,
            timeout_s=30,
            _transport=lambda *a, **k: response,
        )


def test_truncated_cache_file_is_refetched(tmp_path) -> None:
    query = osm.build_query(LAT, LON, 100.0)
    path = tmp_path / f"{osm._hash_query(query)}.json"
    path.write_text(json.dumps(PAYLOAD)[:40], encoding="utf-8")
    calls = []

    def fake_post(q, **kw):
        calls.append(q)
        return PAYLOAD

    with pytest.warns(UserWarning, match="cache"):
        payload = osm.query_buildings(
            LAT, LON, 100.0, cache_dir=tmp_path, _post=fake_post
        )
    assert payload == PAYLOAD and len(calls) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == PAYLOAD


def test_cache_write_is_atomic(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A writer that dies mid-way leaves no partial cache file behind."""

    def exploding_dump(obj, fh, **kw):
        fh.write('{"elements": [')
        raise KeyboardInterrupt

    monkeypatch.setattr(osm.json, "dump", exploding_dump)
    with pytest.raises(KeyboardInterrupt):
        osm.query_buildings(
            LAT, LON, 100.0, cache_dir=tmp_path, _post=lambda q, **kw: PAYLOAD
        )
    assert list(pathlib.Path(tmp_path).iterdir()) == []


def test_cache_key_covers_a_non_default_endpoint(tmp_path) -> None:
    payloads = {
        "https://a.example/api": {"elements": []},
        "https://b.example/api": PAYLOAD,
    }

    def fake_post(q, *, endpoint, **kw):
        return payloads[endpoint]

    a = osm.query_buildings(
        LAT,
        LON,
        100.0,
        cache_dir=tmp_path,
        endpoint="https://a.example/api",
        _post=fake_post,
    )
    b = osm.query_buildings(
        LAT,
        LON,
        100.0,
        cache_dir=tmp_path,
        endpoint="https://b.example/api",
        _post=fake_post,
    )
    assert (a, b) == ({"elements": []}, PAYLOAD)
    # The default endpoint keeps the historic key: sha256(query).
    default = osm.query_buildings(
        LAT, LON, 100.0, cache_dir=tmp_path, _post=lambda q, **kw: PAYLOAD
    )
    assert default == PAYLOAD
    assert (
        tmp_path / f"{osm._hash_query(osm.build_query(LAT, LON, 100.0))}.json"
    ).exists()


# ---------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------


def test_retries_honour_retry_after_and_are_bounded() -> None:
    slept: list[float] = []
    transport = FakeTransport(
        [
            FakeResponse(429, b"slow down", headers={"Retry-After": "7"}),
            FakeResponse(504, b"timeout", headers={"Retry-After": "100000"}),
            _ok(),
        ]
    )
    payload = osm._post_overpass(
        "q",
        endpoint=osm.DEFAULT_ENDPOINT,
        timeout_s=30,
        max_retries=2,
        _sleep=slept.append,
        _transport=transport,
    )
    assert payload == PAYLOAD
    assert slept == [7.0, osm.MAX_RETRY_DELAY_S]


@pytest.mark.parametrize("value", ["nan", "inf", "-3", "Wed, 21 Oct 2026 07:28:00 GMT"])
def test_odd_retry_after_values_give_a_finite_bounded_wait(value: str) -> None:
    transport = FakeTransport([FakeResponse(429, b"", {"Retry-After": value}), _ok()])
    slept: list[float] = []
    osm._post_overpass(
        "q",
        endpoint=osm.DEFAULT_ENDPOINT,
        timeout_s=25,
        _sleep=slept.append,
        _transport=transport,
    )
    assert len(slept) == 1 and 0.0 <= slept[0] <= osm.MAX_RETRY_DELAY_S


def test_cache_file_permissions_follow_the_umask(tmp_path) -> None:
    import os
    import stat

    osm.query_buildings(
        LAT, LON, 100.0, cache_dir=tmp_path, _post=lambda q, **kw: PAYLOAD
    )
    (cache_file,) = pathlib.Path(tmp_path).iterdir()
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(cache_file.stat().st_mode) == 0o666 & ~umask


def test_single_tag_string_is_one_tag() -> None:
    fps = osm.parse_overpass_payload(
        PAYLOAD, Projector(LAT, LON), exclude_building_tags="garage"
    )
    assert [fp.building_use for fp in fps] == ["house", "yes"]


def test_max_retries_is_bounded(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_retries"):
        osm.query_buildings(
            LAT,
            LON,
            100.0,
            cache_dir=tmp_path,
            max_retries=osm.MAX_RETRIES + 1,
            _post=None,
        )


def test_error_body_is_read_bounded() -> None:
    transport = FakeTransport([FakeResponse(500, b"x" * 10_000)])
    with pytest.raises(osm.OverpassError) as info:
        osm._post_overpass(
            "q", endpoint=osm.DEFAULT_ENDPOINT, timeout_s=30, _transport=transport
        )
    assert info.value.status_code == 500
    assert len(info.value.body) <= 1024


# ---------------------------------------------------------------------
# Building-tag filter (explicit and documented)
# ---------------------------------------------------------------------


def test_every_building_tag_is_accepted_by_default() -> None:
    fps = osm.parse_overpass_payload(PAYLOAD, Projector(LAT, LON))
    assert [fp.building_use for fp in fps] == ["house", "garage", "yes"]


def test_building_tag_filter_drops_non_service_buildings() -> None:
    assert {"garage", "shed", "carport"} <= osm.NON_SERVICE_BUILDING_TAGS
    fps = osm.parse_overpass_payload(
        PAYLOAD,
        Projector(LAT, LON),
        exclude_building_tags=osm.NON_SERVICE_BUILDING_TAGS,
    )
    assert [fp.building_use for fp in fps] == ["house", "yes"]


def test_query_and_project_forwards_the_tag_filter(tmp_path) -> None:
    fps, _ = osm.query_and_project(
        LAT,
        LON,
        100.0,
        cache_dir=tmp_path,
        exclude_building_tags={"garage"},
        _post=lambda q, **kw: PAYLOAD,
    )
    assert [fp.osm_id for fp in fps] == [100, 102]


def test_layout_from_osm_forwards_the_tag_filter(tmp_path) -> None:
    from groundfield.generators import LvNetworkLayout

    osm.query_buildings(
        LAT, LON, 100.0, cache_dir=tmp_path, _post=lambda q, **kw: PAYLOAD
    )
    layout = LvNetworkLayout.from_osm(
        center_lat_deg=LAT,
        center_lon_deg=LON,
        radius_m=100.0,
        cache_dir=tmp_path,
        exclude_building_tags=osm.NON_SERVICE_BUILDING_TAGS,
    )
    assert [fp.osm_id for fp in layout.footprints] == [100, 102]


# ---------------------------------------------------------------------
# Real Overpass call (opt-in)
# ---------------------------------------------------------------------


@pytest.mark.network
def test_real_overpass_query(tmp_path) -> None:
    payload = osm.query_buildings(
        LAT, LON, 50.0, cache_dir=tmp_path, force_refresh=True
    )
    assert isinstance(payload.get("elements"), list)
