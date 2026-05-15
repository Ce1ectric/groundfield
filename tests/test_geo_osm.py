"""Tests for :mod:`groundfield.geo.osm` and
:mod:`groundfield.geo.projection`.

Network IO is mocked at the ``_post`` boundary of
:func:`groundfield.geo.osm.query_buildings` so the suite never
opens a socket. Cache, query construction, retry behaviour and
the Overpass-payload parser are exercised end-to-end against
hand-crafted JSON payloads.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from groundfield.geo.osm import (
    DEFAULT_ENDPOINT,
    OverpassError,
    build_query,
    parse_overpass_payload,
    query_buildings,
)
from groundfield.geo.projection import Projector


# ---------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------


def test_projector_origin_maps_to_zero() -> None:
    proj = Projector(52.5, 13.4)
    x, y = proj.to_xy_m(52.5, 13.4)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6


def test_projector_round_trip() -> None:
    proj = Projector(52.5, 13.4)
    for dx in (-500.0, 0.0, 500.0):
        for dy in (-500.0, 0.0, 500.0):
            lat, lon = proj.to_lat_lon(dx, dy)
            x, y = proj.to_xy_m(lat, lon)
            assert abs(x - dx) < 1e-6
            assert abs(y - dy) < 1e-6


def test_projector_one_km_north_matches_latitude_delta() -> None:
    proj = Projector(52.5, 13.4)
    lat_n, _ = proj.to_lat_lon(0.0, 1000.0)
    # 1 km north corresponds to roughly 1 km / 111_320 m/deg of latitude.
    expected_delta = 1000.0 / 111_320.0
    assert lat_n == pytest.approx(52.5 + expected_delta, abs=5e-4)


def test_projector_ring_to_xy_m_round_trip() -> None:
    proj = Projector(52.5, 13.4)
    ring_ll = [
        (52.5000, 13.4000),
        (52.5001, 13.4000),
        (52.5001, 13.4001),
        (52.5000, 13.4001),
    ]
    ring_xy = proj.ring_to_xy_m(ring_ll)
    assert len(ring_xy) == 4
    # Compare against single-point projection.
    for (lat, lon), (x, y) in zip(ring_ll, ring_xy):
        x_ref, y_ref = proj.to_xy_m(lat, lon)
        assert (x, y) == pytest.approx((x_ref, y_ref), abs=1e-9)


def test_projector_rejects_invalid_origin() -> None:
    with pytest.raises(ValueError):
        Projector(120.0, 13.4)
    with pytest.raises(ValueError):
        Projector(52.5, 200.0)


# ---------------------------------------------------------------------
# build_query
# ---------------------------------------------------------------------


def test_build_query_is_deterministic() -> None:
    q1 = build_query(52.5, 13.4, 200.0)
    q2 = build_query(52.5, 13.4, 200.0)
    assert q1 == q2


def test_build_query_rejects_non_positive_radius() -> None:
    with pytest.raises(ValueError):
        build_query(52.5, 13.4, 0.0)
    with pytest.raises(ValueError):
        build_query(52.5, 13.4, -10.0)


def test_build_query_contains_expected_tokens() -> None:
    q = build_query(52.5, 13.4, 250.0, timeout_s=42)
    assert "[out:json][timeout:42]" in q
    assert "around:250.0,52.500000,13.400000" in q
    assert "way[\"building\"]" in q
    assert "relation[\"building\"]" in q


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------


def test_query_buildings_caches_to_disk() -> None:
    payload = {"version": 0.6, "elements": []}
    calls = {"n": 0}

    def fake_post(query, *, endpoint, timeout_s, max_retries=1, _sleep=None):
        calls["n"] += 1
        return payload

    with tempfile.TemporaryDirectory() as td:
        cache_dir = pathlib.Path(td)
        p1 = query_buildings(
            52.5, 13.4, 100.0, cache_dir=cache_dir, _post=fake_post,
        )
        p2 = query_buildings(
            52.5, 13.4, 100.0, cache_dir=cache_dir, _post=fake_post,
        )
        assert calls["n"] == 1
        assert p1 == p2
        files = list(cache_dir.glob("*.json"))
        assert len(files) == 1
        with files[0].open() as fh:
            assert json.load(fh) == payload


def test_query_buildings_force_refresh_bypasses_cache() -> None:
    payload_a = {"version": 0.6, "elements": [{"type": "way", "id": 1}]}
    payload_b = {"version": 0.6, "elements": [{"type": "way", "id": 2}]}
    state = {"current": payload_a, "n": 0}

    def fake_post(query, *, endpoint, timeout_s, max_retries=1, _sleep=None):
        state["n"] += 1
        return state["current"]

    with tempfile.TemporaryDirectory() as td:
        cache_dir = pathlib.Path(td)
        first = query_buildings(
            52.5, 13.4, 100.0, cache_dir=cache_dir, _post=fake_post,
        )
        state["current"] = payload_b
        second = query_buildings(
            52.5, 13.4, 100.0, cache_dir=cache_dir,
            _post=fake_post, force_refresh=True,
        )
        assert first["elements"][0]["id"] == 1
        assert second["elements"][0]["id"] == 2
        assert state["n"] == 2


# ---------------------------------------------------------------------
# Overpass payload parsing
# ---------------------------------------------------------------------


def test_parse_way_extracts_polygon_and_levels() -> None:
    proj = Projector(52.5, 13.4)
    payload = {
        "version": 0.6,
        "elements": [
            {
                "type": "way", "id": 1,
                "geometry": [
                    {"lat": 52.5000, "lon": 13.4000},
                    {"lat": 52.5001, "lon": 13.4000},
                    {"lat": 52.5001, "lon": 13.4001},
                    {"lat": 52.5000, "lon": 13.4001},
                    {"lat": 52.5000, "lon": 13.4000},
                ],
                "tags": {"building": "residential", "building:levels": "2"},
            }
        ],
    }
    fps = parse_overpass_payload(payload, proj)
    assert len(fps) == 1
    fp = fps[0]
    assert fp.osm_id == 1
    assert fp.levels == 2.0
    assert fp.building_use == "residential"
    # The way is roughly an 11 m x 7 m rectangle at this latitude.
    assert 50.0 < fp.area_m2() < 110.0


def test_parse_multipolygon_with_inner_hole() -> None:
    proj = Projector(52.5, 13.4)
    payload = {
        "version": 0.6,
        "elements": [
            {
                "type": "relation", "id": 42,
                "tags": {"type": "multipolygon", "building": "industrial"},
                "members": [
                    {
                        "type": "way", "role": "outer",
                        "geometry": [
                            {"lat": 52.500, "lon": 13.400},
                            {"lat": 52.500, "lon": 13.401},
                            {"lat": 52.501, "lon": 13.401},
                            {"lat": 52.501, "lon": 13.400},
                            {"lat": 52.500, "lon": 13.400},
                        ],
                    },
                    {
                        "type": "way", "role": "inner",
                        "geometry": [
                            {"lat": 52.5002, "lon": 13.4002},
                            {"lat": 52.5002, "lon": 13.4008},
                            {"lat": 52.5008, "lon": 13.4008},
                            {"lat": 52.5008, "lon": 13.4002},
                            {"lat": 52.5002, "lon": 13.4002},
                        ],
                    },
                ],
            }
        ],
    }
    fps = parse_overpass_payload(payload, proj)
    assert len(fps) == 1
    assert len(fps[0].holes_xy_m) == 1
    assert fps[0].building_use == "industrial"


def test_parse_skips_features_below_min_area() -> None:
    proj = Projector(52.5, 13.4)
    payload = {
        "version": 0.6,
        "elements": [
            {
                "type": "way", "id": 1,
                "geometry": [
                    # Tiny ~ 1 m x 1 m polygon — below typical Gartenhaus.
                    {"lat": 52.50000, "lon": 13.40000},
                    {"lat": 52.50001, "lon": 13.40000},
                    {"lat": 52.50001, "lon": 13.40001},
                    {"lat": 52.50000, "lon": 13.40001},
                ],
                "tags": {"building": "yes"},
            }
        ],
    }
    fps = parse_overpass_payload(payload, proj, min_area_m2=10.0)
    assert fps == []


# ---------------------------------------------------------------------
# Error surface
# ---------------------------------------------------------------------


def test_overpass_error_carries_status_code_and_body() -> None:
    err = OverpassError("boom", status_code=429, body="rate limited")
    assert err.status_code == 429
    assert "rate limited" in err.body


def test_default_endpoint_is_canonical_overpass() -> None:
    assert "overpass" in DEFAULT_ENDPOINT
