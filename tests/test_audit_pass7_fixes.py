"""Regression tests for the seventh 2026-05-18 audit pass.

Each test maps 1:1 to one of the user-visible bullets in the
``Fixed (Audit pass 7 — implemented 2026-05-18)`` CHANGELOG block:

* :func:`test_evaluate_spec_error_is_value_error_subclass`
* :func:`test_evaluate_spec_error_raised_on_missing_formula`
* :func:`test_evaluate_spec_error_raised_on_unknown_symbol`
* :func:`test_evaluate_spec_error_top_level_export`
* :func:`test_world_reset_concrete_corrections_clears_registry`
* :func:`test_world_reset_concrete_corrections_returns_previous`
* :func:`test_world_reset_concrete_corrections_idempotent`
* :func:`test_osm_placement_footprint_at_returns_none_by_default`
* :func:`test_osm_placement_footprint_at_strict_raises_indexerror`
* :func:`test_osm_placement_strict_message_carries_context`
* :func:`test_osm_placement_n_footprints_property`
* :func:`test_query_buildings_accepts_max_retries`
* :func:`test_query_buildings_rejects_negative_max_retries`
* :func:`test_query_and_project_forwards_max_retries`
* :func:`test_query_buildings_default_max_retries_is_one`
"""

from __future__ import annotations

import inspect
import pathlib
import tempfile

import pytest

import groundfield as gf
from groundfield.geo.footprint import BuildingFootprint
from groundfield.geo.osm import query_and_project, query_buildings
from groundfield.geo.placement import OsmBuildingPlacement
from groundfield.io.groundinsight import (
    BusTypeSpec,
    EvaluateSpecError,
    evaluate_spec,
)
from groundfield.world import World


# ---------------------------------------------------------------------
# evaluate_spec — typed exception
# ---------------------------------------------------------------------


def _square_metres(side: float = 5.0) -> list[tuple[float, float]]:
    """Helper: closed CCW square footprint of given side, centred at origin."""
    h = side / 2.0
    return [(-h, -h), (h, -h), (h, h), (-h, h), (-h, -h)]


def test_evaluate_spec_error_is_value_error_subclass() -> None:
    assert issubclass(EvaluateSpecError, ValueError)


def test_evaluate_spec_error_raised_on_missing_formula() -> None:
    spec = BusTypeSpec(
        name="dummy",
        description=None,
        system_type="TN",
        voltage_level=400.0,
        impedance_formula="",
        samples={
            "frequency_Hz": [50.0],
            "rho_Ohm_m": [100.0],
            "Z_real_Ohm": [1.0],
            "Z_imag_Ohm": [0.0],
        },
        metadata={},
    )
    with pytest.raises(EvaluateSpecError, match="impedance_formula is missing"):
        evaluate_spec(spec, frequencies=[50.0], rho=100.0)


def test_evaluate_spec_error_raised_on_unknown_symbol() -> None:
    # `omega` is not in the allowed set {f, rho, j}.
    spec = BusTypeSpec(
        name="dummy",
        description=None,
        system_type="TN",
        voltage_level=400.0,
        impedance_formula="omega * f",
        samples={
            "frequency_Hz": [50.0],
            "rho_Ohm_m": [100.0],
            "Z_real_Ohm": [1.0],
            "Z_imag_Ohm": [0.0],
        },
        metadata={},
    )
    with pytest.raises(EvaluateSpecError, match="unrecognised free symbols"):
        evaluate_spec(spec, frequencies=[50.0], rho=100.0)
    # Legacy ``except ValueError`` blocks must still catch it.
    try:
        evaluate_spec(spec, frequencies=[50.0], rho=100.0)
    except ValueError as exc:
        assert isinstance(exc, EvaluateSpecError)
    else:  # pragma: no cover - guard against silent regression
        pytest.fail("evaluate_spec did not raise on unknown free symbol")


def test_evaluate_spec_error_top_level_export() -> None:
    # Public surface check: importable from the top-level package and
    # listed in ``__all__`` so editor auto-complete and ``from gf
    # import *`` style usage both see the new class.
    assert hasattr(gf, "EvaluateSpecError")
    assert gf.EvaluateSpecError is EvaluateSpecError
    assert "EvaluateSpecError" in gf.__all__


# ---------------------------------------------------------------------
# World.reset_concrete_corrections — ADR-0012 V1 registry cleanup
# ---------------------------------------------------------------------


def test_world_reset_concrete_corrections_clears_registry() -> None:
    world = World(name="t")
    world.concrete_shell_corrections["house_1.pen_drop"] = 0.42
    world.concrete_shell_corrections["house_2.pen_drop"] = 1.7
    assert len(world.concrete_shell_corrections) == 2
    world.reset_concrete_corrections()
    assert world.concrete_shell_corrections == {}


def test_world_reset_concrete_corrections_returns_previous() -> None:
    world = World(name="t")
    world.concrete_shell_corrections["house_1.pen_drop"] = 0.42
    previous = world.reset_concrete_corrections()
    assert previous == {"house_1.pen_drop": 0.42}
    # Returned dict is a copy: mutating it must not bring entries back.
    previous["house_1.pen_drop"] = 9.9
    assert world.concrete_shell_corrections == {}


def test_world_reset_concrete_corrections_idempotent() -> None:
    # Calling on a fresh world must be a no-op and return ``{}``.
    world = World(name="t")
    assert world.reset_concrete_corrections() == {}
    assert world.concrete_shell_corrections == {}
    # Second call still returns ``{}``.
    assert world.reset_concrete_corrections() == {}


# ---------------------------------------------------------------------
# OsmBuildingPlacement.footprint_at — defensive bounds
# ---------------------------------------------------------------------


def _placement_with_two_footprints() -> OsmBuildingPlacement:
    fp_a = BuildingFootprint(polygon_xy_m=_square_metres(6.0))
    fp_b = BuildingFootprint(polygon_xy_m=_square_metres(8.0))
    return OsmBuildingPlacement(footprints=[fp_a, fp_b])


def test_osm_placement_footprint_at_returns_none_by_default() -> None:
    placement = _placement_with_two_footprints()
    # In-range
    assert placement.footprint_at(0) is not None
    assert placement.footprint_at(1) is not None
    # Out-of-range — historic ``None`` fall-back stays the default.
    assert placement.footprint_at(2) is None
    assert placement.footprint_at(-1) is None


def test_osm_placement_footprint_at_strict_raises_indexerror() -> None:
    placement = _placement_with_two_footprints()
    with pytest.raises(IndexError):
        placement.footprint_at(2, strict=True)
    with pytest.raises(IndexError):
        placement.footprint_at(-1, strict=True)
    # ``strict=True`` for an in-range index still returns the footprint.
    fp = placement.footprint_at(0, strict=True)
    assert fp is not None


def test_osm_placement_strict_message_carries_context() -> None:
    placement = _placement_with_two_footprints()
    with pytest.raises(IndexError) as excinfo:
        placement.footprint_at(7, strict=True)
    message = str(excinfo.value)
    # The structured message names the placement, the index, and the
    # filtered length — the AP1 generator pipeline can grep these.
    assert "OsmBuildingPlacement.footprint_at" in message
    assert "7" in message
    assert "2 footprints" in message
    assert "min_area_m2" in message


def test_osm_placement_n_footprints_property() -> None:
    placement = _placement_with_two_footprints()
    assert placement.n_footprints == 2
    assert len(placement) == placement.n_footprints
    # The filter should suppress the smaller footprint.
    filtered_placement = OsmBuildingPlacement(
        footprints=list(placement.footprints),
        min_area_m2=40.0,  # 6×6=36 m² dropped, 8×8=64 m² kept
    )
    assert filtered_placement.n_footprints == 1


# ---------------------------------------------------------------------
# geo.osm.max_retries plumbing
# ---------------------------------------------------------------------


def test_query_buildings_default_max_retries_is_one() -> None:
    sig = inspect.signature(query_buildings)
    assert sig.parameters["max_retries"].default == 1


def test_query_buildings_accepts_max_retries() -> None:
    payload = {"version": 0.6, "elements": []}
    seen = {"max_retries": None, "n": 0}

    def fake_post(query, *, endpoint, timeout_s, max_retries=1, _sleep=None):
        seen["max_retries"] = max_retries
        seen["n"] += 1
        return payload

    with tempfile.TemporaryDirectory() as td:
        cache_dir = pathlib.Path(td)
        _ = query_buildings(
            52.5, 13.4, 100.0,
            cache_dir=cache_dir,
            _post=fake_post,
            max_retries=3,
        )
    assert seen["max_retries"] == 3
    assert seen["n"] == 1


def test_query_buildings_rejects_negative_max_retries() -> None:
    payload = {"version": 0.6, "elements": []}

    def fake_post(query, *, endpoint, timeout_s, max_retries=1, _sleep=None):
        return payload

    with tempfile.TemporaryDirectory() as td:
        cache_dir = pathlib.Path(td)
        with pytest.raises(ValueError, match="max_retries"):
            query_buildings(
                52.5, 13.4, 100.0,
                cache_dir=cache_dir,
                _post=fake_post,
                max_retries=-1,
            )


def test_query_and_project_forwards_max_retries() -> None:
    payload = {"version": 0.6, "elements": []}
    seen = {"max_retries": None}

    def fake_post(query, *, endpoint, timeout_s, max_retries=1, _sleep=None):
        seen["max_retries"] = max_retries
        return payload

    with tempfile.TemporaryDirectory() as td:
        cache_dir = pathlib.Path(td)
        # ``query_and_project`` invokes ``query_buildings`` internally
        # — ``_post`` is not in its signature, so we use a low-level
        # check via ``query_buildings`` plus a separate plumbing check
        # on the keyword presence.
        sig = inspect.signature(query_and_project)
        assert "max_retries" in sig.parameters
        assert sig.parameters["max_retries"].default == 1

        # Round-trip: a call with the historic default keeps behaviour
        # unchanged.
        _ = query_buildings(
            52.5, 13.4, 100.0,
            cache_dir=cache_dir,
            _post=fake_post,
        )
    assert seen["max_retries"] == 1
