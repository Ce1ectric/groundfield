"""Tests for the placement-spec layer."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import BaseModel, Field

from groundfield.generators import (
    ExplicitPlacement,
    ManhattanGridPlacement,
    PlacementSpec,
    Uniform,
)


# ---------------------------------------------------------------------
# Manhattan grid
# ---------------------------------------------------------------------


def test_manhattan_grid_returns_n_positions() -> None:
    p = ManhattanGridPlacement(n_per_row=4, spacing_x_m=10.0, spacing_y_m=20.0)
    rng = np.random.default_rng(0)
    pos = p.generate(10, rng)
    assert len(pos) == 10


def test_manhattan_grid_centred_on_centre_xy() -> None:
    p = ManhattanGridPlacement(
        n_per_row=2, spacing_x_m=10.0, spacing_y_m=10.0,
        centre_xy=(100.0, 200.0),
    )
    pos = p.generate(4, np.random.default_rng(0))
    cx = sum(x for x, _ in pos) / 4
    cy = sum(y for _, y in pos) / 4
    assert math.isclose(cx, 100.0, abs_tol=1e-9)
    assert math.isclose(cy, 200.0, abs_tol=1e-9)


def test_manhattan_grid_jitter_spreads_positions() -> None:
    p = ManhattanGridPlacement(
        n_per_row=10, spacing_x_m=20.0, spacing_y_m=20.0,
        jitter_m=5.0,
    )
    rng = np.random.default_rng(0)
    pos = p.generate(20, rng)
    # Every position is within ±2.5 m of its grid point.
    p_no_jitter = ManhattanGridPlacement(
        n_per_row=10, spacing_x_m=20.0, spacing_y_m=20.0,
    )
    grid = p_no_jitter.generate(20, np.random.default_rng(0))
    for (x, y), (gx, gy) in zip(pos, grid):
        assert abs(x - gx) <= 2.5 + 1e-9
        assert abs(y - gy) <= 2.5 + 1e-9


def test_manhattan_grid_is_reproducible_under_seed() -> None:
    p = ManhattanGridPlacement(jitter_m=3.0)
    a = p.generate(15, np.random.default_rng(42))
    b = p.generate(15, np.random.default_rng(42))
    assert a == b


def test_manhattan_grid_zero_returns_empty() -> None:
    p = ManhattanGridPlacement()
    assert p.generate(0, np.random.default_rng(0)) == []


def test_manhattan_grid_accepts_distribution_for_spacing() -> None:
    p = ManhattanGridPlacement(spacing_x_m=Uniform(low=20.0, high=30.0),
                                spacing_y_m=Uniform(low=20.0, high=30.0))
    pos = p.generate(5, np.random.default_rng(0))
    assert len(pos) == 5


# ---------------------------------------------------------------------
# Explicit placement
# ---------------------------------------------------------------------


def test_explicit_placement_returns_provided() -> None:
    p = ExplicitPlacement(positions=[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
    pos = p.generate(3, np.random.default_rng(0))
    assert pos == [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]


def test_explicit_placement_truncates_to_n() -> None:
    p = ExplicitPlacement(positions=[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
    pos = p.generate(2, np.random.default_rng(0))
    assert pos == [(0.0, 0.0), (10.0, 0.0)]


def test_explicit_placement_raises_when_n_exceeds_list() -> None:
    p = ExplicitPlacement(positions=[(0.0, 0.0)])
    with pytest.raises(ValueError, match="requested 5 positions"):
        p.generate(5, np.random.default_rng(0))


# ---------------------------------------------------------------------
# Discriminated union round-trip
# ---------------------------------------------------------------------


class _Wrapper(BaseModel):
    p: PlacementSpec


@pytest.mark.parametrize(
    "p",
    [
        ManhattanGridPlacement(spacing_x_m=20.0, spacing_y_m=15.0,
                                n_per_row=8, jitter_m=2.0),
        ExplicitPlacement(positions=[(1.0, 2.0), (3.0, 4.0)]),
    ],
)
def test_placement_json_roundtrip(p) -> None:
    payload = _Wrapper(p=p).model_dump_json()
    restored = _Wrapper.model_validate_json(payload).p
    assert type(restored) is type(p)
    assert restored.model_dump() == p.model_dump()
