"""Close, interpenetrating and crossing wires (review pass 10, 10.44).

Geometry of R1-12: two 3 m rods (radius 8 mm) 2 mm apart interpenetrate
(about 10x the physical impedance), 2 cm apart +79 %, 5 cm +24 %.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield.conductors import (
    find_close_conductors,
    find_close_wires,
    segment_distance,
)

pytestmark = pytest.mark.finding("10.44")


@pytest.mark.parametrize(
    "gap, kind",
    [(0.002, "interpenetrating"), (0.02, "close"), (0.05, "close"), (0.5, None)],
)
def test_parallel_rods(gap: float, kind: str | None) -> None:
    ep = np.array([[[0, 0, 0.5], [0, 0, 3.5]], [[gap, 0, 0.5], [gap, 0, 3.5]]], float)
    found = find_close_wires(ep, [0.008, 0.008], ["r1", "r2"])
    if kind is None:
        assert found == []
    else:
        assert [p.kind for p in found] == [kind]
        assert found[0].distance == pytest.approx(gap, rel=1e-12)
        assert found[0].clearance == pytest.approx(gap - 0.016, rel=1e-12)


def test_crossing_and_junction() -> None:
    ep = np.array(
        [
            [[0, 0, 0.7], [10, 0, 0.7]],
            [[5, -5, 0.7], [5, 5, 0.7]],
            [[10, 0, 0.7], [10, 10, 0.7]],
        ],
        float,
    )
    found = find_close_wires(ep, [0.005] * 3, ["a", "b", "c"])
    assert [(p.kind, p.name_a, p.name_b) for p in found] == [("crossing", "a", "b")]
    assert found[0].point_a == pytest.approx((5.0, 0.0, 0.7))
    with_j = find_close_wires(ep, [0.005] * 3, ["a", "b", "c"], include_junctions=True)
    assert sorted(p.kind for p in with_j) == ["crossing", "junction"]


def test_t_contact_is_touching() -> None:
    """A drop that starts on the interior of a PEN is not connected there."""
    ep = np.array([[[0, 0, 0.6], [40, 0, 0.6]], [[20, 0, 0.6], [20, 15, 0.6]]], float)
    found = find_close_wires(ep, [0.005, 0.005], ["pen", "drop"])
    assert [p.kind for p in found] == ["touching"]


def test_segments_of_one_owner_are_never_paired() -> None:
    ep = np.array([[[0, 0, 0.7], [1, 0, 0.7]], [[1, 0, 0.7], [2, 0, 0.7]]], float)
    assert (
        find_close_wires(ep, [0.01, 0.01], ["pen", "pen"], include_junctions=True) == []
    )


def test_segment_distance_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    a1, a2, b1, b2 = (rng.normal(size=(150, 3)) for _ in range(4))
    d, s, t = segment_distance(a1, a2, b1, b2)
    grid = np.linspace(0.0, 1.0, 301)
    pa = a1[:, None, :] + grid[None, :, None] * (a2 - a1)[:, None, :]
    pb = b1[:, None, :] + grid[None, :, None] * (b2 - b1)[:, None, :]
    brute = np.linalg.norm(pa[:, :, None, :] - pb[:, None, :, :], axis=-1).min(
        axis=(1, 2)
    )
    assert np.all(d <= brute + 1e-12)
    assert np.all(brute - d <= 2e-4)
    assert np.all((s >= 0) & (s <= 1) & (t >= 0) & (t <= 1))


def test_find_close_conductors_on_a_world() -> None:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="a", position=(0, 0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="b", position=(30, 0, 0.5), length=2.0)
    gf.create_conductor(w, name="pen", start="a", end="b", cross_section=50e-6)
    gf.create_conductor(
        w,
        name="lead",
        start=(0.0, 0.005, 0.5),
        end=(30.0, 0.005, 0.5),
        wire_radius=0.004,
    )
    found = find_close_conductors(w.conductors)
    assert [(p.kind, {p.name_a, p.name_b}) for p in found] == [
        ("interpenetrating", {"pen", "lead"})
    ]


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        find_close_wires(np.zeros((2, 3)), [0.01], ["a"])
    with pytest.raises(ValueError):
        find_close_wires(np.zeros((1, 2, 3)), [0.01, 0.02], ["a"])
    assert (
        find_close_wires(np.zeros((1, 2, 3)) + [[0, 0, 0], [1, 0, 0]], [0.01], ["a"])
        == []
    )
