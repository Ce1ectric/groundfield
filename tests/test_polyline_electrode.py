"""Tests for the :class:`~groundfield.geometry.electrodes.PolylineElectrode`.

Covers
------
* the non-degeneracy guard added on top of the horizontal check:
  - a consecutive duplicate vertex (zero-length edge) is rejected,
  - repeating the first vertex at the end of a ``closed=True`` ring is
    rejected (the closing edge is implicit),
  - ``closed=True`` with only two vertices warns and is treated as an
    open wire,
  - valid open and closed polylines still construct.
* segment-length convergence of the polyline discretiser: the cluster
  impedance of a buried square ring converges as ``segment_length``
  shrinks (mirrors :func:`convergence_study` for rods and strips).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.geometry.electrodes import PolylineElectrode
from groundfield.postprocess.convergence import convergence_study

_SQUARE = [
    (-2.0, -2.0, 0.5),
    (2.0, -2.0, 0.5),
    (2.0, 2.0, 0.5),
    (-2.0, 2.0, 0.5),
]


# ---------------------------------------------------------------------
# Non-degeneracy guard
# ---------------------------------------------------------------------
def test_valid_open_and_closed_polylines_construct() -> None:
    open_pl = PolylineElectrode(
        name="p", wire_radius=0.01, vertices=_SQUARE, closed=False
    )
    # An open polyline has (n - 1) edges, a closed ring has n.
    assert len(open_pl.edges) == 3
    ring = PolylineElectrode(name="p", wire_radius=0.01, vertices=_SQUARE, closed=True)
    assert len(ring.edges) == 4


def test_consecutive_duplicate_vertex_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero length"):
        PolylineElectrode(
            name="p",
            wire_radius=0.01,
            vertices=[
                (0.0, 0.0, 0.5),
                (1.0, 0.0, 0.5),
                (1.0, 0.0, 0.5),  # duplicate of its predecessor
                (1.0, 1.0, 0.5),
            ],
            closed=False,
        )


def test_closed_ring_repeating_first_vertex_is_rejected() -> None:
    # The closing edge is implicit; repeating the first vertex collapses
    # it to zero length.
    with pytest.raises(ValueError, match="zero length"):
        PolylineElectrode(
            name="p",
            wire_radius=0.01,
            vertices=[
                (0.0, 0.0, 0.5),
                (1.0, 0.0, 0.5),
                (1.0, 1.0, 0.5),
                (0.0, 0.0, 0.5),  # == first vertex
            ],
            closed=True,
        )


def test_closed_with_two_vertices_warns_and_is_open() -> None:
    with pytest.warns(UserWarning, match="at least three"):
        pl = PolylineElectrode(
            name="p",
            wire_radius=0.01,
            vertices=[(0.0, 0.0, 0.5), (1.0, 0.0, 0.5)],
            closed=True,
        )
    # No closing edge is added, so it behaves as an open single wire.
    assert len(pl.edges) == 1


# ---------------------------------------------------------------------
# Discretisation convergence
# ---------------------------------------------------------------------
def test_polyline_ring_discretisation_converges() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world,
        "polyline",
        name="ring",
        vertices=_SQUARE,
        closed=True,
        wire_radius=0.01,
    )
    gf.create_source(world, attached_to="ring", magnitude=1.0)
    engine = gf.create_engine(backend="image", frequencies=[50.0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # coarse ds may warn on resolution
        df = convergence_study(world, engine, segment_lengths=[1.0, 0.5, 0.25, 0.125])

    df = df.sort_values("segment_length_m", ascending=False).reset_index(drop=True)
    # Finer discretisation -> strictly more segments.
    assert df["n_segments"].is_monotonic_increasing
    assert df["n_segments"].iloc[-1] > df["n_segments"].iloc[0]

    z = df["abs_Z"].to_numpy()
    steps = np.abs(np.diff(z))
    # Successive refinements change the answer by less and less ...
    assert steps[-1] < steps[0]
    # ... and the finest step is essentially converged (< 2 %).
    assert steps[-1] / z[-1] < 0.02
