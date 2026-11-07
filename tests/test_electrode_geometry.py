"""Geometry queries on the electrode models (review pass 10, 10.79).

Eight ``isinstance`` ladders over the electrode kinds (solver,
diagnostics, plots, VTK export) disagreed with each other — the
resolution check skipped polylines and stars (F53/F54). The kinds now
answer the queries themselves: ``segments(ds)``, ``segment_count(ds)``,
``wire_polylines()``, ``bounds()``, ``total_length()`` and
``min_feature_length()``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.geometry import electrodes as E

_KINDS = {
    "rod": E.RodElectrode(name="rod", position=(0.3, -1.2, 0.5), length=3.0),
    "ring": E.RingElectrode(name="ring", center=(1.0, 2.0, 0.8), radius=4.3),
    "strip": E.StripElectrode(
        name="strip", start=(0.0, 0.0, 0.6), end=(7.3, -2.1, 0.6)
    ),
    "polyline": E.PolylineElectrode(
        name="polyline",
        vertices=[(0, 0, 0.8), (10.3, 0, 0.8), (10.3, 7.7, 0.8), (0.0, 9.1, 0.8)],
    ),
    "star": E.StarElectrode(
        name="star", center=(2.0, -3.0, 0.6), n_arms=5, arm_length=2.7
    ),
    "mesh": E.MeshElectrode(
        name="mesh", corner=(0, 0, 0.5), size=(10.0, 7.0), spacing=2.3
    ),
    "grid_mesh": E.GridMeshElectrode(
        name="grid_mesh", corner=(-3, 1, 0.7), size=(10.0, 7.5), n_x=3, n_y=2
    ),
}


def test_every_kind_is_covered() -> None:
    """The table above lists every member of the ``Electrode`` union."""
    kinds = {cls.model_fields["kind"].default for cls in E.Electrode.__args__}
    assert kinds == set(_KINDS)


@pytest.mark.finding("10.79")
@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_geometry_queries_exist_and_agree(kind: str) -> None:
    e = _KINDS[kind]
    segs = e.segments(0.5)
    assert len(segs) == e.segment_count(0.5) > 0
    assert segs.midpoints.shape == (len(segs), 3)
    np.testing.assert_allclose(np.linalg.norm(segs.directions, axis=1), 1.0)
    # The segments tile the wires: their lengths add up to the wire length.
    assert math.isclose(segs.total_length, e.total_length(), rel_tol=1e-12)
    # End points lie on the polylines' bounding box, midpoints inside it.
    lo, hi = e.bounds()
    poly = np.concatenate(e.wire_polylines(), axis=0)
    assert np.all(poly >= lo - 1e-12) and np.all(poly <= hi + 1e-12)
    assert np.all(segs.midpoints >= lo - 1e-9) and np.all(segs.midpoints <= hi + 1e-9)
    assert 0.0 < e.min_feature_length() <= e.total_length()


@pytest.mark.finding("10.79")
def test_min_feature_length_covers_polyline_and_star() -> None:
    """Basis of the F53/F54 fix: polylines and stars report their feature."""
    poly = E.PolylineElectrode(
        name="p", vertices=[(0, 0, 0.7), (0.3, 0, 0.7), (0.3, 5.0, 0.7)], closed=False
    )
    star = E.StarElectrode(name="s", center=(10, 0, 0.6), n_arms=3, arm_length=0.3)
    assert poly.min_feature_length() == pytest.approx(0.3)
    assert star.min_feature_length() == pytest.approx(0.3)


@pytest.mark.finding("10.79")
@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_segment_count_matches_the_solver(kind: str) -> None:
    """``segment_count`` predicts the point sources of a solve exactly."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    w.add_electrode(_KINDS[kind])
    gf.create_source(w, attached_to=kind, magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = w.solve(gf.Engine(backend="image", segment_length=0.5))
    assert len(res.point_sources) == _KINDS[kind].segment_count(0.5)
    mids = np.array([p.position for p in res.point_sources])
    np.testing.assert_array_equal(mids, _KINDS[kind].segments(0.5).midpoints)


@pytest.mark.finding("10.79")
def test_rod_segments_split_at_layer_interface() -> None:
    rod = E.RodElectrode(name="r", position=(0, 0, 0.6), length=6.0)
    segs = rod.segments(0.5, layer_interfaces=(5.0,))
    tops = segs.midpoints[:, 2] - 0.5 * segs.lengths
    bottoms = segs.midpoints[:, 2] + 0.5 * segs.lengths
    assert not np.any((tops < 5.0 - 1e-12) & (bottoms > 5.0 + 1e-12))
    assert math.isclose(segs.total_length, 6.0, rel_tol=1e-12)


@pytest.mark.finding("10.79")
def test_solver_discretisation_dispatches_on_the_model() -> None:
    """The solver asks the model; no ``isinstance`` ladder over kinds."""
    from groundfield.geometry.primitives import WireSegments
    from groundfield.solver.image import _discretize_electrode

    class TwoSegmentWire(E.RodElectrode):
        def segments(self, ds, layer_interfaces=None):  # noqa: ANN001
            return WireSegments(
                np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
                np.array([1.0, 1.0]),
                np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            )

    segs = _discretize_electrode(
        TwoSegmentWire(name="w", position=(0, 0, 0.5), length=2.0), 0.1
    )
    assert [s.length for s in segs] == [1.0, 1.0]
