"""Regression tests for :func:`groundfield.solve_mutual_field`.

``solve_mutual_field`` must reproduce, for **arbitrary** field points, the
potential of the historic per-excitation construction (one :meth:`Engine.solve`
per anchor with a unit current source; :meth:`FieldResult.potential` at the field
points). It must also agree, at the node positions, with the off-diagonal block of
:func:`solve_mutual_matrix`. Acceptance ``max|ΔR| < 1e-9`` is the pure
BLAS-reordering residual (same kernels, same accumulation order).
"""
from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield import solve_mutual_field, solve_mutual_matrix

_ANCHORS: list[tuple[str, float, float]] = [
    ("g0", 0.0, 0.0),
    ("g1", 12.0, 0.0),
    ("g2", 6.0, 9.0),
]
_SEGMENT_LENGTH = 0.5
_FREQUENCIES = [50.0]
# Field points deliberately DECOUPLED from the anchors: more points than nodes,
# arbitrary surface/buried positions (incl. one on a node, one far away).
_FIELD_POINTS = np.array(
    [[3.0, 0.0, 0.0], [8.0, 2.0, 0.0], [6.0, 9.0, 0.5], [20.0, -5.0, 0.0], [0.0, 0.0, 0.0]]
)


def _build_world(soil: object, length: float) -> "gf.World":
    world = gf.create_world(soil=soil)
    for name, x, y in _ANCHORS:
        gf.create_electrode(world, "rod", name=name, position=(x, y, 0.5), length=length)
    return world


def _reference_R(soil: object, length: float) -> np.ndarray:
    """Potential at the field points via one Engine.solve per excited anchor."""
    names = [a[0] for a in _ANCHORS]
    engine = gf.create_engine(
        backend="image", segment_length=_SEGMENT_LENGTH, frequencies=_FREQUENCIES
    )
    R = np.zeros((len(_FIELD_POINTS), len(names)), dtype=complex)
    for j, anchor_j in enumerate(names):
        world = _build_world(soil, length)
        gf.create_source(world, attached_to=anchor_j, kind="current", magnitude=1.0)
        res = engine.solve(world)
        R[:, j] = np.asarray(res.potential(_FIELD_POINTS, frequency_index=0)).ravel()
    return R


@pytest.mark.parametrize(
    "soil,length",
    [
        (gf.HomogeneousSoil(resistivity=100.0), 3.0),
        (gf.TwoLayerSoil(rho_1=200.0, rho_2=100.0, h_1=10.0), 3.0),
        (gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0), 9.0),  # ADR-0007 cross-layer
    ],
)
def test_mutual_field_matches_per_excitation(soil, length):
    engine = gf.create_engine(
        backend="image", segment_length=_SEGMENT_LENGTH, frequencies=_FREQUENCIES
    )
    world = _build_world(soil, length)
    anchors = [a[0] for a in _ANCHORS]
    R_new = solve_mutual_field(world, engine, anchors, _FIELD_POINTS)
    R_ref = _reference_R(soil, length)
    assert R_new.shape == (len(_FIELD_POINTS), len(anchors))
    assert np.max(np.abs(R_new - R_ref)) < 1e-9


def test_mutual_field_offdiag_matches_mutual_matrix():
    """At the node positions, R equals the off-diagonal of solve_mutual_matrix."""
    soil = gf.TwoLayerSoil(rho_1=200.0, rho_2=100.0, h_1=10.0)
    engine = gf.create_engine(
        backend="image", segment_length=_SEGMENT_LENGTH, frequencies=_FREQUENCIES
    )
    world = _build_world(soil, 3.0)
    anchors = [a[0] for a in _ANCHORS]
    probe = np.array([[x, y, 0.5] for _, x, y in _ANCHORS])
    Z = solve_mutual_matrix(world, engine, anchors, probe, symmetrize=False)
    R = solve_mutual_field(world, engine, anchors, probe)
    nG = len(anchors)
    off = [(i, j) for i in range(nG) for j in range(nG) if i != j]
    assert max(abs(R[i, j] - Z[i, j]) for i, j in off) < 1e-9
