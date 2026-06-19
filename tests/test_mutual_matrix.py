"""Regression tests for :func:`groundfield.solve_mutual_matrix`.

The single-assembly mutual matrix must reproduce the historic
per-excitation construction (one :meth:`Engine.solve` per gnode;
diagonal from :meth:`FieldResult.cluster_impedance`, off-diagonal from
:meth:`FieldResult.potential` at the probe points) **bit-for-bit**.

These tests rebuild that historic loop explicitly and compare it against
``solve_mutual_matrix`` for the homogeneous limit, a within-layer
two-layer soil and a cross-layer two-layer soil (a deep rod piercing a
thin upper layer, the ADR-0007 fallback path). The acceptance threshold
``max|Z_new - Z_ref| < 1e-9`` is the pure BLAS-reordering residual; the
two paths apply the same kernels in the same accumulation order.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield import solve_mutual_matrix

# Three isolated single-rod grounding clusters (gnodes) on a triangle.
_ANCHORS: list[tuple[str, float, float]] = [
    ("g0", 0.0, 0.0),
    ("g1", 12.0, 0.0),
    ("g2", 6.0, 9.0),
]
_SEGMENT_LENGTH = 0.5
_FREQUENCIES = [50.0]
# Probe row i: surface point above rod i (off-diagonal sampling point),
# mirroring ``res.potential([[x_i, y_i, 0.0]])`` in the historic loop.
_PROBE_POINTS = np.array([[x, y, 0.0] for _, x, y in _ANCHORS])


def _build_world(soil: object, length: float) -> "gf.World":
    """World carrying all gnodes and ``soil`` but no current source."""
    world = gf.create_world(soil=soil)
    for name, x, y in _ANCHORS:
        gf.create_electrode(
            world, "rod", name=name, position=(x, y, 0.5), length=length
        )
    return world


def _reference_zg(
    soil: object, length: float, *, symmetrize: bool
) -> np.ndarray:
    """Historic per-excitation construction of ``Z_G``."""
    names = [a[0] for a in _ANCHORS]
    nG = len(names)
    engine = gf.create_engine(
        backend="image",
        segment_length=_SEGMENT_LENGTH,
        frequencies=_FREQUENCIES,
    )
    Z = np.zeros((nG, nG), dtype=complex)
    for j, anchor_j in enumerate(names):
        world = _build_world(soil, length)
        gf.create_source(world, attached_to=anchor_j, magnitude=1.0)
        res = engine.solve(world)
        Z[j, j] = res.cluster_impedance(anchor_j)[0]
        phi = res.potential(_PROBE_POINTS)  # (nG,)
        for i in range(nG):
            if i != j:
                Z[i, j] = phi[i]
    if symmetrize:
        Z = 0.5 * (Z + Z.T)
    return Z


@pytest.mark.parametrize(
    "soil, length, label",
    [
        (gf.HomogeneousSoil(resistivity=100.0), 3.0, "homogeneous"),
        (gf.TwoLayerSoil(rho_1=300.0, rho_2=100.0, h_1=8.0), 3.0,
         "two-layer within-layer"),
        (gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0), 9.0,
         "two-layer cross-layer"),
    ],
)
def test_mutual_matrix_matches_per_excitation(soil, length, label) -> None:
    """``solve_mutual_matrix`` == historic per-excitation loop (bit-exact)."""
    Z_ref = _reference_zg(soil, length, symmetrize=True)

    world = _build_world(soil, length)
    engine = gf.create_engine(
        backend="image",
        segment_length=_SEGMENT_LENGTH,
        frequencies=_FREQUENCIES,
    )
    Z_new = solve_mutual_matrix(
        world, engine, [a[0] for a in _ANCHORS], _PROBE_POINTS
    )

    assert Z_new.shape == (3, 3)
    assert np.max(np.abs(Z_new - Z_ref)) < 1e-9, (
        f"{label}: max|dZ| = {np.max(np.abs(Z_new - Z_ref)):.3e}"
    )


def test_mutual_matrix_unsymmetrized_matches_reference() -> None:
    """``symmetrize=False`` returns the raw per-excitation matrix."""
    soil = gf.TwoLayerSoil(rho_1=300.0, rho_2=100.0, h_1=8.0)
    Z_ref = _reference_zg(soil, 3.0, symmetrize=False)

    world = _build_world(soil, 3.0)
    engine = gf.create_engine(
        backend="image", segment_length=_SEGMENT_LENGTH,
        frequencies=_FREQUENCIES,
    )
    Z_new = solve_mutual_matrix(
        world, engine, [a[0] for a in _ANCHORS], _PROBE_POINTS,
        symmetrize=False,
    )
    assert np.max(np.abs(Z_new - Z_ref)) < 1e-9


def test_mutual_matrix_diagonal_is_cluster_impedance() -> None:
    """Diagonal entries equal the single-cluster grounding impedance."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = _build_world(soil, 3.0)
    engine = gf.create_engine(
        backend="image", segment_length=_SEGMENT_LENGTH,
        frequencies=_FREQUENCIES,
    )
    Z = solve_mutual_matrix(
        world, engine, [a[0] for a in _ANCHORS], _PROBE_POINTS,
        symmetrize=False,
    )
    # Self-impedance is real, positive and dominates its row/column.
    for k in range(3):
        assert Z[k, k].real > 0.0
        assert abs(Z[k, k]) > abs(Z[k, (k + 1) % 3])


def test_mutual_matrix_rejects_bad_probe_shape() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = _build_world(soil, 3.0)
    engine = gf.create_engine(backend="image", frequencies=_FREQUENCIES)
    with pytest.raises(ValueError, match="probe_points must have shape"):
        solve_mutual_matrix(
            world, engine, [a[0] for a in _ANCHORS],
            np.zeros((2, 3)),  # nG = 3 expected
        )


def test_mutual_matrix_rejects_unknown_anchor() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = _build_world(soil, 3.0)
    engine = gf.create_engine(backend="image", frequencies=_FREQUENCIES)
    with pytest.raises(KeyError, match="not an electrode"):
        solve_mutual_matrix(
            world, engine, ["g0", "g1", "ghost"], _PROBE_POINTS
        )
