"""Regression tests for the vectorised partial-inductance assembly.

Per ADR-0010 (Tier 0b) the new vectorised
:func:`build_inductance_matrix` must reproduce the legacy loop
implementation :func:`_build_inductance_matrix_loop` to floating-
point precision on every geometric class used by the existing
test suite.

Tolerance: ``rtol=1e-10, atol=1e-18`` on every matrix entry. The
remaining drift is pure float roundoff in the order in which the
sums are accumulated.
"""

from __future__ import annotations

import numpy as np
import pytest

from groundfield.coupling.inductance import (
    _build_inductance_matrix_loop,
    build_inductance_matrix,
)


# ---------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------


def _segments_random(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``n`` random non-degenerate segments in a 100 m cube."""
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(-50.0, 50.0, size=(n, 3))
    p2 = p1 + rng.uniform(0.5, 5.0, size=(n, 3)) * rng.choice(
        [-1, 1], size=(n, 3),
    )
    return np.stack([p1, p2], axis=1), np.full(n, 0.005)


def _segments_parallel_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` parallel horizontal segments — fully covers the parallel
    closed-form code path."""
    p1 = np.array([[0.0, k * 1.0, 0.6] for k in range(n)])
    p2 = np.array([[10.0, k * 1.0, 0.6] for k in range(n)])
    return np.stack([p1, p2], axis=1), np.full(n, 0.005)


def _segments_mixed() -> tuple[np.ndarray, np.ndarray]:
    """Hand-picked mix: parallel, anti-parallel, vertical, oblique,
    coaxial-touching, far-distance."""
    seg = np.array([
        [[0.0, 0.0, 0.6], [10.0, 0.0, 0.6]],   # horizontal A
        [[10.0, 0.0, 0.6], [0.0, 0.0, 0.6]],   # anti-parallel co-axial-ish
        [[0.0, 1.0, 0.6], [10.0, 1.0, 0.6]],   # parallel, 1 m apart
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],    # vertical rod
        [[0.5, 0.5, 0.0], [3.0, 4.0, 0.6]],    # oblique 3-D
        [[20.0, 0.0, 0.6], [30.0, 0.0, 0.6]],  # far parallel
    ], dtype=float)
    return seg, np.array([0.005, 0.005, 0.005, 0.008, 0.005, 0.005])


# ---------------------------------------------------------------------
# Bit-exact regression
# ---------------------------------------------------------------------


@pytest.mark.parametrize("use_image", [True, False])
def test_mixed_geometry_matches_loop(use_image: bool) -> None:
    seg, radii = _segments_mixed()
    L_loop = _build_inductance_matrix_loop(seg, radii, use_image=use_image)
    L_vec = build_inductance_matrix(seg, radii, use_image=use_image)
    np.testing.assert_allclose(L_vec, L_loop, rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("n", [5, 20, 60])
@pytest.mark.parametrize("use_image", [True, False])
def test_random_segments_match_loop(n: int, use_image: bool) -> None:
    seg, radii = _segments_random(n, seed=42)
    L_loop = _build_inductance_matrix_loop(seg, radii, use_image=use_image)
    L_vec = build_inductance_matrix(seg, radii, use_image=use_image)
    np.testing.assert_allclose(L_vec, L_loop, rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("n", [3, 10, 30])
def test_parallel_grid_matches_loop(n: int) -> None:
    """All-parallel geometry — exercises the closed-form path
    exclusively."""
    seg, radii = _segments_parallel_grid(n)
    L_loop = _build_inductance_matrix_loop(seg, radii, use_image=True)
    L_vec = build_inductance_matrix(seg, radii, use_image=True)
    np.testing.assert_allclose(L_vec, L_loop, rtol=1e-10, atol=1e-18)


# ---------------------------------------------------------------------
# Symmetry properties
# ---------------------------------------------------------------------


def test_matrix_is_symmetric() -> None:
    """Partial inductance is symmetric by construction."""
    seg, radii = _segments_random(15, seed=1)
    L = build_inductance_matrix(seg, radii, use_image=True)
    np.testing.assert_allclose(L, L.T, rtol=1e-12, atol=1e-18)


def test_diagonal_is_positive() -> None:
    """Self-inductance is positive."""
    seg, radii = _segments_random(15, seed=1)
    L = build_inductance_matrix(seg, radii, use_image=True)
    assert (np.diag(L) > 0).all()


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


def test_single_segment() -> None:
    seg = np.array([[[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]]], dtype=float)
    radii = np.array([0.005])
    L_loop = _build_inductance_matrix_loop(seg, radii, use_image=True)
    L_vec = build_inductance_matrix(seg, radii, use_image=True)
    np.testing.assert_allclose(L_vec, L_loop, rtol=1e-12, atol=1e-18)
    assert L_vec.shape == (1, 1)


def test_zero_segments() -> None:
    """Empty input returns a 0×0 matrix."""
    seg = np.zeros((0, 2, 3), dtype=float)
    radii = np.zeros(0, dtype=float)
    L = build_inductance_matrix(seg, radii)
    assert L.shape == (0, 0)


def test_zero_length_segment_raises() -> None:
    seg = np.array([
        [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],   # zero-length
    ], dtype=float)
    radii = np.array([0.005])
    with pytest.raises(ValueError, match="positive"):
        build_inductance_matrix(seg, radii)
