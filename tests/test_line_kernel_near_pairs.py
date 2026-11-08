"""Near-pair search of the line kernel (review pass 10, 10.28; audit round 1, A1-numerics F02).

The exact Galerkin integrals are evaluated for pairs closer than
``NEAR_FACTOR · max(L_i, L_j)``. The search used one radius — that of the
longest segment of the whole set — for every observer, so one conductor
discretised at 50 m made the candidate set (and the solve) quadratic in the
number of segments. Each segment now searches with its own radius; the pair
set is unchanged.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from groundfield.solver._line_kernel import (
    NEAR_FACTOR,
    LineSegments,
    near_pairs,
    reaction_corrections,
)

_FINDING = pytest.mark.finding("10.28")


def _segments(n_short: int, n_long: int, seed: int = 3) -> LineSegments:
    rng = np.random.default_rng(seed)
    mid = np.column_stack(
        [
            rng.uniform(0, 120, n_short),
            rng.uniform(0, 80, n_short),
            np.full(n_short, 0.8),
        ]
    )
    length = np.full(n_short, 0.25)
    tangent = np.tile([1.0, 0.0, 0.0], (n_short, 1))
    if n_long:
        mid = np.vstack([mid, [[60.0 + 35.0 * k, 40.0, 0.6] for k in range(n_long)]])
        length = np.concatenate([length, np.full(n_long, 35.0)])
        tangent = np.vstack([tangent, np.tile([1.0, 0.0, 0.0], (n_long, 1))])
    return LineSegments(mid, length, tangent, np.full(length.shape, 0.005))


def _brute_force(obs_mid, obs_len, src_mid, src_len):
    d = obs_mid[:, None, :] - src_mid[None, :, :]
    r2 = (d[..., 0] ** 2 + d[..., 1] ** 2) + d[..., 2] ** 2
    lim = NEAR_FACTOR * np.maximum.outer(obs_len, src_len)
    return np.nonzero(r2 < lim * lim)


@pytest.mark.parametrize("field_points", [False, True])
def test_near_pairs_are_the_pairs_of_the_definition(field_points: bool) -> None:
    """Guard: the faster search finds exactly the pairs it found before."""
    segs = _segments(400, 2)
    if field_points:
        obs_mid = np.column_stack(
            [np.linspace(-10, 200, 300), np.full(300, 40.2), np.zeros(300)]
        )
        obs_len = np.zeros(300)
    else:
        obs_mid, obs_len = segs.mid, segs.length
    i, j = near_pairs(obs_mid, obs_len, segs.mid, segs.length)
    bi, bj = _brute_force(obs_mid, obs_len, segs.mid, segs.length)
    np.testing.assert_array_equal(i, bi)  # sorted by observer, then source
    np.testing.assert_array_equal(j, bj)


@_FINDING
def test_one_long_segment_keeps_the_correction_memory_bounded() -> None:
    """3 000 short segments plus three 35 m ones: about 5 MiB, not 1.7 GiB."""
    reaction_corrections(_segments(200, 3), 1e-3)  # warm-up (lazy imports)
    segs = _segments(3000, 3)
    tracemalloc.start()
    corrections = reaction_corrections(segs, 1e-3)
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    assert corrections.rows.size < 60_000
    assert peak < 40.0, f"peak {peak:.1f} MiB"
