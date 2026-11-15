"""Memory of the ADR-0007 cross-layer path (review pass 10, 10.56).

Wave 3 of the 2026.11 work (stream w3xl, item 3). Up to this change the
cross-layer path held the reaction matrix, the layered correction and
several ``rows × cols`` temporaries of its largest depth group at once —
41 bytes per segment pair for 16 house grids (``N`` = 1920) against 18 on
the within-layer path — and it never switched to row blocks, whatever the
size of the world. A large depth group is now planned once from the range
of its distances and evaluated row chunk by row chunk (its entries are
those of the one-piece evaluation, bit for bit), and beyond
``MATRIX_BUDGET_BYTES`` the operator works in row blocks like the series
path. Every gate is a deterministic ``tracemalloc`` peak or a bitwise
comparison.
"""

from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pytest

import groundfield as gf

N_SIDE4 = 16 * 120  # segments of the 4 x 4 house world at ds = 0.5 m


def _houses(soil, n_side: int = 4) -> gf.World:
    """``n_side²`` 10 × 10 m house grids (2 × 2 meshes) at 0.8 m, 25 m apart."""
    w = gf.create_world(soil=soil)
    names = []
    for i in range(n_side):
        for j in range(n_side):
            name = f"h{i}{j}"
            gf.create_electrode(
                w,
                "grid_mesh",
                name=name,
                corner=(25.0 * i, 25.0 * j, 0.8),
                size=(10.0, 10.0),
                n_x=2,
                n_y=2,
            )
            names.append(name)
    for k, name in enumerate(names[1:], start=1):
        gf.create_conductor(
            w,
            name=f"c{k}",
            start=names[k - 1],
            end=name,
            conductor_type="pen",
            cross_section=50e-6,
            coupling_to_soil="isolated",
        )
    gf.create_source(w, attached_to=names[0], magnitude=1.0)
    return w


#: h_1 = 0.5 m puts every grid into the lower layer (cross-layer path);
#: h_1 = 5 m keeps them in the upper layer (Tagg/Sunde series).
CROSS = gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=0.5)
WITHIN = gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0)


def _solve(world: gf.World, kernel: str) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world.solve(gf.Engine(segment_length=0.5, kernel=kernel))


def _peak_mb(fn) -> float:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 2**20
    finally:
        tracemalloc.stop()


def _kernels():
    from groundfield.solver import _kernels

    return _kernels


@pytest.mark.finding("10.56")
@pytest.mark.parametrize("kernel", ["point", "line"])
def test_cross_layer_peak_is_that_of_the_within_layer_path(kernel) -> None:
    """The cross-layer solve needs no more than 1.4 times the within-layer peak.

    Measured on this machine (tracemalloc, N = 1920): within-layer 64.4 /
    65.8 MiB (point / line kernel); cross-layer 144.7 / 144.7 MiB before
    (2.2 times), 76.9 / 87.1 MiB now (1.19 / 1.32 times).
    """
    peaks = {}
    for name, soil in (("within", WITHIN), ("cross", CROSS)):
        w = _houses(soil)
        res = _solve(w, kernel)  # warm-up (imports, numba, caches)
        assert res.metadata["series_used"] is (name == "within")
        peaks[name] = _peak_mb(lambda w=w: _solve(w, kernel))
    assert peaks["cross"] < 1.4 * peaks["within"], peaks


@pytest.mark.finding("10.56")
@pytest.mark.parametrize("kernel", ["point", "line"])
def test_cross_layer_block_mode_beyond_the_budget(monkeypatch, kernel) -> None:
    """Above ``MATRIX_BUDGET_BYTES`` the cross-layer path holds no N × N matrix.

    The peak is 6.8 / 9.8 bytes per segment pair (point / line kernel; the
    within-layer path in block mode 3.0 / 8.8), the potentials agree with
    the materialised evaluation to BLAS round-off (1.2e-15 measured). Before,
    the path ignored the budget (41 bytes per pair).
    """
    w = _houses(CROSS)
    ref = _solve(w, kernel)
    monkeypatch.setattr(_kernels(), "MATRIX_BUDGET_BYTES", 0)
    monkeypatch.setattr(_kernels(), "BLOCK_BYTES", 1 << 20)
    res = _solve(w, kernel)
    peak = _peak_mb(lambda: _solve(w, kernel))
    assert peak < 12 * N_SIDE4 * N_SIDE4 / 2**20, f"peak {peak:.1f} MB"
    for name, u in ref.electrode_potentials.items():
        np.testing.assert_allclose(res.electrode_potentials[name], u, rtol=1e-13)
        np.testing.assert_allclose(
            res.electrode_currents[name],
            ref.electrode_currents[name],
            rtol=1e-12,
            atol=1e-15,
        )


@pytest.mark.finding("10.56")
def test_cross_layer_block_mode_grows_linearly(monkeypatch) -> None:
    """Block mode: 2.25 times the segments take 1.5 times the memory (not 5).

    9 and 36 house grids, N = 1920 and 4320: 23.7 and 34.8 MiB measured.
    """
    monkeypatch.setattr(_kernels(), "MATRIX_BUDGET_BYTES", 0)
    monkeypatch.setattr(_kernels(), "BLOCK_BYTES", 1 << 20)
    peaks = []
    for n_side in (4, 6):
        w = _houses(CROSS, n_side)
        _solve(w, "point")
        peaks.append(_peak_mb(lambda w=w: _solve(w, "point")))
    assert peaks[1] < 3.0 * peaks[0], peaks


@pytest.mark.finding("10.56")
@pytest.mark.parametrize("kernel", ["point", "line"])
def test_a_large_depth_group_in_chunks_is_bit_identical(monkeypatch, kernel) -> None:
    """Row chunks of a depth group give the entries of the one-piece evaluation.

    The 16 house grids are one depth group of 1920 × 1920 pairs; with
    ``BLOCK_BYTES`` = 64 KiB it is evaluated in 240 chunks of 8 rows (the
    interpolation is evaluated point by point).
    """
    from groundfield.solver import image_2layer
    from groundfield.solver._pipeline import SegmentArrays, discretise

    w = _houses(CROSS)
    eng = gf.Engine(segment_length=0.5, kernel=kernel)
    disc = discretise(w, eng, layer_interfaces=(CROSS.h_1,))
    arrays = SegmentArrays.from_segments(disc.segments)
    line = arrays.line_segments() if kernel == "line" else None

    def matrices():
        return image_2layer._layered_green_matrices(
            arrays.points, arrays.lengths, arrays.radii, soil=CROSS, line=line
        )

    one_piece = matrices()
    monkeypatch.setattr(_kernels(), "BLOCK_BYTES", 1 << 16)
    chunked = matrices()
    assert np.array_equal(chunked[0], one_piece[0])
    assert np.array_equal(chunked[1], one_piece[1])
