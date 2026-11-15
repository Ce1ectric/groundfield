"""Close segment pairs and long conductors under ``sommerfeld``.

User decision Q34: along the line of closest approach of two segments that
come close (wires beside each other, a core in its shield, a tilted core)
the integrand of the earth-return correction has the kink of the self term,
which the 16x16 tensor rule cannot follow: 3.9e-4 (100 Ohm m) to 1.5e-3
(30 over 1000 Ohm m, h1 = 1 m) of the resistance of two wires 1 mm apart at
a quarter skin depth. Such pairs get a graded rule along that line; it agrees
with the one-dimensional self rule where the two segments coincide.

Audit W2B-05: one lumped conductor of 2 km between a rod and a junction took
28 s and 1.6 GB (0.42 / 4.3 s at 0.25 / 1 km, about L^2.7); its pieces sat at
different depths and were integrated pair by pair, each with its own lambda
grid, and the grid cache kept 512 of them.
"""

from __future__ import annotations

import math
import time
import tracemalloc
import warnings

import numpy as np
import pytest

import groundfield as gf
import groundfield.coupling.sommerfeld_inductance as si
from groundfield.coupling.sommerfeld_inductance import (
    LayeredEarth,
    build_sommerfeld_correction_matrices,
)

pytestmark = pytest.mark.finding("10.24")

MU0 = 4e-7 * math.pi
HOM = LayeredEarth(rhos=(100.0,), thicknesses=())
LAYERED = LayeredEarth(rhos=(30.0, 1000.0), thicknesses=(1.0,))
_REFINED = {
    "_PIECE_SKIN_FRACTION": 1.0 / 32.0,
    "_SELF_T_SPACING": 0.005,
    "_GRADING_START": 1.0 / 16.0,
    "_FAR_WIDTHS": 160.0 * 8.0,
}


def _delta(earth: LayeredEarth, f: float) -> float:
    return math.sqrt(2.0 * min(earth.rhos) / (2.0 * math.pi * f * MU0))


def _matrix(seg, earth, f, *, method="tabulated", refined=False) -> np.ndarray:
    saved = {name: getattr(si, name) for name in _REFINED if hasattr(si, name)}
    if refined:
        for name in saved:
            setattr(si, name, _REFINED[name])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", si.SommerfeldResolutionWarning)
            return build_sommerfeld_correction_matrices(
                np.asarray(seg, dtype=float),
                None,
                omegas=[2.0 * math.pi * f],
                earth=earth,
                image="none",
                method=method,
            )[0]
    finally:
        for name, value in saved.items():
            setattr(si, name, value)


def _pair(name: str, length: float, z: float = 0.7) -> np.ndarray:
    a = [[0.0, 0.0, z], [length, 0.0, z]]
    b = {
        "beside_1mm": [[0.0, 1e-3, z], [length, 1e-3, z]],
        "beside_20cm": [[0.0, 0.2, z], [length, 0.2, z]],
        "beside_1m_shifted": [[0.3 * length, 1.0, z], [1.3 * length, 1.0, z]],
        "tilted_1e-4": [[0.0, 0.0, z], [length, 1e-4 * length, z]],
    }[name]
    return np.array([a, b])


# user decision Q34
@pytest.mark.parametrize("method", ["exact", "tabulated"])
@pytest.mark.parametrize("earth", [HOM, LAYERED], ids=["100", "30_over_1000"])
@pytest.mark.parametrize("f", [50.0, 1000.0])
@pytest.mark.parametrize(
    "name", ["beside_1mm", "beside_20cm", "beside_1m_shifted", "tilted_1e-4"]
)
def test_a_close_pair_is_within_its_bound_of_a_refined_rule(
    name, f, earth, method
) -> None:
    """Mutual correction of two segments a quarter skin depth long: within
    1e-8 of the refined graded rule (measured worst 2.7e-9). Base 3500ec9
    (16x16 tensor rule): 3.4e-4 / 3.9e-4 in 100 Ohm m and 1.5e-3 in 30 over
    1000 Ohm m for wires 1 mm apart and the tilted core at 50 Hz / 1 kHz."""
    seg = _pair(name, 0.25 * _delta(earth, f))
    got = _matrix(seg, earth, f, method=method)[0, 1]
    ref = _matrix(seg, earth, f, method="exact", refined=True)[0, 1]
    assert got.real == pytest.approx(ref.real, rel=1e-8, abs=0)
    assert got.imag == pytest.approx(ref.imag, rel=1e-8, abs=0)


# user decision Q34
@pytest.mark.parametrize("f", [50.0, 1000.0])
def test_the_close_rule_and_the_self_rule_agree_on_one_segment(f) -> None:
    """The graded close-pair rule, given one segment twice, and the
    one-dimensional self rule are two methods for the same integral."""
    length = 0.25 * _delta(LAYERED, f)
    a1, a2 = np.array([0.0, 0.0, 0.7]), np.array([length, 0.0, 0.7])
    om = [2.0 * math.pi * f]
    close = si._close_pair_values(a1, a2, a1, a2, 1.0, om, LAYERED)[0]
    self_ = si._self_values(a1, a2, om, LAYERED)[0]
    assert close.real == pytest.approx(self_.real, rel=5e-9, abs=0)
    assert close.imag == pytest.approx(self_.imag, rel=5e-9, abs=0)


# user decision Q34
@pytest.mark.parametrize("method", ["exact", "tabulated"])
def test_a_core_on_the_shield_axis_has_no_earth_return_in_the_loop(method) -> None:
    """Core and shield as two lumped conductors on one axis: the earth-return
    correction of the loop, dZ_cc + dZ_ss - 2 dZ_cs, vanishes (both segments
    take the self rule), at a length beyond a quarter skin depth."""
    seg = np.array([[[0.0, 0.0, 0.7], [100.0, 0.0, 0.7]]] * 2)
    dz = _matrix(seg, HOM, 1000.0, method=method)
    loop = dz[0, 0] + dz[1, 1] - 2.0 * dz[0, 1]
    assert abs(loop) <= 1e-15 * abs(dz[0, 0])


# user decision Q34
def test_the_close_rule_does_not_depend_on_order_or_direction() -> None:
    a1, a2 = np.array([0.0, 0.0, 0.7]), np.array([40.0, 0.0, 0.7])
    b1, b2 = np.array([3.0, 0.2, 0.7]), np.array([50.0, 0.25, 0.7])
    om = [2.0 * math.pi * 1000.0]
    v = si._close_pair_values(a1, a2, b1, b2, 1.0, om, HOM)[0]
    assert si._close_pair_values(b1, b2, a1, a2, 1.0, om, HOM)[0] == v
    assert si._close_pair_values(a2, a1, b2, b1, 1.0, om, HOM)[0] == v


def _lumped_world(length: float) -> gf.World:
    """The set-up of audit W2B-05 (script b3)."""
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="h", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_junction(w, name="cab", position=(length, 0.0, 0.7))
    gf.create_conductor(
        w,
        name="pen",
        start="h",
        end="cab",
        cross_section=95e-6,
        inductance_model="neumann",
        wire_radius=0.006,
    )
    gf.create_source(w, attached_to="cab", magnitude=1.0)
    return w


# audit W2B-05 (the auditor's proposed test)
@pytest.mark.slow
def test_w2b_05_a_long_inductive_conductor_is_assembled_in_bounded_time() -> None:
    """One lumped Neumann conductor of 2 km between a rod and a junction:
    28 s and 1.6 GB on the base; measured now 1.1 s and 0.21 GB (peak RSS of
    the whole solve)."""
    w = _lumped_world(2000.0)
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gf.create_engine(segment_length=0.5, frequencies=[50.0, 1000.0]).solve(w)
    assert time.perf_counter() - t < 5.0


def _assembly_cost(length: float) -> tuple[int, int]:
    """(J0 entries, traced peak bytes) of the earth-return assembly of the
    W2B-05 conductor at 50 Hz and 1 kHz."""
    count = [0]
    j0 = si.j0

    def counting(x):
        count[0] += int(np.size(x))
        return j0(x)

    seg = np.array([[[0.0, 0.0, 0.5], [length, 0.0, 0.7]]])
    si.j0 = counting
    tracemalloc.start()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", si.SommerfeldResolutionWarning)
            build_sommerfeld_correction_matrices(
                seg,
                None,
                omegas=[2.0 * math.pi * 50.0, 2.0 * math.pi * 1000.0],
                earth=HOM,
                image="none",
                method="tabulated",
            )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        si.j0 = j0
    return count[0], peak


# audit W2B-05
def test_the_cost_of_a_lumped_conductor_grows_at_most_linearly() -> None:
    """Bessel evaluations of the assembly, 0.5 / 1 / 2 km: base 3.0e7 /
    2.1e8 / 1.5e9 (7x per doubling, L^2.8) and 0.53 GiB traced at 2 km;
    measured now 4.9e6 / 1.1e7 / 2.4e7 (2.2x per doubling: the lambda grid
    grows with L, the table with ln L) and 53 MiB."""
    n05, _ = _assembly_cost(500.0)
    n1, _ = _assembly_cost(1000.0)
    n2, peak = _assembly_cost(2000.0)
    assert n1 / n05 < 2.5
    assert n2 / n1 < 2.5
    assert peak < 150 * 2**20


# audit W2B-05
def test_the_grid_caches_stay_bounded() -> None:
    """Large lambda grids go to a cache of eight entries."""
    for length in (
        3000.0,
        3100.0,
        3200.0,
        3300.0,
        3400.0,
        3500.0,
        3600.0,
        3700.0,
        3800.0,
        3900.0,
    ):
        _matrix(np.array([[[0.0, 0.0, 0.7], [length, 0.0, 0.7]]]), HOM, 1000.0)
    assert si._grid_large.cache_info().currsize <= si._LARGE_GRID_CACHE
    assert si._grid_small.cache_info().maxsize == 512
