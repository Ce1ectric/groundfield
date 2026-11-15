"""Sommerfeld earth-return correction of long segments (10.24).

A conductor that the topology keeps lumped is one inductive branch from
start to end; under ``sommerfeld`` its earth-return correction was one
16x16 Gauss–Legendre rule over the whole conductor, which cannot resolve
``exp(-gamma R)`` along a conductor much longer than the skin depth: at
1 kHz in 100 Ohm m (skin depth 159 m) the earth-return resistance of a
straight buried conductor was +1.7 % (1 km), +13.4 % (3 km), +34.9 % (5 km)
against the converged chain, and a 3 km PEN between two rods read +11 %
series resistance (audit round 2 of the calendar release, A2C-2). Since
then a segment longer than a quarter skin depth got a composite outer
rule (16 nodes per piece) and a denser S(rho) table; shorter segments
kept the historic rule, 4.8e-4 off at a quarter skin depth with a step of
3.5e-4 where the composite rule took over.

User decision Q34: the self term of every segment is the exact
one-dimensional reduction 2 int_0^L (L - u) g(u) du (graded 16-node panels,
cubic-spline S(rho)), within 2e-8 of a refined rule from 0.01 skin depths to
3 km at 50 Hz and 1 kHz and without a step (below 1e-9 where the rule
changes).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.conductors import Conductor
from groundfield.coupling.inductance import build_inductance_matrix
from groundfield.coupling.layered_green import SommerfeldResolutionWarning
from groundfield.coupling.sommerfeld_inductance import (
    LayeredEarth,
    build_sommerfeld_correction_matrices,
    build_sommerfeld_correction_matrix,
)

pytestmark = pytest.mark.finding("10.24")

EARTH = LayeredEarth(rhos=(100.0,), thicknesses=())
MU0 = 4e-7 * math.pi


def _chain(length: float, n: int, z: float = 0.7, y: float = 0.0) -> np.ndarray:
    xs = np.linspace(0.0, length, n + 1)
    return np.array([[[xs[k], y, z], [xs[k + 1], y, z]] for k in range(n)])


def _series_z(seg: np.ndarray, oms, method: str) -> list[complex]:
    radii = np.full(seg.shape[0], 0.00691)
    L = build_inductance_matrix(seg, radii, image="none")
    dZ = build_sommerfeld_correction_matrices(
        seg, radii, omegas=oms, earth=EARTH, image="none", method=method
    )
    return [complex((1j * om * L + dZ[k]).sum()) for k, om in enumerate(oms)]


# audit A2C-2
@pytest.mark.parametrize("length", [1000.0, 3000.0, 5000.0])
@pytest.mark.parametrize("method", ["exact", "tabulated"])
def test_lumped_conductor_matches_its_refined_chain(length: float, method) -> None:
    """One branch of 1-5 km against a chain of 50 m pieces (converged to
    1e-4 against 10 m pieces). Base (exact): earth-return resistance at
    1 kHz +1.7 % / +13.4 % / +34.9 %."""
    oms = [2.0 * math.pi * 50.0, 2.0 * math.pi * 1000.0]
    lumped = _series_z(_chain(length, 1), oms, method)
    chain = _series_z(_chain(length, int(length // 50)), oms, "tabulated")
    for k in range(len(oms)):
        # jw L + dZ carries no conductor resistance: the real part is the
        # earth-return resistance
        assert lumped[k].real == pytest.approx(chain[k].real, rel=3e-4)
        assert lumped[k].imag == pytest.approx(chain[k].imag, rel=1e-4)


# audit A2C-2
def test_long_parallel_conductors_couple_as_their_chains() -> None:
    """Mutual correction of two lumped 2 km conductors 1 m apart."""
    om = [2.0 * math.pi * 1000.0]

    def mutual(n: int, method: str) -> complex:
        seg = np.concatenate([_chain(2000.0, n), _chain(2000.0, n, y=1.0)])
        dZ = build_sommerfeld_correction_matrices(
            seg, None, omegas=om, earth=EARTH, image="none", method=method
        )[0]
        return complex(dZ[:n, n:].sum())

    ref = mutual(40, "tabulated")
    assert mutual(1, "exact") == pytest.approx(ref, rel=3e-4)
    assert mutual(1, "tabulated") == pytest.approx(ref, rel=3e-4)


def _two_rods_and_pen(length: float, dsl: float | None) -> gf.World:
    area = 150e-6
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="g2", position=(length, 0.0, 0.5), length=3.0)
    w.add_conductor(
        Conductor(
            name="pen",
            start=(0.0, 0.0, 0.7),
            end=(length, 0.0, 0.7),
            start_electrode="g1",
            end_electrode="g2",
            wire_radius=math.sqrt(area / math.pi),
            resistivity=2.82e-8,
            cross_section=area,
            discretize_segment_length=dsl,
            coupling_to_soil="isolated",
            inductance_model="neumann",
        )
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


# audit A2C-2
def test_lumped_3km_pen_matches_its_chain_in_a_solve() -> None:
    """Series impedance (V1 - V2) / I2 of a lumped 3 km PEN against its 150 m
    chain at 1 kHz (base: real part +11 %)."""

    def series_z(dsl: float | None) -> complex:
        eng = gf.create_engine(
            backend="image", frequencies=[1000.0], segment_length=0.5
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _two_rods_and_pen(3000.0, dsl).solve(eng)
        v1 = np.asarray(res.electrode_potentials["g1"])[0]
        v2 = np.asarray(res.electrode_potentials["g2"])[0]
        i2 = np.asarray(res.electrode_currents["g2"])[0]
        return complex((v1 - v2) / i2)

    lumped, chain = series_z(None), series_z(150.0)
    assert lumped.real == pytest.approx(chain.real, rel=1e-3)
    assert lumped.imag == pytest.approx(chain.imag, rel=1e-4)


# audit A2C-2
@pytest.mark.parametrize("method", ["exact", "tabulated"])
def test_each_frequency_equals_a_single_frequency_assembly(method: str) -> None:
    """The piece count depends on the frequency (50 Hz: one piece for
    600 m, 1 kHz: four); every frequency of a multi-frequency assembly still
    equals its single-frequency assembly bit for bit."""
    seg = np.concatenate([_chain(600.0, 1), _chain(600.0, 3, y=2.0)])
    seg = np.concatenate([seg, np.array([[[0.0, -3.0, -1.0], [40.0, -3.0, -1.0]]])])
    oms = [2.0 * math.pi * f for f in (50.0, 1000.0, 250.0)]
    multi = build_sommerfeld_correction_matrices(
        seg, None, omegas=oms, earth=EARTH, image="none", method=method
    )
    for k, om in enumerate(oms):
        single = build_sommerfeld_correction_matrix(
            seg, None, omega=om, earth=EARTH, image="none", method=method
        )
        assert np.array_equal(multi[k], single)


def test_piece_count_follows_the_skin_depth_of_the_most_conductive_layer() -> None:
    from groundfield.coupling.sommerfeld_inductance import _n_pieces

    om = 2.0 * math.pi * 1000.0
    delta = math.sqrt(2.0 * 100.0 / (om * MU0))
    assert _n_pieces(0.24 * delta, om, EARTH) == 1
    assert _n_pieces(0.26 * delta, om, EARTH) == 2
    assert _n_pieces(0.25 * delta, om, EARTH) == 1  # exactly a quarter: no split
    two_layer = LayeredEarth(rhos=(100.0, 10.0), thicknesses=(5.0,))
    delta_10 = math.sqrt(2.0 * 10.0 / (om * MU0))
    assert _n_pieces(1.01 * delta_10, om, two_layer) == 5
    assert _n_pieces(1e6, 0.0, EARTH) == 1


def test_an_extreme_segment_warns_instead_of_exhausting_memory() -> None:
    from groundfield.coupling.sommerfeld_inductance import _MAX_PIECES, _n_pieces

    om = 2.0 * math.pi * 5000.0
    earth = LayeredEarth(rhos=(10.0,), thicknesses=())
    with pytest.warns(SommerfeldResolutionWarning, match="capped"):
        n = _n_pieces(100e3, om, earth)
    assert n == _MAX_PIECES


_REFINED = {
    "_PIECE_SKIN_FRACTION": 1.0 / 128.0,
    "_RHO_NODE_SPACING": 0.005,
    "_SELF_T_SPACING": 0.005,
    "_GRADING_START": 1.0 / 16.0,
    "_FAR_WIDTHS": 160.0 * 32.0,  # uniform panels up to 40 skin depths
}


def _self_correction(seg, earth, f, *, refined: bool = False, method="exact"):
    """Self correction of one segment; ``refined``: panels of delta/128 up to
    40 skin depths, the first graded panel z/16 and S(rho) node spacings of
    0.005 (user decision
    Q34; the rule of 2026.11.0 was refined with pieces delta/32 and a spacing
    of 0.005, self-consistent to 2e-6)."""
    import groundfield.coupling.sommerfeld_inductance as si

    # (the rule of 2026.11.0 has only the first two constants)
    saved = {name: getattr(si, name) for name in _REFINED if hasattr(si, name)}
    if refined:
        for name in saved:
            setattr(si, name, _REFINED[name])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SommerfeldResolutionWarning)
            return build_sommerfeld_correction_matrices(
                seg,
                None,
                omegas=[2.0 * math.pi * f],
                earth=earth,
                image="none",
                method=method,
            )[0, 0, 0]
    finally:
        for name, value in saved.items():
            setattr(si, name, value)


# audit A3C-4: the stated accuracy holds where it was measured; user
# decision Q34 tightened it (2026.11.0: 6e-4 / 2.5e-4 for the overhead and
# two-layer cases, 2e-4 / 1e-4 buried in 100 Ohm m)
@pytest.mark.parametrize(
    ("name", "z", "rhos", "thicknesses"),
    [
        ("overhead_1km_10ohm", -10.0, (10.0,), ()),
        ("buried_1km_two_layer", 0.7, (1000.0, 30.0), (5.0,)),
        ("buried_1km_100ohm", 0.7, (100.0,), ()),
    ],
)
def test_long_segment_rule_meets_its_stated_accuracy(
    name, z, rhos, thicknesses
) -> None:
    """A lumped 1 km conductor at 1 kHz against the refined rule: within
    2e-8 (2026.11.0: 5.9e-4 / 2.4e-4 for a 3 km overhead line)."""
    seg = np.array([[[0.0, 0.0, z], [1000.0, 0.0, z]]])
    earth = LayeredEarth(rhos=rhos, thicknesses=thicknesses)
    got = _self_correction(seg, earth, 1000.0)
    ref = _self_correction(seg, earth, 1000.0, refined=True)
    assert got.real == pytest.approx(ref.real, rel=2e-8, abs=0)
    assert got.imag == pytest.approx(ref.imag, rel=2e-8, abs=0)


_EARTHS = {
    "buried_100ohm": (0.7, (100.0,), ()),
    "overhead_10ohm": (-10.0, (10.0,), ()),
    "buried_1000_over_30": (0.7, (1000.0, 30.0), (5.0,)),
    "buried_30_over_1000_h1_1m": (0.7, (30.0, 1000.0), (1.0,)),
}


# user decision Q34
@pytest.mark.parametrize("f", [50.0, 1000.0])
@pytest.mark.parametrize("name", list(_EARTHS))
@pytest.mark.parametrize("fraction", [0.01, 0.1, 0.25, 1.0, 5.0])
def test_the_self_correction_is_within_its_bound_of_a_refined_rule(
    name, f, fraction
) -> None:
    """0.01 ... 5 skin depths at 50 Hz and 1 kHz: within 2e-8 of the refined
    rule (measured worst 1.7e-8, 30 over 1000 Ohm m with h1 = 1 m; an
    independent one-dimensional reduction with S(rho) summed at every node
    agrees to the same level, evidence q34_bound_scan). Base 3500ec9: up to
    4.8e-4 at a quarter skin depth (buried, 100 Ohm m), 1.5e-3 at 50 Hz in
    30 over 1000 Ohm m."""
    z, rhos, thicknesses = _EARTHS[name]
    earth = LayeredEarth(rhos=rhos, thicknesses=thicknesses)
    delta = math.sqrt(2.0 * min(rhos) / (2.0 * math.pi * f * MU0))
    length = min(fraction * delta, 3000.0)
    seg = np.array([[[0.0, 0.0, z], [length, 0.0, z]]])
    got = _self_correction(seg, earth, f, method="tabulated")
    ref = _self_correction(seg, earth, f, refined=True)
    assert got.real == pytest.approx(ref.real, rel=2e-8, abs=0)
    assert got.imag == pytest.approx(ref.imag, rel=2e-8, abs=0)


# audit A3C-4, user decision Q34
@pytest.mark.parametrize("name", ["buried_100ohm", "overhead_10ohm"])
def test_the_self_correction_has_no_step_at_a_quarter_skin_depth(name) -> None:
    """Where the composite rule used to take over (L = delta/4, 1 kHz) the
    self correction steps by less than 1e-9 once the smooth change over the
    interval is taken out (base 3500ec9: 3.5e-4 in the resistance and
    1.2e-5 in the reactance, buried in 100 Ohm m; measured now 1.3e-11 and,
    at every other length where the rule changes, at most 5.4e-10)."""
    z, rhos, thicknesses = _EARTHS[name]
    earth = LayeredEarth(rhos=rhos, thicknesses=thicknesses)
    delta = math.sqrt(2.0 * min(rhos) / (2.0 * math.pi * 1000.0 * MU0))

    def at(scale: float) -> complex:
        length = 0.25 * delta * scale
        return _self_correction(
            np.array([[[0, 0, z], [length, 0, z]]]), earth, 1000.0, method="tabulated"
        )

    lo2, lo, hi = at(1.0 - 3e-7), at(1.0 - 1e-7), at(1.0 + 1e-7)
    jump = (hi - lo) - (lo - lo2)  # minus the smooth increment
    assert abs(jump.real / lo.real) < 1e-9
    assert abs(jump.imag / lo.imag) < 1e-9
