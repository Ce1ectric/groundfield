"""Sommerfeld earth-return correction of long segments (10.24).

A conductor that the topology keeps lumped is one inductive branch from
start to end; under ``sommerfeld`` its earth-return correction was one
16x16 Gauss–Legendre rule over the whole conductor, which cannot resolve
``exp(-gamma R)`` along a conductor much longer than the skin depth: at
1 kHz in 100 Ohm m (skin depth 159 m) the earth-return resistance of a
straight buried conductor was +1.7 % (1 km), +13.4 % (3 km), +34.9 % (5 km)
against the converged chain, and a 3 km PEN between two rods read +11 %
series resistance (audit round 2 of the calendar release, A2C-2). Since
then a segment longer than a quarter skin depth gets a composite outer
rule (16 nodes per piece) and a denser S(rho) table; shorter segments
keep the historic rule bit for bit.
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
