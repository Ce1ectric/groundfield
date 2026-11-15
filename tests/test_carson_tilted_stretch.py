"""Carson tiling of tilted near-parallel pairs per stretch (audit W1R3-07).

``carson_series`` tiles Carson's per-metre correction over near-parallel
pairs by their axial overlap. A tilted pair took the per-metre value at its
mid-point distance for the whole overlap, so a lumped pair and its pieces
differed: a core crossing its shield's axis at mid-length, X of the loop at
1 kHz, 4.3e-4 at 2e-3 rad (10 m pieces) up to 5.2e-2 at 4e-2 rad. Where the
distance changes along the overlap the correction is now integrated over it;
pairs parallel up to round-off keep their single entry bit for bit.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.coupling import inductance as ind

pytestmark = pytest.mark.finding("10.46")

RC, RS, Z, LEN = 0.005, 0.015, 0.7, 100.0
OMEGA = [2.0 * math.pi * 1000.0]


def _pieces(p0, p1, n: int) -> np.ndarray:
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    pts = p0 + t * (p1 - p0)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _cross_sum(tilt: float, n_core: int, n_shield: int) -> complex:
    core = _pieces((0.0, -0.5 * LEN * tilt, Z), (LEN, 0.5 * LEN * tilt, Z), n_core)
    shield = _pieces((0.0, 0.0, Z), (LEN, 0.0, Z), n_shield)
    seg = np.concatenate([core, shield])
    radii = np.array([RC] * n_core + [RS] * n_shield)
    dz = ind.build_carson_correction_matrices(
        seg, radii, omegas=OMEGA, sigma_earth=0.01
    )[0]
    return complex(dz[:n_core, n_core:].sum())


# audit W1R3-07
@pytest.mark.parametrize(
    ("tilt", "piece"), [(2e-3, 10.0), (4e-3, 5.0), (2e-2, 5.0), (4e-2, 5.0)]
)
def test_a_tilted_pair_adds_up_in_pieces(tilt, piece) -> None:
    """Core-shield block of the Carson correction, lumped against pieces:
    within 5e-6 (measured 2.9e-6 at most; with 256 nodes per stretch 2e-9;
    base 3500ec9: the loop X 4.3e-4 ... 5.2e-2 apart)."""
    n = round(LEN / piece)
    lumped = _cross_sum(tilt, 1, 1)
    pieces = _cross_sum(tilt, n, n)
    assert pieces.real == pytest.approx(lumped.real, rel=5e-6)
    assert pieces.imag == pytest.approx(lumped.imag, rel=5e-6)


def test_parallel_pairs_keep_one_entry() -> None:
    """Pieces of two parallel conductors (directions equal up to round-off)
    and exactly parallel segments keep the single mid-point entry."""
    a = _pieces((0.0, 0.0, 0.7), (37.3, 11.1, 0.7), 7)
    b = _pieces((0.0, 0.5, 0.7), (37.3, 11.6, 0.7), 5)
    seg = np.concatenate([a, b])
    _, _, _, entries, _ = ind._carson_pair_plan(seg, np.full(len(seg), 0.006))
    pairs = [(i, j) for i, j, *_ in entries]
    assert len(pairs) == len(set(pairs))


# audit W1R3-07
def test_the_solved_loop_adds_up_in_pieces() -> None:
    """The coax loop of the audit (core crossing the shield's axis at 4e-2
    rad, bonded at the far end), X at 1 kHz under carson_series, lumped
    against 5 m pieces: within 1e-5 (base: +5.2e-2)."""

    def loop(ds):
        tilt = 4e-2
        w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
        gf.create_electrode(
            w, "rod", name="rod", position=(LEN + 1.0, 0.0, 0.5), length=1.0
        )
        c0, c1 = (0.0, -0.5 * LEN * tilt, Z), (LEN, 0.5 * LEN * tilt, Z)
        for name, pos in (
            ("A", c0),
            ("S", (0.0, 0.0, Z)),
            ("B", (LEN, 0.0, Z)),
            ("Bc", c1),
        ):
            gf.create_junction(w, name=name, position=pos)
        common = dict(
            resistivity=1.72e-8,
            coupling_to_soil="isolated",
            inductance_model="neumann",
            discretize_segment_length=ds,
        )
        w.add_conductor(
            gf.Conductor(
                name="core",
                start=c0,
                end=c1,
                start_electrode="A",
                end_electrode="Bc",
                wire_radius=RC,
                cross_section=50e-6,
                **common,
            )
        )
        w.add_conductor(
            gf.Conductor(
                name="shield",
                start=(0.0, 0.0, Z),
                end=(LEN, 0.0, Z),
                start_electrode="S",
                end_electrode="B",
                wire_radius=RS,
                cross_section=25e-6,
                conductor_type="cable_shield",
                **common,
            )
        )
        w.add_conductor(
            gf.Conductor(
                name="bond",
                start=(LEN, 0.0, Z),
                end=c1,
                start_electrode="B",
                end_electrode="Bc",
            )
        )
        w.add_conductor(
            gf.Conductor(
                name="tie",
                start=(LEN, 0.0, Z),
                end=(LEN + 1.0, 0.0, 0.5),
                start_electrode="B",
                end_electrode="rod",
            )
        )
        gf.create_source(w, attached_to="A", return_to="S", magnitude=1.0)
        eng = gf.create_engine(
            segment_length=1.0,
            frequencies=[1000.0],
            earth_inductive_model="carson_series",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            jp = eng.solve(w).metadata["junction_potentials"]
        return complex(jp["A"][0] - jp["S"][0]).imag

    assert loop(5.0) == pytest.approx(loop(None), rel=1e-5)
