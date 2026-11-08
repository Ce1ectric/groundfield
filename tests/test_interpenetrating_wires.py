"""Wires lying side by side: interpenetrating parallel segments.

Two near-parallel segments of different conductors whose axes stay closer
than the sum of their radii over a common length — coincident leads, a
service drop routed back along its PEN main, two cables in one trench at a
too small offset — have no finite filament mutual inductance: the 16x16
Gauss–Legendre rule only stopped at its 1e-6 m floor. Two coincident 50 m
leads read +587j Ohm at 50 Hz (stream GEN-2), and a TN network whose drops
ran back along the PEN main had a negative input reactance at 1 kHz
(-0.633 Ohm, stream DOC-B): the inductance matrix was not positive
definite. Since audit round 2 of the calendar release such pairs are
evaluated as touching wires (axis distance r_a + r_b), and the Carson
tiling gives them the mutual earth-return correction of touching wires
(it used to skip them as "the same wire").
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.conductors import Conductor
from groundfield.coupling.carson import carson_mutual_correction
from groundfield.coupling.inductance import (
    _build_inductance_matrix_loop,
    build_carson_correction_matrices,
    build_inductance_matrix,
    parallel_segments_mutual,
)

pytestmark = pytest.mark.finding("10.46")


def _chain(a, b, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pts = a + np.linspace(0.0, 1.0, n + 1)[:, None] * (b - a)
    return np.stack([pts[:-1], pts[1:]], axis=1)


# stream GEN-2 request / DOC-B
def test_coincident_leads_have_the_mutual_of_touching_wires() -> None:
    """Two coincident 50 m leads (5 m segments, r = 5 mm): the mutual
    inductance of two 50 m wires touching side by side (base: 1.9 H, the
    quadrature floor)."""
    lead = _chain((0.0, 0.0, -0.1), (50.0, 0.0, -0.1), 10)
    seg = np.concatenate([lead, lead])
    L = build_inductance_matrix(seg, np.full(20, 0.005), image="none")
    mutual = L[:10, 10:].sum()
    # Segments of the two leads that meet end to end keep the exact coaxial
    # closed form (as chain neighbours do), O(1e-9 H) each from the 1 cm
    # offset of touching wires: 2e-4 of the total.
    assert mutual == pytest.approx(parallel_segments_mutual(50.0, 0.01), rel=1e-3)


# stream DOC-B
@pytest.mark.parametrize("image", ["none", "additive", "antiparallel"])
def test_drops_along_the_pen_main_keep_the_matrix_positive_definite(image) -> None:
    """A PEN main and two service drops routed back along it on the same
    axis and depth (the DOC-B network): positive definite in every image
    convention."""
    seg = np.concatenate(
        [
            _chain((0.0, 0.0, 0.6), (-200.0, 0.0, 0.6), 20),
            _chain((-200.0, 0.0, 0.6), (-20.0, 0.0, 0.6), 18),
            _chain((-200.0, 0.0, 0.6), (20.0, 0.0, 0.6), 22),
        ]
    )
    L = build_inductance_matrix(seg, np.full(len(seg), 0.0069), image=image)
    assert np.linalg.eigvalsh(L).min() > 0.0


def _overlap_world() -> gf.World:
    """Substation rod, a PEN main to a cabinet 200 m away and two service
    drops from the cabinet back along the main (10 m chains on one axis
    and depth, as the TN generator built them for two houses)."""
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    for name, x, y in (("s", 0.0, 0.0), ("kvs", -200.0, 0.0), ("h0", -20.0, 30.0)):
        gf.create_electrode(w, "rod", name=name, position=(x, y, 0.0), length=2.0)
    gf.create_electrode(w, "rod", name="h1", position=(20.0, 30.0, 0.0), length=2.0)
    area = 150e-6
    for name, a, b, xa, xb in (
        ("pen_main", "s", "kvs", 0.0, -200.0),
        ("drop_0", "kvs", "h0", -200.0, -20.0),
        ("drop_1", "kvs", "h1", -200.0, 20.0),
    ):
        w.add_conductor(
            Conductor(
                name=name,
                start=(xa, 0.0, 0.6),
                end=(xb, 0.0, 0.6),
                start_electrode=a,
                end_electrode=b,
                wire_radius=math.sqrt(area / math.pi),
                resistivity=2.82e-8,
                cross_section=area,
                discretize_segment_length=10.0,
                coupling_to_soil="isolated",
                inductance_model="neumann",
            )
        )
    gf.create_source(w, attached_to="s", magnitude=1.0)
    return w


# stream DOC-B
@pytest.mark.parametrize("model", ["perfect_mirror", "sommerfeld", "carson_series"])
def test_overlapping_drops_give_a_passive_input_impedance(model: str) -> None:
    """Base: Im Z_in(1 kHz) < 0 with perfect_mirror and sommerfeld."""
    eng = gf.create_engine(
        backend="image",
        segment_length=1.0,
        frequencies=[50.0, 1000.0],
        earth_inductive_model=model,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = np.asarray(eng.solve(_overlap_world()).input_impedance())
    assert (z.imag > 0.0).all()


# stream DOC-B
def test_carson_tiling_couples_overlapping_coaxial_conductors() -> None:
    """Two conductors overlapping on one axis over 6 m get the Carson
    mutual of touching wires per metre of overlap (base: zero)."""
    seg = np.array(
        [
            [[0.0, 0.0, 0.7], [10.0, 0.0, 0.7]],
            [[4.0, 0.0, 0.7], [20.0, 0.0, 0.7]],
        ]
    )
    om = 2.0 * math.pi * 1000.0
    dZ = build_carson_correction_matrices(
        seg, np.full(2, 0.005), omegas=[om], sigma_earth=0.01
    )[0]
    expected = 6.0 * carson_mutual_correction(om, 0.7, 0.7, 0.01, 0.01)
    assert dZ[0, 1] == pytest.approx(expected, rel=1e-12)


def test_wires_at_the_touching_distance_are_unchanged() -> None:
    """At an axis distance of exactly r_a + r_b the closed form of the
    parallel pair is kept bit for bit (the guard is continuous)."""
    a = _chain((0.0, 0.0, 0.7), (20.0, 0.0, 0.7), 4)
    b = _chain((0.0, 0.01, 0.7), (20.0, 0.01, 0.7), 4)
    seg = np.concatenate([a, b])
    with_guard = build_inductance_matrix(seg, np.full(8, 0.005), image="none")
    no_guard = build_inductance_matrix(seg, np.full(8, 1e-9), image="none")
    assert np.array_equal(with_guard[:4, 4:], no_guard[:4, 4:])


def test_loop_reference_applies_the_same_guard() -> None:
    lead = _chain((0.0, 0.0, 0.5), (12.0, 0.0, 0.5), 3)
    other = _chain((3.0, 0.002, 0.5), (9.0, 0.002, 0.5), 2)  # 2 mm off axis
    seg = np.concatenate([lead, other, _chain((0, 3, 0.5), (12, 3, 0.5), 2)])
    radii = np.full(len(seg), 0.004)
    for image in ("none", "additive", "antiparallel"):
        L_vec = build_inductance_matrix(seg, radii, image=image)
        L_loop = _build_inductance_matrix_loop(seg, radii, image=image)
        np.testing.assert_allclose(L_vec, L_loop, rtol=1e-12, atol=1e-20)
    # the pair 2 mm apart is evaluated as touching wires, 8 mm apart
    from groundfield.coupling.inductance import _parallel_filaments_mutual

    L = build_inductance_matrix(seg, radii, image="none")
    moved = seg[3].copy()
    moved[:, 1] = 0.008
    expected = _parallel_filaments_mutual(seg[1, 0], seg[1, 1], moved[0], moved[1])
    assert L[1, 3] == pytest.approx(expected, rel=1e-12)
