"""Continuous near/far switch of the line kernel (user decision D2 of 2026-11-15).

Wave 3 of the 2026.11 work (stream w3xl, item 5; audit W2B-01/-02). The
line kernel integrated pairs closer than six segment lengths exactly and
used its far form beyond — one step, at the distance where the pairs six
segments apart on a straight wire and, in two-layer soil, every segment's
own Tagg/Sunde image (an image distance of 2nh₁ or 2(nh₁ − z) equal to six
segment lengths) sit exactly. Round-off of the coordinates then decided
their form: the same world at different positions spread by up to 4.4e-7
(a 12 m strip, homogeneous) and 1e-5 ... 2.6e-5 in the impedance in
two-layer soil. The switch is now a blend: exact below 6, a C¹ weight down
to the far form at 7.5 segment lengths, for direct pairs, surface and
Tagg/Sunde images, the images of the cross-layer correction and field
points. Measured over 40 local offsets (0.1 m ... 8.8 km) and the UTM
origin (evidence ``item5_spread.py``): Z spreads by at most 8.6e-14 locally
and the UTM solve lies within 3.4e-11 of the local one, as with the point
kernel.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

UTM = (32_604_321.123, 5_812_345.678)
#: The bound of the point kernel in ``tests/test_projected_coordinates.py``
#: (the rounding of the input vertices at UTM, about 1e-9 m, gives 3e-11).
#: The line kernel shares it since the blend; it was 5e-7 before.
BOUND = 1e-9


def _strip_z(off, h_1, ds=0.5, soil=None):
    soil = soil or gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=h_1)
    w = gf.create_world(soil=soil)
    th = math.radians(23.0)
    gf.create_electrode(
        w,
        "strip",
        name="e",
        start=(off[0], off[1], 0.6),
        end=(off[0] + 12.0 * math.cos(th), off[1] + 12.0 * math.sin(th), 0.6),
    )
    gf.create_source(w, attached_to="e", magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = gf.create_engine(segment_length=ds, frequencies=[50.0])
        return eng.solve(w).cluster_impedance("e")[0]


#: Local shifts (0.1 k, 0.2 k) m, k = 1 ... 39 (k = 12, 17, 22, 27, 32, 37
#: flipped the form of the image pairs before the blend), and the UTM origin.
OFFSETS = [(0.1 * k, 0.2 * k) for k in range(1, 40)] + [UTM]


@pytest.mark.finding("10.28")
def test_w2b_01_image_pairs_on_the_line_kernel_switch_are_translation_invariant():
    """W2B-01/02: a two-layer image six segment lengths away no longer flips.

    2 (h_1 - z) = 2 (2.1 - 0.6) m = 3 m = 6 ds: every segment's own
    Tagg/Sunde image sat on the near/far boundary, whose far form for images
    was the plain point form. The impedance depended on the position by
    1.37e-5 (base 34ad269); now by 1.5e-11 (the UTM origin; 4e-14 locally).
    The test proposed by the audit, with the point kernel's bound.
    """
    # user decision D2
    z0 = _strip_z((0.0, 0.0), 2.1)
    worst = max(abs(_strip_z(off, 2.1) / z0 - 1.0) for off in OFFSETS)
    assert worst < BOUND


@pytest.mark.finding("10.28")
def test_a_straight_strip_is_translation_invariant_in_homogeneous_soil():
    """The direct pairs six segments apart (W2B-01: 4.4e-7 before, 3.5e-14 now)."""
    # user decision D2
    hom = gf.HomogeneousSoil(rho=100.0)
    z0 = _strip_z((0.0, 0.0), None, soil=hom)
    worst = max(abs(_strip_z(off, None, soil=hom) / z0 - 1.0) for off in OFFSETS)
    assert worst < BOUND


def _pair(distance: float, *, side_by_side: bool):
    """Two equal segments (L = 0.5 m, a = 5 mm) at a midpoint distance."""
    from groundfield.solver._line_kernel import LineSegments

    second = [0.0, distance, 1.0] if side_by_side else [distance, 0.0, 1.0]
    return LineSegments(
        np.array([[0.0, 0.0, 1.0], second]),
        np.array([0.5, 0.5]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([0.005, 0.005]),
    )


def _entry(segs) -> float:
    """Off-diagonal entry of the line-kernel half-space reaction matrix (1/m)."""
    from groundfield.solver._kernels import HalfSpaceKernel

    kernel = HalfSpaceKernel(4.0 * np.pi, line=segs)
    return float(kernel.matrix(segs.mid, segs.length, segs.radius)[0, 1])


@pytest.mark.finding("10.28")
@pytest.mark.parametrize(
    "side_by_side", [False, True], ids=["collinear", "side_by_side"]
)
def test_the_pair_value_is_continuous_at_six_segment_lengths(side_by_side) -> None:
    """At R = 6 L (1 ± 1e-12) the entry (direct pair and surface image) jumped
    by 2.8e-5 (collinear) and 9.7e-6 (side by side) between the exact and the
    far value (W2B-02 part A: 5.1e-5 / 1.8e-5 of the direct pair); now by
    round-off."""
    # user decision D2
    below = _entry(_pair(3.0 * (1.0 - 1e-12), side_by_side=side_by_side))
    above = _entry(_pair(3.0 * (1.0 + 1e-12), side_by_side=side_by_side))
    assert abs(above / below - 1.0) < 1e-11


@pytest.mark.finding("10.28")
def test_the_blend_weight_falls_smoothly_from_six_to_seven_and_a_half_lengths() -> None:
    from groundfield.solver._line_kernel import NEAR_FACTOR, SWITCH_FACTOR, _switch

    assert (NEAR_FACTOR, SWITCH_FACTOR) == (6.0, 7.5)
    lim = np.full(7, 6.0)
    r = np.array([5.9, 6.0 - 1e-9, 6.0, 6.0 + 1e-9, 6.75, 7.5, 8.0])
    w, far = _switch(r * r, lim)
    assert far.tolist() == [False, False, True, True, True, True, True]
    assert w[:2].tolist() == [1.0, 1.0]
    assert w[2] == 1.0 and 1.0 - w[3] < 1e-15  # zero slope at 6 L
    assert 0.0 < w[4] < 1.0
    assert w[5] == 0.0 and w[6] == 0.0
    # C¹ at 7.5 L: the weight vanishes quadratically
    w_end, _ = _switch(np.array([(7.5 * (1 - 1e-6)) ** 2]), np.array([6.0]))
    assert w_end[0] < 1e-10


@pytest.mark.finding("10.28")
def test_the_impedance_is_continuous_when_an_image_crosses_the_switch() -> None:
    """h_1 through 2.1 m puts every segment's image pair onto the switch.

    Before the blend Z changed by 1.4e-5 between h_1 = 2.1 (1 ∓ 1e-12) m (the
    image pairs switched from the exact to the point form); now it changes
    smoothly, by dZ/dh_1 · Δh_1.
    """
    # user decision D2
    z_lo = _strip_z((0.0, 0.0), 2.1 * (1.0 - 1e-12))
    z_hi = _strip_z((0.0, 0.0), 2.1 * (1.0 + 1e-12))
    assert abs(z_hi / z_lo - 1.0) < 1e-10


@pytest.mark.finding("10.28")
def test_the_cross_layer_images_are_continuous_at_the_switch() -> None:
    """The images of the cross-layer correction are blended too.

    A straight strip of 0.5 m segments d/2 above the interface with
    d = √35 · 0.5 m: each segment's neighbour lies exactly six segment
    lengths from its interface image (√(L² + d²) = 6 L). A rod through the
    interface puts the world onto the cross-layer path. Before the blend Z
    changed by 3.4e-5 between h_1 = 5 (1 ∓ 1e-12) m; now it is continuous.
    """
    # user decision D2
    d = math.sqrt(35.0) * 0.5

    def z_of(h_1: float) -> complex:
        w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=h_1))
        z = 5.0 - 0.5 * d
        gf.create_electrode(w, "strip", name="s", start=(0, 0, z), end=(10.0, 0, z))
        gf.create_electrode(w, "rod", name="rod", position=(-3.0, 0, 0.6), length=6.0)
        gf.create_conductor(w, name="b", start="s", end="rod", cross_section=50e-6)
        gf.create_source(w, attached_to="s", magnitude=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gf.Engine(segment_length=0.5).solve(w)
        assert res.metadata["series_used"] is False
        return res.cluster_impedance("s")[0]

    assert abs(z_of(5.0 * (1 + 1e-12)) / z_of(5.0 * (1 - 1e-12)) - 1.0) < 1e-10


@pytest.mark.finding("10.28")
def test_field_points_are_continuous_at_the_switch() -> None:
    """A field point six segment lengths from a segment midpoint (1.9e-6
    between the exact and the far value before)."""
    from groundfield.solver._guards import _MIN_DISTANCE
    from groundfield.solver._kernels import _line_field_correction, _point_field

    segs = _pair(0.0, side_by_side=False)
    from groundfield.solver._line_kernel import LineSegments

    one = LineSegments(segs.mid[:1], segs.length[:1], segs.tangent[:1], segs.radius[:1])

    def phi(d: float) -> float:
        pts = np.array([[0.0, d, 1.0]])
        val = _point_field(
            pts, one.mid, np.array([1.0 + 0j]), 4.0 * np.pi, _MIN_DISTANCE
        )
        corr = _line_field_correction(pts, one, [np.array([1.0])], _MIN_DISTANCE)
        return float(val.real[0] + corr[0][0])

    assert abs(phi(3.0 * (1 + 1e-12)) / phi(3.0 * (1 - 1e-12)) - 1.0) < 1e-11
