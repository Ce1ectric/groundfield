"""Image pairs of the line kernel use the transverse distance (audit W3A-02).

The line kernel regularised every segment pair with the wire radius, the
reduced kernel ``1/sqrt(|x - y|² + a²)`` with ``a² = (a_i² + a_j²)/2``, images
included, and its far form carried the matching radius term
``-a²/(2R³)``. For a segment's own surface that is exact; for an image line
that passes a wire at a transverse distance ``d >= a`` it is not: by the 2-D
mean-value property the average of the image line's potential over the
wire's circumference is its value on the axis, so the reduced kernel left out
``a²/(2d²)`` of every near image — 8 % of the interface image of a wire 1 cm
above the interface. A ring 1 cm above a 1000/30 Ohm m interface read
+1.60 % against its exact thin-ring value (w4axi's finding, +1.74 % before
wave 3).

The line potential of an image segment at a point of the observer now uses
``max(rho_perp, a)``, ``rho_perp`` the distance of the point from the image's
axis: the plain kernel for parallel images (``rho_perp >= a``) and the reduced
kernel for collinear ones (``rho_perp = 0``: a rod against its own image, where
every point of the surface circle is ``sqrt(r² + a²)`` from the source).
``ln max(rho, a)`` is the exact circle average of the 2-D line potential for
any ``rho``, so one formula covers both. The far form of images carries no
radius term. The soil surface image keeps the reduced kernel (see the module
docstring of :mod:`groundfield.solver._line_kernel`).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

pytestmark = pytest.mark.finding("10.28")


def _thin_ring_exact(rho1, rho2, h1, r0, t, a, n_theta=64):
    """Thin ring in layer 1 by its Tagg/Sunde image rings (auditor W3A, e7).

    Uniform current on the axis circle; the potential of every image ring
    (``(2/pi) K(m) / sqrt(...)``, elliptic ``K``) averaged over ``n_theta``
    points of the wire surface. Reproduces w4axi's Hankel reference to 2e-6.
    """
    from scipy.special import ellipk

    K = (rho2 - rho1) / (rho2 + rho1)
    imgs = [(1.0, t), (1.0, -t)]
    n = 1
    while abs(K) ** n > 1e-18:
        q = 2.0 * n * h1
        imgs += [(K**n, t + q), (K**n, t - q), (K**n, q - t), (K**n, -q - t)]
        n += 1
    th = (np.arange(n_theta) + 0.5) * 2.0 * math.pi / n_theta
    rho_s, z_s = r0 + a * np.cos(th), t + a * np.sin(th)
    tot = 0.0
    for w, zi in imgs:
        den2 = (r0 + rho_s) ** 2 + (z_s - zi) ** 2
        tot += w * float(
            np.mean((2.0 / math.pi) * ellipk(4.0 * r0 * rho_s / den2) / np.sqrt(den2))
        )
    return rho1 / (4.0 * math.pi) * tot


def _ring(rho1, rho2, gap, r0=2.0, t=2.0, a=0.005, ds=0.05) -> float:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho1, rho_2=rho2, h_1=t + gap))
    gf.create_electrode(w, "ring", name="g", center=(0, 0, t), radius=r0, wire_radius=a)
    gf.create_source(w, attached_to="g", magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gf.create_engine(backend="image_2layer", segment_length=ds).solve(w)
    return float(res.cluster_impedance("g")[0].real)


# audit W3A-02 (w4axi open point 3)
@pytest.mark.parametrize(
    ("rho1", "rho2", "gap"),
    [(1000.0, 30.0, 0.01), (30.0, 1000.0, 0.01), (1000.0, 30.0, 0.1)],
    ids=["1000_30_1cm", "30_1000_1cm", "1000_30_10cm"],
)
def test_w3a_02_a_ring_one_centimetre_above_an_interface_is_its_exact_thin_ring(
    rho1, rho2, gap
) -> None:
    """Within 0.2 % of the exact thin ring and of w4axi's Hankel reference.

    Before: +1.60 % / -0.094 % / +0.004 % (auditor W3A, e7, ds = 0.05 m).
    """
    from tests.benchmarks.references import ring_layered_hankel

    r0, t, a = 2.0, 2.0, 0.005
    exact = _thin_ring_exact(rho1, rho2, t + gap, r0, t, a)
    hankel = ring_layered_hankel([rho1, rho2], [t + gap], r0, t, a)
    assert abs(hankel / exact - 1.0) < 1e-5  # the two references
    value = _ring(rho1, rho2, gap)
    assert abs(value / exact - 1.0) < 2e-3
    assert abs(value / hankel - 1.0) < 2e-3


def _strip_plain_reference(rho1, rho2, h1, length, z, a) -> float:
    """Uniform-current resistance of a straight strip, images axis to axis.

    Direct term with the reduced kernel (the wire's own surface), every
    image at its vertical distance (the transverse-distance convention);
    the double integral of ``1/sqrt((x - y)² + d²)`` in closed form.
    """
    K = (rho2 - rho1) / (rho2 + rho1)

    def double_integral(d):
        return 2.0 * (
            length * math.asinh(length / d) - math.sqrt(length**2 + d * d) + d
        )

    terms = [(1.0, a), (1.0, 2.0 * z)]
    n = 1
    while abs(K) ** n > 1e-18:
        q = 2.0 * n * h1
        for zk in (q + z, q - z, -q + z, -q - z):
            terms.append((K**n, abs(z - zk)))
        n += 1
    total = math.fsum(w * double_integral(d) for w, d in terms)
    return rho1 / (4.0 * math.pi * length * length) * total


def test_w3a_02_a_strip_close_above_the_interface_is_its_plain_image_sum() -> None:
    """A 10 m strip 0.1 m above h_1 against the image sum axis to axis.

    1000/50 Ohm m, h_1 = 0.6 m, strip at 0.5 m: +6.8e-5 before at every
    segment length (the reduced kernel on the interface image, 0.2 m away);
    now -2.7e-6 ... -3.8e-6 for ds = 2 ... 0.125 m. The same strip in
    homogeneous soil reads -1.2e-6 ... -2.1e-6 against its closed form: the
    remainder of the far form and the reduced kernel the soil-surface image
    keeps (evidence ``w3a02_strip.py`` of stream fw3xl).
    """
    rho1, rho2, h1, length, z, a = 1000.0, 50.0, 0.6, 10.0, 0.5, 0.005
    ref = _strip_plain_reference(rho1, rho2, h1, length, z, a)
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho1, rho_2=rho2, h_1=h1))
    gf.create_electrode(
        w, "strip", name="s", start=(0, 0, z), end=(length, 0, z), wire_radius=a
    )
    gf.create_source(w, attached_to="s", magnitude=1.0)
    for ds in (1.0, 0.5, 0.25):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gf.create_engine(segment_length=ds).solve(w)
        dev = float(res.cluster_impedance("s")[0].real) / ref - 1.0
        assert abs(dev) < 5e-6, (ds, dev)


def _rod_two_owner_exact(rho1, rho2, h1, z0, z1, a) -> float:
    """Rod through the interface, one owner per layer (auditor W3A, e4).

    Every term of the Sunde expansion is a collinear vertical image of the
    source owner; the double integral of the reduced kernel over two
    collinear intervals is a second difference of
    ``G(u) = u arsinh(u/a) - sqrt(u² + a²)``.
    """
    K = (rho2 - rho1) / (rho2 + rho1)
    c = 2.0 * rho1 * rho2 / (rho1 + rho2)
    q = 2.0 * h1

    def G(u):
        return u * math.asinh(u / a) - math.sqrt(u * u + a * a)

    def D(a1, b1, a2, b2):
        return G(b1 - a2) - G(b1 - b2) - G(a1 - a2) + G(a1 - b2)

    def P(obs, src):
        up_o, up_s = obs[1] <= h1 + 1e-12, src[1] <= h1 + 1e-12
        terms = []
        if up_o and up_s:
            terms += [(rho1, 0.0, 1.0), (rho1, 0.0, -1.0)]
            m = 1
            while abs(K) ** m > 1e-18:
                w = rho1 * K**m
                terms += [
                    (w, m * q, 1.0),
                    (w, -m * q, 1.0),
                    (w, m * q, -1.0),
                    (w, -m * q, -1.0),
                ]
                m += 1
        elif up_s:
            m = 0
            while abs(K) ** m > 1e-18:
                terms += [(c * K**m, -m * q, 1.0), (c * K**m, -m * q, -1.0)]
                m += 1
        elif up_o:
            m = 0
            while abs(K) ** m > 1e-18:
                terms += [(c * K**m, m * q, 1.0), (c * K**m, -m * q, -1.0)]
                m += 1
        else:
            terms += [(rho2, 0.0, 1.0), (-rho2 * K, q, -1.0)]
            m = 0
            while abs(K) ** m > 1e-18:
                terms += [(rho2 * (1 - K * K) * K**m, -m * q, -1.0)]
                m += 1
        tot = 0.0
        for cf, off, sg in terms:
            lo_, hi_ = sorted((off + sg * src[0], off + sg * src[1]))
            tot += cf * D(obs[0], obs[1], lo_, hi_)
        return tot / (4.0 * math.pi * (obs[1] - obs[0]) * (src[1] - src[0]))

    owners = [(z0, h1), (h1, z1)]
    M = np.array([[P(o, s) for s in owners] for o in owners])
    return 1.0 / float(np.ones(2) @ np.linalg.solve(M, np.ones(2)))


@pytest.mark.parametrize(
    "soil", [(500.0, 100.0), (30.0, 1000.0)], ids=["500_100", "30_1000"]
)
def test_w3a_02_a_rod_through_the_interface_keeps_the_reduced_collinear_images(
    soil,
) -> None:
    """Collinear images keep the reduced kernel: the rod stays within 2e-6.

    A 6 m rod from 0.6 m through h_1 = 5 m against the auditor's exact
    two-owner closed form (snapshot before W3A-02: within 1.6e-6).
    """
    rho1, rho2 = soil
    h1, z0, z1, a = 5.0, 0.6, 6.6, 0.008
    exact = _rod_two_owner_exact(rho1, rho2, h1, z0, z1, a)
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho1, rho_2=rho2, h_1=h1))
    gf.create_electrode(
        w, "rod", name="r", position=(0, 0, z0), length=z1 - z0, wire_radius=a
    )
    gf.create_source(w, attached_to="r", magnitude=1.0)
    for ds in (1.0, 0.25):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gf.create_engine(segment_length=ds).solve(w)
        dev = float(res.cluster_impedance("r")[0].real) / exact - 1.0
        assert abs(dev) < 2e-6, (ds, dev)
