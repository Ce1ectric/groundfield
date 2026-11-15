"""Complex penetration depth of layered soil (roadmap: penetration depth for
multilayer soil).

``penetration_depth(soil, f)`` returns the complex depth p for which the
homogeneous complex-depth formulas (Dubanton, Deri) reproduce the
earth-return impedance of a layered soil: the plane-wave surface impedance of
the stack over j omega mu0, so that exp(-2 p lambda) agrees with the layered
surface reflection coefficient to second order in lambda. Two methods: the
exact layered earth-return integral (``per_metre_mutual_impedance``) and the
formula.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf
from groundfield.coupling.carson import carson_mutual_correction
from groundfield.coupling.sommerfeld_inductance import (
    LayeredEarth,
    per_metre_mutual_impedance,
    reflection_coefficient_layered,
)
from groundfield.references.pollaczek import pollaczek_mutual_impedance

MU0 = 4e-7 * math.pi
FREQS = [50.0, 200.0, 1000.0]


def _err(soil, earth, f, z, x) -> float:
    """Relative error of the earth-return correction (Carson's quantity)."""
    om = 2.0 * math.pi * f
    exact = per_metre_mutual_impedance(z, z, x, omega=om, earth=earth)
    formula = gf.penetration_depth(soil, [f]).mutual_impedance(z, z, x)[0]
    image = 1j * om * MU0 / (2.0 * math.pi) * math.log(math.hypot(2 * abs(z), x) / x)
    return abs(formula - exact) / abs(exact - image)


def test_homogeneous_soil_gives_its_own_depth_and_resistivity() -> None:
    pd = gf.penetration_depth(gf.HomogeneousSoil(rho=100.0), FREQS)
    for k, f in enumerate(FREQS):
        jwm = 1j * 2.0 * math.pi * f * MU0
        assert pd.resistivity[k] == 100.0
        assert pd.depth[k] == pytest.approx(np.sqrt(100.0 / jwm), rel=1e-15)
    assert np.array_equal(pd.frequencies, FREQS)


@pytest.mark.parametrize("f", FREQS)
def test_thin_and_thick_top_layers_tend_to_the_layer_values(f) -> None:
    def rho(h1: float) -> complex:
        soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=h1)
        return complex(gf.penetration_depth(soil, [f]).resistivity[0])

    assert rho(1e-6) == pytest.approx(30.0, rel=1e-6)
    assert rho(1e5) == pytest.approx(1000.0, rel=1e-12)


@pytest.mark.parametrize("f", FREQS)
def test_the_depth_is_the_low_wavenumber_limit_of_the_layered_reflection(f) -> None:
    """Independent method: the layered surface reflection coefficient of the
    Sommerfeld kernel (three layers), Gamma = 1 - 2 p lam + 2 p^2 lam^2 +
    O(lam^3): the residual of exp(-2 p lam) falls eightfold when lam halves."""
    earth = LayeredEarth(rhos=(100.0, 1000.0, 10.0), thicknesses=(2.0, 5.0))
    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=100.0, h=2.0),
            gf.SoilLayer(rho=1000.0, h=5.0),
            gf.SoilLayer(rho=10.0),
        ]
    )
    p = complex(gf.penetration_depth(soil, [f]).depth[0])
    om = 2.0 * math.pi * f
    scale = 1e-3 / abs(p)
    lams = np.array([scale, 0.5 * scale])
    gam = reflection_coefficient_layered(lams, omega=om, earth=earth)
    first = (1.0 - gam) / (2.0 * lams)
    assert first[1] == pytest.approx(p, rel=1e-3)
    resid = np.abs(gam - np.exp(-2.0 * p * lams))
    assert resid[0] / resid[1] == pytest.approx(8.0, rel=0.02)


def test_equal_layers_are_merged() -> None:
    two = gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=5.0)
    multi = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=100.0, h=2.0),
            gf.SoilLayer(rho=100.0, h=3.0),
            gf.SoilLayer(rho=30.0),
        ]
    )
    a = gf.penetration_depth(two, FREQS).depth
    b = gf.penetration_depth(multi, FREQS).depth
    assert np.allclose(a, b, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("f", [50.0, 1000.0])
@pytest.mark.parametrize("x", [0.1, 10.0, 300.0])
def test_the_reference_integral_is_carson_and_pollaczek(f, x) -> None:
    """Method 1 against two independent references: Carson (homogeneous,
    overhead) and the two-layer Pollaczek integral (buried), within 1e-7
    (measured worst 4.9e-8)."""
    om = 2.0 * math.pi * f
    hom = LayeredEarth(rhos=(1000.0,), thicknesses=())
    got = per_metre_mutual_impedance(-10.0, -10.0, x, omega=om, earth=hom)
    ref = 1j * om * MU0 / (2.0 * math.pi) * math.log(math.hypot(20.0, x) / x)
    ref += carson_mutual_correction(om, 10.0, 10.0, x, 1e-3)
    assert got == pytest.approx(ref, rel=1e-7)
    two = LayeredEarth(rhos=(1000.0, 30.0), thicknesses=(5.0,))
    got = per_metre_mutual_impedance(0.7, 0.7, x, omega=om, earth=two)
    ref = pollaczek_mutual_impedance(
        1000.0, 30.0, 5.0, f, depth_i=0.7, depth_j=0.7, separation=x
    )
    assert got == pytest.approx(ref, rel=1e-7)


# The 39 soils of the AP1 grid: measured worst errors of the earth-return
# correction (overhead 10 m / buried 0.7 m, 50 Hz ... 1 kHz, x = 0.1 ...
# 1000 m): 28.4 % / 33.4 % with the equivalent depth (median 2.2 %), the
# formula's own error in the homogeneous soils 12.3 % / 13.0 % (median
# 2.4 %); the homogeneous depth of rho_1 up to 2255 % / 2797 %.
@pytest.mark.parametrize(
    ("rho_1", "rho_2", "h_1"),
    [(1000.0, 30.0, 5.0), (30.0, 200.0, 30.0), (100.0, 30.0, 10.0)],
)
@pytest.mark.parametrize("z", [-10.0, 0.7], ids=["overhead", "buried"])
def test_the_equivalent_depth_reproduces_the_layered_impedance(
    rho_1, rho_2, h_1, z
) -> None:
    soil = gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=h_1)
    earth = LayeredEarth(rhos=(rho_1, rho_2), thicknesses=(h_1,))
    for f in (50.0, 1000.0):
        for x in (0.1, 10.0):
            assert _err(soil, earth, f, z, x) < 0.07
        for x in (100.0, 1000.0):
            assert _err(soil, earth, f, z, x) < 0.35
    # the homogeneous depth of the upper layer is far off at 50 Hz, 10 m
    rho1 = gf.penetration_depth(gf.HomogeneousSoil(rho=rho_1), [50.0])
    om = 2.0 * math.pi * 50.0
    exact = per_metre_mutual_impedance(z, z, 10.0, omega=om, earth=earth)
    image = (
        1j * om * MU0 / (2.0 * math.pi) * math.log(math.hypot(2 * abs(z), 10.0) / 10.0)
    )
    naive = abs(rho1.mutual_impedance(z, z, 10.0)[0] - exact) / abs(exact - image)
    assert naive > 4.0 * _err(soil, earth, 50.0, z, 10.0)


def test_the_error_vanishes_at_large_separations() -> None:
    """The O(lam^3) residual decays with x: the worst AP1 soil (30 over
    200 Ohm m, h1 = 30 m), overhead at 1 kHz: 15 % at 1 km, 0.05 % at
    10 km, 0.007 % at 30 km."""
    soil = gf.TwoLayerSoil(rho_1=30.0, rho_2=200.0, h_1=30.0)
    earth = LayeredEarth(rhos=(30.0, 200.0), thicknesses=(30.0,))
    assert _err(soil, earth, 1000.0, -10.0, 1e3) > 0.1
    assert _err(soil, earth, 1000.0, -10.0, 1e4) < 1e-3
    assert _err(soil, earth, 1000.0, -10.0, 3e4) < 2e-4


def test_the_formula_rejects_wires_on_both_sides_and_bad_input() -> None:
    pd = gf.penetration_depth(gf.HomogeneousSoil(rho=100.0), [50.0])
    with pytest.raises(ValueError, match="one side"):
        pd.mutual_impedance(-1.0, 0.7, 1.0)
    with pytest.raises(ValueError, match="positive"):
        pd.mutual_impedance(0.7, 0.7, 0.0)
    for bad in ([0.0], [-50.0], [math.nan], []):
        with pytest.raises(ValueError, match="positive"):
            gf.penetration_depth(gf.HomogeneousSoil(rho=100.0), bad)
    with pytest.raises(TypeError, match="soil"):
        gf.penetration_depth(100.0, [50.0])


def test_self_impedance_of_a_wire_matches_carson() -> None:
    """The self impedance (distance = radius) in homogeneous soil: the
    complex-depth formula against Carson, within 2 % (Deri's accuracy)."""
    f, h, r = 50.0, 10.0, 0.01
    om = 2.0 * math.pi * f
    pd = gf.penetration_depth(gf.HomogeneousSoil(rho=100.0), [f])
    z_cd = pd.mutual_impedance(-h, -h, r)[0]
    z_carson = 1j * om * MU0 / (2.0 * math.pi) * math.log(2.0 * h / r)
    z_carson += carson_mutual_correction(om, h, h, 0.0, 0.01)
    assert z_cd == pytest.approx(z_carson, rel=0.02)


# audit W3B-11
@pytest.mark.parametrize(
    "bad",
    [
        True,
        False,
        "50",
        ["50"],
        [True, 50.0],
        np.array([True]),
        50.0 + 0j,
        [1.0 + 1e-3j],
    ],
    ids=[
        "True",
        "False",
        "str",
        "str_list",
        "bool_in_list",
        "bool_array",
        "complex",
        "complex_list",
    ],
)
def test_w3b_11_frequencies_are_real_numbers_not_bool_or_str(bad) -> None:
    """Base 098bbbc: ``"50"`` and ``True`` were accepted as frequencies (50 Hz
    and 1 Hz). Frequencies are finite, positive real numbers."""
    with pytest.raises(ValueError, match="real numbers"):
        gf.penetration_depth(gf.HomogeneousSoil(rho=100.0), bad)


# audit W3B-11
@pytest.mark.parametrize("bad", [1e-320, 5e-324, [50.0, 1e-310]])
def test_w3b_11_a_subnormal_frequency_is_refused(bad) -> None:
    """Base 098bbbc: f = 1e-320 raised ZeroDivisionError. A frequency below
    numpy.finfo(float).tiny is refused with ValueError."""
    for soil in (
        gf.HomogeneousSoil(rho=100.0),
        gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0),
    ):
        with pytest.raises(ValueError, match="tiny"):
            gf.penetration_depth(soil, bad)


# audit W3B-11 (own attack): the smallest normal frequencies give a finite depth
@pytest.mark.parametrize("f", [float(np.finfo(float).tiny), 1e-305, 1e-300])
def test_w3b_11_the_smallest_normal_frequency_gives_a_finite_depth(f) -> None:
    """Base 098bbbc: the homogeneous depth sqrt(rho / (j w mu0)) overflowed to
    inf below about 1e-303 Hz; it is 1 / gamma now, as for the bottom layer of
    a stack."""
    for soil in (
        gf.HomogeneousSoil(rho=100.0),
        gf.HomogeneousSoil(rho=1e6),
        gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0),
    ):
        pd = gf.penetration_depth(soil, f)
        assert np.all(np.isfinite(pd.depth)) and np.all(np.isfinite(pd.resistivity))
    pd = gf.penetration_depth(gf.HomogeneousSoil(rho=100.0), f)
    assert pd.resistivity[0] == 100.0
    expected = 1.0 / np.sqrt(2j * math.pi * f * MU0 / 100.0) if f > 1e-302 else None
    if expected is not None:
        assert pd.depth[0] == pytest.approx(expected, rel=1e-12)


# audit W3B-11
def test_w3b_11_penetration_depth_compares_by_value_and_hashes() -> None:
    """Base 098bbbc: ``==`` of two results raised ValueError (the truth value
    of an array) and ``hash`` raised TypeError. A result is a value: equal
    soils and frequencies give equal results with equal hashes, and its
    arrays are read-only copies, so the hash cannot change."""
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    a = gf.penetration_depth(soil, [50.0, 1000.0])
    b = gf.penetration_depth(soil, np.array([50.0, 1000.0]))
    c = gf.penetration_depth(soil, [50.0])
    assert a == b and hash(a) == hash(b)
    assert a != c and a != gf.penetration_depth(
        gf.HomogeneousSoil(rho=30.0), [50.0, 1000.0]
    )
    assert (a == "x") is False
    assert len({a, b, c}) == 2
    f = np.array([50.0])
    d = gf.PenetrationDepth(
        frequencies=f, depth=c.depth.copy(), resistivity=c.resistivity.copy()
    )
    f[0] = 60.0  # the result keeps its own copy
    assert d == c and hash(d) == hash(c)
    for arr in (d.frequencies, d.depth, d.resistivity):
        assert not arr.flags.writeable
    with pytest.raises(ValueError, match="shape"):
        gf.PenetrationDepth(frequencies=[50.0, 60.0], depth=[1.0], resistivity=[1.0])
