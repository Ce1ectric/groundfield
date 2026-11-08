"""Effective resistivity of a two-layer soil, the ``rho`` of a BusType.

User decision Q6 of the calendar release (finding 10.84, the ``rho`` that a
BusType carries for layered soil): the AP1 / AP2 work defines the effective
resistivity of a two-layer soil as

    rho_eff = rho_1**(1 - w) * rho_2**w,   w = w0 * (h_1 / 10 m)**(-beta),

with ``w0 = 0.13`` and ``beta = 0.8`` from the AP1 catalogue; AP2 sets the
``specific_earth_resistance`` of a groundinsight bus to ``rho_eff``.
:func:`groundfield.effective_resistivity` implements it: exact for
homogeneous soil, the formula for two-layer soil and for layer stacks that
are two-layer after merging equal layers, ``w`` limited to ``[0, 1]`` and a
clear error for three or more distinct layers.

The tests compare with the user's research function (reproduced verbatim
below), with two independent evaluations of the formula (log-space weighted
mean, 50-digit ``mpmath``) and with hand-checked anchor values.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import groundfield as gf

pytestmark = pytest.mark.finding("10.84")  # user decision Q6

W0, BETA = 0.13, 0.8


def user_rho_eff(r1, r2, h1, w0=W0, beta=BETA):
    """The user's research function (AP1 recommendation equation), verbatim."""
    w = w0 * (h1 / 10.0) ** (-beta)
    return r1 ** (1.0 - w) * r2**w


#: The two-layer soils of the AP1 catalogue (rho_1, rho_2, h_1).
AP1_GRID = [
    (r1, r2, h1)
    for r1, r2, h1 in itertools.product(
        (30.0, 100.0, 200.0, 500.0, 1000.0), (30.0, 100.0, 200.0), (5.0, 10.0, 30.0)
    )
    if r1 != r2
]

#: h_1 below which w = w0 * (h_1 / 10 m)**(-beta) exceeds 1 (AP1 defaults).
H_CLIP = 10.0 * W0 ** (1.0 / BETA)


def _two_layer(r1: float, r2: float, h1: float) -> gf.TwoLayerSoil:
    return gf.TwoLayerSoil(rho_1=r1, rho_2=r2, h_1=h1)


# ---------------------------------------------------------------------
# homogeneous soil: exact
# ---------------------------------------------------------------------


@pytest.mark.parametrize("rho", [30.0, 100.0, 200.0, 37.3, 123.456, 1e-3, 2.5e5])
def test_homogeneous_soil_is_returned_exactly(rho: float) -> None:
    soil = gf.HomogeneousSoil(rho=rho)
    assert gf.effective_resistivity(soil) == rho
    assert gf.effective_resistivity(soil, w0=0.4, beta=1.7, h_ref=3.0) == rho


@pytest.mark.parametrize("rho", [30.0, 100.0, 200.0, 500.0, 1000.0, 37.3])
@pytest.mark.parametrize("h1", [0.3, 0.9, 5.0, 10.0, 30.0])
def test_equal_layers_are_homogeneous_and_exact(rho: float, h1: float) -> None:
    """Equal layers are one homogeneous soil (canonical stack); the plain
    float formula is not always bit-exact there (200, 200, 10 m gives
    200.00000000000003)."""
    expected = rho
    assert gf.effective_resistivity(_two_layer(rho, rho, h1)) == expected
    assert gf.effective_resistivity(rho_1=rho, rho_2=rho, h_1=h1) == expected
    stack = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=rho, h=h1),
            gf.SoilLayer(rho=rho, h=2.0),
            gf.SoilLayer(rho=rho),
        ]
    )
    assert gf.effective_resistivity(stack) == expected
    single = gf.MultiLayerSoil(layers=[gf.SoilLayer(rho=rho)])
    assert gf.effective_resistivity(single) == expected


def test_the_plain_formula_is_not_exact_for_equal_layers() -> None:
    """Why the homogeneous case is handled explicitly."""
    assert user_rho_eff(200.0, 200.0, 10.0) != 200.0
    assert gf.effective_resistivity(rho_1=200.0, rho_2=200.0, h_1=10.0) == 200.0


# ---------------------------------------------------------------------
# two-layer soil: the formula
# ---------------------------------------------------------------------


@pytest.mark.parametrize("r1, r2, h1", AP1_GRID)
def test_matches_the_user_formula_on_the_ap1_grid(r1, r2, h1) -> None:
    """Bit-identical to the user's function inside the calibration range,
    so a BusType's ``rho`` equals the ``specific_earth_resistance`` that the
    AP2 scripts compute for the same soil."""
    expected = user_rho_eff(r1, r2, h1)
    assert gf.effective_resistivity(_two_layer(r1, r2, h1)) == expected
    assert gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=h1) == expected


@pytest.mark.parametrize("r1, r2, h1", AP1_GRID)
def test_formula_against_independent_evaluations(r1, r2, h1) -> None:
    value = gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=h1)
    w = W0 * (h1 / 10.0) ** (-BETA)
    # (a) weighted mean of the logarithms
    log_mean = math.exp((1.0 - w) * math.log(r1) + w * math.log(r2))
    assert value == pytest.approx(log_mean, rel=1e-14)
    # (b) the defining property: log rho_eff divides [log rho_1, log rho_2] at w
    fraction = (math.log(value) - math.log(r1)) / (math.log(r2) - math.log(r1))
    assert fraction == pytest.approx(w, rel=1e-12)
    assert min(r1, r2) < value < max(r1, r2)


def test_formula_against_a_50_digit_evaluation() -> None:
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    worst = 0.0
    for r1, r2, h1 in AP1_GRID:
        w = mp.mpf(W0) * (mp.mpf(h1) / 10) ** (-mp.mpf(BETA))
        exact = mp.mpf(r1) ** (1 - w) * mp.mpf(r2) ** w
        value = gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=h1)
        worst = max(worst, float(abs(mp.mpf(value) / exact - 1)))
    assert worst < 4e-16  # measured 1.6e-16: round-off of float64


@pytest.mark.parametrize(
    "r1, r2, h1, expected",
    [
        # h_1 = 10 m: w = w0 = 0.13 exactly
        (1000.0, 30.0, 10.0, 633.9073146399232),
        (100.0, 200.0, 10.0, 109.42937012607396),
        # h_1 = 5 m: w = 0.13 * 2**0.8 = 0.226343
        (100.0, 30.0, 5.0, 76.14650585114978),
        (30.0, 100.0, 5.0, 39.39773685563934),
        # h_1 = 30 m: w = 0.13 * 3**-0.8 = 0.053982
        (500.0, 100.0, 30.0, 458.39348170328253),
    ],
)
def test_anchor_values(r1, r2, h1, expected) -> None:
    """Values printed by the evidence script of this change (50-digit check)."""
    assert gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=h1) == pytest.approx(
        expected, rel=1e-15
    )


def test_weight_at_the_reference_thickness_is_w0() -> None:
    """At h_1 = h_ref the weight is w0: rho_eff = rho_1 * (rho_2/rho_1)**w0."""
    for w0 in (0.0, 0.13, 0.5, 1.0):
        value = gf.effective_resistivity(
            rho_1=1000.0, rho_2=30.0, h_1=10.0, w0=w0, beta=0.8
        )
        assert value == pytest.approx(1000.0 * (30.0 / 1000.0) ** w0, rel=1e-14)


# ---------------------------------------------------------------------
# the clip of w to [0, 1]
# ---------------------------------------------------------------------


def test_clip_threshold_is_0p78_m() -> None:
    assert H_CLIP == pytest.approx(0.7806011565155501, rel=1e-15)
    assert W0 * (H_CLIP / 10.0) ** (-BETA) == pytest.approx(1.0, rel=1e-14)


@pytest.mark.parametrize("r1, r2", [(100.0, 30.0), (30.0, 100.0), (1000.0, 200.0)])
@pytest.mark.parametrize("h1", [1e-3, 0.05, 0.2, 0.5, 0.78])
def test_thin_upper_layer_gives_the_lower_resistivity(r1, r2, h1) -> None:
    """Below about 0.78 m the unclipped formula leaves [rho_1, rho_2]; with
    w limited to 1 the soil is the lower layer."""
    assert h1 < H_CLIP
    assert gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=h1) == r2
    assert gf.effective_resistivity(_two_layer(r1, r2, h1)) == r2


def test_the_unclipped_formula_would_leave_the_range() -> None:
    assert user_rho_eff(100.0, 30.0, 0.5) == pytest.approx(17.9169, rel=1e-5)
    assert user_rho_eff(30.0, 100.0, 0.5) == pytest.approx(167.440, rel=1e-5)
    assert gf.effective_resistivity(rho_1=100.0, rho_2=30.0, h_1=0.5) == 30.0
    assert gf.effective_resistivity(rho_1=30.0, rho_2=100.0, h_1=0.5) == 100.0


@pytest.mark.parametrize("r1, r2", [(100.0, 30.0), (30.0, 1000.0), (1000.0, 30.0)])
def test_range_and_monotonicity_over_the_thickness(r1, r2) -> None:
    """rho_eff stays in [min, max], moves monotonically from rho_2 (thin
    upper layer) to rho_1 (thick upper layer) and is continuous at the clip."""
    h = np.geomspace(1e-3, 1e4, 400)
    values = np.array([gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=x) for x in h])
    assert np.all(values >= min(r1, r2)) and np.all(values <= max(r1, r2))
    steps = np.diff(values)
    assert np.all(steps >= 0.0) if r1 > r2 else np.all(steps <= 0.0)
    assert values[0] == r2
    assert abs(values[-1] / r1 - 1.0) < 0.01  # w(10 km) = 0.13 * 1000**-0.8 = 5e-4
    above = gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=H_CLIP * (1 + 1e-9))
    assert above == pytest.approx(r2, rel=1e-8)


# ---------------------------------------------------------------------
# group-specific parameters
# ---------------------------------------------------------------------


@pytest.mark.parametrize("w0, beta", [(0.2, 0.5), (0.05, 1.2), (0.13, 0.0)])
@pytest.mark.parametrize("r1, r2, h1", [(100.0, 30.0, 5.0), (500.0, 200.0, 30.0)])
def test_group_specific_parameters(w0, beta, r1, r2, h1) -> None:
    expected = user_rho_eff(r1, r2, h1, w0=w0, beta=beta)
    assert gf.effective_resistivity(
        _two_layer(r1, r2, h1), w0=w0, beta=beta
    ) == pytest.approx(expected, rel=1e-15)


def test_reference_thickness_scales_h1() -> None:
    """w depends on h_1 / h_ref only."""
    a = gf.effective_resistivity(rho_1=300.0, rho_2=50.0, h_1=3.0, h_ref=5.0)
    b = gf.effective_resistivity(rho_1=300.0, rho_2=50.0, h_1=6.0, h_ref=10.0)
    assert a == pytest.approx(b, rel=1e-15)


def test_w0_zero_is_the_upper_layer() -> None:
    """w0 = 0 reproduces the upper-layer resistivity rho_1."""
    for r1, r2, h1 in AP1_GRID:
        assert gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=h1, w0=0.0) == r1


# ---------------------------------------------------------------------
# canonical stacks and the three-layer error
# ---------------------------------------------------------------------


def test_canonical_two_layer_stacks() -> None:
    """A MultiLayerSoil is reduced to its canonical stack (equal adjacent
    layers merged), as every layered backend does (10.48)."""
    two = gf.effective_resistivity(_two_layer(100.0, 400.0, 2.0))
    merged_bottom = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=100.0, h=2.0),
            gf.SoilLayer(rho=400.0, h=3.0),
            gf.SoilLayer(rho=400.0),
        ]
    )
    assert gf.effective_resistivity(merged_bottom) == two
    plain = gf.MultiLayerSoil(
        layers=[gf.SoilLayer(rho=100.0, h=2.0), gf.SoilLayer(rho=400.0)]
    )
    assert gf.effective_resistivity(plain) == two
    # equal upper layers add their thicknesses: h_1 = 2 + 3 = 5 m
    merged_top = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=100.0, h=2.0),
            gf.SoilLayer(rho=100.0, h=3.0),
            gf.SoilLayer(rho=30.0),
        ]
    )
    assert gf.effective_resistivity(merged_top) == user_rho_eff(100.0, 30.0, 5.0)


def test_three_distinct_layers_raise_a_clear_error() -> None:
    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=80.0, h=0.5),
            gf.SoilLayer(rho=300.0, h=2.0),
            gf.SoilLayer(rho=50.0),
        ]
    )
    with pytest.raises(ValueError) as excinfo:
        gf.effective_resistivity(soil)
    message = str(excinfo.value)
    assert message.startswith("effective_resistivity:")
    assert "3 distinct layers" in message
    assert "80, 300, 50" in message
    assert "two-layer" in message
    assert "rho_1, rho_2 and h_1" in message


# ---------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"rho_1": 0.0}, "rho_1"),
        ({"rho_1": -100.0}, "rho_1"),
        ({"rho_2": math.nan}, "rho_2"),
        ({"rho_2": math.inf}, "rho_2"),
        ({"h_1": 0.0}, "h_1"),
        ({"h_1": math.inf}, "h_1"),
        ({"w0": -0.1}, "w0"),
        ({"w0": math.nan}, "w0"),
        ({"beta": -0.8}, "beta"),
        ({"beta": math.inf}, "beta"),
        ({"h_ref": 0.0}, "h_ref"),
        ({"h_ref": math.nan}, "h_ref"),
        ({"rho_1": "abc"}, "rho_1"),
    ],
)
def test_invalid_values_are_rejected(kwargs, name) -> None:
    args = {"rho_1": 100.0, "rho_2": 30.0, "h_1": 5.0} | kwargs
    with pytest.raises(ValueError, match=f"effective_resistivity: {name} "):
        gf.effective_resistivity(**args)


@pytest.mark.parametrize("bad", [{"w0": -0.1}, {"beta": -1.0}, {"h_ref": -10.0}])
def test_invalid_parameters_are_rejected_for_a_soil_too(bad) -> None:
    with pytest.raises(ValueError):
        gf.effective_resistivity(gf.HomogeneousSoil(rho=100.0), **bad)


def test_soil_and_numbers_are_exclusive() -> None:
    soil = _two_layer(100.0, 30.0, 5.0)
    with pytest.raises(TypeError, match="either a soil model or rho_1, rho_2 and h_1"):
        gf.effective_resistivity(soil, rho_1=100.0)
    with pytest.raises(TypeError, match="missing h_1"):
        gf.effective_resistivity(rho_1=100.0, rho_2=30.0)
    with pytest.raises(TypeError, match="missing rho_1, rho_2, h_1"):
        gf.effective_resistivity()


def test_a_number_as_the_soil_is_a_type_error_with_a_hint() -> None:
    with pytest.raises(TypeError, match=r"rho_1=\.\.\., rho_2=\.\.\., h_1=\.\.\."):
        gf.effective_resistivity(100.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# namespace and results
# ---------------------------------------------------------------------


def test_exported_from_the_soil_package_and_the_top_level() -> None:
    import groundfield.soil

    assert gf.effective_resistivity is groundfield.soil.effective_resistivity
    assert "effective_resistivity" in gf.__all__
    assert "effective_resistivity" in groundfield.soil.__all__


def test_accepts_the_soil_of_a_result() -> None:
    """``FieldResult.soil`` is the canonical soil that was solved."""
    world = gf.create_world(soil=_two_layer(100.0, 30.0, 5.0))
    gf.create_electrode(world, "rod", name="g", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(world, attached_to="g")
    result = gf.create_engine(segment_length=0.5).solve(world)
    assert gf.effective_resistivity(result.soil) == user_rho_eff(100.0, 30.0, 5.0)
