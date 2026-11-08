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

User decision Q36 adds named presets of the weights ``(w0, beta)`` per
electrode class, with the values of the user's AP2 hand-over
(``EFFECTIVE_RESISTIVITY_PRESETS``, ``effective_resistivity(...,
preset=...)``); the default preset ``"lv_network"`` is the AP1 calibration,
so every earlier call keeps its value bit for bit.
"""

from __future__ import annotations

import inspect
import itertools
import math
import warnings
from collections.abc import Mapping

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
# presets per electrode class (user decision Q36)
# ---------------------------------------------------------------------

#: (w0, beta) of the AP2 hand-over (``gruppen_bustypes.json``, generated
#: 2026-10-20 by ``gruppenbildung.py`` from the AP1 catalogue computed with
#: groundfield 0.14.0; method A, median network q50) per LV network group
#: (perimeter equivalent n_U, thresholds 20 / 100 / 250) and penetration of
#: foundation electrodes in %.
HANDOVER_GROUPS = {
    1: {5: (0.06, 1.1), 10: (0.04, 1.6), 20: (0.04, 1.6), 40: (0.06, 1.3)},
    2: {5: (0.06, 1.4), 10: (0.10, 0.9), 20: (0.16, 0.9), 40: (0.24, 0.5)},
    3: {5: (0.12, 0.9), 10: (0.16, 0.6), 20: (0.20, 0.5), 40: (0.24, 0.4)},
    4: {5: (0.12, 0.8), 10: (0.18, 0.6), 20: (0.28, 0.4), 40: (0.36, 0.3)},
}
#: The station electrode alone (4 m x 4 m ring and three 4 m rods).
HANDOVER_STATION = (0.06, 1.3)
#: The global AP1 calibration (the hand-over's method "A0").
AP1_GLOBAL = (W0, BETA)

EXPECTED_PRESETS = {"lv_network": AP1_GLOBAL, "station": HANDOVER_STATION} | {
    f"lv_group_{group}_p{penetration:02d}": weights
    for group, row in HANDOVER_GROUPS.items()
    for penetration, weights in row.items()
}


def _presets() -> Mapping[str, tuple[float, float]]:
    import groundfield.soil

    return groundfield.soil.EFFECTIVE_RESISTIVITY_PRESETS


def test_presets_hold_the_values_of_the_ap2_handover() -> None:
    # user decision Q36
    presets = _presets()
    assert dict(presets) == EXPECTED_PRESETS
    assert len(presets) == 18
    for name, weights in presets.items():
        assert isinstance(weights, tuple) and len(weights) == 2, name
        assert all(type(value) is float for value in weights), name


def test_presets_are_a_read_only_mapping() -> None:
    # user decision Q36
    presets = _presets()
    assert isinstance(presets, Mapping)
    with pytest.raises(TypeError):
        presets["station"] = (0.1, 1.0)  # type: ignore[index]
    with pytest.raises(TypeError):
        del presets["station"]  # type: ignore[attr-defined]
    assert presets["station"] == HANDOVER_STATION


def test_signature_defaults_keep_every_earlier_call() -> None:
    """``preset`` supplies the default of each weight; ``w0`` and ``beta``
    default to ``None`` (take the preset's value)."""
    # user decision Q36
    parameters = inspect.signature(gf.effective_resistivity).parameters
    assert parameters["preset"].default == "lv_network"
    assert parameters["preset"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["w0"].default is None
    assert parameters["beta"].default is None
    assert parameters["h_ref"].default == 10.0


@pytest.mark.parametrize("r1, r2, h1", AP1_GRID)
def test_default_preset_is_the_ap1_calibration_bit_for_bit(r1, r2, h1) -> None:
    # user decision Q36
    expected = user_rho_eff(r1, r2, h1)
    soil = _two_layer(r1, r2, h1)
    assert gf.effective_resistivity(soil, preset="lv_network") == expected
    assert gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=h1) == expected
    # explicit weights override the preset completely
    assert (
        gf.effective_resistivity(soil, preset="station", w0=W0, beta=BETA) == expected
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_PRESETS))
def test_every_preset_is_the_user_formula_with_its_weights(name: str) -> None:
    """Inside the calibration range (w <= 1 on the AP1 grid for every
    preset) the value is bit-identical to the user's function with the
    preset's (w0, beta), for a soil model and for the keywords."""
    # user decision Q36
    w0, beta = EXPECTED_PRESETS[name]
    for r1, r2, h1 in AP1_GRID:
        assert w0 * (h1 / 10.0) ** (-beta) < 1.0
        expected = user_rho_eff(r1, r2, h1, w0=w0, beta=beta)
        assert gf.effective_resistivity(_two_layer(r1, r2, h1), preset=name) == expected
        assert (
            gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=h1, preset=name)
            == expected
        )


def test_station_preset_anchor_values() -> None:
    """w = 0.06 * (h_1 / 10 m)**-1.3: 0.148 / 0.060 / 0.014 at 5 / 10 / 30 m,
    independent of the implementation (log-space mean)."""
    # user decision Q36
    for r1, r2, h1 in [(100.0, 30.0, 5.0), (1000.0, 30.0, 10.0), (30.0, 200.0, 30.0)]:
        w = 0.06 * (h1 / 10.0) ** (-1.3)
        expected = math.exp((1.0 - w) * math.log(r1) + w * math.log(r2))
        value = gf.effective_resistivity(rho_1=r1, rho_2=r2, h_1=h1, preset="station")
        assert value == pytest.approx(expected, rel=1e-14)
    assert gf.effective_resistivity(
        rho_1=1000.0, rho_2=30.0, h_1=10.0, preset="station"
    ) == pytest.approx(1000.0 * 0.03**0.06, rel=1e-14)


def test_an_explicit_weight_overrides_only_that_weight() -> None:
    # user decision Q36
    soil = _two_layer(100.0, 30.0, 5.0)
    assert gf.effective_resistivity(soil, preset="station", w0=0.2) == user_rho_eff(
        100.0, 30.0, 5.0, w0=0.2, beta=1.3
    )
    assert gf.effective_resistivity(soil, preset="station", beta=0.5) == user_rho_eff(
        100.0, 30.0, 5.0, w0=0.06, beta=0.5
    )
    # the default preset supplies the weight that is not given (as before Q36)
    assert gf.effective_resistivity(soil, w0=0.2) == user_rho_eff(
        100.0, 30.0, 5.0, w0=0.2, beta=0.8
    )
    assert gf.effective_resistivity(soil, beta=1.2) == user_rho_eff(
        100.0, 30.0, 5.0, w0=0.13, beta=1.2
    )
    w0, beta = EXPECTED_PRESETS["lv_group_4_p40"]
    assert gf.effective_resistivity(soil, preset="lv_group_4_p40", w0=0.0) == 100.0
    assert gf.effective_resistivity(
        soil, preset="lv_group_4_p40", beta=0.0
    ) == user_rho_eff(100.0, 30.0, 5.0, w0=w0, beta=0.0)


@pytest.mark.parametrize("name", sorted(EXPECTED_PRESETS))
def test_homogeneous_soil_is_exact_with_every_preset(name: str) -> None:
    # user decision Q36
    assert gf.effective_resistivity(gf.HomogeneousSoil(rho=123.456), preset=name) == (
        123.456
    )
    assert (
        gf.effective_resistivity(rho_1=200.0, rho_2=200.0, h_1=10.0, preset=name)
        == 200.0
    )


@pytest.mark.parametrize("name", ["station", "lv_group_1_p10", "lv_group_4_p40"])
def test_the_limit_of_w_holds_for_a_preset(name: str) -> None:
    """The thin-layer limit sits at h_ref * w0**(1/beta), per preset: 1.15 m
    for the station, 1.34 m for lv_group_1_p10, 0.33 m for lv_group_4_p40."""
    # user decision Q36
    w0, beta = EXPECTED_PRESETS[name]
    h_clip = 10.0 * w0 ** (1.0 / beta)
    below = gf.effective_resistivity(
        rho_1=100.0, rho_2=30.0, h_1=0.999 * h_clip, preset=name
    )
    above = gf.effective_resistivity(
        rho_1=100.0, rho_2=30.0, h_1=1.001 * h_clip, preset=name
    )
    assert below == 30.0
    assert 30.0 < above < 30.0 * 1.01
    assert {"station": 1.15, "lv_group_1_p10": 1.34, "lv_group_4_p40": 0.33}[
        name
    ] == pytest.approx(h_clip, abs=0.005)


def test_weights_order_the_classes_by_extent() -> None:
    """Plausibility of the hand-over values: a compact station sees less of
    the lower layer than the average LV network (smaller weight at the three
    AP1 thicknesses), and in groups 2-4 the weight at h_1 = 10 m (w = w0)
    grows with the penetration of foundation electrodes. Group 1 (single
    objects, hamlets) is not monotonic (0.06 / 0.04 / 0.04 / 0.06)."""
    # user decision Q36
    presets = _presets()

    def weight(name: str, h1: float) -> float:
        w0, beta = presets[name]
        return w0 * (h1 / 10.0) ** (-beta)

    for h1 in (5.0, 10.0, 30.0):
        assert weight("station", h1) < weight("lv_network", h1)
    for group in (2, 3, 4):
        w0s = [presets[f"lv_group_{group}_p{p:02d}"][0] for p in (5, 10, 20, 40)]
        assert w0s == sorted(w0s) and w0s[0] < w0s[-1]


@pytest.mark.parametrize(
    "wrong, suggestion",
    [
        ("statoin", "'station'"),
        ("Station", "'station'"),
        ("lv-network", "'lv_network'"),
        ("lv_group_2_p5", "'lv_group_2_p05'"),
        ("group_3_p20", "'lv_group_3_p20'"),
    ],
)
def test_an_unknown_preset_names_the_closest_presets(
    wrong: str, suggestion: str
) -> None:
    # user decision Q36
    with pytest.raises(ValueError) as excinfo:
        gf.effective_resistivity(_two_layer(100.0, 30.0, 5.0), preset=wrong)
    message = str(excinfo.value)
    assert message.startswith(f"effective_resistivity: unknown preset {wrong!r}")
    assert suggestion in message
    assert "EFFECTIVE_RESISTIVITY_PRESETS" in message


@pytest.mark.parametrize("wrong", [None, "", 0.06, ("station",), True, ["station"]])
def test_a_preset_that_is_not_a_name_is_rejected(wrong) -> None:
    # user decision Q36
    with pytest.raises(ValueError, match="effective_resistivity: unknown preset"):
        gf.effective_resistivity(rho_1=100.0, rho_2=30.0, h_1=5.0, preset=wrong)


def test_the_preset_is_checked_for_homogeneous_soil_too() -> None:
    """Homogeneous soil does not use the weights, but a wrong preset or an
    invalid explicit weight is still an error (no silent acceptance)."""
    # user decision Q36
    soil = gf.HomogeneousSoil(rho=100.0)
    with pytest.raises(ValueError, match="unknown preset 'stat'"):
        gf.effective_resistivity(soil, preset="stat")
    with pytest.raises(ValueError, match="effective_resistivity: w0 "):
        gf.effective_resistivity(soil, preset="station", w0=-0.1)
    with pytest.raises(ValueError, match="effective_resistivity: beta "):
        gf.effective_resistivity(soil, preset="station", beta=math.nan)


def test_presets_are_exported_from_the_soil_package() -> None:
    # user decision Q36
    import groundfield.soil
    import groundfield.soil.effective

    assert "EFFECTIVE_RESISTIVITY_PRESETS" in groundfield.soil.__all__
    assert "EFFECTIVE_RESISTIVITY_PRESETS" in groundfield.soil.effective.__all__
    assert (
        groundfield.soil.EFFECTIVE_RESISTIVITY_PRESETS
        is groundfield.soil.effective.EFFECTIVE_RESISTIVITY_PRESETS
    )
    assert gf.soil.EFFECTIVE_RESISTIVITY_PRESETS is _presets()


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


# ---------------------------------------------------------------------
# the station preset against field computations (user decision Q36)
# ---------------------------------------------------------------------

#: The 39 soils of the AP1 grid: the 36 two-layer soils and one
#: homogeneous soil per rho_1 == rho_2 (30, 100, 200 Ohm m).
AP1_SOILS_39 = AP1_GRID + [(rho, rho, 10.0) for rho in (30.0, 100.0, 200.0)]


def _ap1_station(r1: float, r2: float, h1: float) -> gf.World:
    """The AP1 station electrode as the AP1 pipeline builds it: a 4 m x 4 m
    ring (7.5 mm) and three 4 m rods (12.5 mm) on a circle of 1.5 m, both
    at 0.6 m depth, bonded; purely resistive."""
    from groundfield.generators import (
        FoundationElectrodeSpec,
        GroundingSystemSpec,
        rod_circle,
    )

    spec = GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                style="ring",
                size_xy_m=(4.0, 4.0),
                depth_m=0.6,
                wire_radius_m=0.0075,
                concrete_rho_ohm_m=None,
            ),
            *rod_circle(
                3, radius_m=1.5, length_m=4.0, depth_m=0.6, wire_radius_m=0.0125
            ),
        ],
        bond_conductor_type="bare_copper",
    )
    soil = gf.HomogeneousSoil(rho=r1) if r1 == r2 else _two_layer(r1, r2, h1)
    world = gf.create_world(soil=soil)
    anchor = spec.build_at(world, (0.0, 0.0), "station", np.random.default_rng(0))
    gf.create_source(world, attached_to=anchor, magnitude=1.0)
    return world


@pytest.fixture(scope="module")
def ap1_station_resistance() -> np.ndarray:
    """50 Hz input resistance of the AP1 station in the 39 AP1 soils."""
    engine = gf.create_engine(frequencies=[50.0], segment_length=0.5)
    with warnings.catch_warnings():
        # advisory: the rod tips end 0.4 m above the interface at h_1 = 5 m
        warnings.filterwarnings("ignore", "image_2layer: deepest segment", UserWarning)
        z = [engine.solve(_ap1_station(*s)).input_impedance()[0] for s in AP1_SOILS_39]
    return np.array(z).real


def _proportional_errors(resistance: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Relative error of R = c * x over the two-layer soils, c = R / rho of
    the homogeneous soils (a resistive electrode scales exactly with rho)."""
    two = np.array([r1 != r2 for r1, r2, _ in AP1_SOILS_39])
    rho = np.array([r1 for r1, _, _ in AP1_SOILS_39])
    c = resistance[~two] / rho[~two]
    np.testing.assert_allclose(c, c[0], rtol=1e-12)
    return np.abs(c[0] * x[two] - resistance[two]) / resistance[two]


def test_the_station_preset_reproduces_the_handover_for_the_ap1_station(
    ap1_station_resistance,
) -> None:
    """The hand-over reports for the station preset a median error of 1.7 %
    and a maximum of 15.7 % over the AP1 soils; R = c * rho_eff reproduces
    them (median over the 39 soils, the homogeneous ones exact; measured
    1.83 % and 15.69 % with the line kernel, 1.71 % and 15.69 % with the
    point kernel of 0.14.0 at 1.5 m segments). The AP1 calibration
    "lv_network" and the upper layer rho_1 describe the station far worse
    (max 35.9 % and 41.9 %)."""
    # user decision Q36
    errors = {}
    for label, kwargs in (
        ("station", {"preset": "station"}),
        ("lv_network", {}),
        ("rho_1", {"w0": 0.0}),
    ):
        x = np.array(
            [
                gf.effective_resistivity(rho_1=a, rho_2=b, h_1=c, **kwargs)
                for a, b, c in AP1_SOILS_39
            ]
        )
        errors[label] = _proportional_errors(ap1_station_resistance, x)
    station = errors["station"]
    median_39 = float(np.median(np.concatenate([station, np.zeros(3)])))
    assert median_39 == pytest.approx(0.017, abs=0.003)
    assert station.max() == pytest.approx(0.157, abs=0.005)
    assert errors["lv_network"].max() > 2.0 * station.max()
    assert errors["rho_1"].max() > 2.5 * station.max()
    rms = {k: float(np.sqrt(np.mean(v**2))) for k, v in errors.items()}
    assert rms["station"] < 0.06 < 0.11 < rms["lv_network"] < rms["rho_1"]
