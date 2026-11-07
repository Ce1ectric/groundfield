"""Opt-in skin effect of conductors (review pass 10, finding 10.37).

Reference numbers: R3/t19 (exact Bessel solution, cross-checked with the
IEC 60287 skin factor) and the IEC 60287-1-1 formula evaluated here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from groundfield.conductors import Conductor, solid_round_internal_impedance

pytestmark = pytest.mark.finding("10.37")

MU_0 = 4e-7 * math.pi
RHO_AL = 2.82e-8
A_150 = 150e-6


def _pen(**kw) -> Conductor:
    base = dict(
        name="pen",
        start=(0.0, 0.0, 0.7),
        end=(300.0, 0.0, 0.7),
        resistivity=RHO_AL,
        cross_section=A_150,
        wire_radius=math.sqrt(A_150 / math.pi),
    )
    base.update(kw)
    return Conductor(**base)


@pytest.mark.parametrize(
    "freq, ratio", [(250.0, 1.056), (550.0, 1.231), (1000.0, 1.550)]
)
def test_150mm2_aluminium_resistance_ratio(freq: float, ratio: float) -> None:
    z = _pen(resistance_model="skin").internal_impedance_per_length(freq)
    assert z.real / (RHO_AL / A_150) == pytest.approx(ratio, abs=1.5e-3)


def test_internal_inductance_drops_at_1khz() -> None:
    z = _pen(resistance_model="skin").internal_impedance_per_length(1000.0)
    l_int = z.imag / (2.0 * math.pi * 1000.0)
    assert l_int / (MU_0 / (8.0 * math.pi)) == pytest.approx(0.740, abs=2e-3)


@pytest.mark.parametrize(
    "area, rho", [(150e-6, RHO_AL), (95e-6, RHO_AL), (50e-6, 1.72e-8), (16e-6, 1.72e-8)]
)
@pytest.mark.parametrize("freq", [50.0, 250.0, 550.0, 1000.0])
def test_agrees_with_iec60287_skin_factor(area, rho, freq) -> None:
    """IEC 60287-1-1 y_s (k_s = 1) is accurate to ~1 % for x_s <= 2.8."""
    r_dc = rho / area
    x_s2 = 8.0 * math.pi * freq / r_dc * 1e-7
    y_s = x_s2**2 / (192.0 + 0.8 * x_s2**2)
    z = solid_round_internal_impedance(
        freq, radius=math.sqrt(area / math.pi), resistivity=rho
    )
    assert z.real / r_dc == pytest.approx(1.0 + y_s, rel=1.2e-2)


def test_dc_and_high_frequency_limits() -> None:
    a = math.sqrt(A_150 / math.pi)
    z0 = solid_round_internal_impedance(0.0, radius=a, resistivity=RHO_AL)
    assert z0 == pytest.approx(RHO_AL / A_150, rel=1e-15)
    z_lo = solid_round_internal_impedance(1e-3, radius=a, resistivity=RHO_AL)
    assert z_lo.imag / (2 * math.pi * 1e-3) == pytest.approx(
        MU_0 / (8 * math.pi), rel=1e-9
    )
    f = 1e6
    delta = math.sqrt(2.0 * RHO_AL / (2.0 * math.pi * f * MU_0))
    z_hi = solid_round_internal_impedance(f, radius=a, resistivity=RHO_AL)
    surface = (1 + 1j) * RHO_AL / (2.0 * math.pi * a * delta)
    assert abs(z_hi / surface - 1.0) < 0.01


def test_vectorised_and_continuous_across_series_switch() -> None:
    a = math.sqrt(A_150 / math.pi)
    f = np.geomspace(1e-4, 1e5, 400)
    z = solid_round_internal_impedance(f, radius=a, resistivity=RHO_AL)
    scalar = [
        solid_round_internal_impedance(float(x), radius=a, resistivity=RHO_AL)
        for x in f[::37]
    ]
    np.testing.assert_allclose(z[::37], scalar, rtol=1e-15, atol=0)
    assert np.all(np.diff(z.real) >= 0.0)  # resistance rises monotonically


def test_dc_model_is_the_default_and_unchanged() -> None:
    pen = _pen()
    assert pen.resistance_model == "dc"
    z = pen.internal_impedance_per_length(1000.0)
    assert z.real == pytest.approx(pen.series_resistance / pen.length, rel=1e-15)
    assert z.imag == pytest.approx(
        2 * math.pi * 1000.0 * MU_0 / (8 * math.pi), rel=1e-15
    )
    assert pen.skin_correction_per_length(1000.0) == 0j
    np.testing.assert_array_equal(
        pen.skin_correction_per_length(np.array([0.0, 50.0])), 0j
    )


def test_skin_correction_is_relative_to_the_dc_convention() -> None:
    pen = _pen(resistance_model="skin")
    f = 1000.0
    dz = pen.skin_correction_per_length(f)
    z = pen.internal_impedance_per_length(f)
    dc = RHO_AL / A_150 + 1j * 2 * math.pi * f * MU_0 / (8 * math.pi)
    assert dz == pytest.approx(z - dc, rel=1e-14)
    assert pen.skin_correction_per_length(0.0) == pytest.approx(0j, abs=1e-18)


def test_skin_model_needs_a_cross_section() -> None:
    with pytest.raises(ValueError, match="cross_section"):
        _pen(resistance_model="skin", cross_section=None)
    with pytest.raises(ValueError, match="lumped_series_resistance_ohm"):
        _pen(resistance_model="skin", lumped_series_resistance_ohm=0.01)


def test_resistance_model_survives_json_round_trip() -> None:
    pen = _pen(resistance_model="skin")
    again = Conductor.model_validate_json(pen.model_dump_json())
    assert again.resistance_model == "skin"
    assert again.internal_impedance_per_length(
        550.0
    ) == pen.internal_impedance_per_length(550.0)
