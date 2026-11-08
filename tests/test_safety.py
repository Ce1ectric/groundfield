"""Tests for the safety post-processing helpers (touch / step voltage).

Validates :mod:`groundfield.postprocess.safety` against:

- the closed-form Sunde solution for a single rod in homogeneous
  soil (touch and step voltage have known order of magnitude),
- physical monotonicities (U_T grows towards U_E far away from
  the electrode, |U_S| decays towards remote earth),
- symmetry under cylindrically symmetric geometry,
- the published values of EN 50522:2022, Table B.4 (permissible touch
  voltage U_Tp) and Table B.1 (permissible body current I_B), entered
  here as literals,
- the prospective (open-circuit) touch voltage of EN 50522:2022,
  Annex B.4, and the true-RMS touch voltage over a fault spectrum.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.postprocess.safety import (
    permissible_touch_voltage_en50522,
    step_voltage,
    touch_voltage,
    touch_voltage_envelope,
)

# EN 50522:2022 (DIN EN 50522 (VDE 0101-2):2023-10), Table B.4: fault
# duration t_f in s -> permissible touch voltage U_Tp in V (values rounded
# to 5 V in the standard). Literals on purpose: the test must not read the
# module's own table.
_TABLE_B4 = (
    (0.05, 725.0),
    (0.10, 655.0),
    (0.20, 525.0),
    (0.50, 225.0),
    (1.00, 115.0),
    (2.00, 95.0),
    (5.00, 85.0),
    (10.00, 85.0),
)
# EN 50522:2022, Table B.1: fault duration t_f in s -> permissible body
# current I_B in A (curve c2 of IEC 60479-1:2018, left hand to both feet).
_TABLE_B1 = (
    (0.05, 0.900),
    (0.10, 0.800),
    (0.20, 0.600),
    (0.50, 0.200),
    (1.00, 0.080),
    (2.00, 0.060),
    (5.00, 0.051),
    (10.00, 0.050),
)


# ---------------------------------------------------------------------
# Common fixture: a single rod in homogeneous soil
# ---------------------------------------------------------------------


def _single_rod_world(
    *, rho: float = 100.0, length: float = 1.5, current: float = 10.0
) -> tuple[gf.World, gf.FieldResult]:
    soil = gf.HomogeneousSoil(resistivity=rho)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world,
        "rod",
        name="g1",
        position=(0.0, 0.0, 0.5),
        length=length,
        wire_radius=0.005,
    )
    gf.create_source(world, attached_to="g1", magnitude=current)
    eng = gf.create_engine(backend="image", segment_length=0.05)
    return world, eng.solve(world)


# ---------------------------------------------------------------------
# touch_voltage
# ---------------------------------------------------------------------


def test_touch_voltage_positive_close_to_rod() -> None:
    """U_T at 1 m around a current-injecting rod must be positive and below U_E."""
    world, result = _single_rod_world()
    U_E = result.electrode_potentials["g1"][0].real

    U_T = touch_voltage(result, world, electrode="g1", distance=1.0).real
    assert U_T > 0.0
    assert U_T < U_E


def test_touch_voltage_grows_with_distance_to_remote_earth() -> None:
    """As the feet recede, U_T -> U_E (potential at the feet -> 0)."""
    world, result = _single_rod_world()
    U_E = result.electrode_potentials["g1"][0].real

    U_close = touch_voltage(result, world, electrode="g1", distance=1.0).real
    U_mid = touch_voltage(result, world, electrode="g1", distance=10.0).real
    U_far = touch_voltage(result, world, electrode="g1", distance=200.0).real

    assert U_close < U_mid < U_far
    # At 200 m the surface potential of a 1.5 m rod injecting 10 A in
    # 100 Ω·m soil has decayed by far more than 99 % — feet ≈ remote
    # earth.
    assert abs(U_far - U_E) / U_E < 1e-2


def test_touch_voltage_rejects_invalid_distance() -> None:
    world, result = _single_rod_world()
    with pytest.raises(ValueError, match="distance"):
        touch_voltage(result, world, electrode="g1", distance=0.0)
    with pytest.raises(ValueError, match="distance"):
        touch_voltage(result, world, electrode="g1", distance=-1.0)


def test_touch_voltage_rejects_unknown_electrode() -> None:
    world, result = _single_rod_world()
    with pytest.raises(KeyError, match="missing"):
        touch_voltage(result, world, electrode="missing", distance=1.0)


def test_touch_voltage_horizontal_projection_strips_z() -> None:
    """A direction with non-zero z but zero (x, y) is rejected."""
    world, result = _single_rod_world()
    with pytest.raises(ValueError, match="horizontal"):
        touch_voltage(
            result, world, electrode="g1", distance=1.0, direction=(0, 0, 1.0)
        )


# ---------------------------------------------------------------------
# touch_voltage_envelope
# ---------------------------------------------------------------------


def test_envelope_uniform_for_centred_rod() -> None:
    """A vertical rod on the z-axis is cylindrically symmetric — every angle
    must yield the same touch voltage."""
    world, result = _single_rod_world()
    angles, voltages = touch_voltage_envelope(
        result, world, electrode="g1", distance=1.0, n_angles=12
    )
    assert angles.shape == (12,)
    assert voltages.shape == (12,)

    spread = float(voltages.real.max() - voltages.real.min())
    mean = float(voltages.real.mean())
    assert spread / abs(mean) < 1e-6


def test_envelope_matches_pointwise_touch_voltage() -> None:
    """The envelope at angle 0 must equal touch_voltage with direction +x."""
    world, result = _single_rod_world()
    angles, voltages = touch_voltage_envelope(
        result, world, electrode="g1", distance=1.0, n_angles=8
    )
    pointwise = touch_voltage(
        result, world, electrode="g1", distance=1.0, direction=(1.0, 0.0, 0.0)
    )
    # angle index 0 corresponds to (cos 0, sin 0) = (+x).
    assert voltages[0].real == pytest.approx(pointwise.real, rel=1e-12)


def test_envelope_rejects_too_few_angles() -> None:
    world, result = _single_rod_world()
    with pytest.raises(ValueError, match="n_angles"):
        touch_voltage_envelope(result, world, electrode="g1", distance=1.0, n_angles=2)


# ---------------------------------------------------------------------
# step_voltage
# ---------------------------------------------------------------------


def test_step_voltage_decays_to_remote_earth() -> None:
    """|U_S| must decay monotonically with distance from the source."""
    world, result = _single_rod_world()
    U_S_close = abs(
        step_voltage(result, position=(1.0, 0.0, 0.0), step=1.0, direction=(1, 0, 0))
    )
    U_S_mid = abs(
        step_voltage(result, position=(10.0, 0.0, 0.0), step=1.0, direction=(1, 0, 0))
    )
    U_S_far = abs(
        step_voltage(result, position=(100.0, 0.0, 0.0), step=1.0, direction=(1, 0, 0))
    )
    assert U_S_close > U_S_mid > U_S_far
    # 100 m away from a 1.5 m rod, |U_S| over 1 m must be tiny.
    assert U_S_far < 1e-2 * U_S_close


def test_step_voltage_sign_matches_potential_gradient() -> None:
    """Stepping outward from the rod, U_S = phi(near) - phi(far) > 0."""
    _world, result = _single_rod_world()
    U_S = step_voltage(
        result, position=(1.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), step=1.0
    )
    # phi(1, 0, 0) > phi(2, 0, 0) for an outward-pointing source field.
    assert U_S.real > 0.0


def test_step_voltage_rejects_invalid_step() -> None:
    _world, result = _single_rod_world()
    with pytest.raises(ValueError, match="step"):
        step_voltage(result, position=(1.0, 0.0, 0.0), step=0.0)
    with pytest.raises(ValueError, match="step"):
        step_voltage(result, position=(1.0, 0.0, 0.0), step=-0.5)


def test_step_voltage_horizontal_projection_strips_z() -> None:
    _world, result = _single_rod_world()
    with pytest.raises(ValueError, match="horizontal"):
        step_voltage(
            result, position=(1.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), step=1.0
        )


# ---------------------------------------------------------------------
# permissible_touch_voltage_en50522
# ---------------------------------------------------------------------


@pytest.mark.parametrize(("t_f", "u_tp"), _TABLE_B4)
def test_en50522_matches_table_b4_literals(t_f: float, u_tp: float) -> None:
    """U_Tp(t_f) reproduces EN 50522:2022, Table B.4 at every tabulated t_f.

    The expected values are literals transcribed from the standard (the
    v0.15.0 test compared the interpolant with the module constant it
    interpolates, which could never fail). Review pass 10 (10.42)
    confirmed the module values against the standard, so this guard
    passes on v0.15.0 as well and carries no finding marker. The
    log-log interpolation returns an anchor to within one ulp.
    """
    assert permissible_touch_voltage_en50522(t_f) == pytest.approx(u_tp, rel=1e-12)


@pytest.mark.finding("10.42")
def test_en50522_docstrings_cite_the_current_edition() -> None:
    """The helper cites EN 50522:2022 Table B.4, not the 2010 figure."""
    from groundfield.postprocess import safety

    for doc in (safety.__doc__, permissible_touch_voltage_en50522.__doc__):
        assert "EN 50522:2022" in doc
        assert "Table B.4" in doc
        assert "Figure B.3" not in doc
        assert "EN 50522:2010" not in doc
        assert "low-voltage standard curve" not in doc


@pytest.mark.finding("10.42")
@pytest.mark.parametrize("t_f", [10.000001, 10.5, 60.0, 3600.0, 1e6, math.inf])
def test_en50522_beyond_ten_seconds_is_80_volt(t_f: float) -> None:
    """EN 50522:2022 as published: Table B.4 (the curve of Figure 8) up to
    10 s, 80 V beyond.

    The curve of Figure 8 is not defined beyond 10 s; the note to Figure 8
    (5.4.1) and the text after Table B.4 (B.3) allow 80 V for current-flow
    durations considerably longer than 10 s. Resonant-earthed networks carry
    the earth-fault current for such durations, so every ``t_f > 10 s``,
    including a fault that is not cleared (``math.inf``), returns 80 V (below
    the 85 V end value of the table, on the safe side). The base returned
    85 V and rejected ``math.inf`` without a keyword.
    """
    # user decision Q4
    assert permissible_touch_voltage_en50522(t_f) == 80.0
    assert permissible_touch_voltage_en50522(10.0) == 85.0  # last row of Table B.4


@pytest.mark.finding("10.42")
def test_en50522_has_no_long_duration_keyword() -> None:
    """The 80 V rule is the default beyond 10 s; the ``long_duration`` keyword
    of earlier development snapshots is gone (it never shipped)."""
    # user decision Q4
    from groundfield.postprocess.safety import (
        permissible_prospective_touch_voltage_en50522 as u_vtp,
    )

    with pytest.raises(TypeError, match="long_duration"):
        permissible_touch_voltage_en50522(3600.0, long_duration=True)
    with pytest.raises(TypeError, match="long_duration"):
        u_vtp(3600.0, long_duration=True)


@pytest.mark.finding("10.42")
def test_en50522_docstring_cites_the_clauses_of_the_80_volt_rule() -> None:
    """The docstring names where the rule comes from: Figure 8 and its note
    (5.4.1), Table B.4 and the text after it (B.3), and B.4."""
    # user decision Q4
    from groundfield.postprocess.safety import (
        permissible_prospective_touch_voltage_en50522 as u_vtp,
    )

    doc = permissible_touch_voltage_en50522.__doc__
    for clause in ("Figure 8", "5.4.1", "Table B.4", "B.3", "80 V", "math.inf"):
        assert clause in doc, clause
    assert "long_duration" not in doc
    assert "B.4" in u_vtp.__doc__ and "long_duration" not in u_vtp.__doc__


def test_en50522_strictly_decreasing_inside_grid() -> None:
    """U_TP(t) is monotonically decreasing on [50 ms, 10 s]."""
    ts = np.geomspace(0.05, 10.0, 50)
    us = np.array([permissible_touch_voltage_en50522(t) for t in ts])
    diffs = np.diff(us)
    assert (diffs <= 1e-9).all()


@pytest.mark.finding("10.42")
def test_en50522_never_increases_with_the_fault_duration() -> None:
    """From 1 ms to an uncleared fault the permissible value never rises
    (725 V clamp below 50 ms, Table B.4, 80 V beyond 10 s)."""
    # user decision Q4
    ts = [*np.geomspace(1e-3, 1e4, 400), math.inf]
    us = np.array([permissible_touch_voltage_en50522(float(t)) for t in ts])
    assert np.all(np.diff(us) <= 1e-9)
    assert us[0] == 725.0 and us[-1] == 80.0


def test_en50522_clamped_below_the_table() -> None:
    """Below 50 ms the first row of Table B.4 is returned (on the safe side:
    the curve rises towards shorter durations)."""
    assert permissible_touch_voltage_en50522(1e-3) == 725.0
    assert permissible_touch_voltage_en50522(0.05) == 725.0


def test_en50522_rejects_invalid_time() -> None:
    with pytest.raises(ValueError, match="t_clear_s"):
        permissible_touch_voltage_en50522(0.0)
    with pytest.raises(ValueError, match="t_clear_s"):
        permissible_touch_voltage_en50522(-0.1)
    with pytest.raises(ValueError, match="t_clear_s"):
        permissible_touch_voltage_en50522(float("nan"))
    with pytest.raises(ValueError, match="t_clear_s"):
        permissible_touch_voltage_en50522(-math.inf)
    with pytest.raises(TypeError):  # a duration is a number, not text (as in 0.15.0)
        permissible_touch_voltage_en50522("20")


def test_en50522_loglog_interpolation_in_geometric_mean() -> None:
    """Halfway in log time between two anchors the value must equal the
    geometric mean of the corresponding anchor voltages."""
    # Anchors (1.0 s, 115 V) and (2.0 s, 95 V) — geometric mean of t is
    # sqrt(2) s, expected U is sqrt(115 * 95) V.
    t_mid = math.sqrt(2.0)
    u_expected = math.sqrt(115.0 * 95.0)
    assert permissible_touch_voltage_en50522(t_mid) == pytest.approx(
        u_expected, rel=1e-12
    )


def test_en50522_terminal_plateau_constant() -> None:
    """Between t = 5 s and t = 10 s the standard's table is flat at 85 V."""
    for t in (5.0, 6.0, 7.5, 9.0, 10.0):
        assert permissible_touch_voltage_en50522(t) == pytest.approx(85.0, rel=1e-12)


# ---------------------------------------------------------------------
# permissible_prospective_touch_voltage_en50522 (EN 50522:2022, B.4)
# ---------------------------------------------------------------------


@pytest.mark.finding("10.52")
@pytest.mark.parametrize(("t_f", "i_b"), _TABLE_B1)
def test_prospective_touch_voltage_without_resistance_is_u_tp(
    t_f: float, i_b: float
) -> None:
    """R_A = 0 leaves U_vTp = U_Tp (Eq. A.3 with R_H = R_F = 0)."""
    from groundfield.postprocess.safety import (
        permissible_prospective_touch_voltage_en50522 as u_vtp,
    )

    assert u_vtp(t_f) == pytest.approx(permissible_touch_voltage_en50522(t_f))


@pytest.mark.finding("10.52")
@pytest.mark.parametrize(("t_f", "i_b"), _TABLE_B1)
def test_prospective_touch_voltage_uses_table_b1_body_current(
    t_f: float, i_b: float
) -> None:
    """U_vTp - U_Tp = I_B(t_f) / HF * R_A with I_B from Table B.1."""
    from groundfield.postprocess.safety import (
        permissible_prospective_touch_voltage_en50522 as u_vtp,
    )

    r_a = 1000.0
    u_tp = dict(_TABLE_B4)[t_f]
    assert u_vtp(t_f, R_F1=r_a) == pytest.approx(u_tp + i_b * r_a, rel=1e-12)
    # Hand resistance and footwear resistance add up (R_A = R_H + R_F1 + R_F2).
    assert u_vtp(t_f, R_H=400.0, R_F1=600.0) == pytest.approx(
        u_tp + i_b * r_a, rel=1e-12
    )
    # A heart-current factor below 1 raises the admissible voltage.
    assert u_vtp(t_f, R_F1=r_a, HF=0.5) == pytest.approx(
        u_tp + i_b / 0.5 * r_a, rel=1e-12
    )


@pytest.mark.finding("10.52")
@pytest.mark.parametrize(
    ("t_f", "r_f1", "rho_s", "expected"),
    [
        # Figure B.3 parameter sets of EN 50522:2022 (R_F = R_F1 + 1.5 m^-1 rho_s)
        (0.5, 0.0, 500.0, 225.0 + 0.200 * 750.0),
        (1.0, 1000.0, 500.0, 115.0 + 0.080 * 1750.0),
        (2.0, 1000.0, 1000.0, 95.0 + 0.060 * 2500.0),
        (10.0, 1000.0, 2000.0, 85.0 + 0.050 * 4000.0),
    ],
)
def test_prospective_touch_voltage_figure_b3_parameter_sets(
    t_f: float, r_f1: float, rho_s: float, expected: float
) -> None:
    """R_F2 = 1.5 m^-1 * rho_s is added for the surface resistivity."""
    from groundfield.postprocess.safety import (
        permissible_prospective_touch_voltage_en50522 as u_vtp,
    )

    assert u_vtp(t_f, R_F1=r_f1, rho_s=rho_s) == pytest.approx(expected, rel=1e-12)


@pytest.mark.finding("10.52")
def test_prospective_touch_voltage_validates_inputs() -> None:
    from groundfield.postprocess.safety import (
        permissible_prospective_touch_voltage_en50522 as u_vtp,
    )

    with pytest.raises(ValueError, match="R_H"):
        u_vtp(0.5, R_H=-1.0)
    with pytest.raises(ValueError, match="R_F1"):
        u_vtp(0.5, R_F1=float("nan"))
    with pytest.raises(ValueError, match="rho_s"):
        u_vtp(0.5, rho_s=-10.0)
    with pytest.raises(ValueError, match="HF"):
        u_vtp(0.5, HF=0.0)
    # Monotone in the additional resistance and in time (R_A fixed).
    ts = np.geomspace(0.05, 10.0, 40)
    vals = [u_vtp(float(t), R_F1=1000.0, rho_s=100.0) for t in ts]
    assert np.all(np.diff(vals) <= 1e-9)


@pytest.mark.finding("10.42")
@pytest.mark.parametrize("t_f", [20.0, 3600.0, math.inf])
def test_prospective_touch_voltage_beyond_ten_seconds_uses_80_volt(t_f: float) -> None:
    """B.4 with the U_Tp of Figure 8: beyond 10 s, 80 V plus the body current
    of the last row of Table B.1 (50 mA) times R_A / HF."""
    # user decision Q4
    from groundfield.postprocess.safety import (
        permissible_prospective_touch_voltage_en50522 as u_vtp,
    )

    assert u_vtp(t_f) == 80.0
    assert u_vtp(t_f, R_F1=1000.0) == pytest.approx(80.0 + 0.050 * 1000.0, rel=1e-12)
    assert u_vtp(t_f, R_F1=1000.0, rho_s=500.0, HF=0.8) == pytest.approx(
        80.0 + 0.050 / 0.8 * (1000.0 + 750.0), rel=1e-12
    )


# ---------------------------------------------------------------------
# touch_voltage_spectrum (true RMS over a fault spectrum)
# ---------------------------------------------------------------------


def _rod_spectrum_world(frequencies: list[float]) -> tuple[gf.World, gf.FieldResult]:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(world, attached_to="g1", magnitude=2.0, phase_deg=30.0)
    eng = gf.create_engine(backend="mom", segment_length=0.25, frequencies=frequencies)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world, eng.solve(world)


@pytest.mark.finding("10.52")
def test_touch_voltage_spectrum_is_the_rms_of_the_scaled_harmonics() -> None:
    """U_rms = sqrt(sum_k |U_T(f_k) I_k / I_src(f_k)|^2) at one foot position."""
    from groundfield.postprocess.safety import touch_voltage_spectrum

    freqs = [50.0, 150.0, 250.0]
    currents = [100.0, 10.0 - 4.0j, 5.0]
    world, res = _rod_spectrum_world(freqs)
    i_src = 2.0 * np.exp(1j * math.radians(30.0))
    per_harmonic = [
        touch_voltage(res, world, electrode="g1", frequency_index=k) / i_src * i_k
        for k, i_k in enumerate(currents)
    ]
    expected = math.sqrt(sum(abs(u) ** 2 for u in per_harmonic))
    u_rms = touch_voltage_spectrum(res, world, electrode="g1", currents=currents)
    assert u_rms == pytest.approx(expected, rel=1e-12)
    # Galvanic rod: U_T per ampere is frequency independent, so the RMS is
    # U_T(1 A) times the RMS of the current spectrum.
    u_1a = abs(touch_voltage(res, world, electrode="g1") / i_src)
    i_rms = math.sqrt(sum(abs(i) ** 2 for i in currents))
    assert u_rms == pytest.approx(u_1a * i_rms, rel=1e-9)


@pytest.mark.finding("10.52")
def test_touch_voltage_spectrum_accepts_results_per_frequency_and_mappings() -> None:
    from groundfield.postprocess.safety import touch_voltage_spectrum

    freqs = [50.0, 250.0]
    world, multi = _rod_spectrum_world(freqs)
    singles = [_rod_spectrum_world([f])[1] for f in freqs]
    a = touch_voltage_spectrum(multi, world, electrode="g1", currents=[50.0, 7.0])
    b = touch_voltage_spectrum(singles, world, electrode="g1", currents=[50.0, 7.0])
    c = touch_voltage_spectrum(
        multi, world, electrode="g1", currents={250.0: 7.0, 50.0: 50.0}
    )
    assert b == pytest.approx(a, rel=1e-12)
    assert c == pytest.approx(a, rel=1e-12)


@pytest.mark.finding("10.52")
def test_touch_voltage_spectrum_rejects_inconsistent_currents() -> None:
    from groundfield.postprocess.safety import touch_voltage_spectrum

    world, res = _rod_spectrum_world([50.0, 150.0])
    with pytest.raises(ValueError, match="2 frequencies"):
        touch_voltage_spectrum(res, world, electrode="g1", currents=[1.0])
    with pytest.raises(ValueError, match="150"):
        touch_voltage_spectrum(res, world, electrode="g1", currents={50.0: 1.0})
    with pytest.raises(ValueError, match="finite"):
        touch_voltage_spectrum(res, world, electrode="g1", currents=[1.0, math.nan])


@pytest.mark.finding("10.52")
def test_touch_voltage_spectrum_rejects_a_frequency_given_twice() -> None:
    """Audit 1, L4: a frequency carried by two results was counted twice
    (the mapping form gave sqrt(2) times the RMS: 701.6 V instead of
    496.1 V). The harmonics of a spectrum have distinct frequencies."""
    from groundfield.postprocess.safety import touch_voltage_spectrum

    world, r50 = _rod_spectrum_world([50.0])
    _, r_both = _rod_spectrum_world([50.0, 150.0])
    one = touch_voltage_spectrum(r50, world, electrode="g1", currents={50.0: 100.0})
    assert one > 0.0
    for results, currents in (
        ([r50, r50], {50.0: 100.0}),
        ([r50, r50], [100.0, 100.0]),
        ([r_both, r50], {50.0: 100.0, 150.0: 5.0}),
    ):
        with pytest.raises(ValueError, match="50 Hz appears more than once"):
            touch_voltage_spectrum(results, world, electrode="g1", currents=currents)


# ---------------------------------------------------------------------
# Integration: top-level export
# ---------------------------------------------------------------------


def test_top_level_exports_safety_helpers() -> None:
    needed = {
        "touch_voltage",
        "touch_voltage_envelope",
        "step_voltage",
        "permissible_touch_voltage_en50522",
    }
    assert needed.issubset(set(gf.__all__))
    assert all(hasattr(gf, name) for name in needed)
