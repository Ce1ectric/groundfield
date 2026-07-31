r"""Review-pass-9 regression tests for the analytic reference modules.

Covers the findings fixed in 0.15.0:

* **F09** — :func:`groundfield.references.pollaczek.pollaczek_self_impedance`
  and :func:`~groundfield.references.pollaczek.pollaczek_mutual_impedance`
  evaluated the earth-return correction integral with Carson's *air*
  kernel :math:`e^{-2h\lambda}` instead of the buried-conductor
  (Pollaczek/Sunde) kernel :math:`e^{-2h\,u_\text{eff}}`. The air kernel is
  the :math:`\gamma \to 0` limit that holds *above* the interface only;
  below it the vertical decay of the reflected field is set by
  :math:`u = \sqrt{\lambda^2 + j\omega\mu_0/\rho}`. Consequences of the
  bug: :math:`R'` was under-counted (0.5 % at 50 Hz / 1 m, 2.1 % at
  1 kHz / 1 m, 5.8 % at 1 kHz / 3 m) and — the qualitative giveaway —
  :math:`R'` *fell* with burial depth and stayed *below* Carson's
  :math:`\omega\mu_0/8`, whereas a buried conductor's earth-return
  resistance must *rise* with depth and lie *above* that surface-return
  limit.
* **F37** — :func:`groundfield.references.dwight1936.horizontal_strip`
  carried the round-wire constant :math:`-2` instead of the strip
  constant :math:`-1` (~10 % low).
* **F38** — :func:`groundfield.references.dwight1936.vertical_round_plate`
  divided the image term by :math:`4\pi\,(2s)` although ``s`` is already
  the plate-to-image distance :math:`2t`, halving the image term.
* **F55** — the :mod:`groundfield.references.earth_return` module
  docstring claimed the earth-return reactance *decreases* with
  frequency. Only :math:`D_e \propto \sqrt{\rho/f}` does; the
  :math:`\omega` prefactor of
  :math:`X' = (\omega\mu_0/2\pi)\ln(D_e/\mathrm{GMR})` dominates, so
  :math:`X'` rises almost linearly with :math:`f`.

The reference values below are produced by independent quadrature of the
textbook forms, not by the module under test.
"""
from __future__ import annotations

import cmath
import math
import re

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from groundfield.references import earth_return as er
from groundfield.references.dwight1936 import (
    horizontal_round_plate,
    horizontal_strip,
    horizontal_wire,
    vertical_round_plate,
)
from groundfield.references.earth_return import MU_0, carson_self_impedance
from groundfield.references.pollaczek import (
    pollaczek_mutual_impedance,
    pollaczek_self_impedance,
)

# A representative buried PEN conductor (150 mm^2 Al).
GMR = 0.7788 * math.sqrt(150e-6 / math.pi)


# =====================================================================
# F09 — buried-conductor (Pollaczek/Sunde) kernel exp(-2 h u)
# =====================================================================
def _reference_self(
    rho: float, freq: float, depth: float, gmr: float, *, head: str = "log"
) -> complex:
    r"""Independent homogeneous Pollaczek self impedance (Ω/m).

    .. math::

        Z' = j\frac{\omega\mu_0}{2\pi}\Bigl[
             K_0(\gamma\,\mathrm{GMR}) - K_0(2\gamma h)
             + 2\int_0^\infty \frac{e^{-2hu}}{\lambda+u}\,d\lambda\Bigr],
        \quad u = \sqrt{\lambda^2+\gamma^2},\ \gamma^2 = j\omega\mu_0/\rho.

    ``head="log"`` replaces the :math:`K_0` pair by its small-argument
    form :math:`\ln(2h/\mathrm{GMR})`, which is what the module uses —
    that isolates the kernel under test from the head-term
    approximation. ``head="k0"`` is the full textbook form.
    """
    omega = 2.0 * math.pi * freq
    g2 = 1j * omega * MU_0 / rho
    gamma = cmath.sqrt(g2)

    def integrand(lam: float, part: int) -> float:
        u = cmath.sqrt(lam * lam + g2)
        v = cmath.exp(-2.0 * depth * u) / (lam + u)
        return v.real if part == 0 else v.imag

    re_, _ = quad(integrand, 0.0, np.inf, args=(0,), limit=400)
    im_, _ = quad(integrand, 0.0, np.inf, args=(1,), limit=400)
    if head == "log":
        head_term: complex = complex(math.log(2.0 * depth / gmr), 0.0)
    else:
        head_term = kv(0, gamma * gmr) - kv(0, gamma * 2.0 * depth)
    return 1j * omega * MU_0 / (2.0 * math.pi) * (head_term + 2.0 * (re_ + 1j * im_))


def _reference_mutual(
    rho: float, freq: float, h_i: float, h_j: float, x: float
) -> complex:
    """Independent homogeneous Pollaczek mutual impedance (Ω/m), K0 head."""
    omega = 2.0 * math.pi * freq
    g2 = 1j * omega * MU_0 / rho
    gamma = cmath.sqrt(g2)
    d = math.hypot(x, h_i - h_j)
    big_d = math.hypot(x, h_i + h_j)

    def integrand(lam: float, part: int) -> float:
        u = cmath.sqrt(lam * lam + g2)
        v = cmath.exp(-(h_i + h_j) * u) / (lam + u) * math.cos(lam * x)
        return v.real if part == 0 else v.imag

    re_, _ = quad(integrand, 0.0, np.inf, args=(0,), limit=400)
    im_, _ = quad(integrand, 0.0, np.inf, args=(1,), limit=400)
    head = kv(0, gamma * d) - kv(0, gamma * big_d)
    return 1j * omega * MU_0 / (2.0 * math.pi) * (head + 2.0 * (re_ + 1j * im_))


@pytest.mark.parametrize("freq", [50.0, 500.0, 1000.0, 10_000.0])
@pytest.mark.parametrize("depth", [0.7, 1.0, 3.0])
def test_self_uses_buried_kernel_not_carson_air_kernel(freq, depth):
    """R' and X' match the exp(-2 h u) kernel to 1e-5 relative.

    Same head term (``ln(2h/GMR)``) on both sides, so the only thing
    compared is the exponential kernel. Up to 0.14.1 the module used
    ``exp(-2 h lam)``, which is off by -0.5 % (50 Hz / 0.7 m) to -15 %
    (10 kHz / 3 m) in R'.
    """
    z = pollaczek_self_impedance(100.0, 100.0, 5.0, freq, depth=depth, gmr=GMR)
    z_ref = _reference_self(100.0, freq, depth, GMR, head="log")
    assert z.real == pytest.approx(z_ref.real, rel=1e-5)
    assert z.imag == pytest.approx(z_ref.imag, rel=1e-5)


@pytest.mark.parametrize("freq", [50.0, 500.0, 1000.0])
def test_self_matches_full_k0_textbook_form_inside_validity_envelope(freq):
    """Within the documented envelope (|gamma| 2h << 1, f <= 1 kHz, h = 1 m)
    the module also matches the *full* K0-head textbook form to 0.1 %.

    The residual is the small-argument head approximation, not the kernel;
    with the 0.14.1 air kernel the deviation was 0.5 % (50 Hz) to 2.1 %
    (1 kHz) and had the opposite sign convention in depth.
    """
    z = pollaczek_self_impedance(100.0, 100.0, 5.0, freq, depth=1.0, gmr=GMR)
    z_ref = _reference_self(100.0, freq, 1.0, GMR, head="k0")
    assert z.real == pytest.approx(z_ref.real, rel=1e-3)
    assert z.imag == pytest.approx(z_ref.imag, rel=1e-3)


@pytest.mark.parametrize("rho", [30.0, 100.0, 1000.0])
@pytest.mark.parametrize("freq", [50.0, 250.0, 1000.0])
def test_self_resistance_rises_with_burial_depth(rho, freq):
    """R' increases strictly with burial depth (deeper -> more soil in the
    return path). The 0.14.1 air kernel made it *decrease* with depth."""
    depths = [0.5, 0.7, 1.0, 2.0, 3.0]
    r = [
        pollaczek_self_impedance(rho, rho, 10.0, freq, depth=d, gmr=GMR).real
        for d in depths
    ]
    assert all(b > a for a, b in zip(r, r[1:])), r


@pytest.mark.parametrize("rho", [30.0, 100.0, 1000.0])
@pytest.mark.parametrize("freq", [50.0, 250.0, 1000.0])
def test_self_resistance_exceeds_carson_surface_limit(rho, freq):
    """R' lies above omega*mu0/8 — Carson's low-frequency surface-return
    limit — and the excess grows with frequency.

    This inverts the pre-0.15.0 claim (docstrings, docs and
    ``tests/test_pollaczek_reference.py``) that R' approaches
    omega*mu0/8 *from below* with a deficit growing with frequency; that
    was the signature of the air-kernel bug, not physics.
    """
    r_carson = 2.0 * math.pi * freq * MU_0 / 8.0
    z = pollaczek_self_impedance(rho, rho, 10.0, freq, depth=1.0, gmr=GMR)
    assert z.real > r_carson
    assert z.real == pytest.approx(r_carson, rel=5e-2)


@pytest.mark.parametrize("rho", [30.0, 1000.0])
def test_self_resistance_excess_over_carson_grows_with_frequency(rho):
    """R'/(omega*mu0/8) increases with f (it *decreased* up to 0.14.1)."""
    ratios = []
    for freq in (50.0, 250.0, 550.0, 1000.0):
        z = pollaczek_self_impedance(rho, rho, 10.0, freq, depth=1.0, gmr=GMR)
        ratios.append(z.real / (2.0 * math.pi * freq * MU_0 / 8.0))
    assert all(r > 1.0 for r in ratios), ratios
    assert all(b > a for a, b in zip(ratios, ratios[1:])), ratios


def test_self_pinned_values_1khz():
    """Pinned R' (mΩ/m), rho = 30 Ω·m, 1 kHz, GMR = 150 mm² Al PEN.

    0.14.1 gave 0.9739 / 0.9685 / 0.9512 / 0.9348 (falling with depth,
    all below omega*mu0/8 = 0.98696); 0.15.0 gives the values below.
    """
    got = [
        pollaczek_self_impedance(30.0, 30.0, 10.0, 1000.0, depth=d, gmr=GMR).real * 1e3
        for d in (0.7, 1.0, 2.0, 3.0)
    ]
    expected = [0.9991, 1.0036, 1.0166, 1.0267]
    assert got == pytest.approx(expected, abs=5e-4), got


@pytest.mark.parametrize("freq", [50.0, 1000.0])
@pytest.mark.parametrize("x", [0.5, 4.0, 10.0])
def test_mutual_uses_buried_kernel(freq, x):
    """The mutual coupling matches the exp(-(h_i+h_j) u) K0-form reference
    to 0.1 % in both R' and X' (measured worst case 0.048 % in R').

    With the 0.14.1 air kernel R' was 0.47 % low at 50 Hz and 2.0 % low at
    1 kHz — and the existing ``test_mutual_matches_k0_reference_*`` only
    checked the imaginary part, so it did not catch it.
    """
    z = pollaczek_mutual_impedance(
        100.0, 100.0, 10.0, freq, depth_i=0.8, depth_j=1.2, separation=x
    )
    z_ref = _reference_mutual(100.0, freq, 0.8, 1.2, x)
    assert z.real == pytest.approx(z_ref.real, rel=1e-3)
    assert z.imag == pytest.approx(z_ref.imag, rel=1e-3)


@pytest.mark.parametrize("freq", [250.0, 1000.0])
def test_mutual_resistance_rises_with_burial_depth(freq):
    """Like the self, the mutual earth-return resistance grows with the
    burial depth of the coupled pair (it fell with depth up to 0.14.1)."""
    r = [
        pollaczek_mutual_impedance(
            100.0, 100.0, 10.0, freq, depth_i=d, depth_j=d, separation=4.0
        ).real
        for d in (0.7, 1.0, 2.0, 3.0)
    ]
    assert all(b > a for a, b in zip(r, r[1:])), r


def test_self_reactance_still_matches_carson_after_kernel_fix():
    """Guard: fixing R' must not move X' away from Carson (X' shifts by
    only ~0.1 % at 1 kHz), so the reactance benchmark still holds."""
    for freq in (50.0, 250.0, 1000.0):
        z = pollaczek_self_impedance(100.0, 100.0, 10.0, freq, depth=0.7, gmr=GMR)
        z_c = carson_self_impedance(100.0, freq, gmr=GMR)
        assert z.imag == pytest.approx(z_c.imag, rel=5e-3)


# =====================================================================
# F37 — Dwight horizontal strip: constant -1, not -2
# =====================================================================
@pytest.mark.parametrize("width", [0.03, 0.10, 0.30])
@pytest.mark.parametrize("depth", [0.5, 0.8, 1.5])
def test_strip_reduces_to_equivalent_round_wire(width, depth):
    """A vanishingly thin strip of width ``a`` is the round wire of
    self-GMD ``a*exp(-3/2)``.

    This is the constant's fingerprint: with Dwight's strip constant -1
    the two agree to the O(b/a) residual of the cross-section term; with
    the round-wire constant -2 used up to 0.14.1 the strip came out
    10-13 % low.
    """
    b = 1e-9
    r_strip = horizontal_strip(100.0, 10.0, width, b, depth)
    r_wire = horizontal_wire(100.0, 10.0, width * math.exp(-1.5), depth)
    assert r_strip == pytest.approx(r_wire, rel=1e-6)


def test_strip_pinned_value():
    """Pinned: rho = 100 Ω·m, L = 10 m, a = 30 mm, b = 1 µm, t = 0.8 m.
    0.14.1 returned 7.1561 Ω (-10.0 %); the correct value is 7.9519 Ω."""
    r = horizontal_strip(100.0, 10.0, 0.03, 1e-6, 0.8)
    assert r == pytest.approx(7.95188, rel=1e-5)
    assert r > 7.5  # explicitly excludes the -2 (round-wire) constant


def test_strip_sits_at_the_gmd_wire_not_at_the_radius_wire():
    """The strip must be bracketed by the two round wires of radius ``a``
    and radius ``a*e^-1.5``, and sit at the *GMD* end of that bracket.

    Round wire of radius a = 100 mm: 5.800 Ω; round wire of the strip's
    self-GMD a*e^-1.5: 6.994 Ω. 0.14.1 returned 6.198 Ω — inside the
    bracket, but on the wrong side; 0.15.0 returns 6.9938 Ω.
    """
    r_strip = horizontal_strip(100.0, 10.0, 0.10, 1e-6, 0.8)
    r_wire_a = horizontal_wire(100.0, 10.0, 0.10, 0.8)
    r_wire_gmd = horizontal_wire(100.0, 10.0, 0.10 * math.exp(-1.5), 0.8)
    assert r_wire_a < r_strip <= r_wire_gmd * (1.0 + 1e-6)
    assert abs(r_strip - r_wire_gmd) < abs(r_strip - r_wire_a)


# =====================================================================
# F38 — Dwight vertical round plate: image term rho/(4 pi s)
# =====================================================================
def _plate_image_terms(rho: float, radius: float, depth: float):
    """Image contributions of both plate orientations (self term removed)."""
    r_self = rho / (8.0 * radius)
    return (
        horizontal_round_plate(rho, radius, depth) - r_self,
        vertical_round_plate(rho, radius, depth) - r_self,
    )


@pytest.mark.parametrize("depth,rel", [(20.0, 1e-3), (100.0, 1e-4)])
def test_plate_orientations_agree_in_the_far_field(depth, rel):
    """Orientation is invisible in the far field: both plate formulas must
    converge on the same image term rho/(4 pi s), s = 2 t.

    Up to 0.14.1 the vertical plate's image term was exactly half the
    horizontal one at every depth (a factor 2 applied twice), so this
    ratio was 0.5 instead of 1.
    """
    rho, a = 100.0, 0.5
    img_h, img_v = _plate_image_terms(rho, a, depth)
    far_field = rho / (4.0 * math.pi * 2.0 * depth)
    assert img_v == pytest.approx(img_h, rel=rel)
    assert img_v == pytest.approx(far_field, rel=rel)


@pytest.mark.parametrize("depth", [2.0, 5.0, 20.0, 100.0])
def test_vertical_plate_image_term_is_rho_over_4pi_s(depth):
    """The leading image term equals rho/(4 pi s) with the Eq. (38)
    series correction (+7a^2/24s^2 + 99a^4/320s^4) on top — so it is
    slightly *above* rho/(4 pi s), never half of it."""
    rho, a = 100.0, 0.5
    s = 2.0 * depth
    _, img_v = _plate_image_terms(rho, a, depth)
    series = 1.0 + 7.0 * a**2 / (24.0 * s**2) + 99.0 * a**4 / (320.0 * s**4)
    assert img_v == pytest.approx(rho / (4.0 * math.pi * s) * series, rel=1e-12)
    assert img_v > rho / (4.0 * math.pi * s)


def test_vertical_plate_pinned_value():
    """Pinned: rho = 100 Ω·m, a = 0.5 m, t = 2 m. 0.14.1 returned
    25.9993 Ω (-3.7 %); the correct value is 26.9987 Ω."""
    assert vertical_round_plate(100.0, 0.5, 2.0) == pytest.approx(26.99865, rel=1e-6)


def test_vertical_plate_slightly_above_horizontal_plate():
    """Same self term rho/(8a); the vertical plate's Eq. (38) series
    correction is positive where the horizontal Eq. (36) one is negative,
    so R_vertical > R_horizontal at equal depth (a small effect, not the
    factor ~2 the halved image term produced)."""
    for t in (2.0, 5.0, 20.0):
        r_v = vertical_round_plate(100.0, 0.5, t)
        r_h = horizontal_round_plate(100.0, 0.5, t)
        assert r_v > r_h
        assert r_v / r_h < 1.01


# =====================================================================
# F55 — earth_return docstring: X' increases with frequency
# =====================================================================
def test_earth_return_reactance_increases_with_frequency():
    """X' = (omega mu0/2pi) ln(D_e/GMR) rises almost linearly with f even
    though D_e ~ sqrt(rho/f) shrinks: 0.814 -> 14.39 mΩ/m from 50 Hz to
    1 kHz at 1000 Ω·m (17.7x for a 20x frequency step)."""
    xs = [
        carson_self_impedance(1000.0, f, gmr=0.007).imag
        for f in (50.0, 250.0, 1000.0)
    ]
    assert all(b > a for a, b in zip(xs, xs[1:])), xs
    assert xs[0] * 1e3 == pytest.approx(0.8137, abs=1e-4)
    assert xs[2] * 1e3 == pytest.approx(14.3915, abs=1e-4)
    assert xs[2] / xs[0] == pytest.approx(17.7, abs=0.1)


def test_earth_return_docstring_states_the_correct_frequency_trend():
    """The module docstring is the stated acceptance criterion for the
    sommerfeld earth-return stack, so its physics claim is pinned here.
    Up to 0.14.1 it said the reactance *decreases* with frequency."""
    text = re.sub(r"\s+", " ", er.__doc__ or "")
    assert "*decreases* with frequency" not in text
    assert "*increases* with frequency" in text
    # D_e itself is the quantity that shrinks with frequency.
    assert "*shrinks* with frequency" in text
