r"""Carson earth-return line-impedance benchmark.

Validates groundfield's assembled Sommerfeld/Pollaczek earth-return stack
(``build_inductance_matrix`` + ``build_sommerfeld_correction_matrix``,
i.e. ``earth_inductive_model="sommerfeld"``) against Carson's closed-form
per-unit-length line impedance
(:mod:`groundfield.references.earth_return`).

Two independent quantities, with very different convergence behaviour:

* **Reactance** :math:`X'` — the earth-return reactance
  :math:`(\omega\mu_0/2\pi)\ln(D_e/\mathrm{GMR})` with Carson's
  equivalent depth :math:`D_e`. It converges quickly with the modelled
  line length and the assembled value matches Carson to about 1 %. This
  is the headline result: groundfield reproduces the frequency- and
  soil-dependent earth-return **depth** :math:`D_e`, not just the naive
  geometric image at :math:`2h`.
* **Resistance** :math:`R' = \omega\mu_0/8` — soil-independent. The
  *kernel* is correct (pinned at the per-metre level by
  ``tests/test_audit_inductive_stack.py::
  test_buried_pair_reproduces_pollaczek_mutual``), but the earth-return
  current closes over roughly one skin depth
  (:math:`\delta = 503\sqrt{\rho/f}`), so a finite modelled line
  **under-counts** :math:`R'` and approaches :math:`\omega\mu_0/8` only
  from below as the line lengthens. It is checked here as a bounded,
  monotone convergence rather than a tight match.

To keep the O(M²) Sommerfeld assembly affordable the operating points use
a low soil resistivity / high frequency, so :math:`D_e` and
:math:`\delta` are small and a short modelled line already converges.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from groundfield.coupling.inductance import build_inductance_matrix
from groundfield.coupling.sommerfeld_inductance import (
    LayeredEarth,
    build_sommerfeld_correction_matrix,
)
from groundfield.references import earth_return as er


# ---------------------------------------------------------------------
# 1. The reference formula itself.
# ---------------------------------------------------------------------
def test_earth_return_resistance_is_soil_independent() -> None:
    # R' = ω μ0 / 8 depends only on frequency (~0.0493 Ω/km at 50 Hz).
    r50 = er.carson_earth_return_resistance(50.0)
    assert r50 == pytest.approx(math.pi**2 * 50.0 * 1e-7, rel=1e-12)  # π²f·1e-7 Ω/m
    # self impedance real part equals it, for any soil.
    for rho in (10.0, 100.0, 1000.0):
        z = er.carson_self_impedance(rho, 50.0, gmr=0.01)
        assert z.real == pytest.approx(r50, rel=1e-12)


def test_equivalent_depth_scaling() -> None:
    assert er.carson_equivalent_depth(100.0, 50.0) == pytest.approx(
        658.87 * math.sqrt(100.0 / 50.0)
    )
    # De grows with rho, shrinks with f.
    assert er.carson_equivalent_depth(1000.0, 50.0) > er.carson_equivalent_depth(
        100.0, 50.0
    )
    assert er.carson_equivalent_depth(100.0, 1000.0) < er.carson_equivalent_depth(
        100.0, 50.0
    )


def test_self_reactance_increases_with_soil_resistivity() -> None:
    # Deeper current spread in more resistive soil -> larger reactance.
    lo = er.carson_self_impedance(30.0, 50.0, gmr=0.01).imag
    hi = er.carson_self_impedance(1000.0, 50.0, gmr=0.01).imag
    assert hi > lo > 0.0


def test_mutual_below_self_reactance() -> None:
    # Mutual (separation 10 m) reactance < self (gmr 0.01 m): ln(De/d) smaller.
    self_x = er.carson_self_impedance(100.0, 500.0, gmr=0.01).imag
    mutual_x = er.carson_mutual_impedance(100.0, 500.0, separation=10.0).imag
    assert 0.0 < mutual_x < self_x


def test_reference_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        er.carson_self_impedance(-1.0, 50.0, gmr=0.01)
    with pytest.raises(ValueError):
        er.carson_mutual_impedance(100.0, 50.0, separation=0.0)


# ---------------------------------------------------------------------
# 2. Assembled groundfield line impedance vs Carson.
# ---------------------------------------------------------------------
def _assemble_self_per_metre(
    rho: float,
    frequency: float,
    *,
    height: float,
    radius: float,
    half_length: float,
    n_seg: int,
) -> complex:
    """Series self impedance per metre of a straight buried conductor.

    A length ``2·half_length`` wire at depth ``height`` is split into
    ``n_seg`` segments carrying a common (series) current; the total
    series impedance ``jω·ΣL + Σ dZ_sommerfeld`` divided by the length is
    the per-metre value that Carson's formula predicts for an infinite
    line.
    """
    edges = np.linspace(-half_length, half_length, n_seg + 1)
    eps = np.array(
        [[[edges[k], 0.0, height], [edges[k + 1], 0.0, height]] for k in range(n_seg)]
    )
    radii = np.full(n_seg, radius)
    omega = 2.0 * math.pi * frequency
    L = build_inductance_matrix(eps, radii, use_image=True)
    earth = LayeredEarth(rhos=(rho,), thicknesses=())
    dz = build_sommerfeld_correction_matrix(eps, radii, omega=omega, earth=earth)
    return (1j * omega * L.sum() + dz.sum()) / (2.0 * half_length)


@pytest.mark.parametrize("rho,frequency", [(10.0, 1000.0), (30.0, 1000.0)])
def test_assembled_self_reactance_matches_carson(rho, frequency) -> None:
    """The assembled earth-return **reactance** reproduces Carson's
    equivalent-depth value to ~1 % — i.e. groundfield recovers the
    frequency/soil-dependent earth-return depth D_e."""
    radius, height = 0.01, 1.0
    z = _assemble_self_per_metre(
        rho,
        frequency,
        height=height,
        radius=radius,
        half_length=180.0,
        n_seg=22,
    )
    ref = er.carson_self_impedance(rho, frequency, gmr=radius)
    assert z.imag == pytest.approx(ref.imag, rel=0.03), f"X' {z.imag} vs {ref.imag}"


def test_assembled_self_resistance_converges_from_below() -> None:
    """The earth-return **resistance** is positive, bounded above by
    Carson's ω μ0 / 8, and grows toward it as the modelled line lengthens
    (the finite-line truncation the docstring describes)."""
    rho, frequency, radius, height = 10.0, 1000.0, 0.01, 1.0
    r_carson = er.carson_earth_return_resistance(frequency)

    z_short = _assemble_self_per_metre(
        rho, frequency, height=height, radius=radius, half_length=100.0, n_seg=16
    )
    z_long = _assemble_self_per_metre(
        rho, frequency, height=height, radius=radius, half_length=180.0, n_seg=22
    )

    # Positive (the historic kernel had the wrong sign) and below Carson.
    assert 0.0 < z_short.real < r_carson
    assert 0.0 < z_long.real < r_carson
    # Monotone convergence toward the analytic value.
    assert z_long.real > z_short.real
    # A ±180 m line at 1 kHz / 10 Ω·m (δ ≈ 50 m) recovers most of it.
    assert z_long.real / r_carson > 0.75


# ---------------------------------------------------------------------
# 3. Two-layer (and by extension multi-layer) earth-return impedance.
# ---------------------------------------------------------------------
def _assemble_earth(
    earth: "LayeredEarth",
    frequency: float,
    *,
    height: float = 1.0,
    radius: float = 0.01,
    half_length: float = 120.0,
    n_seg: int = 16,
) -> complex:
    """Per-metre self impedance for an arbitrary ``LayeredEarth``.

    The two-layer checks below compare results assembled with *this same*
    configuration, so the finite-line truncation cancels and a short line
    keeps the O(n_seg²) Sommerfeld cost low.
    """
    edges = np.linspace(-half_length, half_length, n_seg + 1)
    eps = np.array(
        [[[edges[k], 0.0, height], [edges[k + 1], 0.0, height]] for k in range(n_seg)]
    )
    radii = np.full(n_seg, radius)
    omega = 2.0 * math.pi * frequency
    L = build_inductance_matrix(eps, radii, use_image=True)
    dz = build_sommerfeld_correction_matrix(eps, radii, omega=omega, earth=earth)
    return (1j * omega * L.sum() + dz.sum()) / (2.0 * half_length)


def test_two_layer_reduces_to_homogeneous() -> None:
    """A two-layer earth with ρ1 == ρ2 reproduces the single-layer
    assembly (the layered kernel's homogeneous limit)."""
    rho, f = 100.0, 1000.0
    z_hom = _assemble_earth(LayeredEarth(rhos=(rho,), thicknesses=()), f)
    z_two = _assemble_earth(LayeredEarth(rhos=(rho, rho), thicknesses=(5.0,)), f)
    assert z_two == pytest.approx(z_hom, rel=1e-9)


def test_two_layer_reactance_brackets_and_transitions() -> None:
    """A two-layer earth (ρ1 over ρ2) has an earth-return reactance
    between the two homogeneous bounds, moving from the deep-layer (ρ2)
    bound for a thin top layer toward the upper-layer (ρ1) bound for a
    thick one."""
    rho1, rho2, f = 100.0, 20.0, 1000.0
    x1 = _assemble_earth(LayeredEarth(rhos=(rho1,), thicknesses=()), f).imag
    x2 = _assemble_earth(LayeredEarth(rhos=(rho2,), thicknesses=()), f).imag
    assert x2 < x1  # more resistive soil -> deeper spread -> larger reactance

    x_thin = _assemble_earth(
        LayeredEarth(rhos=(rho1, rho2), thicknesses=(2.0,)), f
    ).imag
    x_thick = _assemble_earth(
        LayeredEarth(rhos=(rho1, rho2), thicknesses=(320.0,)), f
    ).imag

    tol = 1e-3 * x1  # allow a per-mille numerical overshoot at the bounds
    # Both lie within the homogeneous band...
    assert x2 - tol <= x_thin <= x1 + tol
    assert x2 - tol <= x_thick <= x1 + tol
    # ...the thin top layer sits nearer the deep-layer bound, the thick
    # one nearer the upper-layer bound.
    assert x_thin < x_thick
    assert abs(x_thin - x2) < abs(x_thin - x1)
    assert abs(x_thick - x1) < abs(x_thick - x2)


# ---------------------------------------------------------------------
# 4. Quantitative two-layer validation vs Tsiamitros 2005 Eq.(8).
# ---------------------------------------------------------------------
# Tsiamitros, Papagiannis, Labridis, Dokopoulos, "Earth Return Path
# Impedances of Underground Cables for the Two-Layer Earth Case", IEEE
# Trans. PWRD 20(3), 2005 — Eq.(8), self impedance (validated vs FEM
# <0.9 % in the paper). Independent quadrature implementation used as the
# reference for groundfield's two-layer inductive earth-return.


def _eq8_two_layer_self(
    frequency: float, rho1: float, rho2: float, d: float, h: float, gmr: float
) -> complex:
    """Tsiamitros 2005 Eq.(8) self impedance per metre (Ω/m), two-layer,
    quasi-static (σ-only, matching groundfield)."""
    from scipy.integrate import quad

    mu0 = 4.0e-7 * math.pi
    omega = 2.0 * math.pi * frequency
    g1 = 1j * omega * mu0 / rho1
    g2 = 1j * omega * mu0 / rho2

    def g(u):
        a0 = u
        a1 = np.sqrt(u * u + g1)
        a2 = np.sqrt(u * u + g2)
        s10, s21, d21, d10 = a1 + a0, a1 + a2, a1 - a2, a1 - a0
        num = (
            s10 * s21
            + s10 * d21 * np.exp(-a1 * 2 * (d - h))
            + d10 * s21 * np.exp(-a1 * 2 * h)
            + d10 * d21 * np.exp(-a1 * 2 * d)
        )
        return num / (s10 * s21 - d10 * d21 * np.exp(-2 * a1 * d)) / a1

    re = quad(lambda u: g(u).real, 0, np.inf, weight="cos", wvar=gmr, limit=800)[0]
    im = quad(lambda u: g(u).imag, 0, np.inf, weight="cos", wvar=gmr, limit=800)[0]
    return 1j * omega * mu0 / (2 * math.pi) * (re + 1j * im)


# Paper §VI.A geometry + Table I, CASE V.
_CV = dict(rho1=160.776, rho2=34.074, d=1.848, height=1.2, radius=0.0484)


def test_reflection_layered_reduces_to_halfspace_limits() -> None:
    """The layered surface reflection coefficient must reduce to the deep
    layer's half-space value as the top layer vanishes (h1 → 0) and to the
    upper layer's as it thickens (h1 → ∞). The pre-0.13 form failed the
    thin-layer limit (it under-weighted the deep layer)."""
    from groundfield.coupling.sommerfeld_inductance import (
        reflection_coefficient_homogeneous,
        reflection_coefficient_layered,
    )

    lambdas = np.array([0.01, 0.1, 0.5, 2.0])
    omega = 2.0 * math.pi * 1.0e4
    rho1, rho2 = 300.0, 50.0
    thin = reflection_coefficient_layered(
        lambdas, omega=omega, earth=LayeredEarth(rhos=(rho1, rho2), thicknesses=(1e-4,))
    )
    thick = reflection_coefficient_layered(
        lambdas, omega=omega, earth=LayeredEarth(rhos=(rho1, rho2), thicknesses=(1e5,))
    )
    ref2 = reflection_coefficient_homogeneous(
        lambdas, omega=omega, sigma_earth=1 / rho2
    )
    ref1 = reflection_coefficient_homogeneous(
        lambdas, omega=omega, sigma_earth=1 / rho1
    )
    assert np.max(np.abs(thin - ref2)) < 1e-4
    assert np.max(np.abs(thick - ref1)) < 1e-9


def test_two_layer_self_reactance_matches_eq8() -> None:
    """Assembled two-layer earth-return reactance matches the Tsiamitros
    2005 Eq.(8) reference to a few percent (the same accuracy the
    homogeneous assembly reaches — the earlier layered kernel was ~8 %
    off with the wrong layer weighting)."""
    f = 1.0e4
    z = _assemble_earth(
        LayeredEarth(rhos=(_CV["rho1"], _CV["rho2"]), thicknesses=(_CV["d"],)),
        f,
        height=_CV["height"],
        radius=_CV["radius"],
        half_length=250.0,
        n_seg=30,
    )
    ref = _eq8_two_layer_self(
        f, _CV["rho1"], _CV["rho2"], _CV["d"], _CV["height"], _CV["radius"]
    )
    assert z.imag == pytest.approx(ref.imag, rel=0.05), f"X' {z.imag} vs {ref.imag}"


@pytest.mark.parametrize("frequency", [1.0e2, 1.0e3, 1.0e5, 1.0e6])
def test_two_layer_attribution_matches_eq8_across_frequency(frequency) -> None:
    """The layer *attribution* — how far the two-layer reactance sits
    between the ρ1 and ρ2 bounds — matches Eq.(8) across the full band the
    dissertation cares about, from the low-frequency regime (≤ 1 kHz,
    deep-layer-dominated) up to 1 MHz. Computed as a fraction of each
    method's own bounds, so the finite-line truncation cancels."""
    r1, r2, d = _CV["rho1"], _CV["rho2"], _CV["d"]
    kw = dict(height=_CV["height"], radius=_CV["radius"], half_length=120.0, n_seg=16)
    g1 = _assemble_earth(LayeredEarth(rhos=(r1,), thicknesses=()), frequency, **kw).imag
    g2 = _assemble_earth(LayeredEarth(rhos=(r2,), thicknesses=()), frequency, **kw).imag
    gt = _assemble_earth(
        LayeredEarth(rhos=(r1, r2), thicknesses=(d,)), frequency, **kw
    ).imag
    gf_frac = (gt - g2) / (g1 - g2)

    e1 = _eq8_two_layer_self(frequency, r1, r1, d, _CV["height"], _CV["radius"]).imag
    e2 = _eq8_two_layer_self(frequency, r2, r2, d, _CV["height"], _CV["radius"]).imag
    et = _eq8_two_layer_self(frequency, r1, r2, d, _CV["height"], _CV["radius"]).imag
    eq8_frac = (et - e2) / (e1 - e2)

    assert gf_frac == pytest.approx(
        eq8_frac, abs=0.06
    ), f"f={frequency}: gf frac {gf_frac:.3f} vs Eq.(8) {eq8_frac:.3f}"


def test_two_layer_frequency_regimes_low_vs_high() -> None:
    """Dissertation-relevant frequency behaviour, stated as a pinned claim.

    What happens up to 1 kHz vs up to 1 MHz? The inductive earth-return
    penetration depth δ₂ = 503·√(ρ₂/f) shrinks as √f. For the Case V soil
    (interface d = 1.848 m) δ₂ runs from ~415 m at 50 Hz down to ~3 m at
    1 MHz, so:

    * **≤ 1 kHz — deep-layer-dominated.** δ₂ ≫ d, the return "sees through"
      the thin resistive topsoil, and the two-layer reactance sits within a
      few percent of the *deep-layer* (ρ₂) bound: attribution ``frac`` < 0.05.
    * **up to 1 MHz — transition.** As δ₂ falls toward and below d the top
      layer takes over: ``frac`` rises monotonically and, by 1 MHz, the top
      layer contributes substantially (``frac`` > 0.4).

    This is the low-frequency behaviour the dissertation relies on, so it is
    pinned directly rather than left to the notebook.
    """
    r1, r2, d = _CV["rho1"], _CV["rho2"], _CV["d"]
    kw = dict(height=_CV["height"], radius=_CV["radius"], half_length=120.0, n_seg=16)
    freqs = [50.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
    fracs = []
    for f in freqs:
        g1 = _assemble_earth(LayeredEarth(rhos=(r1,), thicknesses=()), f, **kw).imag
        g2 = _assemble_earth(LayeredEarth(rhos=(r2,), thicknesses=()), f, **kw).imag
        gt = _assemble_earth(
            LayeredEarth(rhos=(r1, r2), thicknesses=(d,)), f, **kw
        ).imag
        fracs.append((gt - g2) / (g1 - g2))

    frac_by_f = dict(zip(freqs, fracs))
    # (1) up to 1 kHz: within a few percent of the deep-layer (ρ2) bound.
    assert frac_by_f[50.0] < 0.05, f"50 Hz frac {frac_by_f[50.0]:.3f}"
    assert frac_by_f[1.0e3] < 0.05, f"1 kHz frac {frac_by_f[1.0e3]:.3f}"
    # (2) monotone rise toward the top layer across 50 Hz .. 1 MHz.
    assert all(a < b for a, b in zip(fracs, fracs[1:])), fracs
    # (3) by 1 MHz the top layer contributes substantially (δ2 ~ interface).
    assert frac_by_f[1.0e6] > 0.4, f"1 MHz frac {frac_by_f[1.0e6]:.3f}"
