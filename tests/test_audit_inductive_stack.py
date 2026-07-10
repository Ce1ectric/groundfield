"""Regression tests for the WP-C inductive-stack fixes (audit 2026-07-08).

Covers the three structural defects of the earth-return stack:

* **F7 — Pollaczek kernel.** The historic ``sommerfeld`` kernel
  (``1/R + ∫Γ e^{-λ(z+z')}J0``) dropped the in-soil attenuation
  ``e^{-γR}`` of the primary term — which carries the buried-wire
  earth-return *resistance* — and produced ``Re Z' ≈ -ωμ0/8``
  (Carson's value with the wrong sign). The new kernel reproduces
  Pollaczek's analytic mutual impedance (K0 form) including the
  positive earth-return resistance.
* **F5 — mirror semantics.** Buried pair at σ→∞ is *screened*
  (total coupling → 0); overhead pair at σ→∞ recovers the
  **anti-parallel** PEC image (Carson's baseline) — the historic
  additive electrostatic mirror is gone from the σ-limits.
* **F6 — Carson matrix tiling.** ``build_carson_correction_matrix``
  no longer accumulates pseudo-mutual terms between collinear
  same-wire segments; the assembled correction is invariant under
  mesh refinement (measured drift before the fix: the branch
  current changed by a factor ~2.5 between dsl = 50 m and 6.25 m).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

import groundfield as gf
from groundfield.coupling.inductance import (
    build_carson_correction_matrix,
    build_inductance_matrix,
)
from groundfield.coupling.sommerfeld_inductance import (
    MU_0,
    sommerfeld_pair_integral_homogeneous,
)

# ---------------------------------------------------------------------
# F7 — Pollaczek per-unit-length mutual (buried pair)
# ---------------------------------------------------------------------


def _pollaczek_mutual_per_m(omega, sigma, x, h1, h2):
    """Analytic Pollaczek mutual impedance per metre, infinite wires."""
    g2 = 1j * omega * MU_0 * sigma
    g = np.sqrt(g2)
    d = math.hypot(x, h1 - h2)
    D = math.hypot(x, h1 + h2)

    def _f(lam, part):
        u = np.sqrt(lam * lam + g2)
        val = np.exp(-(h1 + h2) * u) / (lam + u)
        return val.real if part == "re" else val.imag

    J = (
        2.0 * quad(lambda t: _f(t, "re") * math.cos(t * x), 0, np.inf,
                   limit=400)[0]
        + 2j * quad(lambda t: _f(t, "im") * math.cos(t * x), 0, np.inf,
                    limit=400)[0]
    )
    return 1j * omega * MU_0 / (2 * np.pi) * (kv(0, g * d) - kv(0, g * D) + J)


def test_buried_pair_reproduces_pollaczek_mutual() -> None:
    """1-m probe vs a segment chain: per-m mutual matches the K0 form.

    f = 1 kHz, rho = 30 Ωm (δ ≈ 87 m) so a ±300 m chain captures the
    earth-return closure to within a few per cent; the imaginary part
    converges much faster. The historic kernel failed this test with
    the *opposite sign* of the real part.
    """
    omega = 2 * np.pi * 1000.0
    sigma = 1.0 / 30.0
    x, h = 10.0, 0.7
    seg = 25.0
    edges = np.arange(-300.0, 300.0 + seg, seg)
    a = np.array([[-0.5, 0.0, h], [0.5, 0.0, h]])
    dz = sum(
        sommerfeld_pair_integral_homogeneous(
            a[0], a[1],
            np.array([edges[k], x, h]), np.array([edges[k + 1], x, h]),
            omega=omega, sigma_earth=sigma,
        )
        for k in range(len(edges) - 1)
    )
    segs = [
        np.array([[edges[k], x, h], [edges[k + 1], x, h]])
        for k in range(len(edges) - 1)
    ]
    eps = np.stack([a] + segs)
    L = build_inductance_matrix(eps, np.full(len(eps), 0.005),
                                use_image=True)
    z_pkg = 1j * omega * L[0, 1:].sum() + dz

    z_ref = _pollaczek_mutual_per_m(omega, sigma, x, h, h)
    # Real part: chain truncation at ~3.4 δ leaves a few per cent.
    assert z_pkg.real == pytest.approx(z_ref.real, rel=0.12)
    assert z_pkg.real > 0.0  # the historic kernel gave < 0
    # Imaginary part converges fast.
    assert z_pkg.imag == pytest.approx(z_ref.imag, rel=0.02)


# ---------------------------------------------------------------------
# F5 — σ-limits with the correct image semantics
# ---------------------------------------------------------------------


def _parallel_pair(z: float):
    """Two parallel 10-m segments at 5 m separation, both at height z."""
    a = np.array([[0.0, 0.0, z], [10.0, 0.0, z]])
    b = np.array([[0.0, 5.0, z], [10.0, 5.0, z]])
    return a, b


def test_sigma_infinite_overhead_recovers_pec_anti_image() -> None:
    """Overhead pair, σ→∞: the correction must flip the additive
    mirror into the anti-parallel PEC image, i.e.
    ΔZ → −2·jω·M_image."""
    omega = 2 * np.pi * 50.0
    a, b = _parallel_pair(z=-10.0)
    dz = sommerfeld_pair_integral_homogeneous(
        a[0], a[1], b[0], b[1], omega=omega, sigma_earth=1e9,
    )
    eps = np.stack([a, b])
    radii = np.array([0.005, 0.005])
    M_with = build_inductance_matrix(eps, radii, use_image=True)[0, 1]
    M_free = build_inductance_matrix(eps, radii, use_image=False)[0, 1]
    M_img = M_with - M_free
    assert dz.imag == pytest.approx(-2.0 * omega * M_img, rel=2e-2)


def test_sigma_zero_cancels_the_additive_mirror() -> None:
    """σ→0 (magnetically transparent earth): the correction cancels
    the ADR-0004 image exactly — total coupling = free space."""
    omega = 2 * np.pi * 50.0
    for z in (+0.7, -10.0):  # buried and overhead
        a, b = _parallel_pair(z=z)
        dz = sommerfeld_pair_integral_homogeneous(
            a[0], a[1], b[0], b[1], omega=omega, sigma_earth=1e-12,
        )
        eps = np.stack([a, b])
        radii = np.array([0.005, 0.005])
        M_with = build_inductance_matrix(eps, radii, use_image=True)[0, 1]
        M_free = build_inductance_matrix(eps, radii, use_image=False)[0, 1]
        M_img = M_with - M_free
        assert dz.imag == pytest.approx(-omega * M_img, rel=2e-2), z


# ---------------------------------------------------------------------
# F6 — Carson matrix: mesh convergence
# ---------------------------------------------------------------------


def _carson_total_straight_wire(n_seg: int, omega: float) -> complex:
    """Assembled Carson correction of one straight 100-m buried wire,
    all segments in series (uniform current) → scalar total."""
    edges = np.linspace(0.0, 100.0, n_seg + 1)
    eps = np.array(
        [[[edges[k], 0.0, 0.7], [edges[k + 1], 0.0, 0.7]]
         for k in range(n_seg)]
    )
    dz = build_carson_correction_matrix(
        eps, np.full(n_seg, 0.005), omega=omega, sigma_earth=0.01,
    )
    return complex(dz.sum())


def test_carson_matrix_mesh_convergent_straight_wire() -> None:
    """Total Carson correction of a refined straight wire is
    refinement-invariant (historic behaviour: grew ~linearly with
    the segment count)."""
    omega = 2 * np.pi * 1000.0
    totals = [_carson_total_straight_wire(n, omega) for n in (2, 4, 8, 16)]
    ref = totals[0]
    for t in totals[1:]:
        assert abs(t - ref) / abs(ref) < 1e-9


def test_carson_matrix_parallel_wires_tiling() -> None:
    """Cross-wire mutual block sums to z'_mutual · ℓ independent of
    the segmentation of either wire."""
    omega = 2 * np.pi * 1000.0
    sigma = 0.01

    def cross_sum(n_a: int, n_b: int) -> complex:
        ea = np.linspace(0.0, 100.0, n_a + 1)
        eb = np.linspace(0.0, 100.0, n_b + 1)
        segs = (
            [[[ea[k], 0.0, 0.7], [ea[k + 1], 0.0, 0.7]]
             for k in range(n_a)]
            + [[[eb[k], 8.0, 0.7], [eb[k + 1], 8.0, 0.7]]
               for k in range(n_b)]
        )
        eps = np.array(segs)
        dz = build_carson_correction_matrix(
            eps, np.full(n_a + n_b, 0.005), omega=omega, sigma_earth=sigma,
        )
        return complex(dz[:n_a, n_a:].sum())

    ref = cross_sum(2, 2)
    for n_a, n_b in ((4, 4), (8, 8), (4, 8)):
        assert abs(cross_sum(n_a, n_b) - ref) / abs(ref) < 1e-9


def test_carson_solver_level_mesh_convergence() -> None:
    """End-to-end: the branch current with carson_series active is
    invariant under PEN refinement (the audit measured a drift by a
    factor ~2.5 between dsl = 50 m and 6.25 m)."""
    def run(dsl: float) -> complex:
        w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
        g1 = gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.7),
                                 length=3.0, wire_radius=0.005)
        g2 = gf.create_electrode(w, "rod", name="g2",
                                 position=(100.0, 0, 0.7),
                                 length=3.0, wire_radius=0.005)
        gf.create_conductor(w, name="pen", start=g1, end=g2,
                            cross_section=50e-6,
                            discretize_segment_length=dsl,
                            inductance_model="neumann")
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        eng = gf.create_engine(backend="image", segment_length=0.5,
                               frequencies=[1000.0],
                               earth_inductive_model="carson_series")
        return eng.solve(w).electrode_currents["g2"][0]

    i_coarse = run(50.0)
    i_fine = run(6.25)
    assert abs(i_fine - i_coarse) / abs(i_coarse) < 1e-3
