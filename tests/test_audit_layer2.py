"""Regression tests for the WP-D layer-2 evaluation fixes (audit 2026-07-08).

Covers:

* **D1 — layer dispatch in the post-solve potential paths.**
  ``FieldResult.potential`` (and the ``solve_mutual_*`` probe
  kernels) historically applied the upper-layer Tagg/Sunde image
  series to *every* probe and source point; for layer-2 points the
  measured error was +7 % at z = 7 m and +25 % at z = 12 m
  (K = +0.818, h_1 = 5 m). Pairs involving layer 2 now go through
  the rigorous spectral kernel; pure upper-layer evaluation is
  bit-exact to the historic path.
* **D2 — interface-split discretiser (ADR-0007 step 1).** Rod (and
  sloped-strip) segments no longer straddle the layer interface;
  a margin warning fires when the deepest segment ends within
  2·segment_length of h_1.
* **D3 — mom accepts cross-layer worlds.** The engine that *solves*
  the per-segment current distribution no longer rejects exactly
  the geometry where the image family's uniform-leakage assumption
  is weakest.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf

SOIL = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=5.0)  # K = +0.818


def _rod_world(soil, *, z0=0.5, length=2.0, ds_engine=0.25):
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, z0),
                        length=length, wire_radius=0.005)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image_2layer", segment_length=ds_engine)
    return eng.solve(w)


# ---------------------------------------------------------------------
# D1 — potential evaluation below the interface
# ---------------------------------------------------------------------


def test_potential_continuous_across_interface() -> None:
    """The potential must be continuous at z = h_1: the uu series
    (just above) and the spectral kernel (just below) have to meet.

    Before the fix the layer-2 side used the invalid uu series and
    the mismatch at depth was in the +7 %…+25 % range."""
    res = _rod_world(SOIL)
    eps = 0.02
    pts = np.array([
        [10.0, 0.0, SOIL.h_1 - eps],
        [10.0, 0.0, SOIL.h_1 + eps],
    ])
    phi = res.potential(pts).real
    assert phi[1] == pytest.approx(phi[0], rel=5e-3)


def test_layer2_probe_regression_value() -> None:
    """Pinned regression anchor from the audit verification
    (exp9): rod 0.5–2.5 m, K = +0.818, probe at (10, 0, 7) →
    φ = 5.604 V (was 5.977 V with the uu series, +6.7 %)."""
    res = _rod_world(SOIL)
    phi = float(res.potential(np.array([[10.0, 0.0, 7.0]])).real[0])
    assert phi == pytest.approx(5.6041, rel=1e-3)


def test_layer2_probe_homogeneous_limit() -> None:
    """rho_2 = rho_1: the spectral layer-2 path must reproduce the
    homogeneous evaluation."""
    soil_eq = gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=5.0)
    res = _rod_world(soil_eq)
    pts = np.array([[10.0, 0.0, 7.0], [10.0, 0.0, 12.0]])
    phi = res.potential(pts).real
    # Homogeneous reference: same world in HomogeneousSoil.
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5),
                        length=2.0, wire_radius=0.005)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res_h = gf.create_engine(backend="image", segment_length=0.25).solve(w)
    phi_h = res_h.potential(pts).real
    np.testing.assert_allclose(phi, phi_h, rtol=2e-3)


def test_upper_layer_evaluation_unchanged() -> None:
    """Pure upper-layer probes keep the historic fast path — surface
    potential from the audit verification: φ(10, 0, 0) = 6.084 V."""
    res = _rod_world(SOIL)
    phi = float(res.potential(np.array([[10.0, 0.0, 0.0]])).real[0])
    assert phi == pytest.approx(6.0842, rel=1e-3)


# ---------------------------------------------------------------------
# D2 — interface-split discretiser + margin warning
# ---------------------------------------------------------------------


def test_no_segment_straddles_interface() -> None:
    """A rod through h_1 produces segments entirely inside one layer."""
    res = _rod_world(SOIL, z0=0.5, length=9.0, ds_engine=0.5)
    h_1 = SOIL.h_1
    for ps in res.point_sources:
        z_mid = ps.position[2]
        half = 0.5 * ps.length
        assert (z_mid - half >= h_1 - 1e-9) or (z_mid + half <= h_1 + 1e-9)


def test_interface_margin_warning() -> None:
    """Deepest segment ending just above h_1 triggers the advisory."""
    w = gf.create_world(soil=SOIL)
    # Rod from 0.5 to 4.7 m — 0.3 m above h_1 = 5 with ds = 0.5.
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5),
                        length=4.2, wire_radius=0.005)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image_2layer", segment_length=0.5)
    with pytest.warns(UserWarning, match="interface"):
        eng.solve(w)


# ---------------------------------------------------------------------
# D3 — mom solves cross-layer worlds
# ---------------------------------------------------------------------


def _cross_layer_world():
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=5.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5),
                        length=7.0, wire_radius=0.005)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_mom_solves_cross_layer_and_matches_bem() -> None:
    """mom (Galerkin, solved current distribution) now accepts
    cross-layer worlds and must agree tightly with bem (collocation
    on the same kernel, also solved distribution)."""
    eng_mom = gf.create_engine(backend="mom", segment_length=1.0)
    z_mom = eng_mom.solve(_cross_layer_world()).cluster_impedance("g1")[0]
    eng_bem = gf.create_engine(backend="bem", segment_length=1.0)
    z_bem = eng_bem.solve(_cross_layer_world()).cluster_impedance("g1")[0]
    assert z_mom.real == pytest.approx(z_bem.real, rel=1e-6)


def test_mom_cross_layer_quantifies_uniform_leakage_bias() -> None:
    """The solved distribution (mom) must lie below the prescribed
    uniform-leakage value (image_2layer) for rho_2 > rho_1 — this is
    the audit's P5 finding, now measurable inside the package."""
    eng_ref = gf.create_engine(backend="image_2layer", segment_length=1.0)
    z_ref = eng_ref.solve(_cross_layer_world()).cluster_impedance("g1")[0]
    eng_mom = gf.create_engine(backend="mom", segment_length=1.0)
    z_mom = eng_mom.solve(_cross_layer_world()).cluster_impedance("g1")[0]
    assert 0.7 * z_ref.real < z_mom.real < z_ref.real
