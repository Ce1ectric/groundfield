"""Tests for the 2-layer backend (Engine A — Tagg/Sunde).

Plausibility sources:

- ADR-0001 (``docs/adr/0001-two-layer-method.md``) for the convergence
  expectation.
- Dwight 1936 for the homogeneous limit.
- Tagg 1964 / Sunde 1968 for the series form.

Test strategy
-------------
1. **Limit ρ₁ = ρ₂** must reproduce the homogeneous backend exactly
   (the ``n=0`` series term is mathematically identical).
2. **Small |K|** must keep ``compare_engines(image_homogeneous,
   image_2layer)`` consistent — sanity check of the cross-engine
   helper.
3. **Sign of K** must act in the right direction: a better lower
   layer (ρ₂ < ρ₁) lowers the grounding impedance; a worse lower
   layer raises it.
4. **Auto-dispatch**: ``backend="image"`` with ``TwoLayerSoil``
   transparently picks the 2-layer backend.
5. **Precondition check**: an electrode below the layer interface
   must raise a clear ``ValueError``.
6. **Series convergence** is documented in ``metadata``
   (``n_terms_used``, ``converged``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf


# ---------------------------------------------------------------------
# 1. Homogeneous limit
# ---------------------------------------------------------------------


def test_two_layer_reduces_to_homogeneous_when_rho_equal() -> None:
    """ρ₁ = ρ₂ ⇒ K = 0 ⇒ exactly the homogeneous result."""
    rho = 100.0
    homog = gf.HomogeneousSoil(resistivity=rho)
    twolay = gf.TwoLayerSoil(rho_1=rho, rho_2=rho, h_1=2.0)

    eng = gf.create_engine(backend="image", segment_length=0.05)
    z_homog: list[float] = []
    z_twolay: list[float] = []
    for soil, target in [(homog, z_homog), (twolay, z_twolay)]:
        world = gf.create_world(soil=soil)
        gf.create_electrode(world, "rod", name="g1",
                            position=(0, 0, 0.0), length=1.5)
        gf.create_source(world, attached_to="g1", magnitude=1.0)
        target.append(eng.solve(world).cluster_impedance("g1")[0].real)

    assert z_twolay[0] == pytest.approx(z_homog[0], abs=1e-9), (
        f"K=0 limit: 2-layer={z_twolay[0]:.6f} != homogeneous={z_homog[0]:.6f}"
    )


# ---------------------------------------------------------------------
# 2. Cross-engine comparison at small K
# ---------------------------------------------------------------------


def test_compare_engines_two_layer_vs_homogeneous_close_K() -> None:
    """For small |K| the 2-layer result must be close to the homogeneous
    solution (ρ_eff ≈ ρ₁)."""
    rho_1, rho_2 = 100.0, 110.0
    soil_h = gf.HomogeneousSoil(resistivity=rho_1)
    soil_2 = gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=2.0)

    def _build(soil):
        w = gf.create_world(soil=soil)
        gf.create_electrode(w, "rod", name="g1",
                            position=(0, 0, 0.0), length=1.5)
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    eng = gf.create_engine(backend="image", segment_length=0.05)
    Z_h = eng.solve(_build(soil_h)).cluster_impedance("g1")[0].real
    Z_2 = eng.solve(_build(soil_2)).cluster_impedance("g1")[0].real

    # |K| = 10/210 ≈ 0.048 ⇒ effect on the rod is small, < 5 %.
    assert abs(Z_2 - Z_h) / Z_h < 0.05, (
        f"Z_2layer={Z_2:.2f}, Z_homog={Z_h:.2f}"
    )


# ---------------------------------------------------------------------
# 3. Sign behaviour of K
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "rho_2, expected_direction",
    [
        (10.0, "lower"),    # better lower layer → lower impedance
        (1000.0, "higher"), # worse lower layer  → higher impedance
    ],
)
def test_two_layer_K_sign_changes_impedance_correctly(
    rho_2: float, expected_direction: str
) -> None:
    rho_1 = 100.0
    eng = gf.create_engine(backend="image", segment_length=0.05)

    def _Z(soil):
        w = gf.create_world(soil=soil)
        gf.create_electrode(w, "rod", name="g1",
                            position=(0, 0, 0.0), length=1.5)
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return eng.solve(w).cluster_impedance("g1")[0].real

    Z_homog = _Z(gf.HomogeneousSoil(resistivity=rho_1))
    Z_twolay = _Z(gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=2.0))

    if expected_direction == "lower":
        assert Z_twolay < Z_homog
    else:
        assert Z_twolay > Z_homog


# ---------------------------------------------------------------------
# 4. Engine dispatch
# ---------------------------------------------------------------------


def test_engine_auto_selects_image_2layer_for_two_layer_soil() -> None:
    """``backend='image'`` + ``TwoLayerSoil`` ⇒ backend == 'image_2layer'."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=200.0, h_1=2.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.0), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    result = gf.create_engine(backend="image", segment_length=0.05).solve(world)

    assert result.backend == "image_2layer"
    assert result.metadata["K"] == pytest.approx(
        soil.reflection_coefficient
    )
    assert result.metadata["converged"] is True


# ---------------------------------------------------------------------
# 5. Cross-layer electrodes are now supported (ADR-0007)
# ---------------------------------------------------------------------


def test_two_layer_accepts_electrode_below_layer_boundary() -> None:
    """ADR-0007: a rod that crosses h_1 used to raise; the
    cross-layer Sommerfeld dispatcher now lets it through and
    produces a finite, physically sensible cluster impedance.

    See also tests/test_cross_layer.py for the full validation
    programme of the cross-layer path."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=1.0)
    world = gf.create_world(soil=soil)
    # Rod head at z=0.5, length 2 m → tip at z=2.5 — below h_1=1.0.
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.5), length=2.0)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.5)
    # No precondition error: the cross-layer path runs.
    res = eng.solve(world)
    Z = (res.electrode_potentials["g1"][0]
         / res.electrode_currents["g1"][0])
    assert Z.real > 0.0
    assert math.isfinite(Z.real)


# ---------------------------------------------------------------------
# 6. Series convergence
# ---------------------------------------------------------------------


def test_two_layer_series_converges_within_max_terms() -> None:
    """For moderate K the series must terminate in fewer than 100 terms."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)  # |K|≈0.67
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.0), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    res = gf.create_engine(backend="image", segment_length=0.05).solve(world)

    assert res.metadata["converged"] is True
    assert res.metadata["n_terms_used"] < 100


# ---------------------------------------------------------------------
# 7. Potential field decays monotonically
# ---------------------------------------------------------------------


def test_two_layer_potential_decays_monotonically() -> None:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=10.0, h_1=2.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.0), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    res = gf.create_engine(backend="image", segment_length=0.05).solve(world)

    xs = np.linspace(1.0, 50.0, 60)
    pts = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    phi = res.potential(pts).real
    assert (np.diff(phi) <= 1e-9).all(), \
        f"Potential not monotonic: max Δ = {np.diff(phi).max():.2e}"


# ---------------------------------------------------------------------
# 8. Cross-engine via compare_engines
# ---------------------------------------------------------------------


def test_compare_engines_consistent_at_K_zero() -> None:
    """For ρ₁=ρ₂ the homogeneous and 2-layer (with equal ρ) worlds are
    physically identical and must produce identical Z."""
    rho = 100.0
    soil_h = gf.HomogeneousSoil(resistivity=rho)
    soil_2 = gf.TwoLayerSoil(rho_1=rho, rho_2=rho, h_1=2.0)

    def _build(soil):
        w = gf.create_world(soil=soil)
        gf.create_electrode(w, "rod", name="g1",
                            position=(0, 0, 0.0), length=1.5)
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    # Both worlds have the same electrode configuration; run the same
    # engine but with each soil in turn.
    eng = gf.create_engine(backend="image", segment_length=0.05)
    Z_h = eng.solve(_build(soil_h)).cluster_impedance("g1")[0].real
    Z_2 = eng.solve(_build(soil_2)).cluster_impedance("g1")[0].real
    assert Z_h == pytest.approx(Z_2, abs=1e-9)
