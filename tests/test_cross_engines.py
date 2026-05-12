"""Cross-engine consistency tests (image / image_2layer / mom).

These tests are the central guarantee that the engines stay in sync.
They do *not* check absolute accuracy (that is done in
``test_dwight_references.py`` and ``test_two_layer.py``); their role
is to detect drift between the methodologically independent backends.

Tolerances
----------
- image vs. mom on a homogeneous world:        ≤ 2 % (typical 0.5 %)
- image_2layer vs. mom on a 2-layer world:     ≤ 2 % (typical 0.5 %)
- image_2layer at K=0 vs. image (homogeneous): exact (== 0)
- mom on a 2-layer world at K=0 vs. mom on the equivalent homogeneous
  world:                                       exact (== 0)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf
from groundfield.references import dwight1936 as dw


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _world(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


SEG = 0.05  # segment_length used everywhere — matched between engines


# ---------------------------------------------------------------------
# 1. Homogeneous: image vs. mom
# ---------------------------------------------------------------------


def test_image_vs_mom_homogeneous_consistent() -> None:
    """``image`` and ``mom`` on the same homogeneous world must match
    within 2 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng_image = gf.create_engine(backend="image", segment_length=SEG)
    eng_mom = gf.create_engine(backend="mom", segment_length=SEG)

    Z_image = eng_image.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_mom = eng_mom.solve(_world(soil)).cluster_impedance("g1")[0].real

    rel = abs(Z_image - Z_mom) / Z_mom
    assert rel < 0.02, f"image={Z_image:.3f} vs mom={Z_mom:.3f}, Δ={rel*100:.2f}%"


def test_compare_engines_image_vs_mom_homogeneous() -> None:
    """``compare_engines`` must report 'consistent' for image vs. mom
    on a homogeneous world (tolerance 5 %)."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    report = gf.compare_engines(
        _world(soil),
        engines={
            "image": gf.create_engine(backend="image", segment_length=SEG),
            "mom": gf.create_engine(backend="mom", segment_length=SEG),
        },
        rel_tolerance=0.05,
    )
    assert report.is_consistent, report.summary()


@pytest.mark.parametrize(
    "L, wire_radius",
    [(1.5, 0.005), (3.0, 0.005), (1.5, 0.01)],
)
def test_mom_rod_matches_dwight(L: float, wire_radius: float) -> None:
    """``mom`` alone must agree with Dwight to within 5 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world, "rod", name="g1",
        position=(0, 0, 0.0), length=L, wire_radius=wire_radius,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="mom", segment_length=SEG)
    Z = eng.solve(world).cluster_impedance("g1")[0].real
    R_dw = dw.rod(rho=100.0, length=L, radius=wire_radius)
    rel = abs(Z - R_dw) / R_dw
    assert rel < 0.05, (
        f"L={L}, a={wire_radius}: mom={Z:.3f} vs Dwight={R_dw:.3f}, "
        f"Δ={rel*100:.2f}%"
    )


# ---------------------------------------------------------------------
# 2. 2-layer: image_2layer vs. mom
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "rho_2",
    [50.0, 110.0, 200.0, 500.0, 1000.0],
)
def test_image_2layer_vs_mom_consistent(rho_2: float) -> None:
    """For a range of layer contrasts ``image_2layer`` and ``mom`` must
    agree within 2 %."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=rho_2, h_1=2.0)
    eng_il2 = gf.create_engine(backend="image_2layer", segment_length=SEG)
    eng_mom = gf.create_engine(backend="mom", segment_length=SEG)

    Z_il2 = eng_il2.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_mom = eng_mom.solve(_world(soil)).cluster_impedance("g1")[0].real

    rel = abs(Z_il2 - Z_mom) / Z_mom
    assert rel < 0.02, (
        f"rho_2={rho_2}: image_2layer={Z_il2:.3f} vs mom={Z_mom:.3f}, "
        f"Δ={rel*100:.2f}%"
    )


def test_compare_engines_image_2layer_vs_mom_consistent() -> None:
    """``compare_engines`` reports consistent for image_2layer vs. mom
    on a 2-layer world."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
    report = gf.compare_engines(
        _world(soil),
        engines={
            "image_2layer": gf.create_engine(
                backend="image_2layer", segment_length=SEG
            ),
            "mom": gf.create_engine(backend="mom", segment_length=SEG),
        },
        rel_tolerance=0.05,
    )
    assert report.is_consistent, report.summary()


# ---------------------------------------------------------------------
# 3. K=0 collapse — both engines must reduce exactly
# ---------------------------------------------------------------------


def test_mom_K0_equals_mom_homogeneous_exactly() -> None:
    """``mom`` on a 2-layer world with ρ₁=ρ₂ must equal ``mom`` on the
    homogeneous world bit-exactly."""
    rho = 100.0
    eng_mom = gf.create_engine(backend="mom", segment_length=SEG)

    Z_homog = eng_mom.solve(
        _world(gf.HomogeneousSoil(resistivity=rho))
    ).cluster_impedance("g1")[0].real
    Z_K0 = eng_mom.solve(
        _world(gf.TwoLayerSoil(rho_1=rho, rho_2=rho, h_1=2.0))
    ).cluster_impedance("g1")[0].real

    assert Z_K0 == pytest.approx(Z_homog, abs=1e-9)


def test_three_engines_consistent_at_K0() -> None:
    """At K=0 all three engines on the same world must collapse together."""
    rho = 100.0
    soil = gf.TwoLayerSoil(rho_1=rho, rho_2=rho, h_1=2.0)
    eng_image = gf.create_engine(backend="image", segment_length=SEG)
    eng_il2 = gf.create_engine(backend="image_2layer", segment_length=SEG)
    eng_mom = gf.create_engine(backend="mom", segment_length=SEG)

    # eng_image runs on the homogeneous equivalent
    Z_image = eng_image.solve(
        _world(gf.HomogeneousSoil(resistivity=rho))
    ).cluster_impedance("g1")[0].real
    Z_il2 = eng_il2.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_mom = eng_mom.solve(_world(soil)).cluster_impedance("g1")[0].real

    # image and image_2layer must match exactly (same kernel at K=0)
    assert Z_il2 == pytest.approx(Z_image, abs=1e-9)
    # mom shares the kernel; difference is purely the resolution scheme
    assert abs(Z_mom - Z_image) / Z_image < 0.02


# ---------------------------------------------------------------------
# 4. Multi-electrode cross-check
# ---------------------------------------------------------------------


def test_image_vs_mom_with_cluster() -> None:
    """Two electrodes connected by a conductor: both engines must agree
    on the cluster impedance and the per-electrode current split."""
    soil = gf.HomogeneousSoil(resistivity=100.0)

    def _build(eng):
        w = gf.create_world(soil=soil)
        g1 = gf.create_electrode(w, "rod", name="g1",
                                 position=(0, 0, 0.0), length=1.5)
        g2 = gf.create_electrode(w, "ring", name="g2",
                                 center=(8.0, 0, 0.8), radius=2.0)
        gf.create_conductor(w, name="l1", start=g1, end=g2)
        gf.create_source(w, attached_to=g1, magnitude=10.0)
        return eng.solve(w)

    r_image = _build(gf.create_engine(backend="image", segment_length=SEG))
    r_mom = _build(gf.create_engine(backend="mom", segment_length=SEG))

    Z_image = r_image.cluster_impedance("g1")[0].real
    Z_mom = r_mom.cluster_impedance("g1")[0].real
    assert abs(Z_image - Z_mom) / Z_mom < 0.02

    # Current split: both engines agree on the partitioning.
    I1_image = r_image.electrode_currents["g1"][0].real
    I1_mom = r_mom.electrode_currents["g1"][0].real
    assert abs(I1_image - I1_mom) / abs(I1_mom) < 0.05


# ---------------------------------------------------------------------
# 5. Potential-field consistency
# ---------------------------------------------------------------------


def test_image_vs_mom_potential_field_consistent() -> None:
    """Potential evaluated on a sample grid must agree between image
    and mom within 5 % (the bigger envelope reflects the per-segment
    current redistribution)."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = _world(soil)
    eng_image = gf.create_engine(backend="image", segment_length=SEG)
    eng_mom = gf.create_engine(backend="mom", segment_length=SEG)

    pts = np.array([
        [2.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [2.0, 2.0, 0.5],
    ])
    phi_image = eng_image.solve(world).potential(pts).real
    phi_mom = eng_mom.solve(world).potential(pts).real

    rel = np.max(np.abs(phi_image - phi_mom) / np.abs(phi_mom))
    assert rel < 0.05, f"max relative deviation {rel*100:.2f}%"
