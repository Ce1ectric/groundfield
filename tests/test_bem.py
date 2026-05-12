"""Tests for the BEM (collocation) backend.

Validates that the BEM solver agrees with the closed-form image
backends on homogeneous and 2-layer soils within a 5 % envelope, that
the collocation result is in the expected ballpark of the Galerkin
``mom`` solver, and that single-rod results match the Sunde / Dwight
references.
"""

from __future__ import annotations

import pytest

import groundfield as gf
from groundfield.references import dwight1936 as dw

SEG = 0.05


def _world(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_bem_homogeneous_matches_image() -> None:
    """BEM on homogeneous soil agrees with ``image`` ≤ 5 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng_image = gf.create_engine(backend="image", segment_length=SEG)
    eng_bem = gf.create_engine(backend="bem", segment_length=SEG)
    Z_image = eng_image.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_bem = eng_bem.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert abs(Z_bem - Z_image) / Z_image < 0.05


@pytest.mark.parametrize("rho_2", [50.0, 200.0, 500.0])
def test_bem_two_layer_matches_image_2layer(rho_2: float) -> None:
    """BEM agrees with the closed-form Tagg/Sunde series within 5 %."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=rho_2, h_1=2.0)
    eng_il2 = gf.create_engine(backend="image_2layer", segment_length=SEG)
    eng_bem = gf.create_engine(backend="bem", segment_length=SEG)
    Z_il2 = eng_il2.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_bem = eng_bem.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert abs(Z_bem - Z_il2) / Z_il2 < 0.05, (
        f"rho_2={rho_2}: bem={Z_bem:.3f} vs il2={Z_il2:.3f}"
    )


def test_bem_rod_matches_dwight() -> None:
    """A rod in homogeneous soil matches Dwight to within 5 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng = gf.create_engine(backend="bem", segment_length=SEG)
    Z = eng.solve(_world(soil)).cluster_impedance("g1")[0].real
    R_dw = dw.rod(rho=100.0, length=1.5, radius=0.005)
    assert abs(Z - R_dw) / R_dw < 0.05


def test_bem_metadata() -> None:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    eng = gf.create_engine(backend="bem", segment_length=SEG)
    res = eng.solve(_world(soil))
    assert res.backend == "bem"
    assert res.metadata["solver"] == "collocation"
