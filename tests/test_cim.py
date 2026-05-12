"""Tests for the Complex-Image-Method backend (``cim``).

Checks that the matrix-pencil fit of Γ_1(λ) collapses to zero on
homogeneous soil, that on two-layer soils ``cim`` agrees with the
closed-form Tagg/Sunde series within 2 %, and that ``cim`` is
self-consistent across simple multi-layer reductions (the
``ρ_2 = ρ_1 = ρ_3`` collapse should give the homogeneous result).
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield.references import dwight1936 as dw
from groundfield.solver._layered import LayerStack
from groundfield.solver.cim import fit_complex_images

SEG = 0.05


def _world(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_cim_homogeneous_collapses_to_image() -> None:
    """For HomogeneousSoil the CIM fit produces zero images."""
    stack = LayerStack(rhos=np.array([100.0]), h=np.zeros(0))
    fit = fit_complex_images(stack)
    assert fit.a.size == 0
    assert fit.beta.size == 0


def test_cim_homogeneous_matches_image() -> None:
    """``cim`` on homogeneous soil agrees with ``image`` to within 2 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng_image = gf.create_engine(backend="image", segment_length=SEG)
    eng_cim = gf.create_engine(backend="cim", segment_length=SEG)
    Z_image = eng_image.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_cim = eng_cim.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert abs(Z_cim - Z_image) / Z_image < 0.02


@pytest.mark.parametrize("rho_2", [50.0, 200.0, 500.0])
def test_cim_two_layer_matches_image_2layer(rho_2: float) -> None:
    """For 2-layer soil ``cim`` and ``image_2layer`` must agree within 5 %."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=rho_2, h_1=2.0)
    eng_il2 = gf.create_engine(backend="image_2layer", segment_length=SEG)
    eng_cim = gf.create_engine(backend="cim", segment_length=SEG)
    Z_il2 = eng_il2.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_cim = eng_cim.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert abs(Z_cim - Z_il2) / Z_il2 < 0.05, (
        f"rho_2={rho_2}: cim={Z_cim:.3f} vs il2={Z_il2:.3f}"
    )


def test_cim_three_layer_collapse() -> None:
    """ρ₁ = ρ₂ = ρ₃ → CIM result must collapse to the homogeneous one."""
    rho = 100.0
    # Upper layer must be deep enough to contain the 1.5 m rod.
    multilayer = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=rho, thickness=2.0),
            gf.SoilLayer(resistivity=rho, thickness=2.0),
            gf.SoilLayer(resistivity=rho, thickness=None),
        ]
    )
    homog = gf.HomogeneousSoil(resistivity=rho)
    eng = gf.create_engine(backend="cim", segment_length=SEG)
    Z_ml = eng.solve(_world(multilayer)).cluster_impedance("g1")[0].real
    Z_h = eng.solve(_world(homog)).cluster_impedance("g1")[0].real
    assert abs(Z_ml - Z_h) / Z_h < 0.02


def test_cim_rod_matches_dwight() -> None:
    """A rod in homogeneous soil matches Dwight to within 5 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng = gf.create_engine(backend="cim", segment_length=SEG)
    Z = eng.solve(_world(soil)).cluster_impedance("g1")[0].real
    R_dw = dw.rod(rho=100.0, length=1.5, radius=0.005)
    assert abs(Z - R_dw) / R_dw < 0.05


def test_cim_metadata() -> None:
    """Result metadata exposes the fit diagnostics.

    For ``n_layers = 2`` the engine deliberately uses the closed-form
    Tagg/Sunde self-kernel (the matrix-pencil fit of a constant
    ``Γ_1 = K_1`` is ill-conditioned) and therefore reports
    ``cim_n_images = 0``. For ``n_layers >= 3`` the genuine CIM fit
    runs and produces ``cim_n_images >= 1``.
    """
    eng = gf.create_engine(backend="cim", segment_length=SEG)

    soil_2 = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    res_2 = eng.solve(_world(soil_2))
    assert res_2.backend == "cim"
    assert "cim_n_images" in res_2.metadata
    assert "cim_rms" in res_2.metadata
    assert res_2.metadata["cim_n_images"] == 0

    soil_3 = gf.MultiLayerSoil(layers=[
        gf.SoilLayer(resistivity=100.0, thickness=2.0),
        gf.SoilLayer(resistivity=400.0, thickness=2.0),
        gf.SoilLayer(resistivity=50.0, thickness=None),
    ])
    res_3 = eng.solve(_world(soil_3))
    assert res_3.metadata["cim_n_images"] >= 1
