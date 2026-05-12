"""Tests for the direct-Sommerfeld MoM backend (``mom_sommerfeld``).

The Sommerfeld-quadrature engine is the methodologically independent
reference within the layered family. These tests check that on
homogeneous soil it agrees with ``image``, that on 2-layer soil it
agrees with the closed-form Tagg/Sunde series, and that the kernel
collapses correctly to the homogeneous limit when the layer
contrasts vanish.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver._layered import LayerStack
from groundfield.solver.mom_sommerfeld import sommerfeld_kernel_value

SEG = 0.10  # coarser segments — Sommerfeld quadrature is expensive


def _world(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_kernel_homogeneous_closed_form() -> None:
    """The kernel on a 1-layer stack is the homogeneous closed form."""
    stack = LayerStack(rhos=np.array([100.0]), h=np.zeros(0))
    s = 1.0
    z = 0.5
    z_s = 1.0
    G = sommerfeld_kernel_value(stack, s=s, z=z, z_s=z_s)
    r = float(np.sqrt(s ** 2 + (z - z_s) ** 2))
    r_img = float(np.sqrt(s ** 2 + (z + z_s) ** 2))
    expected = 1.0 / r + 1.0 / r_img
    assert G == pytest.approx(expected, rel=1e-6)


def test_kernel_two_layer_K_zero() -> None:
    """For ρ₁ = ρ₂ the layered kernel must equal the homogeneous one."""
    rho = 100.0
    stack = LayerStack(rhos=np.array([rho, rho]), h=np.array([2.0]))
    s, z, z_s = 1.0, 0.5, 1.0
    G = sommerfeld_kernel_value(stack, s=s, z=z, z_s=z_s)
    r = float(np.sqrt(s ** 2 + (z - z_s) ** 2))
    r_img = float(np.sqrt(s ** 2 + (z + z_s) ** 2))
    expected = 1.0 / r + 1.0 / r_img
    assert G == pytest.approx(expected, rel=1e-4)


def test_mom_sommerfeld_homogeneous() -> None:
    """``mom_sommerfeld`` on homogeneous soil agrees with ``image`` ≤ 5 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng_image = gf.create_engine(backend="image", segment_length=SEG)
    eng_som = gf.create_engine(backend="mom_sommerfeld", segment_length=SEG)
    Z_image = eng_image.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_som = eng_som.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert abs(Z_som - Z_image) / Z_image < 0.05


@pytest.mark.parametrize("rho_2", [200.0, 500.0])
def test_mom_sommerfeld_two_layer_matches_image_2layer(rho_2: float) -> None:
    """``mom_sommerfeld`` agrees with ``image_2layer`` to ≤ 5 %."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=rho_2, h_1=2.0)
    eng_il2 = gf.create_engine(backend="image_2layer", segment_length=SEG)
    eng_som = gf.create_engine(backend="mom_sommerfeld", segment_length=SEG)
    Z_il2 = eng_il2.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_som = eng_som.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert abs(Z_som - Z_il2) / Z_il2 < 0.05


def test_mom_sommerfeld_metadata() -> None:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    eng = gf.create_engine(backend="mom_sommerfeld", segment_length=SEG)
    res = eng.solve(_world(soil))
    assert res.backend == "mom_sommerfeld"
    assert res.metadata["solver"] == "galerkin"
    assert "lambda_max_factor" in res.metadata
