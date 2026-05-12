"""Tests for the ``image_nlayer`` dispatcher.

Validates that the n-layer dispatcher reduces correctly to the
homogeneous and the 2-layer backends, raises a clear error for
``n >= 3``, and that the backend tag in the result is rewritten to
``image_nlayer``.
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


def test_image_nlayer_n1_matches_image() -> None:
    """For ``n_layers = 1`` image_nlayer must equal the image backend."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng_image = gf.create_engine(backend="image", segment_length=SEG)
    eng_nlayer = gf.create_engine(backend="image_nlayer", segment_length=SEG)
    Z_image = eng_image.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_nlayer = eng_nlayer.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert Z_nlayer == pytest.approx(Z_image, rel=1e-9)


def test_image_nlayer_n2_matches_image_2layer() -> None:
    """For ``n_layers = 2`` image_nlayer must equal image_2layer."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    eng_il2 = gf.create_engine(backend="image_2layer", segment_length=SEG)
    eng_nlayer = gf.create_engine(backend="image_nlayer", segment_length=SEG)
    Z_il2 = eng_il2.solve(_world(soil)).cluster_impedance("g1")[0].real
    Z_nlayer = eng_nlayer.solve(_world(soil)).cluster_impedance("g1")[0].real
    assert Z_nlayer == pytest.approx(Z_il2, rel=1e-9)


def test_image_nlayer_backend_tag() -> None:
    """The dispatcher must rewrite the backend tag to ``image_nlayer``."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    eng = gf.create_engine(backend="image_nlayer", segment_length=SEG)
    res = eng.solve(_world(soil))
    assert res.backend == "image_nlayer"
    assert res.metadata["dispatched_to"] in {"image", "image_2layer"}


def test_image_nlayer_n3_raises() -> None:
    """For ``n_layers >= 3`` the dispatcher must raise with a helpful message."""
    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=100.0, thickness=1.0),
            gf.SoilLayer(resistivity=300.0, thickness=2.0),
            gf.SoilLayer(resistivity=50.0, thickness=None),
        ]
    )
    eng = gf.create_engine(backend="image_nlayer", segment_length=SEG)
    with pytest.raises(ValueError, match="cim|mom_sommerfeld|bem"):
        eng.solve(_world(soil))


def test_image_nlayer_rod_sunde() -> None:
    """A single rod in homogeneous soil reproduces Sunde (within 5 %)."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng = gf.create_engine(backend="image_nlayer", segment_length=SEG)
    Z = eng.solve(_world(soil)).cluster_impedance("g1")[0].real
    R_dw = dw.rod(rho=100.0, length=1.5, radius=0.005)
    assert abs(Z - R_dw) / R_dw < 0.05
