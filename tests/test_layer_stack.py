"""Layer-stack canonicalisation of the layered-soil backends (10.48).

A ``MultiLayerSoil`` whose adjacent layers share one resistivity is
physically a soil with fewer layers: ``[100, 400, 400]`` over
``h = [2, 3]`` *is* the two-layer soil ``(100, 400, h_1 = 2)``, and
``[rho, rho, rho]`` is homogeneous. Up to 0.15.0 the layered backends
took the declared layer count literally and rejected or mis-routed such
stacks (``cim`` / ``bem``: ``NotImplementedError``, ``mom``:
``TypeError``, ``mom_sommerfeld`` and ``image_nlayer``: ``ValueError``
as soon as an electrode reached below ``h_1``). The canonical stack
merges such layers and drops zero-thickness layers before any backend
looks at it.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver import _layered
from groundfield.solver._layered import LayerStack, as_layer_stack

SEG = 0.5


def _stack(rhos, h) -> LayerStack:
    return LayerStack(rhos=np.asarray(rhos, float), h=np.asarray(h, float))


def _multi(rhos, h) -> gf.MultiLayerSoil:
    layers = [
        gf.SoilLayer(resistivity=r, thickness=t)
        for r, t in zip(rhos[:-1], h, strict=True)
    ]
    layers.append(gf.SoilLayer(resistivity=rhos[-1], thickness=None))
    return gf.MultiLayerSoil(layers=layers)


def _rod_world(soil, *, length: float = 3.0) -> gf.World:
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world, "rod", name="g1", position=(0.0, 0.0, 0.0), length=length
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    return world


def _z(backend: str, soil, **kw) -> complex:
    eng = gf.create_engine(backend=backend, segment_length=SEG)
    return eng.solve(_rod_world(soil, **kw)).cluster_impedance("g1")[0]


# ---------------------------------------------------------------------
# the canonical form itself
# ---------------------------------------------------------------------


@pytest.mark.finding("10.48")
@pytest.mark.parametrize(
    ("rhos", "h", "rhos_c", "h_c"),
    [
        # duplicated bottom layer -> two layers
        ([100.0, 400.0, 400.0], [2.0, 3.0], [100.0, 400.0], [2.0]),
        # duplicated top layers -> thickness adds up
        ([100.0, 100.0, 400.0], [2.0, 3.0], [100.0, 400.0], [5.0]),
        # all equal -> homogeneous
        ([100.0, 100.0, 100.0], [2.0, 3.0], [100.0], []),
        # declared two-layer soil without contrast -> homogeneous
        ([250.0, 250.0], [4.0], [250.0], []),
        # a zero-thickness layer is invisible; its neighbours then merge
        ([100.0, 50.0, 100.0], [2.0, 0.0], [100.0], []),
        # interior duplicates in a genuine four-layer stack
        ([100.0, 30.0, 30.0, 500.0], [1.0, 2.0, 4.0], [100.0, 30.0, 500.0], [1.0, 6.0]),
        # agreement to 1e-13 relative counts as equal
        ([100.0, 100.0 * (1 + 1e-13)], [2.0], [100.0], []),
    ],
)
def test_canonical_stack_merges_indistinguishable_layers(rhos, h, rhos_c, h_c) -> None:
    canon = _layered.canonical_layer_stack(_stack(rhos, h))
    np.testing.assert_array_equal(canon.rhos, np.asarray(rhos_c))
    np.testing.assert_array_equal(canon.h, np.asarray(h_c))
    assert canon.n_layers == len(rhos_c)
    assert canon.h.size == canon.n_layers - 1


@pytest.mark.finding("10.48")
def test_canonical_stack_keeps_genuine_contrasts_untouched() -> None:
    """A stack that is already canonical is returned as the same object."""
    stack = _stack([100.0, 1000.0, 50.0], [2.0, 1.0])
    assert _layered.canonical_layer_stack(stack) is stack
    # 1e-9 relative is a contrast, not a duplicate.
    near = _stack([100.0, 100.0 * (1 + 1e-9)], [2.0])
    assert _layered.canonical_layer_stack(near).n_layers == 2


@pytest.mark.finding("10.48")
def test_as_layer_stack_returns_the_canonical_stack_by_default() -> None:
    soil = _multi([100.0, 100.0, 100.0], [2.0, 3.0])
    assert as_layer_stack(soil).n_layers == 1
    assert as_layer_stack(soil, canonical=False).n_layers == 3
    two = gf.TwoLayerSoil(rho_1=80.0, rho_2=80.0, h_1=1.5)
    assert as_layer_stack(two).n_layers == 1
    assert as_layer_stack(two, canonical=False).n_layers == 2


@pytest.mark.finding("10.48")
def test_canonical_soil_builds_the_matching_soil_model() -> None:
    hom = gf.HomogeneousSoil(resistivity=100.0)
    assert _layered.canonical_soil(hom) is hom
    two = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    assert _layered.canonical_soil(two) is two

    reduced = _layered.canonical_soil(_multi([100.0, 400.0, 400.0], [2.0, 3.0]))
    assert isinstance(reduced, gf.TwoLayerSoil)
    assert (reduced.rho_1, reduced.rho_2, reduced.h_1) == (100.0, 400.0, 2.0)

    flat = _layered.canonical_soil(gf.TwoLayerSoil(rho_1=70.0, rho_2=70.0, h_1=3.0))
    assert isinstance(flat, gf.HomogeneousSoil)
    assert flat.resistivity == 70.0

    four = _layered.canonical_soil(_multi([100.0, 30.0, 30.0, 500.0], [1.0, 2.0, 4.0]))
    assert isinstance(four, gf.MultiLayerSoil)
    assert [ly.resistivity for ly in four.layers] == [100.0, 30.0, 500.0]
    assert [ly.thickness for ly in four.layers] == [1.0, 6.0, None]


# ---------------------------------------------------------------------
# the layered backends solve the canonical stack
# ---------------------------------------------------------------------


@pytest.mark.finding("10.48")
@pytest.mark.parametrize(
    "backend", ["mom", "mom_sommerfeld", "cim", "bem", "image_nlayer"]
)
def test_duplicated_layer_stack_is_solved_like_two_layers(backend: str) -> None:
    """``[100, 400, 400]`` over ``[2, 3]`` is ``TwoLayerSoil(100, 400, 2)``.

    The 3 m rod crosses ``h_1 = 2 m``, which every backend in this list
    except ``image_nlayer`` solves on two-layer soil; v0.15.0 raised for
    all five on the three-layer spelling. ``image_nlayer`` is checked
    with a rod that stays in the upper layer (it routes to
    ``image_2layer``).
    """
    length = 1.5 if backend == "image_nlayer" else 3.0
    z3 = _z(backend, _multi([100.0, 400.0, 400.0], [2.0, 3.0]), length=length)
    z2 = _z(backend, gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0), length=length)
    assert np.isfinite(z3)
    assert z3 == pytest.approx(z2, rel=1e-12, abs=0.0)


@pytest.mark.finding("10.48")
@pytest.mark.parametrize(
    "backend", ["mom", "mom_sommerfeld", "cim", "bem", "image_nlayer"]
)
def test_equal_resistivity_stack_is_solved_as_homogeneous(backend: str) -> None:
    z3 = _z(backend, _multi([100.0, 100.0, 100.0], [0.5, 2.0]))
    z1 = _z(backend, gf.HomogeneousSoil(resistivity=100.0))
    assert z3 == pytest.approx(z1, rel=1e-12, abs=0.0)


@pytest.mark.finding("10.48")
def test_result_soil_is_the_canonical_model_so_potential_works() -> None:
    """``FieldResult.potential`` cannot evaluate n >= 3 soils; the
    canonical two-layer model it receives instead can."""
    soil = _multi([100.0, 400.0, 400.0], [2.0, 3.0])
    res = gf.create_engine(backend="mom_sommerfeld", segment_length=SEG).solve(
        _rod_world(soil, length=1.5)
    )
    ref = gf.create_engine(backend="mom_sommerfeld", segment_length=SEG).solve(
        _rod_world(gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0), length=1.5)
    )
    assert isinstance(res.soil, gf.TwoLayerSoil)
    pts = np.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
    np.testing.assert_allclose(res.potential(pts), ref.potential(pts), rtol=1e-12)
    assert res.metadata["n_layers"] == 2


@pytest.mark.finding("10.48")
def test_genuine_three_layer_stack_is_still_rejected_where_unsupported() -> None:
    """Canonicalisation must not hide a real third layer."""
    soil = _multi([100.0, 400.0, 50.0], [2.0, 3.0])
    for backend in ("cim", "bem"):
        with pytest.raises(NotImplementedError, match="mom_sommerfeld"):
            _z(backend, soil, length=1.5)
    with pytest.raises(TypeError, match="mom_sommerfeld"):
        _z("mom", soil, length=1.5)


@pytest.mark.finding("10.48")
@pytest.mark.parametrize(
    ("multi", "plain"),
    [
        (
            gf.MultiLayerSoil(
                layers=[
                    gf.SoilLayer(resistivity=100.0, thickness=2.0),
                    gf.SoilLayer(resistivity=400.0, thickness=None),
                ]
            ),
            gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0),
        ),
        (
            gf.MultiLayerSoil(layers=[gf.SoilLayer(resistivity=100.0, thickness=None)]),
            gf.HomogeneousSoil(resistivity=100.0),
        ),
    ],
    ids=["two_layers", "one_layer"],
)
def test_mom_accepts_a_multilayer_soil_with_at_most_two_layers(multi, plain) -> None:
    """Already-canonical MultiLayerSoil with n <= 2 maps to the plain model."""
    assert isinstance(_layered.canonical_soil(multi), type(plain))
    assert _z("mom", multi) == pytest.approx(_z("mom", plain), rel=1e-12, abs=0.0)
