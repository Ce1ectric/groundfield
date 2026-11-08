"""image_2layer with a layer stack: canonical form, consistent errors.

A :class:`MultiLayerSoil` whose canonical stack (equal adjacent layers
merged, review pass 10, 10.48) has two layers passes the registry's
capability check for ``image_2layer``. Up to the audit of the 2026.11 work
(round 1, A1-numerics F09) the backend then raised a bare ``TypeError``;
it now solves the canonical two-layer soil, and a stack that really has
three layers is rejected with the registry's ``BackendCapabilityError``.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver.image_2layer import solve_image_2layer
from groundfield.solver.registry import BackendCapabilityError


def _rod_world(soil) -> gf.World:
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(world, attached_to="g", magnitude=1.0)
    return world


def _stack(*layers: tuple[float, float | None]) -> gf.MultiLayerSoil:
    return gf.MultiLayerSoil(layers=[gf.SoilLayer(rho=r, h=h) for r, h in layers])


@pytest.mark.finding("10.48")
@pytest.mark.parametrize("entry", ["engine", "function"])
def test_two_layer_stack_solves_like_the_two_layer_soil(entry: str) -> None:
    """300 Ω·m over 2 + 4 m, then 50 Ω·m: the two-layer soil 300/50, h_1 = 6 m."""
    stack = _stack((300.0, 2.0), (300.0, 4.0), (50.0, None))
    engine = gf.create_engine(backend="image_2layer", segment_length=0.5)
    if entry == "engine":
        res = engine.solve(_rod_world(stack))
    else:
        res = solve_image_2layer(_rod_world(stack), engine)
    ref = engine.solve(_rod_world(gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=6.0)))
    np.testing.assert_allclose(
        res.cluster_impedance("g"), ref.cluster_impedance("g"), rtol=1e-14
    )


@pytest.mark.finding("10.48")
def test_three_distinct_layers_are_rejected_consistently() -> None:
    stack = _stack((300.0, 2.0), (100.0, 4.0), (50.0, None))
    engine = gf.create_engine(backend="image_2layer", segment_length=0.5)
    with pytest.raises(BackendCapabilityError):
        engine.solve(_rod_world(stack))
    with pytest.raises(BackendCapabilityError):
        solve_image_2layer(_rod_world(stack), engine)
