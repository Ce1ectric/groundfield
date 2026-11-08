"""One error type for a backend that cannot take the soil model (audit A1-08, F09).

:class:`~groundfield.solver.registry.BackendCapabilityError` (a
``ValueError`` and ``NotImplementedError``) is raised by
:meth:`Engine.solve` before any backend code runs. The backend functions
called directly raised four different types for the same situation on
the 2026.11 base: ``TypeError`` (``solve_image`` with a layered soil,
``solve_image_2layer`` with a homogeneous soil, ``solve_mom`` with three
layers), ``ValueError`` (``solve_image_nlayer`` with three layers) and
``NotImplementedError`` (``solve_mutual_*`` with three layers, the
concrete-shell guard of the mom family). They now raise the same
``BackendCapabilityError``; ``TypeError`` is kept for an object that is no
soil model at all.

Changed against v0.15.0 (documented in the changelog): ``image_2layer``
raised ``TypeError`` for a homogeneous soil or a layer stack, ``mom`` for
three or more layers; ``except TypeError`` no longer catches these.
"""

from __future__ import annotations

import warnings

import pytest

import groundfield as gf
from groundfield.solver import image, image_2layer, image_nlayer, mom, mom_sommerfeld
from groundfield.solver.registry import BackendCapabilityError

SOILS = {
    "homogeneous": lambda: gf.HomogeneousSoil(rho=100.0),
    "two_layer": lambda: gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=3.0),
    "stack1": lambda: gf.MultiLayerSoil(layers=[gf.SoilLayer(rho=100.0)]),
    "stack2": lambda: gf.MultiLayerSoil(
        layers=[gf.SoilLayer(rho=100.0, h=3.0), gf.SoilLayer(rho=300.0)]
    ),
    "stack3": lambda: gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=100.0, h=3.0),
            gf.SoilLayer(rho=300.0, h=5.0),
            gf.SoilLayer(rho=50.0),
        ]
    ),
}


def _world(soil: str, **electrode_kw) -> gf.World:
    w = gf.create_world(soil=SOILS[soil]())
    gf.create_electrode(
        w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=1.0, **electrode_kw
    )
    gf.create_source(w, attached_to="g1")
    return w


def _engine(**kw) -> gf.Engine:
    return gf.Engine(segment_length=0.5, **kw)


UNSUPPORTED_SOLVE = [
    ("image_2layer", "homogeneous"),
    ("image_2layer", "stack1"),
    ("image_2layer", "stack3"),
    ("image", "stack3"),
    ("image_nlayer", "stack3"),
    ("mom", "stack3"),
    ("cim", "stack3"),
    ("bem", "stack3"),
]


@pytest.mark.parametrize("backend, soil", UNSUPPORTED_SOLVE)
def test_engine_solve_raises_one_error_type(backend, soil) -> None:
    """Guard (already so on the base): the registry check of Engine.solve."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(BackendCapabilityError) as info:
            _engine(backend=backend).solve(_world(soil))
    assert isinstance(info.value, ValueError | NotImplementedError)
    assert not isinstance(info.value, TypeError)


DIRECT = [
    ("solve_image", image.solve_image, "two_layer"),
    ("solve_image", image.solve_image, "stack1"),
    ("solve_image", image.solve_image, "stack3"),
    ("solve_image_2layer", image_2layer.solve_image_2layer, "homogeneous"),
    ("solve_image_nlayer", image_nlayer.solve_image_nlayer, "stack3"),
    ("solve_mom", mom.solve_mom, "stack3"),
]


@pytest.mark.finding("A1-08", "F09")
@pytest.mark.parametrize(
    "name, solve, soil", DIRECT, ids=[f"{d[0]}-{d[2]}" for d in DIRECT]
)
def test_direct_backend_calls_raise_the_same_error(name, solve, soil) -> None:
    with pytest.raises(BackendCapabilityError, match=name.removeprefix("solve_")):
        solve(_world(soil), _engine())


def test_mutual_matrices_raise_the_same_error_for_three_layers() -> None:
    """Guard: the registry check in front of the assembly raises it."""
    from groundfield.solver.mutual import _prepare

    with pytest.raises(BackendCapabilityError):
        _prepare(_world("stack3"), _engine(), 0, where="solve_mutual_matrix")


@pytest.mark.finding("A1-08")
@pytest.mark.parametrize(
    "solve",
    [mom.solve_mom, mom_sommerfeld.solve_mom_sommerfeld],
    ids=["mom", "mom_sommerfeld"],
)
def test_the_concrete_shell_guard_raises_the_same_error(solve) -> None:
    world = _world("homogeneous", concrete_shell_coefficient_ohm_m=10.0)
    with pytest.raises(BackendCapabilityError, match="concrete"):
        solve(world, _engine())


@pytest.mark.parametrize(
    "solve",
    [
        image.solve_image,
        image_2layer.solve_image_2layer,
        image_nlayer.solve_image_nlayer,
        mom.solve_mom,
        mom_sommerfeld.solve_mom_sommerfeld,
    ],
    ids=["image", "image_2layer", "image_nlayer", "mom", "mom_sommerfeld"],
)
def test_an_object_that_is_no_soil_model_is_a_type_error(solve) -> None:
    world = _world("homogeneous")
    world.__dict__["soil"] = object()  # bypass validation: not a soil model
    with pytest.raises(TypeError, match="object"):
        solve(world, _engine())


@pytest.mark.parametrize(
    "solve, soil",
    [
        (image_2layer.solve_image_2layer, "stack1"),
        (image_2layer.solve_image_2layer, "stack2"),
        (image_nlayer.solve_image_nlayer, "stack2"),
        (mom.solve_mom, "stack2"),
    ],
)
def test_stacks_with_at_most_two_distinct_layers_still_solve(solve, soil) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve(_world(soil), _engine())
    assert result.electrode_potentials["g1"][0].real > 0.0
