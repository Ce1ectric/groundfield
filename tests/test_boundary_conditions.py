"""Boundary conditions: inert fields warn (review pass 10, finding 10.51).

No backend reads :class:`~groundfield.BoundaryConditions`: every engine
imposes φ → 0 at infinity (Dirichlet, remote earth) through the
``1/r`` decay of its Green's function, a Neumann surface ``z = 0``
through the air-side image, and reports potentials relative to remote
earth (``reference_node`` is never consulted). Up to v0.15.0 only
:meth:`World.set_boundary_conditions` warned about non-default values;
the constructor path (``BoundaryConditions(far_field="neumann")`` handed
to :func:`groundfield.create_world`, or a world JSON) was silent.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import groundfield as gf

_NON_DEFAULT = [
    ("far_field", "neumann"),
    ("surface", "dirichlet"),
    ("reference_node", "g1"),
]


@pytest.mark.finding("10.51")
@pytest.mark.parametrize("field, value", _NON_DEFAULT)
def test_non_default_field_warns_on_construction(field, value) -> None:
    with pytest.warns(UserWarning, match=field):
        gf.BoundaryConditions(**{field: value})


@pytest.mark.finding("10.51")
def test_warning_category_and_message() -> None:
    from groundfield.utils import InertConfigurationWarning

    with pytest.warns(InertConfigurationWarning, match="no backend") as record:
        gf.BoundaryConditions(far_field="neumann")
    assert record[0].filename == __file__


@pytest.mark.finding("10.51")
def test_defaults_are_silent() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gf.BoundaryConditions()
        gf.BoundaryConditions(
            far_field="dirichlet", surface="neumann", reference_node=None
        )
        gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    assert not caught, [str(w.message) for w in caught]


@pytest.mark.finding("10.51")
def test_create_world_with_non_default_boundary_warns_once() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        boundary = gf.BoundaryConditions(surface="dirichlet")
        gf.create_world(soil=gf.HomogeneousSoil(rho=100.0), boundary=boundary)
    inert = [w for w in caught if "surface" in str(w.message)]
    assert len(inert) == 1, [str(w.message) for w in caught]


@pytest.mark.finding("10.51")
def test_set_boundary_conditions_still_warns_exactly_once() -> None:
    """The world-level helper keeps its own warning; no duplicate from the model."""
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        world.set_boundary_conditions(far_field="neumann")
    hits = [w for w in caught if "far_field" in str(w.message)]
    assert len(hits) == 1, [str(w.message) for w in caught]


@pytest.mark.finding("10.51")
def test_world_json_with_non_default_boundary_warns_on_load() -> None:
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    data = json.loads(world.model_dump_json())
    data["boundary"]["reference_node"] = "g1"
    with pytest.warns(UserWarning, match="reference_node"):
        gf.World.model_validate(data)


@pytest.mark.finding("10.51")
@pytest.mark.parametrize(
    "soil, backend",
    [
        (gf.HomogeneousSoil(rho=100.0), "image"),
        (gf.HomogeneousSoil(rho=100.0), "mom"),
        (gf.HomogeneousSoil(rho=100.0), "cim"),
        (gf.HomogeneousSoil(rho=100.0), "bem"),
        (gf.HomogeneousSoil(rho=100.0), "mom_sommerfeld"),
        (gf.HomogeneousSoil(rho=100.0), "fem"),
        (gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=3.0), "image_2layer"),
        (gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=3.0), "image_nlayer"),
    ],
)
def test_boundary_conditions_are_inert_in_every_backend(soil, backend) -> None:
    """The evidence behind the warning: non-default values change nothing."""

    def solve(boundary: gf.BoundaryConditions) -> np.ndarray:
        world = gf.create_world(soil=soil, boundary=boundary)
        gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=2.0)
        gf.create_electrode(world, "ring", name="g2", center=(6, 0, 0.6), radius=1.0)
        gf.create_source(world, attached_to="g1", magnitude=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = world.solve(gf.create_engine(backend=backend, segment_length=0.5))
        names = sorted(res.electrode_potentials)
        return np.array(
            [res.electrode_potentials[n][0] for n in names]
            + [res.electrode_currents[n][0] for n in names]
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        odd = gf.BoundaryConditions(
            far_field="neumann", surface="dirichlet", reference_node="g2"
        )
    assert np.array_equal(solve(gf.BoundaryConditions()), solve(odd))
