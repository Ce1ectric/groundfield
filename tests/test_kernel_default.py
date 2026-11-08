"""The exact line-source kernel is the default of the image family (review pass 10, 10.28).

``Engine.kernel`` defaults to ``None``: the backend's preferred kernel, the
exact line kernel for the image family and the backend's own (point) kernel
elsewhere; ``kernel="point"`` reproduces releases up to 0.15.0. The request
is carried in the value (audit round 1, A1-03): an engine rebuilt from JSON,
``with_frequencies`` or a result's provenance block runs like the original —
with "explicit" read from ``model_fields_set`` a rebuilt ``mom`` engine would
have requested the line kernel and raised.
"""

from __future__ import annotations

import warnings

import pytest

import groundfield as gf
from groundfield.solver.registry import BackendCapabilityError

_FINDING = pytest.mark.finding("10.28")


def _rod_world(soil=None) -> gf.World:
    w = gf.create_world(soil=soil or gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="g", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="g", magnitude=1.0)
    return w


def _solve(world: gf.World, engine: gf.Engine) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world.solve(engine)


@_FINDING
@pytest.mark.parametrize(
    "soil",
    [gf.HomogeneousSoil(rho=100.0), gf.TwoLayerSoil(rho_1=200.0, rho_2=50.0, h_1=4.0)],
    ids=["homogeneous", "two_layer"],
)
def test_default_engine_runs_the_line_kernel_in_the_image_family(soil) -> None:
    res = _solve(_rod_world(soil), gf.Engine(segment_length=0.5))
    assert res.metadata["kernel"] == "line"
    line = _solve(_rod_world(soil), gf.Engine(segment_length=0.5, kernel="line"))
    point = _solve(_rod_world(soil), gf.Engine(segment_length=0.5, kernel="point"))
    z = res.cluster_impedance("g")[0]
    assert z == line.cluster_impedance("g")[0]
    assert abs(z / point.cluster_impedance("g")[0] - 1.0) > 3e-3


@_FINDING
def test_default_request_is_carried_in_the_value() -> None:
    engine = gf.Engine(backend="mom")
    assert engine.kernel is None
    assert engine.model_dump()["kernel"] is None


@_FINDING
@pytest.mark.parametrize("rebuild", ["with_frequencies", "json", "provenance"])
def test_a_rebuilt_engine_solves_like_the_original(rebuild: str) -> None:
    """A1-03: the default is not turned into an explicit line request."""
    engine = gf.Engine(backend="mom", segment_length=0.5)
    res = _solve(_rod_world(), engine)
    assert res.metadata["kernel"] == "point"
    assert res.metadata["provenance"]["engine"]["kernel"] is None
    if rebuild == "with_frequencies":
        again = engine.with_frequencies(50.0)
    elif rebuild == "json":
        again = gf.Engine.model_validate_json(engine.model_dump_json())
    else:
        again = gf.Engine(**res.metadata["provenance"]["engine"])
    assert again == engine
    again_res = _solve(_rod_world(), again)
    assert again_res.cluster_impedance("g") == res.cluster_impedance("g")


def test_explicit_line_kernel_on_a_backend_without_it_raises() -> None:
    """Guard: an explicit request the backend cannot honour stays an error."""
    with pytest.raises(BackendCapabilityError, match="line"):
        _solve(_rod_world(), gf.Engine(backend="mom", kernel="line"))


@_FINDING
def test_direct_backend_and_mutual_calls_use_the_default_kernel() -> None:
    from groundfield.solver.image_2layer import solve_image_2layer

    soil = gf.TwoLayerSoil(rho_1=200.0, rho_2=50.0, h_1=4.0)
    engine = gf.Engine(segment_length=0.5)
    direct = solve_image_2layer(_rod_world(soil), engine)
    assert direct.metadata["kernel"] == "line"
    via_engine = _solve(_rod_world(soil), engine)
    assert direct.cluster_impedance("g") == via_engine.cluster_impedance("g")
    w = _rod_world(soil)
    w.sources.clear()
    Z = gf.solve_mutual_matrix(w, engine, ["g"])
    assert Z[0, 0] == pytest.approx(via_engine.cluster_impedance("g")[0], rel=1e-12)


@_FINDING
def test_direct_solve_image_uses_the_default_kernel() -> None:
    from groundfield.solver.image import solve_image

    soil = gf.HomogeneousSoil(rho=100.0)
    engine = gf.Engine(segment_length=0.5)
    direct = solve_image(_rod_world(soil), engine)
    assert direct.metadata["kernel"] == "line"
    via_engine = _solve(_rod_world(soil), engine)
    assert direct.cluster_impedance("g") == via_engine.cluster_impedance("g")
