"""Backend-specific engine options and inert engine settings.

Finding 10.74: the ``mom_sommerfeld`` accuracy knobs were flat
``Engine`` fields that every other backend accepted silently, the five
``fem`` knobs could not be reached through the public API at all, and
the image-series cap had four different defaults. Backend-specific
options now live in ``Engine.options`` keyed by backend name and are
validated against the backend's registered options model; the flat
``sommerfeld_*`` names are still accepted and mapped.

Finding 10.51: ``Engine.tolerance`` and ``Engine.max_iterations`` are
read by no backend. A non-default value now warns that it has no effect
and that the field is deprecated.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pydantic
import pytest

import groundfield as gf

SEG = 0.5

# ``gf.Engine().model_dump_json()`` of v0.15.0, verbatim.
V0150_ENGINE_JSON = (
    '{"backend":"image","frequencies":[50.0],"segment_length":0.5,'
    '"tolerance":1e-06,"max_iterations":200,"earth_inductive_model":"sommerfeld",'
    '"image_max_terms":300,"image_series_tol":1e-06,'
    '"sommerfeld_lambda_max_factor":200.0,"sommerfeld_epsabs":1e-09,'
    '"sommerfeld_epsrel":1e-07,"sommerfeld_max_osc_panels":200000}'
)


def _rod_world(soil=None) -> gf.World:
    w = gf.create_world(
        soil=soil if soil is not None else gf.HomogeneousSoil(resistivity=100.0)
    )
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="g1")
    return w


def _solve(engine: gf.Engine, world: gf.World | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.solve(world if world is not None else _rod_world())


# ---------------------------------------------------------------------
# fem knobs
# ---------------------------------------------------------------------


@pytest.mark.finding("10.74")
def test_fem_knobs_are_reachable_through_engine_options() -> None:
    engine = gf.Engine(
        backend="fem",
        segment_length=SEG,
        options={"fem": {"refinement": 1, "extrapolate": False}},
    )
    res = _solve(engine)
    assert res.metadata["fem_refinement"] == 1
    assert res.metadata["fem_extrapolated"] is False
    assert [m["level"] for m in res.metadata["fem_mesh"]] == [1]


@pytest.mark.finding("10.74")
def test_fem_far_field_option_reaches_the_solver() -> None:
    robin = _solve(gf.Engine(backend="fem", segment_length=SEG))
    dirichlet = _solve(
        gf.Engine(
            backend="fem",
            segment_length=SEG,
            options={"fem": {"far_field": "dirichlet", "r_far_factor": 10.0}},
        )
    )
    assert dirichlet.metadata["far_field"] == "dirichlet"
    z_r = robin.cluster_impedance("g1")[0].real
    z_d = dirichlet.cluster_impedance("g1")[0].real
    # A grounded truncation box biases the resistance low.
    assert z_d < z_r


def test_fem_default_options_reproduce_the_direct_call() -> None:
    from groundfield.solver.fem import solve_fem

    engine = gf.Engine(backend="fem", segment_length=SEG)
    via_engine = _solve(engine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = solve_fem(_rod_world(), engine)
    assert via_engine.electrode_potentials == direct.electrode_potentials


# ---------------------------------------------------------------------
# mom_sommerfeld knobs: nested and flat
# ---------------------------------------------------------------------


@pytest.mark.finding("10.74")
def test_nested_sommerfeld_options_reach_the_backend() -> None:
    from groundfield.solver.registry import MomSommerfeldOptions

    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
    engine = gf.Engine(
        backend="mom_sommerfeld",
        segment_length=SEG,
        options={
            "mom_sommerfeld": MomSommerfeldOptions(
                lambda_max_factor=123.0,
                epsabs=2e-10,
                epsrel=3e-8,
                max_osc_panels=54_321,
            )
        },
    )
    md = _solve(engine, _rod_world(soil)).metadata
    assert md["lambda_max_factor"] == 123.0
    assert md["epsabs"] == 2e-10
    assert md["epsrel"] == 3e-8
    assert md["max_osc_panels"] == 54_321


@pytest.mark.finding("10.74")
def test_flat_sommerfeld_fields_are_mapped_and_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="sommerfeld_epsrel"):
        flat = gf.Engine(backend="mom_sommerfeld", sommerfeld_epsrel=3e-8)
    nested = gf.Engine(
        backend="mom_sommerfeld", options={"mom_sommerfeld": {"epsrel": 3e-8}}
    )
    assert flat == nested
    assert flat.backend_options().epsrel == 3e-8
    with pytest.warns(DeprecationWarning, match="options"):
        assert flat.sommerfeld_epsrel == 3e-8


@pytest.mark.finding("10.74")
def test_options_for_another_backend_warn() -> None:
    from groundfield.solver.registry import BackendOptionsWarning

    with pytest.warns(BackendOptionsWarning, match="'fem'.*'image'"):
        gf.Engine(backend="image", options={"fem": {"refinement": 1}})
    with pytest.warns(BackendOptionsWarning, match="mom_sommerfeld"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gf.Engine(backend="image", sommerfeld_epsrel=1e-12)


def test_options_for_the_selected_backend_are_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gf.Engine(backend="fem", options={"fem": {"refinement": 1}})
        gf.Engine(backend="mom_sommerfeld", options={"mom_sommerfeld": {}})


@pytest.mark.finding("10.74")
def test_misspelled_option_is_rejected_with_a_suggestion() -> None:
    with pytest.raises(pydantic.ValidationError, match="did you mean 'refinement'"):
        gf.Engine(backend="fem", options={"fem": {"refinment": 1}})
    with pytest.raises(pydantic.ValidationError, match="Did you mean 'fem'"):
        gf.Engine(backend="fem", options={"fme": {}})


@pytest.mark.finding("10.74")
def test_series_controls_are_common_fields_not_backend_options() -> None:
    with pytest.raises(pydantic.ValidationError, match="image_max_terms"):
        gf.Engine(options={"image_2layer": {"max_terms": 10}})


def test_invalid_option_values_are_rejected() -> None:
    for options in (
        {"fem": {"refinement": -1}},
        {"fem": {"far_field": "sommerfeld"}},
        {"mom_sommerfeld": {"epsrel": 0.0}},
    ):
        with pytest.raises(pydantic.ValidationError):
            gf.Engine(options=options)


def test_engine_json_round_trip_keeps_typed_options() -> None:
    from groundfield.solver.registry import FemOptions

    engine = gf.Engine(
        backend="fem", options={"fem": {"refinement": 1, "far_field": "dirichlet"}}
    )
    payload = json.loads(engine.model_dump_json())
    assert payload["options"]["fem"]["refinement"] == 1
    back = gf.Engine.model_validate_json(engine.model_dump_json())
    assert back == engine
    assert isinstance(back.options["fem"], FemOptions)


def test_v0150_engine_json_loads_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        engine = gf.Engine.model_validate_json(V0150_ENGINE_JSON)
    assert engine.options == {}
    assert engine.image_max_terms == 300
    assert engine == gf.Engine()


def test_v0150_engine_json_with_custom_sommerfeld_values_is_mapped() -> None:
    payload = json.loads(V0150_ENGINE_JSON)
    payload.update(backend="mom_sommerfeld", sommerfeld_max_osc_panels=2_000_000)
    with pytest.warns(DeprecationWarning, match="sommerfeld_max_osc_panels"):
        engine = gf.Engine.model_validate(payload)
    assert engine.backend_options("mom_sommerfeld").max_osc_panels == 2_000_000


# ---------------------------------------------------------------------
# Inert settings (10.51)
# ---------------------------------------------------------------------


@pytest.mark.finding("10.51")
def test_tolerance_and_max_iterations_warn_that_they_have_no_effect() -> None:
    with pytest.warns(DeprecationWarning, match=r"tolerance.*no effect.*2027\.5"):
        gf.Engine(tolerance=1e-3)
    with pytest.warns(DeprecationWarning, match=r"max_iterations.*no effect"):
        gf.Engine(max_iterations=10)
    with pytest.warns(DeprecationWarning, match="tolerance"):
        gf.Engine().tolerance  # noqa: B018 - attribute access is the test


def test_default_engine_construction_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gf.Engine()
        gf.create_engine()
        gf.create_engine(backend="mom", frequencies=[50.0, 1000.0])
        gf.Engine().with_frequencies(50.0, 150.0)


# ---------------------------------------------------------------------
# Series controls: one default, recorded on every result
# ---------------------------------------------------------------------


@pytest.mark.finding("10.74")
def test_series_default_is_one_documented_value() -> None:
    from groundfield.solver.registry import (
        DEFAULT_IMAGE_MAX_TERMS,
        DEFAULT_IMAGE_SERIES_TOL,
    )

    assert DEFAULT_IMAGE_MAX_TERMS == 300
    assert DEFAULT_IMAGE_SERIES_TOL == 1e-6
    assert gf.Engine().image_max_terms == DEFAULT_IMAGE_MAX_TERMS
    assert gf.create_engine().image_max_terms == DEFAULT_IMAGE_MAX_TERMS
    assert gf.Engine().image_series_tol == DEFAULT_IMAGE_SERIES_TOL


@pytest.mark.finding("10.74")
@pytest.mark.parametrize(
    "soil",
    [
        gf.HomogeneousSoil(resistivity=100.0),
        gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0),
    ],
)
def test_series_controls_are_recorded_on_every_result(soil) -> None:
    engine = gf.Engine(segment_length=SEG, image_max_terms=123, image_series_tol=1e-8)
    res = _solve(engine, _rod_world(soil))
    assert res.metadata["series"] == {"max_terms": 123, "tol": 1e-8}


def test_backend_options_accessor_returns_defaults() -> None:
    from groundfield.solver.registry import FemOptions, MomSommerfeldOptions

    engine = gf.Engine(backend="fem")
    assert engine.backend_options() == FemOptions()
    assert engine.backend_options("mom_sommerfeld") == MomSommerfeldOptions()
    assert gf.Engine(backend="image").backend_options() is None
    assert np.isclose(FemOptions().r_far_factor, 1000.0)
