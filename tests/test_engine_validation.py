"""Validation of engine copies and assignments (audit A1-02, A1-04).

pydantic validates neither ``model_copy(update=...)`` nor attribute
assignment by default. On the 2026.11 base this meant:

* ``engine.model_copy(update={"sommerfeld_epsrel": 1e-2})`` stored the
  value under the name of the deprecated property, which shadows it: the
  copy ran with the default ``1e-7`` and nothing warned (A1-02; v0.15.0,
  where the name was a field, ran with ``1e-2``);
* ``engine.options = {"fem": {...}}`` stored a raw dict and the solve died
  with an ``AttributeError`` (a misspelled option was never named);
  ``engine.frequencies = []`` gave an empty result and ``[nan]`` solved
  (A1-04, the engine part of 10.67).
"""

from __future__ import annotations

import math
import warnings

import pytest

import groundfield as gf
from groundfield.solver.engine import EngineFrequencyOrderWarning
from groundfield.solver.registry import (
    BackendOptionsWarning,
    FemOptions,
    MomSommerfeldOptions,
)

FLAT = [
    ("sommerfeld_epsrel", "epsrel", 1e-2),
    ("sommerfeld_epsabs", "epsabs", 1e-2),
    ("sommerfeld_max_osc_panels", "max_osc_panels", 1000),
    ("sommerfeld_lambda_max_factor", "lambda_max_factor", 50.0),
]


def _rod_world(soil=None) -> gf.World:
    w = gf.create_world(
        soil=soil if soil is not None else gf.HomogeneousSoil(rho=100.0)
    )
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="g1")
    return w


def _solve(engine: gf.Engine, world: gf.World | None = None) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (world if world is not None else _rod_world()).solve(engine)


def _deprecations(caught) -> list[str]:
    return [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]


# ---------------------------------------------------------------------
# model_copy (A1-02)
# ---------------------------------------------------------------------


@pytest.mark.finding("A1-02")
@pytest.mark.parametrize("flat, option, value", FLAT, ids=[f[0] for f in FLAT])
def test_model_copy_maps_a_deprecated_flat_name_onto_the_options(
    flat, option, value
) -> None:
    engine = gf.Engine(backend="mom_sommerfeld", segment_length=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        copy = engine.model_copy(update={flat: value})
    assert getattr(copy.backend_options("mom_sommerfeld"), option) == value
    assert isinstance(copy.options["mom_sommerfeld"], MomSommerfeldOptions)
    messages = _deprecations(caught)
    assert len(messages) == 1
    assert flat in messages[0] and "options" in messages[0] and "2027.5" in messages[0]
    # the original is untouched
    assert getattr(engine.backend_options("mom_sommerfeld"), option) != value


@pytest.mark.finding("A1-02")
def test_model_copy_with_a_flat_name_solves_with_the_copied_accuracy() -> None:
    engine = gf.Engine(backend="mom_sommerfeld", segment_length=0.5)
    with pytest.warns(DeprecationWarning, match="sommerfeld_epsrel"):
        copy = engine.model_copy(update={"sommerfeld_epsrel": 1e-2})
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=2.0)
    assert _solve(copy, _rod_world(soil)).metadata["epsrel"] == 1e-2


@pytest.mark.finding("A1-02")
def test_model_copy_can_set_a_flat_name_back_to_its_default() -> None:
    """The constructor drops flat values equal to the default (0.15.0 JSON);
    a copy must apply them, or a non-default option could not be undone."""
    engine = gf.Engine(
        backend="mom_sommerfeld", options={"mom_sommerfeld": {"epsrel": 1e-2}}
    )
    with pytest.warns(DeprecationWarning):
        copy = engine.model_copy(update={"sommerfeld_epsrel": 1e-7})
    assert copy.backend_options().epsrel == 1e-7
    assert engine.backend_options().epsrel == 1e-2


@pytest.mark.finding("A1-02")
def test_model_copy_merges_flat_names_with_new_options() -> None:
    engine = gf.Engine(backend="mom_sommerfeld")
    with pytest.warns(DeprecationWarning):
        copy = engine.model_copy(
            update={
                "sommerfeld_epsrel": 1e-3,
                "options": {"mom_sommerfeld": {"epsabs": 1e-6}},
            }
        )
    assert (copy.backend_options().epsrel, copy.backend_options().epsabs) == (
        1e-3,
        1e-6,
    )
    with pytest.raises(ValueError, match="epsrel"), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        engine.model_copy(
            update={
                "sommerfeld_epsrel": 1e-3,
                "options": {"mom_sommerfeld": {"epsrel": 1e-4}},
            }
        )


@pytest.mark.finding("A1-02")
def test_model_copy_rejects_an_unknown_field_with_a_suggestion() -> None:
    with pytest.raises(ValueError, match="segment_length"):
        gf.Engine().model_copy(update={"segment_lenght": 1.0})


@pytest.mark.finding("A1-02", "A1-04")
@pytest.mark.parametrize(
    "update",
    [
        {"frequencies": []},
        {"frequencies": [math.nan]},
        {"frequencies": [-50.0]},
        {"segment_length": 0.0},
        {"kernel": "lin"},
        {"backend": "imagee"},
        {"earth_inductive_model": "carson"},
        {"options": {"fem": {"n_radail": 61}}},
        {"options": {"fem": 3}},
    ],
    ids=[
        "empty_frequencies",
        "nan_frequency",
        "negative_frequency",
        "zero_segment",
        "kernel",
        "backend",
        "earth_model",
        "misspelled_option",
        "option_not_a_mapping",
    ],
)
def test_model_copy_validates_the_update(update) -> None:
    engine = gf.Engine(backend="fem")
    with pytest.raises(ValueError):
        engine.model_copy(update=update)


@pytest.mark.finding("A1-02", "A1-04")
def test_model_copy_validates_option_mappings_into_models() -> None:
    copy = gf.Engine(backend="fem").model_copy(
        update={"options": {"fem": {"n_radial": 61}}}
    )
    assert isinstance(copy.options["fem"], FemOptions)
    assert _solve(copy).metadata["n_radial"] == 61


def test_model_copy_repeats_no_warning_of_an_unchanged_field() -> None:
    with pytest.warns(EngineFrequencyOrderWarning):
        engine = gf.Engine(frequencies=[1000.0, 50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        copy = engine.model_copy(update={"segment_length": 1.0})
    assert copy.frequencies == [1000.0, 50.0] and copy.segment_length == 1.0


def test_model_copy_warns_about_what_the_update_introduces() -> None:
    engine = gf.Engine(backend="fem", options={"fem": {"n_radial": 61}})
    with pytest.warns(BackendOptionsWarning, match="fem"):
        engine.model_copy(update={"backend": "image"})


def test_model_copy_keeps_pydantic_copy_semantics() -> None:
    engine = gf.Engine(frequencies=[50.0, 150.0])
    shallow = engine.model_copy()
    assert shallow == engine and shallow is not engine
    deep = engine.model_copy(update={"segment_length": 2.0}, deep=True)
    assert deep.frequencies == engine.frequencies
    assert deep.frequencies is not engine.frequencies
    assert deep.model_fields_set == engine.model_fields_set | {"segment_length"}
    assert engine.segment_length == 0.5


# ---------------------------------------------------------------------
# Assignment (A1-04)
# ---------------------------------------------------------------------


@pytest.mark.finding("A1-04")
def test_assigned_options_are_validated() -> None:
    engine = gf.Engine(backend="fem")
    engine.options = {"fem": {"n_radial": 61}}
    assert isinstance(engine.options["fem"], FemOptions)
    assert _solve(engine).metadata["n_radial"] == 61


@pytest.mark.finding("A1-04")
def test_an_assigned_misspelled_option_is_named() -> None:
    engine = gf.Engine(backend="fem")
    with pytest.raises(ValueError, match="did you mean 'n_radial'"):
        engine.options = {"fem": {"n_radail": 61}}


@pytest.mark.finding("A1-04")
def test_options_mutated_in_place_are_validated_when_read() -> None:
    engine = gf.Engine(backend="fem")
    engine.options["fem"] = {"n_radial": 61}  # bypasses assignment validation
    assert engine.backend_options().n_radial == 61
    assert _solve(engine).metadata["n_radial"] == 61

    engine.options["fem"] = {"n_radail": 61}
    with pytest.raises(ValueError, match="did you mean 'n_radial'"):
        engine.backend_options()
    with pytest.raises(ValueError, match="n_radial"):
        _solve(engine)


@pytest.mark.finding("A1-04")
def test_options_mutated_for_an_unknown_backend_raise_at_solve() -> None:
    engine = gf.Engine(backend="fem")
    engine.options["femm"] = {"n_radial": 61}
    with pytest.raises(ValueError, match="fem"):
        _solve(engine)


@pytest.mark.finding("A1-04", "10.67")
@pytest.mark.parametrize(
    "value",
    [[], [math.nan], [math.inf], [-1.0]],
    ids=["empty", "nan", "inf", "negative"],
)
def test_frequencies_are_validated_on_assignment(value) -> None:
    engine = gf.Engine()
    with pytest.raises(ValueError, match="frequencies"):
        engine.frequencies = value
    assert engine.frequencies == [50.0]


@pytest.mark.finding("A1-04", "10.67")
def test_frequencies_mutated_in_place_are_checked_at_solve() -> None:
    engine = gf.Engine()
    engine.frequencies.clear()
    with pytest.raises(ValueError, match="must not be empty"):
        _solve(engine)
    engine.frequencies.append(math.nan)
    with pytest.raises(ValueError, match="NaN"):
        _solve(engine)


@pytest.mark.finding("A1-04")
@pytest.mark.parametrize(
    "name, value",
    [
        ("segment_length", -1.0),
        ("kernel", "lin"),
        ("backend", "imagee"),
        ("earth_inductive_model", "carson"),
        ("image_max_terms", 0),
        ("sommerfeld_assembly", "table"),
    ],
)
def test_every_field_is_validated_on_assignment(name, value) -> None:
    engine = gf.Engine()
    before = engine.model_dump()
    with pytest.raises(ValueError):
        setattr(engine, name, value)
    assert engine.model_dump() == before


@pytest.mark.finding("A1-04")
def test_an_assigned_backend_warns_about_options_it_ignores() -> None:
    engine = gf.Engine(backend="fem", options={"fem": {"n_radial": 61}})
    with pytest.warns(BackendOptionsWarning, match="fem"):
        engine.backend = "image"


def test_the_deprecated_flat_property_still_sets_the_option() -> None:
    engine = gf.Engine(backend="mom_sommerfeld")
    with pytest.warns(DeprecationWarning, match="sommerfeld_epsrel"):
        engine.sommerfeld_epsrel = 1e-3
    assert engine.backend_options().epsrel == 1e-3
    with pytest.warns(DeprecationWarning):
        assert engine.sommerfeld_epsrel == 1e-3
