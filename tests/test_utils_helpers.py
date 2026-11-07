"""Helpers in :mod:`groundfield.utils`: name suggestions and API warnings.

These helpers back the "did you mean …?" errors of the factories
(finding 10.68) and the deprecated-alias / inert-configuration warnings
(10.69, 10.51) of the model layer.
"""

from __future__ import annotations

import warnings

import pytest

from groundfield.utils import (
    REMOVAL_VERSION,
    InertConfigurationWarning,
    did_you_mean,
    suggest,
    warn_deprecated,
    warn_inert,
)


@pytest.mark.finding("10.68")
def test_suggest_ranks_close_matches_and_ignores_case() -> None:
    kinds = ["rod", "ring", "strip", "polyline", "star", "mesh", "grid_mesh"]
    assert suggest("rood", kinds)[0] == "rod"
    assert suggest("Ring", kinds)[0] == "ring"
    assert suggest("gridmesh", kinds)[0] == "grid_mesh"
    assert suggest("zzz", kinds) == []
    assert suggest("anything", []) == []


@pytest.mark.finding("10.68")
def test_did_you_mean_renders_a_sentence() -> None:
    assert did_you_mean("positon", ["position", "length"]) == (
        " Did you mean 'position'?"
    )
    assert did_you_mean("zzz", ["position"]) == ""
    two = did_you_mean("g", ["g1", "g2"], n=2, cutoff=0.5)
    assert two == " Did you mean 'g1' or 'g2'?"


@pytest.mark.finding("10.69")
def test_removal_version_follows_the_deprecation_policy() -> None:
    assert REMOVAL_VERSION == "2027.5"


@pytest.mark.finding("10.69")
def test_warn_deprecated_names_replacement_and_removal() -> None:
    with pytest.warns(DeprecationWarning) as record:
        warn_deprecated("'resistivity'", "'rho'")
    message = str(record[0].message)
    assert "'resistivity'" in message and "'rho'" in message
    assert REMOVAL_VERSION in message
    assert record[0].filename == __file__


@pytest.mark.finding("10.51")
def test_warn_inert_uses_its_own_category() -> None:
    assert issubclass(InertConfigurationWarning, UserWarning)
    with pytest.warns(InertConfigurationWarning, match="no backend reads") as record:
        warn_inert("Engine.foo = 1 is stored, but no backend reads it.")
    assert record[0].filename == __file__


def _declaring_and_internal_modules() -> tuple[dict, dict]:
    """Two fake groundfield modules: one declares deprecated / inert things,
    the other uses them internally."""
    declaring = {
        "__name__": "groundfield._fake_declaring_module",
        "warn_deprecated": warn_deprecated,
        "warn_inert": warn_inert,
    }
    exec(
        "def old_thing():\n"
        "    warn_deprecated(\"'a'\", \"'b'\")\n"
        "def inert_thing():\n"
        "    warn_inert('x is stored, but no backend reads it.')\n",
        declaring,
    )
    internal = {
        "__name__": "groundfield._fake_internal_caller",
        "old_thing": declaring["old_thing"],
        "inert_thing": declaring["inert_thing"],
    }
    exec(
        "def use_internally():\n    old_thing()\n    inert_thing()\n",
        internal,
    )
    return declaring, internal


@pytest.mark.finding("10.69")
def test_helpers_stay_silent_for_groundfield_callers() -> None:
    """A warning caused by groundfield's own code is not actionable."""
    _, internal = _declaring_and_internal_modules()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        internal["use_internally"]()
    assert not caught, [str(w.message) for w in caught]


@pytest.mark.finding("10.69")
def test_helpers_attribute_the_warning_to_the_user_line() -> None:
    declaring, _ = _declaring_and_internal_modules()
    with pytest.warns(DeprecationWarning) as record:
        declaring["old_thing"]()
    assert record[0].filename == __file__
    with pytest.warns(InertConfigurationWarning) as record:
        declaring["inert_thing"]()
    assert record[0].filename == __file__
