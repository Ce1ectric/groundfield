"""The lumped concrete-shell registry is deprecated (ADR-0012 V1, 10.12–10.14).

The solver consumes concrete shells only as the distributed per-length
coefficient ``concrete_shell_coefficient_ohm_m`` of the electrodes
(user decision 2026-11-07); ``World.concrete_shell_corrections`` was
never read by any backend nor by ``solve_mutual_*`` (finding 10.12:
AP1 foundations lost -28.6 / -15.2 / -2.7 % of their self-impedance
through it). A solve of a world with registry entries now says so; the
field still loads from JSON.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf

pytestmark = pytest.mark.finding("10.12")


def _world(entry: float | None) -> gf.World:
    try:
        soil = gf.HomogeneousSoil(rho=100.0)
    except Exception:  # noqa: BLE001 - v0.15.0 names the field ``resistivity``
        soil = gf.HomogeneousSoil(resistivity=100.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    if entry is not None:
        w.concrete_shell_corrections["g1"] = entry
    return w


def _engine() -> gf.Engine:
    return gf.create_engine(segment_length=0.5, frequencies=[50.0])


def test_solving_with_registry_entries_warns_and_names_the_coefficient() -> None:
    with pytest.warns(
        DeprecationWarning, match=r"(?s)concrete_shell_coefficient_ohm_m.*2027\.5"
    ):
        _world(0.8).solve(_engine())


def test_registry_entries_do_not_change_the_result() -> None:
    """Not consumed by the solver: identical potentials with and without."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _world(0.8).solve(_engine()).electrode_potentials["g1"]
        b = _world(None).solve(_engine()).electrode_potentials["g1"]
    np.testing.assert_array_equal(a, b)


def test_empty_registry_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _world(None).solve(_engine())


def test_registry_still_round_trips_through_json() -> None:
    w = _world(0.8)
    loaded = gf.World.model_validate_json(w.model_dump_json())
    assert loaded.concrete_shell_corrections == {"g1": 0.8}
    assert loaded.reset_concrete_corrections() == {"g1": 0.8}
