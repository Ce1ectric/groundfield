"""Source and conductor references are validated (findings 10.66, 10.51).

Up to 0.15.0 a misspelled ``attached_to`` / ``return_to``, a source on a
conductor, a world without a source, a ``magnitude`` of NaN or zero, a
``VoltageSource`` and a conductor end naming no electrode all solved
silently: zero injection, ``nan`` impedances, or a conductor that simply
vanished from the network. Every one of them now raises at the line
that introduced it (factory / ``add_*``) or, for worlds edited later
or loaded from JSON, when the world is solved, naming the closest
valid name.
"""

from __future__ import annotations

import math
import warnings

import pytest

import groundfield as gf
from groundfield.conductors.conductor import Conductor
from groundfield.sources import CurrentSource, VoltageSource

pytestmark = pytest.mark.finding("10.66")


def _soil(rho: float = 100.0) -> gf.HomogeneousSoil:
    """Homogeneous soil; ``resistivity`` was the field name up to 0.15.0."""
    try:
        return gf.HomogeneousSoil(rho=rho)
    except Exception:  # noqa: BLE001 - v0.15.0 names the field ``resistivity``
        return gf.HomogeneousSoil(resistivity=rho)


def _world() -> gf.World:
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="g2", position=(10.0, 0.0, 0.5), length=2.0)
    return w


def _engine(backend: str = "image") -> gf.Engine:
    return gf.create_engine(backend=backend, segment_length=0.5, frequencies=[50.0])


def _solve(world: gf.World, backend: str = "image") -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world.solve(_engine(backend))


# ---------------------------------------------------------------------
# misspelled node names
# ---------------------------------------------------------------------


def test_unknown_attached_to_raises_with_suggestion_at_creation() -> None:
    w = _world()
    with pytest.raises(KeyError, match=r"'gl'.*Did you mean 'g1'"):
        gf.create_source(w, attached_to="gl", magnitude=1.0)
    assert w.sources == []


def test_unknown_return_to_raises_with_suggestion_at_creation() -> None:
    w = _world()
    with pytest.raises(KeyError, match=r"'g3'.*Did you mean"):
        gf.create_source(w, attached_to="g1", return_to="g3", magnitude=1.0)
    assert w.sources == []


@pytest.mark.parametrize("entry", ["world", "engine"])
def test_reference_broken_after_creation_raises_when_solved(entry: str) -> None:
    w = _world()
    src = gf.create_source(w, attached_to="g1", magnitude=1.0)
    src.attached_to = "gl"  # edited after creation: caught by the solve
    with pytest.raises(KeyError, match=r"'gl'.*Did you mean 'g1'"):
        if entry == "world":
            w.solve(_engine())
        else:
            _engine().solve(w)


def test_world_loaded_from_json_is_validated_when_solved() -> None:
    w = _world()
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    data = w.model_dump(mode="json")
    data["sources"][0]["return_to"] = "aux"  # no such electrode
    loaded = gf.World.model_validate(data)
    with pytest.raises(KeyError, match=r"return_to='aux'"):
        loaded.solve(_engine())


def test_validation_also_guards_the_mom_family() -> None:
    w = _world()
    src = gf.create_source(w, attached_to="g1", magnitude=1.0)
    src.attached_to = "G1"
    with pytest.raises(KeyError, match="Did you mean 'g1'"):
        _solve(w, backend="mom")


# ---------------------------------------------------------------------
# sources on conductors, on their own node, short-circuited
# ---------------------------------------------------------------------


def test_source_on_a_conductor_is_rejected_with_a_junction_hint() -> None:
    w = _world()
    gf.create_conductor(w, name="c1", start="g1", end="g2", cross_section=50e-6)
    with pytest.raises(ValueError, match=r"(?s)conductor.*junction"):
        gf.create_source(w, attached_to="c1", magnitude=1.0)


def test_source_returning_to_its_own_node_is_rejected() -> None:
    w = _world()
    with pytest.raises(ValueError, match="same node"):
        gf.create_source(w, attached_to="g1", return_to="g1", magnitude=1.0)


@pytest.mark.finding("10.16")
def test_source_short_circuited_by_an_ideal_bond_raises() -> None:
    w = _world()
    gf.create_conductor(w, name="bond", start="g1", end="g2")  # ideal
    gf.create_source(w, attached_to="g1", return_to="g2", magnitude=1.0)
    with pytest.raises(ValueError, match="short-circuit"):
        w.solve(_engine())


# ---------------------------------------------------------------------
# missing / invalid excitation
# ---------------------------------------------------------------------


def test_world_without_a_current_source_raises() -> None:
    w = _world()
    with pytest.raises(ValueError, match="no current source"):
        w.solve(_engine())


def test_zero_magnitude_source_raises_when_solved() -> None:
    w = _world()
    gf.create_source(w, attached_to="g1", magnitude=0.0)
    with pytest.raises(ValueError, match="magnitude 0"):
        w.solve(_engine())


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_non_finite_magnitude_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="magnitude"):
        CurrentSource(name="s", attached_to="g1", magnitude=value)
    src = CurrentSource(name="s", attached_to="g1", magnitude=1.0)
    with pytest.raises(ValueError, match="magnitude"):
        src.magnitude = value


def test_non_finite_phase_is_rejected() -> None:
    with pytest.raises(ValueError, match="phase_deg"):
        CurrentSource(name="s", attached_to="g1", magnitude=1.0, phase_deg=math.nan)


@pytest.mark.finding("10.51")
def test_voltage_source_solves_and_fixes_the_feeding_potential() -> None:
    """Since 2026.11 a voltage source is solved (it raised before; up to
    0.15.0 it solved to zero potentials). ``tests/test_voltage_sources.py``
    validates it against scaled current-source solves and closed forms."""
    w = _world()
    w.add_source(VoltageSource(name="v1", attached_to="g1", magnitude=230.0))
    res = _solve(w)
    assert abs(res.electrode_potentials["g1"][0] - 230.0) < 1e-9
    assert res.sources[0].current[0].real > 0.0


# ---------------------------------------------------------------------
# conductor ends
# ---------------------------------------------------------------------


def _cond(start: str | None, end: str | None) -> Conductor:
    return Conductor(
        name="lead",
        start=(0.0, 0.0, 0.5),
        end=(10.0, 0.0, 0.5),
        start_electrode=start,
        end_electrode=end,
        cross_section=50e-6,
    )


def test_unknown_conductor_end_raises_with_suggestion() -> None:
    w = _world()
    with pytest.raises(KeyError, match=r"end_electrode='g3'.*Did you mean"):
        w.add_conductor(_cond("g1", "g3"))
    assert w.conductors == []


def test_conductor_end_broken_after_creation_raises_when_solved() -> None:
    w = _world()
    lead = w.add_conductor(_cond("g1", "g2"))
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    lead.end_electrode = "g_2"
    with pytest.raises(KeyError, match=r"end_electrode='g_2'.*Did you mean 'g2'"):
        w.solve(_engine())


def test_valid_world_still_solves_unchanged() -> None:
    """Control: the validation does not touch a well-formed world."""
    w = _world()
    gf.create_conductor(w, name="pen", start="g1", end="g2", cross_section=50e-6)
    gf.create_source(w, attached_to="g1", magnitude=2.0)
    res = _solve(w)
    total = sum(v[0] for v in res.electrode_currents.values())
    assert abs(total - 2.0) < 1e-12


@pytest.mark.finding("10.66")
@pytest.mark.parametrize("backend", ["image", "mom", "mom_sommerfeld", "fem"])
@pytest.mark.parametrize("entry", ["engine", "world"])
def test_a_world_without_a_source_raises_in_every_backend(
    backend: str, entry: str
) -> None:
    """One rule for every backend and entry point.

    Up to audit round 2 (A2S-05) Engine.solve with fem returned zeros while
    every other backend, and World.solve with fem, raised; the changelog
    claimed the world stayed solvable through Engine.solve.
    """
    # audit A2S-05
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=2.0))
    gf.create_electrode(w, "rod", name="r", position=(0.0, 0.0, 0.5), length=1.0)
    engine = gf.Engine(backend=backend, segment_length=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no current source"):
            engine.solve(w) if entry == "engine" else w.solve(engine)


def _payload() -> dict:
    import json

    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="a", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="b", position=(50.0, 0.0, 0.5), length=3.0)
    gf.create_conductor(w, start="a", end="b", name="c1", cross_section=50e-6)
    gf.create_conductor(w, start="a", end="b", name="c2", cross_section=50e-6)
    gf.create_source(w, attached_to="a")
    gf.create_source(w, attached_to="b", name="s2")
    return json.loads(w.model_dump_json())


@pytest.mark.finding("10.66")
@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (
            lambda d: d["electrodes"][1].update(name="a"),
            "'a' (electrode and electrode)",
        ),
        (
            lambda d: d.update(
                junctions=[
                    {"name": "b", "kind": "junction", "position": [0.0, 0.0, -1.0]}
                ]
            ),
            "'b' (electrode and junction)",
        ),
        (lambda d: d["conductors"][1].update(name="c1"), "conductor name(s)"),
        (
            lambda d: d["sources"][1].update(name=d["sources"][0]["name"]),
            "source name(s)",
        ),
    ],
    ids=["two-electrodes", "electrode-and-junction", "two-conductors", "two-sources"],
)
def test_check_reports_names_used_twice(edit, needle: str) -> None:
    """Uniqueness was enforced by the add_* methods only.

    A world loaded from JSON with two electrodes of one name passed check()
    and failed in the solver with a singular matrix; an electrode and a
    junction of one name were silently merged into one node.
    """
    # audit A2S-06
    data = _payload()
    edit(data)
    world = gf.World.model_validate(data)
    with pytest.raises(ValueError, match="used more than once") as info:
        world.check()
    assert needle in str(info.value)
    with pytest.raises(ValueError, match="used more than once"):
        world.solve(gf.Engine(segment_length=0.5))


# ---------------------------------------------------------------------
# Audit A3S-03: names with the solver's reserved prefixes
# ---------------------------------------------------------------------
#
# The solver names the part of an electrode in one soil layer
# ``__layer_<electrode>__L<k>`` and the nodes of a distributed conductor
# ``__cond_<conductor>__...``. An electrode called ``__layer_r__L1`` next to
# a rod ``r`` through h_1, or ``__cond_pen__seg_0`` next to a galvanic
# conductor ``pen``, passed World.check() and the solve died with
# numpy.linalg.LinAlgError (singular matrix): the internal owner took the
# user electrode's place.


def _layer_clash_payload() -> dict:
    import json

    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0))
    gf.create_electrode(w, "rod", name="r", position=(0, 0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="x", position=(30, 0, 0.5), length=1.0)
    gf.create_source(w, attached_to="r")
    data = json.loads(w.model_dump_json())
    data["electrodes"][1]["name"] = "__layer_r__L1"
    return data


def _conductor_clash_payload() -> dict:
    import json

    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="a", position=(0, 0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="b", position=(30, 0, 0.5), length=2.0)
    gf.create_conductor(
        w,
        name="pen",
        start="a",
        end="b",
        cross_section=50e-6,
        discretize_segment_length=10.0,
        coupling_to_soil="galvanic",
    )
    gf.create_electrode(w, "rod", name="x", position=(0, 40, 0.5), length=1.0)
    gf.create_source(w, attached_to="a")
    data = json.loads(w.model_dump_json())
    data["electrodes"][2]["name"] = "__cond_pen__seg_0"
    return data


@pytest.mark.finding("10.66")
@pytest.mark.parametrize(
    ("payload", "prefix"),
    [(_layer_clash_payload, "__layer_"), (_conductor_clash_payload, "__cond_")],
    ids=["layer", "cond"],
)
def test_check_rejects_a_name_with_a_reserved_prefix(payload, prefix: str) -> None:  # noqa: ANN001
    # audit A3S-03
    world = gf.World.model_validate(payload())
    with pytest.raises(ValueError, match="reserved prefix") as info:
        world.check()
    assert repr(prefix) in str(info.value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="reserved prefix"):
            world.solve(gf.Engine(segment_length=0.5))


@pytest.mark.finding("10.66")
@pytest.mark.parametrize("name", ["__layer_r__L1", "__cond_pen__seg_0", "__cond_x"])
def test_reserved_prefixes_are_rejected_when_a_node_is_added(name: str) -> None:
    # audit A3S-03
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    with pytest.raises(ValueError, match="reserved prefix"):
        gf.create_electrode(w, "rod", name=name, position=(0, 0, 0.5), length=1.0)
    with pytest.raises(ValueError, match="reserved prefix"):
        gf.create_junction(w, name=name, position=(0.0, 0.0, -1.0))
    assert not w.electrodes and not w.junctions


@pytest.mark.finding("10.66")
def test_names_that_only_resemble_the_prefixes_stay_legal() -> None:
    # audit A3S-03: only the two exact prefixes are reserved
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    for name in ("_layer_r", "__layer", "layer__L1", "__conductor", "x__cond_y"):
        gf.create_electrode(
            w, "rod", name=name, position=(len(w.electrodes) * 20.0, 0, 0.5), length=1.0
        )
    gf.create_source(w, attached_to="_layer_r")
    w.check()
