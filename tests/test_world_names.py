"""Name checks of ``World.add_*``: O(1) per add, unchanged semantics.

The ``add_*`` methods scanned every list for the new name, so building a
network was quadratic in its size: ``TnNetworkGenerator.build()`` took 13 s
for 1000 houses (6060 conductors), 11 s of them in these scans, once
generated mesh foundations became one electrode per wire line (audit round
3, note of A3-sec-gen). The names now sit in an index next to the world;
these tests pin that the index changes nothing but the cost.
"""

from __future__ import annotations

import copy
import json
import pickle
import time

import pytest

import groundfield as gf
from groundfield import world as world_module

pytestmark = pytest.mark.finding("10.66")


def _world() -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="a", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="b", position=(20.0, 0.0, 0.5), length=2.0)
    gf.create_junction(w, name="j", position=(10.0, 0.0, -0.5))
    gf.create_conductor(w, name="c", start="a", end="b", cross_section=50e-6)
    gf.create_source(w, attached_to="a", name="s")
    return w


def _rod(name: str, x: float = 40.0) -> gf.RodElectrode:
    return gf.RodElectrode(name=name, position=(x, 0.0, 0.5), length=1.0)


@pytest.mark.parametrize(
    ("add", "match"),
    [
        (lambda w: w.add_electrode(_rod("a")), "Electrode name 'a' already taken"),
        (lambda w: w.add_electrode(_rod("j")), "already taken by a junction"),
        (
            lambda w: gf.create_junction(w, name="b", position=(0, 0, -1)),
            "already taken by an electrode",
        ),
        (
            lambda w: gf.create_conductor(w, name="c", start="a", end="b"),
            "Conductor name 'c' already taken",
        ),
        (
            lambda w: gf.create_source(w, attached_to="b", name="s"),
            "Source name 's' already taken",
        ),
    ],
    ids=["electrode", "electrode-vs-junction", "junction", "conductor", "source"],
)
def test_a_name_taken_is_still_rejected_when_added(add, match: str) -> None:  # noqa: ANN001
    w = _world()
    with pytest.raises(ValueError, match=match):
        add(w)


def test_lists_edited_by_hand_are_seen_by_the_next_add() -> None:
    """An append or a new list changes the list's identity or length."""
    w = _world()
    w.electrodes.append(_rod("x"))
    with pytest.raises(ValueError, match="'x' already taken"):
        w.add_electrode(_rod("x", 60.0))
    w.electrodes = [_rod("y")]
    w.add_electrode(_rod("a", 80.0))  # 'a' is gone with the old list
    with pytest.raises(ValueError, match="'y' already taken"):
        w.add_electrode(_rod("y", 90.0))
    with pytest.raises(KeyError, match="'b'"):
        gf.create_conductor(w, start="a", end="b")


def test_an_object_renamed_in_place_never_blocks_its_old_name() -> None:
    """A stale entry is confirmed by a scan before it raises."""
    w = _world()
    w.electrodes[1].name = "renamed"
    w.add_electrode(_rod("b", 60.0))  # the old name is free again
    # A duplicate made by a rename in place is reported by check().
    w.electrodes[0].name = "j"
    with pytest.raises(ValueError, match="used more than once"):
        w.check()


def test_the_index_is_no_state_of_the_world() -> None:
    """Equality, copies, pickles and JSON do not see it."""
    w = _world()
    loaded = gf.World.model_validate(json.loads(w.model_dump_json()))
    assert loaded == w
    assert pickle.loads(pickle.dumps(w)) == w
    deep = w.model_copy(deep=True)
    deep.add_electrode(_rod("z"))
    assert [e.name for e in w.electrodes] == ["a", "b"]
    with pytest.raises(ValueError, match="'z' already taken"):
        deep.add_electrode(_rod("z", 60.0))
    shallow = copy.copy(w)  # shares the lists: one namespace
    shallow.add_electrode(_rod("q"))
    with pytest.raises(ValueError, match="'q' already taken"):
        w.add_electrode(_rod("q", 60.0))


def test_each_add_costs_constant_name_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """The name set of a list is built once, then updated per add."""
    builds = []
    original = world_module._NameIndex._names

    def counting(self, world, field):  # noqa: ANN001, ANN202
        entry = self._lists.get(field)
        names = original(self, world, field)
        if entry is None or entry[2] is not names:
            builds.append(field)
        return names

    monkeypatch.setattr(world_module._NameIndex, "_names", counting)
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    for k in range(300):
        w.add_electrode(_rod(f"r{k}", 3.0 * k))
    assert builds.count("electrodes") == 1, builds.count("electrodes")
    assert builds.count("junctions") == 1


@pytest.mark.benchmark
def test_adding_names_scales_linearly() -> None:
    """2000 -> 8000 electrodes: about 4 x the time (16 x when quadratic)."""

    def build(n: int) -> float:
        w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
        rods = [_rod(f"r{k}", 3.0 * k) for k in range(n)]
        t0 = time.perf_counter()
        for r in rods:
            w.add_electrode(r)
        return time.perf_counter() - t0

    small, large = (
        min(build(2000) for _ in range(3)),
        min(build(8000) for _ in range(3)),
    )
    assert large / small < 8.0, (small, large)
