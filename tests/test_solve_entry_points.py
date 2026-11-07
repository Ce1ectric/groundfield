"""One source-snapshot semantics for every solve entry point (10.72).

``World.solve``, ``Engine.solve`` and ``gf.run_simulation`` used to
differ: ``World.solve`` (and ``run_simulation``, which calls it) deep
-copied the sources and rolled back any in-flight mutation, while
``Engine.solve`` let a backend rewrite the caller's world. Every entry
point now restores the sources onto the caller's objects, keeps the
list object, and records the sources as the caller built them.
"""

from __future__ import annotations

import warnings

import pytest

import groundfield as gf
import groundfield.solver.image as image_mod


def _rod_world() -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


@pytest.fixture
def mutating_backend(monkeypatch):
    """Make the image backend rewrite and extend the source list in flight."""
    original = image_mod.solve_image

    def mutating(world, engine, **kw):
        world.sources[0].magnitude = 999.0
        extra = world.sources[0].model_copy(deep=True)
        extra.name = "injected_by_backend"
        world.sources.append(extra)
        world.sources = list(world.sources)  # rebind the attribute as well
        return original(world, engine, **kw)

    monkeypatch.setattr(image_mod, "solve_image", mutating)


def _run(entry: str, world: gf.World, engine: gf.Engine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if entry == "world.solve":
            return world.solve(engine)
        if entry == "engine.solve":
            return engine.solve(world)
        return gf.run_simulation(world, engine)


@pytest.mark.parametrize(
    "entry",
    [
        "world.solve",
        pytest.param("engine.solve", marks=pytest.mark.finding("10.72")),
        "run_simulation",
    ],
)
def test_every_entry_point_restores_the_sources(mutating_backend, entry) -> None:
    world = _rod_world()
    handle = world.sources[0]
    source_list = world.sources
    res = _run(entry, world, gf.create_engine(segment_length=0.5))

    assert world.sources is source_list
    assert len(world.sources) == 1
    assert world.sources[0] is handle
    assert handle.magnitude == 1.0
    # The record describes the world as the caller built it.
    assert [s.name for s in res.sources] == [handle.name]
    assert res.sources[0].current == [1.0 + 0j]


@pytest.mark.finding("10.72")
def test_snapshot_opt_out_is_deprecated() -> None:
    world = _rod_world()
    with pytest.warns(DeprecationWarning, match="snapshot_sources"):
        world.solve(gf.create_engine(segment_length=0.5), snapshot_sources=False)


def test_mutating_the_handle_between_solves_still_takes_effect() -> None:
    world = _rod_world()
    engine = gf.create_engine(segment_length=0.5)
    phi_1 = engine.solve(world).electrode_potentials["g1"][0]
    world.sources[0].magnitude = 3.0
    phi_3 = engine.solve(world).electrode_potentials["g1"][0]
    assert phi_3 == pytest.approx(3.0 * phi_1, rel=1e-12)
