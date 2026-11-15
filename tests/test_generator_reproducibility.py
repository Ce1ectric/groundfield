"""Reproducible generator realisations (10.53, pass-9 F07).

* ``GeneratorConfig.sample`` resolves every distribution of the config
  tree — soil, placement, electrode, measurement and cabinet specs, and
  the ``building_counts`` mapping — so the *resolved* config replays the
  realisation.
* ``sample_world(rng)`` draws every random number of the realisation from
  ``rng``; the generator's construction seed has no influence.
* The penetration mask of a layout does not depend on the order of the
  footprints, also when they carry no ``osm_id``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import (
    BuildingTypeSpec,
    CabinetConfig,
    Discrete,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    HomogeneousSoilSpec,
    LvNetworkLayout,
    ManhattanGridPlacement,
    MeasurementInjectionConfig,
    MeasurementSetupConfig,
    RodElectrodeSpec,
    TnNetworkConfig,
    TnNetworkGenerator,
    Uniform,
    buried_lead,
)

pytestmark = pytest.mark.finding("10.53")


def _world_json(world: gf.World) -> str:
    return json.dumps(world.model_dump(mode="json"), sort_keys=True)


def _f07_config() -> TnNetworkConfig:
    """The pass-9 F07 reproduction: distributions in nested specs and a dict."""
    return TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=Uniform(low=50, high=500)),
        placement=ManhattanGridPlacement(spacing_x_m=Uniform(low=20, high=40)),
        building_types=[
            BuildingTypeSpec(
                name="residential",
                grounding=GroundingSystemSpec(
                    electrodes=[FoundationElectrodeSpec(size_m=Uniform(low=8, high=14))]
                ),
            )
        ],
        building_counts={"residential": Discrete(values=[3.0, 9.0])},
    )


def _everything_config() -> TnNetworkConfig:
    return TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=Uniform(low=50, high=500)),
        placement=ManhattanGridPlacement(
            spacing_x_m=Uniform(low=20, high=40), jitter_m=1.0
        ),
        building_types=[
            BuildingTypeSpec(
                name="residential",
                grounding=GroundingSystemSpec(
                    electrodes=[
                        FoundationElectrodeSpec(
                            size_m=Uniform(low=8, high=14),
                            concrete_rho_ohm_m=Uniform(low=50, high=500),
                        ),
                        RodElectrodeSpec(presence_prob=Uniform(low=0.2, high=0.8)),
                    ]
                ),
            )
        ],
        building_counts={"residential": Discrete(values=[3.0, 9.0])},
        cabinets=CabinetConfig(count_per_100_buildings=Uniform(low=5, high=20)),
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                feed_lead=buried_lead(depth_m=0.6).model_copy(
                    update={"depth_m": Uniform(low=0.3, high=0.9)}
                )
            )
        ),
        source_magnitude_a=Uniform(low=1.0, high=5.0),
    )


def test_has_distributions_sees_nested_specs_and_dicts() -> None:
    assert _f07_config().has_distributions()
    only_dict = TnNetworkConfig(
        building_counts={"residential": Discrete(values=[1.0, 2.0])}
    )
    assert only_dict.has_distributions()
    only_soil = TnNetworkConfig(soil=HomogeneousSoilSpec(rho=Uniform(low=1, high=2)))
    assert only_soil.has_distributions()


def test_sample_resolves_every_distribution() -> None:
    resolved = _everything_config().sample(3)
    assert not resolved.has_distributions()
    dumped = resolved.model_dump_json()
    for kind in ("uniform", "discrete"):
        assert f'"kind":"{kind}"' not in dumped


def test_sample_draw_order_ignores_dict_insertion_order() -> None:
    a = TnNetworkConfig(
        building_types=[BuildingTypeSpec(name="x"), BuildingTypeSpec(name="y")],
        building_counts={
            "x": Discrete(values=[1.0, 2.0, 3.0]),
            "y": Discrete(values=[4.0, 5.0]),
        },
    )
    b = a.model_copy(
        update={"building_counts": dict(reversed(a.building_counts.items()))}
    )
    assert a.sample(11).building_counts == b.sample(11).building_counts


def test_resolved_config_replays_the_world_for_any_seed() -> None:
    world, resolved = TnNetworkGenerator(_f07_config(), seed=7).sample_world()
    reference = _world_json(world)
    for seed in (0, 1, 2):
        assert _world_json(TnNetworkGenerator(resolved, seed=seed).build()) == reference


def test_sample_world_rng_pins_every_draw() -> None:
    """R4 N05: presence draws used the construction RNG."""
    cfg = TnNetworkConfig(
        building_types=[
            BuildingTypeSpec(
                name="residential",
                grounding=GroundingSystemSpec(
                    electrodes=[FoundationElectrodeSpec(presence_prob=0.5)]
                ),
            )
        ],
        building_counts={"residential": 12},
    )
    wa, ra = TnNetworkGenerator(cfg, seed=1).sample_world(rng=7)
    wb, rb = TnNetworkGenerator(cfg, seed=2).sample_world(rng=7)
    assert ra == rb
    assert _world_json(wa) == _world_json(wb)
    wc, _ = TnNetworkGenerator(cfg, seed=3).sample_world(rng=np.random.default_rng(7))
    assert _world_json(wc) == _world_json(wa)


def test_build_accepts_an_explicit_rng() -> None:
    cfg = _f07_config()
    w1 = TnNetworkGenerator(cfg, seed=0).build(rng=5)
    w2 = TnNetworkGenerator(cfg, seed=99).build(rng=np.random.default_rng(5))
    assert _world_json(w1) == _world_json(w2)


def test_sample_world_without_rng_keeps_the_generator_stream() -> None:
    """``sample_world()`` without ``rng`` still uses the construction seed."""
    w1, _ = TnNetworkGenerator(_f07_config(), seed=4).sample_world()
    w2, _ = TnNetworkGenerator(_f07_config(), seed=4).sample_world()
    assert _world_json(w1) == _world_json(w2)


# ---------------------------------------------------------------------
# Penetration mask key
# ---------------------------------------------------------------------


def _square(cx: float, cy: float, osm_id: int | None) -> gf.BuildingFootprint:
    return gf.BuildingFootprint(
        polygon_xy_m=[
            (cx - 5, cy - 5),
            (cx + 5, cy - 5),
            (cx + 5, cy + 5),
            (cx - 5, cy + 5),
        ],
        osm_id=osm_id,
    )


def _equipped(footprints: list[gf.BuildingFootprint], p: float) -> set:
    layout = LvNetworkLayout.from_footprints(footprints, substation_xy=(0.0, 0.0))
    mask = layout.foundation_mask(p)
    return {fp.centroid_xy_m() for fp, m in zip(footprints, mask, strict=True) if m}


def test_mask_without_osm_id_ignores_footprint_order() -> None:
    fps = [
        _square(30.0 + 25.0 * (i // 2), 18.0 if i % 2 else -18.0, None)
        for i in range(24)
    ]
    forward = _equipped(fps, 0.5)
    assert 0 < len(forward) < len(fps)
    assert _equipped(list(reversed(fps)), 0.5) == forward
    rotated = [
        fp.model_copy(
            update={"polygon_xy_m": fp.polygon_xy_m[2:] + fp.polygon_xy_m[:2]}
        )
        for fp in fps
    ]
    assert _equipped(rotated, 0.5) == forward


def test_mask_with_osm_id_ignores_footprint_order() -> None:
    fps = [
        _square(30.0 + 25.0 * (i // 2), 18.0 if i % 2 else -18.0, 500 + i)
        for i in range(24)
    ]
    assert _equipped(list(reversed(fps)), 0.5) == _equipped(fps, 0.5)


def test_mask_does_not_depend_on_the_python_hash_seed() -> None:
    code = (
        "import groundfield as gf\n"
        "fps = [gf.BuildingFootprint(polygon_xy_m=[(i, 0), (i + 8, 0), (i + 8, 8), (i, 8)])"
        " for i in range(0, 400, 20)]\n"
        "lay = gf.generators.LvNetworkLayout.from_footprints(fps, substation_xy=(0, 0))\n"
        "print(''.join('1' if m else '0' for m in lay.foundation_mask(0.5)))\n"
    )
    outs = set()
    for seed in ("0", "1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
        outs.add(proc.stdout.strip())
    assert len(outs) == 1


# ---------------------------------------------------------------------
# 10.07 — MD5 marked usedforsecurity=False, digests unchanged
# ---------------------------------------------------------------------


def _osm_layout() -> LvNetworkLayout:
    fps = [
        gf.BuildingFootprint(
            polygon_xy_m=[(i * 20, 0), (i * 20 + 8, 0), (i * 20 + 8, 8), (i * 20, 8)],
            osm_id=4000000 + 7 * i,
        )
        for i in range(20)
    ]
    return LvNetworkLayout.from_footprints(fps, substation_xy=(0.0, 0.0))


@pytest.mark.finding("10.07")
def test_mask_values_are_those_of_v0150() -> None:
    """Literal values measured on v0.15.0: stored AP1 masks stay valid."""
    from groundfield.generators.layout.penetration import _stable_uniform

    layout = _osm_layout()
    bits = "".join("1" if m else "0" for m in layout.foundation_mask(0.3))
    assert bits == "01101001100111001000"
    bits = "".join("1" if m else "0" for m in layout.foundation_mask(0.3, salt=5))
    assert bits == "10000000111010011000"
    assert _stable_uniform(0, 123456789) == 0.061578575410906916
    assert _stable_uniform(3, 42) == 0.2550300460565941


@pytest.mark.finding("10.07")
def test_mask_works_where_md5_is_blocked_for_security(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On FIPS builds ``hashlib.md5()`` raises unless ``usedforsecurity=False``."""
    import hashlib

    from groundfield.generators.layout import penetration

    real_md5 = hashlib.md5

    def fips_md5(data: bytes = b"", *, usedforsecurity: bool = True):
        if usedforsecurity:
            raise ValueError("[digital envelope routines] unsupported (FIPS mode)")
        return real_md5(data, usedforsecurity=False)

    monkeypatch.setattr(penetration.hashlib, "md5", fips_md5)
    bits = "".join("1" if m else "0" for m in _osm_layout().foundation_mask(0.3))
    assert bits == "01101001100111001000"


# ---------------------------------------------------------------------
# plan() and build() of one generator (audit W3R2B, outside the round)
# ---------------------------------------------------------------------


def _jitter_config(jitter: float = 0.5) -> TnNetworkConfig:
    from groundfield.generators import ManhattanGridPlacement

    return TnNetworkConfig(
        building_counts={"residential": 6},
        placement=ManhattanGridPlacement(n_per_row=3, jitter_m=jitter),
    )


# audit W3R2B (outside the round): plan() and build()
def test_build_after_plan_gives_the_world_of_the_plan() -> None:
    """Base 06580ea: with placement jitter, ``plan()`` followed by
    ``build()`` on the same generator drew the jitter twice, so the built
    world was not the world of the plan (``materialise_plan(plan)`` was).
    ``build()`` now materialises the plan the generator returned last when
    nothing has drawn from its RNG since and the configuration is the
    same."""
    from groundfield.generators.materialise import materialise_plan

    gen = TnNetworkGenerator(_jitter_config(), seed=3)
    plan = gen.plan()
    assert _world_json(gen.build()) == _world_json(materialise_plan(plan))


# audit W3R2B (outside the round): plan() and build()
def test_the_generator_stream_still_gives_new_realisations() -> None:
    """Successive ``build()`` (or ``plan()``) calls still draw successive
    realisations from the generator's stream, the first ``build()`` of a
    generator is the one of 06580ea, and a ``build()`` after the plan has
    been used, after another draw from the RNG, with another configuration
    or with an explicit ``rng`` plans anew."""
    from groundfield.generators.materialise import materialise_plan

    cfg = _jitter_config()
    first = _world_json(TnNetworkGenerator(cfg, seed=3).build())
    gen = TnNetworkGenerator(cfg, seed=3)
    plan = gen.plan()
    assert _world_json(materialise_plan(plan)) == first
    assert _world_json(gen.build()) == first  # the plan's world
    second = _world_json(gen.build())  # the plan was used: the next one
    assert second != first
    # another plan, then a draw from the generator's RNG: plan anew
    gen = TnNetworkGenerator(cfg, seed=3)
    p1 = gen.plan()
    gen.rng.random()
    assert _world_json(gen.build()) != _world_json(materialise_plan(p1))
    # another configuration: plan anew, from the stream after the plan
    gen = TnNetworkGenerator(cfg, seed=3)
    gen.plan()
    other = _jitter_config(0.25)
    ref = TnNetworkGenerator(cfg, seed=3)
    ref.plan()
    assert _world_json(gen.build(other)) == _world_json(
        materialise_plan(ref.plan(other))
    )
    # an explicit rng plans from it and leaves the generator's own stream
    # alone: the plan of that stream is still the next world
    gen = TnNetworkGenerator(cfg, seed=3)
    p2 = gen.plan()
    assert _world_json(gen.build(rng=11)) == _world_json(
        TnNetworkGenerator(cfg, seed=0).build(rng=11)
    )
    assert _world_json(gen.build()) == _world_json(materialise_plan(p2))


# audit W3R2B (outside the round): plan() and build()
def test_without_jitter_plan_and_build_agree_as_before() -> None:
    """A configuration without per-site randomness gives the same world from
    every call, as in 06580ea."""
    from groundfield.generators.materialise import materialise_plan

    cfg = TnNetworkConfig(building_counts={"residential": 6})
    gen = TnNetworkGenerator(cfg, seed=3)
    reference = _world_json(TnNetworkGenerator(cfg, seed=3).build())
    assert _world_json(materialise_plan(gen.plan())) == reference
    assert _world_json(gen.build()) == reference
    assert _world_json(gen.build()) == reference
