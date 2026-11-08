"""Generator objects pickled by an earlier release still load and build.

Pydantic restores a pickled model from its ``__dict__`` without running the
validators, so the ``model_validator(mode="before")`` hooks that map renamed
fields of a JSON configuration (``HomogeneousSoilSpec.resistivity`` -> ``rho``,
``SoilLayerSpec.thickness_m`` -> ``h``, ``TnNetworkConfig.kvs`` ->
``cabinets``, ...) never ran for a pickle: a v0.15.0-pickled
``TnNetworkConfig`` unpickled and then failed in ``build()`` with
``AttributeError`` (audit A2G-07; the layout alone was migrated by 10.69 /
audit L6). The fixture ``tests/data/v0150_generator_pickles.pkl`` was written
by v0.15.0 (``dump_v0150_pickles.py`` of the stream evidence) and holds every
renamed field, the old class names ``KvsConfig`` / ``StarKvsTopology``, a
generator object, an ``OrtsnetzLayout`` with a ``KvsPlacement`` and, as the
reference, the v0.15.0 JSON of each config and of the layout.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import subprocess
import sys
import warnings

import pytest

import groundfield as gf
from groundfield.generators import (
    HomogeneousSoilSpec,
    LvNetworkLayout,
    RadialTrunkTopology,
    SoilLayerSpec,
    StarCabinetTopology,
    TnNetworkConfig,
    TnNetworkGenerator,
)
from groundfield.generators.layout.model import CabinetPlacement
from groundfield.generators.tn_network import CabinetConfig

pytestmark = pytest.mark.finding("10.69")

FIXTURE = pathlib.Path(__file__).parent / "data" / "v0150_generator_pickles.pkl"
_PRISTINE_V0150 = pathlib.Path("/home/claude/gf_pristine_v0150/src")
_KEYS = ["homogeneous", "multi_layer", "two_layer_radial", "star_kvs"]


@pytest.fixture(scope="module")
def fixture() -> dict:
    # The fixture is a file of this repository written by v0.15.0. Its
    # layout lives in the deprecated module path ortsnetz_builder, whose
    # import warns.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return pickle.loads(FIXTURE.read_bytes())  # noqa: S301


def _from_json(text: str) -> TnNetworkConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return TnNetworkConfig.model_validate_json(text)


def _world_json(world: gf.World) -> dict:
    return json.loads(world.model_dump_json())


# audit A2G-07
@pytest.mark.parametrize("key", _KEYS)
def test_v0150_pickled_config_equals_the_same_config_loaded_from_json(
    fixture: dict, key: str
) -> None:
    """Equal also in the classes: instances of the deprecated aliases
    ``KvsConfig`` / ``StarKvsTopology`` (the v0.15.0 classes) unpickle as
    ``CabinetConfig`` / ``StarCabinetTopology``, and ``kind="star_kvs"``
    becomes ``"star_cabinet"``, as from JSON."""
    assert fixture["version"] == "0.15.0"
    cfg = fixture["configs"][key]
    assert type(cfg.cabinets) is CabinetConfig
    assert type(cfg.pen_topology) in (StarCabinetTopology, RadialTrunkTopology)
    assert cfg == _from_json(fixture["json"][key])


# audit A2G-07
@pytest.mark.parametrize("key", _KEYS)
def test_v0150_pickled_config_builds_the_same_world(fixture: dict, key: str) -> None:
    """v0.15.0 pickle: AttributeError ('HomogeneousSoilSpec' object has no
    attribute 'rho', 'SoilLayerSpec' ... 'h', 'RadialTrunkTopology' ...
    'slots_per_cabinet')."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = TnNetworkGenerator(fixture["configs"][key], seed=1).build()
        reference = TnNetworkGenerator(_from_json(fixture["json"][key]), seed=1).build()
    assert _world_json(built) == _world_json(reference)


# audit A2G-07
def test_v0150_pickled_generator_builds(fixture: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = fixture["generator"].build()
        reference = TnNetworkGenerator(
            _from_json(fixture["json"]["two_layer_radial"]), seed=3
        ).build()
    assert _world_json(built) == _world_json(reference)


# audit A2G-07
def test_v0150_pickled_layout_unpickles_as_the_current_layout(fixture: dict) -> None:
    """``OrtsnetzLayout`` with ``KvsPlacement`` cabinets (10.69, audit L6):
    the same layout as from its v0.15.0 JSON, in the current classes."""
    layout = fixture["layout"]
    assert type(layout) is LvNetworkLayout
    assert [type(c) for c in layout.cabinet_placements] == [CabinetPlacement]
    assert set(vars(layout)) == set(LvNetworkLayout.model_fields)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        reference = LvNetworkLayout.model_validate_json(fixture["layout_json"])
    assert layout == reference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = layout.to_world(seed=0)
        expected = reference.to_world(seed=0)
    assert _world_json(built) == _world_json(expected)


# audit A2G-07
def test_upgraded_objects_hold_only_the_current_fields(fixture: dict) -> None:
    """Renamed values move to the new field (also in the set of explicitly
    set fields), and the object pickles again in the new form."""
    cfg = fixture["configs"]["two_layer_radial"]
    assert set(vars(cfg)) == set(TnNetworkConfig.model_fields)
    assert {"cabinets", "source_magnitude_a"} <= cfg.model_fields_set
    assert not {"kvs", "source_magnitude_A"} & cfg.model_fields_set
    assert cfg.source_magnitude_a == 5.0
    assert isinstance(cfg.cabinets, CabinetConfig)
    assert cfg.cabinets.count_per_100_buildings.low == 40.0
    topology = cfg.pen_topology
    assert isinstance(topology, RadialTrunkTopology)
    assert (topology.slots_per_cabinet, topology.cabinet_spacing_m) == (2, 30.0)
    layers = fixture["configs"]["multi_layer"].soil.layers
    assert [(layer.rho, layer.h) for layer in layers] == [(300.0, 2.5), (40.0, None)]
    assert all(set(vars(layer)) == set(SoilLayerSpec.model_fields) for layer in layers)
    soil = fixture["configs"]["homogeneous"].soil
    assert isinstance(soil, HomogeneousSoilSpec) and soil.rho == 123.0
    again = pickle.loads(pickle.dumps(cfg))
    assert again == cfg and set(vars(again)) == set(vars(cfg))
    assert "cabinets" in json.loads(cfg.model_dump_json())


# audit A2G-07
def test_a_current_pickle_round_trips_unchanged() -> None:
    cfg = TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=80.0), building_counts={"residential": 2}
    )
    again = pickle.loads(pickle.dumps(cfg))
    assert again == cfg and again.model_fields_set == cfg.model_fields_set


# audit A2G-07
def test_a_field_added_after_pickling_gets_its_default() -> None:
    """The upgrade also covers fields a later release adds: a state without
    them unpickles with the field defaults (pydantic left them missing)."""
    from groundfield.generators.tn_network import PenConfig

    pen = PenConfig()
    state = pen.__getstate__()
    values = dict(state["__dict__"])
    del values["depth_m"]
    restored = PenConfig.__new__(PenConfig)
    restored.__setstate__({**state, "__dict__": values})
    assert restored == pen


_DUMP_V0150_CONFIG = r"""
import pickle, sys, warnings
warnings.simplefilter("ignore")
from groundfield.generators import (TnNetworkConfig, HomogeneousSoilSpec,
    MultiLayerSoilSpec, SoilLayerSpec)
cfgs = {
    "homogeneous": TnNetworkConfig(soil=HomogeneousSoilSpec(resistivity=123.0),
                                   building_counts={"residential": 3}),
    "multi_layer": TnNetworkConfig(soil=MultiLayerSoilSpec(layers=[
        SoilLayerSpec(resistivity=300.0, thickness_m=2.5),
        SoilLayerSpec(resistivity=40.0, thickness_m=None)]),
        building_counts={"residential": 3}),
}
pickle.dump(cfgs, open(sys.argv[1], "wb"))
"""


# audit A2G-07 (the auditor's test, against a fresh v0.15.0 pickle)
@pytest.mark.skipif(
    not _PRISTINE_V0150.is_dir(), reason="pristine v0.15.0 tree missing"
)
@pytest.mark.parametrize("key", ["homogeneous", "multi_layer"])
def test_fresh_v0150_pickle_still_builds(key: str, tmp_path: pathlib.Path) -> None:
    target = tmp_path / "cfg_v0150.pkl"
    env = dict(os.environ, PYTHONPATH=str(_PRISTINE_V0150))
    subprocess.run(
        [sys.executable, "-c", _DUMP_V0150_CONFIG, str(target)],
        env=env,
        check=True,
        timeout=300,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = pickle.loads(target.read_bytes())[key]  # noqa: S301
        world = TnNetworkGenerator(cfg, seed=1).build()
    assert world.soil is not None
