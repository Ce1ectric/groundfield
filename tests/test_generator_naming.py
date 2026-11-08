"""English names in the generators with deprecated aliases (10.69).

The German identifiers of the generator layer (``OrtsnetzLayout``,
``StarKvsTopology``, ``add_kvs``, ``ortsnetz_builder``, ...) and the
capitalised unit suffixes (``source_magnitude_A``) were renamed. Every old
name keeps working, emits a :class:`DeprecationWarning` that names the
replacement and the earliest removal (2027.5), and configurations written
by v0.15.0 still load.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

import groundfield as gf
import groundfield.generators as gen
from groundfield.generators import (
    CabinetConfig,
    CabinetPlacement,
    HomogeneousSoilSpec,
    LvNetworkLayout,
    MultiLayerSoilSpec,
    RadialTrunkTopology,
    SoilLayerSpec,
    StarCabinetTopology,
    TnNetworkConfig,
    TnNetworkGenerator,
)

pytestmark = pytest.mark.finding("10.69")

DATA = pathlib.Path(__file__).parent / "data"


def _deprecated(old: str, new: str):
    return pytest.warns(DeprecationWarning, match=rf"{old}.*{new}.*2027\.5")


# ---------------------------------------------------------------------
# Classes and module
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new", "kwargs"),
    [
        (
            "OrtsnetzLayout",
            "LvNetworkLayout",
            {"footprints": [], "substation_xy": (0, 0)},
        ),
        ("KvsPlacement", "CabinetPlacement", {"name": "c", "position_xy": (1.0, 2.0)}),
        ("KvsConfig", "CabinetConfig", {}),
        ("StarKvsTopology", "StarCabinetTopology", {}),
    ],
)
def test_deprecated_class_aliases_warn_and_subclass(
    old: str, new: str, kwargs: dict
) -> None:
    old_cls, new_cls = getattr(gen, old), getattr(gen, new)
    assert issubclass(old_cls, new_cls)
    with _deprecated(old, new):
        obj = old_cls(**kwargs)
    assert isinstance(obj, new_cls)


def test_deprecated_layout_class_via_classmethod() -> None:
    with _deprecated("OrtsnetzLayout", "LvNetworkLayout"):
        layout = gen.OrtsnetzLayout.from_footprints([], substation_xy=(0.0, 0.0))
    assert isinstance(layout, LvNetworkLayout)


def test_ortsnetz_builder_module_alias_warns() -> None:
    sys.modules.pop("groundfield.generators.ortsnetz_builder", None)
    with _deprecated("ortsnetz_builder", "groundfield.generators.layout"):
        mod = importlib.import_module("groundfield.generators.ortsnetz_builder")
    assert mod.LvNetworkLayout is LvNetworkLayout
    assert issubclass(mod.OrtsnetzLayout, LvNetworkLayout)


def test_import_groundfield_emits_no_deprecation_warning() -> None:
    code = (
        "import warnings; warnings.simplefilter('error', DeprecationWarning); "
        "import groundfield, groundfield.generators"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_public_generator_names_are_english() -> None:
    deprecated = {"KvsConfig", "StarKvsTopology", "OrtsnetzLayout", "KvsPlacement"}
    german = [
        n
        for n in gen.__all__
        if n not in deprecated and any(t in n.lower() for t in ("ortsnetz", "kvs"))
    ]
    assert german == []


# ---------------------------------------------------------------------
# Renamed fields (spec layer)
# ---------------------------------------------------------------------


def test_tn_config_renamed_fields_accept_old_names() -> None:
    with _deprecated("TnNetworkConfig.kvs", "TnNetworkConfig.cabinets"):
        cfg = TnNetworkConfig(kvs=CabinetConfig(fixed_count=2))
    assert cfg.cabinets.fixed_count == 2
    with _deprecated("source_magnitude_A", "source_magnitude_a"):
        cfg = TnNetworkConfig(source_magnitude_A=2.5)
    assert cfg.source_magnitude_a == 2.5
    with _deprecated("TnNetworkConfig.kvs", "TnNetworkConfig.cabinets"):
        assert cfg.kvs is cfg.cabinets
    with _deprecated("source_magnitude_A", "source_magnitude_a"):
        assert cfg.source_magnitude_A == 2.5


def test_both_old_and_new_name_is_an_error() -> None:
    with pytest.raises(ValueError, match="both 'source_magnitude_A'"):
        TnNetworkConfig(source_magnitude_A=1.0, source_magnitude_a=2.0)


def test_v0150_config_json_still_loads() -> None:
    text = (DATA / "v0150_tn_network_config.json").read_text(encoding="utf-8")
    assert '"kvs"' in text and '"source_magnitude_A"' in text and '"star_kvs"' in text
    with pytest.warns(DeprecationWarning):
        cfg = TnNetworkConfig.model_validate_json(text)
    assert cfg.cabinets.count_per_100_buildings == 7.0
    assert cfg.source_magnitude_a == 2.5
    assert cfg.soil.rho == 150.0
    assert isinstance(cfg.pen_topology, StarCabinetTopology)
    assert cfg.pen_topology.kind == "star_cabinet"
    dumped = json.loads(cfg.model_dump_json())
    assert "cabinets" in dumped and "kvs" not in dumped
    # The world does not depend on how the config was written.
    w_old = TnNetworkGenerator(cfg, seed=0).build()
    new_cfg = TnNetworkConfig.model_validate_json(cfg.model_dump_json())
    w_new = TnNetworkGenerator(new_cfg, seed=0).build()
    assert w_old.model_dump_json() == w_new.model_dump_json()


def test_cabinet_config_rate_rename() -> None:
    with _deprecated("quote_per_100_buildings", "count_per_100_buildings"):
        c = CabinetConfig(quote_per_100_buildings=3.0)
    assert c.count_per_100_buildings == 3.0
    with _deprecated("quote_per_100_buildings", "count_per_100_buildings"):
        assert c.quote_per_100_buildings == 3.0


def test_radial_topology_renames() -> None:
    old_json = (
        '{"kind":"radial_trunk","n_feeders":4,"feeder_directions_deg":null,'
        '"max_feeder_length_m":500.0,"slots_per_substation":8,"slots_per_kvs":3,'
        '"kvs_spacing_m":40.0,"name_prefix":"kvs"}'
    )
    with pytest.warns(DeprecationWarning, match="slots_per_cabinet"):
        topo = RadialTrunkTopology.model_validate_json(old_json)
    assert (topo.slots_per_cabinet, topo.cabinet_spacing_m) == (3, 40.0)
    with _deprecated("plan_feeder_kvs_positions", "plan_feeder_cabinet_positions"):
        old = topo.plan_feeder_kvs_positions((0.0, 0.0), 0, 6)
    assert old == topo.plan_feeder_cabinet_positions((0.0, 0.0), 0, 6)


def test_soil_specs_mirror_model_symbols() -> None:
    with _deprecated("HomogeneousSoilSpec.resistivity", "HomogeneousSoilSpec.rho"):
        spec = HomogeneousSoilSpec(resistivity=42.0)
    assert spec.rho == 42.0
    old_json = (
        '{"kind":"multi_layer","layers":[{"resistivity":300.0,"thickness_m":2.0,'
        '"relative_permittivity":10.0},{"resistivity":50.0,"thickness_m":null,'
        '"relative_permittivity":10.0}]}'
    )
    with pytest.warns(DeprecationWarning, match="SoilLayerSpec"):
        multi = MultiLayerSoilSpec.model_validate_json(old_json)
    assert [(lay.rho, lay.h) for lay in multi.layers] == [(300.0, 2.0), (50.0, None)]
    new = MultiLayerSoilSpec(
        layers=[SoilLayerSpec(rho=300.0, h=2.0), SoilLayerSpec(rho=50.0, h=None)]
    )
    rng = np.random.default_rng(0)
    assert multi.to_soil(rng) == new.to_soil(rng)


# ---------------------------------------------------------------------
# Layout builder methods and keyword arguments
# ---------------------------------------------------------------------


def _layout() -> LvNetworkLayout:
    fps = [
        gf.BuildingFootprint(
            polygon_xy_m=[(x - 4, 10), (x + 4, 10), (x + 4, 18), (x - 4, 18)], osm_id=i
        )
        for i, x in enumerate((30.0, 55.0, 80.0))
    ]
    lay = LvNetworkLayout.from_footprints(fps, substation_xy=(0.0, 0.0))
    lay.add_pen_cable(start="substation", end_xy=(100.0, 0.0), min_segment_length_m=8.0)
    lay.connect_buildings()
    return lay


def test_layout_add_kvs_and_kvs_placements() -> None:
    lay = _layout()
    with _deprecated("add_kvs", "add_cabinet"):
        name = lay.add_kvs((50.0, -20.0))
    assert lay.cabinet_placements == [
        CabinetPlacement(name=name, position_xy=(50.0, -20.0))
    ]
    with _deprecated("kvs_placements", "cabinet_placements"):
        assert lay.kvs_placements is lay.cabinet_placements
    with _deprecated("kvs_placements", "cabinet_placements"):
        again = LvNetworkLayout(
            footprints=[],
            substation_xy=(0.0, 0.0),
            kvs_placements=lay.cabinet_placements,
        )
    assert again.cabinet_placements == lay.cabinet_placements


def test_layout_to_world_deprecated_keywords() -> None:
    lay = _layout()
    lay.add_cabinet((50.0, -20.0))
    spec = gf.GroundingSystemSpec(electrodes=[gf.RodElectrodeSpec(length_m=2.0)])
    new = lay.to_world(cabinet_grounding=spec, source_magnitude_a=3.0)
    with _deprecated("kvs_grounding", "cabinet_grounding"):
        old = lay.to_world(kvs_grounding=spec, source_magnitude_a=3.0)
    assert old.model_dump_json() == new.model_dump_json()
    with _deprecated("source_magnitude_A", "source_magnitude_a"):
        old = lay.to_world(cabinet_grounding=spec, source_magnitude_A=3.0)
    assert old.model_dump_json() == new.model_dump_json()
    with pytest.raises(
        TypeError, match="unexpected keyword argument 'source_magnitude'"
    ):
        lay.to_world(source_magnitude=3.0)


def test_layout_from_osm_deprecated_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    from groundfield.geo import osm
    from groundfield.geo.projection import Projector

    def fake_query_and_project(**kwargs):
        return [], Projector(kwargs["lat0_deg"], kwargs["lon0_deg"])

    monkeypatch.setattr(osm, "query_and_project", fake_query_and_project)
    with _deprecated("kvs_xys", "cabinet_xys"):
        lay = LvNetworkLayout.from_osm(
            center_lat_deg=51.9,
            center_lon_deg=10.8,
            radius_m=100.0,
            kvs_xys=[(10.0, 0.0)],
        )
    assert [c.position_xy for c in lay.cabinet_placements] == [(10.0, 0.0)]


def test_verify_current_balance_lower_case_keys() -> None:
    lay = _layout()
    lay.add_auxiliary_electrode(distance_m=200.0)
    world = lay.to_world()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        engine = gf.create_engine(
            backend="image", segment_length=2.0, frequencies=[50.0]
        )
        res = engine.solve(world)
    out = lay.verify_current_balance(res)
    assert "positive_side_a" in out and "positive_side_A" not in set(out)
    with _deprecated("positive_side_A", "positive_side_a"):
        assert out["positive_side_A"] == out["positive_side_a"]
    with _deprecated("source_magnitude_A", "source_magnitude_a"):
        z_old = lay.measured_grounding_impedance(res, source_magnitude_A=2.0)
    assert z_old == lay.measured_grounding_impedance(res, source_magnitude_a=2.0)


_PRISTINE_V0150 = pathlib.Path("/home/claude/gf_pristine_v0150/src")

_DUMP_V0150_LAYOUT = """
import pickle, sys, warnings
warnings.simplefilter("ignore")
import groundfield as gf
from groundfield.generators import OrtsnetzLayout
def house(cx, cy, oid):
    return gf.BuildingFootprint(polygon_xy_m=[(cx - 5, cy - 5), (cx + 5, cy - 5),
                                (cx + 5, cy + 5), (cx - 5, cy + 5)], osm_id=oid)
lay = OrtsnetzLayout.from_footprints([house(30.0, 18.0, 1), house(55.0, -18.0, 2)],
                                     substation_xy=(0.0, 0.0))
k = lay.add_kvs((40.0, 0.0))
lay.add_pen_cable(start="substation", end=k)
lay.add_pen_cable(start=k, end_xy=(80.0, 0.0))
lay.connect_all_buildings()
lay.add_auxiliary_electrode(distance_m=150.0)
pickle.dump(lay, open(sys.argv[1], "wb"))
"""


@pytest.mark.finding("10.69")
@pytest.mark.skipif(
    not _PRISTINE_V0150.is_dir(), reason="pristine v0.15.0 tree not available"
)
def test_layout_pickled_with_v0150_still_builds(tmp_path: pathlib.Path) -> None:
    """Audit L6: a v0.15.0 pickle carries ``kvs_placements``; unpickling
    worked but ``to_world`` raised AttributeError ('cabinet_placements')."""
    import os
    import pickle

    target = tmp_path / "layout_v0150.pkl"
    env = dict(os.environ, PYTHONPATH=str(_PRISTINE_V0150))
    subprocess.run(
        [sys.executable, "-c", _DUMP_V0150_LAYOUT, str(target)],
        env=env,
        check=True,
        timeout=300,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        layout = pickle.loads(target.read_bytes())
    assert [c.name for c in layout.cabinet_placements] == ["kvs_0"]
    assert all(type(c) is CabinetPlacement for c in layout.cabinet_placements)
    assert "kvs_placements" not in layout.__dict__
    world = layout.to_world(seed=0, foundation_outline="spec")
    assert any(e.name.startswith("kvs_0") for e in world.electrodes)
    assert world.sources[0].return_to == "aux_rod_0"
    # The migrated layout pickles and dumps in the new form.
    again = pickle.loads(pickle.dumps(layout))
    assert again == layout
    assert "cabinet_placements" in json.loads(layout.model_dump_json())
