"""Network plan and materialiser shared by the generator front-ends (10.78).

Both front-ends — :class:`TnNetworkGenerator` and the layout builder —
produce a pure-data :class:`NetworkPlan`; one materialiser turns any plan
into a world. These tests pin the contract:

* ``build()`` is exactly ``materialise_plan(plan())``;
* a plan survives a JSON round trip and still materialises to the same
  world (no random numbers are drawn while materialising);
* a plan is concrete (no distributions, no absent electrodes) and its
  node references are consistent;
* ``GroundingSystemSpec.build_at`` is ``plan_site`` + ``materialise_site``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import (
    BuildingTypeSpec,
    Discrete,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    LvNetworkLayout,
    ManhattanGridPlacement,
    MeasurementInjectionConfig,
    MeasurementProbeConfig,
    MeasurementSetupConfig,
    RadialTrunkTopology,
    RodElectrodeSpec,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
    Uniform,
    buried_lead,
    overhead_lead,
)
from groundfield.generators.materialise import materialise_plan, materialise_site
from groundfield.generators.plan import (
    ConductorParameters,
    NetworkPlan,
    PlannedBranch,
    PlannedElectrode,
    PlannedSite,
)

pytestmark = pytest.mark.finding("10.78")


def _canonical(world: gf.World) -> str:
    return json.dumps(world.model_dump(mode="json"), sort_keys=True)


def _tn_configs() -> dict[str, TnNetworkConfig]:
    stochastic = TnNetworkConfig(
        soil=TwoLayerSoilSpec(
            rho_1=Uniform(low=80.0, high=300.0), rho_2=500.0, h_1=2.0
        ),
        placement=ManhattanGridPlacement(
            spacing_x_m=Uniform(low=20, high=30), jitter_m=2.0
        ),
        building_types=[
            BuildingTypeSpec(
                name="residential",
                grounding=GroundingSystemSpec(
                    electrodes=[
                        FoundationElectrodeSpec(
                            size_m=Uniform(low=8, high=12),
                            presence_prob=0.6,
                            concrete_rho_ohm_m=150.0,
                        ),
                        RodElectrodeSpec(
                            length_m=Uniform(low=1, high=3),
                            presence_prob=0.5,
                            offset_xy_m=(6.0, 0.0),
                        ),
                    ]
                ),
            )
        ],
        building_counts={"residential": Discrete(values=[6.0, 9.0])},
    )
    measured = TnNetworkConfig(
        building_counts={"residential": 3},
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(150.0, 0.0), feed_lead=overhead_lead()
            ),
            probe=MeasurementProbeConfig(position_xy=(50.0, 0.0), lead=buried_lead()),
        ),
    )
    radial = TnNetworkConfig(
        building_counts={"residential": 20},
        pen_topology=RadialTrunkTopology(slots_per_substation=3),
    )
    return {"stochastic": stochastic, "measured": measured, "radial": radial}


@pytest.mark.parametrize("case", ["stochastic", "measured", "radial"])
def test_tn_build_is_materialised_plan(case: str) -> None:
    cfg = _tn_configs()[case]
    plan = TnNetworkGenerator(cfg, seed=42).plan()
    assert isinstance(plan, NetworkPlan)
    built = TnNetworkGenerator(cfg, seed=42).build()
    assert _canonical(materialise_plan(plan)) == _canonical(built)


@pytest.mark.parametrize("case", ["stochastic", "measured", "radial"])
def test_tn_plan_json_round_trip_reproduces_world(case: str) -> None:
    plan = TnNetworkGenerator(_tn_configs()[case], seed=7).plan()
    restored = NetworkPlan.model_validate_json(plan.model_dump_json())
    assert restored == plan
    assert _canonical(restored.to_world()) == _canonical(materialise_plan(plan))


def test_tn_plan_records_sites_roles_and_footprint_free_buildings() -> None:
    plan = TnNetworkGenerator(_tn_configs()["measured"], seed=0).plan()
    roles = [s.role for s in plan.all_sites]
    assert roles[0] == "substation"
    assert roles.count("building") == 3
    assert roles[-2:] == ["auxiliary", "probe"]
    assert plan.measurement is not None
    assert plan.measurement.feed_lead is not None
    assert plan.measurement.feed_lead.role == "measurement_lead"
    assert all(b.footprint is None for b in plan.sites_with_role("building"))
    assert plan.sources[0].attached_to == "trafo"
    # The source drives the loop between the substation and the
    # instrument's current terminal, at the end of the feed lead (10.17).
    assert plan.sources[0].return_to == "meas_H"
    assert [t.name for t in plan.measurement.terminals] == ["meas_H", "meas_S"]
    assert plan.measurement.feed_lead.start == "meas_H"
    assert plan.measurement.probe_lead.start == "meas_S"


def test_tn_plan_is_concrete() -> None:
    plan = TnNetworkGenerator(_tn_configs()["stochastic"], seed=3).plan()
    for site in plan.all_sites:
        for planned in site.electrodes:
            assert float(planned.spec.presence_prob) == 1.0
            assert not any(
                isinstance(v, gf.Distribution) for v in planned.spec.__dict__.values()
            )


def test_planned_electrode_rejects_distributions_and_absent_electrodes() -> None:
    with pytest.raises(ValueError, match="distribution"):
        PlannedElectrode(
            index=0, spec=RodElectrodeSpec(length_m=Uniform(low=1, high=2))
        )
    with pytest.raises(ValueError, match="presence_prob"):
        PlannedElectrode(index=0, spec=RodElectrodeSpec(presence_prob=0.5))


def test_network_plan_validates_node_names() -> None:
    site = PlannedSite(name="a", role="building", position_xy=(0.0, 0.0))
    soil = gf.HomogeneousSoil(resistivity=100.0)
    with pytest.raises(ValueError, match="used twice"):
        NetworkPlan(name="w", soil=soil, sites=[site, site])
    branch = PlannedBranch(
        name="c", role="pen", start="a", end="nowhere", conductor=ConductorParameters()
    )
    with pytest.raises(ValueError, match="unknown node 'nowhere'"):
        NetworkPlan(name="w", soil=soil, sites=[site], branches=[branch])


def test_build_at_equals_plan_site_then_materialise_site() -> None:
    spec = GroundingSystemSpec(
        electrodes=[
            RodElectrodeSpec(length_m=Uniform(low=1, high=3), presence_prob=0.5),
            FoundationElectrodeSpec(
                size_m=Uniform(low=8, high=12),
                concrete_rho_ohm_m=Uniform(low=50, high=500),
            ),
            RodElectrodeSpec(length_m=2.0, offset_xy_m=(3.0, 0.0)),
        ]
    )
    for seed in range(6):
        w_a = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
        anchor_a = spec.build_at(w_a, (1.0, 2.0), "s", np.random.default_rng(seed))
        w_b = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
        site = spec.plan_site(
            "s",
            role="building",
            position_xy=(1.0, 2.0),
            rng=np.random.default_rng(seed),
        )
        anchor_b = materialise_site(w_b, site)
        assert anchor_a == anchor_b == site.anchor
        assert _canonical(w_a) == _canonical(w_b)


def test_absent_electrode_keeps_its_index_in_the_name() -> None:
    spec = GroundingSystemSpec(
        electrodes=[RodElectrodeSpec(presence_prob=0.0), RodElectrodeSpec(length_m=2.0)]
    )
    site = spec.plan_site(
        "h", role="building", position_xy=(0.0, 0.0), rng=np.random.default_rng(0)
    )
    assert [p.index for p in site.electrodes] == [1]
    assert site.anchor == "h_rod_1"


def test_materialising_draws_no_random_numbers() -> None:
    plan = TnNetworkGenerator(_tn_configs()["stochastic"], seed=11).plan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        worlds = [_canonical(materialise_plan(plan)) for _ in range(3)]
    assert len(set(worlds)) == 1


# ---------------------------------------------------------------------
# Layout builder front-end
# ---------------------------------------------------------------------


def _house(
    cx: float, cy: float, oid: int, w: float = 10.0, h: float = 8.0
) -> gf.BuildingFootprint:
    return gf.BuildingFootprint(
        polygon_xy_m=[
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ],
        osm_id=oid,
    )


def _layout() -> LvNetworkLayout:
    fps = [
        _house(30.0 + 25.0 * (i // 2), 18.0 if i % 2 == 0 else -18.0, 500 + i)
        for i in range(8)
    ]
    lay = LvNetworkLayout.from_footprints(fps, substation_xy=(0.0, 0.0))
    cab = lay.add_cabinet((80.0, 0.0))
    lay.add_pen_cable(start="substation", end=cab, min_segment_length_m=8.0)
    lay.add_pen_cable(start=cab, end_xy=(160.0, 0.0), min_segment_length_m=8.0)
    lay.connect_all_buildings()
    lay.add_auxiliary_electrode(distance_m=250.0, direction_deg=90.0)
    return lay


def test_layout_to_world_is_materialised_plan() -> None:
    lay = _layout()
    kwargs = dict(seed=3, foundation_mask=lay.foundation_mask(0.75))
    plan = lay.plan(**kwargs)
    assert _canonical(materialise_plan(plan)) == _canonical(lay.to_world(**kwargs))
    restored = NetworkPlan.model_validate_json(plan.model_dump_json())
    assert _canonical(restored.to_world()) == _canonical(lay.to_world(**kwargs))


def test_layout_plan_structure() -> None:
    lay = _layout()
    plan = lay.plan(seed=0)
    assert [s.role for s in plan.sites[:2]] == ["substation", "cabinet"]
    assert all(s.footprint is not None for s in plan.sites_with_role("building"))
    junction_names = {j.name for j in plan.junctions}
    pen = [b for b in plan.branches if b.role == "pen"]
    drops = [b for b in plan.branches if b.role == "service_drop"]
    assert len(drops) == len(lay.connections)
    assert all(b.start in junction_names | {"substation", "kvs_0"} for b in drops)
    assert {b.cable for b in pen} == {c.name for c in lay.pen_cables}
    # One two-terminal source closes the measurement loop through the
    # auxiliary electrode (``return_to``, honoured since 10.16).
    assert plan.measurement is not None
    assert [s.phase_deg for s in plan.sources] == [0.0]
    assert plan.sources[0].return_to is not None


def test_layout_masked_houses_are_not_in_the_plan() -> None:
    lay = _layout()
    mask = [i % 2 == 0 for i in range(len(lay.footprints))]
    plan = lay.plan(foundation_mask=mask)
    names = {s.name for s in plan.sites_with_role("building")}
    assert names == {f"house_{i:03d}" for i in range(len(mask)) if mask[i]}
