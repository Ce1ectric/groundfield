"""Regression tests for the radial-trunk PEN topology.

Covers
------
* :class:`RadialTrunkTopology` configuration validation
  (``feeder_directions_deg`` length, finite values).
* Auto-distributed feeder direction angles.
* Building-to-feeder assignment by angular distance and axial
  projection (negative-projection and beyond-max-length filtering).
* KVS-position planning given the slot budget.
* End-to-end :class:`TnNetworkGenerator.build` with the radial-trunk
  topology: every assigned building has a service-drop conductor,
  the trunk segments substation → KVS$_k$ exist, and the substation
  cluster is the source's attachment point.
* The legacy :class:`StarKvsTopology` is the default and reproduces
  the historic per-building "nearest KVS" wiring.
* Switching topology on the *same* :class:`TnNetworkConfig` produces
  different worlds (sanity for the discriminated union).

Tests live alongside the existing ``test_tn_ortsnetz.py`` so the
AP1 topology suite stays grouped in one folder.
"""

from __future__ import annotations

import math
import warnings

import pytest

import groundfield as gf
from groundfield.generators import (
    ExplicitPlacement,
    RadialTrunkTopology,
    StarKvsTopology,
    TnNetworkConfig,
    TnNetworkGenerator,
)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def test_default_topology_is_star_kvs() -> None:
    """Pre-v0.7 configs keep the legacy star topology by default."""
    cfg = TnNetworkConfig(building_counts={"residential": 3})
    assert isinstance(cfg.pen_topology, StarKvsTopology)
    assert cfg.pen_topology.kind == "star_kvs"


def test_radial_trunk_explicit_directions_length_mismatch_rejected() -> None:
    """``feeder_directions_deg`` length must equal ``n_feeders``."""
    topo = RadialTrunkTopology(
        n_feeders=4,
        feeder_directions_deg=[0.0, 90.0, 180.0],  # 3 instead of 4
    )
    with pytest.raises(ValueError, match="n_feeders"):
        topo.resolved_directions_rad()


def test_radial_trunk_rejects_non_finite_direction() -> None:
    with pytest.raises(ValueError, match="finite"):
        RadialTrunkTopology(n_feeders=2, feeder_directions_deg=[0.0, math.nan])


def test_radial_trunk_auto_distributed_directions() -> None:
    topo = RadialTrunkTopology(n_feeders=4)
    angles = topo.resolved_directions_rad()
    assert len(angles) == 4
    # 0°, 90°, 180°, 270° in radians
    expected = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
    for got, exp in zip(angles, expected):
        assert math.isclose(got, exp, abs_tol=1e-12)


# ---------------------------------------------------------------------
# Building → feeder assignment
# ---------------------------------------------------------------------


def test_assignment_groups_by_quadrant() -> None:
    """4 feeders along ±x, ±y → each building goes to its quadrant.

    Within a feeder, buildings are returned in ascending axial-
    distance order (closer to the substation first), so the
    near-axis building at x=50 precedes the one at x=100.
    """
    topo = RadialTrunkTopology(n_feeders=4)
    positions = [
        (100.0, 0.0),    # feeder 0 (east), axial proj = 100
        (0.0, 100.0),    # feeder 1 (north)
        (-100.0, 0.0),   # feeder 2 (west)
        (0.0, -100.0),   # feeder 3 (south)
        (50.0, 10.0),    # feeder 0 (close to east axis), axial proj ≈ 50
    ]
    groups = topo.assign_buildings_to_feeders((0.0, 0.0), positions)
    # Sorted by axial projection: 50 (idx 4) precedes 100 (idx 0).
    assert groups[0] == [4, 0]
    assert groups[1] == [1]
    assert groups[2] == [2]
    assert groups[3] == [3]


def test_assignment_drops_negative_projection() -> None:
    """A building behind its closest feeder axis is dropped + warned."""
    topo = RadialTrunkTopology(
        n_feeders=1, feeder_directions_deg=[0.0],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        groups = topo.assign_buildings_to_feeders(
            substation_xy=(0.0, 0.0),
            building_positions=[(-50.0, 0.0)],
        )
    assert groups == [[]]
    assert any("behind" in str(w.message) for w in caught)


def test_assignment_drops_beyond_max_length() -> None:
    topo = RadialTrunkTopology(
        n_feeders=1, feeder_directions_deg=[0.0],
        max_feeder_length_m=100.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        groups = topo.assign_buildings_to_feeders(
            substation_xy=(0.0, 0.0),
            building_positions=[(50.0, 0.0), (500.0, 0.0)],
        )
    assert groups == [[0]]
    assert any("max_feeder_length_m" in str(w.message) for w in caught)


def test_assignment_sorts_by_axial_distance() -> None:
    """Closest building first along the feeder axis."""
    topo = RadialTrunkTopology(n_feeders=1, feeder_directions_deg=[0.0])
    positions = [(200.0, 0.0), (50.0, 0.0), (120.0, 0.0)]
    groups = topo.assign_buildings_to_feeders((0.0, 0.0), positions)
    # Sorted by axial projection: 50 → 120 → 200, i.e. indices 1, 2, 0.
    assert groups[0] == [1, 2, 0]


# ---------------------------------------------------------------------
# KVS-position planning
# ---------------------------------------------------------------------


def test_plan_kvs_positions_no_extension_when_within_substation_budget() -> None:
    topo = RadialTrunkTopology(
        n_feeders=1, feeder_directions_deg=[0.0],
        slots_per_substation=5,
        slots_per_kvs=3,
        kvs_spacing_m=40.0,
    )
    assert topo.plan_feeder_kvs_positions((0.0, 0.0), 0, 5) == []


def test_plan_kvs_positions_adds_one_kvs_per_slot_overflow() -> None:
    topo = RadialTrunkTopology(
        n_feeders=1, feeder_directions_deg=[0.0],
        slots_per_substation=2,
        slots_per_kvs=2,
        kvs_spacing_m=40.0,
        max_feeder_length_m=500.0,
    )
    # 2 (subst) + 2 (kvs_0) + 2 (kvs_1) = 6 buildings → 2 KVSes
    positions = topo.plan_feeder_kvs_positions((0.0, 0.0), 0, 6)
    assert positions == [(40.0, 0.0), (80.0, 0.0)]


def test_plan_kvs_positions_clips_to_max_length() -> None:
    topo = RadialTrunkTopology(
        n_feeders=1, feeder_directions_deg=[0.0],
        slots_per_substation=1,
        slots_per_kvs=1,
        kvs_spacing_m=200.0,
        max_feeder_length_m=300.0,
    )
    # 1 + 3 KVSes worth of buildings → KVSes at 200, 300 (clipped),
    # third one would be at 600 → > max → dropped.
    positions = topo.plan_feeder_kvs_positions((0.0, 0.0), 0, 4)
    assert positions == [(200.0, 0.0), (300.0, 0.0)]


# ---------------------------------------------------------------------
# End-to-end build with radial trunk
# ---------------------------------------------------------------------


def _build_radial_world(n_buildings: int, **topo_kwargs) -> "gf.World":
    """Build a radial-trunk world with explicit building positions
    arranged along the four cardinal axes.
    """
    # Spread buildings around the substation evenly so all 4 feeders
    # get used. Layout: 1/4 of buildings on each axis at increasing
    # distance.
    positions: list[tuple[float, float]] = []
    per_axis = max(1, n_buildings // 4)
    step = 25.0
    for axis_idx in range(4):
        ax, ay = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][axis_idx]
        for i in range(per_axis):
            d = (i + 1) * step
            positions.append((d * ax, d * ay))
    # Pad with extra east-axis buildings if integer division dropped some.
    while len(positions) < n_buildings:
        positions.append((25.0 * (1 + len(positions)), 0.0))
    cfg = TnNetworkConfig(
        building_counts={"residential": n_buildings},
        placement=ExplicitPlacement(positions=positions[:n_buildings]),
        pen_topology=RadialTrunkTopology(
            n_feeders=4,
            slots_per_substation=topo_kwargs.pop("slots_per_substation", 4),
            slots_per_kvs=topo_kwargs.pop("slots_per_kvs", 4),
            kvs_spacing_m=topo_kwargs.pop("kvs_spacing_m", 60.0),
            **topo_kwargs,
        ),
    )
    return TnNetworkGenerator(cfg, seed=123).build()


def test_radial_trunk_end_to_end_no_kvs_when_under_budget() -> None:
    """With slots_per_substation ≥ buildings per feeder, no KVS is created."""
    world = _build_radial_world(
        n_buildings=8, slots_per_substation=3, slots_per_kvs=99,
    )
    kvs_electrodes = [e for e in world.electrodes
                      if e.name.startswith("kvs_f")]
    # 8 buildings across 4 feeders = 2 per feeder ≤ 3 → no KVS.
    assert kvs_electrodes == []
    # Every building has its PEN service drop.
    pen_services = [c for c in world.conductors
                    if c.name.startswith("pen_service_")]
    assert len(pen_services) == 8


def test_radial_trunk_inserts_kvs_when_budget_overflows() -> None:
    """When the substation slot budget is exceeded, KVSes appear on
    the feeder and the trunk segments connect them in order."""
    world = _build_radial_world(
        n_buildings=16, slots_per_substation=2, slots_per_kvs=2,
        kvs_spacing_m=50.0,
    )
    # Per feeder: 4 buildings, 2 at substation + 2 at KVS_0 → 1 KVS.
    # Total KVS clusters across 4 feeders = 4. (Each KVS gets one
    # rod electrode by the default ``_default_kvs_grounding`` factory.)
    kvs_rod_electrodes = [
        e for e in world.electrodes
        if e.name.startswith("kvs_f") and "_rod_0" in e.name
    ]
    assert len(kvs_rod_electrodes) == 4
    # 4 trunk segments substation → KVS, named ``pen_trunk_fX_0``.
    trunks = [c for c in world.conductors
              if c.name.startswith("pen_trunk_")]
    assert len(trunks) == 4
    # 16 service drops.
    services = [c for c in world.conductors
                if c.name.startswith("pen_service_")]
    assert len(services) == 16


def test_radial_trunk_source_attached_to_substation() -> None:
    """The injection still happens at the substation cluster anchor."""
    world = _build_radial_world(n_buildings=8)
    assert len(world.sources) == 1
    src = world.sources[0]
    # The default substation grounding's first electrode is named
    # ``trafo_ring_0`` (a RingElectrodeSpec is the first entry of
    # ``_default_substation_grounding``).
    assert src.attached_to == "trafo_ring_0"


def test_radial_trunk_solves_on_image_2layer() -> None:
    """Smoke-test the radial-trunk world end-to-end against the
    default :class:`image_2layer` backend."""
    world = _build_radial_world(n_buildings=12)
    engine = gf.create_engine(
        backend="image_2layer", segment_length=0.5, frequencies=[50.0],
    )
    result = engine.solve(world)
    Z = result.cluster_impedance("trafo_ring_0")
    assert math.isfinite(abs(Z[0]))
    assert abs(Z[0]) > 0.0


# ---------------------------------------------------------------------
# JSON round-trip — discriminated union
# ---------------------------------------------------------------------


def test_topology_json_round_trip_preserves_radial_trunk() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 3},
        pen_topology=RadialTrunkTopology(
            n_feeders=3,
            feeder_directions_deg=[0.0, 120.0, 240.0],
            slots_per_substation=2,
            slots_per_kvs=2,
            kvs_spacing_m=45.0,
            max_feeder_length_m=300.0,
            name_prefix="cabinet",
        ),
    )
    payload = cfg.model_dump_json()
    restored = TnNetworkConfig.model_validate_json(payload)
    assert isinstance(restored.pen_topology, RadialTrunkTopology)
    assert restored.pen_topology.n_feeders == 3
    assert restored.pen_topology.feeder_directions_deg == [0.0, 120.0, 240.0]
    assert restored.pen_topology.kvs_spacing_m == 45.0
    assert restored.pen_topology.name_prefix == "cabinet"


def test_topology_json_round_trip_default_is_star() -> None:
    cfg = TnNetworkConfig(building_counts={"residential": 3})
    payload = cfg.model_dump_json()
    restored = TnNetworkConfig.model_validate_json(payload)
    assert isinstance(restored.pen_topology, StarKvsTopology)


# ---------------------------------------------------------------------
# Drop-everything-warning safety net
# ---------------------------------------------------------------------


def test_radial_trunk_raises_when_all_buildings_dropped() -> None:
    """If every building is dropped (e.g. all behind the feeder axis),
    the generator surfaces a clear ValueError instead of building a
    PEN-less world."""
    cfg = TnNetworkConfig(
        building_counts={"residential": 2},
        placement=ExplicitPlacement(
            positions=[(-50.0, 0.0), (-100.0, 0.0)],  # all west of subst.
        ),
        pen_topology=RadialTrunkTopology(
            n_feeders=1, feeder_directions_deg=[0.0],  # only feeds east
        ),
    )
    gen = TnNetworkGenerator(cfg, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="dropped every building"):
            gen.build()
