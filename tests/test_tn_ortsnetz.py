"""Tests for the AP1 TN low-voltage distribution network generator.

Tests live under the old AP1 file name (``test_tn_ortsnetz.py``)
because the AP1 work-package keeps its German name *TN-Ortsnetz*.
The class names follow the project-wide English convention
(``TnNetworkGenerator`` / ``TnNetworkConfig``).

Validation programme of ADR-0009:

* a default config builds and solves on ``image_2layer``;
* multi-electrode substation grounding (ring + rods + strip +
  foundation) builds without error;
* multiple building types each produce their configured electrodes;
* presence_prob = 0 keeps the matching electrode out of the world;
* explicit-placement coordinates are honoured;
* JSON round-trip preserves the full nested config including
  every distribution kind;
* the deprecation alias still imports the new symbols.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import (
    BuildingTypeSpec,
    Categorical,
    Discrete,
    ExplicitPlacement,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    HomogeneousSoilSpec,
    LogNormal,
    ManhattanGridPlacement,
    MultiLayerSoilSpec,
    Normal,
    PenConfig,
    RingElectrodeSpec,
    RodElectrodeSpec,
    SoilLayerSpec,
    StripElectrodeSpec,
    SubstationConfig,
    KvsConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
    Uniform,
    rod_circle,
)


# ---------------------------------------------------------------------
# Default config — sanity build + solve
# ---------------------------------------------------------------------


def test_default_config_builds_and_solves_minimal() -> None:
    """Smallest sensible run: 5 residential houses, default everything else."""
    cfg = TnNetworkConfig(building_counts={"residential": 5})
    gen = TnNetworkGenerator(cfg, seed=0)
    world = gen.build()

    engine = gf.create_engine(
        backend="image_2layer", segment_length=0.5, frequencies=[50.0],
    )
    result = engine.solve(world)
    Z = result.cluster_impedance("trafo_ring_0")
    assert math.isfinite(abs(Z[0]))
    assert abs(Z[0]) > 0.0


# ---------------------------------------------------------------------
# Substation: ring + rods + strip + foundation, in any combination
# ---------------------------------------------------------------------


def test_substation_with_ring_rods_strip_foundation() -> None:
    """Substation grounding: ring + 4 rods + strip + foundation, all present."""
    substation = SubstationConfig(
        position=(0.0, 0.0),
        grounding=GroundingSystemSpec(
            electrodes=[
                RingElectrodeSpec(radius_m=4.0, depth_m=0.6),
                *rod_circle(n=4, radius_m=2.0, length_m=2.5),
                StripElectrodeSpec(length_m=20.0, depth_m=0.6,
                                   orientation_deg=45.0),
                FoundationElectrodeSpec(size_m=8.0, depth_m=0.8,
                                        n_x=2, n_y=2),
            ],
        ),
    )
    cfg = TnNetworkConfig(
        substation=substation,
        building_counts={"residential": 5},
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    # 1 ring + 4 rods + 1 strip + 1 foundation = 7 trafo electrodes
    n_trafo = sum(1 for e in world.electrodes if e.name.startswith("trafo_"))
    assert n_trafo == 7


def test_substation_presence_prob_can_drop_electrodes() -> None:
    """presence_prob = 0 keeps an electrode out."""
    substation = SubstationConfig(
        grounding=GroundingSystemSpec(
            electrodes=[
                RingElectrodeSpec(radius_m=4.0, presence_prob=1.0),
                StripElectrodeSpec(length_m=20.0, presence_prob=0.0),
            ],
        ),
    )
    cfg = TnNetworkConfig(
        substation=substation, building_counts={"residential": 3},
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    n_trafo_strip = sum(1 for e in world.electrodes
                        if e.name.startswith("trafo_strip_"))
    assert n_trafo_strip == 0
    n_trafo_ring = sum(1 for e in world.electrodes
                       if e.name.startswith("trafo_ring_"))
    assert n_trafo_ring == 1


# ---------------------------------------------------------------------
# Building types: multiple categories with distinct grounding
# ---------------------------------------------------------------------


def test_multiple_building_types_each_get_configured_grounding() -> None:
    catalog = [
        BuildingTypeSpec(
            name="residential",
            grounding=GroundingSystemSpec(electrodes=[
                FoundationElectrodeSpec(size_m=10.0),
            ]),
        ),
        BuildingTypeSpec(
            name="industry",
            grounding=GroundingSystemSpec(electrodes=[
                FoundationElectrodeSpec(size_m=20.0, n_x=3, n_y=3),
                *rod_circle(n=4, radius_m=12.0),
            ]),
        ),
    ]
    cfg = TnNetworkConfig(
        building_types=catalog,
        building_counts={"residential": 5, "industry": 2},
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    # 5 residential foundations
    n_res = sum(1 for e in world.electrodes
                if e.name.startswith("residential_") and e.name.endswith("_foundation_0"))
    assert n_res == 5
    # 2 industry buildings × (1 foundation + 4 rods) = 10 industry electrodes
    n_ind_foundation = sum(1 for e in world.electrodes
                           if e.name.startswith("industry_") and "foundation_" in e.name)
    n_ind_rod = sum(1 for e in world.electrodes
                    if e.name.startswith("industry_") and "_rod_" in e.name)
    assert n_ind_foundation == 2
    assert n_ind_rod == 8


def test_unknown_building_type_in_counts_raises() -> None:
    cfg = TnNetworkConfig(
        building_counts={"no_such_type": 5},
    )
    with pytest.raises(KeyError, match="no matching entry"):
        TnNetworkGenerator(cfg, seed=0).build()


# ---------------------------------------------------------------------
# Placement strategies
# ---------------------------------------------------------------------


def test_explicit_placement_honoured() -> None:
    """Buildings placed at the exact coordinates given."""
    positions = [(50.0, 50.0), (50.0, 100.0), (100.0, 50.0)]
    cfg = TnNetworkConfig(
        placement=ExplicitPlacement(positions=positions),
        building_counts={"residential": 3},
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    foundations = [e for e in world.electrodes if "_foundation_" in e.name
                   and e.name.startswith("residential_")]
    assert len(foundations) == 3
    centres = sorted(((e.corner[0] + e.size[0] / 2.0,
                       e.corner[1] + e.size[1] / 2.0) for e in foundations))
    assert centres == sorted(positions)


def test_manhattan_grid_default_centred_at_origin() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 4},
        placement=ManhattanGridPlacement(spacing_x_m=10.0, spacing_y_m=10.0,
                                          n_per_row=2),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    foundations = [e for e in world.electrodes if "_foundation_" in e.name
                   and e.name.startswith("residential_")]
    cx = sum(e.corner[0] + e.size[0] / 2.0 for e in foundations) / len(foundations)
    cy = sum(e.corner[1] + e.size[1] / 2.0 for e in foundations) / len(foundations)
    assert math.isclose(cx, 0.0, abs_tol=1e-9)
    assert math.isclose(cy, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------
# Soil
# ---------------------------------------------------------------------


def test_homogeneous_soil_works() -> None:
    cfg = TnNetworkConfig(
        soil=HomogeneousSoilSpec(resistivity=200.0),
        building_counts={"residential": 3},
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    assert world.soil.kind == "homogeneous"
    assert world.soil.resistivity == 200.0


def test_multi_layer_soil_works() -> None:
    cfg = TnNetworkConfig(
        soil=MultiLayerSoilSpec(layers=[
            SoilLayerSpec(resistivity=300.0, thickness_m=2.0),
            SoilLayerSpec(resistivity=100.0, thickness_m=5.0),
            SoilLayerSpec(resistivity=50.0, thickness_m=None),
        ]),
        building_counts={"residential": 3},
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    assert world.soil.kind == "multi_layer"
    assert len(world.soil.layers) == 3


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def test_seed_reproduces_world_bit_exact() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 5},
        soil=TwoLayerSoilSpec(
            rho_1=LogNormal(mu=5.0, sigma=0.5),
            rho_2=50.0, h_1=5.0,
        ),
    )
    g1 = TnNetworkGenerator(cfg, seed=42)
    g2 = TnNetworkGenerator(cfg, seed=42)
    w1 = g1.build()
    w2 = g2.build()
    engine = gf.create_engine(backend="image_2layer", segment_length=1.0,
                              frequencies=[50.0])
    Z1 = engine.solve(w1).cluster_impedance("trafo_ring_0")[0]
    Z2 = engine.solve(w2).cluster_impedance("trafo_ring_0")[0]
    assert Z1 == Z2


# ---------------------------------------------------------------------
# JSON round-trip on a heavy stochastic config
# ---------------------------------------------------------------------


def test_full_config_json_roundtrip_with_distributions() -> None:
    cfg = TnNetworkConfig(
        soil=TwoLayerSoilSpec(
            rho_1=LogNormal(mu=5.0, sigma=0.7),
            rho_2=Uniform(low=20.0, high=80.0),
            h_1=Discrete(values=[5.0, 10.0, 30.0]),
        ),
        substation=SubstationConfig(
            grounding=GroundingSystemSpec(electrodes=[
                RingElectrodeSpec(radius_m=Normal(mean=4.0, std=0.4,
                                                  truncate_low=2.5,
                                                  truncate_high=6.0)),
                *rod_circle(n=4, radius_m=2.0, length_m=2.5),
            ]),
        ),
        kvs=KvsConfig(
            quote_per_100_buildings=Uniform(low=4.0, high=8.0),
            grounding=GroundingSystemSpec(electrodes=[
                RodElectrodeSpec(length_m=Uniform(low=1.0, high=2.0)),
            ]),
        ),
        building_counts={"residential": Discrete(values=[5, 10])},
    )
    payload = cfg.model_dump_json()
    restored = TnNetworkConfig.model_validate_json(payload)
    # Spot-check: every distribution survives via discriminated union
    assert isinstance(restored.soil, TwoLayerSoilSpec)
    assert isinstance(restored.soil.rho_1, LogNormal)
    assert isinstance(restored.substation.grounding.electrodes[0],
                      RingElectrodeSpec)
    assert isinstance(restored.substation.grounding.electrodes[0].radius_m,
                      Normal)
    assert isinstance(restored.kvs.quote_per_100_buildings, Uniform)
    assert isinstance(restored.building_counts["residential"], Discrete)


# ---------------------------------------------------------------------
# KVS quota
# ---------------------------------------------------------------------


def test_kvs_count_follows_quote() -> None:
    """50 residentials, quote=10/100 → 5 KVS."""
    cfg = TnNetworkConfig(
        building_counts={"residential": 50},
        kvs=KvsConfig(
            quote_per_100_buildings=10.0,
            placement=ManhattanGridPlacement(
                spacing_x_m=40.0, spacing_y_m=1.0, n_per_row=10,
            ),
        ),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    n_kvs_rod = sum(1 for e in world.electrodes
                    if e.name.startswith("kvs_") and "_rod_" in e.name)
    assert n_kvs_rod == 5


def test_fixed_kvs_count_overrides_quote() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 100},
        kvs=KvsConfig(
            fixed_count=3,
            quote_per_100_buildings=10.0,  # would yield 10 KVS, but ignored
            placement=ManhattanGridPlacement(
                spacing_x_m=40.0, spacing_y_m=1.0, n_per_row=5,
            ),
        ),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    n_kvs_rod = sum(1 for e in world.electrodes
                    if e.name.startswith("kvs_") and "_rod_" in e.name)
    assert n_kvs_rod == 3


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


def test_zero_buildings_raises() -> None:
    cfg = TnNetworkConfig(building_counts={})
    with pytest.raises(ValueError, match="building count"):
        TnNetworkGenerator(cfg, seed=0).build()


def test_substation_with_zero_present_electrodes_raises() -> None:
    cfg = TnNetworkConfig(
        substation=SubstationConfig(
            grounding=GroundingSystemSpec(electrodes=[
                RodElectrodeSpec(length_m=1.5, presence_prob=0.0),
            ]),
        ),
        building_counts={"residential": 3},
    )
    with pytest.raises(ValueError, match="substation grounding"):
        TnNetworkGenerator(cfg, seed=0).build()


# ---------------------------------------------------------------------
# Deprecation alias
# ---------------------------------------------------------------------


def test_deprecated_module_alias_still_imports() -> None:
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from groundfield.generators.tn_ortsnetz import (  # noqa: F401
            TnOrtsnetzGenerator,
            TnOrtsnetzConfig,
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert TnOrtsnetzGenerator is TnNetworkGenerator
    assert TnOrtsnetzConfig is TnNetworkConfig
