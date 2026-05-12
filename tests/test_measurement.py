"""Tests for the measurement-setup layer (AP1 Analysis 1 + 2).

Validates the additions of ADR-0009 v3:

* enabling ``cfg.measurement`` builds the auxiliary current
  electrode and the voltage probe at their configured positions;
* the source's ``return_to`` is wired to the auxiliary anchor when
  the measurement setup is active;
* ``feed_lead`` and ``probe.lead`` add the metallic measurement
  conductors when set, and are absent when ``None``;
* ``overhead_lead`` / ``buried_lead`` factories produce sensible
  defaults;
* JSON round-trip preserves the full nested measurement config.
"""

from __future__ import annotations

import math

import pytest

import groundfield as gf
from groundfield.generators import (
    MeasurementInjectionConfig,
    MeasurementLeadConfig,
    MeasurementProbeConfig,
    MeasurementSetupConfig,
    Normal,
    TnNetworkConfig,
    TnNetworkGenerator,
    Uniform,
    buried_lead,
    neighbour_substation_grounding,
    overhead_lead,
    single_rod_grounding,
)


# ---------------------------------------------------------------------
# Default behaviour: measurement=None keeps the historic shape
# ---------------------------------------------------------------------


def test_no_measurement_means_no_aux_no_probe() -> None:
    """``cfg.measurement = None`` (default) does not add any aux/probe."""
    cfg = TnNetworkConfig(building_counts={"residential": 3})
    world = TnNetworkGenerator(cfg, seed=0).build()
    aux_count = sum(1 for e in world.electrodes if e.name.startswith("aux_"))
    probe_count = sum(1 for e in world.electrodes if e.name.startswith("probe_"))
    assert aux_count == 0
    assert probe_count == 0
    # The source returns to None (remote earth) when no measurement.
    src = world.sources[0]
    assert src.return_to is None


# ---------------------------------------------------------------------
# Galvanic-only setup (Analysis 1): aux + probe, but no leads
# ---------------------------------------------------------------------


def test_galvanic_only_setup_adds_aux_and_probe_without_leads() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 3},
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(150.0, 0.0),
            ),
            probe=MeasurementProbeConfig(
                position_xy=(50.0, 0.0),
            ),
        ),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    aux_rods = [e for e in world.electrodes if e.name.startswith("aux_rod_")]
    probe_rods = [e for e in world.electrodes if e.name.startswith("probe_rod_")]
    assert len(aux_rods) == 1
    assert len(probe_rods) == 1
    # Position check
    ax, ay, _ = aux_rods[0].position
    px, py, _ = probe_rods[0].position
    assert math.isclose(ax, 150.0, abs_tol=1e-9)
    assert math.isclose(ay, 0.0, abs_tol=1e-9)
    assert math.isclose(px, 50.0, abs_tol=1e-9)
    assert math.isclose(py, 0.0, abs_tol=1e-9)
    # No metallic leads
    feed_leads = [c for c in world.conductors if c.name == "meas_feed_lead"]
    probe_leads = [c for c in world.conductors if c.name == "meas_probe_lead"]
    assert feed_leads == []
    assert probe_leads == []


def test_source_return_path_points_to_aux_anchor() -> None:
    """When measurement is set, the source returns through aux."""
    cfg = TnNetworkConfig(
        building_counts={"residential": 3},
        measurement=MeasurementSetupConfig(),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    src = world.sources[0]
    assert src.return_to is not None
    assert src.return_to.startswith("aux_")


# ---------------------------------------------------------------------
# Inductive-coupling setup (Analysis 2): aux + probe + both leads
# ---------------------------------------------------------------------


def test_inductive_setup_adds_both_leads() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 3},
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(150.0, 0.0),
                feed_lead=overhead_lead(),
            ),
            probe=MeasurementProbeConfig(
                position_xy=(50.0, 0.0),
                lead=overhead_lead(),
            ),
        ),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    feed = [c for c in world.conductors if c.name == "meas_feed_lead"]
    probe = [c for c in world.conductors if c.name == "meas_probe_lead"]
    assert len(feed) == 1
    assert len(probe) == 1
    # Both leads have the inductance model active
    assert feed[0].inductance_model == "neumann"
    assert probe[0].inductance_model == "neumann"


def test_buried_lead_helper_uses_cable_shield_and_finite_depth() -> None:
    lead = buried_lead(depth_m=0.6)
    assert lead.depth_m == 0.6
    assert lead.conductor_type == "cable_shield"
    assert lead.coupling_to_soil == "isolated"
    assert lead.inductance_model == "neumann"


def test_overhead_lead_helper_is_surface_bare_copper() -> None:
    lead = overhead_lead()
    assert lead.depth_m == 0.0
    assert lead.conductor_type == "bare_copper"
    assert lead.coupling_to_soil == "isolated"
    assert lead.inductance_model == "neumann"


# ---------------------------------------------------------------------
# Aux electrode variant: neighbour-substation grounding
# ---------------------------------------------------------------------


def test_neighbour_substation_grounding_adds_ring_and_rods() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 3},
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(200.0, 0.0),
                grounding=neighbour_substation_grounding(),
            ),
        ),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    aux_ring = [e for e in world.electrodes if e.name.startswith("aux_ring_")]
    aux_rods = [e for e in world.electrodes if e.name.startswith("aux_rod_")]
    assert len(aux_ring) == 1
    assert len(aux_rods) == 4


# ---------------------------------------------------------------------
# Smallest preset still solves with measurement enabled
# ---------------------------------------------------------------------


def test_galvanic_setup_solves() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 3},
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(position_xy=(80.0, 0.0)),
            probe=MeasurementProbeConfig(position_xy=(40.0, 0.0)),
        ),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    engine = gf.create_engine(
        backend="image_2layer", segment_length=1.0, frequencies=[50.0],
    )
    res = engine.solve(world)
    Z = res.cluster_impedance("trafo_ring_0")[0]
    assert math.isfinite(abs(Z))


# ---------------------------------------------------------------------
# Source magnitude — fault / measurement current
# ---------------------------------------------------------------------


def test_source_magnitude_passthrough() -> None:
    """``source_magnitude_A`` is forwarded to the actual ``CurrentSource``."""
    cfg = TnNetworkConfig(
        building_counts={"residential": 3},
        source_magnitude_A=5000.0,  # ~ single-phase fault current
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    src = world.sources[0]
    assert src.magnitude == 5000.0


def test_source_magnitude_accepts_distribution() -> None:
    """``source_magnitude_A`` accepts a Distribution; build samples it once."""
    from groundfield.generators import Discrete
    cfg = TnNetworkConfig(
        building_counts={"residential": 3},
        source_magnitude_A=Discrete(values=[1.0, 5000.0]),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    src = world.sources[0]
    assert src.magnitude in {1.0, 5000.0}


def test_source_magnitude_seed_reproduces_draw() -> None:
    from groundfield.generators import Discrete
    cfg = TnNetworkConfig(
        building_counts={"residential": 3},
        source_magnitude_A=Discrete(values=[1.0, 10.0, 100.0, 1000.0, 5000.0]),
    )
    a = TnNetworkGenerator(cfg, seed=7).build().sources[0].magnitude
    b = TnNetworkGenerator(cfg, seed=7).build().sources[0].magnitude
    assert a == b


# ---------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------


def test_measurement_config_json_roundtrip() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 5},
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(150.0, 5.0),
                grounding=single_rod_grounding(length_m=Uniform(low=1.0, high=2.0)),
                feed_lead=overhead_lead(),
            ),
            probe=MeasurementProbeConfig(
                position_xy=(60.0, 0.0),
                lead=buried_lead(depth_m=0.6),
            ),
        ),
    )
    payload = cfg.model_dump_json()
    restored = TnNetworkConfig.model_validate_json(payload)
    assert restored.measurement is not None
    assert restored.measurement.injection.position_xy == (150.0, 5.0)
    assert isinstance(restored.measurement.injection.feed_lead,
                      MeasurementLeadConfig)
    assert isinstance(restored.measurement.probe.lead,
                      MeasurementLeadConfig)
    assert restored.measurement.injection.feed_lead.conductor_type == "bare_copper"
    assert restored.measurement.probe.lead.conductor_type == "cable_shield"
