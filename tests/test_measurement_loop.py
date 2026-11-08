r"""Simulated fall-of-potential measurement of a generated TN network.

A real earth tester is wired as

* current loop: earthing system (terminal E) -> soil -> auxiliary
  electrode -> feed lead -> terminal H -> current source -> E;
* voltage path: voltmeter between E and S, probe lead from S to the
  voltage probe, no current through the voltmeter.

Up to v0.15.0 the generator ran both leads from the substation electrode
to the remote electrode: the probe lead bonded the probe to the
substation (the voltmeter read zero) and the feed lead closed the loop
metallically (the test current did not reach the soil). Since 2026.11 the
leads end on leakage-free junction nodes at the instrument, ``meas_H``
and ``meas_S``, and the source drives its current between the substation
and ``meas_H`` (review finding 10.17). The reading is

.. math::

   Z_\text{meas} = \frac{\varphi_E - \varphi_S}{I},

returned by :func:`groundfield.generators.measured_impedance`.
"""

from __future__ import annotations

import math
import warnings

import pytest

import groundfield as gf
from groundfield.generators import (
    HomogeneousSoilSpec,
    MeasurementInjectionConfig,
    MeasurementLeadConfig,
    MeasurementProbeConfig,
    MeasurementSetupConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
    overhead_lead,
)
from groundfield.generators.measurement import (
    MEASUREMENT_H_TERMINAL,
    MEASUREMENT_S_TERMINAL,
    measured_impedance,
)

pytestmark = pytest.mark.finding("10.17")

SUBSTATION = "trafo_ring_0"


def _resistive_lead(**kw: object) -> MeasurementLeadConfig:
    """Overhead lead without inductance: a pure series resistance."""
    return overhead_lead(inductance_model=None, **kw)


def _world(
    *,
    feed: MeasurementLeadConfig | None = None,
    probe_lead: MeasurementLeadConfig | None = None,
    d_aux: float = 200.0,
    probe_fraction: float = 0.618,
    houses: int = 3,
) -> gf.World:
    cfg = TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=100.0),
        building_counts={"residential": houses},
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(d_aux, 0.0), feed_lead=feed
            ),
            probe=MeasurementProbeConfig(
                position_xy=(probe_fraction * d_aux, 0.0), lead=probe_lead
            ),
        ),
    )
    return TnNetworkGenerator(cfg, seed=1).build()


def _solve(world: gf.World, *, f: float = 50.0, **kw: object) -> gf.FieldResult:
    engine = gf.create_engine(
        backend="image", segment_length=2.0, frequencies=[f], kernel="point", **kw
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.solve(world)


# ---------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------


def test_feed_lead_runs_from_the_instrument_to_the_auxiliary_electrode() -> None:
    world = _world(feed=_resistive_lead())
    lead = world.get_conductor("meas_feed_lead")
    h = world.get_junction(MEASUREMENT_H_TERMINAL)
    assert (lead.start_electrode, lead.end_electrode) == (h.name, "aux_rod_0")
    assert lead.start == h.position
    (source,) = world.sources
    assert source.attached_to == SUBSTATION
    assert source.return_to == MEASUREMENT_H_TERMINAL


def test_probe_lead_runs_from_the_instrument_to_the_probe() -> None:
    world = _world(probe_lead=_resistive_lead())
    lead = world.get_conductor("meas_probe_lead")
    s = world.get_junction(MEASUREMENT_S_TERMINAL)
    assert (lead.start_electrode, lead.end_electrode) == (s.name, "probe_rod_0")
    # No feed lead: the source returns straight through the auxiliary electrode.
    (source,) = world.sources
    assert source.return_to == "aux_rod_0"
    assert [j.name for j in world.junctions] == [MEASUREMENT_S_TERMINAL]


def test_galvanic_setup_has_no_instrument_junctions() -> None:
    world = _world()
    assert world.junctions == []
    assert world.sources[0].return_to == "aux_rod_0"


# ---------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------


def test_resistive_leads_do_not_change_the_reading() -> None:
    """The feed lead is in series with an ideal current source and the
    probe lead carries no current, so non-inductive leads leave the
    reading unchanged. v0.15.0 short-circuited the voltmeter
    (1.46 -> 2.5e-4 ohm, review 10.17)."""
    z_bare = measured_impedance(_solve(_world()))[0]
    z_leads = measured_impedance(
        _solve(_world(feed=_resistive_lead(), probe_lead=_resistive_lead()))
    )[0]
    assert abs(z_bare) > 0.1
    assert z_leads == pytest.approx(z_bare, rel=1e-9, abs=1e-12)


def test_feed_lead_adds_its_resistance_to_the_loop_impedance() -> None:
    """``input_impedance`` of the measurement source is the two-pole
    reading E-H: soil loop plus the feed-lead resistance."""
    lead = _resistive_lead()
    world = _world(feed=lead)
    r_lead = world.get_conductor("meas_feed_lead").series_resistance
    z_loop = _solve(world).input_impedance()[0]
    z_soil = _solve(_world()).input_impedance()[0]
    assert r_lead > 1e-3
    assert z_loop - z_soil == pytest.approx(r_lead, rel=1e-8)


def test_two_terminal_reading_is_the_superposition_of_one_terminal_solves() -> None:
    """Independent check of the loop (audit H3): with 1 A injected at the
    substation and taken back at the auxiliary electrode,
    ``phi_E - phi_S`` equals the difference of two remote-earth solves."""
    world = _world()
    z_meas = measured_impedance(_solve(world))[0]

    def one_terminal(attached_to: str) -> gf.FieldResult:
        w = world.model_copy(deep=True)
        w.sources.clear()
        gf.create_source(w, attached_to=attached_to, magnitude=1.0)
        return _solve(w)

    res_e = one_terminal(SUBSTATION)
    res_a = one_terminal("aux_rod_0")

    def phi(res: gf.FieldResult, name: str) -> complex:
        return res.electrode_potentials[name][0]

    expected = (phi(res_e, SUBSTATION) - phi(res_a, SUBSTATION)) - (
        phi(res_e, "probe_rod_0") - phi(res_a, "probe_rod_0")
    )
    assert z_meas == pytest.approx(expected, rel=1e-9)


def test_inductive_leads_shift_but_do_not_short_the_reading() -> None:
    """Neumann leads couple the feed loop into the probe lead (the
    classical inductive error of earth testers): a finite, reactive shift
    of the reading (+0.136j ohm at 50 Hz for the two overhead leads laid
    side by side here), not a short circuit."""
    z_bare = measured_impedance(_solve(_world()))[0]
    world = _world(feed=overhead_lead(), probe_lead=overhead_lead())
    z_ind = measured_impedance(_solve(world, earth_inductive_model="perfect_mirror"))[0]
    shift = abs(z_ind - z_bare) / abs(z_bare)
    assert 1e-3 < shift < 0.5
    assert z_ind.imag > abs(z_bare.imag)
    # The mirror earth return adds no loss: the galvanic part is unchanged.
    assert z_ind.real == pytest.approx(z_bare.real, rel=1e-9)


def test_in_line_leads_lie_side_by_side() -> None:
    """In-line probe (the 61.8 % set-up): both leads leave the instrument
    towards the same side. Laid through each other, their identical 5 m
    segments had an unbounded Neumann mutual inductance (+586.8j ohm at
    50 Hz for the golden geometry); the voltage terminal therefore sits
    the touching distance (sum of the wire radii) beside the current
    terminal."""
    feed, probe = overhead_lead(), overhead_lead(wire_radius_m=0.002)
    world = _world(feed=feed, probe_lead=probe, d_aux=150.0, probe_fraction=1 / 3)
    h = world.get_junction(MEASUREMENT_H_TERMINAL).position
    s = world.get_junction(MEASUREMENT_S_TERMINAL).position
    assert math.dist(h, s) == pytest.approx(0.007, rel=1e-12)
    z = measured_impedance(_solve(world, earth_inductive_model="perfect_mirror"))[0]
    assert 0.0 < z.imag < 0.1


def test_reading_approaches_the_station_impedance_at_61_8_percent() -> None:
    """Fall-of-potential rule (Tagg): with the probe at 61.8 % of a remote
    auxiliary electrode the reading equals the impedance of the network
    against remote earth."""
    world = _world(d_aux=2000.0, probe_fraction=0.618)
    z_meas = measured_impedance(_solve(world))[0]
    reference = world.model_copy(deep=True)
    reference.sources.clear()
    gf.create_source(reference, attached_to=SUBSTATION, magnitude=1.0)
    z_ref = _solve(reference).input_impedance()[0]
    assert z_meas.real == pytest.approx(z_ref.real, rel=5e-3)


# ---------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------


def test_measured_impedance_reads_the_probe_terminal_or_the_probe_electrode() -> None:
    res_lead = _solve(_world(probe_lead=_resistive_lead()))
    explicit = measured_impedance(res_lead, MEASUREMENT_S_TERMINAL)
    assert measured_impedance(res_lead) == explicit
    res_bare = _solve(_world())
    assert measured_impedance(res_bare) == measured_impedance(res_bare, "probe_rod_0")
    # The probe at the substation reads zero.
    assert measured_impedance(res_bare, SUBSTATION) == [0j]


def test_measured_impedance_scales_with_the_source_current() -> None:
    world = _world()
    world.sources[0].magnitude = 5.0
    z5 = measured_impedance(_solve(world))[0]
    world.sources[0].magnitude = 1.0
    z1 = measured_impedance(_solve(world))[0]
    assert z5 == pytest.approx(z1, rel=1e-12)


def test_measured_impedance_rejects_unknown_probes_and_missing_sources() -> None:
    res = _solve(_world())
    with pytest.raises(KeyError, match="no_such_probe"):
        measured_impedance(res, "no_such_probe")
    with pytest.raises(KeyError, match="no_such_source"):
        measured_impedance(res, source="no_such_source")
    hand_built = gf.FieldResult(backend="image", frequencies=[50.0])
    with pytest.raises(ValueError, match="source"):
        measured_impedance(hand_built, "probe_rod_0")


def test_source_return_override_still_warns() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 2},
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(feed_lead=_resistive_lead())
        ),
        source_return_to="kvs_0_rod_0",
    )
    with pytest.warns(UserWarning, match="meas_H"):
        world = TnNetworkGenerator(cfg, seed=0).build()
    assert world.sources[0].return_to == "kvs_0_rod_0"
    assert math.isfinite(_solve(world).input_impedance()[0].real)


# ---------------------------------------------------------------------
# The layout builder (probe = ideal sampling point, no leads)
# ---------------------------------------------------------------------


def _layout(d_aux: float) -> "gf.generators.LvNetworkLayout":
    from groundfield.generators import LvNetworkLayout

    footprints = [
        gf.BuildingFootprint(
            polygon_xy_m=[
                (cx - 5.0, 13.0),
                (cx + 5.0, 13.0),
                (cx + 5.0, 23.0),
                (cx - 5.0, 23.0),
            ],
            osm_id=10 + i,
        )
        for i, cx in enumerate((20.0, 45.0, 70.0))
    ]
    layout = LvNetworkLayout.from_footprints(footprints, substation_xy=(0.0, 0.0))
    layout.add_pen_cable(start="substation", end_xy=(90.0, 0.0))
    layout.connect_buildings()
    layout.add_auxiliary_electrode(distance_m=d_aux, direction_deg=180.0)
    return layout


def test_layout_reading_uses_the_solved_source_current() -> None:
    """Up to 0.15.0 the reading divided by ``source_magnitude_a=1.0``
    unless the caller repeated the build's test current."""
    layout = _layout(300.0)
    layout.add_voltage_probe(inline_fraction=0.618)
    soil = gf.HomogeneousSoil(rho=100.0)
    z = {}
    for amps in (1.0, 5.0):
        world = layout.to_world(soil=soil, source_magnitude_a=amps, seed=0)
        res = _solve(world)
        z[amps] = layout.measured_grounding_impedance(res)
        balance = layout.verify_current_balance(res)
        assert balance["ok"] and balance["expected_positive_a"] == amps
    assert z[5.0] == pytest.approx(z[1.0], rel=1e-12)
    # An explicit current still overrides the solved one.
    assert layout.measured_grounding_impedance(
        res, source_magnitude_a=1.0
    ) == pytest.approx(5.0 * z[1.0], rel=1e-12)


def test_layout_reading_follows_the_61_8_percent_rule() -> None:
    layout = _layout(2000.0)
    layout.add_voltage_probe(inline_fraction=0.618)
    soil = gf.HomogeneousSoil(rho=100.0)
    z_meas = layout.measured_grounding_impedance(
        _solve(layout.to_world(soil=soil, seed=0))
    )
    reference = layout.to_world(soil=soil, seed=0)
    reference.sources[0].return_to = None  # remote earth
    z_ref = _solve(reference).input_impedance()[0]
    assert z_meas.real == pytest.approx(z_ref.real, rel=5e-3)
