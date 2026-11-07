"""Leakage-free junction nodes (prerequisite of findings 10.17 and 10.38).

A junction is a named network node without soil contact: conductors
start or end there and sources attach to it, its potential follows from
Kirchhoff's current law of the conductor network, and it injects no
current into the soil. Up to 0.15.0 the only nodes were electrodes, so
generators modelled cable joints as 5 cm pseudo-rods that leak 3–7 % of
the test current (10.38), and measurement leads had to end on an
electrode, which short-circuits the instrument (10.17).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

pytestmark = pytest.mark.finding("10.38", "10.17")

SEG = 0.5


def _soil(rho: float = 100.0) -> gf.HomogeneousSoil:
    """Homogeneous soil; ``resistivity`` was the field name up to 0.15.0."""
    try:
        return gf.HomogeneousSoil(rho=rho)
    except Exception:  # noqa: BLE001 - v0.15.0 names the field ``resistivity``
        return gf.HomogeneousSoil(resistivity=rho)


def _create_junction(world, **kw):
    from groundfield.api import create_junction

    return create_junction(world, **kw)


def _solve(world: gf.World, **engine_kw) -> gf.FieldResult:
    kw = {"segment_length": SEG, "frequencies": [50.0], **engine_kw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world.solve(gf.create_engine(**kw))


def _junction_potential(res: gf.FieldResult, name: str, k: int = 0) -> complex:
    return complex(res.metadata["junction_potentials"][name][k])


def _rod_world() -> gf.World:
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
    return w


def _lead(world, name, start, end, *, r_ohm: float = 0.5, **kw):
    """Finite lumped lead with the series resistance ``r_ohm``."""
    return gf.create_conductor(
        world,
        name=name,
        start=start,
        end=end,
        conductor_type="generic",
        cross_section=50e-6,
        lumped_series_resistance_ohm=r_ohm,
        **kw,
    )


# ---------------------------------------------------------------------
# model and API
# ---------------------------------------------------------------------


def test_create_junction_registers_a_named_node() -> None:
    from groundfield.geometry.junction import Junction

    w = _rod_world()
    j = _create_junction(w, name="J", position=(1.0, 0.0, 0.5))
    assert isinstance(j, Junction)
    assert w.junctions == [j]
    assert w.get_junction("J") is j
    assert j.connection_point == (1.0, 0.0, 0.5)
    auto = _create_junction(w, position=(2.0, 0.0, 0.5))
    assert auto.name == "junction_0"


def test_node_names_are_unique_across_electrodes_and_junctions() -> None:
    w = _rod_world()
    with pytest.raises(ValueError, match="already taken"):
        _create_junction(w, name="g1", position=(1.0, 0.0, 0.5))
    _create_junction(w, name="J", position=(1.0, 0.0, 0.5))
    with pytest.raises(ValueError, match="already taken"):
        gf.create_electrode(w, "rod", name="J", position=(5.0, 0.0, 0.0), length=1.0)


def test_junction_position_must_be_finite() -> None:
    with pytest.raises(ValueError, match="position"):
        _create_junction(_rod_world(), name="J", position=(math.nan, 0.0, 0.5))


def test_world_json_is_unchanged_without_junctions_and_round_trips_with_them() -> None:
    w = _rod_world()
    assert "junctions" not in w.model_dump(mode="json")
    _create_junction(w, name="J", position=(1.0, 0.0, 0.5))
    _lead(w, "lead", "g1", "J")
    loaded = gf.World.model_validate_json(w.model_dump_json())
    assert [j.name for j in loaded.junctions] == ["J"]
    assert loaded.conductors[0].end_electrode == "J"


def test_conductor_ends_and_sources_accept_junctions() -> None:
    w = _rod_world()
    j = _create_junction(w, name="J", position=(0.0, 0.0, -0.2))
    lead = _lead(w, "lead", "g1", j)
    assert (lead.start_electrode, lead.end_electrode) == ("g1", "J")
    assert lead.end == (0.0, 0.0, -0.2)
    src = gf.create_source(w, attached_to=j, magnitude=1.0)
    assert src.attached_to == "J"
    with pytest.raises(KeyError, match="Did you mean 'J'"):
        gf.create_source(w, attached_to="j", return_to="JJ", magnitude=1.0)


# ---------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------


def test_source_on_a_junction_sees_the_lead_resistance() -> None:
    """Z_in = Z_g1 + R: the lead is in series, the junction leaks nothing."""
    w = _rod_world()
    _create_junction(w, name="J", position=(0.0, 0.0, -0.2))
    _lead(w, "lead", "J", "g1", r_ohm=0.5)
    gf.create_source(w, attached_to="J", magnitude=2.0)
    res = _solve(w)

    w0 = _rod_world()
    gf.create_source(w0, attached_to="g1", magnitude=2.0)
    z_g1 = _solve(w0).electrode_potentials["g1"][0] / 2.0

    assert abs(res.electrode_currents["g1"][0] - 2.0) < 1e-12
    assert all(ps.electrode_name != "J" for ps in res.point_sources)
    z_in = _junction_potential(res, "J") / 2.0
    assert abs(z_in - (z_g1 + 0.5)) <= 1e-12 * abs(z_in)


def test_junction_bonded_ideally_shares_the_electrode_potential() -> None:
    w = _rod_world()
    _create_junction(w, name="J", position=(0.0, 0.0, -0.2))
    gf.create_conductor(w, name="bond", start="J", end="g1")  # ideal
    gf.create_source(w, attached_to="J", magnitude=1.0)
    res = _solve(w)
    phi_g1 = res.electrode_potentials["g1"][0]
    assert abs(_junction_potential(res, "J") - phi_g1) <= 1e-12 * abs(phi_g1)


def test_two_terminal_source_between_two_junctions() -> None:
    """Z_in = R_a + R_b + (Z11 - Z12 - Z21 + Z22) of the two rods."""

    def rods():
        w = gf.create_world(soil=_soil())
        gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
        gf.create_electrode(w, "rod", name="g2", position=(30.0, 0.0, 0.0), length=2.0)
        return w

    w = rods()
    _create_junction(w, name="Ja", position=(0.0, 0.0, -0.3))
    _create_junction(w, name="Jb", position=(30.0, 0.0, -0.3))
    _lead(w, "la", "Ja", "g1", r_ohm=0.2)
    _lead(w, "lb", "Jb", "g2", r_ohm=0.3)
    gf.create_source(w, attached_to="Ja", return_to="Jb", magnitude=1.0)
    res = _solve(w)
    z = np.empty((2, 2), dtype=complex)
    for j, node in enumerate(("g1", "g2")):
        ws = rods()
        gf.create_source(ws, attached_to=node, magnitude=1.0)
        r = _solve(ws)
        z[0, j], z[1, j] = (
            r.electrode_potentials["g1"][0],
            r.electrode_potentials["g2"][0],
        )
    expected = 0.2 + 0.3 + z[0, 0] - z[0, 1] - z[1, 0] + z[1, 1]
    z_in = _junction_potential(res, "Ja") - _junction_potential(res, "Jb")
    assert abs(z_in - expected) <= 1e-12 * abs(expected)
    cur = res.electrode_currents
    assert abs(cur["g1"][0] + cur["g2"][0]) < 1e-12


def test_measurement_leads_through_junctions_do_not_short_the_instrument() -> None:
    """10.17: feed lead H -> aux and probe lead probe -> S end on junctions
    at the instrument, so the soil sees exactly the galvanic-only loop."""
    d = 100.0

    def station():
        w = gf.create_world(soil=_soil())
        gf.create_electrode(w, "ring", name="st", center=(0.0, 0.0, 0.7), radius=2.0)
        gf.create_electrode(w, "rod", name="aux", position=(d, 0.0, 0.0), length=1.5)
        gf.create_electrode(
            w, "rod", name="probe", position=(0.618 * d, 0.0, 0.0), length=0.5
        )
        return w

    w0 = station()
    gf.create_source(w0, attached_to="st", return_to="aux", magnitude=1.0)
    p0 = _solve(w0, segment_length=1.0).electrode_potentials
    z_galvanic = (p0["st"][0] - p0["probe"][0]).real

    w = station()
    _create_junction(w, name="H", position=(2.5, 0.0, -0.1))
    _create_junction(w, name="S", position=(2.5, 0.5, -0.1))
    _lead(w, "feed", "H", "aux", r_ohm=1.2)
    _lead(w, "probe_lead", "probe", "S", r_ohm=0.8)
    gf.create_source(w, attached_to="st", return_to="H", magnitude=1.0)
    res = _solve(w, segment_length=1.0)
    z_meas = (res.electrode_potentials["st"][0] - _junction_potential(res, "S")).real
    assert abs(z_meas - z_galvanic) <= 1e-12 * z_galvanic
    assert abs(res.electrode_currents["aux"][0] + 1.0) < 1e-12
    assert abs(res.electrode_currents["probe"][0]) < 1e-14  # open voltmeter lead

    # The 0.15.0 topology for comparison: the probe lead ends on the station.
    ws = station()
    _lead(ws, "probe_lead", "probe", "st", r_ohm=0.8)
    gf.create_source(ws, attached_to="st", return_to="aux", magnitude=1.0)
    ps = _solve(ws, segment_length=1.0).electrode_potentials
    assert abs((ps["st"][0] - ps["probe"][0]).real) < 0.05 * z_galvanic


def _pen_world(joint: str) -> gf.World:
    """Station, two houses, PEN joint either a junction or a 5 cm rod."""
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "ring", name="st", center=(0.0, 0.0, 0.7), radius=2.0)
    for k, y in enumerate((-8.0, 8.0)):
        gf.create_electrode(w, "ring", name=f"h{k}", center=(40.0, y, 0.8), radius=3.0)
    if joint == "junction":
        _create_junction(w, name="T", position=(40.0, 0.0, 0.7))
    else:
        gf.create_electrode(
            w,
            "rod",
            name="T",
            position=(40.0, 0.0, 0.7),
            length=0.05,
            wire_radius=0.005,
        )
    for name, a, b, r in (
        ("pen", "st", "T", 0.05),
        ("s0", "T", "h0", 0.02),
        ("s1", "T", "h1", 0.02),
    ):
        _lead(w, name, a, b, r_ohm=r)
    gf.create_source(w, attached_to="st", magnitude=1.0)
    return w


def test_junction_replaces_the_leaking_pseudo_rod() -> None:
    """10.38: the 5 cm joint rod leaks a share of the current; a junction
    leaks none and the other electrodes carry all of it."""
    rod = _solve(_pen_world("rod"), segment_length=0.025)
    jun = _solve(_pen_world("junction"), segment_length=0.025)
    leak_rod = rod.electrode_currents["T"][0].real
    assert leak_rod > 1e-3  # the pseudo-rod is a real electrode (0.35 % here)
    assert "T" not in jun.electrode_currents
    total = sum(v[0] for v in jun.electrode_currents.values())
    assert abs(total - 1.0) < 1e-12
    phi_t = _junction_potential(jun, "T")
    i_houses = jun.electrode_currents["h0"][0] + jun.electrode_currents["h1"][0]
    phi_st = jun.electrode_potentials["st"][0]
    assert abs((phi_st - phi_t) - 0.05 * i_houses) <= 1e-12 * abs(phi_st - phi_t)


def test_isolated_distributed_conductor_to_a_junction_at_all_frequencies() -> None:
    w = _rod_world()
    gf.create_electrode(w, "rod", name="g2", position=(60.0, 0.0, 0.0), length=2.0)
    _create_junction(w, name="J", position=(60.0, 0.0, 0.6))
    gf.create_conductor(
        w,
        name="pen",
        start="g1",
        end="J",
        conductor_type="pen",
        cross_section=95e-6,
        resistivity=2.82e-8,
        discretize_segment_length=10.0,
        coupling_to_soil="isolated",
        inductance_model="neumann",
    )
    _lead(w, "drop", "J", "g2", r_ohm=0.01)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res = _solve(w, frequencies=[0.0, 50.0, 1000.0])
    phi_j = res.metadata["junction_potentials"]["J"]
    assert len(phi_j) == 3
    for k in range(3):
        assert abs(sum(v[k] for v in res.electrode_currents.values()) - 1.0) < 1e-12
        # Ohm on the lumped drop: phi_J - phi_g2 = 0.01 * I(g2)
        drop = complex(phi_j[k]) - res.electrode_potentials["g2"][k]
        assert abs(drop - 0.01 * res.electrode_currents["g2"][k]) <= 1e-10 * abs(drop)
    assert abs(complex(phi_j[2]).imag) > abs(complex(phi_j[1]).imag)


def test_galvanic_conductor_between_junctions_is_grounded_by_its_own_leakage() -> None:
    w = _rod_world()
    _create_junction(w, name="J1", position=(10.0, 0.0, 0.6))
    _create_junction(w, name="J2", position=(30.0, 0.0, 0.6))
    gf.create_conductor(
        w,
        name="tape",
        start="J1",
        end="J2",
        conductor_type="bare_copper",
        cross_section=50e-6,
        discretize_segment_length=2.0,
        coupling_to_soil="galvanic",
    )
    gf.create_source(w, attached_to="J1", return_to="g1", magnitude=1.0)
    res = _solve(w)
    leak = sum(v[0] for v in res.metadata["conductor_node_currents"].values())
    assert abs(leak - 1.0) < 1e-12
    assert abs(res.electrode_currents["g1"][0] + 1.0) < 1e-12
    phi1, phi2 = _junction_potential(res, "J1"), _junction_potential(res, "J2")
    assert abs(phi1 - phi2) < 1e-2 * abs(phi1)  # a low-resistance tape


# ---------------------------------------------------------------------
# rejected configurations
# ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["lumped", "isolated_distributed", "dangling"])
def test_subnetwork_without_soil_contact_is_rejected(kind: str) -> None:
    w = _rod_world()
    _create_junction(w, name="J1", position=(5.0, 0.0, 0.5))
    if kind != "dangling":
        _create_junction(w, name="J2", position=(15.0, 0.0, 0.5))
        extra = {} if kind == "lumped" else {"discretize_segment_length": 2.0}
        gf.create_conductor(
            w,
            name="c",
            start="J1",
            end="J2",
            cross_section=50e-6,
            coupling_to_soil="isolated",
            **extra,
        )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    with pytest.raises(ValueError, match=r"(?s)J1.*no soil contact"):
        w.solve(gf.create_engine(segment_length=SEG))


def test_backends_without_junction_support_reject_the_world() -> None:
    from groundfield.solver.registry import BackendCapabilityError

    w = _rod_world()
    _create_junction(w, name="J", position=(0.0, 0.0, -0.2))
    _lead(w, "lead", "J", "g1")
    gf.create_source(w, attached_to="J", magnitude=1.0)
    with pytest.raises(BackendCapabilityError, match=r"(?s)junction.*'image'"):
        w.solve(gf.create_engine(backend="fem", segment_length=SEG))


@pytest.mark.xfail(
    strict=True,
    reason="FieldResult.input_impedance reads electrode potentials only; "
    "junction support in result.py is requested from CORE-2b",
)
def test_input_impedance_of_a_source_on_a_junction() -> None:
    w = _rod_world()
    _create_junction(w, name="J", position=(0.0, 0.0, -0.2))
    _lead(w, "lead", "J", "g1", r_ohm=0.5)
    gf.create_source(w, attached_to="J", magnitude=1.0)
    res = _solve(w)
    assert abs(res.input_impedance()[0] - _junction_potential(res, "J")) < 1e-12
