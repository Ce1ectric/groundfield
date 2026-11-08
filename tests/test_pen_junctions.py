"""PEN cable joints of generated networks are leakage-free junction nodes.

Up to v0.15.0 the layout builder stood a 5 cm driven rod at every corner
and tap point of a routed PEN cable, because the world had no node
without soil contact. Those rods leaked 3 to 7 % of a test current
although the cable is declared ``coupling_to_soil="isolated"`` (review
finding 10.38). Since 2026.11 every joint is a
:class:`~groundfield.geometry.junction.Junction`: a node of the
conductor network with Kirchhoff's current law and no leakage.
"""

from __future__ import annotations

import math

import pytest

import groundfield as gf
from groundfield.generators import LvNetworkLayout


def _house(cx: float, cy: float, osm_id: int) -> gf.BuildingFootprint:
    return gf.BuildingFootprint(
        polygon_xy_m=[
            (cx - 5.0, cy - 5.0),
            (cx + 5.0, cy - 5.0),
            (cx + 5.0, cy + 5.0),
            (cx - 5.0, cy + 5.0),
        ],
        osm_id=osm_id,
    )


def _street(n_houses: int = 4) -> LvNetworkLayout:
    footprints = [
        _house(30.0 + 25.0 * i, 18.0 if i % 2 == 0 else -18.0, 100 + i)
        for i in range(n_houses)
    ]
    layout = LvNetworkLayout.from_footprints(footprints, substation_xy=(0.0, 0.0))
    layout.add_pen_cable(start="substation", end_xy=(30.0 + 25.0 * n_houses, 0.0))
    layout.connect_buildings()
    return layout


def _engine() -> gf.Engine:
    return gf.create_engine(backend="image", segment_length=1.0, frequencies=[50.0])


def _is_joint(name: str) -> bool:
    return "_node_" in name or "_tap_" in name


@pytest.mark.finding("10.38")
def test_cable_corners_and_taps_are_junctions_not_rods() -> None:
    world = _street().to_world(seed=0)
    assert not [e.name for e in world.electrodes if _is_joint(e.name)]
    joints = [j for j in world.junctions if _is_joint(j.name)]
    assert len(joints) == len(world.junctions) > 0
    # At the burial depth of the planned joint (0.6 m below the surface).
    assert all(math.isclose(j.position[2], 0.6) for j in joints)
    # Every PEN section and service drop ends on a junction or an electrode.
    nodes = {e.name for e in world.electrodes} | {j.name for j in world.junctions}
    for c in world.conductors:
        assert c.start_electrode in nodes and c.end_electrode in nodes


@pytest.mark.finding("10.38")
def test_the_earthing_system_carries_the_whole_test_current() -> None:
    """KCL: no part of the 1 A leaves through the cable joints."""
    world = _street().to_world(seed=0)
    res = world.solve(_engine())
    total = sum(i[0] for name, i in res.electrode_currents.items())
    real = sum(
        i[0] for name, i in res.electrode_currents.items() if not _is_joint(name)
    )
    assert abs(total - 1.0) < 1e-12
    assert abs(real - 1.0) < 1e-12
    # Every joint reports its potential.
    assert set(res.metadata["junction_potentials"]) == {j.name for j in world.junctions}


@pytest.mark.finding("10.38")
def test_a_resistive_cable_through_joints_equals_one_series_resistor() -> None:
    """Independent reference: a lumped, isolated cable substation -> tap ->
    house (plus a dead-end stub behind the tap) is one series resistor
    ``R(substation->tap) + R(tap->house)`` between the two earthing
    systems. With the pseudo-rods the joints leaked and the input
    impedance came out lower."""
    layout = _street(n_houses=1)
    world = layout.to_world(seed=0, pen_segment_length_m=None)
    res = world.solve(_engine())
    z_layout = res.input_impedance()[0]

    service = next(c for c in world.conductors if c.name.startswith("service_"))
    tap = service.start_electrode
    feeder = next(
        c
        for c in world.conductors
        if {c.start_electrode, c.end_electrode} == {"substation_ring_0", tap}
    )
    r_series = feeder.series_resistance + service.series_resistance
    assert r_series > 1e-3

    reference = gf.create_world(soil=world.soil)
    for electrode in world.electrodes:
        reference.add_electrode(electrode.model_copy())
    for c in world.conductors:
        if "_bond_" in c.name:
            reference.add_conductor(c.model_copy())
    gf.create_conductor(
        reference,
        name="pen",
        start="substation_ring_0",
        end=service.end_electrode,
        conductor_type="pen",
        cross_section="from_radius",
        lumped_series_resistance_ohm=r_series,
    )
    gf.create_source(reference, attached_to="substation_ring_0", magnitude=1.0)
    z_ref = reference.solve(_engine()).input_impedance()[0]
    assert z_layout == pytest.approx(z_ref, rel=1e-10)


@pytest.mark.finding("10.38")
def test_junction_layout_round_trips_through_json() -> None:
    world = _street().to_world(seed=0)
    restored = gf.World.model_validate_json(world.model_dump_json())
    assert [j.name for j in restored.junctions] == [j.name for j in world.junctions]
    a = world.solve(_engine()).input_impedance()[0]
    b = restored.solve(_engine()).input_impedance()[0]
    assert a == b
