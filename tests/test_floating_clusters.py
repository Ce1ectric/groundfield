"""Source-free clusters float in the image family (finding 10.29, image half).

Up to 0.15.0 the image family dropped every cluster without a source
from the linear system and read its potential back as the incident
field. Under the uniform-current ansatz that is exact for a single
passive electrode (it carries no current either way), but not for a
*bonded* group of passive electrodes, which exchanges current to stay
one equipotential, nor for a passive network joined by conductors: the
members reported different potentials and zero currents (2.5735 V and
1.8055 V where ``mom`` gives one 2.4189 V with ±13 mA, eng1 evidence
e04). Such clusters now enter the system as floating nodes with zero
net current, as 10.29 did for the ``mom`` family.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

pytestmark = pytest.mark.finding("10.29")

SEG = 0.5


def _soil(rho: float = 100.0) -> gf.HomogeneousSoil:
    """Homogeneous soil; ``resistivity`` was the field name up to 0.15.0."""
    try:
        return gf.HomogeneousSoil(rho=rho)
    except Exception:  # noqa: BLE001 - v0.15.0 names the field ``resistivity``
        return gf.HomogeneousSoil(resistivity=rho)


def _solve(world: gf.World, **engine_kw) -> gf.FieldResult:
    kw = {"segment_length": SEG, "frequencies": [50.0], **engine_kw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world.solve(gf.create_engine(**kw))


def _e04_world(soil=None) -> gf.World:
    """Driven rod and a passive ring + rod pair bonded by an ideal wire."""
    w = gf.create_world(soil=soil if soil is not None else _soil())
    gf.create_electrode(w, "rod", name="src", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "ring", name="p_ring", center=(6.0, 0.0, 0.6), radius=1.5)
    gf.create_electrode(w, "rod", name="p_rod", position=(8.5, 0.0, 0.6), length=2.0)
    gf.create_conductor(w, start="p_ring", end="p_rod", coupling_to_soil="isolated")
    gf.create_source(w, attached_to="src", magnitude=1.0)
    return w


# ---------------------------------------------------------------------
# a passive bonded cluster
# ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["image", "image_2layer"])
def test_passive_bonded_cluster_is_one_floating_equipotential(backend: str) -> None:
    soil = (
        _soil()
        if backend == "image"
        else gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=4.0)
    )
    res = _solve(_e04_world(soil), backend=backend)
    phi = {n: res.electrode_potentials[n][0] for n in ("p_ring", "p_rod")}
    cur = {n: res.electrode_currents[n][0] for n in ("p_ring", "p_rod")}
    assert abs(phi["p_ring"] - phi["p_rod"]) <= 1e-12 * abs(phi["p_ring"])
    # The members exchange current, the cluster as a whole carries none.
    assert abs(cur["p_ring"]) > 1e-3
    assert cur["p_ring"] + cur["p_rod"] == 0  # exactly: no round-off residue
    assert math.isnan(res.cluster_impedance("p_ring")[0].real)


def test_passive_bonded_cluster_agrees_with_mom() -> None:
    """Independent method: the point-matching MoM (per-segment currents)."""
    res_img = _solve(_e04_world())
    res_mom = _solve(_e04_world(), backend="mom")
    phi_img = res_img.electrode_potentials["p_ring"][0].real
    phi_mom = res_mom.electrode_potentials["p_ring"][0].real
    assert abs(phi_img - phi_mom) / phi_mom < 0.03
    i_img = res_img.electrode_currents["p_rod"][0].real
    i_mom = res_mom.electrode_currents["p_rod"][0].real
    assert i_img * i_mom > 0.0  # same direction of the equalising current
    assert abs(i_img - i_mom) / abs(i_mom) < 0.3


def _two_bonded_pairs() -> gf.World:
    w = gf.create_world(soil=_soil())
    for tag, x in (("A", 0.0), ("B", 12.0)):
        gf.create_electrode(
            w, "ring", name=f"{tag}_ring", center=(x, 0.0, 0.6), radius=1.5
        )
        gf.create_electrode(
            w, "rod", name=f"{tag}_rod", position=(x + 3.0, 1.0, 0.6), length=2.5
        )
        gf.create_conductor(
            w, start=f"{tag}_ring", end=f"{tag}_rod", coupling_to_soil="isolated"
        )
    return w


def _cluster_potential(res: gf.FieldResult, members: list[str]) -> complex:
    lengths: dict[str, float] = {}
    for ps in res.point_sources:
        lengths[ps.electrode_name] = lengths.get(ps.electrode_name, 0.0) + ps.length
    w = np.array([lengths[m] for m in members])
    u = np.array([res.electrode_potentials[m][0] for m in members])
    return complex(w @ u / w.sum())


def test_two_port_transfer_impedance_is_reciprocal() -> None:
    """Z_BA = Z_AB for two bonded multi-member clusters (open-circuit ports)."""
    z = {}
    for driven, observed in (("A", "B"), ("B", "A")):
        w = _two_bonded_pairs()
        gf.create_source(w, attached_to=f"{driven}_ring", magnitude=1.0)
        res = _solve(w)
        z[(observed, driven)] = _cluster_potential(
            res, [f"{observed}_ring", f"{observed}_rod"]
        )
    assert abs(z[("B", "A")] - z[("A", "B")]) <= 1e-12 * abs(z[("A", "B")])


# ---------------------------------------------------------------------
# passive networks
# ---------------------------------------------------------------------


def test_passive_network_carries_its_equalising_current() -> None:
    """Two passive rods at different incident potentials, joined by a
    finite conductor: the current through the conductor leaks at one rod
    and returns at the other; Ohm's law holds along the conductor."""
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "rod", name="src", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "rod", name="p1", position=(4.0, 0.0, 0.0), length=2.0)
    gf.create_electrode(w, "rod", name="p2", position=(24.0, 0.0, 0.0), length=2.0)
    lead = gf.create_conductor(
        w,
        name="lead",
        start="p1",
        end="p2",
        conductor_type="bare_copper",
        cross_section=1.0e-6,
        coupling_to_soil="isolated",
    )
    gf.create_source(w, attached_to="src", magnitude=1.0)
    res = _solve(w)
    i1 = res.electrode_currents["p1"][0]
    i2 = res.electrode_currents["p2"][0]
    assert abs(i1 + i2) <= 1e-12 * abs(i1)  # KCL of the passive component
    assert i1.real < -1e-3  # p1 sits at the higher incident potential: takes current in
    u1 = res.electrode_potentials["p1"][0]
    u2 = res.electrode_potentials["p2"][0]
    # Ohm: the branch current (p1 -> p2) is -I(p1); its drop is u1 - u2.
    assert abs((u1 - u2) - lead.series_resistance * (-i1)) <= 1e-9 * abs(u1 - u2)


def test_inductively_driven_passive_loop_carries_current() -> None:
    """A grounded measurement lead next to the source lead (10.29 / R2:
    the induced EMF drives a current through the earth loop)."""
    w = gf.create_world(soil=_soil())
    for name, x, y in (("g1", 0, 0), ("g2", 20, 0), ("m1", 0, 1), ("m2", 20, 1)):
        gf.create_electrode(
            w, "rod", name=name, position=(float(x), float(y), 0.5), length=1.5
        )
    for name, a, b in (("src_lead", "g1", "g2"), ("meas_lead", "m1", "m2")):
        gf.create_conductor(
            w,
            name=name,
            start=a,
            end=b,
            conductor_type="bare_copper",
            wire_radius=0.004,
            resistivity=1.72e-8,
            cross_section=50e-6,
            discretize_segment_length=5.0,
            coupling_to_soil="galvanic",
            inductance_model="neumann",
        )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res = _solve(w, frequencies=[50.0, 500.0], earth_inductive_model="perfect_mirror")
    md = res.metadata["conductor_node_currents"]
    for k in range(2):
        i_meas = res.electrode_currents["m1"][k] + res.electrode_currents["m2"][k]
        i_meas += sum(v[k] for n, v in md.items() if "meas_lead" in n)
        assert abs(i_meas) < 1e-12  # the passive component leaks no net current
        assert abs(res.electrode_currents["m1"][k]) > 1e-4
    # The induced share grows with frequency.
    assert abs(res.electrode_currents["m2"][1] - res.electrode_currents["m2"][0]) > 1e-5


# ---------------------------------------------------------------------
# what does not change
# ---------------------------------------------------------------------


def test_single_passive_electrode_keeps_the_read_back_potential() -> None:
    """Uniform current per electrode: a lone passive electrode carries no
    current and its potential is the incident field (exact, as before)."""
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "rod", name="src", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "ring", name="p", center=(6.0, 0.0, 0.6), radius=1.5)
    gf.create_source(w, attached_to="src", magnitude=1.0)
    # Point kernel: the reported potential is the length-weighted mean of
    # the midpoint field. The line kernel reports the segment average of
    # the field (Galerkin), 1e-4 away from the midpoint values here.
    res = _solve(w, kernel="point")
    assert res.electrode_currents["p"][0] == 0
    pts = np.array(
        [ps.position for ps in res.point_sources if ps.electrode_name == "p"]
    )
    lens = np.array([ps.length for ps in res.point_sources if ps.electrode_name == "p"])
    incident = res.potential(pts)
    expected = complex(lens @ incident / lens.sum())
    assert abs(res.electrode_potentials["p"][0] - expected) <= 1e-12 * abs(expected)
