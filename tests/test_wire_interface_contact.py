"""Horizontal wires next to a contrasting layer interface (audits W4A-07, W4B-02).

The thin-wire model carries the current on the wire axis. Next to a more
conductive layer the current of a horizontal wire crowds to its near side,
which a current on the axis cannot follow: a ring (a = 5 mm) over a layer 33
times more conductive reads +3.6 % at two wire radii, +8.6 % at 1.5 and +32 %
at 1.1 radii against the solid torus (the axisymmetric ``fem``, checked
against a Hankel thin ring plus the exact 2-D proximity correction), +89 %
when the interface cuts the wire. Up to these audits only rod ends warned
(``InterfaceContactWarning``, audit A3S-05); the image family, ``mom``,
``mom_sommerfeld`` and the three-dimensional ``fem`` now warn for horizontal
wires within the reach of :func:`groundfield.solver._guards.wire_contact_reach`
too, and ``diagnose()`` lists them under ``interface_contact``. The reach is
where the thin wire is 1 % off (evidence ``fw4axi/e03*``; since audit
W4R2A-04 the largest of six ring geometries plus 0.1 radii, ``fw4r2a/e03``);
no generated network and no world of the AP1 soils warns.
"""

from __future__ import annotations

import math
import warnings

import pytest

import groundfield as gf
from groundfield import diagnostics
from groundfield.solver import InterfaceContactWarning

pytestmark = pytest.mark.filterwarnings(
    "ignore::groundfield.ExperimentalBackendWarning"
)

A, R0, T = 0.005, 2.0, 2.0


def _ring(rho_wire: float, rho_other: float, d_over_a: float, *, below=True):
    """Ring r0 = 2 m, a = 5 mm, axis at 2 m; interface d below (or above)."""
    if below:
        soil = gf.TwoLayerSoil(rho_1=rho_wire, rho_2=rho_other, h_1=T + d_over_a * A)
    else:
        soil = gf.TwoLayerSoil(rho_1=rho_other, rho_2=rho_wire, h_1=T - d_over_a * A)
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "ring", name="g1", center=(0, 0, T), radius=R0, wire_radius=A
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _contact_warnings(world, backend="image_2layer", **kw):
    kw.setdefault("segment_length", 0.05 if backend != "fem" else 0.25)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = gf.create_engine(backend=backend, **kw).solve(world)
    hits = [r for r in rec if issubclass(r.category, InterfaceContactWarning)]
    return res, hits


def test_the_reach_follows_the_measured_table() -> None:
    """The nodes of audit W4R2A-04 (up to then 3.5 / 3.8 / 3.9 radii at 33 /
    100 / 1000, resistive 1.0 at 10, nothing below a contrast of 2)."""
    # audit W4A-07, W4R2A-04
    from groundfield.solver._guards import wire_contact_reach

    assert wire_contact_reach(100.0, 100.0 / 33.0) == pytest.approx(3.9)
    assert wire_contact_reach(1000.0, 30.0) == pytest.approx(3.9, abs=0.01)
    assert wire_contact_reach(100.0, 1.0) == pytest.approx(4.1)
    assert wire_contact_reach(100.0, 1e-3) == pytest.approx(4.2)  # held
    assert wire_contact_reach(30.0, 1000.0) == pytest.approx(1.3, abs=0.01)
    assert wire_contact_reach(100.0, 1000.0) == pytest.approx(1.2)
    assert wire_contact_reach(100.0, 300.0) == pytest.approx(1.0)
    assert wire_contact_reach(100.0, 100.0 / 1.5) == pytest.approx(1.06, abs=0.01)
    assert wire_contact_reach(100.0, 100.0 * 1.5) == pytest.approx(1.0)
    assert wire_contact_reach(100.0, 100.0 / 1.3) == 0.0
    assert wire_contact_reach(100.0, 100.0) == 0.0
    # monotone in the contrast on both sides
    k = [2.0, 3.0, 5.0, 10.0, 20.0, 33.0, 100.0, 1000.0]
    con = [wire_contact_reach(100.0, 100.0 / c) for c in k]
    res = [wire_contact_reach(100.0, 100.0 * c) for c in k]
    assert con == sorted(con) and res == sorted(res)
    assert all(c >= r for c, r in zip(con, res, strict=True))


@pytest.mark.parametrize("backend", ["image", "image_2layer", "mom", "mom_sommerfeld"])
def test_a_ring_two_radii_above_a_conductive_layer_warns(backend) -> None:
    """Up to audit W4A-07 a ring 1 cm (two radii) above a 30 Ohm m layer
    under 1000 Ohm m read +3.6 % without a word."""
    # audit W4A-07
    # the point form of the mom family needs segments shorter than 1.67 cm
    # this close to the interface (it refuses coarser ones)
    kw = {"segment_length": 0.015} if backend.startswith("mom") else {}
    _, hits = _contact_warnings(_ring(1000.0, 30.0, 2.0), backend, **kw)
    assert len(hits) == 1, [str(h.message) for h in hits]
    text = str(hits[0].message)
    assert "'g1'" in text and "2 wire radii" in text and "more conductive" in text
    assert "reach 3.9" in text


def test_inside_the_reach_the_thin_wire_is_more_than_1_percent_off() -> None:
    """At 3 radii over a 33 times more conductive layer (inside the reach of
    3.5) the thin wire reads 1.2 % above the solid torus and warns; at 4 radii
    (outside) 0.59 % and it does not."""
    # audit W4A-07
    rho_o = 1000.0 / 33.0
    for d, inside in ((3.0, True), (4.0, False)):
        w = _ring(1000.0, rho_o, d)
        thin, hits = _contact_warnings(w)
        torus, none = _contact_warnings(w, "fem")
        assert not none  # the axisymmetric fem resolves the wire
        assert torus.metadata["fem_model"] == "axisymmetric"
        error = (
            thin.cluster_impedance("g1")[0].real / torus.cluster_impedance("g1")[0].real
            - 1.0
        )
        assert (error > 0.01) is inside, (d, error)
        assert bool(hits) is inside, (d, [str(h.message) for h in hits])


@pytest.mark.parametrize(
    ("rho_other", "d_over_a", "warns"),
    [
        (1000.0 * 33.0, 1.1, True),  # resistive beyond, reach 1.3
        (1000.0 * 33.0, 1.5, False),
        (1000.0 * 10.0, 1.1, True),  # reach 1.2 (was 1.0, audit W4R2A-04)
        (1000.0 * 10.0, 1.3, False),
        (1000.0 * 3.0, 1.1, False),  # reach 1.0: only a cut wire
        (1000.0 * 3.0, 0.5, True),
        (1000.0 / 1.3, 0.5, False),  # below the contrast 1.4
        (1000.0 / 1.5, 0.5, True),  # was below the contrast 2
        (1000.0 / 3.0, 1.8, True),  # conductive beyond, reach 1.9
        (1000.0 / 3.0, 2.0, False),
    ],
)
def test_the_rule_on_both_sides_and_for_cut_wires(rho_other, d_over_a, warns) -> None:
    # audit W4A-07
    from groundfield.solver._guards import _wires_near_interfaces

    for below in (True, False):
        w = _ring(1000.0, rho_other, d_over_a, below=below)
        hits = _wires_near_interfaces(w)
        assert bool(hits) is warns, (rho_other, d_over_a, below, hits)


def test_diagnose_lists_the_wire_under_interface_contact() -> None:
    # audit W4A-07
    w = _ring(1000.0, 30.0, 1.5)
    for report in (diagnostics.diagnose(w), diagnostics.diagnose(w, gf.Engine())):
        found = report.by_category("interface_contact")
        assert [d.subjects for d in found] == [("g1",)]
        data = found[0].data
        assert found[0].severity == "warning" and data["end"] is None
        assert data["distance_radii"] == pytest.approx(1.5)
        assert data["reach_radii"] == pytest.approx(3.9, abs=0.01)
        assert data["rho_wire_ohm_m"] == 1000.0 and data["rho_other_ohm_m"] == 30.0
        text = str(report)
        assert "interface contact: 1 rod end or wire lies too close" in text
        assert "electrode 'g1'" in text


def test_strips_and_meshes_are_horizontal_wires_and_rods_are_not() -> None:
    # audit W4A-07
    from groundfield.solver._guards import _wires_near_interfaces

    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=0.81))
    gf.create_electrode(
        w, "strip", name="s", start=(0, 0, 0.8), end=(10, 0, 0.8), wire_radius=0.005
    )
    gf.create_electrode(
        w,
        "grid_mesh",
        name="m",
        corner=(20, 0, 0.8),
        size=(5, 5),
        n_x=2,
        n_y=2,
        wire_radius=0.005,
    )
    gf.create_electrode(
        w, "ring", name="far", center=(60, 0, 0.5), radius=2.0, wire_radius=0.005
    )
    gf.create_electrode(
        w, "rod", name="r", position=(40, 0, 0.5), length=3.0, wire_radius=0.008
    )
    gf.create_source(w, attached_to="s", magnitude=1.0)
    assert {h[0] for h in _wires_near_interfaces(w)} == {"s", "m"}


def test_a_galvanic_distributed_conductor_near_an_interface_is_reported() -> None:
    # audit W4A-07
    from groundfield.solver._guards import _wires_near_interfaces

    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=0.81))
    gf.create_electrode(w, "rod", name="a", position=(0, 0, 0.8), length=2.0)
    gf.create_electrode(w, "rod", name="b", position=(20, 0, 0.8), length=2.0)
    gf.create_conductor(
        w,
        name="pen",
        start="a",
        end="b",
        cross_section=50e-6,
        discretize_segment_length=1.0,
        coupling_to_soil="galvanic",
    )
    gf.create_source(w, attached_to="a", magnitude=1.0)
    hits = _wires_near_interfaces(w)
    assert [(h[0], h[1]) for h in hits] == [("pen", "conductor")]


def test_rod_ends_keep_their_rule_and_a_rod_through_the_interface_is_silent() -> None:
    # audit W4A-07 (rod ends: audit A3S-05, unchanged)
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0))
    gf.create_electrode(
        w, "rod", name="through", position=(0, 0, 0.5), length=3.0, wire_radius=0.008
    )
    gf.create_source(w, attached_to="through", magnitude=1.0)
    _, hits = _contact_warnings(w, segment_length=0.25)
    assert not hits
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0))
    gf.create_electrode(
        w, "rod", name="tip", position=(0, 0, 0.5), length=1.5, wire_radius=0.008
    )
    gf.create_source(w, attached_to="tip", magnitude=1.0)
    _, hits = _contact_warnings(w, segment_length=0.25)
    assert len(hits) == 1 and "rod end" in str(hits[0].message)


def test_the_three_dimensional_fem_warns_before_it_solves(monkeypatch) -> None:
    """The 3-D model's smoothed line source shares the thin-wire limit (audit
    W4B-02: a ring 1.5 cm above 1000/30 Ohm m read +9.1e-3 against the torus,
    beyond its estimate of 5.4e-3). The warning comes from the dispatch,
    before the 3-D model runs."""
    # audit W4B-02
    pytest.importorskip("skfem")
    pytest.importorskip("pyamg")
    from groundfield.solver.fem import _solid

    class _Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(_solid, "solid_earth", stop)
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=0.815))
    gf.create_electrode(
        w, "strip", name="s", start=(0, 0, 0.8), end=(10, 0, 0.8), wire_radius=0.005
    )
    gf.create_source(w, attached_to="s", magnitude=1.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(_Stop):
            gf.create_engine(backend="fem", segment_length=0.25).solve(w)
    hits = [r for r in rec if issubclass(r.category, InterfaceContactWarning)]
    assert len(hits) == 1 and "'s'" in str(hits[0].message)


# AP1 corner soils (rho_1, rho_2, h_1) and the homogeneous corner.
_AP1_SOILS = [
    (1000.0, 200.0, 30.0),
    (1000.0, 200.0, 5.0),
    (1000.0, 30.0, 30.0),
    (1000.0, 30.0, 5.0),
    (30.0, 200.0, 30.0),
    (30.0, 200.0, 5.0),
    (30.0, 30.0, 5.0),
]


@pytest.mark.parametrize(("rho_1", "rho_2", "h_1"), _AP1_SOILS)
def test_no_generated_network_of_the_ap1_soils_is_reported(rho_1, rho_2, h_1) -> None:
    """Generated TN networks (10 houses with encased foundations, one small
    commercial building, the substation) in the corner soils of the AP1 grid:
    every horizontal wire lies at least 4 m from the interface, the largest
    reach is 4.2 radii of a 0.205 m shell (0.86 m)."""
    # audit W4A-07
    from groundfield.generators import (
        TnNetworkConfig,
        TnNetworkGenerator,
        TwoLayerSoilSpec,
    )
    from groundfield.solver._guards import _wires_near_interfaces

    cfg = TnNetworkConfig(
        building_counts={"residential": 10, "small_industry": 1},
        soil=TwoLayerSoilSpec(rho_1=rho_1, rho_2=rho_2, h_1=h_1),
    )
    world = TnNetworkGenerator(cfg, seed=1).build()
    assert not _wires_near_interfaces(world)
    report = diagnostics.diagnose(world)
    assert not report.by_category("interface_contact")


def test_no_default_generated_network_is_reported() -> None:
    # audit W4A-07
    from groundfield.generators import TnNetworkConfig, TnNetworkGenerator
    from groundfield.solver._guards import _wires_near_interfaces

    world = TnNetworkGenerator(TnNetworkConfig(), seed=0).build()
    assert not _wires_near_interfaces(world)
    _, hits = _contact_warnings(world, "image_2layer", segment_length=1.0)
    assert not hits


def test_an_ap1_station_is_not_reported() -> None:
    """Square ring 4 m x 4 m at 0.6 m (5 mm) and three 4 m rods (8 mm) in
    1000 / 30 Ohm m with h_1 = 5 m (the station of the 3-D validation)."""
    # audit W4A-07
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0))
    gf.create_electrode(
        w,
        "polyline",
        name="ring",
        vertices=[(0, 0, 0.6), (4, 0, 0.6), (4, 4, 0.6), (0, 4, 0.6)],
        closed=True,
        wire_radius=0.005,
    )
    for k, (x, y) in enumerate([(0, 0), (4, 0), (4, 4)]):
        gf.create_electrode(
            w,
            "rod",
            name=f"rod{k}",
            position=(x, y, 0.6),
            length=4.0,
            wire_radius=0.008,
        )
        gf.create_conductor(w, start="ring", end=f"rod{k}")
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    _, hits = _contact_warnings(w, segment_length=0.5)
    assert not hits
    assert math.isfinite(len(diagnostics.diagnose(w).findings))
    assert not diagnostics.diagnose(w).by_category("interface_contact")


def test_a_wire_in_a_thin_layer_is_counted_once_and_listed_per_interface() -> None:
    """Three layers: a ring in a 12 mm middle layer lies within reach of both
    of its interfaces; the warning counts one wire, diagnose() lists one
    finding per interface (100 / 1000 / 10 Ohm m: reach 2.9 and 3.8 radii)."""
    # audit W4A-07 (self-check of the stream)
    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=100.0, h=1.0),
            gf.SoilLayer(rho=1000.0, h=0.012),
            gf.SoilLayer(rho=10.0),
        ]
    )
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "ring", name="r", center=(0, 0, 1.006), radius=2.0, wire_radius=0.005
    )
    gf.create_source(w, attached_to="r", magnitude=1.0)
    from groundfield.solver._guards import _warn_wires_near_interfaces

    # mom_sommerfeld refuses a wire below h_1 in three layers; the warning
    # itself is checked directly
    with pytest.warns(InterfaceContactWarning) as rec:
        _warn_wires_near_interfaces(w)
    assert len(rec) == 1 and str(rec[0].message).startswith("1 horizontal wire ")
    found = diagnostics.diagnose(w).by_category("interface_contact")
    assert [d.data["interface_m"] for d in found] == [1.0, 1.012]


def _stop_the_3d_model(monkeypatch):
    """Replace the 3-D model by a stub that stops the solve after the dispatch."""
    pytest.importorskip("skfem")
    pytest.importorskip("pyamg")
    from groundfield.solver.fem import _solid

    class _Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(_solid, "solid_earth", stop)
    return _Stop


def _rods_with_a_tip_on_the_interface() -> gf.World:
    """Auditor W4R2A's world: two bonded rods side by side (not coaxial), the
    tip of one on h_1 = 2 m of 1000 / 30 Ohm m."""
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0))
    gf.create_electrode(
        w, "rod", name="r1", position=(0, 0, 0.5), length=1.5, wire_radius=0.008
    )
    gf.create_electrode(
        w, "rod", name="r2", position=(3, 0, 0.5), length=1.0, wire_radius=0.008
    )
    gf.create_conductor(w, start="r1", end="r2")
    gf.create_source(w, attached_to="r1", magnitude=1.0)
    return w


def test_w4r2a_03_the_3d_fem_warns_for_a_rod_end_at_an_interface(monkeypatch) -> None:
    """The 3-D model is a thin wire without end faces too: a rod with its tip
    on 1000 / 30 Ohm m read +43 % against the solid rod with an estimate of
    6e-3 (auditor W4R2A, ``r09``), and its dispatch emitted only the
    horizontal-wire warning; its comment said that its own guards refuse rod
    ends near an interface, which they do not (vertical pieces are skipped).
    The dispatch now warns as the integral backends do, once per solve."""
    # audit W4R2A-03
    stop = _stop_the_3d_model(monkeypatch)
    w = _rods_with_a_tip_on_the_interface()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(stop):
            gf.create_engine(backend="fem", segment_length=0.25).solve(w)
    hits = [
        str(r.message) for r in rec if issubclass(r.category, InterfaceContactWarning)
    ]
    assert len(hits) == 1 and "rod end" in hits[0] and "'r1'" in hits[0], hits


def test_a_rod_end_and_a_wire_warn_once_each_in_the_3d_model(monkeypatch) -> None:
    """A rod end and a strip next to the interface: one warning each, as in
    the integral backends."""
    # audit W4R2A-03
    stop = _stop_the_3d_model(monkeypatch)
    w = _rods_with_a_tip_on_the_interface()
    gf.create_electrode(
        w, "strip", name="s", start=(0, 5, 1.99), end=(5, 5, 1.99), wire_radius=0.005
    )
    gf.create_conductor(w, start="r1", end="s")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(stop):
            gf.create_engine(backend="fem", segment_length=0.25).solve(w)
    hits = [
        str(r.message) for r in rec if issubclass(r.category, InterfaceContactWarning)
    ]
    assert len(hits) == 2, hits
    assert sum("rod end" in h for h in hits) == 1
    assert sum("horizontal wire" in h for h in hits) == 1


def test_the_axisymmetric_fem_solves_a_rod_end_at_an_interface_without_a_warning() -> (
    None
):
    """The axisymmetric model solves the rod with its flat end face, which
    is the remedy of the rod-end rule; it warns for neither rule, and
    ``diagnose()`` with an ``fem`` engine lists no rod end for a world it
    takes (with the default engine it does)."""
    # audit W4R2A-03, W4R2A-05
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0))
    gf.create_electrode(
        w, "rod", name="tip", position=(0, 0, 0.5), length=1.5, wire_radius=0.008
    )
    gf.create_source(w, attached_to="tip", magnitude=1.0)
    res, hits = _contact_warnings(w, "fem")
    assert res.metadata["fem_model"] == "axisymmetric" and not hits
    eng = gf.create_engine(backend="fem", segment_length=0.25)
    assert not diagnostics.diagnose(w, eng).by_category("interface_contact")
    found = diagnostics.diagnose(w, gf.Engine()).by_category("interface_contact")
    assert [f.data["end"] for f in found] == ["tip"]


def test_w4r2a_05_diagnose_agrees_with_an_axisymmetric_fem_solve() -> None:
    """Auditor W4R2A's test: ``diagnose(world, engine)`` with an ``fem``
    engine reported the ``interface_contact`` warning for a ring that the
    axisymmetric model solves as a solid torus without a warning (the
    solve's own remedy)."""
    # audit W4R2A-05
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.01))
    gf.create_electrode(
        w, "ring", name="g1", center=(0, 0, 2.0), radius=2.0, wire_radius=0.005
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="fem", segment_length=0.25)
    res, hits = _contact_warnings(w, "fem")
    assert res.metadata["fem_model"] == "axisymmetric" and not hits
    found = diagnostics.diagnose(w, eng).by_category("interface_contact")
    assert not [f for f in found if f.severity == "warning"], [f.message for f in found]
    # the integral engines and the forced 3-D model keep the finding
    for other in (
        gf.Engine(),
        gf.create_engine(backend="mom", segment_length=0.25),
        gf.create_engine(
            backend="fem", segment_length=0.25, options={"fem": {"model": "3d"}}
        ),
    ):
        found = diagnostics.diagnose(w, other).by_category("interface_contact")
        assert [f.subjects for f in found] == [("g1",)], other.backend


def test_diagnose_with_an_fem_engine_keeps_the_findings_of_a_3d_world() -> None:
    """For a world that goes to the 3-D model, ``diagnose()`` with an ``fem``
    engine keeps the rod end (the 3-D model warns for it too)."""
    # audit W4R2A-03, W4R2A-05
    w = _rods_with_a_tip_on_the_interface()
    eng = gf.create_engine(backend="fem", segment_length=0.25)
    found = diagnostics.diagnose(w, eng).by_category("interface_contact")
    assert [(f.subjects, f.data["end"]) for f in found] == [(("r1",), "tip")]


@pytest.mark.parametrize(
    ("rho_other", "d_over_a", "r0", "a"),
    [
        (100.0 / 33.0, 3.6, 1.0, 0.055),  # thick wire on a 1 m ring: +1.09 %
        (100.0 / 10.0, 3.0, 1.0, 0.055),  # +1.1 %
        (100.0 / 5.5, 2.4, 1.0, 0.055),  # +1.1 %, between the nodes 3 and 10
        (100.0 / 1.9, 0.5, 2.0, 0.005),  # below the contrast 2, cut: -1.2 %
    ],
)
def test_w4r2a_04_a_thin_wire_more_than_1_percent_off_warns(
    rho_other, d_over_a, r0, a
) -> None:
    """Auditor W4R2A's test: the reach table was not conservative for a
    thick wire on a short ring (r0 / a = 18: the 1 % distance lay 0.2 ... 0.4
    radii beyond the reach at contrasts 2.5 ... 100) and below the contrast 2
    for a cut wire; the thin wire was more than 1 % off the solid torus
    without a warning."""
    # audit W4R2A-04
    t = 0.8 if a > 0.01 else 2.0
    w = gf.create_world(
        soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=rho_other, h_1=t + d_over_a * a)
    )
    gf.create_electrode(
        w, "ring", name="g1", center=(0, 0, t), radius=r0, wire_radius=a
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    torus, none = _contact_warnings(w, "fem")
    assert not none and torus.metadata["fem_model"] == "axisymmetric"
    thin, hits = _contact_warnings(
        w, "image_2layer", segment_length=0.1 if a > 0.01 else 0.05
    )
    error = (
        thin.cluster_impedance("g1")[0].real / torus.cluster_impedance("g1")[0].real
        - 1.0
    )
    assert abs(error) > 0.01  # the premise
    assert len(hits) == 1, error


# Auditor W4R2A's G4 ring (r0 = 1 m, a = 55 mm, axis at 0.8 m in 100 Ohm m),
# the widest of six geometries: the distance in wire radii at which the thin
# wire is 1 % off the solid torus at the contrasts between the nodes of the
# table (evidence fw4r2a/e03_summary.out), and G1 (r0 = 2 m, a = 5 mm) with
# the more resistive layer above it.
_MEASURED_1_PERCENT = [
    ("conductive", 1.9, 1.07),
    ("conductive", 2.5, 1.49),
    ("conductive", 4.0, 2.16),
    ("conductive", 7.0, 2.83),
    ("conductive", 14.0, 3.40),
    ("conductive", 25.0, 3.68),
    ("conductive", 50.0, 3.87),
    ("conductive", 300.0, 4.05),
    ("resistive", 10.0, 1.02),
    ("resistive", 100.0, 1.20),
]


@pytest.mark.parametrize(("side", "contrast", "d1"), _MEASURED_1_PERCENT)
def test_w4r2a_04_the_reach_covers_the_measured_1_percent_distances(
    side, contrast, d1
) -> None:
    """Up to audit W4R2A-04 the table was the largest of four geometries
    rounded up, 0.2 ... 0.4 radii short of a thick wire on a short ring and
    linear in log10 of the contrast between nodes where the curve is not.
    Now it is the largest of six geometries at its nodes rounded up to 0.1 a
    plus 0.1 a, and covers every measured point between them (2348 points:
    none 1 % off outside the reach, the largest +0.94 %)."""
    # audit W4R2A-04
    from groundfield.solver._guards import wire_contact_reach

    rho_other = 100.0 / contrast if side == "conductive" else 100.0 * contrast
    assert wire_contact_reach(100.0, rho_other) > d1


def test_w4r2a_04_cut_wires_are_reported_from_the_contrast_1_4() -> None:
    """A wire that the interface cuts reads 1.04 % high at a contrast of 1.5
    (the more resistive layer beyond, G4, cut 0.5 a below the axis) and
    0.77 % at 1.3; up to audit W4R2A-04 nothing was reported below 2."""
    # audit W4R2A-04
    from groundfield.solver._guards import (
        WIRE_CONTACT_MIN_CONTRAST,
        _wires_near_interfaces,
    )

    assert 1.3 < WIRE_CONTACT_MIN_CONTRAST < 1.5
    for k, reported in ((1.3, False), (1.5, True)):
        for rho_other in (1000.0 / k, 1000.0 * k):
            w = _ring(1000.0, rho_other, 0.5)
            assert bool(_wires_near_interfaces(w)) is reported, (k, rho_other)
    # beyond the wire only where the measured 1 % distance needs it: not at
    # 1.5 over the more resistive layer, not at 1.3 on either side
    for rho_other in (1000.0 * 1.5, 1000.0 / 1.3, 1000.0 * 1.3):
        assert not _wires_near_interfaces(_ring(1000.0, rho_other, 1.01))


@pytest.mark.parametrize(
    ("z_end", "reported"), [(1.5, True), (2.0, True), (3.0, False)]
)
def test_w4r2a_04_a_sloped_conductor_piece_is_horizontal_up_to_e_z_0_1(
    z_end, reported
) -> None:
    """Kept as documented (audit W4R2A-04): a piece of a galvanic distributed
    conductor crossing the interface counts as a horizontal wire while its
    slope |e_z| is at most 0.1 (5.7 deg); a steeper piece passes the reach of
    the interface over at most 2 reach / |e_z| wire radii of its length, like
    a rod through an interface, and is not reported (auditor W4R2A, r11:
    |e_z| = 0.050 / 0.075 reported, 0.124 not)."""
    # audit W4R2A-04
    from groundfield.solver._guards import _wires_near_interfaces

    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=1.0))
    gf.create_electrode(w, "rod", name="a", position=(0, 0, 0.5), length=0.1)
    gf.create_electrode(w, "rod", name="b", position=(20, 0, z_end), length=0.1)
    gf.create_conductor(
        w,
        name="pen",
        start="a",
        end="b",
        cross_section=50e-6,
        discretize_segment_length=1.0,
        coupling_to_soil="galvanic",
    )
    gf.create_source(w, attached_to="a", magnitude=1.0)
    hits = [h for h in _wires_near_interfaces(w) if h[0] == "pen"]
    assert bool(hits) is reported, hits
