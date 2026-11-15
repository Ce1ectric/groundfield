"""Fixes of audit W2A (wave 2, impressed voltage sources).

Each test names the finding it guards (``W2A-NN`` of the audit report) and
fails on the base ``8c83d7c`` for the reason given in its docstring. The
tests of W2A-01 … W2A-09 are the auditor's proposals; W2A-10 (accepted by
the coordinator) compares a generated voltage-driven measurement world
with the same world driven by a current source, by two methods.
"""

from __future__ import annotations

import warnings

import pytest

import groundfield as gf
from groundfield.sources import VoltageSource

# The audit ids (W2A-NN) are not finding ids of the review catalogue; the
# tests carry them in their names and comments, the module the finding of
# the voltage sources.
pytestmark = pytest.mark.finding("10.51")


def _engine(**kw) -> gf.Engine:
    return gf.create_engine(**{"segment_length": 0.5, "frequencies": [50.0], **kw})


def _two_rods() -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "rod", name="g2", position=(20.0, 0.0, 0.0), length=2.0)
    return w


# W2A-01 -----------------------------------------------------------------


def _layout():
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
    layout.add_auxiliary_electrode(distance_m=300.0, direction_deg=180.0)
    layout.add_voltage_probe(inline_fraction=0.618)
    return layout


def _layout_worlds():
    """The layout driven by 1 A and by a 230 V source at the same terminals."""
    layout = _layout()
    soil = gf.HomogeneousSoil(rho=100.0)
    w_i = layout.to_world(soil=soil, source_magnitude_a=1.0, seed=0)
    w_v = layout.to_world(soil=soil, source_magnitude_a=1.0, seed=0)
    s = w_v.sources[0]
    w_v.sources = [
        VoltageSource(
            name=s.name,
            attached_to=s.attached_to,
            return_to=s.return_to,
            magnitude=230.0,
        )
    ]
    return layout, w_i, w_v


# audit W2A-01
def test_w2a_01_layout_reading_of_a_voltage_source_uses_the_solved_current() -> None:
    """Base: the reading divides by a fallback of 1 A (5.604 instead of
    1.635 Ohm) and the balance reports ok=False against 1 A."""
    layout, w_i, w_v = _layout_worlds()
    eng = _engine(segment_length=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_i = layout.measured_grounding_impedance(w_i.solve(eng))
        r_v = w_v.solve(eng)
        z_v = layout.measured_grounding_impedance(r_v)
        balance = layout.verify_current_balance(r_v)
    assert z_v == pytest.approx(z_i, rel=1e-12)
    assert balance["ok"]
    assert balance["expected_positive_a"] == pytest.approx(r_v.sources[0].current[0])


# audit W2A-01
def test_w2a_01_layout_reading_of_a_voltage_record_without_current_raises() -> None:
    """Base: a voltage record without a solved current fell back to 1 A."""
    from groundfield.solver.result import SourceRecord

    layout, _, w_v = _layout_worlds()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_v = w_v.solve(_engine(segment_length=1.0))
    rec = r_v.sources[0]
    r_v.sources = [
        SourceRecord(
            name=rec.name,
            kind="voltage",
            attached_to=rec.attached_to,
            return_to=rec.return_to,
            voltage=list(rec.voltage),
        )
    ]
    with pytest.raises(ValueError, match="no solved source current"):
        layout.measured_grounding_impedance(r_v)
    with pytest.raises(ValueError, match="no solved source current"):
        layout.verify_current_balance(r_v)
    # An explicit test current still overrides the snapshot.
    assert layout.measured_grounding_impedance(r_v, source_magnitude_a=2.0) != 0


# W2A-02 -----------------------------------------------------------------


# audit W2A-02
def test_w2a_02_compare_engines_compares_the_input_impedance_of_a_voltage_source() -> (
    None
):
    """Base: only Z_cluster[...] is compared for a voltage world."""
    w = _two_rods()
    gf.create_source(w, attached_to="g1", kind="voltage", magnitude=230.0, name="v")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = gf.compare_engines(
            w, {"image": _engine(backend="image"), "mom": _engine(backend="mom")}
        )
    assert "Z_in[v]" in cmp.impedances
    z = cmp.impedances["Z_in[v]"]
    assert z["image"][0] == pytest.approx(z["mom"][0], rel=0.05)


# W2A-03 -----------------------------------------------------------------


# audit W2A-03
def test_w2a_03_old_alias_keeps_its_keyword_and_passes_current_worlds() -> None:
    """Base: TypeError (unexpected keyword 'stacklevel') for every world."""
    from groundfield.solver import BackendCapabilityError
    from groundfield.solver.image import _warn_ignored_sources

    w = _two_rods()
    gf.create_source(w, attached_to="g1", magnitude=1.0, name="i")
    assert _warn_ignored_sources(w, "custom", stacklevel=3) is None
    assert _warn_ignored_sources(w, "custom") is None
    gf.create_source(w, attached_to="g2", kind="voltage", magnitude=1.0, name="v")
    with pytest.raises(BackendCapabilityError, match="voltage"):
        _warn_ignored_sources(w, "custom", stacklevel=3)


# W2A-04 -----------------------------------------------------------------


def _result_with_an_empty_voltage_record() -> gf.FieldResult:
    from groundfield.solver.result import SourceRecord

    return gf.FieldResult(
        backend="custom",
        frequencies=[50.0, 1000.0],
        sources=[SourceRecord(name="v", kind="voltage", attached_to="g1")],
    )


# audit W2A-04
def test_w2a_04_voltage_record_without_voltage_and_current_raises() -> None:
    """Base: input_impedance() returns [] for a two-frequency result."""
    res = _result_with_an_empty_voltage_record()
    with pytest.raises(ValueError, match="no solved source current"):
        res.input_impedance("v")
    with pytest.raises(ValueError, match="no solved source current"):
        res.input_impedance()


# audit W2A-04
def test_w2a_04_spectrum_source_current_raises_a_value_error() -> None:
    """Base: IndexError from record.current[k]."""
    from groundfield.postprocess.safety import _source_current

    with pytest.raises(ValueError, match="no solved source current"):
        _source_current(_result_with_an_empty_voltage_record(), None, 0)


# W2A-06 -----------------------------------------------------------------


# audit W2A-06
def test_w2a_06_same_node_voltage_source_message_names_the_short() -> None:
    """Base: 'so the source drives no current' for an ideal voltage source."""
    w = _two_rods()
    # create_source and World.check run the same terminal check.
    with pytest.raises(ValueError) as created:
        gf.create_source(
            w, attached_to="g1", return_to="g1", kind="voltage", magnitude=230.0
        )
    w.sources = [
        VoltageSource(name="v", attached_to="g1", return_to="g1", magnitude=230.0)
    ]
    with pytest.raises(ValueError) as checked:
        w.check()
    for excinfo in (created, checked):
        message = str(excinfo.value)
        assert "drives no current" not in message
        assert "short-circuited" in message


# audit W2A-06
def test_w2a_06_same_node_current_source_keeps_its_message() -> None:
    w = _two_rods()
    with pytest.raises(ValueError, match="drives no current"):
        gf.create_source(w, attached_to="g1", return_to="g1", magnitude=1.0)


# W2A-07 -----------------------------------------------------------------


# audit W2A-07
def test_w2a_07_direct_fem_message_names_backends_that_solve_the_world() -> None:
    """Base: names 'image_2layer' for a homogeneous world (Engine.solve does not).

    fem solves voltage sources since its conforming model (user decision
    Q37); its axisymmetric model refuses the two rods side by side, and the
    direct call and Engine.solve name the same backends. (The
    three-dimensional model solves the world, so the axisymmetric one is
    pinned.)
    """
    from groundfield.solver import BackendCapabilityError
    from groundfield.solver.fem import solve_fem

    w = _two_rods()
    gf.create_source(w, attached_to="g1", kind="voltage", magnitude=230.0, name="v")
    axi = {"fem": {"model": "axisymmetric"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BackendCapabilityError) as direct:
            solve_fem(w, _engine(backend="fem"), model="axisymmetric")
        with pytest.raises(BackendCapabilityError) as engine:
            w.solve(_engine(backend="fem", options=axi))
    assert "'image_2layer'" not in str(direct.value)
    assert "voltage" not in str(direct.value)
    for name in ("'image'", "'mom'"):
        assert name in str(direct.value)
        assert name in str(engine.value)


# W2A-08 -----------------------------------------------------------------


def _four_rods() -> gf.World:
    w = _two_rods()
    for k, x in ((3, 0.0), (4, 30.0)):
        gf.create_electrode(w, "rod", name=f"g{k}", position=(x, 12.0, 0.0), length=2.0)
    return w


# audit W2A-08
def test_w2a_08_loop_message_names_the_sources_of_the_loop_only() -> None:
    """Base: lists v1, which is not part of the loop v2-v3-v4."""
    w = _four_rods()
    gf.create_source(w, attached_to="g1", kind="voltage", magnitude=230.0, name="v1")
    gf.create_source(
        w, attached_to="g2", return_to="g1", kind="voltage", magnitude=10.0, name="v2"
    )
    gf.create_source(
        w, attached_to="g3", return_to="g1", kind="voltage", magnitude=20.0, name="v3"
    )
    gf.create_source(
        w, attached_to="g3", return_to="g2", kind="voltage", magnitude=30.0, name="v4"
    )
    with pytest.raises(ValueError, match="loop of voltage sources") as excinfo:
        w.check()
    message = str(excinfo.value)
    assert "'v1'" not in message
    for name in ("'v2'", "'v3'", "'v4'"):
        assert name in message


# audit W2A-08
def test_w2a_08_loop_through_remote_earth_names_its_path() -> None:
    w = _four_rods()
    gf.create_source(w, attached_to="g4", kind="voltage", magnitude=1.0, name="v0")
    gf.create_source(w, attached_to="g1", kind="voltage", magnitude=230.0, name="v1")
    gf.create_source(
        w, attached_to="g2", return_to="g1", kind="voltage", magnitude=10.0, name="v2"
    )
    gf.create_source(w, attached_to="g2", kind="voltage", magnitude=20.0, name="v3")
    with pytest.raises(ValueError, match="loop of voltage sources") as excinfo:
        w.check()
    message = str(excinfo.value)
    assert "'v0'" not in message
    for name in ("'v1'", "'v2'", "'v3'"):
        assert name in message


# audit W2A-08
@pytest.mark.parametrize(
    "bond",
    [
        {"lumped_series_resistance_ohm": 1e-12, "cross_section": 50e-6},
        {"cross_section": 1.0},
    ],
    ids=["1e-12-ohm", "1-m2"],
)
def test_w2a_08_loop_message_names_the_ideal_bond_threshold(bond) -> None:
    """Base: 'put a finite conductor in series' for a 1e-12 Ohm conductor."""
    w = _two_rods()
    gf.create_conductor(w, name="bond", start="g1", end="g2", **bond)
    gf.create_source(w, attached_to="g1", kind="voltage", magnitude=230.0, name="va")
    gf.create_source(w, attached_to="g2", kind="voltage", magnitude=10.0, name="vb")
    with pytest.raises(ValueError, match="loop of voltage sources") as excinfo:
        w.check()
    assert "1e-06" in str(excinfo.value) or "1e-6" in str(excinfo.value)


# W2A-09 -----------------------------------------------------------------


# audit W2A-09
def test_w2a_09_fem_shell_message_names_the_mom_family() -> None:
    """Base: 'Use backend='image' / 'image_2layer'' although mom solves shells.

    fem solves the encased rod since its conforming model (user decision
    Q37); the guard that refused it names the mom family for a backend
    without the correction.
    """
    from groundfield.solver import BackendCapabilityError
    from groundfield.solver._pipeline import _reject_concrete_shells
    from groundfield.solver.fem import solve_fem

    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(
        w,
        "rod",
        name="g1",
        position=(0.0, 0.0, 0.0),
        length=3.0,
        concrete_shell_coefficient_ohm_m=1.0,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0, name="i")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert solve_fem(w, _engine(backend="fem")).cluster_impedance("g1")[0].real > 0
    with pytest.raises(BackendCapabilityError) as excinfo:
        _reject_concrete_shells(w, "toy")
    message = str(excinfo.value)
    for name in ("'image'", "'mom'", "'mom_sommerfeld'"):
        assert name in message
    assert "'image_2layer'" not in message


# W2A-10 -----------------------------------------------------------------


def _generated(kind: str, *, leads: bool) -> gf.World:
    from groundfield.generators.measurement import (
        MeasurementInjectionConfig,
        MeasurementProbeConfig,
        MeasurementSetupConfig,
        overhead_lead,
    )
    from groundfield.generators.soil_specs import HomogeneousSoilSpec
    from groundfield.generators.tn_network import TnNetworkConfig, TnNetworkGenerator

    lead = overhead_lead() if leads else None
    cfg = TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=100.0),
        building_counts={"residential": 3},
        source_kind=kind,
        source_magnitude_a=1.0 if kind == "current" else 230.0,
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(
                position_xy=(200.0, 0.0), feed_lead=lead
            ),
            probe=MeasurementProbeConfig(position_xy=(123.6, 0.0), lead=lead),
        ),
    )
    return TnNetworkGenerator(cfg, seed=1).build()


# audit W2A-10
@pytest.mark.parametrize("leads", [False, True], ids=["galvanic", "inductive-leads"])
def test_w2a_10_generated_voltage_world_reads_like_the_current_world(leads) -> None:
    """Base: measured_impedance refused the voltage record (ValueError).

    Method 1: the ratio of the two readings of ``measured_impedance`` is 1.
    Method 2: the reading of the voltage world by hand, (phi_E - phi_S)/I
    with the solved current, from the potentials of the result.
    """
    from groundfield.generators import measured_impedance
    from groundfield.generators.measurement import MEASUREMENT_S_TERMINAL

    freqs = [50.0, 1000.0] if leads else [50.0]
    eng = gf.create_engine(
        backend="image", segment_length=2.0, frequencies=freqs, kernel="point"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_i = eng.solve(_generated("current", leads=leads))
        r_v = eng.solve(_generated("voltage", leads=leads))
    z_i = measured_impedance(r_i)
    z_v = measured_impedance(r_v)
    z_v_named = measured_impedance(r_v, source=r_v.sources[0].name)
    assert z_v_named == z_v
    rec = r_v.sources[0]
    for k in range(len(freqs)):
        assert abs(z_v[k] / z_i[k] - 1.0) < 1e-12
        if leads:
            u_s = r_v.metadata["junction_potentials"][MEASUREMENT_S_TERMINAL][k]
        else:
            probe = next(n for n in r_v.electrode_potentials if n.startswith("probe_"))
            u_s = r_v._cluster_potential(probe)[1][k]
        u_e = r_v._cluster_potential(rec.attached_to)[1][k]
        assert abs((u_e - u_s) / rec.current[k] / z_v[k] - 1.0) < 1e-12


# audit W2A-10
def test_w2a_10_measured_impedance_of_a_voltage_record_without_current_raises() -> None:
    from groundfield.generators import measured_impedance
    from groundfield.solver.result import SourceRecord

    res = gf.FieldResult(
        backend="custom",
        frequencies=[50.0],
        sources=[
            SourceRecord(name="v", kind="voltage", attached_to="g1", voltage=[1.0])
        ],
    )
    with pytest.raises(ValueError, match="no solved source current"):
        measured_impedance(res, probe="g1")
    with pytest.raises(ValueError, match="no solved source current"):
        measured_impedance(res, probe="g1", source="v")


# audit W2A-10
def test_w2a_10_source_kind_description_no_longer_calls_voltage_rare() -> None:
    from groundfield.generators.tn_network import TnNetworkConfig

    text = TnNetworkConfig.model_fields["source_kind"].description
    assert "rare" not in text
    assert "measured_impedance" in text
