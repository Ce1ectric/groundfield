"""Impressed voltage sources (modified nodal analysis).

Up to 0.15.0 a world with a :class:`~groundfield.sources.VoltageSource`
solved to all-zero potentials with a warning; up to the base of the
calendar release it raised :class:`NotImplementedError` (finding 10.51).
Since 2026.11 every backend on the shared solve pipeline (the image
family including the cross-layer path, ``mom``, ``mom_sommerfeld``)
solves it: the source current is an unknown of the nodal system, with one
constraint row $\\varphi_a - \\varphi_r = U$ per voltage source.

Independent cross-checks used below (at least two methods each):

* linearity — a 1 A current-source solve scaled by $U / Z_\\text{in}$;
* the transfer-impedance matrix of current-source solves, inverted by
  hand for several voltage sources, and the mutual-resistance matrix of
  :func:`~groundfield.solver.mutual.solve_mutual_matrix` for a
  two-terminal source;
* closed forms: Dwight's rod formula and the hemispherical far field
  $\\varphi(r) = \\rho I / (2 \\pi r)$, and the series connection of a
  lumped internal resistance.
"""

# user decision of 2026-11-14: the open item "voltage sources" is part of
# the 2026.11 release scope (BIG_RELEASE_PLAN, stream w2src).

from __future__ import annotations

import cmath
import json
import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.sources import VoltageSource

pytestmark = pytest.mark.finding("10.51")

SEG = 0.5
U_PHASOR = 230.0 * cmath.exp(1j * math.radians(30.0))
#: Round-off bound of the comparison with a scaled current-source solve
#: (measured: at most 1.4e-15, evidence/w2src/validate_voltage_sources.txt).
RTOL = 1e-12


def _soil(rho: float = 100.0) -> gf.HomogeneousSoil:
    return gf.HomogeneousSoil(rho=rho)


def _two_layer() -> gf.TwoLayerSoil:
    return gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=2.0)


def _three_layer() -> gf.MultiLayerSoil:
    return gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=100.0, h=4.0),
            gf.SoilLayer(rho=400.0, h=3.0),
            gf.SoilLayer(rho=50.0, h=None),
        ]
    )


def _rod_world(soil=None, *, length: float = 3.0) -> gf.World:
    w = gf.create_world(soil=soil if soil is not None else _soil())
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=length)
    return w


def _two_rods(soil=None, *, d: float = 20.0) -> gf.World:
    w = _rod_world(soil)
    gf.create_electrode(w, "rod", name="g2", position=(d, 0.0, 0.0), length=2.0)
    return w


def _engine(**kw) -> gf.Engine:
    kw = {"segment_length": SEG, "frequencies": [50.0], **kw}
    return gf.create_engine(**kw)


def _solve(world: gf.World, **kw) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world.solve(_engine(**kw))


def _voltage(world: gf.World, attached_to: str, u: complex = U_PHASOR, **kw):
    return gf.create_source(
        world,
        attached_to=attached_to,
        kind="voltage",
        magnitude=abs(u),
        phase_deg=math.degrees(cmath.phase(u)),
        **kw,
    )


def _record(res: gf.FieldResult, name: str):
    return next(r for r in res.sources if r.name == name)


def _close(a: complex, b: complex, rtol: float = RTOL) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def _check_scaled(res_v: gf.FieldResult, res_i: gf.FieldResult, vname: str, iname):
    """The voltage solve equals the 1 A solve scaled by U / Z_in."""
    rec = _record(res_v, vname)
    z_in = res_i.input_impedance(iname)
    for k in range(len(res_v.frequencies)):
        u = rec.voltage[k]
        i_v = rec.current[k]
        assert _close(i_v, u / z_in[k]), (k, i_v, u / z_in[k])
        assert _close(res_v.input_impedance(vname)[k], z_in[k])
        scale = u / z_in[k]
        for e, values in res_i.electrode_potentials.items():
            assert _close(res_v.electrode_potentials[e][k], scale * values[k])
        for e, values in res_i.electrode_currents.items():
            expected = scale * values[k]
            got = res_v.electrode_currents[e][k]
            assert abs(got - expected) <= RTOL * abs(i_v), (e, got, expected)


# ---------------------------------------------------------------------
# one voltage source against a scaled 1 A solve, every backend
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "soil"),
    [
        ("image", _soil()),
        ("image", _two_layer()),
        ("image_2layer", _two_layer()),
        ("mom", _soil()),
        ("mom", _two_layer()),
        ("mom_sommerfeld", _soil()),
        ("mom_sommerfeld", _three_layer()),
    ],
    ids=["image", "image-2l", "image_2layer", "mom", "mom-2l", "mom_s", "mom_s-3l"],
)
def test_voltage_source_is_a_scaled_current_source(backend, soil) -> None:
    w_v = _two_rods(soil)
    _voltage(w_v, "g1", name="v1")
    w_i = _two_rods(soil)
    gf.create_source(w_i, attached_to="g1", name="i1", magnitude=1.0)
    res_v = _solve(w_v, backend=backend)
    res_i = _solve(w_i, backend=backend)
    _check_scaled(res_v, res_i, "v1", "i1")
    # The impressed voltage is the potential of the feeding cluster.
    assert _close(res_v.electrode_potentials["g1"][0], U_PHASOR, 1e-10)
    # The passive rod floats: zero net current, the potential of the field.
    assert res_v.electrode_currents["g2"][0] == 0j


@pytest.mark.parametrize("backend", ["image_2layer", "mom"])
def test_voltage_source_on_the_cross_layer_path(backend) -> None:
    """A rod through the layer interface (ADR-0007 kernel, one owner per layer)."""
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0)
    w_v = _two_rods(soil)
    gf.create_electrode(w_v, "rod", name="deep", position=(5.0, 0.0, 0.5), length=6.0)
    _voltage(w_v, "deep", name="v1")
    w_i = _two_rods(soil)
    gf.create_electrode(w_i, "rod", name="deep", position=(5.0, 0.0, 0.5), length=6.0)
    gf.create_source(w_i, attached_to="deep", name="i1", magnitude=1.0)
    res_v = _solve(w_v, backend=backend)
    res_i = _solve(w_i, backend=backend)
    if backend == "image_2layer":
        assert res_v.metadata.get("layer_owners", {}).get("deep") == 2
    _check_scaled(res_v, res_i, "v1", "i1")


def _buried_rods() -> gf.World:
    w = gf.create_world(soil=_soil())
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="g2", position=(20.0, 0.0, 0.5), length=2.0)
    return w


def _network(world: gf.World) -> None:
    """A PEN-like inductive galvanic conductor and an isolated inductive lead."""
    gf.create_electrode(world, "ring", name="ring", center=(40.0, 0.0, 0.5), radius=2.0)
    gf.create_conductor(
        world,
        name="pen",
        start="g1",
        end="ring",
        conductor_type="pen",
        cross_section=95e-6,
        resistivity=2.82e-8,
        discretize_segment_length=5.0,
        coupling_to_soil="galvanic",
        inductance_model="neumann",
    )
    gf.create_conductor(
        world,
        name="lead",
        start="g1",
        end="g2",
        conductor_type="generic",
        cross_section=16e-6,
        discretize_segment_length=5.0,
        coupling_to_soil="isolated",
        inductance_model="neumann",
    )


@pytest.mark.parametrize("backend", ["image", "mom"])
@pytest.mark.parametrize("model", ["perfect_mirror", "carson_series"])
def test_voltage_source_with_inductive_conductors_at_50_hz_and_1_khz(
    backend, model
) -> None:
    freqs = [0.0, 50.0, 1000.0]
    w_v = _buried_rods()
    _network(w_v)
    _voltage(w_v, "g1", name="v1")
    w_i = _buried_rods()
    _network(w_i)
    gf.create_source(w_i, attached_to="g1", name="i1", magnitude=1.0)
    kw = {"backend": backend, "frequencies": freqs, "earth_inductive_model": model}
    res_v = _solve(w_v, **kw)
    res_i = _solve(w_i, **kw)
    _check_scaled(res_v, res_i, "v1", "i1")
    z = res_v.input_impedance("v1")
    # The input impedance depends on frequency (the inductive branches).
    assert abs(z[2] - z[0]) > 1e-3 * abs(z[0])
    current = _record(res_v, "v1").current
    assert abs(current[2] - current[0]) > 1e-3 * abs(current[0])


def test_voltage_source_on_a_junction_with_an_internal_resistance() -> None:
    """A lumped series resistance from a junction: I = U / (R_i + Z_E)."""
    r_i = 12.5
    w = _rod_world()
    gf.create_junction(w, name="J", position=(0.0, 0.0, 0.0))
    gf.create_conductor(
        w,
        name="r_internal",
        start="J",
        end="g1",
        cross_section=50e-6,
        lumped_series_resistance_ohm=r_i,
    )
    _voltage(w, "J", name="v1")
    res = _solve(w)
    w_i = _rod_world()
    gf.create_source(w_i, attached_to="g1", magnitude=1.0)
    z_e = _solve(w_i).input_impedance()[0]
    i_v = _record(res, "v1").current[0]
    assert _close(i_v, U_PHASOR / (r_i + z_e))
    assert _close(res.input_impedance("v1")[0], r_i + z_e)
    junction = res.metadata["junction_potentials"]["J"][0]
    assert _close(junction, U_PHASOR, 1e-10)
    assert _close(res.electrode_potentials["g1"][0], z_e * i_v)


@pytest.mark.parametrize("driven", ["g1", "grid"])
def test_voltage_source_with_a_grid_of_one_owner_per_wire(driven) -> None:
    """Owner split of the image family (Q30): a driven or a floating grid."""

    def world() -> gf.World:
        w = _rod_world()
        gf.create_electrode(
            w,
            "grid_mesh",
            name="grid",
            corner=(6.0, -5.0, 0.5),
            size=(10.0, 10.0),
            n_x=2,
            n_y=2,
        )
        return w

    w_v = world()
    _voltage(w_v, driven, name="v1")
    w_i = world()
    gf.create_source(w_i, attached_to=driven, name="i1", magnitude=1.0)
    res_v = _solve(w_v)
    res_i = _solve(w_i)
    assert res_v.metadata["wire_owners"]["grid"] == 6
    _check_scaled(res_v, res_i, "v1", "i1")
    passive = "grid" if driven == "g1" else "g1"
    # The passive electrode floats: exactly zero net current.
    assert res_v.electrode_currents[passive][0] == 0j


def test_post_processing_reads_the_solved_source_current() -> None:
    """split_factor and touch_voltage_spectrum use the solved current."""
    from groundfield.postprocess.current_balance import split_factor

    def world() -> gf.World:
        w = _buried_rods()
        gf.create_conductor(
            w,
            name="lead",
            start="g1",
            end="g2",
            cross_section=16e-6,
            lumped_series_resistance_ohm=5.0,
        )
        return w

    w_v = world()
    _voltage(w_v, "g1", name="v1")
    w_i = world()
    gf.create_source(w_i, attached_to="g1", name="i1", magnitude=1.0)
    res_v, res_i = _solve(w_v), _solve(w_i)
    s_v, s_i = split_factor(res_v, w_v), split_factor(res_i, w_i)
    assert 0.0 < s_i.real < 1.0
    assert _close(s_v, s_i)
    u_v = gf.touch_voltage_spectrum(res_v, w_v, electrode="g1", currents=[100.0])
    u_i = gf.touch_voltage_spectrum(res_i, w_i, electrode="g1", currents=[100.0])
    assert u_i > 0.0 and u_v == pytest.approx(u_i, rel=1e-12)


# ---------------------------------------------------------------------
# two-terminal voltage source, several sources, superposition
# ---------------------------------------------------------------------


@pytest.mark.parametrize("soil", [_soil(), _two_layer()], ids=["hom", "2l"])
def test_two_terminal_voltage_source_against_the_mutual_resistance_matrix(
    soil,
) -> None:
    from groundfield.solver.mutual import solve_mutual_matrix

    w = _two_rods(soil, d=15.0)
    _voltage(w, "g1", name="v1", return_to="g2")
    res = _solve(w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = solve_mutual_matrix(w, _engine(), ["g1", "g2"], symmetrize=False)
    z_loop = Z[0, 0] + Z[1, 1] - Z[0, 1] - Z[1, 0]
    i_v = _record(res, "v1").current[0]
    assert _close(i_v, U_PHASOR / z_loop, 1e-10)
    cur = res.electrode_currents
    assert _close(cur["g1"][0], i_v) and _close(cur["g2"][0], -i_v)
    pot = res.electrode_potentials
    assert _close(pot["g1"][0] - pot["g2"][0], U_PHASOR, 1e-10)
    # Second method: the two-terminal current-source solve.
    w_i = _two_rods(soil, d=15.0)
    gf.create_source(w_i, attached_to="g1", return_to="g2", name="i1", magnitude=1.0)
    _check_scaled(res, _solve(w_i), "v1", "i1")


def _three_rods() -> gf.World:
    w = _two_rods()
    gf.create_electrode(w, "rod", name="g3", position=(0.0, 12.0, 0.0), length=2.5)
    return w


def test_several_voltage_and_current_sources_superpose() -> None:
    """Superposition: other voltage sources shorted (0 V), current sources open."""
    u1, u2 = U_PHASOR, 100.0 * cmath.exp(-1j * math.radians(120.0))
    i3 = 5.0 * cmath.exp(1j * math.radians(45.0))

    def world(a: complex, b: complex, c: complex | None) -> gf.World:
        w = _three_rods()
        _voltage(w, "g1", a, name="v1")
        _voltage(w, "g2", b, name="v2")
        if c is not None:
            gf.create_source(
                w,
                attached_to="g3",
                name="i3",
                magnitude=abs(c),
                phase_deg=math.degrees(cmath.phase(c)),
            )
        return w

    full = _solve(world(u1, u2, i3))
    parts = [
        _solve(world(u1, 0.0, None)),
        _solve(world(0.0, u2, None)),
        _solve(world(0.0, 0.0, i3)),
    ]
    for e in ("g1", "g2", "g3"):
        p = sum(r.electrode_potentials[e][0] for r in parts)
        assert abs(full.electrode_potentials[e][0] - p) <= 1e-10 * abs(u1)
        c = sum(r.electrode_currents[e][0] for r in parts)
        assert abs(full.electrode_currents[e][0] - c) <= 1e-10 * abs(i3)
    for v in ("v1", "v2"):
        c = sum(_record(r, v).current[0] for r in parts)
        assert abs(_record(full, v).current[0] - c) <= 1e-10 * abs(i3)


def test_several_voltage_sources_against_the_transfer_impedance_matrix() -> None:
    """Second method: invert the 3 x 3 matrix of three 1 A solves by hand."""
    u = np.array([U_PHASOR, 0.0, 40.0 * cmath.exp(1j * 0.3)])
    names = ("g1", "g2", "g3")
    Z = np.zeros((3, 3), dtype=complex)
    for j, node in enumerate(names):
        w = _three_rods()
        gf.create_source(w, attached_to=node, magnitude=1.0)
        res = _solve(w)
        Z[:, j] = [res.electrode_potentials[n][0] for n in names]
    currents = np.linalg.solve(Z, u)
    w = _three_rods()
    for node, uk in zip(names, u, strict=True):
        _voltage(w, node, complex(uk), name=f"v_{node}")
    res = _solve(w)
    for node, ik in zip(names, currents, strict=True):
        got = _record(res, f"v_{node}").current[0]
        assert abs(got - ik) <= 1e-10 * np.abs(currents).max(), (node, got, ik)
        # A one-terminal voltage source feeds exactly its electrode.
        assert abs(res.electrode_currents[node][0] - ik) <= 1e-10 * abs(ik) + 1e-14


def test_voltage_source_is_no_manual_counter_source() -> None:
    """The 0.15.0 counter-source warning looks at current sources only."""
    w = _two_rods()
    gf.create_source(w, attached_to="g1", return_to="g2", name="i1", magnitude=5.0)
    _voltage(w, "g2", -5.0, name="v2")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        res = w.solve(_engine())
    assert _close(res.electrode_potentials["g2"][0], -5.0, 1e-10)


def test_zero_volt_source_is_an_ideal_short_to_remote_earth() -> None:
    """0 V at g2 with a current source at g1: g2 is held at remote earth."""
    w = _two_rods(d=8.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    _voltage(w, "g2", 0.0, name="v0")
    res = _solve(w)
    assert abs(res.electrode_potentials["g2"][0]) < 1e-12
    i_short = _record(res, "v0").current[0]
    # The short draws current back from the soil around g2.
    assert i_short.real < 0.0
    assert _close(res.electrode_currents["g2"][0], i_short)


# ---------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------


def test_voltage_driven_rod_against_dwight_and_the_hemispherical_far_field() -> None:
    from groundfield.references import dwight1936

    rho, length = 100.0, 3.0
    w = _rod_world(_soil(rho), length=length)
    _voltage(w, "g1", name="v1")
    res = _solve(w, segment_length=0.25)
    i_v = _record(res, "v1").current[0]
    radius = w.electrodes[0].wire_radius
    r_dwight = dwight1936.rod(rho, length, radius)
    # Dwight's formula is the uniform-current (average-potential) rod too:
    # measured -0.012 % at ds = 0.5 ... 0.125 m.
    assert abs(i_v - U_PHASOR / r_dwight) <= 1e-3 * abs(i_v)
    # Far field of any electrode: the hemispherical electrode.
    r = 2000.0
    phi = res.potential(np.array([[r, 0.0, 0.0]]))[0]
    assert abs(phi - rho * i_v / (2.0 * math.pi * r)) <= 1e-5 * abs(phi)


# ---------------------------------------------------------------------
# results: accessors, JSON, diagnose, sweep
# ---------------------------------------------------------------------


def test_source_record_and_tables_carry_voltage_and_current() -> None:
    w = _two_rods()
    _voltage(w, "g1", name="v1")
    gf.create_source(w, attached_to="g2", name="i2", magnitude=2.0)
    res = _solve(w, frequencies=[50.0, 150.0])
    rec = _record(res, "v1")
    assert rec.kind == "voltage"
    # The source is given as magnitude and phase; the polar round trip is
    # exact with glibc on Linux x86-64 and one unit in the last place off
    # (6.2e-17) with the system libm on macOS arm64.
    assert len(rec.voltage) == 2
    assert all(_close(u, U_PHASOR, rtol=1e-15) for u in rec.voltage)
    assert len(rec.current) == 2 and rec.current[0] != 0j
    irec = _record(res, "i2")
    assert irec.voltage == [] and irec.current == [2.0 + 0j, 2.0 + 0j]
    df = res.to_dataframe("sources")
    row = df[(df.source == "v1") & (df.frequency_hz == 50.0)].iloc[0]
    assert row.current_a == rec.current[0]
    assert row.voltage_v == rec.voltage[0]
    assert _close(row.input_impedance_ohm, U_PHASOR / rec.current[0])
    with pytest.raises(ValueError, match="several sources"):
        res.input_impedance()


def test_direct_backend_call_reports_the_source_current() -> None:
    from groundfield.solver.image import solve_image

    w = _rod_world()
    _voltage(w, "g1", name="v1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = solve_image(w, _engine())
    assert _record(res, "v1").current[0] != 0j
    assert _close(res.input_impedance()[0], U_PHASOR / _record(res, "v1").current[0])


def test_json_round_trip_of_world_and_result() -> None:
    w = _two_rods()
    _voltage(w, "g1", name="v1", return_to="g2")
    w2 = gf.World.model_validate_json(w.model_dump_json())
    assert isinstance(w2.sources[0], VoltageSource)
    res = _solve(w)
    res2 = _solve(w2)
    assert res2.electrode_potentials == res.electrode_potentials
    back = gf.FieldResult.model_validate_json(res.model_dump_json())
    assert back.sources == res.sources
    assert back.input_impedance("v1") == res.input_impedance("v1")
    payload = json.loads(res.model_dump_json())
    assert payload["sources"][0]["kind"] == "voltage"


def test_result_file_round_trip(tmp_path) -> None:
    w = _two_rods()
    _voltage(w, "g1", name="v1")
    res = _solve(w, frequencies=[50.0, 1000.0])
    path = tmp_path / "result.json"
    gf.save_result(res, path)
    back = gf.load_result(path)
    assert back.sources == res.sources
    assert back.input_impedance() == res.input_impedance()


def test_voltage_record_without_a_solved_current_raises() -> None:
    from groundfield.solver.result import SourceRecord

    res = gf.FieldResult(
        backend="custom",
        frequencies=[50.0],
        sources=[
            SourceRecord(name="v", kind="voltage", attached_to="g1", voltage=[1.0])
        ],
    )
    with pytest.raises(ValueError, match="no solved source current"):
        res.input_impedance()


def test_diagnose_counts_one_unknown_per_voltage_source() -> None:
    from groundfield.diagnostics import diagnose, expected_segments

    w = _two_rods()
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    base = expected_segments(w, _engine())["nodal_unknowns"]
    _voltage(w, "g2", name="v2")
    assert expected_segments(w, _engine())["nodal_unknowns"] == base + 1
    _voltage(w, "g1", name="v12", return_to="g2", u=5.0)
    assert expected_segments(w, _engine())["nodal_unknowns"] == base + 2
    report = diagnose(w, _engine())
    assert not [f for f in report.findings if f.severity == "warning"]
    res = _solve(w)
    # v12 fixes g1 at 5 V above g2, which v2 holds at U.
    pot = res.electrode_potentials
    assert _close(pot["g1"][0] - pot["g2"][0], 5.0, 1e-10)
    assert _close(pot["g2"][0], U_PHASOR, 1e-10)


def test_sweep_reads_the_input_impedance_of_a_voltage_source() -> None:
    from groundfield.postprocess.sweep import sweep

    def factory(rho: float) -> gf.World:
        w = _rod_world(_soil(rho))
        _voltage(w, "g1", name="v1")
        return w

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sweep(factory, _engine(frequencies=[50.0]), axes={"rho": [100.0, 200.0]})
    z = df["Z_re"].to_numpy()
    assert z[0] > 0.0 and z[1] == pytest.approx(2.0 * z[0], rel=1e-12)
    # The voltage of the quotient is the impressed voltage.
    assert df["U_E_re"].to_numpy() == pytest.approx([U_PHASOR.real] * 2, rel=1e-12)


# ---------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------


def test_short_circuited_voltage_source_raises() -> None:
    w = _two_rods()
    gf.create_conductor(w, name="bond", start="g1", end="g2")  # ideal bond
    _voltage(w, "g1", name="v1", return_to="g2")
    with pytest.raises(ValueError, match=r"(?s)'v1'.*short-circuited"):
        w.solve(_engine())


@pytest.mark.parametrize(
    "layout",
    ["parallel_to_earth", "bonded_parallel", "triangle", "through_earth"],
)
def test_loop_of_voltage_sources_raises(layout) -> None:
    w = _three_rods()
    if layout == "parallel_to_earth":
        _voltage(w, "g1", name="va")
        _voltage(w, "g1", name="vb", u=10.0)
    elif layout == "bonded_parallel":
        gf.create_conductor(w, name="bond", start="g1", end="g3")  # ideal bond
        _voltage(w, "g1", name="va")
        _voltage(w, "g3", name="vb", u=10.0)
    elif layout == "triangle":
        _voltage(w, "g1", name="va", return_to="g2")
        _voltage(w, "g2", name="vb", return_to="g3", u=10.0)
        _voltage(w, "g3", name="vc", return_to="g1", u=5.0)
    else:
        _voltage(w, "g1", name="va", return_to="g2")
        _voltage(w, "g1", name="vb", u=10.0)
        _voltage(w, "g2", name="vc", u=5.0)
    with pytest.raises(ValueError, match=r"(?s)loop of voltage sources.*v[abc]"):
        w.solve(_engine())


def test_world_of_zero_volt_sources_only_raises() -> None:
    w = _rod_world()
    _voltage(w, "g1", 0.0, name="v0")
    with pytest.raises(ValueError, match="no source with a non-zero magnitude"):
        w.solve(_engine())


def test_non_finite_voltage_is_rejected() -> None:
    w = _rod_world()
    src = _voltage(w, "g1", name="v1")
    with pytest.raises(ValueError):
        src.magnitude = float("nan")


def test_fem_solves_voltage_sources() -> None:
    """Since its conforming model fem runs the network of the shared
    pipeline (user decision Q37; it refused them up to then): a voltage
    source equals the scaled current-source solve."""
    from groundfield.solver.fem import solve_fem

    w = _rod_world()
    _voltage(w, "g1", name="v1")
    res = _solve(w, backend="fem")
    ref = _rod_world()
    gf.create_source(ref, attached_to="g1", magnitude=1.0)
    z = _solve(ref, backend="fem").input_impedance()[0]
    assert abs(res.sources[0].current[0] / (U_PHASOR / z) - 1.0) < RTOL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = solve_fem(w, _engine(backend="fem"))
    assert direct.sources[0].current == res.sources[0].current


def test_capabilities_declare_voltage_sources() -> None:
    from groundfield.solver import get_backend

    for name in ("image", "image_2layer", "mom", "mom_sommerfeld", "fem"):
        assert get_backend(name).capabilities.voltage_sources, name
