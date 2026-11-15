"""Internal impedance of a voltage source (user decision D1, 2026-11-15).

A :class:`~groundfield.sources.VoltageSource` carries an internal series
impedance $Z_q(f) = R + j\\omega L$ (``internal_resistance_ohm``,
``internal_inductance_h``, both 0 by default: the ideal source). It is the
one diagonal entry of the source's constraint row,

$$
\\varphi_a - \\varphi_r + Z_q I_v = U_v ,
$$

with $I_v$ flowing from the source into ``attached_to``, so the terminal
voltage is $U_v - Z_q I_v$. Every backend that solves voltage sources
honours it (image family including the cross-layer path, ``mom``,
``mom_sommerfeld``; galvanic and inductive solves, several frequencies).

Independent cross-checks (two methods each):

* the pattern available before D1: an ideal source at a junction and a
  lumped series resistance from the junction to the electrode;
* the 1 A current-source solve with $I = U / (Z_q + Z_\\text{in})$ and every
  potential scaled by $I$ — at 0 / 50 / 1000 Hz, two backends;
* the transfer-impedance matrix of 1 A solves,
  $(Z_T + \\operatorname{diag} Z_q)\\,\\mathbf I = \\mathbf U$, and the
  mutual-resistance matrix for a two-terminal source.
"""

# user decision D1 (2026-11-15): internal impedance of VoltageSource
# (HANDOVER section 5, "follow every recommendation").

from __future__ import annotations

import cmath
import json
import math
import pickle
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

import groundfield as gf
from groundfield.sources import CurrentSource, VoltageSource

pytestmark = pytest.mark.finding("10.51")

SEG = 0.5
FREQS = [0.0, 50.0, 1000.0]
U_PHASOR = 230.0 * cmath.exp(1j * math.radians(30.0))
R_Q = 2.5  # Ω
L_Q = 4e-3  # H: ωL = 1.26 Ω at 50 Hz, 25.1 Ω at 1 kHz
#: Round-off bound of the comparisons with a scaled current-source solve
#: (measured at most 4.0e-15, evidence/fw2src/d1_validation.txt).
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


def _z_q(f: float, r: float = R_Q, l: float = L_Q) -> complex:
    return complex(r, 2.0 * math.pi * f * l)


def _two_rods(soil=None, *, d: float = 20.0, z: float = 0.0) -> gf.World:
    w = gf.create_world(soil=soil if soil is not None else _soil())
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, z), length=3.0)
    gf.create_electrode(w, "rod", name="g2", position=(d, 0.0, z), length=2.0)
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


def _engine(**kw) -> gf.Engine:
    return gf.create_engine(**{"segment_length": SEG, "frequencies": FREQS, **kw})


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


def _check_against_unit_solve(res_v, res_i, vname, iname, r=R_Q, l=L_Q) -> None:
    """I = U / (Z_q + Z_in) and every quantity is the 1 A solve times I."""
    rec = _record(res_v, vname)
    z_in = res_i.input_impedance(iname)
    for k, f in enumerate(res_v.frequencies):
        i_expected = rec.voltage[k] / (_z_q(f, r, l) + z_in[k])
        i_v = rec.current[k]
        assert _close(i_v, i_expected), (f, i_v, i_expected)
        # input_impedance is the impedance seen by the source, without Z_q.
        assert _close(res_v.input_impedance(vname)[k], z_in[k]), f
        for e, values in res_i.electrode_potentials.items():
            assert _close(res_v.electrode_potentials[e][k], i_v * values[k]), (f, e)
        for e, values in res_i.electrode_currents.items():
            got = res_v.electrode_currents[e][k]
            assert abs(got - i_v * values[k]) <= RTOL * abs(i_v), (f, e)


# ---------------------------------------------------------------------
# the model: fields, validation, factory
# ---------------------------------------------------------------------


def test_the_fields_default_to_the_ideal_source() -> None:
    src = VoltageSource(name="v", attached_to="g1", magnitude=1.0)
    assert src.internal_resistance_ohm == 0.0
    assert src.internal_inductance_h == 0.0
    assert src.internal_impedance(50.0) == 0j
    src = VoltageSource(
        name="v",
        attached_to="g1",
        magnitude=1.0,
        internal_resistance_ohm=R_Q,
        internal_inductance_h=L_Q,
    )
    assert src.internal_impedance(0.0) == complex(R_Q, 0.0)
    assert src.internal_impedance(50.0) == _z_q(50.0)


@pytest.mark.parametrize("field", ["internal_resistance_ohm", "internal_inductance_h"])
@pytest.mark.parametrize("value", [-1e-3, math.inf, math.nan])
def test_a_negative_or_non_finite_internal_impedance_is_rejected(field, value) -> None:
    with pytest.raises(ValidationError):
        VoltageSource(name="v", attached_to="g1", magnitude=1.0, **{field: value})
    src = VoltageSource(name="v", attached_to="g1", magnitude=1.0)
    with pytest.raises(ValidationError):
        setattr(src, field, value)
    # model_copy(update=...) skips the validators; World.check catches it.
    w = _two_rods()
    w.sources = [src.model_copy(update={field: value})]
    with pytest.raises(ValueError, match=field):
        w.check()


def test_create_source_takes_the_internal_impedance_of_a_voltage_source() -> None:
    w = _two_rods()
    src = gf.create_source(
        w,
        attached_to="g1",
        kind="voltage",
        magnitude=230.0,
        internal_resistance_ohm=R_Q,
        internal_inductance_h=L_Q,
    )
    assert isinstance(src, VoltageSource)
    assert (src.internal_resistance_ohm, src.internal_inductance_h) == (R_Q, L_Q)
    with pytest.raises(ValueError, match="current source has no internal impedance"):
        gf.create_source(
            w, attached_to="g2", magnitude=1.0, internal_resistance_ohm=1.0
        )
    with pytest.raises(ValidationError):
        CurrentSource(
            name="i", attached_to="g1", magnitude=1.0, internal_inductance_h=1.0
        )


# ---------------------------------------------------------------------
# method 1: the junction + lumped-resistance pattern
# ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["image", "mom"])
@pytest.mark.parametrize("network", [False, True], ids=["rods", "inductive"])
def test_internal_resistance_equals_a_junction_and_a_lumped_resistance(
    backend, network
) -> None:
    """Before D1 a series resistance was a junction and a lumped conductor."""
    w_q = _two_rods(z=0.5)
    w_j = _two_rods(z=0.5)
    if network:
        _network(w_q)
        _network(w_j)
    _voltage(w_q, "g1", name="v1", internal_resistance_ohm=R_Q)
    gf.create_junction(w_j, name="J", position=(0.0, 0.0, 0.0))
    gf.create_conductor(
        w_j,
        name="r_q",
        start="J",
        end="g1",
        cross_section=50e-6,
        lumped_series_resistance_ohm=R_Q,
    )
    _voltage(w_j, "J", name="v1")
    kw = {"backend": backend, "earth_inductive_model": "carson_series"}
    res_q, res_j = _solve(w_q, **kw), _solve(w_j, **kw)
    for k in range(len(FREQS)):
        i_q = _record(res_q, "v1").current[k]
        assert _close(i_q, _record(res_j, "v1").current[k]), k
        assert _close(
            res_q.input_impedance("v1")[k] + R_Q, res_j.input_impedance("v1")[k]
        )
        for e, values in res_j.electrode_potentials.items():
            assert _close(res_q.electrode_potentials[e][k], values[k]), (k, e)
        for e, values in res_j.electrode_currents.items():
            got = res_q.electrode_currents[e][k]
            assert abs(got - values[k]) <= RTOL * abs(i_q), (k, e)


# ---------------------------------------------------------------------
# method 2: the 1 A solve plus I = U / (Z_q + Z_in), every backend
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
def test_internal_impedance_against_the_unit_current_solve(backend, soil) -> None:
    w_v = _two_rods(soil)
    _voltage(
        w_v, "g1", name="v1", internal_resistance_ohm=R_Q, internal_inductance_h=L_Q
    )
    w_i = _two_rods(soil)
    gf.create_source(w_i, attached_to="g1", name="i1", magnitude=1.0)
    res_v, res_i = _solve(w_v, backend=backend), _solve(w_i, backend=backend)
    _check_against_unit_solve(res_v, res_i, "v1", "i1")
    # The galvanic world is frequency-independent; the source makes the
    # current frequency-dependent (one complex solve per frequency).
    current = _record(res_v, "v1").current
    assert abs(current[2] - current[0]) > 0.1 * abs(current[0])
    # The terminal voltage is U - Z_q I.
    for k, f in enumerate(FREQS):
        u_t = U_PHASOR - _z_q(f) * current[k]
        assert _close(res_v.electrode_potentials["g1"][k], u_t, 1e-10)


@pytest.mark.parametrize("backend", ["image", "mom"])
@pytest.mark.parametrize("model", ["perfect_mirror", "carson_series", "sommerfeld"])
def test_internal_impedance_with_inductive_conductors(backend, model) -> None:
    w_v = _two_rods(z=0.5)
    _network(w_v)
    _voltage(
        w_v, "g1", name="v1", internal_resistance_ohm=R_Q, internal_inductance_h=L_Q
    )
    w_i = _two_rods(z=0.5)
    _network(w_i)
    gf.create_source(w_i, attached_to="g1", name="i1", magnitude=1.0)
    kw = {"backend": backend, "earth_inductive_model": model}
    _check_against_unit_solve(_solve(w_v, **kw), _solve(w_i, **kw), "v1", "i1")


@pytest.mark.parametrize("backend", ["image_2layer", "mom"])
def test_internal_impedance_on_the_cross_layer_path(backend) -> None:
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0)

    def world() -> gf.World:
        w = _two_rods(soil)
        gf.create_electrode(w, "rod", name="deep", position=(5.0, 0.0, 0.5), length=6.0)
        return w

    w_v = world()
    _voltage(
        w_v, "deep", name="v1", internal_resistance_ohm=R_Q, internal_inductance_h=L_Q
    )
    w_i = world()
    gf.create_source(w_i, attached_to="deep", name="i1", magnitude=1.0)
    res_v, res_i = _solve(w_v, backend=backend), _solve(w_i, backend=backend)
    if backend == "image_2layer":
        assert res_v.metadata.get("layer_owners", {}).get("deep") == 2
    _check_against_unit_solve(res_v, res_i, "v1", "i1")


def test_frequency_order_does_not_matter() -> None:
    """The per-frequency loop of a galvanic world with a reactive source."""
    w = _two_rods()
    _voltage(w, "g1", name="v1", internal_resistance_ohm=R_Q, internal_inductance_h=L_Q)
    res = _solve(w, frequencies=[1000.0, 0.0, 50.0])
    for k, f in enumerate([1000.0, 0.0, 50.0]):
        single = _solve(w, frequencies=[f])
        assert _record(res, "v1").current[k] == _record(single, "v1").current[0]
        assert res.electrode_potentials["g2"][k] == single.electrode_potentials["g2"][0]


def test_at_zero_frequency_the_inductance_drops_out() -> None:
    w_rl = _two_rods()
    _voltage(
        w_rl, "g1", name="v1", internal_resistance_ohm=R_Q, internal_inductance_h=L_Q
    )
    w_r = _two_rods()
    _voltage(w_r, "g1", name="v1", internal_resistance_ohm=R_Q)
    res_rl = _solve(w_rl, frequencies=[0.0])
    res_r = _solve(w_r, frequencies=[0.0])
    assert _record(res_rl, "v1").current == _record(res_r, "v1").current
    assert res_rl.electrode_potentials == res_r.electrode_potentials


# ---------------------------------------------------------------------
# several sources, two-terminal source: matrix methods
# ---------------------------------------------------------------------


def _three_rods() -> gf.World:
    w = _two_rods()
    gf.create_electrode(w, "rod", name="g3", position=(0.0, 12.0, 0.0), length=2.5)
    return w


def test_several_sources_against_the_transfer_impedance_matrix() -> None:
    """(Z_T + diag Z_q) I = U with Z_T from three 1 A solves."""
    u = np.array([U_PHASOR, 0.0, 40.0 * cmath.exp(1j * 0.3)])
    rq = [R_Q, 0.0, 7.0]
    lq = [L_Q, 1e-3, 0.0]
    names = ("g1", "g2", "g3")
    f = 50.0
    Z = np.zeros((3, 3), dtype=complex)
    for j, node in enumerate(names):
        w = _three_rods()
        gf.create_source(w, attached_to=node, magnitude=1.0)
        res = _solve(w, frequencies=[f])
        Z[:, j] = [res.electrode_potentials[n][0] for n in names]
    z_q = np.diag([_z_q(f, r, l) for r, l in zip(rq, lq, strict=True)])
    currents = np.linalg.solve(Z + z_q, u)
    w = _three_rods()
    for node, uk, r, l in zip(names, u, rq, lq, strict=True):
        _voltage(
            w,
            node,
            complex(uk),
            name=f"v_{node}",
            internal_resistance_ohm=r,
            internal_inductance_h=l,
        )
    res = _solve(w, frequencies=[f])
    for node, ik in zip(names, currents, strict=True):
        got = _record(res, f"v_{node}").current[0]
        assert abs(got - ik) <= 1e-10 * np.abs(currents).max(), (node, got, ik)


@pytest.mark.parametrize("soil", [_soil(), _two_layer()], ids=["hom", "2l"])
def test_two_terminal_source_against_the_mutual_resistance_matrix(soil) -> None:
    from groundfield.solver.mutual import solve_mutual_matrix

    w = _two_rods(soil, d=15.0)
    _voltage(
        w,
        "g1",
        name="v1",
        return_to="g2",
        internal_resistance_ohm=R_Q,
        internal_inductance_h=L_Q,
    )
    res = _solve(w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = solve_mutual_matrix(w, _engine(), ["g1", "g2"], symmetrize=False)
    z_loop = Z[0, 0] + Z[1, 1] - Z[0, 1] - Z[1, 0]
    for k, f in enumerate(FREQS):
        i_v = _record(res, "v1").current[k]
        assert _close(i_v, U_PHASOR / (_z_q(f) + z_loop), 1e-10), f
        assert _close(res.input_impedance("v1")[k], z_loop, 1e-10), f


# ---------------------------------------------------------------------
# loops and short circuits
# ---------------------------------------------------------------------


def test_a_resistive_source_opens_a_loop_of_voltage_sources() -> None:
    """Two sources from one rod to remote earth: one ideal, one behind R."""
    w = _two_rods()
    _voltage(w, "g1", name="va")
    u_b = 200.0 + 0j
    _voltage(w, "g1", u_b, name="vb", internal_resistance_ohm=R_Q)
    res = _solve(w, frequencies=[50.0])
    # The ideal source fixes the rod at U_a; R carries (U_b - U_a) / R.
    assert _close(res.electrode_potentials["g1"][0], U_PHASOR, 1e-10)
    i_b = _record(res, "vb").current[0]
    assert _close(i_b, (u_b - U_PHASOR) / R_Q, 1e-10)
    # KCL: both sources together feed the rod's leakage.
    i_a = _record(res, "va").current[0]
    assert _close(i_a + i_b, res.electrode_currents["g1"][0], 1e-10)


def test_a_ring_of_sources_with_one_resistive_source_solves() -> None:
    w = _three_rods()
    _voltage(w, "g1", name="va", return_to="g2")
    _voltage(w, "g2", 10.0, name="vb", return_to="g3")
    _voltage(w, "g3", 5.0, name="vc", return_to="g1", internal_resistance_ohm=R_Q)
    res = _solve(w, frequencies=[50.0])
    # The ring current through vc: U_c + U_a + U_b drop across R
    # (the ideal sources fix g1 - g2 and g2 - g3).
    pot = res.electrode_potentials
    assert _close(pot["g1"][0] - pot["g2"][0], U_PHASOR, 1e-10)
    i_c = _record(res, "vc").current[0]
    u_t = pot["g3"][0] - pot["g1"][0]
    assert _close(u_t, 5.0 - R_Q * i_c, 1e-10)


def test_an_inductance_alone_does_not_open_a_loop() -> None:
    """At f = 0 the impedance of L vanishes: the loop stays singular there."""
    w = _two_rods()
    _voltage(w, "g1", name="va")
    _voltage(w, "g1", 10.0, name="vb", internal_inductance_h=L_Q)
    with pytest.raises(
        ValueError, match=r"(?s)loop of voltage sources.*internal_resistance_ohm"
    ):
        w.check()


def test_a_short_circuited_source_with_internal_impedance_raises() -> None:
    w = _two_rods()
    gf.create_conductor(w, name="bond", start="g1", end="g2")  # ideal bond
    _voltage(w, "g1", name="v1", return_to="g2", internal_resistance_ohm=R_Q)
    with pytest.raises(ValueError, match=r"(?s)'v1'.*short-circuited"):
        w.check()


# ---------------------------------------------------------------------
# results: records, tables, round trips, post-processing
# ---------------------------------------------------------------------


def _solved_pair() -> tuple[gf.World, gf.FieldResult]:
    w = _two_rods()
    _voltage(w, "g1", name="v1", internal_resistance_ohm=R_Q, internal_inductance_h=L_Q)
    gf.create_source(w, attached_to="g2", name="i2", magnitude=2.0)
    return w, _solve(w, frequencies=[50.0, 1000.0])


def test_source_record_and_table_carry_the_internal_impedance() -> None:
    _, res = _solved_pair()
    rec = _record(res, "v1")
    assert (rec.internal_resistance_ohm, rec.internal_inductance_h) == (R_Q, L_Q)
    irec = _record(res, "i2")
    assert (irec.internal_resistance_ohm, irec.internal_inductance_h) == (0.0, 0.0)
    df = res.to_dataframe("sources")
    row = df[(df.source == "v1") & (df.frequency_hz == 1000.0)].iloc[0]
    assert row.internal_resistance_ohm == R_Q
    assert row.internal_inductance_h == L_Q
    z_in = U_PHASOR / rec.current[1] - _z_q(1000.0)
    assert _close(row.input_impedance_ohm, z_in)
    current_rows = df[df.source == "i2"]
    assert current_rows.internal_resistance_ohm.isna().all()
    assert current_rows.internal_inductance_h.isna().all()


def test_json_and_pickle_round_trips() -> None:
    w, res = _solved_pair()
    w2 = gf.World.model_validate_json(w.model_dump_json())
    assert w2.sources[0] == w.sources[0]
    payload = json.loads(w.model_dump_json())
    assert payload["sources"][0]["internal_inductance_h"] == L_Q
    w3 = pickle.loads(pickle.dumps(w))
    assert w3.sources[0].internal_resistance_ohm == R_Q
    res2 = _solve(w2, frequencies=[50.0, 1000.0])
    assert res2.electrode_potentials == res.electrode_potentials
    for back in (
        gf.FieldResult.model_validate_json(res.model_dump_json()),
        pickle.loads(pickle.dumps(res)),
    ):
        assert back.sources == res.sources
        assert back.input_impedance("v1") == res.input_impedance("v1")


def test_result_file_round_trip(tmp_path) -> None:
    _, res = _solved_pair()
    path = tmp_path / "result.json"
    gf.save_result(res, path)
    back = gf.load_result(path)
    assert back.sources == res.sources
    assert back.input_impedance("v1") == res.input_impedance("v1")


def test_a_record_written_before_d1_reads_as_an_ideal_source() -> None:
    from groundfield.solver.result import SourceRecord

    rec = SourceRecord.model_validate(
        {
            "name": "v",
            "kind": "voltage",
            "attached_to": "g1",
            "voltage": [10.0],
            "current": [2.0],
        }
    )
    assert (rec.internal_resistance_ohm, rec.internal_inductance_h) == (0.0, 0.0)
    res = gf.FieldResult(backend="custom", frequencies=[50.0], sources=[rec])
    assert res.input_impedance() == [5.0 + 0j]


def test_diagnose_and_expected_segments() -> None:
    from groundfield.diagnostics import diagnose, expected_segments

    w = _two_rods()
    gf.create_source(w, attached_to="g2", magnitude=1.0)
    base = expected_segments(w, _engine())["nodal_unknowns"]
    _voltage(w, "g1", name="v1", internal_resistance_ohm=R_Q, internal_inductance_h=L_Q)
    assert expected_segments(w, _engine())["nodal_unknowns"] == base + 1
    report = diagnose(w, _engine())
    assert not [f for f in report.findings if f.severity == "warning"]


def test_sweep_reads_the_impedance_seen_by_the_source() -> None:
    from groundfield.postprocess.sweep import sweep

    def factory(rho: float) -> gf.World:
        w = gf.create_world(soil=_soil(rho))
        gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
        _voltage(
            w, "g1", name="v1", internal_resistance_ohm=R_Q, internal_inductance_h=L_Q
        )
        return w

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sweep(factory, _engine(frequencies=[50.0]), axes={"rho": [100.0, 200.0]})
    z = df["Z_re"].to_numpy() + 1j * df["Z_im"].to_numpy()
    # Z is the earth (proportional to rho), without Z_q.
    assert abs(z[0].imag) < 1e-12 * abs(z[0])
    assert z[1] == pytest.approx(2.0 * z[0], rel=1e-12)
    i = df["I_re"].to_numpy() + 1j * df["I_im"].to_numpy()
    u_e = df["U_E_re"].to_numpy() + 1j * df["U_E_im"].to_numpy()
    for k in range(2):
        assert _close(i[k], U_PHASOR / (_z_q(50.0) + z[k]))
        # The voltage of the quotient is the terminal voltage U - Z_q I.
        assert _close(u_e[k], U_PHASOR - _z_q(50.0) * i[k])


def test_post_processing_reads_the_solved_current() -> None:
    from groundfield.postprocess.current_balance import split_factor

    def world(**kw) -> gf.World:
        w = _two_rods(z=0.5)
        gf.create_conductor(
            w,
            name="lead",
            start="g1",
            end="g2",
            cross_section=16e-6,
            lumped_series_resistance_ohm=5.0,
        )
        _voltage(w, "g1", name="v1", **kw)
        return w

    w_q, w_0 = world(internal_resistance_ohm=R_Q, internal_inductance_h=L_Q), world()
    res_q, res_0 = _solve(w_q, frequencies=[50.0]), _solve(w_0, frequencies=[50.0])
    assert _close(split_factor(res_q, w_q), split_factor(res_0, w_0))
    u_q = gf.touch_voltage_spectrum(res_q, w_q, electrode="g1", currents=[100.0])
    u_0 = gf.touch_voltage_spectrum(res_0, w_0, electrode="g1", currents=[100.0])
    assert u_q == pytest.approx(u_0, rel=1e-12)


def test_generated_measurement_reading_is_independent_of_the_source_impedance() -> None:
    """AP2: the reading of a generated set-up, driven through a source impedance."""
    from groundfield.generators import measured_impedance
    from groundfield.generators.measurement import (
        MeasurementInjectionConfig,
        MeasurementProbeConfig,
        MeasurementSetupConfig,
        overhead_lead,
    )
    from groundfield.generators.soil_specs import HomogeneousSoilSpec
    from groundfield.generators.tn_network import TnNetworkConfig, TnNetworkGenerator

    def world(kind: str) -> gf.World:
        cfg = TnNetworkConfig(
            soil=HomogeneousSoilSpec(rho=100.0),
            building_counts={"residential": 3},
            source_kind=kind,
            measurement=MeasurementSetupConfig(
                injection=MeasurementInjectionConfig(
                    position_xy=(200.0, 0.0), feed_lead=overhead_lead()
                ),
                probe=MeasurementProbeConfig(
                    position_xy=(123.6, 0.0), lead=overhead_lead()
                ),
            ),
        )
        return TnNetworkGenerator(cfg, seed=1).build()

    eng = gf.create_engine(
        backend="image", segment_length=2.0, frequencies=[50.0, 1000.0], kernel="point"
    )
    w_v = world("voltage")
    w_v.sources[0].internal_resistance_ohm = 10.0
    w_v.sources[0].internal_inductance_h = 1e-3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_i = eng.solve(world("current"))
        r_v = eng.solve(w_v)
    z_in = r_i.input_impedance()
    for k, f in enumerate(eng.frequencies):
        assert (
            abs(measured_impedance(r_v)[k] / measured_impedance(r_i)[k] - 1.0) < 1e-12
        )
        i_v = r_v.sources[0].current[k]
        assert _close(i_v, 1.0 / (_z_q(f, 10.0, 1e-3) + z_in[k]))


def test_fem_keeps_refusing_voltage_sources() -> None:
    from groundfield.solver import BackendCapabilityError

    w = _two_rods()
    _voltage(w, "g1", name="v1", internal_resistance_ohm=R_Q)
    with pytest.raises(BackendCapabilityError, match="voltage source"):
        _solve(w, backend="fem")
