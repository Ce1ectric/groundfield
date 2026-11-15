"""Fixes of audit W2R2 (wave 2, round 2: the internal impedance of voltage sources).

Each test names the finding it guards (``W2R2-NN`` of the audit report) and
fails on the base ``34ad269`` for the reason given in its docstring; the
tests of W2R2-01, -02, -04 and -05 start from the auditor's proposals.

* W2R2-01 — worlds and results pickled before user decision D1 (by
  ``8c83d7c``) load as ideal sources: ``VoltageSource`` and ``SourceRecord``
  fill the two fields of the internal impedance with 0 when a pickle lacks
  them. The fixture ``tests/data/pre_d1_voltage_source_pickles.pkl`` was
  written by ``8c83d7c`` (``dump_pre_d1_pickles.py`` of the stream
  evidence) and holds two worlds with voltage sources, their results at 0
  and 50 Hz, the JSON of each world and the input impedances ``8c83d7c``
  computed.
* W2R2-02 — a galvanic ``mom`` / ``mom_sommerfeld`` world whose voltage
  source has an inductance factorises its segment system once for all
  frequencies (Schur complement on the source rows) instead of once per
  frequency.
* W2R2-04 — the short-circuit message of a source behind $Z_q$.
* W2R2-05 — ``measured_impedance`` and the layout reading prefer the single
  current source of a result, else its single voltage source.
* W2R2-10 — a subnormal internal resistance or inductance is rejected.
"""

from __future__ import annotations

import math
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import groundfield as gf
from groundfield.solver.result import SourceRecord
from groundfield.sources import VoltageSource

# The audit ids (W2R2-NN) are not finding ids of the review catalogue; the
# tests carry them in their names and comments, the module the finding of
# the voltage sources (as tests/test_voltage_source_audit_w2a.py).
pytestmark = pytest.mark.finding("10.51")

HOM = gf.HomogeneousSoil(rho=100.0)
TWO = gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=2.0)
FIXTURE = Path(__file__).parent / "data" / "pre_d1_voltage_source_pickles.pkl"
D1_FIELDS = ("internal_resistance_ohm", "internal_inductance_h")
#: Round-off bound of the comparisons between two exact solution methods
#: (measured at most 4.4e-15 here, evidence/fw2r2/w2r2_02_*.out).
RTOL = 1e-12


def _solve(world: gf.World, **kw) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # frequency order, small-segment notices
        engine = gf.create_engine(
            **{"segment_length": 0.5, "frequencies": [50.0], **kw}
        )
        return world.solve(engine)


def _two_rods(soil=HOM) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "rod", name="g2", position=(20.0, 0.0, 0.0), length=2.0)
    return w


def _record(res: gf.FieldResult, name: str) -> SourceRecord:
    return next(r for r in res.sources if r.name == name)


def _close(a: complex, b: complex, rtol: float = RTOL) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b))


# W2R2-01 ----------------------------------------------------------------


def _old_pickle_state(obj, *fields):
    """The pickle state of ``obj`` as a release before D1 wrote it (fields absent)."""
    state = dict(obj.__getstate__())
    state["__dict__"] = {k: v for k, v in state["__dict__"].items() if k not in fields}
    state["__pydantic_fields_set__"] = set(state["__pydantic_fields_set__"]) - set(
        fields
    )
    return state


@pytest.fixture(scope="module")
def pre_d1() -> dict:
    # A file of this repository written by 8c83d7c (see the module docstring).
    return pickle.loads(FIXTURE.read_bytes())  # noqa: S301


# audit W2R2-01
def test_w2r2_01_a_voltage_source_pickled_before_d1_loads_as_an_ideal_source():
    """W2R2-01: pickles written before D1 raise AttributeError after it.

    On 34ad269 ``World.check`` (``float(getattr(src, field))``) and
    ``impressed_voltages`` raise ``AttributeError: 'VoltageSource' object
    has no attribute 'internal_resistance_ohm'`` for a world unpickled from
    8c83d7c; ``FieldResult.input_impedance`` and ``to_dataframe("sources")``
    raise the same for a ``SourceRecord``. JSON of the same objects loads
    (defaults 0 / 0).
    """
    src = VoltageSource(name="v", attached_to="g1", magnitude=230.0)
    old = VoltageSource.__new__(VoltageSource)
    old.__setstate__(_old_pickle_state(src, *D1_FIELDS))
    assert old == src
    assert list(old.__dict__) == list(VoltageSource.model_fields)
    assert old.model_fields_set == src.model_fields_set
    w = gf.create_world(soil=HOM)
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
    w.sources.append(old)
    w.check()
    res = w.solve(gf.create_engine(frequencies=[50.0]))
    assert res.input_impedance()[0] == pytest.approx(35.9905, abs=1e-3)

    rec = SourceRecord(
        name="v", kind="voltage", attached_to="g1", voltage=[10.0], current=[2.0]
    )
    old_rec = SourceRecord.__new__(SourceRecord)
    old_rec.__setstate__(_old_pickle_state(rec, *D1_FIELDS))
    assert old_rec == rec
    old_res = gf.FieldResult(backend="custom", frequencies=[50.0], sources=[old_rec])
    assert old_res.input_impedance() == [5.0 + 0j]


# audit W2R2-01
def test_w2r2_01_a_current_state_is_restored_unchanged():
    """The upgrade touches pickles without the two fields only."""
    src = VoltageSource(
        name="v",
        attached_to="g1",
        magnitude=230.0,
        internal_resistance_ohm=2.0,
        internal_inductance_h=1e-3,
    )
    back = pickle.loads(pickle.dumps(src))
    assert back == src
    assert back.model_fields_set == src.model_fields_set
    # A pickle that lacks one of the two fields gets 0 for that one only.
    half = VoltageSource.__new__(VoltageSource)
    half.__setstate__(_old_pickle_state(src, "internal_inductance_h"))
    assert (half.internal_resistance_ohm, half.internal_inductance_h) == (2.0, 0.0)


# audit W2R2-01
@pytest.mark.parametrize("name", ["pre_d1_one_terminal", "pre_d1_two_terminal"])
def test_w2r2_01_worlds_pickled_by_8c83d7c_check_and_solve(pre_d1, name):
    """Base 34ad269: ``check()`` and the solve raise AttributeError."""
    assert pre_d1["commit"] == "8c83d7c"
    world = pre_d1["worlds"][name]
    from_json = gf.World.model_validate_json(pre_d1["json"][name])
    assert world.sources == from_json.sources
    for src in world.sources:
        if src.kind == "voltage":
            assert src.internal_impedance(50.0) == 0j
    world.check()
    engine = gf.create_engine(segment_length=0.5, frequencies=[0.0, 50.0])
    res = world.solve(engine)
    ref = from_json.solve(engine)
    for src in world.sources:
        assert res.input_impedance(src.name) == ref.input_impedance(src.name)
        for z, z_old in zip(
            res.input_impedance(src.name),
            pre_d1["input_impedance"][name][src.name],
            strict=True,
        ):
            # The stored value is the line-kernel solve of 8c83d7c; the
            # continuous switch and the far images of stream w3xl moved it
            # by 6.8e-7 (declared up to 6.0e-4 in golden), which is not
            # what this test checks.
            assert _close(z, z_old, rtol=1e-5), (src.name, z, z_old)
    # The world pickles again with the fields and loads unchanged.
    again = pickle.loads(pickle.dumps(world))
    assert again.sources == from_json.sources


# audit W2R2-01
@pytest.mark.parametrize("name", ["pre_d1_one_terminal", "pre_d1_two_terminal"])
def test_w2r2_01_results_pickled_by_8c83d7c_read_back(pre_d1, name):
    """Base 34ad269: ``input_impedance`` and ``to_dataframe('sources')`` raise
    AttributeError for the pickled ``SourceRecord``."""
    pytest.importorskip("pandas")
    res = pre_d1["results"][name]
    for record in res.sources:
        assert (record.internal_resistance_ohm, record.internal_inductance_h) == (
            0.0,
            0.0,
        )
        assert record.internal_impedance(50.0) == 0j
        assert res.input_impedance(record.name) == pytest.approx(
            pre_d1["input_impedance"][name][record.name], rel=1e-15
        )
    table = res.to_dataframe("sources")
    voltage_rows = table[table.kind == "voltage"]
    assert len(voltage_rows) == len(res.frequencies)
    assert (voltage_rows.internal_resistance_ohm == 0.0).all()
    assert (voltage_rows.internal_inductance_h == 0.0).all()
    # The result pickles again and reads the same.
    again = pickle.loads(pickle.dumps(res))
    assert again.sources == res.sources


# W2R2-02 ----------------------------------------------------------------


def _counting_solves(monkeypatch, min_rows: int = 20) -> list[int]:
    """Record the row count of every ``numpy.linalg.solve`` on a large system."""
    import groundfield.solver.mom as mom

    calls: list[int] = []
    real_solve = np.linalg.solve

    def counting_solve(a, b):
        if np.ndim(a) == 2 and a.shape[0] > min_rows:
            calls.append(a.shape[0])
        return real_solve(a, b)

    monkeypatch.setattr(mom.np.linalg, "solve", counting_solve)
    return calls


def _grid_world(soil=HOM, **source) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w,
        "grid_mesh",
        name="grid",
        corner=(0.0, 0.0, 0.5),
        size=(10.0, 10.0),
        n_x=2,
        n_y=2,
    )
    gf.create_source(w, attached_to="grid", kind="voltage", magnitude=230.0, **source)
    return w


# audit W2R2-02
@pytest.mark.parametrize("backend", ["mom", "mom_sommerfeld"])
def test_w2r2_02_a_galvanic_mom_world_with_a_source_inductance_factorises_once(
    monkeypatch, backend
):
    """W2R2-02: the segment system of mom is factorised once per frequency.

    A galvanic world (no inductive conductor) whose voltage source has an
    internal inductance takes the complex path at every frequency: on
    34ad269 ``numpy.linalg.solve`` runs once per frequency on the full
    segment-level matrix (20 frequencies, 1488 segments: 1.53 s against
    0.054 s for the R-only source, x28; evidence ``w2r2_02_cost_base.out``).
    Since the fix: one real factorisation with V + 2 right-hand sides and a
    V x V system per frequency.
    """
    calls = _counting_solves(monkeypatch)
    w = _grid_world(internal_resistance_ohm=1.0, internal_inductance_h=1e-3)
    freqs = [10.0, 50.0, 150.0, 250.0, 1000.0]
    _solve(w, backend=backend, frequencies=freqs)
    assert len(calls) == 1, f"{len(calls)} factorisations of the {calls[0]}-row system"


# audit W2R2-02
@pytest.mark.parametrize("backend", ["mom", "mom_sommerfeld"])
@pytest.mark.parametrize("soil", [HOM, TWO], ids=["hom", "two_layer"])
def test_w2r2_02_the_one_factorisation_equals_the_unit_current_solve(backend, soil):
    """I = U / (Z_q + Z_in) and every potential the 1 A solve times I."""
    freqs = [0.0, 10.0, 50.0, 1000.0]
    res_v = _solve(
        _grid_world(
            soil, name="v", internal_resistance_ohm=1.0, internal_inductance_h=1e-3
        ),
        backend=backend,
        frequencies=freqs,
    )
    w_i = _grid_world(soil)
    w_i.sources = []
    gf.create_source(w_i, attached_to="grid", magnitude=1.0, name="i")
    res_i = _solve(w_i, backend=backend, frequencies=freqs)
    rec = _record(res_v, "v")
    z_in = res_i.input_impedance("i")
    for k, f in enumerate(freqs):
        z_q = complex(1.0, 2.0 * math.pi * f * 1e-3)
        i_v = rec.voltage[k] / (z_q + z_in[k])
        assert _close(rec.current[k], i_v), (f, rec.current[k], i_v)
        assert _close(res_v.input_impedance("v")[k], z_in[k]), f
        for e, values in res_i.electrode_potentials.items():
            assert _close(res_v.electrode_potentials[e][k], i_v * values[k]), (f, e)
        got = np.array([p.current[k] for p in res_v.point_sources])
        ref = np.array([p.current[k] for p in res_i.point_sources])
        assert np.abs(got - i_v * ref).max() <= RTOL * abs(i_v) * np.abs(ref).max()


def _network() -> gf.World:
    """Two reactive and one resistive voltage source, a current source, a
    finite conductor and a floating rod."""
    w = gf.create_world(soil=HOM)
    for name, x, y, length in (
        ("g1", 0.0, 0.0, 3.0),
        ("g2", 8.0, 0.0, 3.0),
        ("g3", 0.0, 9.0, 2.0),
        ("g4", 15.0, 5.0, 2.0),
        ("g5", -6.0, -4.0, 2.0),
    ):
        gf.create_electrode(w, "rod", name=name, position=(x, y, 0.0), length=length)
    gf.create_conductor(
        w,
        name="c1",
        start="g3",
        end="g4",
        conductor_type="generic",
        cross_section=16e-6,
    )
    gf.create_source(
        w,
        attached_to="g1",
        kind="voltage",
        magnitude=100.0,
        phase_deg=30.0,
        name="v1",
        internal_inductance_h=2e-3,
    )
    gf.create_source(
        w,
        attached_to="g2",
        return_to="g3",
        kind="voltage",
        magnitude=50.0,
        name="v2",
        internal_resistance_ohm=3.0,
        internal_inductance_h=5e-4,
    )
    gf.create_source(
        w,
        attached_to="g4",
        kind="voltage",
        magnitude=20.0,
        name="v3",
        internal_resistance_ohm=4.0,
    )
    gf.create_source(w, attached_to="g2", magnitude=2.0, phase_deg=-10.0, name="i1")
    return w


# audit W2R2-02
def test_w2r2_02_several_sources_against_the_transfer_impedance_matrix():
    """(Z_T + diag Z_q) I = U - Z_T,i I_i with Z_T from 1 A solves (mom)."""
    freqs = [0.0, 50.0, 1000.0]
    world = _network()
    res = _solve(world, backend="mom", frequencies=freqs)
    names = ["v1", "v2", "v3"]
    by_name = {s.name: s for s in world.sources}
    unit = {}
    for s in world.sources:
        w = _network()
        w.sources = [
            gf.CurrentSource(
                name=s.name,
                attached_to=s.attached_to,
                return_to=s.return_to,
                magnitude=1.0,
            )
        ]
        unit[s.name] = _solve(w, backend="mom", frequencies=freqs)

    def terminal_voltage(r: gf.FieldResult, s, k: int) -> complex:
        # Every terminal is a rod of its own cluster (c1 is a finite conductor).
        phi = r.electrode_potentials
        return phi[s.attached_to][k] - (
            0j if s.return_to is None else phi[s.return_to][k]
        )

    i_src = 2.0 * np.exp(-1j * math.radians(10.0))
    for k, f in enumerate(freqs):
        z_t = np.array(
            [[terminal_voltage(unit[m], by_name[n], k) for m in names] for n in names]
        )
        z_ti = np.array([terminal_voltage(unit["i1"], by_name[n], k) for n in names])
        z_q = np.diag([by_name[n].internal_impedance(f) for n in names])
        u = np.array([_record(res, n).voltage[k] for n in names])
        expected = np.linalg.solve(z_t + z_q, u - z_ti * i_src)
        got = np.array([_record(res, n).current[k] for n in names])
        assert np.allclose(got, expected, rtol=RTOL, atol=0), (f, got, expected)
        for e in world.electrodes:
            phi = (
                sum(
                    unit[n].electrode_potentials[e.name][k] * got[j]
                    for j, n in enumerate(names)
                )
                + unit["i1"].electrode_potentials[e.name][k] * i_src
            )
            assert _close(res.electrode_potentials[e.name][k], phi), (f, e.name)


# audit W2R2-02
def test_w2r2_02_the_frequencies_are_solved_independently():
    """Every frequency f > 0 of a sweep equals its single-frequency solve, bit
    for bit, in any order. f = 0 inside a sweep equals the R-only source to
    round-off; a solve at f = 0 alone is the R-only solve, bit for bit."""
    world = _network()
    freqs = [1000.0, 0.0, 50.0]
    res = _solve(world, backend="mom", frequencies=freqs)
    for k, f in enumerate(freqs):
        if f == 0.0:
            continue
        single = _solve(world, backend="mom", frequencies=[f])
        for n in ("v1", "v2", "v3"):
            assert _record(res, n).current[k] == _record(single, n).current[0]
        assert res.electrode_potentials["g5"][k] == single.electrode_potentials["g5"][0]
    r_only = _network()
    for s in r_only.sources:
        if s.kind == "voltage":
            s.internal_inductance_h = 0.0
    dc = _solve(r_only, backend="mom", frequencies=[0.0])
    for n in ("v1", "v2", "v3"):
        assert _close(_record(res, n).current[1], _record(dc, n).current[0], 1e-14)
    for e, values in dc.electrode_potentials.items():
        assert _close(res.electrode_potentials[e][1], values[0], 1e-14), e
    only_dc = _solve(world, backend="mom", frequencies=[0.0])
    for n in ("v1", "v2", "v3"):
        assert _record(only_dc, n).current == _record(dc, n).current
    assert only_dc.electrode_potentials == dc.electrode_potentials


# audit W2R2-02
def test_w2r2_02_ideal_and_resistive_sources_keep_their_path(monkeypatch):
    """No reactive source: one solve for all frequencies, as before."""
    calls = _counting_solves(monkeypatch)
    _solve(_grid_world(), backend="mom", frequencies=[10.0, 50.0, 1000.0])
    _solve(
        _grid_world(internal_resistance_ohm=2.0),
        backend="mom",
        frequencies=[10.0, 50.0, 1000.0],
    )
    assert len(calls) == 2


# W2R2-04 ----------------------------------------------------------------


# audit W2R2-04
def test_w2r2_04_a_short_circuited_source_behind_an_internal_impedance_is_described_as_such():
    """W2R2-04: the rejection of a voltage source with R > 0 whose terminals share
    an ideally bonded cluster says it "would impose a voltage across one
    equipotential" (34ad269), which holds for an ideal source only; with Z_q its
    current U / Z_q would circulate through the bond (the ADR-0014 wording)."""
    w = _two_rods()
    gf.create_conductor(w, name="bond", start="g1", end="g2")
    gf.create_source(
        w,
        attached_to="g1",
        return_to="g2",
        kind="voltage",
        magnitude=10.0,
        internal_resistance_ohm=5.0,
    )
    with pytest.raises(ValueError) as info:
        w.check()
    msg = str(info.value)
    assert "impose a voltage across one equipotential" not in msg
    assert "internal" in msg
    assert "U/Z_q would circulate through the bond and the source" in msg
    assert "none through the soil" in msg
    assert "short-circuited" in msg


# audit W2R2-04
@pytest.mark.parametrize(
    "impedance",
    [{"internal_resistance_ohm": 5.0}, {"internal_inductance_h": 1e-3}],
    ids=["R", "L"],
)
def test_w2r2_04_the_same_node_message_of_a_source_behind_an_impedance(impedance):
    w = _two_rods()
    w.sources = [
        VoltageSource(
            name="v", attached_to="g1", return_to="g1", magnitude=10.0, **impedance
        )
    ]
    with pytest.raises(ValueError) as info:
        w.check()
    msg = str(info.value)
    assert "impose a voltage across one equipotential" not in msg
    assert "U/Z_q would circulate through the source" in msg
    assert "short-circuited" in msg


# audit W2R2-04
def test_w2r2_04_an_ideal_short_circuited_source_keeps_its_message():
    w = _two_rods()
    gf.create_conductor(w, name="bond", start="g1", end="g2")
    gf.create_source(
        w, attached_to="g1", return_to="g2", kind="voltage", magnitude=10.0
    )
    with pytest.raises(ValueError, match="impose a voltage across one equipotential"):
        w.check()


# W2R2-05 ----------------------------------------------------------------


def _v_and_i_world(v_first: bool = True) -> gf.World:
    w = _two_rods()
    v = dict(attached_to="g1", kind="voltage", magnitude=1.0, name="v")
    i = dict(attached_to="g2", magnitude=1.0, name="i")
    for spec in (v, i) if v_first else (i, v):
        gf.create_source(w, **spec)
    return w


# audit W2R2-05
@pytest.mark.parametrize("v_first", [True, False], ids=["v_first", "i_first"])
def test_w2r2_05_measured_impedance_keeps_reading_the_only_current_source(v_first):
    """W2R2-05: up to 8c83d7c ``measured_impedance`` read the only current
    source of a result that also carries a voltage source; on 34ad269 it
    raises ``ValueError: The result carries 2 sources (v, i); pass
    source=<name>``. The coordinator restores the base rule."""
    from groundfield.generators import measured_impedance

    res = _solve(_v_and_i_world(v_first))
    assert measured_impedance(res, probe="g1") == measured_impedance(
        res, probe="g1", source="i"
    )


# audit W2R2-05
def test_w2r2_05_a_voltage_only_result_reads_its_voltage_source():
    from groundfield.generators import measured_impedance

    w = _two_rods()
    gf.create_source(w, attached_to="g1", kind="voltage", magnitude=230.0, name="v")
    res = _solve(w)
    assert measured_impedance(res, probe="g2") == measured_impedance(
        res, probe="g2", source="v"
    )


# audit W2R2-05
@pytest.mark.parametrize(
    "sources, listed",
    [
        (
            [("i", "current", "g1"), ("i2", "current", "g2"), ("v", "voltage", "g1")],
            "2 current sources (i, i2)",
        ),
        (
            [("v", "voltage", "g1"), ("v2", "voltage", "g2")],
            "2 voltage sources (v, v2)",
        ),
    ],
    ids=["two_current", "two_voltage"],
)
def test_w2r2_05_several_sources_of_one_kind_still_need_a_name(sources, listed):
    """Base 34ad269: the message counted the sources of both kinds together."""
    from groundfield.generators import measured_impedance

    w = _two_rods()
    for name, kind, node in sources:
        gf.create_source(w, attached_to=node, kind=kind, magnitude=1.0, name=name)
    res = _solve(w)
    with pytest.raises(ValueError, match=re.escape(listed)) as info:
        measured_impedance(res, probe="g2")
    assert "source=<name>" in str(info.value)
    named = measured_impedance(res, probe="g2", source=sources[0][0])
    assert all(math.isfinite(abs(z)) for z in named)


# audit W2R2-05
def test_w2r2_05_one_current_and_two_voltage_sources_read_the_current_source():
    from groundfield.generators import measured_impedance

    w = _two_rods()
    gf.create_source(w, attached_to="g1", kind="voltage", magnitude=1.0, name="v")
    gf.create_source(w, attached_to="g2", kind="voltage", magnitude=2.0, name="v2")
    gf.create_source(w, attached_to="g1", magnitude=1.0, name="i")
    res = _solve(w)
    assert measured_impedance(res, probe="g2") == measured_impedance(
        res, probe="g2", source="i"
    )


def _layout_world_with_a_voltage_source_first():
    """The W2A-01 layout, driven by its 1 A source and a 230 V source at the
    same substation electrode, the voltage source listed first."""
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
    world = layout.to_world(soil=HOM, source_magnitude_a=1.0, seed=0)
    (current,) = world.sources
    world.sources = [
        VoltageSource(name="v_aux", attached_to=current.attached_to, magnitude=230.0),
        current,
    ]
    return layout, world


# audit W2R2-05
def test_w2r2_05_the_layout_reading_prefers_the_current_source():
    """34ad269 took the first record of either kind at the substation anchor,
    here the voltage source (its solved current instead of 1 A)."""
    layout, world = _layout_world_with_a_voltage_source_first()
    res = _solve(world, segment_length=1.0)
    i_src = _record(res, world.sources[1].name).current[0]
    assert i_src == 1.0
    z = layout.measured_grounding_impedance(res)
    assert z == layout.measured_grounding_impedance(res, source_magnitude_a=1.0)
    balance = layout.verify_current_balance(res)
    assert balance["expected_positive_a"] == pytest.approx(1.0)


# W2R2-10 ----------------------------------------------------------------

TINY = sys.float_info.min  # == numpy.finfo(float).tiny, the smallest normal float


# audit W2R2-10
@pytest.mark.parametrize("field", D1_FIELDS)
@pytest.mark.parametrize("value", [1e-310, 5e-324, TINY / 2])
def test_w2r2_10_a_subnormal_internal_impedance_is_rejected(field, value):
    """W2R2-10: R = 1e-310 Ohm was accepted and turned the solve into nan
    (an ideal source and a source behind R on one rod)."""
    with pytest.raises(ValidationError, match="smallest normal"):
        VoltageSource(name="v", attached_to="g1", magnitude=1.0, **{field: value})
    src = VoltageSource(name="v", attached_to="g1", magnitude=1.0)
    with pytest.raises(ValidationError, match="smallest normal"):
        setattr(src, field, value)
    # model_copy(update=...) skips the validators; World.check does not.
    w = _two_rods()
    w.sources = [src.model_copy(update={field: value})]
    with pytest.raises(ValueError, match="smallest normal"):
        w.check()


# audit W2R2-10
@pytest.mark.parametrize("field", D1_FIELDS)
@pytest.mark.parametrize("value", [0.0, -0.0, TINY, 1e-300])
def test_w2r2_10_zero_and_normal_values_stay_accepted(field, value):
    src = VoltageSource(name="v", attached_to="g1", magnitude=1.0, **{field: value})
    w = _two_rods()
    w.sources = [src]
    w.check()


# audit W2R2-10
def test_w2r2_10_the_nan_case_of_the_audit_is_rejected_before_the_solve():
    """The auditor's world: an ideal 230 V source and a 231 V source behind R
    on one rod. R = 1e-310 Ohm solved to nan; R = 1e-300 Ohm is exact."""
    w = _two_rods()
    gf.create_source(w, attached_to="g1", kind="voltage", magnitude=230.0, name="va")
    with pytest.raises(ValidationError, match="smallest normal"):
        gf.create_source(
            w,
            attached_to="g1",
            kind="voltage",
            magnitude=231.0,
            name="vb",
            internal_resistance_ohm=1e-310,
        )
    gf.create_source(
        w,
        attached_to="g1",
        kind="voltage",
        magnitude=231.0,
        name="vb",
        internal_resistance_ohm=1e-300,
    )
    res = _solve(w)
    assert _record(res, "vb").current[0] == pytest.approx(1e300, rel=1e-12)
    assert all(math.isfinite(abs(v[0])) for v in res.electrode_potentials.values())
