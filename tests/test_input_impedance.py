"""``FieldResult.input_impedance`` and the source snapshot (finding 10.18).

The cluster impedance divides the cluster potential by the cluster's
own *leakage* current. As soon as conductors carry part of the
injected current away (PEN, shields, leads) that is not the impedance
an instrument or a fault sees at the feeding point. The input
impedance divides by the impressed source current instead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver.result import FieldResult, SourceRecord

pytestmark = pytest.mark.finding("10.18")


def _two_rods(*, magnitude: float = 1.0, phase_deg: float = 0.0) -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="A", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="B", position=(30.0, 0.0, 0.5), length=3.0)
    gf.create_conductor(
        w,
        name="pen",
        start="A",
        end="B",
        conductor_type="pen",
        resistivity=2.82e-8,
        cross_section=50e-6,
        coupling_to_soil="isolated",
    )
    gf.create_source(w, attached_to="A", magnitude=magnitude, phase_deg=phase_deg)
    return w


def _solve(world: gf.World, **kw) -> FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world.solve(gf.create_engine(frequencies=[0.0, 50.0], **kw))


def test_single_electrode_input_equals_cluster_impedance() -> None:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="rod", magnitude=3.0)
    res = _solve(w)
    np.testing.assert_allclose(
        res.input_impedance(), res.cluster_impedance("rod"), rtol=1e-12
    )


def test_input_impedance_sees_the_conductor_path() -> None:
    res = _solve(_two_rods())
    z_in = np.asarray(res.input_impedance())
    z_cluster = np.asarray(res.cluster_impedance("A"))
    # Definition: feeding-point potential over the impressed current.
    np.testing.assert_allclose(
        z_in, np.asarray(res.electrode_potentials["A"]) / 1.0, rtol=1e-12
    )
    # Half of the current leaves through the PEN: the feeding point sees
    # roughly the two rods in parallel, the cluster impedance one rod.
    assert np.all(z_in.real < 0.6 * z_cluster.real)
    assert np.all(z_in.real > 0.4 * z_cluster.real)


def test_input_impedance_is_independent_of_source_phasor() -> None:
    z1 = _solve(_two_rods()).input_impedance()
    z2 = _solve(_two_rods(magnitude=7.0, phase_deg=40.0)).input_impedance()
    np.testing.assert_allclose(z2, z1, rtol=1e-12)


def test_source_snapshot_is_recorded_and_round_trips() -> None:
    res = _solve(_two_rods(magnitude=2.0, phase_deg=30.0))
    assert len(res.sources) == 1
    rec = res.sources[0]
    assert (rec.kind, rec.attached_to, rec.return_to) == ("current", "A", None)
    expected = 2.0 * np.exp(1j * np.deg2rad(30.0))
    np.testing.assert_allclose(rec.current, [expected, expected], rtol=1e-15)
    back = FieldResult.model_validate_json(res.model_dump_json())
    assert back.input_impedance() == res.input_impedance()


def test_several_sources_need_a_name() -> None:
    w = _two_rods()
    gf.create_source(w, attached_to="B", name="second", magnitude=1.0)
    res = _solve(w)
    with pytest.raises(ValueError, match="several current sources"):
        res.input_impedance()
    first = res.sources[0].name
    z = res.input_impedance(first)
    assert len(z) == 2
    with pytest.raises(KeyError, match="Unknown source"):
        res.input_impedance("nope")


def test_hand_built_result_without_snapshot_raises() -> None:
    res = FieldResult(
        backend="image",
        frequencies=[50.0],
        electrode_potentials={"a": [1.0 + 0j]},
        electrode_currents={"a": [1.0 + 0j]},
    )
    with pytest.raises(ValueError, match="source snapshot"):
        res.input_impedance()


def test_return_cluster_potential_is_subtracted() -> None:
    res = FieldResult(
        backend="image",
        frequencies=[50.0],
        electrode_potentials={"a": [10.0 + 0j], "b": [-4.0 + 0j]},
        electrode_currents={"a": [2.0 + 0j], "b": [-2.0 + 0j]},
        sources=[
            SourceRecord(
                name="s",
                kind="current",
                attached_to="a",
                return_to="b",
                current=[2.0 + 0j],
            )
        ],
    )
    assert res.input_impedance() == [7.0 + 0j]


def test_zero_source_current_gives_nan() -> None:
    res = FieldResult(
        backend="image",
        frequencies=[50.0],
        electrode_potentials={"a": [1.0 + 0j]},
        electrode_currents={"a": [0j]},
        sources=[SourceRecord(name="s", kind="current", attached_to="a", current=[0j])],
    )
    assert np.isnan(res.input_impedance()[0])
