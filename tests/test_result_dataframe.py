"""``FieldResult.to_dataframe()``: the common result query in one call.

Check question U05 (findings 10.69 / 10.72, usability): on v0.15.0 a
tidy table of potentials, currents and impedances needed a hand-written
loop over three dictionaries and two accessor methods.
"""

from __future__ import annotations

import ast
import inspect
import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver import result as result_module
from groundfield.solver.result import FieldResult

ELECTRODE_COLUMNS = [
    "frequency_hz",
    "electrode",
    "cluster",
    "potential_v",
    "current_a",
    "cluster_impedance_ohm",
]
SOURCE_COLUMNS = [
    "frequency_hz",
    "source",
    "kind",
    "attached_to",
    "return_to",
    "current_a",
    "voltage_v",
    "input_impedance_ohm",
]


def _network() -> FieldResult:
    """g1 and g2 bonded ideally (one cluster), g3 tied by a finite PEN."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="g2", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="g1", position=(4.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="g3", position=(40.0, 0.0, 0.5), length=2.0)
    gf.create_conductor(w, name="bond", start="g1", end="g2")
    gf.create_conductor(
        w,
        name="pen",
        start="g1",
        end="g3",
        cross_section=50e-6,
        resistivity=2.82e-8,
        coupling_to_soil="isolated",
    )
    gf.create_source(w, name="feed", attached_to="g1", magnitude=10.0, phase_deg=20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return w.solve(gf.create_engine(segment_length=0.5, frequencies=[0.0, 50.0]))


@pytest.mark.finding("10.69", "10.72")
def test_electrode_table_is_tidy_and_matches_the_accessors() -> None:
    res = _network()
    df = res.to_dataframe()
    assert list(df.columns) == ELECTRODE_COLUMNS
    assert len(df) == len(res.frequencies) * len(res.electrode_potentials)
    for row in df.itertuples(index=False):
        k = res.frequencies.index(row.frequency_hz)
        assert row.potential_v == res.electrode_potentials[row.electrode][k]
        assert row.current_a == res.electrode_currents[row.electrode][k]
        assert row.cluster == res.clusters[row.electrode][0]
        assert row.cluster_impedance_ohm == res.cluster_impedance(row.electrode)[k]
    bonded = df[df.electrode.isin(["g1", "g2"])]
    assert set(bonded.cluster) == {"g1"}
    assert df.loc[df.electrode == "g3", "cluster"].iloc[0] == "g3"
    assert df["potential_v"].dtype == np.complex128


@pytest.mark.finding("10.69", "10.72")
def test_source_table_carries_the_input_impedance() -> None:
    res = _network()
    df = res.to_dataframe("sources")
    assert list(df.columns) == SOURCE_COLUMNS
    assert list(df.source) == ["feed", "feed"]
    np.testing.assert_array_equal(df.input_impedance_ohm, res.input_impedance("feed"))
    np.testing.assert_allclose(
        df.current_a, [10.0 * np.exp(1j * np.deg2rad(20.0))] * 2, rtol=1e-15
    )
    assert df.attached_to.tolist() == ["g1", "g1"]
    assert df.return_to.isna().all()


def test_unknown_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="electrodes.*sources"):
        _network().to_dataframe("branches")


def test_hand_built_result_fills_gaps_with_nan() -> None:
    res = FieldResult(
        backend="hand",
        frequencies=[50.0],
        electrode_potentials={"a": [2.0 + 0j], "b": [1.0 + 0j]},
        electrode_currents={"a": [1.0 + 0j]},
    )
    df = res.to_dataframe()
    assert df.loc[df.electrode == "a", "cluster_impedance_ohm"].iloc[0] == 2.0
    assert math.isnan(df.loc[df.electrode == "b", "current_a"].iloc[0].real)
    sources = res.to_dataframe("sources")
    assert list(sources.columns) == SOURCE_COLUMNS
    assert len(sources) == 0


def test_pandas_is_imported_lazily() -> None:
    tree = ast.parse(inspect.getsource(result_module))
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
        if name.split(".")[0] == "pandas"
    ]
    assert top_level == []
