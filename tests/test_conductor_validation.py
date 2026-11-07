"""Validation of conductor inputs (review pass 10, findings 10.43, 10.67)."""

from __future__ import annotations

import math
import warnings

import pytest

import groundfield as gf
from groundfield.conductors import Conductor


def _world():
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="a", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="b", position=(20.0, 0.0, 0.5), length=2.0)
    return w


@pytest.mark.finding("10.67")
@pytest.mark.parametrize("value", [0.0, -50e-6, math.nan, math.inf])
def test_nonpositive_or_nonfinite_cross_section_raises(value: float) -> None:
    """v0.15.0: 0 -> ZeroDivisionError at solve, negative -> active element."""
    with pytest.raises(ValueError, match="cross_section"):
        gf.create_conductor(_world(), start="a", end="b", cross_section=value)


@pytest.mark.finding("10.67")
def test_cross_section_in_mm2_warns() -> None:
    """cross_section=50 (mm² meant) was a silent ideal short of 6.7 nΩ."""
    from groundfield.conductors import CrossSectionUnitWarning

    with pytest.warns(CrossSectionUnitWarning, match="mm²"):
        gf.create_conductor(_world(), start="a", end="b", cross_section=50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CrossSectionUnitWarning)
        gf.create_conductor(_world(), start="a", end="b", cross_section=50e-6)


@pytest.mark.finding("10.67")
@pytest.mark.parametrize("field", ["resistivity", "wire_radius"])
@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_nonfinite_material_values_raise(field: str, value: float) -> None:
    """v0.15.0: resistivity=inf -> LinAlgError at solve."""
    kw = dict(name="c", start=(0.0, 0.0, 0.5), end=(1.0, 0.0, 0.5), cross_section=50e-6)
    kw[field] = value
    with pytest.raises(ValueError, match=field):
        Conductor(**kw)


@pytest.mark.finding("10.67")
def test_nonfinite_end_point_raises() -> None:
    with pytest.raises(ValueError, match="end"):
        Conductor(name="c", start=(0.0, 0.0, 0.5), end=(math.nan, 0.0, 0.5))


@pytest.mark.finding("10.43")
@pytest.mark.parametrize("coupling", ["isolated", "galvanic"])
def test_ideal_distributed_conductor_raises_with_parameter_name(coupling: str) -> None:
    """v0.15.0: LinAlgError: Singular matrix at solve time."""
    with pytest.raises(ValueError, match="cross_section"):
        gf.create_conductor(
            _world(),
            start="a",
            end="b",
            discretize_segment_length=5.0,
            coupling_to_soil=coupling,
        )


@pytest.mark.finding("10.43")
def test_finite_distributed_conductor_still_solves() -> None:
    w = _world()
    gf.create_conductor(
        w, start="a", end="b", cross_section=50e-6, discretize_segment_length=5.0
    )
    gf.create_source(w, attached_to="a")
    res = w.solve(gf.create_engine(backend="image", segment_length=0.5))
    assert math.isfinite(res.cluster_impedance("a")[0].real)
