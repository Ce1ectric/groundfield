"""Point form of the cross-layer path with horizontal wires (ADR-0007, 10.28).

On the cross-layer path of a two-layer soil (an electrode reaches the
interface h_1) the point form of the reaction matrix — ``kernel="point"``
of the image family and the whole ``mom`` family — averages the layered
correction on the diagonal over a 3-point rule along the *depth* axis,
+-0.387 L around every segment midpoint, whatever the segment's
orientation. For a horizontal wire the rule left the soil (a bare
``ValueError`` about negative depths deep inside ``layered_green``) or
reached across h_1 and returned negative impedances: a rod through
h_1 = 1.2 m with a 20 m strip at 0.8 m, 300/50 Ohm m, ds = 2 m: -19.37 Ohm
against 9.62 Ohm with the line kernel (v0.15.0: -14.24 Ohm). The solve now
raises :class:`BackendCapabilityError` naming the line kernel, and warns
(:class:`PointKernelInterfaceWarning`) for a horizontal wire within 0.6
segment lengths of h_1 (-3 % ... -17 % measured). Rods, cut at h_1, are
unaffected; the line kernel integrates every case exactly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver._guards import PointKernelInterfaceWarning
from groundfield.solver.registry import BackendCapabilityError

_FINDING = pytest.mark.finding("10.28")


def _rod_and_strip(strip_depth: float) -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=1.2))
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(
        w,
        "strip",
        name="strip",
        start=(5.0, 0.0, strip_depth),
        end=(25.0, 0.0, strip_depth),
    )
    gf.create_conductor(w, name="link", start="rod", end="strip")
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    return w


def _ring_below_interface(below: float) -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=2.0))
    gf.create_electrode(
        w, "ring", name="ring", center=(0.0, 0.0, 2.0 + below), radius=3.0
    )
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    return w


def _solve(world: gf.World, **engine_kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = world.solve(gf.create_engine(**engine_kw))
    hits = [r for r in rec if issubclass(r.category, PointKernelInterfaceWarning)]
    return result, hits


@_FINDING
@pytest.mark.parametrize(
    ("strip_depth", "where"),
    [(0.8, "across the layer interface"), (0.1, "above the soil surface")],
)
@pytest.mark.parametrize(
    ("engine_kw", "names"),
    [
        ({"kernel": "point"}, "kernel=None or 'line'"),
        ({"backend": "mom"}, "image_2layer"),
        ({"backend": "mom_sommerfeld"}, "image_2layer"),
    ],
)
def test_point_form_rejects_a_horizontal_wire_its_diagonal_rule_cannot_reach(
    strip_depth: float, where: str, engine_kw: dict, names: str
) -> None:
    """-19.37 Ohm (line kernel: 9.62 Ohm), or a ValueError about depths < 0."""
    # docb item, audit round 2
    with pytest.raises(BackendCapabilityError, match=where) as info:
        _solve(_rod_and_strip(strip_depth), segment_length=2.0, **engine_kw)
    assert names in str(info.value)
    assert "strip" in str(info.value)


@_FINDING
def test_line_kernel_solves_the_same_worlds() -> None:
    # docb item, audit round 2
    for depth in (0.1, 0.8):
        result, hits = _solve(_rod_and_strip(depth), segment_length=2.0)
        assert not hits
        assert 9.0 < result.input_impedance()[0].real < 12.0


@_FINDING
@pytest.mark.parametrize(("ds", "warned"), [(0.5, True), (0.25, False)])
def test_point_form_warns_for_a_horizontal_wire_next_to_the_interface(
    ds: float, warned: bool
) -> None:
    """Ring 0.2 m below h_1: -16.7 % at ds = 0.5 m (warned), -2.1 % at 0.25 m."""
    # docb item, audit round 2
    _, hits = _solve(_ring_below_interface(0.2), segment_length=ds, kernel="point")
    assert bool(hits) is warned
    if warned:
        assert "0.6 segment lengths" in str(hits[0].message)


@_FINDING
def test_rods_through_the_interface_are_unaffected() -> None:
    """Vertical segments cut at h_1: the rule is their own extent."""
    # docb item, audit round 2
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=1.0))
    gf.create_electrode(w, "rod", name="g0", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="g1", position=(6.0, 0.0, 0.0), length=1.005)
    gf.create_conductor(w, name="b", start="g0", end="g1")
    gf.create_source(w, attached_to="g0", magnitude=1.0)
    for kw in ({"kernel": "point"}, {"backend": "mom"}):
        result, hits = _solve(w, segment_length=0.5, **kw)
        assert not hits
        assert np.isfinite(result.input_impedance()[0].real)


@_FINDING
def test_mutual_matrix_point_form_is_guarded_too() -> None:
    # docb item, audit round 2
    from groundfield.solver.mutual import solve_mutual_matrix

    w = _rod_and_strip(0.8)
    engine = gf.create_engine(segment_length=2.0, kernel="point")
    with pytest.raises(BackendCapabilityError, match="across the layer interface"):
        solve_mutual_matrix(w, engine, ["rod"])
    Z = solve_mutual_matrix(w, gf.create_engine(segment_length=2.0), ["rod"])
    assert Z[0, 0].real > 0.0
