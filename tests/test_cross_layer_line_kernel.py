"""Exact line kernel on the cross-layer path (ADR-0007): independent of ds.

The layered correction of the two-layer Green's function contains the
image of a source mirrored in the interface. For wires close to the
interface that image is closer than a segment length, and its point form
(segment midpoint to image midpoint) is the midpoint rule of a peak
narrower than the segment. With the exact line kernel the image term is a
Galerkin integral (review pass 10, 10.28): a ring 5 cm below the interface
gave 30.5 Ω at ds = 0.5 m and 55.7 Ω at ds = 0.025 m before, now 55.6-55.7 Ω
at every ds; the point kernel keeps the old behaviour (29.7 → 54.7 Ω).
"""

from __future__ import annotations

import pytest

import groundfield as gf

_FINDING = pytest.mark.finding("10.28")


def _z(world: gf.World, name: str, ds: float, kernel: str = "line") -> float:
    engine = gf.create_engine(segment_length=ds, kernel=kernel)
    return engine.solve(world).cluster_impedance(name)[0].real


def _ring_below_interface() -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=0.75))
    gf.create_electrode(w, "ring", name="g", center=(0.0, 0.0, 0.8), radius=3.0)
    gf.create_source(w, attached_to="g", magnitude=1.0)
    return w


def _rod_through_interface() -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=2.0))
    gf.create_electrode(
        w, "rod", name="g", position=(0.0, 0.0, 0.5), length=3.0, wire_radius=0.008
    )
    gf.create_source(w, attached_to="g", magnitude=1.0)
    return w


@_FINDING
def test_ring_just_below_the_interface_is_independent_of_ds() -> None:
    values = [_z(_ring_below_interface(), "g", ds) for ds in (0.5, 0.25, 0.1)]
    assert max(values) / min(values) - 1.0 < 2e-3, values
    assert values[0] == pytest.approx(55.7, rel=5e-3)


@_FINDING
def test_rod_through_the_interface_is_independent_of_ds() -> None:
    values = [_z(_rod_through_interface(), "g", ds) for ds in (1.0, 0.5, 0.25)]
    assert max(values) / min(values) - 1.0 < 5e-4, values


def test_point_kernel_keeps_its_numbers_near_the_interface() -> None:
    """``kernel="point"`` reproduces 0.15.0 where its diagonal rule fits.

    At ds = 0.5 m the point form's diagonal rule (+-0.19 m in depth) reaches
    across the interface 5 cm above the ring: 0.15.0 returned 29.68 Ohm
    (line kernel 55.64 Ohm, -47 %); since audit round 2 the solve raises
    (docb item, ``tests/test_cross_layer_point_form.py``). At ds = 0.05 m
    the number is that of 0.15.0.
    """
    from groundfield.solver.registry import BackendCapabilityError

    with pytest.raises(BackendCapabilityError, match="kernel=None or 'line'"):
        _z(_ring_below_interface(), "g", 0.5, "point")
    assert _z(_ring_below_interface(), "g", 0.05, "point") == pytest.approx(
        54.6429, abs=5e-4
    )
