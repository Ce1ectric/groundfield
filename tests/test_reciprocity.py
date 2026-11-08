"""Reciprocity of the image family: Z_ab == Z_ba for every kernel and soil.

The reaction matrix of a reciprocal medium is symmetric, so is every
multi-port derived from it. With the exact line-source kernel as the
default (review pass 10, 10.28), the cross-layer path (ADR-0007) lost
this: its homogeneous part carried the line corrections with the
resistivity of the source layer while the layered correction kept the
compensating terms in point form, so the reaction matrix of a rod through
the interface was 3.8 % asymmetric and a two-port of that rod and a ring
non-reciprocal by 1.1-4.7 % (stream eng2, evidence
``deliver/evidence/eng2/e10_cross_layer_line_kernel_reciprocity.py``; the
point kernel: 1e-8).
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver import mutual as gm


def _rod_and_ring(soil, source: str | None = None, x_ring: float = 4.0) -> gf.World:
    """Rod A 0-3 m (through h_1 = 2 m in the two-layer soil) and ring B."""
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="A", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(world, "ring", name="B", center=(x_ring, 0.0, 0.8), radius=1.5)
    if source is not None:
        gf.create_source(world, attached_to=source, magnitude=1.0)
    return world


_TWO_LAYER = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0)
_SOILS = {
    "homogeneous": gf.HomogeneousSoil(rho=100.0),
    "two_layer": _TWO_LAYER,
    "two_layer_inverse": gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=2.0),
}


@pytest.mark.finding("10.28")
@pytest.mark.parametrize("kernel", ["point", "line"])
@pytest.mark.parametrize("soil", sorted(_SOILS))
@pytest.mark.parametrize("segment_length", [0.4, 0.5])
def test_reaction_matrix_is_symmetric(kernel, soil, segment_length) -> None:
    """The assembled segment reaction matrix is symmetric (cross-layer too)."""
    engine = gf.create_engine(
        backend="image", segment_length=segment_length, kernel=kernel
    )
    problem = gm._prepare(_rod_and_ring(_SOILS[soil]), engine, 0, where="test")
    K = problem.Z_seg_full
    assert np.abs(K - K.T).max() <= 1e-9 * np.abs(K).max()


@pytest.mark.finding("10.28")
@pytest.mark.parametrize("kernel", ["point", "line"])
@pytest.mark.parametrize("soil", sorted(_SOILS))
def test_galerkin_multiport_is_reciprocal(kernel, soil) -> None:
    """``solve_mutual_matrix(offdiag="galerkin")`` is reciprocal unsymmetrised."""
    engine = gf.create_engine(backend="image", segment_length=0.5, kernel=kernel)
    Z = gf.solve_mutual_matrix(
        _rod_and_ring(_SOILS[soil]),
        engine,
        ["A", "B"],
        offdiag="galerkin",
        symmetrize=False,
    )
    assert abs(Z[0, 1] - Z[1, 0]) <= 1e-7 * abs(Z[0, 1])


@pytest.mark.finding("10.28")
@pytest.mark.parametrize("kernel", ["point", "line"])
@pytest.mark.parametrize("soil", sorted(_SOILS))
@pytest.mark.parametrize("x_ring", [4.0, 6.0])
def test_two_port_transfer_impedance_is_reciprocal(kernel, soil, x_ring) -> None:
    """Potential of B per ampere into A equals that of A per ampere into B.

    In the two-layer soils the passive rod has one owner per layer, so it
    floats (zero net current, equalising currents between its layer
    parts, finding 10.29), which makes the transfer impedance reciprocal.
    """
    engine = gf.create_engine(backend="image", segment_length=0.5, kernel=kernel)
    res_a = engine.solve(_rod_and_ring(_SOILS[soil], "A", x_ring))
    res_b = engine.solve(_rod_and_ring(_SOILS[soil], "B", x_ring))
    z_ba = res_a.electrode_potentials["B"][0] / res_a.electrode_currents["A"][0]
    z_ab = res_b.electrode_potentials["A"][0] / res_b.electrode_currents["B"][0]
    assert abs(z_ab - z_ba) <= 1e-7 * abs(z_ba)
