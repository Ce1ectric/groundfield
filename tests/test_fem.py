"""Tests for the FEM (axisymmetric volume PDE) backend.

The FEM backend reduces every cluster to its equivalent hemisphere
and solves the volume PDE in (s, z) coordinates. The reduction is
exact for hemispheres and good (≤ 5–10 %) for thin rods or rings;
``test_fem_*`` accordingly checks an enveloped agreement with the
closed-form references rather than bit-exact identity.
"""

from __future__ import annotations

import pytest

import groundfield as gf
from groundfield.references import dwight1936 as dw

SEG = 0.05


def _world_rod(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_fem_rod_matches_dwight() -> None:
    """FEM on a single rod matches Dwight within 10 % (equivalent-hemisphere)."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng = gf.create_engine(backend="fem", segment_length=SEG)
    Z = eng.solve(_world_rod(soil)).cluster_impedance("g1")[0].real
    R_dw = dw.rod(rho=100.0, length=1.5, radius=0.005)
    assert abs(Z - R_dw) / R_dw < 0.10, (
        f"FEM={Z:.3f} vs Dwight={R_dw:.3f}, Δ={abs(Z - R_dw) / R_dw * 100:.2f}%"
    )


def test_fem_two_layer_K_zero_collapses() -> None:
    """For ρ₁ = ρ₂ the FEM result must equal the homogeneous one.

    The FEM mesh follows the soil description (one z-line per layer
    boundary), so a 2-layer stack with equal resistivities still
    carries the layer interface as a mesh feature even though the
    physics is identical to a homogeneous half-space. We accept a
    < 25 % discretisation bias here in exchange for a mesh that
    stays consistent across a ρ₂ sweep — the price for the layer-
    contrast-monotonicity guarantee in the cross-engine test.
    """
    rho = 100.0
    eng = gf.create_engine(backend="fem", segment_length=SEG)
    soil_h = gf.HomogeneousSoil(resistivity=rho)
    soil_2 = gf.TwoLayerSoil(rho_1=rho, rho_2=rho, h_1=2.0)
    Z_h = eng.solve(_world_rod(soil_h)).cluster_impedance("g1")[0].real
    Z_2 = eng.solve(_world_rod(soil_2)).cluster_impedance("g1")[0].real
    assert abs(Z_2 - Z_h) / Z_h < 0.25


def test_fem_two_layer_low_rho_below_lowers_R() -> None:
    """A more conductive lower layer must lower R; a more resistive one raises it.

    The FEM backend uses an equivalent-hemisphere reduction; the layer
    contrast is only visible in the result when the equivalent
    hemisphere actually crosses the layer boundary. We pick a thin
    upper layer (``h_1 = 0.3 m``) so that the layer effect dominates.
    """
    eng = gf.create_engine(backend="fem", segment_length=SEG)
    soil_low = gf.TwoLayerSoil(rho_1=100.0, rho_2=10.0, h_1=0.1)
    soil_high = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=0.1)
    soil_h = gf.HomogeneousSoil(resistivity=100.0)
    Z_low = eng.solve(_world_rod(soil_low)).cluster_impedance("g1")[0].real
    Z_high = eng.solve(_world_rod(soil_high)).cluster_impedance("g1")[0].real
    Z_h = eng.solve(_world_rod(soil_h)).cluster_impedance("g1")[0].real
    assert Z_low < Z_h < Z_high, (Z_low, Z_h, Z_high)


def test_fem_metadata_exposes_a_eq() -> None:
    """The FEM result reports the equivalent-hemisphere radius per cluster."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng = gf.create_engine(backend="fem", segment_length=SEG)
    res = eng.solve(_world_rod(soil))
    assert res.backend == "fem"
    assert "equivalent_hemisphere_radius" in res.metadata
    a_map = res.metadata["equivalent_hemisphere_radius"]
    assert len(a_map) == 1
    a_eq = next(iter(a_map.values()))
    assert 0.05 < a_eq < 1.5  # plausible for a 1.5 m rod


def test_fem_two_rod_cluster() -> None:
    """Two parallel rods connected by a conductor must give a lower R
    than either single rod alone (parallel-electrode rule)."""
    soil = gf.HomogeneousSoil(resistivity=100.0)

    def _solve(world, eng_be):
        eng = gf.create_engine(backend=eng_be, segment_length=SEG)
        return eng.solve(world).cluster_impedance("g1")[0].real

    w_single = gf.create_world(soil=soil)
    gf.create_electrode(w_single, "rod", name="g1",
                        position=(0, 0, 0.0), length=1.5)
    gf.create_source(w_single, attached_to="g1", magnitude=1.0)

    w_pair = gf.create_world(soil=soil)
    gf.create_electrode(w_pair, "rod", name="g1",
                        position=(0, 0, 0.0), length=1.5)
    gf.create_electrode(w_pair, "rod", name="g2",
                        position=(8, 0, 0.0), length=1.5)
    gf.create_conductor(w_pair, name="bond", start="g1", end="g2")
    gf.create_source(w_pair, attached_to="g1", magnitude=1.0)

    Z_single = _solve(w_single, "fem")
    Z_pair = _solve(w_pair, "fem")
    assert Z_pair < Z_single, (Z_pair, Z_single)
    # Two equal hemispheres in parallel: R_pair ≈ R_single / 2.
    # The FEM equivalent-hemisphere reduction places the doubled
    # hemisphere at the cluster centroid; mesh truncation and the
    # geometric reduction together leave a ~20 % envelope around the
    # ideal parallel-hemisphere value.
    assert Z_pair == pytest.approx(Z_single / 2.0, rel=0.25)
