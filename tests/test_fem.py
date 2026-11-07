"""Tests for the experimental FEM (axisymmetric volume PDE) backend.

The FEM backend reduces every cluster to its equivalent hemisphere
and solves the volume PDE in (s, z) coordinates. The reduction is
exact for hemispheres and, by construction, reproduces the Dwight
resistance of a single electrode in homogeneous soil; for clusters,
several clusters and electrodes reaching across a layer interface it is
not a valid cross-check and warns (10.23). ``test_fem_*`` accordingly
checks an enveloped agreement with the closed-form references rather
than bit-exact identity.
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

    The FEM mesh follows the soil description (the layer interface is
    cut into the elements), so a 2-layer stack with equal
    resistivities still carries the interface as a mesh feature even
    though the physics is identical to a homogeneous half-space. Since
    0.15.0 the mesh is graded relative to the equivalent hemisphere
    instead of the domain height, so that feature costs only the
    logarithmic resolution loss of the larger truncation radius:
    measured 0.06 % at ``h_1 = 2 m`` (19.5 % before). The full sweep
    over ``h_1`` lives in ``tests/test_pass9_fem.py``.
    """
    rho = 100.0
    eng = gf.create_engine(backend="fem", segment_length=SEG)
    soil_h = gf.HomogeneousSoil(resistivity=rho)
    soil_2 = gf.TwoLayerSoil(rho_1=rho, rho_2=rho, h_1=2.0)
    Z_h = eng.solve(_world_rod(soil_h)).cluster_impedance("g1")[0].real
    Z_2 = eng.solve(_world_rod(soil_2)).cluster_impedance("g1")[0].real
    assert abs(Z_2 - Z_h) / Z_h < 0.01


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
    gf.create_electrode(w_single, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_source(w_single, attached_to="g1", magnitude=1.0)

    w_pair = gf.create_world(soil=soil)
    gf.create_electrode(w_pair, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_electrode(w_pair, "rod", name="g2", position=(8, 0, 0.0), length=1.5)
    gf.create_conductor(w_pair, name="bond", start="g1", end="g2")
    gf.create_source(w_pair, attached_to="g1", magnitude=1.0)

    Z_single = _solve(w_single, "fem")
    # Two electrodes in one cluster: the reduction sums the hemisphere
    # radii and says so (10.23).
    with pytest.warns(UserWarning, match="sum of their radii"):
        Z_pair = _solve(w_pair, "fem")
    assert Z_pair < Z_single, (Z_pair, Z_single)
    # Two equal hemispheres in parallel: R_pair ≈ R_single / 2.
    # The FEM equivalent-hemisphere reduction places the doubled
    # hemisphere at the cluster centroid; mesh truncation and the
    # geometric reduction together leave a ~20 % envelope around the
    # ideal parallel-hemisphere value.
    assert Z_pair == pytest.approx(Z_single / 2.0, rel=0.25)


# ---------------------------------------------------------------------
# 10.23 — the equivalent-hemisphere reduction says when it is not a
# valid cross-check; the backend is experimental
# ---------------------------------------------------------------------


def _fem_warnings(world) -> tuple[gf.FieldResult, list[str]]:
    """Solve with fem; return the result and the reduction warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = gf.create_engine(backend="fem", segment_length=0.5).solve(world)
    texts = [
        str(w.message) for w in rec if w.category.__name__ == "FemReductionWarning"
    ]
    return res, texts


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("soil", "builder", "needle"),
    [
        # rod 0-3 m through h_1 = 2 m: a_eq = 0.44 m stays in the top
        # layer while the rod reaches the bottom one (+1092 % vs mom)
        (
            gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0),
            lambda w: gf.create_electrode(
                w, "rod", name="g1", position=(0, 0, 0.0), length=3.0
            ),
            "interface",
        ),
        # ring r = 10 m at 0.8 m over h_1 = 2 m: a_eq = 4.6 m reaches the
        # bottom layer the ring never touches (-92 % vs mom)
        (
            gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0),
            lambda w: gf.create_electrode(
                w, "ring", name="g1", center=(0, 0, 0.8), radius=10.0
            ),
            "interface",
        ),
    ],
)
def test_fem_warns_when_electrode_and_hemisphere_straddle_an_interface(
    soil, builder, needle
) -> None:
    w = gf.create_world(soil=soil)
    builder(w)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    _, texts = _fem_warnings(w)
    assert texts, "no FemReductionWarning"
    assert needle in texts[0]
    assert "h = 2" in texts[0]


@pytest.mark.finding("10.23")
def test_fem_warns_for_a_multi_electrode_cluster() -> None:
    """Ring + three rods: radii summed, mutual resistance neglected
    (-35 % against mom in homogeneous soil)."""
    import numpy as np

    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "ring", name="g1", center=(0, 0, 0.5), radius=2.0)
    for k, ang in enumerate((0.0, 2.1, 4.2)):
        gf.create_electrode(
            w,
            "rod",
            name=f"r{k}",
            position=(2 * np.cos(ang), 2 * np.sin(ang), 0.5),
            length=4.0,
        )
        gf.create_conductor(w, start="g1", end=f"r{k}")
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    _, texts = _fem_warnings(w)
    assert texts and "4 electrodes" in texts[0]


@pytest.mark.finding("10.23")
def test_fem_warns_when_clusters_are_solved_without_coupling() -> None:
    """A second (passive) cluster reports 0 V and 0 transfer impedance."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=3.0)
    gf.create_electrode(w, "ring", name="p", center=(4.0, 0, 0.8), radius=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res, texts = _fem_warnings(w)
    assert res.electrode_potentials["p"][0] == 0j
    assert texts and "2 clusters" in texts[0]


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    "soil",
    [
        gf.HomogeneousSoil(resistivity=100.0),
        gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=5.0),
    ],
)
def test_fem_is_silent_for_one_compact_electrode_inside_one_layer(soil) -> None:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5), length=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    _, texts = _fem_warnings(w)
    assert texts == []


@pytest.mark.finding("10.23")
def test_fem_is_marked_experimental() -> None:
    import groundfield.solver.fem as fem_mod

    assert "experimental" in fem_mod.__doc__.splitlines()[0].lower()
    assert "experimental" in fem_mod.solve_fem.__doc__.strip().splitlines()[0].lower()
    # Claims that the reduction is not entitled to (review pass 10, R2-01).
    assert "forms the third independent cross-check" not in fem_mod.solve_fem.__doc__
    assert "the approximation is appropriate" not in fem_mod.__doc__


# ---------------------------------------------------------------------
# 10.49 — the truncation sphere does not cost near-electrode resolution
# ---------------------------------------------------------------------


@pytest.mark.finding("10.49")
@pytest.mark.parametrize("h_1", [10.0, 100.0, 1000.0, 10000.0])
def test_fem_deep_interface_does_not_degrade_the_resolution(h_1: float) -> None:
    """``TwoLayerSoil(rho, rho, h_1)`` is homogeneous for every h_1.

    The truncation sphere grows with the declared thickness,
    ``r_far = 30 (a_eq + h_1)``; at a fixed radial node count v0.15.0
    lost near-electrode resolution logarithmically: -0.25 % / -0.60 % /
    -1.06 % / -1.62 % at h_1 = 10 m / 100 m / 1 km / 10 km (R2-15).
    """
    eng = gf.create_engine(backend="fem", segment_length=SEG)

    def world(soil):
        w = gf.create_world(soil=soil)
        gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    r_hom = eng.solve(world(gf.HomogeneousSoil(resistivity=100.0)))
    r_lay = eng.solve(world(gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=h_1)))
    z_hom = r_hom.cluster_impedance("g1")[0].real
    z_lay = r_lay.cluster_impedance("g1")[0].real
    assert abs(z_lay / z_hom - 1.0) < 2e-3, (h_1, z_lay, z_hom)
