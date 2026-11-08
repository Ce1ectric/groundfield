"""Electrodes through a soil-layer interface (review pass 10, 10.15 / F02).

The image family carries one uniform current density per owner. Up to
0.15.0 a rod through :math:`h_1` was one owner, so its current density was
the same above and below the interface although the leakage density scales
with :math:`1/\\rho` of the layer: +566 % (1000/30 Ω·m), +148 % (1000/100),
+123 % (100/1000) against the per-segment ``mom`` solution for the R1 case.
Such a rod now has one owner per layer, all in the electrode's cluster —
the same model as two rods bonded at :math:`h_1` (the manual workaround),
for ``Engine.solve`` and for ``solve_mutual_*`` alike.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf

_FINDING = pytest.mark.finding("10.15")
_SOILS = [(1000.0, 100.0), (100.0, 1000.0), (1000.0, 30.0)]


def _rod_world(soil: gf.TwoLayerSoil, *, split: bool) -> gf.World:
    """Rod (0, 0, 0.5) of 12 m through h_1 = 5 m, or the same rod as two bonded rods."""
    w = gf.create_world(soil=soil)
    if split:
        gf.create_electrode(
            w, "rod", name="r", position=(0, 0, 0.5), length=4.5, wire_radius=0.008
        )
        gf.create_electrode(
            w, "rod", name="b", position=(0, 0, 5.0), length=7.5, wire_radius=0.008
        )
        gf.create_conductor(w, start="r", end="b", name="bond")
    else:
        gf.create_electrode(
            w, "rod", name="r", position=(0, 0, 0.5), length=12.0, wire_radius=0.008
        )
    gf.create_source(w, attached_to="r", magnitude=1.0)
    return w


def _z(world: gf.World, backend: str, **kw) -> float:
    eng = gf.create_engine(backend=backend, segment_length=1.0, **kw)
    return world.solve(eng).cluster_impedance("r")[0].real


@_FINDING
@pytest.mark.parametrize(("rho1", "rho2"), _SOILS)
def test_cross_layer_rod_shares_current_per_layer(rho1: float, rho2: float) -> None:
    """R1-01's test: equal to the split workaround, within 2 % of mom."""
    soil = gf.TwoLayerSoil(rho_1=rho1, rho_2=rho2, h_1=5.0)
    z_rod = _z(_rod_world(soil, split=False), "image")
    assert z_rod == pytest.approx(_z(_rod_world(soil, split=True), "image"), rel=1e-9)
    assert z_rod == pytest.approx(_z(_rod_world(soil, split=False), "mom"), rel=0.02)


@_FINDING
def test_current_density_differs_above_and_below_the_interface() -> None:
    """1000/30 Ω·m: the lower layer takes far more current per metre."""
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    res = _rod_world(soil, split=False).solve(gf.create_engine(segment_length=1.0))
    ps = res.point_sources
    density = np.array([p.current[0].real / p.length for p in ps])
    z = np.array([p.position[2] for p in ps])
    upper, lower = density[z < 5.0], density[z >= 5.0]
    np.testing.assert_allclose(upper, upper[0], rtol=1e-12)  # uniform per layer
    np.testing.assert_allclose(lower, lower[0], rtol=1e-12)
    assert lower[0] / upper[0] > 10.0
    # The electrode reports the whole current and one potential.
    assert res.electrode_currents["r"][0] == pytest.approx(1.0, rel=1e-12)
    assert sum(p.current[0] for p in ps) == pytest.approx(1.0, rel=1e-12)
    assert list(res.electrode_currents) == ["r"]
    assert res.metadata["layer_owners"] == {"r": 2}


@_FINDING
def test_mutual_matrix_diagonal_follows_the_owner_split() -> None:
    """solve_mutual_matrix (AP1 Z_G path) and Engine.solve agree for a crossing rod."""
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=100.0, h_1=5.0)
    engine = gf.create_engine(segment_length=1.0)
    z_solve = _rod_world(soil, split=False).solve(engine).cluster_impedance("r")[0]
    w = _rod_world(soil, split=False)
    w.sources.clear()
    gf.create_electrode(w, "rod", name="far", position=(40.0, 0, 0.5), length=3.0)
    probes = np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    Z = gf.solve_mutual_matrix(w, engine, ["r", "far"], probes, symmetrize=False)
    assert Z[0, 0] == pytest.approx(z_solve, rel=1e-6)
    R = gf.solve_mutual_field(w, engine, ["r", "far"], probes)
    res = _rod_world(soil, split=False).solve(engine)
    np.testing.assert_allclose(R[:, 0], res.potential(probes), rtol=1e-9)


@_FINDING
def test_bonded_cluster_with_one_crossing_rod_matches_the_manual_split() -> None:
    """A station ring bonded to a rod through h_1, with a lumped conductor."""
    soil = gf.TwoLayerSoil(rho_1=500.0, rho_2=50.0, h_1=3.0)

    def world(split: bool) -> gf.World:
        w = gf.create_world(soil=soil)
        gf.create_electrode(w, "ring", name="ring", center=(0, 0, 0.6), radius=2.0)
        if split:
            gf.create_electrode(w, "rod", name="rod", position=(2, 0, 0.6), length=2.4)
            gf.create_electrode(w, "rod", name="tip", position=(2, 0, 3.0), length=1.6)
            gf.create_conductor(w, start="rod", end="tip", name="bond_tip")
        else:
            gf.create_electrode(w, "rod", name="rod", position=(2, 0, 0.6), length=4.0)
        gf.create_conductor(w, start="ring", end="rod", name="bond")
        gf.create_electrode(w, "rod", name="house", position=(25, 0, 0.5), length=5.0)
        gf.create_conductor(
            w,
            start="ring",
            end="house",
            name="pen",
            conductor_type="pen",
            resistivity=2.82e-8,
            cross_section=95e-6,
            coupling_to_soil="isolated",
        )
        gf.create_source(w, attached_to="ring", magnitude=1.0)
        return w

    eng = gf.create_engine(segment_length=0.5, frequencies=[50.0])
    one, two = world(False).solve(eng), world(True).solve(eng)
    assert one.cluster_impedance("ring")[0] == pytest.approx(
        two.cluster_impedance("ring")[0], rel=1e-9
    )
    assert one.electrode_currents["house"][0] == pytest.approx(
        two.electrode_currents["house"][0], rel=1e-9
    )
    i_rod = one.electrode_currents["rod"][0]
    assert i_rod == pytest.approx(
        two.electrode_currents["rod"][0] + two.electrode_currents["tip"][0], rel=1e-9
    )


def test_worlds_inside_one_layer_are_unchanged() -> None:
    """Guard: no split, no layer_owners metadata, the in-layer path."""
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="r", position=(0, 0, 0.5), length=4.0)
    gf.create_source(w, attached_to="r", magnitude=1.0)
    res = w.solve(gf.create_engine(segment_length=1.0))
    assert "layer_owners" not in res.metadata
    assert res.metadata["series_used"] is True


def test_equal_resistivities_reproduce_the_homogeneous_backend() -> None:
    """Guard: TwoLayerSoil(rho, rho, h) is homogeneous soil — one owner per rod.

    The interface sits on a segment boundary (h_1 = 2 m, ds = 0.5 m), so the
    two discretisations coincide and the results agree bit for bit.
    """

    def z(soil, backend: str) -> complex:
        w = gf.create_world(soil=soil)
        gf.create_electrode(w, "rod", name="r", position=(0, 0, 0.5), length=3.0)
        gf.create_source(w, attached_to="r", magnitude=1.0)
        eng = gf.create_engine(backend=backend, segment_length=0.5)
        return w.solve(eng).cluster_impedance("r")[0]

    layered = z(gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=2.0), "image_2layer")
    assert layered == z(gf.HomogeneousSoil(rho=100.0), "image")
