"""The point-matching MoM family: ``mom``, ``bem`` and ``mom_sommerfeld``.

Passive clusters (10.29)
------------------------
An electrode cluster without a source is a floating conductor: its
potential is unknown, its net leakage current is zero, and its segment
currents redistribute so that its surface is an equipotential. Up to
0.15.0 the three backends dropped such clusters from the linear system
and read their potential back as the unweighted mean of the incident
field. Transfer impedances obtained from two solves were then
non-reciprocal by 1.1 % (top layer) to 146 % (a rod through the layer
interface, R2-02). With every cluster kept in the system the
transfer impedance is symmetric to round-off.

Documentation (10.31)
---------------------
The three backends assemble the same point-matching (collocation)
system with a point-source near field; none of them is a Galerkin
scheme, and they report the same ``metadata["solver"]``.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf

MOM_FAMILY = ("mom", "bem", "mom_sommerfeld")

#: (label, soil, top of rod A, tolerance). Closed-form kernels are
#: symmetric to round-off; the cross-layer pairs come from the
#: layered_green quadrature/interpolation (segment matrix symmetric to
#: ~5e-11, amplified by the solve).
RECIPROCITY_CASES = [
    ("homogeneous", gf.HomogeneousSoil(resistivity=100.0), 0.0, 1e-10),
    ("top_layer", gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0), 0.0, 1e-10),
    (
        "cross_layer_1000_30",
        gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0),
        0.0,
        1e-7,
    ),
    (
        "cross_layer_100_1000",
        gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=2.0),
        0.0,
        1e-7,
    ),
]


def _two_port_world(soil, source: str, *, rod_top: float = 0.0) -> gf.World:
    """Rod A (3 m) and ring B (r = 1.5 m at 0.8 m), 4 m apart (R2-02)."""
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="A", position=(0.0, 0.0, rod_top), length=3.0)
    gf.create_electrode(w, "ring", name="B", center=(4.0, 0.0, 0.8), radius=1.5)
    gf.create_source(w, attached_to=source, magnitude=1.0)
    return w


def _engine(backend: str, ds: float = 0.5) -> gf.Engine:
    return gf.create_engine(backend=backend, segment_length=ds)


# ---------------------------------------------------------------------
# 10.29 — reciprocity of transfer impedances
# ---------------------------------------------------------------------


@pytest.mark.finding("10.29")
@pytest.mark.parametrize("backend", MOM_FAMILY)
@pytest.mark.parametrize(
    ("label", "soil", "rod_top", "tol"),
    RECIPROCITY_CASES,
    ids=[c[0] for c in RECIPROCITY_CASES],
)
def test_two_port_transfer_impedance_is_reciprocal(
    backend: str, label: str, soil, rod_top: float, tol: float
) -> None:
    """Z_BA (excite A, read B) equals Z_AB (excite B, read A).

    v0.15.0: 1.08e-2 (top layer), 1.46 (rod through h_1 in 1000/30),
    1.47e-2 (rod through h_1 in 100/1000), for all three backends.
    """
    eng = _engine(backend)
    r_a = eng.solve(_two_port_world(soil, "A", rod_top=rod_top))
    r_b = eng.solve(_two_port_world(soil, "B", rod_top=rod_top))
    z_ba = r_a.electrode_potentials["B"][0] / r_a.electrode_currents["A"][0]
    z_ab = r_b.electrode_potentials["A"][0] / r_b.electrode_currents["B"][0]
    assert abs(z_ab - z_ba) / abs(z_ba) < tol, (z_ab, z_ba)


@pytest.mark.finding("10.29")
@pytest.mark.parametrize("backend", MOM_FAMILY)
def test_floating_cluster_potential_matches_the_two_port_matrix(backend: str) -> None:
    """The reported potential of the passive ring is the floating value.

    Reference: the 2-port matrix ``(C^T Z^{-1} C)^{-1}`` assembled
    directly from ``mom``'s own segment reaction matrix (R2 script
    ``r2_10b``: Z_BA = 17.28197 Ω here, where v0.15.0 reported the
    unweighted incident-field mean 17.28043 Ω for the passive ring).
    """
    from groundfield.solver.image import _discretize_electrode
    from groundfield.solver.mom import _build_Z_two_layer

    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    world = _two_port_world(soil, "A")
    segs, owner = [], []
    for e in world.electrodes:
        s = _discretize_electrode(e, 0.5, layer_interfaces=(soil.h_1,))
        segs += s
        owner += [e.name] * len(s)
    pts = np.array([s.midpoint for s in segs])
    lengths = np.array([s.length for s in segs])
    radii = np.array([s.wire_radius for s in segs])
    Z = _build_Z_two_layer(pts, lengths, radii, soil, 300, 1e-6)
    n = len(segs)
    C = np.zeros((n, 2))
    C[np.array(owner) == "A", 0] = 1.0
    C[np.array(owner) == "B", 1] = 1.0
    A = np.block([[Z, -C], [C.T, np.zeros((2, 2))]])
    rhs = np.zeros((n + 2, 2))
    rhs[n, 0] = rhs[n + 1, 1] = 1.0
    z_port = np.linalg.solve(A, rhs)[n:, :]

    res = _engine(backend).solve(world)
    assert res.electrode_potentials["A"][0].real == pytest.approx(
        z_port[0, 0], rel=1e-7
    )
    assert res.electrode_potentials["B"][0].real == pytest.approx(
        z_port[1, 0], rel=1e-7
    )
    assert z_port[1, 0] == pytest.approx(17.28197, abs=5e-6)


@pytest.mark.finding("10.29")
@pytest.mark.parametrize("backend", MOM_FAMILY)
def test_passive_electrode_floats_with_zero_net_current(backend: str) -> None:
    """A source-free electrode carries no net current, but a non-zero
    current distribution (the floating equipotential)."""
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    res = _engine(backend).solve(_two_port_world(soil, "A"))

    assert res.electrode_currents["B"] == [0j]
    assert np.isnan(res.cluster_impedance("B")[0])
    seg_b = np.array(
        [ps.current[0] for ps in res.point_sources if ps.electrode_name == "B"]
    )
    assert abs(seg_b.sum()) < 1e-12 * np.abs(seg_b).max()
    assert np.abs(seg_b).max() > 1e-4  # it does redistribute
    # KCL of the whole world: all injected current leaks through A.
    assert res.electrode_currents["A"][0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.finding("10.29")
@pytest.mark.parametrize("backend", MOM_FAMILY)
def test_passive_bonded_cluster_is_one_equipotential(backend: str) -> None:
    """A passive ring bonded to a rod: one potential, opposite currents."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="src", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "ring", name="p_ring", center=(6.0, 0.0, 0.6), radius=1.5)
    gf.create_electrode(w, "rod", name="p_rod", position=(8.5, 0.0, 0.6), length=2.0)
    gf.create_conductor(w, start="p_ring", end="p_rod", coupling_to_soil="isolated")
    gf.create_source(w, attached_to="src", magnitude=1.0)
    res = _engine(backend).solve(w)

    phi_ring = res.electrode_potentials["p_ring"][0]
    phi_rod = res.electrode_potentials["p_rod"][0]
    assert phi_rod == pytest.approx(phi_ring, rel=1e-10)
    i_ring = res.electrode_currents["p_ring"][0]
    i_rod = res.electrode_currents["p_rod"][0]
    assert abs(i_ring) > 1e-4  # the far member feeds the near one
    assert i_ring + i_rod == 0j
    assert np.isnan(res.cluster_impedance("p_ring")[0])


@pytest.mark.finding("10.29")
@pytest.mark.parametrize("backend", MOM_FAMILY)
def test_passive_network_carries_its_equalising_current(backend: str) -> None:
    """Two passive rods joined by a 1 Ω conductor exchange current.

    The rods sit at different incident potentials, so a current flows
    through the conductor; it leaves the soil through one rod and
    re-enters through the other. v0.15.0 dropped the whole network
    (both currents 0, branch current 0).
    """
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="src", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "rod", name="near", position=(3.0, 0.0, 0.0), length=2.0)
    gf.create_electrode(w, "rod", name="far", position=(30.0, 0.0, 0.0), length=2.0)
    gf.create_conductor(
        w,
        start="near",
        end="far",
        coupling_to_soil="isolated",
        resistivity=1.0e-6,
        cross_section=1.0e-6 * 27.0,  # R = rho * L / A = 1 Ω over 27 m
    )
    gf.create_source(w, attached_to="src", magnitude=1.0)
    res = _engine(backend).solve(w)

    i_near = res.electrode_currents["near"][0]
    i_far = res.electrode_currents["far"][0]
    assert abs(i_near) > 1e-3
    assert abs(i_near + i_far) < 1e-12
    # Ohm's law on the branch near -> far: phi_near - phi_far = R * I_b,
    # with I_b = -I_leak(near) by KCL at "near".
    r_branch = 1.0
    du = res.electrode_potentials["near"][0] - res.electrode_potentials["far"][0]
    assert du == pytest.approx(-r_branch * i_near, rel=1e-9)
