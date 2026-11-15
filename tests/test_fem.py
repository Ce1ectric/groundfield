"""Tests of the conforming axisymmetric ``fem`` backend (user decision Q37).

Since 2026.11 ``fem`` solves rods on one vertical axis and horizontal rings
centred on it as they are (solid cylinder with flat end faces, solid torus,
concrete shells), in any horizontally layered soil, and refuses every
other world. The equivalent-hemisphere reduction and its rules are gone
(they were tested here up to 2026.11; review pass 10, 10.23, user decision
Q23, audits D2A and R2A).

Every number is checked against a second method that shares no code with
``fem``: the exact disc (Weber), the thin-wire ring in closed form, a
Hankel-transform thin ring for any number of layers, the exact 2-D
proximity correction of a wire next to an interface, an independent
finite-volume model of the solid rod (``tests/benchmarks/references.py``)
and the integral engines (``image``, ``image_2layer``, ``mom``,
``mom_sommerfeld``) within their documented biases. The tolerances stay
well above 1e-5, the size by which wave 3 may move the image family.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.geometry.electrodes import concrete_shell_coefficient
from tests.benchmarks import references as ref

pytestmark = pytest.mark.filterwarnings(
    "ignore::groundfield.ExperimentalBackendWarning"
)

SEG = 0.25


def _engine(**kw) -> gf.Engine:
    kw.setdefault("segment_length", SEG)
    return gf.create_engine(backend="fem", **kw)


def _rod_world(soil, *, top=0.0, length=3.0, radius=0.008, shell=0.0, name="g1"):
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w,
        "rod",
        name=name,
        position=(0, 0, top),
        length=length,
        wire_radius=radius,
        concrete_shell_coefficient_ohm_m=shell,
    )
    gf.create_source(w, attached_to=name, magnitude=1.0)
    return w


def _ring_world(soil, *, r0=2.0, depth=0.8, radius=0.008, shell=0.0, name="g1"):
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w,
        "ring",
        name=name,
        center=(0, 0, depth),
        radius=r0,
        wire_radius=radius,
        concrete_shell_coefficient_ohm_m=shell,
    )
    gf.create_source(w, attached_to=name, magnitude=1.0)
    return w


def _z(world, backend="fem", name="g1", **kw) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if backend == "fem":
            eng = _engine(**kw)
        else:
            kw.setdefault("segment_length", 0.05)
            eng = gf.create_engine(backend=backend, **kw)
        return eng.solve(world).cluster_impedance(name)[0].real


def _two(r1, r2, h1):
    return gf.TwoLayerSoil(rho_1=r1, rho_2=r2, h_1=h1)


def _three(rhos, thicknesses):
    return gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=rhos[0], thickness=thicknesses[0]),
            gf.SoilLayer(resistivity=rhos[1], thickness=thicknesses[1]),
            gf.SoilLayer(resistivity=rhos[2]),
        ]
    )


H100 = gf.HomogeneousSoil(resistivity=100.0)


# ---------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------


def test_graded_lines_hit_every_feature_and_grow_at_most_geometrically() -> None:
    # user decision Q37
    from groundfield.solver.fem._mesh import bisect, graded_lines

    feats = [(0.0, math.inf), (0.008, 1.6e-5), (3.0, 1.6e-5), (3000.0, math.inf)]
    lines = graded_lines(feats, 0.2)
    for x, _ in feats:
        assert x in lines
    steps = np.diff(lines)
    assert (steps > 0).all()
    assert steps.min() <= 1.6e-5 * 1.1  # the integer interval counts
    # the spacing grows by at most the ratio 1 + g between neighbours (to
    # the rounding of the integer interval counts)
    assert (steps[1:] / steps[:-1]).max() < 1.25
    assert (steps[:-1] / steps[1:]).max() < 1.25
    fine = bisect(lines, 2)
    assert np.isin(lines, fine).all() and fine.size == 4 * lines.size - 3


def _mesh(rods, rings, interfaces, level=0):
    from groundfield.solver.fem._mesh import MeshParameters, build_mesh

    return build_mesh(rods, rings, interfaces, 3000.0, MeshParameters(), level)


def test_the_rod_is_a_closed_conforming_rectangle() -> None:
    # user decision Q37
    from groundfield.solver.fem._mesh import RodGeometry

    mesh = _mesh([RodGeometry(0.008, 0.5, 3.5)], [], [2.0])
    nodes = mesh.nodes[mesh.port == 0]
    on_face = np.isclose(nodes[:, 0], 0.008) | np.isin(nodes[:, 1], [0.5, 3.5])
    assert on_face.all()
    assert np.isclose(nodes[:, 1], 2.0).any()  # the interface meets the rod
    edges = mesh.port_edges[0]
    # every surface node of the rod has two surface edges, except the two
    # on the axis (the polyline runs from axis to axis)
    count = np.bincount(edges.ravel(), minlength=mesh.nodes.shape[0])[mesh.port == 0]
    on_axis = nodes[:, 0] == 0.0
    assert (count[~on_axis] == 2).all() and (count[on_axis] == 1).all()
    # no triangle inside the rod
    centroid = mesh.nodes[mesh.triangles].mean(axis=1)
    inside = (centroid[:, 0] < 0.008) & (centroid[:, 1] > 0.5) & (centroid[:, 1] < 3.5)
    assert not inside.any()


@pytest.mark.parametrize("h", [2.0, 2.003, 2.01, 1.9999, 2.0 + 1e-6])
def test_the_ring_wire_is_an_inscribed_circle_and_interfaces_are_mesh_lines(h) -> None:
    """The wire's cross section is a closed polygon inscribed in its circle;
    an interface through the wire's mesh block is cut into its elements."""
    # user decision Q37
    from groundfield.solver.fem._mesh import RingGeometry

    ring = RingGeometry(2.0, 2.0, 0.005, 0.015 if abs(h - 2.0) < 0.00525 else 0.0075)
    mesh = _mesh([], [ring], [h])
    nodes = mesh.nodes[mesh.port == 0]
    dist = np.hypot(nodes[:, 0] - 2.0, nodes[:, 1] - 2.0)
    assert dist.max() <= 0.005 * (1 + 1e-12)
    assert dist.min() > 0.005 * math.cos(math.pi / 16)  # chords of 32 at least
    count = np.bincount(mesh.port_edges[0].ravel(), minlength=mesh.nodes.shape[0])
    assert (count[mesh.port == 0] == 2).all()  # closed
    # every triangle lies in one layer: none has vertices on both sides
    dz = mesh.nodes[mesh.triangles, 1] - h
    assert not ((dz > 0).any(axis=1) & (dz < 0).any(axis=1)).any()
    assert (mesh.layer == (mesh.nodes[mesh.triangles, 1].mean(axis=1) > h)).all()


# ---------------------------------------------------------------------
# Exact solution of the kernel
# ---------------------------------------------------------------------


def test_a_disc_on_the_surface_reproduces_webers_resistance() -> None:
    """A coin of radius 1 m and thickness 10 um: rho / (4 b) (Weber)."""
    # user decision Q37
    from groundfield.solver.fem._assembly import solve_ports
    from groundfield.solver.fem._mesh import RodGeometry

    r = []
    for level in (1, 2):
        mesh = _mesh([RodGeometry(1.0, 0.0, 1e-5)], [], [], level)
        r.append(1.0 / solve_ports(mesh, np.array([100.0]), [0.0]).conductance[0, 0])
    extrapolated = r[1] + (r[1] - r[0]) / 3.0
    assert r[0] < r[1] < 25.0  # monotone from below
    assert abs(extrapolated / 25.0 - 1.0) < 2e-4, extrapolated


# ---------------------------------------------------------------------
# Homogeneous soil
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_a_rod_matches_the_solid_rod_finite_volume_model() -> None:
    """3 m rod (a = 8 mm), head 0.5 m deep, 100 Ohm m; FV extrapolated."""
    # user decision Q37
    fv = [
        ref.solid_rod_finite_volume(
            100.0, 100.0, 1e9, 0.5, 3.5, 0.008, h_min=h, growth=g
        )
        for h, g in ((1e-4, 1.07), (5e-5, 1.05))
    ]
    limit = 2.0 * fv[1] - fv[0]  # first order
    z = _z(_rod_world(H100, top=0.5))
    assert abs(z / limit - 1.0) < 1e-4, (z, fv)


def test_a_rod_lies_below_the_uniform_current_value_and_above_point_matching() -> None:
    """Equipotential solid rod: below the uniform current of the line kernel
    (an upper bound), above mom's point kernel (biased low, 10.28)."""
    # user decision Q37
    w = _rod_world(H100, top=0.5)
    z = _z(w)
    image = _z(w, "image", segment_length=0.1)
    mom = _z(w, "mom", segment_length=0.25)
    assert mom < z < image
    assert 0.005 < image / z - 1.0 < 0.015  # +1.0 %
    dwight = 100.0 / (2 * math.pi * 3.0) * (math.log(4 * 3.0 / 0.008) - 1.0)
    assert dwight > z  # Dwight ignores the depth of the head


@pytest.mark.parametrize(("r0", "depth", "a"), [(2.0, 0.8, 0.008), (5.0, 0.5, 0.005)])
def test_a_ring_matches_the_thin_ring_and_image(r0, depth, a) -> None:
    # user decision Q37
    w = _ring_world(H100, r0=r0, depth=depth, radius=a)
    z = _z(w)
    closed = ref.ring_thin_wire_resistance(100.0, r0, a, depth)
    assert abs(z / closed - 1.0) < 2e-4, (z, closed)
    assert abs(z / _z(w, "image", segment_length=0.05) - 1.0) < 3e-4


def test_unextrapolated_levels_converge_monotonically_at_second_order() -> None:
    # user decision Q37
    from groundfield.solver.fem._axisymmetric import axisymmetric_earth

    w = _ring_world(H100)
    eng = _engine()
    r = [
        1.0
        / axisymmetric_earth(w, eng, refinement=k, extrapolate=False).conductance[0, 0]
        for k in range(3)
    ]
    assert r[0] < r[1] < r[2]
    assert 3.5 < (r[1] - r[0]) / (r[2] - r[1]) < 4.6


def test_the_error_estimate_bounds_the_error_of_the_finer_level() -> None:
    # user decision Q37
    w = _ring_world(H100)
    res = _engine().solve(w)
    closed = ref.ring_thin_wire_resistance(100.0, 2.0, 0.008, 0.8)
    fine = res.metadata["fem_port_resistance_levels"][-1][0][0]
    est = res.metadata["fem_error_estimate"]
    assert abs(fine / closed - 1.0) <= 1.5 * est
    assert abs(res.cluster_impedance("g1")[0].real / closed - 1.0) < est / 5


def test_equal_layers_solve_like_homogeneous_soil() -> None:
    """The canonical layer stack: 100 over 100 Ohm m is homogeneous soil."""
    # user decision Q37 (10.48)
    z_h = _z(_rod_world(H100, top=0.5))
    z_2 = _z(_rod_world(_two(100.0, 100.0, 2.0), top=0.5))
    assert z_2 == z_h


def test_the_truncation_does_not_move_the_result() -> None:
    # user decision Q37
    w = _rod_world(_two(30.0, 1000.0, 2.0), top=0.5)
    z = [_z(w, options={"fem": {"r_far_factor": f}}) for f in (200.0, 1000.0, 5000.0)]
    assert abs(z[0] / z[2] - 1.0) < 1e-4 and abs(z[1] / z[2] - 1.0) < 2e-5
    grounded = _z(w, options={"fem": {"far_field": "dirichlet", "r_far_factor": 10.0}})
    assert grounded < z[1]


# ---------------------------------------------------------------------
# Layered soil
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rhos", "h1", "depth"),
    [
        ((1000.0, 30.0), 2.0, 0.8),
        ((30.0, 1000.0), 2.0, 0.8),
        ((100.0, 400.0), 0.5, 2.0),
    ],
)
def test_rings_in_two_layer_soil_match_the_hankel_ring_and_image_2layer(
    rhos, h1, depth
) -> None:
    # user decision Q37
    w = _ring_world(_two(*rhos, h1), depth=depth)
    z = _z(w)
    hankel = ref.ring_layered_hankel(rhos, [h1], 2.0, depth, 0.008)
    assert abs(z / hankel - 1.0) < 2e-4, (z, hankel)
    assert abs(z / _z(w, "image_2layer", segment_length=0.05) - 1.0) < 3e-4


@pytest.mark.parametrize(
    ("rhos", "thicknesses", "depth"),
    [
        ((100.0, 1000.0, 30.0), (1.5, 3.0), 0.8),
        ((1000.0, 30.0, 300.0), (1.0, 2.0), 0.5),
        ((100.0, 1000.0, 30.0), (1.5, 3.0), 2.5),  # in the middle layer
    ],
)
def test_rings_in_three_layer_soil_match_the_hankel_ring(
    rhos, thicknesses, depth
) -> None:
    # user decision Q37
    w = _ring_world(_three(rhos, thicknesses), depth=depth)
    z = _z(w)
    hankel = ref.ring_layered_hankel(rhos, thicknesses, 2.0, depth, 0.008)
    assert abs(z / hankel - 1.0) < 2e-4, (z, hankel)
    if depth < thicknesses[0]:  # mom_sommerfeld: top layer, point kernel low
        msf = _z(w, "mom_sommerfeld", segment_length=0.25)
        assert 0.0 < z / msf - 1.0 < 0.03


@pytest.mark.slow
@pytest.mark.parametrize("rhos", [(1000.0, 30.0), (30.0, 1000.0)])
def test_a_rod_through_the_interface_matches_the_solid_rod_finite_volume_model(
    rhos,
) -> None:
    # user decision Q37
    fv = [
        ref.solid_rod_finite_volume(*rhos, 2.0, 0.5, 3.5, 0.008, h_min=h, growth=g)
        for h, g in ((1e-4, 1.07), (5e-5, 1.05))
    ]
    limit = 2.0 * fv[1] - fv[0]
    w = _rod_world(_two(*rhos, 2.0), top=0.5)
    z = _z(w)
    assert abs(z / limit - 1.0) < 2e-4, (z, fv)
    # the integral engines: one uniform current per layer above, mom below
    assert (
        _z(w, "mom", segment_length=0.25)
        < z
        < _z(w, "image_2layer", segment_length=0.1)
    )


# ---------------------------------------------------------------------
# The limit cases of the equivalent-hemisphere reduction (Q37)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rhos", "r0", "depth"),
    [
        ((1000.0, 30.0), 2.0, 2.0),
        ((30.0, 1000.0), 2.0, 2.0),
        ((1000.0, 30.0), 5.0, 10.0),
    ],
)
def test_a_deep_ring_1_cm_above_an_interface(rhos, r0, depth) -> None:
    """Up to 2026.11: +191 ... +308 % (1000/30) and -27 ... -14 % (30/1000)
    against image_2layer, without a message. Reference: the Hankel thin
    ring plus the exact 2-D proximity correction of the wire, which the
    thin-wire engines lack (-5 % for 1000/30 at two wire radii)."""
    # user decision Q37
    h1 = depth + 0.01
    w = _ring_world(_two(*rhos, h1), r0=r0, depth=depth, radius=0.005)
    z = _z(w)
    k = (rhos[1] - rhos[0]) / (rhos[1] + rhos[0])
    reference = ref.ring_layered_hankel(rhos, [h1], r0, depth, 0.005)
    reference += ref.wire_proximity_correction(rhos[0], r0, 0.005, 0.01, k)
    assert abs(z / reference - 1.0) < 2e-4, (z, reference)


@pytest.mark.parametrize("rhos", [(1000.0, 30.0), (30.0, 1000.0)])
def test_a_ring_exactly_on_the_interface_matches_the_hankel_ring(rhos) -> None:
    """Up to 2026.11: -21 ... +30 % with a warning. image_2layer jumps by
    0.2 % between the two sides of the interface; fem and the Hankel ring
    are continuous."""
    # user decision Q37
    depths = (2.0 - 1e-6, 2.0, 2.0 + 1e-6)
    z = [_z(_ring_world(_two(*rhos, 2.0), depth=d, radius=0.005)) for d in depths]
    hankel = ref.ring_layered_hankel(rhos, [2.0], 2.0, 2.0, 0.005)
    assert abs(z[1] / hankel - 1.0) < 2e-4, (z, hankel)
    assert max(z) / min(z) - 1.0 < 1e-4


@pytest.mark.parametrize("rho_2", [1000.0, 10000.0])
def test_a_rod_reaching_into_a_conductive_top_layer_falls_continuously(rho_2) -> None:
    """3 m rod, top layer 30 Ohm m of h_1 = 0.0885 m (audit D2A-01). Up to
    2026.11: -44 ... -78 % with a warning or an error; the thin wire of the
    integral engines jumps (image_2layer: 2981 / 632 Ohm with the head 1 um
    below / 2 cm into the top layer in 30/10000 Ohm m). The solid rod's head
    face touches the top layer: the resistance falls monotonically as the
    head nears and enters it, and steeply only through the slab of the
    bottom layer left above the face (rho_2 delta / (pi a^2): 50 Ohm for
    1 um of 10000 Ohm m)."""
    # user decision Q37
    h1 = 0.0885
    soil = _two(30.0, rho_2, h1)
    reach = [-1e-3, -1e-6, 0.0, 1e-6, 1e-3, 0.02, 0.08]  # head above h_1 by
    z = [_z(_rod_world(soil, top=h1 - d, length=3.0)) for d in reach]
    assert all(a > b for a, b in zip(z[:-1], z[1:], strict=True)), z
    slab = rho_2 * 1e-6 / (math.pi * 0.008**2)
    assert z[1] - z[2] < slab  # the 1 um slab, in series with the face
    assert z[2] / z[3] - 1.0 < 2e-4  # 1 um of lateral surface: nothing
    # 8 cm in: thin wire and solid rod are close again (a few per cent)
    image = _z(_rod_world(soil, top=h1 - 0.08), "image_2layer", segment_length=0.1)
    assert abs(z[-1] / image - 1.0) < 0.03


@pytest.mark.slow
@pytest.mark.parametrize("p", [0.0, 0.016])
def test_a_rod_tip_on_or_in_a_conductive_layer_matches_the_finite_volume_model(
    p,
) -> None:
    """Rod 0.5 ... 5 m, 1000 over 30 Ohm m, the tip on h_1 or 16 mm below it
    (the case of the known limitation on rod ends at an interface)."""
    # user decision Q37
    h1 = 5.0 - p
    fv = [
        ref.solid_rod_finite_volume(
            1000.0, 30.0, h1, 0.5, 5.0, 0.008, h_min=h, growth=g
        )
        for h, g in ((1e-4, 1.07), (5e-5, 1.05))
    ]
    limit = 2.0 * fv[1] - fv[0]
    z = _z(_rod_world(_two(1000.0, 30.0, h1), top=0.5, length=4.5))
    assert abs(z / limit - 1.0) < 5e-4, (z, fv)


# ---------------------------------------------------------------------
# Concrete shells (ADR-0012)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("depth", [0.8, 3.0])
def test_an_encased_ring_adds_at_least_its_shell_resistance(depth) -> None:
    """R(C) >= R(0) + C / (2 pi r): the shell evens out the current around
    the wire's cross section; the excess fades with depth."""
    # user decision Q37
    c = concrete_shell_coefficient(100.0, 0.005, 0.05)
    bare = _z(_ring_world(H100, depth=depth, radius=0.055))
    encased = _z(_ring_world(H100, depth=depth, radius=0.055, shell=c))
    excess = (encased - bare) / (c / (2 * math.pi * 2.0)) - 1.0
    assert 0.0 < excess < (0.006 if depth < 1 else 0.003), excess


def test_an_encased_rod_follows_sundes_decomposition() -> None:
    """3 m rod from the surface, 5 cm of 100 Ohm m concrete (r_b = 55 mm):
    the shell adds about C / L, a little more (the current evens out)."""
    # user decision Q37
    c = concrete_shell_coefficient(100.0, 0.005, 0.05)
    bare = _z(_rod_world(H100, radius=0.055))
    encased = _z(_rod_world(H100, radius=0.055, shell=c))
    assert 1.0 < (encased - bare) / (c / 3.0) < 1.03
    sunde = c / 3.0 + 100.0 / (2 * math.pi * 3.0) * (math.log(4 * 3.0 / 0.055) - 1.0)
    assert 0.0 < sunde / encased - 1.0 < 0.02  # uniform current: an upper bound


# ---------------------------------------------------------------------
# Network on the shared pipeline
# ---------------------------------------------------------------------


def _ring_and_rod(soil, *, bonded: bool, source: dict | None = None):
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "ring", name="ring", center=(0, 0, 0.8), radius=2.0)
    gf.create_electrode(w, "rod", name="rod", position=(0, 0, 0.5), length=3.0)
    if bonded:
        gf.create_conductor(w, start="ring", end="rod")
    gf.create_source(w, **(source or {"attached_to": "ring", "magnitude": 1.0}))
    return w


def test_a_bonded_ring_and_rod_is_the_parallel_of_the_port_matrix() -> None:
    # user decision Q37
    res = _engine().solve(_ring_and_rod(_two(100.0, 400.0, 2.0), bonded=True))
    r = np.array(res.metadata["fem_port_resistance"])
    assert res.metadata["fem_ports"] == ["ring", "rod"]
    assert np.allclose(r, r.T, rtol=1e-12)
    z = res.cluster_impedance("ring")[0].real
    assert math.isclose(z, 1.0 / np.linalg.inv(r).sum(), rel_tol=1e-10)
    total = sum(res.electrode_currents[n][0].real for n in ("ring", "rod"))
    assert math.isclose(total, 1.0, rel_tol=1e-12)


def test_a_two_terminal_source_measures_the_loop_through_the_soil() -> None:
    # user decision Q37
    src = {"attached_to": "ring", "return_to": "rod", "magnitude": 1.0}
    res = _engine().solve(
        _ring_and_rod(_two(100.0, 400.0, 2.0), bonded=False, source=src)
    )
    r = np.array(res.metadata["fem_port_resistance"])
    expected = r[0, 0] + r[1, 1] - 2 * r[0, 1]
    assert math.isclose(res.input_impedance()[0].real, expected, rel_tol=1e-10)
    assert math.isclose(res.electrode_currents["rod"][0].real, -1.0, rel_tol=1e-12)


def test_a_floating_electrode_takes_the_transfer_potential() -> None:
    # user decision Q37
    res = _engine().solve(_ring_and_rod(H100, bonded=False))
    r = np.array(res.metadata["fem_port_resistance"])
    assert res.electrode_currents["rod"][0] == 0j
    assert math.isclose(res.electrode_potentials["rod"][0].real, r[1, 0], rel_tol=1e-10)
    assert math.isclose(
        res.electrode_potentials["ring"][0].real, r[0, 0], rel_tol=1e-10
    )


def test_voltage_sources_junctions_and_inductive_branches_solve_per_frequency() -> None:
    """A voltage source at a junction feeds a ring and a rod through 16 mm2
    leads, one inductive: fem and image_2layer share the network solve, so
    they differ by the earth model only (a few per mille)."""

    # user decision Q37 (open point 3 of w2src: fem solves voltage sources)
    def world():
        w = gf.create_world(soil=_two(100.0, 400.0, 2.0))
        gf.create_electrode(w, "ring", name="ring", center=(0, 0, 0.8), radius=2.0)
        gf.create_electrode(w, "rod", name="rod", position=(0, 0, 0.5), length=3.0)
        gf.create_junction(w, name="J", position=(0, 0, -0.5))
        gf.create_conductor(w, start="ring", end="J", cross_section=16e-6)
        gf.create_conductor(
            w, start="rod", end="J", cross_section=16e-6, inductance_model="neumann"
        )
        gf.create_source(
            w,
            kind="voltage",
            attached_to="J",
            magnitude=230.0,
            internal_resistance_ohm=1.0,
        )
        return w

    freqs = [0.0, 50.0, 1000.0]
    res = _engine(frequencies=freqs).solve(world())
    other = gf.create_engine(
        backend="image_2layer", segment_length=0.05, frequencies=freqs
    ).solve(world())
    for k in range(3):
        i_fem = res.sources[0].current[k]
        i_img = other.sources[0].current[k]
        assert abs(i_fem / i_img - 1.0) < 5e-3
        assert abs(res.metadata["junction_potentials"]["J"][k] - (230 - i_fem)) < 1e-9
    assert res.sources[0].current[2] != res.sources[0].current[0]  # inductive


def test_an_isolated_distributed_conductor_is_a_chain_of_branches() -> None:
    # user decision Q37
    def world(distributed):
        w = _ring_and_rod(H100, bonded=False)
        gf.create_conductor(
            w,
            start="ring",
            end="rod",
            cross_section=1e-6,
            discretize_segment_length=0.5 if distributed else None,
        )
        return w

    lumped = _engine().solve(world(False)).cluster_impedance("ring")[0]
    chain = _engine().solve(world(True)).cluster_impedance("ring")[0]
    assert abs(chain / lumped - 1.0) < 1e-9


# ---------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------


def test_point_sources_carry_the_model_current_of_every_electrode() -> None:
    # user decision Q37
    src = {"attached_to": "ring", "return_to": "rod", "magnitude": 2.0}
    res = _engine().solve(_ring_and_rod(H100, bonded=False, source=src))
    for name in ("ring", "rod"):
        total = sum(p.current[0] for p in res.point_sources if p.electrode_name == name)
        assert abs(total - res.electrode_currents[name][0]) < 1e-12
    rod = sorted(
        (p for p in res.point_sources if p.electrode_name == "rod"),
        key=lambda p: p.position[2],
    )
    density = [abs(p.current[0]) / p.length for p in rod]
    assert density[-1] > density[len(density) // 2]  # more at the tip


def test_the_finite_element_field_converges_to_the_point_source_field() -> None:
    """Two methods for the field: the FE nodal solution and the point
    sources of the result (the model's leakage with the soil's Green's
    function). The FE field converges to it, by about four per level; the
    point sources carry no far-field discretisation error."""
    # user decision Q37
    from groundfield.solver.fem._assembly import solve_ports
    from groundfield.solver.fem._mesh import MeshParameters, RodGeometry, build_mesh

    w = _rod_world(H100, top=0.5)
    res = _engine().solve(w)
    errors = []
    for level in (0, 1):
        mesh = build_mesh(
            [RodGeometry(0.008, 0.5, 3.5)],
            [],
            [],
            res.metadata["fem_r_far"],
            MeshParameters(),
            level,
        )
        sol = solve_ports(mesh, np.array([100.0]), [0.0])
        if level == 0:
            surface = (mesh.nodes[:, 1] == 0.0) & (mesh.nodes[:, 0] > 1.0)
            s = mesh.nodes[surface & (mesh.nodes[:, 0] < 100.0), 0]
        index = {
            float(x): k
            for k, x in enumerate(mesh.nodes[:, 0])
            if mesh.nodes[k, 1] == 0.0
        }
        rows = [index[float(x)] for x in s]
        phi_fe = sol.potentials[rows, 0] / sol.conductance[0, 0]  # per ampere
        pts = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
        errors.append(np.max(np.abs(phi_fe / res.potential(pts).real - 1.0)))
    assert errors[1] < 5e-3 and errors[1] < errors[0] / 3.0, errors


# ---------------------------------------------------------------------
# Refusals and guards
# ---------------------------------------------------------------------


def _fem_error(world, **kw) -> str:
    """The refusal of the axisymmetric model (the 3-D model of part B solves
    the non-axisymmetric worlds, ``tests/test_fem_3d.py``)."""
    kw.setdefault("options", {"fem": {"model": "axisymmetric"}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(gf.BackendCapabilityError) as info:
            _engine(**kw).solve(world)
    return str(info.value)


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("strip", dict(start=(0, 0, 0.8), end=(10, 0, 0.8))),
        ("grid_mesh", dict(corner=(0, 0, 0.5), size=(10, 10), n_x=2, n_y=2)),
        ("mesh", dict(corner=(0, 0, 0.5), size=(10, 10))),
        ("polyline", dict(vertices=[(0, 0, 0.8), (10, 0, 0.8), (10, 10, 0.8)])),
        ("star", dict(center=(0, 0, 0.8), n_arms=3, arm_length=5.0)),
    ],
)
def test_electrodes_that_are_not_axisymmetric_are_refused(kind, params) -> None:
    # user decision Q37
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, kind, name="g1", **params)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    text = _fem_error(w)
    assert "not axisymmetric" in text and "'g1'" in text, text
    assert "Use backend='image' or 'mom'" in text, text


def test_electrodes_off_the_common_axis_are_refused() -> None:
    # user decision Q37
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, "rod", name="a", position=(0, 0, 0), length=3.0)
    gf.create_electrode(w, "rod", name="b", position=(20, 0, 0), length=3.0)
    gf.create_conductor(w, start="a", end="b")
    gf.create_source(w, attached_to="a", magnitude=1.0)
    text = _fem_error(w)
    assert "off the axis" in text and "'b'" in text, text


@pytest.mark.parametrize(
    ("build", "needle"),
    [
        (
            lambda w: (
                gf.create_electrode(
                    w, "rod", name="g1", position=(0, 0, 0), length=3.0
                ),
                gf.create_electrode(
                    w, "rod", name="g2", position=(0, 0, 3.0), length=1.0
                ),
            ),
            "touch or",
        ),
        (
            lambda w: (
                gf.create_electrode(
                    w, "rod", name="g1", position=(0, 0, 0), length=3.0
                ),
                gf.create_electrode(
                    w, "ring", name="g2", center=(0, 0, 1.0), radius=0.0101
                ),
            ),
            "touches or nearly",
        ),
        (
            lambda w: (
                gf.create_electrode(
                    w, "ring", name="g1", center=(0, 0, 0.8), radius=2.0
                ),
                gf.create_electrode(
                    w, "ring", name="g2", center=(0, 0, 0.81), radius=2.0
                ),
            ),
            "touch or nearly",
        ),
        (
            lambda w: gf.create_electrode(
                w, "ring", name="g1", center=(0, 0, 0.004), radius=2.0
            ),
            "solid torus",
        ),
    ],
)
def test_touching_electrodes_and_a_wire_through_the_surface_are_refused(
    build, needle
) -> None:
    # user decision Q37
    w = gf.create_world(soil=H100)
    build(w)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    assert needle in _fem_error(w)


def test_a_galvanic_distributed_conductor_is_refused() -> None:
    # user decision Q37
    w = _ring_and_rod(H100, bonded=False)
    gf.create_conductor(
        w,
        start="ring",
        end="rod",
        cross_section=50e-6,
        discretize_segment_length=0.5,
        coupling_to_soil="galvanic",
    )
    assert "coupling_to_soil='galvanic'" in _fem_error(w)


def test_the_mesh_size_is_limited_before_anything_is_assembled() -> None:
    # user decision Q37
    from groundfield.diagnostics import ResourceLimitError

    with pytest.raises(ResourceLimitError, match="max_nodes"):
        _engine(options={"fem": {"max_nodes": 1000}}).solve(_ring_world(H100))


@pytest.mark.parametrize(
    ("kind", "params", "needle"),
    [
        (
            "ring",
            dict(center=(0, 0, -0.5), radius=10.0),
            "at or above the soil surface",
        ),
        (
            "rod",
            dict(position=(0, 0, -0.5), length=1.2),
            "reaches above the soil surface",
        ),
        ("ring", dict(center=(0, 0, 0.0025), radius=10.0), "soil-surface plane"),
        (
            "strip",
            dict(start=(0, 0, 0.0), end=(10, 0, 0.0)),
            "at or above the soil surface",
        ),
    ],
)
def test_an_electrode_out_of_the_soil_is_refused_like_every_backend(
    kind, params, needle
) -> None:
    # audit D2A-06
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, kind, name="g1", **params)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    with pytest.raises(ValueError, match=needle) as info:
        _engine().solve(w)
    assert not isinstance(info.value, gf.BackendCapabilityError)


def test_the_point_sources_count_against_the_engines_resource_limit() -> None:
    """A unit error of segment_length fails before anything is allocated,
    as in every backend (the mesh has its own limit, max_nodes)."""
    # user decision Q37 (10.06)
    from groundfield.diagnostics import ResourceLimitError

    with pytest.raises(ResourceLimitError):
        _engine(segment_length=1e-4).solve(_ring_world(H100, r0=100.0))
    with pytest.raises(ResourceLimitError):
        _engine(max_segments=5).solve(_rod_world(H100))


def test_a_world_without_a_source_is_refused_like_every_backend() -> None:
    # audit round 2, A2S-05
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0), length=3.0)
    with pytest.raises(ValueError, match="source"):
        _engine().solve(w)


@pytest.mark.parametrize(
    ("soil", "named"),
    [
        (_three((100.0, 30.0, 300.0), (2.0, 3.0)), "mom_sommerfeld"),
        (_three((100.0, 30.0, 300.0), (0.5, 3.0)), None),
        (gf.HomogeneousSoil(rho=100.0), "image"),
    ],
)
def test_the_remedy_names_only_backends_that_solve_the_world(soil, named) -> None:
    """The remedy comes from the backends' own checks (audit R2A-03)."""
    # audit R2A-03
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "polyline", name="g1", vertices=[(0, 0, 0.8), (10, 0, 0.8), (10, 10, 0.8)]
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    text = _fem_error(w)
    if named is None:
        assert "No backend solves this world yet" in text, text
    else:
        assert f"Use backend='{named}'" in text, text


def test_the_remedy_names_a_finer_segment_length_where_it_lifts_a_refusal() -> None:
    """A strip 5 cm above h_1: mom's point form refuses 0.5 m segments
    (audit R2A-04)."""
    # audit R2A-04
    w = gf.create_world(soil=_two(1000.0, 30.0, 0.85))
    gf.create_electrode(w, "strip", name="g1", start=(0, 0, 0.8), end=(10, 0, 0.8))
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    text = _fem_error(w, segment_length=0.5)
    found = re.search(
        r"'mom' refuses this world at segment_length = 0.5 m .* below ([0-9.]+) m", text
    )
    assert found, text
    assert "image_2layer" in text


def test_fem_is_marked_experimental() -> None:
    # 10.23
    from groundfield.solver.registry import get_backend

    assert get_backend("fem").experimental


def test_the_mesh_arguments_of_0150_warn_and_have_no_effect() -> None:
    # user decision Q37
    from groundfield.solver.fem import solve_fem

    w = _ring_world(H100)
    base = solve_fem(w, _engine()).cluster_impedance("g1")[0]
    with pytest.warns(DeprecationWarning, match="n_radial=..., n_axial="):
        old = solve_fem(w, _engine(), n_radial=60, n_axial=40)
    assert old.cluster_impedance("g1")[0] == base


def test_the_names_of_the_reduction_stay_importable() -> None:
    # user decision Q37
    from groundfield.solver.fem import FemReductionWarning, equivalent_hemisphere_radius

    assert issubclass(FemReductionWarning, UserWarning)
    assert gf.FemReductionWarning is FemReductionWarning
    rod = gf.RodElectrode(name="r", position=(0, 0, 0), length=3.0)
    with pytest.warns(DeprecationWarning, match="equivalent_hemisphere_radius"):
        a = equivalent_hemisphere_radius(rod, 100.0)
    assert math.isclose(a, 0.4423, rel_tol=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FemReductionWarning)
        _engine().solve(_ring_and_rod(H100, bonded=True))  # never emitted


def test_fem_options_are_validated() -> None:
    # user decision Q37
    with pytest.raises(ValueError, match="n_radial"):
        gf.Engine(backend="fem", options={"fem": {"n_radial": 60}})
    with pytest.raises(ValueError):
        gf.Engine(backend="fem", options={"fem": {"refinement": -1}})
    eng = gf.Engine(
        backend="fem", options={"fem": {"refinement": 1, "extrapolate": False}}
    )
    assert eng.backend_options().refinement == 1


def test_mixed_coordinates_warn_at_the_solve() -> None:
    """User decision D5: fem runs the coordinate check like every backend."""
    # user decision D5
    from groundfield.diagnostics import MixedCoordinatesWarning

    w = _ring_world(H100)
    w.electrodes[0] = w.electrodes[0].model_copy(
        update={"center": (500000.0, 5800000.0, 0.8)}
    )
    gf.create_junction(w, name="J", position=(0.0, 0.0, -0.5))
    gf.create_conductor(w, start="g1", end="J")
    with pytest.warns(MixedCoordinatesWarning):
        _engine().solve(w)


def test_projected_coordinates_solve_like_local_ones() -> None:
    # 2026.11 local frame
    local = _z(_ring_world(H100))
    w = _ring_world(H100)
    w.electrodes[0] = w.electrodes[0].model_copy(
        update={"center": (32604321.0, 5812345.0, 0.8)}
    )
    res = _engine().solve(w)
    assert abs(res.cluster_impedance("g1")[0].real / local - 1.0) < 1e-12
    assert res.point_sources[0].position[0] > 3.26e7


def test_the_result_survives_a_json_round_trip_and_reports_its_mesh() -> None:
    # user decision Q37
    res = _engine(frequencies=[0.0, 50.0]).solve(_ring_and_rod(H100, bonded=True))
    back = gf.FieldResult.model_validate_json(res.model_dump_json())
    assert back.cluster_impedance("ring") == res.cluster_impedance("ring")
    md = back.metadata
    assert md["fem_model"] == "axisymmetric" and md["fem_extrapolated"] is True
    assert (
        len(md["fem_mesh"]) == 2
        and md["fem_mesh"][1]["n_nodes"] > md["fem_mesh"][0]["n_nodes"]
    )
    assert md["kernel"] == "line" and md["n_segments"] == len(res.point_sources)
    assert res.cluster_impedance("ring")[0] == res.cluster_impedance("ring")[1]


# ---------------------------------------------------------------------
# Audit W4A (wave 4, round 1)
# ---------------------------------------------------------------------

# Ring r0 = 2 m, a = 5 mm, wire axis at t = 2 m, a layer interface d from
# the wire axis with the more conductive layer beyond it: 30 Ohm m below a
# ring in 1000 Ohm m ("below") or above it ("above"). Reference: the
# axisymmetric boundary elements of auditor W4A (solid torus, Tagg/Sunde
# image rings, three nested levels, observed-order Richardson; shares no
# code with groundfield), resistance in Ohm.
_NEAR_CONTACT_BEM = [
    ("below", 0.95, 7.069669),
    ("below", 0.99, 7.422543),
    ("below", 1.001, 7.856169),
    ("below", 1.01, 8.551903),
    ("below", 1.02, 9.087345),
    ("below", 1.04, 9.908426),
    ("below", 1.05, 10.252321),
    ("above", 0.95, 13.373675),
    ("above", 0.99, 13.726851),
    ("above", 1.001, 14.160322),
    ("above", 1.01, 14.855686),
    ("above", 1.02, 15.390882),
    ("above", 1.04, 16.211641),
    ("above", 1.05, 16.555421),
]
_FAST_NEAR_CONTACT = {("below", 0.99), ("below", 1.01), ("above", 1.04)}


def _near_contact_ring(side: str, d_over_a: float) -> gf.World:
    a, t = 0.005, 2.0
    if side == "below":
        soil = _two(1000.0, 30.0, t + d_over_a * a)
    else:
        soil = _two(30.0, 1000.0, t - d_over_a * a)
    return _ring_world(soil, r0=2.0, depth=t, radius=a)


@pytest.mark.parametrize(
    ("side", "d_over_a", "bem"),
    [
        c
        if (c[0], c[1]) in _FAST_NEAR_CONTACT
        else pytest.param(*c, marks=pytest.mark.slow)
        for c in _NEAR_CONTACT_BEM
    ],
)
def test_a_ring_next_to_a_more_conductive_layer_is_within_1e_3_and_its_estimate(
    side, d_over_a, bem
) -> None:
    """Up to audit W4A-01 an interface within 1.05 a of the wire axis was cut
    into the O-grid with the block's default resolution: the gap between the
    wire and the interface was unresolved, the levels converged at first order
    and not monotonically, and the default read up to 6.0e-3 off with an
    estimate below the error in 7 of these 14 cases."""
    # audit W4A-01
    res = _engine().solve(_near_contact_ring(side, d_over_a))
    error = abs(res.cluster_impedance("g1")[0].real / bem - 1.0)
    assert error < 1e-3, error
    assert res.metadata["fem_error_estimate"] >= error


@pytest.mark.parametrize(
    ("d_over_a", "bem", "fem23"),
    [
        (0.99, 7.422543, 7.414592),
        (1.01, 8.551903, 8.550181),
        (1.04, 9.908426, 9.908191),
    ],
)
def test_w4a_01_the_error_estimate_covers_a_ring_next_to_a_conductive_layer(
    d_over_a, bem, fem23
) -> None:
    """Auditor W4A's test: the default and the finer level lie within the
    error estimate (plus the spread of the two references, the BEM and fem at
    levels 2/3 of the base) of the BEM."""
    # audit W4A-01
    assert abs(fem23 / bem - 1.0) < 1.2e-3  # the two references
    res = _engine().solve(_near_contact_ring("below", d_over_a))
    z = res.cluster_impedance("g1")[0].real
    est = res.metadata["fem_error_estimate"]
    finer = res.metadata["fem_port_resistance_levels"][-1][0][0]
    tol = abs(fem23 / bem - 1.0)
    assert abs(z / bem - 1.0) <= est + tol
    assert abs(finer / bem - 1.0) <= est + tol


def test_a_ring_cut_deeper_by_a_conductive_layer_keeps_second_order() -> None:
    """The interface at 0.9 a through the wire (1000/30 Ohm m): at the
    block's default resolution the default read -1.2e-3 against fem's own
    levels 1/2 at the refined block; levels now converge at second order."""
    # audit W4A-01
    from groundfield.solver.fem._axisymmetric import axisymmetric_earth

    w = _near_contact_ring("below", 0.9)
    eng = _engine()
    r = [
        1.0
        / axisymmetric_earth(w, eng, refinement=k, extrapolate=False).conductance[0, 0]
        for k in range(3)
    ]
    assert r[0] < r[1] < r[2]
    assert 3.5 < (r[1] - r[0]) / (r[2] - r[1]) < 4.6


def test_a_single_level_reports_no_error_estimate() -> None:
    """Up to audit W4A-02 ``extrapolate=False`` reported an estimate of 0.0
    although the level-0 ring is 6.2e-3 low; a single level has no
    estimate."""
    # audit W4A-02
    res = _engine(options={"fem": {"extrapolate": False}}).solve(_ring_world(H100))
    assert math.isnan(res.metadata["fem_error_estimate"])
    assert res.metadata["fem_extrapolated"] is False


def _ring_over_a_rod_head() -> gf.World:
    """Auditor W4A's world: a ring diagonal to a rod's head edge (Euclidean
    gap 1.06 a, so the overlap check passes; Chebyshev gap 0.75 a, no room
    for the ring's O-grid block)."""
    a = 0.008
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "rod", name="rod", position=(0, 0, 0.5), length=3.0, wire_radius=a
    )
    gf.create_electrode(
        w,
        "ring",
        name="ring",
        center=(0, 0, 0.5 - 0.75 * a),
        radius=1.75 * a,
        wire_radius=a,
    )
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    return w


def test_a_ring_without_room_for_its_block_is_not_axisymmetric() -> None:
    """Up to audit W4A-03 the block check ran after the classification, so
    ``auto`` and ``axisymmetric`` ended in a bare ``ValueError`` ("no room
    for the mesh block"). The check is part of the classification now:
    ``auto`` takes the 3-D model, ``axisymmetric`` refuses with a remedy."""
    # audit W4A-03
    from groundfield.solver.fem import _is_axisymmetric

    w = _ring_over_a_rod_head()
    assert _is_axisymmetric(w, _engine()) is False
    text = _fem_error(w)
    assert "no room" in text and "'ring'" in text, text
    assert "'image'" in text and "model='auto'" in text, text


@pytest.mark.slow
def test_w4a_03_auto_mode_does_not_end_in_a_bare_mesh_error() -> None:
    """Auditor W4A's test: ``auto`` either solves the world (3-D model) or
    refuses it with a ``BackendCapabilityError`` that names ``image``."""
    # audit W4A-03
    pytest.importorskip("skfem")
    pytest.importorskip("pyamg")
    w = _ring_over_a_rod_head()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _engine().solve(w)
        except gf.BackendCapabilityError as exc:
            assert "'image'" in str(exc)
        except ValueError as exc:  # pragma: no cover - the base
            pytest.fail(f"bare ValueError in model='auto': {exc}")


@pytest.mark.parametrize("offset", [1e-9, 1e-6])
def test_a_rod_a_micrometre_off_the_ring_axis_is_solved_axisymmetric(offset) -> None:
    """Up to audit W4A-08 the common axis had a tolerance of 1e-9 of the world
    size (3 nm here): a rod 1 um off the ring's axis went to the 3-D model
    (40.7 s, 2.35 GB, -1.43e-3). The tolerance scales with the wire radius
    now: the axisymmetric reading is off by O((offset / r)^2), below 1e-6."""
    # audit W4A-08
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, "ring", name="ring", center=(0, 0, 0.8), radius=2.0)
    gf.create_electrode(w, "rod", name="rod", position=(offset, 0, 0.5), length=3.0)
    gf.create_conductor(w, start="ring", end="rod")
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    res = _engine(options={"fem": {"model": "axisymmetric"}}).solve(w)
    assert res.metadata["fem_model"] == "axisymmetric"
    centred = _ring_and_rod(H100, bonded=True)
    z0 = _engine().solve(centred).cluster_impedance("ring")[0].real
    assert abs(res.cluster_impedance("ring")[0].real / z0 - 1.0) < 1e-6


def test_a_rod_well_off_the_ring_axis_is_not_axisymmetric() -> None:
    # audit W4A-08
    from groundfield.solver.fem import _is_axisymmetric

    w = gf.create_world(soil=H100)
    gf.create_electrode(w, "ring", name="ring", center=(0, 0, 0.8), radius=2.0)
    gf.create_electrode(w, "rod", name="rod", position=(1e-3, 0, 0.5), length=3.0)
    gf.create_conductor(w, start="ring", end="rod")
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    assert _is_axisymmetric(w, _engine()) is False


def test_the_node_limit_defaults_to_two_million() -> None:
    """Audit W4A-05: the axisymmetric model needs about 4 GB per million
    nodes; the old limit of 4 million admitted about 16 GB."""
    # audit W4A-05
    import inspect

    from groundfield.solver.fem import solve_fem
    from groundfield.solver.fem._axisymmetric import axisymmetric_earth
    from groundfield.solver.registry import FemOptions

    assert FemOptions().max_nodes == 2_000_000
    assert inspect.signature(solve_fem).parameters["max_nodes"].default == 2_000_000
    sig = inspect.signature(axisymmetric_earth)
    assert sig.parameters["max_nodes"].default == 2_000_000


def _cost_worlds() -> dict[str, gf.World]:
    """Auditor W4A's worlds of the cost study (s07) and a cut ring."""
    two = _two(100.0, 400.0, 2.0)
    out = {"ring": _ring_world(H100)}
    w = gf.create_world(soil=two)
    gf.create_electrode(
        w, "ring", name="g1", center=(0, 0, 0.8), radius=2.0, wire_radius=0.008
    )
    gf.create_electrode(
        w, "rod", name="rod", position=(0, 0, 0.5), length=3.0, wire_radius=0.008
    )
    gf.create_conductor(w, start="g1", end="rod")
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    out["ring_rod_2l"] = w
    w = gf.create_world(soil=two)
    for k in range(8):
        gf.create_electrode(
            w,
            "ring",
            name="g1" if k == 0 else f"r{k}",
            center=(0, 0, 0.6 + 0.2 * k),
            radius=1.0 + 1.5 * k,
            wire_radius=0.005,
        )
    gf.create_electrode(
        w, "rod", name="rod", position=(0, 0, 0.5), length=3.0, wire_radius=0.008
    )
    for k in range(1, 8):
        gf.create_conductor(w, start="g1", end=f"r{k}")
    gf.create_conductor(w, start="g1", end="rod")
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    out["8rings_rod"] = w
    out["cut_ring"] = _near_contact_ring("below", 0.99)
    out["ring_on_interface"] = _ring_world(_two(1000.0, 30.0, 2.0), depth=2.0)
    return out


@pytest.mark.parametrize("name", list(_cost_worlds()))
def test_the_predicted_node_count_bounds_the_mesh(name) -> None:
    """Up to audit W4A-10 the prediction, documented as an upper bound, was
    1.5 ... 2.3 % low for a ring (its O-grid count grew like 2^level instead
    of 4^level), so ``max_nodes`` let slightly larger meshes through. It
    counts the O-grid exactly now and bounds the nodes an interface cut
    adds: a limit one node below the finer mesh refuses the solve."""
    # audit W4A-10
    from groundfield.diagnostics import ResourceLimitError
    from groundfield.solver._layered import as_layer_stack
    from groundfield.solver.fem import _axisymmetric as ax
    from groundfield.solver.fem._mesh import (
        MeshParameters,
        RingGeometry,
        RodGeometry,
        build_mesh,
        mesh_lines,
        predicted_nodes,
    )

    w = _cost_worlds()[name]
    eng = _engine()
    params = MeshParameters()
    _, els = ax.coaxial_electrodes(w, eng)
    stack = as_layer_stack(w.soil)
    depths = [float(d) for d in np.cumsum(stack.h)]
    rhos = np.asarray(stack.rhos, dtype=float)
    els = ax.block_half_widths(els, depths, params)
    rods = [e.geometry for e in els if isinstance(e.geometry, RodGeometry)]
    rings = [e.geometry for e in els if isinstance(e.geometry, RingGeometry)]
    extent = max(
        [1.0]
        + [r.z_bottom for r in rods]
        + [max(g.radius, g.depth) + g.wire_radius for g in rings]
    )
    contrast = float(rhos.max() / rhos.min())
    r_far = 1000.0 * max(extent, (depths[-1] if depths else 0.0) * contrast)
    inside = [d for d in depths if d < r_far]
    counts = []
    for level in (0, 1):
        lines = mesh_lines(rods, rings, inside, r_far, params, level)
        mesh = build_mesh(rods, rings, inside, r_far, params, level, lines=lines)
        counts.append((lines, mesh.nodes.shape[0]))
    with pytest.raises(ResourceLimitError, match="max_nodes"):
        ax.axisymmetric_earth(w, eng, max_nodes=counts[1][1] - 1)
    for level, (lines, n) in enumerate(counts):
        predicted = predicted_nodes(*lines, rings, level, inside)
        assert n <= predicted <= 1.15 * n, (level, n, predicted)


def test_tutorial_t09_describes_both_models() -> None:
    """Audit W4A-04: T09 said that ``fem`` refuses every world but coaxial rods
    and rings and two rods side by side; since part B ``model="auto"`` solves
    them in three dimensions."""
    # audit W4A-04
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "docs/tutorials/T09_choosing_an_engine_and_convergence.ipynb"
    )
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    text = "".join("".join(c["source"]) for c in cells)
    assert "refuses every other world" not in text
    assert "its conforming model refuses them" not in text
    assert "three-dimensional model" in text and "fem3d" in text
    version = "".join(cells[1]["source"])
    assert "print(gf.__version__)" in version


def test_w4r2a_07_a_ring_of_radius_1_2_a_is_solved_as_the_torus_it_is() -> None:
    """A ring of radius 1.2 a (a = 50 mm: a solid torus with a hole of 0.2 a)
    was refused with the thin-wire ``ValueError`` of the result's display
    segments (chords of 0.94 a), a guard of the integral engines' self-term
    that the axisymmetric model does not use. Auditor W4A's solid-body BEM
    (four levels, last step 1.5e-8; evidence ``fw4r2a/e02``) gives
    89.509319 Ohm."""
    # audit W4R2A-07
    a = 0.05
    w = _ring_world(H100, r0=1.2 * a, depth=0.8, radius=a)
    res = _engine().solve(w)
    assert res.metadata["fem_model"] == "axisymmetric"
    error = abs(res.cluster_impedance("g1")[0].real / 89.509319 - 1.0)
    assert error < 2e-4 and error <= res.metadata["fem_error_estimate"]
    # the integral engines keep their guard
    with pytest.raises(ValueError, match="thin-wire self-term"):
        gf.create_engine(segment_length=0.25).solve(w)


def test_w4r2a_07_tutorial_t09_says_that_fem_has_no_source_kernel() -> None:
    """T09 cell 11 lists ``fem`` among the backends that accept
    ``kernel='line'``, which read as if ``fem`` implemented the line kernel;
    it has no source kernel (the capability table of cell 3 shows none), the
    engine's kernel only selects the field evaluation of its result."""
    # audit W4R2A-07
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "docs/tutorials/T09_choosing_an_engine_and_convergence.ipynb"
    )
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    i = next(k for k, c in enumerate(cells) if 'kernel="line"' in "".join(c["source"]))
    before = "".join(cells[i - 1]["source"])
    assert "that can do it" not in before
    assert "no source kernel of its own" in before
    out = "".join("".join(o.get("text", "")) for o in cells[i].get("outputs", []))
    assert "'fem' (experimental)" in out  # the registry's list, explained above


def test_mesh_lines_that_differ_by_round_off_are_one_line() -> None:
    """Two features computed two ways (a block's centre line at
    0.30000000000000004 m, an interface at 0.3 m) were two lines 5.6e-17 m
    apart: a row of cells of zero height across the mesh (found by this
    stream's reach study, evidence ``fw4r2a/e03_reach_c.out``)."""
    # stream fw4r2a (self-check of W4R2A-02/04)
    from groundfield.solver.fem._mesh import graded_lines

    lines = graded_lines(
        [(0.0, math.inf), (0.3, 1e-3), (0.1 + 0.2, 2e-3), (1.0, math.inf)], 0.2
    )
    assert np.diff(lines).min() > 1e-6
    assert int(np.sum(np.abs(lines - 0.3) < 1e-12)) == 1


def test_a_ring_on_an_interface_and_a_rod_tip_on_it_survive_round_off() -> None:
    """A ring with its axis exactly on an interface at 0.3 m (r0 = 0.5 m,
    a = 10 mm) raised ``RuntimeError: Factor is exactly singular``; a rod
    from 0.1 m of length 0.2 m (tip at 0.30000000000000004 m) with the
    interface at 0.3 m read 6809 Ohm instead of 647 Ohm (levels 646 / 2012),
    silently. Both had mesh lines 5.6e-17 m apart."""
    # stream fw4r2a (self-check of W4R2A-02/04)
    soil = _two(100.0, 100.0 / 1.3, 0.3)
    on = _z(_ring_world(soil, r0=0.5, depth=0.3, radius=0.01))
    hankel = ref.ring_layered_hankel([100.0, 100.0 / 1.3], [0.3], 0.5, 0.3, 0.01)
    assert abs(on / hankel - 1.0) < 2e-3, (on, hankel)
    beside = _z(_ring_world(soil, r0=0.5, depth=0.3 + 1e-9, radius=0.01))
    assert abs(on / beside - 1.0) < 1e-5
    tips = [
        _z(_rod_world(_two(1000.0, 30.0, h), top=0.1, length=0.2), segment_length=0.05)
        for h in (0.3, 0.1 + 0.2, 0.3 + 1e-9)
    ]
    assert max(tips) / min(tips) - 1.0 < 1e-5, tips
