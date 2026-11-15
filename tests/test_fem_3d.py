"""Three-dimensional conforming model of ``fem`` (part B of user decision Q37).

The 3-D model meshes the soil with linear tetrahedra (isotropic background,
longest-edge bisection towards the wires), takes every wire as a smoothed
line source with an analytic radius correction, the layer interfaces (and
resolved concrete, user decision Q40) as partially penalised immersed
elements, and runs the shared network solve. References: the axisymmetric
model of part A (itself within 1e-4 of closed forms and independent
references), exact solutions of the immersed elements, and the integral
engines for worlds part A cannot hold.

Tests are kept small (coarse ``source_radius`` / ``growth``); the
validation numbers of the documentation come from the evidence scripts of
stream w4fem3d.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

pytest.importorskip("skfem")
pytest.importorskip("pyamg")

from groundfield.solver.fem import _assembly3d as asm3  # noqa: E402
from groundfield.solver.fem import _mesh3d as mesh3  # noqa: E402
from groundfield.solver.fem import _wires3d as wires3  # noqa: E402

H100 = gf.HomogeneousSoil(rho=100.0)
#: Coarse but converged-enough settings for the test suite.
COARSE = {"model": "3d", "source_radius": 0.1, "growth": 0.7}


def _engine(**fem) -> gf.Engine:
    return gf.create_engine(
        backend="fem", segment_length=0.25, options={"fem": {**COARSE, **fem}}
    )


def _solve(world, **fem):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _engine(**fem).solve(world)


def _z(world, name="g1", **fem) -> float:
    return _solve(world, **fem).cluster_impedance(name)[0].real


def _axi(world, name="g1") -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = gf.create_engine(
            backend="fem",
            segment_length=0.25,
            options={"fem": {"model": "axisymmetric"}},
        )
        return eng.solve(world).cluster_impedance(name)[0].real


def _rod(soil, *, length=3.0, radius=0.008, shell=0.0, name="g1", xy=(0.0, 0.0)):
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w,
        "rod",
        name=name,
        position=(xy[0], xy[1], 0.0),
        length=length,
        wire_radius=radius,
        concrete_shell_coefficient_ohm_m=shell,
    )
    gf.create_source(w, attached_to=name, magnitude=1.0)
    return w


# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------


def test_the_layer_fraction_is_exact_also_for_equal_depths() -> None:
    """The fraction of a tetrahedron above a plane, against Monte Carlo,
    with the equal vertex depths of tensor-grid elements (a B-spline closed
    form cancels catastrophically there)."""
    # user decision Q37 (part B)
    rng = np.random.default_rng(0)
    worst = 0.0
    for case in range(60):
        p = rng.random((3, 4))
        if case % 3 == 0:
            p[2, 1] = p[2, 0]
        if case % 3 == 1:
            p[2, 1], p[2, 3] = p[2, 0], p[2, 2]
        h = rng.uniform(p[2].min(), p[2].max())
        u = rng.dirichlet(np.ones(4), 100_000)
        mc = float(np.mean(u @ p[2] < h))
        worst = max(worst, abs(mc - float(mesh3.below_fraction(p[2][:, None], h)[0])))
    assert worst < 6e-3  # five standard deviations of the Monte-Carlo estimate


def test_bisection_keeps_the_mesh_conforming_and_the_volume() -> None:
    # user decision Q37 (part B)
    from skfem import MeshTet

    m = MeshTet.init_tensor(
        np.linspace(0, 1, 4), np.linspace(0, 1, 4), np.linspace(0, 1, 4)
    )
    b = mesh3.Bisector(m.p, m.t)
    rng = np.random.default_rng(1)
    v0 = np.abs(mesh3.tet_volumes(b.p, b.t)).sum()
    for _ in range(4):
        b.refine(rng.choice(b.t.shape[1], b.t.shape[1] // 3, replace=False))
    v = mesh3.tet_volumes(b.p, b.t)
    assert abs(np.abs(v).sum() / v0 - 1.0) < 1e-13
    assert np.all(np.abs(v) > 0.0)
    faces = np.sort(
        np.concatenate(
            [b.t[[k for k in range(4) if k != d]] for d in range(4)], axis=1
        ),
        axis=0,
    )
    _, counts = np.unique(faces, axis=1, return_counts=True)
    assert counts.max() == 2  # no hanging node: a face has at most two tetrahedra


@pytest.mark.parametrize(
    ("h", "s1", "s2"), [(0.37, 1.0, 10.0), (0.37, 10.0, 1.0), (0.5003, 1.0, 33.0)]
)
def test_the_immersed_elements_reproduce_a_layered_field_exactly(h, s1, s2) -> None:
    """Uniform current through a layered cube on an irregular bisection
    mesh: the partially penalised immersed elements contain the exact
    piecewise linear potential and reproduce it to round-off."""
    # user decision Q37 (part B)
    from scipy.sparse.linalg import spsolve
    from skfem import MeshTet

    m = MeshTet.init_tensor(
        np.linspace(0, 1, 5), np.linspace(0, 1, 5), np.linspace(0, 1, 5)
    )
    b = mesh3.Bisector(m.p, m.t)
    rng = np.random.default_rng(3)
    for _ in range(3):
        b.refine(rng.choice(b.t.shape[1], b.t.shape[1] // 3, replace=False))
    p, t = b.p, b.t
    k = asm3.stiffness_matrix(p, t, np.array([h]), np.array([s1, s2])).tocsr()
    top, bottom = np.nonzero(p[2] == 0.0)[0], np.nonzero(p[2] == 1.0)[0]
    u = np.zeros(p.shape[1])
    u[bottom] = 1.0
    free = np.setdiff1d(np.arange(p.shape[1]), np.concatenate([top, bottom]))
    u[free] = spsolve(k[free][:, free].tocsc(), -k[free][:, bottom] @ u[bottom])
    flux = 1.0 / (h / s1 + (1.0 - h) / s2)
    exact = np.where(p[2] < h, p[2] / s1, h / s1 + (p[2] - h) / s2) * flux
    assert np.max(np.abs(u - exact)) < 1e-12
    assert abs((u @ (k @ u)) / flux - 1.0) < 1e-12


def test_the_radius_correction_is_sundes_far_inside_a_straight_wire() -> None:
    """Far from the ends, the correction between the smoothed source and the
    wire of radius a is the 2-D radial resistance rho/(2 pi) ln(r_eta / a)."""
    # user decision Q37 (part B)
    model = wires3.WireModel(
        names=("w",),
        pieces=[
            wires3.WirePiece(
                0, np.array([0.0, 0.0, 1.0]), np.array([20.0, 0.0, 1.0]), 0.005, 0.0
            )
        ],
    )
    wires3.axial_quadrature(model, 0.1)
    r_eta = wires3.BUMP_ENERGY_RADIUS * 0.06
    c = wires3.radius_correction(
        model, r_eta, np.full(model.points.shape[0], 100.0), 1.2
    )
    middle = np.abs(model.points[:, 0] - 10.0) < 1.0
    expected = 100.0 / (2.0 * math.pi) * math.log(r_eta / 0.005)
    assert np.allclose(c[middle], expected, rtol=2e-4)
    ends = (model.points[:, 0] < 0.01) | (model.points[:, 0] > 19.99)
    assert np.all(c[ends] < expected)  # half a line beyond a free end


def test_the_bump_energy_radius() -> None:
    """ln r_eta = double integral of eta eta ln|u - v| for the bump."""
    # user decision Q37 (part B)
    pts, w = asm3.bump_disc(1.0, 0.02)
    assert abs(w.sum() - 1.0) < 1e-13
    rng = np.random.default_rng(2)
    i = rng.choice(len(w), 4000, p=w)
    j = rng.choice(len(w), 4000, p=w)
    d = np.linalg.norm(pts[i] - pts[j], axis=1)
    d = d[d > 0]
    assert abs(math.exp(np.mean(np.log(d))) / wires3.BUMP_ENERGY_RADIUS - 1.0) < 0.03


# ---------------------------------------------------------------------
# Against the axisymmetric model (part A)
# ---------------------------------------------------------------------


def test_a_rod_in_homogeneous_soil_matches_the_axisymmetric_model() -> None:
    # user decision Q37 (part B)
    w = _rod(H100)
    z3, za = _z(w), _axi(w)
    assert abs(z3 / za - 1.0) < 1e-3


def test_the_dispatch_keeps_axisymmetric_worlds_on_part_a() -> None:
    # user decision Q37 (part B)
    w = _rod(H100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto = gf.create_engine(backend="fem", segment_length=0.25).solve(w)
    assert auto.metadata["fem_model"] == "axisymmetric"
    assert _solve(w).metadata["fem_model"] == "3d"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("soil", "kind"),
    [
        (gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=2.0), "rod"),
        (gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=4.0), "rod"),
        (gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0), "ring"),
    ],
)
def test_layered_worlds_match_the_axisymmetric_model(soil, kind) -> None:
    """A rod through an interface, above one, and a ring above a conductive
    layer: within the reported error estimate and 3e-3 of part A."""
    # user decision Q37 (part B)
    if kind == "rod":
        w = _rod(soil)
    else:
        w = gf.create_world(soil=soil)
        gf.create_electrode(
            w, "ring", name="g1", center=(0, 0, 0.8), radius=2.0, wire_radius=0.008
        )
        gf.create_source(w, attached_to="g1", magnitude=1.0)
    r = _solve(w, source_radius=None, growth=0.5)
    z3, za = r.cluster_impedance("g1")[0].real, _axi(w)
    assert abs(z3 / za - 1.0) < 3e-3
    assert abs(z3 / za - 1.0) < r.metadata["fem_error_estimate"]


def test_an_encased_rod_adds_its_shell_like_the_axisymmetric_model() -> None:
    """The rise of the input impedance by the shell (same outer radius).

    The totals differ by the end faces of the 55 mm cylinder of the
    axisymmetric model (r_b / L = 1.8 % of its surface; the 3-D model is the
    thin wire of the integral engines), the rise by the shell agrees."""
    # user decision Q37 (part B), ADR-0012
    shell = gf.concrete_shell_coefficient(1000.0, 0.005, 0.05)
    bare, encased = _rod(H100, radius=0.055), _rod(H100, radius=0.055, shell=shell)
    rise_3d = _z(encased, source_radius=0.15) - _z(bare, source_radius=0.15)
    rise_axi = _axi(encased) - _axi(bare)
    assert abs(rise_3d / rise_axi - 1.0) < 1e-2
    assert rise_3d > shell / 3.0  # at least C/L (the current may redistribute)


def test_a_source_radius_below_the_wire_is_refused() -> None:
    # user decision Q37 (part B)
    with pytest.raises(ValueError, match="too small for the bare wires"):
        _solve(_rod(H100, radius=0.055), source_radius=0.05)


# ---------------------------------------------------------------------
# Worlds the axisymmetric model refuses
# ---------------------------------------------------------------------


def _strip_world():
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, "strip", name="g1", start=(0, 0, 0.8), end=(10, 0, 0.8))
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_a_strip_lies_between_mom_and_the_uniform_current_image() -> None:
    """The equipotential strip: below the image family's uniform current,
    within about a per cent of mom (whose point kernel lies low)."""
    # user decision Q37 (part B)
    w = _strip_world()
    z3 = _z(w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zi = (
            gf.create_engine(backend="image", segment_length=0.1)
            .solve(w)
            .cluster_impedance("g1")[0]
            .real
        )
        zm = (
            gf.create_engine(backend="mom", segment_length=0.1)
            .solve(w)
            .cluster_impedance("g1")[0]
            .real
        )
    assert zm < z3 < zi
    assert abs(z3 / zm - 1.0) < 0.02


def test_two_clusters_have_a_symmetric_port_matrix_and_a_transfer_resistance() -> None:
    """Two rods 10 m apart: reciprocity, and the transfer resistance is the
    far field of the other rod, rho/(2 pi d), to a few per cent."""
    # user decision Q37 (part B)
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "rod", name="a", position=(0, 0, 0), length=2.0, wire_radius=0.008
    )
    gf.create_electrode(
        w, "rod", name="b", position=(10, 0, 0), length=2.0, wire_radius=0.008
    )
    gf.create_source(w, attached_to="a", magnitude=1.0, return_to="b")
    r = _solve(w)
    res = np.array(r.metadata["fem_port_resistance"])
    assert abs(res[0, 1] / res[1, 0] - 1.0) < 1e-10
    d = 10.0
    assert abs(res[0, 1] / (100.0 / (2.0 * math.pi * d)) - 1.0) < 0.05


def test_a_galvanic_distributed_conductor_leaks_between_two_rods() -> None:
    """A bare counterpoise between two buried rods: each leaking piece is a
    port, the currents add up to the injected current (Kirchhoff), and the
    input impedance lies within a few per cent of the image family's."""
    # user decision Q37 (part B)
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "rod", name="a", position=(0, 0, 0.6), length=2.0, wire_radius=0.008
    )
    gf.create_electrode(
        w, "rod", name="b", position=(8, 0, 0.6), length=2.0, wire_radius=0.008
    )
    gf.create_conductor(
        w,
        start="a",
        end="b",
        cross_section=50e-6,
        discretize_segment_length=2.0,
        coupling_to_soil="galvanic",
    )
    gf.create_source(w, attached_to="a", magnitude=1.0)
    r = _solve(w)
    leak = sum(r.electrode_currents[n][0] for n in ("a", "b"))
    leak += sum(v[0] for v in r.metadata["conductor_node_currents"].values())
    assert abs(leak - 1.0) < 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zi = (
            gf.create_engine(backend="image", segment_length=0.25)
            .solve(w)
            .cluster_impedance("a")[0]
            .real
        )
    assert abs(r.cluster_impedance("a")[0].real / zi - 1.0) < 0.05


def test_a_voltage_source_a_junction_and_an_inductive_branch_solve_per_frequency() -> (
    None
):
    """A voltage source at a junction feeds a strip and a rod side by side
    through 16 mm2 leads, one inductive: fem (3-D) and the image family share
    the network solve and differ by the earth model only."""

    # user decision Q37 (part B)
    def world():
        w = gf.create_world(soil=H100)
        gf.create_electrode(
            w, "strip", name="strip", start=(0, 0, 0.8), end=(5, 0, 0.8)
        )
        gf.create_electrode(w, "rod", name="rod", position=(10, 0, 0.5), length=2.0)
        gf.create_junction(w, name="J", position=(5, 0, -0.5))
        gf.create_conductor(w, start="strip", end="J", cross_section=16e-6)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gf.create_engine(
            backend="fem",
            segment_length=0.25,
            frequencies=freqs,
            options={"fem": COARSE},
        ).solve(world())
        other = gf.create_engine(
            backend="image", segment_length=0.1, frequencies=freqs
        ).solve(world())
    for k in range(3):
        i_fem = res.sources[0].current[k]
        assert abs(i_fem / other.sources[0].current[k] - 1.0) < 2e-2
        assert abs(res.metadata["junction_potentials"]["J"][k] - (230 - i_fem)) < 1e-9
    assert res.sources[0].current[2] != res.sources[0].current[0]  # inductive


def test_the_field_of_a_3d_result_is_evaluated_from_its_point_sources() -> None:
    # user decision Q37 (part B)
    r = _solve(_strip_world())
    phi = r.potential(np.array([[5.0, 3.0, 0.0], [5.0, 30.0, 0.0]]))
    assert np.all(np.abs(phi[0]) > np.abs(phi[1]))


# ---------------------------------------------------------------------
# Resolved concrete (user decision Q40)
# ---------------------------------------------------------------------


def _encased_loop(rho_c: float = 100.0, side: float = 2.0):
    w = gf.create_world(soil=H100)
    shell = gf.concrete_shell_coefficient(rho_c, 0.005, 0.05)
    gf.create_electrode(
        w,
        "polyline",
        name="g1",
        vertices=[(0, 0, 0.8), (side, 0, 0.8), (side, side, 0.8), (0, side, 0.8)],
        closed=True,
        wire_radius=0.055,
        concrete_shell_coefficient_ohm_m=shell,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _fem3d(world, **kw):
    from groundfield.solver.fem import solve_fem

    eng = gf.create_engine(backend="fem", segment_length=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_fem(world, eng, model="3d", growth=0.7, extrapolate=False, **kw)


@pytest.mark.slow
def test_resolved_concrete_of_the_soils_resistivity_is_the_bare_metal() -> None:
    """Concrete as conductive as the soil: resolving it as a volume gives
    the shell model with the soil's own coefficient on the same mesh (the
    corners differ by the thin-wire terms of the two radii, O(r_b / L))."""
    # user decision Q40
    w = _encased_loop(100.0)
    z_shell = _fem3d(w, _concrete_mesh=100.0).cluster_impedance("g1")[0].real
    z_resolved = _fem3d(w, concrete_resistivity=100.0).cluster_impedance("g1")[0].real
    assert abs(z_resolved / z_shell - 1.0) < 5e-3


def test_resolved_concrete_needs_axis_aligned_wires() -> None:
    # user decision Q40
    shell = gf.concrete_shell_coefficient(1000.0, 0.005, 0.05)
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w,
        "ring",
        name="g1",
        center=(0, 0, 0.8),
        radius=2.0,
        wire_radius=0.055,
        concrete_shell_coefficient_ohm_m=shell,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    with pytest.raises(gf.BackendCapabilityError, match="axis-aligned|along x, y or z"):
        _solve(w, concrete_resistivity=1000.0)


# ---------------------------------------------------------------------
# Guards and options
# ---------------------------------------------------------------------


def test_a_wire_too_close_to_an_interface_is_refused() -> None:
    # user decision Q37 (part B)
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=0.805))
    gf.create_electrode(w, "strip", name="g1", start=(0, 0, 0.8), end=(5, 0, 0.8))
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    with pytest.raises(gf.BackendCapabilityError, match="too close"):
        _solve(w, source_radius=None)


def test_the_3d_mesh_is_limited_before_it_is_built() -> None:
    # user decision Q37 (part B)
    from groundfield.diagnostics import ResourceLimitError

    with pytest.raises(ResourceLimitError, match="max_nodes_3d"):
        _solve(_strip_world(), max_nodes_3d=1000)


def test_the_axisymmetric_model_refuses_a_strip_when_forced() -> None:
    # user decision Q37
    with pytest.raises(gf.BackendCapabilityError, match="not axisymmetric"):
        _solve(_strip_world(), model="axisymmetric")


def test_the_3d_options_are_validated() -> None:
    # user decision Q37 (part B)
    from pydantic import ValidationError

    from groundfield.solver.registry import FemOptions

    for bad in (
        {"model": "2d"},
        {"source_radius": -0.1},
        {"growth": 0.0},
        {"concrete_resistivity": 0.0},
    ):
        with pytest.raises(ValidationError):
            FemOptions(**bad)


def test_without_the_extra_the_3d_model_names_it(monkeypatch) -> None:
    # user decision Q37 (part B), D6
    import builtins

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name in ("skfem", "pyamg"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"groundfield\[fem3d\]"):
        _solve(_strip_world())


def test_a_3d_result_survives_a_json_round_trip_and_reports_its_mesh() -> None:
    # user decision Q37 (part B)
    from groundfield.solver.result import FieldResult

    r = _solve(_strip_world())
    md = r.metadata
    for key in (
        "fem_port_resistance",
        "fem_error_estimate",
        "fem_mesh",
        "fem_source_radius",
        "fem_peak_memory_gb",
    ):
        assert key in md
    assert (
        len(md["fem_mesh"]) == 2
        and md["fem_mesh"][1]["nodes"] > md["fem_mesh"][0]["nodes"]
    )
    back = FieldResult.model_validate_json(r.model_dump_json())
    assert back.cluster_impedance("g1")[0] == r.cluster_impedance("g1")[0]


# ---------------------------------------------------------------------
# Attack: frames, truncation variants, thin layers
# ---------------------------------------------------------------------


def test_projected_coordinates_solve_like_local_ones() -> None:
    # user decision D5, user decision Q37 (part B)
    def strip(x0, y0):
        w = gf.create_world(soil=H100)
        gf.create_electrode(
            w, "strip", name="g1", start=(x0, y0, 0.8), end=(x0 + 6, y0, 0.8)
        )
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    local = _z(strip(0.0, 0.0))
    utm = _z(strip(32_604_321.0, 5_812_345.0))
    assert abs(utm / local - 1.0) < 1e-9


def test_the_unextrapolated_level_lies_below_and_a_grounded_box_lower_still() -> None:
    """The finer level alone is a lower bound of the extrapolated value
    (the conductance falls under refinement); grounding the box biases the
    resistance low by about rho_n / (2 pi R_box)."""
    # user decision Q37 (part B)
    w = _strip_world()
    z_ext = _z(w)
    z_one = _z(w, extrapolate=False)
    z_dir = _z(w, far_field="dirichlet")
    assert z_one < z_ext
    assert z_dir < z_ext
    r_far = _solve(w).metadata["fem_r_far"]
    assert abs((z_ext - z_dir) - 100.0 / (2.0 * math.pi * r_far)) < 0.2 * 100.0 / (
        2.0 * math.pi * r_far
    )


def test_a_rod_through_a_thin_middle_layer() -> None:
    """Three layers with a 0.3 m middle layer that the rod crosses: elements
    cut by both interfaces far away take the laminate; the result moves
    continuously from the two-layer limits (middle layer as the top or the
    bottom resistivity)."""

    # user decision Q37 (part B)
    def rod(rhos):
        soil = gf.MultiLayerSoil(
            layers=[
                gf.SoilLayer(resistivity=rhos[0], thickness=1.0),
                gf.SoilLayer(resistivity=rhos[1], thickness=0.3),
                gf.SoilLayer(resistivity=rhos[2]),
            ]
        )
        return _rod(soil)

    z_mid = _z(rod((100.0, 300.0, 1000.0)))
    z_low = _z(rod((100.0, 100.0, 1000.0)))
    z_high = _z(rod((100.0, 1000.0, 1000.0)))
    assert z_low < z_mid < z_high
