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
        model, r_eta, np.full(model.points.shape[0], 100.0), wires3.NEAR_FACTOR * 0.06
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


def test_the_leakage_per_display_segment_is_its_integral_not_aliased() -> None:
    """Every display segment carries the leakage integrated over its own
    length: uniform around a ring by symmetry. (The nearest segment per
    axial point aliased the axial cells against the 0.1 m chords: 8.9 %
    spread around a 1 m ring.)"""
    # user decision Q37 (part B)
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "ring", name="g1", center=(0, 0, 0.5), radius=1.0, wire_radius=0.008
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = gf.create_engine(
            backend="fem", segment_length=0.1, options={"fem": COARSE}
        )
        r = eng.solve(w)
    i = np.array([ps.current[0].real / ps.length for ps in r.point_sources])
    total = sum(ps.current[0].real for ps in r.point_sources)
    assert abs(total - 1.0) < 1e-12
    assert (i.max() - i.min()) / i.mean() < 5e-3


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


# ---------------------------------------------------------------------
# Audit W4B (stream fw4fem3d)
# ---------------------------------------------------------------------


def _w4b_strip_world(z, extra=None):
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "strip", name="g", start=(-20, 0, z), end=(20, 0, z), wire_radius=0.005
    )
    if extra is not None:
        gf.create_electrode(
            w,
            "strip",
            name="h",
            start=(-20, extra, z),
            end=(20, extra, z),
            wire_radius=0.005,
        )
    return w


#: The window factor of audit W4B-01 in source radii. Before it the window
#: was the caller's argument, 20 R (``_solid.solid_earth`` at 219ca71), so the
#: tests fall back to that and fail on such a base by what they assert, not
#: by an AttributeError (audit W4R2B-07).
_NEAR = getattr(wires3, "NEAR_FACTOR", 20.0)


def _w4b_mid_correction(world, r_src=0.06):
    model = wires3.world_pieces(world)
    wires3.axial_quadrature(model, 0.01)
    r_eta = wires3.BUMP_ENERGY_RADIUS * r_src
    dc = wires3.radius_correction(
        model, r_eta, np.full(model.points.shape[0], 100.0), _NEAR * r_src
    )
    z = model.pieces[0].start[2]
    k = np.argmin(np.linalg.norm(model.points - np.array([0.0, 0.0, z]), axis=1))
    return dc[k], r_eta


def test_w4b_01_no_surface_image_term_in_the_radius_correction() -> None:
    """The finite elements give the exact field of the surface image of a
    smoothed source (it lies 2 z > 2 R away), so the radius correction far
    inside a straight wire 0.1 m deep is Sunde's rho/(2 pi) ln(r_eta/a); the
    base added rho/(4 pi) ln((4 z^2 + r_eta^2)/(4 z^2 + a^2)) (+0.7 %)."""
    # audit W4B-01 (test of auditor W4B)
    dc, r_eta = _w4b_mid_correction(_w4b_strip_world(0.1))
    sunde = 100.0 / (2 * math.pi) * math.log(r_eta / 0.005)
    assert abs(dc / sunde - 1.0) < 1e-3


def test_w4b_01_no_term_of_a_parallel_wire_of_another_port() -> None:
    """A parallel wire of another port 0.15 m away (beyond the disc) must not
    change the radius correction: its field is exact in the mesh, and its
    current is not the observer's."""
    # audit W4B-01 (test of auditor W4B)
    alone, _ = _w4b_mid_correction(_w4b_strip_world(1.5))
    pair, _ = _w4b_mid_correction(_w4b_strip_world(1.5, extra=0.15))
    assert abs(pair / alone - 1.0) < 1e-3


def test_w4b_01_an_arm_folded_back_beside_the_wire_counts_like_another_wire() -> None:
    """A U-shaped polyline whose arms run 0.15 m apart (beyond the disc):
    the far arm is the same electrode's wire but lies beside the point, so
    it must not change the correction (it would add the spurious term of a
    parallel wire, 1.2 %)."""
    # audit W4B-01 (attack of stream fw4fem3d on its own fix)
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w,
        "polyline",
        name="g",
        vertices=[(-20, 0, 1.5), (20, 0, 1.5), (20, 0.15, 1.5), (-20, 0.15, 1.5)],
        closed=False,
        wire_radius=0.005,
    )
    folded, _ = _w4b_mid_correction(w)
    alone, _ = _w4b_mid_correction(_w4b_strip_world(1.5))
    assert abs(folded / alone - 1.0) < 1e-3
    # the far arm tilted by 1e-4 rad (4 mm over 40 m) counts like the
    # parallel one (audit W4R2B-07; W4R2B-02 counted it whole: +1.2 %)
    tilted = gf.create_world(soil=H100)
    gf.create_electrode(
        tilted,
        "polyline",
        name="g",
        vertices=[(-20, 0, 1.5), (20, 0, 1.5), (20, 0.15, 1.5), (-20, 0.154, 1.5)],
        closed=False,
        wire_radius=0.005,
    )
    folded_tilted, _ = _w4b_mid_correction(tilted)
    assert abs(folded_tilted / alone - 1.0) < 1e-3


def _w4b_model_at(pieces, points):
    """A wire model whose axial points are given (each on its piece)."""
    model = wires3.WireModel(
        names=tuple(f"p{k}" for k in range(len(pieces))), pieces=pieces
    )
    pts = np.asarray(points, dtype=float)
    owner = []
    for x in pts:
        dist = []
        for p in pieces:
            t = np.clip((x - p.start) @ p.direction, 0.0, p.length)
            dist.append(np.linalg.norm(x - p.start - t * p.direction))
        owner.append(int(np.argmin(dist)))
    model.points = pts
    model.piece = np.array(owner, dtype=int)
    model.weights = np.ones(len(pts))
    return model


def test_w4b_01_collinear_pieces_and_a_rods_image_count_whole() -> None:
    """The observer's own wire is kept whole: a straight wire cut into
    collinear pieces of separate ports (the pieces of a galvanic conductor)
    has the correction of the uncut wire, and the surface image of a rod
    whose head reaches the surface continues the rod (the closed form over
    the rod and its image)."""
    # audit W4B-01
    r_src = 0.06
    r_eta = wires3.BUMP_ENERGY_RADIUS * r_src
    a = 0.005
    x = [[0.3 * k + 0.05, 0.0, 1.0] for k in range(1, 13)]
    whole = [
        wires3.WirePiece(0, np.array([0.0, 0, 1.0]), np.array([4.0, 0, 1.0]), a, 0.0)
    ]
    cut = [
        wires3.WirePiece(
            k, np.array([0.5 * k, 0, 1.0]), np.array([0.5 * (k + 1), 0, 1.0]), a, 0.0
        )
        for k in range(8)
    ]
    rho = np.full(len(x), 100.0)
    c_whole = wires3.radius_correction(
        _w4b_model_at(whole, x), r_eta, rho, _NEAR * r_src
    )
    c_cut = wires3.radius_correction(_w4b_model_at(cut, x), r_eta, rho, _NEAR * r_src)
    assert np.allclose(c_cut, c_whole, rtol=1e-12, atol=0.0)
    # the same off the coordinate axes, 30 m along the plan direction (0.8,
    # 0.6): the collinear test of the cut pieces holds there as well (audit
    # W4R2B-07; W4R2B-01 clipped such pieces from about 7 m on)
    u = np.array([0.8, 0.6, 0.0])
    o = np.array([0.1234, 0.5678, 1.0])
    x_far = [o + (30.0 + 0.3 * k + 0.05) * u for k in range(1, 13)]
    whole_far = [wires3.WirePiece(0, o + 30.0 * u, o + 34.0 * u, a, 0.0)]
    cut_far = [
        wires3.WirePiece(k, o + (30.0 + 0.5 * k) * u, o + (30.5 + 0.5 * k) * u, a, 0.0)
        for k in range(8)
    ]
    c_whole_far = wires3.radius_correction(
        _w4b_model_at(whole_far, x_far), r_eta, rho, _NEAR * r_src
    )
    c_cut_far = wires3.radius_correction(
        _w4b_model_at(cut_far, x_far), r_eta, rho, _NEAR * r_src
    )
    assert np.allclose(c_cut_far, c_whole_far, rtol=1e-9, atol=0.0)
    assert np.allclose(c_whole_far, c_whole, rtol=1e-9, atol=0.0)

    rod = [
        wires3.WirePiece(
            0, np.array([0.0, 0, 0.0]), np.array([0.0, 0, 3.0]), 0.008, 0.0
        )
    ]
    y = np.array([[0.0, 0.0, 0.3]])
    c_rod = wires3.radius_correction(
        _w4b_model_at(rod, y), r_eta, np.array([100.0]), _NEAR * r_src
    )
    both = 0.0
    for lo, hi in (((0.0, 0, 0.0), (0.0, 0, 3.0)), ((0.0, 0, -3.0), (0.0, 0, 0.0))):
        lo, hi = np.array(lo), np.array(hi)
        both += wires3._line_integral(y, lo, hi, np.array([0.008]))[0]
        both -= wires3._line_integral(y, lo, hi, np.array([r_eta]))[0]
    assert abs(c_rod[0] / (100.0 / (4 * math.pi) * both) - 1.0) < 1e-12


@pytest.mark.slow
def test_w4b_01_two_coaxial_rings_10_cm_apart_within_the_estimate() -> None:
    """Two coaxial rings (1.0 m and 1.1 m, 0.8 m deep) as two ports: R11 of
    the 3-D model lies within its error estimate of part A (base: +4.6e-3
    against an estimate of 3.9e-3)."""
    # audit W4B-01 (test of auditor W4B)
    from groundfield.solver.fem import solve_fem

    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "ring", name="a", center=(0, 0, 0.8), radius=1.0, wire_radius=0.005
    )
    gf.create_electrode(
        w, "ring", name="b", center=(0, 0, 0.8), radius=1.1, wire_radius=0.005
    )
    gf.create_source(w, attached_to="a", magnitude=1.0)
    eng = gf.create_engine(backend="fem", segment_length=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axi = solve_fem(w, eng, model="axisymmetric").metadata["fem_port_resistance"]
        r3 = solve_fem(w, eng, model="3d")
    est = r3.metadata["fem_error_estimate"]
    r = np.asarray(r3.metadata["fem_port_resistance"])
    for i in (0, 1):
        assert abs(r[i, i] / axi[i][i] - 1.0) <= est


@pytest.mark.slow
def test_w4b_01_the_ring_residual_is_flat_in_depth() -> None:
    """A ring r = 1 m (a = 5 mm) at 0.06 m and at 0.8 m: the deviation of the
    3-D model from the solid torus is the same to 3e-4 (base: +2.4e-4 and
    -1.85e-3, the surface-image term drifting with depth)."""
    # audit W4B-01
    from groundfield.solver.fem import solve_fem

    eng = gf.create_engine(backend="fem", segment_length=0.25)
    residual = []
    for z in (0.06, 0.8):
        w = gf.create_world(soil=H100)
        gf.create_electrode(
            w, "ring", name="g", center=(0, 0, z), radius=1.0, wire_radius=0.005
        )
        gf.create_source(w, attached_to="g", magnitude=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            za = solve_fem(w, eng, model="axisymmetric").cluster_impedance("g")[0].real
            z3 = solve_fem(w, eng, model="3d").cluster_impedance("g")[0].real
        residual.append(z3 / za - 1.0)
    assert abs(residual[0] - residual[1]) <= 3e-4


def test_w4b_03_a_failed_cg_solve_names_the_contrast_and_the_alternatives(
    monkeypatch,
) -> None:
    """A conjugate-gradient solve that does not converge raises a
    BackendCapabilityError naming the layer contrast, the tested range, the
    axisymmetric model and the integral engines (base: a bare RuntimeError)."""
    # audit W4B-03
    monkeypatch.setattr(asm3, "CG_MAX_ITERATIONS", 2)
    w = _rod(gf.TwoLayerSoil(rho_1=1.0, rho_2=1000.0, h_1=1.5))
    with pytest.raises(gf.BackendCapabilityError, match="did not converge") as info:
        _solve(w)
    text = str(info.value)
    assert "contrast of 1000" in text and "300" in text
    assert "model='axisymmetric'" in text
    assert "'mom'" in text and "'image_2layer'" in text


@pytest.mark.slow
def test_w4b_03_a_rod_through_a_1_to_1000_interface_solves_or_is_refused() -> None:
    """A rod 0 ... 3 m through a 1 / 1000 Ohm m interface at 1.5 m with the
    3-D model forced: it solves within its estimate of part A, or it is
    refused with BackendCapabilityError (base: RuntimeError after about a
    minute)."""
    # audit W4B-03 (test of auditor W4B)
    from groundfield.solver.fem import solve_fem

    w = _rod(gf.TwoLayerSoil(rho_1=1.0, rho_2=1000.0, h_1=1.5))
    eng = gf.create_engine(backend="fem", segment_length=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        za = solve_fem(w, eng, model="axisymmetric").cluster_impedance("g1")[0].real
        try:
            r3 = solve_fem(w, eng, model="3d")
        except gf.BackendCapabilityError:
            return
    z3 = r3.cluster_impedance("g1")[0].real
    assert abs(z3 / za - 1.0) <= r3.metadata["fem_error_estimate"]


def test_w4b_06_face_terms_do_not_depend_on_the_node_numbering() -> None:
    """The face key of ife_face_terms aliased above 2**21 nodes; the face
    terms of three immersed tetrahedra must not change when their nodes are
    renumbered above 2**21 (base: a spurious face pair, 128 instead of 64
    entries)."""
    # audit W4B-06 (test of auditor W4B)
    big = 1 << 21
    xyz = {
        "P0": (0.0, 0.0, 0.0),
        "P1": (1.0, 0.0, 1.0),
        "P2": (0.0, 1.0, 1.0),
        "P3": (1.0, 1.0, 0.0),
        "P4": (-1.0, -1.0, 0.4),
        "P5": (0.0, -1.0, 1.0),
        "P6": (-1.0, 0.0, 1.0),
        "P7": (-1.0, -1.0, 0.0),
    }

    def terms(ids):
        p = np.zeros((3, max(ids.values()) + 1))
        for name, i in ids.items():
            p[:, i] = xyz[name]
        tets = (
            ("P0", "P1", "P2", "P3"),
            ("P0", "P1", "P2", "P4"),
            ("P0", "P5", "P6", "P7"),
        )
        t = np.array([[ids[n] for n in tet] for tet in tets]).T
        x = np.moveaxis(p[:, t], 0, 2)
        vol, _ = asm3.tet_gradients(p, t)
        ax, val = np.full(3, 2, dtype=np.int64), np.full(3, 0.5)
        sp, sm = np.full(3, 1.0), np.full(3, 0.01)
        _, inv = asm3.ife_matrices(x, ax, val, sp, sm, vol, return_maps=True)
        rows, cols, vals = asm3.ife_face_terms(p, t, inv, ax, val, sp, sm)
        back = {i: n for n, i in ids.items()}
        out = {}
        for r, c, v in zip(rows, cols, vals, strict=True):
            out[(back[r], back[c])] = out.get((back[r], back[c]), 0.0) + v
        return out

    small = terms({n: k for k, n in enumerate(xyz)})
    large = terms(
        {
            "P0": 0,
            "P1": 5,
            "P2": big + 100,
            "P3": big + 200,
            "P4": big + 300,
            "P5": 6,
            "P6": 100,
            "P7": big + 400,
        }
    )
    assert set(small) == set(large)
    assert all(abs(small[k] - large[k]) <= 1e-12 * (1 + abs(small[k])) for k in small)


def _w4b_short_strip():
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, "strip", name="g", start=(0, 0, 0.6), end=(5, 0, 0.6))
    gf.create_source(w, attached_to="g", magnitude=1.0)
    return w


def test_w4b_07_the_3d_solve_is_independent_of_numpys_global_random_state() -> None:
    """pyamg's set-up draws from numpy's global random state: the result must
    be bit-identical whatever the user's seed (base: 1e-14 ... 1e-6 apart)."""
    # audit W4B-07 (script a17 of auditor W4B)
    w = _w4b_short_strip()
    np.random.seed(1)
    z1 = _z(w, "g")
    np.random.seed(2)
    z2 = _z(w, "g")
    np.random.seed(3)
    np.random.rand(1000)
    z3 = _z(w, "g")
    assert z1 == z2 == z3


def test_w4b_07_the_3d_solve_leaves_the_users_random_stream_unchanged() -> None:
    """A 3-D solve must not advance the user's random stream (base: the next
    draws after np.random.seed(7) changed)."""
    # audit W4B-07 (script a17 of auditor W4B)
    w = _w4b_short_strip()
    np.random.seed(7)
    expected = np.random.rand(3)
    np.random.seed(7)
    _z(w, "g")
    assert np.array_equal(np.random.rand(3), expected)


def test_w4b_08_the_segment_budget_is_checked_before_meshing(monkeypatch) -> None:
    """A unit error of segment_length fails before the 3-D mesh is built
    (base: after both levels were solved)."""
    # audit W4B-08
    from groundfield.diagnostics import ResourceLimitError

    def no_mesh(*args, **kwargs):
        raise AssertionError("the 3-D mesh was built before the budget check")

    monkeypatch.setattr(mesh3, "background", no_mesh)
    eng = gf.create_engine(
        backend="fem", segment_length=1e-4, max_segments=1000, options={"fem": COARSE}
    )
    with pytest.raises(ResourceLimitError, match="max_segments"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eng.solve(_w4b_short_strip())


def test_w4b_08_too_many_ports_are_refused_before_meshing(monkeypatch) -> None:
    """Every leaking piece of a galvanic conductor is a port and costs one
    conjugate-gradient solve per level: above the port limit the 3-D model
    refuses before meshing and names the estimated time (base: no limit, a
    20 m conductor in 5 cm pieces ran for about 25 minutes)."""
    # audit W4B-08
    from groundfield.diagnostics import ResourceLimitError
    from groundfield.solver.fem import _solid

    def no_mesh(*args, **kwargs):
        raise AssertionError("the 3-D mesh was built before the port check")

    monkeypatch.setattr(mesh3, "background", no_mesh)
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, "rod", name="a", position=(0, 0, 0.6), length=2.0)
    gf.create_electrode(w, "rod", name="b", position=(8, 0, 0.6), length=2.0)
    gf.create_conductor(
        w,
        start="a",
        end="b",
        cross_section=50e-6,
        coupling_to_soil="galvanic",
        discretize_segment_length=0.05,
    )
    gf.create_source(w, attached_to="a", magnitude=1.0)
    with pytest.raises(ResourceLimitError, match="ports") as info:
        _solve(w)
    text = str(info.value)
    assert f"{_solid.MAX_PORTS_3D}" in text and "minutes" in text


def test_w4b_09_the_memory_figure_is_the_increase_during_the_solve() -> None:
    """fem_peak_memory_gb is what the solve added to the peak memory of the
    process, not the peak of everything the process did before (base: a
    small solve after a large allocation reported the large one); the
    process peak is reported separately."""
    # audit W4B-09
    big = np.ones(80_000_000)  # 0.64 GB, touched
    big[::4096] += 1.0
    del big
    md = _solve(_w4b_short_strip()).metadata
    assert md["fem_process_peak_memory_gb"] >= 0.6
    assert 0.0 <= md["fem_peak_memory_gb"] < md["fem_process_peak_memory_gb"] - 0.5


# ---------------------------------------------------------------------
# Stream fw4int: the port limit as an option, the near-interface warning
# ---------------------------------------------------------------------


class _PastThePortCheck(Exception):
    """Raised by the stubbed mesher: the solve got past the port check."""


def _galvanic_chain(piece: float = 0.05) -> gf.World:
    """Two rods 8 m apart, a galvanic conductor in ``piece`` m pieces: 162
    ports at 5 cm (the world of the W4B-08 test)."""
    w = gf.create_world(soil=H100)
    gf.create_electrode(w, "rod", name="a", position=(0, 0, 0.6), length=2.0)
    gf.create_electrode(w, "rod", name="b", position=(8, 0, 0.6), length=2.0)
    gf.create_conductor(
        w,
        start="a",
        end="b",
        cross_section=50e-6,
        coupling_to_soil="galvanic",
        discretize_segment_length=piece,
    )
    gf.create_source(w, attached_to="a", magnitude=1.0)
    return w


def _stub_the_mesher(monkeypatch) -> None:
    def past(*args, **kwargs):
        raise _PastThePortCheck

    monkeypatch.setattr(mesh3, "background", past)


def test_the_3d_port_limit_is_an_option_of_100_ports() -> None:
    """FemOptions.max_ports_3d carries the port limit of the 3-D model, by
    default the 100 ports of audit W4B-08 (base: a constant of the module,
    not raisable)."""
    # audit W4B-08 (stream fw4int: the limit as an option)
    import inspect

    from groundfield.solver.fem import _solid, solve_fem
    from groundfield.solver.registry import FemOptions

    assert FemOptions().max_ports_3d == 100 == _solid.MAX_PORTS_3D
    assert inspect.signature(solve_fem).parameters["max_ports_3d"].default == 100
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            FemOptions(max_ports_3d=bad)


def test_the_port_refusal_names_the_option(monkeypatch) -> None:
    """Above the limit the refusal names FemOptions.max_ports_3d, the way
    to solve such a world on purpose."""
    # audit W4B-08 (stream fw4int: the limit as an option)
    from groundfield.diagnostics import ResourceLimitError

    _stub_the_mesher(monkeypatch)
    with pytest.raises(ResourceLimitError, match="162 ports") as info:
        _solve(_galvanic_chain())
    assert "FemOptions.max_ports_3d" in str(info.value)
    with pytest.raises(ResourceLimitError, match="above its limit of 150 ports"):
        _solve(_galvanic_chain(), max_ports_3d=150)


def test_a_raised_port_limit_lets_the_3d_model_mesh_the_world(monkeypatch) -> None:
    """With max_ports_3d raised above the port count the solve passes the
    port check and goes on to mesh (stubbed here), through the engine and
    through solve_fem."""
    # audit W4B-08 (stream fw4int: the limit as an option)
    from groundfield.solver.fem import solve_fem

    _stub_the_mesher(monkeypatch)
    with pytest.raises(_PastThePortCheck):
        _solve(_galvanic_chain(), max_ports_3d=162)
    with pytest.raises(_PastThePortCheck), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solve_fem(_galvanic_chain(), _engine(), max_ports_3d=1000, **COARSE)


def test_the_port_limit_survives_a_json_round_trip_of_the_engine() -> None:
    # audit W4B-08 (stream fw4int: the limit as an option)
    import json

    from groundfield.solver.registry import FemOptions

    engine = gf.Engine(backend="fem", options={"fem": {"max_ports_3d": 250}})
    payload = json.loads(engine.model_dump_json())
    assert payload["options"]["fem"]["max_ports_3d"] == 250
    back = gf.Engine.model_validate_json(engine.model_dump_json())
    assert back == engine
    assert isinstance(back.options["fem"], FemOptions)
    assert back.options["fem"].max_ports_3d == 250
    default = gf.Engine.model_validate_json(gf.Engine(backend="fem").model_dump_json())
    assert default.backend_options().max_ports_3d == 100


def _ring_above_the_interface(gap: float = 0.015) -> gf.World:
    """Auditor W4B's ring (r = 1 m, a = 5 mm, 0.8 m deep), its axis ``gap``
    above a 1000 / 30 Ohm m interface."""
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=0.8 + gap))
    gf.create_electrode(
        w, "ring", name="g1", center=(0, 0, 0.8), radius=1.0, wire_radius=0.005
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _interface_warnings(record) -> list:
    from groundfield.solver._guards import InterfaceContactWarning

    return [r for r in record if issubclass(r.category, InterfaceContactWarning)]


def test_the_3d_model_warns_once_for_a_wire_next_to_a_contrasting_interface() -> None:
    """A bare wire 1.5 cm (3 radii) above a 1000 / 30 Ohm m interface: the 3-D
    model's smoothed line source shares the thin-wire error there (+9.2e-3
    against the solid torus, audit W4B-02), so the solve warns once, before
    its own guards (here the node limit) refuse anything."""
    # audits W4A-07, W4B-02 (stream fw4int: the 3-D path)
    from groundfield.diagnostics import ResourceLimitError

    eng = gf.create_engine(
        backend="fem",
        segment_length=0.25,
        options={"fem": {"model": "3d", "max_nodes_3d": 1000}},
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ResourceLimitError, match="max_nodes_3d"):
            eng.solve(_ring_above_the_interface())
    hits = _interface_warnings(record)
    assert len(hits) == 1, [str(h.message) for h in hits]
    text = str(hits[0].message)
    assert "'g1'" in text and "3 wire radii" in text and "reach 3.9 radii" in text
    # 3 cm (6 radii) lies outside the reach of 3.9 radii at contrast 33
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ResourceLimitError):
            eng.solve(_ring_above_the_interface(0.03))
    assert _interface_warnings(record) == []


def test_the_axisymmetric_model_does_not_warn_for_the_same_ring() -> None:
    """The axisymmetric model solves the solid torus and does not warn."""
    # audits W4A-07, W4B-02 (stream fw4int: the 3-D path)
    eng = gf.create_engine(backend="fem", segment_length=0.25)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = eng.solve(_ring_above_the_interface())
    assert result.metadata["fem_model"] == "axisymmetric"
    assert _interface_warnings(record) == []


# ---------------------------------------------------------------------
# Audits W4R2B and W4R2A-01 (stream fw4r2b): oblique pieces, arms folded
# back at an angle, wires beside each other, thin concrete cover, the
# transfer estimate, the CG refusal text
# ---------------------------------------------------------------------

R_SRC = 0.06


def _w4r2b_correction(world, spacing=0.01, r_src=R_SRC, extra=()):
    model = wires3.world_pieces(world, extra)
    wires3.axial_quadrature(model, spacing)
    r_eta = wires3.BUMP_ENERGY_RADIUS * r_src
    near = getattr(wires3, "NEAR_FACTOR", 1.5) * r_src
    dc = wires3.radius_correction(
        model, r_eta, np.full(model.points.shape[0], 100.0), near
    )
    return model, dc, r_eta


def _w4r2b_at(model, dc, point) -> float:
    k = int(np.argmin(np.linalg.norm(model.points - np.asarray(point, float), axis=1)))
    return float(dc[k])


def test_w4r2b_01_the_own_piece_counts_whole_in_any_direction() -> None:
    """A straight 20 m strip in the plan direction (0.8, 0.6): the radius
    correction is the closed form of the whole piece at every axial point.
    Base: p2 = |x - a|^2 - t^2 cancels to about |x - a|^2 eps, exceeds the
    collinear tolerance (1e-6 x 1.5 R)^2 = 8.1e-15 m^2 from about 7 m along
    the piece, and the own piece is clipped to +-1.5 R: 347 of 2668 points
    lose 1.63 %."""
    # audit W4R2B-01 (test of auditor W4R2B)
    w = gf.create_world(soil=H100)
    s0 = np.array([0.1234, 0.5678, 0.8])
    u = np.array([0.8, 0.6, 0.0])
    gf.create_electrode(
        w,
        "strip",
        name="g",
        start=tuple(s0),
        end=tuple(s0 + 20.0 * u),
        wire_radius=0.005,
    )
    model, dc, r_eta = _w4r2b_correction(w, spacing=0.015)
    p = model.pieces[0]
    s = (model.points - p.start) @ p.direction
    a, length = 0.005, p.length
    closed = (
        100.0
        / (4 * math.pi)
        * (
            np.arcsinh((length - s) / a)
            + np.arcsinh(s / a)
            - np.arcsinh((length - s) / r_eta)
            - np.arcsinh(s / r_eta)
        )
    )
    assert np.max(np.abs(dc / closed - 1.0)) < 1e-6


def _w4r2b_hairpin(theta, w_gap=0.15, L=8.0, z=0.8):
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w,
        "polyline",
        name="g",
        closed=False,
        wire_radius=0.005,
        vertices=[
            (-L / 2, 0, z),
            (L / 2, 0, z),
            (L / 2, w_gap, z),
            (-L / 2, w_gap + L * math.tan(theta), z),
        ],
    )
    return w


def test_w4r2b_02_a_slightly_tilted_folded_arm_counts_like_another_wire() -> None:
    """A hairpin whose return arm runs 0.15 m beside the outgoing one:
    tilting the return arm by 1e-4 rad (0.8 mm over 8 m) must not change the
    radius correction in the middle of the outgoing arm. Base: the exactly
    parallel arm is clipped (+3e-7), the tilted one is "own chain, not
    parallel" and counts whole: +1.20e-2 (the full solve +1.41e-3, mom
    -2e-4)."""
    # audit W4R2B-02 (test of auditor W4R2B)
    z = 0.8
    m0, d0, _ = _w4r2b_correction(_w4r2b_hairpin(0.0))
    m1, d1, _ = _w4r2b_correction(_w4r2b_hairpin(1e-4))
    assert abs(_w4r2b_at(m1, d1, (0, 0, z)) / _w4r2b_at(m0, d0, (0, 0, z)) - 1.0) < 1e-3


def test_w4r2b_02_one_v_as_a_polyline_and_as_two_bonded_strips() -> None:
    """A V of two 4 m arms 10 degrees apart: the radius correction 1 m along
    the first arm (the other arm 0.17 m away) does not depend on whether the
    V is one polyline or two strips from the apex. Base: +8.9e-3 for the
    polyline (own chain, counted whole); the full solve +8.5e-4 while mom
    and image give identical results for both."""
    # audit W4R2B-02 (test of auditor W4R2B)
    z, L, t = 0.8, 4.0, math.radians(10.0)
    tip = (L * math.cos(t), L * math.sin(t), z)
    poly = gf.create_world(soil=H100)
    gf.create_electrode(
        poly,
        "polyline",
        name="g",
        closed=False,
        wire_radius=0.005,
        vertices=[(L, 0, z), (0, 0, z), tip],
    )
    two = gf.create_world(soil=H100)
    gf.create_electrode(
        two, "strip", name="g", start=(0, 0, z), end=(L, 0, z), wire_radius=0.005
    )
    gf.create_electrode(
        two, "strip", name="h", start=(0, 0, z), end=tip, wire_radius=0.005
    )
    mp, dp, _ = _w4r2b_correction(poly)
    mt, dt, _ = _w4r2b_correction(two)
    assert (
        abs(_w4r2b_at(mp, dp, (1.0, 0, z)) / _w4r2b_at(mt, dt, (1.0, 0, z)) - 1.0)
        < 1e-3
    )


def _w4r2b_whole(model, x, r_eta, keep) -> float:
    """rho/(4 pi) sum over the pieces ``keep`` (whole) at the point ``x``."""
    y = np.asarray([x], dtype=float)
    total = 0.0
    for k in keep:
        p = model.pieces[k]
        total += wires3._line_integral(y, p.start, p.end, np.array([0.005]))[0]
        total -= wires3._line_integral(y, p.start, p.end, np.array([r_eta]))[0]
    return 100.0 / (4 * math.pi) * total


def test_w4r2b_02_corners_and_rings_keep_their_own_wire_whole() -> None:
    """The own polyline counts whole where it has turned by at most 90
    degrees, by a weight falling linearly to 0 at 150 degrees, and beyond
    only within its window (audit W4R3B-03; up to then whole below 120
    degrees, nothing beyond): an L corner counts both arms whole 0.1 m from
    the corner; on a ring of r = 1 m every chord within 90 degrees of turn
    counts whole, the chords between 90 and 150 degrees by their weight and
    the far third (more than 1.7 m away) nothing; a square loop counts its
    adjacent sides whole and the opposite one nothing."""
    # audits W4R2B-02, W4R3B-03
    z = 0.8
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w,
        "polyline",
        name="g",
        closed=False,
        wire_radius=0.005,
        vertices=[(4, 0, z), (0, 0, z), (0, 4, z)],
    )
    model, dc, r_eta = _w4r2b_correction(w)
    x = model.points[np.argmin(np.linalg.norm(model.points - [0.1, 0, z], axis=1))]
    assert (
        abs(_w4r2b_at(model, dc, x) / _w4r2b_whole(model, x, r_eta, (0, 1)) - 1) < 1e-12
    )
    ring = gf.create_world(soil=H100)
    gf.create_electrode(
        ring, "ring", name="g", center=(0, 0, z), radius=1.0, wire_radius=0.005
    )
    model, dc, r_eta = _w4r2b_correction(ring)
    x = model.points[0]
    d0 = model.pieces[model.piece[0]].direction
    expected = 0.0
    weighted = 0
    for k, p in enumerate(model.pieces):
        turn = math.degrees(math.acos(max(-1.0, min(1.0, float(d0 @ p.direction)))))
        weight = min(1.0, max(0.0, (150.0 - turn) / 60.0))
        weighted += 0.0 < weight < 1.0
        if weight > 0.0:
            expected += weight * _w4r2b_whole(model, x, r_eta, (k,))
    assert weighted > 100
    assert abs(_w4r2b_at(model, dc, x) / expected - 1) < 1e-9
    loop = gf.create_world(soil=H100)
    gf.create_electrode(
        loop,
        "polyline",
        name="g",
        closed=True,
        wire_radius=0.005,
        vertices=[(0, 0, z), (0.5, 0, z), (0.5, 0.5, z), (0, 0.5, z)],
    )
    model, dc, r_eta = _w4r2b_correction(loop)
    x = model.points[np.argmin(np.linalg.norm(model.points - [0.25, 0, z], axis=1))]
    assert (
        abs(_w4r2b_at(model, dc, x) / _w4r2b_whole(model, x, r_eta, (0, 1, 3)) - 1)
        < 1e-12
    )


def _w4r2b_parallel_strips(extra, z=1.5, h_z=None):
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "strip", name="g", start=(-20, 0, z), end=(20, 0, z), wire_radius=0.005
    )
    if extra is not None:
        zz = z if h_z is None else h_z
        gf.create_electrode(
            w,
            "strip",
            name="h",
            start=(-20, extra, zz),
            end=(20, extra, zz),
            wire_radius=0.005,
        )
    return w


def test_w4r2b_03_a_parallel_wire_of_another_port_at_r_over_0_8() -> None:
    """A parallel wire of another port at R/0.8 = 7.5 cm, the distance from
    which the texts claimed 2e-3: the exact 2-D correction of the bump there
    is +0.0014 of a self term of 3.75 (f01 of fw4fem3d), 3.7e-4. Base: the
    1.5 R window adds the reduced-kernel term of the 9 cm of the other wire
    within reach, +2.4e-2, with the observer's current."""
    # audit W4R2B-03 (test of auditor W4R2B)
    m0, d0, _ = _w4r2b_correction(_w4r2b_parallel_strips(None))
    m1, d1, _ = _w4r2b_correction(_w4r2b_parallel_strips(1.25 * R_SRC))
    assert (
        abs(_w4r2b_at(m1, d1, (0, 0, 1.5)) / _w4r2b_at(m0, d0, (0, 0, 1.5)) - 1.0)
        < 2e-3
    )


def test_w4r2b_03_a_wire_of_another_port_that_meets_the_observer_keeps_its_window() -> (
    None
):
    """Wires that meet are one conductor: next to the corner of an L made of
    two strips the other strip's part within 1.5 R counts (as the own
    polyline's corner does), while the same strip 2 cm lower, which does
    not touch, counts nothing."""
    # audit W4R2B-03
    z = 0.8

    def l_world(dz):
        w = gf.create_world(soil=H100)
        gf.create_electrode(
            w, "strip", name="g", start=(4, 0, z), end=(0, 0, z), wire_radius=0.005
        )
        gf.create_electrode(
            w,
            "strip",
            name="h",
            start=(0, 0, z + dz),
            end=(0, 4, z + dz),
            wire_radius=0.005,
        )
        return w

    alone = gf.create_world(soil=H100)
    gf.create_electrode(
        alone, "strip", name="g", start=(4, 0, z), end=(0, 0, z), wire_radius=0.005
    )
    ma, da, _ = _w4r2b_correction(alone)
    mm, dm, _ = _w4r2b_correction(l_world(0.0))
    mf, df, _ = _w4r2b_correction(l_world(0.02))
    x = (0.02, 0.0, z)
    assert _w4r2b_at(mm, dm, x) / _w4r2b_at(ma, da, x) - 1.0 > 0.1
    assert abs(_w4r2b_at(mf, df, x) / _w4r2b_at(ma, da, x) - 1.0) < 1e-12


def _w4r2b_model(world):
    from groundfield.solver.fem._network import port_topology

    disc, _ = port_topology(world)
    return wires3.world_pieces(world, disc.segments[len(world.electrodes) :])


def _w4r2b_station(soil=H100):
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w,
        "polyline",
        name="ring",
        vertices=[(-2, -2, 0.6), (2, -2, 0.6), (2, 2, 0.6), (-2, 2, 0.6)],
        closed=True,
        wire_radius=0.005,
    )
    for k, (x, y) in enumerate([(-2, -2), (2, -2), (2, 2)]):
        gf.create_electrode(
            w,
            "rod",
            name=f"rod{k}",
            position=(x, y, 0.6),
            length=4.0,
            wire_radius=0.008,
        )
        gf.create_conductor(w, start="ring", end=f"rod{k}")
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    return w


def _w4r2b_rings(radius_b, depth_b=0.8, a=0.005):
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "ring", name="g1", center=(0, 0, 0.8), radius=1.0, wire_radius=a
    )
    gf.create_electrode(
        w, "ring", name="g2", center=(0, 0, depth_b), radius=radius_b, wire_radius=a
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _w4a_world():
    """Auditor W4A's rod with a small ring (r0 = 1.75 a) 0.75 a above its head."""
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


def test_w4r2b_03_wires_that_meet_form_one_system() -> None:
    """The AP1 station (a ring with rods from its corners) and a galvanic
    conductor with the rods it joins are one system each; two coaxial rings
    and a strip crossing 2 cm above another are two."""
    # audit W4R2B-03
    assert len(set(wires3.wire_systems(_w4r2b_model(_w4r2b_station())))) == 1
    assert len(set(wires3.wire_systems(_w4r2b_model(_galvanic_chain(0.5))))) == 1
    assert len(set(wires3.wire_systems(_w4r2b_model(_w4r2b_rings(1.075))))) == 2
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "strip", name="g", start=(-2, 0, 0.8), end=(2, 0, 0.8), wire_radius=0.005
    )
    gf.create_electrode(
        w, "strip", name="h", start=(0, -2, 0.78), end=(0, 2, 0.78), wire_radius=0.005
    )
    assert len(set(wires3.wire_systems(_w4r2b_model(w)))) == 2


def test_w4r2b_03_beside_wires_finds_wires_side_by_side_folded_and_around() -> None:
    """The closest wires that run beside each other: coaxial rings (beside),
    a hairpin 6 cm wide (folded), a small ring around a rod's head (around);
    not a V of 10 degrees as one polyline, whose arms meet at the apex
    (audit W4R3B-02: up to then "folded" at the sample point nearest the
    apex, 0.65 mm)."""
    # audits W4R2B-03, W4R2A-01, W4R3B-02
    reach = 1.5 * R_SRC
    b = wires3.beside_wires(_w4r2b_model(_w4r2b_rings(1.075)), reach)
    # (the 1 degree chords lie inside the circle by up to 3.8e-5 r)
    assert b.kind == "beside" and abs(b.distance - 0.075) < 5e-5
    b = wires3.beside_wires(_w4r2b_model(_w4r2b_hairpin(0.0, w_gap=0.06)), reach)
    assert b.kind == "folded" and abs(b.distance - 0.06) < 1e-9
    b = wires3.beside_wires(_w4r2b_model(_w4a_world()), reach)
    assert b.kind == "around" and abs(b.distance - math.hypot(0.014, 0.006)) < 1e-5
    t = math.radians(10.0)
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w,
        "polyline",
        name="g",
        closed=False,
        wire_radius=0.005,
        vertices=[(4, 0, 0.8), (0, 0, 0.8), (4 * math.cos(t), 4 * math.sin(t), 0.8)],
    )
    assert wires3.beside_wires(_w4r2b_model(w), reach) is None


def test_w4r2b_03_junctions_crossings_and_collinear_pieces_are_left_alone() -> None:
    """Not beside each other: the AP1 station (rods on the ring's corners), a
    T-joint, a crossing at 90 degrees in one plane and 2 cm apart, a V of 40
    degrees, a step in a polyline, stacked rods end to end, the 5 cm pieces
    of a galvanic conductor, a ring, an L corner, a hairpin 15 cm wide."""
    # audit W4R2B-03
    z = 0.8
    worlds = [_w4r2b_station(), _galvanic_chain(0.05), _w4r2b_hairpin(0.0)]
    for dz in (0.0, 0.02):
        w = gf.create_world(soil=H100)
        gf.create_electrode(
            w, "strip", name="g", start=(-2, 0, z), end=(2, 0, z), wire_radius=0.005
        )
        gf.create_electrode(
            w,
            "strip",
            name="h",
            start=(0, -2, z - dz),
            end=(0, 2, z - dz),
            wire_radius=0.005,
        )
        worlds.append(w)
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "strip", name="g", start=(-2, 0, z), end=(2, 0, z), wire_radius=0.005
    )
    gf.create_electrode(
        w, "strip", name="h", start=(0, 0, z), end=(0, 2, z), wire_radius=0.005
    )
    worlds.append(w)
    for vertices in (
        [(4, 0, z), (0, 0, z), (4 * math.cos(0.698), 4 * math.sin(0.698), z)],
        [(0, 0, z), (4, 0, z), (4, 0.05, z), (8, 0.05, z)],
        [(4, 0, z), (0, 0, z), (0, 4, z)],
    ):
        w = gf.create_world(soil=H100)
        gf.create_electrode(
            w, "polyline", name="g", closed=False, wire_radius=0.005, vertices=vertices
        )
        worlds.append(w)
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "rod", name="a", position=(0, 0, 0), length=1.0, wire_radius=0.008
    )
    gf.create_electrode(
        w, "rod", name="b", position=(0, 0, 1.0), length=2.0, wire_radius=0.008
    )
    worlds.append(w)
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "ring", name="g", center=(0, 0, z), radius=1.0, wire_radius=0.005
    )
    worlds.append(w)
    for w in worlds:
        assert wires3.beside_wires(_w4r2b_model(w), 1.5 * R_SRC) is None


def test_w4r2b_03_the_source_radius_keeps_wires_beside_each_other_1_5_radii_apart() -> (
    None
):
    """Two coaxial rings 7.5 cm apart: R = 7.5 cm / 1.5 = 5 cm instead of the
    default 6 cm, with the reason; the AP1 station keeps 6 cm. Base: 6 cm
    for both (R11 +4.35e-3 against an estimate of 3.85e-3)."""
    # audit W4R2B-03
    import inspect

    from groundfield.solver.fem import _solid

    eng = gf.create_engine(backend="fem", segment_length=0.25)
    kw = (
        {"notes": []}
        if "notes" in inspect.signature(_solid.source_radius).parameters
        else {}
    )
    w = _w4r2b_rings(1.075)
    r = _solid.source_radius(w, eng, _w4r2b_model(w), [], None, None, **kw)
    assert abs(r - 0.05) < 5e-5
    assert "electrode 'g2' runs beside electrode 'g1' 75 mm apart" in kw["notes"][0]
    w = _w4r2b_station()
    assert _solid.source_radius(w, eng, _w4r2b_model(w), [], None, None) == 0.06


def test_w4r2b_03_wires_too_close_for_the_smallest_source_radius_are_refused(
    monkeypatch,
) -> None:
    """Two rings diagonal to each other 12.7 mm apart (auditor W4R2A's world,
    1.8 wire radii in s and z): R would be 8.5 mm, below the 11 mm that the
    5 mm wires need, so the 3-D model refuses before meshing, naming the
    distance and the integral engines. Base: +12.5 % against the solid tori,
    estimate 4.2e-3, no warning."""
    # audits W4R2B-03, W4R2A-01
    from groundfield.solver.fem import solve_fem

    _stub_the_mesher(monkeypatch)
    w = _w4r2b_rings(1.0 + 1.8 * 0.005, depth_b=0.8 + 1.8 * 0.005)
    with (
        pytest.raises(gf.BackendCapabilityError, match="beside") as info,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        solve_fem(w, gf.create_engine(backend="fem", segment_length=0.25), model="3d")
    text = str(info.value)
    assert "12.7 mm apart" in text and "1.5 source radii" in text
    assert "'image'" in text and "'mom'" in text


def test_w4r2b_03_a_node_refusal_names_the_limited_source_radius() -> None:
    """Coaxial rings 3 cm apart need R = 2 cm, about three times the nodes of
    the default; the node refusal names why the source radius is small and
    does not offer a larger source radius as the remedy."""
    # audit W4R2B-03
    from groundfield.diagnostics import ResourceLimitError
    from groundfield.solver.fem import solve_fem

    with (
        pytest.raises(ResourceLimitError, match="limited to 0.02 m") as info,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        solve_fem(
            _w4r2b_rings(1.03),
            gf.create_engine(backend="fem", segment_length=0.25),
            model="3d",
            max_nodes_3d=100_000,
        )
    text = str(info.value)
    assert "electrode 'g2' runs beside electrode 'g1' 30 mm apart" in text
    assert "FemOptions.source_radius" not in text and "raise max_nodes_3d" in text


def test_w4r2b_03_a_galvanic_piece_is_named_by_its_conductor() -> None:
    """A galvanic conductor 5 cm beside an earthing strip limits the source
    radius to 3.3 cm; the reason names the conductor, not the internal name
    of its piece."""
    # audit W4R2B-03
    from groundfield.solver.fem import _solid

    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "strip", name="g", start=(0, 0, 0.6), end=(8, 0, 0.6), wire_radius=0.005
    )
    gf.create_electrode(w, "rod", name="a", position=(0, 0.05, 0.6), length=2.0)
    gf.create_electrode(w, "rod", name="b", position=(8, 0.05, 0.6), length=2.0)
    gf.create_conductor(
        w,
        name="pen",
        start="a",
        end="b",
        cross_section=50e-6,
        coupling_to_soil="galvanic",
        discretize_segment_length=0.5,
    )
    notes: list[str] = []
    eng = gf.create_engine(backend="fem", segment_length=0.25)
    r = _solid.source_radius(w, eng, _w4r2b_model(w), [], None, None, notes=notes)
    assert abs(r - 0.05 / 1.5) < 1e-9
    assert "conductor 'pen' (its piece" in notes[0] and "__cond_" not in notes[0]


def test_w4r2a_01_the_3d_fallback_of_a_ring_without_room_is_not_silently_off() -> None:
    """Since W4A-03 ``model="auto"`` hands a ring without room for its block
    to the 3-D model, which read +12.5 % against the solid tori (BEM
    22.086716 Ohm; mom +0.18 %) with an estimate of 4.2e-3 and no warning.
    The solve must refuse, warn, or be within its estimate."""
    # audit W4R2A-01 (test of auditor W4R2A)
    w = _w4r2b_rings(1.0 + 1.8 * 0.005, depth_b=0.8 + 1.8 * 0.005)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            res = gf.create_engine(backend="fem", segment_length=0.25).solve(w)
        except gf.BackendCapabilityError as exc:
            assert "'mom'" in str(exc) or "'image'" in str(exc)
            return
    warned = [
        r for r in rec if not issubclass(r.category, gf.ExperimentalBackendWarning)
    ]
    z = res.cluster_impedance("g1")[0].real
    est = res.metadata["fem_error_estimate"]
    assert warned or abs(z / 22.086716 - 1.0) <= est, (z, est)


def test_w4r2a_01_the_w4a_world_is_refused_with_a_remedy() -> None:
    """Auditor W4A's world (a small ring 0.75 a above a rod head, no room for
    its block) goes to the 3-D model, where the whole ring lies within
    15.2 mm of the rod: refused before meshing, naming the integral engines.
    Base: +0.80 % against the boundary elements, estimate 0.66 %."""
    # audit W4R2A-01
    with pytest.raises(
        gf.BackendCapabilityError, match="whole wire of electrode 'ring'"
    ) as info:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gf.create_engine(backend="fem", segment_length=0.25).solve(_w4a_world())
    assert "15.2 mm" in str(info.value) and "'mom'" in str(info.value)


@pytest.mark.slow
def test_w4r2b_03_two_coaxial_rings_at_r_over_0_8_within_the_estimate() -> None:
    """Two coaxial rings 1.0 / 1.075 m (7.5 cm = R/0.8 apart), 0.8 m deep, as
    two ports: R11, R22 and the bonded value of the 3-D model lie within its
    error estimate of part A. Base: R11 +4.35e-3 / R22 +3.53e-3 against an
    estimate of 3.85e-3 (about 40 s, 1.6 GB)."""
    # audit W4R2B-03 (test of auditor W4R2B)
    from groundfield.solver.fem import solve_fem

    w = _w4r2b_rings(1.075)
    eng = gf.create_engine(backend="fem", segment_length=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axi = np.asarray(
            solve_fem(w, eng, model="axisymmetric").metadata["fem_port_resistance"]
        )
        r3 = solve_fem(w, eng, model="3d")
    est = r3.metadata["fem_error_estimate"]
    r = np.asarray(r3.metadata["fem_port_resistance"])
    for i in (0, 1):
        assert abs(r[i, i] / axi[i, i] - 1.0) <= est
    bonded = 1.0 / np.linalg.inv(r).sum() / (1.0 / np.linalg.inv(axi).sum()) - 1.0
    assert abs(bonded) <= est
    assert abs(r3.metadata["fem_source_radius"] - 0.05) < 5e-5


def test_w4r2b_04_resolved_concrete_with_a_thin_cover_is_refused_before_meshing(
    monkeypatch,
) -> None:
    """Resolved concrete (Q40) around a 5 mm bar with 5 mm cover: the metal
    radius (5 mm) exceeds what the energy radius of the largest source the
    prism allows reaches beyond (0.544 x 0.45 x side = 4.1 mm), so the radius
    correction turned negative. Refused with a remedy before meshing. Base:
    a bare ValueError "a non-positive radial resistance of a wire" after
    meshing (6.6 s)."""
    # audit W4R2B-04 (test of auditor W4R2B)
    from groundfield.solver.fem import solve_fem

    _stub_the_mesher(monkeypatch)
    w = gf.create_world(soil=H100)
    c = gf.concrete_shell_coefficient(1000.0, 0.005, 0.005)
    gf.create_electrode(
        w,
        "strip",
        name="g",
        start=(0, 0, 0.8),
        end=(1.0, 0, 0.8),
        wire_radius=0.01,
        concrete_shell_coefficient_ohm_m=c,
    )
    gf.create_source(w, attached_to="g", magnitude=1.0)
    with pytest.raises(gf.BackendCapabilityError, match="concrete") as info:
        solve_fem(
            w,
            gf.create_engine(backend="fem", segment_length=0.25),
            model="3d",
            concrete_resistivity=1000.0,
        )
    assert "concrete_resistivity at None" in str(info.value)
    assert "metal radius of 0.005 m" in str(info.value)


def test_w4r2b_05_the_transfer_estimate_is_the_level_difference_of_the_transfer_entries() -> (
    None
):
    """``fem_error_estimate_transfer``: the largest relative difference of an
    off-diagonal entry of the port resistance matrix between the
    extrapolated and the finer level (base: no such key; the estimate covered
    the diagonal only); NaN for one port and for one level."""
    # audit W4R2B-05
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "rod", name="a", position=(0, 0, 0), length=2.0, wire_radius=0.008
    )
    gf.create_electrode(
        w, "rod", name="b", position=(10, 0, 0), length=2.0, wire_radius=0.008
    )
    gf.create_source(w, attached_to="a", magnitude=1.0, return_to="b")
    md = _solve(w).metadata
    r_ext = np.asarray(md["fem_port_resistance"])
    r_fine = np.asarray(md["fem_port_resistance_levels"][-1])
    expected = abs(r_ext[0, 1] - r_fine[0, 1]) / abs(r_ext[0, 1])
    assert md["fem_error_estimate_transfer"] == pytest.approx(expected, rel=1e-12)
    assert 0.0 < md["fem_error_estimate_transfer"] < 0.05
    one = _solve(_rod(H100, length=2.0), extrapolate=False).metadata
    assert math.isnan(one["fem_error_estimate_transfer"])
    assert math.isnan(
        _solve(w, extrapolate=False).metadata["fem_error_estimate_transfer"]
    )
    from groundfield.solver.fem import _solid

    assert math.isnan(_solid.transfer_estimate(np.ones((1, 1)), np.ones((1, 1))))


def _w4r2b_cg_refusal(monkeypatch, world) -> str:
    monkeypatch.setattr(asm3, "CG_MAX_ITERATIONS", 2)
    with pytest.raises(gf.BackendCapabilityError, match="did not converge") as info:
        _solve(world)
    return str(info.value)


def test_w4r2b_06_a_cg_failure_without_layer_contrast_names_its_circumstances(
    monkeypatch,
) -> None:
    """In homogeneous soil the refusal does not blame "a layer contrast of
    1", and it names model='auto' only for a world the axisymmetric model
    takes (a rod, not a strip). Base: "at a layer contrast of 1" and
    model='auto' for every world."""
    # audit W4R2B-06
    text = _w4r2b_cg_refusal(monkeypatch, _strip_world())
    assert "contrast of 1 " not in text and "contrast of 1." not in text
    assert "homogeneous soil" in text
    assert "model='auto'" not in text and "'image'" in text
    text = _w4r2b_cg_refusal(monkeypatch, _rod(H100))
    assert "homogeneous soil" in text and "model='axisymmetric'" in text


def test_w4r2a_07_solid_earth_defaults_to_the_3d_node_limit() -> None:
    """solid_earth(max_nodes=...) defaults to the 2 million nodes of
    FemOptions.max_nodes_3d (base: 4 million)."""
    # audit W4R2A-07
    import inspect

    from groundfield.solver.fem import _solid
    from groundfield.solver.registry import FemOptions

    default = inspect.signature(_solid.solid_earth).parameters["max_nodes"].default
    assert default == FemOptions().max_nodes_3d == 2_000_000


# ---------------------------------------------------------------------
# Audit W4R3B (stream fw4r3b): collinear wires of two ports, wires that
# meet at a small angle, the turn of the own chain, crossings where the
# surfaces stop touching, the texts
# ---------------------------------------------------------------------


def _w4r3b_until_node_check(world):
    """The exception of a 3-D solve with ``max_nodes_3d = 1`` (before meshing)."""
    eng = gf.create_engine(
        backend="fem",
        segment_length=0.25,
        options={"fem": {"model": "3d", "max_nodes_3d": 1}},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            eng.solve(world)
        except Exception as exc:  # noqa: BLE001
            return exc
    raise AssertionError("the solve passed the node check with max_nodes_3d = 1")


def _w4r3b_stacked(gap, a=0.008, source=True):
    """Rod 'g' 0.5 … 2.0 m and rod 'h' of 1.5 m from 2.0 + gap m on the same
    vertical axis."""
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "rod", name="g", position=(0, 0, 0.5), length=1.5, wire_radius=a
    )
    gf.create_electrode(
        w, "rod", name="h", position=(0, 0, 2.0 + gap), length=1.5, wire_radius=a
    )
    if source:
        gf.create_source(w, attached_to="g", magnitude=1.0)
    return w


def test_w4r3b_01_collinear_wires_of_two_ports_with_a_small_gap_limit_the_source_radius() -> (
    None
):
    """Two rods of two electrodes on one axis, 1 cm end to end: their
    smoothed sources overlap at the gap (R = 6 cm), and the 3-D port entries
    read R11 +1.6e-2 and R12 -7.3e-2 against the axisymmetric model
    (estimates 4.9e-3 / 4.5e-3). The solve must limit the source radius or
    refuse. Base: beside_wires leaves collinear pieces alone, R stays 6 cm
    and the solve reaches the node check unlimited."""
    # audit W4R3B-01 (test of auditor W4R3B)
    from groundfield.diagnostics import ResourceLimitError

    exc = _w4r3b_until_node_check(_w4r3b_stacked(0.01))
    if isinstance(exc, ResourceLimitError):
        assert "source radius was limited" in str(exc)
    else:
        assert type(exc).__name__ == "BackendCapabilityError"


def test_w4r3b_01_end_gaps_below_the_smallest_source_radius_are_refused() -> None:
    """End gaps of 2 mm, 5 mm and 1 cm between rods of a = 8 mm (the source
    radius would be 1.3 … 6.7 mm, below the 17.6 mm that the wires need) are
    refused before meshing, naming the gap; the second wire tilted by 1e-4
    rad, or 0.4 rad (23 degrees), still lies in line. Base: solved at R = 6
    cm (R11 +5.1e-2 / +2.6e-2 / +1.6e-2 against the axisymmetric model)."""
    # audit W4R3B-01
    for gap, mm in ((0.002, "2 mm"), (0.005, "5 mm"), (0.01, "10 mm")):
        exc = _w4r3b_until_node_check(_w4r3b_stacked(gap))
        assert isinstance(exc, gf.BackendCapabilityError), (gap, str(exc))
        text = str(exc)
        assert f"lie end to end, {mm} apart without touching" in text
        assert "'mom'" in text
    a = 0.008
    for tilt in (1e-4, 0.4):
        top = np.array([0.0, 0.0, 2.005])
        foot = top + 1.5 * np.array([math.sin(tilt), 0.0, math.cos(tilt)])
        model = wires3.WireModel(
            names=("g", "h"),
            pieces=[
                wires3.WirePiece(
                    0, np.array([0.0, 0, 0.5]), np.array([0.0, 0, 2.0]), a, 0.0, 0
                ),
                wires3.WirePiece(1, top, foot, a, 0.0, 1),
            ],
        )
        b = wires3.beside_wires(model, 1.5 * R_SRC)
        assert b.kind == "in line" and abs(b.distance - 0.005) < 1e-12, tilt
        assert len(set(wires3.wire_systems(model))) == 2


def test_w4r3b_01_collinear_wires_meet_only_without_a_gap() -> None:
    """Collinear wires meet (one wire system) only where the gap is zero to
    round-off (1e-9 m); a gap of 1 mm, below the 8 mm radius, separates
    them. Base: a gap up to the radius 'met' (the 5 mm gap one system)."""
    # audit W4R3B-01
    for gap, systems in ((0.0, 1), (1e-12, 1), (1e-3, 2), (0.005, 2)):
        model = wires3.world_pieces(_w4r3b_stacked(gap, source=False))
        assert len(set(wires3.wire_systems(model))) == systems, gap


def test_w4r3b_01_a_rod_cut_into_touching_pieces_is_unchanged() -> None:
    """A rod cut into two touching rods of two electrodes (gap 0): one wire
    system, nothing beside each other, and the radius correction of the
    uncut rod at every point (both pieces and both images count whole)."""
    # audit W4R3B-01
    from groundfield.solver.fem import _solid

    cut = _w4r3b_stacked(0.0)
    model = wires3.world_pieces(cut)
    assert wires3.beside_wires(model, 1.5 * R_SRC) is None
    eng = gf.create_engine(backend="fem", segment_length=0.25)
    assert _solid.source_radius(cut, eng, model, [], None, None) == 0.06
    r_eta = wires3.BUMP_ENERGY_RADIUS * R_SRC
    a = 0.008
    pts = [[0.0, 0.0, z] for z in (0.51, 1.0, 1.99, 2.0, 2.01, 3.0, 3.49)]
    pieces = [
        wires3.WirePiece(0, np.array([0.0, 0, 0.5]), np.array([0.0, 0, 2.0]), a, 0.0),
        wires3.WirePiece(1, np.array([0.0, 0, 2.0]), np.array([0.0, 0, 3.5]), a, 0.0),
    ]
    whole = [
        wires3.WirePiece(0, np.array([0.0, 0, 0.5]), np.array([0.0, 0, 3.5]), a, 0.0)
    ]
    rho = np.full(len(pts), 100.0)
    near = 1.5 * R_SRC
    c_cut = wires3.radius_correction(_w4b_model_at(pieces, pts), r_eta, rho, near)
    c_whole = wires3.radius_correction(_w4b_model_at(whole, pts), r_eta, rho, near)
    assert np.allclose(c_cut, c_whole, rtol=1e-12, atol=0.0)


def test_w4r3b_01_a_collinear_wire_that_does_not_meet_counts_nothing() -> None:
    """A collinear rod of another electrode 10 cm below the tip (beyond the
    1.5 R window, so the source radius stays 6 cm), or 5 mm below it (such
    a world is refused), carries its own current: the correction of the
    upper rod is that of the rod alone. Base: the whole lower rod and its
    image counted with the observer's current (+0.82 % / +29 % of a self
    term 5 mm above the tip)."""
    # audit W4R3B-01
    alone = gf.create_world(soil=H100)
    gf.create_electrode(
        alone, "rod", name="g", position=(0, 0, 0.5), length=1.5, wire_radius=0.008
    )
    m1, d1, _ = _w4r2b_correction(alone)
    for gap in (0.10, 0.005):
        m2, d2, _ = _w4r2b_correction(_w4r3b_stacked(gap, source=False))
        for z in (0.6, 1.5, 1.995):
            x = (0.0, 0.0, z)
            assert abs(_w4r2b_at(m2, d2, x) / _w4r2b_at(m1, d1, x) - 1.0) < 1e-12


# audit W4R3B-01 (slow: one 3-D solve of about 600 000 nodes)
@pytest.mark.slow
def test_w4r3b_01_stacked_rods_of_two_ports_3_cm_apart_within_their_estimates() -> None:
    """The same rods 3 cm end to end (a smaller R is possible: 2 cm >= 2.2 a):
    every 3-D port entry and the bonded value must lie within their
    estimates of the axisymmetric model's, or the world be refused. Base:
    R11 +6.35e-3 (estimate 4.90e-3), R12 -2.88e-2 (transfer estimate
    4.46e-3)."""
    # audit W4R3B-01 (test of auditor W4R3B)
    w = _w4r3b_stacked(0.03)
    eng = gf.create_engine(backend="fem", segment_length=0.25)
    from groundfield.solver.fem import solve_fem

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra = solve_fem(w, eng, model="axisymmetric")
        try:
            r3 = solve_fem(w, eng, model="3d")
        except (gf.BackendCapabilityError, gf.ResourceLimitError):
            return
    axi = np.asarray(ra.metadata["fem_port_resistance"])
    r = np.asarray(r3.metadata["fem_port_resistance"])
    dev = r / axi - 1.0
    est = r3.metadata["fem_error_estimate"]
    assert abs(dev[0, 0]) <= est and abs(dev[1, 1]) <= est
    assert abs(dev[0, 1]) <= r3.metadata["fem_error_estimate_transfer"]
    bonded = 1.0 / np.linalg.inv(r).sum() / (1.0 / np.linalg.inv(axi).sum()) - 1.0
    assert abs(bonded) <= est
    assert abs(r3.metadata["fem_source_radius"] - 0.02) < 1e-9


def _w4r3b_v_two(theta_deg, L=4.0, z=0.8, offset=0.0):
    """A V of two bonded strips from the apex (``offset``: the second one
    starts that far beside the apex, so the two do not meet)."""
    w = gf.create_world(soil=H100)
    t = math.radians(theta_deg)
    gf.create_electrode(
        w, "strip", name="g", start=(0, 0, z), end=(L, 0, z), wire_radius=0.005
    )
    gf.create_electrode(
        w,
        "strip",
        name="h",
        start=(0, offset, z),
        end=(L * math.cos(t), offset + L * math.sin(t), z),
        wire_radius=0.005,
    )
    gf.create_conductor(w, start="g", end="h")
    gf.create_source(w, attached_to="g", magnitude=1.0)
    return w


def _w4r3b_v_poly(theta_deg, L=4.0, z=0.8):
    w = gf.create_world(soil=H100)
    t = math.radians(theta_deg)
    gf.create_electrode(
        w,
        "polyline",
        name="g",
        closed=False,
        wire_radius=0.005,
        vertices=[(L, 0, z), (0, 0, z), (L * math.cos(t), L * math.sin(t), z)],
    )
    gf.create_source(w, attached_to="g", magnitude=1.0)
    return w


def test_w4r3b_02_a_v_of_two_bonded_strips_29_degrees_wide_is_not_refused() -> None:
    """Two strips that meet at the apex of a 29-degree V: rule (a) of
    beside_wires reported the sample point nearest the apex (reach / 24
    sin 29 deg = 1.82 mm) and the world was refused before meshing, while
    the 31-degree V solves at R = 6 cm; with the rule off the 25 / 29 /
    31-degree V read +9.54e-3 / +9.82e-3 / +9.93e-3 against mom,
    continuously. Base: BackendCapabilityError "... 1.82 mm apart ..."."""
    # audit W4R3B-02 (test of auditor W4R3B)
    from groundfield.diagnostics import ResourceLimitError

    exc = _w4r3b_until_node_check(_w4r3b_v_two(29.0))
    assert isinstance(exc, ResourceLimitError), str(exc)[:300]


def test_w4r3b_02_the_beside_distance_of_wires_that_meet_does_not_follow_the_sampling() -> (
    None
):
    """The distance beside_wires reported for a 10-degree V was a property
    of the sampling, not of the geometry: 0.651 mm at a reach of 9 cm,
    0.217 mm at 3 cm (base)."""
    # audit W4R3B-02 (test of auditor W4R3B)
    model = wires3.world_pieces(_w4r3b_v_two(10.0))
    a, b = wires3.beside_wires(model, 0.09), wires3.beside_wires(model, 0.03)
    if a is not None and b is not None:
        assert a.distance == pytest.approx(b.distance, rel=1e-6)


def test_w4r3b_02_wires_that_meet_at_a_small_angle_are_left_alone() -> None:
    """A V of 10 and 20 degrees (two bonded strips and one polyline) and a
    branch leaving a strip's middle at 20 degrees keep the default source
    radius; near the apex the window term counts the other wire, whose
    current agrees there. A V whose arms do not meet (the second strip
    starts 3 cm beside the apex) still runs beside: at the closest distance
    of the two axes, whatever the reach."""
    # audit W4R3B-02
    from groundfield.solver.fem import _solid

    eng = gf.create_engine(backend="fem", segment_length=0.25)
    worlds = [_w4r3b_v_two(t) for t in (10.0, 20.0)]
    worlds += [_w4r3b_v_poly(t) for t in (10.0, 20.0)]
    y = gf.create_world(soil=H100)
    gf.create_electrode(
        y, "strip", name="g", start=(-2, 0, 0.8), end=(2, 0, 0.8), wire_radius=0.005
    )
    t = math.radians(20.0)
    gf.create_electrode(
        y,
        "strip",
        name="h",
        start=(0, 0, 0.8),
        end=(3 * math.cos(t), 3 * math.sin(t), 0.8),
        wire_radius=0.005,
    )
    worlds.append(y)
    for w in worlds:
        model = _w4r2b_model(w)
        assert wires3.beside_wires(model, 1.5 * R_SRC) is None
        assert _solid.source_radius(w, eng, model, [], None, None) == 0.06
    model = _w4r2b_model(_w4r3b_v_two(10.0, offset=0.03))
    for reach in (0.09, 0.12):
        b = wires3.beside_wires(model, reach)
        assert b.kind == "beside" and abs(b.distance - 0.03) < 1e-12


def _w4r3b_meander(theta, w=0.10, L=4.0, z=0.8):
    world = gf.create_world(soil=H100)
    gf.create_electrode(
        world,
        "polyline",
        name="g",
        closed=False,
        wire_radius=0.005,
        vertices=[
            (0, 0, z),
            (L, 0, z),
            (L, w, z),
            (0, w, z),
            (0, 2 * w, z),
            (L, 2 * w + L * math.tan(theta), z),
        ],
    )
    return world


def _w4r3b_mean(model, dc, port=None) -> float:
    sel = np.ones(dc.size, bool) if port is None else model.port == port
    return float(np.sum((dc * model.weights)[sel]) / np.sum(model.weights[sel]))


def test_w4r3b_03_a_meander_arm_running_the_same_way_counts_like_another_wire() -> None:
    """An S-meander whose third arm runs the same way as the first, 0.20 m
    beside it (outside 1.5 R): tilting the third arm by 1e-4 rad must not
    change the correction in the middle of the first arm. Base: d_obs . e =
    +1 > -1/2 counted the tilted arm whole, +6.80e-3 (the closed-form line
    term 6.84e-3); the full solve jumped +4.65e-4 while mom moved -1.3e-4."""
    # audit W4R3B-03 (test of auditor W4R3B)
    m0, d0, _ = _w4r2b_correction(_w4r3b_meander(0.0))
    m1, d1, _ = _w4r2b_correction(_w4r3b_meander(1e-4))
    x = (2.0, 0.0, 0.8)
    assert abs(_w4r2b_at(m1, d1, x) / _w4r2b_at(m0, d0, x) - 1.0) < 1e-3


def test_w4r3b_03_a_spiral_as_one_polyline_and_as_bonded_windings() -> None:
    """A flat spiral (pitch 0.15 m > 1.5 R, five windings) as one polyline
    and as one electrode per winding (consecutive windings meet): the mean
    correction must agree. Base: +2.43e-2 (every neighbouring winding
    counted whole); the full solve of three windings read +2.08e-3 in Z
    while mom and image give identical values for both."""
    # audit W4R3B-03 (test of auditor W4R3B)
    z, dphi = 0.8, math.radians(2.0)
    phi = np.arange(0.0, 2 * math.pi * 5 + 1e-9, dphi)
    r = 0.3 + 0.15 * phi / (2 * math.pi)
    v = np.stack([r * np.cos(phi), r * np.sin(phi), np.full_like(phi, z)], axis=1)
    one = gf.create_world(soil=H100)
    gf.create_electrode(
        one,
        "polyline",
        name="g",
        closed=False,
        wire_radius=0.005,
        vertices=[tuple(p) for p in v],
    )
    split = gf.create_world(soil=H100)
    per = int(round(2 * math.pi / dphi))
    for k in range(5):
        gf.create_electrode(
            split,
            "polyline",
            name=f"t{k}",
            closed=False,
            wire_radius=0.005,
            vertices=[tuple(p) for p in v[k * per : (k + 1) * per + 1]],
        )
    mp, dp, _ = _w4r2b_correction(one)
    ms, ds, _ = _w4r2b_correction(split)
    assert abs(_w4r3b_mean(mp, dp) / _w4r3b_mean(ms, ds) - 1.0) < 1e-3


def _w4r3b_v_turn(turn_deg, L, z=0.8):
    """A V polyline whose second arm has turned by ``turn_deg`` at the apex."""
    t = math.radians(180.0 - turn_deg)
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w,
        "polyline",
        name="g",
        closed=False,
        wire_radius=0.005,
        vertices=[(L, 0, z), (0, 0, z), (L * math.cos(t), L * math.sin(t), z)],
    )
    return w


def test_w4r3b_03_the_own_chain_fades_out_continuously_with_its_turn() -> None:
    """The own polyline counts whole up to a turn of 90 degrees, like
    another wire of the electrode from 150 degrees on, and in between by a
    weight linear in the turn: the V of 0.3 m arms changes by less than
    1e-4 across a turn of 120 degrees (base: a jump of -7.7e-3 of its mean
    correction; -1.26e-3 of Z for 0.5 m arms, mom +3.4e-4), and at a turn
    of 120 degrees the other arm counts half whole and half by its window."""
    # audit W4R3B-03
    means = []
    for turn in (119.7, 119.9, 120.1, 120.3):
        m, d, _ = _w4r2b_correction(_w4r3b_v_turn(turn, 0.3))
        means.append(_w4r3b_mean(m, d))
    steps = np.diff(means) / means[0]
    # the V narrows: +1.57e-4 per 0.2 degrees on either side of 120 degrees
    assert abs(steps[1] - 0.5 * (steps[0] + steps[2])) < 1e-5
    assert 0.0 < steps[1] < 2e-4
    model, dc, r_eta = _w4r2b_correction(_w4r3b_v_turn(120.0, 0.3))
    x = model.points[np.argmin(np.linalg.norm(model.points - [0.2, 0, 0.8], axis=1))]
    p = model.pieces[1]
    y = np.asarray([x])
    a = np.array([0.005])
    whole = wires3._line_integral(y, p.start, p.end, a)[0]
    whole -= wires3._line_integral(y, p.start, p.end, np.array([r_eta]))[0]
    t = float((x - p.start) @ p.direction)
    half = math.sqrt(
        max(
            0.0,
            (1.5 * R_SRC) ** 2
            - float(wires3._axis_distance2(y - p.start, p.direction)[0]),
        )
    )
    lo, hi = np.clip([t - half, t + half], 0.0, p.length)
    window = wires3._line_integral(y, p.start, p.end, a, lo, hi)[0]
    window -= wires3._line_integral(y, p.start, p.end, np.array([r_eta]), lo, hi)[0]
    own = _w4r2b_whole(model, x, r_eta, (0,))
    expected = own + 100.0 / (4 * math.pi) * (0.5 * whole + 0.5 * window)
    assert abs(_w4r2b_at(model, dc, x) / expected - 1.0) < 1e-12


def _w4r3b_cross(gap, L=4.0, z=0.8):
    w = gf.create_world(soil=H100)
    gf.create_electrode(
        w, "strip", name="g", start=(-L / 2, 0, z), end=(L / 2, 0, z), wire_radius=0.005
    )
    gf.create_electrode(
        w,
        "strip",
        name="h",
        start=(0, -L / 2, z + gap),
        end=(0, L / 2, z + gap),
        wire_radius=0.005,
    )
    gf.create_conductor(w, start="g", end="h")
    return w


def test_w4r3b_04_the_correction_of_crossing_wires_is_continuous_where_they_stop_meeting() -> (
    None
):
    """Two bonded strips crossing at 90 degrees, axes 4.9 / 5.1 mm apart
    (wire radius 5 mm, the surfaces overlap in both): wire_systems switched,
    and the window term of the other strip disappeared. Base: the mean
    correction of the 4 m strip dropped by 8.5e-3; the full solve of a 2 m
    cross jumped by -7.5e-4 (mom -1.6e-5, image -3.9e-5)."""
    # audit W4R3B-04 (test of auditor W4R3B)
    m0, d0, _ = _w4r2b_correction(_w4r3b_cross(0.0049, L=2.0))
    m1, d1, _ = _w4r2b_correction(_w4r3b_cross(0.0051, L=2.0))
    assert abs(_w4r3b_mean(m1, d1, port=0) / _w4r3b_mean(m0, d0, port=0) - 1.0) < 1e-3


def test_w4r3b_04_crossing_wires_meet_where_their_surfaces_touch_and_fade_out() -> None:
    """Wires meet where their surfaces touch (axis distance at most a1 + a2
    = 10 mm here); beyond, the other strip's window term fades out linearly
    to 2 (a1 + a2), so the correction is continuous at both ends and from
    20 mm on equals that of the strip alone."""
    # audit W4R3B-04
    assert len(set(wires3.wire_systems(_w4r2b_model(_w4r3b_cross(0.0099))))) == 1
    assert len(set(wires3.wire_systems(_w4r2b_model(_w4r3b_cross(0.0101))))) == 2
    alone = gf.create_world(soil=H100)
    gf.create_electrode(
        alone, "strip", name="g", start=(-2, 0, 0.8), end=(2, 0, 0.8), wire_radius=0.005
    )
    ma, da, _ = _w4r2b_correction(alone)
    base = _w4r3b_mean(ma, da)
    mean = {}
    for gap in (0.0099, 0.0101, 0.015, 0.0199, 0.0201, 0.03):
        m, d, _ = _w4r2b_correction(_w4r3b_cross(gap))
        mean[gap] = _w4r3b_mean(m, d, port=0) / base - 1.0
    # base: a jump of 8.5e-3 at the switch (4.9 / 5.1 mm); the change over
    # 0.2 mm is now that of the geometry and of the fading weight
    assert abs(mean[0.0101] - mean[0.0099]) < 3e-4
    assert abs(mean[0.0201] - mean[0.0199]) < 3e-4
    assert mean[0.0099] > mean[0.015] > mean[0.0199] > 0.0
    assert mean[0.0201] == 0.0 and mean[0.03] == 0.0
