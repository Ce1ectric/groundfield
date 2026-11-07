"""Exact line-source kernel, ``Engine(kernel="line")`` (review pass 10, 10.28).

The point kernel converges to a grounding resistance 0.9–1.9 % below
the exact uniform-current solution (rod −1.86 %, ring −0.95 % at every
segment length). The line kernel integrates uniform line currents in
Galerkin form; references are the closed forms of the uniform-current
model (R1 review, ``k01_validate_prototype.py``) and Dwight (1936).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.special import ellipk

import groundfield as gf

RHO = 100.0


def _galerkin_self(L: float, a: float) -> float:
    return 2.0 * (np.arcsinh(L / a) - np.sqrt(1.0 + (a / L) ** 2) + a / L) / L


def _ref_rod(L: float, a: float) -> float:
    """Rod from the surface: rod + image = collinear line of length 2L."""
    return RHO / (4 * np.pi) * 2.0 * _galerkin_self(2 * L, a)


def _ref_strip(L: float, a: float, z: float) -> float:
    return RHO / (4 * np.pi) * (_galerkin_self(L, a) + _galerkin_self(L, 2 * z))


def _ref_ring(r: float, a: float, z: float) -> float:
    k1 = ellipk(4 * r * r / (4 * r * r + a * a)) / np.sqrt(4 * r * r + a * a)
    k2 = ellipk(4 * r * r / (4 * r * r + 4 * z * z)) / np.sqrt(4 * r * r + 4 * z * z)
    return RHO / (2 * np.pi**2) * (k1 + k2)


_GEOMETRIES = {
    "rod": (
        dict(kind="rod", position=(0.0, 0.0, 0.0), length=3.0, wire_radius=0.008),
        _ref_rod(3.0, 0.008),
    ),
    "ring": (
        dict(kind="ring", center=(0.0, 0.0, 0.8), radius=5.0),
        _ref_ring(5.0, 0.005, 0.8),
    ),
    "strip": (
        dict(kind="strip", start=(0.0, 0.0, 0.5), end=(10.0, 0.0, 0.5)),
        _ref_strip(10.0, 0.005, 0.5),
    ),
}


def _resistance(geometry: dict, ds: float, kernel: str, soil=None) -> float:
    w = gf.create_world(soil=soil or gf.HomogeneousSoil(resistivity=RHO))
    params = dict(geometry)
    gf.create_electrode(w, params.pop("kind"), name="g", **params)
    gf.create_source(w, attached_to="g", magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = w.solve(gf.Engine(segment_length=ds, kernel=kernel))
    return res.cluster_impedance("g")[0].real


@pytest.mark.finding("10.28")
@pytest.mark.parametrize("name", sorted(_GEOMETRIES))
def test_line_kernel_matches_the_uniform_current_solution(name: str) -> None:
    """Within 0.05 % at ds = 0.5 m (the point kernel: −0.95 … −1.21 %)."""
    geometry, ref = _GEOMETRIES[name]
    assert _resistance(geometry, 0.5, "line") == pytest.approx(ref, rel=5e-4)


@pytest.mark.finding("10.28")
@pytest.mark.parametrize("name", ["rod", "strip"])
def test_straight_wires_are_exact_at_any_segment_length(name: str) -> None:
    geometry, ref = _GEOMETRIES[name]
    for ds in (2.0, 0.5, 0.125):
        assert _resistance(geometry, ds, "line") == pytest.approx(ref, rel=1e-5)


@pytest.mark.finding("10.28")
@pytest.mark.parametrize("name", sorted(_GEOMETRIES))
def test_line_kernel_agrees_with_dwight(name: str) -> None:
    """Dwight's closed forms are the uniform-current values (≤ 0.12 %)."""
    D = gf.references.dwight1936
    dwight = {
        "rod": D.rod(RHO, 3.0, 0.008),
        "ring": D.buried_ring(RHO, 10.0, 0.010, 0.8),
        "strip": D.horizontal_wire(RHO, 5.0, 0.005, 0.5),
    }[name]
    geometry, _ = _GEOMETRIES[name]
    assert _resistance(geometry, 0.25, "line") == pytest.approx(dwight, rel=2e-3)


@pytest.mark.finding("10.28")
def test_ring_converges_monotonically_and_faster_than_the_point_kernel() -> None:
    geometry, ref = _GEOMETRIES["ring"]
    ladder = (2.0, 1.0, 0.5, 0.25)
    err_line = [abs(_resistance(geometry, ds, "line") / ref - 1) for ds in ladder]
    err_point = [abs(_resistance(geometry, ds, "point") / ref - 1) for ds in ladder]
    assert all(b < a for a, b in zip(err_line, err_line[1:], strict=False))
    assert all(el < ep for el, ep in zip(err_line, err_point, strict=True))
    assert err_line[-1] < 2e-4 < 5e-3 < err_point[-1]


@pytest.mark.finding("10.28")
def test_fine_discretisation_reference_of_a_grid() -> None:
    """20 × 20 m grid 4 × 4 meshes: ds = 1 m equals ds = 0.25 m to 1e-5."""
    grid = dict(kind="grid_mesh", corner=(0, 0, 0.5), size=(20.0, 20.0), n_x=4, n_y=4)
    coarse = _resistance(grid, 1.0, "line")
    fine = _resistance(grid, 0.25, "line")
    assert coarse == pytest.approx(fine, rel=1e-5)
    assert _resistance(grid, 0.25, "point") < 0.997 * fine


@pytest.mark.finding("10.28")
def test_point_kernel_is_the_default_and_unchanged() -> None:
    """``kernel="point"`` stays the default (the golden cases pin it)."""
    assert gf.Engine().kernel == "point"
    geometry, _ = _GEOMETRIES["rod"]
    assert _resistance(geometry, 0.5, "point") == pytest.approx(33.095619, abs=1e-6)


@pytest.mark.finding("10.28")
def test_line_kernel_is_rejected_by_other_backends() -> None:
    with pytest.raises(ValueError, match="kernel='line'"):
        gf.Engine(backend="mom", kernel="line")


@pytest.mark.finding("10.28")
def test_result_records_kernel_and_segment_directions() -> None:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=RHO))
    gf.create_electrode(w, "rod", name="g", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="g", magnitude=1.0)
    for kernel in ("point", "line"):
        res = w.solve(gf.Engine(kernel=kernel))
        assert res.metadata["kernel"] == kernel
        np.testing.assert_allclose(
            [ps.direction for ps in res.point_sources], [[0.0, 0.0, 1.0]] * 4
        )


@pytest.mark.finding("10.28")
def test_old_point_source_json_still_loads() -> None:
    """``PointSource.direction`` is optional: results without it load and
    are evaluated with the point kernel."""
    from groundfield.solver.result import FieldResult

    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=RHO))
    gf.create_electrode(w, "rod", name="g", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="g", magnitude=1.0)
    res = w.solve(gf.Engine(kernel="line"))
    data = res.model_dump(mode="json")
    for ps in data["point_sources"]:
        del ps["direction"]
    old = FieldResult.model_validate(data)
    assert all(ps.direction is None for ps in old.point_sources)
    as_point = FieldResult.model_validate({**data, "metadata": {"kernel": "point"}})
    pts = np.array([[1.5, 0.0, 0.0], [0.0, 0.2, 1.0]])
    np.testing.assert_array_equal(old.potential(pts), as_point.potential(pts))
    assert not np.allclose(old.potential(pts), res.potential(pts), rtol=1e-6)


@pytest.mark.finding("10.28")
def test_line_potential_is_equipotential_on_the_wire_surface() -> None:
    """The potential averaged over the rod surface equals the electrode
    potential (Galerkin consistency) and varies smoothly along the rod;
    with point sources the surface potential spikes at every midpoint
    (16 … 177 V around 32 V)."""
    z_all = np.linspace(0.5, 3.5, 30001)[1:-1]
    z_mid = np.linspace(1.0, 3.0, 2001)
    rough = {}
    for kernel in ("line", "point"):
        w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=RHO))
        gf.create_electrode(
            w, "rod", name="g", position=(0.0, 0.0, 0.5), length=3.0, wire_radius=0.008
        )
        gf.create_source(w, attached_to="g", magnitude=1.0)
        res = w.solve(gf.Engine(segment_length=0.5, kernel=kernel))
        u = res.electrode_potentials["g"][0].real
        surface = np.column_stack([np.full_like(z_all, 0.008), 0 * z_all, z_all])
        interior = np.column_stack([np.full_like(z_mid, 0.008), 0 * z_mid, z_mid])
        if kernel == "line":
            assert np.mean(res.potential(surface).real) == pytest.approx(u, rel=2e-3)
        rough[kernel] = np.abs(np.diff(res.potential(interior).real, 2)).max()
    assert rough["line"] < 1e-3 < 1.0 < rough["point"]


@pytest.mark.finding("10.28")
def test_line_field_far_away_matches_the_point_field() -> None:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=RHO))
    gf.create_electrode(w, "ring", name="g", center=(0.0, 0.0, 0.8), radius=3.0)
    gf.create_source(w, attached_to="g", magnitude=1.0)
    line = w.solve(gf.Engine(kernel="line"))
    point = w.solve(gf.Engine(kernel="point"))
    far = np.array([[200.0, 0.0, 0.0], [0.0, 400.0, 0.0]])
    # far field = (total current) rho / (2 pi r) for both kernels
    np.testing.assert_allclose(line.potential(far), point.potential(far), rtol=1e-6)


@pytest.mark.finding("10.28")
def test_two_layer_line_kernel_reduces_to_homogeneous() -> None:
    """rho_1 = rho_2: the two-layer line kernel equals the homogeneous one."""
    rod = _GEOMETRIES["rod"][0]
    hom = _resistance(rod, 0.5, "line")
    two = _resistance(
        rod, 0.5, "line", soil=gf.TwoLayerSoil(rho_1=RHO, rho_2=RHO, h_1=5.0)
    )
    assert two == pytest.approx(hom, rel=1e-12)


@pytest.mark.finding("10.28")
def test_two_layer_line_kernel_converges_for_a_rod_near_the_interface() -> None:
    """Rod tip 0.4 m above h_1: the series images of the line source."""
    soil = gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=4.0)
    rod = dict(kind="rod", position=(0.0, 0.0, 0.6), length=3.0, wire_radius=0.008)
    line = [_resistance(rod, ds, "line", soil=soil) for ds in (0.5, 0.25, 0.125)]
    point = [_resistance(rod, ds, "point", soil=soil) for ds in (0.5, 0.25, 0.125)]
    assert abs(line[0] / line[2] - 1) < 2e-3
    assert abs(line[1] / line[2] - 1) < 5e-4
    assert abs(point[0] / line[2] - 1) > 5e-3


@pytest.mark.finding("10.28")
@pytest.mark.parametrize(
    "soil",
    [
        gf.HomogeneousSoil(resistivity=RHO),
        gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0),
    ],
    ids=["homogeneous", "two_layer"],
)
def test_mutual_matrix_uses_the_line_kernel(soil) -> None:
    """``solve_mutual_matrix`` diagonal = ``cluster_impedance`` of the solve."""
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="a", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="b", position=(30.0, 0.0, 0.5), length=2.0)
    eng = gf.Engine(segment_length=0.5, kernel="line")
    Z = gf.solve_mutual_matrix(
        w, eng, ["a", "b"], np.array([[0.0, 0, 0], [30.0, 0, 0]]), symmetrize=False
    )
    w1 = w.model_copy(deep=True)
    gf.create_source(w1, attached_to="a", magnitude=1.0)
    res = w1.solve(eng)
    assert Z[0, 0] == pytest.approx(res.cluster_impedance("a")[0], rel=1e-10)
    np.testing.assert_allclose(
        Z[1, 0], res.potential(np.array([[30.0, 0.0, 0.0]]))[0], rtol=1e-10
    )
