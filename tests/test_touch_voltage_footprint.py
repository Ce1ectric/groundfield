"""Touch voltage anchored on the electrode footprint (10.21) and the
uniform-current caveat of the safety helpers (10.22).

The conventional touch-voltage position of EN 50522 is a person standing
1 m away from the touched part. Up to v0.15.0 the helpers measured that
metre from the electrode's *connection point* — the centre of a grid or
star — so the feet of a "1 m" touch voltage could stand above the middle
of a 20 m grid (F13). These tests pin the footprint anchor: the feet stand
``distance`` metres (horizontally) from the nearest wire of the electrode,
on the side given by ``direction``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.geometry.electrodes import GridMeshElectrode
from groundfield.postprocess.safety import (
    step_voltage,
    touch_voltage,
    touch_voltage_envelope,
)


def _solve(world: gf.World, backend: str = "image", ds: float = 1.0):
    eng = gf.create_engine(backend=backend, segment_length=ds, frequencies=[50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eng.solve(world)


def _grid_world(n: int = 4, size: float = 20.0) -> gf.World:
    """Review-pass-10 setup (R5-03): 20 m x 20 m grid, 4 x 4 meshes, 100 A."""
    return gf.World(
        soil=gf.HomogeneousSoil(resistivity=100.0),
        electrodes=[
            GridMeshElectrode(
                name="g1", corner=(0.0, 0.0, 0.5), size=(size, size), n_x=n, n_y=n
            )
        ],
        sources=[gf.CurrentSource(name="s", attached_to="g1", magnitude=100.0)],
    )


def _star_world() -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        w, "star", name="st", center=(0.0, 0.0, 0.6), n_arms=4, arm_length=4.0
    )
    gf.create_source(w, attached_to="st", magnitude=100.0)
    return w


def _ring_world() -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "ring", name="r", center=(2.0, 3.0, 0.8), radius=4.0)
    gf.create_source(w, attached_to="r", magnitude=10.0)
    return w


def _ue(result, name: str) -> complex:
    return complex(result.electrode_potentials[name][0])


def _phi(result, x: float, y: float) -> complex:
    return complex(result.potential(np.array([[x, y, 0.0]]))[0])


# ---------------------------------------------------------------------
# independent geometry: distance from a surface point to the wires
# ---------------------------------------------------------------------


def _dist_to_segments(p: np.ndarray, segs: list[tuple[tuple, tuple]]) -> float:
    best = math.inf
    for a, b in segs:
        a = np.asarray(a[:2], float)
        b = np.asarray(b[:2], float)
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 == 0.0 else min(1.0, max(0.0, float((p - a) @ ab) / L2))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def _grid_segments(size: float, n: int) -> list:
    out = []
    for k in range(n + 1):
        c = size * k / n
        out.append(((0.0, c), (size, c)))
        out.append(((c, 0.0), (c, size)))
    return out


# ---------------------------------------------------------------------
# 10.21 — footprint anchor
# ---------------------------------------------------------------------


@pytest.mark.finding("10.21")
def test_grid_touch_voltage_stands_one_metre_outside_the_edge() -> None:
    """Default direction +x: feet at (21, 10), 1 m outside the east edge.

    v0.15.0 put the feet 1 m from the grid centre (11, 10), above the
    grid: 9.92 V instead of 69.6 V (R5-03).
    """
    w = _grid_world()
    res = _solve(w)
    u_t = touch_voltage(res, w, electrode="g1")
    expected = _ue(res, "g1") - _phi(res, 21.0, 10.0)
    assert u_t == pytest.approx(expected, rel=1e-12)
    assert abs(u_t) > 60.0


@pytest.mark.finding("10.21")
def test_grid_envelope_covers_the_one_metre_offset_curve() -> None:
    """Envelope max >= 99 % of a dense brute-force walk 1 m outside the grid."""
    w = _grid_world()
    res = _solve(w)
    # Independent offset curve of the 20 m square at 1 m: four straight
    # sides and four quarter circles around the corners.
    s = np.linspace(0.0, 20.0, 201)
    pts = [np.column_stack([s, np.full_like(s, -1.0)])]
    pts.append(np.column_stack([s, np.full_like(s, 21.0)]))
    pts.append(np.column_stack([np.full_like(s, -1.0), s]))
    pts.append(np.column_stack([np.full_like(s, 21.0), s]))
    th = np.linspace(0.0, 0.5 * np.pi, 91)
    for cx, cy, sx, sy in (
        (20, 20, 1, 1),
        (0, 20, -1, 1),
        (0, 0, -1, -1),
        (20, 0, 1, -1),
    ):
        pts.append(np.column_stack([cx + sx * np.cos(th), cy + sy * np.sin(th)]))
    p = np.vstack(pts)
    brute = np.abs(
        _ue(res, "g1") - res.potential(np.column_stack([p, np.zeros(len(p))]))
    )
    _, env = touch_voltage_envelope(res, w, electrode="g1", n_angles=24)
    assert np.abs(env).max() >= 0.99 * brute.max()
    # ... and an 11.9x under-estimate like v0.15.0 is impossible.
    assert np.abs(env).max() > 100.0


@pytest.mark.finding("10.21")
def test_grid_envelope_includes_the_mesh_centres() -> None:
    w = _grid_world()
    res = _solve(w)
    angles, env, feet = touch_voltage_envelope(
        res, w, electrode="g1", n_angles=24, return_points=True
    )
    assert feet.shape == (24 + 16, 3)
    assert angles.shape == env.shape == (40,)
    centres = {(2.5 + 5 * i, 2.5 + 5 * j) for i in range(4) for j in range(4)}
    got = {(round(x, 9), round(y, 9)) for x, y in feet[24:, :2]}
    assert got == centres
    u_centres = _ue(res, "g1") - res.potential(feet[24:])
    np.testing.assert_allclose(env[24:], u_centres, rtol=1e-12)
    without = touch_voltage_envelope(
        res, w, electrode="g1", n_angles=24, mesh_centres=False
    )
    assert without[1].shape == (24,)


@pytest.mark.finding("10.21")
def test_coarse_grid_mesh_centre_can_dominate_the_envelope() -> None:
    """A 1 x 1 grid of 20 m (perimeter only): the mesh centre, 10 m from
    every wire, is the worst touch position (158 V against 141 V 1 m
    outside), so the envelope must not stop at the perimeter."""
    w = _grid_world(n=1, size=20.0)
    res = _solve(w, ds=1.0)
    angles, env, feet = touch_voltage_envelope(
        res, w, electrode="g1", n_angles=24, return_points=True
    )
    assert feet.shape == (25, 3)
    perimeter = np.abs(env[:24]).max()
    centre = np.abs(env[24])
    assert centre > 1.05 * perimeter
    assert np.abs(env).max() == centre


@pytest.mark.finding("10.21")
def test_star_touch_voltage_beyond_the_arm_tip() -> None:
    """Along an arm the feet stand 1 m beyond the tip (R5-03: x3.0 before)."""
    w = _star_world()
    res = _solve(w, ds=0.5)
    u_t = touch_voltage(res, w, electrode="st", direction=(1.0, 0.0, 0.0))
    assert u_t == pytest.approx(_ue(res, "st") - _phi(res, 5.0, 0.0), rel=1e-12)


@pytest.mark.finding("10.21")
def test_ring_touch_voltage_on_the_far_side_stands_outside() -> None:
    """Direction -x: feet 1 m outside the ring, not 1 m inside it."""
    w = _ring_world()
    res = _solve(w, ds=0.5)
    u_t = touch_voltage(res, w, electrode="r", direction=(-1.0, 0.0, 0.0))
    expected = _ue(res, "r") - _phi(res, 2.0 - 4.0 - 1.0, 3.0)
    assert u_t == pytest.approx(expected, rel=1e-9)


@pytest.mark.finding("10.21")
def test_connection_point_anchor_reproduces_v0150() -> None:
    """``anchor="connection_point"`` keeps the old definition."""
    w = _grid_world()
    res = _solve(w)
    u_old = touch_voltage(res, w, electrode="g1", anchor="connection_point")
    assert u_old == pytest.approx(_ue(res, "g1") - _phi(res, 11.0, 10.0), rel=1e-12)
    angles, env = touch_voltage_envelope(
        res,
        w,
        electrode="g1",
        n_angles=8,
        anchor="connection_point",
        mesh_centres=False,
    )
    feet = np.column_stack([10.0 + np.cos(angles), 10.0 + np.sin(angles), np.zeros(8)])
    np.testing.assert_allclose(env, _ue(res, "g1") - res.potential(feet), rtol=1e-12)


@pytest.mark.finding("10.21")
@pytest.mark.parametrize("distance", [0.3, 1.0, 2.5])
def test_envelope_feet_keep_the_requested_distance_from_every_wire(
    distance: float,
) -> None:
    """Each offset-curve foot is exactly ``distance`` from the nearest wire —
    for a grid, a star, a strip and a non-convex (L-shaped) open polyline."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        w,
        "grid_mesh",
        name="grid",
        corner=(0.0, 0.0, 0.5),
        size=(12.0, 8.0),
        n_x=3,
        n_y=2,
    )
    gf.create_electrode(
        w, "star", name="star", center=(40.0, 0.0, 0.6), n_arms=3, arm_length=3.0
    )
    gf.create_electrode(
        w, "strip", name="strip", start=(0.0, 30.0, 0.5), end=(9.0, 34.0, 0.5)
    )
    gf.create_electrode(
        w,
        "polyline",
        name="ell",
        closed=False,
        vertices=[(30.0, 30.0, 0.8), (40.0, 30.0, 0.8), (40.0, 40.0, 0.8)],
    )
    for name in ("grid", "star", "strip", "ell"):
        gf.create_source(w, attached_to=name, magnitude=1.0, name=f"s_{name}")
    res = _solve(w, backend="mom", ds=0.5)
    wires = {
        "grid": [((x, 0.0), (x, 8.0)) for x in (0.0, 4.0, 8.0, 12.0)]
        + [((0.0, y), (12.0, y)) for y in (0.0, 4.0, 8.0)],
        "star": [((40.0, 0.0), tip[:2]) for tip in w.get_electrode("star").arm_tips],
        "strip": [((0.0, 30.0), (9.0, 34.0))],
        "ell": [((30.0, 30.0), (40.0, 30.0)), ((40.0, 30.0), (40.0, 40.0))],
    }
    for name, segs in wires.items():
        _, _, feet = touch_voltage_envelope(
            res,
            w,
            electrode=name,
            distance=distance,
            n_angles=36,
            mesh_centres=False,
            return_points=True,
        )
        d = [_dist_to_segments(p[:2], segs) for p in feet]
        np.testing.assert_allclose(d, distance, rtol=1e-9, atol=1e-9)


def test_rod_footprint_anchor_equals_connection_point_anchor() -> None:
    """For a rod (point footprint) both anchors coincide — no change."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="rod", position=(1.0, 2.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    res = _solve(w, backend="mom", ds=0.25)
    for direction in ((1.0, 0.0, 0.0), (0.3, -0.7, 0.0)):
        a = touch_voltage(res, w, electrode="rod", direction=direction)
        e = np.asarray(direction) / np.linalg.norm(direction)
        b = _ue(res, "rod") - _phi(res, 1.0 + e[0], 2.0 + e[1])
        assert a == pytest.approx(b, rel=1e-12)


# ---------------------------------------------------------------------
# 10.22 — uniform-current caveat
# ---------------------------------------------------------------------


def _caveats(record) -> list:
    from groundfield.postprocess.safety import UniformCurrentFieldWarning

    return [x for x in record if issubclass(x.category, UniformCurrentFieldWarning)]


@pytest.mark.finding("10.22")
def test_uniform_current_backend_warns_once_per_call_near_a_grid() -> None:
    w = _grid_world()
    res = _solve(w, backend="image")
    for call in (
        lambda: touch_voltage(res, w, electrode="g1"),
        lambda: touch_voltage_envelope(res, w, electrode="g1", n_angles=36),
        lambda: step_voltage(res, position=(20.5, 10.0, 0.0)),
        lambda: step_voltage(res, position=(20.5, 10.0, 0.0), world=w),
    ):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            call()
        caveats = _caveats(record)
        assert len(caveats) == 1
        assert "mom" in str(caveats[0].message)


@pytest.mark.finding("10.22")
def test_no_caveat_for_per_segment_backends_or_benign_shapes() -> None:
    """mom results, and rings / rods / closed loops (measured within 1 % of
    mom at the surface) do not warn; points far from any wire do not warn."""
    w = _grid_world()
    res_mom = _solve(w, backend="mom")
    ring = _ring_world()
    res_ring = _solve(ring, backend="image", ds=0.5)
    loop = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        loop,
        "polyline",
        name="fdn",
        vertices=[(0, 0, 0.8), (10, 0, 0.8), (10, 10, 0.8), (0, 10, 0.8)],
    )
    gf.create_source(loop, attached_to="fdn", magnitude=1.0)
    res_loop = _solve(loop, backend="image", ds=0.5)
    res_img = _solve(w, backend="image")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        touch_voltage(res_mom, w, electrode="g1")
        touch_voltage_envelope(res_mom, w, electrode="g1")
        touch_voltage(res_ring, ring, electrode="r")
        touch_voltage_envelope(res_ring, ring, electrode="r")
        touch_voltage_envelope(res_loop, loop, electrode="fdn")
        step_voltage(res_ring, position=(7.0, 3.0, 0.0), world=ring)
        step_voltage(res_img, position=(40.0, 10.0, 0.0))  # 20 m from the grid
    assert _caveats(record) == []


@pytest.mark.finding("10.22")
def test_bonded_multi_electrode_cluster_warns() -> None:
    """Station loop + bonded rods (one uniform current per electrode)."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        w,
        "polyline",
        name="loop",
        vertices=[(0, 0, 0.6), (4, 0, 0.6), (4, 4, 0.6), (0, 4, 0.6)],
    )
    gf.create_electrode(w, "rod", name="r0", position=(4.0, 4.0, 0.6), length=4.0)
    gf.create_conductor(w, name="c0", start="loop", end="r0")
    gf.create_source(w, attached_to="loop", magnitude=1.0)
    res = _solve(w, backend="image", ds=0.5)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        touch_voltage(res, w, electrode="loop")
    assert len(_caveats(record)) == 1
