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
    grid: 9.92 V instead of 69.6 V with one current over the grid (R5-03;
    61.31 V since user decision Q30, see the next test).
    """
    w = _grid_world()
    res = _solve(w)
    u_t = touch_voltage(res, w, electrode="g1")
    expected = _ue(res, "g1") - _phi(res, 21.0, 10.0)
    assert u_t == pytest.approx(expected, rel=1e-12)
    assert abs(u_t) > 60.0


# audit W1A-03 (the release-note number), user decision Q30
@pytest.mark.finding("10.21")
def test_grid_touch_voltage_numbers_of_the_release_note() -> None:
    """The 20 m x 20 m grid of the 10.21 row (100 A, 100 Ohm m, ds 1 m):
    61.31 V at +x with the default line kernel, 59.95 V with the point
    kernel, 61.58 V per segment with ``mom``. One current over the whole
    grid, before user decision Q30, read 69.6 V (point kernel); the
    envelope 98.7 V (111.4 V before Q30)."""
    w = _grid_world()
    line = abs(touch_voltage(_solve(w), w, electrode="g1"))
    point_res = gf.create_engine(
        segment_length=1.0, frequencies=[50.0], kernel="point"
    ).solve(w)
    point = abs(touch_voltage(point_res, w, electrode="g1"))
    mom = abs(touch_voltage(_solve(w, "mom"), w, electrode="g1"))
    assert line == pytest.approx(61.31, abs=0.01)
    assert point == pytest.approx(59.95, abs=0.01)
    assert mom == pytest.approx(61.58, abs=0.01)
    _, env = touch_voltage_envelope(point_res, w, electrode="g1", n_angles=24)
    assert float(np.abs(env).max()) == pytest.approx(98.72, abs=0.01)


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
    # ... and an 11.9x under-estimate like v0.15.0 is impossible (99.8 V
    # with one owner per wire line since user decision Q30, 111.4 V with
    # one owner for the grid before).
    assert np.abs(env).max() > 95.0


@pytest.mark.finding("10.21")
def test_grid_envelope_includes_the_mesh_centres_with_interior_meshes() -> None:
    # user decision Q3: mesh centres on request (open-air grids)
    w = _grid_world()
    res = _solve(w)
    angles, env, feet = touch_voltage_envelope(
        res, w, electrode="g1", n_angles=24, interior="meshes", return_points=True
    )
    assert feet.shape == (24 + 16, 3)
    assert angles.shape == env.shape == (40,)
    centres = {(2.5 + 5 * i, 2.5 + 5 * j) for i in range(4) for j in range(4)}
    got = {(round(x, 9), round(y, 9)) for x, y in feet[24:, :2]}
    assert got == centres
    u_centres = _ue(res, "g1") - res.potential(feet[24:])
    np.testing.assert_allclose(env[24:], u_centres, rtol=1e-12)
    without = touch_voltage_envelope(res, w, electrode="g1", n_angles=24)
    assert without[1].shape == (24,)
    np.testing.assert_array_equal(without[1], env[:24])


@pytest.mark.finding("10.21")
def test_coarse_grid_mesh_centre_can_dominate_the_envelope() -> None:
    """A 1 x 1 grid of 20 m (perimeter only): the mesh centre, 10 m from
    every wire, is the worst touch position (158 V against 141 V 1 m
    outside), so on an open-air grid the envelope must not stop at the
    perimeter: ``interior="meshes"``."""
    w = _grid_world(n=1, size=20.0)
    res = _solve(w, ds=1.0)
    angles, env, feet = touch_voltage_envelope(
        res, w, electrode="g1", n_angles=24, interior="meshes", return_points=True
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
        res, w, electrode="g1", n_angles=8, anchor="connection_point"
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
            interior=False,
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


# ---------------------------------------------------------------------
# 10.79 — the footprint comes from the electrode geometry methods
# ---------------------------------------------------------------------


def _footprint_electrodes() -> list:
    from groundfield.geometry.electrodes import (
        MeshElectrode,
        PolylineElectrode,
        RodElectrode,
        StarElectrode,
        StripElectrode,
    )

    return [
        RodElectrode(name="rod", position=(1.0, 2.0, 0.5), length=3.0),
        StripElectrode(name="strip", start=(0.0, 30.0, 0.5), end=(9.0, 34.0, 0.5)),
        PolylineElectrode(
            name="open",
            closed=False,
            vertices=[(30.0, 30.0, 0.8), (40.0, 30.0, 0.8), (40.0, 40.0, 0.8)],
        ),
        PolylineElectrode(
            name="outline",
            vertices=[
                (0.0, 0.0, 0.8),
                (7.3, 0.0, 0.8),
                (7.3, 4.1, 0.8),
                (0.0, 6.6, 0.8),
            ],
        ),
        StarElectrode(name="star", center=(40.0, 0.0, 0.6), n_arms=3, arm_length=3.0),
        MeshElectrode(
            name="mesh", corner=(0.1, 0.2, 0.5), size=(12.0, 9.9), spacing=2.4
        ),
        # 12 m / 5 and 9.9 m / 7: the discretiser's np.linspace positions
        # differ from corner + size * k / n in the last bit.
        GridMeshElectrode(
            name="grid", corner=(0.0, 0.0, 0.5), size=(12.0, 9.9), n_x=5, n_y=7
        ),
    ]


@pytest.mark.parametrize(
    "index", [*range(6), pytest.param(6, marks=pytest.mark.finding("10.79"))]
)
def test_footprint_wires_are_the_electrode_wire_polylines(index: int) -> None:
    """Bit for bit the wires the electrode reports (and the discretiser
    uses), not a local re-implementation of each kind."""
    from groundfield.postprocess._footprint import footprint

    electrode = _footprint_electrodes()[index]
    expected = sorted(
        (*run[k, :2], *run[k + 1, :2])
        for run in electrode.wire_polylines()
        for k in range(len(run) - 1)
    )
    got = sorted(tuple(seg.ravel()) for seg in footprint(electrode).segments)
    assert got == expected


@pytest.mark.finding("10.79")
def test_mesh_centres_lie_midway_between_the_discretiser_wires() -> None:
    from groundfield.postprocess._footprint import mesh_centres

    for electrode in _footprint_electrodes()[-2:]:
        seg = electrode.segments(0.3)
        along_y = np.abs(seg.directions[:, 1]) == 1.0
        xs = np.unique(seg.midpoints[along_y, 0])
        ys = np.unique(seg.midpoints[~along_y, 1])
        expected = sorted(
            (0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1]))
            for i in range(xs.size - 1)
            for j in range(ys.size - 1)
        )
        assert sorted(map(tuple, mesh_centres(electrode))) == expected


def test_ring_footprint_is_the_exact_circle() -> None:
    """Rings keep the exact circle (``wire_polylines`` samples it)."""
    from groundfield.postprocess._footprint import footprint

    ring = gf.RingElectrode(name="r", center=(2.0, 3.0, 0.8), radius=4.0)
    fp = footprint(ring)
    assert fp.segments.shape == (0, 2, 2)
    np.testing.assert_array_equal(fp.circles, [[2.0, 3.0, 4.0]])


# ---------------------------------------------------------------------
# Audit 1, H4 — rotated foundations are several bonded wires
# ---------------------------------------------------------------------

_SIDE = 10.0


def _generated_foundation(angle_deg: float, style: str = "mesh"):
    """10 m x 10 m foundation from the generators (2 x 2 mesh, 0.8 m).

    The generator builds a mesh at every angle, and a ring at any angle
    but 0, from bonded straight wires ``<name>``, ``<name>_w1`` ...
    (10.39, audit A2G-04); an axis-aligned ring is one ``grid_mesh``.
    """
    from groundfield.generators import FoundationElectrodeSpec, GroundingSystemSpec

    world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    GroundingSystemSpec(
        electrodes=[
            FoundationElectrodeSpec(
                size_xy_m=(_SIDE, _SIDE),
                orientation_deg=angle_deg,
                style=style,
                n_x=2,
                n_y=2,
                depth_m=0.8,
            )
        ]
    ).build_at(world, site_xy=(0.0, 0.0), name_prefix="h", rng=np.random.default_rng(0))
    anchor = world.electrodes[0].name
    gf.create_source(world, name="s", attached_to=anchor, magnitude=100.0)
    eng = gf.create_engine(backend="image", segment_length=0.5, frequencies=[50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world, eng.solve(world), anchor


def _rotate(angle_deg: float, lx: float, ly: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return math.cos(a) * lx - math.sin(a) * ly, math.sin(a) * lx + math.cos(a) * ly


def _to_local(angle_deg: float, xy: np.ndarray) -> np.ndarray:
    a = math.radians(-angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.column_stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])


@pytest.mark.finding("10.21")
@pytest.mark.parametrize("side", [(1, 0), (-1, 0), (0, 1), (0, -1)])
def test_touch_voltage_of_a_rotated_foundation_matches_the_unrotated_one(side) -> None:
    """The footprint is every wire of the foundation, not the named edge
    wire: at 30 deg the base put the feet of +y 1 m inside the building
    (93.9 V instead of 167.5 V, -44 %)."""
    w0, r0, a0 = _generated_foundation(0.0)
    w30, r30, a30 = _generated_foundation(30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u0 = abs(touch_voltage(r0, w0, electrode=a0, direction=(*side, 0.0)))
        u30 = abs(
            touch_voltage(
                r30, w30, electrode=a30, direction=(*_rotate(30.0, *side), 0.0)
            )
        )
    # one uniform current per wire vs one per grid: about 1 % (audit 1)
    assert u30 == pytest.approx(u0, rel=0.05)


@pytest.mark.finding("10.21")
def test_envelope_of_a_rotated_foundation_stays_outside_and_samples_its_cells() -> None:
    """31 of 72 foot positions lay inside the foundation and no mesh centre
    was evaluated; now the offset curve surrounds all wires and, with
    ``interior="meshes"``, the four cell centres are sampled, like the mesh
    centres of the 0 deg grid."""
    world, result, anchor = _generated_foundation(30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, feet = touch_voltage_envelope(
            result, world, electrode=anchor, n_angles=72, return_points=True
        )
        _, env, feet_in = touch_voltage_envelope(
            result,
            world,
            electrode=anchor,
            n_angles=72,
            interior="meshes",
            return_points=True,
        )
    local = _to_local(30.0, feet[:, :2])
    assert np.all(np.max(np.abs(local), axis=1) > _SIDE / 2), "foot inside"
    assert feet_in.shape == (72 + 4, 3)
    centres = sorted(map(tuple, np.round(_to_local(30.0, feet_in[72:, :2]), 9)))
    assert centres == [(-2.5, -2.5), (-2.5, 2.5), (2.5, -2.5), (2.5, 2.5)]
    w0, r0, a0 = _generated_foundation(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, env0 = touch_voltage_envelope(
            r0, w0, electrode=a0, n_angles=72, interior="meshes"
        )
    assert np.abs(env[72:]).max() == pytest.approx(np.abs(env0[72:]).max(), rel=0.05)


@pytest.mark.finding("10.21")
def test_any_wire_name_of_a_rotated_foundation_selects_the_whole_foundation() -> None:
    world, result, anchor = _generated_foundation(30.0, style="ring")
    names = [e.name for e in world.electrodes]
    assert names == [anchor, f"{anchor}_w1", f"{anchor}_w2", f"{anchor}_w3"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = touch_voltage_envelope(
            result, world, electrode=anchor, interior="meshes", return_points=True
        )
        for name in names[1:]:
            got = touch_voltage_envelope(
                result, world, electrode=name, interior="meshes", return_points=True
            )
            np.testing.assert_allclose(got[2], ref[2], atol=1e-12)
    # A ring-style foundation is one cell: its centre is sampled.
    np.testing.assert_allclose(ref[2][-1, :2], [0.0, 0.0], atol=1e-9)


def test_wire_suffix_without_a_bonded_root_is_an_ordinary_electrode() -> None:
    """``x_w1`` is grouped only with an electrode ``x`` of its cluster."""
    from groundfield.postprocess._footprint import foundation_wires

    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "strip", name="x_w1", start=(0, 0, 0.5), end=(5, 0, 0.5))
    gf.create_electrode(w, "strip", name="y", start=(0, 9, 0.5), end=(5, 9, 0.5))
    gf.create_electrode(w, "strip", name="y_w1", start=(0, 20, 0.5), end=(5, 20, 0.5))
    assert [e.name for e in foundation_wires(w, "x_w1")] == ["x_w1"]
    clusters = {"y": ["y"], "y_w1": ["y_w1"]}  # not bonded
    assert [e.name for e in foundation_wires(w, "y", clusters)] == ["y"]
    assert [e.name for e in foundation_wires(w, "y")] == ["y", "y_w1"]
    # A grid bonded to a rod that happens to be called ``g_w1`` stays a grid.
    gf.create_electrode(
        w, "grid_mesh", name="g", corner=(30, 0, 0.5), size=(10, 10), n_x=2, n_y=2
    )
    gf.create_electrode(w, "rod", name="g_w1", position=(45.0, 0.0, 0.5), length=3.0)
    assert [e.name for e in foundation_wires(w, "g")] == ["g"]
    assert [e.name for e in foundation_wires(w, "g_w1")] == ["g_w1"]


@pytest.mark.finding("10.22")
def test_the_quoted_corner_deviations_are_those_of_the_default_kernel() -> None:
    """The warning quotes what the default (line) kernel gives at a grid corner.

    20 m x 20 m grid, 4 x 4 meshes, at 0.5 m, 100 Ohm m, ds = 0.5 m, against
    the per-segment ``mom``: touch with the feet 1 m outside the corner
    +6.4 %, step from 1 m to 2 m outside -7.6 %, step from the point above
    the corner 1 m outwards -11.4 % with one owner per wire line (user
    decision Q30; one owner for the grid gave +20.2 / -20.8 / -27.9 %; the
    text used to quote the point kernel's +19 % / -21 % and the engine page
    -28 % without saying which step each was; audit round 3, note of
    nbrefresh).
    """
    # audit round 3 (coordinator item): numbers of UniformCurrentFieldWarning
    w = _grid_world()
    d = (-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0)
    values = {}
    for backend in ("image", "mom"):
        res = _solve(w, backend=backend, ds=0.5)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            values[backend] = (
                touch_voltage(res, w, electrode="g1", direction=d).real,
                step_voltage(res, position=(d[0], d[1], 0.0), direction=d).real,
                step_voltage(res, position=(0.0, 0.0, 0.0), direction=d).real,
            )
        if backend == "image":
            text = str(_caveats(record)[0].message)
    touch, step_out, step_above = (
        values["image"][k] / values["mom"][k] - 1.0 for k in range(3)
    )
    assert f"touch 1 m outside {touch * 100:+.0f} %" in text, (touch, text)
    assert f"from 1 m to 2 m outside {step_out * 100:+.0f} %" in text, text
    assert f"1 m outwards {step_above * 100:+.0f} %" in text, text


# audit D3A-03
@pytest.mark.finding("10.21")
def test_connection_point_anchor_of_a_generated_ring_is_its_corner() -> None:
    """``anchor="connection_point"`` reproduced v0.15.0 for generated
    foundations only while they were one grid mesh connected at the centre.
    Since 2026.11 the electrode named after a rectangular foundation is its
    -y side, connected at a corner: the circle of feet goes round that
    corner (10 m ring, 100 A, 100 ohm m: 270.89 V instead of 245.22 V round
    the centre). The texts say so and quote these numbers."""
    from groundfield.generators import FoundationElectrodeSpec, GroundingSystemSpec
    from groundfield.postprocess import plotting

    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    anchor = GroundingSystemSpec(
        electrodes=[FoundationElectrodeSpec(style="ring", size_m=10.0, depth_m=0.8)]
    ).build_at(w, (0.0, 0.0), "h", np.random.default_rng(0))
    gf.create_source(w, attached_to=anchor, magnitude=100.0)
    res = gf.create_engine(frequencies=[50.0], segment_length=0.5).solve(w)
    assert w.get_electrode(anchor).connection_point == (-5.0, -5.0, 0.8)
    _, env = touch_voltage_envelope(res, w, electrode=anchor, anchor="connection_point")
    assert float(np.max(np.abs(env))) == pytest.approx(270.89, abs=0.005)
    for doc in (touch_voltage.__doc__, plotting.plot_potential_radial.__doc__):
        flat = " ".join(doc.split())
        assert "corner" in flat and "Q38" in flat
    flat = " ".join(touch_voltage.__doc__.split())
    for quoted in (
        "270.89 V",
        "245.22 V",
        "229.71 against 161.76 V",
        "91.10 against 111.38 V",
        "134.24 against 120.12 V",
        "240.82, 156.07 and 109.87 V",
    ):
        assert quoted in flat
    # audit R4B-09: the centre values are this release's circle round the
    # centre of the four strips, not the grid mesh of 77c58bd (159.61 V)
    assert "159.61" not in flat
