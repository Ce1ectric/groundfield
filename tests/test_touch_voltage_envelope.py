"""Interior of closed loops in the touch-voltage envelope (review pass 10, 10.21).

The envelope walks the curve 1 m outside the electrode's wires. Inside a
closed loop the surface potential sags, so the touch voltage there can
exceed every value on that curve: a 20 m x 20 m perimeter loop at 0.5 m
depth (100 Ohm m, 100 A) has at most 141 V on the offset curve and 158 V
in its centre. This is the
mesh voltage of IEEE Std 80 (the worst touch voltage inside a mesh of a
grid, near the mesh centre). ``touch_voltage_envelope(..., interior=...)``
therefore samples the centre of every closed mesh:

* ``interior=None`` (default): the mesh centres of ``mesh`` and
  ``grid_mesh`` electrodes and the centre of a ``ring`` (loops in open
  ground); closed ``polyline`` electrodes, the building-outline
  foundations of the generators, are not sampled (inside lies the
  building);
* ``interior=True``: also the inside of closed polylines;
* ``interior=False``: the offset curve only.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.geometry.electrodes import GridMeshElectrode
from groundfield.postprocess.safety import touch_voltage_envelope


def _solve(world: gf.World, ds: float = 0.5):
    eng = gf.create_engine(backend="image", segment_length=ds, frequencies=[50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eng.solve(world)


def _world(electrode, current: float = 100.0) -> gf.World:
    return gf.World(
        soil=gf.HomogeneousSoil(rho=100.0),
        electrodes=[electrode],
        sources=[
            gf.CurrentSource(name="s", attached_to=electrode.name, magnitude=current)
        ],
    )


def _envelope(res, world, name, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return touch_voltage_envelope(
            res, world, electrode=name, n_angles=24, return_points=True, **kwargs
        )


def _u_t(res, name: str, xy) -> complex:
    phi = res.potential(np.array([[xy[0], xy[1], 0.0]]))[0]
    return complex(res.electrode_potentials[name][0]) - complex(phi)


def _square_outline(size: float = 20.0) -> gf.PolylineElectrode:
    return gf.PolylineElectrode(
        name="outline",
        vertices=[
            (0.0, 0.0, 0.8),
            (size, 0.0, 0.8),
            (size, size, 0.8),
            (0.0, size, 0.8),
        ],
    )


@pytest.mark.finding("10.21")
def test_ring_envelope_samples_the_ring_centre_by_default() -> None:
    ring = gf.RingElectrode(name="r", center=(2.0, 3.0, 0.8), radius=4.0)
    w = _world(ring)
    res = _solve(w)
    angles, env, feet = _envelope(res, w, "r")
    assert feet.shape == (25, 3)
    np.testing.assert_allclose(feet[24], [2.0, 3.0, 0.0], atol=1e-12)
    assert angles[24] == 0.0
    assert env[24] == pytest.approx(_u_t(res, "r", (2.0, 3.0)), rel=1e-12)
    perimeter_only = _envelope(res, w, "r", interior=False)
    assert perimeter_only[2].shape == (24, 3)


@pytest.mark.finding("10.21")
def test_closed_polyline_interior_is_sampled_only_on_request() -> None:
    """A closed polyline is the generators' building-outline foundation:
    its inside is not sampled by default; ``interior=True`` adds the centre,
    which for a 20 m loop exceeds every value 1 m outside."""
    w = _world(_square_outline())
    res = _solve(w)
    _, env, feet = _envelope(res, w, "outline")
    assert feet.shape == (24, 3)
    _, env_in, feet_in = _envelope(res, w, "outline", interior=True)
    assert feet_in.shape == (25, 3)
    np.testing.assert_allclose(feet_in[24, :2], [10.0, 10.0], atol=1e-9)
    assert abs(env_in[24]) > 1.05 * np.abs(env).max()


@pytest.mark.finding("10.21")
def test_coarse_grid_and_closed_loop_agree_on_the_centre() -> None:
    """A 1 x 1 grid and a closed square loop of the same size give the same
    interior point (one mesh, one centre)."""
    grid = GridMeshElectrode(
        name="g", corner=(0.0, 0.0, 0.8), size=(20.0, 20.0), n_x=1, n_y=1
    )
    wg = _world(grid)
    _, _, feet_g = _envelope(_solve(wg), wg, "g")
    wl = _world(_square_outline())
    _, _, feet_l = _envelope(_solve(wl), wl, "outline", interior=True)
    np.testing.assert_allclose(feet_g[24:, :2], feet_l[24:, :2], atol=1e-9)


@pytest.mark.finding("10.21")
def test_interior_false_gives_the_offset_curve_only() -> None:
    grid = GridMeshElectrode(
        name="g", corner=(0.0, 0.0, 0.5), size=(20.0, 20.0), n_x=4, n_y=4
    )
    w = _world(grid)
    res = _solve(w, ds=1.0)
    assert _envelope(res, w, "g")[2].shape == (24 + 16, 3)
    assert _envelope(res, w, "g", interior=False)[2].shape == (24, 3)


@pytest.mark.finding("10.21")
def test_interior_point_of_a_non_convex_outline_lies_inside() -> None:
    """L-shaped outline with 6 m wide wings: the area centroid
    (7.1 m, 7.1 m) lies outside; the sampled point is inside, more than
    3 m from every wire (the largest inscribed circle, on the diagonal of
    the corner square, has a radius of 3.5 m; the candidate grid of
    0.625 m finds 3.18 m)."""
    vertices = [(0, 0), (20, 0), (20, 6), (6, 6), (6, 20), (0, 20)]
    ell = gf.PolylineElectrode(
        name="ell", vertices=[(float(x), float(y), 0.8) for x, y in vertices]
    )
    w = _world(ell)
    res = _solve(w)
    _, _, feet = _envelope(res, w, "ell", interior=True)
    assert feet.shape == (25, 3)
    x, y = feet[24, :2]
    inside_wing = (0 < x < 20 and 0 < y < 6) or (0 < x < 6 and 0 < y < 20)
    assert inside_wing, (x, y)
    edges = list(zip(vertices, vertices[1:] + vertices[:1], strict=True))
    nearest = min(_distance((x, y), a, b) for a, b in edges)
    assert nearest > 3.0


def _distance(p, a, b) -> float:
    p, a, b = (np.asarray(v, dtype=float) for v in (p, a, b))
    ab = b - a
    t = min(1.0, max(0.0, float((p - a) @ ab) / float(ab @ ab)))
    return float(np.linalg.norm(p - (a + t * ab)))


def test_interior_rejects_other_values() -> None:
    ring = gf.RingElectrode(name="r", center=(0.0, 0.0, 0.8), radius=2.0)
    w = _world(ring)
    res = _solve(w)
    with pytest.raises(ValueError, match="interior"):
        touch_voltage_envelope(res, w, electrode="r", interior="all")


def test_open_shapes_and_rods_have_no_interior() -> None:
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(
        w, "star", name="star", center=(30.0, 0.0, 0.6), n_arms=4, arm_length=3.0
    )
    gf.create_electrode(
        w,
        "polyline",
        name="open",
        closed=False,
        vertices=[(60.0, 0.0, 0.8), (70.0, 0.0, 0.8), (70.0, 10.0, 0.8)],
    )
    for name in ("rod", "star", "open"):
        gf.create_source(w, attached_to=name, magnitude=1.0, name=f"s_{name}")
    res = _solve(w)
    for name in ("rod", "star", "open"):
        assert _envelope(res, w, name, interior=True)[2].shape == (24, 3)
    assert math.isfinite(float(np.abs(_envelope(res, w, "rod")[1]).max()))
