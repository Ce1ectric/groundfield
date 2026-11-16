"""Corners of the offset curve in the touch-voltage envelope.

The envelope walks ``n_angles`` equally spaced directions from the centroid
of the wires. They hit the rounded corners of the curve 1 m outside the
wires only by chance, and the touch voltage is largest there. The corners of
a 60 m × 40 m grid lie at 33.7° from its centre, between the 30° and 45°
rays of the default 24 directions. Case study E03 found that envelope 12 …
28 % low in two-layer soil, on the unsafe side; dense rays (360) and a touch
point at the corner post agreed with each other. Square grids were
unaffected, because their diagonals lie on multiples of 15°.

Since the final documentation pass of 2026.11.0 the envelope also stands on
the arc around every corner of the convex hull of the wires
(``corners=True``, five feet per corner). ``corners=False`` keeps the
equally spaced directions alone.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.postprocess._footprint import Footprint, footprint
from groundfield.postprocess.safety import touch_voltage_envelope


def _solve(world: gf.World, backend: str = "image", ds: float = 1.0):
    eng = gf.create_engine(backend=backend, segment_length=ds, frequencies=[50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eng.solve(world)


def _grid(lx: float, ly: float, mesh: float = 5.0, soil=None) -> gf.World:
    w = gf.create_world(soil=soil or gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(
        w,
        "grid_mesh",
        name="grid",
        corner=(0.0, 0.0, 0.8),
        size=(lx, ly),
        n_x=int(lx / mesh),
        n_y=int(ly / mesh),
        wire_radius=0.006,
    )
    gf.create_source(w, attached_to="grid", magnitude=2000.0)
    return w


def _max_touch(res, world, name, **kwargs) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, u = touch_voltage_envelope(res, world, electrode=name, **kwargs)
    return float(np.abs(u).max())


@pytest.mark.parametrize("backend", ["image", "mom"])
def test_the_default_envelope_reaches_the_corners_of_a_rectangular_grid(
    backend: str,
) -> None:
    """60 m × 40 m, 5 m meshes, 100 Ω·m, 2 kA: the default envelope equals the
    maximum over 3600 directions. The 24 directions alone read 20 … 24 % low
    (image 485.1 V against 611.1 V, mom 417.6 V against 544.5 V)."""
    w = _grid(60.0, 40.0)
    res = _solve(w, backend)
    dense = _max_touch(res, w, "grid", n_angles=3600, corners=False)
    default = _max_touch(res, w, "grid")
    rays_only = _max_touch(res, w, "grid", corners=False)
    assert default == pytest.approx(dense, rel=2e-4)
    assert rays_only < 0.85 * dense


def test_a_square_grid_is_unchanged_by_the_corners() -> None:
    """The diagonals of a square lie on the 45° rays: the 24 directions already
    reach its corners, and the corner feet add nothing larger there."""
    w = _grid(40.0, 40.0)
    res = _solve(w)
    assert _max_touch(res, w, "grid") == pytest.approx(
        _max_touch(res, w, "grid", corners=False), rel=1e-9
    )


@pytest.mark.parametrize(
    "electrode",
    [
        gf.StripElectrode(name="e", start=(0.0, 0.0, 0.8), end=(30.0, 0.0, 0.8)),
        gf.RingElectrode(name="e", center=(0.0, 0.0, 0.8), radius=5.0),
        gf.PolylineElectrode(
            name="e",
            vertices=[
                (0.0, 0.0, 0.8),
                (20.0, 0.0, 0.8),
                (20.0, 6.0, 0.8),
                (6.0, 6.0, 0.8),
                (6.0, 20.0, 0.8),
                (0.0, 20.0, 0.8),
            ],
        ),
        gf.RodElectrode(name="e", position=(0.0, 0.0, 0.0), length=3.0),
    ],
    ids=["strip", "ring", "L-outline", "rod"],
)
def test_corners_never_lower_the_envelope_and_keep_the_rays_first(electrode) -> None:
    w = gf.World(
        soil=gf.HomogeneousSoil(rho=100.0),
        electrodes=[electrode],
        sources=[gf.CurrentSource(name="s", attached_to="e", magnitude=100.0)],
    )
    res = _solve(w, ds=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a0, u0 = touch_voltage_envelope(res, w, electrode="e", corners=False)
        a1, u1 = touch_voltage_envelope(res, w, electrode="e")
    assert len(u1) >= len(u0) == 24
    np.testing.assert_array_equal(a1[:24], a0)
    np.testing.assert_allclose(u1[:24], u0, rtol=1e-14, atol=0.0)
    assert np.abs(u1).max() >= np.abs(u0).max()


def test_corner_feet_stand_at_the_distance_on_the_corner_arcs() -> None:
    d = 1.0
    # A 6 m × 4 m rectangle of four wires: 4 corners × 5 feet on arcs of 90°.
    rect = Footprint(
        segments=np.array(
            [
                [[0, 0], [6, 0]],
                [[6, 0], [6, 4]],
                [[6, 4], [0, 4]],
                [[0, 4], [0, 0]],
            ],
            dtype=float,
        ),
        circles=np.empty((0, 3)),
    )
    feet = rect.corner_feet(d)
    assert feet.shape == (20, 2)
    np.testing.assert_allclose(rect.distance(feet), d, rtol=1e-12)
    # Every foot lies outside the rectangle, in the normal cone of a corner.
    outside = (
        (feet[:, 0] <= 0.0)
        | (feet[:, 0] >= 6.0)
        | (feet[:, 1] <= 0.0)
        | (feet[:, 1] >= 4.0)
    )
    assert outside.all()
    # The bisector feet sit diagonally off the corners.
    diag = (np.array([6.0, 4.0]) + d * np.array([1.0, 1.0]) / np.sqrt(2.0))[None, :]
    assert np.min(np.linalg.norm(feet - diag, axis=1)) < 1e-12

    # One straight wire: a half circle of 5 feet around each end.
    strip = footprint(
        gf.StripElectrode(name="s", start=(0.0, 0.0, 0.5), end=(10.0, 0.0, 0.5))
    )
    sf = strip.corner_feet(d)
    assert sf.shape == (10, 2)
    np.testing.assert_allclose(strip.distance(sf), d, rtol=1e-12)
    assert np.all((sf[:, 0] <= 1e-12) | (sf[:, 0] >= 10.0 - 1e-12))

    # One rod and one ring: no corners.
    rod = footprint(gf.RodElectrode(name="r", position=(1.0, 2.0, 0.0), length=3.0))
    ring = footprint(gf.RingElectrode(name="g", center=(0.0, 0.0, 0.5), radius=3.0))
    assert rod.corner_feet(d).shape == (0, 2)
    assert ring.corner_feet(d).shape == (0, 2)


def test_corner_feet_of_a_non_convex_outline_keep_their_distance() -> None:
    """L-shaped outline: only the five convex hull vertices get feet, and no
    foot comes closer than the distance to any wire."""
    ell = footprint(
        gf.PolylineElectrode(
            name="ell",
            vertices=[
                (0.0, 0.0, 0.8),
                (20.0, 0.0, 0.8),
                (20.0, 6.0, 0.8),
                (6.0, 6.0, 0.8),
                (6.0, 20.0, 0.8),
                (0.0, 20.0, 0.8),
            ],
        )
    )
    feet = ell.corner_feet(1.0)
    assert feet.shape == (5 * 5, 2)
    assert np.all(ell.distance(feet) >= 1.0 - 1e-12)


def test_connection_point_anchor_ignores_corners() -> None:
    w = _grid(20.0, 10.0)
    res = _solve(w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, _ = touch_voltage_envelope(
            res, w, electrode="grid", anchor="connection_point"
        )
    assert a.shape == (24,)


@pytest.mark.parametrize("offset", [(0.0, 0.0), (5.0e5, 5.8e6), (3.25e7, 5.8e6)])
@pytest.mark.parametrize("size", [(20.0, 1.0), (10.0, 3.0), (30.0, 20.0)])
def test_corner_feet_do_not_depend_on_projected_coordinates(offset, size) -> None:
    """At an easting of 5e5 m (UTM) or 3.25e7 m (zone-prefixed), a hull
    tolerance scaled by the absolute coordinates merged a 20 m × 1 m outline
    into one straight wire: two corners lost their arc and feet came within
    0.05 m of the wires (audit W5E-01). The hull is built relative to the
    footprint."""
    lx, ly = size
    x0, y0 = offset
    corners = np.array([[0, 0], [lx, 0], [lx, ly], [0, ly]], dtype=float) + [x0, y0]
    fp = Footprint(
        segments=np.stack([corners, np.roll(corners, -1, axis=0)], axis=1),
        circles=np.empty((0, 3)),
    )
    feet = fp.corner_feet(1.0)
    assert feet.shape == (20, 2)
    np.testing.assert_allclose(fp.distance(feet), 1.0, rtol=1e-6)
    local = Footprint(segments=fp.segments - [x0, y0], circles=fp.circles)
    np.testing.assert_allclose(feet - [x0, y0], local.corner_feet(1.0), atol=1e-6)
