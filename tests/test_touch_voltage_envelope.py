"""Interior of closed loops in the touch-voltage envelope (review pass 10, 10.21).

The envelope walks the curve 1 m outside the electrode's wires. Inside a
closed loop the surface potential sags, so the touch voltage there can
exceed every value on that curve: a 20 m x 20 m perimeter loop at 0.5 m
depth (100 Ohm m, 100 A) has at most 141 V on the offset curve and 158 V
in its centre. This is the
mesh voltage of IEEE Std 80 (the worst touch voltage inside a mesh of a
grid, near the mesh centre).

User decision Q3: inside a footprint stands the station, the house, the
steel structure or the mast, so sampling there is not meaningful in
general; the touch points are the accessible metal parts above ground
(:func:`~groundfield.postprocess.safety.touch_voltage_at_part`). The
envelope therefore samples no interior by default and keeps the loops as
an explicit option for open-air grids such as a substation yard:

* ``interior=False`` (default): the offset curve only, every electrode kind;
* ``interior="meshes"``: the mesh centres of ``mesh`` and ``grid_mesh``
  electrodes, of a foundation built from bonded wires, and the centre of a
  ``ring``; closed ``polyline`` electrodes (the building-outline
  foundations of the generators) are not sampled;
* ``interior=True``: also the inside of closed polylines.
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
    # These tests count the equally spaced and the interior points; the
    # corner feet have their own tests (corners=True is the default).
    kwargs.setdefault("corners", False)
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
def test_envelope_samples_no_interior_by_default() -> None:
    """Default: the offset curve only, for every electrode kind.

    The base sampled the centre of a ring and the mesh centres of a grid by
    default (25 and 40 points here).
    """
    # user decision Q3
    ring = gf.RingElectrode(name="r", center=(2.0, 3.0, 0.8), radius=4.0)
    grid = GridMeshElectrode(
        name="g", corner=(30.0, 0.0, 0.5), size=(20.0, 20.0), n_x=4, n_y=4
    )
    outline = gf.PolylineElectrode(
        name="outline",
        vertices=[
            (0.0, 40.0, 0.8),
            (20.0, 40.0, 0.8),
            (20.0, 60.0, 0.8),
            (0.0, 60.0, 0.8),
        ],
    )
    w = gf.World(
        soil=gf.HomogeneousSoil(rho=100.0),
        electrodes=[ring, grid, outline],
        sources=[
            gf.CurrentSource(name=f"s_{e}", attached_to=e, magnitude=1.0)
            for e in ("r", "g", "outline")
        ],
    )
    res = _solve(w, ds=1.0)
    for name in ("r", "g", "outline"):
        default = _envelope(res, w, name)
        explicit = _envelope(res, w, name, interior=False)
        assert default[2].shape == (24, 3), name
        np.testing.assert_array_equal(default[1], explicit[1])


@pytest.mark.finding("10.21")
def test_ring_envelope_samples_the_ring_centre_with_interior_meshes() -> None:
    # user decision Q3: the former default, now an explicit mode
    ring = gf.RingElectrode(name="r", center=(2.0, 3.0, 0.8), radius=4.0)
    w = _world(ring)
    res = _solve(w)
    angles, env, feet = _envelope(res, w, "r", interior="meshes")
    assert feet.shape == (25, 3)
    np.testing.assert_allclose(feet[24], [2.0, 3.0, 0.0], atol=1e-12)
    assert angles[24] == 0.0
    assert env[24] == pytest.approx(_u_t(res, "r", (2.0, 3.0)), rel=1e-12)
    perimeter_only = _envelope(res, w, "r")
    assert perimeter_only[2].shape == (24, 3)
    np.testing.assert_array_equal(perimeter_only[1], env[:24])


@pytest.mark.finding("10.21")
def test_closed_polyline_interior_is_sampled_only_with_interior_true() -> None:
    """A closed polyline is the generators' building-outline foundation:
    its inside is sampled neither by default nor with ``"meshes"``;
    ``interior=True`` adds the centre, which for a 20 m loop exceeds every
    value 1 m outside."""
    w = _world(_square_outline())
    res = _solve(w)
    _, env, feet = _envelope(res, w, "outline")
    assert feet.shape == (24, 3)
    assert _envelope(res, w, "outline", interior="meshes")[2].shape == (24, 3)
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
    _, _, feet_g = _envelope(_solve(wg), wg, "g", interior="meshes")
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
    assert _envelope(res, w, "g", interior="meshes")[2].shape == (24 + 16, 3)
    assert _envelope(res, w, "g", interior=True)[2].shape == (24 + 16, 3)
    assert _envelope(res, w, "g", interior=False)[2].shape == (24, 3)
    assert _envelope(res, w, "g")[2].shape == (24, 3)


@pytest.mark.finding("10.21")
def test_release_note_envelopes_with_the_new_default() -> None:
    """The envelope maxima quoted in the 10.21 release notes (100 Ohm m,
    100 A, image backend, point kernel as in the notes).

    20 m x 20 m grid with 4 x 4 meshes (0.5 m, ds = 1 m): the maximum lies
    on the offset curve 1 m outside a corner, 111.44 V with the one owner
    of the grid that the notes were measured with, 98.72 V with one owner
    per wire line (user decision Q30), with and without the mesh centres,
    so it does not move. Ring of 10 m radius (0.8 m, ds = 0.5 m): 139.37 V
    on the offset curve, the new default; the centre, 182.60 V, only with
    ``interior="meshes"``.
    """
    # user decision Q3
    from groundfield.solver._discretize import _one_owner_per_wire_off

    grid = GridMeshElectrode(
        name="g", corner=(0.0, 0.0, 0.5), size=(20.0, 20.0), n_x=4, n_y=4
    )
    ring = gf.RingElectrode(name="r", center=(0.0, 0.0, 0.8), radius=10.0)
    maxima = {}
    for electrode, ds, one_owner in (
        (grid, 1.0, False),
        (grid, 1.0, True),
        (ring, 0.5, False),
    ):
        w = _world(electrode)
        eng = gf.create_engine(
            backend="image", segment_length=ds, frequencies=[50.0], kernel="point"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if one_owner:
                with _one_owner_per_wire_off():
                    res = eng.solve(w)
            else:
                res = eng.solve(w)
        for mode in (False, "meshes"):
            env = _envelope(res, w, electrode.name, interior=mode)[1]
            maxima[electrode.name, one_owner, mode] = float(np.abs(env).max())
    # user decision Q30
    assert maxima["g", False, False] == pytest.approx(98.72, abs=0.005)
    assert maxima["g", True, False] == pytest.approx(111.44, abs=0.005)
    for one_owner in (False, True):
        assert maxima["g", one_owner, "meshes"] == maxima["g", one_owner, False]
    maxima = {(name, mode): v for (name, _, mode), v in maxima.items() if name == "r"}
    assert maxima["r", False] == pytest.approx(139.37, abs=0.005)
    assert maxima["r", "meshes"] == pytest.approx(182.60, abs=0.005)


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


@pytest.mark.finding("10.21")
def test_interior_rejects_other_values() -> None:
    """Only ``False``, ``"meshes"`` and ``True``; ``None`` (the former
    default) is rejected with a hint instead of silently changing meaning."""
    # user decision Q3
    ring = gf.RingElectrode(name="r", center=(0.0, 0.0, 0.8), radius=2.0)
    w = _world(ring)
    res = _solve(w)
    for value in ("all", "mesh", 1, 0, None):
        with pytest.raises(ValueError, match="interior"):
            touch_voltage_envelope(res, w, electrode="r", interior=value)
    with pytest.raises(ValueError, match="meshes"):
        touch_voltage_envelope(res, w, electrode="r", interior=None)
    # numpy booleans count as booleans
    assert touch_voltage_envelope(res, w, electrode="r", interior=np.True_)[
        0
    ].shape == (25,)


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
        for mode in ("meshes", True):
            assert _envelope(res, w, name, interior=mode)[2].shape == (24, 3)
    assert math.isfinite(float(np.abs(_envelope(res, w, "rod")[1]).max()))


@pytest.mark.finding("10.21")
def test_a_generated_mesh_foundation_is_sampled_in_its_cells_with_meshes() -> None:
    """The generators build a mesh foundation from bonded wires, one per wire line.

    Unlike a closed polyline outline, such a wire group is sampled at the
    centre of every cell with ``interior="meshes"`` (the docstring called it
    a ``grid_mesh``; one electrode per wire line since audit A2G-04); the
    default keeps the offset curve only, because the cells lie under the
    building (user decision Q3).
    """
    # audit round 3 (coordinator item): interior docstring; user decision Q3
    from groundfield.generators import (
        HomogeneousSoilSpec,
        TnNetworkConfig,
        TnNetworkGenerator,
    )
    from groundfield.postprocess._footprint import foundation_wires

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = TnNetworkGenerator(
            TnNetworkConfig(
                soil=HomogeneousSoilSpec(rho=100.0),
                building_counts={"residential": 1},
            ),
            seed=1,
        ).build()
    wires = foundation_wires(net, "residential_0_foundation_0")
    assert len(wires) > 1 and all(e.kind == "polyline" and not e.closed for e in wires)
    names = {e.name for e in wires}
    w = gf.World(
        soil=gf.HomogeneousSoil(rho=100.0),
        electrodes=list(wires),
        conductors=[
            c
            for c in net.conductors
            if c.start_electrode in names and c.end_electrode in names
        ],
        sources=[
            gf.CurrentSource(
                name="s", attached_to="residential_0_foundation_0", magnitude=1.0
            )
        ],
    )
    res = _solve(w)
    _, _, feet = _envelope(res, w, "residential_0_foundation_0", interior="meshes")
    _, _, outside = _envelope(res, w, "residential_0_foundation_0")
    assert outside.shape == (24, 3)
    assert feet.shape[0] > 24
