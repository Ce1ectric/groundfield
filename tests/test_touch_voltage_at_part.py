"""Touch voltage at user-defined touch points (10.21; user decision Q3).

EN 50522 defines the touch voltage between an accessible conductive part
and the feet of a person standing 1 m away from it. The electrode-anchored
helpers (``touch_voltage``, ``touch_voltage_envelope``) put the feet 1 m
from the buried wires, a proxy when the touched parts are not known. Where
they are known — the door of a station housing, a mast, a fence post, a
house connection box bonded to the PEN — ``touch_voltage_at_part`` puts the
feet on the circle of radius ``distance`` around the part, optionally only
in given directions or outside the outlines where nobody can stand (the
housing or building itself), and takes the potential of the electrode or
junction the part is bonded to. A part bonded to a PEN junction far from
the faulted earth carries a transferred potential (U_T,V).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.geometry.electrodes import GridMeshElectrode
from groundfield.postprocess.safety import (
    UniformCurrentFieldWarning,
    touch_voltage,
    touch_voltage_at_part,
)

RHO = 100.0  # Ohm m

#: Station housing 3 m x 2.5 m centred on the origin, (x, y) in m.
HOUSING = [(-1.5, -1.25), (1.5, -1.25), (1.5, 1.25), (-1.5, 1.25)]


def _solve(world: gf.World, backend: str = "image", ds: float = 0.25, freqs=(50.0,)):
    eng = gf.create_engine(backend=backend, segment_length=ds, frequencies=list(freqs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eng.solve(world)


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UniformCurrentFieldWarning)
        return func(*args, **kwargs)


def _rod_world(current: float = 10.0) -> gf.World:
    """3 m rod with its head 0.5 m deep at the origin."""
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=RHO))
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="rod", magnitude=current)
    return w


def _rod_surface_potential(r: float, current: float) -> float:
    """Surface potential in V of the rod of :func:`_rod_world`, independently.

    A uniform line current I / L from depth t = 0.5 m to t + L = 3.5 m and
    its image above the surface give, on the surface at the horizontal
    distance r,

        phi(r) = rho I / (2 pi L) [asinh((t + L) / r) - asinh(t / r)].

    The image backend carries one uniform current on the rod and the line
    kernel integrates it exactly, so the result's potential agrees to the
    quadrature accuracy (3e-7 at 1 m).
    """
    top, length = 0.5, 3.0
    return (
        RHO
        * current
        / (2.0 * math.pi * length)
        * (math.asinh((top + length) / r) - math.asinh(top / r))
    )


def _pen_world(*, source_at: str = "station", inductive: bool = False) -> gf.World:
    """Station rod, a PEN cable 40 m long to a house connection junction."""
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=RHO))
    gf.create_electrode(w, "rod", name="station", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_junction(w, name="house_connection", position=(40.0, 0.0, 0.7))
    gf.create_conductor(
        w,
        name="pen",
        start="station",
        end="house_connection",
        resistivity=2.83e-8,
        cross_section=150e-6,
        inductance_model="neumann" if inductive else None,
    )
    gf.create_source(w, attached_to=source_at, magnitude=100.0)
    return w


def _grid_world() -> gf.World:
    """20 m x 20 m grid with 4 x 4 meshes at 0.5 m, 100 A (as in 10.21)."""
    return gf.World(
        soil=gf.HomogeneousSoil(rho=RHO),
        electrodes=[
            GridMeshElectrode(
                name="g1", corner=(0.0, 0.0, 0.5), size=(20.0, 20.0), n_x=4, n_y=4
            )
        ],
        sources=[gf.CurrentSource(name="s", attached_to="g1", magnitude=100.0)],
    )


# ---------------------------------------------------------------------
# agreement with touch_voltage and with independent evaluations
# ---------------------------------------------------------------------


@pytest.mark.finding("10.21")
def test_part_on_a_wire_matches_touch_voltage_on_the_offset_curve() -> None:
    """A part on the wire, feet away from the electrode: the foot position of
    ``touch_voltage`` (1 m from the nearest wire) and the same value."""
    # user decision Q3
    w = _grid_world()
    res = _solve(w, ds=1.0)
    u_ref = _quiet(touch_voltage, res, w, electrode="g1")  # feet at (21, 10)
    _, u, feet = _quiet(
        touch_voltage_at_part,
        res,
        w,
        position=(20.0, 10.0),
        bonded_to="g1",
        directions=[(1.0, 0.0)],
        return_points=True,
    )
    np.testing.assert_allclose(feet, [[21.0, 10.0, 0.0]], atol=1e-12)
    assert u[0] == pytest.approx(u_ref, rel=1e-12)
    # On the full circle the first foot (azimuth 0) is the same spot.
    angles, circle = _quiet(
        touch_voltage_at_part, res, w, position=(20.0, 10.0), bonded_to="g1"
    )
    assert angles.shape == circle.shape == (24,)
    assert circle[0] == pytest.approx(u_ref, rel=1e-12)

    ring = gf.create_world(soil=gf.HomogeneousSoil(rho=RHO))
    gf.create_electrode(ring, "ring", name="r", center=(2.0, 3.0, 0.8), radius=4.0)
    gf.create_source(ring, attached_to="r", magnitude=10.0)
    res_ring = _solve(ring, ds=0.5)
    a = math.radians(40.0)
    e = (math.cos(a), math.sin(a))
    u_ring = touch_voltage(res_ring, ring, electrode="r", direction=(*e, 0.0))
    _, u_part = touch_voltage_at_part(
        res_ring,
        ring,
        position=(2.0 + 4.0 * e[0], 3.0 + 4.0 * e[1]),
        bonded_to="r",
        directions=[e],
    )
    assert u_part[0] == pytest.approx(u_ring, rel=1e-9)


@pytest.mark.finding("10.21")
@pytest.mark.parametrize("distance", [0.5, 1.0, 2.5])
def test_feet_stand_on_the_circle_around_the_part(distance: float) -> None:
    """Feet at ``position + distance (cos t_k, sin t_k)``, t_k = 2 pi k / n,
    on the surface; U_T = U_part - phi(feet) at exactly these points."""
    # user decision Q3
    w = _rod_world()
    res = _solve(w)
    angles, u, feet = touch_voltage_at_part(
        res,
        w,
        position=(4.0, -1.0, -2.5),  # a part above ground: z is ignored
        bonded_to="rod",
        distance=distance,
        n_angles=12,
        surface_z=0.0,
        return_points=True,
    )
    theta = 2.0 * np.pi * np.arange(12) / 12
    expected = np.column_stack(
        [4.0 + distance * np.cos(theta), -1.0 + distance * np.sin(theta), np.zeros(12)]
    )
    np.testing.assert_allclose(angles, theta, rtol=0, atol=1e-15)
    np.testing.assert_allclose(feet, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.hypot(feet[:, 0] - 4.0, feet[:, 1] + 1.0), distance, rtol=1e-12
    )
    u_e = complex(res.electrode_potentials["rod"][0])
    np.testing.assert_allclose(u, u_e - res.potential(expected), rtol=1e-12)


@pytest.mark.finding("10.21")
def test_part_at_a_rod_head_against_the_analytic_surface_potential() -> None:
    """A part above the rod head (a mast foot): every foot 1 m away has the
    closed-form surface potential of the rod, so U_T is the same all round."""
    # user decision Q3
    current = 10.0
    w = _rod_world(current)
    res = _solve(w)
    _, u = touch_voltage_at_part(res, w, position=(0.0, 0.0), bonded_to="rod")
    u_e = complex(res.electrode_potentials["rod"][0])
    expected = u_e - _rod_surface_potential(1.0, current)
    np.testing.assert_allclose(u, expected, rtol=1e-6)
    assert np.ptp(u.real) <= 1e-9 * abs(u[0])


@pytest.mark.finding("10.21")
def test_part_bonded_to_a_pen_junction_carries_the_transferred_potential() -> None:
    """Transferred potential (U_T,V): a house connection box on the west wall
    of a house 40 m from the station, bonded to the PEN that ends on a
    junction there; the fault current flows into the station earth.

    No current flows on the PEN, so the box carries the station potential;
    the feet stand outside the house on the low surface potential far from
    the station. Independent check: the closed-form surface potential of
    the station rod at every foot.
    """
    # user decision Q3
    w = _pen_world()
    res = _solve(w)
    house = [(40.0, -4.0), (50.0, -4.0), (50.0, 4.0), (40.0, 4.0)]
    angles, u, feet = touch_voltage_at_part(
        res,
        w,
        position=(40.0, 0.0),
        bonded_to="house_connection",
        exclude=[house],
        return_points=True,
    )
    u_e = complex(res.electrode_potentials["station"][0])
    u_j = complex(res.metadata["junction_potentials"]["house_connection"][0])
    assert u_j == pytest.approx(u_e, rel=1e-12)
    # 13 of 24 feet: 90 deg to 270 deg, the two on the wall included
    np.testing.assert_allclose(np.degrees(angles), 90.0 + 15.0 * np.arange(13))
    assert np.all(feet[:, 0] <= 40.0 + 1e-12)
    r = np.hypot(feet[:, 0], feet[:, 1])
    expected = u_e - np.array([_rod_surface_potential(x, 100.0) for x in r])
    np.testing.assert_allclose(u, expected, rtol=1e-6)
    # Most of the earth potential rise is transferred to the house.
    assert np.abs(u).min() > 0.98 * abs(u_e)


@pytest.mark.finding("10.21")
def test_a_junction_part_carries_the_junction_potential() -> None:
    """With the source at the junction, the PEN current makes the junction
    potential exceed the station's by R I; the part bonded to the junction
    sees that difference, the part bonded to the station does not."""
    # user decision Q3
    w = _pen_world(source_at="house_connection")
    res = _solve(w)
    kwargs = {"position": (40.0, 0.0), "directions": [(-1.0, 0.0)]}
    _, at_junction = touch_voltage_at_part(
        res, w, bonded_to="house_connection", **kwargs
    )
    _, at_station = touch_voltage_at_part(res, w, bonded_to="station", **kwargs)
    r_i = w.get_conductor("pen").series_resistance * 100.0
    assert at_junction[0] - at_station[0] == pytest.approx(r_i, rel=1e-9)


@pytest.mark.finding("10.21")
def test_frequency_index_selects_the_potentials_of_that_frequency() -> None:
    """Inductive PEN, source at the junction: the junction potential
    depends on the frequency; every index uses its own potentials."""
    # user decision Q3
    w = _pen_world(source_at="house_connection", inductive=True)
    res = _solve(w, freqs=(50.0, 1000.0))
    u_j = [complex(v) for v in res.metadata["junction_potentials"]["house_connection"]]
    assert abs(u_j[1] - u_j[0]) > 1e-3 * abs(u_j[0])
    for k in range(2):
        _, u, feet = touch_voltage_at_part(
            res,
            w,
            position=(40.0, 0.0),
            bonded_to="house_connection",
            n_angles=6,
            frequency_index=k,
            return_points=True,
        )
        np.testing.assert_allclose(
            u, u_j[k] - res.potential(feet, frequency_index=k), rtol=1e-12
        )


# ---------------------------------------------------------------------
# where the feet may stand
# ---------------------------------------------------------------------


@pytest.mark.finding("10.21")
def test_directions_select_the_foot_positions() -> None:
    # user decision Q3
    w = _rod_world()
    res = _solve(w)
    angles, u, feet = touch_voltage_at_part(
        res,
        w,
        position=(5.0, 0.0),
        bonded_to="rod",
        distance=1.5,
        directions=[(0.0, 2.0), (-1.0, -1.0, 0.7)],  # 3-vectors: z ignored
        surface_z=0.0,
        return_points=True,
    )
    s = 1.5 / math.sqrt(2.0)
    np.testing.assert_allclose(angles, [0.5 * math.pi, 1.25 * math.pi], atol=1e-15)
    np.testing.assert_allclose(
        feet, [[5.0, 1.5, 0.0], [5.0 - s, -s, 0.0]], rtol=0, atol=1e-12
    )
    u_e = complex(res.electrode_potentials["rod"][0])
    np.testing.assert_allclose(u, u_e - res.potential(feet), rtol=1e-12)


@pytest.mark.finding("10.21")
def test_outlines_remove_the_feet_where_nobody_can_stand() -> None:
    """The door in the middle of the south wall of a station housing: the
    feet inside the housing (azimuth 15 ... 165 deg) are dropped, the two on
    the wall line (0 and 180 deg) are kept."""
    # user decision Q3
    w = _rod_world()
    res = _solve(w)
    door = (0.0, -1.25)
    angles, u, feet = touch_voltage_at_part(
        res, w, position=door, bonded_to="rod", exclude=[HOUSING], return_points=True
    )
    kept = [0, *range(12, 24)]
    np.testing.assert_allclose(np.degrees(angles), [15.0 * k for k in kept])
    _, u_all, feet_all = touch_voltage_at_part(
        res, w, position=door, bonded_to="rod", return_points=True
    )
    np.testing.assert_array_equal(u, u_all[kept])
    np.testing.assert_array_equal(feet, feet_all[kept])
    # Several outlines, 3-D vertices: a second building east of the housing
    # removes the foot at 0 deg as well.
    shed = [(0.9, -1.5, 0.0), (3.0, -1.5, 0.0), (3.0, -1.0, 0.0), (0.9, -1.0, 0.0)]
    angles2, _ = touch_voltage_at_part(
        res, w, position=door, bonded_to="rod", exclude=[HOUSING, shed]
    )
    np.testing.assert_allclose(np.degrees(angles2), [15.0 * k for k in kept[1:]])


@pytest.mark.finding("10.21")
def test_no_accessible_foot_position_is_an_error() -> None:
    # user decision Q3
    w = _rod_world()
    res = _solve(w)
    big = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]
    with pytest.raises(ValueError, match="exclude"):
        touch_voltage_at_part(
            res, w, position=(0.0, 0.0), bonded_to="rod", exclude=[big]
        )


@pytest.mark.finding("10.21")
def test_touch_voltage_at_part_validates_its_input() -> None:
    # user decision Q3
    w = _rod_world()
    res = _solve(w)
    ok = {"position": (1.0, 1.0), "bonded_to": "rod"}
    with pytest.raises(KeyError, match="'rod'"):  # names the known nodes
        touch_voltage_at_part(res, w, position=(1.0, 1.0), bonded_to="missing")
    for distance in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="distance"):
            touch_voltage_at_part(res, w, distance=distance, **ok)
    with pytest.raises(ValueError, match="n_angles"):
        touch_voltage_at_part(res, w, n_angles=2, **ok)
    with pytest.raises(ValueError, match="n_angles"):
        touch_voltage_at_part(res, w, n_angles=8, directions=[(1.0, 0.0)], **ok)
    with pytest.raises(ValueError, match="directions"):
        touch_voltage_at_part(res, w, directions=[], **ok)
    with pytest.raises(ValueError, match="horizontal"):
        touch_voltage_at_part(res, w, directions=[(0.0, 0.0, 1.0)], **ok)
    with pytest.raises(ValueError, match="directions"):
        touch_voltage_at_part(res, w, directions=[(1.0,)], **ok)
    for position in ((1.0,), (1.0, 2.0, 3.0, 4.0), (math.nan, 0.0)):
        with pytest.raises(ValueError, match="position"):
            touch_voltage_at_part(res, w, position=position, bonded_to="rod")
    for outline in (
        [(0.0, 0.0), (1.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0), (1.0, math.inf)],
    ):
        with pytest.raises(ValueError, match="exclude"):
            touch_voltage_at_part(res, w, exclude=[outline], **ok)
    with pytest.raises(IndexError, match="frequency_index"):
        touch_voltage_at_part(res, w, frequency_index=3, **ok)


@pytest.mark.finding("10.22")
def test_uniform_current_caveat_near_a_grid() -> None:
    """The image family warns once per call near a grid, with or without the
    world; not for a rod."""
    # user decision Q3
    w = _grid_world()
    res = _solve(w, ds=1.0)
    for world in (w, None):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            touch_voltage_at_part(res, world, position=(20.0, 10.0), bonded_to="g1")
        caveats = [
            r for r in record if issubclass(r.category, UniformCurrentFieldWarning)
        ]
        assert len(caveats) == 1
        assert "touch_voltage_at_part" in str(caveats[0].message)
    rod = _rod_world()
    res_rod = _solve(rod)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        touch_voltage_at_part(res_rod, rod, position=(0.0, 0.0), bonded_to="rod")
    assert not [r for r in record if issubclass(r.category, UniformCurrentFieldWarning)]


@pytest.mark.finding("10.21")
def test_touch_voltage_at_part_is_exported_with_the_other_safety_helpers() -> None:
    # user decision Q3
    import groundfield.postprocess as pp
    import groundfield.postprocess.safety as safety

    assert gf.touch_voltage_at_part is gf.safety.touch_voltage_at_part
    assert gf.touch_voltage_at_part is touch_voltage_at_part
    for module in (gf, gf.safety, pp, safety):
        assert "touch_voltage_at_part" in module.__all__, module.__name__
