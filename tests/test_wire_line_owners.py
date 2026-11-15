r"""One uniform current per straight wire in the image family (Q30, Q44).

The image family (``image``, ``image_2layer``, the cross-layer path and
``solve_mutual_*``) solves one uniform leakage current per owner. Up to
2026.11 an owner was a whole electrode, so a ``GridMeshElectrode`` put the
same current density on its outer and inner wire lines although the current
crowds to the outer ones: the 20 m x 20 m grid with 4 x 4 meshes of the
validation page read +3.9 % against the per-segment ``mom`` solution
(+3.3 % with the point kernel ``mom`` uses). Since the user decision Q30
of 2026.11 every grid has one owner per full wire line, a star one per
arm and a polyline one per edge between the vertices given (Q44: building
outlines are polylines, so a rectangle given as an outline equals the same
rectangle given by its size, one owner per side since Q38). Rods, strips
and rings stay one owner. The owners of an electrode are ideally bonded;
every result is reported per electrode.

Reference used throughout: the same wires built as separate
``StripElectrode`` objects bonded by ideal conductors, which the image
family solves as one owner per strip in every release.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.geometry.electrodes import (
    GridMeshElectrode,
    MeshElectrode,
    PolylineElectrode,
    RingElectrode,
    RodElectrode,
    StarElectrode,
    StripElectrode,
)

# audit A2G-04 (second adversarial audit), user decisions Q30 and Q44
pytestmark = pytest.mark.finding("10.22")

RHO = 100.0
DEPTH = 0.5
TWO_LAYER = gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=3.0)


def _engine(backend: str = "image", kernel: str | None = None, ds: float = 1.0):
    kwargs = {} if kernel is None else {"kernel": kernel}
    return gf.create_engine(
        backend=backend, segment_length=ds, frequencies=[50.0], **kwargs
    )


def _solve(world: gf.World, engine=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (engine or _engine()).solve(world)


def _single(kind: str, soil=None, **params) -> gf.World:
    world = gf.create_world(soil=soil or gf.HomogeneousSoil(rho=RHO))
    gf.create_electrode(world, kind, name="e", **params)
    gf.create_source(world, attached_to="e", magnitude=1.0)
    return world


def _bonded_strips(lines, soil=None, radius: float = 0.005, shell: float = 0.0):
    """The wires as separate strips, bonded by ideal conductors."""
    world = gf.create_world(soil=soil or gf.HomogeneousSoil(rho=RHO))
    for k, (a, b) in enumerate(lines):
        gf.create_electrode(
            world,
            "strip",
            name=f"s{k}",
            start=a,
            end=b,
            wire_radius=radius,
            concrete_shell_coefficient_ohm_m=shell,
        )
    for k in range(1, len(lines)):
        world.add_conductor(
            gf.Conductor(
                name=f"bond{k}",
                start=lines[0][0],
                end=lines[k][0],
                start_electrode="s0",
                end_electrode=f"s{k}",
            )
        )
    gf.create_source(world, attached_to="s0", magnitude=1.0)
    return world


def _grid_lines(corner, size, n_x, n_y):
    cx, cy, cz = corner
    dx, dy = size
    ys = np.linspace(cy, cy + dy, n_y + 1)
    xs = np.linspace(cx, cx + dx, n_x + 1)
    return [((cx, y, cz), (cx + dx, y, cz)) for y in ys] + [
        ((x, cy, cz), (x, cy + dy, cz)) for x in xs
    ]


def _z(result, name: str = "e") -> complex:
    return complex(result.cluster_impedance(name)[0])


# ---------------------------------------------------------------------
# Geometry: the wire index of every segment
# ---------------------------------------------------------------------


# user decision Q30
def test_segments_record_their_straight_wire() -> None:
    """Grid: one wire per full line (longitudinal first); star: per arm;
    polyline: per straight run (a vertex where it continues straight on
    starts no new wire, also across the closing edge); rod, strip and
    ring: one wire."""
    grid = GridMeshElectrode(
        name="g", corner=(0, 0, DEPTH), size=(10.0, 6.0), n_x=2, n_y=3
    )
    segs = grid.segments(1.0)
    assert segs.n_wires == (3 + 1) + (2 + 1)
    for w in range(segs.n_wires):
        on = segs.wire_index == w
        d = segs.directions[on]
        # every wire is one straight line through all its segments
        assert np.allclose(d, d[0])
        lengths = segs.lengths[on].sum()
        assert lengths == pytest.approx(10.0 if w < 4 else 6.0, rel=1e-12)
    mesh = MeshElectrode(name="m", corner=(0, 0, DEPTH), size=(4.0, 4.0), spacing=2.0)
    assert mesh.segments(1.0).n_wires == 6
    star = StarElectrode(name="s", center=(0, 0, DEPTH), n_arms=5, arm_length=4.0)
    assert star.segments(1.0).n_wires == 5
    outline = PolylineElectrode(
        name="p",
        vertices=[(0, 0, DEPTH), (5, 0, DEPTH), (10, 0, DEPTH), (10, 4, DEPTH)],
        closed=True,
    )
    segs = outline.segments(1.0)
    assert segs.n_wires == 3
    assert np.all(segs.wire_index[:10] == 0)
    # the vertex in the middle of the closing side: the last and the first
    # edge are one straight wire (wire 0, which keeps the electrode's name)
    wrapped = PolylineElectrode(
        name="q",
        vertices=[
            (5, 0, DEPTH),
            (10, 0, DEPTH),
            (10, 4, DEPTH),
            (0, 4, DEPTH),
            (0, 0, DEPTH),
        ],
        closed=True,
    )
    segs = wrapped.segments(1.0)
    assert segs.n_wires == 4
    assert np.all(segs.wire_index[-5:] == 0) and np.all(segs.wire_index[:5] == 0)
    for single in (
        RodElectrode(name="r", position=(0, 0, 0.5), length=6.0),
        StripElectrode(name="t", start=(0, 0, DEPTH), end=(7, 0, DEPTH)),
        RingElectrode(name="c", center=(0, 0, DEPTH), radius=3.0),
    ):
        assert single.segments(0.5, layer_interfaces=(3.0,)).n_wires == 1


# ---------------------------------------------------------------------
# The split equals the same wires bonded as separate strips
# ---------------------------------------------------------------------

GRID = dict(corner=(0.0, 0.0, DEPTH), size=(20.0, 20.0), n_x=4, n_y=4)


# user decision Q30
@pytest.mark.parametrize("soil", [None, TWO_LAYER], ids=["homogeneous", "two_layer"])
@pytest.mark.parametrize("kernel", ["line", "point"])
def test_grid_is_one_owner_per_wire_line(soil, kernel: str) -> None:
    """A grid solves as its wire lines bonded as separate strips (base
    e6124cd: +2.7 % homogeneous with the line kernel, one owner)."""
    z_grid = _z(_solve(_single("grid_mesh", soil, **GRID), _engine(kernel=kernel)))
    z_ref = _z(
        _solve(_bonded_strips(_grid_lines(**GRID), soil), _engine(kernel=kernel)),
        "s0",
    )
    assert z_grid == pytest.approx(z_ref, rel=1e-10)


# user decision Q30
def test_mesh_electrode_is_one_owner_per_wire_line() -> None:
    corner, size, spacing = (0.0, 0.0, DEPTH), (12.0, 8.0), 4.0
    z_mesh = _z(_solve(_single("mesh", corner=corner, size=size, spacing=spacing)))
    lines = _grid_lines(corner, size, n_x=3, n_y=2)
    z_ref = _z(_solve(_bonded_strips(lines)), "s0")
    assert z_mesh == pytest.approx(z_ref, rel=1e-10)


# user decision Q30
def test_star_is_one_owner_per_arm_and_unchanged_alone() -> None:
    """A star solves as its arms bonded as separate strips. Alone, its arms
    carry equal currents by symmetry, so the split does not change it (the
    +4.3 % against ``mom`` are the crowding along each arm)."""
    star = StarElectrode(name="e", center=(0, 0, DEPTH), n_arms=3, arm_length=8.0)
    z_star = _z(_solve(_single("star", **star.model_dump(exclude={"name", "kind"}))))
    z_ref = _z(_solve(_bonded_strips(star.edges)), "s0")
    assert z_star == pytest.approx(z_ref, rel=1e-10)


# user decision Q30
def test_star_beside_another_electrode_shares_its_current_unevenly() -> None:
    """Next to a bonded rod the arms see an asymmetric field: one owner per
    arm moves the star's current to the arms away from the rod (base
    e6124cd: one current density on all arms)."""
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=RHO))
    gf.create_electrode(
        world, "star", name="e", center=(0, 0, DEPTH), n_arms=4, arm_length=6.0
    )
    gf.create_electrode(world, "rod", name="r", position=(8.0, 0.0, 0.5), length=6.0)
    world.add_conductor(
        gf.Conductor(
            name="b",
            start=(0, 0, DEPTH),
            end=(8, 0, 0.5),
            start_electrode="e",
            end_electrode="r",
        )
    )
    gf.create_source(world, attached_to="e", magnitude=1.0)
    res = _solve(world)
    arm_current = {}
    for ps in res.point_sources:
        if ps.electrode_name != "e":
            continue
        x, y, _ = ps.position
        arm = int(round(math.degrees(math.atan2(y, x)) / 90.0)) % 4
        arm_current[arm] = arm_current.get(arm, 0.0) + ps.current[0].real
    # the arm towards the rod (+x) leaks least, the one away from it most
    assert arm_current[0] < arm_current[1] < arm_current[2]
    assert arm_current[2] / arm_current[0] > 1.05
    lines = StarElectrode(
        name="x", center=(0, 0, DEPTH), n_arms=4, arm_length=6.0
    ).edges
    ref = _bonded_strips(lines)
    gf.create_electrode(ref, "rod", name="r", position=(8.0, 0.0, 0.5), length=6.0)
    ref.add_conductor(
        gf.Conductor(
            name="b",
            start=(0, 0, DEPTH),
            end=(8, 0, 0.5),
            start_electrode="s0",
            end_electrode="r",
        )
    )
    assert _z(res) == pytest.approx(_z(_solve(ref), "s0"), rel=1e-10)


L_SHAPE = [
    (0.0, 0.0, DEPTH),
    (12.0, 0.0, DEPTH),
    (12.0, 5.0, DEPTH),
    (5.0, 5.0, DEPTH),
    (5.0, 9.0, DEPTH),
    (0.0, 9.0, DEPTH),
]


# user decisions Q30, Q44
@pytest.mark.parametrize("closed", [True, False])
@pytest.mark.parametrize("shell", [0.0, 0.8], ids=["bare", "encased"])
def test_polyline_is_one_owner_per_edge(closed: bool, shell: float) -> None:
    poly = PolylineElectrode(name="e", vertices=L_SHAPE, closed=closed)
    world = _single(
        "polyline",
        vertices=L_SHAPE,
        closed=closed,
        concrete_shell_coefficient_ohm_m=shell,
    )
    z_poly = _z(_solve(world))
    z_ref = _z(_solve(_bonded_strips(poly.edges, shell=shell)), "s0")
    assert z_poly == pytest.approx(z_ref, rel=1e-10)


# user decision Q44 (with Q38)
def test_rectangular_outline_equals_the_rectangle_given_by_its_size() -> None:
    """A rectangle given as an outline (a ``PolylineElectrode``) and as a
    ``GridMeshElectrode`` of one mesh are four owners, like the rectangle
    given by its size (four strips since Q38). Base e6124cd: the outline and
    the grid were one owner, 6.8e-6 above the strips (default engine)."""
    verts = [(-5.0, -4.0, 0.7), (5.0, -4.0, 0.7), (5.0, 4.0, 0.7), (-5.0, 4.0, 0.7)]
    z_poly = _z(_solve(_single("polyline", vertices=verts, wire_radius=0.0075)))
    z_grid = _z(
        _solve(
            _single(
                "grid_mesh",
                corner=(-5.0, -4.0, 0.7),
                size=(10.0, 8.0),
                n_x=1,
                n_y=1,
                wire_radius=0.0075,
            )
        )
    )
    sides = [(verts[k], verts[(k + 1) % 4]) for k in range(4)]
    z_strips = _z(_solve(_bonded_strips(sides, radius=0.0075)), "s0")
    assert z_poly == pytest.approx(z_strips, rel=1e-12)
    assert z_grid == pytest.approx(z_strips, rel=1e-12)


# user decision Q30
def test_generated_mesh_foundation_keeps_one_owner_per_wire_line() -> None:
    """The generators build a mesh foundation as one electrode per wire
    line, an open polyline with a vertex at every crossing (audit A2G-04):
    one straight wire, so the solver keeps one owner per line, also at a
    rotation, and the foundation equals its wire lines as bonded strips."""
    from groundfield.generators import FoundationElectrodeSpec, GroundingSystemSpec

    for angle in (None, 30.0):
        spec = FoundationElectrodeSpec(
            style="mesh",
            size_m=10.0,
            depth_m=0.7,
            wire_radius_m=0.005,
            orientation_deg=angle,
        )
        world = gf.World(soil=gf.HomogeneousSoil(rho=RHO))
        anchor = GroundingSystemSpec(electrodes=[spec]).build_at(
            world, (0.0, 0.0), "h", np.random.default_rng(0)
        )
        gf.create_source(world, attached_to=anchor, magnitude=1.0)
        polylines = [e for e in world.electrodes if e.kind == "polyline"]
        assert polylines and all(len(e.vertices) > 2 for e in polylines)
        res = _solve(world)
        assert "wire_owners" not in res.metadata
        lines = [
            (e.vertices[0], e.vertices[-1])
            if e.kind == "polyline"
            else (e.start, e.end)
            for e in world.electrodes
        ]
        first = world.electrodes[0]
        ref = _bonded_strips(
            lines,
            radius=first.wire_radius,
            shell=first.concrete_shell_coefficient_ohm_m,
        )
        # rotated: the vertices at the crossings differ from the strips'
        # midpoints by round-off (1e-15 m), which the near-pair rules of the
        # line kernel turn into 3e-8 of Z
        assert _z(res, anchor) == pytest.approx(_z(_solve(ref), "s0"), rel=1e-7)


# ---------------------------------------------------------------------
# Results stay per electrode
# ---------------------------------------------------------------------


# user decision Q30
def test_results_are_reported_per_electrode() -> None:
    """Current, potential, impedance and point sources refer to the
    electrode; the owners are internal (``metadata["wire_owners"]``)."""
    res = _solve(_single("grid_mesh", **GRID))
    assert list(res.electrode_currents) == ["e"]
    assert res.electrode_currents["e"][0] == pytest.approx(1.0, abs=1e-12)
    assert res.grounding_impedance("e")[0] == pytest.approx(
        res.cluster_impedance("e")[0], rel=1e-14
    )
    assert res.input_impedance()[0] == pytest.approx(
        res.cluster_impedance("e")[0], rel=1e-12
    )
    assert {ps.electrode_name for ps in res.point_sources} == {"e"}
    assert res.metadata["wire_owners"] == {"e": 10}
    # the line currents differ: the outer lines leak more per metre
    dens = {}
    for ps in res.point_sources:
        y = round(ps.position[1], 6)
        if ps.direction is not None and abs(ps.direction[0]) == 1.0:
            dens.setdefault(y, []).append(ps.current[0].real / ps.length)
    assert dens[0.0][0] > dens[10.0][0] * 1.1
    back = gf.FieldResult.model_validate_json(res.model_dump_json())
    assert back.metadata["wire_owners"] == {"e": 10}
    assert complex(back.cluster_impedance("e")[0]) == pytest.approx(
        complex(res.cluster_impedance("e")[0]), rel=1e-15
    )


# user decision Q30
def test_rods_strips_and_rings_keep_one_owner() -> None:
    for kind, params in (
        ("rod", dict(position=(0, 0, 0.5), length=4.0)),
        ("strip", dict(start=(0, 0, DEPTH), end=(9, 0, DEPTH))),
        ("ring", dict(center=(0, 0, DEPTH), radius=3.0)),
    ):
        res = _solve(_single(kind, **params))
        assert "wire_owners" not in res.metadata


# user decision Q30
def test_passive_grid_floats_with_zero_net_current() -> None:
    """A source-free grid next to a driven rod: its lines exchange
    equalising currents (one potential), its net current is exactly 0 and
    its cluster impedance ``nan``."""
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=RHO))
    gf.create_electrode(world, "grid_mesh", name="g", **GRID)
    gf.create_electrode(world, "rod", name="r", position=(25.0, 10.0, 0.5), length=3.0)
    gf.create_source(world, attached_to="r", magnitude=1.0)
    res = _solve(world)
    assert res.electrode_currents["g"][0] == 0j
    assert math.isnan(abs(res.cluster_impedance("g")[0]))
    line_currents = {}
    for ps in res.point_sources:
        if ps.electrode_name == "g":
            key = ps.position[1] if abs(ps.direction[0]) == 1.0 else -1 - ps.position[0]
            line_currents[key] = line_currents.get(key, 0.0) + ps.current[0].real
    assert max(abs(c) for c in line_currents.values()) > 1e-3


# user decision Q30
def test_solve_mutual_matrix_diagonal_matches_the_solve() -> None:
    """``solve_mutual_matrix`` splits the owners like ``Engine.solve``."""
    from groundfield.solver.mutual import solve_mutual_matrix

    world = gf.create_world(soil=TWO_LAYER)
    gf.create_electrode(world, "grid_mesh", name="g", **GRID)
    gf.create_electrode(
        world,
        "polyline",
        name="p",
        vertices=[(30, 0, DEPTH), (40, 0, DEPTH), (40, 6, DEPTH)],
        closed=True,
    )
    engine = _engine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = solve_mutual_matrix(world, engine, ["g", "p"], symmetrize=False)
    for j, name in enumerate(("g", "p")):
        single = world.model_copy(deep=True)
        gf.create_source(single, attached_to=name, magnitude=1.0)
        z_solve = _z(_solve(single, engine), name)
        assert Z[j, j] == pytest.approx(z_solve, rel=1e-9)
    assert Z[0, 1] == pytest.approx(Z[1, 0], rel=1e-9)


# user decision Q30
def test_cross_layer_rod_bonded_to_a_grid() -> None:
    """The cross-layer path (a rod through h_1) splits the grid by wire and
    the rod by layer; it equals the strips-and-rod reference."""

    def add_rod(world, anchor):
        gf.create_electrode(world, "rod", name="r", position=(0, 0, DEPTH), length=6.0)
        world.add_conductor(
            gf.Conductor(
                name="br",
                start=(0, 0, DEPTH),
                end=(0, 0, DEPTH),
                start_electrode=anchor,
                end_electrode="r",
            )
        )

    world = _single("grid_mesh", TWO_LAYER, **GRID)
    add_rod(world, "e")
    res = _solve(world)
    assert res.metadata["layer_owners"] == {"r": 2}
    assert res.metadata["wire_owners"] == {"e": 10}
    ref = _bonded_strips(_grid_lines(**GRID), TWO_LAYER)
    add_rod(ref, "s0")
    assert _z(res) == pytest.approx(_z(_solve(ref), "s0"), rel=1e-10)


# ---------------------------------------------------------------------
# Accuracy against the per-segment ``mom`` solution
# ---------------------------------------------------------------------


# user decision Q30
@pytest.mark.parametrize(
    "kernel, base, bound",
    [("point", 0.0330, 0.0065), ("line", 0.0386, 0.0120)],
)
def test_grid_against_per_segment_mom(kernel: str, base: float, bound: float) -> None:
    """20 m x 20 m grid, 4 x 4 meshes, 0.5 m deep, 100 Ohm m, ds = 0.5 m
    (the validation-page case): base e6124cd +3.30 % (point) / +3.86 %
    (line) above ``mom``; one owner per wire line +0.57 % / +1.12 %."""
    world = _single("grid_mesh", **GRID)
    z_mom = _z(_solve(world, _engine("mom", ds=0.5))).real
    z_img = _z(_solve(world, _engine(kernel=kernel, ds=0.5))).real
    dev = z_img / z_mom - 1.0
    assert 0.0 < dev < bound < base


# ---------------------------------------------------------------------
# Straight runs at projected coordinates (audit W1A-01)
# ---------------------------------------------------------------------

#: A projected (UTM-size) origin with the zone prefix in the easting.
UTM = (32_604_321.123, 5_812_345.678)


# audit W1A-01, user decision Q30
@pytest.mark.parametrize(
    "angle, n, ds",
    [(33.0, 4, 0.5), (33.0, 10, 0.5), (89.999999, 10, 0.5), (33.0, 20, 0.25)],
)
def test_generated_mesh_foundation_at_utm_coordinates(angle, n: int, ds: float) -> None:
    """A rotated generated mesh foundation is one owner per wire line
    wherever it stands: at projected coordinates the round-off of its
    crossing vertices (sines up to 7e-9 between consecutive edges) is no
    bend. The snapshot 6f63deb split a 10 m 10 x 10 mesh at 33 deg into 142
    extra owners and read Z 0.55 % low (0.64 % for 20 x 20)."""
    from groundfield.generators import FoundationElectrodeSpec, GroundingSystemSpec

    spec = FoundationElectrodeSpec(
        style="mesh",
        size_m=10.0,
        depth_m=0.8,
        wire_radius_m=0.005,
        orientation_deg=angle,
        n_x=n,
        n_y=n,
    )
    zs, owners = [], []
    for origin in ((0.0, 0.0), UTM):
        world = gf.World(soil=gf.HomogeneousSoil(rho=RHO))
        anchor = GroundingSystemSpec(electrodes=[spec]).build_at(
            world, origin, "h", np.random.default_rng(0)
        )
        gf.create_source(world, attached_to=anchor, magnitude=1.0)
        res = _solve(world, _engine(ds=ds))
        owners.append(res.metadata.get("wire_owners"))
        zs.append(complex(res.cluster_impedance(anchor)[0]))
    assert owners == [None, None]
    assert zs[1] == pytest.approx(zs[0], rel=1e-6)


# audit W1A-01, user decision Q30
@pytest.mark.parametrize("origin", [(0.0, 0.0), (604_321.123, 5_812_345.678), UTM])
def test_straight_run_tolerance_follows_the_coordinate_magnitude(origin) -> None:
    """Collinear vertices stay one wire at every origin; a vertex 2 um off
    the chord of its neighbours (local coordinates: 1 nm) starts a new one.
    The tolerance is the perpendicular height of a vertex over the chord of
    its neighbours, scaled to the coordinate magnitude (64 units of
    round-off: 0.46 um at the UTM origin, 2.4e-12 m at the local one)."""
    from groundfield.geometry.primitives import _straight_runs

    x0, y0 = origin
    theta = math.radians(33.0)
    u = np.array([math.cos(theta), math.sin(theta), 0.0])
    v = np.array([-math.sin(theta), math.cos(theta), 0.0])
    o = np.array([x0, y0, 0.8])
    straight = [tuple(o + t * u) for t in (0.0, 0.5, 1.0, 1.5, 7.0)]
    edges = list(zip(straight[:-1], straight[1:], strict=True))
    assert _straight_runs(edges, closed=False) == [0, 0, 0, 0]
    height = 2e-6 if max(abs(x0), abs(y0)) > 1e3 else 1e-9
    a, b, c = o, o + 1.0 * u, o + 2.0 * u + 2.0 * height * v
    assert _straight_runs([(a, b), (b, c)], closed=False) == [0, 1]
    # a vertex where the wire turns back is never a straight continuation
    back = [tuple(o), tuple(o + 2.0 * u), tuple(o + 1.0 * u)]
    edges = list(zip(back[:-1], back[1:], strict=True))
    assert _straight_runs(edges, closed=False) == [0, 1]


# audit W1A-07, user decision Q30
def test_a_lone_floating_grid_reports_a_plain_zero_current() -> None:
    """A source-free grid that is a cluster of its own floats with exactly
    zero net current, reported as +0 (the snapshot 6f63deb gave -0-0j, the
    negated empty sum of the other cluster members)."""
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=RHO))
    gf.create_electrode(world, "grid_mesh", name="g", **GRID)
    gf.create_electrode(world, "rod", name="r", position=(25.0, 10.0, 0.5), length=3.0)
    gf.create_source(world, attached_to="r", magnitude=1.0)
    current = complex(_solve(world).electrode_currents["g"][0])
    assert current == 0j
    assert math.copysign(1.0, current.real) == 1.0
    assert math.copysign(1.0, current.imag) == 1.0


# ---------------------------------------------------------------------
# Straight runs grow against the chord of the whole run (audit W1R2-05)
# ---------------------------------------------------------------------


def _square_with_near_duplicate(origin, d: float) -> list[tuple]:
    """Edges of a 10 m square outline with a second vertex ``d`` from the
    corner (10, 0), on the 45 deg bisector."""
    ox, oy = origin
    pts = [(0, 0), (10, 0), (10 + d / 2**0.5, d / 2**0.5), (10, 10), (0, 10)]
    p = [np.array([x + ox, y + oy, -0.8]) for x, y in pts]
    return [(p[k], p[(k + 1) % 5]) for k in range(5)]


# audit W1R2-05, user decision Q30
@pytest.mark.parametrize("d", [1e-9, 1e-7, 3e-7, 1e-6, 1e-3])
def test_a_near_duplicate_corner_vertex_does_not_join_two_sides(d: float) -> None:
    """The four sides of the square are four wires at the local and the
    UTM origin. Base ecff7b6: each vertex was tested against the chord of
    its neighbours only, so at UTM the corner next to a vertex 0.1 / 0.3 um
    away passed as straight and two perpendicular sides became one wire
    (3 wires). Below the UTM tolerance (0.46 um) the tiny edge itself joins
    one of the two sides it touches (4 wires; at 1e-9 m it is below one
    unit of round-off of the coordinates and points along the second side;
    5 wires at local coordinates, where the tolerance is 2.4e-12 m); above
    the tolerance both origins agree. A polyline
    electrode rejects edges shorter than its wire radius, so such a vertex
    cannot occur in a valid electrode."""
    from groundfield.geometry.primitives import _straight_runs

    for origin in ((0.0, 0.0), UTM):
        w = _straight_runs(_square_with_near_duplicate(origin, d), closed=True)
        assert len({w[0], w[2], w[3], w[4]}) == 4
    local = _straight_runs(_square_with_near_duplicate((0.0, 0.0), d), closed=True)
    utm = _straight_runs(_square_with_near_duplicate(UTM, d), closed=True)
    assert local == [0, 1, 2, 3, 4]
    if d >= 1e-6:
        assert utm == local
    else:
        assert utm in ([0, 0, 1, 2, 3], [0, 1, 1, 2, 3])


# audit W1R2-05, user decision Q30
def test_a_finely_sampled_arc_is_not_one_straight_wire_at_utm() -> None:
    """A quarter arc of R = 2 m in 1 mm edges (each vertex 0.25 um off the
    chord of its neighbours): base ecff7b6 one straight wire turning by 90
    deg at UTM; now no wire turns by more than 0.06 deg (two edges, the
    chord of a run bounds every vertex by the tolerance), and at local
    coordinates every edge stays a wire of its own."""
    from groundfield.geometry.primitives import _straight_runs

    R, n = 2.0, 3142
    t = np.linspace(0.0, math.pi / 2, n + 1)
    for (ox, oy), most in (((0.0, 0.0), 1), (UTM, 2)):
        p = [np.array([ox + R * math.cos(a), oy + R * math.sin(a), -0.8]) for a in t]
        w = _straight_runs([(p[k], p[k + 1]) for k in range(n)], closed=False)
        assert max(np.bincount(w)) == most


# audit W1R2-05, user decision Q30
def test_an_arc_electrode_at_utm_keeps_its_bends() -> None:
    """A valid electrode: a 2 deg arc of R = 50 m in 6 mm edges (r = 5 mm),
    each vertex 0.36 um off the chord of its neighbours. Base ecff7b6 at
    the UTM origin: one straight wire, Z +1.27 % against the local origin
    (291 wires); now 146 wires (pairs of edges below the tolerance) and
    Z +5.5e-4 (+3.6e-5 of it is the round-off of the absolute coordinates
    in the image kernel, measured with every edge a wire)."""
    from groundfield.geometry.primitives import _straight_runs

    R = 50.0
    t = np.linspace(0.0, math.radians(2.0), 292)
    zs, wires = [], []
    for ox, oy in ((0.0, 0.0), UTM):
        v = [(ox + R * math.cos(a), oy + R * math.sin(a), 0.8) for a in t]
        world = gf.create_world(soil=gf.HomogeneousSoil(rho=RHO))
        gf.create_electrode(
            world, "polyline", name="arc", vertices=v, closed=False, wire_radius=0.005
        )
        wires.append(1 + max(_straight_runs(world.electrodes[0].edges, closed=False)))
        gf.create_source(world, attached_to="arc", magnitude=1.0)
        res = _solve(world, _engine(ds=0.5))
        zs.append(complex(res.cluster_impedance("arc")[0]).real)
    assert wires[0] == 291
    assert wires[1] >= 145
    assert zs[1] == pytest.approx(zs[0], rel=1e-3)


# audit W1R2-05 (guard: three vertices are the old test, bit for bit)
def test_the_chord_of_three_vertices_is_the_neighbour_test() -> None:
    from groundfield.geometry.primitives import _continues_straight, _on_chord

    rng = np.random.default_rng(5)
    for origin in (np.zeros(3), np.array([UTM[0], UTM[1], 0.0])):
        for _ in range(500):
            a = origin + rng.uniform(-50, 50, 3)
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            b = a + rng.uniform(0.1, 20) * u
            off = rng.normal(size=3) * 10.0 ** rng.uniform(-13, -5)
            c = b + rng.uniform(0.1, 20) * u + off
            assert _on_chord([a, b, c]) == _continues_straight(a, b, c)


# ---------------------------------------------------------------------
# The chord rule in linear time (audit W1R3-03)
# ---------------------------------------------------------------------


def _on_chord_8c83d7c(vertices) -> bool:
    """The chord test of 8c83d7c (``primitives._on_chord``), the oracle."""
    from groundfield.geometry.primitives import (
        _COLLINEAR_FLOOR_M,
        COLLINEAR_ROUNDOFF_ULPS,
        _coordinate_scale,
    )

    a, c = vertices[0], vertices[-1]
    ac = c - a
    scale = _coordinate_scale(*vertices)
    tol = COLLINEAR_ROUNDOFF_ULPS * float(np.finfo(float).eps) * scale
    if len(vertices) == 3:
        height = float(np.linalg.norm(np.cross(vertices[1] - a, ac))) / float(
            np.linalg.norm(ac)
        )
        return height <= tol + _COLLINEAR_FLOOR_M
    inner = np.asarray(vertices[1:-1], dtype=float) - a
    heights = np.linalg.norm(np.cross(inner, ac), axis=1) / float(np.linalg.norm(ac))
    return bool(np.all(heights <= tol + _COLLINEAR_FLOOR_M))


def _straight_runs_8c83d7c(edges, *, closed: bool) -> list[int]:
    """``_straight_runs`` of 8c83d7c (quadratic in the run length)."""
    pts = [(np.asarray(s, dtype=float), np.asarray(e, dtype=float)) for s, e in edges]

    def forward(prev, nxt) -> bool:
        return float(np.dot(nxt[0] - prev[0], nxt[1] - nxt[0])) > 0.0

    def run_holds(run) -> bool:
        return _on_chord_8c83d7c([pts[k][0] for k in run] + [pts[run[-1]][1]])

    wire = [0] * len(pts)
    first = 0
    for k in range(1, len(pts)):
        if forward(pts[k - 1], pts[k]) and run_holds(range(first, k + 1)):
            wire[k] = wire[k - 1]
        else:
            wire[k] = wire[k - 1] + 1
            first = k
    if closed and len(pts) > 2 and wire[-1] != 0 and forward(pts[-1], pts[0]):
        last = wire[-1]
        joined = [k for k, w in enumerate(wire) if w == last] + [
            k for k, w in enumerate(wire) if w == 0
        ]
        if run_holds(joined):
            wire = [0 if w == last else w for w in wire]
    return wire


def _random_polylines(rng, origin, count: int):
    """Polylines near the chord tolerance: collinear runs, bends of 1e-9
    rad ... 30 deg, zig-zags of 0.05 ... 3 x the tolerance at the
    origin's coordinate magnitude, open and closed."""
    o = np.array([origin[0], origin[1], 0.0])
    tol = 64.0 * float(np.finfo(float).eps) * max(1.0, float(np.max(np.abs(o))))
    for _ in range(count):
        p = o + rng.uniform(-500.0, 500.0, 3) * np.array([1.0, 1.0, 0.0])
        p[2] = rng.uniform(0.3, 2.0)
        th = rng.uniform(0.0, 2.0 * math.pi)
        pts = [p]
        for _run in range(int(rng.integers(1, 5))):
            u = np.array([math.cos(th), math.sin(th), 0.0])
            v = np.array([-u[1], u[0], 0.0])
            h = rng.uniform(0.05, 20.0)
            n = int(rng.integers(1, 30))
            amp = tol * 10.0 ** rng.uniform(-1.3, 0.5) if rng.random() < 0.5 else 0.0
            p0 = pts[-1]
            pts += [p0 + (j + 1) * h * u + amp * (j % 2) * v for j in range(n)]
            th += rng.choice([1.0, -1.0]) * 10.0 ** rng.uniform(-9.0, math.log10(0.52))
        closed = bool(rng.random() < 0.3) and len(pts) > 3
        edges = [(tuple(a), tuple(b)) for a, b in zip(pts[:-1], pts[1:], strict=True)]
        if closed:
            edges.append((tuple(pts[-1]), tuple(pts[0])))
        yield edges, closed


# audit W1R3-03 (guard: the classification of 8c83d7c, bit for bit)
@pytest.mark.parametrize("origin", [(0.0, 0.0), (3_000.0, -2_000.0), UTM])
def test_the_linear_chord_rule_classifies_as_the_quadratic_one(origin) -> None:
    from groundfield.geometry.primitives import _frame_origin, _straight_runs

    rng = np.random.default_rng(20261115)
    for edges, closed in _random_polylines(rng, origin, 400):
        assert _straight_runs(edges, closed=closed) == _straight_runs_8c83d7c(
            edges, closed=closed
        )
    # the same polylines shifted into a local frame, tolerances in the user's
    if origin != (0.0, 0.0):
        shift = np.array([origin[0], origin[1], 0.0])
        rng = np.random.default_rng(7)
        with _frame_origin(shift):
            for edges, closed in _random_polylines(rng, (0.0, 0.0), 100):
                assert _straight_runs(edges, closed=closed) == _straight_runs_8c83d7c(
                    edges, closed=closed
                )


# audit W1R3-03
def test_the_straight_runs_of_a_collinear_polyline_take_linear_time() -> None:
    """``_straight_runs`` re-tested the whole run against its chord at every
    new edge: on 34ad269 500 / 2000 collinear edges took 0.22 / 3.3 s
    (ratio 15, quadratic; ecff7b6 0.011 / 0.042 s, ratio 3.9). The
    vectorised heights are taken only where the bound of the run does not
    decide (best of three)."""
    import time

    from groundfield.geometry.primitives import _straight_runs

    d = np.array([math.cos(0.3), math.sin(0.3), 0.0])

    def run(n: int) -> float:
        pts = [tuple(d * (0.5 * k) + np.array([0.0, 0.0, 0.8])) for k in range(n + 1)]
        edges = list(zip(pts[:-1], pts[1:], strict=True))
        best = math.inf
        for _ in range(3):
            t0 = time.perf_counter()
            assert max(_straight_runs(edges, closed=False)) == 0
            best = min(best, time.perf_counter() - t0)
        return best

    small, big = run(500), run(2000)
    assert big < 8.0 * max(small, 1e-3)
    assert run(4000) < 1.0


# audit W1R3-03
def test_polyline_segments_compute_the_straight_runs_once(monkeypatch) -> None:
    """``PolylineElectrode.segments`` redid the chord rule on every call (a
    2 km polyline in 1 m vertices solved in 17.7 s on 34ad269 against
    0.46 s on ecff7b6). The runs are cached on the vertices' values, the
    flag ``closed`` and the active frame origin, so the cache stays right
    under ``model_copy(update=...)``, assignment and in-place edits."""
    import groundfield.geometry.electrodes as el
    from groundfield.geometry.primitives import (
        _edge_wires,
        _frame_origin,
        _straight_runs,
    )

    cache = getattr(el, "_polyline_runs", None)
    if cache is not None:
        cache.cache_clear()
    calls: list[int] = []

    def counted(edges, *, closed):
        calls.append(1)
        return _straight_runs(edges, closed=closed)

    monkeypatch.setattr(el, "_straight_runs", counted)
    straight = [(0.1 * k + 0.123, 0.0, 0.8) for k in range(40)]
    e = PolylineElectrode(name="p", vertices=straight, closed=False)

    def expected(elec, ds=0.5):
        return _edge_wires(
            elec.edges,
            ds,
            skip_degenerate=True,
            edge_wires=_straight_runs(elec.edges, closed=elec.closed),
        ).wire_index

    for ds in (0.5, 0.05, 0.5):
        assert e.segments(ds).n_wires == 1
    assert len(calls) == 1
    x19 = straight[19][0]
    bent = [*straight[:20], *[(x, 0.5 * (x - x19), z) for x, _, z in straight[20:]]]
    copied = e.model_copy(update={"vertices": bent})
    assert copied.segments(0.5).n_wires == 2
    np.testing.assert_array_equal(copied.segments(0.5).wire_index, expected(copied))
    assert e.segments(0.5).n_wires == 1
    e.vertices = bent
    np.testing.assert_array_equal(e.segments(0.5).wire_index, expected(e))
    e.vertices[30] = (e.vertices[30][0], e.vertices[30][1] + 0.01, 0.8)
    np.testing.assert_array_equal(e.segments(0.5).wire_index, expected(e))
    assert e.segments(0.5).n_wires == 5  # the moved vertex: two edges of their own
    # a zig-zag below the tolerance at UTM: one wire in the user's frame only
    zig = [(0.5 * k, 2e-7 * (k % 2), 0.8) for k in range(21)]
    z = PolylineElectrode(name="z", vertices=zig, closed=False)
    assert z.segments(0.5).n_wires == 20
    with _frame_origin(np.array([UTM[0], UTM[1], 0.0])):
        assert z.segments(0.5).n_wires == 1
    assert z.segments(0.5).n_wires == 20
