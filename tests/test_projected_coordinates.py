"""Solves at projected (UTM-size) coordinates are translation invariant.

Audit fw1sol (open point 1 of its report) found that ``mom`` gives a
2.6e-4 different impedance for the same generated mesh at the UTM origin
(easting 32 604 321.123 m with the zone prefix, northing 5 812 345.678 m)
than at (0, 0). Measured on the base of this stream for every backend
(evidence ``utm_invariance.py`` of stream w2mom), two causes:

* **Segment counts.** An edge whose length is a whole multiple of the
  segment length got one segment more at UTM coordinates: the round-off
  of the vertices (up to a few 1e-9 m) exceeded the tolerance of the count
  rule (1e-9 of ``ds``). The generated 10 m 10 x 10 mesh had 508 instead
  of 440 segments (mom +2.6e-4, image with the point kernel +2.8e-4), an
  L-shaped outline 85 instead of 84 (-7.9e-6).
* **Round-off of the kernels.** The line kernel computed segment end
  points and quadrature nodes at the magnitude of the coordinates: a strip
  and a star moved by 1.4e-7, the mutual matrix of three outlines by 2e-7.

The count rule now allows the round-off of coordinates of the size of the
edge's own (the straight-run test of audit W1A-01 does the same), and the
solvers compute in a local frame — the world shifted by a reference point
of its own — whenever a coordinate exceeds 10 km; results are reported in
the user's coordinates. Worlds within 10 km of the origin are not shifted;
their electrodes are discretised as before, while a distributed conductor
whose length exceeds a whole number of pieces only by round-off now gets
that whole number at any position (audit W2B-03,
``test_w2b_03_local_counterpoise_gets_whole_pieces``).

The yardstick is the round-off of local coordinates: the same world moved
by local offsets of 0.1 m to 8.8 km. With the point kernel (``image``
with ``kernel="point"``, ``mom``, ``mom_sommerfeld``) it is 1e-14, and the
UTM solve now lies 3e-11 from the local one (the rounding of the input
vertices themselves, about 1e-9 m). The line kernel switches from the exact
Galerkin integral to its far form at a midpoint distance of six segment
lengths, where the pairs six segments apart on a straight wire sit exactly:
round-off decides their form at any position, and the impedances of the
worlds here spread by up to 2.4e-7 over the local offsets (audit W2B-01:
up to 5.4e-7 for direct pairs, 8e-6 ... 2.6e-5 in two-layer soil where an
image distance is six segment lengths). The UTM solves lie within that
spread (they were up to 1.5e-6 off).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

ORIGIN = (32_604_321.123, 5_812_345.678)
#: Bounds of the UTM solve against the local one: (impedance and potential,
#: touch voltage). The line kernel's bound is its spread over local offsets
#: (2.4e-7 measured, evidence ``local_offsets_line_kernel.py``) with a margin.
POINT_BOUND = (1e-9, 1e-8)
LINE_BOUND = (5e-7, 2e-6)

HOM = gf.HomogeneousSoil(rho=100.0)
TWO = gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=3.0)
THREE = gf.MultiLayerSoil(
    layers=[
        gf.SoilLayer(rho=100.0, h=5.0),
        gf.SoilLayer(rho=300.0, h=10.0),
        gf.SoilLayer(rho=30.0),
    ]
)

#: Field points around the electrodes, local coordinates (m).
FIELD = np.array(
    [[1.3, 0.7, 0.0], [6.0, 6.0, 0.0], [-3.0, 11.0, 0.0], [15.0, -2.0, 0.0]]
)


def _solve(world, backend="image", ds=0.5, freqs=(50.0,), **kw):
    engine = gf.create_engine(
        backend=backend, segment_length=ds, frequencies=list(freqs), **kw
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.solve(world)


def _at(origin, pts=FIELD):
    p = np.array(pts, dtype=float, copy=True)
    p[:, 0] += origin[0]
    p[:, 1] += origin[1]
    return p


def _rel(a, b) -> float:
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ---------------------------------------------------------------------
# Worlds, built at an origin
# ---------------------------------------------------------------------


def outline_l(origin, soil=HOM, shell=False):
    th = math.radians(21.0)
    u = np.array([math.cos(th), math.sin(th)])
    v = np.array([-u[1], u[0]])
    o = np.asarray(origin, float)
    local = [(0, 0), (6, 0), (12, 0), (12, 5), (5, 5), (5, 9), (0, 9), (0, 4.5)]
    verts = [tuple(o + a * u + b * v) + (0.8,) for a, b in local]
    w = gf.create_world(soil=soil)
    extra = {}
    if shell:
        extra = dict(wire_radius=0.055, concrete_shell_coefficient_ohm_m=38.0)
    gf.create_electrode(w, "polyline", name="h", closed=True, vertices=verts, **extra)
    gf.create_source(w, attached_to="h", magnitude=1.0)
    return w, "h"


def strip_star(origin, soil=HOM):
    ox, oy = origin
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w,
        "strip",
        name="s",
        start=(ox + 1.0, oy + 2.0, 0.6),
        end=(ox + 1.0 + 10.0 * math.cos(0.4), oy + 2.0 + 10.0 * math.sin(0.4), 0.6),
    )
    gf.create_electrode(
        w,
        "star",
        name="st",
        center=(ox - 8.0, oy + 3.0, 0.5),
        n_arms=4,
        arm_length=5.0,
        orientation_deg=17.0,
    )
    gf.create_conductor(w, start="s", end="st")
    gf.create_source(w, attached_to="s", magnitude=1.0)
    return w, "s"


def ring_rods(origin, soil=TWO):
    ox, oy = origin
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "ring", name="ring", center=(ox, oy, 0.8), radius=5.0)
    for k in range(4):
        a = 2 * math.pi * k / 4 + 0.3
        gf.create_electrode(
            w,
            "rod",
            name=f"rod{k}",
            position=(ox + 5.0 * math.cos(a), oy + 5.0 * math.sin(a), 0.8),
            length=6.0,
        )
        gf.create_conductor(w, start="ring", end=f"rod{k}")
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    return w, "ring"


def mesh33(origin, soil=HOM):
    """Generated 10 m 10 x 10 mesh foundation at 33 deg (the fw1sol case)."""
    from groundfield.generators import FoundationElectrodeSpec, GroundingSystemSpec

    spec = FoundationElectrodeSpec(
        style="mesh",
        size_m=10.0,
        depth_m=0.8,
        wire_radius_m=0.005,
        orientation_deg=33.0,
        n_x=10,
        n_y=10,
    )
    world = gf.World(soil=soil)
    anchor = GroundingSystemSpec(electrodes=[spec]).build_at(
        world, origin, "h", np.random.default_rng(0)
    )
    gf.create_source(world, attached_to=anchor, magnitude=1.0)
    return world, anchor


def cables(origin, soil=HOM):
    """Inductive cables 0.2 m apart and a distributed galvanic counterpoise."""
    ox, oy = origin
    c, s = math.cos(math.radians(37.0)), math.sin(math.radians(37.0))
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="a", position=(ox, oy, 0.5), length=2.0)
    gf.create_electrode(
        w, "rod", name="b", position=(ox + 120.0 * c, oy + 120.0 * s, 0.5), length=2.0
    )
    gf.create_electrode(
        w,
        "rod",
        name="d",
        position=(ox + 60.0 * c - 0.2 * s, oy + 60.0 * s + 0.2 * c, 0.5),
        length=2.0,
    )
    gf.create_conductor(
        w,
        name="k1",
        start="a",
        end="b",
        cross_section=95e-6,
        inductance_model="neumann",
        wire_radius=0.006,
    )
    gf.create_conductor(
        w,
        name="k3",
        start="d",
        end="b",
        cross_section=50e-6,
        inductance_model="neumann",
        wire_radius=0.006,
    )
    # A 60 m counterpoise in 10 m pieces at 20 deg: its length is 60 m at
    # (0, 0) and 60.0000000012 m at the UTM origin.
    c20, s20 = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
    gf.create_electrode(w, "rod", name="f", position=(ox + 3.0, oy, 0.5), length=1.0)
    gf.create_electrode(
        w,
        "rod",
        name="g",
        position=(ox + 3.0 + 60.0 * c20, oy + 60.0 * s20, 0.5),
        length=1.0,
    )
    gf.create_conductor(
        w,
        name="cp",
        start="f",
        end="g",
        cross_section=50e-6,
        discretize_segment_length=10.0,
        coupling_to_soil="galvanic",
    )
    gf.create_conductor(w, name="fa", start="a", end="f")
    gf.create_source(w, attached_to="a", magnitude=1.0)
    return w, "a"


# ---------------------------------------------------------------------
# Segment counts
# ---------------------------------------------------------------------


@pytest.mark.finding("10.32")
@pytest.mark.parametrize(
    "build", [outline_l, strip_star, mesh33], ids=lambda b: b.__name__
)
def test_segment_counts_do_not_depend_on_the_position(build) -> None:
    """An edge of a whole number of segments keeps it at UTM coordinates."""
    from groundfield.solver._pipeline import discretise

    engine = gf.create_engine(segment_length=0.5)
    local, _ = build((0.0, 0.0))
    utm, _ = build(ORIGIN)
    lengths_local = [s.length for s in discretise(local, engine).segments]
    lengths_utm = [s.length for s in discretise(utm, engine).segments]
    assert len(lengths_utm) == len(lengths_local)
    assert np.allclose(lengths_utm, lengths_local, rtol=1e-8, atol=0.0)
    # The electrode's own count (diagnostics) agrees with the solver's.
    assert sum(e.segment_count(0.5) for e in utm.electrodes) == len(lengths_utm)


# audit W2B-03
def test_w2b_03_local_counterpoise_gets_whole_pieces() -> None:
    """The count rule allows round-off at any position, near the origin too.

    A 60 m counterpoise in 10 m pieces from (3, 0, 0.5) m at 0.1 deg has the
    computed length 60.00000000000001 m. Up to the projected-coordinate fix
    (base 1579ed8) it got 7 pieces, as in 876 of 3600 directions at 0.1 deg
    steps, now 6 in every one (audit W2B-03; Z -0.30 % ``image``, -2.54 %
    ``mom``, evidence ``w2b03_counterpoise.py`` of stream fw2mom). Pins the
    new behaviour; *Units and conventions* says so.
    """
    a = math.radians(0.1)
    c = gf.Conductor(
        name="cp",
        start=(3.0, 0.0, 0.5),
        end=(3.0 + 60.0 * math.cos(a), 60.0 * math.sin(a), 0.5),
        cross_section=50e-6,
        discretize_segment_length=10.0,
        coupling_to_soil="galvanic",
    )
    assert c.length > 60.0
    assert c.n_segments == 6


@pytest.mark.finding("10.32")
def test_distributed_conductor_pieces_do_not_depend_on_the_position() -> None:
    local, _ = cables((0.0, 0.0))
    utm, _ = cables(ORIGIN)
    n_local = [c.n_segments for c in local.conductors]
    n_utm = [c.n_segments for c in utm.conductors]
    assert n_utm == n_local == [1, 1, 6, 1]
    assert utm.conductors[2].length > 60.0 == local.conductors[2].length


def test_counts_at_local_coordinates_are_unchanged() -> None:
    """Near the origin the count rule is the historic one (tolerance 1e-9 ds)."""
    from groundfield.geometry.primitives import SEGMENT_COUNT_TOL, _segment_count

    assert SEGMENT_COUNT_TOL == 1e-9
    assert _segment_count(3.0 + 4e-16, 1.5) == 2
    assert _segment_count(3.0 * (1.0 + 2e-9), 1.5) == 3
    assert _segment_count(10.0 * (1.0 + 2e-9), 0.5, scale=100.0) == 21
    # The round-off of a coordinate of 3.26e7 m is allowed: 4.6e-7 m.
    assert _segment_count(1.0 + 5e-9, 0.5, scale=3.3e7) == 2
    assert _segment_count(1.0 + 1e-6, 0.5, scale=3.3e7) == 3


# ---------------------------------------------------------------------
# Impedances, potentials and touch voltages of every backend
# ---------------------------------------------------------------------

CASES = [
    ("mesh33", mesh33, HOM, "image", {}),
    ("mesh33", mesh33, HOM, "image", {"kernel": "point"}),
    ("mesh33", mesh33, HOM, "mom", {}),
    ("outline_l", outline_l, HOM, "image", {}),
    ("outline_l", outline_l, HOM, "image", {"kernel": "point"}),
    ("outline_l", outline_l, TWO, "image", {}),
    ("outline_l", outline_l, TWO, "image", {"kernel": "point"}),
    ("outline_l", outline_l, TWO, "mom", {}),
    ("outline_l", outline_l, TWO, "mom_sommerfeld", {}),
    ("strip_star", strip_star, HOM, "image", {}),
    ("strip_star", strip_star, HOM, "mom", {}),
    ("ring_rods", ring_rods, TWO, "image", {}),
    ("ring_rods", ring_rods, TWO, "mom", {}),
]


@pytest.mark.parametrize(
    ("name", "build", "soil", "backend", "kw"),
    CASES,
    ids=[f"{c[0]}-{c[2].kind}-{c[3]}-{c[4].get('kernel', 'default')}" for c in CASES],
)
def test_solves_are_translation_invariant(name, build, soil, backend, kw) -> None:
    # audit fw1sol, open point 1: solves at projected coordinates
    from groundfield.postprocess.safety import touch_voltage

    out = {}
    for origin in ((0.0, 0.0), ORIGIN):
        world, anchor = build(origin, soil)
        res = _solve(world, backend, **kw)
        out[origin] = (
            res.cluster_impedance(anchor)[0],
            res.potential(_at(origin)),
            touch_voltage(res, world, electrode=anchor, direction=(0.3, 1.0, 0.0)),
        )
    local, utm = out[(0.0, 0.0)], out[ORIGIN]
    line = backend == "image" and kw.get("kernel") != "point"
    bound, touch_bound = LINE_BOUND if line else POINT_BOUND
    assert _rel(utm[0], local[0]) < bound
    assert _rel(utm[1], local[1]) < bound
    assert _rel(utm[2], local[2]) < touch_bound


def test_three_layer_solve_is_translation_invariant() -> None:
    z = [
        _solve(outline_l(o, THREE)[0], "mom_sommerfeld").cluster_impedance("h")[0]
        for o in ((0.0, 0.0), ORIGIN)
    ]
    assert _rel(z[1], z[0]) < 1e-9


@pytest.mark.parametrize("backend", ["image", "mom"])
def test_inductive_solve_is_translation_invariant(backend) -> None:
    out = []
    for origin in ((0.0, 0.0), ORIGIN):
        world, anchor = cables(origin)
        res = _solve(world, backend, ds=2.0, freqs=(50.0, 1000.0))
        out.append((res.cluster_impedance(anchor), res.potential(_at(origin), 1)))
    assert _rel(out[1][0], out[0][0]) < 1e-10
    assert _rel(out[1][1], out[0][1]) < 1e-9


def test_concrete_shells_in_mom_are_translation_invariant() -> None:
    z = [
        _solve(outline_l(o, TWO, shell=True)[0], "mom").cluster_impedance("h")[0]
        for o in ((0.0, 0.0), ORIGIN)
    ]
    assert _rel(z[1], z[0]) < 1e-9


@pytest.mark.parametrize("kernel", ["line", "point"])
@pytest.mark.parametrize("soil", [HOM, TWO], ids=["homogeneous", "two_layer"])
def test_mutual_matrices_are_translation_invariant(soil, kernel) -> None:
    from groundfield.solver.mutual import solve_mutual_field, solve_mutual_matrix

    def build(origin):
        w = gf.create_world(soil=soil)
        th = math.radians(28.0)
        u = np.array([math.cos(th), math.sin(th)])
        v = np.array([-u[1], u[0]])
        # Two rectangles and the L outline, whose edges are whole numbers of
        # 0.5 m segments (85 instead of 84 at the UTM origin on the base).
        for k, (bx, by) in enumerate(((0.0, 0.0), (5.0, 16.0))):
            base = np.array([origin[0] + bx, origin[1] + by])
            verts = [
                tuple(base + a * u + b * v) + (0.8,)
                for a, b in ((0, 0), (10, 0), (10, 8), (0, 8))
            ]
            gf.create_electrode(
                w, "polyline", name=f"f{k}", closed=True, vertices=verts
            )
        ell = outline_l((origin[0] + 14.0, origin[1] + 3.0), soil)[0].electrodes[0]
        w.electrodes.append(ell.model_copy(update={"name": "f2"}))
        return w

    engine = gf.create_engine(segment_length=0.5, frequencies=[50.0], kernel=kernel)
    anchors = ["f0", "f1", "f2"]
    out = []
    for origin in ((0.0, 0.0), ORIGIN):
        w = build(origin)
        out.append(
            (
                solve_mutual_matrix(w, engine, anchors),
                solve_mutual_field(w, engine, anchors, _at(origin)),
            )
        )
    bound = (LINE_BOUND if kernel == "line" else POINT_BOUND)[0]
    assert _rel(out[1][0], out[0][0]) < bound
    assert _rel(out[1][1], out[0][1]) < bound


def test_the_solve_runs_in_an_exactly_shifted_frame() -> None:
    """UTM solve == solve of the world moved by the frame origin (bit level).

    Without polylines (whose straight runs the frame decides with the user's
    coordinates) the shifted world is solved like any local world, so the
    two results agree to the last bit, the line kernel included.
    """
    from groundfield.solver._frame import local_origin

    utm, anchor = strip_star(ORIGIN)
    ox, oy, _ = local_origin(utm)
    # The stored coordinates minus the origin: exact (Sterbenz).
    moved = utm.model_copy(deep=True)
    strip, star = moved.electrodes
    moved.electrodes = [
        strip.model_copy(
            update={
                "start": (strip.start[0] - ox, strip.start[1] - oy, strip.start[2]),
                "end": (strip.end[0] - ox, strip.end[1] - oy, strip.end[2]),
            }
        ),
        star.model_copy(
            update={
                "center": (star.center[0] - ox, star.center[1] - oy, star.center[2])
            }
        ),
    ]
    moved.conductors = [
        c.model_copy(
            update={
                "start": (c.start[0] - ox, c.start[1] - oy, c.start[2]),
                "end": (c.end[0] - ox, c.end[1] - oy, c.end[2]),
            }
        )
        for c in moved.conductors
    ]
    assert local_origin(moved) is None
    res_utm, res_moved = _solve(utm), _solve(moved)
    assert res_utm.cluster_impedance(anchor) == res_moved.cluster_impedance(anchor)
    shifted = np.array([p.position for p in res_moved.point_sources]) + [ox, oy, 0.0]
    assert np.array_equal(
        np.array([p.position for p in res_utm.point_sources]), shifted
    )


# ---------------------------------------------------------------------
# Results in the user's coordinates; local worlds untouched
# ---------------------------------------------------------------------


def test_point_sources_are_reported_in_user_coordinates() -> None:
    res_local = _solve(strip_star((0.0, 0.0))[0])
    res_utm = _solve(strip_star(ORIGIN)[0])
    local = np.array([p.position for p in res_local.point_sources])
    utm = np.array([p.position for p in res_utm.point_sources])
    assert np.max(np.abs(utm - _at(ORIGIN, local))) < 1e-8
    # The stored result evaluates the same potentials after a JSON round trip.
    again = gf.FieldResult.model_validate_json(res_utm.model_dump_json())
    assert _rel(again.potential(_at(ORIGIN)), res_utm.potential(_at(ORIGIN))) < 1e-14


def test_worlds_near_the_origin_are_not_moved() -> None:
    """Within 10 km of the origin the solver works in the user's coordinates."""
    from groundfield.solver._frame import local_origin

    near, _ = strip_star((9_000.0, -9_000.0))
    assert local_origin(near) is None
    far, _ = strip_star((12_000.0, 0.0))
    assert local_origin(far) is not None
