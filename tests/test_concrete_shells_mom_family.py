"""Concrete shells in the point-matching family (``mom``, ``mom_sommerfeld``).

Release scope of 2026-11-14 (roadmap item *Concrete shells in the mom
family*): up to the base of this stream both backends rejected electrodes
with ``concrete_shell_coefficient_ohm_m`` (``BackendCapabilityError``).
They now add the Sunde shell term of ADR-0012 per segment, as the image
family does: a segment of length :math:`\\Delta s` carrying the current
:math:`I` drops :math:`C I / \\Delta s` across its shell, so
:math:`C/\\Delta s` joins the diagonal of the segment reaction matrix and
the segment-midpoint potentials (the potential of the metal).

Two methods check it:

* the image family with one owner per segment (every segment its own
  uniform-current owner, all bonded) solves the same matrix physics as
  ``mom`` with the point kernel — they agree to round-off, with and
  without shells, in homogeneous, two-layer and cross-layer soil;
* closed forms: a ring's segments are equivalent (the reaction matrix is
  circulant), so its current stays uniform and the shell adds exactly
  :math:`C/(2\\pi r)` — Sunde's series decomposition — in every soil,
  three layers included (``mom_sommerfeld``); for a dry shell
  (:math:`C \\to \\infty`) the current of any electrode becomes uniform per
  length and :math:`Z - C/L` tends to the uniform-current value.

Worlds without a shell are bit-identical to the base (the golden cases of
``mom`` and ``mom_sommerfeld`` run strict).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

#: rho_c = 100 Ohm m, 5 cm of concrete around a 5 mm wire (r_b = 55 mm).
_C = 100.0 / (2.0 * math.pi) * math.log(0.055 / 0.005)
_RB = 0.055

_HOM = gf.HomogeneousSoil(rho=100.0)
_TWO = gf.TwoLayerSoil(rho_1=30.0, rho_2=200.0, h_1=4.0)
_THREE = gf.MultiLayerSoil(
    layers=[
        gf.SoilLayer(rho=100.0, h=4.0),
        gf.SoilLayer(rho=400.0, h=6.0),
        gf.SoilLayer(rho=30.0),
    ]
)

_KINDS = {
    "rod": dict(position=(0.0, 0.0, 0.6), length=3.0),
    "ring": dict(center=(0.0, 0.0, 0.7), radius=4.0),
    "strip": dict(start=(-4.0, 0.0, 0.7), end=(4.0, 0.0, 0.7)),
    "grid_mesh": dict(corner=(-4.0, -4.0, 0.7), size=(8.0, 6.0), n_x=2, n_y=2),
    "polyline": dict(
        vertices=[(0.0, 0.0, 0.7), (6.0, 0.0, 0.7), (6.0, 3.0, 0.7), (0.0, 3.0, 0.7)]
    ),
    "star": dict(center=(0.0, 0.0, 0.7), n_arms=3, arm_length=3.0),
}


def _world(kind: str, coefficient: float, soil, **extra) -> gf.World:
    world = gf.create_world(soil=soil)
    params = {"wire_radius": _RB if coefficient else 0.005, **_KINDS[kind], **extra}
    gf.create_electrode(
        world,
        kind,
        name="e",
        concrete_shell_coefficient_ohm_m=coefficient,
        **params,
    )
    gf.create_source(world, attached_to="e", magnitude=1.0)
    return world


def _engine(backend: str, ds: float = 0.5, **kw) -> gf.Engine:
    return gf.create_engine(
        backend=backend, segment_length=ds, frequencies=[50.0], **kw
    )


def _solve(world: gf.World, engine: gf.Engine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.solve(world)


def _z(world: gf.World, engine: gf.Engine, name: str = "e") -> float:
    return float(_solve(world, engine).cluster_impedance(name)[0].real)


def _one_owner_per_segment(monkeypatch) -> None:
    """Image family: every segment its own uniform-current owner, all bonded."""
    from groundfield.solver import _pipeline

    original = _pipeline._discretize_electrode

    def every_segment_a_wire(electrode, ds, layer_interfaces=None):
        segments = original(electrode, ds, layer_interfaces)
        for k, segment in enumerate(segments):
            segment.wire = k
        return segments

    monkeypatch.setattr(_pipeline, "_discretize_electrode", every_segment_a_wire)


# ---------------------------------------------------------------------
# Method 1: the image family with one owner per segment
# ---------------------------------------------------------------------


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("kind", sorted(_KINDS))
@pytest.mark.parametrize(
    "soil",
    [_HOM, _TWO],
    ids=["homogeneous", "two_layer"],
)
def test_mom_with_shells_equals_one_owner_per_segment(kind, soil, monkeypatch) -> None:
    # release scope 2026-11-14: concrete shells in the mom family
    z_mom = _z(_world(kind, _C, soil), _engine("mom"))
    _one_owner_per_segment(monkeypatch)
    z_seg = _z(_world(kind, _C, soil), _engine("image", kernel="point"))
    assert z_mom == pytest.approx(z_seg, rel=1e-12)
    # Bounds of the equipotential solution, Z = min I^T (K + S) I over
    # sum(I) = 1: at least Z(0) + C/L (each term minimised on its own; the
    # shell term is smallest for the uniform current), at most the uniform
    # current's value Z_unif(0) + C/L.
    from groundfield.solver._discretize import _one_owner_per_wire_off

    monkeypatch.undo()
    bare = _world(kind, 0.0, soil, wire_radius=_RB)
    z_bare = _z(bare, _engine("mom"))
    with _one_owner_per_wire_off():
        z_unif = _z(bare, _engine("image", kernel="point"))
    shell = _C / bare.electrodes[0].total_length()
    assert shell * (1.0 - 1e-12) <= z_mom - z_bare <= shell + (z_unif - z_bare) + 1e-12


@pytest.mark.finding("10.12")
def test_cross_layer_cluster_with_a_bare_rod(monkeypatch) -> None:
    """An encased ring bonded to two bare rods through h_1, per segment."""
    # release scope 2026-11-14: concrete shells in the mom family

    def build():
        world = gf.create_world(soil=_TWO)
        gf.create_electrode(
            world,
            "ring",
            name="ring",
            center=(0.0, 0.0, 0.8),
            radius=3.0,
            wire_radius=_RB,
            concrete_shell_coefficient_ohm_m=_C,
        )
        for k, x in enumerate((-3.0, 3.0)):
            gf.create_electrode(
                world, "rod", name=f"rod{k}", position=(x, 0.0, 0.8), length=6.0
            )
            gf.create_conductor(world, start="ring", end=f"rod{k}")
        gf.create_source(world, attached_to="ring", magnitude=1.0)
        return world

    res_mom = _solve(build(), _engine("mom"))
    _one_owner_per_segment(monkeypatch)
    res_seg = _solve(build(), _engine("image", kernel="point"))
    assert res_mom.cluster_impedance("ring")[0] == pytest.approx(
        res_seg.cluster_impedance("ring")[0], rel=1e-12
    )
    for name in ("ring", "rod0", "rod1"):
        assert res_mom.electrode_currents[name][0] == pytest.approx(
            res_seg.electrode_currents[name][0], rel=1e-10
        )


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("kind", ["rod", "grid_mesh"])
def test_mom_sommerfeld_reproduces_mom_with_shells(kind) -> None:
    """Two layers: the quadrature kernel against the series kernel."""
    # release scope 2026-11-14: concrete shells in the mom family
    z_mom = _z(_world(kind, _C, _TWO), _engine("mom"))
    z_som = _z(_world(kind, _C, _TWO), _engine("mom_sommerfeld"))
    assert z_som == pytest.approx(z_mom, rel=1e-6)


# ---------------------------------------------------------------------
# Method 2: closed forms
# ---------------------------------------------------------------------


@pytest.mark.finding("10.12")
@pytest.mark.parametrize(
    ("backend", "soil"),
    [
        ("mom", _HOM),
        ("mom", _TWO),
        ("mom_sommerfeld", _HOM),
        ("mom_sommerfeld", _TWO),
        ("mom_sommerfeld", _THREE),
    ],
    ids=["mom-hom", "mom-2L", "sommerfeld-hom", "sommerfeld-2L", "sommerfeld-3L"],
)
def test_ring_shell_adds_exactly_c_over_l(backend, soil) -> None:
    """Sunde's series decomposition: shell and soil add along the radius.

    The segments of a ring are equivalent, so the current stays uniform
    under any shell and ``Z(C) - Z(0) = C / (2 pi r)`` exactly.
    """
    # release scope 2026-11-14: concrete shells in the mom family
    engine = _engine(backend)
    z_shell = _z(_world("ring", _C, soil), engine)
    z_bare = _z(_world("ring", 0.0, soil, wire_radius=_RB), engine)
    assert z_shell - z_bare == pytest.approx(_C / (2.0 * math.pi * 4.0), rel=1e-11)


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("kind", ["rod", "grid_mesh", "star"])
def test_dry_shell_makes_the_current_uniform(kind) -> None:
    """For C -> infinity the shell dominates: I_k / ds_k -> 1 / L, Z - C/L -> uniform value."""
    # release scope 2026-11-14: concrete shells in the mom family
    from groundfield.solver._kernels import HalfSpaceKernel
    from groundfield.solver._pipeline import SegmentArrays, discretise

    big = 1e6 * _C
    world = _world(kind, big, _HOM)
    engine = _engine("mom")
    res = _solve(world, engine)
    disc = discretise(world, engine)
    arrays = SegmentArrays.from_segments(disc.segments)
    length = float(arrays.lengths.sum())
    density = np.array([ps.current[0].real / ps.length for ps in res.point_sources])
    assert np.max(np.abs(density * length - 1.0)) < 1e-5
    K = HalfSpaceKernel(100.0).matrix(arrays.points, arrays.lengths, arrays.radii)
    w = arrays.lengths / length
    uniform = float(w @ K @ w)
    z = float(res.cluster_impedance("e")[0].real)
    assert z - big / length == pytest.approx(uniform, rel=1e-5)


@pytest.mark.finding("10.12")
def test_node_potential_is_the_metal_potential() -> None:
    """Segment potentials carry the shell drop; the field is that of the soil.

    The node potential is the potential of the metal: the soil-side
    potential at r_b plus C I_k / ds_k on every segment. Field points see
    the segment currents in the soil (the shell is a series element).
    """
    # release scope 2026-11-14: concrete shells in the mom family
    from groundfield.solver._kernels import HalfSpaceKernel
    from groundfield.solver._pipeline import SegmentArrays, discretise

    world = _world("strip", _C, _HOM)
    engine = _engine("mom")
    res = _solve(world, engine)
    arrays = SegmentArrays.from_segments(discretise(world, engine).segments)
    current = np.array([ps.current[0] for ps in res.point_sources])
    soil_side = (
        HalfSpaceKernel(100.0).matrix(arrays.points, arrays.lengths, arrays.radii)
        @ current.real
    )
    metal = soil_side + _C / arrays.lengths * current.real
    u = res.electrode_potentials["e"][0].real
    assert np.max(np.abs(metal / u - 1.0)) < 1e-12
    # A surface point far away sees the leakage current, not the shell.
    far = np.array([[200.0, 0.0, 0.0]])
    phi = res.potential(far)[0].real
    assert phi == pytest.approx(100.0 / (2.0 * math.pi * 200.0), rel=2e-3)


# ---------------------------------------------------------------------
# Routing and the bare limit
# ---------------------------------------------------------------------


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("backend", ["mom", "mom_sommerfeld", "bem"])
def test_engine_solve_accepts_shells_in_the_mom_family(backend) -> None:
    # release scope 2026-11-14: concrete shells in the mom family
    res = _solve(_world("strip", _C, _HOM), _engine(backend))
    z_bare = _z(_world("strip", 0.0, _HOM, wire_radius=_RB), _engine(backend))
    assert res.cluster_impedance("e")[0].real > z_bare


def test_direct_backend_calls_accept_shells() -> None:
    from groundfield.solver.mom import solve_mom
    from groundfield.solver.mom_sommerfeld import solve_mom_sommerfeld

    for solve in (solve_mom, solve_mom_sommerfeld):
        res = solve(_world("rod", _C, _TWO), _engine("mom"))
        assert res.cluster_impedance("e")[0].real > 0.0


@pytest.mark.parametrize("backend", ["mom", "mom_sommerfeld"])
def test_zero_coefficient_is_bit_identical(backend) -> None:
    """``C = 0`` and an electrode without the field solve to the same bits."""
    engine = _engine(backend)
    world_a = gf.create_world(soil=_TWO)
    gf.create_electrode(world_a, "rod", name="e", position=(0.0, 0.0, 0.6), length=3.0)
    gf.create_source(world_a, attached_to="e", magnitude=1.0)
    world_b = _world("rod", 0.0, _TWO)
    res_a, res_b = _solve(world_a, engine), _solve(world_b, engine)
    assert res_a.cluster_impedance("e") == res_b.cluster_impedance("e")
    assert [p.current for p in res_a.point_sources] == [
        p.current for p in res_b.point_sources
    ]


def test_fem_still_rejects_shells() -> None:
    from groundfield.solver.registry import BackendCapabilityError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BackendCapabilityError, match="concrete-shell"):
            gf.create_engine(backend="fem").solve(_world("rod", _C, _HOM))
