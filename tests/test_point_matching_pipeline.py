"""The point-matching family on the shared solve pipeline.

Finding 10.75: the solve pipeline (discretisation, source injection,
topology, inductive branch block) was copied into seven backends; the
copies drifted (F10 survived in ``cim``). Since review pass 10 ``mom``,
``mom_sommerfeld`` and the deprecated alias ``bem`` run steps 1–4 of
:mod:`groundfield.solver._pipeline` and its burial-depth guard; only the
per-segment solve and the read-back are their own. An extension of a
shared step (two-terminal sources, junction nodes) therefore reaches
them without a second copy.

Finding 10.59: ``mom``, ``bem`` and ``mom_sommerfeld`` obtained the
reaction matrix by applying the kernel to the identity matrix, an
``O(N³)`` product per image term; the matrix is now assembled directly,
bit-identically.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf

MOM_FAMILY = ("mom", "mom_sommerfeld", "bem")
STEPS = ("discretise", "inject", "build_topology", "assemble_branches")

SOILS = {
    "homogeneous": gf.HomogeneousSoil(rho=100.0),
    "two_layer": gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0),
    "cross_layer": gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0),
}


def _two_port(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="A", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "ring", name="B", center=(4.0, 0.0, 0.8), radius=1.5)
    gf.create_source(w, attached_to="A", magnitude=1.0)
    return w


def _solve(backend: str, world: gf.World, **kw) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gf.create_engine(backend=backend, segment_length=0.5, **kw).solve(world)


# ---------------------------------------------------------------------
# 10.75 — one pipeline
# ---------------------------------------------------------------------


@pytest.mark.finding("10.75")
@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("backend", MOM_FAMILY)
def test_mom_family_runs_the_shared_pipeline_steps(backend, step, monkeypatch) -> None:
    from groundfield.solver import _pipeline

    calls: list[str] = []
    original = getattr(_pipeline, step)

    def spy(*args, **kwargs):
        calls.append(step)
        return original(*args, **kwargs)

    monkeypatch.setattr(_pipeline, step, spy)
    _solve(backend, _two_port(SOILS["two_layer"]))
    assert calls == [step]


@pytest.mark.finding("10.75")
@pytest.mark.parametrize("backend", MOM_FAMILY)
def test_an_extension_of_the_injection_step_reaches_the_mom_family(
    backend, monkeypatch
) -> None:
    """What the shared ``inject`` returns is what the mom family solves.

    The stand-in extension doubles every impressed current; the
    point-matching system is linear, so every potential and current
    doubles. (Two-terminal sources and junction nodes are added to the
    same step.)
    """
    from groundfield.solver import _pipeline

    world = _two_port(SOILS["homogeneous"])
    base = _solve(backend, world)
    original = _pipeline.inject

    def doubled(world, **kwargs):
        return {k: 2.0 * v for k, v in original(world, **kwargs).items()}

    monkeypatch.setattr(_pipeline, "inject", doubled)
    res = _solve(backend, world)
    for name in ("A", "B"):
        np.testing.assert_allclose(
            res.electrode_potentials[name],
            2.0 * np.asarray(base.electrode_potentials[name]),
            rtol=1e-12,
        )
    np.testing.assert_allclose(
        res.electrode_currents["A"],
        2.0 * np.asarray(base.electrode_currents["A"]),
        rtol=1e-12,
    )


@pytest.mark.finding("10.75")
@pytest.mark.parametrize("backend", MOM_FAMILY)
def test_mom_family_rejects_a_segment_partly_above_the_surface(backend) -> None:
    """The burial-depth guard of the pipeline now covers the mom family.

    One 1 m segment from z = -0.1 m to 0.9 m: the midpoint is buried, the
    top is in the air. ``image`` rejected it since 0.15.0; ``mom``,
    ``mom_sommerfeld`` and ``bem`` solved it as if buried (94.27 Ω on
    v0.15.0).
    """
    w = gf.create_world(soil=SOILS["homogeneous"])
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, -0.1), length=1.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    engine = gf.create_engine(backend=backend, segment_length=1.0)
    with pytest.raises(ValueError, match="crosses the soil surface"):
        engine.solve(w)


def test_image_rejects_a_segment_partly_above_the_surface() -> None:
    """Control: the image family's guard (0.15.0) that the mom family now shares."""
    w = gf.create_world(soil=SOILS["homogeneous"])
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, -0.1), length=1.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    engine = gf.create_engine(backend="image", segment_length=1.0)
    with pytest.raises(ValueError, match="crosses the soil surface"):
        engine.solve(w)


@pytest.mark.finding("10.75")
def test_mom_and_image_share_the_shallow_segment_advisory() -> None:
    """A horizontal 5 m segment at 0.7 m warns in both families when the
    self image is a point image (the image family's line kernel has no such
    bias and stays silent, audit F08)."""
    from groundfield.solver._guards import ShallowSegmentWarning

    w = gf.create_world(soil=SOILS["homogeneous"])
    gf.create_electrode(
        w, "strip", name="g1", start=(0.0, 0.0, 0.7), end=(10.0, 0.0, 0.7)
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    for backend in ("image", "mom"):
        with pytest.warns(ShallowSegmentWarning):
            gf.create_engine(backend=backend, segment_length=5.0, kernel="point").solve(
                w
            )


# ---------------------------------------------------------------------
# 10.59 — direct assembly, no identity product
# ---------------------------------------------------------------------


@pytest.mark.finding("10.59")
@pytest.mark.parametrize("soil_label", list(SOILS))
@pytest.mark.parametrize("backend", MOM_FAMILY)
def test_reaction_matrix_is_assembled_without_the_identity_product(
    backend, soil_label, monkeypatch
) -> None:
    """No kernel call receives an ``N × N`` excitation (the identity)."""
    from groundfield.solver import _kernels, image_2layer

    widest: list[int] = []

    def watch(cls):
        original = cls.__call__

        def call(self, seg_points, seg_lengths, wire_radii, currents):
            cur = np.asarray(currents)
            widest.append(cur.shape[1] if cur.ndim == 2 else 1)
            return original(self, seg_points, seg_lengths, wire_radii, currents)

        monkeypatch.setattr(cls, "__call__", call)

    watch(_kernels.HalfSpaceKernel)
    watch(_kernels.TwoLayerSeriesKernel)
    watch(image_2layer._TwoLayerKernel)
    res = _solve(backend, _two_port(SOILS[soil_label]))
    n = res.metadata["n_segments"]
    assert n > 4
    assert max(widest, default=1) <= 2, (n, widest)


@pytest.mark.parametrize("soil_label", list(SOILS))
def test_direct_reaction_matrix_equals_the_identity_extraction(soil_label) -> None:
    """Guard (10.59): the direct matrix is bit-identical to ``kernel(..., eye(N))``."""
    from groundfield.solver._pipeline import SegmentArrays, discretise
    from groundfield.solver.image import _self_corrected_kernel
    from groundfield.solver.image_2layer import _two_layer_self_kernel_factory
    from groundfield.solver.mom import _build_Z_homogeneous, _build_Z_two_layer

    soil = SOILS[soil_label]
    world = _two_port(soil)
    interfaces = (soil.h_1,) if isinstance(soil, gf.TwoLayerSoil) else None
    disc = discretise(world, gf.create_engine(segment_length=0.5), interfaces)
    a = SegmentArrays.from_segments(disc.segments)
    eye = np.eye(a.points.shape[0])
    if isinstance(soil, gf.TwoLayerSoil):
        Z = _build_Z_two_layer(a.points, a.lengths, a.radii, soil, 300, 1e-6)
        kernel = _two_layer_self_kernel_factory(soil, 300, 1e-6, allow_cross_layer=True)
        Z_eye = kernel(a.points, a.lengths, a.radii, eye)
    else:
        Z = _build_Z_homogeneous(a.points, a.lengths, a.radii, soil.rho)
        Z_eye = _self_corrected_kernel(a.points, a.lengths, a.radii, eye, soil.rho)
    assert np.array_equal(Z, Z_eye)


# ---------------------------------------------------------------------
# 10.75 — two-terminal sources and junctions reach the mom family
# through the shared inject / build_topology steps (CORE-2a, 10.16, 10.38)
# ---------------------------------------------------------------------

CORE2A = pytest.mark.skipif(
    "junctions" not in gf.World.model_fields,
    reason="return_to and junction nodes arrive with CORE-2a (release/calver)",
)


def _solver(name: str):
    from groundfield.solver.mom import solve_mom
    from groundfield.solver.mom_sommerfeld import solve_mom_sommerfeld

    return {"mom": solve_mom, "mom_sommerfeld": solve_mom_sommerfeld}[name]


def _two_rods(**source) -> gf.World:
    w = gf.create_world(soil=SOILS["two_layer"])
    gf.create_electrode(w, "rod", name="A", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="B", position=(20.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, magnitude=1.0, **source)
    return w


@CORE2A
@pytest.mark.finding("10.75")
@pytest.mark.parametrize("solver", ["mom", "mom_sommerfeld"])
def test_mom_family_honours_return_to(solver) -> None:
    """+I at ``attached_to``, -I at ``return_to``: KCL and superposition."""
    solve = _solver(solver)
    engine = gf.create_engine(segment_length=0.5)
    res = solve(_two_rods(attached_to="A", return_to="B"), engine)
    i_a = res.electrode_currents["A"][0]
    i_b = res.electrode_currents["B"][0]
    assert abs(i_a + i_b) <= 1e-12
    z_in = res.electrode_potentials["A"][0] - res.electrode_potentials["B"][0]
    r_a = solve(_two_rods(attached_to="A"), engine)
    r_b = solve(_two_rods(attached_to="B"), engine)
    z11, z21 = r_a.electrode_potentials["A"][0], r_a.electrode_potentials["B"][0]
    z12, z22 = r_b.electrode_potentials["A"][0], r_b.electrode_potentials["B"][0]
    assert abs(z_in - (z11 - z12 - z21 + z22)) <= 1e-12 * abs(z_in)


@CORE2A
@pytest.mark.finding("10.75")
@pytest.mark.parametrize("solver", ["mom", "mom_sommerfeld"])
def test_mom_family_reports_the_junction_potential(solver) -> None:
    """A source on a junction sees the lead in series: Z_in = Z_g1 + R."""
    from groundfield.api import create_junction

    solve = _solver(solver)
    engine = gf.create_engine(segment_length=0.5)

    def world(on_junction: bool) -> gf.World:
        w = gf.create_world(soil=SOILS["two_layer"])
        gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
        if on_junction:
            create_junction(w, name="J", position=(0.0, 0.0, -0.2))
            gf.create_conductor(
                w,
                name="lead",
                start="J",
                end="g1",
                conductor_type="generic",
                cross_section=50e-6,
                lumped_series_resistance_ohm=0.5,
            )
        gf.create_source(w, attached_to="J" if on_junction else "g1", magnitude=2.0)
        return w

    res = solve(world(True), engine)
    z_g1 = solve(world(False), engine).electrode_potentials["g1"][0] / 2.0
    z_in = res.metadata["junction_potentials"]["J"][0] / 2.0
    assert abs(res.electrode_currents["g1"][0] - 2.0) <= 1e-12
    assert all(ps.electrode_name != "J" for ps in res.point_sources)
    assert abs(z_in - (z_g1 + 0.5)) <= 1e-12 * abs(z_in)
