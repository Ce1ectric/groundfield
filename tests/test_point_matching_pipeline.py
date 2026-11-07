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
    """A horizontal 5 m segment at 0.7 m warns in both families."""
    from groundfield.solver._guards import ShallowSegmentWarning

    w = gf.create_world(soil=SOILS["homogeneous"])
    gf.create_electrode(
        w, "strip", name="g1", start=(0.0, 0.0, 0.7), end=(10.0, 0.0, 0.7)
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    for backend in ("image", "mom"):
        with pytest.warns(ShallowSegmentWarning):
            gf.create_engine(backend=backend, segment_length=5.0).solve(w)


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
