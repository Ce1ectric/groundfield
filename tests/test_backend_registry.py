"""Backend registry, capability checks and the result metadata core.

Finding 10.74: ``Engine.solve`` was an eight-branch if-chain, a backend
could not be added without editing it, unsupported worlds failed deep
inside a backend with messages that named no alternative, and only two
of 32 metadata keys were common to all backends. The registry in
:mod:`groundfield.solver.registry` replaces the chain; ``Engine.solve``
checks the declared capabilities before any backend runs and fills one
metadata core for every result.

Toy backends registered here are removed again in ``finally`` blocks,
so the registry state does not leak between tests.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver.result import FieldResult

SEG = 0.5
BUILTINS = (
    "image",
    "image_2layer",
    "image_nlayer",
    "cim",
    "mom",
    "mom_sommerfeld",
    "bem",
    "fem",
)
CORE_KEYS = {
    "backend",
    "effective_backend",
    "kernel",
    "groundfield_version",
    "frequencies",
    "n_segments",
    "soil_kind",
    "earth_inductive_model",
    "solve_time_s",
    "series",
    "provenance",
}


def _rod_world(soil=None, *, name: str = "registry-test") -> gf.World:
    w = gf.create_world(
        soil=soil if soil is not None else gf.HomogeneousSoil(resistivity=100.0),
        name=name,
    )
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="g1")
    return w


def _three_layer_soil() -> gf.MultiLayerSoil:
    return gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=100.0, thickness=2.0),
            gf.SoilLayer(resistivity=300.0, thickness=4.0),
            gf.SoilLayer(resistivity=50.0),
        ]
    )


def _toy_result(world: gf.World, engine: gf.Engine) -> FieldResult:
    n = len(engine.frequencies)
    return FieldResult(
        backend="toy",
        frequencies=list(engine.frequencies),
        electrode_potentials={e.name: [1.0 + 0j] * n for e in world.electrodes},
        electrode_currents={e.name: [1.0 + 0j] * n for e in world.electrodes},
        metadata={"toy_key": 1},
    )


def _solve_quietly(engine: gf.Engine, world: gf.World) -> FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.solve(world)


# ---------------------------------------------------------------------
# Registry contents and extension
# ---------------------------------------------------------------------


@pytest.mark.finding("10.74")
def test_builtin_backends_are_registered_with_their_capabilities() -> None:
    from groundfield.solver.registry import available_backends, get_backend

    assert set(BUILTINS) <= set(available_backends())
    caps = {name: get_backend(name).capabilities for name in BUILTINS}
    # n-layer soil: only the Sommerfeld MoM and the volume FEM.
    assert {n for n, c in caps.items() if "multi_layer" in c.soils} == {
        "mom_sommerfeld",
        "fem",
    }
    # Concrete-shell correction (ADR-0012): the image family only.
    assert {n for n, c in caps.items() if c.concrete_shells} == {
        "image",
        "image_2layer",
        "image_nlayer",
    }
    # fem reduces every conductor to a lumped branch without inductance.
    assert caps["fem"].conductors == frozenset({"lumped"})
    assert caps["mom"].conductors == frozenset({"lumped", "distributed", "inductive"})
    # fem has no source kernel; every other backend uses one.
    assert caps["fem"].kernels == frozenset()
    assert "point" in caps["mom"].kernels


@pytest.mark.finding("10.74")
def test_new_backend_runs_without_editing_engine_solve() -> None:
    from groundfield.solver.registry import (
        BackendCapabilities,
        register_backend,
        unregister_backend,
    )

    calls: list[str] = []

    def toy_solve(world: gf.World, engine: gf.Engine) -> FieldResult:
        calls.append(world.name)
        return _toy_result(world, engine)

    register_backend(
        "toy_backend",
        toy_solve,
        capabilities=BackendCapabilities(soils={"homogeneous"}),
    )
    try:
        res = _solve_quietly(gf.Engine(backend="toy_backend"), _rod_world())
    finally:
        unregister_backend("toy_backend")
    assert calls == ["registry-test"]
    assert res.metadata["backend"] == "toy_backend"
    assert res.metadata["effective_backend"] == "toy_backend"
    assert res.metadata["toy_key"] == 1  # backend-specific keys are kept


@pytest.mark.finding("10.74")
def test_unknown_backend_name_suggests_the_closest_registered_one() -> None:
    with pytest.raises(ValueError) as excinfo:
        gf.Engine(backend="imgae")
    assert "Did you mean 'image'" in str(excinfo.value)


def test_registering_a_taken_name_is_rejected() -> None:
    from groundfield.solver.registry import BackendCapabilities, register_backend

    with pytest.raises(ValueError, match="already registered"):
        register_backend(
            "image",
            _toy_result,
            capabilities=BackendCapabilities(soils={"homogeneous"}),
        )


# ---------------------------------------------------------------------
# Capability checks before the solve
# ---------------------------------------------------------------------


@pytest.mark.finding("10.74")
def test_capability_error_names_a_backend_that_can_solve_the_world() -> None:
    from groundfield.solver.registry import BackendCapabilityError

    with pytest.raises(BackendCapabilityError) as excinfo:
        gf.Engine(backend="mom", segment_length=SEG).solve(
            _rod_world(_three_layer_soil())
        )
    msg = str(excinfo.value)
    assert "'mom'" in msg and "mom_sommerfeld" in msg
    # Backwards compatible with the exception types the backends used to
    # raise for the same situation (ValueError, NotImplementedError).
    assert isinstance(excinfo.value, ValueError)
    assert isinstance(excinfo.value, NotImplementedError)


@pytest.mark.finding("10.74")
def test_capability_check_runs_before_the_backend() -> None:
    from groundfield.solver.registry import (
        BackendCapabilities,
        BackendCapabilityError,
        register_backend,
        unregister_backend,
    )

    calls: list[str] = []

    def toy_solve(world: gf.World, engine: gf.Engine) -> FieldResult:
        calls.append(world.name)
        return _toy_result(world, engine)

    register_backend(
        "toy_homogeneous",
        toy_solve,
        capabilities=BackendCapabilities(soils={"homogeneous"}),
    )
    try:
        with pytest.raises(BackendCapabilityError, match="two-layer"):
            gf.Engine(backend="toy_homogeneous").solve(
                _rod_world(gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=5.0))
            )
        assert calls == []
        # A stack of equal layers is homogeneous: the conservative check
        # must not reject it (merging equal layers is exact).
        equal = gf.MultiLayerSoil(
            layers=[
                gf.SoilLayer(resistivity=100.0, thickness=2.0),
                gf.SoilLayer(resistivity=100.0, thickness=3.0),
                gf.SoilLayer(resistivity=100.0),
            ]
        )
        _solve_quietly(gf.Engine(backend="toy_homogeneous"), _rod_world(equal))
        assert calls == ["registry-test"]
    finally:
        unregister_backend("toy_homogeneous")


@pytest.mark.finding("10.74")
def test_concrete_shell_capability_error_names_the_image_family() -> None:
    from groundfield.solver.registry import BackendCapabilityError

    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        w,
        "strip",
        name="f1",
        start=(0.0, 0.0, 0.7),
        end=(10.0, 0.0, 0.7),
        concrete_shell_coefficient_ohm_m=100.0,
    )
    gf.create_source(w, attached_to="f1")
    with pytest.raises(BackendCapabilityError) as excinfo:
        gf.Engine(backend="cim", segment_length=SEG).solve(w)
    msg = str(excinfo.value)
    assert "concrete-shell" in msg
    assert "'image'" in msg


@pytest.mark.finding("10.74")
def test_explicit_kernel_request_is_checked_against_the_backend() -> None:
    """A kernel the backend cannot honour raises only when asked for explicitly.

    ``Engine.kernel`` is added by a parallel work stream; the subclass
    stands in for it, so the check is exercised either way.
    """
    from groundfield.solver.registry import (
        BackendCapabilities,
        BackendCapabilityError,
        register_backend,
        unregister_backend,
    )

    class _KernelEngine(gf.Engine):
        kernel: str = "line"

    register_backend(
        "toy_point_only",
        _toy_result,
        capabilities=BackendCapabilities(soils={"homogeneous"}, kernels={"point"}),
    )
    try:
        with pytest.raises(BackendCapabilityError, match="kernel 'line'"):
            _KernelEngine(backend="toy_point_only", kernel="line").solve(_rod_world())
        # Not requested explicitly: the backend's own kernel is used and
        # recorded, without an error.
        res = _solve_quietly(_KernelEngine(backend="toy_point_only"), _rod_world())
        assert res.metadata["kernel"] == "point"
    finally:
        unregister_backend("toy_point_only")


@pytest.mark.finding("10.74")
def test_two_terminal_source_warns_when_the_backend_ignores_return_to() -> None:
    from groundfield.solver.registry import BackendCapabilityWarning

    w = _rod_world()
    gf.create_electrode(w, "rod", name="aux", position=(80.0, 0.0, 0.5), length=1.0)
    w.sources[0].return_to = "aux"
    with pytest.warns(BackendCapabilityWarning, match="return_to"):
        gf.Engine(segment_length=SEG).solve(w)


@pytest.mark.finding("10.74")
def test_fem_warns_when_it_reduces_a_distributed_conductor() -> None:
    from groundfield.solver.registry import BackendCapabilityWarning

    w = _rod_world()
    gf.create_electrode(w, "rod", name="g2", position=(5.0, 0.0, 0.5), length=2.0)
    gf.create_conductor(
        w,
        name="pen",
        start="g1",
        end="g2",
        cross_section=50e-6,
        discretize_segment_length=1.0,
    )
    with pytest.warns(BackendCapabilityWarning, match="distributed"):
        gf.Engine(backend="fem", segment_length=SEG).solve(w)


# ---------------------------------------------------------------------
# Auto-forwarding, aliases, experimental flag
# ---------------------------------------------------------------------


@pytest.mark.finding("10.74")
@pytest.mark.parametrize(
    ("soil", "effective"),
    [
        (gf.HomogeneousSoil(resistivity=100.0), "image"),
        (gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=5.0), "image_2layer"),
        (
            gf.MultiLayerSoil(
                layers=[
                    gf.SoilLayer(resistivity=100.0, thickness=5.0),
                    gf.SoilLayer(resistivity=300.0),
                ]
            ),
            "image_nlayer",
        ),
    ],
)
def test_image_is_auto_forwarded_by_soil_model(soil, effective: str) -> None:
    res = _solve_quietly(gf.Engine(segment_length=SEG), _rod_world(soil))
    assert res.metadata["backend"] == "image"
    assert res.metadata["effective_backend"] == effective
    assert res.backend == effective


@pytest.mark.finding("10.74")
def test_deprecated_alias_warns_with_replacement_and_removal_version() -> None:
    from groundfield.solver.registry import (
        BackendCapabilities,
        available_backends,
        register_backend,
        unregister_backend,
    )

    register_backend(
        "toy_new",
        _toy_result,
        capabilities=BackendCapabilities(soils={"homogeneous"}),
    )
    register_backend("toy_old", None, deprecated_alias_of="toy_new")
    try:
        with pytest.warns(DeprecationWarning) as record:
            res = gf.Engine(backend="toy_old").solve(_rod_world())
        messages = [str(r.message) for r in record if r.category is DeprecationWarning]
        assert any("toy_new" in m and "2027.5" in m for m in messages), messages
        assert res.metadata["backend"] == "toy_old"
        assert res.metadata["effective_backend"] == "toy_new"
        assert "toy_old" not in available_backends()
        assert "toy_old" in available_backends(include_deprecated=True)
    finally:
        unregister_backend("toy_old")
        unregister_backend("toy_new")


@pytest.mark.finding("10.74")
def test_experimental_backend_warns_once_per_process() -> None:
    from groundfield.solver.registry import (
        BackendCapabilities,
        ExperimentalBackendWarning,
        register_backend,
        unregister_backend,
    )

    register_backend(
        "toy_experimental",
        _toy_result,
        capabilities=BackendCapabilities(soils={"homogeneous"}),
        experimental=True,
    )
    try:
        engine = gf.Engine(backend="toy_experimental")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.solve(_rod_world())
            engine.solve(_rod_world())
        hits = [c for c in caught if issubclass(c.category, ExperimentalBackendWarning)]
        assert len(hits) == 1
        assert "toy_experimental" in str(hits[0].message)
    finally:
        unregister_backend("toy_experimental")


# ---------------------------------------------------------------------
# Metadata core
# ---------------------------------------------------------------------


@pytest.mark.finding("10.74")
@pytest.mark.parametrize("backend", BUILTINS)
def test_every_backend_reports_the_metadata_core(backend: str) -> None:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=5.0)
    engine = gf.Engine(backend=backend, segment_length=SEG, frequencies=[0.0, 50.0])
    res = _solve_quietly(engine, _rod_world(soil))
    md = res.metadata
    assert CORE_KEYS <= set(md), sorted(CORE_KEYS - set(md))
    assert md["backend"] == backend
    assert md["effective_backend"] == res.backend
    assert md["groundfield_version"] == gf.__version__
    assert md["frequencies"] == [0.0, 50.0]
    assert md["soil_kind"] == "two_layer"
    assert md["earth_inductive_model"] == "sommerfeld"
    assert isinstance(md["n_segments"], int)
    assert md["n_segments"] == (0 if backend == "fem" else 4)
    assert md["kernel"] == (None if backend == "fem" else "point")
    assert isinstance(md["solve_time_s"], float) and md["solve_time_s"] >= 0.0
    assert md["series"] == {"max_terms": 300, "tol": 1e-6}


def test_core_metadata_keys_constant_lists_the_contract() -> None:
    from groundfield.solver.engine import CORE_METADATA_KEYS

    assert set(CORE_METADATA_KEYS) == CORE_KEYS


def test_backend_specific_metadata_is_kept() -> None:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=5.0)
    res = _solve_quietly(gf.Engine(segment_length=SEG), _rod_world(soil))
    assert {"K", "n_terms_used", "converged", "world_name"} <= set(res.metadata)
    assert np.isclose(res.metadata["K"], 0.5)


def test_replacing_a_registration_keeps_its_position() -> None:
    """How a later stream turns a built-in name into a deprecated alias."""
    from groundfield.solver.registry import (
        BackendCapabilities,
        available_backends,
        get_backend,
        register_backend,
    )

    original = get_backend("bem")
    order = available_backends(include_deprecated=True)
    try:
        register_backend(
            "bem",
            original.solve_fn,
            capabilities=original.capabilities,
            deprecated_alias_of="mom",
            summary="deprecated alias (test)",
            replace=True,
        )
        assert available_backends(include_deprecated=True) == order
        assert "bem" not in available_backends()
        with pytest.warns(DeprecationWarning, match="'mom'"):
            res = gf.Engine(backend="bem", segment_length=SEG).solve(_rod_world())
        assert res.metadata["effective_backend"] == "bem"
        with pytest.raises(ValueError, match="deprecated aliases of 'mom'"):
            register_backend(
                "mom",
                get_backend("mom").solve_fn,
                capabilities=BackendCapabilities(soils={"homogeneous"}),
                deprecated_alias_of="image",
                replace=True,
            )
    finally:
        register_backend(
            "bem",
            original.solve_fn,
            capabilities=original.capabilities,
            summary=original.summary,
            replace=True,
        )
    assert get_backend("bem") == original
    assert available_backends(include_deprecated=True) == order
