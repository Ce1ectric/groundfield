"""Deprecated engine names and the experimental ``fem`` (finding 10.76).

Review pass 10 found eight backend names for about three distinct
computations: ``bem`` assembled the identical point-matching system as
``mom`` (0.0 to 2.3e-16 apart), ``cim`` evaluated the closed-form image
kernels (bit-identical to ``image`` / ``image_2layer`` since 10.30), and
``image_nlayer`` only routed to them. The user decided: ``cim``, ``bem``
and ``image_nlayer`` become deprecated aliases that keep working, warn
with ``DeprecationWarning`` (removal not before 2027.5) and run their
replacement; ``fem`` is registered as experimental.

The old modules stay importable; :func:`~groundfield.solver.cim.solve_cim`
and :func:`~groundfield.solver.bem.solve_bem` delegate and warn. The
registry entries live in ``groundfield.solver.registry`` (requested from
the coordinator, see the eng2 report): the strict xfails below flip when
that patch lands, and the ``requested_registry`` tests check the patch
itself.
"""

from __future__ import annotations

import warnings

import pytest

import groundfield as gf

SEG = 0.5

SOILS = {
    "homogeneous": gf.HomogeneousSoil(rho=100.0),
    "two_layer": gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=5.0),
    "cross_layer": gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0),
}

#: (alias, soil label, backend whose numbers the alias must reproduce)
CASES = [
    ("bem", "homogeneous", "mom"),
    ("bem", "two_layer", "mom"),
    ("bem", "cross_layer", "mom"),
    ("cim", "homogeneous", "image"),
    ("cim", "two_layer", "image_2layer"),
    ("cim", "cross_layer", "image_2layer"),
]


def _world(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="A", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_electrode(w, "ring", name="B", center=(4.0, 0.0, 0.8), radius=1.5)
    gf.create_conductor(
        w, name="ab", start="A", end="B", lumped_series_resistance_ohm=0.3
    )
    gf.create_source(w, attached_to="A", magnitude=1.0)
    return w


def _quiet(engine: gf.Engine, world: gf.World) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.solve(world)


def _deprecations(record) -> list[str]:
    return [
        str(r.message) for r in record if issubclass(r.category, DeprecationWarning)
    ]


# ---------------------------------------------------------------------
# The aliases run their replacement and warn
# ---------------------------------------------------------------------


@pytest.mark.finding("10.76")
@pytest.mark.parametrize(("alias", "soil_label", "target"), CASES)
def test_deprecated_name_runs_its_replacement_and_warns(
    alias, soil_label, target
) -> None:
    world = _world(SOILS[soil_label])
    with pytest.warns(DeprecationWarning) as record:
        res = gf.Engine(backend=alias, segment_length=SEG).solve(world)
    replacement = "mom" if alias == "bem" else "image"
    messages = _deprecations(record)
    assert any(
        f"'{alias}'" in m and f"'{replacement}" in m and "2027.5" in m for m in messages
    ), messages
    ref = _quiet(
        gf.Engine(backend=target, segment_length=SEG), _world(SOILS[soil_label])
    )
    # The replacement's own numbers, bit for bit.
    assert res.electrode_potentials == ref.electrode_potentials
    assert res.electrode_currents == ref.electrode_currents
    assert [p.current for p in res.point_sources] == [
        p.current for p in ref.point_sources
    ]
    assert res.metadata["backend"] == alias


@pytest.mark.finding("10.76")
def test_direct_calls_of_the_old_solvers_warn_and_delegate() -> None:
    from groundfield.solver.bem import solve_bem
    from groundfield.solver.cim import solve_cim
    from groundfield.solver.image_2layer import solve_image_2layer
    from groundfield.solver.mom import solve_mom

    engine = gf.Engine(segment_length=SEG)
    soil = SOILS["two_layer"]
    with pytest.warns(DeprecationWarning, match=r"'mom'.*2027\.5"):
        res_bem = solve_bem(_world(soil), engine)
    with pytest.warns(DeprecationWarning, match=r"'image'.*2027\.5"):
        res_cim = solve_cim(_world(soil), engine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_mom = solve_mom(_world(soil), engine)
        ref_il2 = solve_image_2layer(
            _world(soil),
            engine,
            max_terms=engine.image_max_terms,
            tol=engine.image_series_tol,
        )
    assert res_bem.electrode_potentials == ref_mom.electrode_potentials
    assert res_cim.electrode_potentials == ref_il2.electrode_potentials
    assert (res_bem.backend, res_bem.metadata["dispatched_to"]) == ("bem", "mom")
    assert (res_cim.backend, res_cim.metadata["dispatched_to"]) == (
        "cim",
        "image_2layer",
    )


def test_old_modules_stay_importable() -> None:
    """Guard: the alias modules and the public spectral helper remain."""
    from groundfield.solver import bem, cim, image_nlayer

    assert callable(bem.solve_bem)
    assert callable(cim.solve_cim)
    assert callable(cim.fit_complex_images)
    assert callable(image_nlayer.solve_image_nlayer)
    assert {"ComplexImageFit", "ComplexImageFitWarning", "fit_complex_images"} <= set(
        cim.__all__
    )


@pytest.mark.parametrize("alias", ["cim", "bem"])
def test_three_or_more_layers_still_point_at_mom_sommerfeld(alias) -> None:
    """Guard: a genuine three-layer stack raises and names the engine that can."""
    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=100.0, h=2.0),
            gf.SoilLayer(rho=400.0, h=3.0),
            gf.SoilLayer(rho=50.0, h=None),
        ]
    )
    with pytest.raises(NotImplementedError, match="mom_sommerfeld"):
        _quiet(gf.Engine(backend=alias, segment_length=SEG), _world(soil))


# ---------------------------------------------------------------------
# compare_engines does not count an alias as a second engine
# ---------------------------------------------------------------------


@pytest.mark.finding("10.76")
def test_compare_engines_notes_that_an_alias_is_no_independent_check() -> None:
    engines = {
        "cim": gf.Engine(backend="cim", segment_length=SEG),
        "image_2layer": gf.Engine(backend="image_2layer", segment_length=SEG),
        "mom": gf.Engine(backend="mom", segment_length=SEG),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = gf.compare_engines(_world(SOILS["two_layer"]), engines)
    same = [n for n in rep.notes if "same computation" in n]
    assert len(same) == 1, rep.notes
    assert "'cim'" in same[0] and "'image_2layer'" in same[0]
    assert "'mom'" not in same[0]


# ---------------------------------------------------------------------
# Registry state (strict xfail until the requested registry patch lands)
# ---------------------------------------------------------------------

REGISTRY_PATCH = pytest.mark.xfail(
    strict=True,
    reason="registry entries requested from the coordinator (eng2 report); "
    "remove this marker together with the registry patch",
)

#: alias -> replacement of the requested registry entries
REQUESTED_ALIASES = {"cim": "image", "bem": "mom", "image_nlayer": "image"}


@REGISTRY_PATCH
@pytest.mark.finding("10.76")
def test_registry_lists_the_deprecated_names_as_aliases() -> None:
    from groundfield.solver.registry import available_backends, get_backend

    for name, target in REQUESTED_ALIASES.items():
        assert get_backend(name).deprecated_alias_of == target
        assert name not in available_backends()
        assert name in available_backends(include_deprecated=True)


@REGISTRY_PATCH
@pytest.mark.finding("10.75")
@pytest.mark.parametrize("name", ["mom", "mom_sommerfeld"])
def test_mom_family_declares_two_terminal_sources_and_junctions(name) -> None:
    """The ported mom family runs the shared inject / build_topology steps."""
    from groundfield.solver.registry import get_backend

    caps = get_backend(name).capabilities
    assert caps.two_terminal_sources
    assert caps.junctions


@REGISTRY_PATCH
@pytest.mark.finding("10.76")
def test_fem_is_registered_as_experimental() -> None:
    from groundfield.solver.registry import get_backend

    assert get_backend("fem").experimental


@REGISTRY_PATCH
@pytest.mark.finding("10.76")
def test_image_nlayer_name_warns_while_the_image_dispatch_does_not() -> None:
    soil = gf.MultiLayerSoil(
        layers=[gf.SoilLayer(rho=100.0, h=5.0), gf.SoilLayer(rho=300.0, h=None)]
    )
    with pytest.warns(DeprecationWarning, match=r"'image'.*2027\.5"):
        gf.Engine(backend="image_nlayer", segment_length=SEG).solve(_world(soil))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        gf.Engine(backend="image", segment_length=SEG).solve(_world(soil))
    assert _deprecations(record) == []


# ---------------------------------------------------------------------
# The requested registry patch, applied in a fixture
# ---------------------------------------------------------------------

_PATCHED = ("mom", "mom_sommerfeld", "image_nlayer", "cim", "bem", "fem")


def apply_requested_registry_patch() -> None:
    """The entries of ``registry._register_builtins`` requested by eng2.

    ``cim`` and ``bem`` become pure aliases (``solve_fn=None``): the
    registry warns and runs the replacement, so the wrappers
    ``solve_cim`` / ``solve_bem`` are not called and no second warning
    is emitted. ``image_nlayer`` keeps its own solve function, because
    ``image`` forwards ``MultiLayerSoil`` worlds to it. ``fem`` becomes
    experimental.
    """
    from dataclasses import replace

    from groundfield.solver import registry as r

    # mom and mom_sommerfeld run the shared inject / build_topology steps
    # (10.75): two-terminal sources and junction nodes (CORE-2a flags,
    # present once CORE-2a is merged).
    flags = {"two_terminal_sources": True}
    if "junctions" in r.BackendCapabilities.__dataclass_fields__:
        flags["junctions"] = True
    for name in ("mom", "mom_sommerfeld"):
        spec = r.get_backend(name)
        r.register_backend(
            name,
            spec.solve_fn,
            capabilities=replace(spec.capabilities, **flags),
            options_model=spec.options_model,
            summary=spec.summary,
            replace=True,
        )
    nlayer = r.get_backend("image_nlayer")
    r.register_backend(
        "image_nlayer",
        nlayer.solve_fn,
        capabilities=nlayer.capabilities,
        deprecated_alias_of="image",
        summary="Deprecated alias of the image auto-dispatch (n <= 2).",
        replace=True,
    )
    r.register_backend(
        "cim",
        None,
        deprecated_alias_of="image",
        summary="Deprecated alias of the image auto-dispatch.",
        replace=True,
    )
    r.register_backend(
        "bem",
        None,
        deprecated_alias_of="mom",
        summary="Deprecated alias of mom (identical point-matching system).",
        replace=True,
    )
    fem = r.get_backend("fem")
    r.register_backend(
        "fem",
        fem.solve_fn,
        capabilities=fem.capabilities,
        options_model=fem.options_model,
        experimental=True,
        summary=fem.summary,
        replace=True,
    )


@pytest.fixture
def requested_registry():
    from groundfield.solver import registry as r

    saved = {name: r.get_backend(name) for name in _PATCHED}
    apply_requested_registry_patch()
    try:
        yield
    finally:
        for name in _PATCHED:
            spec = saved[name]
            r.register_backend(
                name,
                spec.solve_fn,
                capabilities=spec.capabilities,
                options_model=spec.options_model,
                aliases=spec.aliases,
                deprecated_alias_of=spec.deprecated_alias_of,
                experimental=spec.experimental,
                forward=dict(spec.forward),
                summary=spec.summary,
                replace=True,
            )


@pytest.mark.parametrize(("alias", "soil_label", "target"), CASES)
def test_requested_registry_runs_the_replacement_with_one_warning(
    requested_registry, alias, soil_label, target
) -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        res = gf.Engine(backend=alias, segment_length=SEG).solve(
            _world(SOILS[soil_label])
        )
    messages = _deprecations(record)
    assert len(messages) == 1, messages
    assert "2027.5" in messages[0]
    assert res.metadata["backend"] == alias
    assert res.metadata["effective_backend"] == target == res.backend
    ref = _quiet(
        gf.Engine(backend=target, segment_length=SEG), _world(SOILS[soil_label])
    )
    assert res.electrode_potentials == ref.electrode_potentials


def test_requested_registry_keeps_the_image_forwarding_silent(
    requested_registry,
) -> None:
    from groundfield.solver.registry import available_backends

    soil = gf.MultiLayerSoil(
        layers=[gf.SoilLayer(rho=100.0, h=5.0), gf.SoilLayer(rho=300.0, h=None)]
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        res = gf.Engine(backend="image", segment_length=SEG).solve(_world(soil))
    assert _deprecations(record) == []
    assert res.metadata["effective_backend"] == "image_nlayer"
    assert not set(REQUESTED_ALIASES) & set(available_backends())


@pytest.mark.parametrize("alias", ["cim", "bem", "image_nlayer"])
def test_requested_registry_rejects_three_layers_naming_mom_sommerfeld(
    requested_registry, alias
) -> None:
    from groundfield.solver.registry import BackendCapabilityError

    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=100.0, h=2.0),
            gf.SoilLayer(rho=400.0, h=3.0),
            gf.SoilLayer(rho=50.0, h=None),
        ]
    )
    with pytest.raises(BackendCapabilityError, match="mom_sommerfeld"):
        _quiet(gf.Engine(backend=alias, segment_length=SEG), _world(soil))


def test_requested_registry_marks_fem_experimental_once(requested_registry) -> None:
    from groundfield.solver.registry import ExperimentalBackendWarning

    w = gf.create_world(soil=SOILS["homogeneous"])
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    engine = gf.Engine(backend="fem", segment_length=SEG)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        engine.solve(w)
        engine.solve(w)
    hits = [r for r in record if issubclass(r.category, ExperimentalBackendWarning)]
    assert len(hits) == 1
