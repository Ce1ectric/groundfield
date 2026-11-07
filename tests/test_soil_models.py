"""Soil models: canonical parameter names, deprecated aliases, validation.

Naming convention of the model layer (review pass 10, finding 10.69):
physics objects carry bare SI units and the literature symbols — the
resistivity is ``rho`` (Ω·m), a layer thickness ``h`` (m). The pre-2026.11
names ``resistivity`` / ``thickness`` keep working as deprecated aliases
(keyword, JSON key, attribute and ``model_copy(update=...)``), so worlds
written by v0.15.0 still load.

Physical validation (finding 10.67): resistivities and thicknesses must
be finite and positive; ``inf`` used to be accepted and failed much later
inside the solver (``LinAlgError: Singular matrix``).

Inert configuration (finding 10.51 / F57): ``relative_permittivity`` is
not read by any backend (every backend uses the quasi-static
conduction-current formulation), so a non-default value warns.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

import groundfield as gf

INF = math.inf
NAN = math.nan


def _no_warnings(fn, *args, **kwargs):
    """Call ``fn`` and assert that it emits no warning at all."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    assert not caught, [f"{w.category.__name__}: {w.message}" for w in caught]
    return out


# ---------------------------------------------------------------------
# canonical names (10.69)
# ---------------------------------------------------------------------


@pytest.mark.finding("10.69")
def test_homogeneous_soil_uses_rho() -> None:
    soil = _no_warnings(gf.HomogeneousSoil, rho=100.0)
    assert soil.rho == 100.0
    dumped = soil.model_dump()
    assert dumped["rho"] == 100.0
    assert "resistivity" not in dumped


@pytest.mark.finding("10.69")
def test_soil_layer_uses_rho_and_h() -> None:
    layer = _no_warnings(gf.SoilLayer, rho=50.0, h=2.0)
    assert (layer.rho, layer.h) == (50.0, 2.0)
    assert _no_warnings(gf.SoilLayer, rho=50.0).h is None
    assert set(layer.model_dump()) == {"rho", "h", "relative_permittivity"}


@pytest.mark.finding("10.69")
def test_two_layer_soil_names_unchanged() -> None:
    soil = _no_warnings(gf.TwoLayerSoil, rho_1=100.0, rho_2=500.0, h_1=2.0)
    assert (soil.rho_1, soil.rho_2, soil.h_1) == (100.0, 500.0, 2.0)


@pytest.mark.finding("10.69")
def test_json_schema_names_the_canonical_fields() -> None:
    hom = gf.HomogeneousSoil.model_json_schema()["properties"]
    layer = gf.SoilLayer.model_json_schema()["properties"]
    assert "rho" in hom and "resistivity" not in hom
    assert {"rho", "h"} <= set(layer) and "thickness" not in layer


# ---------------------------------------------------------------------
# deprecated aliases (10.69)
# ---------------------------------------------------------------------


_OLD_KEYWORDS = [
    (gf.HomogeneousSoil, {"resistivity": 120.0}, {"rho": 120.0}),
    (gf.SoilLayer, {"resistivity": 30.0}, {"rho": 30.0}),
    (gf.SoilLayer, {"rho": 30.0, "thickness": 4.0}, {"rho": 30.0, "h": 4.0}),
    (
        gf.SoilLayer,
        {"resistivity": 30.0, "thickness": 4.0},
        {"rho": 30.0, "h": 4.0},
    ),
]


@pytest.mark.finding("10.69")
@pytest.mark.parametrize("cls, old_kwargs, expected", _OLD_KEYWORDS)
def test_old_keyword_warns_and_maps_to_canonical_field(
    cls, old_kwargs, expected
) -> None:
    with pytest.warns(DeprecationWarning) as record:
        obj = cls(**old_kwargs)
    for name, value in expected.items():
        assert getattr(obj, name) == value
    messages = [str(w.message) for w in record]
    old_names = sorted(set(old_kwargs) - set(expected))
    for old in old_names:
        new = {"resistivity": "rho", "thickness": "h"}[old]
        assert any(
            f"'{old}'" in m and f"'{new}'" in m and "2027.5" in m for m in messages
        ), messages


@pytest.mark.finding("10.69")
def test_deprecation_warning_points_at_the_caller() -> None:
    with pytest.warns(DeprecationWarning) as record:
        gf.HomogeneousSoil(resistivity=100.0)
    assert record[0].filename == __file__


@pytest.mark.finding("10.69")
def test_old_and_new_name_together_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"rho.*resistivity|resistivity.*rho"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gf.HomogeneousSoil(rho=100.0, resistivity=200.0)
    with pytest.raises(ValueError, match=r"thickness"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gf.SoilLayer(rho=10.0, h=1.0, thickness=2.0)


@pytest.mark.finding("10.69")
def test_old_attribute_names_read_and_write_with_warning() -> None:
    soil = gf.HomogeneousSoil(rho=100.0)
    with pytest.warns(DeprecationWarning, match=r"'resistivity'.*'rho'"):
        assert soil.resistivity == 100.0
    with pytest.warns(DeprecationWarning, match=r"'resistivity'.*'rho'"):
        soil.resistivity = 250.0
    assert soil.rho == 250.0

    layer = gf.SoilLayer(rho=10.0, h=3.0)
    with pytest.warns(DeprecationWarning, match=r"'thickness'.*'h'"):
        assert layer.thickness == 3.0
    with pytest.warns(DeprecationWarning, match=r"'resistivity'.*'rho'"):
        assert layer.resistivity == 10.0
    with pytest.warns(DeprecationWarning, match=r"'thickness'.*'h'"):
        layer.thickness = 5.0
    assert layer.h == 5.0


@pytest.mark.finding("10.69")
def test_model_copy_with_old_name_is_translated() -> None:
    """``model_copy(update={"resistivity": ...})`` must not be a silent no-op.

    pydantic's ``model_copy`` does not validate ``update``; without the
    translation the old key would land in ``__dict__`` next to ``rho`` and
    a sweep written against v0.15.0 would silently keep the old value.
    """
    soil = gf.HomogeneousSoil(rho=100.0)
    with pytest.warns(DeprecationWarning, match="resistivity"):
        copy = soil.model_copy(update={"resistivity": 300.0})
    assert copy.rho == 300.0
    assert "resistivity" not in copy.__dict__
    layer = gf.SoilLayer(rho=10.0, h=2.0)
    with pytest.warns(DeprecationWarning, match="thickness"):
        assert layer.model_copy(update={"thickness": 7.0}).h == 7.0
    assert _no_warnings(soil.model_copy, update={"rho": 5.0}).rho == 5.0


@pytest.mark.finding("10.68")
def test_model_copy_rejects_unknown_fields() -> None:
    """A misspelled key in ``model_copy(update=...)`` was a silent no-op."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=10.0, h_1=2.0)
    with pytest.raises(ValueError, match=r"'rho1'.*Did you mean 'rho_1'"):
        soil.model_copy(update={"rho1": 5.0})
    with pytest.raises(ValueError, match=r"'thikness'"):
        gf.SoilLayer(rho=1.0).model_copy(update={"thikness": 2.0})


_V0150_WORLD = {
    "name": "v0150",
    "soil": {
        "kind": "multi_layer",
        "layers": [
            {"resistivity": 100.0, "thickness": 2.0, "relative_permittivity": 10.0},
            {"resistivity": 300.0, "thickness": None, "relative_permittivity": 10.0},
        ],
    },
    "electrodes": [],
    "conductors": [],
    "sources": [],
    "boundary": {
        "far_field": "dirichlet",
        "surface": "neumann",
        "reference_node": None,
    },
    "concrete_shell_corrections": {},
}


@pytest.mark.finding("10.69")
def test_world_json_written_by_v0150_loads() -> None:
    # Homogeneous soil: the canonical world JSON of a v0.15.0 golden case.
    from tests.golden import harness

    stored = str(harness.load_case("gen_layout_builder")["world/json"])
    assert '"resistivity": 100.0' in stored
    with pytest.warns(DeprecationWarning, match="resistivity"):
        world = gf.World.model_validate_json(stored)
    assert isinstance(world.soil, gf.HomogeneousSoil)
    assert world.soil.rho == 100.0
    assert len(world.electrodes) > 0
    # Multi-layer soil with the old layer keys.
    with pytest.warns(DeprecationWarning):
        world = gf.World.model_validate_json(json.dumps(_V0150_WORLD))
    layers = world.soil.layers
    assert [(ly.rho, ly.h) for ly in layers] == [(100.0, 2.0), (300.0, None)]
    # Round trip with the canonical names is silent and lossless.
    again = _no_warnings(gf.World.model_validate_json, world.model_dump_json())
    assert again == world
    assert '"thickness"' not in world.model_dump_json()


@pytest.mark.finding("10.69")
def test_internal_callers_do_not_trigger_deprecation_warnings() -> None:
    """groundfield code that still uses an old name must not warn the user.

    Such a warning is not actionable for the user. Simulated by running the
    construction with module globals that claim to be a groundfield module.
    """
    code = (
        "soil = HomogeneousSoil(resistivity=100.0)\n"
        "value = soil.resistivity\n"
        "layer = SoilLayer(resistivity=5.0, thickness=1.0)\n"
    )
    namespace = {
        "__name__": "groundfield._internal_caller_test",
        "HomogeneousSoil": gf.HomogeneousSoil,
        "SoilLayer": gf.SoilLayer,
    }
    _no_warnings(exec, code, namespace)
    assert namespace["value"] == 100.0
    assert namespace["layer"].h == 1.0


# ---------------------------------------------------------------------
# physical validation (10.67)
# ---------------------------------------------------------------------


_FINITE_POSITIVE = [
    ("rho", lambda v: gf.HomogeneousSoil(rho=v)),
    ("resistivity", lambda v: gf.HomogeneousSoil(resistivity=v)),
    ("rho_1", lambda v: gf.TwoLayerSoil(rho_1=v, rho_2=100.0, h_1=2.0)),
    ("rho_2", lambda v: gf.TwoLayerSoil(rho_1=100.0, rho_2=v, h_1=2.0)),
    ("h_1", lambda v: gf.TwoLayerSoil(rho_1=100.0, rho_2=10.0, h_1=v)),
    ("rho", lambda v: gf.SoilLayer(rho=v)),
    ("h", lambda v: gf.SoilLayer(rho=10.0, h=v)),
    ("thickness", lambda v: gf.SoilLayer(resistivity=10.0, thickness=v)),
    (
        "relative_permittivity",
        lambda v: gf.HomogeneousSoil(rho=10.0, relative_permittivity=v),
    ),
]


@pytest.mark.finding("10.67")
@pytest.mark.parametrize("value", [INF, -INF, NAN, 0.0, -5.0])
@pytest.mark.parametrize(
    "param, build", _FINITE_POSITIVE, ids=[p for p, _ in _FINITE_POSITIVE]
)
def test_soil_parameters_must_be_finite_and_positive(param, build, value) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match=param):
            build(value)


@pytest.mark.finding("10.67")
def test_infinite_resistivity_fails_at_construction_not_in_the_solver() -> None:
    """v0.15.0 accepted ``inf`` and failed in the solve (``LinAlgError``)."""
    with pytest.raises(ValueError, match="finite"):
        gf.HomogeneousSoil(rho=INF)


@pytest.mark.finding("10.67")
def test_assignment_is_validated() -> None:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=10.0, h_1=2.0)
    with pytest.raises(ValueError, match="h_1"):
        soil.h_1 = INF
    with pytest.raises(ValueError, match="rho_1"):
        soil.rho_1 = -1.0
    assert soil.h_1 == 2.0 and soil.rho_1 == 100.0
    hom = gf.HomogeneousSoil(rho=100.0)
    with pytest.raises(ValueError, match="rho"):
        hom.rho = NAN


@pytest.mark.finding("10.67")
def test_multilayer_structure_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        gf.MultiLayerSoil(layers=[])
    with pytest.raises(ValueError, match=r"single layer.*h=None.*HomogeneousSoil"):
        gf.MultiLayerSoil(layers=[gf.SoilLayer(rho=100.0, h=5.0)])
    with pytest.raises(ValueError, match=r"layer 2 of 3.*h=None.*only the last"):
        gf.MultiLayerSoil(
            layers=[
                gf.SoilLayer(rho=100.0, h=1.0),
                gf.SoilLayer(rho=50.0),
                gf.SoilLayer(rho=20.0),
            ]
        )
    with pytest.raises(ValueError, match=r"last layer.*half-space.*h=None"):
        gf.MultiLayerSoil(
            layers=[gf.SoilLayer(rho=100.0, h=1.0), gf.SoilLayer(rho=50.0, h=3.0)]
        )


@pytest.mark.finding("10.67")
def test_single_layer_multilayer_soil_is_a_valid_half_space() -> None:
    """A one-layer stack is the homogeneous half-space; it stays accepted."""
    soil = _no_warnings(gf.MultiLayerSoil, layers=[gf.SoilLayer(rho=100.0)])
    assert soil.layers[0].h is None


# ---------------------------------------------------------------------
# inert relative permittivity (10.51 / F57)
# ---------------------------------------------------------------------


@pytest.mark.finding("10.51")
@pytest.mark.parametrize(
    "build",
    [
        lambda: gf.HomogeneousSoil(rho=100.0, relative_permittivity=81.0),
        lambda: gf.TwoLayerSoil(
            rho_1=100.0, rho_2=10.0, h_1=2.0, relative_permittivity=81.0
        ),
        lambda: gf.SoilLayer(rho=100.0, relative_permittivity=81.0),
    ],
    ids=["homogeneous", "two_layer", "layer"],
)
def test_non_default_relative_permittivity_warns(build) -> None:
    with pytest.warns(UserWarning, match="relative_permittivity"):
        build()


@pytest.mark.finding("10.51")
def test_relative_permittivity_warning_category() -> None:
    from groundfield.utils import InertConfigurationWarning

    with pytest.warns(InertConfigurationWarning, match="no backend"):
        gf.HomogeneousSoil(rho=100.0, relative_permittivity=1.0)


@pytest.mark.finding("10.51")
def test_default_relative_permittivity_is_silent() -> None:
    _no_warnings(gf.HomogeneousSoil, rho=100.0)
    _no_warnings(gf.HomogeneousSoil, rho=100.0, relative_permittivity=10.0)
    _no_warnings(gf.TwoLayerSoil, rho_1=1.0, rho_2=2.0, h_1=3.0)
    _no_warnings(gf.SoilLayer, rho=1.0)


@pytest.mark.finding("10.51")
def test_relative_permittivity_does_not_change_results() -> None:
    """The evidence behind the warning: ε_r = 1 and 81 solve bit-identically."""

    def solve(eps_r: float) -> complex:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            soil = gf.HomogeneousSoil(rho=100.0, relative_permittivity=eps_r)
        world = gf.create_world(soil=soil)
        gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=2.0)
        gf.create_source(world, attached_to="g1", magnitude=1.0)
        eng = gf.create_engine(segment_length=0.5, frequencies=[50.0, 1000.0])
        return np.asarray(world.solve(eng).cluster_impedance("g1"))

    assert np.array_equal(solve(1.0), solve(81.0))
