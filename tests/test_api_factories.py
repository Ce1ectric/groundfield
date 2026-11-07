"""Top-level factories: static typing, actionable errors, engine forwarding.

Finding 10.68 (review pass 10): ``create_electrode(world, kind: str,
**params: Any)`` hid every per-kind parameter from IDEs and mypy, and the
string options of the factories were typed as plain ``str``. The
factories now carry one ``@overload`` per electrode kind (returning the
concrete class) and ``Literal`` types for their string options, and the
runtime errors suggest the closest valid name.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import textwrap
import typing
import warnings
from pathlib import Path

import pytest

import groundfield as gf
from groundfield import api
from groundfield.conductors.conductor import InductanceModel, SoilCoupling
from groundfield.solver.engine import Backend, EarthInductiveModel

_KIND_TO_CLASS = {
    "rod": gf.RodElectrode,
    "ring": gf.RingElectrode,
    "strip": gf.StripElectrode,
    "polyline": gf.PolylineElectrode,
    "star": gf.StarElectrode,
    "mesh": gf.MeshElectrode,
    "grid_mesh": gf.GridMeshElectrode,
}


def _world() -> gf.World:
    # The factories do not need a soil model; only solving does.
    return gf.create_world()


# ---------------------------------------------------------------------
# typed overloads (10.68)
# ---------------------------------------------------------------------


@pytest.mark.finding("10.68")
def test_create_electrode_has_one_overload_per_kind() -> None:
    overloads = typing.get_overloads(api.create_electrode)
    by_kind = {}
    for ov in overloads:
        hints = typing.get_type_hints(ov)
        (kind,) = typing.get_args(hints["kind"])
        by_kind[kind] = hints["return"]
    assert by_kind == _KIND_TO_CLASS
    assert set(typing.get_args(api.ElectrodeKind)) == set(_KIND_TO_CLASS)


@pytest.mark.finding("10.68")
@pytest.mark.parametrize("kind", sorted(_KIND_TO_CLASS))
def test_overload_parameters_match_the_electrode_model(kind: str) -> None:
    """Every overload lists exactly the model's fields, with the same
    required/optional split — so the static view cannot drift from the
    runtime models when a field is added."""
    cls = _KIND_TO_CLASS[kind]
    (ov,) = [
        o
        for o in typing.get_overloads(api.create_electrode)
        if typing.get_args(typing.get_type_hints(o)["kind"]) == (kind,)
    ]
    params = inspect.signature(ov).parameters
    keyword = {
        n: p for n, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    fields = {n: f for n, f in cls.model_fields.items() if n != "kind"}
    assert set(keyword) == set(fields)
    for name, field in fields.items():
        has_default = keyword[name].default is not inspect.Parameter.empty
        assert has_default == (not field.is_required() or name == "name"), name


@pytest.mark.finding("10.68")
def test_string_options_are_literal_types() -> None:
    conductor = typing.get_type_hints(api.create_conductor)
    assert conductor["coupling_to_soil"] == SoilCoupling
    assert conductor["inductance_model"] == InductanceModel | None
    assert conductor["conductor_type"] == gf.ConductorType
    source = typing.get_type_hints(api.create_source)
    assert source["kind"] == api.SourceKind
    assert set(typing.get_args(api.SourceKind)) == {"current", "voltage"}
    engine = typing.get_type_hints(api.create_engine)
    assert engine["backend"] == Backend
    assert engine["earth_inductive_model"] == EarthInductiveModel | None


# ---------------------------------------------------------------------
# runtime errors with suggestions (10.68)
# ---------------------------------------------------------------------


@pytest.mark.finding("10.68")
def test_unknown_electrode_kind_suggests_the_closest() -> None:
    with pytest.raises(ValueError, match=r"'rood'.*Did you mean 'rod'"):
        gf.create_electrode(_world(), "rood", position=(0, 0, 0.5), length=1.0)
    with pytest.raises(ValueError, match=r"Did you mean 'grid_mesh'"):
        gf.create_electrode(_world(), "gridmesh")


@pytest.mark.finding("10.68")
def test_unknown_electrode_keyword_suggests_the_closest() -> None:
    world = _world()
    with pytest.raises(TypeError) as excinfo:
        gf.create_electrode(world, "rod", name="g1", positon=(0, 0, 0.5), length=1.0)
    assert str(excinfo.value) == (
        "create_electrode(kind='rod') got an unexpected keyword argument "
        "'positon' (did you mean 'position'?). Parameters of kind 'rod': "
        "name, position, length, wire_radius."
    )
    assert world.electrodes == []


@pytest.mark.finding("10.68")
def test_keyword_of_another_kind_is_explained() -> None:
    with pytest.raises(TypeError, match=r"'position'.*kind 'ring'.*center"):
        gf.create_electrode(_world(), "ring", position=(0, 0, 0.8), radius=2.0)


@pytest.mark.finding("10.68")
def test_unknown_source_kind_suggests_the_closest() -> None:
    world = _world()
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.0)
    with pytest.raises(ValueError, match=r"'Current'.*Did you mean 'current'"):
        gf.create_source(world, attached_to="g1", kind="Current")  # type: ignore[arg-type]
    assert world.sources == []


@pytest.mark.finding("10.68")
def test_unknown_conductor_end_suggests_the_closest() -> None:
    world = _world()
    gf.create_electrode(world, "rod", name="house_1", position=(0, 0, 0.5), length=1.0)
    gf.create_electrode(world, "rod", name="house_2", position=(5, 0, 0.5), length=1.0)
    with pytest.raises(KeyError, match=r"'house_3'.*Did you mean"):
        gf.create_conductor(world, start="house_1", end="house_3")
    assert world.conductors == []


@pytest.mark.finding("10.68")
def test_valid_calls_return_the_concrete_class() -> None:
    world = _world()
    rod = gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.0)
    ring = gf.create_electrode(world, "ring", center=[5.0, 0.0, 0.8], radius=1.0)
    assert type(rod) is gf.RodElectrode
    assert type(ring) is gf.RingElectrode
    assert ring.name == "electrode_0"  # first free auto-name
    assert world.electrodes == [rod, ring]


# ---------------------------------------------------------------------
# mypy acceptance (10.68): the reviewer's four typos are found
# ---------------------------------------------------------------------

_TYPO_SCRIPT = textwrap.dedent(
    """\
    import groundfield as gf

    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="g1", positon=(0.0, 0.0, 0.5), length=3.0)  # TYPO
    gf.create_electrode(w, "rood", name="g2", position=(5.0, 0.0, 0.5), length=3.0)  # TYPO
    gf.create_conductor(w, start="g1", end="g2", coupling_to_soil="galvanik")  # TYPO
    gf.create_source(w, attached_to="g1", kind="Current")  # TYPO
    rod = gf.create_electrode(w, "rod", name="g3", position=[0.0, 0.0, 0.5], length=3.0)
    ring = gf.create_electrode(w, "ring", center=(0.0, 0.0, 0.8), radius=5.0)
    mesh = gf.create_electrode(w, "grid_mesh", corner=(0, 0, 0.5), size=(10, 10), n_x=3, n_y=3)
    gf.create_conductor(w, start=rod, end=ring, coupling_to_soil="galvanic")
    gf.create_source(w, attached_to=rod, kind="current", magnitude=2.0)
    eng = gf.create_engine(backend="image", earth_inductive_model="sommerfeld")
    reveal_type(rod)
    reveal_type(ring)
    reveal_type(mesh)
    """
)


@pytest.mark.slow
@pytest.mark.finding("10.68")
def test_mypy_finds_the_four_typos(tmp_path: Path) -> None:
    pytest.importorskip("mypy")
    script = tmp_path / "user_typos.py"
    script.write_text(_TYPO_SCRIPT, encoding="utf-8")
    src = Path(gf.__file__).resolve().parents[1]
    env = {**os.environ, "MYPYPATH": str(src)}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-incremental",
            "--follow-imports=silent",
            "--no-error-summary",
            "--hide-error-context",
            "--no-color-output",
            "--cache-dir",
            str(tmp_path / ".mypy_cache"),
            str(script),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=600,
    )
    out = proc.stdout + proc.stderr
    error_lines = {
        int(match.group(1))
        for match in re.finditer(r"user_typos\.py:(\d+): error:", out)
    }
    typo_lines = {
        i
        for i, line in enumerate(_TYPO_SCRIPT.splitlines(), start=1)
        if line.endswith("# TYPO")
    }
    assert error_lines == typo_lines, out
    assert 'Revealed type is "groundfield.geometry.electrodes.RodElectrode"' in out
    assert 'Revealed type is "groundfield.geometry.electrodes.RingElectrode"' in out
    assert 'Revealed type is "groundfield.geometry.electrodes.GridMeshElectrode"' in out


# ---------------------------------------------------------------------
# create_engine forwards only what the caller chose
# ---------------------------------------------------------------------


@pytest.mark.finding("10.68")
def test_create_engine_forwards_only_explicit_arguments() -> None:
    """Defaults come from :class:`Engine` itself (one source of truth), so
    option-level warnings of the engine fire only for options the caller
    actually set, and a changed engine default is not shadowed."""
    eng = gf.create_engine()
    assert eng.model_fields_set <= {"backend"}
    assert eng.frequencies == [50.0]
    assert eng.segment_length == gf.Engine().segment_length
    eng = gf.create_engine(backend="mom", segment_length=0.25, image_max_terms=50)
    assert {"backend", "segment_length", "image_max_terms"} <= eng.model_fields_set
    assert "tolerance" not in eng.model_fields_set
    assert (eng.backend, eng.segment_length, eng.image_max_terms) == ("mom", 0.25, 50)


@pytest.mark.finding("10.68")
def test_create_engine_frequencies_are_validated_by_the_engine() -> None:
    """A numpy array is accepted; a string or a scalar is rejected, never
    iterated ('50' must not become [5.0, 0.0])."""
    import numpy as np

    eng = gf.create_engine(frequencies=np.array([50.0, 150.0]))
    assert eng.frequencies == [50.0, 150.0]
    assert gf.create_engine(frequencies=(50, 1000)).frequencies == [50.0, 1000.0]
    for bad in ("50", 50.0):
        with pytest.raises(ValueError, match="frequencies"):
            gf.create_engine(frequencies=bad)  # type: ignore[arg-type]


@pytest.mark.finding("10.68")
def test_create_engine_accepts_the_point_kernel() -> None:
    eng = gf.create_engine(kernel="point")
    if "kernel" in gf.Engine.model_fields:
        assert eng.kernel == "point"


@pytest.mark.finding("10.68")
def test_create_engine_rejects_unknown_keywords() -> None:
    with pytest.raises(TypeError, match="segment_lenght"):
        gf.create_engine(segment_lenght=0.5)  # type: ignore[call-arg]


# ---------------------------------------------------------------------
# one solve entry point semantics (10.72)
# ---------------------------------------------------------------------


@pytest.mark.finding("10.72")
def test_run_simulation_has_world_solve_semantics(monkeypatch) -> None:
    """``run_simulation`` is ``World.solve``: the world's sources are restored
    even if a backend mutates them in flight, and the result is identical."""
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.0)
    src = gf.create_source(world, attached_to="g1", magnitude=2.0)
    eng = gf.create_engine(segment_length=0.25)
    reference = world.solve(eng)

    real_solve = gf.Engine.solve

    def mutating_solve(self, w):
        w.sources[0].magnitude = 99.0
        return real_solve(self, w)

    monkeypatch.setattr(gf.Engine, "solve", mutating_solve)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gf.run_simulation(world, eng)
    assert src.magnitude == 2.0
    assert world.sources[0] is src
    assert result.backend == reference.backend
    assert inspect.signature(gf.run_simulation).parameters.keys() == {
        "world",
        "engine",
    }
