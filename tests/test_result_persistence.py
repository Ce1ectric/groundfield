"""Provenance, lossless result JSON and the versioned result file (10.77).

On v0.15.0 a :class:`FieldResult` carried no provenance (which version,
which engine configuration, which world), its JSON lost information
(the float keys and the ``inf`` of ``metadata["penetration_depth"]``),
numpy values in the metadata could not be serialised at all, and there
was no versioned on-disk format.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import stat
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import scipy

import groundfield as gf
from groundfield.solver.result import FieldResult

SEG = 0.5


def _strict_json(text: str):
    """Parse ``text`` as RFC 8259 JSON (no NaN / Infinity tokens)."""

    def _reject(token: str):
        raise ValueError(f"non-standard JSON token {token!r}")

    return json.loads(text, parse_constant=_reject)


def _rod_world(rho: float = 100.0) -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=rho), name="prov")
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(w, attached_to="rod", magnitude=2.0, phase_deg=30.0)
    return w


def _inductive_world() -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0), name="ind")
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="g2", position=(30.0, 0.0, 0.5), length=2.0)
    gf.create_conductor(
        w,
        name="pen",
        start="g1",
        end="g2",
        cross_section=50e-6,
        discretize_segment_length=5.0,
        inductance_model="neumann",
    )
    gf.create_source(w, attached_to="g1", magnitude=10.0)
    return w


def _solve(world: gf.World, **engine_kw) -> FieldResult:
    engine_kw.setdefault("segment_length", SEG)
    return world.solve(gf.create_engine(**engine_kw))


# ---------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------


@pytest.mark.finding("10.77")
def test_every_solve_records_provenance() -> None:
    from groundfield.io import world_fingerprint

    world = _rod_world()
    engine = gf.create_engine(segment_length=SEG, frequencies=[0.0, 50.0])
    before = datetime.now(timezone.utc)
    res = world.solve(engine)
    prov = res.metadata["provenance"]
    assert prov["groundfield_version"] == gf.__version__
    assert prov["python"] == platform.python_version()
    assert prov["numpy"] == np.__version__
    assert prov["scipy"] == scipy.__version__
    assert prov["platform"] == platform.platform()
    assert prov["engine"] == engine.model_dump(mode="json")
    assert prov["world_fingerprint"] == world_fingerprint(world)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", prov["world_fingerprint"])
    stamp = prov["solved_at_utc"]
    assert stamp.endswith("Z")
    solved = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert before - timedelta(seconds=1) <= solved
    assert solved <= datetime.now(timezone.utc) + timedelta(seconds=1)


@pytest.mark.finding("10.77")
def test_world_fingerprint_is_canonical_and_sensitive() -> None:
    from groundfield.io import world_fingerprint

    a, b = _rod_world(), _rod_world()
    assert world_fingerprint(a) == world_fingerprint(b)
    assert world_fingerprint(a) != world_fingerprint(_rod_world(rho=101.0))
    roundtrip = gf.World.model_validate_json(a.model_dump_json())
    assert world_fingerprint(roundtrip) == world_fingerprint(a)


@pytest.mark.finding("10.77")
def test_fingerprint_describes_the_world_as_built(monkeypatch) -> None:
    """A backend that rewrites a source in flight does not change it."""
    import groundfield.solver.image as image_mod
    from groundfield.io import world_fingerprint

    original = image_mod.solve_image

    def mutating(world, engine, **kw):
        world.sources[0].magnitude = 999.0
        return original(world, engine, **kw)

    world = _rod_world()
    expected = world_fingerprint(world)
    monkeypatch.setattr(image_mod, "solve_image", mutating)
    res = _solve(world)
    assert res.metadata["provenance"]["world_fingerprint"] == expected
    assert world_fingerprint(world) == expected  # and the world is restored


# ---------------------------------------------------------------------
# Lossless JSON of FieldResult
# ---------------------------------------------------------------------


@pytest.mark.finding("10.77")
def test_result_json_round_trip_is_lossless() -> None:
    res = _solve(_inductive_world(), frequencies=[0.0, 50.0, 1000.0])
    depth = res.metadata["penetration_depth"]
    assert depth[0.0] == math.inf  # premise: float keys and an inf value
    text = res.model_dump_json()
    _strict_json(text)
    back = FieldResult.model_validate_json(text)
    assert back == res
    assert back.metadata["penetration_depth"] == depth
    assert all(isinstance(k, float) for k in back.metadata["penetration_depth"])


@pytest.mark.finding("10.77")
def test_numpy_and_non_json_metadata_survive_the_round_trip() -> None:
    res = _solve(_rod_world())
    extra = {
        "np_float": np.float64(1.5),
        "np_int": np.int64(3),
        "np_bool": np.bool_(True),
        "array": np.arange(6.0).reshape(2, 3),
        "complex_array": np.array([1.0 + 2.0j, complex("nan")]),
        "int_array": np.array([1, 2, 3], dtype=np.int32),
        "empty": np.zeros((0, 3)),
        "tuple": (1, "a", 2.5),
        "keys": {1: "int", 2.5: "float", (1, 2): "tuple", None: "none"},
        "complex": 1.0 - 2.0j,
        "nan": float("nan"),
        "neg_inf": -math.inf,
        "lookalike": {"$float": "inf"},
        "nested": [{"inf": math.inf}, (np.float32(0.25),)],
    }
    res.metadata.update(extra)
    text = res.model_dump_json()
    _strict_json(text)
    md = FieldResult.model_validate_json(text).metadata

    assert md["np_float"] == 1.5 and type(md["np_float"]) is float
    assert md["np_int"] == 3 and type(md["np_int"]) is int
    assert md["np_bool"] is True
    for key in ("array", "complex_array", "int_array", "empty"):
        assert isinstance(md[key], np.ndarray)
        assert md[key].dtype == extra[key].dtype
        assert md[key].shape == extra[key].shape
        np.testing.assert_array_equal(md[key], extra[key])
    assert md["tuple"] == (1, "a", 2.5)
    assert md["keys"] == extra["keys"]
    assert md["complex"] == 1.0 - 2.0j
    assert math.isnan(md["nan"])
    assert md["neg_inf"] == -math.inf
    assert md["lookalike"] == {"$float": "inf"}
    assert md["nested"] == [{"inf": math.inf}, (0.25,)]


def test_unserialisable_metadata_names_the_offending_key() -> None:
    res = _solve(_rod_world())
    res.metadata["handle"] = object()
    with pytest.raises(Exception, match="handle"):
        res.model_dump_json()


# ---------------------------------------------------------------------
# Versioned result file
# ---------------------------------------------------------------------


@pytest.mark.finding("10.77")
def test_save_and_load_result_round_trip(tmp_path: Path) -> None:
    from groundfield.io import load_result, save_result

    res = _solve(_inductive_world(), frequencies=[0.0, 50.0])
    path = save_result(res, tmp_path / "result.json")
    assert path == tmp_path / "result.json"
    payload = _strict_json(path.read_text(encoding="utf-8"))
    assert payload["format"] == "groundfield.result"
    assert payload["schema_version"] == 1
    assert payload["provenance"]["world_fingerprint"].startswith("sha256:")
    assert payload["provenance"]["groundfield_version"] == gf.__version__
    assert {"format", "schema_version", "provenance", "result"} <= set(payload)

    back = load_result(path)
    assert back == res
    assert back.cluster_impedance("g1") == res.cluster_impedance("g1")
    assert back.input_impedance() == res.input_impedance()
    probes = np.array([[5.0, 0.0, 0.0], [12.0, 3.0, 0.0]])
    np.testing.assert_array_equal(back.potential(probes), res.potential(probes))


@pytest.mark.finding("10.77")
def test_load_result_rejects_a_newer_schema(tmp_path: Path) -> None:
    from groundfield.io import ResultFormatError, load_result, save_result

    path = save_result(_solve(_rod_world()), tmp_path / "r.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResultFormatError, match="newer"):
        load_result(path)


def test_load_result_rejects_foreign_json(tmp_path: Path) -> None:
    from groundfield.io import ResultFormatError, load_result

    bare = tmp_path / "bare.json"
    bare.write_text(_solve(_rod_world()).model_dump_json(), encoding="utf-8")
    with pytest.raises(ResultFormatError, match="groundfield.result"):
        load_result(bare)
    other = tmp_path / "other.json"
    other.write_text('{"format": "something.else", "schema_version": 1}')
    with pytest.raises(ResultFormatError, match="something.else"):
        load_result(other)


def test_save_result_refuses_to_overwrite(tmp_path: Path) -> None:
    from groundfield.io import load_result, save_result

    res = _solve(_rod_world())
    path = save_result(res, tmp_path / "r.json")
    with pytest.raises(FileExistsError):
        save_result(res, path)
    save_result(res, path, overwrite=True)
    assert load_result(path) == res


def test_save_result_writes_one_plain_json_file(tmp_path: Path) -> None:
    from groundfield.io import save_result

    path = save_result(_solve(_rod_world()), tmp_path / "r.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.json"]
    assert path.read_bytes()[:1] == b"{"


def test_hand_built_result_without_provenance_round_trips(tmp_path: Path) -> None:
    from groundfield.io import load_result, save_result

    res = FieldResult(
        backend="hand",
        frequencies=[50.0],
        electrode_potentials={"a": [1.0 + 0j]},
        electrode_currents={"a": [0.5 + 0j]},
    )
    path = save_result(res, tmp_path / "hand.json")
    assert json.loads(path.read_text(encoding="utf-8"))["provenance"] is None
    assert load_result(path) == res


def test_python_version_is_recorded_in_full() -> None:
    res = _solve(_rod_world())
    assert res.metadata["provenance"]["python"].startswith(
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )


# ---------------------------------------------------------------------
# File permissions (audit A1-05)
# ---------------------------------------------------------------------

_POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")


@_POSIX
@pytest.mark.finding("A1-05")
@pytest.mark.parametrize("umask", [0o022, 0o077, 0o002], ids=oct)
def test_result_file_mode_follows_the_umask(tmp_path: Path, umask: int) -> None:
    from groundfield.io import save_result

    old = os.umask(umask)
    try:
        path = save_result(
            FieldResult(backend="hand", frequencies=[50.0]), tmp_path / "r.json"
        )
        plain = tmp_path / "plain.json"
        plain.write_text("{}", encoding="utf-8")
    finally:
        os.umask(old)
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)


@_POSIX
@pytest.mark.finding("A1-05")
def test_overwriting_keeps_the_mode_of_the_replaced_file(tmp_path: Path) -> None:
    from groundfield.io import load_result, save_result

    res = FieldResult(backend="hand", frequencies=[50.0])
    path = save_result(res, tmp_path / "r.json")
    path.chmod(0o640)
    save_result(res, path, overwrite=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert load_result(path) == res


def test_a_failed_write_leaves_the_target_and_no_temporary_file(
    tmp_path: Path, monkeypatch
) -> None:
    from groundfield.io import load_result, save_result
    from groundfield.io import results as results_mod

    old = FieldResult(backend="old", frequencies=[50.0])
    path = save_result(old, tmp_path / "r.json")

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(results_mod.os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        save_result(
            FieldResult(backend="new", frequencies=[50.0]), path, overwrite=True
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.json"]
    assert load_result(path) == old
