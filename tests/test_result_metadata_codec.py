"""Validation of the tagged metadata codec of result files (audit A1-01).

``FieldResult.metadata`` is stored as standard JSON with tagged objects
(``$float``, ``$complex``, ``$tuple``, ``$dict``, ``$ndarray``). The
reader must only accept what the writer can write: an untrusted result
file of a few hundred bytes must not allocate megabytes, create object /
datetime / void arrays, or escape with a bare ``KeyError`` /
``TypeError``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pydantic
import pytest

from groundfield.io import ResultFormatError, load_result, save_result
from groundfield.solver.result import FieldResult

#: Errors a malformed result file may raise (``ResultFormatError`` and
#: ``pydantic.ValidationError`` both subclass ``ValueError``).
MALFORMED = (ResultFormatError, pydantic.ValidationError)


def _crafted(tmp_path: Path, value: object, name: str = "crafted.json") -> Path:
    """A valid result file whose metadata holds ``{"x": value}`` verbatim."""
    base = save_result(
        FieldResult(backend="image", frequencies=[50.0]),
        tmp_path / "base.json",
        overwrite=True,
    )
    envelope = json.loads(base.read_text(encoding="utf-8"))
    envelope["result"]["metadata"] = {"x": value}
    path = tmp_path / name
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return path


def _ndarray(dtype: object, shape: object, data: object) -> dict:
    return {"$ndarray": {"dtype": dtype, "shape": shape, "data": data}}


def _names(tag: str) -> str:
    """Pattern of the decoder's message for a bad ``tag`` under key ``x``.

    pydantic repeats the input value in its message, so matching the tag
    alone would not show that the decoder itself raised.
    """
    return rf"metadata\['x'\]: {re.escape(tag)} "


# ---------------------------------------------------------------------
# Memory amplification
# ---------------------------------------------------------------------


@pytest.mark.finding("A1-01")
@pytest.mark.parametrize(
    "payload",
    [
        # 4 MiB from one element of an allowed kind (unicode)
        _ndarray("<U1048576", [1], ["x"]),
        # 8 MiB from one element: a sub-array dtype of kind f
        _ndarray("(1024,1024)<f8", [1, 1024, 1024], [0.0]),
        # one element, but the shape claims 2**20 of them
        _ndarray("<f8", [1024, 1024], [0.0]),
        # 2 000 empty strings in 4 KiB characters each: 32 MiB from 6 kB
        _ndarray("<U4096", [2000], [""] * 2000),
    ],
    ids=["wide_unicode", "subarray", "shape_exceeds_data", "padded_strings"],
)
def test_a_small_file_cannot_allocate_a_large_array(tmp_path, payload) -> None:
    path = _crafted(tmp_path, payload)
    assert path.stat().st_size < 10_000
    with pytest.raises(MALFORMED, match=_names("$ndarray")):
        load_result(path)


def test_decoded_arrays_stay_linear_in_the_file_size(tmp_path) -> None:
    """Control: legitimate arrays load and stay within 64 bytes per file byte."""
    payloads = [
        _ndarray("<U8", [3], ["a", "bb", ""]),
        _ndarray("<U64", [1], [""]),  # within the per-array allowance
        _ndarray("<c16", [4], [{"$complex": [0, 0]}] * 4),
        _ndarray("|b1", [2, 2], [True, False, True, True]),
        _ndarray("<u8", [2], [0, 2**64 - 1]),
        _ndarray("<f8", [0, 1000], []),
        [_ndarray("<U1024", [1], [""]) for _ in range(3)],
    ]
    for i, payload in enumerate(payloads):
        path = _crafted(tmp_path, payload, name=f"ok{i}.json")
        value = load_result(path).metadata["x"]
        arrays = value if isinstance(value, list) else [value]
        nbytes = sum(a.nbytes for a in arrays)
        assert nbytes <= 64 * path.stat().st_size, (payload, nbytes)


# ---------------------------------------------------------------------
# Only what the writer writes
# ---------------------------------------------------------------------


@pytest.mark.finding("A1-01")
@pytest.mark.parametrize(
    "dtype, data",
    [
        ("|O", [1, "a"]),
        ("<M8[s]", [0]),
        ("<m8[s]", [0]),
        ("|V8", [0]),
        ("|S5", ["abc"]),
        ("[('a', '<f8')]", [0.0]),
        ("{'names': ['a'], 'formats': ['<f8']}", [0.0]),
        ("(2,)<f8", [0.0, 1.0]),
        ("<U0", ["a"]),
        ("<f16", [0.0]),
        ("<c32", [0.0]),
        ("f3", [0.0]),
        (8, [0.0]),
        (None, [0.0]),
    ],
    ids=[
        "object",
        "datetime",
        "timedelta",
        "void",
        "bytes",
        "structured_list",
        "structured_dict",
        "subarray",
        "zero_width_unicode",
        "longdouble",
        "clongdouble",
        "unknown",
        "integer",
        "null",
    ],
)
def test_only_dtypes_the_writer_writes_are_read(tmp_path, dtype, data) -> None:
    path = _crafted(tmp_path, _ndarray(dtype, [len(data)], data))
    with pytest.raises(MALFORMED, match=_names("$ndarray")):
        load_result(path)


@pytest.mark.finding("A1-01")
@pytest.mark.parametrize(
    "dtype, data",
    [
        ("<i8", [1.5]),  # numpy would truncate silently
        ("|b1", [2]),
        ("<f8", ["1.0"]),
        ("<f8", [[1.0], [2.0]]),  # nested lists
        ("<U3", [1]),
        ("<c16", ["1+2j"]),
        ("<f8", [_ndarray("<f8", [1], [0.0])]),  # an array as an item
        ("|u1", [300]),  # out of range
        ("<i8", [2**70]),
        ("<f8", [10**400]),
    ],
    ids=[
        "float_in_int",
        "int_in_bool",
        "str_in_float",
        "nested_list",
        "int_in_str",
        "str_in_complex",
        "array_item",
        "uint8_overflow",
        "int64_overflow",
        "float_overflow",
    ],
)
def test_items_must_match_the_dtype(tmp_path, dtype, data) -> None:
    path = _crafted(tmp_path, _ndarray(dtype, [len(data)], data))
    with pytest.raises(MALFORMED, match=_names("$ndarray")):
        load_result(path)


@pytest.mark.finding("A1-01")
@pytest.mark.parametrize(
    "payload, tag",
    [
        ({"$ndarray": {"dtype": "<f8", "shape": [1]}}, "$ndarray"),
        (
            {"$ndarray": {"dtype": "<f8", "shape": [1], "data": [0.0], "x": 1}},
            "$ndarray",
        ),
        ({"$ndarray": [1, 2]}, "$ndarray"),
        (_ndarray("<f8", [-1], []), "$ndarray"),
        (_ndarray("<f8", [True], [0.0]), "$ndarray"),
        (_ndarray("<f8", "1", [0.0]), "$ndarray"),
        (_ndarray("<f8", [2], [0.0]), "$ndarray"),
        (_ndarray("<f8", [1], {"a": 0.0}), "$ndarray"),
        (_ndarray("<f8", [0, 2**62], []), "$ndarray"),
        ({"$dict": [[[1, 2], 3]]}, "$dict"),
        ({"$dict": [[1, 2, 3]]}, "$dict"),
        ({"$dict": [5]}, "$dict"),
        ({"$dict": {"a": 1}}, "$dict"),
        ({"$float": "1e5"}, "$float"),
        ({"$float": 5.0}, "$float"),
        ({"$complex": ["a", 1.0]}, "$complex"),
        ({"$complex": [1.0]}, "$complex"),
        ({"$complex": [True, 1.0]}, "$complex"),
        ({"$tuple": "abc"}, "$tuple"),
    ],
    ids=[
        "ndarray_missing_data",
        "ndarray_extra_key",
        "ndarray_body_list",
        "negative_dimension",
        "bool_dimension",
        "shape_string",
        "shape_data_mismatch",
        "data_mapping",
        "too_big_empty",
        "unhashable_key",
        "dict_triple",
        "dict_scalar_pair",
        "dict_mapping_body",
        "float_spelling",
        "float_number",
        "complex_string",
        "complex_one_part",
        "complex_bool",
        "tuple_string",
    ],
)
def test_malformed_tags_raise_value_errors(tmp_path, payload, tag) -> None:
    path = _crafted(tmp_path, payload)
    with pytest.raises(MALFORMED, match=_names(tag)):
        load_result(path)


@pytest.mark.finding("A1-01")
def test_the_error_names_the_metadata_key(tmp_path) -> None:
    path = _crafted(tmp_path, {"inner": [1, _ndarray("|O", [1], [1])]})
    with pytest.raises(MALFORMED, match=r"metadata\['x'\]\['inner'\]\[1\]"):
        load_result(path)


# ---------------------------------------------------------------------
# Unreadable files are format errors
# ---------------------------------------------------------------------


@pytest.mark.finding("A1-01")
def test_binary_and_deeply_nested_files_are_format_errors(tmp_path) -> None:
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00{garbage")
    with pytest.raises(ResultFormatError, match="UTF-8"):
        load_result(binary)

    deep = tmp_path / "deep.json"
    depth = sys.getrecursionlimit() + 100
    deep.write_text("[" * depth + "]" * depth, encoding="utf-8")
    with pytest.raises(ResultFormatError):
        load_result(deep)

    nested = _crafted(tmp_path, json.loads("[" * 150 + "]" * 150), name="nested.json")
    with pytest.raises(MALFORMED, match="nested deeper"):
        load_result(nested)


# ---------------------------------------------------------------------
# Controls: everything the writer writes still round-trips
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "array",
    [
        np.array([True, False]),
        np.arange(6, dtype=np.int8).reshape(2, 3),
        np.array([-(2**63), 2**63 - 1], dtype=np.int64),
        np.array([0, 2**64 - 1], dtype=np.uint64),
        np.array([1.5, -np.inf, np.nan], dtype=np.float16),
        np.array([1.5, np.inf], dtype=np.float32),
        np.array([1 + 2j, complex("nan")], dtype=np.complex64),
        np.array([1.0, 2.0], dtype=">f8"),
        np.array(["a", "bb", "ünï"]),
        np.array(["a", ""], dtype="U12"),
        np.array(7.0),
        np.zeros((0, 3)),
        np.zeros((2, 0, 4), dtype=np.complex128),
        np.ones((1,) * 40),
    ],
    ids=lambda a: f"{a.dtype.str}{list(a.shape)}",
)
def test_every_array_the_writer_writes_round_trips(tmp_path, array) -> None:
    res = FieldResult(backend="image", frequencies=[50.0], metadata={"a": array})
    back = load_result(save_result(res, tmp_path / "r.json")).metadata["a"]
    assert back.dtype == array.dtype and back.dtype.str == array.dtype.str
    assert back.shape == array.shape
    np.testing.assert_array_equal(back, array)


@pytest.mark.finding("A1-01")
def test_the_writer_refuses_a_string_array_the_reader_would_refuse() -> None:
    padded = np.array([""] * 2000, dtype="U4096")
    res = FieldResult(backend="image", frequencies=[50.0], metadata={"names": padded})
    with pytest.raises(Exception, match=r"names.*width"):
        res.model_dump_json()
    # the same strings at their own width are fine
    res.metadata["names"] = np.array(padded.tolist())
    assert FieldResult.model_validate_json(res.model_dump_json()).metadata[
        "names"
    ].shape == (2000,)


@pytest.mark.finding("A1-01")
def test_writer_and_reader_share_the_nesting_limit(tmp_path) -> None:
    from groundfield.solver.result import MAX_METADATA_DEPTH

    def nest(levels: int) -> list:
        value: list = []
        for _ in range(levels - 1):
            value = [value]
        return value

    ok = FieldResult(backend="image", frequencies=[50.0])
    ok.metadata["deep"] = nest(MAX_METADATA_DEPTH)
    back = load_result(save_result(ok, tmp_path / "ok.json")).metadata["deep"]
    assert back == ok.metadata["deep"]

    too_deep = FieldResult(backend="image", frequencies=[50.0])
    too_deep.metadata["deep"] = nest(MAX_METADATA_DEPTH + 1)
    with pytest.raises(Exception, match="nested deeper"):
        too_deep.model_dump_json()


def test_python_construction_still_passes_arrays_through() -> None:
    array = np.array([[1.0, 2.0]])
    res = FieldResult(backend="image", frequencies=[50.0], metadata={"a": array})
    assert res.metadata["a"] is array


def test_tag_lookalike_keys_still_round_trip(tmp_path) -> None:
    md = {"lookalike": {"$ndarray": "not an array"}, "f": {"$float": "nan"}}
    res = FieldResult(backend="image", frequencies=[50.0])
    res.metadata.update(md)
    back = load_result(save_result(res, tmp_path / "r.json")).metadata
    assert back["lookalike"] == {"$ndarray": "not an array"}
    assert back["f"] == {"$float": "nan"}


def test_a_crafted_file_is_rejected_before_its_array_is_allocated(tmp_path) -> None:
    """The check runs before numpy allocates (no MemoryError, fast)."""
    huge = _ndarray("<U2000000000", [1], ["x"])  # 8 GB if allocated
    path = _crafted(tmp_path, huge)
    with pytest.raises(MALFORMED, match=_names("$ndarray")):
        load_result(path)
    assert os.path.getsize(path) < 1000


@pytest.mark.finding("A1-01")
@pytest.mark.parametrize(
    "value",
    [
        {"$complex": [10**400, 0]},
        {"$complex": [0, -(10**400)]},
        {"$tuple": [{"$complex": [1, 10**400]}]},
        {"$dict": [[{"$complex": [10**400, 0]}, 1]]},
        {
            "$ndarray": {
                "dtype": "<c16",
                "shape": [1],
                "data": [{"$complex": [10**400, 1]}],
            }
        },
    ],
    ids=["complex-re", "complex-im", "tuple", "dict-key", "ndarray-item"],
)
def test_a_complex_tag_beyond_the_float_range_is_a_value_error(
    tmp_path: Path, value: dict
) -> None:
    """A 400-digit integer escaped as a bare OverflowError (not a ValueError)."""
    # audit A2S-04
    with pytest.raises(ValueError, match="outside the float range"):
        load_result(_crafted(tmp_path, value))
