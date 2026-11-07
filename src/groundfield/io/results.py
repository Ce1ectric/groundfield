"""Versioned on-disk format of a :class:`FieldResult`.

A result file is one UTF-8 JSON document (standard JSON, no ``NaN`` /
``Infinity`` tokens, no pickle) with an envelope::

    {
      "format": "groundfield.result",
      "schema_version": 1,
      "written_by": "groundfield <version>",
      "provenance": {...},        # FieldResult.metadata["provenance"] or null
      "result": {...}             # FieldResult JSON (lossless, see below)
    }

``provenance`` repeats the solve provenance of the result
(groundfield / numpy / scipy / Python versions, platform, engine
configuration, ``world_fingerprint`` and the UTC timestamp), so it can
be read without parsing the result. It is ``null`` for results that
were not produced by :meth:`Engine.solve
<groundfield.solver.engine.Engine.solve>` (hand-built results).

The ``result`` body is :meth:`FieldResult.model_dump_json`, which
round-trips losslessly: complex numbers are strings, non-finite floats
of typed fields are ``"Infinity"`` / ``"NaN"`` strings, and the
metadata uses tagged objects for values JSON cannot express (see
:mod:`groundfield.solver.result`).

Schema versions
---------------
``schema_version`` increases whenever a reader of the previous version
could misread a file. :func:`load_result` reads every version up to
:data:`RESULT_SCHEMA_VERSION` and raises :class:`ResultFormatError` for
a newer one, instead of guessing.

The world itself is not part of the file; save it alongside with
``world.model_dump_json()`` (exact round trip) and compare it with
:func:`world_fingerprint` against ``provenance["world_fingerprint"]``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from groundfield.solver.result import FieldResult, world_fingerprint

__all__ = [
    "RESULT_FORMAT",
    "RESULT_SCHEMA_VERSION",
    "ResultFormatError",
    "load_result",
    "save_result",
    "world_fingerprint",
]

#: Value of the ``format`` key of a result file.
RESULT_FORMAT = "groundfield.result"

#: Schema version written by :func:`save_result`.
RESULT_SCHEMA_VERSION = 1


class ResultFormatError(ValueError):
    """A file is not a readable groundfield result file."""


def save_result(
    result: FieldResult,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    indent: int | None = None,
) -> Path:
    """Write a result to a versioned JSON file.

    Parameters
    ----------
    result
        The result to store.
    path
        Target file. The parent directory must exist.
    overwrite
        Replace an existing file. Default ``False``: an existing file
        raises :class:`FileExistsError`.
    indent
        Indentation of the JSON text; ``None`` (default) writes it
        compactly.

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    TypeError
        If ``result`` is not a :class:`FieldResult`.
    FileExistsError
        If ``path`` exists and ``overwrite`` is false.
    ValueError
        If the result holds values that cannot be stored (the message
        names the metadata key).

    Notes
    -----
    The file is written to a temporary file in the target directory and
    moved into place with :func:`os.replace`, so an interrupted write
    never leaves a truncated result file behind.
    """
    if not isinstance(result, FieldResult):
        raise TypeError(
            f"save_result expects a FieldResult, got {type(result).__name__}."
        )
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it.")
    from groundfield import __version__

    body: dict[str, Any] = json.loads(result.model_dump_json())
    envelope = {
        "format": RESULT_FORMAT,
        "schema_version": RESULT_SCHEMA_VERSION,
        "written_by": f"groundfield {__version__}",
        "provenance": body.get("metadata", {}).get("provenance"),
        "result": body,
    }
    text = json.dumps(envelope, allow_nan=False, indent=indent, ensure_ascii=False)
    directory = target.parent if str(target.parent) else Path(".")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return target


def load_result(path: str | os.PathLike[str]) -> FieldResult:
    """Read a result file written by :func:`save_result`.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    FieldResult
        The stored result, equal to the one that was saved.

    Raises
    ------
    ResultFormatError
        If the file is not JSON, not a groundfield result file, uses a
        schema version newer than :data:`RESULT_SCHEMA_VERSION`, or
        lacks the ``result`` body.
    pydantic.ValidationError
        If the ``result`` body does not validate as a
        :class:`FieldResult`.
    """
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultFormatError(f"{source} is not a JSON file: {exc}") from exc
    if not isinstance(payload, dict) or "format" not in payload:
        raise ResultFormatError(
            f"{source} is not a groundfield result file (no 'format' key, "
            f"expected '{RESULT_FORMAT}'). Result files are written by "
            "groundfield.io.save_result; a bare FieldResult JSON loads with "
            "FieldResult.model_validate_json."
        )
    if payload["format"] != RESULT_FORMAT:
        raise ResultFormatError(
            f"{source}: format {payload['format']!r} is not {RESULT_FORMAT!r}."
        )
    version = payload.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ResultFormatError(
            f"{source}: invalid schema_version {version!r} (expected an integer >= 1)."
        )
    if version > RESULT_SCHEMA_VERSION:
        writer = payload.get("written_by", "a newer groundfield")
        raise ResultFormatError(
            f"{source} uses schema_version {version}, newer than the "
            f"supported {RESULT_SCHEMA_VERSION} (written by {writer}); "
            "upgrade groundfield to read it."
        )
    if "result" not in payload:
        raise ResultFormatError(f"{source} has no 'result' body.")
    return FieldResult.model_validate(payload["result"])
