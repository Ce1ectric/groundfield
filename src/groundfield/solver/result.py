"""Result object of a simulation run.

A :class:`FieldResult` is the unified return value of
:meth:`Engine.solve` and :meth:`World.solve`. It contains:

- ``electrode_potentials`` / ``electrode_currents`` — node values per
  frequency, used to derive input and transfer impedances.
- ``point_sources`` — the discretised current distribution. The
  post-processing layer can use these to evaluate the potential at
  arbitrary field points (contour plots, profiles, touch and step
  voltages).
- ``soil_resistivity`` and ``soil`` — provide the parameters needed to
  evaluate the appropriate Green's function (homogeneous or 2-layer)
  inside :meth:`potential`.
- ``sources`` — the source snapshot of the solve, read by
  :meth:`FieldResult.input_impedance`: the impressed current of every
  current source; the impressed voltage and the solved current of every
  voltage source.
- ``metadata`` — the core keys every backend reports (see
  :data:`groundfield.solver.engine.CORE_METADATA_KEYS`), including the
  ``provenance`` block, plus backend-specific keys.

Serialisation
-------------
``model_dump_json`` / ``model_validate_json`` round-trip a result
losslessly and produce standard JSON (no ``NaN`` / ``Infinity``
tokens). Values that JSON cannot express in ``metadata`` are written as
single-key tagged objects and restored on validation:

* ``{"$float": "inf" | "-inf" | "nan"}`` — non-finite float;
* ``{"$complex": [re, im]}`` — complex number;
* ``{"$tuple": [...]}`` — tuple;
* ``{"$dict": [[key, value], ...]}`` — mapping with non-string keys
  (e.g. ``penetration_depth`` keyed by frequency) or with a key that
  looks like a tag;
* ``{"$ndarray": {"dtype": ..., "shape": [...], "data": [...]}}`` —
  numpy array (bool, integer, float, complex or string dtype).

numpy scalars are stored as the equal Python scalar. The tag names are
reserved: a metadata value that is a single-key mapping with one of the
keys above is decoded on validation. The versioned file format around
this JSON is :func:`groundfield.io.save_result` /
:func:`groundfield.io.load_result`.

The decoder reads only what the encoder writes, so a result file from
an untrusted source cannot allocate more memory than a small multiple
of its own size: ``$ndarray`` dtypes are limited to bool, integer,
``float16`` … ``float64``, ``complex64`` / ``complex128`` and unicode
strings (no object, datetime, void, structured or sub-array dtypes);
the shape must match the number of items before anything is allocated;
every item must be a scalar of the dtype's kind; a string array may
take at most 64 bytes per character of its items (plus 4 KiB); and
metadata may nest at most :data:`MAX_METADATA_DEPTH` levels. A tag that
breaks these rules raises :class:`ValueError` naming the metadata key
(a ``pydantic.ValidationError`` when it is read through validation).
The encoder applies the same limits, so every file it writes loads.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
)

from groundfield.soil.models import (
    HomogeneousSoil,
    MultiLayerSoil,
    TwoLayerSoil,
)
from groundfield.sources import _internal_impedance, _with_ideal_internal_impedance

if TYPE_CHECKING:  # pragma: no cover - type-hint imports only
    import pandas as pd

    from groundfield.solver.engine import Engine
    from groundfield.sources import Source
    from groundfield.world import World

__all__ = [
    "MAX_METADATA_DEPTH",
    "FieldResult",
    "PointSource",
    "SourceRecord",
    "solve_provenance",
    "world_fingerprint",
]


# Local discriminated union of all soil types — kept here to avoid a
# circular import.
_SoilUnion = Union[HomogeneousSoil, TwoLayerSoil, MultiLayerSoil]


# 3-D coordinate
_Point3D = tuple[float, float, float]

#: Source kinds with a driving-point impedance (:meth:`FieldResult.input_impedance`).
_SOURCE_KINDS = ("current", "voltage")


class PointSource(BaseModel):
    """A discretised current source (segment midpoint, length, direction).

    Filled by the backend; not instantiated directly by the user. With
    the point kernel the current sits at ``position``; with the line
    kernel (``FieldResult.metadata["kernel"] == "line"``) it is spread
    uniformly along the segment given by ``length`` and ``direction``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    position: _Point3D = Field(..., description="(x, y, z) midpoint in m.")
    current: list[complex] = Field(
        ..., description="Current per frequency in A (complex phasor)."
    )
    electrode_name: str = Field(..., description="Owning electrode.")
    length: float = Field(..., gt=0.0, description="Represented length in m.")
    direction: _Point3D | None = Field(
        default=None,
        description=(
            "Unit tangent of the represented segment, which then runs from "
            "position - length/2·direction to position + length/2·direction. "
            "Used by the exact line-source kernel (kernel='line') of "
            ":meth:`FieldResult.potential`. ``None`` for results written "
            "before 2026.11 or by backends that do not record it."
        ),
    )


class SourceRecord(BaseModel):
    """Snapshot of one source as it was solved.

    Filled by :meth:`Engine.solve` (and, for a world with voltage
    sources, by the backend); not instantiated directly by the user.
    :meth:`FieldResult.input_impedance` reads it, so the result stays
    self-contained after the world has changed.

    For a current source ``current`` is the impressed current and
    ``voltage`` is empty. For a voltage source ``voltage`` is the
    impressed voltage $U$ and ``current`` the source current $I$ the
    solve returned — the current the source drives into ``attached_to``
    (and takes back from ``return_to``), the same sign convention as a
    current source; ``internal_resistance_ohm`` and
    ``internal_inductance_h`` are its internal impedance
    $Z_q = R + j\\omega L$ (0 for an ideal source and for current
    sources), so the terminal voltage is $U - Z_q I$.

    A record pickled before 2026.11 lacks the two fields of the internal
    impedance and unpickles with 0 / 0, as its JSON loads (audit W2R2-01).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Source name in the world.")
    kind: str = Field(..., description="Source kind: ``'current'`` or ``'voltage'``.")
    attached_to: str = Field(..., description="Feeding electrode / node.")
    return_to: str | None = Field(
        default=None,
        description="Return electrode; ``None`` means remote earth.",
    )
    current: list[complex] = Field(
        default_factory=list,
        description=(
            "Source current per frequency in A (complex phasor): impressed "
            "(current source) or solved (voltage source; empty if the "
            "backend did not report it)."
        ),
    )
    voltage: list[complex] = Field(
        default_factory=list,
        description=(
            "Impressed source voltage per frequency in V (complex phasor), "
            "attached_to against return_to (or remote earth), before the "
            "drop across the internal impedance. Empty for current sources."
        ),
    )
    internal_resistance_ohm: float = Field(
        default=0.0,
        description=(
            "Internal series resistance R of a voltage source in Ω; 0 for an "
            "ideal source, a current source and records written before "
            "2026.11 (JSON and pickles)."
        ),
    )
    internal_inductance_h: float = Field(
        default=0.0,
        description=(
            "Internal series inductance L of a voltage source in H; 0 for an "
            "ideal source, a current source and records written before "
            "2026.11 (JSON and pickles)."
        ),
    )

    def __setstate__(self, state: Any) -> None:
        super().__setstate__(_with_ideal_internal_impedance(type(self), state))

    def internal_impedance(self, frequency: float) -> complex:
        """Internal impedance $Z_q = R + j\\omega L$ in Ω at ``frequency`` (Hz).

        ``0j`` for a current source and an ideal voltage source.
        """
        return _internal_impedance(
            self.internal_resistance_ohm,
            self.internal_inductance_h,
            2.0 * np.pi * float(frequency),
        )


def _source_records(
    sources: Sequence[Source],
    n_freq: int,
    voltage_source_currents: Sequence[dict[str, complex]] | None = None,
) -> list[SourceRecord]:
    """Snapshot of ``sources`` for :attr:`FieldResult.sources`.

    Parameters
    ----------
    sources
        Sources of the solved world.
    n_freq
        Number of solved frequencies.
    voltage_source_currents
        Solved current in A per voltage-source name, one mapping per
        frequency; ``None`` leaves the currents of voltage sources empty.

    Returns
    -------
    list[SourceRecord]
        One record per source, in world order.
    """
    records = []
    for src in sources:
        phasor = complex(src.magnitude * np.exp(1j * np.deg2rad(src.phase_deg)))
        current: list[complex] = []
        voltage: list[complex] = []
        impedance: dict[str, float] = {}
        if src.kind == "current":
            current = [phasor] * n_freq
        elif src.kind == "voltage":
            voltage = [phasor] * n_freq
            if voltage_source_currents:
                current = [complex(m[src.name]) for m in voltage_source_currents]
            impedance = {
                "internal_resistance_ohm": float(src.internal_resistance_ohm),
                "internal_inductance_h": float(src.internal_inductance_h),
            }
        records.append(
            SourceRecord(
                name=src.name,
                kind=src.kind,
                attached_to=src.attached_to,
                return_to=src.return_to,
                current=current,
                voltage=voltage,
                **impedance,
            )
        )
    return records


# ---------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------


def world_fingerprint(world: World) -> str:
    """SHA-256 fingerprint of the canonical JSON of a world.

    The canonical form is ``json.dumps(world.model_dump(mode="json"),
    sort_keys=True, separators=(",", ":"))``. Pydantic writes floats in
    their shortest round-trip form, so equal worlds give equal
    fingerprints, also after a JSON round trip of the world. The
    fingerprint depends on the world schema of the installed version:
    renamed or added model fields change it.

    Parameters
    ----------
    world
        The world to fingerprint.

    Returns
    -------
    str
        ``"sha256:"`` followed by 64 hexadecimal digits.
    """
    canonical = json.dumps(
        world.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _package_version(module_name: str) -> str:
    """Version of an imported module, else of its installed distribution."""
    module = sys.modules.get(module_name)
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import version as dist_version

        return dist_version(module_name)
    except Exception:  # noqa: BLE001 - provenance must never break a solve
        return "unknown"


@functools.lru_cache(maxsize=1)
def _environment() -> tuple[tuple[str, str], ...]:
    """Versions and platform of this process (constant per process)."""
    import pydantic

    from groundfield import __version__

    return (
        ("groundfield_version", str(__version__)),
        ("python", platform.python_version()),
        ("numpy", str(np.__version__)),
        ("scipy", _package_version("scipy")),
        ("pydantic", str(pydantic.VERSION)),
        ("platform", platform.platform()),
    )


def solve_provenance(world: World, engine: Engine) -> dict[str, Any]:
    """Provenance block of a solve (without the timestamp).

    :meth:`Engine.solve <groundfield.solver.engine.Engine.solve>` calls
    this before the backend runs and adds ``solved_at_utc`` when it
    returns; the block is stored as ``FieldResult.metadata["provenance"]``.

    Parameters
    ----------
    world
        The world as built by the caller.
    engine
        The engine configuration.

    Returns
    -------
    dict
        ``groundfield_version``, ``python``, ``numpy``, ``scipy``,
        ``pydantic``, ``platform``, ``engine`` (the JSON dump of the
        engine) and ``world_fingerprint`` (see :func:`world_fingerprint`).
    """
    block: dict[str, Any] = dict(_environment())
    block["engine"] = engine.model_dump(mode="json")
    block["world_fingerprint"] = world_fingerprint(world)
    return block


# ---------------------------------------------------------------------
# Lossless JSON codec of the metadata
# ---------------------------------------------------------------------

_TAG_FLOAT = "$float"
_TAG_COMPLEX = "$complex"
_TAG_TUPLE = "$tuple"
_TAG_DICT = "$dict"
_TAG_NDARRAY = "$ndarray"
_TAGS = frozenset({_TAG_FLOAT, _TAG_COMPLEX, _TAG_TUPLE, _TAG_DICT, _TAG_NDARRAY})
_FLOAT_SPELLINGS = ("inf", "-inf", "nan")

#: Deepest nesting of containers (lists, mappings, tuples, tagged
#: objects) the metadata codec writes and reads.
MAX_METADATA_DEPTH = 100

#: ``numpy.dtype.str`` of every array dtype the codec stores: an optional
#: byte-order character, then bool, integer of 1-8 bytes, float of 2, 4
#: or 8 bytes, complex of 8 or 16 bytes, or unicode of at least one
#: character. ``float128`` / ``complex256`` are excluded because their
#: items have no lossless Python scalar; object, datetime, timedelta,
#: void, bytes, structured and sub-array dtypes never match.
_NDARRAY_DTYPE = re.compile(r"[<>|=]?(?:b1|[iu][1248]|f[248]|c(?:8|16)|U[1-9]\d{0,8})")

#: A string array may take at most this many bytes per character of its
#: JSON items (two quotes and a separator counted per item) ...
_NDARRAY_MAX_EXPANSION = 64
#: ... or this many bytes in total, whichever is larger. Together with
#: the shape check this keeps the decoded size linear in the file size.
_NDARRAY_FREE_BYTES = 4096


def _encode_float(value: float) -> Any:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return {_TAG_FLOAT: "nan"}
    return {_TAG_FLOAT: "inf" if value > 0 else "-inf"}


def _check_depth(depth: int, path: str) -> None:
    if depth > MAX_METADATA_DEPTH:
        raise ValueError(
            f"{path}: metadata is nested deeper than {MAX_METADATA_DEPTH} levels."
        )


def _check_string_array_size(itemsize: int, items: list[str], path: str) -> None:
    """Bound the bytes of a unicode array by the text of its items."""
    nbytes = itemsize * len(items)
    text = sum(len(item) + 3 for item in items)
    if nbytes > max(_NDARRAY_FREE_BYTES, _NDARRAY_MAX_EXPANSION * text):
        raise ValueError(
            f"{path}: {_TAG_NDARRAY} string array of width {itemsize // 4} "
            f"with {len(items)} item(s) takes {nbytes} bytes, more than "
            f"{_NDARRAY_MAX_EXPANSION} bytes per character of its items; "
            "store the strings at the width of the longest item "
            "(numpy.array(list_of_strings))."
        )


def _encode_metadata(value: Any, path: str, depth: int = 0) -> Any:
    """Encode a metadata value into standard-JSON-compatible data."""
    _check_depth(depth, path)
    inner = depth + 1
    if isinstance(value, np.ndarray):
        if not _NDARRAY_DTYPE.fullmatch(value.dtype.str):
            raise TypeError(
                f"{path}: numpy arrays of dtype {value.dtype} cannot be "
                "stored losslessly in FieldResult.metadata (bool, integer, "
                "float16/32/64, complex64/128 and unicode arrays can)."
            )
        items = value.ravel().tolist()
        if value.dtype.kind == "U":
            _check_string_array_size(value.dtype.itemsize, items, path)
        data = [
            _encode_metadata(item, f"{path}[{i}]", inner)
            for i, item in enumerate(items)
        ]
        return {
            _TAG_NDARRAY: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "data": data,
            }
        }
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _encode_float(value)
    if isinstance(value, complex):
        return {_TAG_COMPLEX: [_encode_float(value.real), _encode_float(value.imag)]}
    if isinstance(value, tuple):
        return {
            _TAG_TUPLE: [
                _encode_metadata(v, f"{path}[{i}]", inner) for i, v in enumerate(value)
            ]
        }
    if isinstance(value, list):
        return [_encode_metadata(v, f"{path}[{i}]", inner) for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        plain = all(isinstance(k, str) for k in value) and not (
            len(value) == 1 and next(iter(value)) in _TAGS
        )
        if plain:
            return {
                k: _encode_metadata(v, f"{path}[{k!r}]", inner)
                for k, v in value.items()
            }
        return {
            _TAG_DICT: [
                [
                    _encode_metadata(k, f"{path} key", inner),
                    _encode_metadata(v, f"{path}[{k!r}]", inner),
                ]
                for k, v in value.items()
            ]
        }
    raise TypeError(
        f"{path}: cannot store a {type(value).__name__} losslessly in "
        "FieldResult.metadata; use None, bool, int, float, complex, str, "
        "lists, tuples, mappings, numpy scalars or numpy arrays."
    )


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ndarray_item_fits(kind: str, item: Any) -> bool:
    """True if ``item`` is a scalar the encoder writes for ``kind``."""
    if kind == "b":
        return isinstance(item, bool)
    if kind in "iu":
        return isinstance(item, int) and not isinstance(item, bool)
    if kind == "f":
        return _is_real(item)
    if kind == "c":
        return _is_real(item) or isinstance(item, complex)
    return isinstance(item, str)


def _decode_ndarray(body: Any, path: str, depth: int) -> np.ndarray:
    """Decode an ``$ndarray`` body; validate everything before allocating."""
    bad = f"{path}: {_TAG_NDARRAY} tag"
    if not isinstance(body, dict) or set(body) != {"dtype", "shape", "data"}:
        raise ValueError(
            f"{bad} needs a mapping with exactly the keys 'dtype', 'shape' and 'data'."
        )
    spec, shape, data = body["dtype"], body["shape"], body["data"]
    if not isinstance(spec, str) or not _NDARRAY_DTYPE.fullmatch(spec):
        raise ValueError(
            f"{bad} has dtype {spec!r}; only bool, integer, float16/32/64, "
            "complex64/128 and unicode dtypes are stored (no object, "
            "datetime, void, bytes, structured or sub-array dtypes)."
        )
    try:
        dtype = np.dtype(spec)
    except (TypeError, ValueError):
        raise ValueError(
            f"{bad} has dtype {spec!r}, which numpy cannot build."
        ) from None
    if not isinstance(shape, list) or not all(
        isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape
    ):
        raise ValueError(f"{bad} has shape {shape!r}; expected non-negative integers.")
    size = math.prod(shape)
    if not isinstance(data, list) or len(data) != size:
        n_items = len(data) if isinstance(data, list) else type(data).__name__
        raise ValueError(
            f"{bad} has shape {shape} ({size} items) but data of {n_items} item(s)."
        )
    items = [_decode_metadata(v, f"{path}[{i}]", depth + 1) for i, v in enumerate(data)]
    for i, item in enumerate(items):
        if not _ndarray_item_fits(dtype.kind, item):
            raise ValueError(
                f"{bad}: item {i} ({type(item).__name__}) does not match "
                f"dtype {spec!r}."
            )
    if dtype.kind == "U":
        _check_string_array_size(dtype.itemsize, items, path)
    try:
        return np.array(items, dtype=dtype).reshape(tuple(shape))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{bad}: data do not fit dtype {spec!r} ({exc}).") from None


def _decode_tag(tag: str, body: Any, path: str, depth: int) -> Any:
    """Decode the body of one tagged object written by the encoder."""
    inner = depth + 1
    bad = f"{path}: {tag} tag"
    if tag == _TAG_FLOAT:
        if isinstance(body, str) and body in _FLOAT_SPELLINGS:
            return float(body)
        raise ValueError(f"{bad} must hold 'inf', '-inf' or 'nan', got {body!r}.")
    if tag == _TAG_COMPLEX:
        if isinstance(body, list) and len(body) == 2:
            # scalar parts: no extra nesting level (the encoder counts none)
            parts = [_decode_metadata(v, path, depth) for v in body]
            if all(_is_real(p) for p in parts):
                try:
                    return complex(parts[0], parts[1])
                except OverflowError:
                    # A JSON integer beyond the float range (audit round 2,
                    # A2S-04: it escaped load_result as a bare OverflowError).
                    raise ValueError(
                        f"{bad} holds a number outside the float range (|x| > 1.8e308)."
                    ) from None
        raise ValueError(f"{bad} must hold two real numbers, got {body!r:.80}.")
    if tag == _TAG_TUPLE:
        if isinstance(body, list):
            return tuple(
                _decode_metadata(v, f"{path}[{i}]", inner) for i, v in enumerate(body)
            )
        raise ValueError(f"{bad} must hold a list, got {type(body).__name__}.")
    if tag == _TAG_DICT:
        if not isinstance(body, list) or not all(
            isinstance(pair, list) and len(pair) == 2 for pair in body
        ):
            raise ValueError(f"{bad} must hold a list of [key, value] pairs.")
        out: dict[Any, Any] = {}
        for raw_key, raw_value in body:
            key = _decode_metadata(raw_key, f"{path} key", inner)
            try:
                hash(key)
            except TypeError:
                raise ValueError(
                    f"{bad} has an unhashable key of type {type(key).__name__}."
                ) from None
            out[key] = _decode_metadata(raw_value, f"{path}[{key!r}]", inner)
        return out
    return _decode_ndarray(body, path, depth)


def _decode_metadata(value: Any, path: str = "metadata", depth: int = 0) -> Any:
    """Inverse of :func:`_encode_metadata`.

    Raises
    ------
    ValueError
        If a tagged object is malformed or describes data the encoder
        cannot have written (see the module docstring), or if the value
        is nested deeper than :data:`MAX_METADATA_DEPTH`. The message
        names the metadata key.
    """
    _check_depth(depth, path)
    if isinstance(value, list):
        return [
            _decode_metadata(v, f"{path}[{i}]", depth + 1) for i, v in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        tag, body = next(iter(value.items()))
        if tag in _TAGS:
            return _decode_tag(tag, body, path, depth)
    return {
        k: _decode_metadata(v, f"{path}[{k!r}]", depth + 1) for k, v in value.items()
    }


class FieldResult(BaseModel):
    """Result of a field computation."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        # Non-finite floats in typed fields become "Infinity" / "NaN"
        # strings, so the JSON stays standard and round-trips exactly.
        ser_json_inf_nan="strings",
    )

    backend: str
    frequencies: list[float]
    electrode_potentials: dict[str, list[complex]] = Field(default_factory=dict)
    electrode_currents: dict[str, list[complex]] = Field(default_factory=dict)
    point_sources: list[PointSource] = Field(default_factory=list)
    soil_resistivity: float | None = Field(
        default=None,
        description=(
            "Effective resistivity used to evaluate the image-charge sum. "
            "For layered soils this stores $\\rho_1$; ``soil`` then "
            "carries the full layered model."
        ),
    )
    soil: _SoilUnion | None = Field(
        default=None,
        description=(
            "Full soil model used in the solution. :meth:`potential` reads "
            "this to pick the correct Green's-function kernel "
            "(homogeneous vs. 2-layer)."
        ),
    )
    clusters: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Cluster mapping: ``electrode_name -> sorted list of all "
            "electrodes that share the same galvanic cluster (including "
            "the electrode itself). Stand-alone electrodes map to "
            "``[name]``."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Core keys of every solve (groundfield.solver.engine."
            "CORE_METADATA_KEYS, including the provenance block) plus "
            "backend-specific keys. Serialised losslessly (tagged JSON "
            "objects for non-finite floats, complex numbers, tuples, "
            "non-string keys and numpy arrays)."
        ),
    )
    sources: list[SourceRecord] = Field(
        default_factory=list,
        description=(
            "Sources of the solved world (snapshot), used by "
            ":meth:`input_impedance`. Empty for hand-built results."
        ),
    )
    # Private, not serialised: numpy arrays of ``point_sources`` for
    # :meth:`potential` (10.65), keyed by the identity and length of the
    # list (see ``_point_source_arrays``).
    _ps_cache: tuple | None = PrivateAttr(default=None)

    @field_validator("metadata", mode="before")
    @classmethod
    def _restore_metadata(cls, value: Any) -> Any:
        """Decode the tagged JSON objects written by the serialiser."""
        if not isinstance(value, dict):
            return value
        if len(value) == 1 and _TAG_DICT in value:
            return _decode_metadata(value)
        return {k: _decode_metadata(v, f"metadata[{k!r}]", 1) for k, v in value.items()}

    @field_serializer("metadata", when_used="json")
    def _store_metadata(self, value: dict[str, Any]) -> Any:
        """Encode the metadata losslessly into standard JSON data."""
        return _encode_metadata(value, "metadata")

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def grounding_impedance(self, electrode_name: str) -> list[complex]:
        """Input impedance $Z(f) = U/I$ of an electrode.

        Parameters
        ----------
        electrode_name : str
            Name of the electrode whose input impedance is requested.
            Must be present in both :attr:`electrode_potentials` and
            :attr:`electrode_currents`.

        Returns
        -------
        list[complex]
            Complex impedance per frequency in ohms, one entry per
            entry in :attr:`frequencies`. Frequencies for which the
            electrode current is exactly zero return ``nan``.

        Raises
        ------
        KeyError
            If ``electrode_name`` is unknown to the result.

        Notes
        -----
        For galvanically connected electrodes (cluster with more than
        one member) this quantity is the cluster potential divided by
        the electrode's share of the total current. The physically
        meaningful quantity is the **cluster impedance** (see
        :meth:`cluster_impedance`).
        """
        if electrode_name not in self.electrode_potentials:
            raise KeyError(f"No potential data for '{electrode_name}'.")
        if electrode_name not in self.electrode_currents:
            raise KeyError(f"No current data for '{electrode_name}'.")
        u = self.electrode_potentials[electrode_name]
        i = self.electrode_currents[electrode_name]
        return [
            (uk / ik) if ik != 0 else complex("nan")
            for uk, ik in zip(u, i, strict=False)
        ]

    def cluster_impedance(self, electrode_name: str) -> list[complex]:
        """Grounding impedance of the galvanic cluster containing an electrode.

        Definition: $Z_{\\text{cluster}}(f) =
        \\varphi_{\\text{cluster}}/\\sum_{e \\in c} I_e$. For a
        stand-alone electrode this is identical to
        :meth:`grounding_impedance`. For connected electrodes it
        corresponds to the parallel combination of the individual
        grounding admittances.

        Parameters
        ----------
        electrode_name : str
            Name of any electrode in the target cluster. The cluster
            members are looked up in :attr:`clusters`; if the
            electrode does not appear there it is treated as a
            stand-alone cluster ``[electrode_name]``.

        Returns
        -------
        list[complex]
            Complex cluster impedance per frequency in ohms, one entry
            per entry in :attr:`frequencies`. Frequencies for which
            the summed cluster current is exactly zero return ``nan``.

        Raises
        ------
        KeyError
            If the resolved cluster is empty, or if none of its members
            carries potential *and* current data.

        Notes
        -----
        The cluster potential is the **length-weighted (Galerkin)
        average** over all members,

        .. math::
            \\varphi_c \\;=\\;
            \\frac{\\sum_{e \\in c} L_e\\, \\varphi_e}
                  {\\sum_{e \\in c} L_e},
            \\qquad
            L_e = \\sum_{k \\in e} L_k ,

        with the per-electrode lengths taken from
        :attr:`point_sources`. Since every member potential
        :math:`\\varphi_e` is itself the length-weighted average over
        that electrode's segments, :math:`\\varphi_c` is exactly the
        length-weighted average over the whole cluster surface — the
        quantity the augmented system constrains to one value (see
        ``solver/image.py``, block 1 of the multi-port system).

        Up to 0.14.1 this method read ``members[0]``, i.e. the
        *alphabetically first* member name. Because the reported
        member potentials were unweighted segment means they did not
        agree with each other, so renaming an electrode changed the
        reported cluster impedance of an unchanged geometry (19.4393 Ω
        → 19.2314 Ω in the audit case, 2026-07-28, F11). With the
        weighted reduction the members agree to round-off for the
        image family and the result is invariant under renaming for
        every backend.

        Numerator and denominator run over **different** member sets,
        deliberately:

        * :math:`\\varphi_c` averages over members that carry *both*
          potential and current data — a member with no potential
          cannot contribute a term to the average.
        * :math:`\\sum_e I_e` sums over every member that carries
          current data, whether or not its potential was stored,
          because its injected current is part of the cluster total
          regardless.

        Every backend in this package populates both mappings for
        every member, so the two sets coincide in practice. They
        differ only for hand-built :class:`FieldResult` instances,
        where restricting the current sum to the potential-carrying
        members would silently under-count the cluster current
        (audit 2026-07-29).
        """
        members, u = self._cluster_potential(electrode_name)
        n_freq = len(u)
        # The *denominator* runs over every member that has current
        # data, not over ``usable``: a member without potential data
        # cannot contribute to the weighted average, but its injected
        # current is still part of the cluster's total. Restricting the
        # sum to ``usable`` (0.15.0 development state) silently dropped
        # it and returned a too-large impedance instead of the
        # ``KeyError`` v0.14.1 raised — e.g. potentials {"a"} with
        # currents {"a": 1 A, "b": 1 A} and cluster ["a", "b"] gave
        # 10 V / 1 A = 10 Ω where 10 V / 2 A = 5 Ω is meant
        # (audit 2026-07-29). Not reachable through any shipped
        # backend, but ``FieldResult`` is public and user-constructible.
        current_members = [
            m
            for m in members
            if m in self.electrode_currents
            and len(self.electrode_currents[m]) >= n_freq
        ]
        i_sum = [
            sum(self.electrode_currents[m][k] for m in current_members)
            for k in range(n_freq)
        ]
        return [
            (uk / ik) if ik != 0 else complex("nan")
            for uk, ik in zip(u, i_sum, strict=False)
        ]

    def _cluster_potential(
        self, electrode_name: str
    ) -> tuple[list[str], list[complex]]:
        """Members and length-weighted potential of a galvanic cluster.

        Shared by :meth:`cluster_impedance` and :meth:`input_impedance`;
        see :meth:`cluster_impedance` for the weighting.
        """
        members = self.clusters.get(electrode_name, [electrode_name])
        if not members:
            raise KeyError(f"No cluster for '{electrode_name}'.")
        usable = [
            m
            for m in members
            if m in self.electrode_potentials and m in self.electrode_currents
        ]
        if not usable:
            raise KeyError(
                f"No potential/current data for any member of the cluster "
                f"of '{electrode_name}' (members: {members})."
            )

        # Per-electrode total leakage length (weights of the Galerkin
        # average). Falls back to equal weights when the backend did
        # not populate ``point_sources`` (stub results).
        lengths: dict[str, float] = {}
        for ps in self.point_sources:
            lengths[ps.electrode_name] = lengths.get(ps.electrode_name, 0.0) + float(
                ps.length
            )
        w = np.array([lengths.get(m, 0.0) for m in usable], dtype=float)
        if w.sum() <= 0.0:
            w = np.ones(len(usable), dtype=float)

        n_freq = len(self.electrode_potentials[usable[0]])
        u = [
            complex(
                sum(
                    float(wm) * self.electrode_potentials[m][k]
                    for wm, m in zip(w, usable, strict=False)
                )
                / float(w.sum())
            )
            for k in range(n_freq)
        ]
        return members, u

    def input_impedance(self, source: str | None = None) -> list[complex]:
        """Driving-point impedance seen by a source.

        Definition:

        $$
        Z_\\text{in}(f) \\;=\\;
        \\frac{\\varphi_a(f) - \\varphi_r(f)}{I_s(f)},
        $$

        with $\\varphi_a$ the potential of the galvanic cluster the
        source feeds (``attached_to``), $\\varphi_r$ the potential of
        the return cluster (``return_to``; zero for a return through
        remote earth) and $I_s$ the source current. Cluster potentials
        are the length-weighted averages used by
        :meth:`cluster_impedance`.

        * **Current source:** $I_s$ is the impressed current.
        * **Voltage source** (since 2026.11): the solve fixes
          $\\varphi_a - \\varphi_r = U - Z_q I_s$, so
          $Z_\\text{in} = U / I_s - Z_q$ with the impressed voltage $U$,
          the solved source current $I_s$ and the internal impedance
          $Z_q = R + j\\omega L$ of the source
          (:attr:`SourceRecord.voltage`, :attr:`SourceRecord.current`,
          :meth:`SourceRecord.internal_impedance`). It is the impedance
          the source sees at its terminals, **without** its own $Z_q$
          (the current is $U / (Z_q + Z_\\text{in})$). For a world with
          this one source it equals the impedance a 1 A current source
          at the same terminals reports, to round-off; with several
          sources it is an apparent impedance (see *Notes*).

        This is the impedance a measurement instrument or a fault sees
        at the feeding point. It differs from :meth:`cluster_impedance`
        (cluster potential over the cluster's own *leakage* current)
        as soon as part of the injected current leaves the cluster
        through conductors (PEN, cable shields, measurement leads):
        then the cluster impedance describes the local electrode only,
        the input impedance the whole connected system.

        Parameters
        ----------
        source
            Name of the source. May be omitted when the result carries
            exactly one source.

        Returns
        -------
        list[complex]
            Complex input impedance per frequency in ohms. Frequencies
            with zero source current return ``nan``.

        Raises
        ------
        ValueError
            If the result carries no source snapshot (hand-built
            results), no source, or several sources and ``source`` is
            omitted; if the record of a voltage source carries no solved
            current (a backend that does not report it).
        KeyError
            If ``source`` is unknown, or the feeding / return electrode
            has no potential data in this result.

        Notes
        -----
        With several simultaneous sources the potential difference
        includes the contribution of the other sources, so the ratio is
        an *apparent* impedance of the named source. Two 230 V sources
        on two 3 m rods 5 m apart (100 Ω·m, ``image``) report 38.90 Ω
        each, against 35.99 Ω of the 1 A solve of one rod (+8.1 %).
        """
        if not self.sources:
            raise ValueError(
                "input_impedance needs the source snapshot of the solve; "
                "this result carries none (hand-built FieldResult?)."
            )
        candidates = [s for s in self.sources if s.kind in _SOURCE_KINDS]
        if source is None:
            if not candidates:
                raise ValueError("The result carries no current source.")
            if len(candidates) > 1:
                kinds = {s.kind for s in candidates}
                label = f"{kinds.pop()} sources" if len(kinds) == 1 else "sources"
                names = ", ".join(s.name for s in candidates)
                raise ValueError(
                    f"The result carries several {label} ({names}); pass source=<name>."
                )
            record = candidates[0]
        else:
            by_name = {s.name: s for s in self.sources}
            if source not in by_name:
                known = ", ".join(sorted(by_name)) or "none"
                raise KeyError(f"Unknown source {source!r} (known: {known}).")
            record = by_name[source]
            if record.kind not in _SOURCE_KINDS:
                raise ValueError(
                    f"Source {source!r} is a {record.kind!r} source; the input "
                    "impedance is defined for current and voltage sources."
                )
        if record.kind == "voltage":
            n_freq = len(self.frequencies)
            if len(record.current) != n_freq or len(record.voltage) != n_freq:
                raise ValueError(
                    f"Voltage source {record.name!r}: the result carries no "
                    "solved source current (the backend did not report it), "
                    f"or not one per frequency ({len(record.current)} currents, "
                    f"{len(record.voltage)} voltages, {n_freq} frequencies)."
                )
            z_in = [
                u / i if i != 0 else complex("nan")
                for u, i in zip(record.voltage, record.current, strict=True)
            ]
            if record.internal_resistance_ohm or record.internal_inductance_h:
                # The impedance seen by the source, without its own Z_q.
                z_in = [
                    z - record.internal_impedance(f)
                    for z, f in zip(z_in, self.frequencies, strict=True)
                ]
            return z_in
        u_a = self._terminal_potential(record.name, "feeds", record.attached_to)
        if record.return_to is None:
            u_r = [0j] * len(u_a)
        else:
            u_r = self._terminal_potential(record.name, "returns to", record.return_to)
        return [
            (ua - ur) / ik if ik != 0 else complex("nan")
            for ua, ur, ik in zip(u_a, u_r, record.current, strict=False)
        ]

    def _terminal_potential(self, source: str, verb: str, node: str) -> list[complex]:
        """Potential of a source terminal: a junction or an electrode cluster."""
        junctions = self.metadata.get("junction_potentials") or {}
        if node in junctions:
            return [complex(v) for v in junctions[node]]
        if node not in self.electrode_potentials:
            raise KeyError(
                f"Source {source!r} {verb} {node!r}, which has no potential in "
                "this result."
            )
        return self._cluster_potential(node)[1]

    # ------------------------------------------------------------------
    # Potential evaluation at arbitrary field points
    # ------------------------------------------------------------------

    def potential(
        self,
        points: np.ndarray,
        frequency_index: int = 0,
        min_distance: float = 1e-3,
    ) -> np.ndarray:
        """Evaluate the potential at field points (image-charge sum).

        Picks the appropriate Green's-function kernel automatically:

        - **homogeneous** (:class:`HomogeneousSoil` or no ``self.soil``
          set but ``soil_resistivity`` available): classic image-charge
          sum $1/r + 1/r_{\\text{img}}$.
        - **2-layer** (:class:`TwoLayerSoil`, or :class:`MultiLayerSoil`
          with exactly two layers): Tagg/Sunde series truncated with the
          controls of the solve, ``metadata["series"]`` (the engine's
          ``image_max_terms`` and ``image_series_tol``; defaults 300 and
          ``1e-6`` for results without the record).
        - **1-layer** (:class:`MultiLayerSoil` with a single layer):
          degenerate case, dispatched to the homogeneous kernel.

        The dispatch reads the canonical layer stack of ``self.soil``
        (adjacent layers of one resistivity merged, review pass 10,
        10.48): ``TwoLayerSoil(rho, rho, h)`` and a ``MultiLayerSoil`` of
        one resistivity are evaluated like homogeneous soil, exactly as
        they are solved (before audit A3S-04 their sources below ``h``
        took the point form of the two-layer kernel, +2.7 % next to a
        rod), and a ``MultiLayerSoil`` with at most two distinct layers
        like the two-layer soil.

        Parameters
        ----------
        points
            Field points, array of shape ``(M, 3)`` in metres.
        frequency_index
            Index into :attr:`frequencies`. Default 0.
        min_distance
            Numerical cutoff for 1/r singularities, in metres.

        Returns
        -------
        phi : np.ndarray, shape (M,)
            Complex potential in V.

        Warns
        -----
        SeriesTruncationWarning
            If the two-layer series reaches ``max_terms`` before its tail
            bound falls below ``tol``.

        Raises
        ------
        NotImplementedError
            If ``self.soil`` is a :class:`MultiLayerSoil` with three or
            more distinct layers. The post-solve potential path does not yet
            carry an n-layer Green's-function kernel. The solve itself
            is still correct via the ``cim`` / ``mom_sommerfeld`` /
            ``bem`` backends — only the explicit
            :meth:`potential` evaluation is unavailable for n ≥ 3.
            Cluster impedances, electrode currents and potentials
            stored in :attr:`electrode_potentials` /
            :attr:`cluster_impedance` remain accessible.

        Notes
        -----
        Prior to 0.2.0 a :class:`MultiLayerSoil` with n ≥ 3 silently
        fell through to the homogeneous kernel, returning incorrect
        potentials, profiles, touch / step voltages and VTK exports.
        The error is now raised explicitly so the regime is visible
        rather than silent.

        Point sources at projected (UTM) coordinates, beyond 10 km from
        the origin, are evaluated in a local frame: points and sources are
        shifted by the rounded centre of the sources' bounding box (exact
        for nearby doubles) before the kernel runs (since 2026.11; see
        *Units and conventions* in the reference).
        """
        if not self.point_sources:
            raise RuntimeError(
                "FieldResult has no point_sources — the backend did not "
                "populate the post-processing data (stub?)."
            )

        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] != 3:
            raise ValueError(f"points must have shape (M, 3), got {pts.shape}.")

        sources, all_currents = self._point_source_arrays()
        # Contiguous, as before the cache: the real/imaginary views then
        # have the stride the BLAS calls of v0.15.0 saw.
        currents = np.ascontiguousarray(all_currents[:, frequency_index])
        line = self._line_sources()
        # Sources at projected (UTM) coordinates: evaluate in a local frame
        # of the sources, exactly shifted (2026.11, see solver._frame); a
        # result within 10 km of the origin is evaluated as before.
        from groundfield.solver._frame import points_origin

        origin = points_origin(sources)
        if origin is not None:
            pts = pts - origin
            sources = sources - origin
            if line is not None:
                from groundfield.solver._line_kernel import LineSegments

                line = LineSegments(
                    line.mid - origin, line.length, line.tangent, line.radius
                )

        # Dispatch to the kernel that matches the effective number of
        # soil layers: that of the canonical layer stack, the soil the
        # backends actually solve (review pass 10, 10.48). A
        # ``TwoLayerSoil(rho, rho, h)`` or a ``MultiLayerSoil`` of one
        # resistivity is homogeneous soil — its sources below h took the
        # point form of the two-layer spectral kernel here instead of the
        # line sources of the homogeneous result (+2.7 % next to a rod,
        # audit A3S-04); a ``MultiLayerSoil`` with one or two distinct
        # layers is cast on the fly. For n ≥ 3 there is no closed-form
        # image-charge series (Γ_1(λ) is no longer constant in λ — see
        # ``solver/image_nlayer.py``), so the call is rejected explicitly
        # rather than silently falling back to the homogeneous kernel.
        soil = self.soil
        if soil is not None and not isinstance(soil, HomogeneousSoil):
            from groundfield.solver._layered import canonical_soil

            soil = canonical_soil(soil)
        if isinstance(soil, TwoLayerSoil):
            return self._potential_two_layer(
                pts, sources, currents, soil, min_distance, line=line
            )
        if isinstance(soil, MultiLayerSoil):
            n = len(soil.layers)
            raise NotImplementedError(
                f"FieldResult.potential: MultiLayerSoil with n={n} ≥ 3 "
                "layers is not supported on the post-solve potential "
                "path. The solve itself is correct (use the 'cim', "
                "'mom_sommerfeld', or 'bem' backend), but a closed-form "
                "n-layer Green's-function kernel is not yet wired into "
                "FieldResult.potential. Access cluster impedances and "
                "electrode potentials directly via "
                "result.electrode_potentials / result.cluster_impedance "
                "instead, or reduce the soil model to "
                "TwoLayerSoil / HomogeneousSoil for explicit field "
                "evaluation."
            )
        if isinstance(soil, HomogeneousSoil) and soil is not self.soil:
            # A layered declaration of one resistivity (see above).
            return self._potential_homogeneous(
                pts,
                sources,
                currents,
                min_distance,
                rho_override=float(soil.rho),
                line=line,
            )
        return self._potential_homogeneous(
            pts, sources, currents, min_distance, line=line
        )

    # -- private helpers per Green's-function kind --------------------

    def _point_source_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions ``(N, 3)`` in m and currents ``(N, n_freq)`` in A.

        Built once and cached privately (10.65): converting the pydantic
        :class:`PointSource` list dominates a single-point
        :meth:`potential` call (8.4 ms rebuilt vs 0.13 ms cached at
        N = 3687). The cache is keyed by the
        identity and the length of :attr:`point_sources`, so assigning a
        new list or appending invalidates it; the list is otherwise
        treated as read-only after the solve (in-place edits of single
        :class:`PointSource` objects are not detected).
        """
        ps_list = self.point_sources
        cache = self._ps_cache
        if cache is not None and cache[0] is ps_list and cache[1] == len(ps_list):
            return cache[2], cache[3]
        sources = np.array([ps.position for ps in ps_list])
        # Backends store one current per frequency for every source; a
        # hand-built ragged list is cut to the shortest entry.
        n_freq = min((len(ps.current) for ps in ps_list), default=0)
        currents = np.array(
            [ps.current[:n_freq] for ps in ps_list], dtype=complex
        ).reshape(len(ps_list), n_freq)
        self._ps_cache = (ps_list, len(ps_list), sources, currents)
        return sources, currents

    def _line_sources(self):
        """Point sources as line segments when the solve used ``kernel="line"``.

        Returns ``None`` for point-kernel results and for results whose
        point sources carry no ``direction`` (then the point kernel is
        used, as before).
        """
        if self.metadata.get("kernel") != "line":
            return None
        if any(ps.direction is None for ps in self.point_sources):
            return None
        from groundfield.solver._line_kernel import LineSegments

        sources, _ = self._point_source_arrays()
        return LineSegments(
            sources,
            np.array([ps.length for ps in self.point_sources], dtype=float),
            np.array([ps.direction for ps in self.point_sources], dtype=float),
            np.zeros(len(self.point_sources)),
        )

    def _potential_homogeneous(
        self,
        pts: np.ndarray,
        sources: np.ndarray,
        currents: np.ndarray,
        min_distance: float,
        rho_override: float | None = None,
        line=None,
    ) -> np.ndarray:
        rho = float(rho_override) if rho_override is not None else self.soil_resistivity
        if rho is None:
            raise RuntimeError(
                "FieldResult.soil_resistivity is not set — homogeneous "
                "image-charge evaluation requires it."
            )
        from groundfield.solver._kernels import _point_field

        return _point_field(pts, sources, currents, rho, min_distance, line=line)

    def _potential_two_layer(
        self,
        pts: np.ndarray,
        sources: np.ndarray,
        currents: np.ndarray,
        soil: TwoLayerSoil,
        min_distance: float,
        max_terms: int | None = None,
        tol: float | None = None,
        line=None,
    ) -> np.ndarray:
        """Tagg/Sunde series for 2-layer soil (upper-layer fast path).

        Layer dispatch (audit 2026-07-08, WP-D1): the image series is
        valid for source AND observer in the upper layer only. Any
        pair involving layer 2 is routed through the rigorous
        spectral kernel of :mod:`groundfield.coupling.layered_green`
        (measured uu-series error before the fix: +7 % at z = 7 m,
        +25 % at z = 12 m for K = +0.818, h_1 = 5 m). Pure
        upper-layer evaluations are bit-exact to the historic path.

        The series controls default to those recorded with the solve
        (``metadata["series"]``, see
        :func:`~groundfield.solver._series.series_controls`); up to
        2026.11 they were hard-coded to 100 terms and ``1e-6`` whatever
        ``Engine.image_max_terms`` said (review pass 10, 10.25: +0.83 % at
        100 m, +3.6 % at 500 m for :math:`K = -0.94`). A truncated series
        emits :class:`~groundfield.solver._series.SeriesTruncationWarning`.

        With the line kernel (``line`` given) the upper-layer series
        uses line sources; pairs involving layer 2 keep the point form
        of the spectral kernel.
        """
        if max_terms is None or tol is None:
            from groundfield.solver._series import series_controls

            rec_terms, rec_tol = series_controls(self.metadata)
            max_terms = rec_terms if max_terms is None else max_terms
            tol = rec_tol if tol is None else tol
        h_1 = soil.h_1
        rho_1 = soil.rho_1

        pts_l1 = pts[:, 2] < h_1
        src_l1 = sources[:, 2] < h_1
        if not (pts_l1.all() and src_l1.all()):
            from groundfield.coupling.layered_green import (
                two_layer_probe_matrix,
            )

            phi = np.zeros(pts.shape[0], dtype=complex)
            if pts_l1.any() and src_l1.any():
                phi[pts_l1] += self._potential_two_layer(
                    pts[pts_l1],
                    sources[src_l1],
                    currents[src_l1],
                    soil,
                    min_distance,
                    max_terms,
                    tol,
                    line=None if line is None else line.subset(src_l1),
                )
            if (~src_l1).any():
                G = two_layer_probe_matrix(
                    pts,
                    sources[~src_l1],
                    rho_1=rho_1,
                    rho_2=soil.rho_2,
                    h_1=h_1,
                    min_distance=min_distance,
                )
                phi += G @ currents[~src_l1]
            if (~pts_l1).any() and src_l1.any():
                G = two_layer_probe_matrix(
                    pts[~pts_l1],
                    sources[src_l1],
                    rho_1=rho_1,
                    rho_2=soil.rho_2,
                    h_1=h_1,
                    min_distance=min_distance,
                )
                phi[~pts_l1] += G @ currents[src_l1]
            return phi

        from groundfield.solver._kernels import _series_field
        from groundfield.solver._series import ImageSeries

        series = ImageSeries.for_soil(soil, max_terms, tol)
        series.warn_if_truncated("FieldResult.potential")
        # An all-zero imaginary part (galvanic solves) sums to exactly
        # zero and is skipped.
        parts = [currents.real]
        if np.any(currents.imag):
            parts.append(currents.imag)
        from groundfield.solver._series import series_far_field

        sums = _series_field(
            pts,
            sources,
            parts,
            series,
            min_distance,
            line=line,
            far_field=series_far_field(self.metadata),
        )
        phi_re = sums[0]
        phi_im = sums[1] if len(sums) > 1 else np.zeros_like(phi_re)
        return (rho_1 / (4.0 * np.pi)) * (phi_re + 1j * phi_im)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def to_dataframe(self, table: str = "electrodes") -> pd.DataFrame:
        """Tidy table of the result, one row per frequency and object.

        Parameters
        ----------
        table
            ``"electrodes"`` (default): one row per frequency and
            electrode with the columns

            * ``frequency_hz`` — frequency in Hz;
            * ``electrode`` — electrode name;
            * ``cluster`` — galvanic cluster, labelled by its
              alphabetically first member (see :attr:`clusters`);
            * ``potential_v`` — electrode potential in V (complex);
            * ``current_a`` — leakage current of the electrode into the
              soil in A (complex);
            * ``cluster_impedance_ohm`` — :meth:`cluster_impedance` of
              the electrode's cluster in Ω (complex), the cluster
              potential over the cluster's own leakage current.

            ``"sources"``: one row per frequency and recorded source
            with ``frequency_hz``, ``source``, ``kind``,
            ``attached_to``, ``return_to``, ``current_a`` (source
            current in A, complex: impressed, or solved for a voltage
            source), ``voltage_v`` (impressed voltage in V of a voltage
            source, complex; ``nan`` for a current source),
            ``internal_resistance_ohm`` and ``internal_inductance_h``
            (internal impedance of a voltage source in Ω and H, float;
            ``nan`` for a current source) and ``input_impedance_ohm``
            (:meth:`input_impedance` in Ω, complex), the impedance seen
            at the feeding point (without the internal impedance).

        Returns
        -------
        pandas.DataFrame
            The table; rows ordered by frequency, then by electrode /
            source in result order. Values that the result does not
            carry (hand-built results) are ``nan``.

        Raises
        ------
        ValueError
            If ``table`` is not ``"electrodes"`` or ``"sources"``.
        ImportError
            If pandas is not installed (it is imported on first use).

        Examples
        --------
        >>> df = result.to_dataframe()                       # doctest: +SKIP
        >>> df.pivot(index="frequency_hz", columns="electrode",
        ...          values="potential_v").abs()             # doctest: +SKIP
        """
        if table not in ("electrodes", "sources"):
            raise ValueError(
                f"Unknown table {table!r}; choose 'electrodes' or 'sources'."
            )
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas is a dependency
            raise ImportError(
                "FieldResult.to_dataframe needs pandas (pip install pandas)."
            ) from exc
        if table == "sources":
            return self._source_frame(pd)
        return self._electrode_frame(pd)

    def _electrode_frame(self, pd: Any) -> Any:
        nan = complex("nan")
        names = list(self.electrode_potentials)
        names += [n for n in self.electrode_currents if n not in names]
        labels: dict[str, str] = {}
        z_cluster: dict[str, list[complex]] = {}
        for name in names:
            members = self.clusters.get(name) or [name]
            label = members[0]
            labels[name] = label
            if label not in z_cluster:
                try:
                    z_cluster[label] = list(self.cluster_impedance(name))
                except (KeyError, ValueError, ZeroDivisionError):
                    z_cluster[label] = []

        def _at(values: list[complex] | None, k: int) -> complex:
            return complex(values[k]) if values is not None and k < len(values) else nan

        rows: dict[str, list[Any]] = {
            "frequency_hz": [],
            "electrode": [],
            "cluster": [],
            "potential_v": [],
            "current_a": [],
            "cluster_impedance_ohm": [],
        }
        for k, f in enumerate(self.frequencies):
            for name in names:
                rows["frequency_hz"].append(float(f))
                rows["electrode"].append(name)
                rows["cluster"].append(labels[name])
                rows["potential_v"].append(_at(self.electrode_potentials.get(name), k))
                rows["current_a"].append(_at(self.electrode_currents.get(name), k))
                rows["cluster_impedance_ohm"].append(_at(z_cluster[labels[name]], k))
        return pd.DataFrame(
            {
                "frequency_hz": pd.Series(rows["frequency_hz"], dtype=float),
                "electrode": pd.Series(rows["electrode"], dtype=object),
                "cluster": pd.Series(rows["cluster"], dtype=object),
                "potential_v": pd.Series(rows["potential_v"], dtype=complex),
                "current_a": pd.Series(rows["current_a"], dtype=complex),
                "cluster_impedance_ohm": pd.Series(
                    rows["cluster_impedance_ohm"], dtype=complex
                ),
            }
        )

    def _source_frame(self, pd: Any) -> Any:
        nan = complex("nan")
        z_in: dict[str, list[complex]] = {}
        for record in self.sources:
            if record.kind not in _SOURCE_KINDS:
                continue
            try:
                z_in[record.name] = list(self.input_impedance(record.name))
            except (KeyError, ValueError, ZeroDivisionError):
                pass
        rows: dict[str, list[Any]] = {
            "frequency_hz": [],
            "source": [],
            "kind": [],
            "attached_to": [],
            "return_to": [],
            "current_a": [],
            "voltage_v": [],
            "internal_resistance_ohm": [],
            "internal_inductance_h": [],
            "input_impedance_ohm": [],
        }
        for k, f in enumerate(self.frequencies):
            for record in self.sources:
                z = z_in.get(record.name, [])
                rows["frequency_hz"].append(float(f))
                rows["source"].append(record.name)
                rows["kind"].append(record.kind)
                rows["attached_to"].append(record.attached_to)
                rows["return_to"].append(record.return_to)
                rows["current_a"].append(
                    complex(record.current[k]) if k < len(record.current) else nan
                )
                rows["voltage_v"].append(
                    complex(record.voltage[k]) if k < len(record.voltage) else nan
                )
                voltage = record.kind == "voltage"
                rows["internal_resistance_ohm"].append(
                    float(record.internal_resistance_ohm) if voltage else math.nan
                )
                rows["internal_inductance_h"].append(
                    float(record.internal_inductance_h) if voltage else math.nan
                )
                rows["input_impedance_ohm"].append(complex(z[k]) if k < len(z) else nan)
        return pd.DataFrame(
            {
                "frequency_hz": pd.Series(rows["frequency_hz"], dtype=float),
                "source": pd.Series(rows["source"], dtype=object),
                "kind": pd.Series(rows["kind"], dtype=object),
                "attached_to": pd.Series(rows["attached_to"], dtype=object),
                "return_to": pd.Series(rows["return_to"], dtype=object),
                "current_a": pd.Series(rows["current_a"], dtype=complex),
                "voltage_v": pd.Series(rows["voltage_v"], dtype=complex),
                "internal_resistance_ohm": pd.Series(
                    rows["internal_resistance_ohm"], dtype=float
                ),
                "internal_inductance_h": pd.Series(
                    rows["internal_inductance_h"], dtype=float
                ),
                "input_impedance_ohm": pd.Series(
                    rows["input_impedance_ohm"], dtype=complex
                ),
            }
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Compact one-line description of the result.

        Returns
        -------
        str
            Human-readable summary listing the backend name, the
            number of frequencies, the number of electrodes for which
            potentials were stored and the number of discretised
            point sources. Intended for logging and notebook output;
            the format is informational and not stable across
            versions.
        """
        return (
            f"FieldResult(backend='{self.backend}', "
            f"n_freq={len(self.frequencies)}, "
            f"n_electrodes={len(self.electrode_potentials)}, "
            f"n_segments={len(self.point_sources)})"
        )
