"""CSV exports for groundfield results.

This module provides three convenience writers that turn a
:class:`FieldResult` (and optionally its companion :class:`World`)
into machine-readable, tool-agnostic CSV files. They wrap the
existing ``postprocess`` helpers — no new science here, just a
clean disk format for sharing typical results across notebooks,
spreadsheets, and downstream pipelines.

Functions
---------
:func:`save_potential_path_csv`
    Sample :meth:`FieldResult.potential` along a straight line on
    or below the soil surface and write
    ``(s, x, y, z, frequency_Hz, phi_re, phi_im, abs_phi)`` rows.
:func:`save_electrode_table_csv`
    Wrap :func:`groundfield.postprocess.electrode_current_table`
    and dump the per-electrode summary.
:func:`save_cluster_impedances_csv`
    Wrap :func:`groundfield.postprocess.cluster_current_balance`
    and dump the per-cluster summary.

Column-name convention
----------------------
All complex-valued quantities follow the same ``<symbol>_re``,
``<symbol>_im``, ``abs_<symbol>`` triple where ``symbol`` is the
physical symbol of the quantity:

* potential — ``phi_re`` / ``phi_im`` / ``abs_phi``;
* current — ``I_re`` / ``I_im`` / ``abs_I``;
* impedance — ``Z_re`` / ``Z_im`` / ``abs_Z``.

The three writers therefore deliberately use *different* magnitude
columns (``abs_phi`` for the potential path, ``abs_I`` for the
electrode table, ``abs_Z`` for the cluster table). They are **not**
intended to be ``concat``/``merge``-ed on the magnitude column
without an explicit rename; the column constants below
(:data:`POTENTIAL_PATH_COLUMNS`, etc.) lock the schema for
downstream consumers.

The identifier columns follow the ``postprocess`` tables the writers
wrap: an electrode row is keyed by ``name``, a cluster row by
``cluster_root`` (the canonical member of the galvanic cluster).
There is no ``electrode`` or ``cluster`` column.

All writers use UTF-8, comma-separated, with a header row;
floating-point values are written at full precision so the
files round-trip without loss of accuracy.

Frequency and provenance (review pass 10, 10.73)
------------------------------------------------
Every table carries a ``frequency_Hz`` column, and every file starts
with ``#`` comment lines that record where it came from::

    # groundfield csv: electrode_table
    # groundfield_version: 2026.11.0
    # written_utc: 2026-11-07T19:30:00Z
    # backend: image
    # frequency_Hz: 50
    # frequency_index: 0

followed by the provenance block of the solve (``solve.*`` keys) when
the result carries one. Read such a file with
:func:`read_csv_with_provenance`, which skips exactly the leading
comment lines and returns the metadata as a dict, or with
``pandas.read_csv(path, comment="#")`` when no value contains ``#``.
Pass ``provenance=False`` for a plain CSV. The writers refuse to
replace an existing file unless ``overwrite=True``.
"""

from __future__ import annotations

import datetime as _dt
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = [
    "save_potential_path_csv",
    "save_electrode_table_csv",
    "save_cluster_impedances_csv",
    "read_csv_with_provenance",
    "POTENTIAL_PATH_COLUMNS",
    "ELECTRODE_TABLE_REQUIRED_COLUMNS",
    "CLUSTER_IMPEDANCE_REQUIRED_COLUMNS",
]


#: Frozen column list for :func:`save_potential_path_csv`. The
#: writer always emits exactly these columns in exactly this order.
#: Locking this contract here makes regressions on the column name
#: convention test-detectable.
POTENTIAL_PATH_COLUMNS: tuple[str, ...] = (
    "s",
    "x",
    "y",
    "z",
    "frequency_Hz",
    "phi_re",
    "phi_im",
    "abs_phi",
)

#: Mandatory columns of :func:`save_electrode_table_csv` (additional
#: columns such as ``kind`` / ``depth_m`` may appear when a
#: ``world`` is supplied). The electrode is identified by ``name``
#: and its galvanic cluster by ``cluster_root`` — the column names
#: emitted by :func:`groundfield.postprocess.electrode_current_table`.
ELECTRODE_TABLE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "name",
    "cluster_root",
    "I_re",
    "I_im",
    "abs_I",
    "frequency_Hz",
)

#: Mandatory columns of :func:`save_cluster_impedances_csv`. A cluster
#: is identified by ``cluster_root``, the canonical (alphabetically
#: first) member of the galvanic cluster, as emitted by
#: :func:`groundfield.postprocess.cluster_current_balance`.
CLUSTER_IMPEDANCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "cluster_root",
    "Z_re",
    "Z_im",
    "abs_Z",
    "frequency_Hz",
)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _ensure_path(path: str | Path, overwrite: bool) -> Path:
    """Create parent directories; refuse to replace an existing file."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"{p} exists; pass overwrite=True to replace it (review pass 10: "
            "the CSV writers no longer overwrite silently)."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _one_line(value: Any) -> str:
    """Single-line, control-character-free text for a comment line."""
    text = str(value)
    return "".join(
        " " if unicodedata.category(ch)[0] in ("C", "Z") and ch != " " else ch
        for ch in text
    ).strip()


def _fmt_number(x: float) -> str:
    return f"{float(x):.17g}"


def _provenance_lines(
    result: "FieldResult", kind: str, extra: dict[str, Any]
) -> list[str]:
    """``# key: value`` header lines describing the written table."""
    from groundfield import __version__

    entries: dict[str, Any] = {
        "groundfield_version": __version__,
        "written_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "backend": result.backend,
    }
    entries.update(extra)
    solve = (
        result.metadata.get("provenance") if isinstance(result.metadata, dict) else None
    )
    if isinstance(solve, dict):
        for key, value in solve.items():
            if isinstance(value, (str, int, float, bool)):
                entries[f"solve.{key}"] = value
    lines = [f"# groundfield csv: {kind}"]
    lines += [f"# {_one_line(k)}: {_one_line(v)}" for k, v in entries.items()]
    return lines


def _write(df: pd.DataFrame, out: Path, header: list[str] | None) -> Path:
    body = df.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    text = ("\n".join(header) + "\n" if header else "") + body
    out.write_text(text, encoding="utf-8")
    return out


def read_csv_with_provenance(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a CSV written by this module, with its provenance header.

    Parameters
    ----------
    path
        File written by one of the ``save_*_csv`` writers (with or
        without provenance header).

    Returns
    -------
    df : pandas.DataFrame
        The table. Only the leading ``#`` lines are skipped, so values
        that contain ``#`` (electrode names) stay intact.
    provenance : dict[str, str]
        ``key -> value`` of the header lines (the first line's
        ``groundfield csv: <table>`` appears as key ``groundfield csv``);
        empty for a plain CSV.
    """
    p = Path(path)
    meta: dict[str, str] = {}
    n_comment = 0
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            n_comment += 1
            key, sep, value = line[1:].strip().partition(":")
            if sep:
                meta[key.strip()] = value.strip()
    return pd.read_csv(p, skiprows=n_comment), meta


# ---------------------------------------------------------------------
# Public writers
# ---------------------------------------------------------------------


def save_potential_path_csv(
    result: "FieldResult",
    path: str | Path,
    *,
    start: tuple[float, float, float],
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    distance: float = 30.0,
    n: int = 200,
    frequency_indices: Sequence[int] | None = None,
    overwrite: bool = False,
    provenance: bool = True,
) -> Path:
    """Sample the potential along a line and write to CSV.

    Builds an evenly spaced set of ``n`` field points starting at
    ``start`` and walking ``distance`` metres along the unit
    vector of ``direction``, evaluates
    :meth:`FieldResult.potential` at every selected frequency
    index, and writes the result to ``path``.

    Parameters
    ----------
    result
        Solver output. Must carry ``point_sources`` (i.e. not a
        stub backend).
    path
        Destination CSV path. Parent directories are created
        automatically.
    start
        Path start point ``(x, y, z)`` in metres.
    direction
        Direction vector. Will be normalised to unit length.
        Default ``(1, 0, 0)``.
    distance
        Path length in metres. Default 30.
    n
        Number of sample points along the path. Default 200.
    frequency_indices
        Iterable of integer indices into
        :attr:`FieldResult.frequencies`. ``None`` (default)
        uses every available frequency.
    overwrite
        Replace an existing file (default ``False``: raise).
    provenance
        Write the ``#`` provenance header (default ``True``, see the
        module docstring).

    Returns
    -------
    pathlib.Path
        The path the file was written to.

    Raises
    ------
    ValueError
        If ``distance <= 0``, ``n < 2``, ``direction`` is the zero
        vector, or any ``frequency_indices`` value is out of range.
    FileExistsError
        If ``path`` exists and ``overwrite`` is ``False``.
    """
    if distance <= 0.0:
        raise ValueError(f"distance must be > 0, got {distance!r}.")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n!r}.")
    direction_arr = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction_arr))
    if norm == 0.0:
        raise ValueError(f"direction must be non-zero, got {tuple(direction)}.")
    direction_arr = direction_arr / norm

    if frequency_indices is None:
        f_idx = list(range(len(result.frequencies)))
    else:
        f_idx = list(frequency_indices)
        for k in f_idx:
            if not (0 <= k < len(result.frequencies)):
                raise ValueError(
                    f"frequency index {k} out of range [0, {len(result.frequencies)})."
                )

    out = _ensure_path(path, overwrite)
    s = np.linspace(0.0, float(distance), int(n))
    pts = np.column_stack(
        [
            start[0] + s * direction_arr[0],
            start[1] + s * direction_arr[1],
            start[2] + s * direction_arr[2],
        ]
    )

    rows = []
    for k in f_idx:
        f_hz = float(result.frequencies[k])
        phi = result.potential(pts, frequency_index=k)
        for i in range(len(s)):
            rows.append(
                {
                    "s": float(s[i]),
                    "x": float(pts[i, 0]),
                    "y": float(pts[i, 1]),
                    "z": float(pts[i, 2]),
                    "frequency_Hz": f_hz,
                    "phi_re": float(phi[i].real),
                    "phi_im": float(phi[i].imag),
                    "abs_phi": float(abs(phi[i])),
                }
            )

    header = None
    if provenance:
        header = _provenance_lines(
            result,
            "potential_path",
            {
                "frequencies_Hz": ", ".join(
                    _fmt_number(result.frequencies[k]) for k in f_idx
                ),
                "frequency_indices": ", ".join(str(k) for k in f_idx),
            },
        )
    return _write(pd.DataFrame(rows, columns=list(POTENTIAL_PATH_COLUMNS)), out, header)


def save_electrode_table_csv(
    result: "FieldResult",
    path: str | Path,
    *,
    world: "World | None" = None,
    frequency_index: int = 0,
    overwrite: bool = False,
    provenance: bool = True,
) -> Path:
    """Wrap :func:`electrode_current_table` and write to CSV.

    Parameters
    ----------
    result
        Solver output.
    path
        Destination CSV path.
    world
        Optional companion world; when given, the table includes
        the ``kind`` and ``depth_m`` columns.
    frequency_index
        Index into :attr:`FieldResult.frequencies`. Default 0. The
        frequency is written to the ``frequency_Hz`` column.
    overwrite
        Replace an existing file (default ``False``: raise).
    provenance
        Write the ``#`` provenance header (default ``True``).

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is ``False``.
    """
    from groundfield.postprocess.current_balance import electrode_current_table

    out = _ensure_path(path, overwrite)
    df = electrode_current_table(result, world=world, frequency_index=frequency_index)
    f_hz = float(result.frequencies[frequency_index])
    df.insert(min(2, len(df.columns)), "frequency_Hz", f_hz)
    header = None
    if provenance:
        header = _provenance_lines(
            result,
            "electrode_table",
            {"frequency_Hz": _fmt_number(f_hz), "frequency_index": frequency_index},
        )
    return _write(df, out, header)


def save_cluster_impedances_csv(
    result: "FieldResult",
    path: str | Path,
    *,
    frequency_index: int = 0,
    overwrite: bool = False,
    provenance: bool = True,
) -> Path:
    """Wrap :func:`cluster_current_balance` and write to CSV.

    Note that the ``members`` column of
    :func:`cluster_current_balance` carries Python lists; CSV is
    a flat format, so the lists are joined into ``';'``-separated
    strings before writing.

    Parameters
    ----------
    result
        Solver output.
    path
        Destination CSV path.
    frequency_index
        Index into :attr:`FieldResult.frequencies`. Default 0. The
        frequency is written to the ``frequency_Hz`` column.
    overwrite
        Replace an existing file (default ``False``: raise).
    provenance
        Write the ``#`` provenance header (default ``True``).

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is ``False``.

    Notes
    -----
    The ``Z`` columns are cluster impedances (a cluster's potential over
    its own soil leakage), not the input impedance a source sees.
    """
    from groundfield.postprocess.current_balance import cluster_current_balance

    out = _ensure_path(path, overwrite)
    df = cluster_current_balance(result, frequency_index=frequency_index)
    if "members" in df.columns:
        df = df.copy()
        df["members"] = df["members"].apply(lambda lst: ";".join(lst))
    f_hz = float(result.frequencies[frequency_index])
    df.insert(min(1, len(df.columns)), "frequency_Hz", f_hz)
    header = None
    if provenance:
        header = _provenance_lines(
            result,
            "cluster_impedances",
            {"frequency_Hz": _fmt_number(f_hz), "frequency_index": frequency_index},
        )
    return _write(df, out, header)
