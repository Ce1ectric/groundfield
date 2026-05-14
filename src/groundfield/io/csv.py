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
downstream consumers (fourth 2026-05-12 audit pass).

All writers use UTF-8, comma-separated, with a header row;
floating-point values are written at full precision so the
files round-trip without loss of accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = [
    "save_potential_path_csv",
    "save_electrode_table_csv",
    "save_cluster_impedances_csv",
    "POTENTIAL_PATH_COLUMNS",
    "ELECTRODE_TABLE_REQUIRED_COLUMNS",
    "CLUSTER_IMPEDANCE_REQUIRED_COLUMNS",
]


#: Frozen column list for :func:`save_potential_path_csv`. The
#: writer always emits exactly these columns in exactly this order.
#: Locking this contract here makes regressions on the column name
#: convention test-detectable (fourth 2026-05-12 audit pass).
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
#: ``world`` is supplied).
ELECTRODE_TABLE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "electrode",
    "I_re",
    "I_im",
    "abs_I",
)

#: Mandatory columns of :func:`save_cluster_impedances_csv`.
CLUSTER_IMPEDANCE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "cluster",
    "Z_re",
    "Z_im",
    "abs_Z",
)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _ensure_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


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

    Returns
    -------
    pathlib.Path
        The path the file was written to.

    Raises
    ------
    ValueError
        If ``distance <= 0``, ``n < 2``, ``direction`` is the zero
        vector, or any ``frequency_indices`` value is out of range.
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
                    f"frequency index {k} out of range "
                    f"[0, {len(result.frequencies)})."
                )

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

    out = _ensure_path(path)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.17g")
    return out


def save_electrode_table_csv(
    result: "FieldResult",
    path: str | Path,
    *,
    world: "World | None" = None,
    frequency_index: int = 0,
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
        Index into :attr:`FieldResult.frequencies`. Default 0.

    Returns
    -------
    pathlib.Path
    """
    from groundfield.postprocess.current_balance import electrode_current_table

    df = electrode_current_table(
        result, world=world, frequency_index=frequency_index
    )
    out = _ensure_path(path)
    df.to_csv(out, index=False, float_format="%.17g")
    return out


def save_cluster_impedances_csv(
    result: "FieldResult",
    path: str | Path,
    *,
    frequency_index: int = 0,
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
        Index into :attr:`FieldResult.frequencies`. Default 0.

    Returns
    -------
    pathlib.Path
    """
    from groundfield.postprocess.current_balance import cluster_current_balance

    df = cluster_current_balance(result, frequency_index=frequency_index)
    if "members" in df.columns:
        df = df.copy()
        df["members"] = df["members"].apply(lambda lst: ";".join(lst))
    out = _ensure_path(path)
    df.to_csv(out, index=False, float_format="%.17g")
    return out
