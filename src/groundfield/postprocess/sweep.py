"""Cartesian-product parameter sweeps for AP1 axis studies.

This module turns the AP1 work-package axes — soil resistivity
:math:`\\rho_1` / :math:`\\rho_2`, layer thickness :math:`h_1`,
electrode geometry, frequency — into a single tabular response
that is cheap to plot and easy to feed into vector-fitting,
:math:`\\rho`-:math:`f` regression, or downstream
``groundinsight``.

API surface
-----------
:func:`sweep`
    Walk the Cartesian product of an arbitrary number of named
    axes, build a fresh :class:`World` (and optionally a fresh
    :class:`Engine`) per combination, solve, and extract a scalar
    response per frequency. Returns a long-format
    :class:`pandas.DataFrame` with one row per
    *(axis values × frequency)*.
:func:`plot_sweep_lines`
    Line plot of one response column against a chosen axis,
    optionally with one curve per value of a second axis.
:func:`plot_sweep_heatmap`
    Pivot-table heatmap of one response column over a
    *(x_axis, y_axis)* pair (e.g. :math:`\\rho_1` vs. :math:`h_1`).

Mathematical / physical content
-------------------------------
The default response extractor picks up the
**cluster impedance** :math:`Z_c = U_c / I_c` at the source's
galvanic cluster, with :math:`U_c` the cluster potential and
:math:`I_c = \\sum_{e \\in c_\\text{src}} I_e` the net soil
leakage. Both numerator and denominator are read from
:class:`FieldResult` per ``frequency_index``, so the response is
the engineering :math:`Z(\\rho_1, h_1, f)` curve that
:mod:`groundfield.postprocess.rho_f_standard` and
:mod:`groundfield.postprocess.vector_fitting` consume.

Validity envelope
-----------------
* Frequency: dissertation envelope :math:`f \\le 1\\,\\mathrm{kHz}`.
* Linearity: the sweep does not interpolate between samples — for
  a smooth :math:`Z(\\rho, f)` surface, hand the resulting
  DataFrame to :func:`fit_rho_f_standard` /
  :func:`fit_to_sympy_standard`.
* Cost: every Cartesian combination triggers one full
  :meth:`Engine.solve`. For 50 ρ-values × 10 h-values × 1 frequency
  list with N segments each, expect 500 dense system solves;
  budget accordingly with :func:`expected_segments`.
"""

from __future__ import annotations

import cmath
import itertools
import math
from typing import TYPE_CHECKING, Any, Callable, Sequence

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = ["sweep", "plot_sweep_lines", "plot_sweep_heatmap"]


# ---------------------------------------------------------------------
# Default response extractor
# ---------------------------------------------------------------------


def _default_response(
    result: "FieldResult", world: "World", frequency_index: int
) -> dict[str, float]:
    """Cluster impedance and EPR at the source's cluster.

    Picks the (single) current source's cluster, sums the
    per-electrode soil leakages and returns ``Z_c = U_c / I_c``
    plus convenience fields.

    Returns an empty dict when the world has no current source —
    that signals "no scalar response to extract" to :func:`sweep`,
    which then emits a row with only the axis values and
    ``frequency_Hz``.
    """
    current_sources = [s for s in world.sources if s.kind == "current"]
    if not current_sources:
        return {}
    src = current_sources[0]
    members = result.clusters.get(src.attached_to, [src.attached_to])
    if not members:
        return {}

    U = complex(result.electrode_potentials.get(members[0], [0j])[frequency_index])
    I = 0 + 0j
    for m in members:
        if m in result.electrode_currents:
            I += complex(result.electrode_currents[m][frequency_index])

    if I != 0:
        Z = U / I
        Z_re, Z_im = Z.real, Z.imag
        abs_Z = abs(Z)
        arg_Z = math.degrees(cmath.phase(Z))
    else:
        Z_re = Z_im = abs_Z = arg_Z = float("nan")

    return {
        "U_E_re": U.real, "U_E_im": U.imag, "abs_U_E": abs(U),
        "I_re": I.real, "I_im": I.imag, "abs_I": abs(I),
        "Z_re": Z_re, "Z_im": Z_im, "abs_Z": abs_Z, "arg_Z_deg": arg_Z,
    }


# ---------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------


def sweep(
    world_factory: Callable[..., "World"],
    engine: "Engine | Callable[..., Engine]",
    *,
    axes: dict[str, Sequence[Any]],
    response: Callable[["FieldResult", "World", int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Cartesian-product parameter sweep across user-defined axes.

    For every combination ``(a_1 = v_1, a_2 = v_2, ...)`` in the
    Cartesian product of ``axes``, the function

    1. builds a fresh world via ``world_factory(**combination)``,
    2. resolves the engine — either the static :class:`Engine`
       passed in or a per-combination engine via
       ``engine(**combination)`` if a callable is given,
    3. solves with :meth:`World.solve`,
    4. iterates over every frequency in
       :attr:`FieldResult.frequencies` and extracts a row via
       ``response(result, world, frequency_index)``.

    The axis values and ``frequency_Hz`` are added to every row
    automatically.

    Parameters
    ----------
    world_factory
        Callable that builds a fresh :class:`World` from the
        per-combination kwargs. Must return a fully populated
        world (soil, electrodes, sources, optional conductors).
    engine
        Either a static :class:`Engine` (reused for every
        combination) or a callable
        ``engine(**combination) -> Engine`` (rebuilt per
        combination — useful when the segment length depends on
        the geometry).
    axes
        Mapping ``axis_name -> Sequence`` of values. Must contain
        at least one axis. Empty sequences raise immediately.
    response
        Optional response extractor. Receives
        ``(result, world, frequency_index)`` and must return a
        ``dict[str, float|complex]``. Defaults to
        :func:`_default_response` (cluster impedance + EPR at
        the source's cluster).

    Returns
    -------
    pandas.DataFrame
        Long-format. Columns are the axis names, ``frequency_Hz``,
        and every key returned by ``response``. One row per
        Cartesian-product point per frequency.

    Raises
    ------
    ValueError
        If ``axes`` is empty or any axis is empty.
    """
    from groundfield.solver.engine import Engine

    if not axes:
        raise ValueError("axes must contain at least one axis.")
    for name, values in axes.items():
        if len(list(values)) == 0:
            raise ValueError(f"axis '{name}' is empty.")

    if response is None:
        response = _default_response

    keys = list(axes.keys())
    values_list = [list(axes[k]) for k in keys]

    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*values_list):
        params = dict(zip(keys, combo))
        world = world_factory(**params)
        eng = engine if isinstance(engine, Engine) else engine(**params)
        result = eng.solve(world)
        for f_idx, f_hz in enumerate(result.frequencies):
            row = dict(params)
            row["frequency_Hz"] = float(f_hz)
            row.update(response(result, world, f_idx))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------


def plot_sweep_lines(
    df: pd.DataFrame,
    *,
    x: str,
    y: str = "abs_Z",
    color: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    log_x: bool = False,
    log_y: bool = False,
    title: str | None = None,
):
    """Line plot of ``y`` versus ``x``, one curve per ``color`` value.

    Parameters
    ----------
    df
        Long-format DataFrame as produced by :func:`sweep`.
    x
        Column name to use on the x-axis.
    y
        Column name of the response. Default ``"abs_Z"``.
    color
        Optional second column name; one line per distinct value
        is drawn. ``None`` (default) collapses to a single line.
    figsize, log_x, log_y, title
        Standard matplotlib options.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if x not in df.columns:
        raise KeyError(f"column '{x}' not in DataFrame; have: {list(df.columns)}")
    if y not in df.columns:
        raise KeyError(f"column '{y}' not in DataFrame; have: {list(df.columns)}")

    fig, ax = plt.subplots(figsize=figsize)
    if color is None:
        sub = df.sort_values(x)
        ax.plot(sub[x], sub[y], marker="o")
    else:
        if color not in df.columns:
            raise KeyError(
                f"color column '{color}' not in DataFrame; have: {list(df.columns)}"
            )
        for val, sub in df.groupby(color):
            sub = sub.sort_values(x)
            ax.plot(sub[x], sub[y], marker="o", label=f"{color} = {val}")
        ax.legend()

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, which="both", alpha=0.3)
    if title is None:
        title = f"{y} vs. {x}" + (f"  (color: {color})" if color else "")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_sweep_heatmap(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    response: str = "abs_Z",
    frequency_Hz: float | None = None,
    agg: str = "mean",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (7.5, 5.5),
    title: str | None = None,
):
    """Heatmap of ``response`` over the ``(x, y)`` axis pair.

    Pivot-tables ``df`` by ``y`` (rows) and ``x`` (columns),
    aggregating ``response`` with the chosen ``agg`` (default
    ``"mean"`` — useful when the sweep contains additional axes
    that should collapse). Selects a single frequency slice
    when ``frequency_Hz`` is given.

    Parameters
    ----------
    df
        Long-format DataFrame from :func:`sweep`.
    x, y
        Column names for the heatmap axes. Both must be numeric.
    response
        Column to colour-code. Default ``"abs_Z"``.
    frequency_Hz
        If given and the DataFrame has a ``frequency_Hz`` column,
        keep only rows that match this frequency. Raises if no
        rows match.
    agg
        Aggregation passed to :meth:`pd.DataFrame.pivot_table`.
        Default ``"mean"``.
    cmap, figsize, title
        Standard matplotlib options.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    for col in (x, y, response):
        if col not in df.columns:
            raise KeyError(
                f"column '{col}' not in DataFrame; have: {list(df.columns)}"
            )

    sub = df
    if frequency_Hz is not None and "frequency_Hz" in df.columns:
        sub = df[df["frequency_Hz"] == frequency_Hz]
        if sub.empty:
            raise ValueError(
                f"No rows match frequency_Hz={frequency_Hz}; "
                f"available: {sorted(df['frequency_Hz'].unique())}"
            )

    pivot = sub.pivot_table(index=y, columns=x, values=response, aggfunc=agg)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=(
            float(pivot.columns.min()), float(pivot.columns.max()),
            float(pivot.index.min()), float(pivot.index.max()),
        ),
        cmap=cmap,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label=response)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title is None:
        title = f"{response}({x}, {y})"
        if frequency_Hz is not None:
            title += f"  @ f = {frequency_Hz:g} Hz"
    ax.set_title(title)
    fig.tight_layout()
    return fig
