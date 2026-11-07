"""Cartesian-product parameter sweeps for parameter axis studies.

This module turns the typical study axes — soil resistivity
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
The default response extractor reports one impedance per
frequency, selected by ``quantity`` (review pass 10, 10.18):

* ``"input"`` (default) — the **input (driving-point) impedance**
  :math:`Z_\\text{in} = (\\varphi_a - \\varphi_r) / I_s` of the
  current source: what a measurement instrument or a fault sees at
  the feeding point (:meth:`FieldResult.input_impedance`);
* ``"cluster"`` — the **cluster impedance**
  :math:`Z_c = \\varphi_c / \\sum_{e \\in c} I_e` of the source's
  galvanic cluster over its own soil leakage (the default up to
  v0.15.0);
* ``"electrode"`` — :math:`\\varphi_e / I_e` of the feeding
  electrode alone.

As soon as conductors (PEN, shields, leads) carry part of the
injected current away, the cluster impedance describes only the local
electrode: for a station with three houses on a PEN it is 1.79 times
the input impedance (1.26 times with one house, 3.09 times with eight;
review pass 10). The response is the engineering
:math:`Z(\\rho_1, h_1, f)` curve that
:mod:`groundfield.postprocess.rho_f_standard` and
:mod:`groundfield.postprocess.vector_fitting` consume.

Validity envelope
-----------------
* Frequency: quasi-static envelope :math:`f \\le 1\\,\\mathrm{kHz}`.
* Linearity: the sweep does not interpolate between samples — for
  a smooth :math:`Z(\\rho, f)` surface, hand the resulting
  DataFrame to :func:`fit_rho_f_standard` /
  :func:`fit_to_sympy_standard`.
* Cost: every Cartesian combination triggers one full
  :meth:`Engine.solve`. For 50 ρ-values × 10 h-values × 1 frequency
  list with N segments each, expect 500 dense system solves;
  budget accordingly with :func:`expected_segments`.
* Parallelism: the combinations are independent, so
  ``sweep(..., n_jobs=k)`` solves them in ``k`` worker processes
  (review pass 10, 10.64). The dense LU factorisation is only
  0.2–1.5 % of a solve (kernel assembly dominates), so a multi-threaded
  BLAS does not parallelise a solve; separate processes do (measured
  1.98× on 2 workers for a 20-house world). Workers run their BLAS on
  one thread; the result is bit-identical to the serial run when the
  calling process also runs a single-threaded BLAS (results differ at
  the 1e-15 level between 1 and 2 BLAS threads).
"""

from __future__ import annotations

import cmath
import contextlib
import functools
import itertools
import math
import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from groundfield.postprocess._impedance import (
    Quantity,
    check_quantity,
    impedance_series,
    select_current_source,
)

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = ["sweep", "plot_sweep_lines", "plot_sweep_heatmap"]


# ---------------------------------------------------------------------
# Default response extractor
# ---------------------------------------------------------------------


def _impedance_response(
    result: "FieldResult",
    world: "World",
    frequency_index: int,
    *,
    quantity: str = "input",
    source: str | None = None,
) -> dict[str, float]:
    """Impedance, its denominator current and the voltage of one source.

    Picks the current source (``source`` or the only one), reads the
    impedance selected by ``quantity`` (see the module docstring) and
    returns ``Z = U / I`` plus convenience fields:

    * ``U_E_*`` — the voltage of the quotient: the driving-point voltage
      :math:`\\varphi_a - \\varphi_r` (= EPR for a return through
      remote earth) for ``"input"``, the cluster potential for
      ``"cluster"``, the electrode potential for ``"electrode"``;
    * ``I_*`` — the impressed source current, the cluster leakage or
      the electrode leakage;
    * ``Z_*``, ``abs_Z``, ``arg_Z_deg`` — the impedance.

    Returns an empty dict when the world has no current source — that
    signals "no scalar response to extract" to :func:`sweep`, which then
    emits a row with only the axis values and ``frequency_Hz``.
    """
    pool = result.sources or world.sources
    src = select_current_source(
        result, source=source, electrode_name=None, caller="sweep", sources=pool
    )
    if src is None:
        return {}
    name = None if quantity == "input" else src.attached_to
    z_all, i_all = impedance_series(
        result,
        quantity,
        electrode_name=name,
        source=src.name if quantity == "input" else None,
        caller="sweep",
    )
    Z = complex(z_all[frequency_index])
    I = complex(i_all[frequency_index])
    if I != 0 and np.isfinite(Z.real) and np.isfinite(Z.imag):
        U = Z * I
        Z_re, Z_im = Z.real, Z.imag
        abs_Z = abs(Z)
        arg_Z = math.degrees(cmath.phase(Z))
    else:
        U = complex(result.electrode_potentials[src.attached_to][frequency_index])
        Z_re = Z_im = abs_Z = arg_Z = float("nan")

    return {
        "U_E_re": U.real,
        "U_E_im": U.imag,
        "abs_U_E": abs(U),
        "I_re": I.real,
        "I_im": I.imag,
        "abs_I": abs(I),
        "Z_re": Z_re,
        "Z_im": Z_im,
        "abs_Z": abs_Z,
        "arg_Z_deg": arg_Z,
    }


#: Default response of :func:`sweep` and :func:`convergence_study`: the
#: input impedance of the only current source.
_default_response = functools.partial(_impedance_response, quantity="input")


def _resolve_response(
    response: Callable[["FieldResult", "World", int], dict[str, float]] | None,
    quantity: str,
    source: str | None,
    caller: str,
) -> Callable[["FieldResult", "World", int], dict[str, float]]:
    """Default response for ``quantity`` / ``source`` or the user's one."""
    check_quantity(quantity)
    if response is None:
        return functools.partial(_impedance_response, quantity=quantity, source=source)
    if quantity != "input" or source is not None:
        raise ValueError(
            f"{caller}: quantity= and source= configure the default response; "
            "they cannot be combined with a custom response."
        )
    return response


# ---------------------------------------------------------------------
# Process-pool helpers (n_jobs > 1)
# ---------------------------------------------------------------------

#: Environment variables that cap the thread count of common BLAS /
#: OpenMP runtimes; set to "1" for the spawned worker processes.
_BLAS_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextlib.contextmanager
def _single_threaded_blas_env() -> Iterator[None]:
    """Set the BLAS thread variables to 1 while worker processes start."""
    saved = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
    try:
        for k in _BLAS_THREAD_VARS:
            os.environ[k] = "1"
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _worker_init() -> None:
    """Pin the worker's BLAS to one thread (threadpoolctl when available)."""
    try:  # pragma: no cover - optional dependency
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass


def _check_parallel_payload(**objects: Any) -> None:
    """Fail early if a callable cannot reach a spawned worker process."""
    import pickle

    import __main__

    for name, obj in objects.items():
        target = getattr(obj, "func", obj)  # functools.partial
        if getattr(target, "__module__", None) == "__main__" and not hasattr(
            __main__, "__file__"
        ):
            raise TypeError(
                f"sweep(n_jobs>1): {name} {target!r} is defined in an "
                "interactive session (notebook); worker processes cannot "
                "import it. Move it into a .py module next to the notebook "
                "and import it from there, or use n_jobs=1."
            )
        try:
            pickle.dumps(obj)
        except Exception as exc:  # noqa: BLE001 - any pickling failure
            raise TypeError(
                f"sweep(n_jobs>1): {name} must be picklable (a module-level "
                f"function, not a lambda or a nested function): {exc}. Use "
                "n_jobs=1 or move it into a module."
            ) from exc


def _sweep_cell(
    world_factory: Callable[..., "World"],
    engine: Any,
    response: Callable[["FieldResult", "World", int], dict[str, float]],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple[str, type, str, int]]]:
    """Solve one Cartesian combination; return its rows and warnings."""
    from groundfield.solver.engine import Engine

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        world = world_factory(**params)
        eng = engine if isinstance(engine, Engine) else engine(**params)
        result = eng.solve(world)
        rows = []
        for f_idx, f_hz in enumerate(result.frequencies):
            row = dict(params)
            row["frequency_Hz"] = float(f_hz)
            row.update(response(result, world, f_idx))
            rows.append(row)
    caught = [(str(w.message), w.category, w.filename, w.lineno) for w in record]
    return rows, caught


# ---------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------


def sweep(
    world_factory: Callable[..., "World"],
    engine: "Engine | Callable[..., Engine]",
    *,
    axes: dict[str, Sequence[Any]],
    response: Callable[["FieldResult", "World", int], dict[str, float]] | None = None,
    quantity: Quantity = "input",
    source: str | None = None,
    n_jobs: int = 1,
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
        ``dict[str, float|complex]``. Defaults to the impedance
        selected by ``quantity`` with the columns ``U_E_re``,
        ``U_E_im``, ``abs_U_E``, ``I_re``, ``I_im``, ``abs_I``,
        ``Z_re``, ``Z_im``, ``abs_Z``, ``arg_Z_deg`` (see the module
        docstring).
    quantity
        Impedance of the default response: ``"input"`` (default since
        review pass 10, 10.18) — the driving-point impedance of the
        source; ``"cluster"`` — the source cluster's potential over its
        own leakage (the v0.15.0 default); ``"electrode"`` — the feeding
        electrode's own U/I. Only with the default response.
    source
        Name of the current source the default response refers to; may
        be omitted when the world has exactly one current source (with
        several, v0.15.0 silently used the first one; now it raises).
    n_jobs
        Number of worker processes (default 1: serial, in this process).
        ``-1`` uses every logical CPU. With ``n_jobs > 1`` the
        combinations are solved in a process pool (``spawn`` start
        method, BLAS pinned to one thread per worker); rows keep the
        order of the serial run and warnings raised in the workers are
        re-issued here. ``world_factory``, ``engine`` and ``response``
        must then be picklable: module-level functions or an
        :class:`Engine`, not lambdas or functions defined in a notebook
        (put those into a small ``.py`` module).

    Returns
    -------
    pandas.DataFrame
        Long-format. Columns are the axis names, ``frequency_Hz``,
        and every key returned by ``response``. One row per
        Cartesian-product point per frequency. With the default
        response, ``df.attrs["impedance_quantity"]`` records
        ``quantity``.

    Raises
    ------
    ValueError
        If ``axes`` is empty or any axis is empty, ``quantity`` is
        unknown, ``quantity`` / ``source`` are combined with a custom
        ``response``, the default response finds several current
        sources and ``source`` is omitted, or ``n_jobs`` is 0 or below -1.
    TypeError
        With ``n_jobs > 1``, if a callable cannot be sent to the worker
        processes.

    Notes
    -----
    With ``n_jobs > 1`` the result is bit-identical to ``n_jobs=1``
    when the calling process runs a single-threaded BLAS as well
    (``OPENBLAS_NUM_THREADS=1``); otherwise the two agree to round-off.
    Process start-up costs about a second per worker, so parallel runs
    pay off for solves of a second or more.
    """
    from groundfield.solver.engine import Engine

    if not axes:
        raise ValueError("axes must contain at least one axis.")
    for name, values in axes.items():
        if len(list(values)) == 0:
            raise ValueError(f"axis '{name}' is empty.")

    default = response is None
    response = _resolve_response(response, quantity, source, "sweep")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        raise ValueError(f"n_jobs must be an integer, got {n_jobs!r}.")
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1 or -1, got {n_jobs}.")

    keys = list(axes.keys())
    values_list = [list(axes[k]) for k in keys]
    combos = [dict(zip(keys, c, strict=True)) for c in itertools.product(*values_list)]

    rows: list[dict[str, Any]] = []
    if n_jobs == 1 or len(combos) == 1:
        for params in combos:
            world = world_factory(**params)
            eng = engine if isinstance(engine, Engine) else engine(**params)
            result = eng.solve(world)
            for f_idx, f_hz in enumerate(result.frequencies):
                row = dict(params)
                row["frequency_Hz"] = float(f_hz)
                row.update(response(result, world, f_idx))
                rows.append(row)
    else:
        rows = _parallel_rows(world_factory, engine, response, combos, n_jobs)
    df = pd.DataFrame(rows)
    if default:
        df.attrs["impedance_quantity"] = quantity
    return df


def _parallel_rows(
    world_factory: Callable[..., "World"],
    engine: Any,
    response: Callable[["FieldResult", "World", int], dict[str, float]],
    combos: list[dict[str, Any]],
    n_jobs: int,
) -> list[dict[str, Any]]:
    """Solve ``combos`` in a process pool; rows in the order of ``combos``."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    _check_parallel_payload(
        world_factory=world_factory, engine=engine, response=response
    )
    cell = functools.partial(_sweep_cell, world_factory, engine, response)
    registry: dict[Any, Any] = {}
    rows: list[dict[str, Any]] = []
    with _single_threaded_blas_env():
        with ProcessPoolExecutor(
            max_workers=min(n_jobs, len(combos)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
        ) as pool:
            for cell_rows, caught in pool.map(cell, combos):
                for message, category, filename, lineno in caught:
                    warnings.warn_explicit(
                        message, category, filename, lineno, registry=registry
                    )
                rows.extend(cell_rows)
    return rows


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
            raise KeyError(f"column '{col}' not in DataFrame; have: {list(df.columns)}")

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
            float(pivot.columns.min()),
            float(pivot.columns.max()),
            float(pivot.index.min()),
            float(pivot.index.max()),
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
