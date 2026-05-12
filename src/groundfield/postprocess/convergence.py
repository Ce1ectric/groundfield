"""Convergence study over the engine's segment_length.

The PDE / field model in ``groundfield`` is a *reference*
computation (see ``CLAUDE.md``); to honour that role every
non-trivial result should be backed up by a refinement study.
This module turns the canonical *"halve the segment length, watch
what happens"* experiment into one function call.

Mathematical / physical content
-------------------------------
The image-family discretiser splits each electrode into segments
of length :math:`\\Delta s`. As :math:`\\Delta s \\to 0` the
multi-port grounding matrix approaches the continuous integral
operator and the cluster impedance :math:`Z_c` converges to the
PDE-grade reference. The convergence is monotone in :math:`\\Delta
s` for the average-potential method (cf. Sunde 1968; Tagg 1964).
A practical refinement plot therefore answers two questions at
once:

* "Has my chosen :math:`\\Delta s` already converged within X %?"
  — the curve flattens out.
* "What is the asymptotic (PDE-grade) value?" — extrapolation /
  Richardson if needed.

Validity envelope
-----------------
* Frequency: dissertation envelope :math:`f \\le 1\\,\\mathrm{kHz}`.
* Backends: any image-family solver (image / image_2layer /
  image_nlayer / mom / mom_sommerfeld / cim / bem). FEM is
  supported but its mesh is generated independently — the
  ``segment_length`` knob does not directly control its accuracy.
* The function clones the engine via :meth:`Engine.model_copy`,
  so the original engine is not mutated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import pandas as pd

from groundfield.postprocess.sweep import _default_response

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.engine import Engine
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = ["convergence_study", "plot_convergence"]


def convergence_study(
    world: "World",
    engine: "Engine",
    *,
    segment_lengths: Sequence[float],
    response: Callable[["FieldResult", "World", int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Solve the same world repeatedly with refining ``segment_length``.

    For every :math:`\\Delta s_k` in ``segment_lengths`` the
    function builds a clone of the engine (only ``segment_length``
    differs), solves the world, and extracts a scalar response
    per frequency. The returned DataFrame is sorted **descending**
    in ``segment_length_m`` so finer resolutions land on the right
    in the default plot.

    Parameters
    ----------
    world
        World to solve. **Not modified.**
    engine
        Base engine. Cloned per refinement step via
        :meth:`Engine.model_copy`; the original is left untouched.
    segment_lengths
        Sequence of segment lengths in metres. Must be strictly
        positive and contain at least two distinct values.
    response
        Optional response extractor. Defaults to
        :func:`groundfield.postprocess.sweep._default_response`
        (cluster impedance and EPR at the source's cluster).

    Returns
    -------
    pandas.DataFrame
        Long-format. Columns ``segment_length_m``,
        ``frequency_Hz``, ``n_segments`` and every key of the
        response (e.g. ``Z_re``, ``Z_im``, ``abs_Z``,
        ``arg_Z_deg``, ``U_E_re``, ...).

    Raises
    ------
    ValueError
        If ``segment_lengths`` is empty, contains non-positive
        values, or has fewer than two distinct values (a
        convergence study below two refinement steps is
        meaningless).
    """
    ds_list = [float(ds) for ds in segment_lengths]
    if len(ds_list) < 2:
        raise ValueError(
            f"segment_lengths must contain at least 2 values, got {len(ds_list)}."
        )
    if any(not (ds > 0) for ds in ds_list):
        raise ValueError(
            f"segment_lengths must be strictly positive, got {ds_list}."
        )
    if len(set(ds_list)) < 2:
        raise ValueError(
            "segment_lengths must contain at least 2 distinct values."
        )

    if response is None:
        response = _default_response

    rows: list[dict[str, Any]] = []
    for ds in ds_list:
        eng = engine.model_copy(update={"segment_length": ds})
        result = world.solve(eng)
        n_seg = len(result.point_sources)
        for f_idx, f_hz in enumerate(result.frequencies):
            row: dict[str, Any] = {
                "segment_length_m": ds,
                "frequency_Hz": float(f_hz),
                "n_segments": n_seg,
            }
            row.update(response(result, world, f_idx))
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(
        ["frequency_Hz", "segment_length_m"], ascending=[True, False]
    ).reset_index(drop=True)
    return df


def plot_convergence(
    df: pd.DataFrame,
    *,
    response: str = "abs_Z",
    reference: float | None = None,
    figsize: tuple[float, float] = (7.5, 4.5),
    title: str | None = None,
):
    """Plot ``response`` versus ``segment_length`` (one line per frequency).

    The x-axis is logarithmic and **inverted** so finer
    resolutions sit on the right (the convergence "asymptote
    direction"). When the DataFrame contains a single frequency
    only, the legend is suppressed.

    Parameters
    ----------
    df
        DataFrame returned by :func:`convergence_study`.
    response
        Column name to plot. Default ``"abs_Z"``.
    reference
        Optional asymptotic value to draw as a horizontal dashed
        reference line — useful when an analytical reference
        (Sunde, Dwight, IEEE Std 80) is known.
    figsize, title
        Standard matplotlib options.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if response not in df.columns:
        raise KeyError(
            f"column '{response}' not in DataFrame; have: {list(df.columns)}"
        )
    if "segment_length_m" not in df.columns:
        raise KeyError(
            "DataFrame must have a 'segment_length_m' column "
            "(produced by convergence_study)."
        )

    fig, ax = plt.subplots(figsize=figsize)
    if "frequency_Hz" in df.columns and df["frequency_Hz"].nunique() > 1:
        for f_hz, sub in df.groupby("frequency_Hz"):
            sub = sub.sort_values("segment_length_m", ascending=False)
            ax.plot(
                sub["segment_length_m"], sub[response],
                marker="o", label=f"f = {f_hz:g} Hz",
            )
        ax.legend()
    else:
        sub = df.sort_values("segment_length_m", ascending=False)
        ax.plot(sub["segment_length_m"], sub[response], marker="o")

    if reference is not None:
        ax.axhline(
            float(reference), ls="--", color="k", alpha=0.6,
            label=f"reference = {reference:g}",
        )
        # Re-render legend so the reference line appears even in
        # the single-frequency case.
        ax.legend()

    ax.set_xscale("log")
    ax.invert_xaxis()  # finer ds (smaller value) on the right
    ax.set_xlabel("segment_length in m")
    ax.set_ylabel(response)
    ax.grid(True, which="both", alpha=0.3)
    if title is None:
        title = f"Convergence of {response} vs. segment_length"
    ax.set_title(title)
    fig.tight_layout()
    return fig
