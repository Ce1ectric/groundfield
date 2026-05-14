"""Current sharing and split-factor analysis for a ``FieldResult``.

This module turns the per-electrode currents stored in
:class:`groundfield.solver.result.FieldResult` into the engineering
quantities used to answer the research question *"where does the
injected source current actually return?"*. In a TN-Ortsnetz with
hundreds of houses, dozens of cable cabinets and one or more
metallic return paths (PEN trunk, measurement leads, cable
shields), the soil-leakage current of every cluster matters and
cannot be read off ``electrode_currents`` directly without aggregation.

Quantities
----------
- **Per-cluster soil leakage**
  :math:`I_{c} = \\sum_{e \\in c} I_e`. With ideal galvanic
  bonds (``cross_section=None``) the cluster members share a
  potential :math:`U_c`; the cluster impedance is then
  :math:`Z_c = U_c / I_c`. Returned as a tabular summary by
  :func:`cluster_current_balance`.
- **Per-electrode share of cluster current** — what fraction of
  the cluster's net soil leakage flows through this specific
  electrode:
  :math:`s_{e \\mid c} = I_e / I_c` (complex). Returned alongside
  the per-electrode potential / impedance by
  :func:`electrode_current_table`.
- **Split factor**
  :math:`s = I_{c_\\text{src}} / I_\\text{src} =
  \\sum_{e \\in c_\\text{src}} I_e / I_\\text{src}`,
  with :math:`c_\\text{src}` the cluster of the source's
  ``attached_to`` electrode. ``s = 1`` means the injected current
  leaves the source cluster *entirely* through the soil (no
  metallic short-cut to the return-path cluster). ``s < 1`` means
  a metallic conductor (PEN trunk, parallel measurement lead,
  cable shield) carries part of the current as a parallel
  resistive path.

  This is the **galvanic** current-split between parallel paths.
  It is **not** the *Reduktionsfaktor* in the German EVU /
  Schirmtechnik sense (Oeding & Oswald 2016): that latter
  quantity is the additional **transformatorische / inductive
  coupling correction** between a current-carrying conductor and
  a parallel grounding / shield conductor. The Reduktionsfaktor
  vanishes when the two conductors are perpendicular (no flux
  linkage) but the split factor still applies — the current
  always splits among parallel resistive paths irrespective of
  the geometric angle. A future helper may add the proper
  Reduktionsfaktor based on the Neumann mutual-inductance / Carson
  / Sommerfeld backends already present in :mod:`groundfield.coupling`.

Validity envelope
-----------------
* Frequency: quasi-static envelope :math:`f \\le 1\\,\\mathrm{kHz}`.
* Conventions: :class:`FieldResult.electrode_currents` carries
  the **per-electrode soil-leakage current** in A (complex
  phasor), with positive sign in the direction *electrode → soil*.
  All quantities here are complex per ``frequency_index``.
* Cluster discovery uses :attr:`FieldResult.clusters` — the
  same map produced by every backend.
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.figure as mpl_fig

    from groundfield.solver.result import FieldResult
    from groundfield.sources import Source
    from groundfield.world import World

__all__ = [
    "cluster_current_balance",
    "electrode_current_table",
    "split_factor",
    "plot_current_sharing",
]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _source_current(source: "Source") -> complex:
    """Complex phasor of a current source — :math:`I = |I|\\,e^{j\\varphi}`."""
    return float(source.magnitude) * cmath.exp(
        1j * math.radians(float(source.phase_deg))
    )


def _unique_clusters(result: "FieldResult") -> dict[str, list[str]]:
    """Deduplicated cluster mapping ``cluster_root -> sorted members``.

    Uses the lexicographically smallest member as the cluster
    identifier, mirroring the convention of
    :meth:`FieldResult.cluster_impedance`.
    """
    out: dict[str, list[str]] = {}
    for members in result.clusters.values():
        if not members:
            continue
        root = sorted(members)[0]
        out.setdefault(root, sorted(members))
    return out


def _resolve_source(world: "World", source_name: str | None) -> "Source":
    """Pick a current source from the world (single-source default)."""
    current_sources = [s for s in world.sources if s.kind == "current"]
    if not current_sources:
        raise ValueError("World contains no current source.")
    if source_name is None:
        if len(current_sources) > 1:
            raise ValueError(
                "World contains multiple current sources "
                f"({[s.name for s in current_sources]}); please pass "
                "``source_name`` explicitly."
            )
        return current_sources[0]
    for s in current_sources:
        if s.name == source_name:
            return s
    raise KeyError(
        f"Current source '{source_name}' not found. Known: "
        f"{[s.name for s in current_sources]}."
    )


# ---------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------


def cluster_current_balance(
    result: "FieldResult",
    *,
    frequency_index: int = 0,
) -> pd.DataFrame:
    """Per-cluster soil leakage, potential and impedance.

    Returns one row per *unique* galvanic cluster present in the
    result. The cluster impedance is computed as
    :math:`Z_c = U_c / I_c` with :math:`I_c = \\sum_{e \\in c} I_e`.
    Where :math:`I_c = 0` (purely passive observer cluster) the
    impedance columns are ``NaN``.

    Parameters
    ----------
    result
        Solver output.
    frequency_index
        Index into :attr:`FieldResult.frequencies`. Default 0.

    Returns
    -------
    pandas.DataFrame
        Columns
        ``cluster_root``,
        ``n_members``,
        ``members`` (list of strings),
        ``U_re``, ``U_im``, ``abs_U``,
        ``sum_I_re``, ``sum_I_im``, ``abs_sum_I``,
        ``Z_re``, ``Z_im``, ``abs_Z``, ``arg_Z_deg``.
        Rows are sorted by descending ``abs_sum_I`` so the
        dominant clusters surface first — the typical
        debugging order.
    """
    rows = []
    for root, members in _unique_clusters(result).items():
        # Cluster potential = potential of any member (ideal cluster
        # bond -> shared phi). Use the canonical root.
        U = complex(result.electrode_potentials[root][frequency_index])
        sum_I = 0.0 + 0j
        for m in members:
            if m not in result.electrode_currents:
                continue
            sum_I += complex(result.electrode_currents[m][frequency_index])
        if sum_I != 0:
            Z = U / sum_I
        else:
            Z = complex("nan")
        rows.append(
            {
                "cluster_root": root,
                "n_members": len(members),
                "members": list(members),
                "U_re": U.real,
                "U_im": U.imag,
                "abs_U": abs(U),
                "sum_I_re": sum_I.real,
                "sum_I_im": sum_I.imag,
                "abs_sum_I": abs(sum_I),
                "Z_re": Z.real if np.isfinite(Z.real) else float("nan"),
                "Z_im": Z.imag if np.isfinite(Z.imag) else float("nan"),
                "abs_Z": (
                    abs(Z) if (np.isfinite(Z.real) and np.isfinite(Z.imag)) else float("nan")
                ),
                "arg_Z_deg": (
                    math.degrees(cmath.phase(Z))
                    if (np.isfinite(Z.real) and np.isfinite(Z.imag))
                    else float("nan")
                ),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("abs_sum_I", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Per-electrode table
# ---------------------------------------------------------------------


def electrode_current_table(
    result: "FieldResult",
    world: "World | None" = None,
    *,
    frequency_index: int = 0,
) -> pd.DataFrame:
    """Per-electrode potential, current and share of cluster current.

    For each electrode in the result the table reports the
    potential :math:`U_e` (= cluster potential), the soil-leakage
    current :math:`I_e`, the implied two-terminal impedance
    :math:`Z_e = U_e / I_e`, and the **fractional share** of the
    cluster total :math:`s_{e \\mid c} = I_e / I_c`. The latter is
    the engineering key indicator for "of all the soil current
    leaving cluster *c*, how much physically flows through this
    specific electrode" — useful when looking at a single
    transformer station with a ring + several rods, or a building
    cluster with multiple foundation parts.

    Parameters
    ----------
    result
        Solver output.
    world
        Optional companion world. When given, the table includes
        the electrode kind (``rod`` / ``ring`` / ``mesh`` / ...)
        and the connection-point depth in metres — a small but
        very helpful annotation when scanning a 200-EFH typical run.
    frequency_index
        Index into :attr:`FieldResult.frequencies`. Default 0.

    Returns
    -------
    pandas.DataFrame
        Columns
        ``name``, ``cluster_root``, ``kind`` (if ``world``),
        ``depth_m`` (if ``world``),
        ``U_re``, ``U_im``, ``abs_U``,
        ``I_re``, ``I_im``, ``abs_I``,
        ``Z_re``, ``Z_im``, ``abs_Z``, ``arg_Z_deg``,
        ``share_of_cluster_re``, ``share_of_cluster_im``,
        ``share_of_cluster_pct``. Sorted by descending ``abs_I``.
    """
    # Map every electrode to its cluster root and to the cluster's
    # net soil-leakage current at the chosen frequency.
    cluster_root_of: dict[str, str] = {}
    cluster_sum_I: dict[str, complex] = {}
    for root, members in _unique_clusters(result).items():
        s = 0.0 + 0j
        for m in members:
            if m in result.electrode_currents:
                s += complex(result.electrode_currents[m][frequency_index])
        cluster_sum_I[root] = s
        for m in members:
            cluster_root_of[m] = root

    geo_lookup: dict[str, tuple[str, float]] = {}
    if world is not None:
        for e in world.electrodes:
            cp = e.connection_point
            geo_lookup[e.name] = (e.kind, float(cp[2]))

    rows = []
    for name in sorted(result.electrode_potentials.keys()):
        U = complex(result.electrode_potentials[name][frequency_index])
        I = complex(
            result.electrode_currents.get(name, [0.0 + 0j] * (frequency_index + 1))[
                frequency_index
            ]
        )
        if I != 0:
            Z = U / I
        else:
            Z = complex("nan")
        root = cluster_root_of.get(name, name)
        I_c = cluster_sum_I.get(root, 0.0 + 0j)
        if I_c != 0:
            s_e = I / I_c
            s_pct = abs(s_e) * 100.0
        else:
            s_e = complex("nan")
            s_pct = float("nan")

        row: dict[str, object] = {
            "name": name,
            "cluster_root": root,
        }
        if world is not None:
            kind, depth = geo_lookup.get(name, ("?", float("nan")))
            row["kind"] = kind
            row["depth_m"] = depth
        row.update(
            {
                "U_re": U.real,
                "U_im": U.imag,
                "abs_U": abs(U),
                "I_re": I.real,
                "I_im": I.imag,
                "abs_I": abs(I),
                "Z_re": Z.real if np.isfinite(Z.real) else float("nan"),
                "Z_im": Z.imag if np.isfinite(Z.imag) else float("nan"),
                "abs_Z": (
                    abs(Z) if (np.isfinite(Z.real) and np.isfinite(Z.imag)) else float("nan")
                ),
                "arg_Z_deg": (
                    math.degrees(cmath.phase(Z))
                    if (np.isfinite(Z.real) and np.isfinite(Z.imag))
                    else float("nan")
                ),
                "share_of_cluster_re": (
                    s_e.real if np.isfinite(s_e.real) else float("nan")
                ),
                "share_of_cluster_im": (
                    s_e.imag if np.isfinite(s_e.imag) else float("nan")
                ),
                "share_of_cluster_pct": s_pct,
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("abs_I", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Split factor
# ---------------------------------------------------------------------


def split_factor(
    result: "FieldResult",
    world: "World",
    *,
    source_name: str | None = None,
    frequency_index: int = 0,
) -> complex:
    """Split factor :math:`s = I_{c_\\text{src}} / I_\\text{src}`.

    Computes the **galvanic split factor** of a current source:
    the fraction of the injected current that leaves the source
    cluster through the **soil** rather than through any metallic
    parallel path (PEN trunk, parallel measurement lead, cable
    shield).

    Mathematically

    .. math::

        s \\;=\\; \\frac{\\sum_{e \\in c_\\text{src}} I_e}{I_\\text{src}},

    with :math:`c_\\text{src}` the galvanic cluster of the source's
    ``attached_to`` electrode and :math:`I_\\text{src}` the complex
    phasor of the source. By construction:

    * For a stand-alone source on an electrode without metallic
      parallel paths, KCL forces :math:`s = 1 + 0\\,j`. The
      injected current leaves the cluster entirely through the
      soil.
    * For a source whose cluster is connected to *another*
      grounding cluster (e.g. the auxiliary electrode of a
      fall-of-potential measurement) by a metallic conductor,
      :math:`s < 1`. The smaller the conductor's series
      impedance, the smaller :math:`s`.
    * The complex argument :math:`\\arg s` reveals the inductive
      content of the parallel path — a Carson- or Sommerfeld-
      corrected lead at frequency >0 produces an imaginary share.

    Not to be confused with the *Reduktionsfaktor*
    -----------------------------------------------
    In the German EVU / Schirmtechnik literature (Oeding & Oswald
    2016), the *Reduktionsfaktor* refers to the **transformatorische
    / inductive coupling correction** between a current-carrying
    conductor and a parallel grounding / shield conductor. That
    quantity is angle-dependent: it vanishes when the two conductors
    are perpendicular (no flux linkage) but is large for collinear
    runs.

    The split factor implemented here is **purely galvanic** — the
    resistive division of current across parallel paths. It is
    present whenever there are multiple parallel paths, regardless
    of their geometric orientation.

    Parameters
    ----------
    result
        Solver output.
    world
        Companion world used to resolve the source object.
    source_name
        Optional name of the current source. ``None`` (default)
        picks the unique current source; raises if the world
        contains more than one.
    frequency_index
        Index into :attr:`FieldResult.frequencies`. Default 0.

    Returns
    -------
    complex
        Phasor :math:`s(f)` (dimensionless).

    Raises
    ------
    ValueError
        If the world has no current source, multiple current
        sources without explicit ``source_name``, or the source
        has zero magnitude.
    KeyError
        If ``source_name`` does not exist among current sources,
        or the source's ``attached_to`` electrode is unknown to
        the result.
    """
    source = _resolve_source(world, source_name)
    I_src = _source_current(source)
    if abs(I_src) == 0.0:
        raise ValueError(
            f"Source '{source.name}' has zero magnitude — split factor "
            "is undefined."
        )

    attached = source.attached_to
    members = result.clusters.get(attached, [attached])
    if not members:
        raise KeyError(
            f"Source '{source.name}' is attached to '{attached}', "
            "which has no cluster entry in the result."
        )

    sum_I = 0.0 + 0j
    for m in members:
        if m in result.electrode_currents:
            sum_I += complex(result.electrode_currents[m][frequency_index])
    return sum_I / I_src


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------


def plot_current_sharing(
    result: "FieldResult",
    world: "World | None" = None,
    *,
    by: Literal["electrode", "cluster"] = "electrode",
    top_n: int = 15,
    frequency_index: int = 0,
    figsize: tuple[float, float] = (8.0, 5.0),
    title: str | None = None,
):
    """Bar chart of the ``top_n`` current contributors.

    Renders ``|I|`` (in A) for either every electrode or every
    galvanic cluster, sorted descending. The default ``by =
    "electrode"`` is the default — *which physical electrode
    actually carries the test current?* The companion option
    ``by = "cluster"`` aggregates over each cluster.

    Parameters
    ----------
    result
        Solver output.
    world
        Optional companion world. Forwarded to
        :func:`electrode_current_table` for the kind / depth
        annotations; ignored otherwise.
    by
        ``"electrode"`` (default) or ``"cluster"``.
    top_n
        Maximum number of bars to render. Smaller contributors are
        truncated. Pass ``0`` to render all.
    frequency_index
        Index into :attr:`FieldResult.frequencies`.
    figsize
        Matplotlib figure size in inches.
    title
        Optional title override.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if by == "electrode":
        df = electrode_current_table(
            result, world=world, frequency_index=frequency_index
        )
        labels = df["name"].tolist()
        magnitudes = df["abs_I"].to_numpy()
        ylabel = "|I| in A (per electrode)"
    elif by == "cluster":
        df = cluster_current_balance(result, frequency_index=frequency_index)
        labels = df["cluster_root"].tolist()
        magnitudes = df["abs_sum_I"].to_numpy()
        ylabel = "|ΣI| in A (per cluster)"
    else:
        raise ValueError(f"by must be 'electrode' or 'cluster', got {by!r}.")

    if top_n and top_n > 0:
        labels = labels[:top_n]
        magnitudes = magnitudes[:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(labels)), magnitudes, color="C0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    if title is None:
        f_hz = result.frequencies[frequency_index]
        suffix = f"top {top_n}" if (top_n and top_n > 0) else "all"
        title = f"Current sharing by {by} — {suffix}, f = {f_hz:g} Hz"
    ax.set_title(title)
    fig.tight_layout()
    return fig
