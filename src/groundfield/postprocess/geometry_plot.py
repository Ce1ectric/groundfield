"""World-geometry plots (no solve required).

This module provides quick visualisations of the *physical world*
— electrodes, conductors and current sources — **before** the
solver runs. In default setups with several hundred electrodes
(200 EFH plus KVS plus substation plus measurement aux/probe) it
is very useful to inspect the geometry first to catch typos in
positions, accidental clusters, missing conductors, or sources
attached to the wrong electrode, **without** paying the cost of
a field solve.

Functions
---------
world_bounds_3d
    Smallest axis-aligned bounding box in :math:`(x, y, z)` of
    the world's electrodes plus conductor endpoints. Extension of
    :func:`groundfield.postprocess.plotting.world_bounds_xy` to
    three dimensions.
plot_world
    2-D top-down (``plane="xy"``) or vertical (``plane="xz"``)
    geometry plot. Electrodes drawn via the existing
    :func:`_draw_electrodes` helper of
    :mod:`groundfield.postprocess.plotting`; conductors as
    colour-coded line segments; sources as red star markers with
    an optional arrow to ``return_to``.
plot_world_3d
    3-D wireframe with the soil-surface plane shown in light
    grey and the :math:`z`-axis pointing downwards
    (groundfield convention; positive depth is below ground).

Conductor colour scheme
-----------------------
The conductor colour follows :data:`groundfield.conductors.ConductorType`:

==================  =========================  =====================
``conductor_type``  Colour                     Default style
==================  =========================  =====================
``pen``             ``#2c7a2c`` (green)        solid
``bare_copper``     ``#d97300`` (orange)       solid
``cable_shield``    ``#888888`` (grey)         solid
``overhead``        ``#1f77b4`` (steel blue)   solid
``generic``         ``#444444`` (dark grey)    solid
==================  =========================  =====================

The line **style** flags the soil-coupling mode of the conductor:
``coupling_to_soil = "galvanic"`` is drawn solid, ``"isolated"``
is drawn dashed.

Validity
--------
Pure geometry — no solver result is required. The plot does
*not* visualise any field quantity; for that, see
:func:`groundfield.postprocess.plotting.plot_potential_contour`
and friends. Best used as a debugging step before
``world.solve(...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.figure as mpl_fig

    from groundfield.conductors.conductor import Conductor
    from groundfield.sources import Source
    from groundfield.world import World

__all__ = [
    "world_bounds_3d",
    "plot_world",
    "plot_world_3d",
]


# ---------------------------------------------------------------------
# Conductor styling
# ---------------------------------------------------------------------

_CONDUCTOR_COLORS: dict[str, str] = {
    "pen": "#2c7a2c",
    "bare_copper": "#d97300",
    "cable_shield": "#888888",
    "overhead": "#1f77b4",
    "generic": "#444444",
}


def _conductor_style(c: "Conductor") -> tuple[str, str, float]:
    """Return ``(color, linestyle, linewidth)`` for a conductor."""
    color = _CONDUCTOR_COLORS.get(c.conductor_type, "#444444")
    linestyle = "-" if c.coupling_to_soil == "galvanic" else "--"
    linewidth = 1.6
    return color, linestyle, linewidth


# ---------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------


def world_bounds_3d(
    world: "World",
) -> tuple[float, float, float, float, float, float]:
    """Smallest axis-aligned :math:`(x, y, z)` bounding box of the world.

    Inspects every electrode in :attr:`World.electrodes` and every
    conductor endpoint in :attr:`World.conductors`, returning the
    six bounds ``(x_min, x_max, y_min, y_max, z_min, z_max)`` in
    metres.

    The :math:`(x, y)` portion is consistent with
    :func:`groundfield.postprocess.plotting.world_bounds_xy` but
    additionally includes conductor endpoints — important for
    overhead lines or measurement leads that extend well beyond
    the electrode footprint. The :math:`z` portion uses

    * :class:`RodElectrode`: ``[position[2], position[2] + length]``
      (head + foot of the rod).
    * :class:`RingElectrode`, :class:`StripElectrode`,
      :class:`MeshElectrode`, :class:`GridMeshElectrode`: the
      :math:`z` coordinate of the electrode (single buried depth).
    * Conductors: ``min/max`` over both endpoint :math:`z` values
      (catches overhead :math:`z<0` and buried :math:`z>0`).

    Returns
    -------
    tuple
        ``(x_min, x_max, y_min, y_max, z_min, z_max)`` in metres.
        For an empty world the result is the trivial
        ``(0, 0, 0, 0, 0, 0)``; callers should add positive
        padding before plotting.
    """
    from groundfield.geometry.electrodes import (
        GridMeshElectrode,
        MeshElectrode,
        RingElectrode,
        RodElectrode,
        StripElectrode,
    )

    if not world.electrodes and not world.conductors:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for e in world.electrodes:
        if isinstance(e, RodElectrode):
            x, y, z = e.position
            xs.append(x)
            ys.append(y)
            zs.extend([z, z + e.length])
        elif isinstance(e, RingElectrode):
            cx, cy, cz = e.center
            xs.extend([cx - e.radius, cx + e.radius])
            ys.extend([cy - e.radius, cy + e.radius])
            zs.append(cz)
        elif isinstance(e, StripElectrode):
            xs.extend([e.start[0], e.end[0]])
            ys.extend([e.start[1], e.end[1]])
            zs.extend([e.start[2], e.end[2]])
        elif isinstance(e, (MeshElectrode, GridMeshElectrode)):
            cx, cy, cz = e.corner
            dx, dy = e.size
            xs.extend([cx, cx + dx])
            ys.extend([cy, cy + dy])
            zs.append(cz)
        else:  # pragma: no cover — defensive
            cp = e.connection_point
            xs.append(cp[0])
            ys.append(cp[1])
            zs.append(cp[2])

    for c in world.conductors:
        for endpoint in (c.start, c.end):
            xs.append(endpoint[0])
            ys.append(endpoint[1])
            zs.append(endpoint[2])

    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


# ---------------------------------------------------------------------
# 2-D world plot
# ---------------------------------------------------------------------


def _project_2d(
    point: tuple[float, float, float], plane: str
) -> tuple[float, float]:
    if plane == "xy":
        return point[0], point[1]
    if plane == "xz":
        return point[0], point[2]
    raise ValueError(f"plane must be 'xy' or 'xz', got {plane!r}.")


def _draw_conductor_2d(ax, c: "Conductor", plane: str) -> None:
    color, linestyle, linewidth = _conductor_style(c)
    p1 = _project_2d(c.start, plane)
    p2 = _project_2d(c.end, plane)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.9,
            zorder=3, label="_nolegend_")


def _draw_source_2d(
    ax, source: "Source", world: "World", plane: str
) -> None:
    """Draw a source as a red star at its anchor; arrow to return_to if any."""
    try:
        anchor_e = world.get_electrode(source.attached_to)
    except KeyError:
        # The source might be attached to a conductor — for now skip
        # the marker; the conductor itself is already visible.
        return
    a = _project_2d(anchor_e.connection_point, plane)
    ax.plot(a[0], a[1], marker="*", color="#cc0000", markersize=12,
            markeredgecolor="black", linewidth=0, zorder=5,
            label="_nolegend_")
    if getattr(source, "return_to", None):
        try:
            ret_e = world.get_electrode(source.return_to)
        except KeyError:
            return
        b = _project_2d(ret_e.connection_point, plane)
        ax.annotate(
            "",
            xy=b, xytext=a,
            arrowprops={"arrowstyle": "->", "color": "#cc0000",
                        "alpha": 0.6, "linewidth": 1.0,
                        "shrinkA": 6, "shrinkB": 6},
            zorder=4,
        )


def _build_legend(ax, world: "World", show_conductors: bool, show_sources: bool) -> None:
    """Build a synthetic legend covering the conductor types and source marker."""
    handles = []
    labels = []
    if show_conductors and world.conductors:
        types_present = sorted({c.conductor_type for c in world.conductors})
        couplings_present = sorted({c.coupling_to_soil for c in world.conductors})
        for t in types_present:
            color = _CONDUCTOR_COLORS.get(t, "#444444")
            handles.append(_legend_line(color, "-"))
            labels.append(f"conductor: {t}")
        if "isolated" in couplings_present and "galvanic" in couplings_present:
            handles.append(_legend_line("#444444", "-"))
            labels.append("solid: galvanic")
            handles.append(_legend_line("#444444", "--"))
            labels.append("dashed: isolated")
    if show_sources and world.sources:
        from matplotlib.lines import Line2D
        handles.append(
            Line2D([], [], marker="*", color="#cc0000", markersize=10,
                   markeredgecolor="black", linewidth=0)
        )
        labels.append("source")
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=8, framealpha=0.9)


def _legend_line(color: str, linestyle: str):
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, linestyle=linestyle, linewidth=1.6)


def plot_world(
    world: "World",
    *,
    plane: Literal["xy", "xz"] = "xy",
    extent: tuple[float, float, float, float] | None = None,
    padding_m: float = 5.0,
    show_conductors: bool = True,
    show_sources: bool = True,
    annotate_electrodes: bool = False,
    figsize: tuple[float, float] = (8.0, 6.0),
    ax=None,
    title: str | None = None,
):
    """Top-down or vertical 2-D geometry plot of a :class:`World`.

    Pure-geometry visualisation; no field quantity is evaluated
    and no solver is invoked. Useful as a sanity check before
    :meth:`World.solve` on a large typical network.

    Parameters
    ----------
    world
        World to draw.
    plane
        ``"xy"`` (default — top-down) or ``"xz"`` (side / vertical
        slice).
    extent
        Optional explicit ``(a_min, a_max, b_min, b_max)`` of the
        plotted area in metres. ``None`` (default) derives the
        extent from :func:`world_bounds_3d` plus ``padding_m``.
    padding_m
        Extra padding on each side of the bounding box in metres
        (used only when ``extent`` is ``None``).
    show_conductors
        If ``True`` (default), draw every entry of
        :attr:`World.conductors` as a colour-coded line.
    show_sources
        If ``True`` (default), draw every entry of
        :attr:`World.sources` as a red star at its anchor
        electrode, plus an arrow to ``return_to`` if set.
    annotate_electrodes
        If ``True``, attach a small text label with the electrode
        name to each electrode. Default ``False`` because
        production-grade worlds with > 50 electrodes look cluttered.
    figsize
        Matplotlib figure size in inches (used when ``ax`` is
        ``None``).
    ax
        Optional pre-existing :class:`matplotlib.axes.Axes`. When
        passed, ``figsize`` is ignored and the host figure is
        returned.
    title
        Optional title override; default ``World 'name' — plane
        geometry``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from groundfield.postprocess.plotting import _draw_electrodes

    if extent is None:
        bounds = world_bounds_3d(world)
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        if plane == "xy":
            extent = (
                x_min - padding_m, x_max + padding_m,
                y_min - padding_m, y_max + padding_m,
            )
        elif plane == "xz":
            # In z, "padding" should not push the surface (z=0)
            # below the soil — but z<0 (overhead) is legitimate.
            # We pad symmetrically and rely on plot inversion.
            extent = (
                x_min - padding_m, x_max + padding_m,
                z_min - padding_m, z_max + padding_m,
            )
        else:
            raise ValueError(f"plane must be 'xy' or 'xz', got {plane!r}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # 1. Surface line for the xz view — emphasises the soil
    #    boundary at z = 0.
    if plane == "xz":
        ax.axhline(0.0, color="#666666", linestyle=":", linewidth=0.8,
                   alpha=0.7, zorder=1)

    # 2. Conductors first (so electrodes draw on top).
    if show_conductors:
        for c in world.conductors:
            _draw_conductor_2d(ax, c, plane)

    # 3. Electrodes — reuse the existing helper for consistency
    #    with the field plots.
    _draw_electrodes(ax, world, plane)

    # 4. Sources on top.
    if show_sources:
        for s in world.sources:
            _draw_source_2d(ax, s, world, plane)

    # 5. Optional annotations.
    if annotate_electrodes:
        for e in world.electrodes:
            cp = e.connection_point
            x, y = _project_2d(cp, plane)
            ax.annotate(
                e.name, (x, y),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, alpha=0.85, zorder=6,
            )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if plane == "xy":
        ax.set_xlabel("x in m")
        ax.set_ylabel("y in m")
    else:
        ax.set_xlabel("x in m")
        ax.set_ylabel("z in m (depth)")
        ax.invert_yaxis()  # soil downwards

    if title is None:
        n_e = len(world.electrodes)
        n_c = len(world.conductors)
        n_s = len(world.sources)
        title = (
            f"World '{world.name}' — {plane} geometry "
            f"({n_e} electrodes, {n_c} conductors, {n_s} sources)"
        )
    ax.set_title(title)

    _build_legend(ax, world, show_conductors, show_sources)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# 3-D world plot
# ---------------------------------------------------------------------


def _draw_electrode_3d(ax, e) -> None:
    """Draw one electrode in 3-D using its native geometric shape."""
    from groundfield.geometry.electrodes import (
        GridMeshElectrode,
        MeshElectrode,
        RingElectrode,
        RodElectrode,
        StripElectrode,
    )

    kw = {"color": "k", "linewidth": 1.6, "alpha": 0.9}
    if isinstance(e, RodElectrode):
        x, y, z = e.position
        ax.plot([x, x], [y, y], [z, z + e.length], **kw)
    elif isinstance(e, RingElectrode):
        cx, cy, cz = e.center
        theta = np.linspace(0, 2 * np.pi, 64)
        ax.plot(cx + e.radius * np.cos(theta),
                cy + e.radius * np.sin(theta),
                np.full_like(theta, cz),
                **kw)
    elif isinstance(e, StripElectrode):
        sx, sy, sz = e.start
        ex, ey, ez = e.end
        ax.plot([sx, ex], [sy, ey], [sz, ez], **kw)
    elif isinstance(e, (MeshElectrode, GridMeshElectrode)):
        cx, cy, cz = e.corner
        dx, dy = e.size
        # Outer rectangle.
        xs = [cx, cx + dx, cx + dx, cx, cx]
        ys = [cy, cy, cy + dy, cy + dy, cy]
        zs = [cz] * 5
        ax.plot(xs, ys, zs, **kw)
        if isinstance(e, GridMeshElectrode):
            for k in range(1, e.n_x):
                xk = cx + dx * k / e.n_x
                ax.plot([xk, xk], [cy, cy + dy], [cz, cz],
                        color="k", linewidth=0.7, alpha=0.5)
            for k in range(1, e.n_y):
                yk = cy + dy * k / e.n_y
                ax.plot([cx, cx + dx], [yk, yk], [cz, cz],
                        color="k", linewidth=0.7, alpha=0.5)
    else:  # pragma: no cover — defensive
        cp = e.connection_point
        ax.scatter([cp[0]], [cp[1]], [cp[2]], color="k", s=20)


def _draw_conductor_3d(ax, c: "Conductor") -> None:
    color, linestyle, linewidth = _conductor_style(c)
    ax.plot(
        [c.start[0], c.end[0]],
        [c.start[1], c.end[1]],
        [c.start[2], c.end[2]],
        color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.9,
    )


def _draw_source_3d(ax, source: "Source", world: "World") -> None:
    try:
        anchor_e = world.get_electrode(source.attached_to)
    except KeyError:
        return
    cp = anchor_e.connection_point
    ax.scatter([cp[0]], [cp[1]], [cp[2]],
               marker="*", color="#cc0000", s=140,
               edgecolor="black", linewidth=0.5,
               depthshade=False, zorder=10)


def plot_world_3d(
    world: "World",
    *,
    show_conductors: bool = True,
    show_sources: bool = True,
    show_surface: bool = True,
    figsize: tuple[float, float] = (9.0, 7.0),
    elev: float = 22.0,
    azim: float = -55.0,
    title: str | None = None,
):
    """3-D wireframe of a :class:`World` using ``mpl_toolkits.mplot3d``.

    The :math:`z` axis is **inverted** so that depth points
    downwards on screen (groundfield convention: positive
    :math:`z` is into the soil). A faint grey surface plane at
    :math:`z = 0` marks the soil surface.

    Parameters
    ----------
    world
        World to draw.
    show_conductors, show_sources
        See :func:`plot_world`.
    show_surface
        If ``True`` (default), render a translucent grey square
        at :math:`z = 0` over the world's :math:`(x, y)` bounding
        box plus 5 m padding.
    figsize
        Matplotlib figure size in inches.
    elev, azim
        Initial viewing angles (passed to
        :meth:`Axes3D.view_init`).
    title
        Optional title override.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    bounds = world_bounds_3d(world)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Optional surface plane at z = 0.
    if show_surface and (world.electrodes or world.conductors):
        pad = 5.0
        xs = np.array([[x_min - pad, x_max + pad],
                       [x_min - pad, x_max + pad]])
        ys = np.array([[y_min - pad, y_min - pad],
                       [y_max + pad, y_max + pad]])
        zs = np.zeros_like(xs)
        ax.plot_surface(xs, ys, zs, color="#cccccc", alpha=0.25,
                        edgecolor="none", linewidth=0)

    if show_conductors:
        for c in world.conductors:
            _draw_conductor_3d(ax, c)

    for e in world.electrodes:
        _draw_electrode_3d(ax, e)

    if show_sources:
        for s in world.sources:
            _draw_source_3d(ax, s, world)

    ax.set_xlabel("x in m")
    ax.set_ylabel("y in m")
    ax.set_zlabel("z in m (depth)")
    # Invert z so that depth points downwards on screen.
    ax.invert_zaxis()
    ax.view_init(elev=elev, azim=azim)
    if title is None:
        title = (
            f"World '{world.name}' — 3-D geometry "
            f"({len(world.electrodes)} electrodes, "
            f"{len(world.conductors)} conductors, "
            f"{len(world.sources)} sources)"
        )
    ax.set_title(title)
    fig.tight_layout()
    return fig
