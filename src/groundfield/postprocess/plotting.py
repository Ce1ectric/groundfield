"""Matplotlib plots for potential distributions and profiles.

Functions
---------
plot_potential_contour
    Contour / pseudo-colour plot of the potential in a slice plane
    (``xy`` at fixed $z$ or ``xz`` at fixed $y$).
plot_potential_profile
    Line plot of the potential along an arbitrary direction for one
    or several $z$ depths.
plot_potential_radial
    Radial profile $\\varphi(r)$ starting from an electrode for
    several depths — the standard "how far does the trumpet reach?"
    plot.

All functions return the ``matplotlib.figure.Figure`` they produce so
that notebooks can post-process them (titles, save, sub-plots).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.figure as mpl_fig

    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = [
    "plot_potential_contour",
    "plot_potential_profile",
    "plot_potential_radial",
    "plot_surface_potential",
    "world_bounds_xy",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _draw_electrodes(ax, world: "World", plane: str) -> None:
    """Draw the electrodes' geometric outlines onto an existing axes.

    Names are *not* annotated on purpose: in dense AP1 networks
    (200 EFH plus KVS plus substation plus measurement aux/probe
    electrodes) the labels overlap heavily and clutter the
    surface-potential plot. If a name annotation is genuinely
    needed, the caller can iterate ``world.electrodes`` and call
    :py:meth:`matplotlib.axes.Axes.annotate` themselves.

    Parameters
    ----------
    plane
        ``"xy"`` or ``"xz"`` — selects the projection.
    """
    from groundfield.geometry.electrodes import (
        GridMeshElectrode,
        MeshElectrode,
        RingElectrode,
        RodElectrode,
        StripElectrode,
    )

    for e in world.electrodes:
        if isinstance(e, RodElectrode):
            x, y, z = e.position
            if plane == "xy":
                ax.plot(x, y, "ks", markersize=6, label="_nolegend_")
            else:  # xz
                ax.plot([x, x], [z, z + e.length], "k-", linewidth=2)
        elif isinstance(e, RingElectrode):
            cx, cy, cz = e.center
            theta = np.linspace(0, 2 * np.pi, 64)
            xs = cx + e.radius * np.cos(theta)
            ys = cy + e.radius * np.sin(theta)
            if plane == "xy":
                ax.plot(xs, ys, "k-", linewidth=1.5)
            else:  # xz
                # Side view: cannot pick out the y-slice without the
                # slice y-value. Draw the projected width at depth cz.
                ax.plot([cx - e.radius, cx + e.radius], [cz, cz],
                        "k-", linewidth=1.5)
        elif isinstance(e, StripElectrode):
            sx, sy, sz = e.start
            ex, ey, ez = e.end
            if plane == "xy":
                ax.plot([sx, ex], [sy, ey], "k-", linewidth=1.8)
            else:  # xz — project both endpoints onto x; depth is fixed
                ax.plot([sx, ex], [sz, sz], "k-", linewidth=1.8)
        elif isinstance(e, (MeshElectrode, GridMeshElectrode)):
            cx, cy, cz = e.corner
            dx, dy = e.size
            if plane == "xy":
                # Outer rectangle …
                ax.plot([cx, cx + dx, cx + dx, cx, cx],
                        [cy, cy, cy + dy, cy + dy, cy], "k-", linewidth=1.5)
                # … plus inner mesh wires for GridMeshElectrode.
                if isinstance(e, GridMeshElectrode):
                    for k in range(1, e.n_x):
                        x_k = cx + dx * k / e.n_x
                        ax.plot([x_k, x_k], [cy, cy + dy],
                                "k-", linewidth=0.7, alpha=0.6)
                    for k in range(1, e.n_y):
                        y_k = cy + dy * k / e.n_y
                        ax.plot([cx, cx + dx], [y_k, y_k],
                                "k-", linewidth=0.7, alpha=0.6)
            else:
                ax.plot([cx, cx + dx], [cz, cz], "k-", linewidth=1.5)


def _make_grid(
    plane: Literal["xy", "xz"],
    extent: tuple[float, float, float, float],
    fixed: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2-D grid and the flattened array of evaluation points."""
    a_min, a_max, b_min, b_max = extent
    a = np.linspace(a_min, a_max, n)
    b = np.linspace(b_min, b_max, n)
    A, B = np.meshgrid(a, b)
    flat = np.empty((A.size, 3))
    if plane == "xy":
        flat[:, 0] = A.ravel()
        flat[:, 1] = B.ravel()
        flat[:, 2] = fixed
    elif plane == "xz":
        flat[:, 0] = A.ravel()
        flat[:, 1] = fixed
        flat[:, 2] = B.ravel()
    else:
        raise ValueError(f"plane must be 'xy' or 'xz', got {plane!r}.")
    return A, B, flat


# ---------------------------------------------------------------------
# Contour plot
# ---------------------------------------------------------------------


def plot_potential_contour(
    result: "FieldResult",
    *,
    world: "World | None" = None,
    plane: Literal["xy", "xz"] = "xy",
    z: float = 0.0,
    y: float = 0.0,
    extent: tuple[float, float, float, float] | None = None,
    n: int = 120,
    frequency_index: int = 0,
    levels: int = 20,
    log: bool = False,
    cmap: str = "viridis",
):
    """Contour plot of the potential in a slice plane.

    Parameters
    ----------
    result
        Result object from :meth:`Engine.solve`.
    world
        Optional companion world; if given, electrodes are drawn into
        the plot.
    plane
        ``"xy"`` (horizontal slice at depth $z$) or ``"xz"``
        (vertical slice at $y$).
    z, y
        Depth or $y$ value of the slice plane in metres.
    extent
        ``(a_min, a_max, b_min, b_max)`` of the slice plane in metres.
        Default is derived from the point-source distribution.
    n
        Resolution per axis.
    frequency_index
        Index into :attr:`FieldResult.frequencies`.
    levels
        Number of contour levels.
    log
        Logarithmic colour scale (better for fields that span several
        decades).
    cmap
        Matplotlib colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if extent is None:
        srcs = np.array([ps.position for ps in result.point_sources])
        if plane == "xy":
            ax = srcs[:, 0]
            bx = srcs[:, 1]
        else:
            ax = srcs[:, 0]
            bx = srcs[:, 2]
        pad = max(5.0, 0.5 * (ax.ptp() + bx.ptp()))
        extent = (ax.min() - pad, ax.max() + pad,
                  bx.min() - pad, bx.max() + pad)

    fixed = z if plane == "xy" else y
    A, B, flat = _make_grid(plane, extent, fixed, n)

    phi = result.potential(flat, frequency_index=frequency_index).real
    phi = phi.reshape(A.shape)

    fig, axx = plt.subplots(figsize=(7, 5))
    if log:
        # Drop non-positive values, otherwise the log scale breaks.
        phi_plot = np.where(phi > 0, phi, np.nan)
        cs = axx.contourf(A, B, phi_plot, levels=levels, cmap=cmap,
                          locator=__import__("matplotlib.ticker",
                                             fromlist=["LogLocator"]).LogLocator())
    else:
        cs = axx.contourf(A, B, phi, levels=levels, cmap=cmap)
    contours = axx.contour(A, B, phi, levels=levels, colors="k",
                           linewidths=0.4, alpha=0.5)
    axx.clabel(contours, inline=True, fontsize=7, fmt="%.0f")

    cbar = fig.colorbar(cs, ax=axx)
    cbar.set_label("Potential φ in V")

    if plane == "xy":
        axx.set_xlabel("x in m")
        axx.set_ylabel("y in m")
        axx.set_title(f"Potential φ(x, y, z={z:g} m), "
                      f"f={result.frequencies[frequency_index]} Hz")
        axx.set_aspect("equal")
    else:
        axx.set_xlabel("x in m")
        axx.set_ylabel("z in m (depth)")
        axx.invert_yaxis()  # soil downwards
        axx.set_title(f"Potential φ(x, y={y:g} m, z), "
                      f"f={result.frequencies[frequency_index]} Hz")

    if world is not None:
        _draw_electrodes(axx, world, plane)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Line / radial profiles
# ---------------------------------------------------------------------


def plot_potential_profile(
    result: "FieldResult",
    *,
    start: tuple[float, float, float],
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    distance: float = 30.0,
    n: int = 200,
    depths: Iterable[float] | None = None,
    frequency_index: int = 0,
):
    """Potential along an arbitrary line, optionally for several depths.

    Parameters
    ----------
    result
        Result object.
    start
        Start point ``(x, y, z)`` of the line in metres. ``z`` is
        overridden by ``depths`` if given.
    direction
        Direction vector (will be normalised).
    distance
        Length of the line in metres.
    n
        Number of evaluation points.
    depths
        List of depths $z$ in metres. One curve per depth.
    frequency_index
        Index into :attr:`FieldResult.frequencies`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if depths is None:
        depths = [0.0]
    direction_arr = np.asarray(direction, dtype=float)
    direction_arr = direction_arr / np.linalg.norm(direction_arr)
    s = np.linspace(0.0, distance, n)

    fig, ax = plt.subplots(figsize=(7, 4))
    for z_d in depths:
        pts = np.column_stack(
            [start[0] + s * direction_arr[0],
             start[1] + s * direction_arr[1],
             np.full_like(s, z_d)]
        )
        phi = result.potential(pts, frequency_index=frequency_index).real
        ax.plot(s, phi, label=f"z = {z_d:g} m")

    ax.set_xlabel("distance s along profile in m")
    ax.set_ylabel("Potential φ in V")
    ax.set_title(
        f"Potential profile from {start}, "
        f"f = {result.frequencies[frequency_index]} Hz"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def world_bounds_xy(world: "World") -> tuple[float, float, float, float]:
    """Compute the horizontal bounding box of a world's electrodes.

    Inspects every electrode in ``world.electrodes`` and returns
    the smallest axis-aligned rectangle in the $(x, y)$ plane that
    contains the geometry. Used by :func:`plot_surface_potential` as
    the natural "the whole world" extent for a surface contour plot.

    The footprint of each electrode kind is treated explicitly:

    * :class:`RodElectrode` — point at ``position[:2]``.
    * :class:`RingElectrode` — square of side $2r$ around the centre.
    * :class:`StripElectrode` — bounding box of ``start`` and ``end``.
    * :class:`MeshElectrode` / :class:`GridMeshElectrode` —
      ``corner`` to ``corner + size``.

    Returns
    -------
    tuple
        ``(x_min, x_max, y_min, y_max)`` in metres. For a world with
        no electrodes the result is the trivial ``(0, 0, 0, 0)``;
        callers should add positive padding before plotting.
    """
    from groundfield.geometry.electrodes import (
        GridMeshElectrode,
        MeshElectrode,
        RingElectrode,
        RodElectrode,
        StripElectrode,
    )

    if not world.electrodes:
        return (0.0, 0.0, 0.0, 0.0)

    xs: list[float] = []
    ys: list[float] = []
    for e in world.electrodes:
        if isinstance(e, RodElectrode):
            xs.append(e.position[0])
            ys.append(e.position[1])
        elif isinstance(e, RingElectrode):
            cx, cy, _ = e.center
            xs.extend([cx - e.radius, cx + e.radius])
            ys.extend([cy - e.radius, cy + e.radius])
        elif isinstance(e, StripElectrode):
            xs.extend([e.start[0], e.end[0]])
            ys.extend([e.start[1], e.end[1]])
        elif isinstance(e, (MeshElectrode, GridMeshElectrode)):
            cx, cy, _ = e.corner
            dx, dy = e.size
            xs.extend([cx, cx + dx])
            ys.extend([cy, cy + dy])
        else:  # pragma: no cover — defensive against future kinds
            cp = e.connection_point
            xs.append(cp[0])
            ys.append(cp[1])
    return (min(xs), max(xs), min(ys), max(ys))


def plot_surface_potential(
    result: "FieldResult",
    world: "World",
    *,
    z: float = 0.0,
    padding_m: float = 15.0,
    n: int = 200,
    extent: tuple[float, float, float, float] | None = None,
    frequency_index: int = 0,
    levels: int = 25,
    log: bool = False,
    symmetric: bool = False,
    cmap: str = "viridis",
    show_electrodes: bool = True,
    show_contour_lines: bool = True,
    figsize: tuple[float, float] = (8.5, 7.0),
    title: str | None = None,
):
    """Surface-potential pseudo-colour plot over the entire world.

    Differs from :func:`plot_potential_contour` in two ways:

    1. The default ``extent`` is derived from the *world's
       electrode bounding box* (via :func:`world_bounds_xy`) plus
       ``padding_m``, not from the discretised current point
       sources. This makes the plot naturally cover all
       buildings, cable cabinets and the substation in a TN
       network — including the "boundary regions" where the
       potential decays back to remote earth.
    2. The plot is locked to a horizontal slice (always
       ``plane="xy"``); this is what AP1 calls the *surface
       potential*. Use :func:`plot_potential_contour` for vertical
       slices.

    Mathematical content
    --------------------
    For each grid point $(x_i, y_j, z)$ the function evaluates

    .. math::

        \\varphi(x_i, y_j, z) \\;=\\; \\frac{1}{4\\pi\\sigma_0}
        \\sum_k \\frac{I_k}{|\\mathbf{r}_{ij} - \\mathbf{r}_k|}
        \\;+\\; \\text{image-charge series}

    via :meth:`FieldResult.potential`, which dispatches to the
    correct Green's function for the soil model the result was
    computed with (homogeneous / 2-layer / multi-layer). The
    real part is plotted; for ``log=True`` the colour scale uses
    $\\log_{10} |\\varphi|$ so the boundary decay is visible across
    several decades.

    Parameters
    ----------
    result
        Solver output from :meth:`Engine.solve`.
    world
        Companion world used both to derive the plot extent and
        (optionally) to overlay electrodes.
    z
        Depth of the slice in metres. ``0.0`` is the ground
        surface; positive values go into the soil.
    padding_m
        Extra space added on each side of the electrode bounding
        box, in metres. Larger values let you see how the
        potential decays towards remote earth.
    n
        Resolution per axis (the grid is ``n × n``).
    extent
        Optional explicit ``(x_min, x_max, y_min, y_max)``
        override. ``None`` (default) uses
        ``world_bounds_xy(world)`` plus ``padding_m``.
    frequency_index
        Index into :attr:`FieldResult.frequencies`.
    levels
        Number of contour fill levels.
    log
        Logarithmic colour scale based on ``|φ|`` (better for the
        boundary decay).
    symmetric
        Use a symmetric ``[-φ_max, +φ_max]`` colour scale around 0.
        Useful when the potential takes both signs (multi-source
        or net-injection-zero studies).
    cmap
        Matplotlib colormap name.
    show_electrodes
        Draw the electrode geometry on top of the contour.
    show_contour_lines
        Draw black iso-potential lines on top of the fill.
    figsize
        Matplotlib figure size in inches.
    title
        Optional title override; if ``None`` a sensible default is
        constructed from ``z`` and the chosen frequency.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    if extent is None:
        x_min, x_max, y_min, y_max = world_bounds_xy(world)
        extent = (
            x_min - padding_m, x_max + padding_m,
            y_min - padding_m, y_max + padding_m,
        )

    A, B, flat = _make_grid("xy", extent, z, n)
    phi = result.potential(flat, frequency_index=frequency_index).real
    phi = phi.reshape(A.shape)

    fig, ax = plt.subplots(figsize=figsize)

    if log:
        phi_abs = np.abs(phi)
        # Avoid log(0) by clipping at the smallest positive value;
        # mask-zero so the contourf renders the under-range as bg.
        positive = phi_abs[phi_abs > 0]
        v_min = float(positive.min()) if positive.size else 1e-9
        v_max = float(phi_abs.max()) if phi_abs.size else 1.0
        norm = LogNorm(vmin=max(v_min, 1e-9), vmax=max(v_max, 10 * v_min))
        cs = ax.contourf(A, B, phi_abs, levels=levels, cmap=cmap, norm=norm)
        cbar_label = "|φ| in V (log scale)"
    elif symmetric:
        v = float(np.max(np.abs(phi)))
        norm = Normalize(vmin=-v, vmax=v)
        cs = ax.contourf(A, B, phi, levels=levels, cmap=cmap, norm=norm)
        cbar_label = "Potential φ in V"
    else:
        cs = ax.contourf(A, B, phi, levels=levels, cmap=cmap)
        cbar_label = "Potential φ in V"

    if show_contour_lines and not log:
        contours = ax.contour(A, B, phi, levels=levels, colors="k",
                              linewidths=0.4, alpha=0.4)
        ax.clabel(contours, inline=True, fontsize=7, fmt="%.0f")
    elif show_contour_lines and log:
        # Log mode: contour lines on the absolute value, fewer levels
        # so labels stay legible.
        n_lines = max(5, levels // 3)
        contours = ax.contour(A, B, np.abs(phi), levels=n_lines,
                              colors="k", linewidths=0.4, alpha=0.4)

    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel("x in m")
    ax.set_ylabel("y in m")
    ax.set_aspect("equal")
    if title is None:
        f_hz = result.frequencies[frequency_index]
        title = (
            f"Surface potential φ(x, y, z = {z:g} m), "
            f"f = {f_hz:g} Hz"
        )
    ax.set_title(title)

    if show_electrodes:
        _draw_electrodes(ax, world, "xy")

    fig.tight_layout()
    return fig


def plot_potential_radial(
    result: "FieldResult",
    *,
    around: str | tuple[float, float, float],
    world: "World | None" = None,
    r_max: float = 30.0,
    n: int = 200,
    depths: Iterable[float] = (0.0, 0.5, 1.0),
    frequency_index: int = 0,
    log_x: bool = False,
):
    """Trumpet-shape decay around an electrode (or fixed point).

    Evaluates $\\varphi(r)$ along the ``+x`` direction starting
    at the connection point of the given electrode (or at a fixed
    point). Multiple ``depths`` produce multiple curves — the typical
    "surface vs. deeper soil" comparison.

    Parameters
    ----------
    around
        Electrode name (resolved via ``world``) or ``(x, y, z)`` tuple.
    world
        Required when ``around`` is an electrode name.
    r_max
        Maximum radius in metres.
    n
        Number of evaluation points.
    depths
        $z$ depths for the comparison curves.
    frequency_index
        Index into :attr:`FieldResult.frequencies`.
    log_x
        If ``True`` use a logarithmic x-axis (better visibility of the
        trumpet shape).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if isinstance(around, str):
        if world is None:
            raise ValueError(
                "When 'around' is an electrode name, 'world' is required."
            )
        cp = world.get_electrode(around).connection_point
        x0, y0 = cp[0], cp[1]
        label_origin = around
    else:
        x0, y0 = around[0], around[1]
        label_origin = f"{around}"

    rs = np.linspace(0.1 if not log_x else 0.05, r_max, n)
    fig, ax = plt.subplots(figsize=(7, 4))
    for z_d in depths:
        pts = np.column_stack(
            [x0 + rs, np.full_like(rs, y0), np.full_like(rs, z_d)]
        )
        phi = result.potential(pts, frequency_index=frequency_index).real
        ax.plot(rs, phi, label=f"z = {z_d:g} m")

    ax.set_xlabel(f"distance r from {label_origin} in m")
    ax.set_ylabel("Potential φ in V")
    ax.set_title(
        f"Radial profile around {label_origin}, "
        f"f = {result.frequencies[frequency_index]} Hz"
    )
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
