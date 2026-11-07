"""Matplotlib views of a layout (plan view and surface potential).

Part of :mod:`groundfield.generators.layout`. Drawing helpers only; no
solver work happens here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from groundfield.world import World

if TYPE_CHECKING:  # pragma: no cover - typing only
    import matplotlib.axes as _mpl_axes

__all__: list[str] = []


class _PlotMixin:
    """Plot methods of :class:`~groundfield.generators.layout.LvNetworkLayout`."""

    def plot(
        self,
        *,
        ax: Optional["_mpl_axes.Axes"] = None,
        show_unconnected: bool = True,
        annotate_houses: bool = False,
        foundation_mask: Optional[list[bool]] = None,
        figsize: tuple[float, float] = (12.0, 8.0),
    ) -> "_mpl_axes.Axes":
        """Render the layout with matplotlib.

        Parameters
        ----------
        foundation_mask
            Optional per-house bool list (one entry per footprint).
            When supplied, houses with ``True`` get a saturated
            fill colour (foundation electrode present) and the
            remaining houses get a faded fill (no foundation).
            Combine with :meth:`foundation_mask` to visualise
            different penetration scenarios.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the layout was drawn on. The figure can be
            retrieved via ``ax.figure``.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        connected_idx = {c.house_idx for c in self.connections}

        if foundation_mask is not None and len(foundation_mask) != len(
            self.footprints,
        ):
            raise ValueError(
                "LvNetworkLayout.plot: foundation_mask length "
                f"{len(foundation_mask)} does not match footprints "
                f"count {len(self.footprints)}."
            )

        # Building footprints.
        for i, fp in enumerate(self.footprints):
            xs = [p[0] for p in fp.polygon_xy_m] + [fp.polygon_xy_m[0][0]]
            ys = [p[1] for p in fp.polygon_xy_m] + [fp.polygon_xy_m[0][1]]
            if foundation_mask is not None:
                # With-foundation: saturated green-grey; without: faded.
                if foundation_mask[i]:
                    colour = "#7fbf7f"  # has a foundation electrode
                    edge = "#2a6f2a"
                else:
                    colour = "#f4f4f4"  # no foundation electrode
                    edge = "0.7"
            else:
                colour = "0.92" if i in connected_idx else "#fde0dc"
                edge = "0.4"
            ax.fill(xs, ys, color=colour, edgecolor=edge, linewidth=0.6, zorder=1)
            if annotate_houses:
                cx, cy = fp.centroid_xy_m()
                ax.text(
                    cx,
                    cy,
                    str(i),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="0.3",
                    zorder=2,
                )

        # PEN cables.
        cable_colours = {}
        palette = plt.get_cmap("tab10").colors
        for k, cable in enumerate(self.pen_cables):
            colour = palette[k % len(palette)]
            cable_colours[cable.name] = colour
            xs = [p[0] for p in cable.waypoints]
            ys = [p[1] for p in cable.waypoints]
            ax.plot(xs, ys, color=colour, lw=2.2, zorder=4, label=f"PEN {cable.name}")
            # Mark every corner.
            for x, y in cable.waypoints:
                ax.plot(x, y, marker="o", color=colour, markersize=3, zorder=5)

        # Service drops.
        drew_stub_label = False
        for c in self.connections:
            colour = cable_colours.get(c.cable_name, "saddlebrown")
            xs = [p[0] for p in c.stub_waypoints]
            ys = [p[1] for p in c.stub_waypoints]
            ax.plot(
                xs,
                ys,
                color=colour,
                lw=1.0,
                ls="--",
                zorder=3,
                label=("PEN service drop" if not drew_stub_label else None),
            )
            drew_stub_label = True
            ax.plot(
                c.tap_point_xy[0],
                c.tap_point_xy[1],
                marker="x",
                color=colour,
                markersize=6,
                zorder=6,
            )

        # Substation marker.
        ax.plot(
            *self.substation_xy,
            marker="D",
            color="tab:orange",
            markersize=14,
            zorder=7,
            label="Substation",
        )

        # Cable-cabinet markers.
        for k in self.cabinet_placements:
            ax.plot(
                *k.position_xy, marker="s", color="tab:blue", markersize=11, zorder=7
            )
        if self.cabinet_placements:
            ax.plot(
                [],
                [],
                marker="s",
                color="tab:blue",
                markersize=11,
                linestyle="None",
                label="Cable cabinet",
            )

        # Auxiliary current electrode.
        if self.auxiliary_electrode is not None:
            aux = self.auxiliary_electrode
            # Dotted return-current line from substation to aux.
            ax.plot(
                [self.substation_xy[0], aux.position_xy[0]],
                [self.substation_xy[1], aux.position_xy[1]],
                color="purple",
                lw=0.8,
                ls=":",
                zorder=2,
                label="Measurement feed (return)",
            )
            ax.plot(
                *aux.position_xy,
                marker="v",
                color="purple",
                markersize=13,
                zorder=7,
                label=f"Auxiliary electrode {aux.name!s}",
            )

        # Voltage probe.
        if self.voltage_probe is not None:
            probe = self.voltage_probe
            ax.plot(
                *probe.position_xy,
                marker="*",
                color="lime",
                markersize=18,
                zorder=8,
                markeredgecolor="darkgreen",
                markeredgewidth=1.2,
                label=f"Voltage probe {probe.name!s}",
            )
        if (
            show_unconnected
            and foundation_mask is None
            and any(i not in connected_idx for i in range(len(self.footprints)))
        ):
            ax.plot(
                [],
                [],
                marker="s",
                color="#fde0dc",
                markersize=11,
                linestyle="None",
                label="Unconnected house",
            )
        if foundation_mask is not None:
            ax.plot(
                [],
                [],
                marker="s",
                color="#7fbf7f",
                markersize=11,
                linestyle="None",
                label="House with foundation electrode",
            )
            ax.plot(
                [],
                [],
                marker="s",
                color="#f4f4f4",
                markeredgecolor="0.7",
                markersize=11,
                linestyle="None",
                label="House without foundation electrode",
            )

        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_title("LV network layout (Manhattan PEN routing)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_surface_potential(
        self,
        result: object,
        world: World,
        *,
        frequency_index: int = 0,
        padding_m: float = 50.0,
        n: int = 150,
        levels: int = 41,
        log: bool = False,
        symmetric: bool = False,
        two_slope: bool = True,
        cmap: str = "RdBu_r",
        show_electrodes: bool = True,
        figsize: tuple[float, float] = (11.0, 8.5),
        title: Optional[str] = None,
    ):
        r"""Surface-potential pseudo-colour plot with measurement
        markers (substation, auxiliary electrode, voltage probe).

        Renders $\varphi(x, y, z=0)$ on a regular grid sized to the
        world bounding box plus ``padding_m`` and overlays the
        substation, auxiliary-electrode and voltage-probe markers plus a
        dotted return-current line. Three colour-axis modes are
        supported:

        * ``two_slope=True`` (default) -- uses
          :class:`matplotlib.colors.TwoSlopeNorm` centred at
          $\varphi = 0$ so the colour resolution above and below
          zero is balanced regardless of the actual range. Ideal
          for the fall-of-potential setup, where the auxiliary electrode
          sits at a strongly negative potential (e.g. -10 V) and
          the substation + foundations together rise to only a
          few volts -- on a single linear scale the positive
          trumpet collapses into a single colour. The colourbar
          uses non-uniform contour levels so each half of the
          scale carries the same number of fills.
        * ``symmetric=True`` -- the classic
          :math:`[-|\varphi|_\text{max}, +|\varphi|_\text{max}]`
          range. Equivalent to ``TwoSlopeNorm`` only when the
          extremes are symmetric; otherwise large negative
          potentials clip the positive side. Setting
          ``symmetric=True`` *implicitly* turns off
          ``two_slope``.
        * ``log=True`` -- log-scale on $|\varphi|$. Useful when the
          potential decays across several decades toward remote
          earth.

        Parameters
        ----------
        result
            :class:`FieldResult` from :meth:`Engine.solve`.
        world
            Companion world used to derive the plot extent and
            (when ``show_electrodes=True``) to overlay the
            electrode geometry.
        frequency_index
            Index into :attr:`FieldResult.frequencies`.
        padding_m
            Extra space around the world bounding box in m.
        n
            Grid resolution per axis (``n × n`` evaluation points).
        levels
            Number of contour-fill levels. Split half-and-half
            between the negative and positive ranges when
            ``two_slope=True``.
        log, symmetric, two_slope
            Mutually-influencing scaling modes (see above).
        cmap
            Matplotlib colormap (diverging recommended).
        show_electrodes
            Overlay the electrode geometry (small markers + lines).
        figsize, title
            Figure size and optional title override.

        Returns
        -------
        matplotlib.figure.Figure
            The figure the plot was drawn on. Use ``fig.axes[0]``
            for further annotations.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

        from groundfield.postprocess.plotting import (
            _draw_electrodes,
            _make_grid,
            world_bounds_xy,
        )

        # Mutual exclusion of the scaling modes -- log dominates;
        # symmetric overrides two_slope. Default is two_slope.
        if log:
            two_slope = False
            symmetric = False
        elif symmetric:
            two_slope = False

        x_min, x_max, y_min, y_max = world_bounds_xy(world)
        extent = (
            x_min - padding_m,
            x_max + padding_m,
            y_min - padding_m,
            y_max + padding_m,
        )
        A, B, flat = _make_grid("xy", extent, 0.0, n)
        phi = result.potential(flat, frequency_index=frequency_index).real
        phi = phi.reshape(A.shape)

        fig, ax = plt.subplots(figsize=figsize)

        if log:
            phi_abs = np.abs(phi)
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
        elif two_slope:
            v_neg = float(min(phi.min(), -1e-9))
            v_pos = float(max(phi.max(), +1e-9))
            norm = TwoSlopeNorm(vmin=v_neg, vcenter=0.0, vmax=v_pos)
            # Split contour levels half-and-half so each side of
            # zero carries the same visual resolution.
            n_per_side = max(2, levels // 2)
            neg_levels = np.linspace(v_neg, 0.0, n_per_side + 1)[:-1]
            pos_levels = np.linspace(0.0, v_pos, n_per_side + 1)
            level_array = np.concatenate([neg_levels, pos_levels])
            cs = ax.contourf(
                A, B, phi, levels=level_array, cmap=cmap, norm=norm, extend="both"
            )
            cbar_label = (
                "Potential φ in V (TwoSlopeNorm: equal colour "
                "resolution above / below 0)"
            )
        else:
            cs = ax.contourf(A, B, phi, levels=levels, cmap=cmap)
            cbar_label = "Potential φ in V"

        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label(cbar_label)

        ax.set_xlabel("x in m")
        ax.set_ylabel("y in m")
        ax.set_aspect("equal")
        if title is None:
            f_hz = result.frequencies[frequency_index]
            title = f"Surface potential φ(x, y, z = 0 m), f = {f_hz:g} Hz"
        ax.set_title(title)

        if show_electrodes:
            _draw_electrodes(ax, world, "xy")

        # Substation marker.
        ax.plot(
            *self.substation_xy,
            marker="D",
            color="tab:orange",
            markersize=14,
            zorder=10,
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="Substation",
        )
        # Auxiliary electrode.
        if self.auxiliary_electrode is not None:
            aux = self.auxiliary_electrode
            ax.plot(
                [self.substation_xy[0], aux.position_xy[0]],
                [self.substation_xy[1], aux.position_xy[1]],
                color="purple",
                lw=0.8,
                ls=":",
                zorder=9,
            )
            ax.plot(
                *aux.position_xy,
                marker="v",
                color="purple",
                markersize=13,
                zorder=10,
                markeredgecolor="black",
                markeredgewidth=0.8,
                label="Auxiliary electrode",
            )
        # Voltage probe.
        if self.voltage_probe is not None:
            probe = self.voltage_probe
            ax.plot(
                *probe.position_xy,
                marker="*",
                color="lime",
                markersize=18,
                zorder=11,
                markeredgecolor="darkgreen",
                markeredgewidth=1.2,
                label="Voltage probe",
            )
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        return fig
