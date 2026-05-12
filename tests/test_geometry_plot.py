"""Tests for the world-geometry plot helpers (no solve required).

Validates :mod:`groundfield.postprocess.geometry_plot`:

- :func:`world_bounds_3d` correctness across electrode kinds and
  conductor endpoints (rod foot, ring extremes, conductor depth),
- :func:`plot_world` smoke tests on both planes (xy / xz), with
  conductors and sources toggled on/off,
- :func:`plot_world_3d` smoke test on a multi-electrode world,
- empty-world edge case (no crash, returns a zero-size box and a
  blank figure),
- top-level export check.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import groundfield as gf  # noqa: E402
from groundfield.postprocess.geometry_plot import (  # noqa: E402
    plot_world,
    plot_world_3d,
    world_bounds_3d,
)


# ---------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------


def _mixed_world() -> gf.World:
    """Small world with all electrode kinds + a few conductors + a source."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil, name="mixed")

    g_rod = gf.create_electrode(
        world, "rod", name="rod_a", position=(0.0, 0.0, 0.5), length=1.5
    )
    g_ring = gf.create_electrode(
        world, "ring", name="ring_a", center=(10.0, 0.0, 0.8), radius=2.5
    )
    g_strip = gf.create_electrode(
        world,
        "strip",
        name="strip_a",
        start=(0.0, 5.0, 0.6),
        end=(8.0, 5.0, 0.6),
    )
    g_mesh = gf.create_electrode(
        world,
        "grid_mesh",
        name="mesh_a",
        corner=(15.0, -3.0, 0.7),
        size=(6.0, 4.0),
        n_x=3,
        n_y=2,
    )
    # Two conductors with different conductor_type and coupling.
    gf.create_conductor(
        world, name="bond", start=g_rod, end=g_ring,
        conductor_type="bare_copper",
    )
    gf.create_conductor(
        world,
        name="pen_overhead",
        start=(-5.0, 0.0, -0.2),       # above ground
        end=(20.0, 0.0, -0.2),
        conductor_type="pen",
        cross_section=50e-6,
        coupling_to_soil="isolated",
    )
    gf.create_source(
        world, name="src", attached_to=g_rod, magnitude=10.0,
        return_to=g_ring,
    )
    return world


def _empty_world() -> gf.World:
    return gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))


# ---------------------------------------------------------------------
# world_bounds_3d
# ---------------------------------------------------------------------


def test_world_bounds_3d_covers_all_electrode_extents() -> None:
    world = _mixed_world()
    x_min, x_max, y_min, y_max, z_min, z_max = world_bounds_3d(world)

    # x range: rod at 0, ring centered at 10 with radius 2.5,
    # strip from 0 to 8, mesh corner 15 + size 6 -> 21,
    # pen conductor from -5 to +20.
    assert x_min == pytest.approx(-5.0)
    assert x_max == pytest.approx(21.0)

    # y range: ring centered at y=0 +/- 2.5, mesh from -3 to 1.
    assert y_min == pytest.approx(-3.0)
    assert y_max == pytest.approx(5.0)

    # z range: rod head at 0.5, foot at 2.0; pen conductor at -0.2.
    assert z_min == pytest.approx(-0.2)
    assert z_max == pytest.approx(2.0)


def test_world_bounds_3d_includes_rod_foot() -> None:
    """The rod's lower end (head + length) must be in the box."""
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=2.0
    )
    bounds = world_bounds_3d(world)
    assert bounds[5] == pytest.approx(2.5)


def test_world_bounds_3d_includes_conductor_endpoints() -> None:
    """Even without electrodes outside, conductor endpoints extend the box."""
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    g1 = gf.create_electrode(
        world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=1.0
    )
    g2 = gf.create_electrode(
        world, "rod", name="g2", position=(2.0, 0.0, 0.5), length=1.0
    )
    gf.create_conductor(
        world, name="lead", start=(-50.0, 0.0, -0.3), end=g1,
        conductor_type="overhead",
    )
    bounds = world_bounds_3d(world)
    assert bounds[0] == pytest.approx(-50.0)
    assert bounds[4] == pytest.approx(-0.3)


def test_world_bounds_3d_empty_world() -> None:
    assert world_bounds_3d(_empty_world()) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------
# plot_world (2-D)
# ---------------------------------------------------------------------


def test_plot_world_xy_returns_figure() -> None:
    fig = plot_world(_mixed_world(), plane="xy")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_world_xz_returns_figure_with_inverted_y() -> None:
    fig = plot_world(_mixed_world(), plane="xz")
    ax = fig.axes[0]
    # Inverted y axis means depth grows downward on screen
    # (top number > bottom number).
    y_low, y_high = ax.get_ylim()
    assert y_low > y_high
    plt.close(fig)


def test_plot_world_omits_conductors_and_sources_when_disabled() -> None:
    world = _mixed_world()
    fig_full = plot_world(world)
    fig_geo = plot_world(world, show_conductors=False, show_sources=False)

    # The "geometry-only" version must have strictly fewer artists.
    n_full = sum(len(a.lines) for a in fig_full.axes)
    n_geo = sum(len(a.lines) for a in fig_geo.axes)
    assert n_geo < n_full
    plt.close(fig_full)
    plt.close(fig_geo)


def test_plot_world_extent_uses_padding_around_bounds() -> None:
    world = _mixed_world()
    bounds = world_bounds_3d(world)
    fig = plot_world(world, plane="xy", padding_m=10.0)
    ax = fig.axes[0]
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    assert x_lo == pytest.approx(bounds[0] - 10.0)
    assert x_hi == pytest.approx(bounds[1] + 10.0)
    assert y_lo == pytest.approx(bounds[2] - 10.0)
    assert y_hi == pytest.approx(bounds[3] + 10.0)
    plt.close(fig)


def test_plot_world_explicit_extent_overrides_bounds() -> None:
    fig = plot_world(_mixed_world(), plane="xy", extent=(-100, 100, -100, 100))
    ax = fig.axes[0]
    assert ax.get_xlim() == (-100.0, 100.0)
    assert ax.get_ylim() == (-100.0, 100.0)
    plt.close(fig)


def test_plot_world_rejects_unknown_plane() -> None:
    with pytest.raises(ValueError, match="plane"):
        plot_world(_mixed_world(), plane="yz")  # type: ignore[arg-type]


def test_plot_world_handles_empty_world() -> None:
    """An empty world still returns a figure (blank canvas) and does not raise."""
    fig = plot_world(_empty_world())
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_world_annotate_electrodes_adds_text() -> None:
    world = _mixed_world()
    fig = plot_world(world, annotate_electrodes=True)
    ax = fig.axes[0]
    # Each electrode contributes one annotation; mixed world has 4.
    n_text = sum(1 for child in ax.get_children()
                 if isinstance(child, matplotlib.text.Annotation))
    assert n_text >= len(world.electrodes)
    plt.close(fig)


def test_plot_world_accepts_external_axes() -> None:
    world = _mixed_world()
    fig, ax = plt.subplots()
    out = plot_world(world, ax=ax)
    assert out is fig
    plt.close(fig)


# ---------------------------------------------------------------------
# plot_world_3d
# ---------------------------------------------------------------------


def test_plot_world_3d_returns_figure() -> None:
    fig = plot_world_3d(_mixed_world())
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_world_3d_inverts_z_axis() -> None:
    """Soil convention: positive z grows downwards. The 3D z-axis must be inverted."""
    fig = plot_world_3d(_mixed_world())
    ax = fig.axes[0]
    # Axes3D.get_zlim returns (low, high) but post-inversion the first value > second.
    z_low, z_high = ax.get_zlim()
    assert z_low > z_high
    plt.close(fig)


def test_plot_world_3d_handles_empty_world() -> None:
    fig = plot_world_3d(_empty_world(), show_surface=True)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_world_3d_with_options_disabled() -> None:
    fig = plot_world_3d(
        _mixed_world(),
        show_conductors=False,
        show_sources=False,
        show_surface=False,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------


def test_top_level_exports_geometry_plot_helpers() -> None:
    needed = {"plot_world", "plot_world_3d", "world_bounds_3d"}
    assert needed.issubset(set(gf.__all__))
    assert all(hasattr(gf, name) for name in needed)
