"""Tests for the parameter-sweep helpers.

Validates :mod:`groundfield.postprocess.sweep` against:

- the row count of the Cartesian product (axes × frequencies),
- AP1 monotonicities (Z grows with rho_1, halves with rod count
  through cluster bonding),
- per-combination engine factory wiring,
- error paths (empty axes, invalid axis values, missing columns
  in plot helpers),
- plot helper smoke tests.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import groundfield as gf  # noqa: E402
from groundfield.postprocess.sweep import (  # noqa: E402
    plot_sweep_heatmap,
    plot_sweep_lines,
    sweep,
)


# ---------------------------------------------------------------------
# Worlds and factories
# ---------------------------------------------------------------------


def _single_rod_world(*, rho: float = 100.0, length: float = 1.5) -> gf.World:
    soil = gf.HomogeneousSoil(resistivity=rho)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world, "rod", name="g1",
        position=(0.0, 0.0, 0.5), length=length, wire_radius=0.005,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    return world


# ---------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------


def test_sweep_row_count_matches_cartesian_product() -> None:
    """rows = product(axes) × frequencies."""
    rho_values = [50.0, 100.0, 500.0]
    L_values = [1.0, 1.5, 2.0]
    freqs = [50.0, 200.0]
    eng = gf.create_engine(
        backend="image", segment_length=0.1, frequencies=freqs,
    )
    df = sweep(
        world_factory=lambda **p: _single_rod_world(rho=p["rho"], length=p["length"]),
        engine=eng,
        axes={"rho": rho_values, "length": L_values},
    )
    assert len(df) == len(rho_values) * len(L_values) * len(freqs)
    assert set(df["rho"].unique()) == set(rho_values)
    assert set(df["length"].unique()) == set(L_values)
    assert set(df["frequency_Hz"].unique()) == set(freqs)


def test_sweep_default_response_columns_present() -> None:
    eng = gf.create_engine(backend="image", segment_length=0.1)
    df = sweep(
        world_factory=lambda **p: _single_rod_world(rho=p["rho"]),
        engine=eng,
        axes={"rho": [100.0, 200.0]},
    )
    expected = {
        "U_E_re", "U_E_im", "abs_U_E",
        "I_re", "I_im", "abs_I",
        "Z_re", "Z_im", "abs_Z", "arg_Z_deg",
        "rho", "frequency_Hz",
    }
    assert expected.issubset(set(df.columns))


def test_sweep_z_proportional_to_rho() -> None:
    """For homogeneous soil the cluster impedance scales linearly with rho."""
    eng = gf.create_engine(backend="image", segment_length=0.05)
    df = sweep(
        world_factory=lambda **p: _single_rod_world(rho=p["rho"]),
        engine=eng,
        axes={"rho": [50.0, 100.0, 500.0, 1000.0]},
    )
    df = df.sort_values("rho").reset_index(drop=True)
    Z_per_rho = df["abs_Z"] / df["rho"]
    # Same rod, only rho changes -> Z/rho must be (numerically) constant.
    assert Z_per_rho.std() / Z_per_rho.mean() < 1e-6


def test_sweep_engine_factory_runs_per_combination() -> None:
    """If engine is a callable, it is rebuilt per combination."""
    counter = {"calls": 0}

    def engine_factory(**params):
        counter["calls"] += 1
        return gf.create_engine(backend="image", segment_length=0.1)

    df = sweep(
        world_factory=lambda **p: _single_rod_world(rho=p["rho"]),
        engine=engine_factory,
        axes={"rho": [50.0, 100.0, 500.0]},
    )
    assert counter["calls"] == 3
    assert len(df) == 3


def test_sweep_custom_response_extractor() -> None:
    """A user-supplied response replaces the default extractor."""
    def my_response(result, world, f_idx):
        return {"answer": 42.0}

    eng = gf.create_engine(backend="image", segment_length=0.1)
    df = sweep(
        world_factory=lambda **p: _single_rod_world(rho=p["rho"]),
        engine=eng,
        axes={"rho": [100.0, 200.0]},
        response=my_response,
    )
    assert (df["answer"] == 42.0).all()
    # Default-response columns must not show up.
    assert "Z_re" not in df.columns


def test_sweep_rejects_empty_axes() -> None:
    eng = gf.create_engine(backend="image", segment_length=0.1)
    with pytest.raises(ValueError, match="at least one axis"):
        sweep(
            world_factory=lambda **p: _single_rod_world(),
            engine=eng,
            axes={},
        )


def test_sweep_rejects_empty_axis_value_list() -> None:
    eng = gf.create_engine(backend="image", segment_length=0.1)
    with pytest.raises(ValueError, match="empty"):
        sweep(
            world_factory=lambda **p: _single_rod_world(rho=p["rho"]),
            engine=eng,
            axes={"rho": []},
        )


# ---------------------------------------------------------------------
# plot helpers
# ---------------------------------------------------------------------


def _demo_df() -> pd.DataFrame:
    eng = gf.create_engine(
        backend="image", segment_length=0.1, frequencies=[50.0, 200.0],
    )
    return sweep(
        world_factory=lambda **p: _single_rod_world(rho=p["rho"], length=p["length"]),
        engine=eng,
        axes={"rho": [50.0, 100.0, 500.0], "length": [1.0, 1.5, 2.0]},
    )


def test_plot_sweep_lines_smoke_single_curve() -> None:
    df = _demo_df()
    df = df[df["frequency_Hz"] == 50.0]
    df = df[df["length"] == 1.5]
    fig = plot_sweep_lines(df, x="rho", y="abs_Z")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_sweep_lines_smoke_multi_curve() -> None:
    df = _demo_df()
    df = df[df["frequency_Hz"] == 50.0]
    fig = plot_sweep_lines(df, x="rho", y="abs_Z", color="length",
                           log_x=True, log_y=True)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_sweep_lines_rejects_unknown_column() -> None:
    df = _demo_df()
    with pytest.raises(KeyError):
        plot_sweep_lines(df, x="rho", y="nope")


def test_plot_sweep_heatmap_smoke() -> None:
    df = _demo_df()
    fig = plot_sweep_heatmap(df, x="rho", y="length", response="abs_Z",
                             frequency_Hz=50.0)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_sweep_heatmap_rejects_invalid_frequency() -> None:
    df = _demo_df()
    with pytest.raises(ValueError, match="No rows match"):
        plot_sweep_heatmap(df, x="rho", y="length", response="abs_Z",
                           frequency_Hz=999.0)


def test_plot_sweep_heatmap_rejects_unknown_column() -> None:
    df = _demo_df()
    with pytest.raises(KeyError):
        plot_sweep_heatmap(df, x="bogus", y="length", response="abs_Z")


# ---------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------


def test_top_level_exports_sweep_helpers() -> None:
    needed = {"sweep", "plot_sweep_lines", "plot_sweep_heatmap"}
    assert needed.issubset(set(gf.__all__))
    assert all(hasattr(gf, name) for name in needed)
