"""Tests for the segment-length convergence study.

Validates :mod:`groundfield.postprocess.convergence` against:

- monotone refinement (n_segments grows as ``segment_length`` shrinks),
- convergence of the cluster impedance towards Sunde for a single rod,
- the engine is **cloned** (not mutated) by the helper,
- the ``segment_lengths`` validation error paths,
- the plot helper smoke tests with and without a reference line.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import groundfield as gf  # noqa: E402
from groundfield.postprocess.convergence import (  # noqa: E402
    convergence_study,
    plot_convergence,
)


# ---------------------------------------------------------------------
# Worlds
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
# convergence_study
# ---------------------------------------------------------------------


def test_convergence_study_n_segments_grows_with_refinement() -> None:
    world = _single_rod_world()
    eng = gf.create_engine(backend="image", segment_length=0.5)
    df = convergence_study(world, eng, segment_lengths=[0.5, 0.2, 0.1, 0.05])
    df = df.sort_values("segment_length_m", ascending=False).reset_index(drop=True)
    diffs = np.diff(df["n_segments"].to_numpy())
    # n_segments must monotonically grow (>= since same ds shouldn't repeat).
    assert (diffs >= 0).all()
    # Strict growth between distinct segment lengths.
    assert df["n_segments"].iloc[-1] > df["n_segments"].iloc[0]


def test_convergence_study_approaches_sunde() -> None:
    """For a single rod the cluster impedance must converge towards
    Sunde's analytical formula as segment_length shrinks."""
    rho, L, d = 100.0, 1.5, 0.01  # diameter = 2 * wire_radius (5 mm)
    R_sunde = rho / (2.0 * math.pi * L) * (math.log(4.0 * L / d) - 1.0)

    world = _single_rod_world(rho=rho, length=L)
    eng = gf.create_engine(backend="image", segment_length=0.5)
    df = convergence_study(
        world, eng, segment_lengths=[0.5, 0.2, 0.1, 0.05, 0.02],
    )

    df = df.sort_values("segment_length_m", ascending=False).reset_index(drop=True)
    Zs = df["abs_Z"].to_numpy()
    # The finest run must be within 5 % of Sunde.
    rel_err = abs(Zs[-1] - R_sunde) / R_sunde
    assert rel_err < 0.05, (
        f"Z_finest = {Zs[-1]:.3f} Ω, R_sunde = {R_sunde:.3f} Ω, "
        f"rel error = {rel_err*100:.1f} %"
    )
    # And the error must decrease monotonically with refinement.
    rel_errs = np.abs(Zs - R_sunde) / R_sunde
    assert (np.diff(rel_errs) <= 1e-6).all(), (
        f"convergence is not monotone: {rel_errs}"
    )


def test_convergence_study_does_not_mutate_engine() -> None:
    """Engine.segment_length must be unchanged after the helper runs."""
    world = _single_rod_world()
    eng = gf.create_engine(backend="image", segment_length=0.5)
    original = eng.segment_length
    _ = convergence_study(world, eng, segment_lengths=[0.5, 0.1, 0.05])
    assert eng.segment_length == original


def test_convergence_study_rejects_empty_or_too_few() -> None:
    world = _single_rod_world()
    eng = gf.create_engine(backend="image", segment_length=0.5)
    with pytest.raises(ValueError, match="at least 2"):
        convergence_study(world, eng, segment_lengths=[0.1])
    with pytest.raises(ValueError, match="distinct values"):
        convergence_study(world, eng, segment_lengths=[0.1, 0.1])


def test_convergence_study_rejects_non_positive_segment_lengths() -> None:
    world = _single_rod_world()
    eng = gf.create_engine(backend="image", segment_length=0.5)
    with pytest.raises(ValueError, match="strictly positive"):
        convergence_study(world, eng, segment_lengths=[0.1, 0.0, 0.05])
    with pytest.raises(ValueError, match="strictly positive"):
        convergence_study(world, eng, segment_lengths=[0.1, -0.05, 0.02])


def test_convergence_study_multi_frequency() -> None:
    """rows = len(segment_lengths) * len(frequencies)."""
    world = _single_rod_world()
    eng = gf.create_engine(
        backend="image", segment_length=0.5, frequencies=[50.0, 200.0],
    )
    df = convergence_study(world, eng, segment_lengths=[0.2, 0.1, 0.05])
    assert len(df) == 3 * 2
    assert set(df["frequency_Hz"].unique()) == {50.0, 200.0}


# ---------------------------------------------------------------------
# plot_convergence
# ---------------------------------------------------------------------


def test_plot_convergence_smoke_single_frequency() -> None:
    world = _single_rod_world()
    eng = gf.create_engine(backend="image", segment_length=0.5)
    df = convergence_study(world, eng, segment_lengths=[0.5, 0.2, 0.1])
    fig = plot_convergence(df)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_convergence_smoke_multi_frequency() -> None:
    world = _single_rod_world()
    eng = gf.create_engine(
        backend="image", segment_length=0.5, frequencies=[50.0, 200.0],
    )
    df = convergence_study(world, eng, segment_lengths=[0.5, 0.2, 0.1])
    fig = plot_convergence(df, response="abs_Z")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_convergence_with_reference_line() -> None:
    world = _single_rod_world()
    eng = gf.create_engine(backend="image", segment_length=0.5)
    df = convergence_study(world, eng, segment_lengths=[0.5, 0.2, 0.1])
    fig = plot_convergence(df, response="abs_Z", reference=60.0)
    # Reference shows up as an extra line in the axes.
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_convergence_inverts_x_axis() -> None:
    """The x-axis must be inverted so finer ds lands on the right."""
    world = _single_rod_world()
    eng = gf.create_engine(backend="image", segment_length=0.5)
    df = convergence_study(world, eng, segment_lengths=[0.5, 0.2, 0.1])
    fig = plot_convergence(df)
    ax = fig.axes[0]
    x_lo, x_hi = ax.get_xlim()
    # Inverted -> first value > second value.
    assert x_lo > x_hi
    plt.close(fig)


def test_plot_convergence_rejects_unknown_response() -> None:
    world = _single_rod_world()
    eng = gf.create_engine(backend="image", segment_length=0.5)
    df = convergence_study(world, eng, segment_lengths=[0.2, 0.1])
    with pytest.raises(KeyError):
        plot_convergence(df, response="bogus")


# ---------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------


def test_top_level_exports_convergence_helpers() -> None:
    needed = {"convergence_study", "plot_convergence"}
    assert needed.issubset(set(gf.__all__))
    assert all(hasattr(gf, name) for name in needed)
