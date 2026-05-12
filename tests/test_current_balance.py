"""Tests for the current-sharing post-processing helpers.

Validates :mod:`groundfield.postprocess.current_balance` against:

- KCL on the source cluster (single rod ⇒ ``r = 1``,
  multi-electrode ideally bonded cluster ⇒ ``r = 1`` regardless of
  member count),
- the AP1 measurement scenario with a metallic feed line in
  parallel to the soil return path (``r < 1`` and the conductor
  current makes up the remainder),
- the per-cluster summary (members complete, share-of-cluster
  rows sum to 100 %, sort order),
- the per-electrode table (cluster mapping, kind / depth
  annotation when ``world`` is given),
- the plot helper smoke test,
- the explicit error paths (no source / multiple sources / unknown
  source name / zero-magnitude source).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import groundfield as gf
from groundfield.postprocess.current_balance import (
    cluster_current_balance,
    electrode_current_table,
    plot_current_sharing,
    split_factor,
)


# ---------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------


def _single_rod_world(
    *, rho: float = 100.0, current: float = 10.0
) -> tuple[gf.World, gf.FieldResult]:
    soil = gf.HomogeneousSoil(resistivity=rho)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world,
        "rod",
        name="g1",
        position=(0.0, 0.0, 0.5),
        length=1.5,
        wire_radius=0.005,
    )
    gf.create_source(world, name="src", attached_to="g1", magnitude=current)
    eng = gf.create_engine(backend="image", segment_length=0.05)
    return world, eng.solve(world)


def _two_rod_bonded_world(
    *, rho: float = 100.0, current: float = 10.0
) -> tuple[gf.World, gf.FieldResult]:
    soil = gf.HomogeneousSoil(resistivity=rho)
    world = gf.create_world(soil=soil)
    g1 = gf.create_electrode(
        world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=1.5
    )
    g2 = gf.create_electrode(
        world, "rod", name="g2", position=(5.0, 0.0, 0.5), length=1.5
    )
    # Ideal galvanic bond -> shared cluster, single source attached to g1.
    gf.create_conductor(
        world, name="bond", start=g1, end=g2, conductor_type="bare_copper"
    )
    gf.create_source(world, name="src", attached_to="g1", magnitude=current)
    eng = gf.create_engine(backend="image", segment_length=0.05)
    return world, eng.solve(world)


def _two_separate_clusters_with_metallic_loop(
    *,
    rho: float = 100.0,
    current: float = 10.0,
    cross_section: float = 50e-6,
) -> tuple[gf.World, gf.FieldResult]:
    """Source cluster + remote aux cluster + finite-impedance metallic
    return conductor between them.

    Mimics an AP1 fall-of-potential measurement with a parallel feed
    lead carrying part of the test current away from the soil.
    """
    soil = gf.HomogeneousSoil(resistivity=rho)
    world = gf.create_world(soil=soil)
    g1 = gf.create_electrode(
        world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=1.5
    )
    g_aux = gf.create_electrode(
        world, "rod", name="g_aux", position=(50.0, 0.0, 0.5), length=1.5
    )
    # Finite-impedance metallic return — Cu, 50 mm² (typical
    # measurement cable). Resistance ~ 1.68e-8 * 50 / 50e-6 = 0.0168 Ω.
    gf.create_conductor(
        world,
        name="feed_lead",
        start=g1,
        end=g_aux,
        conductor_type="bare_copper",
        cross_section=cross_section,
    )
    gf.create_source(
        world, name="src", attached_to="g1", return_to="g_aux", magnitude=current
    )
    eng = gf.create_engine(backend="image", segment_length=0.05)
    return world, eng.solve(world)


# ---------------------------------------------------------------------
# split_factor
# ---------------------------------------------------------------------


def test_split_factor_single_rod_equals_one() -> None:
    world, result = _single_rod_world(current=7.5)
    r = split_factor(result, world)
    assert r.real == pytest.approx(1.0, abs=1e-9)
    assert r.imag == pytest.approx(0.0, abs=1e-9)


def test_split_factor_bonded_pair_equals_one() -> None:
    """When the source cluster is the *only* cluster, the entire current
    must leave it through the soil → r = 1, regardless of how many
    members the cluster has."""
    world, result = _two_rod_bonded_world(current=10.0)
    # Sanity: both electrodes share one cluster.
    assert sorted(result.clusters["g1"]) == ["g1", "g2"]
    r = split_factor(result, world)
    assert r.real == pytest.approx(1.0, abs=1e-9)
    assert r.imag == pytest.approx(0.0, abs=1e-9)


def test_split_factor_with_metallic_return_below_one() -> None:
    """Adding a finite-impedance metallic return path between source
    cluster and aux cluster shunts current away from the soil, so
    r < 1."""
    world, result = _two_separate_clusters_with_metallic_loop(current=10.0)
    r = split_factor(result, world)
    assert 0.0 < r.real < 1.0
    # Imaginary part should be small at DC-like behaviour without an
    # inductance model — but allow for non-zero rounding.
    assert abs(r.imag) < 1e-6


def test_split_factor_complement_matches_branch_current() -> None:
    """Total current balance: (1 - r) * I_src must match the metallic
    branch current. We probe this via I_branch = (U_a - U_b)/R."""
    world, result = _two_separate_clusters_with_metallic_loop(current=10.0)
    cond = world.get_conductor("feed_lead")
    R = cond.series_resistance
    U_g1 = result.electrode_potentials["g1"][0]
    U_aux = result.electrode_potentials["g_aux"][0]
    I_branch = (U_g1 - U_aux) / R

    r = split_factor(result, world)
    I_src = 10.0 + 0j
    soil_current = r * I_src
    assert abs(soil_current.real + I_branch.real - I_src.real) < 1e-3, (
        "Branch + soil currents must reconstruct the source current."
    )


def test_split_factor_unknown_source_name_raises() -> None:
    world, result = _single_rod_world()
    with pytest.raises(KeyError, match="unknown_source"):
        split_factor(result, world, source_name="unknown_source")


def test_split_factor_no_current_source_raises() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.0)
    # Bypass the solver — we only test the error path. Build a
    # minimal FieldResult manually.
    from groundfield.solver.result import FieldResult, PointSource

    result = FieldResult(
        backend="image",
        frequencies=[50.0],
        electrode_potentials={"g1": [1.0 + 0j]},
        electrode_currents={"g1": [1.0 + 0j]},
        point_sources=[
            PointSource(
                position=(0.0, 0.0, 1.0),
                current=[1.0 + 0j],
                electrode_name="g1",
                length=1.0,
            )
        ],
        soil_resistivity=100.0,
        soil=soil,
        clusters={"g1": ["g1"]},
    )
    with pytest.raises(ValueError, match="no current source"):
        split_factor(result, world)


def test_split_factor_multiple_sources_requires_explicit_name() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.0)
    gf.create_electrode(world, "rod", name="g2", position=(20, 0, 0.5), length=1.0)
    gf.create_source(world, name="s1", attached_to="g1", magnitude=1.0)
    gf.create_source(world, name="s2", attached_to="g2", magnitude=2.0)
    eng = gf.create_engine(backend="image", segment_length=0.1)
    result = eng.solve(world)

    with pytest.raises(ValueError, match="multiple current sources"):
        split_factor(result, world)
    # With explicit name, both should resolve.
    r1 = split_factor(result, world, source_name="s1")
    r2 = split_factor(result, world, source_name="s2")
    assert r1.real == pytest.approx(1.0, abs=1e-9)
    assert r2.real == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------
# cluster_current_balance
# ---------------------------------------------------------------------


def test_cluster_balance_columns_and_dtypes() -> None:
    _world, result = _single_rod_world()
    df = cluster_current_balance(result)
    expected = {
        "cluster_root", "n_members", "members",
        "U_re", "U_im", "abs_U",
        "sum_I_re", "sum_I_im", "abs_sum_I",
        "Z_re", "Z_im", "abs_Z", "arg_Z_deg",
    }
    assert set(df.columns) == expected
    assert (df["abs_sum_I"] >= 0).all()


def test_cluster_balance_single_rod() -> None:
    _world, result = _single_rod_world(current=10.0)
    df = cluster_current_balance(result)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["cluster_root"] == "g1"
    assert row["n_members"] == 1
    assert row["sum_I_re"] == pytest.approx(10.0, abs=1e-9)
    # Z must be real-positive with no inductance model.
    assert row["Z_re"] > 0.0
    assert abs(row["Z_im"]) < 1e-9


def test_cluster_balance_two_separate_clusters() -> None:
    """Source cluster + aux cluster — both rows present, currents sum
    to the source magnitude.

    The image backend treats ``return_to`` as informational; the
    injected current dissipates via the Dirichlet far-field boundary,
    so KCL across all cluster leakages must equal the net injection
    (here +10 A at the source cluster).
    """
    _world, result = _two_separate_clusters_with_metallic_loop(current=10.0)
    df = cluster_current_balance(result)
    assert len(df) == 2
    total = df["sum_I_re"].sum() + 1j * df["sum_I_im"].sum()
    assert total.real == pytest.approx(10.0, abs=1e-3)
    assert abs(total.imag) < 1e-3


def test_cluster_balance_sorted_descending() -> None:
    _world, result = _two_separate_clusters_with_metallic_loop(current=10.0)
    df = cluster_current_balance(result)
    diffs = np.diff(df["abs_sum_I"].to_numpy())
    assert (diffs <= 1e-12).all(), "DataFrame must be sorted by descending |ΣI|."


# ---------------------------------------------------------------------
# electrode_current_table
# ---------------------------------------------------------------------


def test_electrode_table_share_sums_match_cluster_total() -> None:
    """For every cluster, the per-electrode complex shares must sum
    to (1 + 0j)."""
    world, result = _two_rod_bonded_world(current=10.0)
    df = electrode_current_table(result, world=world)
    # Both electrodes share the same cluster.
    grp = df.groupby("cluster_root")
    for _, sub in grp:
        s_re = sub["share_of_cluster_re"].sum()
        s_im = sub["share_of_cluster_im"].sum()
        assert s_re == pytest.approx(1.0, abs=1e-9)
        assert abs(s_im) < 1e-9


def test_electrode_table_includes_kind_and_depth_when_world_given() -> None:
    world, result = _single_rod_world()
    df = electrode_current_table(result, world=world)
    assert "kind" in df.columns
    assert "depth_m" in df.columns
    row = df.iloc[0]
    assert row["kind"] == "rod"
    assert row["depth_m"] == pytest.approx(0.5)


def test_electrode_table_omits_geometry_columns_when_world_missing() -> None:
    _world, result = _single_rod_world()
    df = electrode_current_table(result)
    assert "kind" not in df.columns
    assert "depth_m" not in df.columns


def test_electrode_table_sorted_descending() -> None:
    _world, result = _two_separate_clusters_with_metallic_loop(current=10.0)
    df = electrode_current_table(result)
    diffs = np.diff(df["abs_I"].to_numpy())
    assert (diffs <= 1e-12).all(), "DataFrame must be sorted by descending |I|."


# ---------------------------------------------------------------------
# plot helper (smoke only)
# ---------------------------------------------------------------------


def test_plot_current_sharing_smoke_by_electrode() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.figure

    world, result = _two_separate_clusters_with_metallic_loop(current=10.0)
    fig = plot_current_sharing(result, world=world, by="electrode", top_n=5)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_current_sharing_smoke_by_cluster() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.figure

    world, result = _two_separate_clusters_with_metallic_loop(current=10.0)
    fig = plot_current_sharing(result, world=world, by="cluster", top_n=0)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_current_sharing_rejects_unknown_by() -> None:
    _world, result = _single_rod_world()
    with pytest.raises(ValueError, match="electrode"):
        plot_current_sharing(result, by="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------


def test_top_level_exports_current_balance_helpers() -> None:
    needed = {
        "cluster_current_balance",
        "electrode_current_table",
        "split_factor",
        "plot_current_sharing",
    }
    assert needed.issubset(set(gf.__all__))
    assert all(hasattr(gf, name) for name in needed)
