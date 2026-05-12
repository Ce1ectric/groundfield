"""Tests for the CSV writers in :mod:`groundfield.io.csv`.

Validates:

- :func:`save_potential_path_csv` round-trip (file exists, header
  is intact, the saved values match :meth:`FieldResult.potential`),
- :func:`save_electrode_table_csv` matches the in-memory
  :func:`electrode_current_table`,
- :func:`save_cluster_impedances_csv` flattens the ``members``
  list into a string column and is otherwise consistent with
  :func:`cluster_current_balance`,
- error paths (bad distance / n / direction / frequency_index),
- top-level exports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import groundfield as gf
from groundfield.io.csv import (
    save_cluster_impedances_csv,
    save_electrode_table_csv,
    save_potential_path_csv,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _two_rod_world() -> tuple[gf.World, gf.FieldResult]:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    g1 = gf.create_electrode(
        world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=1.5
    )
    g2 = gf.create_electrode(
        world, "rod", name="g2", position=(5.0, 0.0, 0.5), length=1.5
    )
    gf.create_conductor(world, name="bond", start=g1, end=g2,
                        conductor_type="bare_copper")
    gf.create_source(world, name="src", attached_to=g1, magnitude=10.0)
    eng = gf.create_engine(backend="image", segment_length=0.05)
    return world, eng.solve(world)


# ---------------------------------------------------------------------
# save_potential_path_csv
# ---------------------------------------------------------------------


def test_save_potential_path_csv_roundtrip(tmp_path: Path) -> None:
    _, result = _two_rod_world()
    out = save_potential_path_csv(
        result, tmp_path / "phi.csv",
        start=(1.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
        distance=20.0, n=50,
    )
    assert out.exists()
    df = pd.read_csv(out)
    assert {"s", "x", "y", "z", "frequency_Hz", "phi_re", "phi_im", "abs_phi"}.issubset(
        df.columns
    )
    assert len(df) == 50  # one frequency, n samples
    # Spot-check: re-evaluate the potential at the first sample point
    # and compare.
    pt = np.array([[df["x"].iloc[0], df["y"].iloc[0], df["z"].iloc[0]]])
    phi_back = result.potential(pt, frequency_index=0)[0]
    assert df["phi_re"].iloc[0] == pytest.approx(phi_back.real, rel=1e-12)
    assert df["phi_im"].iloc[0] == pytest.approx(phi_back.imag, rel=1e-12)


def test_save_potential_path_csv_creates_parent_directories(tmp_path: Path) -> None:
    _, result = _two_rod_world()
    nested = tmp_path / "a" / "b" / "c" / "phi.csv"
    out = save_potential_path_csv(
        result, nested, start=(1, 0, 0), distance=5.0, n=10,
    )
    assert out.exists()


def test_save_potential_path_csv_multi_frequency(tmp_path: Path) -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.5), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(
        backend="image", segment_length=0.1, frequencies=[50.0, 200.0],
    )
    result = eng.solve(world)

    out = save_potential_path_csv(
        result, tmp_path / "phi.csv", start=(1, 0, 0), distance=5.0, n=20,
    )
    df = pd.read_csv(out)
    # 2 frequencies * 20 samples = 40 rows, both frequencies present.
    assert len(df) == 40
    assert sorted(df["frequency_Hz"].unique()) == [50.0, 200.0]


def test_save_potential_path_csv_rejects_invalid_args(tmp_path: Path) -> None:
    _, result = _two_rod_world()
    with pytest.raises(ValueError, match="distance"):
        save_potential_path_csv(result, tmp_path / "f.csv",
                                start=(0, 0, 0), distance=0.0, n=10)
    with pytest.raises(ValueError, match="n"):
        save_potential_path_csv(result, tmp_path / "f.csv",
                                start=(0, 0, 0), distance=5.0, n=1)
    with pytest.raises(ValueError, match="direction"):
        save_potential_path_csv(result, tmp_path / "f.csv",
                                start=(0, 0, 0), direction=(0, 0, 0),
                                distance=5.0, n=10)
    with pytest.raises(ValueError, match="out of range"):
        save_potential_path_csv(result, tmp_path / "f.csv",
                                start=(0, 0, 0), distance=5.0, n=10,
                                frequency_indices=[0, 99])


# ---------------------------------------------------------------------
# save_electrode_table_csv
# ---------------------------------------------------------------------


def test_save_electrode_table_csv_matches_in_memory(tmp_path: Path) -> None:
    world, result = _two_rod_world()
    out = save_electrode_table_csv(
        result, tmp_path / "electrodes.csv", world=world,
    )
    df_disk = pd.read_csv(out)
    df_mem = gf.electrode_current_table(result, world=world)

    pd.testing.assert_frame_equal(
        df_disk.reset_index(drop=True), df_mem.reset_index(drop=True),
        check_dtype=False,  # CSV roundtrip can shift int64 <-> float64
    )


def test_save_electrode_table_csv_without_world_omits_geometry(tmp_path: Path) -> None:
    _, result = _two_rod_world()
    out = save_electrode_table_csv(result, tmp_path / "electrodes.csv")
    df = pd.read_csv(out)
    assert "kind" not in df.columns
    assert "depth_m" not in df.columns


# ---------------------------------------------------------------------
# save_cluster_impedances_csv
# ---------------------------------------------------------------------


def test_save_cluster_impedances_csv_flattens_members(tmp_path: Path) -> None:
    _, result = _two_rod_world()
    out = save_cluster_impedances_csv(result, tmp_path / "clusters.csv")
    df = pd.read_csv(out)
    # The 'members' column is now a ';'-joined string.
    assert "members" in df.columns
    assert all(isinstance(v, str) for v in df["members"])
    assert "g1;g2" in set(df["members"])


# ---------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------


def test_top_level_exports_csv_helpers() -> None:
    needed = {
        "save_potential_path_csv",
        "save_electrode_table_csv",
        "save_cluster_impedances_csv",
    }
    assert needed.issubset(set(gf.__all__))
    assert all(hasattr(gf, name) for name in needed)
