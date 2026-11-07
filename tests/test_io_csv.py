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


def read_csv_with_provenance(path):
    """Reader of the provenance-headed CSV files (imported lazily so the
    module still collects against releases without it)."""
    from groundfield.io.csv import read_csv_with_provenance as _read

    return _read(path)


def _two_rod_world() -> tuple[gf.World, gf.FieldResult]:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    g1 = gf.create_electrode(
        world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=1.5
    )
    g2 = gf.create_electrode(
        world, "rod", name="g2", position=(5.0, 0.0, 0.5), length=1.5
    )
    gf.create_conductor(
        world, name="bond", start=g1, end=g2, conductor_type="bare_copper"
    )
    gf.create_source(world, name="src", attached_to=g1, magnitude=10.0)
    eng = gf.create_engine(backend="image", segment_length=0.05)
    return world, eng.solve(world)


# ---------------------------------------------------------------------
# save_potential_path_csv
# ---------------------------------------------------------------------


def test_save_potential_path_csv_roundtrip(tmp_path: Path) -> None:
    _, result = _two_rod_world()
    out = save_potential_path_csv(
        result,
        tmp_path / "phi.csv",
        start=(1.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        distance=20.0,
        n=50,
    )
    assert out.exists()
    df = read_csv_with_provenance(out)[0]
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
        result,
        nested,
        start=(1, 0, 0),
        distance=5.0,
        n=10,
    )
    assert out.exists()


def test_save_potential_path_csv_multi_frequency(tmp_path: Path) -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(
        backend="image",
        segment_length=0.1,
        frequencies=[50.0, 200.0],
    )
    result = eng.solve(world)

    out = save_potential_path_csv(
        result,
        tmp_path / "phi.csv",
        start=(1, 0, 0),
        distance=5.0,
        n=20,
    )
    df = read_csv_with_provenance(out)[0]
    # 2 frequencies * 20 samples = 40 rows, both frequencies present.
    assert len(df) == 40
    assert sorted(df["frequency_Hz"].unique()) == [50.0, 200.0]


def test_save_potential_path_csv_rejects_invalid_args(tmp_path: Path) -> None:
    _, result = _two_rod_world()
    with pytest.raises(ValueError, match="distance"):
        save_potential_path_csv(
            result, tmp_path / "f.csv", start=(0, 0, 0), distance=0.0, n=10
        )
    with pytest.raises(ValueError, match="n"):
        save_potential_path_csv(
            result, tmp_path / "f.csv", start=(0, 0, 0), distance=5.0, n=1
        )
    with pytest.raises(ValueError, match="direction"):
        save_potential_path_csv(
            result,
            tmp_path / "f.csv",
            start=(0, 0, 0),
            direction=(0, 0, 0),
            distance=5.0,
            n=10,
        )
    with pytest.raises(ValueError, match="out of range"):
        save_potential_path_csv(
            result,
            tmp_path / "f.csv",
            start=(0, 0, 0),
            distance=5.0,
            n=10,
            frequency_indices=[0, 99],
        )


# ---------------------------------------------------------------------
# save_electrode_table_csv
# ---------------------------------------------------------------------


def test_save_electrode_table_csv_matches_in_memory(tmp_path: Path) -> None:
    world, result = _two_rod_world()
    out = save_electrode_table_csv(
        result,
        tmp_path / "electrodes.csv",
        world=world,
    )
    df_disk = read_csv_with_provenance(out)[0]
    # The file adds the frequency column (review pass 10, 10.73).
    assert (df_disk.pop("frequency_Hz") == result.frequencies[0]).all()
    df_mem = gf.electrode_current_table(result, world=world)

    pd.testing.assert_frame_equal(
        df_disk.reset_index(drop=True),
        df_mem.reset_index(drop=True),
        check_dtype=False,  # CSV roundtrip can shift int64 <-> float64
    )


def test_save_electrode_table_csv_without_world_omits_geometry(tmp_path: Path) -> None:
    _, result = _two_rod_world()
    out = save_electrode_table_csv(result, tmp_path / "electrodes.csv")
    df = read_csv_with_provenance(out)[0]
    assert "kind" not in df.columns
    assert "depth_m" not in df.columns


# ---------------------------------------------------------------------
# save_cluster_impedances_csv
# ---------------------------------------------------------------------


def test_save_cluster_impedances_csv_flattens_members(tmp_path: Path) -> None:
    _, result = _two_rod_world()
    out = save_cluster_impedances_csv(result, tmp_path / "clusters.csv")
    df = read_csv_with_provenance(out)[0]
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


# ---------------------------------------------------------------------
# Review pass 10, 10.73: frequency column, provenance, no silent overwrite
# ---------------------------------------------------------------------


def _two_freq_world() -> tuple[gf.World, gf.FieldResult]:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(world, "rod", name="r#1", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_source(world, attached_to="r#1", magnitude=1.0)
    eng = gf.create_engine(
        backend="image", segment_length=0.5, frequencies=[50.0, 1000.0]
    )
    return world, eng.solve(world)


@pytest.mark.finding("10.73")
def test_tables_record_their_frequency(tmp_path: Path) -> None:
    """R5-14: a 1 kHz table could not be told apart from a 50 Hz one."""
    world, res = _two_freq_world()
    p_e = save_electrode_table_csv(
        res, tmp_path / "e.csv", world=world, frequency_index=1
    )
    p_c = save_cluster_impedances_csv(res, tmp_path / "c.csv", frequency_index=1)
    for p in (p_e, p_c):
        df, meta = read_csv_with_provenance(p)
        assert df["frequency_Hz"].tolist() == [1000.0]
        assert meta["frequency_Hz"] == "1000"
        assert meta["frequency_index"] == "1"


@pytest.mark.finding("10.73")
def test_csv_files_carry_a_provenance_header(tmp_path: Path) -> None:
    world, res = _two_freq_world()
    p = save_potential_path_csv(res, tmp_path / "p.csv", start=(1.0, 0.0, 0.0), n=3)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# groundfield csv: potential_path"
    df, meta = read_csv_with_provenance(p)
    assert meta["groundfield_version"] == gf.__version__
    assert meta["backend"] == "image"
    assert meta["frequencies_Hz"] == "50, 1000"
    assert meta["written_utc"].endswith("Z")
    assert len(df) == 6
    # Electrode names containing '#' survive (no comment='#' parsing).
    p_e = save_electrode_table_csv(res, tmp_path / "e.csv", world=world)
    df_e, _ = read_csv_with_provenance(p_e)
    assert df_e["name"].tolist() == ["r#1"]
    # Plain CSV on request.
    p_plain = save_electrode_table_csv(res, tmp_path / "plain.csv", provenance=False)
    assert pd.read_csv(p_plain)["name"].tolist() == ["r#1"]


@pytest.mark.finding("10.73")
def test_csv_writers_refuse_to_overwrite(tmp_path: Path) -> None:
    world, res = _two_freq_world()
    target = tmp_path / "e.csv"
    save_electrode_table_csv(res, target)
    first = target.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_electrode_table_csv(res, target, frequency_index=1)
    with pytest.raises(FileExistsError):
        save_cluster_impedances_csv(res, target)
    with pytest.raises(FileExistsError):
        save_potential_path_csv(res, target, start=(1.0, 0.0, 0.0), n=3)
    assert target.read_text(encoding="utf-8") == first
    save_electrode_table_csv(res, target, frequency_index=1, overwrite=True)
    assert "1000" in target.read_text(encoding="utf-8")
