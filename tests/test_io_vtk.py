"""Tests for the VTK writers in :mod:`groundfield.io.vtk`.

Validates:

- :func:`export_geometry_vtk` produces a syntactically valid
  legacy-ASCII VTK PolyData file with the expected number of
  points / lines and a ``role`` cell-data scalar,
- :func:`export_field_vtk` produces a STRUCTURED_POINTS file
  with the right ``DIMENSIONS``, ``ORIGIN`` and ``SPACING`` and
  the correct point-count payload,
- the empty-world edge case (header-only output, no crash),
- error paths for ``export_field_vtk`` (bad extent, n < 2),
- top-level exports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import groundfield as gf
from groundfield.io.vtk import export_field_vtk, export_geometry_vtk


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _mixed_world() -> gf.World:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil, name="mixed")
    gf.create_electrode(world, "rod", name="rod_a", position=(0, 0, 0.5), length=1.5)
    gf.create_electrode(world, "ring", name="ring_a", center=(10, 0, 0.8), radius=2.5)
    gf.create_electrode(
        world, "strip", name="strip_a", start=(0, 5, 0.6), end=(8, 5, 0.6)
    )
    gf.create_electrode(
        world,
        "grid_mesh",
        name="mesh_a",
        corner=(15, -3, 0.7),
        size=(6, 4),
        n_x=3,
        n_y=2,
    )
    gf.create_conductor(
        world, name="bond", start="rod_a", end="ring_a", conductor_type="bare_copper"
    )
    gf.create_source(world, attached_to="rod_a", magnitude=10.0)
    return world


def _solved_single_rod() -> gf.FieldResult:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=10.0)
    eng = gf.create_engine(backend="image", segment_length=0.05)
    return eng.solve(world)


# ---------------------------------------------------------------------
# export_geometry_vtk
# ---------------------------------------------------------------------


def test_export_geometry_vtk_produces_valid_polydata(tmp_path: Path) -> None:
    out = export_geometry_vtk(_mixed_world(), tmp_path / "world.vtk")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# vtk DataFile Version 3.0")
    assert "ASCII" in text
    assert "DATASET POLYDATA" in text
    assert "POINTS " in text
    assert "LINES " in text
    assert "CELL_DATA " in text
    assert "SCALARS role int 1" in text


def test_export_geometry_vtk_line_count_matches_world(tmp_path: Path) -> None:
    """The number of LINES in the file must equal the number of polylines
    expected from the geometry.

    Mixed world: 1 rod (1 polyline) + 1 ring (1) + 1 strip (1) +
    1 grid_mesh with n_x=3, n_y=2 ((n_y+1) + (n_x+1) = 3 + 4 = 7)
    + 1 conductor = 11 polylines total.
    """
    out = export_geometry_vtk(_mixed_world(), tmp_path / "world.vtk")
    text = out.read_text(encoding="utf-8")
    # Find the LINES header.
    for line in text.splitlines():
        if line.startswith("LINES"):
            n_lines = int(line.split()[1])
            break
    else:
        pytest.fail("LINES header not found in VTK output")
    assert n_lines == 11


def test_export_geometry_vtk_empty_world(tmp_path: Path) -> None:
    """An empty world produces a header-only POLYDATA file."""
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    out = export_geometry_vtk(world, tmp_path / "empty.vtk")
    text = out.read_text(encoding="utf-8")
    assert "DATASET POLYDATA" in text
    assert "POINTS 0 double" in text  # double since review pass 10 (10.10)


def test_export_geometry_vtk_role_scalar_distinguishes_electrodes_and_conductors(
    tmp_path: Path,
) -> None:
    out = export_geometry_vtk(_mixed_world(), tmp_path / "world.vtk")
    text = out.read_text(encoding="utf-8")
    # Pull out the role values that follow LOOKUP_TABLE default.
    lines = text.splitlines()
    idx = next(i for i, l in enumerate(lines) if l == "LOOKUP_TABLE default")
    # Skip the first LOOKUP_TABLE inside the LINES header? No, only one
    # CELL_DATA scalar block exists; this is it. Read forward as ints
    # until we hit a non-int line or EOF.
    role_values = []
    for l in lines[idx + 1 :]:
        try:
            role_values.append(int(l.strip()))
        except ValueError:
            break
    assert 0 in role_values  # at least one electrode
    assert 1 in role_values  # at least one conductor


# ---------------------------------------------------------------------
# export_field_vtk
# ---------------------------------------------------------------------


def test_export_field_vtk_structured_points_header(tmp_path: Path) -> None:
    result = _solved_single_rod()
    out = export_field_vtk(
        result,
        tmp_path / "phi.vtk",
        extent=(-10.0, 10.0, -10.0, 10.0),
        z=0.0,
        n=(20, 25),
    )
    text = out.read_text(encoding="utf-8")
    assert "DATASET STRUCTURED_POINTS" in text
    assert "DIMENSIONS 20 25 1" in text
    assert "ORIGIN -10" in text
    # Two scalar blocks: real and imag part.
    assert text.count("SCALARS") == 2
    assert "potential_re" in text
    assert "potential_im" in text


def test_export_field_vtk_payload_size(tmp_path: Path) -> None:
    result = _solved_single_rod()
    n = (12, 8)
    out = export_field_vtk(
        result,
        tmp_path / "phi.vtk",
        extent=(-5.0, 5.0, -3.0, 3.0),
        n=n,
    )
    text = out.read_text(encoding="utf-8")
    # Count numerical lines after each LOOKUP_TABLE default.
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if l == "LOOKUP_TABLE default"]
    n_pts = n[0] * n[1]
    for s in starts:
        # n_pts numeric lines must follow.
        for k in range(1, n_pts + 1):
            try:
                float(lines[s + k])
            except ValueError:
                pytest.fail(f"Non-numeric payload at line {s + k}")


def test_export_field_vtk_rejects_bad_extent(tmp_path: Path) -> None:
    result = _solved_single_rod()
    with pytest.raises(ValueError, match="extent"):
        export_field_vtk(result, tmp_path / "phi.vtk", extent=(5.0, 5.0, -1.0, 1.0))
    with pytest.raises(ValueError, match="extent"):
        export_field_vtk(result, tmp_path / "phi.vtk", extent=(-5.0, 5.0, 1.0, -1.0))


def test_export_field_vtk_rejects_too_few_grid_points(tmp_path: Path) -> None:
    result = _solved_single_rod()
    with pytest.raises(ValueError, match="n"):
        export_field_vtk(
            result, tmp_path / "phi.vtk", extent=(-5.0, 5.0, -5.0, 5.0), n=(1, 10)
        )


# ---------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------


def test_top_level_exports_vtk_helpers() -> None:
    needed = {"export_geometry_vtk", "export_field_vtk"}
    assert needed.issubset(set(gf.__all__))
    assert all(hasattr(gf, name) for name in needed)


# ---------------------------------------------------------------------
# Review pass 10, 10.10: header sanitising, double precision
# ---------------------------------------------------------------------


@pytest.mark.finding("10.10")
def test_world_name_cannot_break_the_header(tmp_path: Path) -> None:
    """R5-15: a newline in world.name produced an invalid file."""
    world = gf.create_world(
        soil=gf.HomogeneousSoil(resistivity=100.0),
        name="site 4711\nDATASET STRUCTURED_POINTS\tx",
    )
    gf.create_electrode(world, "rod", name="r", position=(0, 0, 0.5), length=2.0)
    lines = export_geometry_vtk(world, tmp_path / "g.vtk").read_text().splitlines()
    assert lines[0] == "# vtk DataFile Version 3.0"
    assert lines[1] == "groundfield geometry: site 4711 DATASET STRUCTURED_POINTS x"
    assert lines[2] == "ASCII"
    assert lines[3] == "DATASET POLYDATA"


@pytest.mark.finding("10.10")
def test_header_line_is_limited_to_256_bytes(tmp_path: Path) -> None:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0), name="ä" * 400)
    gf.create_electrode(world, "rod", name="r", position=(0, 0, 0.5), length=2.0)
    raw = export_geometry_vtk(world, tmp_path / "g.vtk").read_bytes().split(b"\n")
    assert len(raw[1]) <= 256
    raw[1].decode("utf-8")  # no character cut in half
    assert raw[2] == b"ASCII"


@pytest.mark.finding("10.10")
def test_points_are_double_precision_at_utm_scale(tmp_path: Path) -> None:
    """float32 quantised UTM coordinates (y ~ 5.79e6 m) by up to 0.19 m."""
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world, "rod", name="r", position=(612345.678, 5791234.567, 0.5), length=2.0
    )
    text = export_geometry_vtk(world, tmp_path / "g.vtk").read_text()
    lines = text.splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("POINTS"))
    assert lines[i] == "POINTS 2 double"
    x, y, z = (float(v) for v in lines[i + 1].split())
    assert (x, y, z) == (612345.678, 5791234.567, 0.5)


@pytest.mark.finding("10.10")
def test_field_scalars_are_double(tmp_path: Path) -> None:
    result = _solved_single_rod()
    text = export_field_vtk(
        result, tmp_path / "f.vtk", extent=(-2.0, 2.0, -2.0, 2.0), n=(3, 3)
    ).read_text()
    assert "SCALARS potential_re double 1" in text
    assert "SCALARS potential_im double 1" in text
    assert " float" not in text
