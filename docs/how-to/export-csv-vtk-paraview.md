# Export to CSV, VTK and ParaView

This guide writes the numbers of a solve to CSV tables for spreadsheets
and other tools, and the geometry and the surface potential to VTK files
for a 3-D viewer such as ParaView. Read it when results have to leave
Python.

`groundfield.io` writes plain tables (CSV) and legacy VTK files. Every CSV
starts with `# key: value` lines that record where the numbers come from
(groundfield version, UTC time, backend, frequency and the solve
provenance), and no writer overwrites an existing file unless you pass
`overwrite=True`. For complete, lossless result files use `gf.save_result`
instead (tutorial [T10](../tutorials/T10_reproducible_studies.ipynb)).

## 1. Solve the world to export

```python
from pathlib import Path

import numpy as np

import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0), name="station")
gf.create_electrode(world, "ring", name="ring", center=(0.0, 0.0, 0.6), radius=4.0)
for k, (x, y) in enumerate([(4.0, 0.0), (-4.0, 0.0)]):
    gf.create_electrode(world, "rod", name=f"rod_{k}", position=(x, y, 0.6), length=3.0)
    gf.create_conductor(world, name=f"bond_{k}", start="ring", end=f"rod_{k}")
gf.create_source(world, attached_to="ring", magnitude=100.0)
result = gf.create_engine(segment_length=0.5, frequencies=[50.0, 250.0]).solve(world)
out = Path("export")
out.mkdir(exist_ok=True)
```

## 2. Write and read tables

```python
gf.save_cluster_impedances_csv(result, out / "clusters.csv")
gf.save_electrode_table_csv(result, out / "electrodes.csv", world=world, frequency_index=1)
gf.save_potential_path_csv(result, out / "profile.csv", start=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), distance=50.0, n=101)

table, header = gf.read_csv_with_provenance(out / "electrodes.csv")
print(header["backend"], header["frequency_hz"], "Hz")
print(table[["name", "cluster_root", "abs_I", "share_of_cluster_pct"]])
```

<!-- doctest-check
assert header["backend"] == "image" and float(header["frequency_hz"]) == 250.0
assert abs(table["share_of_cluster_pct"].sum() - 100.0) < 1e-6
profile, _ = gf.read_csv_with_provenance(out / "profile.csv")
assert len(profile) == 2 * 101  # one row per point and frequency
-->

**What you get:** three CSV files in `export/`, and the electrode table
read back with its header. `gf.read_csv_with_provenance` returns the table
and the header lines as a dictionary; the hidden check asserts the backend
and frequency of the header and that the current shares add up to 100 %. The potential path holds one row per point and frequency
(columns `s`, `x`, `y`, `z`, `frequency_Hz`, `phi_re`, `phi_im`,
`abs_phi`). Other tools read the files with
`pandas.read_csv(path, comment="#")`; write a plain CSV without the header
with `provenance=False`. A second write to the same path raises
`FileExistsError`:

```python
try:
    gf.save_cluster_impedances_csv(result, out / "clusters.csv")
except FileExistsError as exc:
    print("FileExistsError:", exc)
gf.save_cluster_impedances_csv(result, out / "clusters.csv", overwrite=True)
```

## 3. Write VTK files

Two writers produce legacy VTK files (ASCII, double precision):

- `gf.export_geometry_vtk(world, path)` — every electrode and conductor
  as poly-lines;
- `gf.export_field_vtk(result, path, extent=..., z=..., n=...)` — the
  potential on a horizontal grid at depth `z` as a structured-points data
  set with the scalar fields of the complex potential.

```python
geometry = gf.export_geometry_vtk(world, out / "geometry.vtk")
field = gf.export_field_vtk(result, out / "surface_potential.vtk", extent=(-30.0, 30.0, -30.0, 30.0), z=0.0, n=(121, 121))
print(geometry.read_text().splitlines()[:4])
print(field.read_text().splitlines()[:5])
```

<!-- doctest-check
lines = field.read_text().splitlines()
assert lines[0].startswith("# vtk DataFile") and lines[2] == "ASCII"
assert any("double" in line for line in lines[:15])
-->

**What you get:** two ASCII VTK files; the header lines printed above
start with `# vtk DataFile`.

## 4. Open them in ParaView

1. *File → Open* both files, *Apply*.
2. The coordinates are those of groundfield: $z$ is the depth and points
   **downwards**. To see the electrodes below the surface, add a
   *Transform* filter with scale `(1, 1, -1)` to both data sets.
3. Colour the field data set by the real part of the potential (or its
   magnitude); add *Contour* for equipotential lines and *Tube* on the
   geometry for visible wires.
4. For several frequencies, export one field file per
   `frequency_index`.

## See also

- Tutorial [T10](../tutorials/T10_reproducible_studies.ipynb): lossless
  result files and reproducible studies.
- [Plot results](plot-results.md): the same quantities as matplotlib
  figures.
- API: [IO](../reference/api/io.md) and
  [result files](../reference/api/results-io.md).
