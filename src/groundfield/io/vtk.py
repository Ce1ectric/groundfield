"""Legacy ASCII VTK exports for ParaView / VisIt / Mayavi.

This module writes two flavours of the legacy ASCII VTK file
format (``.vtk``), keeping the dependency surface zero — no
``pyvista`` or ``vtk`` Python bindings are required:

* :func:`export_geometry_vtk` — POLYDATA with the electrode
  wires (rod, ring, strip, mesh, grid_mesh) and the conductor
  line segments. Useful for "drag the world into ParaView and
  rotate it" inspections of large typical networks.
* :func:`export_field_vtk` — STRUCTURED_POINTS with the soil
  surface (or any horizontal slice) sampled on a regular
  ``(N_x, N_y)`` grid. The potential is exported as a single
  scalar field; ParaView's contour / colour-map filters take
  it from there.

Why legacy ASCII VTK?
---------------------
The legacy format is the smallest common denominator: every VTK
reader since the 90s opens it, the on-disk layout is plain text
(diff-able, version-controllable), and the writer is ~30 lines
of pure-Python without any external library. For large research
runs prefer ``pyvista`` once you need binary I/O and unstructured
grids; for production-grade networks at notebook scale this format is
fast enough.

References
----------
- Schroeder, W., Martin, K., Lorensen, B. (2006). *The
  Visualization Toolkit*, 4th ed., Kitware. Section 19.5
  (Legacy file format).
- VTK File Formats reference:
  https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = ["export_geometry_vtk", "export_field_vtk"]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _ensure_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _polylines_for_electrode(e) -> list[np.ndarray]:
    """Return one or more 3-D polylines (each a (N, 3) array)
    sampling the electrode geometry."""
    from groundfield.geometry.electrodes import (
        GridMeshElectrode,
        MeshElectrode,
        RingElectrode,
        RodElectrode,
        StripElectrode,
    )

    if isinstance(e, RodElectrode):
        x, y, z = e.position
        return [np.array([[x, y, z], [x, y, z + e.length]], dtype=float)]
    if isinstance(e, RingElectrode):
        cx, cy, cz = e.center
        theta = np.linspace(0.0, 2.0 * np.pi, 64)
        ring = np.column_stack(
            [cx + e.radius * np.cos(theta),
             cy + e.radius * np.sin(theta),
             np.full_like(theta, cz)]
        )
        return [ring]
    if isinstance(e, StripElectrode):
        return [np.array([list(e.start), list(e.end)], dtype=float)]
    if isinstance(e, GridMeshElectrode):
        cx, cy, cz = e.corner
        dx, dy = e.size
        polys = []
        # n_y + 1 wires running parallel to x at distinct y values.
        for k in range(e.n_y + 1):
            yk = cy + dy * k / e.n_y
            polys.append(np.array([[cx, yk, cz], [cx + dx, yk, cz]], dtype=float))
        # n_x + 1 wires running parallel to y at distinct x values.
        for k in range(e.n_x + 1):
            xk = cx + dx * k / e.n_x
            polys.append(np.array([[xk, cy, cz], [xk, cy + dy, cz]], dtype=float))
        return polys
    if isinstance(e, MeshElectrode):
        cx, cy, cz = e.corner
        dx, dy = e.size
        # Outer rectangle only (the spacing-driven inner mesh would
        # require us to mirror the discretiser; not worth the
        # complexity for an on-disk visualisation).
        rect = np.array(
            [[cx, cy, cz], [cx + dx, cy, cz],
             [cx + dx, cy + dy, cz], [cx, cy + dy, cz],
             [cx, cy, cz]],
            dtype=float,
        )
        return [rect]
    return []  # pragma: no cover - defensive


def _format_lines_block(polylines: list[np.ndarray], pad: int = 0) -> tuple[list[np.ndarray], list[list[int]], int]:
    """Return ``(points, line_indices, total_n_points)`` for VTK LINES."""
    points: list[np.ndarray] = []
    lines: list[list[int]] = []
    offset = pad
    for poly in polylines:
        idx = list(range(offset, offset + poly.shape[0]))
        lines.append(idx)
        points.append(poly)
        offset += poly.shape[0]
    return points, lines, offset


# ---------------------------------------------------------------------
# Geometry export — POLYDATA
# ---------------------------------------------------------------------


def export_geometry_vtk(world: "World", path: str | Path) -> Path:
    """Export the world geometry as a legacy ASCII VTK PolyData file.

    Writes electrodes (rods, rings, strips, mesh / grid_mesh
    perimeters) **and** conductors as 3-D polylines. The on-disk
    layout uses VTK's ``DATASET POLYDATA`` with a ``LINES`` block;
    cell data carries an integer ``role`` field (0 = electrode,
    1 = conductor) so colour-by-role works directly in ParaView.

    Parameters
    ----------
    world
        World to export. May be empty (the writer still produces
        a syntactically valid header-only file).
    path
        Destination ``.vtk`` path. Parent directories are created
        automatically.

    Returns
    -------
    pathlib.Path
        The path the file was written to.
    """
    polylines: list[np.ndarray] = []
    roles: list[int] = []

    for e in world.electrodes:
        for poly in _polylines_for_electrode(e):
            polylines.append(poly)
            roles.append(0)

    for c in world.conductors:
        seg = np.array([list(c.start), list(c.end)], dtype=float)
        polylines.append(seg)
        roles.append(1)

    out = _ensure_path(path)
    buf = StringIO()
    buf.write("# vtk DataFile Version 3.0\n")
    buf.write(f"groundfield geometry: {world.name}\n")
    buf.write("ASCII\n")
    buf.write("DATASET POLYDATA\n")

    if not polylines:
        buf.write("POINTS 0 float\n")
        out.write_text(buf.getvalue(), encoding="utf-8")
        return out

    n_points = sum(p.shape[0] for p in polylines)
    n_lines = len(polylines)
    n_line_data = n_lines + n_points  # one count + entries per line

    buf.write(f"POINTS {n_points} float\n")
    for poly in polylines:
        for x, y, z in poly:
            buf.write(f"{x:.17g} {y:.17g} {z:.17g}\n")

    buf.write(f"LINES {n_lines} {n_line_data}\n")
    offset = 0
    for poly in polylines:
        n = poly.shape[0]
        idx_list = " ".join(str(offset + k) for k in range(n))
        buf.write(f"{n} {idx_list}\n")
        offset += n

    # Cell data: role per polyline (0 = electrode, 1 = conductor).
    buf.write(f"CELL_DATA {n_lines}\n")
    buf.write("SCALARS role int 1\n")
    buf.write("LOOKUP_TABLE default\n")
    for r in roles:
        buf.write(f"{r}\n")

    out.write_text(buf.getvalue(), encoding="utf-8")
    return out


# ---------------------------------------------------------------------
# Field export — STRUCTURED_POINTS
# ---------------------------------------------------------------------


def export_field_vtk(
    result: "FieldResult",
    path: str | Path,
    *,
    extent: tuple[float, float, float, float],
    z: float = 0.0,
    n: tuple[int, int] = (120, 120),
    frequency_index: int = 0,
) -> Path:
    """Sample the potential on a horizontal grid and write a VTK file.

    Evaluates :meth:`FieldResult.potential` on a regular
    :math:`N_x \\times N_y` grid in the plane :math:`z = z_0` and
    writes a ``DATASET STRUCTURED_POINTS`` (a.k.a. uniform grid)
    with a single scalar field ``potential_re``. The imaginary
    part is included as a second scalar ``potential_im`` for
    above-DC typical studies.

    Parameters
    ----------
    result
        Solver output. Must carry ``point_sources`` (not a stub).
    path
        Destination ``.vtk`` path.
    extent
        Plot extent ``(x_min, x_max, y_min, y_max)`` in metres.
    z
        Slice depth in metres (default ``0.0`` — soil surface).
        Positive ``z`` is below ground.
    n
        Grid resolution ``(n_x, n_y)``. Default ``(120, 120)``.
    frequency_index
        Index into :attr:`FieldResult.frequencies`. Default 0.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    ValueError
        On bad extent or non-positive grid sizes.
    """
    x_min, x_max, y_min, y_max = extent
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"extent must satisfy x_max > x_min and y_max > y_min, "
            f"got {extent!r}."
        )
    n_x, n_y = n
    if n_x < 2 or n_y < 2:
        raise ValueError(f"n must be >= (2, 2), got {n!r}.")

    xs = np.linspace(float(x_min), float(x_max), int(n_x))
    ys = np.linspace(float(y_min), float(y_max), int(n_y))
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [X.ravel(), Y.ravel(), np.full(X.size, float(z))]
    )
    phi = result.potential(pts, frequency_index=frequency_index)
    phi = phi.reshape(X.shape)

    dx = (x_max - x_min) / (n_x - 1)
    dy = (y_max - y_min) / (n_y - 1)

    out = _ensure_path(path)
    buf = StringIO()
    buf.write("# vtk DataFile Version 3.0\n")
    buf.write(
        f"groundfield potential @ z={z} m, "
        f"f={result.frequencies[frequency_index]} Hz\n"
    )
    buf.write("ASCII\n")
    buf.write("DATASET STRUCTURED_POINTS\n")
    buf.write(f"DIMENSIONS {n_x} {n_y} 1\n")
    buf.write(f"ORIGIN {x_min:.17g} {y_min:.17g} {z:.17g}\n")
    buf.write(f"SPACING {dx:.17g} {dy:.17g} 1\n")

    n_pts = n_x * n_y
    buf.write(f"POINT_DATA {n_pts}\n")
    buf.write("SCALARS potential_re float 1\n")
    buf.write("LOOKUP_TABLE default\n")
    # The order in STRUCTURED_POINTS is x fastest, then y, then z.
    # numpy meshgrid with indexing='xy' gives Y[i, j], X[i, j];
    # ravel() with C-order then walks j (x) before i (y) — exactly
    # what VTK expects.
    for v in phi.real.ravel():
        buf.write(f"{float(v):.17g}\n")

    buf.write("SCALARS potential_im float 1\n")
    buf.write("LOOKUP_TABLE default\n")
    for v in phi.imag.ravel():
        buf.write(f"{float(v):.17g}\n")

    out.write_text(buf.getvalue(), encoding="utf-8")
    return out
