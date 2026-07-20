"""Regression tests for the grid-crossing discretisation trap.

A rectangular grounding grid is crossed at its mesh corners. If the
segment count per wire is chosen so that segment **midpoints** (the
point-source locations of the image/MoM kernels) land exactly on those
crossings, a longitudinal and a transverse segment share a point source.
Their mutual distance is then zero, the 1 mm singularity clamp engages,
and the grid resistance comes back **~2x too large** for the
uniform-current ``image`` backend — silently, for an otherwise perfectly
valid grid (e.g. ``segment_length = 2.0`` on a 7 m pitch: 3.5 segments
per span, midpoints on every second crossing).

The discretiser now aligns the segment count per wire to a multiple of
the mesh-span count, so crossings fall on segment endpoints and no two
point sources coincide. These tests pin that invariant, the helper, and
the resulting resistance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.references import ieee80
from groundfield.solver.image import (
    _MIN_DISTANCE,
    _discretize_electrode,
    _segments_per_wire,
)


# ---------------------------------------------------------------------
# 1. The alignment helper.
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "span,ds,n_spans",
    [(70.0, 2.0, 10), (70.0, 2.8, 10), (70.0, 1.0, 5), (4.0, 0.7, 2), (3.0, 0.7, 3)],
)
def test_segments_per_wire_aligns_to_spans(span, ds, n_spans) -> None:
    n = _segments_per_wire(span, ds, n_spans)
    # A whole number of segments per mesh span -> crossings on endpoints.
    assert n % n_spans == 0
    # Never coarser than the request.
    assert n >= int(np.ceil(span / ds))


def test_segments_per_wire_noop_without_interior_crossings() -> None:
    # 0 or 1 span: nothing to align, plain ceil count.
    assert _segments_per_wire(10.0, 0.7, 1) == int(np.ceil(10.0 / 0.7))
    assert _segments_per_wire(10.0, 0.7, 0) == int(np.ceil(10.0 / 0.7))


# ---------------------------------------------------------------------
# 2. Geometric invariant: no two segment midpoints coincide.
# ---------------------------------------------------------------------
@pytest.mark.parametrize("ds", [2.0, 2.8, 3.5, 2.5, 1.4, 2.3333333])
def test_grid_has_no_coincident_point_sources(ds) -> None:
    """For a 7 m-pitch grid, no segment midpoint may sit on a crossing
    (which would make a longitudinal and a transverse point source
    coincide within the singularity clamp)."""
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world,
        "grid_mesh",
        name="grid",
        corner=(-35.0, -35.0, 0.5),
        size=(70.0, 70.0),
        n_x=10,
        n_y=10,
        wire_radius=0.005,
    )
    segs = _discretize_electrode(world.electrodes[0], ds)
    pts = np.array([s.midpoint for s in segs])
    # Pairwise horizontal distances; the smallest off-diagonal must clear
    # the singularity clamp.
    d = np.linalg.norm(pts[:, None, :2] - pts[None, :, :2], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > _MIN_DISTANCE, f"coincident point sources at ds={ds}"


# ---------------------------------------------------------------------
# 3. Physical: no 2x resistance error, no clamp warning, any ds.
# ---------------------------------------------------------------------
@pytest.mark.parametrize("ds", [2.0, 2.8, 2.5, 3.5])
def test_grid_resistance_stable_across_segment_length(ds) -> None:
    """The image-backend grid resistance at a *trap* ``segment_length``
    now matches Sverak's estimate (previously ~2x too large) and raises
    no singularity-clamp warning."""
    rho, side, ndiv, depth = 100.0, 70.0, 10, 0.5  # 7 m pitch
    total_length = 2.0 * (ndiv + 1) * side
    area = side * side
    r_ref = ieee80.grid_resistance_sverak(rho, total_length, area, depth)

    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=rho))
    gf.create_electrode(
        world,
        "grid_mesh",
        name="grid",
        corner=(-side / 2.0, -side / 2.0, depth),
        size=(side, side),
        n_x=ndiv,
        n_y=ndiv,
        wire_radius=0.005,
    )
    gf.create_source(world, attached_to="grid", magnitude=1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine = gf.create_engine(backend="image", segment_length=ds, frequencies=[0.0])
        r_num = world.solve(engine).cluster_impedance("grid")[0].real

    assert abs(r_num - r_ref) / r_ref < 0.05, f"ds={ds}: {r_num} vs {r_ref}"
    clamp = [w for w in caught if "singularity clamp" in str(w.message)]
    assert not clamp, f"unexpected clamp warning at ds={ds}: {clamp}"
