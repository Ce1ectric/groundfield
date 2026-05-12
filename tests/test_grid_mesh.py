"""Tests for the ``GridMeshElectrode`` primitive (Maschenerder).

Covers:

- Pydantic-level validation (``n_x``, ``n_y`` ≥ 1).
- Discretisation regression: ``GridMeshElectrode(n_x=k, n_y=k)`` and
  ``MeshElectrode(spacing=...)`` chosen so that both end up with the
  same wire layout must produce identical cluster impedances.
- Monotonicity: refining the inner grid must lower the cluster
  impedance (more current-emitting wire surface).
- Image backend vs. ``MeshElectrode`` cross-check on a small grid.
"""

from __future__ import annotations

import pytest

import groundfield as gf
from groundfield.solver.image import _discretize_electrode


SOIL = gf.HomogeneousSoil(resistivity=100.0)
ENG = gf.create_engine(backend="image", segment_length=0.1)


# ---------------------------------------------------------------------
# 1. Geometry validation
# ---------------------------------------------------------------------


def test_grid_mesh_rejects_zero_meshes() -> None:
    with pytest.raises(ValueError):
        gf.GridMeshElectrode(
            name="bad", corner=(0.0, 0.0, 0.5), size=(6.0, 6.0),
            n_x=0, n_y=3,
        )


def test_grid_mesh_connection_point_is_centre() -> None:
    g = gf.GridMeshElectrode(
        name="g", corner=(0.0, 0.0, 0.5), size=(6.0, 4.0),
        n_x=3, n_y=2,
    )
    assert g.connection_point == (3.0, 2.0, 0.5)


# ---------------------------------------------------------------------
# 2. Discretisation regression vs. legacy MeshElectrode
# ---------------------------------------------------------------------


def test_grid_mesh_segment_count_matches_legacy_mesh() -> None:
    """A 3×3 GridMesh equals a ``spacing = size / 3`` MeshElectrode.

    Both definitions produce a 4×4 wire layout (``n_x + 1`` and
    ``n_y + 1`` wires per direction), and therefore the same number
    of segments at the same midpoints for a given ``segment_length``.
    """
    legacy = gf.MeshElectrode(
        name="m", corner=(0.0, 0.0, 0.5), size=(6.0, 6.0), spacing=2.0,
    )
    grid = gf.GridMeshElectrode(
        name="g", corner=(0.0, 0.0, 0.5), size=(6.0, 6.0), n_x=3, n_y=3,
    )

    segs_legacy = _discretize_electrode(legacy, ds=0.1)
    segs_grid = _discretize_electrode(grid, ds=0.1)

    assert len(segs_legacy) == len(segs_grid)


def test_grid_mesh_cluster_impedance_matches_legacy_mesh() -> None:
    """The two equivalent layouts must yield the same cluster impedance."""
    w_legacy = gf.create_world(soil=SOIL)
    gf.create_electrode(
        w_legacy, "mesh", name="g1",
        corner=(0.0, 0.0, 0.5), size=(6.0, 6.0), spacing=2.0,
        wire_radius=0.005,
    )
    gf.create_source(w_legacy, attached_to="g1", magnitude=1.0)

    w_grid = gf.create_world(soil=SOIL)
    gf.create_electrode(
        w_grid, "grid_mesh", name="g1",
        corner=(0.0, 0.0, 0.5), size=(6.0, 6.0), n_x=3, n_y=3,
        wire_radius=0.005,
    )
    gf.create_source(w_grid, attached_to="g1", magnitude=1.0)

    Z_legacy = ENG.solve(w_legacy).cluster_impedance("g1")[0].real
    Z_grid = ENG.solve(w_grid).cluster_impedance("g1")[0].real
    assert Z_legacy == pytest.approx(Z_grid, rel=1e-6)


# ---------------------------------------------------------------------
# 3. Monotonicity: refining the inner grid lowers the resistance
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_grid_mesh_resistance_decreases_with_refinement(n: int) -> None:
    """``Z(n_x = n_y = n)`` is monotonically non-increasing in ``n``.

    A denser mesh adds wire surface inside the same footprint;
    extra surface only ever lowers the grounding impedance.
    """
    if n == 1:
        pytest.skip("baseline run; comparison happens via the next steps")
    w_coarse = gf.create_world(soil=SOIL)
    gf.create_electrode(
        w_coarse, "grid_mesh", name="g1",
        corner=(0.0, 0.0, 0.5), size=(6.0, 6.0),
        n_x=max(1, n - 1), n_y=max(1, n - 1), wire_radius=0.005,
    )
    gf.create_source(w_coarse, attached_to="g1", magnitude=1.0)

    w_fine = gf.create_world(soil=SOIL)
    gf.create_electrode(
        w_fine, "grid_mesh", name="g1",
        corner=(0.0, 0.0, 0.5), size=(6.0, 6.0),
        n_x=n, n_y=n, wire_radius=0.005,
    )
    gf.create_source(w_fine, attached_to="g1", magnitude=1.0)

    Z_coarse = ENG.solve(w_coarse).cluster_impedance("g1")[0].real
    Z_fine = ENG.solve(w_fine).cluster_impedance("g1")[0].real
    assert Z_fine <= Z_coarse + 1e-6, (
        f"n={n}: Z_fine = {Z_fine:.4f} should be <= Z_coarse = {Z_coarse:.4f}"
    )


# ---------------------------------------------------------------------
# 4. Sanity: cluster bonded with a rod
# ---------------------------------------------------------------------


def test_grid_mesh_bonded_to_rod_shares_cluster_potential() -> None:
    import numpy as np

    world = gf.create_world(soil=SOIL)
    grid = gf.create_electrode(
        world, "grid_mesh", name="grid",
        corner=(-3.0, -3.0, 0.5), size=(6.0, 6.0),
        n_x=2, n_y=2, wire_radius=0.005,
    )
    rod = gf.create_electrode(
        world, "rod", name="rod",
        position=(5.0, 0.0, 0.0), length=1.5, wire_radius=0.0125,
    )
    gf.create_conductor(world, name="bond", start=grid, end=rod)
    gf.create_source(world, attached_to="grid", magnitude=10.0)

    res = ENG.solve(world)
    u_grid = complex(np.mean(res.electrode_potentials["grid"][0]))
    u_rod = complex(np.mean(res.electrode_potentials["rod"][0]))
    assert abs(u_grid - u_rod) / abs(u_grid) < 1e-9
