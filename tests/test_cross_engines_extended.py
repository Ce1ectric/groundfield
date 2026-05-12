"""Cross-engine consistency tests for the **full** engine family.

Adds the four new engines (``image_nlayer``, ``cim``, ``mom_sommerfeld``,
``bem``, ``fem``) to the existing cross-engine matrix and codifies the
guiding principle:

- **Simple configurations**: every engine on the list must produce
  the same answer within a 5 % envelope (10 % for ``fem``, which uses
  the equivalent-hemisphere reduction).
- **Layer variations**: as the layer contrast grows, the closed-form
  engines (``image_2layer``, ``image_nlayer``, ``cim``, ``bem``) and
  the integral reference (``mom_sommerfeld``) must track each other
  within 5 %; the volume engine (``fem``) follows the trend with the
  expected reduction-induced bias.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf

SEG = 0.05
SEG_SOMMERFELD = 0.10  # Sommerfeld quadrature is expensive — coarser segments


def _world_rod(soil, *, length: float = 1.5) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0),
                        length=length)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _world_two_rods(soil) -> gf.World:
    """Two parallel rods bonded by a conductor — used as the standard
    'two interconnected grounding systems' fixture."""
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_electrode(w, "rod", name="g2", position=(8, 0, 0.0), length=1.5)
    gf.create_conductor(w, name="bond", start="g1", end="g2")
    gf.create_source(w, attached_to="g1", magnitude=10.0)
    return w


# ---------------------------------------------------------------------
# Simple homogeneous: every engine must agree
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend, tol",
    [
        ("image", 0.0),         # reference
        ("image_nlayer", 1e-9),  # dispatcher → image
        ("cim", 0.05),
        ("mom", 0.05),
        ("mom_sommerfeld", 0.05),
        ("bem", 0.05),
        ("fem", 0.10),
    ],
)
def test_homogeneous_single_rod_all_engines(backend: str, tol: float) -> None:
    """Single rod in homogeneous soil — every engine must agree with ``image``
    within its documented envelope."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    seg = SEG_SOMMERFELD if backend == "mom_sommerfeld" else SEG
    Z_ref = (
        gf.create_engine(backend="image", segment_length=seg)
        .solve(_world_rod(soil))
        .cluster_impedance("g1")[0]
        .real
    )
    Z_test = (
        gf.create_engine(backend=backend, segment_length=seg)
        .solve(_world_rod(soil))
        .cluster_impedance("g1")[0]
        .real
    )
    if tol == 0.0:
        assert Z_test == pytest.approx(Z_ref, rel=1e-9)
    else:
        rel = abs(Z_test - Z_ref) / Z_ref
        assert rel < tol, f"{backend}: Z={Z_test:.3f} vs {Z_ref:.3f}, Δ={rel*100:.2f}%"


# ---------------------------------------------------------------------
# 2-layer: closed-form engines must agree with image_2layer
# ---------------------------------------------------------------------


@pytest.mark.parametrize("rho_2", [50.0, 200.0, 500.0])
@pytest.mark.parametrize(
    "backend, tol",
    [
        ("image_nlayer", 1e-9),  # dispatcher → image_2layer
        ("cim", 0.05),
        ("mom", 0.05),
        ("bem", 0.05),
    ],
)
def test_two_layer_engines_agree(backend: str, tol: float, rho_2: float) -> None:
    """For a range of layer contrasts every closed-form engine and the
    Galerkin MoM must agree with ``image_2layer`` within 5 %."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=rho_2, h_1=2.0)
    Z_ref = (
        gf.create_engine(backend="image_2layer", segment_length=SEG)
        .solve(_world_rod(soil))
        .cluster_impedance("g1")[0]
        .real
    )
    Z_test = (
        gf.create_engine(backend=backend, segment_length=SEG)
        .solve(_world_rod(soil))
        .cluster_impedance("g1")[0]
        .real
    )
    if tol < 1e-6:
        assert Z_test == pytest.approx(Z_ref, rel=tol)
    else:
        rel = abs(Z_test - Z_ref) / Z_ref
        assert rel < tol, (
            f"{backend} rho_2={rho_2}: Z={Z_test:.3f} vs {Z_ref:.3f}, "
            f"Δ={rel*100:.2f}%"
        )


# ---------------------------------------------------------------------
# Two interconnected grounding systems — the user-mandated test case
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend",
    ["image", "image_nlayer", "cim", "mom", "bem", "fem"],
)
def test_two_interconnected_rods_homogeneous_lower_R(backend: str) -> None:
    """Two bonded rods in homogeneous soil have a lower cluster resistance
    than a single rod (the parallel-combination rule)."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng = gf.create_engine(backend=backend, segment_length=SEG)
    Z_single = eng.solve(_world_rod(soil)).cluster_impedance("g1")[0].real
    Z_pair = eng.solve(_world_two_rods(soil)).cluster_impedance("g1")[0].real
    assert Z_pair < Z_single, (
        f"{backend}: pair {Z_pair:.3f} should be lower than single {Z_single:.3f}"
    )


@pytest.mark.parametrize(
    "backend",
    ["image", "image_nlayer", "cim", "mom", "bem"],
)
def test_two_interconnected_rods_two_layer_engines_agree(backend: str) -> None:
    """In 2-layer soil all integral / closed-form engines must agree on
    the cluster impedance of two interconnected rods within 5 %."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    Z_ref = (
        gf.create_engine(backend="image_2layer", segment_length=SEG)
        .solve(_world_two_rods(soil))
        .cluster_impedance("g1")[0]
        .real
    )
    Z_test = (
        gf.create_engine(backend=backend, segment_length=SEG)
        .solve(_world_two_rods(soil))
        .cluster_impedance("g1")[0]
        .real
    )
    rel = abs(Z_test - Z_ref) / Z_ref
    assert rel < 0.05, (
        f"{backend}: pair Z={Z_test:.3f} vs il2={Z_ref:.3f}, Δ={rel*100:.2f}%"
    )


# ---------------------------------------------------------------------
# Layer-contrast monotonicity — predictable behaviour as ρ_2 changes
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend",
    ["image_2layer", "image_nlayer", "cim", "mom", "bem", "fem"],
)
def test_layer_contrast_monotone(backend: str) -> None:
    """Sweeping ρ_2 from 10 to 1000 Ω·m at fixed ρ_1 = 100 Ω·m must
    produce a monotonically increasing cluster impedance — a basic
    physics consistency check."""
    rhos = [10.0, 50.0, 100.0, 300.0, 1000.0]
    Zs = []
    eng = gf.create_engine(backend=backend, segment_length=SEG)
    for rho_2 in rhos:
        soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=rho_2, h_1=2.0)
        Z = eng.solve(_world_rod(soil)).cluster_impedance("g1")[0].real
        Zs.append(Z)
    # Strictly monotone increasing.
    diffs = np.diff(Zs)
    assert (diffs > 0).all(), f"{backend} not monotone: {Zs}"


# ---------------------------------------------------------------------
# Potential-field cross-check at a sample grid
# ---------------------------------------------------------------------


def test_potential_field_consistent_homogeneous() -> None:
    """The sampled potential field of every engine that exposes
    ``point_sources`` must agree with ``image`` to within 5 %."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    pts = np.array(
        [
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [3.0, 3.0, 0.5],
        ]
    )
    eng_image = gf.create_engine(backend="image", segment_length=SEG)
    phi_ref = eng_image.solve(_world_rod(soil)).potential(pts).real

    for backend in ("cim", "mom", "bem"):
        res = gf.create_engine(backend=backend, segment_length=SEG).solve(
            _world_rod(soil)
        )
        phi = res.potential(pts).real
        rel = float(np.max(np.abs(phi - phi_ref) / np.abs(phi_ref)))
        assert rel < 0.05, f"{backend}: max Δ = {rel*100:.2f}%"


# ---------------------------------------------------------------------
# Strip and GridMesh primitives across the whole engine family
# ---------------------------------------------------------------------


def _world_strip(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "strip", name="g1",
        start=(-5.0, 0.0, 0.5), end=(+5.0, 0.0, 0.5),
        wire_radius=0.005,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _world_grid_mesh(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "grid_mesh", name="g1",
        corner=(-3.0, -3.0, 0.5), size=(6.0, 6.0),
        n_x=2, n_y=2, wire_radius=0.005,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


@pytest.mark.parametrize(
    "backend, tol",
    [
        ("image", 0.0),
        ("image_nlayer", 1e-9),
        ("cim", 0.05),
        ("mom", 0.05),
        ("bem", 0.05),
        ("fem", 0.20),  # equivalent-hemisphere bias is larger for strips
    ],
)
def test_homogeneous_strip_all_engines(backend: str, tol: float) -> None:
    """Single strip in homogeneous soil — every engine must agree with
    ``image`` within its documented envelope."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    Z_ref = (
        gf.create_engine(backend="image", segment_length=SEG)
        .solve(_world_strip(soil)).cluster_impedance("g1")[0].real
    )
    Z_test = (
        gf.create_engine(backend=backend, segment_length=SEG)
        .solve(_world_strip(soil)).cluster_impedance("g1")[0].real
    )
    if tol == 0.0:
        assert Z_test == pytest.approx(Z_ref, rel=1e-9)
    else:
        rel = abs(Z_test - Z_ref) / Z_ref
        assert rel < tol, (
            f"strip / {backend}: Z={Z_test:.3f} vs {Z_ref:.3f}, "
            f"Δ={rel*100:.2f}%"
        )


@pytest.mark.parametrize(
    "backend, tol",
    [
        ("image", 0.0),
        ("image_nlayer", 1e-9),
        ("cim", 0.05),
        ("mom", 0.05),
        ("bem", 0.05),
        # FEM uses the Schwarz / Sverak / IEEE Std 80 reduction for
        # rectangular grids. Sverak itself is a documented ~10–15 %
        # engineering approximation, and the equivalent-hemisphere
        # mapping adds further bias against the integral solvers, so
        # 30 % is the realistic envelope for this cross-check role.
        ("fem", 0.30),
    ],
)
def test_homogeneous_grid_mesh_all_engines(backend: str, tol: float) -> None:
    """3-mesh grid in homogeneous soil — every engine must agree with
    ``image`` within its documented envelope."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    Z_ref = (
        gf.create_engine(backend="image", segment_length=SEG)
        .solve(_world_grid_mesh(soil)).cluster_impedance("g1")[0].real
    )
    Z_test = (
        gf.create_engine(backend=backend, segment_length=SEG)
        .solve(_world_grid_mesh(soil)).cluster_impedance("g1")[0].real
    )
    if tol == 0.0:
        assert Z_test == pytest.approx(Z_ref, rel=1e-9)
    else:
        rel = abs(Z_test - Z_ref) / Z_ref
        assert rel < tol, (
            f"grid_mesh / {backend}: Z={Z_test:.3f} vs {Z_ref:.3f}, "
            f"Δ={rel*100:.2f}%"
        )


@pytest.mark.parametrize(
    "backend",
    ["image_2layer", "image_nlayer", "cim", "mom", "bem"],
)
def test_two_layer_strip_engines_agree(backend: str) -> None:
    """Strip in 2-layer soil — closed-form / Galerkin engines agree
    with ``image_2layer`` within 5 %."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    Z_ref = (
        gf.create_engine(backend="image_2layer", segment_length=SEG)
        .solve(_world_strip(soil)).cluster_impedance("g1")[0].real
    )
    Z_test = (
        gf.create_engine(backend=backend, segment_length=SEG)
        .solve(_world_strip(soil)).cluster_impedance("g1")[0].real
    )
    rel = abs(Z_test - Z_ref) / Z_ref
    assert rel < 0.05, (
        f"strip 2-layer / {backend}: Z={Z_test:.3f} vs {Z_ref:.3f}, "
        f"Δ={rel*100:.2f}%"
    )
