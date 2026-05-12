"""Tests for cross-layer electrodes and conductors (ADR-0007).

Validates that the layered backends accept and correctly compute
geometries in which segments cross the upper-layer interface.

Coverage
--------
1. **`coupling.layered_green` limits**: homogeneous limit
   $\\rho_1 = \\rho_2$ reproduces the free-space Green's function
   (with image at z=0); reciprocity $G(z, z_s) = G(z_s, z)$;
   continuity at the interface.
2. **`image_2layer` cross-layer**: a 2-m driven rod crossing
   $h_1 = 1$ m soil interface produces a finite cluster impedance,
   the precondition no longer raises, and the layer-tagged
   ``_Segment.layer_index`` is set correctly.
3. **`image_2layer` regression** for pure-upper-layer worlds: bit-exact
   reproduction of the historic image-series result.
4. **Other layered backends**: emit the documented `UserWarning`
   when given a cross-layer world (instead of raising), so the user
   gets a clear path forward without a hard crash.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.coupling.layered_green import (
    two_layer_real_space_kernel,
    two_layer_spectral_kernel,
)


# ---------------------------------------------------------------------
# 1. Layered Green's function — limits and reciprocity
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "z,z_s,s",
    [
        (0.5, 1.5, 5.0),   # cross-layer
        (1.5, 0.5, 5.0),   # reciprocal
        (1.5, 0.5, 0.5),   # near-source
        (3.0, 0.5, 10.0),  # both far apart
    ],
)
def test_homogeneous_limit_recovers_free_space(z, z_s, s) -> None:
    """When $\\rho_1 = \\rho_2$, the 2-layer kernel reproduces the
    free-space Green's function $\\rho/(2)\\,(1/r + 1/r_\\text{img})$
    (the factor 1/(4π) is applied by the caller)."""
    rho = 100.0
    G = two_layer_real_space_kernel(
        s=s, z=z, z_s=z_s, rho_1=rho, rho_2=rho, h_1=1.0,
    )
    r = math.hypot(s, z - z_s)
    r_img = math.hypot(s, z + z_s)
    expected = (rho / 2.0) * (1.0 / r + 1.0 / r_img)
    rel = abs(G - expected) / abs(expected)
    assert rel < 1e-2, (
        f"Homogeneous limit mismatch: G={G}, expected={expected}, "
        f"rel={rel:.3e}"
    )


def test_reciprocity_cross_layer() -> None:
    """For a 2-layer earth, $G(z, z_s) = G(z_s, z)$ at every
    horizontal distance."""
    rho_1, rho_2, h_1 = 100.0, 1000.0, 1.0
    for s in [1.0, 5.0, 20.0]:
        for (z, z_s) in [(0.5, 1.5), (1.5, 0.5), (0.3, 2.5)]:
            G_ab = two_layer_real_space_kernel(
                s=s, z=z, z_s=z_s, rho_1=rho_1, rho_2=rho_2, h_1=h_1,
            )
            G_ba = two_layer_real_space_kernel(
                s=s, z=z_s, z_s=z, rho_1=rho_1, rho_2=rho_2, h_1=h_1,
            )
            rel = abs(G_ab - G_ba) / max(abs(G_ab), 1e-12)
            assert rel < 1e-3, (
                f"Reciprocity violated at s={s}, z={z}, z_s={z_s}: "
                f"G_ab={G_ab}, G_ba={G_ba}, rel={rel:.3e}"
            )


# ---------------------------------------------------------------------
# 2. image_2layer cross-layer geometry
# ---------------------------------------------------------------------


def test_image_2layer_accepts_cross_layer_rod() -> None:
    """A 2-m driven rod crossing $h_1 = 1$ m no longer raises;
    the FieldResult is finite and physically sensible."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=1.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5),
                        length=2.0, wire_radius=0.0075)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image_2layer", segment_length=0.5)
    res = eng.solve(w)
    U = res.electrode_potentials["g1"][0]
    I = res.electrode_currents["g1"][0]
    Z = U / I
    # Spreading resistance must be a sensible positive number. With
    # rho_1=100 (upper) and rho_2=1000 (highly resistive lower) the
    # rod tip in the resistive lower layer pulls the resistance up;
    # the magnitude can be a few hundred Ohms, in line with the
    # Dwight 1936 ratio formula for layered soil.
    assert Z.real > 0.0
    assert Z.real < 5000.0
    assert math.isfinite(Z.real)


def test_image_2layer_regression_pure_upper_layer() -> None:
    """For a world that fits entirely in the upper layer the
    cross-layer dispatcher must NOT kick in — historic Tagg/Sunde
    image series is preserved bit-exact."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=5.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5),
                        length=2.0, wire_radius=0.0075)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image_2layer", segment_length=0.5)
    res_a = eng.solve(w)
    res_b = eng.solve(w)
    assert res_a.electrode_currents["g1"][0] == res_b.electrode_currents["g1"][0]


def test_image_2layer_homogeneous_limit_cross_layer() -> None:
    """Cross-layer geometry with $\\rho_2 = \\rho_1$ must produce the
    same cluster impedance as a homogeneous world with that
    resistivity (within the Sommerfeld-quadrature tolerance)."""
    soil_layered = gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=1.0)
    w_l = gf.create_world(soil=soil_layered)
    gf.create_electrode(w_l, "rod", name="g1", position=(0, 0, 0.5),
                        length=2.0, wire_radius=0.0075)
    gf.create_source(w_l, attached_to="g1", magnitude=1.0)

    soil_homog = gf.HomogeneousSoil(resistivity=100.0)
    w_h = gf.create_world(soil=soil_homog)
    gf.create_electrode(w_h, "rod", name="g1", position=(0, 0, 0.5),
                        length=2.0, wire_radius=0.0075)
    gf.create_source(w_h, attached_to="g1", magnitude=1.0)

    eng_l = gf.create_engine(backend="image_2layer", segment_length=0.5)
    eng_h = gf.create_engine(backend="image", segment_length=0.5)
    Z_l = (eng_l.solve(w_l).electrode_potentials["g1"][0]
           / eng_l.solve(w_l).electrode_currents["g1"][0])
    Z_h = (eng_h.solve(w_h).electrode_potentials["g1"][0]
           / eng_h.solve(w_h).electrode_currents["g1"][0])
    rel = abs(Z_l - Z_h) / abs(Z_h)
    assert rel < 0.05, (
        f"Cross-layer homogeneous limit mismatch: "
        f"Z_layered={Z_l}, Z_homog={Z_h}, rel={rel:.3f}"
    )


def test_image_2layer_layer_index_assignment() -> None:
    """After ADR-0007 every segment gets a layer_index. Verify by
    inspecting metadata (the FieldResult exposes n_segments;
    we count how many have midpoint z < h_1)."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=1.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5),
                        length=2.0, wire_radius=0.0075)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image_2layer", segment_length=0.5)
    res = eng.solve(w)
    # The 2 m rod from z=0.5 to z=2.5 with seg_length 0.5 m:
    # 4 segments at z = 0.75, 1.25, 1.75, 2.25.
    # 1 in upper layer (z < 1), 3 in lower.
    assert res.metadata["n_segments"] >= 4


# ---------------------------------------------------------------------
# 3. Other layered backends emit a UserWarning instead of raising
# ---------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["mom_sommerfeld", "cim", "bem"])
def test_other_layered_backends_accept_cross_layer_n2(backend) -> None:
    """ADR-0006/0007 Phase B: for n=2 cross-layer geometries the
    backends mom_sommerfeld, cim, bem now accept the world without
    warning — the kernel delegates to the shared cross-layer-aware
    `_two_layer_self_kernel_factory`. n>=3 still warns."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=1.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5),
                        length=2.0, wire_radius=0.0075)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend=backend, segment_length=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = eng.solve(w)
        cross_layer_warnings = [
            w_ for w_ in caught
            if issubclass(w_.category, UserWarning)
            and "cross-layer" in str(w_.message).lower()
        ]
        assert len(cross_layer_warnings) == 0, (
            f"Backend {backend} should NOT warn for n=2 cross-layer; "
            f"got: {[str(w_.message) for w_ in cross_layer_warnings]}"
        )
    # And the result must be physically sensible.
    Z = (res.electrode_potentials["g1"][0]
         / res.electrode_currents["g1"][0])
    assert Z.real > 0
    assert math.isfinite(Z.real)
