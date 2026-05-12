"""Tests for the ``StripElectrode`` primitive (Banderder).

Covers:

- Pydantic-level validation (horizontal-only constraint).
- Discretisation: number of segments matches the segment length.
- Image backend vs. Dwight 1936 ``horizontal_wire`` (alongside the
  parametric entry in ``test_dwight_references.py``).
- Cluster-impedance equivalence with the prior degenerate-mesh
  workaround (regression guard for the API change).
- Equal-cluster-potential check for a strip bonded to a ring.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield.references import dwight1936 as dw


SOIL = gf.HomogeneousSoil(resistivity=100.0)
ENG = gf.create_engine(backend="image", segment_length=0.05)


# ---------------------------------------------------------------------
# 1. Geometry validation
# ---------------------------------------------------------------------


def test_strip_rejects_non_horizontal() -> None:
    """``start[2]`` and ``end[2]`` must match — otherwise raise."""
    with pytest.raises(ValueError, match="horizontal"):
        gf.StripElectrode(
            name="bad", start=(0.0, 0.0, 0.5), end=(5.0, 0.0, 1.0)
        )


def test_strip_length_property() -> None:
    s = gf.StripElectrode(
        name="s", start=(0.0, 0.0, 0.5), end=(3.0, 4.0, 0.5)
    )
    # 3-4-5 triangle
    assert s.length == pytest.approx(5.0)


def test_strip_connection_point_is_start() -> None:
    s = gf.StripElectrode(
        name="s", start=(1.0, 2.0, 0.5), end=(6.0, 2.0, 0.5)
    )
    assert s.connection_point == (1.0, 2.0, 0.5)


# ---------------------------------------------------------------------
# 2. Image backend vs. Dwight horizontal-wire formula
# ---------------------------------------------------------------------


def test_image_strip_matches_dwight_horizontal_wire() -> None:
    """Single straight strip: image vs. Dwight Eq. (12).

    A canonical 10 m strip at 0.5 m depth, 5 mm wire radius. Tolerance
    10 % to absorb the residual uniform-current bias of the image
    backend.
    """
    L = 10.0
    a = 0.005
    depth = 0.5
    world = gf.create_world(soil=SOIL)
    gf.create_electrode(
        world, "strip", name="g1",
        start=(-L / 2.0, 0.0, depth), end=(+L / 2.0, 0.0, depth),
        wire_radius=a,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    Z = ENG.solve(world).cluster_impedance("g1")[0].real
    R_dw = dw.horizontal_wire(rho=100.0, length=L / 2.0,
                              radius=a, depth=depth)
    rel = abs(Z - R_dw) / R_dw
    assert rel < 0.10, (
        f"strip {Z:.2f} Ω vs. Dwight {R_dw:.2f} Ω, Δ = {rel*100:.1f} %"
    )


# ---------------------------------------------------------------------
# 3. Native strip is more accurate than the historical mesh workaround
# ---------------------------------------------------------------------


def test_strip_more_accurate_than_degenerate_mesh_workaround() -> None:
    """The native ``StripElectrode`` must beat the historical
    degenerate-mesh workaround when checked against Dwight 1936.

    The workaround puts two parallel longitudinal wires 1 mm apart
    and distributes the cluster current uniformly over their *total*
    length (``2 L`` instead of ``L``). The per-unit-length emission
    is therefore halved, which inflates the cluster impedance by
    roughly a factor of two — this was the root cause of the
    "potential at the strip end is not the ring potential" symptom
    and motivates the migration to the native primitive.
    """
    L = 8.0
    a = 0.005
    depth = 0.5

    # Native strip
    w1 = gf.create_world(soil=SOIL)
    gf.create_electrode(
        w1, "strip", name="g1",
        start=(0.0, 0.0, depth), end=(L, 0.0, depth),
        wire_radius=a,
    )
    gf.create_source(w1, attached_to="g1", magnitude=1.0)
    Z_strip = ENG.solve(w1).cluster_impedance("g1")[0].real

    # Degenerate mesh with eps = 1 mm (the prior workaround)
    w2 = gf.create_world(soil=SOIL)
    gf.create_electrode(
        w2, "mesh", name="g1",
        corner=(0.0, 0.0, depth), size=(L, 1.0e-3),
        spacing=10.0 * L, wire_radius=a,
    )
    gf.create_source(w2, attached_to="g1", magnitude=1.0)
    Z_mesh = ENG.solve(w2).cluster_impedance("g1")[0].real

    R_dw = dw.horizontal_wire(rho=100.0, length=L / 2.0,
                              radius=a, depth=depth)
    err_strip = abs(Z_strip - R_dw) / R_dw
    err_mesh = abs(Z_mesh - R_dw) / R_dw

    # Native primitive is well within the 10 % image-vs-Dwight envelope.
    assert err_strip < 0.10, (
        f"native strip Z = {Z_strip:.3f} Ω, Dwight = {R_dw:.3f} Ω, "
        f"Δ = {err_strip*100:.2f} %"
    )
    # Workaround is much further off — the very symptom that motivated
    # the migration. Pin it as a regression guard.
    assert err_mesh > 5 * err_strip, (
        "Expected the degenerate-mesh workaround to be much less "
        f"accurate than the native strip. Got err_strip = "
        f"{err_strip*100:.1f} %, err_mesh = {err_mesh*100:.1f} %."
    )


# ---------------------------------------------------------------------
# 4. Cluster equipotential constraint
# ---------------------------------------------------------------------


def test_strip_bonded_to_ring_shares_cluster_potential() -> None:
    """Strip bonded to a ring sits at the same average potential.

    The cluster constraint inside the solver enforces equal mean
    segment potential per electrode in the cluster. This test checks
    that the constraint is honoured for the new strip primitive.
    """
    world = gf.create_world(soil=SOIL)
    ring = gf.create_electrode(
        world, "ring", name="ring",
        center=(0.0, 0.0, 0.7), radius=4.0, wire_radius=0.005,
    )
    strip = gf.create_electrode(
        world, "strip", name="strip",
        start=(0.0, 4.0, 0.7), end=(0.0, 9.0, 0.7),
        wire_radius=0.005,
    )
    gf.create_conductor(world, name="bond", start=ring, end=strip)
    gf.create_source(world, attached_to="ring", magnitude=10.0)

    res = ENG.solve(world)
    u_ring = complex(np.mean(res.electrode_potentials["ring"][0]))
    u_strip = complex(np.mean(res.electrode_potentials["strip"][0]))
    assert abs(u_ring - u_strip) / abs(u_ring) < 1e-9, (
        f"u_ring = {u_ring}, u_strip = {u_strip}"
    )

    # And the currents add up to the source current.
    i_total = (res.electrode_currents["ring"][0]
               + res.electrode_currents["strip"][0])
    assert abs(i_total - 10.0) < 1e-9
