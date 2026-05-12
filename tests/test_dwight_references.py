"""Comparison tests image-backend vs. Dwight 1936.

Source for every reference value: Dwight, H. B., *Calculation of
Resistances to Ground*, AIEE Transactions, December 1936,
pp. 1319-1328 (Table I and the worked examples on pp. 1321-1325).

Layout
------
1. **`test_dwight_module_against_paper_examples`** — verifies the pure
   reference module against the numbers published in the paper. These
   tests must always pass: they check our implementation of the
   formulas, independent of the solver.

2. **`test_image_backend_against_dwight`** — for every geometry that
   ``groundfield`` currently supports, compare the numerical result
   from the image backend with the corresponding Dwight formula.
   Tolerance: 10 % for the image vs. Dwight comparison. Dwight
   himself reports "several per cent" accuracy for the star formulas;
   the image backend additionally suffers ~5 % from the point-source
   approximation. The result must still be consistent with Dwight.

How to extend
-------------
Whenever a new geometry is added to the data model, add another entry
to the parametric test list — every geometry should have *at least*
one comparison test against Dwight.
"""

from __future__ import annotations

import math

import pytest

import groundfield as gf
from groundfield.references import dwight1936 as dw


# ---------------------------------------------------------------------
# 1. Pure reference module vs. paper examples
# ---------------------------------------------------------------------


def test_dwight_horizontal_wire_paper_example() -> None:
    """Dwight 1936, p. 1323: horizontal wire 200 ft, 10 ft deep,
    No. 4/0, rho = 200,000 Ω·cm³ → R = 57.6 Ω."""
    R = dw.horizontal_wire(rho=2000.0, length=30.48,
                           radius=0.005842, depth=3.048)
    assert R == pytest.approx(57.6, abs=0.2)


def test_dwight_right_angle_paper_example() -> None:
    """Dwight 1936, p. 1323: right-angle wire 2 × 100 ft → R = 59.4 Ω."""
    R = dw.right_angle_wire(rho=2000.0, arm_length=30.48,
                            radius=0.005842, depth=3.048)
    assert R == pytest.approx(59.4, abs=0.2)


@pytest.mark.parametrize(
    "n_arms, expected",
    [(3, 43.9), (4, 37.3), (6, 31.1), (8, 28.2)],
)
def test_dwight_star_paper_examples(n_arms: int, expected: float) -> None:
    """Dwight 1936, pp. 1323-1324: 3/4/6/8-point star, same dimensions
    as the right-angle example."""
    R = dw.n_point_star(rho=2000.0, arm_length=30.48,
                        radius=0.005842, depth=3.048, n_arms=n_arms)
    assert R == pytest.approx(expected, abs=0.2)


def test_dwight_hemisphere_basic() -> None:
    """Hemisphere, R = ρ/(2πA). Dwight 1936, p. 1320."""
    R = dw.hemisphere(rho=100.0, radius=0.5)
    assert R == pytest.approx(100.0 / (2 * math.pi * 0.5))


def test_dwight_two_rods_far_self_term_equals_single_rod() -> None:
    """For very wide spacing (s → ∞): R → 0.5 · R_single (parallel
    combination of two completely decoupled electrodes, no mutual
    term)."""
    L, a = 1.5, 0.005
    R_single = dw.rod(rho=100.0, length=L, radius=a)
    R_pair = dw.two_rods_far(rho=100.0, length=L, radius=a, spacing=10000.0)
    assert R_pair == pytest.approx(0.5 * R_single, rel=1e-3)


# ---------------------------------------------------------------------
# 2. Image backend vs. Dwight
# ---------------------------------------------------------------------


SOIL = gf.HomogeneousSoil(resistivity=100.0)
ENG = gf.create_engine(backend="image", segment_length=0.05)


@pytest.mark.parametrize(
    "L, wire_radius",
    [(1.5, 0.005), (3.0, 0.005), (1.5, 0.01)],
)
def test_image_rod_vs_dwight(L: float, wire_radius: float) -> None:
    """Driven rod: image backend must agree with Dwight to within 10 %."""
    world = gf.create_world(soil=SOIL)
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.0), length=L, wire_radius=wire_radius)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    Z = ENG.solve(world).cluster_impedance("g1")[0].real
    R_dw = dw.rod(rho=100.0, length=L, radius=wire_radius)
    rel = abs(Z - R_dw) / R_dw
    assert rel < 0.10, (
        f"image {Z:.2f} Ω vs. Dwight {R_dw:.2f} Ω, Δ = {rel*100:.1f} %"
    )


@pytest.mark.parametrize(
    "spacing",
    [3.0, 5.0, 10.0],
)
def test_image_two_rods_vs_dwight(spacing: float) -> None:
    """Two parallel rods (galvanically connected, equal current per rod):
    cluster impedance of both rods in parallel must match Dwight's
    `two_rods_far`."""
    L, a = 1.5, 0.005
    world = gf.create_world(soil=SOIL)
    g1 = gf.create_electrode(world, "rod", name="g1",
                             position=(0.0, 0.0, 0.0), length=L,
                             wire_radius=a)
    g2 = gf.create_electrode(world, "rod", name="g2",
                             position=(spacing, 0.0, 0.0), length=L,
                             wire_radius=a)
    # Connect with a conductor → cluster impedance is the parallel
    # combination of both rods including mutual coupling.
    gf.create_conductor(world, name="l1", start=g1, end=g2)
    gf.create_source(world, attached_to=g1, magnitude=1.0)
    Z = ENG.solve(world).cluster_impedance("g1")[0].real

    R_dw = dw.two_rods_far(rho=100.0, length=L, radius=a, spacing=spacing)
    rel = abs(Z - R_dw) / R_dw
    assert rel < 0.10, (
        f"spacing={spacing}: image {Z:.2f} Ω vs. Dwight {R_dw:.2f} Ω, "
        f"Δ = {rel*100:.1f} %"
    )


def test_image_buried_ring_vs_dwight() -> None:
    """Buried ring electrode against Dwight Eq. (29).

    Dwight's preconditions: d ≪ s ≪ D. We pick D = 4 m, d = 10 mm,
    depth = 0.5 m → s = 1.0 m.
    """
    D = 4.0
    radius = D / 2.0
    wire_radius = 0.005
    depth = 0.5

    world = gf.create_world(soil=SOIL)
    gf.create_electrode(
        world, "ring", name="g1",
        center=(0.0, 0.0, depth), radius=radius, wire_radius=wire_radius,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    Z = ENG.solve(world).cluster_impedance("g1")[0].real

    R_dw = dw.buried_ring(
        rho=100.0, ring_diameter=D, wire_diameter=2 * wire_radius, depth=depth
    )
    rel = abs(Z - R_dw) / R_dw
    assert rel < 0.15, (
        f"ring electrode: image {Z:.2f} Ω vs. Dwight {R_dw:.2f} Ω, "
        f"Δ = {rel*100:.1f} %"
    )


@pytest.mark.parametrize(
    "total_length, wire_radius, depth",
    [(10.0, 0.005, 0.5), (20.0, 0.005, 0.5), (10.0, 0.01, 1.0)],
)
def test_image_horizontal_wire_vs_dwight(
    total_length: float, wire_radius: float, depth: float
) -> None:
    """Buried straight horizontal wire (StripElectrode) vs. Dwight Eq. (12).

    Dwight's ``horizontal_wire`` formula uses the half-length ``L``
    (total length is ``2 L``). The image backend must agree to within
    10 % across a few aspect ratios.
    """
    world = gf.create_world(soil=SOIL)
    gf.create_electrode(
        world, "strip", name="g1",
        start=(-total_length / 2.0, 0.0, depth),
        end=(+total_length / 2.0, 0.0, depth),
        wire_radius=wire_radius,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    Z = ENG.solve(world).cluster_impedance("g1")[0].real
    R_dw = dw.horizontal_wire(
        rho=100.0, length=total_length / 2.0,
        radius=wire_radius, depth=depth,
    )
    rel = abs(Z - R_dw) / R_dw
    assert rel < 0.10, (
        f"strip {Z:.2f} Ω vs. Dwight {R_dw:.2f} Ω, Δ = {rel*100:.1f} %"
    )


def test_image_n_point_star_vs_dwight_skip() -> None:
    """Star electrode (3/4/6/8) against Dwight Eqs. (23)–(26).

    Reserved: the ``StarElectrode`` geometry is not yet in the data
    model. The test will be activated as soon as that geometry exists.
    """
    pytest.skip("StarElectrode is not implemented yet.")


# ---------------------------------------------------------------------
# 3. Consistency sanity checks
# ---------------------------------------------------------------------


def test_dwight_module_units_consistent() -> None:
    """Doubling ρ doubles R (every formula is linear in ρ)."""
    args = dict(rho=1.0, length=1.5, radius=0.005)
    R1 = dw.rod(**args)
    args["rho"] = 2.0
    R2 = dw.rod(**args)
    assert R2 == pytest.approx(2.0 * R1)


def test_dwight_rod_decreases_with_length() -> None:
    """Longer rod → lower grounding resistance."""
    R_short = dw.rod(rho=100, length=1.5, radius=0.005)
    R_long = dw.rod(rho=100, length=3.0, radius=0.005)
    assert R_long < R_short
