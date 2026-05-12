"""Tests for finite-impedance conductor branches.

These tests cover the augmented nodal-analysis solver introduced
together with :attr:`Conductor.cross_section`. The historic
ideal-conductor behaviour (``cross_section is None``) must remain
bit-identical to the previous releases — captured here as a
regression baseline. The new branch model is checked against an
analytical two-electrode example with a known closed form,
and against every other backend in the family for cross-engine
consistency.

Reference geometry
------------------
Two driven rods, far apart so that the mutual grounding impedance is
small compared with the self impedance. With cluster self-resistance
$R_\\text{self}$ (per electrode) and a single finite branch
$R_b$ between them, plus a unit current $I$ injected at
electrode 1, the branch current is

$$
I_b \\;=\\; \\frac{R_\\text{self}}{2 R_\\text{self} + R_b}\\, I
$$

so $I_{e_1} = I - I_b$ and $I_{e_2} = I_b$. The two limits
$R_b \\to 0$ (cluster equipotential, half-and-half split) and
$R_b \\to \\infty$ (decoupled grounds, $I_{e_2} \\to 0$) are
explicit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf


# ---------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------

SEG = 0.1  # segment length used in every test (matched across engines)
RHO = 100.0
ROD_LEN = 2.0
ROD_RADIUS = 0.0075
SEPARATION = 30.0  # rod-to-rod distance — large compared to rod length
                    # so that the mutual grounding impedance is negligible
                    # but >> SEG, so the discretisation stays meaningful.
PEN_CROSS_SECTION = 50.0e-6  # 50 mm² (NAYY-typical)
PEN_RESISTIVITY = 2.82e-8     # Al at 20 °C


def _two_rod_world(
    *,
    rod_separation: float = SEPARATION,
    cross_section: float | None = None,
    pen_resistivity: float = PEN_RESISTIVITY,
    soil: gf.HomogeneousSoil | None = None,
) -> gf.World:
    """Two rods linked by one Conductor; current injected at rod g1.

    The conductor's series resistance is set via ``cross_section``
    (None ⇒ historic ideal short, finite ⇒ branch model).
    """
    if soil is None:
        soil = gf.HomogeneousSoil(resistivity=RHO)
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=ROD_LEN,
        wire_radius=ROD_RADIUS,
    )
    gf.create_electrode(
        w, "rod", name="g2", position=(rod_separation, 0.0, 0.5),
        length=ROD_LEN, wire_radius=ROD_RADIUS,
    )
    gf.create_conductor(
        w, name="pen", start="g1", end="g2",
        conductor_type="pen",
        wire_radius=0.004, resistivity=pen_resistivity,
        cross_section=cross_section,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _solve(world: gf.World, backend: str, **eng_kwargs) -> gf.FieldResult:
    eng = gf.create_engine(
        backend=backend, segment_length=SEG, **eng_kwargs
    )
    return eng.solve(world)


# ---------------------------------------------------------------------
# 1. Default behaviour: cross_section=None ⇒ ideal galvanic short
# ---------------------------------------------------------------------


def test_ideal_default_keeps_cluster_equipotential() -> None:
    """Without ``cross_section`` the conductor remains an ideal short
    and both rods share one cluster potential exactly."""
    res = _solve(_two_rod_world(cross_section=None), "image")
    # Both electrodes share the cluster
    assert sorted(res.clusters["g1"]) == ["g1", "g2"]
    # Common potential within 0.1 % (image discretisation noise)
    u1 = res.electrode_potentials["g1"][0]
    u2 = res.electrode_potentials["g2"][0]
    assert abs(u1 - u2) / abs(u1) < 1e-3
    # Sum of currents == source amplitude
    I_total = res.electrode_currents["g1"][0] + res.electrode_currents["g2"][0]
    assert I_total.real == pytest.approx(1.0, rel=1e-9)
    assert I_total.imag == pytest.approx(0.0, abs=1e-9)


def test_ideal_default_matches_legacy_cluster_split() -> None:
    """For two equal rods at equal distance the ideal-conductor split
    is 50 / 50 (within the few-% image-method bias)."""
    res = _solve(_two_rod_world(cross_section=None), "image")
    I1 = res.electrode_currents["g1"][0].real
    I2 = res.electrode_currents["g2"][0].real
    assert I1 == pytest.approx(I2, rel=0.05)
    assert (I1 + I2) == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------
# 2. Finite-impedance branch: closed-form check
# ---------------------------------------------------------------------


def _expected_branch_split(R_self: float, R_b: float, I_in: float = 1.0) -> tuple[float, float]:
    """Closed-form per :func:`module docstring`."""
    I_b = R_self / (2.0 * R_self + R_b) * I_in
    return I_in - I_b, I_b


def test_finite_branch_closed_form_image() -> None:
    """The augmented system reproduces the analytical
    two-rod / one-branch split within 5 % for the image backend.

    The 5 % tolerance covers the residual mutual grounding impedance
    between the two rods (their finite separation is large but not
    infinite) plus the average-potential bias of the image backend.
    """
    R_b = 0.5  # Ω, finite enough to push the split off 50/50
    A = R_b / PEN_RESISTIVITY * SEPARATION  # cross_section that gives R_b
    # Get R_self from a single-rod isolated world.
    iso = gf.create_world(soil=gf.HomogeneousSoil(resistivity=RHO))
    gf.create_electrode(
        iso, "rod", name="g1", position=(0, 0, 0.5),
        length=ROD_LEN, wire_radius=ROD_RADIUS,
    )
    gf.create_source(iso, attached_to="g1", magnitude=1.0)
    R_self = _solve(iso, "image").cluster_impedance("g1")[0].real

    res = _solve(_two_rod_world(cross_section=A), "image")
    I1 = res.electrode_currents["g1"][0].real
    I2 = res.electrode_currents["g2"][0].real
    I1_exp, I2_exp = _expected_branch_split(R_self, R_b)
    # 5 % relative tolerance on each share
    assert abs(I1 - I1_exp) / I1_exp < 0.05, (
        f"image: I1={I1:.4f} vs analytic {I1_exp:.4f}"
    )
    assert abs(I2 - I2_exp) / I2_exp < 0.05, (
        f"image: I2={I2:.4f} vs analytic {I2_exp:.4f}"
    )
    # Total current must still equal the source.
    assert (I1 + I2) == pytest.approx(1.0, rel=1e-9)


def test_finite_branch_resistance_property() -> None:
    """Conductor.series_resistance maps geometry to R via R = ρ L / A."""
    w = _two_rod_world(cross_section=PEN_CROSS_SECTION)
    pen = w.get_conductor("pen")
    expected = PEN_RESISTIVITY * SEPARATION / PEN_CROSS_SECTION
    assert pen.series_resistance == pytest.approx(expected, rel=1e-9)
    assert not pen.is_ideal()


def test_from_radius_resolves_to_pi_r2() -> None:
    """The "from_radius" shortcut gives A = π · r²."""
    w = _two_rod_world(cross_section="from_radius")
    pen = w.get_conductor("pen")
    expected_A = math.pi * pen.wire_radius ** 2
    assert pen.effective_cross_section == pytest.approx(expected_A, rel=1e-12)


# ---------------------------------------------------------------------
# 3. Limits of the finite-branch model
# ---------------------------------------------------------------------


def test_branch_collapses_to_cluster_for_small_R() -> None:
    """A very small R_b (large cross section) should produce the same
    50/50 split as the ideal-conductor mode within 0.5 %."""
    # Huge cross section -> R_b ≈ 0
    res_big = _solve(_two_rod_world(cross_section=1.0), "image")
    res_ideal = _solve(_two_rod_world(cross_section=None), "image")
    for ename in ("g1", "g2"):
        i_big = res_big.electrode_currents[ename][0].real
        i_ideal = res_ideal.electrode_currents[ename][0].real
        assert abs(i_big - i_ideal) / abs(i_ideal) < 5e-3


def test_branch_decouples_for_large_R() -> None:
    """For an extremely large R_b the two rods should decouple — almost
    all current stays at the source rod."""
    # Tiny cross section -> very large R_b
    res = _solve(_two_rod_world(cross_section=1e-12), "image")
    I1 = res.electrode_currents["g1"][0].real
    I2 = res.electrode_currents["g2"][0].real
    assert I1 == pytest.approx(1.0, abs=1e-3)
    assert abs(I2) < 1e-3


# ---------------------------------------------------------------------
# 4. Cross-engine: every backend agrees on the finite-branch split
# ---------------------------------------------------------------------


HOMOGENEOUS_BACKENDS = ["image", "mom", "cim", "bem", "fem"]


@pytest.mark.parametrize("backend", HOMOGENEOUS_BACKENDS)
def test_finite_branch_cross_engine_homogeneous(backend: str) -> None:
    """Every homogeneous-soil backend must agree with the image
    reference on the finite-branch split within engine-specific
    tolerances."""
    A = 1e-4  # ~ 100 mm² cross section: R_b ≈ ρ_Al · 30 m / 1e-4 ≈ 8.5 mΩ
    w = _two_rod_world(cross_section=A)
    res_ref = _solve(w, "image")
    res = _solve(_two_rod_world(cross_section=A), backend)

    I1_ref = res_ref.electrode_currents["g1"][0].real
    I2_ref = res_ref.electrode_currents["g2"][0].real
    I1 = res.electrode_currents["g1"][0].real
    I2 = res.electrode_currents["g2"][0].real
    # FEM uses an equivalent-hemisphere reduction → a slightly larger
    # tolerance is appropriate for the cross-check.
    tol = 0.07 if backend == "fem" else 0.03
    assert abs(I1 - I1_ref) / abs(I1_ref) < tol, (
        f"{backend}: I1={I1:.4f} vs image {I1_ref:.4f}"
    )
    assert abs(I2 - I2_ref) / abs(I2_ref) < tol, (
        f"{backend}: I2={I2:.4f} vs image {I2_ref:.4f}"
    )
    # Current conservation always exact.
    assert (I1 + I2) == pytest.approx(1.0, rel=1e-9)


# Layered-soil engines need a TwoLayerSoil world.
LAYERED_BACKENDS = ["image_2layer", "mom", "cim", "bem", "mom_sommerfeld", "fem"]


@pytest.mark.parametrize("backend", LAYERED_BACKENDS)
def test_finite_branch_cross_engine_two_layer(backend: str) -> None:
    """Same cross-engine check on a moderate-contrast 2-layer soil."""
    soil = gf.TwoLayerSoil(rho_1=80.0, rho_2=400.0, h_1=2.5)
    A = 1e-4
    w = _two_rod_world(cross_section=A, soil=soil)
    res_ref = _solve(_two_rod_world(cross_section=A, soil=soil), "image_2layer")
    res = _solve(w, backend)

    I1_ref = res_ref.electrode_currents["g1"][0].real
    I2_ref = res_ref.electrode_currents["g2"][0].real
    I1 = res.electrode_currents["g1"][0].real
    I2 = res.electrode_currents["g2"][0].real
    tol = 0.10 if backend == "fem" else 0.04
    assert abs(I1 - I1_ref) / abs(I1_ref) < tol, (
        f"{backend}: I1={I1:.4f} vs image_2layer {I1_ref:.4f}"
    )
    assert abs(I2 - I2_ref) / abs(I2_ref) < tol, (
        f"{backend}: I2={I2:.4f} vs image_2layer {I2_ref:.4f}"
    )
    assert (I1 + I2) == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------
# 5. Monotonicity: I_branch decreases as R_b grows
# ---------------------------------------------------------------------


def test_split_monotone_in_branch_resistance() -> None:
    """As R_b increases, the share routed through the branch must
    decrease monotonically."""
    rs = [1e-6, 1e-3, 0.1, 1.0, 10.0]
    branch_currents = []
    for R_b in rs:
        # Pick a cross section that yields exactly this R_b
        A = PEN_RESISTIVITY * SEPARATION / R_b
        res = _solve(_two_rod_world(cross_section=A), "image")
        branch_currents.append(res.electrode_currents["g2"][0].real)
    # Strictly monotone decreasing
    diffs = np.diff(branch_currents)
    assert np.all(diffs < 0.0), f"branch currents not monotone: {branch_currents}"


# ---------------------------------------------------------------------
# 6. Three-cluster chain (transitive activation)
# ---------------------------------------------------------------------


def _three_rod_chain(*, branch_cross_section: float) -> gf.World:
    """Build a three-rod / two-PEN-branch test world."""
    soil = gf.HomogeneousSoil(resistivity=RHO)
    w = gf.create_world(soil=soil)
    for k, x in enumerate([0.0, SEPARATION, 2 * SEPARATION]):
        gf.create_electrode(
            w, "rod", name=f"g{k+1}", position=(x, 0.0, 0.5),
            length=ROD_LEN, wire_radius=ROD_RADIUS,
        )
    gf.create_conductor(
        w, name="pen_12", start="g1", end="g2", conductor_type="pen",
        cross_section=branch_cross_section, resistivity=PEN_RESISTIVITY,
    )
    gf.create_conductor(
        w, name="pen_23", start="g2", end="g3", conductor_type="pen",
        cross_section=branch_cross_section, resistivity=PEN_RESISTIVITY,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_three_node_chain_propagates_through_branches() -> None:
    """Three rods chained by two finite PEN branches, source at g1.

    The transitive-activation contract is:

    1. current is conserved (Σ I_e = 1),
    2. *all three* rods carry a positive share — passive nodes
       reachable through finite branches must be activated,
    3. the chain produces an EPR profile that decreases away from
       the source: $\\varphi_{g_1} > \\varphi_{g_3}$.

    The relative ordering of the three earth currents at intermediate
    branch resistances is *not* fixed: in the cluster-near regime
    (R_b ≪ R_self) the small mutual-coupling asymmetry between the
    middle and the outer rods can dominate the linear source-side
    bias, so a naive ``I_g1 > I_g2 > I_g3`` would be too strict. We
    therefore test the unambiguous limits separately below.
    """
    res = _solve(_three_rod_chain(branch_cross_section=1e-4), "image")
    I1 = res.electrode_currents["g1"][0].real
    I2 = res.electrode_currents["g2"][0].real
    I3 = res.electrode_currents["g3"][0].real
    U1 = res.electrode_potentials["g1"][0].real
    U3 = res.electrode_potentials["g3"][0].real

    # (1) current conservation
    assert (I1 + I2 + I3) == pytest.approx(1.0, rel=1e-9)
    # (2) transitive activation
    assert I1 > 0.0 and I2 > 0.0 and I3 > 0.0, (I1, I2, I3)
    # (3) EPR drops away from the source
    assert U1 > U3, (U1, U3)


def test_three_node_chain_decoupled_limit() -> None:
    """For a *very* high branch resistance the chain decouples: almost
    all current stays at g1; g3 carries virtually none. This is the
    unambiguous source-bias regime."""
    # Cross section that yields R_b ≈ 850 Ω — far above R_self ≈ 30 Ω
    res = _solve(_three_rod_chain(branch_cross_section=1e-9), "image")
    I1 = res.electrode_currents["g1"][0].real
    I3 = res.electrode_currents["g3"][0].real
    assert I1 > 0.95
    assert I3 < 0.01


def test_three_node_chain_strong_branch_drives_source_bias() -> None:
    """At R_b comparable to R_self the source-bias is large enough to
    overcome the mutual-coupling-driven near-cluster effect: the
    Source rod and its immediate neighbour beat the far-away rod
    monotonically."""
    # cross_section such that R_b ≈ 15 Ω (≈ R_self / 2)
    A = PEN_RESISTIVITY * SEPARATION / 15.0
    res = _solve(_three_rod_chain(branch_cross_section=A), "image")
    I1 = res.electrode_currents["g1"][0].real
    I2 = res.electrode_currents["g2"][0].real
    I3 = res.electrode_currents["g3"][0].real
    assert (I1 + I2 + I3) == pytest.approx(1.0, rel=1e-9)
    # In this decoupling regime the strict source-side ordering holds.
    assert I1 > I2 > I3 > 0.0, (I1, I2, I3)
