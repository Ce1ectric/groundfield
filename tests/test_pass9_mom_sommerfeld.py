"""Review-pass-9 regression tests for ``mom_sommerfeld`` (F04, F40).

Physical background
-------------------
The top-layer Sommerfeld kernel of a point current source at depth
$z_s$, observed at $(s, z)$ inside the upper layer, is

$$
G = \\int_0^{\\infty}
\\frac{e^{-\\lambda d_z} + \\Gamma_1 e^{-\\lambda (2 h_1 - d_z)}
     + e^{-\\lambda p}  + \\Gamma_1 e^{-\\lambda (2 h_1 - p)}}
     {1 - \\Gamma_1(\\lambda) e^{-2 \\lambda h_1}}
J_0(\\lambda s)\\, d\\lambda ,
\\qquad d_z = |z - z_s|,\\; p = z + z_s .
$$

Two properties of that integral are exploited here as *exact*
references, independent of the implementation under test:

1. **Lipschitz' integral.** For $\\Gamma_1 \\equiv 0$ (a two-layer
   stack declared with $\\rho_1 = \\rho_2$, so
   $K_1 = 0$) the integral collapses to
   $1/\\sqrt{s^2 + d_z^2} + 1/\\sqrt{s^2 + p^2}$ — the
   homogeneous point-source pair. This is the strongest available
   test: any deviation is pure quadrature error.
2. **Tagg / Sunde image series.** For $n = 2$,
   $\\Gamma_1 \\equiv K_1$ is constant in $\\lambda$, the
   multiplier expands geometrically and every term integrates by
   Lipschitz, giving the exact closed form
   $G = 1/r(d_z) + 1/r(p) + \\sum_{m \\ge 1} K_1^m [\\,
   1/r(2mh_1 - d_z) + 1/r(2mh_1 + d_z)
   + 1/r(2mh_1 - p) + 1/r(2mh_1 + p)\\,]$ with
   $r(d) = \\sqrt{s^2 + d^2}$. It is summed independently in
   :func:`_tagg_series` below.

F04 (pre-0.15.0): the kernel was a bare ``scipy.integrate.quad`` on a
finite $[0, \\lambda_{\\max}]$. The direct term
$e^{-\\lambda d_z} J_0(\\lambda s)$ does not decay for
$d_z = 0$ (two segments at the same depth), so the integral is only
conditionally convergent and the truncation left a sign-alternating
error: the kernel returned **-0.012551** at
$s = 50, z = z_s = 0.8, h_1 = 1$ where the exact value is
**+0.039990**, and a 25 m ring came out 5.1 % low.

F40 (pre-0.15.0): $\\lambda_{\\max}$ was derived from
$\\min(h_1, s + z + z_s)$ instead of the decay scale of the
integrand, so the largest off-diagonal entries (two close segments
deep in the layer) were truncated by up to 45 %, and got *worse* under
mesh refinement.

F04 second pass (the 0.15.0 audit of the F04 fix)
-------------------------------------------------
The analytic split above is exact, but the *panel budget* that
resolves the $J_0$ oscillation reintroduced the very same failure
mode in a narrow corner:

1. ``_MAX_OSC_PANELS`` truncated the half-period panel *count* while
   the step was $\\pi / (s \\cdot \\texttt{refine})$, so once the
   budget bound, the resolved $\\lambda$-range **shrank** with every
   refinement. Because :func:`_reflected_integral` returns the *last*
   rung of the ladder, more refinement produced a worse answer.
2. Exhausting the ladder — i.e. detecting exactly this — only emitted
   a ``logger.debug`` record. Zero Python-level warnings.

Measured on ``LayerStack([100, 1000, 50], h=[2, h_2])`` at
$z = z_s$, checked against the far-field asymptote
$G \\to (2/s)\\,\\rho_3/\\rho_1$ (a *pure* reference: at
$s \\gg$ every layer depth the stack behaves like a homogeneous
half-space of resistivity $\\rho_n$):

=========  ======  =======  ==================  ===========
$h_2$ [m]  z [m]   s [m]    0.14.x              exact
=========  ======  =======  ==================  ===========
0.01       2.0     2000     **-9.4749e-05**     +5.000e-04
0.001      2.0     200      **-9.4761e-04**     +5.000e-03
0.01       2.0     500      1.92531e-03         2.000e-03
0.01       1.99    1000     9.58002e-04         1.000e-03
=========  ======  =======  ==================  ===========

The first two rows are sign-flipped Green's functions (-119 %) in the
package's designated *reference* engine, returned silently.

The third gap of the same audit: ``Engine.solve`` called
``solve_mom_sommerfeld(world, self)`` with no keywords, so
``lambda_max_factor`` / ``epsabs`` / ``epsrel`` were unreachable from
the public API and a user in that corner had nothing to tighten.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver._layered import LayerStack
from groundfield.solver.mom_sommerfeld import (
    SommerfeldConvergenceWarning,
    sommerfeld_kernel_value,
)

# Coarse mesh — the reference engine is the slow one.
SEG = 5.0
RING_RADIUS = 25.0
RING_DEPTH = 0.8


# ---------------------------------------------------------------------
# Independent analytic references
# ---------------------------------------------------------------------


def _homogeneous_pair(s: float, z: float, z_s: float) -> float:
    """Exact $\\Gamma_1 = 0$ kernel: source plus air mirror."""
    return 1.0 / float(np.hypot(s, z - z_s)) + 1.0 / float(np.hypot(s, z + z_s))


def _tagg_series(
    rho_1: float, rho_2: float, h_1: float,
    s: float, z: float, z_s: float, m_max: int = 20_000,
) -> float:
    """Exact two-layer kernel as the Tagg / Sunde image series.

    Independent of the module under test: it uses only the geometric
    expansion of the multiple-reflection multiplier and Lipschitz'
    integral, summed until the term drops below 1e-15 relative.
    """
    k_1 = (rho_2 - rho_1) / (rho_2 + rho_1)
    d_z = abs(z - z_s)
    p = z + z_s
    total = 1.0 / np.hypot(s, d_z) + 1.0 / np.hypot(s, p)
    for m in range(1, m_max + 1):
        two_mh = 2.0 * m * h_1
        term = k_1 ** m * (
            1.0 / np.hypot(s, two_mh - d_z)
            + 1.0 / np.hypot(s, two_mh + d_z)
            + 1.0 / np.hypot(s, two_mh - p)
            + 1.0 / np.hypot(s, two_mh + p)
        )
        total += term
        if abs(term) < 1e-15 * abs(total):
            break
    return float(total)


def _tagg_reflected_series(
    rho_1: float, rho_2: float, h_1: float,
    s: float, z: float, z_s: float, m_max: int = 200_000,
) -> float:
    """Exact *reflected* part only ($m \\ge 1$ image families).

    This is the layered correction that the $n \\ge 3$ diagonal of
    the reaction matrix adds on top of the homogeneous line
    self-potential; unlike the full kernel it stays finite at
    $s = 0,\\; z = z_s$.
    """
    k_1 = (rho_2 - rho_1) / (rho_2 + rho_1)
    d_z = abs(z - z_s)
    p = z + z_s
    total = 0.0
    for m in range(1, m_max + 1):
        two_mh = 2.0 * m * h_1
        term = k_1 ** m * (
            1.0 / np.hypot(s, two_mh - d_z)
            + 1.0 / np.hypot(s, two_mh + d_z)
            + 1.0 / np.hypot(s, two_mh - p)
            + 1.0 / np.hypot(s, two_mh + p)
        )
        total += term
        if abs(term) < 1e-16 * abs(total):
            break
    return float(total)


def _ring_world(soil) -> gf.World:
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world, "ring", name="g1",
        center=(0.0, 0.0, RING_DEPTH), radius=RING_RADIUS,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    return world


# ---------------------------------------------------------------------
# F04 — the conditionally convergent J0 tail
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("h_1", "s", "z", "z_s"),
    [
        # The finding's headline case: -0.012551 instead of +0.039990.
        (1.0, 50.0, 0.8, 0.8),
        (1.0, 100.0, 0.5, 0.5),
        (1.0, 0.1, 0.05, 0.95),
        (5.0, 0.5, 0.8, 0.8),
        (5.0, 2.0, 0.8, 0.8),
        (5.0, 5.0, 0.8, 0.8),
        (5.0, 20.0, 0.8, 0.8),
        (2.0, 30.0, 1.9, 1.999),   # both points at the interface
        (2.0, 0.02, 1.0, 1.5),
    ],
)
def test_degenerate_stack_reproduces_lipschitz_pair(
    h_1: float, s: float, z: float, z_s: float,
) -> None:
    """F04: for $\\rho_1 = \\rho_2$ the kernel is the exact pair.

    $K_1 = 0 \\Rightarrow \\Gamma_1 \\equiv 0$, so the Sommerfeld
    integral *is* $1/r + 1/r_{\\text{img}}$. Any deviation is
    quadrature error of the backend and nothing else.
    """
    stack = LayerStack(rhos=np.array([100.0, 100.0]), h=np.array([h_1]))
    got = sommerfeld_kernel_value(stack, s=s, z=z, z_s=z_s)
    assert got == pytest.approx(_homogeneous_pair(s, z, z_s), rel=1e-10)


def test_degenerate_stack_kernel_is_positive() -> None:
    """F04: the Green's function may never come out negative.

    The pre-0.15.0 truncation of the alternating $J_0$ tail flipped
    the sign at large radial distance (-0.012551 at s = 50 m).
    """
    stack = LayerStack(rhos=np.array([100.0, 100.0]), h=np.array([1.0]))
    for s in (10.0, 20.0, 50.0, 80.0, 120.0):
        assert sommerfeld_kernel_value(stack, s=s, z=0.8, z_s=0.8) > 0.0


@pytest.mark.parametrize("rho_2", [20.0, 500.0, 1000.0])
@pytest.mark.parametrize("h_1", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("s", [0.05, 0.5, 2.0, 20.0, 50.0])
def test_two_layer_kernel_matches_tagg_series(
    rho_2: float, h_1: float, s: float,
) -> None:
    """F04: real contrast — kernel equals the exact image series."""
    stack = LayerStack(rhos=np.array([100.0, rho_2]), h=np.array([h_1]))
    z = z_s = 0.4 * h_1
    got = sommerfeld_kernel_value(stack, s=s, z=z, z_s=z_s)
    expected = _tagg_series(100.0, rho_2, h_1, s, z, z_s)
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("rho_2", [1.0, 1.0e4])
@pytest.mark.parametrize("s", [0.5, 5.0, 50.0])
def test_two_layer_kernel_extreme_contrast(rho_2: float, s: float) -> None:
    """F04: hard contrasts $|K_1| \\to 1$ stay accurate.

    $\\rho_2 / \\rho_1 = 10^{\\pm 2}$ gives
    $K_1 = \\pm 0.9802$, i.e. the multiple-reflection multiplier
    $1 / (1 - \\Gamma_1 e^{-2\\lambda h_1})$ is near-singular at
    $\\lambda \\to 0$ — the regime the reference engine exists for.
    """
    stack = LayerStack(rhos=np.array([100.0, rho_2]), h=np.array([2.0]))
    got = sommerfeld_kernel_value(stack, s=s, z=0.8, z_s=0.8)
    expected = _tagg_series(100.0, rho_2, 2.0, s, 0.8, 0.8, m_max=4_000_000)
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("depth_fraction", [0.5, 0.9, 0.999, 1.0])
def test_two_layer_kernel_at_the_interface(depth_fraction: float) -> None:
    """F04 corner: $z + z_s \\to 2 h_1$ (both points at the interface).

    There the *remainder* of the analytic split loses its decay, which
    is why the leading interface reflection is extracted analytically
    as well ($\\Gamma_1(\\infty) = K_1$).
    """
    h_1 = 2.0
    stack = LayerStack(rhos=np.array([100.0, 700.0]), h=np.array([h_1]))
    z = z_s = depth_fraction * h_1
    for s in (0.5, 5.0, 50.0):
        got = sommerfeld_kernel_value(stack, s=s, z=z, z_s=z_s)
        expected = _tagg_series(100.0, 700.0, h_1, s, z, z_s)
        assert got == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------
# F40 — lambda_max must follow the decay scale, not the geometry span
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("z", "z_s"),
    [(3.0, 3.02), (4.0, 4.02), (3.0, 3.05), (3.0, 3.002), (4.99, 4.995)],
)
def test_close_deep_pair_not_truncated(z: float, z_s: float) -> None:
    """F40: small $|z - z_s|$ with large $z + z_s$.

    These are the largest off-diagonal reaction-matrix entries. The old
    bound $\\lambda_{\\max} = 200 / \\min(h_1, s + z + z_s)$ dropped
    up to 45 % of them (27.6997 instead of 50.1661).
    """
    stack = LayerStack(rhos=np.array([100.0, 100.0]), h=np.array([5.0]))
    got = sommerfeld_kernel_value(stack, s=0.0, z=z, z_s=z_s)
    assert got == pytest.approx(_homogeneous_pair(0.0, z, z_s), rel=1e-10)


@pytest.mark.parametrize("d_z", [0.2, 0.05, 0.02, 0.005, 0.002])
def test_refinement_does_not_degrade_accuracy(d_z: float) -> None:
    """F40: the error must not grow when the mesh is refined.

    Pre-0.15.0 the relative error went -0.0 % / -13.4 % / -44.8 % /
    -81.8 % / -92.3 % along this sequence, i.e. the *reference* engine
    became less accurate the finer the discretisation.
    """
    stack = LayerStack(rhos=np.array([100.0, 100.0]), h=np.array([5.0]))
    z = 3.0
    got = sommerfeld_kernel_value(stack, s=0.0, z=z, z_s=z + d_z)
    assert got == pytest.approx(_homogeneous_pair(0.0, z, z + d_z), rel=1e-10)


def test_lambda_max_factor_is_stable() -> None:
    """F40 / F14: raising the accuracy knob may not move the answer.

    Pre-0.15.0 this sweep gave 1.8932 / 1.9037 / 1.9153 / 1.9379 /
    1.9506 / **0.8249** — non-monotone and finally 57 % low, because
    ``quad(..., limit=400)`` cannot resolve the added Bessel
    half-waves. The exact value is 1.9424598179.
    """
    stack = LayerStack(rhos=np.array([100.0, 500.0]), h=np.array([2.0]))
    exact = _tagg_series(100.0, 500.0, 2.0, 2.0, 0.8, 0.8)
    values = [
        sommerfeld_kernel_value(
            stack, s=2.0, z=0.8, z_s=0.8, lambda_max_factor=factor,
        )
        for factor in (50.0, 100.0, 200.0, 400.0, 2000.0, 20000.0)
    ]
    for value in values:
        assert value > 0.0
        assert value == pytest.approx(exact, rel=1e-9)
    assert max(values) - min(values) < 1e-9 * abs(exact)


@pytest.mark.parametrize(("z", "z_s"), [(0.8, 0.8), (0.8, 1.2), (1.99, 1.99)])
@pytest.mark.parametrize("s", [0.5, 5.0, 50.0])
def test_n3_kernel_reduces_to_two_layer_series(
    s: float, z: float, z_s: float,
) -> None:
    """F04 on the $n \\ge 3$ path: $\\rho_2 = \\rho_3$.

    A three-layer stack whose two lower layers have equal resistivity
    *is* the two-layer soil, so the recursive
    $\\Gamma_1(\\lambda)$ branch must reproduce the same exact image
    series. This is the only exact reference available for
    $n \\ge 3$.
    """
    stack = LayerStack(
        rhos=np.array([100.0, 1000.0, 1000.0]), h=np.array([2.0, 5.0]),
    )
    got = sommerfeld_kernel_value(stack, s=s, z=z, z_s=z_s)
    expected = _tagg_series(100.0, 1000.0, 2.0, s, z, z_s)
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("z", [0.8, 1.0, 1.9, 1.98, 1.995, 1.999])
def test_n3_diagonal_reflected_integral_near_interface(z: float) -> None:
    """F40, second occurrence: the $n \\ge 3$ diagonal correction.

    The reflection-only integrand at $s = 0,\\; z = z_s$ decays on
    the scale $2 (h_1 - z)$, but the pre-0.15.0 bound was
    $\\lambda_{\\max} = 200 / \\min(h_1, z)$, which discards the
    tail as soon as a segment approaches the interface. Measured on
    ``LayerStack([100, 1000, 1000], h=[2, 5])`` against the exact
    reflected series:

    ==========  ==========  ===================
    ``z`` [m]   exact       v0.14.1 quadrature
    ==========  ==========  ===================
    1.90         5.556553    5.556553 (-0.00 %)
    1.98        21.925159   21.565356 (-1.64 %)
    1.995       83.289807   53.265922 (-36.1 %)
    1.999      410.562808   75.661009 (-81.6 %)
    ==========  ==========  ===================
    """
    from groundfield.solver.mom_sommerfeld import _reflected_integral

    stack = LayerStack(
        rhos=np.array([100.0, 1000.0, 1000.0]), h=np.array([2.0, 5.0]),
    )
    got = _reflected_integral(
        stack, 0.0, 0.0, 2.0 * z,
        lambda_max_factor=200.0, epsabs=1e-9, epsrel=1e-7,
        reference=1.0 / (2.0 * z),
    )
    expected = _tagg_reflected_series(100.0, 1000.0, 2.0, 0.0, z, z)
    assert got == pytest.approx(expected, rel=1e-9)


def test_cross_layer_misuse_raises_for_n3() -> None:
    """The top-layer form is rejected instead of silently diverging."""
    stack = LayerStack(
        rhos=np.array([100.0, 500.0, 50.0]), h=np.array([2.0, 3.0]),
    )
    with pytest.raises(ValueError, match="z_s"):
        sommerfeld_kernel_value(stack, s=1.0, z=3.5, z_s=3.5)


# ---------------------------------------------------------------------
# Propagation into a solve
# ---------------------------------------------------------------------


def test_ring_impedance_degenerate_matches_image_and_mom() -> None:
    """F04: a declared 2-layer soil with $\\rho_1 = \\rho_2$.

    Physically homogeneous, so ``image`` and ``mom`` are exact for this
    world. Pre-0.15.0 ``mom_sommerfeld`` returned 1.495211 Ω against
    1.575051 Ω (-5.07 %) and needed ~150 s.
    """
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=1.0)
    z_image = gf.create_engine(
        backend="image", segment_length=SEG,
    ).solve(_ring_world(soil)).cluster_impedance("g1")[0]
    z_mom = gf.create_engine(
        backend="mom", segment_length=SEG,
    ).solve(_ring_world(soil)).cluster_impedance("g1")[0]
    z_som = gf.create_engine(
        backend="mom_sommerfeld", segment_length=SEG,
    ).solve(_ring_world(soil)).cluster_impedance("g1")[0]
    assert z_som.real == pytest.approx(z_image.real, rel=1e-9)
    assert z_som.real == pytest.approx(z_mom.real, rel=1e-9)


def test_ring_impedance_contrast_matches_image_2layer() -> None:
    """F04: real contrast — the reference engine must agree with the
    closed-form series it is supposed to validate.

    Pre-0.15.0: 4.072423 Ω vs 4.074735 Ω (-5.7e-4 relative) in 121 s;
    after the fix the two agree to ~4e-8 in ~1 s.
    """
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
    z_ref = gf.create_engine(
        backend="image_2layer", segment_length=SEG,
    ).solve(_ring_world(soil)).cluster_impedance("g1")[0]
    z_som = gf.create_engine(
        backend="mom_sommerfeld", segment_length=SEG,
    ).solve(_ring_world(soil)).cluster_impedance("g1")[0]
    assert z_som.real == pytest.approx(z_ref.real, rel=1e-4)


def test_multilayer_reduces_to_two_layer_solve() -> None:
    """F40 (n ≥ 3 branch): a 3-layer stack with $\\rho_2 = \\rho_3$.

    Such a stack *is* the two-layer soil $(\\rho_1, \\rho_2, h_1)$,
    so the $n \\ge 3$ code path — the $\\Gamma_1(\\lambda)$
    recursion in the off-diagonals plus the reflection-only diagonal
    correction, whose $\\lambda_{\\max}$ ignored the
    $2 (h_1 - z_i)$ decay scale before 0.15.0 — must reproduce the
    $n = 2$ result.

    Note: with the rod tip 0.2 m clear of the interface this
    consistency check also held before 0.15.0 (rel 4.7e-10); it guards
    the *dispatch* between the two code paths. The quantitative
    regression of the diagonal is
    :func:`test_n3_diagonal_reflected_integral_near_interface`.
    """
    rho_1, rho_2, h_1 = 100.0, 500.0, 3.0
    two_layer = gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=h_1)
    multi = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=rho_1, thickness=h_1),
            gf.SoilLayer(resistivity=rho_2, thickness=5.0),
            gf.SoilLayer(resistivity=rho_2, thickness=None),
        ],
    )
    engine = gf.create_engine(backend="mom_sommerfeld", segment_length=0.5)

    def _rod_world(soil):
        world = gf.create_world(soil=soil)
        gf.create_electrode(
            world, "rod", name="g1", position=(0.0, 0.0, 0.0), length=2.8,
        )
        gf.create_source(world, attached_to="g1", magnitude=1.0)
        return world

    z_2 = engine.solve(_rod_world(two_layer)).cluster_impedance("g1")[0]
    z_3 = engine.solve(_rod_world(multi)).cluster_impedance("g1")[0]
    assert z_3.real == pytest.approx(z_2.real, rel=1e-2)


# ---------------------------------------------------------------------
# F04 second pass — the oscillation panel budget
# ---------------------------------------------------------------------

#: Three-layer stack with a *thin* second layer. The remainder decay
#: scale is d_rem = min(2 h_1 - |z - z_s|, d_int + 2 min(h_1, h_2)),
#: so a centimetre-scale h_2 drives d_rem to ~2 h_2 and the number of
#: J_0 half periods to resolve, ~34 s / (pi d_rem), explodes with s.
_THIN = (100.0, 1000.0, 50.0)


def _thin_stack(h_2: float) -> LayerStack:
    return LayerStack(rhos=np.array(_THIN), h=np.array([2.0, h_2]))


def _far_field_asymptote(s: float) -> float:
    """Exact $s \\to \\infty$ limit of the kernel: $(2/s)\\,\\rho_n/\\rho_1$.

    At a horizontal separation far larger than every layer depth the
    stack is indistinguishable from a homogeneous half-space of the
    *bottom* resistivity $\\rho_n$, whose kernel (in the
    $\\rho_1 / 4\\pi$ normalisation this module uses) is
    $2 \\rho_n / (\\rho_1 s)$. Independent of the implementation and,
    crucially, **strictly positive** — which is what makes it a valid
    sign reference.
    """
    return (2.0 / s) * _THIN[-1] / _THIN[0]


#: (h_2, z, s) of the four measured corner cases. The first two were
#: sign-flipped before 0.15.0 and are still budget-limited (they warn);
#: the last two are now converged and silent.
_CORNER_CASES = [
    (0.01, 2.0, 2000.0, True),
    (0.001, 2.0, 200.0, True),
    (0.01, 2.0, 500.0, False),
    (0.01, 1.99, 1000.0, False),
]


@pytest.mark.parametrize(("h_2", "z", "s", "warns"), _CORNER_CASES)
def test_thin_layer_kernel_is_never_negative(
    h_2: float, z: float, s: float, warns: bool,
) -> None:
    """F04 second pass: no sign flip where the asymptote says positive.

    The pre-0.15.0 values for these four rows were -9.4749e-05,
    -9.4761e-04, 1.92531e-03 and 9.58002e-04 against exact
    +5.000e-04, +5.000e-03, 2.000e-03 and 1.000e-03 — the first two
    with the **wrong sign** (-119 %).

    The tolerance below is deliberately loose (2 %): the point of this
    test is the *sign* and the order of magnitude, which no amount of
    quadrature tuning may get wrong. The sharp accuracy assertions live
    in :func:`test_thin_layer_corner_converges_when_budget_is_raised`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SommerfeldConvergenceWarning)
        got = sommerfeld_kernel_value(_thin_stack(h_2), s=s, z=z, z_s=z)
    exact = _far_field_asymptote(s)
    assert got > 0.0, f"negative Green's function {got!r} (F04 regression)"
    assert got == pytest.approx(exact, rel=2e-2)


@pytest.mark.parametrize(
    ("h_2", "z", "s"),
    [(h_2, z, s) for h_2, z, s, warns in _CORNER_CASES if warns],
)
def test_unresolved_oscillation_raises_a_warning(
    h_2: float, z: float, s: float,
) -> None:
    """F04 second pass: non-convergence must be loud.

    Up to 0.15.0 the refinement loop detected this case (it exhausted
    all four grids and fell into the ``else`` branch) but only emitted
    ``logger.debug`` — **zero** Python-level warnings — and returned the
    most refined, i.e. worst covered, value.
    """
    with pytest.warns(SommerfeldConvergenceWarning) as record:
        sommerfeld_kernel_value(_thin_stack(h_2), s=s, z=z, z_s=z)
    message = str(record[0].message)
    # The warning has to say what went wrong, that the sign is at risk
    # and which knob to turn.
    assert "did not converge" in message
    assert "SIGN may be wrong" in message
    assert "max_osc_panels" in message
    assert "epsrel=1.0e-07" in message


def test_in_envelope_evaluations_do_not_warn() -> None:
    """The escalated warning must not spam ordinary users.

    A grid of in-envelope parameter sets (the regime the engine is
    documented for) has to stay silent, otherwise escalating the
    ``else`` branch from DEBUG to a warning would be a usability
    regression rather than a fix.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", SommerfeldConvergenceWarning)
        for rho_2 in (20.0, 200.0, 1000.0):
            for h_1 in (1.0, 2.0, 5.0):
                stack = LayerStack(
                    rhos=np.array([100.0, rho_2]), h=np.array([h_1]),
                )
                for s in (0.05, 0.5, 5.0, 50.0, 200.0):
                    for frac in (0.2, 0.6, 1.0):
                        sommerfeld_kernel_value(
                            stack, s=s, z=frac * h_1, z_s=frac * h_1,
                        )


def test_thin_layer_corner_converges_when_budget_is_raised() -> None:
    """``max_osc_panels`` is a *working* escape hatch.

    Same geometry, three budgets. With a budget far too small the
    coverage is a few per cent of the required $\\lambda$-range and
    the answer is 28 % low (and warns); at the default budget the
    coverage is complete and the kernel hits the far-field asymptote to
    2e-5 relative without warning.
    """
    stack = _thin_stack(0.01)
    exact = _far_field_asymptote(200.0)

    def _run(cap: int) -> tuple[float, int]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = sommerfeld_kernel_value(
                stack, s=200.0, z=2.0, z_s=2.0, max_osc_panels=cap,
            )
        n = sum(
            1 for c in caught
            if issubclass(c.category, SommerfeldConvergenceWarning)
        )
        return value, n

    coarse, n_coarse = _run(3_000)
    medium, n_medium = _run(10_000)
    fine, n_fine = _run(200_000)

    assert n_coarse == 1 and n_medium == 1 and n_fine == 0
    # Raising the budget moves the answer monotonically towards the
    # independent reference.
    assert abs(coarse - exact) > abs(medium - exact) > abs(fine - exact)
    assert coarse == pytest.approx(exact, rel=0.30)
    assert fine == pytest.approx(exact, rel=1e-4)


def test_oscillation_coverage_never_shrinks_under_refinement() -> None:
    """The refinement ladder may not cover *less* than its first rung.

    Design bug behind the sign flip: the half-period step was
    ``pi / (s * refine)`` and the panel *count* was truncated at the
    budget, so the covered range ``n_osc * step`` was
    ``budget * pi / (s * refine)`` — a factor 8 smaller on the last
    rung than on the first. Since ``_reflected_integral`` returns the
    last rung, refining made the result worse. Measured remainder
    values for the s = 2000 corner were -0.000913, -0.000925,
    -0.001341, -0.001504: monotonically *away* from the answer.
    """
    from groundfield.solver.mom_sommerfeld import _REFINEMENTS, _panel_edges

    # s / d_rem = 1e5 — far into the regime where the budget binds.
    d_rem = 0.02
    lam_max, lam_osc, s = 60.0 / d_rem, 34.0 / d_rem, 2000.0
    covered = [
        _panel_edges(
            lam_max, s, lam_osc, refine, max_osc_panels=200_000,
        ).lam_covered
        for refine in _REFINEMENTS
    ]
    assert covered[0] > 0.0
    assert covered == sorted(covered), covered   # non-decreasing
    # Budget-bound case: coverage is refinement-independent, not 1/8.
    assert covered[-1] == pytest.approx(covered[0])

    # And when the budget does *not* bind, coverage is always complete.
    covered_ok = [
        _panel_edges(
            lam_max, 50.0, lam_osc, refine, max_osc_panels=200_000,
        ).lam_covered
        for refine in _REFINEMENTS
    ]
    assert all(c == pytest.approx(lam_osc, rel=1e-3) for c in covered_ok)


def test_matrix_assembly_aggregates_into_one_warning() -> None:
    """An $O(N^2)$ assembly must not emit $O(N^2)$ warnings.

    The message text embeds the geometry, so the default
    ``UserWarning`` filter (which deduplicates by message *text*) would
    not collapse them. ``_build_Z_sommerfeld`` therefore collects the
    per-pair diagnostics and reports the worst offender once, with the
    number of affected entries.
    """
    from groundfield.solver.mom_sommerfeld import _build_Z_sommerfeld

    stack = LayerStack(rhos=np.array(_THIN), h=np.array([2.0, 0.0005]))
    seg_points = np.array([[0.0, 0.0, 1.999], [200.0, 0.0, 1.999]])
    seg_lengths = np.array([1.0, 1.0])
    wire_radii = np.array([0.01, 0.01])

    with pytest.warns(SommerfeldConvergenceWarning) as record:
        Z = _build_Z_sommerfeld(
            seg_points, seg_lengths, wire_radii, stack,
            lambda_max_factor=200.0, epsabs=1e-9, epsrel=1e-7,
        )
    hits = [
        r for r in record
        if issubclass(r.category, SommerfeldConvergenceWarning)
    ]
    assert len(hits) == 1
    assert "2 reaction-matrix entries are affected" in str(hits[0].message)
    assert np.isfinite(Z).all()


# ---------------------------------------------------------------------
# Extreme layer contrast — the same silent path
# ---------------------------------------------------------------------


def _resummed_two_layer_kernel(
    rho_1: float, rho_2: float, h_1: float,
    s: float, z: float, z_s: float, m_max: int = 2_000_000,
) -> float:
    """Tagg / Sunde series with an analytic tail — usable at $|K| \\to 1$.

    :func:`_tagg_series` needs $O(1/(1-|K|))$ terms, i.e. $10^8$ at
    $\\rho_2/\\rho_1 = 10^8$. Here the first ``m_max`` terms are summed
    exactly and the remainder is closed in form using
    $g(m) \\to 2/(m h_1)$:
    $\\sum_{m > M} K^m \\cdot 2/(m h_1)
    = (2/h_1)\\,[-\\ln(1 - K) - \\sum_{m \\le M} K^m/m]$.
    The neglected correction is $O(M^{-2})$.
    """
    import math

    k_1 = (rho_2 - rho_1) / (rho_2 + rho_1)
    d_z, p = abs(z - z_s), z + z_s
    m = np.arange(1, m_max + 1, dtype=float)
    two_mh = 2.0 * m * h_1
    g = (
        1.0 / np.hypot(s, two_mh - d_z) + 1.0 / np.hypot(s, two_mh + d_z)
        + 1.0 / np.hypot(s, two_mh - p) + 1.0 / np.hypot(s, two_mh + p)
    )
    k_m = np.exp(m * math.log(k_1))
    head = float((k_m * g).sum())
    tail = (2.0 / h_1) * (-math.log1p(-k_1) - float((k_m / m).sum()))
    direct = 1.0 / np.hypot(s, d_z) + 1.0 / np.hypot(s, p)
    return float(direct + head + tail)


@pytest.mark.parametrize(
    ("contrast", "before"),
    [
        (1.0e2, -2.2e-16),
        (1.0e4, -1.5e-13),
        (1.0e6, -3.4e-4),    # was returned after a failed convergence test
        (1.0e8, -5.2e-2),    # ... and logged at DEBUG only
    ],
)
def test_extreme_contrast_resolves_the_multiplier_pole(
    contrast: float, before: float,
) -> None:
    """The $|K_1| \\to 1$ error-control claim now holds.

    For $|K_1| \\to 1$ the multiple-reflection multiplier
    $1/(1 - \\Gamma_1 e^{-2\\lambda h_1})$ concentrates on the
    $\\lambda$-scale $(1 - \\Gamma_1(0)) / 2 h_1$ — 2.5e-9 1/m at
    $\\rho_2/\\rho_1 = 10^8$, $h_1 = 2\\,\\text{m}$, against an
    innermost quadratically-graded panel at 1.6e-3 1/m. The graded grid
    therefore missed the pole entirely and the kernel came out 5.2 %
    low, *after* the internal convergence test had already failed and
    been logged at DEBUG. A geometric panel family covering the pole
    scale (8 panels/decade) closes it.

    ``before`` records the pre-0.15.0 relative error at this point for
    documentation; the assertion is that we are now well below it.
    """
    stack = LayerStack(
        rhos=np.array([100.0, 100.0 * contrast]), h=np.array([2.0]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", SommerfeldConvergenceWarning)
        got = sommerfeld_kernel_value(stack, s=0.5, z=0.8, z_s=0.8)
    exact = _resummed_two_layer_kernel(
        100.0, 100.0 * contrast, 2.0, 0.5, 0.8, 0.8,
    )
    assert got == pytest.approx(exact, rel=1e-9)
    # ``max(..., 1e-10)`` absorbs the round-off of the resummed
    # reference itself (2e6 float64 terms), which is at the 1e-13 level
    # and is the floor the two mild contrasts already sat on.
    assert abs(got / exact - 1.0) <= max(abs(before), 1e-10)


# ---------------------------------------------------------------------
# The accuracy knobs must be reachable from the public API
# ---------------------------------------------------------------------


def test_engine_forwards_the_sommerfeld_knobs_to_the_backend() -> None:
    """``Engine.solve`` used to drop every accuracy knob on the floor.

    Up to 0.14.x the dispatch was ``solve_mom_sommerfeld(world, self)``,
    so ``lambda_max_factor`` / ``epsabs`` / ``epsrel`` could not be set
    by a user at all. They are now four ``Engine`` fields, following the
    ``image_max_terms`` / ``image_series_tol`` precedent.
    """
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
    engine = gf.Engine(
        backend="mom_sommerfeld",
        segment_length=SEG,
        sommerfeld_lambda_max_factor=123.0,
        sommerfeld_epsabs=2e-10,
        sommerfeld_epsrel=3e-8,
        sommerfeld_max_osc_panels=54_321,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = engine.solve(_ring_world(soil)).metadata
    assert meta["lambda_max_factor"] == 123.0
    assert meta["epsabs"] == 2e-10
    assert meta["epsrel"] == 3e-8
    assert meta["max_osc_panels"] == 54_321


def test_engine_lambda_max_factor_changes_the_solve_result() -> None:
    """The forwarded knob must reach the *kernel*, not just the metadata."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)

    def _impedance(factor: float) -> float:
        engine = gf.Engine(
            backend="mom_sommerfeld", segment_length=SEG,
            sommerfeld_lambda_max_factor=factor,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = engine.solve(_ring_world(soil))
        return result.cluster_impedance("g1")[0].real

    z_default = _impedance(200.0)
    z_truncated = _impedance(1.0)
    # A factor of 1 discards the remainder tail above 1/d_rem, which
    # must be visible. Pre-0.15.0 both calls returned the same number
    # because the keyword never left ``Engine.solve``.
    assert z_truncated != pytest.approx(z_default, rel=1e-6)


@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0, 5.0, 20.0, 60.0])
def test_raising_lambda_max_factor_moves_towards_the_reference(
    factor: float,
) -> None:
    """Tightening the knob must monotonically improve the answer.

    Measured relative error against the exact Tagg / Sunde series at
    ``s = 2, z = z_s = 0.8, h_1 = 2, rho_2/rho_1 = 5``:
    -1.70e-01, -7.63e-02, -1.58e-02, +2.60e-04, +7.6e-11, 0.0 — i.e.
    the knob is a genuine accuracy control and saturates at 60, as
    documented.
    """
    stack = LayerStack(rhos=np.array([100.0, 500.0]), h=np.array([2.0]))
    exact = _tagg_series(100.0, 500.0, 2.0, 2.0, 0.8, 0.8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SommerfeldConvergenceWarning)
        got = sommerfeld_kernel_value(
            stack, s=2.0, z=0.8, z_s=0.8, lambda_max_factor=factor,
        )
        coarser = sommerfeld_kernel_value(
            stack, s=2.0, z=0.8, z_s=0.8, lambda_max_factor=0.5 * factor,
        )
    assert abs(got - exact) <= abs(coarser - exact)
    if factor >= 20.0:
        assert got == pytest.approx(exact, rel=1e-9)


def test_engine_rejects_nonsensical_sommerfeld_knobs() -> None:
    """The new fields are validated, not silently accepted."""
    import pydantic

    for kwargs in (
        {"sommerfeld_lambda_max_factor": 0.0},
        {"sommerfeld_epsabs": -1e-9},
        {"sommerfeld_epsrel": 0.0},
        {"sommerfeld_max_osc_panels": 0},
    ):
        with pytest.raises(pydantic.ValidationError):
            gf.Engine(backend="mom_sommerfeld", **kwargs)
