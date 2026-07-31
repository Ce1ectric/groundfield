"""Regression tests for review-pass-9 findings F14 and F15.

Both findings concern the Sommerfeld machinery of
:mod:`groundfield.coupling.layered_green`, i.e. the numerical
evaluation of

.. math::

   \\varphi(s, z, z_s) \\;=\\; \\int_0^{\\infty}\\!
   \\Phi(\\lambda, z, z_s)\\,J_0(\\lambda s)\\,\\lambda\\,d\\lambda .

**F14 — oscillation resolution of the $\\lambda$ grid.** The linear
region of the quadrature grid places one 16-node Gauss panel per
$J_0(\\lambda s)$ period, but the panel count was hard-capped at 4096
with no adequacy check. Once $n_\\text{osc} = \\lambda_\\text{max}s/2\\pi$
exceeded ~8192 the grid fell below eight nodes per oscillation and the
result degraded *silently* — including when the user raised
``lambda_max_factor``, the documented accuracy knob. Measured on
v0.14.1 for $\\rho_1 = 100$, $\\rho_2 = 1000$, $h_1 = 5$ m,
$z = z_s = 6$ m, $s = 200$ m:

===========================  ==============  ==============
``lambda_max_factor``        v0.14.1         converged
===========================  ==============  ==============
200                          −0.23113144     −0.23113144
4000                         −0.23121361     −0.23113144
6000                         **+0.03487**    −0.23113144
8000                         **+4.32609**    −0.23113144
===========================  ==============  ==============

**F15 — large-group interpolation.** Groups above 64 distance pairs
were sampled on a *fixed* 48-node log grid and linearly interpolated,
so the accuracy depended on the network extent (0.94 % over a 400 m
span, 2.2e-3 for ``two_layer_probe_matrix`` against a documented
1e-4) and the *same* matrix entry changed by ~3 % when the group
crossed the 64-pair threshold.

Both fixes rest on one structural change: the primary (source-layer
particular) spectral term $\\rho_k/(2\\lambda)\\,e^{-\\lambda|z-z_s|}$,
the only part of $\\Phi$ that does **not** decay in $\\lambda$, is now
inverted in closed form as $\\rho_k/(2R)$,
$R = \\sqrt{s^2 + (z-z_s)^2}$. Its $\\lambda$-truncated quadrature used
to leave an $O((\\lambda_\\text{max}R)^{-1/2})$ ripple of period
$2\\pi/\\lambda_\\text{max}$ in ``s`` — up to 4.5 % on the total kernel,
non-monotone in ``lambda_max_factor``, and impossible for any coarse
interpolant to follow.

Reference values marked "adaptive" were produced with an independent
per-period QUADPACK integration of the reflected spectral difference
(``scipy.integrate.quad``, ``epsrel=1e-11``), agreement ~2e-7 limited
by the 32-node logarithmic region of the production grid.

Re-audit (2026-07-30): the first F14 fix was **not sufficient**
-------------------------------------------------------------
Splitting off the primary term removed the only spectral term that does
not decay in $\\lambda$ at all, but the *reflected* remainder decays
only like $e^{-\\lambda d_\\text{img}}$ — and $\\lambda_\\text{max}$ was
still ``lambda_max_factor / min(h_1, s + z + z_s)``, which has no
relation to $d_\\text{img}$. Every image distance can degenerate:

======================================  ===========================
geometry                                degenerate image distance
======================================  ===========================
pair at the air/soil surface            $z + z_s \\to 0$
pair at the layer interface             $|2h_1 - z - z_s| \\to 0$
pair straddling the interface           $z_l - z_u \\to 0$
======================================  ===========================

Measured on those three regimes at the default knob (full kernel,
against the closed form below): −2.7e-2 at $z = z_s = 0$, +4.1 % at
$z = z_s = h_1$, and **−39.5 %** at $z = z_s = h_1 + 0.5$ mm — the last
one *worse* than v0.14.1's +10.7 % and reached by any mesh fine enough
to place a segment centre within centimetres of $h_1$, so the error
*grew under mesh refinement*. No warning in any case.

The tests below therefore check every one of those regimes against
:func:`_closed_form_kernel`, an independent analytic reference (see its
docstring) rather than against the module's own output at another knob
setting — a self-consistency check cannot see a systematic truncation
error, which is exactly how the first fix passed review.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from groundfield.coupling import layered_green as lg
from groundfield.coupling.layered_green import (
    SommerfeldResolutionWarning,
    SommerfeldTailTruncationWarning,
    two_layer_layered_correction_group,
    two_layer_layered_correction_real_space,
    two_layer_probe_matrix,
    two_layer_real_space_kernel,
)

# rho_1 = 100, rho_2 = 1000, h_1 = 5 m — the two-layer soil of the
# audit's cross-layer reference set (K = +0.818).
_SOIL = dict(rho_1=100.0, rho_2=1000.0, h_1=5.0)


# =====================================================================
# Independent analytic reference
# =====================================================================
def _closed_form_terms(z, z_s, rho_1, rho_2, h_1, n_gen):
    """Image terms $(c_j, d_j)$ of the exact 2-layer Green's function.

    Derived by hand, **not** taken from the module: solve the matching
    problem for the spectral kernel in closed form,

    .. math::

       \\Phi_{uu} = \\frac{\\rho_1}{2\\lambda}\\Big[e^{-\\lambda|z-z_s|}
       + \\frac{e^{-\\lambda(z+z_s)} + K e^{-\\lambda(2h_1+z-z_s)}
       + K e^{-\\lambda(2h_1-z-z_s)} + K e^{-\\lambda(2h_1-z+z_s)}}
       {1 - K e^{-2\\lambda h_1}}\\Big],

    with $K = (\\rho_2-\\rho_1)/(\\rho_2+\\rho_1)$, expand
    $1/(1 - Ke^{-2\\lambda h_1}) = \\sum_m K^m e^{-2m\\lambda h_1}$ and
    invert each term with the *exact* Hankel pair
    $\\int_0^\\infty \\frac{c}{2\\lambda}e^{-\\lambda d}J_0(\\lambda s)
    \\lambda\\,d\\lambda = \\frac{c}{2\\sqrt{s^2+d^2}}$. The layer-2 and
    cross-layer kernels follow from the same solve; for a source in
    layer 2 the amplitude is
    $\\alpha = (1-K)\\,\\frac{\\rho_2}{2\\lambda}
    e^{-\\lambda(z_s-h_1)}/(1-Ke^{-2\\lambda h_1})$, giving the
    coefficients below.

    This is the Sunde/Tagg image series, i.e. the same physics the
    ``image_2layer`` engine implements for the upper–upper case, but
    written out for all four layer combinations. Its only error source
    is the geometric truncation after ``n_gen`` generations, bounded by
    $4\\max(\\rho)|K|^{n_\\text{gen}}/((1-|K|)\\,2 d_\\text{min})$ —
    checked explicitly in
    :func:`test_closed_form_reference_matches_the_spectral_kernel`.
    """
    K = (rho_2 - rho_1) / (rho_2 + rho_1)
    q = 2.0 * h_1
    out = []
    if (z <= h_1) == (z_s <= h_1):
        out.append((rho_1 if z_s <= h_1 else rho_2, abs(z - z_s)))
    for m in range(n_gen):
        Km = K ** m
        if z <= h_1 and z_s <= h_1:
            S, D = z + z_s, abs(z - z_s)
            if m == 0:
                out.append((rho_1, S))
            else:
                c = rho_1 * Km
                out += [(c, m * q + S), (c, m * q + D),
                        (c, m * q - D), (c, m * q - S)]
        elif z > h_1 and z_s > h_1:
            S = z + z_s
            if m == 0:
                out += [(-rho_2 * K, S - q), (rho_2 * (1.0 - K * K), S)]
            else:
                out.append((rho_2 * (1.0 - K * K) * Km, S + m * q))
        else:
            z_u, z_l = (z, z_s) if z <= h_1 else (z_s, z)
            c = 2.0 * rho_1 * rho_2 / (rho_1 + rho_2) * Km
            out += [(c, m * q + (z_l - z_u)), (c, m * q + (z_l + z_u))]
    return out


def _closed_form_kernel(
    s, z, z_s, *, rho_1, rho_2, h_1, n_gen=4000, r_min=0.0,
):
    """Full kernel $G(s, z, z_s)$ from :func:`_closed_form_terms`."""
    s = np.asarray(s, dtype=float)
    acc = np.zeros(np.shape(s), dtype=float)
    for c, d in _closed_form_terms(z, z_s, rho_1, rho_2, h_1, n_gen):
        acc = acc + 0.5 * c / np.maximum(np.hypot(s, d), max(r_min, 1e-300))
    return acc


def _closed_form_correction(
    s, z, z_s, *, rho_1, rho_2, h_1, rho_baseline=None, n_gen=4000,
    r_min=0.0,
):
    """Layered minus homogeneous baseline, differenced **term by term**.

    Differencing the term lists rather than two summed kernels is what
    keeps the reference usable at small ``s``: both kernels are
    dominated there by a $1/R$ direct term that cancels identically, and
    subtracting the sums would lose it to rounding.
    """
    if rho_baseline is None:
        rho_baseline = rho_1
    merged: dict[float, float] = {}
    for c, d in _closed_form_terms(z, z_s, rho_1, rho_2, h_1, n_gen):
        merged[round(d, 12)] = merged.get(round(d, 12), 0.0) + c
    for c, d in _closed_form_terms(
        z, z_s, rho_baseline, rho_baseline, h_1, n_gen,
    ):
        merged[round(d, 12)] = merged.get(round(d, 12), 0.0) - c
    s = np.asarray(s, dtype=float)
    acc = np.zeros(np.shape(s), dtype=float)
    for d, c in merged.items():
        acc = acc + 0.5 * c / np.maximum(np.hypot(s, d), max(r_min, 1e-300))
    return acc


def test_closed_form_reference_matches_the_spectral_kernel() -> None:
    """Validate the reference itself against the module's *spectral* kernel.

    :func:`two_layer_spectral_kernel` solves the 4x4 matching problem
    numerically at each $\\lambda$ and shares no code with the image
    expansion, so agreement node by node pins the reference's
    coefficients and distances — including the signs, which is where a
    hand-derived series goes wrong.
    """
    lambdas = np.logspace(-4.0, 2.0, 60)
    for z, z_s, rho_1, rho_2, h_1 in (
        (0.5, 0.5, 100.0, 1000.0, 5.0),      # uu
        (0.0, 0.0, 100.0, 1000.0, 5.0),      # uu, surface pair
        (5.0, 5.0, 100.0, 1000.0, 5.0),      # uu, at the interface
        (6.0, 6.0, 100.0, 1000.0, 5.0),      # ll
        (5.0005, 5.0005, 100.0, 1000.0, 5.0),  # ll, just below h_1
        (2.0, 8.0, 100.0, 1000.0, 5.0),      # cross-layer
        (8.0, 2.0, 100.0, 1000.0, 5.0),      # cross-layer, reciprocal
        (0.7, 0.7, 1000.0, 30.0, 5.0),       # K < 0
        (6.0, 6.0, 100.0, 10000.0, 6.096),   # K = +0.98
    ):
        phi = lg.two_layer_spectral_kernel(
            lambdas, z, z_s, rho_1=rho_1, rho_2=rho_2, h_1=h_1,
        )
        terms = _closed_form_terms(z, z_s, rho_1, rho_2, h_1, n_gen=4000)
        series = sum(
            c / (2.0 * lambdas) * np.exp(-lambdas * d) for c, d in terms
        )
        rel = np.abs(series - phi) / np.abs(phi)
        assert rel.max() < 1e-12, (z, z_s, rho_2, rel.max())


# =====================================================================
# F14 — the accuracy knob must not make the answer worse
# =====================================================================
def test_lambda_max_factor_never_flips_the_sign() -> None:
    """Raising ``lambda_max_factor`` must converge, not destroy.

    The layered correction decays like $e^{-2\\lambda h_1}$, so every
    factor beyond the default integrates the same physical quantity and
    the results must agree. On v0.14.1 the 4096-panel cap left 2.7
    nodes per oscillation at factor 6000 and the sign flipped
    (−0.2311 -> +0.0349, 115 % error).
    """
    kw = dict(rho_baseline=1000.0, **_SOIL)
    reference = two_layer_layered_correction_real_space(
        200.0, 6.0, 6.0, lambda_max_factor=200.0, **kw,
    )
    assert reference < 0.0
    for factor in (2000.0, 4000.0, 6000.0, 8000.0):
        value = two_layer_layered_correction_real_space(
            200.0, 6.0, 6.0, lambda_max_factor=factor, **kw,
        )
        assert value < 0.0, (factor, value)
        assert value == pytest.approx(reference, rel=1e-6), factor


def test_default_grid_resolves_remote_radii() -> None:
    """At the default factor a 6 km pair must still be converged.

    v0.14.1: correction 0.13994183 (−6.7 %) and kernel 0.15563183
    (−6.6 %) because n_osc = 38 197 was clamped to 4096 panels.
    Reference: adaptive per-period QUADPACK of the spectral difference.
    """
    corr = two_layer_layered_correction_real_space(
        6000.0, 0.7, 0.7, **_SOIL,
    )
    assert corr == pytest.approx(0.1499885467, rel=1e-5)
    # Independent of the truncation knob (the integrand is < 1e-13 well
    # before lambda_max in either case).
    corr_refined = two_layer_layered_correction_real_space(
        6000.0, 0.7, 0.7, lambda_max_factor=400.0, **_SOIL,
    )
    assert corr == pytest.approx(corr_refined, rel=1e-5)


def test_thin_top_layer_at_one_km_is_resolved() -> None:
    """h_1 = 1 m raises lambda_max fivefold; s = 1 km then needs
    31 831 panels. v0.14.1 returned 0.98836860 (−1.15 %)."""
    value = two_layer_real_space_kernel(
        1000.0, 0.7, 0.7, rho_1=100.0, rho_2=1000.0, h_1=1.0,
    )
    refined = two_layer_real_space_kernel(
        1000.0, 0.7, 0.7, rho_1=100.0, rho_2=1000.0, h_1=1.0,
        lambda_max_factor=400.0,
    )
    assert value == pytest.approx(refined, rel=1e-5)
    assert value == pytest.approx(0.9999004087, rel=1e-6)


def test_panel_count_scales_with_oscillation_content() -> None:
    """The grid must keep at least eight nodes per J0 oscillation.

    Eight nodes per oscillation is the module's own adequacy criterion
    (the single-panel fast path uses exactly that bound). v0.14.1
    delivered 2.74 nodes per oscillation for the last case below.
    """
    for lambda_max, char_length, s in (
        (40.0, 5.0, 200.0),        # n_osc ~ 1273  (historically fine)
        (200.0, 5.0, 200.0),       # n_osc ~ 6366  (at the old cap)
        (40.0, 6000.0, 200.0),     # n_osc ~ 38197 (historically broken)
        (200.0, 1.0, 1000.0),      # n_osc ~ 31831 (thin top layer)
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error", SommerfeldResolutionWarning)
            lambdas, _ = lg._hankel_lambda_grid(
                lambda_max=lambda_max, char_length=char_length, s=s,
                n_log=32, n_lin=96,
            )
        n_osc = lambda_max * s / (2.0 * math.pi)
        n_linear = lambdas.size - 32
        nodes_per_osc = n_linear / n_osc
        assert nodes_per_osc >= lg._HANKEL_MIN_NODES_PER_OSCILLATION - 0.01, (
            lambda_max, char_length, s, nodes_per_osc,
        )


def test_soft_budget_keeps_the_historic_full_resolution() -> None:
    """Below the soft budget the historic one-panel-per-period target is
    kept, so results the 4096-panel cap resolved adequately are
    unchanged (16 nodes per oscillation)."""
    lambdas, _ = lg._hankel_lambda_grid(
        lambda_max=40.0, char_length=5.0, s=200.0, n_log=32, n_lin=96,
    )
    n_osc = 40.0 * 200.0 / (2.0 * math.pi)
    n_linear = lambdas.size - 32
    assert n_linear == lg._HANKEL_PANEL_ORDER * math.ceil(n_osc)


def test_exhausted_node_budget_warns_and_names_the_parameter() -> None:
    """A grid that cannot resolve the oscillations must warn, not lie.

    v0.14.1 silently returned +4.33 (wrong sign, 1900 % error) for
    ``lambda_max_factor = 8000`` at s = 200 m and worse beyond; the
    failure mode had no diagnostic whatsoever.

    Since 0.15.0 ``lambda_max`` is no longer taken from the knob — it is
    lowered until the panel budget fits, and the tail is bounded by the
    closed-form image split instead (:func:`lg._plan_truncation`), so
    the knob can no longer *drive* the grid into this state. The guard
    is now the second line of defence behind the image-term budget, and
    is exercised by asking the grid builder directly for a truncation
    that cannot be resolved at the given radius.
    """
    with pytest.warns(SommerfeldResolutionWarning) as record:
        lg._hankel_lambda_grid(
            lambda_max=2.0e6, char_length=5.0, s=2000.0,
            n_log=32, n_lin=96,
        )
    message = str(record[0].message)
    assert "lambda_max_factor" in message
    assert "nodes per oscillation" in message


def test_absurd_lambda_max_factor_no_longer_degrades_the_grid() -> None:
    """The knob cannot reach the unresolvable regime any more.

    Same call that produced +4.33 (v0.14.1) and a
    ``SommerfeldResolutionWarning`` at 0.15.0.dev: it must now simply
    return the converged value, silently.
    """
    reference = two_layer_layered_correction_real_space(
        200.0, 6.0, 6.0, rho_baseline=1000.0, **_SOIL,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value = two_layer_layered_correction_real_space(
            200.0, 6.0, 6.0, rho_baseline=1000.0,
            lambda_max_factor=5.0e4, **_SOIL,
        )
    assert value == pytest.approx(reference, rel=1e-9)


def test_production_scale_group_emits_no_resolution_warning() -> None:
    """A 400 m AP1-scale reaction block (n_osc ~ 9095) is within the
    adequacy floor and must stay silent — the warning is for genuinely
    unconverged grids only."""
    s_values = np.linspace(0.0, 400.0, 1600)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SommerfeldResolutionWarning)
        two_layer_layered_correction_group(
            s_values, 0.7, 0.7, rho_1=1000.0, rho_2=30.0, h_1=5.0,
            rho_baseline=1000.0,
        )


# =====================================================================
# F14 / F15 — closed-form primary term
# =====================================================================
def test_primary_term_is_the_closed_form_1_over_R() -> None:
    """``_primary_term_real_space`` is the exact Hankel inverse of the
    particular spectral term, clamped at the quadrature's resolution
    limit 1/lambda_max."""
    s = np.array([0.0, 0.5, 3.0, 50.0])
    got = lg._primary_term_real_space(100.0, s, 2.0, 40.0)
    want = 0.5 * 100.0 / np.hypot(s, 2.0)
    np.testing.assert_allclose(got, want, rtol=1e-14)
    # Coincident points: clamped, finite, equal to rho*lambda_max/2.
    at_zero = float(lg._primary_term_real_space(100.0, 0.0, 0.0, 40.0))
    assert at_zero == pytest.approx(0.5 * 100.0 * 40.0, rel=1e-12)


@pytest.mark.parametrize("s", [1.0, 10.0, 30.0])
def test_full_kernel_homogeneous_limit_is_machine_accurate(s) -> None:
    """rho_2 = rho_1 with z = z_s: the kernel must reproduce
    $\\rho/2\\,(1/r + 1/r')$.

    v0.14.1 was off by −4.5 % (s = 1 m), −0.46 % (s = 10 m) and
    −0.88 % (s = 30 m) — the truncation ripple of the numerically
    integrated 1/R term. The equal-depth case is exactly the
    reaction-matrix geometry of a buried grid.
    """
    rho, z = 100.0, 0.5
    got = two_layer_real_space_kernel(
        s, z, z, rho_1=rho, rho_2=rho, h_1=5.0,
    )
    want = 0.5 * rho * (1.0 / s + 1.0 / math.hypot(s, 2.0 * z))
    assert got == pytest.approx(want, rel=1e-8)


def test_full_kernel_is_smooth_in_s() -> None:
    """No $2\\pi/\\lambda_\\text{max}$ ripple: the kernel of an
    equal-depth pair is strictly decreasing in ``s``.

    v0.14.1 swung by ±4 % with a 0.031 m period around s = 10 m
    (14.49 … 16.44 over s in [9.5, 10.5]), which is what made the
    documented 1e-4 interpolation tolerance unreachable.
    """
    s = np.linspace(9.5, 10.5, 61)
    values = np.array([
        two_layer_real_space_kernel(
            float(x), 0.5, 0.5, rho_1=300.0, rho_2=100.0, h_1=6.096,
        )
        for x in s
    ])
    assert np.all(np.diff(values) < 0.0)
    # And convex (no sign change of the second difference).
    assert np.all(np.diff(values, 2) > 0.0)


def test_full_kernel_converges_in_lambda_max_factor() -> None:
    """The public accuracy knob must be monotone for the full kernel
    too. v0.14.1 gave 81.49 / 85.88 / 84.56 for factors 200 / 1000 /
    5000 around an exact 85.36 — oscillation, not convergence."""
    args = (10.0, 0.5, 0.5)
    kw = dict(rho_1=300.0, rho_2=100.0, h_1=6.096)
    base = two_layer_real_space_kernel(*args, **kw)
    for factor in (400.0, 1000.0):
        assert two_layer_real_space_kernel(
            *args, lambda_max_factor=factor, **kw,
        ) == pytest.approx(base, rel=1e-7)


# =====================================================================
# F15 — large-group interpolation
# =====================================================================
def test_interpolated_branch_matches_exact_branch() -> None:
    """The same distances must give the same numbers whether the group
    is small (exact contraction) or large (interpolated).

    A 3 % step as a function of *problem size* is indefensible in a
    reference solver. Measured on v0.14.1 for this configuration:
    4.88e-2 relative at s = 180 m.
    """
    query = np.linspace(2.0, 190.0, 40)
    padding = np.linspace(0.0, 200.0, 320)
    kw = dict(rho_baseline=1000.0, **_SOIL)

    exact = two_layer_layered_correction_group(query, 6.0, 6.0, **kw)
    padded = two_layer_layered_correction_group(
        np.concatenate([query, padding]), 6.0, 6.0, **kw,
    )[: query.size]

    rel = np.abs(padded - exact) / np.maximum(np.abs(exact), 1e-12)
    assert rel.max() < 1e-4, (rel.argmax(), rel.max())


def test_group_matches_scalar_over_a_400m_span() -> None:
    """AP1 production shape (rho_1 = 1000, rho_2 = 30, h_1 = 5 m, both
    conductors at 0.7 m, 400 m span, ``rho_baseline`` matched to the
    source layer as ``image_2layer`` does).

    v0.14.1: 9.449e-3 worst relative deviation from the per-pair
    scalar kernel at s = 351 m, where |dG| ~ |G_hom| — i.e. ~0.5 % on
    the total far-field mutual entry.
    """
    s_values = np.linspace(0.0, 400.0, 400)
    kw = dict(rho_1=1000.0, rho_2=30.0, h_1=5.0, rho_baseline=1000.0)
    grouped = two_layer_layered_correction_group(s_values, 0.7, 0.7, **kw)
    scalar = np.array([
        two_layer_layered_correction_real_space(float(x), 0.7, 0.7, **kw)
        for x in s_values
    ])
    rel = np.abs(grouped - scalar) / np.maximum(np.abs(scalar), 1e-12)
    assert rel.max() < 1e-4, (s_values[int(rel.argmax())], rel.max())


def test_group_matches_scalar_for_a_lower_layer_pair() -> None:
    """Both points in layer 2 with the default (mismatched)
    ``rho_baseline``: the residual primary term then carries a genuine
    1/R singularity, which v0.14.1 pushed through the interpolation and
    got 17 % wrong at s = 6 m."""
    s_values = np.linspace(0.0, 200.0, 200)
    grouped = two_layer_layered_correction_group(
        s_values, 6.0, 6.0, **_SOIL,
    )
    scalar = np.array([
        two_layer_layered_correction_real_space(float(x), 6.0, 6.0, **_SOIL)
        for x in s_values
    ])
    rel = np.abs(grouped - scalar) / np.maximum(np.abs(scalar), 1e-12)
    assert rel.max() < 1e-4, (s_values[int(rel.argmax())], rel.max())


def test_probe_matrix_meets_its_documented_tolerance() -> None:
    """``two_layer_probe_matrix`` documents "relative error <~ 1e-4" for
    the interpolated large-group path; v0.14.1 measured 2.171e-3."""
    rng = np.random.default_rng(3)
    xy = rng.uniform(-45.0, 45.0, size=(400, 2))
    probes = np.column_stack([xy, np.zeros(400)])
    sources = np.array([[0.0, 0.0, 0.5], [5.0, 3.0, 0.5]])
    kw = dict(rho_1=300.0, rho_2=100.0, h_1=6.096)

    grouped = two_layer_probe_matrix(probes, sources, **kw)
    per_pair = np.vstack([
        two_layer_probe_matrix(probes[i:i + 1], sources, **kw)
        for i in range(probes.shape[0])
    ])
    rel = np.abs(grouped - per_pair) / np.maximum(np.abs(per_pair), 1e-30)
    assert rel.max() < 1e-4, rel.max()


def test_interp_node_count_scales_with_the_dynamic_range() -> None:
    """A fixed node count cannot hold a tolerance over an arbitrary
    span — the count must grow with ``t_hi - t_lo``."""
    narrow = lg._interp_node_count(math.log(1.0), math.log(10.0))
    wide = lg._interp_node_count(math.log(1.0), math.log(10000.0))
    assert wide > narrow
    assert narrow >= lg._HANKEL_INTERP_NODES_MIN
    assert wide <= lg._HANKEL_INTERP_NODES_MAX
    # ~12 nodes per e-fold over the wide range.
    span = math.log(10000.0) - math.log(1.0)
    assert wide == pytest.approx(
        lg._HANKEL_INTERP_NODES_PER_LN * span, rel=0.05,
    )


def test_log_grid_offset_picks_the_nearest_image_distance() -> None:
    """The interpolation coordinate ``t = ln(s + c)`` must use the
    smallest image distance, not the 1 mm floor that every equal-depth
    group collapsed to."""
    # Equal depths just below the surface: nearest image is the surface
    # image of the source at z + z_s.
    assert lg._log_grid_offset(0.7, 0.7, 5.0) == pytest.approx(1.4)
    # Both points in layer 2: nearest image is |2 h_1 - z - z_s|.
    assert lg._log_grid_offset(6.0, 6.0, 5.0) == pytest.approx(2.0)
    # Cross-layer: the transmitted term sets the scale.
    assert lg._log_grid_offset(0.5, 8.0, 6.096) == pytest.approx(7.5)
    # Degenerate (z = z_s = h_1): falls back to the floor.
    assert lg._log_grid_offset(5.0, 5.0, 5.0) == pytest.approx(1e-3)
    # Since 0.15.0 the degenerate image term is evaluated in closed form
    # *outside* the interpolation, so the offset must be the decay length
    # of what is actually interpolated — not the 1 mm floor, which would
    # spend half the nodes on an s-range where nothing happens.
    assert lg._log_grid_offset(
        5.0, 5.0, 5.0, d_rest=10.0,
    ) == pytest.approx(10.0)
    # ...and it is never *lowered* by d_rest.
    assert lg._log_grid_offset(
        0.5, 8.0, 6.096, d_rest=1.0,
    ) == pytest.approx(7.5)


def test_homogeneous_limit_of_the_group_is_exactly_zero() -> None:
    """Unchanged contract: rho_2 == rho_1 gives an identically zero
    correction for any group size and any branch."""
    s_values = np.linspace(0.0, 400.0, 500)
    out = two_layer_layered_correction_group(
        s_values, 0.7, 3.0, rho_1=120.0, rho_2=120.0, h_1=5.0,
    )
    assert np.array_equal(out, np.zeros_like(out))


# =====================================================================
# F14 re-audit — the truncation must be guaranteed, not hoped for
# =====================================================================
# Every test in this block compares against _closed_form_kernel /
# _closed_form_correction, i.e. an analytic reference, never against the
# module at another knob setting. The tolerance 1e-7 is two orders
# looser than what is actually measured (~3e-10) and three orders
# tighter than the module's documented 1e-4.
_TOL = 1.0e-7


@pytest.mark.parametrize("z", [0.0, 0.001, 0.05, 0.25])
@pytest.mark.parametrize("s", [1.0, 10.0, 30.0, 100.0])
def test_surface_pair_is_truncation_bounded(z: float, s: float) -> None:
    """(1a) Both points at the air/soil surface: ``z + z_s -> 0``.

    The free-surface image sits at $d = z + z_s$, so its
    $\\lambda$-truncation is of order one when both points are shallow.
    Measured at 0.15.0.dev (default knob, full kernel): −2.71e-2 at
    z = 0/s = 1, −4.56e-3 at s = 10, −8.83e-3 at s = 30, and still
    −8.0e-5 at z = 0.05 — the same magnitudes v0.14.1 had at z = 0.5,
    i.e. the defect had moved rather than gone.
    """
    got = two_layer_real_space_kernel(s, z, z, **_SOIL)
    want = float(_closed_form_kernel(s, z, z, **_SOIL))
    assert got == pytest.approx(want, rel=_TOL), (z, s, got, want)


@pytest.mark.parametrize("s", [0.5, 1.0, 10.0, 100.0])
def test_interface_depth_pair_is_truncation_bounded(s: float) -> None:
    """(1b) ``z = z_s = h_1``: the first interface image sits at d = 0.

    Measured at 0.15.0.dev for s = 1 m: +4.11 % at the default knob,
    and non-monotone in it (+13.9 / −9.6 / +2.19 / +1.91 / +4.11 /
    +1.57 / −1.77 / +0.79 % for factors 10 … 3000) — worse at the
    default than at 100.
    """
    got = two_layer_real_space_kernel(s, 5.0, 5.0, **_SOIL)
    want = float(_closed_form_kernel(s, 5.0, 5.0, **_SOIL))
    assert got == pytest.approx(want, rel=_TOL), (s, got, want)


@pytest.mark.parametrize(
    "z", [5.0005, 5.0015, 5.005, 5.015, 5.05, 5.0 + 1e-6, 4.9999, 4.99],
)
def test_near_interface_pair_does_not_degrade_under_refinement(
    z: float,
) -> None:
    """(1c) The worst regime, and the one that broke mesh convergence.

    A pair of segment centres a few millimetres from $h_1$ has
    $|2h_1 - z - z_s| \\to 0$. Measured at 0.15.0.dev (s = 1 m, default
    knob): −39.5 % at z = 5.0005, −36.4 % at 5.0015, −27.5 % at 5.005,
    −12.3 % at 5.015, −0.73 % at 5.05 — *worse* than v0.14.1 (+10.7 %,
    +13.8 %, +22.7 %) and growing as the mesh is refined towards the
    interface, which destroys mesh convergence for exactly the
    cross-layer rods ADR-0007 exists for.
    """
    got = two_layer_real_space_kernel(1.0, z, z, **_SOIL)
    want = float(_closed_form_kernel(1.0, z, z, **_SOIL))
    assert got == pytest.approx(want, rel=_TOL), (z, got, want)


@pytest.mark.parametrize("half_gap", [1e-6, 1e-4, 1e-2, 0.1, 1.0])
def test_pair_straddling_the_interface_is_truncation_bounded(
    half_gap: float,
) -> None:
    """The cross-layer transmitted term degenerates too ($z_l - z_u \\to 0$).

    Not in the auditor's table but the same defect class, and it is the
    geometry of every rod that crosses $h_1$: the *correction* is worse
    than the kernel here, because the homogeneous baseline contributes
    its own $1/R$ direct term through the reflected amplitudes (where
    ``_primary_rho`` does not see it).
    """
    z, z_s = 5.0 - half_gap, 5.0 + half_gap
    got = two_layer_real_space_kernel(1.0, z, z_s, **_SOIL)
    want = float(_closed_form_kernel(1.0, z, z_s, **_SOIL))
    assert got == pytest.approx(want, rel=_TOL), (half_gap, got, want)
    got_c = two_layer_layered_correction_real_space(1.0, z, z_s, **_SOIL)
    want_c = float(_closed_form_correction(1.0, z, z_s, **_SOIL))
    assert got_c == pytest.approx(want_c, rel=_TOL), (half_gap, got_c, want_c)


@pytest.mark.parametrize("rho_2", [1.0, 10000.0])
def test_extreme_contrast_group_meets_the_documented_kernel_tolerance(
    rho_2: float,
) -> None:
    """K = ±0.98 overshoot (re-audit item 2, the part that is F14).

    rho_1 = 100, h_1 = 6.096 m, z = z_s = 6 m: the first interface image
    sits at d = 0.192 m, so at 0.15.0.dev
    $\\lambda_\\text{max} d_\\text{img} = 32.8 \\times 0.192 = 6.3$ and
    the group was 2.4e-4 … 2.9e-4 off the closed form over
    s = 0 … 90 m — 2.4–2.9x the documented 1e-4, at points carrying
    57–73 % of the group's max |dG| (so "near-cancellation" is no
    excuse).
    """
    s_values = np.linspace(0.5, 90.0, 400)
    kw = dict(rho_1=100.0, rho_2=rho_2, h_1=6.096, rho_baseline=100.0)
    got = two_layer_layered_correction_group(s_values, 6.0, 6.0, **kw)
    want = _closed_form_correction(s_values, 6.0, 6.0, **kw)
    err = np.abs(got - want) / np.abs(want).max()
    assert err.max() < 1e-6, (rho_2, s_values[int(err.argmax())], err.max())


@pytest.mark.parametrize(
    "z,s",
    [
        (0.0, 10.0), (0.05, 10.0), (5.0, 1.0), (5.0005, 1.0),
        (6.0, 200.0), (0.7, 4000.0), (0.7, 20000.0),
    ],
)
def test_lambda_max_factor_is_inert_over_five_orders(
    z: float, s: float,
) -> None:
    """The public knob must not change the answer — in either direction.

    This is the whole of F14 in one assertion, and it is checked on the
    geometries that used to fail rather than on the one comfortable
    (s = 200, z = 6) point the previous test suite used. At 0.15.0.dev
    the sweep 30 … 30 000 spanned +6.28e-3 … +3.75e-4 with sign changes
    at z = z_s = 0, s = 10 m alone.
    """
    values = [
        two_layer_real_space_kernel(
            s, z, z, lambda_max_factor=f, **_SOIL,
        )
        for f in (30.0, 100.0, 200.0, 1000.0, 10000.0, 30000.0)
    ]
    want = float(_closed_form_kernel(s, z, z, **_SOIL))
    for f, v in zip((30.0, 100.0, 200.0, 1000.0, 10000.0, 30000.0), values):
        assert v == pytest.approx(want, rel=1e-6), (f, v, want)
    spread = (max(values) - min(values)) / abs(want)
    assert spread < 1e-7, (values, spread)


@pytest.mark.parametrize("s", [1000.0, 4000.0, 20000.0])
@pytest.mark.parametrize("h_1", [1.0, 5.0])
def test_kilometre_spans_are_accurate_and_bounded_in_cost(
    s: float, h_1: float,
) -> None:
    """Remote injection (AP1) — accuracy *and* the panel-cost bound.

    Accuracy: v0.14.1 was −6.6 % at s = 6 km (h_1 = 5 m) and −1.15 % at
    s = 1 km (h_1 = 1 m). Cost: the 0.15.0.dev fix removed the wrong
    4096-panel accuracy cap but put nothing in its place, so the node
    count grew like ``lambda_max * s`` and a 4 km pair cost 42x a 90 m
    one. The node count is asserted here rather than a wall-clock time
    so the guard is deterministic and machine-independent.
    """
    kw = dict(rho_1=100.0, rho_2=1000.0, h_1=h_1)
    got = two_layer_real_space_kernel(s, 0.7, 0.7, **kw)
    want = float(_closed_form_kernel(s, 0.7, 0.7, **kw))
    assert got == pytest.approx(want, rel=1e-6), (s, h_1, got, want)

    plan = lg._plan_truncation(
        z=0.7, z_s=0.7, rho_1=100.0, rho_2=1000.0, h_1=h_1,
        rho_baseline=None, s_near=s, s_far=s, lambda_max_factor=200.0,
        warn_stacklevel=3,
    )
    lambdas, _ = lg._hankel_lambda_grid(
        lambda_max=plan.lambda_max, char_length=plan.char_length, s=s,
        n_log=32, n_lin=96,
    )
    # One panel per J0 period at the *capped* lambda_max, i.e. at most
    # _HANKEL_PANEL_TARGET panels — v0.14.1 allowed 4096 and 0.15.0.dev
    # allowed 38 197 for the 6 km case.
    assert lambdas.size <= 32 + lg._HANKEL_PANEL_ORDER * (
        lg._HANKEL_PANEL_TARGET + 1
    ), (s, h_1, lambdas.size)


def test_panel_cost_target_scales_with_the_knob() -> None:
    """Raising ``lambda_max_factor`` must buy quadrature, not silence.

    The cost target is proportional to the knob, so a convergence check
    genuinely moves work from the closed-form image sum into the
    integral (and must still not change the answer — asserted by
    :func:`test_lambda_max_factor_is_inert_over_five_orders`).
    """
    plans = [
        lg._plan_truncation(
            z=0.7, z_s=0.7, rho_1=100.0, rho_2=1000.0, h_1=5.0,
            rho_baseline=None, s_near=4000.0, s_far=4000.0,
            lambda_max_factor=f, warn_stacklevel=3,
        )
        for f in (200.0, 800.0)
    ]
    assert plans[1].lambda_max > 3.5 * plans[0].lambda_max
    # More quadrature range means fewer terms have to be split off.
    assert plans[1].coeffs.size < plans[0].coeffs.size


# =====================================================================
# F14 re-audit — the guarantee itself
# =====================================================================
def test_every_retained_image_term_is_decayed_at_lambda_max() -> None:
    """``lambda_max * d_rest >= 30`` is the load-bearing invariant.

    The truncated tail is bounded by
    ``sum_j |c_j| exp(-lambda_max d_j) / (2 d_j)``, so this product —
    and nothing about ``min(h_1, s + z + z_s)`` — is what makes
    truncation legitimate. Checked over the whole degenerate zoo,
    including several ``h_1`` and both signs of K.
    """
    for h_1 in (0.5, 1.0, 5.0, 6.096):
        for rho_2 in (1.0, 30.0, 1000.0, 10000.0):
            for z in (0.0, 1e-6, 0.7, h_1 - 1e-6, h_1, h_1 + 1e-6,
                      h_1 + 0.05, 2.0 * h_1):
                for s in (0.0, 1.0, 90.0, 4000.0):
                    plan = lg._plan_truncation(
                        z=z, z_s=z, rho_1=100.0, rho_2=rho_2, h_1=h_1,
                        rho_baseline=None, s_near=s, s_far=s,
                        lambda_max_factor=200.0, warn_stacklevel=3,
                    )
                    assert (
                        plan.d_rest * plan.lambda_max
                        >= lg._HANKEL_TAIL_DECAY_TARGET - 1e-9
                    ), (h_1, rho_2, z, s, plan.d_rest, plan.lambda_max)


def test_image_split_never_skips_the_degenerate_generation() -> None:
    """The distance floor must be the infimum over *all* later generations.

    For two points in layer 1 generation 0 contributes only $z + z_s$
    while generation 1 contains $2h_1 - z - z_s$, which is the
    degenerate term. Using generation 0's own minimum as the loop bound
    skips it silently — the first version of this fix did exactly that
    and was still +4.1 % at ``z = z_s = h_1``.
    """
    # z = z_s = h_1: gen 0 sits at 2 h_1 = 10 m, gen 1 contains d = 0.
    assert lg._image_distance_floor(0, 5.0, 5.0, 5.0) == 0.0
    coeffs, dists, d_rest, exhausted = lg._reflected_image_terms(
        5.0, 5.0, rho_1=100.0, rho_2=1000.0, h_1=5.0,
        rho_baseline=None, d_cut=0.75,
    )
    assert not exhausted
    assert dists.min() == pytest.approx(0.0)
    assert d_rest == pytest.approx(10.0)
    # And the coefficient is the interface reflection K * rho_1.
    K = (1000.0 - 100.0) / (1000.0 + 100.0)
    assert coeffs[int(np.argmin(dists))] == pytest.approx(100.0 * K)


def test_image_terms_cancel_for_a_matched_baseline() -> None:
    """A matched ``rho_baseline`` must cancel the surface image exactly.

    Otherwise the correction would carry two closed-form terms that only
    cancel to rounding — which at a surface pair (d = 0, clamped 1/R)
    would be a catastrophic-cancellation trap.
    """
    coeffs, dists, _, _ = lg._reflected_image_terms(
        0.0, 0.0, rho_1=100.0, rho_2=1000.0, h_1=5.0,
        rho_baseline=100.0, d_cut=1.0,
    )
    assert coeffs.size == 0 and dists.size == 0


def test_tail_truncation_warns_when_the_term_budget_is_exhausted(
    monkeypatch,
) -> None:
    """Silence is the one unacceptable outcome (the point of F14).

    The analytic split closes every reachable geometry, so this guard is
    exercised by shrinking the term budget — which is exactly the
    condition (very thin top layer, |K| -> 1, multi-kilometre span) the
    warning documents.
    """
    monkeypatch.setattr(lg, "_HANKEL_IMAGE_TERM_BUDGET", 4)
    with pytest.warns(SommerfeldTailTruncationWarning) as record:
        two_layer_real_space_kernel(
            2000.0, 0.2, 0.2, rho_1=100.0, rho_2=10000.0, h_1=0.4,
        )
    message = str(record[0].message)
    assert "lambda_max_factor" in message
    assert "image terms" in message


@pytest.mark.parametrize(
    "call",
    [
        lambda: two_layer_real_space_kernel(1.0, -0.1, 0.5, **_SOIL),
        lambda: two_layer_real_space_kernel(1.0, 0.5, -0.1, **_SOIL),
        lambda: two_layer_layered_correction_real_space(
            0.0075, -1.759, -1.759, **_SOIL,
        ),
        lambda: two_layer_layered_correction_group(
            np.array([1.0, 2.0]), -1.0, 1.0, **_SOIL,
        ),
        lambda: two_layer_probe_matrix(
            np.array([[0.0, 0.0, -1.0]]), np.array([[1.0, 0.0, 0.5]]),
            **_SOIL,
        ),
        lambda: two_layer_real_space_kernel(
            1.0, float("nan"), 0.5, **_SOIL,
        ),
    ],
)
def test_negative_depth_raises_instead_of_returning_nan(call) -> None:
    """``z < 0`` used to return ``nan`` and propagate to the impedance.

    Reproduced on v0.14.1 and on 0.15.0.dev with
    ``two_layer_layered_correction_real_space(s=0.0075, z=z_s=-1.759)``.
    The producer is the diagonal Gauss-Legendre averaging in
    ``image_2layer`` (``z_lo = z_i - 0.5 * L_i``), which is a caller bug
    reported separately; this module's job is to refuse the input rather
    than to invent a number for it.
    """
    with pytest.raises(ValueError, match="depths must be|non-finite"):
        call()


def test_warning_stacklevel_points_at_the_caller(monkeypatch) -> None:
    """``stacklevel`` must resolve to user code for *every* entry point.

    At 0.15.0.dev the fixed ``stacklevel=3`` resolved to
    ``layered_green.py`` itself for :func:`two_layer_probe_matrix` (one
    frame deeper), which also mis-keys the default once-per-location
    warning de-duplication — a user could silence the warning for their
    own code by triggering it once from a different call site.
    """
    monkeypatch.setattr(lg, "_HANKEL_IMAGE_TERM_BUDGET", 4)
    thin = dict(rho_1=100.0, rho_2=10000.0, h_1=0.4)
    calls = [
        lambda: two_layer_real_space_kernel(2000.0, 0.2, 0.2, **thin),
        lambda: two_layer_layered_correction_real_space(
            2000.0, 0.2, 0.2, **thin,
        ),
        lambda: two_layer_layered_correction_group(
            np.linspace(1.0, 2000.0, 100), 0.2, 0.2, **thin,
        ),
        lambda: two_layer_probe_matrix(
            np.array([[0.0, 0.0, 0.2], [2000.0, 0.0, 0.2]]),
            np.array([[1.0, 0.0, 0.2]]), **thin,
        ),
    ]
    for i, call in enumerate(calls):
        with pytest.warns(SommerfeldTailTruncationWarning) as record:
            call()
        assert record[0].filename == __file__, (i, record[0].filename)


# =====================================================================
# F15 re-audit — chunking and interpolation stay range-independent
# =====================================================================
def test_group_is_independent_of_the_matmul_chunk_budget(monkeypatch) -> None:
    """Blocking the ``J0`` contraction must not move the result.

    Varied over five budgets down to one row per block, on the
    degenerate geometries as well — the closed-form image terms are
    added *outside* the contraction, so a chunk boundary must not
    interact with them.
    """
    s_values = np.linspace(0.0, 400.0, 777)
    for z in (0.0, 0.7, 5.0, 6.0):
        reference = None
        for budget in (512 * 1024 * 1024, 1 << 20, 1 << 14, 1 << 10, 8):
            monkeypatch.setattr(lg, "_HANKEL_MATMUL_BUDGET_BYTES", budget)
            out = two_layer_layered_correction_group(s_values, z, z, **_SOIL)
            if reference is None:
                reference = out
            else:
                np.testing.assert_allclose(
                    out, reference, rtol=1e-11, atol=1e-11,
                )


@pytest.mark.parametrize("n_pairs", [63, 64, 65, 66, 96, 129])
@pytest.mark.parametrize("z", [0.0, 5.0, 6.0])
def test_branch_switch_is_not_observable(n_pairs: int, z: float) -> None:
    """The same distance must give the same number at any group size.

    The interpolation threshold is a *problem-size* switch, so a step
    there is indefensible; and it must hold for the degenerate
    geometries too, where the closed-form terms are added per pair on
    one side of the switch and per pair on the other.
    """
    s_values = np.linspace(0.0, 200.0, n_pairs)
    got = two_layer_layered_correction_group(s_values, z, z, **_SOIL)
    want = _closed_form_correction(
        s_values, z, z,
        r_min=1.0 / (200.0 / max(min(5.0, 2.0 * z + 1e-9), 1e-3)),
        **_SOIL,
    )
    scale = np.abs(want).max()
    assert np.abs(got - want).max() / scale < 1e-5, (
        n_pairs, z, np.abs(got - want).max() / scale,
    )
