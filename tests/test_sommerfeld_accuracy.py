"""Accuracy of the Sommerfeld earth-return kernel (review pass 10).

* 10.36 — the lambda grid resolves J0 for distant pairs and no longer
  truncates silently;
* 10.35 — the buried kernel is exact for n >= 3 layers and for
  conductors below the first interface;
* 10.46 — F50 (sign of the layered point kernel), exact removal of the
  additive mirror (vertical and long segments), F52 (segments crossing
  the soil surface);
* 10.60 — the multi-frequency assembly is bit-identical to the
  single-frequency one.

Reference values for the far pairs come from an independent quadrature
(``deliver/evidence/coup1/e04_far_pair_reference.py``, the method of
review script R3/t05; no groundfield code involved).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.coupling.inductance import build_inductance_matrix
from groundfield.coupling.sommerfeld_inductance import (
    LayeredEarth,
    build_sommerfeld_correction_matrix,
    earth_return_correction_layered,
    sommerfeld_pair_integral_homogeneous,
    sommerfeld_pair_integral_layered,
)

OMEGA_1K = 2.0 * math.pi * 1000.0


def _seg(x0: float, x1: float, z: float, y: float = 0.0):
    return np.array([x0, y, z]), np.array([x1, y, z])


# ---------------------------------------------------------------------
# 10.36 — lambda grid
# ---------------------------------------------------------------------

# Collinear 10 m segments at h = 1 m, 10 Ohm m, 1 kHz; centre distance
# D -> independent reference (additive-image baseline), e04_output.txt.
_FAR_PAIR_REFERENCE = {
    100.0: 1.4696340e-04 - 1.1865504e-03j,
    200.0: 2.1669450e-05 - 6.2995070e-04j,
    400.0: 2.3864389e-06 - 3.1427961e-04j,
}


@pytest.mark.finding("10.36")
@pytest.mark.parametrize("distance", sorted(_FAR_PAIR_REFERENCE))
def test_distant_pair_entry_matches_independent_quadrature(distance: float) -> None:
    """v0.15.0: Re -24 % at 200 m, +274 % at 400 m; Im +2.35 % at 400 m."""
    a1, a2 = _seg(0.0, 10.0, 1.0)
    b1, b2 = _seg(distance, distance + 10.0, 1.0)
    v = sommerfeld_pair_integral_homogeneous(
        a1, a2, b1, b2, omega=OMEGA_1K, sigma_earth=0.1
    )
    ref = _FAR_PAIR_REFERENCE[distance]
    assert v.real == pytest.approx(ref.real, rel=1e-4)
    assert v.imag == pytest.approx(ref.imag, rel=1e-5)


@pytest.mark.finding("10.36")
def test_lambda_grid_budget_warns_instead_of_truncating() -> None:
    """Beyond the node budget the grid warns (v0.15.0: silent 4096 cap)."""
    from groundfield.coupling import sommerfeld_inductance as si
    from groundfield.coupling.layered_green import SommerfeldResolutionWarning

    with pytest.warns(SommerfeldResolutionWarning, match="NOT converged"):
        si._build_lambda_grid(
            z_sum=1.4, rho_max=200_000.0, omega=OMEGA_1K, sigma_top=0.01
        )


@pytest.mark.finding("10.36")
def test_lambda_grid_log_region_is_oscillation_free() -> None:
    """The log region ends below pi / (4 rho_max) (F14 policy)."""
    from groundfield.coupling import sommerfeld_inductance as si

    for rho_max in (5.0, 50.0, 400.0, 2000.0):
        lam, _ = si._build_lambda_grid(
            z_sum=1.4, rho_max=rho_max, omega=OMEGA_1K, sigma_top=0.01
        )
        # the 64 log nodes come first
        assert lam[:64].max() * rho_max <= 0.25 * math.pi * (1.0 + 1e-12)
        # linear panels: at least 16 nodes per J0 period
        span = lam.max() - lam[:64].max()
        n_lin = lam.size - 64
        assert n_lin >= 16 * span * rho_max / (2.0 * math.pi) - 16


# ---------------------------------------------------------------------
# 10.35 — exact layered kernel
# ---------------------------------------------------------------------


@pytest.mark.finding("10.35")
@pytest.mark.parametrize("distance", [0.0, 30.0, 300.0])
def test_three_layer_duplicate_equals_two_layer(distance: float) -> None:
    """rho_2 == rho_3 must reproduce the two-layer earth (v0.15.0: 10-25 %)."""
    e2 = LayeredEarth(rhos=(1000.0, 30.0), thicknesses=(5.0,))
    e3 = LayeredEarth(rhos=(1000.0, 30.0, 30.0), thicknesses=(5.0, 20.0))
    a1, a2 = _seg(0.0, 10.0, 0.7)
    b1, b2 = _seg(distance, distance + 10.0, 0.7)
    z2 = sommerfeld_pair_integral_layered(a1, a2, b1, b2, omega=OMEGA_1K, earth=e2)
    z3 = sommerfeld_pair_integral_layered(a1, a2, b1, b2, omega=OMEGA_1K, earth=e3)
    assert z3 == pytest.approx(z2, rel=1e-12)


@pytest.mark.finding("10.35")
@pytest.mark.parametrize("distance", [0.0, 20.0])
def test_conductor_below_first_interface(distance: float) -> None:
    """A duplicated top layer must not change a conductor in layer 2.

    (100, 100, 10) Ohm m over (0.5, 2.5) m is the earth (100, 10) over
    3 m; a conductor at 0.7 m lies in layer 2 of the first description
    and in layer 1 of the second. v0.15.0 used the single air-image
    approximation for the first one.
    """
    e3 = LayeredEarth(rhos=(100.0, 100.0, 10.0), thicknesses=(0.5, 2.5))
    e2 = LayeredEarth(rhos=(100.0, 10.0), thicknesses=(3.0,))
    a1, a2 = _seg(0.0, 10.0, 0.7)
    b1, b2 = _seg(distance, distance + 10.0, 0.7)
    z3 = sommerfeld_pair_integral_layered(a1, a2, b1, b2, omega=OMEGA_1K, earth=e3)
    z2 = sommerfeld_pair_integral_layered(a1, a2, b1, b2, omega=OMEGA_1K, earth=e2)
    assert z3 == pytest.approx(z2, rel=1e-6)


@pytest.mark.finding("10.35")
def test_closed_forms_match_interface_solve() -> None:
    """Generalised-reflection closed forms == direct interface solve."""
    from groundfield.coupling import sommerfeld_inductance as si

    lam = np.concatenate([np.geomspace(1e-6, 1.0, 120), np.linspace(1.0, 40.0, 200)])
    earth = LayeredEarth(rhos=(300.0, 50.0, 1000.0, 20.0), thicknesses=(0.4, 2.0, 6.0))
    u = si._vertical_wavenumbers(lam, OMEGA_1K, earth)
    tops = (0.0, *earth.interface_depths)
    for k in range(1, earth.n_layers + 1):
        top = tops[k - 1]
        thick = earth.thicknesses[k - 1] if k < earth.n_layers else 3.0
        za, zb = top + 0.3 * thick, top + 0.8 * thick
        closed = si._same_layer_weight(lam, u, earth, k, za - top, zb - top)
        solved = si._generic_weight(lam, u, earth, za, zb)
        np.testing.assert_allclose(closed, solved, rtol=1e-12, atol=1e-14)
    mixed = si._mixed_layer1_weight(lam, u, earth, 0.2, 3.0)
    np.testing.assert_allclose(
        mixed, si._generic_weight(lam, u, earth, -3.0, 0.2), rtol=1e-12, atol=1e-14
    )


# ---------------------------------------------------------------------
# 10.46 — F50, image removal, F52
# ---------------------------------------------------------------------


@pytest.mark.finding("10.46", "F50")
def test_layered_point_kernel_has_the_exact_sign() -> None:
    """Earth (100, 10) Ohm m, h1 = 3 m, 1 kHz, z = z' = 0.6 m, rho = 0.

    Exact reflected term -6.3463e-3 - 4.5795e-3j (R3, independent
    derivation, t10); v0.15.0 returned +8.739e-3 + 6.484e-3j.
    """
    earth = LayeredEarth(rhos=(100.0, 10.0), thicknesses=(3.0,))
    corr = earth_return_correction_layered(
        rho=0.0, z_i=0.6, z_j=0.6, omega=OMEGA_1K, earth=earth
    )
    gamma = complex(np.sqrt(1j * OMEGA_1K * 4e-7 * math.pi / 100.0))
    reflected = corr - (-gamma - 1.0 / 1.2)
    assert reflected.real == pytest.approx(-6.3463e-3, rel=1e-4)
    assert reflected.imag == pytest.approx(-4.5795e-3, rel=1e-4)


@pytest.mark.finding("10.46")
@pytest.mark.parametrize(
    "eps",
    [
        # 50 m horizontal segments at 0.7 m (segment length >> 2h)
        np.array(
            [[[50.0 * k, 0.0, 0.7], [50.0 * (k + 1), 0.0, 0.7]] for k in range(4)]
        ),
        # vertical riser from 0.5 m to 2.5 m
        np.array(
            [[[0.0, 0.0, 0.5], [0.0, 0.0, 1.5]], [[0.0, 0.0, 1.5], [0.0, 0.0, 2.5]]]
        ),
    ],
    ids=["long_horizontal", "vertical_riser"],
)
def test_transparent_earth_cancels_the_additive_mirror(eps: np.ndarray) -> None:
    """sigma -> 0: L(additive) + dZ/(j w) must be the free-space L.

    v0.15.0: the riser kept twice the image (2.000x), the 50 m segments
    lost 2 % of it (the -1/R' removal was 16x16-quadrature-integrated).
    """
    omega = 2.0 * math.pi * 50.0
    radii = np.full(len(eps), 0.004)
    L_add = build_inductance_matrix(eps, radii, use_image=True)
    L_free = build_inductance_matrix(eps, radii, use_image=False)
    transparent = LayeredEarth(rhos=(1e16,), thicknesses=())
    dz = build_sommerfeld_correction_matrix(eps, radii, omega=omega, earth=transparent)
    total = L_add + (dz / (1j * omega)).real
    assert np.abs(total - L_free).max() <= 1e-6 * np.abs(L_free).max()


@pytest.mark.finding("10.46", "F52")
@pytest.mark.parametrize(
    "p1, p2",
    [((0.0, 0.0, 0.6), (0.0, 0.0, -3.0)), ((0.0, 0.0, 2.0), (0.0, 0.0, -2.0))],
    ids=["asymmetric_riser", "symmetric_riser"],
)
def test_surface_crossing_segment_is_rejected_with_the_right_message(p1, p2) -> None:
    """v0.15.0 accepted the first (dispatch by midpoint) and rejected the
    second as 'lying in the soil-surface plane'."""
    eps = np.array([[p1, p2]], dtype=float)
    with pytest.raises(ValueError, match="crosses the soil surface"):
        build_inductance_matrix(eps, np.array([0.01]))


@pytest.mark.finding("10.46", "F52")
@pytest.mark.parametrize("model", ["perfect_mirror", "carson_series"])
def test_world_solve_rejects_surface_crossing_inductive_conductor(model) -> None:
    """v0.15.0 solved this silently (dispatch by segment midpoint).

    Only the image conventions reject it; ``sommerfeld`` assembles on the
    image-free baseline and accepts such a conductor
    (``tests/test_inductive_solver_integration.py``).
    """
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5), length=2.0)
    gf.create_electrode(w, "rod", name="g2", position=(20, 0, 0.5), length=2.0)
    c = gf.create_conductor(
        w,
        name="lead",
        start=(0.0, 0.0, 0.5),
        end=(20.0, 0.0, -1.0),
        cross_section=50e-6,
        discretize_segment_length=5.0,
        inductance_model="neumann",
    )
    w.conductors[-1] = c.model_copy(
        update={"start_electrode": "g1", "end_electrode": "g2"}
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(
        backend="image",
        segment_length=0.5,
        frequencies=[50.0],
        earth_inductive_model=model,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="crosses the soil surface"):
            w.solve(eng)


@pytest.mark.finding("10.46", "F52")
def test_surface_crossing_pair_is_split_at_the_surface() -> None:
    """With the exact baseline a riser is the sum of its two halves."""
    earth = LayeredEarth(rhos=(100.0,), thicknesses=())
    riser = (np.array([0.0, 0.0, 0.6]), np.array([0.0, 0.0, -3.0]))
    buried = (np.array([0.0, 0.0, 0.6]), np.array([0.0, 0.0, 0.0]))
    air = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -3.0]))
    lead = (np.array([1.0, 0.0, 0.5]), np.array([1.0, 0.0, -2.0]))
    whole = sommerfeld_pair_integral_layered(
        *riser, *lead, omega=OMEGA_1K, earth=earth, image="none"
    )
    parts = sum(
        sommerfeld_pair_integral_layered(
            *part, *half, omega=OMEGA_1K, earth=earth, image="none"
        )
        for part in (buried, air)
        for half in (
            (lead[0], np.array([1.0, 0.0, 0.0])),
            (np.array([1.0, 0.0, 0.0]), lead[1]),
        )
    )
    assert whole == pytest.approx(parts, rel=1e-12)


# ---------------------------------------------------------------------
# 10.20 preparation — image conventions are consistent
# ---------------------------------------------------------------------


@pytest.mark.finding("10.20")
@pytest.mark.parametrize("model", ["sommerfeld", "carson"])
def test_total_impedance_does_not_depend_on_the_image_convention(model: str) -> None:
    """j w L(image) + dZ(image) is the same physical matrix for every image."""
    from groundfield.coupling.inductance import build_carson_correction_matrix

    eps = np.array(
        [
            [[0.0, 0.0, 0.7], [10.0, 0.0, 0.7]],
            [[10.0, 0.0, 0.7], [20.0, 0.0, 0.7]],
            [[0.0, 3.0, 0.8], [20.0, 3.0, 0.8]],
            [[20.0, 0.0, 0.7], [20.0, 0.0, 2.7]],
        ]
    )
    radii = np.full(len(eps), 0.005)
    earth = LayeredEarth(rhos=(100.0, 30.0), thicknesses=(5.0,))
    totals = {}
    for image in ("none", "additive", "antiparallel"):
        L = build_inductance_matrix(eps, radii, image=image)
        if model == "sommerfeld":
            dz = build_sommerfeld_correction_matrix(
                eps, radii, omega=OMEGA_1K, earth=earth, image=image
            )
        else:
            dz = build_carson_correction_matrix(
                eps, radii, omega=OMEGA_1K, sigma_earth=0.01, image=image
            )
        totals[image] = 1j * OMEGA_1K * L + dz
    scale = np.abs(totals["none"]).max()
    for image in ("additive", "antiparallel"):
        assert np.abs(totals[image] - totals["none"]).max() <= 1e-12 * scale


@pytest.mark.finding("10.20", "F32")
@pytest.mark.parametrize("length", [100.0, 1000.0])
@pytest.mark.parametrize("freq", [50.0, 1000.0])
def test_carson_on_the_antiparallel_baseline_matches_carson(length, freq) -> None:
    """Carson's correction on its own (anti-parallel) image baseline.

    The per-metre reactance of a straight buried wire equals Carson's
    closed form with GMR = a e^{-1/4} (internal inductance included,
    ``include_internal=True``), independent of the length. With the
    additive baseline the solver assembles today it is 1.66-2.18x.
    """
    from groundfield.coupling.inductance import build_carson_correction_matrix
    from groundfield.references.earth_return import carson_self_impedance

    a, h, rho = 0.004, 0.7, 100.0
    n = int(length / 10.0)
    x = np.linspace(0.0, length, n + 1)
    eps = np.array([[[x[k], 0, h], [x[k + 1], 0, h]] for k in range(n)])
    radii = np.full(n, a)
    om = 2.0 * math.pi * freq
    L = build_inductance_matrix(eps, radii, image="antiparallel")
    dz = build_carson_correction_matrix(eps, radii, omega=om, sigma_earth=1.0 / rho)
    z = (1j * om * L.sum() + dz.sum()) / length
    ref = carson_self_impedance(rho, freq, gmr=a * math.exp(-0.25))
    assert z.imag == pytest.approx(ref.imag, rel=2e-3)


# ---------------------------------------------------------------------
# 10.60 — multi-frequency assembly
# ---------------------------------------------------------------------


@pytest.mark.finding("10.60")
def test_multifrequency_matrices_are_bit_identical() -> None:
    from groundfield.coupling.sommerfeld_inductance import (
        build_sommerfeld_correction_matrices,
    )

    eps = np.array(
        [
            [[0.0, 0.0, 0.7], [8.0, 0.0, 0.7]],
            [[8.0, 0.0, 0.7], [8.0, 9.0, 0.7]],
            [[40.0, 2.0, 0.6], [55.0, 2.0, 0.6]],
            [[300.0, 0.0, 0.7], [310.0, 0.0, 0.7]],
        ]
    )
    earth = LayeredEarth(rhos=(200.0, 50.0), thicknesses=(5.0,))
    omegas = [2.0 * math.pi * f for f in (0.0, 16.7, 50.0, 550.0, 1000.0)]
    multi = build_sommerfeld_correction_matrices(eps, None, omegas=omegas, earth=earth)
    for k, om in enumerate(omegas):
        single = build_sommerfeld_correction_matrix(eps, None, omega=om, earth=earth)
        assert np.array_equal(multi[k], single)


@pytest.mark.finding("10.36")
def test_conductor_on_an_interface_is_resolved_without_warnings() -> None:
    """A PEN exactly on a layer interface has a vanishing decay length.

    lambda_max = 30/z alone then forces ~10^6 nodes for distant pairs and
    the node budget binds; the oscillatory tail bound caps lambda_max.
    The result is continuous with the PEN 1 cm below the interface.
    Guard for the new grid: v0.15.0 passes it only because it truncated
    at 4096 panels without a check.
    """
    from groundfield.coupling.layered_green import SommerfeldResolutionWarning

    on = LayeredEarth(rhos=(100.0, 30.0), thicknesses=(0.7,))
    below = LayeredEarth(rhos=(100.0, 30.0), thicknesses=(0.69,))
    a1, a2 = _seg(0.0, 10.0, 0.7)
    for distance in (100.0, 400.0):
        b1, b2 = _seg(distance, distance + 10.0, 0.7)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SommerfeldResolutionWarning)
            z_on = sommerfeld_pair_integral_layered(
                a1, a2, b1, b2, omega=OMEGA_1K, earth=on
            )
        z_below = sommerfeld_pair_integral_layered(
            a1, a2, b1, b2, omega=OMEGA_1K, earth=below
        )
        assert z_on == pytest.approx(z_below, rel=1e-3)


@pytest.mark.finding("10.60")
def test_multifrequency_with_frequency_dependent_grids() -> None:
    """Deep conductors in very conductive soil: p_skin > 1/z at the higher
    frequencies, so the lambda grid differs per frequency."""
    from groundfield.coupling.sommerfeld_inductance import (
        build_sommerfeld_correction_matrices,
    )

    eps = np.array(
        [[[0.0, 0.0, 10.0], [10.0, 0.0, 10.0]], [[30.0, 0.0, 10.0], [40.0, 0.0, 10.0]]]
    )
    earth = LayeredEarth(rhos=(0.2,), thicknesses=())
    omegas = [2.0 * math.pi * f for f in (5.0, 50.0, 1000.0)]
    multi = build_sommerfeld_correction_matrices(eps, None, omegas=omegas, earth=earth)
    for k, om in enumerate(omegas):
        single = build_sommerfeld_correction_matrix(eps, None, omega=om, earth=earth)
        assert np.array_equal(multi[k], single)
    empty = build_sommerfeld_correction_matrices(
        np.zeros((0, 2, 3)), None, omegas=omegas, earth=earth
    )
    assert empty.shape == (3, 0, 0)
