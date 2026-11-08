"""Validity domain of the ``carson_series`` earth-return model.

Carson's correction is a per-metre result for long, parallel, horizontal
wires. Since audit round 2 of the calendar release:

* pairs on **opposite sides** of the soil surface (an overhead lead over
  a buried PEN) use the field transmitted into the soil,
  :func:`~groundfield.coupling.carson.carson_mixed_mutual_correction`
  (their anti-parallel image baseline is zero: a perfect conductor
  screens the soil). Up to then the buried wire was treated as an
  overhead wire at height ``|z|`` on the mirrored-endpoint image: -36 %
  mutual reactance for a lead 1 m above a PEN at 0.7 m depth (audit A2C-1);
* oblique pairs, and every pair or self term of an inclined or vertical
  segment, use the exact Sommerfeld kernel of a homogeneous earth. Up to
  then oblique pairs got ``sqrt(l_i l_j) |cos phi|`` per segment pair,
  which grows with the number of segments (a probe lead turning 8.1 deg
  against the feed lead changed the reading 17.6-fold; stream DOC-B).

References: an mpmath quadrature of the transmitted-field integral
(written here independently, interval-wise for x > 0) and groundfield's
own Sommerfeld kernel.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest

from groundfield.coupling.carson import (
    MU_0,
    carson_mixed_mutual_correction,
    carson_mutual_correction,
)
from groundfield.coupling.inductance import (
    build_carson_correction_matrices,
    build_inductance_matrix,
)
from groundfield.coupling.sommerfeld_inductance import (
    LayeredEarth,
    build_sommerfeld_correction_matrices,
)
from groundfield.solver._inductive import IMAGE_OF_EARTH_MODEL

RHO = 100.0
SIG = 1.0 / RHO


def _reference_mixed(h: float, d: float, x: float, f: float, rho: float) -> complex:
    """(j w mu0 / pi) int exp(-lam h - u d) / (lam + u) cos(lam x) dlam."""
    mp.mp.dps = 20
    om = 2.0 * math.pi * f
    g2 = 1j * om * MU_0 / rho

    def integrand(lam):
        u = mp.sqrt(lam * lam + g2)
        return mp.exp(-lam * h - u * d) / (lam + u) * mp.cos(lam * x)

    k = math.sqrt(om * MU_0 / rho)
    lam_end = 45.0 / (h + d)  # exp(-45) ~ 3e-20
    cuts = sorted({0.0, 1e-3 * k, 1e-2 * k, 0.1 * k, k, 10.0 * k, 1.0 / (h + d)})
    cuts = [c for c in cuts if c < lam_end] + [lam_end]
    if x > 0.0:  # resolve cos(lam x): interval-wise, half a period at most
        fine = np.arange(0.0, lam_end, math.pi / x).tolist()
        cuts = sorted(set(cuts) | set(fine))
    val = mp.fsum(
        mp.quad(integrand, [a, b]) for a, b in zip(cuts[:-1], cuts[1:], strict=True)
    )
    return complex(1j * om * MU_0 / math.pi * val)


def _wire(z: float, y: float, length: float, n: int) -> np.ndarray:
    xs = np.linspace(0.0, length, n + 1)
    return np.array([[[xs[k], y, z], [xs[k + 1], y, z]] for k in range(n)])


def _carson_series_per_metre(z1: float, z2: float, x: float, f: float) -> complex:
    """Per-metre mutual impedance as the solver assembles carson_series."""
    image = IMAGE_OF_EARTH_MODEL["carson_series"]
    om = 2.0 * math.pi * f
    vals = []
    for length, n in ((2000.0, 40), (1000.0, 20)):
        seg = np.concatenate([_wire(z1, 0.0, length, n), _wire(z2, x, length, n)])
        radii = np.full(seg.shape[0], 0.005)
        L = build_inductance_matrix(seg, radii, image=image)
        dZ = build_carson_correction_matrices(
            seg, radii, omegas=[om], sigma_earth=SIG, image=image
        )[0]
        vals.append((1j * om * L + dZ)[:n, n:].sum())
    return (vals[0] - vals[1]) / 1000.0


# ---------------------------------------------------------------------
# Opposite sides of the surface (audit A2C-1)
# ---------------------------------------------------------------------


# audit A2C-1
@pytest.mark.finding("10.20")
@pytest.mark.parametrize(
    "h, d, x, f",
    [
        (1.0, 0.7, 0.0, 1000.0),  # base: -35.9 % reactance
        (1.0, 0.7, 1.0, 1000.0),  # base: -13.5 %
        (0.1, 0.7, 0.5, 1000.0),  # generator surface lead; base: -3.4 %
        (1.0, 0.7, 1.0, 50.0),  # base: -10.3 %
        (8.0, 0.7, 2.0, 1000.0),  # base: -5.0 %
    ],
)
def test_overhead_wire_over_buried_wire_matches_the_transmitted_field(
    h: float, d: float, x: float, f: float
) -> None:
    """Per-metre mutual impedance of an overhead wire (z = -h) parallel to a
    buried one (z = d), as the solver assembles carson_series."""
    z = _carson_series_per_metre(-h, d, x, f)
    ref = _reference_mixed(h, d, x, f, RHO)
    assert z.imag == pytest.approx(ref.imag, rel=1e-6)
    assert z.real == pytest.approx(ref.real, rel=1e-6)


# audit A2C-1
@pytest.mark.finding("10.20")
@pytest.mark.parametrize(
    "h, d, x, f, rho",
    [
        (1.0, 0.7, 0.0, 50.0, 10.0),
        (0.1, 0.7, 0.5, 50.0, 3000.0),  # tiny q: QUADPACK's Fourier rule fails here
        (10.0, 10.0, 0.0, 5000.0, 10.0),  # remainder 38 % of Carson's J
        (1.0, 3.0, 200.0, 1000.0, 10.0),  # 82 oscillation panels
        (0.01, 0.05, 0.0, 1000.0, 100.0),
    ],
)
def test_mixed_mutual_correction_matches_an_independent_quadrature(
    h: float, d: float, x: float, f: float, rho: float
) -> None:
    om = 2.0 * math.pi * f
    z = carson_mixed_mutual_correction(om, h, d, x, 1.0 / rho)
    ref = _reference_mixed(h, d, x, f, rho)
    assert abs(z - ref) <= 1e-9 * abs(ref)


# audit A2C-1
@pytest.mark.finding("10.20")
def test_mixed_correction_tends_to_carson_as_the_depth_vanishes() -> None:
    om = 2.0 * math.pi * 1000.0
    carson = carson_mutual_correction(om, 1.0, 1e-9, 2.0, SIG)
    mixed = carson_mixed_mutual_correction(om, 1.0, 1e-9, 2.0, SIG)
    assert mixed == pytest.approx(carson, rel=1e-9)
    deeper = carson_mixed_mutual_correction(om, 1.0, 0.7, 2.0, SIG)
    assert abs(deeper - carson_mutual_correction(om, 1.0, 0.7, 2.0, SIG)) > 1e-3 * abs(
        deeper
    )


# audit A2C-1
@pytest.mark.finding("10.20")
def test_mixed_pair_agrees_with_the_sommerfeld_model_per_metre() -> None:
    """Two independent kernels: carson_series (per-metre transmitted field)
    and sommerfeld (3-D quasi-static Green's function), long wires at 1 kHz
    (skin depth 159 m), per metre from a 2000 m - 1000 m difference."""
    om = 2.0 * math.pi * 1000.0
    earth = LayeredEarth(rhos=(RHO,), thicknesses=())
    somm = []
    for length, n in ((2000.0, 40), (1000.0, 20)):
        seg = np.concatenate([_wire(-1.0, 0.0, length, n), _wire(0.7, 1.0, length, n)])
        radii = np.full(seg.shape[0], 0.005)
        L = build_inductance_matrix(seg, radii, image="none")
        dZ = build_sommerfeld_correction_matrices(
            seg, radii, omegas=[om], earth=earth, image="none", method="tabulated"
        )[0]
        somm.append((1j * om * L + dZ)[:n, n:].sum())
    z_somm = (somm[0] - somm[1]) / 1000.0
    z_carson = _carson_series_per_metre(-1.0, 0.7, 1.0, 1000.0)
    assert z_carson.imag == pytest.approx(z_somm.imag, rel=1e-3)
    assert z_carson.real == pytest.approx(z_somm.real, rel=1e-2)


# audit A2C-1
@pytest.mark.finding("10.20")
def test_antiparallel_baseline_of_a_pair_across_the_surface_is_zero() -> None:
    seg = np.array(
        [
            [[0.0, 0.0, -1.0], [10.0, 0.0, -1.0]],
            [[0.0, 0.0, 0.7], [10.0, 0.0, 0.7]],
            [[0.0, 2.0, 0.7], [10.0, 2.0, 0.7]],
        ]
    )
    L = build_inductance_matrix(seg, np.full(3, 0.005), image="antiparallel")
    assert L[0, 1] == 0.0 and L[0, 2] == 0.0
    assert L[1, 2] > 0.0  # same side: the PEC image baseline


def _mixed_geometry() -> np.ndarray:
    def seg(a, b):
        return [list(map(float, a)), list(map(float, b))]

    return np.array(
        [
            seg((0, 0, 0.7), (10, 0, 0.7)),  # buried chain
            seg((10, 0, 0.7), (20, 0, 0.7)),
            seg((0, 1, -1.0), (10, 1, -1.0)),  # overhead chain over it
            seg((10, 1, -1.0), (20, 1, -1.0)),
            seg((0, 3, -0.1), (10, 4.5, -0.1)),  # oblique surface lead
            seg((5, 0, 0.7), (15, 0, 0.7)),  # coaxial, overlapping the chain
            seg((20, 0, -0.1), (20, 0, -3.0)),  # vertical riser above ground
            seg((2, -2, 0.5), (8, -1, 3.0)),  # buried, inclined by 22 deg
        ]
    )


# audit A2C-1
@pytest.mark.finding("10.20")
@pytest.mark.parametrize("model", ["carson", "sommerfeld"])
def test_every_image_convention_gives_one_impedance(model: str) -> None:
    """jw L(image) + dZ(image) is the same matrix for none / additive /
    antiparallel — also for pairs across the surface, oblique, inclined,
    vertical and overlapping conductors."""
    seg = _mixed_geometry()
    radii = np.full(len(seg), 0.005)
    oms = [2.0 * math.pi * 50.0, 2.0 * math.pi * 1000.0]
    earth = LayeredEarth(rhos=(RHO,), thicknesses=())

    def correction(image: str) -> np.ndarray:
        if model == "carson":
            return build_carson_correction_matrices(
                seg, radii, omegas=oms, sigma_earth=SIG, image=image
            )
        return build_sommerfeld_correction_matrices(
            seg, radii, omegas=oms, earth=earth, image=image
        )

    totals = {}
    for image in ("none", "additive", "antiparallel"):
        L = build_inductance_matrix(seg, radii, image=image)
        dZ = correction(image)
        totals[image] = [1j * om * L + dZ[k] for k, om in enumerate(oms)]
    for image in ("additive", "antiparallel"):
        for k in range(len(oms)):
            ref = totals["none"][k]
            assert np.abs(totals[image][k] - ref).max() <= 1e-12 * np.abs(ref).max()


# ---------------------------------------------------------------------
# Oblique, inclined and vertical segments: the Sommerfeld kernel
# ---------------------------------------------------------------------


# audit A2C-1 / stream DOC-B
@pytest.mark.finding("10.20")
def test_pairs_outside_the_domain_get_the_sommerfeld_correction() -> None:
    """Oblique pairs and inclined / vertical segments (self terms too)
    carry exactly the Sommerfeld correction of a homogeneous earth."""
    seg = _mixed_geometry()
    radii = np.full(len(seg), 0.005)
    oms = [2.0 * math.pi * 1000.0]
    carson = build_carson_correction_matrices(
        seg, radii, omegas=oms, sigma_earth=SIG, image="antiparallel"
    )[0]
    somm = build_sommerfeld_correction_matrices(
        seg,
        radii,
        omegas=oms,
        earth=LayeredEarth(rhos=(RHO,), thicknesses=()),
        image="antiparallel",
    )[0]
    for i, j in ((4, 0), (4, 2), (6, 6), (7, 7), (7, 0), (6, 2), (4, 7)):
        # tabulated (carson fallback) against the per-pair rule
        assert carson[i, j] == pytest.approx(somm[i, j], rel=1e-4, abs=1e-12)
    # inside the domain Carson's own values remain
    assert carson[0, 0] != pytest.approx(somm[0, 0], rel=1e-3)


# stream DOC-B
@pytest.mark.finding("10.20")
def test_oblique_coupling_does_not_grow_with_the_number_of_segments() -> None:
    """Two 200 m conductors at 20 deg: the total mutual correction is a
    property of the geometry, not of the discretisation (base: about
    proportional to the number of segments)."""
    om = 2.0 * math.pi * 1000.0
    th = math.radians(20.0)

    def total(n: int) -> complex:
        a = _wire(0.7, 0.0, 200.0, n)
        t = np.linspace(0.0, 200.0, n + 1)
        b = np.array(
            [
                [[t[k] * math.cos(th), 3.0 + t[k] * math.sin(th), 0.7]]
                + [[t[k + 1] * math.cos(th), 3.0 + t[k + 1] * math.sin(th), 0.7]]
                for k in range(n)
            ]
        )
        seg = np.concatenate([a, b])
        dZ = build_carson_correction_matrices(
            seg, np.full(2 * n, 0.005), omegas=[om], sigma_earth=SIG
        )[0]
        return complex(dZ[:n, n:].sum())

    coarse, fine = total(4), total(16)
    assert abs(fine - coarse) <= 2e-3 * abs(coarse)


# stream DOC-B
@pytest.mark.finding("10.20")
@pytest.mark.parametrize("angle_deg", [9.0, 12.0, 30.0])
def test_oblique_leads_couple_as_in_the_sommerfeld_model(angle_deg: float) -> None:
    """Two 1 km overhead leads 2 m apart at 1 kHz, the second at an angle
    beyond the near-parallel bound (8.1 deg): carson_series gives the
    mutual impedance of the Sommerfeld kernel (base at 9 deg: about 20x,
    the oblique heuristic counted every segment pair). Inside the bound
    the per-metre tiling remains (7 deg: 11.8 % above the Sommerfeld
    model for these leads, which end only 6 skin depths long)."""
    om = 2.0 * math.pi * 1000.0
    n = 50
    th = math.radians(angle_deg)
    a = _wire(-1.0, 0.0, 1000.0, n)
    t = np.linspace(0.0, 1000.0, n + 1)
    b = np.array(
        [
            [[t[k] * math.cos(th), 2.0 + t[k] * math.sin(th), -1.0]]
            + [[t[k + 1] * math.cos(th), 2.0 + t[k + 1] * math.sin(th), -1.0]]
            for k in range(n)
        ]
    )
    seg = np.concatenate([a, b])
    radii = np.full(2 * n, 0.005)
    L_anti = build_inductance_matrix(seg, radii, image="antiparallel")
    dZ_c = build_carson_correction_matrices(seg, radii, omegas=[om], sigma_earth=SIG)
    z_carson = complex((1j * om * L_anti + dZ_c[0])[:n, n:].sum())
    L_none = build_inductance_matrix(seg, radii, image="none")
    dZ_s = build_sommerfeld_correction_matrices(
        seg,
        radii,
        omegas=[om],
        earth=LayeredEarth(rhos=(RHO,), thicknesses=()),
        image="none",
        method="tabulated",
    )
    z_somm = complex((1j * om * L_none + dZ_s[0])[:n, n:].sum())
    assert z_carson == pytest.approx(z_somm, rel=1e-9)


def _chain(a, b, n: int) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pts = a + np.linspace(0.0, 1.0, n + 1)[:, None] * (b - a)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _tiling_geometry() -> np.ndarray:
    """Overhead lead, buried PEN, an oblique branch, a riser, an inclined
    lead (the auditor's e08/e13 set) plus two leads 5 deg apart from one
    point and a drop 1 deg off its main."""
    return np.concatenate(
        [
            _chain((0, 0, -1.0), (60, 0, -1.0), 3),
            _chain((0, 0.5, 0.7), (60, 0.5, 0.7), 4),
            _chain((20, 0.5, 0.7), (50, 20, 0.7), 2),
            _chain((60, 0.5, 0.7), (60, 0.5, 3.0), 1),
            _chain((0, 3, -0.5), (40, 3, -6.0), 2),
            _chain((0, 10, -0.2), (80, 10 + 80 * math.tan(math.radians(5)), -0.2), 4),
            _chain((0, 10, -0.2), (80, 10, -0.2), 4),
            _chain((10, 1.0, 0.7), (40, 1.0 + 30 * math.tan(math.radians(1)), 0.7), 3),
        ]
    )


# audit round 3 (A3C-1 class; auditor's evidence e08 (b), e13)
@pytest.mark.finding("10.20")
@pytest.mark.parametrize("image", ["antiparallel", "none"])
def test_carson_tiling_does_not_depend_on_the_segment_order(image: str) -> None:
    """Near-parallel pairs that are not exactly parallel were projected on
    the axis of the first segment of the pair: reversing the segment order
    moved entries by up to 3.1 % of the largest one (a pair got an overlap
    in one order and none in the other). Every order and direction now
    gives the same matrix, up to round-off."""
    seg = _tiling_geometry()
    radii = np.full(len(seg), 0.005)
    om = [2.0 * math.pi * 1000.0]
    dz = build_carson_correction_matrices(
        seg, radii, omegas=om, sigma_earth=0.01, image=image
    )[0]
    rng = np.random.default_rng(3)
    for perm in (np.arange(len(seg))[::-1], rng.permutation(len(seg))):
        flip = rng.random(len(seg)) < 0.5
        s = seg[perm].copy()
        s[flip] = s[flip][:, ::-1]
        dzp = build_carson_correction_matrices(
            s, radii[perm], omegas=om, sigma_earth=0.01, image=image
        )[0]
        sign = np.where(flip, -1.0, 1.0)
        back = np.empty_like(dzp)
        back[np.ix_(perm, perm)] = sign[:, None] * sign[None, :] * dzp
        assert np.abs(back - dz).max() <= 1e-12 * np.abs(dz).max()


# audit A3C-3: the documentation describes the validity domain
@pytest.mark.finding("10.20")
def test_docs_describe_the_carson_validity_domain() -> None:
    """Audit round 3: the explanation and how-to pages still described
    carson_series as Carson's correction "scaled by the geometric length of
    each segment pair" (the heuristic removed in audit round 2)."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    pages = {
        name: " ".join((root / "docs" / name).read_text(encoding="utf-8").split())
        for name in ("explanation/earth-return.md", "how-to/inductive-coupling.md")
    }
    for text in pages.values():
        assert "geometric length of each segment pair" not in text
        assert "times the pair length" not in text
        assert "8.1°" in text
        assert "transmitted into the soil" in text
    explanation = pages["explanation/earth-return.md"]
    assert "Sommerfeld correction of a homogeneous earth" in explanation
    assert "quarter skin depth" in explanation  # long-segment rule
    assert "Wires beside each other" in explanation  # touching rule
