"""The layered correction of the cross-layer path in line form (10.28, ADR-0007).

Wave 3 of the 2026.11 work (stream w3xl, item 2). For electrodes that cross
the interface, ``image_2layer`` builds the reaction matrix from the
homogeneous line kernel and a layered correction ``ΔZ`` evaluated by
spectral quadrature between segment midpoints. Only the interface image of
near pairs in one layer was a Galerkin integral; the rest of ``ΔZ`` stayed
in point form, so a rod through the interface changed by 1.3e-4 ... 2.4e-4
between 2 and 0.125 m segments and the path disagreed with the Tagg/Sunde
series by up to 6.2e-4 where both apply. The correction is an exact image
expansion in every layer combination (Sunde); its images now get the
line-kernel treatment of the series path (far quadrupole term, blended
Galerkin integrals of near pairs, tapered Galerkin self images).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

SOILS = [(500.0, 100.0), (1000.0, 30.0), (30.0, 1000.0)]


def _rod(rho):
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho[0], rho_2=rho[1], h_1=5.0))
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.6), length=6.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    return w


def _z(world, ds, anchor="rod"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gf.create_engine(segment_length=ds, frequencies=[50.0]).solve(world)
    assert res.metadata["series_used"] is False
    return res.cluster_impedance(anchor)[0].real


@pytest.mark.finding("10.28")
@pytest.mark.parametrize("rho", SOILS, ids=["500-100", "1000-30", "30-1000"])
def test_a_rod_through_the_interface_does_not_depend_on_the_segment_length(rho) -> None:
    """One owner per layer, each straight: the Galerkin solution is exact at
    any segment length once every pair term is. Between 2 and 0.125 m:
    6.4e-7 (500/100), 6.5e-7 (1000/30), 9.0e-7 (30/1000); before 1.3e-4,
    1.6e-4, 2.4e-4 (evidence ``item2_crosslayer.py``)."""
    world = _rod(rho)
    z = [_z(world, ds) for ds in (2.0, 1.0, 0.5, 0.25, 0.125)]
    assert max(z) / min(z) - 1.0 < 2e-6


def _upper_world(kind: str, h_1: float = 1.0) -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1.0, rho_2=1.0, h_1=h_1))
    if kind == "ring":
        gf.create_electrode(w, "ring", name="e", center=(0, 0, h_1 - 0.2), radius=3.0)
    elif kind == "strip":
        gf.create_electrode(
            w, "strip", name="e", start=(0, 0, h_1 - 0.5), end=(10, 0, h_1 - 0.5)
        )
    else:
        gf.create_electrode(
            w,
            "grid_mesh",
            name="e",
            corner=(0, 0, h_1 - 0.3),
            size=(10.0, 10.0),
            n_x=2,
            n_y=2,
        )
    return w


@pytest.mark.finding("10.28")
@pytest.mark.parametrize("kind", ["ring", "strip", "grid"])
@pytest.mark.parametrize(
    "rho", [(1000.0, 30.0), (30.0, 1000.0)], ids=["1000-30", "30-1000"]
)
@pytest.mark.parametrize("ds", [0.5, 0.25])
def test_the_cross_layer_path_matches_the_series_where_both_apply(
    kind, rho, ds
) -> None:
    """For wires in the upper layer the two paths of ``image_2layer`` agree.

    The resistance of the uniform current, ``iᵀ M i`` with ``i`` the length
    weights: within 1.7e-7 (the spectral quadrature), matrix entries within
    2.5e-7 of the largest; before up to 6.2e-4 / 2.9e-4 (a ring 0.2 m above
    the interface, a strip 0.5 m and a grid 0.3 m above it, h_1 = 1 m).
    """
    from groundfield.solver import image_2layer
    from groundfield.solver._kernels import TwoLayerSeriesKernel
    from groundfield.solver._pipeline import SegmentArrays, discretise
    from groundfield.solver._series import ImageSeries

    soil = gf.TwoLayerSoil(rho_1=rho[0], rho_2=rho[1], h_1=1.0)
    disc = discretise(_upper_world(kind), gf.Engine(segment_length=ds))
    a = SegmentArrays.from_segments(disc.segments)
    line = a.line_segments()
    series = ImageSeries.for_soil(soil, 2000, 1e-14)
    m_series = TwoLayerSeriesKernel(soil.rho_1, series, line=line).matrix(
        a.points, a.lengths, a.radii
    )
    phi, dz = image_2layer._layered_green_matrices(
        a.points, a.lengths, a.radii, soil=soil, line=line
    )
    m_cross = phi + dz
    i = a.lengths / a.lengths.sum()
    assert abs((i @ m_cross @ i) / (i @ m_series @ i) - 1.0) < 5e-7
    assert np.max(np.abs(m_cross - m_series)) < 6e-7 * np.max(np.abs(m_series))


@pytest.mark.finding("10.28")
@pytest.mark.parametrize(
    ("z", "z_s"),
    [(0.8, 2.3), (4.6, 4.9), (5.4, 7.0), (6.0, 5.2), (1.0, 6.5), (7.5, 0.6)],
    ids=["uu", "uu-interface", "ll", "ll-interface", "ul", "lu"],
)
def test_the_image_families_are_the_expansion_of_the_spectral_kernel(z, z_s) -> None:
    """The families' coefficients and image depths reproduce the exact image
    expansion of ``layered_green`` (``_image_generation``, written from the
    closed form of the matching problem) for every layer combination.

    ``ΔZ`` relative to the baseline of the source layer, as the sum of
    ``c/(2R)`` over the images at a horizontal distance s, from both: the
    families (the solver's tables) and the module's generations summed to
    convergence (the ``m = 0`` transmitted pair of different layers is the
    transmitted resistivity of ``Φ_hom`` and appears in neither).
    """
    from groundfield.coupling.layered_green import _image_generation
    from groundfield.solver._line_kernel import LineSegments
    from groundfield.solver.image_2layer import _LayeredImageFamilies

    rho_1, rho_2, h_1 = 500.0, 100.0, 5.0
    soil = gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=h_1)
    K = (rho_2 - rho_1) / (rho_2 + rho_1)
    # Two vertical segments (observer at z, source at z_s), 0.5 m long.
    segs = LineSegments(
        np.array([[0.0, 0.0, z], [3.0, 0.0, z_s]]),
        np.array([0.5, 0.5]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        np.array([0.008, 0.008]),
    )
    lower = segs.mid[:, 2] >= h_1
    families = _LayeredImageFamilies(segs, lower, soil)
    s = 3.0
    lay = lower.astype(int)
    total = 0.0
    for images, tab in families.groups:
        for weight, offset, sign in images:
            d = z - (offset + sign * z_s)
            total += (
                weight * tab[lay[0], lay[1]] * 4.0 * np.pi / (2.0 * math.hypot(s, d))
            )
    # The module's expansion minus the baseline of the source layer, over the
    # generations the families carry (beyond them the line form equals the
    # point form: the taper of the far-image term is zero there).
    m_max = max(len(images) for images, _ in families.groups)
    assert m_max >= 5
    rho_b = rho_1 if z_s < h_1 else rho_2
    ref = 0.0
    for m in range(0, m_max + 1):
        for c, d in _image_generation(m, z, z_s, rho_1, rho_2, h_1, K):
            if (z <= h_1) != (z_s <= h_1) and m == 0:
                continue  # transmitted direct and surface pair: in Φ_hom
            ref += c / (2.0 * math.hypot(s, d))
    if (z <= h_1) == (z_s <= h_1):
        for c, d in _image_generation(0, z, z_s, rho_b, rho_b, h_1, 0.0):
            ref -= c / (2.0 * math.hypot(s, d))
    assert total == pytest.approx(ref, rel=1e-12, abs=1e-14 * rho_1)


@pytest.mark.finding("10.28")
def test_the_cross_layer_line_matrix_stays_reciprocal() -> None:
    """Symmetric to 4e-12 (the spectral path alone: 3e-12)."""
    from groundfield.solver import image_2layer
    from groundfield.solver._pipeline import SegmentArrays, discretise

    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    w = _rod((1000.0, 30.0))
    gf.create_electrode(w, "ring", name="ring", center=(0.0, 0.0, 4.8), radius=3.0)
    disc = discretise(w, gf.Engine(segment_length=0.5), layer_interfaces=(5.0,))
    a = SegmentArrays.from_segments(disc.segments)
    phi, dz = image_2layer._layered_green_matrices(
        a.points, a.lengths, a.radii, soil=soil, line=a.line_segments()
    )
    m = phi + dz
    assert np.max(np.abs(m - m.T)) < 1e-10 * np.max(np.abs(m))
