"""Regression tests for :func:`groundfield.solve_mutual_matrix`.

With probe points, the single-assembly mutual matrix reproduces the
historic per-excitation construction (one :meth:`Engine.solve` per
gnode; diagonal from :meth:`FieldResult.cluster_impedance`, off-diagonal
from :meth:`FieldResult.potential` at the probe points) to round-off
(``max|Z_new - Z_ref| < 1e-9``), for the homogeneous limit, a
within-layer two-layer soil and a cross-layer two-layer soil (a deep rod
piercing a thin upper layer, the ADR-0007 path).

The diagonal is the length-weighted cluster impedance over all members
(review pass 10, 10.26); without probe points the off-diagonal is the
reciprocal reaction of the leakage distributions (``offdiag="galerkin"``,
10.34), which reproduces the solve of the clusters joined by branches.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield import solve_mutual_matrix

# Three isolated single-rod grounding clusters (gnodes) on a triangle.
_ANCHORS: list[tuple[str, float, float]] = [
    ("g0", 0.0, 0.0),
    ("g1", 12.0, 0.0),
    ("g2", 6.0, 9.0),
]
_SEGMENT_LENGTH = 0.5
_FREQUENCIES = [50.0]
# Probe row i: surface point above rod i (off-diagonal sampling point),
# mirroring ``res.potential([[x_i, y_i, 0.0]])`` in the historic loop.
_PROBE_POINTS = np.array([[x, y, 0.0] for _, x, y in _ANCHORS])


def _build_world(soil: object, length: float) -> "gf.World":
    """World carrying all gnodes and ``soil`` but no current source."""
    world = gf.create_world(soil=soil)
    for name, x, y in _ANCHORS:
        gf.create_electrode(
            world, "rod", name=name, position=(x, y, 0.5), length=length
        )
    return world


def _reference_zg(soil: object, length: float, *, symmetrize: bool) -> np.ndarray:
    """Historic per-excitation construction of ``Z_G``."""
    names = [a[0] for a in _ANCHORS]
    nG = len(names)
    engine = gf.create_engine(
        backend="image",
        segment_length=_SEGMENT_LENGTH,
        frequencies=_FREQUENCIES,
    )
    Z = np.zeros((nG, nG), dtype=complex)
    for j, anchor_j in enumerate(names):
        world = _build_world(soil, length)
        gf.create_source(world, attached_to=anchor_j, magnitude=1.0)
        res = engine.solve(world)
        Z[j, j] = res.cluster_impedance(anchor_j)[0]
        phi = res.potential(_PROBE_POINTS)  # (nG,)
        for i in range(nG):
            if i != j:
                Z[i, j] = phi[i]
    if symmetrize:
        Z = 0.5 * (Z + Z.T)
    return Z


@pytest.mark.parametrize(
    "soil, length, label",
    [
        (gf.HomogeneousSoil(resistivity=100.0), 3.0, "homogeneous"),
        (
            gf.TwoLayerSoil(rho_1=300.0, rho_2=100.0, h_1=8.0),
            3.0,
            "two-layer within-layer",
        ),
        (
            gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0),
            9.0,
            "two-layer cross-layer",
        ),
    ],
)
def test_mutual_matrix_matches_per_excitation(soil, length, label) -> None:
    """``solve_mutual_matrix`` == historic per-excitation loop (bit-exact)."""
    Z_ref = _reference_zg(soil, length, symmetrize=True)

    world = _build_world(soil, length)
    engine = gf.create_engine(
        backend="image",
        segment_length=_SEGMENT_LENGTH,
        frequencies=_FREQUENCIES,
    )
    Z_new = solve_mutual_matrix(world, engine, [a[0] for a in _ANCHORS], _PROBE_POINTS)

    assert Z_new.shape == (3, 3)
    assert np.max(np.abs(Z_new - Z_ref)) < 1e-9, (
        f"{label}: max|dZ| = {np.max(np.abs(Z_new - Z_ref)):.3e}"
    )


def test_mutual_matrix_unsymmetrized_matches_reference() -> None:
    """``symmetrize=False`` returns the raw per-excitation matrix."""
    soil = gf.TwoLayerSoil(rho_1=300.0, rho_2=100.0, h_1=8.0)
    Z_ref = _reference_zg(soil, 3.0, symmetrize=False)

    world = _build_world(soil, 3.0)
    engine = gf.create_engine(
        backend="image",
        segment_length=_SEGMENT_LENGTH,
        frequencies=_FREQUENCIES,
    )
    Z_new = solve_mutual_matrix(
        world,
        engine,
        [a[0] for a in _ANCHORS],
        _PROBE_POINTS,
        symmetrize=False,
    )
    assert np.max(np.abs(Z_new - Z_ref)) < 1e-9


def test_mutual_matrix_diagonal_is_cluster_impedance() -> None:
    """Diagonal entries equal the single-cluster grounding impedance."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = _build_world(soil, 3.0)
    engine = gf.create_engine(
        backend="image",
        segment_length=_SEGMENT_LENGTH,
        frequencies=_FREQUENCIES,
    )
    Z = solve_mutual_matrix(
        world,
        engine,
        [a[0] for a in _ANCHORS],
        _PROBE_POINTS,
        symmetrize=False,
    )
    # Self-impedance is real, positive and dominates its row/column.
    for k in range(3):
        assert Z[k, k].real > 0.0
        assert abs(Z[k, k]) > abs(Z[k, (k + 1) % 3])


def test_mutual_matrix_rejects_bad_probe_shape() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = _build_world(soil, 3.0)
    engine = gf.create_engine(backend="image", frequencies=_FREQUENCIES)
    with pytest.raises(ValueError, match="probe_points must have shape"):
        solve_mutual_matrix(
            world,
            engine,
            [a[0] for a in _ANCHORS],
            np.zeros((2, 3)),  # nG = 3 expected
        )


def test_mutual_matrix_rejects_unknown_anchor() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = _build_world(soil, 3.0)
    engine = gf.create_engine(backend="image", frequencies=_FREQUENCIES)
    with pytest.raises(KeyError, match="not an electrode"):
        solve_mutual_matrix(world, engine, ["g0", "g1", "ghost"], _PROBE_POINTS)


# ---------------------------------------------------------------------
# Review pass 10: weighted diagonal (10.26), backend check and galerkin
# off-diagonal (10.34)
# ---------------------------------------------------------------------

_L_FOOTPRINT = [
    (0.0, 0.0, 0.8),
    (12.0, 0.0, 0.8),
    (12.0, 7.0, 0.8),
    (5.0, 7.0, 0.8),
    (5.0, 10.0, 0.8),
    (0.0, 10.0, 0.8),
]


def _house_and_far_rod(soil: object, *, split_rod: bool = False) -> "gf.World":
    world = gf.create_world(soil=soil)
    if split_rod:
        # 4.509 m rod from 0.5 m: crosses h_1 = 5 m by 9 mm.
        gf.create_electrode(
            world,
            "rod",
            name="house",
            position=(0, 0, 0.5),
            length=4.509,
            wire_radius=0.008,
        )
    else:
        gf.create_electrode(
            world, "polyline", name="house", closed=True, vertices=_L_FOOTPRINT
        )
    gf.create_electrode(
        world, "rod", name="far", position=(40, 0, 0.0), length=3.0, wire_radius=0.008
    )
    return world


@pytest.mark.finding("10.26")
@pytest.mark.parametrize(
    "soil, split_rod, segment_length",
    [
        (gf.HomogeneousSoil(rho=100.0), False, 2.0),  # v0.15.0: +0.150 %
        (gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0), False, 2.0),  # +0.179 %
        (gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=5.0), True, 0.5),  # -3.81 %
        (gf.TwoLayerSoil(rho_1=1000.0, rho_2=100.0, h_1=5.0), True, 0.5),  # -9.08 %
    ],
)
def test_diagonal_is_the_weighted_cluster_impedance(
    soil, split_rod, segment_length
) -> None:
    """``Z[j, j]`` equals ``cluster_impedance`` of the per-excitation solve.

    Up to 0.15.0 the diagonal was the unweighted mean over the segments
    of the alphabetically first member: wrong for an L-shaped footprint
    at ds = 2 m (unequal segment lengths) and for a rod split at the
    layer interface.
    """
    engine = gf.create_engine(
        backend="image", segment_length=segment_length, image_max_terms=500
    )
    world = _house_and_far_rod(soil, split_rod=split_rod)
    probes = np.array([[6.0, 5.0, 0.0], [40.0, 0.0, 0.0]])
    Z = solve_mutual_matrix(world, engine, ["house", "far"], probes)

    reference = _house_and_far_rod(soil, split_rod=split_rod)
    gf.create_source(reference, attached_to="house", magnitude=1.0)
    z_cluster = engine.solve(reference).cluster_impedance("house")[0]
    assert abs(Z[0, 0] / z_cluster - 1.0) < 1e-12


@pytest.mark.finding("10.34")
@pytest.mark.parametrize("backend", ["mom", "mom_sommerfeld"])
def test_non_image_backend_is_rejected(backend) -> None:
    """The mutual assembly is the image family's; other engines raise.

    Up to 0.15.0 ``Engine.backend`` was ignored: an engine for ``mom``
    returned image numbers.
    """
    from groundfield import solve_mutual_field

    world = _build_world(gf.HomogeneousSoil(rho=100.0), 3.0)
    engine = gf.create_engine(
        backend=backend, segment_length=0.5, frequencies=_FREQUENCIES
    )
    names = [a[0] for a in _ANCHORS]
    with pytest.raises(ValueError, match="image family"):
        solve_mutual_matrix(world, engine, names, _PROBE_POINTS)
    with pytest.raises(ValueError, match="image family"):
        solve_mutual_field(world, engine, names, _PROBE_POINTS)


@pytest.mark.finding("10.34")
@pytest.mark.parametrize(
    "soil, length",
    [
        (gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0), 3.0),
        (gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0), 9.0),  # rods through h_1
    ],
)
def test_galerkin_offdiagonal_is_reciprocal(soil, length) -> None:
    """``offdiag="galerkin"`` is reciprocal without symmetrisation.

    The probe definition is not: entry pairs differ by up to 35 % in
    this soil (review pass 10, R1-08).
    """
    world = _build_world(soil, length)
    engine = gf.create_engine(
        backend="image", segment_length=_SEGMENT_LENGTH, image_max_terms=500
    )
    names = [a[0] for a in _ANCHORS]
    Z = solve_mutual_matrix(world, engine, names, offdiag="galerkin", symmetrize=False)
    assert np.linalg.norm(Z - Z.T) / np.linalg.norm(Z) < 1e-10
    # Same diagonal as the probe call: the definitions differ off the diagonal only.
    Zp = solve_mutual_matrix(world, engine, names, _PROBE_POINTS, symmetrize=False)
    np.testing.assert_allclose(np.diag(Z), np.diag(Zp), rtol=1e-13)


@pytest.mark.finding("10.34")
@pytest.mark.parametrize(
    "soil",
    [gf.HomogeneousSoil(rho=100.0), gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)],
)
def test_galerkin_matrix_reproduces_the_joined_network(soil) -> None:
    """``(Y_L + Z_G^-1) u = i`` with the galerkin ``Z_G`` equals the network solve.

    The three rods joined by resistive branches, 1 A into ``g0``, solved
    once by :meth:`Engine.solve`: the reduced network reproduces every
    node potential to round-off. With probe off-diagonals it does not
    (0.2-0.9 % on the input impedance of an AP1-like site, evidence
    ``e1026_mutual_ap1_station.py``).
    """
    names = [a[0] for a in _ANCHORS]
    engine = gf.create_engine(
        backend="image", segment_length=_SEGMENT_LENGTH, image_max_terms=500
    )
    Z = solve_mutual_matrix(_build_world(soil, 3.0), engine, names)

    network = _build_world(soil, 3.0)
    branches = []
    for a, b, R in (("g0", "g1", 0.5), ("g0", "g2", 0.8), ("g1", "g2", 2.0)):
        conductor = gf.create_conductor(
            network,
            start=a,
            end=b,
            name=f"{a}{b}",
            cross_section=50e-6,
            lumped_series_resistance_ohm=R,
        )
        branches.append((a, b, conductor.series_resistance))
    gf.create_source(network, attached_to="g0", magnitude=1.0)
    res = engine.solve(network)
    u_solve = np.array([res.electrode_potentials[n][0] for n in names])

    Y = np.zeros((3, 3))
    for a, b, R in branches:
        i, j = names.index(a), names.index(b)
        Y[[i, j], [i, j]] += 1.0 / R
        Y[i, j] -= 1.0 / R
        Y[j, i] -= 1.0 / R
    u_reduced = np.linalg.solve(Y + np.linalg.inv(Z), np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(u_reduced, u_solve, rtol=1e-10)


def test_offdiag_selection_and_validation() -> None:
    """``offdiag=None`` follows the probe points; inconsistent calls raise."""
    world = _build_world(gf.HomogeneousSoil(rho=100.0), 3.0)
    engine = gf.create_engine(backend="image", segment_length=_SEGMENT_LENGTH)
    names = [a[0] for a in _ANCHORS]
    np.testing.assert_array_equal(
        solve_mutual_matrix(world, engine, names, _PROBE_POINTS),
        solve_mutual_matrix(world, engine, names, _PROBE_POINTS, offdiag="probe"),
    )
    np.testing.assert_array_equal(
        solve_mutual_matrix(world, engine, names),
        solve_mutual_matrix(world, engine, names, offdiag="galerkin"),
    )
    with pytest.raises(ValueError, match="needs probe_points"):
        solve_mutual_matrix(world, engine, names, offdiag="probe")
    with pytest.raises(ValueError, match="not used with offdiag='galerkin'"):
        solve_mutual_matrix(world, engine, names, _PROBE_POINTS, offdiag="galerkin")
    with pytest.raises(ValueError, match="offdiag must be"):
        solve_mutual_matrix(world, engine, names, offdiag="center")
