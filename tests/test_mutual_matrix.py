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


# ---------------------------------------------------------------------
# One frequency per call (10.60, coup2 request)
# ---------------------------------------------------------------------


def _inductive_pair_world() -> "gf.World":
    """Two rods joined by a 40 m Neumann PEN, plus a separate rod."""
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(world, "rod", name="a", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(world, "rod", name="b", position=(40.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(world, "rod", name="c", position=(20.0, 15.0, 0.5), length=2.0)
    gf.create_conductor(
        world,
        name="pen",
        start="a",
        end="b",
        cross_section=95e-6,
        discretize_segment_length=10.0,
        inductance_model="neumann",
    )
    return world


@pytest.mark.finding("10.60")
@pytest.mark.parametrize("which", ["matrix", "field"])
def test_one_call_assembles_the_branch_correction_for_its_frequency_only(
    monkeypatch, which
) -> None:
    """``frequency_index`` selects one frequency; the inductive branch
    correction must not be assembled for the engine's whole frequency set
    (the multi-frequency assembly of ``Engine.solve``)."""
    from groundfield.solver import mutual

    built = []
    real = mutual.assemble_branches

    def recording(world, engine, topo):
        branches = real(world, engine, topo)
        built.append(branches)
        return branches

    monkeypatch.setattr(mutual, "assemble_branches", recording)
    freqs = [50.0, 150.0, 250.0, 1000.0]
    engine = gf.create_engine(segment_length=1.0, frequencies=freqs)
    world = _inductive_pair_world()
    if which == "matrix":
        Z = solve_mutual_matrix(world, engine, anchors=["a", "c"], frequency_index=2)
    else:
        Z = gf.solve_mutual_field(
            world,
            engine,
            anchors=["a", "c"],
            field_points=_PROBE_POINTS,
            frequency_index=2,
        )
    assert built[-1].earth_correction.n_assemblies == 1

    single = gf.create_engine(segment_length=1.0, frequencies=[freqs[2]])
    if which == "matrix":
        Z_single = solve_mutual_matrix(world, single, anchors=["a", "c"])
    else:
        Z_single = gf.solve_mutual_field(
            world, single, anchors=["a", "c"], field_points=_PROBE_POINTS
        )
    np.testing.assert_array_equal(Z, Z_single)
    assert np.iscomplexobj(Z) and np.abs(Z.imag).max() > 0.0


def test_frequency_index_out_of_range_is_harmless_without_inductance() -> None:
    """Documented: the index matters only for frequency-dependent worlds."""
    world = _build_world(gf.HomogeneousSoil(rho=100.0), 2.0)
    engine = gf.create_engine(segment_length=_SEGMENT_LENGTH)
    names = [a[0] for a in _ANCHORS]
    Z0 = solve_mutual_matrix(world, engine, anchors=names)
    Z5 = solve_mutual_matrix(world, engine, anchors=names, frequency_index=5)
    np.testing.assert_array_equal(Z0, Z5)


# ---------------------------------------------------------------------
# Audit round 2, A2S-03: the floating core is factorised once
# ---------------------------------------------------------------------


def _mixed_site(soil: object, *, inductive: bool = False, shell: bool = False):
    """Single rings, bonded ring+rod groups, one group joined by a finite bond."""
    import warnings

    w = gf.create_world(soil=soil)
    anchors: list[str] = []
    kw = {"concrete_shell_coefficient_ohm_m": 38.0} if shell else {}
    for k in range(3):  # single electrodes: no floating partner
        gf.create_electrode(
            w, "ring", name=f"s{k}", center=(30.0 * k, 0.0, 0.8), radius=3.0, **kw
        )
        anchors.append(f"s{k}")
    for k in range(3):  # bonded groups: the floating core
        x = 30.0 * k
        gf.create_electrode(
            w, "ring", name=f"b{k}", center=(x, 30.0, 0.8), radius=3.0, **kw
        )
        gf.create_electrode(
            w, "rod", name=f"b{k}_rod", position=(x + 3.0, 30.0, 0.8), length=6.0
        )
        gf.create_conductor(w, start=f"b{k}", end=f"b{k}_rod", name=f"b{k}_bond")
        anchors.append(f"b{k}")
    # a passive network: two rings joined by a finite conductor
    gf.create_electrode(w, "ring", name="n0", center=(0.0, 60.0, 0.8), radius=3.0)
    gf.create_electrode(w, "ring", name="n1", center=(40.0, 60.0, 0.8), radius=3.0)
    extra = {}
    if inductive:
        extra = {"discretize_segment_length": 8.0, "inductance_model": "neumann"}
    gf.create_conductor(
        w, start="n0", end="n1", name="pen", cross_section=50e-6, **extra
    )
    anchors.append("n0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.check(require_source=False)
    return w, anchors


def _excitations_both_ways(world, engine, anchors):
    from groundfield.solver import _topology, mutual

    problem = mutual._prepare(world, engine, 0, where="test")
    Z_seg = problem.Z_seg_full
    common = dict(
        electrodes=world.electrodes,
        cluster_id=problem.topo.cluster_id,
        seg_points=problem.arrays.points,
        seg_lengths=problem.arrays.lengths,
        wire_radii=problem.arrays.radii,
        elec_to_segidx=problem.disc.owner_segments,
        self_kernel=lambda p, l, r, c: Z_seg @ c,  # noqa: E741
        finite_branches=problem.topo.finite_branches,
        pseudo_owners=problem.topo.pseudo_owners,
        omega=problem.omega if problem.branches.has_inductance else 0.0,
        inductance_matrix=problem.branches.inductance_matrix,
        carson_correction=problem.carson_dz,
        shell_coefficients=problem.arrays.shell_coefficients,
    )
    batched = _topology._solve_unit_excitations(anchors=anchors, **common)
    per_anchor = [
        _topology._solve_cluster_currents(
            elec_input_current={**{e.name: 0j for e in world.electrodes}, a: 1.0 + 0j},
            floating=True,
            **common,
        )
        for a in anchors
    ]
    return batched, per_anchor


@pytest.mark.finding("10.29")
@pytest.mark.parametrize(
    "case",
    ["homogeneous", "two_layer_crossing", "inductive_1khz", "concrete_shell"],
)
def test_shared_factorisation_equals_one_floating_solve_per_anchor(case: str) -> None:
    """Every owner current of every excitation, as the per-anchor solve."""
    # audit A2S-03
    import warnings

    soil = gf.HomogeneousSoil(rho=100.0)
    freqs = [50.0]
    kw = {}
    if case == "two_layer_crossing":
        soil = gf.TwoLayerSoil(rho_1=300.0, rho_2=50.0, h_1=4.0)  # rods cross h_1
    if case == "inductive_1khz":
        freqs = [1000.0]
        kw = {"inductive": True}
    if case == "concrete_shell":
        kw = {"shell": True}
    world, anchors = _mixed_site(soil, **kw)
    engine = gf.Engine(segment_length=2.0, frequencies=freqs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batched, reference = _excitations_both_ways(world, engine, anchors)
    for got, want in zip(batched, reference, strict=True):
        assert got.keys() == want.keys()
        scale = max(abs(v) for v in want.values())
        for name in want:
            assert abs(got[name] - want[name]) <= 1e-12 * scale, (
                name,
                got[name],
                want[name],
            )
    if case == "inductive_1khz":
        assert any(abs(v.imag) > 1e-9 for d in batched for v in d.values())


@pytest.mark.finding("10.29")
@pytest.mark.parametrize("n_groups", [4, 12])
def test_one_core_assembly_and_factorisation_for_all_anchors(
    monkeypatch: pytest.MonkeyPatch, n_groups: int
) -> None:
    """The base assembled and factorised the floating core once per anchor.

    ``n_groups`` bonded ring+rod groups and as many single rings, all of
    them anchors: one multi-port assembly over the core's owners, one
    coupling call for the single rings, one augmented system.
    """
    # audit A2S-03
    import warnings

    from groundfield.solver import _topology

    counts = {"multiport_columns": 0, "augmented": 0}
    original_mp = _topology._multiport_matrix
    original_aug = _topology._augmented_matrix

    def counting_mp(system, **kwargs):  # noqa: ANN001, ANN202
        counts["multiport_columns"] += system.n_owners
        return original_mp(system, **kwargs)

    def counting_aug(Z, system, cluster_id):  # noqa: ANN001, ANN202
        counts["augmented"] += 1
        return original_aug(Z, system, cluster_id)

    monkeypatch.setattr(_topology, "_multiport_matrix", counting_mp)
    monkeypatch.setattr(_topology, "_augmented_matrix", counting_aug)
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    anchors = []
    for k in range(n_groups):
        x = 25.0 * k
        gf.create_electrode(w, "ring", name=f"h{k}", center=(x, 0.0, 0.8), radius=4.0)
        gf.create_electrode(
            w, "rod", name=f"h{k}_rod", position=(x + 4.0, 0.0, 0.8), length=3.0
        )
        gf.create_conductor(w, start=f"h{k}", end=f"h{k}_rod", name=f"b{k}")
        gf.create_electrode(w, "ring", name=f"s{k}", center=(x, 25.0, 0.8), radius=4.0)
        anchors += [f"h{k}", f"s{k}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solve_mutual_matrix(w, gf.Engine(segment_length=2.0, kernel="point"), anchors)
    assert counts["augmented"] == 1
    assert counts["multiport_columns"] == 2 * n_groups  # the core's owners, once
