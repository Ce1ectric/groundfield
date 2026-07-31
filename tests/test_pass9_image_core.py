"""Review-pass-9 regression tests for the closed-form galvanic core.

Findings covered (audit 2026-07-28, fixed in 0.15.0):

- **F10 / F11** — the reported node potential must be the
  *length-weighted* (Galerkin) average of the segment-midpoint
  potentials, i.e. exactly the reduction the multi-port matrix
  :math:`Z_{ij}` enforces inside the augmented system, and
  :meth:`FieldResult.cluster_impedance` must be built from the
  cluster-wide weighted potential instead of the alphabetically
  first member. Physics: a galvanic node is one equipotential; the
  discrete stand-in dual to the uniform-per-unit-length current
  ansatz :math:`I_k = I_e L_k / \\sum_j L_j` is the average over the
  electrode *surface*, :math:`\\varphi_e = \\sum_k L_k \\varphi_k /
  \\sum_k L_k`, not over the segment *list*.
- **F26 / F29** — the half-space image construction
  :math:`G = \\rho/(4\\pi)\\,(1/r + 1/r')` requires a finite image
  separation :math:`2 z \\gg a`. A conductor at (or above) the soil
  surface coincides with its own mirror image and the diagonal
  self-image term diverges; up to 0.14.1 it was silently *clamped*
  at ``_MIN_DISTANCE = 1 mm``, which returned a mesh-dependent,
  up to ~15x too large grounding impedance. It is now a hard
  ``ValueError`` (mirroring the guard the inductive path received
  in ``build_inductance_matrix``), with a
  :class:`ShallowSegmentWarning` for the merely shallow regime
  :math:`L > 4 z`.

Follow-up audit 2026-07-29 (same release):

- the ``z <= 0`` rejection moved into the **shared** reaction-matrix
  kernel, so ``mom`` / ``bem`` / ``cim`` / ``mom_sommerfeld`` /
  ``mutual`` reject an airborne electrode too instead of mirroring it
  into the soil;
- the depth validation looks at the segment *extent*, not just its
  midpoint, so a conductor that pokes out of the ground is rejected
  rather than warned about;
- :meth:`FieldResult.cluster_impedance` sums the cluster current over
  *all* members again, not only over those that also carry potential
  data.

Provenance of the quoted numbers
--------------------------------
Every number labelled "0.14.1" was measured on the released 0.14.1
code (tag ``v0.14.1``, commit ``5ba740d``) with ``PYTHONPATH`` pointed
at a checkout of that tag.

The 0.15.0 changes are **not bit-identical** to 0.14.1 even for
uniform segment lengths, and the tests below do not claim they are.
The reported node potential changed from ``np.mean(phi[idxs])`` to
``(w @ phi[idxs]) / w.sum()``; for equal weights those are the same
number mathematically but not the same *floating-point summation* —
``np.mean`` uses pairwise summation, the BLAS dot product does not.
Measured on an isolated tree (v0.14.1 plus only
``solver/{image,image_2layer,result}.py`` from this branch), 4 of 11
uniform-segment reference worlds move by 1-2 ulp. Two of them are
re-measured directly for this module:

===============================  ===================  ===================
world (rho = 100 Ohm-m)          0.14.1               0.15.0
===============================  ===================  ===================
10 m strip z=0.5 ds=0.5          14.798310924481832   14.798310924481834
ring r=4 z=0.7 ds=0.5            7.447920231050293    7.447920231050295
10 m tape z=0.01 ds=0.5          35.49116014344861    35.4911601434486
===============================  ===================  ===================

That is a relative change of ~1e-16, i.e. physically irrelevant, but
the correct claim is "unchanged to floating-point rounding", not
"bit-identical". Up to the 0.15.0 development state the docstrings
here asserted the opposite and, worse, labelled the *new* value
``14.798310924481834`` as "measured on v0.14.1". The assertions below
compare against the genuine 0.14.1 value with an explicit ulp budget.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.conductors.conductor import Conductor
from groundfield.solver.image import (
    ShallowSegmentWarning,
    _discretize_electrode,
    _self_corrected_kernel,
)
from groundfield.solver.result import FieldResult, PointSource


def _ulps_apart(a: float, b: float, budget: int = 4) -> bool:
    """True when ``a`` and ``b`` are at most ``budget`` ulps apart."""
    lo, hi = (a, b) if a <= b else (b, a)
    for _ in range(budget):
        if lo >= hi:
            return True
        lo = float(np.nextafter(lo, np.inf))
    return lo >= hi


# ---------------------------------------------------------------------
# F10 — reported potential == the potential the solver enforced
# ---------------------------------------------------------------------


def _multiport_diagonal(
    electrode, rho: float, ds: float, layer_interfaces=None
) -> float:
    """Independent reference for :math:`Z_{00}` of a single electrode.

    Re-implements the Galerkin reduction of the multi-port matrix from
    first principles: excite the electrode with the
    uniform-per-unit-length ansatz (:math:`\\sum I_k = 1` A), evaluate
    the segment-midpoint potentials with the same self-corrected
    kernel the solver uses, and reduce the row with the
    length-weighted average.
    """
    segs = _discretize_electrode(electrode, ds, layer_interfaces)
    lengths = np.array([s.length for s in segs])
    points = np.array([s.midpoint for s in segs])
    radii = np.array([s.wire_radius for s in segs])
    excitation = lengths / lengths.sum()
    phi = _self_corrected_kernel(points, lengths, radii, excitation, rho)
    return float((lengths @ phi) / lengths.sum())


def test_rod_split_at_layer_interface_reports_weighted_potential() -> None:
    """F10: the -10.9 % case of the audit — rod split at ``h_1``.

    ``TwoLayerSoil(200, 200, h_1=5)`` has ``K = 0``, so the two-layer
    backend must reproduce the homogeneous kernel exactly. The
    interface split of the 4.509 m rod produces five 0.900 m segments
    plus a 0.009 m remainder, i.e. the mixed-length case: 0.2 % of the
    electrode length used to receive 1/6 = 16.7 % of the weight in the
    reported potential.

    0.14.1 reported 40.70144871 Ω while its own constraint row
    enforced 45.68625169 Ω (-10.9 %).
    """
    rho = 200.0
    soil = gf.TwoLayerSoil(rho_1=rho, rho_2=rho, h_1=5.0)
    world = gf.create_world(soil=soil)
    rod = gf.create_electrode(
        world, "rod", name="r",
        position=(0.0, 0.0, 0.5), length=4.509, wire_radius=0.008,
    )
    gf.create_source(world, attached_to="r", magnitude=1.0)
    result = gf.create_engine(
        backend="image_2layer", segment_length=1.0
    ).solve(world)

    segs = _discretize_electrode(rod, 1.0, (soil.h_1,))
    lengths = np.array([s.length for s in segs])
    assert lengths.min() == pytest.approx(0.009, abs=1e-9), (
        "geometry precondition: the interface split must produce the "
        "0.009 m remainder segment"
    )

    z_reported = result.grounding_impedance("r")[0]
    z_reference = _multiport_diagonal(rod, rho, 1.0, (soil.h_1,))

    # The reported impedance is the impedance the solver enforced.
    assert z_reported.real == pytest.approx(z_reference, rel=1e-12)
    assert z_reported.real == pytest.approx(45.68625169, rel=1e-8)
    # ... and no longer the unweighted mean of 0.14.1.
    assert abs(z_reported.real - 40.70144871) > 1.0


def test_grounding_impedance_equals_multiport_diagonal() -> None:
    """F10: ``grounding_impedance(e) == Z[i, i]`` to machine precision.

    A polyline with legs of 3.0 m and 1.1 m discretised at
    ``ds = 1.0`` carries segments of three different lengths, so the
    weighted and unweighted reductions differ (0.14.1: 33.60176075 Ω
    reported vs. 33.82386779 Ω enforced, -0.66 %). For a single
    isolated electrode driven with 1 A the input impedance *is* the
    diagonal of the multi-port matrix, so the two must agree to
    round-off.
    """
    rho = 150.0
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=rho))
    poly = gf.create_electrode(
        world, "polyline", name="p",
        vertices=[(0.0, 0.0, 0.7), (3.0, 0.0, 0.7), (3.0, 1.1, 0.7)],
        wire_radius=0.006,
    )
    gf.create_source(world, attached_to="p", magnitude=1.0)
    result = gf.create_engine(backend="image", segment_length=1.0).solve(world)

    lengths = np.array([s.length for s in _discretize_electrode(poly, 1.0)])
    assert len(np.unique(np.round(lengths, 9))) > 1, (
        "geometry precondition: segment lengths must be mixed"
    )

    z_reference = _multiport_diagonal(poly, rho, 1.0)
    assert result.grounding_impedance("p")[0].real == pytest.approx(
        z_reference, rel=1e-12
    )
    assert result.cluster_impedance("p")[0].real == pytest.approx(
        z_reference, rel=1e-12
    )


# 10 m strip, z = 0.5 m, ds = 0.5 m, rho = 100: 20 equal segments.
# Measured on ``v0.14.1`` (PYTHONPATH at the tag). 0.15.0 returns
# 14.798310924481834, one ulp higher — see the module docstring.
Z_STRIP_10M_0_14_1 = 14.798310924481832


def test_uniform_segment_lengths_unchanged() -> None:
    """Control: for uniform segments the two averages coincide.

    A plain 10 m strip at 0.5 m depth discretises into 20 equal
    segments, so the length-weighted reduction is mathematically the
    plain mean and the 0.14.1 impedance is reproduced — **to
    floating-point rounding, not bit-identically**. ``np.mean`` sums
    pairwise, ``(w @ phi) / w.sum()`` is a BLAS dot product, so the
    two differ by 1 ulp here (0.14.1: ...832, 0.15.0: ...834). The
    assertion pins the 0.14.1 value with an explicit 4-ulp budget so a
    real regression still fails while a summation-order change does
    not.
    """
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world, "strip", name="s",
        start=(0.0, 0.0, 0.5), end=(10.0, 0.0, 0.5), wire_radius=0.005,
    )
    gf.create_source(world, attached_to="s", magnitude=1.0)
    result = gf.create_engine(backend="image", segment_length=0.5).solve(world)
    z = result.grounding_impedance("s")[0].real
    assert _ulps_apart(z, Z_STRIP_10M_0_14_1, budget=4), (
        f"{z!r} is more than 4 ulps from the 0.14.1 value "
        f"{Z_STRIP_10M_0_14_1!r}"
    )


# ---------------------------------------------------------------------
# F11 — cluster potential is a physical quantity, not a name lookup
# ---------------------------------------------------------------------


def _bonded_world(rod_name: str, ring_name: str) -> gf.World:
    """Rod crossing the layer interface, ideally bonded to a ring."""
    world = gf.create_world(
        soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
    )
    gf.create_electrode(
        world, "rod", name=rod_name,
        position=(0.0, 0.0, 0.1), length=5.0, wire_radius=0.008,
    )
    gf.create_electrode(
        world, "ring", name=ring_name,
        center=(10.0, 0.0, 0.7), radius=2.0, wire_radius=0.008,
    )
    gf.create_conductor(world, name="bond", start=rod_name, end=ring_name)
    gf.create_source(world, attached_to=rod_name, magnitude=1.0)
    return world


def test_cluster_members_report_one_common_potential() -> None:
    """F11: ideally bonded electrodes must report the same potential.

    The augmented system constrains every member of a galvanic
    cluster to a single potential :math:`\\varphi_c`. With the
    unweighted mean of 0.14.1 the rod (split at ``h_1 = 2 m`` into
    0.6667 m and 0.6 m segments) and the ring reported 17.90603210 V
    and 17.62820953 V respectively — a 1.6 % disagreement for two
    nodes the model holds at one potential.
    """
    world = _bonded_world("a_rod", "b_ring")
    result = gf.create_engine(
        backend="image_2layer", segment_length=0.7
    ).solve(world)

    potentials = np.array(
        [result.electrode_potentials[n][0] for n in ("a_rod", "b_ring")]
    )
    spread = abs(potentials[0] - potentials[1]) / abs(potentials.mean())
    assert spread < 1e-12, f"cluster members disagree by {spread:.2e}"


def test_cluster_impedance_invariant_under_renaming() -> None:
    """F11: renaming an electrode must not change the physics.

    ``cluster_impedance`` used to read ``members[0]`` — the
    alphabetically first name — so swapping the names of the two
    members changed the reported value of an identical geometry
    (0.14.1: 17.90603210 Ω for ``('a_rod', 'b_ring')`` vs.
    17.62820953 Ω for ``('z_rod', 'a_ring')``).
    """
    engine = gf.create_engine(backend="image_2layer", segment_length=0.7)
    r_rod_first = engine.solve(_bonded_world("a_rod", "b_ring"))
    r_ring_first = engine.solve(_bonded_world("z_rod", "a_ring"))

    z_a = r_rod_first.cluster_impedance("a_rod")[0]
    z_b = r_ring_first.cluster_impedance("z_rod")[0]
    assert z_a.real == pytest.approx(z_b.real, rel=1e-12)
    # Both entry points into the same cluster agree as well.
    assert r_rod_first.cluster_impedance("b_ring")[0].real == pytest.approx(
        z_a.real, rel=1e-12
    )


def test_cluster_impedance_uses_length_weighted_members() -> None:
    """F11: the cluster potential is the length-weighted member average.

    Pure :mod:`solver.result` unit test with synthetic data — a
    backend that still reports slightly disagreeing member potentials
    must not make ``cluster_impedance`` depend on the member order.
    Weights are the per-electrode total leakage lengths taken from
    ``point_sources``: here 3 m (``big``) and 1 m (``small``), so
    :math:`\\varphi_c = (3 \\cdot 10 + 1 \\cdot 20)/4 = 12.5` V and
    :math:`Z_c = 12.5 / 2 = 6.25\\,\\Omega`. Picking ``members[0]``
    would give 10/2 = 5 Ω or 20/2 = 10 Ω depending on the names.
    """
    members = ["big", "small"]
    result = FieldResult(
        backend="unit-test",
        frequencies=[50.0],
        electrode_potentials={"big": [10.0 + 0j], "small": [20.0 + 0j]},
        electrode_currents={"big": [1.0 + 0j], "small": [1.0 + 0j]},
        point_sources=[
            PointSource(
                position=(0.0, 0.0, 0.7), current=[1.0 + 0j],
                electrode_name="big", length=3.0,
            ),
            PointSource(
                position=(5.0, 0.0, 0.7), current=[1.0 + 0j],
                electrode_name="small", length=1.0,
            ),
        ],
        soil_resistivity=100.0,
        clusters={"big": members, "small": members},
    )
    assert result.cluster_impedance("big")[0].real == pytest.approx(6.25)
    # Order independence: reversing the member list changes nothing.
    result.clusters = {"big": members[::-1], "small": members[::-1]}
    assert result.cluster_impedance("small")[0].real == pytest.approx(6.25)


# ---------------------------------------------------------------------
# F26 / F29 — no image separation, no result
# ---------------------------------------------------------------------


def _surface_strip_world(z: float, wire_radius: float = 0.005) -> gf.World:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world, "strip", name="tape",
        start=(0.0, 0.0, z), end=(10.0, 0.0, z), wire_radius=wire_radius,
    )
    gf.create_source(world, attached_to="tape", magnitude=1.0)
    return world


@pytest.mark.parametrize("ds", [1.0, 0.5, 0.1])
def test_surface_laid_strip_is_rejected(ds: float) -> None:
    """F29: ``z = 0`` used to return a divergent, mesh-dependent R.

    0.14.1 solved this world silently to 810.35 Ω (``ds = 1.0``),
    413.49 Ω (0.5) and 97.67 Ω (0.1) — roughly doubling with every
    mesh halving, against a physical value of ~23 Ω. The image
    separation is zero, so there is nothing to salvage: hard error.
    """
    with pytest.raises(ValueError, match="soil surface"):
        gf.create_engine(backend="image", segment_length=ds).solve(
            _surface_strip_world(0.0)
        )


def test_near_surface_segment_is_rejected_via_wire_radius() -> None:
    """F26: ``2|z| <= a`` — the mirror image touches the conductor.

    Same criterion as the inductive guard in
    ``build_inductance_matrix``; 0.14.1 returned 47.05 Ω for the
    2 mm-deep ring of the audit without any diagnostic.
    """
    with pytest.raises(ValueError, match=r"wire_radius/2"):
        gf.create_engine(backend="image", segment_length=0.5).solve(
            _surface_strip_world(0.001, wire_radius=0.005)
        )


def test_electrode_above_the_surface_is_rejected() -> None:
    """F26: a conductor in air was mirrored into the soil silently.

    A ring at ``z = -100 m`` solved to 5.52 Ω in 0.14.1 because the
    kernel only ever sees ``|z|``. ``z`` is a depth, positive
    downwards, so a negative value is a sign-convention mistake.
    """
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world, "ring", name="airborne",
        center=(0.0, 0.0, -100.0), radius=4.0, wire_radius=0.005,
    )
    gf.create_source(world, attached_to="airborne", magnitude=1.0)
    with pytest.raises(ValueError, match="above the soil surface"):
        gf.create_engine(backend="image", segment_length=0.5).solve(world)


def test_surface_laid_ring_is_rejected_by_two_layer_backend() -> None:
    """F26: the same guard must hold for ``image_2layer``.

    ``image_2layer.py`` carried a verbatim copy of the clamped
    self-image line, and its ``n = 0`` term is the same perfect air
    mirror. 0.14.1 returned 164.08 Ω for this ring at ``z = 0``
    (physical limit ~11 Ω).
    """
    world = gf.create_world(
        soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=3.0)
    )
    gf.create_electrode(
        world, "ring", name="r",
        center=(0.0, 0.0, 0.0), radius=4.0, wire_radius=0.005,
    )
    gf.create_source(world, attached_to="r", magnitude=1.0)
    with pytest.raises(ValueError, match="soil surface"):
        gf.create_engine(backend="image_2layer", segment_length=0.5).solve(
            world
        )


def test_shared_kernel_guard_protects_every_backend() -> None:
    """F26: the guard sits in the kernel shared by mom / bem / cim.

    ``_self_corrected_kernel`` is imported by ``mom``, ``mom_sommerfeld``,
    ``bem``, ``cim`` and ``mutual``, so guarding it closes the hole for
    the whole family rather than for ``solve_image`` alone. Both
    branches are checked: ``2|z| <= a`` (segment at 2 mm with a 5 mm
    wire) and ``z <= 0``.
    """
    lengths = np.array([0.5, 0.5])
    radii = np.array([0.005, 0.005])
    excitation = np.array([0.5, 0.5])

    shallow = np.array([[0.0, 0.0, 0.002], [1.0, 0.0, 0.002]])
    with pytest.raises(ValueError, match="soil-surface plane"):
        _self_corrected_kernel(shallow, lengths, radii, excitation, 100.0)

    airborne = np.array([[0.0, 0.0, -0.7], [1.0, 0.0, -0.7]])
    with pytest.raises(ValueError, match="above the soil surface"):
        _self_corrected_kernel(airborne, lengths, radii, excitation, 100.0)

    # Without owner labels the message must still locate the segment
    # (index plus coordinates) — the sibling backends call positionally.
    with pytest.raises(ValueError, match=r"segment 0 at \(x, y, z\)"):
        _self_corrected_kernel(airborne, lengths, radii, excitation, 100.0)
    # With labels it names what the user wrote.
    with pytest.raises(ValueError, match=r"electrode 'tape'"):
        _self_corrected_kernel(
            airborne, lengths, radii, excitation, 100.0,
            owner_names=["electrode 'tape'"] * 2,
        )


@pytest.mark.parametrize(
    "backend,soil",
    [
        ("image", gf.HomogeneousSoil(resistivity=100.0)),
        ("image_2layer", gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=3.0)),
        ("mom", gf.HomogeneousSoil(resistivity=100.0)),
        ("bem", gf.HomogeneousSoil(resistivity=100.0)),
        ("cim", gf.HomogeneousSoil(resistivity=100.0)),
        ("mom_sommerfeld", gf.HomogeneousSoil(resistivity=100.0)),
    ],
)
def test_airborne_electrode_is_rejected_by_every_backend(
    backend: str, soil
) -> None:
    """Audit 2026-07-29: the ``z <= 0`` hole was open in 4 of 6 backends.

    ``_check_segment_depths`` runs from ``solve_image`` /
    ``solve_image_2layer`` only, and the shared kernel guard tested
    above checked ``2|z| <= a`` alone — a *negative* ``z`` passes that.
    So the airborne ring of F26 still solved silently in four
    backends::

        image / image_2layer          -> ValueError
        mom / bem / cim / mom_somm.   -> Z = 5.5162 Ohm, warnings = []

    ``compare_engines`` on an airborne electrode therefore raised for
    one half of the family and returned a number for the other. The
    ``z <= 0`` rejection now lives in the shared kernel.
    """
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world, "ring", name="airborne",
        center=(0.0, 0.0, -100.0), radius=4.0, wire_radius=0.005,
    )
    gf.create_source(world, attached_to="airborne", magnitude=1.0)
    with pytest.raises(ValueError, match="above the soil surface"):
        gf.create_engine(backend=backend, segment_length=0.5).solve(world)


def test_segment_straddling_the_surface_is_rejected() -> None:
    """Audit 2026-07-29: midpoint-only depth checks miss a rod in air.

    F26 asked for a validation that *every electrode segment* has
    ``z > 0``; only midpoints were checked. A 1.2 m rod whose top
    0.5 m is above the surface has its single segment's midpoint at
    ``z = +0.1 m``, so it passed every guard and solved to
    106.2444 Ω with nothing but a ``ShallowSegmentWarning`` claiming
    the entry was overestimated by 65 % — a figure computed from the
    *horizontal* closed form for a strictly vertical segment.
    """
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world, "rod", name="poking_out",
        position=(0.0, 0.0, -0.5), length=1.2, wire_radius=0.008,
    )
    gf.create_source(world, attached_to="poking_out", magnitude=1.0)
    with pytest.raises(ValueError, match="crosses the soil surface"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ShallowSegmentWarning)
            gf.create_engine(backend="image", segment_length=5.0).solve(world)


def test_rod_driven_from_the_surface_stays_legal() -> None:
    """Control for the extent guard: ``z_top == 0`` is the normal rod.

    A driven rod starting at ``position=(x, y, 0)`` has its top end
    *exactly* in the surface plane. Source plus image form the
    length-``2 L`` line Dwight's closed form is derived from, so the
    potential is perfectly finite and this must not be rejected — the
    extent guard therefore tests ``z_top < 0``, not ``z_top <= 0``.
    """
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world, "rod", name="g1",
        position=(0.0, 0.0, 0.0), length=1.5, wire_radius=0.008,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = gf.create_engine(
            backend="image", segment_length=0.05
        ).solve(world)
    # Dwight's driven rod, rho/(2 pi L) [ln(4L/a) - 1] = 59.63 Ohm for
    # 100 Ohm-m, L = 1.5 m, a = 8 mm; the solver returns 58.489 Ohm
    # (-1.9 %, the uniform-current ansatz residual).
    dwight = (
        100.0 / (2.0 * np.pi * 1.5) * (np.log(4.0 * 1.5 / 0.008) - 1.0)
    )
    assert result.grounding_impedance("g1")[0].real == pytest.approx(
        dwight, rel=0.05
    )


def test_depth_error_names_the_conductor_not_the_pseudo_node() -> None:
    """Audit 2026-07-29: the message must name what the user wrote.

    A galvanically coupled distributed conductor's leakage segments
    carry the reserved pseudo-node name
    ``__cond_<name>__seg_<k>`` in ``_Segment.electrode_name``. The
    depth error used to report ``electrode '__cond_pen__seg_0'`` for a
    conductor the user called ``pen``. ``_Segment.conductor_name`` is
    available, so the message now reads ``conductor 'pen'``.
    """
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world, "rod", name="a",
        position=(0.0, 0.0, 0.7), length=2.0, wire_radius=0.008,
    )
    gf.create_electrode(
        world, "rod", name="b",
        position=(20.0, 0.0, 0.7), length=2.0, wire_radius=0.008,
    )
    # Explicit endpoints put the conductor axis into the surface plane
    # while both end electrodes stay legally buried.
    world.add_conductor(Conductor(
        name="pen",
        start=(0.0, 0.0, 0.0), end=(20.0, 0.0, 0.0),
        start_electrode="a", end_electrode="b",
        cross_section=50e-6, discretize_segment_length=5.0,
        coupling_to_soil="galvanic",
    ))
    gf.create_source(world, attached_to="a", magnitude=1.0)
    with pytest.raises(ValueError, match=r"conductor 'pen'") as excinfo:
        gf.create_engine(backend="image", segment_length=0.5).solve(world)
    assert "__cond_pen__seg_" not in str(excinfo.value)


def test_shallow_segment_warns_and_quantifies_the_bias() -> None:
    """F29: the shallow but legal regime is now diagnosed, not hidden.

    A tape at ``z = 0.01 m`` with the default-ish ``ds = 0.5 m``
    satisfies ``L > 4 z``, where the point image at distance ``2 z``
    overestimates the diagonal image term of the exact image line
    ``(2/L)·arsinh(L/(4z))`` by a large factor. 0.14.1 returned
    35.49116014344861 Ω against Dwight's 21.02 Ω (+69 %) with zero
    warnings.

    0.15.0 returns 35.4911601434486 — the same number to 1 ulp, not
    bit-identical (see the module docstring: the reported potential is
    now a BLAS dot product rather than ``np.mean``). Only the
    diagnostic is new.
    """
    with pytest.warns(ShallowSegmentWarning, match="shallower than"):
        result = gf.create_engine(backend="image", segment_length=0.5).solve(
            _surface_strip_world(0.01)
        )
    z = result.grounding_impedance("tape")[0].real
    assert _ulps_apart(z, 35.49116014344861, budget=4), (
        f"{z!r} is more than 4 ulps from the 0.14.1 value "
        "35.49116014344861"
    )


def test_shallow_warning_states_a_bias_valid_for_the_case_it_fires_on() -> None:
    """Audit 2026-07-29: the quoted percentage must be the right formula.

    The message quantifies the bias with the *horizontal* closed form
    ``(L/4z)/arsinh(L/4z) - 1``. Before the extent guard existed the
    worst offender could be a strictly **vertical** segment, for which
    the vertical closed form ``(1/L)·ln((4z+L)/(4z-L))`` has no real
    value at all when ``4z < L`` and the point form *under*-estimates
    — so both the sign and the number were wrong for exactly the class
    that triggered the warning.

    With the extent guard in place a segment can only reach the
    warning if ``L > 4 z`` *and* ``L·|e_z| <= 2 z``, which forces
    ``|e_z| < 1/2`` (inclination below 30 degrees). The message states
    the inclination it assumed; this test pins both facts.
    """
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(
        world, "ring", name="g1",
        center=(0.0, 0.0, 0.8), radius=25.0, wire_radius=0.005,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    with pytest.warns(ShallowSegmentWarning) as record:
        gf.create_engine(backend="image", segment_length=5.0).solve(world)
    message = str(record[0].message)

    # Plain ring electrode, no distributed conductor in sight.
    assert "electrode 'g1'" in message
    assert "L = 4.909 m at z = 0.8 m" in message
    # Horizontal arc chords: the quoted formula is the applicable one.
    assert "inclination |e_z| = 0" in message
    assert "horizontal segment" in message
    ratio = 4.9087385212340517 / (4.0 * 0.8)
    expected = ratio / np.arcsinh(ratio) - 1.0
    assert f"by {expected * 100.0:.0f} %" in message
    assert "by 26 %" in message
    # The asymmetry across the family is stated, not hidden.
    assert "mom / bem / cim / mom_sommerfeld" in message


def test_vertical_segments_can_never_reach_the_shallow_warning() -> None:
    """The two guards interlock: no vertical worst offender survives.

    For a vertical segment ``|e_z| = 1``, so the extent guard requires
    ``L <= 2 z`` while the shallow warning requires ``L > 4 z``. The
    two are mutually exclusive, which is what makes the horizontal
    bias formula in the message sound. Checked directly on the
    validator with a hand-built vertical segment.
    """
    from groundfield.solver.image import _check_segment_depths, _Segment

    seg = _Segment(
        midpoint=np.array([0.0, 0.0, 0.1]),
        length=1.2,
        electrode_name="rod",
        wire_radius=0.008,
        direction=np.array([0.0, 0.0, 1.0]),
    )
    # L = 1.2 > 4 z = 0.4, so 0.14.1..0.15.0-dev warned; now it raises.
    with pytest.raises(ValueError, match="crosses the soil surface"):
        _check_segment_depths([seg], where="unit-test")

    # The deepest vertical segment that the extent guard admits
    # (z_top == 0, L == 2 z) is silent, as the interlock argument
    # requires.
    ok = _Segment(
        midpoint=np.array([0.0, 0.0, 0.6]),
        length=1.2,
        electrode_name="rod",
        wire_radius=0.008,
        direction=np.array([0.0, 0.0, 1.0]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_segment_depths([ok], where="unit-test")


def test_ordinary_burial_depth_is_silent() -> None:
    """Control: the guard must not fire for ordinary geometries.

    A 10 m tape at 0.5 m depth with 0.5 m segments has
    ``L = 4 z / 4`` and a clean image separation, so it neither
    raises nor warns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = gf.create_engine(backend="image", segment_length=0.5).solve(
            _surface_strip_world(0.5)
        )
    z = result.grounding_impedance("tape")[0].real
    assert _ulps_apart(z, Z_STRIP_10M_0_14_1, budget=4)


# ---------------------------------------------------------------------
# cluster_impedance must not drop a member's current
# ---------------------------------------------------------------------


def test_cluster_impedance_sums_current_over_all_members() -> None:
    """Audit 2026-07-29: the ``usable`` filter turned a KeyError into
    a silently wrong number.

    F11 introduced a filter that drops cluster members without both
    potential *and* current data, and applied it to the current sum as
    well as to the potential average. v0.14.1 summed over ``members``
    and raised ``KeyError`` when a member was missing; the finding
    never asked for ``KeyError`` tolerance, and swallowing it changed
    a loud failure into a wrong answer::

        potentials {"a": 10 V}, currents {"a": 1 A, "b": 1 A},
        cluster ["a", "b"]
            0.14.1 -> 5.0 Ohm   (10 V / 2 A)
            0.15.0 dev -> 10.0 Ohm   (member b's 1 A dropped)

    The average now runs over the potential-carrying members while the
    current sum runs over every member with current data, so both
    properties hold at once.
    """
    result = FieldResult(
        backend="unit-test",
        frequencies=[50.0],
        electrode_potentials={"a": [10.0 + 0j]},
        electrode_currents={"a": [1.0 + 0j], "b": [1.0 + 0j]},
        clusters={"a": ["a", "b"], "b": ["a", "b"]},
    )
    assert result.cluster_impedance("a")[0].real == pytest.approx(5.0)
    # Entering through the potential-less member gives the same physics.
    assert result.cluster_impedance("b")[0].real == pytest.approx(5.0)
    # A cluster with no potential data anywhere is still a KeyError.
    orphan = FieldResult(
        backend="unit-test",
        frequencies=[50.0],
        electrode_potentials={},
        electrode_currents={"a": [1.0 + 0j]},
        clusters={"a": ["a"]},
    )
    with pytest.raises(KeyError):
        orphan.cluster_impedance("a")


# ---------------------------------------------------------------------
# Export hygiene
# ---------------------------------------------------------------------


def test_shallow_segment_warning_is_reachable_from_the_solver_package() -> None:
    """The category a user must name in ``simplefilter`` must be public.

    ``ShallowSegmentWarning`` was only importable from
    ``groundfield.solver.image``, i.e. from a backend implementation
    module, although silencing it is the documented remedy. It is now
    re-exported from ``groundfield.solver`` and listed in
    ``docs/api/solver.md``.
    """
    import groundfield.solver as solver
    from groundfield.solver.image import (
        ShallowSegmentWarning as from_backend,
    )

    assert solver.ShallowSegmentWarning is from_backend
    assert "ShallowSegmentWarning" in solver.__all__
    assert issubclass(from_backend, UserWarning)
