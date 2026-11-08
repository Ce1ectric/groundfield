"""Exact line kernel on the cross-layer path (ADR-0007): independent of ds.

The layered correction of the two-layer Green's function contains the
image of a source mirrored in the interface. For wires close to the
interface that image is closer than a segment length, and its point form
(segment midpoint to image midpoint) is the midpoint rule of a peak
narrower than the segment. With the exact line kernel the image term is a
Galerkin integral (review pass 10, 10.28): a ring 5 cm below the interface
gave 30.5 Ω at ds = 0.5 m and 55.7 Ω at ds = 0.025 m before, now 55.6-55.7 Ω
at every ds; the point kernel keeps the old behaviour (29.7 → 54.7 Ω).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

_FINDING = pytest.mark.finding("10.28")


def _z(world: gf.World, name: str, ds: float, kernel: str = "line") -> float:
    engine = gf.create_engine(segment_length=ds, kernel=kernel)
    return engine.solve(world).cluster_impedance(name)[0].real


def _ring_below_interface() -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=0.75))
    gf.create_electrode(w, "ring", name="g", center=(0.0, 0.0, 0.8), radius=3.0)
    gf.create_source(w, attached_to="g", magnitude=1.0)
    return w


def _rod_through_interface() -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=2.0))
    gf.create_electrode(
        w, "rod", name="g", position=(0.0, 0.0, 0.5), length=3.0, wire_radius=0.008
    )
    gf.create_source(w, attached_to="g", magnitude=1.0)
    return w


@_FINDING
def test_ring_just_below_the_interface_is_independent_of_ds() -> None:
    values = [_z(_ring_below_interface(), "g", ds) for ds in (0.5, 0.25, 0.1)]
    assert max(values) / min(values) - 1.0 < 2e-3, values
    assert values[0] == pytest.approx(55.7, rel=5e-3)


@_FINDING
def test_rod_through_the_interface_is_independent_of_ds() -> None:
    values = [_z(_rod_through_interface(), "g", ds) for ds in (1.0, 0.5, 0.25)]
    assert max(values) / min(values) - 1.0 < 5e-4, values


def test_point_kernel_keeps_its_numbers_near_the_interface() -> None:
    """``kernel="point"`` reproduces 0.15.0 where its diagonal rule fits.

    At ds = 0.5 m the point form's diagonal rule (+-0.19 m in depth) reaches
    across the interface 5 cm above the ring: 0.15.0 returned 29.68 Ohm
    (line kernel 55.64 Ohm, -47 %); since audit round 2 the solve raises
    (docb item, ``tests/test_cross_layer_point_form.py``). At ds = 0.05 m
    the number is that of 0.15.0.
    """
    from groundfield.solver.registry import BackendCapabilityError

    with pytest.raises(BackendCapabilityError, match="kernel=None or 'line'"):
        _z(_ring_below_interface(), "g", 0.5, "point")
    assert _z(_ring_below_interface(), "g", 0.05, "point") == pytest.approx(
        54.6429, abs=5e-4
    )


# ---------------------------------------------------------------------
# Audit A3S-01: the regularisation of the closed-form 1/R terms
# ---------------------------------------------------------------------
#
# The layered correction evaluates its closed-form 1/R terms with a clamp at
# min(h_1, s + z + z_s)/200 (25 mm for h_1 = 5 m), meant for coincident
# points. The solver evaluates the diagonal at the wire radius s = a and then
# exchanges the *unclamped* point image coeff/sqrt(a^2 + d^2) of the
# interface for its Galerkin integral, so a part within h_1/400 of the
# interface lost 68-84 % of its layered correction: a ring at h_1 = 5 m read
# -5.29 Ohm with the default engine. The solver now passes its own
# regularisation length (``min_distance``: the wire radius on the diagonal,
# the 1 mm of its point forms off it).


def _ring_at(depth: float, rho_1: float, rho_2: float, h_1: float) -> float:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=h_1))
    gf.create_electrode(
        w, "ring", name="e", center=(0.0, 0.0, depth), radius=3.0, wire_radius=0.005
    )
    gf.create_source(w, attached_to="e")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = w.solve(gf.Engine(segment_length=0.5))
    return result.cluster_impedance("e")[0].real


@_FINDING
@pytest.mark.parametrize("h_1", [5.0, 10.0])
@pytest.mark.parametrize(("rho_1", "rho_2"), [(1000.0, 30.0), (30.0, 1000.0)])
def test_ring_resistance_is_continuous_through_a_deep_interface(
    rho_1: float, rho_2: float, h_1: float
) -> None:
    """Default engine: 4.18 -> -5.29 Ohm (1000/30), 6.80 -> 322 Ohm (30/1000)."""
    # audit A3S-01
    above = _ring_at(h_1 - 1e-6, rho_1, rho_2, h_1)  # Tagg/Sunde series path
    at = _ring_at(h_1, rho_1, rho_2, h_1)  # ADR-0007 cross-layer path
    below = _ring_at(h_1 + 1e-6, rho_1, rho_2, h_1)
    assert at > 0.0 and below > 0.0, (above, at, below)
    assert abs(at / above - 1.0) < 1e-3, (above, at, below)
    assert abs(below / above - 1.0) < 1e-3, (above, at, below)


def _image_solution_lower(
    s: float, z: float, z_s: float, h_1: float, rho_1: float, rho_2: float
) -> float:
    """Two-layer kernel (rho/2 convention) for two points in the lower layer.

    Sunde (1968), sect. 3.5: ``rho_2/2 [1/r - K/r' + (1 - K^2) sum_n K^n /
    r_n]`` with ``K = (rho_2 - rho_1)/(rho_2 + rho_1)``, the interface image
    ``r'`` at ``z + z_s - 2 h_1`` and the transmitted images ``r_n`` at
    ``z + z_s + 2 n h_1``; summed to convergence.
    """
    K = (rho_2 - rho_1) / (rho_2 + rho_1)
    total = 1.0 / math.hypot(s, z - z_s) - K / math.hypot(s, z + z_s - 2.0 * h_1)
    n, term = 0, 1.0
    while abs(term) > 1e-18:
        term = (1.0 - K * K) * K**n / math.hypot(s, z + z_s + 2.0 * n * h_1)
        total += term
        n += 1
    return 0.5 * rho_2 * total


def _image_solution_cross(
    s: float, z_u: float, z_l: float, h_1: float, rho_1: float, rho_2: float
) -> float:
    """Two-layer kernel (rho/2 convention) between the two layers.

    Sunde (1968), sect. 3.5: ``rho_1 (1 + K)/2 sum_n K^n [1/r_n^- +
    1/r_n^+]`` at ``2 n h_1 + z_l -+ z_u``.
    """
    K = (rho_2 - rho_1) / (rho_2 + rho_1)
    total, n, term = 0.0, 0, 1.0
    while abs(term) > 1e-18:
        term = K**n * (
            1.0 / math.hypot(s, 2.0 * n * h_1 + z_l - z_u)
            + 1.0 / math.hypot(s, 2.0 * n * h_1 + z_l + z_u)
        )
        total += term
        n += 1
    return 0.5 * rho_1 * (1.0 + K) * total


def _homogeneous(s: float, z: float, z_s: float, rho: float) -> float:
    return 0.5 * rho * (1.0 / math.hypot(s, z - z_s) + 1.0 / math.hypot(s, z + z_s))


@_FINDING
@pytest.mark.parametrize("h_1", [5.0, 10.0])
@pytest.mark.parametrize(("rho_1", "rho_2"), [(1000.0, 30.0), (100.0, 1000.0)])
@pytest.mark.parametrize("eps", [5e-7, 2e-3, 8e-3, 0.013])
def test_diagonal_correction_at_the_wire_radius_matches_the_image_solution(
    eps: float, rho_1: float, rho_2: float, h_1: float
) -> None:
    """The diagonal entry of a layer-2 part whose midpoint is eps below h_1.

    Evaluated as the solver does (s = a = 8 mm, ``min_distance = a``): equal
    to the image solution minus the homogeneous rho_2 baseline. With the
    default clamp it was -68 % (h_1 = 5 m) and -84 % (h_1 = 10 m) at 0.5 um,
    -28 % / -64 % at 8 mm (audit script p04b).
    """
    # audit A3S-01
    from groundfield.coupling.layered_green import (
        two_layer_layered_correction_real_space,
    )

    a, z = 0.008, h_1 + eps
    code = two_layer_layered_correction_real_space(
        s=a,
        z=z,
        z_s=z,
        rho_1=rho_1,
        rho_2=rho_2,
        h_1=h_1,
        rho_baseline=rho_2,
        min_distance=a,
    )
    ref = _image_solution_lower(a, z, z, h_1, rho_1, rho_2) - _homogeneous(
        a, z, z, rho_2
    )
    assert code == pytest.approx(ref, rel=1e-9)


@_FINDING
@pytest.mark.parametrize("eps", [0.0126, 0.05, 0.25])
def test_min_distance_leaves_every_farther_image_bit_for_bit(eps: float) -> None:
    """Only R below the old clamp h_1/200 = 25 mm changes; the rest is bit-identical."""
    # audit A3S-01
    from groundfield.coupling.layered_green import (
        two_layer_layered_correction_group,
        two_layer_layered_correction_real_space,
        two_layer_real_space_kernel,
    )

    kw = dict(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    z = 5.0 + eps  # interface image at 2 eps >= 25.2 mm
    old = two_layer_layered_correction_real_space(
        s=0.008, z=z, z_s=z, rho_baseline=30.0, **kw
    )
    new = two_layer_layered_correction_real_space(
        s=0.008, z=z, z_s=z, rho_baseline=30.0, min_distance=0.008, **kw
    )
    assert new == old
    s = np.array([0.03, 0.5, 2.0, 40.0])  # primary R = s >= 30 mm
    np.testing.assert_array_equal(
        two_layer_layered_correction_group(s, z, z, rho_baseline=30.0, **kw),
        two_layer_layered_correction_group(
            s, z, z, rho_baseline=30.0, min_distance=1e-3, **kw
        ),
    )
    assert two_layer_real_space_kernel(0.03, z, z, **kw) == two_layer_real_space_kernel(
        0.03, z, z, min_distance=1e-3, **kw
    )


@_FINDING
def test_pairs_closer_than_the_clamp_match_the_image_solution() -> None:
    """Off-diagonal entries closer than h_1/200 (fine meshes, deep interfaces).

    A pair straddling h_1 = 30.1 m 0.125 m apart (old clamp 0.15 m): the group
    correction read 93.6 instead of 112.8 Ohm·(2π); a probe 1 cm from a
    layer-2 source (h_1 = 5 m, clamp 25 mm) and the kernel of
    ``mom_sommerfeld``'s off-diagonal entries likewise.
    """
    # audit A3S-01
    from groundfield.coupling.layered_green import (
        two_layer_layered_correction_group,
        two_layer_probe_matrix,
        two_layer_real_space_kernel,
    )

    rho_1, rho_2 = 1000.0, 30.0
    z_u, z_l, h_1 = 30.05, 30.175, 30.1
    code = two_layer_layered_correction_group(
        np.array([0.0]),
        z_u,
        z_l,
        rho_1=rho_1,
        rho_2=rho_2,
        h_1=h_1,
        rho_baseline=rho_2,
        min_distance=1e-3,
    )[0]
    ref = _image_solution_cross(0.0, z_u, z_l, h_1, rho_1, rho_2) - _homogeneous(
        0.0, z_u, z_l, rho_2
    )
    assert code == pytest.approx(ref, rel=1e-8)

    src = np.array([[0.0, 0.0, 7.0]])
    probe = np.array([[0.01, 0.0, 7.0]])
    ref = _image_solution_lower(0.01, 7.0, 7.0, 5.0, rho_1, rho_2)
    G = two_layer_probe_matrix(probe, src, rho_1=rho_1, rho_2=rho_2, h_1=5.0)
    assert G[0, 0] == pytest.approx(ref / (2.0 * math.pi), rel=1e-8)
    G_msf = two_layer_real_space_kernel(
        0.01, 7.0, 7.0, rho_1=rho_1, rho_2=rho_2, h_1=5.0, min_distance=1e-3
    )
    assert G_msf == pytest.approx(ref, rel=1e-8)


@_FINDING
@pytest.mark.parametrize("bad", [0.0, -1e-3, float("nan"), float("inf")])
def test_min_distance_must_be_finite_and_positive(bad: float) -> None:
    # audit A3S-01
    from groundfield.coupling.layered_green import two_layer_real_space_kernel

    with pytest.raises(ValueError, match="min_distance"):
        two_layer_real_space_kernel(
            0.01, 7.0, 7.0, rho_1=1000.0, rho_2=30.0, h_1=5.0, min_distance=bad
        )
