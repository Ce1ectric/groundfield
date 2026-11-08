"""Short segments and interface contact (review pass 10, 10.32; audit round 2).

* The collocation self term :math:`2\\ln(L/a)/L` of the point kernel tends to
  zero as :math:`L \\to a`: a sliver left by the split at a layer interface
  acted as a short, so ``mom`` gave a rod whose tip lies 9 mm below
  :math:`h_1` 27 % less resistance than the rod 1 mm above it, and a rod
  whose tip lies 1 mm below :math:`h_1` raised a ``ValueError``. Below
  :math:`L/a = 2.99` the Galerkin self term takes over.
* Audit round 2 (A2S-01): the intermediate rule that merged a part shorter
  than the wire diameter into its neighbour dropped the contact with the
  layer beyond the interface: a 4.5 m rod (a = 8 mm) in 1000 over 30 Ω·m read
  195.0 Ω with its tip 15.9 mm into the lower layer and 128.6 Ω at 16.1 mm
  (+52 %), in ``image_2layer`` and in the ``mom`` family. Every interface cut
  is now kept, however short the part beyond it; the Galerkin self term and
  the pair integrals of the line kernel are evaluated without cancellation
  for parts much shorter than the wire radius (two parts of 1e-11 m gave a
  pair integral of 0 instead of 125 1/m). The impedance is continuous in the
  penetration depth; the step that remains is the onset of the contact
  between a tip that ends at the interface and one that reaches past it.
* ``ceil(L / ds)`` added a segment when ``L / ds`` exceeded an integer only
  through floating-point noise (GEN-1: 52 instead of 50 segments).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf

_FINDING = pytest.mark.finding("10.32")
_SOIL = dict(rho_1=1000.0, rho_2=100.0, h_1=5.0)


def _rod_z(bottom: float, backend: str, **engine_kw) -> float:
    w = gf.create_world(soil=gf.TwoLayerSoil(**_SOIL))
    gf.create_electrode(
        w, "rod", name="r", position=(0, 0, 0.5), length=bottom - 0.5, wire_radius=0.008
    )
    gf.create_source(w, attached_to="r", magnitude=1.0)
    eng = gf.create_engine(backend=backend, segment_length=0.5, **engine_kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return w.solve(eng).cluster_impedance("r")[0].real


@_FINDING
def test_rod_tip_just_below_the_interface_solves_and_touches_the_layer() -> None:
    """A tip 1 mm below h_1 raised (segment shorter than the wire radius).

    It solves and is in contact with the lower, more conductive layer: below
    the rod that ends at h_1 and equal to the contact limit of a tip that
    reaches only 1 µm past the interface (audit A2S-01; the merge rule of
    the base read the tip 1 mm below h_1 like the rod that ends there).
    """
    # audit A2S-01
    at_interface = _rod_z(5.0, "image_2layer")
    one_mm = _rod_z(5.001, "image_2layer")
    one_um = _rod_z(5.0 + 1e-6, "image_2layer")
    # 1000 over 100 Ohm m: the contact lowers Z by 11 % (199.7 -> 177.5 Ohm)
    assert one_mm < 0.95 * at_interface
    assert one_mm == pytest.approx(one_um, rel=2e-3)


@_FINDING
def test_mom_resistance_decreases_with_rod_length_across_the_interface() -> None:
    """R1-04: mom gave 196.6 -> 144.2 -> 166.8 Ohm for bottoms 4.99 / 5.009 / 5.02 m."""
    bottoms = [4.99, 5.009, 5.02, 5.05, 5.2]
    z = [_rod_z(b, "mom") for b in bottoms]
    assert all(b < a for a, b in zip(z, z[1:], strict=False)), z


@pytest.mark.parametrize("kernel", ["point", "line"])
def test_image_resistance_decreases_with_rod_length_across_the_interface(
    kernel: str,
) -> None:
    bottoms = [4.99, 5.001, 5.009, 5.02, 5.05, 5.2]
    z = [_rod_z(b, "image_2layer", kernel=kernel) for b in bottoms]
    assert all(b < a for a, b in zip(z, z[1:], strict=False)), z


@_FINDING
def test_point_self_term_takes_the_galerkin_form_for_short_segments() -> None:
    from groundfield.solver._line_kernel import (
        SELF_TERM_CROSSOVER,
        galerkin_self,
        point_self_term,
    )

    a = np.full(7, 0.008)
    ratio = np.array([1.05, 1.13, 1.5, 2.0, 2.5, 2.9, 2.98])
    L = ratio * a
    np.testing.assert_array_equal(point_self_term(L, a), galerkin_self(L, a))
    long_ratio = np.array([2.991, 3.0, 5.0, 62.5, 1e3])
    L = long_ratio * a[:5]
    np.testing.assert_array_equal(
        point_self_term(L, a[:5]), 2.0 * np.log(L / a[:5]) / L
    )
    # continuous at the switch
    a0 = np.array([0.01])
    below = point_self_term(np.array([SELF_TERM_CROSSOVER * (1 - 1e-9)]) * a0, a0)
    above = point_self_term(np.array([SELF_TERM_CROSSOVER * (1 + 1e-9)]) * a0, a0)
    assert below[0] == pytest.approx(above[0], rel=1e-8)


@_FINDING
@pytest.mark.parametrize("part", [1e-9, 0.1, 1.9, 2.1])
def test_every_interface_cut_is_kept(part: float) -> None:
    """A part beyond an interface is a segment of its own, however short.

    ``part`` is the length beyond the cut in wire radii (1e-9 m for the
    first case). The base dropped cuts closer than two wire radii (audit
    A2S-01).
    """
    # audit A2S-01
    a, h = 0.008, (5.0,)
    p = part if part < 1e-6 else part * a
    tip = gf.RodElectrode(name="r", position=(0, 0, 0.5), length=4.5 + p, wire_radius=a)
    segs = tip.segments(0.5, h)
    below = segs.midpoints[:, 2] >= 5.0
    assert below.sum() == 1
    assert segs.lengths[below][0] == pytest.approx(p, rel=1e-6, abs=1e-15)
    assert segs.ends[below][0][2] == pytest.approx(5.0 + p)
    # the same at the head of a rod that starts just above the interface
    head = gf.RodElectrode(
        name="r", position=(0, 0, 5.0 - p), length=3.0, wire_radius=a
    )
    segs = head.segments(0.5, h)
    above = segs.midpoints[:, 2] < 5.0
    assert above.sum() == 1
    assert segs.lengths[above][0] == pytest.approx(p, rel=1e-6, abs=1e-15)


@_FINDING
def test_the_thin_wire_guard_concerns_the_segment_length_only() -> None:
    """A part shorter than the wire radius is accepted at an interface only."""
    # audit A2S-01
    w = gf.create_world(soil=gf.TwoLayerSoil(**_SOIL))
    gf.create_electrode(
        w, "rod", name="r", position=(0, 0, 0.5), length=4.501, wire_radius=0.008
    )
    gf.create_source(w, attached_to="r", magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = w.solve(gf.create_engine(backend="image_2layer", segment_length=0.5))
    assert min(ps.length for ps in res.point_sources) == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="wire_radius"):
        w.solve(gf.create_engine(backend="image_2layer", segment_length=0.008))


@_FINDING
def test_segment_count_ignores_floating_point_noise() -> None:
    # Star arms at 120 / 240 degrees: 2.9999999999999996 and 3.000000000000001 m.
    star = gf.StarElectrode(name="s", center=(0, 0, 0.6), n_arms=3, arm_length=3.0)
    assert star.segment_count(0.5) == 18
    edge = 3.0000000000000004
    poly = gf.PolylineElectrode(
        name="p", vertices=[(0, 0, 0.5), (edge, 0, 0.5)], closed=False
    )
    assert poly.segment_count(1.5) == 2
    rod = gf.RodElectrode(name="r", position=(0, 0, 0.5), length=0.1 + 0.2)
    assert rod.segment_count(0.1) == 3
    assert math.isclose(rod.segments(0.1).lengths[0], 0.1)


# ---------------------------------------------------------------------
# Audit round 2, A2S-01: the contact with a layer is never dropped
# ---------------------------------------------------------------------

_WIRE_RADIUS = 0.008  # 2a = 16 mm
_CONFIGS = [
    ("image_2layer", "line"),
    ("image_2layer", "point"),
    ("mom", None),
    ("mom_sommerfeld", None),
]


def _tip_z(penetration: float, backend: str, kernel: str | None = None) -> float:
    """4.5 m rod 0.5-5.0 m; h_1 lies ``penetration`` m above its tip (1000/30)."""
    w = gf.create_world(
        soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0 - penetration)
    )
    gf.create_electrode(
        w,
        "rod",
        name="r",
        position=(0.0, 0.0, 0.5),
        length=4.5,
        wire_radius=_WIRE_RADIUS,
    )
    gf.create_source(w, attached_to="r")
    kw: dict = {"backend": backend, "segment_length": 0.5}
    if kernel is not None:
        kw["kernel"] = kernel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return w.solve(gf.Engine(**kw)).cluster_impedance("r")[0].real


@_FINDING
@pytest.mark.parametrize(("backend", "kernel"), _CONFIGS)
def test_rod_resistance_is_continuous_in_h1_at_the_former_sliver_threshold(
    backend: str, kernel: str | None
) -> None:
    """0.2 mm of h_1 across 2a moved the base by +52 % (195.0 vs 128.6 Ohm)."""
    # audit A2S-01
    just_below = _tip_z(0.0159, backend, kernel)  # p = 15.9 mm
    just_above = _tip_z(0.0161, backend, kernel)  # p = 16.1 mm
    assert abs(just_below / just_above - 1.0) < 2e-3, (just_below, just_above)


@_FINDING
@pytest.mark.parametrize(("backend", "kernel"), _CONFIGS)
def test_rod_resistance_is_smooth_in_the_tip_penetration(
    backend: str, kernel: str | None
) -> None:
    """Z(p) falls monotonically and without steps for every p > 0.

    The tip p below h_1 from 10 pm to 25 mm: no relative step larger than
    1 % per mm of penetration (the base: 33 % per 0.1 mm at p = 2a; the
    line kernel without the short-pair integrals: 72.7 instead of
    133.5 Ohm for p = 1e-11 m), and the contact limit is reached for
    p -> 0+ (the parts of 10 pm and 1 µm agree to 1e-4).
    """
    # audit A2S-01
    ps = [1e-11, 1e-6, 1e-4, 1e-3, 4e-3, 8e-3, 12e-3, 15.9e-3, 16.1e-3, 20e-3, 25e-3]
    z = [_tip_z(p, backend, kernel) for p in ps]
    assert z[0] == pytest.approx(z[1], rel=1e-4)
    for k in range(len(ps) - 1):
        (p0, z0), (p1, z1) = (ps[k], z[k]), (ps[k + 1], z[k + 1])
        assert z1 <= z0 * (1.0 + 1e-5), (p0, z0, p1, z1)
        if p1 - p0 >= 1e-4:
            assert abs(z1 / z0 - 1.0) / ((p1 - p0) * 1e3) < 0.01, (p0, z0, p1, z1)


@_FINDING
def test_a_tip_twelve_millimetres_into_a_conductive_layer_is_felt() -> None:
    """p = 12 mm (1.5 a): the base ignored the lower layer completely.

    The bound on the 12 -> 16.1 mm step was 2 % while the h_1/200 clamp of
    the diagonal (audit A3S-01) flattened the curve below 12.5 mm; the solid
    rod falls 7.7 % from 8 to 16 mm (141.16 -> 130.23 Ohm, audit p20), the
    thin wire now 3.5 % from 12 to 16.1 mm.
    """
    # audit A2S-01, A3S-01
    touching = _tip_z(-0.001, "image_2layer")  # tip 1 mm above h_1
    penetrating = _tip_z(0.012, "image_2layer")  # tip 12 mm below h_1
    deeper = _tip_z(0.0161, "image_2layer")  # tip 16.1 mm below h_1
    assert penetrating < 0.8 * touching, (touching, penetrating, deeper)
    assert deeper < penetrating < 1.05 * deeper


@_FINDING
@pytest.mark.parametrize("backend", ["image_2layer", "mom"])
def test_a_head_just_inside_a_conductive_upper_layer_is_felt(backend: str) -> None:
    """The same at the head: a rod 0.5-5.0 m below a 30 Ohm m top layer."""
    # audit A2S-01

    def z(q: float) -> float:
        w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=0.5 + q))
        gf.create_electrode(
            w, "rod", name="r", position=(0.0, 0.0, 0.5), length=4.5, wire_radius=0.008
        )
        gf.create_source(w, attached_to="r")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eng = gf.Engine(backend=backend, segment_length=0.5)
            return w.solve(eng).cluster_impedance("r")[0].real

    below_only, q8, q16, q20 = z(0.0), z(0.008), z(0.016), z(0.020)
    assert q8 < 0.85 * below_only  # the base merged the 8 mm head part
    assert q20 < q16 < q8
    assert abs(q16 / q8 - 1.0) / 8.0 < 0.01  # per mm


@_FINDING
def test_galerkin_self_term_is_exact_for_parts_much_shorter_than_the_radius() -> None:
    """(2/L)[arsinh x - sqrt(1 + 1/x²) + 1/x] lost every digit below x = 1e-8."""
    # audit A2S-01
    from groundfield.solver._line_kernel import galerkin_self

    a = 0.008
    x = np.array([1e-12, 1e-9, 1e-6, 1e-4, 1e-3])
    series = (1.0 - x**2 / 12.0 + x**4 / 40.0) / a  # small-x expansion
    np.testing.assert_allclose(
        galerkin_self(x * a, np.full(x.shape, a)), series, rtol=1e-14
    )
    # continuous where the evaluation switches forms (L/a = 0.25)
    lo, hi = galerkin_self(np.array([0.25 * (1 - 1e-12), 0.25]) * a, np.full(2, a))
    assert lo == pytest.approx(hi, rel=1e-12)


@_FINDING
def test_parallel_pair_integral_of_parts_much_shorter_than_their_distance() -> None:
    """Parts much shorter than their axis distance a (audit A2S-01).

    Two collinear touching parts of length eps (a stub and its mirror image
    in the interface): (1/a)(1 - 7 eps²/(12 a²)). One part of eps touching
    the end of a 0.5 m segment: arsinh(L/a)/L + (eps/2)(1/sqrt(a² + L²) -
    1/a)/L. The closed form returned 0 instead of 125 1/m for eps = 1e-11 m.
    Placed at a depth of 5 m as in the solver.
    """
    # audit A2S-01
    from groundfield.solver._line_kernel import LineSegments, galerkin_pairs

    a, L = 0.008, 0.5
    t = np.array([[0.0, 0.0, 1.0]])
    idx = np.array([0])
    for eps in (1e-12, 1e-11, 1e-9, 1e-6):
        stub = LineSegments(
            np.array([[0.0, 0.0, 5.0 + eps / 2]]), np.array([eps]), t, np.array([a])
        )
        mirror = LineSegments(
            np.array([[0.0, 0.0, 5.0 - eps / 2]]), np.array([eps]), -t, np.array([a])
        )
        rod = LineSegments(
            np.array([[0.0, 0.0, 5.0 - L / 2]]), np.array([L]), t, np.array([a])
        )
        both = galerkin_pairs(stub, idx, mirror, idx)[0]
        assert both == pytest.approx(
            (1.0 - 7.0 * eps**2 / (12.0 * a**2)) / a, rel=1e-12
        )
        one = galerkin_pairs(rod, idx, stub, idx)[0]
        slope = (1.0 / math.hypot(a, L) - 1.0 / a) / L
        assert one == pytest.approx(math.asinh(L / a) / L + 0.5 * eps * slope, rel=1e-9)


# ---------------------------------------------------------------------
# Audit A3S-01: the contact value against an independent solid-rod model
# ---------------------------------------------------------------------
#
# Axisymmetric finite-volume model of the solid rod (flat end faces, a = 8 mm)
# of the round-3 audit (script p20; two grids agree to 0.03 %; homogeneous
# control: thin wire +0.9 %). Tip: rod 0.5-5.0 m, h_1 = 5.0 - p, 1000/30 Ohm m.
# Head: rod (5 - q)-9.5 m, h_1 = 5 m, 30/1000 Ohm m (the head part q lies in
# the conductive top layer).
_SOLID_ROD_TIP = {0.004: 148.548, 0.008: 141.159}
_SOLID_ROD_HEAD = {0.004: 147.475, 0.008: 140.343}


def _head_z(q: float, backend: str, kernel: str | None = None) -> float:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=5.0))
    gf.create_electrode(
        w,
        "rod",
        name="r",
        position=(0.0, 0.0, 5.0 - q),
        length=4.5 + q,
        wire_radius=_WIRE_RADIUS,
    )
    gf.create_source(w, attached_to="r")
    kw: dict = {"backend": backend, "segment_length": 0.5}
    if kernel is not None:
        kw["kernel"] = kernel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return w.solve(gf.Engine(**kw)).cluster_impedance("r")[0].real


@_FINDING
@pytest.mark.parametrize("p", sorted(_SOLID_ROD_TIP))
@pytest.mark.parametrize(("backend", "kernel"), _CONFIGS)
def test_rod_tip_contact_against_the_solid_rod_reference(
    backend: str, kernel: str | None, p: float
) -> None:
    """Tip 4 / 8 mm below h_1: line -1.2 / +0.7 %, point -2.6 / -0.8 %, mom -3.6 / -1.6 %.

    Before (diagonal clamped at h_1/200 = 25 mm): -10.7 / -6.8 % (line) ...
    -12.7 / -9.1 % (``mom``).
    """
    # audit A3S-01
    value = _tip_z(p, backend, kernel)
    assert value == pytest.approx(_SOLID_ROD_TIP[p], rel=0.04), value


@_FINDING
@pytest.mark.parametrize("q", sorted(_SOLID_ROD_HEAD))
@pytest.mark.parametrize(("backend", "kernel"), _CONFIGS)
def test_rod_head_contact_against_the_solid_rod_reference(
    backend: str, kernel: str | None, q: float
) -> None:
    """Head 4 / 8 mm above h_1: line -1.1 / +0.7 %; before -10.3 / -6.5 %."""
    # audit A3S-01
    value = _head_z(q, backend, kernel)
    assert value == pytest.approx(_SOLID_ROD_HEAD[q], rel=0.04), value


@_FINDING
@pytest.mark.parametrize(("rho_1", "rho_2"), [(1000.0, 30.0), (10000.0, 10.0)])
def test_rod_tip_contact_does_not_depend_on_the_interface_clamp(
    monkeypatch: pytest.MonkeyPatch, rho_1: float, rho_2: float
) -> None:
    """Contact limit (tip 1 um below h_1): 149.2 and 180.7 Ohm (were 133.5, 123.2).

    The comparison value raises ``lambda_max_factor`` of the diagonal
    correction until its 1/R clamp lies below the wire radius (and changes
    its quadrature); the solver's value must not depend on that.
    """
    # audit A3S-01
    import groundfield.coupling.layered_green as lg

    def solve() -> float:
        w = gf.create_world(
            soil=gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=5.0 - 1e-6)
        )
        gf.create_electrode(
            w, "rod", name="r", position=(0, 0, 0.5), length=4.5, wire_radius=0.008
        )
        gf.create_source(w, attached_to="r")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eng = gf.Engine(backend="image_2layer", segment_length=0.5)
            return w.solve(eng).cluster_impedance("r")[0].real

    shipped = solve()
    original = lg.two_layer_layered_correction_real_space

    def unclamped(s, z, z_s, *, lambda_max_factor=200.0, **kw):  # noqa: ANN001
        char = max(min(kw["h_1"], s + z + z_s + 1e-9), 1e-3)
        lmf = max(lambda_max_factor, 4.0 * char / max(s, 1e-12))
        return original(s, z, z_s, lambda_max_factor=lmf, **kw)

    monkeypatch.setattr(lg, "two_layer_layered_correction_real_space", unclamped)
    assert shipped == pytest.approx(solve(), rel=1e-5)


# ---------------------------------------------------------------------
# Audit A3S-05: the step at zero penetration is a thin-wire limit
# ---------------------------------------------------------------------
#
# The thin wire has no end face. Against the solid rod (flat end, same
# model as above; tip p below h_1, line kernel): +3.3 % at p = -16 mm,
# +3.9 % at -8 mm, +7.2 % at -1 mm, +21.4 % at p = 0, -4.6 % at +1 mm,
# +0.7 % at +8 mm (point kernel and mom alike, +19.0 ... +20.5 % at p = 0).
# Rod ends within one wire radius of an interface warn.


def _tip_solve_warnings(p: float, backend: str, kernel: str | None) -> list:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0 - p))
    gf.create_electrode(
        w, "rod", name="r", position=(0.0, 0.0, 0.5), length=4.5, wire_radius=0.008
    )
    gf.create_source(w, attached_to="r")
    kw: dict = {"backend": backend, "segment_length": 0.5}
    if kernel is not None:
        kw["kernel"] = kernel
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = w.solve(gf.Engine(**kw))
    assert res.cluster_impedance("r")[0].real > 0.0
    return [r for r in rec if issubclass(r.category, gf.solver.InterfaceContactWarning)]


@_FINDING
@pytest.mark.parametrize(
    ("p", "warned"),
    [
        (0.0, True),
        (-1e-3, True),
        (1e-3, True),
        (-0.008, True),
        (0.008, True),
        (-0.0082, False),
        (0.0082, False),
        (-0.016, False),
        (0.016, False),
    ],
)
@pytest.mark.parametrize(("backend", "kernel"), _CONFIGS)
def test_rod_tip_within_one_wire_radius_of_the_interface_warns(
    backend: str, kernel: str | None, p: float, warned: bool
) -> None:
    """+21 % at the interface against the solid rod: warned, not silently returned."""
    # audit A3S-05
    hits = _tip_solve_warnings(p, backend, kernel)
    assert bool(hits) is warned
    if warned:
        text = str(hits[0].message)
        assert "electrode 'r', tip" in text and "end face" in text


@_FINDING
@pytest.mark.parametrize("bonded", [True, False])
def test_a_rod_split_at_the_interface_has_no_free_end_there(bonded: bool) -> None:
    """Two rods meeting at h_1 form one rod: no end face, no warning.

    Splitting a rod at the interface into two bonded pieces is the way to
    give each layer its own piece (tutorial T02). The warning concerns free
    rod ends only; a third, unrelated rod ending at h_1 still warns.
    """
    # audit A3S-05 (coordinator: no false alarm for joined ends)
    def solve(extra: bool) -> list:
        w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=3.0))
        gf.create_electrode(
            w, "rod", name="up", position=(0.0, 0.0, 0.5), length=2.5, wire_radius=0.008
        )
        gf.create_electrode(
            w, "rod", name="low", position=(0.0, 0.0, 3.0), length=3.5, wire_radius=0.008
        )
        if bonded:
            gf.create_conductor(w, name="bond", start="up", end="low")
        if extra:
            gf.create_electrode(
                w, "rod", name="free", position=(5.0, 0.0, 0.5), length=2.5,
                wire_radius=0.008,
            )
            gf.create_conductor(w, name="b2", start="up", end="free")
        gf.create_source(w, attached_to="up")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            w.solve(gf.Engine(segment_length=0.5))
        return [r for r in rec if issubclass(r.category, gf.solver.InterfaceContactWarning)]

    assert solve(extra=False) == []
    hits = solve(extra=True)
    assert len(hits) == 1 and "'free'" in str(hits[0].message)


@_FINDING
@pytest.mark.parametrize(("q", "warned"), [(0.0, True), (0.008, True), (0.02, False)])
def test_rod_head_within_one_wire_radius_of_the_interface_warns(
    q: float, warned: bool
) -> None:
    # audit A3S-05
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=5.0))
    gf.create_electrode(
        w, "rod", name="r", position=(0.0, 0.0, 5.0 - q), length=4.5, wire_radius=0.008
    )
    gf.create_source(w, attached_to="r")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w.solve(gf.Engine(segment_length=0.5))
    hits = [r for r in rec if issubclass(r.category, gf.solver.InterfaceContactWarning)]
    assert bool(hits) is warned
    if warned:
        assert "head" in str(hits[0].message)


@_FINDING
def test_no_physical_interface_no_contact_warning() -> None:
    """TwoLayerSoil(rho, rho, h) has no interface to touch."""
    # audit A3S-05
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=5.0))
    gf.create_electrode(
        w, "rod", name="r", position=(0.0, 0.0, 0.5), length=4.5, wire_radius=0.008
    )
    gf.create_source(w, attached_to="r")
    with warnings.catch_warnings():
        warnings.simplefilter("error", gf.solver.InterfaceContactWarning)
        w.solve(gf.Engine(segment_length=0.5))
