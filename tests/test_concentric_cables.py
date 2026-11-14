r"""Concentric cables by an explicit marker (user decisions Q32 and Q33).

Since 2026.11 (review finding 10.46, audit A3C-5) a thinner conductor drawn
on the axis of a thicker one couples to it at the radius of the thicker one
(a core inside a cable shield: a filament inside a thin tube). Up to the
user decision Q32 that reading applied to every such pair, so two separate
cables of different radii drawn on one axis were read as one concentric
cable. The reading now needs the explicit marker
``conductor_type="cable_shield"`` on the thicker conductor; two unmarked
conductors on one axis are separate conductors that touch, as any pair
lying side by side. A shield is a thin-walled tube, so its self inductance
carries no internal inductance of a solid wire (Q33): the 100 m coaxial pair
with a 5 mm core and a 15 mm shield reads the analytic coaxial loop
inductance

.. math::

    L = \frac{\mu_0 \ell}{2\pi}\Bigl(\ln\frac{r_s}{r_c} + \frac14\Bigr)
      = 26.97\ \mu\mathrm{H}

instead of 31.97 µH (base e6124cd: the internal inductance
:math:`\mu_0\ell/(8\pi)` = 5 µH of a solid shield on top).
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.conductors import Conductor
from groundfield.coupling.inductance import (
    _build_inductance_matrix_loop,
    build_inductance_matrix,
    parallel_segments_mutual,
)

# audit A3C-5 (the inside reading), user decisions Q32 and Q33
pytestmark = pytest.mark.finding("10.46")

LENGTH, R_CORE, R_SHIELD, DEPTH = 100.0, 0.005, 0.015, 0.7
MU0 = 4e-7 * math.pi
#: Analytic loop inductance of the coaxial pair (core out, shield back).
COAX = MU0 / (2.0 * math.pi) * LENGTH * (math.log(R_SHIELD / R_CORE) + 0.25)
#: Two touching wires (centre distance r_c + r_s), both solid (GMR a e^-1/4).
TOUCHING = (
    MU0
    / (2.0 * math.pi)
    * LENGTH
    * (math.log((R_CORE + R_SHIELD) ** 2 / (R_CORE * R_SHIELD)) + 0.5)
)


def _loop_world(shield_type: str, ds: float | None = None) -> gf.World:
    """Core out from junction A, shield back to junction S, both bonded at
    the far end B (tied to a rod for the soil contact); 1 A from A to S."""
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(
        w, "rod", name="rod", position=(LENGTH + 1.0, 0.0, 0.5), length=1.0
    )
    gf.create_junction(w, name="A", position=(0.0, 0.0, DEPTH))
    gf.create_junction(w, name="S", position=(0.0, 0.0, DEPTH))
    gf.create_junction(w, name="B", position=(LENGTH, 0.0, DEPTH))
    common = dict(
        start=(0.0, 0.0, DEPTH),
        end=(LENGTH, 0.0, DEPTH),
        end_electrode="B",
        resistivity=1.72e-8,
        coupling_to_soil="isolated",
        inductance_model="neumann",
        discretize_segment_length=ds,
    )
    w.add_conductor(
        Conductor(
            name="core",
            start_electrode="A",
            wire_radius=R_CORE,
            cross_section=50e-6,
            **common,
        )
    )
    w.add_conductor(
        Conductor(
            name="shield",
            start_electrode="S",
            wire_radius=R_SHIELD,
            cross_section=25e-6,
            conductor_type=shield_type,
            **common,
        )
    )
    w.add_conductor(
        Conductor(
            name="tie",
            start=(LENGTH, 0.0, DEPTH),
            end=(LENGTH + 1.0, 0.0, 0.5),
            start_electrode="B",
            end_electrode="rod",
        )
    )
    gf.create_source(w, attached_to="A", return_to="S", magnitude=1.0)
    return w


def _loop_inductance(world: gf.World, model: str) -> float:
    engine = gf.create_engine(
        segment_length=1.0, frequencies=[50.0], earth_inductive_model=model
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = engine.solve(world)
    potentials = res.metadata["junction_potentials"]
    z = complex(potentials["A"][0] - potentials["S"][0])
    return z.imag / (2.0 * math.pi * 50.0)


# user decisions Q32, Q33
@pytest.mark.parametrize("model", ["sommerfeld", "perfect_mirror", "carson_series"])
@pytest.mark.parametrize("ds", [None, 10.0], ids=["lumped", "distributed"])
def test_marked_shield_reads_the_analytic_coaxial_loop(model: str, ds) -> None:
    """26.97 µH analytic; the solver -2.2e-4 (the finite length and the
    end terms of the thin-wire self terms; carson_series -7e-5), lumped and
    in 10 m sub-pieces alike (audit W1A-02: the pieces read -0.2 % on the
    snapshot 6f63deb, which the former bound of 3e-3 hid). Base e6124cd:
    31.97 µH (+18.5 %)."""
    loop = _loop_inductance(_loop_world("cable_shield", ds), model)
    assert loop == pytest.approx(COAX, rel=3e-4)
    assert loop < COAX + 0.5 * MU0 * LENGTH / (8.0 * math.pi)


# user decision Q32
@pytest.mark.parametrize("shield_type", ["generic", "pen", "bare_copper"])
def test_unmarked_cables_on_one_axis_are_separate_touching_conductors(
    shield_type: str,
) -> None:
    """Without the marker the two cables touch (centre distance r_c + r_s,
    both solid): 43.48 µH analytic, 43.47 µH solved. Base e6124cd read them
    as a core inside a shield, 31.97 µH."""
    loop = _loop_inductance(_loop_world(shield_type), "sommerfeld")
    assert loop == pytest.approx(TOUCHING, rel=1e-3)


# user decision Q32
def test_the_thicker_wire_must_be_the_shield() -> None:
    """A thin wire marked as a shield on the axis of a thicker unmarked
    one is no core inside a shield: the two touch."""
    seg = np.array(
        [[[0, 0, DEPTH], [LENGTH, 0, DEPTH]], [[0, 0, DEPTH], [LENGTH, 0, DEPTH]]],
        dtype=float,
    )
    radii = np.array([R_CORE, R_SHIELD])
    touching = parallel_segments_mutual(LENGTH, R_CORE + R_SHIELD)
    inside = parallel_segments_mutual(LENGTH, R_SHIELD)
    marked = build_inductance_matrix(
        seg, radii, image="none", shields=np.array([False, True])
    )
    wrong_one = build_inductance_matrix(
        seg, radii, image="none", shields=np.array([True, False])
    )
    unmarked = build_inductance_matrix(seg, radii, image="none")
    assert marked[0, 1] == pytest.approx(inside, rel=1e-12)
    assert wrong_one[0, 1] == pytest.approx(touching, rel=1e-12)
    assert unmarked[0, 1] == pytest.approx(touching, rel=1e-12)
    # the marker does not touch the self terms
    assert np.array_equal(np.diag(marked), np.diag(unmarked))


# user decision Q32
def test_a_core_off_the_axis_but_inside_the_shield_couples_at_its_radius() -> None:
    """rho + r_c <= r_s: inside (shield radius); farther out: touching."""
    radii = np.array([R_CORE, R_SHIELD])
    for offset, expected in ((0.009, R_SHIELD), (0.0105, R_CORE + R_SHIELD)):
        seg = np.array(
            [
                [[0, offset, DEPTH], [LENGTH, offset, DEPTH]],
                [[0, 0, DEPTH], [LENGTH, 0, DEPTH]],
            ],
            dtype=float,
        )
        L = build_inductance_matrix(
            seg, radii, image="none", shields=np.array([False, True])
        )
        assert L[0, 1] == pytest.approx(
            parallel_segments_mutual(LENGTH, expected), rel=1e-9
        )


# user decision Q32
@pytest.mark.parametrize("image", ["none", "additive", "antiparallel"])
def test_vectorised_assembly_matches_the_loop_oracle_with_shields(image: str) -> None:
    """A three-core arrangement in one shield and a cable beside it."""
    seg = np.array(
        [
            [[0, 0.0, DEPTH], [30, 0.0, DEPTH]],
            [[0, 0.004, DEPTH], [30, 0.004, DEPTH]],
            [[0, -0.004, DEPTH], [30, -0.004, DEPTH]],
            [[0, 0.0, DEPTH], [30, 0.0, DEPTH]],
            [[0, 0.03, DEPTH], [30, 0.03, DEPTH]],
        ],
        dtype=float,
    )
    radii = np.array([0.003, 0.003, 0.003, 0.012, 0.012])
    shields = np.array([False, False, False, True, False])
    internal = ~shields
    fast = build_inductance_matrix(
        seg, radii, image=image, shields=shields, include_internal=internal
    )
    slow = _build_inductance_matrix_loop(
        seg, radii, image=image, shields=shields, include_internal=internal
    )
    assert np.allclose(fast, slow, rtol=1e-13, atol=0.0)
    assert np.linalg.eigvalsh(fast).min() > 0.0


# user decision Q33
def test_a_shield_has_no_internal_inductance_of_a_solid_wire() -> None:
    shield = Conductor(
        name="s",
        start=(0, 0, DEPTH),
        end=(10, 0, DEPTH),
        conductor_type="cable_shield",
        cross_section=25e-6,
    )
    core = shield.model_copy(update={"conductor_type": "generic"})
    assert shield.is_shield and not core.is_shield
    z_shield = shield.internal_impedance_per_length(50.0)
    z_core = core.internal_impedance_per_length(50.0)
    assert z_shield.imag == 0.0
    assert z_core.imag == pytest.approx(2 * math.pi * 50.0 * MU0 / (8 * math.pi))
    assert z_shield.real == z_core.real
    with pytest.raises(ValueError, match="thin-walled"):
        Conductor(
            name="s2",
            start=(0, 0, DEPTH),
            end=(10, 0, DEPTH),
            conductor_type="cable_shield",
            cross_section=25e-6,
            resistance_model="skin",
        )


# user decision Q33
def test_a_lone_shield_loses_exactly_the_internal_inductance() -> None:
    """One inductive shield alone (no core): its self inductance drops by
    mu0 l/(8 pi) = 50 nH/m against the same conductor unmarked."""
    from groundfield.coupling.inductance import thin_wire_self_inductance

    def lone(kind: str) -> float:
        w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
        gf.create_junction(w, name="A", position=(0.0, 0.0, DEPTH))
        gf.create_electrode(w, "rod", name="r", position=(LENGTH, 0.0, 0.5), length=2)
        w.add_conductor(
            Conductor(
                name="c",
                start=(0, 0, DEPTH),
                end=(LENGTH, 0, DEPTH),
                start_electrode="A",
                end_electrode="r",
                conductor_type=kind,
                wire_radius=R_SHIELD,
                cross_section=25e-6,
                inductance_model="neumann",
            )
        )
        gf.create_source(w, attached_to="A", magnitude=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gf.create_engine(
                segment_length=1.0,
                frequencies=[50.0],
                earth_inductive_model="perfect_mirror",
            ).solve(w)
        return complex(res.input_impedance()[0]).imag / (2 * math.pi * 50.0)

    drop = lone("generic") - lone("cable_shield")
    expected = thin_wire_self_inductance(
        LENGTH, R_SHIELD, include_internal=True
    ) - thin_wire_self_inductance(LENGTH, R_SHIELD, include_internal=False)
    assert expected == pytest.approx(MU0 * LENGTH / (8 * math.pi), rel=1e-12)
    assert drop == pytest.approx(expected, rel=1e-9)


# user decision Q32
def test_the_marker_survives_a_json_round_trip() -> None:
    world = _loop_world("cable_shield")
    data = json.loads(world.model_dump_json())
    types = {c["name"]: c["conductor_type"] for c in data["conductors"]}
    assert types == {"core": "generic", "shield": "cable_shield", "tie": "generic"}
    back = gf.World.model_validate_json(world.model_dump_json())
    assert _loop_inductance(back, "sommerfeld") == pytest.approx(
        _loop_inductance(world, "sommerfeld"), rel=1e-12
    )


# ---------------------------------------------------------------------
# Pieces of conductors lying on one axis (audit W1A-02)
# ---------------------------------------------------------------------


def _axis_pieces(n: int, x0: float = 0.0, x1: float = LENGTH) -> np.ndarray:
    xs = np.linspace(x0, x1, n + 1)
    return np.array(
        [[[xs[k], 0.0, DEPTH], [xs[k + 1], 0.0, DEPTH]] for k in range(n)], dtype=float
    )


def _matrix_loop(
    n_core: int, n_shield: int, marked: bool, image: str = "none"
) -> float:
    """Loop inductance (core out, shield back) from the matrix of the pieces,
    with one conductor label per conductor."""
    seg = np.concatenate([_axis_pieces(n_core), _axis_pieces(n_shield)])
    radii = np.array([R_CORE] * n_core + [R_SHIELD] * n_shield)
    shields = np.array([False] * n_core + [marked] * n_shield)
    L = build_inductance_matrix(
        seg,
        radii,
        image=image,
        shields=shields,
        include_internal=~shields,
        conductors=np.array([0] * n_core + [1] * n_shield),
    )
    s = np.array([1.0] * n_core + [-1.0] * n_shield)
    return float(s @ L @ s)


# audit W1A-02, user decisions Q32 / Q33
@pytest.mark.parametrize("shield_type", ["cable_shield", "generic"])
@pytest.mark.parametrize("ds", [10.0, 5.0, 1.0, 0.25])
def test_the_coax_loop_does_not_depend_on_the_conductor_pieces(
    shield_type: str, ds: float
) -> None:
    """The core and shield pieces of a coax (and of a touching unmarked
    pair) add up to the lumped loop: every pair of pieces of two conductors
    lying on one axis couples at the on-axis distance (the shield radius or
    r_c + r_s), also end to end and apart. Snapshot 6f63deb: 26.912 /
    26.852 / 26.374 / 24.608 uH marked (-0.2 / -0.4 / -2.2 / -8.7 %),
    43.400 ... 40.343 uH unmarked, against 26.966 / 43.472 uH lumped."""
    lumped = _loop_inductance(_loop_world(shield_type, None), "perfect_mirror")
    loop = _loop_inductance(_loop_world(shield_type, ds), "perfect_mirror")
    assert loop == pytest.approx(lumped, rel=1e-4)
    assert lumped == pytest.approx(
        COAX if shield_type == "cable_shield" else TOUCHING, rel=3e-4
    )


# audit W1A-02, user decisions Q32 / Q33
@pytest.mark.parametrize("model", ["sommerfeld", "carson_series"])
def test_the_coax_pieces_add_up_under_every_earth_model(model: str) -> None:
    lumped = _loop_inductance(_loop_world("cable_shield", None), model)
    loop = _loop_inductance(_loop_world("cable_shield", 1.0), model)
    assert loop == pytest.approx(lumped, rel=1e-4)


# audit W1A-02, user decision Q32
@pytest.mark.parametrize("marked", [True, False])
@pytest.mark.parametrize("image", ["none", "additive", "antiparallel"])
def test_matrix_pieces_on_one_axis_add_up(marked: bool, image: str) -> None:
    """Core and shield in different pieces (100 / 37 / 7 / 1 m and 1 m /
    0.5 m), the loop of the matrix equals the lumped one to round-off; the
    lumped mutual is the closed form at the on-axis distance."""
    lumped = _matrix_loop(1, 1, marked, image)
    for n_core, n_shield in ((10, 10), (3, 14), (100, 200)):
        assert _matrix_loop(n_core, n_shield, marked, image) == pytest.approx(
            lumped, rel=1e-10
        )
    d = R_SHIELD if marked else R_CORE + R_SHIELD
    L = build_inductance_matrix(
        np.concatenate([_axis_pieces(1), _axis_pieces(1)]),
        np.array([R_CORE, R_SHIELD]),
        image="none",
        shields=np.array([False, marked]),
    )
    assert L[0, 1] == pytest.approx(parallel_segments_mutual(LENGTH, d), rel=1e-13)


# audit W1A-02, user decision Q32
def test_a_shield_made_of_two_conductors_equals_one_shield() -> None:
    """A shield of two conductors meeting end to end at 40 m around one
    core: the two shield halves continue one tube (filament on the axis,
    no lateral distance), each couples to the core at the shield radius."""
    core = _axis_pieces(10)
    shield = np.concatenate([_axis_pieces(4, 0.0, 40.0), _axis_pieces(6, 40.0, LENGTH)])
    seg = np.concatenate([core, shield])
    radii = np.array([R_CORE] * 10 + [R_SHIELD] * 10)
    shields = np.array([False] * 10 + [True] * 10)
    labels = np.array(["core"] * 10 + ["shield_a"] * 4 + ["shield_b"] * 6)
    L = build_inductance_matrix(
        seg,
        radii,
        image="none",
        shields=shields,
        include_internal=~shields,
        conductors=labels,
    )
    s = np.array([1.0] * 10 + [-1.0] * 10)
    assert float(s @ L @ s) == pytest.approx(_matrix_loop(1, 1, True), rel=1e-10)


# audit W1A-02 (guard: worlds without conductors beside each other)
def test_conductors_meeting_end_to_end_on_one_axis_keep_the_filament() -> None:
    """A PEN main split into conductors at its taps lies on one axis but no
    two of its conductors lie beside each other: the labels change nothing,
    and a straight wire in pieces keeps the self inductance of the whole."""
    seg = np.concatenate([_axis_pieces(5, 0.0, 50.0), _axis_pieces(7, 50.0, LENGTH)])
    radii = np.full(12, 0.008)
    plain = build_inductance_matrix(seg, radii, image="additive")
    labelled = build_inductance_matrix(
        seg, radii, image="additive", conductors=np.array([0] * 5 + [1] * 7)
    )
    assert np.array_equal(plain, labelled)
    whole = build_inductance_matrix(_axis_pieces(1), radii[:1], image="none")
    pieces = build_inductance_matrix(seg, radii, image="none", conductors=np.zeros(12))
    assert pieces.sum() == pytest.approx(whole.sum(), rel=1e-13)


# audit W1A-02, user decision Q32
def test_the_on_axis_rule_is_free_of_order_and_direction() -> None:
    """Permuting and reversing the pieces permutes the core-shield block
    bit for bit (up to the orientation sign), and the loop oracle agrees
    with the vectorised assembly."""
    seg = np.concatenate([_axis_pieces(4), _axis_pieces(3, 10.0, LENGTH - 20.0)])
    radii = np.array([R_CORE] * 4 + [R_SHIELD] * 3)
    shields = np.array([False] * 4 + [True] * 3)
    labels = np.array([0] * 4 + [1] * 3)
    for image in ("none", "additive", "antiparallel"):
        kw = dict(image=image, shields=shields, include_internal=~shields)
        L = build_inductance_matrix(seg, radii, conductors=labels, **kw)
        oracle = _build_inductance_matrix_loop(seg, radii, conductors=labels, **kw)
        assert np.array_equal(L, oracle)
    L = build_inductance_matrix(
        seg, radii, conductors=labels, image="none", shields=shields
    )
    perm = np.array([5, 2, 0, 6, 3, 1, 4])
    flip = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
    seg_p = seg[perm].copy()
    seg_p[flip < 0] = seg_p[flip < 0][:, ::-1, :]
    L_p = build_inductance_matrix(
        seg_p,
        radii[perm],
        conductors=labels[perm],
        image="none",
        shields=shields[perm],
    )
    expected = L[np.ix_(perm, perm)] * np.outer(flip, flip)
    cross = labels[perm][:, None] != labels[perm][None, :]
    assert np.array_equal(L_p[cross], expected[cross])
