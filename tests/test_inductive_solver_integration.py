"""Solver-level behaviour of the inductive stack (review pass 10, wave 2).

The coupling-side APIs of stream COUP-1 (``image=`` conventions,
``touching="exact"``, multi-frequency builders,
``Conductor.resistance_model="skin"``) as the solver uses them (stream
COUP-2): Carson's correction on its anti-parallel image baseline (10.20),
Neumann inductance of lumped and single-segment conductors (10.24), the
skin effect in the branch impedance (10.37), exact touching pairs and the
image-free Sommerfeld baseline (10.46), one multi-frequency assembly per
solve (10.60). The first eight tests were strict ``xfail`` markers of
wave 1.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.conductors import Conductor

RHO_SOIL = 100.0
RHO_AL = 2.82e-8
A_150 = 150e-6
A_RADIUS = math.sqrt(A_150 / math.pi)


def _pen_world(
    length: float,
    *,
    dsl: float | None,
    inductance_model: str | None = "neumann",
    resistance_model: str = "dc",
    end_z: float = 0.7,
) -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=RHO_SOIL))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="g2", position=(length, 0.0, 0.5), length=3.0)
    w.add_conductor(
        Conductor(
            name="pen",
            start=(0.0, 0.0, 0.7),
            end=(length, 0.0, end_z),
            start_electrode="g1",
            end_electrode="g2",
            wire_radius=A_RADIUS,
            resistivity=RHO_AL,
            cross_section=A_150,
            discretize_segment_length=dsl,
            coupling_to_soil="isolated",
            inductance_model=inductance_model,
            resistance_model=resistance_model,
        )
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _series_impedance(
    w: gf.World, model: str, freqs: list[float], **engine_kw: object
) -> np.ndarray:
    eng = gf.create_engine(
        backend="image",
        segment_length=0.5,
        frequencies=freqs,
        earth_inductive_model=model,
        **engine_kw,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = w.solve(eng)
    v1 = np.asarray(res.electrode_potentials["g1"])
    v2 = np.asarray(res.electrode_potentials["g2"])
    i2 = np.asarray(res.electrode_currents["g2"])
    return (v1 - v2) / i2


@pytest.mark.finding("10.20", "F32")
@pytest.mark.parametrize("freq", [50.0, 1000.0])
def test_carson_series_reactance_matches_carson(freq: float) -> None:
    """1 km isolated PEN: X = l * X'_Carson(GMR = a e^{-1/4}) within 1 %.

    v0.15.0: 2.04x (50 Hz) and 2.19x (1 kHz) — Carson's correction on
    top of the additive image (the solver now assembles it on the
    anti-parallel image, Carson's own baseline).
    """
    from groundfield.references.earth_return import carson_self_impedance

    length = 1000.0
    z = _series_impedance(_pen_world(length, dsl=10.0 + 1e-9), "carson_series", [freq])[
        0
    ]
    ref = length * carson_self_impedance(RHO_SOIL, freq, gmr=A_RADIUS * math.exp(-0.25))
    assert z.imag == pytest.approx(ref.imag, rel=1e-2)


@pytest.mark.finding("10.24")
@pytest.mark.parametrize("dsl", [None, 500.0], ids=["lumped", "single_segment"])
def test_neumann_is_honoured_on_lumped_and_single_segment_conductors(dsl) -> None:
    """300 m PEN at 1 kHz: v0.15.0 gives 0.0564 + 0j (inductance dropped)."""
    z_dist = _series_impedance(_pen_world(300.0, dsl=10.0), "sommerfeld", [1000.0])[0]
    z = _series_impedance(_pen_world(300.0, dsl=dsl), "sommerfeld", [1000.0])[0]
    assert z.imag == pytest.approx(z_dist.imag, rel=2e-2)


@pytest.mark.finding("10.37")
def test_skin_model_raises_the_pen_resistance_at_1khz() -> None:
    """150 mm² Al: R_ac/R_dc = 1.5496 at 1 kHz (solid-round Bessel)."""
    kw = dict(dsl=10.0)
    z_dc = _series_impedance(_pen_world(300.0, **kw), "sommerfeld", [1000.0])[0]
    z_skin = _series_impedance(
        _pen_world(300.0, resistance_model="skin", **kw), "sommerfeld", [1000.0]
    )[0]
    r_dc = RHO_AL * 300.0 / A_150
    assert (z_skin - z_dc).real == pytest.approx(0.5496 * r_dc, rel=1e-2)


@pytest.mark.finding("10.60")
def test_sommerfeld_pair_work_does_not_grow_with_the_number_of_frequencies(
    monkeypatch,
) -> None:
    """The solver builds the Sommerfeld matrices for all frequencies at once.

    The per-pair rule (``sommerfeld_assembly="exact"``; the default is the
    tabulated assembly since the user decision Q16).
    """
    from groundfield.coupling import sommerfeld_inductance as si

    calls = {"n": 0}
    original = si._pair_values_none

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(si, "_pair_values_none", counting)
    world, exact = _pen_world(60.0, dsl=10.0), {"sommerfeld_assembly": "exact"}
    _series_impedance(world, "sommerfeld", [50.0], **exact)
    one = calls["n"]
    assert one > 0
    calls["n"] = 0
    _series_impedance(world, "sommerfeld", [50.0, 250.0, 1000.0], **exact)
    assert calls["n"] == one


@pytest.mark.finding("10.46")
def test_inductance_is_refinement_invariant_for_a_straight_conductor() -> None:
    """Exact touching pairs: the chain sum does not drift with refinement
    (16x16 quadrature: about 8e-5 between 4 and 40 segments)."""
    z4 = _series_impedance(_pen_world(300.0, dsl=75.0), "perfect_mirror", [1000.0])[0]
    z40 = _series_impedance(_pen_world(300.0, dsl=7.5), "perfect_mirror", [1000.0])[0]
    assert z40.imag == pytest.approx(z4.imag, rel=1e-7)


@pytest.mark.finding("10.46", "F52")
def test_sommerfeld_accepts_a_surface_crossing_conductor() -> None:
    """With the exact (image-free) baseline a lead may leave the soil.

    With the additive image the crossing segment is singular and the
    solve raised; with ``image="none"`` the coupling side splits the
    segment at z = 0.
    """
    w = _pen_world(60.0, dsl=10.0, end_z=-1.0)
    z = _series_impedance(w, "sommerfeld", [1000.0])[0]
    assert math.isfinite(z.real) and z.imag > 0.0


@pytest.mark.finding("10.60")
@pytest.mark.parametrize("model", ["carson_series", "sommerfeld"])
def test_multifrequency_solve_equals_single_frequency_solves(model: str) -> None:
    """One assembly for all frequencies; every frequency equals its own solve.

    Bit for bit with OpenBLAS on Linux x86-64. With Accelerate on macOS
    arm64 the electrode potentials differ by up to one unit in the last
    place (2.0e-16), which the difference ``v1 - v2`` of the series
    impedance amplifies by ``|v1| / |v1 - v2|`` (3088 at DC): 1.26e-12.
    The test therefore compares the solved potentials and currents, not
    the cancelling difference.
    """

    def solved(freqs: list[float]) -> np.ndarray:
        eng = gf.create_engine(
            backend="image",
            segment_length=0.5,
            frequencies=freqs,
            earth_inductive_model=model,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _pen_world(60.0, dsl=10.0).solve(eng)
        return np.array(
            [
                res.electrode_potentials["g1"],
                res.electrode_potentials["g2"],
                res.electrode_currents["g2"],
            ]
        )

    freqs = [0.0, 50.0, 250.0, 1000.0]
    all_f = solved(freqs)
    for k, f in enumerate(freqs):
        one_f = solved([f])[:, 0]
        np.testing.assert_allclose(all_f[:, k], one_f, rtol=1e-15, atol=0)


@pytest.mark.finding("10.60")
def test_branch_correction_is_built_once_for_all_frequencies() -> None:
    """The correction object assembles the engine's frequency set at once."""
    from groundfield.solver import _pipeline

    w = _pen_world(60.0, dsl=10.0)
    eng = gf.create_engine(
        backend="image",
        segment_length=0.5,
        frequencies=[50.0, 250.0, 1000.0],
        earth_inductive_model="sommerfeld",
    )
    disc = _pipeline.discretise(w, eng)
    topo = _pipeline.build_topology(w, disc)
    branches = _pipeline.assemble_branches(w, eng, topo)
    correction = branches.earth_correction
    assert correction.n_assemblies == 0
    first = correction(2.0 * np.pi * 250.0)
    assert correction.n_assemblies == 3  # the whole frequency set at once
    assert correction(2.0 * np.pi * 250.0) is first
    assert not first.flags.writeable
    # a frequency outside the engine's set is built on demand
    correction(2.0 * np.pi * 60.0)
    assert correction.n_assemblies == 4
    assert not np.any(correction(0.0))


@pytest.mark.finding("10.24")
@pytest.mark.parametrize("model", ["perfect_mirror", "carson_series"])
@pytest.mark.parametrize("dsl", [None, 500.0], ids=["lumped", "single_segment"])
def test_lumped_branch_equals_the_distributed_chain(model: str, dsl) -> None:
    """With exact touching pairs (perfect_mirror) and the per-metre Carson
    tiling the single branch is the sum of the refined chain."""
    z_dist = _series_impedance(_pen_world(300.0, dsl=10.0), model, [50.0, 1000.0])
    z = _series_impedance(_pen_world(300.0, dsl=dsl), model, [50.0, 1000.0])
    np.testing.assert_allclose(z, z_dist, rtol=1e-9)


@pytest.mark.finding("10.24")
def test_lumped_inductive_conductor_couples_to_a_parallel_chain() -> None:
    """The lumped branch takes part in the mutual inductances too: a lumped
    PEN parallel to a distributed one splits the current like two chains."""

    def world(dsl_b):
        w = gf.create_world(soil=gf.HomogeneousSoil(rho=RHO_SOIL))
        gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=3.0)
        gf.create_electrode(w, "rod", name="g2", position=(200.0, 0.0, 0.5), length=3.0)
        for name, y, dsl in (("a", 0.0, 20.0), ("b", 0.3, dsl_b)):
            w.add_conductor(
                Conductor(
                    name=name,
                    start=(0.0, y, 0.7),
                    end=(200.0, y, 0.7),
                    start_electrode="g1",
                    end_electrode="g2",
                    wire_radius=A_RADIUS,
                    resistivity=RHO_AL,
                    cross_section=A_150,
                    discretize_segment_length=dsl,
                    inductance_model="neumann",
                )
            )
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    z_chain = _series_impedance(world(20.0), "perfect_mirror", [1000.0])[0]
    z_lumped = _series_impedance(world(None), "perfect_mirror", [1000.0])[0]
    assert z_lumped == pytest.approx(z_chain, rel=1e-9)


def _skin_ratio(f: float) -> complex:
    """Z_int(f) / R_dc of the 150 mm2 Al conductor (independent Bessel form)."""
    from scipy.special import iv

    a = math.sqrt(A_150 / math.pi)
    m = np.sqrt(1j * 2.0 * math.pi * f * 4e-7 * math.pi / RHO_AL)
    z_int = m * RHO_AL / (2.0 * math.pi * a) * iv(0, m * a) / iv(1, m * a)
    return z_int / (RHO_AL / A_150)


@pytest.mark.finding("10.37")
@pytest.mark.parametrize("dsl", [None, 10.0], ids=["lumped", "chain"])
def test_skin_correction_of_an_inductive_conductor(dsl) -> None:
    """Z_skin - Z_dc = R_dc (Z_int/R_dc - 1) - j w mu0 l / (8 pi) at 1 kHz:
    the Bessel internal impedance replaces R_dc and the DC internal
    inductance of the inductance matrix."""
    z_dc = _series_impedance(_pen_world(300.0, dsl=dsl), "perfect_mirror", [1000.0])
    z_skin = _series_impedance(
        _pen_world(300.0, dsl=dsl, resistance_model="skin"), "perfect_mirror", [1000.0]
    )
    r_dc = RHO_AL * 300.0 / A_150
    l_int_dc = 4e-7 * math.pi * 300.0 / (8.0 * math.pi)
    expected = r_dc * (_skin_ratio(1000.0) - 1.0) - 1j * 2e3 * math.pi * l_int_dc
    assert z_skin[0] - z_dc[0] == pytest.approx(expected, rel=1e-9)


@pytest.mark.finding("10.37")
def test_skin_correction_covers_the_half_stubs_of_a_galvanic_chain() -> None:
    """Galvanic chain (R/2 stubs at both ends): the diagonal corrections of
    all branches add up to l * dZ'_skin(f)."""
    from groundfield.solver import _pipeline

    w = _pen_world(300.0, dsl=10.0, resistance_model="skin")
    w.conductors[0] = w.conductors[0].model_copy(
        update={"coupling_to_soil": "galvanic"}
    )
    eng = gf.create_engine(
        backend="image",
        segment_length=0.5,
        frequencies=[1000.0],
        earth_inductive_model="perfect_mirror",
    )
    disc = _pipeline.discretise(w, eng)
    topo = _pipeline.build_topology(w, disc)
    branches = _pipeline.assemble_branches(w, eng, topo)
    dz = branches.earth_correction(2.0 * math.pi * 1000.0)
    assert len(topo.finite_branches) == 31  # 30 segments, two half stubs
    assert np.count_nonzero(dz - np.diag(np.diagonal(dz))) == 0
    expected = 300.0 * w.conductors[0].skin_correction_per_length(1000.0)
    assert np.trace(dz) == pytest.approx(expected, rel=1e-12)


@pytest.mark.finding("10.37")
def test_skin_effect_without_inductance_model_changes_the_resistance_only() -> None:
    """No inductance model: R_ac(f) only, no reactance; f = 0 stays R_dc."""
    z_dc = _series_impedance(
        _pen_world(300.0, dsl=None, inductance_model=None),
        "perfect_mirror",
        [0.0, 1000.0],
    )
    z_skin = _series_impedance(
        _pen_world(300.0, dsl=None, inductance_model=None, resistance_model="skin"),
        "perfect_mirror",
        [0.0, 1000.0],
    )
    r_dc = RHO_AL * 300.0 / A_150
    assert z_dc[0] == pytest.approx(r_dc, rel=1e-12)
    # DC: no skin effect. The two worlds run different solve paths, and
    # (v1 - v2)/i2 cancels the potentials by about 600x: equal to round-off
    # (1.3e-13 with the line kernel of wave 3, bit-equal before by chance).
    assert z_skin[0] == pytest.approx(z_dc[0], rel=1e-12)
    assert z_skin[1].real == pytest.approx(r_dc * _skin_ratio(1000.0).real, rel=1e-9)
    assert z_skin[1].imag == pytest.approx(0.0, abs=1e-12)
    assert z_dc[1] == z_dc[0]  # the dc model stays frequency independent


@pytest.mark.finding("10.37")
def test_skin_effect_is_frequency_dependent() -> None:
    """R_ac/R_dc of 150 mm2 Al through the solver: 1.0556 / 1.2308 / 1.5496."""
    freqs = [250.0, 550.0, 1000.0]
    z_dc = _series_impedance(_pen_world(300.0, dsl=10.0), "perfect_mirror", freqs)
    z_skin = _series_impedance(
        _pen_world(300.0, dsl=10.0, resistance_model="skin"), "perfect_mirror", freqs
    )
    r_dc = RHO_AL * 300.0 / A_150
    ratio = 1.0 + (z_skin - z_dc).real / r_dc
    np.testing.assert_allclose(ratio, [1.0556, 1.2308, 1.5496], atol=5e-4)


@pytest.mark.finding("10.60")
def test_branch_correction_memory_bound_keeps_the_numbers(monkeypatch) -> None:
    """With a tiny memory budget every frequency is assembled on its own and
    evicted matrices are rebuilt: the same matrices, bit for bit."""
    from groundfield.solver import _inductive, _pipeline

    w = _pen_world(60.0, dsl=10.0)
    eng = gf.create_engine(
        backend="image",
        segment_length=0.5,
        frequencies=[50.0, 250.0, 1000.0],
        earth_inductive_model="sommerfeld",
    )
    disc = _pipeline.discretise(w, eng)
    topo = _pipeline.build_topology(w, disc)
    omegas = [2.0 * np.pi * f for f in eng.frequencies]
    full = [
        _pipeline.assemble_branches(w, eng, topo).earth_correction(om) for om in omegas
    ]
    monkeypatch.setattr(_inductive, "_CACHE_BYTES", 1)
    small = _pipeline.assemble_branches(w, eng, topo).earth_correction
    for om, ref in zip(omegas + omegas, full + full, strict=True):
        assert np.array_equal(small(om), ref)
    assert small.n_assemblies == 6  # one at a time, and rebuilt after eviction


# ---------------------------------------------------------------------
# User decision Q27 — perfect_mirror warns for inductive pairs on opposite
# sides of the soil surface
# ---------------------------------------------------------------------


def _lead_and_pen_world(
    *,
    lead_z: float = -1.0,
    lead_dsl: float | None = 25.0,
    lead_inductive: bool = True,
    lead_name: str = "lead",
) -> gf.World:
    """Rods 50 m apart, bonded by a lead at ``lead_z`` and a PEN at 0.7 m.

    The lead hangs 1 m above the soil surface by default (``z < 0``);
    with ``lead_dsl=25`` it has two inductive segments, the PEN
    (``discretize_segment_length=10``) five: 10 pairs across the surface.
    """
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=RHO_SOIL))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="g2", position=(50.0, 0.0, 0.5), length=3.0)
    for name, y, z, dsl, inductive in (
        (lead_name, 1.0, lead_z, lead_dsl, lead_inductive),
        ("pen", 0.0, 0.7, 10.0, True),
    ):
        w.add_conductor(
            Conductor(
                name=name,
                start=(0.0, y, z),
                end=(50.0, y, z),
                start_electrode="g1",
                end_electrode="g2",
                wire_radius=A_RADIUS,
                resistivity=RHO_AL,
                cross_section=A_150,
                discretize_segment_length=dsl,
                coupling_to_soil="isolated",
                inductance_model="neumann" if inductive else None,
            )
        )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _surface_pair_warnings(
    world: gf.World, model: str = "perfect_mirror", freqs: tuple = (50.0,)
) -> tuple[gf.FieldResult, list[warnings.WarningMessage]]:
    eng = gf.create_engine(
        backend="image",
        segment_length=0.5,
        frequencies=list(freqs),
        earth_inductive_model=model,
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = world.solve(eng)
    hits = [w for w in rec if "opposite sides of the soil surface" in str(w.message)]
    return res, hits


@pytest.mark.finding("10.46")
def test_perfect_mirror_warns_for_inductive_pairs_across_the_surface() -> None:
    """The additive image has no physical basis for an overhead/buried pair.

    ``perfect_mirror`` adds the mirrored-endpoint image to every pair,
    also to a lead in the air over a buried PEN; up to bbaea9f silently.
    The numbers stay (a diagnostic baseline); the solve says so once and
    points to ``"sommerfeld"``.
    """
    # user decision Q27
    res, hits = _surface_pair_warnings(_lead_and_pen_world())
    assert len(hits) == 1, [str(w.message) for w in hits]
    hit = hits[0]
    text = str(hit.message)
    assert hit.category is UserWarning  # a plain UserWarning
    assert hit.filename == __file__  # attributed to the caller's line
    assert "perfect_mirror" in text and "10 inductive segment pairs" in text, text
    assert "'lead'" in text and "'pen'" in text, text
    assert "'sommerfeld'" in text, text
    assert np.isfinite(res.input_impedance()[0])


@pytest.mark.finding("10.46")
def test_perfect_mirror_surface_warning_is_emitted_once_per_solve() -> None:
    # user decision Q27
    _, hits = _surface_pair_warnings(_lead_and_pen_world(), freqs=(50.0, 150.0, 1000.0))
    assert len(hits) == 1


@pytest.mark.finding("10.46")
def test_perfect_mirror_surface_warning_names_a_lumped_conductor() -> None:
    """A lumped lead is one inductive branch: 1 x 5 pairs, still named."""
    # user decision Q27
    _, hits = _surface_pair_warnings(_lead_and_pen_world(lead_dsl=None))
    assert len(hits) == 1
    text = str(hits[0].message)
    assert "5 inductive segment pairs" in text and "'lead'" in text, text


@pytest.mark.finding("10.46")
@pytest.mark.parametrize("name", ["le\nad", "lead\n", "\nlead", "le\r\nad"])
def test_perfect_mirror_surface_warning_names_a_chain_conductor_across_line_breaks(
    name: str,
) -> None:
    """A chain conductor whose name contains a line break is still named.

    A conductor chain names its conductor in its pseudo-nodes
    (``__cond_<name>__seg_<k>``); the pattern that reads them back lacked
    ``re.DOTALL``, so ``'le\\nad'`` was reported as "unnamed branches".
    """
    # audit D2A-07
    _, hits = _surface_pair_warnings(_lead_and_pen_world(lead_name=name))
    assert len(hits) == 1
    text = str(hits[0].message)
    assert repr(name) in text and "unnamed" not in text, text


@pytest.mark.finding("10.46")
@pytest.mark.parametrize("model", ["carson_series", "sommerfeld"])
def test_other_earth_models_do_not_warn_about_surface_pairs(model: str) -> None:
    # user decision Q27
    _, hits = _surface_pair_warnings(_lead_and_pen_world(), model=model)
    assert hits == []


@pytest.mark.finding("10.46")
@pytest.mark.parametrize(
    "world",
    [
        pytest.param(lambda: _lead_and_pen_world(lead_z=1.2), id="both-buried"),
        pytest.param(
            lambda: _lead_and_pen_world(lead_inductive=False), id="lead-not-inductive"
        ),
    ],
)
def test_perfect_mirror_is_silent_with_inductive_conductors_on_one_side(world) -> None:
    # user decision Q27
    _, hits = _surface_pair_warnings(world())
    assert hits == []
